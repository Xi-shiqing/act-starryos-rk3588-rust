mod rknn_sys;
use rknn_sys::*;
use std::os::raw::c_void;
use std::ptr;
use std::time::Instant;

// ---- 固定的模型 / 数据集参数（与 act-rust 一致）----------------------------
const IMAGE_W: usize = 224;
const IMAGE_H: usize = 224;
const IMAGE_C: usize = 3;
const STATE_DIM: usize = 2;
const ACTION_CHUNK: usize = 8;
const ACTION_DIM: usize = 3;

const IMAGE_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGE_STD: [f32; 3] = [0.229, 0.224, 0.225];
const STATE_Q01: [f32; STATE_DIM] = [-0.079_000_004, 0.0];
const STATE_Q99: [f32; STATE_DIM] = [0.200_000_003, 0.200_000_003];
const ACTION_Q01: [f32; ACTION_DIM] = [-0.100_000_001, 0.0, 0.0];
const ACTION_Q99: [f32; ACTION_DIM] = [0.200_000_003, 0.200_000_003, 0.0];

// ---- 预处理 ----------------------------------------------------------------
struct Rgb {
    w: usize,
    h: usize,
    pixels: Vec<u8>, // 交错 RGB，行优先
}

fn decode_jpeg(path: &str) -> Result<Rgb, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("open {path}: {e}"))?;
    let mut dec = jpeg_decoder::Decoder::new(std::io::BufReader::new(file));
    let pixels = dec.decode().map_err(|e| format!("decode {path}: {e}"))?;
    let info = dec.info().ok_or_else(|| format!("no jpeg info {path}"))?;
    let (w, h) = (info.width as usize, info.height as usize);
    let rgb = match info.pixel_format {
        jpeg_decoder::PixelFormat::RGB24 => pixels,
        jpeg_decoder::PixelFormat::L8 => {
            let mut out = Vec::with_capacity(w * h * 3);
            for g in pixels {
                out.extend_from_slice(&[g, g, g]);
            }
            out
        }
        other => return Err(format!("unsupported jpeg fmt: {other:?}")),
    };
    Ok(Rgb { w, h, pixels: rgb })
}

/// 双线性缩放到 224×224 + ImageNet 归一化，输出 **NHWC**（HWC 交错）。
/// RKNN 图像输入要 NHWC；像素中心映射与 act-rust/参考 C++ 完全一致。
fn preprocess_image_nhwc(img: &Rgb) -> Vec<f32> {
    let mut hwc = vec![0.0f32; IMAGE_H * IMAGE_W * IMAGE_C];
    let scale_x = img.w as f32 / IMAGE_W as f32;
    let scale_y = img.h as f32 / IMAGE_H as f32;
    for y in 0..IMAGE_H {
        let src_y = (y as f32 + 0.5) * scale_y - 0.5;
        let y0 = (src_y.floor() as isize).clamp(0, img.h as isize - 1) as usize;
        let y1 = (y0 + 1).min(img.h - 1);
        let wy = src_y - y0 as f32;
        for x in 0..IMAGE_W {
            let src_x = (x as f32 + 0.5) * scale_x - 0.5;
            let x0 = (src_x.floor() as isize).clamp(0, img.w as isize - 1) as usize;
            let x1 = (x0 + 1).min(img.w - 1);
            let wx = src_x - x0 as f32;
            let p00 = (y0 * img.w + x0) * 3;
            let p01 = (y0 * img.w + x1) * 3;
            let p10 = (y1 * img.w + x0) * 3;
            let p11 = (y1 * img.w + x1) * 3;
            for c in 0..3 {
                let top = img.pixels[p00 + c] as f32 * (1.0 - wx) + img.pixels[p01 + c] as f32 * wx;
                let bot = img.pixels[p10 + c] as f32 * (1.0 - wx) + img.pixels[p11 + c] as f32 * wx;
                let pixel = (top * (1.0 - wy) + bot * wy) / 255.0;
                hwc[(y * IMAGE_W + x) * IMAGE_C + c] = (pixel - IMAGE_MEAN[c]) / IMAGE_STD[c];
            }
        }
    }
    hwc
}

fn normalize_state(state: [f32; STATE_DIM]) -> [f32; STATE_DIM] {
    let mut out = [0.0f32; STATE_DIM];
    for i in 0..STATE_DIM {
        let mut d = STATE_Q99[i] - STATE_Q01[i];
        if d == 0.0 {
            d = 1e-8;
        }
        out[i] = 2.0 * (state[i] - STATE_Q01[i]) / d - 1.0;
    }
    out
}

fn denormalize_action(action_norm: &[f32]) -> Vec<f32> {
    action_norm
        .iter()
        .enumerate()
        .map(|(i, &a)| {
            let d = i % ACTION_DIM;
            let mut denom = ACTION_Q99[d] - ACTION_Q01[d];
            if denom == 0.0 {
                denom = 1e-8;
            }
            (a + 1.0) * 0.5 * denom + ACTION_Q01[d]
        })
        .collect()
}

fn turn_decision(diff: f32, deadband: f32) -> &'static str {
    if diff.abs() <= deadband {
        "straight"
    } else if diff > 0.0 {
        "right"
    } else {
        "left"
    }
}

// ---- 参数 ------------------------------------------------------------------
struct Args {
    model: String,
    image: String,
    state: [f32; STATE_DIM],
    deadband: f32,
}

fn parse_args() -> Args {
    let mut a = Args {
        model: "act_rk3588_fp16.rknn".to_string(),
        image: "frame_000227.jpg".to_string(),
        state: [0.0, 0.0],
        deadband: 0.0,
    };
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => { i += 1; a.model = argv[i].clone(); }
            "--image" => { i += 1; a.image = argv[i].clone(); }
            "--state" => {
                a.state[0] = argv[i + 1].parse().unwrap_or(0.0);
                a.state[1] = argv[i + 2].parse().unwrap_or(0.0);
                i += 2;
            }
            "--deadband" => { i += 1; a.deadband = argv[i].parse().unwrap_or(0.0); }
            other => eprintln!("warning: ignoring unknown arg {other}"),
        }
        i += 1;
    }
    a
}

// ---- 主程序 ----------------------------------------------------------------
fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = parse_args();

    // 1) 读入 .rknn 模型
    let model_bytes = std::fs::read(&args.model).map_err(|e| format!("read model {}: {e}", args.model))?;
    eprintln!("loading rknn model: {} ({} bytes)", args.model, model_bytes.len());

    // 2) rknn_init
    let mut ctx: rknn_context = 0;
    let rc = unsafe {
        rknn_init(
            &mut ctx,
            model_bytes.as_ptr() as *mut c_void,
            model_bytes.len() as u32,
            0,
            ptr::null_mut(),
        )
    };
    if rc != 0 {
        return Err(format!("rknn_init failed: {rc}"));
    }
    eprintln!("rknn_init OK (ctx={ctx})");

    // 3) 收集待推理的帧：--image 指向目录则取目录下全部 *.jpg（排序），否则单帧。
    //    板上不便传数据，故支持一次烧录、一次启动跑完多帧。
    let frames = collect_frames(&args.image)?;
    eprintln!("frames to run: {}", frames.len());

    // state 对所有帧相同（默认 (0,0)）
    let mut state_norm = normalize_state(args.state);

    let mut total_ms = 0.0f64;
    let mut ok = 0usize;
    for path in &frames {
        match infer_one(ctx, path, &mut state_norm, args.deadband) {
            Ok((left, right, diff, decision, ms)) => {
                ok += 1;
                total_ms += ms;
                let name = std::path::Path::new(path)
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or(path);
                println!(
                    "{name}: left_vel={left:.6} right_vel={right:.6} diff={diff:+.6} decision={decision} ({ms:.1} ms)"
                );
            }
            Err(e) => eprintln!("{path}: ERROR {e}"),
        }
    }
    if ok > 0 {
        println!("---");
        println!("summary: {ok}/{} frames, avg inference {:.1} ms", frames.len(), total_ms / ok as f64);
    }

    unsafe { rknn_destroy(ctx) };
    Ok(())
}

/// 列出待推理帧：路径是目录→目录内全部 .jpg（按文件名排序）；否则当作单个文件。
fn collect_frames(path: &str) -> Result<Vec<String>, String> {
    let p = std::path::Path::new(path);
    if p.is_dir() {
        let mut v: Vec<String> = std::fs::read_dir(p)
            .map_err(|e| format!("read_dir {path}: {e}"))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().and_then(|s| s.to_str()).map(|s| s.eq_ignore_ascii_case("jpg")).unwrap_or(false))
            .map(|p| p.to_string_lossy().into_owned())
            .collect();
        v.sort();
        if v.is_empty() {
            return Err(format!("no .jpg under {path}"));
        }
        Ok(v)
    } else {
        Ok(vec![path.to_string()])
    }
}

/// 对单帧做一次完整推理，返回 (左轮速, 右轮速, diff, 判向, 推理耗时ms)。
/// 计时只覆盖 rknn_run（NPU 实际计算），不含 JPEG 解码/预处理。
fn infer_one(
    ctx: rknn_context,
    image_path: &str,
    state_norm: &mut [f32],
    deadband: f32,
) -> Result<(f32, f32, f32, &'static str, f64), String> {
    let rgb = decode_jpeg(image_path)?;
    let mut image_nhwc = preprocess_image_nhwc(&rgb);

    let mut inputs = [
        rknn_input {
            index: 0,
            buf: image_nhwc.as_mut_ptr() as *mut c_void,
            size: (image_nhwc.len() * 4) as u32,
            pass_through: 0,
            type_: RKNN_TENSOR_FLOAT32,
            fmt: RKNN_TENSOR_NHWC,
        },
        rknn_input {
            index: 1,
            buf: state_norm.as_mut_ptr() as *mut c_void,
            size: (state_norm.len() * 4) as u32,
            pass_through: 0,
            type_: RKNN_TENSOR_FLOAT32,
            fmt: RKNN_TENSOR_NCHW,
        },
    ];
    let rc = unsafe { rknn_inputs_set(ctx, inputs.len() as u32, inputs.as_mut_ptr()) };
    if rc != 0 {
        return Err(format!("rknn_inputs_set failed: {rc}"));
    }

    let t0 = Instant::now();
    let rc = unsafe { rknn_run(ctx, ptr::null_mut()) };
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    if rc != 0 {
        return Err(format!("rknn_run failed: {rc}"));
    }

    let mut outputs = [rknn_output {
        want_float: 1,
        is_prealloc: 0,
        index: 0,
        buf: ptr::null_mut(),
        size: 0,
    }];
    let rc = unsafe { rknn_outputs_get(ctx, 1, outputs.as_mut_ptr(), ptr::null_mut()) };
    if rc != 0 {
        return Err(format!("rknn_outputs_get failed: {rc}"));
    }
    let n = (outputs[0].size as usize) / 4;
    let action_norm: Vec<f32> =
        unsafe { std::slice::from_raw_parts(outputs[0].buf as *const f32, n).to_vec() };
    unsafe { rknn_outputs_release(ctx, 1, outputs.as_mut_ptr()) };

    let action = denormalize_action(&action_norm);
    let left = action[0];
    let right = action.get(1).copied().unwrap_or(0.0);
    let diff = left - right;
    let decision = turn_decision(diff, deadband);
    let _ = ACTION_CHUNK;
    Ok((left, right, diff, decision, ms))
}
