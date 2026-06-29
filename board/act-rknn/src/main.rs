mod rknn_sys;
use rknn_sys::*;
use std::io::{Read, Seek, SeekFrom};
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
    rgb_pack: Option<String>,
    state: [f32; STATE_DIM],
    deadband: f32,
    feedback: bool, // false=开环(每帧恒 --state)；true=闭环(下一帧 state=上一帧预测轮速)
    start: usize,   // 仅当 --image 是目录：从第 start 帧开始
    count: usize,   // 仅当 --image 是目录：最多跑 count 帧（0=到末尾）。配合 start 做分段跑。
    io_mode: IoMode,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum IoMode {
    ZeroCopyFloat,
    OldOutputsGet,
}

impl IoMode {
    fn parse(s: &str) -> Self {
        match s {
            "old" | "outputs-get" => Self::OldOutputsGet,
            "zc-float" | "zero-copy" | "zerocopy" => Self::ZeroCopyFloat,
            other => {
                eprintln!("warning: unknown --io-mode {other}, using zc-float");
                Self::ZeroCopyFloat
            }
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::ZeroCopyFloat => "zc-float",
            Self::OldOutputsGet => "old",
        }
    }
}

fn parse_args() -> Args {
    let mut a = Args {
        model: "act_rk3588_fp16.rknn".to_string(),
        image: "frame_000227.jpg".to_string(),
        rgb_pack: None,
        state: [0.0, 0.0],
        deadband: 0.0,
        feedback: false,
        start: 0,
        count: 0,
        io_mode: IoMode::ZeroCopyFloat,
    };
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => { i += 1; a.model = argv[i].clone(); }
            "--image" => { i += 1; a.image = argv[i].clone(); }
            "--rgb-pack" => { i += 1; a.rgb_pack = Some(argv[i].clone()); }
            "--state" => {
                a.state[0] = argv[i + 1].parse().unwrap_or(0.0);
                a.state[1] = argv[i + 2].parse().unwrap_or(0.0);
                i += 2;
            }
            "--deadband" => { i += 1; a.deadband = argv[i].parse().unwrap_or(0.0); }
            // --loop open|closed ；闭环也可写 --feedback
            "--loop" => { i += 1; a.feedback = argv[i].eq_ignore_ascii_case("closed"); }
            "--feedback" => { a.feedback = true; }
            "--start" => { i += 1; a.start = argv[i].parse().unwrap_or(0); }
            "--count" => { i += 1; a.count = argv[i].parse().unwrap_or(0); }
            "--io-mode" => { i += 1; a.io_mode = IoMode::parse(&argv[i]); }
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

    // 2.5) IO 模式：
    // - zc-float：零拷贝，查询输入/输出张量属性，预分配一次 NPU 内存并 set_io_mem 绑定。
    //   之后每帧只 rknn_run + 直接读写固定内存，不再 outputs_get/release，用来验证/规避第 96 帧 DMA 泄漏。
    // - old：保留原 inputs_set + outputs_get/release 路径，用作 A/B 复现与应急对照。
    eprintln!("io mode: {}", args.io_mode.label());
    let zc_io = if args.io_mode == IoMode::ZeroCopyFloat {
        let mut io_num = rknn_input_output_num { n_input: 0, n_output: 0 };
        let rc = unsafe {
            rknn_query(
                ctx,
                RKNN_QUERY_IN_OUT_NUM,
                &mut io_num as *mut _ as *mut c_void,
                std::mem::size_of::<rknn_input_output_num>() as u32,
            )
        };
        if rc != 0 {
            return Err(format!("rknn_query(IN_OUT_NUM) failed: {rc}"));
        }
        eprintln!("model io: n_input={} n_output={}", io_num.n_input, io_num.n_output);
        if io_num.n_input < 2 || io_num.n_output < 1 {
            return Err(format!(
                "unexpected io num: n_input={} n_output={} (expect 2 in / 1 out)",
                io_num.n_input, io_num.n_output
            ));
        }

        let (img_mem, img_elems) =
            unsafe { setup_io_mem(ctx, RKNN_QUERY_INPUT_ATTR, 0, Some(RKNN_TENSOR_NHWC), "input[0] image")? };
        let (state_mem, _state_elems) =
            unsafe { setup_io_mem(ctx, RKNN_QUERY_INPUT_ATTR, 1, Some(RKNN_TENSOR_NCHW), "input[1] state")? };
        let (out_mem, out_elems) =
            unsafe { setup_io_mem(ctx, RKNN_QUERY_OUTPUT_ATTR, 0, None, "output[0] action")? };
        Some(ZeroCopyIo { img_mem, img_elems, state_mem, out_mem, out_elems })
    } else {
        None
    };

    // 3) 收集待推理的帧：--image 指向目录则取目录下全部 *.jpg（排序），否则单帧。
    //    板上不便传数据，故支持一次烧录、一次启动跑完多帧。
    let mut frames = collect_frames(&args.image)?;
    // 分段跑：仅目录模式有意义。配合外部脚本多次调用（每段独立进程）周期性释放资源。
    if std::path::Path::new(&args.image).is_dir() {
        let n = frames.len();
        let s = args.start.min(n);
        let e = if args.count == 0 { n } else { (s + args.count).min(n) };
        frames = frames[s..e].to_vec();
    }
    eprintln!("frames to run: {} (start={}, count={})", frames.len(), args.start, args.count);
    eprintln!(
        "state mode: {}",
        if args.feedback { "closed-loop (feedback predicted state)" } else { "open-loop (fixed --state)" }
    );
    let mut rgb_pack = match &args.rgb_pack {
        Some(path) => {
            let file = std::fs::File::open(path).map_err(|e| format!("open rgb pack {path}: {e}"))?;
            eprintln!("image source: rgb-pack {path} ({} bytes/frame)", IMAGE_W * IMAGE_H * IMAGE_C);
            Some(RgbPack { file, frame_size: IMAGE_W * IMAGE_H * IMAGE_C })
        }
        None => {
            eprintln!("image source: jpeg files");
            None
        }
    };

    use std::io::Write;
    let trace = std::env::var("ACT_TRACE").is_ok(); // 设置则在每帧 NPU 调用前后打阶段标记

    // 开环：每帧恒用 args.state。闭环：state 随上一帧预测轮速滚动更新。
    let mut state = args.state;

    let mut total_ms = 0.0f64;
    let mut ok = 0usize;
    for (frame_index, path) in frames.iter().enumerate() {
        let state_norm = normalize_state(state);
        let image_nhwc = match load_image_nhwc(path, args.start + frame_index, rgb_pack.as_mut(), trace) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("{path}: ERROR {e}");
                continue;
            }
        };
        let result = if let Some(zc) = &zc_io {
            infer_one_zc_float(ctx, zc, &image_nhwc, &state_norm, args.deadband, trace)
        } else {
            infer_one_old(ctx, &image_nhwc, &state_norm, args.deadband, trace)
        };
        match result {
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
                let _ = std::io::stdout().flush(); // 每帧立即刷出，卡点帧号才精确
                if args.feedback {
                    state = [left, right]; // 反馈到下一帧
                }
            }
            Err(e) => eprintln!("{path}: ERROR {e}"),
        }
    }
    if ok > 0 {
        println!("---");
        println!("summary: {ok}/{} frames, avg inference {:.1} ms", frames.len(), total_ms / ok as f64);
    }

    unsafe {
        if let Some(zc) = zc_io {
            rknn_destroy_mem(ctx, zc.img_mem);
            rknn_destroy_mem(ctx, zc.state_mem);
            rknn_destroy_mem(ctx, zc.out_mem);
        }
        rknn_destroy(ctx);
    }
    Ok(())
}

struct ZeroCopyIo {
    img_mem: *mut rknn_tensor_mem,
    img_elems: usize,
    state_mem: *mut rknn_tensor_mem,
    out_mem: *mut rknn_tensor_mem,
    out_elems: usize,
}

struct RgbPack {
    file: std::fs::File,
    frame_size: usize,
}

/// 查询单个输入/输出张量属性，按 FLOAT32 预分配一块 NPU 内存并 set_io_mem 绑定。
/// `want_fmt=Some(..)` 时把缓冲布局覆写为该 fmt（输入用），`None` 保持模型原生 fmt（输出用）。
/// 返回 (内存指针, 元素个数)。同时打印属性，便于在板上排查。
unsafe fn setup_io_mem(
    ctx: rknn_context,
    query_cmd: std::os::raw::c_int,
    index: u32,
    want_fmt: Option<std::os::raw::c_int>,
    label: &str,
) -> Result<(*mut rknn_tensor_mem, usize), String> {
    let mut attr = rknn_tensor_attr::default();
    attr.index = index;
    let rc = rknn_query(
        ctx,
        query_cmd,
        &mut attr as *mut _ as *mut c_void,
        std::mem::size_of::<rknn_tensor_attr>() as u32,
    );
    if rc != 0 {
        return Err(format!("rknn_query attr({label}) failed: {rc}"));
    }
    let n_elems = attr.n_elems as usize;
    let dims: Vec<u32> = attr.dims[..(attr.n_dims as usize).min(RKNN_MAX_DIMS)].to_vec();
    eprintln!(
        "{label}: index={} dims={:?} n_elems={} size={} size_with_stride={} fmt={} type={}",
        attr.index, dims, attr.n_elems, attr.size, attr.size_with_stride, attr.fmt, attr.type_
    );
    if n_elems == 0 {
        return Err(format!("{label}: n_elems=0"));
    }

    // 以 FLOAT32 描述我方缓冲：驱动按 type/fmt 做转换（非 pass_through）。
    attr.type_ = RKNN_TENSOR_FLOAT32;
    if let Some(f) = want_fmt {
        attr.fmt = f;
    }
    attr.pass_through = 0;
    let bytes = (n_elems * 4) as u32;
    attr.size = bytes;

    let mem = rknn_create_mem(ctx, bytes);
    if mem.is_null() {
        return Err(format!("{label}: rknn_create_mem({bytes}) returned NULL"));
    }
    let rc = rknn_set_io_mem(ctx, mem, &mut attr);
    if rc != 0 {
        return Err(format!("{label}: rknn_set_io_mem failed: {rc}"));
    }
    Ok((mem, n_elems))
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

fn normalize_rgb224_nhwc(rgb: &[u8]) -> Vec<f32> {
    let mut hwc = vec![0.0f32; IMAGE_H * IMAGE_W * IMAGE_C];
    for y in 0..IMAGE_H {
        for x in 0..IMAGE_W {
            let base = (y * IMAGE_W + x) * IMAGE_C;
            for c in 0..IMAGE_C {
                let pixel = rgb[base + c] as f32 / 255.0;
                hwc[base + c] = (pixel - IMAGE_MEAN[c]) / IMAGE_STD[c];
            }
        }
    }
    hwc
}

fn load_image_nhwc(
    image_path: &str,
    frame_index: usize,
    rgb_pack: Option<&mut RgbPack>,
    trace: bool,
) -> Result<Vec<f32>, String> {
    if let Some(pack) = rgb_pack {
        if trace {
            eprint!("[pack>]");
            let _ = { use std::io::Write; std::io::stderr().flush() };
        }
        let mut rgb = vec![0u8; pack.frame_size];
        let off = (frame_index as u64)
            .checked_mul(pack.frame_size as u64)
            .ok_or_else(|| format!("rgb pack offset overflow for frame {frame_index}"))?;
        pack.file
            .seek(SeekFrom::Start(off))
            .map_err(|e| format!("seek rgb pack frame {frame_index}: {e}"))?;
        pack.file
            .read_exact(&mut rgb)
            .map_err(|e| format!("read rgb pack frame {frame_index}: {e}"))?;
        if trace {
            eprint!("[pack<]");
            let _ = { use std::io::Write; std::io::stderr().flush() };
        }
        return Ok(normalize_rgb224_nhwc(&rgb));
    }

    if trace {
        eprint!("[jpg>]");
        let _ = { use std::io::Write; std::io::stderr().flush() };
    }
    let rgb = decode_jpeg(image_path)?;
    if trace {
        eprint!("[jpg<][prep>]");
        let _ = { use std::io::Write; std::io::stderr().flush() };
    }
    let image_nhwc = preprocess_image_nhwc(&rgb);
    if trace {
        eprint!("[prep<]");
        let _ = { use std::io::Write; std::io::stderr().flush() };
    }
    Ok(image_nhwc)
}

/// 对单帧做一次完整推理，返回 (左轮速, 右轮速, diff, 判向, 推理耗时ms)。
/// 计时只覆盖 rknn_run（NPU 实际计算），不含 JPEG 解码/预处理。
#[allow(clippy::too_many_arguments)]
fn infer_one_zc_float(
    ctx: rknn_context,
    zc: &ZeroCopyIo,
    image_nhwc: &[f32],
    state_norm: &[f32],
    deadband: f32,
    trace: bool,
) -> Result<(f32, f32, f32, &'static str, f64), String> {
    if image_nhwc.len() != zc.img_elems {
        return Err(format!(
            "preprocess produced {} floats but model image input expects {}",
            image_nhwc.len(),
            zc.img_elems
        ));
    }

    // 零拷贝：把图像/状态直接写进绑定的输入 DMA 内存（FLOAT32）。
    // 这几块内存在 run() 里一次性分配并 set_io_mem 绑定，整轮复用，不再每帧分配/释放。
    unsafe {
        let dst = (*zc.img_mem).virt_addr as *mut f32;
        if dst.is_null() {
            return Err("image input mem virt_addr is NULL".into());
        }
        std::ptr::copy_nonoverlapping(image_nhwc.as_ptr(), dst, zc.img_elems);

        let sdst = (*zc.state_mem).virt_addr as *mut f32;
        if sdst.is_null() {
            return Err("state input mem virt_addr is NULL".into());
        }
        std::ptr::copy_nonoverlapping(state_norm.as_ptr(), sdst, state_norm.len());
    }

    // 阶段标记（stderr，便于卡死时定位卡在 NPU 还是别处；由 ACT_TRACE 控制）
    if trace { eprint!("[npu>]"); let _ = { use std::io::Write; std::io::stderr().flush() }; }
    let t0 = Instant::now();
    let rc = unsafe { rknn_run(ctx, ptr::null_mut()) };
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    if rc != 0 {
        return Err(format!("rknn_run failed: {rc}"));
    }
    if trace { eprint!("[npu<]"); let _ = { use std::io::Write; std::io::stderr().flush() }; }

    // 直接读绑定的输出 DMA 内存（驱动在 rknn_run 里已写入并刷/失效 cache，
    // 因为未设置 RKNN_FLAG_DISABLE_FLUSH_OUTPUT_MEM_CACHE）。不再 outputs_get/release，
    // 从根上消除每帧输出 DMA 泄漏 → 不再卡在第 96 帧。
    let action_norm: Vec<f32> = unsafe {
        let src = (*zc.out_mem).virt_addr as *const f32;
        if src.is_null() {
            return Err("output mem virt_addr is NULL".into());
        }
        std::slice::from_raw_parts(src, zc.out_elems).to_vec()
    };
    if trace { eprint!("[rd]"); let _ = { use std::io::Write; std::io::stderr().flush() }; }

    let action = denormalize_action(&action_norm);
    let left = action[0];
    let right = action.get(1).copied().unwrap_or(0.0);
    let diff = left - right;
    let decision = turn_decision(diff, deadband);
    let _ = ACTION_CHUNK;
    Ok((left, right, diff, decision, ms))
}

fn infer_one_old(
    ctx: rknn_context,
    image_nhwc: &[f32],
    state_norm: &[f32],
    deadband: f32,
    trace: bool,
) -> Result<(f32, f32, f32, &'static str, f64), String> {
    let mut image_nhwc = image_nhwc.to_vec();
    let mut state_buf = state_norm.to_vec();

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
            buf: state_buf.as_mut_ptr() as *mut c_void,
            size: (state_buf.len() * 4) as u32,
            pass_through: 0,
            type_: RKNN_TENSOR_FLOAT32,
            fmt: RKNN_TENSOR_NCHW,
        },
    ];
    if trace { eprint!("[set]"); let _ = { use std::io::Write; std::io::stderr().flush() }; }
    let rc = unsafe { rknn_inputs_set(ctx, inputs.len() as u32, inputs.as_mut_ptr()) };
    if rc != 0 {
        return Err(format!("rknn_inputs_set failed: {rc}"));
    }

    if trace { eprint!("[npu>]"); let _ = { use std::io::Write; std::io::stderr().flush() }; }
    let t0 = Instant::now();
    let rc = unsafe { rknn_run(ctx, ptr::null_mut()) };
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    if rc != 0 {
        return Err(format!("rknn_run failed: {rc}"));
    }
    if trace { eprint!("[npu<]"); let _ = { use std::io::Write; std::io::stderr().flush() }; }

    let mut outputs = [rknn_output {
        want_float: 1,
        is_prealloc: 0,
        index: 0,
        buf: ptr::null_mut(),
        size: 0,
    }];
    if trace { eprint!("[get>]"); let _ = { use std::io::Write; std::io::stderr().flush() }; }
    let rc = unsafe { rknn_outputs_get(ctx, 1, outputs.as_mut_ptr(), ptr::null_mut()) };
    if rc != 0 {
        return Err(format!("rknn_outputs_get failed: {rc}"));
    }
    if trace { eprint!("[get<]"); let _ = { use std::io::Write; std::io::stderr().flush() }; }
    let n = (outputs[0].size as usize) / 4;
    let action_norm: Vec<f32> =
        unsafe { std::slice::from_raw_parts(outputs[0].buf as *const f32, n).to_vec() };
    unsafe { rknn_outputs_release(ctx, 1, outputs.as_mut_ptr()) };
    if trace { eprint!("[rel]"); let _ = { use std::io::Write; std::io::stderr().flush() }; }

    let action = denormalize_action(&action_norm);
    let left = action[0];
    let right = action.get(1).copied().unwrap_or(0.0);
    let diff = left - right;
    let decision = turn_decision(diff, deadband);
    let _ = ACTION_CHUNK;
    Ok((left, right, diff, decision, ms))
}
