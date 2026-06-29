#!/usr/bin/env bash
# 组装板上部署包（Linux/NPU 版）：可执行 + librknnrt.so + 模型 + 测试帧 + 运行脚本。
# 用法：bash scripts/package_board.sh
set -e
ROOT="$(cd "$(dirname "$0")/.."; pwd)"
TC=/root/OScompetition/toolchains/aarch64--glibc--stable-2025.08-1/bin
OUT="$ROOT/board_pkg/linux"
rm -rf "$OUT"; mkdir -p "$OUT/frames"

# 1) 交叉编译 + strip
( cd "$ROOT/board/act-rknn" && cargo build --release --target aarch64-unknown-linux-gnu )
"$TC/aarch64-linux-strip" -o "$OUT/act-rknn" \
  "$ROOT/board/act-rknn/target/aarch64-unknown-linux-gnu/release/act-rknn"

# 2) 运行时库 + 模型（fp16：真硬件实测最准，见 results/board_rk3588_real.md）
cp "$ROOT/board/rknn_runtime/aarch64/librknnrt.so" "$OUT/"
cp "$ROOT/model/act_rk3588_fp16.rknn"              "$OUT/"

# 3) 测试帧：完整 666 帧 + 参考方向（以原模型 fp32 判向为权威，供肉眼对照）
#    同时生成连续 RGB224 pack，板上默认读它，绕开 StarryOS 上 JPEG 解码/小文件读在第 96 帧卡住的问题。
python3 - "$ROOT" "$OUT" <<'PY'
import csv,shutil,sys
from PIL import Image
root,out=sys.argv[1],sys.argv[2]
ref={r['frame']:r for r in csv.DictReader(open(f"{root}/results/666帧_原模型_fp32.csv"))}
pack_path=f"{out}/frames_rgb224.bin"
with open(f"{out}/EXPECTED.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["frame","gt_dir","fp32_sim_dir"])
    with open(pack_path, "wb") as pack:
        for fn in sorted(ref):
            src=f"{root}/data/frames/{fn}"
            shutil.copy(src, f"{out}/frames/{fn}")
            img=Image.open(src).convert("RGB").resize((224,224), Image.Resampling.BILINEAR)
            pack.write(img.tobytes())
            w.writerow([fn, ref[fn]['decision'], ref[fn]['decision']])
print("copied", len(ref), "frames; wrote RGB224 pack:", pack_path)
PY

# 4) 板上运行脚本
cat > "$OUT/run.sh" <<'RUN'
#!/bin/sh
# 【板子+Ubuntu】Orange Pi 5 Plus / RK3588 真 NPU 上批量推理 frames/ 下全部 666 帧，打印判向+耗时。
# 默认走 RKNN zero-copy (zc-float)，不再每帧 outputs_get/release。
cd "$(dirname "$0")"
echo "============================================================"
echo "【板子+Ubuntu / RK3588 真 NPU】ACT 推理 · smoke 120 帧"
echo "============================================================"
ACT_TRACE=1 ./act-rknn --model act_rk3588_fp16.rknn --image frames --rgb-pack frames_rgb224.bin --state 0 0 --loop open --io-mode zc-float --count 120
echo
echo "============================================================"
echo "【板子+Ubuntu / RK3588 真 NPU】ACT 推理 · 开环 666 帧"
echo "============================================================"
./act-rknn --model act_rk3588_fp16.rknn --image frames --rgb-pack frames_rgb224.bin --state 0 0 --loop open --io-mode zc-float
echo
echo "============================================================"
echo "【板子+Ubuntu / RK3588 真 NPU】ACT 推理 · 闭环 666 帧"
echo "============================================================"
./act-rknn --model act_rk3588_fp16.rknn --image frames --rgb-pack frames_rgb224.bin --state 0 0 --loop closed --io-mode zc-float
echo
echo "manual A/B command if needed:"
echo "  ACT_TRACE=1 ./act-rknn --model act_rk3588_fp16.rknn --image frames --state 0 0 --loop open --io-mode zc-float --count 120"
RUN
chmod +x "$OUT/run.sh"

# 5) 说明
cat > "$OUT/README.txt" <<'TXT'
RK3588 普通 Linux 上的 ACT NPU 推理测试包（部署模型 = fp16）
文件：act-rknn(可执行) librknnrt.so act_rk3588_fp16.rknn frames/(完整 666 帧) frames_rgb224.bin run.sh EXPECTED.csv
跑法：  sh run.sh（默认读 frames_rgb224.bin，先 zc-float 零拷贝跑 120 帧 smoke，过 frame_000096 后再跑完整开环/闭环）
输出：  每帧打印 left/right/diff/decision 和推理耗时(ms)，最后给平均耗时。
对照：  EXPECTED.csv 里 gt_dir/fp32_sim_dir 为原模型 fp32 判向（权威参考），肉眼比对 decision 是否一致。
诊断：  若需要复现 JPEG 卡点，可去掉 --rgb-pack 再跑；默认路径绕开 JPEG 解码/小文件读。
前提：  板子的 NPU 驱动正常(/dev/dri + RKNPU driver)；librknnrt.so 与板上驱动版本兼容。
实测：  fp16 在真 RK3588 NPU 上 ~26.6ms/帧；56 转向帧子集 46/56 vs gt、右召回18/19；
        StarryOS 真 NPU 666 帧 RGBPack 结果见 starryos_npu/RESULT_starryos_board.md。
TXT

echo "=== package done: $OUT ==="
ls -la "$OUT"; du -sh "$OUT"
