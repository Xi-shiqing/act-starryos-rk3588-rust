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

# 3) 测试帧：56 个明确转向帧 + 期望方向（供肉眼对照）
python3 - "$ROOT" "$OUT" <<'PY'
import csv,os,shutil,sys
root,out=sys.argv[1],sys.argv[2]
turn=[l.strip() for l in open(f"{root}/data/eval_turn.txt")]
ref={r['frame']:r for r in csv.DictReader(open(f"{root}/results/ref_fp32.csv"))}
with open(f"{out}/EXPECTED.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["frame","gt_dir","fp32_sim_dir"])
    for s in turn:
        fn=s+".jpg"
        shutil.copy(f"{root}/data/frames/{fn}", f"{out}/frames/{fn}")
        w.writerow([fn, ref[fn]['gt'], ref[fn]['pred']])
print("copied", len(turn), "frames")
PY

# 4) 板上运行脚本
cat > "$OUT/run.sh" <<'RUN'
#!/bin/sh
# 在 RK3588（普通 Linux）上运行：批量跑 frames/ 下全部帧，打印判向+耗时。
cd "$(dirname "$0")"
./act-rknn --model act_rk3588_fp16.rknn --image frames --state 0 0
RUN
chmod +x "$OUT/run.sh"

# 5) 说明
cat > "$OUT/README.txt" <<'TXT'
RK3588 普通 Linux 上的 ACT NPU 推理测试包（部署模型 = fp16）
文件：act-rknn(可执行) librknnrt.so act_rk3588_fp16.rknn frames/(56 转向帧) run.sh EXPECTED.csv
跑法：  sh run.sh
输出：  每帧打印 left/right/diff/decision 和推理耗时(ms)，最后给平均耗时。
对照：  EXPECTED.csv 里 gt_dir/fp32_sim_dir 是参考方向，肉眼比对 decision 是否一致。
前提：  板子的 NPU 驱动正常(/dev/dri + RKNPU driver)；librknnrt.so 与板上驱动版本兼容。
实测：  fp16 在真 RK3588 NPU 上 46/56 vs gt、右召回18/19、24.7ms/帧、峰值206MB；
        详见 results/board_rk3588_real.md（含与 hybrid/int8 的真硬件对比）。
TXT

echo "=== package done: $OUT ==="
ls -la "$OUT"; du -sh "$OUT"
