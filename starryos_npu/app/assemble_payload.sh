# 把 act-rknn + librknnrt.so + fp16 模型 + 测试帧 组装成 StarryOS rootfs 里的 /act_rknn 载荷。
# 依赖先跑过 scripts/package_board.sh（产出 board_pkg/linux）。
set -e
ROOT="$(cd "$(dirname "$0")/../.."; pwd)"
SRC="$ROOT/board_pkg/linux"
OUT="$ROOT/starryos_npu/app/payload/act_rknn"
[ -d "$SRC" ] || { echo "先跑 scripts/package_board.sh 生成 board_pkg/linux"; exit 1; }
rm -rf "$OUT"; mkdir -p "$OUT/lib" "$OUT/model" "$OUT/frames"
cp "$SRC/act-rknn"                       "$OUT/"
cp "$SRC/librknnrt.so"                   "$OUT/lib/"
cp "$SRC/act_rk3588_fp16.rknn"           "$OUT/model/"
cp "$SRC"/frames/*.jpg                    "$OUT/frames/"
cp "$ROOT/starryos_npu/app/init.sh"      "$OUT/"
cp "$SRC/EXPECTED.csv"                    "$OUT/" 2>/dev/null || true
echo "payload 组好: $OUT"
echo "  -> 烤进 StarryOS rootfs 的 /act_rknn/"
du -sh "$OUT"
