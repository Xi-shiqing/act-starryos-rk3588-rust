#!/usr/bin/env bash
# Prepare the StarryOS/RISC-V QEMU rootfs so ACT can run the full 666-frame eval inside QEMU.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QEMU_REPO="${QEMU_REPO:-/root/OScompetition/repos/act-starryos-qemu-infer}"
STARRY="${STARRY:-/root/OScompetition/repos/StarryOS}"
DISK="${DISK:-${STARRY}/make/disk.img}"

MODEL="${MODEL:-${QEMU_REPO}/models/balancedcalib_static_qdq_conv_matmul_keep_action_head_fp16.onnx}"
PARAMS="${PARAMS:-${QEMU_REPO}/deploy/cpp_onnxruntime/config/act_params.json}"
MANIFEST="${MANIFEST:-${QEMU_REPO}/deploy/cpp_onnxruntime/data/eval_manifest.csv}"
FRAME_DIR="${FRAME_DIR:-${ROOT}/data/frames}"
GUEST_ROOT="${GUEST_ROOT:-/root/proj57-act}"
GUEST_FRAME_DIR="${GUEST_ROOT}/data/dataset/videos/observation.images.fpv/chunk-000"

for f in "$DISK" "$MODEL" "$PARAMS" "$MANIFEST"; do
  [[ -f "$f" ]] || { echo "missing file: $f" >&2; exit 1; }
done
[[ -d "$FRAME_DIR" ]] || { echo "missing frame dir: $FRAME_DIR" >&2; exit 1; }

frame_count="$(find "$FRAME_DIR" -maxdepth 1 -type f -name 'frame_*.jpg' | wc -l)"
[[ "$frame_count" -eq 666 ]] || { echo "expected 666 frames, got $frame_count in $FRAME_DIR" >&2; exit 1; }

cmds="$(mktemp)"
log="$(mktemp)"
trap 'rm -f "$cmds" "$log"' EXIT

cat >"$cmds" <<EOF
mkdir ${GUEST_ROOT}
mkdir ${GUEST_ROOT}/bin
mkdir ${GUEST_ROOT}/lib
mkdir ${GUEST_ROOT}/models
mkdir ${GUEST_ROOT}/config
mkdir ${GUEST_ROOT}/data
mkdir ${GUEST_ROOT}/data/dataset
mkdir ${GUEST_ROOT}/data/dataset/videos
mkdir ${GUEST_ROOT}/data/dataset/videos/observation.images.fpv
mkdir ${GUEST_FRAME_DIR}
rm ${GUEST_ROOT}/models/balancedcalib_static_qdq_conv_matmul_keep_action_head_fp16.onnx
write ${MODEL} ${GUEST_ROOT}/models/balancedcalib_static_qdq_conv_matmul_keep_action_head_fp16.onnx
rm ${GUEST_ROOT}/config/act_params.json
write ${PARAMS} ${GUEST_ROOT}/config/act_params.json
rm ${GUEST_ROOT}/data/eval_manifest.csv
write ${MANIFEST} ${GUEST_ROOT}/data/eval_manifest.csv
EOF

while IFS= read -r frame; do
  base="$(basename "$frame")"
  {
    printf 'rm %s/%s\n' "$GUEST_FRAME_DIR" "$base"
    printf 'write %s %s/%s\n' "$frame" "$GUEST_FRAME_DIR" "$base"
  } >>"$cmds"
done < <(find "$FRAME_DIR" -maxdepth 1 -type f -name 'frame_*.jpg' | sort)

if ! debugfs -w -f "$cmds" "$DISK" >"$log" 2>&1; then
  cat "$log" >&2
  exit 1
fi

guest_count="$(
  debugfs -R "ls -p ${GUEST_FRAME_DIR}" "$DISK" 2>/dev/null \
    | grep -c 'frame_[0-9][0-9][0-9][0-9][0-9][0-9]\.jpg'
)"

echo "QEMU disk prepared:"
echo "  disk:     $DISK"
echo "  model:    ${GUEST_ROOT}/models/balancedcalib_static_qdq_conv_matmul_keep_action_head_fp16.onnx"
echo "  manifest: ${GUEST_ROOT}/data/eval_manifest.csv"
echo "  frames:   ${GUEST_FRAME_DIR}/frame_000000.jpg ... frame_000665.jpg"
echo "  count:    ${guest_count}/666 frames"
