#!/usr/bin/env bash
# 用 qemu-user 在 x86 上验证 rootfs 里的 act-rknn 能加载+链接+跑到 rknn_init（NPU 缺设备失败属正常）。
set -e
IMG="${1:?用法: verify_qemu.sh <rootfs.img>}"
QEMU="${QEMU:-/tmp/qemu-aarch64-static}"
M=$(mktemp -d); mount -o loop "$IMG" "$M"
trap 'umount "$M"; rmdir "$M"' EXIT
FRAME=$(ls "$M"/act_rknn/frames/*.jpg | head -1)
QEMU_LD_PREFIX="$M" "$QEMU" -E LD_LIBRARY_PATH=/act_rknn/lib:/usr/lib/aarch64-linux-gnu \
  "$M/act_rknn/act-rknn" --model "$M/act_rknn/model/act_rk3588_fp16.rknn" --image "$FRAME" --state 0 0 || true
echo "（期望：打印 loading rknn model... 然后 'failed to open rknpu module' —— 用户态链路 OK，只差真 NPU）"
