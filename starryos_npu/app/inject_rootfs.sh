#!/usr/bin/env bash
# 往 musl rootfs 注入 aarch64 glibc 运行时 + 我们的 /act_rknn 载荷。
# 用法: sudo bash inject_rootfs.sh <rootfs.img>
set -e
IMG="${1:?用法: inject_rootfs.sh <rootfs.img>}"
SR=/root/OScompetition/toolchains/aarch64--glibc--stable-2025.08-1/aarch64-buildroot-linux-gnu/sysroot
ROOT="$(cd "$(dirname "$0")/../.."; pwd)"
PAYLOAD="$ROOT/starryos_npu/app/payload/act_rknn"
M=$(mktemp -d)
mount -o loop "$IMG" "$M"
trap 'sync; umount "$M"; rmdir "$M"' EXIT

# 1) glibc 加载器 + 运行库（解引用真身；libstdc++ 必须取 .so.6.0.xx ELF，别取 -gdb.py）
cp -L "$SR/lib/ld-linux-aarch64.so.1" "$M/lib/ld-linux-aarch64.so.1"
mkdir -p "$M/usr/lib/aarch64-linux-gnu"
for so in libc.so.6 libm.so.6 libpthread.so.0 libdl.so.2 librt.so.1 libgcc_s.so.1; do
  cp -L "$(find "$SR/lib" "$SR/usr/lib" -name "$so" -type f | head -1)" "$M/usr/lib/aarch64-linux-gnu/$so"
done
STDCPP=$(find "$SR" -name "libstdc++.so.6.[0-9]*" -type f ! -name "*.py" | head -1)
cp -L "$STDCPP" "$M/usr/lib/aarch64-linux-gnu/libstdc++.so.6"

# 2) /act_rknn 载荷
rm -rf "$M/act_rknn"; mkdir -p "$M/act_rknn"
cp -r "$PAYLOAD"/* "$M/act_rknn/"
chmod +x "$M/act_rknn/act-rknn" "$M/act_rknn/init.sh" 2>/dev/null || true
echo "注入完成: glibc 运行时 + /act_rknn -> $IMG"
