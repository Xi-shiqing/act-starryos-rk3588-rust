set -euo pipefail
HERE="$(cd "$(dirname "$0")/.."; pwd)"          # .../starryos_npu
DELIV="$HERE/deliverable"
BOOTBLOB="$HERE/bootloader/bootblob.bin"
OUT="$HERE/sdcard"
WORK="$(mktemp -d)"
R="$WORK/root.img"
IMG="$OUT/starryos-act-orangepi5plus-sd.img"
mkdir -p "$OUT"
cleanup(){ umount "$WORK/m" 2>/dev/null||true; [ -n "${LOOP:-}" ]&&losetup -d "$LOOP" 2>/dev/null||true; rm -rf "$WORK"; }
trap cleanup EXIT

# 1) 取 rootfs，收缩到 320MB，去 orphan_file，加 /boot + autorun 钩子
cp "$DELIV/rootfs-aarch64-act.img" "$R"
e2fsck -fy "$R" >/dev/null 2>&1 || true
resize2fs "$R" 81920                       # 81920 x 4k = 320MB
# 对齐板子 known-good ext4 特性集（StarryOS 已验证能 sync）：去 orphan_file、加 metadata_csum
tune2fs -O ^orphan_file "$R"               # rsext4 不维护 orphan_file 写回 → 必去
tune2fs -O metadata_csum "$R"              # 基准有 metadata_csum；rsext4 支持(config.rs 默认集含)
e2fsck -fy "$R" >/dev/null 2>&1 || true
mkdir -p "$WORK/m"; mount "$R" "$WORK/m"
mkdir -p "$WORK/m/boot/dtb" "$WORK/m/boot/extlinux"
cp "$DELIV/starryos-orangepi5plus-rknpu.bin" "$WORK/m/boot/Image"
# 关键：dtb 必须去掉自带的 /chosen/bootargs。否则本板 rockchip u-boot 走 "android 改写" 路径,
# 把 console 强成 tty1(StarryOS device_id_from_bootargs 只认硬件串口,tty1→/dev/console 不绑→
# act-rknn println 失败 Rust panic)。去掉后 u-boot 像对板子原版 Ubuntu 那样原样用 extlinux append,
# 其中 console=ttyS2,1500000 得以存活 → StarryOS 绑定 ttyS2 → 串口看到推理输出。
python3 - "$DELIV/orangepi-5-plus.dtb" "$WORK/m/boot/dtb/orangepi-5-plus.dtb" <<'PY'
import struct, sys
d=bytearray(open(sys.argv[1],"rb").read())
magic,total,off_s,off_str,_,_,_,_,sz_str,sz_s=struct.unpack(">10I",d[:40]); assert magic==0xd00dfeed
nameoff=d[off_str:off_str+sz_str].find(b"bootargs\x00"); assert nameoff>=0
p=off_s; end=off_s+sz_s
while p<end:
    tok=struct.unpack(">I",d[p:p+4])[0]
    if tok==1: p+=4; p=(d.index(b"\x00",p)+1+3)&~3
    elif tok in (2,4): p+=4
    elif tok==9: break
    elif tok==3:
        plen,pno=struct.unpack(">II",d[p+4:p+12])
        if pno==nameoff:
            for w in range(3+((plen+3)//4)): d[p+w*4:p+w*4+4]=struct.pack(">I",4)
        p+=12+((plen+3)&~3)
    else: p+=4
open(sys.argv[2],"wb").write(d)
PY
# extlinux(镜像板子原版 Ubuntu 的工作启动路径)：append 带 console=ttyS2,1500000
cat > "$WORK/m/boot/extlinux/extlinux.conf" <<'CONF'
default starry
menu title StarryOS ACT NPU
prompt 0
timeout 10
label starry
	menu label StarryOS ACT NPU
	linux /boot/Image
	fdt /boot/dtb/orangepi-5-plus.dtb
	append console=ttyS2,1500000 console=tty1 root=PARTLABEL=rootfs rootfstype=ext4 rootwait earlycon=uart8250,mmio32,0xfeb50000
CONF
# 推理脚本走正常 stdout（console=ttyS2 绑定后即到串口）+ autorun 钩子
cat > "$WORK/m/act_rknn/init.sh" <<'RUN'
#!/bin/sh
cd /act_rknn
export LD_LIBRARY_PATH=/act_rknn/lib:/usr/local/lib:/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}
echo "==== ACT NPU inference start ===="
./act-rknn --model model/act_rk3588_fp16.rknn --image frames --state 0 0
echo "==== ACT NPU inference end (exit=$?) ===="
RUN
chmod +x "$WORK/m/act_rknn/init.sh"
cat > "$WORK/m/usr/bin/starry-run-case-tests" <<'HOOK'
#!/bin/sh
echo "==== ACT NPU autorun ===="
sh /act_rknn/init.sh
HOOK
chmod +x "$WORK/m/usr/bin/starry-run-case-tests"
sync; umount "$WORK/m"
e2fsck -fy "$R" >/dev/null 2>&1 || true
# 关键：置 EXT4_ERROR_FS 标志，逼 StarryOS 走只读挂载分支(fs/ext4/rsext4/fs.rs device_has_error_state→
# mount_readonly_no_replay)，从而跳过挂载后那次会卡死的 sync_filesystem(mount.rs `if !readonly`)。
# act-rknn 只读不写盘，只读根完全够用；也彻底绕开 SD 写入卡死(无论烂卡还是 dwmmc 写 bug)。
# 必须是最后一步——之后绝不能再 e2fsck/读写挂载，否则标志被清。
debugfs -w -R 'ssv state 2' "$R" >/dev/null 2>&1

# 2) 整盘 GPT：引导块@扇区64，根分区@扇区32768（GPT 名=rootfs）
truncate -s 360M "$IMG"
sgdisk --zap-all "$IMG" >/dev/null 2>&1 || true
sgdisk -n 1:32768:0 -t 1:8300 -c 1:"rootfs" "$IMG" >/dev/null
dd if="$BOOTBLOB" of="$IMG" bs=512 seek=64    conv=notrunc status=none
dd if="$R"        of="$IMG" bs=512 seek=32768 conv=notrunc status=none

# 3) 压缩 + 校验
gzip -cf "$IMG" > "$IMG.gz"
( cd "$OUT" && sha256sum "$(basename "$IMG")" > SHA256SUMS )
echo "done -> $IMG(.gz)"; ls -lh "$OUT"
