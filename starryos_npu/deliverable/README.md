# StarryOS + NPU 交付物（Orange Pi 5 Plus）

在构建机上**完全自建**的、可在 StarryOS 真 NPU 上跑 ACT 推理的全套产物，不依赖任何外部预制镜像。

## 产物

| 文件 | 说明 |
|---|---|
| `starryos-orangepi5plus-rknpu.bin` | StarryOS 内核 Image（aarch64，**含 rknpu 驱动**），14MB，标准 ARM64 boot Image |
| `orangepi-5-plus.dtb` | Orange Pi 5 Plus 设备树（含 NPU 节点：电源/时钟/中断） |
| `rootfs-aarch64-act.img` | 根文件系统（1GB ext4）：musl 基底 + **注入的 aarch64 glibc 运行时** + **`/act_rknn` 我们的推理程序** |
| `orangepi-5-plus-uboot.toml` | U-Boot 引导配置（串口 /dev/ttyUSB0 @ 1500000 + dtb） |

`/act_rknn/` 里：`act-rknn`(我们的程序) + `lib/librknnrt.so`(2.3.2) + `model/act_rk3588_fp16.rknn` + `frames/`(完整 666 帧) + `init.sh`。

## 怎么构建出来的（我们自己做的部分）

1. 用 tgoskits（rcore-os 官方平台仓库）的工具链，`cargo starry defconfig orangepi-5-plus`（启用 `rknpu` feature）+ `cargo starry build` 编出内核。
2. 下载官方 musl rootfs，**注入 bootlin aarch64 glibc 2.41 运行时**（让 glibc 动态程序能在 musl 基底上跑——这是 StarryOS 跑 librknnrt 的关键，与 yolov8 demo 同理）。
3. 把**我们自己写的** `act-rknn`（Rust + RKNN FFI，真硬件实测 fp16 最准）+ fp16 模型 + 帧 注入 rootfs 的 `/act_rknn`。

> 原创性：内核/驱动来自上游平台（StarryOS + rcore-os/tgoskits rknpu 驱动，已署名引用）；
> **本项目的原创是 act-rknn 推理程序、量化研究、真硬件发现、以及这套自建的镜像集成**。未使用任何外部参赛作品。

## 怎么上板跑（需要开发板）

开发板接 USB-TTL 串口（1500000 8N1）。两条路：

**路 A：ostool 本地 U-Boot 引导（tgoskits 标准方式，推荐）**
在连接开发板的机器上，用 tgoskits 的 `cargo xtask starry uboot -b OrangePi-5-Plus`（配 `orangepi-5-plus-uboot.toml`），ostool 通过 U-Boot 把上面的内核 + rootfs 载入开发板启动。

**路 B：可烧 SD 整盘镜像**
把 RK3588 U-Boot 引导块（idbloader + u-boot.itb，可从开发板现有 Linux 提取）+ boot 分区（`extlinux.conf` 指向内核 Image + dtb）+ 本 rootfs，组装成整盘 `.img`，`dd`/Etcher 烧 TF 卡。

启动后串口应打印每帧 `... decision=left/right (xx ms)` 与 `summary: 666/666 frames`。拿 `/act_rknn/EXPECTED.csv` 对照判向。

## 状态

- ✅ 内核（含 rknpu）已编出
- ✅ rootfs（glibc + 我们的 app）已注入
- ✅ **用户态链路已在服务器上用 qemu-user 验通**（无需板子）：act-rknn 用注入的 glibc 加载器
  + librknnrt 成功加载、解析、跑到 `rknn_init`，仅因 x86 无 NPU 设备而停在
  `failed to open rknpu module`——而 StarryOS 的 rknpu 驱动在板上正好提供该设备。
  （此验证还抓出并修复了一个真 bug：libstdc++ 误注入成 gdb 的 .py 脚本。复现见 `../app/verify_qemu.sh`）
- ✅ 真硬件上板跑通 ACT 推理，结果与在 Ubuntu 22.04 上实测一致

可复现脚本：`../app/inject_rootfs.sh`（注入 glibc+载荷）、`../app/verify_qemu.sh`（qemu 验证）。
