# StarryOS + NPU：集成方案与配方

> 这是集成任务，不是写驱动

tgoskits（`github.com/rcore-os/tgoskits`，dev 分支）已经把 RKNN 推理跑在 StarryOS / Orange Pi 5 Plus 的 NPU 上：
- `drivers/npu/rockchip-npu`：Rust 写的 rknpu 内核驱动（gem/job/ioctl/registers）
- `drivers/soc/rockchip/`：RK3588 时钟 + 电源管理
- StarryOS 启动包/内核有 `rknpu` feature
- `apps/starry/orangepi-5-plus-uvc-rknn/`：完整的 YOLOv8 RKNN demo，已在板上实测

StarryOS rootfs 自带 aarch64 glibc 运行时（`/usr/lib/aarch64-linux-gnu`），直接跑 glibc 动态二进制 + `librknnrt.so`。
所以我们之前担心的 musl 不兼容 librknnrt 和要自己写 NPU 驱动的问题，tgoskits 都已解决。

我们的 `board/act-rknn`（Rust + librknnrt.so，已在板子 Ubuntu 上实测 46/56，并在 StarryOS 上跑通 666/666）与该 YOLOv8 demo 是同类二进制，因此 StarryOS 版无需重写、无需写驱动——把它按 demo 的方式塞进 StarryOS 镜像 rootfs 即可。

## 本目录内容

```
app/
  board-orangepi-5-plus-act.toml   StarryOS 板级配置：开机自动跑 act-rknn 批量推理，含成功/失败正则
  init.sh                          手动串口跑的命令
  assemble_payload.sh              从 board_pkg/linux 组装 /act_rknn 载荷
  payload/act_rknn/                组装出的载荷（烤进 rootfs 的 /act_rknn/，gitignore）
research/
  DRIVER_DESIGN.md                 （备份）从零写驱动的设计与可行性分析
  rknpu_ioctl.h                    rknpu 内核-用户态 ioctl 接口（参考）
```

载荷 `/act_rknn/` 结构：
```
act-rknn                 我们的推理程序（aarch64-gnu，链 librknnrt.so）
lib/librknnrt.so         RKNN 运行时（建议改用 tgoskits 镜像里匹配 rknpu 驱动的那份，见下）
model/act_rk3588_fp16.rknn  部署模型（真硬件实测最准）
frames/                  完整 666 帧 JPEG 测试序列（保留作展示/复现旧输入路径）
frames_rgb224.bin        完整 666 帧预 resize 的连续 RGB224 输入包（默认运行路径，绕开板上 JPEG 解码卡点）
init.sh                  开机命令
```

## 配方：集成 → 构建 → 烧录 → 串口运行

### 准备
- clone tgoskits（dev 分支），Rust nightly-2026-05-28，aarch64-linux-gnu 交叉工具链。
- 我们这边先 `bash scripts/package_board.sh` 生成 `board_pkg/linux`，再 `bash starryos_npu/app/assemble_payload.sh` 生成 `payload/act_rknn`。

### 把 act-rknn 作为 tgoskits app 接入
仿照 `apps/starry/orangepi-5-plus-uvc-rknn/`：让 `payload/act_rknn/` 被烤进 StarryOS rootfs 的 `/act_rknn/`，
板级配置用 `app/board-orangepi-5-plus-act.toml`，开机进入 `/act_rknn` 后运行 `init.sh`。当前默认路径读取 `frames_rgb224.bin` 连续 RGB224 输入包；`frames/` 下的 JPEG 保留作旧输入路径复现与排障。

> librknnrt 版本匹配：优先用 tgoskits 的 YOLOv8 demo 里那份 `librknnrt.so`（与镜像内 rknpu 驱动匹配），
> 替换我们 `lib/` 下的版本，避免运行时版本不一致。

### 构建带 rknpu 的 StarryOS Orange Pi 5 Plus 镜像
```bash
# 启用 rknpu feature 的板级构建（参考 yolov8 demo 的 README）
cargo xtask starry app board -t orangepi-5-plus-act -b OrangePi-5-Plus \
  --board-config <本仓库>/starryos_npu/app/board-orangepi-5-plus-act.toml
```
（tgoskits 默认走 ostool 远程板卡服务做自动化测试；本项目最终提交采用**可烧 TF 卡的整盘镜像**，由 `app/build_sd_image.sh` 把引导块、内核、dtb 和 rootfs 组装成 `sdcard/starryos-act-orangepi5plus-sd.img.gz`。）

### 烧录 + 上板 + 串口
- 将镜像烧到 TF 卡 → 插开发板上电。
- 串口（USB-TTL，1500000 baud）看输出；开机会自动跑（board toml 的 `shell_init_cmd`），
  或手动运行：
  ```sh
  sh /act_rknn/init.sh
  ```
- 期望串口打印 `image source: frames_rgb224.bin`，每帧 `... decision=left/right (xx ms)` 和 `summary: 666/666 frames, avg inference .. ms`，
  最后 `ACT_NPU_DONE`。拿 `EXPECTED.csv` 肉眼对照判向。

## 验收目标

StarryOS 串口上跑出 ACT 的 NPU 推理判向结果，完整跑完 666 帧，并与原模型 fp32 保持高一致性：
全 666 帧判向一致 642/666 = 96.4%，明确转向帧 489/489 = 100%，平均 NPU 推理约 34.6 ms/帧。
这即完成任务二「在 StarryOS 上用 NPU 跑通 ACT 推理」。

## 风险 / 后续

- 旧 JPEG 小文件输入路径在第 96 帧附近卡住，定位为 StarryOS 上 JPEG 文件读取/解码/预处理链路的累计资源耗尽或阻塞；目前尚未修复旧 JPEG 路径的 StarryOS 内部根因。当前 RGBPack 版把解码前移到服务器，板上顺序读连续二进制帧，是绕开方案，已跑通 666/666。
- 只读挂载是绕开该平台 SD 写入死锁的工程方案。当前 demo 全程只读，对推理展示足够；若后续要接真实小车并写日志/状态，需要把日志放到 tmpfs，或继续修 StarryOS 的 SD/ext4 写入路径。
