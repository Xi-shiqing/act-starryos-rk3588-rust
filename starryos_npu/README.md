# StarryOS + NPU：集成方案与配方

> 这是集成任务，不是写驱动

tgoskits（`github.com/rcore-os/tgoskits`，dev 分支）已经把 RKNN 推理跑在 StarryOS / Orange Pi 5 Plus 的 NPU 上：
- `drivers/npu/rockchip-npu`：Rust 写的 rknpu 内核驱动（gem/job/ioctl/registers）
- `drivers/soc/rockchip/`：RK3588 时钟 + 电源管理
- StarryOS 启动包/内核有 `rknpu` feature
- `apps/starry/orangepi-5-plus-uvc-rknn/`：完整的 YOLOv8 RKNN demo，已在板上实测

StarryOS rootfs 自带 aarch64 glibc 运行时（`/usr/lib/aarch64-linux-gnu`），直接跑 glibc 动态二进制 + `librknnrt.so`。
所以我们之前担心的 musl 不兼容 librknnrt 和要自己写 NPU 驱动的问题，tgoskits 都已解决。

我们的 `board/act-rknn`（Rust + librknnrt.so，已在板子 Ubuntu 上实测 46/56）与该 YOLOv8 demo 是同类二进制，
因此 StarryOS 版无需重写、无需写驱动——把它按 demo 的方式塞进 StarryOS 镜像 rootfs 即可。

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
frames/                  56 个测试帧
init.sh                  开机命令
```

## 配方：集成 → 构建 → 烧录 → 串口运行

### 准备
- clone tgoskits（dev 分支），Rust nightly-2026-05-28，aarch64-linux-gnu 交叉工具链。
- 我们这边先 `bash scripts/package_board.sh` 生成 `board_pkg/linux`，再 `bash starryos_npu/app/assemble_payload.sh` 生成 `payload/act_rknn`。

### 把 act-rknn 作为 tgoskits app 接入
仿照 `apps/starry/orangepi-5-plus-uvc-rknn/`：让 `payload/act_rknn/` 被烤进 StarryOS rootfs 的 `/act_rknn/`，
板级配置用 `app/board-orangepi-5-plus-act.toml`（开机自动 `cd /act_rknn && ./act-rknn --model model/act_rk3588_fp16.rknn --image frames`）。

> librknnrt 版本匹配：优先用 tgoskits 的 YOLOv8 demo 里那份 `librknnrt.so`（与镜像内 rknpu 驱动匹配），
> 替换我们 `lib/` 下的版本，避免运行时版本不一致。

### 构建带 rknpu 的 StarryOS Orange Pi 5 Plus 镜像
```bash
# 启用 rknpu feature 的板级构建（参考 yolov8 demo 的 README）
cargo xtask starry app board -t orangepi-5-plus-act -b OrangePi-5-Plus \
  --board-config <本仓库>/starryos_npu/app/board-orangepi-5-plus-act.toml
```
（tgoskits 默认走 ostool 远程板卡服务做自动化测试；要**可烧 TF 卡的整盘镜像**，用上游 tgoskits 的烧录流程
/ 整盘镜像产出步骤，把内核+rootfs 写进 SD。）

### 烧录 + 上板 + 串口
- 将镜像烧到 TF 卡 → 插开发板上电。
- 串口（USB-TTL，1500000 baud）看输出；开机会自动跑（board toml 的 `shell_init_cmd`），
  或手动 `sh /act_rknn/init.sh`。
- 期望串口打印每帧 `... decision=left/right (xx ms)` 和 `summary: 56/56 frames, avg inference .. ms`，
  最后 `ACT_NPU_DONE`。拿 `EXPECTED.csv` 肉眼对照判向。

## 验收目标

StarryOS 串口上跑出 ACT 的 NPU 推理判向结果（与板子 Ubuntu 上 fp16 的 46/56 对齐），
即完成任务二「在 StarryOS 上用 NPU 跑通 ACT 推理」。

## 风险 / 待确认

- tgoskits 出可烧 TF 整盘镜像的具体命令（远程板卡服务之外的路径）有待进一步确认。
- librknnrt 与镜像内 rknpu 驱动版本匹配。
- 构建侧无 StarryOS 板可自测，需在板上烧卡 + 串口回传日志迭代。
