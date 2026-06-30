# StarryOS + Orange Pi 5 Plus / RK3588 上板推理结果

自烧 TF 卡（`sdcard/` 同名一键包）烧入开发板，StarryOS 启动 → 真 RKNPU 加载 fp16 模型 → 对 56 个明确转向帧推理，串口完整输出。

## 666 帧 RGBPack 版

最新镜像把完整 666 帧预处理为 `frames_rgb224.bin` 连续 RGB224 输入包，板上不再逐帧 JPEG 解码。该版本已在 Orange Pi 5 Plus / StarryOS / 真 RKNPU 上跑完：

```text
image source: frames_rgb224.bin (pre-resized RGB224 pack, no JPEG decode on board)
---- smoke: first 120 frames, must pass frame_000096 ----
summary: 120/120 frames, avg inference 34.8 ms
---- full open-loop (666 frames, state=0,0) ----
summary: 666/666 frames, avg inference 34.6 ms
---- full closed-loop (666 frames, feedback) ----
summary: 666/666 frames, avg inference 34.6 ms
==== ACT NPU inference end (exit=0) ====
```

产物：

- `results/666帧_板子-StarryOS真NPU_RGBPack_开环.csv`
- `results/666帧_板子-StarryOS真NPU_RGBPack_闭环.csv`
- `results/gif/666帧_板子-StarryOS真NPU_RGBPack_开环.gif`
- `results/gif/666帧_板子-StarryOS真NPU_RGBPack_闭环.gif`

开环与原模型 fp32 对照：全 666 判向一致 642/666 = 96.4%，明确转向帧（`|diff|>=0.002`）一致 489/489 = 100%，`diff` 相关系数 0.9954。

## 第 96 帧卡点定位

旧 JPEG 路径在串口中停在：

```text
frame_000095.jpg ... OK
[dec]
```

`[dec]` 在读取/解码下一帧前打印；若进入 NPU 会继续出现 `[npu>]`。因此旧版本卡在 `frame_000096.jpg` 的 JPEG 文件读取/解码/预处理阶段，尚未进入 RKNPU。

RGBPack 版本同一位置为：

```text
[pack>][pack<][npu>][npu<][rd]frame_000096.jpg ... OK
```

所以第 96 帧问题不在 ACT 模型、RKNPU 或 RKNN `rknn_run`，而在 StarryOS 上逐 JPEG 小文件输入链路。它表现为长时间反复读取/解码 JPEG 小文件后，在第 96 帧附近停在输入阶段；现阶段只能判断为该链路存在累计资源耗尽或阻塞问题，尚未精确定位到 StarryOS 内部是哪一个 VFS/ext4/buffer/cache/JPEG 解码相关结构或锁导致卡住。

需要说明：RGBPack 并不是对该 StarryOS JPEG 小文件输入问题的根因修复，而是工程绕过方案。我们把 JPEG 解码和 resize 前移到服务器，板上只顺序读取连续 `frames_rgb224.bin` 二进制帧，从而避开旧输入链路；该方案已证明 StarryOS + 真 RKNPU 可以稳定完成 666 帧推理。旧 JPEG 路径仍保留为后续排查对象。

## 56 帧早期结果

## 关键启动日志（成功标志）
```
NPU registered successfully
/dev/console bound to ttyS2
ext4 filesystem is in error state; mounting read-only without journal replay   ← 故意置位以走只读、绕开 SD 写卡死
Welcome to Starry OS!
==== ACT NPU inference start ====
rknn_init OK (ctx=1074006784)
...
summary: 56/56 frames, avg inference 26.6 ms
==== ACT NPU inference end (exit=0) ====
root@starry:/root #
```

## 判向精度
| 指标 | StarryOS+真NPU | 对照：Linux+真NPU（board_rk3588_real.md） |
|---|---|---|
| vs 标准答案 gt | 46/56 = 82.1% | 46/56 |
| 右召回（gt右=19） | 18/19 | 18/19 |
| 左召回（gt左=37） | 28/37 | — |
| vs fp32 模拟器 | 55/56 | — |
| 平均时延 | 26.6 ms/帧 | 24.7 ms/帧 |

StarryOS 结果与普通 Linux 真机实测逐项一致，证明 StarryOS 真 NPU 推理可复现、fp16 量化近乎无损（vs fp32 基线 55/56）。10 帧与 gt 不一致者全为轮速差极小（diff ±0.001~0.002）的临界帧，与 fp16 数值噪声同量级（已知现象，非缺陷）。

复现脚本：`app/build_sd_image.sh`；引导块 `bootloader/bootblob.bin`；去 bootargs 的 dtb `app/orangepi-5-plus.nobootargs.dtb`。
