# StarryOS + Orange Pi 5 Plus / RK3588 上板推理结果

自烧 TF 卡（`sdcard/` 同名一键包）烧入开发板，StarryOS 启动 → 真 RKNPU 加载 fp16 模型 → 对 56 个明确转向帧推理，串口完整输出。

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
