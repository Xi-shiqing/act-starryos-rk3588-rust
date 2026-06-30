# 关于在 RK3588 上的模型转换与量化

本项目在香橙派 **RK3588** 的 NPU 上推理。
NPU 推理本身只能在板子上执行，但模型转换、量化、精度验证可以在服务器上用官方 [rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2) 的模拟器完成。本项目当前阶段主要完成这一部分的工作。

## 工作概览

把官方 ACT 模型转成 RK3588 的 RKNN 格式，并系统对比了几种量化策略的判向精度（左转/右转）。最终选择把占算力大头的视觉卷积骨干量化成 INT8、transformer 头保留 float16 的混合量化。

> 评测集 = 官方 `eval_manifest.csv` 中全部 **56 个明确转向帧**；判向 = `left_vel − right_vel` 的符号；state=(0,0)。
> "与 fp32 一致率" = 与 fp32 ONNX 逐帧判向相同的比例。

### 全部改成 INT8 会掉精度，而混合量化能救回

ACT 输出的极小的轮速差，对输出端的量化噪声极其敏感。
- full-INT8 把 state 输入、action 输出、整个 transformer 全部量化成 INT8，细粒度回归输出的符号判别力被量化噪声淹没。
- 混合量化只量化 vision 卷积（ResNet，50 个 conv 层，是 224×224 图像推理的 FLOPs 大头），把对输出敏感的 transformer encoder/decoder + action_head 保留 float16。结果与 fp32 逐帧判向完全一致，diff 相关系数 0.9985，实测优于纯 fp16。

实验还发现只要量化任何一个 encoder 层就会出错，但是只量化卷积骨干则无损失。

## 上板实测与模拟器结论结论不同

后续拿到 Orange Pi 5 Plus 后，在 Ubuntu 22.04 / RKNPU driver v0.9.6 进行实测，上面基于 RKNN 模拟器得出的混合量化是无损最优的结论不成立：

| 模型(真 NPU) | 判向 vs gt | 右召回(gt右=19) | 时延 | 峰值内存 |
|---|---|---|---|---|
| fp16 | 46/56 | 18/19 | 24.7 ms | 206 MB |
| hybrid | 40/56 | 3/19 | 20 ms | ~200 MB |
| full-int8 | 31/56 | 18/19 | ~18 ms | ~180 MB |

模拟器判定无损的混合量化，在真 NPU 上反而最差。根因：本模型轮速差信号极小（~0.005），与真硬件量化噪声（~0.006）同量级，足以使得临界帧方向相反。而 RKNN 模拟器数值精度高于真硬件，在模拟器上无损到板子上就不成立了。所以最终部署模型选 fp16（46/56 ≈ fp32 基线 47/56，已跑满模型精度上限）。
完整实测见 [`results/board_rk3588_real.md`](results/board_rk3588_real.md)。

## 在 StarryOS + RK3588 上跑通

进一步把推理从普通 Linux 推进到 StarryOS：做出一张自包含、可直接烧录的整盘 TF 卡镜像，插卡上电即从 StarryOS 启动、在 RKNPU 上加载 fp16 模型并完成完整 666 帧连续推理，经调试串口打印判向。
最新 RGBPack 版已在 StarryOS + 真 RKNPU 上跑通 666/666 帧：开环、闭环均完整结束，平均 NPU 推理约 34.6 ms/帧；开环与原模型 fp32 对照，全 666 帧判向一致 642/666 = 96.4%，明确转向帧（`|diff| >= 0.002`）一致 489/489 = 100%，`diff` 相关系数 0.9954。早期 56 个明确转向帧结果与普通 Linux 真机实测逐项一致：判向 vs 真值 46/56，右召回 18/19。

- 可烧录镜像：[Release v1.0.0](https://github.com/Xi-shiqing/act-starryos-rk3588-rust/releases/tag/v1.0.0)
- StarryOS 集成与上板运行：[`starryos_npu/README.md`](starryos_npu/README.md)
- 上板结果存档：[`starryos_npu/RESULT_starryos_board.md`](starryos_npu/RESULT_starryos_board.md)
- 上板调试历程：[`starryos_npu/上板调试历程.md`](starryos_npu/上板调试历程.md)
- 镜像构建/重建：[`starryos_npu/app/build_sd_image.sh`](starryos_npu/app/build_sd_image.sh)、[`starryos_npu/如何重建镜像.md`](starryos_npu/如何重建镜像.md)

## 提交材料索引

- 设计思路、实现描述与代码说明：本 README、[`board/README.md`](board/README.md)、[`starryos_npu/README.md`](starryos_npu/README.md)。
- 问题定位与解决方法：[`starryos_npu/上板调试历程.md`](starryos_npu/上板调试历程.md)、[`starryos_npu/RESULT_starryos_board.md`](starryos_npu/RESULT_starryos_board.md)。
- 第三方来源、基础版本、增量贡献与授权：[`第三方来源与许可.md`](第三方来源与许可.md)、[`基础版本与增量贡献.md`](基础版本与增量贡献.md)、[`LICENSE`](LICENSE)。
- 技术文档、答辩 PPT 与演示视频统一采用 CC-BY-SA 4.0；源代码采用 Apache License 2.0。

## 演示复现命令

### 镜像构建与 release 文件

录制“如何烧出可上板镜像”时，先展示重建文档，再执行构建链路：

```bash
cd /root/OScompetition/task2_rknn
sed -n '1,90p' starryos_npu/如何重建镜像.md
bash scripts/package_board.sh
bash starryos_npu/app/assemble_payload.sh
bash starryos_npu/app/inject_rootfs.sh starryos_npu/deliverable/rootfs-aarch64-act.img
bash starryos_npu/app/build_sd_image.sh
ls -lh starryos_npu/sdcard/starryos-act-orangepi5plus-sd.img.gz
gzip -t starryos_npu/sdcard/starryos-act-orangepi5plus-sd.img.gz
```

`starryos-act-orangepi5plus-sd.img.gz` 即 release 和队友上板测试使用的镜像。Etcher 可以直接选择 `.img.gz` 烧录，不需要先解压。

### 官方原模型 666 帧

用 Python + ONNX Runtime 跑由原始 PyTorch 权重导出的 fp32 ONNX，生成 666 帧参考结果：

```bash
cd /root/OScompetition/task2_rknn
python3 tools/run_666_原模型_fp32.py
wc -l results/666帧_原模型_fp32.csv
tail -n 5 results/666帧_原模型_fp32.csv
```

`wc -l` 应为 `667`，表示 1 行表头 + 666 帧结果。

### QEMU + StarryOS 666 帧

QEMU/StarryOS 版本的实现说明见 `/root/OScompetition/repos/act-starryos-qemu-infer/deploy/cpp_onnxruntime/README.md` 和 `DELIVERABLE.md`。录制时必须在 QEMU 启动的 StarryOS shell 内运行静态链接的 `act_static`，不要用 host 侧脚本替代。

先把 666 帧评测数据、manifest 和 ONNX 模型写入 QEMU 使用的 `disk.img`：

```bash
cd /root/OScompetition/task2_rknn
bash scripts/prepare_qemu_666_disk.sh
```

启动 QEMU/StarryOS：

```bash
cd /root/OScompetition/repos/StarryOS/make
qemu-system-riscv64 \
  -m 1G -smp 1 -machine virt -bios default \
  -kernel ../StarryOS_riscv64-qemu-virt.bin \
  -device virtio-blk-pci,drive=disk0 \
  -drive id=disk0,if=none,format=raw,file=disk.img \
  -nographic -monitor none
```

如果 QEMU 报 `Failed to get "write" lock`，说明已有另一个 QEMU 正在占用 `disk.img`。先在旧 QEMU 窗口按 `Ctrl+A`，再按 `X` 退出；如果找不到旧窗口，再执行：

```bash
pkill -f 'qemu-system-riscv64.*disk.img'
```

如果不想杀旧 QEMU，也可以复制一份运行盘，把启动命令里的 `file=disk.img` 改成 `file=disk-run.img`：

```bash
cd /root/OScompetition/repos/StarryOS/make
cp disk.img disk-run.img
```

进入 StarryOS shell 后运行完整 666 帧评测：

```sh
cd /root/proj57-act
export LD_LIBRARY_PATH=/root/proj57-act/lib:/lib
bin/act_static \
  --model models/balancedcalib_static_qdq_conv_matmul_keep_action_head_fp16.onnx \
  --params config/act_params.json \
  --eval-manifest data/eval_manifest.csv \
  --dataset-root data/dataset \
  --threads 1 \
  --eval-turn-eps 0.005 \
  --track-allocator
```

建议先跑一帧 smoke test，确认静态二进制和模型能正常启动：

```sh
cd /root/proj57-act
bin/act_static \
  --model models/balancedcalib_static_qdq_conv_matmul_keep_action_head_fp16.onnx \
  --image data/dataset/videos/observation.images.fpv/chunk-000/frame_000000.jpg \
  --params config/act_params.json \
  --state 0 0 \
  --threads 1 \
  --warmup 0 \
  --runs 1 \
  --deadband 0.005
```

QEMU TCG 纯 CPU 推理很慢，完整 666 帧可能需要较长时间；退出 QEMU 用 `Ctrl+A`，再按 `X`。不要使用动态链接的 `bin/act_ort_infer` 做录制演示，当前 StarryOS 环境下它可能在动态加载阶段 segfault。运行过程中如果屏幕反复刷 `starry_kernel::syscall::mm::mmap`，通常是 ONNX Runtime 加载模型和分配内存时触发的内核调试日志；只要没有 `segmentation fault`，且没有回到 `starry:~#`，就说明进程仍在运行。

### 结果文件

官方原模型和板子 StarryOS 真 NPU 的 666 帧 CSV/GIF 保存在 `results/`；QEMU+StarryOS 的 666 帧结果以 QEMU 终端中的 `act_static --eval-manifest` 输出为准。

```bash
cd /root/OScompetition/task2_rknn
ls -lh results/666帧_原模型_fp32*.csv
ls -lh results/666帧_板子-StarryOS真NPU_RGBPack_*.csv
wc -l results/666帧_原模型_fp32.csv results/666帧_板子-StarryOS真NPU_RGBPack_开环.csv results/666帧_板子-StarryOS真NPU_RGBPack_闭环.csv
ls -lh results/gif/666帧_原模型_fp32_*.gif results/gif/666帧_板子-StarryOS真NPU_RGBPack_*.gif
```

## 目录结构

```
model/
  act_rknn_4d.onnx                 NPU 友好的 4D 输入 ONNX（image[1,3,224,224]+state[1,2]）
  act_rk3588_fp16.rknn             fp16（最终部署：真硬件实测最准，101.7MB）
  act_rk3588_hybrid_backbone.rknn  混合量化：骨干INT8+transformerFP16（90MB，模拟器无损但真硬件掉精度）
  act_rk3588_int8.rknn             full-INT8（最小52.7MB，有损，作对照）
tools/
  export_rknn_onnx.py              PyTorch checkpoint -> 4D 输入 ONNX
  _onnx_compat.py                  onnx>=1.16/新版 protobuf 的兼容垫片（见下）
  eval_reference.py                fp32 ONNX 基线（ORT）
  convert_and_sim.py               ONNX -> RKNN(fp16/int8) + 模拟器判向评测
  hybrid_step1.py / hybrid_step2.py 混合量化两步流程
  quant_sweep.py                   量化粒度/算法扫描
  gen_inputs.py                    预处理 666 帧为 npy + 校准集
data/
  frames/                          官方数据集 666 帧（含 56 转向帧，从 HF bobodai/proj57_dataset 下载）
  npy/                             预处理好的归一化输入（可由 gen_inputs.py 重生成）
  dataset_calib.txt                INT8 量化校准集（178 帧）
results/
  ref_fp32.csv / sim_*.csv         各模型逐帧判向明细
  board_rk3588_real.md             真 NPU（普通 Linux）实测
board/                             板上 NPU 推理程序（Rust + RKNN FFI，见下）
starryos_npu/                      StarryOS + 真 NPU 整盘镜像、构建脚本与上板文档（见上）
```

## 复现步骤（在 x86 服务器上，无需板子）

```bash
# 0) 环境：rknn-toolkit2 2.3.2 装在 ../rknn-libs（--no-deps + 轻量依赖）
export PYTHONPATH=$(pwd)/../rknn-libs:$(pwd)/tools

# 1) 导出 4D 输入 ONNX（从 /tmp/model.pt）
python3 tools/export_rknn_onnx.py /tmp/model.pt model/act_rknn_4d.onnx

# 2) 预处理评测/校准数据
python3 tools/gen_inputs.py

# 3) fp32 基线 + fp16 / full-int8
python3 tools/eval_reference.py
python3 tools/convert_and_sim.py fp16
python3 tools/convert_and_sim.py int8

# 4) 混合量化（推荐）：骨干INT8 + transformer FP16
python3 tools/hybrid_step1.py
python3 tools/hybrid_step2.py "/encoder,decoder,state_encoder,action_head,action,/m/Concat" hybrid_backbone
```

## 环境兼容性说明

rknn-toolkit2 2.3.2 的依赖与 Python 3.12 / 新版 onnx·protobuf 有两处冲突，已用 `tools/_onnx_compat.py` 解决：
1. **`onnx.mapping`**：onnx ≥1.16 移除，用 `onnx.helper` 重建 `TENSOR_TYPE_TO_NP_TYPE` 等并注册回去；
2. **`onnx.helper.strip_doc_string`**：在新版 protobuf 下用了已改名的 `FieldDescriptor.label`，替换为不依赖该 API 的安全版（文档字符串与转换结果无关）。

另：模型用 `--no-deps` 安装，跳过 tensorflow/torch（ONNX 前端用不到）。

## 板上推理程序

板上的 NPU 推理程序在 [`board/`](board/)：用 Rust + RKNN 运行时 C API（手写 FFI） 实现，与任务三 `act-rust` 共用预处理/后处理，交叉编译到 aarch64。已在真 RK3588 NPU 上跑通 `rknn_run`——既在普通 Linux（见 [`results/board_rk3588_real.md`](results/board_rk3588_real.md)），也在 StarryOS。详见 [board/README.md](board/README.md)。
