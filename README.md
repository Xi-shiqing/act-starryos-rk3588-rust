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

板上的 NPU 推理程序在 [`board/`](board/)：用 Rust + RKNN 运行时 C API（手写 FFI） 实现，与任务三 `act-rust` 共用预处理/后处理，交叉编译到 aarch64。服务器侧已验证交叉编译、FFI ABI 对齐、预处理与模拟器输入数值一致（corr 0.99997）；`rknn_run` 上 NPU 的实测留待上板。详见 [board/README.md](board/README.md)。

