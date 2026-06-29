#!/usr/bin/env python3
# 【原模型结果】在完整 666 帧上，用 ONNX Runtime 跑【原始 ACT 模型 fp32】，输出逐帧判向。
#   原模型文件：task2_rknn/model/act_rknn_4d.onnx
#       （由原始 PyTorch 权重 model.pt 经 tools/export_rknn_onnx.py 导出的 fp32 ONNX）
#   输出：results/666帧_原模型_fp32.csv  （列：frame,left,right,diff,decision）
# 判向 = left_vel − right_vel 的符号（正=右，负=左）。
import os, csv, glob
import numpy as np, onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONNX = f"{ROOT}/model/act_rknn_4d.onnx"                 # ← 原模型（fp32）
OUT  = f"{ROOT}/results/666帧_原模型_fp32.csv"
NPY, FRM = f"{ROOT}/data/npy", f"{ROOT}/data/frames"
aq01 = np.array([-0.1,0.0,0.0],np.float32); aq99 = np.array([0.2,0.2,0.0],np.float32)
def denorm(a):
    return (float((a[0]+1)*0.5*(aq99[0]-aq01[0])+aq01[0]),
            float((a[1]+1)*0.5*(aq99[1]-aq01[1])+aq01[1]))

print("="*60)
print("【原模型结果 / Original Model (fp32, ONNX Runtime)】")
print(f"  模型文件: model/act_rknn_4d.onnx")
print("="*60)

s = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
in_img = [i.name for i in s.get_inputs() if len(i.shape) >= 4][0]
in_st  = [i.name for i in s.get_inputs() if len(i.shape) == 2][0]
state  = np.load(f"{NPY}/state.npy").astype(np.float32)

rows = []
for fp in sorted(glob.glob(f"{FRM}/frame_*.jpg")):
    fn = os.path.basename(fp); stem = fn[:-4]
    img = np.load(f"{NPY}/{stem}_img.npy").astype(np.float32)
    a = s.run(None, {in_img: img, in_st: state})[0].reshape(-1)
    l, r = denorm(a); diff = l - r
    dec = "right" if diff > 0 else "left"
    rows.append((fn, f"{l:.6f}", f"{r:.6f}", f"{diff:+.6f}", dec))
    if len(rows) % 100 == 0:
        print(f"  原模型 {len(rows)}/666  {fn} -> {dec}")

os.makedirs(f"{ROOT}/results", exist_ok=True)
with open(OUT, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["frame","left","right","diff","decision"]); w.writerows(rows)
from collections import Counter
c = Counter(x[4] for x in rows)
print(f"\n【原模型结果】共 {len(rows)} 帧: left={c['left']} right={c['right']}")
print(f"  -> {OUT}")
