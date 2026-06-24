#!/usr/bin/env python3
# 把所有评测帧统一预处理成 RKNN 输入用的 .npy：
#   image: [1,3,224,224] fp32（Resize224 + ImageNet 归一化，与官方/任务三一致）
#   state: [1,2] fp32（QUANTILES 归一化，state=(0,0)）
# 同时生成：量化校准集 dataset_calib.txt（均匀抽样 N 帧），评测列表 eval_turn.txt（56 转向帧）。
import csv, os, numpy as np
from PIL import Image
from torchvision import transforms

MANIFEST="repos/act-starryos-qemu-infer/deploy/cpp_onnxruntime/data/eval_manifest.csv"
FRAMES="task2_rknn/data/frames"; NPY="task2_rknn/data/npy"
os.makedirs(NPY,exist_ok=True)
q01=np.array([-0.079,0.0],np.float32); q99=np.array([0.2,0.2],np.float32)
TF=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
d=np.where(q99-q01==0,1e-8,q99-q01); st=(2*(np.zeros(2,np.float32)-q01)/d-1).astype(np.float32).reshape(1,2)
np.save(f"{NPY}/state.npy", st)

rows=list(csv.DictReader(open(MANIFEST)))
all_fn=[]; turn_fn=[]
for r in rows:
    fn=r['image_path'].split('/')[-1]; stem=fn[:-4]
    img=TF(Image.open(f"{FRAMES}/{fn}").convert("RGB")).unsqueeze(0).numpy().astype(np.float32)
    np.save(f"{NPY}/{stem}_img.npy", img)
    all_fn.append(stem)
    if float(r['gt_left_vel'])!=float(r['gt_right_vel']): turn_fn.append(stem)

# 校准集：均匀抽 120 帧 + 全部 56 转向帧（去重），写成 rknn dataset 格式（每行: img.npy state.npy）
step=max(1,len(all_fn)//120)
calib=sorted(set(all_fn[::step]) | set(turn_fn))
with open("task2_rknn/data/dataset_calib.txt","w") as f:
    for s in calib: f.write(f"{NPY}/{s}_img.npy {NPY}/state.npy\n")
with open("task2_rknn/data/eval_turn.txt","w") as f:
    for s in turn_fn: f.write(s+"\n")
print(f"npy frames={len(all_fn)}  calib={len(calib)}  turn={len(turn_fn)}")
