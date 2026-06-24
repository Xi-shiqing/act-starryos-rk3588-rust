#!/usr/bin/env python3
# 用 ONNX Runtime 跑 4D fp32 模型，在 eval_manifest 的 56 个明确转向帧上建立基线。
# 预处理与官方 infer.ipynb 一致；state=(0,0)。判向=left-right 符号（正=右,负=左）。
# 输出每帧 left/right/diff/pred 到 csv，供 RKNN 模拟器结果对比。
import csv, sys, numpy as np, onnxruntime as ort
from PIL import Image
from torchvision import transforms

MANIFEST = "repos/act-starryos-qemu-infer/deploy/cpp_onnxruntime/data/eval_manifest.csv"
FRAMES   = "task2_rknn/data/frames"
MODEL    = sys.argv[1] if len(sys.argv) > 1 else "task2_rknn/model/act_rknn_4d.onnx"
OUT      = sys.argv[2] if len(sys.argv) > 2 else "task2_rknn/results/ref_fp32.csv"

q01=np.array([-0.079,0.0],np.float32); q99=np.array([0.2,0.2],np.float32)
aq01=np.array([-0.1,0.0,0.0],np.float32); aq99=np.array([0.2,0.2,0.0],np.float32)
TF=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])

def state_norm(s):
    d=np.where(q99-q01==0,1e-8,q99-q01); return (2*(s-q01)/d-1).astype(np.float32).reshape(1,2)

def main():
    rows=[r for r in csv.DictReader(open(MANIFEST))
          if float(r['gt_left_vel'])!=float(r['gt_right_vel'])]
    s=ort.InferenceSession(MODEL,providers=["CPUExecutionProvider"])
    in_img=[i.name for i in s.get_inputs() if len(i.shape)>=4][0]
    in_st =[i.name for i in s.get_inputs() if len(i.shape)==2][0]
    st=state_norm(np.zeros(2,np.float32))
    out=[]; correct=0
    for r in rows:
        fn=r['image_path'].split('/')[-1]
        img=TF(Image.open(f"{FRAMES}/{fn}").convert("RGB")).unsqueeze(0).numpy().astype(np.float32)
        a=s.run(None,{in_img:img,in_st:st})[0].reshape(-1)
        left=(a[0]+1)*0.5*(aq99[0]-aq01[0])+aq01[0]
        right=(a[1]+1)*0.5*(aq99[1]-aq01[1])+aq01[1]
        diff=left-right; pred='right' if diff>0 else 'left'
        gt='right' if float(r['gt_left_vel'])>float(r['gt_right_vel']) else 'left'
        correct+=(pred==gt)
        out.append((fn,f"{left:.6f}",f"{right:.6f}",f"{diff:.6f}",pred,gt))
    import os; os.makedirs(os.path.dirname(OUT),exist_ok=True)
    with open(OUT,'w',newline='') as f:
        w=csv.writer(f); w.writerow(['frame','left','right','diff','pred','gt']); w.writerows(out)
    print(f"frames={len(rows)} dir_acc_vs_gt={correct}/{len(rows)}={correct/len(rows):.3f} -> {OUT}")

if __name__=="__main__": main()
