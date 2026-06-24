#!/usr/bin/env python3
# INT8 量化扫描：对比 量化粒度(layer/channel) × 算法(normal/kl/mmse) 的判向精度。
# 用法: python3 quant_sweep.py <method> <algo> <calib_txt> [tag]
import csv, os, sys, numpy as np
import _onnx_compat  # noqa: 必须在 rknn 之前
from rknn.api import RKNN

METHOD = sys.argv[1] if len(sys.argv)>1 else "channel"
ALGO   = sys.argv[2] if len(sys.argv)>2 else "normal"
CALIB  = sys.argv[3] if len(sys.argv)>3 else "task2_rknn/data/dataset_calib.txt"
TAG    = sys.argv[4] if len(sys.argv)>4 else f"{METHOD}_{ALGO}"
ONNX="task2_rknn/model/act_rknn_4d.onnx"; NPY="task2_rknn/data/npy"
REF="task2_rknn/results/ref_fp32.csv"
RKNN_OUT=f"task2_rknn/model/act_rk3588_int8_{TAG}.rknn"
OUTCSV=f"task2_rknn/results/sim_int8_{TAG}.csv"
aq01=np.array([-0.1,0.0,0.0],np.float32); aq99=np.array([0.2,0.2,0.0],np.float32)

def denorm(a):
    return ((a[0]+1)*0.5*(aq99[0]-aq01[0])+aq01[0],
            (a[1]+1)*0.5*(aq99[1]-aq01[1])+aq01[1])

rknn=RKNN(verbose=False)
rknn.config(mean_values=[[0,0,0],[0,0]], std_values=[[1,1,1],[1,1]],
            target_platform='rk3588',
            quantized_method=METHOD, quantized_algorithm=ALGO)
assert rknn.load_onnx(model=ONNX, inputs=['image','state'],
                      input_size_list=[[1,3,224,224],[1,2]])==0
assert rknn.build(do_quantization=True, dataset=CALIB)==0
assert rknn.export_rknn(RKNN_OUT)==0
sz=os.path.getsize(RKNN_OUT)/1e6
assert rknn.init_runtime()==0

ref={r['frame']:r for r in csv.DictReader(open(REF))}
turn=[l.strip() for l in open("task2_rknn/data/eval_turn.txt")]
state=np.load(f"{NPY}/state.npy")
rows=[]; acc=0; agree=0
for stem in turn:
    img=np.transpose(np.load(f"{NPY}/{stem}_img.npy"),(0,2,3,1))
    out=rknn.inference(inputs=[img,state],data_format=['nhwc','nchw'])[0].reshape(-1)
    l,r=denorm(out); diff=l-r; pred='right' if diff>0 else 'left'
    fn=stem+".jpg"; gt=ref[fn]['gt']; rp=ref[fn]['pred']
    acc+=(pred==gt); agree+=(pred==rp)
    rows.append((fn,f"{diff:.6f}",pred,gt,rp))
n=len(turn)
with open(OUTCSV,'w',newline='') as f:
    w=csv.writer(f); w.writerow(['frame','diff','pred','gt','ref_fp32']); w.writerows(rows)
print(f"RESULT tag={TAG} size={sz:.1f}MB acc_vs_gt={acc}/{n}={acc/n:.3f} agree_vs_fp32={agree}/{n}={agree/n:.3f}")
rknn.release()
