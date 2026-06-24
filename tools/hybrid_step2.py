#!/usr/bin/env python3
# 混合量化 step2：把指定子串命中的层在 cfg 里标成 float16，其余保持 int8，重建并评测。
# 用法: python3 hybrid_step2.py "<substr1,substr2,...>" <tag>
import csv, os, re, sys, numpy as np
import _onnx_compat  # noqa
from rknn.api import RKNN

KEEP = [s for s in (sys.argv[1] if len(sys.argv)>1 else "decoder,action").split(",") if s]
TAG  = sys.argv[2] if len(sys.argv)>2 else "hybrid"
CFG="act_rknn_4d.quantization.cfg"; MODELF="act_rknn_4d.model"; DATAF="act_rknn_4d.data"
NPY="task2_rknn/data/npy"; REF="task2_rknn/results/ref_fp32.csv"
RKNN_OUT=f"task2_rknn/model/act_rk3588_{TAG}.rknn"; OUTCSV=f"task2_rknn/results/sim_{TAG}.csv"
aq01=np.array([-0.1,0.0,0.0],np.float32); aq99=np.array([0.2,0.2,0.0],np.float32)

# ---- 改写 cfg：custom_quantize_layers 里把命中层设 float16 ----
lines=open(CFG).read().split("\n")
names=[]; inqp=False
for l in lines:
    if l.startswith("quantize_parameters:"): inqp=True; continue
    if inqp and re.match(r"^    [^\s].*:$", l): names.append(l.strip()[:-1])
ACT_SUFFIX=("-rs","_rs","_mm","_sw","_tp","_sdpa","ln","_mm_sdpa","output_0_mm")
def _valid(n):
    if n.startswith("m.") or "onnx::" in n: return False
    if n.endswith(".weight") or n.endswith(".bias"): return False
    if n.endswith("_int8") or n in ("image","state"): return False
    if "Expand_output" in n or "Unsqueeze_output" in n: return False
    return n=="action" or any(suf in n for suf in ACT_SUFFIX)
sel=[n for n in names if any(k in n for k in KEEP) and _valid(n)]
# 不把输入本身设 float（image/state 保持），但输出 action 要 float
cfg_txt=open(CFG).read()
block="custom_quantize_layers: {}"
newblock="custom_quantize_layers:\n" + "".join(f"    {n}: float16\n" for n in sel)
cfg_txt=cfg_txt.replace(block,newblock,1)
open(CFG,"w").write(cfg_txt)
print(f"[cfg] set float16 on {len(sel)} layers (keys={KEEP})")

rknn=RKNN(verbose=False)
r=rknn.hybrid_quantization_step2(model_input=MODELF, data_input=DATAF, model_quantization_cfg=CFG)
assert r==0, f"step2 rc={r}"
assert rknn.export_rknn(RKNN_OUT)==0
sz=os.path.getsize(RKNN_OUT)/1e6
assert rknn.init_runtime()==0

ref={x['frame']:x for x in csv.DictReader(open(REF))}
turn=[l.strip() for l in open("task2_rknn/data/eval_turn.txt")]
state=np.load(f"{NPY}/state.npy")
rows=[]; acc=0; agree=0
for stem in turn:
    img=np.transpose(np.load(f"{NPY}/{stem}_img.npy"),(0,2,3,1))
    out=rknn.inference(inputs=[img,state],data_format=['nhwc','nchw'])[0].reshape(-1)
    l=(out[0]+1)*0.5*(aq99[0]-aq01[0])+aq01[0]; rr=(out[1]+1)*0.5*(aq99[1]-aq01[1])+aq01[1]
    diff=l-rr; pred='right' if diff>0 else 'left'
    fn=stem+".jpg"; gt=ref[fn]['gt']; rp=ref[fn]['pred']
    acc+=(pred==gt); agree+=(pred==rp); rows.append((fn,f"{diff:.6f}",pred,gt,rp))
n=len(turn)
with open(OUTCSV,'w',newline='') as f:
    w=csv.writer(f); w.writerow(['frame','diff','pred','gt','ref_fp32']); w.writerows(rows)
print(f"RESULT tag={TAG} size={sz:.1f}MB acc_vs_gt={acc}/{n}={acc/n:.3f} agree_vs_fp32={agree}/{n}={agree/n:.3f}")
rknn.release()
