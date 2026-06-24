#!/usr/bin/env python3
# ONNX -> RKNN(rk3588) 转换 + 模拟器推理评测。
# 用法: python3 convert_and_sim.py [fp16|int8]
# - 预处理已在 npy 里做好，故 rknn.config 不再做 mean/std（mean=0,std=1）。
# - 输出 .rknn，并在 56 转向帧上跑模拟器，对比 gt 与 fp32 基线(ref_fp32.csv)。
import csv, os, sys, numpy as np
import _onnx_compat  # 必须在 rknn 之前
from rknn.api import RKNN

MODE = sys.argv[1] if len(sys.argv)>1 else "fp16"
ONNX = "task2_rknn/model/act_rknn_4d.onnx"
NPY  = "task2_rknn/data/npy"
RKNN_OUT = f"task2_rknn/model/act_rk3588_{MODE}.rknn"
REF  = "task2_rknn/results/ref_fp32.csv"
OUTCSV = f"task2_rknn/results/sim_{MODE}.csv"
aq01=np.array([-0.1,0.0,0.0],np.float32); aq99=np.array([0.2,0.2,0.0],np.float32)

def build():
    rknn=RKNN(verbose=False)
    # 两路输入都已是归一化 fp32，故不在 NPU 内做均值/方差。
    cfg=dict(mean_values=[[0,0,0],[0,0]], std_values=[[1,1,1],[1,1]],
             target_platform='rk3588')
    if MODE=="int8":
        # 逐通道量化（channel）比逐层精度好；algorithm 默认 normal。
        # 注：quantized_algorithm='mmse' 精度更高但在 CPU 模拟器上极慢（>30min），
        # 留待板上/更快环境再开。
        cfg.update(quantized_method='channel')
    rknn.config(**cfg)
    print("[load_onnx]")
    assert rknn.load_onnx(model=ONNX, inputs=['image','state'],
                          input_size_list=[[1,3,224,224],[1,2]])==0
    quant = (MODE=="int8")
    print(f"[build] do_quantization={quant}")
    assert rknn.build(do_quantization=quant,
                      dataset="task2_rknn/data/dataset_calib.txt" if quant else None)==0
    os.makedirs(os.path.dirname(RKNN_OUT),exist_ok=True)
    assert rknn.export_rknn(RKNN_OUT)==0
    print("[export]", RKNN_OUT, round(os.path.getsize(RKNN_OUT)/1e6,1),"MB")
    assert rknn.init_runtime()==0   # target=None -> 模拟器
    return rknn

def denorm(a):
    l=(a[0]+1)*0.5*(aq99[0]-aq01[0])+aq01[0]
    r=(a[1]+1)*0.5*(aq99[1]-aq01[1])+aq01[1]
    return l,r

def main():
    rknn=build()
    ref={row['frame']:row for row in csv.DictReader(open(REF))} if os.path.exists(REF) else {}
    turn=[l.strip() for l in open("task2_rknn/data/eval_turn.txt")]
    state=np.load(f"{NPY}/state.npy")
    rows=[]; acc_gt=0; agree_ref=0
    for stem in turn:
        img=np.load(f"{NPY}/{stem}_img.npy")               # NCHW [1,3,224,224]
        img_nhwc=np.transpose(img,(0,2,3,1))               # RKNN 图像输入要 NHWC [1,224,224,3]
        out=rknn.inference(inputs=[img_nhwc,state],data_format=['nhwc','nchw'])[0].reshape(-1)
        l,r=denorm(out); diff=l-r; pred='right' if diff>0 else 'left'
        fn=stem+".jpg"; gt=ref.get(fn,{}).get('gt'); refpred=ref.get(fn,{}).get('pred')
        acc_gt+=(pred==gt); agree_ref+=(pred==refpred)
        rows.append((fn,f"{l:.6f}",f"{r:.6f}",f"{diff:.6f}",pred,gt,refpred))
    n=len(turn)
    with open(OUTCSV,'w',newline='') as f:
        w=csv.writer(f); w.writerow(['frame','left','right','diff','pred','gt','ref_fp32']); w.writerows(rows)
    print(f"[{MODE}] frames={n} dir_acc_vs_gt={acc_gt}/{n}={acc_gt/n:.3f}  agree_vs_fp32={agree_ref}/{n}={agree_ref/n:.3f} -> {OUTCSV}")
    rknn.release()

if __name__=="__main__": main()
