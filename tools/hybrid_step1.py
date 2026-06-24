#!/usr/bin/env python3
import sys
import _onnx_compat  # noqa
from rknn.api import RKNN
ONNX="task2_rknn/model/act_rknn_4d.onnx"
CALIB="task2_rknn/data/dataset_calib.txt"
rknn=RKNN(verbose=False)
rknn.config(mean_values=[[0,0,0],[0,0]], std_values=[[1,1,1],[1,1]],
            target_platform='rk3588', quantized_method='channel')
assert rknn.load_onnx(model=ONNX, inputs=['image','state'],
                      input_size_list=[[1,3,224,224],[1,2]])==0
r=rknn.hybrid_quantization_step1(dataset=CALIB, proposal=False)
print("step1 rc=", r)
rknn.release()
