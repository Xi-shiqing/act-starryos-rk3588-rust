#!/bin/sh
# 在 StarryOS 串口里手动跑：批量推理 frames/ 下全部帧。
cd /act_rknn
export LD_LIBRARY_PATH=/act_rknn/lib:/usr/local/lib:/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}
./act-rknn --model model/act_rk3588_fp16.rknn --image frames --state 0 0
