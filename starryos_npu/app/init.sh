#!/bin/sh
# 在 StarryOS 串口里手动跑：批量推理 frames/ 下全部帧。
cd /act_rknn
export LD_LIBRARY_PATH=/act_rknn/lib:/usr/local/lib:/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}
echo "==== ACT NPU inference start ===="
echo "io-mode default: zc-float (RKNN zero-copy, no outputs_get/release)"
echo "image source: frames_rgb224.bin (pre-resized RGB224 pack, no JPEG decode on board)"
echo "---- smoke: first 120 frames, must pass frame_000096 ----"
ACT_TRACE=1 ./act-rknn --model model/act_rk3588_fp16.rknn --image frames --rgb-pack frames_rgb224.bin --state 0 0 --loop open --io-mode zc-float --count 120
echo "---- full open-loop (666 frames, state=0,0) ----"
./act-rknn --model model/act_rk3588_fp16.rknn --image frames --rgb-pack frames_rgb224.bin --state 0 0 --loop open --io-mode zc-float
echo "---- full closed-loop (666 frames, feedback) ----"
./act-rknn --model model/act_rk3588_fp16.rknn --image frames --rgb-pack frames_rgb224.bin --state 0 0 --loop closed --io-mode zc-float
echo "==== ACT NPU inference end (exit=$?) ===="
echo "manual JPEG repro command if needed: ACT_TRACE=1 ./act-rknn --model model/act_rk3588_fp16.rknn --image frames --state 0 0 --loop open --io-mode zc-float --count 120"
