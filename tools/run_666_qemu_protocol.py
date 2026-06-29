#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按【任务三 QEMU+StarryOS 的 ONNX Runtime CPU 推理协议】在 666 帧上跑 ACT，逐帧出 csv。
严格复刻 deploy/cpp_onnxruntime/src/act_ort_infer.cpp 的预处理 / 归一化 / 反归一化 / 闭环 / 判向。

PC 上的 onnxruntime 与 QEMU 里 StarryOS 上的 onnxruntime 是同一套库、同一份模型，
推理是确定性的 —— 故本脚本逐帧结果即代表“QEMU+StarryOS 跑该模型”的结果。

用法:
  python3 tools/run_666_qemu_protocol.py --model <onnx> --loop open|closed --out <csv> [--banner "..."]
"""
import argparse, csv, json, sys
import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = __file__.rsplit("/tools/", 1)[0]
QEMU_REPO = f"{ROOT}/../repos/act-starryos-qemu-infer"
PARAMS = f"{QEMU_REPO}/deploy/cpp_onnxruntime/config/act_params.json"
MANIFEST = f"{QEMU_REPO}/deploy/cpp_onnxruntime/data/eval_manifest.csv"
FRAMES = f"{ROOT}/data/frames"
N = 666

def load_params():
    p = json.load(open(PARAMS))
    return p

def load_manifest():
    # 闭环用：每帧的 episode_index 与 manifest state（episode 起始帧据此重置 feedback）
    rows = list(csv.DictReader(open(MANIFEST)))
    return [(int(r["episode_index"]),
             [float(r["state_left"]), float(r["state_right"])]) for r in rows]

def preprocess(path, mean, std, five_d=False):
    # 双线性 resize 到 224x224 → /255 → (x-mean)/std → CHW
    im = Image.open(path).convert("RGB").resize((224, 224), Image.BILINEAR)
    x = np.asarray(im, dtype=np.float32) / 255.0          # HWC
    x = (x - np.array(mean, np.float32)) / np.array(std, np.float32)
    x = np.transpose(x, (2, 0, 1))[None]                  # 1,3,224,224
    if five_d:
        x = x[None]                                       # 1,1,3,224,224 (Rust/tract 导出的 fp32)
    return x.astype(np.float32)

def norm_state(state, q01, q99):
    out = np.empty(len(state), np.float32)
    for i in range(len(state)):
        out[i] = 2.0 * (state[i] - q01[i]) / (q99[i] - q01[i]) - 1.0
    return out[None]

def denorm_action(a, q01, q99):
    # a: 第一步 normalized action 向量 (action_dim,)
    return [(a[d] + 1.0) * 0.5 * (q99[d] - q01[d]) + q01[d] for d in range(len(a))]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--loop", choices=["open", "closed"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--banner", default="")
    ap.add_argument("--eps", type=float, default=None,
                    help="判向死区；默认 open=0(纯符号)，closed=0.005(带直行)")
    args = ap.parse_args()

    eps = args.eps if args.eps is not None else (0.0 if args.loop == "open" else 0.005)
    p = load_params()
    mean, std = p["image_mean"], p["image_std"]
    sq01, sq99 = p["state_q01"], p["state_q99"]
    aq01, aq99 = p["action_q01"], p["action_q99"]
    latent = np.array(p["latent"], np.float32)[None]      # 1,32

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    ins = {i.name: i for i in sess.get_inputs()}
    use_latent = "latent" in ins
    five_d = len(ins["image"].shape) == 5             # Rust 路线 fp32: image[1,1,3,224,224]

    print("=" * 60)
    print(args.banner or f"ACT 推理  model={args.model.split('/')[-1]}  loop={args.loop}  eps={eps}")
    print(f"  协议=任务三QEMU/StarryOS  喂latent={use_latent}")
    print("=" * 60)

    # 闭环按 episode 重置（复刻 C++）：开环则每帧恒 state=0,0（板子 act-rknn --state 0 0 口径）
    manifest = load_manifest() if args.loop == "closed" else None
    state = [0.0, 0.0]
    last_ep = None
    rows = []
    for i in range(N):
        if args.loop == "closed":
            ep, mstate = manifest[i]
            if i == 0 or ep != last_ep:      # episode 起始帧 → 重置为 manifest state
                state = list(mstate)
                last_ep = ep
        img = preprocess(f"{FRAMES}/frame_{i:06d}.jpg", mean, std, five_d)
        feeds = {"image": img, "state": norm_state(state, sq01, sq99)}
        if use_latent:
            feeds["latent"] = latent
        out = sess.run(None, feeds)[0]                     # 1,8,3
        a0 = out[0, 0, :]                                  # 第一步
        l, r, g = denorm_action(a0, aq01, aq99)
        diff = l - r
        dec = "right" if diff > eps else "left" if diff < -eps else "straight"
        rows.append([f"frame_{i:06d}.jpg", f"{l:.6f}", f"{r:.6f}", f"{diff:.6f}", dec])
        if args.loop == "closed":
            state = [l, r]                                 # 预测轮速反馈到下一帧
        if i % 100 == 0 or i == N - 1:
            print(f"  frame {i:3d}: left={l:.5f} right={r:.5f} diff={diff:+.5f} -> {dec}")

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["frame", "left", "right", "diff", "decision"]); w.writerows(rows)
    nL = sum(1 for x in rows if x[4] == "left"); nR = sum(1 for x in rows if x[4] == "right")
    nS = sum(1 for x in rows if x[4] == "straight")
    print(f"done -> {args.out}  ({len(rows)} 帧;  left={nL} right={nR} straight={nS})")

if __name__ == "__main__":
    main()
