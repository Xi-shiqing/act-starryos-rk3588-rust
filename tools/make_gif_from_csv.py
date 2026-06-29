#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
由逐帧判向 csv（frame,left,right,diff,decision）+ data/frames 渲染演示 gif：
行驶画面 + 实时转向指针。csv 判向是纯符号 left/right；gif 里 |diff|<deadband 仅为视觉显示 STRAIGHT。

用法: python3 tools/make_gif_from_csv.py --csv <csv> --out <gif> --title "环境名" [--deadband 0.002] [--stride 1] [--width 320] [--fps 20]
"""
import argparse, csv, os
from PIL import Image, ImageDraw

ROOT = __file__.rsplit("/tools/", 1)[0]
FRAMES = f"{ROOT}/data/frames"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--deadband", type=float, default=0.002)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    gif = []
    W = args.width
    for idx, r in enumerate(rows):
        if idx % args.stride:
            continue
        fn = r["frame"]; l = float(r["left"]); rr = float(r["right"]); diff = float(r["diff"])
        dec = r["decision"]
        show = "straight" if abs(diff) < args.deadband else dec       # gif 视觉死区
        im = Image.open(f"{FRAMES}/{fn}").convert("RGB")
        H = int(im.height * W / im.width)
        im = im.resize((W, H))
        panel = 64
        canvas = Image.new("RGB", (W, H + panel), (18, 18, 22))
        canvas.paste(im, (0, 0))
        d = ImageDraw.Draw(canvas)
        cx, midy = W // 2, H + panel // 2
        # 顶部环境横幅
        if args.title:
            d.rectangle([(0, 0), (W, 16)], fill=(12, 12, 16))
            d.text((6, 3), args.title, fill=(120, 210, 150))
        d.line([(cx, H + 8), (cx, H + panel - 8)], fill=(90, 90, 100), width=1)
        if show == "straight":
            col = (230, 185, 80); arrow = "STRAIGHT"
            d.ellipse([(cx-5, midy-5), (cx+5, midy+5)], fill=col)
        else:
            col = (90, 200, 120)
            mag = max(-1.0, min(1.0, diff / 0.02))
            nx = int(cx + mag * (W // 2 - 16))
            d.line([(cx, midy), (nx, midy)], fill=col, width=6)
            tri = [(nx, midy)] + ([(nx-14, midy-9), (nx-14, midy+9)] if mag >= 0 else [(nx+14, midy-9), (nx+14, midy+9)])
            d.polygon(tri, fill=col)
            arrow = "RIGHT >" if show == "right" else "< LEFT"
        d.text((8, H + 6), f"frame {idx+1:3d}/{len(rows)}", fill=(200, 200, 210))
        d.text((8, H + 26), f"L={l:+.4f}  R={rr:+.4f}  diff={diff:+.4f}", fill=(170, 170, 180))
        d.text((W - 96, H + 16), arrow, fill=col)
        gif.append(canvas)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    gif[0].save(args.out, save_all=True, append_images=gif[1:],
                duration=int(1000/args.fps), loop=0, optimize=True)
    sz = os.path.getsize(args.out) / 1e6
    print(f"gif -> {args.out}  ({len(gif)} 帧, {W}px, {args.fps}fps, {sz:.1f}MB)")

if __name__ == "__main__":
    main()
