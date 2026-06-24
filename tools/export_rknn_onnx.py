# 导出 RKNN 友好的 ONNX
import sys, torch
from pathlib import Path
PROJ = Path("/root/OScompetition/repos/proj57")
sys.path.insert(0, str(PROJ))
from act.configuration_act import ACTConfig
from act.modeling_act import ACTModel

CKPT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/model.pt")
OUT  = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/root/OScompetition/task2_rknn/model/act_rknn_4d.onnx")

class Wrap4D(torch.nn.Module):
    """接收 4D 图像，内部补出相机维，调用 ACT。"""
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, image, state):          # image: [1,3,224,224]
        image5d = image.unsqueeze(1)          # -> [1,1,3,224,224]
        return self.m(image5d, state, action_target=None, infer_cvae=True)["action"]

def main():
    ckpt = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    cfg = ACTConfig(**ckpt.get("config", {}))
    model = ACTModel(cfg); model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    wrap = Wrap4D(model).eval()
    img = torch.zeros(1, 3, 224, 224); st = torch.zeros(1, 2)
    with torch.no_grad():
        out = wrap(img, st)
    print("dummy out:", tuple(out.shape))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrap, (img, st), str(OUT),
        input_names=["image", "state"], output_names=["action"],
        opset_version=12,                     # RKNN 对 opset 12 支持最稳
        do_constant_folding=True, dynamo=False,
    )
    print("exported ->", OUT)

if __name__ == "__main__":
    main()
