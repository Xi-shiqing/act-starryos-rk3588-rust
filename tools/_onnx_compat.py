# 兼容垫片：onnx>=1.16 移除了 onnx.mapping，rknn-toolkit2 2.3.2 仍依赖它。
# 用 onnx.helper 重建 TENSOR_TYPE_TO_NP_TYPE / NP_TYPE_TO_TENSOR_TYPE 并注册回 onnx.mapping。
import sys, types, numpy as np, onnx
from onnx import TensorProto, helper
if not hasattr(onnx, "mapping"):
    m = types.ModuleType("onnx.mapping")
    t2n = {}
    for _name, _val in TensorProto.DataType.items():
        if _val == 0:
            continue
        try:
            t2n[_val] = np.dtype(helper.tensor_dtype_to_np_dtype(_val))
        except Exception:
            pass
    m.TENSOR_TYPE_TO_NP_TYPE = t2n
    m.NP_TYPE_TO_TENSOR_TYPE = {v: k for k, v in t2n.items()}
    # 少数旧代码还会用到的别名
    m.TENSOR_TYPE_TO_STORAGE_TENSOR_TYPE = {k: k for k in t2n}
    sys.modules["onnx.mapping"] = m
    onnx.mapping = m

# onnx.helper.strip_doc_string 在新版 protobuf(FieldDescriptor.label -> _label)下崩溃。
# 文档字符串与转换结果无关，替换成不依赖 descriptor.label 的安全版（直接清 doc_string，失败则跳过）。
def _strip_doc_string_safe(proto):
    try:
        if proto.HasField("doc_string"):
            proto.ClearField("doc_string")
    except Exception:
        pass
    g = getattr(proto, "graph", None)
    if g is not None:
        for n in g.node:
            try: n.ClearField("doc_string")
            except Exception: pass
        try: g.ClearField("doc_string")
        except Exception: pass
    return proto
helper.strip_doc_string = _strip_doc_string_safe
