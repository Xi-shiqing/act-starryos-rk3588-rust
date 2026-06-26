# 拉取 RK3588 的 RKNN 运行时库与头文件。
# 来自官方 airockchip/rknn-toolkit2 的 rknpu2 运行时。
set -e
cd "$(dirname "$0")"
BASE="https://github.com/airockchip/rknn-toolkit2/raw/master/rknpu2/runtime/Linux/librknn_api"
mkdir -p rknn_runtime/include rknn_runtime/aarch64
curl -sL -o rknn_runtime/include/rknn_api.h        "$BASE/include/rknn_api.h"
curl -sL -o rknn_runtime/aarch64/librknnrt.so      "$BASE/aarch64/librknnrt.so"
echo "fetched rknn_api.h + librknnrt.so(aarch64)"
