# RK3588 板推理程序

```
act-rknn/                Rust crate
  src/main.rs            主程序
  src/rknn_sys.rs        rknn_api FFI 绑定（最小子集）
  build.rs               链接 librknnrt.so
  .cargo/config.toml     交叉编译目标 + 链接器
rknn_runtime/            RKNN 运行时（由 fetch_runtime.sh 拉取）
  include/rknn_api.h
  aarch64/librknnrt.so
fetch_runtime.sh         拉取运行时库与头文件
```

## 在 x86 机器上交叉编译构建

```bash
# 1) 拉取 RKNN 运行时（头文件 + aarch64 .so）
./fetch_runtime.sh

# 2) 准备交叉工具链（自带 sysroot 的 aarch64 glibc gcc），并配好 .cargo/config.toml 的 linker 路径
#    本项目用 bootlin aarch64--glibc--stable；rustup target add aarch64-unknown-linux-gnu
cd act-rknn
cargo build --release --target aarch64-unknown-linux-gnu

# 产物：target/aarch64-unknown-linux-gnu/release/act-rknn
```

## 部署 & 运行（在 RK3588 上）

部署模型用 fp16。`--image` 给目录会批量跑目录下全部 .jpg。

```bash
# 把可执行文件、librknnrt.so、.rknn 模型、测试图放到同一目录
#   act-rknn  librknnrt.so  act_rk3588_fp16.rknn  frames/
./act-rknn --model act_rk3588_fp16.rknn --image frames --state 0 0
# 每帧输出：<frame>: left_vel=.. right_vel=.. diff=±.. decision=left/right (xx ms)
```

可执行文件 RPATH 设为 `$ORIGIN`，因此 `librknnrt.so` 放同目录即可被找到。
一键打包部署目录见 `../scripts/package_board.sh`（产出 `../board_pkg/linux/`）。

## Orange Pi 5 Plus / RK3588 / Ubuntu 22.04 实测

已在 NPU 上跑通（RKNPU driver v0.9.6 + librknnrt 2.3.2）。实测结论与 RKNN 模拟器相反：

| 模型 | 判向 vs gt | 右召回(gt右=19) | 时延 | 峰值内存 |
|---|---|---|---|---|
| fp16（部署） | 46/56 | 18/19 | 24.7 ms | 206 MB |
| hybrid（模拟器里最优） | 40/56 | 3/19 | 20 ms | ~200 MB |
| full-int8 | 31/56 | 18/19 | ~18 ms | ~180 MB |

模拟器判定无损最优的混合量化，在真硬件上反而最差——轮速差信号(~0.005)与 NPU 量化噪声(~0.006)同量级。故部署模型选用 fp16。

完整对比见 [`../results/board_rk3588_real.md`](../results/board_rk3588_real.md)。


## 关于 musl / StarryOS

接下来要做是将 starryOS 烧到板子上，再在上面进行推理。

官方 `librknnrt.so` 是 glibc(aarch64) 链接的，所以本程序当前目标是 `aarch64-unknown-linux-gnu`，对应 RK3588 上的普通 Linux。

要跑在 StarryOS 上，需解决 glibc 运行时库与 musl 的兼容——这与当时在 QEMU 上跑的时候动态链接 ORT 崩在 musl是同一类问题。
