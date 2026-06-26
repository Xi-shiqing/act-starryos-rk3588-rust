# RK3588 板上推理程序（Rust + RKNN FFI）

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

```bash
# 把可执行文件、librknnrt.so、.rknn 模型、测试图放到同一目录
#   act-rknn  librknnrt.so  act_rk3588_hybrid_backbone.rknn  frame_000227.jpg
./act-rknn --model act_rk3588_hybrid_backbone.rknn --image frame_000227.jpg --state 0 0
# 期望输出：first_step: ... diff=+0.00xx decision=right
```

可执行文件 RPATH 设为 `$ORIGIN`，因此 `librknnrt.so` 放同目录即可被找到。


## 关于 musl / StarryOS

x86 机器侧已验证

唯一只能上板验证的是 `rknn_run` 在 NPU 上的实际执行。


官方 `librknnrt.so` 是 glibc(aarch64) 链接的，所以本程序当前目标是 `aarch64-unknown-linux-gnu`，对应 RK3588 上的普通 Linux。

要跑在 StarryOS 上，需解决 glibc 运行时库与 musl 的兼容——这与当时在 QEMU 上跑的时候动态链接 ORT 崩在 musl是同一类问题。
