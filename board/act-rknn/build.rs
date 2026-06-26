// 链接 RKNN 运行时库 librknnrt.so（aarch64）。
// 运行时库放在仓库的 board/rknn_runtime/aarch64/ 下。
use std::path::PathBuf;

fn main() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // board/act-rknn -> board/rknn_runtime/aarch64
    let lib_dir = manifest.parent().unwrap().join("rknn_runtime/aarch64");
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=rknnrt");
    println!("cargo:rerun-if-changed=build.rs");
}
