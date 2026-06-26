// 手写的 RKNN C API FFI 绑定。
// 对应 board/rknn_runtime/include/rknn_api.h。aarch64 下 rknn_context = u64。

#![allow(non_camel_case_types, dead_code)]
use std::os::raw::{c_int, c_void};

pub type rknn_context = u64;

// rknn_tensor_type（节选）
pub const RKNN_TENSOR_FLOAT32: c_int = 0;
pub const RKNN_TENSOR_FLOAT16: c_int = 1;
pub const RKNN_TENSOR_INT8: c_int = 2;
pub const RKNN_TENSOR_UINT8: c_int = 3;

// rknn_tensor_format
pub const RKNN_TENSOR_NCHW: c_int = 0;
pub const RKNN_TENSOR_NHWC: c_int = 1;

#[repr(C)]
pub struct rknn_input {
    pub index: u32,
    pub buf: *mut c_void,
    pub size: u32,
    pub pass_through: u8,
    pub type_: c_int, // rknn_tensor_type
    pub fmt: c_int,   // rknn_tensor_format
}

#[repr(C)]
pub struct rknn_output {
    pub want_float: u8,
    pub is_prealloc: u8,
    pub index: u32,
    pub buf: *mut c_void,
    pub size: u32,
}

extern "C" {
    pub fn rknn_init(
        context: *mut rknn_context,
        model: *mut c_void,
        size: u32,
        flag: u32,
        extend: *mut c_void, // rknn_init_extend*，此处传 NULL
    ) -> c_int;

    pub fn rknn_inputs_set(context: rknn_context, n_inputs: u32, inputs: *mut rknn_input) -> c_int;

    pub fn rknn_run(context: rknn_context, extend: *mut c_void) -> c_int;

    pub fn rknn_outputs_get(
        context: rknn_context,
        n_outputs: u32,
        outputs: *mut rknn_output,
        extend: *mut c_void,
    ) -> c_int;

    pub fn rknn_outputs_release(
        context: rknn_context,
        n_outputs: u32,
        outputs: *mut rknn_output,
    ) -> c_int;

    pub fn rknn_destroy(context: rknn_context) -> c_int;
}
