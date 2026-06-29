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

// rknn_query_cmd（节选，零拷贝/属性查询用）
pub const RKNN_QUERY_IN_OUT_NUM: c_int = 0;
pub const RKNN_QUERY_INPUT_ATTR: c_int = 1;
pub const RKNN_QUERY_OUTPUT_ATTR: c_int = 2;
pub const RKNN_QUERY_NATIVE_INPUT_ATTR: c_int = 8;
pub const RKNN_QUERY_NATIVE_OUTPUT_ATTR: c_int = 9;

pub const RKNN_MAX_DIMS: usize = 16;
pub const RKNN_MAX_NAME_LEN: usize = 256;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct rknn_input_output_num {
    pub n_input: u32,
    pub n_output: u32,
}

// 与 rknn_api.h 的 _rknn_tensor_attr 严格对齐（字段顺序/类型一致，
// #[repr(C)] 负责按 C ABI 插入对齐填充）。供 rknn_query / rknn_set_io_mem 用。
#[repr(C)]
#[derive(Clone, Copy)]
pub struct rknn_tensor_attr {
    pub index: u32,
    pub n_dims: u32,
    pub dims: [u32; RKNN_MAX_DIMS],
    pub name: [u8; RKNN_MAX_NAME_LEN],
    pub n_elems: u32,
    pub size: u32,
    pub fmt: c_int,      // rknn_tensor_format
    pub type_: c_int,    // rknn_tensor_type
    pub qnt_type: c_int, // rknn_tensor_qnt_type
    pub fl: i8,
    pub zp: i32,
    pub scale: f32,
    pub w_stride: u32,
    pub size_with_stride: u32,
    pub pass_through: u8,
    pub h_stride: u32,
}

impl Default for rknn_tensor_attr {
    fn default() -> Self {
        // 全零初始化，等价于 C 侧 memset(attr, 0, sizeof(attr))。
        unsafe { std::mem::zeroed() }
    }
}

// 与 rknn_api.h 的 _rknn_tensor_memory 严格对齐。
#[repr(C)]
pub struct rknn_tensor_mem {
    pub virt_addr: *mut c_void,
    pub phys_addr: u64,
    pub fd: i32,
    pub offset: i32,
    pub size: u32,
    pub flags: u32,
    pub priv_data: *mut c_void,
}

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

    // ---- 零拷贝 / 属性查询 ----
    pub fn rknn_query(
        context: rknn_context,
        cmd: c_int, // rknn_query_cmd
        info: *mut c_void,
        size: u32,
    ) -> c_int;

    /// 在内部分配一块 NPU 可访问的 tensor 内存（DMA），返回其信息指针。
    pub fn rknn_create_mem(context: rknn_context, size: u32) -> *mut rknn_tensor_mem;

    /// 释放 rknn_create_mem 分配的内存。
    pub fn rknn_destroy_mem(context: rknn_context, mem: *mut rknn_tensor_mem) -> c_int;

    /// 把一块 tensor 内存按 attr 描述绑定为输入/输出。绑定后 rknn_run 直接读写该内存，
    /// 无需再 inputs_set / outputs_get（从根上避免每帧分配输出 DMA）。
    pub fn rknn_set_io_mem(
        context: rknn_context,
        mem: *mut rknn_tensor_mem,
        attr: *mut rknn_tensor_attr,
    ) -> c_int;
}
