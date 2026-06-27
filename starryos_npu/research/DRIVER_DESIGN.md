# StarryOS 上的 RKNPU 驱动 —— 设计与可行性

目标：让 ACT 推理在 **StarryOS（RK3588）的真 NPU** 上跑起来。本文是写驱动前的接口圈定、
StarryOS 能力盘点、差距分析与实现计划。

## 0. 核心认知（为什么可做）

rknpu 内核驱动是**"笨"的内存+寄存器管道**：NPU 的"智能"（regcmd 指令编码）在**用户态闭源 `librknnrt.so`** 里，
它把寄存器命令拼好后通过 ioctl 把"内存地址 + 提交"交给内核；内核只负责**分配 DMA 内存、映射寄存器、踢硬件、等完成**。

→ 我们**不逆向 NPU**，只需在 StarryOS 上**重实现这套开源的 ioctl 接口**。librknnrt 不改。
问题从"无界逆向"收敛为"有界的 ioctl 重实现"。

## 1. 要实现的内核接口（来自 rockchip-linux/kernel develop-5.10，与板子内核一致）

驱动以 **DRM render 节点**（`/dev/dri/renderD12X`）暴露，对外是 DRM 私有 ioctl，magic `'r'`：

| ioctl | 结构体 | 作用 |
|---|---|---|
| `RKNPU_ACTION` | `rknpu_action{flags,value}` | 查 HW/驱动版本、查/设频率电压、复位等（动作枚举 0..12+） |
| `RKNPU_MEM_CREATE` | `rknpu_mem_create{handle,flags,size,obj_addr,dma_addr,iommu_domain_id,...}` | 分配 NPU 可见的 DMA 内存，返回 handle + dma_addr |
| `RKNPU_MEM_MAP` | `rknpu_mem_map{handle,offset}` | 把该内存 mmap 进用户态 |
| `RKNPU_MEM_DESTROY` | handle | 释放 |
| `RKNPU_MEM_SYNC` | flags(TO/FROM_DEVICE) | cache 一致性刷写 |
| `RKNPU_SUBMIT` | `rknpu_submit{task_obj_addr,task_base_addr,core_mask,subcore_task[5],timeout,...}` | 提交一次推理：regcmd 缓冲地址→NPU，踢，等完成 |

内存 flag 含 `RKNPU_MEM_CONTIGUOUS`（连续，可绕 IOMMU）/ `RKNPU_MEM_IOMMU`（走 IOMMU）。
**优先用连续(CMA)内存绕开 IOMMU**，降低首版复杂度。

接口头：`rknpu_ioctl.h`（已存本目录）。

## 2. StarryOS 已具备的能力（够得着的部分）

| 需求 | StarryOS 现状 | 参考 |
|---|---|---|
| 自定义 ioctl 的字符设备 | ✅ `DeviceOps::ioctl(cmd,arg)` | `kernel/src/pseudofs/dev/event.rs`、`tty` |
| 设备 mmap 物理内存 | ✅ `DeviceMmap::Physical(PhysAddrRange)` | **`fb.rs`（framebuffer，MMIO 映射+ioctl，最佳模板）** |
| ioctl 系统调用分发 | ✅ `sys_ioctl → device.ioctl` | `kernel/src/syscall/fs/ctl.rs` |
| /dev 节点注册 | ✅ devfs `SimpleFs` builder | `kernel/src/pseudofs/dev/mod.rs` |
| 物理地址类型 | ✅ `memory_addr::PhysAddr/PhysAddrRange` | — |

→ **设备节点 + ioctl 分发 + mmap 这一层照 `fb.rs` 就能搭。**

## 3. 差距（要补的硬骨头，按风险排序）

1. **连续 DMA 内存分配**：StarryOS 无 CMA/DMA 分配器。NPU 的 regcmd/权重/输入输出缓冲需要物理连续
   （或 IOMMU 映射）的内存。需在内核加一个连续物理内存分配器（预留一段物理内存做池）。
2. **NPU 寄存器编程（SUBMIT 核心）**：照开源驱动 `rknpu_job.c` 的提交序列——写 base 寄存器、
   设 core_mask、踢 PC_OP、轮询/中断等完成。寄存器偏移与序列开源可查。
3. **NPU 上电/时钟/复位**：RK3588 NPU 有独立电源域 + 时钟 + reset。StarryOS 平台初始化里**大概率没拉起 NPU**。
   需在 StarryOS 的 RK3588 平台层使能 NPU 电源域/时钟（device tree 里 npu 节点的 power-domain/clocks）。
   **这是最大未知数——若 StarryOS 平台层没有 PMU/CRU 驱动，要先补。**
4. **IOMMU（可选规避）**：首版用连续内存绕开；若必须走 IOMMU，则要实现 RK IOMMU 页表。
5. **用户态 glibc ABI**：librknnrt 依赖 glibc(仅 GLIBC_2.17) + libstdc++ 等。StarryOS 是 musl。
   方案：rootfs 里塞 aarch64 glibc 运行时 + glibc ld.so，靠 StarryOS 的 Linux syscall ABI 跑（见 `USERSPACE_ABI.md` 待写）。

## 4. 实现顺序（每步可独立验证）

- **S0 用户态 ABI 验证**：先让一个最小 glibc 动态二进制在 StarryOS 上跑起来（不碰 NPU），证明 librknnrt 能被加载。
- **S1 设备骨架**：照 `fb.rs` 在 devfs 加 `/dev/dri/renderD12X`（或 rknpu 节点），ioctl 全部返回桩值；
  让 librknnrt `rknn_init` 走到"查版本"那步不崩。
- **S2 内存管理**：实现连续物理内存池 + MEM_CREATE/MAP/DESTROY/SYNC；librknnrt 能分配/映射缓冲。
- **S3 上电+寄存器**：拉起 NPU 电源/时钟，映射寄存器块，实现 SUBMIT 的寄存器序列 + 完成等待。
- **S4 端到端**：跑通一帧推理，串口打印判向，与 Ubuntu/NPU 的 fp16 结果对照。

## 5. 关键依赖与风险

- **必须有 StarryOS 跑在开发板上**才能测（烧卡 + 串口）。若板上当前是 Ubuntu，需另起一张卡烧 StarryOS。
- **NPU 电源/时钟在 StarryOS 平台层能否拉起**是成败关键，需先核实 tgoskits 的 RK3588 StarryOS 平台初始化。
- 这是按周计的内核研究型工作；与 Linux/NPU（已完成）并行，作为冲高档项。

## 6. 参考

- 开源驱动：`rockchip-linux/kernel` `drivers/rknpu/`（ioctl 头、job 提交、内存管理）
- mainline：`accel/rocket`（更干净的参考实现）
- StarryOS 设备模板：`kernel/src/pseudofs/dev/fb.rs`
