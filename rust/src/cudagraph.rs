//! Rust-launched CUDA graph backend for ByteFight self-play inference.

use std::ffi::{c_void, CStr};
use std::mem::size_of;
use std::slice;

use cudarc::runtime::sys as cuda;
use ndarray::{ArrayView, Ix3};
use pyo3::exceptions::PyRuntimeError;
use pyo3::ffi;
use pyo3::prelude::*;

use crate::eval::PolicyValue;
use crate::queue::BatchCompletion;

#[allow(non_camel_case_types)]
type cudaStream_t = cuda::cudaStream_t;

#[allow(non_camel_case_types)]
type cudaGraphExec_t = cuda::cudaGraphExec_t;

type CudaError = cuda::cudaError_t;

const DL_TENSOR_NAME: &[u8] = b"dltensor\0";
const DL_DEVICE_CPU: i32 = 1;
const DL_DEVICE_CUDA: i32 = 2;
const DL_DTYPE_FLOAT: u8 = 2;
const DL_DTYPE_UINT: u8 = 1;

const BYTEFIGHT_OBS_SIDE: usize = 18;
const BYTEFIGHT_OBS_WIDTH: usize = 16;
const BYTEFIGHT_OBS_CELLS: usize = BYTEFIGHT_OBS_SIDE * BYTEFIGHT_OBS_WIDTH;
const BYTEFIGHT_ACTIONS: usize = 7;

#[repr(C)]
struct DLDevice {
    device_type: i32,
    device_id: i32,
}

#[repr(C)]
struct DLDataType {
    code: u8,
    bits: u8,
    lanes: u16,
}

#[repr(C)]
struct DLTensor {
    data: *mut c_void,
    device: DLDevice,
    ndim: i32,
    dtype: DLDataType,
    shape: *mut i64,
    strides: *mut i64,
    byte_offset: usize,
}

#[repr(C)]
struct DLManagedTensor {
    dl_tensor: DLTensor,
    manager_ctx: *mut c_void,
    deleter: Option<extern "C" fn(*mut DLManagedTensor)>,
}

struct DLPackContext {
    shape: Box<[i64]>,
}

unsafe extern "C" fn dlpack_capsule_destructor(capsule: *mut ffi::PyObject) {
    if capsule.is_null() {
        return;
    }

    let name = ffi::PyCapsule_GetName(capsule);
    if name.is_null() {
        return;
    }

    let c_name = CStr::from_ptr(name);
    if c_name.to_bytes() != b"dltensor" {
        return;
    }

    let ptr = ffi::PyCapsule_GetPointer(capsule, DL_TENSOR_NAME.as_ptr() as *const i8);
    if ptr.is_null() {
        return;
    }

    let managed = ptr as *mut DLManagedTensor;
    if let Some(deleter) = unsafe { (*managed).deleter } {
        deleter(managed);
    }
}

extern "C" fn dlpack_deleter(ptr: *mut DLManagedTensor) {
    if ptr.is_null() {
        return;
    }

    unsafe {
        let ctx_ptr = (*ptr).manager_ctx as *mut DLPackContext;
        if !ctx_ptr.is_null() {
            drop(Box::from_raw(ctx_ptr));
        }
        drop(Box::from_raw(ptr));
    }
}

fn dlpack_capsule(
    py: Python<'_>,
    data: *mut c_void,
    shape: &[i64],
    device_type: i32,
    device_id: i32,
    dtype_code: u8,
    dtype_bits: u8,
) -> PyResult<Py<PyAny>> {
    let ctx = Box::new(DLPackContext {
        shape: shape.to_vec().into_boxed_slice(),
    });
    let shape_ptr = ctx.shape.as_ptr() as *mut i64;
    let ctx_ptr = Box::into_raw(ctx);

    let managed = Box::new(DLManagedTensor {
        dl_tensor: DLTensor {
            data,
            device: DLDevice {
                device_type,
                device_id,
            },
            ndim: shape.len() as i32,
            dtype: DLDataType {
                code: dtype_code,
                bits: dtype_bits,
                lanes: 1,
            },
            shape: shape_ptr,
            strides: std::ptr::null_mut(),
            byte_offset: 0,
        },
        manager_ctx: ctx_ptr as *mut c_void,
        deleter: Some(dlpack_deleter),
    });

    let managed_ptr = Box::into_raw(managed);
    let capsule = unsafe {
        ffi::PyCapsule_New(
            managed_ptr as *mut c_void,
            DL_TENSOR_NAME.as_ptr() as *const i8,
            Some(dlpack_capsule_destructor),
        )
    };

    if capsule.is_null() {
        dlpack_deleter(managed_ptr);
        return Err(PyErr::new::<PyRuntimeError, _>(
            "failed to create DLPack capsule",
        ));
    }

    Ok(unsafe { Py::from_owned_ptr(py, capsule) })
}

fn check_cuda(code: CudaError, context: &str) -> PyResult<()> {
    if code == cuda::cudaError::cudaSuccess {
        Ok(())
    } else {
        Err(PyErr::new::<PyRuntimeError, _>(format!(
            "{} failed with CUDA error {:?}",
            context, code
        )))
    }
}

fn check_cuda_or_panic(code: CudaError, context: &str) {
    if code != cuda::cudaError::cudaSuccess {
        panic!("{} failed with CUDA error {:?}", context, code);
    }
}

fn cuda_malloc_host_f32(count: usize, context: &str) -> PyResult<*mut f32> {
    let mut ptr: *mut c_void = std::ptr::null_mut();
    let bytes = count
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| PyErr::new::<PyRuntimeError, _>("host allocation size overflow"))?;
    unsafe {
        check_cuda(
            cuda::cudaMallocHost(&mut ptr as *mut *mut c_void, bytes),
            context,
        )?;
    }
    Ok(ptr.cast::<f32>())
}

fn cuda_malloc_host_u8(count: usize, context: &str) -> PyResult<*mut u8> {
    let mut ptr: *mut c_void = std::ptr::null_mut();
    let bytes = count;
    unsafe {
        check_cuda(
            cuda::cudaMallocHost(&mut ptr as *mut *mut c_void, bytes),
            context,
        )?;
    }
    Ok(ptr.cast::<u8>())
}

fn cuda_malloc_device_f32(count: usize, context: &str) -> PyResult<*mut c_void> {
    let mut ptr: *mut c_void = std::ptr::null_mut();
    let bytes = count
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| PyErr::new::<PyRuntimeError, _>("device allocation size overflow"))?;
    unsafe {
        check_cuda(
            cuda::cudaMalloc(&mut ptr as *mut *mut c_void, bytes),
            context,
        )?;
    }
    Ok(ptr)
}

fn cuda_malloc_device_u8(count: usize, context: &str) -> PyResult<*mut c_void> {
    let mut ptr: *mut c_void = std::ptr::null_mut();
    let bytes = count;
    unsafe {
        check_cuda(
            cuda::cudaMalloc(&mut ptr as *mut *mut c_void, bytes),
            context,
        )?;
    }
    Ok(ptr)
}

struct ByteFightCudaGraphLane {
    stream: cudaStream_t,
    graph_exec: cudaGraphExec_t,
    /// Owns Python-side graph/tensor objects for this lane.
    _py_owner: Py<PyAny>,
    obs_host: *mut u8,
    obs_dev: *mut c_void,
    policy_host: *mut f32,
    policy_dev: *mut c_void,
    value_host: *mut f32,
    value_dev: *mut c_void,
}

struct LaneCompletionContext {
    policy_host: *const f32,
    value_host: *const f32,
    batch_size: usize,
    completion: Option<BatchCompletion<PolicyValue<7>>>,
}

unsafe impl Send for LaneCompletionContext {}

unsafe extern "C" fn lane_completion_callback(user_data: *mut c_void) {
    if user_data.is_null() {
        return;
    }

    let mut ctx = unsafe { Box::from_raw(user_data.cast::<LaneCompletionContext>()) };
    let policy_src =
        unsafe { slice::from_raw_parts(ctx.policy_host, ctx.batch_size * BYTEFIGHT_ACTIONS) };
    let value_src = unsafe { slice::from_raw_parts(ctx.value_host, ctx.batch_size) };

    let mut outputs = vec![PolicyValue::<7>::default(); ctx.batch_size];
    for (i, out) in outputs.iter_mut().enumerate() {
        let start = i * BYTEFIGHT_ACTIONS;
        out.policy
            .copy_from_slice(&policy_src[start..start + BYTEFIGHT_ACTIONS]);
        out.value = value_src[i];
    }

    if let Some(completion) = ctx.completion.take() {
        completion.complete(&outputs);
    }
}

impl Drop for ByteFightCudaGraphLane {
    fn drop(&mut self) {
        unsafe {
            let _ = cuda::cudaFree(self.obs_dev);
            let _ = cuda::cudaFree(self.policy_dev);
            let _ = cuda::cudaFree(self.value_dev);
            let _ = cuda::cudaFreeHost(self.obs_host.cast::<c_void>());
            let _ = cuda::cudaFreeHost(self.policy_host.cast::<c_void>());
            let _ = cuda::cudaFreeHost(self.value_host.cast::<c_void>());
            let _ = cuda::cudaStreamDestroy(self.stream);
        }
    }
}

/// Per-lane CUDA graph executor for ByteFight self-play inference.
pub struct ByteFightCudaGraphRunner {
    batch_size: usize,
    lanes: Vec<ByteFightCudaGraphLane>,
}

// SAFETY: Lane buffers/streams are independent per batch_idx and queue dispatch
// ensures a lane is not reused before dispatch returns for that lane.
unsafe impl Send for ByteFightCudaGraphRunner {}
unsafe impl Sync for ByteFightCudaGraphRunner {}

impl ByteFightCudaGraphRunner {
    pub fn new(
        py: Python<'_>,
        model: Py<PyAny>,
        gpu_id: usize,
        num_lanes: usize,
        batch_size: usize,
        precision: &str,
    ) -> PyResult<Self> {
        if num_lanes == 0 {
            return Err(PyErr::new::<PyRuntimeError, _>("num_lanes must be > 0"));
        }
        if batch_size == 0 {
            return Err(PyErr::new::<PyRuntimeError, _>("batch_size must be > 0"));
        }

        let module = PyModule::import(py, "siebren.cudagraph_backend")?;
        let capture_fn = module.getattr("capture_bytefight_lane_graph")?;

        let obs_count = batch_size * BYTEFIGHT_OBS_CELLS;
        let policy_count = batch_size * BYTEFIGHT_ACTIONS;
        let value_count = batch_size;

        let obs_shape = [
            batch_size as i64,
            BYTEFIGHT_OBS_SIDE as i64,
            BYTEFIGHT_OBS_WIDTH as i64,
        ];
        let policy_shape = [batch_size as i64, BYTEFIGHT_ACTIONS as i64];
        let value_shape = [batch_size as i64];

        let mut lanes = Vec::with_capacity(num_lanes);

        for lane_idx in 0..num_lanes {
            let mut stream: cudaStream_t = std::ptr::null_mut();
            unsafe {
                check_cuda(
                    cuda::cudaStreamCreate(&mut stream as *mut cudaStream_t),
                    "cudaStreamCreate",
                )?;
            }

            let obs_host =
                cuda_malloc_host_u8(obs_count, &format!("cudaMallocHost obs lane {}", lane_idx))?;
            let policy_host = cuda_malloc_host_f32(
                policy_count,
                &format!("cudaMallocHost policy lane {}", lane_idx),
            )?;
            let value_host = cuda_malloc_host_f32(
                value_count,
                &format!("cudaMallocHost value lane {}", lane_idx),
            )?;

            let obs_dev =
                cuda_malloc_device_u8(obs_count, &format!("cudaMalloc obs lane {}", lane_idx))?;
            let policy_dev = cuda_malloc_device_f32(
                policy_count,
                &format!("cudaMalloc policy lane {}", lane_idx),
            )?;
            let value_dev = cuda_malloc_device_f32(
                value_count,
                &format!("cudaMalloc value lane {}", lane_idx),
            )?;

            let obs_host_capsule = dlpack_capsule(
                py,
                obs_host.cast::<c_void>(),
                &obs_shape,
                DL_DEVICE_CPU,
                gpu_id as i32,
                DL_DTYPE_UINT,
                8,
            )?;
            let obs_dev_capsule = dlpack_capsule(
                py,
                obs_dev,
                &obs_shape,
                DL_DEVICE_CUDA,
                gpu_id as i32,
                DL_DTYPE_UINT,
                8,
            )?;
            let policy_host_capsule = dlpack_capsule(
                py,
                policy_host.cast::<c_void>(),
                &policy_shape,
                DL_DEVICE_CPU,
                gpu_id as i32,
                DL_DTYPE_FLOAT,
                32,
            )?;
            let policy_dev_capsule = dlpack_capsule(
                py,
                policy_dev,
                &policy_shape,
                DL_DEVICE_CUDA,
                gpu_id as i32,
                DL_DTYPE_FLOAT,
                32,
            )?;
            let value_host_capsule = dlpack_capsule(
                py,
                value_host.cast::<c_void>(),
                &value_shape,
                DL_DEVICE_CPU,
                gpu_id as i32,
                DL_DTYPE_FLOAT,
                32,
            )?;
            let value_dev_capsule = dlpack_capsule(
                py,
                value_dev,
                &value_shape,
                DL_DEVICE_CUDA,
                gpu_id as i32,
                DL_DTYPE_FLOAT,
                32,
            )?;

            let (exec_handle, py_owner): (u64, Py<PyAny>) = capture_fn
                .call1((
                    model.clone_ref(py),
                    obs_host_capsule,
                    obs_dev_capsule,
                    policy_host_capsule,
                    policy_dev_capsule,
                    value_host_capsule,
                    value_dev_capsule,
                    stream as u64,
                    precision,
                ))?
                .extract()?;

            let lane = ByteFightCudaGraphLane {
                stream,
                graph_exec: exec_handle as cudaGraphExec_t,
                _py_owner: py_owner,
                obs_host,
                obs_dev,
                policy_host,
                policy_dev,
                value_host,
                value_dev,
            };
            lanes.push(lane);
        }

        Ok(Self { batch_size, lanes })
    }

    pub fn dispatch_async(
        &self,
        batch_idx: usize,
        obs_view: ArrayView<u8, Ix3>,
        completion: BatchCompletion<PolicyValue<7>>,
    ) {
        debug_assert_eq!(
            obs_view.shape(),
            &[self.batch_size, BYTEFIGHT_OBS_SIDE, BYTEFIGHT_OBS_WIDTH]
        );

        let lane = &self.lanes[batch_idx % self.lanes.len()];

        let obs_src = obs_view
            .as_slice()
            .expect("bytefight queue observation batch must be contiguous");
        let obs_dst = unsafe {
            slice::from_raw_parts_mut(lane.obs_host, self.batch_size * BYTEFIGHT_OBS_CELLS)
        };
        obs_dst.copy_from_slice(obs_src);

        unsafe {
            check_cuda_or_panic(
                cuda::cudaGraphLaunch(lane.graph_exec, lane.stream),
                "cudaGraphLaunch",
            );

            let ctx = Box::new(LaneCompletionContext {
                policy_host: lane.policy_host,
                value_host: lane.value_host,
                batch_size: self.batch_size,
                completion: Some(completion),
            });
            check_cuda_or_panic(
                cuda::cudaLaunchHostFunc(
                    lane.stream,
                    Some(lane_completion_callback),
                    Box::into_raw(ctx).cast::<c_void>(),
                ),
                "cudaLaunchHostFunc",
            );
        }
    }
}
