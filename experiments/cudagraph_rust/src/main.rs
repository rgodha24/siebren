use cudarc::runtime::sys as cuda;
use pyo3::ffi;
use pyo3::prelude::*;
use std::env;
use std::ffi::{c_void, CStr};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

static CALLBACK_FIRED: AtomicBool = AtomicBool::new(false);

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
    strides: Option<Box<[i64]>>,
}

const DL_TENSOR_NAME: &[u8] = b"dltensor\0";
const DL_DEVICE_CPU: i32 = 1;
const DL_DEVICE_CUDA: i32 = 2;
const DL_DTYPE_FLOAT: u8 = 2;

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
) -> PyResult<PyObject> {
    let ctx = Box::new(DLPackContext {
        shape: shape.to_vec().into_boxed_slice(),
        strides: None,
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
                code: DL_DTYPE_FLOAT,
                bits: 32,
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
        unsafe {
            dlpack_deleter(managed_ptr);
        }
        return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "failed to create DLPack capsule",
        ));
    }

    Ok(unsafe { PyObject::from_owned_ptr(py, capsule) })
}

#[allow(non_camel_case_types)]
type cudaStream_t = cuda::cudaStream_t;

#[allow(non_camel_case_types)]
type cudaGraphExec_t = cuda::cudaGraphExec_t;

type CudaError = cuda::cudaError_t;

fn check_cuda(code: CudaError, context: &str) {
    if code != cuda::cudaError::cudaSuccess {
        panic!("{} failed with error {:?}", context, code);
    }
}

extern "C" fn host_callback(_user_data: *mut c_void) {
    CALLBACK_FIRED.store(true, Ordering::Release);
}

fn workspace_root() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .expect("missing parent")
        .parent()
        .expect("missing workspace root")
        .to_path_buf()
}

fn ensure_python_path(py: Python<'_>) -> PyResult<()> {
    let sys = PyModule::import(py, "sys")?;
    let sys_path = sys.getattr("path")?;
    sys_path.call_method1("insert", (0, workspace_root().to_string_lossy().as_ref()))?;
    Ok(())
}

fn call_python_callback<'py>(
    callback: &pyo3::Bound<'py, PyAny>,
    input: &pyo3::Bound<'py, PyAny>,
) -> PyResult<pyo3::Bound<'py, PyAny>> {
    callback.call1((input,))
}

fn cpu_tensor_from_dlpack<'py>(
    py: Python<'py>,
    data: &mut [f32],
    shape: &[i64],
) -> PyResult<pyo3::Bound<'py, PyAny>> {
    let dlpack = PyModule::import(py, "torch.utils.dlpack")?;
    let capsule = dlpack_capsule(
        py,
        data.as_mut_ptr() as *mut c_void,
        shape,
        DL_DEVICE_CPU,
        0,
    )?;
    dlpack.getattr("from_dlpack")?.call1((capsule,))
}

fn load_callback<'py>(py: Python<'py>) -> PyResult<pyo3::Bound<'py, PyAny>> {
    let spec = env::var("CPU_CALLBACK")
        .unwrap_or_else(|_| "experiments.cpu_callback:cpu_callback".to_string());
    let (module_name, attr_name) = spec.split_once(':').ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>("CPU_CALLBACK must be 'module:callable'")
    })?;
    let module = PyModule::import(py, module_name)?;
    module.getattr(attr_name)
}

fn run_cpu_callback_with<'py>(
    py: Python<'py>,
    callback: &pyo3::Bound<'py, PyAny>,
) -> PyResult<Vec<f32>> {
    let shape = [4_i64, 8_i64];
    let elems = (shape[0] * shape[1]) as usize;
    let mut input: Vec<f32> = (0..elems).map(|i| i as f32).collect();
    let input_tensor = cpu_tensor_from_dlpack(py, &mut input, &shape)?;
    let output = call_python_callback(callback, &input_tensor)?;
    let output_flat = output.call_method0("flatten")?;
    let output_vec: Vec<f32> = output_flat.call_method0("tolist")?.extract()?;
    Ok(output_vec)
}

fn run_cpu_callback() -> PyResult<Vec<f32>> {
    Python::with_gil(|py| {
        ensure_python_path(py)?;
        let callback = load_callback(py)?;
        run_cpu_callback_with(py, &callback)
    })
}

fn main() -> PyResult<()> {
    let cpu_output = run_cpu_callback()?;
    if let Some(sample) = cpu_output.first() {
        println!("cpu callback output[0]={:.5}", sample);
    }

    let device_id = 0;
    let batch = 8_i64;
    let channels = 24_i64;
    let heuristics = 18_i64;
    let actions = 11_i64;

    let board_shape = [batch, channels, 32_i64, 32_i64];
    let heur_shape = [batch, heuristics];
    let value_shape = [batch, 2_i64];
    let policy_shape = [batch, actions];

    let board_elems = (batch * channels * 32 * 32) as usize;
    let heur_elems = (batch * heuristics) as usize;
    let value_elems = (batch * 2) as usize;
    let policy_elems = (batch * actions) as usize;

    let board_bytes = board_elems * std::mem::size_of::<f32>();
    let heur_bytes = heur_elems * std::mem::size_of::<f32>();
    let value_bytes = value_elems * std::mem::size_of::<f32>();
    let policy_bytes = policy_elems * std::mem::size_of::<f32>();

    let mut stream: cudaStream_t = std::ptr::null_mut();
    unsafe {
        check_cuda(cuda::cudaStreamCreate(&mut stream), "cudaStreamCreate");
    }

    let mut d_board: *mut c_void = std::ptr::null_mut();
    let mut d_heur: *mut c_void = std::ptr::null_mut();
    let mut d_value: *mut c_void = std::ptr::null_mut();
    let mut d_policy: *mut c_void = std::ptr::null_mut();
    unsafe {
        check_cuda(
            cuda::cudaMalloc(&mut d_board, board_bytes),
            "cudaMalloc board",
        );
        check_cuda(cuda::cudaMalloc(&mut d_heur, heur_bytes), "cudaMalloc heur");
        check_cuda(
            cuda::cudaMalloc(&mut d_value, value_bytes),
            "cudaMalloc value",
        );
        check_cuda(
            cuda::cudaMalloc(&mut d_policy, policy_bytes),
            "cudaMalloc policy",
        );
    }

    let host_board: Vec<f32> = (0..board_elems).map(|i| (i % 256) as f32 / 255.0).collect();
    let host_heur: Vec<f32> = vec![0.5; heur_elems];
    let mut host_value: Vec<f32> = vec![0.0; value_elems];
    let mut host_policy: Vec<f32> = vec![0.0; policy_elems];

    unsafe {
        check_cuda(
            cuda::cudaMemcpyAsync(
                d_board,
                host_board.as_ptr() as *const c_void,
                board_bytes,
                cuda::cudaMemcpyKind::cudaMemcpyHostToDevice,
                stream,
            ),
            "cudaMemcpyAsync board H2D",
        );
        check_cuda(
            cuda::cudaMemcpyAsync(
                d_heur,
                host_heur.as_ptr() as *const c_void,
                heur_bytes,
                cuda::cudaMemcpyKind::cudaMemcpyHostToDevice,
                stream,
            ),
            "cudaMemcpyAsync heur H2D",
        );
    }

    let exec_handle = Python::with_gil(|py| -> PyResult<u64> {
        ensure_python_path(py)?;

        let module = PyModule::import(py, "experiments.cudagraph_capture")?;
        let board_capsule = dlpack_capsule(py, d_board, &board_shape, DL_DEVICE_CUDA, device_id)?;
        let heur_capsule = dlpack_capsule(py, d_heur, &heur_shape, DL_DEVICE_CUDA, device_id)?;
        let value_capsule = dlpack_capsule(py, d_value, &value_shape, DL_DEVICE_CUDA, device_id)?;
        let policy_capsule =
            dlpack_capsule(py, d_policy, &policy_shape, DL_DEVICE_CUDA, device_id)?;
        let exec_handle: u64 = module
            .getattr("capture_graph")?
            .call1((
                board_capsule,
                heur_capsule,
                value_capsule,
                policy_capsule,
                stream as u64,
            ))?
            .extract()?;
        Ok(exec_handle)
    })?;

    let exec_ptr = exec_handle as cudaGraphExec_t;
    unsafe {
        check_cuda(cuda::cudaGraphLaunch(exec_ptr, stream), "cudaGraphLaunch");
        check_cuda(
            cuda::cudaLaunchHostFunc(stream, Some(host_callback), std::ptr::null_mut()),
            "cudaLaunchHostFunc",
        );
        check_cuda(
            cuda::cudaMemcpyAsync(
                host_value.as_mut_ptr() as *mut c_void,
                d_value,
                value_bytes,
                cuda::cudaMemcpyKind::cudaMemcpyDeviceToHost,
                stream,
            ),
            "cudaMemcpyAsync value D2H",
        );
        check_cuda(
            cuda::cudaMemcpyAsync(
                host_policy.as_mut_ptr() as *mut c_void,
                d_policy,
                policy_bytes,
                cuda::cudaMemcpyKind::cudaMemcpyDeviceToHost,
                stream,
            ),
            "cudaMemcpyAsync policy D2H",
        );
        check_cuda(cuda::cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    }

    if !CALLBACK_FIRED.load(Ordering::Acquire) {
        eprintln!("host callback did not fire");
    }

    let value_sample = host_value.get(0).copied().unwrap_or(0.0);
    let policy_sample = host_policy.get(0).copied().unwrap_or(0.0);
    println!(
        "value[0]={:.5} policy[0]={:.5}",
        value_sample, policy_sample
    );

    unsafe {
        check_cuda(cuda::cudaFree(d_board), "cudaFree board");
        check_cuda(cuda::cudaFree(d_heur), "cudaFree heur");
        check_cuda(cuda::cudaFree(d_value), "cudaFree value");
        check_cuda(cuda::cudaFree(d_policy), "cudaFree policy");
        check_cuda(cuda::cudaStreamDestroy(stream), "cudaStreamDestroy");
    }

    Ok(())
}
