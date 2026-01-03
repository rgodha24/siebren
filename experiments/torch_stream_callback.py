import ctypes
import time
import torch


def _load_cudart() -> ctypes.CDLL | None:
    for name in ("libcudart.so", "libcudart.so.12", "libcudart.so.11.0"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def main() -> None:
    print(f"torch {torch.__version__}")
    if not torch.cuda.is_available():
        print("cuda not available")
        return

    libcudart = _load_cudart()
    if libcudart is None:
        print("could not load libcudart")
        return

    cuda_launch_host_func = libcudart.cudaLaunchHostFunc
    cuda_launch_host_func.restype = ctypes.c_int
    cuda_launch_host_func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

    flag = ctypes.c_int(0)

    @ctypes.CFUNCTYPE(None, ctypes.c_void_p)
    def _callback(_user_data):
        flag.value = 1

    stream = torch.cuda.Stream()
    stream_handle = ctypes.c_void_p(stream.cuda_stream)

    with torch.cuda.stream(stream):
        x = torch.randn((1024, 1024), device="cuda")
        y = x @ x
        _ = y
        err = cuda_launch_host_func(stream_handle, _callback, None)

    if err != 0:
        print(f"cudaLaunchHostFunc error: {err}")
        return

    t0 = time.time()
    while flag.value == 0:
        time.sleep(0.001)
        if time.time() - t0 > 5.0:
            print("timeout waiting for callback")
            return

    print("callback fired")


if __name__ == "__main__":
    main()
