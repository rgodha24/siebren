import os


def _ensure_cuda_env_for_torch_compile() -> None:
    # torch.compile (inductor->triton) tries to discover libcuda via `/sbin/ldconfig`.
    # In Nix shells this binary may not exist, so point Triton at libcuda explicitly.
    if os.environ.get("TRITON_LIBCUDA_PATH"):
        return

    candidates: list[str] = []
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    candidates.extend([p for p in ld_library_path.split(":") if p])
    candidates.append("/run/opengl-driver/lib")

    for p in candidates:
        if os.path.exists(os.path.join(p, "libcuda.so.1")):
            os.environ["TRITON_LIBCUDA_PATH"] = p
            return


_ensure_cuda_env_for_torch_compile()

print(os.environ["TRITON_LIBCUDA_PATH"])
