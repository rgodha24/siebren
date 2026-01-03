import torch
import torch.utils.dlpack as dlpack


def main() -> None:
    print(f"torch {torch.__version__}")
    x = torch.empty((256, 17), dtype=torch.float32, pin_memory=True)
    to_dlpack = getattr(dlpack, "to_dlpack")
    from_dlpack = getattr(dlpack, "from_dlpack")
    capsule = to_dlpack(x)
    y = from_dlpack(capsule)

    print(f"x data_ptr: {x.data_ptr()}")
    print(f"y data_ptr: {y.data_ptr()}")
    print(f"zero-copy: {x.data_ptr() == y.data_ptr()}")


if __name__ == "__main__":
    main()
