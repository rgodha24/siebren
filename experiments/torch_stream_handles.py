import torch


def main() -> None:
    print(f"torch {torch.__version__}")
    if not torch.cuda.is_available():
        print("cuda not available")
        return

    stream = torch.cuda.Stream()
    print(f"stream cuda_stream handle: {stream.cuda_stream}")

    ext = torch.cuda.ExternalStream(stream.cuda_stream)
    x = torch.randn((1024,), device="cuda")
    event = torch.cuda.Event()

    with torch.cuda.stream(ext):
        y = x * 2
        event.record()
        _ = y

    print(f"event query before sync: {event.query()}")
    event.synchronize()
    print(f"event query after sync: {event.query()}")

    if hasattr(event, "cuda_event"):
        print(f"event cuda_event handle: {getattr(event, 'cuda_event')}")
    if hasattr(event, "_as_parameter_"):
        print(f"event _as_parameter_: {getattr(event, '_as_parameter_')}")


if __name__ == "__main__":
    main()
