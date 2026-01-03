import torch
import torch.utils.dlpack as dlpack

from perceiver_bench import ModelConfig, Perceiver

_GRAPH_HOLD = []


def _validate_io(board, heur, value, policy):
    if board.device.type != "cuda" or heur.device.type != "cuda":
        raise ValueError("board/heur must be CUDA tensors")
    if value.device.type != "cuda" or policy.device.type != "cuda":
        raise ValueError("value/policy must be CUDA tensors")
    if (
        board.dtype != heur.dtype
        or board.dtype != value.dtype
        or board.dtype != policy.dtype
    ):
        raise ValueError("all tensors must share the same dtype")

    if board.ndim != 4:
        raise ValueError("board must be (B, C, 32, 32)")
    if board.shape[2] != 32 or board.shape[3] != 32:
        raise ValueError("board must be (B, C, 32, 32)")
    if heur.ndim != 2:
        raise ValueError("heur must be (B, H)")
    if value.ndim != 2 or value.shape[1] != 2:
        raise ValueError("value must be (B, 2)")
    if policy.ndim != 2:
        raise ValueError("policy must be (B, A)")

    batch = board.shape[0]
    if heur.shape[0] != batch or value.shape[0] != batch or policy.shape[0] != batch:
        raise ValueError("batch dimension mismatch")


def capture_graph(
    board_dlpack, heur_dlpack, value_dlpack, policy_dlpack, stream_handle, *, device=0
):
    torch.cuda.set_device(device)
    stream = torch.cuda.ExternalStream(stream_handle)

    board = dlpack.from_dlpack(board_dlpack)
    heur = dlpack.from_dlpack(heur_dlpack)
    value_out = dlpack.from_dlpack(value_dlpack)
    policy_out = dlpack.from_dlpack(policy_dlpack)

    _validate_io(board, heur, value_out, policy_out)

    cfg = ModelConfig(
        channels=board.shape[1],
        heuristics_dim=heur.shape[1],
        actions=policy_out.shape[1],
    )
    model = Perceiver(cfg).to(device=board.device, dtype=board.dtype).eval()

    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph(keep_graph=True)

    with torch.inference_mode():
        with torch.cuda.stream(stream):
            _ = model(board, heur)
        torch.cuda.synchronize()

        with torch.cuda.graph(graph, stream=stream, capture_error_mode="thread_local"):
            value, policy = model(board, heur)
            value_out.copy_(value)
            policy_out.copy_(policy)

    graph.instantiate()
    _GRAPH_HOLD.append((graph, model, board, heur, value_out, policy_out))
    return int(graph.raw_cuda_graph_exec())
