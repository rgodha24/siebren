import torch


def main() -> None:
    print(f"torch {torch.__version__}")
    if not torch.cuda.is_available():
        print("cuda not available")
        return

    x = torch.randn((1024,), device="cuda")
    y = torch.empty_like(x)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        y.copy_(x * 2)

    graph.instantiate()
    graph.replay()
    torch.cuda.synchronize()

    if hasattr(graph, "raw_cuda_graph"):
        print(f"raw_cuda_graph: {graph.raw_cuda_graph()}")
    if hasattr(graph, "raw_cuda_graph_exec"):
        print(f"raw_cuda_graph_exec: {graph.raw_cuda_graph_exec()}")

    candidates = [
        "cuda_graph",
        "_cuda_graph",
        "graph",
        "_graph",
        "graph_exec",
        "_graph_exec",
        "cuda_graph_exec",
    ]
    for name in candidates:
        if hasattr(graph, name):
            print(f"graph attr {name}: {getattr(graph, name)}")


if __name__ == "__main__":
    main()
