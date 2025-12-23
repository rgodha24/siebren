# AGENTS.md

AlphaZero-style game engine training infrastructure. Rust core with Python bindings via PyO3/maturin.

## Build & Test Commands

- **Rust tests**: `cd rust && cargo test` (single test: `cargo test test_name`)
- **Python build**: `maturin develop` (builds Rust extension for Python)
- **Python tests**: `pytest src/tests/` (single test: `pytest src/tests/test_all.py::test_name`)

## Code Style

- **Rust**: Edition 2021, use `cargo fmt` and `cargo clippy`
- **Python**: Follow PEP 8, use type hints
- **Naming**: snake_case for functions/variables, PascalCase for types/traits
- **Imports**: Group std lib, external crates, then local modules

## Architecture

- Core traits in `rust/src/lib.rs`: `Action`, `Environment`, `Player`, `TerminalState`
- MCTS implementation in `rust/src/mcts.rs` with `Evaluator` trait for neural network interface
- Environments in `rust/src/environments/` (implement `Environment` trait)

## Key Patterns

- Environments use apply/rollback for efficient tree search without cloning
- Q values represent "how good was this action for the player who chose it"
- Network returns value from current player's perspective; MCTS negates during backprop

## other stuff

ALWAYS PREPEND COMMANDS WITH `nix develop -c` and IF IT IS PYTHON THEN `nix develop -c uv run ...`
