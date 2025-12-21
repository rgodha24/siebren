# Siebren

Training infrastructure for AlphaZero-style game engines.

## Architecture

```
rust/src/
├── lib.rs              # Core traits (Action, Environment, Player, TerminalState)
├── mcts.rs             # MCTS implementation
├── evaluator.rs        # ONNX Runtime neural network evaluator
├── replay_buffer.rs    # Experience replay buffer (PyO3 bindings)
└── environments/
    ├── tictactoe.rs    # TicTacToe environment
    └── connect4.rs     # Connect4 environment

src/siebren/
├── __init__.py         # Python package (re-exports Rust bindings)
└── export.py           # ONNX model export with type introspection
```

## Core Concepts

### Action

Represents a move in a game. Must map to/from indices in `0..NUM_ACTIONS` for neural network policy vectors.

### Environment

A game state that supports:

- Checking terminal conditions (win/draw)
- Listing valid actions
- Apply/rollback for efficient tree search without cloning

### Evaluator

Neural network interface. Takes an environment, returns `(policy, value)` where:

- `policy`: probability distribution over actions
- `value`: position evaluation in [-1, 1] from current player's perspective

### MCTS

Monte Carlo Tree Search with:

- UCB action selection with configurable `c_puct`
- Dirichlet noise at root for exploration
- Efficient rollback-based tree traversal

## Adding a New Environment

1. Create `rust/src/environments/yourgame.rs`
2. Define action type implementing `Action`
3. Define game state implementing `Environment`
4. Add `pub mod yourgame;` to `environments/mod.rs`

## Value Semantics

Q values in MCTS nodes represent "how good was this action for the player who chose it". Terminal values:

- Win for player who just moved: +1
- Loss for player who just moved: -1
- Draw: 0

Network returns value from `current_player`'s perspective; MCTS negates appropriately during backprop.

## Development Setup

Requires Nix for reproducible builds with CUDA and OpenSSL. RUN EVERYTHING WRAPPED IN `nix develop -c "WHATEVER YOU WANT TO RUN"`
USE UV FOR EVERYTHING TOO. e.g. to run a script `nix develop -c uv run python3 some_script.py`, install `uv add torch`, etc.
