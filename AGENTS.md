# AGENTS.md

AlphaZero-style game engine training infrastructure. Rust core with Python bindings via PyO3/maturin.

## Quick Reference

```bash
# IMPORTANT: All commands must be prefixed with `nix develop -c`
# Python commands: `nix develop -c uv run ...`

# Build
nix develop -c maturin develop          # Build Rust extension for Python

# Rust tests
nix develop -c cargo test               # Run all tests (from rust/ directory)
nix develop -c cargo test test_name     # Run single test
nix develop -c cargo test -- --nocapture  # Show println! output

# Python tests
nix develop -c uv run pytest src/tests/              # Run all tests
nix develop -c uv run pytest src/tests/test_all.py::test_name  # Single test

# Linting
nix develop -c cargo fmt                # Format Rust code
nix develop -c cargo clippy             # Lint Rust code

# adding deps. ALWAYS USE COMMANDS TO ADD DEPS TO MAKE SURE WE GET LATEST VERSIONS.
uv add numpy
cargo add tokio
```

## Project Structure

```
siebren/
├── rust/                      # Rust crate (core logic)
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs             # Core traits: Action, Environment, Player, TerminalState
│       ├── mcts.rs            # Async MCTS with GPU-batched inference
│       ├── eval.rs            # Evaluator trait and implementations
│       ├── executor.rs        # Single-threaded async executor for workers
│       ├── queue.rs           # Lock-free GPU job queue for batching
│       ├── training.rs        # Thread spawning and coordination
│       ├── worker.rs          # Worker loop and training data collection
│       └── environments/      # Game implementations
│           ├── tictactoe.rs
│           └── connect4.rs
├── src/
│   ├── scripts/train.py       # Training script
│   └── siebren/               # Python package (compiled extension)
└── pyproject.toml             # Python project config (maturin build)
```

## Rust Code Style

### General

- **Edition**: 2021
- **Formatting**: Use `cargo fmt` before committing
- **Linting**: Run `cargo clippy` and fix warnings

### Naming Conventions

- `snake_case` for functions, methods, variables, modules
- `PascalCase` for types, traits, enums
- `SCREAMING_SNAKE_CASE` for constants
- Prefix private helper functions with descriptive names, not underscores

### Imports

Group imports in this order, separated by blank lines:

1. Standard library (`std::...`)
2. External crates (`rand::...`, `pyo3::...`)
3. Local modules (`crate::...`)

```rust
use std::future::Future;
use std::sync::Arc;

use rand::Rng;
use pyo3::prelude::*;

use crate::eval::Evaluator;
use crate::{Action, Environment};
```

### Types and Traits

- Prefer `impl Trait` in return position for iterators: `fn valid_actions(&self) -> impl Iterator<Item = Self::Action>`
- Use associated types for trait-specific types
- Derive common traits: `#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]`
- Use `#[inline]` for small, frequently-called methods

### Error Handling

- Use `Option` for values that may not exist
- Use `.expect("descriptive message")` when panicking is acceptable (e.g., invariant violations)
- Avoid `unwrap()` - prefer `expect()` with context

### Documentation

- Use `//!` for module-level docs at the top of files
- Use `///` for public items
- Document safety invariants for `unsafe` blocks with `// SAFETY:` comments

```rust
//! Lock-free GPU job queue for batching inference requests.

/// A training sample from a single game step.
pub struct TrainingSample<E: Environment> {
    /// The environment state at this step.
    pub env: E,
}

// SAFETY: We own this slot exclusively until we increment batch_writes
unsafe {
    *self.inputs[slot_idx].get() = input;
}
```

### Async Patterns

- Use `Box::pin` for recursive async calls
- Signal progress with `signal_progress()` when making forward progress in futures
- Tests use `Executor::new(|| event.listen())` pattern for running async code

### Testing

- Place tests in `#[cfg(test)] mod tests` at the bottom of each file
- Use `#[test]` attribute for test functions
- Test names should be descriptive: `test_apply_and_rollback`, `test_horizontal_win`
- Use `Rc<RefCell<_>>` pattern for capturing values in async tests

## Python Code Style

### General

- Follow PEP 8
- Use type hints for all function signatures
- Use `torch.Tensor` type hints for tensor arguments

### Imports

```python
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
```

### Neural Network Patterns

- Inherit from `nn.Module`
- Define layers in `__init__`, use them in `forward`
- Use descriptive comments for tensor shapes: `# (B, 64)`

## Architecture Patterns

### Environment Trait

Games implement `Environment` with:

- `apply_action()` returns `RollbackState` for efficient tree search
- `rollback()` undoes an action using the rollback state
- No cloning needed during MCTS - use apply/rollback pairs

### Value Semantics

- Q values represent "how good was this action for the player who chose it"
- Network returns value from current player's perspective
- MCTS negates values during backpropagation when players alternate

### GPU Batching

- Workers submit observations to `GpuJobQueue`
- When batch fills (256 jobs), dispatch callback runs inference
- Lock-free design using atomic operations for slot assignment

## Common Patterns

### Implementing a New Game

1. Create `environments/game.rs`
2. Define `GameAction` implementing `Action` trait
3. Define `Game` implementing `Environment` trait
4. Define `GameRollback` for rollback state
5. Add to `environments/mod.rs`
6. Write tests for action trait, new game, apply/rollback, win detection

### Running MCTS Search

```rust
let config = MCTSConfig::default();
let evaluator = UniformEvaluator;  // or GpuEvaluator
let mcts = MCTS::new(&evaluator, &config);
let visits = mcts.search(&mut env, &mut rng).await;
let policy = visits_to_policy(&visits, temperature);
```

## Dependencies

### Rust (Cargo.toml)

- `pyo3`: Python bindings
- `ndarray`: N-dimensional arrays (observations)
- `rand`, `rand_chacha`, `rand_distr`: Random number generation
- `event-listener`: Async event signaling

### Python (pyproject.toml)

- `torch`: PyTorch for neural networks
- `numpy`: Array operations
- Build: `maturin` (Rust-Python build tool)
