//! Python callback evaluator for neural network inference.
//!
//! This module provides an evaluator that calls back into Python for neural network
//! inference, allowing users to define custom board representations and networks.

use numpy::PyReadonlyArrayDyn;
use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::environments::TicTacToe;
use crate::mcts::Evaluator;
use crate::{Action, Environment};

/// A Python-callable evaluator that batches requests for efficiency.
///
/// The evaluator accumulates game states and calls Python in batches to minimize
/// the overhead of crossing the Python/Rust boundary.
#[pyclass]
pub struct PyEvaluator {
    /// Python function: (list[GameState]) -> (policy_batch: ndarray, value_batch: ndarray)
    eval_fn: PyObject,
}

#[pymethods]
impl PyEvaluator {
    #[new]
    fn new(eval_fn: PyObject) -> Self {
        Self { eval_fn }
    }
}

/// Wrapper around TicTacToe to expose to Python
#[pyclass(name = "TicTacToeState")]
#[derive(Clone)]
pub struct PyTicTacToeState {
    pub inner: TicTacToe,
}

#[pymethods]
impl PyTicTacToeState {
    #[new]
    fn new() -> Self {
        Self {
            inner: TicTacToe::new(),
        }
    }

    /// Get board as a flat list [0-8], where 0=empty, 1=X, 2=O
    fn board(&self) -> Vec<u8> {
        self.inner.board.to_vec()
    }

    /// Get current player: 1 for PlayerA (X), -1 for PlayerB (O)
    fn current_player(&self) -> i8 {
        match self.inner.current_player() {
            crate::Player::PlayerA => 1,
            crate::Player::PlayerB => -1,
        }
    }

    /// Check if game is terminal. Returns None if ongoing, or (winner, is_draw)
    fn is_terminal(&self) -> Option<(i8, bool)> {
        match self.inner.is_terminal() {
            None => None,
            Some(crate::TerminalState::Win(winner)) => {
                let w = match winner {
                    crate::Player::PlayerA => 1,
                    crate::Player::PlayerB => -1,
                };
                Some((w, false))
            }
            Some(crate::TerminalState::Draw) => Some((0, true)),
        }
    }

    /// Get valid action indices
    fn valid_actions(&self) -> Vec<usize> {
        use crate::Action;
        self.inner
            .valid_actions()
            .map(|a| a.to_index())
            .collect()
    }

    /// Apply an action by index, returns True if successful
    fn apply_action(&mut self, action_idx: usize) -> bool {
        use crate::Action;
        use crate::environments::TicTacToeAction;
        if let Some(action) = TicTacToeAction::from_index(action_idx) {
            self.inner.apply_action(action);
            true
        } else {
            false
        }
    }

    fn __repr__(&self) -> String {
        let b = &self.inner.board;
        let sym = |i: usize| match b[i] {
            1 => 'X',
            2 => 'O',
            _ => '.',
        };
        format!(
            "TicTacToe:\n{} {} {}\n{} {} {}\n{} {} {}",
            sym(0), sym(1), sym(2),
            sym(3), sym(4), sym(5),
            sym(6), sym(7), sym(8)
        )
    }
}

/// Benchmark utilities for measuring Python callback overhead
#[pyclass]
pub struct CallbackBenchmark {
    states: Vec<PyTicTacToeState>,
}

#[pymethods]
impl CallbackBenchmark {
    #[new]
    fn new() -> Self {
        Self { states: Vec::new() }
    }

    /// Generate n random game states for benchmarking
    fn generate_states(&mut self, n: usize) {
        use rand::Rng;
        let mut rng = rand::rng();

        self.states.clear();
        for _ in 0..n {
            let mut state = PyTicTacToeState::new();
            // Make 0-5 random moves
            let num_moves = rng.random_range(0..=5);
            for _ in 0..num_moves {
                let valid = state.valid_actions();
                if valid.is_empty() {
                    break;
                }
                let idx = rng.random_range(0..valid.len());
                state.apply_action(valid[idx]);
            }
            self.states.push(state);
        }
    }

    /// Benchmark: call Python function once per state (worst case)
    fn bench_single_calls(&self, py: Python<'_>, eval_fn: PyObject) -> PyResult<f64> {
        use std::time::Instant;

        let start = Instant::now();
        for state in &self.states {
            let list = PyList::new(py, [state.clone()])?;
            let _result = eval_fn.call1(py, (list,))?;
        }
        let elapsed = start.elapsed();

        Ok(elapsed.as_secs_f64())
    }

    /// Benchmark: call Python function with all states at once (best case)
    fn bench_batched_call(&self, py: Python<'_>, eval_fn: PyObject) -> PyResult<f64> {
        use std::time::Instant;

        let start = Instant::now();
        let list = PyList::new(py, self.states.clone())?;
        let _result = eval_fn.call1(py, (list,))?;
        let elapsed = start.elapsed();

        Ok(elapsed.as_secs_f64())
    }

    /// Benchmark: call Python function with batches of given size
    fn bench_batched_calls(&self, py: Python<'_>, eval_fn: PyObject, batch_size: usize) -> PyResult<f64> {
        use std::time::Instant;

        let start = Instant::now();
        for chunk in self.states.chunks(batch_size) {
            let list = PyList::new(py, chunk.to_vec())?;
            let _result = eval_fn.call1(py, (list,))?;
        }
        let elapsed = start.elapsed();

        Ok(elapsed.as_secs_f64())
    }

    /// Pure Rust baseline: just iterate states without Python
    fn bench_rust_baseline(&self) -> f64 {
        use std::time::Instant;

        let start = Instant::now();
        let mut sum = 0u64;
        for state in &self.states {
            // Simulate some work
            for &cell in &state.inner.board {
                sum = sum.wrapping_add(cell as u64);
            }
        }
        let elapsed = start.elapsed();
        // Prevent optimization
        std::hint::black_box(sum);

        elapsed.as_secs_f64()
    }

    fn num_states(&self) -> usize {
        self.states.len()
    }
}

/// Run self-play with Python evaluator callback
/// This is the key function that demonstrates the architecture.
#[pyfunction]
pub fn run_self_play_with_callback(
    py: Python<'_>,
    eval_fn: PyObject,
    num_games: usize,
    num_simulations: usize,
) -> PyResult<Vec<(Vec<u8>, Vec<f32>, f32)>> {
    use crate::mcts::{MCTS, MCTSConfig, visits_to_policy};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    // Create a callback-based evaluator
    struct CallbackEvaluator<'py> {
        py: Python<'py>,
        eval_fn: &'py PyObject,
    }

    impl Evaluator<TicTacToe> for CallbackEvaluator<'_> {
        fn evaluate(&self, env: &TicTacToe) -> (Vec<f32>, f32) {
            // Convert to Python state
            let state = PyTicTacToeState { inner: env.clone() };
            let list = PyList::new(self.py, [state]).unwrap();

            // Call Python
            let result = self.eval_fn.call1(self.py, (list,)).unwrap();
            let (policy, value): (PyReadonlyArrayDyn<f32>, PyReadonlyArrayDyn<f32>) =
                result.extract(self.py).unwrap();

            let policy_vec: Vec<f32> = policy.as_array().iter().copied().collect();
            let value_scalar = value.as_array()[0];

            (policy_vec, value_scalar)
        }
    }

    let config = MCTSConfig {
        num_simulations,
        c_puct: 1.5,
        dirichlet_alpha: 0.3,
        dirichlet_epsilon: 0.25,
    };

    let evaluator = CallbackEvaluator { py, eval_fn: &eval_fn };
    let mcts = MCTS::new(&evaluator, &config);

    let mut experiences = Vec::new();
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    for _ in 0..num_games {
        let mut game = TicTacToe::new();
        let mut game_experiences: Vec<([u8; 9], Vec<f32>)> = Vec::new();

        while game.is_terminal().is_none() {
            let visits = mcts.search(&mut game, &mut rng);
            let policy = visits_to_policy(&visits, 1.0);

            // Store experience
            game_experiences.push((game.observation(), policy.clone()));

            // Sample action
            let action_idx = crate::mcts::sample_action_index(&policy, &mut rng)
                .unwrap_or(0);
            if let Some(action) = crate::environments::TicTacToeAction::from_index(action_idx) {
                game.apply_action(action);
            }
        }

        // Compute final value
        let final_value = match game.is_terminal() {
            Some(crate::TerminalState::Win(winner)) => {
                match winner {
                    crate::Player::PlayerA => 1.0,
                    crate::Player::PlayerB => -1.0,
                }
            }
            _ => 0.0,
        };

        // Add experiences with alternating values
        for (i, (board, policy)) in game_experiences.into_iter().enumerate() {
            let value = if i % 2 == 0 { final_value } else { -final_value };
            experiences.push((board.to_vec(), policy, value));
        }
    }

    Ok(experiences)
}

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEvaluator>()?;
    m.add_class::<PyTicTacToeState>()?;
    m.add_class::<CallbackBenchmark>()?;
    m.add_function(wrap_pyfunction!(run_self_play_with_callback, m)?)?;
    Ok(())
}
