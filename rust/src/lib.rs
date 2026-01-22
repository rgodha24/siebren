use ndarray::{ArrayView, ArrayViewMut, Dimension, Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, RemoveAxis};
use numpy::{PyArray, PyArrayMethods};
use pyo3::prelude::*;
use std::{fmt::Debug, hash::Hash};

/// Extension trait for prepending a batch dimension to a shape.
///
/// The `BatchedDim` associated type is the dimension with batch prepended.
/// We use an associated type instead of `Dimension::Larger` so we can
/// constrain that `BatchedDim::Smaller == Self`.
pub trait BatchDim: Dimension + Clone {
    type BatchedDim: Dimension<Smaller = Self> + RemoveAxis;

    fn with_batch(batch_size: usize, obs_shape: Self) -> Self::BatchedDim;
}

impl BatchDim for Ix0 {
    type BatchedDim = Ix1;

    fn with_batch(batch_size: usize, _obs: Self) -> Ix1 {
        Ix1(batch_size)
    }
}

impl BatchDim for Ix1 {
    type BatchedDim = Ix2;

    fn with_batch(batch_size: usize, obs: Self) -> Ix2 {
        Ix2(batch_size, obs[0])
    }
}

impl BatchDim for Ix2 {
    type BatchedDim = Ix3;

    fn with_batch(batch_size: usize, obs: Self) -> Ix3 {
        Ix3(batch_size, obs[0], obs[1])
    }
}

impl BatchDim for Ix3 {
    type BatchedDim = Ix4;

    fn with_batch(batch_size: usize, obs: Self) -> Ix4 {
        Ix4(batch_size, obs[0], obs[1], obs[2])
    }
}

impl BatchDim for Ix4 {
    type BatchedDim = Ix5;

    fn with_batch(batch_size: usize, obs: Self) -> Ix5 {
        Ix5(batch_size, obs[0], obs[1], obs[2], obs[3])
    }
}

impl BatchDim for Ix5 {
    type BatchedDim = Ix6;

    fn with_batch(batch_size: usize, obs: Self) -> Ix6 {
        Ix6(batch_size, obs[0], obs[1], obs[2], obs[3], obs[4])
    }
}

pub mod environments;
pub mod eval;
pub mod executor;
pub mod future;
mod integration_tests;
pub mod mcts;
pub mod queue;
pub mod replay_buffer;
pub mod training;
pub mod worker;

/// Run TicTacToe self-play with Python model callback.
///
/// The callback receives observations as a (BATCH_SIZE, 9) int8 numpy array.
/// Board encoding: 0 = empty, 1 = PlayerA (X), -1 = PlayerB (O)
///
/// Returns (policy, value) tuple where:
/// - policy: (BATCH_SIZE, 9) float32 - action probabilities
/// - value: (BATCH_SIZE,) float32 - position evaluations in [-1, 1]
#[pyfunction]
#[pyo3(signature = (num_threads, workers_per_thread, target_samples, seed, execute_model))]
fn selfplay_tictactoe(
    py: Python<'_>,
    num_threads: usize,
    workers_per_thread: usize,
    target_samples: usize,
    seed: u64,
    execute_model: Py<PyAny>,
) -> PyResult<(usize, usize)> {
    use environments::TicTacToe;
    use eval::PolicyValue;
    use replay_buffer::ReplayBuffer;
    use training::{run_training, TrainingConfig};
    use worker::WorkerConfig;

    let config = TrainingConfig {
        num_threads,
        workers_per_thread,
        seed,
        target_samples,
        worker: WorkerConfig::default(),
    };

    // Create replay buffer with enough capacity for target samples
    let replay_buffer = ReplayBuffer::new(target_samples);

    let dispatch = move |obs_view: ArrayView<i8, Ix2>, outputs: &mut [PolicyValue<9>]| {
        Python::attach(|py| {
            // Zero-copy numpy array from obs_view
            // SAFETY: obs_view is valid for the duration of this callback,
            // and the numpy array doesn't escape the callback scope.
            let np_obs = unsafe {
                PyArray::borrow_from_array(&obs_view, py.None().into_bound(py))
            };

            // Call Python: (BATCH_SIZE, 9) -> ((BATCH_SIZE, 9), (BATCH_SIZE,))
            let result = execute_model
                .call1(py, (np_obs,))
                .expect("execute_model call failed");

            let (policy_arr, value_arr): (
                Bound<'_, PyArray<f32, Ix2>>,
                Bound<'_, PyArray<f32, Ix1>>,
            ) = result
                .extract(py)
                .expect("expected (policy, value) tuple of numpy arrays");

            // Copy results back to PolicyValue outputs
            let policy = unsafe { policy_arr.as_slice().unwrap() };
            let value = unsafe { value_arr.as_slice().unwrap() };

            for (i, out) in outputs.iter_mut().enumerate() {
                out.policy.copy_from_slice(&policy[i * 9..(i + 1) * 9]);
                out.value = value[i];
            }
        });
    };

    // Release GIL while running training, reacquire in dispatch callback
    let result = py.detach(|| run_training::<TicTacToe, 9, _>(config, &replay_buffer, dispatch));

    Ok((result.games_completed, result.samples_collected))
}

/// Run Connect4 self-play with Python model callback.
///
/// The callback receives observations as a (BATCH_SIZE, 6, 7) int8 numpy array.
/// Board encoding: 0 = empty, 1 = PlayerA, -1 = PlayerB
/// Row 0 is top, row 5 is bottom. Pieces fall down.
///
/// Returns (policy, value) tuple where:
/// - policy: (BATCH_SIZE, 7) float32 - action probabilities for each column
/// - value: (BATCH_SIZE,) float32 - position evaluations in [-1, 1]
#[pyfunction]
#[pyo3(signature = (num_threads, workers_per_thread, target_samples, seed, execute_model))]
fn selfplay_connect4(
    py: Python<'_>,
    num_threads: usize,
    workers_per_thread: usize,
    target_samples: usize,
    seed: u64,
    execute_model: Py<PyAny>,
) -> PyResult<(usize, usize)> {
    use environments::Connect4;
    use eval::PolicyValue;
    use replay_buffer::ReplayBuffer;
    use training::{run_training, TrainingConfig};
    use worker::WorkerConfig;

    let config = TrainingConfig {
        num_threads,
        workers_per_thread,
        seed,
        target_samples,
        worker: WorkerConfig::default(),
    };

    // Create replay buffer with enough capacity for target samples
    let replay_buffer = ReplayBuffer::new(target_samples);

    let dispatch = move |obs_view: ArrayView<i8, Ix3>, outputs: &mut [PolicyValue<7>]| {
        Python::attach(|py| {
            // Zero-copy numpy array from obs_view
            // SAFETY: obs_view is valid for the duration of this callback,
            // and the numpy array doesn't escape the callback scope.
            let np_obs =
                unsafe { PyArray::borrow_from_array(&obs_view, py.None().into_bound(py)) };

            // Call Python: (BATCH_SIZE, 6, 7) -> ((BATCH_SIZE, 7), (BATCH_SIZE,))
            let result = execute_model
                .call1(py, (np_obs,))
                .expect("execute_model call failed");

            let (policy_arr, value_arr): (
                Bound<'_, PyArray<f32, Ix2>>,
                Bound<'_, PyArray<f32, Ix1>>,
            ) = result
                .extract(py)
                .expect("expected (policy, value) tuple of numpy arrays");

            // Copy results back to PolicyValue outputs
            let policy = unsafe { policy_arr.as_slice().unwrap() };
            let value = unsafe { value_arr.as_slice().unwrap() };

            for (i, out) in outputs.iter_mut().enumerate() {
                out.policy.copy_from_slice(&policy[i * 7..(i + 1) * 7]);
                out.value = value[i];
            }
        });
    };

    // Release GIL while running training, reacquire in dispatch callback
    let result = py.detach(|| run_training::<Connect4, 7, _>(config, &replay_buffer, dispatch));

    Ok((result.games_completed, result.samples_collected))
}

#[pymodule]
fn siebren(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(selfplay_tictactoe, m)?)?;
    m.add_function(wrap_pyfunction!(selfplay_connect4, m)?)?;
    Ok(())
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
#[repr(i8)]
pub enum Player {
    PlayerA = 1,
    PlayerB = -1,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TerminalState {
    Win(Player),
    Draw,
}

/// Actions must be convertible to/from a unique index in `0..NUM_ACTIONS`.
pub trait Action: Copy + Eq + Hash {
    fn to_index(self) -> usize;
    fn from_index(index: usize) -> Option<Self>;
}

/// Trait for serializing/deserializing game states to/from a string notation.
pub trait GameNotation: Sized {
    type Error: std::error::Error + Send + Sync + 'static;
    fn to_notation(&self) -> String;
    fn from_notation(s: &str) -> Result<Self, Self::Error>;
}

/// An environment implements a game that we want to train a model to play.
///
/// Environments should support efficient rollback to step in and out of states
/// without cloning.
pub trait Environment: Clone + Hash + Debug + GameNotation {
    /// Element type of observations (u8, i8, f32, etc.)
    type ObsElem: Clone + Default + Send + Sync;
    /// Dimension of a single observation (Ix1, Ix2, etc.)
    type ObsDim: BatchDim;
    /// Shape of a single observation as a compile-time constant.
    const OBS_SHAPE: Self::ObsDim;

    type Action: Action;
    type RollbackState;
    const NUM_ACTIONS: usize;

    /// Creates an environment. Should be randomly generated if possible to
    /// avoid the network overfitting on a single starting position.
    fn new() -> Self;

    /// Returns None if the game is still going, Some(Win/Draw) if it's over.
    fn is_terminal(&self) -> Option<TerminalState>;

    /// Returns an iterator over valid actions.
    fn valid_actions(&self) -> impl Iterator<Item = Self::Action>;

    fn current_player(&self) -> Player;

    /// Write the observation into the provided buffer.
    /// The buffer is a mutable view into the queue's contiguous storage.
    fn observation(&self, out: ArrayViewMut<Self::ObsElem, Self::ObsDim>);

    /// Applies an action and returns state needed for rollback.
    /// Caller must ensure the action is valid per `valid_actions`.
    fn apply_action(&mut self, action: Self::Action) -> Self::RollbackState;

    /// Undoes `apply_action`. After `rollback(apply_action(a))`, the
    /// environment should be back to its original state.
    fn rollback(&mut self, rollback: Self::RollbackState);
}
