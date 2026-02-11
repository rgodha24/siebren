use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::sync::{Mutex, OnceLock};
use std::{fmt::Debug, hash::Hash};

use ndarray::{ArrayView, ArrayViewMut, Dimension, Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, RemoveAxis};
use numpy::{PyArray, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

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

pub mod cudagraph;
pub mod environments;
pub mod eval;
pub mod executor;
pub mod future;
mod integration_tests;
pub mod mcts;
pub mod observation_replay_buffer;
pub mod queue;
pub mod replay_buffer;
pub mod training;
pub mod worker;

use environments::{ByteFight, Connect4, TicTacToe};
use observation_replay_buffer::ObservationReplayBuffer;

struct ByteFightGraphCacheEntry {
    model_ptr: usize,
    num_batches: usize,
    precision: String,
    runner: Arc<cudagraph::ByteFightCudaGraphRunner>,
}

static BYTEFIGHT_GRAPH_CACHE: OnceLock<Mutex<Option<ByteFightGraphCacheEntry>>> = OnceLock::new();

fn bytefight_graph_cache() -> &'static Mutex<Option<ByteFightGraphCacheEntry>> {
    BYTEFIGHT_GRAPH_CACHE.get_or_init(|| Mutex::new(None))
}

/// Macro to generate typed ephemeral replay buffer classes for each environment.
///
/// Each generated class wraps an `Arc<ObservationReplayBuffer<...>>` and
/// exposes numpy sampling.
macro_rules! typed_ephemeral_replay_buffer {
    (
        $name:ident,
        $obs_ty:ty,
        $obs_single_dim:ty,
        $obs_batched_dim:ty,
        $obs_shape_const:expr,
        $obs_shape_fn:expr,
        $num_actions:expr
    ) => {
        #[doc = concat!("Typed ephemeral replay buffer for ", stringify!($name), ".")]
        #[doc = ""]
        #[doc = "Stores contiguous observations, policies, and values in memory."]
        #[pyclass]
        pub struct $name {
            inner: Arc<ObservationReplayBuffer<$obs_ty, $obs_single_dim, $num_actions>>,
        }

        #[pymethods]
        impl $name {
            #[new]
            fn new(capacity: usize) -> Self {
                Self {
                    inner: Arc::new(ObservationReplayBuffer::new(capacity, $obs_shape_const)),
                }
            }

            fn __len__(&self) -> usize {
                self.inner.len()
            }

            #[getter]
            fn capacity(&self) -> usize {
                self.inner.capacity()
            }

            /// Sample `n` items and return (observations, policies, values) as numpy arrays.
            ///
            /// Args:
            ///     n: Number of samples to draw
            ///     seed: Random seed for reproducible sampling
            ///
            /// Returns:
            ///     Tuple of (observations, policies, values) numpy arrays
            fn sample<'py>(
                &self,
                py: Python<'py>,
                n: usize,
                seed: u64,
            ) -> PyResult<(
                Bound<'py, PyArray<$obs_ty, $obs_batched_dim>>,
                Bound<'py, PyArray<f32, Ix2>>,
                Bound<'py, PyArray<f32, Ix1>>,
            )> {
                let mut rng = ChaCha8Rng::seed_from_u64(seed);
                let batch = self.inner.sample(n, &mut rng);
                let num_samples = batch.values.len();

                let obs_data = batch.observations.into_raw_vec_and_offset().0;
                let policy_data = batch.policies.into_raw_vec_and_offset().0;

                let shape_fn: fn(usize) -> $obs_batched_dim = $obs_shape_fn;
                let obs = PyArray::from_vec(py, obs_data).reshape(shape_fn(num_samples))?;
                let policies =
                    PyArray::from_vec(py, policy_data).reshape(Ix2(num_samples, $num_actions))?;
                let values = PyArray::from_vec(py, batch.values);

                Ok((obs, policies, values))
            }
        }

        impl $name {
            pub fn inner(
                &self,
            ) -> &Arc<ObservationReplayBuffer<$obs_ty, $obs_single_dim, $num_actions>> {
                &self.inner
            }
        }
    };
}

// TicTacToe: observations (9,) i8, sampled as (n, 9)
typed_ephemeral_replay_buffer!(
    TicTacToeEphemeralReplayBuffer,
    i8,
    Ix1,
    Ix2,
    TicTacToe::OBS_SHAPE,
    |n| Ix2(n, 9),
    9
);

// Connect4: observations (6, 7) i8, sampled as (n, 6, 7)
typed_ephemeral_replay_buffer!(
    Connect4EphemeralReplayBuffer,
    i8,
    Ix2,
    Ix3,
    Connect4::OBS_SHAPE,
    |n| Ix3(n, 6, 7),
    7
);

// ByteFight: observations (16, 16) u8 bit-packed planes, sampled as (n, 16, 16)
typed_ephemeral_replay_buffer!(
    ByteFightEphemeralReplayBuffer,
    u8,
    Ix2,
    Ix3,
    ByteFight::OBS_SHAPE,
    |n| Ix3(n, 16, 16),
    11
);

/// Run TicTacToe self-play with Python model callback.
///
/// The callback receives observations as a (BATCH_SIZE, 9) int8 numpy array.
/// Board encoding: 0 = empty, 1 = PlayerA (X), -1 = PlayerB (O)
///
/// Returns (policy, value) tuple where:
/// - policy: (BATCH_SIZE, 9) float32 - action probabilities
/// - value: (BATCH_SIZE,) float32 - position evaluations in [-1, 1]
#[pyfunction]
#[pyo3(signature = (
    replay_buffer,
    num_threads,
    workers_per_thread,
    target_samples,
    seed,
    execute_model,
    mcts_num_simulations = 20
))]
fn selfplay_tictactoe_ephemeral(
    py: Python<'_>,
    replay_buffer: &TicTacToeEphemeralReplayBuffer,
    num_threads: usize,
    workers_per_thread: usize,
    target_samples: usize,
    seed: u64,
    execute_model: Py<PyAny>,
    mcts_num_simulations: usize,
) -> PyResult<(usize, usize, Py<PyDict>)> {
    use eval::PolicyValue;
    use mcts::MCTSConfig;
    use training::{run_training, TrainingConfig};
    use worker::WorkerConfig;

    if mcts_num_simulations == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "mcts_num_simulations must be >= 1",
        ));
    }

    let config = TrainingConfig {
        num_threads,
        workers_per_thread,
        seed,
        target_samples,
        worker: WorkerConfig {
            mcts: MCTSConfig {
                num_simulations: mcts_num_simulations,
                ..Default::default()
            },
            ..Default::default()
        },
    };

    let dispatch = move |_batch_idx: usize,
                         obs_view: ArrayView<i8, Ix2>,
                         outputs: &mut [PolicyValue<9>]| {
        Python::attach(|py| {
            // Zero-copy numpy array from obs_view
            // SAFETY: obs_view is valid for the duration of this callback,
            // and the numpy array doesn't escape the callback scope.
            let np_obs = unsafe { PyArray::borrow_from_array(&obs_view, py.None().into_bound(py)) };

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
    let result =
        py.detach(|| run_training::<TicTacToe, 9, _>(config, replay_buffer.inner(), dispatch));

    let stats = PyDict::new(py);
    stats.set_item("poll_rounds", result.executor.poll_rounds)?;
    stats.set_item("futures_polled", result.executor.futures_polled)?;
    stats.set_item("poll_ready", result.executor.poll_ready)?;
    stats.set_item("poll_pending", result.executor.poll_pending)?;
    stats.set_item("wait_count", result.executor.wait_count)?;

    Ok((
        result.games_completed,
        result.samples_collected,
        stats.into(),
    ))
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
#[pyo3(signature = (
    replay_buffer,
    num_threads,
    workers_per_thread,
    target_samples,
    seed,
    execute_model,
    mcts_num_simulations = 20
))]
fn selfplay_connect4_ephemeral(
    py: Python<'_>,
    replay_buffer: &Connect4EphemeralReplayBuffer,
    num_threads: usize,
    workers_per_thread: usize,
    target_samples: usize,
    seed: u64,
    execute_model: Py<PyAny>,
    mcts_num_simulations: usize,
) -> PyResult<(usize, usize, Py<PyDict>)> {
    use eval::PolicyValue;
    use mcts::MCTSConfig;
    use training::{run_training, TrainingConfig};
    use worker::WorkerConfig;

    if mcts_num_simulations == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "mcts_num_simulations must be >= 1",
        ));
    }

    let config = TrainingConfig {
        num_threads,
        workers_per_thread,
        seed,
        target_samples,
        worker: WorkerConfig {
            mcts: MCTSConfig {
                num_simulations: mcts_num_simulations,
                ..Default::default()
            },
            ..Default::default()
        },
    };

    let dispatch = move |_batch_idx: usize,
                         obs_view: ArrayView<i8, Ix3>,
                         outputs: &mut [PolicyValue<7>]| {
        Python::attach(|py| {
            // Zero-copy numpy array from obs_view
            // SAFETY: obs_view is valid for the duration of this callback,
            // and the numpy array doesn't escape the callback scope.
            let np_obs = unsafe { PyArray::borrow_from_array(&obs_view, py.None().into_bound(py)) };

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
    let result =
        py.detach(|| run_training::<Connect4, 7, _>(config, replay_buffer.inner(), dispatch));

    let stats = PyDict::new(py);
    stats.set_item("poll_rounds", result.executor.poll_rounds)?;
    stats.set_item("futures_polled", result.executor.futures_polled)?;
    stats.set_item("poll_ready", result.executor.poll_ready)?;
    stats.set_item("poll_pending", result.executor.poll_pending)?;
    stats.set_item("wait_count", result.executor.wait_count)?;

    Ok((
        result.games_completed,
        result.samples_collected,
        stats.into(),
    ))
}

/// Run ByteFight self-play with Python model callback.
///
/// The callback receives observations as a (BATCH_SIZE, 16, 16) uint8 numpy array.
/// Each cell stores 8 one-hot planes packed into bits 0..7.
///
/// Returns (policy, value) tuple where:
/// - policy: (BATCH_SIZE, 11) float32 - action probabilities
///   Actions: 0-7 = directions (N,NE,E,SE,S,SW,W,NW), 8=Trap, 9=FF, 10=EndTurn
/// - value: (BATCH_SIZE,) float32 - position evaluations in [-1, 1]
#[pyfunction]
#[pyo3(signature = (
    replay_buffer,
    num_threads,
    workers_per_thread,
    target_samples,
    seed,
    execute_model = None,
    *,
    mcts_num_simulations = 20,
    use_rust_cudagraph = false,
    model = None,
    selfplay_precision = "fp32"
))]
fn selfplay_bytefight_ephemeral(
    py: Python<'_>,
    replay_buffer: &ByteFightEphemeralReplayBuffer,
    num_threads: usize,
    workers_per_thread: usize,
    target_samples: usize,
    seed: u64,
    execute_model: Option<Py<PyAny>>,
    mcts_num_simulations: usize,
    use_rust_cudagraph: bool,
    model: Option<Py<PyAny>>,
    selfplay_precision: &str,
) -> PyResult<(usize, usize, Py<PyDict>)> {
    use cudagraph::ByteFightCudaGraphRunner;
    use eval::PolicyValue;
    use mcts::MCTSConfig;
    use queue::{queue_shape_for_workers, BATCH_SIZE};
    use training::{run_training, TrainingConfig};
    use worker::WorkerConfig;

    if mcts_num_simulations == 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "mcts_num_simulations must be >= 1",
        ));
    }

    let config = TrainingConfig {
        num_threads,
        workers_per_thread,
        seed,
        target_samples,
        worker: WorkerConfig {
            mcts: MCTSConfig {
                num_simulations: mcts_num_simulations,
                ..Default::default()
            },
            ..Default::default()
        },
    };

    let batches_dispatched = Arc::new(AtomicUsize::new(0));

    let result = if use_rust_cudagraph {
        let model = model.ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "model is required when use_rust_cudagraph=True",
            )
        })?;
        let total_workers = num_threads.checked_mul(workers_per_thread).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "num_threads * workers_per_thread overflow",
            )
        })?;
        let (num_batches, _total_slots) = queue_shape_for_workers(total_workers);
        let model_ptr = model.bind(py).as_ptr() as usize;
        let runner = {
            let cache = bytefight_graph_cache();
            let mut guard = cache.lock().expect("bytefight graph cache mutex poisoned");

            let needs_rebuild = match guard.as_ref() {
                Some(entry) => {
                    entry.model_ptr != model_ptr
                        || entry.num_batches != num_batches
                        || entry.precision != selfplay_precision
                }
                None => true,
            };

            if needs_rebuild {
                let runner = Arc::new(ByteFightCudaGraphRunner::new(
                    py,
                    model.clone_ref(py),
                    num_batches,
                    BATCH_SIZE,
                    selfplay_precision,
                )?);
                *guard = Some(ByteFightGraphCacheEntry {
                    model_ptr,
                    num_batches,
                    precision: selfplay_precision.to_string(),
                    runner: runner.clone(),
                });
                runner
            } else {
                guard
                    .as_ref()
                    .expect("cached runner should exist")
                    .runner
                    .clone()
            }
        };

        let batches_dispatched_ref = batches_dispatched.clone();
        let dispatch = move |batch_idx: usize,
                             obs_view: ArrayView<u8, Ix3>,
                             outputs: &mut [PolicyValue<11>]| {
            batches_dispatched_ref.fetch_add(1, Ordering::Relaxed);
            runner.dispatch(batch_idx, obs_view, outputs);
        };

        py.detach(|| run_training::<ByteFight, 11, _>(config, replay_buffer.inner(), dispatch))
    } else {
        let execute_model = execute_model.ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "execute_model is required when use_rust_cudagraph=False",
            )
        })?;
        let batches_dispatched_ref = batches_dispatched.clone();
        let dispatch = move |_batch_idx: usize,
                             obs_view: ArrayView<u8, Ix3>,
                             outputs: &mut [PolicyValue<11>]| {
            batches_dispatched_ref.fetch_add(1, Ordering::Relaxed);
            Python::attach(|py| {
                // Zero-copy numpy array from obs_view
                // SAFETY: obs_view is valid for the duration of this callback,
                // and the numpy array doesn't escape the callback scope.
                let np_obs =
                    unsafe { PyArray::borrow_from_array(&obs_view, py.None().into_bound(py)) };

                // Call Python: (BATCH_SIZE, 16, 16) -> ((BATCH_SIZE, 11), (BATCH_SIZE,))
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
                    out.policy.copy_from_slice(&policy[i * 11..(i + 1) * 11]);
                    out.value = value[i];
                }
            });
        };

        // Release GIL while running training, reacquire in dispatch callback.
        py.detach(|| run_training::<ByteFight, 11, _>(config, replay_buffer.inner(), dispatch))
    };

    let stats = PyDict::new(py);
    stats.set_item("poll_rounds", result.executor.poll_rounds)?;
    stats.set_item("futures_polled", result.executor.futures_polled)?;
    stats.set_item("poll_ready", result.executor.poll_ready)?;
    stats.set_item("poll_pending", result.executor.poll_pending)?;
    stats.set_item("wait_count", result.executor.wait_count)?;
    stats.set_item(
        "batches_dispatched",
        batches_dispatched.load(Ordering::Relaxed),
    )?;

    Ok((
        result.games_completed,
        result.samples_collected,
        stats.into(),
    ))
}

#[pymodule]
fn siebren(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TicTacToeEphemeralReplayBuffer>()?;
    m.add_class::<Connect4EphemeralReplayBuffer>()?;
    m.add_class::<ByteFightEphemeralReplayBuffer>()?;
    m.add_function(wrap_pyfunction!(selfplay_tictactoe_ephemeral, m)?)?;
    m.add_function(wrap_pyfunction!(selfplay_connect4_ephemeral, m)?)?;
    m.add_function(wrap_pyfunction!(selfplay_bytefight_ephemeral, m)?)?;
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
