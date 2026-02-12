use std::sync::Arc;
use std::sync::{Mutex, OnceLock};
use std::{fmt::Debug, hash::Hash};

use ndarray::{ArrayView, ArrayViewMut, Dimension, Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, RemoveAxis};
use numpy::{PyArray, PyArrayMethods};
use pyo3::prelude::*;
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

use environments::{bytefight::types as bytefight_types, ByteFight, Connect4, TicTacToe};
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

// ByteFight: observations (18, 16) u8, sampled as (n, 18, 16)
typed_ephemeral_replay_buffer!(
    ByteFightEphemeralReplayBuffer,
    u8,
    Ix2,
    Ix3,
    ByteFight::OBS_SHAPE,
    |n| Ix3(
        n,
        bytefight_types::OBS_SERIALIZED_SIDE,
        bytefight_types::OBS_SERIALIZED_WIDTH,
    ),
    7
);

/// Macro to generate a persistent SelfPlay pyclass with a Python callback dispatch.
macro_rules! typed_selfplay {
    (
        $name:ident,
        $env:ty,
        $obs_ty:ty,
        $obs_dim:ty,
        $obs_batched_dim:ty,
        $replay_buf_class:ident,
        $num_actions:expr,
        callback_dispatch
    ) => {
        #[doc = concat!("Persistent self-play session for ", stringify!($name), ".")]
        #[pyclass]
        struct $name {
            session: Option<training::SelfPlaySession>,
        }

        #[pymethods]
        impl $name {
            #[new]
            #[rustfmt::skip]
            #[pyo3(signature = (
                replay_buffer,
                num_threads,
                workers_per_thread,
                seed,
                execute_model,
                mcts_num_simulations = 20,
                mcts_c_puct = 1.5,
                mcts_dirichlet_alpha = 0.3,
                mcts_dirichlet_epsilon = 0.25,
                temperature = 1.0,
                exploration_moves = 30,
            ))]
            fn new(
                replay_buffer: &$replay_buf_class,
                num_threads: usize,
                workers_per_thread: usize,
                seed: u64,
                execute_model: Py<PyAny>,
                mcts_num_simulations: usize,
                mcts_c_puct: f32,
                mcts_dirichlet_alpha: f32,
                mcts_dirichlet_epsilon: f32,
                temperature: f32,
                exploration_moves: usize,
            ) -> PyResult<Self> {
                use eval::PolicyValue;
                use mcts::MCTSConfig;
                use training::{SelfPlaySession, SessionConfig};
                use worker::WorkerConfig;

                if mcts_num_simulations == 0 {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "mcts_num_simulations must be >= 1",
                    ));
                }
                if mcts_c_puct <= 0.0 {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "mcts_c_puct must be > 0",
                    ));
                }
                if mcts_dirichlet_alpha <= 0.0 {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "mcts_dirichlet_alpha must be > 0",
                    ));
                }
                if !(0.0..=1.0).contains(&mcts_dirichlet_epsilon) {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "mcts_dirichlet_epsilon must be in [0, 1]",
                    ));
                }
                if temperature < 0.0 {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "temperature must be >= 0",
                    ));
                }

                let config = SessionConfig {
                    num_threads,
                    workers_per_thread,
                    seed,
                    worker: WorkerConfig {
                        mcts: MCTSConfig {
                            num_simulations: mcts_num_simulations,
                            c_puct: mcts_c_puct,
                            dirichlet_alpha: mcts_dirichlet_alpha,
                            dirichlet_epsilon: mcts_dirichlet_epsilon,
                            ..Default::default()
                        },
                        temperature,
                        exploration_moves,
                        ..Default::default()
                    },
                };

                let dispatch = move |_batch_idx: usize,
                                     obs_view: ArrayView<$obs_ty, $obs_batched_dim>,
                                     completion: queue::BatchCompletion<
                    PolicyValue<$num_actions>,
                >| {
                    let mut outputs =
                        vec![PolicyValue::<$num_actions>::default(); queue::BATCH_SIZE];
                    Python::attach(|py| {
                        // SAFETY: obs_view is valid for the duration of this callback,
                        // and the numpy array doesn't escape the callback scope.
                        let np_obs = unsafe {
                            PyArray::borrow_from_array(&obs_view, py.None().into_bound(py))
                        };

                        let result = execute_model
                            .call1(py, (np_obs,))
                            .expect("execute_model call failed");

                        let (policy_arr, value_arr): (
                            Bound<'_, PyArray<f32, Ix2>>,
                            Bound<'_, PyArray<f32, Ix1>>,
                        ) = result
                            .extract(py)
                            .expect("expected (policy, value) tuple of numpy arrays");

                        let policy = unsafe { policy_arr.as_slice().unwrap() };
                        let value = unsafe { value_arr.as_slice().unwrap() };

                        for (i, out) in outputs.iter_mut().enumerate() {
                            out.policy
                                .copy_from_slice(&policy[i * $num_actions..(i + 1) * $num_actions]);
                            out.value = value[i];
                        }
                    });
                    completion.complete(&outputs);
                };

                let session = SelfPlaySession::new::<$env, $num_actions, _>(
                    config,
                    replay_buffer.inner().clone(),
                    dispatch,
                );

                Ok(Self {
                    session: Some(session),
                })
            }

            /// Start self-play with no sample limit.
            fn start(&self) -> PyResult<()> {
                self.session
                    .as_ref()
                    .ok_or_else(|| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("session already dropped")
                    })?
                    .start();
                Ok(())
            }

            /// Block until absolute target_samples is reached, then pause and quiesce.
            fn wait_for(&self, py: Python<'_>, target_samples: usize) -> PyResult<usize> {
                let session = self.session.as_ref().ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("session already dropped")
                })?;
                let result = py.detach(|| session.wait_for(target_samples));
                Ok(result)
            }

            /// Return the current absolute sample count.
            fn samples(&self) -> PyResult<usize> {
                Ok(self
                    .session
                    .as_ref()
                    .ok_or_else(|| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("session already dropped")
                    })?
                    .samples())
            }

            /// Shut down the session. Idempotent.
            #[pyo3(name = "drop")]
            fn py_drop(&mut self) {
                if let Some(mut session) = self.session.take() {
                    session.shutdown();
                }
            }
        }

        impl Drop for $name {
            fn drop(&mut self) {
                if let Some(mut session) = self.session.take() {
                    session.shutdown();
                }
            }
        }
    };
}

// TicTacToe self-play: callback-based dispatch
typed_selfplay!(
    TicTacToeSelfPlay,
    TicTacToe,
    i8,
    Ix1,
    Ix2,
    TicTacToeEphemeralReplayBuffer,
    9,
    callback_dispatch
);

// Connect4 self-play: callback-based dispatch
typed_selfplay!(
    Connect4SelfPlay,
    Connect4,
    i8,
    Ix2,
    Ix3,
    Connect4EphemeralReplayBuffer,
    7,
    callback_dispatch
);

/// Persistent ByteFight self-play session.
///
/// Uses the CUDA graph runner for GPU dispatch (no Python callback).
#[pyclass]
struct ByteFightSelfPlay {
    session: Option<training::SelfPlaySession>,
}

#[pymethods]
impl ByteFightSelfPlay {
    #[new]
    #[pyo3(signature = (
        replay_buffer,
        num_threads,
        workers_per_thread,
        seed,
        *,
        mcts_num_simulations = 20,
        mcts_c_puct = 1.5,
        mcts_dirichlet_alpha = 0.3,
        mcts_dirichlet_epsilon = 0.25,
        temperature = 1.0,
        exploration_moves = 30,
        model,
        selfplay_precision = "fp32"
    ))]
    fn new(
        py: Python<'_>,
        replay_buffer: &ByteFightEphemeralReplayBuffer,
        num_threads: usize,
        workers_per_thread: usize,
        seed: u64,
        mcts_num_simulations: usize,
        mcts_c_puct: f32,
        mcts_dirichlet_alpha: f32,
        mcts_dirichlet_epsilon: f32,
        temperature: f32,
        exploration_moves: usize,
        model: Py<PyAny>,
        selfplay_precision: &str,
    ) -> PyResult<Self> {
        use cudagraph::ByteFightCudaGraphRunner;
        use eval::PolicyValue;
        use mcts::MCTSConfig;
        use queue::{queue_shape_for_workers, BATCH_SIZE};
        use training::{SelfPlaySession, SessionConfig};
        use worker::WorkerConfig;

        if mcts_num_simulations == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "mcts_num_simulations must be >= 1",
            ));
        }
        if mcts_c_puct <= 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "mcts_c_puct must be > 0",
            ));
        }
        if mcts_dirichlet_alpha <= 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "mcts_dirichlet_alpha must be > 0",
            ));
        }
        if !(0.0..=1.0).contains(&mcts_dirichlet_epsilon) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "mcts_dirichlet_epsilon must be in [0, 1]",
            ));
        }
        if temperature < 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "temperature must be >= 0",
            ));
        }

        let config = SessionConfig {
            num_threads,
            workers_per_thread,
            seed,
            worker: WorkerConfig {
                mcts: MCTSConfig {
                    num_simulations: mcts_num_simulations,
                    c_puct: mcts_c_puct,
                    dirichlet_alpha: mcts_dirichlet_alpha,
                    dirichlet_epsilon: mcts_dirichlet_epsilon,
                    ..Default::default()
                },
                temperature,
                exploration_moves,
                ..Default::default()
            },
        };

        let total_workers = num_threads.checked_mul(workers_per_thread).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "num_threads * workers_per_thread overflow",
            )
        })?;
        let (num_batches, _total_slots) = queue_shape_for_workers(total_workers);
        let model_ptr = model.bind(py).as_ptr() as usize;

        // Build or reuse the CUDA graph runner.
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

        let dispatch =
            move |batch_idx: usize,
                  obs_view: ArrayView<u8, Ix3>,
                  completion: queue::BatchCompletion<PolicyValue<7>>| {
                runner.dispatch_async(batch_idx, obs_view, completion);
            };

        let session = SelfPlaySession::new::<ByteFight, 7, _>(
            config,
            replay_buffer.inner().clone(),
            dispatch,
        );

        Ok(Self {
            session: Some(session),
        })
    }

    /// Start self-play with no sample limit.
    fn start(&self) -> PyResult<()> {
        self.session
            .as_ref()
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("session already dropped")
            })?
            .start();
        Ok(())
    }

    /// Block until absolute target_samples is reached, then pause and quiesce.
    fn wait_for(&self, py: Python<'_>, target_samples: usize) -> PyResult<usize> {
        let session = self.session.as_ref().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("session already dropped")
        })?;
        let result = py.detach(|| session.wait_for(target_samples));
        Ok(result)
    }

    /// Return the current absolute sample count.
    fn samples(&self) -> PyResult<usize> {
        Ok(self
            .session
            .as_ref()
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("session already dropped")
            })?
            .samples())
    }

    /// Shut down the session. Idempotent.
    #[pyo3(name = "drop")]
    fn py_drop(&mut self) {
        if let Some(mut session) = self.session.take() {
            session.shutdown();
        }
    }
}

impl Drop for ByteFightSelfPlay {
    fn drop(&mut self) {
        if let Some(mut session) = self.session.take() {
            session.shutdown();
        }
    }
}

#[pymodule]
fn siebren(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TicTacToeEphemeralReplayBuffer>()?;
    m.add_class::<Connect4EphemeralReplayBuffer>()?;
    m.add_class::<ByteFightEphemeralReplayBuffer>()?;
    m.add_class::<TicTacToeSelfPlay>()?;
    m.add_class::<Connect4SelfPlay>()?;
    m.add_class::<ByteFightSelfPlay>()?;
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
