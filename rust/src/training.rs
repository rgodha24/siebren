//! Training infrastructure - thread spawning and coordination.
//!
//! This module provides the entry point for running parallel self-play
//! with GPU-batched inference.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;

use ndarray::ArrayView;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::eval::{GpuEvaluator, PolicyValue};
use crate::executor::{
    reset_executor_counters, take_executor_counters, Executor, ExecutorCounters,
};
use crate::observation_replay_buffer::ObservationReplayBuffer;
use crate::queue::{BatchCompletion, GpuJobQueue};
use crate::worker::{worker_loop, WorkerConfig};
use crate::{BatchDim, Environment};

/// Configuration for the training run.
#[derive(Clone)]
pub struct TrainingConfig {
    /// Number of OS threads to spawn.
    pub num_threads: usize,
    /// Number of workers per thread.
    pub workers_per_thread: usize,
    /// Worker configuration (MCTS params, temperature, etc).
    pub worker: WorkerConfig,
    /// Random seed for reproducibility.
    pub seed: u64,
    /// Target number of samples to collect across all workers.
    pub target_samples: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            num_threads: 32,
            workers_per_thread: 16,
            worker: WorkerConfig::default(),
            seed: 42,
            target_samples: 102400,
        }
    }
}

/// Result of a training run.
pub struct TrainingResult {
    /// Total number of games completed.
    pub games_completed: usize,
    /// Total number of samples collected.
    pub samples_collected: usize,
    /// Aggregated executor counters across worker threads.
    pub executor: ExecutorCounters,
}

#[derive(Default)]
struct ExecutorCountersAtomic {
    poll_rounds: AtomicU64,
    futures_polled: AtomicU64,
    poll_ready: AtomicU64,
    poll_pending: AtomicU64,
    wait_count: AtomicU64,
}

impl ExecutorCountersAtomic {
    fn add(&self, counters: ExecutorCounters) {
        self.poll_rounds
            .fetch_add(counters.poll_rounds, Ordering::Relaxed);
        self.futures_polled
            .fetch_add(counters.futures_polled, Ordering::Relaxed);
        self.poll_ready
            .fetch_add(counters.poll_ready, Ordering::Relaxed);
        self.poll_pending
            .fetch_add(counters.poll_pending, Ordering::Relaxed);
        self.wait_count
            .fetch_add(counters.wait_count, Ordering::Relaxed);
    }

    fn snapshot(&self) -> ExecutorCounters {
        ExecutorCounters {
            poll_rounds: self.poll_rounds.load(Ordering::Relaxed),
            futures_polled: self.futures_polled.load(Ordering::Relaxed),
            poll_ready: self.poll_ready.load(Ordering::Relaxed),
            poll_pending: self.poll_pending.load(Ordering::Relaxed),
            wait_count: self.wait_count.load(Ordering::Relaxed),
        }
    }
}

/// Run self-play training with the given configuration.
///
/// This spawns `num_threads` OS threads, each running `workers_per_thread`
/// async workers. All workers share a single GPU job queue for batched inference.
///
/// The `dispatch` callback is called when a batch of observations is ready
/// for GPU inference. It receives the queue batch slot index, a zero-copy view
/// of the batched observations, and a completion handle that must be completed
/// with policy/value outputs.
///
/// Workers share an atomic counter for samples collected. When the target is
/// reached, remaining workers are cancelled via the executor. This allows
/// fast workers to complete more games while slow workers are still playing.
///
/// Samples are pushed directly to the shared `replay_buffer` after each game.
pub fn run_training<E, const NUM_ACTIONS: usize, F>(
    config: TrainingConfig,
    replay_buffer: &ObservationReplayBuffer<E::ObsElem, E::ObsDim, NUM_ACTIONS>,
    dispatch: F,
) -> TrainingResult
where
    E: Environment + Clone + Send + 'static,
    E::ObsDim: BatchDim,
    F: Fn(
            usize,
            ArrayView<E::ObsElem, <E::ObsDim as BatchDim>::BatchedDim>,
            BatchCompletion<PolicyValue<NUM_ACTIONS>>,
        ) + Send
        + Sync
        + 'static,
{
    let target_samples = config.target_samples;

    // Shared counters for samples collected and games completed across all workers
    let samples_collected = Arc::new(AtomicUsize::new(0));
    let games_completed = Arc::new(AtomicUsize::new(0));
    let executor_counters = Arc::new(ExecutorCountersAtomic::default());

    // Create shared GPU queue with observation shape
    let total_workers = config
        .num_threads
        .checked_mul(config.workers_per_thread)
        .expect("num_threads * workers_per_thread overflowed usize");

    let queue: Arc<GpuJobQueue<E::ObsElem, E::ObsDim, PolicyValue<NUM_ACTIONS>>> =
        Arc::new(GpuJobQueue::new(E::OBS_SHAPE, total_workers, dispatch));

    // Use scoped threads to allow borrowing replay_buffer across threads
    thread::scope(|s| {
        for thread_id in 0..config.num_threads {
            let queue = queue.clone();
            let config = config.clone();
            let samples_collected = samples_collected.clone();
            let games_completed = games_completed.clone();
            let executor_counters = executor_counters.clone();

            s.spawn(move || {
                run_thread::<E, NUM_ACTIONS>(
                    thread_id,
                    queue,
                    config,
                    samples_collected,
                    games_completed,
                    target_samples,
                    executor_counters,
                    replay_buffer,
                )
            });
        }
    });

    let final_samples = samples_collected.load(Ordering::Acquire);
    let final_games = games_completed.load(Ordering::Acquire);
    let executor = executor_counters.snapshot();
    TrainingResult {
        games_completed: final_games,
        samples_collected: final_samples,
        executor,
    }
}

/// Run a single worker thread with multiple async workers.
fn run_thread<E, const NUM_ACTIONS: usize>(
    thread_id: usize,
    queue: Arc<GpuJobQueue<E::ObsElem, E::ObsDim, PolicyValue<NUM_ACTIONS>>>,
    config: TrainingConfig,
    samples_collected: Arc<AtomicUsize>,
    games_completed: Arc<AtomicUsize>,
    target_samples: usize,
    executor_counters: Arc<ExecutorCountersAtomic>,
    replay_buffer: &ObservationReplayBuffer<E::ObsElem, E::ObsDim, NUM_ACTIONS>,
) where
    E: Environment + Clone + 'static,
    E::ObsDim: BatchDim,
{
    let base_seed = config.seed.wrapping_add(thread_id as u64 * 1000);
    let evaluator = GpuEvaluator::<E, NUM_ACTIONS>::new(&*queue);

    let futures: Vec<_> = (0..config.workers_per_thread)
        .map(|i| {
            let samples_collected = samples_collected.clone();
            let games_completed = games_completed.clone();
            let mut rng = ChaCha8Rng::seed_from_u64(base_seed + i as u64);
            let evaluator_ref = &evaluator;
            let worker_config = &config.worker;

            async move {
                worker_loop::<E, _, _, NUM_ACTIONS>(
                    evaluator_ref,
                    worker_config,
                    &mut rng,
                    samples_collected,
                    games_completed,
                    target_samples,
                    replay_buffer,
                )
                .await;
            }
        })
        .collect();

    let executor = Executor::new(|| queue.listen());
    reset_executor_counters();
    executor.run(
        futures
            .into_iter()
            .map(|f| Box::pin(f) as std::pin::Pin<Box<dyn std::future::Future<Output = ()>>>)
            .collect(),
        || {
            let done = samples_collected.load(Ordering::Acquire) >= target_samples;
            if done {
                queue.notify_all();
            }
            done
        },
    );

    let counters = take_executor_counters();
    executor_counters.add(counters);
}
