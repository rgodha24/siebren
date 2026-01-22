//! Training infrastructure - thread spawning and coordination.
//!
//! This module provides the entry point for running parallel self-play
//! with GPU-batched inference.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;

use ndarray::ArrayView;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::eval::{GpuEvaluator, PolicyValue};
use crate::executor::Executor;
use crate::queue::GpuJobQueue;
use crate::replay_buffer::ReplayBuffer;
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
}

/// Run self-play training with the given configuration.
///
/// This spawns `num_threads` OS threads, each running `workers_per_thread`
/// async workers. All workers share a single GPU job queue for batched inference.
///
/// The `dispatch` callback is called when a batch of observations is ready
/// for GPU inference. It receives a zero-copy view of the batched observations
/// and should fill the output policy/value pairs.
///
/// Workers share an atomic counter for samples collected. When the target is
/// reached, remaining workers are cancelled via the executor. This allows
/// fast workers to complete more games while slow workers are still playing.
///
/// Samples are pushed directly to the shared `replay_buffer` after each game.
pub fn run_training<E, const NUM_ACTIONS: usize, F>(
    config: TrainingConfig,
    replay_buffer: &ReplayBuffer,
    dispatch: F,
) -> TrainingResult
where
    E: Environment + Clone + Send + 'static,
    E::ObsDim: BatchDim,
    F: Fn(
            ArrayView<E::ObsElem, <E::ObsDim as BatchDim>::BatchedDim>,
            &mut [PolicyValue<NUM_ACTIONS>],
        ) + Send
        + Sync
        + 'static,
{
    let target_samples = config.target_samples;

    // Shared counters for samples collected and games completed across all workers
    let samples_collected = Arc::new(AtomicUsize::new(0));
    let games_completed = Arc::new(AtomicUsize::new(0));

    // Create shared GPU queue with observation shape
    let queue: Arc<GpuJobQueue<E::ObsElem, E::ObsDim, PolicyValue<NUM_ACTIONS>>> =
        Arc::new(GpuJobQueue::new(E::OBS_SHAPE, dispatch));

    // Use scoped threads to allow borrowing replay_buffer across threads
    thread::scope(|s| {
        for thread_id in 0..config.num_threads {
            let queue = queue.clone();
            let config = config.clone();
            let samples_collected = samples_collected.clone();
            let games_completed = games_completed.clone();

            s.spawn(move || {
                run_thread::<E, NUM_ACTIONS>(
                    thread_id,
                    queue,
                    config,
                    samples_collected,
                    games_completed,
                    target_samples,
                    replay_buffer,
                )
            });
        }
    });

    let final_samples = samples_collected.load(Ordering::Acquire);
    let final_games = games_completed.load(Ordering::Acquire);
    TrainingResult {
        games_completed: final_games,
        samples_collected: final_samples,
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
    replay_buffer: &ReplayBuffer,
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
                worker_loop::<E, _, _>(
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

    eprintln!(
        "Thread {thread_id}: finished, total samples so far: {}",
        samples_collected.load(Ordering::Acquire)
    );
}
