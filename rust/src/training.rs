//! Training infrastructure - thread spawning and coordination.
//!
//! This module provides the entry point for running parallel self-play
//! with GPU-batched inference.

use std::sync::Arc;
use std::thread;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::eval::{GpuEvaluator, PolicyValue};
use crate::executor::Executor;
use crate::queue::GpuJobQueue;
use crate::worker::{worker_loop, TrainingSample, WorkerConfig};
use crate::Environment;

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
    /// Number of games each worker should play.
    /// Total games = num_threads * workers_per_thread * games_per_worker.
    pub games_per_worker: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            num_threads: 32,
            workers_per_thread: 16,
            worker: WorkerConfig::default(),
            seed: 42,
            games_per_worker: 20,
        }
    }
}

/// Result of a training run.
pub struct TrainingResult<E: Environment> {
    /// Total number of games completed.
    pub games_completed: usize,
    /// All training samples collected (from all threads).
    pub samples: Vec<TrainingSample<E>>,
}

/// Run self-play training with the given configuration.
///
/// This spawns `num_threads` OS threads, each running `workers_per_thread`
/// async workers. All workers share a single GPU job queue for batched inference.
///
/// The `dispatch` callback is called when a batch of observations is ready
/// for GPU inference. It should fill the output policy/value pairs.
///
/// Each worker plays exactly `games_per_worker` games. This ensures all workers
/// stay alive together and prevents the deadlock where fewer than BATCH_SIZE
/// workers remain active.
///
/// Note: Due to Rust's const generics limitations, `NUM_ACTIONS` must be specified
/// as a const generic parameter rather than inferred from the Environment.
pub fn run_training<E, const NUM_ACTIONS: usize, F>(
    config: TrainingConfig,
    dispatch: F,
) -> TrainingResult<E>
where
    E: Environment + Clone + Send + 'static,
    E::Observation: Copy + Default + Send + Sync,
    F: Fn(&[E::Observation], &mut [PolicyValue<NUM_ACTIONS>]) + Send + Sync + 'static,
{
    let total_workers = config.num_threads * config.workers_per_thread;
    let total_games = total_workers * config.games_per_worker;

    // Create shared GPU queue
    let queue: Arc<GpuJobQueue<E::Observation, PolicyValue<NUM_ACTIONS>>> =
        Arc::new(GpuJobQueue::new(dispatch));

    // Spawn worker threads
    let mut handles = Vec::with_capacity(config.num_threads);

    for thread_id in 0..config.num_threads {
        let queue = queue.clone();
        let config = config.clone();

        let handle = thread::spawn(move || run_thread::<E, NUM_ACTIONS>(thread_id, queue, config));

        handles.push(handle);
    }

    // Collect samples from all threads
    let mut all_samples = Vec::new();
    for handle in handles {
        let thread_samples = handle.join().expect("worker thread panicked");
        all_samples.extend(thread_samples);
    }

    TrainingResult {
        games_completed: total_games,
        samples: all_samples,
    }
}

/// Run a single worker thread with multiple async workers.
/// Returns all samples collected by this thread.
fn run_thread<E, const NUM_ACTIONS: usize>(
    thread_id: usize,
    queue: Arc<GpuJobQueue<E::Observation, PolicyValue<NUM_ACTIONS>>>,
    config: TrainingConfig,
) -> Vec<TrainingSample<E>>
where
    E: Environment + Clone + 'static,
    E::Observation: Copy + Default,
{
    use std::cell::RefCell;
    use std::rc::Rc;

    // Create thread-local RNGs for each worker
    let base_seed = config.seed.wrapping_add(thread_id as u64 * 1000);

    // Create evaluator that uses the shared queue
    let evaluator = Rc::new(GpuEvaluator::<E, NUM_ACTIONS>::new(&*queue));
    let worker_config = Rc::new(config.worker.clone());
    let games_per_worker = config.games_per_worker;

    // Create executor
    let executor = Executor::new(|| queue.listen());

    // Collect samples from all workers
    let all_samples: Rc<RefCell<Vec<TrainingSample<E>>>> = Rc::new(RefCell::new(Vec::new()));

    // Create worker futures
    let futures: Vec<_> = (0..config.workers_per_thread)
        .map(|i| {
            let evaluator = evaluator.clone();
            let worker_config = worker_config.clone();
            let all_samples = all_samples.clone();
            let mut rng = ChaCha8Rng::seed_from_u64(base_seed + i as u64);

            async move {
                let samples = worker_loop::<E, _, _>(
                    &*evaluator,
                    &*worker_config,
                    &mut rng,
                    games_per_worker,
                )
                .await;
                all_samples.borrow_mut().extend(samples);
            }
        })
        .collect();

    // Run all workers until all games are complete
    executor.run(
        futures
            .into_iter()
            .map(|f| Box::pin(f) as std::pin::Pin<Box<dyn std::future::Future<Output = ()>>>)
            .collect(),
        || false,
    );

    // Log completion
    let samples_count = all_samples.borrow().len();
    eprintln!(
        "Thread {}: finished with {} samples collected locally",
        thread_id, samples_count
    );

    // Extract and return samples
    match Rc::try_unwrap(all_samples) {
        Ok(cell) => cell.into_inner(),
        Err(_) => panic!("samples Rc should be unique after executor completes"),
    }
}

#[cfg(test)]
mod tests {
    // Integration tests would go here, but they require a lot of setup
    // and would take too long to run in unit test context.
    // See the examples/ directory for full integration tests.
}
