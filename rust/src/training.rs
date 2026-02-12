//! Training infrastructure - thread spawning and coordination.
//!
//! Provides `SelfPlaySession`: a persistent session with pause/resume semantics
//! that preserves in-progress game state across boundaries.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

use ndarray::ArrayView;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::eval::{GpuEvaluator, PolicyValue};
use crate::executor::Executor;
use crate::observation_replay_buffer::ObservationReplayBuffer;
use crate::queue::{BatchCompletion, GpuJobQueue};
use crate::worker::{worker_loop_forever, WorkerConfig};
use crate::{BatchDim, Environment};

/// Shared control state for the persistent self-play session.
///
/// Counters are behind `Arc<AtomicUsize>` so they can be cloned directly
/// into worker futures that expect `Arc<AtomicUsize>`.
struct SessionControl {
    /// Whether workers should be actively polling.
    running: AtomicBool,
    /// Whether the session is being torn down.
    shutdown: AtomicBool,
    /// Absolute target: workers pause when `samples_collected >= target`.
    target_samples: AtomicUsize,
    /// Total samples collected (monotonic across the session lifetime).
    samples_collected: Arc<AtomicUsize>,
    /// Total games completed.
    games_completed: Arc<AtomicUsize>,
    /// Number of threads currently inside the executor polling loop.
    active_pollers: AtomicUsize,
    /// Condvar + mutex for coordinating start/pause/quiesce/shutdown.
    condvar: Condvar,
    condvar_mutex: Mutex<()>,
}

impl SessionControl {
    fn new() -> Self {
        Self {
            running: AtomicBool::new(false),
            shutdown: AtomicBool::new(false),
            target_samples: AtomicUsize::new(0),
            samples_collected: Arc::new(AtomicUsize::new(0)),
            games_completed: Arc::new(AtomicUsize::new(0)),
            active_pollers: AtomicUsize::new(0),
            condvar: Condvar::new(),
            condvar_mutex: Mutex::new(()),
        }
    }

    /// Wake all threads blocked on the condvar.
    fn wake_all(&self) {
        self.condvar.notify_all();
    }

    /// Check if the thread should stop polling (pause or shutdown).
    fn should_pause(&self) -> bool {
        if self.shutdown.load(Ordering::Acquire) {
            return true;
        }
        if !self.running.load(Ordering::Acquire) {
            return true;
        }
        self.samples_collected.load(Ordering::Acquire)
            >= self.target_samples.load(Ordering::Acquire)
    }
}

/// Configuration for creating a persistent self-play session.
#[derive(Clone)]
pub struct SessionConfig {
    /// Number of OS threads to spawn.
    pub num_threads: usize,
    /// Number of workers per thread.
    pub workers_per_thread: usize,
    /// Worker configuration (MCTS params, temperature, etc).
    pub worker: WorkerConfig,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            num_threads: 32,
            workers_per_thread: 16,
            worker: WorkerConfig::default(),
            seed: 42,
        }
    }
}

/// Trait-object wrapper so we can call `notify_all()` on the queue without
/// leaking the full generic type into `SelfPlaySession`.
trait QueueNotify: Send + Sync {
    fn notify_all(&self);
}

impl<A, D, O> QueueNotify for GpuJobQueue<A, D, O>
where
    A: Clone + Default + Send + Sync,
    D: BatchDim,
    O: Copy + Default + Send + Sync,
{
    fn notify_all(&self) {
        GpuJobQueue::notify_all(self);
    }
}

/// A persistent self-play session that owns worker threads and preserves
/// in-progress game state across pause/resume boundaries.
///
/// # Lifecycle
///
/// 1. `new(...)` — creates threads and futures (paused).
/// 2. `start()` — sets target to `usize::MAX` and wakes threads.
/// 3. `wait_for(target)` — ensures running, blocks until `samples >= target`,
///    then pauses and waits for all pollers to quiesce. Safe to read replay
///    buffer after this returns.
/// 4. `samples()` — returns current absolute sample count.
/// 5. `shutdown()` / Rust `Drop` — sets shutdown, joins threads.
pub struct SelfPlaySession {
    control: Arc<SessionControl>,
    /// Queue used by all workers. Kept alive for `notify_all` on drop.
    queue_notify: Arc<dyn QueueNotify>,
    /// Join handles for worker threads. `None` after `shutdown`.
    threads: Option<Vec<thread::JoinHandle<()>>>,
}

impl SelfPlaySession {
    /// Create a new persistent session.
    ///
    /// Threads are spawned immediately but start paused. The `dispatch` callback
    /// is invoked when a batch of observations is ready for GPU inference.
    pub fn new<E, const NUM_ACTIONS: usize, F>(
        config: SessionConfig,
        replay_buffer: Arc<ObservationReplayBuffer<E::ObsElem, E::ObsDim, NUM_ACTIONS>>,
        dispatch: F,
    ) -> Self
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
        let total_workers = config
            .num_threads
            .checked_mul(config.workers_per_thread)
            .expect("num_threads * workers_per_thread overflowed usize");

        let queue: Arc<GpuJobQueue<E::ObsElem, E::ObsDim, PolicyValue<NUM_ACTIONS>>> =
            Arc::new(GpuJobQueue::new(E::OBS_SHAPE, total_workers, dispatch));

        let control = Arc::new(SessionControl::new());

        let mut threads = Vec::with_capacity(config.num_threads);
        for thread_id in 0..config.num_threads {
            let queue = queue.clone();
            let control = control.clone();
            let config = config.clone();
            let replay_buffer = replay_buffer.clone();

            let handle = thread::spawn(move || {
                session_thread_main::<E, NUM_ACTIONS>(
                    thread_id,
                    queue,
                    config,
                    control,
                    &replay_buffer,
                );
            });
            threads.push(handle);
        }

        Self {
            control,
            queue_notify: queue,
            threads: Some(threads),
        }
    }

    /// Start self-play with no sample limit (runs until explicitly paused or
    /// `wait_for` is called).
    pub fn start(&self) {
        self.control
            .target_samples
            .store(usize::MAX, Ordering::Release);
        self.control.running.store(true, Ordering::Release);
        self.control.wake_all();
        self.queue_notify.notify_all();
    }

    /// Block until at least `target_samples` absolute samples have been
    /// collected, then pause and quiesce all workers.
    ///
    /// Returns the actual number of samples collected (may exceed target).
    ///
    /// After this returns, no worker thread is inside the executor polling
    /// loop, so it is safe to read the replay buffer.
    pub fn wait_for(&self, target_samples: usize) -> usize {
        // Set target and ensure running.
        self.control
            .target_samples
            .store(target_samples, Ordering::Release);
        self.control.running.store(true, Ordering::Release);
        self.control.wake_all();
        self.queue_notify.notify_all();

        // Wait until target is reached (condvar-based, no spinning).
        {
            let mut guard = self
                .control
                .condvar_mutex
                .lock()
                .expect("condvar mutex poisoned");
            while self.control.samples_collected.load(Ordering::Acquire) < target_samples
                && !self.control.shutdown.load(Ordering::Acquire)
            {
                guard = self
                    .control
                    .condvar
                    .wait(guard)
                    .expect("condvar wait failed");
            }
        }

        // Pause workers.
        self.control.running.store(false, Ordering::Release);
        self.queue_notify.notify_all();
        self.control.wake_all();

        // Wait for all pollers to exit (quiesce).
        {
            let mut guard = self
                .control
                .condvar_mutex
                .lock()
                .expect("condvar mutex poisoned");
            while self.control.active_pollers.load(Ordering::Acquire) > 0
                && !self.control.shutdown.load(Ordering::Acquire)
            {
                guard = self
                    .control
                    .condvar
                    .wait(guard)
                    .expect("condvar wait failed");
            }
        }

        self.control.samples_collected.load(Ordering::Acquire)
    }

    /// Return the current absolute sample count.
    pub fn samples(&self) -> usize {
        self.control.samples_collected.load(Ordering::Acquire)
    }

    /// Return the current absolute game count.
    pub fn games(&self) -> usize {
        self.control.games_completed.load(Ordering::Acquire)
    }

    /// Shut down the session. Idempotent.
    pub fn shutdown(&mut self) {
        if let Some(threads) = self.threads.take() {
            self.control.shutdown.store(true, Ordering::Release);
            self.control.running.store(false, Ordering::Release);
            self.control.wake_all();
            self.queue_notify.notify_all();

            for handle in threads {
                let _ = handle.join();
            }
        }
    }
}

impl Drop for SelfPlaySession {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Main loop for a single thread in a persistent session.
///
/// 1. Wait on condvar until `running || shutdown`.
/// 2. If shutdown => exit.
/// 3. Increment `active_pollers`, run executor until pause/shutdown/target.
/// 4. Decrement `active_pollers`, notify condvar so `wait_for()` can observe
///    quiesce.
/// 5. Goto 1.
fn session_thread_main<E, const NUM_ACTIONS: usize>(
    thread_id: usize,
    queue: Arc<GpuJobQueue<E::ObsElem, E::ObsDim, PolicyValue<NUM_ACTIONS>>>,
    config: SessionConfig,
    control: Arc<SessionControl>,
    replay_buffer: &ObservationReplayBuffer<E::ObsElem, E::ObsDim, NUM_ACTIONS>,
) where
    E: Environment + Clone + 'static,
    E::ObsDim: BatchDim,
{
    let base_seed = config.seed.wrapping_add(thread_id as u64 * 1000);
    let evaluator = GpuEvaluator::<E, NUM_ACTIONS>::new(&*queue);

    // Clone the session's counters for workers.
    let samples_collected = control.samples_collected.clone();
    let games_completed = control.games_completed.clone();

    // Create futures once. They live for the entire session.
    let mut futures: Vec<std::pin::Pin<Box<dyn std::future::Future<Output = ()> + '_>>> = (0
        ..config.workers_per_thread)
        .map(|i| {
            let samples_collected = samples_collected.clone();
            let games_completed = games_completed.clone();
            let mut rng = ChaCha8Rng::seed_from_u64(base_seed + i as u64);
            let evaluator_ref = &evaluator;
            let worker_config = &config.worker;

            let fut = async move {
                worker_loop_forever::<E, _, _, NUM_ACTIONS>(
                    evaluator_ref,
                    worker_config,
                    &mut rng,
                    samples_collected,
                    games_completed,
                    replay_buffer,
                )
                .await;
            };
            Box::pin(fut) as std::pin::Pin<Box<dyn std::future::Future<Output = ()> + '_>>
        })
        .collect();

    let executor = Executor::new(|| queue.listen());

    loop {
        // 1. Wait until running or shutdown.
        {
            let mut guard = control
                .condvar_mutex
                .lock()
                .expect("condvar mutex poisoned");
            while !control.running.load(Ordering::Acquire)
                && !control.shutdown.load(Ordering::Acquire)
            {
                guard = control.condvar.wait(guard).expect("condvar wait failed");
            }
        }

        // 2. If shutdown, exit.
        if control.shutdown.load(Ordering::Acquire) {
            return;
        }

        // 3. Increment active_pollers and run executor.
        control.active_pollers.fetch_add(1, Ordering::AcqRel);

        let control_ref = &control;
        let queue_ref = &queue;
        executor.run(&mut futures, &mut || {
            let should_pause = control_ref.should_pause();
            if should_pause {
                queue_ref.notify_all();
            }
            // Notify the condvar when samples cross the target so wait_for()
            // wakes up.
            if control_ref.samples_collected.load(Ordering::Acquire)
                >= control_ref.target_samples.load(Ordering::Acquire)
            {
                control_ref.wake_all();
            }
            should_pause
        });

        // 4. Decrement active_pollers and notify.
        control.active_pollers.fetch_sub(1, Ordering::AcqRel);
        control.wake_all();
    }
}
