//! Integration tests for the full GPU batching stack.

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::thread;

    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use crate::environments::TicTacToe;
    use crate::eval::{GpuEvaluator, PolicyValue, SyncEvaluator};
    use crate::executor::Executor;
    use crate::mcts::{MCTSConfig, MCTS};
    use crate::queue::{GpuJobQueue, BATCH_SIZE};
    use crate::worker::{worker_loop, WorkerConfig};
    use crate::Environment;

    /// Simple test: multiple futures doing GPU eval on a single thread.
    #[test]
    fn test_simple_multi_future() {
        let dispatch_count = Arc::new(AtomicUsize::new(0));
        let dispatch_count_clone = dispatch_count.clone();

        type Obs = [u8; 9];
        type Output = PolicyValue<9>;
        let queue: Arc<GpuJobQueue<Obs, Output>> = Arc::new(GpuJobQueue::new(
            move |_inputs: &[Obs], outputs: &mut [Output]| {
                dispatch_count_clone.fetch_add(1, Ordering::Relaxed);
                for output in outputs.iter_mut() {
                    output.policy = [1.0 / 9.0; 9];
                    output.value = 0.0;
                }
            },
        ));

        let evaluator = Rc::new(GpuEvaluator::<TicTacToe, 9>::new(&*queue));
        let executor = Executor::new(|| queue.listen());

        // Create BATCH_SIZE futures that each do one eval
        let completed = Rc::new(RefCell::new(0usize));
        let futures: Vec<_> = (0..BATCH_SIZE)
            .map(|_| {
                let completed = completed.clone();
                let evaluator = evaluator.clone();
                let env = TicTacToe::new();
                async move {
                    use crate::eval::Evaluator;
                    let (policy, value) = evaluator.evaluate(&env).await;
                    assert_eq!(policy.len(), 9);
                    assert_eq!(value, 0.0);
                    *completed.borrow_mut() += 1;
                }
            })
            .collect();

        executor.run(
            futures
                .into_iter()
                .map(|f| Box::pin(f) as std::pin::Pin<Box<dyn std::future::Future<Output = ()>>>)
                .collect(),
            || false,
        );

        assert_eq!(*completed.borrow(), BATCH_SIZE);
        assert_eq!(dispatch_count.load(Ordering::Relaxed), 1);
    }

    /// Test multiple batches with simple futures.
    #[test]
    fn test_multiple_batches_simple() {
        let dispatch_count = Arc::new(AtomicUsize::new(0));
        let dispatch_count_clone = dispatch_count.clone();

        type Obs = [u8; 9];
        type Output = PolicyValue<9>;
        let queue: Arc<GpuJobQueue<Obs, Output>> = Arc::new(GpuJobQueue::new(
            move |_inputs: &[Obs], outputs: &mut [Output]| {
                dispatch_count_clone.fetch_add(1, Ordering::Relaxed);
                for output in outputs.iter_mut() {
                    output.policy = [1.0 / 9.0; 9];
                    output.value = 0.0;
                }
            },
        ));

        let evaluator = Rc::new(GpuEvaluator::<TicTacToe, 9>::new(&*queue));
        let executor = Executor::new(|| queue.listen());

        // Create 3 batches worth of futures
        let num_evals = BATCH_SIZE * 3;
        let completed = Rc::new(RefCell::new(0usize));
        let futures: Vec<_> = (0..num_evals)
            .map(|_| {
                let completed = completed.clone();
                let evaluator = evaluator.clone();
                let env = TicTacToe::new();
                async move {
                    use crate::eval::Evaluator;
                    let (_policy, _value) = evaluator.evaluate(&env).await;
                    *completed.borrow_mut() += 1;
                }
            })
            .collect();

        executor.run(
            futures
                .into_iter()
                .map(|f| Box::pin(f) as std::pin::Pin<Box<dyn std::future::Future<Output = ()>>>)
                .collect(),
            || false,
        );

        assert_eq!(*completed.borrow(), num_evals);
        assert_eq!(dispatch_count.load(Ordering::Relaxed), 3);
    }

    /// Test multiple MCTS searches concurrently.
    #[test]
    fn test_multiple_mcts_searches() {
        let dispatch_count = Arc::new(AtomicUsize::new(0));
        let dispatch_count_clone = dispatch_count.clone();

        type Obs = [u8; 9];
        type Output = PolicyValue<9>;
        let queue: Arc<GpuJobQueue<Obs, Output>> = Arc::new(GpuJobQueue::new(
            move |_inputs: &[Obs], outputs: &mut [Output]| {
                dispatch_count_clone.fetch_add(1, Ordering::Relaxed);
                for output in outputs.iter_mut() {
                    output.policy = [1.0 / 9.0; 9];
                    output.value = 0.0;
                }
            },
        ));

        let evaluator = Rc::new(GpuEvaluator::<TicTacToe, 9>::new(&*queue));
        let mcts_config = MCTSConfig {
            num_simulations: 5,
            ..Default::default()
        };
        let executor = Executor::new(|| queue.listen());

        // Run multiple MCTS searches concurrently
        let num_searches = BATCH_SIZE * 2;

        let completed = Rc::new(RefCell::new(0usize));
        let futures: Vec<_> = (0..num_searches)
            .map(|i| {
                let completed = completed.clone();
                let evaluator = evaluator.clone();
                let mcts_config = mcts_config.clone();
                async move {
                    let mcts = MCTS::new(&*evaluator, &mcts_config);
                    let mut env = TicTacToe::new();
                    let mut rng = ChaCha8Rng::seed_from_u64(i as u64);
                    let visits = mcts.search(&mut env, &mut rng).await;
                    assert_eq!(visits.len(), 9);
                    *completed.borrow_mut() += 1;
                }
            })
            .collect();

        executor.run(
            futures
                .into_iter()
                .map(|f| Box::pin(f) as std::pin::Pin<Box<dyn std::future::Future<Output = ()>>>)
                .collect(),
            || false,
        );

        assert_eq!(*completed.borrow(), num_searches);
        let batches = dispatch_count.load(Ordering::Relaxed);
        assert!(batches > 0);
    }

    /// Test worker_loop runs until a global target of games is reached.
    #[test]
    fn test_worker_loop_with_games_per_worker() {
        let evaluator = SyncEvaluator::new(|_env: &TicTacToe| {
            let mut policy = vec![0.0; 9];
            policy[0] = 1.0;
            (policy, 0.0)
        });
        let config = WorkerConfig {
            mcts: MCTSConfig {
                num_simulations: 3,
                ..Default::default()
            },
            ..Default::default()
        };
        let executor = Executor::new(|| event_listener::Event::new().listen());

        let num_workers = 8;
        let target_games = 32;
        let total_games = Arc::new(AtomicUsize::new(0));

        let samples_collected = Rc::new(RefCell::new(0usize));
        let games_completed = Rc::new(RefCell::new(0usize));

        let futures: Vec<_> = (0..num_workers)
            .map(|i| {
                let evaluator = &evaluator;
                let config = config.clone();
                let samples_collected = samples_collected.clone();
                let games_completed = games_completed.clone();
                let total_games = total_games.clone();
                let mut rng = ChaCha8Rng::seed_from_u64(i as u64);
                async move {
                    loop {
                        let idx = total_games.fetch_add(1, Ordering::AcqRel);
                        if idx >= target_games {
                            break;
                        }
                        let samples =
                            worker_loop::<TicTacToe, _, _>(evaluator, &config, &mut rng, 1).await;
                        *samples_collected.borrow_mut() += samples.len();
                        *games_completed.borrow_mut() += 1;
                    }
                }
            })
            .collect();

        executor.run(
            futures
                .into_iter()
                .map(|f| Box::pin(f) as std::pin::Pin<Box<dyn std::future::Future<Output = ()>>>)
                .collect(),
            || false,
        );

        let completed = *games_completed.borrow();
        let samples = *samples_collected.borrow();
        assert_eq!(completed, target_games);
        assert!(samples >= target_games * 5);
    }

    /// Test multithreaded worker_loop with fixed games per worker using the sync evaluator.
    #[test]
    fn test_multithreaded_worker_loop() {
        const NUM_THREADS: usize = 2;
        const WORKERS_PER_THREAD: usize = 4;
        const TARGET_GAMES: usize = 32;

        let total_samples = Arc::new(AtomicUsize::new(0));
        let total_games = Arc::new(AtomicUsize::new(0));

        let handles: Vec<_> = (0..NUM_THREADS)
            .map(|thread_id| {
                let total_games = total_games.clone();
                let total_samples = total_samples.clone();

                thread::spawn(move || {
                    let evaluator = SyncEvaluator::new(|_env: &TicTacToe| {
                        let mut policy = vec![0.0; 9];
                        policy[0] = 1.0;
                        (policy, 0.0)
                    });
                    let config = WorkerConfig {
                        mcts: MCTSConfig {
                            num_simulations: 3,
                            ..Default::default()
                        },
                        ..Default::default()
                    };
                    let executor = Executor::new(|| event_listener::Event::new().listen());

                    let samples_collected = Rc::new(RefCell::new(0usize));
                    let games_collected = Rc::new(RefCell::new(0usize));

                    let futures: Vec<_> = (0..WORKERS_PER_THREAD)
                        .map(|i| {
                            let evaluator = &evaluator;
                            let config = config.clone();
                            let samples_collected = samples_collected.clone();
                            let games_collected = games_collected.clone();
                            let total_games = total_games.clone();
                            let total_samples = total_samples.clone();
                            let mut rng = ChaCha8Rng::seed_from_u64((thread_id * 1000 + i) as u64);
                            async move {
                                loop {
                                    let current = total_games.load(Ordering::Acquire);
                                    if current >= TARGET_GAMES {
                                        break;
                                    }
                                    if total_games
                                        .compare_exchange(
                                            current,
                                            current + 1,
                                            Ordering::AcqRel,
                                            Ordering::Acquire,
                                        )
                                        .is_err()
                                    {
                                        continue;
                                    }

                                    let samples = worker_loop::<TicTacToe, _, _>(
                                        evaluator, &config, &mut rng, 1,
                                    )
                                    .await;
                                    total_samples.fetch_add(samples.len(), Ordering::AcqRel);
                                    *samples_collected.borrow_mut() += samples.len();
                                    *games_collected.borrow_mut() += 1;
                                }
                            }
                        })
                        .collect();

                    executor.run(
                        futures
                            .into_iter()
                            .map(|f| {
                                Box::pin(f)
                                    as std::pin::Pin<Box<dyn std::future::Future<Output = ()>>>
                            })
                            .collect(),
                        || false,
                    );

                    // Return (samples, games) collected by this thread
                    let samples = *samples_collected.borrow();
                    let games = *games_collected.borrow();
                    (samples, games)
                })
            })
            .collect();

        // Collect results from all threads
        for handle in handles {
            let _ = handle.join().expect("thread panicked");
        }

        let completed = total_games.load(Ordering::Relaxed);
        let samples = total_samples.load(Ordering::Relaxed);

        assert_eq!(completed, TARGET_GAMES);
        assert!(samples >= TARGET_GAMES * 5);
    }
}
