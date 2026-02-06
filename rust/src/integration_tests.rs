//! Integration tests for the full GPU batching stack.

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::thread;

    use ndarray::Ix1;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use crate::environments::TicTacToe;
    use crate::eval::{GpuEvaluator, PolicyValue, SyncEvaluator};
    use crate::executor::Executor;
    use crate::mcts::{MCTSConfig, MCTS};
    use crate::observation_replay_buffer::ObservationReplayBuffer;
    use crate::queue::{GpuJobQueue, BATCH_SIZE};
    use crate::worker::{worker_loop, WorkerConfig};
    use crate::Environment;

    /// Simple test: multiple futures doing GPU eval on a single thread.
    #[test]
    fn test_simple_multi_future() {
        let dispatch_count = Arc::new(AtomicUsize::new(0));
        let dispatch_count_clone = dispatch_count.clone();

        type Output = PolicyValue<9>;
        let queue: Arc<GpuJobQueue<i8, Ix1, Output>> = Arc::new(GpuJobQueue::new(
            TicTacToe::OBS_SHAPE,
            BATCH_SIZE,
            move |_inputs, outputs: &mut [Output]| {
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
        let num_evals = BATCH_SIZE * 3;

        type Output = PolicyValue<9>;
        let queue: Arc<GpuJobQueue<i8, Ix1, Output>> = Arc::new(GpuJobQueue::new(
            TicTacToe::OBS_SHAPE,
            num_evals,
            move |_inputs, outputs: &mut [Output]| {
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
        let num_searches = BATCH_SIZE * 2;

        type Output = PolicyValue<9>;
        let queue: Arc<GpuJobQueue<i8, Ix1, Output>> = Arc::new(GpuJobQueue::new(
            TicTacToe::OBS_SHAPE,
            num_searches,
            move |_inputs, outputs: &mut [Output]| {
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

    /// Test worker_loop runs until a global target of samples is reached.
    #[test]
    fn test_worker_loop_with_shared_counter() {
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
        let target_samples = 200; // ~32 games * 6 samples/game
        let samples_collected = Arc::new(AtomicUsize::new(0));
        let games_completed = Arc::new(AtomicUsize::new(0));
        let replay_buffer = ObservationReplayBuffer::<i8, Ix1, 9>::new(1000, TicTacToe::OBS_SHAPE);

        let futures: Vec<_> = (0..num_workers)
            .map(|i| {
                let evaluator = &evaluator;
                let config = &config;
                let replay_buffer = &replay_buffer;
                let samples_collected = samples_collected.clone();
                let games_completed = games_completed.clone();
                let mut rng = ChaCha8Rng::seed_from_u64(i as u64);
                async move {
                    worker_loop::<TicTacToe, _, _, 9>(
                        evaluator,
                        config,
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

        executor.run(
            futures
                .into_iter()
                .map(|f| Box::pin(f) as std::pin::Pin<Box<dyn std::future::Future<Output = ()>>>)
                .collect(),
            || false,
        );

        let completed_games = games_completed.load(Ordering::Relaxed);
        let collected_samples = samples_collected.load(Ordering::Relaxed);
        // We collect at least target_samples (may be slightly more due to race)
        assert!(collected_samples >= target_samples);
        // TicTacToe games are 5-9 moves, so ~22-40 games for 200 samples
        assert!(
            completed_games >= 20,
            "expected at least 20 games, got {completed_games}"
        );
        assert_eq!(replay_buffer.len(), collected_samples);
    }

    /// Test multithreaded worker_loop with shared counter using the sync evaluator.
    #[test]
    fn test_multithreaded_worker_loop() {
        const NUM_THREADS: usize = 2;
        const WORKERS_PER_THREAD: usize = 4;
        const TARGET_SAMPLES: usize = 200; // ~32 games * 6 samples/game

        let total_samples = Arc::new(AtomicUsize::new(0));
        let games_completed = Arc::new(AtomicUsize::new(0));
        let replay_buffer = ObservationReplayBuffer::<i8, Ix1, 9>::new(1000, TicTacToe::OBS_SHAPE);

        thread::scope(|s| {
            for thread_id in 0..NUM_THREADS {
                let samples_collected = total_samples.clone();
                let games_completed = games_completed.clone();
                let replay_buffer = &replay_buffer;

                s.spawn(move || {
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

                    let futures: Vec<_> = (0..WORKERS_PER_THREAD)
                        .map(|i| {
                            let evaluator = &evaluator;
                            let config = &config;
                            let samples_collected = samples_collected.clone();
                            let games_completed = games_completed.clone();
                            let mut rng = ChaCha8Rng::seed_from_u64((thread_id * 1000 + i) as u64);
                            async move {
                                worker_loop::<TicTacToe, _, _, 9>(
                                    evaluator,
                                    config,
                                    &mut rng,
                                    samples_collected,
                                    games_completed,
                                    TARGET_SAMPLES,
                                    replay_buffer,
                                )
                                .await;
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
                });
            }
        });

        let completed_games = games_completed.load(Ordering::Relaxed);
        let collected_samples = total_samples.load(Ordering::Relaxed);

        // We collect at least TARGET_SAMPLES (may be slightly more due to race)
        assert!(collected_samples >= TARGET_SAMPLES);
        assert!(completed_games >= 30); // At least ~30 games to get 200 samples
        assert_eq!(replay_buffer.len(), collected_samples);
    }
}
