//! Async evaluator trait and implementations for GPU inference.

use std::future::Future;

use crate::queue::GpuJobQueue;
use crate::Environment;

/// Output from neural network evaluation: (policy logits, value).
/// Policy has one entry per possible action, value is in [-1, 1].
#[derive(Clone, Copy)]
pub struct PolicyValue<const NUM_ACTIONS: usize> {
    pub policy: [f32; NUM_ACTIONS],
    pub value: f32,
}

impl<const NUM_ACTIONS: usize> Default for PolicyValue<NUM_ACTIONS> {
    fn default() -> Self {
        Self {
            policy: [0.0; NUM_ACTIONS],
            value: 0.0,
        }
    }
}

/// Async evaluator trait for neural network inference.
pub trait Evaluator<E: Environment> {
    /// Evaluate the environment and return (policy, value).
    /// Policy is over all actions, value is in [-1, 1] from current player's perspective.
    fn evaluate(&self, env: &E) -> impl Future<Output = (Vec<f32>, f32)>;
}

/// GPU-backed evaluator that batches inference requests.
///
/// Wraps a GpuJobQueue and converts between Environment observations
/// and the queue's I/O types.
pub struct GpuEvaluator<'a, E: Environment, const NUM_ACTIONS: usize> {
    queue: &'a GpuJobQueue<E::Observation, PolicyValue<NUM_ACTIONS>>,
}

impl<'a, E, const NUM_ACTIONS: usize> GpuEvaluator<'a, E, NUM_ACTIONS>
where
    E: Environment,
    E::Observation: Copy + Default,
{
    pub fn new(queue: &'a GpuJobQueue<E::Observation, PolicyValue<NUM_ACTIONS>>) -> Self {
        Self { queue }
    }
}

impl<'a, E, const NUM_ACTIONS: usize> Evaluator<E> for GpuEvaluator<'a, E, NUM_ACTIONS>
where
    E: Environment,
    E::Observation: Copy + Default,
{
    fn evaluate(&self, env: &E) -> impl Future<Output = (Vec<f32>, f32)> {
        let obs = env.observation();
        let future = self.queue.eval(obs);

        async move {
            let result = future.await;
            (result.policy.to_vec(), result.value)
        }
    }
}

/// Synchronous CPU evaluator for testing.
///
/// Returns uniform policy and zero value.
pub struct UniformEvaluator;

impl<E: Environment> Evaluator<E> for UniformEvaluator {
    fn evaluate(&self, _env: &E) -> impl Future<Output = (Vec<f32>, f32)> {
        let num_actions = E::NUM_ACTIONS;
        let policy = vec![1.0 / num_actions as f32; num_actions];
        std::future::ready((policy, 0.0))
    }
}

/// CPU evaluator that uses a sync evaluation function.
///
/// Useful for testing or CPU-only inference.
pub struct SyncEvaluator<E: Environment, F>
where
    F: Fn(&E) -> (Vec<f32>, f32),
{
    eval_fn: F,
    _phantom: std::marker::PhantomData<E>,
}

impl<E: Environment, F> SyncEvaluator<E, F>
where
    F: Fn(&E) -> (Vec<f32>, f32),
{
    pub fn new(eval_fn: F) -> Self {
        Self {
            eval_fn,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<E: Environment, F> Evaluator<E> for SyncEvaluator<E, F>
where
    F: Fn(&E) -> (Vec<f32>, f32),
{
    fn evaluate(&self, env: &E) -> impl Future<Output = (Vec<f32>, f32)> {
        let result = (self.eval_fn)(env);
        std::future::ready(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::environments::TicTacToe;

    #[test]
    fn test_uniform_evaluator() {
        use std::pin::Pin;
        use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

        fn dummy_waker() -> Waker {
            fn clone(_: *const ()) -> RawWaker {
                RawWaker::new(std::ptr::null(), &VTABLE)
            }
            fn wake(_: *const ()) {}
            fn wake_by_ref(_: *const ()) {}
            fn drop(_: *const ()) {}
            static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, wake, wake_by_ref, drop);
            unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) }
        }

        let env = TicTacToe::new();
        let evaluator = UniformEvaluator;

        let mut future = evaluator.evaluate(&env);
        let waker = dummy_waker();
        let mut cx = Context::from_waker(&waker);

        match Pin::new(&mut future).poll(&mut cx) {
            Poll::Ready((policy, value)) => {
                assert_eq!(policy.len(), 9);
                assert!((policy[0] - 1.0 / 9.0).abs() < 0.001);
                assert_eq!(value, 0.0);
            }
            Poll::Pending => panic!("UniformEvaluator should be ready immediately"),
        }
    }

    #[test]
    fn test_sync_evaluator() {
        use std::pin::Pin;
        use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

        fn dummy_waker() -> Waker {
            fn clone(_: *const ()) -> RawWaker {
                RawWaker::new(std::ptr::null(), &VTABLE)
            }
            fn wake(_: *const ()) {}
            fn wake_by_ref(_: *const ()) {}
            fn drop(_: *const ()) {}
            static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, wake, wake_by_ref, drop);
            unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) }
        }

        let env = TicTacToe::new();
        let evaluator = SyncEvaluator::new(|_env: &TicTacToe| {
            let mut policy = vec![0.0; 9];
            policy[4] = 1.0; // Center is best
            (policy, 0.5)
        });

        let mut future = evaluator.evaluate(&env);
        let waker = dummy_waker();
        let mut cx = Context::from_waker(&waker);

        match Pin::new(&mut future).poll(&mut cx) {
            Poll::Ready((policy, value)) => {
                assert_eq!(policy[4], 1.0);
                assert_eq!(value, 0.5);
            }
            Poll::Pending => panic!("SyncEvaluator should be ready immediately"),
        }
    }

    #[test]
    fn test_gpu_evaluator() {
        use crate::queue::BATCH_SIZE;
        use std::sync::Arc;

        // Create a mock GPU queue that returns uniform policy
        type Output = PolicyValue<{ TicTacToe::NUM_ACTIONS }>;
        let queue: Arc<GpuJobQueue<[u8; 9], Output>> = Arc::new(GpuJobQueue::new(
            |_inputs: &[[u8; 9]], outputs: &mut [Output]| {
                for output in outputs.iter_mut() {
                    output.policy = [1.0 / 9.0; 9];
                    output.value = 0.0;
                }
            },
        ));

        let evaluator = GpuEvaluator::new(&*queue);

        // Submit BATCH_SIZE evaluations to trigger dispatch
        let envs: Vec<TicTacToe> = (0..BATCH_SIZE).map(|_| TicTacToe::new()).collect();

        // Use the executor to run the futures
        use crate::executor::Executor;
        use std::cell::Cell;
        use std::rc::Rc;

        let results: Rc<Cell<usize>> = Rc::new(Cell::new(0));

        let futures: Vec<_> = envs
            .iter()
            .map(|env| {
                let results = results.clone();
                let fut = evaluator.evaluate(env);
                async move {
                    let (policy, value) = fut.await;
                    assert_eq!(policy.len(), 9);
                    assert!((policy[0] - 1.0 / 9.0).abs() < 0.001);
                    assert_eq!(value, 0.0);
                    results.set(results.get() + 1);
                }
            })
            .collect();

        let executor = Executor::new(|| queue.listen());
        executor.run(
            futures
                .into_iter()
                .map(|f| Box::pin(f) as std::pin::Pin<Box<dyn std::future::Future<Output = ()>>>)
                .collect(),
            || false,
        );

        assert_eq!(results.get(), BATCH_SIZE);
    }
}
