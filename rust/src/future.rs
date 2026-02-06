//! Future implementation for GPU evaluation requests.
//!
//! GpuEvalFuture represents a pending GPU inference job. The observation
//! is submitted immediately when the future is created, so the future
//! just polls for completion.

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

use ndarray::Dimension;

use crate::queue::GpuJobQueue;
use crate::BatchDim;

// Thread-local flag for tracking whether any future made progress.
// Used by the executor to decide whether to park.
std::thread_local! {
    static MADE_PROGRESS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Signal that progress was made (a future submitted or completed work).
pub fn signal_progress() {
    MADE_PROGRESS.with(|p| p.set(true));
}

/// Check if progress was made and reset the flag.
pub fn take_progress() -> bool {
    MADE_PROGRESS.with(|p| p.replace(false))
}

/// A future representing a GPU evaluation request.
///
/// The observation is submitted when this future is created (not on first poll).
/// Polling checks if the batch containing this job is complete.
pub struct GpuEvalFuture<'a, A, D, O>
where
    A: Clone + Default + Send + Sync,
    D: BatchDim,
    D::Larger: Dimension,
    O: Copy + Default + Send + Sync,
{
    queue: &'a GpuJobQueue<A, D, O>,
    ticket: u64,
    completed: bool,
}

impl<'a, A, D, O> GpuEvalFuture<'a, A, D, O>
where
    A: Clone + Default + Send + Sync,
    D: BatchDim,
    D::Larger: Dimension,
    O: Copy + Default + Send + Sync,
{
    /// Create a new future with an already-submitted ticket.
    pub fn new(queue: &'a GpuJobQueue<A, D, O>, ticket: u64) -> Self {
        // Signal progress on creation since we just submitted
        signal_progress();
        Self {
            queue,
            ticket,
            completed: false,
        }
    }
}

impl<'a, A, D, O> Future for GpuEvalFuture<'a, A, D, O>
where
    A: Clone + Default + Send + Sync,
    D: BatchDim,
    D::Larger: Dimension,
    O: Copy + Default + Send + Sync,
{
    type Output = O;

    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        // SAFETY: We don't move out of self, just update completed flag
        let this = unsafe { self.get_unchecked_mut() };

        if this.completed {
            panic!("GpuEvalFuture polled after completion");
        }

        if let Some(&output) = this.queue.poll(this.ticket) {
            this.completed = true;
            signal_progress();
            Poll::Ready(output)
        } else {
            Poll::Pending
        }
    }
}

impl<A, D, O> GpuJobQueue<A, D, O>
where
    A: Clone + Default + Send + Sync,
    D: BatchDim,
    D::Larger: Dimension,
    O: Copy + Default + Send + Sync,
{
    /// Submit an observation and create a future that will resolve to the output.
    ///
    /// The observation is written immediately via the callback.
    /// The returned future polls for the batch to complete.
    pub fn eval<F>(&self, write_obs: F) -> GpuEvalFuture<'_, A, D, O>
    where
        F: FnOnce(ndarray::ArrayViewMut<A, D>),
    {
        let ticket = self.submit(write_obs);
        GpuEvalFuture::new(self, ticket)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::queue::BATCH_SIZE;
    use ndarray::Ix0;
    use std::sync::Arc;
    use std::task::{RawWaker, RawWakerVTable, Waker};

    // Create a dummy waker that does nothing
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

    #[test]
    fn test_future_submits_immediately() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let submit_count = Arc::new(AtomicUsize::new(0));
        let submit_count_clone = submit_count.clone();

        let queue: Arc<GpuJobQueue<u64, Ix0, u64>> = Arc::new(GpuJobQueue::new(
            Ix0(),
            BATCH_SIZE,
            move |inputs, outputs| {
                submit_count_clone.fetch_add(1, Ordering::SeqCst);
                for (i, input) in inputs.iter().enumerate() {
                    outputs[i] = input * 2;
                }
            },
        ));

        assert_eq!(submit_count.load(Ordering::SeqCst), 0);

        // Create futures - they submit immediately
        let mut futures: Vec<_> = (0..BATCH_SIZE as u64)
            .map(|i| queue.eval(|mut out| out[()] = i))
            .collect();

        // Batch should have been dispatched (last eval triggered it)
        assert_eq!(submit_count.load(Ordering::SeqCst), 1);

        // Poll all futures - they should all be ready
        let waker = dummy_waker();
        let mut cx = Context::from_waker(&waker);

        for fut in &mut futures {
            match Pin::new(fut).poll(&mut cx) {
                Poll::Ready(_) => {}
                Poll::Pending => panic!("future should be ready"),
            }
        }
    }

    #[test]
    fn test_future_returns_correct_result() {
        let queue: Arc<GpuJobQueue<u64, Ix0, u64>> =
            Arc::new(GpuJobQueue::new(Ix0(), BATCH_SIZE, |inputs, outputs| {
                for (i, input) in inputs.iter().enumerate() {
                    outputs[i] = input + 100;
                }
            }));

        let mut futures: Vec<_> = (0..BATCH_SIZE as u64)
            .map(|i| queue.eval(|mut out| out[()] = i))
            .collect();

        let waker = dummy_waker();
        let mut cx = Context::from_waker(&waker);

        // All should be ready immediately (batch was triggered)
        for (i, fut) in futures.iter_mut().enumerate() {
            match Pin::new(fut).poll(&mut cx) {
                Poll::Ready(result) => {
                    assert_eq!(result, (i as u64) + 100);
                }
                Poll::Pending => panic!("future {} should be ready", i),
            }
        }
    }

    #[test]
    fn test_progress_tracking() {
        let queue: Arc<GpuJobQueue<u64, Ix0, u64>> =
            Arc::new(GpuJobQueue::new(Ix0(), BATCH_SIZE, |inputs, outputs| {
                for (i, input) in inputs.iter().enumerate() {
                    outputs[i] = *input;
                }
            }));

        // Clear any previous progress
        take_progress();

        // Create one future (partial batch) - should signal progress on creation
        let _fut = queue.eval(|mut out| out[()] = 0);

        // Should have made progress (submitted)
        assert!(take_progress());
    }
}
