//! Future implementation for GPU evaluation requests.
//!
//! GpuEvalFuture represents a pending GPU inference job. It transitions through:
//! 1. NotSubmitted - contains input, first poll submits to queue
//! 2. Pending - waiting for batch to complete
//! 3. Ready - result available (Future returns Poll::Ready)

use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::task::{Context, Poll};

use crate::queue::GpuJobQueue;

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

/// State of a GPU evaluation future.
enum State<I> {
    /// Input ready to submit on first poll.
    NotSubmitted(I),
    /// Submitted, waiting for result.
    Pending { ticket: u64 },
    /// Already completed (for safety, shouldn't be polled again).
    Completed,
}

/// A future representing a GPU evaluation request.
///
/// On first poll, submits the input to the queue and transitions to Pending.
/// Subsequent polls check if the batch is complete.
pub struct GpuEvalFuture<'a, I, O>
where
    I: Copy + Default,
    O: Copy + Default,
{
    queue: &'a GpuJobQueue<I, O>,
    state: State<I>,
    _phantom: PhantomData<O>,
}

impl<'a, I, O> GpuEvalFuture<'a, I, O>
where
    I: Copy + Default,
    O: Copy + Default,
{
    /// Create a new future that will submit the input on first poll.
    pub fn new(queue: &'a GpuJobQueue<I, O>, input: I) -> Self {
        Self {
            queue,
            state: State::NotSubmitted(input),
            _phantom: PhantomData,
        }
    }
}

impl<'a, I, O> Future for GpuEvalFuture<'a, I, O>
where
    I: Copy + Default,
    O: Copy + Default,
{
    type Output = O;

    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        // SAFETY: We don't move out of self, just update state
        let this = unsafe { self.get_unchecked_mut() };

        match &this.state {
            State::NotSubmitted(input) => {
                // First poll: submit to queue
                let input = *input;
                let ticket = this.queue.submit(input);
                this.state = State::Pending { ticket };

                // Signal progress: we did work (submitted a job)
                signal_progress();

                // Immediately check if already complete (might be if we triggered batch)
                if let Some(&output) = this.queue.poll(ticket) {
                    this.state = State::Completed;
                    signal_progress();
                    Poll::Ready(output)
                } else {
                    Poll::Pending
                }
            }
            State::Pending { ticket } => {
                // Subsequent polls: check if result ready
                if let Some(&output) = this.queue.poll(*ticket) {
                    this.state = State::Completed;
                    signal_progress();
                    Poll::Ready(output)
                } else {
                    // No progress - still waiting
                    Poll::Pending
                }
            }
            State::Completed => {
                panic!("GpuEvalFuture polled after completion");
            }
        }
    }
}

impl<I, O> GpuJobQueue<I, O>
where
    I: Copy + Default,
    O: Copy + Default,
{
    /// Create a future that will evaluate the input on the GPU.
    pub fn eval(&self, input: I) -> GpuEvalFuture<'_, I, O> {
        GpuEvalFuture::new(self, input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
    fn test_future_submits_on_first_poll() {
        use crate::queue::BATCH_SIZE;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let submit_count = Arc::new(AtomicUsize::new(0));
        let submit_count_clone = submit_count.clone();

        let queue: Arc<GpuJobQueue<u64, u64>> =
            Arc::new(GpuJobQueue::new(move |inputs, outputs| {
                submit_count_clone.fetch_add(1, Ordering::SeqCst);
                for (i, &input) in inputs.iter().enumerate() {
                    outputs[i] = input * 2;
                }
            }));

        // Create futures but don't poll yet
        let mut futures: Vec<_> = (0..BATCH_SIZE as u64).map(|i| queue.eval(i)).collect();

        assert_eq!(submit_count.load(Ordering::SeqCst), 0);

        // Poll all futures once
        let waker = dummy_waker();
        let mut cx = Context::from_waker(&waker);

        for fut in &mut futures {
            let _ = Pin::new(fut).poll(&mut cx);
        }

        // Batch should have been dispatched
        assert_eq!(submit_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_future_returns_correct_result() {
        use crate::queue::BATCH_SIZE;

        let queue: Arc<GpuJobQueue<u64, u64>> = Arc::new(GpuJobQueue::new(|inputs, outputs| {
            for (i, &input) in inputs.iter().enumerate() {
                outputs[i] = input + 100;
            }
        }));

        let mut futures: Vec<_> = (0..BATCH_SIZE as u64).map(|i| queue.eval(i)).collect();

        let waker = dummy_waker();
        let mut cx = Context::from_waker(&waker);

        // Collect results - some futures may complete on first poll (the one that
        // triggers batch), others need a second poll
        let mut results: Vec<Option<u64>> = vec![None; BATCH_SIZE];

        // Poll until all complete
        for _ in 0..2 {
            for (i, fut) in futures.iter_mut().enumerate() {
                if results[i].is_some() {
                    continue; // Already completed
                }
                if let Poll::Ready(result) = Pin::new(fut).poll(&mut cx) {
                    results[i] = Some(result);
                }
            }
        }

        // Verify all completed with correct values
        for (i, result) in results.iter().enumerate() {
            let result = result.expect(&format!("Future {} should have completed", i));
            assert_eq!(result, (i as u64) + 100);
        }
    }

    #[test]
    fn test_progress_tracking() {
        let queue: Arc<GpuJobQueue<u64, u64>> = Arc::new(GpuJobQueue::new(|inputs, outputs| {
            for (i, &input) in inputs.iter().enumerate() {
                outputs[i] = input;
            }
        }));

        let waker = dummy_waker();
        let mut cx = Context::from_waker(&waker);

        // Clear any previous progress
        take_progress();

        // Create and poll one future (partial batch)
        let mut fut = queue.eval(0);
        let _ = Pin::new(&mut fut).poll(&mut cx);

        // Should have made progress (submitted)
        assert!(take_progress());

        // Poll again - still pending, no progress
        let _ = Pin::new(&mut fut).poll(&mut cx);
        assert!(!take_progress());
    }
}
