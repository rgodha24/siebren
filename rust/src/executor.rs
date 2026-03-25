//! Single-threaded async executor for GPU inference workers.
//!
//! This executor is designed for polling many workers that submit GPU inference
//! requests. It doesn't use wakers - instead, it polls all futures in a tight
//! loop and parks when no progress is made.

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

use event_listener::{EventListener, Listener};

use crate::future::{signal_progress, take_progress};

/// Create a dummy waker that does nothing.
/// We don't use wakers for signaling - we use event_listener + progress tracking.
fn dummy_waker() -> Waker {
    fn clone(_: *const ()) -> RawWaker {
        RawWaker::new(std::ptr::null(), &VTABLE)
    }
    fn wake(_: *const ()) {}
    fn wake_by_ref(_: *const ()) {}
    fn drop(_: *const ()) {}

    static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, wake, wake_by_ref, drop);

    // SAFETY: The vtable functions are valid and the data pointer is null (unused)
    unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) }
}

/// A single-threaded executor for GPU inference workers.
///
/// Polls all futures in round-robin until no progress is made, then parks
/// waiting for GPU batch completion.
pub struct Executor<'a> {
    /// Function to get an event listener for parking.
    listen_fn: Box<dyn Fn() -> EventListener + 'a>,
}

impl<'a> Executor<'a> {
    /// Create a new executor that will park using the given listen function.
    ///
    /// The listen function should return an EventListener from the GPU queue's
    /// completion_event.
    pub fn new<F>(listen_fn: F) -> Self
    where
        F: Fn() -> EventListener + 'a,
    {
        Self {
            listen_fn: Box::new(listen_fn),
        }
    }

    /// Run futures until completion or cancellation.
    ///
    /// Polls all futures in round-robin. When no future makes progress,
    /// parks until the GPU signals batch completion.
    ///
    /// Completed futures are removed (swap_remove). Pending futures remain
    /// alive in the vec. Returns when `cancel()` returns true or all futures
    /// complete. This allows preserving in-progress game state across
    /// pause/resume cycles.
    pub fn run<F, C>(&self, futures: &mut Vec<Pin<Box<F>>>, cancel: &mut C)
    where
        F: Future<Output = ()> + ?Sized,
        C: FnMut() -> bool,
    {
        let waker = dummy_waker();
        let mut cx = Context::from_waker(&waker);

        loop {
            if cancel() || futures.is_empty() {
                return;
            }

            // Poll all futures until no progress
            loop {
                if cancel() || futures.is_empty() {
                    return;
                }

                // Clear progress flag before polling round
                take_progress();

                // Poll all pending futures
                let mut i = 0;
                while i < futures.len() {
                    if cancel() {
                        return;
                    }

                    let poll_result = futures[i].as_mut().poll(&mut cx);
                    match poll_result {
                        Poll::Ready(()) => {
                            futures.swap_remove(i);
                            signal_progress();
                        }
                        Poll::Pending => {
                            i += 1;
                        }
                    }
                }

                if cancel() || futures.is_empty() {
                    return;
                }

                // If no progress was made, break to park
                if !take_progress() {
                    break;
                }
            }

            if futures.is_empty() {
                return;
            }

            // Set up listener BEFORE re-checking (avoid race).
            let listener = (self.listen_fn)();

            // Double-check before parking and re-evaluate cancellation.
            take_progress();
            let mut i = 0;
            while i < futures.len() {
                let poll_result = futures[i].as_mut().poll(&mut cx);
                match poll_result {
                    Poll::Ready(()) => {
                        futures.swap_remove(i);
                        signal_progress();
                    }
                    Poll::Pending => {
                        i += 1;
                    }
                }
            }

            if cancel() || futures.is_empty() {
                return;
            }

            if !take_progress() {
                listener.wait();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::rc::Rc;

    /// A simple counter future that completes after N polls
    struct CountdownFuture {
        remaining: Cell<usize>,
    }

    impl CountdownFuture {
        fn new(count: usize) -> Self {
            Self {
                remaining: Cell::new(count),
            }
        }
    }

    impl Future for CountdownFuture {
        type Output = ();

        fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<()> {
            let remaining = self.remaining.get();
            if remaining == 0 {
                Poll::Ready(())
            } else {
                self.remaining.set(remaining - 1);
                signal_progress();
                Poll::Pending
            }
        }
    }

    #[test]
    fn test_executor_runs_single_future() {
        let completed = Rc::new(Cell::new(false));
        let completed_clone = completed.clone();

        let fut = async move {
            completed_clone.set(true);
        };

        let executor = Executor::new(|| event_listener::Event::new().listen());
        executor.run(&mut vec![Box::pin(fut)], &mut || false);

        assert!(completed.get());
    }

    #[test]
    fn test_executor_runs_multiple_futures() {
        let count = Rc::new(Cell::new(0));

        let mut futures: Vec<Pin<Box<dyn Future<Output = ()>>>> = (0..10)
            .map(|_| {
                let count = count.clone();
                let fut = async move {
                    count.set(count.get() + 1);
                };
                Box::pin(fut) as Pin<Box<dyn Future<Output = ()>>>
            })
            .collect();

        let executor = Executor::new(|| event_listener::Event::new().listen());
        executor.run(&mut futures, &mut || false);

        assert_eq!(count.get(), 10);
    }

    #[test]
    fn test_executor_cancels_pending_futures() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        let cancelled = Arc::new(AtomicBool::new(false));
        let cancelled_clone = cancelled.clone();
        let event = Arc::new(event_listener::Event::new());

        // A future that never completes.
        let fut = async move {
            std::future::pending::<()>().await;
        };

        let event_for_exec = event.clone();
        let executor = Executor::new(move || event_for_exec.listen());

        // Cancel shortly after starting and notify to wake the executor.
        std::thread::spawn({
            let event = event.clone();
            let cancelled = cancelled.clone();
            move || {
                std::thread::sleep(std::time::Duration::from_millis(5));
                cancelled.store(true, Ordering::Relaxed);
                event.notify(usize::MAX);
            }
        });

        executor.run(&mut vec![Box::pin(fut)], &mut || {
            cancelled_clone.load(Ordering::Relaxed)
        });
    }

    #[test]
    fn test_executor_handles_multi_poll_futures() {
        let completed = Rc::new(Cell::new(0));

        let mut futures: Vec<Pin<Box<dyn Future<Output = ()>>>> = (0..5)
            .map(|i| {
                let completed = completed.clone();
                let countdown = CountdownFuture::new(i + 1);
                let fut = async move {
                    // Wrap countdown in a custom future that signals progress
                    std::future::poll_fn(|_cx| {
                        // Need to poll countdown, but poll requires Pin<&mut Self>
                        // so we'll inline the countdown logic
                        let remaining = countdown.remaining.get();
                        if remaining == 0 {
                            Poll::Ready(())
                        } else {
                            countdown.remaining.set(remaining - 1);
                            signal_progress();
                            Poll::Pending
                        }
                    })
                    .await;
                    completed.set(completed.get() + 1);
                };
                Box::pin(fut) as Pin<Box<dyn Future<Output = ()>>>
            })
            .collect();

        let executor = Executor::new(|| event_listener::Event::new().listen());
        executor.run(&mut futures, &mut || false);

        assert_eq!(completed.get(), 5);
    }

    #[test]
    fn test_executor_with_event_notification() {
        use std::sync::Arc;

        let event = Arc::new(event_listener::Event::new());
        let event_clone = event.clone();

        // Future that waits for event then completes
        let completed = Rc::new(Cell::new(false));
        let completed_clone = completed.clone();

        // Spawn a thread that will notify after a short delay
        let notify_thread = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(10));
            event_clone.notify(usize::MAX);
        });

        // Future that needs external notification to complete
        let polls = Rc::new(Cell::new(0));
        let polls_clone = polls.clone();

        let fut = std::future::poll_fn(move |_cx| {
            let p = polls_clone.get();
            polls_clone.set(p + 1);

            // Complete after being woken up (second poll after notification)
            if p > 0 {
                completed_clone.set(true);
                signal_progress();
                Poll::Ready(())
            } else {
                Poll::Pending
            }
        });

        let executor = Executor::new(move || event.listen());
        executor.run(&mut vec![Box::pin(fut)], &mut || false);

        notify_thread.join().unwrap();
        assert!(completed.get());
    }

    #[test]
    fn test_executor_checks_cancel_during_progress_loop() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let polls = Arc::new(AtomicUsize::new(0));
        let polls_for_future = polls.clone();

        let fut = std::future::poll_fn(move |_cx| {
            let n = polls_for_future.fetch_add(1, Ordering::Relaxed) + 1;
            if n < 1000 {
                signal_progress();
            }
            Poll::<()>::Pending
        });

        let executor = Executor::new(|| event_listener::Event::new().listen());
        let polls_for_cancel = polls.clone();
        executor.run(&mut vec![Box::pin(fut)], &mut || {
            polls_for_cancel.load(Ordering::Relaxed) >= 10
        });

        // If cancel is only checked outside the inner progress loop,
        // this would be close to 1000 polls before returning.
        assert!(
            polls.load(Ordering::Relaxed) < 100,
            "cancel should stop polling quickly"
        );
    }

    #[test]
    fn test_run_preserves_futures() {
        // Test that run does NOT drop pending futures.
        // We use futures that always signal progress and complete after enough polls.
        // Cancel immediately on first check to guarantee futures are still pending.

        let completed = Rc::new(Cell::new(0usize));

        let mut futures: Vec<Pin<Box<dyn Future<Output = ()>>>> = (0..3)
            .map(|_| {
                let completed = completed.clone();
                let polls = Cell::new(0usize);
                let fut = std::future::poll_fn(move |_cx| {
                    let p = polls.get();
                    polls.set(p + 1);
                    signal_progress();
                    if p >= 10 {
                        completed.set(completed.get() + 1);
                        Poll::Ready(())
                    } else {
                        Poll::Pending
                    }
                });
                Box::pin(fut) as Pin<Box<dyn Future<Output = ()>>>
            })
            .collect();

        let executor = Executor::new(|| event_listener::Event::new().listen());

        // Cancel immediately - futures should not be polled at all.
        executor.run(&mut futures, &mut || true);

        // Futures should still be alive (cancel was true from the start).
        assert_eq!(
            futures.len(),
            3,
            "all futures must be preserved when cancel is immediate"
        );
        assert_eq!(completed.get(), 0, "no futures should have completed");

        // Now let them finish.
        executor.run(&mut futures, &mut || false);

        assert_eq!(completed.get(), 3, "all futures should have completed");
        assert!(futures.is_empty(), "completed futures should be removed");
    }

    #[test]
    fn test_run_does_not_drop_pending() {
        // Verify a future's drop impl is NOT called between run calls.

        let was_dropped = Rc::new(Cell::new(false));
        let was_dropped_clone = was_dropped.clone();

        struct DropDetector {
            flag: Rc<Cell<bool>>,
            polls: Cell<usize>,
        }
        impl Drop for DropDetector {
            fn drop(&mut self) {
                self.flag.set(true);
            }
        }
        impl Future for DropDetector {
            type Output = ();
            fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<()> {
                let this = unsafe { self.get_unchecked_mut() };
                let p = this.polls.get();
                this.polls.set(p + 1);
                signal_progress();
                if p >= 5 {
                    Poll::Ready(())
                } else {
                    Poll::Pending
                }
            }
        }

        let mut futures: Vec<Pin<Box<dyn Future<Output = ()>>>> = vec![Box::pin(DropDetector {
            flag: was_dropped_clone,
            polls: Cell::new(0),
        })];

        let executor = Executor::new(|| event_listener::Event::new().listen());

        // Cancel immediately
        executor.run(&mut futures, &mut || true);

        // The future should NOT have been dropped
        assert!(
            !was_dropped.get(),
            "future must not be dropped between run calls"
        );
        assert_eq!(futures.len(), 1, "future should still be in the vec");

        // Now let it finish
        executor.run(&mut futures, &mut || false);

        assert!(futures.is_empty());
        assert!(
            was_dropped.get(),
            "future should be dropped after completion"
        );
    }
}
