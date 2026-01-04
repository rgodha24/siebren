//! Lock-free GPU job queue for batching inference requests.
//!
//! Uses atomic fetch_add for slot assignment and batch completion tracking.
//! Designed for 512 workers (32 threads x 16 workers) submitting to batches of 256.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU64, Ordering};

use event_listener::Event;

/// Number of jobs per batch. In production this would be 256.
/// Using a smaller value for tests to avoid deadlock with few workers.
#[cfg(test)]
pub const BATCH_SIZE: usize = 16;
#[cfg(not(test))]
pub const BATCH_SIZE: usize = 256;

pub const NUM_BATCHES: usize = 8;
pub const TOTAL_SLOTS: usize = BATCH_SIZE * NUM_BATCHES;

/// A lock-free queue for batching GPU inference jobs.
///
/// Workers submit inputs and receive tickets. When a batch fills (256 jobs),
/// the completing worker triggers GPU dispatch. Workers poll for results
/// using their tickets.
pub struct GpuJobQueue<I, O> {
    /// Monotonically increasing counter for slot assignment.
    write_ticket: AtomicU64,
    /// Count of completed writes per batch slot.
    /// When this reaches BATCH_SIZE, the batch is ready for GPU dispatch.
    batch_writes: [AtomicU64; NUM_BATCHES],

    /// Ticket number at which each batch was completed (end of batch).
    /// Workers check this to know if their result is ready.
    batch_complete: [AtomicU64; NUM_BATCHES],

    /// Input buffer. Size = TOTAL_SLOTS.
    /// UnsafeCell because multiple threads write to different slots concurrently.
    inputs: Box<[UnsafeCell<I>]>,

    /// Output buffer. Size = TOTAL_SLOTS.
    outputs: Box<[UnsafeCell<O>]>,

    /// Event for parking threads when waiting for GPU completion.
    completion_event: Event,

    /// Callback invoked when a batch is ready.
    /// Receives (inputs slice, outputs slice) and should fill outputs.
    dispatch: Box<dyn Fn(&[I], &mut [O]) + Send + Sync>,
}

// SAFETY: The queue is designed for concurrent access:
// - write_ticket ensures each slot is claimed by exactly one writer
// - batch_writes/batch_complete use atomic operations
// - inputs/outputs slots are only written by their ticket owner, read after batch_complete
// - dispatch is Send + Sync
unsafe impl<I: Send, O: Send> Send for GpuJobQueue<I, O> {}
unsafe impl<I: Send + Sync, O: Send + Sync> Sync for GpuJobQueue<I, O> {}

impl<I, O> GpuJobQueue<I, O>
where
    I: Copy + Default,
    O: Copy + Default,
{
    /// Creates a new job queue with the given dispatch callback.
    ///
    /// The callback is invoked when a batch of BATCH_SIZE jobs is ready.
    /// It receives input and output slices and should fill the outputs.
    pub fn new<F>(dispatch: F) -> Self
    where
        F: Fn(&[I], &mut [O]) + Send + Sync + 'static,
    {
        // Initialize input/output buffers
        let inputs: Box<[UnsafeCell<I>]> = (0..TOTAL_SLOTS)
            .map(|_| UnsafeCell::new(I::default()))
            .collect();
        let outputs: Box<[UnsafeCell<O>]> = (0..TOTAL_SLOTS)
            .map(|_| UnsafeCell::new(O::default()))
            .collect();

        Self {
            write_ticket: AtomicU64::new(0),
            batch_writes: std::array::from_fn(|_| AtomicU64::new(0)),
            batch_complete: std::array::from_fn(|_| AtomicU64::new(0)),
            inputs,
            outputs,
            completion_event: Event::new(),
            dispatch: Box::new(dispatch),
        }
    }

    /// Submit a job and return a ticket for polling the result.
    ///
    /// If this submission completes a batch, the current thread will
    /// synchronously dispatch the batch (blocking until complete).
    pub fn submit(&self, input: I) -> u64 {
        // Claim a slot
        let ticket = self.write_ticket.fetch_add(1, Ordering::Relaxed);
        let slot_idx = (ticket as usize) % TOTAL_SLOTS;
        let batch_idx = ((ticket as usize) / BATCH_SIZE) % NUM_BATCHES;

        // Write input to our slot
        // SAFETY: We own this slot exclusively until we increment batch_writes
        unsafe {
            *self.inputs[slot_idx].get() = input;
        }

        // Release fence ensures the write is visible before we signal completion
        std::sync::atomic::fence(Ordering::Release);

        // Signal that our write is complete
        let writes_in_batch = self.batch_writes[batch_idx].fetch_add(1, Ordering::AcqRel) + 1;

        // If we completed the batch, dispatch it
        if writes_in_batch == BATCH_SIZE as u64 {
            self.dispatch_batch(batch_idx, ticket);
        }

        ticket
    }

    /// Dispatch a completed batch to the GPU.
    fn dispatch_batch(&self, batch_idx: usize, trigger_ticket: u64) {
        let batch_start_slot = batch_idx * BATCH_SIZE;

        // SAFETY: All writes to this batch are complete (batch_writes == BATCH_SIZE)
        let inputs: Vec<I> = (batch_start_slot..batch_start_slot + BATCH_SIZE)
            .map(|i| unsafe { *self.inputs[i].get() })
            .collect();

        let mut outputs: Vec<O> = vec![O::default(); BATCH_SIZE];

        (self.dispatch)(&inputs, &mut outputs);

        // SAFETY: We're the only one writing outputs for this batch
        for (i, output) in outputs.into_iter().enumerate() {
            unsafe {
                *self.outputs[batch_start_slot + i].get() = output;
            }
        }

        // Calculate the batch end ticket (first ticket of next batch)
        // trigger_ticket is somewhere in [batch_start, batch_end)
        // batch_end_ticket = (batch_number + 1) * BATCH_SIZE
        let batch_number = trigger_ticket / BATCH_SIZE as u64;
        let batch_end_ticket = (batch_number + 1) * BATCH_SIZE as u64;

        // Mark batch complete (release ensures outputs are visible)
        self.batch_complete[batch_idx].store(batch_end_ticket, Ordering::Release);

        // Reset batch_writes for next use of this slot
        self.batch_writes[batch_idx].store(0, Ordering::Relaxed);

        // Wake all waiting threads
        self.completion_event.notify(usize::MAX);
    }

    /// Poll for a result. Returns Some(&O) if ready, None if still pending.
    pub fn poll(&self, ticket: u64) -> Option<&O> {
        let batch_idx = ((ticket as usize) / BATCH_SIZE) % NUM_BATCHES;
        let batch_end_ticket = ((ticket / BATCH_SIZE as u64) + 1) * BATCH_SIZE as u64;

        // Check if this batch is complete
        if self.batch_complete[batch_idx].load(Ordering::Acquire) < batch_end_ticket {
            return None;
        }

        // Batch is complete, return reference to output
        let slot_idx = (ticket as usize) % TOTAL_SLOTS;
        // SAFETY: batch_complete >= batch_end_ticket means output is written and won't change
        Some(unsafe { &*self.outputs[slot_idx].get() })
    }

    /// Get a listener for the completion event.
    /// Use this before polling to avoid missing notifications.
    pub fn listen(&self) -> event_listener::EventListener {
        self.completion_event.listen()
    }

    /// Wake all waiters (used for GPU completion or external cancellation).
    pub fn notify_all(&self) {
        self.completion_event.notify(usize::MAX);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_single_batch_completion() {
        let queue: Arc<GpuJobQueue<u64, u64>> = Arc::new(GpuJobQueue::new(|inputs, outputs| {
            // Simple transform: output = input * 2
            for (i, &input) in inputs.iter().enumerate() {
                outputs[i] = input * 2;
            }
        }));

        // Submit BATCH_SIZE jobs
        let tickets: Vec<u64> = (0..BATCH_SIZE as u64).map(|i| queue.submit(i)).collect();

        // All should be complete now (last submit triggered dispatch)
        for (i, &ticket) in tickets.iter().enumerate() {
            let result = queue.poll(ticket);
            assert!(result.is_some(), "ticket {} should be ready", ticket);
            assert_eq!(*result.unwrap(), (i as u64) * 2);
        }
    }

    #[test]
    fn test_partial_batch_not_ready() {
        let queue: Arc<GpuJobQueue<u64, u64>> = Arc::new(GpuJobQueue::new(|inputs, outputs| {
            for (i, &input) in inputs.iter().enumerate() {
                outputs[i] = input * 2;
            }
        }));

        // Submit less than a full batch
        let tickets: Vec<u64> = (0..BATCH_SIZE as u64 - 1)
            .map(|i| queue.submit(i))
            .collect();

        // None should be ready
        for &ticket in &tickets {
            assert!(
                queue.poll(ticket).is_none(),
                "partial batch should not be ready"
            );
        }

        // Complete the batch
        queue.submit(BATCH_SIZE as u64 - 1);

        // Now all should be ready
        for &ticket in &tickets {
            assert!(
                queue.poll(ticket).is_some(),
                "batch should be ready after completion"
            );
        }
    }

    #[test]
    fn test_multiple_batches() {
        let queue: Arc<GpuJobQueue<u64, u64>> = Arc::new(GpuJobQueue::new(|inputs, outputs| {
            for (i, &input) in inputs.iter().enumerate() {
                outputs[i] = input + 1000;
            }
        }));

        // Submit 3 full batches
        let all_tickets: Vec<u64> = (0..(BATCH_SIZE * 3) as u64)
            .map(|i| queue.submit(i))
            .collect();

        // All should be ready
        for (i, &ticket) in all_tickets.iter().enumerate() {
            let result = queue.poll(ticket).expect("should be ready");
            assert_eq!(*result, (i as u64) + 1000);
        }
    }

    #[test]
    fn test_batch_slot_reuse() {
        // This test verifies that batch slots can be reused correctly.
        // Key invariant: a slot can't be resubmitted until its owner reads the result.
        // With TOTAL_SLOTS = 2048 and submitting in batches, we need to read results
        // before we can wrap around and reuse slots.
        let queue: Arc<GpuJobQueue<u64, u64>> = Arc::new(GpuJobQueue::new(|inputs, outputs| {
            for (i, &input) in inputs.iter().enumerate() {
                outputs[i] = input;
            }
        }));

        // Submit exactly NUM_BATCHES batches (fills all slots)
        let tickets_round1: Vec<u64> = (0..TOTAL_SLOTS as u64).map(|i| queue.submit(i)).collect();

        // Read all results from round 1
        for (i, &ticket) in tickets_round1.iter().enumerate() {
            let result = queue.poll(ticket).expect("should be ready");
            assert_eq!(
                *result, i as u64,
                "round 1 ticket {} has wrong value",
                ticket
            );
        }

        // Now submit another round (reusing slots)
        let tickets_round2: Vec<u64> = (TOTAL_SLOTS as u64..(TOTAL_SLOTS * 2) as u64)
            .map(|i| queue.submit(i))
            .collect();

        // Read all results from round 2
        for (i, &ticket) in tickets_round2.iter().enumerate() {
            let expected = (TOTAL_SLOTS + i) as u64;
            let result = queue.poll(ticket).expect("should be ready");
            assert_eq!(
                *result, expected,
                "round 2 ticket {} has wrong value",
                ticket
            );
        }
    }

    #[test]
    fn test_concurrent_submissions() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::thread;

        let queue: Arc<GpuJobQueue<u64, u64>> = Arc::new(GpuJobQueue::new(|inputs, outputs| {
            for (i, &input) in inputs.iter().enumerate() {
                outputs[i] = input * 2;
            }
        }));

        let batch_count = Arc::new(AtomicUsize::new(0));
        let num_threads = 4;
        let jobs_per_thread = BATCH_SIZE * 2; // Each thread submits 2 batches worth

        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let queue = queue.clone();
                let batch_count = batch_count.clone();
                thread::spawn(move || {
                    let base = (thread_id * jobs_per_thread) as u64;
                    let mut results = Vec::new();

                    for i in 0..jobs_per_thread as u64 {
                        let ticket = queue.submit(base + i);
                        results.push((ticket, base + i));
                    }

                    // Wait for results
                    for (ticket, expected_input) in results {
                        loop {
                            if let Some(&result) = queue.poll(ticket) {
                                assert_eq!(result, expected_input * 2);
                                batch_count.fetch_add(1, Ordering::Relaxed);
                                break;
                            }
                            // Busy wait (in real code we'd use the event listener)
                            std::hint::spin_loop();
                        }
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("thread panicked");
        }

        assert_eq!(
            batch_count.load(Ordering::Relaxed),
            num_threads * jobs_per_thread
        );
    }
}
