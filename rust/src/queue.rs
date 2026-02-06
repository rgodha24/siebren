//! Lock-free GPU job queue for batching inference requests.
//!
//! Uses atomic fetch_add for slot assignment and batch completion tracking.
//! Queue storage is sized from worker count at construction time.
//!
//! Observations are stored in a single contiguous array with shape
//! `(total_slots, ...obs_shape)`.
//! This enables zero-copy batch slicing for GPU dispatch.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU64, Ordering};

use event_listener::Event;
use ndarray::{Array, ArrayView, ArrayViewMut, Axis, Slice};

use crate::BatchDim;

/// Number of jobs per batch. In production this would be 256.
/// Using a smaller value for tests to avoid deadlock with few workers.
#[cfg(test)]
pub const BATCH_SIZE: usize = 16;
#[cfg(not(test))]
pub const BATCH_SIZE: usize = 256;

const SLOT_MULTIPLIER: usize = 2;

/// Compute queue shape for a given worker count.
///
/// Returns `(num_batches, total_slots)` where `total_slots` is rounded up to a
/// whole number of `BATCH_SIZE` lanes and is at least `2 * num_workers`.
pub fn queue_shape_for_workers(num_workers: usize) -> (usize, usize) {
    assert!(num_workers > 0, "num_workers must be > 0");
    let min_slots = num_workers.saturating_mul(SLOT_MULTIPLIER).max(BATCH_SIZE);
    let num_batches = min_slots.div_ceil(BATCH_SIZE);
    let total_slots = num_batches * BATCH_SIZE;
    (num_batches, total_slots)
}

/// A lock-free queue for batching GPU inference jobs.
///
/// Workers submit observations via callback and receive tickets. When a batch fills,
/// the completing worker triggers GPU dispatch with a zero-copy view of the batch.
pub struct GpuJobQueue<A, D, O>
where
    A: Clone + Default + Send + Sync,
    D: BatchDim,
    O: Copy + Default + Send + Sync,
{
    /// Monotonically increasing counter for slot assignment.
    write_ticket: AtomicU64,
    /// Count of completed writes per batch slot.
    /// When this reaches BATCH_SIZE, the batch is ready for GPU dispatch.
    batch_writes: Box<[AtomicU64]>,

    /// Ticket number at which each batch was completed (end of batch).
    /// Workers check this to know if their result is ready.
    batch_complete: Box<[AtomicU64]>,

    /// Number of batch slots in the ring.
    num_batches: usize,

    /// Total number of observation/output slots.
    total_slots: usize,

    /// Observation storage: shape is `(total_slots, ...obs_shape)`.
    /// Single contiguous allocation for zero-copy batch slicing.
    observations: UnsafeCell<Array<A, D::BatchedDim>>,

    /// Output buffer. Size = `total_slots`.
    outputs: Box<[UnsafeCell<O>]>,

    /// Event for parking threads when waiting for GPU completion.
    completion_event: Event,

    /// Callback invoked when a batch is ready.
    /// Receives the batch slot index, a view of batch observations, and should fill outputs.
    dispatch: Box<dyn Fn(usize, ArrayView<A, D::BatchedDim>, &mut [O]) + Send + Sync>,
}

// SAFETY: The queue is designed for concurrent access:
// - write_ticket ensures each slot is claimed by exactly one writer
// - batch_writes/batch_complete use atomic operations
// - observation slots are only written by their ticket owner, read after batch_complete
// - dispatch is Send + Sync
unsafe impl<A, D, O> Send for GpuJobQueue<A, D, O>
where
    A: Clone + Default + Send + Sync,
    D: BatchDim,
    O: Copy + Default + Send + Sync,
{
}
unsafe impl<A, D, O> Sync for GpuJobQueue<A, D, O>
where
    A: Clone + Default + Send + Sync,
    D: BatchDim,
    O: Copy + Default + Send + Sync,
{
}

impl<A, D, O> GpuJobQueue<A, D, O>
where
    A: Clone + Default + Send + Sync,
    D: BatchDim,
    O: Copy + Default + Send + Sync,
{
    fn compute_queue_shape(num_workers: usize) -> (usize, usize) {
        queue_shape_for_workers(num_workers)
    }

    /// Creates a new job queue with the given observation shape, worker count,
    /// and dispatch callback.
    ///
    /// Queue storage is provisioned to at least `2 * num_workers` slots,
    /// rounded up to a whole number of batches.
    ///
    /// The callback is invoked when a batch of BATCH_SIZE jobs is ready.
    /// It receives a view of the batch observations (shape: BATCH_SIZE x obs_shape)
    /// and should fill the outputs.
    pub fn new<F>(obs_shape: D, num_workers: usize, dispatch: F) -> Self
    where
        F: Fn(usize, ArrayView<A, D::BatchedDim>, &mut [O]) + Send + Sync + 'static,
    {
        let (num_batches, total_slots) = Self::compute_queue_shape(num_workers);

        // Build the batched shape: `(total_slots, ...obs_shape)`
        let full_shape = D::with_batch(total_slots, obs_shape);
        let observations = Array::default(full_shape);

        let outputs: Box<[UnsafeCell<O>]> = (0..total_slots)
            .map(|_| UnsafeCell::new(O::default()))
            .collect();

        let batch_writes = (0..num_batches).map(|_| AtomicU64::new(0)).collect();
        let batch_complete = (0..num_batches).map(|_| AtomicU64::new(0)).collect();

        Self {
            write_ticket: AtomicU64::new(0),
            batch_writes,
            batch_complete,
            num_batches,
            total_slots,
            observations: UnsafeCell::new(observations),
            outputs,
            completion_event: Event::new(),
            dispatch: Box::new(dispatch),
        }
    }

    #[inline]
    pub fn num_batches(&self) -> usize {
        self.num_batches
    }

    #[inline]
    pub fn total_slots(&self) -> usize {
        self.total_slots
    }

    /// Submit a job by writing an observation via callback.
    ///
    /// The callback receives a mutable view into the queue's contiguous storage
    /// for zero-copy observation writing.
    ///
    /// If this submission completes a batch, the current thread will
    /// synchronously dispatch the batch (blocking until complete).
    pub fn submit<F>(&self, write_obs: F) -> u64
    where
        F: FnOnce(ArrayViewMut<A, D>),
    {
        // Claim a slot
        let ticket = self.write_ticket.fetch_add(1, Ordering::Relaxed);
        let slot_idx = (ticket as usize) % self.total_slots;
        let batch_idx = ((ticket as usize) / BATCH_SIZE) % self.num_batches;

        // Get mutable view of our slot and let caller write the observation
        // SAFETY: We own this slot exclusively until we increment batch_writes
        // index_axis_mut on Array<A, D::BatchedDim> returns ArrayViewMut<A, D>
        // because D::BatchedDim::Smaller == D (guaranteed by BatchDim trait)
        let slot_view = unsafe { (*self.observations.get()).index_axis_mut(Axis(0), slot_idx) };
        write_obs(slot_view);

        // AcqRel: Release our write, Acquire if we trigger dispatch to see others' writes
        let writes_in_batch = self.batch_writes[batch_idx].fetch_add(1, Ordering::AcqRel) + 1;

        // If we completed the batch, dispatch it
        if writes_in_batch == BATCH_SIZE as u64 {
            self.dispatch_batch(batch_idx, ticket);
        }

        ticket
    }

    /// Dispatch a completed batch to the GPU.
    fn dispatch_batch(&self, batch_idx: usize, trigger_ticket: u64) {
        let batch_start = batch_idx * BATCH_SIZE;

        // Zero-copy slice of the batch observations
        // SAFETY: All writes to this batch are complete (batch_writes == BATCH_SIZE)
        let obs_array = unsafe { &*self.observations.get() };
        let batch_view =
            obs_array.slice_axis(Axis(0), Slice::from(batch_start..batch_start + BATCH_SIZE));
        debug_assert!(
            batch_view.is_standard_layout(),
            "batch_view should be contiguous for efficient GPU transfer"
        );

        let mut outputs: Vec<O> = vec![O::default(); BATCH_SIZE];

        (self.dispatch)(batch_idx, batch_view, &mut outputs);

        // SAFETY: We're the only one writing outputs for this batch
        for (i, output) in outputs.into_iter().enumerate() {
            unsafe {
                *self.outputs[batch_start + i].get() = output;
            }
        }

        // Calculate the batch end ticket (first ticket of next batch)
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
        let batch_idx = ((ticket as usize) / BATCH_SIZE) % self.num_batches;
        let batch_end_ticket = ((ticket / BATCH_SIZE as u64) + 1) * BATCH_SIZE as u64;

        // Check if this batch is complete
        if self.batch_complete[batch_idx].load(Ordering::Acquire) < batch_end_ticket {
            return None;
        }

        // Batch is complete, return reference to output
        let slot_idx = (ticket as usize) % self.total_slots;
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
    use ndarray::Ix0;
    use std::sync::Arc;

    #[test]
    fn test_single_batch_completion() {
        // Use Ix0 (scalar) for simple tests
        let queue: Arc<GpuJobQueue<u64, Ix0, u64>> = Arc::new(GpuJobQueue::new(
            Ix0(),
            BATCH_SIZE,
            |_batch_idx, inputs, outputs| {
                // Simple transform: output = input * 2
                for (i, input) in inputs.iter().enumerate() {
                    outputs[i] = input * 2;
                }
            },
        ));

        // Submit BATCH_SIZE jobs
        let tickets: Vec<u64> = (0..BATCH_SIZE as u64)
            .map(|i| queue.submit(|mut out| out[()] = i))
            .collect();

        // All should be complete now (last submit triggered dispatch)
        for (i, &ticket) in tickets.iter().enumerate() {
            let result = queue.poll(ticket);
            assert!(result.is_some(), "ticket {} should be ready", ticket);
            assert_eq!(*result.unwrap(), (i as u64) * 2);
        }
    }

    #[test]
    fn test_partial_batch_not_ready() {
        let queue: Arc<GpuJobQueue<u64, Ix0, u64>> = Arc::new(GpuJobQueue::new(
            Ix0(),
            BATCH_SIZE,
            |_batch_idx, inputs, outputs| {
                for (i, input) in inputs.iter().enumerate() {
                    outputs[i] = input * 2;
                }
            },
        ));

        // Submit less than a full batch
        let tickets: Vec<u64> = (0..BATCH_SIZE as u64 - 1)
            .map(|i| queue.submit(|mut out| out[()] = i))
            .collect();

        // None should be ready
        for &ticket in &tickets {
            assert!(
                queue.poll(ticket).is_none(),
                "partial batch should not be ready"
            );
        }

        // Complete the batch
        queue.submit(|mut out| out[()] = BATCH_SIZE as u64 - 1);

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
        let num_jobs = BATCH_SIZE * 3;
        let queue: Arc<GpuJobQueue<u64, Ix0, u64>> = Arc::new(GpuJobQueue::new(
            Ix0(),
            num_jobs,
            |_batch_idx, inputs, outputs| {
                for (i, input) in inputs.iter().enumerate() {
                    outputs[i] = input + 1000;
                }
            },
        ));

        // Submit 3 full batches
        let all_tickets: Vec<u64> = (0..num_jobs as u64)
            .map(|i| queue.submit(|mut out| out[()] = i))
            .collect();

        // All should be ready
        for (i, &ticket) in all_tickets.iter().enumerate() {
            let result = queue.poll(ticket).expect("should be ready");
            assert_eq!(*result, (i as u64) + 1000);
        }
    }

    #[test]
    fn test_batch_slot_reuse() {
        let queue: Arc<GpuJobQueue<u64, Ix0, u64>> = Arc::new(GpuJobQueue::new(
            Ix0(),
            BATCH_SIZE,
            |_batch_idx, inputs, outputs| {
                for (i, input) in inputs.iter().enumerate() {
                    outputs[i] = *input;
                }
            },
        ));

        let total_slots = queue.total_slots();

        // Submit exactly enough jobs to fill every slot once.
        let tickets_round1: Vec<u64> = (0..total_slots as u64)
            .map(|i| queue.submit(|mut out| out[()] = i))
            .collect();

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
        let tickets_round2: Vec<u64> = (total_slots as u64..(total_slots * 2) as u64)
            .map(|i| queue.submit(|mut out| out[()] = i))
            .collect();

        // Read all results from round 2
        for (i, &ticket) in tickets_round2.iter().enumerate() {
            let expected = (total_slots + i) as u64;
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

        let batch_count = Arc::new(AtomicUsize::new(0));
        let num_threads = 4;
        let jobs_per_thread = BATCH_SIZE * 2; // Each thread submits 2 batches worth

        let queue: Arc<GpuJobQueue<u64, Ix0, u64>> = Arc::new(GpuJobQueue::new(
            Ix0(),
            num_threads * jobs_per_thread,
            |_batch_idx, inputs, outputs| {
                for (i, input) in inputs.iter().enumerate() {
                    outputs[i] = input * 2;
                }
            },
        ));

        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let queue = queue.clone();
                let batch_count = batch_count.clone();
                thread::spawn(move || {
                    let base = (thread_id * jobs_per_thread) as u64;
                    let mut results = Vec::new();

                    for i in 0..jobs_per_thread as u64 {
                        let val = base + i;
                        let ticket = queue.submit(|mut out| out[()] = val);
                        results.push((ticket, val));
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
