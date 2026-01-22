//! Lock-free replay buffer for concurrent training data storage.

use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicU64, Ordering};

use rand::seq::index::sample;
use rand::Rng;

/// A training sample stored in the replay buffer.
#[derive(Clone, Debug)]
pub struct Sample {
    pub notation: String,
    pub policy: Vec<f32>,
    pub value: f32,
}

/// Lock-free ring buffer for storing training samples.
pub struct ReplayBuffer {
    data: Box<[UnsafeCell<MaybeUninit<Sample>>]>,
    capacity: usize,
    head: AtomicU64,
    writers: AtomicU64,
}

unsafe impl Sync for ReplayBuffer {}
unsafe impl Send for ReplayBuffer {}

/// RAII guard for writing to reserved slots.
pub struct ReserveGuard<'a> {
    buffer: &'a ReplayBuffer,
    start: u64,
    len: usize,
    written: usize,
}

impl<'a> ReserveGuard<'a> {
    #[inline]
    pub fn push(&mut self, sample: Sample) {
        assert!(self.written < self.len, "wrote more samples than reserved");
        let idx = (self.start + self.written as u64) as usize % self.buffer.capacity;
        unsafe {
            (*self.buffer.data[idx].get()).write(sample);
        }
        self.written += 1;
    }

    pub fn extend(&mut self, samples: impl IntoIterator<Item = Sample>) {
        for sample in samples {
            self.push(sample);
        }
    }
}

impl Drop for ReserveGuard<'_> {
    fn drop(&mut self) {
        self.buffer.writers.fetch_sub(1, Ordering::Release);
    }
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        let data: Vec<UnsafeCell<MaybeUninit<Sample>>> = (0..capacity)
            .map(|_| UnsafeCell::new(MaybeUninit::uninit()))
            .collect();
        Self {
            data: data.into_boxed_slice(),
            capacity,
            head: AtomicU64::new(0),
            writers: AtomicU64::new(0),
        }
    }

    pub fn reserve(&self, n: usize) -> ReserveGuard<'_> {
        self.writers.fetch_add(1, Ordering::Acquire);
        let start = self.head.fetch_add(n as u64, Ordering::AcqRel);
        ReserveGuard {
            buffer: self,
            start,
            len: n,
            written: 0,
        }
    }

    #[inline]
    fn valid_range(&self) -> (u64, u64) {
        let head = self.head.load(Ordering::Acquire);
        let tail = head.saturating_sub(self.capacity as u64);
        (tail, head)
    }

    #[inline]
    pub fn len(&self) -> usize {
        let (tail, head) = self.valid_range();
        (head - tail) as usize
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Sample `n` items uniformly. Panics if writers are active.
    pub fn sample(&self, n: usize, rng: &mut impl Rng) -> Vec<Sample> {
        assert_eq!(
            self.writers.load(Ordering::Acquire),
            0,
            "cannot sample while writers are active"
        );

        let (tail, head) = self.valid_range();
        let count = (head - tail) as usize;
        if count == 0 || n == 0 {
            return Vec::new();
        }

        let indices = sample(rng, count, n.min(count));
        indices
            .iter()
            .map(|offset| {
                let idx = tail + offset as u64;
                let slot = idx as usize % self.capacity;
                unsafe { (*self.data[slot].get()).assume_init_ref().clone() }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use std::sync::Arc;

    fn make_sample(id: usize) -> Sample {
        Sample {
            notation: format!("_________|A"),
            policy: vec![id as f32 / 10.0; 9],
            value: id as f32 / 100.0,
        }
    }

    #[test]
    fn test_reserve_guard_push() {
        let buffer = ReplayBuffer::new(100);

        {
            let mut guard = buffer.reserve(3);
            guard.push(make_sample(0));
            guard.push(make_sample(1));
            guard.push(make_sample(2));
        }

        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer.writers.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_reserve_guard_extend() {
        let buffer = ReplayBuffer::new(100);

        {
            let mut guard = buffer.reserve(5);
            guard.extend((0..5).map(make_sample));
        }

        assert_eq!(buffer.len(), 5);
    }

    #[test]
    fn test_sample() {
        let buffer = ReplayBuffer::new(100);
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        {
            let mut guard = buffer.reserve(10);
            guard.extend((0..10).map(make_sample));
        }

        let samples = buffer.sample(5, &mut rng);
        assert_eq!(samples.len(), 5);
    }

    #[test]
    fn test_concurrent_writes() {
        let buffer = Arc::new(ReplayBuffer::new(1000));
        let num_threads = 4;
        let samples_per_thread = 100;

        std::thread::scope(|s| {
            for _ in 0..num_threads {
                let buffer = Arc::clone(&buffer);
                s.spawn(move || {
                    let mut guard = buffer.reserve(samples_per_thread);
                    guard.extend((0..samples_per_thread).map(make_sample));
                });
            }
        });

        assert_eq!(buffer.len(), num_threads * samples_per_thread);
        assert_eq!(buffer.writers.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_buffer_wraparound() {
        let buffer = ReplayBuffer::new(10);

        for i in 0..25 {
            let mut guard = buffer.reserve(1);
            guard.push(make_sample(i));
        }

        assert_eq!(buffer.len(), 10);
        let (tail, head) = buffer.valid_range();
        assert_eq!(tail, 15);
        assert_eq!(head, 25);
    }

    #[test]
    #[should_panic(expected = "cannot sample while writers are active")]
    fn test_sample_during_write_panics() {
        let buffer = ReplayBuffer::new(100);
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let _guard = buffer.reserve(5);
        let _ = buffer.sample(5, &mut rng);
    }
}
