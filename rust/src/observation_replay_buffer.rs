//! Lock-free ring buffer for contiguous observation replay storage.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU64, Ordering};

use ndarray::{Array, Array2, ArrayViewMut, Axis};
use rand::seq::index::sample;
use rand::Rng;

use crate::BatchDim;

/// Batched sample output from [`ObservationReplayBuffer::sample`].
pub struct ObservationSampleBatch<A, D, const NUM_ACTIONS: usize>
where
    A: Clone + Default + Send + Sync,
    D: BatchDim,
{
    pub observations: Array<A, D::BatchedDim>,
    pub policies: Array2<f32>,
    pub values: Vec<f32>,
}

/// Lock-free ring buffer for storing observations, policies, and values.
///
/// Observations and policies are kept in single contiguous arrays:
/// - observations: `(capacity, ...obs_shape)`
/// - policies: `(capacity, NUM_ACTIONS)`
/// - values: `(capacity,)`
pub struct ObservationReplayBuffer<A, D, const NUM_ACTIONS: usize>
where
    A: Clone + Default + Send + Sync,
    D: BatchDim,
{
    observations: UnsafeCell<Array<A, D::BatchedDim>>,
    policies: UnsafeCell<Array2<f32>>,
    values: UnsafeCell<Vec<f32>>,
    capacity: usize,
    obs_shape: D,
    obs_elems_per_sample: usize,
    head: AtomicU64,
    writers: AtomicU64,
}

// SAFETY: Each writer reserves unique slots through an atomic ticket and writes
// only to its owned slots until drop. Readers require no active writers.
unsafe impl<A, D, const NUM_ACTIONS: usize> Sync for ObservationReplayBuffer<A, D, NUM_ACTIONS>
where
    A: Clone + Default + Send + Sync,
    D: BatchDim,
{
}
unsafe impl<A, D, const NUM_ACTIONS: usize> Send for ObservationReplayBuffer<A, D, NUM_ACTIONS>
where
    A: Clone + Default + Send + Sync,
    D: BatchDim,
{
}

/// RAII guard for writing to reserved slots.
pub struct ReserveGuard<'a, A, D, const NUM_ACTIONS: usize>
where
    A: Clone + Default + Send + Sync,
    D: BatchDim,
{
    buffer: &'a ObservationReplayBuffer<A, D, NUM_ACTIONS>,
    start: u64,
    len: usize,
    written: usize,
}

impl<'a, A, D, const NUM_ACTIONS: usize> ReserveGuard<'a, A, D, NUM_ACTIONS>
where
    A: Clone + Default + Send + Sync,
    D: BatchDim,
{
    /// Push one sample by writing observation directly into the reserved slot.
    #[inline]
    pub fn push_with_observation<F>(&mut self, policy: &[f32], value: f32, write_observation: F)
    where
        F: FnOnce(ArrayViewMut<A, D>),
    {
        assert!(self.written < self.len, "wrote more samples than reserved");
        assert_eq!(
            policy.len(),
            NUM_ACTIONS,
            "policy length must match NUM_ACTIONS"
        );

        let idx = (self.start + self.written as u64) as usize % self.buffer.capacity;

        unsafe {
            let slot_view = (*self.buffer.observations.get()).index_axis_mut(Axis(0), idx);
            write_observation(slot_view);

            let policy_storage = &mut *self.buffer.policies.get();
            let policy_slice = policy_storage
                .as_slice_memory_order_mut()
                .expect("policy storage must be contiguous");
            let policy_start = idx * NUM_ACTIONS;
            policy_slice[policy_start..policy_start + NUM_ACTIONS].copy_from_slice(policy);

            (&mut *self.buffer.values.get())[idx] = value;
        }

        self.written += 1;
    }

    /// Push one sample from flattened observation data.
    pub fn push(&mut self, observation: &[A], policy: &[f32], value: f32) {
        assert_eq!(
            observation.len(),
            self.buffer.obs_elems_per_sample,
            "observation length must match env observation size"
        );

        self.push_with_observation(policy, value, |mut out| {
            for (dst, src) in out.iter_mut().zip(observation.iter()) {
                *dst = src.clone();
            }
        });
    }
}

impl<A, D, const NUM_ACTIONS: usize> Drop for ReserveGuard<'_, A, D, NUM_ACTIONS>
where
    A: Clone + Default + Send + Sync,
    D: BatchDim,
{
    fn drop(&mut self) {
        self.buffer.writers.fetch_sub(1, Ordering::Release);
    }
}

impl<A, D, const NUM_ACTIONS: usize> ObservationReplayBuffer<A, D, NUM_ACTIONS>
where
    A: Clone + Default + Send + Sync,
    D: BatchDim,
{
    pub fn new(capacity: usize, obs_shape: D) -> Self {
        assert!(capacity > 0, "capacity must be > 0");
        let obs_elems_per_sample = obs_shape.clone().size();

        Self {
            observations: UnsafeCell::new(Array::default(D::with_batch(
                capacity,
                obs_shape.clone(),
            ))),
            policies: UnsafeCell::new(Array2::<f32>::zeros((capacity, NUM_ACTIONS))),
            values: UnsafeCell::new(vec![0.0; capacity]),
            capacity,
            obs_shape,
            obs_elems_per_sample,
            head: AtomicU64::new(0),
            writers: AtomicU64::new(0),
        }
    }

    pub fn reserve(&self, n: usize) -> ReserveGuard<'_, A, D, NUM_ACTIONS> {
        assert!(
            n <= self.capacity,
            "cannot reserve more samples than buffer capacity"
        );

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
    pub fn sample(
        &self,
        n: usize,
        rng: &mut impl Rng,
    ) -> ObservationSampleBatch<A, D, NUM_ACTIONS> {
        assert_eq!(
            self.writers.load(Ordering::Acquire),
            0,
            "cannot sample while writers are active"
        );

        let (tail, head) = self.valid_range();
        let count = (head - tail) as usize;
        if count == 0 || n == 0 {
            return ObservationSampleBatch {
                observations: Array::default(D::with_batch(0, self.obs_shape.clone())),
                policies: Array2::<f32>::zeros((0, NUM_ACTIONS)),
                values: Vec::new(),
            };
        }

        let sample_count = n.min(count);
        let indices = sample(rng, count, sample_count);

        let observations = unsafe { &*self.observations.get() };
        let observation_slice = observations
            .as_slice_memory_order()
            .expect("observation storage must be contiguous");

        let policies = unsafe { &*self.policies.get() };
        let policy_slice = policies
            .as_slice_memory_order()
            .expect("policy storage must be contiguous");

        let values = unsafe { &*self.values.get() };

        let mut obs_data = Vec::with_capacity(sample_count * self.obs_elems_per_sample);
        let mut policy_data = Vec::with_capacity(sample_count * NUM_ACTIONS);
        let mut value_data = Vec::with_capacity(sample_count);

        for offset in indices.iter() {
            let idx = (tail + offset as u64) as usize % self.capacity;

            let obs_start = idx * self.obs_elems_per_sample;
            obs_data.extend_from_slice(
                &observation_slice[obs_start..obs_start + self.obs_elems_per_sample],
            );

            let policy_start = idx * NUM_ACTIONS;
            policy_data.extend_from_slice(&policy_slice[policy_start..policy_start + NUM_ACTIONS]);

            value_data.push(values[idx]);
        }

        let observations = Array::from_shape_vec(
            D::with_batch(sample_count, self.obs_shape.clone()),
            obs_data,
        )
        .expect("sample observation shape mismatch");
        let policies = Array2::from_shape_vec((sample_count, NUM_ACTIONS), policy_data)
            .expect("shape mismatch");

        ObservationSampleBatch {
            observations,
            policies,
            values: value_data,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Ix1;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_push_and_sample() {
        let buffer = ObservationReplayBuffer::<i8, Ix1, 3>::new(10, Ix1(4));

        {
            let mut guard = buffer.reserve(2);
            guard.push(&[1, 2, 3, 4], &[0.1, 0.2, 0.7], 0.5);
            guard.push(&[5, 6, 7, 8], &[0.6, 0.2, 0.2], -0.5);
        }

        assert_eq!(buffer.len(), 2);

        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let batch = buffer.sample(2, &mut rng);
        assert_eq!(batch.observations.shape(), &[2, 4]);
        assert_eq!(batch.policies.shape(), &[2, 3]);
        assert_eq!(batch.values.len(), 2);
    }

    #[test]
    fn test_wraparound_len() {
        let buffer = ObservationReplayBuffer::<i8, Ix1, 2>::new(3, Ix1(2));

        for i in 0..10 {
            let mut guard = buffer.reserve(1);
            guard.push(&[i as i8, (i + 1) as i8], &[0.5, 0.5], i as f32);
        }

        assert_eq!(buffer.len(), 3);
    }

    #[test]
    #[should_panic(expected = "cannot sample while writers are active")]
    fn test_sample_during_write_panics() {
        let buffer = ObservationReplayBuffer::<i8, Ix1, 2>::new(4, Ix1(2));
        let _guard = buffer.reserve(1);

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let _ = buffer.sample(1, &mut rng);
    }
}
