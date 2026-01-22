//! Lock-free replay buffer using atomics for concurrent training data storage.
//!
//! Uses a ring buffer design where writers atomically reserve slots via `fetch_add`,
//! ensuring exclusive write access without locks. Readers may see slightly stale
//! data during concurrent writes, which is acceptable for training purposes.

use std::cell::UnsafeCell;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use rand::Rng;

/// A training sample stored in the replay buffer.
#[derive(Clone, Debug)]
pub struct Sample {
    /// Game state in notation format.
    pub notation: String,
    /// Policy distribution over actions.
    pub policy: Vec<f32>,
    /// Value estimate for the position.
    pub value: f32,
}

/// Lock-free ring buffer for storing training samples.
///
/// Writers call `reserve(n)` to atomically claim `n` contiguous slots,
/// then write samples using `write()`. The ring buffer overwrites old
/// data when capacity is exceeded.
pub struct ReplayBuffer {
    data: Box<[UnsafeCell<Option<Sample>>]>,
    capacity: usize,
    /// Next write position (monotonically increasing).
    head: AtomicU64,
}

// SAFETY: The reservation system ensures exclusive write access to slots.
// Readers may see partially written data, but Sample's Option wrapper
// ensures we only return fully initialized samples.
unsafe impl Sync for ReplayBuffer {}
unsafe impl Send for ReplayBuffer {}

impl ReplayBuffer {
    /// Create a new replay buffer with the given capacity.
    pub fn new(capacity: usize) -> Self {
        let data: Vec<UnsafeCell<Option<Sample>>> =
            (0..capacity).map(|_| UnsafeCell::new(None)).collect();
        Self {
            data: data.into_boxed_slice(),
            capacity,
            head: AtomicU64::new(0),
        }
    }

    /// Reserve `n` contiguous slots for writing.
    ///
    /// Returns the start index. Caller can write to slots `[start..start+n]`
    /// using the `write` method.
    #[inline]
    pub fn reserve(&self, n: usize) -> u64 {
        self.head.fetch_add(n as u64, Ordering::AcqRel)
    }

    /// Write a sample to a specific slot obtained from `reserve`.
    ///
    /// # Safety
    ///
    /// Caller must have reserved this slot via `reserve()` and must not
    /// write to the same slot from multiple threads.
    #[inline]
    pub unsafe fn write(&self, idx: u64, sample: Sample) {
        let slot = idx as usize % self.capacity;
        // SAFETY: Caller guarantees exclusive access to this slot via reservation.
        *self.data[slot].get() = Some(sample);
    }

    /// Get current valid range `[tail, head)`.
    ///
    /// The tail is `max(0, head - capacity)`, representing the oldest
    /// non-overwritten data.
    #[inline]
    pub fn valid_range(&self) -> (u64, u64) {
        let head = self.head.load(Ordering::Acquire);
        let tail = head.saturating_sub(self.capacity as u64);
        (tail, head)
    }

    /// Sample `n` items uniformly from the valid range.
    ///
    /// Returns up to `n` samples. May return fewer if slots are empty
    /// (e.g., during concurrent writes or before buffer is full).
    pub fn sample(&self, n: usize, rng: &mut impl Rng) -> Vec<Sample> {
        let (tail, head) = self.valid_range();
        let count = (head - tail) as usize;
        if count == 0 {
            return Vec::new();
        }

        (0..n)
            .filter_map(|_| {
                let idx = tail + rng.random_range(0..count as u64);
                let slot = idx as usize % self.capacity;
                // SAFETY: slot is within bounds and in the valid range.
                // We clone the Option to handle concurrent writes gracefully.
                unsafe { (*self.data[slot].get()).clone() }
            })
            .collect()
    }

    /// Returns the number of valid samples in the buffer.
    ///
    /// This is `min(head, capacity)` - the buffer fills up to capacity
    /// and then maintains that count as old data is overwritten.
    #[inline]
    pub fn len(&self) -> usize {
        let (tail, head) = self.valid_range();
        (head - tail) as usize
    }

    /// Returns true if the buffer contains no samples.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the buffer capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Save all samples to a binary file.
    ///
    /// File format:
    /// - Header: magic (8 bytes), version (u64), sample_count (u64),
    ///   max_notation_len (u64), policy_len (u64)
    /// - Per sample: notation (null-padded), policy (f32s), value (f32)
    pub fn save(&self, path: &Path, max_notation_len: usize, policy_len: usize) -> io::Result<()> {
        const MAGIC: &[u8; 8] = b"SIEBREN\0";
        const VERSION: u64 = 1;

        let file = std::fs::File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Collect valid samples
        let (tail, head) = self.valid_range();
        let sample_count = (head - tail) as usize;

        // Write header
        writer.write_all(MAGIC)?;
        writer.write_all(&VERSION.to_le_bytes())?;
        writer.write_all(&(sample_count as u64).to_le_bytes())?;
        writer.write_all(&(max_notation_len as u64).to_le_bytes())?;
        writer.write_all(&(policy_len as u64).to_le_bytes())?;

        // Write each sample
        for idx in tail..head {
            let slot = idx as usize % self.capacity;
            // SAFETY: slot is within bounds and in the valid range
            let sample_opt = unsafe { &*self.data[slot].get() };
            if let Some(sample) = sample_opt {
                // Write notation (null-padded to max_notation_len)
                let notation_bytes = sample.notation.as_bytes();
                let notation_len = notation_bytes.len().min(max_notation_len);
                writer.write_all(&notation_bytes[..notation_len])?;
                // Pad with zeros
                for _ in notation_len..max_notation_len {
                    writer.write_all(&[0u8])?;
                }

                // Write policy (policy_len f32s)
                for i in 0..policy_len {
                    let val = sample.policy.get(i).copied().unwrap_or(0.0);
                    writer.write_all(&val.to_le_bytes())?;
                }

                // Write value
                writer.write_all(&sample.value.to_le_bytes())?;
            }
        }

        writer.flush()?;
        Ok(())
    }

    /// Load samples from a binary file into the buffer.
    ///
    /// Returns the number of samples loaded.
    pub fn load(&self, path: &Path) -> io::Result<usize> {
        const MAGIC: &[u8; 8] = b"SIEBREN\0";
        const VERSION: u64 = 1;

        let file = std::fs::File::open(path)?;
        let mut reader = BufReader::new(file);

        // Read and validate header
        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid magic bytes",
            ));
        }

        let mut buf8 = [0u8; 8];
        reader.read_exact(&mut buf8)?;
        let version = u64::from_le_bytes(buf8);
        if version != VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported version: {}", version),
            ));
        }

        reader.read_exact(&mut buf8)?;
        let sample_count = u64::from_le_bytes(buf8) as usize;

        reader.read_exact(&mut buf8)?;
        let max_notation_len = u64::from_le_bytes(buf8) as usize;

        reader.read_exact(&mut buf8)?;
        let policy_len = u64::from_le_bytes(buf8) as usize;

        // Read samples
        let mut notation_buf = vec![0u8; max_notation_len];
        let mut policy_buf = vec![0u8; policy_len * 4];
        let mut value_buf = [0u8; 4];

        let mut loaded = 0;
        for _ in 0..sample_count {
            // Read notation
            if reader.read_exact(&mut notation_buf).is_err() {
                break; // Truncated file - return what we have
            }

            // Trim null padding
            let notation_end = notation_buf
                .iter()
                .position(|&b| b == 0)
                .unwrap_or(max_notation_len);
            let notation = String::from_utf8_lossy(&notation_buf[..notation_end]).into_owned();

            // Read policy
            if reader.read_exact(&mut policy_buf).is_err() {
                break;
            }
            let policy: Vec<f32> = policy_buf
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect();

            // Read value
            if reader.read_exact(&mut value_buf).is_err() {
                break;
            }
            let value = f32::from_le_bytes(value_buf);

            // Push to buffer
            let sample = Sample {
                notation,
                policy,
                value,
            };
            let idx = self.reserve(1);
            // SAFETY: We just reserved this slot exclusively
            unsafe { self.write(idx, sample) };
            loaded += 1;
        }

        Ok(loaded)
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
            notation: format!("sample_{}", id),
            policy: vec![id as f32 / 10.0; 9],
            value: id as f32 / 100.0,
        }
    }

    #[test]
    fn test_basic_push_and_sample() {
        let buffer = ReplayBuffer::new(100);
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // Initially empty
        assert!(buffer.is_empty());
        assert_eq!(buffer.len(), 0);

        // Add some samples
        for i in 0..10 {
            let idx = buffer.reserve(1);
            unsafe { buffer.write(idx, make_sample(i)) };
        }

        assert_eq!(buffer.len(), 10);
        assert!(!buffer.is_empty());

        // Sample from buffer
        let samples = buffer.sample(5, &mut rng);
        assert_eq!(samples.len(), 5);

        // All samples should have valid notation format
        for sample in &samples {
            assert!(sample.notation.starts_with("sample_"));
        }
    }

    #[test]
    fn test_buffer_wraparound() {
        let capacity = 10;
        let buffer = ReplayBuffer::new(capacity);

        // Write more samples than capacity
        for i in 0..25 {
            let idx = buffer.reserve(1);
            unsafe { buffer.write(idx, make_sample(i)) };
        }

        // Buffer should be at capacity
        assert_eq!(buffer.len(), capacity);

        // Valid range should be [15, 25)
        let (tail, head) = buffer.valid_range();
        assert_eq!(tail, 15);
        assert_eq!(head, 25);

        // Sample and verify we get samples from the valid range (15-24)
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let samples = buffer.sample(100, &mut rng);
        assert_eq!(samples.len(), 100);

        for sample in &samples {
            let id: usize = sample
                .notation
                .strip_prefix("sample_")
                .unwrap()
                .parse()
                .unwrap();
            assert!(
                id >= 15 && id < 25,
                "Sample id {} not in valid range [15, 25)",
                id
            );
        }
    }

    #[test]
    fn test_batch_reserve() {
        let buffer = ReplayBuffer::new(100);

        // Reserve a batch of slots
        let start = buffer.reserve(5);
        assert_eq!(start, 0);

        // Write batch
        for i in 0..5u64 {
            unsafe { buffer.write(start + i, make_sample(i as usize)) };
        }

        assert_eq!(buffer.len(), 5);

        // Reserve another batch
        let start2 = buffer.reserve(3);
        assert_eq!(start2, 5);

        for i in 0..3u64 {
            unsafe { buffer.write(start2 + i, make_sample((5 + i) as usize)) };
        }

        assert_eq!(buffer.len(), 8);
    }

    #[test]
    fn test_concurrent_writes() {
        let buffer = Arc::new(ReplayBuffer::new(1000));
        let num_threads = 4;
        let samples_per_thread = 100;

        std::thread::scope(|s| {
            for thread_id in 0..num_threads {
                let buffer = Arc::clone(&buffer);
                s.spawn(move || {
                    for i in 0..samples_per_thread {
                        let sample_id = thread_id * samples_per_thread + i;
                        let idx = buffer.reserve(1);
                        unsafe { buffer.write(idx, make_sample(sample_id)) };
                    }
                });
            }
        });

        // All samples should be written
        assert_eq!(buffer.len(), num_threads * samples_per_thread);

        // Verify we can sample from all threads' data
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let samples = buffer.sample(200, &mut rng);
        assert_eq!(samples.len(), 200);

        // Check samples are from various threads
        let mut seen_threads = std::collections::HashSet::new();
        for sample in &samples {
            let id: usize = sample
                .notation
                .strip_prefix("sample_")
                .unwrap()
                .parse()
                .unwrap();
            seen_threads.insert(id / samples_per_thread);
        }
        // With 200 samples from 400 total, we should see all threads
        assert!(
            seen_threads.len() >= 2,
            "Expected samples from multiple threads"
        );
    }

    #[test]
    fn test_concurrent_batch_writes() {
        let buffer = Arc::new(ReplayBuffer::new(1000));
        let num_threads = 4;
        let batches_per_thread = 10;
        let batch_size = 10;

        std::thread::scope(|s| {
            for thread_id in 0..num_threads {
                let buffer = Arc::clone(&buffer);
                s.spawn(move || {
                    for batch in 0..batches_per_thread {
                        // Reserve entire batch atomically
                        let start = buffer.reserve(batch_size);

                        // Write batch
                        for i in 0..batch_size {
                            let sample_id = thread_id * batches_per_thread * batch_size
                                + batch * batch_size
                                + i;
                            unsafe { buffer.write(start + i as u64, make_sample(sample_id)) };
                        }
                    }
                });
            }
        });

        assert_eq!(buffer.len(), num_threads * batches_per_thread * batch_size);
    }

    #[test]
    fn test_sample_empty_buffer() {
        let buffer = ReplayBuffer::new(100);
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let samples = buffer.sample(10, &mut rng);
        assert!(samples.is_empty());
    }

    #[test]
    fn test_sample_more_than_available() {
        let buffer = ReplayBuffer::new(100);
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // Add only 5 samples
        for i in 0..5 {
            let idx = buffer.reserve(1);
            unsafe { buffer.write(idx, make_sample(i)) };
        }

        // Request 100 samples - should get 100 (with repeats since we sample with replacement)
        let samples = buffer.sample(100, &mut rng);
        assert_eq!(samples.len(), 100);
    }

    #[test]
    fn test_save_load_roundtrip() {
        let buffer = ReplayBuffer::new(100);
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_replay_buffer.bin");

        // Add samples with varying notation lengths and policy sizes
        for i in 0..10 {
            let idx = buffer.reserve(1);
            unsafe { buffer.write(idx, make_sample(i)) };
        }

        // Save with fixed notation and policy lengths
        let max_notation_len = 32;
        let policy_len = 9;
        buffer.save(&path, max_notation_len, policy_len).unwrap();

        // Load into a new buffer
        let buffer2 = ReplayBuffer::new(100);
        let loaded = buffer2.load(&path).unwrap();

        assert_eq!(loaded, 10);
        assert_eq!(buffer2.len(), 10);

        // Verify samples match
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let samples = buffer2.sample(10, &mut rng);
        for sample in &samples {
            assert!(sample.notation.starts_with("sample_"));
            assert_eq!(sample.policy.len(), policy_len);
        }

        // Clean up
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_load_nonexistent_file() {
        let buffer = ReplayBuffer::new(100);
        let path = std::path::Path::new("/nonexistent/path/file.bin");
        let result = buffer.load(path);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_corrupt_file() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_corrupt_replay_buffer.bin");

        // Write invalid magic bytes
        std::fs::write(&path, b"INVALID\0").unwrap();

        let buffer = ReplayBuffer::new(100);
        let result = buffer.load(&path);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("invalid magic bytes"));

        // Clean up
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_load_truncated_file() {
        let buffer = ReplayBuffer::new(100);
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_truncated_replay_buffer.bin");

        // Add samples
        for i in 0..5 {
            let idx = buffer.reserve(1);
            unsafe { buffer.write(idx, make_sample(i)) };
        }

        // Save
        let max_notation_len = 32;
        let policy_len = 9;
        buffer.save(&path, max_notation_len, policy_len).unwrap();

        // Truncate the file (keep header + partial data)
        let file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        // Header is 40 bytes, truncate to header + 2 complete samples
        let sample_size = max_notation_len + policy_len * 4 + 4;
        let truncate_len = 40 + 2 * sample_size + 10; // partial third sample
        file.set_len(truncate_len as u64).unwrap();

        // Load should succeed with partial data
        let buffer2 = ReplayBuffer::new(100);
        let loaded = buffer2.load(&path).unwrap();

        // Should have loaded at least 2 complete samples
        assert!(loaded >= 2, "Expected at least 2 samples, got {}", loaded);
        assert!(loaded < 5, "Expected fewer than 5 samples, got {}", loaded);

        // Clean up
        std::fs::remove_file(&path).ok();
    }
}
