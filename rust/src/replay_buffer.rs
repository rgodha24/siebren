//! Lock-free replay buffer for concurrent training data storage.

use std::cell::UnsafeCell;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::mem::MaybeUninit;
use std::path::Path;
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

    /// Save buffer to binary file.
    ///
    /// File format (version 2):
    /// - magic: 8 bytes "SIEBREN\0"
    /// - version: u64 (little endian)
    /// - generation_id: u64 (little endian)
    /// - sample_count: u64 (little endian)
    /// - max_notation_len: u64 (little endian) - max length of any notation
    /// - policy_len: u64 (little endian)
    /// - samples: for each sample:
    ///   - notation_len: u64 (little endian) - actual length of this notation
    ///   - notation: [u8; max_notation_len] - UTF-8 bytes, padded with zeros
    ///   - policy: [f32; policy_len] (little endian)
    ///   - value: f32 (little endian)
    pub fn save(&self, path: &Path, generation_id: u64, policy_len: usize) -> io::Result<()> {
        const MAGIC: &[u8; 8] = b"SIEBREN\0";
        const VERSION: u64 = 2;

        assert_eq!(
            self.writers.load(Ordering::Acquire),
            0,
            "cannot save while writers are active"
        );

        let (tail, head) = self.valid_range();
        let count = (head - tail) as u64;

        // Find max notation length
        let mut max_notation_len = 0usize;
        for idx in tail..head {
            let slot = idx as usize % self.capacity;
            let sample = unsafe { (*self.data[slot].get()).assume_init_ref() };
            max_notation_len = max_notation_len.max(sample.notation.len());
        }

        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Write header
        writer.write_all(MAGIC)?;
        writer.write_all(&VERSION.to_le_bytes())?;
        writer.write_all(&generation_id.to_le_bytes())?;
        writer.write_all(&count.to_le_bytes())?;
        writer.write_all(&(max_notation_len as u64).to_le_bytes())?;
        writer.write_all(&(policy_len as u64).to_le_bytes())?;

        // Pre-allocate padding buffer
        let mut notation_buf = vec![0u8; max_notation_len];

        // Write samples
        for idx in tail..head {
            let slot = idx as usize % self.capacity;
            let sample = unsafe { (*self.data[slot].get()).assume_init_ref() };

            // Write notation length and padded notation
            let notation_bytes = sample.notation.as_bytes();
            writer.write_all(&(notation_bytes.len() as u64).to_le_bytes())?;
            notation_buf[..notation_bytes.len()].copy_from_slice(notation_bytes);
            notation_buf[notation_bytes.len()..].fill(0);
            writer.write_all(&notation_buf)?;

            // Write policy
            for &p in &sample.policy {
                writer.write_all(&p.to_le_bytes())?;
            }

            // Write value
            writer.write_all(&sample.value.to_le_bytes())?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Load samples from binary file.
    ///
    /// Returns (samples_loaded, generation_id).
    /// Panics if writers are active.
    pub fn load(&self, path: &Path) -> io::Result<(usize, u64)> {
        const MAGIC: &[u8; 8] = b"SIEBREN\0";

        assert_eq!(
            self.writers.load(Ordering::Acquire),
            0,
            "cannot load while writers are active"
        );

        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Read and validate magic
        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid magic bytes",
            ));
        }

        // Read and validate version
        let mut buf8 = [0u8; 8];
        reader.read_exact(&mut buf8)?;
        let version = u64::from_le_bytes(buf8);
        if version != 2 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported version: {version}, expected 2"),
            ));
        }

        // Read header fields
        reader.read_exact(&mut buf8)?;
        let generation_id = u64::from_le_bytes(buf8);

        reader.read_exact(&mut buf8)?;
        let sample_count = u64::from_le_bytes(buf8) as usize;

        reader.read_exact(&mut buf8)?;
        let max_notation_len = u64::from_le_bytes(buf8) as usize;

        reader.read_exact(&mut buf8)?;
        let policy_len = u64::from_le_bytes(buf8) as usize;

        // Pre-allocate buffers
        let mut notation_buf = vec![0u8; max_notation_len];
        let mut policy_buf = vec![0u8; policy_len * 4];
        let mut buf4 = [0u8; 4];

        // Reserve space and read samples
        let mut guard = self.reserve(sample_count);

        for _ in 0..sample_count {
            // Read notation length
            reader.read_exact(&mut buf8)?;
            let notation_len = u64::from_le_bytes(buf8) as usize;

            if notation_len > max_notation_len {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "notation length {notation_len} exceeds max_notation_len {max_notation_len}"
                    ),
                ));
            }

            // Read padded notation
            reader.read_exact(&mut notation_buf)?;
            let notation =
                String::from_utf8(notation_buf[..notation_len].to_vec()).map_err(|e| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("invalid UTF-8 in notation: {e}"),
                    )
                })?;

            // Read policy
            reader.read_exact(&mut policy_buf)?;
            let policy: Vec<f32> = policy_buf
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect();

            // Read value
            reader.read_exact(&mut buf4)?;
            let value = f32::from_le_bytes(buf4);

            guard.push(Sample {
                notation,
                policy,
                value,
            });
        }

        Ok((sample_count, generation_id))
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

    #[test]
    fn test_save_load_roundtrip() {
        let buffer = ReplayBuffer::new(100);
        let policy_len = 9;
        let generation_id = 42u64;

        // Add samples
        {
            let mut guard = buffer.reserve(5);
            guard.extend((0..5).map(make_sample));
        }

        // Save to temp file
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_replay_buffer.bin");

        buffer.save(&path, generation_id, policy_len).unwrap();

        // Load into new buffer
        let buffer2 = ReplayBuffer::new(100);
        let (loaded_count, loaded_gen) = buffer2.load(&path).unwrap();

        assert_eq!(loaded_count, 5);
        assert_eq!(loaded_gen, generation_id);
        assert_eq!(buffer2.len(), 5);

        // Verify samples match
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let original = buffer.sample(5, &mut rng);
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let loaded = buffer2.sample(5, &mut rng);

        for (orig, load) in original.iter().zip(loaded.iter()) {
            assert_eq!(orig.notation, load.notation);
            assert_eq!(orig.policy, load.policy);
            assert!((orig.value - load.value).abs() < 1e-6);
        }

        // Cleanup
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_save_load_generation_id() {
        let buffer = ReplayBuffer::new(100);

        {
            let mut guard = buffer.reserve(1);
            guard.push(make_sample(0));
        }

        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_replay_gen_id.bin");

        // Save with specific generation ID
        let gen_id = 12345u64;
        buffer.save(&path, gen_id, 9).unwrap();

        // Load and verify generation ID is preserved
        let buffer2 = ReplayBuffer::new(100);
        let (_, loaded_gen) = buffer2.load(&path).unwrap();
        assert_eq!(loaded_gen, gen_id);

        // Cleanup
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_save_load_varying_notation_lengths() {
        let buffer = ReplayBuffer::new(100);

        // Add samples with different notation lengths
        {
            let mut guard = buffer.reserve(3);
            guard.push(Sample {
                notation: "A".to_string(),
                policy: vec![0.1; 9],
                value: 0.5,
            });
            guard.push(Sample {
                notation: "ABCDEFGHIJ".to_string(),
                policy: vec![0.2; 9],
                value: 0.6,
            });
            guard.push(Sample {
                notation: "XYZ".to_string(),
                policy: vec![0.3; 9],
                value: 0.7,
            });
        }

        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_replay_varying_notation.bin");

        buffer.save(&path, 1, 9).unwrap();

        let buffer2 = ReplayBuffer::new(100);
        let (count, _) = buffer2.load(&path).unwrap();
        assert_eq!(count, 3);

        // Verify all notations preserved correctly
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let original = buffer.sample(3, &mut rng);
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let loaded = buffer2.sample(3, &mut rng);

        for (orig, load) in original.iter().zip(loaded.iter()) {
            assert_eq!(orig.notation, load.notation);
        }

        // Cleanup
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_save_empty_buffer() {
        let buffer = ReplayBuffer::new(100);

        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_replay_empty.bin");

        buffer.save(&path, 0, 9).unwrap();

        let buffer2 = ReplayBuffer::new(100);
        let (count, gen) = buffer2.load(&path).unwrap();
        assert_eq!(count, 0);
        assert_eq!(gen, 0);
        assert_eq!(buffer2.len(), 0);

        // Cleanup
        std::fs::remove_file(&path).ok();
    }
}
