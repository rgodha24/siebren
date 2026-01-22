# Training Pipeline Implementation Plan (REVISED)

## Key Design Changes
1. **No separate writer thread** - write to disk inline after pushing to replay buffer (can overlap with GPU during training later if needed)
2. **Binary format** - simple, fast, no JSON parsing
3. **Lock-free deque** - atomic head/tail, `reserve_n()` returns `&mut [Sample]` for exclusive write access
4. **Target samples not games**

---

## Replay Buffer Design

```rust
struct ReplayBuffer<const N: usize> {
    // Ring buffer storage - pre-allocated
    data: Box<[MaybeUninit<Sample>]>,
    capacity: usize,
    
    // Atomic positions
    head: AtomicU64,  // next write position
    tail: AtomicU64,  // oldest valid position (head - min(writes, capacity))
}

impl ReplayBuffer {
    /// Reserve n slots for writing. Returns mutable slice.
    /// Caller has exclusive access to these slots.
    fn reserve(&self, n: usize) -> &mut [Sample] {
        let start = self.head.fetch_add(n as u64, Ordering::AcqRel);
        // ... return slice of data[start..start+n]
    }
    
    /// Sample randomly from valid range [tail, head)
    fn sample(&self, n: usize, rng: &mut impl Rng) -> Vec<&Sample> { ... }
    
    /// Write to disk (binary format)
    fn save(&self, path: &Path) -> io::Result<()> { ... }
    
    /// Load from disk
    fn load(&self, path: &Path) -> io::Result<usize> { ... }
}
```

---

## Binary File Format

Simple header + raw samples:
```
[8 bytes] magic: "SIEBREN\0"
[8 bytes] version: u64
[8 bytes] sample_count: u64
[8 bytes] notation_max_len: u64  // max PEN string length (padded)
[8 bytes] policy_len: u64        // NUM_ACTIONS
[samples...]
    [notation_max_len bytes] notation (null-padded)
    [policy_len * 4 bytes]   policy (f32 array)
    [4 bytes]                value (f32)
```

---

## Task Order

### 1. GameNotation trait + implementations
### 2. Replay buffer (lock-free deque)
### 3. Binary save/load
### 4. Update training.rs to use replay buffer
### 5. PyO3 bindings for replay buffer
### 6. selfplay_bytefight + ByteFightSelfPlay wrapper
### 7. ByteFightNet
### 8. Training loop (loss, optimizer)
### 9. Checkpointing
### 10. Wire everything together in train.py
