//! Replay buffer implementation for storing self-play experience.
//!
//! Uses a circular buffer to store (state, policy, value) tuples from self-play games.
//! Supports uniform random sampling for training batches.

use ndarray::{Array2, ArrayD, Axis, IxDyn};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::seq::index::sample;
use std::collections::HashMap;

/// A single experience tuple stored in the buffer.
#[derive(Clone)]
struct Experience {
    /// Named state tensors (flattened)
    states: HashMap<String, Vec<f32>>,
    /// State tensor shapes for reconstruction
    state_shapes: HashMap<String, Vec<usize>>,
    /// Target policy distribution
    policy: Vec<f32>,
    /// Target value
    value: f32,
}

/// Circular replay buffer for storing self-play experience.
///
/// The buffer has a fixed maximum capacity. When full, new experiences
/// overwrite the oldest ones.
#[pyclass]
pub struct ReplayBuffer {
    buffer: Vec<Experience>,
    capacity: usize,
    position: usize,
    size: usize,
    /// Number of actions (policy size)
    num_actions: usize,
    /// Expected state keys and their shapes
    state_schema: Option<HashMap<String, Vec<usize>>>,
}

#[pymethods]
impl ReplayBuffer {
    /// Create a new replay buffer.
    ///
    /// Args:
    ///     capacity: Maximum number of experiences to store
    ///     num_actions: Size of the policy vector
    #[new]
    #[pyo3(signature = (capacity, num_actions))]
    fn new(capacity: usize, num_actions: usize) -> Self {
        ReplayBuffer {
            buffer: Vec::with_capacity(capacity),
            capacity,
            position: 0,
            size: 0,
            num_actions,
            state_schema: None,
        }
    }

    /// Add a single experience to the buffer.
    ///
    /// Args:
    ///     states: Dict of state tensors (numpy arrays)
    ///     policy: Target policy distribution (numpy array)
    ///     value: Target value (float)
    fn add(
        &mut self,
        _py: Python<'_>,
        states: &Bound<'_, PyDict>,
        policy: PyReadonlyArrayDyn<'_, f32>,
        value: f32,
    ) -> PyResult<()> {
        let policy_vec: Vec<f32> = policy.as_array().iter().cloned().collect();
        if policy_vec.len() != self.num_actions {
            return Err(PyValueError::new_err(format!(
                "Policy size mismatch: expected {}, got {}",
                self.num_actions,
                policy_vec.len()
            )));
        }

        let mut state_map = HashMap::new();
        let mut shape_map = HashMap::new();

        for (key, value) in states.iter() {
            let name: String = key.extract()?;
            let arr: PyReadonlyArrayDyn<'_, f32> = value.extract()?;
            let shape: Vec<usize> = arr.shape().to_vec();
            let data: Vec<f32> = arr.as_array().iter().cloned().collect();

            state_map.insert(name.clone(), data);
            shape_map.insert(name, shape);
        }

        // Initialize or validate schema
        if self.state_schema.is_none() {
            self.state_schema = Some(shape_map.clone());
        }

        let exp = Experience {
            states: state_map,
            state_shapes: shape_map,
            policy: policy_vec,
            value,
        };

        if self.buffer.len() < self.capacity {
            self.buffer.push(exp);
        } else {
            self.buffer[self.position] = exp;
        }

        self.position = (self.position + 1) % self.capacity;
        self.size = self.size.saturating_add(1).min(self.capacity);

        Ok(())
    }

    /// Add a batch of experiences to the buffer.
    ///
    /// Args:
    ///     states: Dict of batched state tensors (first dim is batch)
    ///     policies: Batched policy distributions (batch, num_actions)
    ///     values: Batched values (batch,)
    fn add_batch(
        &mut self,
        _py: Python<'_>,
        states: &Bound<'_, PyDict>,
        policies: PyReadonlyArrayDyn<'_, f32>,
        values: PyReadonlyArrayDyn<'_, f32>,
    ) -> PyResult<()> {
        let policies_arr = policies.as_array();
        let values_arr = values.as_array();

        let batch_size = values_arr.len();

        // Extract all state arrays
        let mut state_arrays: HashMap<String, ArrayD<f32>> = HashMap::new();
        for (key, value) in states.iter() {
            let name: String = key.extract()?;
            let arr: PyReadonlyArrayDyn<'_, f32> = value.extract()?;
            state_arrays.insert(name, arr.as_array().to_owned());
        }

        // Add each experience
        for i in 0..batch_size {
            let mut state_map = HashMap::new();
            let mut shape_map = HashMap::new();

            for (name, arr) in &state_arrays {
                // Slice along batch dimension
                let slice = arr.index_axis(Axis(0), i);
                let shape: Vec<usize> = slice.shape().to_vec();
                let data: Vec<f32> = slice.iter().cloned().collect();

                state_map.insert(name.clone(), data);
                shape_map.insert(name.clone(), shape);
            }

            // Initialize schema on first add
            if self.state_schema.is_none() {
                self.state_schema = Some(shape_map.clone());
            }

            let policy_slice = policies_arr.index_axis(Axis(0), i);
            let policy_vec: Vec<f32> = policy_slice.iter().cloned().collect();

            let exp = Experience {
                states: state_map,
                state_shapes: shape_map,
                policy: policy_vec,
                value: values_arr[i],
            };

            if self.buffer.len() < self.capacity {
                self.buffer.push(exp);
            } else {
                self.buffer[self.position] = exp;
            }

            self.position = (self.position + 1) % self.capacity;
            self.size = self.size.saturating_add(1).min(self.capacity);
        }

        Ok(())
    }

    /// Sample a batch of experiences uniformly at random.
    ///
    /// Args:
    ///     batch_size: Number of experiences to sample
    ///
    /// Returns:
    ///     Tuple of (states_dict, policies, values) where:
    ///     - states_dict: Dict[str, np.ndarray] with batched states
    ///     - policies: np.ndarray of shape (batch_size, num_actions)
    ///     - values: np.ndarray of shape (batch_size, 1)
    fn sample<'py>(
        &self,
        py: Python<'py>,
        batch_size: usize,
    ) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyArrayDyn<f32>>, Bound<'py, PyArrayDyn<f32>>)>
    {
        if self.size == 0 {
            return Err(PyValueError::new_err("Buffer is empty"));
        }

        let actual_batch = batch_size.min(self.size);
        let mut rng = rand::rng();
        let indices = sample(&mut rng, self.size, actual_batch);

        // Initialize output arrays
        let schema = self.state_schema.as_ref().ok_or_else(|| {
            PyValueError::new_err("Buffer schema not initialized")
        })?;

        // Build batched state arrays
        let mut state_batches: HashMap<String, Vec<f32>> = HashMap::new();
        for name in schema.keys() {
            state_batches.insert(name.clone(), Vec::new());
        }

        let mut policies: Vec<f32> = Vec::with_capacity(actual_batch * self.num_actions);
        let mut values: Vec<f32> = Vec::with_capacity(actual_batch);

        for idx in indices {
            let exp = &self.buffer[idx];

            for (name, data) in &exp.states {
                state_batches.get_mut(name).unwrap().extend(data.iter());
            }

            policies.extend(&exp.policy);
            values.push(exp.value);
        }

        // Convert to numpy arrays
        let states_dict = PyDict::new(py);
        for (name, data) in state_batches {
            let shape = schema.get(&name).unwrap();
            let mut full_shape = vec![actual_batch];
            full_shape.extend(shape);

            let arr = ArrayD::from_shape_vec(IxDyn(&full_shape), data)
                .map_err(|e| PyValueError::new_err(format!("Shape error: {}", e)))?;
            states_dict.set_item(&name, arr.into_pyarray(py))?;
        }

        let policies_arr = Array2::from_shape_vec((actual_batch, self.num_actions), policies)
            .map_err(|e| PyValueError::new_err(format!("Policy shape error: {}", e)))?
            .into_dyn()
            .into_pyarray(py);

        let values_arr = Array2::from_shape_vec((actual_batch, 1), values)
            .map_err(|e| PyValueError::new_err(format!("Value shape error: {}", e)))?
            .into_dyn()
            .into_pyarray(py);

        Ok((states_dict, policies_arr, values_arr))
    }

    /// Return the current number of experiences in the buffer.
    fn __len__(&self) -> usize {
        self.size
    }

    /// Return the maximum capacity of the buffer.
    #[getter]
    fn capacity(&self) -> usize {
        self.capacity
    }

    /// Return the number of actions (policy size).
    #[getter]
    fn num_actions(&self) -> usize {
        self.num_actions
    }

    /// Clear all experiences from the buffer.
    fn clear(&mut self) {
        self.buffer.clear();
        self.position = 0;
        self.size = 0;
    }

    fn __repr__(&self) -> String {
        format!(
            "ReplayBuffer(size={}, capacity={}, num_actions={})",
            self.size, self.capacity, self.num_actions
        )
    }
}
