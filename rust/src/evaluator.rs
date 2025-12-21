//! ONNX Runtime evaluator for neural network inference during MCTS.
//!
//! Uses the `ort` crate to run ONNX models efficiently on CPU or GPU.

use ndarray::{ArrayD, ArrayViewD};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use std::collections::HashMap;
use std::path::Path;

/// Neural network evaluator using ONNX Runtime.
///
/// Runs inference on batches of observations and returns (policy, value) pairs.
pub struct Evaluator {
    session: Session,
    input_names: Vec<String>,
    output_names: Vec<String>,
}

impl Evaluator {
    /// Create a new evaluator from an ONNX model file.
    ///
    /// # Arguments
    /// * `model_path` - Path to the ONNX model file
    ///
    /// # Errors
    /// Returns an error if the model cannot be loaded.
    pub fn new<P: AsRef<Path>>(model_path: P) -> ort::Result<Self> {
        let model_bytes = std::fs::read(model_path.as_ref())
            .map_err(|e| ort::Error::new(format!("Failed to read model file: {}", e)))?;

        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_memory(&model_bytes)?;

        let input_names: Vec<String> = session.inputs.iter().map(|i| i.name.clone()).collect();

        let output_names: Vec<String> = session.outputs.iter().map(|o| o.name.clone()).collect();

        Ok(Evaluator {
            session,
            input_names,
            output_names,
        })
    }

    /// Run inference on a batch of inputs.
    ///
    /// # Arguments
    /// * `inputs` - HashMap mapping input names to batched arrays
    ///
    /// # Returns
    /// Tuple of (policy_logits, values) as dynamic arrays.
    /// - policy_logits: shape (batch_size, num_actions)
    /// - values: shape (batch_size, 1)
    pub fn evaluate(
        &mut self,
        inputs: HashMap<String, ArrayD<f32>>,
    ) -> ort::Result<(ArrayD<f32>, ArrayD<f32>)> {
        // Build input vector in the order expected by the model
        let mut ort_inputs: Vec<(&str, ort::value::DynValue)> = Vec::new();

        for name in &self.input_names {
            let arr = inputs
                .get(name)
                .ok_or_else(|| ort::Error::new(format!("Missing input: {}", name)))?;

            // Clone to owned array for ort
            let owned = arr.clone();
            let value = ort::value::Tensor::from_array(owned)?.into_dyn();
            ort_inputs.push((name.as_str(), value));
        }

        let outputs = self.session.run(ort_inputs)?;

        // Extract outputs - assume first is policy, second is value
        let policy = outputs[self.output_names[0].as_str()]
            .try_extract_array::<f32>()?
            .to_owned();

        let value = if self.output_names.len() > 1 {
            outputs[self.output_names[1].as_str()]
                .try_extract_array::<f32>()?
                .to_owned()
        } else {
            // Single output model - use policy as value too (shouldn't happen)
            policy.clone()
        };

        Ok((policy, value))
    }

    /// Convenience method for single-input models.
    pub fn evaluate_single(
        &mut self,
        input_name: &str,
        input: ArrayD<f32>,
    ) -> ort::Result<(ArrayD<f32>, ArrayD<f32>)> {
        let mut inputs = HashMap::new();
        inputs.insert(input_name.to_string(), input);
        self.evaluate(inputs)
    }

    /// Get the input names expected by the model.
    pub fn input_names(&self) -> &[String] {
        &self.input_names
    }

    /// Get the output names produced by the model.
    pub fn output_names(&self) -> &[String] {
        &self.output_names
    }
}

/// Trait for types that can be evaluated by the neural network.
///
/// This allows the MCTS to request evaluations without knowing
/// the specific observation format.
pub trait Evaluable {
    /// Convert this observation to named input arrays for the evaluator.
    fn to_eval_inputs(&self) -> HashMap<String, ArrayD<f32>>;
}

/// Batch evaluator that accumulates observations and evaluates them together.
///
/// Useful for MCTS where leaf nodes are expanded in waves.
pub struct BatchEvaluator {
    evaluator: Evaluator,
    pending: Vec<HashMap<String, ArrayD<f32>>>,
}

impl BatchEvaluator {
    pub fn new(evaluator: Evaluator) -> Self {
        BatchEvaluator {
            evaluator,
            pending: Vec::new(),
        }
    }

    /// Queue an observation for evaluation.
    /// Returns an index to retrieve the result after `flush()`.
    pub fn queue(&mut self, inputs: HashMap<String, ArrayD<f32>>) -> usize {
        let idx = self.pending.len();
        self.pending.push(inputs);
        idx
    }

    /// Queue an evaluable observation.
    pub fn queue_evaluable<E: Evaluable>(&mut self, obs: &E) -> usize {
        self.queue(obs.to_eval_inputs())
    }

    /// Evaluate all pending observations and return results.
    ///
    /// Returns a vector of (policy, value) pairs in queue order.
    pub fn flush(&mut self) -> ort::Result<Vec<(ArrayD<f32>, ArrayD<f32>)>> {
        if self.pending.is_empty() {
            return Ok(Vec::new());
        }

        // Stack all pending inputs into batches
        let input_names: Vec<String> = self.evaluator.input_names().to_vec();
        let batch_size = self.pending.len();

        let mut batched: HashMap<String, ArrayD<f32>> = HashMap::new();

        for name in &input_names {
            // Collect all arrays for this input
            let arrays: Vec<ArrayViewD<f32>> = self
                .pending
                .iter()
                .map(|p| p.get(name).unwrap().view())
                .collect();

            // Stack along new batch dimension
            let stacked = ndarray::stack(ndarray::Axis(0), &arrays)
                .map_err(|e| ort::Error::new(format!("Stack error: {}", e)))?;

            batched.insert(name.clone(), stacked);
        }

        let (policies, values) = self.evaluator.evaluate(batched)?;

        // Split results back into individual samples
        let mut results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let policy = policies
                .index_axis(ndarray::Axis(0), i)
                .to_owned()
                .into_dyn();
            let value = values
                .index_axis(ndarray::Axis(0), i)
                .to_owned()
                .into_dyn();
            results.push((policy, value));
        }

        self.pending.clear();
        Ok(results)
    }

    /// Clear pending observations without evaluating.
    pub fn clear(&mut self) {
        self.pending.clear();
    }

    /// Number of pending observations.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[test]
    fn test_evaluator_with_onnx_model() {
        // This test requires the ONNX model to exist
        // Run: uv run python -c "from siebren.export import save_model; ..." first
        let model_path = "models/snake_test2/snake_test2_epoch_0000.onnx";

        if !std::path::Path::new(model_path).exists() {
            eprintln!("Skipping test: model not found at {}", model_path);
            return;
        }

        let mut evaluator = Evaluator::new(model_path).expect("Failed to load model");

        // Create test inputs matching the model's expected shapes
        let board = Array::from_shape_fn((1, 7, 32, 32), |_| rand::random::<f32>()).into_dyn();
        let history = Array::from_shape_fn((1, 1, 18), |_| rand::random::<f32>()).into_dyn();

        let mut inputs = HashMap::new();
        inputs.insert("board".to_string(), board);
        inputs.insert("heuristic_history".to_string(), history);

        let (policy, value) = evaluator.evaluate(inputs).expect("Inference failed");

        // Check output shapes
        assert_eq!(policy.shape(), &[1, 10], "Policy shape mismatch");
        assert_eq!(value.shape(), &[1, 1], "Value shape mismatch");

        // Policy should be log probabilities (sum of exp should be ~1)
        let policy_sum: f32 = policy.iter().map(|x| x.exp()).sum();
        assert!(
            (policy_sum - 1.0).abs() < 0.01,
            "Policy doesn't sum to 1: {}",
            policy_sum
        );

        // Value should be in [-1, 1] (tanh output)
        let v = value[[0, 0]];
        assert!(v >= -1.0 && v <= 1.0, "Value out of range: {}", v);

        println!("Evaluator test passed!");
    }

    #[test]
    fn test_batch_evaluator() {
        let model_path = "models/snake_test2/snake_test2_epoch_0000.onnx";

        if !std::path::Path::new(model_path).exists() {
            eprintln!("Skipping test: model not found at {}", model_path);
            return;
        }

        let evaluator = Evaluator::new(model_path).expect("Failed to load model");
        let mut batch_eval = BatchEvaluator::new(evaluator);

        // Queue multiple observations
        for _ in 0..4 {
            let board = Array::from_shape_fn((7, 32, 32), |_| rand::random::<f32>()).into_dyn();
            let history = Array::from_shape_fn((1, 18), |_| rand::random::<f32>()).into_dyn();

            let mut inputs = HashMap::new();
            inputs.insert("board".to_string(), board);
            inputs.insert("heuristic_history".to_string(), history);
            batch_eval.queue(inputs);
        }

        assert_eq!(batch_eval.pending_count(), 4);

        let results = batch_eval.flush().expect("Batch inference failed");
        assert_eq!(results.len(), 4);

        for (policy, value) in &results {
            assert_eq!(policy.shape(), &[10], "Policy shape mismatch");
            assert_eq!(value.shape(), &[1], "Value shape mismatch");
        }

        assert_eq!(batch_eval.pending_count(), 0);
        println!("BatchEvaluator test passed!");
    }
}
