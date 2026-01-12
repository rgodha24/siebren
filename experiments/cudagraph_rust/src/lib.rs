use ndarray::{Array3, Array4, Axis};
use numpy::PyArray4;
use pyo3::prelude::*;

const OBS_H: usize = 8;
const OBS_W: usize = 8;
const OBS_C: usize = 64;
const BATCH: usize = 256;

fn build_batch() -> Array4<f32> {
    let mut batch = Array4::<f32>::zeros((BATCH, OBS_H, OBS_W, OBS_C));
    for b in 0..BATCH {
        let obs = Array3::from_shape_fn((OBS_H, OBS_W, OBS_C), |(i, j, k)| {
            (b as f32 + i as f32 + j as f32 + k as f32) / 1000.0
        });
        batch.index_axis_mut(Axis(0), b).assign(&obs);
    }
    batch
}

#[pyfunction]
fn play_games(num_games: usize, callback: PyObject) -> PyResult<Vec<PyObject>> {
    Python::with_gil(|py| {
        let callback = callback.bind(py);
        let torch = PyModule::import(py, "torch")?;
        let mut outputs = Vec::with_capacity(num_games);
        for _ in 0..num_games {
            let batch = build_batch();
            let py_batch = PyArray4::from_owned_array(py, batch);
            let tensor = torch.call_method1("from_numpy", (py_batch,))?;
            let output = callback.call1((tensor,))?;
            outputs.push(output.unbind());
        }
        Ok(outputs)
    })
}

#[pymodule]
fn cudagraph_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(play_games, m)?)?;
    Ok(())
}
