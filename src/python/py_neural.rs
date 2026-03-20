use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyfunction;

use super::convert;
use crate::diagram::PersistenceDiagram;
use crate::neural::{ActivationAnalyzer, WeightAnalyzer};

/// Compute the topological signature of a neural network layer's activations.
///
/// Treats each row of the activation matrix as a point in feature space,
/// builds a Vietoris-Rips filtration on the resulting point cloud, and
/// returns the persistence diagram summarising the topology.
///
/// Parameters
/// ----------
/// activations : numpy.ndarray
///     Activation matrix of shape ``(n_samples, n_neurons)`` — one row per
///     input sample, one column per neuron in the layer.
/// max_dim : int, optional
///     Maximum homological dimension (default ``1``).
/// subsample : int or None, optional
///     If set, randomly subsample to this many points before computing
///     (useful for large batches). ``None`` keeps all points (default).
/// modulus : int, optional
///     Prime modulus for the coefficient field Z/pZ (default ``2``).
///
/// Returns
/// -------
/// numpy.ndarray
///     Persistence diagram as an array of shape ``(n_pairs, 3)`` where each
///     row is ``[dimension, birth, death]``.
///
/// Examples
/// --------
/// Analyse the topology of a hidden layer with 128 neurons on 500 samples:
///
/// >>> import numpy as np
/// >>> import synaplex
/// >>> activations = np.random.randn(500, 128)
/// >>> dgm = synaplex.neural.analyze_activations(activations, max_dim=1)
/// >>> dgm.shape[1]
/// 3
///
/// Subsample for faster computation:
///
/// >>> dgm = synaplex.neural.analyze_activations(activations, subsample=100)
#[gen_stub_pyfunction(module = "synaplex.neural")]
#[pyfunction]
#[pyo3(signature = (activations, max_dim=1, subsample=None, modulus=2))]
pub fn analyze_activations(
	py: Python<'_>,
	activations: PyReadonlyArray2<'_, f64>,
	max_dim: usize,
	subsample: Option<usize>,
	modulus: u16,
) -> PyResult<Py<PyArray2<f64>>> {
	let data = convert::numpy_to_array2(activations);
	let mut analyzer = ActivationAnalyzer::new(max_dim).with_modulus(modulus);
	if let Some(s) = subsample {
		analyzer = analyzer.with_subsample(s);
	}
	let diagrams = analyzer.analyze_layer(&data);
	Ok(diagrams_to_array(py, &diagrams))
}

/// Compute the topological signature of a weight matrix.
///
/// Interprets the weight matrix as a collection of row (or column) vectors
/// in parameter space, computes the persistence diagram of the resulting
/// point cloud, revealing clusters, loops, and higher-dimensional structure.
///
/// Parameters
/// ----------
/// weights : numpy.ndarray
///     Weight matrix of shape ``(n_out, n_in)`` (or transposed if
///     ``row_wise=False``).
/// max_dim : int, optional
///     Maximum homological dimension (default ``1``).
/// row_wise : bool, optional
///     If ``True`` (default), each *row* is treated as a point.
///     If ``False``, each *column* is treated as a point.
/// modulus : int, optional
///     Prime modulus for the coefficient field Z/pZ (default ``2``).
///
/// Returns
/// -------
/// numpy.ndarray
///     Persistence diagram of shape ``(n_pairs, 3)`` with columns
///     ``[dimension, birth, death]``.
///
/// Examples
/// --------
/// Analyse a 256×128 weight matrix (e.g. a linear layer):
///
/// >>> import numpy as np
/// >>> import synaplex
/// >>> W = np.random.randn(256, 128)
/// >>> dgm = synaplex.neural.analyze_weights(W, max_dim=1)
/// >>> dgm.shape[1]
/// 3
///
/// Analyse column-wise (one point per input neuron):
///
/// >>> dgm = synaplex.neural.analyze_weights(W, row_wise=False)
#[gen_stub_pyfunction(module = "synaplex.neural")]
#[pyfunction]
#[pyo3(signature = (weights, max_dim=1, row_wise=true, modulus=2))]
pub fn analyze_weights(
	py: Python<'_>,
	weights: PyReadonlyArray2<'_, f64>,
	max_dim: usize,
	row_wise: bool,
	modulus: u16,
) -> PyResult<Py<PyArray2<f64>>> {
	let data = convert::numpy_to_array2(weights);
	let mut analyzer = WeightAnalyzer::new(max_dim);
	analyzer.row_wise = row_wise;
	analyzer.modulus = modulus;
	let diagrams = analyzer.analyze(&data);
	Ok(diagrams_to_array(py, &diagrams))
}

/// Compare the topology of multiple neural network layers.
///
/// Computes the persistence diagram for each layer's activations, then
/// returns a pairwise bottleneck distance matrix measuring how
/// topologically different the layers are.
///
/// Parameters
/// ----------
/// layers : list[numpy.ndarray]
///     A list of activation matrices, one per layer. Each array has shape
///     ``(n_samples, n_neurons_in_layer)``.
/// max_dim : int, optional
///     Maximum homological dimension (default ``1``).
///
/// Returns
/// -------
/// numpy.ndarray
///     Symmetric distance matrix of shape ``(n_layers, n_layers)`` where
///     entry ``(i, j)`` is the bottleneck distance between the persistence
///     diagrams of layer *i* and layer *j*.
///
/// Examples
/// --------
/// Compare three layers of a neural network:
///
/// >>> import numpy as np
/// >>> import synaplex
/// >>> layers = [
/// ...     np.random.randn(200, 64),   # layer 1
/// ...     np.random.randn(200, 128),  # layer 2
/// ...     np.random.randn(200, 64),   # layer 3
/// ... ]
/// >>> dist = synaplex.neural.compare_layers(layers, max_dim=1)
/// >>> dist.shape
/// (3, 3)
/// >>> dist[0, 0]  # distance of a layer to itself
/// 0.0
#[gen_stub_pyfunction(module = "synaplex.neural")]
#[pyfunction]
#[pyo3(signature = (layers, max_dim=1))]
pub fn compare_layers(
	py: Python<'_>,
	layers: Vec<PyReadonlyArray2<'_, f64>>,
	max_dim: usize,
) -> PyResult<Py<PyArray2<f64>>> {
	let data: Vec<ndarray::Array2<f64>> =
		layers.into_iter().map(convert::numpy_to_array2).collect();

	let analyzer = ActivationAnalyzer::new(max_dim);
	let diagrams = analyzer.analyze_layers(&data);
	let result = crate::neural::compare_layers(&diagrams);

	Ok(PyArray2::from_owned_array(py, result).into())
}

fn diagrams_to_array(py: Python<'_>, diagrams: &[PersistenceDiagram]) -> Py<PyArray2<f64>> {
	let total: usize = diagrams.iter().map(|d| d.len()).sum();
	let mut result = ndarray::Array2::zeros((total, 3));
	let mut idx = 0;
	for diag in diagrams {
		for &(b, d) in &diag.points {
			result[[idx, 0]] = diag.dimension as f64;
			result[[idx, 1]] = b;
			result[[idx, 2]] = d;
			idx += 1;
		}
	}
	PyArray2::from_owned_array(py, result).into()
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
	m.add_function(wrap_pyfunction!(analyze_activations, m)?)?;
	m.add_function(wrap_pyfunction!(analyze_weights, m)?)?;
	m.add_function(wrap_pyfunction!(compare_layers, m)?)?;
	Ok(())
}
