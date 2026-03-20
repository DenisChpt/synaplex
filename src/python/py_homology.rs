use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3_stub_gen::derive::gen_stub_pyfunction;

use super::convert;
use crate::distance::{DistanceMatrix, Euclidean};
use crate::homology::{CohomologyConfig, compute_persistent_cohomology};

/// Compute persistent cohomology of a point cloud.
///
/// Builds a Vietoris-Rips filtration from the Euclidean distance matrix of the
/// input points, then computes persistent cohomology with coefficients in Z/pZ.
///
/// Parameters
/// ----------
/// points : numpy.ndarray
///     Input point cloud of shape ``(n_points, n_features)``.
/// max_dim : int, optional
///     Maximum homological dimension to compute (default ``1``).
///     Dimension 0 captures connected components, dimension 1 captures loops, etc.
/// threshold : float or None, optional
///     Maximum filtration value (edge length). ``None`` means no cutoff
///     (default ``None``).
/// modulus : int, optional
///     Prime modulus for the coefficient field Z/pZ (default ``2``).
///     Use a different prime (e.g. 3, 5, 7) to detect torsion.
/// cocycles : bool, optional
///     If ``True``, also return cocycle representatives (default ``False``).
///
/// Returns
/// -------
/// numpy.ndarray or dict
///     * If ``cocycles=False``: an array of shape ``(n_pairs, 3)`` where each
///       row is ``[dimension, birth, death]``.
///     * If ``cocycles=True``: a ``dict`` with keys ``"diagram"`` (the array)
///       and ``"cocycles"`` (nested list of cocycle representatives).
///
/// Examples
/// --------
/// Compute H0 and H1 of a small point cloud:
///
/// >>> import numpy as np
/// >>> import synaplex
/// >>> pts = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float64)
/// >>> dgm = synaplex.persistence_diagram(pts, max_dim=1)
/// >>> dgm.shape[1]
/// 3
///
/// Compute with cocycle representatives:
///
/// >>> result = synaplex.persistence_diagram(pts, max_dim=1, cocycles=True)
/// >>> result["diagram"].shape[1]
/// 3
///
/// Use Z/3Z coefficients to detect torsion:
///
/// >>> dgm = synaplex.persistence_diagram(pts, max_dim=1, modulus=3)
#[gen_stub_pyfunction(module = "synaplex.homology")]
#[pyfunction]
#[pyo3(signature = (points, max_dim=1, threshold=None, modulus=2, cocycles=false))]
pub fn persistence_diagram(
	py: Python<'_>,
	points: PyReadonlyArray2<'_, f64>,
	max_dim: usize,
	threshold: Option<f64>,
	modulus: u16,
	cocycles: bool,
) -> PyResult<Py<pyo3::types::PyAny>> {
	let data = convert::numpy_to_array2(points);
	let dm = DistanceMatrix::from_point_cloud(&data.view(), &Euclidean);
	let result = compute_persistent_cohomology(
		&dm,
		&CohomologyConfig {
			max_dimension: max_dim,
			threshold,
			modulus,
			cocycles,
		},
	);

	let n = result.pairs.len();
	let mut diagram = ndarray::Array2::zeros((n, 3));
	for (i, p) in result.pairs.iter().enumerate() {
		diagram[[i, 0]] = p.dimension as f64;
		diagram[[i, 1]] = p.birth;
		diagram[[i, 2]] = p.death;
	}

	if cocycles {
		let cocycles_py = PyList::empty(py);
		for dim_cocycles in &result.cocycles {
			let dim_list = PyList::empty(py);
			for cocycle in dim_cocycles {
				let entries_list = PyList::new(py, &cocycle.entries)?;
				dim_list.append(entries_list)?;
			}
			cocycles_py.append(dim_list)?;
		}

		let dict = pyo3::types::PyDict::new(py);
		dict.set_item("diagram", PyArray2::from_owned_array(py, diagram))?;
		dict.set_item("cocycles", cocycles_py)?;
		Ok(dict.into())
	} else {
		Ok(PyArray2::from_owned_array(py, diagram).into())
	}
}

/// Compute Betti numbers at a given filtration value.
///
/// The k-th Betti number counts the number of k-dimensional topological
/// features (connected components, loops, voids, ...) that are alive at the
/// specified filtration value.
///
/// Parameters
/// ----------
/// points : numpy.ndarray
///     Input point cloud of shape ``(n_points, n_features)``.
/// at_filtration : float
///     Filtration value at which to count alive features.
/// max_dim : int, optional
///     Maximum homological dimension (default ``1``).
/// threshold : float or None, optional
///     Maximum filtration value (default ``None``).
/// modulus : int, optional
///     Prime modulus for the coefficient field Z/pZ (default ``2``).
///
/// Returns
/// -------
/// list[int]
///     A list of Betti numbers ``[beta_0, beta_1, ..., beta_{max_dim}]``.
///
/// Examples
/// --------
/// >>> import numpy as np
/// >>> import synaplex
/// >>> pts = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float64)
/// >>> betti = synaplex.betti_numbers(pts, at_filtration=1.5)
/// >>> len(betti)  # max_dim+1 entries
/// 2
/// >>> betti[0]  # number of connected components
/// 1
#[gen_stub_pyfunction(module = "synaplex.homology")]
#[pyfunction]
#[pyo3(signature = (points, at_filtration, max_dim=1, threshold=None, modulus=2))]
pub fn betti_numbers(
	_py: Python<'_>,
	points: PyReadonlyArray2<'_, f64>,
	at_filtration: f64,
	max_dim: usize,
	threshold: Option<f64>,
	modulus: u16,
) -> PyResult<Vec<usize>> {
	let data = convert::numpy_to_array2(points);
	let dm = DistanceMatrix::from_point_cloud(&data.view(), &Euclidean);
	let result = compute_persistent_cohomology(
		&dm,
		&CohomologyConfig {
			max_dimension: max_dim,
			threshold,
			modulus,
			cocycles: false,
		},
	);
	Ok(crate::homology::betti_numbers(&result.pairs, at_filtration))
}

/// Test function to verify that the native extension loads correctly.
///
/// Parameters
/// ----------
/// a : int
///     First value.
/// b : int, optional
///     Second value (default ``2``).
/// c : int, optional
///     Third value (default ``3``).
///
/// Returns
/// -------
/// int
///     The sum ``a + b + c``.
///
/// Examples
/// --------
/// >>> import synaplex
/// >>> synaplex.test_fn(1)
/// 6
/// >>> synaplex.test_fn(1, b=10, c=20)
/// 31
#[gen_stub_pyfunction(module = "synaplex.homology")]
#[pyfunction]
#[pyo3(name = "test_fn", signature = (a, b=2, c=3))]
pub fn test_fn_py(_py: Python<'_>, a: u32, b: u32, c: u32) -> PyResult<u32> {
	Ok(a + b + c)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
	m.add_function(wrap_pyfunction!(persistence_diagram, m)?)?;
	m.add_function(wrap_pyfunction!(betti_numbers, m)?)?;
	m.add_function(wrap_pyfunction!(test_fn_py, m)?)?;
	Ok(())
}
