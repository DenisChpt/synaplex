use numpy::{PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyfunction;

use super::convert;
use crate::distance::{Cosine, DistanceMatrix, DistanceMetric, Euclidean, Manhattan};

/// Compute the pairwise distance matrix for a point cloud.
///
/// Returns the upper-triangular entries of the symmetric distance matrix
/// stored as a flat array (condensed form, same layout as
/// ``scipy.spatial.distance.pdist``).
///
/// Parameters
/// ----------
/// points : numpy.ndarray
///     Input point cloud of shape ``(n_points, n_features)``.
/// metric : str, optional
///     Distance metric to use (default ``"euclidean"``).
///     Supported values: ``"euclidean"``, ``"cosine"``, ``"manhattan"``.
///
/// Returns
/// -------
/// numpy.ndarray
///     1-D array of length ``n_points * (n_points - 1) / 2`` containing the
///     upper-triangle of the pairwise distance matrix.
///
/// Raises
/// ------
/// ValueError
///     If *metric* is not one of the supported values.
///
/// Examples
/// --------
/// >>> import numpy as np
/// >>> import synaplex
/// >>> pts = np.array([[0, 0], [3, 0], [0, 4]], dtype=np.float64)
/// >>> d = synaplex.distance.distance_matrix(pts)
/// >>> d.shape
/// (3,)
/// >>> d[0]  # distance between point 0 and point 1
/// 3.0
///
/// Use cosine distance:
///
/// >>> d = synaplex.distance.distance_matrix(pts, metric="cosine")
#[gen_stub_pyfunction(module = "synaplex.distance")]
#[pyfunction]
#[pyo3(signature = (points, metric="euclidean"))]
pub fn distance_matrix(
	py: Python<'_>,
	points: PyReadonlyArray2<'_, f64>,
	metric: &str,
) -> PyResult<Py<PyArray1<f64>>> {
	let data = convert::numpy_to_array2(points);
	let m: Box<dyn DistanceMetric> = match metric {
		"euclidean" => Box::new(Euclidean),
		"cosine" => Box::new(Cosine),
		"manhattan" => Box::new(Manhattan),
		_ => {
			return Err(pyo3::exceptions::PyValueError::new_err(format!(
				"Unknown metric: {metric}. Use 'euclidean', 'cosine', or 'manhattan'."
			)));
		}
	};

	let dm = DistanceMatrix::from_point_cloud(&data.view(), m.as_ref());
	Ok(PyArray1::from_slice(py, dm.data()).into())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
	m.add_function(wrap_pyfunction!(distance_matrix, m)?)?;
	Ok(())
}
