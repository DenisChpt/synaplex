use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use super::convert;
use crate::complex::VietorisRipsComplex;
use crate::distance::{DistanceMatrix, Euclidean};

/// A Vietoris-Rips simplicial complex built from a point cloud.
///
/// The Vietoris-Rips complex is constructed by adding a k-simplex for every
/// set of (k+1) points that are pairwise within the given distance threshold.
/// It is the standard construction used in topological data analysis to study
/// the shape of data.
///
/// Parameters
/// ----------
/// points : numpy.ndarray
///     Input point cloud of shape ``(n_points, n_features)``.
/// max_dim : int, optional
///     Maximum simplex dimension to construct (default ``2``).
/// threshold : float or None, optional
///     Maximum edge length. ``None`` uses the diameter of the point cloud
///     (default ``None``).
///
/// Examples
/// --------
/// Build a complex from four points in the plane:
///
/// >>> import numpy as np
/// >>> import synaplex
/// >>> pts = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float64)
/// >>> cpx = synaplex.complex.VietorisRipsComplex(pts, max_dim=2)
/// >>> cpx.dimension()
/// 2
/// >>> cpx.len()  # total number of simplices
/// 11
///
/// Restrict by a distance threshold:
///
/// >>> cpx = synaplex.complex.VietorisRipsComplex(pts, threshold=1.0)
/// >>> cpx.threshold()
/// 1.0
#[gen_stub_pyclass]
#[pyclass(name = "VietorisRipsComplex", module = "synaplex.complex")]
pub struct PyVietorisRipsComplex {
	inner: VietorisRipsComplex,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyVietorisRipsComplex {
	/// Create a new Vietoris-Rips complex from a point cloud.
	///
	/// Parameters
	/// ----------
	/// points : numpy.ndarray
	///     Point cloud of shape ``(n_points, n_features)``.
	/// max_dim : int, optional
	///     Maximum simplex dimension (default ``2``).
	/// threshold : float or None, optional
	///     Maximum edge length (default ``None`` — uses full diameter).
	///
	/// Returns
	/// -------
	/// VietorisRipsComplex
	///     The constructed simplicial complex.
	#[new]
	#[pyo3(signature = (points, max_dim=2, threshold=None))]
	fn new(points: PyReadonlyArray2<'_, f64>, max_dim: usize, threshold: Option<f64>) -> Self {
		let data = convert::numpy_to_array2(points);
		let dm = DistanceMatrix::from_point_cloud(&data.view(), &Euclidean);
		let inner = VietorisRipsComplex::new(&dm, max_dim, threshold);
		Self { inner }
	}

	/// Return the total number of simplices in the complex.
	///
	/// Returns
	/// -------
	/// int
	///     Total simplex count across all dimensions.
	///
	/// Examples
	/// --------
	/// >>> cpx.len()
	/// 11
	fn len(&self) -> usize {
		self.inner.len()
	}

	/// Count the simplices of a given dimension.
	///
	/// Parameters
	/// ----------
	/// dim : int
	///     The dimension to count (0 = vertices, 1 = edges, 2 = triangles, ...).
	///
	/// Returns
	/// -------
	/// int
	///     Number of simplices in that dimension.
	///
	/// Examples
	/// --------
	/// >>> cpx.count_by_dim(0)  # number of vertices
	/// 4
	/// >>> cpx.count_by_dim(1)  # number of edges
	/// 6
	fn count_by_dim(&self, dim: usize) -> usize {
		self.inner.count_by_dim(dim)
	}

	/// Return the maximum dimension of the complex.
	///
	/// Returns
	/// -------
	/// int
	///     The highest dimension of any simplex in the complex.
	///
	/// Examples
	/// --------
	/// >>> cpx.dimension()
	/// 2
	fn dimension(&self) -> usize {
		self.inner.dimension()
	}

	/// Return the filtration threshold used to build the complex.
	///
	/// Returns
	/// -------
	/// float
	///     The maximum edge length in the filtration.
	///
	/// Examples
	/// --------
	/// >>> cpx.threshold()
	/// 1.4142135623730951
	fn threshold(&self) -> f64 {
		self.inner.threshold()
	}

	fn __repr__(&self) -> String {
		format!(
			"VietorisRipsComplex(simplices={}, dim={}, threshold={:.4})",
			self.inner.len(),
			self.inner.dimension(),
			self.inner.threshold()
		)
	}
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
	m.add_class::<PyVietorisRipsComplex>()?;
	Ok(())
}
