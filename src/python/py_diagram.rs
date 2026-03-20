use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyfunction;

use crate::diagram::{self, PersistenceDiagram};

/// Compute the bottleneck distance between two persistence diagrams.
///
/// The bottleneck distance is the L-infinity cost of the optimal matching
/// between points of the two diagrams (including matches to the diagonal).
/// It is stable under small perturbations of the input data.
///
/// Parameters
/// ----------
/// diagram1 : list[tuple[float, float]]
///     First persistence diagram as a list of ``(birth, death)`` pairs.
/// diagram2 : list[tuple[float, float]]
///     Second persistence diagram as a list of ``(birth, death)`` pairs.
///
/// Returns
/// -------
/// float
///     The bottleneck distance between the two diagrams.
///
/// Examples
/// --------
/// >>> import synaplex
/// >>> d1 = [(0.0, 1.0), (0.5, 2.0)]
/// >>> d2 = [(0.0, 1.1), (0.5, 1.9)]
/// >>> dist = synaplex.diagram.bottleneck_distance(d1, d2)
/// >>> dist < 0.2
/// True
///
/// Identical diagrams have zero distance:
///
/// >>> synaplex.diagram.bottleneck_distance(d1, d1)
/// 0.0
#[gen_stub_pyfunction(module = "synaplex.diagram")]
#[pyfunction]
pub fn bottleneck_distance(diagram1: Vec<(f64, f64)>, diagram2: Vec<(f64, f64)>) -> f64 {
	let d1 = PersistenceDiagram {
		points: diagram1,
		dimension: 0,
	};
	let d2 = PersistenceDiagram {
		points: diagram2,
		dimension: 0,
	};
	diagram::bottleneck_distance(&d1, &d2)
}

/// Compute the p-Wasserstein distance between two persistence diagrams.
///
/// Uses the Hungarian algorithm to find the optimal matching that minimises
/// the p-th power of the sum of L-infinity matching costs. Points can be
/// matched to their projection on the diagonal at cost
/// ``(death - birth) / 2``.
///
/// Parameters
/// ----------
/// diagram1 : list[tuple[float, float]]
///     First persistence diagram as a list of ``(birth, death)`` pairs.
/// diagram2 : list[tuple[float, float]]
///     Second persistence diagram as a list of ``(birth, death)`` pairs.
/// p : float, optional
///     The exponent for the Wasserstein distance (default ``2.0``).
///     ``p=1.0`` gives the earth-mover distance, ``p=2.0`` the standard
///     2-Wasserstein distance.
///
/// Returns
/// -------
/// float
///     The p-Wasserstein distance between the two diagrams.
///
/// Examples
/// --------
/// >>> import synaplex
/// >>> d1 = [(0.0, 1.0), (0.5, 2.0)]
/// >>> d2 = [(0.0, 1.1), (0.5, 1.9)]
/// >>> w2 = synaplex.diagram.wasserstein_distance(d1, d2)
/// >>> w2 > 0.0
/// True
///
/// Earth-mover distance (p=1):
///
/// >>> w1 = synaplex.diagram.wasserstein_distance(d1, d2, p=1.0)
#[gen_stub_pyfunction(module = "synaplex.diagram")]
#[pyfunction]
#[pyo3(signature = (diagram1, diagram2, p=2.0))]
pub fn wasserstein_distance(diagram1: Vec<(f64, f64)>, diagram2: Vec<(f64, f64)>, p: f64) -> f64 {
	let d1 = PersistenceDiagram {
		points: diagram1,
		dimension: 0,
	};
	let d2 = PersistenceDiagram {
		points: diagram2,
		dimension: 0,
	};
	diagram::wasserstein_distance(&d1, &d2, p)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
	m.add_function(wrap_pyfunction!(bottleneck_distance, m)?)?;
	m.add_function(wrap_pyfunction!(wasserstein_distance, m)?)?;
	Ok(())
}
