use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_stub_gen::derive::gen_stub_pyfunction;

use super::convert;
use crate::distance::Euclidean;
use crate::mapper::{
	BalancedCover, ClusteringAlgorithm, CoverStrategy, Dbscan, MapperPipeline, SingleLinkage,
	UniformCover,
};

/// Run the Mapper pipeline to build a topological graph of a dataset.
///
/// Mapper is a tool from topological data analysis that produces a
/// simplified graph representation of high-dimensional data. It works by:
///
/// 1. Projecting data through a *filter function* (user-supplied values).
/// 2. Covering the filter range with overlapping intervals.
/// 3. Clustering within each interval.
/// 4. Building the *nerve* — a graph whose nodes are clusters and whose
///    edges connect clusters that share data points.
///
/// Parameters
/// ----------
/// points : numpy.ndarray
///     Input data of shape ``(n_points, n_features)``.
/// filter_values : numpy.ndarray
///     1-D array of length ``n_points`` with the filter function evaluated
///     on each point (e.g. eccentricity, density, a PCA coordinate, …).
/// num_intervals : int, optional
///     Number of intervals in the cover (default ``10``).
/// overlap : float, optional
///     Fractional overlap between consecutive intervals, in ``(0, 1)``
///     (default ``0.3``).
/// clustering : str, optional
///     Clustering algorithm: ``"single_linkage"`` (default) or
///     ``"dbscan"``.
/// clustering_threshold : float or None, optional
///     Distance threshold for clustering. For single-linkage: cut height.
///     For DBSCAN: the ``eps`` parameter (default ``None`` — auto).
/// cover_type : str, optional
///     Cover strategy: ``"uniform"`` (default) or ``"balanced"``
///     (quantile-based, equal point counts per interval).
///
/// Returns
/// -------
/// dict
///     A dictionary with keys:
///
///     * ``"nodes"`` — list of lists of point indices per node.
///     * ``"num_nodes"`` — total number of nodes.
///     * ``"num_edges"`` — total number of edges.
///     * ``"edges"`` — list of ``(source, target, shared_count)`` tuples.
///
/// Raises
/// ------
/// ValueError
///     If *clustering* or *cover_type* is not a recognised value.
///
/// Examples
/// --------
/// Build a Mapper graph from 2-D data using the first coordinate as filter:
///
/// >>> import numpy as np
/// >>> import synaplex
/// >>> pts = np.random.randn(500, 2)
/// >>> filt = pts[:, 0]  # use x-coordinate as filter
/// >>> g = synaplex.mapper.mapper(pts, filt, num_intervals=15, overlap=0.25)
/// >>> g["num_nodes"] > 0
/// True
/// >>> len(g["edges"]) >= 0
/// True
///
/// Use DBSCAN clustering with a balanced cover:
///
/// >>> g = synaplex.mapper.mapper(
/// ...     pts, filt,
/// ...     clustering="dbscan",
/// ...     clustering_threshold=0.5,
/// ...     cover_type="balanced",
/// ... )
#[gen_stub_pyfunction(module = "synaplex.mapper")]
#[pyfunction]
#[pyo3(signature = (points, filter_values, num_intervals=10, overlap=0.3, clustering="single_linkage", clustering_threshold=None, cover_type="uniform"))]
pub fn mapper<'py>(
	py: Python<'py>,
	points: PyReadonlyArray2<'_, f64>,
	filter_values: numpy::PyReadonlyArray1<'_, f64>,
	num_intervals: usize,
	overlap: f64,
	clustering: &str,
	clustering_threshold: Option<f64>,
	cover_type: &str,
) -> PyResult<Bound<'py, PyDict>> {
	let data = convert::numpy_to_array2(points);
	let fv = convert::numpy_to_array1(filter_values);

	let cover: Box<dyn CoverStrategy> = match cover_type {
		"uniform" => Box::new(UniformCover::new(num_intervals, overlap)),
		"balanced" => Box::new(BalancedCover::new(num_intervals, overlap)),
		_ => {
			return Err(pyo3::exceptions::PyValueError::new_err(format!(
				"Unknown cover type: {cover_type}"
			)));
		}
	};

	let clust: Box<dyn ClusteringAlgorithm> = match clustering {
		"single_linkage" => Box::new(SingleLinkage::new(clustering_threshold)),
		"dbscan" => Box::new(Dbscan::new(clustering_threshold.unwrap_or(1.0), 3)),
		_ => {
			return Err(pyo3::exceptions::PyValueError::new_err(format!(
				"Unknown clustering: {clustering}"
			)));
		}
	};

	let pipeline = MapperPipeline::new(cover, clust, Box::new(Euclidean));
	let graph = pipeline.run(&data, &fv);

	let dict = PyDict::new(py);

	let nodes: Vec<Vec<usize>> = graph
		.graph
		.node_weights()
		.map(|n| n.member_indices.clone())
		.collect();
	dict.set_item("nodes", nodes)?;
	dict.set_item("num_nodes", graph.num_nodes())?;
	dict.set_item("num_edges", graph.num_edges())?;

	let edges: Vec<(usize, usize, usize)> = graph
		.graph
		.edge_indices()
		.filter_map(|ei| {
			let (a, b) = graph.graph.edge_endpoints(ei)?;
			let w = graph.graph.edge_weight(ei)?;
			Some((a.index(), b.index(), w.shared_count))
		})
		.collect();
	dict.set_item("edges", edges)?;

	Ok(dict)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
	m.add_function(wrap_pyfunction!(mapper, m)?)?;
	Ok(())
}
