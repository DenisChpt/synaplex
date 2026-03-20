pub mod clustering;
pub mod cover;
pub mod graph;
pub mod nerve;

pub use clustering::{ClusteringAlgorithm, Dbscan, SingleLinkage};
pub use cover::{BalancedCover, CoverElement, CoverStrategy, UniformCover};
pub use graph::{MapperEdge, MapperGraph, MapperNode};
pub use nerve::build_nerve;

use crate::distance::{DistanceMatrix, DistanceMetric, Euclidean};
use ndarray::{Array1, Array2};

/// Full Mapper pipeline: data -> filter -> cover -> cluster -> nerve -> graph.
pub struct MapperPipeline {
	pub cover: Box<dyn CoverStrategy>,
	pub clustering: Box<dyn ClusteringAlgorithm>,
	pub metric: Box<dyn DistanceMetric>,
}

impl Default for MapperPipeline {
	fn default() -> Self {
		Self {
			cover: Box::new(UniformCover::new(10, 0.3)),
			clustering: Box::new(SingleLinkage::new(None)),
			metric: Box::new(Euclidean),
		}
	}
}

impl MapperPipeline {
	pub fn new(
		cover: Box<dyn CoverStrategy>,
		clustering: Box<dyn ClusteringAlgorithm>,
		metric: Box<dyn DistanceMetric>,
	) -> Self {
		Self {
			cover,
			clustering,
			metric,
		}
	}

	/// Run the full Mapper pipeline.
	///
	/// - `data`: point cloud (n_points x n_dims)
	/// - `filter_values`: filter function output per point (length n_points)
	pub fn run(&self, data: &Array2<f64>, filter_values: &Array1<f64>) -> MapperGraph {
		// 1. Build the cover on the filter range
		let fv: Vec<f64> = filter_values.iter().copied().collect();
		let cover_elements = self.cover.cover(&fv);

		// 2. For each cover element, cluster the pulled-back points
		let dm = DistanceMatrix::from_point_cloud(&data.view(), self.metric.as_ref());
		let mut all_clusters: Vec<Vec<usize>> = Vec::new();

		for elem in &cover_elements {
			if elem.point_indices.is_empty() {
				continue;
			}
			let clusters = self.clustering.cluster(&dm, &elem.point_indices);
			all_clusters.extend(clusters);
		}

		// 3. Build nerve (connect clusters that share points)
		let nerve_edges = build_nerve(&all_clusters);

		// 4. Build the graph
		MapperGraph::from_clusters_and_edges(data, all_clusters, nerve_edges)
	}
}
