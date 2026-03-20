use ndarray::{Array1, Array2};
use petgraph::graph::{Graph, NodeIndex};

/// A node in the Mapper graph, representing a cluster.
#[derive(Clone, Debug)]
pub struct MapperNode {
	/// Original data point indices in this cluster.
	pub member_indices: Vec<usize>,
	/// Number of members.
	pub size: usize,
	/// Centroid of the cluster in the original data space.
	pub centroid: Array1<f64>,
}

/// An edge in the Mapper graph, representing cluster overlap.
#[derive(Clone, Debug)]
pub struct MapperEdge {
	/// Number of shared data points.
	pub shared_count: usize,
}

/// The Mapper graph output.
pub struct MapperGraph {
	pub graph: Graph<MapperNode, MapperEdge>,
}

impl MapperGraph {
	/// Build from clusters and nerve edges.
	pub fn from_clusters_and_edges(
		data: &Array2<f64>,
		clusters: Vec<Vec<usize>>,
		edges: Vec<(usize, usize, usize)>,
	) -> Self {
		let mut graph = Graph::new();
		let mut node_indices: Vec<NodeIndex> = Vec::with_capacity(clusters.len());

		for cluster in &clusters {
			let ndim = data.ncols();
			let mut centroid = Array1::zeros(ndim);
			for &idx in cluster {
				centroid += &data.row(idx).to_owned();
			}
			if !cluster.is_empty() {
				centroid /= cluster.len() as f64;
			}

			let node = MapperNode {
				member_indices: cluster.clone(),
				size: cluster.len(),
				centroid,
			};
			node_indices.push(graph.add_node(node));
		}

		for (a, b, shared) in edges {
			if a < node_indices.len() && b < node_indices.len() {
				graph.add_edge(
					node_indices[a],
					node_indices[b],
					MapperEdge {
						shared_count: shared,
					},
				);
			}
		}

		Self { graph }
	}

	pub fn num_nodes(&self) -> usize {
		self.graph.node_count()
	}

	pub fn num_edges(&self) -> usize {
		self.graph.edge_count()
	}

	/// Total number of data points covered.
	pub fn total_members(&self) -> usize {
		use std::collections::HashSet;
		let all: HashSet<usize> = self
			.graph
			.node_weights()
			.flat_map(|n| n.member_indices.iter().copied())
			.collect();
		all.len()
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use ndarray::array;

	#[test]
	fn test_mapper_graph_construction() {
		let data = array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]];
		let clusters = vec![vec![0, 1], vec![1, 2], vec![2, 3]];
		let edges = vec![(0, 1, 1), (1, 2, 1)];

		let mg = MapperGraph::from_clusters_and_edges(&data, clusters, edges);
		assert_eq!(mg.num_nodes(), 3);
		assert_eq!(mg.num_edges(), 2);
		assert_eq!(mg.total_members(), 4);
	}
}
