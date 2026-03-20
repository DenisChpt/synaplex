use rayon::prelude::*;

use super::DistanceMatrix;

/// k-Nearest Neighbors graph for sparse Vietoris-Rips optimization.
/// Each point stores its k nearest neighbors and their distances.
pub struct KnnGraph {
	k: usize,
	/// neighbors[i] = [(neighbor_index, distance), ...] sorted by distance.
	neighbors: Vec<Vec<(u32, f64)>>,
}

impl KnnGraph {
	/// Build a k-NN graph from a dense distance matrix, in parallel.
	pub fn from_distance_matrix(dm: &DistanceMatrix, k: usize) -> Self {
		let n = dm.size();
		let k = k.min(n - 1);

		let neighbors: Vec<Vec<(u32, f64)>> = (0..n)
			.into_par_iter()
			.map(|i| {
				let mut dists: Vec<(u32, f64)> = (0..n)
					.filter(|&j| j != i)
					.map(|j| (j as u32, dm.get(i, j)))
					.collect();
				dists.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
				dists.truncate(k);
				dists
			})
			.collect();

		Self { k, neighbors }
	}

	pub fn k(&self) -> usize {
		self.k
	}

	/// Get the k nearest neighbors of point i.
	pub fn neighbors(&self, i: usize) -> &[(u32, f64)] {
		&self.neighbors[i]
	}

	/// Maximum k-NN distance across all points (useful as Rips threshold).
	pub fn max_knn_distance(&self) -> f64 {
		self.neighbors
			.par_iter()
			.map(|nn| nn.last().map_or(0.0, |&(_, d)| d))
			.reduce(|| 0.0f64, f64::max)
	}

	pub fn size(&self) -> usize {
		self.neighbors.len()
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::distance::{DistanceMatrix, Euclidean};
	use ndarray::array;

	#[test]
	fn test_knn_graph() {
		let points = array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [10.0, 0.0]];
		let dm = DistanceMatrix::from_point_cloud(&points.view(), &Euclidean);
		let knn = KnnGraph::from_distance_matrix(&dm, 2);

		assert_eq!(knn.k(), 2);
		let nn0 = knn.neighbors(0);
		assert_eq!(nn0[0].0, 1); // closest to 0 is 1
		assert_eq!(nn0[1].0, 2); // second closest is 2
	}
}
