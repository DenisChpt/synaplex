use crate::distance::DistanceMatrix;
use crate::utils::union_find::UnionFind;

/// Trait for clustering algorithms used within each cover element.
pub trait ClusteringAlgorithm: Send + Sync {
	/// Cluster a subset of points given by `indices` using distances from `dm`.
	/// Returns a vec of clusters, each cluster being a vec of original point indices.
	fn cluster(&self, dm: &DistanceMatrix, indices: &[usize]) -> Vec<Vec<usize>>;
}

/// Single-linkage (hierarchical agglomerative) clustering.
/// Merges clusters as long as the minimum inter-cluster distance is below threshold.
pub struct SingleLinkage {
	/// Distance threshold. If None, uses the first "big gap" heuristic.
	pub threshold: Option<f64>,
}

impl SingleLinkage {
	pub fn new(threshold: Option<f64>) -> Self {
		Self { threshold }
	}
}

impl ClusteringAlgorithm for SingleLinkage {
	fn cluster(&self, dm: &DistanceMatrix, indices: &[usize]) -> Vec<Vec<usize>> {
		let n = indices.len();
		if n <= 1 {
			return vec![indices.to_vec()];
		}

		// Build sorted edge list among the subset
		let mut edges: Vec<(usize, usize, f64)> = Vec::new();
		for i in 0..n {
			for j in (i + 1)..n {
				let d = dm.get(indices[i], indices[j]);
				edges.push((i, j, d));
			}
		}
		edges.sort_unstable_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

		let threshold = self.threshold.unwrap_or_else(|| {
			// Heuristic: use the first "big gap" in the sorted edge distances
			if edges.len() < 2 {
				return f64::INFINITY;
			}
			let mut max_gap = 0.0f64;
			let mut best_threshold = edges.last().unwrap().2;
			for w in edges.windows(2) {
				let gap = w[1].2 - w[0].2;
				if gap > max_gap {
					max_gap = gap;
					best_threshold = w[0].2;
				}
			}
			best_threshold
		});

		let mut uf = UnionFind::new(n);
		for &(i, j, d) in &edges {
			if d > threshold {
				break;
			}
			uf.union(i, j);
		}

		// Collect clusters
		let mut cluster_map: std::collections::HashMap<usize, Vec<usize>> =
			std::collections::HashMap::new();

		#[allow(clippy::needless_range_loop)]
		for i in 0..n {
			let root = uf.find(i);
			cluster_map.entry(root).or_default().push(indices[i]);
		}

		cluster_map.into_values().collect()
	}
}

/// DBSCAN-style density-based clustering.
pub struct Dbscan {
	pub eps: f64,
	pub min_points: usize,
}

impl Dbscan {
	pub fn new(eps: f64, min_points: usize) -> Self {
		Self { eps, min_points }
	}
}

impl ClusteringAlgorithm for Dbscan {
	fn cluster(&self, dm: &DistanceMatrix, indices: &[usize]) -> Vec<Vec<usize>> {
		let n = indices.len();
		let mut labels = vec![None::<usize>; n];
		let mut cluster_id = 0;

		for i in 0..n {
			if labels[i].is_some() {
				continue;
			}

			// Find neighbors of i within eps
			let neighbors: Vec<usize> = (0..n)
				.filter(|&j| j != i && dm.get(indices[i], indices[j]) <= self.eps)
				.collect();

			if neighbors.len() + 1 < self.min_points {
				continue; // noise point (will be assigned later or left out)
			}

			labels[i] = Some(cluster_id);
			let mut queue = neighbors;
			let mut qi = 0;

			while qi < queue.len() {
				let j = queue[qi];
				qi += 1;

				if labels[j].is_some() {
					continue;
				}
				labels[j] = Some(cluster_id);

				let j_neighbors: Vec<usize> = (0..n)
					.filter(|&k| k != j && dm.get(indices[j], indices[k]) <= self.eps)
					.collect();

				if j_neighbors.len() + 1 >= self.min_points {
					for &k in &j_neighbors {
						if labels[k].is_none() && !queue.contains(&k) {
							queue.push(k);
						}
					}
				}
			}

			cluster_id += 1;
		}

		// Assign noise points to their nearest cluster, or put each in its own cluster
		for i in 0..n {
			if labels[i].is_none() {
				// Find nearest labeled point
				let nearest = (0..n).filter(|&j| labels[j].is_some()).min_by(|&a, &b| {
					dm.get(indices[i], indices[a])
						.partial_cmp(&dm.get(indices[i], indices[b]))
						.unwrap()
				});

				labels[i] = match nearest {
					Some(j) => labels[j],
					None => {
						let id = cluster_id;
						cluster_id += 1;
						Some(id)
					}
				};
			}
		}

		// Collect clusters
		let mut cluster_map: std::collections::HashMap<usize, Vec<usize>> =
			std::collections::HashMap::new();
		for i in 0..n {
			if let Some(label) = labels[i] {
				cluster_map.entry(label).or_default().push(indices[i]);
			}
		}

		cluster_map.into_values().collect()
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::distance::{DistanceMatrix, Euclidean};
	use ndarray::array;

	#[test]
	fn test_single_linkage() {
		let points = array![
			[0.0, 0.0],
			[0.1, 0.0],
			[0.2, 0.0], // cluster 1
			[10.0, 0.0],
			[10.1, 0.0] // cluster 2
		];
		let dm = DistanceMatrix::from_point_cloud(&points.view(), &Euclidean);
		let sl = SingleLinkage::new(Some(1.0));
		let clusters = sl.cluster(&dm, &[0, 1, 2, 3, 4]);
		assert_eq!(clusters.len(), 2);
	}

	#[test]
	fn test_dbscan() {
		let points = array![
			[0.0, 0.0],
			[0.1, 0.0],
			[0.2, 0.0],
			[10.0, 0.0],
			[10.1, 0.0],
			[10.2, 0.0]
		];
		let dm = DistanceMatrix::from_point_cloud(&points.view(), &Euclidean);
		let dbscan = Dbscan::new(0.5, 2);
		let clusters = dbscan.cluster(&dm, &[0, 1, 2, 3, 4, 5]);
		assert_eq!(clusters.len(), 2);
	}
}
