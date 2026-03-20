use ndarray::ArrayView2;
use rayon::prelude::*;

use super::metrics::DistanceMetric;

/// Dense symmetric distance matrix storing only the upper triangle.
/// For n points, stores n*(n-1)/2 entries in row-major order.
/// Entry (i, j) with i < j is at index: i*n - i*(i+1)/2 + j - i - 1.
pub struct DistanceMatrix {
	n: usize,
	data: Vec<f64>,
}

impl DistanceMatrix {
	/// Build a distance matrix from a point cloud (n_points x n_dims) in parallel.
	pub fn from_point_cloud(points: &ArrayView2<f64>, metric: &dyn DistanceMetric) -> Self {
		let n = points.nrows();
		if n == 0 {
			return Self { n: 0, data: vec![] };
		}

		// Pre-extract rows as slices for cache-friendly access
		let rows: Vec<&[f64]> = (0..n)
			.map(|i| {
				points
					.row(i)
					.to_slice()
					.expect("point cloud must be contiguous C-order")
			})
			.collect();

		let size = n * (n - 1) / 2;

		// Parallel computation: each row i computes distances to all j > i
		// Row-level parallelism avoids excessive Rayon task overhead.
		let row_data: Vec<Vec<f64>> = (0..n)
			.into_par_iter()
			.map(|i| {
				((i + 1)..n)
					.map(|j| metric.distance(rows[i], rows[j]))
					.collect()
			})
			.collect();

		let mut data = Vec::with_capacity(size);
		for row in row_data {
			data.extend(row);
		}

		Self { n, data }
	}

	/// Build from a pre-computed flat upper-triangle vector.
	pub fn from_raw(n: usize, data: Vec<f64>) -> Self {
		debug_assert_eq!(data.len(), n * (n - 1) / 2);
		Self { n, data }
	}

	#[inline]
	pub fn size(&self) -> usize {
		self.n
	}

	#[inline]
	pub fn get(&self, i: usize, j: usize) -> f64 {
		if i == j {
			return 0.0;
		}
		let (lo, hi) = if i < j { (i, j) } else { (j, i) };
		self.data[pair_to_flat(lo, hi, self.n)]
	}

	/// Raw data slice (upper triangle, row-major).
	pub fn data(&self) -> &[f64] {
		&self.data
	}

	pub fn max_distance(&self) -> f64 {
		self.data.iter().cloned().fold(0.0f64, f64::max)
	}

	/// Compute the enclosing radius (minimum radius to cover all points).
	/// This is the smallest epsilon such that the Rips complex at epsilon
	/// is contractible (has trivial homology). Used as an upper bound for
	/// persistent homology computation to avoid building unnecessary simplices.
	pub fn enclosing_radius(&self) -> f64 {
		// For each point, find the maximum distance to any other point
		// The enclosing radius is the minimum of these maximum distances
		(0..self.n)
			.map(|i| {
				// Find max distance from point i to all other points
				(0..self.n)
					.filter(|&j| i != j)
					.map(|j| self.get(i, j))
					.fold(0.0f64, f64::max)
			})
			.fold(f64::INFINITY, f64::min)
	}

	/// Compute a threshold based on k-nearest-neighbor distances.
	/// For each point, find its k-th nearest neighbor distance, then return
	/// the given percentile of these values. This avoids both the outlier
	/// problem of using the maximum and the connectivity loss of using the
	/// median, while dramatically reducing the number of edges in high-
	/// dimensional settings.
	pub fn knn_threshold(&self, k: usize, percentile: f64) -> f64 {
		let k = k.min(self.n - 1);
		let mut knn_dists = Vec::with_capacity(self.n);
		for i in 0..self.n {
			let mut dists: Vec<f64> = (0..self.n)
				.filter(|&j| j != i)
				.map(|j| self.get(i, j))
				.collect();
			dists.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
			if k > 0 && k <= dists.len() {
				knn_dists.push(dists[k - 1]);
			}
		}
		if knn_dists.is_empty() {
			return 0.0;
		}
		knn_dists.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
		let idx = ((percentile / 100.0) * (knn_dists.len() - 1) as f64).round() as usize;
		knn_dists[idx.min(knn_dists.len() - 1)]
	}

	/// Iterator over all edges (i, j, distance) sorted by distance.
	pub fn edges_sorted(&self) -> Vec<(usize, usize, f64)> {
		let mut edges: Vec<(usize, usize, f64)> = (0..self.data.len())
			.map(|flat| {
				let (i, j) = flat_to_pair(flat, self.n);
				(i, j, self.data[flat])
			})
			.collect();
		edges.sort_unstable_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
		edges
	}
}

/// Convert (i, j) with i < j to flat index in upper triangle.
#[inline]
fn pair_to_flat(i: usize, j: usize, n: usize) -> usize {
	debug_assert!(i < j && j < n);
	i * n - i * (i + 1) / 2 + j - i - 1
}

/// Convert flat index back to (i, j) pair.
#[inline]
fn flat_to_pair(flat: usize, n: usize) -> (usize, usize) {
	// i is the row: largest i such that i*n - i*(i+1)/2 <= flat
	let mut i = 0;
	let mut offset = 0;
	loop {
		let row_len = n - i - 1;
		if offset + row_len > flat {
			let j = i + 1 + (flat - offset);
			return (i, j);
		}
		offset += row_len;
		i += 1;
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use ndarray::array;

	#[test]
	fn test_pair_flat_roundtrip() {
		let n = 5;
		let mut flat = 0;
		for i in 0..n {
			for j in (i + 1)..n {
				assert_eq!(pair_to_flat(i, j, n), flat);
				assert_eq!(flat_to_pair(flat, n), (i, j));
				flat += 1;
			}
		}
	}

	#[test]
	fn test_distance_matrix_from_points() {
		use crate::distance::metrics::Euclidean;
		let points = array![[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]];
		let dm = DistanceMatrix::from_point_cloud(&points.view(), &Euclidean);
		assert_eq!(dm.size(), 3);
		assert!((dm.get(0, 1) - 3.0).abs() < 1e-10);
		assert!((dm.get(0, 2) - 4.0).abs() < 1e-10);
		assert!((dm.get(1, 2) - 5.0).abs() < 1e-10);
		assert!((dm.get(1, 0) - 3.0).abs() < 1e-10); // symmetric
	}

	#[test]
	fn test_edges_sorted() {
		use crate::distance::metrics::Euclidean;
		let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]];
		let dm = DistanceMatrix::from_point_cloud(&points.view(), &Euclidean);
		let edges = dm.edges_sorted();
		assert!(edges[0].2 <= edges[1].2);
		assert!(edges[1].2 <= edges[2].2);
	}
}
