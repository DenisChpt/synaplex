use super::DistanceMatrix;

/// Sparse distance matrix in CSR format.
/// Only stores distances below a threshold, drastically reducing memory
/// for large point clouds.
pub struct SparseDistanceMatrix {
	n: usize,
	row_offsets: Vec<usize>,
	col_indices: Vec<u32>,
	values: Vec<f64>,
}

impl SparseDistanceMatrix {
	/// Build from a dense distance matrix by keeping only entries below threshold.
	pub fn from_dense(dm: &DistanceMatrix, threshold: f64) -> Self {
		let n = dm.size();
		let mut row_offsets = Vec::with_capacity(n + 1);
		let mut col_indices = Vec::new();
		let mut values = Vec::new();

		row_offsets.push(0);
		for i in 0..n {
			for j in 0..n {
				if i != j {
					let d = dm.get(i, j);
					if d <= threshold {
						col_indices.push(j as u32);
						values.push(d);
					}
				}
			}
			row_offsets.push(col_indices.len());
		}

		Self {
			n,
			row_offsets,
			col_indices,
			values,
		}
	}

	pub fn size(&self) -> usize {
		self.n
	}

	/// Neighbors of point `i` with their distances, sorted by column index.
	pub fn neighbors(&self, i: usize) -> &[u32] {
		let start = self.row_offsets[i];
		let end = self.row_offsets[i + 1];
		&self.col_indices[start..end]
	}

	/// Distances to neighbors of point `i`.
	pub fn neighbor_distances(&self, i: usize) -> &[f64] {
		let start = self.row_offsets[i];
		let end = self.row_offsets[i + 1];
		&self.values[start..end]
	}

	/// Get distance between i and j, or None if not stored (above threshold).
	pub fn get(&self, i: usize, j: usize) -> Option<f64> {
		if i == j {
			return Some(0.0);
		}
		let start = self.row_offsets[i];
		let end = self.row_offsets[i + 1];
		let cols = &self.col_indices[start..end];
		let vals = &self.values[start..end];
		cols.iter()
			.position(|&c| c == j as u32)
			.map(|pos| vals[pos])
	}

	/// Check if an edge (i, j) exists below threshold.
	pub fn has_edge(&self, i: usize, j: usize) -> bool {
		if i == j {
			return true;
		}
		let start = self.row_offsets[i];
		let end = self.row_offsets[i + 1];
		self.col_indices[start..end].contains(&(j as u32))
	}

	pub fn nnz(&self) -> usize {
		self.values.len()
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::distance::{DistanceMatrix, Euclidean};
	use ndarray::array;

	#[test]
	fn test_sparse_from_dense() {
		let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 10.0]];
		let dm = DistanceMatrix::from_point_cloud(&points.view(), &Euclidean);
		let sp = SparseDistanceMatrix::from_dense(&dm, 2.0);

		// Only edge (0,1) with distance 1.0 should be within threshold
		assert!(sp.has_edge(0, 1));
		assert!(sp.has_edge(1, 0));
		assert!(!sp.has_edge(0, 2));
		assert!(!sp.has_edge(1, 2));
	}
}
