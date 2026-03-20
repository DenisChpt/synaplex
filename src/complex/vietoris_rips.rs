use std::collections::HashMap;

use crate::distance::DistanceMatrix;
use crate::utils::combinatorics::{self, BinomialTable};

/// Vietoris-Rips complex construction (matrix-free).
///
/// Simplices are identified by their combinatorial index (u64) rather than
/// stored as explicit vertex lists. Vertices are decoded on-the-fly from
/// the combinatorial index when needed (e.g. for boundary computation).
/// This eliminates per-simplex `Vec<u32>` allocations, dramatically
/// reducing memory usage for large complexes.
///
/// A simplex {v0, ..., vk} is included at filtration value
/// max(d(vi, vj)) for all pairs i < j.
pub struct VietorisRipsComplex {
	/// Combinatorial index (combinatorial number system) for each column.
	comb_indices: Vec<u64>,
	/// Filtration value for each column.
	filtrations: Vec<f64>,
	/// Simplex dimension for each column.
	dimensions: Vec<usize>,
	/// Number of points in the underlying point cloud.
	n: usize,
	max_dim: usize,
	threshold: f64,
	/// For each dimension d, maps combinatorial index → column position.
	dim_index: Vec<HashMap<u64, usize>>,
	binomial: BinomialTable,
}

impl VietorisRipsComplex {
	/// Build a Vietoris-Rips complex up to `max_dim` from a distance matrix.
	/// `threshold` limits the maximum filtration value (None = use enclosing radius).
	pub fn new(dm: &DistanceMatrix, max_dim: usize, threshold: Option<f64>) -> Self {
		let n = dm.size();
		let threshold = threshold.unwrap_or_else(|| dm.max_distance());
		let binomial = BinomialTable::new(n + 1, max_dim + 2);

		let mut comb_indices: Vec<u64> = Vec::new();
		let mut filtrations: Vec<f64> = Vec::new();
		let mut dims: Vec<usize> = Vec::new();
		let mut dim_index: Vec<HashMap<u64, usize>> = vec![HashMap::new(); max_dim + 1];

		// Dimension 0: vertices
		for i in 0..n {
			let comb_idx = combinatorics::simplex_to_index(&[i as u32], &binomial);
			let idx = comb_indices.len();
			dim_index[0].insert(comb_idx, idx);
			comb_indices.push(comb_idx);
			filtrations.push(0.0);
			dims.push(0);
		}

		// Dimension 1: edges
		for i in 0..n {
			for j in (i + 1)..n {
				let d = dm.get(i, j);
				if d <= threshold {
					let comb_idx =
						combinatorics::simplex_to_index(&[i as u32, j as u32], &binomial);
					let idx = comb_indices.len();
					dim_index[1].insert(comb_idx, idx);
					comb_indices.push(comb_idx);
					filtrations.push(d);
					dims.push(1);
				}
			}
		}

		// Higher dimensions: expand by checking cofaces
		if max_dim >= 2 {
			let mut prev_dim_range = n..comb_indices.len();

			for dim in 2..=max_dim {
				let mut new_entries: Vec<(u64, f64)> = Vec::new();
				let mut seen: HashMap<u64, f64> = HashMap::new();

				for fs_idx in prev_dim_range.clone() {
					// Decode vertices on-the-fly from combinatorial index
					let verts = combinatorics::index_to_simplex(
						comb_indices[fs_idx],
						dims[fs_idx],
						n,
						&binomial,
					);
					let filt = filtrations[fs_idx];
					let max_v = *verts.last().unwrap();

					for w in (max_v + 1)..(n as u32) {
						let mut all_edges_exist = true;
						let mut max_filt = filt;

						for &v in &verts {
							let d = dm.get(v as usize, w as usize);
							if d > threshold {
								all_edges_exist = false;
								break;
							}
							if d > max_filt {
								max_filt = d;
							}
						}

						if all_edges_exist {
							let mut new_verts = verts.clone();
							new_verts.push(w);
							let comb_idx =
								combinatorics::simplex_to_index(&new_verts, &binomial);
							seen.entry(comb_idx)
								.and_modify(|f| {
									if max_filt < *f {
										*f = max_filt;
									}
								})
								.or_insert(max_filt);
							if seen.len() > new_entries.len() {
								new_entries.push((comb_idx, max_filt));
							}
						}
					}
				}

				let start = comb_indices.len();
				for (comb_idx, mut filt) in new_entries {
					if let Some(&f) = seen.get(&comb_idx) {
						filt = f;
					}
					let idx = comb_indices.len();
					dim_index[dim].insert(comb_idx, idx);
					comb_indices.push(comb_idx);
					filtrations.push(filt);
					dims.push(dim);
				}
				let end = comb_indices.len();

				prev_dim_range = start..end;
				if prev_dim_range.is_empty() {
					break;
				}
			}
		}

		// Sort all simplices by (filtration, dimension, combinatorial index)
		let mut permutation: Vec<usize> = (0..comb_indices.len()).collect();
		permutation.sort_by(|&a, &b| {
			filtrations[a]
				.partial_cmp(&filtrations[b])
				.unwrap()
				.then_with(|| dims[a].cmp(&dims[b]))
				.then_with(|| comb_indices[a].cmp(&comb_indices[b]))
		});

		let sorted_comb: Vec<u64> = permutation.iter().map(|&i| comb_indices[i]).collect();
		let sorted_filt: Vec<f64> = permutation.iter().map(|&i| filtrations[i]).collect();
		let sorted_dims: Vec<usize> = permutation.iter().map(|&i| dims[i]).collect();

		// Rebuild combinatorial index with new positions
		let mut new_dim_index: Vec<HashMap<u64, usize>> = vec![HashMap::new(); max_dim + 1];
		for (new_idx, (&comb_idx, &dim)) in
			sorted_comb.iter().zip(sorted_dims.iter()).enumerate()
		{
			new_dim_index[dim].insert(comb_idx, new_idx);
		}

		Self {
			comb_indices: sorted_comb,
			filtrations: sorted_filt,
			dimensions: sorted_dims,
			n,
			max_dim,
			threshold,
			dim_index: new_dim_index,
			binomial,
		}
	}

	pub fn len(&self) -> usize {
		self.comb_indices.len()
	}

	pub fn is_empty(&self) -> bool {
		self.comb_indices.is_empty()
	}

	pub fn dimension(&self) -> usize {
		self.max_dim
	}

	pub fn threshold(&self) -> f64 {
		self.threshold
	}

	/// Filtration values for all simplices (in filtration order).
	pub fn filtrations(&self) -> &[f64] {
		&self.filtrations
	}

	/// Dimensions for all simplices (in filtration order).
	pub fn dims(&self) -> &[usize] {
		&self.dimensions
	}

	pub fn num_points(&self) -> usize {
		self.n
	}

	pub fn count_by_dim(&self, dim: usize) -> usize {
		if dim < self.dim_index.len() {
			self.dim_index[dim].len()
		} else {
			0
		}
	}

	/// Decode the vertices of the simplex at the given index.
	/// This is computed on-the-fly from the combinatorial index.
	pub fn vertices_of(&self, idx: usize) -> Vec<u32> {
		combinatorics::index_to_simplex(
			self.comb_indices[idx],
			self.dimensions[idx],
			self.n,
			&self.binomial,
		)
	}

	/// Get the column index of a simplex by its vertices.
	pub fn simplex_idx(&self, vertices: &[u32]) -> Option<usize> {
		let dim = vertices.len().checked_sub(1)?;
		if dim >= self.dim_index.len() {
			return None;
		}
		let comb_idx = combinatorics::simplex_to_index(vertices, &self.binomial);
		self.dim_index[dim].get(&comb_idx).copied()
	}

	/// Get boundary simplex indices for a given simplex index.
	/// Vertices are decoded on-the-fly from the combinatorial index.
	pub fn boundary_indices(&self, idx: usize) -> Vec<usize> {
		let dim = self.dimensions[idx];
		if dim == 0 {
			return vec![];
		}
		let face_dim = dim - 1;
		if face_dim >= self.dim_index.len() {
			return vec![];
		}
		let verts = combinatorics::index_to_simplex(
			self.comb_indices[idx],
			dim,
			self.n,
			&self.binomial,
		);
		let mut indices = Vec::with_capacity(verts.len());
		let mut face_verts = Vec::with_capacity(verts.len() - 1);
		for skip in 0..verts.len() {
			face_verts.clear();
			for (i, &v) in verts.iter().enumerate() {
				if i != skip {
					face_verts.push(v);
				}
			}
			let comb_idx = combinatorics::simplex_to_index(&face_verts, &self.binomial);
			if let Some(&pos) = self.dim_index[face_dim].get(&comb_idx) {
				indices.push(pos);
			}
		}
		indices
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::distance::{DistanceMatrix, Euclidean};
	use ndarray::array;

	#[test]
	fn test_triangle() {
		let points = array![[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]];
		let dm = DistanceMatrix::from_point_cloud(&points.view(), &Euclidean);
		let rips = VietorisRipsComplex::new(&dm, 2, None);

		assert_eq!(rips.count_by_dim(0), 3);
		assert_eq!(rips.count_by_dim(1), 3);
		assert_eq!(rips.count_by_dim(2), 1);
		assert_eq!(rips.len(), 7);
	}

	#[test]
	fn test_threshold_limits() {
		let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 10.0]];
		let dm = DistanceMatrix::from_point_cloud(&points.view(), &Euclidean);
		let rips = VietorisRipsComplex::new(&dm, 2, Some(2.0));
		assert_eq!(rips.count_by_dim(0), 3);
		assert_eq!(rips.count_by_dim(1), 1);
		assert_eq!(rips.count_by_dim(2), 0);
	}

	#[test]
	fn test_filtration_order() {
		let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]];
		let dm = DistanceMatrix::from_point_cloud(&points.view(), &Euclidean);
		let rips = VietorisRipsComplex::new(&dm, 2, None);

		// Verify filtration order: each simplex's filtration <= next
		let filts = rips.filtrations();
		let dims = rips.dims();
		for i in 0..filts.len() - 1 {
			assert!(
				filts[i] <= filts[i + 1] || dims[i] < dims[i + 1],
				"Filtration order violated at index {i}"
			);
		}
	}

	#[test]
	fn test_boundary_indices() {
		let points = array![[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]];
		let dm = DistanceMatrix::from_point_cloud(&points.view(), &Euclidean);
		let rips = VietorisRipsComplex::new(&dm, 2, None);

		let tri_idx = rips.simplex_idx(&[0, 1, 2]).unwrap();
		let bdry = rips.boundary_indices(tri_idx);
		assert_eq!(bdry.len(), 3);
	}

	#[test]
	fn test_vertices_roundtrip() {
		let points = array![[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]];
		let dm = DistanceMatrix::from_point_cloud(&points.view(), &Euclidean);
		let rips = VietorisRipsComplex::new(&dm, 2, None);

		// The triangle should decode to [0, 1, 2]
		let tri_idx = rips.simplex_idx(&[0, 1, 2]).unwrap();
		assert_eq!(rips.vertices_of(tri_idx), vec![0, 1, 2]);

		// An edge should decode correctly
		let edge_idx = rips.simplex_idx(&[0, 1]).unwrap();
		assert_eq!(rips.vertices_of(edge_idx), vec![0, 1]);
	}
}
