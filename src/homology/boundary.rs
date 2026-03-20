use std::collections::BinaryHeap;

use crate::complex::vietoris_rips::VietorisRipsComplex;

/// Lazy boundary matrix over Z/2 coefficients (matrix-free approach).
///
/// Columns are generated on-demand from the Vietoris-Rips complex during
/// reduction. Cleared columns are never materialized, saving significant
/// memory on large complexes.
///
/// Uses a heap-based representation for columns. Adding two columns is
/// done by pushing elements onto the heap, and duplicates cancel (Z/2)
/// lazily when the pivot is queried.
pub struct BoundaryMatrix<'a> {
	/// columns[j] = None (not yet loaded) or Some(heap of row indices).
	/// Lazy loading: columns are computed from the complex on first access.
	columns: Vec<Option<BinaryHeap<usize>>>,
	/// Dimension of the simplex corresponding to each column.
	pub dimensions: Vec<usize>,
	/// Filtration value of each simplex.
	pub filtrations: Vec<f64>,
	/// Reference to the complex for on-demand boundary computation.
	complex: &'a VietorisRipsComplex,
}

impl<'a> BoundaryMatrix<'a> {
	/// Build a lazy boundary matrix from a Vietoris-Rips complex.
	///
	/// Only borrows metadata (dimensions, filtrations) from the complex.
	/// Boundary columns are computed on-demand during reduction, not upfront.
	pub fn from_rips(complex: &'a VietorisRipsComplex) -> Self {
		let n = complex.len();

		Self {
			columns: vec![None; n],
			dimensions: complex.dims().to_vec(),
			filtrations: complex.filtrations().to_vec(),
			complex,
		}
	}

	pub fn num_columns(&self) -> usize {
		self.columns.len()
	}

	/// Ensure column j is loaded into memory. If it hasn't been accessed
	/// yet, compute its boundary from the complex.
	#[inline]
	fn ensure_loaded(&mut self, j: usize) {
		if self.columns[j].is_none() {
			let heap = if self.dimensions[j] == 0 {
				BinaryHeap::new()
			} else {
				BinaryHeap::from(self.complex.boundary_indices(j))
			};
			self.columns[j] = Some(heap);
		}
	}

	/// Get the pivot (largest nonzero row index) of column j,
	/// cancelling duplicate entries over Z/2 lazily.
	/// Returns None if the column is zero.
	#[inline]
	pub fn pivot(&mut self, j: usize) -> Option<usize> {
		self.ensure_loaded(j);
		let heap = self.columns[j].as_mut().unwrap();
		loop {
			heap.peek()?;
			let top_val = heap.pop().unwrap();
			if let Some(&next) = heap.peek() {
				if next == top_val {
					// Cancel: 1 + 1 = 0 mod 2
					heap.pop();
					continue;
				}
			}
			// Unique top — push it back and return
			heap.push(top_val);
			return Some(top_val);
		}
	}

	/// Add column `src` to column `dst` over Z/2.
	/// Instead of computing symmetric difference, we push all elements
	/// of src onto dst's heap. Duplicates cancel lazily when pivot() is called.
	pub fn add_column(&mut self, dst: usize, src: usize) {
		self.ensure_loaded(dst);
		self.ensure_loaded(src);
		// Clone src's elements (iterate without borrowing dst mutably).
		let src_elements: Vec<usize> = self.columns[src]
			.as_ref()
			.unwrap()
			.iter()
			.copied()
			.collect();
		self.columns[dst].as_mut().unwrap().extend(src_elements);
	}

	/// Clear column j (set to zero) without loading it first.
	/// This is the key win: cleared columns are never materialized.
	pub fn clear_column(&mut self, j: usize) {
		self.columns[j] = Some(BinaryHeap::new());
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::distance::{DistanceMatrix, Euclidean};
	use ndarray::array;

	#[test]
	fn test_boundary_matrix_triangle() {
		let points = array![[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]];
		let dm = DistanceMatrix::from_point_cloud(&points.view(), &Euclidean);
		let rips = VietorisRipsComplex::new(&dm, 2, None);
		let mut bm = BoundaryMatrix::from_rips(&rips);

		assert_eq!(bm.num_columns(), 7); // 3 verts + 3 edges + 1 triangle

		// Vertices have empty boundary
		for i in 0..3 {
			assert!(bm.pivot(i).is_none());
		}

		// The triangle should have a pivot (one of its boundary edges)
		let tri_pivot = bm.pivot(6);
		assert!(tri_pivot.is_some());
	}

	#[test]
	fn test_z2_column_addition() {
		let points = array![[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]];
		let dm = DistanceMatrix::from_point_cloud(&points.view(), &Euclidean);
		let rips = VietorisRipsComplex::new(&dm, 2, None);
		let mut bm = BoundaryMatrix::from_rips(&rips);

		// Adding a column's content then checking that double-add cancels
		if bm.num_columns() > 5 {
			// Get pivot of col 4 before
			let pivot_before = bm.pivot(4);
			assert!(pivot_before.is_some());

			// Add col 5 twice — should cancel out and leave col 4 unchanged
			bm.add_column(4, 5);
			bm.add_column(4, 5);
			let pivot_after = bm.pivot(4);
			assert_eq!(pivot_before, pivot_after);
		}
	}

	#[test]
	fn test_lazy_loading() {
		let points = array![[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]];
		let dm = DistanceMatrix::from_point_cloud(&points.view(), &Euclidean);
		let rips = VietorisRipsComplex::new(&dm, 2, None);
		let mut bm = BoundaryMatrix::from_rips(&rips);

		// Initially no columns are loaded
		assert!(bm.columns.iter().all(|c| c.is_none()));

		// Accessing pivot loads the column
		let _ = bm.pivot(6);
		assert!(bm.columns[6].is_some());

		// Clearing without loading
		let mut bm2 = BoundaryMatrix::from_rips(&rips);
		bm2.clear_column(3);
		assert!(bm2.columns[3].is_some()); // set to empty heap
		// Other columns remain unloaded
		assert!(bm2.columns[0].is_none());
	}
}
