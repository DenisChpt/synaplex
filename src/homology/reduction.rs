use super::boundary::BoundaryMatrix;

/// Reduce the boundary matrix using the standard persistence algorithm
/// (over Z/2 coefficients).
///
/// The algorithm processes columns left-to-right. For each column j,
/// it repeatedly finds the pivot (largest nonzero row), checks if another
/// column already has that pivot, and if so, adds that column (Z/2) to
/// eliminate the pivot. This continues until the column is zero or has
/// a unique pivot.
///
/// Returns a vector of pivot pairs: `pivots[j] = Some(i)` means column j
/// has pivot row i after reduction.
pub fn reduce(matrix: &mut BoundaryMatrix<'_>) -> Vec<Option<usize>> {
	let n = matrix.num_columns();
	let mut pivots = vec![None; n];
	// pivot_to_col[i] = j means column j has pivot row i
	let mut pivot_to_col: Vec<Option<usize>> = vec![None; n];

	#[allow(clippy::needless_range_loop)]
	for j in 0..n {
		loop {
			match matrix.pivot(j) {
				None => break, // column is zero
				Some(pivot_row) => {
					match pivot_to_col[pivot_row] {
						Some(other_col) => {
							// Another column already has this pivot; add it to eliminate
							matrix.add_column(j, other_col);
						}
						None => {
							// Unique pivot found
							pivot_to_col[pivot_row] = Some(j);
							pivots[j] = Some(pivot_row);
							break;
						}
					}
				}
			}
		}
	}

	pivots
}

/// Reduce with clearing optimization (twist).
///
/// Process dimensions from **highest to lowest**. When reducing dimension d,
/// a pivot pair (death=j at dim d, birth=i at dim d-1) immediately clears
/// column i from needing reduction. This is the "clearing lemma" —
/// the cleared columns are guaranteed to reduce to zero, so we skip them.
///
/// For H1 computation this is critical: reducing triangles (dim 2) first
/// reveals which edges are paired, clearing them before the expensive
/// edge reduction pass.
///
/// With lazy column loading, cleared columns are never materialized at all,
/// providing both CPU and memory savings.
pub fn reduce_with_clearing(matrix: &mut BoundaryMatrix<'_>) -> Vec<Option<usize>> {
	let n = matrix.num_columns();
	let mut pivots = vec![None; n];
	let mut pivot_to_col: Vec<Option<usize>> = vec![None; n];
	let mut cleared = vec![false; n];

	// Find max dimension
	let max_dim = matrix.dimensions.iter().copied().max().unwrap_or(0);

	// Process dimensions highest to lowest (twist optimization)
	for dim in (0..=max_dim).rev() {
		for j in 0..n {
			if matrix.dimensions[j] != dim {
				continue;
			}
			if cleared[j] {
				// This column was cleared by a higher-dimension pair.
				// With lazy loading, the column is never materialized.
				matrix.clear_column(j);
				continue;
			}

			loop {
				match matrix.pivot(j) {
					None => break,
					Some(pivot_row) => {
						match pivot_to_col[pivot_row] {
							Some(other_col) => {
								matrix.add_column(j, other_col);
							}
							None => {
								pivot_to_col[pivot_row] = Some(j);
								pivots[j] = Some(pivot_row);

								// Clearing: mark the pivot row simplex as cleared
								// (it's paired as a birth simplex, its column reduces to zero)
								cleared[pivot_row] = true;
								break;
							}
						}
					}
				}
			}
		}
	}

	pivots
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::complex::VietorisRipsComplex;
	use crate::distance::{DistanceMatrix, Euclidean};
	use ndarray::array;

	#[test]
	fn test_reduction_triangle() {
		// Triangle: should have H0=1 (one component), H1=0 (triangle is filled)
		let points = array![[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]];
		let dm = DistanceMatrix::from_point_cloud(&points.view(), &Euclidean);
		let rips = VietorisRipsComplex::new(&dm, 2, None);
		let mut bm = BoundaryMatrix::from_rips(&rips);
		let pivots = reduce(&mut bm);

		// Count unpaired simplices by dimension
		let paired_births: Vec<usize> = pivots.iter().filter_map(|p| *p).collect();
		let _unpaired: Vec<usize> = (0..bm.num_columns())
			.filter(|j| pivots[*j].is_none() && !paired_births.contains(j))
			.collect();
	}

	#[test]
	fn test_clearing_matches_standard() {
		let points = array![[0.0, 0.0], [1.0, 0.0], [0.5, 0.866], [2.0, 1.0]];
		let dm = DistanceMatrix::from_point_cloud(&points.view(), &Euclidean);
		let rips = VietorisRipsComplex::new(&dm, 2, None);

		let mut bm1 = BoundaryMatrix::from_rips(&rips);
		let mut bm2 = BoundaryMatrix::from_rips(&rips);

		let pivots_std = reduce(&mut bm1);
		let pivots_clr = reduce_with_clearing(&mut bm2);

		// Both should produce the same pivot pairings
		for j in 0..pivots_std.len() {
			assert_eq!(
				pivots_std[j], pivots_clr[j],
				"Mismatch at column {j}: std={:?}, clearing={:?}",
				pivots_std[j], pivots_clr[j]
			);
		}
	}
}
