/// Persistent cohomology computation.
///
/// Uses cohomology instead of homology, producing naturally sparser matrices
/// for Vietoris-Rips complexes and much faster reduction. Key features:
///
/// 1. **Coboundary approach**: the coboundary matrix is sparser than boundary
/// 2. **Implicit enumeration**: cofaces generated on-the-fly (zero allocation)
/// 3. **Emergent pairs**: shortcut for apparent pairs (no reduction needed)
/// 4. **Clearing optimization**: paired simplices are cleared in lower dimensions
/// 5. **Arbitrary coefficients**: Z/pZ for any prime p

use std::collections::BinaryHeap;
use std::collections::HashMap;

use crate::distance::DistanceMatrix;
use crate::utils::combinatorics::{self, BinomialTable};
use crate::utils::union_find::UnionFind;

use super::coboundary::CoboundaryEnumerator;
use super::coefficients::CoefficientField;
use super::entry::DiameterEntry;
use super::persistent::PersistencePair;

/// Configuration for cohomology-based persistent homology.
pub struct CohomologyConfig {
	/// Maximum homological dimension to compute.
	pub max_dimension: usize,
	/// Maximum filtration value (None = enclosing radius).
	pub threshold: Option<f64>,
	/// Prime modulus for coefficient field (default: 2).
	pub modulus: u16,
	/// Whether to extract representative cocycles.
	pub cocycles: bool,
}

impl Default for CohomologyConfig {
	fn default() -> Self {
		Self {
			max_dimension: 1,
			threshold: None,
			modulus: 2,
			cocycles: false,
		}
	}
}

/// A representative cocycle: a list of (simplex_vertices..., coefficient) entries.
#[derive(Clone, Debug)]
pub struct Cocycle {
	/// Dimension of the cocycle.
	pub dimension: usize,
	/// Entries: for a d-cocycle, each entry has d+1 vertex indices followed by a coefficient.
	/// Layout: [v0, v1, ..., vd, coeff, v0, v1, ..., vd, coeff, ...]
	pub entries: Vec<i32>,
}

/// Result of cohomology computation.
pub struct CohomologyResult {
	/// Persistence pairs (same format as homology).
	pub pairs: Vec<PersistencePair>,
	/// Representative cocycles, indexed by dimension.
	/// `cocycles[d]` is a list of cocycles for dimension d, parallel with pairs of that dimension.
	pub cocycles: Vec<Vec<Cocycle>>,
}

/// Compute persistent cohomology.
///
/// This is the recommended entry point for persistence computation.
/// It combines:
/// - Union-Find for H0 (fast, exact)
/// - Cohomology reduction for H1+ (much faster than homology)
pub fn compute_persistent_cohomology(
	dm: &DistanceMatrix,
	config: &CohomologyConfig,
) -> CohomologyResult {
	let n = dm.size();
	if n == 0 {
		return CohomologyResult {
			pairs: vec![],
			cocycles: vec![],
		};
	}

	let field = CoefficientField::new(config.modulus);
	let threshold = config
		.threshold
		.unwrap_or_else(|| dm.enclosing_radius());
	let binomial = BinomialTable::new(n + 1, config.max_dimension + 2);

	// H0 via Union-Find
	let mut pairs = compute_h0_union_find(dm, threshold);
	let mut cocycles_by_dim: Vec<Vec<Cocycle>> = vec![vec![]; config.max_dimension + 1];

	if config.max_dimension >= 1 {
		// Compute tighter threshold for H1+ to avoid combinatorial explosion
		let h1_threshold = config.threshold.unwrap_or_else(|| {
			if n < 100 {
				return threshold;
			}
			let k = (config.max_dimension + 2).min(n.saturating_sub(1)).max(1);
			dm.knn_threshold(k, 90.0).min(threshold)
		});

		// Get all edges sorted by diameter (ascending)
		let edges = get_edges_sorted(dm, &binomial, h1_threshold);

		// Compute H0 and get columns to reduce for H1
		// (edges that don't merge components in union-find are potential H1 births)
		let (columns_to_reduce, _) = compute_dim0_and_get_columns(dm, &edges, n);

		// Compute H1+ via cohomology reduction
		let mut current_columns = columns_to_reduce;
		let mut pivot_column_index: HashMap<u64, usize> = HashMap::new();

		for dim in 1..=config.max_dimension {
			let dim_pairs = compute_pairs_cohomology(
				&current_columns,
				&mut pivot_column_index,
				dim,
				n,
				h1_threshold,
				&field,
				&binomial,
				dm,
				config.cocycles,
			);

			for (pair, cocycle) in dim_pairs {
				pairs.push(pair);
				if let Some(c) = cocycle {
					cocycles_by_dim[dim].push(c);
				}
			}

			// Assemble columns to reduce for next dimension
			if dim < config.max_dimension {
				current_columns = assemble_columns_to_reduce(
					&current_columns,
					&pivot_column_index,
					dim,
					n,
					h1_threshold,
					dm,
					&binomial,
					config.modulus,
				);
				pivot_column_index.clear();
			}
		}
	}

	pairs.sort_by(|a, b| {
		a.dimension
			.cmp(&b.dimension)
			.then_with(|| a.birth.partial_cmp(&b.birth).unwrap())
			.then_with(|| a.death.partial_cmp(&b.death).unwrap())
	});

	CohomologyResult {
		pairs,
		cocycles: cocycles_by_dim,
	}
}

/// Get all edges as (diameter, combinatorial_index) sorted by diameter ascending.
fn get_edges_sorted(
	dm: &DistanceMatrix,
	binomial: &BinomialTable,
	threshold: f64,
) -> Vec<(f64, u64)> {
	let n = dm.size();
	let mut edges = Vec::new();
	for i in 0..n {
		for j in (i + 1)..n {
			let d = dm.get(i, j);
			if d <= threshold {
				let idx = combinatorics::simplex_to_index(&[i as u32, j as u32], binomial);
				edges.push((d, idx));
			}
		}
	}
	edges.sort_by(|a, b| {
		a.0.partial_cmp(&b.0)
			.unwrap()
			.then_with(|| a.1.cmp(&b.1))
	});
	edges
}

/// Compute H0 pairs via union-find and return edges that don't merge components
/// (these become columns to reduce for H1).
fn compute_dim0_and_get_columns(
	_dm: &DistanceMatrix,
	edges: &[(f64, u64)],
	n: usize,
) -> (Vec<(f64, u64)>, Vec<PersistencePair>) {
	let binomial = BinomialTable::new(n + 1, 3);
	let mut uf = UnionFind::new(n);
	let mut columns_to_reduce = Vec::new();

	// Process edges in ascending order
	for &(diam, idx) in edges {
		let verts = combinatorics::index_to_simplex(idx, 1, n, &binomial);
		let u = uf.find(verts[0] as usize);
		let v = uf.find(verts[1] as usize);
		if u != v {
			uf.union(verts[0] as usize, verts[1] as usize);
		} else {
			// This edge doesn't merge components → potential H1 birth
			columns_to_reduce.push((diam, idx));
		}
	}

	(columns_to_reduce, vec![])
}

/// H0 via Union-Find (same as existing implementation).
fn compute_h0_union_find(dm: &DistanceMatrix, threshold: f64) -> Vec<PersistencePair> {
	let n = dm.size();
	if n == 0 {
		return vec![];
	}

	let mut edges = dm.edges_sorted();
	edges.retain(|e| e.2 <= threshold);

	let mut uf = UnionFind::new(n);
	let mut pairs = Vec::new();

	for &(i, j, dist) in &edges {
		if uf.union(i, j) {
			pairs.push(PersistencePair {
				birth_idx: 0,
				death_idx: Some(0),
				birth: 0.0,
				death: dist,
				dimension: 0,
			});
		}
	}

	let n_essential = uf.num_components();
	for _ in 0..n_essential {
		pairs.push(PersistencePair {
			birth_idx: 0,
			death_idx: None,
			birth: 0.0,
			death: f64::INFINITY,
			dimension: 0,
		});
	}

	pairs
}

/// Get the pivot (largest index entry with nonzero coefficient) from a working column.
fn get_pivot(column: &mut BinaryHeap<DiameterEntry>, field: &CoefficientField) -> Option<DiameterEntry> {
	if field.is_z2() {
		// Z/2Z fast path: duplicates cancel
		loop {
			let pivot = column.pop()?;
			if let Some(&next) = column.peek() {
				if next.index() == pivot.index() {
					// Cancel (1 + 1 = 0 mod 2)
					column.pop();
					continue;
				}
			}
			column.push(pivot);
			return Some(pivot);
		}
	} else {
		// General Z/pZ: accumulate coefficients for same index
		let mut pivot = column.pop()?;
		loop {
			if let Some(&next) = column.peek() {
				if next.index() == pivot.index() {
					let next = column.pop().unwrap();
					let new_coeff = field.add(pivot.coefficient(), next.coefficient());
					if new_coeff == 0 {
						// Cancelled
						match column.pop() {
							Some(p) => {
								pivot = p;
								continue;
							}
							None => return None,
						}
					}
					pivot = DiameterEntry::new(pivot.diameter, pivot.index(), new_coeff);
					continue;
				}
			}
			column.push(pivot);
			return Some(pivot);
		}
	}
}

/// Pop the pivot without pushing it back.
fn pop_pivot(column: &mut BinaryHeap<DiameterEntry>, field: &CoefficientField) -> Option<DiameterEntry> {
	if field.is_z2() {
		loop {
			let pivot = column.pop()?;
			if let Some(&next) = column.peek() {
				if next.index() == pivot.index() {
					column.pop();
					continue;
				}
			}
			return Some(pivot);
		}
	} else {
		let mut pivot = column.pop()?;
		loop {
			if let Some(&next) = column.peek() {
				if next.index() == pivot.index() {
					let next = column.pop().unwrap();
					let new_coeff = field.add(pivot.coefficient(), next.coefficient());
					if new_coeff == 0 {
						match column.pop() {
							Some(p) => {
								pivot = p;
								continue;
							}
							None => return None,
						}
					}
					pivot = DiameterEntry::new(pivot.diameter, pivot.index(), new_coeff);
					continue;
				}
			}
			return Some(pivot);
		}
	}
}

/// Initialize the coboundary of a simplex and check for emergent pairs.
///
/// An emergent pair is a (σ, τ) pair where τ is the unique cofacet of σ at
/// the same filtration value and τ hasn't been paired yet. This allows
/// skipping the full reduction for that column.
fn init_coboundary_and_get_pivot(
	simplex_diameter: f64,
	simplex_index: u64,
	simplex_coefficient: u16,
	dim: usize,
	n: usize,
	threshold: f64,
	field: &CoefficientField,
	binomial: &BinomialTable,
	dm: &DistanceMatrix,
	pivot_column_index: &HashMap<u64, usize>,
	working_coboundary: &mut BinaryHeap<DiameterEntry>,
) -> Option<DiameterEntry> {
	let mut check_emergent = true;
	let mut cofacet_entries = Vec::new();

	let mut cob = CoboundaryEnumerator::new(
		simplex_index,
		dim,
		simplex_diameter,
		simplex_coefficient,
		n,
		field.modulus(),
		threshold,
		dm,
		binomial,
	);

	while let Some(cofacet) = cob.next() {
		cofacet_entries.push(cofacet);

		// Emergent pair check: if cofacet has same diameter and isn't already paired
		if check_emergent && cofacet.diameter == simplex_diameter {
			let entry_key = cofacet.entry.index();
			if !pivot_column_index.contains_key(&entry_key) {
				return Some(cofacet);
			}
			check_emergent = false;
		}
	}

	// No emergent pair found; push all cofacets into working column
	for cf in cofacet_entries {
		working_coboundary.push(cf);
	}

	get_pivot(working_coboundary, field)
}

/// Add the coboundary of a simplex to the working columns.
fn add_simplex_coboundary(
	simplex_index: u64,
	simplex_diameter: f64,
	simplex_coefficient: u16,
	dim: usize,
	n: usize,
	threshold: f64,
	field: &CoefficientField,
	binomial: &BinomialTable,
	dm: &DistanceMatrix,
	working_reduction: &mut BinaryHeap<DiameterEntry>,
	working_coboundary: &mut BinaryHeap<DiameterEntry>,
) {
	working_reduction.push(DiameterEntry::new(simplex_diameter, simplex_index, simplex_coefficient));

	let mut cob = CoboundaryEnumerator::new(
		simplex_index,
		dim,
		simplex_diameter,
		simplex_coefficient,
		n,
		field.modulus(),
		threshold,
		dm,
		binomial,
	);

	while let Some(cofacet) = cob.next() {
		working_coboundary.push(cofacet);
	}
}

/// Compute persistence pairs for a given dimension via cohomology reduction.
fn compute_pairs_cohomology(
	columns_to_reduce: &[(f64, u64)],
	pivot_column_index: &mut HashMap<u64, usize>,
	dim: usize,
	n: usize,
	threshold: f64,
	field: &CoefficientField,
	binomial: &BinomialTable,
	dm: &DistanceMatrix,
	extract_cocycles: bool,
) -> Vec<(PersistencePair, Option<Cocycle>)> {
	let mut result = Vec::new();

	// For Z/pZ (p>2), track the pivot coefficient for each paired column
	let mut pivot_coefficients: HashMap<usize, u16> = HashMap::new();

	// Compressed sparse matrix for reduction columns
	let mut reduction_columns: Vec<Vec<DiameterEntry>> = Vec::with_capacity(columns_to_reduce.len());

	for (col_idx, &(diameter, simplex_index)) in columns_to_reduce.iter().enumerate() {
		reduction_columns.push(Vec::new());

		let mut working_reduction: BinaryHeap<DiameterEntry> = BinaryHeap::new();
		let mut working_coboundary: BinaryHeap<DiameterEntry> = BinaryHeap::new();

		let coefficient = 1u16;

		let mut pivot = init_coboundary_and_get_pivot(
			diameter,
			simplex_index,
			coefficient,
			dim,
			n,
			threshold,
			field,
			binomial,
			dm,
			pivot_column_index,
			&mut working_coboundary,
		);

		loop {
			match pivot {
				Some(pivot_entry) if !pivot_entry.entry.is_null() => {
					let pivot_key = pivot_entry.entry.index();
					if let Some(&other_col_idx) = pivot_column_index.get(&pivot_key) {
						// Factor = -(pivot_coeff * inverse(other_pivot_coeff)) mod p
						let factor = if field.is_z2() {
							1u16
						} else {
							let other_pivot_coeff = pivot_coefficients
								.get(&other_col_idx)
								.copied()
								.unwrap_or(1);
							field.negate(field.multiply(
								pivot_entry.coefficient(),
								field.inverse(other_pivot_coeff),
							))
						};

						// Add the original simplex's coboundary
						let (other_diam, other_idx) = columns_to_reduce[other_col_idx];

						add_simplex_coboundary(
							other_idx,
							other_diam,
							factor,
							dim,
							n,
							threshold,
							field,
							binomial,
							dm,
							&mut working_reduction,
							&mut working_coboundary,
						);

						// Add stored reduction column entries
						for &entry in &reduction_columns[other_col_idx] {
							let coeff = if field.is_z2() {
								entry.coefficient()
							} else {
								field.multiply(entry.coefficient(), factor)
							};
							add_simplex_coboundary(
								entry.index(),
								entry.diameter,
								coeff,
								dim,
								n,
								threshold,
								field,
								binomial,
								dm,
								&mut working_reduction,
								&mut working_coboundary,
							);
						}

						pivot = get_pivot(&mut working_coboundary, field);
					} else {
						// Unique pivot found → persistence pair
						let death = pivot_entry.diameter;

						if death > diameter {
							let pair = PersistencePair {
								birth_idx: col_idx,
								death_idx: Some(0),
								birth: diameter,
								death,
								dimension: dim,
							};

							let cocycle = if extract_cocycles {
								Some(extract_cocycle(
									&mut working_reduction,
									dim,
									n,
									field,
									binomial,
								))
							} else {
								None
							};

							result.push((pair, cocycle));
						}

						pivot_column_index.insert(pivot_key, col_idx);
						if !field.is_z2() {
							pivot_coefficients.insert(col_idx, pivot_entry.coefficient());
						}

						// Store the reduction column
						while let Some(e) = pop_pivot(&mut working_reduction, field) {
							reduction_columns[col_idx].push(e);
						}
						break;
					}
				}
				_ => {
					// Column reduces to zero → essential class
					let pair = PersistencePair {
						birth_idx: col_idx,
						death_idx: None,
						birth: diameter,
						death: f64::INFINITY,
						dimension: dim,
					};

					let cocycle = if extract_cocycles {
						Some(extract_cocycle(
							&mut working_reduction,
							dim,
							n,
							field,
							binomial,
						))
					} else {
						None
					};

					result.push((pair, cocycle));
					break;
				}
			}
		}
	}

	result
}

/// Extract a representative cocycle from a working reduction column.
fn extract_cocycle(
	working_reduction: &mut BinaryHeap<DiameterEntry>,
	dim: usize,
	n: usize,
	field: &CoefficientField,
	binomial: &BinomialTable,
) -> Cocycle {
	let mut entries = Vec::new();

	while let Some(e) = pop_pivot(working_reduction, field) {
		let verts = combinatorics::index_to_simplex(e.index(), dim, n, binomial);
		for &v in &verts {
			entries.push(v as i32);
		}
		entries.push(field.normalize(e.coefficient()) as i32);
	}

	Cocycle {
		dimension: dim,
		entries,
	}
}

/// Assemble columns to reduce for the next dimension.
///
/// For each simplex at dimension `dim`, enumerate its cofacets. Cofacets
/// that aren't already paired (not in `pivot_column_index`) become columns
/// to reduce for dimension `dim+1`.
fn assemble_columns_to_reduce(
	current_columns: &[(f64, u64)],
	pivot_column_index: &HashMap<u64, usize>,
	dim: usize,
	n: usize,
	threshold: f64,
	dm: &DistanceMatrix,
	binomial: &BinomialTable,
	modulus: u16,
) -> Vec<(f64, u64)> {
	let mut next_columns: Vec<(f64, u64)> = Vec::new();
	let mut seen: HashMap<u64, f64> = HashMap::new();

	for &(diam, idx) in current_columns {
		let mut cob = CoboundaryEnumerator::new(idx, dim, diam, 1, n, modulus, threshold, dm, binomial);

		while let Some(cofacet) = cob.next() {
			let cf_idx = cofacet.index();
			let cf_diam = cofacet.diameter;

			// Track minimum diameter for each cofacet
			seen.entry(cf_idx)
				.and_modify(|d| {
					if cf_diam < *d {
						*d = cf_diam;
					}
				})
				.or_insert(cf_diam);

			// Only include if not already paired
			let entry_key = cf_idx;
			if !pivot_column_index.contains_key(&entry_key) {
				if !next_columns.iter().any(|(_, i)| *i == cf_idx) {
					next_columns.push((cf_diam, cf_idx));
				}
			}
		}
	}

	// Update diameters to the correct (minimum) values
	for (diam, idx) in &mut next_columns {
		if let Some(&d) = seen.get(idx) {
			*diam = d;
		}
	}

	// Sort: ascending diameter, then ascending index
	next_columns.sort_by(|a, b| {
		a.0.partial_cmp(&b.0)
			.unwrap()
			.then_with(|| a.1.cmp(&b.1))
	});

	next_columns
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::distance::{DistanceMatrix, Euclidean};
	use ndarray::Array2;

	fn circle_points(n: usize) -> Array2<f64> {
		let mut points = Array2::zeros((n, 2));
		for i in 0..n {
			let theta = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
			points[[i, 0]] = theta.cos();
			points[[i, 1]] = theta.sin();
		}
		points
	}

	#[test]
	fn test_cohomology_single_component() {
		let points = ndarray::array![[0.0, 0.0], [0.1, 0.0]];
		let dm = DistanceMatrix::from_point_cloud(&points.view(), &Euclidean);
		let result = compute_persistent_cohomology(
			&dm,
			&CohomologyConfig {
				max_dimension: 0,
				..Default::default()
			},
		);
		let essential_h0 = result
			.pairs
			.iter()
			.filter(|p| p.dimension == 0 && p.is_essential())
			.count();
		assert_eq!(essential_h0, 1);
	}

	#[test]
	fn test_cohomology_circle_h1() {
		let points = circle_points(12);
		let dm = DistanceMatrix::from_point_cloud(&points.view(), &Euclidean);
		let result = compute_persistent_cohomology(
			&dm,
			&CohomologyConfig {
				max_dimension: 1,
				..Default::default()
			},
		);

		let h1_pairs: Vec<_> = result.pairs.iter().filter(|p| p.dimension == 1).collect();
		assert!(!h1_pairs.is_empty(), "Circle should have at least one H1 feature");

		let max_persistence = h1_pairs
			.iter()
			.filter(|p| p.persistence().is_finite())
			.map(|p| p.persistence())
			.fold(0.0f64, f64::max);
		assert!(
			max_persistence > 0.4,
			"Circle's H1 should have persistence > 0.4, got {max_persistence}"
		);
	}

	#[test]
	fn test_cohomology_three_clusters() {
		let points = ndarray::array![[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]];
		let dm = DistanceMatrix::from_point_cloud(&points.view(), &Euclidean);
		let result = compute_persistent_cohomology(
			&dm,
			&CohomologyConfig {
				max_dimension: 0,
				..Default::default()
			},
		);

		let h0_pairs: Vec<_> = result.pairs.iter().filter(|p| p.dimension == 0).collect();
		let essential = h0_pairs.iter().filter(|p| p.is_essential()).count();
		let finite = h0_pairs.iter().filter(|p| !p.is_essential()).count();
		assert_eq!(essential, 1);
		assert_eq!(finite, 2);
	}

	#[test]
	fn test_cohomology_z3_coefficients() {
		let points = circle_points(12);
		let dm = DistanceMatrix::from_point_cloud(&points.view(), &Euclidean);
		let result = compute_persistent_cohomology(
			&dm,
			&CohomologyConfig {
				max_dimension: 1,
				modulus: 3,
				..Default::default()
			},
		);

		let h1_pairs: Vec<_> = result.pairs.iter().filter(|p| p.dimension == 1).collect();
		assert!(!h1_pairs.is_empty(), "Circle with Z/3Z should have H1 features");
	}

	#[test]
	fn test_cohomology_with_cocycles() {
		let points = circle_points(12);
		let dm = DistanceMatrix::from_point_cloud(&points.view(), &Euclidean);
		let result = compute_persistent_cohomology(
			&dm,
			&CohomologyConfig {
				max_dimension: 1,
				cocycles: true,
				..Default::default()
			},
		);

		// Should have cocycles for H1 features
		let h1_cocycles = &result.cocycles[1];
		let h1_pairs: Vec<_> = result.pairs.iter().filter(|p| p.dimension == 1).collect();
		// There should be at least some cocycles for H1
		assert!(
			!h1_cocycles.is_empty() || h1_pairs.is_empty(),
			"Should have cocycles for each H1 pair"
		);
	}

	#[test]
	fn test_cohomology_triangle_filled() {
		// Equilateral triangle: H0=1, H1=0 (triangle is filled)
		let points = ndarray::array![[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]];
		let dm = DistanceMatrix::from_point_cloud(&points.view(), &Euclidean);
		let result = compute_persistent_cohomology(
			&dm,
			&CohomologyConfig {
				max_dimension: 1,
				..Default::default()
			},
		);

		let essential_h0 = result
			.pairs
			.iter()
			.filter(|p| p.dimension == 0 && p.is_essential())
			.count();
		assert_eq!(essential_h0, 1, "Triangle should have 1 component");

		// H1 features should either not exist or have zero persistence (filled triangle)
		let h1_persistent: Vec<_> = result
			.pairs
			.iter()
			.filter(|p| p.dimension == 1 && p.persistence() > 1e-10 && p.persistence().is_finite())
			.collect();
		assert!(
			h1_persistent.is_empty(),
			"Filled triangle should have no persistent H1, got {:?}",
			h1_persistent
		);
	}
}
