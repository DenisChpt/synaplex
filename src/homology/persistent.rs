use crate::complex::VietorisRipsComplex;
use crate::distance::DistanceMatrix;

use super::boundary::BoundaryMatrix;
use super::reduction;

/// A persistence pair: records when a topological feature is born and dies.
#[derive(Clone, Debug)]
pub struct PersistencePair {
	/// Index of the birth simplex in the filtration.
	pub birth_idx: usize,
	/// Index of the death simplex (None for essential/infinite classes).
	pub death_idx: Option<usize>,
	/// Filtration value at birth.
	pub birth: f64,
	/// Filtration value at death (f64::INFINITY for essential classes).
	pub death: f64,
	/// Homological dimension of the feature.
	pub dimension: usize,
}

impl PersistencePair {
	pub fn persistence(&self) -> f64 {
		self.death - self.birth
	}

	pub fn is_essential(&self) -> bool {
		self.death_idx.is_none()
	}
}

/// Configuration for persistent homology computation.
pub struct PersistenceConfig {
	/// Maximum homological dimension to compute.
	pub max_dimension: usize,
	/// Maximum filtration value (None = use all).
	pub threshold: Option<f64>,
	/// Use clearing optimization (recommended).
	pub clearing: bool,
}

impl Default for PersistenceConfig {
	fn default() -> Self {
		Self {
			max_dimension: 1,
			threshold: None,
			clearing: true,
		}
	}
}

/// Compute persistent homology from a distance matrix.
///
/// This is the main entry point. Delegates to the cohomology engine for
/// optimal performance:
/// - H0: Union-Find on sorted edges (O(E log E), no matrix needed)
/// - H1+: Persistent cohomology reduction with clearing
pub fn compute_persistent_homology(
	dm: &DistanceMatrix,
	config: &PersistenceConfig,
) -> Vec<PersistencePair> {
	use super::cohomology::{CohomologyConfig, compute_persistent_cohomology};

	let cohomology_config = CohomologyConfig {
		max_dimension: config.max_dimension,
		threshold: config.threshold,
		modulus: 2,
		cocycles: false,
	};

	compute_persistent_cohomology(dm, &cohomology_config).pairs
}

/// Compute persistent homology from a pre-built Vietoris-Rips complex.
pub fn compute_from_rips(
	rips: &VietorisRipsComplex,
	config: &PersistenceConfig,
) -> Vec<PersistencePair> {
	let mut bm = BoundaryMatrix::from_rips(rips);

	let pivots = if config.clearing {
		reduction::reduce_with_clearing(&mut bm)
	} else {
		reduction::reduce(&mut bm)
	};

	extract_pairs(&bm, &pivots, 0, config.max_dimension)
}

/// Compute persistent homology from a distance matrix (matrix-free).
pub fn compute_matrix_free(
	dm: &DistanceMatrix,
	config: &PersistenceConfig,
) -> Vec<PersistencePair> {
	compute_persistent_homology(dm, config)
}

/// Extract persistence pairs from the reduced boundary matrix.
fn extract_pairs(
	bm: &BoundaryMatrix<'_>,
	pivots: &[Option<usize>],
	min_dim: usize,
	max_dim: usize,
) -> Vec<PersistencePair> {
	let n = bm.num_columns();
	let mut pairs = Vec::new();
	let mut is_paired = vec![false; n];

	// Finite pairs: pivot[j] = Some(i) means (birth=i, death=j)
	for j in 0..n {
		if let Some(i) = pivots[j] {
			let dim = bm.dimensions[i];
			if dim >= min_dim && dim <= max_dim {
				pairs.push(PersistencePair {
					birth_idx: i,
					death_idx: Some(j),
					birth: bm.filtrations[i],
					death: bm.filtrations[j],
					dimension: dim,
				});
				is_paired[i] = true;
				is_paired[j] = true;
			}
		}
	}

	// Essential (infinite) classes: unpaired simplices whose columns are zero
	for j in 0..n {
		if !is_paired[j] && pivots[j].is_none() {
			let dim = bm.dimensions[j];
			if dim >= min_dim && dim <= max_dim {
				pairs.push(PersistencePair {
					birth_idx: j,
					death_idx: None,
					birth: bm.filtrations[j],
					death: f64::INFINITY,
					dimension: dim,
				});
			}
		}
	}

	pairs
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
	fn test_single_component() {
		// Two close points: H0 should show 1 essential class (1 component)
		let points = ndarray::array![[0.0, 0.0], [0.1, 0.0]];
		let dm = DistanceMatrix::from_point_cloud(&points.view(), &Euclidean);
		let pairs = compute_persistent_homology(
			&dm,
			&PersistenceConfig {
				max_dimension: 0,
				..Default::default()
			},
		);

		let essential_h0: Vec<_> = pairs
			.iter()
			.filter(|p| p.dimension == 0 && p.is_essential())
			.collect();
		assert_eq!(
			essential_h0.len(),
			1,
			"Should have exactly 1 connected component"
		);
	}

	#[test]
	fn test_circle_has_h1() {
		// Circle with 12 points: should have 1 prominent H1 class
		let points = circle_points(12);
		let dm = DistanceMatrix::from_point_cloud(&points.view(), &Euclidean);
		let pairs = compute_persistent_homology(
			&dm,
			&PersistenceConfig {
				max_dimension: 1,
				threshold: None,
				clearing: true,
			},
		);

		let h1_pairs: Vec<_> = pairs.iter().filter(|p| p.dimension == 1).collect();

		// There should be at least one H1 class
		assert!(
			!h1_pairs.is_empty(),
			"Circle should have at least one H1 feature"
		);

		// The most persistent H1 feature should have significant persistence
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
	fn test_three_clusters() {
		// Three well-separated points: H0 should show 3 births, 2 deaths
		let points = ndarray::array![[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]];
		let dm = DistanceMatrix::from_point_cloud(&points.view(), &Euclidean);
		let pairs = compute_persistent_homology(
			&dm,
			&PersistenceConfig {
				max_dimension: 0,
				..Default::default()
			},
		);

		let h0_pairs: Vec<_> = pairs.iter().filter(|p| p.dimension == 0).collect();
		let essential = h0_pairs.iter().filter(|p| p.is_essential()).count();
		let finite = h0_pairs.iter().filter(|p| !p.is_essential()).count();

		assert_eq!(essential, 1, "Should have 1 essential H0 class");
		assert_eq!(
			finite, 2,
			"Should have 2 finite H0 pairs (merging 3 components)"
		);
	}

	#[test]
	fn test_h0_union_find_matches_matrix() {
		// Verify that UF-based H0 gives same results as matrix reduction
		let points = ndarray::array![
			[0.0, 0.0],
			[1.0, 0.0],
			[0.5, 0.866],
			[5.0, 0.0],
			[6.0, 0.0]
		];
		let dm = DistanceMatrix::from_point_cloud(&points.view(), &Euclidean);

		// UF-based
		let pairs_uf = compute_persistent_homology(
			&dm,
			&PersistenceConfig {
				max_dimension: 0,
				..Default::default()
			},
		);

		// Matrix-based (via compute_from_rips)
		let rips = VietorisRipsComplex::new(&dm, 1, None);
		let pairs_mat = compute_from_rips(
			&rips,
			&PersistenceConfig {
				max_dimension: 0,
				..Default::default()
			},
		);

		let uf_essential = pairs_uf.iter().filter(|p| p.is_essential()).count();
		let mat_essential = pairs_mat.iter().filter(|p| p.is_essential()).count();
		assert_eq!(uf_essential, mat_essential, "Essential H0 count should match");

		let uf_finite = pairs_uf.iter().filter(|p| !p.is_essential()).count();
		let mat_finite = pairs_mat.iter().filter(|p| !p.is_essential()).count();
		assert_eq!(uf_finite, mat_finite, "Finite H0 count should match");

		// Death values should match
		let mut uf_deaths: Vec<f64> = pairs_uf
			.iter()
			.filter(|p| !p.is_essential())
			.map(|p| p.death)
			.collect();
		let mut mat_deaths: Vec<f64> = pairs_mat
			.iter()
			.filter(|p| !p.is_essential())
			.map(|p| p.death)
			.collect();
		uf_deaths.sort_by(|a, b| a.partial_cmp(b).unwrap());
		mat_deaths.sort_by(|a, b| a.partial_cmp(b).unwrap());
		assert_eq!(uf_deaths.len(), mat_deaths.len());
		for (a, b) in uf_deaths.iter().zip(mat_deaths.iter()) {
			assert!(
				(a - b).abs() < 1e-10,
				"Death values should match: {a} vs {b}"
			);
		}
	}
}
