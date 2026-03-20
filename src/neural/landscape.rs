use ndarray::{Array1, Array2};

use crate::diagram::PersistenceDiagram;
use crate::distance::{DistanceMatrix, Euclidean};
use crate::homology::{CohomologyConfig, compute_persistent_cohomology};

/// Analyze the topology of a loss landscape by sampling.
///
/// Given a function that evaluates loss at a point in parameter space,
/// samples the landscape along random directions and computes
/// persistent homology of the resulting sublevel sets.
pub struct LandscapeAnalyzer {
	pub max_dimension: usize,
	/// Number of sample points per direction.
	pub resolution: usize,
	/// Range around center to sample (symmetric).
	pub extent: f64,
}

impl Default for LandscapeAnalyzer {
	fn default() -> Self {
		Self {
			max_dimension: 1,
			resolution: 20,
			extent: 1.0,
		}
	}
}

impl LandscapeAnalyzer {
	pub fn new(max_dimension: usize, resolution: usize, extent: f64) -> Self {
		Self {
			max_dimension,
			resolution,
			extent,
		}
	}

	/// Analyze a 2D slice of the loss landscape.
	///
	/// `center`: point in parameter space.
	/// `dir1`, `dir2`: two orthogonal directions to sample along.
	/// `loss_fn`: evaluates the loss at a given parameter vector.
	///
	/// Returns persistence diagrams of the sampled landscape.
	pub fn analyze_slice(
		&self,
		center: &Array1<f64>,
		dir1: &Array1<f64>,
		dir2: &Array1<f64>,
		loss_fn: &dyn Fn(&Array1<f64>) -> f64,
	) -> Vec<PersistenceDiagram> {
		let n = self.resolution;
		let mut points = Array2::zeros((n * n, 3)); // (coord1, coord2, loss_value)

		let step = 2.0 * self.extent / (n as f64 - 1.0);
		for i in 0..n {
			for j in 0..n {
				let alpha = -self.extent + i as f64 * step;
				let beta = -self.extent + j as f64 * step;
				let param = center + &(dir1 * alpha) + &(dir2 * beta);
				let loss = loss_fn(&param);

				let idx = i * n + j;
				points[[idx, 0]] = alpha;
				points[[idx, 1]] = beta;
				points[[idx, 2]] = loss;
			}
		}

		// Compute persistence of the point cloud (coordinates + loss as height)
		let dm = DistanceMatrix::from_point_cloud(&points.view(), &Euclidean);
		let config = CohomologyConfig {
			max_dimension: self.max_dimension,
			..Default::default()
		};

		let result = compute_persistent_cohomology(&dm, &config);
		PersistenceDiagram::from_pairs(&result.pairs)
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_landscape_analysis() {
		let center = Array1::from_vec(vec![0.0, 0.0]);
		let dir1 = Array1::from_vec(vec![1.0, 0.0]);
		let dir2 = Array1::from_vec(vec![0.0, 1.0]);

		// Simple quadratic loss landscape
		let loss_fn = |x: &Array1<f64>| x.mapv(|v| v * v).sum();

		let analyzer = LandscapeAnalyzer::new(1, 8, 2.0);
		let diagrams = analyzer.analyze_slice(&center, &dir1, &dir2, &loss_fn);
		assert!(!diagrams.is_empty());
	}
}
