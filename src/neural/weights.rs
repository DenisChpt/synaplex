use ndarray::Array2;

use crate::diagram::PersistenceDiagram;
use crate::distance::{DistanceMatrix, DistanceMetric, Euclidean};
use crate::homology::{CohomologyConfig, compute_persistent_cohomology};

/// Analyze the topology of weight matrices.
///
/// Treats rows (or columns) of a weight matrix as a point cloud
/// to reveal structural patterns in learned representations.
pub struct WeightAnalyzer {
	pub max_dimension: usize,
	pub metric: Box<dyn DistanceMetric>,
	/// Analyze rows (true) or columns (false).
	pub row_wise: bool,
	/// Prime modulus for coefficient field (default: 2 for Z/2Z).
	pub modulus: u16,
}

impl Default for WeightAnalyzer {
	fn default() -> Self {
		Self {
			max_dimension: 1,
			metric: Box::new(Euclidean),
			row_wise: true,
			modulus: 2,
		}
	}
}

impl WeightAnalyzer {
	pub fn new(max_dimension: usize) -> Self {
		Self {
			max_dimension,
			..Default::default()
		}
	}

	/// Analyze a weight matrix's topology.
	///
	/// `weights`: (num_output x num_input) weight matrix.
	pub fn analyze(&self, weights: &Array2<f64>) -> Vec<PersistenceDiagram> {
		let data = if self.row_wise {
			weights.clone()
		} else {
			weights.t().to_owned()
		};

		let dm = DistanceMatrix::from_point_cloud(&data.view(), self.metric.as_ref());
		let config = CohomologyConfig {
			max_dimension: self.max_dimension,
			threshold: None,
			modulus: self.modulus,
			cocycles: false,
		};

		let result = compute_persistent_cohomology(&dm, &config);
		PersistenceDiagram::from_pairs(&result.pairs)
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_weight_analysis() {
		let weights = Array2::from_shape_fn((8, 4), |(i, j)| ((i * 4 + j) as f64).sin());
		let analyzer = WeightAnalyzer::new(1);
		let diagrams = analyzer.analyze(&weights);
		assert!(!diagrams.is_empty());
	}
}
