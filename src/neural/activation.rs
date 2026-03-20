use ndarray::Array2;
use rayon::prelude::*;

use crate::diagram::PersistenceDiagram;
use crate::distance::{DistanceMatrix, DistanceMetric, Euclidean};
use crate::homology::{CohomologyConfig, compute_persistent_cohomology};

/// Analyze the topology of neural network activation spaces.
///
/// Treats activation vectors (one per sample) as a point cloud
/// and computes persistent homology to reveal topological features.
pub struct ActivationAnalyzer {
	/// Maximum homological dimension to compute.
	pub max_dimension: usize,
	/// Subsample size for large layers (None = use all).
	pub subsample_size: Option<usize>,
	/// Distance metric to use.
	pub metric: Box<dyn DistanceMetric>,
	/// Filtration threshold (None = auto).
	pub threshold: Option<f64>,
	/// Prime modulus for coefficient field (default: 2 for Z/2Z).
	pub modulus: u16,
}

impl Default for ActivationAnalyzer {
	fn default() -> Self {
		Self {
			max_dimension: 1,
			subsample_size: None,
			metric: Box::new(Euclidean),
			threshold: None,
			modulus: 2,
		}
	}
}

impl ActivationAnalyzer {
	pub fn new(max_dimension: usize) -> Self {
		Self {
			max_dimension,
			..Default::default()
		}
	}

	pub fn with_subsample(mut self, size: usize) -> Self {
		self.subsample_size = Some(size);
		self
	}

	pub fn with_threshold(mut self, threshold: f64) -> Self {
		self.threshold = Some(threshold);
		self
	}

	pub fn with_modulus(mut self, modulus: u16) -> Self {
		self.modulus = modulus;
		self
	}

	/// Analyze a single layer's activations.
	///
	/// `activations`: (num_samples x num_neurons) matrix.
	/// Returns persistence diagrams for each dimension up to max_dimension.
	pub fn analyze_layer(&self, activations: &Array2<f64>) -> Vec<PersistenceDiagram> {
		let data = self.maybe_subsample(activations);
		let dm = DistanceMatrix::from_point_cloud(&data.view(), self.metric.as_ref());

		let config = CohomologyConfig {
			max_dimension: self.max_dimension,
			threshold: self.threshold,
			modulus: self.modulus,
			cocycles: false,
		};

		let result = compute_persistent_cohomology(&dm, &config);
		PersistenceDiagram::from_pairs(&result.pairs)
	}

	/// Analyze multiple layers in parallel.
	pub fn analyze_layers(&self, layers: &[Array2<f64>]) -> Vec<Vec<PersistenceDiagram>> {
		layers
			.par_iter()
			.map(|layer| self.analyze_layer(layer))
			.collect()
	}

	fn maybe_subsample<'a>(&self, data: &'a Array2<f64>) -> std::borrow::Cow<'a, Array2<f64>> {
		match self.subsample_size {
			Some(k) if k < data.nrows() => {
				// Deterministic subsample: take evenly spaced points
				let step = data.nrows() as f64 / k as f64;
				let indices: Vec<usize> = (0..k).map(|i| (i as f64 * step) as usize).collect();
				let subsampled = ndarray::Array2::from_shape_fn((k, data.ncols()), |(i, j)| {
					data[[indices[i], j]]
				});
				std::borrow::Cow::Owned(subsampled)
			}
			_ => std::borrow::Cow::Borrowed(data),
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_analyze_layer() {
		// Small activation matrix
		let activations = Array2::from_shape_fn((10, 3), |(i, j)| {
			let theta = 2.0 * std::f64::consts::PI * (i as f64) / 10.0;
			if j == 0 {
				theta.cos()
			} else if j == 1 {
				theta.sin()
			} else {
				0.0
			}
		});

		let analyzer = ActivationAnalyzer::new(1);
		let diagrams = analyzer.analyze_layer(&activations);
		assert!(!diagrams.is_empty());
	}
}
