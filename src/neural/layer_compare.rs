use ndarray::Array2;
use rayon::prelude::*;

use crate::diagram::{PersistenceDiagram, bottleneck_distance, wasserstein_distance};

/// Compare topological signatures across neural network layers.
///
/// Given persistence diagrams for each layer, compute pairwise distances
/// to quantify how the topology changes through the network.
///
/// Compute a pairwise bottleneck distance matrix between layers.
///
/// `layer_diagrams[i][d]` = persistence diagram of layer i in dimension d.
/// Returns a matrix where entry (i, j) = sum of bottleneck distances
/// across all dimensions.
pub fn compare_layers(layer_diagrams: &[Vec<PersistenceDiagram>]) -> Array2<f64> {
	let n = layer_diagrams.len();
	let mut result = Array2::zeros((n, n));

	// Compute upper triangle in parallel
	let pairs: Vec<(usize, usize, f64)> = (0..n)
		.into_par_iter()
		.flat_map(|i| {
			((i + 1)..n).into_par_iter().map(move |j| {
				let dist = total_bottleneck(&layer_diagrams[i], &layer_diagrams[j]);
				(i, j, dist)
			})
		})
		.collect();

	for (i, j, d) in pairs {
		result[[i, j]] = d;
		result[[j, i]] = d;
	}

	result
}

/// Compute a pairwise Wasserstein distance matrix.
pub fn compare_layers_wasserstein(
	layer_diagrams: &[Vec<PersistenceDiagram>],
	p: f64,
) -> Array2<f64> {
	let n = layer_diagrams.len();
	let mut result = Array2::zeros((n, n));

	let pairs: Vec<(usize, usize, f64)> = (0..n)
		.into_par_iter()
		.flat_map(|i| {
			((i + 1)..n).into_par_iter().map(move |j| {
				let dist = total_wasserstein(&layer_diagrams[i], &layer_diagrams[j], p);
				(i, j, dist)
			})
		})
		.collect();

	for (i, j, d) in pairs {
		result[[i, j]] = d;
		result[[j, i]] = d;
	}

	result
}

/// Sum of bottleneck distances across matching dimensions.
fn total_bottleneck(a: &[PersistenceDiagram], b: &[PersistenceDiagram]) -> f64 {
	let max_dim = a.len().max(b.len());
	let empty = PersistenceDiagram {
		points: vec![],
		dimension: 0,
	};

	(0..max_dim)
		.map(|d| {
			let da = a.get(d).unwrap_or(&empty);
			let db = b.get(d).unwrap_or(&empty);
			bottleneck_distance(da, db)
		})
		.sum()
}

fn total_wasserstein(a: &[PersistenceDiagram], b: &[PersistenceDiagram], p: f64) -> f64 {
	let max_dim = a.len().max(b.len());
	let empty = PersistenceDiagram {
		points: vec![],
		dimension: 0,
	};

	(0..max_dim)
		.map(|d| {
			let da = a.get(d).unwrap_or(&empty);
			let db = b.get(d).unwrap_or(&empty);
			wasserstein_distance(da, db, p)
		})
		.sum()
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_compare_identical_layers() {
		let diagrams = vec![
			vec![PersistenceDiagram {
				points: vec![(0.0, 1.0), (0.5, 2.0)],
				dimension: 0,
			}],
			vec![PersistenceDiagram {
				points: vec![(0.0, 1.0), (0.5, 2.0)],
				dimension: 0,
			}],
		];

		let dist_matrix = compare_layers(&diagrams);
		assert!(dist_matrix[[0, 1]].abs() < 1e-10);
		assert!(dist_matrix[[1, 0]].abs() < 1e-10);
	}
}
