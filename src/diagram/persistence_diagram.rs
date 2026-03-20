use crate::homology::PersistencePair;

/// A persistence diagram: a collection of (birth, death) points
/// for a single homological dimension.
#[derive(Clone, Debug)]
pub struct PersistenceDiagram {
	/// (birth, death) pairs. death may be f64::INFINITY for essential classes.
	pub points: Vec<(f64, f64)>,
	/// Homological dimension.
	pub dimension: usize,
}

impl PersistenceDiagram {
	/// Build diagrams (one per dimension) from persistence pairs.
	pub fn from_pairs(pairs: &[PersistencePair]) -> Vec<Self> {
		let max_dim = pairs.iter().map(|p| p.dimension).max().unwrap_or(0);
		let mut diagrams: Vec<Self> = (0..=max_dim)
			.map(|d| PersistenceDiagram {
				points: Vec::new(),
				dimension: d,
			})
			.collect();

		for pair in pairs {
			diagrams[pair.dimension]
				.points
				.push((pair.birth, pair.death));
		}

		diagrams
	}

	pub fn len(&self) -> usize {
		self.points.len()
	}

	pub fn is_empty(&self) -> bool {
		self.points.is_empty()
	}

	/// Only finite (non-essential) points.
	pub fn finite_points(&self) -> Vec<(f64, f64)> {
		self.points
			.iter()
			.filter(|(_, d)| d.is_finite())
			.copied()
			.collect()
	}

	/// Essential (infinite death) points.
	pub fn essential_points(&self) -> Vec<(f64, f64)> {
		self.points
			.iter()
			.filter(|(_, d)| d.is_infinite())
			.copied()
			.collect()
	}

	/// Maximum persistence among finite points.
	pub fn max_persistence(&self) -> f64 {
		self.finite_points()
			.iter()
			.map(|(b, d)| d - b)
			.fold(0.0f64, f64::max)
	}

	/// Filter out points with persistence below a threshold.
	pub fn threshold(&self, min_persistence: f64) -> Self {
		Self {
			points: self
				.points
				.iter()
				.filter(|(b, d)| (d - b) >= min_persistence || d.is_infinite())
				.copied()
				.collect(),
			dimension: self.dimension,
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_from_pairs() {
		let pairs = vec![
			PersistencePair {
				birth_idx: 0,
				death_idx: None,
				birth: 0.0,
				death: f64::INFINITY,
				dimension: 0,
			},
			PersistencePair {
				birth_idx: 1,
				death_idx: Some(3),
				birth: 0.0,
				death: 1.0,
				dimension: 0,
			},
			PersistencePair {
				birth_idx: 4,
				death_idx: Some(6),
				birth: 1.0,
				death: 2.0,
				dimension: 1,
			},
		];

		let diagrams = PersistenceDiagram::from_pairs(&pairs);
		assert_eq!(diagrams.len(), 2);
		assert_eq!(diagrams[0].len(), 2); // H0: 2 points
		assert_eq!(diagrams[1].len(), 1); // H1: 1 point
	}

	#[test]
	fn test_threshold() {
		let diag = PersistenceDiagram {
			points: vec![(0.0, 0.1), (0.0, 1.0), (0.5, f64::INFINITY)],
			dimension: 0,
		};
		let filtered = diag.threshold(0.5);
		assert_eq!(filtered.len(), 2); // (0,1) and essential
	}
}
