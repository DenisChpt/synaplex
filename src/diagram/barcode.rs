use super::PersistenceDiagram;

/// Barcode representation: a list of intervals [birth, death) per dimension.
/// Equivalent to a persistence diagram but in interval form.
#[derive(Clone, Debug)]
pub struct Barcode {
	/// Intervals grouped by dimension. bars[d] = vec of (birth, death).
	pub bars: Vec<Vec<(f64, f64)>>,
}

impl Barcode {
	/// Build from persistence diagrams (one per dimension).
	pub fn from_diagrams(diagrams: &[PersistenceDiagram]) -> Self {
		let bars = diagrams.iter().map(|d| d.points.clone()).collect();
		Self { bars }
	}

	pub fn count(&self, dim: usize) -> usize {
		self.bars.get(dim).map_or(0, |b| b.len())
	}

	pub fn total_bars(&self) -> usize {
		self.bars.iter().map(|b| b.len()).sum()
	}

	/// Bars for a given dimension, sorted by birth time.
	pub fn sorted_bars(&self, dim: usize) -> Vec<(f64, f64)> {
		let mut bars = self.bars.get(dim).cloned().unwrap_or_default();
		bars.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
		bars
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_barcode_from_diagrams() {
		let diagrams = vec![
			PersistenceDiagram {
				points: vec![(0.0, f64::INFINITY), (0.0, 1.0)],
				dimension: 0,
			},
			PersistenceDiagram {
				points: vec![(1.5, 3.0)],
				dimension: 1,
			},
		];
		let bc = Barcode::from_diagrams(&diagrams);
		assert_eq!(bc.count(0), 2);
		assert_eq!(bc.count(1), 1);
		assert_eq!(bc.total_bars(), 3);
	}
}
