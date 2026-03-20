/// A single element of the cover: an interval [lower, upper] with
/// the indices of points whose filter value falls in this interval.
#[derive(Clone, Debug)]
pub struct CoverElement {
	pub lower: f64,
	pub upper: f64,
	pub point_indices: Vec<usize>,
}

/// Trait for cover strategies on the filter range.
pub trait CoverStrategy: Send + Sync {
	fn cover(&self, filter_values: &[f64]) -> Vec<CoverElement>;
}

/// Uniform cover: divide the filter range into equally-spaced intervals
/// with a fixed overlap percentage.
pub struct UniformCover {
	pub num_intervals: usize,
	/// Overlap fraction in [0, 1). E.g., 0.3 = 30% overlap.
	pub overlap: f64,
}

impl UniformCover {
	pub fn new(num_intervals: usize, overlap: f64) -> Self {
		Self {
			num_intervals: num_intervals.max(1),
			overlap: overlap.clamp(0.0, 0.99),
		}
	}
}

impl CoverStrategy for UniformCover {
	fn cover(&self, filter_values: &[f64]) -> Vec<CoverElement> {
		if filter_values.is_empty() {
			return vec![];
		}

		let min_val = filter_values.iter().cloned().fold(f64::INFINITY, f64::min);
		let max_val = filter_values
			.iter()
			.cloned()
			.fold(f64::NEG_INFINITY, f64::max);
		let range = max_val - min_val;

		if range < f64::EPSILON {
			return vec![CoverElement {
				lower: min_val - 1.0,
				upper: max_val + 1.0,
				point_indices: (0..filter_values.len()).collect(),
			}];
		}

		let step = range / self.num_intervals as f64;
		let overlap_size = step * self.overlap;

		let mut elements = Vec::with_capacity(self.num_intervals);
		for i in 0..self.num_intervals {
			let lower = min_val + (i as f64) * step - overlap_size;
			let upper = min_val + ((i + 1) as f64) * step + overlap_size;

			let point_indices: Vec<usize> = filter_values
				.iter()
				.enumerate()
				.filter(|(_, v)| **v >= lower && **v <= upper)
				.map(|(idx, _)| idx)
				.collect();

			elements.push(CoverElement {
				lower,
				upper,
				point_indices,
			});
		}

		elements
	}
}

/// Balanced cover: each interval contains approximately the same
/// number of points (quantile-based).
pub struct BalancedCover {
	pub num_intervals: usize,
	pub overlap: f64,
}

impl BalancedCover {
	pub fn new(num_intervals: usize, overlap: f64) -> Self {
		Self {
			num_intervals: num_intervals.max(1),
			overlap: overlap.clamp(0.0, 0.99),
		}
	}
}

impl CoverStrategy for BalancedCover {
	fn cover(&self, filter_values: &[f64]) -> Vec<CoverElement> {
		if filter_values.is_empty() {
			return vec![];
		}

		// Sort indices by filter value
		let mut sorted_indices: Vec<usize> = (0..filter_values.len()).collect();
		sorted_indices.sort_by(|&a, &b| filter_values[a].partial_cmp(&filter_values[b]).unwrap());

		let n = sorted_indices.len();
		let chunk_size = n.div_ceil(self.num_intervals);
		let overlap_count = (chunk_size as f64 * self.overlap) as usize;

		let mut elements = Vec::with_capacity(self.num_intervals);
		for i in 0..self.num_intervals {
			let start = if i == 0 {
				0
			} else {
				(i * chunk_size).saturating_sub(overlap_count)
			};
			let end = ((i + 1) * chunk_size + overlap_count).min(n);

			if start >= n {
				break;
			}

			let point_indices: Vec<usize> = sorted_indices[start..end].to_vec();
			let lower = filter_values[point_indices[0]];
			let upper = filter_values[*point_indices.last().unwrap()];

			elements.push(CoverElement {
				lower,
				upper,
				point_indices,
			});
		}

		elements
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_uniform_cover() {
		let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
		let cover = UniformCover::new(10, 0.1);
		let elements = cover.cover(&values);
		assert_eq!(elements.len(), 10);
		// All points should be covered
		let all_covered: std::collections::HashSet<usize> = elements
			.iter()
			.flat_map(|e| e.point_indices.iter().copied())
			.collect();
		for i in 0..100 {
			assert!(all_covered.contains(&i), "Point {i} not covered");
		}
	}

	#[test]
	fn test_balanced_cover() {
		let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
		let cover = BalancedCover::new(5, 0.2);
		let elements = cover.cover(&values);
		assert_eq!(elements.len(), 5);
	}
}
