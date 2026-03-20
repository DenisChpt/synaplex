/// Trait for distance metrics between points in R^n.
pub trait DistanceMetric: Send + Sync {
	fn distance(&self, a: &[f64], b: &[f64]) -> f64;
}

/// Standard Euclidean (L2) distance.
pub struct Euclidean;

impl DistanceMetric for Euclidean {
	#[inline]
	fn distance(&self, a: &[f64], b: &[f64]) -> f64 {
		debug_assert_eq!(a.len(), b.len());
		let mut sum = 0.0f64;
		for i in 0..a.len() {
			let d = a[i] - b[i];
			sum += d * d;
		}
		sum.sqrt()
	}
}

/// Cosine distance: 1 - cos(a, b).
pub struct Cosine;

impl DistanceMetric for Cosine {
	#[inline]
	fn distance(&self, a: &[f64], b: &[f64]) -> f64 {
		debug_assert_eq!(a.len(), b.len());
		let mut dot = 0.0f64;
		let mut norm_a = 0.0f64;
		let mut norm_b = 0.0f64;
		for i in 0..a.len() {
			dot += a[i] * b[i];
			norm_a += a[i] * a[i];
			norm_b += b[i] * b[i];
		}
		let denom = (norm_a * norm_b).sqrt();
		if denom < f64::EPSILON {
			return 1.0;
		}
		1.0 - (dot / denom)
	}
}

/// Manhattan (L1) distance.
pub struct Manhattan;

impl DistanceMetric for Manhattan {
	#[inline]
	fn distance(&self, a: &[f64], b: &[f64]) -> f64 {
		debug_assert_eq!(a.len(), b.len());
		let mut sum = 0.0f64;
		for i in 0..a.len() {
			sum += (a[i] - b[i]).abs();
		}
		sum
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_euclidean() {
		let m = Euclidean;
		let a = [0.0, 0.0];
		let b = [3.0, 4.0];
		assert!((m.distance(&a, &b) - 5.0).abs() < 1e-10);
	}

	#[test]
	fn test_cosine() {
		let m = Cosine;
		let a = [1.0, 0.0];
		let b = [0.0, 1.0];
		assert!((m.distance(&a, &b) - 1.0).abs() < 1e-10); // orthogonal

		let c = [1.0, 0.0];
		assert!(m.distance(&a, &c).abs() < 1e-10); // same direction
	}

	#[test]
	fn test_manhattan() {
		let m = Manhattan;
		let a = [0.0, 0.0];
		let b = [3.0, 4.0];
		assert!((m.distance(&a, &b) - 7.0).abs() < 1e-10);
	}
}
