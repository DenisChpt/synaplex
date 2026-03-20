use super::PersistenceDiagram;

/// Compute the p-Wasserstein distance between two persistence diagrams.
///
/// W_p(D1, D2) = (inf_M sum_{(x,y) in M} ||x - y||_inf^p)^(1/p)
///
/// Uses the Hungarian algorithm (Kuhn-Munkres) for exact optimal matching.
/// Points can be matched to their diagonal projection.
pub fn wasserstein_distance(d1: &PersistenceDiagram, d2: &PersistenceDiagram, p: f64) -> f64 {
	let pts1 = d1.finite_points();
	let pts2 = d2.finite_points();

	if pts1.is_empty() && pts2.is_empty() {
		return 0.0;
	}

	// Build a cost matrix for the augmented matching problem.
	// Each point from D1 can match to a point from D2 or to the diagonal.
	// Each point from D2 can also go to the diagonal.
	//
	// We create an (n1+n2) x (n1+n2) cost matrix:
	// - Top-left n1 x n2: cost of matching pts1[i] to pts2[j]
	// - Top-right n1 x n1: cost of matching pts1[i] to diagonal (diagonal entries only)
	// - Bottom-left n2 x n2: cost of matching pts2[j] to diagonal (diagonal entries only)
	// - Bottom-right n2 x n1: zero (diagonal-to-diagonal)
	let n1 = pts1.len();
	let n2 = pts2.len();
	let n = n1 + n2;

	if n == 0 {
		return 0.0;
	}

	let mut cost = vec![vec![0.0f64; n]; n];

	// Top-left: real-to-real matching
	for i in 0..n1 {
		for j in 0..n2 {
			cost[i][j] = l_inf_pow(pts1[i], pts2[j], p);
		}
	}

	// Top-right: pts1[i] to diagonal
	for i in 0..n1 {
		let diag_cost = diagonal_cost(pts1[i], p);
		for j in 0..n1 {
			cost[i][n2 + j] = if i == j { diag_cost } else { f64::MAX / 2.0 };
		}
	}

	// Bottom-left: pts2[j] to diagonal
	for j in 0..n2 {
		let diag_cost = diagonal_cost(pts2[j], p);
		for i in 0..n2 {
			cost[n1 + i][j] = if i == j { diag_cost } else { f64::MAX / 2.0 };
		}
	}

	// Bottom-right: diagonal-to-diagonal (zero cost)
	// Already initialized to 0.0

	let assignment = hungarian(&cost);
	let total: f64 = assignment
		.iter()
		.enumerate()
		.map(|(i, &j)| cost[i][j])
		.sum();

	total.powf(1.0 / p)
}

/// L-infinity distance raised to power p.
#[inline]
fn l_inf_pow(a: (f64, f64), b: (f64, f64), p: f64) -> f64 {
	let d = (a.0 - b.0).abs().max((a.1 - b.1).abs());
	d.powf(p)
}

/// Cost of matching a point to its diagonal projection, raised to power p.
#[inline]
fn diagonal_cost(pt: (f64, f64), p: f64) -> f64 {
	let d = (pt.1 - pt.0) / 2.0;
	d.powf(p)
}

/// Hungarian algorithm (Kuhn-Munkres) for minimum-cost perfect matching.
/// Input: n x n cost matrix. Output: assignment[i] = j for row i.
fn hungarian(cost: &[Vec<f64>]) -> Vec<usize> {
	let n = cost.len();
	if n == 0 {
		return vec![];
	}

	// u[i] = potential for row i, v[j] = potential for column j
	let mut u = vec![0.0f64; n + 1];
	let mut v = vec![0.0f64; n + 1];
	// p[j] = row assigned to column j (1-indexed, 0 = unassigned)
	let mut p = vec![0usize; n + 1];
	let mut way = vec![0usize; n + 1];

	for i in 1..=n {
		p[0] = i;
		let mut j0 = 0usize;
		let mut minv = vec![f64::MAX; n + 1];
		let mut used = vec![false; n + 1];

		loop {
			used[j0] = true;
			let i0 = p[j0];
			let mut delta = f64::MAX;
			let mut j1 = 0usize;

			for j in 1..=n {
				if used[j] {
					continue;
				}
				let cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
				if cur < minv[j] {
					minv[j] = cur;
					way[j] = j0;
				}
				if minv[j] < delta {
					delta = minv[j];
					j1 = j;
				}
			}

			for j in 0..=n {
				if used[j] {
					u[p[j]] += delta;
					v[j] -= delta;
				} else {
					minv[j] -= delta;
				}
			}

			j0 = j1;
			if p[j0] == 0 {
				break;
			}
		}

		loop {
			let j1 = way[j0];
			p[j0] = p[j1];
			j0 = j1;
			if j0 == 0 {
				break;
			}
		}
	}

	// Convert to 0-indexed assignment
	let mut assignment = vec![0usize; n];
	for j in 1..=n {
		if p[j] > 0 {
			assignment[p[j] - 1] = j - 1;
		}
	}
	assignment
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_identical_diagrams() {
		let d = PersistenceDiagram {
			points: vec![(0.0, 1.0), (0.5, 2.0)],
			dimension: 1,
		};
		let w = wasserstein_distance(&d, &d, 2.0);
		assert!(w < 1e-10, "Same diagram should have W=0, got {w}");
	}

	#[test]
	fn test_empty_diagrams() {
		let d1 = PersistenceDiagram {
			points: vec![],
			dimension: 0,
		};
		let d2 = PersistenceDiagram {
			points: vec![],
			dimension: 0,
		};
		assert_eq!(wasserstein_distance(&d1, &d2, 2.0), 0.0);
	}

	#[test]
	fn test_one_empty_w1() {
		let d1 = PersistenceDiagram {
			points: vec![(0.0, 2.0)],
			dimension: 1,
		};
		let d2 = PersistenceDiagram {
			points: vec![],
			dimension: 1,
		};
		// W1: point goes to diagonal, cost = (2-0)/2 = 1.0
		let w = wasserstein_distance(&d1, &d2, 1.0);
		assert!((w - 1.0).abs() < 1e-10, "Expected 1.0, got {w}");
	}

	#[test]
	fn test_triangle_inequality() {
		let d1 = PersistenceDiagram {
			points: vec![(0.0, 1.0)],
			dimension: 1,
		};
		let d2 = PersistenceDiagram {
			points: vec![(0.5, 1.5)],
			dimension: 1,
		};
		let d3 = PersistenceDiagram {
			points: vec![(1.0, 2.0)],
			dimension: 1,
		};
		let w12 = wasserstein_distance(&d1, &d2, 1.0);
		let w23 = wasserstein_distance(&d2, &d3, 1.0);
		let w13 = wasserstein_distance(&d1, &d3, 1.0);
		assert!(w13 <= w12 + w23 + 1e-10, "Triangle inequality violated");
	}
}
