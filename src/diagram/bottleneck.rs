use super::PersistenceDiagram;

/// Compute the bottleneck distance between two persistence diagrams.
///
/// The bottleneck distance is the infimum over all matchings M of
/// sup_{(p,q) in M} ||p - q||_inf, where points can be matched
/// to the diagonal (at cost (death - birth) / 2).
///
/// Algorithm: binary search over candidate distances + bipartite
/// matching check via augmenting paths.
pub fn bottleneck_distance(d1: &PersistenceDiagram, d2: &PersistenceDiagram) -> f64 {
	let pts1 = d1.finite_points();
	let pts2 = d2.finite_points();

	if pts1.is_empty() && pts2.is_empty() {
		return 0.0;
	}

	// Collect all candidate distance values
	let mut candidates: Vec<f64> = Vec::new();

	// Distances between all pairs of points
	for &(b1, d1) in &pts1 {
		for &(b2, d2) in &pts2 {
			candidates.push(l_inf(b1, d1, b2, d2));
		}
	}

	// Distances from each point to the diagonal
	for &(b, d) in &pts1 {
		candidates.push((d - b) / 2.0);
	}
	for &(b, d) in &pts2 {
		candidates.push((d - b) / 2.0);
	}

	candidates.push(0.0);
	candidates.sort_by(|a, b| a.partial_cmp(b).unwrap());
	candidates.dedup_by(|a, b| (*a - *b).abs() < 1e-12);

	// Binary search: find smallest delta such that a perfect matching exists
	let mut lo = 0;
	let mut hi = candidates.len() - 1;

	while lo < hi {
		let mid = (lo + hi) / 2;
		if can_match(&pts1, &pts2, candidates[mid]) {
			hi = mid;
		} else {
			lo = mid + 1;
		}
	}

	candidates[lo]
}

/// L-infinity distance between two diagram points.
#[inline]
fn l_inf(b1: f64, d1: f64, b2: f64, d2: f64) -> f64 {
	(b1 - b2).abs().max((d1 - d2).abs())
}

/// Check if a perfect matching exists at distance delta.
/// Uses a greedy augmenting path algorithm on the bipartite graph.
fn can_match(pts1: &[(f64, f64)], pts2: &[(f64, f64)], delta: f64) -> bool {
	let n1 = pts1.len();
	let n2 = pts2.len();
	// Total nodes: n1 "real" from diagram 1, n2 "real" from diagram 2,
	// plus diagonal projections for each point.
	// We model this as: each point in pts1 can match to a point in pts2
	// or to the diagonal; same for pts2.

	// Build adjacency: pts1[i] can match to pts2[j] if l_inf <= delta
	// pts1[i] can match to diagonal if (d-b)/2 <= delta
	// pts2[j] can match to diagonal (always available as a fallback)

	// Simple greedy + augmenting path approach
	let mut match_for_2: Vec<Option<usize>> = vec![None; n2];

	for i in 0..n1 {
		let diag_ok = (pts1[i].1 - pts1[i].0) / 2.0 <= delta + 1e-12;

		let mut visited = vec![false; n2];
		if !augment(i, pts1, pts2, delta, &mut match_for_2, &mut visited) && !diag_ok {
			return false; // Can't match this point at all
		}
	}

	// Check that all unmatched pts2 can go to diagonal
	for j in 0..n2 {
		if match_for_2[j].is_none() {
			let diag_ok = (pts2[j].1 - pts2[j].0) / 2.0 <= delta + 1e-12;
			if !diag_ok {
				return false;
			}
		}
	}

	true
}

/// Try to find an augmenting path for pts1[i].
fn augment(
	i: usize,
	pts1: &[(f64, f64)],
	pts2: &[(f64, f64)],
	delta: f64,
	match_for_2: &mut [Option<usize>],
	visited: &mut [bool],
) -> bool {
	for j in 0..pts2.len() {
		if visited[j] {
			continue;
		}
		let dist = l_inf(pts1[i].0, pts1[i].1, pts2[j].0, pts2[j].1);
		if dist > delta + 1e-12 {
			continue;
		}
		visited[j] = true;

		// If j is unmatched, or we can reroute its current match
		let can_reroute = match match_for_2[j] {
			None => true,
			Some(prev_i) => {
				// Can prev_i go to diagonal instead?
				let diag_ok = (pts1[prev_i].1 - pts1[prev_i].0) / 2.0 <= delta + 1e-12;
				diag_ok || augment(prev_i, pts1, pts2, delta, match_for_2, visited)
			}
		};

		if can_reroute {
			match_for_2[j] = Some(i);
			return true;
		}
	}
	false
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
		assert!((bottleneck_distance(&d, &d)).abs() < 1e-10);
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
		assert_eq!(bottleneck_distance(&d1, &d2), 0.0);
	}

	#[test]
	fn test_one_empty() {
		let d1 = PersistenceDiagram {
			points: vec![(0.0, 2.0)],
			dimension: 1,
		};
		let d2 = PersistenceDiagram {
			points: vec![],
			dimension: 1,
		};
		// Point must go to diagonal at cost (2-0)/2 = 1.0
		assert!((bottleneck_distance(&d1, &d2) - 1.0).abs() < 1e-10);
	}

	#[test]
	fn test_shifted_diagrams() {
		let d1 = PersistenceDiagram {
			points: vec![(0.0, 1.0)],
			dimension: 1,
		};
		let d2 = PersistenceDiagram {
			points: vec![(0.5, 1.5)],
			dimension: 1,
		};
		// L-inf distance between (0,1) and (0.5,1.5) = max(0.5, 0.5) = 0.5
		assert!((bottleneck_distance(&d1, &d2) - 0.5).abs() < 1e-10);
	}
}
