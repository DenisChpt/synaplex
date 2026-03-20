/// Build the nerve of a collection of clusters.
/// Two clusters are connected if they share at least one point.
/// Returns a list of edges (cluster_i, cluster_j, shared_count).
pub fn build_nerve(clusters: &[Vec<usize>]) -> Vec<(usize, usize, usize)> {
	let mut edges = Vec::new();

	// Build a point-to-cluster index for efficiency
	let mut point_to_clusters: std::collections::HashMap<usize, Vec<usize>> =
		std::collections::HashMap::new();
	for (ci, cluster) in clusters.iter().enumerate() {
		for &pt in cluster {
			point_to_clusters.entry(pt).or_default().push(ci);
		}
	}

	// For each point that belongs to multiple clusters, create edges
	let mut edge_counts: std::collections::HashMap<(usize, usize), usize> =
		std::collections::HashMap::new();
	for cluster_ids in point_to_clusters.values() {
		if cluster_ids.len() < 2 {
			continue;
		}
		for i in 0..cluster_ids.len() {
			for j in (i + 1)..cluster_ids.len() {
				let (a, b) = if cluster_ids[i] < cluster_ids[j] {
					(cluster_ids[i], cluster_ids[j])
				} else {
					(cluster_ids[j], cluster_ids[i])
				};
				*edge_counts.entry((a, b)).or_insert(0) += 1;
			}
		}
	}

	for ((a, b), count) in edge_counts {
		edges.push((a, b, count));
	}

	edges
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_nerve_construction() {
		let clusters = vec![
			vec![0, 1, 2], // cluster 0
			vec![2, 3, 4], // cluster 1 (shares point 2 with cluster 0)
			vec![5, 6],    // cluster 2 (isolated)
		];
		let edges = build_nerve(&clusters);
		assert_eq!(edges.len(), 1);
		assert_eq!(edges[0], (0, 1, 1)); // shared 1 point
	}

	#[test]
	fn test_nerve_multiple_shared() {
		let clusters = vec![vec![0, 1, 2, 3], vec![2, 3, 4, 5]];
		let edges = build_nerve(&clusters);
		assert_eq!(edges.len(), 1);
		assert_eq!(edges[0].2, 2); // 2 shared points
	}
}
