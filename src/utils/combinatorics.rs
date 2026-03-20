/// Precomputed binomial coefficient table for O(1) lookup.
/// `table[n][k]` = C(n, k). Used throughout the library for
/// combinatorial number system encoding of simplices.
pub struct BinomialTable {
	table: Vec<Vec<u64>>,
}

impl BinomialTable {
	/// Build a table for n in [0, max_n] and k in [0, max_k].
	pub fn new(max_n: usize, max_k: usize) -> Self {
		let mut table = vec![vec![0u64; max_k + 1]; max_n + 1];
		for n in 0..=max_n {
			table[n][0] = 1;
			for k in 1..=max_k.min(n) {
				table[n][k] = table[n - 1][k - 1].saturating_add(table[n - 1][k]);
			}
		}
		Self { table }
	}

	/// C(n, k) with bounds check.
	#[inline]
	pub fn get(&self, n: usize, k: usize) -> u64 {
		if k > n || n >= self.table.len() || k >= self.table[0].len() {
			return 0;
		}
		self.table[n][k]
	}
}

/// Encode a k-simplex `{v0 < v1 < ... < vk}` as a single u64 index
/// using the combinatorial number system.
/// index = C(v_k, k+1) + C(v_{k-1}, k) + ... + C(v_0, 1)
#[inline]
pub fn simplex_to_index(vertices: &[u32], binomial: &BinomialTable) -> u64 {
	let dim = vertices.len();
	let mut index = 0u64;
	for (i, &v) in vertices.iter().enumerate() {
		index += binomial.get(v as usize, i + 1);
	}
	debug_assert!(dim > 0);
	index
}

/// Decode a combinatorial index back to sorted vertex indices.
pub fn index_to_simplex(
	mut index: u64,
	dim: usize,
	n: usize,
	binomial: &BinomialTable,
) -> Vec<u32> {
	let mut vertices = vec![0u32; dim + 1];
	let mut v = n as u64;
	for k in (0..=dim).rev() {
		// Find largest v such that C(v, k+1) <= index
		while v > 0 && binomial.get(v as usize, k + 1) > index {
			v -= 1;
		}
		vertices[k] = v as u32;
		index -= binomial.get(v as usize, k + 1);
		// As v is unsigned, we can use saturating_sub to avoid underflow when v is 0.
		v = v.saturating_sub(1);
	}
	vertices
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_binomial_table() {
		let bt = BinomialTable::new(10, 5);
		assert_eq!(bt.get(5, 2), 10);
		assert_eq!(bt.get(10, 3), 120);
		assert_eq!(bt.get(0, 0), 1);
		assert_eq!(bt.get(5, 0), 1);
		assert_eq!(bt.get(3, 5), 0); // k > n
	}

	#[test]
	fn test_simplex_round_trip() {
		let bt = BinomialTable::new(100, 5);

		let vertices = vec![2, 5, 7];
		let idx = simplex_to_index(&vertices, &bt);
		let decoded = index_to_simplex(idx, 2, 100, &bt);
		assert_eq!(decoded, vertices);

		let vertices = vec![0, 1, 2, 3];
		let idx = simplex_to_index(&vertices, &bt);
		let decoded = index_to_simplex(idx, 3, 100, &bt);
		assert_eq!(decoded, vertices);
	}

	#[test]
	fn test_edge_encoding() {
		let bt = BinomialTable::new(100, 5);
		// Edge {0,1} should map to index C(1,2) + C(0,1) = 0 + 0 = 0
		let idx = simplex_to_index(&[0, 1], &bt);
		assert_eq!(idx, 0);
		// Edge {0,2} -> C(2,2) + C(0,1) = 1 + 0 = 1
		let idx = simplex_to_index(&[0, 2], &bt);
		assert_eq!(idx, 1);
		// Edge {1,2} -> C(2,2) + C(1,1) = 1 + 1 = 2
		let idx = simplex_to_index(&[1, 2], &bt);
		assert_eq!(idx, 2);
	}
}
