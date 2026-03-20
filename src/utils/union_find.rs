/// Union-Find (Disjoint Set) data structure with union by rank
/// and path compression. Used for H0 (connected components) computation.
pub struct UnionFind {
	parent: Vec<usize>,
	rank: Vec<u8>,
	num_components: usize,
}

impl UnionFind {
	pub fn new(n: usize) -> Self {
		Self {
			parent: (0..n).collect(),
			rank: vec![0; n],
			num_components: n,
		}
	}

	/// Find the representative of the set containing `x`, with path compression.
	pub fn find(&mut self, x: usize) -> usize {
		if self.parent[x] != x {
			self.parent[x] = self.find(self.parent[x]);
		}
		self.parent[x]
	}

	/// Union the sets containing `x` and `y`.
	/// Returns `true` if they were in different sets (i.e., a merge happened).
	pub fn union(&mut self, x: usize, y: usize) -> bool {
		let rx = self.find(x);
		let ry = self.find(y);
		if rx == ry {
			return false;
		}
		match self.rank[rx].cmp(&self.rank[ry]) {
			std::cmp::Ordering::Less => self.parent[rx] = ry,
			std::cmp::Ordering::Greater => self.parent[ry] = rx,
			std::cmp::Ordering::Equal => {
				self.parent[ry] = rx;
				self.rank[rx] += 1;
			}
		}
		self.num_components -= 1;
		true
	}

	pub fn num_components(&self) -> usize {
		self.num_components
	}

	pub fn connected(&mut self, x: usize, y: usize) -> bool {
		self.find(x) == self.find(y)
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_union_find_basic() {
		let mut uf = UnionFind::new(5);
		assert_eq!(uf.num_components(), 5);

		assert!(uf.union(0, 1));
		assert_eq!(uf.num_components(), 4);
		assert!(uf.connected(0, 1));
		assert!(!uf.connected(0, 2));

		assert!(uf.union(2, 3));
		assert!(uf.union(0, 3));
		assert_eq!(uf.num_components(), 2);
		assert!(uf.connected(0, 2));
		assert!(uf.connected(1, 3));

		assert!(!uf.union(0, 1)); // already connected
	}
}
