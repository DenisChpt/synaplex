use crate::utils::combinatorics::BinomialTable;

/// A simplex identified by sorted vertex indices.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Simplex {
	pub vertices: Vec<u32>,
}

impl Simplex {
	/// Create a new simplex from vertices. Sorts them.
	pub fn new(mut vertices: Vec<u32>) -> Self {
		vertices.sort_unstable();
		Self { vertices }
	}

	#[inline]
	pub fn dimension(&self) -> usize {
		self.vertices.len() - 1
	}

	pub fn to_index(&self, binomial: &BinomialTable) -> u64 {
		crate::utils::combinatorics::simplex_to_index(&self.vertices, binomial)
	}

	/// Get boundary simplices (all codimension-1 faces).
	pub fn boundary(&self) -> Vec<Simplex> {
		(0..self.vertices.len())
			.map(|skip| {
				let verts: Vec<u32> = self
					.vertices
					.iter()
					.enumerate()
					.filter(|&(i, _)| i != skip)
					.map(|(_, &v)| v)
					.collect();
				Simplex { vertices: verts }
			})
			.collect()
	}
}

/// A simplex with an associated filtration value (the "time" it enters).
#[derive(Clone, Debug)]
pub struct FilteredSimplex {
	pub simplex: Simplex,
	pub filtration: f64,
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_simplex_dimension() {
		assert_eq!(Simplex::new(vec![0]).dimension(), 0);
		assert_eq!(Simplex::new(vec![0, 1]).dimension(), 1);
		assert_eq!(Simplex::new(vec![0, 1, 2]).dimension(), 2);
	}

	#[test]
	fn test_boundary() {
		let tri = Simplex::new(vec![0, 1, 2]);
		let bdry = tri.boundary();
		assert_eq!(bdry.len(), 3);
		assert_eq!(bdry[0].vertices, vec![1, 2]);
		assert_eq!(bdry[1].vertices, vec![0, 2]);
		assert_eq!(bdry[2].vertices, vec![0, 1]);
	}
}
