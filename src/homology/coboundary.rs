/// Zero-allocation simplex coboundary enumerator.
///
/// Generates cofaces of a simplex procedurally, without allocating memory
/// for storing them. The coboundary matrix is never materialized; cofaces
/// are computed on-the-fly during reduction.
///
/// For a d-simplex σ = {v_0, ..., v_d} with combinatorial index `idx`,
/// each cofacet is obtained by inserting a vertex w ∉ σ. The enumerator
/// walks through candidate vertices from n-1 down to 0, tracking the
/// combinatorial index incrementally (no full recomputation per cofacet).

use crate::distance::DistanceMatrix;
use crate::utils::combinatorics::BinomialTable;

use super::entry::DiameterEntry;

/// Enumerates cofacets of a simplex on the fly (zero heap allocation).
pub struct CoboundaryEnumerator<'a> {
	/// Index below the current insertion point in the combinatorial encoding.
	idx_below: u64,
	/// Index above the current insertion point.
	idx_above: u64,
	/// Current candidate vertex (decreasing from n-1).
	v: i64,
	/// Current position in the vertex list (dimension + 1 down to 0).
	k: i64,
	/// Vertices of the simplex (decoded once at construction).
	vertices: Vec<u32>,
	/// Diameter (filtration value) of the original simplex.
	simplex_diameter: f64,
	/// Coefficient of the original simplex.
	simplex_coefficient: u16,
	/// Modulus for the coefficient field.
	modulus: u16,
	/// Distance matrix for computing cofacet diameters.
	dist: &'a DistanceMatrix,
	/// Binomial coefficient table.
	binomial: &'a BinomialTable,
	/// Maximum filtration threshold.
	threshold: f64,
}

impl<'a> CoboundaryEnumerator<'a> {
	/// Create an enumerator for the cofacets of a simplex.
	///
	/// - `simplex_index`: combinatorial index of the simplex
	/// - `dim`: dimension of the simplex
	/// - `diameter`: filtration value of the simplex
	/// - `coefficient`: coefficient of the simplex in Z/pZ
	/// - `n`: number of vertices in the point cloud
	/// - `modulus`: prime modulus for coefficients
	/// - `threshold`: maximum filtration value
	/// - `dist`: distance matrix
	/// - `binomial`: precomputed binomial table
	pub fn new(
		simplex_index: u64,
		dim: usize,
		diameter: f64,
		coefficient: u16,
		n: usize,
		modulus: u16,
		threshold: f64,
		dist: &'a DistanceMatrix,
		binomial: &'a BinomialTable,
	) -> Self {
		// Decode the simplex vertices
		let vertices = crate::utils::combinatorics::index_to_simplex(simplex_index, dim, n, binomial);

		Self {
			idx_below: simplex_index,
			idx_above: 0,
			v: n as i64 - 1,
			k: dim as i64 + 1,
			vertices,
			simplex_diameter: diameter,
			simplex_coefficient: coefficient,
			modulus,
			dist,
			binomial,
			threshold,
		}
	}

	/// Check if there are more cofacets to enumerate.
	///
	/// When `all_cofacets` is false, only enumerates cofacets that insert
	/// a vertex *above* the highest vertex of the simplex (the "apparent
	/// pairs" optimization used during assembly of columns to reduce).
	#[inline]
	pub fn has_next(&self, all_cofacets: bool) -> bool {
		self.v >= self.k
			&& (all_cofacets || self.binomial.get(self.v as usize, self.k as usize) > self.idx_below)
	}

	/// Advance to and return the next cofacet.
	///
	/// The combinatorial index is updated incrementally as we "insert"
	/// vertex v into the simplex.
	pub fn next(&mut self) -> Option<DiameterEntry> {
		if !self.has_next(true) {
			return None;
		}

		// Walk v downward until C(v, k) > idx_below
		while self.binomial.get(self.v as usize, self.k as usize) <= self.idx_below {
			self.idx_below -= self.binomial.get(self.v as usize, self.k as usize);
			self.idx_above += self.binomial.get(self.v as usize, self.k as usize + 1);
			self.v -= 1;
			self.k -= 1;
			if self.k < 0 {
				return None;
			}
		}

		// Compute cofacet diameter: max of simplex diameter and distances to new vertex v
		let mut cofacet_diameter = self.simplex_diameter;
		for &w in &self.vertices {
			let d = self.dist.get(self.v as usize, w as usize);
			if d > cofacet_diameter {
				cofacet_diameter = d;
			}
		}

		// Compute cofacet combinatorial index
		let cofacet_index =
			self.idx_above + self.binomial.get(self.v as usize, self.k as usize + 1) + self.idx_below;

		// Compute cofacet coefficient: (-1)^position * simplex_coefficient
		let cofacet_coefficient = if self.k & 1 != 0 {
			// odd position: negate
			((self.modulus - 1) as u32 * self.simplex_coefficient as u32 % self.modulus as u32) as u16
		} else {
			self.simplex_coefficient
		};

		self.v -= 1;

		// Only return cofacets within threshold
		if cofacet_diameter <= self.threshold {
			Some(DiameterEntry::new(
				cofacet_diameter,
				cofacet_index,
				cofacet_coefficient,
			))
		} else {
			// Skip this one, try next
			self.next()
		}
	}

	/// Advance and return the next cofacet, but only those inserting a vertex
	/// above the highest vertex (for apparent pairs shortcut).
	pub fn next_above_max(&mut self) -> Option<DiameterEntry> {
		if !self.has_next(false) {
			return None;
		}
		self.next()
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::distance::{DistanceMatrix, Euclidean};
	use ndarray::array;

	#[test]
	fn test_coboundary_vertex() {
		// For a vertex in a 3-point cloud, coboundary should give 2 edges
		let points = array![[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]];
		let dm = DistanceMatrix::from_point_cloud(&points.view(), &Euclidean);
		let binomial = BinomialTable::new(4, 4);
		let threshold = dm.max_distance() + 1.0;

		// Vertex 0 has comb index 0, dim 0
		let vertex_idx = crate::utils::combinatorics::simplex_to_index(&[0], &binomial);
		let mut cob = CoboundaryEnumerator::new(vertex_idx, 0, 0.0, 1, 3, 2, threshold, &dm, &binomial);

		let mut cofacets = Vec::new();
		while let Some(cf) = cob.next() {
			cofacets.push(cf);
		}
		assert_eq!(cofacets.len(), 2, "vertex 0 should have 2 cofacets (edges)");
	}

	#[test]
	fn test_coboundary_edge() {
		// For an edge in a triangle, coboundary should give 1 triangle
		let points = array![[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]];
		let dm = DistanceMatrix::from_point_cloud(&points.view(), &Euclidean);
		let binomial = BinomialTable::new(4, 4);
		let threshold = dm.max_distance() + 1.0;

		// Edge {0,1} has comb index C(1,2)+C(0,1) = 0
		let edge_idx = crate::utils::combinatorics::simplex_to_index(&[0, 1], &binomial);
		let edge_diam = dm.get(0, 1);
		let mut cob = CoboundaryEnumerator::new(edge_idx, 1, edge_diam, 1, 3, 2, threshold, &dm, &binomial);

		let mut cofacets = Vec::new();
		while let Some(cf) = cob.next() {
			cofacets.push(cf);
		}
		assert_eq!(cofacets.len(), 1, "edge {{0,1}} in a triangle has 1 cofacet");
	}

	#[test]
	fn test_coboundary_with_threshold() {
		// With a low threshold, some cofacets should be filtered out
		let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 10.0]];
		let dm = DistanceMatrix::from_point_cloud(&points.view(), &Euclidean);
		let binomial = BinomialTable::new(4, 4);
		let threshold = 2.0; // Only the edge {0,1} is within threshold

		let vertex_idx = crate::utils::combinatorics::simplex_to_index(&[0], &binomial);
		let mut cob = CoboundaryEnumerator::new(vertex_idx, 0, 0.0, 1, 3, 2, threshold, &dm, &binomial);

		let mut cofacets = Vec::new();
		while let Some(cf) = cob.next() {
			cofacets.push(cf);
		}
		assert_eq!(
			cofacets.len(),
			1,
			"only 1 edge within threshold 2.0"
		);
	}
}
