/// Bit-packed simplex entry: stores a combinatorial index and a coefficient
/// in a single `u64` for cache-friendly access.
///
/// Layout (when using coefficients):
///   - Lower 56 bits: simplex index (max ~7.2 × 10^16)
///   - Upper 8 bits: coefficient in Z/pZ
///
/// When using Z/2Z (the common case), the coefficient is implicit (always 1
/// for nonzero entries) and all 64 bits store the index.

const COEFF_BITS: u32 = 8;
const INDEX_MASK: u64 = (1u64 << (64 - COEFF_BITS)) - 1;
const COEFF_SHIFT: u32 = 64 - COEFF_BITS;

/// A packed (index, coefficient) pair fitting in 64 bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Entry(u64);

impl Entry {
	/// Pack an index and coefficient into a single `u64`.
	#[inline(always)]
	pub fn new(index: u64, coefficient: u16) -> Self {
		debug_assert!(index <= INDEX_MASK, "simplex index overflow");
		Self(index | ((coefficient as u64) << COEFF_SHIFT))
	}

	/// Entry with coefficient 1 (the common case for Z/2Z).
	#[inline(always)]
	pub fn from_index(index: u64) -> Self {
		Self::new(index, 1)
	}

	#[inline(always)]
	pub fn index(self) -> u64 {
		self.0 & INDEX_MASK
	}

	#[inline(always)]
	pub fn coefficient(self) -> u16 {
		(self.0 >> COEFF_SHIFT) as u16
	}

	#[inline(always)]
	pub fn set_coefficient(&mut self, c: u16) {
		self.0 = (self.0 & INDEX_MASK) | ((c as u64) << COEFF_SHIFT);
	}

	/// Raw packed value (for use as hash key or comparison).
	#[inline(always)]
	pub fn raw(self) -> u64 {
		self.0
	}

	#[inline(always)]
	pub fn is_null(self) -> bool {
		self.index() == INDEX_MASK
	}

	#[inline(always)]
	pub fn null() -> Self {
		Self(INDEX_MASK)
	}
}

impl PartialOrd for Entry {
	#[inline]
	fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
		Some(self.cmp(other))
	}
}

impl Ord for Entry {
	#[inline]
	fn cmp(&self, other: &Self) -> std::cmp::Ordering {
		self.index().cmp(&other.index())
	}
}

/// A diameter-entry pair: (filtration_value, packed_entry).
/// Used in priority queues during cohomology reduction.
#[derive(Clone, Copy, Debug)]
pub struct DiameterEntry {
	pub diameter: f64,
	pub entry: Entry,
}

impl DiameterEntry {
	#[inline(always)]
	pub fn new(diameter: f64, index: u64, coefficient: u16) -> Self {
		Self {
			diameter,
			entry: Entry::new(index, coefficient),
		}
	}

	#[inline(always)]
	pub fn index(self) -> u64 {
		self.entry.index()
	}

	#[inline(always)]
	pub fn coefficient(self) -> u16 {
		self.entry.coefficient()
	}
}

impl PartialEq for DiameterEntry {
	fn eq(&self, other: &Self) -> bool {
		self.entry == other.entry
	}
}

impl Eq for DiameterEntry {}

/// Ordering for the priority queue: smaller diameter first, then larger index first.
impl PartialOrd for DiameterEntry {
	fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
		Some(self.cmp(other))
	}
}

impl Ord for DiameterEntry {
	fn cmp(&self, other: &Self) -> std::cmp::Ordering {
		// Reverse: we want a min-heap by diameter, max-heap by index
		other
			.diameter
			.partial_cmp(&self.diameter)
			.unwrap_or(std::cmp::Ordering::Equal)
			.then_with(|| self.entry.index().cmp(&other.entry.index()))
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_entry_pack_unpack() {
		let e = Entry::new(42, 3);
		assert_eq!(e.index(), 42);
		assert_eq!(e.coefficient(), 3);
	}

	#[test]
	fn test_entry_z2() {
		let e = Entry::from_index(12345);
		assert_eq!(e.index(), 12345);
		assert_eq!(e.coefficient(), 1);
	}

	#[test]
	fn test_entry_large_index() {
		let max_idx = INDEX_MASK;
		let e = Entry::new(max_idx, 7);
		assert_eq!(e.index(), max_idx);
		assert_eq!(e.coefficient(), 7);
	}

	#[test]
	fn test_entry_null() {
		let e = Entry::null();
		assert!(e.is_null());
		assert!(!Entry::from_index(0).is_null());
	}

	#[test]
	fn test_diameter_entry_ordering() {
		let a = DiameterEntry::new(1.0, 10, 1);
		let b = DiameterEntry::new(2.0, 5, 1);
		// a has smaller diameter → should come first in min-heap (a > b in Ord)
		assert!(a > b);
	}
}
