/// Modular arithmetic over Z/pZ for arbitrary prime p.
///
/// Supports Z/2Z (the default, optimized via bitwise AND) and arbitrary
/// primes (Z/3Z, Z/5Z, etc.) for detecting torsion and more complex
/// topological structures.

/// A coefficient field Z/pZ.
#[derive(Clone, Debug)]
pub struct CoefficientField {
	modulus: u16,
	/// Precomputed multiplicative inverses: `inverse[a]` = a^(-1) mod p.
	inverse: Vec<u16>,
}

impl CoefficientField {
	/// Create a new coefficient field Z/pZ.
	/// `modulus` must be a prime number.
	pub fn new(modulus: u16) -> Self {
		debug_assert!(modulus >= 2, "modulus must be >= 2");
		let inverse = multiplicative_inverse_table(modulus);
		Self { modulus, inverse }
	}

	/// The default Z/2Z field.
	pub fn z2() -> Self {
		Self::new(2)
	}

	#[inline(always)]
	pub fn modulus(&self) -> u16 {
		self.modulus
	}

	/// Whether this is the Z/2Z field (enables fast-path optimizations).
	#[inline(always)]
	pub fn is_z2(&self) -> bool {
		self.modulus == 2
	}

	#[inline(always)]
	pub fn reduce(&self, val: u16) -> u16 {
		if self.modulus == 2 {
			val & 1
		} else {
			val % self.modulus
		}
	}

	#[inline(always)]
	pub fn add(&self, a: u16, b: u16) -> u16 {
		self.reduce(a + b)
	}

	#[inline(always)]
	pub fn multiply(&self, a: u16, b: u16) -> u16 {
		self.reduce(((a as u32) * (b as u32)) as u16)
	}

	#[inline(always)]
	pub fn negate(&self, a: u16) -> u16 {
		if a == 0 {
			0
		} else {
			self.modulus - a
		}
	}

	#[inline(always)]
	pub fn inverse(&self, a: u16) -> u16 {
		debug_assert!(a > 0 && (a as usize) < self.inverse.len());
		self.inverse[a as usize]
	}

	/// Normalize a coefficient to the range (-p/2, p/2].
	/// Used for cocycle output.
	#[inline]
	pub fn normalize(&self, n: u16) -> i16 {
		let half = self.modulus / 2;
		if n > half {
			n as i16 - self.modulus as i16
		} else {
			n as i16
		}
	}
}

/// Precompute the multiplicative inverse table for Z/pZ.
/// Uses the recurrence: inverse[a] = -(p / a) * inverse[p % a] mod p
/// which is O(p) and works for any prime p.
fn multiplicative_inverse_table(p: u16) -> Vec<u16> {
	let m = p as usize;
	let mut inv = vec![0u16; m];
	if m <= 1 {
		return inv;
	}
	inv[1] = 1;
	for a in 2..m {
		// p = (p/a)*a + (p%a)
		// => 0 ≡ (p/a)*a + (p%a) (mod p)
		// => inv[a] = -(p/a) * inv[p%a] (mod p)
		inv[a] = (p - ((p / a as u16) * inv[(p % a as u16) as usize]) % p) % p;
	}
	inv
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_z2_field() {
		let f = CoefficientField::z2();
		assert!(f.is_z2());
		assert_eq!(f.reduce(0), 0);
		assert_eq!(f.reduce(1), 1);
		assert_eq!(f.reduce(2), 0);
		assert_eq!(f.reduce(3), 1);
		assert_eq!(f.add(1, 1), 0); // 1+1=0 mod 2
		assert_eq!(f.negate(1), 1); // -1=1 mod 2
	}

	#[test]
	fn test_z3_field() {
		let f = CoefficientField::new(3);
		assert!(!f.is_z2());
		assert_eq!(f.add(2, 2), 1); // 4 mod 3 = 1
		assert_eq!(f.multiply(2, 2), 1); // 4 mod 3 = 1
		assert_eq!(f.negate(1), 2); // -1 mod 3 = 2
		assert_eq!(f.inverse(1), 1);
		assert_eq!(f.inverse(2), 2); // 2*2=4≡1 mod 3
	}

	#[test]
	fn test_z5_field() {
		let f = CoefficientField::new(5);
		assert_eq!(f.inverse(2), 3); // 2*3=6≡1 mod 5
		assert_eq!(f.inverse(3), 2);
		assert_eq!(f.inverse(4), 4); // 4*4=16≡1 mod 5
		for a in 1..5u16 {
			assert_eq!(f.multiply(a, f.inverse(a)), 1, "a={a}");
		}
	}

	#[test]
	fn test_z7_inverses() {
		let f = CoefficientField::new(7);
		for a in 1..7u16 {
			let inv = f.inverse(a);
			assert_eq!(
				f.multiply(a, inv),
				1,
				"inverse of {a} should be {inv} mod 7"
			);
		}
	}

	#[test]
	fn test_normalize() {
		let f = CoefficientField::new(5);
		assert_eq!(f.normalize(0), 0);
		assert_eq!(f.normalize(1), 1);
		assert_eq!(f.normalize(2), 2);
		assert_eq!(f.normalize(3), -2); // 3 > 5/2
		assert_eq!(f.normalize(4), -1); // 4 > 5/2
	}
}
