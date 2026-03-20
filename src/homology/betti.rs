use super::PersistencePair;

/// Compute Betti numbers at a given filtration value from persistence pairs.
///
/// beta_k(t) = number of k-dimensional features alive at filtration value t
///           = |{pairs (b,d) in dim k : b <= t < d}|
pub fn betti_numbers(pairs: &[PersistencePair], at_filtration: f64) -> Vec<usize> {
	let max_dim = pairs.iter().map(|p| p.dimension).max().unwrap_or(0);
	let mut betti = vec![0usize; max_dim + 1];

	for pair in pairs {
		if pair.birth <= at_filtration && at_filtration < pair.death {
			betti[pair.dimension] += 1;
		}
	}

	betti
}

/// Compute Betti numbers at a range of filtration values.
/// Returns a vector of (filtration_value, betti_numbers) pairs.
pub fn betti_curve(pairs: &[PersistencePair], values: &[f64]) -> Vec<(f64, Vec<usize>)> {
	values
		.iter()
		.map(|&t| (t, betti_numbers(pairs, t)))
		.collect()
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_betti_numbers() {
		let pairs = vec![
			PersistencePair {
				birth_idx: 0,
				death_idx: None,
				birth: 0.0,
				death: f64::INFINITY,
				dimension: 0,
			},
			PersistencePair {
				birth_idx: 1,
				death_idx: Some(3),
				birth: 0.0,
				death: 1.0,
				dimension: 0,
			},
			PersistencePair {
				birth_idx: 4,
				death_idx: Some(6),
				birth: 1.5,
				death: 3.0,
				dimension: 1,
			},
		];

		let b = betti_numbers(&pairs, 0.5);
		assert_eq!(b[0], 2); // both H0 alive
		assert_eq!(b[1], 0); // H1 not born yet

		let b = betti_numbers(&pairs, 1.5);
		assert_eq!(b[0], 1); // one H0 died
		assert_eq!(b[1], 1); // H1 alive

		let b = betti_numbers(&pairs, 5.0);
		assert_eq!(b[0], 1); // essential H0 still alive
		assert_eq!(b[1], 0); // H1 died
	}
}
