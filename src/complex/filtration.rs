use ordered_float::OrderedFloat;

/// Wraps f64 with total ordering for filtration-based sorting.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FiltrationValue(pub OrderedFloat<f64>);

impl FiltrationValue {
	pub fn new(val: f64) -> Self {
		Self(OrderedFloat(val))
	}

	pub fn value(&self) -> f64 {
		self.0.into_inner()
	}
}

impl From<f64> for FiltrationValue {
	fn from(val: f64) -> Self {
		Self::new(val)
	}
}
