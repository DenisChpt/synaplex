use thiserror::Error;

#[derive(Debug, Error)]
pub enum SynaplexError {
	#[error("dimension mismatch: expected {expected}, got {got}")]
	DimensionMismatch { expected: usize, got: usize },

	#[error("empty input: {0}")]
	EmptyInput(&'static str),

	#[error("invalid parameter: {0}")]
	InvalidParameter(String),

	#[error("index out of bounds: {index} >= {size}")]
	IndexOutOfBounds { index: usize, size: usize },

	#[error("computation failed: {0}")]
	ComputationError(String),
}

pub type Result<T> = std::result::Result<T, SynaplexError>;
