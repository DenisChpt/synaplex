pub mod complex;
pub mod diagram;
pub mod distance;
pub mod error;
pub mod homology;
pub mod mapper;
pub mod neural;
mod python;
pub mod utils;

pub use error::{Result, SynaplexError};

pyo3_stub_gen::define_stub_info_gatherer!(stub_info);
