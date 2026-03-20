mod convert;
mod py_complex;
mod py_diagram;
mod py_distance;
mod py_homology;
mod py_mapper;
mod py_neural;

use pyo3::prelude::*;

// Module-level documentation for stub files
pyo3_stub_gen::module_doc!(
	"synaplex",
	"Synaplex — high-performance Topological Data Analysis engine for neural networks.\n\
	 \n\
	 Synaplex provides Rust-accelerated persistent homology, Mapper, and\n\
	 topological neural-network analysis tools with a Pythonic interface.\n\
	 \n\
	 Submodules\n\
	 ----------\n\
	 homology : Persistent cohomology computation.\n\
	 neural   : Topological signatures for activations and weights.\n\
	 complex  : Vietoris-Rips simplicial complex construction.\n\
	 distance : Pairwise distance matrix computation.\n\
	 diagram  : Bottleneck and Wasserstein distances on persistence diagrams.\n\
	 mapper   : The Mapper pipeline for graph-based data summarisation."
);

pyo3_stub_gen::module_doc!(
	"synaplex.homology",
	"Persistent cohomology computation.\n\
	 \n\
	 This module computes persistence diagrams and Betti numbers from\n\
	 point clouds using cohomology, clearing optimisation, and arbitrary\n\
	 Z/pZ coefficients."
);

pyo3_stub_gen::module_doc!(
	"synaplex.neural",
	"Topological analysis tools for neural network layers.\n\
	 \n\
	 Analyse the shape of activation spaces, weight matrices, and compare\n\
	 the topological signatures of different layers."
);

pyo3_stub_gen::module_doc!(
	"synaplex.complex",
	"Simplicial complex construction.\n\
	 \n\
	 Build Vietoris-Rips complexes from point clouds using matrix-free,\n\
	 combinatorial-index encoding for memory efficiency."
);

pyo3_stub_gen::module_doc!(
	"synaplex.distance",
	"Distance matrix computation.\n\
	 \n\
	 Compute dense pairwise distance matrices with Euclidean, cosine, or\n\
	 Manhattan metrics. Computation is parallelised with Rayon."
);

pyo3_stub_gen::module_doc!(
	"synaplex.diagram",
	"Distances between persistence diagrams.\n\
	 \n\
	 Compute the bottleneck distance (L-infinity optimal matching) and\n\
	 p-Wasserstein distance (optimal transport) between persistence\n\
	 diagrams."
);

pyo3_stub_gen::module_doc!(
	"synaplex.mapper",
	"The Mapper pipeline for topological graph summarisation.\n\
	 \n\
	 Mapper projects data through a filter function, covers the range\n\
	 with overlapping intervals, clusters within each interval, and\n\
	 builds the nerve complex — a graph that reveals the global shape\n\
	 of the data."
);

// Re-export top-level convenience functions from homology into the root module
pyo3_stub_gen::reexport_module_members!(
	"synaplex",
	"synaplex.homology",
	"persistence_diagram",
	"betti_numbers",
	"test_fn"
);

/// Python module: synaplex
///
/// High-performance Topological Data Analysis engine for neural networks.
#[pymodule]
fn synaplex(m: &Bound<'_, PyModule>) -> PyResult<()> {
	// Distance submodule
	let distance_mod = PyModule::new(m.py(), "distance")?;
	py_distance::register(&distance_mod)?;
	m.add_submodule(&distance_mod)?;

	// Complex submodule
	let complex_mod = PyModule::new(m.py(), "complex")?;
	py_complex::register(&complex_mod)?;
	m.add_submodule(&complex_mod)?;

	// Homology submodule
	let homology_mod = PyModule::new(m.py(), "homology")?;
	py_homology::register(&homology_mod)?;
	m.add_submodule(&homology_mod)?;

	// Diagram submodule
	let diagram_mod = PyModule::new(m.py(), "diagram")?;
	py_diagram::register(&diagram_mod)?;
	m.add_submodule(&diagram_mod)?;

	// Mapper submodule
	let mapper_mod = PyModule::new(m.py(), "mapper")?;
	py_mapper::register(&mapper_mod)?;
	m.add_submodule(&mapper_mod)?;

	// Neural submodule
	let neural_mod = PyModule::new(m.py(), "neural")?;
	py_neural::register(&neural_mod)?;
	m.add_submodule(&neural_mod)?;

	// Top-level convenience functions
	m.add_function(wrap_pyfunction!(py_homology::persistence_diagram, m)?)?;
	m.add_function(wrap_pyfunction!(py_homology::betti_numbers, m)?)?;
	m.add_function(wrap_pyfunction!(py_homology::test_fn_py, m)?)?;

	Ok(())
}
