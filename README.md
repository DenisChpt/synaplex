# Synaplex

[![Rust](https://img.shields.io/badge/rust-2024%20edition-orange.svg)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)

**Synaplex** is a high-performance Topological Data Analysis (TDA) engine for neural networks. Written in Rust with Python bindings via PyO3, it provides optimised algorithms for persistent homology computation, simplicial complex construction, and topological analysis of neural networks.

---

## Overview

Synaplex combines advanced topological data analysis techniques with a performance-oriented Rust implementation:

- **Persistent homology** via cohomology with clearing optimisation and arbitrary Z/pZ coefficients
- **Simplicial complex construction** using Vietoris-Rips with matrix-free combinatorial index encoding
- **Topological neural network analysis**: activation spaces, weight matrices, loss landscapes
- **Mapper algorithm** for data visualisation and summarisation
- **Diagram distances**: bottleneck and Wasserstein distances

---

## Features

### Persistent Homology

- **Cohomological approach**: implicit cofacet enumeration, emergent pairs, clearing
- **Clearing optimisation**: removal of paired simplices to reduce complexity
- **Arbitrary coefficients**: support for Z/2Z, Z/3Z, or any Z/pZ for prime p
- **H0 via Union-Find**: exact and fast computation of connected components
- **Cocycle extraction**: representatives of persistent cohomology classes

### Neural Network Analysis

- **`ActivationAnalyzer`**: analyses the topology of activation spaces through sampling
- **`WeightAnalyzer`**: topological analysis of weight matrices (rows or columns as point clouds)
- **`LandscapeAnalyzer`**: sampling and topological analysis of the loss landscape
- **Layer comparison**: bottleneck and Wasserstein distances between layers

### Simplicial Complex Construction

- **Vietoris-Rips Complex**: matrix or matrix-free construction with combinatorial encoding
- **Filtrations**: management of filtration values per simplex
- **Combinatorial indexing**: compact simplex encoding (significant memory reduction)

### Distances and Metrics

- **Bottleneck distance**: optimal L∞ matching with binary search and augmenting paths
- **Wasserstein distance**: optimal transport of dimension p
- **Supported metrics**: Euclidean, cosine, Manhattan

### Mapper Algorithm

The Mapper implementation provides a basic pipeline for topological graph summarisation. **Note: This implementation is relatively naive and not highly optimised.** It includes:

- **Complete pipeline**: filter → cover → clustering → nerve → graph
- **Cover strategies**: uniform, balanced
- **Clustering algorithms**: DBSCAN, single linkage
- **petgraph integration**: usable Mapper graph output

---

## Installation

### Prerequisites

- [Rust](https://www.rust-lang.org/tools/install) (2024 edition)
- [Python](https://www.python.org/) ≥ 3.9
- [uv](https://docs.astral.sh/uv/) (recommended) or [Maturin](https://www.maturin.rs/)

### From Source

```bash
git clone <repo-url>
cd synaplex

# With uv (recommended)
uv sync
uv run python -c "import synaplex"

# Or with maturin directly
maturin develop --release
```

### As a Rust Dependency

Add to your `Cargo.toml`:

```toml
[dependencies]
synaplex = { path = "path/to/synaplex" }
```

---

## Modules

### `synaplex.homology`

Persistent homology computation via cohomology.

### `synaplex.complex`

Simplicial complex construction.

### `synaplex.diagram`

Distances between persistence diagrams.

### `synaplex.neural`

Topological analysis of neural networks.

### `synaplex.mapper`

Mapper pipeline for topological graph summarisation.

### `synaplex.distance`

Distance matrices and metrics.

---

## Performance

Synaplex is optimised for performance through:

- **Rust**: systems language with zero-cost abstractions
- **Parallelism**: Rayon for data parallelism
- **Optimised algorithms**: cohomology + clearing rather than standard reduction
- **Combinatorial encoding**: compact simplex indexing (no Vec per simplex)
- **Implicit enumeration**: cofacets generated on-the-fly without allocation

---

## Licence

This project is licensed under the MIT Licence.

---

## Acknowledgements

- [PyO3](https://github.com/PyO3/pyo3) for Python bindings
- [ndarray](https://github.com/rust-ndarray/ndarray) for multidimensional arrays in Rust
- [petgraph](https://github.com/petgraph/petgraph) for graph structures
