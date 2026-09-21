# Targeted Ricci Curvature

A Python package for computing Ollivier-Ricci curvature on graphs and hypergraphs, including directed and undirected variants, weighted and unweighted data, and parallel execution workflows for larger experiments.

## What this package includes

- A reusable `Hypergraph` abstraction with concrete `UndirectedHypergraph` and `DirectedHypergraph` implementations in `hypergraph.py`
- Utility helpers for timing, CSV output, and normalization in `ricciutil.py`
- MPI-based parallel workflows for targeted and untargeted curvature computations in `targeted_parallel.py`, `untargeted_parallel.py`, and `untargeted_community_comparison_parallel.py`
- Sphinx API documentation generated from the module source under `docs/`

## Installation

Install the project dependencies from the repository root:

```bash
pip install -r requirements.txt
```

For local documentation builds, install the docs extras as well:

```bash
pip install -r docs/requirements.txt
```

## Quick start

The library is centered around a hypergraph object. The simplest workflow is to construct a graph from a DataFrame, update the node index, and compute a distance matrix before evaluating curvature on an edge.

```python
import pandas as pd
from hypergraph import UndirectedHypergraph

edge_data = pd.DataFrame([
    {"source": "A", "target": "B", "weight": 1.0},
    {"source": "B", "target": "C", "weight": 1.0},
    {"source": "C", "target": "A", "weight": 1.0},
])

graph = UndirectedHypergraph()
graph.build_from_dataframe(edge_data, verbose=False)
graph.update_node_index()

distance_matrix = graph.floyd_warshall()
print(distance_matrix)

edge_id = "A_to_B"
orc = graph.earthmover_distance_hyperedge_combinations(
    edge_id,
    distance_matrix,
    approx_emd=False,
    gurobi_flag=False,
    verbose=False,
)
print(f"ORC for {edge_id}: {orc}")
```

## Repository layout

- `hypergraph.py` defines the core hypergraph models and curvature logic.
- `ricciutil.py` contains shared helpers for output formatting and timing.
- `targeted_parallel.py` runs targeted ORC calculations across MPI workers.
- `untargeted_parallel.py` and `untargeted_community_comparison_parallel.py` support broader comparison workflows.
- `docs/` contains the Sphinx documentation sources.

## Documentation

The generated documentation can be built locally with:

```bash
sphinx-build -b html docs docs/_build/html
```

See the docs index in `docs/index.rst` for the full project overview and module reference.

## Notes

This project is designed for research and experimentation with Ollivier-Ricci curvature on hypergraphs. The implementation is built around custom data structures rather than a general-purpose graph library API, so the most reliable usage is to follow the object model in `hypergraph.py` and the patterns in the parallel modules.