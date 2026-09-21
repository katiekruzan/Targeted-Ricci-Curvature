Getting started
===============

Installation
------------

Install the project requirements from the repository root:

.. code-block:: bash

   pip install -r requirements.txt

If you want to build the HTML documentation locally as well, install the docs dependencies:

.. code-block:: bash

   pip install -r docs/requirements.txt

Core concepts
-------------

The package is built around a small set of graph objects:

- ``Hypergraph`` is the abstract base class for all graph representations.
- ``UndirectedHypergraph`` stores each hyperedge as a list of nodes and keeps edge weights in a parallel dictionary.
- ``DirectedHypergraph`` stores hyperedges as tail and head sets, which is useful when the edge direction matters.

The general workflow is:

1. Construct or load a graph.
2. Add nodes and edges.
3. Update the node index.
4. Compute shortest-path distances with ``floyd_warshall``.
5. Compute curvature or compare graphs using the ORC APIs.

Example: building a small undirected graph
------------------------------------------

.. code-block:: python

   import pandas as pd
   from hypergraph import UndirectedHypergraph

   df = pd.DataFrame([
       {"source": "A", "target": "B", "weight": 1.0},
       {"source": "B", "target": "C", "weight": 1.0},
       {"source": "C", "target": "A", "weight": 1.0},
   ])

   graph = UndirectedHypergraph()
   graph.build_from_dataframe(df, verbose=False)
   graph.update_node_index()

   distance_matrix = graph.floyd_warshall()
   print(distance_matrix)

This creates a simple triangle-like network and computes the all-pairs distance matrix that the ORC functions use internally.

Example: computing edge curvature
----------------------------------

.. code-block:: python

   edge_id = "A_to_B"
   orc = graph.earthmover_distance_hyperedge_combinations(
       edge_id,
       distance_matrix,
       approx_emd=False,
       gurobi_flag=False,
       verbose=False,
   )
   print(f"ORC for {edge_id}: {orc}")

The exact ORC value depends on the graph structure, edge weights, and the probability distributions defined by the hypergraph model.

Parallel workflows
------------------

For larger graphs, the package also provides MPI-based parallel modules:

- ``targeted_parallel`` performs targeted curvature calculations.
- ``untargeted_parallel`` computes untargeted curvature across the whole graph.
- ``untargeted_community_comparison_parallel`` compares communities or graph states in parallel.

These modules are designed for research experiments and larger datasets, where a single-threaded pass is too slow.
