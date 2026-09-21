Targeted Ricci Curvature documentation
=====================================

This project implements Ollivier-Ricci curvature computations for graphs and hypergraphs, with support for directed and undirected structures, weighted and unweighted data, and parallel analysis workflows. It was developed as part of the research from Bhaskar DasGupta's `Creative Algorithms lab <https://bdasgup.github.io/professional/algo-lab.html>`.

The package is organized around the core ``Hypergraph`` abstraction, which stores nodes, hyperedges, edge weights, and curvature values. In practice, users build a graph from a DataFrame or a custom object model, compute shortest-path distances, and then evaluate curvature on each hyperedge or pair of nodes.

Getting started
---------------

The fastest way to begin is to install the project dependencies and use the object model in :mod:`hypergraph`.

.. code-block:: bash

   pip install -r requirements.txt

Then construct a graph object, populate it with edges, and call the curvature utilities.

.. code-block:: python

   import pandas as pd
   from hypergraph import UndirectedHypergraph

   data = pd.DataFrame([
       {"source": "A", "target": "B", "weight": 1.0},
       {"source": "B", "target": "C", "weight": 1.0},
       {"source": "C", "target": "A", "weight": 1.0},
   ])

   graph = UndirectedHypergraph()
   graph.build_from_dataframe(data)
   graph.update_node_index()

   distance_matrix = graph.floyd_warshall()
   print(distance_matrix)

Project structure
-----------------

The repository is organized into a few main components:

- :mod:`hypergraph` contains the graph and hypergraph classes and the main ORC logic.
- :mod:`ricciutil` provides CSV output and timing helpers used by the analysis workflows.
- :mod:`targeted_parallel` and :mod:`untargeted_parallel` run larger computations with MPI.
- :mod:`untargeted_community_comparison_parallel` supports community-level comparison routines.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
