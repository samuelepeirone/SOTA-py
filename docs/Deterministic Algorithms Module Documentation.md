# Deterministic Algorithms Module Documentation

## `class Dijkstra`
### Overview

The `Dijkstra` class provides a deterministic shortes-path computation module based on NetworkX's implementation of Dijkstra's algorithm.
It supports:
- directed graphs
- weighted adjacency matrices
- multiple predecessor path reconstruction
- extraction of nodes and edges belonging to all optimal paths
- reverse-direction computation (for destination-based searches)
This class is suitable for integration with **deterministic Reach** and **deterministic Arc-Flags preprocessing**.

### Functionalities provided

The following functionalities are required by `detReach` and `detArcFlags` classes:

|Requirement|Provided by|
|---|---|
|compute distances from a source|`compute_path()`|
|compute distances to a destination|`compute_path(invert=True)`|
|extract nodes in optimal paths|`get_nodes_from_optimal_paths()`|
|extract edges in optimal paths|`get_edges_from_optimal_paths()`|
|support for multiple optimal predecessors|`get_all_optimal_paths()`|

---

## Example Usage

Adjacency matrix is retrieved from the `StochasticGraph` instance

```python
adj = graph.get_adjacency_matrix()

solver = Dijkstra(adj)

pred, dist = solver.compute_path(start_node=0)
paths = solver.get_all_optimal_paths(pred, dest_node=2)

nodes = solver.get_nodes_from_optimal_paths(paths)
edges = solver.get_edges_from_optimal_paths(paths)

print(dist)
print(paths)
print(nodes)
print(edges)
```