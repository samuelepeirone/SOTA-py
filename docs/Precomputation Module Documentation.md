# Precomputation Module Documentation

## Overview

This codebase implements two families of graph-based preprocessing techniques aimed at improving shortest-path search:
1. **Reach-based pruning**: removes nodes that cannon belong to shortest (or optimal) paths.
2. **Arc-Flags pruning**: partitions the graph and assigns flags to edges, allowing the search to ignore edges that are irrilevant to the destination region.

Both approaches are implemented in brute-force and deterministic variants:

|Technique|Brute-Force Variant|Deterministic Variant|
|---|---|---|
|Reach|`bfReach`|`detReach`|
|Arc-Flags|`bfArcFlags`|`detArcFlags`|
The system is designed to work with:
- a `graph` object providing adjacency/variance matrices and pruning functions
- either a SOTA solver or a deterministic shortest-path algorithm

---
## Class Documentation

---
### 1. Reach Framework

---
#### `class Reach(ABC)`

Base class defining the interface for Reach computation.

##### Responsibilities

- Stores graph reference and node count
- Defines structure for reach values
- Implements basic utilities:
	- `m(i,j)`: travel time measure between nodes
	- printing / retrieving reach values

##### Abstract Methods to be Implemented

| Method                   | Description                                      |
| ------------------------ | ------------------------------------------------ |
| `reach_computation(...)` | Computes reach values for each node in the graph |
| `reach_pruning(...)`     | Pruning of the nodes, based on reach values      |

##### Pruning Rule

A node `i` is removed from the graph if:

```python
reach(i) < min( m(s,i), m(i,d) )
```

---
#### `class bfReach(Reach)`

Brute-force reach computation based on enumerating optimal paths using a SOTA solver.

###### Key Characteristics

- For each destination `d`, SOTA is solved
- For every source `s`, all optimal paths from `s` to `d` are collected
- Reach of node `i` updated using:

```python
reach(i) = max( reach(i), min(m(s,i), m(i,d)) )
```

###### Main methods

| Method                         | Description                                      |
| ------------------------------ | ------------------------------------------------ |
| `get_optimal_path_nodes(s, d)` | Returns nodes appearing in any optimal SOTA path |
| `reach_computation()`          | Computes reach for all nodes exhaustively        |
| `reach_pruning(s, d)`          | Uses inherited pruning rules                     |

---
#### `class detReach(Reach)`

Deterministic reach computation using a classical shortest-path algorithm

##### Key Characteristics

- Computes distances and optimal predecessor chains from every source
- Optimal paths are recontructed to determine visited nodes
- Reach computed using:

```python
m_si = dist_s[i]
m_id = dist_s[d]-dist_s[i]
```

##### Main methods

| Method                | Description                           |
| --------------------- | ------------------------------------- |
| `reach_computation()` | Computes reach deterministically      |
| `reach_pruning(s, d)` | Removes nodes with insufficient reach |

---
#### Example Usage - detReach

```python
import numpy as np
from reach import detReach
from dijkstra import Dijkstra
from graph import StochasticGraph
from Grid_network_and_Gamma_distribution import Matrix

# --- build a graph from randomly-generated matrices ---
matrix = Matrix(5,5)
adj, var = matrix.compute_matrices()

graph = StochasticGraph(adj, var)

# --- initialize deterministic shortest path algorithm ---
det_algo = Dijkstra(adj)

# --- initialize deterministic reach computation ---
reach_solver = detReach(graph, det_algo)

# --- compute reach values ---
reach_values = reach_solver.reach_computation()

# --- print results ---
reach_solver.print_reach_values()

# --- prune unreachable / irrelevant nodes from s to d ---
source = 0
destination = 24

pruned_nodes = reach_solver.reach_pruning(source, destination)
print("Pruned nodes:", pruned_nodes)
```

---
### 2. Arc-Flags Framework

---
#### `class ArcFlags(ABC)`

Base class for Arc-Flags preprocessing.

##### Responsibilities

- Stores graph and source / destination
- Builds partitioning of nodes into regions using Canopy-accelerated K-means
- Maintains arc-flag table per edge per region
- Provides printing and initialization utilities

##### Abstract Methods to be Implemented

| Method                      | Description                                    |
| --------------------------- | ---------------------------------------------- |
| `arcflags_computation(...)` | Computes arcflaìgs values for each edge        |
| `arcflags_pruning(...)`     | Pruning of the edges, based on arcflags values |

##### Graph Partitioning Components

`canopy_clustering(X, T1, T2)`: produces rough clusters (canopies) based on relaxed distance rules

`canopy_kmeans(X, canopies, centers)`: refines clustering by restricting distance checks only to point sharing a canopy

`partition_graph()`: constructs feature vectors:

```python
x_i = [ mean_outgoing_edge_weight , variance_outgoing_edge_weight ]
```

Then:
1. scales features
2. forms canopies
3. refines via Canopy-kMeans
4. assigns region to each node

##### Pruning Rule

An edge `(u,v)` is removed from the graph if:

```python
arc_flags[(u,v)][ region(d) ] == False
```

---
#### `class bfArcFlags(ArcFlags)`

Brute-force Arc-Flags computation based on SOTA optimal policies.

##### Approach

For each destination `d`:
1. Retrieve optimal SOTA policy
2. Extract all edges appearing in optimal paths to `d`
3. Set flags for edges relevant to region of `d`

##### Main methods

| Method                      | Description                                  |
| --------------------------- | -------------------------------------------- |
| `compute_optimal_policy(d)` | Extracts edges used in optimal SOTA policies |
| `collect_relevant_edges(d)` | Aggregates edge sets                         |
| `arcflags_computation()`    | Computes flag table                          |

---
#### `class detArcFlags(ArcFlags)`

Deterministic Arc-Flags based on deterministic optimal path.

##### Approach

For each source `s`:
1. Compute predecessors and distances
2. Reconstruct all optimal paths to all `d`
3. Mark visited edges with the region of `d`

##### Main methods

|Method|Description|
|---|---|
|`arcflags_computation()`|deterministic multi-source computation|
|`arcflags_pruning()`|removes edges irrelevant to destination region|

---

#### Example Usage - detArcFlags

```python
import numpy as np
from reach import detReach
from dijkstra import Dijkstra
from graph import StochasticGraph
from Grid_network_and_Gamma_distribution import Matrix

# --- build a graph from randomly-generated matrices ---
matrix = Matrix(5,5)
adj, var = matrix.compute_matrices()

graph = StochasticGraph(adj, var)

det_algo = Dijkstra(adj)

# choose destination
node_d = 24

arcflags = detArcFlags(graph, det_algo, node_d=node_d)

# compute region partition + arc flags
arcflags.arcflags_computation()

# print flags
arcflags.print_arcflags()

# prune edges irrelevant for destination region
arcflags.arcflags_pruning()

# visualize graph sections (optional)
arcflags.print_graph_sections()
```
