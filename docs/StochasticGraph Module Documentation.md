# Stochastic Module Documentation

## Overview

The `StochasticGraph` module defines a directed graph where each edge represents a travel time modeled as a stochastic gamma-distributres random variable. The graph is represented using:
- an adjacency matrix (mean travel times)
- a variance matrix (travel-time variances)
The module also provides visualization utilities using NetworkX and Matplotlib.

## Dependencies

```lua
numpy
networkx
matplotlib
math
```

## Key Features

### Graph Representation

- Directed graph based on adjacency matrix
- Variance matrix aligned with adjacency matrix
- Automatic detection of:
	- number of nodes
	- minimum non-zero edge weight
	- active ndoes (with inbound/outbound edges)

### Stochastic Edge Modeling

- Travel time sampled using gamma distribution
- Shape and scale computed from mean and variance
- Sampling lower-bounded by minimum edge time

### Graph Inspection Utilities

- Print adjacency matrix
- Print variance matrix
- List nodes and edges
- Query incoming and outgoing neighbors
- Retrieve edge mean and variance

### Graph Manipulation

- Reverse all edges
- Prune nodes
- Prune edges
- Recalculate min edge when graph changes

### Visualization Tools

Supports different display modes:

`print_graph_all_nodes()`
- Shows entire graph including isolated nodes

`print_graph()`
- Shows only active nodes
- Optional
	- highlight a path
	- toggle edge labels
	- fixed grid layout

`print_graph_sections()`
- Colors nodes by user-defined grouping
- Displays legend automatically
- Can highlight a path simultaneously

---

### Constructor Parameters

|Name|Description|
|---|---|
|`adjacency_matrix`|square matrix of edge mean travel times|
|`variance_matrix`|square matrix of edge variances|

### Important attributes

|Attribute|Meaning|
|---|---|
|`adjacency_matrix`|mean travel times|
|`variance_matrix`|variance values|
|`min_edge`|smallest positive edge|
|`num_nodes`|number of nodes in graph|

---
## Example Usage

```python
from stochastic_graph import StochasticGraph
import numpy as np

# Build graph
adj = np.array([
    [0, 4, 5],
    [0, 0, 3],
    [0, 0, 0]
])

var = np.array([
    [0, 1/2, 1/3],
    [0, 0, 1/2],
    [0, 0, 0]
])

graph = StochasticGraph(adj, var)

# Inspect
print(graph.get_nodes())
print(graph.get_edges())
print("Min edge:", graph.get_min_edge())

# Sample travel time
sample = graph.sample_distance(0, 1)
print("Sampled travel time:", sample)

# Visualize
graph.print_graph(path=[0,1,2])
```

## Notes

- Zero entries in matrices represent absence of edges
- Means and variances must be positive when edge exists
- Sampling returns `inf` in no edge exists
- Node pruning preserves index consistency
- Visualization ignores isolated nodes unless using `print_graph_aa_nodes()`
