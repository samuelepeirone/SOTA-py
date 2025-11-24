# SOTA Module Documentation

## Overview

The software implements the **Stochastic On-Time Arrival (SOTA)** routing model. The SOTA problem returns the policy that maximizes the probability of arriving at the destination node within the time budget $T$.

The module includes:
- An abstract base class:
	- `SOTA`: defines the shared mechanics of the SOTA programming model
- Two concrete solver implementation:
	- `StandardSOTASolver`: successive approiximations solver
	- `SingleIterationSOTASolver`: single iteration SOTA solver
The two solvers operate over a `StochasticGraph` instance.

## Dependencies

```python
numpy
scipy
math
abc
stochastic_graph.StochasticGraph
```

Travel time density evaluation uses:

```python
from scipy.stats import gamma
```

## Time Discretization

The time budget is discretized according to the minimum edge weight

## Core Data Structures

#### `sota_matrix`

A matrix of dimension `[num_nodes, num_cols]`, with `num_cols = ceil(time_budget / min_edge)`. Each entry stores the probability of reaching the destination node from node `i` with remaining time corresponding to column `t`.

#### `policy_matrix`

A matrix of equal dimension as `sota_matrix`, storing:
- the optimal successotr node for each `(node, time)` pair
- `-1` indicates no feasible policy

---
## Class Documentation

---

### `class SOTA(ABC)`

#### Purpose

Abstract base class defining the dynamic programming structure and shared functionality required for computing SOTA probabilities and extraction of optimal policies.

#### Constructor Parameters

|Parameter|Description|
|---|---|
|`graph`|A `StochasticGraph` instance providing means, variances, adjacency and sampling|
|`node_d`|Index of the destination node|
|`time_budget`|Maximum allowed travel time|
#### Abstract methods to implement

| Method                     | Description                                             |
| -------------------------- | ------------------------------------------------------- |
| `compute_convolution(...)` | Computes discrete convolution for a node–successor pair |
| `update_node(...)`         | Updates one node row of SOTA values                     |
| `update_sota(...)`         | Performs one SOTA iteration step                        |
| `solve(...)`               | Runs the SOTA solver until completion                   |
| `extract_path(...)`        | Extracts best route from a given starting node          |

#### Provided Core Methods

##### Density computation

`compute_density(node_i, node_j, s)`
- evaluates gamma travel-time density using mean and variance
- avoids overflow via log-domain computation

##### Path extraction

- `extract_path_from_time(...)`
- `extract_path(...)`
- `extract_path_from_policy(...)` *(static)*

---

### `class StandardSOTASolver(SOTA)`

#### Purpose

Implements the successiv approximation approach to solving SOTA problem.

#### Algorithm Characteristics

- iteratively updates all nodes until convergence
- uses previous iteration matrix for stable updates
- stops when `delta` is below threshold.

#### Key Methods

|Method|Description|
|---|---|
|`compute_convolution(...)`|Uses base class convolution|
|`update_node(...)`|Updates full probability row and best successor|
|`update_sota(...)`|Performs full matrix update, returns update magnitude|
|`solve(...)`|Iterates until convergence or max iterations|

#### Output

- final `sota_matrix`
- fully populated `policy_matrix`

---

### `class SingleIterationSOTASolver(SOTA)`

#### Purpose

Implements a single iteration approach, updating probability estimates incrementally from low to high time indices. In this approach, the matrix is updated one cell at a time, iteratively proceeding forward.

#### Algorithm Characteristics

- updates only one time slice per iteration
- progresses through time dimension sequentially
- avoids global iterative refinement

#### Key Methods

|Method|Description|
|---|---|
|`update_node(...)`|Updates probability for a single time value|
|`update_row(...)`|Updates a full row for a time slice|
|`update_sota(...)`|Processes all nodes for given time slice|
|`solve(...)`|Runs updates across all time indices|

---
## Example Usage

```python
from stochastic_graph import StochasticGraph
from sota_module import StandardSOTASolver

graph = StochasticGraph(...)
destination = 5
time_budget = 120

solver = StandardSOTASolver(graph, destination, time_budget)
solver.solve()

probabilities = solver.get_sota_matrix()
policy = solver.get_policy_matrix()

path = solver.extract_path(start_node=0)
print("Optimal path:", path)
```

## Output Interpretation

Higher values in **SOTA matrix** mean higher probability of arriving on time, while in **Policy matrix** each entry indicates the successor node yielding maximum probability.
**Path Extraction** function supports both deterministic expectations and stochastic sampled traversal.
