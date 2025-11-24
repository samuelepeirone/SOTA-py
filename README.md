# SOTA solver - documentation

This repository provides a modular implementation of the Stochastic On-Time Arrival (SOTA) problem, supporting:
- Gamma-distributed stochastic travel times
- Dynamic programming-based SOTA solver
- Single Iteration approximation
- Deterministic Dijkstra routing
- Preprocessing acceleration techniques:
	- Reach-based pruning
	- Arc-Flags pruning
The framework allows benchmarking, comparison, pruning effectiveness evaluation and policy/path extraction.

## 📁 Project Structure

```python
📁 SOTA solver
	📁 graph
	   🔹 Grid_network_and_Gamma_distribution.py
		   🔸Matrix()
	📁 docs
	📁 notebook
		📁 instances/graphs
		📁 pruning techniques-results
			📁 graphs
			📁 tables
	📁 src
	   🔹 deterministic_algorithms.py
		   🔸Dijkstra
	   🔹 main.py
	   🔹 preprocessing.py
		   🔸Reach(ABC)
		   🔸bfReach
		   🔸detReach
		   🔸bfArcFlags
	   🔹 SOTA.py
		   🔸SOTA(ABC)
		   🔸StandardSOTASolver(SOTA)
		   🔸SingleIterationSOTASolver(SOTA)
	   🔹 stochastic_graph.py
		   🔸StochasticGraph
	   🔹 utilities.py
		   🔸TestFunctions
		   🔸Utils
```

**All modules' full documentation can be found in `docs` folder**:

`src`:
- [Deterministic Algorithms Module Documentation](./docs/Deterministic%20Algorithms%20Module%20Documentation.md)
- [Precomputation Module Documentation](./docs/Precomputation%20Module%20Documentation.md)
- [SOTA Module Documentation](./docs/SOTA%20Module%20Documentation.md)
- [StochasticGraph Module Documentation](./docs/StochasticGraph%20Module%20Documentation.md)
- [Utilities Module Documentation](./docs/Utilities%20Module%20Documentation.md)

`graph`:
- [Grid Network Module Documentation](./docs/Grid%20Network%20Module%20Documentation.md)

## 🖥️ Requirements

Recommended packages:

```python
numpy
networkx
scipy
math
abc
```

---
## 📄 Basic Usage Example

### 1) Build a grid graph with stochastic travel times

```python
from graph.Grid_network_and_Gamma_distribution import Matrix

matrix = Matrix(line_numbers=5, column_numbers=5)
mean_matrix, var_matrix = matrix.compute_matrices()
```

### 2) Construct a stochastic graph

```python
from src.stochastic_graph import StochasticGraph

graph = StochasticGraph(mean_matrix, var_matrix)
```

### 3) Run Standard SOTA

```python
from src.SOTA import StandardSOTASolver

solver = StandardSOTASolver(graph, node_d=10, time_budget=50)
solver.solve()

policy = solver.get_policy_matrix()
path = solver.extract_path(start_node=0)
```

### 4) Run preprocessing + accelerated SOTA

```python
from src.preprocessing import detReach
from src.deterministic_algorithms import Dijkstra

dijkstra = Dijkstra(graph.get_adjacency_matrix())
reach = detReach(graph, dijkstra)

reach_values = reach.reach_computation()
pruned_graph = reach.reach_pruning(source=0, destination=10)

# accelerated SOTA

solver = StandardSOTASolver(pruned_graph, node_d=10, time_budget=50)
solver.solve()

policy = solver.get_policy_matrix()
path = solver.extract_path(start_node=0)
```

---
## 📓 Notebooks and Experimental Results

The repository includes a `notebook` directory containing Jupyter notebooks used for running, visualizing and analyzing experimental results. Specifically, this folder includes:
- Executable notebooks demonstrating:
	- SOTA solver execution
	- visualization of probability curves
	- comparison between Standard SOTA, Single Iteration SOTA, Reach and Arc-Flags
	- performance plots
- Experiment runs performed with varying conditions:
	- multiple time budgets
	- multiple graph dimensions
- Pickle files storing graph instances, allowing:
	- reproducibility of experiments
	- consinstent comparison across algorithms
	- loading pre-generated stochastic graids without re-generating them
- CSV tables containing aggregated benchmark results, including:
	- execution times
	- speedups
These assets make possible to replicate the evaluation process and extend the analysis withut rerunning all solver executions

## 📊 Benchmarking & Comparison

The repository includes `TestFunctions`, that allow:
- timing execution
- comparing policy matrices
- comparing probability distributions
- path similarity scoring (LCS)
- evaluating pruning speedups