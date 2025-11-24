# Grid Network Module Documentation

## `class Matrix`

### Overview

The `Matrix` class generates:
- an adjacency matrix representing a rectangular grid graph
- a mean travel-time matrix assigned to existing links
- a variance matrix associated with stochastic link travel times

The grid is defined by:
- `line_numbers`: number of rows
- `column_numbers`: number of columns

Each cell represents a node, and connections exist only between adjacency grid cells (up, down, left, right)

### Parameters

| Parameter        | Type  | Meaning                      |
| ---------------- | ----- | ---------------------------- |
| `line_numbers`   | int   | number of grid rows          |
| `column_numbers` | int   | number of grid columns       |
| `link_mean_min`  | float | minimum travel time mean     |
| `link_mean_max`  | float | maximum travel time mean     |
| `link_var_min`   | float | minimum travel time variance |
| `link_var_max`   | float | maximum travel time variance |

### Functions

`compute_matrices()`: generates all required metrices in sequence:
1. `build_adj()`
2. `build_link_mean()`
3. `build_link_var()`
4. returns mean + variance matrices
Returns: `(mean_matrix, variance_matrix)`

## Example Usage

```python
from matrix import Matrix

# create a 5x5 grid
M = Matrix(5, 5)

# generate all matrices
mean_matrix, var_matrix = M.compute_matrices()

print("Adjacency:\n", M.get_adjmatrix())
print("Mean matrix:\n", mean_matrix)
print("Variance matrix:\n", var_matrix)
```
