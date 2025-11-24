# Utilities Module Documentation

## Overview

This module defines a collection of benchmarking and comparison utilities for evaluating different routin and stochastic on-time arrival (SOTA) algorithms on stochastic graphs. It provides:
- execution wrappers for:
	- Standard SOTA
	- Single-Iteration SOTA
	- Reach pruning
	- ArcFlags pruning
- unified timing and metrics reporting
- policy and probability matrix comparisons
- extracted path comparison

---
## `class TestFunctions`

A static utility class providing ready-to-run experimental procedures for SOTA-based routing methods.

| Method                                   | Description                                                                                                                                                                                                               |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `run_standard_sota(...)`                 | Runs the Standard SOTA solver and measures execution time                                                                                                                                                                 |
| `run_single_iteration_sota(...)`         | Runs the Single-Iteration SOTA solver                                                                                                                                                                                     |
| `run_reach(...)`                         | Runs deterministic Reach preprocessing + pruned SOTA executions                                                                                                                                                           |
| `run_arcflags(...)`                      | Runs deterministic ArcFlags preprocessing + pruned SOTA executions                                                                                                                                                        |
| `reach_comparison_run(...)`              | Comparison version involving:<br>- Standard SOTA<br>- Single Iteration SOTA<br>- Reach-pruned version<br>Includes:<br>- timing<br>- speedup<br>- policy similarity<br>- probability distance<br>- path similarity         |
| `single_path_comparison_run(...)`        | Extracts and prints for two solvers from the same starting node                                                                                                                                                           |
| `general_comparison_run_with_paths(...)` | Runs all four:<br>- Standard SOTA<br>- Single Iteration SOTA<br>- Reach-pruned SOTA<br>- ArcFlags-pruned SOTA<br>and prints:<br>- execution times<br>- speedups<br>- matrix similarity metrics<br>- path distance metrics |

---
## `class Utils`

| Method             | Description                                                                |
| ------------------ | -------------------------------------------------------------------------- |
| `save_object(...)` | Pickles an object, verifying directory existence                           |
| `load_object(...)` | Loads a pickled object, verifying file existence                           |
| `lcs_length(a, b)` | Computes the longest common subsequence lenght between two node sequences. |
