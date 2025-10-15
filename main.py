import os
import sys

sys.path.append(os.path.abspath("./src"))

from stochastic_graph import StochasticGraph
from preprocessing import bfReach, detReach, bfArcFlags, detArcFlags
from SOTA import StandardSOTASolver, SingleIterationSOTASolver
from deterministic_algorithms import Dijkstra

import warnings
warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak")

sys.path.append(os.path.abspath("./graph"))

from Grid_network_and_Gamma_distribution import Matrix

def main():
    matrix = Matrix(5, 5, link_var_max=0.5)
    adj_matrix, var_matrix = matrix.compute_matrices()

    graph = StochasticGraph(adj_matrix, var_matrix)
    d = Dijkstra(adj_matrix)
    r = detReach(graph, d, 2, 19)
    r.reach_computation()
    r.print_reach_values()
    r.reach_pruning()
    graph.print_graph()

if __name__ == "__main__":
    main()