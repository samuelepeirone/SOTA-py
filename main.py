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
    matrix = Matrix(3, 3)
    adj_matrix, var_matrix = matrix.compute_matrices()

    graph = StochasticGraph(adj_matrix, var_matrix)
    s = SingleIterationSOTASolver(graph, 4, 20)
    s.solve()
    print(s.get_sota_matrix()[:, -1])
    graph.print_graph()

if __name__ == "__main__":
    main()