from stochastic_graph import StochasticGraph
from SOTA import StandardSOTASolver, SingleIterationSOTASolver
from preprocessing import Reach, ArcFlags

import warnings
warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak")


def main():
    graph = StochasticGraph()
    sota = SingleIterationSOTASolver(graph, 8, 10)
    reach = Reach(graph, sota)
    reach.reach_computation()
    reach.reach_pruning()

if __name__ == "__main__":
    main()