from stochastic_graph import StochasticGraph
from SOTA import StandardSOTASolver, SingleIterationSOTASolver
from preprocessing import Reach

def main():
    graph = StochasticGraph()
    sota = SingleIterationSOTASolver(graph, 8, 10)
    sota.set_destination(6)
    reach = Reach(graph, 10, sota)
    reach.reach_computation()
    reach.reach_pruning()
    sota.solve()
    print(sota.extract_path(1))

if __name__ == "__main__":
    main()