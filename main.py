from stochastic_graph import StochasticGraph
from SOTA import StandardSOTASolver, SingleIterationSOTASolver

def main():
    graph = StochasticGraph()
    sota = SingleIterationSOTASolver(graph, 2, 8, 10)
    sota.solve()
    sota.print_sota_matrix()
    sota.print_policy_matrix()
    print("Optimal path (minimal extraction):", sota.extract_path())

if __name__ == "__main__":
    main()