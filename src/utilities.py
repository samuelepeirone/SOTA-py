import time
from stochastic_graph import StochasticGraph
from SOTA import StandardSOTASolver, SingleIterationSOTASolver
from preprocessing import detReach, detArcFlags
from deterministic_algorithms import Dijkstra
import pickle
import os

class TestFunctions:
    """
    Defining test functions that can be used as a run for all algorithms
    """
    @staticmethod
    def run_standard_sota_from_graph(graph, destination_node, time_budget, eps=1e-4, max_iter=100, delta_t=None, print_results=True):
        """
        Basic run of Sota solver from a given graph
        """
        s = StandardSOTASolver(graph, destination_node, time_budget)

        # changing discretization variable if needed
        if delta_t is not None:
            s.set_min_edge(delta_t)

        # solving and timing
        start = time.time()
        s.solve(eps=eps, max_iter=max_iter)
        end = time.time()

        # printing results
        if print_results:
            print(f"Policy computed on {end-start:.4f} seconds")

            print("policy last column:")
            print(s.get_policy_matrix()[:, -1])

            print("sota last column:")
            print(s.get_sota_matrix()[:, -1])

        return end-start

    @staticmethod
    def run_standard_sota(matrix, destination_node, time_budget, eps=1e-4, max_iter=100, delta_t=None, print_results=True):
        """
        Running Standard SOTA and showing time results.
        """
        # retrieving matrices and creating graph
        adj_matrix, var_matrix = matrix.get_matrices()
        graph = StochasticGraph(adjacency_matrix=adj_matrix, variance_matrix=var_matrix)

        time = TestFunctions.run_standard_sota_from_graph(graph, destination_node, time_budget, eps, max_iter, delta_t, print_results=print_results)

        return time
    
    @staticmethod
    def run_single_iteration_sota_from_graph(graph, destination_node, time_budget, delta_t=None, print_results=True):
        """
        Basic run of Single Iteration Sota solver from a given graph
        """
        s = SingleIterationSOTASolver(graph, destination_node, time_budget)

        # changing discretization variable if needed
        if delta_t is not None:
            s.set_min_edge(delta_t)

        # solving and timing
        start = time.time()
        s.solve()
        end = time.time()

        # printing results
        if print_results:
            print(f"Policy computed on {end-start:.4f} seconds")

            print("policy last column:")
            print(s.get_policy_matrix()[:, -1])

            print("sota last column:")
            print(s.get_sota_matrix()[:, -1])

        return end-start
    
    @staticmethod
    def run_single_iteration_sota(matrix, destination_node, time_budget, delta_t=None, print_results=True):
        """
        Running Single Iteration SOTA and showing time results.
        """
        adj_matrix, var_matrix = matrix.get_matrices()
        graph = StochasticGraph(adjacency_matrix=adj_matrix, variance_matrix=var_matrix)
        
        time = TestFunctions.run_single_iteration_sota_from_graph(graph, destination_node, time_budget, delta_t, print_results=print_results)

        return time
        

    def run_reach(matrix, destination_node, time_budget, eps=1e-4, max_iter=100, delta_t=None, print_pruned_graph=False, show_edge_labels=True, print_results=True, print_summary=False):
        """
        Running reach pruning and showing time results.

        Return the triple (reach_time, standard_sota_time, single_iteration_sota_time)
        """
        adj_matrix, var_matrix = matrix.get_matrices()
        graph = StochasticGraph(adjacency_matrix=adj_matrix, variance_matrix=var_matrix)
        d = Dijkstra(adj_matrix)
        r = detReach(graph, d, destination_node)

        start_reach = time.time()
        r.reach_computation()
        r.reach_pruning()
        end_reach = time.time()

        reach_time = end_reach-start_reach
        if print_results:
            print(f"Reach computed and pruned in {reach_time:.4f} seconds")

        if print_pruned_graph:
            print("Graph after reach pruning:")
            graph.print_graph(show_edge_labels=show_edge_labels)

        if print_results:
            print("\n------- STANDARD SOTA on Reach-pruned graph ---------\n")
        standard_sota_time = TestFunctions.run_standard_sota_from_graph(graph, destination_node, time_budget, eps, max_iter, delta_t, print_results=print_results)
        
        if print_results:
            print("\n--- SINGLE ITERATION SOTA on Reach-pruned graph -----\n")
        single_iteration_sota_time = TestFunctions.run_single_iteration_sota_from_graph(graph, destination_node, time_budget, delta_t, print_results=print_results)
        
        if print_results:
            print(f"\nStandard SOTA policy computed in {standard_sota_time:.4f} seconds")
            print(f"Standard-pruned query time: {standard_sota_time+reach_time:.4f} seconds")
            print(f"Single iteration policy computed in {single_iteration_sota_time:.4f} seconds")
            print(f"SingleIteration-pruned query time: {single_iteration_sota_time+reach_time:.4f} seconds")
        
        # printing the final summary with times.
        if print_summary:
            print("\n===============================\n=========== SUMMARY ===========\n===============================\n")
            print("\n------------ TIMES ------------\n")

            print(f"reach+ss: {standard_sota_time+reach_time:.2f}s;    reach+si: {single_iteration_sota_time+reach_time:.2f}s")

        return reach_time, standard_sota_time + reach_time, single_iteration_sota_time + reach_time

    @staticmethod
    def run_arcflags(matrix, destination_node, time_budget, eps=1e-4, max_iter=100, delta_t=None, print_pruned_graph=False, show_edge_labels=True, print_results=True, print_summary=False):
        """
        Running arcflags pruning and showing time results.

        Return the triple (arcflags_time, standard_sota_time, single_iteration_sota_time)
        """
        adj_matrix, var_matrix = matrix.get_matrices()
        graph = StochasticGraph(adjacency_matrix=adj_matrix, variance_matrix=var_matrix)
        d = Dijkstra(adj_matrix)
        f = detArcFlags(graph, d, destination_node)

        start_af = time.time()
        f.arcflags_computation()
        f.arcflags_pruning()
        end_af = time.time()

        af_time = end_af-start_af
        if print_results:
            print(f"Arcflags computed and pruned in {af_time:.4f} seconds")

        if print_pruned_graph:
            print("Graph after arcflags pruning:")
            f.print_graph_sections(show_edge_labels=show_edge_labels)

        if print_results:
            print("\n------- STANDARD SOTA on Arcflags-pruned graph ---------\n")
        standard_sota_time = TestFunctions.run_standard_sota_from_graph(graph, destination_node, time_budget, eps, max_iter, delta_t, print_results=print_results)

        if print_results:
            print("\n---- SINGLE ITERATION SOTA on Arcflags-pruned graph ----\n")
        single_iteration_sota_time = TestFunctions.run_single_iteration_sota_from_graph(graph, destination_node, time_budget, delta_t, print_results=print_results)
    
        if print_results:
            print(f"Standard SOTA policy computed in {standard_sota_time:.4f} seconds")
            print(f"Standard-pruned query time: {standard_sota_time + af_time:.4f} seconds")
            print(f"Single iteration policy computed in {single_iteration_sota_time:.4f} seconds")
            print(f"SingleIteration-pruned query time: {single_iteration_sota_time + af_time:.4f} seconds")

        # printing the final summary with times.
        if print_summary:
            print("\n===============================\n=========== SUMMARY ===========\n===============================\n")
            print("\n------------ TIMES ------------\n")

            print(f"af+ss: {standard_sota_time+af_time:.2f}s;    af+si: {single_iteration_sota_time+af_time:.2f}s")
            
        return af_time, standard_sota_time + af_time, single_iteration_sota_time + af_time

    @staticmethod
    def general_comparison_run(matrix, destination_node, time_budget, eps=1e-4, max_iter=100, delta_t=None, print_graph=False, print_all_graphs=False, show_edge_labels=True, print_iterm_results=True):
        """
        Printing results for all algorithms
        """
        if print_graph:
            adj_matrix, var_matrix = matrix.get_matrices()
            graph = StochasticGraph(adjacency_matrix=adj_matrix, variance_matrix=var_matrix)

            graph.print_graph(show_edge_labels=show_edge_labels)

        if print_iterm_results:
            print("\n======== STANDARD SOTA ========\n")

        ss = TestFunctions.run_standard_sota(matrix, destination_node, time_budget, eps, max_iter, delta_t, print_results=print_iterm_results)

        if print_iterm_results:
            print("\n==== SINGLE ITERATION SOTA ====\n")

        si = TestFunctions.run_single_iteration_sota(matrix, destination_node, time_budget, delta_t, print_results=print_iterm_results)

        if print_iterm_results:
            print("\n============ REACH ============\n")

        _, reach_ss, reach_si = TestFunctions.run_reach(matrix, destination_node, time_budget, eps, max_iter, delta_t, print_pruned_graph=print_all_graphs, show_edge_labels=show_edge_labels, print_results=print_iterm_results)
        
        if print_iterm_results:
            print("\n========== ARCFLAGS ===========\n")

        _, af_ss, af_si = TestFunctions.run_arcflags(matrix, destination_node, time_budget, eps, max_iter, delta_t, print_pruned_graph=print_all_graphs, show_edge_labels=show_edge_labels, print_results=print_iterm_results)

        print("\n===============================\n=========== SUMMARY ===========\n===============================\n")
        
        print("\n------------ TIMES ------------\n")

        print(f"ss: {ss:.1f}s\nsi: {si:.2f}s\nreach+ss: {reach_ss:.1f}s;    reach+si: {reach_si:.1f}s\naf+ss: {af_ss:.1f}s;         af+si: {af_si:.1f}s")

        print("\n----------- SPEEDUP -----------\n")

        print(f"reach+ss: {ss/reach_ss:.2f}x;   reach+si: {si/reach_si:.2f}x\naf+ss: {ss/af_ss:.2f}x;        af+si: {si/af_si:.2f}x")

        print("\a")

class Utils:
    @staticmethod
    def save_object(obj, filepath):
        """
        Save a Python object to a file using pickle.
        """
        dirpath = os.path.dirname(filepath)
        if dirpath !="" and not os.path.exists(dirpath):
            raise FileNotFoundError(f"The directory {dirpath} does not exists.")

        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)

        print(f"Object saved to {filepath}")

    @staticmethod
    def load_object(filepath):
        """
        Load a Python object from a pickle file.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"The file {filepath} does not exists.")

        with open(filepath, 'rb') as f:
            obj = pickle.load(f)

        print(f"Object loaded from {filepath}")
        
        return obj