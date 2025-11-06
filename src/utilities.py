import time
from stochastic_graph import StochasticGraph
from SOTA import StandardSOTASolver, SingleIterationSOTASolver
from preprocessing import detReach, detArcFlags
from deterministic_algorithms import Dijkstra
import pickle
import os
import copy
import numpy as np

class TestFunctions:
    """
    Defining test functions that can be used as a run for all algorithms
    """
    @staticmethod
    def run_standard_sota(graph, destination_node, time_budget, eps=1e-4, max_iter=100, delta_t=None, print_results=True):
        """
        Running Standard SOTA and showing time results.
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

        return end-start, s.get_policy_matrix(), s.get_sota_matrix()
    
    @staticmethod
    def run_single_iteration_sota(graph, destination_node, time_budget, delta_t=None, print_results=True):
        """
        Running Single Iteration SOTA and showing time results.
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

        return end-start, s.get_policy_matrix(), s.get_sota_matrix()
        

    def run_reach(graph, destination_node, time_budget, eps=1e-4, max_iter=100, delta_t=None, print_pruned_graph=False, show_edge_labels=True, show_node_labels=True, print_results=True, print_summary=False):
        """
        Running reach pruning and showing time results.

        Return the triple (reach_time, standard_sota_time, single_iteration_sota_time)
        """
        adj_matrix = graph.get_adjacency_matrix()
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
            graph.print_graph(show_edge_labels=show_edge_labels, show_node_labels=show_node_labels)

        if print_results:
            print("\n------- STANDARD SOTA on Reach-pruned graph ---------\n")
        standard_sota_time, ss_policy_matrix, ss_sota_matrix = TestFunctions.run_standard_sota(graph, destination_node, time_budget, eps, max_iter, delta_t, print_results=print_results)
        
        if print_results:
            print("\n--- SINGLE ITERATION SOTA on Reach-pruned graph -----\n")
        single_iteration_sota_time, si_policy_matrix, si_sota_matrix = TestFunctions.run_single_iteration_sota(graph, destination_node, time_budget, delta_t, print_results=print_results)
        
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

        return reach_time, standard_sota_time + reach_time, single_iteration_sota_time + reach_time, ss_policy_matrix, ss_sota_matrix, si_policy_matrix, si_sota_matrix

    @staticmethod
    def run_arcflags(graph, destination_node, time_budget, eps=1e-4, max_iter=100, delta_t=None, print_pruned_graph=False, show_edge_labels=True, show_node_labels=True, print_results=True, print_summary=False):
        """
        Running arcflags pruning and showing time results.

        Return the triple (arcflags_time, standard_sota_time, single_iteration_sota_time)
        """
        adj_matrix = graph.get_adjacency_matrix()
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
            f.print_graph_sections(show_edge_labels=show_edge_labels, show_node_labels=show_node_labels)

        if print_results:
            print("\n------- STANDARD SOTA on Arcflags-pruned graph ---------\n")
        standard_sota_time, ss_policy_matrix, ss_sota_matrix = TestFunctions.run_standard_sota(graph, destination_node, time_budget, eps, max_iter, delta_t, print_results=print_results)

        if print_results:
            print("\n---- SINGLE ITERATION SOTA on Arcflags-pruned graph ----\n")
        single_iteration_sota_time, si_policy_matrix, si_sota_matrix = TestFunctions.run_single_iteration_sota(graph, destination_node, time_budget, delta_t, print_results=print_results)
    
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
            
        return af_time, standard_sota_time + af_time, single_iteration_sota_time + af_time, ss_policy_matrix, ss_sota_matrix, si_policy_matrix, si_sota_matrix

    @staticmethod
    def general_comparison_run(graph, destination_node, time_budget, eps=1e-4, max_iter=100, delta_t=None, print_graph=False, print_all_graphs=False, show_edge_labels=True, show_node_labels=True, print_iterm_results=True):
        """
        Printing results for all algorithms
        """
        if print_graph:
            graph.print_graph(show_edge_labels=show_edge_labels, show_node_labels=show_node_labels)

        if print_iterm_results:
            print("\n======== STANDARD SOTA ========\n")

        ss, ss_policy, ss_sota = TestFunctions.run_standard_sota(graph, destination_node, time_budget, eps, max_iter, delta_t, print_results=print_iterm_results)

        if print_iterm_results:
            print("\n==== SINGLE ITERATION SOTA ====\n")

        si, si_policy, si_sota = TestFunctions.run_single_iteration_sota(graph, destination_node, time_budget, delta_t, print_results=print_iterm_results)

        if print_iterm_results:
            print("\n============ REACH ============\n")

        g_reach = copy.deepcopy(graph)
        _, reach_ss, reach_si, reach_ss_policy, reach_ss_sota, reach_si_policy, reach_si_sota = TestFunctions.run_reach(g_reach, destination_node, time_budget, eps, max_iter, delta_t, print_pruned_graph=print_all_graphs, show_edge_labels=show_edge_labels, show_node_labels=show_node_labels, print_results=print_iterm_results)
        
        if print_iterm_results:
            print("\n========== ARCFLAGS ===========\n")

        g_arcflags = copy.deepcopy(graph)
        _, af_ss, af_si, af_ss_policy, af_ss_sota, af_si_policy, af_si_sota = TestFunctions.run_arcflags(g_arcflags, destination_node, time_budget, eps, max_iter, delta_t, print_pruned_graph=print_all_graphs, show_edge_labels=show_edge_labels, show_node_labels=show_node_labels, print_results=print_iterm_results)

        print("\n===============================\n=========== SUMMARY ===========\n===============================\n")
        
        print("\n------------ TIMES ------------\n")

        print(f"ss: {ss:.1f}s\nsi: {si:.1f}s\nreach+ss: {reach_ss:.1f}s;    reach+si: {reach_si:.1f}s\naf+ss: {af_ss:.1f}s;         af+si: {af_si:.1f}s")

        print("\n----------- SPEEDUP -----------\n")

        print(f"reach+ss: {ss/reach_ss:.2f}x;   reach+si: {si/reach_si:.2f}x\naf+ss: {ss/af_ss:.2f}x;        af+si: {si/af_si:.2f}x")

        print("\n---------- DISTANCES ----------")
        print("\n----------- policy  -----------\n")
        print("same elements in policy matrix ratio:")
        pol_dist_ss_reach = np.mean(ss_policy[:,-1] == reach_ss_policy[:,-1])
        pol_dist_si_reach = np.mean(si_policy[:,-1] == reach_si_policy[:,-1])

        pol_dist_ss_af = np.mean(ss_policy == af_ss_policy)
        pol_dist_si_af = np.mean(si_policy == af_si_policy)

        print(f"reach+ss: {pol_dist_ss_reach:.3f};    reach+si: {pol_dist_si_reach:.3f}\naf+ss: {pol_dist_ss_af:.3f};        af+si: {pol_dist_si_af:.3f}")

        print("\n-------- probabilities --------\n")
        print("distances between probabilities matrices:")
        dist_ss_reach = np.linalg.norm(ss_sota[:,-1] - reach_ss_sota[:,-1])
        dist_si_reach = np.linalg.norm(si_sota[:,-1] - reach_si_sota[:,-1])

        dist_ss_af = np.linalg.norm(ss_sota - af_ss_sota, 'fro')
        dist_si_af = np.linalg.norm(si_sota - af_si_sota, 'fro')

        print(f"reach+ss: {dist_ss_reach:.3f};    reach+si: {dist_si_reach:.3f}\naf+ss: {dist_ss_af:.3f};        af+si: {dist_si_af:.3f}")

        print("\a")
    
    @staticmethod
    def arcflags_ss(graph, destination_node, time_budget):
        d = Dijkstra(graph.get_adjacency_matrix())
        afsi = detArcFlags(graph, d, destination_node, time_budget)
        afsi.arcflags_computation()
        afsi.arcflags_pruning()

        si = StandardSOTASolver(graph, destination_node, time_budget)
        si.solve()
        return si, graph

    @staticmethod
    def arcflags_si(graph, destination_node, time_budget):
        d = Dijkstra(graph.get_adjacency_matrix())
        afsi = detArcFlags(graph, d, destination_node, time_budget)
        afsi.arcflags_computation()
        afsi.arcflags_pruning()

        si = SingleIterationSOTASolver(graph, destination_node, time_budget)
        si.solve()
        return si, graph
    
    @staticmethod
    def single_path_comparison_run(graph1, graph2, solver1, solver2, starting_node, stochastic_sampling=False, print_graph=True, show_edge_labels=True, show_node_labels=True):
        """
        given two solvers, it will compare the paths found by the two of them
        """
        print(f"====================================\n======== PATH FROM {starting_node} to {solver1.get_destination()} ========\n====================================\n")

        path1 = solver1.extract_path(starting_node, stochastic_sampling = stochastic_sampling)
        print("\n======== SOLVER 1 PATH ========\n")
        print(f"path: {path1}")

        if print_graph:
            graph1.print_graph(path = path1, show_edge_labels=show_edge_labels, show_node_labels=show_node_labels)

        path2 = solver2.extract_path(starting_node, stochastic_sampling=stochastic_sampling)
        print("\n======== SOLVER 2 PATH ========\n")
        print(f"path: {path2}")
        
        if print_graph:
            graph2.print_graph(path = path2, show_edge_labels=show_edge_labels, show_node_labels=show_node_labels)

    @staticmethod
    def paths_comparison_run(graph1, graph2, solver1, solver2, starting_nodes, stochastic_sampling=False, print_graph=True, show_edge_labels=True, show_node_labels=True):
        """
        Paths comparison run for different starting nodes
        """
        for starting_node in starting_nodes:
            TestFunctions.single_path_comparison_run(graph1, graph2, solver1, solver2, starting_node, stochastic_sampling=stochastic_sampling, print_graph=print_graph, show_edge_labels=show_edge_labels, show_node_labels=show_node_labels)

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