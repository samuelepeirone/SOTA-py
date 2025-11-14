import time
from stochastic_graph import StochasticGraph
from SOTA import StandardSOTASolver, SingleIterationSOTASolver
from preprocessing import detReach, detArcFlags
from deterministic_algorithms import Dijkstra
import pickle
import os
import copy
import numpy as np
from stdnum.util import lcs

class TestFunctions:
    """
    Defining test functions that can be used as a run for all algorithms
    """
    @staticmethod
    def run_standard_sota(graph, destination_node, time_budget, eps=1e-4, max_iter=100, delta_t=None, print_results=True):
        """
        Running Standard SOTA and showing time results.
        
        Return:
        - time
        - policy matrix
        - sota matrix
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

        Return:
        - time
        - policy matrix
        - sota matrix
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

    @staticmethod
    def run_reach(graph, starting_node, destination_node, time_budget, eps=1e-4, max_iter=100, delta_t=None, print_pruned_graph=False, print_paths=False, show_edge_labels=True, show_node_labels=True, print_results=True, print_summary=False, stochastic_sampling=False):
        """
        Running reach pruning and showing time results.

        Returning:
        [TIMES]:
        - reach_time
        - standard_sota_time 
        - single_iteration_sota_time
        [MATRICES]:
        - ss_policy_matrix
        - ss_sota_matrix
        - si_policy_matrix
        - si_policy_matrix
        [PATHS]:
        - ss_sota_path
        - si_sota_path
        """
        adj_matrix = graph.get_adjacency_matrix()
        d = Dijkstra(adj_matrix)
        r = detReach(graph, d)

        start_reach = time.time()
        r.reach_computation()
        pruned = r.reach_pruning(starting_node, destination_node)
        end_reach = time.time()

        reach_time = end_reach-start_reach
        if print_results:
            print(f"Reach computed and pruned in {reach_time:.4f} seconds. Pruned {len(pruned)} nodes: {pruned}")

        if print_pruned_graph:
            print(f"Graph after reach pruning with s={starting_node}, d={destination_node}")
            graph.print_graph(show_edge_labels=show_edge_labels, show_node_labels=show_node_labels)

        if print_results:
            print("\n------- STANDARD SOTA on Reach-pruned graph ---------\n")
        
        # computing successive approximations sota
        standard_sota_time, ss_policy_matrix, ss_sota_matrix = TestFunctions.run_standard_sota(graph, destination_node, time_budget, eps, max_iter, delta_t, print_results=print_results)
        
        # extracting path
        ss_sota_path = StandardSOTASolver.extract_path_from_policy(starting_node, destination_node, ss_policy_matrix, graph, stochastic_sampling=stochastic_sampling)

        if print_results:
            print(f"Path: {ss_sota_path}")

        if print_paths:
            graph.print_graph(path=ss_sota_path, show_edge_labels=show_edge_labels, show_node_labels=show_node_labels)

        if print_results:
            print("\n--- SINGLE ITERATION SOTA on Reach-pruned graph -----\n")
        
        # computing single iteration
        single_iteration_sota_time, si_policy_matrix, si_sota_matrix = TestFunctions.run_single_iteration_sota(graph, destination_node, time_budget, delta_t, print_results=print_results)
        
        # extracting path
        si_sota_path = StandardSOTASolver.extract_path_from_policy(starting_node, destination_node, si_policy_matrix, graph, stochastic_sampling=stochastic_sampling)

        if print_results:
            print(f"Path: {si_sota_path}")

        if print_paths:
            graph.print_graph(path=si_sota_path, show_edge_labels=show_edge_labels, show_node_labels=show_node_labels)

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

        return reach_time, standard_sota_time + reach_time, single_iteration_sota_time + reach_time, ss_policy_matrix, ss_sota_matrix, si_policy_matrix, si_sota_matrix, ss_sota_path, si_sota_path

    @staticmethod
    def run_arcflags(graph, node_d, time_budget, node_s=None, eps=1e-4, max_iter=100, delta_t=None, print_pruned_graph=False, show_edge_labels=True, show_node_labels=True, print_results=True, print_summary=False, print_paths=False):
        """
        Running arcflags pruning and showing time results.

        Return the triple (arcflags_time, standard_sota_time, single_iteration_sota_time)
        """
        adj_matrix = graph.get_adjacency_matrix()
        d = Dijkstra(adj_matrix)
        f = detArcFlags(graph, d, node_d)

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
        standard_sota_time, ss_policy_matrix, ss_sota_matrix = TestFunctions.run_standard_sota(graph, node_d, time_budget, eps, max_iter, delta_t, print_results=print_results)

        if print_results:
            print("\n---- SINGLE ITERATION SOTA on Arcflags-pruned graph ----\n")
        single_iteration_sota_time, si_policy_matrix, si_sota_matrix = TestFunctions.run_single_iteration_sota(graph, node_d, time_budget, delta_t, print_results=print_results)

        af_ss_path = None
        af_si_path = None

        if node_s is not None:
            af_ss_path = StandardSOTASolver.extract_path_from_policy(node_s, node_d, ss_policy_matrix, graph)
            af_si_path = StandardSOTASolver.extract_path_from_policy(node_s, node_d, si_policy_matrix, graph)
            if print_paths:
                print("\nAF ss path")
                f.print_graph_sections(path=af_ss_path, show_edge_labels=show_edge_labels, show_node_labels=show_node_labels)
                print("\nAF si path")
                f.print_graph_sections(path=af_si_path, show_edge_labels=show_edge_labels, show_node_labels=show_node_labels)

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
            
        return af_time, standard_sota_time + af_time, single_iteration_sota_time + af_time, ss_policy_matrix, ss_sota_matrix, si_policy_matrix, si_sota_matrix, af_ss_path, af_si_path

    @staticmethod
    def general_comparison_run(graph, destination_node, time_budget, eps=1e-4, max_iter=100, delta_t=None, print_graph=False, print_all_graphs=False, show_edge_labels=True, show_node_labels=True, print_interm_results=True):
        """
        Printing results for all algorithms
        """
        if print_graph:
            graph.print_graph(show_edge_labels=show_edge_labels, show_node_labels=show_node_labels)

        if print_interm_results:
            print("\n======== STANDARD SOTA ========\n")

        ss, ss_policy, ss_sota = TestFunctions.run_standard_sota(graph, destination_node, time_budget, eps, max_iter, delta_t, print_results=print_interm_results)

        if print_interm_results:
            print("\n==== SINGLE ITERATION SOTA ====\n")

        si, si_policy, si_sota = TestFunctions.run_single_iteration_sota(graph, destination_node, time_budget, delta_t, print_results=print_interm_results)

        if print_interm_results:
            print("\n============ REACH ============\n")

        g_reach = copy.deepcopy(graph)
        _, reach_ss, reach_si, reach_ss_policy, reach_ss_sota, reach_si_policy, reach_si_sota = TestFunctions.run_reach(g_reach, destination_node, time_budget, eps, max_iter, delta_t, print_pruned_graph=print_all_graphs, show_edge_labels=show_edge_labels, show_node_labels=show_node_labels, print_results=print_interm_results)
        
        if print_interm_results:
            print("\n========== ARCFLAGS ===========\n")

        g_arcflags = copy.deepcopy(graph)
        _, af_ss, af_si, af_ss_policy, af_ss_sota, af_si_policy, af_si_sota = TestFunctions.run_arcflags(g_arcflags, destination_node, time_budget, eps, max_iter, delta_t, print_pruned_graph=print_all_graphs, show_edge_labels=show_edge_labels, show_node_labels=show_node_labels, print_results=print_interm_results)

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
        dist_ss_reach = dist_ss_reach / graph.get_num_nodes() # normalizing by the number of nodes in the graph
        dist_si_reach = np.linalg.norm(si_sota[:,-1] - reach_si_sota[:,-1])
        dist_si_reach = dist_si_reach / graph.get_num_nodes()

        dist_ss_af = np.linalg.norm(ss_sota - af_ss_sota, 'fro')
        dist_ss_af = dist_ss_af / graph.get_num_nodes()
        dist_si_af = np.linalg.norm(si_sota - af_si_sota, 'fro')
        dist_si_af = dist_si_af / graph.get_num_nodes()

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

    @staticmethod
    def reach_comparison_run(graph, node_s, node_d, time_budget, eps=1e-4, max_iter=100, delta_t=None, print_all_graphs=False, print_paths=False, show_edge_labels=True, show_node_labels=True, print_results=True, print_interm_results=True, print_summary=False, stochastic_sampling=False):
        """
        Making a comparison run between ss, si and reach-pruned ss and si
        """
        if print_interm_results:
            print("\n======== STANDARD SOTA ========\n")

        ss, ss_policy, ss_sota = TestFunctions.run_standard_sota(graph, node_d, time_budget, eps, max_iter, delta_t, print_results=print_interm_results)

        if print_interm_results:
            print("\n==== SINGLE ITERATION SOTA ====\n")

        si, si_policy, si_sota = TestFunctions.run_single_iteration_sota(graph, node_d, time_budget, delta_t, print_results=print_interm_results)

        if print_interm_results:
            print("\n============ REACH ============\n")

        g_reach = copy.deepcopy(graph)
        _, reach_ss, reach_si, reach_ss_policy, reach_ss_sota, reach_si_policy, reach_si_sota, reach_ss_path, reach_si_path = TestFunctions.run_reach(g_reach, node_s, node_d, time_budget, eps, max_iter, delta_t, print_pruned_graph=print_all_graphs, show_edge_labels=show_edge_labels, show_node_labels=show_node_labels, print_results=print_interm_results, print_paths=print_paths)

        print("\n===============================\n=========== SUMMARY ===========\n===============================\n")
        
        print("\n------------ TIMES ------------\n")

        print(f"ss: {ss:.1f}s\nsi: {si:.1f}s\nreach+ss: {reach_ss:.1f}s;    reach+si: {reach_si:.1f}s\n")

        print("\n----------- SPEEDUP -----------\n")

        print(f"reach+ss: {ss/reach_ss:.2f}x;   reach+si: {si/reach_si:.2f}x\n")

        print("\n---------- DISTANCES ----------")
        print("\n----------- policy  -----------\n")
        print("same elements in policy matrix ratio:")
        pol_dist_ss_reach = np.mean(ss_policy[:,-1] == reach_ss_policy[:,-1])
        pol_dist_si_reach = np.mean(si_policy[:,-1] == reach_si_policy[:,-1])

        print(f"reach+ss: {pol_dist_ss_reach:.3f};    reach+si: {pol_dist_si_reach:.3f}\n")

        print("\n-------- probabilities --------\n")
        print("distances between probabilities matrices:")
        dist_ss_reach = np.linalg.norm(ss_sota[:,-1] - reach_ss_sota[:,-1])
        dist_ss_reach = dist_ss_reach / graph.get_num_nodes() # normalizing by the number of nodes in the graph
        dist_si_reach = np.linalg.norm(si_sota[:,-1] - reach_si_sota[:,-1])
        dist_si_reach = dist_si_reach / graph.get_num_nodes()

        print(f"reach+ss: {dist_ss_reach:.3f};    reach+si: {dist_si_reach:.3f}\n")

        print("\n------------ paths ------------\n")
        print("paths:")
        ss_path = StandardSOTASolver.extract_path_from_policy(node_s, node_d, ss_policy, graph)
        si_path = StandardSOTASolver.extract_path_from_policy(node_s, node_d, si_policy, graph)

        print(f"ss: {ss_path};    reach+ss: {reach_ss_path}\nsi: {si_path};     reach+si: {reach_si_path}")


        path_dist_ss_reach = Utils.lcs_length(reach_ss_path, ss_path) / max(len(reach_ss_path), len(ss_path))   # normalizing by the longest sequence
        path_dist_si_reach = Utils.lcs_length(reach_si_path, si_path) / max(len(reach_si_path), len(si_path))
        print("distances between probabilities paths:\n")
        print(f"reach+ss: {path_dist_ss_reach:.3f};    reach+si: {path_dist_si_reach:.3f}\n")
    
    @staticmethod
    def general_comparison_run_with_paths(graph, node_s, node_d, time_budget, eps=1e-4, max_iter=100, delta_t=None, print_all_graphs=False, print_paths=False, show_edge_labels=True, show_node_labels=True, print_results=True, print_interm_results=True, print_summary=False, stochastic_sampling=False):
        """
        Making a comparison run between ss, si and reach-pruned ss and si
        """
        if print_interm_results:
            print("\n======== STANDARD SOTA ========\n")

        ss, ss_policy, ss_sota = TestFunctions.run_standard_sota(graph, node_d, time_budget, eps, max_iter, delta_t, print_results=print_interm_results)

        if print_interm_results:
            print("\n==== SINGLE ITERATION SOTA ====\n")

        si, si_policy, si_sota = TestFunctions.run_single_iteration_sota(graph, node_d, time_budget, delta_t, print_results=print_interm_results)

        if print_interm_results:
            print("\n============ REACH ============\n")

        g_reach = copy.deepcopy(graph)
        _, reach_ss, reach_si, reach_ss_policy, reach_ss_sota, reach_si_policy, reach_si_sota, reach_ss_path, reach_si_path = TestFunctions.run_reach(
            g_reach, 
            node_s, 
            node_d, 
            time_budget, 
            eps, 
            max_iter, 
            delta_t, 
            print_pruned_graph=print_all_graphs, 
            show_edge_labels=show_edge_labels, 
            show_node_labels=show_node_labels, 
            print_results=print_interm_results, 
            print_paths=print_paths)

        if print_interm_results:
            print("\n========== ARCFLAGS ===========\n")

        g_arcflags = copy.deepcopy(graph)
        _, af_ss, af_si, af_ss_policy, af_ss_sota, af_si_policy, af_si_sota, af_ss_path, af_si_path = TestFunctions.run_arcflags(
            g_arcflags, 
            node_d=node_d,
            time_budget=time_budget,
            eps=eps,
            max_iter=max_iter,
            delta_t=delta_t,
            node_s=node_s,
            print_pruned_graph=print_all_graphs,
            show_edge_labels=show_edge_labels,
            show_node_labels=show_node_labels,
            print_results=print_interm_results,
            print_paths=print_paths)

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
        dist_ss_reach = dist_ss_reach / graph.get_num_nodes() # normalizing by the number of nodes in the graph
        dist_si_reach = np.linalg.norm(si_sota[:,-1] - reach_si_sota[:,-1])
        dist_si_reach = dist_si_reach / graph.get_num_nodes()

        dist_ss_af = np.linalg.norm(ss_sota - af_ss_sota, 'fro')
        dist_ss_af = dist_ss_af / graph.get_num_nodes()
        dist_si_af = np.linalg.norm(si_sota - af_si_sota, 'fro')
        dist_si_af = dist_si_af / graph.get_num_nodes()

        print(f"reach+ss: {dist_ss_reach:.3f};    reach+si: {dist_si_reach:.3f}\naf+ss: {dist_ss_af:.3f};        af+si: {dist_si_af:.3f}")

        print("\n------------ paths ------------\n")
        print("paths:")
        ss_path = StandardSOTASolver.extract_path_from_policy(node_s, node_d, ss_policy, graph)
        si_path = StandardSOTASolver.extract_path_from_policy(node_s, node_d, si_policy, graph)

        # in case we dind't compute it before
        if af_ss_path is None or af_si_path is None:
            af_ss_path = StandardSOTASolver.extract_path_from_policy(node_s, node_d, af_ss_policy, g_arcflags)
            af_si_path = StandardSOTASolver.extract_path_from_policy(node_s, node_d, af_si_policy, g_arcflags)

        print(f"ss: {ss_path};    reach+ss: {reach_ss_path}     af+ss: {af_ss_path}\nsi: {si_path};     reach+si: {reach_si_path}   af+si: {af_si_path}")

        path_dist_ss_reach = Utils.lcs_length(reach_ss_path, ss_path) / max(len(reach_ss_path), len(ss_path))   # normalizing by the longest sequence
        path_dist_si_reach = Utils.lcs_length(reach_si_path, si_path) / max(len(reach_si_path), len(si_path))

        path_dist_ss_af = Utils.lcs_length(af_ss_path, ss_path) / max(len(af_ss_path), len(ss_path))
        path_dist_si_af = Utils.lcs_length(af_si_path, si_path) / max(len(af_si_path), len(si_path))
        print("distances between probabilities paths:\n")
        print(f"reach+ss: {path_dist_ss_reach:.3f};    reach+si: {path_dist_si_reach:.3f}\naf+ss: {path_dist_ss_af:.3f}     af+si: {path_dist_si_af:.3f}")

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
    
    @staticmethod
    def lcs_length(a, b):
        """
        Compute the longest common subsequence
        """
        a = [tuple(x) if isinstance(x, (list, np.ndarray)) else x for x in a]
        b = [tuple(x) if isinstance(x, (list, np.ndarray)) else x for x in b]

        n, m = len(a), len(b)
        dp = [[0]*(m+1) for _ in range(n+1)]

        for i in range(1, n+1):
            for j in range(1, m+1):
                if a[i-1] == b[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        return dp[-1][-1]