import numpy as np
import time
from sklearn.cluster import KMeans
from abc import ABC, abstractmethod

from sklearn.preprocessing import MinMaxScaler

class Reach(ABC):
    """
    Metric that quantifies the radius of node's relevance.
    A node will have a small reach value if it only belong to shortest paths
    whose sources or destinations are close to the node; large reach for shortest 
    paths involving distante sources and destinations.
    """
    def __init__(self, graph, node_s, node_d):
        self.graph = graph
        self.num_nodes = graph.get_num_nodes()
        self.min_edge = graph.get_min_edge()
        # array of reach values; initializing all reach values to zero
        self.reach_values = np.zeros(self.num_nodes)
        # m_id and m_si cache, as they are used multiple times
        self.m_id_cache = {}    # key: dest_node, value: array m(id)
        self.m_si_cache = {}    # key: (s,i), value: m(s,i)
        # source and destination nodes -> to protect them from pruning
        self.node_d = node_d
        self.node_s = node_s
        self.visited_nodes = {} # dictionary

    def get_reach_values(self):
        return self.reach_values

    def print_reach_values(self):
        print("Reach values for each node:")
        for i, reach in enumerate(self.reach_values):
            print(f"Node {i}: Reach = {reach:.2f}")
    
    def m(self, node_i, node_j):
        """
        Computing the m metric as average travel time from node_i to node_j.
        Returns np.inf if there is no link between i and j.
        """
        m = self.graph.get_adjacency_matrix_value(node_i, node_j)

        if m == 0:
            m = np.inf

        return m

    @abstractmethod
    def reach_computation(self):
        pass

    @abstractmethod
    def reach_pruning(self):
        """
        Pruning if r(i,T) < min(m(s,i), m(i,d)). We use the cached values to avoid
        recomputing m(s,i) and m(i,d), as we computed them in reach_computation function.
        """
        print("Pruning nodes...")

        pruned_nodes = set()

        nodes = self.graph.get_nodes()
        
        for d in nodes:
            # retrieving m_id for all nodes i
            m_id = self.m_id_cache.get(d)
            # if not cached, compute it
            if m_id is None:
                m_id = [self.m(i, d) for i in nodes]
                self.m_id_cache[d] = m_id

            for s in nodes:
                if s == d:
                    continue

                visited_nodes = self.visited_nodes.get((s, d), [])
                #print(visited_nodes)
                
                for i in nodes:
                    # don't prune the start or destination node
                    if i == self.node_d or i == self.node_s:
                        continue

                    if i == s or i == d:
                        continue

                    if (s, i) in self.m_si_cache:
                        m_si = self.m_si_cache[(s, i)]
                    else:
                        m_si = self.m(s, i)
                        self.m_si_cache[(s, i)] = m_si

                    #print(f"RV: {self.reach_values[i]}, SI: {m_si}, ID: {m_id[i]}")
                    
                    if np.isinf(m_si) or np.isinf(m_id[i]):
                        continue

                    if self.reach_values[i] < min(m_si, m_id[i]):
                        self.graph.prune_node(i)
                        #print(f"Node {i} pruned")
                        pruned_nodes.add(i)
        
        print(f"Pruned nodes: {pruned_nodes}")
        return pruned_nodes

    def reach_test(self):
        print(self.visited_nodes)

class bfReach(Reach):
    """
    Brute force approach to compute Reach values
    """
    def __init__(self, graph, SOTASolver, node_s=None):
        super().__init__(graph, node_s, node_d=SOTASolver.get_destination())
        self.SOTASolver = SOTASolver
        self.num_cols = SOTASolver.get_num_cols()
        self.time_budget = SOTASolver.get_time_budget()

    def get_optimal_path_nodes(self, start_node, dest_node):
        """
        Returning the set of nodes that are in at least one optimal path from 
        start_node to dest_node for times <= T.
        Uses SOTA functions.
        """
        visited = set()

        for t in range(0, self.num_cols-1):
            path = self.SOTASolver.extract_path_from_time(start_node, t)
            # if the path exists and 
            if path and path[-1] == dest_node:
                visited.update(path)
        
        return visited

    def reach_computation(self):
        """
        Computes the reach values by running a SOTA search for all possible destinations in the graph.
        """
        print("Computing reach values...")
        # reset reach values and caches
        self.reach_values = np.zeros(self.num_nodes)
        self.m_id_cache.clear()
        self.m_si_cache.clear()

        for d in range(self.num_nodes):
            # d is the destination node
            self.SOTASolver.set_destination(d)
            self.SOTASolver.solve()

            # pre-computing m(i,d) for all nodes i
            m_id = [self.m(i, d) for i in range(self.num_nodes)]
            self.m_id_cache[d] = m_id

            # iterating over all possible source nodes
            for s in range(self.num_nodes):
                if s == d:
                    continue
                
                # getting the set of nodes in optimal paths from s to d
                visited_nodes = self.get_optimal_path_nodes(s, d)
                self.visited_nodes[(s, d)] = visited_nodes

                # updating the reach values for each visited node
                for i in visited_nodes:
                    if (s, i) in self.m_si_cache:
                        m_si = self.m_si_cache[(s, i)]
                    else:
                        m_si = self.m(s, i)
                        self.m_si_cache[(s, i)] = m_si
                    
                    # if a path doesn't exist, the reach is not updated
                    if np.isinf(m_si) or np.isinf(m_id[i]):
                        continue

                    self.reach_values[i] = max(self.reach_values[i], min(m_si, m_id[i]))
        
        print("Reach values computed")
        return self.reach_values

    def reach_pruning(self):
        return super().reach_pruning()

class detReach(Reach):
    def __init__(self, graph, DetAlgorithm, node_d, node_s=None):
        super().__init__(graph, node_s, node_d)
        self.DetAlgorithm = DetAlgorithm
        self.node_d = node_d

    def reach_computation(self):
        """
        Computing reach values using a deterministic algorithm
        """
        print("Computing reach values...")
        # reset reach values and caches
        self.reach_values = np.zeros(self.num_nodes)
        self.m_id_cache.clear()
        self.m_si_cache.clear()

        nodes = self.graph.get_nodes()

        for s in nodes:
            # computing optimal predecessors and distances from source s to all destinations s
            pred_s = self.DetAlgorithm.compute_path(s)

            m_si = [self.m(s,i) for i in nodes]
            self.m_si_cache[s] = m_si

            for d in nodes:
                if s == d:
                    continue
                    
                # getting the set of nodes in optimal paths from s to d
                opt_paths = self.DetAlgorithm.get_all_optimal_paths(pred_s, d)
                visited_nodes = self.DetAlgorithm.get_nodes_from_optimal_paths(opt_paths)
                
                self.visited_nodes[(s, d)] = visited_nodes

                # updating the reach values for each visited node
                for i in visited_nodes:
                    if (i, d) in self.m_id_cache:
                        m_id = self.m_id_cache[(i, d)]
                    else:
                        m_id = self.m(i, d)
                        self.m_id_cache[(i, d)] = m_id
                    
                    # if a path doesn't exist, the reach is not updated
                    if np.isinf(m_id) or np.isinf(m_si[i]):
                        continue

                    self.reach_values[i] = max(self.reach_values[i], min(m_si[i], m_id))

        print("Reach values computed")
        return self.reach_values

    def reach_pruning(self):
        return super().reach_pruning()

class ArcFlags(ABC):
    def __init__(self, graph, node_s, node_d):
        self.graph = graph
        self.regions = None
        self.arc_flags = None
        self.node_s = node_s
        self.node_d = node_d
        self.num_regions = None

    def print_arcflags(self):
        print("Arc-Flags for each edge:")
        for edge, flags in self.arc_flags.items():
            print(f"Edge {edge}: {flags}")
    
    def print_graph_sections(self, path=None):
        self.graph.print_graph_sections(self.regions, path)

    def initialize_arcflags(self):
        self.arc_flags = {e: {r: False for r in range(self.num_regions)} for e in self.graph.get_edges()}
        return

    def canopy_clustering(self, X, T1=0.8, T2=0.4):
        """
        Canopy clustering algorithm

        @param X:   feature matrix
        @param T1:  a point within this distance from canopy center will be included in the canopy
        @param T2:  a point within this distance from canopy center will be removed from candidates pool
        """
        unassigned = set(self.graph.get_nodes())
        canopies = []
        centers = []

        while unassigned:
            # choose casually a point as center of the new canopy
            center_idx = unassigned.pop()
            center = X[center_idx]
            current_canopy = [center_idx]   # initializing the current canopy
            to_remove = set()

            # find all points in range T1
            for i in unassigned:
                # euclidean distance betweeen center and unassigned point
                d = np.linalg.norm(center - X[i])
                if d < T1:
                    current_canopy.append(i)
                if d < T2:
                    to_remove.add(i)
            
            # updating the sets
            unassigned -= to_remove

            # saving canopy and center
            canopies.append(current_canopy)
            centers.append(center_idx)
        
        return canopies, centers

    def partition_graph(self, **kwargs):
        """
        Partitioning the graph using the Canopy Kmeans algorithm.
        """
        # building the feature vector x_i = [mu_i, sigma_i]
        mean_edges = np.mean(self.graph.adjacency_matrix, axis=1)
        var_edges = np.mean(self.graph.variance_matrix, axis=1)

        X = np.column_stack([mean_edges, var_edges])

        # bounding the values between 0 and 1
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        canopies, centers = self.canopy_clustering(X_scaled, **kwargs)

        self.num_regions = len(centers)

        # retrieving the feature values of the centroids found with canopy clustering
        initial_centroids = X_scaled[centers]

        # using kmeans with canopy-found centroids
        kmeans = KMeans(n_clusters=len(centers),
                        init=initial_centroids,
                        n_init=1,
                        random_state=42)
        
        kmeans.fit(X_scaled)

        labels = kmeans.labels_

        self.regions = {i: int(labels[i]) for i in range(self.graph.get_num_nodes())}

        return self.regions

    @abstractmethod
    def arcflags_computation(self):
        pass

    @abstractmethod
    def arcflags_pruning(self):
        """
        Prunes from the graph edges whose flag is False for the destination-node's region.
        """
        print("Pruning edges...")

        region_d = self.regions.get(self.node_d)
        
        pruned_edges = 0

        for (u, v) in self.graph.get_edges():
            # pruning condition
            if not self.arc_flags[(u,v)][region_d]:
                self.graph.prune_edge(u,v)
                pruned_edges += 1
        
        print(f"[ArcFlags] Pruned {pruned_edges} edges for destination {self.node_d} (region {region_d}).")

class bfArcFlags(ArcFlags):
    def __init__(self, graph, SOTASolver, node_s=None):
        super().__init__(graph, node_s, node_d=SOTASolver.get_destination())
        self.SOTASolver = SOTASolver
        self.time_budget = SOTASolver.get_time_budget()

    def compute_optimal_policy(self, dest_node):
        """
        Uses SOTA solver to compute the optimal policy to the given destination node.
        returns a list of arcs (i,j) that are part of the optimal policy to dest_node from a given node.
        """
        self.SOTASolver.set_destination(dest_node)
        self.SOTASolver.solve()

        policy = self.SOTASolver.get_policy_matrix()

        if np.all(policy[:, -1] == -1):
            print("[DEBUG] Last column of policy matrix only contains -1, no path possible.")
            self.SOTASolver.print_policy_matrix()

        # initializing the set of arcs
        paths = {}

        for s in range(self.graph.get_num_nodes()):
            if s == dest_node:
                continue
            
            path = self.SOTASolver.extract_path(s)

            nodes = list(path)

            if len(nodes) < 2:
                paths[s] = []
            else:
                edges = [(nodes[i], nodes[i+1]) for i in range(len(nodes)-1)]
                paths[s] = edges
        
        return paths

    def collect_relevant_edges(self, dest_node):
        """
        Returns the set of edges that are part of an optimal 
        path to the destination node.
        """
        paths = self.compute_optimal_policy(dest_node)

        relevant_edges = set()

        for _, path_edges in paths.items():
            for e in path_edges:    # for each edge in the optimal path
                relevant_edges.add(e)
        
        return relevant_edges
    
    def arcflags_computation(self):
        """
        Computes the arc-flags for all edges and all regions with brute force approach.
        """
        print("Executing partition...")
        start = time.time()
        self.partition_graph()
        end = time.time()
        print(f"Partitioning of the graph in {self.num_regions} regions executed in {end-start:.4f} seconds!")

        print("Computing arcflags...")
        
        self.initialize_arcflags()

        for d in range(self.graph.get_num_nodes()):
            region_d = self.regions[d]
            relevant_edges = self.collect_relevant_edges(d)
            
            if len(relevant_edges) == 0:
                print(f"[WARN] No relevant edges found for destination {d}, region {region_d}")

            for e in relevant_edges:
                self.arc_flags[e][region_d] = True
        
        print("Arcflags computed!")

        return self.arc_flags

    def arcflags_pruning(self):
        return super().arcflags_pruning()

class detArcFlags(ArcFlags):
    def __init__(self, graph, DetAlgorithm, node_d, node_s = None):
        super().__init__(graph, node_s, node_d)
        self.DetAlgorithm = DetAlgorithm

    def arcflags_computation(self):
        """
        Computing arcflags using deterministic algorithm.
        """
        print("Executing partition...")
        start = time.time()
        self.partition_graph()
        end = time.time()
        print(f"Partitioning of the graph in {self.num_regions} regions executed in {end-start:.4f} seconds!")

        print("Computing arcflags...")

        self.initialize_arcflags()

        nodes = self.graph.get_nodes()

        for s in nodes:
            pred_s = self.DetAlgorithm.compute_path(s)

            for d in nodes:
                if s == d:
                    continue
                    
                opt_paths = self.DetAlgorithm.get_all_optimal_paths(pred_s, d)
                visited_edges = self.DetAlgorithm.get_edges_from_optimal_paths(opt_paths)

                if len(visited_edges) == 0:
                    print(f"[WARN] No relevant edges found for destination {d}, region {region_d}")

                region_d = self.regions[d]

                for e in visited_edges:
                    self.arc_flags[e][region_d] = True

        print("Arcflags computed!")

        return self.arc_flags           

    def arcflags_pruning(self):
        return super().arcflags_pruning()