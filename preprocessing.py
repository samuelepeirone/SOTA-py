import numpy as np
from sklearn.cluster import KMeans

class Reach:
    """
    Metric that quantifies the radius of node's relevance.
    A node will have a small reach value if it only belong to shortest paths
    whose sources or destinations are close to the node; large reach for shortest 
    paths involving distante sources and destinations.
    """
    def __init__(self, graph, SOTASolver):
        self.graph = graph
        self.SOTASolver = SOTASolver
        self.time_budget = SOTASolver.get_time_budget()
        self.num_nodes = graph.get_num_nodes()
        # array of reach values; initializing all reach values to zero
        self.reach_values = np.zeros(self.num_nodes)
        # m_id and m_si cache, as they are used multiple times
        self.m_id_cache = {}    # key: dest_node, value: array m(id)
        self.m_si_cache = {}    # key: (s,i), value: m(s,i)
        self.node_d = SOTASolver.get_destination()
    
    def get_reach_values(self):
        return self.reach_values

    def print_reach_values(self):
        print("Reach values for each node:")
        for i, reach in enumerate(self.reach_values):
            print(f"Node {i}: Reach = {reach:.2f}")

    def m(self, node_i, node_j):
        """
        Computing the m metric as average travel time from node_i to node_j
        """
        return self.graph.get_adjacency_matrix_value(node_i, node_j)

    def get_optimal_path_nodes(self, start_node, dest_node):
        """
        Returning the set of nodes that are in at least one optimal path from 
        start_node to dest_node for times <= T.
        Uses SOTA functions.
        """
        visited = set()

        for t in range(1, self.time_budget + 1):
            path = self.SOTASolver.extract_path_from_time(start_node, t)
            # if the path exists and 
            if path and path[-1] == dest_node:
                visited.update(path)
        
        return visited

    def reach_computation(self):
        """
        Computes the reach values by running a SOTA search for all possible destinations in the graph.
        """
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

                # updating the reach values for each visited node
                for i in visited_nodes:
                    if (s, i) in self.m_si_cache:
                        m_si = self.m_si_cache[(s, i)]
                    else:
                        m_si = self.m(s, i)
                        self.m_si_cache[(s, i)] = m_si

                    self.reach_values[i] = max(self.reach_values[i], min(m_si, m_id[i]))
        
        return self.reach_values

    def reach_pruning(self):
        """
        Pruning if r(i,T) < min(m(s,i), m(i,d)). We use the cached values to avoid
        recomputing m(s,i) and m(i,d), as we computed them in reach_computation function.
        """
        pruned_nodes = set()
        
        for d in range(self.num_nodes):
            # retrieving m_id for all nodes i
            m_id = self.m_id_cache.get(d)
            # if not cached, compute it
            if m_id is None:
                m_id = [self.m(i, d) for i in range(self.num_nodes)]
                self.m_id_cache[d] = m_id

            for s in range(self.num_nodes):
                if s == d:
                    continue
                
                for i in range(self.num_nodes):
                    # don't prune the destination node
                    if i == self.node_d:
                        continue

                    if (s, i) in self.m_si_cache:
                        m_si = self.m_si_cache[(s, i)]
                    else:
                        m_si = self.m(s, i)
                        self.m_si_cache[(s, i)] = m_si
                    
                    if self.reach_values[i] < min(m_si, m_id[i]):
                        self.graph.prune_node(i)
                        pruned_nodes.add(i)
        
        print(f"Pruned nodes: {pruned_nodes}")
        return pruned_nodes

class ArcFlags:
    def __init__(self, graph, num_regions, SOTASolver):
        self.graph = graph
        self.num_regions = num_regions
        self.regions = None
        self.arc_flags = {e: {r: False for r in range(self.num_regions)} for e in self.graph.get_edges()}
        self.SOTASolver = SOTASolver
        self.time_budget = SOTASolver.get_time_budget()
        self.node_d = SOTASolver.get_destination()

    def print_arcflags(self):
        print("Arc-Flags for each edge:")
        for edge, flags in self.arc_flags.items():
            print(f"Edge {edge}: {flags}")
    
    def print_graph_sections(self):
        self.graph.print_graph_sections(self.regions)
    
    def partition_graph(self, alpha=1.0, beta=1.0):
        """
        Divides the graph into num_regions regions using K-means algorithm on node vectors.
        Each node is represented by the vector:
        x_i = alpha * A[i,:] + beta * Var[i,:]
        """
        # combining the two matrices
        X = alpha * self.graph.adjacency_matrix + beta * self.graph.variance_matrix

        # applying K-means clustering
        kmeans = KMeans(n_clusters=self.num_regions, random_state=42, n_init=10)
        region_labels = kmeans.fit_predict(X)

        self.regions = {i: int(region_labels[i]) for i in range(self.graph.get_num_nodes())}

        return self.regions

    def compute_optimal_policy(self, dest_node):
        """
        Uses SOTA solver to compute the optimal policy to the given destination node.
        returns a list of arcs (i,j) that are part of the optimal policy to dest_node from a given node.
        """
        self.SOTASolver.set_destination(dest_node)
        self.SOTASolver.solve()

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
    
    def compute_arcflags(self):
        """
        Computes the arc-flags for all edges and all regions.
        """
        self.partition_graph()

        for d in range(self.graph.get_num_nodes()):
            region_d = self.regions[d]
            relevant_edges = self.collect_relevant_edges(d)

            for e in relevant_edges:
                self.arc_flags[e][region_d] = True
        
        return self.arc_flags

    def arcflags_pruning(self):
        """
        Prunes from the graph edges whose flag is False for the destination-node's region.
        """
        region_d = self.regions.get(self.node_d)
        
        pruned_edges = 0

        for (u, v) in self.graph.get_edges():
            # pruning condition
            if not self.arc_flags[(u,v)][region_d]:
                self.graph.prune_edge(u,v)
                pruned_edges += 1
        
        print(f"[ArcFlags] Pruned {pruned_edges} edges for destination {self.node_d} (region {region_d}).")