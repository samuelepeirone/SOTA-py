import numpy as np

class Reach:
    """
    Metric that quantifies the radius of node's relevance.
    A node will have a small reach value if it only belong to shortest paths
    whose sources or destinations are close to the node; large reach for shortest 
    paths involving distante sources and destinations.
    """
    def __init__(self, graph, time_budget, SOTASolver):
        self.graph = graph
        self.time_budget = time_budget
        self.SOTASolver = SOTASolver
        self.num_nodes = graph.get_num_nodes()
        # array of reach values; initializing all reach values to zero
        self.reach_values = np.zeros(self.num_nodes)
        # m_id and m_si cache, as they are used multiple times
        self.m_id_cache = {}    # key: dest_node, value: array m(id)
        self.m_si_cache = {}    # key: (s,i), value: m(s,i)
    
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