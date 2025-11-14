import networkx as nx
import numpy as np

class Dijkstra:
    def __init__(self, adj_matrix):
        self.adj_matrix = adj_matrix
        self.G = nx.DiGraph()
        self.num_nodes = adj_matrix.shape[0]

        self.initialize_graph()

    def get_adj_matrix(self):
        return self.adj_matrix
    
    def initialize_graph(self):
        """
        Initializing NetworkX graph with adjacency matrix values.
        """
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                weight = self.adj_matrix[i, j]
                if weight not in (None, np.inf) and weight != 0 and i != j:
                    self.G.add_edge(i, j, weight=float(weight))
        
        return self.G

    def compute_path(self, start_node, invert=False):
        """
        Computes the optimal path from start node to every other node, with Dijkstra algorithm. 
        With invert set to False, we compute a path from a source node to all the others, with
        True the opposite we compute to all the nodes to a destination.
        """
        if invert == False:
            # computing from source to all destinations
            pred, dist = nx.dijkstra_predecessor_and_distance(self.G, source=start_node)
        else:
            # computing from destination to all sources, by inverting the graph
            revG = self.G.reverse(copy=True)
            pred, dist = nx.dijkstra_predecessor_and_distance(revG, source=start_node)
        
        return pred, dist
    
    def get_all_optimal_paths(self, predecessors, dest_node):
        """
        Returns the list of optimal paths, given the predecessors structure.
        It works with nodes with multiple predecessors
        """
        def dfs(current):
            if not predecessors[current]:
                return [[current]]
            
            paths = []
            for pred in predecessors[current]:
                for path in dfs(pred):
                    paths.append(path + [current])
            return paths

        return dfs(dest_node)
    
    def get_nodes_from_optimal_paths(self, optimal_paths):
        """
        Returns a set of nodes that are part of at least one optimal path
        from source to destination
        """
        nodes = set()

        for path in optimal_paths:
            nodes.update(path)

        return nodes
    
    def get_edges_from_optimal_paths(self, optimal_paths):
        """
        Returns a set of nodes that are part of at least one optimal path
        from source to destination.
        """
        edges = set()

        for path in optimal_paths:
            for i in range(len(path) - 1):
                edges.add((path[i], path[i + 1]))

        return edges