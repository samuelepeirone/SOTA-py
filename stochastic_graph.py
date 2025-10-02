import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


adjacency_matrix = np.array([
    [0, 4, 5, 9, 0, 0, 0, 0, 0],
    [0, 0, 5, 0, 5, 4, 0, 0, 0],
    [0, 5, 0, 5, 0, 5, 5, 0, 0],
    [0, 0, 5, 0, 0, 0, 9, 5, 0],
    [0, 0, 0, 0, 0, 5, 0, 0, 5],
    [0, 0, 0, 0, 5, 0, 5, 0, 4],
    [0, 0, 0, 0, 0, 5, 0, 5, 9],
    [0, 0, 0, 0, 0, 0, 5, 0, 5],
    [0, 0, 0, 0, 0, 0, 0, 0, 1]
])

variance_matrix = np.array([
    [0, 1/1.2, 1/2, 1/0.6, 0, 0, 0, 0, 0],
    [0, 0, 1/2, 0, 1/2, 1/1.2, 0, 0, 0],
    [0, 1/2, 0, 1/2, 0, 1/2, 1/2, 0, 0],
    [0, 0, 1/2, 0, 0, 0, 1/0.6, 1/2, 0],
    [0, 0, 0, 0, 0, 1/2, 0, 0, 1/2],
    [0, 0, 0, 0, 1/2, 0, 1/2, 0, 1/1.2],
    [0, 0, 0, 0, 0, 1/2, 0, 1/2, 1/0.6],
    [0, 0, 0, 0, 0, 0, 1/2, 0, 1/2],
    [0, 0, 0, 0, 0, 0, 0, 0, 1]
])

class StochasticGraph:
    def __init__(self, adjacency_matrix=adjacency_matrix, variance_matrix=variance_matrix):
        """
        @param adjacency_matrix: numpy 2D array representing the adjacency matrix of the graph
        @param variance_matrix: numpy 2D array representing the variance matrix of the graph
        Initializes the StochasticGraph with the given adjacency and variance matrices.
        """
        self.adjacency_matrix = adjacency_matrix
        self.variance_matrix = variance_matrix
        self.min_edge = self.find_min_edge()
        self.num_nodes = self.count_nodes()

    def print_adjacency_matrix(self):
        print("Adjacency Matrix:")
        for row in self.adjacency_matrix:
            print(" ".join(str(val) for val in row))
        print()

    def print_variance_matrix(self):
        print("Variance Matrix:")
        for row in self.variance_matrix:
            print(" ".join(f"{val:.2f}" for val in row))
        print()

    def print_graph(self):
        G = nx.from_numpy_array(self.adjacency_matrix, create_using=nx.DiGraph)
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', arrows=True)
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        plt.show()

    def find_min_edge(self):
        """
        Find the minimum edge in the adjacency matrix.
        """
        mat = self.adjacency_matrix
        # i don't consider zero entries (no edge)
        mask = (mat > 0)
        if not np.any(mask):
            return None  # no edges
        min_val = np.min(mat[mask])
        return min_val

    def sample_distance(self, node1, node2):
        """
        Returns a sampled distance between node1 and node2
        using a gamma distribution with mean from adjacency_matrix and variance from variance_matrix.
        """
        mean = self.adjacency_matrix[node1, node2]
        var = self.variance_matrix[node1, node2]
        if mean == 0 or var == 0:
            return np.inf  # No edge or zero variance

        shape = mean**2 / var
        scale = var / mean
        # Ensure the sampled distance is at least min_edge to lower-bound it
        return max(np.random.gamma(shape, scale), self.min_edge)

    def count_nodes(self):
        """
        Returns number of nodes in the graph, calculated from the adjacency matrix.
        """
        return self.adjacency_matrix.shape[0]

    def get_min_edge(self):
        return self.min_edge
    
    def get_num_nodes(self):
        return self.num_nodes
    
    def get_adjacency_matrix_value(self, i, j):
        return self.adjacency_matrix[i][j]
    
    def get_variance_matrix_value(self, i, j):
        return self.variance_matrix[i][j]
    
    def get_incoming_nodes(self, node):
        """
        Returns a list of nodes that have an edge to the given node,
        by checking the adjacency matrix.
        """
        incoming = []
        for i in range(self.adjacency_matrix.shape[0]):
            if self.adjacency_matrix[i, node] > 0:
                incoming.append(i)
        return incoming
    
    def get_successor_nodes(self, node):
        """
        Returns a list of nodes that are reachable from the given node,
        by checking the adjacency matrix.
        """
        successors = []
        for j in range(self.adjacency_matrix.shape[1]):
            if self.adjacency_matrix[node, j] > 0:
                successors.append(j)
        return successors
        