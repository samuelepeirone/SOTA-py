import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import math

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

    def print_graph_all_nodes(self):
        """
        Simple displaying of the graph
        """
        G = nx.from_numpy_array(self.adjacency_matrix, create_using=nx.DiGraph)
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', arrows=True)
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        plt.show()

    def print_graph(self, path=None, grid_shape=None):
        """
        Dislaying the graph without isolated nodes. Nodes without incoming or outgoing edges are ignored.
        If a path is specified (as a list of nodes), it will be displayed
        """
        n_total = self.adjacency_matrix.shape[0]

        # find active nodes
        active_nodes = np.where(self.adjacency_matrix.sum(axis=0) + self.adjacency_matrix.sum(axis=1) > 0)[0]

        # Reduced matrix
        reduced_matrix = self.adjacency_matrix[np.ix_(active_nodes, active_nodes)].astype(float)

        # creates the graph
        G = nx.DiGraph()
        for i_idx, i in enumerate(active_nodes):
            for j_idx, j in enumerate(active_nodes):
                weight = reduced_matrix[i_idx, j_idx]
                if weight != 0:
                    G.add_edge(i, j, weight=weight)

        # === Fixed grid layout for all nodes ===
        cols = math.ceil(math.sqrt(n_total))
        pos = {}
        for node in range(n_total):
            r, c = divmod(node, cols)
            pos[node] = (c, -r)  # c -> x, -r -> y

        # drawing
        nx.draw(G, {node: pos[node] for node in active_nodes},
                with_labels=True, node_color='lightblue', edge_color='gray', arrows=True)

        # Edge labels
        edge_labels = {(u, v): f"{d['weight']:.1f}" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, {node: pos[node] for node in active_nodes}, edge_labels=edge_labels)

        # showing the path
        if path and len(path) > 1:
            path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
            
            nx.draw_networkx_edges(
                G,
                pos={node: pos[node] for node in active_nodes},
                edgelist=path_edges,
                edge_color='red',
                width=2.5,
                arrows=True
            )
            nx.draw_networkx_nodes(
                G,
                pos={node: pos[node] for node in active_nodes},
                nodelist=path,
                node_color='orange'
            )

        plt.axis('equal')
        plt.show()

    def print_graph_sections(self, node_sections=None, path=None):
        """
        Displaying the graph with nodes colored based on their sections if provided.
        If a path is specified (as a list of nodes), it will be displayed
        """
        n_total = self.adjacency_matrix.shape[0]  # numero totale di nodi

        # === find active nodes ===
        active_nodes = np.where(self.adjacency_matrix.sum(axis=0) + self.adjacency_matrix.sum(axis=1) > 0)[0]
        reduced_matrix = self.adjacency_matrix[np.ix_(active_nodes, active_nodes)]

        # === creating graph ===
        G = nx.from_numpy_array(reduced_matrix, create_using=nx.DiGraph)
        mapping = {i: active_nodes[i] for i in range(len(active_nodes))}
        G = nx.relabel_nodes(G, mapping)

        # === fixed-grid layout ===
        cols = math.ceil(math.sqrt(n_total))
        pos = {}
        for node in range(n_total):
            r, c = divmod(node, cols)
            pos[node] = (c, -r)

        # === different nodes colors for different sections ===
        if node_sections is not None:
            unique_sections = sorted(set(node_sections.values()))
            color_map = cm.get_cmap('tab10', len(unique_sections))
            node_colors = [
                color_map(unique_sections.index(node_sections[n])) if n in node_sections else (0.8, 0.8, 0.8, 1.0)
                for n in G.nodes()
            ]
        else:
            node_colors = 'lightblue'

        # === drawing base graph ===
        nx.draw(
            G,
            {n: pos[n] for n in G.nodes()},
            with_labels=True,
            node_color=node_colors,
            edge_color='gray',
            arrows=True
        )

        # weights
        edge_labels = {(u, v): f"{d['weight']:.1f}" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, {n: pos[n] for n in G.nodes()}, edge_labels=edge_labels)

        # === legend ===
        if node_sections is not None:
            legend_handles = [
                mpatches.Patch(color=color_map(i), label=f'Section {unique_sections[i]}')
                for i in range(len(unique_sections))
            ]
            plt.legend(handles=legend_handles, title="Sections")

        # === lighting the path ===
        if path and len(path) > 1:
            path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]

            # edges
            pos_active = {node: pos[node] for node in active_nodes}
            nx.draw_networkx_edges(
                G,
                pos=pos_active,
                edgelist=path_edges,
                edge_color='orange',
                width=1.75,
                arrows=True
            )

            # border of the nodes
            nx.draw_networkx_nodes(
                G,
                pos=pos_active,
                nodelist=path,
                node_color='none',
                edgecolors='orange',
                linewidths=2.5,
                node_size=440
            )

            # Secondo strato: colore originale della sezione
            if node_sections is not None:
                path_colors = [
                    color_map(unique_sections.index(node_sections[n])) if n in node_sections else (0.8, 0.8, 0.8, 1.0)
                    for n in path
                ]
            else:
                path_colors = ['lightblue'] * len(path)

            nx.draw_networkx_nodes(
                G,
                pos=pos_active,
                nodelist=path,
                node_color=path_colors
            )

        plt.axis('equal')
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

    def get_nodes(self):
        """
        Returns the list of nodes in the graph
        """
        return list(range(self.num_nodes))

    def get_edges(self):
        """
        Returns a list of edges in the graph as (i, j) tuples.
        """
        edges = []
        rows, cols = np.where(self.adjacency_matrix > 0)
        for i, j in zip(rows, cols):
            edges.append((i, j))
        return edges

    def get_min_edge(self):
        return self.min_edge
    
    def get_num_nodes(self):
        return self.num_nodes
    
    def get_adjacency_matrix_value(self, i, j):
        return self.adjacency_matrix[i][j]
    
    def get_variance_matrix_value(self, i, j):
        return self.variance_matrix[i][j]
    
    def get_edge_mean(self, node_i, node_j):
        return self.adjacency_matrix[node_i, node_j]
    
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
        
    def prune_node(self, node):
        """
        Removes a node from the graph and all associated edges by setting
        the adjacency and variance matrix rows and columns to zero.
        Non-disruptive pruning, as the index are maintained.
        """
        if node < 0 or node >= self.num_nodes:
            raise ValueError("Node index out of bounds")

        # removing corresponding row and columm in both adjacency and variance matrices
        self.adjacency_matrix[:, node] = 0
        self.adjacency_matrix[node, :] = 0

        self.variance_matrix[:, node] = 0
        self.variance_matrix[node, :] = 0

        # updating attributes
        self.min_edge = self.find_min_edge()
    
    def prune_edge(self, node_i, node_j):
        """
        Prune the edge in between node_i and node_j
        """
        for node in (node_i, node_j):
            if node < 0 or node >= self.num_nodes:
                raise ValueError("Node index out of bounds")
        
        self.adjacency_matrix[node_i, node_j] = 0
        self.variance_matrix[node_i, node_j] = 0

        self.min_edge = self.find_min_edge()