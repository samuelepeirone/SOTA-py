from stochastic_graph import StochasticGraph
import math
from math import exp, gamma, log
import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import gamma

class SOTA(ABC):
    def __init__(self, graph, node_d, time_budget):
        """
        @param graph: StochasticGraph instance
        @param node_d: destination node index
        @param time_budget: time budget (integer)
        Initializes the Standard SOTA Solver with the given graph, source and destination nodes, and time budget.
        """
        self.graph = graph
        self.node_d = node_d
        self.time_budget = time_budget
        self.num_nodes = graph.get_num_nodes()
        self.min_edge = graph.min_edge
        self.num_cols = math.ceil(self.time_budget / self.min_edge)

        self.sota_matrix = self.initialize_matrix()
        # -1 indicates no policy set
        self.policy_matrix = -1 * np.ones((self.num_nodes, self.num_cols), dtype=int)
    
    def initialize_matrix(self):
        """
        Initialize the SOTA matrix with zeros and set the destination node row to 1s.
        """
        n_rows = self.graph.get_num_nodes()
        # n_cols = math.ceil(self.time_budget / self.min_edge) + 1
        
        matrix = np.zeros((n_rows, self.num_cols), dtype=float)
        matrix[self.node_d, :] = 1
        
        return matrix
    
    def print_matrix(self, matrix, string):
        print(string)
        print(np.array2string(matrix, formatter={'float_kind':lambda x: f"{x:6.2f}"}))

    def print_sota_matrix(self):
        self.print_matrix(self.sota_matrix, "SOTA Matrix:")
    
    def print_policy_matrix(self):
        self.print_matrix(self.policy_matrix, "Policy Matrix:")

    def get_sota_matrix(self):  
        return self.sota_matrix
    
    def get_policy_matrix(self):
        return self.policy_matrix
    
    def get_destination(self):
        return self.node_d
    
    def get_time_budget(self):
        return self.time_budget
    
    def get_min_edge(self):
        return self.min_edge
    
    def get_num_cols(self):
        return self.num_cols
    
    def set_destination(self, node_d):
        """
        Sets the new destination node and re-initializes the SOTA matrix.
        """
        self.node_d = node_d
        self.sota_matrix = self.initialize_matrix()
    
    def compute_density(self, node_i, node_j, s):
        """
        Compute the density function for the edge from node_i to node_j
        using a gamma distribution with parameters from adjacency_matrix and variance_matrix.
        We use the logaritmic function to avoid overflow.
        @param s: weight(time-travel) that we are computing density for
        """
        mu = self.graph.get_adjacency_matrix_value(node_i, node_j)
        sigma2 = self.graph.get_variance_matrix_value(node_i, node_j)

        # no edge case
        if mu == 0 or sigma2 == 0:
            return 0.0

        alpha = mu**2 / sigma2
        beta = mu / sigma2
        scale = 1 / beta

        if s < self.min_edge:
            return 0.0

        # we use the logaritmic function
        try:
            log_G = gamma.logpdf(s, a=alpha, scale=scale)
            G = np.exp(log_G)
        except OverflowError:
            # if we still have overflow, we return 0.0, as the density is almost null
            G = 0.0

        # we assume that if s < min_edge, then the density is 0, 
        # as the time is less than the real minimum time in the graph
        return G

    @abstractmethod
    def compute_convolution(self, node_i, node_j, t, matrix): 
        """ 
        Discrete convolution for edge (node_i, node_j) in matrix at time t.
        @param:t: remaining time
        @return:m: convolution result
        """
        m = 0.0
        # we transform time into discretized index
        num_steps = int(t // self.min_edge)

        for s_step in range(1, num_steps + 1):
            s_time = s_step * self.min_edge  # corresponding real time
            p = self.compute_density(node_i, node_j, s_time)

            matrix_col = int((t - s_time) // self.min_edge)
            if 0 <= matrix_col < self.num_cols:
                m += p * matrix[int(node_j), int(matrix_col)]

        return m

    @abstractmethod
    def update_node(self, node_i):
        pass

    @abstractmethod
    def update_sota(self):
        pass

    @abstractmethod
    def solve(self, eps=1e-4, max_iter=100):
        pass

    def extract_path_from_time(self, start_node, t_idx):
        """
        Extracts the optimal path for time-index watching the policy row.
        It stops if we get to destination, we get to a node -1 or if a loop is detected.
        @return: list of nodes representing the optimal path from source to destination
        """
        path = [start_node]
        current = start_node
        visited = set()

        policy = self.policy_matrix[:, t_idx]

        while True:
            # to avoid infinite loops
            if current in visited:
                print(f"Loop detected from {start_node} at time_idx {t_idx}")
                break

            visited.add(current)

            # retrieving the next node
            next_node = policy[current]

            # stop if next = -1 or destination
            if next_node == -1 or current == self.node_d:
                break

            path.append(int(next_node))
            current = next_node

        return path

    @abstractmethod
    def extract_path(self, start_node):
        """
        Extracts the optimal path using only the policy row and the last column.
        @return: list of nodes representing the optimal path from source to destination
        """
        return self.extract_path_from_time(start_node, self.num_cols-1)

class StandardSOTASolver(SOTA):
    def __init__(self, graph, node_d, time_budget):
        super().__init__(graph, node_d, time_budget)

    def compute_convolution(self, node_i, node_j, t):
        return super().compute_convolution(node_i, node_j, t, self.sota_matrix)

    def update_node(self, node_i):
        """
        Update the row of the SOTA matrix corresponding to node_i
        for all times from 1 to time budget, using compute_convolution function.
        Updates also the policy matrix.
        """
        # if the node is the destination, set all values to 1
        if node_i == self.node_d:
            self.sota_matrix[node_i, :] = 1.0
            return
        
        # the steps of the matrix are in multiples of min_edge, 
        # but the index is in integers, so we use steps of 1
        for t_idx in range(1, self.num_cols+1):
            # current time
            t = t_idx * self.min_edge
            max_val = 0.0
            best_successor = -1
            successors = self.graph.get_successor_nodes(node_i)  # finds the successors of node_i

            for j in successors:
                conv = self.compute_convolution(node_i, j, t)
                if conv > max_val:
                    max_val = conv
                    best_successor = j

            self.sota_matrix[node_i, t_idx-1] = max_val
            self.policy_matrix[node_i, t_idx-1] = best_successor

    def update_sota(self):
        """
        Executing an iteration of the SOTA algorithm on all the nodes
        Returns the norm of the difference between the old and new SOTA matrix
        """
        old_matrix = self.sota_matrix.copy()  # copy of the current SOTA matrix
        
        for node in range(self.num_nodes):
            self.update_node(node)
        
        # compute the norm of the difference between old and new matrix
        norm = np.sum(np.abs(self.sota_matrix - old_matrix))
        
        return norm

    def solve(self, eps=1e-4, max_iter=100):
        """
        Standard SOTA solving algorithm implementation.
        It iteratively updates the SOTA matrix until convergence or max iterations reached.
        @param eps: convergence threshold
        """
        for _ in range(max_iter):
            delta = self.update_sota()
            if delta < eps:
                # convergence
                print("Convergence reached!")
                break
        
        print("End of iterations")
        return self.sota_matrix
    
    def extract_path(self, node_s):
        return super().extract_path(node_s)
    
class SingleIterationSOTASolver(SOTA):
    def __init__(self, graph, node_d, time_budget):
        super().__init__(graph, node_d, time_budget)

    def compute_convolution(self, node_i, node_j, t, prev_sota_matrix):
        return super().compute_convolution(node_i, node_j, t, prev_sota_matrix)

    def update_node(self, node_i, t, prev_sota_matrix):
        """
        Update the maximum probability for node_i at each time t
        considering all the successors
        """
        max_value = 0.0
        best_successor = -1

        # loop over all successors of node_i
        for node_j in self.graph.get_successor_nodes(node_i):
            m = self.compute_convolution(node_i, node_j, t, prev_sota_matrix)
            if m > max_value:
                max_value = m
                best_successor = node_j
        
        # updating the sota_matrix and policy_matrix for node_i
        t_idx = math.floor(t / self.min_edge)
        # to avoid "out of bounds" error with min_edge = 1
        t_idx = min(t_idx, self.num_cols-1)
        self.sota_matrix[node_i, t_idx] = max_value
        self.policy_matrix[node_i, t_idx] = best_successor

    def update_row(self, node_i, k, prev_sota_matrix):
        """
        Update the row of the SOTA matrix corresponding to node_i
        from tauk-min-edge+1 to tauk.
        """
        # skipping destination node
        if node_i == self.node_d:
            return
        
        # compute the current maximum time
        tauk = min(self.time_budget, k * self.min_edge)

        t_start_idx = int(max(0, tauk - self.min_edge + 1) / self.min_edge)
        t_end_idx = int(tauk / self.min_edge)

        for t_idx in range(t_start_idx, t_end_idx + 1):
            t_real = t_idx * self.min_edge
            self.update_node(node_i, t_real, prev_sota_matrix)

    def update_sota(self, k):
        """
        Performes a single iteration of SIA approach.
        Updates the SOTA matrix for all nodes for the selected time slice k.        
        """
        # copying current SOTA matrix
        prev_sota_matrix = self.sota_matrix.copy()

        for node_i in range(self.num_nodes):
            if node_i != self.node_d:
                self.update_row(node_i, k, prev_sota_matrix)

    def solve(self):
        """
        Solves the SOTA problem using a single iteration approach.
        Updates the SOTA matrix for L iterations, where L is the maximum number of steps
        that can be taken within the time budget.
        """
        # loop over all iterations
        for k in range(1, self.num_cols):
            self.update_sota(k)

    def extract_path(self, node_s):
        return super().extract_path(node_s)