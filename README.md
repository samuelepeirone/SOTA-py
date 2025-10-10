# SOTA-py

## SOTA Solvers

### SOTA solver - documentation

Implementation of the successive approximations algorithm, page $3$ of paper *[[A tractable class of algorithms for reliable routing in stochastic neworks]]*.

```cmd
📁 SOTA solver
   🔹 main.py
   🔹 SOTA.py
	   🔸SOTA(ABC)
	   🔸StandardSOTASolver(SOTA)
	   🔸SingleIterationSOTASolver(SOTA)
   🔹 stochastic_graph.py
	   🔸StochasticGraph
   🔹 preprocessing.py
	   🔸Reach
	   🔸ArcFlags
```

#### `StochasticGraph`

attributes:

- `adjacency_matrix`
- `variance_matrix`
- `min_edge`: minimum edge in the graph

functions:

- `print_adjacency_matrix()`, `print_variance_matrix()`, `print_graph_all_nodes()`, `print_graph()`: simple print functions
- `find_min_edge()`: it finds the minimum edge across all the adjacency matrix. We will need this information as we discretize time, using $\Delta t = \text{min\_edge}$ and to lower bound `sample_distance` function.
- `sample_distance(node1, node2)`: it returns the distance between the nodes, sampled using a *Gamma distribution*, retrieving mean and variance from respectively adjacency and variance matrices. The function returns $\infty$ for $0$ values in both adjacency matrix and variance matrix. The return value is lower bounded to *min_edge*.
- `get_min_edge`: we only use the `find_min_edge()` function one time and then retrieve the value from the attributes, as we need it to avoid the complexity going up
- `get_edges()`: returns a list of edges
- `get_incoming_nodes(node)`: returns a list of nodes that have an edge to the given node, by checking the adjacency matrix.
- `get_successor_nodes(node)`: returns a list of nodes that are reachable from the given node, by checking the adjacency matrix.
- `prune_node(node)`: non-disruptive pruning technique, that removes a node from the graph and all associated edges by setting both the adjacency and variance matrix rows and columns to zero. The indexes of the matrices are mantained.
- `prune_edge(node_i, node_j)`: non-disruptive pruning technique, that removes the edge between node $i$ and node $j$ from the graph. The indixes of the matrices are mantained.

#### `SOTA`

attributes:

- `graph`
- `node_d`: destination node
- `time_budget`
- `num_nodes`
- `min_edge`: shortest time-travelling edge in the whole `StochasticGraph`

functions:

- `initialize_matrix()`: the function initializes the SOTA matrix. We set all entries to $0$, except for the row of *node_d*, that we set to $1$, as we are already on the destination node.
- `compute_density(node_i, node_j, s)`: computes the density function for the edge between nodes $i$ and $j$.
  The density function of a continue distribution $f(s)$ represents the relative probability of a random variable having a value near $s$. Density is defined as:
  $$
  G(s) = \frac{\beta^\alpha s^{\alpha-1}e^{-\beta s}}{\Gamma(\alpha)}
  $$
  Where $\alpha$ and $\beta$ are parameters of the gamma distribution, $\Gamma(\alpha)$ is the gamma function and $s$ is the time past on the arch. $s\approx\omega$, as $\omega$ is the continous time on the arch and $s$ is the discrete time unit corresponding to $w$.
  $$
  \alpha = \frac{(\text{mean})^2}{\text{variance}},\qquad\beta=\frac{\text{mean}}{\text{variance}}
  $$
  We will use the density value in the convolution, as it approssimates $p(s)$ in the convolution formula:
  $$
  \begin{aligned}
  \sum_{s=1}^tp(s)u_j(t-s)\approx\int_0^t p_{ij}(\omega)u_j(t-\omega)d\omega \\
  u_i(t) = \max_j\sum_{s=1}^{t}p(s)\cdot u_j(t-s+1)
  \end{aligned}
  $$
  As we are dealing with big numbers, we will use the following log function to compute the $G(s)$ value to avoid overflow.
  $$
  \begin{aligned}
  &\log G(s) = \alpha\log\beta+(\alpha-1)\log s-\beta s-\log\Gamma(\alpha)\\ \\
  &G = \exp(\log G(s))
  \end{aligned}
  $$
  
  We then return $0$ if $s<\text{min\_edge}$, as the time is less than the minimum physical time between two nodes in the whole graph. In this way we avoid irrealistic probabilities on too small travel-times.
  Overflow detection and handling: if the log produces overflow anyway, we will return $0.0$, as the density is almost null in those cases.
- `printmatrix(matrix, string)`, `print_sota_matrix()`, `print_policy_matrix()`: trivial print functions
- `compute_convolution(node_i, node_j, t)`: computes the discrete convolution for the edge $(i,j)$, using the `compute_density` function and multiplying it to the *sota_matrix* values.
- `extract_path_from_time(start_node, t_idx)`: it extracts the optimal path for a specific time-index. It stops if: we get to destination, we get to a node $-1$.
  There is no control on already visited nodes, as the optimal path in SOTA can contain loops: in stochastic SOTA, it is possible to be convenient for a path to re-visit a node that has already been visited in the past.
- `extract_path(start_node)`: extracts the optimal path from the last column of the policy matrix. From the starting node, reads the entry at the last column, then goes to that index, read the next one and so on, until it gets to destination node.

#### `StandardSOTASolver(SOTA)`

>[!note]
>In this approach, we update the whole sota_matrix until the solution converges (the differences between two consecutive iterations is less than a threshold). Each iteration re-compute all the cells of the matrix, this is the reason why it's kind of a slow algorithm.

attributes:

- `sota_matrix`: this matrix will contain all the probabilities $u_i(t)$ for each row $i$ and time step $t$.
  Rows correspond to nodes and columns to time steps.
  If maximum time budget is $T$, then the number of columns is:
  $$
  N_{col} = \lceil\frac{T}{\Delta t}\rceil+1
  $$
  "$+1$" because we want to include the column $t=0$.
  SOTA matrix $U$ will be a matrix of dimension $|V| \times (T+1)$.
- `policy_matrix`: matrix that will contain the optimal successor index for each node $i$ and time $t$. It's like a map of the optimal decisions for each state $(i,t)$.

functions:

- `update_node(node_i)`: updates the node of the sota_matrix corresponding to node $i$ for all columns of step $\Delta t$ from $1$ to $T$. Updates also the values of the *policy_matrix* with the best successor for each node and time step.
- `update_sota()`: executing an iteration of the SOTA algorithm on all the nodes. It returns the norm of the difference between the old and the new sota_matrix, as a measure of how much the update changed the matrix.
- `solve(eps, max_iter)`: iteratively updates the sota_matrix until convergence or max iterations reached. The $\text{eps}$ param is the convergence threshold.
- `get_sota_matrix()`, `get_policy_matrix()`

#### `SingleIterationSOTASolver(SOTA)`

>[!note]
>With the the SIA approach, we update the matrix one cell at a time, iteratively proceding forward, as each iteration computes only the new temporal window $[\tau_k-\Delta t+1 .. \tau_k]$.
>This makes this algorithm faster than the standard approach, as it computes each cell only one time.

functions:

- `compute_convolution(node_i, node_j, t, prev_sota_matrix)`: computes the convolution on *prev_sota_matrix*.
- `update_node(node_i, t, prev_sota_matrix)`: update the maximum probability for node $i$ at time $t$ considering all its successors. We compute the convolution for all the successor nodes and we take the maximum value. We update only the $(i,t)$ entry in the matrix, not the whole row as we did in the StandardSotaSolver.
  The update is based on the previous matrix, as an in-place update would "contaminate" the calculus by reading updated values of another node in the same iteration.
- `update_row(node_i, k, prev_sota_matrix)`: perform an iteration of the single iteration algorithm. Updates the cell $(i,k)$ of sota_matrix and policy_matrix for time interval: $[\tau_k-\Delta t+1 .. \tau_k]$.
  Note that $\tau_k$ is the maximum temporal instant considered at the iteration $k$. It's the cumulated time that you can spend after $k$ steps of discretization. 
  After the first iteration: $\tau_1=\Delta t$, after two steps: $\tau_2=2\Delta t$. 
  As we start from $k=1$ to $k=L(=\text{number of instants})$, at each iteration $k$ I don't need to re-compute the values from $0$ to $\tau_k-\Delta t$, because I've already updated them in the previous iterations. I only need to update values between $\tau_k-\Delta t+1$ and $t_k$.
  In the code: $\tau_k = \text{tauk}$, $\Delta t = \text{t}$.
- `update_sota(k)`: performs a single iteration of SIA approach. Updates the SOTA matrix for all nodes for the selected time slice $k\in[1..L]$, where $L=T/\Delta t$.
- `solve()`: it solves the SOTA problem using a single iteration approach. Updates the sota_matrix for $\text{num\_cols}$ iterations.

#### `Reach`

[[Precomputation techniques for the stochastic on-time arrival problem#Reach]]

>**for** $d\in V$ **do**
>$\quad$compute the optimal policy with budget $T$; $\forall s \neq d \in V$
>$\quad$**for** $i\in V$ **do**
>$\quad$$\quad$compute $m(i,d)$
>$\quad$**for** $s\in V$ **do**
>$\quad$$\quad$compute $\bigcup_{t\leq T}V_{sd}(t)$
>$\quad$$\quad$**for** $i\in\bigcup_{t\leq T}V_{sd}(t)$ **do**
>$\quad$$\quad$$\quad$compute $m(s,i)$
>$\quad$$\quad$$\quad$set $r(i,T) = \max(r(i,T), \min(m(s,i),m(i,d)))$
>**return** r
>
>Notations:
>
>- $V_{sd}(t)$ is the set of nodes that are in at least one optimal path from $s$ to $d$ with temporal budget $t$. In the algorithm we take $\bigcup_{t\leq T}V_{sd}(t)$, so the union on all the times less or equal than $T$.

attributes:

- `graph`
- `time_budget`
- `SOTASolver`: istance of SOTASolver, either `StandardSOTASolver` or `SingleIterationSOTASolver`
- `reach_values`: array that will contain the reach values
- `m_id_cache`, `m_si_cache`: cache values, used to not recompute the same values multiple times.

functions:

- `print_reach_values()`
- `m(node_i, node_j)`: computing the $m(i,j)$ metric as the average travel time from node $i$ to node $j$
- `get_optimal_path_nodes(start_node, dest_node)`: returns the set of nodes that are in, at least, one optimal path from start node to destination for times $t\in[0..T]$.
  Returns $\cup_{t\leq T}V_{sd}(t)$.
- `reach_computation()`: computes the reach values by running a SOTA search for all possible destinations in the graph. The algorithm corresponds to the example algorithm.
- `reach_pruning()`: prunes a node if
  $$
  r(i,t)<\min(m(s,i),m(i,d))
  $$
  using the `graph.prune_node(node)` function. It uses cache values, as reach_pruning and reach_computation fuction compute the same values.
  The pruning technique implemented avoid the destination node to be pruned from the graph.

>[!danger]
>No mention in the paper about strategies to protect the source nodes to be pruned

#### `ArcFlags`

[[Precomputation techniques for the stochastic on-time arrival problem#Arc-flags]]

>**input:** a graph $G$, a partition of the edges $R$ and a time budget T
>**output:** the arc-flags $AF(.,T,.)$
>**initialization:** $AF(e, T, r) = \text{FALSE}, \forall(e,r)\in E\times R$
>**for** $d\in V$ **do**
>$\quad$compute the optimal policy with budget $T;\forall s\neq d \in V$
>$\quad$compute $\bigcup_{t\leq T, s\in V}E_{sd}(t)$
>$\quad$**for** $e \in \bigcup_{t\leq T, s\in V}E_{sd}(t)$ **do**
>$\quad\quad$ set $AF(e, T, r(d)) = \text{TRUE}$
>**return** $AF$

$AF(e, T, r)$ is initially $\text{FALSE}$ for each arc and each region.
$$
AF[e] = [f_{e,1}, f_{e,2}, ..., f_{e,|R|}]
$$
Each arc is associated to a boolean vector of length $|R|$, where $f_{e,r}=\text{TRUE}$ if arc $e$ is relevant for region $r$.

>[!note]
>$e$ is an arc of the graph, while $r$ represents a region that the graph has been partitioned in.
>We can represent $AF$ as a matrix $|E|\times|R|$.
>
>We will mantain the aggregated information until the current temporal timestep $t$ (we are not interested about intermediate configurations). At the end, we will have the information about timestep $T$.

attributes:

- `graph`
- `num_regions`: the number of partitions of the graph
- `regions`: the regions that have been partitioned from the graph
- `arc_flags`: structure where arc flags are stored. It's a dictionary of dictionaries, that contains values for each region. The basic structure is the following (example for 3 regions):

```python
Edge (0, 1): {0: False, 1: False, 2: False}
Edge (0, 2): {0: False, 1: False, 2: False}
Edge (0, 3): {0: False, 1: False, 2: False}
```
  
- `time_budget`
- `SOTASolver`
- `node_d`: destination node

functions:

- `print_graph_sections()`: wrapper function for `graph.print_graph_sections(sections)`
- `partition_graph(alpha, beta)`: divides the graph into *num_regions* regions using $K$-means algorithm (with $k=\text{num\_regions}$). Each node will be represented as:
  $$
  x_i = \alpha Adj[i, :] + \beta Var[i, :]
  $$
- `compute_optimal_policy(dest_node)`: uses the SOTASolver to compute the optimal policy to dest_node from each source node. Returns a list of arcs $(i,j)$ that are part of the optimal policy from the starting node to dest_node.
- `collect_relevant_edges(dest_node)`: returns the set of edges that are part of an optimal path to the destination node. This corresponds to $\cup_{t\leq T, s\in V}E_{sd}(t)$.
- `arcflags_computation()`: computes the arc-flags structure for all edges and all regions.
- `arcflags_pruning()`: we prune an edge if $AF(e,t,r)=\text{FALSE}$, where $r$ is the region where destination node $d$ belongs.
