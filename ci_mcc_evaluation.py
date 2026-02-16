import numpy as np
import networkx as nx
from itertools import combinations


# ============================================================
# D-SEPARATION UTILITIES
# ============================================================

def is_d_separated(G, X, Y, Z):
    """
    Returns True if X and Y are d-separated given conditioning set Z.
    Uses active trail logic via moralization.
    """
    if X == Y:
        return True

    # Step 1: Get ancestors of X, Y, Z
    ancestors = set([X, Y]) | set(Z)
    for node in list(ancestors):
        ancestors |= nx.ancestors(G, node)

    # Step 2: Induced subgraph on ancestors
    subG = G.subgraph(ancestors).copy()

    # Step 3: Moralize
    moral_graph = subG.to_undirected()
    for node in subG.nodes():
        parents = list(subG.predecessors(node))
        for p1, p2 in combinations(parents, 2):
            moral_graph.add_edge(p1, p2)

    # Step 4: Remove conditioned nodes
    moral_graph.remove_nodes_from(Z)

    # Step 5: Check connectivity
    return not nx.has_path(moral_graph, X, Y)


def compute_ci_matrix(adj):
    """
    Computes 0th and 1st order conditional independence matrix.
    Output shape: (n+1, n, n)
    """
    n = adj.shape[0]
    G = nx.from_numpy_array(adj, create_using=nx.DiGraph)

    ci_tensor = np.zeros((n + 1, n, n), dtype=int)

    nodes = list(range(n))

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            # 0th order (no conditioning)
            if is_d_separated(G, i, j, []):
                ci_tensor[0, i, j] = 1

            # 1st order
            for k in nodes:
                if k == i or k == j:
                    continue
                if is_d_separated(G, i, j, [k]):
                    ci_tensor[k + 1, i, j] = 1

    return ci_tensor


# ============================================================
# CI_MCC METRIC
# ============================================================

def ci_mcc(pred_adj, true_adj):
    """
    Compute CI_MCC between predicted and true DAG adjacency matrices.
    """

    print("→ Computing CI tensors")

    pred_ci = compute_ci_matrix(pred_adj)
    true_ci = compute_ci_matrix(true_adj)

    pred = pred_ci.flatten()
    true = true_ci.flatten()

    mask_1 = (pred == 1)
    mask_0 = (pred == 0)

    tp = np.sum(true[mask_1])
    fp = np.sum(1 - true[mask_1])
    tn = np.sum(1 - true[mask_0])
    fn = np.sum(true[mask_0])

    numerator = tp * tn - fp * fn
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    if denominator == 0:
        return 0.0

    return numerator / denominator


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":

    # Example small test
    true_adj = np.array([
        [0,1,0],
        [0,0,1],
        [0,0,0]
    ])

    pred_adj = np.array([
        [0,1,0],
        [0,0,1],
        [0,0,0]
    ])

    score = ci_mcc(pred_adj, true_adj)

    print("\nCI_MCC:", score)
