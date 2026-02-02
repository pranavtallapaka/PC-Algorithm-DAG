import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.readwrite import BIFReader
from pgmpy.sampling import BayesianModelSampling
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import chisq
from sklearn.preprocessing import LabelEncoder

def log_likelihood(data, edges):
    model = BayesianNetwork(edges)
    model.fit(data, estimator=MaximumLikelihoodEstimator)

    ll = 0.0
    for _, row in data.iterrows():
        ll += model.log_probability(row.to_dict())
    return ll

def visualize_dag(edges, title="Learned Causal DAG", save_path=None):
    G = nx.DiGraph()
    G.add_edges_from(edges)

    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(7, 5))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=2500,
        node_color="lightblue",
        font_size=11,
        arrowsize=20,
        arrowstyle="->"
    )
    plt.title(title)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

def dag_to_adjacency_matrix(edges, nodes):
    """
    Create adjacency matrix A where A[i, j] = 1 if node i -> node j
    """
    idx = {node: i for i, node in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)), dtype=int)

    for u, v in edges:
        A[idx[u], idx[v]] = 1

    return pd.DataFrame(A, index=nodes, columns=nodes)

def pc_plus_likelihood(bif_file, n=5000):
    true_model = BIFReader(bif_file).get_model()
    sampler = BayesianModelSampling(true_model)
    data = sampler.forward_sample(size=n, seed=42)

    for c in data.columns:
        data[c] = LabelEncoder().fit_transform(data[c])

    nodes = list(data.columns)

    cg = pc(
        data.to_numpy(),
        alpha=0.05,
        indep_test=chisq,
        node_names=nodes
    )

    pc_edges = []
    for e in cg.G.get_graph_edges():
        u = e.get_node1().get_name()
        v = e.get_node2().get_name()
        pc_edges.append((u, v))

    print("\nPC edges:")
    for e in pc_edges:
        print(e)

    ambiguous_sets = [
        {"Cancer", "Xray"},
        {"Cancer", "Dyspnoea"}
    ]

    fixed_edges = [
        (u, v) for (u, v) in pc_edges
        if {u, v} not in ambiguous_sets
    ]

    candidates = [
        fixed_edges + [("Cancer", "Xray"), ("Cancer", "Dyspnoea")],
        fixed_edges + [("Xray", "Cancer"), ("Dyspnoea", "Cancer")]
    ]

    scores = []
    for edges in candidates:
        try:
            scores.append(log_likelihood(data, edges))
        except Exception:
            scores.append(-np.inf)

    best_edges = candidates[np.argmax(scores)]

    print("\nFinal DAG edges (PC + likelihood):")
    for e in best_edges:
        print(e)

    adj_matrix = dag_to_adjacency_matrix(best_edges, nodes)

    print("\nAdjacency Matrix (rows → columns):")
    print(adj_matrix)

    adj_matrix.to_csv("pc_cancer_adjacency_matrix.csv")

    visualize_dag(
        best_edges,
        title="PC + Likelihood Causal DAG",
        save_path="pc_cancer_final_dag.png"
    )

    return best_edges, adj_matrix

if __name__ == "__main__":
    pc_plus_likelihood("data/cancer.bif")
