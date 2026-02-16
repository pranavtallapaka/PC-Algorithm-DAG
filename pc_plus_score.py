import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import time

from pgmpy.readwrite import BIFReader
from pgmpy.sampling import BayesianModelSampling
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator

from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import chisq
from sklearn.preprocessing import LabelEncoder

def shd(dag_pred, dag_true):
    """
    Structural Hamming Distance:
    Counts number of edge additions, removals, and direction errors.
    """
    return int(np.sum(np.abs(dag_pred - dag_true)))


def log_likelihood(data, edges):
    model = BayesianNetwork(edges)
    model.fit(data, estimator=MaximumLikelihoodEstimator)
    return model.log_likelihood(data)


def edges_to_numpy_adj(edges, nodes):
    idx = {node: i for i, node in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)), dtype=np.int32)

    for u, v in edges:
        A[idx[u], idx[v]] = 1

    return A


def true_model_to_numpy_adj(model, nodes):
    idx = {node: i for i, node in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)), dtype=np.int32)

    for u, v in model.edges():
        A[idx[u], idx[v]] = 1

    return A


def visualize_dag(edges, title="Learned Causal DAG", save_path=None):
    print("→ Visualizing DAG (close window to continue)")

    G = nx.DiGraph()
    G.add_edges_from(edges)

    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(9, 7))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=2500,
        node_color="lightblue",
        font_size=9,
        arrowsize=20,
        arrowstyle="->"
    )
    plt.title(title)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def pc_plus_likelihood(bif_file, n=2000):

    print("\n===== STARTING PC + LIKELIHOOD PIPELINE =====")

    print("1️⃣ Loading BIF model")
    reader = BIFReader(bif_file)
    true_model = reader.get_model()
    nodes = list(true_model.nodes())
    print("Nodes:", nodes)

    print("2️⃣ Sampling data")
    sampler = BayesianModelSampling(true_model)
    data = sampler.forward_sample(size=n, seed=42)

    for c in data.columns:
        data[c] = LabelEncoder().fit_transform(data[c])

    print("Data shape:", data.shape)

    print("3️⃣ Running PC algorithm")
    start_time = time.time()

    cg = pc(
        data[nodes].to_numpy(),
        alpha=0.05,
        indep_test=chisq,
        node_names=nodes
    )

    print("PC completed in", round(time.time() - start_time, 2), "seconds")

    directed_edges = []
    undirected_edges = []

    for e in cg.G.get_graph_edges():
        u = e.get_node1().get_name()
        v = e.get_node2().get_name()

        ep1 = e.get_endpoint1().name
        ep2 = e.get_endpoint2().name

        if ep1 == "TAIL" and ep2 == "ARROW":
            directed_edges.append((u, v))
        elif ep1 == "ARROW" and ep2 == "TAIL":
            directed_edges.append((v, u))
        else:
            undirected_edges.append((u, v))

    print("Directed edges:", directed_edges)
    print("Undirected edges:", undirected_edges)

    print("4️⃣ Orienting undirected edges using likelihood")

    final_edges = directed_edges.copy()

    for (u, v) in undirected_edges:
        print(f"   Testing {u} <-> {v}")

        candidate1 = final_edges + [(u, v)]
        candidate2 = final_edges + [(v, u)]

        try:
            score1 = log_likelihood(data, candidate1)
        except:
            score1 = -np.inf

        try:
            score2 = log_likelihood(data, candidate2)
        except:
            score2 = -np.inf

        if score1 >= score2:
            final_edges.append((u, v))
            print(f"   Chose {u} -> {v}")
        else:
            final_edges.append((v, u))
            print(f"   Chose {v} -> {u}")

    print("Final edges:", final_edges)

    print("5️⃣ Computing SHD")

    pred_adj = edges_to_numpy_adj(final_edges, nodes)
    true_adj = true_model_to_numpy_adj(true_model, nodes)

    shd_value = shd(pred_adj, true_adj)

    print("\n===== EVALUATION RESULTS =====")
    print("SHD:", shd_value)

    print("6️⃣ Saving adjacency matrix")

    adj_df = pd.DataFrame(pred_adj, index=nodes, columns=nodes)
    adj_df.to_csv("pc_sachs_adjacency_matrix.csv")

    print("7️⃣ Visualizing final DAG")

    visualize_dag(
        final_edges,
        title="SACHS: PC + Likelihood DAG",
        save_path="pc_sachs_final_dag.png"
    )

    print("\n===== PIPELINE COMPLETE =====")

    return final_edges, pred_adj, true_adj

if __name__ == "__main__":
    pc_plus_likelihood("data/sachs.bif", n=2000)
