import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import logging
import networkx as nx

from pgmpy.readwrite import BIFReader
from pgmpy.sampling import BayesianModelSampling
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator

from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import chisq
from sklearn.preprocessing import LabelEncoder

logging.getLogger('pgmpy').setLevel(logging.ERROR)

def shd(pred_adj, true_adj):
    return int(np.sum(np.abs(pred_adj - true_adj)))

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

def log_likelihood(data, edges):
    model = BayesianNetwork(edges)
    model.fit(data, estimator=MaximumLikelihoodEstimator)
    return model.log_likelihood(data)

def plot_shd_progression(shd_history):

    plt.figure(figsize=(8, 5))
    plt.plot(shd_history)
    plt.xlabel("Orientation Step")
    plt.ylabel("Structural Hamming Distance (SHD)")
    plt.title("Hailfinder: SHD Progression During Likelihood Orientation")
    plt.grid(True)
    plt.savefig("hailfinder_shd_progression.png", dpi=300)
    plt.show()

def visualize_dag(edges, title="Final Learned DAG"):

    G = nx.DiGraph()
    G.add_edges_from(edges)

    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(12, 10))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=1000,
        node_color="lightblue",
        font_size=7,
        arrowsize=10,
        arrowstyle="->"
    )

    plt.title(title)
    plt.savefig("hailfinder_final_dag.png", dpi=300)
    plt.show()


def pc_plus_likelihood_hailfinder(bif_file, n=1000):

    print("\n===== PC + LIKELIHOOD ON HAILFINDER =====")

    print("1️⃣ Loading true model")
    reader = BIFReader(bif_file)
    true_model = reader.get_model()
    nodes = list(true_model.nodes())

    print("Number of nodes:", len(nodes))

    print("2️⃣ Sampling data")
    sampler = BayesianModelSampling(true_model)
    data = sampler.forward_sample(size=n, seed=42)

    for col in data.columns:
        data[col] = LabelEncoder().fit_transform(data[col])

    print("Data shape:", data.shape)

    print("3️⃣ Running PC")
    start = time.time()

    cg = pc(
        data[nodes].to_numpy(),
        alpha=0.01,
        indep_test=chisq,
        node_names=nodes,
        stable=True
    )

    print("PC finished in", round(time.time() - start, 2), "seconds")

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

    print("Directed edges from PC:", len(directed_edges))
    print("Undirected edges from PC:", len(undirected_edges))

    print("4️⃣ Computing SHD after pure PC")

    true_adj = true_model_to_numpy_adj(true_model, nodes)
    pc_only_adj = edges_to_numpy_adj(directed_edges, nodes)

    initial_shd = shd(pc_only_adj, true_adj)
    print("SHD after pure PC:", initial_shd)

    shd_history = [initial_shd]

    print("5️⃣ Orienting edges via likelihood")

    final_edges = directed_edges.copy()
    step = 0

    for (u, v) in undirected_edges:
        step += 1
        print(f"   Step {step}: testing {u} <-> {v}")

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
        else:
            final_edges.append((v, u))

        # Compute SHD at this step
        current_adj = edges_to_numpy_adj(final_edges, nodes)
        current_shd = shd(current_adj, true_adj)
        shd_history.append(current_shd)

        print("      SHD:", current_shd)

    print("Orientation complete.")

    print("6️⃣ Final SHD")

    final_adj = edges_to_numpy_adj(final_edges, nodes)
    final_shd = shd(final_adj, true_adj)

    print("\n===== RESULTS =====")
    print("Final SHD:", final_shd)

    print("7️⃣ Plotting SHD progression")
    plot_shd_progression(shd_history)

    print("8️⃣ Visualizing final DAG")
    visualize_dag(final_edges)

    print("\n===== DONE =====")

    return final_edges, final_adj

if __name__ == "__main__":
    pc_plus_likelihood_hailfinder("data/hailfinder.bif", n=1000)
