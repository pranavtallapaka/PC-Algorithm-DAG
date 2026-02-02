import numpy as np
import pandas as pd
from pgmpy.readwrite import BIFReader
from pgmpy.sampling import BayesianModelSampling
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import chisq
from causallearn.utils.GraphUtils import GraphUtils
from sklearn.preprocessing import LabelEncoder
import networkx as nx
import matplotlib.pyplot as plt


def learn_pc_only(bif_file, sample_size=20000):
    reader = BIFReader(bif_file)
    true_model = reader.get_model()
    print("True DAG edges:", true_model.edges())

    sampler = BayesianModelSampling(true_model)
    data = sampler.forward_sample(size=sample_size, seed=42)

    data_enc = data.copy()
    for col in data_enc.columns:
        data_enc[col] = LabelEncoder().fit_transform(data_enc[col])

    X = data_enc.to_numpy()
    cg = pc(
        X,
        alpha=0.05,
        indep_test=chisq,
        node_names=list(data_enc.columns)
    )

    print("\nPC Algorithm DAG edges (may have ambiguous directions):")
    for e in cg.G.get_graph_edges():
        print(f"{e.get_node1().get_name()} -> {e.get_node2().get_name()}")

    G = nx.DiGraph()
    nodes = list(data_enc.columns)
    G.add_nodes_from(nodes)
    for e in cg.G.get_graph_edges():
        G.add_edge(e.get_node1().get_name(), e.get_node2().get_name())

    plt.figure(figsize=(7, 5))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=2600,
            node_color="lightcoral", arrowsize=20)
    plt.title("PC Algorithm DAG (may have wrong directions)")
    plt.show()

    try:
        pyd = GraphUtils.to_pydot(cg.G)
        pyd.write_png("pc_cancer_dag.png")
        print("\nSaved graph to pc_cancer_dag.png")
    except Exception as e:
        print("\nVisualization via pydot skipped:", e)
        
    idx = {n: i for i, n in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)), dtype=int)
    for e in cg.G.get_graph_edges():
        u = e.get_node1().get_name()
        v = e.get_node2().get_name()
        A[idx[u], idx[v]] = 1

    df = pd.DataFrame(A, index=nodes, columns=nodes)
    print("\nAdjacency Matrix:")
    print(df)

    return cg


if __name__ == "__main__":
    cg = learn_pc_only("data/cancer.bif", sample_size=20000)
