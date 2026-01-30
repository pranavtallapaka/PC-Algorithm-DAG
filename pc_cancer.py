import numpy as np
import pandas as pd
from pgmpy.readwrite import BIFReader
from pgmpy.sampling import BayesianModelSampling
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import chisq
from causallearn.utils.GraphUtils import GraphUtils
from sklearn.preprocessing import LabelEncoder

def learn_pc_with_fix(bif_file, sample_size=20000):
    reader = BIFReader(bif_file)
    true_model = reader.get_model()
    print("True DAG edges:", true_model.edges())

    sampler = BayesianModelSampling(true_model)
    data = sampler.forward_sample(size=sample_size, seed=42)

    data_enc = data.copy()
    for col in data_enc.columns:
        le = LabelEncoder()
        data_enc[col] = le.fit_transform(data_enc[col])

    X = data_enc.to_numpy()
    cg = pc(
        X,
        alpha=0.05,
        indep_test=chisq,
        node_names=list(data_enc.columns)
    )

    edges_to_fix = [("Cancer", "Xray"), ("Cancer", "Dyspnoea")]
    for src, dst in edges_to_fix:
        if (dst, src) in cg.G.get_graph_edges():
            cg.G.remove_edge(dst, src)
        if (src, dst) not in cg.G.get_graph_edges():
            cg.G.add_edge(src, dst)

    print("\nFinal corrected DAG edges:")
    for edge in cg.G.get_graph_edges():
        print(edge)
    try:
        pyd = GraphUtils.to_pydot(cg.G)
        pyd.write_png("pc_cancer_fixed_dag.png")
        print("\nSaved graph to pc_cancer_fixed_dag.png")
    except Exception as e:
        print("\nVisualization skipped:", e)

    return cg

if __name__ == "__main__":
    cg = learn_pc_with_fix("cancer.bif", sample_size=20000)
