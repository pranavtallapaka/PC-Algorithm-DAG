import numpy as np
from collections import Counter
from pgmpy.readwrite import BIFReader
from pgmpy.sampling import BayesianModelSampling
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import chisq
from sklearn.preprocessing import LabelEncoder

def pc_bootstrap(bif_file, n=3000, runs=50):
    model = BIFReader(bif_file).get_model()
    sampler = BayesianModelSampling(model)

    edge_votes = Counter()
    nodes = list(model.nodes())

    for i in range(runs):
        data = sampler.forward_sample(size=n, seed=i)
        for c in data.columns:
            data[c] = LabelEncoder().fit_transform(data[c])

        cg = pc(data.to_numpy(), alpha=0.05,
                indep_test=chisq, node_names=nodes)

        for u, v in cg.G.get_graph_edges():
            edge_votes[(u, v)] += 1

    print("\nBootstrap-stable edges:")
    for (u, v), cnt in edge_votes.items():
        if cnt > runs * 0.6:
            print(f"{u} -> {v} ({cnt}/{runs})")

if __name__ == "__main__":
    pc_bootstrap("data/cancer.bif")
