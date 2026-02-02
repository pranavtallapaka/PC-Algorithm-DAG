# PC-Algorithm-DAG

Causal structure learning with the **PC algorithm**, supplemented by **score-based orientation** (plus-score) to overcome limitations of PC used alone.

## Why the PC Algorithm Alone Is Not Enough

The **PC algorithm** is a constraint-based method that learns a causal graph by testing conditional independences in the data. It has two main limitations:

1. **It does not uniquely orient edges.**  
   PC returns a *Markov equivalence class* (a CPDAG): some edges are directed, but others remain undirected because the conditional independence tests cannot distinguish between \(X \to Y\) and \(Y \to X\). So PC alone does not give you a single DAG—you get ambiguous edges that need to be resolved some other way.

2. **It is sensitive to finite samples and noise.**  
   PC relies on a sequence of independence tests. With limited or noisy data, tests can be wrong (false positives/negatives), and the order in which variables are processed can change the result. So a single run of PC on one dataset can be unstable and may not reflect the true structure.

Because of this, we supplement PC with a **plus-score** approach: use a likelihood score to orient the ambiguous edges that PC leaves undirected.

---

## PC with Manual Fixes (`pc_cancer.py`)

This script runs PC on the cancer network and then **manually** orients the ambiguous edges (Cancer→Xray, Cancer→Dyspnoea) to illustrate that PC by itself does not determine a unique DAG—you have to add extra information (here, by hand; in `pc_plus_score.py`, by a likelihood score).

**Usage:**
```bash
python pc_cancer.py
```

Outputs: corrected DAG edges and a graph image (`pc_cancer_fixed_dag.png`).

---

## PC + Plus-Score (`pc_plus_score.py`)

**Idea:** Use PC to get the skeleton and orient everything it can. For edges that PC leaves undirected (e.g. Cancer–Xray, Cancer–Dyspnoea), try both possible orientations and choose the one that fits the data best (here, via **log-likelihood**).

**Why it helps:**  
- PC alone cannot pick a single DAG when edges are in the same equivalence class.  
- Score-based comparison uses the data to choose among the remaining orientations, so you get a single, well-defined DAG that is consistent with both the PC skeleton and the data.

**Usage:**
```bash
python pc_plus_score.py
```

Outputs: final DAG edges, adjacency matrix (`pc_cancer_adjacency_matrix.csv`), and a plot (`pc_cancer_final_dag.png`).

---

## Project Structure

| File | Description |
|------|-------------|
| `data/cancer.bif` | Cancer network (Bayesian network) in BIF format |
| `pc_cancer.py` | PC only + manual orientation of ambiguous edges |
| `pc_plus_score.py` | PC + likelihood score to orient ambiguous edges and output DAG/adjacency/plot |

## Dependencies

- `numpy`, `pandas`
- `pgmpy` (BIF reading, sampling, Bayesian network fitting)
- `causallearn` (PC algorithm, chi-squared independence test)
- `scikit-learn` (LabelEncoder)
- `networkx`, `matplotlib` (for `pc_plus_score.py` visualization)

Install with:
```bash
pip install numpy pandas pgmpy causallearn scikit-learn networkx matplotlib
```

## Summary

| Approach | Addresses |
|----------|-----------|
| **PC alone** (manual fix in `pc_cancer.py`) | Illustrates that PC gives an equivalence class; ambiguous edges are fixed by hand. |
| **PC + plus-score** (`pc_plus_score.py`) | Orientation: which way to direct ambiguous edges using a likelihood score, yielding a single data-driven DAG. |

Plus-score supplements PC by using the data to resolve ambiguous orientations and produce a single, well-defined DAG instead of an unresolved equivalence class.
