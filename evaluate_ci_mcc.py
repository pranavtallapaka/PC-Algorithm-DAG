from pc_plus_score import pc_plus_likelihood
from ci_mcc_evaluation import ci_mcc

# Run PC + Likelihood
edges, pred_adj, true_adj = pc_plus_likelihood("data/sachs.bif", n=2000)

print("\nNow computing CI_MCC...")
score = ci_mcc(pred_adj, true_adj)

print("\nCI_MCC:", score)
