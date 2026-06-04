# Metric & inference conventions

The published paper (Information Fusion 2026, doi:10.1016/j.inffus.2026.104517)
is the source of truth. Every inference rule returns the prediction that
maximises the **expected** value of its target metric, and is verified against
brute-force enumeration over all `2^L` candidates in
`tests/test_inference_optimality.py`.

## Vacuous-case conventions (important and intentionally asymmetric)

Some metrics are undefined for boundary predictions. The paper assigns:

| Situation | Value | Used by |
|---|---|---|
| Precision with an **empty** prediction `Fpre(y, 0_K)` | **1** | Markedness (internal) |
| NPV with an **all-ones** prediction `Fneg(y, 1_K)` | **1** | NPV metric, Markedness (internal) |
| Recall when the **true** set is empty | 1 | Recall, Informedness |
| Specificity when the **true** set is full | 1 | Informedness |

Note the asymmetry: the **standalone** `precision_score` keeps the scikit-learn
convention (precision = 0 when nothing is predicted but true positives exist),
whereas the **markedness-internal** precision uses the vacuous `= 1`. This is
exactly what reproduces the published result tables (e.g. the `F_neg` and
`F_rec` rows are identical; the NPV/Recall diagonals are 100; the Markedness
diagonal is high). Changing the standalone precision to vacuous=1 would *not*
reproduce the paper.

## BOP summary

| Metric | Bayes-optimal prediction | Needs |
|---|---|---|
| Hamming | `ŷ_j = 1 ⇔ p_j > ½` | marginals |
| Subset 0/1 | most probable label vector | full joint |
| F-β | sort by `q^β`, best prefix (Algorithm 1) | pairwise `P(y_k=1,|y|=s)` |
| Markedness | rank by marginals, compare sizes (Algorithm 4) | marginals |
| Precision | single top-marginal label | marginals |
| NPV | all-ones `1_K` (Corollary 2) | — |
| Recall | all-ones `1_K` | — |
| Specificity | all-zeros `0_K` | — |
| Informedness | per-label threshold (see below) | pairwise |

## Informedness — the paper appendix is wrong; use the per-label rule

The manuscript's appendix derives a size-`l` informedness rule and claims the
expected score is monotone in `l`, so that only `{0_K, ŷ^{L-1}, 1_K}` need be
compared. **That claim is false** (brute force finds size-2 predictions that
beat `ŷ^{L-1}`), which is why Informedness was dropped from the published main
text. The correct exact BOP is a **per-label threshold**:

```
include label k  ⇔  q_sens[k] + q_spec_cost[k] > C
  q_sens[k]      = Σ_{s=1}^L     P(y_k=1, |y|=s) / s
  q_spec_cost[k] = Σ_{s=1}^{L-1} P(y_k=1, |y|=s) / (L-s)
  C              = P(|y|=0)/L + Σ_{s=1}^{L-1} P(|y|=s) / (L-s)
```

This is derived from `E[F_Inf(y,ŷ)] = const + ½ Σ_{k∈ŷ}(q^Inf_k − β_k)` and is
verified optimal in `tests/test_inference_optimality.py::test_informedness_optimal`.
The corresponding metric is the example-based `½(Specificity + Recall)`.
