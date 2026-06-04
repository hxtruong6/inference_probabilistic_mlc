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
| Recall when the **true** set is empty | 1 | Recall |

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

These are the seven metrics that make up the paper's target × evaluation table
(Hamming, Subset 0/1, Precision, NPV, Recall, Markedness, F-β). Each `predict_*`
rule is verified against brute-force enumeration in
`tests/test_inference_optimality.py`.
