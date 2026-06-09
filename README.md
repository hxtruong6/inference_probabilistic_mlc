# Probabilistic Multi-Label Classification via Divide-and-Conquer and Fusion (DaCaF)

[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.inffus.2026.104517-blue)](https://doi.org/10.1016/j.inffus.2026.104517)
[![Journal](https://img.shields.io/badge/Information%20Fusion-2026-green)](https://doi.org/10.1016/j.inffus.2026.104517)
[![Software DOI](https://img.shields.io/badge/Software%20DOI-10.5281%2Fzenodo.20572637-blue)](https://doi.org/10.5281/zenodo.20572637)
[![Code Ocean](https://img.shields.io/badge/Code%20Ocean-Reproduce-blue?logo=codeocean)](https://doi.org/10.24433/CO.1580907.v1)
[![PyPI](https://img.shields.io/pypi/v/dacaf-mlc?logo=pypi&logoColor=white)](https://pypi.org/project/dacaf-mlc/)
[![Python](https://img.shields.io/pypi/pyversions/dacaf-mlc)](https://pypi.org/project/dacaf-mlc/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

Official code for the paper **published in _Information Fusion_ (2026)**:

> **Probabilistic multi-label classification via a divide-and-conquer and fusion approach**
> Vu-Linh Nguyen, Xuan-Truong Hoang, Anh Hoang, Van-Nam Huynh.
> *Information Fusion*, 2026, Article 104517. <https://doi.org/10.1016/j.inffus.2026.104517>

---

## What is this about? (in one picture)

In multi-label classification, each instance can carry *any subset* of the labels, and **different evaluation metrics want different predictions**. A model that is great for one metric (e.g. F₁) can be poor for another (e.g. subset accuracy).

**DaCaF** is a generic recipe that, given a probabilistic model `P(y | x)`, finds the **Bayes-optimal prediction (BOP)** for a chosen metric: the prediction `ŷ` that maximises the *expected* score of that metric.

```text
              Training data
                   │ learn
                   ▼
        ┌──────────────────────────┐
        │  Probabilistic classifier │   PCC estimates P(y | x)
        │  chain (PCC)              │
        └──────────────────────────┘
                   │ inference, per instance x
                   ▼
          P(y | x)   (probabilistic prediction)
                   │
                   ▼
        ╔═════════════ DaCaF ══════════════╗
        ║  1. DIVIDE & CONQUER             ║
        ║     split the 2^L predictions    ║
        ║     into L+1 groups by #labels;  ║
        ║     solve each group by sorting  ║
        ║                                  ║
        ║  2. FUSION                       ║
        ║     fuse the chain's binary      ║
        ║     classifiers (ancestral       ║
        ║     sampling) to supply the      ║
        ║     marginal / pairwise probs    ║
        ╚══════════════════════════════════╝
                   │
                   ▼
          ŷ = Bayes-optimal prediction
              for the chosen metric
```

**Two building blocks:**

1. **Divide & Conquer**: partition the `2^L` possible predictions into `L+1` groups (by how many labels are predicted relevant). Within each group the best prediction is found just by **sorting labels by a score**; the global best is the best across groups.
2. **Fusion**: the final step that produces the prediction. The sorting scores need certain marginal/pairwise probabilities, which are supplied by **fusing the predictions of the dependent binary classifiers** that make up the chain (via ancestral sampling).

The paper proves this works for **two whole families of metrics** (so it covers many metrics at once, not one at a time) and shows when a metric's optimal prediction is *trivial*, a useful warning sign when choosing a metric.

---

## Results at a glance

**The headline finding: mismatch hurts.** When you evaluate with metric *E* but optimise for a different metric *T* during prediction, performance usually drops. Optimising the metric you actually care about is (almost always) best — verified on 5 tabular datasets plus a chest-X-ray image dataset, using the *exact* computation paradigm (no approximation blurring the picture).

Read the per-dataset table **column by column**: each column is an evaluation metric, each row the metric you optimised for, and the **diagonal** (optimise the metric you evaluate) is the largest value in its column. On CHD-49, all **7 of 7** columns confirm this.

📊 **Full results, datasets, and the target × evaluation table** → see the [reproduction guide](https://github.com/hxtruong6/inference_probabilistic_mlc/blob/main/docs/REPRODUCING.md).

---

## Install

To **use DaCaF as a library**, install the released package from PyPI:

```bash
pip install dacaf-mlc          # core (tabular); add "dacaf-mlc[image]" for the ChestX-ray experiments
```

To **reproduce the paper** (datasets, sweeps, lockfile), use the editable install from a clone below.

## Quickstart (one run)

Using [**uv**](https://docs.astral.sh/uv/) (recommended, fast; a checked-in `uv.lock` pins exact versions):

```bash
uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install -e .          # core (tabular) deps; add ".[image]" for the ChestX-ray experiments
# reproducible install from the lockfile instead: uv sync            (add --extra image for ChestX-ray)

# one (dataset, seed) run:
dacaf-mlc --dataset emotions --seed 1 --output-dir result
# or without activating a venv: uv run dacaf-mlc --dataset emotions --seed 1 --output-dir result
```

<details><summary>Alternative: plain pip / conda</summary>

```bash
python -m venv .venv && source .venv/bin/activate    # or conda create -n dacaf python=3.10
pip install -e .            # core (tabular) deps; add ".[image]" for the ChestX-ray experiments
dacaf-mlc --dataset emotions --seed 1 --output-dir result
```
</details>

This writes `result/emotions/seed1_all.csv` and a cross-tab of **target metric × evaluation metric**, the table at the heart of the paper.

---

## The metrics and their optimal predictions

For a probabilistic prediction `P(y | x)` over `L` labels, each rule returns the prediction that maximises the expected metric. `pⱼ = P(yⱼ = 1 | x)` is the marginal.

**How to read the columns:** *Needs* is the probabilistic information the rule consumes (cheap **marginals** `pⱼ`, the harder **pairwise** terms, or the full joint). *Cost* is the per-instance time once that information is available. Rules marked *trivial* / *near-trivial* have a BOP you can write down without looking at any data.

| Metric | Optimal prediction (BOP) | Needs | Cost |
|---|---|---|---|
| **Hamming** | `ŷⱼ = 1 ⇔ pⱼ > ½` | marginals | `O(L)` |
| **Subset 0/1** | the single most probable label vector | full joint | intractable |
| **F-β / F₁** | sort by an F-score, pick best prefix size | pairwise `P(yⱼ=1, |y|=s)` | `O(L³)` |
| **Markedness** | rank by marginals, compare prefix sizes | marginals | `O(L log L)` |
| **Precision** | predict only the top-marginal label | marginals | `O(L)` *(near-trivial)* |
| **NPV** | predict all ones `1…1` (same BOP as Recall here); falls back to `ŷ^{K-1}` (all ones but the lowest-marginal label) only if `1…1` is disallowed | marginals | `O(L)` *(near-trivial)* |
| **Recall** | always predict `1…1` | none | trivial |
| **Specificity** | always predict `0…0` | none | trivial |

> **Why "trivial" matters:** Recall/Specificity (and near-trivial Precision/NPV) have optimal predictions you can write down *without looking at any data*. The paper argues such metrics are weak *standalone* evaluation metrics, a practical takeaway when designing a metric for a new domain.

---

## Library usage

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from dacaf_mlc.probability_classifier_chains import ProbabilisticClassifierChain
from dacaf_mlc.evaluation_metrics import EvaluationMetrics as EM

# toy multi-label data: 200 instances, 8 features, 4 labels (Y is (n, L) binary)
rng = np.random.default_rng(0)
X = rng.normal(size=(200, 8))
Y = (rng.random((200, 4)) > 0.5).astype(int)
X_train, Y_train, X_test, Y_test = X[:150], Y[:150], X[150:], Y[150:]

pcc = ProbabilisticClassifierChain(LogisticRegression(max_iter=10_000))
pcc.fit(X_train, Y_train)

y_f1  = pcc.predict_fmeasure(X_test, beta=1)   # Bayes-optimal for F1
y_ham = pcc.predict_hamming(X_test)            # ... for Hamming
y_mar = pcc.predict_markedness(X_test)         # ... for Markedness

print("F1:        ", EM.f_beta(Y_test, y_f1))
print("Hamming:   ", EM.hamming_accuracy(Y_test, y_ham))
print("Markedness:", EM.markedness(Y_test, y_mar))
```

Every `predict_*` rule returns the prediction that maximises the expected value of its
target metric (see [`docs/CONVENTIONS.md`](https://github.com/hxtruong6/inference_probabilistic_mlc/blob/main/docs/CONVENTIONS.md) for the exact rules and conventions).

**Extending DaCaF** — adding a new evaluation metric, a new Bayes-optimal target,
or a new dataset is each a small, registry-based change; see the
[extension guide](https://github.com/hxtruong6/inference_probabilistic_mlc/blob/main/docs/EXTENDING.md).

---

## Reproducing the paper

The full experimental protocol — the 6 datasets, the `make reproduce` command, the
cluster sweep, the online Code Ocean capsule, the complete target × evaluation results
table, the repository layout, and the test suite — lives in the **[reproduction guide](https://github.com/hxtruong6/inference_probabilistic_mlc/blob/main/docs/REPRODUCING.md)**.

In short: `make reproduce` runs the tabular datasets and aggregates the crosstabs, and
every inference rule is checked against brute-force enumeration of the expected metric
(`python -m pytest tests/ -v`).

---

## How to cite

```bibtex
@article{nguyen2026probabilistic,
  title   = {Probabilistic multi-label classification via a divide-and-conquer and fusion approach},
  author  = {Nguyen, Vu-Linh and Hoang, Xuan-Truong and Hoang, Anh and Huynh, Van-Nam},
  journal = {Information Fusion},
  year    = {2026},
  pages   = {104517},
  issn    = {1566-2535},
  doi     = {10.1016/j.inffus.2026.104517}
}
```

---

## References

- K. Dembczyński, W. Cheng, E. Hüllermeier. [*Bayes Optimal Multilabel Classification via Probabilistic Classifier Chains.*](https://dl.acm.org/doi/10.5555/3104322.3104359) ICML 2010.
- K. Dembczyński, W. Waegeman, W. Cheng, E. Hüllermeier. [*An Exact Algorithm for F-Measure Maximization.*](https://proceedings.neurips.cc/paper/2011/hash/71ad16ad2c4d81f348082ff6c4b20768-Abstract.html) NeurIPS 2011.
- W. Waegeman, K. Dembczyński, A. Jachnik, W. Cheng, E. Hüllermeier. [*On the Bayes-Optimality of F-Measure Maximizers.*](https://jmlr.org/papers/v15/waegeman14a.html) JMLR 2014.
- D. M. W. Powers. [*Evaluation: From Precision, Recall and F-Measure to ROC, Informedness, Markedness & Correlation.*](https://arxiv.org/abs/2010.16061) 2011.
- G. Tsoumakas, I. Katakis, I. Vlahavas. [*Mining Multi-label Data.*](https://doi.org/10.1007/978-0-387-09823-4_34) 2010 (MULAN).

## Acknowledgements

The `dacaf_mlc/skmultiflow/` directory contains a trimmed, vendored subset of
[scikit-multiflow](https://github.com/scikit-multiflow/scikit-multiflow)
(the `ClassifierChain` base and its supporting utilities), redistributed under
its original 3-clause BSD license. See
[`dacaf_mlc/skmultiflow/LICENSE`](dacaf_mlc/skmultiflow/LICENSE) for the full text.

## License

MIT for the original DaCaF code, see [LICENSE](LICENSE). Vendored third-party
code retains its own license as noted in Acknowledgements above.
