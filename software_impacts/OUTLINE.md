# DaCaF — Software Impacts metapaper: outline & writing plan

Working plan for `dacaf_software_impacts.tex`. Reference exemplar: BOOMER (`boomer_reference.md`),
a *published* Software Impacts OSP. Content source: the journal paper `../papers/dacaf.tex`
(Information Fusion 2026). This metapaper is a **software-framed condensation** of that paper —
no new research, just the software story.

## Workflow
- We write **one section at a time**.
- **You draft first → send to me → I review & enhance** (prose, BOOMER voice, tightness, accuracy).
- Citations: I add only **DOI-verified** references (web/API), never from memory.

## Constraints (Guide for Authors)
- ~**3 pages** total, two-column `cas-dc`, including Impact Overview + references.
- Abstract ≤ 250 words. Highlights: 3–5 bullets, ≤ 85 chars each (also a separate file).
- Keywords 1–7 (Guide prefers short single terms).
- Required: CRediT per author, competing-interest declaration, GenAI declaration.
- **Body prose budget ≈ 1,400–1,500 words** (see per-section budgets below).

## Structural decision
Keep the **official 5-section** Software Impacts structure (Motivation → Software description →
Illustrative examples → Impact → Conclusions). BOOMER uses only 3 sections because it predates
the current template — so we **borrow BOOMER's prose voice, not its section count**.

### BOOMER voice = what "good" means here
- Finished **paragraphs**, not bullet talking points.
- **Declarative, confident, no hedging** ("We introduce…", "A key functionality is…").
- **Citations woven into sentences**, each doing argumentative work.
- Feature lists led by **bold term + explanation**.
- Examples/results prose tells the reader **what to observe** ("the X shows Y").

---

## Front matter

| Element | Plan | Source in `dacaf.tex` | Status |
|---|---|---|---|
| Title | "DaCaF: Bayes-optimal per-metric inference for probabilistic multi-label classification" | l.163 | done |
| Abstract (~150 w) | Condense journal abstract: metrics conflict → probabilistic MLC enables per-metric optimization → DaCaF = divide-and-conquer + fusion → ships 7 verified `predict_*` rules, CLI, reproducible eval | l.193–200 | TODO write |
| Highlights (3–5) | Align with journal highlights | l.276–281 | draft exists |
| Keywords | Trim to short terms | l.284 | revise |
| Code metadata table | **Fix stale lines** (see "Must-fix" below) | — | stale |

---

## Body sections

### 1. Motivation and significance  — ~280 words
BOOMER analogue: Introduction. Source: `dacaf.tex` l.301–320, Primary Goals l.474.
Narrative arc:
1. MLC + real applications (each instance → any subset of labels).
2. The hard fact: **different metrics demand different Bayes-optimal predictions (BOPs)**; one model can't be optimal for all.
3. Naive BOP is intractable (2^L candidate predictions).
4. Gap: prior tools cover one metric / assume label independence / aren't released.
5. "We introduce DaCaF…" — generic recipe + reference implementation returning the exact BOP for a chosen metric, given any P(y|x).
6. Significance: optimize the metric you actually care about; study metric-mismatch *without approximation noise*.

### 2. Software description  — ~450 words (two subsections)
BOOMER analogue: Technical overview. Source: method l.485–1026.

**2.1 Software architecture** — lettered components like BOOMER's (a)(b)(c):
- (a) Installable `dacaf_mlc` package; `ProbabilisticClassifierChain` (PCC) wraps any scikit-learn estimator, estimates P(y|x) via a classifier chain.
- (b) The two algorithmic blocks behind every rule:
  - **Divide-and-conquer**: partition 2^L predictions into L+1 groups by label count; best-in-group by sorting; global best = best across groups.
  - **Fusion**: estimate marginal/pairwise probabilities by fusing the chain's dependent binary classifiers.
- (c) Supporting modules: `evaluation_metrics`, `metrics_registry`, `pipeline`, `evaluate` (CLI), ARFF/MULAN loaders, `chest_xray_dataset` (NIH features, [image] extra); vendored skmultiflow base.

**2.2 Software functionalities** — bold-led feature list:
- **Seven per-metric BOP rules** (`predict_fmeasure/hamming/markedness/precision/npv/recall/subset`).
- **Trivial-optimum diagnostic** — flags metrics whose optimum is degenerate.
- **Batched predictor** — verified numerically equal to brute-force enumeration.
- **Reproducible evaluation** — CLI + scripts over MULAN tabular + NIH ChestX-ray.
- **Correctness harness + CI** (Python 3.10/3.12).

### 3. Illustrative examples  — ~200 words + code listing + table
BOOMER analogue: (folded into Technical overview / Impact). Source: Experiments l.1318–1371; CHD-49 table.
- Minimal library snippet: fit once → query any metric's BOP (already in skeleton).
- **CHD-49 cross-tab table**: rows = target metric, cols = eval metric; bold diagonal = column max. Prose tells reader what to observe: optimizing the evaluated metric wins its column; NPV & Recall collapse to the predict-all trivial optimum.
- One command reproduces it: `make reproduce`.

### 4. Impact  — ~350 words (reviewers weight this most)
BOOMER analogue: Impact (two paragraphs). Source: l.314, Applications l.1027, Conclusion l.1535.
- ¶1: enables an **exact** metric-mismatch study (no approximation blurring); model-agnostic BOP layer on any P(y|x); covers two whole metric families; MIT license → reuse in clinical/image MLC (NIH ChestX-ray demo), benchmarking, teaching Bayes-optimal decision-making.
- ¶2: forward look — more base models, more metric families, optional approximate fast paths; low barrier (pip + Docker + scripted reproduction).

### 5. Conclusions  — ~60 words
Source: Conclusion l.1535. 2–3 sentences: DaCaF turns a proved recipe into a reusable, verified, exact per-metric inference tool; future directions.

---

## Declarations + references
- **Competing interest** — state plainly (not TODO).
- **GenAI declaration** — standard Guide wording, or remove if unused.
- **Acknowledgements** — funding/grant numbers, or "none".
- **References** (keep short, ~6–8): the **software (Zenodo)** + the **journal article** + foundational refs
  (Dembczyński PCC 2010, Dembczyński F-measure 2011, Waegeman 2014, MULAN/Tsoumakas 2010). DOI-verified only.

## Must-fix metadata (independent of prose)
| Line in `.tex` | Currently | Fix to |
|---|---|---|
| C1 | v1.0.0 | **v1.1.0** |
| C2 Zenodo | TODO software DOI | **10.5281/zenodo.20572638** |
| C3 capsule | TODO Code Ocean | **https://codeocean.com/capsule/1580907/tree** |
| bib `{software}` | Zenodo, TODO DOI | real Zenodo DOI |
| C9 / contacts | hxtruong@jaist.ac.jp | confirm vs CITATION.cff |

## Page/word budget summary
| Part | Budget |
|---|---|
| Abstract | ~150 w |
| 1 Motivation | ~280 w |
| 2 Software description | ~450 w |
| 3 Illustrative examples | ~200 w + listing + table |
| 4 Impact | ~350 w |
| 5 Conclusions | ~60 w |
| **Body total** | **~1,490 w (≈3 pp)** |
