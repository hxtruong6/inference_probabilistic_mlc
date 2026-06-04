**Dataset:**
    + Water-quality:  ==== 🦈 Dataset evaluation time: 42101.88221 seconds ~ 12 hours

---

## Reproduction vs. paper discussion — summary (2026-05-19)

Source of comparison: `docs/paper_tables.tex` (auto-generated 7×7 OURS tables + Δ vs. paper tables + new-dataset tables + PCC-vs-BR head-to-head).

> Historical note: this analysis was written when the repo still shipped the Binary Relevance baseline and the extra (non-paper) datasets `flags` / `VirusGO` / `PlantPseAAC`. The final release keeps only the paper's protocol (PCC + LogReg on the paper datasets), so the BR and new-dataset sections below refer to runs no longer reproducible from this codebase; they are retained for provenance.

### Conclusion

The paper's main claim (matched-loss principle: optimizing the metric used for evaluation gives the best score on that metric) **still holds** in the reproduction, including on the 3 added datasets (Flags, VirusGO_sparse, PlantPseAAC). Two systematic off-diagonal differences exist and are explained — they do not invalidate the paper.

### Agreement with paper

1. **Matched-loss (diagonal) values** match paper within:
   - ~±1pp on Emotions, CHD-49, Scene, Water-quality, Yeast
   - ~±0.1pp on ChestX-ray8 (all three feature extractors: ResNet, resnetAE, Densenet)
   - Exception: F_mar diagonal (see below)
2. **F_pre exception on chest-xray** (paper: optimizing F_pre does NOT yield top F_pre on images). Reproduced. Example (ResNet): predict_F_pre → F_pre = 33.24, while predict_F_ham → F_pre = 50.86.
3. **Triplet (F_1, F_ham, F_sub) is the most stable across columns** — reproduced.
4. **New datasets (Flags, VirusGO_sparse, PlantPseAAC)**: diagonal is the top-3 entry of its column on essentially every (target, evaluation) pair → bolsters the paper's claim, doesn't contradict it.

### Systematic differences (and why)

| Difference | Root cause | Affects paper claim? |
|---|---|---|
| Entire F_mar column and F_mar target row deviate by tens of pp | `predict_markedness` was re-derived under sklearn `zero_division=0` convention (paper's original closed form assumed Precision=1 in the vacuous |ŷ|=0 case, which made the rule predict ŷ≈0 → near-zero F_1/F_pre/F_rec). New rule yields sensible labels. Diagonal F_mar slightly lower; off-diagonals (F_1, F_pre, F_rec, F_sub) much higher. | No. Diagonal is still highest in F_mar column. **Must be flagged as a correction in the paper revision.** |
| F_rec target × F_neg eval cell: Δ = −100pp | predict_recall outputs ŷ=1 always. Paper convention: vacuous NPV = 1. sklearn: vacuous NPV = 0. | No. Pure convention difference; already noted in caption of `tab:ours-delta`. |
| F_neg target × F_ham eval: Δ = +6–16pp, consistent sign | Suspect: optimal-NPV chooses min-marginal label → ŷ ≈ 1 vector → Hamming behaves differently from paper's implementation. To re-verify. | No. F_neg target was never claimed optimal for F_ham. |

### Bonus result (not in the paper)

`tab:pcc-vs-br` in `paper_tables.tex` — PCC vs. Binary Relevance head-to-head over 33 (dataset × base estimator) cells per metric:

- F_sub: PCC wins 25 / tie 0 / BR 8
- F_1: PCC wins 20 / tie 5 / BR 8
- F_ham: PCC 11 / tie 7 / BR 15 (essentially tied)
- Avg. Precision: PCC 16 / tie 2 / BR 15 (essentially tied)

Supports the paper's implicit thesis that chain-aware inference matters for structured losses but not for label-decomposable ones.

### TODO if pushing this into the paper revision

- Add a paragraph explaining the corrected `predict_markedness` derivation (sklearn convention) and the F_neg vs. F_rec convention; cite the Δ table.
- Decide whether to include the 3 new tabular datasets in the main results table or as an appendix.
- Decide whether to include the PCC-vs-BR head-to-head as a stand-alone subsection.
- Re-verify the F_neg/F_ham off-diagonal discrepancy (current explanation is a hypothesis).
