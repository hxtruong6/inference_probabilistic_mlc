# Software Impacts submission checklist (DaCaF)

Companion software metapaper for the *Information Fusion* (2026) paper
(DOI `10.1016/j.inffus.2026.104517`). Software Impacts reviews the **software**
(availability, documentation, reusability, license, impact), not the science.

Deliverables: (1) a ~2-3 page metapaper + C1-C9 metadata table, (2) a public,
permanently-archived, license-clear, reproducible repository.

---

## Phase 0 - Confirm the invitation
- [ ] Re-read the Information Fusion editor email: confirm Software Impacts (not SoftwareX),
      note any reference code / submission link / deadline, and whether they want the
      original-paper DOI cited as the associated research.

## Phase 1 - Artifact hardening
- [ ] Clean-clone test: fresh venv -> `pip install -e ".[dev]"` -> `pytest tests/ -q`
      -> `make reproduce` (CHD-49). All green from a clean checkout.
- [ ] Regenerate one clean result table from current code (CHD-49) so the metapaper
      numbers match the shipped code exactly. Replace/annotate stale `result/*.csv` and
      `docs/paper_tables.tex` (flagged in README as an older multi-model run).
- [ ] Reconcile the README off-diagonal Markedness/F-measure caveat: make sure the
      example used in the paper is current and reproducible (diagonal claim, ~1.8pp).
- [ ] Verify README quick-start (4-line install + 1-command reproduce) works as written.
- [ ] Verify version/DOI agree across `pyproject.toml` (1.0.0), `CITATION.cff`,
      README BibTeX, and the planned git tag.

## Phase 2 - Permanent archive (required) + capsule (optional)
- [ ] Tag release: `git tag v1.0.0` + push + GitHub Release.
- [ ] Archive on Zenodo (GitHub<->Zenodo integration) -> software DOI = metadata field C2.
      Optionally upload the ~200 MB pre-extracted feature `.npy` files here too.
- [ ] (Optional, rewarded) Code Ocean capsule running `reproduce_tabular.sh` on
      CHD-49 -> metadata field C3.

## Phase 3 - Write the metapaper (Elsevier Software Impacts LaTeX template)
- [ ] Download the official Software Impacts LaTeX template + metadata-table template.
- [x] Fill the C1-C9 code metadata table (support email C9 = hxtruong@jaist.ac.jp;
      Zenodo DOI still TODO into C2).
- [ ] Sec 1 Motivation and significance (~250 words).
- [ ] Sec 2 Software description: 2.1 Architecture, 2.2 Functionalities.
- [ ] Sec 3 Illustrative examples: library snippet + clean CHD-49 table.
- [ ] Sec 4 Impact (the section reviewers weight most, ~300-400 words).
- [ ] Sec 5 Conclusions + Acknowledgements + Conflict of interest + short references.
- [ ] Keep to ~1,300-1,500 words / within the 3-page budget.

## Phase 4 - Submit
- [ ] Submit via the Software Impacts Editorial Manager portal.
- [ ] Cover letter references the Information Fusion DOI as associated research +
      the editor invitation.
- [ ] Attach/declare: metapaper PDF + source, public GitHub repo, Zenodo DOI,
      Code Ocean capsule (if built).
- [ ] Required statements: license (MIT), data availability (ARFFs in repo; NIH via
      download script), author contributions, conflict of interest.

---

## NIH chest-X-ray data: already handled correctly
- The 5.8 GB of raw PNGs and the large `nih_feature_vectors_*.npy` files are **git-ignored**,
  not committed. Repo ships only the small ARFFs + the 832 KB derived label CSV, with
  `dacaf_mlc/chest_xray_dataset/Readme.md` giving the academic-torrents download link and
  exact regeneration steps. This is the right pattern; no change required.
- NIH ChestX-ray14 is public (NIH Clinical Center), so shipping the derived label CSV is fine.
- Reviewers cannot reproduce the chest-xray rows without a GPU + 5.8 GB download, so make the
  paper's illustrative example a **tabular** dataset (CHD-49) that runs in seconds; present
  chest-xray as a "scales to real images" demo.
- Optional: host the three feature `.npy` files (~200 MB) on Zenodo so the chest-xray path is
  reproducible without a GPU.

## Highest-leverage items
1. Zenodo DOI (blocks the metadata table; required).
2. One clean, reproducible CHD-49 table (paper numbers == code output).
3. A sharp Sec 4 Impact (the main thing reviewers judge).
