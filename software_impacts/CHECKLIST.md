# Software Impacts submission checklist (DaCaF)

Companion **Original Software Publication (OSP)** for the *Information Fusion* (2026)
paper (DOI `10.1016/j.inffus.2026.104517`). Software Impacts reviews the **software**
(availability, documentation, reusability, license, impact) via the already-published
results, not the science. Single-anonymized peer review.

Deliverables: (1) a ~3-page metapaper + C1-C9 metadata table + Impact Overview,
(2) a public, permanently-archived, OSI-licensed, reproducible repository (CodeOcean
verifies reproducibility and awards a Reproducibility Badge).

Questions to: software.impacts@elsevier.com. Submit via Editorial Manager,
**Article Type = "Original Software"**.

---

## Phase 0 - Confirm the invitation
- [ ] Re-read the Information Fusion editor email: confirm Software Impacts (not SoftwareX),
      note any reference code / submission link / deadline.
- [ ] Corresponding author = Xuan-Truong Hoang (JAIST). Only the corresponding author's
      affiliation sets open-access APC / agreement eligibility, so check whether JAIST has
      an Elsevier open-access agreement that covers the APC.

## Phase 1 - Artifact hardening
- [x] Clean-clone test: fresh venv -> `pip install -e ".[dev]"` -> `pytest tests/ -q`
      (42 passed) -> `dacaf-mlc --dataset CHD_49 ...` (CLI run OK, CSV written).
      Verified 2026-06-05 from a clean clone of this branch.
- [x] Regenerate one clean result table from current code (CHD-49, PCC + LR, 5 seeds x
      10-fold). Saved to `chd49_crosstab_regenerated.csv`; all 7 diagonals are the column
      max (paper claim holds); NPV/Recall show the predict-all trivial optimum.
      Table embedded in the metapaper (Table 1 there).
- [ ] Decide whether to also refresh the repo's stale `result/*.csv` and
      `docs/paper_tables.tex` (older multi-model run) or leave the README caveat in place.
- [ ] Verify README quick-start (4-line install + 1-command reproduce) works as written.
- [ ] Verify version/DOI agree across `pyproject.toml` (1.0.0), `CITATION.cff`,
      README BibTeX, and the planned git tag.

## Phase 2 - Permanent archive (required) + CodeOcean (verification)
- [ ] Tag release: `git tag v1.0.0` + push + GitHub Release.
- [ ] Archive on Zenodo (GitHub<->Zenodo integration) -> software DOI = metadata field C2.
      Optionally upload the ~200 MB pre-extracted feature `.npy` files here too.
- [ ] CodeOcean capsule running `reproduce_tabular.sh` on CHD-49 -> metadata field C3.
      (CodeOcean is the journal's reproducibility-certification platform; capsule earns
      the Reproducibility Badge.)

## Phase 3 - Write the metapaper (official Elsevier CAS template, already wired up)
- [x] Use the official template: `els-cas/cas-dc.cls` (double column), copied from
      els-cas-templates.zip v2.4. Skeleton = `dacaf_software_impacts.tex` (compiles).
- [x] Fill the C1-C9 code metadata table (C9 = hxtruong@jaist.ac.jp; C2 Zenodo DOI TODO).
- [ ] Abstract (<= 250 words).
- [ ] Sec 1 Motivation and significance.
- [ ] Sec 2 Software description (2.1 Architecture, 2.2 Functionalities).
- [ ] Sec 3 Illustrative examples (snippet + Table 1, already inserted).
- [ ] Sec 4 Impact / Impact Overview (the section reviewers weight most).
- [ ] Sec 5 Conclusions.
- [ ] Fill `\credit{}` CRediT roles for all 4 authors; set affiliations + ORCIDs.
- [x] Highlights: 4 bullets, all <= 85 chars, in `highlights.txt` (separate submission file)
      and mirrored in the metapaper `highlights` environment.
- [ ] Declaration of competing interest (complete the declarations tool too).
- [ ] **Declaration of generative AI use**: required because AI tooling (Claude Code) was
      used to help prepare this manuscript/skeleton. Add the Guide's standard statement in
      the dedicated section before References, or remove if ultimately unused.
- [ ] Acknowledgements / funding (use the Guide's standard wording, or state none).
- [ ] Keep to ~3 pages including references (skeleton is 4 pages with verbose bullets;
      prose will tighten it).

## Phase 4 - Submit
- [ ] Submit via Editorial Manager; select Article Type = "Original Software".
- [ ] Upload editable sources (.tex + els-cas class files + figures/tables), not just PDF.
- [ ] Separate files: highlights file; competing-interest declaration (.doc/.docx).
- [ ] Cover letter references the Information Fusion DOI as the associated peer-reviewed
      research where the software's results were published.
- [ ] Data statement (encouraged): datasets bundled (MULAN ARFFs) or fetched via script
      (NIH ChestX-ray14); code at GitHub + Zenodo DOI.
- [ ] License confirmation: MIT (OSI-approved). Software is freely available.

---

## NIH chest-X-ray data: already handled correctly
- The 5.8 GB of raw PNGs and the large `nih_feature_vectors_*.npy` files are **git-ignored**,
  not committed. Repo ships only the small ARFFs + the 832 KB derived label CSV, with
  `dacaf_mlc/chest_xray_dataset/Readme.md` giving the academic-torrents download link and
  exact regeneration steps. This is the right pattern; no change required.
- NIH ChestX-ray14 is public (NIH Clinical Center), so shipping the derived label CSV is fine.
- Reviewers cannot reproduce the chest-xray rows without a GPU + 5.8 GB download, so the
  paper's illustrative example uses a **tabular** dataset (CHD-49) that runs in seconds;
  chest-xray is a "scales to real images" demo.
- Optional: host the three feature `.npy` files (~200 MB) on Zenodo so the chest-xray path
  is reproducible without a GPU.

## Highest-leverage items
1. Zenodo DOI (blocks the metadata table; required).
2. CodeOcean capsule (CHD-49) for the Reproducibility Badge.
3. A sharp Sec 4 Impact Overview (the main thing reviewers judge).
