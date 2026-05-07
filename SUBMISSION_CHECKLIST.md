# Pre-submission checklist — NeurIPS 2026 D&B Track

Deadline: **May 6, 2026 AoE** (14 days from today, 2026-04-22).

## Content — already done ✓
- [x] Paper draft (`paper/main.tex`, 702 lines, 7 sections)
- [x] Bibliography (`paper/refs.bib`, 31 entries)
- [x] Datasheet (`paper/datasheet.md`, Gebru template, per-benchmark license table)
- [x] 5 publication-grade figures (PDF + PNG in `figures_v2/`)
- [x] 5 tables (CSV + TeX `\input{}`)
- [x] README.md (reproduction guide)
- [x] SCHEMA.md (on-disk format documentation)
- [x] CITATION.cff
- [x] requirements.txt (pinned versions)
- [x] Ground-truth Pythia-1B variants (5 main + 12 light-regime)
- [x] v1 and v2 ensemble pickles
- [x] Per-cell contamination scores under both calibrations
- [x] Statistical robustness (LOO-cell, LOO-model, BH-FDR, within-difficulty)
- [x] Responsible-release / Broader Impact expanded

## Must do before upload (cannot automate from this env)
- [ ] **Compile main.tex with pdflatex + bibtex** on a machine with TeX Live.
      The current `.tex` uses `neurips_2024.sty` as a placeholder; swap for
      `neurips_2026.sty` (or equivalent) once the template is available.
- [ ] **Skim the PDF** for overflow boxes, missing figures, wrong references.
- [ ] **Host artifacts on HuggingFace and/or Zenodo** and get a DOI. R5
      flagged this as the D&B-track bar gap. Repositories to create:
  - HF dataset for per-cell scores (v1 + v2 under one dataset, two
    configurations, with README referencing this paper)
  - HF model repo for each of the 5 main ground-truth Pythia-1B variants
  - GitHub public mirror of the code + a Zenodo snapshot with DOI
- [ ] **Update the paper** with the DOIs once obtained (two lines in §7 Release and in CITATION.cff).
- [ ] **Anonymize** — double-check no `.tex`, `.bib`, or released artifact
      metadata leaks identifying info (HF repo slugs, dataset uploader,
      API keys, /home/USER/ paths in logs).
- [ ] **Double-check license files** in the GitHub repo:
  - LICENSE (Apache-2.0 for code)
  - LICENSE.data (CC BY 4.0 for scores)
  - NOTICE (per-benchmark attributions)
- [ ] **File-tree sanity** — the submitted code archive should be
      self-contained and NOT include /local/scratch paths or HF tokens.
- [ ] **NeurIPS submission form** — checklist the track asks for
      (dataset card, broader-impact statement, reproducibility checklist).

## Nice-to-have (stretch, if time)
- [ ] LLM-as-judge validation on 100 flagged items (3 models as judges).
      Labeled HONESTLY as LLM-as-judge, not human eval.
- [ ] Add Gemma-2-27B or Llama-3-70B (quantized) to audit for breadth.
- [ ] Closed-API audit (Claude, GPT-4) via logprobs where available —
      unique value for D&B track but requires vendor API access.
- [ ] Pre-registration snapshot (before running any additional
      experiments) demonstrating that the uniform/variant-difficulty
      split was planned.

## Anticipated reviewer concerns NOT yet addressed
- R5: **DOI/persistent hosting** — hard blocker for D&B-track accept
  without it. Only you can action this.
- R1: **Control ensemble** — independent t-test / random shuffled
  negatives as an additional null. Could be done in ~30 min if
  reviewers push on it during rebuttal.
- R2: **Closed-API model audit** — reviewers may ask "why not Claude/
  GPT-4?" Answer in rebuttal: we scope to open-weight (paper §5.1,
  broader impact). Closed API audit requires logprobs generally not
  exposed.
- R3: **Other-benchmark negatives in prior work** — be prepared to
  name specific cases if asked. Our honest answer: the v1 setup is
  the natural default we stumbled into; we have not found a specific
  published pipeline that uses it as-described.

## Submission checklist items most-likely asked by NeurIPS form
- [x] Discussed limitations in a dedicated section
- [x] Discussed broader societal impact in a dedicated section
- [x] Justified potential harms and ethical considerations
- [x] Included hyperparameters for all experiments
- [x] Included error bars / confidence intervals / significance tests
- [ ] Compute infrastructure described (do in final camera-ready)
- [x] Stated if assets are released with license
- [x] Datasheet included
- [x] Used open-source code/models; cited appropriately
- [x] Discussed whether results use randomness; reported seeds (SCHEMA.md)

## Rebuttal-period preparation
A 1-page "response to reviewers" template in `paper/REBUTTAL_TEMPLATE.md`
would help when reviews arrive. I can draft one if you want.
