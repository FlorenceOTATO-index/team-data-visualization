# Contributing to team-data-visualization

SJSU FA26 DATA-230 (Section 11) team project — exploratory data analysis of US
motor-vehicle accidents, a Tableau dashboard, and an ML direction grounded in
the EDA.

**Team:** Jue Wang · Saida Mahmood · Tzu-Yang Huang · Nino Pelko

## Workflow

1. **One branch per task**, named `<initials>/<short-topic>` — e.g.
   `jue/reorganize`, `saida/cleaned-parquet`. Branch off `main`, keep it
   focused.
2. **Everything merges via pull request.** No direct pushes to `main`.
3. **Request a teammate as reviewer** on every PR.
4. **Squash-and-merge**, so `main` stays a clean one-commit-per-PR history.
5. **Verify before requesting review.** Run the relevant checks
   (e.g. `scripts/verify_*.py`) and paste the results in the PR description —
   "128 checks, ALL CHECKS PASSED" beats "should be fine".

## Repo conventions

| Work | Goes in |
| ---- | ------- |
| EDA notebooks | `notebooks/NN_initials_topic.ipynb` (e.g. `03_jw_dashboard_data_prep.ipynb`) |
| Reusable code | `src/cleaning/`, `src/eda/`, `src/dashboard/` |
| One-off utilities | `scripts/` |
| Cleaned data shared by the team | `data/processed/` — committed (parquet parts) |
| Raw dataset | `data/raw/` — **never committed** (gitignored); see `scripts/download_data.py` |
| Figures used in slides | `reports/figures/` with captions |
| Slides | `reports/mid-presentation/` |
| Dataset/cleaning/decision docs | `docs/` |

## Boundaries (hard rules)

- **Don't restructure, rename, or delete a teammate's committed files**
  without their explicit, itemized approval. Propose it in an issue or as a
  PR comment instead and let the owner act.
- **Diagnose before you fix** on shared work: report what you found in plain
  words first, implement after the owner agrees.

## Data license

The US Accidents dataset (Kaggle, Sobhan Moosavi) is licensed
**CC BY-NC-SA 4.0** — non-commercial use, share-alike. Keep that intact:
don't strip attribution, don't use the data commercially. The repo's
`LICENSE` file covers our code and docs, not the dataset.

## Issues

Use the templates (bug report / task). One issue per problem, with enough
context (branch, file paths, what you ran) that someone else can reproduce it.
