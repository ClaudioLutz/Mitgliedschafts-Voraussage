# AGENTS.md

This repository is a lead-generation ML pipeline that predicts membership conversion for Swiss companies, trained on historical SQL Server snapshots and used to rank current non-members by probability of joining within a configurable horizon.

This file is the operating manual for AI coding agents (and humans) working in this repo. Follow it unless the user explicitly overrides it.

---

## Quick rules (do these every time)

1. **Start with the repo map** (below) and edit the *closest* existing module/script first.
2. **Prefer existing patterns and dependencies.** Do not introduce new libraries unless necessary and justified.
3. **Avoid data leakage:** membership outcomes and snapshot timing must not bleed into features.
4. **Keep changes small and testable.** Add/adjust tests for functional changes.
5. **If the change is functional,** you **must** add a Story file under `docs/stories/` in the same PR/commit (see “Change Documentation”).

---

## Project map (where things live)

### Core pipeline (primary)

* `training_lead_generation_model.py` — end-to-end training, evaluation, calibration, ranking, and export workflow.
* `column_transformer_lead_gen.py` — canonical preprocessing and feature engineering (explicit column lists; encoding strategy).
* `time_estimation.py` — runtime estimation using the same preprocessing pipeline.

### Analysis & reporting (optional/diagnostics)

* `run_visualizations.py` / `visualize_lead_model.py` — diagnostic plots and reports.
* `analyze_current_snapshot_data.py` — descriptive analysis of the current snapshot.

### Tools (manual / dev utilities)

* `tools/manual_preprocessor_check.py` — manual preprocessor validation.
* `tools/benchmark_enhanced_sampling.py` — sampling benchmark (CPU/memory heavy; do not run by default).

### Docs & outputs

* `docs/` — design, evaluation, deployment, roadmap.
* `tests/` — pytest smoke tests.
* `outputs/`, `artifacts/`, `figures/` — generated results (generally not committed unless explicitly required).

---

## Tech stack and constraints

### Language/runtime

* Python (virtualenv-based workflow).

### Core libraries (do not replace casually)

* Data: `pandas`, `numpy`
* ML: `scikit-learn`, `imbalanced-learn`, `category-encoders`
* DB: `sqlalchemy`, `pyodbc`
* Serialization: `joblib`
* Viz: `matplotlib`, `seaborn` (optional: `shap`)

If you add/remove dependencies, update the appropriate `requirements*.txt` file(s) and document the change in a Story.

### External system dependency

* SQL Server connectivity via ODBC (Microsoft ODBC Driver 17 is referenced in setup/connection patterns). Do not commit secrets.

---

## Common commands (copy/paste)

### Setup

```bash
python -m venv .venv
# activate your venv (platform-specific)
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Run the pipeline

```bash
python training_lead_generation_model.py
```

### Estimate runtime

```bash
python time_estimation.py
```

### Visualizations / analysis

```bash
python run_visualizations.py
python analyze_current_snapshot_data.py
```

### Tests

```bash
pytest
```

---

## Development standards (how to make changes safely)

### 1) Preserve the core modeling design

The pipeline is intentionally:

* **Time-aware** (train/val/test split by snapshot date; time-series CV)
* **Leakage-safe** (membership outcome fields are excluded from features)
* **Centralized preprocessing** in `column_transformer_lead_gen.py`

When modifying model behavior, ensure the above invariants still hold.

### 2) Preprocessing rules (high impact)

* Treat `column_transformer_lead_gen.py` as the **single source of truth** for feature engineering and encoding.
* If you add/remove/rename feature columns:

  * Update the explicit column lists (numeric/ordinal/low-card/high-card).
  * Update drop/leakage handling consistently.
  * Add/adjust tests to catch schema regressions.

### 3) Compatibility guards are intentional

This repo includes defensive code paths for library version differences (e.g., scikit-learn API changes). Do not remove these guards unless you also:

* constrain/lock versions explicitly, and
* document the rationale in a Story.

### 4) Database I/O and safety

* Never hardcode credentials or tokens.
* Do not log sensitive row-level data.
* Prefer parameterized SQL and clear inclusion criteria when editing queries.

### 5) Performance and “don’t melt the laptop”

* Training can be heavy. Avoid running full training loops unless explicitly needed.
* Do not run `tools/benchmark_enhanced_sampling.py` by default.
* Prefer small/synthetic samples for tests and quick validation.

### 6) Output hygiene

* Keep generated artifacts in `outputs/`, `artifacts/`, `figures/`.
* Avoid committing large generated files unless explicitly requested.

---

## Testing expectations

* Run `pytest` for any functional change.
* Add tests when changing:

  * preprocessing/feature engineering,
  * sampling logic,
  * scoring/ranking outputs,
  * file output schemas (CSV columns, artifact filenames).

Tests should remain fast and deterministic (fixed `random_state` where applicable).

---

## Change Documentation (required)

For every change that modifies behavior, adds/removes features, changes dependencies, alters configuration, or impacts performance/security:

* Create **one** new Markdown file under `docs/stories/` in the **same PR/commit** as the code change.

### Filename format

* `docs/stories/YYYYMMddHHmmss-topic-of-the-code-change.md`
* `YYYYMMddHHmmss` is a **14-digit timestamp** (recommend **UTC** to avoid timezone ambiguity).
* `topic-of-the-code-change` is a short **kebab-case** slug (ASCII, no spaces, no underscores).

Examples:

* `docs/stories/20251228143005-fix-dedup-merge-logic.md`
* `docs/stories/20251228160219-add-address-normalization-step.md`

### Minimum required contents

Each story file must include these sections:

#### Summary

1–3 sentences describing the change.

#### Context / Problem

Why this change is needed (bug, requirement, refactor driver).

#### What Changed

Bulleted list of key implementation changes (include modules/components touched).

#### How to Test

Exact commands and/or manual steps to validate.

#### Risk / Rollback Notes

What could go wrong, and how to revert/mitigate.

### When a story is NOT required

* Pure formatting (whitespace), typo fixes in comments/docs, or non-functional refactors that do not change behavior.

  * If in doubt, create a story.
