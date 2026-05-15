# Evaluation post-processing & figures

Standalone scripts that consume the metrics JSONs produced by the
canonical reproducer scripts (`scripts/{retrieval,localization,
dialogue}/...`) and emit (a) headline numbers used in the paper /
rebuttal, and (b) the figures bundled with the rebuttal report.

The scripts here do **not** run the retrieval / localization / dialog
pipelines themselves — they read the cached outputs (`eval/*.json`,
`eval/new_data/*.json`, `eval/rebuttal_plots/*`, `/tmp/dialog_*.log`).
Run the reproducer scripts first; then run a tool from this directory.

## Scripts

| Script | Reads | Writes | Used for |
|---|---|---|---|
| [`recall_at_threshold.py`](recall_at_threshold.py) | `eval/*_metrics.json`, `/tmp/dialog_*.log` | stdout table | Reviewer EEmB hook — Recall@(τ_pos, τ_ang) at five thresholds across methods × datasets. |
| [`error_distribution_plots.py`](error_distribution_plots.py) | `eval/*_metrics.json` | `eval/rebuttal_plots/error_*.png` | Reviewer EEmB hook — KDE, CDF, scatter, and direction-error panels showing skew of the position-error distribution. |
| [`recall_cdf_with_dialog.py`](recall_cdf_with_dialog.py) | `eval/new_data/*` (+ optional `eval/baseline_eval_metrics_qwen_*.json`) | `eval/rebuttal_plots/recall_cdf_*.png` | Figure-5-style position-error CDFs (Midpoint / Qwen / LangLoc / LangLoc-top-10 oracle). Qwen series is skipped with a `[WARN]` if its cache is absent. |
| [`rebuttal_figures.py`](rebuttal_figures.py) | hard-coded numbers (see top of file) | `eval/rebuttal_plots/rebuttal_{scoreboard,qwen_scatter}.png` | The two summary plots used in the rebuttal PDF — Tables 1–5 reproduction scoreboard and Qwen-reproduction scatter. |
| [`dialog_log_stats.py`](dialog_log_stats.py) | `/tmp/dialog_*.log` (runner stdout) | stdout table + `eval/rebuttal_plots/dialog_*.png` | Reviewer WxoL hook — how many dialog rounds the system needs to converge, per-scene posterior trajectories. |

## Conventions

- All scripts derive their repo root from `Path(__file__).resolve().parents[2]`,
  so they run from any cwd inside the repo.
- All output goes under `eval/rebuttal_plots/` (gitignored).
- All randomness uses the canonical project seed `42`
  (see `langloc/utils/seed.py`). No script in this directory is itself
  stochastic — they are post-processors over cached numbers.
- Where a required input is missing, the scripts print a `[WARN]` and
  either skip that series (CDFs) or fail loudly with a clear message
  about which cache to regenerate. No silent fallbacks.
