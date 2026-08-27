# Experiment inventory

Only `logisticregression_only.py` and `boostedtrees_ab.py` are maintained
training pipelines. They use the committed `cleaned_data.csv`, write governed
artifacts to `model_artifacts/`, and are covered by tests, Ruff, mypy-backed
shared modules, deterministic training smoke, and provenance verification.

The other root-level model scripts are retained as historical research. They
are not production entrypoints, are not used by inference, and are outside the
Ruff/typecheck boundary. Moving them would break old commands and obscure their
history, so this inventory governs them in place instead of creating cosmetic
directory churn.

## Classification

| Script | Status | Expected data | Output and reproducibility |
| --- | --- | --- | --- |
| `logisticregression_only.py` | MAINTAINED | committed `cleaned_data.csv` | governed `model_artifacts/`; reproducible from locks |
| `boostedtrees_ab.py` | MAINTAINED | committed `cleaned_data.csv` | governed `model_artifacts/`; reproducible from locks |
| `bernouli_nb.py` | HISTORICAL-VALUABLE | uncommitted `cleaned_data_upd.csv` | reports metrics only; optional Plotly dependency is not locked |
| `bilayered_neural.py` | HISTORICAL-VALUABLE | uncommitted `cleaned_data_upd.csv` | `experiment_results/final_results_bilayer_nn.csv`; optional plotting dependency is not locked |
| `BoostedTrees+Narrow_NN.py` | HISTORICAL-VALUABLE | uncommitted `cleaned_data_upd.csv` | `experiment_results/final_results_boosted_nn_stack.csv`; not release-governed |
| `boostedTrees.py` | HISTORICAL-VALUABLE | uncommitted `cleaned_data_upd.csv` | `experiment_results/final_results_boostedtrees.csv`; predates the maintained boosted pipeline |
| `coarseKnn.py` | HISTORICAL-VALUABLE | uncommitted `cleaned_data_upd.csv` | `experiment_results/final_results_coarseknn.csv`; optional plotting dependency is not locked |
| `conditionalStacking.py` | HISTORICAL-VALUABLE | uncommitted `cleaned_data_upd.csv` | `experiment_results/final_results_conditional_stack_grid.csv`; not release-governed |
| `efficient_lr.py` | HISTORICAL-VALUABLE | uncommitted `cleaned_data_upd.csv` | `experiment_results/final_results_efficient_logreg.csv`; superseded for serving by the maintained logistic pipeline |
| `enc1_smoteenn.py` | HISTORICAL-VALUABLE | uncommitted `enc1.csv` | `experiment_results/enc1_smoteenn.csv`; requires unpinned imbalanced-learn |
| `ensemble_xgb_narrow.py` | HISTORICAL-VALUABLE | uncommitted `cleaned_data_upd.csv` | `experiment_results/final_results_stacking.csv`; not release-governed |
| `logreg+clustering.py` | HISTORICAL-VALUABLE | uncommitted `cleaned_data_upd.csv` | four CSVs under `experiment_results/`; requires unpinned clustering/plotting packages |
| `mlp_narrowNeural.py` | HISTORICAL-VALUABLE | uncommitted `cleaned_data_upd.csv` | `experiment_results/final_results_narrow_nn.csv`; not release-governed |
| `nb_gaussian.py` | HISTORICAL-VALUABLE | uncommitted `cleaned_data_upd.csv` | declares result/profile paths but does not persist them; incomplete historical study |
| `neko.py` | HISTORICAL-VALUABLE | uncommitted `cleaned_data_upd.csv` | `experiment_results/final_results_catboost.csv`; requires unpinned CatBoost/Plotly |
| `qsvm.py` | HISTORICAL-VALUABLE | uncommitted `cleaned_data_upd.csv` | `experiment_results/final_results_qsvm.csv`; not release-governed |
| `qsvm_fast.py` | HISTORICAL-VALUABLE | uncommitted `cleaned_data_upd.csv` | `experiment_results/final_results_qsvm_fast.csv`; reduced-budget historical comparison |
| `subspaceKNN+boostedTrees+NarrowNN.py` | HISTORICAL-VALUABLE | uncommitted `cleaned_data_upd.csv` | `experiment_results/final_results_triple_stack.csv`; not release-governed |
| `tuning.py` | HISTORICAL-VALUABLE | uncommitted `cleaned_data_upd.csv` | `experiment_results/final_results_boostedtrees.csv`; requires several unpinned model libraries |
| `xgboost_only.py` | HISTORICAL-VALUABLE | uncommitted `cleaned_data_upd.csv` | `experiment_results/final_results_xgboost.csv`; predates the maintained boosted pipeline |
| `voting_ensemble.py` | REDUNDANT/DEAD | uncommitted relative `enc.csv` | writes `final_results_ensemble.csv` to the current directory; no CLI/import guard and unsafe to import |

## Known archival hazards

- Eighteen historical scripts use `experiment_config` to expose `--data-path`
  and `--results-dir`, then intentionally execute top to bottom. Import is
  refused before training, but these are still scripts rather than libraries.
- `voting_ensemble.py` is the exception: it retains import side effects,
  current-working-directory paths, and a stale output location.
- `logreg+clustering.py` contains two bare `except:` handlers. They are recorded
  here rather than suppressed or rewritten without rerunning the original study.
- Several studies duplicate model families now owned by the maintained logistic
  and boosted pipelines. Their outputs are research CSVs, never serving
  artifacts, and are gitignored under `experiment_results/`.
- The historical dataset schemas and optional dependencies are not committed as
  a reproducible environment. `python SCRIPT --help` documents accepted paths;
  a successful historical rerun additionally requires the named dataset and
  libraries imported by that script.

No historical script is claimed to reproduce the committed serving artifacts.
