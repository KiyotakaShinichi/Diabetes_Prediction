# Changelog

Engineering milestones for this repository, newest first.

**No version has been released and no tag exists.** Everything below is
`Unreleased`. Dates are the authored dates of the commits cited beside each
entry, taken from the git history rather than assigned retrospectively, and each
milestone names the commit that completed it so any claim here can be checked
against the diff.

This file records how the system was built. It is not a product release note,
and it deliberately does not describe model quality — measured performance lives
in `model_artifacts/metrics.json` and is reported by the admin dashboard as
held-out evaluation, never as live accuracy.

## Unreleased

### Resource-constrained tabular model zoo — 2026-09-04

Thirty-one algorithms from six families behind one declared contract, trained
on a deterministic 1,000-row subset of the training partition and evaluated on
the same held-out rows as Track K. Capability is declared per model and
asserted against the built model, so the benchmark carries no per-model special
cases and a spec cannot overclaim; a model with no ranking score reports its
threshold-free metrics as undefined rather than computing them from hard
labels. Failures, skips and resource limits stay in the results table with
reasons attached.

The zoo reuses rather than reimplements: Track K's three networks are imported,
its subset machinery selects the training rows, and its evaluation and
threshold code scores every model. Seven new deep architectures each encode a
different structural hypothesis, including a neural additive model that cannot
represent feature interactions at all.

Exploratory by construction and labelled as such. Production serving, the
committed artifacts and Track K's evidence are untouched. (`034e058`,
`f651f4c`)

### Deep-learning challenger lab — 2026-08-28

Track K asks whether modern deep tabular models improve diabetes-risk
prediction over strong classical baselines, and answers it under one frozen
protocol committed before any model saw the test partition. Four families —
logistic regression, XGBoost, an MLP and an FT-Transformer — were trained fresh
on an identical split, with a residual tower added later as a third challenger.
The evaluation contract, calibration selection, paired bootstrap, promotion
policy, research provenance and error analysis are all new, and all live under
`research/`, which is measured by its own coverage gate rather than the
maintained-module one.

Nothing about production changed. The research package cannot address
`model_artifacts/`, and the deployed artifacts are hashed before and after the
benchmark tests to prove it.

A second, resource-constrained arm trains every family on one deterministic,
fingerprinted, nested 5,000-row subset drawn from the training partition alone,
with budgets set from measured per-epoch cost. Runs persist per-row test
predictions so a corrected metric can be re-derived without retraining — used
once, when average precision was found to mishandle the tied scores isotonic
calibration produces. (`15a0921`, `cf96f6b`, `b9b31b0`, `6ef01e6`, `9f10bd7`,
`9f86d3d`)

### Buyer readiness and static governance — 2026-08-27

Dependency inputs and deterministic locks are verified against each other by
`tools/verify_dependency_contract.py`, mypy checks the stable shared modules,
and CI exposes lint, typecheck, tests, artifact compatibility, provenance,
deterministic training smoke, container health smoke and dependency audit as
named gates. The archival experiment inventory in `experiments/README.md`
classifies every root-level script that sits outside the lint and typecheck
boundary. (`46ead99`, `4a9cfdf`)

### Single authoritative serving path — 2026-08-26

The public Streamlit app stopped loading model bundles and scoring requests
itself. It now posts to the FastAPI service, which owns validation, canonical
feature ordering, bundle verification, A/B routing, thresholding, request
correlation and inference telemetry. Equivalence with the previous direct
scoring was proved against captured fixtures for both variants before the old
path was removed. Render wiring passes the API address to the public service by
Blueprint reference. (`c8a53d4`, `1899003`, `02f4e9f`)

### Frontend accessibility and shared presentation — 2026-08-26

The Streamlit theme is pinned in `.streamlit/config.toml`, closing a dark-mode
failure that rendered body text at roughly 1:1 contrast. The assessment form no
longer pre-fills answers, the result leads with a single risk estimate and
states that it is not a diagnosis, and presentation moved into `ui/` shared by
both apps. Interaction and accessibility tests cover what the launch smokes
never reached. (`4ba20ba`, `b72a32c`, `73b13e5`)

### Composable training pipelines — 2026-08-26

Training was decomposed into independently callable stages, then the duplicated
orchestration those stages had produced in both pipelines was collapsed into
`ml_core/pipeline.py`, verified bit-identical against the pre-refactor
behaviour. (`265ff5b`, `0ed6684`)

### Canonical feature contract — 2026-08-26

`ml_core/feature_contract.py` became the single definition of the served
features and their order, with training and serving migrated onto it so a
mismatch fails loudly instead of scoring misaligned columns. (`f9e8d71`,
`40eb6f1`)

### Training provenance and artifact integrity — 2026-08-26

Training runs emit a manifest fingerprinting dataset, features, environment and
source; committed artifacts are inventoried by SHA-256 and verified in CI. The
attestation states explicitly that the lineage of the pre-existing artifacts is
unknown, because reconstructing it would be fabrication. (`1f489c3`, `18a4f50`)

### Shared evaluation core — 2026-08-26

Evaluation, bootstrap and threshold selection moved into `ml_core/`, fixing a
Youden threshold that could be returned as infinite. (`586fff3`)

### Fail-closed administration — 2026-08-26

The implicit default administrator was removed; with no provider configured no
login can succeed and the dashboard says so. (`431f099`)

### Reproducible runtime — 2026-08-25 to 2026-08-26

Artifact-compatible dependency locks, a canonical Python version shared by CI,
the Dockerfile, the devcontainer and Render, and a workflow running tests, lint
and a container smoke. (`935387b`, `0b78bf6`, `8cdefad`, `c0f193c`, `c09334a`)

### Test foundation — 2026-08-25

The first regression suite covering the API and the ML utilities, isolated from
any production database. (`3ffcc0d`)

### Initial deployment configuration — 2026-03-11

Render configuration, container image and optional shared PostgreSQL for the
deployed services. (`6572dbe`, `235d965`)
