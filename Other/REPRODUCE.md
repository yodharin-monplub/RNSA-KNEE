# How this project is laid out and how to reproduce it without the Claude Code session

## Layout (AGENTS.md, 2026-09-20)
- `AGENTS.md` - the only file in the root (plus hidden tool folders `.git`, `.claude`, `.codex`, `.agents`).
- `Data/competition/` - Kaggle competition CSVs (`train.csv` holds the reports and the 58 gold labels), pixel spec.
- `Data/labels/` - report-derived training labels: `llm_relabel/` (ours: `targets_g38f_low.csv` = Gemini v1 used by the
  best models; `targets_hybrid.csv`; `all_g38f_low_v2.*` = official-criteria relabel), `weak_labels*/`, `raptor_labels/` (public).
- `Data/external/` - external knee MRI: `mrnet/`, `kneemri/`, `fastmri/` (signed links only), `fastmri_plus/`, `oai/`.
  Research-use licences accepted by the account owner; never redistribute. Compact private copies live on Kaggle.
- `Model/modelN/` - one folder per model: `readme.txt` (what and why), `config.json` or notebook source, `score.txt`.
- `Other/` - everything else: `scripts/` (experiments, Kaggle notebook builders), `cloud/` (Vast.ai rental tooling),
  `knee/` (library code), `artifacts/` (results copied back from rentals and Kaggle), `docs/`, `EXPERIMENTS.md`
  (full chronological log), `README.md`, `.env` (API keys), `email.txt`, `.venv/`.

## Path convention - important
All code uses `ROOT = Other/` and addresses `ROOT/data/...` and `ROOT/modelN/...`. Those exist as links:
`Other/data/*` are junctions / hard links into `Data/`, and `Other/modelN` are junctions to `Model/modelN`.
Recreate them on a fresh checkout with:
    cd Other
    python scripts/layout_data_links.py
    # model junctions: for each folder in ..\Model:  mklink /J Other\modelN ..\Model\modelN
Always run commands from `Other/`, e.g. `.\.venv\Scripts\python scripts\compare_fulltrain.py`.

## Environment
    cd Other
    python -m venv .venv          # the venv is not relocatable; recreate it after moving the project
    .\.venv\Scripts\python -m pip install -r requirements.txt
`.env` needs KAGGLE_API_KEY, VAST_AI_API_KEY (rentals), OPENROUTER_API_KEY (relabels), REDIVIS_API_TOKEN (MRNet).

## Reproducing the best submission (model61, public 0.943)
1. Inputs are Kaggle datasets/kernels, not local files: the public ensemble notebook (mattiaangeli), our fine-tuned
   CoAtNet checkpoints `yodharinmonplub/rsna-knee-model{38,39,40,41}-weights`, and the retrained heads written by the
   kernels `yodharinmonplub/rsna-knee-head-retrain-{maxspan,dense,v8,fine}` (`model61_<arm>_best.pt`).
2. `python scripts/run_model23_24.py` builds the offline submission notebook for a given model name (patch set per
   model is defined in that file); `Model/model61/` holds the exact notebook source that was submitted.
3. Training a CoAtNet arm: `Model/model38/config.json` + `scripts/train_model16.py`, launched on a Vast rental with
   `RSNA_CLOUD_RUN=model38 python cloud/launch_only.py`, guarded by `python cloud/guard_once.py` every ~20 minutes
   (destroys the rental after a verified backup). Dense image caches are the Kaggle kernel outputs
   `yodharinmonplub/rsna-knee-dense-cache-part{0..3}` (built by `scripts/run_dense_cache.py`).
4. Head retrain (the +0.001 step): `python scripts/run_head_retrain.py maxspan|dense|v8|fine` (free Kaggle T4).
5. Labels: `python scripts/llm_relabel.py --subset all --model google/gemini-3.8-flash --effort low --tag all_g38f_low`
   (`--prompt v2` = host's official criteria, see `docs/official_label_criteria.md`).

## Where results are recorded
`Other/EXPERIMENTS.md` (every run, cost and conclusion), `Model/modelN/score.txt`, `Other/artifacts/remote_modelN/`
(history.json, metrics.json, gold_predictions.csv, best.pt for recent runs), `Other/artifacts/cloud/modelN/finished.json` (cost).
