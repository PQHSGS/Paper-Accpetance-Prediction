# Project Guidelines

## Code Style
- Preserve the existing split in style by area:
  - `Models/`: typed, sklearn-like API, NumPy-first implementations.
  - `DataPipeline/`: script-oriented ETL code.
- For all new or modified Python code, prefer:
  - Type hints on public functions and class methods.
  - Small, testable functions instead of long monolithic `main` blocks.
  - Explicit error messages for missing files/dirs and invalid arguments.
- Keep naming consistent with existing exported model names in `Models/__init__.py`.

## Architecture
- Keep a strict boundary:
  - `DataPipeline/` builds feature artifacts (`*.svmlite`, labels, ids, feature maps).
  - `Models/` consumes numeric matrices and exposes classifier APIs (`fit/predict/predict_proba`).
  - Root scripts orchestrate only:
    - `run_combined_extraction.py` for feature extraction.
    - `train_models.py` for training/evaluation.
- Avoid introducing model logic into `DataPipeline/` or data-loading logic into model classes.

## Build and Run
- Run from repository root (required by current relative path/import behavior).
- Use this repository-local environment for Python work:
  - `conda activate ml`
  - If running in a non-interactive shell: `source /mnt/disk1/miniconda3/etc/profile.d/conda.sh && conda activate ml`
- Main commands:
  - `python run_combined_extraction.py pipeline_config.json`
  - `python train_models.py pipeline_config.json`
- Dataset assumptions:
  - Per source dataset: `Dataset/<name>/{train,dev,test}/{reviews,parsed_pdfs}`
  - Combined output: `Dataset/all_combined/{train,dev,test}/dataset`

## Conventions
- Maintain sklearn-like behavior in `Models/`:
  - Validate input shapes and class cardinality in `fit`.
  - Keep `predict_proba` output as `(n_samples, 2)` with class-0/class-1 columns.
  - Preserve `random_state` behavior for reproducibility.
- In `DataPipeline/`, keep extraction deterministic where possible:
  - Avoid uncontrolled randomness unless explicitly needed.
  - If shuffling is needed, allow a seed or document why nondeterminism is acceptable.

## Performance and Complexity Priorities
- Prefer sparse-friendly flows for high-dimensional features:
  - Do not densify sparse matrices unless a model strictly requires dense input.
  - If needed, document memory impact and feature-size constraints.
- Prefer vectorized NumPy operations over Python loops in hot paths.
- Avoid repeated expensive work in loops:
  - Reuse loaded resources (e.g., vocab/features/stopwords).
  - Move invariant computations outside per-paper/per-row loops.
- For algorithms with high time complexity (notably KNN and kernel SVM paths), call out complexity tradeoffs and expected dataset size limits in code comments/docstrings.

## Known Pitfalls
- Import path assumptions are fragile:
  - Current scripts rely on `sys.path` manipulation and repo-root execution.
  - Do not change working-directory assumptions silently; update entrypoints consistently if refactoring imports.
- File naming/argument conventions are string-encoded (`"False"`, `"True"`, encoder name). Keep compatibility with existing artifact names unless migration is intentional.
- For any refactor that changes artifact filenames or directory layout, update both extraction and training entrypoints in the same change.

## Key Files
- `run_combined_extraction.py`
- `train_models.py`
- `DataPipeline/config/pipeline_config.py`
- `DataPipeline/feature_pipeline.py`
- `DataPipeline/feature/handcrafted.py`
- `Models/base.py`
- `Models/decision_tree.py`
