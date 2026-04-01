# Paper Acceptance Prediction (Config-Driven Pipeline)

This repository provides a config-driven machine learning pipeline for paper acceptance prediction.

The workflow is now centered around a single JSON config file and a sklearn-like pipeline class:

- `Config(...)` in `DataPipeline/config/pipeline.py`
- `FeaturePipeline(...)` in `DataPipeline/feature_pipeline.py`

The pipeline supports:

1. Data loading and label resolution
2. Text preprocessing and corpus statistics
3. Hand-crafted feature extraction and SVMLite export
4. Model training and evaluation

## 1. Repository Entry Points

Primary scripts:

- `run_combined_extraction.py`: run feature extraction/export only
- `train_models.py`: run model training/evaluation only
- `pipeline_config.json`: default config template

Core implementation:

- `DataPipeline/config/pipeline.py`: config schema and JSON parsing
- `DataPipeline/feature_pipeline.py`: end-to-end pipeline logic
- `DataPipeline/feature/handcrafted.py`: feature definitions

## 2. Environment Setup

Run commands from repository root.

Recommended environment from project conventions:

```bash
source /mnt/disk1/miniconda3/etc/profile.d/conda.sh
conda activate ml
```

Install required packages if needed:

```bash
pip install numpy scikit-learn nltk
python -m nltk.downloader stopwords
```

## 3. Quick Start

### 3.1 Feature extraction only

```bash
python run_combined_extraction.py pipeline_config.json
```

### 3.2 Training only

```bash
python train_models.py pipeline_config.json
```

### 3.3 Run both in one Python call

```python
from DataPipeline.config import Config
from DataPipeline.feature_pipeline import FeaturePipeline

cfg = Config("pipeline_config.json")
pipeline = FeaturePipeline(cfg)
pipeline.run()
```

## 4. Config File Overview

Default config file: `pipeline_config.json`

Top-level sections:

1. `DataConfig`
2. `PreprocessConfig`
3. `FeatureConfig`
4. `TrainingConfig`

### 4.1 DataConfig

Controls where data is read and where combined artifacts are written.

Fields:

- `base_dir`: base dataset folder (default: `Dataset`)
- `combined_name`: name of combined output directory (default: `all_combined`)
- `datasets`: list of datasets to include
- `splits`: split order to process (must include `train`)
- `reviews_subdir`: reviews folder name inside split
- `parsed_subdir`: parsed PDFs folder name inside split
- `output_subdir`: feature output folder name inside each split

Example:

```json
"DataConfig": {
  "base_dir": "Dataset",
  "combined_name": "all_combined",
  "datasets": ["acl_2017", "iclr_2017"],
  "splits": ["train", "dev", "test"],
  "reviews_subdir": "reviews",
  "parsed_subdir": "parsed_pdfs",
  "output_subdir": "dataset"
}
```

### 4.2 PreprocessConfig

Controls text normalization and corpus frequency bucket creation.

Fields:

- `methods`: ordered data-preprocess method list (default: `build_corpus_words`, `compute_frequency_buckets`)
- `only_char`: keep alphanumeric tokens only
- `lower`: lowercase text
- `stop_remove`: remove English stopwords
- `hfw_proportion`: top fraction treated as highest-frequency words
- `freq_proportion`: next fraction treated as frequent words
- `min_freq_threshold`: words below this count are infrequent

Example:

```json
"PreprocessConfig": {
  "methods": [
    "build_corpus_words",
    "compute_frequency_buckets"
  ],
  "only_char": true,
  "lower": true,
  "stop_remove": true,
  "hfw_proportion": 0.01,
  "freq_proportion": 0.05,
  "min_freq_threshold": 3
}
```

### 4.3 FeatureConfig

Controls feature extraction behavior and leakage/label fallback policy.

Fields:

- `max_vocab`: `false` or integer (also used in output file naming token)
- `encoder_type`: string or `false` (naming token)
- `use_hand_features`: whether hand features are emitted
- `allow_recommendation_fallback`: enable score-based label fallback when acceptance label is missing
- `recommendation_threshold`: threshold for fallback label rule
- `drop_post_review_leakage`: drop feature names containing recommendation/score/confidence tokens
- `preserve_corpus_cache`: reuse split `corpus.pkl` if present
- `shuffle_seed`: deterministic paper shuffle seed

Example:

```json
"FeatureConfig": {
  "max_vocab": false,
  "encoder_type": "w2v",
  "use_hand_features": true,
  "allow_recommendation_fallback": true,
  "recommendation_threshold": 3.5,
  "drop_post_review_leakage": true,
  "preserve_corpus_cache": true,
  "shuffle_seed": 42
}
```

### 4.4 TrainingConfig

Controls model training/evaluation.

Fields:

- `enabled`: run training stage
- `data_dir`: optional override for feature artifact folder; if null uses `DataConfig.combined_dir`
- `preprocess_methods`: ordered feature-matrix preprocess method list (default: `standardize_for_linear_models`)
- `standardize_linear_models`: backward-compatible on/off switch used by `standardize_for_linear_models`
- `threshold_min`: lower bound for fixed threshold grid
- `threshold_max`: upper bound for fixed threshold grid
- `threshold_steps`: number of points in fixed threshold grid
- `use_probability_midpoints`: if true, threshold search uses exact probability boundaries
- `ensemble_top_k`: top-k model probability blending on dev/test
- `model_candidates`: list of models and hyperparameters

Model candidate schema:

- `name`: display name
- `model_type`: one of `LogisticRegression`, `SVMClassifier`, `RandomForestClassifier`, `AdaBoostClassifier`, `GradientBoostingClassifier`
- `requires_scaling`: use standardized features for this model
- `hyperparams`: kwargs passed into model constructor

Example:

```json
"TrainingConfig": {
  "enabled": true,
  "data_dir": null,
  "preprocess_methods": ["standardize_for_linear_models"],
  "use_probability_midpoints": true,
  "ensemble_top_k": 3,
  "model_candidates": [
    {
      "name": "LogReg L2 lambda=0.01",
      "model_type": "LogisticRegression",
      "requires_scaling": true,
      "hyperparams": {
        "balance": true,
        "penalty": "l2",
        "reg_lambda": 0.01,
        "lr": 0.05,
        "max_iter": 2500,
        "random_state": 42
      }
    }
  ]
}
```

## 5. Common Configuration Tasks

### 5.1 Use only selected datasets

Edit:

```json
"DataConfig": {
  "datasets": ["acl_2017", "iclr_2017"]
}
```

### 5.2 Disable recommendation-score fallback labels

Edit:

```json
"FeatureConfig": {
  "allow_recommendation_fallback": false
}
```

### 5.3 Run extraction without training

Edit:

```json
"TrainingConfig": {
  "enabled": false
}
```

Then run:

```bash
python run_combined_extraction.py pipeline_config.json
```

### 5.4 Change random seed

Edit:

```json
"FeatureConfig": {
  "shuffle_seed": 7
}
```

### 5.5 Add or remove model candidates

Edit the `TrainingConfig.model_candidates` list.

Each entry maps directly to a model constructor in `Models/` via `FeaturePipeline` model registry.

## 5.6 Pipeline Stage Methods

`FeaturePipeline` exposes readable stage methods aligned with execution flow:

1. `load_raw_data(split)`
2. `preprocess_data(process_data)`
3. `extract_features(process_data, fit_mode)`
4. `load_features()`
5. `train(loaded_features=None)`
6. `predict_proba(model, X)` / `predict_labels(probabilities, threshold)`
7. `eval(name, model, X_dev, y_dev, X_test, y_test)`

Internal preprocess registries are split by domain:

- data preprocess registry: `DataPipeline/preprocess/__init__.py` via `get_data_preprocess_methods`
- feature-matrix preprocess registry: `DataPipeline/preprocess/__init__.py` via `get_feature_preprocess_methods`
- default feature transform implementation: `DataPipeline/preprocess/feature_transform.py`

## 6. Output Artifacts

For each split, the pipeline writes under:

`Dataset/all_combined/<split>/dataset/`

Files:

- `features.svmlite_<max_vocab>_<encoder>_<hand>.txt`
- `labels_<max_vocab>_<encoder>_<hand>.tsv`
- `ids_<max_vocab>_<encoder>_<hand>.tsv`

Train split also writes feature map:

- `Dataset/all_combined/train/dataset/features_<max_vocab>_<encoder>_<hand>.dat`

Notes:

- Dev/test use the train feature map schema.
- `corpus.pkl` is used for corpus caching within split output dirs.

## 7. Feature Groups (Current)

The current hand-crafted feature set (from `DataPipeline/feature/handcrafted.py`) includes:

1. Abstract keyword flags
2. Citation counts and recency
3. Citation context statistics
4. Structural reference counts (figures/tables/sections/equations/theorems)
5. Lexical/style signals (unique words, sentence length, frequent-word ratio)
6. Metadata signals (title length, number of authors)
7. Additional normalized and interaction features (for example ratios per section and recency interactions)

## 8. Troubleshooting

1. Error: missing NLTK stopwords

Run:

```bash
python -m nltk.downloader stopwords
```

2. Error: file not found for feature artifacts during training

- Ensure extraction completed successfully.
- Confirm `TrainingConfig.data_dir` and naming tokens in `FeatureConfig` match generated files.

3. Empty or very small training set

- Check dataset paths in `DataConfig.datasets`.
- Check whether many papers are skipped due to missing labels.
- If needed, enable fallback labels with `FeatureConfig.allow_recommendation_fallback`.

4. Memory pressure during training

- Some models currently convert sparse input to dense arrays.
- Reduce data scope or simplify model candidate list first.

## 9. Helper Scripts

You can also run the config-driven extraction wrapper scripts:

- `DataPipeline/featurize_acl_229.sh [config_path]`
- `DataPipeline/featurize_acl_229.ps1 [config_path]`

If no path is given, they default to `pipeline_config.json`.

## 10. Minimal End-to-End Example

```bash
python run_combined_extraction.py pipeline_config.json
python train_models.py pipeline_config.json
```

Or in one Python call:

```python
from DataPipeline.config import Config
from DataPipeline.feature_pipeline import FeaturePipeline

cfg = Config("pipeline_config.json")
result = FeaturePipeline(cfg).run()
print(result.keys())
```
