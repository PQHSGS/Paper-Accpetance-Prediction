# Paper Acceptance Prediction

This repository provides a config-driven pipeline for paper acceptance prediction.

Core files:

- `DataPipeline/config/pipeline.py`: config loading
- `DataPipeline/feature_pipeline.py`: feature extraction and single-model training
- `DataPipeline/preprocess/__init__.py`: preprocess registries
- `DataPipeline/feature/__init__.py`: feature registries

## 1. Entry Points

- `run_combined_extraction.py`: feature extraction only
- `train_models.py`: training and evaluation only

Shipped config templates:

- `pipeline_config.json`: logistic regression
- `pipeline_config_svm.json`: SVM
- `pipeline_config_random_forest.json`: random forest
- `pipeline_config_adaboost.json`: AdaBoost
- `pipeline_config_gradient_boosting.json`: gradient boosting

## 2. Environment Setup

Run commands from repository root.

```bash
source /mnt/disk1/miniconda3/etc/profile.d/conda.sh
conda activate ml
pip install numpy scikit-learn nltk
python -m nltk.downloader stopwords
```

## 3. Quick Start

Feature extraction:

```bash
python run_combined_extraction.py pipeline_config.json
```

Training:

```bash
python train_models.py pipeline_config.json
```

End-to-end:

```python
from DataPipeline.config import Config
from DataPipeline.feature_pipeline import FeaturePipeline

cfg = Config("pipeline_config.json")
result = FeaturePipeline(cfg).run()
print(result.keys())
```

## 4. Config Layout

Each config JSON has:

1. `DataConfig`
2. `PreprocessConfig`
3. `FeatureConfig`
4. `TrainingConfig`

### 4.1 DataConfig

Dataset and output layout.

Fields:

- `base_dir`
- `combined_name`
- `datasets`
- `splits`
- `reviews_subdir`
- `parsed_subdir`
- `output_subdir`

### 4.2 PreprocessConfig

Raw-text preprocessing and label-resolution behavior.

Fields:

- `methods`
- `only_char`
- `lower`
- `stop_remove`
- `hfw_proportion`
- `freq_proportion`
- `min_freq_threshold`
- `allow_recommendation_fallback`
- `recommendation_threshold`
- `preserve_corpus_cache`
- `shuffle_seed`

### 4.3 FeatureConfig

Feature extraction behavior.

Fields:

- `methods`
- `max_vocab`
- `encoder_type`
- `use_hand_features`
- `drop_post_review_leakage`

### 4.4 TrainingConfig

Only shared training setup lives here.

Fields:

- `enabled`: run training stage
- `data_dir`: optional override for feature artifact directory
- `preprocess_methods`: ordered feature-matrix preprocess methods
- `model`: model class name from `Models/`
- `model_param`: constructor kwargs for that specific model

Example:

```json
"TrainingConfig": {
  "enabled": true,
  "data_dir": null,
  "preprocess_methods": [
    "standardize_for_linear_models"
  ],
  "model": "LogisticRegression",
  "model_param": {
    "balance": true,
    "penalty": "l2",
    "reg_lambda": 0.01,
    "lr": 0.05,
    "max_iter": 2500,
    "random_state": 42
  }
}
```

There is no threshold search or training-time hyperparameter search in the pipeline. Predictions use the model probability output with a fixed `0.5` decision threshold.

## 5. Training Flow

`FeaturePipeline` runs:

1. `load_raw_data(split)`
2. `preprocess_data(process_data)`
3. `extract_features(process_data, fit_mode)`
4. `load_features()`
5. `prepare_training_data()`
6. `build_model()`
7. `fit(model, X_train, y_train)`
8. `eval(name, model, X_dev, y_dev, X_test, y_test)`
9. `run(extract_features=True)`

Training is one model per config file. The selected model is instantiated from `TrainingConfig.model` and `TrainingConfig.model_param`.

## 6. Available Preprocess Methods

Data preprocess methods are registered in `DataPipeline/preprocess/__init__.py`.

Current names:

- `build_corpus_words`
- `compute_frequency_buckets`

Feature-matrix preprocess methods are also registered there.

Current names:

- `standardize_for_linear_models`

Use `standardize_for_linear_models` in `TrainingConfig.preprocess_methods` for linear models, and use `[]` for tree-based models.

## 7. Available Feature Methods

Feature methods are registered in `DataPipeline/feature/__init__.py`.

Current names:

- `handcrafted_features`

## 8. Available Models And `model_param`

Supported `TrainingConfig.model` values:

- `LogisticRegression`
- `SVMClassifier`
- `RandomForestClassifier`
- `AdaBoostClassifier`
- `GradientBoostingClassifier`

### 8.1 LogisticRegression

Defined in `Models/logistic_regression.py`.

Possible `model_param` keys:

- `balance`
- `penalty`
- `reg_lambda`
- `fit_intercept`
- `max_iter`
- `lr`
- `tol`
- `random_state`

Example:

```json
"model": "LogisticRegression",
"model_param": {
  "balance": true,
  "penalty": "l2",
  "reg_lambda": 0.01,
  "lr": 0.05,
  "max_iter": 2500,
  "random_state": 42
}
```

### 8.2 SVMClassifier

Defined in `Models/svm.py`.

Possible `model_param` keys:

- `balance`
- `kernel`
- `C`
- `gamma`
- `degree`
- `coef0`
- `max_iter_linear`
- `max_passes`
- `tol`
- `eps_alpha`
- `fit_intercept`
- `use_sample_weight_resample`
- `resample_n`
- `probability_scale`
- `random_state`

Example:

```json
"model": "SVMClassifier",
"model_param": {
  "balance": true,
  "kernel": "linear",
  "C": 1.0,
  "max_iter_linear": 6000,
  "random_state": 42
}
```

### 8.3 RandomForestClassifier

Defined in `Models/random_forest.py`.

Possible `model_param` keys:

- `balance`
- `n_estimators`
- `bootstrap`
- `max_features`
- `random_state`
- `criterion`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`

### 8.4 AdaBoostClassifier

Defined in `Models/ada_boost.py`.

Possible `model_param` keys:

- `balance`
- `n_estimators`
- `learning_rate`
- `criterion`
- `base_max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `random_state`

### 8.5 GradientBoostingClassifier

Defined in `Models/gradient_boosting.py`.

Possible `model_param` keys:

- `balance`
- `n_estimators`
- `learning_rate`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `l2_leaf_reg`
- `random_state`

## 9. Common Tasks

Use only selected datasets:

```json
"DataConfig": {
  "datasets": ["acl_2017", "iclr_2017"]
}
```

Disable recommendation-score fallback labels:

```json
"PreprocessConfig": {
  "allow_recommendation_fallback": false
}
```

Run extraction without training:

```json
"TrainingConfig": {
  "enabled": false
}
```

Switch from logistic regression to random forest:

```json
"TrainingConfig": {
  "preprocess_methods": [],
  "model": "RandomForestClassifier",
  "model_param": {
    "balance": true,
    "n_estimators": 300,
    "max_depth": 10,
    "min_samples_leaf": 2,
    "random_state": 42
  }
}
```

## 10. How To Extend The Pipeline

### 10.1 Propose New Hyperparameters For An Existing Model

If the parameter already exists in that model constructor:

1. Edit the config JSON.
2. Update `TrainingConfig.model_param`.
3. Run `train_models.py`.

No pipeline code changes are needed if the model class already accepts the parameter.

If the parameter does not exist yet:

1. Add it to the target model constructor in `Models/*.py`.
2. Use it inside that model’s fit/predict implementation.
3. Put it in the config file under `TrainingConfig.model_param`.
4. Update this README section for that model.

### 10.2 Add A New Data Preprocess Function

1. Implement the function in `DataPipeline/preprocess/`.
2. Match the existing signature used by preprocess functions.
3. Register the name in `DATA_PREPROCESS_METHODS` in `DataPipeline/preprocess/__init__.py`.
4. Add that method name to `PreprocessConfig.methods`.
5. Update the README method list.

### 10.3 Add A New Feature-Matrix Preprocess Function

1. Implement the function in `DataPipeline/preprocess/feature_transform.py` or another file under `DataPipeline/preprocess/`.
2. Register it in `FEATURE_PREPROCESS_METHODS` in `DataPipeline/preprocess/__init__.py`.
3. Add the method name to `TrainingConfig.preprocess_methods`.
4. Update the README method list.

### 10.4 Add A New Feature Extraction Function

1. Implement the function in `DataPipeline/feature/`.
2. Match the existing feature-method signature.
3. Register it in `FEATURE_METHODS` in `DataPipeline/feature/__init__.py`.
4. Add the method name to `FeatureConfig.methods`.
5. Update the README method list.

### 10.5 Add A New Model Class

1. Implement the model in `Models/`.
2. Export it from `Models/__init__.py`.
3. Add it to `_MODEL_REGISTRY` in `DataPipeline/feature_pipeline.py`.
4. Set `TrainingConfig.model` to the new class name.
5. Put constructor kwargs in `TrainingConfig.model_param`.
6. Update the README model list and parameter list.

## 11. Output Artifacts

Artifacts are written under:

`Dataset/all_combined/<split>/dataset/`

Files:

- `features.svmlite_<max_vocab>_<encoder>_<hand>.txt`
- `labels_<max_vocab>_<encoder>_<hand>.tsv`
- `ids_<max_vocab>_<encoder>_<hand>.tsv`
- `features_<max_vocab>_<encoder>_<hand>.dat` for the train split

## 12. Troubleshooting

Missing NLTK stopwords:

```bash
python -m nltk.downloader stopwords
```

Training cannot find feature files:

- Run extraction first.
- Confirm `TrainingConfig.data_dir` and `FeatureConfig` naming fields match the generated files.

Very small training set:

- Check `DataConfig.datasets`.
- Check skipped unlabeled counts during loading.
- If needed, enable `PreprocessConfig.allow_recommendation_fallback`.

## 13. Helper Scripts

- `DataPipeline/featurize_acl_229.sh [config_path]`
- `DataPipeline/featurize_acl_229.ps1 [config_path]`

If no config path is passed, they default to `pipeline_config.json`.
