from .data import DataConfig
from .feature import FeatureConfig
from .pipeline import PipelineConfig
from .preprocess import PreprocessConfig
from .training import TrainingConfig

# Backward-compatible alias for existing entry scripts and docs.
Config = PipelineConfig

__all__ = [
    "DataConfig",
    "PreprocessConfig",
    "FeatureConfig",
    "TrainingConfig",
    "PipelineConfig",
    "Config",
]
