from typing import Any, Dict, List, Union


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _coerce_bool_or_int(value: Any) -> Union[bool, int]:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        v = value.strip()
        if v.lower() in {"false", "none", "off"}:
            return False
        if v.lower() in {"true", "on"}:
            return True
        if v.isdigit():
            return int(v)
    raise ValueError(f"Expected bool|int compatible value, got: {value!r}")


def _coerce_str_list(value: Any, field_name: str) -> List[str]:
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    raise ValueError(f"Expected list[str] for {field_name}, got: {value!r}")


def _split_known(d: Dict[str, Any], known_keys: set[str]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    known: Dict[str, Any] = {}
    extra: Dict[str, Any] = {}
    for k, v in d.items():
        if k in known_keys:
            known[k] = v
        else:
            extra[k] = v
    return known, extra


def _get_section(payload: Dict[str, Any], *keys: str) -> Dict[str, Any]:
    for k in keys:
        value = payload.get(k)
        if isinstance(value, dict):
            return value
    return {}


def _deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for k, v in update.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def _with_extra(base_dict: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base_dict)
    out.pop("extra", None)
    out.update(extra)
    return out


def _extract_top_level_extra(payload: Dict[str, Any]) -> Dict[str, Any]:
    reserved = {
        "DataConfig",
        "data",
        "data_config",
        "PreprocessConfig",
        "preprocess",
        "preprocess_config",
        "FeatureConfig",
        "feature",
        "feature_config",
        "TrainingConfig",
        "training",
        "training_config",
    }
    return {k: v for k, v in payload.items() if k not in reserved}
