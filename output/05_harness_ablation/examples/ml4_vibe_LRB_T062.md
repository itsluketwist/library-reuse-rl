# ml4_vibe_LRB_T062

````python
import math

try:
    import yaml
except ImportError:  # pragma: no cover - fallback when PyYAML is unavailable
    yaml = None


def _parse_scalar(text):
    text = text.strip()
    if text == "" or text in ("null", "Null", "NULL", "~"):
        return None
    if text in ("true", "True", "TRUE"):
        return True
    if text in ("false", "False", "FALSE"):
        return False
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        return text[1:-1]
    return text


def _simple_parse(yaml_text):
    """Minimal fallback for a flat top-level mapping of scalar values."""
    data = {}
    for line in yaml_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, sep, value = line.partition(":")
        if not sep:
            continue
        data[key.strip()] = _parse_scalar(value)
    return data


def cosine_yaml_key(yaml_text, key):
    if yaml is not None:
        data = yaml.safe_load(yaml_text)
    else:
        data = _simple_parse(yaml_text)
    if not isinstance(data, dict):
        return data
    if key in data:
        data[key] = math.cos(data[key])
    return data
````
