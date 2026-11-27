# glm53_vibe_LRB_T062

````python
import math

import yaml


def cosine_yaml_key(yaml_text, key):
    data = yaml.safe_load(yaml_text)
    if isinstance(data, dict) and key in data:
        value = data[key]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            data[key] = math.cos(value)
    return data
````
