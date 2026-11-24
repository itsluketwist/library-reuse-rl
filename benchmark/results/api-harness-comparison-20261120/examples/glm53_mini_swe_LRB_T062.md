# glm53_mini_swe_LRB_T062

````python
"""Parse a YAML mapping and replace the value under a given key by its cosine.

The primary implementation uses PyYAML (``yaml.safe_load``).  A tiny,
dependency-free parser for a subset of YAML is included as a fallback so that
this module remains usable if PyYAML is not installed.
"""

import math


# --------------------------------------------------------------------------
# Fallback (subset) YAML parsing -- only used when PyYAML is unavailable.
# --------------------------------------------------------------------------

def _split_flow(text):
    """Split a flow-collection body on top-level commas."""
    parts, depth, cur = [], 0, []
    for ch in text:
        if ch in '[{':
            depth += 1
        elif ch in ']}':
            depth -= 1
        if ch == ',' and depth == 0:
            parts.append(''.join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append(''.join(cur))
    return [p.strip() for p in parts if p.strip()]


def _strip_comment(line):
    """Remove a trailing comment while respecting simple quotes."""
    out, quote = [], None
    for ch in line:
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
        elif ch in ('"', "'"):
            quote = ch
            out.append(ch)
        elif ch == '#':
            break
        else:
            out.append(ch)
    return ''.join(out).rstrip()


def _parse_scalar(text):
    text = text.strip()
    if text == '' or text.lower() in ('null', '~'):
        return None
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'"):
        return text[1:-1]
    if text.startswith('[') and text.endswith(']'):
        inner = text[1:-1].strip()
        return [_parse_scalar(p) for p in _split_flow(inner)] if inner else []
    if text.startswith('{') and text.endswith('}'):
        inner = text[1:-1].strip()
        out = {}
        if inner:
            for part in _split_flow(inner):
                k, _, v = part.partition(':')
                out[_parse_scalar(k)] = _parse_scalar(v)
        return out
    low = text.lower()
    if low in ('true', 'yes', 'on'):
        return True
    if low in ('false', 'no', 'off'):
        return False
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def _fallback_load(yaml_text):
    """Parse a small subset of YAML: top-level mapping with scalars, flow
    collections, block mappings and block sequences."""
    if yaml_text is None:
        return None
    if isinstance(yaml_text, bytes):
        yaml_text = yaml_text.decode('utf-8')

    lines = []
    for raw in yaml_text.splitlines():
        line = _strip_comment(raw)
        if not line.strip() or line.strip() in ('---', '...'):
            continue
        lines.append(line)
    if not lines:
        return None

    root = {}
    stack = [(-1, root)]

    for index, line in enumerate(lines):
        indent = len(line) - len(line.lstrip())
        body = line.strip()

        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if body.startswith('- ') or body == '-':
            rest = body[2:].strip() if body != '-' else ''
            if isinstance(parent, dict):
                # first item of a block sequence: convert container to a list
                prev = stack[-2][1] if len(stack) >= 2 else None
                seq = []
                if isinstance(prev, dict):
                    for k in list(prev):
                        if prev[k] is parent:
                            prev[k] = seq
                            break
                stack.pop()
                stack.append((indent, seq))
                parent = seq
            if rest:
                if ':' in rest and not (rest[0] in ('"', "'")):
                    k, _, v = rest.partition(':')
                    item = {_parse_scalar(k): _parse_scalar(v)}
                else:
                    item = _parse_scalar(rest)
                parent.append(item)
                if isinstance(item, (dict, list)):
                    stack.append((indent, item))
            else:
                child = {}
                parent.append(child)
                stack.append((indent, child))
        else:
            if ':' not in body:
                continue
            k, _, v = body.partition(':')
            k = _parse_scalar(k)
            v = v.strip()
            if v == '':
                # decide between a nested mapping and a nested sequence
                child_indent = None
                for nxt in lines[index + 1:]:
                    ni = len(nxt) - len(nxt.lstrip())
                    if ni <= indent:
                        break
                    if child_indent is None:
                        child_indent = ni
                is_seq = False
                if child_indent is not None:
                    for nxt in lines[index + 1:]:
                        ni = len(nxt) - len(nxt.lstrip())
                        if ni < child_indent:
                            break
                        if ni == child_indent:
                            nb = nxt.strip()
                            if nb.startswith('- ') or nb == '-':
                                is_seq = True
                            break
                child = [] if is_seq else {}
                parent[k] = child
                stack.append((indent, child))
            else:
                parent[k] = _parse_scalar(v)

    return root


def _load(yaml_text):
    try:
        import yaml
    except ImportError:
        return _fallback_load(yaml_text)
    try:
        return yaml.safe_load(yaml_text)
    except Exception:
        return _fallback_load(yaml_text)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def _cosine(value):
    """Cosine of a numeric value; non-numeric values are returned unchanged."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return math.cos(value)
    if isinstance(value, str):
        stripped = value.strip()
        try:
            return math.cos(int(stripped))
        except ValueError:
            pass
        try:
            return math.cos(float(stripped))
        except ValueError:
            return value
    return value


def cosine_yaml_key(yaml_text, key):
    """Parse ``yaml_text`` and replace the value under ``key`` by its cosine.

    ``yaml_text`` is the text of a YAML document whose top level is a
    mapping.  The returned dict contains every value exactly as parsed, with
    the single exception that the number stored under ``key`` is replaced by
    its cosine (the angle is interpreted in radians).  If ``key`` is not
    present in the document, the data is returned unchanged.
    """
    data = _load(yaml_text)
    if not isinstance(data, dict):
        return data
    if key in data:
        data[key] = _cosine(data[key])
    return data
````
