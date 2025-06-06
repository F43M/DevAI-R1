def safe_load(stream):
    content = stream.read() if hasattr(stream, "read") else str(stream)
    data = {}
    for line in content.strip().splitlines():
        if ':' in line:
            k, v = line.split(':', 1)
            data[k.strip()] = _parse_value(v.strip())
    return data

def _parse_value(val):
    if val.isdigit():
        return int(val)
    try:
        return float(val)
    except ValueError:
        return val
