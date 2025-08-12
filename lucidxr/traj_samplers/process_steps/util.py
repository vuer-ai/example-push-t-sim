def pick(d, *keys, strict=False):
    """Pick keys"""
    _d = {}
    for k in keys:
        if k in d:
            _d[k] = d[k]
        elif strict:
            raise KeyError(k)
    return _d