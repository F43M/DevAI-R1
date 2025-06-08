class BaseModel:
    def __init__(self, **data):
        for k, v in data.items():
            setattr(self, k, v)


def Field(default=None, **kwargs):
    return default


def validator(*fields, **kwargs):
    def decorator(func):
        return func

    return decorator
