"""Stages looked up by name, so a scenario can name a solver without importing it."""
_STAGES = {}


def register(cls):
    if not getattr(cls, "name", ""):
        raise ValueError("a stage needs a name")
    _STAGES[cls.name] = cls
    return cls


def create(name, **kwargs):
    try:
        cls = _STAGES[name]
    except KeyError as exc:
        known = ", ".join(sorted(_STAGES))
        raise KeyError(f"unknown stage {name!r}; known: {known}") from exc
    return cls(**kwargs)


def known():
    return tuple(sorted(_STAGES))
