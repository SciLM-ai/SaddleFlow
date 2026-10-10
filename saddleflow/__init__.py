__version__ = "0.0.1"

__all__ = ["SaddleFlow", "SaddleFlowSearch", "find_saddles"]


def __getattr__(name):
    # Lazy, so `import saddleflow` stays light for the training code paths.
    if name == "SaddleFlow":
        from .pretrained import SaddleFlow
        return SaddleFlow
    if name in ("SaddleFlowSearch", "find_saddles"):
        from . import search
        return getattr(search, name)
    raise AttributeError(f"module 'saddleflow' has no attribute {name!r}")
