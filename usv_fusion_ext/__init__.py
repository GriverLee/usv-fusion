"""Registration entry point for the SeaPersea BEVFusion adapter.

Model modules are imported only when mmdet3d is importable, so pure data
conversion can run before a model environment is configured.
"""
try:
    import mmdet3d  # noqa: F401
except ModuleNotFoundError:
    mmdet3d = None

if mmdet3d is not None:
    from . import dataset, transforms  # noqa: F401

__all__ = ['dataset', 'transforms']
