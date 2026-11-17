"""SeaPersea-specific data transforms."""
from __future__ import annotations

import numpy as np
from mmcv.transforms.base import BaseTransform
from mmdet3d.registry import TRANSFORMS


@TRANSFORMS.register_module()
class SeaPerseaClipIntensity(BaseTransform):
    """Clip only the intensity column of the in-memory point tensor.

    The on-disk .bin keeps raw intensity; no other point column is touched.
    """

    def __init__(self, min_value: float = 0.0, max_value: float = 255.0):
        self.min_value = float(min_value)
        self.max_value = float(max_value)

    def transform(self, results: dict) -> dict:
        points = results['points']
        tensor = points.tensor
        if tensor.shape[-1] < 4:
            raise ValueError(f'expected >=4 point features, got {tensor.shape}')
        intensity = tensor[:, 3]
        if hasattr(intensity, 'clamp_'):
            intensity.clamp_(self.min_value, self.max_value)
        else:
            intensity[...] = np.clip(
                np.asarray(intensity), self.min_value, self.max_value)
        points.tensor = tensor
        return results
