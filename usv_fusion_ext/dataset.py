"""Native three-class SeaPersea dataset for MMDetection3D 1.4.0."""
from __future__ import annotations

import numpy as np
from mmdet3d.datasets import Det3DDataset
from mmdet3d.registry import DATASETS
from mmdet3d.structures import LiDARInstance3DBoxes

from .runtime import CLASS_ORDER, DATA_VERSION


@DATASETS.register_module()
class SeaPerseaDataset(Det3DDataset):
    """Single-frame LiDAR+CAM_FRONT dataset with native ship/buoy/other IDs."""

    METAINFO = dict(
        classes=CLASS_ORDER,
        palette=[(255, 0, 0), (0, 255, 0), (0, 0, 255)],
        version=DATA_VERSION)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        meta = self.metainfo
        if meta.get('dataset') != 'seepersea' or meta.get('version') != DATA_VERSION:
            raise ValueError(
                f'SeaPerseaDataset rejects pkl metainfo: dataset='
                f'{meta.get("dataset")!r}, version={meta.get("version")!r}')
        if tuple(meta.get('classes', ())) != CLASS_ORDER:
            raise ValueError(f'SeaPerseaDataset expects {CLASS_ORDER} classes')
        if meta.get('categories') != {name: i for i, name in enumerate(CLASS_ORDER)}:
            raise ValueError('SeaPerseaDataset expects native 0/1/2 category IDs')
        if meta.get('box_origin') != 'gravity_center' or \
                meta.get('box_dim_order') != 'lwh':
            raise ValueError('SeaPerseaDataset expects gravity-center lwh boxes')

    def parse_ann_info(self, info: dict) -> dict:
        ann_info = super().parse_ann_info(info)
        if ann_info is None:
            ann_info = dict(
                gt_bboxes_3d=np.zeros((0, 7), dtype=np.float32),
                gt_labels_3d=np.zeros((0, ), dtype=np.int64),
                instances=[])
        boxes = np.asarray(ann_info['gt_bboxes_3d'], dtype=np.float32)
        labels = np.asarray(ann_info['gt_labels_3d'], dtype=np.int64)
        if boxes.ndim != 2 or boxes.shape[1] != 7:
            raise ValueError(f'invalid gt_bboxes_3d shape {boxes.shape}')
        ann_info['gt_bboxes_3d'] = LiDARInstance3DBoxes(
            boxes, box_dim=7, origin=(0.5, 0.5, 0.5))
        ann_info['gt_labels_3d'] = labels
        return ann_info
