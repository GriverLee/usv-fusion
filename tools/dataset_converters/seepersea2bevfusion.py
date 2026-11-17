#!/usr/bin/env python3
"""Convert a SeaPersea clip to the legacy ten-class nuScenes-style pkl.

This is NOT the formal three-class training entry point. The formal dataset
version ``seepersea_3class_v1`` is produced by ``build_seepersea_v1.py``.

The converted layout mimics nuScenes so that ``NuScenesDataset`` and the
official BEVFusion configs can read it without any change to the model::

    data/seepersea/
        samples/LIDAR_TOP/000000.bin   # float32, 5 channels: x,y,z,intensity,0
        samples/CAM_FRONT/000000.png   # symlink (or copy) of undist image
        seepersea_infos_train.pkl      # all frames
        seepersea_infos_val.pkl        # a small subset for fast validation

Usage::

    python tools/dataset_converters/seepersea2bevfusion.py \
        --src-root seepersea-data/barbados-20230223-240 \
        --out-dir data/seepersea
"""
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
from typing import Dict, List, Optional

import mmengine
import numpy as np

# Legacy ten-class inference converter only. NOT the formal three-class
# training entry point; use build_seepersea_v1.py for seepersea_3class_v1.
# The three semantic classes of this dataset and their KITTI aliases.
CATEGORY_TO_KITTI = {
    'ship': 'Car',
    'buoy': 'Pedestrian',
    'other': 'Cyclist',
}

# BEVFusion is pretrained on nuScenes (10 classes). Keep the head untouched
# and project the three KITTI aliases onto nuScenes classes. The ids below
# follow ``NuScenesDataset.METAINFO['classes']`` (the standard nuScenes order
# expected inside a v2 info pkl); ``Det3DDataset`` will re-map them to the
# class order declared by the BEVFusion config.
NUSCENES_CLASSES = (
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier')
KITTI_TO_NUSCENES_ID = {
    'Car': NUSCENES_CLASSES.index('car'),
    'Cyclist': NUSCENES_CLASSES.index('bicycle'),
    'Pedestrian': NUSCENES_CLASSES.index('pedestrian'),
}

EYE4 = np.eye(4, dtype=np.float64)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--src-root', required=True, help='clip root')
    parser.add_argument('--out-dir', default='data/seepersea')
    parser.add_argument(
        '--max-frames',
        type=int,
        default=-1,
        help='convert only the first N frames; -1 means all (default)')
    parser.add_argument(
        '--val-size',
        type=int,
        default=8,
        help='number of frames in the quick-validation info pkl')
    parser.add_argument(
        '--copy-images',
        action='store_true',
        help='copy images instead of creating symlinks')
    return parser.parse_args()


def read_pcd_ascii(path: str) -> np.ndarray:
    """Read an ASCII PCD with FIELDS x y z intensity."""
    with open(path, 'r') as f:
        lines = f.read().splitlines()
    data_idx = None
    fields = []
    for i, line in enumerate(lines):
        if line.startswith('FIELDS'):
            fields = line.split()[1:]
        if line.startswith('DATA'):
            if line.split()[1].lower() != 'ascii':
                raise ValueError(f'{path}: only ASCII PCD is supported')
            data_idx = i + 1
            break
    if data_idx is None or fields[:4] != ['x', 'y', 'z', 'intensity']:
        raise ValueError(f'{path}: unexpected PCD header {fields}')
    rows = [l.split() for l in lines[data_idx:] if l.strip()]
    if not rows:
        return np.zeros((0, 4), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)[:, :4]


def read_calib(path: str) -> Dict[str, np.ndarray]:
    """Parse a KITTI-style calib file into cam2img (3x3) / lidar2cam (4x4)."""
    calib = {}
    with open(path, 'r') as f:
        for line in f:
            if ':' not in line:
                continue
            key, value = line.split(':', 1)
            calib[key.strip()] = np.asarray(
                [float(x) for x in value.split()], dtype=np.float64)

    p2 = calib['P2'].reshape(3, 4)
    r0 = calib.get('R0_rect', np.eye(3)).reshape(3, 3)
    tr = calib['Tr_velo_to_cam'].reshape(3, 4)

    # Rectified-camera coordinates (KITTI convention): x_rect = R0 @ Tr @ x_lidar
    rect = np.eye(4, dtype=np.float64)
    rect[:3, :3] = r0
    tr4 = np.eye(4, dtype=np.float64)
    tr4[:3, :4] = tr
    lidar2cam = rect @ tr4

    return dict(cam2img=p2[:3, :3], lidar2cam=lidar2cam)


def read_timestamps(path: str) -> Dict[int, float]:
    raw = {}
    if not osp.isfile(path):
        return raw
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                try:
                    raw[int(parts[0])] = float(parts[1])
                except ValueError:
                    continue
    valid = [v for v in raw.values() if v > 0]
    first_valid = min(valid) if valid else 0.0
    # frame 0 of this clip carries a -99999999 sentinel
    return {k: (v if v > 0 else first_valid - 0.1) for k, v in raw.items()}


def read_instances(path: str) -> List[dict]:
    """Turn annotations/*.json (LiDAR-frame 3D boxes) into v2 instances."""
    if not osp.isfile(path):
        return []
    with open(path, 'r') as f:
        text = f.read().strip()
    if not text:
        return []
    ann = json.loads(text)
    instances = []
    for obj in ann.get('labels', []):
        category = obj.get('category')
        if category not in CATEGORY_TO_KITTI:
            continue
        box = obj['box3d']
        loc = box['location']
        dim = box['dimension']
        yaw = box['orientation'].get('rotationYaw', 0.0)
        label = KITTI_TO_NUSCENES_ID[CATEGORY_TO_KITTI[category]]
        # [x, y, z, l, w, h, yaw]; ann.width is the along-heading length l.
        bbox_3d = [
            float(loc['x']), float(loc['y']), float(loc['z']),
            float(dim['width']), float(dim['length']), float(dim['height']),
            float(yaw)
        ]
        instances.append(
            dict(
                bbox_3d=bbox_3d,
                bbox_label_3d=label,
                bbox_label=label,
                bbox_3d_isvalid=True,
                num_lidar_pts=1,  # >0 so NuScenesDataset keeps the box
                num_radar_pts=0,
                velocity=[0.0, 0.0]))  # NuScenesDataset.with_velocity=True
    return instances


def link_or_copy(src: str, dst: str, copy: bool) -> None:
    if osp.exists(dst) or osp.islink(dst):
        return
    if copy:
        import shutil
        shutil.copy2(src, dst)
    else:
        os.symlink(osp.abspath(src), dst)


def convert(args) -> None:
    src = osp.abspath(args.src_root)
    out = osp.abspath(args.out_dir)
    pts_dir = osp.join(out, 'samples', 'LIDAR_TOP')
    img_dir = osp.join(out, 'samples', 'CAM_FRONT')
    mmengine.mkdir_or_exist(pts_dir)
    mmengine.mkdir_or_exist(img_dir)

    frame_ids = sorted(
        int(osp.splitext(f)[0])
        for f in os.listdir(osp.join(src, 'point_clouds'))
        if f.endswith('.pcd'))
    if args.max_frames > 0:
        frame_ids = frame_ids[:args.max_frames]
    if not frame_ids:
        raise RuntimeError(f'no .pcd files under {src}/point_clouds')

    timestamps = read_timestamps(osp.join(src, 'timestamp', 'timestamp.txt'))
    fallback_ts = min([v for v in timestamps.values() if v > 0], default=0.0)

    metainfo = dict(
        categories={name: i for i, name in enumerate(CATEGORY_TO_KITTI)},
        dataset='seepersea',
        version='v1.0-custom',
        info_version='1.1')

    data_list = []
    for idx in frame_ids:
        stem = f'{idx:06d}'
        pcd_path = osp.join(src, 'point_clouds', stem + '.pcd')
        img_path = osp.join(src, 'undist_images', 'CAM_FRONT', stem + '.png')
        calib_path = osp.join(src, 'calib', stem + '.txt')
        ann_path = osp.join(src, 'annotations', stem + '.json')
        if not osp.isfile(img_path):
            raise FileNotFoundError(img_path)

        # ---- LiDAR: 4-dim ASCII PCD -> 5-dim float32 .bin ----
        pts = read_pcd_ascii(pcd_path)
        if pts.size:
            # nuScenes stores intensity in [0, 255]; clip for numerical sanity
            intensity = np.clip(pts[:, 3:4], 0.0, 255.0)
            pts5 = np.concatenate(
                [pts[:, :3], intensity, np.zeros((len(pts), 1))],
                axis=1).astype(np.float32)
        else:
            pts5 = np.zeros((0, 5), dtype=np.float32)
        pts5.tofile(osp.join(pts_dir, stem + '.bin'))

        # ---- camera ----
        link_or_copy(img_path, osp.join(img_dir, stem + '.png'),
                     args.copy_images)
        calib = read_calib(calib_path) if osp.isfile(calib_path) else dict(
            cam2img=np.eye(3), lidar2cam=EYE4)

        ts = timestamps.get(idx, fallback_ts + idx * 0.1)
        info = dict(
            sample_idx=idx,
            token=f'seepersea_{stem}',
            timestamp=float(ts),
            ego2global=EYE4.tolist(),
            lidar_points=dict(
                num_pts_feats=5,
                lidar_path=stem + '.bin',
                lidar2ego=EYE4.tolist()),
            images={
                'CAM_FRONT':
                dict(
                    img_path=stem + '.png',
                    cam2img=calib['cam2img'].tolist(),  # 3x3
                    lidar2cam=calib['lidar2cam'].tolist(),  # 4x4
                    cam2ego=EYE4.tolist(),
                    timestamp=float(ts))
            },
            instances=read_instances(ann_path))
        data_list.append(info)

    train_pkl = dict(metainfo=metainfo, data_list=data_list)
    val_pkl = dict(
        metainfo=metainfo, data_list=data_list[:max(1, args.val_size)])

    mmengine.dump(train_pkl, osp.join(out, 'seepersea_infos_train.pkl'))
    mmengine.dump(val_pkl, osp.join(out, 'seepersea_infos_val.pkl'))

    n_boxes = sum(len(x['instances']) for x in data_list)
    print(f'[seepersea2bevfusion] frames      : {len(data_list)}')
    print(f'[seepersea2bevfusion] 3D boxes    : {n_boxes}')
    print(f'[seepersea2bevfusion] output root : {out}')
    print(f'[seepersea2bevfusion] train pkl   : seepersea_infos_train.pkl')
    print(f'[seepersea2bevfusion] val pkl     : seepersea_infos_val.pkl '
          f'({len(val_pkl["data_list"])} frames)')


if __name__ == '__main__':
    convert(parse_args())
