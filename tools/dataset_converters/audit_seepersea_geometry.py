#!/usr/bin/env python3
"""Audit the raw SeaPersea annotations and geometry for BEVFusion (issue L6).

The script uses ``annotations/*.json`` as the candidate ground truth and checks
how it was converted into ``data/seepersea/*_infos_*.pkl``: class definitions,
box centre, dimension order, yaw unit/direction, point-cloud/3D-box projection,
in-range object ratio, points per box and the reason for empty label frames.

It only depends on numpy (matplotlib is optional, for one overlay image) and
does not import mmdet3d, so it can be run without compiling CUDA ops.

Usage::

    python tools/dataset_converters/audit_seepersea_geometry.py \
        --src-root seepersea-data/barbados-20230223-240 \
        --pkl data/seepersea/seepersea_infos_train.pkl \
        --out-dir outputs/seepersea_l6_audit
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# BEVFusion lidar-cam config (test_pipeline PointsRangeFilter)
PC_RANGE = np.array([-54.0, -54.0, -5.0, 54.0, 54.0, 3.0])
IMG_W, IMG_H = 640, 480
CLASSES = ('ship', 'buoy', 'other')
# Confirmed by the data provider (L1): buoy == Pedestrian, other == Cyclist.
CATEGORY_TO_KITTI_CONFIRMED = {
    'ship': 'Car',
    'buoy': 'Pedestrian',
    'other': 'Cyclist',
}
# Mapping currently hard-coded in seepersea2bevfusion.py.
CATEGORY_TO_KITTI_CURRENT = {
    'ship': 'Car',
    'buoy': 'Cyclist',
    'other': 'Pedestrian',
}
NUSCENES_CLASSES = (
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier')
KITTI_TO_NUSCENES_NAME = {
    'Car': 'car', 'Cyclist': 'bicycle', 'Pedestrian': 'pedestrian'}

EDGES = ((0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7),
         (4, 5), (4, 6), (5, 7), (6, 7))


# ---------------------------------------------------------------------------
# readers
# ---------------------------------------------------------------------------

def read_pcd(path: Path) -> np.ndarray:
    lines = path.read_text().splitlines()
    fields: List[str] = []
    data_idx = None
    for i, line in enumerate(lines):
        if line.startswith('FIELDS'):
            fields = line.split()[1:]
        elif line.startswith('DATA'):
            data_idx = i + 1
            break
    if data_idx is None:
        raise ValueError(f'{path}: no DATA section')
    rows = [line.split() for line in lines[data_idx:] if line.strip()]
    if not rows:
        return np.zeros((0, 4), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)[:, :len(fields)]


def read_calib(path: Path) -> Dict[str, np.ndarray]:
    calib = {}
    for line in path.read_text().splitlines():
        if ':' in line:
            key, value = line.split(':', 1)
            calib[key.strip()] = np.asarray([float(x) for x in value.split()])
    p2 = calib['P2'].reshape(3, 4)
    r0 = np.eye(4)
    r0[:3, :3] = calib.get('R0_rect', np.eye(9)).reshape(3, 3)
    tr = np.eye(4)
    tr[:3, :4] = calib['Tr_velo_to_cam'].reshape(3, 4)
    return dict(
        P2=p2,
        cam2img=p2[:3, :3],
        lidar2cam=r0 @ tr,
        lidar2img=p2 @ r0 @ tr,
        p2_last_col=p2[:, 3].copy())


def read_ann(path: Path) -> List[dict]:
    text = path.read_text().strip()
    if not text:
        return []
    return json.loads(text).get('labels', [])


def read_kitti(path: Path) -> List[dict]:
    out = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        out.append(dict(
            type=parts[0],
            alpha=float(parts[3]),
            bbox=np.array([float(x) for x in parts[4:8]]),
            dims=np.array([float(x) for x in parts[8:11]]),   # h, w, l
            loc=np.array([float(x) for x in parts[11:14]]),    # bottom centre
            ry=float(parts[14])))
    return out


def read_camera_label(path: Path) -> List[dict]:
    text = path.read_text().strip()
    if not text:
        return []
    shapes = json.loads(text).get('shapes', [])
    out = []
    for s in shapes:
        pts = np.asarray(s['points'], dtype=np.float64).reshape(-1, 2)
        x1y1 = pts.min(axis=0)
        x2y2 = pts.max(axis=0)
        out.append(dict(
            label=s['label'],
            bbox=np.array([x1y1[0], x1y1[1], x2y2[0], x2y2[1]])))
    return out


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def wrap_pi(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def box_of(label: dict) -> tuple:
    """Return (gravity centre, [l, w, h], yaw) of a raw annotation."""
    b = label['box3d']
    loc, dim, ori = b['location'], b['dimension'], b['orientation']
    centre = np.array([loc['x'], loc['y'], loc['z']], dtype=np.float64)
    # Candidate A: annotation 'width' == KITTI l (along heading),
    # annotation 'length' == KITTI w (lateral).  Verified below.
    dims = np.array([dim['width'], dim['length'], dim['height']],
                    dtype=np.float64)
    return centre, dims, float(ori.get('rotationYaw', 0.0))


def box_corners(centre: np.ndarray, dims: np.ndarray,
                yaw: float) -> np.ndarray:
    """8 corners, mmdet3d LiDARInstance3DBoxes (bottom-centre origin)."""
    l, w, h = dims
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    local = np.array([[x, y, z] for x in (-l / 2, l / 2)
                      for y in (-w / 2, w / 2) for z in (0.0, h)])
    bottom = centre - np.array([0.0, 0.0, h / 2])
    return local @ rot.T + bottom


def points_in_box(pts: np.ndarray, centre: np.ndarray, dims: np.ndarray,
                  yaw: float) -> np.ndarray:
    """Boolean mask of points inside a rotated 3D box (gravity centre)."""
    if len(pts) == 0:
        return np.zeros(0, dtype=bool)
    l, w, h = dims
    d = pts - centre
    c, s = np.cos(yaw), np.sin(yaw)
    lx = c * d[:, 0] + s * d[:, 1]
    ly = -s * d[:, 0] + c * d[:, 1]
    return (np.abs(lx) <= l / 2 + 1e-9) & \
           (np.abs(ly) <= w / 2 + 1e-9) & \
           (np.abs(d[:, 2]) <= h / 2 + 1e-9)


def project(pts: np.ndarray, calib: Dict[str, np.ndarray]):
    """Project LiDAR points to image pixels; return (uv, depth)."""
    homo = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    cam = homo @ calib['lidar2cam'].T
    depth = cam[:, 2]
    uvh = homo @ calib['lidar2img'].T
    uv = uvh[:, :2] / np.maximum(uvh[:, 2:3], 1e-9)
    return uv, depth


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.maximum(0.0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))
    ai = inter[0] * inter[1]
    ua = max((a[2] - a[0]) * (a[3] - a[1]), 0.0) + \
         max((b[2] - b[0]) * (b[3] - b[1]), 0.0) - ai
    return float(ai / ua) if ua > 0 else 0.0


def summarise(values) -> dict:
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return dict(n=0)
    return dict(
        n=int(v.size), min=float(v.min()), max=float(v.max()),
        mean=float(v.mean()), median=float(np.median(v)),
        p10=float(np.percentile(v, 10)), p90=float(np.percentile(v, 90)))


def ratio(num: int, den: int) -> float:
    return float(num / den) if den else 0.0


# ---------------------------------------------------------------------------
# main audit
# ---------------------------------------------------------------------------

def audit(src: Path, pkl_path: Optional[Path], out_dir: Path) -> dict:
    frames = sorted(int(p.stem) for p in (src / 'point_clouds').glob('*.pcd'))
    ann_dir, cal_dir = src / 'annotations', src / 'calib'
    kitti_dir, cam_dir = src / 'kitti_label', src / 'camera_label'
    img_dir = src / 'undist_images' / 'CAM_FRONT'
    ts_path = src / 'timestamp' / 'timestamp.txt'

    timestamps = {}
    if ts_path.is_file():
        for line in ts_path.read_text().splitlines():
            parts = line.strip().split(',')
            if len(parts) >= 2:
                try:
                    timestamps[int(parts[0])] = float(parts[1])
                except ValueError:
                    pass

    R: dict = dict(pc_range=PC_RANGE.tolist(), frames=len(frames))
    empty_ann, empty_cam, empty_kitti = set(), set(), set()
    cat_counts = collections.Counter()
    cam_label_counts = collections.Counter()
    kitti_type_counts = collections.Counter()
    ann_per_frame: Dict[int, int] = {}
    cam_shapes_per_frame: Dict[int, int] = {}
    kitti_rows_per_frame: Dict[int, int] = {}
    pts_per_frame: Dict[int, int] = {}
    pcd_hashes: Dict[int, tuple] = {}
    calib_signatures = collections.Counter()

    # raw geometry accumulators
    all_pts_xyz: List[np.ndarray] = []
    all_intensity: List[np.ndarray] = []
    pts_total = pts_in_range = pts_out_z = pts_out_xy = 0
    pts_proj_in_image = pts_proj_front = 0
    centres: Dict[str, List[np.ndarray]] = {c: [] for c in CLASSES}
    dims_by_cat: Dict[str, List[np.ndarray]] = {c: [] for c in CLASSES}
    yaws: List[float] = []
    yaw_by_cat: Dict[str, List[float]] = {c: [] for c in CLASSES}
    pitch_roll_nonzero = 0
    yaw_key_missing = 0
    unknown_categories = collections.Counter()
    id_reuse = 0

    per_box = {c: dict(
        total=0, in_range_3d=0, in_range_bev=0, all_corners_in=0,
        any_corner_in=0, proj_visible=0, z_out=0, xy_out=0,
        pts=[]) for c in CLASSES}
    pts_nearest_center: List[float] = []
    center_exact_point = 0
    center_total = 0
    exact_matches = 0
    exact_by_cat = collections.Counter()
    exact_iou: Dict[str, List[float]] = {'A': [], 'B': []}
    exact_center_err: Dict[str, List[float]] = {'A': [], 'B': []}
    inexact_dim_err: List[float] = []
    inexact_loc_err: List[float] = []
    kitti_match = collections.Counter()
    err_dim_a, err_dim_b = [], []
    err_loc_bottom, err_loc_gravity = [], []
    err_ry = {k: [] for k in ('minus_yaw_minus_pi2', 'yaw',
                              'minus_yaw_plus_pi2', 'yaw_minus_pi2',
                              'minus_yaw')}
    kitti_bbox_in_cam = 0
    kitti_bbox_area: List[float] = []
    cam_bbox_area: List[float] = []
    cam_bbox_area_by_cat: Dict[str, List[float]] = {
        c: [] for c in CLASSES}
    kitti_2d_unmatched = 0
    pts_gt_255 = unique_pts = 0
    depth_in_dbound = depth_in_dbound_image = 0
    depth_positive: List[np.ndarray] = []

    for fid in frames:
        stem = f'{fid:06d}'
        ann = read_ann(ann_dir / f'{stem}.json')
        kitti = read_kitti(kitti_dir / f'{stem}.txt')
        cam_shapes = read_camera_label(cam_dir / f'{stem}.json')
        calib = read_calib(cal_dir / f'{stem}.txt')
        pts = read_pcd(src / 'point_clouds' / f'{stem}.pcd')
        xyz = pts[:, :3]
        calib_signatures[tuple(np.round(calib['P2'].ravel(), 6))] += 1

        if not ann:
            empty_ann.add(fid)
        if not cam_shapes:
            empty_cam.add(fid)
        if not kitti:
            empty_kitti.add(fid)
        ann_per_frame[fid] = len(ann)
        cam_shapes_per_frame[fid] = len(cam_shapes)
        kitti_rows_per_frame[fid] = len(kitti)
        pts_per_frame[fid] = len(xyz)
        pcd_hashes[fid] = (len(xyz), tuple(np.round(xyz.ravel(), 4)[:12]))
        all_pts_xyz.append(xyz)
        all_intensity.append(pts[:, 3])
        for s in cam_shapes:
            cam_label_counts[s['label']] += 1
            b = s['bbox']
            area = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
            cam_bbox_area.append(area)
            if s['label'] in cam_bbox_area_by_cat:
                cam_bbox_area_by_cat[s['label']].append(area)
        for k in kitti:
            kitti_type_counts[k['type']] += 1
            b = k['bbox']
            kitti_bbox_area.append(max(0.0, (b[2] - b[0]) * (b[3] - b[1])))
            if any(np.allclose(b, s['bbox'], atol=1e-6) for s in cam_shapes):
                kitti_bbox_in_cam += 1
            else:
                kitti_2d_unmatched += 1

        # --- point cloud statistics -----------------------------------
        in_r = np.all((xyz >= PC_RANGE[:3]) & (xyz <= PC_RANGE[3:]), axis=1)
        pts_total += len(xyz)
        pts_in_range += int(in_r.sum())
        pts_out_z += int(((xyz[:, 2] < PC_RANGE[2]) |
                          (xyz[:, 2] > PC_RANGE[5])).sum())
        pts_out_xy += int((np.any(xyz[:, :2] < PC_RANGE[:2], axis=1) |
                           np.any(xyz[:, :2] > PC_RANGE[3:5], axis=1)).sum())
        uv, depth = project(xyz, calib)
        in_img = ((depth > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < IMG_W) &
                  (uv[:, 1] >= 0) & (uv[:, 1] < IMG_H))
        pts_proj_front += int((depth > 0).sum())
        pts_proj_in_image += int(in_img.sum())
        band = (depth >= 1.0) & (depth <= 60.0)  # LSS dbound
        depth_in_dbound += int(band.sum())
        depth_in_dbound_image += int((band & in_img).sum())
        depth_positive.append(depth[depth > 0])
        pts_gt_255 += int((pts[:, 3] > 255).sum())
        unique_pts += len(np.unique(np.round(xyz, 4), axis=0))

        # --- boxes -----------------------------------------------------
        ids_seen = collections.defaultdict(set)
        boxes = []
        for lab in ann:
            cat = lab.get('category')
            if cat not in CLASSES:
                unknown_categories[cat] += 1
                continue
            c, d, yaw = box_of(lab)
            ori = lab['box3d']['orientation']
            if 'rotationYaw' not in ori:
                yaw_key_missing += 1
            if abs(float(ori.get('rotationPitch', 0))) > 1e-9 or \
                    abs(float(ori.get('rotationRoll', 0))) > 1e-9:
                pitch_roll_nonzero += 1
            if lab.get('id') in ids_seen[cat]:
                id_reuse += 1
            ids_seen[cat].add(lab.get('id'))
            cat_counts[cat] += 1
            centres[cat].append(c)
            dims_by_cat[cat].append(d)
            yaws.append(yaw)
            yaw_by_cat[cat].append(yaw)
            boxes.append((cat, c, d, yaw, lab))
            pb = per_box[cat]
            pb['total'] += 1
            pb['in_range_3d'] += int(np.all((c >= PC_RANGE[:3]) &
                                            (c <= PC_RANGE[3:]), axis=0))
            pb['in_range_bev'] += int((c[0] > PC_RANGE[0]) &
                                      (c[1] > PC_RANGE[1]) &
                                      (c[0] < PC_RANGE[3]) &
                                      (c[1] < PC_RANGE[4]))
            pb['z_out'] += int((c[2] < PC_RANGE[2]) | (c[2] > PC_RANGE[5]))
            pb['xy_out'] += int((c[0] < PC_RANGE[0]) | (c[0] > PC_RANGE[3]) |
                                (c[1] < PC_RANGE[1]) | (c[1] > PC_RANGE[4]))
            corners = box_corners(c, d, yaw)
            cin = np.all((corners >= PC_RANGE[:3]) &
                         (corners <= PC_RANGE[3:]), axis=1)
            pb['all_corners_in'] += int(cin.all())
            pb['any_corner_in'] += int(cin.any())
            _, dep_c = project(corners, calib)
            pb['proj_visible'] += int((dep_c > 0).all())
            # points inside the box under the two candidate dim orders
            pb['pts'].append(int(points_in_box(xyz, c, d, yaw).sum()))
            # centre vs. point cloud
            center_total += 1
            if len(xyz):
                dist = np.linalg.norm(xyz - c, axis=1).min()
                pts_nearest_center.append(float(dist))
                center_exact_point += int(dist < 1e-3)
            # swapped dims (hypothesis B): x_size=length, y_size=width
            d_swap = np.array([d[1], d[0], d[2]])
            pb.setdefault('pts_swap', []).append(
                int(points_in_box(xyz, c, d_swap, yaw).sum()))
            # 2D labels are a separate annotation source, so projection
            # consistency is checked below against exactly matched KITTI
            # rows (which share the 3D parameters) instead of a greedy
            # 2D/3D nearest-centre match that can explode for tiny depths.

        # --- KITTI cross-check -----------------------------------------
        for k in kitti:
            best = None
            for cat, c, d, yaw, lab in boxes:
                # candidate A: [h, w, l] = [height, length, width]
                pred_a = np.array([d[2], d[1], d[0]])
                bottom = c - np.array([0.0, 0.0, d[2] / 2])
                loc_b = (calib['lidar2cam'] @ np.r_[bottom, 1.0])[:3]
                loc_g = (calib['lidar2cam'] @ np.r_[c, 1.0])[:3]
                ry_pred = wrap_pi(-yaw - np.pi / 2)
                score = np.linalg.norm(
                    np.r_[pred_a - k['dims'], loc_b - k['loc'],
                          wrap_pi(ry_pred - k['ry'])])
                if best is None or score < best[0]:
                    best = (score, cat, c, d, yaw, pred_a, loc_b, loc_g,
                            ry_pred)
            if best is None:
                continue
            _, cat, c, d, yaw, pred_a, loc_b, loc_g, ry_pred = best
            kitti_match[(k['type'], cat)] += 1
            exact = max(np.abs(pred_a - k['dims']).max(),
                        np.abs(loc_b - k['loc']).max(),
                        abs(wrap_pi(ry_pred - k['ry']))) < 1e-5
            if exact:
                exact_matches += 1
                exact_by_cat[cat] += 1
                # Use exactly matched rows as trusted 2D/3D anchors.
                for tag, dims in (
                        ('A', d), ('B', np.array([d[1], d[0], d[2]]))):
                    corners = box_corners(c, dims, yaw)
                    uv_b, dep_b = project(corners, calib)
                    if not (dep_b > 1.0).all():
                        continue
                    pb2d = np.array([uv_b[:, 0].min(), uv_b[:, 1].min(),
                                     uv_b[:, 0].max(), uv_b[:, 1].max()])
                    if (pb2d[2] - pb2d[0]) > 4 * IMG_W or \
                            (pb2d[3] - pb2d[1]) > 4 * IMG_H:
                        continue
                    exact_iou[tag].append(bbox_iou(pb2d, k['bbox']))
                    exact_center_err[tag].append(float(np.linalg.norm(
                        (pb2d[:2] + pb2d[2:]) / 2 -
                        (k['bbox'][:2] + k['bbox'][2:]) / 2)))
            else:
                inexact_dim_err.append(float(np.abs(pred_a - k['dims']).max()))
                inexact_loc_err.append(float(np.abs(loc_b - k['loc']).max()))
            pred_b = np.array([d[2], d[0], d[1]])
            err_dim_a.append(float(np.linalg.norm(pred_a - k['dims'])))
            err_dim_b.append(float(np.linalg.norm(pred_b - k['dims'])))
            err_loc_bottom.append(float(np.linalg.norm(loc_b - k['loc'])))
            err_loc_gravity.append(float(np.linalg.norm(loc_g - k['loc'])))
            err_ry['minus_yaw_minus_pi2'].append(abs(wrap_pi(
                -yaw - np.pi / 2 - k['ry'])))
            err_ry['yaw'].append(abs(wrap_pi(yaw - k['ry'])))
            err_ry['minus_yaw_plus_pi2'].append(abs(wrap_pi(
                -yaw + np.pi / 2 - k['ry'])))
            err_ry['yaw_minus_pi2'].append(abs(wrap_pi(
                yaw - np.pi / 2 - k['ry'])))
            err_ry['minus_yaw'].append(abs(wrap_pi(-yaw - k['ry'])))

    # --- aggregate --------------------------------------------------------
    R['files'] = dict(
        empty_annotations=sorted(empty_ann),
        empty_camera_label=sorted(empty_cam),
        empty_kitti_label=sorted(empty_kitti),
        empty_annotations_count=len(empty_ann),
        empty_camera_label_count=len(empty_cam),
        empty_kitti_label_count=len(empty_kitti),
        annotations_and_camera_label_empty_sets_equal=(empty_ann == empty_cam),
        empty_sets_identical=(empty_ann == empty_cam == empty_kitti),
        calib_unique_p2=len(calib_signatures),
        p2_last_col_max_abs=float(max(
            np.abs(read_calib(cal_dir / f'{fid:06d}.txt')['p2_last_col']).max()
            for fid in frames)),
        unique_pcd_xyz_ratio=ratio(unique_pts, pts_total),
        intensity_gt_255_ratio=ratio(pts_gt_255, pts_total),
        images_present=sum((img_dir / f'{fid:06d}.png').is_file()
                           for fid in frames))
    R['categories'] = dict(
        annotations=dict(cat_counts),
        camera_label=dict(cam_label_counts),
        kitti_label=dict(kitti_type_counts),
        unknown_annotations=dict(unknown_categories),
        kitti_type_by_annotation_category={
            k[0]: collections.Counter() for k in kitti_match},
        confirmed_mapping=CATEGORY_TO_KITTI_CONFIRMED,
        current_mapping_in_converter=CATEGORY_TO_KITTI_CURRENT,
        id_reused_within_category=id_reuse,
        annotations_boxes=sum(cat_counts.values()),
        kitti_rows=sum(kitti_type_counts.values()),
        camera_label_boxes=sum(cam_label_counts.values()),
        camera_label_2d_area_by_category={
            c: summarise(v) for c, v in cam_bbox_area_by_cat.items()})
    for (ktype, cat), n in kitti_match.items():
        R['categories']['kitti_type_by_annotation_category'][ktype][cat] = n
    R['categories']['kitti_type_by_annotation_category'] = {
        k: dict(v) for k, v in
        R['categories']['kitti_type_by_annotation_category'].items()}

    R['kitti_crosscheck'] = dict(
        matched_pairs={f'{a}->{b}': n for (a, b), n in kitti_match.items()},
        matched_rows=sum(kitti_match.values()),
        exact_geometric_matches=exact_matches,
        exact_geometric_match_ratio=ratio(exact_matches,
                                          sum(kitti_match.values())),
        exact_matches_by_annotation_category=dict(exact_by_cat),
        inexact_rows=len(inexact_dim_err),
        inexact_max_dim_abs_err=summarise(inexact_dim_err),
        inexact_max_loc_abs_err=summarise(inexact_loc_err),
        dim_err_hypothesis_A_lwh_from_ann=summarise(err_dim_a),
        dim_err_hypothesis_B_wlh_from_ann=summarise(err_dim_b),
        loc_err_bottom_centre=summarise(err_loc_bottom),
        loc_err_gravity_centre=summarise(err_loc_gravity),
        yaw_err_by_formula={k: summarise(v) for k, v in err_ry.items()},
        kitti_2d_boxes_found_in_camera_label=kitti_bbox_in_cam,
        kitti_2d_boxes_unmatched=kitti_2d_unmatched,
        kitti_2d_box_area=summarise(kitti_bbox_area),
        camera_label_2d_box_area=summarise(cam_bbox_area))

    R['yaw'] = dict(
        overall=summarise(yaws),
        by_category={c: summarise(v) for c, v in yaw_by_cat.items()},
        abs_gt_pi=int(sum(abs(y) > np.pi for y in yaws)),
        pitch_roll_nonzero=pitch_roll_nonzero,
        rotationYaw_missing=yaw_key_missing,
        most_common_values=[
            [float(v), int(n)] for v, n in
            collections.Counter(np.round(yaws, 3)).most_common(10)])

    for cat in CLASSES:
        d = np.asarray(dims_by_cat[cat]).reshape(-1, 3)
        c = np.asarray(centres[cat]).reshape(-1, 3)
        R.setdefault('dims_by_category', {})[cat] = dict(
            width_aka_l=summarise(d[:, 0]), length_aka_w=summarise(d[:, 1]),
            height=summarise(d[:, 2]),
            width_eq_length_ratio=ratio(int((np.abs(d[:, 0] - d[:, 1]) <
                                             1e-6).sum()), len(d)),
            centre_x=summarise(c[:, 0]), centre_y=summarise(c[:, 1]),
            centre_z=summarise(c[:, 2]))
    all_xyz = np.concatenate(all_pts_xyz, axis=0)
    all_int = np.concatenate(all_intensity, axis=0)
    all_ctr = np.concatenate([np.asarray(centres[c]).reshape(-1, 3)
                              for c in CLASSES], axis=0)
    depth_all = np.concatenate(depth_positive)
    R['range_coverage'] = {}
    for xy in (54, 75, 100.8, 108):
        for zr in ((-5, 3), (-10, 10), (-20, 20)):
            pm = (np.abs(all_xyz[:, 0]) <= xy) & \
                 (np.abs(all_xyz[:, 1]) <= xy) & \
                 (all_xyz[:, 2] >= zr[0]) & (all_xyz[:, 2] <= zr[1])
            cm = (np.abs(all_ctr[:, 0]) <= xy) & \
                 (np.abs(all_ctr[:, 1]) <= xy) & \
                 (all_ctr[:, 2] >= zr[0]) & (all_ctr[:, 2] <= zr[1])
            R['range_coverage'][f'xy±{xy}_z{zr[0]}..{zr[1]}'] = dict(
                point_ratio=float(pm.mean()), box_center_ratio=float(cm.mean()))
    R['point_cloud'] = dict(
        total_points=pts_total,
        per_frame_points=summarise(list(pts_per_frame.values())),
        in_range_3d=pts_in_range,
        in_range_3d_ratio=ratio(pts_in_range, pts_total),
        out_of_range_points=pts_total - pts_in_range,
        out_of_z_only=pts_out_z,
        out_of_xy_only=pts_out_xy,
        x=summarise(all_xyz[:, 0]), y=summarise(all_xyz[:, 1]),
        z=summarise(all_xyz[:, 2]),
        intensity=summarise(all_int),
        proj_depth_positive=pts_proj_front,
        proj_depth_positive_ratio=ratio(pts_proj_front, pts_total),
        proj_in_image=pts_proj_in_image,
        proj_in_image_ratio=ratio(pts_proj_in_image, pts_total),
        depth_in_LSS_dbound_1_60=depth_in_dbound,
        depth_in_LSS_dbound_ratio=ratio(depth_in_dbound, pts_total),
        depth_in_dbound_and_image=depth_in_dbound_image,
        depth_in_dbound_and_image_ratio=ratio(depth_in_dbound_image,
                                             pts_total),
        depth_percentiles={
            p: float(np.percentile(depth_all, p))
            for p in (10, 50, 90, 99)},
        unique_xyz_ratio=ratio(unique_pts, pts_total),
        intensity_gt_255=pts_gt_255,
        intensity_gt_255_ratio=ratio(pts_gt_255, pts_total),
        xyz=dict(
            x_lt_xmin=int((all_xyz[:, 0] < PC_RANGE[0]).sum()),
            x_gt_xmax=int((all_xyz[:, 0] > PC_RANGE[3]).sum()),
            y_lt_ymin=int((all_xyz[:, 1] < PC_RANGE[1]).sum()),
            y_gt_ymax=int((all_xyz[:, 1] > PC_RANGE[4]).sum()),
            z_lt_zmin=int((all_xyz[:, 2] < PC_RANGE[2]).sum()),
            z_gt_zmax=int((all_xyz[:, 2] > PC_RANGE[5]).sum())))

    R['boxes'] = {}
    for cat in CLASSES:
        pb = per_box[cat]
        R['boxes'][cat] = dict(
            total=pb['total'],
            centre_in_range_3d=pb['in_range_3d'],
            centre_in_range_3d_ratio=ratio(pb['in_range_3d'], pb['total']),
            centre_in_range_bev_ObjectRangeFilter=pb['in_range_bev'],
            centre_in_range_bev_ratio=ratio(pb['in_range_bev'], pb['total']),
            all_corners_in_range_3d=pb['all_corners_in'],
            any_corner_in_range_3d=pb['any_corner_in'],
            all_corners_front_of_camera=pb['proj_visible'],
            centre_z_out_of_range=pb['z_out'],
            centre_xy_out_of_range=pb['xy_out'],
            points_per_box=summarise(pb['pts']),
            points_per_box_zeros=int(sum(p == 0 for p in pb['pts'])),
            points_per_box_one=int(sum(p == 1 for p in pb['pts'])),
            points_per_box_ge2=int(sum(p >= 2 for p in pb['pts'])),
            points_per_box_ge5=int(sum(p >= 5 for p in pb['pts'])),
            points_per_box_swapped_dims=summarise(pb['pts_swap']))
    tot = sum(per_box[c]['total'] for c in CLASSES)
    R['boxes']['overall'] = dict(
        total=tot,
        centre_in_range_3d=sum(per_box[c]['in_range_3d'] for c in CLASSES),
        centre_in_range_bev=sum(per_box[c]['in_range_bev'] for c in CLASSES),
        points_per_box=summarise(
            [p for c in CLASSES for p in per_box[c]['pts']]),
        points_per_box_zeros=sum(
            sum(p == 0 for p in per_box[c]['pts']) for c in CLASSES))

    R['projection'] = dict(
        valid_projected_exact_match_anchors=len(exact_iou['A']),
        projection_iou_A=summarise(exact_iou['A']),
        projection_iou_B=summarise(exact_iou['B']),
        projection_centre_err_A_px=summarise(exact_center_err['A']),
        projection_centre_err_B_px=summarise(exact_center_err['B']),
        boxes_all_corners_in_front_of_camera=sum(
            per_box[c]['proj_visible'] for c in CLASSES),
        centre_nearest_point_distance=summarise(pts_nearest_center),
        centre_exactly_matches_a_point=center_exact_point,
        centre_exactly_matches_a_point_ratio=ratio(center_exact_point,
                                                   center_total),
        note='2D camera_label boxes are an independent annotation source; '
             'IoU is computed against exactly matched KITTI rows only.')

    # --- empty annotation frames ------------------------------------------
    empty = sorted(empty_ann)
    nonempty = [f for f in frames if f not in empty_ann]
    prev_ann = {f: ann_per_frame[f] for f in frames}
    ef = []
    for f in empty:
        prev_f = max((x for x in nonempty if x < f), default=None)
        next_f = min((x for x in nonempty if x > f), default=None)
        ef.append(dict(
            frame=f,
            n_points=pts_per_frame[f],
            camera_label_shapes=cam_shapes_per_frame[f],
            kitti_rows=kitti_rows_per_frame[f],
            timestamp=timestamps.get(f),
            prev_nonempty_frame=prev_f,
            prev_nonempty_boxes=prev_ann.get(prev_f) if prev_f else None,
            next_nonempty_frame=next_f,
            next_nonempty_boxes=prev_ann.get(next_f) if next_f else None,
            image_isfile=(img_dir / f'{f:06d}.png').is_file(),
            pcd_identical_to_prev=(pcd_hashes[f] == pcd_hashes.get(f - 1))))
    R['empty_frames'] = dict(
        count=len(empty),
        frames=empty,
        kitti_label_empty_frames=len(empty_kitti),
        kitti_empty_extra_vs_annotations=len(empty_kitti - empty_ann),
        camera_label_empty_set_equals_annotations=(empty_ann == empty_cam),
        points_per_empty_frame=summarise([pts_per_frame[f] for f in empty]),
        points_per_annotated_frame=summarise(
            [pts_per_frame[f] for f in nonempty]),
        frames_with_camera_label=sum(cam_shapes_per_frame[f] > 0 for f in empty),
        frames_with_kitti_rows=sum(kitti_rows_per_frame[f] > 0 for f in empty),
        frames_with_image=sum((img_dir / f'{f:06d}.png').is_file()
                              for f in empty),
        pcd_identical_to_prev_frame=sum(e['pcd_identical_to_prev'] for e in ef),
        timestamp_anomalies=sum(
            (timestamps.get(f, 0) <= 0) for f in empty),
        neighbour_boxes_both_sides=sum(
            (e['prev_nonempty_boxes'] or 0) > 0 and
            (e['next_nonempty_boxes'] or 0) > 0 for e in ef),
        sample=ef[:8])
    R['annotations_per_frame'] = summarise(list(ann_per_frame.values()))
    R['kitti_rows_per_frame'] = summarise(list(kitti_rows_per_frame.values()))
    R['camera_label_shapes_per_frame'] = summarise(
        list(cam_shapes_per_frame.values()))

    # --- converted pkl ----------------------------------------------------
    R['pkl_check'] = dict(path=str(pkl_path) if pkl_path else None)
    if pkl_path and pkl_path.is_file():
        with open(pkl_path, 'rb') as f:
            pkl = pickle.load(f)
        infos = pkl['data_list']
        label_counts = collections.Counter()
        n_box = 0
        max_geom_diff = 0.0
        bad_frames = 0
        for info in infos:
            fid = int(info['sample_idx'])
            raw = [box_of(l) for l in read_ann(ann_dir / f'{fid:06d}.json')
                   if l.get('category') in CLASSES]
            inst = info['instances']
            if len(inst) != len(raw):
                bad_frames += 1
                continue
            for i, (c, d, yaw) in enumerate(raw):
                n_box += 1
                label_counts[int(inst[i]['bbox_label_3d'])] += 1
                got = np.asarray(inst[i]['bbox_3d'], dtype=np.float64)
                expected = np.r_[c, d, yaw]
                max_geom_diff = max(max_geom_diff,
                                    float(np.abs(got - expected).max()))
        R['pkl_check'].update(
            frames=len(infos), boxes=n_box,
            frames_with_box_count_mismatch=bad_frames,
            max_abs_bbox_3d_diff_vs_raw_annotation=max_geom_diff,
            bbox_label_3d_counts={
                NUSCENES_CLASSES[k]: v for k, v in sorted(label_counts.items())},
            metainfo_categories=pkl['metainfo'].get('categories'),
            label_map_current={
                cat: KITTI_TO_NUSCENES_NAME[CATEGORY_TO_KITTI_CURRENT[cat]]
                for cat in CLASSES},
            label_map_after_L1_fix={
                cat: KITTI_TO_NUSCENES_NAME[
                    CATEGORY_TO_KITTI_CONFIRMED[cat]] for cat in CLASSES},
            label_ids_after_L1_fix={
                cat: NUSCENES_CLASSES.index(KITTI_TO_NUSCENES_NAME[
                    CATEGORY_TO_KITTI_CONFIRMED[cat]]) for cat in CLASSES},
            expected_label_counts_after_L1_fix={
                cat: int(cat_counts[cat]) for cat in CLASSES})

    # --- overlay image for one frame --------------------------------------
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.image import imread
        fid = frames[0]
        img = imread(img_dir / f'{fid:06d}.png')
        calib = read_calib(cal_dir / f'{fid:06d}.txt')
        xyz = read_pcd(src / 'point_clouds' / f'{fid:06d}.pcd')[:, :3]
        uv, depth = project(xyz, calib)
        fig, ax = plt.subplots(figsize=(10, 7.5))
        ax.imshow(img)
        ok = (depth > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < IMG_W) & \
             (uv[:, 1] >= 0) & (uv[:, 1] < IMG_H)
        ax.scatter(uv[ok, 0], uv[ok, 1], s=9, c='yellow',
                   edgecolors='black', linewidths=0.3, label='PCD points')
        for lab in read_ann(ann_dir / f'{fid:06d}.json'):
            if lab.get('category') not in CLASSES:
                continue
            c, d, yaw = box_of(lab)
            corners = box_corners(c, d, yaw)
            cuv, cdep = project(corners, calib)
            if not (cdep > 0).all():
                continue
            for a, b in EDGES:
                ax.plot([cuv[a, 0], cuv[b, 0]], [cuv[a, 1], cuv[b, 1]],
                        color='lime', linewidth=0.9)
        for s in read_camera_label(cam_dir / f'{fid:06d}.json'):
            x1, y1, x2, y2 = s['bbox']
            ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1],
                    color='red', linewidth=0.8, linestyle='--')
        ax.set_xlim(0, IMG_W)
        ax.set_ylim(IMG_H, 0)
        ax.set_title(f'frame {fid:06d}: green=3D boxes (l,w,h + yaw), '
                     'red dashed=2D labels, yellow=PCD')
        ax.legend(loc='upper right', fontsize=8)
        overlay = out_dir / f'projection_{fid:06d}.png'
        fig.savefig(overlay, dpi=120, bbox_inches='tight')
        plt.close(fig)
        R['projection']['overlay_image'] = str(overlay)
    except Exception as exc:  # pragma: no cover - optional visual aid
        R['projection']['overlay_image'] = f'not generated: {exc}'

    return R


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--src-root',
                        default='seepersea-data/barbados-20230223-240')
    parser.add_argument('--pkl', default='data/seepersea/'
                        'seepersea_infos_train.pkl')
    parser.add_argument('--out-dir', default='outputs/seepersea_l6_audit')
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = Path(args.pkl) if args.pkl else None
    R = audit(Path(args.src_root), pkl_path, out_dir)
    out_json = out_dir / 'audit_stats.json'
    out_json.write_text(json.dumps(R, indent=2, ensure_ascii=False))
    print(f'[audit] frames={R["frames"]}  boxes='
          f'{R["categories"]["annotations_boxes"]}  '
          f'points={R["point_cloud"]["total_points"]}')
    print(f'[audit] empty annotation frames={R["empty_frames"]["count"]}')
    print(f'[audit] stats -> {out_json}')


if __name__ == '__main__':
    main()
