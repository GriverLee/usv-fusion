"""Shared path/data/config helpers for the SeaPersea adapter.

Data helpers are intentionally independent of mmdet3d so that the converter,
validator and synthetic tests can run before any model environment is set up.
Model/config helpers import mmdet3d lazily and always take explicit roots.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import os.path as osp
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ADAPTER_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
DEFAULT_SPEC = osp.join(ADAPTER_ROOT, 'configs', 'seepersea_dataset_v1.json')
CLASS_ORDER = ('ship', 'buoy', 'other')
DATA_VERSION = 'seepersea_3class_v1'


def ensure_adapter_on_path() -> None:
    if ADAPTER_ROOT not in sys.path:
        sys.path.insert(0, ADAPTER_ROOT)


def abspath(path: str) -> str:
    return osp.abspath(osp.expanduser(path))


def load_spec(path: Optional[str] = None) -> dict:
    spec_path = abspath(path or DEFAULT_SPEC)
    with open(spec_path, 'r', encoding='utf-8') as f:
        spec = json.load(f)
    for key in ('version', 'clip', 'classes', 'categories', 'splits',
                'point_cloud_range', 'data_prefix'):
        if key not in spec:
            raise ValueError(f'{spec_path}: missing spec key {key}')
    if tuple(spec['classes']) != CLASS_ORDER:
        raise ValueError(f'{spec_path}: classes must be {CLASS_ORDER}')
    if spec['categories'] != {c: i for i, c in enumerate(CLASS_ORDER)}:
        raise ValueError(f'{spec_path}: category ids must be 0/1/2 in order')
    return spec


def frame_token(clip: str, frame_id: int) -> str:
    return f'{clip}:{frame_id:06d}'


def candidate_role(frame_id: int, spec: dict) -> str:
    for item in spec['splits']:
        if int(item['start']) <= frame_id <= int(item['end']):
            return str(item['role'])
    raise ValueError(f'frame {frame_id}: no candidate interval in spec')


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_pcd_ascii(path: str) -> Tuple[np.ndarray, dict]:
    """Read a strict ASCII x/y/z/intensity PCD; return float64 points+header."""
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
    header: Dict[str, List[str]] = {}
    data_idx = None
    for i, line in enumerate(lines):
        if line.startswith('DATA'):
            parts = line.split()
            if len(parts) != 2 or parts[1].lower() != 'ascii':
                raise ValueError(f'{path}: only ASCII PCD is supported')
            data_idx = i + 1
            break
        if line.startswith('#') or not line.strip():
            continue
        parts = line.split()
        header[parts[0]] = parts[1:]
    if data_idx is None:
        raise ValueError(f'{path}: DATA ascii header missing')
    if header.get('FIELDS', [])[:4] != ['x', 'y', 'z', 'intensity']:
        raise ValueError(f'{path}: FIELDS must start with x y z intensity')
    if header.get('TYPE', [])[:4] != ['F', 'F', 'F', 'F']:
        raise ValueError(f'{path}: x/y/z/intensity must be FLOAT32')
    rows = [line.split() for line in lines[data_idx:] if line.strip()]
    expected = int(header['POINTS'][0])
    width = int(header['WIDTH'][0])
    height = int(header.get('HEIGHT', ['1'])[0])
    if expected <= 0 or width * height != expected or len(rows) != expected:
        raise ValueError(
            f'{path}: POINTS={expected}, WIDTH*HEIGHT={width * height}, '
            f'rows={len(rows)} disagree')
    pts = np.asarray([row[:4] for row in rows], dtype=np.float64)
    if any(len(row) < 4 for row in rows) or not np.isfinite(pts).all():
        raise ValueError(f'{path}: non-finite or incomplete point row')
    return pts, {k: v for k, v in header.items()}


def read_calib(path: str) -> Dict[str, np.ndarray]:
    """Parse and validate KITTI calib. P2's 4th column must be exactly zero."""
    raw: Dict[str, np.ndarray] = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if ':' not in line:
                continue
            key, value = line.split(':', 1)
            raw[key.strip()] = np.asarray(
                [float(x) for x in value.split()], dtype=np.float64)
    for key, size in (('P2', 12), ('R0_rect', 9), ('Tr_velo_to_cam', 12)):
        if key not in raw or raw[key].shape != (size, ):
            raise ValueError(f'{path}: missing or malformed {key}')
        if not np.isfinite(raw[key]).all():
            raise ValueError(f'{path}: non-finite {key}')
    p2 = raw['P2'].reshape(3, 4)
    if np.abs(p2[:, 3]).max() > 1e-6:
        raise ValueError(f'{path}: unsupported projection form (P2 column 4)')
    r0 = raw['R0_rect'].reshape(3, 3)
    tr = raw['Tr_velo_to_cam'].reshape(3, 4)
    rot = r0 @ tr[:, :3]
    for name, mat in (('R0_rect', r0), ('Tr_velo_to_cam rotation', rot)):
        if abs(np.linalg.det(mat)) < 1e-6:
            raise ValueError(f'{path}: {name} is not invertible')
        if not np.allclose(mat.T @ mat, np.eye(3), atol=1e-3):
            raise ValueError(f'{path}: {name} is not a rigid rotation')
        if not np.isclose(np.linalg.det(mat), 1.0, atol=1e-3):
            raise ValueError(f'{path}: {name} has improper rotation determinant')
    rect4 = np.eye(4, dtype=np.float64)
    rect4[:3, :3] = r0
    tr4 = np.eye(4, dtype=np.float64)
    tr4[:3, :4] = tr
    return dict(cam2img=p2[:3, :3].copy(), lidar2cam=rect4 @ tr4)


def read_timestamps(path: str, spec: dict) -> Tuple[Dict[int, float], List[dict]]:
    """Read seconds; repair only frame 0's documented sentinel."""
    raw: Dict[int, float] = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',')
            if not parts or not parts[0]:
                continue
            fid = int(parts[0])
            if len(parts) < 2:
                raise ValueError(f'{path}: frame {fid} has no timestamp')
            raw[fid] = float(parts[1])
    sentinel_frame = int(spec['timestamp']['sentinel_frame'])
    sentinel = float(spec['timestamp']['sentinel_value'])
    fixes: List[dict] = []
    if sentinel_frame in raw and raw[sentinel_frame] == sentinel:
        t1, t2 = raw.get(sentinel_frame + 1), raw.get(sentinel_frame + 2)
        if t1 is None or t2 is None or not (np.isfinite(t1) and np.isfinite(t2)) \
                or t1 <= 0 or t2 <= 0 or t2 <= t1:
            raise ValueError(
                f'{path}: frame {sentinel_frame} sentinel cannot be repaired: '
                f't1={t1}, t2={t2}')
        repaired = float(t1 - (t2 - t1))
        fixes.append(dict(frame_id=sentinel_frame, original=sentinel,
                          repaired=repaired, rule='t0=t1-(t2-t1)'))
        raw[sentinel_frame] = repaired
    ordered = sorted(raw)
    previous: Optional[float] = None
    for fid in ordered:
        value = raw[fid]
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{path}: frame {fid} has invalid timestamp {value}')
        if previous is not None and value <= previous:
            raise ValueError(
                f'{path}: frame {fid} timestamp {value} not increasing '
                f'(previous {previous})')
        previous = value
    return raw, fixes


def read_annotations(path: str, spec: dict) -> Optional[List[dict]]:
    """Return native 3-class boxes, or None for an explicitly empty frame.

    A malformed/non-empty JSON file is an error and is never treated as empty.
    """
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    if not text:
        return None
    try:
        ann = json.loads(text)
    except (ValueError, TypeError) as exc:
        raise ValueError(f'{path}: invalid JSON: {exc}') from exc
    if not isinstance(ann, dict) or 'labels' not in ann:
        raise ValueError(f'{path}: missing labels field')
    if not isinstance(ann['labels'], list):
        raise ValueError(f'{path}: labels must be a list')
    if len(ann['labels']) == 0:
        return None
    boxes: List[dict] = []
    for index, label in enumerate(ann['labels']):
        try:
            category = label['category']
            if category not in spec['categories']:
                raise ValueError(f'unknown category {category!r}')
            box = label['box3d']
            loc, dim, ori = box['location'], box['dimension'], box['orientation']
            center = [float(loc[k]) for k in ('x', 'y', 'z')]
            # Annotation width is the along-heading length l; annotation
            # length is the lateral width w (see L6 audit section 3.2).
            length = float(dim['width'])
            width = float(dim['length'])
            height = float(dim['height'])
            yaw = float(ori['rotationYaw'])
            pitch = float(ori.get('rotationPitch', 0.0))
            roll = float(ori.get('rotationRoll', 0.0))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'{path}: label {index} malformed: {exc}') from exc
        values = center + [length, width, height, yaw]
        if not np.isfinite(values).all():
            raise ValueError(f'{path}: label {index} has non-finite geometry')
        if min(length, width, height) <= 0.0:
            raise ValueError(f'{path}: label {index} has non-positive dimension')
        if abs(pitch) > 1e-6 or abs(roll) > 1e-6:
            raise ValueError(f'{path}: label {index} has non-zero pitch/roll')
        boxes.append(dict(
            category=category,
            label_id=int(spec['categories'][category]),
            raw_id=label.get('id'),
            label_index=index,
            bbox_3d=[float(v) for v in values]))
    return boxes


def points_in_box(points_xyz: np.ndarray, bbox_3d: Sequence[float]) -> int:
    """Count points in a rotated gravity-center box (float64, inclusive half-size)."""
    xyz = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    if xyz.size == 0:
        return 0
    x, y, z, length, width, height, yaw = [float(v) for v in bbox_3d]
    dx = xyz[:, 0] - x
    dy = xyz[:, 1] - y
    dz = xyz[:, 2] - z
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    local_x = cos_yaw * dx + sin_yaw * dy
    local_y = -sin_yaw * dx + cos_yaw * dy
    inside = ((np.abs(local_x) <= length / 2.0 + 1e-9)
              & (np.abs(local_y) <= width / 2.0 + 1e-9)
              & (np.abs(dz) <= height / 2.0 + 1e-9))
    return int(inside.sum())


def box_center_in_range(center: Sequence[float], pc_range: Sequence[float]) -> bool:
    return bool(pc_range[0] <= center[0] < pc_range[3]
                and pc_range[1] <= center[1] < pc_range[4]
                and pc_range[2] <= center[2] < pc_range[5])


def range_retention(points_xyz: np.ndarray, centers: np.ndarray,
                    pc_range: Sequence[float]) -> dict:
    lo, hi = np.asarray(pc_range[:3]), np.asarray(pc_range[3:])
    pts = np.asarray(points_xyz).reshape(-1, 3)
    ctrs = np.asarray(centers).reshape(-1, 3)
    keep_pts = np.ones(len(pts), dtype=bool)
    keep_ctr = np.ones(len(ctrs), dtype=bool)
    for axis in range(3):
        keep_pts &= (pts[:, axis] >= lo[axis]) & (pts[:, axis] < hi[axis])
        keep_ctr &= (ctrs[:, axis] >= lo[axis]) & (ctrs[:, axis] < hi[axis])
    total_pts, total_ctr = len(pts), len(ctrs)
    return dict(points_total=total_pts, points_kept=int(keep_pts.sum()),
                points_kept_ratio=float(keep_pts.sum() / total_pts)
                if total_pts else 0.0,
                box_centers_total=total_ctr,
                box_centers_kept=int(keep_ctr.sum()),
                box_centers_kept_ratio=float(keep_ctr.sum() / total_ctr)
                if total_ctr else 0.0)


def atomic_write_json(path: str, payload: dict) -> None:
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write('\n')
    os.replace(tmp, path)


def write_pickle(path: str, payload: dict) -> None:
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(payload, f, protocol=4)


def setup_source_root(mmdet3d_root: str, import_ext: bool = True) -> dict:
    """Put the explicit SOURCE_ROOT first and verify the imported package."""
    root = abspath(mmdet3d_root)
    if not osp.isdir(root) or not osp.isfile(
            osp.join(root, 'mmdet3d', '__init__.py')):
        raise RuntimeError(f'{root}: not an MMDetection3D source root')
    ensure_adapter_on_path()
    if root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)
    import mmdet3d
    from mmdet3d.utils import register_all_modules
    imported = osp.abspath(mmdet3d.__file__)
    if osp.commonpath([root, imported]) != root:
        raise RuntimeError(
            f'mmdet3d imported from {imported}, expected under {root}')
    register_all_modules()
    install_loader_patch()
    for module in ('projects.BEVFusion.bevfusion',
                   'projects.BEVFusion.bevfusion.ops'):
        try:
            __import__(module)
        except Exception as exc:
            raise RuntimeError(
                f'cannot import {module}; compile BEVFusion CUDA ops first: '
                f'{exc}') from exc
    try:
        import spconv
    except ModuleNotFoundError as exc:
        raise RuntimeError('spconv 2.x is required for controlled weight transfer'
                           ) from exc
    from mmdet3d.models.layers.spconv import IS_SPCONV2_AVAILABLE
    if not IS_SPCONV2_AVAILABLE:
        raise RuntimeError('spconv 2.x is required; found incompatible spconv')
    if import_ext:
        import usv_fusion_ext  # noqa: F401
    return dict(mmdet3d_root=root, mmdet3d_file=imported,
                mmdet3d_version=getattr(mmdet3d, '__version__', 'unknown'),
                spconv_version=getattr(spconv, '__version__', 'unknown'))


def explicit_load_state_dict(module, state_dict, strict=False, logger=None):
    """Load by name using explicit tensor copies (spconv-safe).

    spconv 2.3.8 with torch 2.0 reports shape mismatches from its own
    state_dict when ``_load_from_state_dict`` is used. Copying by registered
    parameter/buffer name keeps the strict key contract without that defect.
    """
    import torch
    from mmengine.logging import print_log

    model_keys = set(module.state_dict())
    source_keys = set(state_dict)
    missing = sorted(model_keys - source_keys)
    unexpected = sorted(source_keys - model_keys)
    used = set()
    for name, param in module.named_parameters():
        if name in source_keys:
            if tuple(param.shape) != tuple(state_dict[name].shape):
                raise RuntimeError(f'load shape mismatch: {name}')
            with torch.no_grad():
                param.copy_(state_dict[name].to(param.device))
            used.add(name)
    for name, buffer in module.named_buffers():
        if name in source_keys:
            if tuple(buffer.shape) != tuple(state_dict[name].shape):
                raise RuntimeError(f'load shape mismatch: {name}')
            with torch.no_grad():
                buffer.copy_(state_dict[name].to(buffer.device))
            used.add(name)
    if used != model_keys:
        missing = sorted(model_keys - used)
    if strict and (missing or unexpected):
        raise RuntimeError(
            f'strict load failed: missing={missing[:8]}, '
            f'unexpected={unexpected[:8]}')
    if missing or unexpected:
        print_log(f'weights not matched: missing={len(missing)}, '
                  f'unexpected={len(unexpected)}', logger=logger, level=30)


def install_loader_patch() -> None:
    import mmengine.runner.checkpoint as ckpt
    ckpt.load_state_dict = explicit_load_state_dict


def register_monitor_hook() -> None:
    """Register a small hook that records losses, timing and peak VRAM."""
    from mmengine.hooks import Hook
    from mmengine.registry import HOOKS

    if 'SeaPerseaMonitorHook' in HOOKS:
        return

    @HOOKS.register_module()
    class SeaPerseaMonitorHook(Hook):
        def __init__(self, out_dir: str):
            self.out_dir = abspath(out_dir)
            self.records: List[dict] = []

        def before_run(self, runner) -> None:
            import time
            import torch
            self.start = time.perf_counter()
            self.last = self.start
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

        def after_train_iter(self, runner, batch_idx, data_batch=None,
                             outputs=None) -> None:
            import time
            import torch
            now = time.perf_counter()
            losses = {}
            for key, value in (outputs or {}).items():
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    losses[key] = float(value.detach().cpu())
            total = sum(v for k, v in losses.items() if k != 'matched_ious')
            self.records.append(dict(
                iter=runner.iter + 1,
                total_loss=total,
                losses=losses,
                iter_seconds=now - self.last,
                peak_memory_mb=(torch.cuda.max_memory_allocated() / 2**20
                                if torch.cuda.is_available() else 0.0)))
            self.last = now

        def after_run(self, runner) -> None:
            import time
            os.makedirs(self.out_dir, exist_ok=True)
            atomic_write_json(osp.join(self.out_dir, 'smoke_summary.json'),
                              dict(iterations=self.records,
                                   total_wall_seconds=time.perf_counter()
                                   - self.start))


UPSTREAM_CONFIG = 'projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
OVERLAY_CONFIG = 'configs/bevfusion_seepersea_3class.py'


def build_seepersea_config(mmdet3d_root: str, data_root: str, work_dir: str,
                           checkpoint: Optional[str] = None,
                           mode: str = 'train',
                           resume_from: Optional[str] = None,
                           max_iters: Optional[int] = None) -> tuple:
    """Build one resolved config used by every stage-4 entry point."""
    from mmengine.config import Config

    provenance = setup_source_root(mmdet3d_root)
    upstream = osp.join(provenance['mmdet3d_root'], UPSTREAM_CONFIG)
    overlay = osp.join(ADAPTER_ROOT, OVERLAY_CONFIG)
    base_cfg = Config.fromfile(upstream)
    override = Config.fromfile(overlay)
    base_cfg.merge_from_dict(override.to_dict())
    cfg = base_cfg
    for key in ('db_sampler', 'class_names', 'dataset_type'):
        cfg.pop(key, None)
    cfg.data_root = abspath(data_root)
    cfg.work_dir = abspath(work_dir)
    cfg.load_from = abspath(checkpoint) if checkpoint else None
    cfg.resume = False
    cfg.seed = 2026
    cfg.custom_imports = dict(
        imports=['projects.BEVFusion.bevfusion', 'usv_fusion_ext'],
        allow_failed_imports=False)

    dataset_cfg = cfg.train_dataloader.dataset
    dataset_cfg.data_root = cfg.data_root
    ann_name = 'infos_smoke.pkl' if mode == 'smoke' else 'infos_train.pkl'
    dataset_cfg.ann_file = osp.join(cfg.data_root, ann_name)
    if mode in ('smoke', 'resume') or max_iters is not None:
        limit = int(max_iters or (21 if mode == 'resume' else 20))
        cfg.train_cfg = dict(type='IterBasedTrainLoop', max_iters=limit,
                             val_interval=1000000)
        cfg.default_hooks.checkpoint = dict(
            type='CheckpointHook', interval=limit, by_epoch=False,
            save_last=True)
        cfg.default_hooks.logger = dict(type='LoggerHook', interval=1)
        cfg.log_processor = dict(window_size=1, by_epoch=False)
    if mode == 'resume':
        if not resume_from:
            raise ValueError('resume mode requires --resume-from')
        cfg.resume = True
        cfg.load_from = abspath(resume_from)

    register_monitor_hook()
    cfg.custom_hooks = [dict(type='SeaPerseaMonitorHook', out_dir=cfg.work_dir)]

    cfg.seepersea_provenance = dict(
        **provenance,
        upstream_config=upstream,
        upstream_sha256=sha256_file(upstream),
        overlay_config=overlay,
        overlay_sha256=sha256_file(overlay),
        data_root=cfg.data_root,
        work_dir=cfg.work_dir,
        load_from=cfg.load_from,
        mode=mode,
        adapter_root=ADAPTER_ROOT)
    if cfg.model.img_backbone.get('init_cfg', 'missing') is not None:
        raise RuntimeError('img_backbone.init_cfg must be None')
    os.makedirs(cfg.work_dir, exist_ok=True)
    cfg.dump(osp.join(cfg.work_dir, 'resolved_config.py'))
    atomic_write_json(
        osp.join(cfg.work_dir, 'config_provenance.json'),
        cfg.seepersea_provenance)
    return cfg, provenance
