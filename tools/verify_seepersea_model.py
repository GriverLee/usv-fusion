#!/usr/bin/env python3
"""Stage-4 model verification: config, pipeline, tensors, grads, reload, val.

Only the minimal checks required by the task spec are performed. No AP/NDS or
full fine-tuning is computed here.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import os.path as osp
import subprocess
import sys
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
from usv_fusion_ext.runtime import (  # noqa: E402
    atomic_write_json, build_seepersea_config, explicit_load_state_dict,
    setup_source_root)

CHECKS: Dict[str, dict] = {}
RUN_TOOL = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))),
                    'tools', 'run_seepersea.py')
EXPECTED_SHAPES = dict(
    pts_middle_encoder=[1, 256, 180, 180],
    view_transform=[1, 80, 180, 180],
    fusion_layer=[1, 256, 180, 180],
    pts_neck=[1, 512, 180, 180])


def record(name: str, status: str, detail: str) -> None:
    CHECKS[name] = dict(status=status, detail=str(detail))


def require(name: str, ok: bool, detail: str) -> None:
    record(name, 'PASS' if ok else 'FAIL', detail)
    if not ok:
        raise RuntimeError(f'{name}: {detail}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mmdet3d-root', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--checkpoint', required=True, help='iter_20.pth')
    parser.add_argument('--work-dir', required=True)
    return parser.parse_args()


def build_dataset(cfg, split: str):
    from mmdet3d.registry import DATASETS

    dataset_cfg = copy.deepcopy(cfg.train_dataloader.dataset)
    dataset_cfg.ann_file = osp.join(cfg.data_root, f'infos_{split}.pkl')
    is_test = split in ('val', 'test')
    dataset_cfg.test_mode = is_test
    dataset_cfg.pipeline = copy.deepcopy(
        cfg.test_pipeline if is_test else cfg.train_pipeline)
    return DATASETS.build(dataset_cfg)


def make_batch(model, packed: dict, training: bool = False) -> dict:
    batch = dict(
        inputs=dict(points=[packed['inputs']['points']],
                    img=[packed['inputs']['img']]),
        data_samples=[packed['data_samples']])
    return model.data_preprocessor(batch, training=training)


def check_pipeline(cfg, split: str, limit: int = None) -> dict:
    dataset = build_dataset(cfg, split)
    total = len(dataset) if limit is None else min(limit, len(dataset))
    stats = dict(split=split, frames=total, max_points=0, issues=[])
    for index in range(total):
        packed = dataset[index]
        points = packed['inputs']['points']
        img = packed['inputs']['img']
        meta = packed['data_samples'].metainfo
        if points.ndim != 2 or points.shape[1] != 5:
            stats['issues'].append(f'{index}: bad point shape {tuple(points.shape)}')
        if not torch.isfinite(points).all():
            stats['issues'].append(f'{index}: non-finite points')
        intensity = points[:, 3]
        if float(intensity.min()) < 0.0 or float(intensity.max()) > 255.0:
            stats['issues'].append(f'{index}: intensity out of [0,255]')
        if tuple(img.shape) != (1, 3, 256, 352):
            stats['issues'].append(f'{index}: img shape {tuple(img.shape)}')
        if meta.get('num_views') != 1:
            stats['issues'].append(f'{index}: num_views={meta.get("num_views")}')
        required_keys = ['cam2img', 'lidar2img', 'cam2lidar',
                         'img_aug_matrix']
        if split in ('train', 'smoke'):
            required_keys.append('lidar_aug_matrix')
        for key in required_keys:
            value = np.asarray(meta.get(key))
            if value.shape[-2:] != (4, 4) or value.ndim not in (2, 3):
                stats['issues'].append(f'{index}: bad {key} {value.shape}')
        ann = dataset.get_ann_info(index)
        boxes = np.asarray(ann['gt_bboxes_3d'].tensor)
        labels = np.asarray(ann['gt_labels_3d'])
        if boxes.ndim != 2 or boxes.shape[-1] != 7:
            stats['issues'].append(f'{index}: gt box shape {boxes.shape}')
        if len(labels) and (labels.min() < 0 or labels.max() > 2):
            stats['issues'].append(f'{index}: label outside 0..2')
        raw_path = packed['data_samples'].metainfo['lidar_path']
        if not osp.isfile(raw_path):
            stats['issues'].append(f'{index}: missing {raw_path}')
        stats['max_points'] = max(stats['max_points'], int(points.shape[0]))
    return stats


def predict(model, batch: dict):
    with torch.no_grad():
        return model.predict(batch['inputs'], batch['data_samples'])


def pred_arrays(outputs):
    pred = outputs[0].pred_instances_3d
    return (pred.bboxes_3d.tensor.detach().cpu(),
            pred.labels_3d.detach().cpu(),
            pred.scores_3d.detach().cpu(), pred.bboxes_3d)


def main() -> None:
    args = parse_args()
    os.makedirs(osp.abspath(args.work_dir), exist_ok=True)
    provenance = setup_source_root(args.mmdet3d_root)
    cfg, _ = build_seepersea_config(
        args.mmdet3d_root, args.data_root, args.work_dir,
        checkpoint=args.checkpoint, mode='train')
    from mmdet3d.registry import DATASETS, MODELS

    resolved = osp.join(args.work_dir, 'resolved_config.py')
    text = open(resolved, encoding='utf-8').read()
    forbidden = ['CBGS', 'NuScenesMetric', 'LoadPointsFromMultiSweeps',
                 'ObjectSample', 'GridMask', 'BEVFusionRandomFlip3D',
                 '-54.0', 'vel=[2, 2]']
    found = [token for token in forbidden if token in text]
    head = cfg.model.bbox_head
    require('config_no_inherited_artifacts', not found,
            f'forbidden tokens={found}')
    require('config_native_three_classes',
            tuple(cfg.train_dataloader.dataset.metainfo['classes']) ==
            ('ship', 'buoy', 'other')
            and head.num_classes == 3
            and set(head.common_heads) == {'center', 'height', 'dim', 'rot'}
            and head.bbox_coder.code_size == 8,
            f'classes={cfg.train_dataloader.dataset.metainfo["classes"]}, '
            f'heads={list(head.common_heads)}')
    require('config_geometry',
            list(cfg.point_cloud_range) ==
            [-108.0, -108.0, -20.0, 108.0, 108.0, 20.0]
            and list(cfg.model.pts_middle_encoder.sparse_shape) ==
            [1440, 1440, 41]
            and list(cfg.model.view_transform.dbound) == [1.0, 60.0, 0.5]
            and head.train_cfg.grid_size == [1440, 1440, 40],
            'geometry fields mismatch')
    resolve_paths = all(
        osp.isfile(osp.join(cfg.data_root, f'infos_{name}.pkl'))
        for name in ('train', 'val', 'test', 'smoke')) and all(
            osp.isdir(osp.join(cfg.data_root, prefix))
            for prefix in cfg.train_dataloader.dataset.data_prefix.values())
    require('config_paths_resolve', resolve_paths, f'data_root={cfg.data_root}')
    require('mmdet3d_from_explicit_root',
            osp.commonpath([osp.abspath(args.mmdet3d_root),
                            osp.abspath(provenance['mmdet3d_file'])]) ==
            osp.abspath(args.mmdet3d_root),
            str(provenance))

    pipeline_stats = {}
    for split in ('smoke', 'train', 'val', 'test'):
        pipeline_stats[split] = check_pipeline(cfg, split)
        require(f'pipeline_{split}', not pipeline_stats[split]['issues'],
                f'frames={pipeline_stats[split]["frames"]}, '
                f'issues={pipeline_stats[split]["issues"][:5]}')

    model = MODELS.build(copy.deepcopy(cfg.model)).cuda().eval()
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    explicit_load_state_dict(model, ckpt['state_dict'], strict=True)
    require('checkpoint_strict_reload', True,
            f'{args.checkpoint}: keys={len(ckpt["state_dict"])}')

    dataset = build_dataset(cfg, 'smoke')
    packed = dataset[0]
    batch = make_batch(model, packed, training=False)
    captures: Dict[str, object] = {}
    for name in EXPECTED_SHAPES:
        module = dict(model.named_modules())[name]
        module.register_forward_hook(
            lambda mod, inp, out, key=name: captures.__setitem__(key, out))
    outputs = predict(model, batch)
    shapes = {}
    for name, expected in EXPECTED_SHAPES.items():
        value = captures[name]
        shape = ([tuple(x.shape) for x in value]
                 if isinstance(value, (list, tuple)) else [tuple(value.shape)])
        shapes[name] = shape
        require(f'forward_shape_{name}', shape == [tuple(expected)],
                f'got={shape}, expected={[tuple(expected)]}')
    vt = model.view_transform
    geom = dict(nx=vt.nx.detach().cpu().tolist(),
                frustum=list(vt.frustum.shape))
    require('view_transform_geometry',
            geom == dict(nx=[360, 360, 1], frustum=[118, 32, 44, 3]),
            str(geom))
    boxes, labels, scores, box_obj = pred_arrays(outputs)
    require('forward_outputs',
            boxes.ndim == 2 and boxes.shape[1] == 7
            and labels.min().item() >= 0 and labels.max().item() <= 2
            and torch.isfinite(boxes).all() and torch.isfinite(scores).all(),
            f'boxes={tuple(boxes.shape)}, labels='
            f'{torch.unique(labels).tolist()}, scores finite='
            f'{bool(torch.isfinite(scores).all())}')

    model.train()
    train_batch = make_batch(model, build_dataset(cfg, 'train')[0],
                             training=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    losses = model(train_batch['inputs'], train_batch['data_samples'],
                   mode='loss')
    total = sum(v for k, v in losses.items()
                if k != 'matched_ious' and isinstance(v, torch.Tensor))
    total.backward()
    require('backward_loss_finite',
            bool(torch.isfinite(total)) and total.item() > 0,
            f'total={total.item()}, matched_ious_excluded=True')
    branch_report = {}
    for branch, prefix in (('img', 'img_backbone'),
                           ('lidar', 'pts_middle_encoder'),
                           ('fusion', 'fusion_layer'),
                           ('head', 'bbox_head')):
        params = [(name, p) for name, p in model.named_parameters()
                  if name.startswith(prefix) and p.requires_grad]
        grads = [(name, p) for name, p in params if p.grad is not None]
        finite = all(bool(torch.isfinite(p.grad).all()) for _, p in grads)
        snapshot = [(name, p.detach().clone()) for name, p in grads]
        optimizer.step()
        changed = [name for (name, before), (n2, p) in
                   zip(snapshot, grads)
                   if not torch.equal(before, p.detach())]
        branch_report[branch] = dict(trainable=len(params),
                                     with_grad=len(grads),
                                     changed_after_step=len(changed),
                                     finite_grad=finite)
        require(f'grad_and_update_{branch}',
                bool(grads) and finite and bool(changed),
                str(branch_report[branch]))
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    explicit_load_state_dict(model, ckpt['state_dict'], strict=True)

    smoke_summary = osp.join(osp.dirname(osp.abspath(args.checkpoint)),
                             'smoke_summary.json')
    if osp.isfile(smoke_summary):
        summary = json.load(open(smoke_summary, encoding='utf-8'))
        iterations = summary['iterations']
        finite = all(math.isfinite(i['total_loss']) for i in iterations)
        peak = max(i['peak_memory_mb'] for i in iterations)
        require('iter20_smoke', len(iterations) == 20
                and iterations[-1]['iter'] == 20 and finite
                and osp.isfile(args.checkpoint),
                f'iterations={len(iterations)}, finite={finite}, '
                f'peak_memory_mb={peak:.1f}')
    else:
        record('iter20_smoke', 'NOT_RUN', f'missing {smoke_summary}')

    # Save/reload prediction comparison on the same processed input.
    reference = predict(model, batch)
    ref_boxes, ref_labels, ref_scores, _ = pred_arrays(reference)
    reloaded = MODELS.build(copy.deepcopy(cfg.model)).cuda().eval()
    explicit_load_state_dict(reloaded, ckpt['state_dict'], strict=True)
    re_pred = predict(reloaded, batch)
    boxes2, labels2, scores2, _ = pred_arrays(re_pred)
    same = (torch.equal(ref_labels, labels2)
            and torch.allclose(ref_scores, scores2, rtol=1e-4, atol=1e-5)
            and torch.allclose(ref_boxes, boxes2, rtol=1e-4, atol=1e-5))
    require('reload_predictions_match', same,
            f'boxes_close={torch.allclose(ref_boxes, boxes2, rtol=1e-4, atol=1e-5)}, '
            f'labels_equal={torch.equal(ref_labels, labels2)}')
    del reloaded
    torch.cuda.empty_cache()

    resume_dir = osp.join(osp.abspath(args.work_dir), 'resume_21')
    command = [sys.executable, RUN_TOOL,
               '--mmdet3d-root', args.mmdet3d_root,
               '--data-root', args.data_root,
               '--checkpoint', args.checkpoint,
               '--work-dir', resume_dir, '--mode', 'resume',
               '--resume-from', args.checkpoint, '--max-iters', '21']
    process = subprocess.run(command, capture_output=True, text=True)
    resume_ckpt = osp.join(resume_dir, 'iter_21.pth')
    resume_ok = process.returncode == 0 and osp.isfile(resume_ckpt)
    if resume_ok:
        state = torch.load(resume_ckpt, map_location='cpu')
        meta = state.get('meta', {})
        resume_ok = (meta.get('iter') == 21
                     and bool(state.get('optimizer', {}).get('state')))
    record('resume_to_iter21', 'PASS' if resume_ok else 'FAIL',
           f'exit={process.returncode}, checkpoint={resume_ok}, '
           f'log={osp.join(resume_dir, "resume.log")}')
    with open(osp.join(resume_dir, 'resume.log'), 'w',
              encoding='utf-8') as f:
        f.write(process.stdout + '\n' + process.stderr)

    val_dataset = build_dataset(cfg, 'val')
    pc_range = cfg.point_cloud_range
    raw_results, roi_results, val_issues = [], [], []
    for index in range(len(val_dataset)):
        packed = val_dataset[index]
        result = predict(model, make_batch(model, packed, training=False))
        boxes, labels, scores, box_obj = pred_arrays(result)
        token = packed['data_samples'].metainfo['token']
        labels_ok = len(labels) == 0 or (labels.min().item() >= 0
                                        and labels.max().item() <= 2)
        if not (torch.isfinite(boxes).all() and torch.isfinite(scores).all()
                and labels_ok and boxes.ndim == 2 and boxes.shape[1] == 7):
            val_issues.append(token)
        entry = dict(token=token, num_detections=int(boxes.shape[0]),
                     boxes=boxes.tolist(), labels=labels.tolist(),
                     scores=scores.tolist())
        raw_results.append(entry)
        centers = np.asarray(box_obj.gravity_center.detach().cpu())
        keep = np.ones(len(centers), dtype=bool)
        for axis in range(3):
            keep &= (centers[:, axis] >= pc_range[axis]) & \
                    (centers[:, axis] < pc_range[axis + 3])
        roi_results.append(dict(entry, boxes=[boxes[i].tolist()
                                              for i in np.where(keep)[0]],
                                labels=[labels[i].tolist()
                                        for i in np.where(keep)[0]],
                                scores=[scores[i].tolist()
                                        for i in np.where(keep)[0]],
                                num_detections=int(keep.sum())))
    atomic_write_json(osp.join(args.work_dir, 'val_predictions_raw.json'),
                      dict(results=raw_results, score_threshold=0.0,
                           note='no AP/NDS computed'))
    atomic_write_json(osp.join(args.work_dir, 'val_predictions_roi.json'),
                      dict(results=roi_results, score_threshold=0.0,
                           roi_range=pc_range,
                           roi_basis='gravity_center',
                           note='no AP/NDS computed'))
    require('val_inference_records',
            len(raw_results) == len(val_dataset) and not val_issues
            and len({r['token'] for r in raw_results}) == len(val_dataset),
            f'frames={len(val_dataset)}, records={len(raw_results)}, '
            f'invalid={val_issues[:5]}')

    summary = dict(
        status='PASS' if all(v['status'] == 'PASS' for v in CHECKS.values())
        else 'FAIL',
        mmdet3d=provenance,
        checkpoint=args.checkpoint,
        work_dir=args.work_dir,
        checks=CHECKS,
        tensor_shapes=shapes,
        view_transform=geom,
        backward=branch_report,
        pipeline_stats=pipeline_stats,
        val_frames=len(val_dataset),
        score_threshold=0.0,
        ap_nds='NOT_RUN: no evaluator in this task')
    atomic_write_json(osp.join(args.work_dir, 'model_verification.json'),
                      summary)
    print(f'[verify_seepersea_model] status={summary["status"]} '
          f'checks={len(CHECKS)}')
    if summary['status'] != 'PASS':
        print('failures:', [k for k, v in CHECKS.items()
                            if v['status'] == 'FAIL'])
        sys.exit(1)


if __name__ == '__main__':
    main()
