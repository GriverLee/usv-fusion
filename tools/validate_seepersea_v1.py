#!/usr/bin/env python3
"""Validate the formal seepersea_3class_v1 data version.

Pure-data checks do not need CUDA. Dataset/pipeline checks use an explicitly
provided MMDetection3D source root and never modify SOURCE_ROOT.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import os.path as osp
import pickle
import sys
from collections import Counter
from typing import Dict, List

import numpy as np

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
from usv_fusion_ext.runtime import (  # noqa: E402
    CLASS_ORDER, DATA_VERSION, atomic_write_json, box_center_in_range,
    candidate_role, frame_token, load_spec, points_in_box, read_annotations,
    read_calib, read_pcd_ascii, read_timestamps)

CHECKS: Dict[str, dict] = {}


def record(name: str, ok: bool, detail: str) -> None:
    CHECKS[name] = dict(status='PASS' if ok else 'FAIL', detail=str(detail))


def require(name: str, condition: bool, detail: str) -> None:
    record(name, condition, detail)
    if not condition:
        raise RuntimeError(f'{name}: {detail}')


def load_pkl(path: str) -> dict:
    with open(path, 'rb') as f:
        return pickle.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mmdet3d-root', required=True)
    parser.add_argument('--src-root', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--spec', default=None)
    parser.add_argument('--reference-data-root', default=None,
                        help='second independent conversion for determinism')
    return parser.parse_args()


def check_dataset(data_root: str, spec: dict, split: str) -> dict:
    from usv_fusion_ext.dataset import SeaPerseaDataset

    ann_file = f'infos_{split}.pkl'
    dataset = SeaPerseaDataset(
        data_root=data_root,
        ann_file=ann_file,
        data_prefix=spec['data_prefix'],
        pipeline=[],
        modality=dict(use_lidar=True, use_camera=True),
        box_type_3d='LiDAR',
        default_cam_key=None,
        filter_empty_gt=False,
        test_mode=False)
    labels, dims, centers, bad_paths = [], [], [], []
    for index in range(len(dataset)):
        info = dataset.get_data_info(index)
        ann = dataset.get_ann_info(index)
        labels.extend(ann['gt_labels_3d'].tolist())
        dims.extend(list(ann['gt_bboxes_3d'].tensor.shape[-1:]))
        centers.extend(ann['gt_bboxes_3d'].gravity_center.tolist())
        for key, rel in (('pts', info['lidar_points']['lidar_path']),
                         ('CAM_FRONT', info['images']['CAM_FRONT']['img_path'])):
            prefix_abs = osp.join(osp.abspath(data_root),
                                  spec['data_prefix'][key])
            if not rel.startswith(prefix_abs + os.sep) or not osp.isfile(rel):
                bad_paths.append(rel)
    return dict(frames=len(dataset), labels=sorted(set(labels)),
                box_dims=sorted(set(dims)), gravity_centers=centers,
                bad_paths=bad_paths[:10], metainfo=dict(dataset.metainfo))


def main() -> None:
    args = parse_args()
    os.makedirs(osp.abspath(args.out_dir), exist_ok=True)
    spec = load_spec(args.spec)
    src, data_root = osp.abspath(args.src_root), osp.abspath(args.data_root)
    mmdet3d_root = osp.abspath(args.mmdet3d_root)
    sys.path.insert(0, mmdet3d_root)
    try:
        import mmdet3d
    except Exception as exc:
        require('mmdet3d_import', False, f'cannot import from {mmdet3d_root}: {exc}')
        return
    require('mmdet3d_root',
            osp.commonpath([mmdet3d_root, osp.abspath(mmdet3d.__file__)])
            == mmdet3d_root,
            f'mmdet3d.__file__={mmdet3d.__file__}, root={mmdet3d_root}')

    first = int(spec.get('first_frame_id', 0))
    expected_ids = list(range(first, first + int(spec['expected_frame_count'])))
    layout = spec['raw_layout']
    manifests = load_pkl(osp.join(data_root, 'infos_train.pkl'))
    manifest = json.load(open(osp.join(data_root, 'manifest.json'),
                              encoding='utf-8'))
    report = json.load(open(osp.join(data_root, 'conversion_report.json'),
                            encoding='utf-8'))

    meta_ok = (manifests['metainfo']['version'] == DATA_VERSION
               and tuple(manifests['metainfo']['classes']) == CLASS_ORDER
               and manifests['metainfo']['categories'] ==
               {c: i for i, c in enumerate(CLASS_ORDER)}
               and manifests['metainfo']['box_origin'] == 'gravity_center'
               and manifests['metainfo']['box_dim_order'] == 'lwh'
               and manifests['metainfo']['has_velocity'] is False)
    require('metainfo_contract', meta_ok,
            f'metainfo={manifests["metainfo"]}')

    tokens = {}
    for split in ('train', 'val', 'test', 'smoke', 'unlabeled', 'temporal_gap'):
        path = osp.join(data_root, 'splits', f'{split}.txt')
        lines = [line.strip() for line in open(path, encoding='utf-8')
                 if line.strip()]
        tokens[split] = lines
        require(f'split_sorted_{split}', lines == sorted(lines),
                f'{split}: not lexicographically sorted')
    train, val, test = set(tokens['train']), set(tokens['val']), set(tokens['test'])
    unlabeled, gap = set(tokens['unlabeled']), set(tokens['temporal_gap'])
    pairs = [('train', train, 'val', val), ('train', train, 'test', test),
             ('val', val, 'test', test), ('train', train, 'unlabeled', unlabeled),
             ('val', val, 'unlabeled', unlabeled), ('test', test, 'unlabeled', unlabeled),
             ('train', train, 'gap', gap), ('val', val, 'gap', gap),
             ('test', test, 'gap', gap), ('unlabeled', unlabeled, 'gap', gap)]
    require('splits_disjoint',
            all(not (a & b) for _, a, _, b in pairs),
            '; '.join(f'{n1}&{n2}={len(a & b)}' for n1, a, n2, b in pairs))
    all_tokens = train | val | test | unlabeled | gap
    require('splits_cover_all_frames',
            len(all_tokens) == len(spec['expected_frame_count'] * [0]) + 0
            and len(all_tokens) == len(expected_ids)
            and all_tokens == {frame_token(spec['clip'], fid)
                               for fid in expected_ids},
            f'unique tokens={len(all_tokens)}, expected={len(expected_ids)}')
    require('smoke_is_train_prefix',
            tokens['smoke'] == sorted(
                frame_token(spec['clip'], fid)
                for fid in sorted(int(t.split(':')[1])
                                  for t in tokens['train']))[:8],
            f'smoke={tokens["smoke"][:2]}...')

    labeled = load_pkl(osp.join(data_root, 'infos_labeled_all.pkl'))
    expected_audit = spec['expected_audit']
    counts = Counter(inst['bbox_label_3d'] for info in labeled['data_list']
                     for inst in info['instances'])
    require('labeled_all_counts',
            len(labeled['data_list']) == expected_audit['labeled_all_frames']
            and sum(counts.values()) == expected_audit['labeled_all_boxes']
            and all(counts[i] == expected_audit['labeled_all_by_class'][c]
                    for i, c in enumerate(CLASS_ORDER)),
            f'frames={len(labeled["data_list"])}, boxes={dict(counts)}')

    empty_detected, mismatches, point_errors, center_errors = set(), [], [], []
    raw_sha_mismatch = []
    for frame_id in expected_ids:
        stem = f'{frame_id:06d}'
        paths = {key: osp.join(src, layout[key].format(frame=frame_id))
                 for key in ('pcd', 'image', 'calib', 'annotation')}
        pts, _ = read_pcd_ascii(paths['pcd'])
        calib = read_calib(paths['calib'])
        boxes = read_annotations(paths['annotation'], spec)
        if boxes is None:
            empty_detected.add(stem)
        frame = next((f for f in manifest['frames'] if f['frame_id'] == frame_id),
                     None)
        if frame is None:
            mismatches.append(f'{stem}: missing in manifest')
            continue
        for key, path in paths.items():
            digest = hashlib.sha256(open(path, 'rb').read()).hexdigest()
            if frame['raw_file_sha256'][key] != digest:
                raw_sha_mismatch.append(f'{stem}:{key}')

        formal_names = [name for name, ids in (('train', train), ('val', val),
                                               ('test', test))
                        if frame_token(spec['clip'], frame_id) in ids]
        if not formal_names:
            continue
        pkl_name = formal_names[0]
        pkl = load_pkl(osp.join(data_root, f'infos_{pkl_name}.pkl'))
        info = next((i for i in pkl['data_list'] if i['frame_id'] == frame_id),
                    None)
        if info is None:
            mismatches.append(f'{stem}: missing in {pkl_name} pkl')
            continue
        if boxes is None:
            mismatches.append(f'{stem}: unlabeled frame in formal pkl')
            continue
        bin_path = osp.join(data_root, spec['data_prefix']['pts'],
                            info['lidar_points']['lidar_path'])
        disk = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 5)
        expected_bin = np.concatenate([
            pts.astype(np.float32),
            np.zeros((len(pts), 1), dtype=np.float32)], axis=1)
        if not np.array_equal(disk, expected_bin):
            mismatches.append(f'{stem}: bin mismatch')
        cam = info['images']['CAM_FRONT']
        if not np.allclose(cam['cam2img'], calib['cam2img'], atol=1e-9) or \
                not np.allclose(cam['lidar2cam'], calib['lidar2cam'], atol=1e-9):
            mismatches.append(f'{stem}: calibration mismatch')
        if len(info['instances']) != len(boxes):
            mismatches.append(f'{stem}: box count mismatch')
            continue
        for inst, raw in zip(info['instances'], boxes):
            if not np.allclose(inst['bbox_3d'], raw['bbox_3d'], atol=1e-5):
                mismatches.append(f'{stem}: box geometry mismatch')
            if inst['bbox_label_3d'] != raw['label_id']:
                mismatches.append(f'{stem}: native label mismatch')
            actual = points_in_box(pts[:, :3], raw['bbox_3d'])
            if inst['num_lidar_pts'] != actual or actual < 1:
                point_errors.append(f'{stem}: got {inst["num_lidar_pts"]}, '
                                    f'expected {actual}')
            if not box_center_in_range(raw['bbox_3d'][:3],
                                       spec['point_cloud_range']):
                center_errors.append(f'{stem}: GT center outside range')

    require('per_frame_roundtrip', not mismatches, '; '.join(mismatches[:8]))
    require('raw_sha256_match', not raw_sha_mismatch,
            '; '.join(raw_sha_mismatch[:8]))
    require('num_lidar_pts_exact', not point_errors, '; '.join(point_errors[:8]))
    require('formal_gt_centers_in_range', not center_errors,
            '; '.join(center_errors[:8]))
    empty_tokens = {frame_token(spec['clip'], int(stem))
                    for stem in empty_detected}
    require('empty_annotation_set', len(empty_detected) == 66
            and empty_tokens == unlabeled,
            f'detected={len(empty_detected)}, listed={len(unlabeled)}, '
            f'diff={sorted(empty_tokens ^ unlabeled)[:6]}')

    dataset_checks = {}
    for split in ('train', 'val', 'test', 'smoke'):
        dataset_checks[split] = check_dataset(data_root, spec, split)
        result = dataset_checks[split]
        require(f'dataset_{split}',
                set(result['labels']).issubset({0, 1, 2})
                and result['box_dims'] == [7]
                and not result['bad_paths'],
                f'labels={result["labels"]}, dims={result["box_dims"]}, '
                f'bad_paths={result["bad_paths"]}')

    relocated = osp.join(osp.abspath(args.out_dir), 'relocated_data')
    if not osp.lexists(relocated):
        os.symlink(data_root, relocated)
    check_dataset(relocated, spec, 'smoke')
    record('relocated_data_root', True,
           f'dataset resolved through {relocated}')

    if args.reference_data_root:
        reference = osp.abspath(args.reference_data_root)
        same = True
        details = []
        for name in ('train', 'val', 'test', 'smoke', 'labeled_all'):
            left = load_pkl(osp.join(data_root, f'infos_{name}.pkl'))
            right = load_pkl(osp.join(reference, f'infos_{name}.pkl'))
            fields = ('token', 'sample_idx', 'lidar_points', 'images',
                      'instances')
            if [tuple(i[f] if f != 'images' else tuple(i[f].keys())
                      for f in fields) for i in left['data_list']] != \
               [tuple(i[f] if f != 'images' else tuple(i[f].keys())
                      for f in fields) for i in right['data_list']]:
                same = False
                details.append(name)
        for stem in [f'{i:06d}' for i in expected_ids[:8]]:
            bin_rel = osp.join(data_root, 'samples/LIDAR_TOP', spec['clip'],
                               stem + '.bin')
            ref_bin = osp.join(reference, 'samples/LIDAR_TOP', spec['clip'],
                               stem + '.bin')
            if open(bin_rel, 'rb').read() != open(ref_bin, 'rb').read():
                same = False
                details.append(stem)
        require('two_conversions_identical', same,
                f'mismatches={details[:8]}')
    else:
        record('two_conversions_identical', False,
               'NOT_RUN: --reference-data-root not provided')

    record('conversion_report_present',
           all(k in report for k in ('counts', 'range_retention',
                                     'points_per_box', 'exclusion_reasons')),
           f'splits={list(report.get("counts", {}))}')

    failures = [name for name, value in CHECKS.items()
                if value['status'] == 'FAIL']
    summary = dict(version=DATA_VERSION, data_root=data_root,
                   src_root=src, mmdet3d_root=mmdet3d_root,
                   checks=CHECKS, datasets=dataset_checks,
                   failed=failures,
                   status='PASS' if not failures else 'FAIL')
    atomic_write_json(osp.join(osp.abspath(args.out_dir),
                               'validation_report.json'), summary)
    print(f'[validate_seepersea_v1] status={summary["status"]} '
          f'checks={len(CHECKS)} failures={len(failures)}')
    if failures:
        print('failed:', failures)
        sys.exit(1)


if __name__ == '__main__':
    main()
