#!/usr/bin/env python3
"""Build the formal seepersea_3class_v1 dataset.

Only annotations/*.json are used as box ground truth. All 438 sensor frames
are converted; formal pkl files contain only the configured labeled subsets.
"""
from __future__ import annotations

import argparse
import os
import os.path as osp
import shutil
import statistics
import sys
from collections import Counter, defaultdict
from typing import Dict, List

import numpy as np
from PIL import Image

sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
from usv_fusion_ext.runtime import (  # noqa: E402
    ADAPTER_ROOT, CLASS_ORDER, DATA_VERSION, atomic_write_json, box_center_in_range,
    candidate_role, frame_token, load_spec, points_in_box, range_retention,
    read_annotations, read_calib, read_pcd_ascii, read_timestamps, sha256_file,
    write_pickle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--src-root', required=True, help='raw clip root')
    parser.add_argument('--out-dir', required=True, help='fresh output DATA_ROOT')
    parser.add_argument('--spec', default=None, help='dataset spec JSON')
    return parser.parse_args()


def class_counts(infos: List[dict]) -> Dict[str, int]:
    names = [CLASS_ORDER[inst['bbox_label_3d']]
             for info in infos for inst in info['instances']]
    counter = Counter(names)
    return {name: int(counter.get(name, 0)) for name in CLASS_ORDER}


def check_nonempty(infos: List[dict], name: str) -> None:
    counts = class_counts(infos)
    if len(infos) == 0 or any(counts[c] == 0 for c in CLASS_ORDER):
        raise RuntimeError(
            f'{name}: invalid split: frames={len(infos)}, boxes={counts}; '
            'every formal split needs frames and >=1 box of every class')


def convert(args: argparse.Namespace) -> None:
    spec_path = osp.abspath(args.spec) if args.spec else osp.join(
        ADAPTER_ROOT, 'configs', 'seepersea_dataset_v1.json')
    spec = load_spec(spec_path)
    src = osp.abspath(args.src_root)
    out = osp.abspath(args.out_dir)
    if osp.exists(out):
        if not osp.isdir(out) or os.listdir(out):
            raise RuntimeError(
                f'{out} exists and is not empty; supply a fresh version path')
    else:
        os.makedirs(out)

    clip = spec['clip']
    first = int(spec.get('first_frame_id', 0))
    expected_ids = list(range(first, first + int(spec['expected_frame_count'])))
    layout = spec['raw_layout']
    paths_for = lambda frame, key: osp.join(src, layout[key].format(frame=frame))

    directories = ('pcd', 'image', 'calib', 'annotation')
    stem_sets = {}
    for key in directories:
        rel = layout[key].split('/')[0]
        abs_dir = osp.dirname(paths_for(expected_ids[0], key))
        files = os.listdir(abs_dir)
        stems = [osp.splitext(f)[0] for f in files if f.endswith(
            osp.splitext(layout[key])[1])]
        if len(stems) != len(set(stems)):
            raise RuntimeError(f'{abs_dir}: duplicate file stems')
        stem_sets[rel] = set(stems)
    expected_stems = {f'{fid:06d}' for fid in expected_ids}
    for rel, stems in stem_sets.items():
        if stems != expected_stems:
            missing = sorted(expected_stems - stems)[:10]
            extra = sorted(stems - expected_stems)[:10]
            raise RuntimeError(
                f'{src}/{rel}: frame set mismatch; missing={missing}, '
                f'extra={extra}')

    empty_stems = set()
    with open(paths_for(expected_ids[0], 'empty_list'), 'r',
              encoding='utf-8') as f:
        for line in f:
            token = line.strip().lstrip('./')
            if token:
                empty_stems.add(osp.splitext(osp.basename(token))[0])

    timestamps, timestamp_fixes = read_timestamps(
        paths_for(expected_ids[0], 'timestamp'), spec)
    if set(timestamps) != set(expected_ids):
        raise RuntimeError('timestamp frame set does not match sensor frames')

    pts_dir = osp.join(out, spec['data_prefix']['pts'], clip)
    img_dir = osp.join(out, spec['data_prefix']['CAM_FRONT'], clip)
    os.makedirs(pts_dir)
    os.makedirs(img_dir)
    split_dir = osp.join(out, 'splits')
    os.makedirs(split_dir)

    metainfo = dict(
        dataset='seepersea',
        version=DATA_VERSION,
        classes=list(CLASS_ORDER),
        categories={c: i for i, c in enumerate(CLASS_ORDER)},
        box_origin='gravity_center',
        box_dim_order='lwh',
        point_features=list(spec['pcd']['output_features']),
        has_global_pose=False,
        has_velocity=False)

    all_infos: List[dict] = []
    manifest_frames: List[dict] = []
    all_points: List[np.ndarray] = []
    all_centers: List[np.ndarray] = []
    point_counts: List[int] = []
    empty_set_from_ann = set()
    script_hashes = {
        osp.relpath(osp.join(ADAPTER_ROOT, path), ADAPTER_ROOT):
            sha256_file(osp.join(ADAPTER_ROOT, path))
        for path in ('tools/dataset_converters/build_seepersea_v1.py',
                     'usv_fusion_ext/runtime.py',
                     'configs/seepersea_dataset_v1.json')}

    for frame_id in expected_ids:
        stem = f'{frame_id:06d}'
        paths = {key: paths_for(frame_id, key) for key in
                 ('pcd', 'image', 'calib', 'annotation')}
        pts, _ = read_pcd_ascii(paths['pcd'])
        calib = read_calib(paths['calib'])
        boxes = read_annotations(paths['annotation'], spec)
        with Image.open(paths['image']) as image:
            if image.size != (spec['image']['width'], spec['image']['height']):
                raise ValueError(
                    f'frame {stem}: image size {image.size} != '
                    f'{(spec["image"]["width"], spec["image"]["height"])}')

        instances = []
        if boxes is None:
            empty_set_from_ann.add(stem)
        else:
            for box in boxes:
                count = points_in_box(pts[:, :3], box['bbox_3d'])
                if count < 1:
                    raise ValueError(
                        f'frame {stem}, label {box["label_index"]}: '
                        'no LiDAR points inside box')
                point_counts.append(count)
                instances.append(dict(
                    bbox_3d=box['bbox_3d'],
                    bbox_label_3d=box['label_id'],
                    num_lidar_pts=count))
            centers = np.asarray([b['bbox_3d'][:3] for b in boxes])
            all_centers.append(centers)
        all_points.append(pts[:, :3])
        centers = centers if boxes is not None else np.zeros((0, 3))

        pts5 = np.concatenate([
            pts.astype(np.float32),
            np.zeros((len(pts), 1), dtype=np.float32)], axis=1)
        pts5.tofile(osp.join(pts_dir, stem + '.bin'))
        shutil.copyfile(paths['image'], osp.join(img_dir, stem + '.png'))

        role = candidate_role(frame_id, spec)
        empty = boxes is None
        if empty:
            identity = 'unlabeled'
        elif role == 'temporal_gap':
            identity = 'temporal_gap'
        else:
            identity = role
        token = frame_token(clip, frame_id)
        info = dict(
            sample_idx=frame_id,
            token=token,
            scene_id=clip,
            frame_id=frame_id,
            timestamp=float(timestamps[frame_id]),
            lidar_points=dict(num_pts_feats=5,
                              lidar_path=f'{clip}/{stem}.bin'),
            images={'CAM_FRONT': dict(
                img_path=f'{clip}/{stem}.png',
                cam2img=calib['cam2img'].tolist(),
                lidar2cam=calib['lidar2cam'].tolist(),
                timestamp=float(timestamps[frame_id]))},
            instances=instances)
        all_infos.append(info)
        manifest_frames.append(dict(
            frame_id=frame_id,
            token=token,
            candidate_role=role,
            final_identity=identity,
            empty_annotation=empty,
            raw_timestamp=(float(spec['timestamp']['sentinel_value'])
                           if any(f['frame_id'] == frame_id
                                  for f in timestamp_fixes)
                           else float(timestamps[frame_id])),
            timestamp=float(timestamps[frame_id]),
            timestamp_fixed=any(f['frame_id'] == frame_id
                                for f in timestamp_fixes),
            num_points=int(len(pts)),
            num_boxes=len(instances),
            raw_file_sha256={key: sha256_file(paths[key]) for key in paths}))

    if empty_set_from_ann != empty_stems:
        raise RuntimeError(
            f'empty_label_files.txt mismatch: listed_only='
            f'{sorted(empty_stems - empty_set_from_ann)[:10]}, '
            f'annotation_only={sorted(empty_set_from_ann - empty_stems)[:10]}, '
            f'counts={len(empty_stems)} vs {len(empty_set_from_ann)}')

    expected = spec.get('expected_audit')
    if expected is not None:
        actual = dict(
            unlabeled_frames=len(empty_stems),
            labeled_all_frames=sum(not f['empty_annotation']
                                   for f in manifest_frames),
            labeled_all_boxes=sum(len(i['instances']) for i in all_infos))
        for key, value in actual.items():
            if key in expected and actual[key] != expected[key]:
                raise RuntimeError(
                    f'audit mismatch for {key}: got {actual[key]}, '
                    f'expected {expected[key]}')
        all_counts = class_counts(all_infos)
        for name, value in expected['labeled_all_by_class'].items():
            if all_counts[name] != value:
                raise RuntimeError(
                    f'audit mismatch for class {name}: got {all_counts[name]}, '
                    f'expected {value}')

    by_identity: Dict[str, List[dict]] = defaultdict(list)
    for info, manifest_frame in zip(all_infos, manifest_frames):
        by_identity[manifest_frame['final_identity']].append(info)
    for name in ('train', 'val', 'test'):
        check_nonempty(by_identity[name], name)

    smoke_count = int(spec['smoke']['count'])
    smoke_infos = sorted(by_identity[spec['smoke']['source']],
                         key=lambda i: i['frame_id'])[:smoke_count]
    check_nonempty(smoke_infos, 'smoke')

    labeled_all = [i for i, m in zip(all_infos, manifest_frames)
                   if not m['empty_annotation']]
    write_pickle(osp.join(out, 'infos_labeled_all.pkl'),
                 dict(metainfo=metainfo, data_list=labeled_all))
    for name in ('train', 'val', 'test'):
        write_pickle(osp.join(out, f'infos_{name}.pkl'),
                     dict(metainfo=metainfo, data_list=by_identity[name]))
    write_pickle(osp.join(out, 'infos_smoke.pkl'),
                 dict(metainfo=metainfo, data_list=smoke_infos))

    for name in ('train', 'val', 'test', 'smoke', 'unlabeled', 'temporal_gap'):
        infos = (smoke_infos if name == 'smoke' else by_identity[name])
        tokens = sorted(i['token'] for i in infos)
        with open(osp.join(split_dir, f'{name}.txt'), 'w',
                  encoding='utf-8') as f:
            for token in tokens:
                f.write(token + '\n')

    retention = range_retention(
        np.concatenate(all_points, axis=0),
        np.concatenate(all_centers, axis=0) if all_centers else np.zeros((0, 3)),
        spec['point_cloud_range'])
    distribution = dict(
        min=min(point_counts), max=max(point_counts),
        mean=float(statistics.mean(point_counts)),
        median=float(statistics.median(point_counts)),
        zero_boxes=sum(c == 0 for c in point_counts),
        one_box=sum(c == 1 for c in point_counts),
        at_least_two=sum(c >= 2 for c in point_counts),
        at_least_five=sum(c >= 5 for c in point_counts))
    counts = {}
    for name in ('train', 'val', 'test', 'smoke'):
        infos = smoke_infos if name == 'smoke' else by_identity[name]
        counts[name] = dict(
            frames=len(infos), boxes=sum(len(i['instances']) for i in infos),
            by_class=class_counts(infos),
            points=sum(m['num_points'] for m in manifest_frames
                       if m['token'] in {i['token'] for i in infos}))
    counts['labeled_all'] = dict(
        frames=sum(not m['empty_annotation'] for m in manifest_frames),
        boxes=sum(len(i['instances']) for i in all_infos),
        by_class=class_counts(all_infos))
    for name in ('unlabeled', 'temporal_gap'):
        counts[name] = dict(frames=len(by_identity[name]),
                            boxes=sum(len(i['instances']) for i in
                                      by_identity[name]))

    manifest = dict(
        version=DATA_VERSION,
        script_version=spec['script_version'],
        clip=clip,
        spec_path=spec_path,
        spec_sha256=sha256_file(spec_path),
        script_sha256=script_hashes,
        data_prefix=spec['data_prefix'],
        point_cloud_range=spec['point_cloud_range'],
        voxel_size=spec['voxel_size'],
        split_strategy=spec['split_strategy'],
        conversions=dict(point_bin_dtype='float32', point_bin_features=5,
                         intensity='raw', image_copy=True, symlink=False,
                         sweeps=False),
        frames=manifest_frames,
        timestamp_fixes=timestamp_fixes,
        timestamp_file_sha256=sha256_file(
            paths_for(expected_ids[0], 'timestamp')))
    atomic_write_json(osp.join(out, 'manifest.json'), manifest)
    atomic_write_json(osp.join(out, 'conversion_report.json'), dict(
        version=DATA_VERSION,
        clip=clip,
        counts=counts,
        exclusion_reasons=dict(
            unlabeled=len(empty_stems),
            temporal_gap=len(by_identity['temporal_gap']),
            train=len(by_identity['train']),
            val=len(by_identity['val']),
            test=len(by_identity['test'])),
        empty_label_files=dict(listed=len(empty_stems),
                               detected=len(empty_set_from_ann),
                               agree=True),
        range_retention=retention,
        points_per_box=distribution,
        timestamp_fixes=timestamp_fixes,
        note='Range check uses lower-inclusive/upper-exclusive bounds.'))
    print(f'[build_seepersea_v1] {out}')
    for name in ('train', 'val', 'test', 'smoke', 'unlabeled', 'temporal_gap',
                 'labeled_all'):
        print(f'  {name:12s}: {counts[name]}')


if __name__ == '__main__':
    convert(parse_args())
