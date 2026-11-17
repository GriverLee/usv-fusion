#!/usr/bin/env python3
"""Controlled checkpoint migration for the native three-class model.

The source checkpoint is never loaded with ``strict=False`` into the target:
a complete target state_dict is produced explicitly, with the entire detection
head and all geometry-derived LSS parameters kept from the target model.
"""
from __future__ import annotations

import argparse
import os
import os.path as osp
import random
import sys

import numpy as np
import torch
from mmdet3d.registry import MODELS

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
from usv_fusion_ext.runtime import (  # noqa: E402
    atomic_write_json, build_seepersea_config, setup_source_root, sha256_file)

GEOMETRY_KEYS = {
    'view_transform.dx', 'view_transform.bx', 'view_transform.nx',
    'view_transform.frustum'}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mmdet3d-root', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--source-checkpoint', required=True)
    parser.add_argument('--out-dir', required=True)
    return parser.parse_args()


def seed_all(seed: int = 2026) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_target_model(cfg):
    import copy
    seed_all()
    return MODELS.build(copy.deepcopy(cfg.model))


def main() -> None:
    args = parse_args()
    provenance = setup_source_root(args.mmdet3d_root)
    source_path = osp.abspath(args.source_checkpoint)
    out_dir = osp.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    cfg, _ = build_seepersea_config(
        args.mmdet3d_root, args.data_root, out_dir, mode='base')

    model = build_target_model(cfg)
    target = {k: v.detach().cpu().clone()
              for k, v in model.state_dict().items()}
    source_ckpt = torch.load(source_path, map_location='cpu')
    if 'state_dict' not in source_ckpt:
        raise RuntimeError(f'{source_path}: no state_dict')
    source = dict(source_ckpt['state_dict'])
    module_keys = [k for k in source if k.startswith('module.')]
    if module_keys and len(module_keys) != len(source):
        raise RuntimeError('source has mixed module./no-module. prefixes')
    if module_keys:
        source = {k[len('module.'):]: v for k, v in source.items()}

    migrated: dict = {}
    kept_target: dict = {}
    for key, target_value in target.items():
        keep = key.startswith('bbox_head.') or key in GEOMETRY_KEYS
        if keep:
            kept_target[key] = target_value
            migrated[key] = target_value.clone()
            continue
        if key not in source:
            raise RuntimeError(f'target key {key} missing from source')
        if tuple(source[key].shape) != tuple(target_value.shape):
            raise RuntimeError(
                f'shape mismatch for {key}: source={tuple(source[key].shape)}, '
                f'target={tuple(target_value.shape)}')
        migrated[key] = source[key].detach().cpu().clone()
    allowed_external = lambda k: (k.startswith('bbox_head.')
                                  or k in GEOMETRY_KEYS)
    unexpected = sorted(k for k in source if k not in target
                        and not allowed_external(k))
    if unexpected:
        raise RuntimeError(f'source has {len(unexpected)} unexpected keys: '
                           f'{unexpected[:8]}')

    reloaded = build_target_model(cfg)
    # spconv 2.3.8 + torch 2.0 rejects even a model's own state_dict through
    # Module.load_state_dict (a shape-proxy bug). Perform the same strict key
    # check and copy explicitly, then verify tensor equality.
    model_keys = set(reloaded.state_dict())
    missing = sorted(model_keys - set(migrated))
    unexpected = sorted(set(migrated) - model_keys)
    if missing or unexpected:
        raise RuntimeError(f'strict reload failed: missing={missing[:8]}, '
                           f'unexpected={unexpected[:8]}')
    used = set()
    for name, param in reloaded.named_parameters():
        if name in migrated:
            if tuple(param.shape) != tuple(migrated[name].shape):
                raise RuntimeError(f'reload shape mismatch: {name}')
            with torch.no_grad():
                param.copy_(migrated[name].to(param.device))
            used.add(name)
    for name, buffer in reloaded.named_buffers():
        if name in migrated:
            if tuple(buffer.shape) != tuple(migrated[name].shape):
                raise RuntimeError(f'reload shape mismatch: {name}')
            with torch.no_grad():
                buffer.copy_(migrated[name].to(buffer.device))
            used.add(name)
    if used != model_keys:
        raise RuntimeError(f'not all keys loaded: {sorted(model_keys - used)[:8]}')
    result = type('StrictResult', (), dict(missing_keys=missing,
                                           unexpected_keys=unexpected))()
    for key, value in migrated.items():
        if key not in kept_target and not torch.equal(
                reloaded.state_dict()[key].cpu(), value):
            raise RuntimeError(f'reloaded tensor differs from source: {key}')

    vt = reloaded.view_transform
    expected = dict(
        dx=[0.6, 0.6, 40.0],
        bx=[-107.7, -107.7, 0.0],
        nx=[360, 360, 1],
        frustum_shape=[118, 32, 44, 3])
    actual = dict(
        dx=vt.dx.detach().cpu().tolist(),
        bx=vt.bx.detach().cpu().tolist(),
        nx=vt.nx.detach().cpu().tolist(),
        frustum_shape=list(vt.frustum.shape))
    geometry_ok = (actual['nx'] == expected['nx']
                   and actual['frustum_shape'] == expected['frustum_shape']
                   and np.allclose(actual['dx'], expected['dx'], atol=1e-6)
                   and np.allclose(actual['bx'], expected['bx'], atol=1e-5))
    if not geometry_ok:
        raise RuntimeError(f'geometry mismatch: got {actual}, expected {expected}')

    dataset_meta = dict(
        dataset='seepersea',
        version='seepersea_3class_v1',
        classes=['ship', 'buoy', 'other'],
        categories=dict(ship=0, buoy=1, other=2))
    destination = osp.join(out_dir, 'bevfusion_seepersea_3class_init.pth')
    torch.save(dict(
        state_dict=migrated,
        meta=dict(dataset_meta=dataset_meta,
                  provenance=dict(**provenance,
                                  source_checkpoint=source_path,
                                  source_sha256=sha256_file(source_path),
                                  strict_reload=True))),
        destination)
    atomic_write_json(osp.join(out_dir, 'checkpoint_transfer.json'), dict(
        source_checkpoint=source_path,
        source_sha256=sha256_file(source_path),
        target_checkpoint=destination,
        target_sha256=sha256_file(destination),
        source_keys=len(source),
        target_keys=len(target),
        migrated_keys=len(migrated) - len(kept_target),
        kept_target_keys=sorted(kept_target),
        kept_target_count=len(kept_target),
        module_prefix_removed=bool(module_keys),
        strict_reload=dict(missing=result.missing_keys,
                           unexpected=result.unexpected_keys,
                           method='explicit key/shape check + tensor copy '
                                  '(spconv 2.3.8/torch 2.0 load_state_dict '
                                  'shape-proxy defect)',
                           tensors_equal_to_source=True),
        geometry=actual,
        prohibited_source_fields=[
            k for k in ('optimizer', 'epoch', 'iter', 'scheduler', 'resume')
            if k in source_ckpt],
        dataset_meta=dataset_meta))
    print(f'[prepare_seepersea_checkpoint] {destination} '
          f'({len(migrated)} keys, {len(kept_target)} kept from target)')


if __name__ == '__main__':
    main()
