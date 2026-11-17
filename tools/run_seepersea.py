#!/usr/bin/env python3
"""Build one resolved config and run smoke/train/resume for SeaPersea."""
from __future__ import annotations

import argparse
import os.path as osp
import sys

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
from usv_fusion_ext.runtime import build_seepersea_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mmdet3d-root', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--checkpoint', required=True,
                        help='controlled 3-class init checkpoint')
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--mode', choices=['smoke', 'train', 'resume'],
                        default='smoke')
    parser.add_argument('--resume-from', default=None,
                        help='checkpoint to resume from in resume mode')
    parser.add_argument('--max-iters', type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from mmengine.runner import Runner

    cfg, provenance = build_seepersea_config(
        args.mmdet3d_root, args.data_root, args.work_dir,
        checkpoint=args.checkpoint, mode=args.mode,
        resume_from=args.resume_from, max_iters=args.max_iters)
    runner = Runner.from_cfg(cfg)
    runner.train()
    print(f'[run_seepersea] mode={args.mode}, work_dir={runner.work_dir}, '
          f'iter={runner.iter}, mmdet3d={provenance["mmdet3d_file"]}')


if __name__ == '__main__':
    main()
