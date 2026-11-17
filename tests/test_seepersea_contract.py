"""Synthetic contract tests for the seepersea_3class_v1 adapter."""
from __future__ import annotations

import json
import math
import os
import os.path as osp
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, ROOT)
from usv_fusion_ext.runtime import (  # noqa: E402
    candidate_role, load_spec, points_in_box, read_annotations, read_calib,
    read_timestamps)

CONVERTER = osp.join(ROOT, 'tools', 'dataset_converters',
                     'build_seepersea_v1.py')


def write_pcd(path: str, points: np.ndarray) -> None:
    header = (f'# .PCD v0.7\nVERSION 0.7\nFIELDS x y z intensity\n'
              f'SIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n'
              f'WIDTH {len(points)}\nHEIGHT 1\nPOINTS {len(points)}\n'
              f'DATA ascii\n')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(header)
        for x, y, z, intensity in points:
            f.write(f'{x:.6f} {y:.6f} {z:.6f} {intensity:.6f}\n')


def write_calib(path: str, p2_col3: float = 0.0, sheared: bool = False) -> None:
    p2 = [600, 0, 8, p2_col3, 0, 600, 6, 0, 0, 0, 1, 0]
    r0 = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    rows = [[1, 0.1, 0], [0, 1, 0], [0, 0, 1]] if sheared else [
        [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    tr = [v for row, t in zip(rows, (0, 0, 0)) for v in row + [t]]
    with open(path, 'w', encoding='utf-8') as f:
        f.write('P2: ' + ' '.join(map(str, p2)) + '\n')
        f.write('R0_rect: ' + ' '.join(map(str, r0)) + '\n')
        f.write('Tr_velo_to_cam: ' + ' '.join(map(str, tr)) + '\n')


def box(category: str, x: float, y: float, z: float,
        length: float, width: float, height: float, yaw: float = 0.0) -> dict:
    return dict(id=0, category=category, box3d=dict(
        location=dict(x=x, y=y, z=z),
        dimension=dict(width=length, length=width, height=height),
        orientation=dict(rotationYaw=yaw, rotationPitch=0, rotationRoll=0)))


class SyntheticFixture:
    def __init__(self, root: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.clip = 'synthetic-test'
        self.spec = json.loads(json.dumps(load_spec()))
        self.spec.update(clip=self.clip, expected_frame_count=6,
                         expected_audit=None,
                         image=dict(width=16, height=12),
                         splits=[
                             dict(role='train', start=0, end=1),
                             dict(role='temporal_gap', start=2, end=2),
                             dict(role='val', start=3, end=3),
                             dict(role='temporal_gap', start=4, end=4),
                             dict(role='test', start=5, end=5)],
                         smoke=dict(source='train', count=1))
        self.spec_path = self.root / 'spec.json'
        self.spec_path.write_text(json.dumps(self.spec), encoding='utf-8')
        for sub in ('point_clouds', 'calib', 'annotations',
                    'undist_images/CAM_FRONT', 'timestamp'):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        for fid in range(6):
            stem = f'{fid:06d}'
            write_pcd(str(self.root / 'point_clouds' / f'{stem}.pcd'),
                      np.asarray([[0, 0, 0, 300.0], [0.1, 0.1, 0.1, 2.0],
                                  [80, 0, 0, 1.0]], dtype=np.float64))
            Image.new('RGB', (16, 12)).save(
                self.root / 'undist_images' / 'CAM_FRONT' / f'{stem}.png')
            write_calib(str(self.root / 'calib' / f'{stem}.txt'))
            if fid == 1:
                (self.root / 'annotations' / f'{stem}.json').write_text(
                    '', encoding='utf-8')
            else:
                labels = [box('ship', 0, 0, 0, 2, 2, 1),
                          box('buoy', 0.1, 0, 0, 1, 1, 1),
                          box('other', 0, 0.1, 0, 2, 2, 1)]
                (self.root / 'annotations' / f'{stem}.json').write_text(
                    json.dumps(dict(name=stem, labels=labels)), encoding='utf-8')
        ts = ['0,-99999999', '1,100.0', '2,101.0', '3,102.0',
              '4,103.0', '5,104.0']
        (self.root / 'timestamp' / 'timestamp.txt').write_text(
            '\n'.join(ts) + '\n', encoding='utf-8')
        (self.root / 'empty_label_files.txt').write_text(
            '000001.json\n000001.txt\n', encoding='utf-8')

    @property
    def src(self) -> str:
        return str(self.root)

    def convert(self, out: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, CONVERTER, '--src-root', self.src,
             '--out-dir', out, '--spec', str(self.spec_path)],
            capture_output=True, text=True)


class ContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.fixture = SyntheticFixture(osp.join(cls.tmp.name, 'raw'))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_rotated_nonsquare_point_count(self):
        pts = np.asarray([[0.4, 1.9, 0.0], [1.9, 0.4, 0.0]], dtype=np.float64)
        yaw = math.pi / 2
        self.assertEqual(points_in_box(pts, [0, 0, 0, 4, 1, 1, yaw]), 1)
        self.assertEqual(points_in_box(pts, [0, 0, 0, 1, 4, 1, yaw]), 1)
        self.assertEqual(points_in_box(
            np.asarray([[3.9, 0.4, 0], [0.4, 3.9, 0]]),
            [0, 0, 0, 4, 1, 1, yaw]), 0)

    def test_gravity_to_bottom_center(self):
        try:
            from mmdet3d.structures import LiDARInstance3DBoxes
        except ImportError:
            self.skipTest('mmdet3d unavailable')
        boxes = LiDARInstance3DBoxes(
            np.asarray([[1, 2, 3, 4, 2, 2, 0]], dtype=np.float32),
            box_dim=7, origin=(0.5, 0.5, 0.5))
        np.testing.assert_allclose(boxes.tensor[0, :3], [1, 2, 2], atol=1e-6)
        np.testing.assert_allclose(boxes.gravity_center[0], [1, 2, 3], atol=1e-6)

    def test_native_class_order(self):
        out = osp.join(self.tmp.name, 'class_order')
        result = self.fixture.convert(out)
        self.assertEqual(result.returncode, 0, result.stderr)
        with open(osp.join(out, 'infos_train.pkl'), 'rb') as f:
            pkl = pickle.load(f)
        self.assertEqual(tuple(pkl['metainfo']['classes']),
                         ('ship', 'buoy', 'other'))
        self.assertEqual([i['bbox_label_3d']
                          for i in pkl['data_list'][0]['instances']], [0, 1, 2])
        self.assertEqual(pkl['metainfo']['categories'],
                         dict(ship=0, buoy=1, other=2))

    def test_empty_and_corrupt_json(self):
        spec = load_spec()
        empty = osp.join(self.tmp.name, 'empty.json')
        Path(empty).write_text('  \n', encoding='utf-8')
        self.assertIsNone(read_annotations(empty, spec))
        no_labels = osp.join(self.tmp.name, 'no_labels.json')
        Path(no_labels).write_text('{}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'missing labels'):
            read_annotations(no_labels, spec)
        broken = osp.join(self.tmp.name, 'broken.json')
        Path(broken).write_text('{"labels": [', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'invalid JSON'):
            read_annotations(broken, spec)
        filled = osp.join(self.tmp.name, 'filled.json')
        Path(filled).write_text(json.dumps(dict(labels=[box(
            'ship', 0, 0, 0, 1, 1, 1)])), encoding='utf-8')
        self.assertEqual(len(read_annotations(filled, spec)), 1)

    def test_split_boundaries(self):
        spec = load_spec()
        expected = {279: 'train', 280: 'temporal_gap', 299: 'temporal_gap',
                    300: 'val', 359: 'val', 360: 'temporal_gap',
                    379: 'temporal_gap', 380: 'test', 437: 'test'}
        for fid, role in expected.items():
            self.assertEqual(candidate_role(fid, spec), role)

    def test_timestamp_repair(self):
        path = osp.join(self.fixture.src, 'timestamp', 'timestamp.txt')
        values, fixes = read_timestamps(path, self.fixture.spec)
        self.assertEqual(values[0], 99.0)
        self.assertEqual(values[1], 100.0)
        self.assertEqual(len(fixes), 1)
        self.assertEqual(fixes[0]['rule'], 't0=t1-(t2-t1)')

    def test_illegal_calibration_rejected(self):
        bad_projection = osp.join(self.tmp.name, 'bad_p2.txt')
        write_calib(bad_projection, p2_col3=0.5)
        with self.assertRaisesRegex(ValueError, 'unsupported projection'):
            read_calib(bad_projection)
        bad_rotation = osp.join(self.tmp.name, 'bad_rot.txt')
        write_calib(bad_rotation, sheared=True)
        with self.assertRaisesRegex(ValueError, 'rigid rotation'):
            read_calib(bad_rotation)

    def test_dirty_output_rejected(self):
        out = osp.join(self.tmp.name, 'dirty')
        os.makedirs(out)
        Path(out, 'keep.txt').write_text('old', encoding='utf-8')
        result = self.fixture.convert(out)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('not empty', result.stderr)
        self.assertTrue(Path(out, 'keep.txt').exists())


if __name__ == '__main__':
    unittest.main()
