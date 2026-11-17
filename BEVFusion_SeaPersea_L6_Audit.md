# SeaPersea 标注与几何审计报告（遗留问题 L6）

> 生成日期：2026-11-16  
> 工作区：`/data/coding/mmdetection3d`  
> 数据段：`seepersea-data/barbados-20230223-240`，438 帧（`000000`~`000437`）  
> 真值口径：**以原始 `annotations/*.json` 为候选 3D 真值**，`kitti_label` / `camera_label` / `calib` / `point_clouds` 作为交叉校验与几何证据  
> 关联结论：L1 已由数据提供方确认 `buoy → Pedestrian`、`other → Cyclist`（本报告第 2 节给出数据旁证与当前代码需要修正之处）

## 可复现产物

| 路径 | 说明 |
|---|---|
| `tools/dataset_converters/audit_seepersea_geometry.py` | 本次新增的审计脚本，仅依赖 numpy（matplotlib 可选），不导入 mmdet3d、无需 CUDA 算子 |
| `outputs/seepersea_l6_audit/audit_stats.json` | 全部统计原始结果 |
| `outputs/seepersea_l6_audit/projection_000000.png` | 首帧 3D 框（绿）/ 2D 标注（红虚线）/ 点云点（黄）叠加图 |
| `BEVFusion_SeaPersea_L6_Audit.md` | 本报告 |

```bash
cd /data/coding/mmdetection3d
python tools/dataset_converters/audit_seepersea_geometry.py \
  --src-root seepersea-data/barbados-20230223-240 \
  --pkl data/seepersea/seepersea_infos_train.pkl \
  --out-dir outputs/seepersea_l6_audit
```

---

## 0. 结论摘要

| # | 审计项 | 结论 | 是否需要改代码 |
|---|---|---|---|
| 1 | 三类定义 | `annotations` 只有 `ship/buoy/other` 三类，共 1804 框（ship 968 / buoy 464 / other 372），无未知类别；`id` 仅在类别内唯一，不是全局跟踪 ID | **是**：L1 类别映射 `CATEGORY_TO_KITTI` 写反了 |
| 2 | 框中心 | `box3d.location` 是 **3D 重力中心**（LiDAR 系）；KITTI `location` 是底面中心，差 `h/2` | 否（现有转换正确） |
| 3 | 尺寸顺序 | KITTI `(h,w,l) = (ann.height, ann.length, ann.width)`，即 **`ann.width` 是沿航向长度 l，`ann.length` 是横向宽度 w**；pkl 需要 `[l,w,h]` | **仅改注释**：数值顺序正确但注释写成 `[w,l,h]`，有误导性 |
| 4 | yaw | 弧度、绕 LiDAR +z、从 +x 向 +y 增加，范围 `[-π, π]`，pitch/roll 恒 0；与 KITTI 的关系 `ry = -yaw - π/2` | 否（无需取负/加 π/2） |
| 5 | 点云/框投影 | 标定链路正确（`P2` 第 4 列全 0，`lidar2img` 等价 `P2·R0·Tr`）；3D 框投影与 2D 标注大致对齐但不同源，**2D 框不能当严格 GT** | 否 |
| 6 | 范围内目标比例 | 训练 `ObjectRangeFilter` 只看 xy 中心 → **79.3%** 框保留；按 3D 范围只有 **65.1%**；点云 **20.0%** 被 `PointsRangeFilter` 丢弃（HANDOFF 中「约 14%」应更正） | **是**：重设 `point_cloud_range` 等几何参数 |
| 7 | 每框点数 | 全部 1804 框都 ≥1 点（均值 50.0 / 中位 18），无空框；`num_lidar_pts=1` 是低估的占位值 | **建议是**：写入真实点数 |
| 8 | 空标注帧 | 66 帧（15.1%）`annotations` 与 `camera_label` 同时为空，点云/图像/时间戳均正常，前后相邻帧都有目标 → **整帧标注缺失/有意跳过**，不是场景无目标 | **是**：训练/评估时显式剔除，别当负样本 |
| 9 | pkl 转换 | 438 帧 1804 框，`bbox_3d` 与原始标注数值零误差、无漏框；但标签 id 仍是错误的 L1 映射 | **是**：修正映射后重跑转换 |

---

## 1. 审计范围与方法

- **候选真值**：`annotations/*.json` 的 `labels[].box3d`（`location` / `dimension` / `orientation`）与 `category`，438 个文件、1804 个 3D 框、无未知类别。
- **交叉校验**：`kitti_label/*.txt`（KITTI 15 列，373 行）、`camera_label/*.json`（LabelMe 2D 框，2419 个）、`calib/*.txt`、`point_clouds/*.pcd`（ASCII，139391 点）、`timestamp/timestamp.txt`、`empty_label_files.txt`。
- **转换产物**：`data/seepersea/seepersea_infos_train.pkl`（438 帧 / 1804 框）。
- **几何判定**：按 mmdet3d `LiDARInstance3DBoxes` 约定（底面中心 origin、绕 z 旋转）生成 8 角点，做旋转框内点判定、`lidar2img` 投影、2D IoU、范围过滤复现。
- **范围定义**：沿用 BEVFusion 融合配置 `point_cloud_range = [-54,-54,-5,54,54,3]`。

框架侧关键约定（决定「原始标注该怎样写进 pkl」）：

```258:258:tools/dataset_converters/nuscenes_converter.py
            gt_boxes = np.concatenate([locs, dims[:, [1, 0, 2]], rots], axis=1)
```

官方转换器把 devkit 的 `wlh` 显式换成 `lwh`（即 `[x_size, y_size, z_size]`），所以 **v2 info pkl 的 `bbox_3d` 是 `[x, y, z, l, w, h, yaw]`**；`NuScenesMetric.output_to_nusc_box` 再用 `box_dims[:, [1,0,2]]` 换回 `wlh`。

```184:187:mmdet3d/datasets/nuscenes_dataset.py
            gt_bboxes_3d = LiDARInstance3DBoxes(
                ann_info['gt_bboxes_3d'],
                box_dim=ann_info['gt_bboxes_3d'].shape[-1],
                origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)
```

`origin=(0.5,0.5,0.5)` 表示 **pkl 中的 z 是 3D 中心**，框架内部才转成底面中心：

```26:29:mmdet3d/structures/bbox_3d/lidar_box3d.py
    The relative coordinate of bottom center in a LiDAR box is (0.5, 0.5, 0),
    and the yaw is around the z axis, thus the rotation axis=2. The yaw is 0 at
    the positive direction of x axis, and increases from the positive direction
    of x to the positive direction of y.
```

---

## 2. 三类定义与 L1 闭环

### 2.1 三套标注互不等价

| 标注源 | ship / Car | buoy / Pedestrian | other / Cyclist | 合计 | 说明 |
|---|---|---|---|---|---|
| `annotations`（3D 真值） | 968 (53.7%) | 464 (25.7%) | 372 (20.6%) | **1804** | 每帧均值 4.12 框，最多 28 框 |
| `camera_label`（2D 人工框） | 1648 | 244 | 527 | 2419 | 每帧均值 5.52 框，与 3D 数量不一致 |
| `kitti_label`（KITTI 3D） | Car 369 | Pedestrian 4 | **无 Cyclist** | 373 | 每帧最多 2 行，是 2D 大框的子集 |

- 373 个 KITTI 行的 2D bbox **100% 精确出现在 `camera_label` 中**（`kitti_2d_boxes_found_in_camera_label=373`），说明 `kitti_label` 是 `camera_label` 的筛选子集。
- 筛选偏向大 2D 框：`camera_label` 2D 框面积中位数 209 px²，而 `kitti_label` 保留的框面积最小 45.5 px²、p10 1585 px²、中位数 12330 px²。分类看：`other` 2D 框面积中位 20.3 px²、**最大仅 556 px²、`>1000 px²` 的数量为 0**；`buoy` 中位 20.7 px²、`>1000 px²` 仅 4 个；`ship` 中位 815.6 px²、p90 13500 px²。
- 这就是 **L1 疑点的根因**：`other`（应为 `Cyclist`）和绝大多数 `buoy`（应为 `Pedestrian`）目标在图像里太小，被 `kitti_label` 过滤，故全段只出现 `Car(369)` 与 `Pedestrian(4)`。

### 2.2 L1 结论的数据旁证

- 4 个 `Pedestrian` 行全部与 `buoy` 的 3D 框**精确几何匹配**（dim/loc/yaw 误差 <1e-5）；369 个 `Car` 行中有 294 个与 `ship` 精确匹配。
- 因此数据证据与数据提供方结论一致：**`ship→Car`、`buoy→Pedestrian`、`other→Cyclist`**。
- 当前转换器仍是旧映射，必须修正：

```32:37:tools/dataset_converters/seepersea2bevfusion.py
CATEGORY_TO_KITTI = {
    'ship': 'Car',
    'buoy': 'Cyclist',
    'other': 'Pedestrian',
}
```

改为 `'buoy': 'Pedestrian'`、`'other': 'Cyclist'` 后，pkl 的 `bbox_label_3d` 应从 `buoy→bicycle(5) / other→pedestrian(7)` 变为 `buoy→pedestrian(7) / other→bicycle(5)`（`ship→car(0)` 不变）。

### 2.3 `kitti_label` 不能当完整 3D 真值

373 行中 **298 行（79.9%）** 与同帧 `annotations` 的 3D 参数完全一致（判定：`dims`、`loc`、`ry` 的最大绝对误差 <1e-5），其余 **75 行（20.1%）** 不一致：尺寸最大差 21.3 m（中位 6.9 m）、位置最大差 5.11 m（中位 1.73 m）。典型例子第 183 帧：`kitti_label` 的 `Car` 为 `[h,w,l]=[8.17,10.75,9.96]`、对应 `camera_label` 的大船 2D 框 `[199,0,335,105]`，而该帧 `annotations` 只有 5 个框、最大仅 1.46 m。**这 75 行的 3D 参数来自另一套单目/2D 驱动的估计，且 2D 框筛选偏差大**，只能用作类别与朝向的旁证，不能做 GT。

### 2.4 类别尺寸先验与 nuScenes 严重失配（供 L2/L3 参考）

| 类别 | l 中位 / 最大 (m) | w 中位 / 最大 (m) | h 中位 / 最大 (m) | 中心 z 中位 (m) | 映射到的 nuScenes 类 |
|---|---|---|---|---|---|
| ship | 3.90 / 19.79 | 3.73 / 15.79 | 1.19 / 20.77 | -1.82 | Car |
| buoy | 0.37 / 1.79 | 0.35 / 1.43 | 0.82 / 2.59 | +0.67 | Pedestrian |
| other | 4.57 / 9.14 | 4.65 / 9.58 | 1.07 / 2.49 | -0.20 | Cyclist |

`buoy` 多为 0.2 m 立方体（远小于行人），`other` 约 4.6×4.7×1.1 m（远大于自行车），`ship` 存在 20 m 级异常大框。预训练权重的尺寸回归先验、以及将来 nuScenes 风格的评估阈值，都会与这些类别系统性失配；改 3 类头时应同步重设尺寸/距离阈值，不能只换类名。

---

## 3. 3D 框几何约定（中心 / 尺寸 / yaw）

用 373 个 KITTI 行做「原始标注 → KITTI 标注」的正向复现，逐一验证候选公式。

### 3.1 框中心：3D 重力中心

| 假设 | 位置误差中位数 | 位置误差均值 | 判定 |
|---|---|---|---|
| `loc_kitti = lidar2cam @ (loc_ann - [0,0,h/2])`（底面中心） | **6.7e-7 m** | 0.456 m（受 75 行不同源标注污染） | ✅ 298 行误差 <1e-5 |
| `loc_kitti = lidar2cam @ loc_ann`（不修正） | 1.83 m | 2.00 m | ❌ |

首帧可手算复核：`ship` 中心 `(65.395, 15.484, -10.498)` 减 `h/2=0.522` 后经 `Tr_velo_to_cam` 得 `(-19.887, -14.699, 63.626)`，与 `kitti_label/000000.txt` 的 `(-19.888, -14.698, 63.624)` 吻合。

**结论**：`annotations` 的 `location` 是 3D 中心，转换脚本直接把它写入 `bbox_3d` 是正确的（框架以 `origin=(0.5,0.5,0.5)` 读取）。注意这里的 `z` 不是「地面以上高度」，而是海面/传感器坐标系的垂直坐标：点云 z 中位 -0.80 m、p1 -7.96 m、p99 11.64 m，框中心 z ∈ [-17.94, 10.93]。

### 3.2 尺寸顺序：`ann.width` 是沿航向长度

| 假设 | 与 KITTI `(h,w,l)` 的 dim 误差中位数 | 判定 |
|---|---|---|
| A：KITTI `[h,w,l] = [ann.height, ann.length, ann.width]` | **0.0**（298 行 <1e-5） | ✅ |
| B：KITTI `[h,w,l] = [ann.height, ann.width, ann.length]` | 1.33 m | ❌ |

首帧 `buoy` 例：annotation `dimension = {width:0.8656, length:0.7533, height:1.6014}`，KITTI `Pedestrian` 的 dims 为 `1.6014 0.7533 0.8656`（h,w,l），即 `ann.width` 对应 KITTI 的 `l`（沿航向），`ann.length` 对应 `w`（横向）。

**结论**：mmdet3d pkl 需要 `[l,w,h]`，转换脚本写 `[ann.width, ann.length, ann.height]` **数值顺序正确**；但字段名注释写反了，建议修正以免后续维护者按 devkit 的 `wlh` 语义「纠正」成错误顺序：

```161:166:tools/dataset_converters/seepersea2bevfusion.py
        # v2 nuScenes convention: [x, y, z, w, l, h, yaw]
        bbox_3d = [
            float(loc['x']), float(loc['y']), float(loc['z']),
            float(dim['width']), float(dim['length']), float(dim['height']),
            float(yaw)
        ]
```

应改为 `# [x, y, z, l, w, h, yaw]` 并注明 `ann.width` 实为沿航向长度。辅助证据：按 A 顺序做旋转框内点判定，1804 框**无 0 点框**、均值 50.0 点/框；按 B（交换 l/w）会出现 0 点框且均值下降。

### 3.3 yaw：弧度、绕 +z、方向从 +x 到 +y

| 公式 | 与 KITTI `rotation_y` 的最大误差 |
|---|---|
| `ry = -yaw - π/2` | **4.4e-16 rad**（373/373 行） ✅ |
| `ry = yaw` | 3.14 rad |
| `ry = -yaw + π/2` / `ry = -yaw` | 恒为 π / π/2 |
| `ry = yaw - π/2` | 3.13 rad |

全段 yaw ∈ [-3.140, 3.130]，`|yaw| > π` 的数量为 **0**，pitch/roll 非 0 的数量为 **0**，`rotationYaw` 缺失数为 0。因此 yaw 已是弧度、且与 `LiDARInstance3DBoxes` 的 yaw 定义一致，**转换脚本直接写 `ann.rotationYaw` 即可，不需要取负或加常量**；KITTI 的 `ry` 只是坐标变换后的相机系角度。

### 3.4 转换产物校验

`data/seepersea/seepersea_infos_train.pkl`：438 帧、1804 框、**框数不一致帧 0**、逐框 `bbox_3d` 与原始标注 `max|diff| = 0.0`。**几何写入没有丢精度，唯一错误是类别 id（见 2.2）**。另需注意：pkl 的 `metainfo.categories = {ship:0, buoy:1, other:2}` 是自定义 3 类 id，而 `bbox_label_3d` 用的是 nuScenes 10 类 id；`Det3DDataset.label_mapping` 只认 `metainfo['classes']`，这里的 `categories` 只是未使用的元数据，容易误导，建议改 3 类头时一并统一。

---

## 4. 点云 / 3D 框投影检查

### 4.1 标定链路

- 全段 438 个 `calib` 的 `P2` 完全相同，`P2` 第 4 列最大绝对值 **0.0**，因此 `cam2img(3×3) @ lidar2cam(4×4)` 与 `P2 @ R0_rect @ Tr_velo_to_cam` 完全等价，转换器拆分标定矩阵的方式正确。
- `R0_rect` 为单位阵，`Tr_velo_to_cam` 全段一致（与 HANDOFF 一致）。

### 4.2 点云投影

| 指标 | 数值 |
|---|---|
| 总点数 | 139391（每帧 134~570，均值 318.2） |
| 相机前方（depth>0） | 85893（61.62%） |
| 落在 640×480 图像内 | 35347（**25.36%**） |
| depth ∈ LSS `dbound=[1,60]` | 80610（57.83%） |
| 同时在图像内且 ∈ dbound | 34323（24.62%） |
| 前向点深度 p10/p50/p90/p99 | 2.18 / 23.02 / 44.80 / **61.86 m** |

单相机视野只覆盖约 1/4 的点；深度上界 60 m 覆盖约 99% 的前向点，略紧但基本合适（若要覆盖 p99 以上可放宽到 70~80 m）。点云坐标无重复（`unique_xyz_ratio=1.0`）。

### 4.3 3D 框投影

- 1804 框中 **951 框（52.7%）** 的 8 个角点全部位于相机前方；其余在身后或跨过成像平面。
- 用 298 个「3D 参数精确匹配」的 KITTI 行作可信锚点，把 3D 框 8 角点投影外接框与对应 2D 框比较：IoU 均值 **0.285** / 中位 **0.270**，中心误差均值 **34.5 px** / 中位 **31.3 px**。假设 A（l,w,h）与假设 B（w,l,h）的 IoU 几乎相同（0.285 vs 0.283），因为多数框近立方体 —— **投影无法区分 l/w 顺序**，该结论以 3.2 的 KITTI dims 匹配为准。
- 叠加图 `projection_000000.png` 显示：左侧大船的绿色 3D 框与红色 2D 框吻合，右侧 `buoy` 的绿色 3D 框明显大于其 2D 框，且 2D 红框数量（6）多于可见 3D 框。原因是 `camera_label` 是独立的 2D 人工标注（可见区域/远处目标），`annotations` 是点云驱动的 3D 标注：**两者大致对齐但不等价，2D 框不能作为几何校准或评估的严格真值**。

---

## 5. 范围内目标比例与点云裁剪（L6 核心）

### 5.1 点云被丢弃的真实比例

`PointsRangeFilter` 的 3D 范围 `[-54,-54,-5,54,54,3]`：

| 项 | 点数 | 占比 |
|---|---|---|
| 保留 | 111507 | **80.00%** |
| 丢弃 | **27884** | **20.00%** |
| 其中 z 越界（`z<-5` 或 `z>3`） | 25423 | 18.24%（`z<-5`: 11445；`z>3`: 13978） |
| 其中 xy 越界 | 8705 | 6.24% |
| 同时 z 与 xy 越界（重叠） | 6244 | 4.48% |

> **更正 HANDOFF L6 的「约 14%」**：按原始 PCD 逐点复现，实际是 **20.00%**；若只看 xy 是 6.24%，只看 z 是 18.24%。推测旧数字来自不同统计口径（例如只统计某一方向或某帧子集）。后续以本报告与 `audit_stats.json` 为准。主因是 **nuScenes 的 z 高度假设 `[-5,3]` 不适用于海面坐标系**（点云 z 范围 [-67.79, 71.71]，p10 -4.70、p90 3.01）。

### 5.2 目标（3D 框）在范围内的比例

| 类别 | 框数 | 中心在 3D 范围内 | 中心在 BEV 范围内（`ObjectRangeFilter`） | z 越界 | xy 越界 | 8 角点全在 3D 范围内 |
|---|---|---|---|---|---|---|
| ship | 968 | 416 (**43.0%**) | 595 (61.5%) | 401 | 373 | 270 |
| buoy | 464 | 412 (88.8%) | 464 (**100%**) | 52 | 0 | 391 |
| other | 372 | 346 (93.0%) | 372 (**100%**) | 26 | 0 | 330 |
| **合计** | **1804** | **1174 (65.1%)** | **1431 (79.3%)** | **479 (26.6%)** | **373 (20.7%)** | 991 |

- `ObjectRangeFilter` 通过 `in_range_bev`（`mmdet3d/structures/bbox_3d/base_box3d.py:181`）只检查 **框中心 x/y**，不看 z；所以它保留 79.3% 的框，但其中不少目标的高度已超出体素化范围，`ship` 尤其严重。
- **训练/测试过滤不对称**：训练 pipeline 同时有 `ObjectRangeFilter`，而 test pipeline 只有 `PointsRangeFilter`：

```98:99:projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
```

```163:164:projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py
    dict(
        type='PointsRangeFilter',
        point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]),
```

将来做真实评估（L3）时，训练可见的 GT 集合与评估使用的 GT 集合会不一致，必须统一。

### 5.3 候选范围覆盖率（`audit_stats.json: range_coverage`）

| xy 范围 | z 范围 | 点覆盖率 | 框中心覆盖率 |
|---|---|---|---|
| ±54 | [-5, 3]（当前） | 80.00% | 65.08% |
| ±54 | [-10, 10] | 93.70% | 79.21% |
| ±75 | [-10, 10] | 98.17% | 96.84% |
| ±75 | [-20, 20] | 99.33% | 98.61% |
| ±100.8 | [-20, 20] | 99.80% | **100%** |
| ±108（= 1440×0.075） | [-20, 20] | **99.81%** | **100%** |

**参数建议（两档，均需实测显存/权重加载）**：

- **A 档（改动最小，推荐先做）**：`point_cloud_range=[-54,-54,-10,54,54,10]`，保持 `voxel_size=[0.075,0.075,0.2]` → `sparse_shape` 由 `[1440,1440,41]` 改为 `[1440,1440,101]`；`bbox_coder.pc_range=[-54,-54]`、`xbound/ybound=[-54,54,0.3]`、`zbound=[-10,10,20]` 不变；`dbound` 可放宽到 `[1,80,0.5]`。点覆盖率 80.0%→93.7%，框中心 65.1%→79.2%，代价是体素 z 维度 41→101。
- **B 档（覆盖完整但需降分辨率）**：`point_cloud_range=[-100.8,-100.8,-20,100.8,100.8,20]`，`voxel_size` 放大到 `[0.3,0.3,0.4]` → `sparse_shape=[672,672,100]`，同步 `xbound/ybound=[-100.8,100.8,0.6]`（保持 336 bins）、`zbound=[-20,20,40]`、`bbox_coder.pc_range=[-100.8,-100.8]`、`voxel_size=[0.3,0.3]`。覆盖率 99.8% / 100%，但 BEV 网格从 360×360 变为 336×336，LiDAR 分支分辨率变粗，需验证预训练权重加载与显存。

必须同步的字段：`point_cloud_range`、`voxel_size`、`sparse_shape`、`view_transform.xbound/ybound/zbound/dbound`（`gen_dx_bx` 生成 BEV 网格，xbound/ybound 的 bin 数需与 BEV 特征尺寸配套；`zbound=[-10,10,20]` 只有 1 个 z bin，LSS 输出被压成 2D BEV，故 z 维度不决定特征形状）、`bbox_coder.pc_range` / `voxel_size`、以及 train/test 两个 pipeline 的 `PointsRangeFilter`。

### 5.4 sweeps 相关（L6 另一半）

`LoadPointsFromMultiSweeps(sweeps_num=9, pad_empty_sweeps=True)` 在 `ego2global` 恒为单位阵、无 `lidar_sweeps` 的情况下，只是把同一关键帧点集重复 10 次，**不提供任何新的几何/时序信息**；`remove_close=True` 只移除 `|x|<1 且 |y|<1` 的点（`mmdet3d/datasets/transforms/loading.py:374-396`），对本段几乎无影响。建议在没有真实位姿前，直接把 `sweeps_num=0` 或关闭该 transform，避免 10 倍冗余体素化开销与「伪多帧」误导。

---

## 6. 每框点数与稀疏性

按 `[l,w,h] + yaw` 的旋转框判定（假设 A）：

| 类别 | 框数 | 点数均值 / 中位 / p10 / p90 | 0 点 | 1 点 | ≥2 点 | ≥5 点 | 交换 l/w 后均值 |
|---|---|---|---|---|---|---|---|
| ship | 968 | 54.6 / 12.5 / 2 / 169 | 0 | 59 | 909 | 723 | 50.7（且出现 0 点框） |
| buoy | 464 | 48.6 / 7 / 2 / 183 | 0 | 27 | 437 | 284 | 42.3（且出现 0 点框） |
| other | 372 | 39.8 / 40 / 20 / 61 | 0 | 0 | 372 | 369 | 36.9（且出现 0 点框） |
| 合计 | 1804 | 50.0 / 18 / 2 / 144 | **0** | 86 | 1718 | 1376 | — |

- 「每帧约 300 点」并不等于「每个目标只有几个点」：点云是**局部密集点簇**（无重复点），例如首帧第一个 `buoy` 框 `0.87×0.75×1.60 m` 内即含 89 点；同时少量 `ship` 框达 20 m 级，单框最多 513 点。均值被大框/密簇拉高，中位 18 点更能代表典型情况。
- 每个框中心到最近点云点的距离：中位 **0.30 m**、p90 1.16 m、最大 2.86 m；100 个框（5.5%）中心与某点坐标完全重合 → **3D 标注是点云驱动的**，几何自洽。
- **结论**：转换器里 `num_lidar_pts=1` 的占位值虽能绕过 `NuScenesDataset._filter_with_mask`（`num_lidar_pts>0`），但严重低估真实点数。建议写入真实框内点数，便于后续按点数过滤、按点数/距离分桶评估（nuScenes 指标依赖此类分桶）。

### 附带发现：强度被裁剪

PCD intensity 范围 2~7256，中位 30、p90 285；转换器的 `np.clip(pts[:,3], 0, 255)` 会把 **14454 点（10.37%）** 截断到 255（其中 9944 点 >1000），强度信息受损：

```228:232:tools/dataset_converters/seepersea2bevfusion.py
        if pts.size:
            # nuScenes stores intensity in [0, 255]; clip for numerical sanity
            intensity = np.clip(pts[:, 3:4], 0.0, 255.0)
            pts5 = np.concatenate(
                [pts[:, :3], intensity, np.zeros((len(pts), 1))],
```

建议改为保序映射到 `[0,255]`（如 `255*log1p(I)/log1p(I_max)` 或按分位数截断缩放），并在配置侧确认 VFE 输入分布；代价是进一步偏离 nuScenes 预训练分布，需权衡。

---

## 7. 66 帧空标注的原因

| 检查项 | 结果 |
|---|---|
| `annotations` / `camera_label` / `kitti_label` 空文件数 | **66 / 66 / 109**（前两者空帧集合完全相同） |
| `empty_label_files.txt` 清单 | 66 个 `.json` + 66 个 `.txt`，与检测到的 66 帧**完全一致** |
| 空帧点云 | 148~570 点（均值 291.8）；有标注帧 134~569 点（均值 322.9）→ 同分布，非空帧 |
| 空帧图像 / 时间戳 | 66/66 图像存在；时间戳异常 0（第 0 帧哨兵值不在这 66 帧内） |
| 点云是否为上一帧的重复 | 0/66 |
| 前后相邻有标注帧是否都有目标 | **66/66** 是 |
| `kitti_label` 额外空的 43 帧 | 集中在 5、21~23、332~334、386~436 等段，属于「2D 大框为空」，不是 3D 标注缺失 |

**结论**：这 66 帧在 3D 与 2D 标注上被整帧清空，但点云、图像、时间戳全部正常，且相邻帧目标持续存在 —— 属于**标注/导出环节整帧跳过（数据方也用 `empty_label_files.txt` 显式标记）**，而不是「场景确实无目标」或传感器故障。现有证据无法区分「有意标记无目标」与「漏标」，但两种解释下都不应把这些帧当作「无目标负样本」用于训练或评估。

**处理建议**：训练/评估时按清单显式剔除；如需补齐，可用相邻帧目标做插值并标记为低置信度，但不要静默当作空 GT。`kitti_label` 额外的 43 帧是另一回事（2D 筛选结果），不应混用。

---

## 8. 修复清单（按优先级）

| # | 优先级 | 改动 | 涉及文件 | 依据 |
|---|---|---|---|---|
| 1 | P0 | `CATEGORY_TO_KITTI` 改为 `buoy→Pedestrian`、`other→Cyclist`，重跑转换生成 pkl（标签 id 变为 `ship=0, buoy=7, other=5`） | `tools/dataset_converters/seepersea2bevfusion.py` | §2.2、§2.3 |
| 2 | P0 | 修正 `bbox_3d` 注释为 `[x,y,z,l,w,h,yaw]`，并注明 `ann.width` 是沿航向长度（**数值不要改**） | 同上 | §3.2 |
| 3 | P0 | `num_lidar_pts` 改为真实旋转框内点数 | 同上 | §6 |
| 4 | P0 | 空标注 66 帧：训练/评估显式剔除，或补齐并标低置信度 | 转换脚本 / 配置文件 | §7 |
| 5 | P1 | 重设几何范围（建议先用 A 档 `[-54,-54,-10,54,54,10]`），同步 `sparse_shape`、LSS `xbound/ybound/dbound`、`bbox_coder.pc_range`、train/test 两个 `PointsRangeFilter` | 新建 seepersea 专用配置（L10） | §5.3 |
| 6 | P1 | 统一 train/test 的 `ObjectRangeFilter` 口径，避免评估 GT 与训练 GT 集合不一致 | 配置 / evaluator（L3） | §5.2 |
| 7 | P1 | 强度映射替代硬裁剪；关闭 `LoadPointsFromMultiSweeps` 或补真实 sweeps 位姿（L7） | 转换脚本 / 配置 | §6、§5.4 |
| 8 | P2 | 明确 2D/3D 标注不同源：评估只用 `annotations` 作 GT；`kitti_label` 仅作旁证 | evaluator / 文档 | §2.3、§4.3 |
| 9 | P2 | 改 3 类头时同步重设尺寸先验/评估阈值，并统一 pkl 的 `metainfo.classes` | 配置 / 转换脚本（L2、L3） | §2.4、§3.4 |

---

## 9. 证据索引

| 结论 | 证据位置 |
|---|---|
| 全部统计数值 | `outputs/seepersea_l6_audit/audit_stats.json` |
| 类别计数、KITTI 精确匹配率、yaw 公式残差 | `audit_stats.json: categories / kitti_crosscheck / yaw` |
| 框中心/尺寸/yaw 复现逻辑 | `tools/dataset_converters/audit_seepersea_geometry.py: read_kitti / box_of / box_corners / KITTI cross-check` |
| 投影与可视化 | `audit_stats.json: projection`、`outputs/seepersea_l6_audit/projection_000000.png` |
| 范围丢弃比例与候选范围覆盖率 | `audit_stats.json: point_cloud / boxes / range_coverage` |
| 空帧诊断 | `audit_stats.json: files / empty_frames`，`seepersea-data/.../empty_label_files.txt` |
| 框架约定 | `tools/dataset_converters/nuscenes_converter.py:258`、`mmdet3d/datasets/nuscenes_dataset.py:175-189`、`mmdet3d/structures/bbox_3d/lidar_box3d.py:26-29`、`mmdet3d/evaluation/metrics/nuscenes_metric.py:588-590` |
| 范围过滤器 | `mmdet3d/structures/bbox_3d/base_box3d.py:181 / 260`、`mmdet3d/datasets/transforms/transforms_3d.py:845 / 905` |
