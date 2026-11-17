# BEVFusion 本地数据集接入 · 交接文档

> 生成日期：2026-11-16  
> 工作区：`/data/coding/mmdetection3d`  
> 面向：接手本项目的其他 agent。本文汇总两轮对话的全部结论、产物、实测数据与待办。  
> 语言：简体中文（与原对话一致）。所有命令默认在仓库根目录执行。

---

## 1. 项目背景与目标

仓库是 **MMDetection3D v1.4.0**，BEVFusion 位于 `projects/BEVFusion/`（`projects` 扩展，非核心包）。用户最终目标是把 BEVFusion 用于**自有海面感知数据集**，分两步推进：

1. **阶段一**：先基于仓库现有配置验证官方 BEVFusion 是否可正常运行（环境、权重、示例数据、命令、判定标准）。
2. **阶段二**：把本地样本段 `seepersea-data/barbados-20230223-240` 转成 BEVFusion 可读格式，用 nuScenes 预训练权重验证**可行性**。用户明确表示「暂时不用关注是否准确」。

**当前结论**：两步均已完成，可行性已验证 —— 438 帧全部通过前向推理。尚未进入精度/训练阶段。

---

## 2. 环境与关键约束（已实测通过）

| 组件 | 实测值 |
|---|---|
| Python | 3.10.14 |
| PyTorch / CUDA | `2.0.1+cu118` / CUDA 11.8，`nvcc` V11.8.89 |
| GPU | Quadro RTX 6000，24 GB（sm_75） |
| mmcv / mmengine / mmdet / mmdet3d | 2.1.0 / 0.10.5 / 3.3.0 / 1.4.0 |
| spconv | `spconv-cu118` 2.3.8，`IS_SPCONV2_AVAILABLE=True` |
| 运行时断言 | `mmcv>=2.0.0rc4,<2.2.0`、`mmengine>=0.8.0,<1.0.0`、`mmdet>=3.0.0rc5,<3.4.0`（`mmdet3d/__init__.py`、`docs/en/notes/faq.md:14`） |

两条**必须记住**的前置条件：

- **编译 BEVFusion 自有 CUDA 算子**：`python projects/BEVFusion/setup.py develop`，生成 `bev_pool_ext` 与 `voxel_layer` 两个 `.so`。未编译时 `Config.fromfile()` 会报 `Failed to import custom modules ... You should set PYTHONPATH`，该提示具有误导性，真因是算子缺失。`setup.py:26-29` 硬编码 `sm_70/75/80/86`。
- **必须安装 spconv 2.x**（`pip install spconv-cu118`）。否则会回退到 `mmcv.ops.SparseConv3d`，官方权重出现 **21 条 `size mismatch for pts_middle_encoder.*`**，稀疏卷积权重被静默跳过；此时程序退出码仍为 0、仍输出检测框，但 LiDAR 分支是随机初始化。**"能跑通"不等于"权重正确加载"**。

其他环境事实：`pip install -v -e .` 不安装 mmcv/mmdet/mmengine，必须用 `mim`；`model-index.yml` 无 BEVFusion 条目，`mim download` 不可用；纯 CPU 不支持；`docs/zh_cn/get_started.md:7` 与英文版版本要求不一致，以英文版和运行时断言为准。

---

## 3. 已完成工作清单及关键产出

### 3.1 阶段一：官方 BEVFusion 可用性验证

| # | 工作 | 结论 / 产出 |
|---|---|---|
| 1 | 梳理版本、依赖、编译与权重来源 | 见第 2 节；明确官方权重 URL、无 `metafile.yml`、Swin `init_cfg` 会从 GitHub 下载 |
| 2 | 实际编译 CUDA 算子 | 两个 `.so` 生成，`import projects.BEVFusion.bevfusion` 通过 |
| 3 | 安装 `spconv-cu118` | 消除 21 条 size mismatch（前后对比已实测） |
| 4 | 下载官方 lidar-cam 权重 | `checkpoints/bevfusion_lidar-cam_5239b1af.pth`（160 MB / 167113973 B，582 个参数键） |
| 5 | nuScenes 官方 demo 实测 | `outputs/bevfusion_demo.png`（9.6 MB）；200 个预测框、最高分 ≈0.82、峰值显存 ≈2.2 GB |
| 6 | mini 评估方案梳理 | `tools/create_data.py nuscenes --version v1.0-mini`；无 mini 版配置，需 `--cfg-options`；`NuScenesMetric` 需 `metainfo.version=v1.0-mini` 才会选 `mini_val`；官方 mini pkl 无 `lidar_sweeps` |
| 7 | 判定标准 | 版本断言通过 + 算子可导入 + 权重零 mismatch + demo 有框有分数 + `tools/test.py` 输出有限 mAP/NDS + 训练 loss 有限 |

**重要纠错（已确认）**：`docs/en/user_guides/inference.md:84` 的示例已过时 —— 引用的 `bevfusion_voxel0075_...py` 不存在，根目录 `demo/multi_modality_demo.py` 不识别 `--score-thr` 且会丢弃 `--cam-type`。**必须使用 `projects/BEVFusion/demo/multi_modality_demo.py`**。

**未执行**：nuScenes mini 的定量评估（需下载约 4 GB 官方 mini 数据，官网需注册）。当前环境 `data/` 下只有 lyft/s3dis/scannet/sunrgbd，没有 nuScenes。

### 3.2 阶段二：SeaPersea 数据集转换与验证

#### 数据勘察结论

`seepersea-data/barbados-20230223-240`，**438 帧**（`000000`~`000437`）：

| 目录 | 内容 |
|---|---|
| `point_clouds/` | 438 个 ASCII `.pcd`，`FIELDS x y z intensity`，约 300 点/帧，共 139391 点 |
| `undist_images/CAM_FRONT/` | 438 张 640×480 RGB PNG，**仅 1 个相机** |
| `calib/` | KITTI 格式 `P2`(3×4)、`R0_rect`(单位阵)、`Tr_velo_to_cam`(3×4)，全段一致 |
| `annotations/` | 核心标注，`box3d.location` 已在 LiDAR 坐标系；**66 个文件为空** |
| `kitti_label/` | KITTI 3D 标注 |
| `camera_label/` | 2D 框（未使用） |
| `timestamp/timestamp.txt` | 帧号→unix 秒；**第 0 帧是哨兵值 `-99999999`** |
| `motion_data/` | `gps.csv`、`imu.csv`（未使用） |
| `empty_label_files.txt` | 空标签清单（未使用） |

标注类别统计：`ship` 968、`buoy` 464、`other` 372。

**坐标系已交叉校验**：`annotations/000000.json` 的 ship 中心 `(65.395, 15.484, -10.498)` 经 `Tr_velo_to_cam` 得 `≈(-19.87, -15.18, 63.6)`，与 `kitti_label/000000.txt` 的 Car `(-19.888, -14.698, 63.624)` 在 KITTI「底面中心 vs 3D 中心」修正后一致 → 标定拆分方向正确。

#### 关键方案决策

- **点云**：4 维 ASCII PCD → `float32` **5 通道** `.bin`（`x,y,z,intensity,0`），强度裁剪到 `[0,255]`。原因：配置 `LoadPointsFromFile(load_dim=5)`、`HardSimpleVFE(num_features=5)`。
- **相机**：仅 `CAM_FRONT`，不伪造 6 目。`bevfusion.py:141` 与 `depth_lss.py:80` 的 `N` 是动态的，单视角可直接跑通（已实测）。
- **标定**：`cam2img = P2[:3,:3]`（3×3）；`lidar2cam = R0_rect(4×4) @ Tr_velo_to_cam(4×4)`（4×4）。`BEVLoadMultiViewImageFromFiles` 自行推导 `lidar2img`。
- **标注**：`instances[].bbox_3d=[x,y,z,w,l,h,yaw]`（v2 nuScenes 约定）；额外补 `velocity=[0,0]`（`NuScenesDataset.with_velocity=True` 必需）、`num_lidar_pts=1`（`_filter_with_mask` 用 `num_lidar_pts>0` 过滤）、`bbox_3d_isvalid=True`。
- **类别头不改**：保持 `bbox_head.num_classes=10` 与 10 类 `metainfo`，确保预训练权重零 mismatch。按标准 nuScenes 类序映射 `ship→Car→car(0)`、`buoy→Cyclist→bicycle(5)`、`other→Pedestrian→pedestrian(7)`，再由 `Det3DDataset.label_mapping` 转到 BEVFusion 配置类序。
- **图片不复制**：用 symlink 指向原图，产物仅 4.2 MB。
- **评估器**：`NuScenesMetric(format_only=True)` —— 只做结果格式化、不实例化 `NuScenes()`，因此**不需要 nuScenes devkit 元数据**，自定义数据也能走官方 `tools/test.py` 流水线并输出 nuScenes 格式 `results_nusc.json`。

#### 生成的文件

| 路径 | 说明 |
|---|---|
| `tools/dataset_converters/seepersea2bevfusion.py` | **新增代码**，转换脚本（9.9 KB） |
| `data/seepersea/samples/LIDAR_TOP/*.bin` | 438 个 5 通道点云，无 NaN、无空帧 |
| `data/seepersea/samples/CAM_FRONT/*.png` | 438 个 symlink，全部可解析 |
| `data/seepersea/seepersea_infos_train.pkl` | 438 帧，v2 格式 `metainfo`+`data_list`，共 1804 个 3D 框 |
| `data/seepersea/seepersea_infos_val.pkl` | 前 8 帧，快速验证用 |
| `outputs/seepersea_demo.png` | 自定义数据首帧可视化（315 KB） |
| `work_dirs/seepersea/results/pred_instances_3d/results_nusc.json` | 8 帧结果 |
| `work_dirs/seepersea_full/results/pred_instances_3d/results_nusc.json` | 438 帧结果 |
| `checkpoints/bevfusion_lidar-cam_5239b1af.pth` | 阶段一下载的预训练权重 |

#### 实测结果（验证结论）

| 验证 | 命令要点 | 结果 |
|---|---|---|
| 官方 demo 单帧 | `projects/BEVFusion/demo/multi_modality_demo.py ... --cam-type all --score-thr 0.05` | 退出码 0，权重零 mismatch，生成 `outputs/seepersea_demo.png` |
| 程序化前向 | `init_model` + `inference_multi_modality_detector` | 200 框、`has_nan=False`、最高分 0.146、峰值显存 ≈731 MB |
| `tools/test.py` 8 帧 | `--cfg-options ... format_only=True` | `Epoch(test) [8/8]`；类别分布 car 14 / bicycle 99 / pedestrian 7（证明标签映射生效）；1564 条检测 |
| `tools/test.py` 全段 | 指向 `seepersea_infos_train.pkl` | `Epoch(test) [438/438]  time: 0.1280`，无 error/traceback，438 个 `sample_token`、75599 条检测 |

**分数低（max ≈0.146）是域外数据的预期现象**（海面场景、单相机、稀疏点云、nuScenes 权重），与可行性无关。用户已明确不关注准确性。

#### 可复现命令

```bash
cd /data/coding/mmdetection3d

# 1) 转换（全 438 帧）
python tools/dataset_converters/seepersea2bevfusion.py \
  --src-root seepersea-data/barbados-20230223-240 \
  --out-dir data/seepersea --val-size 8

# 2) 官方 demo 可视化
python projects/BEVFusion/demo/multi_modality_demo.py \
  data/seepersea/samples/LIDAR_TOP/000000.bin \
  data/seepersea/samples/CAM_FRONT/ \
  data/seepersea/seepersea_infos_val.pkl \
  projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
  checkpoints/bevfusion_lidar-cam_5239b1af.pth \
  --cam-type all --score-thr 0.05 --out-dir outputs/seepersea_demo

# 3) 标准测试流水线（8 帧）
python tools/test.py \
  projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
  checkpoints/bevfusion_lidar-cam_5239b1af.pth \
  --work-dir work_dirs/seepersea \
  --cfg-options \
    test_dataloader.dataset.data_root=data/seepersea/ \
    test_dataloader.dataset.ann_file=seepersea_infos_val.pkl \
    test_dataloader.num_workers=0 test_dataloader.persistent_workers=False \
    test_evaluator.data_root=data/seepersea/ \
    test_evaluator.ann_file=data/seepersea/seepersea_infos_val.pkl \
    test_evaluator.format_only=True \
    test_evaluator.jsonfile_prefix=work_dirs/seepersea/results
# 全 438 帧：把两处 ann_file 改为 seepersea_infos_train.pkl，工作目录改 work_dirs/seepersea_full
```

---

## 4. 当前进度与遗留问题

**进度**：可行性验证闭环已完成（环境 → 算子编译 → 权重 → 数据转换 → 单帧 demo → 批量 `tools/test.py` → 全段 438 帧）。**尚未开始精度优化、训练、真实评估**。

| # | 遗留问题 | 影响 |
|---|---|---|
| L1 | **类别映射存在疑点**：本段 `kitti_label` 只出现 `Car`(369) 与 `Pedestrian`(4)，**没有 `Cyclist`**；且第 0 帧的 `buoy` 对应的是 `Pedestrian`，与用户描述的「buoy→Cyclist、other→Pedestrian」不一致 | 后续训练/评估的标签语义可能错，需与数据提供方确认 |
| L2 | 类别头仍是 nuScenes **10 类**，仅把 3 类投影到其中 3 个 id | 无法输出 ship/buoy/other 语义；改 3 类会导致 head 权重 mismatch（预期但需接受） |
| L3 | **无真实评估器**：`format_only=True` 只验流水线，没有 mAP/NDS | 无法量化，无法判断微调效果 |
| L4 | 只有 1 个相机（CAM_FRONT） | 未利用多目信息；BEVFusion 的 LSS 视角优势未发挥 |
| L5 | `ImageAug3D(final_dim=[256,704], resize_lim=[0.48,0.48])` 对 640×480 产生大量黑边补零 | 图像特征质量下降 |
| L6 | 约 14% 点云落在 nuScenes `[-54,-54,-5,54,54,3]` 之外被 `PointsRangeFilter` 丢弃；无 sweeps，`pad_empty_sweeps=True` 重复关键帧 9 次 | 几何与 nuScenes 假设不匹配 |
| L7 | `ego2global` = 单位阵，`motion_data/`（GPS/IMU）未使用 | 无法做多帧位姿对齐 |
| L8 | 66 帧 `annotations/*.json` 为空；点云仅 ~300 点/帧 | 训练样本质量/稀疏性问题 |
| L9 | 未做训练 smoke test；nuScenes mini 官方定量评估从未实际执行 | 训练链路与官方指标复现尚未验证 |
| L10 | 自定义配置未落地成文件，依赖一长串 `--cfg-options` | 复用成本高 |
| L11 | 产物多为未跟踪文件（`git status` 显示 `outputs/`、`seepersea-data/`、转换器脚本等为 `??`）；`data/` 被 gitignore | 交接/归档时注意 |

---

## 5. 下一步行动项（按优先级）

### P0（阻塞后续精度工作，必须先做）

1. **确认三类 ↔ KITTI 名称映射**：与数据提供方核对 `ship/buoy/other` → `Car/Cyclist/Pedestrian` 的真实对应（当前数据证据与用户描述冲突）。确认后同步修改 `seepersea2bevfusion.py` 的 `CATEGORY_TO_KITTI` / `KITTI_TO_NUSCENES_ID`。
2. **确定类别头方案**：改为 `bbox_head.num_classes=3` + `metainfo.classes=['ship','buoy','other']`，并在转换 pkl 中直接写 0/1/2；或保留 10 类做 baseline。改 3 类时接受 head 权重的 mismatch 日志，用 `--cfg-options` 或新配置覆盖。
3. **实现可量化的评估**：自定义 3 类 evaluator，或生成 nuScenes devkit 兼容的标注文件以复用 `NuScenesMetric`；去掉 `format_only=True`，产出真实 mAP/NDS 基线。

### P1（工程化，提升可用性）

4. **新增专用配置文件** `projects/BEVFusion/configs/bevfusion_lidar-cam_..._seepersea.py`，封装 `data_root`、`ann_file`、`data_prefix`、单相机 `images`、evaluator、`ImageAug3D` 参数（注意：mmengine 的 `_base_` 不会把父配置的变量注入子配置命名空间，需显式重写 dataloader dict，或继续用 `--cfg-options`）。
5. **按海面场景重设几何参数**：`point_cloud_range`、`voxel_size`、`sparse_shape`、`view_transform.xbound/ybound/zbound/dbound`、`bbox_coder.pc_range`，四者必须保持一致（`sparse_shape=[1440,1440,41]` 对应 `108/0.075`）；同时调整 `ImageAug3D.final_dim` / `resize_lim` 适配 640×480。
6. **训练 smoke test**：用 lidar-cam 融合配置（其 `train_pipeline` **没有 `ObjectSample`**，不需要 `nuscenes_dbinfos_train.pkl`），`--amp` + `batch_size=1` + `train_cfg.max_epochs=1` + `val_interval=2`，确认 loss 有限、可保存 ckpt。
7. **接入多相机**：若数据实际有多目，补齐 5 个 `CAM_*` 键、各自 `img_path` / `cam2img` / `lidar2cam`，并建立 `data_prefix` 对应目录；`N` 动态，无需改模型。

### P2（优化与收尾）

8. 用 `motion_data/` 构建 `ego2global` 与 sweeps 位姿对齐，替换关键帧重复。
9. 处理空标注帧与点云稀疏问题；确定默认 `--score-thr`（当前需用 0.05 才能看到框）。
10. 若需严格对照官方指标：下载 nuScenes mini（约 4 GB，需官网账号）跑 L2 定量评估，复现 NDS 71.4 / mAP 68.6。
11. 清理/归档临时产物（`/tmp` 已清理；`work_dirs/`、两个 `.png`、`bev_pool.egg-info/`、`build/` 可按需保留或删除）。

---

## 6. 上下文说明与注意事项

### 6.1 复用时必须避开的坑

- **不要用根目录 `demo/multi_modality_demo.py` 跑 BEVFusion**：它把 `--cam-type` 丢弃、不识别 `--score-thr`；用 `projects/BEVFusion/demo/multi_modality_demo.py`。
- **无 `spconv` 时权重会静默部分加载**：日志中只要出现 `size mismatch for pts_middle_encoder...`，结果就不可信。
- **`--score-thr 0.2` 跑自定义数据会零输出**：所有框被过滤，`out_file` 不生成；用 `0.05`。
- **`test_dataloader.num_workers=0` 必须同时设 `persistent_workers=False`**，否则 `ValueError: persistent_workers option needs num_workers > 0`。
- **无 GUI 环境不要加 `--show`**；`Failed to add LocalVisBackend ... save_dir`、`build_sampler/build_assigner deprecated`、`torch.meshgrid`、`CuDNN nvrtc` 等 warning 均可忽略。
- **离线环境**：`model.img_backbone.init_cfg.checkpoint` 是 GitHub URL，首次会下载到 `~/.cache/torch/hub/checkpoints/`；可改本地路径（如 `swint-nuimages-pretrained.pth`）。
- **权重文件名中的 8 位十六进制串不是md5/sha256 前缀**（实测 md5 `5873a41f…`、sha256 `e68c6c08…`），用文件大小 + `torch.load` 判断完整性。
- **`data/seepersea/samples/CAM_FRONT/*.png` 是 symlink**，指向 `seepersea-data/...`；移动或删除源目录会断链，归档时用 `--copy-images` 重跑转换。
- **`.bin` 是 `float32` 5 通道裸数据**（`np.tofile`），不能把原始 `.pcd` 直接喂给 `LoadPointsFromFile`。
- 所有命令依赖相对路径，**必须在 `/data/coding/mmdetection3d` 下执行**。
- 若要把 `tools/test.py` 用于其他目录布局：`NuScenesDataset.parse_data_info` 会把 `data_prefix[CAM_*]` / `data_prefix['pts']` 前缀拼到 pkl 里的相对 `img_path` / `lidar_path` 前，路径不匹配会报 `FileNotFoundError`。

### 6.2 关键证据文件索引

| 文件 | 说明 |
|---|---|
| `projects/BEVFusion/README.md` | 编译命令、demo 命令、训练/测试命令、权重 URL 与官方指标表 |
| `projects/BEVFusion/setup.py:19-30` | CUDA 算子编译开关与硬编码 `sm_70/75/80/86` |
| `projects/BEVFusion/configs/bevfusion_lidar_voxel0075_...py` | 10 类 `class_names`、`data_root=data/nuscenes/`、`data_prefix` 6 目、`db_sampler`、`test_dataloader=val_dataloader`、`test_evaluator=val_evaluator` |
| `projects/BEVFusion/configs/bevfusion_lidar-cam_...py` | 融合配置；`input_modality=use_camera=True`；重写的 `train/test_pipeline`（**无 ObjectSample**）；Swin `init_cfg` |
| `projects/BEVFusion/bevfusion/loading.py:133-158` | 从 `images` 推导 `cam2img` / `lidar2cam` / `lidar2img` |
| `projects/BEVFusion/bevfusion/bevfusion.py:141`、`depth_lss.py:80` | 视图数 `N` 动态，证明单相机可行 |
| `mmdet3d/datasets/det3d_dataset.py:200-260,263-323` | `parse_ann_info` / `parse_data_info` 的路径与标注字段要求 |
| `mmdet3d/datasets/nuscenes_dataset.py:58-63,133-191` | 标准 nuScenes 类序、`with_velocity=True`、`_filter_with_mask` |
| `mmdet3d/evaluation/metrics/nuscenes_metric.py:149-184,233-246` | `format_only` 提前返回；`eval_set_map` 的 `v1.0-mini → mini_val` |
| `mmdet3d/apis/inference.py:190-289` | `inference_multi_modality_detector` 的 `--cam-type all` 行为与 pkl `data_list` 要求 |
| `docs/en/get_started.md:80-91` | spconv 2.x 安装说明（官方只列 cu102/111/113/114，实测 cu118 可用） |
| `docs/en/user_guides/dataset_prepare.md:264` | 官方离线 `nuscenes_mini_infos_{train,val}.pkl` 链接 |
| `tools/create_data.py:360-373` | `--version v1.0-mini` 分支只生成 train/val pkl |
| `tools/dataset_converters/seepersea2bevfusion.py` | 本次新增的转换脚本（含类别映射、pkl 字段、CLI 参数） |

### 6.3 数据来源与用户口径

- 用户描述的三类映射：`ship→Car`、`buoy→Cyclist`、`other→Pedestrian`（已在脚本中按此固定，但数据证据有冲突，见 L1）。
- 用户明确：「暂时不用关注是否准确，只需验证可行性」→ 本项目至今**未做任何精度优化或指标计算**。
- 阶段一的完整流程说明曾以会话形式给出，并存在工作区根目录的 `验证BEVFusion可用性的完整流程_*.json` 导出文件中，可按需查阅。
