# SeaPersea 阶段 3–4 实施任务书

日期：2026-11-17。状态：待实施；本文固定实施方案，不代表代码或 CUDA 验证已完成。

适用任务：3. 建立正式数据集；4. 落地三类训练配置。
依据：同目录 `BEVFusion_SeaPersea_L6_Audit.md`，以及本地 MMDetection3D 1.4.0 / BEVFusion 源码。
旧 HANDOFF 中的 buoy/other 映射、点云丢弃比例和尺寸注释已被审计报告更正。

## 0. 接手 agent 先读这部分

### 0.1 本次要交付什么

交付一个有固定划分、原生三类标签、可追溯转换结果的数据集版本，以及能加载预训练特征提取器、完成前向/反向/保存/重载的三类训练配置。

按第 3 节完成数据验收后，再进入第 4 节。只执行文中规定的最小运行验证；完整微调、精度调参、AP/NDS evaluator、多帧对齐、多相机接入均不在本次范围。

### 0.2 双目录及修改边界

本机目录：

```text
data/coding/
  mmdetection3d-main/          # 上游源码，以下称 SOURCE_ROOT
  usv-fusion/                 # 适配仓库，以下称 ADAPTER_ROOT
    seepersea-dataset/        # 原始数据文件夹
```

远程源码目录可以叫 `mmdetection3d`，不能从目录名字推断安装位置。所有运行入口接收显式 `--mmdet3d-root`；数据、权重和输出位置也通过 CLI 指定并转换为绝对路径。不要硬编码 `/data/coding/mmdetection3d`。

所有新增或修改代码放在 ADAPTER_ROOT。不得修改 SOURCE_ROOT，不得在适配仓库新建顶层 `projects` 或 `mmdet3d` 包，以免遮蔽上游包。新增扩展包统一叫 `usv_fusion_ext`。

开始前记录两边 git 状态及源码版本；源码若无 `.git`，记录版本和所用 BEVFusion 源文件 SHA256。保留现有未提交文件。不要自动提交、推送、删除旧数据或覆盖旧推理产物。

目前本地没有报告所列的原始片段、完整权重和远程运行产物。先完成可独立实现的代码和合成样例测试；真实数据或 GPU 缺失时，把相应验收项标成“未执行：缺少……”，不得伪造通过。

### 0.3 固定决定，不再自行选方案

| 项目 | 本次唯一方案 |
|---|---|
| 真值源 | 只用 `annotations/*.json`；KITTI/2D 标注只作旁证 |
| 类别顺序 | `('ship', 'buoy', 'other')`，ID 严格为 `0,1,2` |
| KITTI 别名 | 仅历史兼容：ship→Car、buoy→Pedestrian、other→Cyclist；正式三类转换不经过别名映射 |
| 框定义 | pkl 使用重力中心 `[x,y,z,l,w,h,yaw]`；l=ann.width，w=ann.length，h=ann.height；yaw 原值、弧度 |
| 空标注 | 66 帧单独列出，排除训练/验证/测试；不补框、不插值、不当负样本 |
| 时间信息 | 单帧，不生成 sweeps，不生成伪全局位姿，不监督速度 |
| 点特征 | bin 为 float32 `[x,y,z,raw_intensity,0]`；模型输入时才固定 clip intensity 到 `[0,255]` |
| 相机 | 仅真实 `CAM_FRONT`，`num_views=1` |
| 检测范围 | `[-108,-108,-20,108,108,20]` |
| 体素 | `[0.15,0.15,1.0]`，稀疏形状 `[1440,1440,41]` |
| 权重 | 加载兼容的非检测头参数；整个 `bbox_head` 重新初始化；几何派生参数重新生成 |
| 训练采样 | 普通 DefaultSampler；不使用 CBGS、重复采样、ObjectSample |
| 本轮训练验收 | 单 GPU、batch=1、FP32、20 次迭代；不开展超参数搜索 |

这些是本任务书选定的工程基线，不是审计报告已经实测的训练配置，也不是最佳精度声明。不得因首轮分数低而更换划分、范围、类别或网络。

## 1. 为什么不能直接套审计报告的 A/B 配置

上游 `BEVFusionSparseEncoder.forward()` 把最后的深度维 D 折叠到通道，输出是 `[B,128*D,H,W]`。按现有卷积结构：

- sparse z=41：41→21→11→5→2，输出 256 通道。
- sparse z=101：101→51→26→12→5，输出 640 通道。
- 审计 B 档的 xy=672，使 LiDAR BEV 为 84×84；它建议的 LSS 为 336/2=168×168，两路也不一致。

因此，不执行“只扩大 z 范围、保持 0.2/0.4 的 z 体素、其余网络不变”。本任务固定 ±108 / z±20，并用 0.15/0.15/1.0 保持原来的网格和通道尺寸。审计报告给出的该范围覆盖率是全片段点约 99.81%、框中心 100%；剔除空帧或划分后须重新统计，不能沿用该百分比作为各子集结果。

代价：xy 和 z 的物理分辨率变粗，尤其 z=1m；本轮先得到完整目标覆盖和结构一致的可训练基线。不得为改善该代价自行增加通道、改变稀疏编码器或扩大张量。

## 2. 文件交付清单

以下路径均相对 ADAPTER_ROOT；只创建必要文件：

```text
configs/seepersea_dataset_v1.json               # 数据格式、类别、范围、划分策略
configs/bevfusion_seepersea_3class.py            # 相对上游融合配置的覆盖项
usv_fusion_ext/__init__.py                      # 导入注册项
usv_fusion_ext/dataset.py                       # SeaPerseaDataset
usv_fusion_ext/transforms.py                    # SeaPerseaClipIntensity
tools/dataset_converters/build_seepersea_v1.py   # 新建正式转换入口
tools/validate_seepersea_v1.py                  # 数据及 dataset/pipeline 校验
tools/prepare_seepersea_checkpoint.py           # 受控迁移权重
tools/run_seepersea.py                         # 解析双目录、生成配置、运行 smoke/train
tools/verify_seepersea_model.py                 # 张量、梯度、重载及预测检查
tests/test_seepersea_contract.py                # 合成数据合同测试
README_SeaPersea.md                            # 依赖、命令、产物、限制
```

允许把上述工具的共享路径/配置构建逻辑放到 `usv_fusion_ext/runtime.py`，不得复制多份。测试只覆盖数据语义及实际易错点，不写“测试常量等于自身”的测试。

保留旧 `seepersea2bevfusion.py` 作为 legacy 十类推理转换器，只修正已知类别映射及 `[l,w,h]` 注释，添加“非正式三类训练入口”的说明。不得静默把其原有 CLI 改成另一种输出格式。旧审计脚本的统计口径属于历史版本；正式三类产物由新 validator 检查，不能拿其硬编码的 nuScenes 标签检查当验收。

## 3. 建立正式数据集

### 3.1 固定划分，先分原始帧号，再剔除

版本名：`seepersea_3class_v1`。本次只支持审计过的 `barbados-20230223-240`；遇到其他片段不自动套用相同策略，报未配置。

| 原始帧号（两端均包含） | 身份 | 候选帧数 |
|---|---|---:|
| 000000–000279 | train | 280 |
| 000280–000299 | gap，隔离 | 20 |
| 000300–000359 | val | 60 |
| 000360–000379 | gap，隔离 | 20 |
| 000380–000437 | test | 58 |

执行规则：

1. 先扫描全部 438 个原始 frame_id，按上表赋值；不得先移除空帧再按序号切片。
2. 将缺标注帧排除，最终 train/val/test 数量由实际清单计算，不编造确切数字。
3. 排除原因主分类优先级：`unlabeled` 优先于 `temporal_gap`；另存原始候选区间，避免空帧与 gap 双重计数。
4. `infos_labeled_all.pkl` 保留全部 372 个有标注帧，包括 gap 内有标注帧，仅用于审计。它的 1804 框应按类别为 968/464/372。它不是训练入口。
5. `infos_smoke.pkl` 固定取最终 train 中按 frame_id 排序的前 8 个有效帧，明确是 train 子集，绝不称为独立验证集。
6. 每个 train/val/test 必须非空且三类各至少有一个框。若不满足，输出具体计数并停止数据版本验收；不得自动重排、随机抽帧或移动测试帧。
7. 此划分叫“单片段时间留出基线”。20 帧间隔用于减少紧邻帧重叠，不代表场景或目标轨迹独立；不以 annotation.id 检验跨集合跟踪独立性，该 ID 不是全局跟踪 ID。

### 3.2 扫描、异常和时间戳处理

- 确认帧号恰为 0–437，无缺文件、重复 stem；点云、图像、calib、annotation 文件都必须存在。
- 空文件或 `labels=[]` 归为 unlabeled。非空但 JSON 无效、缺少 `labels`、未知类别或字段缺失属于错误，不得伪装为空帧跳过。
- 归一化 `empty_label_files.txt` 的 `.json/.txt` stem 后核对空标注集合，预期 66 个唯一帧。清单不一致时报告差异，停止验收。
- 原始时间戳以秒保存。只允许修复已知第 0 帧哨兵：`t0=t1-(t2-t1)`，要求 t1/t2 有效且递增；保留原值和修复记录。其他缺失或不递增时间戳报错，不默认为 10Hz。
- calib 必须包含 P2、R0_rect、Tr_velo_to_cam，维度正确、值有限、旋转可逆且满足刚体变换容差。禁止缺标定时补单位阵。
- 验证当前片段 P2 第四列绝对值≤1e-6；不符合则报“不支持该投影形式”，不得直接丢弃第四列。
- 验证图片 640×480、点特征有限、PCD 为支持的 ASCII x/y/z/intensity 格式，非空，点数与头部声明一致；异常一律带 frame_id 报错。
- 框中心/尺寸/yaw 全部有限，尺寸严格大于零，pitch/roll 绝对值≤1e-6；当前任务不擅自修正异常大框，不因尺寸离群自动删 ship。

### 3.3 正式产物与路径协议

```text
DATA_ROOT/
  samples/LIDAR_TOP/barbados-20230223-240/000000.bin
  samples/CAM_FRONT/barbados-20230223-240/000000.png
  infos_labeled_all.pkl
  infos_train.pkl
  infos_val.pkl
  infos_test.pkl
  infos_smoke.pkl
  splits/{train,val,test,smoke,unlabeled,temporal_gap}.txt
  manifest.json
  conversion_report.json
```

转换全部 438 帧的传感器文件，正式 pkl 仅纳入规定的有标注集合。图片默认复制，禁止默认 symlink。生成目录必须全新或为空；非空则报错，要求调用者提供新版本路径，不自动覆盖或清理。

`splits/*.txt` 每行一个稳定 token，字典序排序。token 格式 `barbados-20230223-240:000000`；sample_idx 本版为原始整数 frame_id，在各子集不重新编号。

pkl 的 `lidar_path` / `img_path` 为 `barbados-20230223-240/000000.bin|png`，只保存相对路径。`data_prefix` 固定：

```python
dict(pts='samples/LIDAR_TOP', CAM_FRONT='samples/CAM_FRONT')
```

不要把 `samples/LIDAR_TOP` 重复写进 pkl path；不要保存远程绝对路径。

### 3.4 固定数据合同

每个 pkl 顶层为 `dict(metainfo=..., data_list=[...])`，元数据至少含：

```python
dict(
    dataset='seepersea', version='seepersea_3class_v1',
    classes=('ship', 'buoy', 'other'),
    categories={'ship': 0, 'buoy': 1, 'other': 2},
    box_origin='gravity_center', box_dim_order='lwh',
    point_features=('x', 'y', 'z', 'raw_intensity', 'time_lag'),
    has_global_pose=False, has_velocity=False,
)
```

每个 sample：`sample_idx/token/scene_id/frame_id/timestamp/lidar_points/images/instances`。其中：

- `lidar_points=dict(num_pts_feats=5, lidar_path=...)`。
- bin 为连续 float32 N×5，xyz 保持原坐标，强度保存原始数值，第五列严格为零。保存时不裁剪坐标、强度或重采样点。
- `images` 只含 CAM_FRONT：`img_path`、3×3 `cam2img=P2[:3,:3]`、4×4 `lidar2cam=R0_rect_4x4 @ Tr_4x4`、timestamp。
- 不写伪 `ego2global`、`cam2ego`、`lidar_sweeps`。当前单帧加载器不依赖这些字段。
- `instances` 每项只要求 `bbox_3d`、`bbox_label_3d`、`num_lidar_pts`；不伪造 2D bbox、速度、属性标签。
- `bbox_3d=[loc.x,loc.y,loc.z,dim.width,dim.length,dim.height,yaw]`，数值顺序不交换，不预先把 z 减 h/2，不修改 yaw 符号。
- `num_lidar_pts` 使用原始完整单帧 xyz 计算，复用审计的旋转框逻辑：先中心平移，再旋转 -yaw，判断三个轴的绝对值≤半尺寸+1e-9。点同时位于多个框可以被分别计数。不是裁剪后点数，也不是虚设 1。
- 原始 annotation.id 不用作实例唯一键。若报告需要逐框索引，使用 sample token 加原始 labels 列表下标。

`manifest.json` 必须列出全部帧的 token、候选区间、最终身份、空标注状态、时间戳修复、原始文件 SHA256、转换配置摘要及脚本版本。`conversion_report.json` 必须给出各集合帧数/类别框数/点数、范围保留率、框内点数分布、排除原因计数；范围对点和框中心都用下界包含、上界不包含。

### 3.5 SeaPerseaDataset 的实现约束

直接继承 `Det3DDataset`，不要继承 NuScenesDataset。注册名为 `SeaPerseaDataset`。

1. METAINFO.classes 固定三类原生顺序，palette 固定三色即可。
2. `parse_ann_info` 调用父类以获得标签映射和数组；无标注时显式构造 float32 `(0,7)` 框及 int64 `(0,)` 标签。
3. 非空/空框都包装成 `LiDARInstance3DBoxes(..., box_dim=7, origin=(0.5,0.5,0.5))`；由框架转到底面中心，不能二次转换 z。
4. `default_cam_key=None`，继承父类路径解析。当前父类设置 CAM_FRONT 后可能直接计算 3×3 cam2img @ 4×4 lidar2cam，避免走该分支；投影矩阵由 BEVLoadMultiViewImageFromFiles 构造。
5. 不加入 nuScenes 的类别/速度/点数有效性过滤；转换阶段已审计数据。`filter_empty_gt=False`，避免运行时随机换帧掩盖问题；真实三类集合本应无空标注，validator 必须断言这一点。
6. 数据集加载时校验数据版本和三类元数据，拒绝 legacy nuScenes ID pkl。

### 3.6 阶段 3 验收门槛

- 原始 438 帧、unlabeled 66、labeled_all 372；labeled_all 共 1804 框，三类 968/464/372；不能把 train+val+test 的和误要求为 372，因为 gap 中有标注帧也被排除。
- train/val/test token 两两交集为空；与 unlabeled/gap 均不相交；smoke 是 train 前 8 帧；所有 438 帧恰有一种最终身份。
- 当前片段所有框 `num_lidar_pts>=1`；逐框计数与审计的 float64 原始几何算法一致。框参数往返误差≤1e-5，ID 与原始类别逐项一致。
- 重新定位 DATA_ROOT 后仍能读到所有图片和 bin；两次独立转换的列表、框和 bin 内容一致（运行时间字段不比较）。
- 对全部正式样本构建 dataset 并检查标签在 0–2、框维度为 7、重力中心还原正确、无路径二次拼接。
- 合成测试必须覆盖：旋转非正方形框的点数、重力中心→底面中心、类别顺序、空文件与损坏 JSON 区分、划分边界、已知时间戳修复、非法标定拒绝。

## 4. 落地三类训练配置

### 4.1 配置构建和启动方式

复用上游融合配置：

```text
SOURCE_ROOT/projects/BEVFusion/configs/
bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py
```

为适应本地/远程目录差异，不在新配置里写跨目录 `_base_`。统一运行工具先把 SOURCE_ROOT、ADAPTER_ROOT 加入 Python import 路径，校验 `mmdet3d.__file__` 确实来自指定源码，然后：

1. `Config.fromfile()` 读取上游融合配置。
2. 读取新配置覆盖项，用 MMEngine 配置合并能力覆盖；本节要求整体替换的 dict 必须真正清掉父项，不能遗留 CBGS、velocity 或 evaluator。
3. 设置绝对 data_root/work_dir/load_from，注册 mmdet3d 与 `usv_fusion_ext` 模块。
4. 输出完整 `resolved_config.py` 和来源/版本/哈希，然后构建模型或 Runner；所有验证与训练使用同一个配置构建函数。

`custom_imports` 包含 `projects.BEVFusion.bevfusion` 和 `usv_fusion_ext`，`allow_failed_imports=False`。所有模式的最终配置都设置 `model.img_backbone.init_cfg=None`，图像预训练参数只由受控 checkpoint 提供。实际 CUDA 运行以移交记录的 Linux 环境为基线；不把 Windows 原生 CUDA 移植纳入本任务。

### 4.2 几何参数必须整组设置

| 字段 | 固定值 |
|---|---|
| point_cloud_range / voxelize_cfg.point_cloud_range | `[-108,-108,-20,108,108,20]` |
| voxel_size / voxelize_cfg.voxel_size | `[0.15,0.15,1.0]` |
| 实际体素网格 xyz | `[1440,1440,40]` |
| pts_middle_encoder.sparse_shape | `[1440,1440,41]`，z 的 +1 沿用上游布局 |
| bbox_head.train_cfg.grid_size | `[1440,1440,40]` |
| bbox_head.test_cfg.grid_size | `[1440,1440,40]` |
| bbox_head.train_cfg.point_cloud_range | 同 point_cloud_range |
| bbox_head.train_cfg.voxel_size | `[0.15,0.15,1.0]` |
| bbox_head.test_cfg.voxel_size | `[0.15,0.15]` |
| bbox_head.test_cfg.pc_range | `[-108,-108]` |
| bbox_head.bbox_coder.pc_range | `[-108,-108]` |
| bbox_head.bbox_coder.voxel_size | `[0.15,0.15]` |
| train/test/coder 的 out_size_factor | `8` |
| view_transform.xbound / ybound | 各 `[-108,108,0.6]` |
| view_transform.zbound | `[-20,20,40]`，1 个 z bin |
| view_transform.dbound | `[1,60,0.5]`，118 个 depth bins |
| view_transform.downsample | `2` |
| bbox_coder.post_center_range | `[-108,-108,-40,108,108,40]` |

不扩大 dbound 到 80：本轮保留 depth 分支维度以复用权重。±108 的框中心由 LiDAR 分支覆盖，不表示单相机具有该完整视野/深度覆盖。

`post_center_range` 的 z 与训练范围不同是有意的：上游 coder 在恢复底面中心之后才执行该过滤，而审计范围按重力中心定义。它只是宽松的解码保护。后续 evaluator/标准导出必须按 `boxes.gravity_center` 再执行正式 3D ROI：`min <= center < max`，不能用底面 z 假装统一口径。本轮预测验证同时记录 ROI 前/后的数量，不实现 AP。

静态及实测均检查：LiDAR BEV `[B,256,180,180]`，camera BEV `[B,80,180,180]`，fusion `[B,256,180,180]`，pts_neck 输出供 head 使用的通道为 512。禁止靠修改 fusion_layer.in_channels 掩盖 z 网格错误。

### 4.3 图像与 pipeline 固定方案

ImageAug3D（train/test 相同几何）：

```python
final_dim=[256,352]
resize_lim=[0.55,0.55]
bot_pct_lim=[0.0,0.0]
rot_lim=[0.0,0.0]
rand_flip=False
# train is_train=True；test is_train=False
```

640×480 缩成 352×264，裁掉顶部 8 像素，得到 352×256；无补黑边。同步 `view_transform.image_size=[256,352]`、`feature_size=[32,44]`。保留上游 RGB 加载及 mean/std/bgr_to_rgb=False，验证不发生二次颜色翻转。

训练 pipeline 按以下顺序整体替换：

1. `BEVLoadMultiViewImageFromFiles(to_float32=True,color_type='color',num_views=1,num_ref_frames=0)`。
2. `LoadPointsFromFile(coord_type='LIDAR',load_dim=5,use_dim=5)`。
3. `SeaPerseaClipIntensity`：只修改 points.tensor[:,3]，clip 到 [0,255]，其余四列不变，不改磁盘 bin。此为固定兼容基线，log/分位数归一化留待后续对照实验。
4. `LoadAnnotations3D(with_bbox_3d=True,with_label_3d=True,with_attr_label=False)`。
5. 上述 ImageAug3D。
6. `BEVFusionGlobalRotScaleTrans(scale_ratio_range=[1,1],rot_range=[0,0],translation_std=0)`，生成一致的 lidar_aug_matrix。本轮不启用随机几何增强。
7. `PointsRangeFilter` 使用新范围。
8. `ObjectNameFilter(classes=['ship','buoy','other'])`。
9. `PointShuffle`。
10. `Pack3DDetInputs(keys=['points','img','gt_bboxes_3d','gt_labels_3d'])`。

不使用 `ObjectRangeFilter`：当前数据所有 GT 重力中心应位于固定新范围内，由 validator 逐框断言；避免仅 xy 的旧过滤器掩盖 GT 不一致。若断言不成立则停止验收，不擅自扩范围或删框。后续新增数据须单独制定统一的 GT/预测 ROI 过滤策略。

Pack 的 meta_keys 继承上游融合 train_pipeline 列表并去重，追加 `token,scene_id,frame_id,timestamp,num_views`；必须保留所有标定、img_aug_matrix、lidar_aug_matrix、box_type_3d、sample_idx、路径、num_pts_feats 字段。

推理 pipeline 使用步骤 1/2/3/5/7 和 Pack；Pack keys 为 points/img，其 meta_keys 使用相同清单，允许训练专属 key 不存在。GT 校验直接使用 dataset.get_ann_info，不把评估 GT 从随机 train pipeline 取出。

所有 pipeline 删除 LoadPointsFromMultiSweeps、ObjectSample、GridMask、BEVFusionRandomFlip3D；删除 db_sampler 和相关 hook。不得用 repeats 伪造时序，也不得伪造其他相机。

### 4.4 原生三类检测头和无速度监督

复用上游 `TransFusionHead`，本任务不需要复制/改写检测头源码：

```python
num_classes=3
num_proposals=200
nms_kernel_size=3
train_cfg.dataset='SeaPersea'
test_cfg.dataset='SeaPersea'
test_cfg.nms_type=None
common_heads=dict(center=[2,2], height=[1,2], dim=[3,2], rot=[2,2])
bbox_coder.code_size=8
train_cfg.code_weights=[1.0]*8
```

common_heads 必须整体替换，确认 resolved config 没有残留 vel。模型 GT 为 7D，回归编码为 8D（yaw→sin/cos）。不设置假的 velocity=[0,0]，不设置 with_velocity=True，也不借用 Waymo 的类别分支。

`test_cfg.dataset='SeaPersea'` 用于绕过 nuScenes 热图索引 8/9 的专用逻辑；nms_type=None 使预测不访问 nuScenes/Waymo 的固定 tasks。二者必须一起固定。现有 loss/encode/decode 原生支持不含 vel。

其余 Hungarian assigner、loss 权重、gaussian_overlap=0.1、min_radius=2 先继承上游；本轮不擅自添加类别尺寸先验或 AP 距离阈值。该检测头不需要为 ship/buoy/other 新造 anchor 尺寸表。

### 4.5 dataloader 与训练运行参数

整体替换训练 dataset dict，不保留父配置的 CBGSDataset 包装：

```python
type='SeaPerseaDataset'
ann_file='infos_train.pkl'
metainfo=dict(classes=('ship','buoy','other'))
modality=dict(use_lidar=True,use_camera=True)
box_type_3d='LiDAR'
default_cam_key=None
filter_empty_gt=False
test_mode=False
```

batch_size=1，num_workers=0，persistent_workers=False，DefaultSampler(shuffle=True)，drop_last=False。固定 seed=2026；记录环境，不承诺 CUDA 位级确定性。

基础配置：AdamW(lr=1e-4,weight_decay=0.01)，clip_grad(max_norm=35,norm_type=2)，OptimWrapper（FP32），auto_scale_lr.enable=False，param_scheduler=[]，max_epochs=6。不冻结分支。训练轮数与 LR 为初始工程基线，完整 6 epoch 不在本轮执行范围。

本阶段没有真实 evaluator，因此有效配置中统一设 `val_dataloader=val_cfg=val_evaluator=None`、`test_dataloader=test_cfg=test_evaluator=None`；注意这些是顶层 Runner 字段，不能清掉 model.bbox_head.test_cfg。另由验证工具直接构建 val/test dataset 和 inference pipeline。不得保留 NuScenesMetric，不得用 format_only 冒充精度验证。

smoke 模式使用 ann_file=infos_smoke.pkl，并且**整体替换**顶层 train_cfg 为：

```python
dict(type='IterBasedTrainLoop', max_iters=20, val_interval=1000000)
```

必须删除父配置遗留的 by_epoch/max_epochs，不能仅做字段追加。同步 `default_hooks.checkpoint.by_epoch=False`、`interval=20`、`save_last=True`；logger.interval=1；log_processor.by_epoch=False。所有模型/几何参数与基础配置一致。验证恢复到 iter21 时也整体替换 loop，只将 max_iters 改为 21。

### 4.6 权重迁移必须受控，不能只容忍 mismatch

源权重为已经验证过的官方融合权重 `bevfusion_lidar-cam_5239b1af.pth`。禁止直接在三类模型里加载该文件然后忽略所有警告。

`prepare_seepersea_checkpoint.py` 按以下算法生成一个完整目标模型初始化 checkpoint：

1. 根据最终配置构建目标模型，固定 seed=2026；`img_backbone.init_cfg=None`，防止模型构造时联网下载或覆盖迁移参数。加载前先确认 spconv 2.x 与自有 CUDA 算子可用。
2. 读取源 checkpoint 的 state_dict。若含 module. 前缀，仅允许一致移除并写入报告，不模糊匹配键名。
3. 以目标新初始化 state_dict 为底稿；**整个 bbox_head.* 保留目标初始化值**，不从十类头挑三类复制。
4. **view_transform.dx、bx、nx、frustum 始终保留目标配置生成值**，即使 tensor shape 一样也不加载源值。这些是注册在 state_dict 中的几何参数，加载旧 dx/bx 会悄悄覆盖新范围。
5. 对其余全部目标键，要求源中存在且 shape 完全相同，逐一复制；源多出、目标缺失或形状不同的键，若不属于上述允许集合，立即失败。禁止仅过滤 shape 后静默忽略其他键，尤其 pts_middle_encoder。
6. 产出完整目标 state_dict、三类 dataset_meta、配置/源权重哈希、加载清单 `checkpoint_transfer.json`；不携带源 optimizer、epoch、iter、scheduler 或 resume 状态。
7. 用 `strict=True` 加载到另一个同配置模型，要求 missing/unexpected 为零；核对已迁移的张量与源逐项相等、几何参数与新配置一致。

最终初始化权重命名 `bevfusion_seepersea_3class_init.pth`。Runner.load_from 指向该文件，resume=False。正式续训才允许使用 Runner 保存的 checkpoint 并恢复 optimizer/iter。

### 4.7 阶段 4 验收门槛

按顺序执行，任一步失败先修实现，不改本任务书的实验定义：

1. **配置检查**：resolved_config 无 CBGS、旧范围、NuScenesMetric、sweeps、vel；三个类别一致，数据路径全部可解析，网络导入来自指定源码。
2. **数据流水线**：完整遍历 smoke/train/val/test 的 pipeline，检查点数不因重复 sweeps 膨胀、所有值有限；模型输入 intensity 在 [0,255]；图片为单视图 256×352，变换矩阵正确。
3. **前向尺寸**：记录第 4.2 节各分支尺寸；view_transform.nx=[360,360,1]，frustum=[118,32,44,3]；GT N×7，输出标签范围 0–2、输出框 N×7、无 vel。
4. **反向与更新**：至少一次真实 loss.backward，loss/梯度有限；图像分支、LiDAR 分支、融合层、检测头各选一个参与前向的可训练参数，确认梯度非空且该分支至少一个参数在 optimizer.step 后改变。不得将 matched_ious 计入总 loss。
5. **20 iter smoke**：连续完成、无 NaN/Inf，无未经许可的参数缺失；保存 checkpoint，记录峰值显存和实际迭代时间。本轮不把 loss 必须单调下降或分数达到某阈值作为验收。
6. **重载与恢复**：同一已处理输入、eval 模式，对保存/重载前后的预测做合理浮点容差比较；额外在独立 work_dir 从 iter20 恢复到 iter21，确认 optimizer/iter 连续，不能只重新加载参数当作 resume。
7. **val 推理接口**：用 iter20 checkpoint 对完整 val 做无梯度预测，仅验证每个 token 有一条结果记录、标签/框维度/数值合法；允许零检测，不以低分调整阈值。以 score threshold=0 导出原始结果及按重力中心 ROI 筛选的结果；不要调用 NuScenesMetric 或报告 mAP/NDS。test 留作后续最终评估，不用来调参。

记录到 `model_verification.json`、`checkpoint_transfer.json`、训练日志和 `resolved_config.py`。只凭 exit code=0 不算完成。

## 5. 统一 CLI 合同和执行顺序

接手 agent 按此设计 CLI。下列路径变量由运行者设置为真实绝对路径，示例展示参数合同，不表示本地已有数据/权重：

```text
python ADAPTER_ROOT/tools/dataset_converters/build_seepersea_v1.py
  --src-root RAW_CLIP_ROOT
  --out-dir DATA_ROOT
  --spec ADAPTER_ROOT/configs/seepersea_dataset_v1.json

python ADAPTER_ROOT/tools/validate_seepersea_v1.py
  --mmdet3d-root SOURCE_ROOT --src-root RAW_CLIP_ROOT
  --data-root DATA_ROOT --out-dir WORK_ROOT/data_validation

python ADAPTER_ROOT/tools/prepare_seepersea_checkpoint.py
  --mmdet3d-root SOURCE_ROOT --data-root DATA_ROOT
  --source-checkpoint OFFICIAL_CHECKPOINT
  --out-dir WORK_ROOT/init

python ADAPTER_ROOT/tools/run_seepersea.py
  --mmdet3d-root SOURCE_ROOT --data-root DATA_ROOT
  --checkpoint WORK_ROOT/init/bevfusion_seepersea_3class_init.pth
  --work-dir WORK_ROOT/smoke --mode smoke

python ADAPTER_ROOT/tools/verify_seepersea_model.py
  --mmdet3d-root SOURCE_ROOT --data-root DATA_ROOT
  --checkpoint WORK_ROOT/smoke/iter_20.pth
  --work-dir WORK_ROOT/model_validation
```

这些示例分行只是展示参数；README 必须另外给出可直接运行的 Linux bash 命令及本机 PowerShell 的纯数据转换/校验命令，不混用续行符。run 工具可提供 train 模式，但本任务执行到 smoke 和上述验证即止。

## 6. 禁止自行扩展与最终回报格式

禁止：随机重划数据、伪造负样本、KITTI 替代完整 GT、插值补标、跨片段假定 track ID 唯一、伪多目/伪 sweeps、为跑通静默跳异常、直接改上游代码、未列明的权重忽略、复制 nuScenes 指标作为本项目精度。

若数据事实与报告计数不符、三类在某集合缺失、受控权重加载失败、或固定几何实测张量不符：保留已完成工作，输出实际证据和具体阻塞点，不自行变更范围/划分/模型后宣称完成。缺资源时继续完成不依赖资源的代码与测试，再明确列出待执行项。

最终回报必须包含：

1. 修改文件及用途；源码目录是否保持不变。
2. train/val/test/smoke/unlabeled/gap 的实际数量及类别计数。
3. 转换、配置构建、权重迁移、模型验证的完整复现命令。
4. 实测张量尺寸、loss/梯度/参数更新、checkpoint 重载与 resume 结果。
5. 验收表：逐项 PASS / FAIL / NOT_RUN，附对应日志或 JSON 路径。
6. 本次没有执行完整训练或真实精度评价；后续任务是小样本过拟合验证、正式 evaluator 和完整微调。

## 7. 关键源码索引（SOURCE_ROOT 内）

- `projects/BEVFusion/bevfusion/sparse_encoder.py`：H/W/D 顺序、C×D 折叠。
- `projects/BEVFusion/bevfusion/depth_lss.py`：dx/bx/nx/frustum 注册参数、BEV 网格、depth bin 数。
- `projects/BEVFusion/bevfusion/transfusion_head.py`：nuScenes 类别索引 8/9、nms tasks、可选 vel、8D regression loss。
- `projects/BEVFusion/bevfusion/utils.py`：TransFusionBBoxCoder encode/decode、底面中心 post_center_range。
- `projects/BEVFusion/bevfusion/transforms_3d.py`：图像 resize/crop 和增强矩阵。
- `projects/BEVFusion/bevfusion/loading.py`：从 images 构造单相机标定矩阵。
- `mmdet3d/datasets/det3d_dataset.py`：类别映射、路径拼接、annotation 解析。
- `mmdet3d/datasets/dataset_wrappers.py`：CBGS 零频类别除零风险。

