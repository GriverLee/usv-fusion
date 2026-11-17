# SeaPersea 阶段 3–4 实施说明

实施依据：`BEVFusion_SeaPersea_STAGE3_4_SPEC.md`。本文件记录依赖、命令、产物和本轮验收结果。所有路径通过 CLI 显式传入并转成绝对路径；不修改 `SOURCE_ROOT`，不覆盖旧数据或旧推理产物。

## 1. 环境与边界

- `SOURCE_ROOT=/data/coding/mmdetection3d`：MMDetection3D 1.4.0，git `90b3b0d94d40ec778b00189d541cb08fb1e2b5c7`，分支 `dev-1.x`（ahead 2），工作树未修改。
- `ADAPTER_ROOT=/data/coding/usv-fusion`：git `45c837a5898f8629154da5050a0921f1a8720c0d`，仅新增/修改本仓库文件，未提交。
- Python 3.10.14，torch 2.0.1+cu118，CUDA 可用，Quadro RTX 6000 24 GB，mmcv 2.1.0，mmengine 0.10.5，mmdet 3.3.0，spconv 2.3.8。BEVFusion 自有 CUDA 算子已编译。
- 源权重：`/data/coding/mmdetection3d/checkpoints/bevfusion_lidar-cam_5239b1af.pth`。
- 原始数据：`/data/coding/usv-fusion/seepersea-dataset/barbados-20230223-240`。
- 关键 BEVFusion 源文件 SHA256 与上游版本记录在 `config_provenance.json`（由 `build_seepersea_config` 写入每个 work_dir）。

类别固定为 `ship/buoy/other`（ID 0/1/2）；划分、范围和几何参数完全取自任务书，不在本实现中重新选择。

## 2. 新增/修改文件

| 路径 | 用途 |
|---|---|
| `configs/seepersea_dataset_v1.json` | 数据版本、类别、范围、438 帧划分策略与路径协议 |
| `configs/bevfusion_seepersea_3class.py` | 阶段 4 配置覆盖项，无跨目录 `_base_` |
| `usv_fusion_ext/__init__.py` | 模块注册入口；纯数据转换时延迟导入模型模块 |
| `usv_fusion_ext/runtime.py` | 共享数据解析/几何、配置构建、spconv 安全加载、监控 hook |
| `usv_fusion_ext/dataset.py` | `SeaPerseaDataset`，继承 `Det3DDataset`，原生三类 |
| `usv_fusion_ext/transforms.py` | `SeaPerseaClipIntensity`，仅内存裁剪强度 |
| `tools/dataset_converters/build_seepersea_v1.py` | 正式数据转换入口 |
| `tools/validate_seepersea_v1.py` | 数据/dataset 验收与两次转换一致性比较 |
| `tools/prepare_seepersea_checkpoint.py` | 受控权重迁移 |
| `tools/run_seepersea.py` | smoke/train/resume 统一入口 |
| `tools/verify_seepersea_model.py` | 模型/训练/推理接口验收 |
| `tests/test_seepersea_contract.py` | 合成数据合同测试 |
| `tools/dataset_converters/seepersea2bevfusion.py` | 旧十类推理转换器：修正 KITTI 别名、框顺序注释，标注非正式训练入口 |
| `README_SeaPersea.md` | 本文件 |

## 3. 可复现命令

下列 Linux 命令均为单行，可在任意目录执行（路径为真实绝对路径）。

```bash
python /data/coding/usv-fusion/tools/dataset_converters/build_seepersea_v1.py --src-root /data/coding/usv-fusion/seepersea-dataset/barbados-20230223-240 --out-dir /data/coding/usv-fusion/work_stage34/data/v1_run3 --spec /data/coding/usv-fusion/configs/seepersea_dataset_v1.json
python /data/coding/usv-fusion/tools/dataset_converters/build_seepersea_v1.py --src-root /data/coding/usv-fusion/seepersea-dataset/barbados-20230223-240 --out-dir /data/coding/usv-fusion/work_stage34/data/v1_run4 --spec /data/coding/usv-fusion/configs/seepersea_dataset_v1.json
python /data/coding/usv-fusion/tools/validate_seepersea_v1.py --mmdet3d-root /data/coding/mmdetection3d --src-root /data/coding/usv-fusion/seepersea-dataset/barbados-20230223-240 --data-root /data/coding/usv-fusion/work_stage34/data/v1_run3 --out-dir /data/coding/usv-fusion/work_stage34/data_validation --reference-data-root /data/coding/usv-fusion/work_stage34/data/v1_run4
python -m unittest discover -s /data/coding/usv-fusion/tests -v
python /data/coding/usv-fusion/tools/prepare_seepersea_checkpoint.py --mmdet3d-root /data/coding/mmdetection3d --data-root /data/coding/usv-fusion/work_stage34/data/v1_run3 --source-checkpoint /data/coding/mmdetection3d/checkpoints/bevfusion_lidar-cam_5239b1af.pth --out-dir /data/coding/usv-fusion/work_stage34/init
python /data/coding/usv-fusion/tools/run_seepersea.py --mmdet3d-root /data/coding/mmdetection3d --data-root /data/coding/usv-fusion/work_stage34/data/v1_run3 --checkpoint /data/coding/usv-fusion/work_stage34/init/bevfusion_seepersea_3class_init.pth --work-dir /data/coding/usv-fusion/work_stage34/smoke_v2 --mode smoke
python /data/coding/usv-fusion/tools/verify_seepersea_model.py --mmdet3d-root /data/coding/mmdetection3d --data-root /data/coding/usv-fusion/work_stage34/data/v1_run3 --checkpoint /data/coding/usv-fusion/work_stage34/smoke_v2/iter_20.pth --work-dir /data/coding/usv-fusion/work_stage34/model_validation
```

Windows PowerShell（仅纯数据转换/校验，不含 CUDA/GPU 训练；单行）：

```powershell
python C:\data\coding\usv-fusion\tools\dataset_converters\build_seepersea_v1.py --src-root C:\data\coding\usv-fusion\seepersea-dataset\barbados-20230223-240 --out-dir C:\data\coding\usv-fusion\work_stage34\data\ps_v1 --spec C:\data\coding\usv-fusion\configs\seepersea_dataset_v1.json
python C:\data\coding\usv-fusion\tools\validate_seepersea_v1.py --mmdet3d-root C:\data\coding\mmdetection3d --src-root C:\data\coding\usv-fusion\seepersea-dataset\barbados-20230223-240 --data-root C:\data\coding\usv-fusion\work_stage34\data\ps_v1 --out-dir C:\data\coding\usv-fusion\work_stage34\ps_validation
```

注意：转换输出目录必须不存在或为空；本轮已使用 `v1_run3`/`v1_run4`。早期失败重试留下的 `data/seepersea_3class_v1`、`v1_run1`、`v1_run2` 不是有效产物，未删除、未作为验收输入。

## 4. 本轮实测结果

### 数据（阶段 3）

| 集合 | 帧数 | ship | buoy | other | 框数 |
|---|---:|---:|---:|---:|---:|
| train | 232 | 505 | 324 | 232 | 1061 |
| val | 54 | 162 | 54 | 54 | 270 |
| test | 54 | 205 | 54 | 54 | 313 |
| smoke（train 前 8） | 8 | 16 | 100 | 8 | 124 |
| unlabeled | 66 | - | - | - | 0 |
| temporal_gap（gap 内有标注帧） | 32 | - | - | - | 160 |
| labeled_all（审计用） | 372 | 968 | 464 | 372 | 1804 |

所有 438 帧恰有一种身份；集合两两不相交；范围 `[-108,-108,-20,108,108,20]` 保留点 139129/139391（99.812%），框中心 1804/1804（100%）；1804 框点数均值 50.0、中位 18、最小 1、最大 513。证据：`work_stage34/data/v1_run3/conversion_report.json`、`work_stage34/data_validation/validation_report.json`。

### 模型（阶段 4）

- LiDAR BEV `[1,256,180,180]`，camera BEV `[1,80,180,180]`，fusion `[1,256,180,180]`，pts_neck `[1,512,180,180]`；`view_transform.nx=[360,360,1]`，`frustum=[118,32,44,3]`。
- GT 7 维、标签 0–2；预测框 `[200,7]`、标签 0–2、无 vel。
- 一次真实 `loss.backward()`：总 loss 有限；img/lidar/fusion/head 分支分别有 175/63/3/64 个参数获得非空有限梯度，`optimizer.step()` 后各分支均有参数改变；`matched_ious` 未计入总 loss。
- 20 iter smoke：20/20 完成，无 NaN/Inf，保存 `iter_20.pth`；峰值显存约 2421.9 MB；日志 `work_stage34/smoke_v2.log`，统计 `work_stage34/smoke_v2/smoke_summary.json`。
- 重载：同一输入预测框/分数浮点容差比较通过，标签完全一致。
- resume：独立 work_dir 从 iter20 恢复到 iter21，checkpoint meta `iter=21`，optimizer state 非空。
- val 推理接口：54/54 token 各一条记录，框/标签/分数合法；score threshold=0 的原始结果 10800 条，按重力中心 ROI 结果 10800 条。未计算 mAP/NDS。

证据：`work_stage34/model_validation/model_verification.json`、`checkpoint_transfer.json`、`resolved_config.py`、`val_predictions_raw.json`、`val_predictions_roi.json`、`resume_21/resume.log`。

spconv 2.3.8 与 torch 2.0 的 `Module.load_state_dict` 对自有 state_dict 会误报 sparse 卷积形状不匹配；工具采用显式 key/shape 校验后按注册参数名复制，并在 `checkpoint_transfer.json` 中记录 `strict_reload.method`。权重迁移边界未改变：整个 `bbox_head.*` 与 `view_transform.dx/bx/nx/frustum` 保留目标初始化，其余键要求源中存在且形状一致。

## 5. 验收表

| 项目 | 结果 | 证据 |
|---|---|---|
| 原始 438 帧、unlabeled 66、labeled_all 372/1804（968/464/372） | PASS | `conversion_report.json`, `validation_report.json` |
| 划分不相交、smoke=train 前 8、438 帧身份唯一 | PASS | `validation_report.json` |
| 框几何往返、逐框点数、GT 中心范围内 | PASS | `validation_report.json` |
| dataset 全部样本、版本/类别校验、重定位路径 | PASS | `validation_report.json` |
| 两次独立转换一致 | PASS | `validation_report.json` |
| 合成合同测试（8 项） | PASS | `python -m unittest discover -s tests -v` |
| 受控权重迁移与严格 key/shape 核对 | PASS | `checkpoint_transfer.json` |
| resolved config 无 CBGS/旧范围/NuScenesMetric/sweeps/vel | PASS | `resolved_config.py`, `model_verification.json` |
| 四个 pipeline 完整遍历 | PASS | `model_verification.json` |
| 前向张量尺寸与 LSS 几何 | PASS | `model_verification.json` |
| 反向、梯度、参数更新 | PASS | `model_verification.json` |
| 20 iter smoke、checkpoint 保存 | PASS | `smoke_summary.json`, `smoke_v2.log` |
| 重载预测一致、resume 到 iter21 | PASS | `model_verification.json`, `resume_21/resume.log` |
| val 推理接口与两种导出 | PASS | `val_predictions_raw.json`, `val_predictions_roi.json` |
| 6 epoch 完整微调 | NOT_RUN | 本轮范围止于 20 iter smoke |
| 真实 AP/NDS evaluator | NOT_RUN | 任务书未要求本轮实现 |

## 6. 限制与后续

- 本轮没有执行完整训练、超参数搜索、AP/NDS 评价、多帧对齐或多相机接入；不以低分调阈值或更换类别/划分/范围。
- 单片段时间留出不保证场景或轨迹独立；`annotation.id` 不是全局跟踪 ID。
- 后续任务：小样本过拟合验证、正式三类 evaluator、完整微调。
