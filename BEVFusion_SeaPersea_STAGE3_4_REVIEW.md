# SeaPersea 阶段 3、4 验收复核

日期：2026-11-18。审阅对象：当前 README、阶段 3–4 任务书和适配仓库代码。

## 1. 结论

**主体实现可承接；阶段 3 有条件通过代码审查，阶段 4 暂不能签署全部 PASS。先修复验收与加载问题、补齐运行证据，再开始阶段 5 的学习能力实验。**

这不是认定远程训练失败。README 记录了远程成功运行，但本地没有 `work_stage34/`、原始片段、checkpoint、原始日志和验收 JSON，无法独立确认其中数值；且当前验收代码存在可能产生假 PASS 的缺口。

本次只新增审阅及下一阶段操作文档，没有修改实现代码或执行 GPU 训练。

## 2. 已确认的成果及证据边界

| 项目 | 本次结论 |
|---|---|
| 原生类别、框 lwh/重力中心/yaw、空标注剔除、固定时间段划分 | 转换器实现与任务书主要要求一致 |
| 原始强度落盘、内存 clip、真实框内点数、复制图片 | 源码可确认 |
| 独立 SeaPerseaDataset、单相机单帧、7D GT / 8D 回归、无 vel | 源码可确认；原始 pkl 元数据拒绝机制仍需补强 |
| ±108/z±20、体素 0.15/0.15/1、网格/图像/融合尺寸配置 | 静态配置符合任务书；实际 CUDA 尺寸依赖远程证据 |
| train=232/1061、val=54/270、test=54/313、gap=32/160 | README 记录，算术自洽，本地未从真实数据复算 |
| 372 有标注帧、1804 框、968/464/372 三类统计 | 与历史审计及 README 一致，本地尚无产物可独立复算 |
| 合同测试 | 本地实际运行 8 项：7 PASS，1 SKIP（缺少 mmdet3d 的重力中心测试） |
| GPU 前向、反向、20 iter、峰值显存、重载、val 推理 | README 记录；本次 NOT_RUN，不能直接继承为本次实测 PASS |
| 全量微调、实际 AP | 未执行，符合阶段 3–4 的范围 |

本地 SOURCE_ROOT 当前名称已经是 `D:/my_project/usv-fusion/mmdetection3d`，旧任务书中的 `mmdetection3d-main` 是历史目录名。运行仍必须由显式 CLI 根路径决定。

## 3. 必须修复的发现

### R1 / P1：训练监控会把真实 loss 记成 0

位置：`usv_fusion_ext/runtime.py:407–409`；上游 `projects/BEVFusion/bevfusion/bevfusion.py:102–113`。

BEVFusion.parse_losses 在返回日志前对值执行 `.item()`，标准训练日志是 Python float。MonitorHook 仅收集 Tensor，导致 `losses={}`、`total_loss=0`；`verify_seepersea_model.py` 只检查记录数量和有限性，无法识别这种空记录。

本地对真实 hook 方法做 AST 提取后最小复现：输入 `loss=3.0`、三个分量各 1.0、matched_ious=0.2，输出 `total_loss=0, losses={}`。该检查可复现，不依赖 GPU。

修复：接受 Python 数值/NumPy 数值/标量 Tensor；总损失直接读取 `outputs['loss']`。缺少 loss、非有限值或空分量必须报错；不得把总 loss 与分量再次相加。补 float/Tensor/NaN/空字典回归测试，重新运行 smoke，旧 smoke_summary 不再用于证明 loss 正常。

### R2 / P1：checkpoint 元数据被丢弃，不能归因于上游缺陷

位置：`tools/prepare_seepersea_checkpoint.py:61–73`；`usv_fusion_ext/runtime.py:333–375`；上游 `mmdet3d/models/layers/spconv/overwrite_spconv/write_spconv2.py:44,63,85`。

稀疏模块声明 `_version=2`。加载器在 local_metadata.version 不是 2 时，会进行 v1→v2 的维度转置。迁移工具将目标 state_dict 改为普通 dict，再构造 `migrated={}`，丢失 OrderedDict 的 `_metadata`；随后本应已是 v2 的张量被再次转置。

本地 CPU 最小复现使用上游原加载函数和一个 `_version=2` 的参数模块：保留 state_dict._metadata 时 strict load 全匹配；改为 dict(state_dict) 后出现 shape mismatch。环境是 torch 2.12.0+cpu，复现的是元数据分支机制，不代替远程 torch 2.0.1/spconv 实测。

修复：以 `deepcopy(target_model.state_dict())` 为底稿替换键的张量，完整保留目标 `_metadata` 并序列化；重新在远程验证原生 `load_state_dict(strict=True)` 和正常 MMEngine load/resume。不得手工给来源不明的 checkpoint 注入 version=2。新实验使用新迁移权重，不覆盖旧文件。移除针对全部 MMEngine 模型的无条件 loader monkeypatch；如仍失败，保存完整异常及最小复现，停止加载器验收，不继续归因或扩大绕过范围。

显式张量复制本身不意味着现有远程张量一定错，但“相同布局前提、完整加载及可移植性”需要重新验证。README 的“shape-proxy defect”说法应改为已验证的实际原因。

### R3 / P1：resume 会静默切换数据集，现有恢复检查不够

位置：`usv_fusion_ext/runtime.py:460–477`；`tools/verify_seepersea_model.py:287–306`。

只有 mode=smoke 使用 infos_smoke.pkl，mode=resume 落到 infos_train.pkl。因此从 20 iter 恢复到 21 iter 同时从 8 帧变成 232 帧；当前检查只看 iter=21 和 optimizer state 非空，无法证明原 optimizer/数据身份被恢复。

修复：resume 从保存的配置/运行清单继承 split、pkl SHA256、模型配置和初始化来源，禁止由 mode 推断新数据集。恢复第一步前检查 iter=20 和 optimizer 的 step/exp_avg/exp_avg_sq 与 checkpoint 一致，执行一步后检查 iter=21、step 增长且采样 token 属于原 smoke 集。单独记录采样/RNG 是否恢复；没有验证不得宣称逐步轨迹等价。

### R4 / P1：固定 seed 没有传给 Runner

位置：`usv_fusion_ext/runtime.py:453`。

代码只设置 cfg.seed；MMEngine 0.10.5 Runner.from_cfg 使用 cfg.randomness，默认 seed=None。初始化权重工具的 seed_all 不等于独立训练进程已经固定 seed。

修复：设置 `cfg.randomness=dict(seed=2026, deterministic=False)`，实际记录 runner.seed、采样 seed 和初始化权重哈希。证据：[MMEngine v0.10.5 Runner.from_cfg](https://raw.githubusercontent.com/open-mmlab/mmengine/v0.10.5/mmengine/runner/runner.py)。

### R5 / P2：数据验收对泄漏、元数据和一致性的检查不完整

位置：`tools/validate_seepersea_v1.py:129–166,189–254,289–325`；`usv_fusion_ext/dataset.py:18–35`。

- TXT 集合互斥不代表 pkl 互斥。没有要求 pkl token 列表与对应 TXT 完全相等，也没有拒绝额外/重复记录或逐帧复算固定区间。向 train.pkl 追加一条 val 记录可绕过当前这些验收条件；这是检查漏洞，不是已经发现真实数据泄漏。
- SeaPerseaDataset 检查合并后的 self.metainfo；类 METAINFO 和构造参数会遮住 pkl 的同名版本/类别。必须在合并之前检查原始 pkl 元数据。依据：[MMEngine v0.10.5 BaseDataset.load_data_list](https://raw.githubusercontent.com/open-mmlab/mmengine/v0.10.5/mmengine/dataset/base_dataset.py)。
- 两次转换检查只比较 images 的 key，没有比较标定/图片路径/时间戳；bin 只比前 8 帧。
- relocation 检查忽略 check_dataset 返回的 bad_paths；软链接也没有证明数据完全自包含。

修复：对所有 pkl 做原始元数据及有序 token 精确检查；按 spec+原始空标注逐帧重建角色；完整比较所有样本语义和全部 438 个 bin/图片哈希；在独立复制的数据根验证全部路径且不能回指原根。增加故意篡改 pkl、标定、后段 bin、版本的拒绝测试。data-only 校验与依赖框架的 dataset 校验分开输出状态。

### R6 / P2：反向验收进行了四次 optimizer.step

位置：`tools/verify_seepersea_model.py:231–252`。

optimizer.step 位于四分支循环内部，实际使用同一组未重新计算的梯度更新四次。不能解释为一次训练更新后的四分支验证。

修复：一次 forward/backward 后同时保存四分支参数，所有梯度检查完毕后仅一次 step，再统一比较变化。该测试模型不得用于后续评估；评估要重新从指定 checkpoint 建模/加载。

### R7 / P2：入口仍依赖预先正确安装的源码

位置：`tools/prepare_seepersea_checkpoint.py:18`；`usv_fusion_ext/__init__.py`；`usv_fusion_ext/runtime.py:294–330`。

prepare 在处理 --mmdet3d-root 前导入 mmdet3d。扩展包 __init__ 也会提前尝试导入；若首次不可用，后续再次 import 已缓存的扩展包不会自动执行注册。

修复：包入口保持惰性；先设置/验证 SOURCE_ROOT，再显式 importlib.import_module(dataset/transforms)。MODELS 等框架导入移动到 setup 之后。不同工作目录、没有 editable install 的正确根路径及错误根路径均要测试。

### R8 / P2：合法零检测会被验收脚本误判为运行错误

位置：`tools/verify_seepersea_model.py:213–216`。

首次输出检查直接对 labels.min/max 求值，零检测时异常。当前配置及初始化不保证一定有框，任务书允许零检测。

修复：空预测应以 `(0,7)` 框、`(0,)` labels/scores 通过结构验证，并在指标中正常产生 FN；不得为通过验收放宽阈值。

## 4. 下一阶段特别注意的评估合同

现有 `val_predictions_raw.json` / `roi.json` 由 `box_obj.tensor` 导出，**框 z 是底面中心**。`roi_basis='gravity_center'` 只说明过滤依据，不会把保存的 tensor 转成重力中心。

正式 GT pkl z 是重力中心。阶段 6 必须显式统一：预测 `cz=z_bottom+h/2`，只转换一次。新导出加 `schema_version/box_origin/box_dim_order/yaw_unit/coordinate_frame`，拒绝没有原点声明的旧 JSON直接进入 evaluator。不能用 xy/BEV 指标通过就断言 3D 几何无误。

## 5. 补证与重新签署验收

必须从远程交付一个与代码哈希绑定的轻量证据包：转换报告、全部 split 清单、manifest、validation_report、resolved_config、config_provenance、checkpoint_transfer、原始训练日志、修正后的 smoke_summary、model_verification、resume 日志、预测样例和各大文件 SHA256。原始数据与完整权重可以保留远程，但必须有可访问路径和哈希。

先完成 R1–R8，再运行对应回归测试和必要的远程 smoke/重载/resume。无需为了审查重复完整训练；不得编辑旧 PASS JSON 伪装重测，应使用新 work_dir 并保留旧产物。

本次本地测试命令：

```powershell
& 'D:\application\Miniconda\envs\learn-pytorch\python.exe' -B -m unittest discover -s 'D:\my_project\usv-fusion\usv-fusion\tests' -v
```

结果：8 项，7 通过，重力中心框架测试 1 跳过。最初默认 Python 缺少 numpy；随后复用现有 learn-pytorch 环境，未安装或修改项目依赖。测试产生的缓存变更已恢复，实施源码保持不变。

最终状态应使用 PASS / FAIL / NOT_RUN / DOCUMENTED_ONLY 四种值：README 中尚未交付独立证据的远程结果标记 DOCUMENTED_ONLY；本机无法运行的 GPU 项标记 NOT_RUN；不能把这两类折算成 PASS。
