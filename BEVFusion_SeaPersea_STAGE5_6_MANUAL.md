# SeaPersea 阶段 5、6：agent 操作手册

日期：2026-11-18。状态：待执行。本文是固定实验合同，指标数值是验收门槛，不是已取得的成绩。

先读 `BEVFusion_SeaPersea_STAGE3_4_REVIEW.md`，再读本文。README 的远程 PASS 不替代该复核要求。

## 0. 目标、边界和执行顺序

阶段 5：用固定 8 帧完成小样本过拟合，验证模型能够学到位置、尺寸和类别，而不只是 loss 可以反传。

阶段 6：实现有明确协议、通过独立合成测试的三类评估器，完成固定 6 epoch 全训练集微调及一次锁定测试集评价。

**实际顺序固定为：G0 修复阶段 3–4验收 → E0 实现并验证阶段 6 的公共评估核心 → S5 过拟合 → S6 完整基线。** 先实现 E0 是为了用同一套真实几何指标验收过拟合，不代表提前使用 val/test 调参。

修改仅限 ADAPTER_ROOT，不改 SOURCE_ROOT，不改已有数据 v1 的三类、划分、范围、体素、相机或强度规则。所有新产物放 `work_stage56/` 下的新目录，拒绝覆盖已有运行；远程目录必须通过参数传入，不硬编码 `/data/coding/...`。

本机当前 SOURCE_ROOT 为 `D:/my_project/usv-fusion/mmdetection3d`，ADAPTER_ROOT 为同父目录下的 `usv-fusion`。远程 GPU 继续使用已验证的 Linux/torch2.0.1+cu118/spconv2.3.8 环境，Windows CUDA 移植不在范围内。

不进行：超参数搜索、数据扩充/重新划分、伪标签、时序/多目、模型替换、LiDAR-only 消融、nuScenes 元数据伪造或 NDS 计算。首次基线效果差时如实报告，不自动追加训练或调整阈值。

## 1. G0：必须先完成的修复和补证

逐项执行复核报告 R1–R8，保持原模型结构和数据版本：

| 项目 | 固定操作 | 放行条件 |
|---|---|---|
| R1 loss 监控 | 接收普通数值/标量 Tensor；直接取 outputs.loss；验证分量存在且有限；修复空值假 PASS、总值重复累加 | float/Tensor/NaN/空记录回归测试通过；20 iter 记录逐项与 Runner 日志一致 |
| R2 权重加载 | deepcopy 目标 OrderedDict state_dict，保留 `_metadata`；按原迁移边界复制非 head 权重；重新生成 init checkpoint | 原生 strict load、磁盘往返、正常 MMEngine load/resume 全通过，几何参数正确；关闭全局 loader patch |
| R3 resume | 从运行配置/manifest 继承原 ann_file、sampler 和哈希；恢复前后核对 optimizer moments/step | iter20→21 只执行一次更新，使用原 smoke 8 帧；恢复前 moments 与 iter20 checkpoint 完全相同 |
| R4 seed | `randomness=dict(seed=2026,deterministic=False)` | 实际 runner.seed=2026，记录抽样种子；不宣称 CUDA 位级确定性 |
| R5 数据验证 | 检查原始 pkl 元数据；pkl/TXT/manifest 身份精确一致；完整数据一致性和独立根路径校验 | 错版本、重复 token、val 混入 train、后段 bin/标定变更均被拒绝；真实数据新报告通过 |
| R6 更新验证 | 四分支同时快照，一次 step，再同时比较；采用实际训练 optimizer 和 clipping | 一次更新后各分支存在有限梯度和参数变化 |
| R7 导入路径 | SOURCE_ROOT 设置/校验先于框架导入；显式注册扩展模块 | 任意 cwd 可运行；错误源码根拒绝；不依赖恰好安装了同一 clone |
| R8 空预测 | `(0,7)` boxes 和空 labels/scores 合法 | 零检测结构验证通过，评估将对应 GT 计为 FN |

同时补强输入合同：重复 timestamp frame_id、NaN pitch/roll、重叠 split 和 spec 版本错误必须拒绝；真实 dataset 往返测试检查非零高度/非方框/非零 yaw，不能只测试框架盒子类。

每个 validator 都使用 try/finally 保存结构化报告，即使失败也有 FAIL 原因；缺依赖标 NOT_RUN，不用异常后遗留旧 PASS 文件。普通数据检查与 mmdet3d/GPU 检查分层，前者只依赖 NumPy/Pillow/标准库。

R2 不允许直接重写旧模型 checkpoint 的数值或无依据添加版本。优先从可信官方融合权重重新迁移生成新 init；必要时只读诊断旧 checkpoint。保留旧产物。若 native load 仍失败，交付最小复现并停止 G0，不回退为宽松忽略缺失键。

运行目录：`work_stage56/gate34/{data_validation,init,smoke,model_validation,evidence}`。保存代码/配置/数据/权重 SHA256。收集原始训练日志及修正后的 JSON，不仅引用 README。

完成后输出 `stage34_reacceptance.json`：逐项 PASS/FAIL/NOT_RUN/DOCUMENTED_ONLY。只有本表全部通过才允许执行 S5 训练；E0 的 CPU 实现与合成测试可在等待 GPU 时进行。

## 2. 固定输入与文件交付

固定集合（需要实读产物再次确认）：

| split | 帧数 | ship/buoy/other | 用途 |
|---|---:|---|---|
| smoke | 8 | 16/100/8，共124框 | S5 训练与同集诊断 |
| train | 232 | 505/324/232，共1061框 | S6 微调 |
| val | 54 | 162/54/54，共270框 | S6 选 checkpoint |
| test | 54 | 205/54/54，共313框 | 选定 checkpoint 后评价一次 |

不得将 labeled_all 的372帧或 temporal_gap 加入训练。S5 的评价是在训练样本上进行，仅证明记忆/拟合能力，不证明泛化。

新增文件固定职责：

```text
configs/seepersea_eval_v1.json            # 下面全部指标协议，不从数据拟合阈值
usv_fusion_ext/evaluation.py             # 纯 NumPy：合同校验、ROI、IoU、匹配、AP
tools/export_seepersea_predictions.py    # checkpoint→版本化预测JSON
tools/evaluate_seepersea.py              # GT pkl + 预测JSON→metrics/PR/matches
tools/run_seepersea_stage56.py           # 固定 overfit/baseline/resume 入口
tests/test_seepersea_evaluation.py        # 独立解析答案与边界测试
README_SeaPersea_STAGE5_6.md              # 实际命令、结果、限制及证据索引
```

复用并修复现有 runtime/config/dataset，不复制第二套数据转换或模型实现。允许在现有 runtime 中增加共享 checkpoint/seed/运行清单逻辑。图表可用 matplotlib，但指标核心不得依赖 GPU、nuScenes devkit 或联网。

## 3. E0：评估输入合同

### 3.1 预测导出必须统一为重力中心

新 JSON 顶层格式：

```json
{
  "schema_version": "seepersea_predictions_v1",
  "prediction_source": "model",
  "dataset_version": "seepersea_3class_v1",
  "classes": ["ship", "buoy", "other"],
  "split": "smoke",
  "coordinate_frame": "lidar",
  "box_origin": "gravity_center",
  "box_dim_order": "lwh",
  "yaw_unit": "rad",
  "checkpoint_sha256": "...",
  "config_sha256": "...",
  "ann_sha256": "...",
  "manifest_sha256": "...",
  "protocol_sha256": "...",
  "results": [
    {"token": "barbados-20230223-240:000000",
     "boxes": [], "labels": [], "scores": [], "prediction_ids": []}
  ]
}
```

- 一条 box 为 `[cx,cy,cz,l,w,h,yaw]`；`prediction_ids` 为该帧模型输出中的稳定原始序号。
- 模型 bboxes_3d.tensor 是底面中心；导出时从 `bboxes_3d.gravity_center` 取 xyz，并拼接 tensor[:,3:7]。**不要再对 gravity_center 加 h/2。**
- GT 直接读正式 pkl 的重力中心框，不通过会增广的训练 pipeline 获取。
- 不直接接收阶段4旧 JSON；其框是底面中心，且缺少原点/版本声明。本轮从指定 checkpoint 重新导出，不添加自动猜测原点的逻辑。
- checkpoint 必须 strict load；校验 LSS dx/bx/nx/frustum 和三类配置。推理时 model.eval + no_grad；不能用经过反向检查后又更新过的模型冒充 checkpoint。
- 使用阶段4确定性的 test_pipeline，score_threshold=0.0，不附加0.05/0.2显示阈值，不新增 NMS，保留模型返回的全部至多200个候选。

### 3.2 校验与 ROI

- 导出器按所选 split 的 pkl 顺序输出；evaluator先检查 token 唯一且集合与pkl完全一致，再按pkl顺序规范化records，允许输入文件仅行序不同。每帧恰一条结果，零预测保留空数组。
- boxes N×7、labels/scores/prediction_ids 长度都是 N；ID唯一且整数，labels 为整数0–2，score有限且在[0,1]，尺寸正且所有几何值有限。非法值报错，不静默丢弃。
- GT 和预测统一按**重力中心**执行 ROI：`[-108,-108,-20,108,108,20]`，下界含、上界不含。不裁剪框尺寸，不按角点剔除，不限制相机视野。
- 导出文件保留解码输出全量候选；evaluator 负责统一 ROI，并同时记录 ROI 前后数量。上游 coder 已有底面中心范围保护，不把导出文件称为“未经过任何过滤的网络原始张量”。
- 实际各集合 GT 中心应全部在 ROI，若不成立则数据门槛失败，不悄悄减少 GT 分母。
- 没有预测的帧仍贡献 GT/FN；已排除的66个缺标注帧不能作为空GT负样本重新加入。

## 4. E0：评估协议固定为 seepersea_eval_v1

### 4.1 指标及用途

| 指标 | 固定阈值 | 用途 |
|---|---|---|
| 每类 AP3D 与 mAP3D | IoU≥0.25、≥0.50 | 主指标；checkpoint 按 mAP3D@0.25 选择 |
| 每类 APBEV 与 mAPBEV | IoU≥0.25、≥0.50 | 辅助判断 z/高度问题 |
| 每类 RecallCenter2m | xy 中心距离≤2.0m，同类 | S5 定位诊断；使用全部候选，不是部署阈值表现 |
| 每类 TP/FP/FN、最终 precision/recall | 各 IoU 阈值 | 可核对 AP 的原始统计 |
| 定位/尺寸误差 | AP3D@0.25 的匹配 TP | 中心3D距离中位数、绝对z误差中位数、三轴尺寸相对误差中位数 |

主排序指标名称固定 `SeaPersea/mAP3D@0.25`，值域0–1；表格可另列百分数。禁止把它叫 nuScenes mAP 或 NDS。阈值是本项目固定基线协议，不宣称行业标准或最佳选择。

### 4.2 旋转框 IoU

- 使用 float64。
- 在框局部坐标中构造 `(+/-l/2,+/-w/2)` 矩形顶点，按 CCW 顺序，以 yaw 绕+z旋转后平移到(cx,cy)。不得把 l/w 对调。
- BEV 交面积：凸多边形 Sutherland–Hodgman 裁剪 + shoelace 面积；面积退化为0时交面积=0。边界几何判断 eps=1e-9；结果夹到合法范围以抵消数值舍入。
- `IoU_BEV=A_inter/(l1*w1+l2*w2-A_inter)`。
- 重力中心 z 区间为 `[cz-h/2,cz+h/2]`；`h_inter=max(0,min(zmax1,zmax2)-max(zmin1,zmin2))`。
- `IoU_3D=A_inter*h_inter/(l1*w1*h1+l2*w2*h2-A_inter*h_inter)`。
- 不使用相机2D框/轴对齐矩形近似旋转框。IoU接近阈值时直接比较计算值≥阈值，不加宽匹配阈值。

### 4.3 匹配和 AP

对每个类别、每个 IoU 阈值独立执行：

1. 收集整个 split、ROI 内同类预测，按 `(-score,token,prediction_id)` 稳定排序。
2. 每条预测只与同帧同类尚未匹配 GT 比较；取 IoU 最大者，达到阈值则 TP 并占用该 GT，否则 FP。最大 IoU 相同时取 GT 原 labels 下标较小者。
3. 一个 GT 只能匹配一次；重复检测是 FP。不同类别不匹配。每个阈值重新清空匹配状态。
4. 累积 TP/FP，`recall=TP_cum/N_GT`，`precision=TP_cum/(TP_cum+FP_cum)`。
5. AP 固定101点：在 r=0,0.01,...,1 上取所有 recall≥r 的 precision 最大值，无满足点时为0，再取101点平均。不得切换梯形积分/11点/其他插值。
6. 类别有 GT 但无预测，AP=0；无 GT 时 AP=null并输出该类FP数，宏平均只含有GT的类并列出 supported_classes。正式四个集合都必须包含三类，故正式报告的宏平均分母应为3。
7. RecallCenter2m 使用同样分数排序和一对一规则，以最近未匹配同类 GT 的 xy 距离≤2m 作为匹配条件，独立于 IoU 匹配。

不使用 Hungarian 匹配计算 AP；训练 assigner 的 Hungarian 与评估排名匹配是不同用途。

输出 `metrics.json`、`pr_curves.json`、`matches.json`；后者列出每个阈值下逐预测 TP/FP、匹配 GT 下标和 IoU、未匹配 GT，确保不是只输出一个不可检查的平均数。JSON 空指标用 null，不写 NaN/Infinity。

### 4.4 必须先通过的独立测试

测试预期值必须来自手算/明确构造，不能用被测 evaluator 自己生成“标准答案”：

1. 三类完美预测：各 AP=1，TP=GT，FP=FN=0。
2. 有GT无预测：AP=0，FN=GT；空结果序列合法。
3. 错类别、完全不相交、相同 BEV 但 z 不重叠：3D 匹配失败，后者 BEV IoU=1而3D IoU=0。
4. 重复预测：一个GT只有一个TP，多余为FP；若FP排在满召回之后，AP可能仍为1，不错误要求所有重复预测都降低AP。
5. 一个高分FP后跟一个TP、一个GT：AP=0.5。
6. 旋转非正方形框、yaw+2π等价、90°旋转、只接触边界、包含框；验证解析面积及体积。
7. 底面→重力中心恰好加h/2一次；相同预测经过新导出与 GT 完全重合，不二次偏移。
8. ROI的x/y/z下界保留、上界排除；ROI过滤按中心，框跨界不裁尺寸。
9. 同分预测稳定排序；打乱JSON records后按合同排序得到相同数值；缺失/额外/重复token、非法类别/score/维度/尺寸必须拒绝。
10. 无GT类别、混合空帧、无TP误差统计为null；各阈值独立匹配；修改schema/原点/数据哈希应拒绝。

在框架环境可用时，另抽取100对固定seed的合法旋转框，与本地 mmdet3d/mmcv 的3D IoU实现交叉比对，先转换到它要求的底面中心约定，非退化样例绝对误差≤1e-4。该交叉检查验证实现，不把框架实现当作唯一测试答案。

E0通过后，用 GT 构造 oracle 预测跑完整 smoke 和 val，三类 AP=1、TP数等于124/270，token数量8/54。oracle文件必须标记 `prediction_source='oracle'`、checkpoint_sha256=null（model config哈希也允许null），但ann/manifest/protocol哈希必填；模型导出固定 `prediction_source='model'` 且checkpoint/config哈希必填，S5/S6模型成绩表拒绝oracle输入。oracle不读取test，不混入模型结果。

## 5. S5：固定8帧过拟合

### 5.1 唯一实验配置

| 参数 | 固定值 |
|---|---|
| 训练/诊断集合 | infos_smoke.pkl，同一8帧/124框 |
| 起点 | G0新生成的受控三类 init；不从iter20或之前overfit恢复来冒充新实验 |
| 模型、范围、图像、强度、三类 | 全部沿用修复后的阶段4配置 |
| FP32 / batch / workers | FP32，batch=1，workers=0，persistent_workers=False |
| seed | Runner randomness.seed=2026 |
| 优化器 | AdamW lr=1e-4，weight_decay=0.01，grad clip norm=35 |
| scheduler / 冻结 | 无scheduler，不冻结分支 |
| pipeline | 保留阶段4固定图像变换、单位3D增强和PointShuffle；不加新增强 |
| sampler | DefaultSampler(shuffle=True)，固定seed |
| 迭代 | IterBasedTrainLoop，max_iters=1000 |
| checkpoint | 每100iter保存，save_last=True |
| 同集评价 | iter0/100/200/.../1000，各8帧，eval模式、test_pipeline |

迭代0为 init 权重尚未更新的结果。每次评价另建推理模型/独立进程，避免切换mode或BN统计影响训练；推理中不更新参数。记录总loss、三个loss分量、梯度有限性、实际学习率、显存峰值、step数和采样token。

训练结束绘制loss曲线与同集AP3D/APBEV/RecallCenter2m曲线；可用PNG/SVG，不需网页。固定可视化8帧全部GT和预测：同时保存BEV与相机投影，图上注明显示阈值0.05，**该阈值不参与指标**。

### 5.2 阶段5通过标准

必须同时满足：

1. G0/E0通过；1000次更新全部实际完成，loss/梯度/参数无NaN/Inf，全部token来自固定8帧。
2. 总loss取真实Runner loss；最后50iter中位数≤最初50iter中位数的0.5倍。分量曲线必须同时交付，不只给比值。
3. iter1000的同集 `mAP3D@0.25≥0.80`，每类 `AP3D@0.25≥0.70`。
4. 同集每类 `RecallCenter2m≥0.90`。
5. strict重载 iter1000 后，固定输入的预测在 rtol=1e-4/atol=1e-5 内一致，标签完全一致；所有结果与 checkpoint 哈希绑定。

这些是预先固定的学习能力门槛，不保证当前架构必然达到。失败后必须先区分实现错误、评估错误与模型/标注限制，不能把门槛改成实际成绩。

### 5.3 失败处理：只允许诊断，不自行调参

若未通过，输出 `stage5_failure.md`：

- loss各分量与AP/定位召回是否一致；BEV好但3D差时检查重力中心、h、z和导出原点。
- 每类GT/TP/FP/FN，误差分位数；buoy低时检查密集目标在180×180热图中的中心碰撞及3×3局部极大抑制，记录而不改kernel/min_radius。
- 检查原始GT、数据集GT、训练目标decode回来的框是否一致，输出逐框差值。
- 核对梯度、更新、受控权重、seed、数据身份；检查BN train/eval差异，不自动冻结BN或换batch。

发现代码缺陷时修复、加回归测试，用同一个规范init从0重新跑；最多再完整跑一次1000iter。若无明确实现缺陷或第二次仍不通过，停止S6微调并提交证据。不得通过换样本、提高LR、额外epoch或降低阈值来制造PASS。E0及文档整理可继续完成。

## 6. S6：正式评估与6 epoch基线

### 6.1 模型运行与数据使用规则

S5通过后，从**同一个G0三类init重新开始**；不能从已记忆8帧的overfit权重接着训练。

- train=232帧，val=54帧；参数同阶段4固定基线：AdamW 1e-4、weight_decay=.01、clip35、FP32、batch1、无scheduler、seed2026、无新增强。
- EpochBasedTrainLoop max_epochs=6；每epoch保存；无重复采样、无drop_last，应完成232×6=1392次更新。
- 分别保存 epoch1–6 checkpoint；训练后用独立推理入口对这6个checkpoint逐一导出val并评价即可。本轮固定使用**离线验证**，无需再实现MMEngine BaseMetric/在线evaluator包装。
- 顶层 Runner val/test 配置继续为None；正式评价由CLI完成，避免为了接入指标更改训练过程。
- 另对 init 做一次val基线。表中把它标为“随机三类head + 预训练特征，未微调”，不是原nuScenes十类模型的海面精度。
- test 只在下面选择规则冻结后，对最终选中checkpoint评价一次。不得用 test 选择epoch、score阈值、IoU阈值或修模型。

run_seepersea_stage56 在overfit/baseline/resume中显式保存 experiment_id、split/pkl/config/init hash；resume必须继承它们，禁止由mode自动换数据或重新初始化optimizer。

### 6.2 checkpoint选择规则

从epoch1–6中选择val `mAP3D@0.25`最高者，使用未四舍五入的值；差值≤1e-12视为并列，选择更早的epoch。不得按test成绩选择，亦不选择init作为“已训练模型”。

生成不可变 `selection.json`：候选每个epoch的文件路径/哈希/val指标、选中epoch、规则、数据/协议哈希、UTC时间。生成后才开放test推理。若test发现程序错误，只允许修复该错误并在**同一个锁定checkpoint和协议**重算，记录所有运行和原因；不能重新选epoch。

若错误同时影响val指标或候选排序，原selection标为INVALID，阶段6选模验收FAIL；可以报告原锁定checkpoint的修正test诊断，但不能继续称其为修正后val最优模型。不得看过test后重新选模来完成本轮验收，应把后续重新设计实验的需求单列提交。

### 6.3 固定输出

```text
work_stage56/
  gate34/...
  evaluator_tests/{unit_results,oracle_smoke,oracle_val,crosscheck}/...
  overfit/run01/
    resolved_config.py, run_manifest.json, training_log.jsonl
    iter_*.pth, predictions/, metrics/, curves/, visualizations/
    stage5_acceptance.json
  baseline/run01/
    resolved_config.py, run_manifest.json, training_log.jsonl
    epoch_1.pth ... epoch_6.pth
    val/init/, val/epoch_1/ ... val/epoch_6/
    selection.json
    test/selected/
    results_table.csv
    stage6_acceptance.json
  evidence_manifest.json
```

原始日志与manifest逐行可追踪；所有指标json包含代码版本、协议、配置、checkpoint、数据清单哈希。

结果表包含：init-val、6个epoch-val、selected-test；每行列三类AP3D@.25、宏AP3D@.25/.50、宏APBEV@.25/.50、GT/预测数、耗时、显存。S5同集成绩放单独表，不与val/test泛化成绩并列排名。

距离和点数诊断固定使用 **Recall@IoU3D0.25**，不创建新的按桶AP：

- GT水平距离桶 `[0,20),[20,40),[40,60),[60,+inf)` 米。
- GT原始num_lidar_pts桶 `[1,5),[5,20),[20,+inf)`。
- 使用全split同类全候选匹配结果，按GT所属桶累积TP/GT和FN；不通过删除桶外预测重算AP，避免匹配/FP口径变化。

初版不挑阈值计算部署精确率，不报告跨场景泛化。说明本数据是单片段时间留出，目标可能跨集合持续出现。

### 6.4 阶段6通过标准

- 指标实现通过E0测试、oracle和交叉检查；固定协议与单位/坐标没有歧义。
- 6epoch/1392次更新真实完成，loss及梯度有限，无宽松加载，所有checkpoint及val结果齐全。
- 选模只用val，test token恰54，test只针对锁定checkpoint；所有身份/哈希校验通过。
- 逐类AP/PR/TP/FP/FN、距离/点数诊断和完整复现命令齐全。
- 指标可从同一预测文件重复计算完全一致（JSON运行时间字段不比较）；checkpoint重导出预测允许规定浮点容差。

本阶段“流程通过”和“精度达标”分开：不预设test AP必须达到某数值，也不要求6epoch一定优于init。若未改善，报告ΔAP与错误分布，标明基线效果不足；不得隐藏结果或扩大实验预算。本轮产出用于决定下一轮范围/分辨率/监督/数据规模研究。

## 7. CLI 合同与交付顺序

下面入口是需要实现的接口，当前仓库尚不存在；不要把示例当成已经可运行的命令。参数均接受绝对路径。README最终须给出真实Linux单行命令。

```text
python tools/export_seepersea_predictions.py --mmdet3d-root SOURCE_ROOT --data-root DATA_ROOT --checkpoint CHECKPOINT --split smoke|val|test --protocol configs/seepersea_eval_v1.json --out PREDICTIONS_JSON

python tools/evaluate_seepersea.py --data-root DATA_ROOT --split smoke|val|test --predictions PREDICTIONS_JSON --protocol configs/seepersea_eval_v1.json --out-dir METRICS_DIR

python tools/run_seepersea_stage56.py --mmdet3d-root SOURCE_ROOT --data-root DATA_ROOT --init-checkpoint INIT --protocol configs/seepersea_eval_v1.json --mode overfit --work-dir work_stage56/overfit/run01

python tools/run_seepersea_stage56.py --mmdet3d-root SOURCE_ROOT --data-root DATA_ROOT --init-checkpoint INIT --protocol configs/seepersea_eval_v1.json --mode baseline --work-dir work_stage56/baseline/run01
```

resume增加 `--resume-from CHECKPOINT`，从run_manifest获取原始模式/数据/config，拒绝冲突CLI。test导出额外要求 `--selection-json selection.json`，并验证checkpoint哈希等于选择文件，防止随意读取测试集结果。

按以下四个独立工作单交付，允许分别提交代码但不得自动推送：

1. **G0工作单**：修复R1–R8、数据/加载/监控回归测试、新阶段34证据包；不训练1000iter。
2. **E0工作单**：协议、导出器、CPU evaluator、合成测试和oracle/IoU交叉检查；不跑test模型指标。
3. **S5工作单**：按表运行1000iter、同集评价、可视化和验收；失败按固定流程报告，不自动扩展。
4. **S6工作单**：只有前面通过后执行6epoch、逐epoch-val、冻结选模、一次test、结果表与局限说明。

若接手agent缺少远程资源，先做代码和CPU测试，列明所缺路径/环境及NOT_RUN项；不要重新索要文档中已确定的实验选择，不把“没有GPU”改写成“模型不支持”。

## 8. 最终回报模板

必须回答：

1. G0/E0/S5/S6各自PASS/FAIL/NOT_RUN，证据路径；哪些实测，哪些只继承历史记录。
2. R1–R8如何修复，原生checkpoint加载及真实resume验证结果。
3. 数据/模型/协议有无偏离本文；若有偏离，不宣称按本合同验收通过。
4. S5初末loss、每类AP3D@.25、RecallCenter2m、是否通过全部门槛。
5. S6选中epoch、选择依据、val与test逐类/宏指标；test是否参与了任何决策。
6. 完整命令、代码/权重/数据/协议哈希、日志及可视化链接。
7. 对失败或低精度给出有证据的解释；禁止以“出框了”“退出码0”“mAP有限”替代本手册的验收。
