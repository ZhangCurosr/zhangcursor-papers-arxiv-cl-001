---
title: "VISCAD-A-FOUNDATION-MODEL-SUITE-WITH-MUL-TIMODAL-INDUSTRIAL"
source: https://arxiv.org/pdf/2609.03811v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-07 23:13:55"
field: "多模态程序生成与 CAD 智能"
keywords: ["CAD generation", "multimodal foundation model", "test-time scaling", "assembly generation", "domain-specific harness", "intermediate representation", "industrial design"]
innovations: ["27B多模态零件生成模型VisCAD-M1在PubCADBench/RealCADBench上超越SOTA前沿模型", "并行测试时缩放(27B模型自reranker)带来约5%相对提升", "DSL-agnostic装配级中间表示与四阶段gated工作流VisCAD-H1"]
benchmarks: ["PubCADBench", "RealCADBench"]
---

# 论文速读：VISCAD-A-FOUNDATION-MODEL-SUITE-WITH-MUL-TIMODAL-INDUSTRIAL

## 一句话总结
VisCAD 提出了一个面向工业 CAD 的多模态基础模型套件，包含 27B 零件级生成模型 VisCAD-M1 和基于结构化中间表示（IR）的装配级 harness VisCAD-H1，在 PubCADBench 和 RealCADBench 上超越 Gemini-3.1-Pro 等前沿模型，并引入 parallel test-time scaling 实现额外约 5% 相对提升。

## 研究问题与动机
- **输入模态覆盖不足**：现有专用 CAD 模型多仅支持文本或渲染图单一模态，2D 工程图与真实产品照片在工业场景中常见但严重缺乏，尚无统一接口覆盖全谱系输入。
- **域泛化差**：多数方法在 ABC、Fusion 360 Gallery 等合成/CAD-native 数据集上训练和评估，分布局限导致面对真实工业意图（工厂自动化产品照片、长文本描述）时泛化能力弱。
- **装配级问题被忽视**：现有工作几乎全部聚焦单零件生成，复杂装配需额外处理零件分解、mating 关系推理、全局放置等，通用 coding harness 缺乏领域结构化能力（BoM、逐零件 IR 门控等）。
- **数据质量与规模瓶颈**：工业 CAD 数据存在稀疏性，且"golden program"（精确对应设计意图的可执行代码）稀缺，难以直接监督训练。

## 核心贡献（创新点）
- **27B 多模态零件生成模型 VisCAD-M1**：统一将文本、2D 工程图、真实产品照片、渲染图映射为可执行 FreeCAD Python 程序，在 PubCADBench/RealCADBench 上平均 profile score 0.5540，超越 SOTA 前沿模型 Gemini-3.1-Pro（0.5496）。
- **并行测试时缩放（parallel-tts）自 reranker**：将同一输入的多条 rollout 交由模型本身进行 listwise 重排序选择，较基础模型提升 2.7 分（绝对），相对 SOTA 提升约 5%，而序列式迭代精化无效。
- **综合数据治理流水线**：通过 off-the-shelf 图像编辑与 image-to-3D 模型填充缺失意图/程序/几何三元组，并采用 RSI-based 程序搜索获取更高质量监督信号；多阶段训练（mid-training → robust SFT → continual high-quality SFT）最大化利用不同质量和规模的数据。
- **DSL-agnostic 装配级中间表示（IR）与 VisCAD-H1**：将装配问题分解为 BoM 解析、逐零件 IR 门控、全局放置审查与 CAD 后端翻译四个 gated 阶段，同一 IR 可移植到 FreeCAD、Fusion、SolidWorks 三种后端，装配 judge score 达 85.0，显著高于 Codex（68.0）与 Claude Code（52.0）。

## 方法详解
- **统一三元组抽象**：所有数据视作 (intent, program, shape) 三元组，支持前向（i → p → s）与后向（s → p → i）两条路径，前者用于从真实意图重建程序和形状，后者用于从形状生成多样化意图与近优程序。
- **数据过滤**：四步类别过滤保留刚性机械产品（21 大类、128 子分类），剔除软件、电子、耗材及弱 CAD 类别。
- **多阶段训练**：
  - Mid-training：约 100 万 CadQuery 公开程序在环境中执行导出实体，从多角度投影生成多视图图像，建立"黄金"意图-程序对，教授 CAD 语法与 API。
  - Robust SFT：约 15 万电商/爬取产品图像经清理后由教师模型蒸馏出多条候选程序，提升对噪声轨迹的鲁棒性。
  - Continual high-quality SFT：约 1 万精选样本在 generate-verify-feedback 循环中微调，恢复精细几何保真度。
- **Test-time Scaling（parallel-tts）**：对同一输入生成 k 条 rollout（程序 + 六视图导出模型），用模型自身作为 listwise reranker 打分并选最优；k=8 时提升 1.3–1.7 分，k=16 时进一步至 0.5833。
- **VisCAD-H1 四阶段 gated 工作流**：
  1. Multimodal understanding & BoM：从图像/图纸/文本提取零件列表、数量、装配顺序及 mating 提示。
  2. Parallel part IR：为每个零件分配与后端无关的局部几何/接口/约束表示，并在生成固体前进行语法与局部有效性门控。
  3. Global IR review：结合参考视图检查跨零件几何、接口与放置一致性，输出带位姿的已审查 IR。
  4. CAD realization：将 IR 翻译为目标后端 DSL 并行构建并装配。

## 实验与结果
- **数据集**：PubCADBench（1,100 任务，含 BenchCAD/CADBench/正交重建/P3D-Text/P3D-Image 五个切片）；RealCADBench（1,745 任务，含 Text/2D Drawing/Real Picture/Rendered Image 四个切片）。
- **评估指标**：Executability（成功率）、Solid IoU、Surface IoU、Judge Score（Kimi-K2.6 无 GT judge），profile average 为四项等权平均（式 1）。
- **零件级 SOTA**：VisCAD-M1 平均 0.5540（Pub 0.5596 / Real 0.5485），领先 Gemini-3.1-Pro（0.5496）；parallel-tts 后达 0.5797（Pub 0.5833 / Real 0.5761），相对 SOTA 提升约 5.5%。
- **Open-weight 基线全面落后**：Qwen3-VL-8B（0.2680）、Qwen3-VL-32B（0.3199）、Qwen3.8-27B（0.4759）。
- **各切片差异**：VisCAD-M1 在 Solid/Surface IoU 上优势明显（BenchCAD 0.4422、CADBench 0.5484、正交重建 0.4604），但在 Judge 分数上部分切片低于 GPT-5.5/Gemini；真实照片切片 Executability 达 0.9877，显著高于竞品（Gemini 0.9085、GPT-5.5 0.9208）；2D 工程图 Judge 分仍落后 Gemini（69.13 vs 79.08）。
- **装配级**：50 实例子集上 VisCAD-H1 judge score 85.0，Codex 68.0，Claude Code 52.0，有效样本数分别为 49/49/50。
- **消融**：Mid-training 在各切片 IoU 上提升约 0.01、Judge 提升约 5 分；Robust SFT 中 sample_k=3（3 条候选）优于 sample_k=1。

## 相关工作脉络
- **DeepCAD / SketchGraphs / Text2CAD / CAD-Recode**：从文本/草图/点云恢复参数化命令序列，本文在输入模态多样性（加入照片/工程图）与规模上显著扩展。
- **BRepNet / BrepGen / CAD-Llama / ParaCAD-RL**：直接以 B-Rep 或 RL 为训练目标，本文坚持 program-native 范式，强调可执行 DSL 程序而非隐式几何表示。
- **OpenECAD / CadVLM / ChatCAD / PICASSO**：拓展输入模态的工作，但输出仍为单零件且工程图/产品照片覆盖有限，本文在同一接口下覆盖全谱系意图。
- **ABC / Fusion 360 Gallery / BenchCAD / CADBench / P3D-Bench**：现有基准以 CAD-native 或合成数据为主，本文引入 RealCADBench 衡量真实工业意图（工厂自动化产品）下的表现。
- **通用 coding harness（Codex / Claude Code）**：可执行-观察-重写，但缺乏 BoM 结构化分解、逐零件 IR 门控与 mating 推理，本文证明领域特定 harness 在装配任务上显著占优。
- **Gencad-SelfRepairing / Gencad-3d**：强调程序自检与合成数据平衡，本文通过 RSI-based 程序搜索和递归自改进获取高质量监督信号，路线更系统。

## 局限性与未来方向
- ** Judge 分数与几何指标不完全一致**：部分切片中 VisCAD-M1 几何重叠优但 judge 分落后，说明视觉语义理解仍有短板（如 2D 工程图 judge 差距较大）。
- **IR-to-API 翻译依赖前端 VLM 能力**：H1 的最后一环仍需 VLM 将 IR 翻译为后端 DSL，若翻译错误会导致上游合理装配计划无法落地。
- **TTS 增加推理开销**：parallel-tts 需生成 k 条 rollout 再 rerank，计算成本随 k 线性增长，k=16 虽效果好但代价更高。
- **真实工业场景复杂度**：当前装配评估仅 50 实例，且未覆盖大规模多部件（>10 零件）或含运动学链的复杂装配。
- **未来方向**：论文展望将 M1 与 H1 整合为单一 agent-native 闭环，支持从意图到装配的全流程端到端生成。

## 研究启发与可借鉴点
- **RSI-based 程序搜索构建高质量监督**：通过自改进循环迭代优化 CAD 程序，可有效缓解 golden program 稀缺问题，该思路可迁移至其他程序生成领域（如脚本、SQL）。
- **多阶段训练的渐进式能力构建**：mid-training（语法 grounding）→ robust SFT（域泛化）→ continual SFT（精细保真）的三段式训练策略，对任何代码生成模型均有借鉴价值。
- **DSL-agnostic 中间表示解耦规划与实现**：将装配推理与后端语法隔离，使同一计划可在多种 CAD 环境间迁移，该设计模式适用于任何多后端工具生成任务。
- **测试时扩展中的判别式选择优于生成式精化**：序列迭代精化无效而并行 rerank 有效，提示在复杂程序生成任务中，"筛选最优"比"逐步修正"更可靠，可作为 TTS 设计的通用经验。
- **领域特定 harness 在结构化生成任务中的压倒性优势**：即使使用相同前端 VLM，定制化 workflow（BoM→IR→mating→placement）远超通用 coding agent，说明任务结构先验的重要性。

## 关键术语表
- **VisCAD-M1**：27B 多模态零件级 CAD 生成基础模型，支持文本/工程图/照片/渲染图到 FreeCAD Python 程序的映射。
- **VisCAD-H1**：基于领域特定 harness 的装配级生成系统，通过四阶段 gated 工作流实现结构化装配规划。
- **PubCADBench**：聚合 BenchCAD、CADBench、正交重建、P3D-Text、P3D-Image 五个公开基准的零件级评测集合（1,100 任务）。
- **RealCADBench**：面向真实工业产品意图的零件级评测集合，含 Text/2D Drawing/Real Picture/Rendered Image 四切片共 1,745 任务。
- **Parallel test-time scaling（parallel-tts）**：对同一输入生成多条 rollout 并由模型自身进行 listwise rerank 选择的测试时优化技术。
- **Intermediate Representation（IR）**：与 CAD DSL 无关的结构化中间表示，编码 BoM、零件几何、接口约束与全局放置，支持跨后端移植。
- **Bill of Materials（BoM）**：装配体中各零件的名称、数量及层级关系清单，是装配规划的起点。
- **Profile Average**： Executability、Solid IoU、Surface IoU 与归一化 Judge Score 四项等权均值，作为零件级综合评测指标。

## 可复现要素
- **数据集**：PubCADBench（公开基准聚合）、RealCADBench（作者发布的工业 benchmark，论文未声明开源状态）；训练数据包含公开 CadQuery 程序库、电商产品图像及采购工业资产（STEP/STL/工程图），未声明完全开源。
- **代码**：论文未明确说明代码是否开源。
- **权重**：VisCAD-M1 为 27B 模型，论文未说明是否开源；基线模型引用 GPT-5.4/5.5、Gemini-3.1-Pro、Claude-Opus-4.8、Kimi-K3、Doubao-Seed-2.0-pro、Qwen3-VL-8/32B、Qwen3.8-27B。
- **关键超参**：mid-training 约 100 万程序；robust SFT 约 15 万图像，sample_k=3；continual SFT 约 1 万精选样本；parallel-tts 中 k=8 和 k=16；voxel 分辨率 R=96，padding=2%。
