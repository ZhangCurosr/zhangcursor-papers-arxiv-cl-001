---
title: "REALCADBENCH-BENCHMARKING-PARAMETRIC-CAD-MODELING-FROM-INDUS"
source: https://arxiv.org/pdf/2609.03773v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-07 20:20:23"
field: "CAD 生成与评测"
keywords: ["Parametric CAD", "CAD Benchmark", "Industrial Design", "Large Language Model", "Geometric Deep Learning", "Multi-modal Generation"]
innovations: ["提出 RealCADBench 多模态参数化 CAD 基准（12,632 任务/19 类别/四项互补指标）", "引入 rubric-based 视觉-语义一致性 Judge 独立于参考模型评估产品身份", "系统对比前沿大模型与 Agent 在真实工业 CAD 任务上的四项指标 trade-off"]
benchmarks: ["RealCADBench", "RCB-Assm25", "BenchCAD", "CADGenBench"]
---

# 论文速读：REALCADBENCH — BENCHMARKING PARAMETRIC CAD MODELING FROM INDUSTRIAL DESIGN INTENTS

## 一句话总结
本文提出了 **RealCADBench**，一个面向真实工业设计意图的参数化 CAD 生成基准，包含来自 19 个工业自动化类别的 12,632 个任务，覆盖文本、2D 工程图、实物照片和渲染图四种输入模态，以及 Part 和 Assembly 两个建模赛道；通过四项互补指标（可执行性、Solid IoU、Surface IoU 和基于 rubric 的视觉-语义一致性 Judge）对九个前沿大模型和两个 Agent 进行了系统评测，揭示了现有 benchmark 中常被忽视的关键问题：可执行性高不代表几何或语义质量好，不同输入模态在不同指标上呈现不同的难度顺序。

## 研究问题与动机
1. **单一指标不足以评估真实工业场景下的参数化 CAD 建模**。现有基准多依赖可执行性或 IoU 单一维度，但语法正确程序可能遗漏安装孔，视觉上合理的装配可能放置错误零件。
2. **现有基准偏向合成数据或 CAD 原生设置**。ABC、Fusion 360 Gallery、DeepCAD、SketchGraphs 等提供大量几何语料，但未在真实工业设计意图下分离可执行性、IoU 和语义一致性三项独立结果；Text2CAD、CAD-Recode 等工作评估范围仍受限。
3. **输入模态多样且信息不完整**。真实工业输入（文本描述、2D 工程图、实物照片、渲染图）各自揭示不同且常不完备的几何视图，需评估模型能否产出有效产物、恢复底层形状并保留目标零件/装配的可见身份。
4. **前沿模型与 Agent 在扩展评价指标后的表现分化尚未被系统研究**。多数基准仅评测独立模型，未对 Agent 配置（含工具调用与推理时修复）进行同等对比。

## 核心贡献（创新点）
1. **首个面向真实工业设计意图的多模态参数化 CAD 基准**：收集 12,632 个任务、覆盖 19 个工业自动化类别，统一使用 FreeCAD API Python 输出契约，区别于 ABC、Fusion 360 Gallery 等以合成/程序化序列为主的语料库。
2. **四项互补评测指标体系（超越可执行性与 IoU）**：提出可执行性 + Solid IoU + Surface IoU + 基于 rubric 的视觉-语义一致性 Judge 的综合评估协议，其中 Judge 独立于参考模型、仅比较渲染视图与原始输入，弥补 IoU 无法捕捉产品身份和装配关系的不足，区别于 BenchCAD、P3D-Bench 等仅关注几何或参数的基准。
3. **前沿模型与 Agent 在真实 CAD 任务上的系统横向评测**：对 9 个独立大模型（Part Track，1,745 任务）和 2 个通用 Agent（RCB-Assm25，25 个装配任务）进行对比，揭示"无任何模型同时领先四项指标"的结论，区别于 CADGenBench 仅报告单一 Agent 或与参考模型直接比对的做法。
4. **揭示"可执行≠高质量"的普遍现象及典型失败模式**：发现缺失细结构、零件身份丢失、装配位置错误等反复出现的失败模式，为后续研究提供了清晰的评测维度分离视角。

## 方法详解
- **任务定义**：每个任务由输入 $x_i$、参考 3D 模型 $C_i^\star$ 和输入模态/类别元数据定义；模型输出调用 FreeCAD API 的 Python 程序 $p_i$，由共享运行时执行并导出 `result.stl` 作为最终 3D 模型，评分仅基于导出模型而非源代码或构造路径。
- **两条评测赛道**：
  - **Multimodal Part CAD Modeling Track**：12,465 个任务（568 文本、236 2D 工程图、11,288 实物照片、373 渲染图）。
  - **Image-based Assembly CAD Modeling Track**：167 个实物照片装配任务，报告中使用分层 25 任务子集 RCB-Assm25。
- **四项评测指标**：
  - **Executability（可执行性）** $e_i = \mathbb{I}[p_i \text{ 执行并产出合法 } \hat{C}_i]$：涵盖语法错误、运行失败、缺失输出、空几何和非法模型。
  - **Solid IoU**：确定性 signed-PCA 对齐（消除姿态和均匀缩放差异）后，计算填充体素占用率的交并比 $g_i^{\text{solid}} = \frac{|V_{\text{filled}}(\tilde{C}_i) \cap V_{\text{filled}}(C_i^\star)|}{|V_{\text{filled}}(\tilde{C}_i) \cup V_{\text{filled}}(C_i^\star)|}$，分辨率 $R=96$，padding=2%。
  - **Surface IoU**：在相同网格上计算边界体素占用率的交并比 $g_i^{\text{surface}} = \frac{|V_{\text{boundary}}(\tilde{C}_i) \cap V_{\text{boundary}}(C_i^\star)|}{|V_{\text{boundary}}(\tilde{C}_i) \cup V_{\text{boundary}}(C_i^\star)|}$，对薄结构和局部细节更敏感。
  - **Visual-Semantic Identity Judge**：基于 Kimi K2.6 冻结的 rubric 评分器，比较导出模型的渲染视图与原始输入（对文本任务则使用对应的实物照片证据），**从不查看参考模型**；Part rubric 侧重身份与显著特征（P1 几何实现质量 65%、P2 功能设计质量 35%）；Assembly rubric 分 Q（零件几何质量）、F（装配准确性）、D（系统设计质量）三个维度。
- **Profile Average（综合平均）**：$P_c = \frac{1}{4}(E_c + G_c^{\text{solid}} + G_c^{\text{surface}} + J_c/100)$，用于报告汇总；Part Track 的 regime-balanced PA 为四个输入模态 PA 的宏平均。
- **基准参考构建**：对渲染图任务直接使用验证过的 STEP 源文件导出；对文本/2D 工程图/实物照片任务，由候选方法提出参考 3D 模型，经 Gemini 3 Pro 筛选和 5 名 CAD 专家评分（仅约 17% 得分≥8 的被采纳），文本任务由 Gemini 3.0 Pro 从对应实物照片生成。

## 实验与结果
- **数据集规模**：基准 12,632 个任务（19 个工业自动化类别），报告切片 1,770 个任务（1,745 Part 任务 + 25 装配任务 RCB-Assm25）。
- **评测模型**：9 个独立前沿大模型（Qwen3-VL-8B/32B、Qwen3.8-27B、Claude Opus 4.8、Gemini 3.1 Pro Preview、GPT-5.4、GPT-5.5、Kimi K3、Doubao Seed 2.0 Pro）+ 2 个 Agent（Codex+GPT-5.5、Claude Code+Claude Opus 4.8）。
- **Part Track 主要结果（Table 4，regime-balanced）**：
  - **Gemini 3.1 Pro Preview** 以 PA=**0.5507** 综合最高，执行率 0.8825，Solid IoU=0.4291，Surface IoU=0.1578，Judge=73.33。
  - **GPT-5.5** 执行率最高 **0.9311**，Judge=73.79（最高），但 Surface IoU=0.1393 并非最优。
  - **Kimi K3** Solid IoU=0.4481（最高）、Surface IoU=0.1657（最高），但执行率仅 0.3422。
  - 四个指标**无单一模型同时领先**：执行率最高 ≠ IoU 最高 ≠ PA 最高。
  - 四种输入模态难度顺序不一致：渲染图在最上面四项指标均领先（Exec=0.8123，Solid IoU=0.5379，Surface IoU=0.2165，Judge=75.80）；其余三种模态（文本、2D 工程图、实物照片）在不同指标下排名发生变化。
- **RCB-Assm25 装配评测结果（Table 5）**：
  - **Codex+GPT-5.5** 执行率 1.0000（最高），Solid IoU=0.2817（最高），Surface IoU=0.1161（最高），但 Judge=69.35（低于 GPT-5.5 的 76.33，下降 6.98 pp）。
  - 相对独立 GPT-5.5：执行率提升 +0.1600，Solid IoU 提升 +0.0714，Surface IoU 提升 +0.0190，但 Judge 下降 6.98 pp。
  - 装配 Solid IoU（0.2290）明显低于实物照片 Part（0.4071），差距 0.1781。
  - **GPT-5.5 独立模型是 Judge 领跑者**。
- **核心结论**：执行成功不等于几何质量好（Surface IoU 普遍偏低，最高仅 0.2165），几何质量高不等于视觉语义一致（Judge 与 IoU 相关性弱），Agent 的工具调用和迭代修复能改善执行和几何但未必改善产品身份。

## 相关工作脉络
1. **ABC / DeepCAD / Fusion 360 Gallery / SketchGraphs**：提供大规模 CAD 几何和构造历史语料，但不评估真实工业设计意图下的可执行性、IoU 和语义一致性三者分离问题；RealCADBench 在此基础上引入真实工业输入模态和四项指标体系。
2. **Text2CAD (Khan et al., 2024)**：语言条件化 CAD 生成，仅支持文本输入和 Part Track，无执行/语义评测；RealCADBench 扩展至多模态和 Assembly 赛道。
3. **CAD-Recode (Rukhovich et al., 2025)**：从点云恢复可执行代码，但仅 Part Track、单模态输入，且无视觉语义一致性评价；RealCADBench 的多模态覆盖和 Judge 设计显著超越。
4. **BenchCAD (Zhang et al., 2026) / P3D-Bench (Yang et al., 2026) / CADEngBench (Singh et al., 2026)**：引入语义或工程标准，但主要面向合成/CAD 原生任务，评测独立模型而非 Agent；RealCADBench 聚焦真实工业意图并同时对独立模型和 Agent 进行系统性对比。
5. **CADGenBench (Hugging Face, 2026)**：支持 2D 工程图和 CAD 编辑输入、可执行/可编辑输出、Agent 评测；但与 RealCADBench 的关键区别在于后者覆盖 19 个真实工业自动化类别、使用 FreeCAD 统一输出契约并引入独立的视觉-语义一致性 Judge。
6. **Zero-to-CAD (Ataei et al., 2026)**：推理时执行修复，强调 runtime feedback 和 tool use；RealCADBench 通过与独立模型的配对对比，明确界定 Agent 和基础模型在四项指标上的不同 trade-off。

## 局限性与未来方向
- **当前仅报告小规模评估切片**：Part Track 仅报告 1,745/12,632 个任务，Assembly Track 仅报告 25/167 个任务；全文明确承认将逐步发布完整基准。
- **参考模型构建依赖专家人工审核**：约 83% 的候选参考模型因评分不足被丢弃，构建成本较高，且对无配对供应商 CAD 的任务（文本/2D 工程图/实物照片）引入潜在的参考偏差。
- **Judge 虽经专家审阅但仍为自动化评分**：Kimi K2.6 Judge 的跨数据集一致性检查显示 rank 一致性良好（BenchCAD 上 13-14/15 pair 一致），但存在最大 2.02 分的分数波动。
- **Agent 对比未控制计算预算**：Codex 均值 12.1 min、Claude Code 均值 31.6 min，token 使用量差异大，直接比较公平性受限。
- **未来方向**：（1）向完整基准扩展；（2）对比更多专用 AI-for-CAD 算法与前沿大模型；（3）研究如何改善细结构保留和装配关系推理等反复出现的失败模式。

## 研究启发与可借鉴点
1. **四项互补指标分离评测范式**：将"能否运行"（Executability）、"几何重叠度"（Solid/Surface IoU）、"产品身份一致性"（Judge）作为独立维度报告，避免单一指标掩盖关键失败模式，可直接迁移至任何程序化生成任务（如代码生成、3D 打印路径规划）的评测设计。
2. **Rubric-based 视觉-语义一致性 Judge 的设计**：使用冻结 LLM 作为 Judge、直接比较渲染视图与原始输入（而非与参考模型比较）、结合 CAD 专家审阅迭代 prompt，这一流程可迁移至其他几何/视觉生成任务的自动化质量评估。
3. **输入模态难度顺序非单调的发现**：同一模型在不同输入模态下的性能排名随指标类型变化（执行率→IoU→Judge 的顺序不同），提示评测设计应避免跨模态简单平均，而应分别报告各模态 profile，这对多模态生成任务（如图像到 3D、文本到程序）的评测有直接参考价值。
4. **Agent 作为完整方法而非基础模型包装**：将 Agent 与配对独立模型在同一任务集、记录预算下进行对比，明确区分"执行修复能力"与"语义一致性保持能力"，为 Agent 评测提供了严谨的方法学范式。
5. **signed-PCA 确定性对齐 + 双分辨率 IoU 设计**：使用 signed-PCA 消除姿态和均匀缩放差异后再计算体素级 IoU（volume + boundary 双通道），既保证了度量不变性又兼顾了整体形状和薄结构评估，可直接复用到其他 CAD/3D 形状生成 benchmark。

## 关键术语表
**Parametric CAD Modeling**：参数化 CAD 建模，通过调用 CAD API（如 FreeCAD Python API）的程序化方式生成具有构造历史的 3D 模型。
**Executability**：可执行性，模型输出的 FreeCAD Python 程序能否在共享运行时中成功执行并导出合法 3D 模型（.stl）。
**Solid IoU**：实体 IoU，在确定性 signed-PCA 对齐后，计算预测模型与参考模型填充体素占用率的交并比，衡量整体体积恢复程度。
**Surface IoU**：表面 IoU，在相同对齐网格上计算边界体素占用率的交并比，对薄结构和局部细节更敏感。
**Visual-Semantic Identity Judge**：视觉-语义一致性裁判，基于冻结 LLM（Kimi K2.6）的 rubric 评分器，直接比较导出模型的渲染视图与原始输入（不查看参考模型），评估产品身份、显著特征和装配关系。
**Profile Average (PA)**：综合平均，$P_c = \frac{1}{4}(E_c + G_c^{\text{solid}} + G_c^{\text{surface}} + J_c/100)$，对四项指标等权平均的汇总分数。
**RCB-Assm25**：RealCADBench Assembly 25，从 167 个装配任务中按类别、组件数、可见性和几何复杂度分层选取的 25 任务评测子集。
**FreeCAD API Python**：FreeCAD 的参数化建模 Python API，所有基准方法统一输出的程序格式，由共享运行时执行后导出 .stl 文件。

## 可复现要素
- **数据集**：RealCADBench 共 12,632 个任务（报告切片 1,770 个），论文声明将公开完整基准（Section 6 Conclusion）。
- **代码**：共享运行时和评估代码随基准发布（Appendix A.2）。
- **权重**：评测使用的模型（Qwen3-VL-8B/32B、Kimi K3 等开源模型权重公开；GPT-5.5、Claude Opus 4.8 等闭源模型通过 API 调用）。
- **关键超参**：体素网格分辨率 $R=96$，padding=2%；Judge 使用 Kimi K2.6 冻结版本；FreeCAD 1.1.1 + Python 3.11。
- **环境约束**：FreeCAD API Python 输出、mm 单位、至少一个非空 solid（Part）/多个 solids（Assembly）。
- **未提及**：具体的推理温度、top-p 等生成超参数（论文说明使用 provider default）。
