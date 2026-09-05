---
title: "Reactivating-Test-Time-Scaling-for-Plane-Geometry-Problem-So"
source: https://arxiv.org/pdf/2608.30156v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 16:28:34"
field: "多模态几何推理"
keywords: ["平面几何问题求解", "测试时间扩展", "多模态大语言模型", "多轨迹合成", "感知增强训练", "神经符号推理"]
innovations: ["提出MTS框架将符号程序扩展为异构推理轨迹以重新激活TTS", "设计PA训练实现视觉到符号的显式grounding", "提出CG-MTE自适应性推理策略实现8倍采样成本降低"]
benchmarks: ["PGPS9K", "Geometry3K", "GeoQA"]
---

# 论文速读：Reactivating-Test-Time-Scaling-for-Plane-Geometry-Problem-So

## 一句话总结
本文针对平面几何问题（PGP）求解中测试时间扩展（TTS）失效的问题，提出了多轨迹合成（MTS）框架与感知增强（PA）训练方法，将刚性符号程序转化为异构推理轨迹并引入显式视觉 grounding，结合自适应推理策略 CG-MTE，成功在三个几何基准上重新激活了 TTS 的收益，同时以约 1/8 的采样成本达到高预算 self-consistency 的精度。

## 研究问题与动机
- **TTS 在平面几何求解中失效**：与一般数学推理不同，在 PGP 上用 self-consistency（SC）无法获得显著收益，采样轨迹难以收敛到多数一致答案。
- **推理多样性受限**：现有几何数据集依赖刚性符号程序（symbolic programs），这种紧凑的形式语言缺乏自然语言推导所需的探索空间，导致不同采样轨迹高度重复。
- **视觉感知与符号推理存在 gap**：模型常因误读图示意图（如将 101° 读为 104°）而产生错误推断，且缺乏推理前的显式视觉 grounding 步骤。
- **计算效率问题**：即便能扩展 TTS，朴素增加采样预算在简单问题上会造成冗余计算，需要高效的自适应性策略。

## 核心贡献（创新点）
1. **提出 MTS（Multi-Trace Synthesis）框架**：将每个符号程序扩展为四种异构推理轨迹（Program / PAL / CoT-Program / CoT-PAL），本质区别在于通过格式转换与 CoT 增强重新引入自然语言推导语境，从而恢复 TTS 所需的推理多样性。
2. **引入 PA（Perception-Augmented）训练**：在推理前强制模型先解析图为结构化语义子句（semantic clauses），本质区别在于将视觉 grounding 作为显式前置步骤，而非直接跳到符号推导。
3. **设计 CG-MTE（Consensus-Guided Multi-Trace Ensemble）自适应性推理策略**：按深度逐步检查跨轨迹共识，仅在分歧持续时才扩展采样，本质区别在于以共识停止为启发式规则而非固定预算，实现精度-计算 Pareto 优化。
4. **构建 MTS-All 数据集**：在 PGPS9K、Geometry3K、GeoQA 三个基准上分别构建 MTS-All 训练集，证明该方法跨模型规模均有效且可泛化。

## 方法详解

### 问题形式化
几何问题实例表示为 $\mathcal{I} = (D, Q, S, P, A)$，其中 $D$ 为图示意图，$Q$ 为文本问题，$S$ 为语义子句（显式几何关系/度量），$P$ 为符号解程序，$A$ 为数值答案。目标是训练 MLLM 先预测 $S$，再基于 $S$ 生成推理轨迹 $T$ 得到答案 $A$。

### 符号程序（Formal Solution Program）
程序步骤 $s_t = (\mathsf{op}_t, \mathsf{args}_t)$，算子集 $\mathcal{O}$ 包含 34 个几何定理/公理（如 Gougu 表示勾股定理）。操作数分三类：问题变量 N、过程变量 V、常量 C。程序实例化时将 N 替换为数值、C 替换为常量，使程序自包含。

### Multi-Trace Synthesis（MTS）
每种程序扩展为四种轨迹：
- **Program**：原始符号程序。
- **PAL**：通过规则翻译将程序转为可执行 Python 脚本，并经沙箱执行验证（相对误差 $\epsilon = 0.001$）。
- **CoT-PAL**：在 PAL 每步方程前注入自然语言几何理由（如"几何平均定理 $BD^2 = AB \times BC$"），采用 generate-and-verify 循环（最多 3 次）确保可执行性。
- **CoT-Program**：将程序逐步骤改写为带自然语言解释的结构化推导，推理过程中提取程序并执行获取答案。

### Perception-Augmented（PA）训练
将推理过程分为两步：（i）Perception：MLLM 解析图 $D$ 为语义子句 $S$；（ii）Reasoning：基于 $S$ 生成推理轨迹 $T$。联合损失函数：
$$\mathcal{L} = -\log P_\theta(S|D,Q) - \log P_\theta(T|S,D,Q) = \mathcal{L}_{\text{perc}} + \mathcal{L}_{\text{reason}}$$
两项均以标准交叉熵自回归优化。

### Inference：测试时间扩展策略
- **Self-Consistency（SC）**：对统一模型采样 K 条轨迹，执行后多数投票。
- **Multi-Trace Ensemble（MTE）**：对 V=4 种轨迹各采样 D=10 条，全部投票。
- **CG-MTE**：从浅层深度 d=1 开始逐步扩展，在每个深度 d 汇聚所有轨迹的前 d 个候选答案，若出现唯一众数则提前终止；否则 d ← d+1 继续。最高深度 D=10 仍未达成共识时回退到 CoT-PAL top-1。成本用平均采样数（ASN）衡量：$ASN_{CG} = \frac{V}{|\mathcal{T}|}\sum_{x\in\mathcal{T}} d_x$。

## 实验与结果

### 数据集与评估
- 三个基准：**PGPS9K**（训练 8021 / 测试 1000）、**Geometry3K**（训练 8432 / 测试 589）、**GeoQA**（训练 3485 / 测试 754）。GeoQA 无语义子句标注，需人工标注。
- MTS-All 数据量：PGPS9K-All（32.1K）、Geometry3K-All（33.7K）、GeoQA-All（13.9K）。
- 评估指标：答案准确率（相对误差 $\epsilon = 10^{-3}$ 内匹配即正确）。

### 主要结果（Table 1）
| 模型 | PGPS9K | Geometry3K | GeoQA |
|---|---|---|---|
| **Qwen3-VL-8B（PA+MTS）** | **71.2%**（+12.6 vs 基线） | **74.5%**（+9.1） | **67.2%**（+6.3） |
| Qwen2.5-VL-3B（PA+MTS） | 58.3%（+11.6） | 64.7%（+11.6） | 58.7%（+5.3） |
| Qwen2-VL-2B（PA+MTS） | 50.8%（+7.4） | 52.0%（+10.6） | 53.5%（+6.4） |

- 最强结果：**Qwen3-VL-8B** 在 PGPS9K 达 71.2%，超越最强专用求解器 LANS（66.7%）+4.5 点，超越 Qwen2.5-VL-72B（53.3%）+17.9 点。

### 测试时间扩展（Table 9 + Figure 5）
- SC@40（beam search）：PGPS9K 从 71.2% → **76.7%**，Geometry3K 从 74.5% → **80.7%**，GeoQA 从 67.2% → **76.1%**。
- CG-MTE：**76.0%**（PGPS9K），ASN ≈ 4.89，对比 SC@40（ASN=40），采样成本降低约 **8×**，精度损失仅 0.7 点。

### 消融实验（Table 2-4）
- w/o PA 训练：8B 模型 PGPS9K 下降 5.8 点，验证视觉 grounding 的重要性。
- w/o MTS（仅重复原程序）：8B 模型下降 3.6 点，验证推理多样性的贡献。
- 单轨迹 vs 多轨迹混合：CoT-PAL 单轨迹在 8B 上达 68.3%，多轨迹混合（PGPS9K-All）达 71.2%。
- Token 控制留一法（Table 4）：移除任一轨迹类型均导致性能下降，CoT-PAL 贡献最大（2B: -3.1, 3B: -3.0）。

### InternVL3.5 跨骨干验证（Table 7-8）
- InternVL3.5-8B：基线 40.2% → PA+MTS 48.5%（+8.3），SC@40 达 54.9%，验证方法对 MLLM 骨干的泛化性。

## 相关工作脉络
1. **神经符号几何求解器**（InterGPS、GeoDRL、Pi-GPS）：依赖精确解析，对感知误差敏感；本文不依赖外部符号引擎，而是让 MLLM 直接生成可执行轨迹。
2. **纯神经几何求解器**（NGS、Geoformer、PGPSNet、LANS、GeoX）：端到端程序生成，但推理路径单一；本文通过 MTS 扩展推理多样性。
3. **MLLM 几何推理**（G-LLaVA、Geouni）：侧重视觉-语言预训练与逻辑微调；本文聚焦于测试时间扩展的激活与效率优化。
4. **程序辅助推理**（PAL、AlphaGeometry）：AlphaGeometry 结合神经网络与形式化推理引擎达到 IMO 水平；本文专注于利用 MTS 扩展 TTS 在 PGP 中的效果。
5. **几何数据合成**（GeoFM、GeoThought、TR-CoT）：前者合成问题但与人类表达差异大，后者存在幻觉风险；本文从已有验证程序出发合成轨迹，保持符号严谨性。
6. **测试时间缩放**（Snell et al., 2025; Self-Consistency）：在一般数学推理中成功，但在 PGP 的符号程序范式下失效；本文揭示了失效原因并提出修复方案。

## 局限性与未来方向
- **依赖结构化标注**：MTS 需要可靠的形式化解程序，PA 训练需要语义子句标注，限制了在无此类标注数据集上的直接应用。
- **仅限于 2D 平面几何**：未验证方法在 3D 几何、物理或其他多模态推理任务上的泛化性。
- **CoT 理由的语义忠实度无法完全验证**：执行验证仅保证代码输出正确，不能保证自然语言理由与形式化推理步骤完全对齐，偶发幻觉仍可能存在。
- **未来方向**：扩展至更通用的推理种子、探索 3D/物理领域、改进 CoT 理由的验证机制。

## 研究启发与可借鉴点
1. **MTS 轨迹合成范式可迁移**：将形式化程序扩展为多种异构轨迹（代码+CoT）的策略，可推广至其他符号推理密集型领域（如代数、逻辑推理）。
2. **PA 训练的感知-推理两阶段设计**：先显式解析视觉输入为结构化语义再推理，可有效缓解视觉幻觉引发的连锁错误，适用于其他视觉符号任务。
3. **CG-MTE 的共识早停机制**：以跨轨迹共识为启发式停止规则，实现精度-计算 Pareto 优化，思路可迁移至其他需要 TTS 的场景（如数学证明、代码生成）。
4. **可执行验证保障轨迹质量**：generate-and-verify 循环（执行失败则返回错误信息让 MLLM 修复）是提升合成轨迹可靠性的有效手段。
5. **Beam search 在 TTS 中的优越性**：发现 beam search 随深度增加逐步激活 CoT 推理，比 temperature sampling 产生更多样化的答案分布，对解码策略选择有指导意义。

## 关键术语表
- **Test-Time Scaling（TTS）**：通过在推理阶段增加计算预算（采样更多轨迹）来提升模型推理性能的方法，典型代表为 self-consistency。
- **Multi-Trace Synthesis（MTS）**：将每条符号程序扩展为多种异构推理轨迹（程序/PAL/CoT-Program/CoT-PAL）的数据合成框架。
- **Perception-Augmented（PA）训练**：在两阶段推理中，先让模型解析图为结构化语义子句，再基于子句生成推理轨迹的训练范式。
- **Consensus-Guided Multi-Trace Ensemble（CG-MTE）**：从浅层深度开始逐步检查跨轨迹共识，一旦达成共识即提前终止采样的自适应性推理策略。
- **Program-Aided Language（PAL）**：让语言模型生成可执行 Python 脚本来求解问题的推理范式。
- **Semantic Clauses（语义子句）**：从图中提取的显式几何关系或度量描述（如 $AC \perp DB$、$AB = 3\sqrt{2}$）。
- **Average Sampling Number（ASN）**：衡量推理成本的指标，表示每个测试实例的平均采样数。
- **Same-Wrong 错误耦合**：不同轨迹在错误答案上的一致程度，用于评估多轨迹的错误相关性。

## 可复现要素
- **数据集**：PGPS9K、Geometry3K、GeoQA 均为公开数据集；MTS-All 合成数据已开源。
- **代码**：已开源，地址 https://github.com/Jason8Kang/ReTTS-PGPS。
- **权重**：论文未明确说明开源权重，需访问 GitHub 仓库确认。
- **关键超参**：训练 epoch=10，学习率 2B/3B 为 $2\times10^{-5}$、8B 为 $5\times10^{-6}$，cosine 调度+10% warmup，最大序列长度 1024 tokens；解码设 beam search/temperature sampling（T=0.9, top-p=0.9），K=40，V=4 种轨迹，D=10；执行验证容差 $\epsilon=0.001$。
- **训练硬件**：8× NVIDIA A100（80GB），DeepSpeed ZeRO-2。
