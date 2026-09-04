---
title: "JUDGESTEALER-Extracting-LLM-Judging-Capabilities-across-Eval"
source: https://arxiv.org/pdf/2608.26982v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 19:27:05"
field: "LLM安全与隐私"
keywords: ["模型提取攻击", "LLM-as-a-Judge", "黑盒提取", "多协议评估", "主动学习", "模型 stealing"]
innovations: ["首个利用跨协议一致性实现黑盒LLM judge多协议提取的框架", "融合语义多样性、预测不确定性和评判偏差的动态样本选择机制", "自适应高斯平滑评分目标与多协议联合巩固防遗忘策略"]
benchmarks: ["Alpaca", "GPT4All", "GPT-5.4", "Claude Sonnet 4.5", "Qwen3-235B-A22B", "DeepSeek V4 Pro", "UniRRM"]
---

# 论文速读：JUDGESTEALER: Extracting LLM Judging Capabilities across Evaluation Protocols

## 一句话总结
本文提出了 **JUDGESTEALER**，首个面向黑盒 LLM judge 的多协议模型提取框架。该方法利用点wise评分与pairwise/listwise评判间的高度一致性，仅需查询 victim 获取点wise分数，即可将监督信号跨协议转换，以极低的查询预算复制 LLM 的评判能力。

## 研究问题与动机
1. **LLM judge 能力具有商业价值**：LLM-as-a-Judge、Reward Model 及安全评分等应用依赖模型评判能力，服务提供商通过黑盒 API 暴露该能力，但面临被窃取风险。
2. **现有提取方法存在不足**：① 多针对传统 CNN，专门面向 LLM judge 的研究几乎空白；② 多数方法假设已知 victim 架构或需访问 logits，不适用于真实 API 场景；③ 仅在单一协议下评估，跨协议协同利用不足；④ 低查询预算下性能显著退化。
3. **跨协议一致性被低估**：实验发现，不同评估协议（点wise评分、pairwise比较、listwise排序）之间存在高达 88%–95% 的一致性，说明其共享同一底层评判准则，为跨协议监督复用提供了可行性。
4. **查询预算受限**：victim API 通常有调用频率和成本限制，如何在有限 query budget 下最大化提取效率是关键挑战。

## 核心贡献（创新点）
1. **首个跨协议黑盒 LLM judge 提取框架**：提出 JUDGESTEALER，首次实现 pointwise/pairwise/listwise 三种评估协议下的统一提取，与仅关注单一任务或组件的前序工作形成本质区别。
2. **动态高信息量样本选择机制**：融合语义多样性（semantic diversity）、预测不确定性（predictive uncertainty）和潜在评判偏差（judge bias）三路信号构建综合选择分数，区别于 Vanilla 随机采样或仅依赖单一信息的基线。
3. **跨协议监督转换与多协议联合巩固**：利用点wise分数无损推导pairwise偏好与listwise排序，避免额外victim查询；并通过多协议联合训练防止灾难性遗忘，优于逐协议顺序微调的策略。
4. **自适应高斯平滑评分目标**：将硬 one-hot 标签替换为以预测分数为中心的可训练高斯分布，更好地保留评分的序数结构，优于固定平滑强度或无平滑设置。
5. **全面的防御鲁棒性验证**：系统测试对抗异常检测、抗蒸馏扰动和所有权追踪（watermark）三种代表性防御，证明方法在现实安全场景下的实用性。

## 方法详解
### 整体框架
JUDGESTEALER 分为两阶段：
- **Stage I**：动态选择高信息量候选实例 → 向 victim 提交点wise评分查询 → 训练初始 surrogate。
- **Stage II**：将点wise监督跨协议转换为 pairwise/listwise 数据 → 依次适配各协议 → 多协议联合巩固（避免遗忘）。

### Stage I：点wise评分提取
1. **样本选择（4.2.1）**：对每个候选实例 $G = \{(q, a_i)\}_{i=1}^n$，计算三路信号：
   - **语义多样性** $D(G)$：用文本嵌入模型的 CLS 向量表征实例，计算与已选实例集合的最小余弦距离，并结合局部密度估计 $\rho(G)$ 排除孤立异常点。
   - **预测不确定性** $U(G)$：用当前 surrogate $M_\Phi$ 对 $G$ 中所有响应的评分分布求归一化熵，越高越具信息量。
   - **潜在评判偏差** $B(G)$：融合两种常见偏差：
     - *冗长性偏差* $B_v$：在响应前拼接中性前缀，测量 surrogate 对长度变化的敏感度。
     - *位置偏差* $B_p$：交换响应顺序，用 KL 散度衡量 surrogate 输出分布变化。
   - **综合得分**：$\Gamma(G) = \lambda_1 D(G) + \lambda_2 U(G) + \lambda_3 B(G)$，每轮取 Top-K。

2. **Victim 评分**：将选定实例中的每个 $(q, a_i)$ 提交至 victim 的 pointwise prompt $\mathsf{P}_s$，获得分数 $y_i \in \mathcal{Y}$。

3. **Surrogate 训练**：采用**自适应高斯平滑**，将 one-hot 标签 $\mathbf{e}_y$ 平滑为 $\tilde{\mathbf{e}}_y = (1-\lambda)\mathbf{e}_y + \lambda \mathbf{g}_y$（$\mathbf{g}_y$ 为离散高斯分布），平滑系数 $\lambda$ 可训练。损失函数：
$$\mathcal{L}_{\mathrm{point}}(\Phi) = \frac{1}{|S_t|} \sum_{(q,a,y) \in S_t} \mathrm{CE}(\tilde{\mathbf{e}}_y, \mathbf{p})$$

### Stage II：多协议扩展（4.3）
1. **跨协议转换（无需额外 victim 查询）**：
   - **Pairwise**：从同一实例中选响应对 $(a_j, a_k)$，按分数大小推断偏好 $y_{j,k} \in \{a_j \succ a_k, a_j \prec a_k, a_j \sim a_k\}$。
   - **Listwise**：采样 $n_l$ 个响应，按分数降序排列得到排序标签。
2. **渐进式训练**：分别在 $\mathcal{C}$（pairwise）和 $\mathcal{R}$（listwise）上微调 surrogate，损失函数为标准交叉熵。
3. **多协议巩固**：构建联合训练集 $\mathcal{T} = S \cup \mathcal{C} \cup \mathcal{R}$，加权损失：
$$\mathcal{L}_{\mathrm{con}}(\Phi) = \alpha_{\mathrm{point}} \mathcal{L}_{\mathrm{point}} + \alpha_{\mathrm{pair}} \mathcal{L}_{\mathrm{pair}} + \alpha_{\mathrm{list}} \mathcal{L}_{\mathrm{list}}$$
其中 $\alpha$ 为各协议样本占比，防止后续协议覆盖早期点wise行为。

## 实验与结果
### 数据集与模型
- **数据集**：Alpaca（52K）和 GPT4All（437K），各采样 2K–18K 条指令构建候选实例。
- **Victim 模型**：GPT-5.4、Claude Sonnet 4.5（专有）；Qwen3-235B-A22B、DeepSeek V4 Pro、UniRRM（开源）共5个。
- **Surrogate 模型**：主实验用 Llama-3.2-1B-Instruct、Qwen3-1.7B；规模实验覆盖 Qwen3-0.6B 至 Qwen3-32B。
- **基线**：Vanilla（随机采样）、LoRD（基于偏好的强化提取）、Lion（对抗蒸馏）、Proxy-KD（中间白盒代理蒸馏）。

### 主要结果（Table 2）
- JUDGESTEALER 在 **480 个指标中有 475 个** 超越所有基线。
- **GPT-5.4 + Qwen3-1.7B + Alpaca**：Pointwise $\mathrm{Acc_{\pm1}} = 0.5905$（提升 +0.1040 vs 最强基线 Proxy-KD 的 0.4865）；Pairwise $\mathrm{Acc} = 0.7707$；Listwise $\mathrm{Acc} = 0.6345$。
- **GPT-4All + GPT-5.4 + Qwen3-1.7B**：Pointwise $\mathrm{Acc_{\pm1}} = 0.6100$，Pairwise $\mathrm{Acc} = 0.7833$，MAE 降至 1.6111。
- **DeepSeek V4 Pro + GPT4All**：Pointwise $\mathrm{Acc}$ 相对提升 93%（0.2800 → 0.5400），Pairwise $\mathrm{Acc} = 0.8700$，Listwise $\mathrm{Acc} = 0.7167$。
- 各协议平均提升：Pointwise +0.094，Pairwise +0.064，Listwise +0.024。

### 其他实验
- **Surrogate 规模**（0.6B–32B）：性能随规模递增，pairwise 在 4B 后趋于饱和。
- **训练策略**：Full fine-tuning 与 LoRA 性能相当，LoRA 计算成本更低。
- **CoT 推理设置**（Table 3）：在 victim 提供推理链的情况下，JUDGESTEALER 仍显著优于 Vanilla（如 Alpaca Qwen3-1.7B 点wise $\mathrm{Acc_{\pm1}}$ 提升 +0.0604）。
- **Reward Model 实验**（Table 4，UniRRM）：Pointwise MAE 0.6354，Pairwise Acc 0.8356，Listwise $\mathrm{Acc@Top}$ 0.7967，超越基线 1.9%–7.5%。
- **Query Budget**（Figure 4）：仅 0.5% 预算时已实现 Pointwise $\mathrm{Acc_{\pm1}} \approx 50\%$、Pairwise $\mathrm{Acc} > 70\%$。

### 最强结果汇总
| 协议 | 最佳 Acc/指标 |
|---|---|
| Pointwise $\mathrm{Acc_{\pm1}}$ | 73.3%（GPT-4All + Qwen3-1.7B）|
| Pairwise Acc | 87.0%（DeepSeek V4 Pro + GPT4All）|
| Listwise Acc | 71.6%（DeepSeek V4 Pro + GPT4All）|

## 相关工作脉络
1. **Model Leeching [2]**：最早针对 LLM 的功能提取工作，通过任务特定 prompt 蒸馏 ChatGPT-3.5-Turbo 行为；但仅关注单一任务，未考虑多协议协同。
2. **LoRD [28]**：引入偏好引导的强化学习提取，通过构造 neighborhood 样本最大化偏好差距；但同样针对通用 LLM，未利用跨协议一致性。
3. **Lion [16]**：对抗性蒸馏框架，迭代生成挑战性指令提升模仿效果；属于通用提取范式，未针对 judge 能力设计。
4. **Proxy-KD [5]**：利用中间白盒 proxy 对齐黑盒 soft output；需要 logits 访问或白盒代理，不适用于纯黑盒 API 场景。
5. **KnockoffNet / Activethief / Inversenet** 等传统 DNN 提取方法：针对图像分类网络，查询效率和协议多样性与本文场景不兼容。
6. **PRADA [17] / SEAT [48]**：查询检测防御，基于查询距离分布异常检测提取行为；本文验证了 JUDGESTEALER 的查询模式可规避此类检测。
7. **ModelShield [36]**：黑盒水印防御，通过词表分区嵌入绿色 token；本文验证 watermark 难以传递至 surrogate。

## 局限性与未来方向
1. **依赖高质量响应池**：需要预先收集包含多样响应的多响应实例（本文使用 25 个 LLM 生成），若响应池质量不足会影响跨协议转换的信噪比。
2. **对弱 judge 效果有限**：Figure 8 显示，低能力 judge 的跨协议一致性较低（<80%），监督转换误差更大，攻击有效性随之下降。
3. **未全面评估所有防御**：仅测试了异常检测、抗蒸馏扰动和 watermark；对 rate limiting、API 身份认证等基础设施层防御未涉及。
4. **潜在伦理边界**：提取框架可能被用于绕过后门保护、安全评分等关键下游应用，需配套制定行业防护标准。
5. **未来方向**：探索跨语言/跨领域的 judge 提取；研究基于协议一致性的 judge 能力量化指标；开发更鲁棒的对抗性防御。

## 研究启发与可借鉴点
1. **跨协议监督复用思路可迁移**：若团队有其他多模态或多任务评估场景（如代码生成评测、多文档 QA），可借鉴"单一细粒度监督 → 多任务信号推导"的策略，降低查询成本。
2. **动态样本选择机制**：融合多样性、不确定性和偏差信号的三路评估框架，可作为通用主动学习/查询选择器应用于其他黑盒模型提取或 API 调用优化场景。
3. **自适应高斯平滑技巧**：将离散 label 平滑为序数分布的思想，可用于任意带有序关系的分类任务（如 Likert 量表预测、等级评分），提升 surrogate 泛化性。
4. **多协议联合巩固防遗忘**：按比例加权多任务 loss 的 consolidation 策略，可有效缓解多阶段微调中的灾难性遗忘，适用于任何需要分阶段引入新监督信号的迁移场景。
5. **防御鲁棒性验证框架**：本文系统性测试三类防御的方法学，可作为团队后续设计/评估提取防御的参考模板。

## 关键术语表
**Black-box model extraction**：仅通过 API 查询观察输入输出行为，无需访问模型参数或内部状态，从而构建功能等价的本地 surrogate 模型。
**Cross-protocol agreement**：不同评估协议（点wise/pairwise/listwise）下 victim judge 输出的一致性比例，本文实测高达 88%–95%。
**Catastrophic forgetting**：模型在顺序学习新任务时，原有知识被严重覆盖或遗忘的现象，本文通过多协议联合训练缓解。
**Adaptive score smoothing**：将 hard one-hot 目标分布替换为以预测分数为中心的可训练高斯分布，保留评分的序数邻域结构。
**Informative sample selection**：基于语义多样性、预测不确定性和潜在评判偏差的综合评分，动态筛选最有价值的查询实例。
**Multi-protocol consolidation**：将 pointwise/pairwise/listwise 三协议训练数据联合采样并加权优化，避免单一协议主导导致的遗忘。
**Chain-of-Thought (CoT) defense-aware extraction**：在 victim 提供推理链的复杂 prompt 设置下仍保持提取有效性的能力。

## 可复现要素
- **数据集**：Alpaca 和 GPT4All 为公开数据集；候选响应由 25 个开源/专有 LLM 生成（见 Table 7），预收集后加载模拟 victim 交互。
- **代码/权重**：论文声明将在 acceptance 后公开 artifact repository，包含 JUDGESTEALER 实现、prompt 模板、配置、随机种子和评估脚本；**不公开**商业 API 凭证和从专有 judge 蒸馏的 checkpoint。
- **关键超参**：
  - LoRA rank=8，scaling=16，dropout=0.05，learning rate=$1\times10^{-4}$（full fine-tuning 为 $1\times10^{-5}$）
  - 选择权重：$\lambda_1=1.0, \lambda_2=0.25, \lambda_3=1.0$
  - 每轮采样 $K=20$ 实例，候选子集大小 100，k近邻密度过滤（排除底部10%）
  - 高斯平滑 $\sigma=1.0$，初始 $\lambda=0.10$，学习率 $5\times10^{-6}$
  - 跨协议采样：$n_c = n_l = n_r = 3$
- **硬件**：384-core Intel Xeon 6972P + NVIDIA H200 NVL GPU，Ubuntu 22.04。
