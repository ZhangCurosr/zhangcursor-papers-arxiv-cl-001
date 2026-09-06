---
title: "Predict-Don-t-Iterate-Efficient-Adaptive-Length-Infilling-fo"
source: https://arxiv.org/pdf/2609.02108v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 22:37:18"
field: "扩散语言模型推理优化"
keywords: ["Diffusion Language Model", "Adaptive-length Infilling", "Multi-slot Parallel Decoding", "Post-hoc Selection", "Predictive Length Estimation", "Bidirectional Attention"]
innovations: ["提出单次前向传播的轻量 MLP 探针直接预测 DLM 补全跨度长度，无需预设初始长度", "设计 slot-wise 并行解码与线性插值位置 ID，多候选在近似单候选代价下并行生成", "构造单次前向传播的严格因果 masked scoring probe 实现多候选后验联合评分"]
benchmarks: ["HumanEval-S/M", "MBPP-S/M", "Java (MultiPL-E)", "C/C++ (MultiPL-E)", "WikiText", "ArXiv Abstract"]
---

# 论文速读：Predict-Don't-Iterate-Efficient-Adaptive-Length-Infilling-for-Diffusion-Language-Models

## 一句话总结
本文提出 PILL（Probing-based InfiLling with preset-Length-free decoding），一种面向扩散语言模型（DLM）的自适应长度补全框架，通过单次前向传播预测掩码跨度长度、多槽并行解码和后置置信度评分三个阶段，在不需预设初始长度的前提下仅需额外两次前向传播即可完成变长补全，在八项代码/文本基准上显著超越最强基线且推理速度提升 1.82×。

## 研究问题与动机
- **DLM 的固定长度限制**：现有 DLM 在解码前必须预先指定掩码位置数量 $L$，但补全任务中正确长度 $L^*$ 未知且因例而异，模型必须在不知道内容的前提下决定长度，造成严重约束。
- **既有自适应方法对初始长度敏感**：CAL 等方法需要预设初始长度启动搜索，容易陷入局部最优；DAEDAL 仅支持追加式扩展，无法利用后缀条件。
- **既有方法推理开销大**：CAL 依赖多步去噪置信度反复搜索长度，DAEDAL 和 DreamOn 需在生成过程中插入长度变化操作，均引入大量额外前向传播。
- **补全质量对跨度长度高度敏感**：如图 1(a) 所示，长度偏差很小也会导致代码补全从正确变为错误，因此准确预测或覆盖真实长度至关重要。

## 核心贡献（创新点）
- **无预设长度的轻量探针**：通过冻结骨干 DLM 单次前向传播，读取 mask token 的双向隐藏状态并输入 MLP 直接预测目标长度，避免 CAL 等方法的迭代搜索和初始长度敏感问题。
- **多槽并行解码设计**：将预测长度 $\hat{L}$ 扩展为邻域候选集 $\{\hat{L}-r,\ldots,\hat{L}+r\}$，借助 slot-wise attention mask 与线性插值位置 ID，所有候选在同一序列中并行去噪，代价接近单次解码。
- **单次后验选择性评分**：针对 DLM 无法用 next-token likelihood 评分的特性，设计因果读取出入的 masked scoring probe 机制，在一次前向传播中联合衡量候选内部连贯性与续接后缀的自然度，实现端到端两额外 pass 的完整流程。
- **系统性实验验证**：在五个不同架构/家族的 DLM（LLaDA/Dream，dense/MoE）及八个代码与文本基准上全面验证，较最强基线 CAL 平均提升 +4.8 Pass@1（代码）和 +6.0 BLEU-2（文本），且推理速度提高 1.82×。

## 方法详解
PILL 分为三个串联阶段：

**Stage I — Length Probing（长度探针）**
- 在 prefix $P$ 和 suffix $S$ 之间插入单个 `[M]` token 构造序列 $x_{\text{probe}} = P \oplus [\mathsf{M}] \oplus S$，执行一次冻结 DLM 前向传播。
- 取第 $\ell$ 层的 mask 隐藏状态 $h_{[\mathsf{M}]}$，拼接前后各 4 个 token 的 mean-pooled 表示得到 $h = [\text{MeanPool}(h_{p-4:p-1});\; h_{[\mathsf{M}]};\; \text{MeanPool}(h_{s:s+3})] \in \mathbb{R}^{3d}$。
- 经 3 层 MLP $f_\phi$ 回归出 $\log \hat{L}$：$\hat{L} = \lfloor \exp(f_\phi(h)) \rceil$，以正整数形式输出，保障估计恒正并缓解长尾分布影响。

**Stage II — Multi-slot Parallel Decoding（多槽并行解码）**
- 将 $\hat{L}$ 扩展为半径 $r$ 的候选集 $\{\ell_1,\ldots,\ell_N\}$，$N=2r+1$。
- 序列布局为 $P \oplus [\mathsf{M}]^{\hat{L}} \oplus S \oplus [S_1]\cdots[S_N]$，其中每个槽 $S_j$ 含 $\ell_j$ 个 mask token，共享上下文 $P$、锚位 $[\mathsf{M}]^{\hat{L}}$、$S$。
- Slot-wise attention mask：各槽只 attending 自身与共享上下文，槽间互不干扰，实现并行去噪且 KV-cache 共享。
- **位置 ID 插值设计**（关键）：固定后缀起始位置 $p_S = p_P + \hat{L} + 1$，槽 $S_j$ 的第 $k$ 个 token 的位置 ID 线性插值于 $[p_P+1, p_S-1]$ 区间：$\text{pos}(S_j,k) = (p_P+1) + (k-1)\cdot(\hat{L}-1)/(\ell_j-1)$。该设计消除 padding 造成的位置空缺，避免 DLM 产生"欠生成"（under-generation）。

**Stage III — Post-hoc Selection（后置选择）**
- 候选无法用 DLM 自身预测分布评分（已 commit token 的分布非预测性），故构造 masked scoring probe：对每个候选 $C_j$，将其与少量后缀子集 $\tilde{S}$ 拼接为可见块 $v$，以 mask token 序列 $\mu$ 对齐 $v$ 的位置 ID。
- 设计严格因果注意力 mask：每个 probe $\mu_k$ 只能 attend 到 $P$ 与 $v_{<k}$，读取 $p_\theta(v_k|P,v_{<k})$，防止标签泄漏。
- 内部连贯性得分与续接后缀得分的凸组合：
$$j^\star = \arg\max_j \alpha\, s_j^{\text{in}} + (1-\alpha)\, s_j^{\text{suf}}$$
其中 $s_j^{\text{in}} = \frac{1}{\ell_j}\sum_{k=1}^{\ell_j}\log p_\theta(c_k|P,c_{<k})$，$s_j^{\text{suf}} = \frac{1}{m}\sum_{t=1}^m\log p_\theta(\tilde{s}_t|P,C_j,s_{<t})$，实践中 $\alpha=0.5$。
- 后缀采样包含 head tokens（最近过渡信号）与高注意力 token（远距离依赖），$m\in[4,8]$，总长度上限 32。N 个候选块通过 block-wise attention mask 打包至单次前向传播完成评分。

**全程仅引入两次额外前向传播**（探针一次、评分一次），多槽解码在几乎零额外代价下并行产出候选。

## 实验与结果
- **模型**：LLaDA-8B-Base/Instruct、LLaDA-MoE-Base、Dream-7B-Base、Dream-Coder-7B-Base（共 5 个，覆盖 dense/MoE、base/instruct、code/general 等差异）。
- **数据集（8 项）**：代码 HumanEval-S/M、MBPP-S/M、Java、C/C++；文本 WikiText、arXiv 摘要。均独立于探针训练集。
- **评估指标**：代码用 Pass@1，文本用 BLEU-2/ROUGE-L。
- **最强结果**：相较最强基线 CAL，PILL 在代码基准平均 +4.8 Pass@1、文本基准平均 +6.0 BLEU-2；在 LLaDA-8B-Base HumanEval-S 上达 71.35 Pass@1，超越 CAL 的 64.74。
- **效率**：PILL 额外开销仅 +7% wall-clock（vs Backbone），CAL 为 +94%、DAEDAL 为 +117%；PILL 比 CAL 快 1.82×，且只需 2 次额外前向传播（CAL 约 14 次）。
- **Oracle 上界**：PILL (Oracle) 达 81.90（Human-S），与 PILL (71.35) 的差距主要来自选择阶段而非候选生成能力。
- **iLLaDA-8B-Base 新模型**：PILL 同样超越所有基线，耗时仅 +5% overhead，验证跨模型泛化性。
- **多槽扩展**：在 Multi-span infilling（MBPP 构建）上 PILL 较 backbone 提升 11.79 点，较 CAL 提升 4.86 点，耗时增加仅 1.5%。

## 相关工作脉络
- **Diffusion Language Models（D3PM、MDLM、LLaDA、Dream 等）**：DLM 以双向注意力和任意顺序生成为特征，天然适合利用前后文条件的 infilling 任务，但传统实现依赖固定 $L$（Lin et al., 2026a）。
- **DAEDAL（Li et al., 2025a）**：仅支持在序列末尾追加扩展，不满足 infilling 同时依赖 prefix 与 suffix 的要求，在文本基准上甚至低于固定长度 backbone。
- **DreamOn（Wu et al., 2026）**：通过 fine-tune DLM 引入显式长度变化操作实现变长补全，但需重新训练骨干模型，可能损害通用能力；本文在无 fine-tune 前提下实现更强效果。
- **CAL（Liu et al., 2026）**：使用早期去噪置信度反复搜索合适长度，计算开销大（约 14 次额外前向传播）且高度依赖预设初始长度；本文用单次探针替代迭代搜索。
- **FlexMDM / DDOT（Kim et al., 2025；Zhang et al., 2025a）**：同样引入额外 fine-tune 成本，本文以 training-free 探针 + 后验评分方式避免此类开销。
- **FIM-style / InCoder（Bavarian et al., 2022；Fried et al., 2022）**：自回归模型通过 causal attention 进行 infilling，只能单向利用后缀条件；本文借助 DLM 双向注意力原生利用双边上下文，定位更优。

## 局限性与未来方向
- **探针训练数据依赖**：当前探针基于公开代码/文本语料（Py150、LeetCode、C4 等）训练，未见针对特定任务分布做适配，领域偏移时可能影响预测精度。
- **长跨度不确定性未充分建模**：预测不确定性随跨度增长而增大（表 15），现有固定半径 $r=2$ 未能动态适应，仅 bootstrap 扩展策略（A.9）带来有限改善（+0.09 Pass@1）。
- **模型规模上限**：实验集中在 7B/8B 级别单 GPU 环境，探针与后验评分对更大 DLM（如 LLaDA-100B）的缩放行为仍需验证（作者自述）。
- **复杂编辑场景未覆盖**：嵌套 span、仓库级代码 patch 等实际工程中常见的多跨度/跨文件补全场景未涉及，仅 A.5 验证了简单多 span 场景。
- **选择阶段仍有 ~10% 误差空间**：Failure analysis（表 14）显示选择失败占 11.4%，探针覆盖失败占 24.3%，提示候选集与评分机制仍有改进余地。

## 研究启发与可借鉴点
- **"预测而非搜索"范式**：用单次轻量前向传播+MLP 探针直接预测任务关键超参（长度）替代反复搜索，为 DLM 类模型的其他自适应生成任务（如上下文长度控制、停止条件判断）提供了可迁移的设计模式。
- **Slot-wise 并行解码 + 位置插值**：将多候选在同一序列内并行生成并隔离注意力，通过线性插值位置 ID 消除 padding 带来的位置偏移，对任何需要多方案并行推理的场景均有参考价值。
- **后验因果评分机制**：利用严格因果 mask 使 probe token 仅 access $v_{<k}$，实现单次前向传播的多候选联合评分，避免了 pseudo-log-likelihood 的逐 token 计算，可复用于其他 DLM 评分难题。
- **Oracle 上界对照实验设计**：以 Oracle 选择器替代真实选择器，清晰分离"候选生成能力"与"选择决策能力"，为后续工作的消融定位提供了标准化评估基准。
- **跨领域鲁棒性验证**：探针仅用代码训练却在文本 OOD 任务上超越 DreamOn（Table 17），说明长度预测特征具有较强跨域泛化性，为本团队探索多模态/跨域适配提供了先例。

## 关键术语表
- **Diffusion Language Model (DLM)**：将文本生成建模为逐步去噪过程的离散扩散模型，使用双向注意力支持任意顺序生成，区别于 AR 模型的自回归范式。
- **Infilling**：给定 prefix 与 suffix 的条件下生成中间缺失 span 的任务，DLM 可原生利用双边上下文，而 AR 模型仅能单向依赖前缀。
- **Adaptive-length decoding**：在生成过程中动态确定或调整掩码跨度长度的解码策略，解决 DLM 固定长度限制与真实任务变长需求之间的矛盾。
- **Slot-wise attention mask**：在多槽并行解码中隔离各槽的注意力，使每个槽仅 attend 共享上下文与自身 token，避免候选间干扰。
- **Post-hoc selection**：生成多个候选后用额外一次前向传播的评分机制从中选择最优，避免 AR 模型直接依赖 next-token likelihood 评分的局限。
- **Position-ID interpolation**：通过线性插值将不同长度候选槽映射到相同的连续位置 ID 区间，消除 padding 导致的"欠生成"结构性误差。
- **Pseudo-log-likelihood**：对序列中每个 token 依次 mask 后计算条件概率的均值，理论上可用于 DLM 评分，但需 $O(N)$ 次前向传播，成本过高。
- **Probe coverage failure**：预测长度 $\hat{L}$ 及其邻域候选无法覆盖真实长度 $L^*$ 的情况，是当前 PILL 的主要误差来源（24.3%）。

## 可复现要素
- **数据集**：所有数据集均为公开数据集（HumanEval、MBPP、MultiPL-E、WikiText、arXiv、C4、Py150、LeetCode、CodeContests），实验结果使用确定性解码单次运行得出。
- **代码开源**：已开源，地址 https://github.com/Hsu1023/PILL。
- **模型**：LLaDA-8B-Base/Instruct、LLaDA-MoE-Base、Dream-7B-Base、Dream-Coder-7B-Base 等均为公开或研究预览模型。
- **探针训练**：每模型约 196k 样本（代码+文本），AdamW，LR=1e-3，weight decay=1e-4，batch size=16，dropout=0.1，最多 50 轮，MSE loss；单卡 A40 约 100 分钟。
- **关键超参**：探针 MLP 3 层，隐藏维度 512/128，GeLU；候选半径 $r=2$（5 个候选）；选择权重 $\alpha=0.5$；后缀采样 $m\in[4,8]$，上限 32 token。
- **硬件**：单张 NVIDIA A40 GPU。
