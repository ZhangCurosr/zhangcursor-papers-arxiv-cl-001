---
title: "THE-FIRST-TOKEN-IS-A-CLUE-VERBALIZING-MULTI-TOKEN-CONCEPTS-F"
source: https://arxiv.org/pdf/2608.31084v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 18:44:56"
field: "大语言模型可解释性"
keywords: ["J-lens", "multi-token concepts", "interpretability", "verbalizable representations", "causal intervention", "Jacobian lens", "concept vectors"]
innovations: ["发现多token概念首token可读性接近单token概念，首token+源prompt可88.3%恢复第二token", "提出PROPOSE+SCORE两阶段框架，无需固定词表或微调即可恢复多token概念向量", "恢复向量支持因果概念交换，succ@10达61.4%显著优于Template Lens的26.2%"]
benchmarks: ["496 multi-hop clozes on Gemma-3-12B-IT/Llama-3.1-8B/Qwen3-14B", "Rank@10 readout", "succ@10 causal swap"]
---

# 论文速读：THE-FIRST-TOKEN-IS-A-CLUE-VERBALIZING-MULTI-TOKEN-CONCEPTS-F

## 一句话总结
本文发现 J-lens 虽只能输出单 token 排序，但多 token 概念的**第一个 token 可读性接近单 token 概念**，结合源 prompt 后冻结模型能以 88.3% 概率恢复第二个 token；进一步利用后续 hidden state 可在单次前向传播中恢复完整概念向量，实现无需固定短语词表、无需额外微调的端到端多 token 概念读出与因果干预。

## 研究问题与动机
- **J-lens 的多 token 盲区**：J-lens 通过 Jacobian 与 unembedding 映射，为每个词汇表 token 提供一个向量，但"black hole"等多 token 概念无独立向量， constituent tokens（如 black）出现时无法区分"black hole"与"black coffee"。
- **既有扩展方案的代价**：Template Lens 需预定义约 12,700 条短语词表并前向传播数百次预计算向量；Oracle Lens 需微调 proposer + reconstructor 两个组件。两者均引入额外训练/维护负担。
- **核心科学问题**：能否不引入新词表或微调组件，直接从 J-lens 与冻结模型中恢复多 token 概念及其向量？
- **现有方法的不足**：Template Lens 受限于固定词表覆盖度；Oracle Lens 工程复杂度高、代码未公开；J-lens 原始形式完全无法处理跨 token 概念。

## 核心贡献（创新点）
- **诊断发现**：多 token 概念的首 token 在 J-lens top-10 中出现率（54.6%）与单 token 概念（56.9%）几乎相当；给定正确首 token + 源 prompt，冻结模型在 88.3% 的两 token 案例中恢复出第二 token。→ 与已有工作本质区别：首次系统量化"首 token 作为线索"的可读性，而前作关注整体短语级别表征。
- **向量恢复方法**：将候选概念置于 Carrier Prompt 中，利用后续 hidden states 池化得到概念向量，该向量与独立定义的 ground-truth 单 token J-lens 向量 Top-1 匹配率达 97.4%–99.2%。→ 本质区别：无需微调、无需对比学习，仅靠冻结模型自身后续层即可恢复高质量向量。
- **端到端两阶段框架（PROPOSE + SCORE）**：PROPOSE 从 J-lens 多个排名区间选取首 token 候选，由冻结 LLM 补全概念；SCORE 恢复向量并通过 Jacobian 映射到源层打分，与完整词汇表联合排序。→ 本质区别：完全基于推理时冻结模型，不依赖固定短语词表或训练好的 proposer/reconstructor。
- **因果概念交换验证**：恢复的向量支持 causal swap，succ@10 达 61.4%，显著优于 Template Lens 的 26.2%。→ 本质区别：在同一向量恢复框架下同时支持读出与因果干预，无需额外拟合干预强度系数。

## 方法详解
- **PROPOSE（候选概念生成）**：
  - 在源层 $\ell_s$ 获取 J-lens 排序列表，按排名划分为三个不相交区间 $B_m$（ranks 1–20, 21–50, 51–100），每区间独立分配搜索预算。
  - 对每个选定首 token $s$，构造 Scaffold Prompt：`when model is answering the cloze below: <cloze>. it will think a concept: <seed>`，固定首 token，让冻结 LLM 按正常 next-token 分布生成后续 token（核采样 $p=0.85$，最多扩展 4 token）。
  - 去重并移除严格前缀，保留 top-25 候选。

- **SCORE（向量恢复与打分）**：
  - **载体提示**：`Remember the following concept in your mind. Concept: <C>. Fact: The capital of France is Paris.`（不含源 cloze）。
  - **向量恢复公式**：
    $$
    v_C^{(\ell_c)} = \sum_{t \in \mathcal{T}} \alpha_t \left( h_{\ell_c, t}(\tau(C)) - \mu_t \right)
    $$
    其中 $\mu_t$ 为位置特定的均值（消除概念共享成分），$\alpha_t$ 为 pooling 权重（基于独立校准语料上各位置 hidden state 与对应 J-lens token vector 的平均 cosine 一致性确定）。
  - **向量运输**：将载体层 $\ell_c$ 的向量映射到源层 $\ell_s$：
    $$
    v_C^{(\ell_s)} = \arg\min_v \|J_{\ell_s} v - J_{\ell_c} v_C^{(\ell_c)}\|^2 + \lambda \|v\|^2, \quad \lambda=10^{-2}
    $$
  - **打分**：对源 hidden state 做中心化并应用对角白化矩阵 $A = \text{diag}(\sigma^2 + \varepsilon)^{-1/2}$，最终 score 为 $G_C(h_{\ell_s}) = \langle A v_C^{(\ell_s)}, A \tilde{h}_{\ell_s} \rangle$，与词汇表 token 采用同一打分规则联合排序。

- **关键超参**：
  - 排名区间：1–20（宽度 12）、21–50（宽度 6）、51–100（宽度 6）
  - 载体层 $\ell_c$：Gemma=43, Llama=27, Qwen=37
  - 运输 ridge $\lambda=10^{-2}$，白化 $\varepsilon=10^{-3}\text{median}(\sigma^2)$

## 实验与结果
- **数据集**：496 个多跳 cloze 提示（248 对反事实对），目标多 token 概念均不出现在源文本中；测试模型为 Gemma-3-12B-IT、Llama-3.1-8B、Qwen3-14B（使用 Neuronpedia 官方 J-lens fits）。
- **端到端读出（Rank@k）**：

  | 方法 | Gemma Rank@10 | Llama Rank@10 | Qwen Rank@10 | 平均 Rank@10 |
  |---|---|---|---|---|
  | Ours | 52.8% | 44.2% | 32.3% | **43.1%** |
  | Template Lens | 29.0% | 29.4% | 24.4% | 27.6% |
  | Ours (无 J-lens) | 22.4% | 25.7% | 16.6% | 21.6% |
  | Random | 0–8.5% | — | — | — |

- **因果概念交换（succ@k）**：

  | 方法 | Gemma succ@10 | Llama succ@10 | Qwen succ@10 | 平均 succ@10 |
  |---|---|---|---|---|
  | Ours (vC→vD) | 64.2% | 62.2% | 57.7% | **61.4%** |
  | Template Lens | 22.8% | 35.3% | 20.4% | 26.2% |
  | Delete only | ~7% | — | — | — |
  | Add only | 43.8–56.0% | — | — | — |

- **向量恢复诊断**（500 词拆分实验）：恢复向量 Top-1 准确率达 97.4%（Gemma）、98.8%（Llama）、99.2%（Qwen），显著高于首 fragment 向量（26.2–51.4%）与均值向量（17.6–59.6%）。
- **最强结果**：端到端 Rank@10 平均 43.1%（vs Template Lens 27.6%，绝对提升 15.5pp）；causal succ@10 平均 61.4%（vs 26.2%，绝对提升 35.2pp）。
- **计算成本**：单 prompt 0.4–0.8s（A800），详见附录 D。

## 相关工作脉络
- **Logit Lens / Tuned Lens / J-lens 一脉**：均以词汇表 token 为粒度读出 hidden state；本文沿用 J-lens 框架但突破其多 token 限制，无需 affine correction 或 Jacobian 之外的额外训练。
- **Template Lens（Gurnee et al., 2026）**：需预定义 ~12,700 条短语词表并前向数百次预计算向量；本文无需固定词表，候选概念在推理时动态生成，覆盖度不受预定义列表限制。
- **Oracle Lens（Gurnee et al., 2026）**：需微调 proposer + reconstructor；本文完全冻结主模型，仅利用其正常 next-token 生成能力，零额外训练。
- **Patchscopes（Ghandeharioun et al., 2024）**：将激活注入新上下文让 LLM 解释；本文 LLM 从不接收源激活，仅从文本级首 token 线索补全概念，PROPOSE 与 SCORE 解耦。
- **SelfIE（Chen et al., 2024）**：让 LLM 用自然语言描述 embedding；本文聚焦向量级读出与因果干预，而非自然语言释义。
- **稀疏自编码器 / Activation Steering**：通过字典学习或对比样本构建概念向量；本文向量直接从冻结模型后续 hidden state 池化获得，无学习过程。

## 局限性与未来方向
- **候选生成依赖 J-lens 排名**：若首 token 排名过低未被 PROPOSE 覆盖，正确概念将永久缺失，即便 SCORE 能识别也无济于事。
- **当前仅验证短英语多词短语**：未测试长描述或任意自然语言表达，泛化性存疑。
- **未建立跨层表征演化的因果证据**：Discussion 提出"早期层逐渐组合 token 为更高阶概念"的假说，但未直接验证。
- **Oracle Lens 因代码未公开而未对比**：缺少与另一主流扩展方法的完整对照。
- **计算开销**：虽仅 0.4–0.8s/prompt，但对于大规模分析场景仍有优化空间。

## 研究启发与可借鉴点
- **首 token 线索机制可迁移**：任何基于 token-level 读出的可解释性方法（如 Logit Lens、Tuned Lens）均可借鉴"首 token + 源 prompt 补全"策略扩展至多 token 概念。
- **载体提示池化恢复向量**：Carrier Prompt + 位置特定均值减去 + cosine 加权池化的设计简洁有效，可复用于其他需要概念向量的任务（如 cross-context 对齐、概念编辑）。
- **Jacobian 运输无需微调**：用 $v_C^{(\ell_s)} = \arg\min_v \|J_{\ell_s} v - J_{\ell_c} v_C^{(\ell_c)}\|^2 + \lambda\|v\|^2$ 在不同层间运输向量，避免了层间分布偏移的手动校准。
- **诊断实验设计**：拆分单 token 词为两 fragment 并对比恢复向量与 fragment 向量/均值向量，是验证"整体概念表征独立于 constituent tokens"的干净实验范式。
- **与本团队方向结合机会**：可探索将该方法应用于多 hop 推理链追踪、多语言概念对齐、或与其他解释工具（如 Sparse Autoencoders）的向量空间对比。

## 关键术语表
- **J-lens（Jacobian Lens）**：通过层特异性 Jacobian 将中间 hidden state 映射到最终表示空间，再用 unembedding 解码为词汇表 token 排名的可解释性读出炉。
- **Template Lens**：预定义固定短语词表（约 12,700 条）并前向传播预计算每条短语的向量，推理时直接打分。
- **Oracle Lens**：微调 proposer（从激活生成短语）和 reconstructor（将短语映射为向量）两个组件以支持多 token 概念读出。
- **PROPOSE 阶段**：从 J-lens 排名列表中选取首 token 候选，由冻结 LLM 基于源 cloze prompt 补全完整多 token 概念。
- **SCORE 阶段**：将候选概念置于 Carrier Prompt 中，从后续 hidden states 池化恢复概念向量，经 Jacobian 运输到源层后打分。
- **Causal Concept Swap**：在隐藏状态中投影删除源概念向量并叠加目标概念向量，验证恢复向量能否因果性地改变模型输出。
- **Rank@k**：目标概念进入联合排序 top-k 的比例，衡量读出能力。
- **succ@k**：因果干预后目标答案进入模型 top-k 续写的比例，衡量向量操控能力。

## 可复现要素
- **数据集**：496 个多跳 cloze 提示（248 对反事实对），由 GPT-5.5 生成；论文未声明开放，但附录 A 提供了模板与构建约束，可据此复现。
- **代码/权重**：J-lens fits 来自 Neuronpedia（Lin & Bloom, 2023）；论文未提供本项目代码开源声明。
- **关键超参**：排名区间 [1–20, 21–50, 51–100]，各区间搜索宽度 [12, 6, 6]；核采样 $p=0.85$，最大生成 4 token；载体层 $\ell_c$（Gemma=43, Llama=27, Qwen=37）；ridge $\lambda=10^{-2}$；白化 $\varepsilon=10^{-3}\text{median}(\sigma^2)$；保留 top-25 候选。
