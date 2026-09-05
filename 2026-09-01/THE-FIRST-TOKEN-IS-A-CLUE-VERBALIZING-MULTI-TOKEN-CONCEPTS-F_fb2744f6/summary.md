---
title: "THE-FIRST-TOKEN-IS-A-CLUE-VERBALIZING-MULTI-TOKEN-CONCEPTS-F"
source: https://arxiv.org/pdf/2608.31084v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 18:45:07"
field: "大语言模型可解释性"
keywords: ["J-lens", "可解释性", "多token概念", "activation steering", "概念向量", "causal intervention", "LLM interpretability"]
innovations: ["首个token作为多token概念读出的强线索，冻结模型88.3%恢复率", "无需固定词表或微调，从后续隐藏状态直接恢复概念向量（Top-1达97.4%+）"]
benchmarks: ["Rank@10读取出准率", "succ@10因果概念交换成功率"]
---

# 论文速读：THE-FIRST-TOKEN-IS-A-CLUE-VERBALIZING-MULTI-TOKEN-CONCEPTS-F

## 一句话总结
论文提出一种无需固定短语词表或微调的新方法，利用J-lens的**首个token作为线索**，让冻结的LLM完成候选多token概念，并从后续隐藏状态中恢复完整概念向量，实现了对未命名多token概念的端到端可读性与因果干预。

---

## 研究问题与动机

1. **J-lens的多token局限**：Jacobian Lens（J-lens）通过将中间隐藏状态乘以Jacobian矩阵再经unembedding解码，给出词汇token的排名列表，但每个多token概念（如"black hole"）没有自己的独立向量， constituent token向量简单相加会丢失顺序信息。
2. **已有扩展方法的不足**：Template Lens依赖预定义的~12,700条固定短语词表；Oracle Lens需要fine-tune proposer和reconstructor两个组件——两者都需要额外构建显式模块。
3. **核心疑问**：能否直接从J-lens和冻结模型中恢复多token概念及其向量，而无需任何额外训练或固定词表？
4. **可解释性读出需求**：许多中间概念跨越多个token（如多跳推理中的隐式实体），现有方法使这部分模型中间计算无法被读出。

---

## 核心贡献（创新点）

1. **诊断性发现：首个token的可读性**——多token概念的首个token进入J-lens top-10的比例（54.6%）与单token概念（56.9%）几乎相同；给定正确首token和源提示，冻结模型在88.3%的两token案例中能恢复第二个token。
2. **向量恢复机制**——一旦完整概念已知，仅通过固定Carrier Prompt将概念输入模型，从后续隐藏状态中恢复的概念向量与ground truth的Top-1匹配率达97.4%~99.2%，远高于首token向量（26.2%~51.4%）或均值向量（17.6%~59.6%）。
3. **PROPOSE + SCORE两阶段方法**——PROPOSE从J-lens排名列表不同区间选取首token候选，冻结LLM从源提示补全概念；SCORE用固定载体提示恢复每个候选的概念向量，经Jacobian空间映射到源层后与词汇token统一打分排序。
4. **因果干预验证**—— recovered向量支持概念交换（causal concept swap），succ@10平均61.4%，显著高于Template Lens的26.2%。

---

## 方法详解

### PROPOSE：候选概念生成

1. 在源层$\ell_s$处，J-lens对隐藏状态$h_{\ell_s}$给出token排名列表。
2. 将排名列表前100名划分为三个不相交区间$B_m$（ranks 1–20, 21–50, 51–100），每个区间独立分配搜索预算$q_m$（active widths分别为12, 6, 6），使低排名首token也能进入候选集。
3. 对每个候选首token $s$，使用固定Scaffold Prompt：
   ```
   when model is answering the cloze below:
   <cloze>
   it will think a concept: <seed>
   ```
   其中`<cloze>`为原始源提示，`<seed>`为J-lens选出的首token。冻结LLM在核采样（nucleus mass=0.85）下自由生成后续token，最多生成4个token，丢弃特殊/格式化token（除终止符外）。
4. 合并所有区间候选，去重并移除严格前缀，保留25个最终候选。

### SCORE：概念向量恢复与打分

1. **向量恢复**：对每个候选概念$C$，使用固定Carrier Prompt：
   ```
   Remember the following concept in your mind.
   Concept:<C>. Fact: The capital of France is Paris.
   ```
   从载体层$\ell_c$处提取概念$C$之后的隐藏状态窗口$\mathcal{T}$，按位置加权池化得到概念向量：
   $$v_C^{(\ell_c)} = \sum_{t \in \mathcal{T}} \alpha_t \left(h_{\ell_c, t}(\tau(C)) - \mu_t\right)$$
   其中$\mu_t$为位置特定的均值（去除跨概念共享分量），$\alpha_t$为 pooling 权重（基于独立校准语料上居中隐藏状态与J-lens token向量的平均余弦相似度确定）。
   
2. **Jacobian空间映射（Transport）**：载体层$\ell_c$的向量需映射到源层$\ell_s$，通过最小化Jacobian图像距离：
   $$v_C^{(\ell_s)} = \arg\min_v \|J_{\ell_s} v - J_{\ell_c} v_C^{(\ell_c)}\|^2 + \lambda\|v\|^2, \quad \lambda = 10^{-2}$$
   使用预处理共轭梯度法求解。

3. **打分**：对源隐藏状态进行中心化和对角白化（$A = \text{diag}(\sigma^2+\varepsilon)^{-1/2}$），候选概念$C$的分数为：
   $$G_C(h_{\ell_s}) = \langle A v_C^{(\ell_s)}, A \tilde{h}_{\ell_s} \rangle$$
   词汇token同样用其J-lens token向量代入同一公式打分，所有候选与完整词汇表联合排序。

---

## 实验与结果

**数据集**：496个多跳cloze提示（248个反事实对），中间多token概念从不出现于源文本中。三模型：Gemma-3-12B-IT、Llama-3.1-8B、Qwen3-14B。

**基线**：Template Lens（预定义短语词表+预计算向量）、OURS (WITHOUT J-LENS，去掉J-lens首token线索)、Random。

| 方法 | Gemma Rank@10 | Llama Rank@10 | Qwen Rank@10 | **平均 Rank@10** |
|------|-------------|-------------|------------|---------------|
| Ours | **52.8%** | **44.2%** | **32.3%** | **43.1%** |
| Template Lens | 29.0% | 29.4% | 24.4% | 27.6% |
| Ours (no J-lens) | 22.4% | 25.7% | 16.6% | 21.6% |
| Random | 0.0% | 5.6% | 8.5% | — |

**因果概念交换（succ@10）**：

| 方法 | Gemma | Llama | Qwen | **平均 succ@10** |
|------|-------|-------|------|---------------|
| v_C → v_D (Ours) | **64.2%** | **62.2%** | **57.7%** | **61.4%** |
| Template Lens | 22.8% | 35.3% | 20.4% | 26.2% |
| Add only | 56.0% | 49.7% | 43.8% | — |
| Delete only | 7.0% | 7.9% | 7.6% | — |

**向量恢复诊断（Table 1）**：将500个单token词拆分为两段，用碎片形式恢复向量后与500个原生J-lens token向量做余弦排序，Ours恢复向量Top-1准确率：Gemma 97.4%、Llama 98.8%、Qwen 99.2%；首token向量最高仅51.4%，均值向量最高59.6%。

**计算成本**：单A800上每提示0.40–0.78秒（Table 4）。

---

## 相关工作脉络

1. **J-lens与Template/Oracle Lens**（Gurnee et al., 2026）——本文直接对比的基线方法，Template Lens依赖固定短语词表预计算向量，Oracle Lens需fine-tune proposer+reconstructor，本文方法无需任何额外训练模块。
2. **Logit Lens / Tuned Lens / Future Lens**（nostalgebraist 2020; Belrose et al. 2023; Pal et al. 2023）——均通过unembedding解码中间状态为词汇token，本质上都只能读出单token概念，本文方法与它们同属"lens-based readout"一脉。
3. **Patchscopes**（Ghandeharioun et al., 2024）——将内部表示注入新推理上下文让LLM解读，而本文的冻结LLM从未见到源激活本身，仅在PROPOSE阶段接收源提示+首token线索。
4. **SelfIE**（Chen et al., 2024）——让LLM用自然语言解释隐藏嵌入，本文不涉及自然语言解释，而是向量层面的打分与因果干预。
5. **Token erasure / 多token表示研究**（Feucht et al., 2024; Kaplan et al., 2025）——发现中间计算中子token身份信息逐渐消失而完整实体表示仍存在，本文为此提供了J-lens框架下的实证支撑。
6. **Activation Steering / Sparse Autoencoders**（Turner et al. 2023; Cunningham et al. 2024; Panickssery et al. 2024）——通过对比学习或稀疏编码获得概念向量用于干预，本文的向量来自冻结模型的后续隐藏状态，无需对比数据或学习字典。

---

## 局限性与未来方向

1. **候选生成的瓶颈**：方法高度依赖J-lens将有用首token排到足够高的名次；若首token排名过差，正确概念永远不会进入候选集。
2. **源提示的信息需求**：冻结模型需从源提示+首token充分推断完整概念，对信息不足的提示效果会下降。
3. **仅验证了短英语多词表达**：实验集中于标准多token概念（短英语短语），长描述或任意自然语言表达的泛化性尚未验证。
4. **未建立跨层表示演化的因果机制**：论文推测早期层逐步组合token级片段形成概念级表示，但这一跨层演化路径未被实验直接证实，是重要的未来研究方向。
5. **Oracle Lens未被复现对比**：因无官方公开代码且多阶段训练管线复现不确定性大，未纳入对比。

---

## 研究启发与可借鉴点

1. **首token作为读出的强线索**：多token概念的首token可读性与单token相当——这一诊断性发现可迁移到其他lens-based读出手法（如Tuned Lens、Future Lens），作为跳过复杂短语建模的简洁替代方案。
2. **Carrier Prompt向量恢复范式**：用"Remember the following concept in your mind"类固定提示从后续隐藏状态恢复概念向量，无需对比数据或微调——这一范式可与activation steering结合，为无监督概念向量提取提供新路径。
3. **Jacobian空间transport作为跨层对齐工具**：方程(2)的ridge-regularized Jacobian匹配可用于不同层间表示的对齐，不依赖层间线性假设，值得在跨层分析中推广。
4. **多区间搜索预算策略**：将J-lens排名列表分区间独立搜索（ranks 1–20/21–50/51–100各分配不同width）避免了高排名token独占候选预算的问题，可作为通用候选生成策略。
5. **可与本团队方向结合**：若团队关注多跳推理的中间概念可视化，此方法可直接应用于现有J-lens管线，以零额外训练成本扩展至多token概念；亦可与sparse autoencoder的concept dictionary结合，实现混合读出。

---

## 关键术语表

**J-lens（Jacobian Lens）**：通过将中间层隐藏状态乘以该层对最终表示的平均Jacobian矩阵，再用unembedding解码，输出模型即将 verbalize 的token排名列表的可解释性工具。

**PROPOSE**：方法的第一阶段，从J-lens排名列表的不同区间选取候选首token，由冻结LLM基于源提示补全为完整多token概念。

**SCORE**：方法的第二阶段，用固定Carrier Prompt从模型后续隐藏状态中恢复每个候选概念的向量，经Jacobian映射到源层后与完整词汇表统一打分排序。

**Carrier Prompt**：固定格式的提示模板（"Remember the following concept in your mind. Concept:<C>. Fact: The capital of France is Paris."），用于从模型隐藏状态中提取概念向量。

**Scaffold Prompt**：固定格式的提示模板（"when model is answering the cloze below: <cloze> it will think a concept: <seed>"），用于引导冻结LLM从首token补全候选概念。

**Causal Concept Swap**：将源概念向量从激活中投影移除并叠加伙伴概念向量，验证 recovered 向量是否能真正控制模型的中间计算而非仅相关。

**Token Vector**：J-lens中由$d_\ell(w) = (W_U J_\ell)_w$定义的词汇token $w$在层$\ell$的表示向量。

**Concept Vector**：多token概念$C$在层$\ell$的表示向量$v_C^{(\ell)}$，由Carrier Prompt下的隐藏状态池化恢复。

---

## 可复现要素

- **数据集**：496个多跳cloze提示（248反事实对），论文Appendix A详述构造流程；数据集是否公开需进一步确认（论文未明确声明开源链接）。
- **代码/权重**：使用Neuronpedia上发布的J-lens fits（未自行refit），载体reader在633个独立概念上拟合后固定；论文未声明代码开源。
- **关键超参**：
  - J-lens排名区间：ranks 1–20, 21–50, 51–100
  - 各区间active width：12, 6, 6
  - Nucleus mass：0.85
  - 最大概念长度：4 token
  - 保留候选数：25
  - Ridge正则化系数$\lambda$：$10^{-2}$
  - Whitening $\varepsilon$：$10^{-3} \times \text{median}(\sigma^2)$
  - 载体层$\ell_c$：Gemma=43, Llama=27, Qwen=37
  - 计算环境：单A800-SXM4-80GB, bfloat16

---
