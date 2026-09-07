---
title: "ViSAR-Training-Free-Adaptive-k-Retrieval-for-Visual-Document"
source: https://arxiv.org/pdf/2609.02486v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-07 00:32:32"
field: "文档视觉问答与检索增强生成"
keywords: ["Document Visual Question Answering", "Late-Interaction Retrieval", "Adaptive-k", "RAG", "Visual Document Retrieval", "Training-Free"]
innovations: ["训练无关的三级嵌入空间加权机制实现自适应k检索，无需微调编码器", "构建查询条件化页面级相似度矩阵，其稀疏结构与答案准确率相关", "基于相干性与泄漏代价的O(N)优化目标替代固定top-k和分数分布启发式"]
benchmarks: ["MMLongBench", "LongDocURL"]
---

# 论文速读：ViSAR-Training-Free-Adaptive-k-Retrieval-for-Visual-Document

## 一句话总结
ViSAR 是一种训练无关的自适应 k 检索方法，通过在晚交互编码器（late-interaction encoder）的嵌入空间中对多向量表示进行加权，构建查询条件化的页面级相似度矩阵，从而在推理时动态决定检索页数，在 MMLongBench 上将 RAG 端到端延迟降低最高达 58.7%，同时保持或提升 DocVQA 答案准确率。

## 研究问题与动机
- **固定 top-k 检索无法适应查询复杂度**：现有晚交互方法（如 ColPali）对所有查询统一检索固定数量的页面，复杂查询需要更多页但会引入无关内容，简单查询则浪费计算资源，均增加 LVLM 推理延迟并可能降低准确率。
- **现有自适应 k 方法不适用于视觉嵌入**：文本检索中的 embedding 加权（基于词频统计）和自适应 k 策略依赖离散 token 结构，无法直接迁移至无 OCR 的视觉 patch 嵌入空间；而现有的视觉方案通常需要额外训练。
- **晚交互表示蕴含页面语义结构，但未加利用**：晚交互编码的每个 query embedding 在各页面间产生稀疏、查询依赖的激活分布，这种跨页面的语义组织结构可用于指导自适应检索，但此前未被系统挖掘。
- **页面排序质量与下游答案准确率的关系有待揭示**：固定 top-k 排序在复杂查询中可能需检索中间无关页才能覆盖全部证据页，说明排序本身存在局限，如何构建更贴合查询的页面排序是提升 DocVQA 的关键。

## 核心贡献（创新点）
- **提出 ViSAR 训练无关自适应 k 检索机制**：直接在晚交互编码器的嵌入空间中进行 Query-to-Page、Page-to-Query、Page-to-Page 三级加权交互，无需对编码器微调即可实现查询感知的动态页数选择。与已有工作的本质区别在于：不同于文本领域的词频加权或需训练的自适应方法，ViSAR 利用视觉 patch 嵌入的语义激活模式，完全训练无关。
- **构建查询条件化的页面级相似度矩阵以揭示语义定位结构**：通过加权后的 MaxSim 操作生成非对称的 Sim(p, p') 矩阵，其稀疏性刻画了查询相关证据在文档中的局部化程度。与已有工作的本质区别在于：这是首个将晚交互的多向量结构显式用于构建跨页面相似度矩阵并关联到答案准确率的工作。
- **设计基于相干性与泄漏代价的自适应 k 优化目标**：通过最小化成本函数 I(k) = Σ w_sp (c_k^p - γ l_k^p) 在 O(N) 复杂度内确定最优页数 k*，同时引入过渡尖锐性检验防止泄漏残留。与已有工作的本质区别在于：相比 Largest-Gap（最大分数间隔）和 Score-Cluster（聚类）等仅依赖单点分数的启发式方法，ViSAR 同时利用页面间相似度结构做出决策。
- **系统实验验证延迟-精度联合收益**：在 MMLongBench 和 LongDocURL 两个多页文档基准上，ViSAR 平均检索页数显著少于 Oracle 及其他自适应基线，RAG 延迟最高降低 58.7%，答案准确率在 60 种配置（3 编码器×5 LVLM×2 Max-k×2 数据集）中 24 种提升、36 种持平。

## 方法详解

**1. 前期基础：MaxSim 与晚交互**
- 文档 D = {P¹, …, P^N}，每页 P^p 和查询 Q 分别表示为多向量嵌入集合：P^p ∈ ℝ^(n_p×D)，Q ∈ ℝ^(m×D)。
- 标准晚交互相关度分数（Equation 1）：S_{Q,P^p} = Σ_{i=1}^{m} max_j sim(q_i, v_j^p)，对每个 query embedding 取其与页面所有 patch embedding 的最大余弦相似度后求和。

**2. 激活分数定义（Equation 2）**
- A_{p,i} = max_j ⟨q_i, v_j^p⟩，衡量 query embedding q_i 的语义在页面 P^p 中的实现强度，保留完整激活矩阵而非合并为单一分数。

**3. Query-to-Page 交互加权（Equations 3–7）**
- 将 A_{p,i} 按页面均值缩放得 Â_{p,i}，并按跨页面的标准化标准差调制得 ã_{p,i}（Equations 3–4）。
- 聚合 ã_{p,i} 得到 query embedding 权重 w_i = log(N / (1 + a_i))，其中 a_i = Σ_p ã_{p,i}（Equation 5）——抑制在所有页面都出现的通用语义，突出稀疏激活。
- 计算页面内 query 共激活 C_{i,i'}^p = ã_{p,i} · ã_{p,i'}，聚合得页面权重 w_p（Equation 7）——识别与查询强相关的页面。

**4. Page-to-Query 交互加权（Equations 8–9）**
- 对 w_i、w_p 做 Min-Max 归一化（得 ŵ_i, ŵ_p ∈ [0,1]），调制 patch-query 余弦相似度后，反转 MaxSim 方向得页面内每个 patch 的相关度 r_j^p（Equation 8）。
- 对 r_j^p 中心化并阈值化得 patch 权重 w_j^p = max(0, r_j^p - mean(r_j^p))（Equation 9）——抑制低激活 patch。

**5. Page-to-Page 交互加权（Equations 10–11）**
- 归一化得 ŵ_j^p，构造源页面 P^p 到目标页面 P^{p'} 的加权相似度 S_j^{p→p'}（Equation 10）。
- 取 T=50 个最大交互的平均再开方，得到非对称页面相似度矩阵 Sim(p, p')（Equation 11）。

**6. 自适应 k 检索（Equations 12–14）**
- 以自相似度 s_p = Sim(p, p) 排序，取 top-k 为候选相关集 R_k，其余为无关集 I_k。
- 计算相干性 c_k^p = mean_{p'∈R_k} Sim(p, p') 和泄漏度 l_k^p = mean_{p'∈I_k} Sim(p, p')（Equations 12–13）。
- 优化目标 I(k) = Σ_{p∈R_k} w_{s_p}(c_k^p - γ l_k^p)，γ=10⁵（Equation 14），在 O(N) 候选集中寻找最优 k*。
- 额外引入过渡尖锐性检验：仅当 I(k) 在 k* 之后的变化率大于之前的变化率时才接受 k*+1，防止泄漏残留。
- 实现优化：w_j^p=0 的页面被排除出后续计算；Equation 10 以 block-wise 方式计算以降低峰值内存。

## 实验与结果

**数据集**：MMLongBench（长上下文多模态文档理解）和 LongDocURL（多模态长文档基准，含证据页标注），支持召回率/精确率评估。

**编码器**：ColPali、ColQwen2.5、ColModernVBERT（三类无 OCR 视觉多向量编码器）；对比基线含 ColBERTv2（文本多向量）、VisRAG-Ret（单向量视觉）。

**自适应 k 基线**：Oracle（枚举所有证据页的最小 k）、Largest-Gap（最大分数间隔启发式）、Score-Cluster（分数聚类启发式）。

**生成模型**：Qwen2.5-VL-7B-Instruct（LVLM），额外测试 5 种其他 LVLM。

**主要结果**：
- **检索效率**：ViSAR 平均检索页数显著少于 Oracle——MMLongBench 上 ColQwen2.5 编码器下 ViSAR 均值 4.7 页 vs Oracle 8.3 页；LongDocURL 下 7.9 页 vs 10.7 页。Largest-Gap 和 Score-Cluster 反而检索更多页面（MMLongBench 分别为 15.4 和 18.6 页）。
- **排序质量**：ViSAR 在 Recall@5 和 NDCG@10 上均优于标准晚交互——ColQwen2.5 下 Recall@5: 79.73% vs 78.60%，NDCG@10: 0.799 vs 0.793（Table 2）。
- **答案准确率**：ViSAR 在 MMLongBench Max-10 下达到 36.63%（ColQwen2.5），超过固定 top-k 的 35.69%；LongDocURL Max-10 达 60.97% vs 59.27%。在 60 种配置中 24 种显著提升、36 种持平，无下降。
- **RAG 延迟**：端到端延迟最高降低 **58.7%**（MMLongBench，Max-10），LongDocURL 降低 38.5%。检索开销随文档规模增长但仅在最长的 468 页文档中显著。
- **最强结果**：MMLongBench Max-10 + ColQwen2.5 编码器，ViSAR 准确率达 36.63%，较固定 top-k 提升约 2.6%，延迟降低 58.7%。

## 相关工作脉络
- **ColPali / ColBERT 系列（[12–14]）**：晚交互多向量检索的开创与视觉扩展工作，ViSAR 在其嵌入空间之上进行训练无关加权，不改变编码器本身，区别于这些工作的编码器训练范式。
- **ColBERTv2 / PLaid（[20, 21]）**：文本领域的高效晚交互检索优化，依赖 OCR 提取文本，无法处理视觉内容；ViSAR 直接作用于视觉 patch 嵌入，实现 OCR-free。
- **Token 加权与稀疏化方法（[15–18]）**：如 SPLADE、End-to-end query term weighting，基于词频统计或学习重要性估计，依赖离散 token 结构；ViSAR 将类似思想迁移至连续视觉嵌入空间且无需训练。
- **Adaptive-k 文本检索（[19, 32, 33]）**：Largest-Gap 和 Score-Cluster 等启发式方法基于分数分布确定截断点，仅依赖单点相关性分数；ViSAR 利用跨页面相似度结构，在 O(N) 内联合优化相干性与泄漏，且实验显示基线方法普遍过检索。
- **VisRAG / M3DocRAG（[1, 3]）**：VisRAG 使用单向量视觉检索器，M3DocRAG 采用 ColPali 固定 top-k 检索；ViSAR 与其正交，可无缝接入这些框架的检索模块。
- **Iterative RAG 方法（[27–31]）**：如 Self-RAG、Adaptive-RAG、Doc-ReAct 通过多轮 LLM/LVLM 推理隐式调节 k；ViSAR 在单次传递中直接估计 k，无需 LLM 参与决策，延迟更低。

## 局限性与未来方向
- **小编码器性能受限**：ColModernVBERT（250M 参数）相比 ColQwen2.5（3B）语义分离能力较弱，导致 ViSAR 在其上的提升不显著甚至略降，说明方法对底层编码器质量有依赖。
- **极端长文档的相似度矩阵计算开销**：468 页文档的端到端延迟增益下降，补充材料虽提供了近似策略，但未在主实验中详细展示。
- **相似度矩阵稀疏性与准确率的因果性待验证**：论文仅报告了相关性（Figure 5-c），未证明矩阵结构可直接用于改进检索策略。
- **未来方向**：利用 Sim(p,p') 矩阵结构作为无标签反馈信号，支持迭代查询精炼或证据选择；探索矩阵稀疏性的定量度量用于检索质量感知。

## 研究启发与可借鉴点
- **训练无关的嵌入空间加权范式可迁移**：ViSAR 完全不微调编码器，仅在后处理阶段对 MaxSim 激活值进行统计加权，此思路可迁移至其他多向量检索场景（如多模态知识图谱检索、视频检索）中实现零样本自适应。
- **跨对象的相似度矩阵用于自适应决策**：将 pair-wise 相似度矩阵的结构特性（稀疏性、连通性）转化为自适应截断的优化目标，这一设计模式可推广至任意基于多向量表示的检索系统中，不限于文档页面。
- **相干性-泄漏度的折中优化目标具有通用性**：I(k) 中的 c_k^p（组内相似度）与 l_k^p（组间相似度）构成类似聚类质量的分离度指标，可在聚类感知检索、动态上下文选择等场景复用。
- **与团队方向的结合机会**：若团队研究方向涉及多页文档理解、长上下文 RAG 或检索效率优化，ViSAR 的训练无关特性和即插即用接口使其易于集成到现有 VisRAG/M3DocRAG 管道中；相似度矩阵的结构分析可为检索质量评估提供新信号。

## 关键术语表
- **Late-Interaction（晚交互）**：将 query 和 document 分别编码为多向量集合，通过逐向量最大相似度聚合计算相关度（如 ColBERT 的 MaxSim 操作），支持细粒度语义匹配。
- **MaxSim**：Late-interaction 的核心算子，对每个 query embedding 在其对应 document 的所有 embedding 中找最大余弦相似度后求和。
- **Adaptive-k Retrieval**：根据查询复杂度动态确定检索页面/段落数量 k，而非使用固定阈值，以平衡召回率与无关上下文引入。
- **DocVQA（Document Visual Question Answering）**：面向视觉上丰富的多页文档的问答任务，要求模型在图文混排文档中定位证据并生成答案。
- **RAG（Retrieval-Augmented Generation）**：检索增强生成，先通过编码器检索相关文档片段，再送入 LVLM 生成答案的范式。
- **ColPali / ColQwen2.5**：基于晚交互框架的无 OCR 视觉文档检索编码器，将每页图像编码为 patch embedding 集合，支持多模态语义匹配。
- **LLM-as-a-Judge**：使用大语言模型作为评价器，根据语义等价性判断生成答案与参考答案是否正确的自动评估方法。

## 可复现要素
- **数据集**：MMLongBench 和 LongDocURL，均公开可用。
- **代码**：已开源，地址 https://github.com/adrienmialland/ViSAR。
- **模型权重**：ColPali、ColQwen2.5、ColModernVBERT、VisRAG-Ret、ColBERTv2 均为公开模型；生成模型使用 Qwen2.5-VL-7B-Instruct 和 Qwen2.5-14B-Instruct（公开权重）。
- **关键超参**：T=50（Equation 11 中取 Top-T 交互）、γ=10⁵（Equation 14 中泄漏惩罚系数）；论文声明完整敏感性分析见补充材料。
