---
title: "BiG-SURE-Bipartite-Graph-for-Semantic-Uncertainty-and-Reliab"
source: https://arxiv.org/pdf/2608.30646v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 01:53:44"
field: "大语言模型可靠性与不确定性估计"
keywords: ["uncertainty estimation", "black-box LLM", "semantic entropy", "bipartite graph", "multimodal QA", "NLI entailment"]
innovations: ["提出跨温度锚点-探测二分图框架以捕捉语义漂移", "将归一化平方谱能量（Frobenius范数）定义为SURE不确定性分数", "在无监督黑盒设定下统一文本/多语言/多模态QA的不确定性估计"]
benchmarks: ["Trivia-QA", "SVAMP", "SciQ (en/fr/ja/zh)", "OK-VQA"]
---

# 论文速读：BiG-SURE-Bipartite-Graph-for-Semantic-Uncertainty-and-Reliab

## 一句话总结
提出 BiG-SURE，一种基于跨温度二分图语义一致性的黑盒不确定性估计方法：以低温度响应为稳定语义锚点、高温度响应为探测探针，构建锚点-探测二分图并用其归一化平方谱能量（即 Frobenius 范数平方除以 MN）作为置信度，其补即为不确定性分数；在文本问答、多语言问答及多模态问答任务上均优于已有黑盒基线。

## 研究问题与动机
- 黑盒场景下（无法访问内部 logits/token 概率）对 LLM/VLM 的预测进行可靠不确定性估计，是金融、法律、医疗等安全关键领域部署的核心瓶颈。
- 现有黑盒不确定性估计方法（DSE、Graph-based、KLE、SNNE、LexSim 等）几乎全部基于**固定温度**采样并从中提取不确定性信号，忽略了低/高温度响应对比所蕴含的语义漂移信息。
- 多语言与多模态场景相对于纯英文文本更易产生幻觉与误判，亟需一种不依赖模型参数、同时可迁移至多语言/多模态的黑色方框不确定性方法。
- 已有文本不确定性方法难以直接复用：词法重叠指标在多语言下失效、聚类类方法在多语言/多模态扩展性不足，而光谱/核类方法仍仅使用同温采样。

## 核心贡献（创新点）
1. 首次将**跨温度响应动力学**识别为可用于 LLM/VLM 黑盒不确定性估计的有效信号，突破了以往固定温度采样的范式。
2. 提出**锚点-探测二分图（BiG）** 框架：用低温度稳定响应作锚点、高温度多样响应作探针，并通过语义相似边连接；区别于同温图方法（如 NumSet/Deg/EigV），其核心信号来自跨温度对齐而非同温内聚类结构。
3. 定义 **SURE 分数**：将锚点-探测相似矩阵的归一化平方谱能量（等价于归一化 Frobenius 范数平方）作为置信度，不确定性为其补；相较于 DSE（需语义聚类+熵）和 KLE（需 von Neumann 熵）避免复杂聚类或矩阵求熵操作。
4. 系统评测覆盖纯文本 QA（Trivia-QA、SVAMP）、多语言 QA（SciQ 四语种）与多模态 QA（OK-VQA），并在多个模型族（Llama、Qwen、Gemma、Aya、Apertus、Pixtral、LLaVA 等）上展示跨设置优越性与稳定性。
5. 给出**理论性质证明**：语义同一性（U=0 当且仅当所有响应语义一致）、语义互斥性（U=1 当且仅当跨温度完全不相干）与单调性（相似度下降则不确定度不降），确立 SURE 作为有效不确定性估计量的数学基础。

## 方法详解
- **采样策略**：给定输入 x，设低温度 T_L 生成 M 个锚点响应 A={a_i}，高温度 T_H>T_L 生成 N 个探测响应 H={h_j}。默认 M=3、N=10、T_L=0.1、T_H=1.0。
- **输入增强**：N 个探测由 K=5 个语义等价变换（文本 paraphrase；多模态再加 mild 图像扰动：均值模糊 kernel=5、±10° 仿射旋转、高斯噪声 N(0,10) 截断至 [0,255]）各采样 2 次拼合而成，用以引入有意义但语义保持的变异。
- **双边相似度矩阵**：构建 W∈R^{M×N}，W_{ij}=sem-sim(a_i,h_j)，语义相似度由 NLI 蕴涵模块输出 P(entail|a_i,h_j)∈[0,1] 提供（文本/多模态用 DeBERTa-large，多语言用 mDeBERTa），未进行微调。
- **SURE 计算公式**：
  - 谱能量：E_SURE(W)=Σ_i σ_i^2=‖W‖_F^2=Σ_{i,j}W_{ij}^2（Frobenius 恒等）。
  - 归一化置信度：E_norm(W)=E_SURE(W)/(MN)，落在 [0,1]。
  - 归一化不确定度：U_norm(W)=1−E_norm(W)，落在 [0,1]。
- **验证理论性质**：U_norm=0 当且仅当所有跨温度对完全语义一致；U_norm=1 当且仅当所有跨温度对完全不相干；在 W 元素逐点不增的条件下 U_norm 单调不减。
- **Top-k 截断谱分析**：实验显示 Top-1/Top-2 奇异值已能复现全谱 SURE 排名（因 M=3 时 rank≤3），表明主要不确定信号集中在主导奇异模态。
- **算法复杂度**：构造 W 为 O(MN·C_sem)，谱能量计算开销远小于 LLM 推理成本；相较全对相似图 O(N^2·C_sem) 及 KLE 的 O(N^3)。

## 实验与结果
- **数据集与模型**：
  - 文本 QA：Trivia-QA（6.68% accuracy 均准参考）、SVAMP（75.5%）。
  - 多语言 QA：SciQ 英/法/日/中（Aya-expanse-8b、Apertus-8B）。
  - 多模态 QA：OK-VQA（63.7%）。
  - LLM：Llama-3.1-8B、Llama-3.3-70B、Qwen-2.5-7B、Qwen-2.5-72B、Gemma-3-12b、Gemma-3-27b；VLM：Pixtral-12B、LLaVa-v1.6-mistral-7b、Qwen3-VL-8B。
- **评估指标**：以 greedy(T=0) 与 ground truth 匹配判定正误，AUROC 作为 abstention 性能；另附 AURC。
- **主要数值结果（ Abstention AUROC，均值±std，5 seeds）**：
  - Trivia-QA 平均：SURE **0.753±0.017**，最强基线 Deg 0.722±0.007；SURE 在 6 模型中 5 个最优。
  - SVAMP 平均：SURE **0.877±0.022**，最强基线 SNNE 0.846±0.018。
  - SciQ 平均：en 0.677、ja 0.685、zh 0.691（三者均为最优），fr 0.590（LexSim 最优）。
  - OK-VQA 平均：SURE **0.752±0.014**，最强基线 SNNE 0.735±0.016；三 VLM 上均最优。
  - 在七个数据/语言组中，SURE 在六组取得平均 AUROC 最高。
- **AURC 结果**：SURE 在 SVAMP（0.084±0.033）与 Trivia-QA（0.198±0.052）上均最低（更好）。
- **预算公平对比**：将基线也用 N=13 样本重测，SURE 仍保持优势，证明增益非来自更多调用次数。

## 相关工作脉络
1. **Semantic Entropy / DSE**（Kuhn et al., 2023; Farquhar et al., 2024）：在同温高温度采样上做语义聚类后求熵；SURE 不聚类、改用跨温度二分图能量。
2. **Kernel Language Entropy (KLE)**（Nikitin et al., 2024）：以单位迹语义核的 von Neumann 熵度量不确定性；仅用同温样本，SURE 引入低温度锚点提供稳定语义参考。
3. **Semantic Nearest-Neighbor Entropy (SNNE)**（Nguyen et al., 2025）：避免显式聚类，直接在成对相似度上算最近邻熵；仍属同温策略，SURE 通过跨温度对齐捕获更大判别力。
4. **Graph-based**（Lin et al., 2024）：将同温响应作为图节点、用度/特征值等统计量评分；SURE 将图限定为锚点-探测二分结构，度量对象不同。
5. **Lexical Similarity (LexSim)**：基于 ROUGE-L 等词法重叠；对多语言无效，SURE 使用 NLI 蕴涵概率以语言无关地衡量语义一致。
6. **Multimodal uncertainty**（FESTA、TREA、Uncertainty-O、UMPIRE 等）：多数聚焦视觉扰动或函数等价采样；SURE 同时用 paraphrase 与 mild 图像扰动，并与跨温度框架统一。

## 局限性与未来方向
- **依赖语义相似度后端**：NLI 器在多语言/数值/频率表述下会出现"相关≠等价"误判（如"heat"与"combustion"、"monthly"与"yearly"被高估），影响 SURE 准确性。
- **一致性 ≠ 正确性**：若模型反复以不同措辞给出同一错误答案，SURE 会低估不确定性（一致但错误幻觉）。
- **视觉接地缺失**：当前后端仅比较生成文本，不检查与图像的视觉对齐，可能导致对"视觉上被忽略的正确答案"给出过高置信度。
- **主观评测规模有限**：paraphrase 等价性与多语言 correctness judge 的人评样本小，尚不足以做全面验证。
- **评测范围局限**：聚焦 QA 类基准，未扩展到长文生成、摘要、对话、规划、代码合成等开放生成场景。
- **未来方向**：开发多模态联合 grounding-aware 一致度量、改进多语言与数值型 NLI 后端、扩展到更广泛的生成任务、系统化大规模人工评测。

## 研究启发与可借鉴点
1. **跨温度锚点-探测范式可迁移**：将低温度稳定响应作为"语义锚"、高温度响应作为探针的思路，可推广至检索增强、代码生成、agent 规划中用于检测漂移与幻觉。
2. **Frobenius 谱能量替代复杂聚类/熵计算**：SURE 用 ‖W‖_F^2/(MN) 代替语义聚类+熵，简化实现并保持可解释性；可在其他相似性图上复用以快速得置信度。
3. **输入增强与交叉混合策略**：SURE 的 "Mixed-Rephrased" 组合（多个等价变体各采少量）优于同输入多次采样或仅 paraphrase，可作为其他采样式方法的数据增强模板。
4. **Top-k 奇异值诊断**：Spectral energy 的截断分析能揭示不确定信号集中在主模态，可用于诊断模型在哪些语义维度上不稳定。
5. **NLI 后端的公平性设计**：本文在 SURE 与所有相似度基线中锁定同一 entailment 后端，避免后端差异干扰对比；该控制变量策略值得在方法评测中复用。

## 关键术语表
- **BiG-SURE**：本文提出的黑盒不确定性估计框架，基于锚点-探测二分图的跨温度语义一致性能量度量。
- **Anchor（锚点响应）**：在低温度（T_L）下生成的稳定响应集合，作为语义参考基准。
- **Probe（探测响应）**：在高温度（T_H）下生成的多样响应集合，用于测试锚点语义的稳定性。
- **SURE 分数**：归一化锚点-探测相似矩阵的平方谱能量，其补即为不确定度，值域 [0,1]。
- **NLI entailment 后端**：基于自然语言推理模型的蕴涵概率，用作锚点-探测语义相似度度量。
- **Abstention AUROC**：以不确定性分数排序后拒绝部分预测的性能曲线下的面积，越高表示不确定性估计越能区分对错。
- **Cross-temperature semantic agreement**：跨温度语义一致性，指低/高温度响应在意义上保持对齐的程度，是 SURE 的核心信号来源。
- **Spectral energy**：矩阵奇异值平方和，等于 Frobenius 范数平方，本文作为锚点-探测相似矩阵的能量度量。

## 可复现要素
- **数据集**：Trivia-QA、SVAMP、SciQ（en/fr/ja/zh）、OK-VQA，均为公开数据集。
- **代码**：已开源，见 https://github.com/iiscleap/BiG-SURE。
- **项目页**：https://iiscleap.github.io/projects/BiG-SURE。
- **权重/后端**：DeBERTa-large（文本）、mDeBERTa（多语言）NLI 模型使用公开权重，未微调。
- **关键超参**：M=3、N=10、T_L=0.1、T_H=1.0、5 种输入增强各采样 2 次；论文未提及额外随机种子外的调参。
