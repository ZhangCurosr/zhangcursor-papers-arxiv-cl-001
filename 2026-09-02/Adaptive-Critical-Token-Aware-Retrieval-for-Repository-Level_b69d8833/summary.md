---
title: "Adaptive-Critical-Token-Aware-Retrieval-for-Repository-Level"
source: https://arxiv.org/pdf/2609.01601v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 20:53:23"
field: "仓库级代码生成"
keywords: ["仓库级代码生成", "关键token", "检索增强生成", "大语言模型", "动态检索"]
innovations: ["首次系统定义并量化关键token，揭示其在生成错误中的决定性作用", "提出生成过程中动态按需检索框架，在关键token处触发针对性上下文修正", "设计位置感知双端高斯加权池化方法增强密集检索器的上下文聚合质量"]
benchmarks: ["RepoExec", "CoderEval"]
---

# 论文速读：Adaptive-Critical-Token-Aware-Retrieval-for-Repository-Level

## 一句话总结
本文提出了ACTOR框架，通过识别代码生成过程中的"关键token"并在推理时按需触发针对性检索，解决了现有仓库级代码生成方法仅停留在任务级一次性检索、缺乏细粒度token级别上下文支持的缺陷，在RepoExec和CoderEval上分别实现最高8.4%和15.4%的相对提升。

## 研究问题与动机
1. 现有仓库级RAG方法将检索到的上下文作为任务级整体支持，隐含假设一组相关上下文可贯穿整个生成过程，忽略了代码生成是高度路径依赖的自回归过程，不同生成位置可能需要不同的上下文信息。
2. 代码生成错误并非均匀累积，而是集中在少数"关键token"位置（如错误的API名、变量引用、索引等），一旦这些位置生成错误，后续代码将偏离正确语义路径，最终导致功能级失败。
3. 关键token具有语法多样性（可能是API调用、注释、索引表达式等），无法通过固定的语法规则或单一预生成检索步骤覆盖，需要一种生成时的动态识别与按需检索机制。

## 核心贡献（创新点）
1. **首次系统提出并量化"关键token"概念**：提出token不匹配（Token Mismatch）、不确定性（Uncertainty）和后续注意力影响（Subsequent Attention Influence）三个维度来定义和识别关键token，并揭示其仅占生成位置5–11%却集中了大部分错误。
2. **设计了基于关键token的动态按需检索框架**：在自回归生成过程中实时监测每个token，当识别为关键token时立即触发二次检索并修正当前token，区别于现有方法的事前一次性检索。
3. **提出位置感知加权密集检索器**：设计了双端高斯加权池化方法（Position-Aware Weighted Pooling），赋予序列头部（如函数签名）和尾部（如紧邻未生成代码处）更高权重，提升检索上下文对生成过程的 informative程度。
4. **系统性分析了关键token的构成与句法特征**：发现模型错误集中在keywords（异常偏高的7.64倍），而注意力集中于operators/delimiters，揭示了大模型在关键信息定位能力上的差异。

## 方法详解
**整体架构**：分为离线训练和在线推理两个阶段，核心由关键token分类器、位置感知加权检索器和动态修正推理三部分组成。

**离线训练——关键token标签构建**：
- **Token Mismatch**：模型top-1预测与ground truth不一致，即 $x_i \neq \hat{x}_i$。
- **Uncertainty**：使用熵度量 $\mathcal{H}_i = -\sum_{v \in \mathcal{V}} p_i(v) \log p_i(v)$，熵越高表示不确定性越大。
- **Subsequent Attention Influence**：定义 $a_{\mathrm{mean}}(i) = \mathrm{Mean}_{j \in \{i+1, \dots, i+k\}} \mathrm{Top}_K(A_{j,i})$，衡量当前位置被后续位置注意力的集中程度（取 $K=5$）。
- 满足任一条件（或不匹配、或超过阈值）则标记为critical；使用teacher-forcing抑制误差传播，并通过自信息（$I(x_i) = -\log_2 P(x_i | x_{<i})$）筛选"hard negatives"平衡类别。

**离线训练——分类器**：
- 使用3层MLP作为分类器，输入为Code-LLM最后一层隐藏状态 $h_n^{(L)}$，输出critical/non-critical概率，采用交叉熵损失优化，参数量仅4.01–10.01M，远小于基础模型（1B–13B）。

**在线推理——位置感知加权检索器**：
- 对长度为 $L$ 的文本片段，定义双端高斯权重：
  $$w_i = \exp\left(-\frac{i^2}{2\sigma^2 L^2}\right) + \exp\left(-\frac{(i-(L-1))^2}{2\sigma^2 L^2}\right)$$
- 经softmax归一化后加权池化token嵌入：$\mathbf{v} = \sum_{i=0}^{L-1} \alpha_i \mathbf{e}_i$，使检索更偏向函数签名（头部）和紧邻未生成代码处（尾部）。

**在线推理——关键token引导的动态修正**（Algorithm 1）：
- 初始化：使用目标函数prompt检索初始上下文。
- 逐token生成：提取当前hidden state，经分类器判断是否为关键token。
- 若为关键token：用当前prompt+已生成代码+当前token构建新query，进行二次检索获取更新上下文，重新解码当前token，并保持新上下文指导后续生成。
- 跳过line break和space token的检查以降低开销。

## 实验与结果
**数据集**：
- RepoExec：355个Python仓库级代码生成任务，含单元测试套件。
- CoderEval：230个真实开源项目Python代码生成任务，统一Docker环境评测。

**基线方法**：RawPrompt、RawRAG（单次RAG）、RepoCoder（迭代RAG）、RLCoder（RL优化检索）。

**生成模型**：DeepSeek-Coder（1.3B、6.7B）、CodeLlama（7B、13B）；检索器：UniXcoder。

**主要结果**：
- **CodeLlama-7B**：RepoExec Pass@5提升 **8.4%**（相对基线），CoderEval Pass@5提升 **15.4%**。
- **CodeLlama-13B**：CoderEval Pass@5达到 **39.57%**，相对RLCoder提升 **11.0%**。
- **DeepSeek-Coder-1.3B**：CoderEval Pass@5达33.48%，相对RLCoder提升5.5%。
- ACTOR在不同模型规模上均一致超越SOTA基线，且多pass成功率（Pass@5）提升尤为显著。

**效率分析**（DSCoder-1B）：
- 单token延迟仅增加约3.3ms（22.6ms vs 19.3ms），因为修正仅在稀疏的关键token位置触发。
- 端到端执行时间2.66s，远低于RepoCoder的4.50s；生成token数最少（94.59），代码结构更紧凑。

**消融实验**：
- 三种标注规则（Mismatch、Uncertainty、Attention）缺一不可，不同规则在不同数据集上重要性不同。
- 移除动态推理导致CoderEval Pass@5下降10.4%（中位数约15%），移除加权检索器导致约7–9%下降，两者互补不可替代。
- 超参数敏感性低：25种配置下Pass@1方差仅≈2.83，性能稳定。

## 相关工作脉络
1. **RepoCoder**（迭代RAG）：通过多轮生成-检索循环逐步精炼上下文，但与ACTOR不同，其检索仍发生在任务级而非token级，且计算开销更高（4.50s/sample vs ACTOR的2.66s）。
2. **RLCoder**（RL优化检索）：使用强化学习训练检索器选择有价值上下文，并引入stop signal机制；ACTOR在此基础上进一步细化到生成过程中的token级别动态修正。
3. **Cocomic**（图结构检索）：基于项目依赖图进行上下文检索，关注跨文件依赖关系；ACTOR聚焦于自回归过程中的关键位置感知，两者正交可结合。
4. **RepoFormer**（选择性检索决策）：判断是否对当前任务触发检索；ACTOR更进一步，在生成过程中实时判断每个token是否需要检索支持。
5. **仓库级代码生成错误分析**（Wang等、Zhang等）：聚焦于最终输出的错误类型分类和hallucination分析；ACTOR从token级别的错误前兆（不匹配、不确定性、注意力）切入，建立"token→错误"的因果链路。

## 局限性与未来方向
1. **计算开销限制**：迭代上下文更新和KV cache重计算在大规模实时系统中可能构成部署挑战，需进一步优化（如集成KV cache压缩技术）。
2. **基准范围局限**：仅在RepoExec和CoderEval上验证，可能无法覆盖所有真实软件项目的复杂性；未来可扩展至更多样化的基准（如SWE-bench等）。
3. **语言覆盖**：当前实验仅限Python，关键token的识别阈值和分布特征可能在不同语言中有所不同。
4. **关键token标签为代理信号**：真实的关键性取决于对最终程序的下游影响，当前通过不匹配、不确定性、注意力等可观测代理近似，可能遗漏部分功能决定性token。

## 研究启发与可借鉴点
1. **token级错误归因分析思路**：将生成过程中的token划分为critical/non-critical两类，并结合不匹配、不确定性、注意力三维度进行联合判断，可迁移到文本生成、翻译等序列生成任务中的错误分析。
2. **位置感知加权池化设计**：双端高斯加权策略为密集检索器的embedding聚合提供了新思路，可应用于任何需要保留序列首尾语义信息的检索场景。
3. **稀疏动态修正机制**：ACTOR在每步生成时仅需调用轻量分类器（4–10M参数），仅在高置信度关键token处触发昂贵的二次检索，实现了精度与效率的平衡，可作为"稀疏干预"范式的参考。
4. **实验设计的可借鉴**：通过teacher-forcing抑制误差传播、用自信息筛选hard negatives进行类别平衡、多维度消融（规则级+模块级）和敏感性分析，均为严谨的实证研究提供了方法论范例。
5. **结合本团队方向的机会**：关键token识别与仓库级代码生成的结合思路，可拓展至代码审查（识别高风险diff位置）、bug修复（定位修复关键点）等下游任务。

## 关键术语表
**Critical Token（关键token）**：在自回归生成过程中，其正确与否对最终代码结果具有决定性影响的少量token，其错误会导致后续代码沿错误语义路径生成。

**Position-Aware Weighted Pooling（位置感知加权池化）**：为密集检索器设计的加权聚合方法，对序列头部和尾部赋予更高权重（双端高斯分布），使检索上下文更利于生成。

**Subsequent Attention Influence（后续注意力影响）**：衡量当前位置被后续若干位置注意力权重的Top-K均值，反映当前token对后续生成的潜在影响力。

**Teacher-Forcing（教师强制）**：在评估token重要性时，用ground truth替换模型预测，以抑制误差传播并更准确地评估单个token的关键性。

**Self-Information/Supriscal（自信息/惊讶值）**：基于信息论的token重要性度量，$I(x_i) = -\log_2 P(x_i|x_{<i})$，自信息越大表示该token越难预测、携带信息越多。

**Pass@k**：在k次生成采样中至少有一次通过所有单元测试的概率估计，是代码生成任务的核心评测指标。

**Dense Retriever（密集检索器）**：将文本映射到高密度向量空间并通过余弦相似度进行检索的模型，如UniXcoder。

**Repository-Level Code Generation（仓库级代码生成）**：要求在已有代码仓库上下文中生成与项目API、依赖、编码约定一致的代码的任务。

## 可复现要素
- **数据集**：RepoExec、CoderEval、RepoST-Train（用于训练数据构建）
- **代码/权重开源**：代码和数据已公开于 https://github.com/DeepSoftwareAnalytics/ACToR
- **关键超参数**：检索上下文长度1K tokens，最多检索10个片段；温度0.6；最大生成token数512；后续注意力阈值0.05；不确定性阈值0.8；Top-K=5（注意力）；σ为超参数控制高斯峰值锐度
- **硬件环境**：8× NVIDIA A800 GPU（80GB），Intel Xeon Gold 6348 CPU
- **检索器**：UniXcoder
- **生成模型**：DeepSeek-Coder（1.3B、6.7B）、CodeLlama（7B、13B）
