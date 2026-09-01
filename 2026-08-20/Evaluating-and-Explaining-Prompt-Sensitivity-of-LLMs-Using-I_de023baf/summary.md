---
title: "Evaluating-and-Explaining-Prompt-Sensitivity-of-LLMs-Using-I"
source: https://arxiv.org/pdf/2608.18539v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 21:52:07"
field: "大语言模型可解释性与鲁棒性"
keywords: ["prompt sensitivity", "interaction explanation", "LLM interpretability", "prompt robustness", "fine-grained evaluation", "model stability"]
innovations: ["提出基于交互分解的细粒度IPS指标，揭示输出稳定下潜在的内部不稳定性", "系统发现SFT/规模/架构/few-shot四个降敏因素及其共性机制——主要稳定低阶交互"]
benchmarks: ["ARC", "MMLU", "Dolly-15k"]
---

# 论文速读：Evaluating-and-Explaining-Prompt-Sensitivity-of-LLMs-Using-I

## 一句话总结
本文提出基于"交互（interactions）"的细粒度方法 IPS 指标来评估和解释 LLM 的 prompt 敏感性，揭示了 SFT、模型规模、密集架构和 few-shot 学习能降低敏感性，且其共同机制是主要稳定低阶交互。

## 研究问题与动机
- **现有方法的粗粒度局限**：当前评估 prompt 敏感性的方法（如 Sclar et al., 2024; Chatterjee et al., 2024）仅关注最终输出的变化（如任务准确率或输出一致性），无法解释内部原因。
- **输出稳定不等于内部稳定**：论文发现即使 LLM 最终输出相同，输入变量间的交互模式也可能发生剧烈波动，说明粗粒度指标会漏检潜在不稳定性。
- **缺乏对影响因素的系统分析**：目前尚不清楚哪些模型设计因素能降低 prompt 敏感性，以及其内在机制是什么。
- **细粒度分析工具的缺失**：需要一种能从内部逻辑层面分解和量化 prompt 敏感性的可解释方法。

## 核心贡献（创新点）
1. **提出 IPS 指标**：将 DNN 输出分解为交互效应之和，通过量化不同 prompt 模板下交互的变化来度量 prompt 敏感性，区别于传统的输出级粗粒度指标。
2. **发现输出稳定下的潜在不稳定性**：即使 LLM 最终预测不变，60%-80% 的显著交互仍不稳定（符号反转或幅度剧变），揭示传统指标无法捕捉的隐性风险。
3. **系统识别四个降敏因素**：通过控制变量实验发现 SFT、增大模型规模、使用密集架构（vs MoE）、few-shot 学习均能显著降低 prompt 敏感性。
4. **揭示共性机制**：四个因素的共同机制是主要降低**低阶交互**（涉及少量输入变量的交互）的敏感性，而非高阶交互——这与直觉相悖（高阶交互本身最不稳定）。

## 方法详解
- **交互理论框架**：给定输入 x 含 n 个变量（词/token），任意子集 S ⊆ N 对应一个 AND 交互，其效应为 I_S = Σ_{S'⊂S} (-1)^{|S|-|S'|} · v(x_{S'})，其中 v(x) 是 DNN 对正确答案的对数几率。逻辑模型 φ(x) = Σ_{S⊆N} I_S 精确等于 v(x)（万能匹配性质，Theorem 3.1）。
- **AND-OR 交互提取**：采用 Li & Zhang (2023) 的方法提取 AND 和 OR 两种交互，通过 LASSO 类损失（最小化 |I_S^{AND}| + |I_S^{OR}|）实现稀疏性。
- **IPS 指标定义**：IPS = E_x[E_{T,Ť}[ (1/|Ω_union|) Σ_{S∈Ω_union} |Ĩ_S(x|T) - Ĩ_S(x|Ť)| / (|Ĩ_S(x|T)| + |Ĩ_S(x|Ť)|) ]]，即对显著交互集合取对称平均绝对百分比误差，阈值 τ=0.1 区分显著交互与噪声。
- **阶数分类**：将交互按涉及变量数分为低阶（≤n/3）、中阶、高阶（>2n/3），分别计算 IPS^{low}, IPS^{mid}, IPS^{high}。

## 实验与结果
- **数据集**：ARC（推理题）和 MMLU（多任务理解），另在 Dolly-15k（开放生成）上验证。
- **模型**：50 个开源 LLM，覆盖 Llama、Mistral、Qwen、OLMo、InternLM 五个家族，含 Base/Instruct/Chat/MoE/Dense 多种变体。
- **核心结果**：
  - IPS 范围 1.268（Qwen2.5-72B-Instruct，最稳定）至 1.752（Mistral-7B-v0.3，最敏感）。
  - IPS 与输出一致性呈负相关（验证可靠性），τ 敏感性实验中 Spearman ρ=0.9905、Pearson r=0.9957。
  - 四个因素均降低敏感性；从 0-shot 到 1-shot 下降幅度最大。
  - MoE 比 Dense 更敏感（即使控制激活参数量，Table 7）。
  - **阶数分析**：IPS^{high} > IPS^{mid} > IPS^{low}；四个因素主要降低 IPS^{low}，对 IPS^{high} 改善有限。
- **稳健性**：在开放生成任务（Dolly-15k）、语义改写和指令重排扰动下结论一致。

## 相关工作脉络
- **Prompt 敏感性评测**：Sclar et al. (2024) 提出量化 prompt 格式变化的敏感性；Chatterjee et al. (2024) 的 POSIX 指标；Lu et al. (2024) 的 promptrobust——均为输出级粗粒度评估，本文转向内部交互层面。
- **博弈论交互解释 DNN**：Ren et al. (2023a) 提出交互解释框架并给出忠实性理论保证；Li & Zhang (2023) 证明 DNN 编码稀疏交互；Chen et al. (2024)、Zhou et al. (2024) 将其用于泛化分析——本文首次将交互框架应用于 prompt 敏感性分析。
- **模型可解释性基线对比**：Appendix H 证明 Cosine Similarity 和 L2 Distance 等表征级指标与 IPS 趋势矛盾，凸显交互方法的解释力优势。

## 局限性与未来方向
- **计算复杂度**：交互计算需评估 2^n 个掩码样本，长文本场景成本极高；虽可通过选择性变量和短语聚合缓解（Appendix K/L），但仍未根本解决。
- **高阶交互不稳定性难抑制**：四个因素主要稳定低阶交互，对本质上最不稳定的高阶交互改善有限——这是未来训练方法需要攻克的方向。
- **实验集中在 MCQ 和简单扰动**：虽然验证了开放任务和语义改写，但更复杂的对抗性 prompt 攻击下的行为尚待探索。

## 研究启发与可借鉴点
- **交互分解用于模型诊断**：将模型输出分解为交互项之和的思路可迁移至其他稳定性/鲁棒性分析任务（如 adversarial robustness、distribution shift）。
- **细粒度指标设计范式**：IPS 的对称 MAPE 形式和显著交互筛选策略可作为细粒度评估指标的通用设计模板。
- **阶数分层分析**：按交互阶数分层研究模型行为（低阶 vs 高阶）提供了新的可解释性分析维度，值得在模型压缩、泛化等方向复用。
- **控制变量实验设计**：通过同家族模型对比（Base/Instruct、Dense/MoE、不同 scale）分离多因素影响的实验设计严谨，可借鉴于其他模型属性研究。

## 关键术语表
- **Interaction（交互）**：输入变量集合 S 之间的非线性联合效应 I_S，表示这些变量共同作用对模型输出的贡献。
- **AND/OR 交互**：AND 交互要求 S 中所有变量同时存在才触发；OR 交互要求 S 中至少一个变量存在即触发。
- **Salient Interaction（显著交互）**：绝对值超过阈值 τ 的交互，代表对模型推理有实质影响的交互模式。
- **IPS（Interaction-based Prompt Sensitivity）**：基于交互变化量度化的 prompt 敏感性指标，值越大表示越敏感。
- **Order of Interaction（交互阶数）**：交互涉及的输入变量数目 |S|，分低/中/高三档。
- **Universal Matching Property**：定理保证逻辑模型 φ(x) 对所有 2^n 个掩码样本的输出与 DNN 完全一致。

## 可复现要素
- **数据集**：ARC（公开）、MMLU（公开）、Dolly-15k（公开）；prompt 模板设计见 Appendix E.5。
- **代码/权重**：论文未明确声明开源代码，但使用 50 个开源 LLM（HuggingFace 可获取）。
- **关键超参**：显著交互阈值 τ=0.1（附录 F.8 验证 0.05–0.20 范围结论稳健）；生成策略为 greedy decoding（do_sample=False）；mask token 因模型而异（Appendix E.4）。
- **硬件**：4× NVIDIA Tesla V100-DGXS 32GB，torch.float16。
