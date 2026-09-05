---
title: "The-Fragility-of-Jailbreak-Robustness-Across-Operational-Sta"
source: https://arxiv.org/pdf/2608.30748v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 18:45:39"
field: "LLM 安全与越狱鲁棒性评估"
keywords: ["jailbreak robustness", "operational state", "LLM safety", "prompt sensitivity", "state-induced robustness shift", "SSI", "refusal representation"]
innovations: ["首次系统揭示操作状态变化可独立于攻击本身导致 ASR 大幅偏移（最大 56pp）", "建立拒绝相关表征轴预测状态依赖鲁棒性变化（r=0.94）", "提出 SSI 指标量化跨状态鲁棒性稳定性"]
benchmarks: ["AdvBench-50", "MaliciousInstruct", "JailbreakBench"]
---

# 论文速读：The Fragility of Jailbreak Robustness Across Operational States

## 一句话总结
论文首次系统揭示**操作状态诱导的鲁棒性偏移**现象：即使越狱攻击保持不变，仅通过普通的非安全相关系统提示词改变模型的操作状态，就能使攻击成功率(ASR)发生巨大波动（最大偏移达 56 个百分点）。这表明基于单一 vanilla 状态的评估无法全面刻画模型的越狱鲁棒性。

## 研究问题与动机
1. **现有评估范式盲点**：当前越狱评估普遍仅在单一 vanilla 状态（无上下文/默认配置）下测量 ASR，但部署中的 LLM 会因用户交互产生多样化的操作状态（如角色设定、行为指令等）。
2. **未验证的状态稳定性假设**：攻击经过 vanilla 状态优化后，能否在不同操作状态下保持同等成功率尚无研究，存在"看似鲁棒实则脆弱"的潜在盲区。
3. **机制黑箱**：操作状态变化如何影响越狱鲁棒性缺乏表征层面的解释。
4. **实践危害**：模型发布时的 vanilla 评估可能掩盖实际应用中的安全风险。

## 核心贡献（创新点）
1. **发现并命名状态诱导的鲁棒性偏移**：首次系统证明操作状态变化可独立于攻击本身大幅改变越狱成功率，揭示现有 vanilla-state 评估范式的根本缺陷。
2. **建立表征层面的预测机制**：识别出拒绝相关表示轴，证明操作状态引起的鲁棒性变化与该轴上的系统表征差异高度相关（Pearson r=0.94），提供可解释的表征级解释。
3. **提出 SSI（State Sensitivity Indicator）**：设计轻量级指标量化模型跨操作状态的鲁棒性稳定性，补充单一 ASR 的不足，揭示 vanilla ASR 相似但稳定性差异显著的模型。
4. **系统性实证验证**：在 7 个对齐模型、3 类攻击（黑盒/灰盒/白盒）、6 种操作状态下完成全面评估，并提供泛化验证（ paraphrase 稳定性、用户共享角色提示词迁移）。

## 方法详解
**威胁模型扩展**：
- 将鲁棒性从 $ASR(m, a)$ 扩展为 $ASR(m, a, s)$，其中 $s$ 为操作状态，vanilla 状态仅为 $s = s_{vanilla}$ 的特例。
- 操作状态定义为上下文诱导状态，本研究聚焦由系统提示词诱导的状态。

**评估协议**：
1. **状态实例化**：采用 Big Five 人格框架（OCEAN）的 persona-inducing prompts，为每个维度生成独立系统提示词，创建 5 个非 vanilla 操作状态。
2. **攻击生成**：在 vanilla 状态下使用三种越狱攻击（PAIR/LAA/AutoDAN）生成攻击工件。
3. **鲁棒性评估**：固定攻击工件，在不同操作状态下查询目标模型，分别计算 ASR。

**表征分析**：
- 收集 vanilla 与五个 Big Five 状态下共 300 个查询的隐藏表示（生成前最后输入 token）。
- 训练 logistic regression probe 区分越狱成功/失败，提取拒绝相关方向权重向量。
- 将各状态的隐藏表示投影到该轴，计算均值并检验与 ASR 的相关性。

**SSI 指标**：
$$SSI(m, a) = 1 - 2 \cdot \text{Std}_{s \in S}(\text{ASR}(m, a, s))$$
其中 $S$ 包含 vanilla 和五个非 vanilla 状态，ASR 归一化至 0–1 范围。

## 实验与结果
**数据集**：AdvBench-50（主实验）、MaliciousInstruct、JailbreakBench（泛化验证）

**目标模型**：7 个开源对齐 LLM（Llama-2-7B/13B, Llama3-8B, Llama3.1-8B, Qwen2.5-7B, Mistral-7B, Vicuna-7B）

**攻击方法**：PAIR（黑盒）、LAA（灰盒）、AutoDAN（白盒）

**关键结果**：
| 模型-攻击对 | Vanilla ASR | 非 Vanilla 范围 | 最大偏移 |
|---|---|---|---|
| Llama-2-7B + LAA | 2% | 10%–58% | **+56 pp** |
| Llama-2-7B + PAIR | 4% | 0%–14% | +10 pp |
| Llama-2-7B + AutoDAN | 16% | 16%–18% | +2 pp |
| Llama-3.1-8B + AutoDAN | 54% | 2%–58% | **-52 pp** |

**核心发现**：
- 21 个模型-攻击组合中，18 个存在至少一个非 vanilla 状态 ASR 高于 vanilla 状态。
- ASR 偏移方向不一致：部分状态可降低 ASR（如 Llama-3.1-8B 的 Agreeableness 状态从 54%→38%），部分显著升高。
- Llama-3-8B 与 Llama-3.1-8B vanilla ASR 相近（88% vs 90%），但 SSI 差异大（0.881 vs 0.621），揭示稳定性维度的重要性。
- 表征分析：probe AUROC 0.971–0.976，投影与 ASR 强相关（r=0.94），且跨提示集迁移仍保持预测力（Big Five→Public: AUROC 0.890, r=0.61）。
- Paraphrase 实验：同一人格特质的 10 个语义等价改写版本产生一致 ASR 排序（O>N>E>C>A），ANOVA 显著（F=31.27, p<.001, η²=0.735）。

## 相关工作脉络
1. **Jailbreak Evaluation（Chao et al., 2024; Mazeika et al., 2024; Xu et al., 2024a）**：现有评估以单一 vanilla-state ASR 为核心指标，本文将其扩展为状态条件化评估，揭示单点评估的信息损失。
2. **Jailbreak Attacks（PAIR/Chao et al., 2025; LAA/Andriushchenko et al., 2025; AutoDAN/Liu et al., 2024）**：本文使用已有攻击而非开发新攻击，重点考察攻击在状态变化下的鲁棒性而非攻击本身性能。
3. **Persona/Role-Play Jailbreaks（DAN/Shen et al., 2024; DeepInception/Li et al., 2023b）**：前人将 persona 作为攻击组成部分；本文固定攻击，将 persona 作为操作状态的控制变量，方法论定位截然不同。
4. **Safety-Oriented System Prompting（Xu et al., 2024a; SYSFORMER/Sharma et al., 2026）**：安全提示词被优化用于提升安全性；本文使用**非安全相关**提示词实例化操作状态，强调"无意的状态变化"也会产生显著影响。
5. **Refusal-Related Representations（Arditi et al., 2024; Zheng et al., 2024）**：前置工作建立拒绝行为与表征方向的关系；本文将此发现应用于操作状态分析，证明状态差异通过该轴系统分布，形成表征级解释。
6. **Prompt Sensitivity in LLM Evaluation（Mizrahi et al., 2024; Sclar et al., 2024）**：前人研究任务指令/格式变化对评估结果的影响；本文保持有害查询与攻击不变，仅改变目标模型上下文状态，关注的是鲁棒性稳定性而非评估可比性。

## 局限性与未来方向
1. **操作状态覆盖有限**：仅使用 Big Five persona prompts 作为受控状态实例化手段，未涵盖多轮对话、历史累积、个性化、工具使用等更丰富的部署态场景。
2. **表征分析的提示集泛化局限**：probe 在相同提示族内表现更优（Big Five→Big Five AUROC 0.976，跨族降至 0.890），可能捕获提示家族特定伪影，需跨更多模型族/攻击策略/状态类型验证。
3. **代表性状态选择难题**：Big Five 框架仅提供受控分析基准，不能视为部署态空间的规范基础；确定哪些状态集合最能捕捉真实风险评估仍是开放问题。
4. **攻击生成策略**：本文所有攻击工件均在 vanilla 状态下生成，未考虑攻击者针对目标操作状态定制攻击的情况（附录 C.2 表明 state-conditioned 生成收益有限）。

## 研究启发与可借鉴点
1. **评估范式迁移价值**：将"状态敏感度"概念引入其他安全评估场景（如红队测试、对齐评估），建议 Future Work 构建多状态鲁棒性基准。
2. **实验设计借鉴**：
   - 使用 Big Five 人格框架作为**可控状态扰动源**的方法论，可用于研究其他系统提示词对模型行为的影响。
   - 固定攻击+变状态的实验设计逻辑，适用于评估任何"上下文敏感性"研究场景。
3. **分析方法复用**：logistic regression probe + 投影分析的组合可迁移至其他表征可解释性任务（如检测模型在哪种状态下更易产生幻觉、偏见等）。
4. **SSI 指标的推广潜力**：状态敏感性指标的设计思路可扩展至其他评估维度（如语言切换敏感性、领域迁移敏感性）。
5. **团队结合机会**：若团队关注对齐/安全，可将 SSI 纳入模型发布前的常规评估流程，识别"高 vanilla ASR 低稳定性"的模型风险点。

## 关键术语表
**Operational State（操作状态）**：由上下文（系统提示词、对话历史等）诱导的模型运行条件，本文聚焦系统提示词定义的状态。
**State-Induced Robustness Shift（状态诱导鲁棒性偏移）**：攻击固定时，仅因操作状态变化导致的越狱成功率波动现象。
**ASR（Attack Success Rate，攻击成功率）**：越狱成功的查询比例，本文核心评估指标。
**SSI（State Sensitivity Indicator，状态敏感性指标）**：基于跨状态 ASR 标准差计算的鲁棒性稳定性指标，范围 0–1，越高越稳定。
**Refusal-Related Representation Axis（拒绝相关表征轴）**：由 Arditi et al. (2024) 发现的、区分模型拒绝/顺从行为的线性表示方向。
**Vanilla State（Vanilla 状态）**：无额外上下文/默认配置的模型操作状态，当前评估基准。
**Big Five Persona Prompts（大五人格提示词）**：基于 OCEAN 人格框架的系统提示词，用于实例化五种非 vanilla 操作状态。
**Probe（探针）**：线性分类器，用于从隐藏表示中预测特定行为属性（本文为越狱成功/失败）。

## 可复现要素
- **数据集**：AdvBench-50、MaliciousInstruct、JailbreakBench（均为公开基准）；用户共享角色提示词来自 Hugging Face 社区数据集（论文提供过滤标准与随机种子 random_state=42）。
- **代码/权重**：论文未提供开源代码仓库声明；目标模型均从 HuggingFace 公开下载（表 4 提供链接）；攻击方法引用官方 GitHub 仓库（表 5）。
- **关键超参**：解码温度 T=0（保证可重复性）；GPT-4o 生成 paraphrase；BERTScore F1≥0.9 筛选；paraphrase 词数约束 ±10%。
- **评估工具**：GPT-4（gpt-4-0613）作为 judge，评分≥10 视为越狱成功；keyword-based judge 用于附录泛化实验。
