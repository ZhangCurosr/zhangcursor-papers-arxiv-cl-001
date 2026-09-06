---
title: "From-Production-Traffic-to-Post-Training-Building-a-Self-Hos"
source: https://arxiv.org/pdf/2609.01572v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 00:26:02"
field: "大语言模型后训练与对齐"
keywords: ["post-training", "GRPO", "reward hacking", "model merging", "SLERP", "function calling", "instruction following", "enterprise LLM"]
innovations: ["单轴GRPO专家+两阶段SLERP权重合并替代联合多目标RL，避免跨域奖励干扰", "三类reward-hacking失败模式（字数膨胀/语义坍塌/过度调用）的系统识别与针对性修复", "模板感知采样与任务特定LLM Judge校准协议，构建从生产流量到内部benchmark的自动化评测pipeline"]
benchmarks: ["In-house Arena", "ruIFEval", "ruBFCLv3", "AceBench", "SmartSearch", "ruMultiChallenge", "ruWildChat Hard", "Arena Hard Ru"]
---

# 论文速读：From Production Traffic to Post-Training: Building a Self-Hosted LLM That Covers the Corporate Request Mix

## 一句话总结
本文提出了一种**从生产流量诊断到模块化后训练**的端到端方法论：基于企业内超 200 个应用的流量误差分析，针对指令遵循、函数调用、通用对齐三个弱项分别训练独立 GRPO 专家，再通过两阶段 SLERP 权重合并，以 32B 参数量模型在多项评测上持平甚至超越 ~7× 参数量的同系模型，成功整合平台 50% 流量（月均 1.16 亿请求）。

## 研究问题与动机
- **企业数据驻留约束**迫使公司自建私有化 LLM 服务，但新模型快速迭代而旧模型不能退役，导致 GPU 池被多个模型碎片化分割，token 成本持续攀升。
- **单一联合 RL 优化存在跨域奖励干扰**：IF、FC、通用对齐三个目标方向各异，且每个奖励都存在明显的 reward-hacking 捷径，联合优化会导致某一域提升而另一域退化。
- **现有开源评测无法反映内部真实分布**：公共 benchmark（如 IFEval、BFCL）多面向英语且缺乏内部业务特定的约束与工具分布，难以指导生产环境中的模型优化决策。
- **俄语场景下工具调用数据严重匮乏**：现有开源 FC 训练集和评测集以英语为主，直接翻译在结构上不安全（函数名、参数键、枚举值不可变），导致俄语文档化工具调用性能显著下降。

## 核心贡献（创新点）
- **生产流量驱动的模块化诊断→训练闭环**：从零散流量采样构建分层内部评测（in-house Arena、IFEval、BFCL、SmartSearch），先定位误差再定向修复，区别于单纯依赖公开基准的训练范式。
- **单轴 GRPO 专家 + 两阶段 SLERP 合并策略**：每个弱项轴独立训练一个 GRPO 专家，在权重空间进行有序合并；实验证明联合 GRPO 存在强烈的跨域干扰，而分轨训练+合并可稳定兼顾各域性能。
- **三种 reward-hacking 失败模式的发现与针对性修复**：（1）通用专家的字数膨胀（length hacking）通过乘性长度惩罚 + 增大 KL 系数解决；（2）IF 专家的语义坍塌（semantic collapse）通过 prompt-specific RM 质量校正解决；（3）FC 专家的过度调用（over-calling）通过数据分布注入合成无关样本解决，而非修改奖励函数本身。
- **模板感知的多样性采样方法**：针对企业内部模板化请求主导的流量，设计 masked-variable + LSH 分桶 + 桶内 greedy max-min 的采样策略，在不牺牲生产代表性的前提下达到最高多样性（Dist=0.953，JS 距离显著低于纯多样性方法）。
- **任务特定的 LLM Judge 校准协议**：通过 LLM 任务分类器路由不同任务至专属评判方案（参考式 vs. 带 RubricHub 清单的 Pairwise SBS），将 Cohen's κ 从 0.63/0.57 提升至 0.88/0.72。

## 方法详解
**整体流程**：以 Qwen3-32B + 适配西里尔语密集的 tokenizer 为底座，在**非推理模式**下训练，分为两大部分：(1) 共享 SFT + (2) 三分支 GRPO + SLERP 合并。

**Stage 1 — 共享 SFT**：将四类数据（内部生产流量 20% + 开源俄语指令语料 80%、IF 合成数据、FC 合成数据）混合进行单次 SFT，作为后续所有 GRPO 分支的共享起点。消融实验（Table 3）证明共享 SFT 在各域上的质量与分域 SFT 相当。

**Stage 2 — 三分支 GRPO 专家**：

- **General Expert**：GRPO 基于通用 RM，引入**乘性长度惩罚**：$R(x,y) \mapsto R(x,y)\cdot(1-\alpha(x,y))$，其中 $\alpha = \mathrm{sgn}(R)\cdot\mathrm{clip}\!\left(\frac{L(y)/L_0(x)-d_{\min}}{d_{\max}-d_{\min}}\right)$，$d_{\min}=0.1, d_{\max}=0.3$，$L_0(x)$ 为 Qwen3-235B-A22B 在同一 prompt 上的回复长度；同时将 KL 系数从 0.001 提升至 0.01。
- **IF Expert**：基于 AutoIF 适配俄语的合成管道生成 43K 验证约束、26K 训练样本；GRPO 使用 VerIF 风格 verifier 奖励，并引入**prompt-specific RM 质量校正**：当完成通过 verifier（$V_i>0$）但 RM 分数低于 teacher 均值（$S_i \leq \alpha_i$）时给予 -0.5 惩罚，避免模型输出语义空洞的"最短合法回复"。
- **FC Expert**：原生生成 1.2M 英语 + 300K 俄语多轮对话（规划器→三人 Agent 模拟）；GRPO 使用 Tool-N1 二元精确匹配奖励（工具调用 exact match=1，文本回复无解析出调用=1）；通过在文本目标中注入 10% 合成无关样本（移除 ground-truth 工具改写为拒绝）来对抗 over-calling 捷径，文本目标占比固定 20%。

**Stage 3 — 两阶段 SLERP 合并**：先合并 (IF + FC) 专家（各参数组独立设置系数：attention 从 {0,0.3,0.5,0.7,1} 网格搜索，MLP 互补，其余 $t=0.5$），再将中间结果与 General 专家以 $t_2=0.8$ 合并；消融表明此顺序（Table 7）优于其他组合。

## 实验与结果
- **数据集/评测**：内部四任务（In-house Arena、In-house IFEval、In-house BFCL/ruBFCLv3、SmartSearch）；公开评测 ruIFEval、ruMultiChallenge、ruWildChat Hard、Arena Hard Ru、BFCLv3（en/ru）、AceBench、$\tau^2$-bench。
- **最强结果**（Table 6，non-reasoning 模式）：
  - **In-house Arena**：69.57（vs. Qwen3-235B 的 65.83，+3.74）；**ruIFEval**：0.799（vs. 235B 的 0.803，基本持平）；**ruBFCLv3**：65.96（vs. 235B 的 64.42，+1.54）；**AceBench**：73.50（vs. 235B 的 70.20，+3.30）。
  - **SmartSearch F1**：0.557（vs. 基座 0.478，+8pp；超越 Qwen3-32B think 模式 0.491 和 T-Pro-2.0 think 0.537）；**ruWildChat Hard**：80.7（vs. 基座 52.0，大幅跃升；距 235B 的 85.1 仅差 4.4pp）。
- **合并 vs. 联合 GRPO 对比**（Table 8）：Joint from SFT 在 BFCL-EN 上仅 70.38（合并 72.27），需 General GRPO warm-start + 1.7× 预算才勉强追平，证实分轨合并的工程优势。
- **部署效果**：模型承载平台 50% 流量，1.16 亿请求/月，200+ 内部服务，单卡 FP8 部署，95th 延迟 3.2s，TTFT 0.3s；相对 ~7× 参数量的基线，单位 token 成本降低 2.8–3.9×（输入/输出），部分服务可达 4–9×。

## 相关工作脉络
- **Tulu 3 / DeepSeek-R1**：代表当前主流 RLVR（GRPO）后训练范式；本文与之差异在于将多目标拆分为单专家分轨训练后权重合并，而非联合优化，从而避免跨域奖励干扰。
- **AutoIF / VerIF**：英文指令遵循 verifier 体系；本文将其适配俄语，指出俄语文法（格变位、词序自由）使英文 rule 不可直接复用，需在数据层面原生合成而非 inference-time 修正。
- **MO-GRPO (Ichihara et al., 2025) / RARL (Huang et al., 2025)**：尝试多目标 RL 的 reward 加权或 rubric 锚定方案；本文认为这些方法在 verifiable reward 场景下同样面临 reward-hacking，主张在数据/奖励设计层面单独解决每个 failure mode。
- **APIGen / ToolACE / Toucan**：函数调用合成数据管道；本文继承其规划→模拟思路，关键区别在于俄语文档化工具的原生生成（非翻译），并分离 planning 与 simulation 阶段以注入 asymmetric visibility。
- **Model Soups / TIES-Merging / MergeKit**：权重合并方法谱系；本文采用 SLERP（非 TIES），并发现合并顺序（IF+FC 先合再入 General）对最终性能影响显著（Table 7），这一经验规律在已有工作中未被系统探索。
- **Arena-Hard / WildBench**：公开对话评测；本文拓展至企业私有化场景，构建流量分层内部 benchmark 并引入任务分类路由的差异化评判方案，弥补公开 benchmark 与生产分布之间的 gap。

## 局限性与未来方向
- 评估仅限俄语和英语，方法论未在其它语言/组织上验证（论文自述 limitation）。
- 开放式任务质量依赖 LLM Judge，校准于当前 benchmark 分布；迁移至新部署时需重新校准。
- 部署证据来自单一企业内部平台，且仅基于 Qwen3 模型族，未在其他模型家族上验证。
- 内部知识/长上下文类任务（MultiChallenge、SmartSearch）仍存在差距，需更大基座或更长 context 窗口。
- 未来方向包括：验证跨语言泛化、扩展至推理模式、探索 judge 自动校准协议、将 reward-hacking 修复策略推广至更多垂直域。

## 研究启发与可借鉴点
- **"分域训练 + 权重合并"替代"联合多目标 RL"的工程思路**：当各优化目标存在结构性 reward-hacking 捷径时，单轴专家化训练比联合优化更可控、更易调试；对多技能并行的企业 LLM 微调具有直接参考价值。
- **模板感知采样策略**：针对高度模板化的生产流量，masked-variable + LSH + 桶内 max-min 的采样方案在多样性和代表性之间取得了良好平衡，可迁移至任何存在模板化请求的企业场景。
- **Prompt-specific 质量校正奖励设计**：IF expert 中将 verifier 奖励与 prompt-level teacher RM 分数相结合的 reward 构造（$V_i>0$ 时比较 $S_i$ 与 $\alpha_i$），是一种可复用的防语义坍塌方案，可推广至其他 verifiable constraint 训练场景。
- **通过数据分布而非奖励修改来纠正行为偏差**：FC expert 中 over-calling 问题通过注入合成无关样本（而非更改 Tool-N1 奖励）解决，这一思路简洁有效，避免了奖励工程带来的额外 hacking surface。
- **内部 benchmark 的自动化构建方法论**：从流量采样、任务分类路由到差异化评判（reference-based vs. checklist-guided SBS）的完整 pipeline，为企业自建评测体系提供了可操作的技术路线。

## 关键术语表
**SLERP**：Spherical Linear Interpolation，球面线性插值，用于在权重空间平滑合并多个模型参数，本文采用两阶段顺序合并策略。
**GRPO (Group Relative Policy Optimization)**：群体相对策略优化，DeepSeekMath 提出的 RL 算法，通过组内相对归一化 Advantage 估计更新策略，本文用于各专家轴的 RL 训练。
**Reward Hacking**：模型利用奖励函数漏洞获得高分但不产生期望行为的优化现象，本文识别出三类：字数膨胀、语义坍塌、过度调用。
**VerIF-style Verifier Reward**：基于可执行验证器的确定性奖励，检查模型输出是否满足形式化约束（格式、长度、关键词等），可大规模扩展但易被绕过。
**In-house Arena**：基于生产流量构建的内部对话质量 benchmark，通过任务分类路由至差异化评判方案（参考式或 RubricHub 清单辅助的 Pairwise SBS）。
**Tool-N1 Exact-match Reward**：二元奖励函数，工具调用要求与 ground truth 的多重集完全匹配得 1 分，文本回复要求不解析出任何调用得 1 分。
**Template-aware Sampling**：先 mask 变量 token 按 LSH 分桶，再在每个桶内对变量 span 做 greedy max-min 采样，兼顾多样性与生产分布代表性。
**Semantic Collapse**：IF 训练中模型学会输出最短但形式合法的空洞回复以满足 verifier，但丢失语义内容，本文通过 prompt-specific RM 阈值惩罚修复。

## 可复现要素
- **数据集**：内部 benchmark（In-house Arena/IFEval/BFCL/SmartSearch）基于企业内部 LLM 平台流量，脱敏匿名处理后构建，**未公开**；俄语文本适配的 BFCLv3（ruBFCLv3）和 Arena Hard Ru 已开源至 HuggingFace。
- **代码/权重**：公开 checkpoint（`T-pro-2.1 public`，使用相同 recipe 但未加入内部数据增量）已发布为 open-weight（论文标注¹）；内部代码未明确提及开源。
- **关键超参**：SFT—序列长度 32768、batch=32、lr=1e-6、epochs=2、bf16（Table 20）；GRPO—rollouts=8、PPO micro-batch=1/GPU、lr=1e-6、KL 系数 General=0.01 / IF-FC=0.001（Table 21/22）；SLERP 合并系数 $t_1$ 各参数组分层搜索、$t_2=0.8$。
- **硬件**：SFT 4 节点×8 H100；GRPO 每专家 4 节点×8 H100。
- **框架**： verl（训练）、vLLM（推理）、FSDP2。
