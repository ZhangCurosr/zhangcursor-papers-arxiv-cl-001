---
title: "Groundhog-Bit-Flip-Attack-Seeding-Infinite-Generation-Loops"
source: https://arxiv.org/pdf/2608.25276v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 23:41:32"
field: "大语言模型安全与可靠性"
keywords: ["Mixture-of-Experts", "Bit-Flip Attack", "Denial-of-Wallet", "LLM Security", "Expert Specialization", "Routing Vulnerability"]
innovations: ["发现MoE专家对终止令牌的高度专业化并用于可用性攻击", "提出缓存隐藏状态加速的无推理位翻转攻击方法", "首个针对MoE路由层的拒绝支付位翻转攻击框架"]
benchmarks: ["AGNews", "SST-2", "Samsum", "SQuAD 2.0", "Qwen3-Coder Agentic Tasks"]
---

# 论文速读：Groundhog-Bit-Flip-Attack-Seeding-Infinite-Generation-Loops

## 一句话总结
提出了GBFA，首个针对MoE架构LLM的位翻转拒绝支付（DoW）攻击，通过识别并翻转路由层中与EOS/EOT生成强相关的专家参数，以平均少于4个专家的微小扰动，使模型输出膨胀达平均5912%，同时保持语义连贯性。

## 研究问题与动机
1. **MoE架构引入新的攻击面**：MoE通过稀疏专家激活提升推理效率，但特定专家与终止令牌（如EOS）的高度相关性使得路由机制成为可用性攻击的脆弱点。
2. **现有位翻转攻击仅关注完整性破坏**：已有BFA工作（如Rakin等）主要研究精度下降或输出篡改，而针对MoE的DoW攻击尚未探索。
3. **Agentic应用中的成本风险**：随着LLM Agent广泛应用，输出膨胀直接转化为用户账单激增，构成现实威胁。
4. **终止行为可由少量专家控制**：论文发现EOS和EOT生成集中于极少数专家，这为精准攻击提供了目标。

## 核心贡献（创新点）
1. **发现专家-令牌专业化现象**：提出Target Activation Shift（τ）和Target Gate Shift（Δg）度量，证明MoE专家对EOS/EOT存在高度专业化。
2. **系统性专家检测方法**：提出GLOBAL和LOCAL两种互补检测策略，分别适用于跨层集中控制和逐层路径控制的模型。
3. **无推理位搜索技术**：基于缓存隐藏状态实现router-centric位搜索，每个专家仅需单次前向传播评估所有候选位。
4. **结构化的DoW攻击框架**：将位翻转攻击从完整性破坏重构为可用性攻击，在6个MoE模型上验证了攻击有效性。
5. **多场景攻击泛化**：扩展至Agent规划模式和End-of-Thinking令牌攻击，证明方法通用性。

## 方法详解
**Step 1：目标相关专家识别**
- 定义路由专家的目标激活偏移：$\tau_{l,i} = \text{target EAF} - \text{non-target EAF}$，高τ值表示专家专门负责目标令牌生成。
- 定义共享专家的目标门控偏移：$\Delta g_{l,i}$，衡量共享专家在目标令牌时的门控幅度变化。

**Step 2：GLOBAL与LOCAL检测**
- GLOBAL：跨所有层排名后，筛选得分超过阈值的专家集合。
- LOCAL：逐层选择top-k专家（k为模型默认路由数），测试每层独立去活化后的长度变化，选择最大增量的层$l^*$。

**Step 3：位翻转攻击实施**
- 缓存目标层隐藏状态H（[M, D]张量），避免重复推理。
- 对每个目标专家的路由参数$v_i$，依次翻转每位，计算位翻转有效性：$\mathrm{BF\text{-}eff}(b) = \frac{\mathrm{EAF}_i^{\mathrm{orig}} - \mathrm{EAF}_i^{(b)}}{\mathrm{EAF}_i^{\mathrm{orig}}}$。
- 选取每个专家得分最高的B位（默认B=3），实施在线攻击。

**物理实现**：通过Rowhammer（DRAM干扰）或电压毛刺在共享硬件中注入位翻转，一旦成功，单点攻击即可长期生效。

## 实验与结果
**评测模型**：Mixtral-8x7B、Phi-3.5-MoE、DeepSeek-V2-Lite、Qwen3-30B-A3B、Qwen3-Coder-Next、GPT-OSS-20B。

**评测数据集**：AGNews、SST-2（分类）；Samsum、SQuAD 2.0（生成）。

**攻击效果**（以P值衡量输出膨胀百分比）：
- **手动去活化**：DeepSeek在AGNews达$3.81 \times 10^4\%$，GPT-OSS GLOBAL达556%，平均输出膨胀5912%。
- **GBFA位翻转攻击**：DeepSeek AGNews达$3.84 \times 10^4\%$，Phi-3.5 GLOBAL达131%，GPT-OSS GLOBAL达556%。
- **Agent场景**：Qwen3-Coder在10个代码沙盒上GLOBAL去活化平均P=+501.5%，全部触发最大步骤限制。
- **EOT攻击**：GPT-OSS GLOBAL在SST-2达$1.27 \times 10^3\%$，Qwen3达224%~271%。

**模型效用**：GBFA在保持语义连贯性的同时实现攻击。Mixtral局部GBFA在AGNews上CA从0.838仅降至0.840（反而略有提升）；Phi-3.5在Samsum上ROUGE-1从0.362提升至0.382。DeepSeek GLOBAL位翻转出现PPL爆炸（最高5.3M），需改进比特搜索策略。

## 相关工作脉络
1. **MoEcho (Ding et al., 2025)**：利用MoE专家执行的时空痕迹进行侧信道推理，关注隐私而非可用性攻击。
2. **SAFEx (Lai et al., 2025)**：揭示MoE安全对齐依赖特定位置专家，定位脆弱点用于安全检查。
3. **传统BFA (Rakin et al., 2019, 2022)**：聚焦模型完整性（精度下降、输出污染），本文首次将BFA导向可用性攻击。
4. **Rowhammer攻击 (Kim et al., 2014; Lin et al., 2025)**：硬件级位翻转技术基础，本文将其应用于MoE路由层。
5. **Breaking the Loop (Yu et al., 2025)**：检测LLM循环输出的缓解方案，与本文攻击形成对抗关系。
6. **BadMoE (Wang et al., 2025)**：通过路由触发器注入后门，与本文关注终止专家形成不同攻击面。

## 局限性与未来方向
1. **威胁模型较强**：假设白盒访问+硬件位翻转能力，对具备强隔离或完整性检查的MLaaS不适用。
2. **缺少端到端硬件验证**：仅完成易感位搜索和模型级攻击演示，未在实际硬件上物理翻转。
3. **理论分析缺失**：未形式化证明路由logits、专家特化与输出膨胀之间的理论关联。
4. **部分模型困惑度爆炸**：DeepSeek GLOBAL攻击出现PPL极高，需引入初始token困惑度约束优化比特搜索。
5. **系统级成本未量化**：硬件部署、攻击成功率、时间成本等工程因素未深入分析。

## 研究启发与可借鉴点
1. **专家-令牌专业化检测方法可迁移**：通过激活差异度量识别功能特异性专家的思路，可扩展至其他攻击面（如安全对齐专家、工具调用专家）的发现。
2. **缓存隐藏状态加速位搜索**：仅一次前向传播获取H，后续位评估零推理开销，可推广至其他权重扰动攻击的效率优化。
3. **从完整性攻击到可用性攻击的思维转换**：为LLM安全研究开辟新视角，避免被"输出仍然有用"的伪装攻击绕过检测。
4. **bias参数作为更优攻击面**：GPT-OSS因攻击bias参数实现近似手动去活化效果，提示设计层参数比权重矩阵更适合精确控制。
5. **与团队方向结合机会**：若团队研究LLM安全加固，可借鉴此方法设计MoE路由完整性验证机制；若研究模型效率，可评估专家去活化对输出控制的影响。

## 关键术语表
**Mixture-of-Experts (MoE)**：通过路由器选择top-k专家子网络处理每Token的LLM架构，实现参数扩展与计算效率的平衡。

**Denial-of-Wallet (DoW)**：通过技术手段迫使模型生成过量输出，从而增加用户API调用成本的可用性攻击类型。

**EOS / EOT**：End-of-Sequence（序列结束）和End-of-Thinking（思维结束）令牌，标记模型应停止生成的特殊标记。

**Bit-Flip Attack (BFA)**：通过翻转模型权重中极少数比特位引发预期行为变化的硬件级攻击。

**Target Activation Shift (τ)**：目标令牌位置专家激活频率与非目标位置频率的差值，用于量化专家专业化程度。

**BF-eff**：位翻转有效性指标，衡量翻转某位后目标专家激活频率的相对下降比例。

**Rowhammer**：通过高频访问DRAM相邻行引发干扰错误，实现无需物理接触的非侵入式位翻转技术。

## 可复现要素
- **数据集**：Stanford Alpaca（专家检测）、AGNews、SST-2、Samsum、SQuAD 2.0（评测）；代码/沙盒任务将在论文发表时公开。
- **代码与权重**：所有评测模型为开源权重；实验代码和配置预计随论文发布。
- **关键超参**：目标令牌EOS；max_new_tokens=1024；GLOBAL选取top-k₉专家（37/16/40/25）；LOCAL按模型默认top-k；每个专家翻转3位；样本数N=100（攻击评估）/N=1000（专家检测）。
