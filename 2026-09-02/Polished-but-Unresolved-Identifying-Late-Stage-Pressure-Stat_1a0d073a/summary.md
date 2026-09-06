---
title: "Polished-but-Unresolved-Identifying-Late-Stage-Pressure-Stat"
source: https://arxiv.org/pdf/2609.00823v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 13:18:52"
field: "Agent可靠性与可解释性"
keywords: ["long-horizon agents", "tool-use", "activation engineering", "late-stage pressure", "probe", "PSPR", "constraint satisfaction"]
innovations: ["识别并量化长周期Agent晚期压力状态，证明其在隐藏空间中线性可分", "提出PSPR轻量插件，通过探针监控+分级干预缓解晚期过早闭合", "揭示约束清晰度与行动映射是缓解晚期压力的关键因素"]
benchmarks: ["DeepPlanning-Travel", "DeepPlanning-Shop", "TravelPlanner"]
---

# 论文速读：Polished-but-Unresolved-Identifying-Late-Stage-Pressure-Stat

## 一句话总结
本文识别出长周期工具使用智能体中存在一种"晚期压力状态"——智能体倾向于在约束未满足时提交看似完整的回答；通过线性探针检测该状态并结合激活干预，提出轻量插件PSPR，在多基准上持续提升Agent质量。

## 研究问题与动机
1. **核心现象**：长周期工具使用Agent常在任务后期产生"表面完整但关键约束未解决"的提交（polished but unresolved），用户仅能看到最终答案而难以察觉此类失败。
2. **现有研究空白**：已有工作从行为层面分析约束未满足的原因（如虚假完成、早期错误积累、证据整合衰减），但未回答是否存在一个可识别的、驱动晚期过早闭合的内部状态。
3. **方法不足**：当前Agent评估多关注最终输出质量，缺乏对"即将提交但约束未满足"这一决策边界的隐式状态分析与干预。
4. **动机**：若晚期压力状态在隐藏空间中可线性分离且可通过激活干预改变行为，则可开发轻量在线监控与缓解机制。

## 核心贡献（创新点）
1. **识别晚期压力状态**：证明在长周期工具使用中，"表面完整但约束未满足"的提交对应隐藏空间中可线性分离的内部状态（AUROC=0.916），而非单纯的行为标签。
2. **揭示压力缓解因素**：通过受控上下文操作发现，约束清晰度（constraint clarity）和行动映射（action mapping）是缓解晚期压力的两个关键因素。
3. **提出PSPR插件**：设计轻量在线干预方法，通过探针持续监控压力信号，在中等压力时施加激活缓解方向，在高压力风险时触发显式状态组织，无需额外训练。
4. **验证普适性**：在DeepPlanning-Travel、DeepPlanning-Shop、TravelPlanner等多个基准上，PSPR对CoT、ReAct、Reflexion等基线方法均产生一致提升。

## 方法详解
1. **探针构建**：在动作边界（prefill结束后、下一个action开始前）提取首个token隐藏状态，训练线性探针区分三类状态：
   - PresC（压力驱动提交）：约束满足度$S_{sat} \leq 0.3$且交付完善度DPS $\geq 4$
   - HealC（健康提交）：$S_{sat} \geq 0.85$且DPS $\geq 4$
   - ProdC（生产性继续）：后续动作继续处理未满足约束
   
2. **激活方向构造**：使用对比激活加法（CAA）构造干预方向：
   - $v_{PresC-HealC} = \mu(PresC) - \mu(HealC)$：隔离压力成分
   - $v_{PresC-ProdC}^{\perp commit}$：去除泛化提交倾向后的残余压力方向
   
3. **PSPR两阶段干预**：
   - 中等压力（$a \leq s_t < b$）：施加压力缓解方向$v_{relief} = \mu_{LP} - \mu_{HP}$，其中HP来自PresC节点，LP来自带结构化上下文的ProdC节点
   - 高压力（$s_t \geq b$）：触发显式状态组织，要求模型总结约束状态并规划下一步行动
   
4. **超参数**：阈值$a=0.4, b=0.65$，干预窗口$K=10$ tokens，干预幅度$\alpha=1$，探针层选用模型中间层（Qwen3-14B用layer 26）。

## 实验与结果
1. **主要基准DeepPlanning-Travel**（120个任务）：
   - Qwen3-32B：CoT的CP从25.2%提升至28.1%，ReAct从29.3%提升至33.2%
   - Qwen3-14B：ReAct的CS从41.5%提升至45.6%（95% CI: [0.4, 8.0]）
   - OLMo-3.1-32B：ReAct的CP从27.0%提升至28.9%
   
2. **泛化基准**：
   - DeepPlanning-Shop：Match从65.9%提升至68.0%，Acc从19.0%提升至23.0%
   - TravelPlanner：DR、CS、HC各项指标均有提升
   
3. **消融实验**：
   - 显式状态组织贡献更大（CP提升21.4% vs 19.9%）
   - 非探针触发控制（随机触发/周期性触发）效果均低于PSPR，表明干预时机的重要性
   - 中文版本DeepPlanning-Travel同样有效
   
4. **执行成本**：平均回合数从4.62增至7.94，工具调用从7.57增至8.95，压力分数显著下降（缓解方向-0.31，显式组织-0.57）。

## 相关工作脉络
1. **长周期工具使用失败研究**：Ma et al. (2024) AgentBoard、Garikaparthi et al. (2026) Researchgym、Wang et al. (2026a) TRAJECT-bench诊断了轨迹级失败；本文聚焦"表面完整但约束未满足"这一特定失败模式。
2. **晚期约束失败**：Ko et al. (2026) 虚假完成、Wang et al. (2026b) 早期错误累积、Yu et al. (2026) 证据整合衰减——这些工作从行为层面分析原因，本文从表示层面识别内部状态。
3. **激活工程**：Rimsky et al. (2024) CAA、Stolfo et al. (2025) 指令遵循改进、Højer et al. (2025) PCA优化——本文将其应用于长周期Agent的晚期压力缓解。
4. **推理时预算控制**：Fang et al. (2026) 研究搜索与承诺的边界；本文进一步识别导致错误承诺的内部状态并可直接干预。

## 局限性与未来方向
1. **适用场景限制**：研究主要在文本结构化、可验证的长周期工具使用环境中进行，扩展到开放动态或多模态Agent环境是重要方向。
2. **非因果解释**：晚期压力状态并非过早闭合的唯一或充分原因，早期规划错误、证据缺失、上下文丢失、工具错误等也可能导致失败。
3. **隐状态访问需求**：PSPR需要访问隐藏状态进行探针监控和激活干预，限制了其对闭源模型的应用。
4. **离线标注成本**：构建探针和激活方向需要额外的离线数据标注，引入一定成本。

## 研究启发与可借鉴点
1. **探针设计策略**：通过构造互补负类（HealC控制提交形式、ProdC控制任务难度）避免探针学习捷径，值得在类似状态识别任务中借鉴。
2. **残余化方向构造**：去除泛化倾向后的压力方向更精准地捕获目标行为，可推广至其他需要隔离特定心理状态的研究。
3. **双阈值分级干预**：根据压力强度选择不同干预策略（轻量激活 vs 显式组织），为实时Agent控制提供可扩展框架。
4. **注意力验证**：通过计算追加上下文块的注意力集中比（1.6×）验证模型真正利用了诊断信息，增强了干预有效性的说服力。

## 关键术语表
**Late-stage pressure**：智能体在任务后期倾向于过早闭合、提交看似完整但约束未满足答案的内部状态。
**Pressure-driven commit (PresC)**：约束满足度低但交付完善度高的一类动作边界，代表晚期压力状态。
**Healthy commit (HealC)**：约束满足度高且交付完善的动作边界，作为压力状态的对照正例。
**Productive continue (ProdC)**：继续处理未满足约束的动作边界，防止探针混淆压力与任务难度。
**Activation intervention**：在推理时向特定层的隐藏状态添加/减去方向向量以改变模型行为的技术。
**Constraint clarity**：使已验证和未解决的硬约束显式化的上下文增强方式，可缓解晚期压力。
**Action mapping**：将未解决的约束映射到具体下一步工具调用的上下文增强方式。
**PSPR**：Probe-Sensed Pressure Relief，基于探针感知压力的轻量干预插件。

## 可复现要素
- 数据集：DeepPlanning-Travel（公开）、DeepPlanning-Shop（公开）、TravelPlanner（公开）
- 代码/权重：论文未明确提及代码开源状态；使用Qwen3-14B/32B、OLMo-3.1-32B-Instruct开源模型
- 关键超参：阈值$a=0.4, b=0.65$，干预窗口$K=10$，干预幅度$\alpha=1$，探针层layer 26（Qwen3-14B）/layer 46（Qwen3-32B）
- 评估协议：pass@3，两折留一评估（因无官方训练集）
