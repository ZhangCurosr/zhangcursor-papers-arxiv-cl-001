---
title: "Polished-but-Unresolved-Identifying-Late-Stage-Pressure-Stat"
source: https://arxiv.org/pdf/2609.00823v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 13:19:12"
field: "长期工具使用智能体的表示干预"
keywords: ["long-horizon agent", "late-stage pressure", "activation steering", "linear probe", "tool-use planning", "PSPR"]
innovations: ["在隐藏空间中线性可分离地识别长程agent晚期压力状态并证实行为因果性", "发现约束清晰度与行动映射双因素可显著缓解晚期压力", "提出PSPR双阈值轻量插件，在线结合激活干预与显式状态组织"]
benchmarks: ["DeepPlanning-Travel", "DeepPlanning-Shop", "TravelPlanner"]
---

# 论文速读：Polished-but-Unresolved-Identifying-Late-Stage-Pressure-Stat

## 一句话总结
本文识别了长程工具使用智能体在即将提交时出现的"晚期压力状态"——模型被偏向于提前提交看似光滑完整的答案，但关键约束仍未满足。作者通过线性探针可识别该状态，并提出 PSPR（Probe-Sensed Pressure Relief）插件，通过激活干预和显式状态组织在线缓解压力，在多组长程规划基准上稳定提升了最终答案质量与约束满足率。

## 研究问题与动机
- **光滑但未解的提交模式**：长程工具使用智能体常在任务后期倾向于"提前闭合"，生成结构完整、语言流利但约束未充分验证的最终答案；用户只能看到提交结果，使这类失败比明显崩溃更难察觉。
- **行为诊断的局限**：已有研究从轨迹层面分析了进度控制失效、证据衰减、早期错误放大等问题，但并未回答：在提交决策点附近，模型是否包含一种独立的、偏向承诺的内部状态。
- **任务时域越长越脆弱**：随着 horizons 增长，智能体对约束跟踪的可靠性下降，且已积累的证据在最终组装时更难被充分利用。
- **可干预性的缺口**：即使识别出问题，也缺少一种轻量、在线、不依赖额外训练的方法在提交前纠正压力状态。

## 核心贡献（创新点）
- **在隐空间中线性可分离地识别晚期压力状态**：通过对比 PresC（压力驱动提交）、HealC（健康提交）与 ProdC（持续 productive 执行）三类边界的第一 token 隐状态，训练线性探针达到 AUROC=0.916；与已有"事后行为标签"的本质区别在于，本文证明了该状态是可读取、可干预的表征实体。
- **揭示晚期压力的两个缓解因素**：通过受控上下文操纵发现，约束清晰度（constraint clarity）与行动映射（action mapping）同时引入时，压力从 0.62 降至 0.13，repair 率升至 0.85；既有工作多归因于证据衰减或预算控制，本文从表示层面给出更精细的条件解释。
- **提出 PSPR 轻量在线干预框架**：基于探针分数采用双阈值调度——中压时施加压力缓解方向（activation steering），高压时触发显式状态组织（prompted constraint reorganization），不需要额外微调；与单纯 prompt engineering 的区别在于干预信号来自模型内部表示而非外部指令。
- **证实干预方向的行为因果性**：减去 v_PresC-HealC 使 CS 从 0.21 提升至 0.25，下一边界压力从 0.67 降至 0.53；沿反方向则恶化；这超越了相关性，提供了方向性因果证据。

## 方法详解
- **探针构造**：在每轮 action boundary（prefill 结束、下一 token 生成前）取第一 token 的隐藏状态 h_T；正类 PresC（S_sat ≤ 0.3 且 DPS ≥ 4），负类由 HealC（S_sat ≥ 0.85, DPS ≥ 4）与 ProdC（后续两步仍指向未满足约束）组成；用线性分类器在层 ℓ=26（Qwen3-14B）训练。
- **激活方向构造**：按式 v_{C^+-C^-}=μ(C^+)-μ(C^-) 计算 contrastive direction；进一步 residualize v_PresC-ProdC 去除 generic commit 分量：v^{⊥commit}=v-Proj_{v_commit}(v)，其中 v_commit=μ(HealC)-μ(ProdC)。
- **干预机制**：在每个 boundary 评估 s_t=Probe(h_{t,1}^{(ℓ)})；按阈值 a=0.4、b=0.65 分三档：s_t<a 不干预；a≤s_t<b 施加 v_relief=μ_LP-μ_HP（HP 为原始 PresC 节点激活，LP 为经过 clarification+action map 增强后 productive 节点的激活）；s_t≥b 触发显式组织，让模型总结当前约束状态并将不确定项映射为下一步行动。
- **显式状态组织**：高压分支要求模型输出（1）当前满足/不满足约束列表（2）不确定或未支持项（3）下一步可能行动；结果拼入 prefill 后继续生成，每条轨迹最多进入该分支两次。

## 实验与结果
- **基准与设置**：主实验 DeepPlanning-Travel（120 题，含 location/budget/user 硬约束）；扩展实验 DeepPlanning-Shop、TravelPlanner；基线含 CoT、ReAct、Reflexion_1；三层回测 Qwen3-14B/32B、OLMo-3.1-32B-Instruct；temperature=0.6，pass@3。
- **主表提升**：Qwen3-32B+CoT 的 CP 从 25.2%→28.1%；Qwen3-32B+ReAct 的 CP 从 29.3%→33.2%；Qwen3-14B+ReAct 的 CP 从 26.1%→29.1%；CS、PS、CP 在所有模型×方法组合上均稳定提升。
- **扩展基准**：DeepPlanning-Shop 上 ReAct+PSPR Match=68.0（+2.1）、Acc=23.0（+4.0）；TravelPlanner 上 DR、CS、HC 均提升。
- **干预方向验证**：Table 2 显示 −v_PresC-HealC 使 PresC-risk 节点 Valid p_next 从 0.67→0.53，Continuation Rate 0.28→0.38；同向干预则反向恶化。
- **消融**：Relief direction 与 Explicit organization 均有贡献；Random/Periodic trigger 不如 PSPR，说明压力信号调度是关键；中文版本 DeepPlanning-Travel 同样提升。
- **开销**：平均轮次 4.62→7.94，工具调用 7.57→8.95，Δp_next 分别 −0.31（relief）与 −0.57（organization）。

## 相关工作脉络
- **长程工具失败诊断**（Garikaparthi et al., 2026; Wang et al., 2026a; He et al., 2026）：聚焦轨迹级进度/资源/工具组织失败；本文与之定位差异在于聚焦"光滑提交"这一具体子类并从内部表示角度刻画。
- **晚期约束失败**（Ko et al., 2026 illusory completion; Wang et al., 2026b early-error cascade; Yu et al., 2026 evidence decay; Fang et al., 2026 inference-time budget）：均为行为/过程层面解释；本文进一步证明存在可读取的内部压力表征。
- **激活工程**（Rimsky et al., 2024 CAA; Højer et al., 2025 PCA-CAA; Stolfo et al., 2025; Wang et al., 2025 adaptive steering）：通用 LLM 干预框架；本文的关键区别是将其专门化到"长程工具使用晚期压力"这一细分现象，并设计双阈值调度与双分支干预。
- **概念擦除/线性解耦**（Belrose et al., 2023 LEACE; Todd et al., 2024 function vectors）：支撑本文 residualize commit 分量的技术背景。

## 局限性与未来方向
- **设定局限**：主要在文本化、结构化、可验证的长程工具使用场景下验证，难以直接推广到开放动态或多模态 agent 环境。
- **因果解释边界**：作者明确称晚期压力并非过早闭合的唯一或充分原因，早期规划错误、证据缺失、工具错误、benchmark 激励等仍起作用。
- **适用性约束**：PSPR 需要访问 hidden states，仅适用于 open-weight 或可检查模型；训练探针与方向需额外离线 rollout 标注，带来一定成本。
- **阈值固定**：a=0.4, b=0.65 经少量预试验确定并在多设置上复用，可能存在 per-domain 微调空间。

## 研究启发与可借鉴点
- **"表示诊断+轻量干预"范式**：先训练线性探针建立内部状态可观测性，再以 activation addition 实施无训练干预；该方法可迁移至其他 agent 失败模式（如过早收敛、反复循环、幻觉漂移）。
- **双阈值分层干预设计**：按压力强度切换"隐式激活偏移"与"显式 prompt 重组"，兼顾效率与力度；可作为一般 agent 在线监控架构的模板。
- **负类对照设计**：用 HealC（光滑且满足）与 ProdC（未满足但持续工作）共同排除"提交形式"和"任务难度"两类混淆，值得在其它探针研究中复用。
- **约束清晰度+行动映射双因素**：本文实证的两类 context augmentation 结构简单、实现成本低，可直接嵌入现有 agent loop 作为"压力检查点"模块。
- **团队结合点**：若团队研究涉及长程推理/多步规划/工具调用，可将 PSPR 作为 post-hoc plugin 接入，在不改动主模型的前提下试水；也可将其探针思路迁移到代码生成、数学推理等"结尾提交敏感"场景。

## 关键术语表
- **Late-stage pressure（晚期压力）**：智能体在长程任务接近提交时产生的、偏向提前闭合而非继续验证的内部状态。
- **PresC（Pressure-driven Commit）**：低约束满足度（S_sat≤0.3）但高提交光滑度（DPS≥4）的决策边界，作为压力正类样本。
- **HealC / ProdC**：健康提交（S_sat≥0.85, DPS≥4）与持续 productive 执行（后续两步仍修复未满足约束）两类负样本，用于排除形式/难度混淆。
- **Delivery Polish Score (DPS)**：基于 Gemini 3.1 Pro Preview 对最终答案"完整可用感"的 1–5 分评估，仅看文本形式不核查事实。
- **Constraint clarity（约束清晰度）**：将当前已满足/未满足的硬约束显式列出并拼入上下文的增强手段。
- **Action mapping（行动映射）**：将未满足约束映射为具体下一步工具调用或行动的结构化提示。
- **Activation steering / Contrastive Activation Addition (CAA)**：在推理时沿正负样本平均隐状态之差的方向对前 K 个 token 的隐藏状态做加减干预。
- **PSPR（Probe-Sensed Pressure Relief）**：论文提出的轻量插件，按探针分数双阈值调度压力缓解方向或显式状态组织。

## 可复现要素
- **数据集**：DeepPlanning-Travel（120 tasks，公开许可）、DeepPlanning-Shop（100 instances）、TravelPlanner（公开）；论文采用两折 held-out 划分，无官方训练集。
- **代码/权重**：论文未明确声明代码开源仓库；基线模型 Qwen3-14B/32B、OLMo-3.1-32B-Instruct 为 open-weight。
- **关键超参**：干预层 ℓ=26（Qwen3-14B）/46（Qwen3-32B）；K=5（干预窗口，部分实验 K=10）；|α|=1；阈值 a=0.4、b=0.65；temperature=0.6；pass@3；probe 训练集 ~545 例、测试集 ~160 例。
