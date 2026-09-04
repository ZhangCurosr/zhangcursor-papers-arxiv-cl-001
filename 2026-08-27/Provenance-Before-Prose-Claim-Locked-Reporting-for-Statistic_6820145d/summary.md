---
title: "Provenance-Before-Prose-Claim-Locked-Reporting-for-Statistic"
source: https://arxiv.org/pdf/2608.25336v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 01:48:19"
field: "可信大模型生成与统计报告可靠性"
keywords: ["statistical reporting", "claim-locked reporting", "LLM faithfulness", "cross-run reproducibility", "evidence-bound claims", "deterministic rendering", "risk auditor"]
innovations: ["提出 claim-locked reporting 协议，将证据源/数值/方向/强度在 prose 生成前锁定到 claim ledger", "揭示 hybrid template 的 slot 级控制不足以稳定证据承载内容", "建立可复现性/治理/正确性三轴评估并校准 FActScore 类指标的局限性"]
benchmarks: ["Evidence Inference 2.0", "HCP S1200 fMRI cohort", "In-house obesity fMRI cohort"]
---

# 论文速读：Provenance-Before-Prose-Claim-Locked-Reporting-for-Statistic

## 一句话总结
论文提出"claim-locked reporting"协议，通过将结构化统计证据（来源、数值、方向、语言强度上限）在自然语言生成前锁定到 claim ledger 中，使 LLM 仅负责连接性 prose，从而解决大模型生成统计报告时存在的数值漂移、方向反转和断言过度等统计声明失真问题。

## 研究问题与动机
1. **统计报告失真现象**：现有 LLM 生成的 fMRI/临床试验报告常在多次运行中出现数值漂移、方向颠倒或阈值对比被重新表述为类别效应的问题，这类"统计声明失真"与事实性幻觉不同。
2. **现有控制粒度不足**：text-level 控制（提示/检索/结构化输出/事后验证）只约束输出形式或可访问证据，但声明选择、数值实现、方向框架和修辞强度仍由 LLM 决定；slot-level 混合模板仅实现确定性渲染，LLM 仍能选择填入 slot 的证据内容。
3. **缺乏声明级控制**：现有方法没有"在 prose 生成之前将可报告统计声明绑定到证据源"的机制，导致生成过程本质上是采样而非确定性映射。

## 核心贡献（创新点）
1. **将统计声明失真重新定义为可靠性控制问题**：提出以跨运行可复现性作为应力测试，检验证据承载内容是否在 prose 生成前被固定，区别于事后检测型幻觉评估。
2. **提出 claim-locked reporting 三级控制协议**：包含 claim builder → risk auditor → policy controller → deterministic renderer → LLM writer 的完整流水线，将控制单元从文本/槽位迁移至证据绑定声明。
3. **在 fMRI 和功能连接报告和 RCT 报告上验证显著提升**：相比 hybrid template，claim-locked 在 fMRI 可复现性上提升 +37.4 点（61.1%→98.5%），在 RCT 提升 +20.5 点（79.5%→100.0%），同时实现最低的 token 消耗和生成延迟。
4. **揭示 FActScore 等词法支持度指标的局限性**：证明现有忠实度指标会因词法匹配/采样一致性而给出与统计声明保留相悖的排序，提出需建立针对统计方向的直接评估轴。

## 方法详解
**整体架构**：证据记录（Statistical Evidence Record）→ claim ledger → 确定性渲染 → LLM 连接 prose。

1. **Statistical Evidence Record**：由领域分析流水线输出的结构化计算结果（如 FC 的 FDR 边缘计数、BMI 调整后的折叠率；RCT 的 intervention/comparator/outcome/direction span）。
2. **Claim Builder**：将证据记录转换为 typed claim 集合，每条 claim 包含 claim type、canonical text、evidence pointers、numerical fields、effect direction、literature-support level、allowed language strength。
3. **Risk Auditor**：只读层，按五条证据承载属性（数值支持、方向保留、实体/来源支持、推断范围、解释强度）检查风险；FC 触发器覆盖边缘/枢纽/ROI/协变量/脑行为关联；RCT 覆盖 Evidence Inference 字段。
4. **Policy Controller**：对标记 claim 执行单调强度降级（只能降低不允许提高），并将 forbidden language 列表 + 每类 NL 指令传给 LLM。
5. **Deterministic Renderer + LLM Writer**：数值/表格/方向标签完全由 ledger 渲染，LLM 仅写限定范围内的连接段落；输出不完整则 generation fails。

**关键公式（数值匹配容差）**：绝对容差 $10^{-3}$ 或相对容差 5%，用于 Reproducibility 评估中的数值匹配；跨种子 Jaccard 重叠 $J$ 作为主指标。

**组件分析**：添加 deterministic renderer 使 fMRI 复现率从 85.0% 升至 98.0%，激活 risk auditor + policy controller 后 Strong 从 1.40 降至 0.75。

## 实验与结果
- **数据集**：① fMRI：机构肥胖队列 n=428 + 公开 HCP S1200（n=712），Brainnetome-246 分箱；② RCT：Evidence Inference 2.0，112 增加 + 88 减少样本（共 200 条去重）。
- **评估基线**：Free-form / Prompt-only / Structured / Retrieval / Post-hoc verifier / Hybrid template / Claim-locked，两个 writer 提供商（Moonshot kimi-k2.6、DeepSeek deepseek-v4-pro）。
- **主要结果**：
  - fMRI 可复现性：Claim-locked 98.5% vs Hybrid 61.1%，提升 **+37.4 点**（95% CI [+15.1, +59.7]）。
  - RCT 可复现性：Claim-locked 100.0% vs Hybrid 79.5%，提升 **+20.5 点**（95% CI [+17.8, +23.1]）。
  - RCT 方向审计：Claim-locked 方向保留 116/120，反转 0/120；inversion rate = **0.0%**，基线范围 5.6%–14.3%。
  - fMRI 人力审计：Claim-locked 强语言违规均值 0.60，不支持内容违规均值 0.53，均最低。
  - 资源效率（DeepSeek writer）：Claim-locked input 12,482 tokens / output 4,209 tokens / latency 77.3s，为最低。
  - RCT 数值校准：Claim-locked **0.0%** harmful fabrication，Hybrid 2.5%，Free-form 5.9%。

## 相关工作脉络
1. **Faithful generation (FActScore / SelfCheckGPT)**：评估已生成文本的词法支持或采样一致性；本文认为它们不测量统计声明保留，会错过方向反转和强度过度。
2. **Constrained generation (RAG / structured output / post-hoc verifier)**：约束证据访问或输出结构但不锁定声明选择；本文定位为在 prose 之前锁定声明的第三级控制。
3. **Classical NLG content planning vs realization (Reiter & Dale; Gatt & Krahmer)**：强调内容与形式分离但选择不保证统计正确性；本文继承此思想但进一步在证据层面锁定数值/方向。
4. **Neural data-to-text with content selection (Lebret et al.; Wiseman et al.; Puduppully et al.)**：端到端联合建模内容选择；本文指出此类方法仍允许 LLM 在推理阶段改变方向/强度。
5. **Neuroimaging reproducibility (Marek et al.; Botvinik-Nezer et al.; Power et al.; Kriegeskorte et al.)**：证明 fMRI 分析可变异性；本文通过声明级锁定避免将可变结果错误地表述为确定发现。

## 局限性与未来方向
1. **上游错误传播**：claim-locked 只控制如何表述证据，不对统计有效性/模型设定/协变量选择负责，上游错误会直接传递到报告。
2. **风险分类库非穷尽**：当前 taxonomy 是固定且 benchmark-specific 的，不能覆盖所有统计报告风险。
3. **可复现性不等于正确性**：报告可能"稳定地错误"，需结合治理/正确性轴解释。
4. **RCT 未覆盖 null 方向**：当前仅处理增加/减少两类，中性结论需额外设计 ledger state。
5. **文献支持标签由 LLM 辅助分配**：作者建议未来替换为专家策展标签或进行验证。
6. **机构 fMRI 数据不可分发**：仅能发布 HCP 公开数据集和相关代码/提示。

## 研究启发与可借鉴点
1. **将"证据绑定声明"作为控制单元**：对于任何涉及数值/方向统计内容的生成任务（如临床摘要、科学报告、金融新闻），可借鉴先构建 claim ledger 再渲染的范式。
2. **单调强度降级策略**：风险标注只允许语言强度向下走，不向上升级，为"防止过度自信"提供了简单而可操作的接口。
3. **双向评估轴设计**：可复现性（跨种子 Jaccard）+ 治理（Strong/Hedge）+ 正确性（数值/方向审计）分离报告，避免单一指标掩盖不同失效模式。
4. **对现有忠实度指标的批判性使用**：FActScore/SelfCheckGPT 会因词法匹配或采样一致性给出误导性排序；建议结合手动/规则校准来识别 harmful fabrication。
5. **可复现性作为协议测试而非最终真理**：用跨运行一致性检验控制是否生效，但需配合人工审计和校准才能构成完整可靠性论证。

## 关键术语表
**Claim-locked reporting**：一种在 prose 生成前将声明的证据源、数值、方向和语言强度上限锁定到 ledger 的协议，LLM 仅生成连接文本。
**Statistical claim distortion**：统计报告中出现的数值漂移、方向反转或将阈值对比重写为类别效应等失真错误。
**Risk auditor**：只读检查层，依据领域风险类别对照证据字段标记声明，不修改 ledger。
**Policy controller**：将 auditor 标记转化为渲染/写作约束，执行单调强度降级并生成 forbidden language 列表。
**Cross-run reproducibility**：以跨种子 Jaccard 重叠衡量相同证据在不同运行下报告可见数值的稳定性。
**Harmful fabrication**：报告中出现的支持集中不存在、且构成错误统计结论的数值（区别于 benign metadata mismatch）。
**Hybrid template**：将 slot 渲染确定性化的混合方案，但仍允许 LLM 选择填入 slot 的内容。
**Monotone strength downgrade**：策略规则，风险标记声明的语言强度只能降级不允许升级。

## 可复现要素
- **数据集**：机构肥胖 fMRI 队列不可再分发；HCP S1200 公开；Evidence Inference 2.0 公开。
- **代码**：开源，链接 https://github.com/XiuFan719/Claim-Locked-Reporting（论文明确声明代码可用）。
- **关键超参**：数值匹配容差绝对 10^-3 或相对 5%；fMRI 作者使用 g=0（BMI<25）与 g=1（BMI≥30）及 WHO 阈值；HCP 排除超重层。
- **Writer 提供商与温度**：Moonshot kimi-k2.6（temperature 1.0）和 DeepSeek deepseek-v4-pro（temperature 0.7）。
- **评估重复**：fMRI 每方法 20 个 cell（两队列×两 provider×五 seed）；RCT 每方法 800 cells（112+88 样本×两 seed×两 provider）。
