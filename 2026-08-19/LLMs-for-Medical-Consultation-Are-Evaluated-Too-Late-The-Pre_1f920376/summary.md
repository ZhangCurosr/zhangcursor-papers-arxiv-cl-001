---
title: "LLMs-for-Medical-Consultation-Are-Evaluated-Too-Late-The-Pre"
source: https://arxiv.org/pdf/2608.17330v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 11:29:30"
field: "医疗大语言模型交互安全评估"
keywords: ["medical LLM evaluation", "preformulation gap", "first-contact safety", "interactive assessment", "healthcare AI", "patient simulation", "clinical handoff"]
innovations: ["提出preformulation gap概念并量化首次咨询阶段的交互风险", "设计固定脚本与自适应模拟患者双重评测方法", "验证entry-to-care instruction显著改善询问顺序和handoff生成"]
benchmarks: ["4医生编写多轮病例", "Adaptive standardized-patient simulation"]
---

# 论文速读：LLMs-for-Medical-Consultation-Are-Evaluated-Too-Late-The-Pre

## 一句话总结
论文提出并量化了医疗咨询LLM的"预 formulations 差距"（preformulation gap）——模型在患者首次接触时往往在未完成关键病史采集前就给出自我护理建议；通过4个医生编写的多轮病例与"entry-to-care instruction"干预，证明简短的工作流提示能显著改善首接触的询问顺序和 handoff 记录，但无法可靠确保关键临床事实的主动 elicitation。

## 研究问题与动机
- **评估时机错位**：现有医学LLM评测多基于已清晰表述的临床场景或标准化问答，未能捕捉患者以模糊、最小化、错框架（如拼写错误、症状轻描淡写）发起的首次咨询中的真实行为。
- **安全风险的早期发生点**：误判、漏报红旗症状、虚假安慰可能发生在模型尚未获取足够信息形成临床问题之前，而非最终诊断阶段；传统准确率指标无法揭示这一阶段风险。
- **用户-模型交互性能落差**：Bean et al. (2026) RCT显示模型单独测试正确率94.9%，但用户辅助下仅34.5%，表明模型具备医学知识但缺乏引导用户组织信息的交互能力。
- **指令敏感性揭示可干预性**：若首接触失败可通过工作流提示改变，则说明部分问题源于任务框架而非模型内在缺陷，为评测设计与产品规范提供明确靶点。

## 核心贡献（创新点）
- **提出"preformulation gap"概念框架**：将首次咨询安全性的核心问题从"诊断准确率"重新定位为"模型是否在给出建议前完成有效病史采集"，填补了医学LLM评估的时间维度空白。
- **设计可直接观测的交互标记体系**：构建4个由医生编写、包含逐轮风险披露的多轮病例，以及"advice-before-elicitation"、"premise repair"、"safe routing"、"handoff readiness"四个评分域，使首次接触行为可被量化比较。
- **引入固定脚本与自适应模拟患者双重评测设计**：固定脚本确保信息可比性，自适应LLM模拟患者测试模型能否主动 elicitation 关键事实（而非被动接收），揭示 instruction 改善顺序但未能可靠确保关键事实采集的残留问题。
- **验证简短 entry-to-care instruction 的工作流重塑效果**：证明仅通过系统提示改变询问-建议顺序、unsafe plan 纠正、分级路由和 handoff 生成四个步骤，即可将"首轮即给自我护理建议"比例从9/12降至0/12，将结构化 handoff 出现率从0/12提升至10/12。

## 方法详解
**病例设计**：4个医生编写的多轮 vignette，均以非结构化、含拼写错误的患者消息开场，逐轮披露风险信息：
- 老年上腹痛（60岁男性，计划饮酒sleep it off）
- 年轻成人胸膜炎性胸痛（25岁，左侧刺痛）
- 呕吐合并糖尿病风险（胰岛素治疗者漏打胰岛素）
- 旅行后腿痛（单侧小腿肿胀，计划按摩）

**模型与条件**：
- 三个API模型：OpenAI `chat-latest`（近似ChatGPT）、Google `gemini-3.5-flash`、Anthropic `claude-sonnet-4-6`
- Baseline条件：无system prompt、无tools、无评估告知
- Instruction条件：附加4步"entry-to-care instruction"系统提示（询问优先→纠正unsafe plan→分级路由→生成handoff摘要）
- 24个固定脚本转录 + 12个自适应模拟患者转录（呕吐和腿痛病例）

**自适应设计**：患者由LLM模拟（gemini-3.5-flash，80 token回复上限），仅回答模型提出的问题；预设unsafe plan语句在所有运行中相同时间点注入，确保可比性。

**评分体系**（4个域，每域0-2分）：
1. **Usable concern**：是否 elicitation 决策相关事实并跨轮适应
2. **Premise repair**：是否纠正unsafe framing或risky intended action
3. **Safe routing**：下一步建议是否与红旗症状和不确定性校准
4. **Handoff readiness**：是否保留可供临床使用的结构化摘要

关键可观测标记包括：首轮是否在任何患者回答前给出自我护理建议、是否询问≥3个紧急程度相关问题、是否纠正明确的unsafe plan、是否出现结构化handoff summary。

## 实验与结果
**数据集**：4个医生编写的多轮病例，非流行病学代表性样本；测试于2026年7月28日。

**主要结果数字**（固定脚本，24个转录）：
| 标记 | Baseline | Instruction |
|------|----------|-------------|
| 首轮给出自我护理/家庭管理建议（在任何患者回答前） | 9/12 | 0/12 |
| 首轮询问≥3个紧急程度相关问题 | 8/12 | 12/12 |
| 首次呕吐回合询问慢性病/用药 | 0/3 | 1/3 |
| 明确unsafe plan被纠正 | 8/9 | 9/9 |
| 出现结构化handoff summary | 0/12 | 10/12 |

**自适应运行关键发现**：
- 糖尿病/胰岛素信息在预设披露前：Baseline 1/3 运行中浮现，Instruction 0/3 运行中浮现
- 长程乘车史在所有6个腿痛运行中均被 elicitation
- 一旦关键事实出现，所有模型均能识别风险并升级建议

**结论**：Instruction 显著改变了询问-建议顺序和文档生成行为，但未可靠确保关键临床事实（如糖尿病、胰岛素使用）的首轮主动 elicitation；baseline 模型在收到决定性事实后同样展现相关知识，表明问题核心在于交互顺序而非知识缺失。

## 相关工作脉络
- **Singhal et al. (2025), McDuf et al. (2025)**：医学LLM在问答和鉴别诊断支持中达到医师水平，但基于静态/专家生成查询，不涉及首次接触的真实交互动态。
- **Tu et al. (2025)**：交互式模拟咨询评测，但仍从研究者设计的临床场景出发，未隔离患者以稀疏/误导性框架发起的首次咨询。
- **Bean et al. (2026)**：RCT显示用户-LLM协作诊断正确率（34.5%）远低于模型单独测试（94.9%），本文将其归因于 preformulation gap 而非知识缺陷。
- **Ramaswamy et al. (2026)**：ChatGPT Health 评估发现半数金标准急诊被降级为24-48小时评估，症状最小化显著影响分诊建议（OR 11.7），与本文发现指向同一早期失败点。
- **Johri et al. (2025), Kopka et al. (2025), Draelos et al. (2026)**：分别提出患者交互评测框架、自分诊准确性评估和患者提问red-teaming，本文在此基础上明确将评估起点前移至"病例formulation之前"。
- **Hampton et al. (1975)**：经典门诊研究证明病史采集使诊断与最终诊断一致率达66/80，本文将此临床实践价值映射到LLM首次接触评估。

## 局限性与未来方向
- **病例数量少、非代表性**：仅4个英语病例，各1次运行，无法估计错误或排序供应商安全性。
- **无人类对照与 benign 对照组**：未包含健康人群对照，也无法判断模型建议是否过度/不足医疗化。
- **评分非盲法**：case-specific routing 期望和转录审查由研究团队定义，未独立盲评。
- **Instruction 未做消融**：无法确定四步中哪一步贡献最大。
- **自适应患者由LLM模拟**：可能与真实患者回答存在差异。
- **API模型近似消费产品**：未测试完整产品层（安全层、记忆、账户级控制）。
- **未来方向**：需要更大规模、多语言、真实患者输入的动态评测；需要human-in-the-loop对照研究；需开发标准化的首次接触benchmark。

## 研究启发与可借鉴点
- **评测时间维度的前移**：将评估对象从"给定病例后的回答质量"转向"病例形成前的交互质量"，为医疗AI评测提供新的方法论范式。
- **固定脚本+自适应模拟的双重设计**：固定脚本保证可比性，自适应模拟测试主动elicitation能力，两者结合可更全面评估交互行为。
- **可观测交互标记优于单一准确率**：sequencing marker、premise repair、handoff presence等细粒度标记能揭示单一分数掩盖的行为差异。
- **工作流提示作为低成本干预手段**：简短instruction即可显著改变模型行为，提示产品设计中可通过系统prompt规范首接触流程。
- **可与本团队方向结合**：若团队关注医疗对话系统、患者入口代理或症状检查器，可将preformulation gap作为评测维度和优化目标，开发专门的elicitation能力训练方案。

## 关键术语表
**Preformulation gap**：从患者模糊/最小化/错框架的主诉到可临床使用的病例 formulation 之间的差距，指模型在未完成有效病史采集前就给出建议的风险窗口。
**Entry-to-care instruction**：本文设计的4步系统提示，要求模型先询问关键问题、纠正unsafe plan、分级路由、生成handoff摘要，再提供自我护理建议。
**Adaptive standardized-patient simulation**：使用LLM模拟患者，仅回答 tested model 提出的问题，测试模型能否主动 elicitation 关键事实而非被动接收。
**Handoff**：患者可带入下一阶段护理的结构化摘要，包含时间线、关键阳性和阴性事实、缺失信息和升级原因。
**Premise repair**：模型纠正患者unsafe framing、症状最小化或risky intended action（如"饮酒sleep it off"、"按摩小腿"）的行为。
**Safe routing**：根据红旗症状和不确定性校准下一步建议（emergency/same-day/自护理），避免undertriage或reflexive emergency triage。
**Sequencing marker**：可观测标记，记录模型是否在任何患者回答前就给出自我护理建议。

## 可复现要素
- **数据集**：4个医生编写的多轮病例脚本公开于 https://github.com/ningkko/preformulation-gap
- **代码/权重**：完整转录（固定脚本24个、自适应12个）公开于上述GitHub仓库；附录S1-S6提供脚本、评分标准、instruction全文、模型清单和参数
- **关键超参**：gemini-3.5-flash 和 claude-sonnet-4-6 使用 temperature 0.2、response cap 4096 tokens；OpenAI chat-latest 使用默认temperature（alias不接受override）；自适应患者模拟器 gemini-3.5-flash response cap 80 tokens
- **测试日期**：2026年7月28日
