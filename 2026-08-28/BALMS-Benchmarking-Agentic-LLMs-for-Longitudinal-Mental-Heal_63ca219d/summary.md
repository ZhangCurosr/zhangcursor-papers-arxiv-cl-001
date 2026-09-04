---
title: "BALMS-Benchmarking-Agentic-LLMs-for-Longitudinal-Mental-Heal"
source: https://arxiv.org/pdf/2608.27219v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 06:45:46"
field: "可穿戴感知与大模型结合的精神健康预测"
keywords: ["Longitudinal Mental Health", "Agentic LLM", "Wearable Sensing", "LLM-as-Judge", "Chain-of-Thought", "RAG", "Mode Collapse", "Temporal Reasoning"]
innovations: ["首个纵向精神健康感知 LLM Agent 基准（BALMS），联合评测封闭分数预测与基于证据的论述生成", "提出双层 LLM-as-Judge 评判框架（Tier-1 六维时序推理 + Tier-2 五维论述质量）", "揭示工具类 Agent 的模式坍塌与 schema 敏感故障，发现语义紧凑特征优于原始高频流"]
benchmarks: ["BALMS", "Health-LLM", "PHIA", "TemporalBench", "GLOBEM", "PMData", "DiversityOne"]
---

# 论文速读：BALMS-Benchmarking-Agentic-LLMs-for-Longitudinal-Mental-Heal

## 一句话总结
本文提出 BALMS，首个针对**纵向精神健康感知**的 LLM Agent 系统评测基准，在 3 个真实可穿戴/手机数据集上，以 2 类任务（封闭分数预测 + 基于证据的论述生成）评测 3 种 Agent 范式（Prompt、Tool、Memory）跨 5 个 LLM 主干，发现零样本 Agent 仅在强模型或紧凑语义特征上才能超越简单均值基线，CoT 仅对推理型模型有效，工具类 Agent 存在严重的模式坍塌与 schema 敏感问题。

## 研究问题与动机
1. **现有自陈量表评估频度低**：临床精神健康评估依赖周期性问卷（如 PHQ-4、PMSys），只能提供稀疏快照，无法捕捉数周至数年的动态波动。
2. **可穿戴传感提供连续信号但缺乏推理能力**：Fitbit、智能手机可被动收集睡眠、步数、心率等纵向多通道信号，但现有系统仅能做短期事实查询（如"本周最高步数"），无法从长期行为模式中推断精神健康状态。
3. **LLM 直接处理长数值序列不可靠**：多通道、高频传感数据迅速超出 LLM 上下文窗口；LLM 已知在长数值序列上的数值推理能力有限（Spathis & Kawsar, 2024）。
4. **不同数据集的 schema 差异导致泛化困难**：DiversityOne（原始手机流）、PMData（Fitbit 预聚合）、GLOBEM（混合手机+智能手表）的传感器类型和特征格式差异显著，一个数据集上设计良好的 Agent 难以直接迁移。

## 核心贡献（创新点）
1. **首个纵向精神健康感知 Agent 基准（BALMS）**：形式化为"封闭分数预测 + 基于证据的论述生成"两类任务，要求模型同时输出可验证数字分数和支持性时序论证。
2. **统一评测三路 Agent 范式 × 五路 LLM 主干**：Prompt-based（Health-LLM）、Tool-based（PHIA）、Memory-based（Chunk RAG / RAPTOR）在三个真实数据集上一致评测，建立标准化基础设施。
3. **LLM-as-Judge 双层评判框架**：Tier-1 改编自 TemporalBench 的六维时序推理能力（对齐/切分/差异/滞后/结构/交互）；Tier-2 改编自 PHIA 的五维论述质量（可信度/证据 grounding/领域知识/安全性/清晰度），使开放论述可系统自动评分。
4. **揭示三种关键经验规律**：①零样本 Agent 仅在强 backbone 或紧凑语义特征下优于均值基线；②CoT 仅对 DeepSeek/Claude 等推理型模型有效，对 instruct 模型反而恶化；③工具类 Agent（PHIA）在 Fitbit 格式数据上表现良好，但在原始手机流上存在模式坍塌（mode collapse）与 schema 盲聚合等系统性故障。

## 方法详解

### 数据集
| 数据集 | 参与人数 | 时长 | 传感器 | 预测目标 |
|---|---|---|---|---|
| DiversityOne | 782（只用蒙古子集，1,355 样本） | 28 天 | 10（加速度计、陀螺仪、Wi-Fi 等） | Mood（1-5） |
| PMData | 16（1,448 样本） | 5 月 | 5（步数、静息心率、卡路里等） | Stress（1-5 Likert） |
| GLOBEM | 497（只用 INS-W_1，1,640 样本） | 多年（只取一个 10 周窗口） | 8（位置、屏幕、步数、睡眠等） | PHQ-4 Anxiety（0-3） |

### 任务定义
- **T1（封闭预测）**：给定目标日 T 和前向回看窗口（DiversityOne=7 天，PMData/GLOBEM=14 天）的多变量传感信号，输出单整数自陈分数；以 **MAE** 评测。
- **T2（开放论述）**：要求模型输出自由形式 chain-of-thought 论述，引用具体通道、数值并推理多天模式（趋势、周循环、恢复）；以 **LLM-as-Judge** 双层 rubric 评分。

### 三种 Agent 范式
1. **Prompt-Based（Health-LLM）**：将每日每通道信号格式化为数值数组拼成单一 prompt，直接让 LLM 输出整数；无检索、无工具、无记忆。
2. **Tool-Based（PHIA）**：将用户完整记录预加载为内存中 pandas DataFrame，通过 ReAct 循环（最多 10 步）由 LLM 生成 Python 代码执行日期过滤、groupby 聚合等操作，执行结果追加到轨迹，直至输出最终答案。
3. **Memory-Based**：
   - **Chunk RAG**：按"天"粒度（非 token 粒度）用 sentence-transformer 编码历史每日数据入库；推理时按余弦相似度检索 top-k（k=3）最相似天数，连同查询日一起送入 LLM。
   - **RAPTOR**：两级树结构——叶节点为亚日记录（PMData 取半小时槽，GLOBEM 取一天四时段），内部节点为 LLM 生成的该日抽象摘要；检索在同一树平面上竞争 top-k。

### LLM-as-Judge 详细设计
使用 **Llama-3.3-70B-Instruct** 作为单一 judge，greedy decoding，对每条 agent 输出给出：
- **Tier-1 时序推理**（C1 对齐、C2 切分、C3 差异判断、C4 滞后、C5 结构、C6 交互）：打分 {正确=1, 错误=0, 未调用=null}，不罚未调用；报告调用率、条件准确率与覆盖率。
- **Tier-2 通用质量**（Faithfulness / Evidence Grounding / Domain Knowledge / Safety / Clarity / Overall）：1-5 Likert，其中 Faithfulness 只看"预测是否从论述中逻辑推出"，不要求与 ground truth 一致。

### 实现细节
- 开源模型用 vLLM 在 4×A6000 本地部署；Claude-Haiku-4.5 走 Anthropic API。
- PHIA ReAct 迭代上限 10 步；Chunk RAG 用 LangChain 官方实现；RAPTOR 按数据集亚日粒度构建两级树，top-3 检索。

## 实验与结果

### T1 主要结果（MAE↓，括号内为均值基线）
| Backbone | 方法 | DiversityOne(0.58) | PMData(0.48) | GLOBEM(0.80) |
|---|---|---|---|---|
| **Claude-Haiku-4.5** | Health-LLM | **0.42** ✓ | 0.64 | **0.83** |
| | PHIA | 1.11 | **0.54** | 1.41 |
| **DeepSeek-R1-14B** | Health-LLM | 0.49 | 0.65 | 1.23 |
| | RAG | 0.48 | 0.58 | 1.11 |
| **Qwen2.5-14B** | Health-LLM | **0.44** ✓ | 0.62 | 0.86 |
| | RAG | 0.53 | 0.71 | 0.90 |
| Mistral-7B | Health-LLM | 1.06 | 0.88 | 1.83 |
| | PHIA | 0.84 | 0.63 | 1.29 |

**关键发现**：
- 仅 **Claude-Haiku-4.5 + Health-LLM** 在 DiversityOne 上显著超越均值基线（0.42 vs 0.58）；PHIA 在 PMData 上接近基线（0.54 vs 0.48）。
- 小模型（Mistral-7B）在所有数据集上均大幅落后于均值基线。
- PHIA **不随 backbone 增强而改善**：Qwen2.5-14B 的 PHIA 在 GLOBEM 上 MAE=1.52，远差于 0.80 的均值基线，而 Claude-Haiku-4.5 的 PHIA 为 1.41——瓶颈不在语言推理而在代码执行契约。
- **CoT 效应**：DeepSeek 与 Claude 的 +CoT 普遍降低 MAE（DeepSeek-RAG 在 DiversityOne 从 0.48→0.41，降幅 14.9%；Claude Health-LLM 0.42→0.41）；Qwen/Mistral 的 +CoT 反而恶化。

### 纵向扩展（Section 4.3）
- **Health-LLM** 随回看窗口增长而退化（文本化记录稀释目标日证据）。
- **RAG** 在 PMData 上 MAE 下降约 29%（更多历史可供检索），在 GLOBEM 上仅降约 9%（原始手机流信息密度低）。
- **PHIA** 在 PMData 上稳定（DataFrame 与 prompt 长度解耦），在 GLOBEM 上仍脆弱且延迟高。

### 传感器敏感性（Table 4）
在 GLOBEM 上，**丢弃原始手机流仅保留 Fitbit 语义特征（Fitbit-only）**，多数配置**匹配或超过 Full 设置**（如 Qwen2.5-14B PHIA：1.52→1.05；DeepSeek PHIA：1.14→1.08），说明当代 LLM 擅长推理高级语义原语而不会从原始高频流中提取信号。

### T2 LLM-as-Judge（Figure 4 + Table 6）
- 仅 **Claude-Haiku-4.5** 同时在 Tier-1 时序推理（各维度调用率与准确率）和 Tier-2 总体质量（Overall ~4.3）上双优。
- Qwen2.5-14B 的 Health-LLM RAG 在 Evidence Grounding 上仅 4.00，远低于 PHIA 的 4.77（后者虽数值对但论述空泛）。
- Case study（Appendix D.1）揭示 Qwen2.5-7B 的 CoT 存在**错误日期归因**（如"day 2 睡眠 139 min"实际是 day 3），而 Claude 正确引用所有数值并能识别跨 14 天的 location entropy 下降趋势（1.80→0.85）。

### PHIA 失败模式分析（Appendix B）
归纳 5 类故障：
- **F1 静默代码执行**：`print()` 缺失或截断，agent 误认为计算成功。
- **F2 跨 [Act] 块状态丢失**：之前定义的变量（`avg_steps`、`filtered_df` 等）在后续块中 NameError。
- **F3 预算耗尽后幻觉论述**：force-finish 生成看似合理但从未计算的数值声称。
- **F4 魔术数字阈值**：硬编码 `steps > 20000` 等，导致**模式坍塌**（GLOBEM 上 85.9% 输出同一标签"2"）。
- **F5 schema 盲聚合**：对半小时槽直接 `.mean()` 而非先重索引到日粒度，语义错误。

## 相关工作脉络
1. **Health-LLM (Kim et al., 2024)**：最早的 wearable-sensing LLM 零样本预测框架，按日数值数组 prompt 格式；BALMS 直接复用其模板作为 prompt-based 基线，并提出 T2 论述任务与之区分。
2. **PHIA (Merrill et al., 2026)**：首个将 ReAct + pandas DataFrame 引入被动感知的 agent；BALMS 用其作为 tool-based 代表，并首次揭示其在 GLOBEM/DiversityOne 上的模式坍塌与 schema 敏感问题。
3. **GLOSS (Choube et al., 2025) / PHA (Heydari et al., 2025)**：同属 tool-based 家族，但各自绑定特定传感器 schema，缺乏跨队列泛化评测；BALMS 将其定位为"数据集中心化"工作对比。
4. **TemporalBench (Cai et al., 2024)**：视频时序理解评测；BALMS 将其 C1-C6 六维操作改编为 Tier-1 时序推理 rubric，首次用于可穿戴纵向传感论述评估。
5. **LifeAgentBench (Tian et al., 2026)**：个人健康助手多维度 agent 评测；与之相比，BALMS 专注纵向多日/多年传感的数值 grounding，而非应用操控或对话交互。
6. **MedAgentBench (Jiang et al., 2025) / AgentClinic (Schmidgall et al., 2025)**：电子病历与临床环境 agent 评测；均属文本模态，BALMS 首次将 agent 评测扩展到**多通道时间序列数值推理**场景。

## 局限性与未来方向
1. **仅评测单 agent 范式**，未涉及 multi-agent 协作、planner-executor 架构、或动态组合 tool+memory+reflection 的设计。
2. **仅限回溯预测**，未在实际交互式或干预场景中测试 agent 的在线表现。
3. **LLM-as-Judge 不能替代临床专家或人类受试者评估**；尽管两 tier rubric 系统化，仍存在 judge 偏差风险。
4. **GLOBEM 仅用一个 10 周窗口**，未评估跨学年连续性（参与者变动、校准漂移、行为基线偏移）。
5. **人口代表性有限**：DiversityOne 只用蒙古子集，PMData 仅 16 人，GLOBEM 为学生队列，泛化到异质临床人群需谨慎。
6. **未来方向**：结合 clinician/user 评估、前瞻性部署测试、安全/校准/个性化研究。

## 研究启发与可借鉴点
1. **"语义紧凑特征胜过原始高频流"**：GLOBEM 的 Fitbit-only 配置经常匹敌或超越 Full 设置——提示团队在构建 sensing agent 时应优先提取高级行为原语（如日均步数、睡眠效率）而非直接喂 raw stream。
2. **工具类 Agent 需严格执行契约**：PHIA 的五类故障（尤其 F1 静默执行、F4 魔术阈值）提示：任何 code-interpreter agent 都应内置"执行结果必须可解析"的断言机制，以及阈值自适应推导而非硬编码。
3. **CoT 与 backbone 类型强耦合**：推理蒸馏模型（DeepSeek-R1）从 CoT 受益，而 instruct 模型反而受损——团队在设计推理链路时应先做 backbone 类型分层实验，不盲目加 CoT。
4. **以天为粒度的 chunk RAG 适配传感数据**：将传统 token/段落级 RAG 改为 day-level chunk + sentence-transformer 余弦检索，是平衡上下文节省与信息完整性的有效折衷；RAPTOR 的两级树进一步引入亚日叶节点，值得在细粒度场景尝试。
5. **LLM-as-Judge 双层 rubric 可直接迁移**：Tier-1 六维时序操作 + Tier-2 五维论述质量框架，可用于任何需要"数值预测 + 证据支撑论述"的 agent 评测（如金融时间序列解读、医疗报表推理）。

## 关键术语表
- **BALMS**：Benchmarking Agentic LLMs for Longitudinal Mental Health Sensing，首个面向纵向精神健康感知的 LLM Agent 系统评测基准。
- **Look-back window**：用于预测目标日分数的前向历史传感窗口长度（本文默认 7/14 天）。
- **Mode collapse（模式坍塌）**：Agent 对绝大多数输入输出同一预测标签（如 GLOBEM 上 85.9% 输出"2"），丧失对个体传感历史的区分能力。
- **LLM-as-Judge**：用另一个强 LLM（本文 Llama-3.3-70B）按结构化 rubric 自动评分 agent 生成的开放论述质量与时序 grounding。
- **Tier-1 / Tier-2**：Tier-1 为六维时序推理正确性（C1-C6）；Tier-2 为五维论述通用质量（Faithfulness/Evidence/Domain/Safety/Clarity）。
- **Schema sensitivity**：Agent 性能随底层传感数据的字段结构（日粒度预聚合 vs 半小时间原始流）剧烈波动的现象。
- **ReAct loop**：Reasoning + Acting 循环，agent 在每步生成思维链并选择工具动作（如执行 Python 代码），直到输出最终答案。
- **PHQ-4**：Patient Health Questionnaire-4，包含抑郁和焦虑各两个条目的简短自陈量表，本文取其焦虑 subscale（0-3）作预测目标。

## 可复现要素
- **数据集**：DiversityOne（公开）、PMData（公开）、GLOBEM（restricted access）——均为二手再分析，已完成去标识化。
- **代码/权重**：论文使用官方实现（Health-LLM、PHIA、LangChain RAPTOR），**未声明单独开源 BALMS 代码**；模型推理在 4×A6000 本地（vLLM）+ Claude API 完成。
- **关键超参**：Look-back window（DiversityOne=7 天，PMData/GLOBEM=14 天）；RAG top-k=3；RAPTOR 叶粒度（PMData=半小时槽，GLOBEM=一天四时段）；PHIA ReAct 最大迭代=10；Judge 模型 Llama-3.3-70B-Instruct greedy decode，8-bit 量化。
