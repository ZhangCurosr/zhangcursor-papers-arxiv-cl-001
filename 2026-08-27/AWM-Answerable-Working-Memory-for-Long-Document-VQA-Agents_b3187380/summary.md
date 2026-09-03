---
title: "AWM-Answerable-Working-Memory-for-Long-Document-VQA-Agents"
source: https://arxiv.org/pdf/2608.25618v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 18:39:03"
field: "多模态文档理解与Agent"
keywords: ["长文档VQA", "working memory", "GRPO", "多模态agent", "memory evaluation", "reward design"]
innovations: ["提出memory-only answerability诊断与四象限分类", "设计AWM Reward将终止态memory可回答性纳入GRPO奖励", "通过EP-given受控设置隔离memory提取与页面访问质量"]
benchmarks: ["MMLONGBENCH-DOC", "LONG-DOCURL"]
---

# 论文速读：AWM-Answerable-Working-Memory-for-Long-Document-VQA-Agents

## 一句话总结
本文针对长文档VQA Agent的记忆质量盲区，提出memory-only answerability诊断指标与AWM-GRPO训练框架，通过将终止态working memory的可回答性纳入GRPO奖励，在保持最终答案优先的同时显著提升了记忆质量。

## 研究问题与动机
- 现有长文档VQA Agent评估仅关注最终答案正确性与证据页访问，忽视了working memory本身是否承载足够支撑答案的证据。
- 即便提供gold证据页，42.5%的正确回答仍无法仅从agent写入的终止态memory中独立作答（memory-missing correct，MMC）。
- 答案正确性与memory质量之间存在脱节：agent可能依赖原始页面上下文作答，而留下的memory过于泛化或缺失关键细节。
- 受控实验表明，仅提升terminal working memory质量即可独立提升最终答案与memory-only回答准确率，证明优化memory artifact本身具有直接价值。

## 核心贡献（创新点）
1. **提出memory-only answerability诊断**：通过冻结reader仅用问题与终止态memory作答，量化agent记忆中答案支撑证据的完整性，与已有工作仅评估最终答案的做法形成本质区别。
2. **设计四象限奖励函数AWM Reward**：将轨迹划分为四种细胞（11/10/01/00），通过最小标量奖励实现"记忆细化偏好"、"答案优先门控"与"有用记忆失败"三级排序，区别于answer-only GRPO仅依赖最终答案的二值奖励。
3. **构建AWM-GRPO训练框架**：在线GRPO中使用冻结Qwen3-14B生成memory-only答案并计算双评分，仅更新4B策略，在相同backbone、检索与工具设置下实现端到端可控对比。
4. **揭示并量化memory-quality gap**：通过EP-given受控设置证明即使消除检索噪声，terminal memory的可回答性仍显著低于最终答案准确率，暴露现有评估盲点。

## 方法详解
**Agent-Memory框架**：
- 长文档VQA实例为三元组 $(D, q, y)$，其中 $D = \{p_1, \dotsc, p_N\}$ 为多页文档。
- Agent执行retrieve-inspect-update循环：检索Top-3候选页图像 → 检查页面 → 写入带source-link的发现到working memory $M_t$，最多15步。
- 终止时输出最终答案 $\hat{y}$ 与terminal working memory $M^{\mathrm{term}}$。

**Memory-only Answerability诊断**：
- 冻结reader $R$ 接收 $(q, M^{\mathrm{term}})$，无轨迹与页面图像，生成memory-only答案 $R(q, M^{\mathrm{term}})$。
- 官方judge $J$ 评估最终答案与memory-only答案的正确性，得到二值变量 $s_{\mathrm{ans}}$ 与 $s_{\mathrm{mem}}$。
- 定义 $P_{\mathrm{mmc}} = \sum_i \mathbf{1}[s_{\mathrm{ans}}^{(i)}=1 \land s_{\mathrm{mem}}^{(i)}=0] / \sum_i \mathbf{1}[s_{\mathrm{ans}}^{(i)}=1]$，衡量正确回答中memory不可回答的比例。

**AWM Reward设计**：
$$r_{11}=\beta, \quad r_{10}=0, \quad r_{01}=\gamma, \quad r_{00}=\omega, \quad \beta > 0 > \gamma > \omega$$
- 默认值：$(\beta, \gamma, \omega) = (2, -0.1, -1)$。
- 归一化优势：$A_i = (r(\tau_i) - \mu_g) / (\sigma_g + \epsilon)$。
- 三种偏好：
  - Memory refinement：$(1,1)$ 优于 $(1,0)$，Margin = $\beta / (\sigma_g + \epsilon)$
  - Final-answer gate：$(1,0)$ 优于 $(0,1)$，Margin = $-\gamma / (\sigma_g + \epsilon)$
  - Useful-memory failure：$(0,1)$ 优于 $(0,0)$，Margin = $(\gamma - \omega) / (\sigma_g + \epsilon)$

**训练设置**：
- SFT阶段使用DOC-750K的994条轨迹（kimi-k2.5为teacher）。
- GRPO阶段使用226个mixed-difficulty组，每组8条轨迹。
- 在线训练使用冻结Qwen3-14B生成memory-only答案与计算奖励，仅更新Qwen3-VL-4B策略。

## 实验与结果
**数据集**：MMLONGBENCH-DOC（N=1082）、LONG-DOCURL（N=2325）。

**主要结果**（Table 4）：
| 方法 | Backbone | MMLONGBENCH-DOC | LONG-DOCURL |
|------|----------|-----------------|-------------|
| VLM + RAG Top-3 | Qwen3-VL-4B | 45.8 | 48.2 |
| AWM-Agent (SFT) | Qwen3-VL-4B | 49.2 | 55.7 |
| + Answer-GRPO | Qwen3-VL-4B | 51.6 | 57.4 |
| **+ AWM-GRPO** | Qwen3-VL-4B | **53.9** | **60.1** |

- AWM-GRPO较RAG Top-3提升 +8.1（MMLONGBENCH-DOC）/ +11.9（LONG-DOCURL）点。
- 较Answer-GRPO提升 +2.3 / +2.7 点。

**Memory质量分析**（Table 5-7）：
- SFT：$P_{\mathrm{mmc}} = 17.7\%$，Mem acc = 38.5%
- + Answer-GRPO：$P_{\mathrm{mmc}} = 19.9\%$，Mem acc = 42.5%（答案精度提升但memory质量恶化）
- **+ AWM-GRPO**：$P_{\mathrm{mmc}} = 17.2\%$，Mem acc = 44.5%（同时提升最终答案与memory质量）
- EP-given受控设置下，AWM-GRPO将$P_{\mathrm{mmc}}$从19.1%降至16.4%，证明reward对memory提取本身的贡献独立于检索质量。

**训练动态**：Memory-only accuracy从step 40的38.8%单调提升至step 280的43.5%。

**按证据源分类**（Table 7）：AWM-GRPO在mixed visual+text、text-multi、figure、chart等类别降低$P_{\mathrm{mmc}}$，但table与layout类别表现较差。

## 相关工作脉络
1. **Doc-V\***（Zheng et al., 2026）：将长文档VQA视为顺序页面导航任务，使用 imitation learning + GRPO训练navigation policy；本文保持相同agent架构，但引入memory answerability作为额外RL信号。
2. **MemSearcher**（Yuan et al., 2025）：研究文本搜索agent中compact working memory的跨轮传递，使用multi-context GRPO；本文聚焦page-image VQA中的多轮页面检查记忆保留。
3. **AgenticRAG-R1**（Jiang et al., 2026）与**TC-RAG**（Jiang et al., 2025）：分别使用stack memory与Turing-complete RAG处理复杂推理；本文不改变memory schema，仅优化其内容质量。
4. **CapRL**（Xing et al., 2026）与**Vision-SR1**（Li et al., 2026b）：通过RL奖励可回答caption与question-conditioned视觉描述；本文将中间状态监督扩展到多页检查累积的terminal memory。
5. **Evidence-citation工作**（Liu et al., 2023; Gao et al., 2023）：检查答案是否被引用证据支持；本文测试memory是否独立可回答，两者正交——可回答memory可能含无支持声明，grounded memory可能遗漏必要事实。

## 局限性与未来方向
- 实验仅限Qwen3-VL-4B在LONG-DOCURL与MMLONGBENCH-DOC上，未验证30B+模型或其他文档分布（科学论文、财务文件、幻灯片）。
- Memory-only answerability依赖固定Qwen3-14B reader，未测试第二reader的鲁棒性。
- Judge的answer-extraction步骤存在误差（100例人工审计发现6处分歧），更大规模审计可提供更强证据。
- AWM reward仅测量answerability而非完整source-grounding验证；未来可添加per-finding source-image grounding或self-consistency over multiple reader rollouts。
- Memory-only answerability要求额外reader pass与extraction pass，虽不影响部署但增加离线评估开销。

## 研究启发与可借鉴点
1. **中间状态监督的迁移价值**：将working memory作为可评估的artifact，而非仅关注最终输出，这一思路可迁移至多模态agent、code agent等需要跨步状态维护的场景。
2. **可控对照实验设计**：EP-given设置通过替换检索为gold页面隔离memory提取质量，四细胞分型（11/10/01/00）提供细粒度诊断，值得在同类工作中复用。
3. **Minimal reward design**：AWM使用4个标量奖励实现三级偏好，避免复杂per-finding grounding或length penalty，平衡了信号强度与训练稳定性，可作为RL reward设计的参考范式。
4. **与团队方向的结合机会**：若团队研究知识图谱RAG或文档理解agent，可将memory-only answerability作为额外诊断指标；若研究多模态agent，可探索将memory reward扩展至跨模态证据聚合场景。

## 关键术语表
**Memory-only answerability**：仅凭问题与agent写入的终止态working memory，由冻结reader独立作答并判断正确性的诊断指标。
**AWM-GRPO**：将memory-only answerability纳入GRPO奖励的强化学习训练框架，通过四象限奖励函数实现记忆细化偏好。
**$P_{\mathrm{mmc}}$（Memory-Missing-Correct rate）**：在最终答案正确的样本中，terminal working memory不可回答的比例，衡量记忆质量缺口。
**EP-given（Evidence-Page-given）**：将检索替换为gold证据页面的受控设置，用于隔离memory提取质量与页面访问质量的影响。
**Four-cell taxonomy**：按$(s_{\mathrm{ans}}, s_{\mathrm{mem}})$划分的四种结果细胞：memory-supported correct、memory-missing correct、answering error、unresolved error。
**Source-linked finding**：agent写入working memory的带页面来源标注的发现条目，格式为"来源页+发现内容"。
**Qwen3-VL-4B-Instruct**：本文使用的4B参数多模态VLM backbone，通过SGLang serving。
**GRPO（Group Relative Policy Optimization）**：DeepSeekMath提出的RL算法，通过组内归一化优势进行策略更新。

## 可复现要素
- **数据集**：MMLONGBENCH-DOC（N=1082）与LONG-DOCURL（N=2325）均为公开benchmark。
- **代码/权重**：Qwen3-VL-4B-Instruct为公开开源模型；AWM相关代码与训练脚本论文未明确提及开源声明（需查阅项目主页或arxiv附件）。
- **关键超参**：
  - 检索：Top-3，Jina v4 page-image embeddings + Qdrant索引
  - Agent步数上限：15步
  - Memory schema：每发现限制40词，需带source_page引用
  - GRPO组大小：G=8
  - Reward权重：$(\beta, \gamma, \omega) = (2, -0.1, -1)$
  - SFT轨迹：994条（DOC-750K的2000题rendered子集）
  - GRPO轨迹：226个mixed-difficulty组
  - Teacher模型：kimi-k2.5（SFT）、Qwen3-14B（在线GRPO reward scoring）
  - 评估reader：冻结Qwen3-14B
  - 评估judge：GPT-4o temperature=0提取答案 + 各benchmark官方确定性规则评分
