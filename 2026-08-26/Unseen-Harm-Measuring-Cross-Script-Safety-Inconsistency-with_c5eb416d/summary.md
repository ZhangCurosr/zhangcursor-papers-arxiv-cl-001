---
title: "Unseen-Harm-Measuring-Cross-Script-Safety-Inconsistency-with"
source: https://arxiv.org/pdf/2608.24191v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 18:38:11"
field: "多语言仇恨言论检测与跨脚本安全评估"
keywords: ["cross-script safety", "hate speech detection", "Urdu NLP", "LLM bias", "missed-in-Urdu", "WOAH", "label instability", "low-resource language"]
innovations: ["提出missed-in-Urdu指标量化跨脚本安全不一致性", "四条件交叉脚本评测框架分离脚本/转写/混合代码效应", "首个ALW/WOAH九年文献全面审计揭示乌尔都语零覆盖"]
benchmarks: ["HateInsights", "Cyberbullying", "HS-RU-20", "RU-EN Emotion", "Abusive Tweets", "HateXplain"]
---

# 论文速读：Unseen-Harm-Measuring-Cross-Script-Safety-Inconsistency-with

## 一句话总结
本论文首次系统审计 ALW/WOAH 近九年文献中乌尔都语的缺失，并通过四条件（Nastaliq 乌尔都语 / 英文翻译 / Roman 乌尔都语 / 混合代码）对比评测五个 LLM 的仇恨言论分类一致性，揭示前沿模型平均 18.0% 的标签不稳定率及 4.3% 的"missed-in-Urdu"（英译判有害、原文判正常）漏检率，且开源模型表现显著劣于前沿模型。

## 研究问题与动机
- **文献覆盖盲区**：乌尔都语是全球第十大语言（2.46 亿使用者），但在 ALW/WOAH 全部 205 篇论文中零篇专门涉及乌尔都语，且 WOAH 6（2022）已明确点名 Urdu 为"急需技术支持的语言"，却无任何跟进。
- **跨脚本分类一致性未知**：乌尔都语存在 Nastaliq 字体、Roman 转写、Urdu-English 混合代码等多种书写形态，LLM 对同一语义内容在不同脚本下是否给出稳定分类尚无系统度量。
- **现有资源结构性缺口**：不存在专用的 Nastaliq Urdu-English 混合代码仇恨言论数据集，现有数据集多为二元标签或单一脚本，无法覆盖真实社交媒体内容形态。
- **标注可靠性存疑**：5 个模型在 C1 下均将 1,216 条原标注为 Hate 的实例判为 Normal，可能存在标注偏差，但缺乏定性审查，难以判断是模型失败还是标注噪声。

## 核心贡献（创新点）
- **首个 ALW/WOAH 全面文献审计**：通过 ACL Anthology Python API 枚举全部 205 篇论文，量化乌尔都语在顶级在线伤害研究会议中的零代表，并发现孟加拉语（全球第七大语言）同样为零。
- **提出"missed-in-Urdu"指标**：首次量化"在英文翻译中被识别为有害、但在原始乌尔都语脚本中被静默放行"的安全不一致率，衡量实际线上危害漏检规模。
- **四条件交叉脚本评测框架**：设计 C1（Nastaliq 原文）、C2（同模型翻译→英文）、C3（Roman 乌尔都语转写）、C4（混合代码代理）四个实验条件，分离脚本、转写与混合代码对分类结果的影响。
- **开源模型与前沿模型的系统性差距验证**：通过 McNemar / chi-square / Stuart-Maxwell 检验，证实开源模型（Llama-3.1、Qwen-2.5）的标签不稳定率和 missed-in-Urdu 率均显著高于前沿闭源模型（p < 0.0005）。
- **揭示结构性资源缺口作为研究发现**：将"不存在专用 Nastaliq Urdu-English 混合代码数据集"本身作为一个可测量的 gap 直接报告，而非仅仅作为实验限制。

## 方法详解
- **数据集映射**：6 个数据集统一映射为三分类体系（Hate / Offensive / Normal），原始标签经如下转换：THREAT→Hate；INSULT/OFFENSIVE/NAMECALLING/PROFANE/CURSE→Offensive；NONE→Normal。HateXplain 作为英文控制集，理论上应在 C1/C2 下产生 0.0% 不稳定性，以排除 pipeline 噪声。
- **四个实验条件**：
  - **C1（Original Nastaliq Urdu）**：直接使用原始 Nastaliq 乌尔都语文本进行分类。
  - **C2（English translation）**：由被评测模型自身将 C1 文本翻译为英文，再用同一模型对译文分类，确保 C1/C2 差异仅反映脚本敏感性，而非外部翻译系统引入的噪声。
  - **C3（Roman Urdu transliteration）**：使用 HS-RU-20 数据集的 Roman 乌尔都语等价文本，隔离纯转写效应。
  - **C4（Code-switched Roman Urdu-English）**：使用 RU-EN Emotion 语料作为近似代理（注：非 Nastaliq，情绪标签映射为危害标签）。
- **核心指标**：
  - **Label Instability** = 同一实例在 C1 与 C2 下获得不同分类标签的比例。
  - **Missed-in-Urdu** = 在 C2（英文）中被判为 Hate/Offensive、但在 C1（Nastaliq 原文）中被判为 Normal 的比例，代表静默放行的有害内容。
- **统计检验**：
  - **McNemar 检验**：用于配对二元不稳定性比较，关注不一致对（discordant pairs），对方向性误差敏感。
  - **Chi-square 检验**：用于 Missed-in-Urdu 的两两比较（2×2 列联表）。
  - **Stuart-Maxwell 检验**：三类标签（H/O/N）下 C1 vs C2 边际分布差异检验。
  - **Bonferroni 校正**：10 次两两比较，显著性阈值 α = 0.005。
- **提示模板**：分类 prompt 为标准化 zero-shot 指令（HATE/OFFENSIVE/NORMAL 三选一）；翻译 prompt 要求保留语气和强度不做软化处理。两种 prompt 在所有模型和条件下严格一致。

## 实验与结果
- **数据集规模**：5 个乌尔都语相关数据集共约 4,396 条独立文本（N=4,531–4,553/模型），HateXplain 1,000 条作控制。
- **模型列表**：GPT-4o、Claude Sonnet 4.5、Gemini 2.5 Flash（前沿）；Qwen-2.5-7B、Llama-3.1-8B（开源）。
- **主要结果**（Table 3）：

  | 模型 | 标签不稳定率 | Missed-in-Urdu |
  |------|------------|----------------|
  | GPT-4o | 18.0% | 2.4% |
  | Claude Sonnet 4.5 | 16.3% | 3.6% |
  | Gemini 2.5 Flash | 15.9% | 4.3% |
  | Qwen-2.5-7B | 31.6% | 9.9% |
  | Llama-3.1-8B | 27.3% | 6.5% |
  | **中位数（SD）** | **18.0%（7.1%）** | **4.3%（2.9%）** |

- **最强结果**：Gemini 2.5 Flash 以最低不稳定率（15.9%）领先；GPT-4o 在 missed-in-Urdu 上表现最优（2.4%），显著低于 Claude Sonnet 4.5（p < 0.0005）。
- **开源 vs 前沿差距**：Qwen-2.5 和 Llama-3.1 的不稳定率约为前沿模型的 2 倍；missed-in-Urdu 率也显著更高。
- **统计可靠性**：Stuart-Maxwell 检验确认所有模型的 C1→C2 标签分布均有显著偏移（p < 0.01）；所有前沿 vs 开源的两两比较均高度显著（p < 0.0005）。
- **WOAH 文献审计结果**（Table 2）：Hindi 6–8 篇、Arabic 4–5 篇、German 5–6 篇；Urdu 和 Bengali 均为 0 篇。

## 相关工作脉络
- **Ghorbanpour et al. (2025)**：评测 LLM 在八种非英语语言中的仇恨检测，覆盖五大语种（Hindi、Spanish、French、Arabic、Portuguese），但恰好遗漏 Urdu；本文定位差异在于填补此空白并以跨脚本一致性为核心指标。
- **Chan et al. (2024)**：直接证明翻译对 code-mixed 内容无效，检测性能显著退化；本文为跨脚本评测提供了更系统的实验设计，且首次引入"missed-in-Urdu"指标。
- **Nozza (2021)**：发现 zero-shot 跨语言仇恨检测在低资源语言上失效；本文将其结论推广到"同一语言内不同脚本"维度，扩展了低资源语言安全评估的粒度。
- **Ahmad et al. (2025)**：唯一一篇专门针对乌尔都语 LLM 仇恨检测的工作，但仅评测单一模型在单一数据集上的性能，未涉及跨脚本一致性；本文将其定位为孤岛式研究并指出其不足。
- **Dey et al. (2024)**：发现将 South Asian 低资源语言输入翻译为英文后再分类优于原始语言 prompt，与本文"翻译不能替代原文检测"的发现形成张力，本文强调原始脚本评估的必要性。
- **Francielle Vargas (2024) / HausaHate**：在 WOAH 框架内研究 Hausa 仇恨检测，与本文形成结构性类比，共同揭示低资源语言仇恨研究是系统性缺口而非孤立问题。

## 局限性与未来方向
- 数据集样本量偏小（700–1,000/数据集），二进制标签数据集（Abusive Tweets、HS-RU-20）缺少 Offensive 类别，可能人为 inflate 不稳定率。
- C2 翻译步骤由被评测模型自身完成，无法分离"翻译质量"与"分类一致性"两个维度的贡献。
- 1,216 条模型与 gold Hate 标注冲突的实例未经人工审查，无法区分标注偏差与模型失败。
- RU-EN Emotion 作为 C4 代理数据集使用了 Roman 而非 Nastaliq 脚本，且情绪标签映射为危害标签，是近似而非精确替代。
- ACL Anthology 审计仅基于标题和摘要，未涵盖全文提及 Urdu 但未在摘要中出现的情况。
- 所有模型通过商业 API 访问，结果受 provider 版本控制影响，长期复现性不保证。
- Perspective API 覆盖 18 种语言却唯独不支持 Urdu，反映了工业界现有的安全评估盲区。
- 未来方向：构建专用 Nastaliq Urdu-English 混合代码仇恨数据集、对冲突标注进行定性审查、探究开源模型跨脚本一致性劣化的根因（训练数据脚本分布偏差）。

## 研究启发与可借鉴点
- **"missed-in-X"指标的可迁移性**：该指标逻辑（原文判正常 vs 翻译判有害）可直接推广至其他多脚本/多转写系统的低资源语言（如阿拉伯语方言、西里尔/拉丁双写系统语言），作为安全一致性评测标准。
- **四条件交叉脚本实验设计**：C1–C4 设计可有效分离"脚本效应""转写效应""混合代码效应"，适用于任何具有多书写变体的语言安全评估研究。
- **英文翻译作为内部对照**：使用被评测模型自身完成翻译（而非外部翻译服务），有效控制了翻译质量噪声，这一设计可复用于其他语言的跨脚本评测。
- **HateXplain 作为 pipeline 控制集**：在纯英文数据集上验证 0.0% 不稳定率，可快速排除实验基础设施问题，是跨语言评测的可靠 sanity check 范式。
- **统计检验组合策略**：McNemar（二元配对）+ chi-square（比例比较）+ Stuart-Maxwell（三类标签分布）+ Bonferroni 校正的组合，为 NLP 分类器对比提供了完整的统计验证模板，可直接沿用。

## 关键术语表
- **Nastaliq Urdu**：乌尔都语的传统书法字体风格，广泛用于巴基斯坦和印度日常书写，与拉丁字母转写形式（Roman Urdu）在视觉和模型处理上截然不同。
- **Roman Urdu**：使用拉丁字母转写的乌尔都语，常见于社交媒体和即时通讯，与 Nastaliq 形式在字符集和模型 tokenization 上存在系统性差异。
- **Code-switching（混合代码）**：在同一话语中交替使用乌尔都语和英语，巴基斯坦社交媒体中的普遍现象，现有数据集对此覆盖严重不足。
- **Missed-in-Urdu**：本文提出的核心指标，指内容在英文翻译中被模型判为有害（Hate/Offensive）但在原始 Nastaliq 乌尔都语文本中被判为正常的比例，反映脚本条件导致的安全漏洞。
- **Label Instability（标签不稳定性）**：同一内容在 C1（Nastaliq）与 C2（英文翻译）条件下获得不同分类标签的比例，衡量模型跨脚本分类一致性。
- **McNemar 检验**：用于配对二元分类结果的显著性检验，关注不一致对的数量，适合比较同一测试集上两个分类器的差异。
- **Stuart-Maxwell 检验**：McNemar 检验的多类扩展，用于检验配对的三类（或更多类）标签分布是否存在显著边际差异。
- **ALW/WOAH**：Abusive Language Online / Workshop on Online Abuse and Harms，NLP 领域研究在线虐待和伤害内容的旗舰研讨会，本文审计了其 2017–2025 年全部九届会议论文。

## 可复现要素
- **数据集**：HateInsights、Cyberbullying、Abusive Tweets、HS-RU-20、RU-EN Emotion、HateXplain 均为公开数据集（部分需注册获取），代码仓库匿名链接见附录。
- **代码开源**：匿名评审链接 https://anonymous.4open.science/r/WOAHJul26-047C，正式录用后将替换为去匿名化链接。
- **模型访问**：通过 OpenAI、Anthropic、Google、Together.ai、Groq 的云端 API 调用，非本地推理。
- **关键超参**：zero-shot 分类模式，无 in-context examples；标准化三分类 prompt（HATE/OFFENSIVE/NORMAL）；Bonferroni 校正 α = 0.005（10 次比较）。
- **运行环境**：Windows 机器，Anaconda Python 虚拟环境，NVIDIA GeForce RTX 5090 GPU（仅用于本地备用，实际分类通过 API）。
- **API 速率限制**：主要约束来自 Groq 免费层的 Llama-3.1 调用速率；支持断点续跑（增量写入磁盘）。
