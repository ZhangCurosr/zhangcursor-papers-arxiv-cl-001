---
title: "Expectations-and-Practices-around-AI-Disclosure-in-CS-Resear"
source: https://arxiv.org/pdf/2608.23271v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 19:50:16"
field: "科研伦理与AI治理"
keywords: ["AI Disclosure", "Generative AI in Research", "Research Integrity", "Policy Analysis", "Computer Science Publishing", "Human-AI Collaboration"]
innovations: ["首次系统性量化CS领域21项研究任务×3个人类参与水平的AI披露必要性", "揭示EMNLP/ICLR实际披露行为与读者期望的严重错位（责任声明缺失、写作辅助过度披露）", "提出任务分级（mandatory/recommended/optional）与标准化boilerplate模板框架"]
benchmarks: ["EMNLP 2025", "ICLR 2026", "CSRankings 65 conferences", "Gemini 2.5 Flash annotation F1 96.5%/90.6%"]
---

# 论文速读：Expectations-and-Practices-around-AI-Disclosure-in-CS-Resear

## 一句话总结
本文通过政策分析（65个顶会）、研究者调查（N=109）和披露声明审计（13,867条）三重方法，系统考察了计算机科学领域AI使用披露政策的现状、研究者的期望与实际操作之间的差距，并提出了任务分级披露政策与标准化模板的建议。

## 研究问题与动机
- **政策空白**：尽管AAAI、ACL、ACM、IEEE等主流学会及35/65个CS顶会已建立AI披露政策，但这些政策自2023年引入后变化极少，缺乏具体指导何时披露、披露哪些细节，且对"研究设计"与"写作辅助"等任务的区分不够明确。
- **期望不明**：现有文献（如Resnik & Hosseini, 2026）主张AI辅助直接影响研究结果或生成/分析内容时需披露，但不同研究阶段（idea generation vs. data analysis vs. writing）的披露必要性可能存在差异，且受人类参与程度（high/assumed/low）影响，尚缺乏实证数据支撑。
- **实践脱节**：ACLPaper和ICLR等会议的作者实际披露行为是否与读者期望一致？是否存在"表演性披露"（performative disclosure）——即仅为合规而非提供实质信息的重复模板声明——目前尚无系统审计。

## 核心贡献（创新点）
1. **首次系统性CS领域AI披露政策审计**：覆盖65个CS顶会与4大学会政策，揭示了政策同质化（29/35个会议直接引用ACM政策）且演进停滞（多数仅修订1-4次）的现状，区别于先前零散的政策综述。
2. **多维度披露必要性量化**：通过N=109的研究者调查，首次实证刻画了21项研究任务×3个人类参与水平组合下的披露必要性评分（均值2.95/5），发现"研究设计"阶段（如生成合成数据集，mean=4.02）比"写作报告"阶段（如编辑论文可读性，mean=1.8）更需披露，且低人类参与度显著提升披露需求（β=0.49, p<0.001）。
3. **期望与实践的巨大差距揭示**：审计EMNLP 2025（40%披露率）和ICLR 2026（64%披露率）的13,867条声明，发现71%受访者期望包含"责任声明"（responsibility declaration），但实际仅2%（EMNLP）和23%（ICLR）包含；而最不需要的"文本润色"（editing）和"代码编辑"却是最常被披露的任务（ICLR中96.5%和16.1%），暴露出政策与行为的严重错位。
4. **任务分级披露框架与模板**：基于调查数据提出三级分类（mandatory rating≥3.5、recommended 2.5-3.5、optional <2.5），并设计包含7个细节维度（task/model/reason/non-use/oversight/responsibility/purpose）的boilerplate模板，直接推动政策从" blanket requirement"转向"task-specific guidelines"。

## 方法详解
- **政策分析**：从CSRankings选取65个AI/Systems/Theory/Interdisciplinary顶会，结合Internet Archive Wayback Machine追溯AAAI/ACL/ACM/IEEE政策演变（2023-2026），编码政策存在性、来源链接及修订次数。
- **混合方法调查**：在线问卷（LimeSurvey），21项任务（分属idea generation、research design、data collection、data analysis、writing/reporting五阶段）×3个人类参与水平（high/assumed/low，见表1），5点Likert量表评分；拟合线性混合效应模型 $y = \beta_0 + \beta_{task}x_{task} + \beta_{human}x_{human} + u_{part} + \epsilon$，以参与者为随机截距，控制个体变异。
- **披露声明审计**：通过OpenReview API获取ICLR 2026的19,525篇论文及ACL Anthology获取EMNLP 2025的3,216篇论文，使用Gemini 2.5 Flash进行两阶段提取（检查表E/E1→定位section→抽取disclosure），人工验证F1达96.5%（details）和90.6%（tasks）；按表2/8的7维度标注细节与任务类型，统计覆盖率与长度分布。

## 实验与结果
- **政策普及**：35/65（54%）CS会议有披露政策，其中29个直接引用ACM政策；学会政策修订次数：AAAI 1次、ACL 2次、ACM 1次、IEEE 4次，整体演进缓慢。
- **调查关键数字**：总体必要性和均值2.95（95% CI [2.78, 3.13]）；研究设计阶段最高（mean≈3.5+），写作阶段最低（mean≈2.0-）；"Generate synthetic data sets"均值4.02（最高），"Edit a research paper for readability"均值1.8（最低）；低人类参与度显著提升必要性（β=0.49, p<0.001），高参与度降低（β=-0.44, p<0.001），但存在相位不对称性（如data分析阶段low vs. assumed差1分，idea generation阶段high vs. assumed差1分而low vs. assumed仅差0.5分）。
- **审计结果**：ICLR披露率64%（12,577条），EMNLP披露率40%（1,290条）；"Editing a research paper"披露比例ICLR 96.5%/EMNLP 81.4%，而"Generating synthetic data sets"仅ICLR 2.1%/EMNLP 2.5%；责任声明缺失严重（EMNLP 2%、ICLR 23% vs. 期望71%）；约50% ICLR声明包含"非AI使用任务"（least expected detail）；91%声明为1-2句简短形式（EMNLP尤为突出）；发现1个声明在95篇ICLR论文中逐字重复，疑似表演性披露。

## 相关工作脉络
- **Ganjavi et al. / Bhavsar et al. (2025)**：跨学科出版商AI政策文献计量分析，揭示政策异质性但缺乏CS领域细粒度任务分类；本文在此基础上量化任务级必要性，填补CS内部差异空白。
- **Resnik & Hosseini (2026)**：主张AI影响研究结果或生成内容时需披露；本文发现其标准与实证期望存在 mismatch（如"creating scientific figures"mean=3.0低于其预期，而"translating papers"mean=3.31被低估）。
- **Akpinar et al. (2026)**：证明披露影响信任度（人类撰写+LLM编辑获最高清晰度）；本文延伸揭示实践中的披露内容偏离期望（如过度披露写作辅助而忽略责任声明），为信任机制提供底层数据支撑。
- **Yusuf et al. (2025) (N=777)**：指出自我决定适当性与AI使用正常化是非披露主因；本文通过任务分级和模板设计降低披露摩擦，针对性缓解该障碍。
- **Andersen et al. (2025)**：提出跨学科AI使用感知差异；本文将其taxonomy适配CS场景（移除peer review相关任务、合并coding任务），并首次引入"人类参与水平"作为关键调节变量。

## 局限性与未来方向
- **样本偏差**：调查依赖purposive+snowball抽样，参与者可能已持有较强观点；未来需分层随机抽样（按CS子领域/会议邮件列表）验证普适性。
- **领域局限**：聚焦CS（AI工具最熟悉、规范演进最快），其他学科（如生物、医学）的披露期望可能迥异，需跨学科扩展。
- **LLM标注误差**：Gemini 2.5 Flash在rare details/task categories上可能存在残余错误（micro F1 96.5%/90.6%非完美）；需更大规模人工复核或交叉验证。
- **政策滞后性**：基于2025-2026年政策快照，AI能力快速演进可能导致现有政策迅速过时；建议建立动态修订机制。
- **伦理风险未完全解决**：披露语言辅助可能暴露非母语作者身份，引发审稿偏见；虽建议review期间 masking，但未评估实际效果。

## 研究启发与可借鉴点
1. **任务分级框架可直接迁移**：本研究构建的mandatory/recommended/optional三级分类（基于rating≥3.5/2.5-3.5/<2.5阈值）可复用于其他领域的AI使用规范制定（如法学、医学研究），只需重新校准任务清单与阈值。
2. **混合效应模型控制个体变异的方法学价值**：采用participant-level random intercept分离固定效应（task/human involvement）与噪声，适用于主观评分类调查，未来可推广至类似的用户期望研究。
3. **LLM辅助大尺度文本审计 pipeline**：两阶段提示工程（先提取结构再精准抽取）+ 人工验证F1>90%的流程，为政策合规性审计提供可复用技术方案，尤其适合需要处理大量非结构化声明的场景。
4. **人类参与水平作为关键调节变量**：首次实证"high/assumed/low"介入程度显著影响披露必要性（且存在相位不对称），提示后续研究应将"如何用时"而非仅"何时用"纳入政策设计。
5. **表演性披露的检测信号**：通过查重发现同一声明重复95次可作为合规疲劳指标；建议会议引入披露内容多样性检测或人工抽检机制，提升政策有效性。

## 关键术语表
- **AI Disclosure Statement**：作者声明其在研究/写作过程中使用生成式AI工具的具体说明，通常包含任务、模型、监督程度等细节。
- **Human Involvement Level**：调查定义的三种干预程度——High（作者主导、AI输出严格验证）、Assumed（基线假设参与）、Low（AI主导、 loosely supervised）。
- **Performative Disclosure**：为合规而非提供实质信息而重复/模板化的披露声明，本文发现95篇论文共享同一声明即为此类典型案例。
- **Linear Mixed Effects Model**：统计模型 $y = \beta_0 + \beta_{task}x_{task} + \beta_{human}x_{human} + u_{part} + \epsilon$，以参与者为随机截距控制个体差异，隔离任务与参与水平的固定效应。
- **Boilerplate Template**：标准化披露模板（Figure 5），强制包含task列表、human oversight描述、责任声明等7个期望维度，以减少政策模糊性。
- **Likert Scale Rating**：1-5分量表，1表示"无需披露"，5表示"必须披露"，用于量化21项研究任务的披露必要性感知。

## 可复现要素
- **数据集**：EMNLP 2025（3,216篇）与ICLR 2026（19,525篇）公开论文；调查数据与伦理审批信息见附录Table 15；**未声明公开存储库**。
- **代码/权重**：未开源（仅提及使用Gemini 2.5 Flash进行提取与标注）；提示模板见附录Table 16-18。
- **关键超参**：调查样本N=109；Likert 5点量表；必要性分级阈值≥3.5（mandatory）、2.5-3.5（recommended）、<2.5（optional）；LLM标注验证集N=100。
