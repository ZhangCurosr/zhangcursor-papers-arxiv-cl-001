---
title: "LLM-Judges-Verify-Presence-Not-Absence-Omission-Blindness-in"
source: https://arxiv.org/pdf/2608.31016v1.pdf
model: agnes-2.5-flash
chunks: 6
summarized_at: "2026-09-05 11:10:33"
---

# 论文速读：LLM-Judges-Verify-Presence-Not-Absence-Omission-Blindness-in

## 一句话总结
本文揭示LLM judge在验证AI临床笔记时存在结构性盲区：因Transformer注意力机制无法“关注到空缺”，omission（遗漏）检测性能接近随机，而commission（增补/篡改）检测则相对可靠。作者通过构建基于transcript blind extraction的高保真基准、系统测试18种judge配置与消融设计，提出并验证“先枚举事实清单、再逐条核对存在性”的enumerate-then-check pipeline修复路径。

## 研究问题与动机
1. **omission检测失效**：现有LLM judge对AI临床笔记的信息遗漏检测接近随机水平（AUC仅0.503–0.575），无法支撑实际质控部署。
2. **机制根源**：omission不留下任何可供引用的文本片段，transformer注意力机制本质上无法对“缺席内容”建立key-value关注，导致faithfulness-only框架天然盲区。
3. **基准公钥缺陷**：既有开源基准（PriMock57、ACI-Bench）均借用临床参考note构建，实测发现100%携带与原始transcript的材料性偏差（平均偏差7–10.7处，missing facts占比56%），无法作为可信ground truth。
4. **优化路径未知**：在明确结构性局限后，通过检查范围扩展、输出格式切换、多次采样、prompt wording调整与GEPA自动优化等常见手段能否突破噪声边界，缺乏系统实验验证。

## 核心贡献（创新点）
1. **构建首个transcript-blind提取基准**：摒弃临床参考note，直接从transcript盲提取事实并经3个model critic交叉审计，生成500对单错误note pairs（覆盖117次consultation），规避既有基准的材料性偏差。
2. **揭示commission/omission检测不对称性**：首次以控制实验量化LLM judge在增补/篡改与遗漏两类错误上的性能鸿沟，确立“存在性核对优于缺失检测”的评测范式。
3. **设计可复现的enumerate-then-check pipeline**：提出两阶段与三阶段审计流水线（Extraction→Check→Audit/Second Look），配套开源提示词模板、模型锁定文件与完整run manifests，实现高保真omission定位与severity分级。
4. **提供细粒度成本-效能基准**：公开18种judge配置的paired discrimination、single-note detection与false-alarm率，并拆解per-note-per-call定价（$0.005–$0.45），为后续部署研究提供经济性参照。

## 方法详解
- **基准构建协议**：从transcript blind extraction生成事实清单，经3个model critic审计；severity分级采用cross-family双model + 书面rubric（kappa 0.177→0.662），不一致时取较低grade；移除验证显示complete removal成功率56.3%，partial removal成功率75.5%。
- **Judge配置矩阵**：8种消融设计交叉因子为检查范围（faithfulness-only vs +completeness）、输出格式（yes/no vs 0-10 score）、采样次数（k1 vs k8 winsorised mean）；5种reference baseline（部署faithfulness / G-Eval / checklist / engineered / RAGAS-style）；2种pipeline架构（two-stage / three-stage）。
- **Pipeline架构**：
  - **Extraction Stage**：从transcript独立枚举事实与fact-site map，成本$0.0868/note（按consultation缓存）。
  - **Check Stage**：逐条核对note是否包含枚举事实；二阶段pipeline直接输出，三阶段pipeline增加后续Audit。
  - **Audit/Second Look Stage**：对已标记遗漏进行quote-verified second look，要求引文可验证；救援79/299个遗漏（26.4%），零不可验证引文被错误拒绝，但在note层面指标净零（互相抵消）。
- **评估协议**：
  - **Paired discrimination**：flawed note得分低于clean twin的概率（0.5=coin flip），需clean twin，属capability ceiling。
  - **Single-note detection & Usable detection**：部署指标，定义在≤10% false-alarm率下显著优于chance为usable。
  - **Answer cap**：monolithic judges与pipeline各stage统一1024-token限制，无reasoning effort开放。

## 实验与结果
- **数据集**：500对单错误note pairs（495用于evaluation），117次consultation（112次evaluation）；Omission half 293对（complete removal 150 / fragment trace 86 / restatement trace 57），Commission half 202对（addition 95 / alteration 107），Clean twins 112个。
- **主要结果**：
  - Commission paired discrimination（8 designs）：**0.792–0.944**（6种≥0.87）
  - Omission paired discrimination（8 designs）：**0.500–0.634**（最佳FC-score-k8）
  - Omission trace梯度（最佳design）：complete 0.690 / fragment 0.607 / restatement 0.526（chance）
  - Omission severity梯度（最佳design）：critical 0.683 / supporting 0.586 / peripheral 0.568
  - AUC（omission，8 designs）：**0.503–0.575**（threshold-free）
  - 最佳monolithic single-note（FC-score-k8）：8.3% detection @ 6.5% false alarms（差距1.02 SE，噪声内）
  - 最佳跨family结果：**Gemini-family completeness-scored设计**首次突破single-note噪声，达19.1–20.5% detection @ 0.9–3.6% FA
  - MEDEC外部commission锚点验证：paired 0.827 / AUC 0.811 / 51.3% detection @ 10% FA；同design在omission上AUC仅0.575
- **五种修复尝试均告失败**：扩展检查范围（F→FC）+0.036–0.082仍噪声；输出格式（bin→score）同步拉低detection/false alarm；投票（k1→k8）最小杠杆；prompt wording修改（删除“omissions are NOT errors”或添加affirmative instruction）未改善FA；GEPA三波优化均未通过held-out验证（exploratory winner 0.549无reasoning budget时性能差）。

## 相关工作脉络
1. **PriMock57 / ACI-Bench**：两者均基于临床参考note构建公钥，实测100%携带与transcript的材料性偏差（missing facts占56%），本文指出其无法作为omission检测的可靠ground truth。
2. **RAGAS / Faithfulness-only judges**：传统faithfulness检查仅验证note是否包含transcript支持的陈述，本文证明其天然忽略completeness维度，导致commission敏感而omission盲。
3. **GEPA与自动Prompt优化**：本文系统性测试多轮GE
