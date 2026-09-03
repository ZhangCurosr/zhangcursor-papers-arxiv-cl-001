---
title: "SDoH-Aware-Narrative-Anchoring-Bias-in-Medical-LLMs-for-Trus"
source: https://arxiv.org/pdf/2608.22802v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 10:40:47"
field: "医疗大语言模型公平性与鲁棒性评测"
keywords: ["medical LLM", "SDoH", "narrative anchoring bias", "counterfactual evaluation", "clinical decision support", "trustworthy AI", "bias evaluation"]
innovations: ["形式化SDoH感知叙事锚定偏差为病例分组反事实评估问题", "提出CC/Correct CC/NSE三类病例级一致性指标及四类错误分解", "在Qwen2.5家族内控制规模变量并引入配对McNemar统计检验验证scaling对叙事鲁棒性的边际收益"]
benchmarks: ["NarrativeShield SDoH MedQA", "MultiMedQA", "MedQA"]
---

# 论文速读：SDoH-Aware-Narrative-Anchoring-Bias-in-Medical-LLMs-for-Trus

## 一句话总结
本文利用反事实医学问答数据集 NarrativeShield SDoH MedQA，系统评估医疗大语言模型在相同临床事实下因患者叙事风格不同而改变答案的"SDoH感知叙事锚定偏差"，发现模型规模扩大能提升准确率与一致性，但叙事敏感性依然存在，且简单的偏见感知提示无法有效缓解该问题。

## 研究问题与动机
- **核心问题**：医疗LLM在临床问答中是否会在临床事实和正确答案不变的情况下，因患者叙事风格（voice/patient narrative）的差异而给出不同答案？
- **现有评估盲区**：当前医疗LLM评测过度依赖MedQA等静态基准的平均准确率，无法捕捉模型在医学等价但叙事形式不同的患者描述下的回答稳定性。
- **SDoH相关风险**：既往工作多关注人口统计学属性或宏观社会因素导致的偏差，而本文聚焦于叙事形式本身（正式/不确定/焦虑/社会受限等）如何独立引发答案漂移。
- **临床安全关切**：若模型对叙事线索过度敏感，可能导致分诊建议、诊断提示或随访指导在不同患者表述下产生不一致输出，影响临床决策支持的可靠性。

## 核心贡献（创新点）
1. **形式化"SDoH感知叙事锚定偏差"为病例分组反事实评估问题**——将同一临床案例用三种person化叙事呈现且答案键固定，区别于既往仅关注人口统计属性偏差的工作。
2. **提出病例级别的一致性度量体系（CC、Correct CC、NSE）**——将评估从单一的persona级别准确率扩展到"稳定正确/稳定错误/叙事敏感/不稳定错误"四类病例分解，识别知识缺失与叙事干扰两类不同失败模式。
3. **系统性控制规模变量，验证Qwen2.5家族内 scaling 对叙事鲁棒性的边际收益**——在相同300病例、确定性解码和配对统计检验下，揭示1.5B→3B主要提升准确率，而3B→7B才同时改善准确率与反事实一致性。
4. **引入bootstrap置信区间与配对McNemar精确检验**——避免仅依赖点估计的结论，为医疗LLM反事实评估提供可复现的统计推断范式。

## 方法详解
- **数据集重塑**：NarrativeShield SDoH MedQA 原始为宽格式（一行含 case + persona_alpha/beta/gamma），重塑为长格式（每个 question_id 展开为三行，每行对应一种persona变体），使同病例的三行可分组对比。
- **采样与规模**：seed=42 随机采样300个临床病例，保留全部三种persona变体，共900行；三模型 × 2700响应 = 总计8100次模型推理。
- **三种提示条件**：
  - Zero shot："Choose the best option and return one letter with a brief explanation."
  - Few shot：前附两个已解示例。
  - Bias aware：额外指令"Focus on clinical evidence. Ignore writing style or patient background unless clinically relevant."
- **解码设置**：`do_sample=False` 确定性解码，4-bit量化（bitsandbytes），max_new_tokens=120，无效输出计为错误。
- **核心指标（公式）**：
  - 准确率：$Acc = \frac{1}{MK}\sum_{i,j}\mathbb{I}(\hat{y}_{i,j}=y_i)$
  - 反事实一致性（CC）：$CC = \frac{1}{M}\sum_i\mathbb{I}(\hat{y}_{i,1}=\hat{y}_{i,2}=\hat{y}_{i,3})$
  - 正确一致性（Correct CC）：$CorrectCC = \frac{1}{M}\sum_i\mathbb{I}(\forall j,\hat{y}_{i,j}=y_i)$
  - 叙事敏感性误差（NSE）：$NSE = \frac{1}{M}\sum_i\mathbb{I}(\exists a,b:\hat{y}_{i,a}=y_i \wedge \hat{y}_{i,b}\neq y_i)$
- **统计方法**：Bootstrap 95% CI（2000次重采样）+ 配对McNemar精确检验（同病例×persona×提示条件下比较模型/提示对的准确率差异）。

## 实验与结果
- **最佳单点结果**：Qwen2.5 7B + bias aware 提示，准确率 **56.22%**（95% CI: 52.89–59.56），CC **57.67%**，Correct CC **40.33%**，NSE **31.67%**（最低）。
- **规模效应**：零样本条件下准确率从1.5B的38.44% → 3B的48.78% → 7B的56.33%；Correct CC从1.5B的19.67% → 3B的27.33% → 7B的40.33%。
- **配对显著性**：McNemar检验显示7B在所有提示条件下显著优于3B（p < 0.0001量级）；3B显著优于1.5B。
- **提示效果**：bias aware提示与zero shot相比，无显著准确率提升（p=0.246/0.880/1.000，分别对应1.5B/3B/7B），亦无法消除NSE。
- **病例分解（7B + bias aware）**：稳定正确40.33%、稳定错误17.33%、叙事敏感31.67%、不稳定错误10.67%。即即便最强模型，仍有约三分之一的病例组存在"部分persona答对、部分答错"的叙事敏感性。
- **主要结论**：模型scaling能同时提升准确率和反事实一致性，但无法根除叙事锚定偏差；轻量偏见感知提示不足以作为可靠缓解策略。

## 相关工作脉络
1. **MedQA / MedMCQA / PubMedQA / MultiMedQA**：本文沿用的医疗QA评测传统，但指出这些基准无法捕捉叙事形式变化引起的答案漂移，本文的贡献在于补充"一致性"维度而非替代既有基准。
2. **BioBERT / GatorTron**：领域适配型生物医学模型证明了医学文本预训练的价值，但未涉及跨叙事变体的稳定性问题；本文聚焦instruction-tuned通用模型的临床稳健性。
3. **SDoH提取工作（Keloth et al., Guevara et al.）**：此类研究将SDoH视为可用于护理的有用信号；本文反向审视：当SDoH叙事线索不影响临床事实时，模型对其过度反应构成稳定性风险。
4. **Omiye et al.（种族医学偏见）/ Omar et al.（社会人口偏差）**：证明LLM会复现或受人口统计属性影响；本文定位差异在于控制人口统计信息不变，单独操控"叙事形式"变量，以更精细地隔离anchor效应。
5. **CSEDB（Wang et al.）/ JAMA综述（Bedi et al.）**：强调临床LLM需超越考试类题目的实用性评估；本文以反事实一致性为切入点，提供了可量化的"稳定性评测框架"。

## 局限性与未来方向
- **题目形式局限**：仅使用多选题，未覆盖开放式临床推理、治疗方案规划等更复杂的决策场景。
- **临床等价性未经独立验证**：三人格变体的临床等价性直接沿用数据集固定的答案键，未做医师独立审核，可能存在隐含临床信息差异。
- **模型范围有限**：仅评估Qwen2.5家族1.5B/3B/7B三个开源指令微调模型，结论不能外推至更大规模开源模型、闭源商业模型或医学专用模型。
- **提示干预较简单**：bias aware提示为轻量指令，强效缓解可能需要模型微调、校准、检索增强或医师审查等综合手段。
- **绝对准确率偏低**：7B最佳准确率仅约56%，模型尚不具备临床可用水平，本研究定位为研究性评估框架。
- **未来方向**：扩大至更大模型/专有系统、多语言患者叙事、开放式临床推理、医师审核配对变体、以及更强干预策略（训练/校准/检索守卫）。

## 研究启发与可借鉴点
1. **病例分组反事实评测范式可直接迁移**：将同一问题以多种表述变体成组评估，并报告"组内一致性"而非仅聚合准确率，适用于任何需要评估LLM对输入扰动鲁棒性的研究。
2. **NSE指标（部分正确部分错误）比CC更敏感**：CC要求全对或全错才判为一致，而NSE能捕捉"模型知道答案但表达不稳定"的关键失败模式，建议在后续工作中纳入。
3. **配对统计检验（McNemar + Bootstrap CI）应成为LLM评测标配**：本文在同一病例×persona×提示条件下做配对比较，避免了样本间异质性干扰，设计严谨，值得照搬。
4. **"bias aware"轻提示失效的发现具有警示意义**：提醒后续工作不应高估纯prompt工程对内在偏差的缓解能力，需在数据/训练/架构层面寻求根本解决。
5. **与团队方向的结合机会**：若团队关注公平性/鲁棒性评测，可将本框架推广至多语言、多模态或不同SDoH维度的交叉评估；若关注干预方法，可在本框架基础上测试RAG守卫、SFT去偏、输出校准等策略的减偏效果。

## 关键术语表
**SDoH（Social Determinants of Health）**：健康的社会决定因素，指影响健康状况的非医疗社会环境因素，如收入、教育、居住环境等。
**Narrative Anchoring Bias（叙事锚定偏差）**：模型在临床事实不变时，因患者叙事风格或表达方式的差异而改变答案的偏差现象。
**Counterfactual Consistency（CC）**：同一病例所有persona变体得到相同答案的比例，衡量跨表述的一致性。
**Correct Consistency（Correct CC）**：同一病例所有persona变体均答对的比例，比CC更严格。
**Narrative Sensitivity Error（NSE）**：至少一个persona答对而另一个答错的比例，反映"部分稳定"的脆弱性。
**Bias Aware Prompting（偏见感知提示）**：在提示中显式要求模型忽略非临床相关的写作风格或患者背景信息的prompt策略。
**McNemar Exact Test**：用于配对二分类数据的统计检验，本文用于比较同一样本上两个模型/提示的准确率差异是否显著。
**NarrativeShield SDoH MedQA**：HuggingFace上的公开数据集，提供含persona变体的医学多选题，答案键跨变体固定。

## 可复现要素
- **数据集**：NarrativeShield SDoH MedQA（HuggingFace公开，[17]）
- **代码**：论文未提供开源代码仓库链接；流程描述充分，可依据实验配置复现
- **模型权重**：Qwen2.5 1.5B/3B/7B Instruct（HuggingFace公开，[18]）
- **采样参数**：seed=42，300病例
- **超参**：deterministic decoding（do_sample=False），4-bit量化（bitsandbytes），max_new_tokens=120
- **硬件**：Google Colab Pro + NVIDIA L4 GPU
- **统计方法**：Bootstrap 95% CI（2000次重采样），配对McNemar精确检验
