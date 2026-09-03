---
title: "STONIC-A-Layered-Measurement-Contract-for-LLM-Value-Profilin"
source: https://arxiv.org/pdf/2608.23411v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 00:59:14"
field: "LLM价值对齐与评估"
keywords: ["LLM value profiling", "Schwartz values", "cross-interface measurement", "behavioral consistency", "semantic scorer audit", "hidden-state decoding", "value alignment evaluation"]
innovations: ["提出STONIC分层测量合同，将评分/选择/生成分离为L1-L3四层独立接口并设报告门槛", "发现跨界面行为连续性（10/17配置L1→L2一致，所有符合条件配置C5偏好自身答案），但未支持跨界面语义等价性", "五视角冻结语义审计+隐藏状态一次前向探测，区分预测性与因果性"]
benchmarks: ["ValuePortrait", "AIRiskDilemmas", "DailyDilemmas", "MoralChoice", "FULCRA", "ValueLlama", "DeBERTa-19"]
---

# 论文速读：STONIC-A-Layered-Measurement-Contract-for-LLM-Value-Profilin

## 一句话总结
本文提出STONIC（Schwartz-Theory-Oriented Normative Instrumentation Contract），一种分层测量合同，通过四个独立接口（L0-L3 + L2*）分别在5,144个情境中测试LLM在评分、成对选择与自由生成三种模式下的价值一致性。核心发现：虽然模型在行为层面显示出可重复的连续性（如10/17配置评分能预测冲突选择，所有符合条件配置偏好自身答案），但跨界面的语义等价性未被支持——当前证据不足以建立跨接口的单一"价值身份"。

## 研究问题与动机
- **核心问题**：现有LLM价值研究常将问卷评分、成对选择与生成文本中推断的价值合并为一个统一画像，但这一合并假设三种观测描述的是同一稳定偏好，该假设是否成立？
- **现有方法不足**：
  1. 各接口（评分、选择、生成）存在提示敏感性和位置效应，不能直接互换或平均；
  2. 现有基准分别研究道德判断、提示评分、成对选择和生成文本，缺乏匹配同一物品的跨接口检验；
  3. 不同语义评分器（Scorer）对同一文本给出差异巨大的价值标签，缺乏可比性基准。

## 核心贡献（创新点）
1. **四层接口控制性设计**：首次在同一5,144情境数据集上，以严格匹配的L1评分、L2反平衡选择、L3自由回答、L2*（模型自答 vs 人工备选）四层接口测量，此前无工作实现全匹配跨接口比对。
2. **可报告的跨界面声称条件**：预设覆盖率≥80%、80个有效聚类、Holm校正、跨银行方向一致性等报告条件，明确界定何时一个跨接口结论可被报告，区别于以往"有数字就报告"的做法。
3. **35个配置的系统实验**：覆盖22个命名checkpoint的instruction/base对照实验，揭示10个跨银行支持的L1→L2一致性、所有符合条件的17个配置的自身答案偏好（中位效应+0.790），以及L1-L2语义传递最强、L2-L3次之、L1-L3最弱的梯度特征。
4. **五视角语义审计与隐藏状态探测**：同时评估ValueLlama、DeBERTa presence/segmentation、FULCRA五个冻结评分器在200 L3响应上的一致性，并通过Hidden-State审计证明生成后状态比纯prompt更能解码价值表征。

## 方法详解
- **四层接口设计**：
  - **L0**：40项PVQ-40描述性问卷锚点（独立银行，不与场景项对应）。
  - **L1**：每个备选回答独立展示，请求六档 Likert 式赞同短语（not like me at all → very much like me）。
  - **L2**：每对备选以A/B和B/A两种顺序呈现，仅接受结构化选择（`{"choice": "A"}` 或 `{"choice": "B"}`）。
  - **L3**：不给备选，请求简洁自由回答（why + what to do）。
  - **L2\***：将L3模型自答与每个人工备选以正反序配对比较（类似L2格式）。
- **统计检验**：
  - **C1（评分→选择一致性）**：$\widehat{\Delta}_{C1} = \frac{1}{4}\sum_{b=1}^{4}\left(\frac{1}{N_b}\sum_{i \in b} s_{bi} - 0.5\right)$，$s=1$选更高分、$s=0$选更低分、$s=0.5$平局。
  - **C2（跨界面语义转移）**：同情境余弦相似度减去同银行随机配对基线，评估跨接口价值向量一致性。
  - **C5（自身答案偏好）**：稳定偏好自答+1，稳定偏好人工备选-1，顺序不一致记0，均值反映偏好强度。
  - **E2（位置敏感性）**：$P(\text{choose A}) - 0.5$。
- **报告门槛**：每个单元格需 ≥80%有效覆盖率 + ≥80有效聚类 + Holm校正（FWER=0.05）+ 至少3/4银行同向 + 留一银行检验无符号反转。
- **隐藏状态审计**：每次前向传播记录每个decoder block在4个位置（prompt_end, decision_first, decision_mean, decision_last）的隐状态，用FP32累积后存FP16，经CountSketch（256维）降至低维后用ridge回归探测Spearman/balanced accuracy。

## 实验与结果
- **数据集**：4个场景银行（ValuePortrait 104项、AIRiskDilemmas 3000项、DailyDilemmas 1360项、MoralChoice 680项，合计5144情境），共享同一情境脊柱。
- **模型**：35个固定配置（20 instruction + 15 base），涵盖Gemma、Granite、Llama、Mistral/Ministral、Qwen、SOLAR等22个checkpoint。
- **主要结果**：
  - **L1→L2一致性**：17个配置可估计，10个跨银行通过（中位+0.228），均为instruction调优模型。
  - **跨界面语义转移（C2）**：数值估计均为正（L1-L2中位ρ=0.73，L1-L3=0.43，L2-L3=0.50），但**无一配置**在全部4银行达到80%语义覆盖率门槛，故跨界面语义等价性未被证实。
  - **自身答案偏好（C5）**：17个符合条件配置全部偏好自身答案，范围0.508-0.932，中位+0.790——这是最强的跨界面行为结果。
  - **位置敏感性（E2）**：18个资格单元格全部显著，中位+0.0218，范围[-0.1136, +0.1572]，十偏A方向、八偏B方向。
  - **五评分器对比**：FULCRA与人类多数标注最接近（AUROC=0.893，AP=0.809）；DeBERTa保留有用排序信息（AUROC=0.784，AP=0.640）；ValueLlama覆盖仅53.1%（近半数L3答案为精确零向量）。
  - **Top-3探索性排名**：Qwen3.6 27B Instruct (53.11)、Qwen3.6 35B-A3B Instruct (51.20)、Qwen2.5 Coder 14B Instruct (42.62)。
  - **隐藏状态**：post-output位置比prompt_end更容易解码（L1中位提升+0.052，L2\*提升+0.112）；L2（冲突选择）跨银行探针转移为负（中位-0.121），表明冲突表征更依赖特定银行。

## 相关工作脉络
1. **Schwartz价值理论**：STONIC沿用Ten-Value框架作为可解释报告空间，但强调"识别某价值≠偏好该价值"，将Schwartz视为测量维度而非道德等级。
2. **Hendrycks et al. (2021) MMLU/BIG-bench道德判断基准**：独立研究道德判断，无跨接口匹配测试；STONIC将rating/choice/generation统一至相同情境作直接对照。
3. **Han et al. (2025) ValuePortrait**：将情境响应与人类心理测量档案关联；STONIC沿用其104项情境但剥离单一画像假设，改为分层报告。
4. **Yao et al. (2024a) FULCRA / 2024b CLAVE**：分别用生成式回归和参考无关评估预测价值轮廓；STONIC以五视角冻结比较凸显 scorer-dependent 差异，反对任何单一视角作为ground truth。
5. **Liu et al. (2025) INVP**：通过社会场景中决策调查价值优先级；STONIC的同物匹配设计使其成为INVP的自然延伸，可直接复用以测试跨接口稳定性。
6. **Mitchell et al. (2019) Model Cards / Gebru et al. (2021) Datasheets**：STONIC继承其"条件与局限性优先"精神，将覆盖度、顺序敏感性、scorer身份等元数据纳入"measurement card"而非单一数字排行榜。

## 局限性与未来方向
- **数据覆盖**：仅4个场景银行，不代表所有文化/语言/部署场景；银行转换保留了来源特征但也引入了共同的提取shell。
- **严格解析导致缺失非随机**：Base模型因输出格式不符产生高缺失率，限制跨mode直接比较；缺失未被填为中性证据。
- **Schwartz-10非穷尽本体**：作为可解释报告空间而非完备价值分类体系。
- **任务本地人类验证仅200项**：用于校准语义审计，不能确立任何scorer为通用ground truth。
- **隐藏状态审计只证明预测性，不证明因果机制**：post-output可解码性高并不意味着内部表征即为价值本身。
- **未来方向**：可扩展至更多文化与语言银行；改进语义覆盖（如多scorer投票或校准）；探究instruction tuning对跨界面一致性的因果效应（当前matched family未通过Holm校正）；将分层合同迁移至其他模型能力测量（如推理一致性、工具使用可靠性）。

## 研究启发与可借鉴点
1. **分层合同设计思路可直接复用**：将同一情境经多接口（prompt/choice/generation）提取，并以报告门槛区分"数值估计"与"可支持声称"的做法，可迁移至模型对齐、可信度、工具使用等多维评估。
2. **跨界面语义转移检验框架**：用余弦减去银行内随机配对基线的方法（Eq.2），可用于衡量任意两类LLM输出之间价值分布的一致性，而不依赖单一scorer。
3. **隐藏状态审计协议**：单次前向传播记录四个逻辑位置、FP32累积+FP16存储、CountSketch降维+ridge probing的流程可作为可复现的标准组件。
4. **与团队方向的结合机会**：可将STONIC的分层设计用于团队关注的"模型价值观漂移检测"（monitoring value drift across prompt variants or fine-tuning stages），或以C5自身答案偏好指标评估RLHF/RLAIF过程中偏好保真度。

## 关键术语表
- **STONIC（Schwartz-Theory-Oriented Normative Instrumentation Contract）**：一种分层测量合同，将LLM价值证据按接口类型分离并设置报告门槛，而非合并为单一画像。
- **L1/L2/L3/L2\***：四层接口缩写，分别对应独立评分、反平衡成对选择、自由回答、模型自答vs人工备选的二次比较。
- **跨界面声称（Cross-interface claim）**：仅在匹配物品、覆盖率和稳定性均满足时才被报告的结论，避免跨接口随意平均。
- **行为连续性（Behavioral continuity）**：模型在独立评分、冲突选择、自发回答中表现出的可重复模式，但不等于跨接口价值身份等价。
- **C5（Own-answer preference）**：模型在L2*中对自己L3生成的答案的偏好强度，中位+0.790为最强跨界面行为结果。
- **E2（Order sensitivity）**：选项呈现顺序对选择率的影响，所有合格单元格均显著，提醒单一顺序结果不可靠。
- **C2（Same-situation semantic transfer）**：跨接口价值向量余弦相似度减去银行内随机配对基线，衡量语义可转移性。
- **Measurement Card**：STONIC推荐的产出格式，包含各接口覆盖度、位置敏感性、scorer身份与对应声称边界，替代单一分数。

## 可复现要素
- **数据集**：四个场景银行（ValuePortrait、AIRiskDilemmas、Daily-Dilemmas、MoralChoice），总5,144情境；文章提供了request artifact的SHA-256哈希，但未提供原始数据下载链接，需从各源（Han et al. 2025; Chiu et al. 2025/2026; Scherrer et al. 2023）获取。
- **代码/权重**：论文未开源代码库；模型权重为各checkpoint公开权重的调用（非自行训练）；vLLM 0.19.1固化容器。
- **关键超参**：temperature=0，top-p=0.95，top-k=-1，seed=13，单sample；L0≤32 tokens，L1≤16，L2/L2*≤128，L3≤512；max context=2048，eager execution，无prefix caching。
- **统计细节**：20,000 within-bank permutations（C2）；10,000 item-cluster bootstrap（CI）；Holm校正（FWER=0.05）用于model-cell family；BH校正（q=0.05）用于exploratory matrix。
