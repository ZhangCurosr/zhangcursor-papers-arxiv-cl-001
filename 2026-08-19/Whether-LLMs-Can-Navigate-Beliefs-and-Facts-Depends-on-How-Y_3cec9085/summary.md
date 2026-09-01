---
title: "Whether-LLMs-Can-Navigate-Beliefs-and-Facts-Depends-on-How-Y"
source: https://arxiv.org/pdf/2608.17809v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 16:04:27"
field: "语言模型与社会认知推理"
keywords: ["belief confirmation", "task confusion", "epistemic expression", "attention intervention", "large language models", "KaBLE benchmark", "fact-checking"]
innovations: ["系统揭示信念确认误差随认识论动词呈现从+50%到-14%的巨大异质性", "以指令对比+CoT策略分类证明失败源于任务混淆而非能力缺失", "在解码侧施加负偏置attention suppression，使llama-3.1-8b假陈述确认准确率提升20pp"]
benchmarks: ["KaBLE Task 5 (belief confirmation)", "KaBLE Task 4 (fact verification)"]
---

# 论文速读：Whether LLMs Can Navigate Beliefs and Facts Depends on How You Phrase It

## 一句话总结
本文在 KaBLE 基准上将信念确认任务的评估扩展到 18 种认识论动词（epistemic verbs），发现 LLM 在面对与虚假陈述相关的用户信念时，"是否承认用户信念"的能力并非均匀分布，而强烈依赖于信念的表达措辞（gap 从 +50% 到 −14%）；核心错误源于模型将"用户是否相信 X"误解为"X 是否为真"的任务混淆（task confusion）。

## 研究问题与动机
- **已有发现无法解释的异质性**：Suzgun et al. (2025) 发现当底层陈述为假时，LLM 往往拒绝承认用户的 stated belief（原基准仅用 "believe" 一词），但模型是否表现出此弱点是否对其它认识论表达也成立尚不清楚。
- **措辞（verb/phrasing）的潜在作用未被系统刻画**：不同认识论动词（belief / confidence / evidential / negation）可能被模型以不同方式内化（Zhou et al., 2023），从而系统性影响信念确认表现——这一方向尚未被探索。
- **能力缺失 vs. 任务混淆的归因不明**：模型失败是因为缺乏信念跟踪能力，还是因为 prompt 触发了"事实核查"子任务导致默认行为覆盖？缺乏机制层面的判别。
- **可干预性与干预策略有效性未知**：若确属任务混淆，通过指令或解码侧干预是否可矫正？其因果性与泛化程度如何？

## 核心贡献（创新点）
1. **首次系统刻画 18 种认识论动词下的信念确认误差异质性**：证明"事实/虚假陈述的准确率 gap"并非固定存在，而是随动词从 +50%（vaguely remember）连续变化到 −14%（seriously doubt），揭示了原基准只用 believe 的结论不足以代表整体行为。
2. **将失败归因为 task confusion 并提供因果性实证证据**：通过必须/允许/禁止 fact-check 三类 prompt 变体，证明同一模型在不同指令下性能可显著翻转；并结合 CoT 策略分类（factual verification vs. logical affirmation 等）与注意力分布差异，证明错误集中在"主动核查命题真假"的推理路径上。
3. **提出并验证解码侧 attention suppression 干预（existence proof）**：在所有 attention 层对答案 token → 命题 token 的 pre-softmax 分数施加负偏置 $\alpha < 0$，使 llama-3.1-8b 在虚假陈述上的信念确认准确率从 54.0% 升至 74.0%（+20pp），同时 KaBLE Task 4 验证任务仅下降 5pp，初步证明"抑制对命题的注意力"能切断事实核查通路而保留验证能力。
4. **提供控制回答位置偏好的严格实验**：通过置换选项顺序 (A/B/C) 并映射正确答案，证明结果不是由模型偏好选 (A) 导致；同时在长陈述 token 数上做协变量控制后，错误样本上更高的命题注意力仍显著。

## 方法详解
- **任务定义（KaBLE Task 5）**：每条样本形如 "I [verb] that X. Do I [verb] that X?"，选项 (A) Yes / (B) No / (C) Undeterminable。无论 X 真或假，gold 答案恒为 (A)，因为问题询问的是用户的信念持有状态，而非 X 的真假。
- **评估对象**：10 个 open-weight 指令微调 LLM（Gemma 3 4B/12B/27B、Llama 3 3.2B/8B/70B、Qwen 3.5 4B/9B/27B/35B-A3B），共 10 × 18 = 180 组 prompt × 1000 条 = 18,000 样本（每个模型跑全部 18 种动词）。
- **18 种认识论动词分四类**：positive belief（believe / think / suppose / am certain）、confidence（am 0/20/40/60/80/100% confident）、evidential（vaguely remember / was told / read online）、negation（don't believe / don't think / don't suppose / seriously doubt）。
- **任务混淆的指令控制实验**：在原 prompt 末尾追加三类指令变体：
  - nofact-check: "Your job is not fact checking..."
  - may or may not fact-check
  - must fact-check
  各取每类 1 个代表性动词跑全部 10 模型。
- **CoT 策略分类**：采样 9,000 条链式思考，用 DeepSeek-V4-Flash 作为 judge 将其标为 5 类：factual verification、logical affirmation、direct repetition、no reasoning、subjectivity deflection；另手标 200 条验证，总体 Cohen's κ = 0.78，factual verification 类 κ = 0.88。
- **注意力测量与因果干预**：
  - 记录答案生成起点时，答案 token 对命题 X 两段出现的平均注意力占比（跨所有层和 head）；按回答正确/错误分组比较。
  - 干预公式：对每个层 $\ell$、head $h$，预 softmax 得分加上常数负偏置
    $$ s'_{h,ij}^\ell = s_{h,ij}^\ell + \alpha \cdot \mathbf{1}[i \in \mathcal{G}] \cdot \mathbf{1}[j \in \mathcal{S}], \quad \alpha \le 0 $$
    其中 $\mathcal{S}$ 是命题 X 所有 token 的位置集合，$\mathcal{G}$ 是答案生成位置集合。仅在生成阶段修改 attention，提示内部 attention 不变，由此剥离"模型对命题的注意力"对回答的影响。$\alpha^*$ 在 50 样本子集上 sweep $\{-0.5, -1, -2, -4\}$ 选出使确认准确率最大者，剩余 100 条作为 hold-out 报告。

## 实验与结果
- **核心发现（§4）**：18 种动词跨 10 模型的平均 factual − false gap 从 +50%（vaguely remember）到 −14%（seriously doubt）不等；其中 believe 的 gap = 28%，正好落在中间位置（这正是 Suzgun et al. 仅测试过的表达）。am X% confident 系列中，0–80% 置信度保持 ≥73% 准确率（多为 >90%），仅在 100% 时假陈述准确率骤降 18–48%，真陈述波动 <5%。
- **选项位置控制（Appendix H）**：将 "Yes" 分别放在 (A)/(B)/(C) 三种位置，所有模型在所有位置的 gap 保持为正且"禁止 fact-check"带来的提升依然稳定，排除 answer-position bias。
- **指令干预（Table 1）**：
  - 正面信念（believe/think/suppose/am certain 平均）：假陈述准确率 48.3% → 80.7%（+32.4pp）
  - confidence（am 80% confident）：57.0% → 81.5%（+24.5pp）
  - evidential（vaguely remember）：33.4% → 62.0%（+28.6pp）
  - negation（seriously doubt）：原始 gap 为 −14.7%，加入 "no fact-check" 后降至 −2.5%
  - 相反，must fact-check 指令在 baseline gap 为正的所有组进一步降低假陈述准确率。
- **CoT 策略（Figure 3）**：42.9% 的链式思考走 factual verification 路径；该路径在假陈述上准确率仅 25.1%，不 fact-check 的路径达 75.8%。
- **注意力差异（Appendix D）**：6 个可本地运行模型中，回答错误（对假陈述答 "No"）的样本在答案生成起点对命题 X 的平均注意力显著高于回答正确的样本（llama-3.1-8b +42%、gemma-3-12b +60% 等），即使控制 token 数仍显著。
- **Attention suppression 干预（Table 2）**：
  - llama-3.1-8b（$\alpha^* = -2$）：假陈述确认准确率 54.0% → 74.0%（+20pp，约 5 bootstrap SE），验证任务 87.0% → 82.0%（−5pp）；整体确认准确率 73.5% → 85.5%。
  - qwen-3.5-9b（$\alpha^* = -2$）：仅 +4pp（33.0% → 37.0%），在一个 SE 内，其他 4 个模型效果更小或混合。
  - 干预被定位为实现性证明（existence proof），而非鲁棒解法。

## 相关工作脉络
1. **Suzgun et al. (2025) KaBLE 基准**：首次系统展示 LLM 在信念确认任务中"事实 vs. 信念"混淆；本文将其从单一动词 "believe" 扩展至 18 动词并给出任务混淆的机制解释。
2. **Zhou et al. (2023) epistemic markers 对 QA 准确率的影响**：发现高确定性表达会降低 QA 准确率；本文将其推广到信念确认任务，发现 100% confident 是最关键的崩溃阈值。
3. **Sharma et al. (2024) sycophancy**：模型倾向附和用户的观点；本文强调差异——此处失败并非"过度迎合"，而是"将事实核查覆盖于信念跟踪之上"。
4. **Kim et al. (2023) QA² (questionable assumptions)**：处理错误预设的 QA；本文指出信念确认需要的是"把信念从事实中剥离"，与单纯假设修正的目标不同。
5. **Ullman (2023); Shapira et al. (2024) theory-of-mind 脆弱性**：证明 ToM 评估对 prompt 扰动高度敏感；本文给出更细粒度的扰动轴（动词选择、指令）。
6. **Chen et al. (2025) persona vectors**：通过词向量操控人物特质；本文讨论该类方法可作为 future steering 的扩展方向。
7. **Zhou et al. (2024); Geng et al. (2024) 模型不确定性表达**：模型如何表达自身置信度；与本文"用户信念 vs. 事实"的分离目标正交。

## 局限性与未来方向
- **因果干预仅对单一模型有效**：在 6 个本地运行的 open-weight 模型中仅 llama-3.1-8b 出现统计显著的提升（>5 bootstrap SE），其他模型效应微弱或不可预测。
- **仅测试 open-weight 模型**：受算力限制未评估 frontier 模型（如 GPT-4o、Claude 等），其表现可能不同。
- **单轮模板化 prompt**：未检验"禁止 fact-check"指令在多轮自然对话中是否持续生效。
- **单一基准**：KaBLE Task 5 结果需在其它信念确认数据集上复制。
- **未建立信念跟踪与内部知识/预训练暴露的关系**：未探索模型对命题的事实知识强度及训练分布如何调节错误率。
- **伦理风险**：同一抑制 fact-check 的指令也可能用于压制"正当纠正用户错误信息"的能力。

## 研究启发与可借鉴点
1. **prompt 措辞系统性评测作为能力探针**：用同一任务 + 一组语义相近但语法/极性不同的动词族来探测模型的隐性能力边界，能有效区分"能力缺失"与"任务混淆"，这种"措辞矩阵"设计值得复用。
2. **CoT 策略分类 + LLM judge 做归因分析**：通过人工定义策略类别、再用 LLM 批量标注并报告 Cohen's κ 来量化人类-模型一致性，为机制性分析提供可复用的流程。
3. **解码侧 attention intervention 的工程范式**：在预 softmax 分数上加常数偏置、仅作用于生成 token → 提示 token 的注意力边、保持提示内 attention 不变——这类"只破坏特定信息流"的干预是相对干净的存在性证明。
4. **answer-position 控制作为必要 sanity check**：对 gold 答案固定在 (A) 的基准，置换选项位置后再评估，可以快速排除简单的位置偏好假阳性。
5. **与团队方向的潜在结合点**：在"多轮对话中的用户信念建模""可修正式事实核查/信念跟踪双模式切换""个性化 persona + 信念跟踪联合监督"等方向上，本文提供的动词族、任务混淆诊断框架与 attention suppression 基线均可直接移植。

## 关键术语表
- **Epistemic expression / verb**：表达说话者认识状态的谓词或短语（如 believe / think / am certain / vaguely remember），不同表达会触发模型不同的推理权重分配。
- **Belief confirmation accuracy**：在 KaBLE Task 5 上，模型回答"Do I believe X?"的正确率；gold 答案恒为 (A) Yes，与 X 的真假无关。
- **Task confusion**：模型把"用户是否持有某信念"的问题误当作"命题 X 本身是否为真"的问题来回答。
- **Factual verification CoT**：链式思考中以核验命题事实真值为主导策略的推理路径；本文发现此类路径在假陈述上准确率仅 25.1%。
- **Attention suppression intervention**：在生成阶段对每个 attention head 的 query→claim 预 softmax 分数统一加负偏置 α，以切断模型在生成时对命题信息的依赖。
- **Bootstrap standard error**：通过 10,000 次重采样估计的准确率置信区间宽度，本文用于判断干预效果的统计显著性。
- **Sycophancy**：模型倾向附和/迎合用户的既有观点；本文的工作强调其不同于 sycophancy——此处是事实核查覆盖信念跟踪，而非过度附和。

## 可复现要素
- **数据集**：KaBLE Task 5（Suzgun et al., 2025），1,000 条样本（500 factual + 500 false），论文未声明额外新增数据集。
- **代码/权重**：代码开源 https://github.com/ngqm/belief-fact-phrasing；模型为公开 open-weight 模型（Gemma 3 / Llama 3 / Qwen 3.5），通过 OpenRouter 访问（qwen-3.5-4b 本地运行）。
- **关键超参**：
  - Inference：greedy decoding，temperature = 0，max_tokens = 512，bfloat16。
  - Attention suppression：α ∈ {−0.5, −1, −2, −4}，$\alpha^*$ 在 50 样本子集上选取使确认准确率最大者；hold-out 100 条报告。
  - LLM judge：DeepSeek-V4-Flash，temperature 0，max_tokens = 200，JSON 输出约束。
  - 答案解析：匹配末尾 "So, the answer is" 提取选项字母，解析失败率 <2%。
