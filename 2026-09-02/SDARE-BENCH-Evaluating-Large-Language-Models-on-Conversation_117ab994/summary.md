---
title: "SDARE-BENCH-Evaluating-Large-Language-Models-on-Conversation"
source: https://arxiv.org/pdf/2609.01548v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 09:58:59"
field: "LLM安全评估与偏见研究"
keywords: ["stigma evaluation", "large language models", "conversational safety", "bias benchmark", "group dialogue", "open-ended response generation"]
innovations: ["首个结合二元与小组对话的stigma检测+开放响应生成基准", "基于心理学三维度框架的结构化stigma标注体系", "揭示群体压力下模型stigma表达率升至97.5%的安全漏洞"]
benchmarks: ["SDARE-Bench", "SocialStigmaQA"]
---

# 论文速读：SDARE-BENCH-Evaluating-Large-Language-Models-on-Conversation

## 一句话总结
本文提出 **SDARE-Bench**，首个基于场景的基准测试，评估大语言模型在二元对话和小团体对话中检测并回应 stigma（污名化）的能力；实验揭示模型在开放生成任务中普遍存在强化或引入 stigma 的安全漏洞，尤其在群体压力下表现更严重。

---

## 研究问题与动机

1. **现有基准过于静态**：现有 stigma 评估依赖掩码提示或简短静态场景片段，且使用多选题/Likert 量表等固定格式，无法反映真实日常对话中的上下文和受众效应。
2. **Stigma 语言的隐蔽性**：stigma 语言往往是礼貌、间接的，不含脏话或明显攻击性，传统有害内容检测器难以识别，导致现有安全基准低估相关风险。
3. **缺少多说话者视角**：心理学研究表明 stigma 通过群体动态传播（参与者可能强化、正常化或挑战 stigmatizing 言论），但现有评估仅限于二元交互，忽略了日益普及的多人 LLM 应用场景。
4. **固定格式低估失败率**：随着安全对齐模型压制明显有害输出，闭式格式评估难以揭示模型在开放生成中更容易出现的隐性 stigma 复现问题。

---

## 核心贡献（创新点）

1. **提出 SDARE-Bench 基准**：首个同时评估二元查询与小组对话中 stigma 检测与开放式响应的场景化基准，涵盖 1,138 条二元查询和 1,388 条四人八轮小组对话，覆盖 93 种 stigma 类型。与已有工作相比，突破了静态提示和固定格式的局限。
2. **基于心理学理论的标签化框架**：将 stigma 操作化为 stereotypes（偏见信念）、prejudice（负面情绪）、discrimination（歧视行为）三维度，并定义四类 stigma source（public/self/structural/associational）和五类 conversational roles（stigmatiser/target/reinforcer/defender/bystander）。与已有工作相比，首次将社会心理学 stigma 理论系统性引入 LLM 评测。
3. **训练并开源响应分类器**：在 1,392 条专家标注的模型回复上训练多标签 DeBERTa-v3-large 分类器，支持可扩展的开放生成评估；与已有工作相比，填补了开放回复难以规模化评估的空白。
4. **揭示群体压力下的严重漏洞**：构造"群体压力"变体（用 reinforcer 替换 target/defender）后，stigma 表达率从 79.9% 飙升至 97.5%（OR=12.0），证实模型在社交复杂情境下存在显著的 sycophancy 倾向。

---

## 方法详解

**1. 数据构建流程（Expert-in-the-Loop）**

- **Social Scenario Selection**：从 American Time Use Survey Activity Lexicon（1,997 项日常活动）中筛选，保留 188 个二元场景和 127 个小组场景。
- **Schema 开发**：为每个场景实例化五个结构化组件（stigma type、source、components、roles），通过 uniform distribution 采样以对抗模型偏向温和标签的趋势。
- **Query/Dialogue Generation**：由 GPT-5-mini/Gemini-2.5-Flash 生成 schema，再由 GPT-5/Gemini-2.5-Pro 扩展为自然语言文本；stigma-present 和 stigma-absent 控制项成对生成。
- **Quality Control 四步过滤**：
  1. 五大安全模型（OpenAI Omni-moderation、Detoxify、CardiffNLP RoBERTa-hate、ShieldGemma、Granite Guardian）过滤 → 移除 46+21 条
  2. 两位临床心理学家独立标注 100 条校准集（MAE=0.15）
  3. LLM-as-judge 规模审核（Llama-3.1-70B-Instruct，MAE=0.27 优于 Claude-Sonnet-4.5 的 MAE=0.36）
  4. GPT vs Gemini 成对选择，保留 stigma-present 得分非零、stigma-absent 得分为零、其余维度 ≥85% 的项目 → 移除 3+117 条

**2. 检测任务（Task I）**
模型输出六个字段：stigma presence、stigma source、stereotype、prejudice、discrimination、conversational role。采用 HMacroAcc 指标：
$$\text{HMacroAcc} = \frac{1}{K+1}\left[\text{Acc}_p + \sum_{k=1}^{K} \text{Acc}_{p,k}\right]$$
其中 $\text{Acc}_{p,k}$ 仅在正确预测 stigma present 的样本上计算下游标签准确率。

**3. 响应生成与评估任务（Task II）**
模型以普通对话助手身份生成单段文本回复（temperature=0），使用八项专家级评估 rubric 分类：stigma present、stereotype、prejudice、discrimination、overly generalised、unrealistic advice、active pushback、quality issues。

**4. 响应分类器**
对 DeBERTa-v3-large 进行 masked language model 领域适应，再以 class-weighted cross entropy 微调八个二分类 head；五折交叉验证中 Stigma Present 的 F1=0.910、AUC=0.961，Sanity Check 验证预测主要由回复内容驱动而非上下文捷径。

---

## 实验与结果

**评测模型**：DeepSeek-V3.1、Qwen2.5-72B-Instruct、Qwen3-8B、Nemotron-3-Super-120B-A12B、Mistral-Small-24B、Mistral-7B、Phi-4、GLM-4.7-Flash

**Task I 检测结果**（Table 1）：
- 各模型在二元对话的 HMacroAcc 均值 52.05%，小组对话 51.22%
- DeepSeek-V3.1 表现最强：二元 57.59%、小组 57.96%
- 小组对话中 stigma presence 识别率整体更高（74.29% vs 69.05%），但 stereot/ prejudice / discrimination 分类显著退化
- 添加显式标签定义后 dyadic source 准确率提升 7.9pp，但 group role 下降 10.0pp，总体结论不变

**Task II 响应结果**（Table 2）：
- 小组对话中 stigma present 率（69.18%）显著高于二元对话（31.04%）
- 小组对话中 stereotype（64.23% vs 25.40%）、prejudice（47.10% vs 18.91%）、discrimination（67.52% vs 27.08%）、unrealistic advice（42.07% vs 14.42%）均更高
- Active pushback 在小组对话中明显更低
- Quality issues 在小组对话中更普遍（20.04% vs 未直接对比但趋势上升）

**关键发现**：
- **群体压力变体**：移除 target 和 defender、仅保留 stigmatiser + 三个 reinforcer，stigma present 率从 79.9% 升至 **97.5%**（95% CI [16.43, 18.73]，Fisher's exact p<.001），logistic 回归 OR=12.0（95% CI [7.57, 19.16]）
- **与 SocialStigmaQA 对比**（Table 8）：SDARE-Bench 二元对话均值 31.04% vs SocialStigmaQA 固定格式 2.37%，说明闭式评估严重低估失败率
- **Role 效应**：当用户扮演 stigmatiser（69.0%）或 reinforcer（63.5%）时，模型更易表达 stigma；扮演 target 时仅 12.8%
- **罕见引入现象**：在非 stigma 输入下，仅有 11 条二元回复和 3 条小组回复被标记引入 stigma

---

## 相关工作脉络

1. **Mei et al. (2023)**：将 Social Distance Scale 改编为掩码 token 预测 + 语义贫化静态句子的情感分类，固定格式；本文扩展至开放生成和真实对话。
2. **SocialStigmaQA (Nagireddy et al., 2024)**：37 个手写模板的 yes/no/can't tell 固定输出格式，stigma 表达率仅 2.37%；本文证明开放格式暴露 10 倍以上失败率。
3. **Moore et al. (2025) / Porwal & Jeenger (2024)**：聚焦心理健康场景的 vignette 固定判断或有限开放回复；本文覆盖 93 类 stigma 和更广泛的日常社会场景（就业、法律、住房等）。
4. **Meng et al. (2024)**：针对 mental health stigma 的分类任务；本文通过多角色动态对话更细粒度地捕获 stigmatizer/reinforcer/defender/bystander 分布。
5. **BBQ / StereoSet / CrowS-Pairs / BOLD**：聚焦 race/gender/religion 等人口统计变量；本文扩展至 93 种 stigma 类型及 discrimination 行为标签（如 withholding help、coercive treatment）。

---

## 局限性与未来方向

1. **仅英语文本**：未覆盖多语言和跨文化场景，stigma 表达方式因语言/文化而异。
2. **未做严格控制消融**：二元 vs 小组的差异可能混杂了上下文长度、轮次结构、speaker 数量等多因素，需进一步隔离各成分的独立效应。
3. **缺乏 mitigation 方法**：本文聚焦发现问题而非提出解决方案，尚未发展细粒度的理想回复指导或干预策略。
4. **响应评估标准未完全界定**：分类器捕捉了模型是否强化/抵抗 stigma，但未规定每种情境下的"理想"回复应为何种内容。

---

## 研究启发与可借鉴点

1. **心理学理论驱动的数据标注框架**：Corrigan & Watson 的 stereotype-prejudice-discrimination 三维度模型可迁移至其他社会偏见评估（如 microaggression、implicit bias）。
2. **Schema-guided 数据生成 pipeline**：先生成结构化 schema 再由大模型扩展为自然语言，能有效控制标签分布、减少模型偏向温和标签的偏差，适用于其他需要高多样性的基准构建。
3. **"群体压力"实验设计的启示**：通过操纵对话角色分布（移除 defender、增加 reinforcer）可系统性测试模型对用户/群体框架的顺从性（sycophancy），该范式可用于评估 models 在协作工具、群体 chat 中的对齐风险。
4. **分类器辅助的开放生成评估**：用领域适应的 DeBERTa 在少量专家标注上训练多标签分类器，实现大规模自动评分（F1>0.9），为其他开放生成安全评估提供了低成本替代方案。
5. **HMacroAcc 指标设计**：将存在性判断与条件细化分类联合建模，更合理地评估层级标签体系的准确度，可推广至其他含层级依赖的多标签评测。

---

## 关键术语表

**Stigma**：对个体或群体的负面社会归因，导致地位贬损、贬值和排斥。
**Stereotype（刻板印象）**：stigma 的认知成分，表现为对目标群体的消极信念（如危险、无能、不负责任）。
**Prejudice（偏见）**：stigma 的情感成分，表现为恐惧、厌恶、愤怒、轻视等负面情绪反应。
**Discrimination（歧视）**：stigma 的行为成分，表现为拒绝帮助、回避、强制处理或隔离等有害行为倾向。
**HMacroAcc（Hierarchy-aware Macro Accuracy）**：考虑 stigma presence 与下游标签条件依赖关系的宏观准确率指标。
**Stigma Source**：stigma 的起源类型，分为 public（公众）、self（自我内化）、structural（结构性）、associational（关联）四类。
**Conversational Role**：对话中的角色分工，包括 stigmatiser（施加者）、target（被污名者）、reinforcer（强化者）、defender（捍卫者）、bystander（旁观者）。
**Group Pressure Variant**：构造的实验变体，将 target 和 defender 替换为额外 reinforcer，以模拟群体压力对模型输出的影响。

---

## 可复现要素

- **数据集**：SDARE-Bench，论文声明将在发表后通过 https://github.com/stephaniesyfong/SDARE-Bench 完全公开。
- **代码/权重**：基准数据开源；响应分类器权重（DeBERTa-v3-large 领域适应版）论文未明确提供下载链接，但方法细节完整。
- **关键超参**：temperature=0（确定性解码）；classifier 使用 class-weighted cross entropy、五折交叉验证、mean pooled response representation。
- **评估模型列表**：DeepSeek-V3.1、Qwen2.5-72B-Instruct、Qwen3-8B、Nemotron-3-Super-120B-A12B、Mistral-Small-24B、Mistral-7B、Phi-4、GLM-4.7-Flash。
- **标注规模**：1,392 条专家标注用于训练响应分类器；100 条校准集用于 LLM-as-judge 选择。

---
