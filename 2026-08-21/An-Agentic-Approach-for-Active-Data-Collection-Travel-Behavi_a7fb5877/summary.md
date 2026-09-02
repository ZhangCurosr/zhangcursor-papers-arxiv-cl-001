---
title: "An-Agentic-Approach-for-Active-Data-Collection-Travel-Behavi"
source: https://arxiv.org/pdf/2608.20320v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 02:04:20"
field: "出行行为建模与LLM预测"
keywords: ["Large Language Models", "mode choice", "travel behavior", "multi-agent system", "stated preference survey", "conversational agent", "weather-sensitive demand"]
innovations: ["三智能体端到端工作流整合对话式调查、数据处理与LLM预测", "图像增强SP调查与LLM视觉输入的联合评估", "系统比较Prompt框架与上下文配置对LLM出行选择预测的影响"]
benchmarks: ["Random Forest 69.6%", "Gemma 4:12B零样本69.9%", "Vision+FS3最佳71.5%"]
---

# 论文速读：An-Agentic-Approach-for-Active-Data-Collection-Travel-Behavi

## 一句话总结
本文提出并实现了一个三智能体（Data Collection Agent → Data Processing Agent → Data Modeling Agent）工作流，将对话式图像增强陈述偏好调查、结构化数据处理与传统离散选择模型/机器学习/LLM预测整合于同一可审计流程中；在92名McGill学生通勤者、5种天气场景的数据上验证，最佳LLM零样本五分类准确率达69.9%，与随机森林的69.6%基本持平，引入视觉输入后进一步提升至71.5%。

## 研究问题与动机
1. **数据收集与建模脱节**：现有出行行为研究常将数字数据采集与预测建模分阶段独立开发，缺乏端到端的可审计工作流。
2. **传统SP调查的信息贫乏**：陈述偏好（SP）调查多依赖固定文字描述天气等情境，受访者需自行构建上下文，难以嵌入更丰富的多模态信息（如视觉场景）。
3. **LLM出行预测的系统性评估缺失**：尽管LLM作为"合成受访者"在其他行为领域已有探索，但在出行选择预测中，Prompt策略、上下文丰富度、Persona、Few-shot及视觉输入等因素对LLM预测性能的影响尚未在同一任务上进行系统比较。
4. **多智能体框架在交通领域的落地空白**：跨数据收集、数据处理与行为建模的全流程多智能体工作流在交通研究中长期停留在概念层面，缺乏实证实现与评测。

## 核心贡献（创新点）
1. **端到端三智能体研究工作流**：将调查部署、数据处理与建模预测解耦为三个职责独立的Agent，并通过标准化数据对象与接口串联，区别于以往仅用单一LLM完成端到端任务的尝试。
2. **图像增强的对话式SP调查**：通过Voiceflow构建旅行行为调查Chatbot，向受访者呈现AI生成的5种天气场景照片，首次在该设定下将视觉上下文与SP任务结合。
3. **LLM出行预测的系统性Prompt评测**：在9个本地部署LLM（2B–35B参数）上，交叉评估Expert vs Role-Play框架与Base/Richer Context两组维度，并进一步扩展至Persona增强、Few-shot学习与视觉输入四种高级配置。
4. **视觉上下文用于LLM模式选择预测**：首次将受访者看到的同天气图像直接作为Vision-capable LLM的输入进行模式选择预测，最佳视觉配置达到71.5%五分类准确率。

## 方法详解

### 3.1 三Agent工作流架构
框架由三个专用Agent组成：
- **Data Collection Agent**：接收研究者定义的问卷规格 $\mathscr{Q}$，经构建程序 $\Gamma_{\mathrm{col}}$ 生成可部署Chatbot $\mathcal{C}$，收集原始响应 $\mathscr{D}^{\mathrm{raw}}$ 及部署包 $\mathscr{B}_{\mathrm{col}}$。
- **Data Processing Agent**：对原始数据进行去标识化、类型转换、缺失值处理、特征构造等，输出 $\mathscr{D}^{\mathrm{pro}}$ 及质量标志 $\mathscr{Z}_{\mathrm{pro}}$。
- **Data Modeling Agent**：基于 $\mathscr{D}^{\mathrm{pro}}$ 估计MNL（Biogeme）、Logistic Regression与Random Forest等传统模型，同时对九类LLM进行零样本/少样本/视觉配置预测。

信息流形式化表示为：
$$\mathscr{Q} \xrightarrow{\mathcal{A}_{\mathrm{col}}} (\mathscr{D}^{\mathrm{raw}}, \mathscr{B}_{\mathrm{col}}) \xrightarrow{\mathcal{A}_{\mathrm{pro}}} (\mathscr{D}^{\mathrm{pro}}, \mathscr{Z}_{\mathrm{pro}}) \xrightarrow{\mathcal{A}_{\mathrm{mod}}} (\widehat{\mathcal{M}}, \mathscr{R}_{\mathrm{mod}})$$

每个Agent以映射 $(\mathbf{x}_k, \mathbf{u}_k, \mathbf{s}_k; \Theta_k) \mapsto (\mathbf{y}_k, \mathbf{s}_k', \ell_k)$ 运行，通过策略函数 $\pi_k$ 选择工具并更新状态。

### 3.2 调查设计
- 92名McGill学生通勤者，覆盖5种天气场景：Sunny、Hot–humid、Rainy、Foggy/cold、Snowy。
- 每场景嵌入1张AI生成的写实天气图像，受访者从5种模式中单选：步行、骑行、公共交通、骑行+公交联合、驾车。
- 调查同时收集人口统计、出行习惯、季节性通勤模式、周期频率、影响因素重要性评分等。

### 3.3 传统建模
- **MNL**（Biogeme）：以Walking和Sunny为参照，含天气场景指标、移动资源、骑行经验、年龄、收入等。
- **Logistic Regression** 与 **Random Forest**：同一五折交叉验证（按被访者分组），确保同一个人不出现于训练和测试集。

### 3.4 LLM预测配置
LLM预测函数：$\widehat{y}_{is}^{(r)} = F_{\psi}[\rho(\mathbf{p}_i, \mathbf{z}_{is}, \mathbf{a}_{is}, \mathbf{u}_{\mathrm{mod}}); \epsilon_r]$，Empirical probability：$\widehat{P}_{isj}^{\mathrm{LLM}} = \frac{1}{R}\sum_{r=1}^{R}\mathbb{I}(\widehat{y}_{is}^{(r)} = j)$。

| 维度 | 配置 |
|---|---|
| Prompt框架 | Expert (EXP) vs Role-Play (RP) |
| 上下文丰富度 | Base Context (BC) vs Richer Context (RC, 增加习惯性出行信息) |
| 高级配置 | Persona增强、Few-shot（k=3/5/10等）、Vision输入 |

LLM候选：Gemma系列（2B–31B）、Llama 3.2:3B、Qwen 3.6:35B，共9个本地部署模型。

## 实验与结果

**数据集**：McGill大学92名学生通勤者，454条有效"受访者–场景"观测（6条因缺失/不可解析被剔除）。

**主要结果**：

| 模型/配置 | 五分类准确率 | 二元（active vs non-active）准确率 |
|---|---|---|
| MNL | 44.7% | 81.2% |
| Logistic Regression | 60.2% | 85.1% |
| Random Forest | **69.6%** | 88.8% |
| 最佳LLM零样本（Gemma 4:12B, EXP-RC） | **69.9%** | — |
| 最佳视觉配置（EXP-ImgFS3） | **71.5%** | — |
| 随机猜测基线 | 20.0% | 50.0% |

关键发现：
- RC（加入习惯性出行信息）相较BC带来最一致的预测提升：Gemma 3:4B 从41.3% → 64.2%。
- EXP框架普遍优于RP框架，尤其在较小模型中差异更显著。
- Few-shot提示在少量示例后即趋于稳定（约10个示例后增益饱和）。
- Snowy/Rainy场景更易预测（向公交集中），Sunny/Warm更异质。
- Bike+Transit是各类模型最难预测的选项。
- 视觉输入对不同模型效果不均：Qwen 3.6:35B在替换文本天气描述为图像后提升；Gemma 4:26B在视觉+三样本组合下收益更大。

## 相关工作脉络

1. **MNL与离散选择模型**（McFadden 1974; Ben-Akiva & Lerman 1985; Train 2009）：本文MNL作为可解释行为基准，与LLM预测形成互补对照——DCM提供系数解释，LLM无需拟合即可做出个体级预测。
2. **ML分类器在出行选择中的应用**（Hensher & Ton 2000; Ali & Fissha 2026）：本文Random Forest（69.6%）作为最强训练模型基准，证明LLM在零样本设定下可达到相近水平。
3. **对话式AI调查**（Yu et al. 2025; Ren et al. 2026; Krajcovic et al. 2026）：本文首次将图像增强SP任务嵌入Chatbot调查，填补了视觉多模态内容在交通SP调查中的空白。
4. **LLM作为合成受访者**（Horton 2023; Argyle et al. 2023; Liu et al. 2024, 2025a; Sameen et al. 2025）：本文在单一任务上系统比较多种Prompt配置，优于以往逐个实验的研究风格。
5. **多智能体系统框架**（Chu et al. 2025; Liu et al. 2025b, 2026; Wu et al. 2026）：本文是首个在交通领域实现"调查→处理→建模"全链路三Agent落地的实证研究。

## 局限性与未来方向

1. **样本代表性不足**：仅92名McGill学生，非总体代表性样本，结果外推需谨慎。
2. **陈述偏好而非显示偏好**：数据基于假设情境选择，未与实际天气条件下的真实出行进行比较验证。
3. **图像条件混淆**：每种天气仅用1张固定图片，天气条件与图像身份未分离识别。
4. **小规模与不平衡数据**：限制了小众模式和子群体的细致分析；小样本上多配置比较可能导致最优结果的偶然性。
5. **框架未与传统工作流对照实验**：多智能体工作流的贡献在于集成性、可追溯性和模块化，而非已测量的成本节约。
6. **未来方向**：更大/更多样化样本验证；扩大至其他出行目的和地点；控制图像的视觉实验；校准、子群分析与跨人群迁移；在严格验证后探索自适应调查与合成受访者。

## 研究启发与可借鉴点

1. **多Agent解耦设计可复用**：将数据收集、处理、建模拆分为独立Agent并通过标准化接口传递，降低了排查错误的难度，适用于多领域行为研究。
2. **习惯出行信息是LLM预测最强信号**：RC包含周期性出行模式后显著提升零样本准确率，提示其他领域（如健康行为、消费选择）亦应优先纳入历史行为特征作为Prompt上下文。
3. **专家框架优于角色扮演框架**：对较小模型尤为明显，说明"让LLM扮演专家推理"比"扮演被访者直接决策"在复杂任务上更稳定，可推广至一般预测场景。
4. **视觉输入对部分LLM有效但并非普适**：需针对性评估Vision-capable模型的适配性，避免对所有模型盲目添加视觉模态。
5. **以受访者为单位进行数据分割**：防止同一个人的信息泄漏到训练/测试集，这对纵向或重复测量设计具有普适意义。

## 关键术语表

**Stated Preference (SP) Survey**：陈述偏好调查，通过假设情境 eliciting 受访者在可控条件下做出的选择，弥补显示偏好数据无法覆盖极端/新型情境的不足。
**Multinomial Logit (MNL) Model**：多项Logit模型，基于随机效用理论的离散选择建模经典方法，系数具直接行为学解释。
**Zero-shot Prompting**：零样本提示，向LLM提供任务指令和上下文信息但不提供标注示例，依赖模型先验知识进行预测。
**Few-shot Prompting**：少样本提示，在Prompt中嵌入若干 labeled examples，引导LLM参考示例模式进行输出。
**Richer Context (RC)**：比Base Context多出习惯性出行信息的Prompt配置，包含受访者的季节性常规出行模式等。
**Persona-based Augmentation**：基于潜类分析提取行为Persona后注入Prompt，作为习惯性出行信息不可用时的替代信息源。
**Vision-capable LLM**：具备图像处理能力的LLM，可直接接收天气场景图片作为输入而非仅依赖文字描述。
**Multi-Agent Workflow**：由多个专业化Agent通过结构化数据接口串联完成端到端任务的协作框架。

## 可复现要素

- **数据集**：92名McGill学生通勤者SP调查数据，454条观测；论文未声明是否公开（仅描述为Phase 1样本）。
- **代码**：调查Chatbot基于Voiceflow构建，具体代码未声明开源；传统模型使用Biogeme（开源软件）；LLM本地部署运行。
- **关键超参**：9个LLM参数规模（2B–35B），prompt框架2×2设计，few-shot k值取3/5/10等，五折交叉验证（按受访者分组），重复次数R未明确给出。
- **开源情况**：论文未明确声明代码与数据公开。
