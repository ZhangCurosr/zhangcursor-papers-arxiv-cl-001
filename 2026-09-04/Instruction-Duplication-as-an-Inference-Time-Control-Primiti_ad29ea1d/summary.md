---
title: "Instruction-Duplication-as-an-Inference-Time-Control-Primiti"
source: https://arxiv.org/pdf/2609.04024v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-07 11:04:35"
field: "大语言模型推理控制与轨迹编辑"
keywords: ["instruction duplication", "inference-time control", "trajectory editing", "protocol compliance", "black-box intervention", "Answer Engineering", "medical reasoning"]
innovations: ["定义指令重复为最小黑盒推理时控制原语，仅重复程序性指令即可改变机器可消费的显式轨迹状态", "通过2×2×2因子设计分离复制次数与放置位置，发现trailing duplicate为最强干预", "揭示协议状态改善与答案准确性不变之间的解离，并通过下游Answer Engineering实验验证其操作性价值"]
benchmarks: ["MedQA", "MedXpertQA", "AfriMed-QA"]
---

# 论文速读：Instruction-Duplication-as-an-Inference-Time-Control-Primitive

## 一句话总结
论文提出一种最小化的黑盒推理时控制原语——**指令重复（instruction duplication）**，仅重复程序性指令而不重训或改动解码，能显著提升模型在医学诊断任务中可被机器直接消费的显式轨迹状态（All-8诊断提升+2.95pp），同时保持最终答案准确性不变（60.21%）；并在下游轨迹编辑器Answer Engineering中验证了其操作性价值。

## 研究问题与动机
- **核心问题**：可控LLM系统往往不仅需要正确答案，还需要可检查、可修复的中间轨迹状态；现有方法多依赖重新训练或解码改造，成本高且改变模型行为模式。
- **动机1**：重复完整prompt（whole-prompt repetition）已被证明可提升非推理准确率，但干预对象是任务内容本身；本文旨在探究仅重复**程序性指令**能否改变下游控制器可感知的显式状态。
- **动机2**：医学诊断推理通常是一个迭代假设检验过程，需要模型暴露对比、暂定答案、决定性区分、 reconsideration 等显式状态，以便下游系统进行确定性编辑或修复。
- **动机3**：程序性状态合规（protocol compliance）与任务正确性（task accuracy）是独立的；一个模型可能给出正确答案但遗漏下游系统所需的中间状态，因此需要一种不依赖参数更新的轻量级控制手段。

## 核心贡献（创新点）
1. **定义指令重复为最小黑盒控制机制**：仅通过重复提示词中的程序性指令（如“使用八个标题并按顺序执行”），无需任何模型内部修改或重新训练即可干预生成轨迹。
2. **构建完整的2×2×2位置因子实验设计**：将指令复制次数（1 vs 2）与放置位置（system S、before question B、after question A）分离，得到8种条件，精确识别干预效果与位置交互。
3. **揭示协议状态与答案准确性的解离现象**：指令重复显著提升All-8诊断（+2.95pp）和TF–IDF召回（+1.38pp），但答案准确性完全不变（60.21% vs 60.21%），证明该干预专门改变显式轨迹状态而非整体推理能力。
4. **通过盲审挑战与下游AE实验验证操作性价值**：盲审显示多数机器检测到的状态变化对人是感知平局，但在Answer Engineering轨迹编辑器中，指令重复将SSNHL endpoint从84.2%提升至97.1%（+12.9pp），证明其对确定下游系统的实际意义。

## 方法详解
- **模型与数据**：使用7个指令微调模型（Gemma 3 12B, Llama 3.3 70B Instruct, Llama 4 Scout, Ministral 3 14B Instruct, Mistral Large 3, Qwen3 30B-A3B Instruct, Qwen3 235B-A22B Instruct），300道医学多选题（MedQA、MedXpertQA、AfriMed-QA各100题）。
- **程序性指令**：要求模型按顺序使用八个标题：(1) Facts; (2) Implications; (3) Provisional answer; (4) Best alternative; (5) Decisive distinction; (6) What would change the answer; (7) Reconsideration; (8) Final answer，并禁止在暂定答案部分前选择答案。
- **实验设计**：2×2×2因子设计，指令可出现在系统消息(S)、问题前(B)、问题后(A)，形成0/1/2/3份复制的8种条件，共7×300×8 = 16,800次生成，温度=0，固定种子。
- **评估指标**：
  - **All-8诊断**：响应通过所有八个可观察测试的比例，作为协议状态完整性的确定性度量。
  - **Pre-provisional TF–IDF recall**：在暂定答案之前生成的Facts和Implications部分中恢复的问题题干TF–IDF加权词汇内容比例。
  - **Premature commitment**：在暂定答案部分之前过早做出确定选择的比例。
  - **Accuracy (ITT)**：最终多选题答案的准确率。
- **统计方法**：Holm校正控制多重比较，10,000次问题聚类bootstrap置信区间，50,000次配对符号翻转检验；加和性检验通过从0/1复制单元拟合加和模型并检验残差。

## 实验与结果
- **主要结果**（1份 vs 2份指令复制）：
  - All-8诊断：90.22% → 93.17%（+2.95pp，消除30.2%的机器检测失败）
  - TF–IDF召回：73.44% → 74.81%（+1.38pp，Holm-adjusted p < .001）
  - 对比讨论完成：97.07% → 97.76%（+0.69pp）
  - 角色完成数：7.818 → 7.845（+0.027）
  - 提前承诺：1.52% → 2.30%（+0.77pp， adverse effect）
  - **准确率：60.21% → 60.21%（无变化）**
- **最强放置**：**trailing duplicate**（在问题后增加第二份指令）使TF–IDF召回从73.33%提升至75.53%（+2.19pp，p < .0001），All-8提升+3.31pp，效果在全部七个模型中一致为正。
- **盲审挑战**：30个机器正向转换中，20个为感知平局，10个非平局全部支持重复指令响应；未达预设的28/30确认标准，但说明人眼感知与机器可消费状态存在差异。
- **下游Answer Engineering实验**：
  - SSNHL治疗endpoint：无编辑25.1% → AE编辑84.2% → AE+重复指令97.1%（+12.9pp）
  - 传导性诊断分支保留：无编辑58.9% → AE 78.6% → AE+重复指令73.8%（低于系统-only AE但仍高于无编辑基线14.9pp）

## 相关工作脉络
1. **Prompt repetition (Leviathan et al., 2025)**：重复整个prompt提升非推理准确率；本文仅重复程序性指令，针对显式轨迹状态而非答案质量。
2. **Re-reading (Xu et al., 2023)**：重新阅读问题可提升推理；本文聚焦指令重复对协议状态的影响而非内容重曝。
3. **Instruction position (Liu et al., 2024)**：指令位置影响序列生成；本文通过因子设计分离复制次数与位置，并发现trailing duplicate最强。
4. **Chain-of-thought & process supervision**：CoT改变可见轨迹但未必忠实（Turpin et al., 2023）；本文是纯黑盒干预，不涉及内部激活或逐步验证。
5. **Grammar-constrained decoding & activation steering**：需修改解码或访问模型内部；本文仅在提示词层面添加token，完全黑盒。
6. **Clinical diagnostic reasoning (Kassirer, 1983; Charlin et al., 2000)**：提供领域动机，将诊断视为迭代假设检验，需要显式的对比、reconsideration等状态供机器消费。
7. **Answer Engineering (Lavrenko & Molodnitskaia, 2026)**：轨迹编辑器仅能编辑可观测的中间状态；本文证明指令重复可改善该可观测底物，从而提升下游编辑效果。

## 局限性与未来方向
- **领域局限**：仅测试医学多选题，未验证于其他领域或指令族。
- **指标饱和**：大多数协议指标在单次指令后已接近天花板（如7.94/8非平凡部分），限制了重复指令的提升空间。
- **TF–IDF局限**：仅衡量词汇暴露而非语义理解，可能遗漏同义词、 paraphrases、极性变化。
- **盲审局限**：单人非临床作者、 oversampled machine-positive cases，未达预设确认阈值。
- **因果中介未验证**：TF–IDF、All-8等指标与下游AE效果的因果中介关系未通过因子交叉实验证实。
- **长度影响**：重复指令增加预答案内容长度约7.5%，长度调整回归显示效果部分减弱但仍显著。
- **未来方向**：测试语义等价指令paraphrase的鲁棒性；设计冻结测量堆栈的confirmatory study；与下游控制器交叉因子实验以建立因果中介。

## 研究启发与可借鉴点
1. **控制参数解耦**：将指令复制次数与放置位置分离为独立控制参数，为轨迹控制器提供可微调的轻量级接口。
2. **协议状态与答案准确性的解离**：证明可通过黑盒干预专门改善机器可消费状态而不影响人类导向的准确率，为可修复系统架构提供新思路。
3. **Trailing duplicate作为强干预**：在问题后追加第二份指令是最有效的放置策略，可考虑作为下游编辑系统的标准配置。
4. **人机评价差异的量化**：盲审挑战显示机器检测到的许多状态变化对人眼是感知平局，提示需为不同消费者（人 vs 机器控制器）设计不同评估指标。
5. **结合Answer Engineering等轨迹编辑器**：指令重复可视为上游状态增强层，与下游确定性编辑器结合可显著提升端到端性能，未来可探索更多此类组合。

## 关键术语表
- **Instruction Duplication**：仅重复提示词中的程序性指令（如“使用八个标题”）而不重复实质性查询或完整prompt的黑盒推理时控制原语。
- **All-8 Diagnostic**：确定性协议状态度量，指响应完整包含所有八个请求标题且每个部分均有实质性内容。
- **TF–IDF Recall (Pre-provisional)**：在模型选择暂定答案之前，其生成的事实和含义部分中恢复的问题题干TF–IDF加权词汇比例。
- **Premature Commitment**：模型在到达暂定答案部分之前就做出确定答案选择的比例，反映过度自信或过早收敛。
- **Answer Engineering (AE)**：一种轨迹编辑器，仅能对模型生成轨迹中已显式出现的中间状态和决策进行局部机械编辑。
- **Contrastive Discussion**：协议指标，衡量模型是否完成了暂定答案、最佳替代方案、决定性区分、改变因素、 reconsideration 五个部分。
- **Factorial Placement**：将指令放置位置（system/before/after）与复制次数作为独立因子，通过2×2×2设计分离其主效应与交互效应。
- **Black-box Control Primitive**：无需访问模型内部、仅通过提示词修改即可实现的轻量级控制机制。

## 可复现要素
- **数据集**：MedQA (Jin et al., 2021), MedXpertQA (Zuo et al., 2025), AfriMed-QA (Nimo et al., 2025) – 公开基准，作者使用其中300题。
- **代码与权重**：代码与复现材料在GitHub公开：github.com/victorlavrenko/answer-engineering；源数据与冻结输出绑定至GitHub Release `instruction-duplication-arxiv-v1`，包含16,800单元格主运行、4,000响应AE重复运行及完整性哈希。
- **关键超参**：temperature=0，固定cell-specific种子，模型特定输出上限，10,000 bootstrap重采样，50,000 sign-flip检验，Holm校正。
