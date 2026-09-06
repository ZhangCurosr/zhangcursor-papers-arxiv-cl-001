---
title: "FROM-REWEIGHTING-TO-REWRITING-UNLOCKINGTHE-INTERVENTION-EFFE"
source: https://arxiv.org/pdf/2609.02771v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 20:59:50"
field: "训练数据归因与大语言模型干预"
keywords: ["Training Data Attribution", "Influence Functions", "Response Rewriting", "Epistemic Abstention", "Intervention Effects", "EK-FAC"]
innovations: ["提出影响引导的响应重写框架，解耦样本选择与干预设计", "揭示IF选择样本在重写干预下隐藏的强行为杠杆效应", "系统对比删除/加权/重写在相同IF选择集上的轨迹效果并刻画靶向性"]
benchmarks: ["Abstention-Bench", "WildGuard", "HarmBench", "XSTest", "WMDP"]
---

# 论文速读：FROM REWEIGHTING TO REWRITING: UNLOCKING THE INTERVENTION EFFECTS OF INFLUENTIAL SAMPLES IN TRAINING DATA ATTRIBUTION

## 一句话总结
本文提出**影响引导的响应重写**方法，将训练数据归因（TDA）中的样本选择与干预设计解耦：先用影响函数（IF）识别高杠杆样本，再通过替换其响应内容（而非调整权重）来实现定向行为干预。在四个开源LLM上的认知性拒答（epistemic abstention）实验中，响应重写产生了比传统重加权更强、更稳定且可逆的行为改变，揭示出IF选择样本中隐藏的干预价值。

## 研究问题与动机
- **核心问题**：TDA通过影响函数识别与特定行为相关的训练样本，但在现代LLM中，基于权重的干预（upweighting/deletion）对IF选择样本的效果与随机选择相差无几，引发疑问——是IF选择的样本本身缺乏干预价值，还是重加权策略未能释放其行为杠杆？
- **现有方法不足**：传统TDA干预聚焦于"是否该删改样本的权重"，未考虑"改变样本教给模型的内容"这一更直接的干预路径；此前工作（如Infusion）仅在视觉领域实现可靠的行为修改，在Transformer/LLM中扰动效果随训练延长而衰减。
- **动机**：在监督微调（SFT）设置下，可固定指令（instruction）直接重写响应（response），从"调整训练贡献强度"转向"修改训练信号内容"，从而分离样本选择质量与干预策略有效性两个因素。

## 核心贡献（创新点）
1. **影响引导的响应重写框架**：将TDA中的样本选择（IF评分排序）与干预设计（响应替换）解耦，首次系统比较删除、加权和响应重写在同一IF选择集上的效果。*与已有工作的本质区别*：不同于Infusion等仅依赖局部扰动的数据投毒，本文聚焦语义级响应重写并在真实LLM SFT训练轨迹中追踪效果演化。
2. **揭示IF选择样本的隐藏行为杠杆**：在OLMo2、Qwen3.5、Gemma3等四模型上证明，行为对齐/对抗的响应重写可实现强、持久、双向的拒答能力改变，而同样的样本经重加权后效果微弱且不一致。*与已有工作的本质区别*：反驳了"IF在LLM中无用"的结论，指出此前负面结果源于干预策略局限而非样本选择失效。
3. **刻画干预效用的来源与范围**：分析表明IF优先选择与目标行为（不可答性/安全性）相关的样本而非最极端内部表征样本；重写后行为改变集中在目标场景（answer unknown/false premise），且未导致通用能力退化。*与已有工作的本质区别*：超越单纯的"效果验证"，通过投影分析、跨模型排名重叠、定点曲率诊断揭示"为何重写有效"的机制。

## 方法详解
- **影响函数计算**：使用标准SFT response-only loss $\mathcal{L}(z_i, \theta) = -\sum_t \log p_\theta(y_{i,t}|x_i, y_{i,<t})$，目标函数 $f(\theta)$ 为hold-out查询集的均值响应log-likelihood。影响分数 $\mathcal{I}_f(z_i) = -\nabla_\theta f(\theta)^\top H_\theta^{-1} \nabla_\theta \mathcal{L}(z_i, \theta)$，用 **EK-FAC**（Eigenvalue-corrected Kronecker-Factored Approximate Curvature）近似逆Hessian，避免显式求逆全量曲率矩阵。
- **样本选择**：按 $\mathcal{I}_f$ 排序取 Top-K 为 $S_k^{\mathrm{helpful}}$（预期增强目标行为），Bottom-K 为 $S_k^{\mathrm{harmful}}$（预期抑制目标行为），默认干预预算 $k=1600$（占SFT数据集2.5%）。
- **三类干预设计**：
  - **重加权**：对选定集合 $\mathcal{S}$ 设权重 $w_i(\alpha)=\alpha$（$\alpha>1$ upweighting，$\alpha=0$ deletion），保留原始响应，仅改变训练贡献强度。
  - **行为对齐重写**：固定指令 $x_i$，将响应替换为 abstention/safety-aligned 模板（从80/100个多样本拒绝模板池中采样）。
  - **行为对抗重写**：固定指令，替换为鼓励回答/顺从的模板（20个comply模板）。
- **评估协议**：沿两个维度评估干预 $\mathcal{O}$ 在集合 $\mathcal{S}$ 上的效果 $\Delta(\mathcal{O},\mathcal{S}) = B(\theta_{\mathcal{O},\mathcal{S}}) - B(\theta_{\mathrm{base}})$：(1) **方向有效性**：是否沿预期方向移动目标行为；(2) **选择优势**：IF选择 vs. 匹配随机选择在相同干预下的相对收益。全程追踪SFT轨迹中各checkpoint的 $\Delta R_t$。

## 实验与结果
- **数据集与评估**：主要使用 **Abstention-Bench**（Kirichenko et al., 2026）的 answer unknown 与 false premise 场景（300 held-out 查询用于IF计算，1800 prompt 用于评估）；主指标为 abstention recall（不可答查询中被正确拒答的比例）。安全性泛化使用 9 个benchmark（WildGuard、HarmBench、XSTest等）。
- **基线模型**：**OLMo2-1B、Qwen3.5-2B、Gemma3-4B、OLMo2-7B**，均在 Tulu 3 SFT mixture 上继续训练（固定前64K样本、8×H200、1 epoch、lr $3\sim9\times10^{-5}$、max len 4096、bf16）。
- **主要结果**：
  - **响应重写vs重加权**：Across all four models，behavior-aligned rewriting 持续提升 abstention recall，behavior-opposed 显著降低，效果**强于随机选择且稳定**；而重加权（upweighting α=2 / deletion）效果不稳定，甚至出现反向效应（如图3所示varying α无剂量响应规律）。
  - **最强结果**：OLMo2-7B 上 harmful-aligned 重写使 recall 达 **0.6328**（baseline 0.4689，+16.4pp）；Qwen3.5-2B 上 harmful-aligned 达 **0.6321**（baseline 0.4754，+15.7pp）。
  - **选择优势**：IF选择显著优于 TRAK、梯度内积、probing等其他selector（Appendix C.2 Figure 8），且在干预预算 $k\in\{800,1600,3200\}$ 下重写效果单调缩放，而重加权无清晰剂量响应。
  - **靶向性**：重写增益集中于 attribution target 对应的 answer unknown / false premise 场景，非目标场景（underspecified context）无显著增益（Figure 10），且通用能力（GSM8K/HellaSwag/ARC）未系统性退化（Table 4）。
  - **安全拒答泛化**：相同范式在 OLMo2-7B 安全refusal上复现 qualitatively similar 模式（Table 2），但伴随 **over-refusal trade-off**（XSTest 得分下降），提示需更细粒度优化。
  - **跨模型迁移**：用 OLMo2-1B 排名为其他模型选样，仍显著优于随机（Figure 13），支持小模型排名向大模型迁移的可行性。

## 相关工作脉络
1. **Influence Functions for LLMs**（Grosse et al., 2023; Kou et al., 2025）：用IF追踪LLM行为/能力的训练起源；本文与其定位差异在于不局限于"归因解释"，而是将IF用于**主动干预**，并系统区分"选择质量"与"干预策略"的贡献。
2. **Data filtering/reweighting with IF**（Li et al., 2025; Lee et al., 2026a; Chen et al., 2026）：发现IF选择样本在权重干预下对LLM几乎无效；本文承接这一负面结论，但提出**重写**作为替代干预路径以重新挖掘IF选择价值，完成从"IF无用"到"IF+重写有效"的范式转换。
3. **Infusion**（Rosser et al., 2026）：同样用IF选样并扰动训练样本做 targeted data poisoning；但仅适用于视觉领域，且在LLM上扰动效果随训练衰减；本文的**语义级响应重写**在真实LLM SFT中产生稳定持久的效果。
4. **Abstention-aware instruction tuning**（Zhang et al., 2024; Yang et al., 2024; Zhu et al., 2025a,b）：通过构造/替换响应或梯度采样提升拒答能力；本文与之互补——从**训练数据归因视角**追溯拒答行为的单个SFT样本来源，并比较删除/加权/重写三种干预路径。
5. **TRAK / gradient-based attribution**（Park et al., 2023; Kowal et al., 2026）：另一类可扩展TDA方法；本文在Appendix C.2中与IF并列对比，证明**曲率感知的IF选择**在重写干预下取得最大且最持续的行为改变。
6. **Linear probing for unanswerability**（Lavi et al., 2026）：训练线性探针区分可答/不可答输入并提取内部方向；本文借用其方向做分析基线，但发现高投影样本在 aligned rewriting 下增益有限（因已含拒答信号），反衬IF选择更善于定位"有改写空间"的样本。

## 局限性与未来方向
- **行为改写适用范围受限**：当前框架最适合具有清晰"对齐/对抗"响应模板的行为（如拒答、安全拒绝）；对缺乏明确对立响应的行为（如推理能力、风格偏好）难以直接套用。
- **安全领域的 over-refusal trade-off**：在 safety refusal 实验中，aligned rewriting 显著提升拒答率但同时损害 XSTest 表现，提示粗粒度安全目标可能导致过度泛化，需更细粒度的子类别干预。
- **跨模型排名迁移的不确定性**：虽证明 OLMo2-1B 排名可迁移至更大/不同家族模型，但 Gemma3-4B 的 bottom-ranked 与 OLMo 家族 top-ranked 存在异常交叉重叠（Figure 12），反映预训练数据混合差异导致的 ranking 结构异质性，迁移条件的理论刻画仍是开放问题。
- **干预效果的非单调性**：Gemma3-4B 上 harmful-example rewriting 的最终效应并非最大（与其余三模型相反），显示模型架构/预训练数据的交互影响尚未完全理解。
- **长期训练下的效果衰减**：部分模型中 helpful-aligned 重写早期出现大跳跃后渐趋衰减，需在更长训练轨迹中观察持续性。

## 研究启发与可借鉴点
1. **"选择-干预解耦"的实验设计范式**：固定IF选择集，仅改变干预方式（删除/加权/重写）并对比，可有效分离 attribution quality 与 intervention strategy 的贡献——这一对照思路可直接复用于其他TDA应用场景（如数据清洗、隐私审计）的评估。
2. **响应重写作为可操作的数据干预工具**：在SFT/continued-pretraining中，通过**指令保留+响应替换**实现语义级干预，比权重调整更具行为操控力；可迁移至模型调试（debugging misbehaving predictions by rewriting influential examples）、红队测试（red-teaming via safety rewriting）等任务。
3. **训练轨迹中的 $\Delta R_t$ 追踪评估**：不只报告最终checkpoint，而是绘制与 baseline 的差值曲线并做移动平均，可捕捉干预效果的时序动态（早期突变 vs. 渐进积累），避免单次评估的误导性结论——建议作为后续干预类工作的标准评估协议。
4. **固定训练数据顺序以控制混杂变量**：所有干预条件使用相同的 shuffled dataset order（seed固定），排除 example presentation order 对结果的干扰；这一细节在复现对比实验时至关重要。
5. **跨模型排名迁移的性价比策略**：在大型模型上计算IF成本高昂时，可在小模型上预计算排名并迁移选样（本文Figure 13初步验证），为大规模TDA部署提供实用路径；可进一步探索迁移可靠性的自动检测指标。

## 关键术语表
- **Training Data Attribution (TDA)**：量化训练集中各个样本对模型最终行为/预测贡献程度的方法论体系。
- **Influence Functions (IF)**：基于微扰理论估计" infinitesimally reweight 单个训练样本"对模型参数及目标量的影响，常用于归因与干预规划。
- **EK-FAC**：Eigenvalue-corrected Kronecker-Factored Approximate Curvature，一种可扩展的Hessian逆近似方法，通过Kronecker因子化+特征值修正实现LLM规模下的曲率计算。
- **Epistemic Abstention**：模型在面对无法可靠回答的问题时主动选择不作答（而非编造答案）的认知性拒答能力。
- **Behavior-aligned / Behavior-opposed Rewriting**：分别指将选定样本的响应替换为鼓励目标行为或抑制目标行为的监督信号。
- **Directional Effectiveness vs. Selection Advantage**：前者衡量干预是否沿预期方向改变目标行为；后者衡量IF选择是否比随机选择在相同干预下产生更大收益。
- **Over-refusal Trade-off**：提升安全拒答能力的同时，对 benign 查询也出现过度拒绝，损害有用性。
- **Fixed-reference Influence Comparison**：在同一checkpoint的局部几何下比较原始响应与重写响应的 influence score 差异，避免跨checkpoint几何不对称。

## 可复现要素
- **数据集**：Tulu 3 SFT mixture（OLMo版本）；Abstention-Bench（18子集×100 prompt）；Safety benchmarks（WildGuard/HarmBench/XSTest/ToxiGen/WMDP/BBQ等）；Query set 300 abstention prompts（CoCoNot/SelfAware/KUQ）+ 300 safety prompts（HarmBench/WildJailbreak/WildGuard）。
- **是否公开**：论文声明"all templates are released with the code"；附录详细描述了template pool构建与query generation procedure（temperature=1.5, presence/frequency penalty=1.0）。
- **代码/权重**：代码随template一并发布（具体仓库URL见原文，未在主文列出）；基础模型使用开源权重（OLMo2-1B/7B、Qwen3.5-2B、Gemma3-4B）。
- **关键超参**：干预预算 k=1600（2.5%）；upweighting α=2；EK-FAC 使用 Kronfluence 包默认设置；训练 lr=3~9×10⁻⁵、batch 有效128、epoch=1、warmup=0.03、max_len=4096、bf16、8×H200 GPU。
