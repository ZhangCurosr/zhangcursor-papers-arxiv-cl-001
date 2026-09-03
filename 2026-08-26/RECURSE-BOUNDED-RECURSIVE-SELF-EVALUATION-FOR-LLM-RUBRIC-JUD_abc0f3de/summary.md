---
title: "RECURSE-BOUNDED-RECURSIVE-SELF-EVALUATION-FOR-LLM-RUBRIC-JUD"
source: https://arxiv.org/pdf/2608.24231v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 18:37:07"
field: "LLM评估与对齐"
keywords: ["LLM-as-judge", "Recursive Self-Improvement", "Rubric Evaluation", "Process Reward", "RL from AI Feedback", "Early Stopping Monitor", "Reward Hacking Mitigation"]
innovations: ["提出界面解耦（interface decoupling）切断judge-checker闭环中的表面token复制捷径", "构造理论支撑的PAV监控器，通过judge accuracy与checker fidelity加权实现可靠的有界早停", "在零外部gold RL reward条件下，实现judge能力的闭环有界递归自我改进与跨域泛化"]
benchmarks: ["HealthBench", "RubricBench", "CheckEval-Summ", "ProfBench", "SV-HARD", "SV-FULL"]
---

# 论文速读：RECURSE-BOUNDED-RECURSIVE-SELF-EVALUATION-FOR-LLM-RUBRIC-JUD

## 一句话总结
本文提出了RECURSE框架，通过有界递归自我改进（RSI）实现LLM裁判能力的自我提升：可训练的裁判在Pass 1生成推理与判决，同步的checker副本在Pass 2对该推理进行审计并产出标量过程奖励，两者共享参数并在每步同步；通过界面解耦关闭表面token复制捷径，并借助PAV（Pairwise Advantage Validity）监控器可靠定位最优早停窗口，最终在零外部金标准RL奖励的情况下实现了跨架构、跨规模的泛化提升。

## 研究问题与动机
- **现有裁判改进高度依赖外部监督**：LLM-as-judge提升裁判能力通常需要昂贵的人工标注、外部奖励模型或从更强教师模型蒸馏，成本高昂且形成循环依赖。
- **静态教师蒸馏在双变量耦合任务中失效**：与数学/代码不同，rubric裁判的difficulty同时受判据（discriminative rubric）和边界候选response两者耦合控制，固定训练语料会快速退化出“平凡样本”（明显正确或错误），不再产生有效修正梯度。
- **无锚定递归优化必然有界**：若完全去除外部锚点，递归优化在reward或ranking fidelity饱和后会引发分布外泛化退化，需可靠的早停机制防止过优化。
- **闭环自生成的奖励可能诱发表面捷径**：若judging与auditing共享相同YES/NO输出接口，策略梯度可直接利用token层面复制捷径（inflating YES tokens）虚增reward而无需真实改善判读能力。

## 核心贡献（创新点）
1. **首次将"有界递归自我改进（RSI）"形式化于LLM rubric judge场景**：设计judge–checker闭环，模型自身的判读能力既作为学习target又作为标量reward来源，零外部gold RL reward，与Self-Taught Evaluator / Meta-Rewarding / Grad2Reward 依赖合成偏好对、外部meta-judge或参考RM的本质区别在于切断训练奖励链中的外部锚点。
2. **发现并关闭surface coupling退化捷径（interface decoupling）**：指出"相同YES/NO读出口径 + 权重同步"会形成退化token复制通道（$b>0$），使reward被表面YES倾向（$B$）支配；通过将checker输出改为独立的5级标量Final Score（$s_i\in\{0,...,4\}$）切断token级直接复制，使reward增长转化为真实accuracy提升。
3. **提出理论支撑的PAV早停监控器**：基于pairwise score-difference误差三角不等式导出配对排名误差上界$2e_{C,t}$，构造复合指标$V_t = \frac{A_t + 2(1-e_{C,t})}{3}$，double-weight checker fidelity以反映pairwise比较的核心作用；在SV-HARD（仅100条人工核验prompt）上无偏估计、稳定定位有效性窗口，避免单纯依赖rule accuracy导致的"validation deception"。
4. **在三种架构/规模上验证bounded RSI可行并具跨域泛化**：Qwen3.5-9B（+12.9点SV-HARD）、Gemma-4-E4B-it（+5.2）、Qwen3.6-27B（+3.9）均在多套out-of-distribution benchmark（HealthBench、RubricBench、CheckEval-Summ、ProfBench）上获得一致增益，并证明由该judge生成的preference pairs可用于下游DPO对齐并进一步增益。

## 方法详解
- **Judge–Checker Recurrence（§3.1）**：输入为rubric实例$x=(h,y,r_{1:K})$。Pass 1：训练裁判$\pi_{\theta_t}$对$n$条rollout生成逐步推理与每条rubric的YES/NO判决。Pass 2：同步快照$C_{\bar{\theta}_t}$（$\bar{\theta}_t\leftarrow\theta_t$）依据meta-rubric对完整推理轨迹审计，产出独立标量过程奖励$s_{t,i}\in\{0,...,4\}$；参数更新$\theta_{t+1}=\text{PolicyUpdate}(\theta_t,\{z_{t,i},r_{t,i}=s_{t,i}\})$，随后$\bar{\theta}_{t+1}\leftarrow\theta_{t+1}$。Pass 2仅作reward-only审计、不反传梯度。
- **Interface Decoupling（§3.2）**：分解自产reward $S=aU+bB+\varepsilon$。当共用YES/NO读出时$b>0$导致退化捷径（inflated YES → reward→1.0，U停滞）；解耦后保留共享evidence layout（对话、响应、rubric三槽位）但checker输出升级为"5级标量Final Score"而非逐项YES/NO，切断token复制路径（$b\approx 0$）。绝对数值范围无关紧要，因group-relative advantage对正仿射变换不变。
- **Group-relative Optimization（§3.3）**：对合法format rollout计算组内均值$\mu_t$与标准差$\sigma_t$，得到$\hat{A}_{t,i}=(R_{t,i}-\mu_t)/\sigma_t$；采用clipped sequence-level policy loss（clip $\epsilon=0.005$）更新，无效format直接mask并从组统计中剔除。
- **PAV早停监控（§3.3）**：SV-HARD上评估judge rule accuracy $\hat{A}_t$与checker规范化误差$\hat{e}_{C,t}=\text{MAE}/4$，构造$V_t=\frac{\hat{A}_t+2(1-\hat{e}_{C,t})}{3}$。基于三角不等式知pairwise error被$2e_{C,t}$界定，故checker fidelity得2倍权重。使用prompt-bootstrap 95%置信区间评估不确定性，在有效性平台期选择checkpoint（Qwen-9B @130、Gemma @160）。

## 实验与结果
- **数据集/架构**：训练集来自RubricHub cluster-split，response由16模型三层池合成（Low:Mid:High=1:2:1）；零eval重叠。模型：Qwen3.5-9B（主实验）、Gemma-4-E4B-it（跨架构复现）、Qwen3.6-27B（扩展）。训练框架verl+FSDP+vLLM，lr $3\times10^{-6}$、batch 32/48、n=8、clip 0.005。
- **评测三阶梯**：SV-HARD（100 prompt，每prompt约2.5条hard规则，用于PAV）、SV-FULL（同100 prompt完整rule集）、out-of-distribution迁移（HealthBench 1459 prompt，RubricBench 2294，CheckEval-Summ 1600，ProfBench 120）。
- **主要数字**：
  - Qwen3.5-9B Best@130：SV-HARD R 73.7%（Base 60.8%，+12.9）；HealthBench 85.9%（+3.4）；RubricBench 79.1%（+3.1）；CheckEval $\rho$ 0.441（+0.019）；ProfBench 74.7%（+1.6）。Final@200 SV-HARD 75.9%看似更高但迁移全面退化。
  - Gemma-4-E4B-it Best@160：SV-HARD +5.2；HealthBench +3.7；CheckEval +0.024。
  - Qwen3.6-27B Best：SV-HARD +3.9，四套迁移均提升。
- **消融/RQ2-5**：
  - RQ2：共用YES/NO使YES率0.730→0.791、reward 0.465→0.698但accuracy停滞0.729→0.730；解耦后YES率稳定0.707→0.695、reward 0.484→0.656、accuracy 0.724→0.787同步上升。
  - RQ3：PAV在@130定位到$\hat{V}=0.742$（CI[0.685,0.797]），SV-FULL独立印证峰值（95.3%），fold-3稳定性验证稳定。
  - RQ4：Frozen checker SV-HARD 62.7%；External 27B 73.1%但HealthBench仅79.8%；Self-consistency坍塌至56.5%；Teacher SFT (6.4k) 64.2%、(17k) 70.2%，均落后于Main 73.7%。
  - RQ5下游DPO（Qwen-27B）：Base-judge GPQA 79.02（退化）；RECURSE-judge GPQA 81.46（+2.44 over base）、GuideBench 85.51（+1.63）、SOP-Maze 36.71（+2.37）。

## 相关工作脉络
- **LLM judges & rubric evaluation**：Zheng et al. 2023、JUDGE 2024、Evolm (Li et al. 2026a)、Dynamic-Rubric (Wang et al. 2026a)等；本文与之差异在于固定criteria并优化judge本身，而非共演rubric或仅做离线评测。
- **Self-improving evaluators**：Self-Taught Evaluator (Wang et al. 2024，合成偏好对)、Meta-Rewarding (Wu et al. 2025，外部frozen meta-judge)、SELF-JUDGE (Lee et al. 2024，execution-based)、Grad2Reward (Zhang et al. 2026b，reference RM)；本文核心定位：把这些外部anchor从RL reward中完全移除，仅靠compact holdout用于早停。
- **Process auditing / generative verification**：Lightman et al. 2024等需ground-truth训练process RM；本文process checker纯inference-time reward-only审计、无需单独SFT，且通过decoupling规避self-verification奖励通胀（Stechly et al. 2025; Simonds et al. 2025; Zhou 2026）。
- **RLHF框架**：verl (Sheng et al. 2025)、DeepSeekMath (Shao et al. 2024) group-relative advantage；本文继承并扩展至judge训练场景。
- **DPO downstream**：Rafailov et al. 2023；本文证明由本方法judge生成的preference pair能直接改善DPO训练效果。

## 局限性与未来方向
- **仅覆盖英文与rubric格式**：未扩展到multilingual、multimodal或非rubric（holistic unconstrained）打分；Appendix R指出原理可扩展但未经验证。
- **仍需小规模人工核验holdout**：SV-HARD 100条需人工逐条核验规则标签，虽远小于全量标注但仍是有成本的前置步骤；PAV在线sequential testing的多重检验问题未完全解决。
- **仅固定任务规格，不做open-ended self-modification**：模型不会自动修改optimizer、meta-rubric或check protocol，属"bounded"而非"autonomous" RSI。
- **后期checker退化模式仍在**：post-boundary仍出现score inflation、hallucinated meta-justification、verbosity drift等（Appendix P），需进一步robust设计。
- **模型规模上限27B**：未验证更大规模（如Qwen3.5-397B）下的scaling行为。

## 研究启发与可借鉴点
1. **Surface-coupling诊断范式可迁移**：凡涉及"模型自产奖励+同步权重更新"的闭环系统均需警惕token级复制捷径；提出$S=aU+bB+\varepsilon$分解框架，通过对比coupled vs decoupled输出接口即可检测。
2. **PAV指标构造思路适用于任何self-auditing RL**：当训练reward由checker实时产出时，用"目标准确度+checker校准保真度"加权组合代替单一accuracy，可避免validation deception；pairwise误差上界推导具有通用性。
3. **Group-relative advantage + format gate联合设计**：仅用组内相对rank即可消除标度依赖，配合严格format gate排除注入式作弊（judge伪造checker标签），是闭环reward design的良好模板。
4. **On-policy dual-variable coupling优于static distillation**：对于"判据×候选response"双变量耦合任务（rubric judging、安全合规判断等），静态teacher SFT会迅速耗尽边界样本；本文的on-policy持续生成+同步审计模式值得在其他双变量校准任务中复用。
5. **Holdout-only用于早停、零gold入reward**：可将SV-HARD式100条核验集作为"哨兵"嵌入任意RL微调流程中，以极低标注成本阻止过优化。

## 关键术语表
- **Bounded Recursive Self-Improvement (RSI)**：在固定任务与协议下，模型利用自身能力闭环地产生训练奖励以实现有限步内的自我提升，区别于自主修改结构/目标的open-ended RSI。
- **Interface Decoupling**：将judge的逐项YES/NO判读与checker的独立标量评分解耦，切断token级复制捷径，使reward增长转化为真实判读能力提升。
- **Pairwise Advantage Validity (PAV)**：基于judge rule accuracy与checker ranking fidelity双指标的复合早停监控，checker误差因pairwise advantage对误差被$\times 2$加权，用以定位有效性窗口。
- **Surface Coupling Shortcut**：当judge与checker共用同一YES/NO读出且参数同步时，策略梯度可直接放大YES token概率而不提升真实判读质量，使reward虚高。
- **Validation Deception**：仅监控同一风格validation accuracy会误判后期过优化为持续进步，因分布外泛化已退化；需引入checker fidelity等正交信号识别。
- **Group-Relative Advantage**：对同prompt下$n$条rollout的reward做组内Z-score归一化，使reward仅依赖relative ranking，对正仿射变换不变。
- **Rubric Instance**：形式$x=(h,y,r_{1:K})$，含对话上下文$h$、候选response $y$与$K$条预定义判据$r_k$。
- **Process Reward (Process Checking)**：checker对judge完整推理轨迹（而非最终判决本身）依据meta-rubric审计所得标量奖励，用于鼓励合理的推理过程。

## 可复现要素
- **数据集**：训练数据为RubricHub（cluster-split），候选response由16模型三层池合成；SV-HARD 100 prompt人工核验、SV-FULL 100 prompt LLM panel核验；迁移集HealthBench/RubricBench/CheckEval-Summ/ProfBench均为公开基准。
- **代码/权重**：论文未明确声明开源，使用verl+FSDP+vLLM训练框架；checker为vLLM快照每步同步。
- **关键超参**：lr $3\times10^{-6}$、batch 32（27B用48）、n=8 rollouts、clip 0.005、无KL（27B加KL 0.005）；checker 5-tier标量（0–4）；SV-HARD每prompt生成4条trace聚合（PAV计算）；每10步checkpoint一次，总步数Qwen-9B/Gemma 200步、27B 75步。
- **训练prompt/response长度**：8192/32768 tokens。
