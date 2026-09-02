---
title: "CREDIT-WITHOUT-GROUND-TRUTH-AUDITING-STEP-LEVEL-CREDIT-ASSIG"
source: https://arxiv.org/pdf/2608.19760v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-09-02 00:03:36"
---

# 论文速读：CREDIT-WITHOUT-GROUND-TRUTH-AUDITING-STEP-LEVEL-CREDIT-ASSIG

## 一句话总结
本文在无人工标注的条件下，以**Executed Replay**作为因果真值代理，对LLM Agent训练中广泛使用的步骤级信用信号（Implicit与Judge两大家族）进行了预注册的多臂对照审计，发现所有现有信用信号均处于**机会水平（placebo-level）**，其中Implicit信用实质仅为模型流畅度的回声，并不包含因果贡献信息。

## 研究问题与动机
- **核心问题**：LLM Agent训练所用的步骤级信用信号能否真正识别对该步结果具有因果贡献的关键步骤？
- **现有方法的盲区**：主流工作依赖“步骤正确性标注”（annotated step correctness）或反事实 advantage 打分，但**正确性≠因果贡献**——正确的冗余步骤因果增量为零，而错误的步骤反而可能触发关键状态转移。
- **审计需求**：缺乏一个不依赖人工标签、可在单Agent交互环境中直接验证信用信号因果保真度的标准化测试床。

## 核心贡献（创新点）
1. **提出 Executed Replay 因果基真构建协议**：通过重采样策略支持的替代动作并多次前向rollout，以事实与反事实结果的均值差量化步骤因果增量，无需人类标注即可搭建审计基准。
2. **建立预注册七臂对照审计框架**：在双重模型族（Qwen2.5-7B / Llama-3.1-8B）与校正仪器下，共享三重检验（秩忠义vs shuffle控制、逐步符号一致性vs机会、控制fluency后的部分相关），冻结阈值与排除规则，确保结论可复现。
3. **揭示 Implicit 信用的 fluency 混淆机制**：证明 implicit 信用与 fluency 的中位数秩相关高达 +0.752，且在条件化 outcome 后偏相关跌至 −0.004（p=0.87），说明其本质是语言流畅度的回声而非因果信号。
4. **提供信用审计的方法论规范**：指出比较不同 credit 规则时必须匹配 effective sample size，否则测量的仅是训练剂量（optimizer steps）而非 credit 信息内容；同时暴露 outcome-only 训练在选定 substrate 下的发散缺陷。

## 方法详解
- **环境与被审计模型**：ALFWorld 可重放单Agent工具环境；策略模型 Qwen2.5-7B-Instruct（主测）与 Llama-3.1-8B（交叉族校验）。
- **Executed Replay 因果基真**：在每个决策点从策略支持中采样 K=4 个替代动作，各 forward rollout ≥3 次，计算：
  $$A_{\text{replay}}(t) = \mathbb{E}[\text{outcome} \mid \text{factual replays}] - \mathbb{E}[\text{outcome} \mid \text{alternative rollouts}]$$
  噪声基线 $\sigma_{\text{floor}}(t)$ 取事实回放的结果标准差；排除无法采到4个不同替代动作的 turn（非填补）。
- **被审计信用信号家族**：
  1. **Implicit**：HCAPO 条件化 token log-prob ratio（$\rho_t$），由策略自身打分。
  2. **Judge**：Qwen2.5-7B 作为 Task Execution Evaluation Judge 输出 1/-1/0 评分。
  3. **Confidence**：策略自身置信度，用于成本路由实验。
- **三重审计协议**：① 秩忠义（Spearman中位数）与边际匹配 shuffle 控制带比较；② 逐步骤符号一致性 vs 随机机会；③ 控制 fluency 后的部分相关（Pearson/Mantel）。所有阈值、排除规则、判定顺序在数据收集前预注册冻结。
- **评估量表（TARL）**：五分量表（5=必要/高效，4=有用非必要，3=中性，2=浪费，1=有害），强制使用全范围；Judge 三分制变体聚焦写操作（take/put/open/close/clean/heat/cool/toggle）的偏差归因。
- **成本路由规则**：低置信度 turn 路由至 judge，记录 pivotal turn 回收率与总路由比例。
- **诊断层（PC/PC2/PC3）**：PC2 通过 ∆ addenda 携带扰动强度与位置单调性，输出 Tier-1 ledger rows；PC3 边界表用于验证 v1.4 stabiliser family 在 outcome-only relative-advantage 信号下的稳定性。

## 实验与结果
- **数据集与设置**：ALFWorld，50条轨迹（temperature=0.7，HCAPO模板，历史长度=2），128个 held-out 任务，10,000次 bootstrap 估计 Spearman 中位数。
- **因果稀疏度特征**：
  - Qwen 下仅 **30.5%** complete turns 为 pivotal（n=1,768），**69.5%** 的 $A_{\text{replay}}=0$；**86.0%** complete turns 的 $\sigma_{\text{floor}}=0$。
  - Llama 排除率（26.8%）约为 Qwen（13.1%）的2.05倍，但 Llama pivotal 率反而更高（38.3% vs 30.5%）。
- **信用信号审计结果（全部通过 H3  placebo 检验）**：
  - Implicit 秩忠义（Qwen）：**0.0193** [−0.109, 0.081]，与控制带重叠。
  - Judge 秩忠义（Qwen）：**0.1142** [0.027, 0.168]，与控制带重叠。
  - Implicit 秩忠义（Llama校正）：**−0.043** [−0.125, −0.016]，与控制带重叠。
  - 条件化 fluency 后部分相关：**—0.004**（p=0.87），因果信息完全消失。
  - OLS 标准化系数：fluency 权重 **0.955** vs 因果增量 **0.402**（稳定 2.29–2.37 倍）；显式信用与 fluency 中位数秩相关 **+0.752**（Holm-adjusted p=0.0002）。
- **训练与成本实验**：
  - 七臂训练 vs 未训练基线（0.422成功率）：无臂可靠超越，54/128任务达标。
  - 每轮 optimizer steps 剂量范围 **112→8**（随 credit 稀疏度变化）；跨臂 JS 散度 0.0520 vs 臂内 0.0198（2.6倍），但 partial Mantel ρ=+0.078（p=0.774），签名差异完全由剂量解释，不含 credit 内容。
  - 置信度 router 节省 judge 调用 **13.1%/turn**（14.0%/trajectory），但 pivotal 回收率仅 **11.9%** → 机会水平。
  - 最小可检测效应（MDE）≈ **11.8 pp**；未训练 policy vs expert cloning 差距 −10.9 pp（McNemar p=0.0488）。
- **核心结论**：所有信用信号与随机 shuffle 控制无显著差异；implicit 信用是 fluency 的回声；outcome-only 训练在现行超参下发散，训练 substrate 存在缺陷。

## 相关工作脉络
- **HCAPO** (Tan et al., 2026)：Implicit 信用信号来源，原报道在 ALFWorld 达 SOTA；本文指出其信号实质为 fluency 回声，未捕捉因果增量。
- **C3** (Chen et al., 2026)：多 Agent 场景下的 credit 审计；本文强调二者审计对象不同（单步 vs 多智能体交互），不可直接类比。
- **CARL** (Shen et al., 2025)：报告类似的稀疏模式，用 entropy 分离关键/非关键状态（Cliff's δ=0.42）；本文通过 executed replay 提供更直接的因果代理验证。
- **StepOPSD** (Zhang et al., 2026) / **Verified Critical Step Optimization** (Li et al., 2026a) / **ProcessSupervised RL** (Tan et al., 2025)：依赖 annotated step correctness 或过程监督信号；本文论证正确性标注易与因果贡献混淆。
- **Who&when Pro** (Liu et al., 2026) / **Counterfactual Credit Policy Optimization** (Li et al., 2026b) / **Exact is Easier** (Chen et al., 2026)：属 failure attribution 或合作场景反事实信用方法；本文的预注册审计协议为这类工作提供了可复用的因果保真度检验范式。

## 局限性与未来方向
- **环境局限性**：审计仅在 ALFWorld 可重放环境中进行，尚未验证于开放域多轮对话、长视界视觉导航或多 Agent 协作场景。
- **可测量性过滤偏差**：约 13–27% 的 turn 因无法采到 4 个不同替代动作被排除，极端确定性子策略可能被系统性低估。
- **训练 substrate 缺陷**：outcome-only 训练在三种超参配置下均发散，提示当前相对 advantage 信号在单 Agent 步级信用场景下缺乏稳定优化基底，需引入正则或混合信号。
- **未来方向**：开发解耦 fluency 与因果贡献的信用构造方法；探索 dose-invariant 的 credit 评估协议；将 Executed Replay 泛化至多步多目标、带噪声反馈的复杂环境。

## 研究启发与可借鉴点
- **预注册多臂对照协议**：将阈值、排除规则、判定顺序在数据收集前冻结，并共享 CRN，可大幅降低信用信号研究中的 p-hacking 与剂量混淆风险，值得纳入团队实验规范。
- **Fluency 作为显式协变量**：在评估任何基于 log-prob/ratio 的策略梯度或 reward 信号时，应将 fluency 作为标准化协变量纳入部分相关或 OLS 分解，避免误将流畅度归因为因果归因。
- **Effective sample size 匹配原则**：不同 credit 规则的可测量步骤数差异巨大（如 Qwen 13.1% vs Llama 26.8% 排除率），比较时必须重加权或截断至共同支撑集，否则结论仅反映训练剂量而非算法差异。
- **成本路由的验证边界**：置信度 router 虽能节省约 13%/turn 的计算成本，但无法提升 pivotal 识别率；后续可将此类路由仅定位为工程优化，而非信号增强手段。

## 关键术语表
- **Executed Replay**：通过重采样替代动作并多次 roll-out，以事实与反事实结果均值差构建步骤因果贡献的无标注代理基准。
- **Implicit Credit**：由策略模型自身计算的条件 token log-prob ratio（$\rho_t$），常被用作步骤重要性打分，但本文证明其为 fluency 回声。
- **Judge Credit**：依赖 LLM 裁判对单步执行质量进行语义评分的信号家族，审计显示其秩忠义仍与 shuffle 控制重叠。
- **TARL（Turn-Level Evaluation）**：五分量表（1–5）评估框架，强制全范围使用，用于标注步骤对任务推进的真实贡献度。
- **H3（Placebo-level）**：审计原假设，指信用信号的分布与随机 shuffle 控制无显著差异，即不具备因果识别能力。
- **Effective Sample Size**：满足可测量性且具区分度的步骤子集规模；本文强调比较不同 credit 规则时须匹配此指标以避免剂量混淆。
- **Parameter Displacement Signature**：跨训练臂的模型参数分布散度（JS 散度），本文发现其差异完全由 optimizer steps 剂量解释，不含 credit 信息。
- **Causal Sparsity**：因果关键步骤在总步骤中的稀疏比例（本文测得 pivotal 率约 30.5%，零增量步占 69.5%）。

## 可复现要素
- **数据集**：ALFWorld（公开 embodied 环境）；128 个 held-out 任务，50 条初始轨迹（temperature=0.7）。
- **代码/权重**：论文未明确声明开源状态。
- **关键超参**：K=4 替代动作采样，每个 roll-out ≥3 次；历史长度=2；optimizer steps 剂量 112→8（随信用稀疏度调节）；Spearman 中位数经 10,000 次 bootstrap；PC2/PC3 诊断层携带扰动强度与位置单调性元数据。
- **评估工具**：TARL 五分量表与 LLM Judge 三分制变体（1/correct, -1/deviation
