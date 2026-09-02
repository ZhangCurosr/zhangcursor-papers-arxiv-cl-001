---
title: "TASK-COEVOLVE-EFFICIENT-HARNESS-OPTIMIZA-TION-VIA-ADAPTIVE-V"
source: https://arxiv.org/pdf/2608.20169v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 12:37:04"
field: "LLM Agent Harness 自动化优化"
keywords: ["harness optimization", "adaptive validation task selection", "variance-weighted sampling", "Horvitz-Thompson estimation", "efficient LLM evaluation", "agent self-improvement"]
innovations: ["提出方差加权自适应验证任务选择（VWS）以聚焦能力前沿任务", "设计采样感知全量性能估计（SaFE）实现跨迭代可比性评估", "在7%-20%评估预算下逼近或超越全量搜索，Terminal-Bench成本降67%-80%"]
benchmarks: ["LawBench", "Symptom2Disease", "USPTO-50k", "Terminal-Bench 2.1"]
---

# 论文速读：TASK-COEVOLVE: EFFICIENT HARNESS OPTIMIZATION VIA ADAPTIVE VALIDATION TASK SELECTION

## 一句话总结
本文针对LLM Agent harness（控制代码）自动化优化过程中的评估成本过高问题，提出 **Task-CoEvolve** 框架，通过**方差加权自适应选择验证任务子集**并结合**采样感知全量性能估计**，在仅使用少量评估预算（7%~20%）的情况下实现与全量搜索相当甚至更优的最终性能，大幅降低计算开销。

## 研究问题与动机
1. **评估成本高昂**：现有harness优化方法（如Meta-Harness）在每次迭代中都对完整固定验证集进行全量评估，当任务执行代价大（如长周期终端任务）时，评估成本主导整个优化循环。
2. **静态验证集低效**：随着harness演化，能区分候选的差异性任务分布会变化；始终被解决或始终失败的“天花板/地板”任务持续消耗评估预算却提供微弱信号。
3. **子集复用导致过拟合**：若简单复用固定子集评估，meta-agent易偏向在该子集上表现好的修改，损害泛化性；若每次随机重采样，则不同迭代的得分因子集难度不同而不可比。
4. **正交于现有效率优化**：已有工作多聚焦于减少候选数量或改进候选生成/选择策略，本文从**每次评估所需任务数**这一正交维度切入，直接削减每个候选的评估成本。

## 核心贡献（创新点）
1. **引入“验证任务选择优化”新设定**：将优化对象从仅harness扩展至“harness + 用于评估的验证任务子集”，与减少候选数的方法正交互补。  
2. **方差加权自适应任务选择（VWS）**：基于历史成功率的伯努利方差度量任务信息量，优先采样处于能力前沿（成功/失败均衡）的任务，采样分布随harness演化动态调整。  
3. **采样感知全量性能估计（SaFE）**：利用Horvitz-Thompson思想，结合Hajek估计器或锚定差值估计器，从非均匀采样的部分子集无偏估计全量得分，使跨迭代比较一致。  
4. **在两个高价值基准上验证**：在线文本分类中7%预算即逼近全量搜索，20%预算超越全量搜索；Terminal-Bench 2.1上以20%预算匹配全量搜索性能，整体搜索成本降低67%~80%，时间减半。

## 方法详解
Task-CoEvolve包含三个主要阶段（对应论文Fig. 2）：

### Phase 0：初始化
在搜索开始前，用两个初始harness（zero-shot与few-shot-all）在全量验证集T上各运行一次，记录历史成功率$\bar{p}_t$作为种子分布。

### Phase 1：方差加权任务选择（VWS）
在迭代$k$，对每个任务$t$计算采样权重：
$$
w_t = \max\big(\bar{p}_t(1-\bar{p}_t),\ \ell_t\big) + \frac{\lambda}{\sqrt{n_t}}
$$
- 第一项为伯努利方差，当$\bar{p}_t=0.5$时最大，持续成功/失败的任务方差为零。
- $\ell_t$为保底权重：从未被解决的 task 设为小常数$\ell$，其余为0，避免早期完全无人解决的难题被永久忽略。
- 第二项为不确定性探索项，观察次数$n_t$少时权重增大。
从该分布中无放回抽样大小为$m=\lceil\rho N\rceil$的子集$S_k$进行评估。

### Phase 2：采样感知全量估计（SaFE）
根据任务池结构选用两种无偏估计之一：
- **Hajek估计器**（适用于均值接近0或1、分为若干子池的场景）：
$$
\hat{S}(h) = \frac{\sum_{t\in S_k} x_t(h)/\pi_t}{\sum_{t\in S_k} 1/\pi_t}
$$
- **锚定差值估计器**（适用于均值居中、易被极端任务主导的场景）：
$$
\hat{S}(h) = \frac{1}{N}\sum_{t\in\mathcal{T}}\bar{p}_t + \frac{1}{N}\sum_{t\in S_k}\frac{x_t(h)-\bar{p}_t}{\pi_t}
$$
其中$\pi_t$为任务$t$被抽入子集的概率，通过4000次Monte Carlo模拟估计。$\hat{S}$作为跨迭代可比的全量性能代理，选择使$\hat{S}$最大的候选作为最终harness。

## 实验与结果
**数据集与设置**：
- 在线文本分类：LawBench（215类）、Symptom2Disease（22类）、USPTO-50k（180类），验证集共130例。
- Terminal-Bench 2.1：89个长周期终端任务，使用GPT-5.6-Luna与Qwen3.6-35B-A3B两个模型。
- 基线：Meta-Harness（Full Search, $\rho=100\%$）、Naive（固定子集复用）、Random-Resample（均匀重采样+原始子集均值）。
- 迭代次数：文本分类20轮（每轮3候选）、Terminal-Bench 10轮（每轮1候选）。

**主要结果**：
- 文本分类7%预算：Task-CoEvolve平均测试准确率**47.6±0.9**，接近Full Search的48.6±0.8；Few-shot起始值41.6%。
- 文本分类20%预算：Task-CoEvolve **49.3±0.8**，超越Full Search的48.6±0.8，提升约1个百分点。
- Terminal-Bench 2.1（20%预算）：GPT-5.6-Luna上61.8%（Full Search 62.9%）、Qwen3.6上41.6%（Full Search 42.7%），差距仅约1个任务。
- 成本缩减：Terminal-Bench 2.1搜索token输入减少**67%~80%**，搜索时间缩短约**一半**（GPT-5.6: 22.2h→11.5h；Qwen3.6: 38.0h→20.5h）。

**消融与诊断**：
- 组件贡献（$\rho=20\%$）：Naive 47.2 → +随机重采样 48.2 → +全量估计 48.8 → +VWS 49.3。
- 估算准确性：20%预算下$\hat{S}$与真实分排名相关系数Spearman=0.62，Top-1候选真实排名12/60；7%预算相关性降至0.13但仍保持在Top 1/6。
- 难度分布随迭代变化：几乎人人解决的样本从34增至58，最具有判别力的“一半样本”占比约10%~24%，支持自适应必要性。

## 相关工作脉络
1. **Meta-Harness (Lee et al., 2026)**：代表性固定全量验证集的harness自进化框架，Task-CoEvolve在其基础上正交地降低每次评估的任务数，二者可结合。
2. **DemoEvolve / ShinkaEvolve / TurboEvolve / HarnessCompass**：从候选生成、选择、约束优化等搜索侧提升效率，不触碰评估侧；本文从“评估哪个子集”切入，形成互补。
3. **课程学习 & 自适应任务选择**：关注训练过程中的样本/课程调度以更新模型参数；本文关注评估过程中的任务调度以筛选最优harness，目标不同。
4. **Active Testing / tinyBenchmarks / AcTracer**：面向固定模型的样本高效评估，通过重要性加权或紧凑子集估计性能；本文扩展至**演化中候选比较**与**非均匀采样无偏估计**两个核心差异。
5. **自动Harness设计（Zhang et al., 2026c; Ye et al., 2026; Merrill et al., 2026）**：多为人工设计或单次搜索，本文提供迭代自动优化的通用范式与低成本评估工具。
6. **Terminal-Bench 系列（Merrill et al., 2026）**：本文使用其2.1版本验证方法在昂贵长周期任务上的有效性，填补了高效评估机制在该场景的空白。

## 局限性与未来方向
1. **固定每候选评估任务数**：当前预先设定$m$，无法对明显差的候选提前止损，也无法对难以区分的候选动态追加评估。
2. **模型过强时优化空间枯竭**：附录C指出，使用DeepSeek-V4-Flash时起始harness已解70.8%任务，搜索无法找到更好候选，方法无法“无中生有”。
3. **估算器依赖初始分布假设**：Hajek与差值估计的选择基于Phase-0全量评估的经验判断，对未知任务池可能需额外调参。
4. **未来方向**：作者明确建议将评估预算在候选间动态分配（类似置信上界或主动停止机制）作为下一步工作。

## 研究启发与可借鉴点
1. **方差加权选择范式可迁移**：基于历史结果方差的“能力前沿聚焦”策略可推广至模型微调的数据选择、RLHF的prompt采样等需要高效评估的场景。
2. **采样感知无偏估计（Horvitz-Thompson）在贝叶斯优化中的应用潜力**：将包含入样概率的调整纳入效用函数估计，有助于缓解非均匀采样导致的策略偏差。
3. **Hajek vs. 锚定差值估计器的选择准则**：论文提供的启发式（均值近0/1用Hajek，居中用锚定差值）对设计其他基准的高效评估协议有参考意义。
4. **与现有搜索侧优化方法叠加**：Task-CoEvolve与DemoEvolve、HarnessCompass等在搜索效率上正交，可组合使用以获得双重收益。
5. **验证集难度分布动态追踪**：通过维护“无人解决/少数解决/半数解决/几乎全解决”四象限统计，可诊断优化过程是否陷入局部饱和，辅助调试harness演化流程。

## 关键术语表
**Harness Optimization**：在固定LLM参数前提下，迭代改写围绕模型的“控制代码”（提示、检索策略、记忆管理等），以提升其在目标基准上的性能。

**Variance-Weighted Sampling (VWS)**：以历史成功率的伯努利方差为主要权重依据，优先采样判别力最强的验证任务，同时保留保底与探索项。

**Hajek Estimator**：对 sampled outcomes 按 inclusion probability 倒数加权后归一化的全量性能估计量，适合成功率贴近0或1、任务可分组的场景。

**Anchored Difference Estimator**：以历史成功率为锚，对当前候选偏离锚的增量按逆概率加权并累加，适合任务均值居中、极端任务易主导Hajek估计的场景。

**Evaluation Budget ($\rho$)**：每次迭代实际评估任务数占全量验证集的比例，本文测试7%与20%。

**Candidate Harness**：每轮迭代由meta-agent生成的待评估控制代码变体，最终从所有候选中选取$\hat{S}$最高者。

**$\hat{S}$-Max Selection**：依据全量估计得分$\hat{S}$选择最终harness，打破跨迭代不可比性。

**Meta-Agent**：负责基于历史评估结果提议新harness的LLM控制器（本文使用Claude Opus 4.6）。

## 可复现要素
- **数据集**：LawBench、Symptom2Disease、USPTO-50k（均为公开学术数据集）；Terminal-Bench 2.1（arXiv:2601.11868）。
- **代码**：论文声明将开源至 `https://github.com/Agent4Science-UTokyo/Task-CoEvolve`。
- **模型**：LLM backbone使用 GPT-OSS-120B（分类）与 GPT-5.6-Luna / Qwen3.6-35B-A3B（Terminal-Bench）；meta-agent使用 Claude Opus 4.6。
- **关键超参**：见附录 Table D——保底权重 $\ell=0.125$，探索系数 $\lambda=0.025$，Monte Carlo重复4000次，历史窗口为全部过去迭代，$\rho\in\{7\%, 20\%\}$。
- **随机种子与重复**：每个设置运行3次不同随机种子，报告均值±标准差。
