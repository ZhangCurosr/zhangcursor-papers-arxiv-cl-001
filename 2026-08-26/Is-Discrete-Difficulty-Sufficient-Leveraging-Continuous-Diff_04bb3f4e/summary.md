---
title: "Is-Discrete-Difficulty-Sufficient-Leveraging-Continuous-Diff"
source: https://arxiv.org/pdf/2608.24590v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 14:53:02"
field: "大模型高效推理与测试时计算"
keywords: ["Self-Consistency", "Test-Time Scaling", "continuous difficulty", "output entropy", "token-efficient reasoning", "probe", "dynamic budget allocation"]
innovations: ["用输出熵替代离散难度类别作为连续信号指导采样预算", "仅用最后token嵌入训练轻量线性探针预测熵值", "在保持准确率的同时最高节省76% token消耗"]
benchmarks: ["MATH500", "AMC23", "AIME2024", "AIME2025", "GPQA-Diamond", "MMLU-Pro"]
---

# 论文速读：Is-Discrete-Difficulty-Sufficient-Leveraging-Continuous-Diff

## 一句话总结
论文提出了 Flexible Self-Consistency (FSC) 框架，将问题难度建模为**连续信号（输出熵）**而非离散类别，通过轻量级线性探针预测熵值并动态调整推理路径数量；在多个模型和基准上，FSC 保持了与 Self-Consistency (SC) 相当的准确率，同时最高节省了 **76%** 的 token 消耗。

## 研究问题与动机
- **SC 的 token 消耗过高**：Self-Consistency 通过采样多条推理路径进行多数投票，虽然能显著提升复杂推理任务的表现，但随路径数增加，token 消耗线性增长，计算效率受限。
- **现有难度自适应方法的粒度不足**：已有方法（如 DSC）将难度划分为少数离散等级（如"简单/困难"），同一等级内的问题仍存在显著的推理需求差异，导致资源分配不精准——或过度计算，或探索不足。
- **输出熵可作为连续的细粒度难度信号**：实验表明，随着问题难度提升，模型回答的多样性（尤其是错误答案的多样性）单调上升，输出分布的熵值与感知难度呈正相关，因此可用熵替代离散难度标签。

## 核心贡献（创新点）
1. **提出 FSC 框架，首次用连续熵信号替代离散难度类别指导推理资源分配**：与 DSC 等基于离散分类的方法本质不同，FSC 允许每个问题获得细粒度的专属采样预算。
2. **设计轻量级线性探针，仅用最后 token 嵌入即可预测输出熵**：无需额外 LLM 自评估调用，训练和推理开销极低。
3. **揭示输出熵作为连续难度信号的泛化性**：跨模型（3B–14B）、跨数据集（MATH、AMC、AIME、GPQA、MMLU-Pro）均验证熵值随难度单调递增。
4. **实现最高 76% 的 token 节省且保持准确率持平甚至提升**：在 Qwen2.5-14B 的 MATH500 上，FSC 以 6.1×10³ token 达到 82.2% 准确率，相比 SC 节省 75.1% token 且准确率+0.6pp。

## 方法详解
FSC 分为两个阶段：

**阶段一：训练轻量级线性探针**
- 对 MATH 训练集（7,500 题）的每题生成 N=40 条推理轨迹，提取最终答案集合 $\mathcal{A}_q$，计算答案分布的相对频率 $p_q(a) = c_q(a)/N$，进而得到输出熵标签：
$$H_q = -\sum_{a \in \mathcal{U}_q} p_q(a) \log_2 p_q(a)$$
- 使用 LLM 对输入问题 $q$ 的最后 token 隐藏表示 $h_q$ 作为特征，训练无激活函数的线性回归模型：$\hat{H}_q = w^\top h_q + b$，优化 MSE 损失 $\mathcal{L} = \mathbb{E}[(H_q - \hat{H}_q)^2]$。

**阶段二：熵引导的自适应采样**
- 对测试题，探针预测熵值 $\hat{H}_q$，并进行裁剪：$\hat{H}_q \leftarrow \text{clip}(\hat{H}_q, 0, \log_2 N)$。
- 将预测熵归一化为相对难度得分 $r_q = \hat{H}_q / \log_2 N \in [0,1]$，据此计算自适应采样预算：
$$N_{\text{adj}} = \lceil 1 + (N-1) \cdot r_q \rceil$$
- 用 $N_{\text{adj}}$ 条推理路径进行多数投票确定最终答案。

## 实验与结果
- **数据集**：MATH500、AMC23、AIME2024、AIME2025、GPQA-Diamond（主实验）；MMLU-Pro（OOD 泛化）。
- **模型**：Qwen2.5-Instruct（3B、7B、14B）、Gemma-3-4B-it。
- **基线**：SC（固定 40 条路径）、AC（自适应一致性）、ESC（早停 SC）、DSC（离散难度自适应 SC）。
- **主要结果**：
  - Qwen2.5-14B + MATH500：FSC 82.2% acc / 6.1×10³ tok（**-75.1%** vs SC 的 24.5×10³ tok）。
  - Qwen2.5-14B + GPQA-Diamond：FSC 46.7% acc / 7.5×10³ tok（**-68.2%** vs SC）。
  - Gemma-3-4B + AMC23：FSC 52.5% acc / 18.5×10³ tok（**-64.8%** vs SC）。
  - 跨所有模型-数据集组合，FSC 均实现最低 token 消耗，且准确率持平或优于 DSC。
  - MMLU-Pro 跨 14 个学科域，FSC 在多数域达到最高效率，验证 OOD 泛化能力。
- **对比 DSC 的优势**：FSC 在推理路径分配分布上呈现更连续的梯度，避免 DSC 在"简单"类别集中分配 1 条路径导致的探索不足。

## 相关工作脉络
- **SC (Wang et al., 2022)**：多路径采样 + 多数投票，FSC 在其基础上引入动态预算控制。
- **AC (Aggarwal et al., 2023) / ESC (Li et al., 2024)**：基于中间一致性/窗口收敛的早停策略；FSC 与之区别在于不依赖答案一致性判定，而是用熵信号预分配预算。
- **DSC (Wang et al., 2025)**：基于 LLM 自评估的离散难度分级；FSC 将其推广为连续信号，解决同等级内粒度不足的问题。
- **Zhu et al. (2025)**：用值函数（无生成）估计内部难度；FSC 进一步使用输出熵这一可解释且易计算的信号。
- **Lee et al. (2025)**：线性探针预测离散难度标签；FSC 将探针输出从离散标签扩展为连续熵值，提升分配精度。

## 局限性与未来方向
- **模型规模受限**：仅在 ≤14B 参数模型上验证，未涉及更大模型（如 70B+），且熵-难度关系可能随模型规模变化。
- **探针训练数据单一**：仅在数学数据集（MATH）上训练，跨领域泛化（如代码、多语言）有待验证。
- **需访问隐藏表示**：探针依赖最后 token 的隐藏层输出，无法直接应用于 GPT 等封闭 API 模型。
- **未来方向**：扩展探针训练至更多元数据；开发不依赖内部表示的代理指标；探索在更大模型族上的通用性。

## 研究启发与可借鉴点
1. **连续信号优于离散分类**：将问题难度/不确定性建模为连续变量（而非几档分类）可实现更细粒度的资源分配，该思想可迁移至推理步数控制、思考深度调节等场景。
2. **轻量探针的低成本高效性**：仅用最后 token 嵌入训练线性模型即可捕捉复杂属性，这种"冻结 LLM + 轻量头"范式值得在其他可控生成任务中复用。
3. **输出熵作为模型不确定性的代理**：实验证明熵与难度单调相关，且错误答案多样性更高；可进一步探索熵在其他风险敏感决策中的应用。
4. **采样预算的归一化映射策略**：公式 $N_{\text{adj}} = \lceil 1 + (N-1) \cdot r_q \rceil$ 提供了一种简单且可解释的连续值→离散预算映射，易于在其他自适应推理框架中复用。

## 关键术语表
- **Self-Consistency (SC)**：对同一问题采样多条推理链，通过多数投票聚合得到最终答案的解码策略。
- **Flexible Self-Consistency (FSC)**：本文提出的新框架，基于输出熵预测动态调整采样预算。
- **输出熵 (Output Entropy)**：由答案分布频率计算的香农熵，作为模型对问题不确定性的连续度量。
- **轻量级线性探针 (Lightweight Linear Probe)**：接在 LLM 最后 token 嵌入后的无激活线性回归模型，用于预测熵值。
- **Test-Time Scaling (TTS)**：在推理阶段动态分配计算资源以提升性能的范式。
- **推理路径 (Reasoning Path/Chain)**：LLM 针对单个问题生成的一条完整推理轨迹。
- **采样预算 (Sampling Budget)**：为某个问题分配的推理路径数量上限。

## 可复现要素
- **数据集**：MATH（训练，7,500 题，MIT License）；评测集 MATH500、AMC23、AIME2024/2025、GPQA-Diamond、MMLU-Pro 均有开源。
- **代码/权重**：论文未明确声明开源仓库链接；探针权重需自行训练。
- **关键超参**：训练时 temperature=1.0、top-p=1.0、N=40；推理时 temperature=0.7、top-p=0.95；探针为纯线性层，无隐藏层；AC 阈值设为 0.95，ESC 窗口大小 5，DSC judge 窗口 24。
