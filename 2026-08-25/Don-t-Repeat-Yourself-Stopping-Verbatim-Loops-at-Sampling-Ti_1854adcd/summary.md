---
title: "Don-t-Repeat-Yourself-Stopping-Verbatim-Loops-at-Sampling-Ti"
source: https://arxiv.org/pdf/2608.22761v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 19:50:26"
---

# 论文速读：Don't Repeat Yourself: Stopping Verbatim Loops at Sampling Time

## 一句话总结
提出 **DRY（Don't Repeat Yourself）**，一种在采样时基于“当前词缀是否精确续接历史片段”动态调整 logit 的解码惩罚机制；通过序列感知匹配与结构化断点设计，在将逐字循环率降低约 47% 的同时保留格式与核心推理能力，且推理开销可控（128K 上下文下 <3%）。

## 研究问题与动机
- **核心问题**：自回归 LLM 在长上下文聊天与生成中易陷入 **verbatim looping**（逐字循环），即模型开始重复上下文中已出现的 token 序列，且一旦启动往往自我强化数十至数百 token。
- **现有方法不足**：工业界广泛采用的重复惩罚（repetition penalty）、存在/频率惩罚（presence/frequency penalty）及 n-gram 硬屏蔽均基于**词元级别的先验出现次数**施加惩罚，无法区分“恶性循环起始”与“良性结构复用”（如对话模板中的换行符、角色标签、Markdown 列表标记），导致抑制循环时必然牺牲格式完整性与输出流畅度。
- **评估空白**：尽管 DRY 已被 llama.cpp、ExLlamaV2、text-generation-webui 等主流推理框架工程化集成，但此前缺乏系统的对照实验与机制验证，本文填补该空白。

## 核心贡献（创新点）
1. **将重复控制从“词元频次”重构为“序列延续”**：提出 DRY 采样时 logit 调整器，仅当候选 token 会将当前后缀精确延伸至历史已出现片段时才施加惩罚，理论上证明其在多数解码步对分布无扰动。
2. **指数渐变惩罚 + 可配置序列断点**：设计 $\lambda \beta^{n-L}$ 的指数放大函数实现软控制，并通过断点集合 $B$（换行、冒号、引号等）阻止匹配跨越结构边界，避免对 chat template 与格式 token 的误伤。
3. **多尺度、多场景系统性评测**：在 1.5B~120B 参数模型、九类提示家族、双解码 regime 下验证，DRY 使 SER@4 下降 47% 且 distinct-4、MAUVE 同步提升。
4. **前沿模型能力保留验证**：在 AWQ 4-bit 量化的 Llama-3-70B-Instruct 与 GPT-OSS-120B 上，DRY 将循环率减半的同时在 MT-Bench、MMLU、GSM8k 上与无干预基线持平，而传统惩罚方法出现可测量能力下降。
5. **机制特异性与人类感知双重验证**：通过干预匹配的安慰剂对照（placebo）确认后缀匹配是生效机制；600 对盲评 MTurk 实验证明 DRY 在“避免循环”上显著优于基线，且人类感知的格式保护优势与自动化指标一致。

## 方法详解
- **参数定义**：允许重复阈值 $L$、乘数 $\lambda>0$、底数 $\beta\ge1$、断点集合 $B$（tokenizer 相关，默认含 newline、colon、quotation mark、asterisk 等）。
- **匹配机制**：对候选 token $v$ 在步 $t+1$，向后扫描当前上下文，寻找满足以下条件的最长匹配后缀长度 $n_t(v)$：
  1. 该后缀曾在更早位置 $i<t$ 完整出现过；
  2. 扩展后不跨越断点 token（$x_{t-j} \notin B$）。
- **Logit 调整公式**：
  $$
  z'_t(v) = 
  \begin{cases}
  z_t(v) - \lambda \beta^{n_t(v) - L}, & \text{if } n_t(v) \ge L \text{ and } v \notin B \\
  z_t(v), & \text{otherwise}
  \end{cases}
  $$
- **关键性质**：
  - **选择性干预**：仅当 $v$ 会延续出长度 $\ge L$ 的历史后缀时才修改 logit，绝大多数候选 token 不受影响。
  - **指数增长**：短匹配（接近 $L$）仅受轻微 nudging，长匹配（循环已成型）被快速压制，避免硬截断带来的多样性崩溃。
  - **结构保留**：断点 $B$ 使匹配在换行、引号等结构边界处终止，保护必要的模板复用。
- **计算效率**：C++ 反向搜索有界实现，遇首处 mismatch 或断点即停止，平均时间复杂度 $O(1)$；128K 上下文额外延迟 2.6%， worst case 6.4%（1.5B 模型），显著低于全词表 pass 的传统重复惩罚。

## 实验与结果
- **模型与规模**：主实验 Qwen 2.5-1.5B、Llama 3.2-3B、Qwen 2.5-7B；扩展 Qwen 2.5-14B、Llama-3-70B-Instruct、GPT-OSS-120B（AWQ 4-bit 量化）。
- **基线**：无干预、Repetition penalty、Presence penalty、Frequency penalty、No-repeat n-gram blocking、Contrastive decoding、Placebo control。
- **核心指标**：SER@L（词缀延伸率）、distinct-4、loop-free length、MAUVE（vs WikiText-103 人类参考 / baseline 参考）、MT-Bench、MMLU、GSM8k。
- **主要数值结果**：
  - **SER@4**：DRY 从 0.124 降至 0.065，相对
