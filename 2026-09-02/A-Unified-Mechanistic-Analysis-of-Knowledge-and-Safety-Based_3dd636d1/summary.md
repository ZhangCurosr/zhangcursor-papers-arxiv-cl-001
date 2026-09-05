---
title: "A-Unified-Mechanistic-Analysis-of-Knowledge-and-Safety-Based"
source: https://arxiv.org/pdf/2609.00760v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 20:53:10"
---

# 论文速读：A-Unified-Mechanistic-Analysis-of-Knowledge-and-Safety-Based

## 一句话总结
本文首次在同一受控对比设置下系统揭示了大语言模型中知识型拒绝（KR）与安全型拒绝（SR）的共享与分化机制：两者共享一个主导的拒绝方向，但跨类型迁移呈现显著的非对称性，且类型特异性信号在顶层分层涌现，支持“先承诺后细分（commit-then-specify）”的两阶段拒绝架构。

## 研究问题与动机
- **KR与SR长期被孤立研究**：KR主要归属于不确定性估计与幻觉缓解文献，SR主要归属于LLM对齐、越狱防御与策略合规文献，两者表面输出相似却缺乏统一的机制对比。
- **现有基准存在严重混淆**：KR与SR评测集在提示词结构、语义域、表层词汇上差异巨大，难以剥离主题/格式因素，直接归因于“拒绝类型”本身。
- **模型训练历史干扰表征归因**：标准指令微调模型通常在不同阶段、使用不同来源的数据分别习得KR与SR，观察到的差异可能仅反映训练轨迹而非内部组织方式。
- **核心科学问题**：（RQ1）KR与SR是否共享底层机制？（RQ2）若共享，跨类型迁移是否对称？（RQ3）类型特异性信号在何处、以何种形式涌现？

## 核心贡献（创新点）
- **构建受控对比四元组数据集**：提出213组(KR, SR, KC, SC)四元组，通过最小化编辑构造控制对，首次在同一主题与句法下隔离出拒绝特异性表征。与已有工作相比，突破了以往KR/SR基准格式不一、无法直接对比的局限。
- **证明主导拒绝方向共享但残差显著**：发现KR−KC与SR−SC位移在峰值层的余弦对齐度达0.699~0.794，公共投影范数占0.92~0.95，正交残差仍保持0.32~0.39。与仅关注单一拒绝类型的既往机制研究不同，本文量化了共享与特异成分的几何比例。
- **揭示稳定的跨类型迁移非对称性**：SR→KR探针迁移准确率（0.80~0.86）显著高于KR→SR（0.58~0.63），差距Δ=0.22~0.27且在全部基座/微调模型及两种微调顺序下稳健。区别于以往将拒绝视为同质行为的假设，本文指出SR更深度依赖共享拒绝结构。
- **提出“承诺-细分”两阶段机制并验证行为效应**：通过层间投影、Logit Lens与组件消融证明通用拒绝倾向在早期层形成，KR/SR特异性语义分化集中于顶层（relative depth > 0.9）；激活导向实验进一步表明类型特异性方向可定向调制拒绝理由框架。该框架为后续分层干预提供了可解释的理论基础。

## 方法详解
- **受控数据集构建**：基于20个语义类别（如Personal data access、Cybersecurity abuse等）与共享模板，由GPT-4o-mini并行生成KR（虚构/不可验证指代）与SR（真实敏感/违规指代）候选对，经规则过滤、GPT-5质量评估、KC/SC控制对最小化编辑构造及六模型行为验证，最终保留213组通过模板相似度、行为合理性与拒绝触发器移除校验的四元组。
- **受控拒绝微调**：在Llama-3-8B-Instruct、Qwen2.5-7B-Instruct、Gemma-2-9B-it上使用CRaFT（知识拒绝）与FalseReject（安全拒绝）进行全参数SFT；对比Seq-KS（先知识后安全）、Seq-SK（先安全后知识）与Merged配置，选定Seq-KS变体（Llama*/Qwen*/Gemma*）以在保留MMLU/GSM8K通用能力的前提下同步获得两类拒绝行为。
- **拒绝方向估计与几何分解**：提取各层末token残差流激活 $h_\ell(x)$，计算均值位移 $d_\ell^r = \mathbb{E}[h_\ell(x^{rR}) - h_\ell(x^{rC})]$（$r \in \{K,S\}$）；定义单位公共方向 $\hat{d}_\ell^{\mathrm{common}} = (\hat{d}_\ell^K + \hat{d}_\ell^S)/\|\hat{d}_\ell^K + \hat{d}_\ell^S\|$，并将各方向分解为公共投影 $d_{\ell,\mathrm{proj}}^r = (\hat{d}_\ell^r \cdot \hat{d}_\ell^{\mathrm{common}})\hat{d}_\ell^{\mathrm{common}}$ 与正交残差 $d_{\ell,\mathrm{spec}}^r = \hat{d}_\ell^r - d_{\ell,\mathrm{proj}}^r$。
- **探针与层间轨迹分析**：训练L2正则逻辑回归探针（$C=1.0$, 5-fold交叉验证），评估同类准确率与跨类迁移；以终层KR特异性残差为固定参考方向 $\hat{d}_{\mathrm{ref}}$，逐层计算投影 $p_\ell^r = d_\ell^r \cdot \hat{d}_{\mathrm{ref}}$ 以追踪分化时机。
- **可解释性解码与组件消融**：使用Logit Lens将终层方向经归一化层与未嵌入矩阵 $W_U$ 投影至词表，过滤3~18字符ASCII首字母词以观察语义偏好；选取对KR特异性方向贡献Top-10的注意力头进行零消融，并逐层零消融MLP块计算不对称评分 $A^{(\ell)} = \Delta_{\mathrm{SR}}^{(\ell)} - \Delta_{\mathrm{KR}}^{(\ell)}$。
- **激活导向（Activation Steering）**：在峰值层 $\ell^*$ 向残差流叠加 $+\alpha \hat{d}^r$， sweeps $\alpha$ 并统计 Lexicon
