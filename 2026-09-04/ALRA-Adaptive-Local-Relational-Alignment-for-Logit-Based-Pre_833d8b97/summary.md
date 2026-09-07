---
title: "ALRA-Adaptive-Local-Relational-Alignment-for-Logit-Based-Pre"
source: https://arxiv.org/pdf/2609.03355v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-07 05:30:21"
---

# 论文速读：ALRA: Adaptive Local Relational Alignment for Logit-Based Pre-training Distillation of Autoregressive Language Models

## 一句话总结
本文提出ALRA，一种面向自回归语言模型预训练知识蒸馏的位置自适应局部关系对齐方法。该方法通过学生提议与教师锚定动态构建局部token集，并结合区域散度（ALD）与学生加权成对关系对齐（SWPRA），在不丢失全词表监督的前提下精准匹配高概率候选token的相对偏好，显著提升小参数模型的zero-shot能力。

## 研究问题与动机
- 现有logit-based KD通常对全词表进行全局前向KL对齐，将不同预测上下文的不确定性视为同质，忽视了部分位置高度集中、部分位置多候选竞争的结构差异。
- 已有局部选择方法存在单向依赖缺陷：仅选教师高分token会遗漏学生当前认为可能的token；仅选学生高分token在随机初始化初期排序极不可靠。
- 固定局部集合大小无法随预测位置的上下文不确定性自适应调整，且现有方法未显式建模局部高概率token之间的相对偏好关系。
- 需要一种统一的目标函数，既能按位置动态决定监督的token范围，又能同时保留局部区域与剩余词表的结构化监督，并在局部集内显式建模成对关系。

## 核心贡献（创新点）
1. **教师锚定的自适应局部token选择机制**：每个预测位置由学生提议$d_{max}$个高概率token，强制纳入教师top-1作为锚点，再依据候选集内教师条件有效支撑集与批次平均的比值动态确定局部预算$d_u$。与仅依赖教师截断或学生排名的已有方法本质不同，该设计兼顾了学生当前预测状态与教师可靠性，并实现位置级动态资源分配。
2. **自适应局部散度（ALD）**：从全词表前向KL的精确local-rest分解出发，保留质量匹配项，但将局部条件与剩余条件KL的系数从教师概率质量替换为1。与Vanilla KD、TAD等直接保留质量权重或简单截断重归一化的基线本质不同，ALD防止低概率区域的散度因教师质量系数过小而被动削弱。
3. **学生加权成对关系对齐（SWPRA）**：在自适应局部集内对token对赋予权重，高学生总概率且小概率差的token对获得更大权重。与LDRLD等基于固定类别排名差加权的方式本质不同，SWPRA直接利用学生当前全词表概率分布，聚焦于学生尚难区分的高潜备选对。
4. **从零预训练蒸馏的系统验证**：在冻结Qwen1.5-1.8B教师与随机初始化200M/500M学生之间，于The Pile上进行约1B token预训练，并在9个zero-shot基准上全面验证，证明各模块的协同增益与良好的泛化性。

## 方法详解
- **教师锚定自适应局部token集**：学生输出top-$d_{max}$ token构成提议集$S_u^{\max}$，教师top-1 token $a_u=\arg\max_i p_{u,i}^T$ 强制加入形成候选集$C_u^{\max}$（若$a_u$已在其中则直接保留，否则替换最低排名学生提议）。计算候选集内教师条件分布的有效支撑集 $E_u^{\text{loc}}=\exp(-\sum_{i\in C_u^{\max}}\rho_{u,i}\log\rho_{u,i})$，并与批次平均 $\bar{E}_{\mathcal{T}_B}^{\text{loc}}$ 比较，通过 $d_u=\text{clip}(\text{round}[d_{\min}+(d_{\max}-d_{\min})\frac{E_u^{\text{loc}}}{\bar{E}_{\mathcal{T}_B}^{\text{loc}}+\epsilon}], d_{\min}, d_{\max})$ 得到位置自适应预算。最终局部集 $\mathcal{I}_u$ 取候选集内教师概率最高的$d_u$个token，补集为 $\mathcal{R}_u$。
- **自适应局部散度（ALD）**：将全词表forward KL精确分解为 $\mathrm{KL}(P^T\|P^S)=\mathrm{KL}(b^T\|b^S)+\alpha^T\mathrm{KL}(\tilde{P}^{T,I}\|\tilde{P}^{S,I})+\bar{\alpha}^T\mathrm{KL}(\tilde{P}^{T,R}\|\tilde{P}^{S,R})$。ALD保留质量匹配项，但将后两项系数置为1，即 $\mathcal{L}_{\text{ALD}}=\mathcal{L}_{\text{mass}}+\mathcal{L}_{\text{local}}+\mathcal{L}_{\text{rest}}$，使局部与剩余区域的分布对齐均不受教师概率质量衰减。
- **学生加权成对关系对齐（SWPRA）**：在 $\mathcal{I}_u$ 内枚举所有无序对 $(i,j)$，用配对温度 $\tau_p$ 计算两token相对偏好分布。未归一化分数 $s_{u,ij}=\exp(-\gamma|p_{u,i}^S-p_{u,j}^S|)(p_{u,i}^S+p_{u,j}^S)$，归一化得权重 $\omega_{u,ij}$。成对损失 $\mathcal{L}_{\text{SWPRA}}=\sum_{(i,j)}\omega_{u,ij}\mathrm{KL}(r^T(i,j)\|r^S(i,j))$，优先对齐学生当前概率相近且总质量高的高潜备选对。
- **总体目标**：$\mathcal{L}_{\text{ALRA}}=\frac{1}{|\mathcal{T}_B|}\sum_{u}[\mathcal{L}_{\text{ALD}}(u)+\lambda_{\text{pair}}\mathcal{L}_{\text{SWPRA}}(u)]$，可附加交叉熵项（本文 $\lambda_{\text{CE}}=0$）。仅更新学生参数，教师冻结。

## 实验与结果
- **数据集与设置**：处理自The Pile Uncopyrighted（剔除Books3/BookCorpus2/OpenSubtitles/YTSubtitles/OWT2）；教师Qwen1.5-1.8B冻结，学生为随机初始化的Pretrain-Qwen-200M与500M；共享优化配置（AdamW, $\beta_1=0.9,\beta_2=0.98,\epsilon=10^{-6},
