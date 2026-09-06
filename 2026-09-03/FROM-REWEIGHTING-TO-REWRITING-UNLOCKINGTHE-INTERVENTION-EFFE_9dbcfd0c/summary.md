---
title: "FROM-REWEIGHTING-TO-REWRITING-UNLOCKINGTHE-INTERVENTION-EFFE"
source: https://arxiv.org/pdf/2609.02771v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 20:59:39"
---

# 论文速读：FROM-REWEIGHTING-TO-REWRITING-UNLOCKINGTHE-INTERVENTION-EFFE

## 一句话总结
提出 influence-guided response rewriting 框架，将影响函数（IF）仅用作目标样本选择器，并通过改写选定样本的响应内容（而非调整权重）实施干预；实验表明该方式能激活 IF 选中样本被传统重加权掩盖的行为杠杆，在四个开源 LLM 的 epistemic abstention 任务上实现更强、更持久且双向的行为偏移。

## 研究问题与动机
- 训练数据归因（TDA）旨在识别塑造模型行为的训练样本，但近期研究指出 IF 选中的高影响力样本在常规权重干预（upweighting/deletion）下效果常不优于随机选择，引发“IF 筛选本身无效”还是“重加权未能释放其潜力”的争议。
- 传统 TDA 干预将注意力集中在“样本权重调整”上，却忽略了 SFT 场景下可直接修改样本提供的监督内容（保留 instruction，替换 response）。
- 论文动机：解耦“选择”与“干预”两个阶段，系统对比相同 IF 选中样本在重加权与响应改写下的干预轨迹，重新评估 IF 的真实行为杠杆价值。
- 以 epistemic abstention（知识性拒答）为主要测试床，并验证结论在安全拒绝（safety refusal）场景的泛化性。

## 核心贡献（创新点）
1. 提出 influence-guided response rewriting 框架，将 IF 归因与监督级干预解耦，实现按行为方向（align/opp）改写响应内容的细粒度干预；与传统权重干预的本质区别在于，前者改变样本教授的**行为信号内容**，后者仅放大或缩小原信号的**强度**。
2. 揭示 IF 选中样本的隐藏行为杠杆：在四个 LLM 上验证响应改写可产生比重加权强得多且更稳定的双向偏移，证明“重加权失效”源于干预形式局限而非样本价值不足。
3. 刻画干预杠杆的来源与作用边界：通过对照实验证明 IF 选择并非简单捕获内部表征对齐度最高的样本，而是定位出可被有效重定向的关键训练位置；改写后的行为变化集中体现在目标场景，且不引发通用能力退化。

## 方法详解
- **影响分数计算**：采用标准 SFT 响应损失 $\mathcal{L}(z_i,\theta)=-\sum\log p_\theta(y_{i,t}|x_i,y_{i,<t})$，利用 EK-FAC 近似逆 Hessian $H_\theta^{-1}$，计算 $\mathcal{I}_f(z_i)=-\nabla_\theta f(\theta)^\top H_\theta^{-1}\nabla_\theta\mathcal{L}(z_i,\theta)$；以 abstention 查询集的均值响应 log-likelihood 作为可微行为代理 $f(\theta)$，按分数排序得到 supposedly helpful / supposedly harmful 集合。
- **两类干预设计**：
  - *Reweighting*：对选中集合 $\mathcal{S}$ 施加权重 $w_i(\alpha)$，$\alpha=2$ 为上加权，$\alpha=0$ 为删除，响应内容保持不变。
  - *Response Rewriting*：固定 instruction $x_i$，将响应替换为行为对齐或对抗模板 $\tilde{y}_i^d$，构成新样本 $(x_i,\tilde{y}_i^d)$；改写不沿用 IF 的局部重加权解释，IF 仅决定“改哪里”，改写决定“改成什么”。
- **评估协议**：定义干预前后行为指标变化 $\Delta(\mathcal{O},\mathcal{S})=B(\theta_{\mathcal{O},\mathcal{S}})-B(\theta_{\text{base}})$，同步追踪 *directional effectiveness*（是否朝预期方向移动）与 *selection advantage*（IF 选择是否显著优于随机选择在同类干预下的表现），并在 SFT 全周期记录轨迹以减少 checkpoint 噪声干扰。

## 实验与结果
- **设置**：SFT 数据采用 Tulu 3 mixture 前 64k 条（固定 seed 与顺序）；模型覆盖 OLMo2-1B、Qwen3.5-2B、Gemma3-4B、OLMo2-7B；默认干预比例 2.5%（k=1,600）。
- **基线**：Random selection 配合同类干预；替代选择器包括 TRAK、Grad Inner Product、Grad Norm、Probing、Loss；abstention 评估基于 AbstentionBench（LLM Judge 判定），核心指标为 Recall，另报告 Precision/Accuracy/F1 及 GSM8K/HellaSwag/ARC 通用能力。
- **主要结果**：
  - Reweighting 轨迹高度不稳定，调整 $\alpha$ 亦无法恢复预期双向效应，部分设置甚至出现反向效果。
  - Response rewriting 效果显著更强且持久：行为对齐改写普遍提升 abstention recall（如 OLMo2-7B 从 Baseline 0.4689 升至 Harmful-Aligned 0.6328，+16.39%），行为对抗改写显著降低 recall；效果随干预剂量 k 单调递增。
  - 选择优势明确：IF 选择在全程重写中持续优于 TRAK、梯度内积、表征投影等其他选择器。
  - 目标特异性验证：增益集中于 attribution target 对应的 answer unknown 与 false premise 场景，非目标场景（subjective/underspecified）增益微弱，证明改写并非无差别增加拒答倾向。
  - 安全拒绝泛化：在 OLMo2-7B 上对齐改写显著提升 WJB-Harmful（0.792→0.880）等多项安全指标，但伴随 XSTest 明显下降，暴露 over-refusal trade-off；重加权仍无系统性效果。
  - 跨模型可迁移性：使用 OLMo2-1B 排名选择的数据在其他模型上重写仍显著优于随机，但幅度受模型家族差异影响（如 Gemma3-4B 的 top/bottom 重合模式与其他模型迥异）。

## 相关工作脉络
- **Influence Functions 与大模型归因（Koh & Liang, 2017; Grosse et al., 2023）**：IF 传统被用于参数更新预测与数据过滤，本文指出其在现代 LLM 权重干预中失效，并将 IF 重新定位为“可操作样本的选择器”而非“干预执行器”。
- **Infusion（Rosser et al., 2026）**：同样利用 IF 选择并扰动训练样本，但聚焦 vision 领域的对抗性 data poisoning，且在长训练下扰动效果衰减；本文将其思路移植至 LLM SFT 语义响应改写，并系统对比 deletion/upweight
