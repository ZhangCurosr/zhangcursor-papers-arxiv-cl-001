---
title: "Beyond-Token-Level-Guidance-Inference-Time-Alignment-of-Spec"
source: https://arxiv.org/pdf/2608.30319v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 01:53:28"
---

# 论文速读：Beyond-Token-Level-Guidance-Inference-Time-Alignment-of-Spec

## 一句话总结
本文针对领域特化微调导致大模型安全性退化的问题，提出 CREST 方法，通过在隐藏表征空间进行跨家族方向引导，从根本上避免了 Token 级干预带来的 EOS 抑制与答案埋没问题，在显著提升安全指标的同时完整保留领域能力。

## 研究问题与动机
1. 领域特化微调（如代码、数学、医疗垂直领域）常伴随灾难性遗忘，导致模型安全对齐水平显著下降，易生成高风险内容。
2. 现有推理时对齐方法在跨家族场景下存在结构性缺陷：通用指导模型与特化基座模型的专长呈“互补正交”，指导信号在领域硬核位置不可靠甚至反向干扰。
3. Token 级方法会对所有位置均匀施加指导分布，导致指导模型的通用续写偏好压制基座模型的停止信号（EOS 抑制），使正确答案被后续错误续写“埋没”。
4. 基于不确定性的触发机制无法拦截特化模型利用领域知识“自信”生成的有害内容，且调整阈值会在安全与能力间造成不可调和的权衡。

## 核心贡献（创新点）
1. 系统识别并形式化了“互补专长正交性”这一根本挑战，揭示了特化模型推理时对齐失效的内在结构瓶颈。该发现区别于以往仅将跨家族引导失败归咎于指导模型选择不当或超参调节的经验性分析。
2. 提出 CREST 表征空间导向框架，通过 Procrustes 对齐将安全方向从任意家族的指导模型投影至基座模型隐空间。该方法本质上跳出了 Token 概率干预范式，从而彻底规避了 EOS 抑制与答案埋没的结构性缺陷。
3. 设计基于探针生成的威胁检测与自适应强度控制机制，替代传统的不确定性触发逻辑。与依赖 max-prob 门限的现有方法不同，该机制直接评估实际生成轨迹的安全距离，可有效拦截特化模型自信输出的高危内容。
4. 构建单次离线预计算结合低开销在线推理的即插即用流程，支持跨家族引导且无需修改权重。相比需要多轮逐 Token 调用指导模型的基线方法，本方案仅增加 1.3× 延迟，大幅提升了工程可用性与部署兼容性。

## 方法详解
CREST 分为**一次性设置（One-Time Setup）**与**逐查询推理（Per-Query Inference）**两个阶段，全程不修改任何模型参数。

**1. 一次性设置**
- **层选择与跨模型映射**：在指导模型 $M_g$ 各层计算安全/不安全提示的隐藏状态均值 L2 距离 $\sigma_l = \|\bar{\mathbf{h}}_l^{\text{safe}} - \bar{\mathbf{h}}_l^{\text{unsafe}}\|_2$，选取 Top-$k$ 层；若基座与指导模型层数不同，按比例映射 $l_b = \lfloor (l_g / L_g) \cdot L_b \rfloor$。
- **表示对齐**：针对每对选定层，基于 $N=100$ 组配对提示的隐藏状态矩阵 $H_b, H_g$，求解正交投影矩阵 $W_l = \arg\min_W \|H_b - H_g W^\top\|_F$，实现跨词表家族的隐空间对齐。
- **检测向量与阈值校准**：在基座模型 $M_b$ 的最后层，对安全/不安全探针输出取隐藏状态均值差并单位化得到检测向量 $\mathbf{v}$；自适应阈值 $\tau$ 设为两类得分分布的中点 $\tau = \frac{1}{2}(\mathbb{E}[s(\mathbf{h}^S)] + \mathbb{E}[s(\mathbf{h}^U)])$，威胁得分 $s(\mathbf{h}) = (1 - \cos(\mathbf{h}, \mathbf{v}))/2$。
- **范数缩放**：记录各层隐藏状态的平均 L2 范数 $\rho_l$，用于统一不同架构的 Steering 强度量纲。

**2. 逐查询推理（三阶段）**
- **Phase 1 探针生成与威胁评估**：无干预生成 $T_p$ 个 Token 作为探针 $x_{\text{probe}}$，计算其末层隐藏状态与 $\mathbf{v}$ 的夹角得分 $s(x)$。
- **Phase 2 自适应 Steering 强度**：$\alpha = \sigma(10 \cdot (s(x) - \tau)) \cdot \alpha_{\max}$。当 $s(x) < \tau$ 时 $\alpha=0$，完全跳过干预以保护正常领域生成。
- **Phase 3 安全方向计算与 Hook 注入**：在 $M_g$ 中计算安全前缀 $p_{\text{safe}} \oplus x$ 与探针 $x_{\text{probe}}$ 的隐藏状态差，归一化得到安全方向 $\mathbf{d}_x$，经 $W_l$ 投影后以 Hook 形式注入基座模型选定层：$\mathbf{h}_l \leftarrow \mathbf{h}_l + \alpha \cdot \rho_l \cdot W_l \mathbf{d}_x$，持续 $T_i$ 个 Token 后移除。该设计确保仅在生成初期建立安全轨迹，随后由基座模型的领域专长接管完成输出。

## 实验与结果
- **数据集与评估**：代码（CyberSecEval↓、HumanEval）、数学（XSTest↑、GSM8K）、医疗（PatientSafetyBench↑、MedQA）。基座模型分别为 Qwen2.5-Coder-7B、Mathstral-7B-v0.1、MedGemma-1.5-4B-it；跨家族指导模型统一使用 Llama-3.1-8B-Instruct。
- **主要结果**：代码不安全率 Base=0.18，Nudging=0.17，BlendIn=0.18，**CREST=0.14**（相对最强基线提升 17.6%，相对 Base 提升 22.2%）；数学安全率 Base=0.92，Nudging=0.95，BlendIn=0.92，**CREST=0.96**；医疗拒绝率 Base=0.99，Nudging=1.00，BlendIn=0.99，**CREST=0.99**（完美保持原有安全水平）。
- **能力保留**：各领域 Capability $\Delta$ 均在 0.00~0.01 范围内，证明 Steering 未损害领域性能。
- **效率**：代码域单查询延迟较 Base 仅增加 1.3×（6.0s vs 4.6s），远低于 Nudging（7.0×）与 BlendIn（2.8×）；内存增量仅约 57MB 的对齐缓存。

## 相关工作脉络
1. **Token 级推理时对齐（Nudging, BlendIn）**：在候选 Token 分布层面进行替换或插值，受限于通用指导模型的领域正交性，易引发 EOS 压制；CREST 转向表征空间，从根本上解耦了语义内容与结构控制信号。
2. **单模型内部激活导向（Rimsky et al., Arditi et al., Zou et al.）**：假设目标模型自身仍编码可靠的安全几何，通过对比激活加法提取方向；CREST 从相反前提出发，承认特化微调已破坏该几何，转而从外部对齐模型移植安全信号。
3. **跨模型激活直传（INFERALIGNER）**：依赖基座与指导模型的层维度与拓扑完全匹配，仅适用于同家族模型；CREST 引入 Procrustes 对齐打破架构壁垒，支持任意家族组合。
4. **微调后安全恢复（SafetyLock）**：需访问模型微调前的祖辈权重；CREST 仅依赖当前特化模型权重与现成通用对齐模型，更贴合实际部署约束。
5. **训练期安全保护（Vaccine, Booster, GradShield, Antidote, L
