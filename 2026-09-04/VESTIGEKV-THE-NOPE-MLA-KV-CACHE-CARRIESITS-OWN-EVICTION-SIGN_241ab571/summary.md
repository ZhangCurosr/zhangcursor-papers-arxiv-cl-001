---
title: "VESTIGEKV-THE-NOPE-MLA-KV-CACHE-CARRIESITS-OWN-EVICTION-SIGN"
source: https://arxiv.org/pdf/2609.03949v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-07 23:13:50"
---

# 论文速读：VESTIGEKV-THE-NOPE-MLA-KV-CACHE-CARRIESITS-OWN-EVICTION-SIGN

## 一句话总结
提出 VestigeKV，利用 Kimi Linear NoPE-MLA 架构中废弃的 64 维位置分支作为查询无关的 token 重要性信号，实现零训练/零权重的 KV Cache 分区与可召回驱逐；在 8k–65k 上下文、最高 128× 压缩比下保持完全 needle 检索，并从数学上证明 RoPE 架构下此类信号因旋转轨道膨胀而无法存在。

## 研究问题与动机
- **压缩早于查询的信息盲区**：长上下文 KV Cache 必须在生成前压缩，H2O/SnapKV 等依赖已观测注意力或最近窗口的方法在此设定下完全失效（needle 检索率跌至 0.00–0.33）。
- **架构残迹的信号潜力未被挖掘**：NoPE-MLA 解除了旋转模块，原 64 维解耦分支不再承担位置编码任务，训练后该通道自发继承 salience 语义，但尚无方法将其转化为缓存管理信号。
- **RoPE 下排名随时间老化**：旋转使注意力分数依赖查询位置，预压缩阶段冻结的排序在生成过程中随步数振荡失效，驱逐决策无法一次性固化。
- **可召回驱逐的摘要松散**：现有 Recall 方案（如 ArkVale）依赖包围球半径估计，RoPE 下旋转轨道使半径膨胀至快频分量量级，紧致上界在理论上不可达。

## 核心贡献（创新点）
- **残迹分支查询无关信号**：仅读取每行缓存 11% 维度即可获得稳定 token 重要性排序，无需观测任何历史注意力。
- **固定双线性形式的理论分割**：证明 NoPE-MLA 注意力输出为缓存行多重集的对称函数（Lemma 1），且精确合并被证明不可能（Proposition 1），强制选择而非压缩合并。
- **带证书的回叫层设计**：基于精确分支和、内容低秩草图与残差范数构建索引，导出查询均匀的误差上界（Lemma 2），使触发召回具有数学保证（Lemma 3）。
- **零侵入工程落地**：纯缓存策略，不修改权重、算子与推理内核；KV 路径读带宽降低 4.0×（576→~145 dims/token），推理路径延迟仅增加 <1%。
- **RoPE/NoPE 适用域的形式化对比**：严格推导旋转如何使排名失效、摘要松散、精确合并消失，解释既往基线在无查询先验时的崩溃根源。

## 方法详解
- **信号提取与压缩分区**：每 token 每层缓存行 $\tilde{C}_u = [\hat{c}_u; r_u]$（512 维内容 + 64 维分支）。对分支序列做 FFT 低通滤波后取残差范数 $\sigma_u = \|r_u - \text{low-pass}(r)_u\|$ 作为 row-intrinsic 评分；保留 $\sigma$-top-m 行在 attended tier，其余行精确迁移至 GPU 常驻 archive，**不删除任何行**。
- **排名稳定性**：由于 NoPE 下分数 $s_h(t,u) = \lambda \langle Q_t^h, \tilde{C}_u \rangle$ 不含步数 $t$ 与行年龄，一旦计算完毕 $\sigma$ 排序对所有未来步有效（Corollary 1），驱逐决定一次固化，无需重评分。
- **召回层索引与触发生成**：归档行维护三元组 $(r_u,\ V_r^\top \hat{c}_u,\ \eta_u)$。每解码步计算代理分数：
  $\text{score}(t,u) = \underbrace{\lambda q_t^{r\top} r_u}_{\text{精确分支和}} + \underbrace{\lambda (V_r^\top q_t^{c'})^\top (V_r^\top \hat{c}_u)}_{\text{内容草图}} + z \underbrace{\lambda \|(I-V_rV_r^\top)q_t^{c'}\|\eta_u/\sqrt{d_c-r}}_{\text{证书缩放}}$
  超过 attended tier 最大分数的 top-j 行被召回参与 softmax。$z$ 与 $V_r$ 均在当前上下文 prefix 内自校准，无需外部语料。
- **误差界与召回完备性**：Lemma 2 给出 $|\Delta s| \le \lambda \|Q\|\varepsilon_u$ 的查询均匀界；Lemma 3 证明仅召回分数竞争性行时，被排除 softmax 权重有界（$\le (T-|F|)e^{-\tau}$），输出误差可控。
- **RoPE 障碍的代数根源**：旋转使 $\|R_u k - R_v k\|^2 > 0$，所有行 pairwise distinct 导致合并类为空；同时摘要
