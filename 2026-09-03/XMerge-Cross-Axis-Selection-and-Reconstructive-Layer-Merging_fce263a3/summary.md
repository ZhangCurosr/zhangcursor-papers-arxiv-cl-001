---
title: "XMerge-Cross-Axis-Selection-and-Reconstructive-Layer-Merging"
source: https://arxiv.org/pdf/2609.02083v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-07 00:33:49"
field: "大语言模型压缩与效率"
keywords: ["depth compression", "LLM pruning", "layer merging", "post-training reconstruction", "cross-axis selection", "CORE benchmark"]
innovations: ["跨轴选择（RM+BI max融合）无参数选块", "重建合并算子将两块的输出映射梯度拟合进一个现存标准非线性块", "零崩溃鲁棒性：唯一在14个(模型×零/少样本)单元格中无CORE<0.10崩溃的算子"]
benchmarks: ["CORE (22-task aggregate)", "MMLU (zero-shot)", "WikiText-2 perplexity"]
---

# 论文速读：XMerge-Cross-Axis-Selection-and-Reconstructive-Layer-Merging

## 一句话总结
XMERGE 是一种无任务标签、无需端到端微调的深度压缩后训练方法，通过**跨轴选择**识别可安全移除的 Transformer 块，并结合**重建合并算子**将相邻两块映射重构进一个现存标准块，在不改变推理架构和参数的条件下显著减少层数。在 Llama/Qwen 7 个骨干模型、5 个基线、3 个压缩强度下，XMERGE 在激进移除（k=4）时于 CORE 和 MMLU 基准上全面领先，且是唯一零崩溃的操作符。

## 研究问题与动机
- **问题**：Transformer 模型的深度直接决定推理延迟和 KV 缓存流量；现有深度压缩方法（直接丢弃、解析折叠、平均等）在移除整层后质量损失大且波动不可预测。
- **现有方法不足**：
  1. **单纯丢弃（如 ShortGPT）**：不吸收被删块的映射，导致较大精度下降；
  2. **解析合并（LaCo/MKA/SWM/CoMe）**：依赖闭式参数折叠或窗口平均，无法拟合非线性激活；
  3. **模块替换（如 LLM-Streamline）**：引入新模块，破坏标准服务接口；
  4. **线性化（ReplaceMe）**：将两层折叠成线性映射，损失非线性表达能力。

## 核心贡献（创新点）
1. **重建合并算子**：将相邻两块的输出映射用 Adam 优化重写到一个现存标准 Transformer 块中，压缩后仍是普通 L−k 层 Transformer，无新增推理参数。
2. **无参数跨轴选择器**：结合相对幅值轴（RM）与角度轴（BI），以 max 准则融合两轴，仅在两轴均"安静"时选定移除块，降低选错高风险块的概率。
3. **系统评估与新度量视角**：在 7 个骨干、5 个基线、3 个压缩强度下全面对比；首次将 CORE 聚合指标与零/少样本分拆（zero-shot/ICL）崩溃阈值结合，论证 XMERGE 是 14 个 (模型×场景) 单元格中唯一零崩溃的操作符。

## 方法详解
**深度压缩 = 选择 + 合并（𝒞 = ℳ ∘ 𝒮）**，两步解耦。

### 3.1 选择器：跨轴融合（Cross-Axis Selection）
对每一块 f_l 计算其输入→输出变换的两个独立信号：
- **相对幅值轴（RM）**：衡量残差位移大小
  $$\mathrm{RM}_l = \mathbb{E}_{x,t}\left[\frac{\|h_{l+1}-h_l\|_2}{\|h_l\|_2+\epsilon}\right]$$
- **角度轴（BI，Block-Influence）**：衡量表示方向的旋转程度
  $$\mathrm{BI}_l = \mathbb{E}_{x,t}\left[1-\frac{\langle h_l, h_{l+1}\rangle}{\|h_l\|\|h_{l+1}\|+\epsilon}\right]$$
  将两轴标准化后用 **max 融合**：$s_l = \max(z_l^{\mathrm{RM}}, z_l^{\mathrm{BI}})$，选中得分最低的候选块。max 比平均更安全——避免单轴高得分被另一轴掩盖。

### 3.2 算子：重建深度坍缩（Reconstructive Depth Collapse）
选定被删块 f_ℓ 后，与相邻块 f_{ℓ+1} 配对，保留一侧为"存活块" $\tilde{f}_\ell$（按相邻激活修补冗余度 S_patch 决定方向），用原预训练权重初始化，然后优化全部参数以最小化原始两块的输出 MSE：
$$\min_{\tilde{\theta}_\ell} \mathbb{E}_x\left[\|\tilde{f}_\ell(h_\ell(x);\tilde{\theta}_\ell) - f_{\ell+1}(f_\ell(h_\ell(x)))\|_2^2\right]$$
- 优化器：Adam，300 步，lr=1e−5，batch=16，128 条 WikiText-2 序列（512 tokens）。
- 无任务标签、无端到端微调、无新增推理参数。
- 迭代压缩（k>1）：选择顺序一次性从原模型计算；每步合并后刷新目标激活。

## 实验与结果
- **数据集与骨干**：7 个 Llama/Qwen 模型（0.5B–8B），WikiText-2 校准；评估指标 CORE（22 任务聚合）、MMLU、WikiText-2 PPL。
- **基线**：ShortGPT（丢弃）、LaCo、MKA、SWM、CoMe（合并/折叠）。
- **主要结果（k=4，表 3）**：
  - **CORE**：XMERGE 在 7 个骨干中**6 个第一**（仅 Qwen3-1.7B 被 MKA 超越 −0.018）。
  - **MMLU**：同样**6 个第一**（仅 Qwen3-0.6B 差 0.001）。
  - **PPL**：XMERGE 在 6/7 上低于 LaCo（LaCo 在各行都是最低 PPL 基线）。
  - **崩溃规避**：其他 4 个算子在 k=4 时 PPL 出现 10²–10⁴ 量级爆炸（如 MKA 在 Llama-3.2-1B 达 8343，CoMe 在 Qwen3-1.7B 达 5.3×10⁴），XMERGE 与 LaCo 是唯一避免极端崩溃的算子。
- **Bootstrap 置信区间**：3 个最大 CORE 边距（Llama-3.2-1B +0.070、Llama-3.2-3B +0.033、Qwen3-8B +0.048）的 95% CI 排除零；其余边距为"平局"，单侧符号检验 p=0.06。
- **零崩溃鲁棒性**（表 5）：XMERGE 是唯一在 14 个 (模型×zero-shot/ICL) 单元格中**从不崩溃**（CORE ≥ 0.10）的算子；其他基线崩溃 1–7 次。
- **校准**：在 Llama-3-8B 上，XMERGE 的 ECE 恶化仅 +0.010，为最优。

## 相关工作脉络
1. **ShortGPT**（Men et al. 2025）：BI 角度轴引导的单层丢弃；XMERGE 借鉴 BI 作为融合轴之一，但用重建吸收而非丢弃。
2. **LaCo / MKA / SWM / CoMe**（Yang 2024; Liu 2024; Ding 2026; Wang 2025）：解析折叠、窗口平均、头组拼接等，均无法拟合非线性激活；XMERGE 与之的本质区别是**梯度拟合整个非线性块**。
3. **ReplaceMe**（Shopkhoev et al. 2025）：最接近的无模块邻居，但线性化被删块（折入存活权重）；XMERGE 保留非线性。
4. **LLM-Streamline**（Chen et al. 2025）： learned-replacement 添加新模块；XMERGE 不需新模块。
5. **Prune&Comp**（Chen et al. 2026）：研究隐藏状态幅值差；本文的 RM 轴与其一脉相承。
6. **Post-pruning 重建范式**（Wagner 2025; van der Ouderaa 2024）：本文将其实例化为**服务接口保真**版本。

## 局限性与未来方向
- **构建成本**：一次压缩耗时数分钟到 4.4 小时（线性于 k），虽在约 1.9k–24k 请求后可回本，但仍高于免训练基线。
- **单种子**：仅用种子 42，未测量重建种子方差；需多种子研究验证稳定性。
- **仅覆盖 Dense Decoder-only**：未涉及 MoE、Encoder-Decoder、多模态、状态空间模型；8B 上限源于单卡 40GB MIG 预算。
- **校准仅在一骨干验证**：ECE 提升结论推广性有限；幻觉/拒答等安全维度未测。
- **重建目标与评估指标同源**：CORE/MMLU/PPL 部分衡量与重建目标（MSE 到稠密激活）的接近度，非完全独立验证。
- **未消融合并方向启发式**（S_patch）；**WikiText-2 训练/测试部分同域**，PPL 略有偏差。

## 研究启发与可借鉴点
1. **选择-合并解耦框架**：将失败模式拆分为"选错块"与"合并操作差"两类，便于分别分析、诊断，可迁移到其他结构压缩（宽度、稀疏）任务。
2. **双轴 max 融合代替加权平均**：无超参的 robust 选择策略；对重要性评分多源融合有参考意义。
3. **重建保留标准服务接口**：所有压缩后模型仍是纯 L−k Transformer，兼容任何生产 serving 栈；对比引入 LoRA/新模块的方法更具工程友好性。
4. **零崩溃评测协议**：以中心化 CORE < 0.10 为崩溃阈值，跨 zero-shot/ICL 双场景统计崩溃次数，可作为压缩方法的鲁棒性通用基准。
5. **构建-服务成本摊销分析**（§L）：给出一次性构建时间与 per-token 节省的盈亏平衡公式，可复用于评估其他"建设时重、推理时轻"压缩策略。

## 关键术语表
**Depth Compression（深度压缩）**：移除整个 Transformer 块以缩减模型深度，保留隐藏维度与推理接口。
**Cross-axis Selection（跨轴选择）**：同时用相对幅值（RM）和角度（BI）两个正交信号选块，取 max 防止单轴误判。
**Block-Influence（BI）**：ShortGPT 提出的角度变化指标，衡量层对表示方向的旋转程度。
**Reconstructive Merge（重建合并）**：用 Adam 在 300 步内把两块的输出映射拟合进一个现存标准块。
**CORE Metric**：22 任务零样本/少样本聚合指标，比 MMLU 更敏感，适合深度压缩评估。
**Setting A / Setting B**：A 仅比较构造算子；B 给所有有恢复签点的算子相同的后压缩 KD 预算进行公平对比。
**Regime Collapse（场景崩溃）**：某 (模型, 场景) 下中心化 CORE < 0.10，视为完全失效。
**Patch-based Redundancy Score S_patch**：通过逐层修复激活的因果 patching 估算相邻块冗余度，用于决定合并方向。

## 可复现要素
- **数据集**：WikiText-2（校准/重建用）、MMLU、CORE（nanochat bundle）。
- **代码/权重**：论文未提供开源链接；基线引用作者官方实现（§G 给出审计说明）。
- **关键超参**：300 步、lr=1e−5、batch=16、128 条 512-token 序列；统一用于所有模型；单种子 42。
- **硬件**：NVIDIA A100-SXM4-80GB（MIG slice 3g.40gb），float16。
