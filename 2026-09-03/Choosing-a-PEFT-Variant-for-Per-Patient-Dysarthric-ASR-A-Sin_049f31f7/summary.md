---
title: "Choosing-a-PEFT-Variant-for-Per-Patient-Dysarthric-ASR-A-Sin"
source: https://arxiv.org/pdf/2609.02735v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 16:44:59"
---

# 论文速读：Choosing-a-PEFT-Variant-for-Per-Patient-Dysarthric-ASR-A-Sin

## 一句话总结
本文针对临床构音障碍语音识别的 per-patient adapter 生产架构，首次在相同说话人、相同语料、固定训练配方下系统横评了七种 LoRA 家族 PEFT 变体，证实 LoRA 与 DoRA 性能统计持平且显著优于真实 4-bit QLoRA 及其他变体，推荐以 LoRA r=16 作为单患者适配默认方案，并给出了可复现的生产级 recipe 与 enrollment 时长收益曲线。

## 研究问题与动机
1. 临床构音障碍 ASR 部署普遍采用 per-patient adapter 架构（每位患者独立训练小型适配器，共享底座不变），但 PEFT 变体在说话人相关（speaker-dependent）、小数据、单患者场景下的最优选择尚未被系统性对比。
2. 现有 dysarthric PEFT 文献多关注 speaker-disjoint 条件或不同适配机制（如 adapter fusion、FiLM、prompt-tuning），结论无法直接迁移至 per-patient 场景，因为 speaker overlap 是该场景的核心特征。
3. 实际临床部署需严格权衡适配器体积、训练耗时、显存占用与识别精度；生产环境中“每个 MB 与每分钟训练成本都随患者数线性增长”，亟需明确的工程决策依据。
4. 希望输出一套固定配方下的横评结果与开源脚本，回答“在稀疏单患者数据下默认选用哪种低秩适配器”这一生产落地问题。

## 核心贡献（创新点）
1. 首次开展说话人相关、per-patient 模式下的七种 LoRA 家族变体横评，填补了单患者构音障碍适配的 PEFT 对比空白。
2. 首次在两款架构迥异的生产底座（encoder-decoder Transformer 的 Whisper-large-v3 + HU FT，与 LLM-decoder 的 Qwen3-ASR-1.7B）上进行同说话人、同数据、同配方的跨架构对比。
3. 提出并公开两项负结果方法论分析：Qwen3 底座上匈牙利语 warm-base 微调引发的 dysarthric CER 回归现象，及其由预训练混合分布过窄（而非健康对照内容）导致的归因结论。
4. 构建 6 点 enrollment 时长网格，首次刻画了 LoRA 家族在固定 per-patient recipe 下的音频采集收益曲线（~5 min 捕获 45.6% 的零样本到 30 min 总 CER 降幅）。
5. 发布 source-available 的 per-patient PEFT 训练脚本与 recipe，明确推荐 LoRA r=16 为生产默认方案，并提供 DoRA 作为无损替代品。

## 方法详解
- **数据与说话人**：1 名卒中后匈牙利男性患者 S1（重度构音障碍，auditory-perceptual 临床评估），409 条语音（55 min）。按内容划分：训练 262 条（195 read-sentence + 67 narrative）、验证 40 条、测试 107 条；测试集与训练/验证集无文本重叠（text-disjoint）。
- **底座模型**：① Whisper-large-v3（1.55B encoder-decoder Transformer），合并 38K-utterance 匈牙利语微调权重（HU FT）；② Qwen3-ASR-1.7B（内部多语言生产 checkpoint，基于 1.7B AuT encoder + Qwen LLM decoder，在含 TORGO/UASPEECH/EasyCall 等 10 种语言的构音障碍语料池上微调）。
- **PEFT 变体与目标模块**：基于 HuggingFace PEFT 库，统一锁定 attention projections 为目标层。Whisper 选取 q/k/v/out_proj（encoder self-attention + decoder self/cross-attention，共 384 处）；Qwen3 选取 LLM decoder 的 q/k/v/o 及 audio-encoder 投影层（out_proj, conv_out, proj1, proj2）。七种变体参数如下：LoRA (r=16, α=32)、DoRA (r=16)、QLoRA (r=16, 真实 4-bit NF4 冻结基座 + double quantisation + paged optimiser)、LoHA (r=8, 等效秩 ~64)、AdaLoRA (init_r=24 → target_r=16)、VeRA (r=256)、VB-LoRA (r=16, vector_bank=256)。
- **训练配方**：AdamW (weight decay=0), lr=1×10⁻⁴, bf16, batch=4 × grad_acc=4, 固定 5 epochs（约 80 步优化器
