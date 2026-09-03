---
title: "Quantization-Efects-on-Bangla-Language-Understanding-in-Larg"
source: https://arxiv.org/pdf/2608.24615v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 18:36:51"
field: "低资源语言大模型量化评估"
keywords: ["post-training quantization", "Bangla NLU", "low-resource language", "GPTQ", "GGUF", "multilingual LLM", "model compression"]
innovations: ["首次控制变量对比三种 LLM 家族在五种 Bangla NLU 基准上的 PTQ 效应", "揭示量化鲁棒性由架构与格式主导而非语言本身，GPTQ 下 Qwen/LLaMA 退化<1.5%", "在形态复杂低资源语言中验证推理任务比阅读理解对量化更敏感跨语言普适性"]
benchmarks: ["Bangla MMLU", "CommonsenseQA-BN", "OpenBookQA-BN", "PIQA-BN", "BoolQ-BN"]
---

# 论文速读：Quantization-Effects-on-Bangla-Language-Understanding-in-Large-Language-Models

## 一句话总结
本文首次系统评估了后训练量化（PTQ）对低资源形态复杂语言孟加拉语自然语言理解性能的影响，发现量化鲁棒性主要由模型架构与量化格式决定，而非语言本身：GPTQ 格式下的 Qwen 与 LLaMA 在五大 Bangla NLU 基准上近乎无损，而 GGUF-W8A16 格式下的 GPT-OSS-20B 在推理类任务上精度损失高达 57.4%。

## 研究问题与动机
- **核心问题**：现有 PTQ 评估几乎全部基于英语基准，缺乏对形态复杂、低资源语言（如孟加拉语）的系统考察，量化误差是否会因孟加拉语的黏着 morphology、辅音连字集群及 Unicode 字符组合规则而累积放大，尚不清楚。
- **现有方法不足**：Prior work (Jin 2024; Li 2024; Ding 2024) 聚焦 INT4/INT8 在 MMLU、GSM8K 等英语任务的退化规律，未覆盖低资源语言；TituLLMs 强调 tokenizer 与语料质量是 Bangla NLU 性能瓶颈，但未验证模型压缩是否会侵蚀该优势。
- **实践缺口**：南亚等地算力受限，部署端侧 Bangla NLP 需量化，但架构/格式选择缺乏实证指引。

## 核心贡献（创新点）
- **首创 Bangla PTQ 对照评估**：在 15 个 model–benchmark 对上首次控制变量（仅变精度，固定 prompt/解码/harness），覆盖三种主流模型家族与三种量化格式，填补低资源语言 PTQ 评测空白。
- **揭示架构主导的鲁棒性分化**：GPTQ-Int8/Q8 使 Qwen-2.5-7B 与 LLaMA-3.1-8B 在五项基准上退化均 <1.5%，部分任务甚至反升；GGUF-W8A16 使 GPT-OSS-20B 在 CommonsenseQA-BN 上损失 57.4%，证明格式与架构比位宽更关键。
- **验证任务敏感性跨语言普适**：推理/常识类任务（MMLU、CSQA、OBQ）比阅读理解（BoolQ-BN）对量化更脆弱，该模式在 Bangla 中同样成立，首次将英语观测扩展至形态复杂低资源语言。
- **提供可落地的部署建议**：给出针对 Bangla NLU 的三种具体推荐——优先选用 GPTQ-Int8/Q8、避免在推理场景使用 GGUF-W8A16、纯 comprehension 场景可放宽格式选择。

## 方法详解
- **基准与任务划分**：选用 five 公开 Bangla NLU 基准（均来自 Hugging Face hishab 组织），按任务类别分为 reasoning（Bangla MMLU、OpenBookQA-BN）、commonsense（CommonsenseQA-BN、PIQA-BN）、comprehension（BoolQ-BN），共覆盖 497–1840 条样本不等。
- **模型与量化变体**：三组 family × two precision 对照：
  - Qwen2.5-7B-Instruct (FP16) vs GPTQ-Int8
  - Meta-Llama-3.1-8B-Instruct (FP16/BF16) vs GPTQ-Q8
  - GPT-OSS-20B (FP16) vs GGUF-W8A16（经 llama.cpp 加载）
- **评估流水线**：使用 EleutherAI `lm-evaluation-harness`，zero-shot 模式，greedy log-likelihood 选答；全精度通过 `transformers` 加载，量化版通过 `auto-gptq` 或 `llama.cpp` 加载；固定随机种子与解码策略以消除采样方差。
- **度量公式**：主指标为 accuracy；退化用绝对差 Δ = Acc_full − Acc_quant 与相对退化率 Δ% = Δ/Acc_full × 100% 报告，负值（量化优于全精度）保留原符号并在讨论中解释。
- **复现保障**：随机种子固定、greedy 解码，重跑方差 <0.3%；评估在 NVIDIA A100/V100 云 GPU 完成，总算力约 40 GPU-hours；结构化 JSON 日志与代码将随论文开源。

## 实验与结果
- **全精度基线**：Qwen-2.5-7B 在 Bangla MMLU 上最高（0.440）；LLaMA-3.1-8B 在 BoolQ-BN 上达 0.914（显著高于参数更多的 GPT-OSS 的 0.539）；GPT-OSS-20B 在 OBQ（0.584）与 PIQA（0.641）领先；三族无一家通吃，验证基线具备可比 Bangla NLU 能力。
- **量化后性能**：
  - GPT-OSS (GGUF-W8A16)：全维度下跌，CSQA 从 0.440→0.188（−57.4%），OBQ 从 0.584→0.268（−54.1%），BoolQ 仅降 5.6%。
  - LLaMA (GPTQ-Q8)：五基准绝对退化均 ≤0.025（最大 MMLU −7.0%），OBQ 零退化，PIQA 微升 +0.4%。
  - Qwen (GPTQ-Int8)：五基准退化均 <1.5%，MMLU/CSQA/PIQA 分别 +0.6%/+0.5%/+0.1%，BoolQ 持平。
- **关键结论**：最强结果为 Qwen-GPTQ-Int8（几乎无损）与 LLaMA-GPTQ-Q8（<1% 退化）；最大退化为 GPT-OSS-GGUF-W8A16 在 CSQA 上 57.4%；BoolQ-BN 为唯一跨 family 稳定的任务。

## 相关工作脉络
- **Multilingual LLM / Low-Resource**：Bhowmik 2025、Nahin 2025 (TituLLMs) 指出 Bangla 受 token 稀疏与形态复杂制约；本文定位差异在于首次追问“即使预训练与 tokenizer 已优化，后训练压缩是否仍会侵蚀 Bangla 性能”，填补 PTQ×低资源语言交叉空白。
- **PTQ 方法学**：GPTQ (Frantar 2023)、AWQ (Liu 2025)、GGUF 系列；本文与 Ding 2024、Jin 2024 的关键差异是评测语言从英语转为 Bangla，且覆盖 zero-shot NLU 而非仅 code/math。
- **Quantized LLM 评估**：Li 2024、Jin 2024 发现 INT4 损害复杂推理、INT8 安全；本文复现该 pattern 于 Bangla，并新增“format 比 bit-width 更重要”与“架构鲁棒性分化”的发现。
- **TituLLMs 呼应**：Nahin 2025 主张 tokenizer/corpus 主导 Bangla NLU；本文与之衔接但未直接对比，明确声明因未纳入 TituLLMs 量化版而无法判断语言特异性预训练能否增强压缩韧性——为后续工作留出接口。

## 局限性与未来方向
- **格式/尺寸/架构未完全正交**：GPT-OSS 是唯一使用 GGUF、唯一 20B、唯一缺少 GPTQ 对照的 family，导致“架构效应 vs 格式效应 vs 尺寸效应”无法彻底分离。
- **仅覆盖 INT8 区间**：未评估 INT4（AWQ、Q4_K_M 等），高精度压缩下的 Bangla 行为未知。
- **评估场景单一**：仅 zero-shot classification-style 任务，few-shot/fine-tuned/generative（summarization、translation、dialogue）未测。
- **缺失延迟与显存实测**：仅报告精度，未报 inference latency、peak memory，工程落地指标不完整。
- **校准数据不透明**：GPT-OSS GGUF 的 calibration corpus 未公开，可能引入英语偏向混杂 Bangla 特有结论。
- **未来方向**：① INT4 区间扫描；② 量化 Bangla-native 模型（如 TituLLMs）验证语言特异性预训练收益；③ Bangla PTQ-aware training；④ 延迟/显存 benchmark；⑤ 扩展至 generative Bangla 任务。

## 研究启发与可借鉴点
- **对照设计范式**：family × precision 严格单因子控制（固定 prompt/harness/seed），是低资源语言 PTQ 评估的可复用模板，可直接迁移至其他南亚/形态复杂语言（印地语、泰米尔语等）。
- **任务敏感性分层结论**：BoolQ-style comprehension 对量化天然稳健，可作部署快速 sanity check；推理/常识类任务需重点监控，建议将此类基准纳入 Bangla 端侧部署门禁。
- **负退化现象的工程解读**：GPTQ Hessian 重建偶有“平滑 FP16 overfitting/数值不稳定”的正则化效应，Qwen 的 <1.5% 反升应视为鲁棒性信号而非真实增益，评估报告中宜附注避免误读。
- **架构选择优先于格式选择**：部署资源受限时，先锁定 GPTQ 兼容的高鲁棒架构（Qwen/LLaMA 系列），再调格式位宽，比盲目追求更低 bit-width 更有效。
- **可复现资产复用**：lm-evaluation-harness zero-shot + greedy log-likelihood pipeline、JSON 日志结构、40 GPU-h 量级预算设计，均可直接移植至同类多语言 PTQ 评测流程。

## 关键术语表
- **Post-Training Quantization (PTQ)**：模型训练完成后直接压缩权重精度（如 FP16→INT8），无需重训练即可降低内存与加速推理。
- **GPTQ-Int8 / GPTQ-Q8**：基于 GPTQ 算法的 8 比特量化变体，利用近似二阶信息最小化逐层重构误差，分别以 tensor-wise 或 grouped-wise 方式量化。
- **GGUF-W8A16**：llama.cpp 序列化格式，权重 8-bit、激活 16-bit，面向 CPU/GPU 端侧推理优化，但未使用 GPTQ 校准。
- **Bangla MMLU / CommonsenseQA-BN / OpenBookQA-BN / PIQA-BN / BoolQ-BN**：五类由英语基准翻译而来的孟加拉语 NLU 评测集，分别覆盖百科全书推理、常识推理、多步科学推理、物理程序常识、段落阅读理解。
- **lm-evaluation-harness**：EleutherAI 维护的标准化 LLM 评测工具包，支持 zero-shot/few-shot、greedy log-likelihood 选答，被 OpenAI/Meta 及学界广泛采用。
- **Δ 与 Δ%**：绝对退化与相对退化率，用于量化 PTQ 对 accuracy 的影响，负值表示量化后反而提升。
- **Calibration set**：GPTQ 等二阶量化方法用于估计 Hessian 或校正权重的校准数据集，其语言构成直接影响跨语言迁移性能。
- **TituLLMs**：Nahin 2025 提出的孟加拉语原生 decoder-only 模型族，使用 Bangla-aware subword tokenizer，本文指出其量化鲁棒性尚待验证。

## 可复现要素
- **数据集**：五类 Bangla NLU 基准（hishab/bangla-mmlu、commonsenseqa-bn、openbookqa-bn、piqa-bn、boolq_bn）均在 Hugging Face 公开。
- **代码**：论文声明结构化 JSON 日志与代码将随论文发布（Section 3.3 "will be released alongside the code"）。
- **权重**：三款基线模型（Qwen2.5-7B-Instruct、Meta-Llama-3.1-8B-Instruct、gpt-oss-20b）及三款量化 checkpoint 均为公开开源权重。
- **关键超参**：zero-shot、greedy log-likelihood 选答、固定随机种子；环境在 NVIDIA A100/V100 云 GPU；总算力约 40 GPU-hours。
- **评估框架**：lm-evaluation-harness；全精度加载 via transformers，量化版 via auto-gptq 或 llama.cpp。
