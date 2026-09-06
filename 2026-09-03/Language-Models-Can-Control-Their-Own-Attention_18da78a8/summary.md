---
title: "Language-Models-Can-Control-Their-Own-Attention"
source: https://arxiv.org/pdf/2609.02737v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 21:02:24"
field: "长上下文高效推理"
keywords: ["Declarative Attention", "long-context inference", "sparse attention", "KV cache masking", "zero-shot protocol", "system-2 reasoning"]
innovations: ["提出DA协议，让模型通过声明性标记控制注意力范围", "零样本elicited注意力控制，无需训练", "块对齐动态掩码与vLLM集成"]
benchmarks: ["RULER", "LongBench v1/v2", "LooGLE", "ZeroSCROLLS"]
---

# 论文速读：Language-Models-Can-Control-Their-Own-Attention

## 一句话总结
论文提出 **Declarative Attention（DA）**，一种零样本协议，让语言模型在链式思考中显式声明自己的注意力范围，通过三种模式（`<global>`、`<focus>`、`<local>`）将生成过程分区，推理引擎解析这些声明并动态构建注意力掩码，从而跳过大部分KV cache读取，显著降低长上下文解码成本而几乎不损失准确率。

## 研究问题与动机
1. **长上下文解码成本高**：Transformer每次解码都需要读取整个KV cache，但在100K-token级别上下文中，90%以上的注意力权重集中在极少数token上，却仍需付出O(N)的内存访问代价。
2. **现有稀疏注意力方法的局限**：当前主流做法（如QUEST、H2O、DeepSeek索引器等）是通过轻量级代理分数每步扫描KV cache来预选相关token，虽降低了常数因子，但每步复杂度仍为O(N)，且引入额外计算开销。
3. **动机：让模型自己“说”出需要关注什么**：既然模型在隐藏状态中已编码了未来token的信息，且链式思考能外显这种隐含计算，能否直接让模型用自然语言声明其注意力计划？这与系统2推理的思想一脉相承——将内部不可观察的选择过程转化为可解析、可干预的外部指令。

## 核心贡献（创新点）
1. **提出Declarative Attention协议**：通过固定提示让模型在生成时交替使用`<global>`（全局扫描）、`<focus>`（聚焦特定段落）、`<local>`（仅看自身已生成文本）三种模式，将注意力计划显式化为可解析的文本标记。
2. **零样本即有效，无需训练或微调**：在Gemma-4-31B和Qwen-3.6-27B等现成模型上，仅通过提示工程即可 elicited 该行为，Across 15个长上下文任务验证其有效性。
3. **推理引擎层面的动态掩码构建**：设计DA状态机，在解码过程中监听模式标记转换，实时重写vLLM的KV cache块表，实现块对齐的硬掩码，跳过大部分KV读取。
4. **揭示模型规模与DA性能的缩放关系**：随着模型从4B扩展到31B，DA的准确率差距持续缩小（相对准确率从29%提升至99%），证明该协议能充分利用模型能力。
5. **提供roofline墙钟时间分析框架**：将DA节省转化为理论上的解码延迟改进（Gemma-4-31B降至0.71×，Qwen-3.6-27B降至0.77×），为长上下文 serving 的效率优化提供量化依据。

## 方法详解
**1. 三种注意力模式**
- `<global>`：模型扫描全部上下文片段，用于定位相关段落并简要说明理由。每步仍需读取完整KV cache。
- `<focus magic_chunks="K">`：模型仅查看编号为K的特定上下文片段，提取事实值。每步只需读取该片段对应的KV块。
- `<local>`：模型仅基于已生成的回复内容进行推理，完全不接触上下文。每步只读取自身已生成token的KV。

**2. Prompt结构设计**
- **脚手架区域（始终可见）**：包括系统指令（占位attention sink）、问题、DA指令说明。
- **上下文区域（动态可见）**：长文本被分割为约2048-token的“magic chunk”，以模拟工具调用转录的形式呈现（assistant调用`get_magic_chunk`，tool response返回片段内容），但实际不执行任何工具调用。

**3. 上下文分段策略**
- 按语义边界分层切割：段空行→单换行→句子结束（.!?后跟空格）→分句结束（;:,后跟空格）→词边界。
- 保证切分点不落在单词中间，且拼接后可无损还原原文。

**4. 解码时干预机制**
- **状态机解析**：默认处于`<global>`模式，遇到`<focus>`或`<local>`标签时切换，遇到对应闭合标签时恢复。
- **块对齐掩码**：vLLM以固定大小块（通常16-32 token）管理KV cache，掩码必须整块跳过才能节省内存读取。状态机将保留的token区间向上取整到块边界，每个保留区间边缘最多增加一个额外块（几十token vs 2048-token片段）。
- **vLLM集成**：通过hook attention metadata builder重写请求的KV cache块表，不修改底层kernel，兼容FlashAttention等现有高效算子。

**5. 作用范围**
- 仅应用于全局注意力层（global attention layers），不影响滑动窗口注意力（SWA）或Gated DeltaNet（GDN）等高效层，因为这些层的每步成本已与上下文长度无关。

## 实验与结果
**1. 实验设置**
- **模型**：Gemma-4系列（E4B、12B、31B）和Qwen-3.5/3.6系列（4B、9B、27B），上下文上限244K（Gemma-4-E4B为116K）。
- **基准**：15个长上下文来源，涵盖RULER、LongBench v1/v2、LooGLE、ZeroSCROLLS，分为单段检索/推理和多段推理两类。
- **基线**：Vanilla（完整因果注意力）、DA-nm（使用DA提示但无自定义掩码，用于隔离掩码效果）。
- **评估指标**：准确率（LLM-judge评分）、注意力token总数、roofline墙钟时间。

**2. 主要结果**
- **准确率变化微小**：Gemma-4-31B从87.01%降至85.74%（-1.27pp），Qwen-3.6-27B从85.31%降至82.56%（-2.75pp）。在15个任务中有7个（Gemma）和5个（Qwen）持平或提升。
- **注意力token显著减少**：Gemma-4-31B平均减少52.0%（13.43M→6.45M），Qwen-3.6-27B减少31.1%（22.54M→15.52M）。最长上下文任务（如code_repo、dialogue_history）绝对节省可达21M-52M token。
- **效率来源隔离**：DA-nm相比Vanilla注意力token增加66.2%（Gemma）/28.8%（Qwen），说明DA协议本身导致生成步骤增多（+15-35%）；但掩码将这一开销逆转，使DA相对DA-nm再减少71.1%（Gemma）/46.5%（Qwen）的token读取。
- **缩放趋势**：相对准确率随模型规模单调上升（Gemma从29%→99%，Qwen从64%→97%）；相对注意力token在多数模型上稳定在Vanilla的46%-69%。
- **上下文长度效应**：DA在32K token内准确率与Vanilla基本持平，更长上下文略有下降；绝对token节省随上下文线性增长（从~1M到~21M）。
- **墙钟时间估计**：基于B200 GPU的roofline分析，DA将解码墙钟时间降至Vanilla的0.71×（Gemma）和0.77×（Qwen），节省全部来自全局注意力KV读取的降低。

**3. 最强结果与提升**
- 最大绝对节省：LBv2/code_repo任务，Gemma-4-31B节省41.8M token，Qwen-3.6-27B节省52.0M token。
- 最大相对节省：Gemma-4-31B在多数任务上达~50-60%的token减少。
- 唯一显著提升：LooGLE/longdep_qa在Gemma上+3.1pp，LBv2/code_repo在Qwen上+5.6pp。

## 相关工作脉络
1. **动态稀疏注意力（QUEST、H2O、SnapKV、DeepSeek索引器等）**：这些方法每步通过轻量扫描预测重要token并应用掩码，仍保持O(N)每步复杂度；DA通过模型声明直接获取掩码，消除扫描开销，且选择过程可解释、可干预。
2. **KV缓存驱逐（Scissorhands、FastGen等）**：永久丢弃不重要的KV条目以释放内存；DA采用可逆硬掩码，保留完整缓存，允许后续步骤重新访问被跳过的内容，避免信息丢失风险。
3. **模型控制的推理与自生成控制token（ReAct、Self-RAG、MemGPT等）**：这些工作让模型生成动作、反思或记忆调用token来影响生成流程；DA将这一范式扩展到控制“读什么”而非“写什么”，改变的是KV cache可见性而非token序列本身。
4. **自声明与受控注意力（SSAS、PASTA、Attention-Gate等）**：SSAS训练模型在输出中命名所需上下文跨度并硬掩码；DA零样本 elicited，无需任务特定训练。PASTA/Attention-Gate通过重新加权或学习门控实现注意力引导，但不减少KV读取量。
5. **系统2注意力概念**：DA被定位为“系统2稀疏注意力”——模型通过显式推理决定关注点，而非系统1式的基于激活的快速推断；这为可审计的注意力控制提供了新途径。

## 局限性与未来方向
1. **零样本策略次优**：DA增加15-35%解码步骤，模式分解不一定适合所有任务形状；需通过SFT/RLVR进一步优化模式选择和分解策略。
2. **人工分段局限**：静态基准需要将上下文切分为人工定义的magic chunks；实际agent系统中存在自然分段（工具调用、用户回合、检索段落），DA可更自然地应用。
3. **不支持thinking模式**：初步实验中模型无法在thinking标签内遵循DA协议；未来可将DA操作暴露为标准工具声明，使其融入thinking traces。
4. **全局模式仍有成本**：`<global>`步骤需全量注意力，占DA总注意力token的80%以上，且随上下文增长而上升；可通过在上下文中的短索引（如各片段摘要）替代全量扫描来缓解。
5. **小模型协议遵循率低**：Gemma-4-E4B的focus解析成功率仅58%，准确率大幅崩溃；证明DA需要一定的基础能力阈值。
6. **结构化数据/跨段聚合任务失败**：当答案依赖跨所有段落的聚合信息（如全局计数、完整表格）时，分段会破坏证据完整性，导致准确率暴跌。

## 研究启发与可借鉴点
1. **协议驱动的注意力控制范式**：将模型内部选择过程外显为可解析文本，为其他模型内部状态（如记忆访问、工具调用时机）的控制提供新思路——可从“预测/推断”转向“声明/执行”。
2. **块对齐掩码的工程实现**：通过重写KV cache块表实现细粒度注意力控制，同时保持与现有kernel（FlashAttention等）兼容，为未来研究提供了可直接复用的vLLM集成方案。
3. **零样本elicitation的可能性边界**：证明现代大模型无需训练即可在复杂协议下可靠运行，只要提示设计充分利用模型已有的边界感知能力（如段落、对话轮次结构）；这激励了对其他“模型本应具备但未充分激活”能力的探索。
4. **与投机解码、KV offloading的协同潜力**：DA降低每步代价，投机解码压缩序列步数，两者正交互补；DA的声明式访问模式也为KV cache分层调度（如按需预取）提供了天然的时间窗口。
5. **可扩展的评估框架**：结合roofline墙钟时间分析与绝对token计数，既能反映硬件层面的效率收益，又能隔离协议开销与掩码效果，为稀疏注意力研究提供更细粒度的评估维度。

## 关键术语表
**Declarative Attention (DA)**：一种让语言模型通过链式思考中的声明性标记（`<global>`、`<focus>`、`<local>`）显式指定其注意力范围的零样本协议。
**Magic Chunk**：将长上下文分割成的约2048-token语义单元，作为`<focus>`模式下的可寻址片段，通过模拟工具调用转录的形式呈现给模型。
**Attention Sink**：提示开头的固定token（通常为系统指令），吸引并保持注意力权重，防止上下文token被忽略。
**Block-Aligned Masking**：将注意力掩码的保留区间向上取整到KV cache块边界（通常16-32 token），确保内存读取可以跳过整个块，同时避免切割单个token。
**Roofline Wall-Time**：基于硬件峰值性能和利用率（MFU/MBU）估算的操作理论耗时，用于分离compute-bound和memory-bound组件的贡献。
**DA-nm (DA-no-mask)**：使用完整DA提示模板但关闭自定义掩码的消融变体，用于隔离提示格式本身与掩码机制的贡献。
**Focus Success Rate**：模型发出的`<focus>`声明成功解析为有效chunk引用的比例，衡量协议遵循能力。
**System-2 Sparse Attention**：将稀疏注意力决策从内部激活推断（system-1）转化为模型显式推理输出（system-2）的范式，具有可解释性和可干预性。

## 可复现要素
- **数据集**：来自15个公开长上下文基准（RULER、LongBench v1/v2、LooGLE、ZeroSCROLLS），部分使用Gemini-3-Flash生成的合成QA；原始基准可公开获取。
- **代码**：论文提供了vLLM集成细节（Appendix B），但未明确声明开源仓库；DA状态机和块表重写逻辑可基于所述设计复现。
- **模型权重**：Gemma-4-31B/12B/E4B、Qwen-3.6-27B/3.5-9B/4B均为公开可用模型。
- **关键超参**：片段大小~2048 token（硬上限2560）；最大生成长度8K token；采样参数按各模型官方推荐（Gemma: temperature=1.0, top-p=0.95, top-k=64；Qwen: temperature=0.7, top-p=0.80, top-k=20, presence penalty=1.5）；roofline分析假设MFU=40%、MBU=70%。
