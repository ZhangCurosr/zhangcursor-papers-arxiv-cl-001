---
title: "Language-Models-Can-Control-Their-Own-Attention"
source: https://arxiv.org/pdf/2609.02737v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 21:02:14"
field: "高效大语言模型推理"
keywords: ["sparse attention", "long-context inference", "KV cache optimization", "chain-of-thought", "inference efficiency"]
innovations: ["提出声明式注意力协议让模型通过思维链显式声明关注范围", "零样本激活开箱即用模型的稀疏注意力能力无需微调", "基于block-aligned掩码重写实现与现有attention kernel兼容的高效集成"]
benchmarks: ["RULER", "LongBench v1", "LongBench v2", "LooGLE", "ZeroSCROLLS"]
---

# 论文速读：Language-Models-Can-Control-Their-Own-Attention

## 一句话总结
本文提出**声明式注意力（Declarative Attention, DA）**协议，让语言模型在推理过程中通过思维链显式声明其关注范围（全局/焦点/局部），推理引擎据此动态构建注意力掩码，从而在零样本条件下显著减少长上下文解码的KV缓存读取成本，同时保持接近原始精度的表现。

## 研究问题与动机
- **长上下文推理的KV缓存读取瓶颈**：Transformer在解码时每步需扫描整个KV缓存（如1M token上下文需加载约15GB），内存带宽成为主要开销。
- **稀疏注意力的选择代价高**：现有稀疏注意力方法虽能减少计算量，但每步需扫描整个上下文以近似注意力分数，仍为O(N)复杂度，无法消除选择开销。
- **人类阅读的局部性直觉**：人类阅读长文档时不会每步重读所有词，而是聚焦于相关片段；模型注意力天然稀疏，但如何高效利用这一特性仍是开放问题。
- **模型已隐含注意力规划信息**：研究表明模型隐状态中已编码未来token信息，Chain-of-Thought可将其外显，本文进一步延伸至"关注位置"的显式声明。

## 核心贡献（创新点）
1. **声明式注意力协议（DA）**：引入三种可解析的注意力模式标签（<global>/<focus>/<local>），让模型在思维链中声明其关注范围，推理引擎据此构建动态掩码，完全消除每步的显式选择开销。
2. **无需训练的零样本实现**：DA通过固定prompt在开箱即用的模型上零样本激活，无需微调或额外模块，适用于Gemma-4和Qwen-3系列等主流架构。
3. **vLLM高效集成方案**：基于block-aligned的KV缓存掩码重写，兼容FlashAttention等现有内核，无需修改底层attention kernel，仅需hook attention metadata builder即可实现。
4. **系统2稀疏注意力新范式**：将稀疏注意力从隐式（系统1）提升至显式可解释（系统2）层面，注意力计划以自然语言形式呈现，兼具可审计性与效率增益。

## 方法详解
**整体框架**：DA将生成过程划分为三种注意力模式，模型通过特定标签语法声明当前模式的切换：
- **<global>模式**：关注全部上下文分段，用于导航和定位相关信息（默认模式）
- **<focus magic_chunks="K">模式**：仅关注指定分段K，用于从特定区域提取精确值
- **<local>模式**：不关注任何上下文分段，仅基于已生成的回复进行推理

**上下文分段机制**：将长上下文按~2048 token切分为"magic chunks"，使用语义边界启发式分段（段落→句子→从句→词边界），并通过模拟工具调用格式呈现，使分段边界对齐模型训练见过的特殊token。

**解码时干预**：DA状态机监控模型输出流，在闭合标签处切换模式并更新注意力掩码；掩码按block粒度对齐（16-32 token），避免碎片化读取损失；全局注意力层应用掩码，滑动窗口注意力（SWA）和线性注意力层（GDN）保持不变。

**Prompt设计**：包含固定scaffold（system instruction、question、instruction）+ 分段上下文，instruction明确定义三种模式的使用规则和策略示例。

## 实验与结果
**评测设置**：
- 6个模型：Gemma-4-{31B, 12B, E4B}、Qwen-3.6-27B、Qwen-3.5-{9B, 4B}
- 15个长上下文数据集：RULER、LongBench v1/v2、LooGLE、ZeroSCROLLS
- 上下文长度覆盖9K至244K token
- 使用LLM judge（Qwen-3.5-4B，与Gemini-3.1-Pro一致性r=0.99）评估

**主要结果**：
| 指标 | Gemma-4-31B | Qwen-3.6-27B |
|------|-------------|--------------|
| 准确率下降 | 1.27pp (87.01%→85.74%) | 2.75pp (85.31%→82.56%) |
| 注意力tokens减少 | 52.0% (13.43M→6.45M) | 31.1% (22.54M→15.52M) |
| 最长上下文节省 | ~21M tokens/response | ~52M tokens/response |
| 理论解码时间缩减 | 0.71× vanilla | 0.77× vanilla |

**关键发现**：
- 模型规模越大，DA准确率越接近vanilla（Gemma-4-E4B仅保留29%相对准确率，Gemma-4-31B达99%）
- 上下文越长，绝对token节省越大（从1M到21M）
- 掩码本身是效率增益来源（DA vs DA-nm：Gemma减少71.1%，Qwen减少46.5%）
- <focus>和<local>模式共占生成token的73%，每token注意力节省76-99%

## 相关工作脉络
- **动态稀疏注意力**（QUEST、SparQ、H2O、SnapKV等）：每步仍需扫描上下文近似注意力分数（O(N)），DA通过模型声明完全消除此开销；DA可与轻量扫描方法结合用于global模式。
- **KV缓存淘汰**（Scissorhands、FastGen等）：永久丢弃KV条目以释放空间，存在不可逆信息丢失风险；DA采用可逆掩码，保持完整缓存可用性。
- **模型可控推理**（ReAct、Self-RAG、MemGPT、Quiet-STaR）：通过生成控制token影响执行流程，但多数编辑token序列或触发外部操作；DA仅改变KV缓存读取范围而不修改序列。
- **自选择注意力跨度**（SSAS, Jin et al. 2024）：同样基于模型声明选择关注区间，但需针对任务微调且仅支持2K上下文；DA零样本扩展到244K上下文。
- **注意力引导**（PASTA、Attention-Gate）：通过重加权或学习门控影响注意力分布；DA采用硬掩码直接减少KV读取量，实现真正的带宽节省。

## 局限性与未来方向
- **零样本策略次优**：DA比vanilla多15-35%解码步数，分段方式未必匹配任务结构，需通过SFT/RL优化模式选择策略。
- **人为分段限制**：静态基准强制切分上下文为人工chunk；Agent场景中工具调用、用户轮次等天然分段更适配DA。
- **仅支持非thinking模式**：初步实验中模型在thinking标签内无法遵循协议，需将DA操作封装为标准工具声明以扩展至thinking trace。
- **global模式仍有开销**：global步骤占DA注意力tokens的80%+且随上下文增长；可探索在-context索引（各分段摘要）替代全量扫描。
- **最小能力门槛**：小模型（如E4B）协议遵循成功率仅58%，需基础能力达标才能有效使用DA。

## 研究启发与可借鉴点
- **可解释稀疏注意力新范式**：将隐式注意力模式转化为显式可读文本，为可审计AI推理提供新思路，可延伸至安全监控场景。
- **零样本协议设计模式**：通过精心设计的prompt和分段格式，无需训练即可激发模型潜在能力，为其他推理优化提供方法论参考。
- **推理引擎集成技巧**：基于block-aligned的KV缓存掩码重写方案，兼容现有FlashAttention等内核，为高效实现稀疏注意力提供工程范例。
- **与现有技术的正交组合**：DA可与speculative decoding、轻量扫描索引、KV卸载等技术叠加，形成多层效率优化策略。
- **面向Agent系统的适配潜力**：天然匹配工具调用场景，每个tool response可作为独立segment，适合构建可追溯的agent推理链路。

## 关键术语表
**Declarative Attention (DA)**：声明式注意力，让模型通过思维链中的可解析标签声明其关注范围，推理引擎据此构建动态注意力掩码的方法。
**Focus mode**：焦点模式，模型仅关注指定上下文分段进行信息提取的推理模式。
**Global mode**：全局模式，模型扫描全部上下文分段的导航推理模式。
**Local mode**：本地模式，模型仅基于已生成回复进行自包含推理的模式。
**Magic chunk**：魔块，将上下文按~2048 token切分并通过模拟工具调用格式呈现的可寻址分段单元。
**Roofline wall-time**：屋檐图墙时，基于硬件峰值性能估算各操作理论耗时的分析框架。
**KV cache masking**：KV缓存掩码，通过重写block table限制attention kernel实际读取的缓存范围的技术。
**System-2 sparse attention**：系统2稀疏注意力，模型通过显式推理声明关注范围的稀疏注意力范式。

## 可复现要素
- **数据集**：RULER、LongBench v1/v2、LooGLE、ZeroSCROLLS（均为公开基准）
- **代码**：论文提供vLLM集成代码（Appendix B详述），需在vLLM基础上添加attention metadata builder hook
- **模型权重**：使用开源Gemma-4和Qwen-3系列模型
- **关键超参**：segment大小~2048 token，block大小16-32 token，max generation length 8K tokens
- **硬件**：NVIDIA B200 GPU，bf16精度
- **推理设置**：禁用thinking mode，temperature/top-p/top-k按各模型官方推荐值
