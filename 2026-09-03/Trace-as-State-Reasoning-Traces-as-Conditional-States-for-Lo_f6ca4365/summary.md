---
title: "Trace-as-State-Reasoning-Traces-as-Conditional-States-for-Lo"
source: https://arxiv.org/pdf/2609.02702v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-07 05:29:01"
field: "长上下文语言模型推理优化"
keywords: ["长上下文推理", "因果Transformer", "推理轨迹", "输入顺序", " TRACE AS STATE", "状态更新", "推理扩展"]
innovations: ["形式化因果状态更新任务中的条件时序内存指数分离", "提出将reasoning traces作为任务状态文本代理并前置的TRACE AS STATE方法", "系统性验证跨模型跨任务的条件前置顺序增益"]
benchmarks: ["GraphWalks 256K", "MRCRv2 8-needle", "NUB-1M Season 2"]
---

# 论文速读：Trace-as-State-Reasoning-Traces-as-Conditional-States-for-Lo

## 一句话总结
本文提出TRACE AS STATE方法，将模型首轮推理生成的reasoning traces作为任务状态的文本代理，置于长上下文之前进行第二轮因果处理，从而弥补因果Transformer中"晚发现的条件状态无法影响早期表征"的结构缺陷；在三个前沿模型与三个长上下文基准上，该方法在27组实验中26组优于对照方法TRACE APPEND。

## 研究问题与动机
- **因果处理与任务状态发现的时序错配**：前沿大模型支持数十万至百万token输入，但自回归Transformer仅支持因果注意力，后出现的信息无法影响前序位置的表征；而许多长上下文推理任务所需的关键状态（如活跃目标、搜索边界、已排除假设）往往在读完部分上下文后才被发现。
- **条件状态更新任务的内存复杂度分离**：论文形式化证明，对于一次读取的确定性因果状态更新处理器，若任务条件先于信息序列出现，只需$\lceil b \rceil$位即可跟踪单一实现状态路径；若条件后出现，则最坏情况下需保留$|S|$个状态到自身的映射，至少需要$\lceil b 2^b \rceil$位，两者呈指数级差距。
- **推理轨迹可作为任务状态的文本代理**：现代推理模型在输出可见答案前会生成reasoning traces，其中包含中间计算信息（如活跃目标、解析后的引用关系），尽管可能不完整或有误，但仍可作为 imperfect textual proxy 承载任务状态。
- **现有方法未充分解决排序敏感性**：既往工作（如Re2、CoRe、Racing Thoughts等）已表明输入顺序、重复与上下文定位会影响模型行为，但缺乏将"晚发现状态前置"作为通用推理扩展策略的系统验证。

## 核心贡献（创新点）
1. **形式化因果状态更新任务中的条件时序内存分离**：首次以条件状态更新任务抽象刻画因果处理器在处理顺序上的指数级内存差距，为长上下文推理的顺序设计提供理论依据。与已有工作相比，本文不仅观察到排序效应，还给出了最坏情况下的信息论下界。
2. **提出TRACE AS STATE推理扩展方法**：将首轮reasoning traces序列化后置于长上下文之前进行第二轮因果处理，使晚发现的状态在重新处理上下文时即可影响表征形成。与TRACE APPEND（同等trace置于上下文之后）相比，该方法仅在排序上 differ，突出了"状态可用时机"的作用。
3. **系统性实证验证跨模型、跨任务的顺序增益**：在DeepSeek V4 Pro Preview、GLM-5.2、Qwen 3.7 Max三个来自不同厂商的模型上，于GraphWalks 256K、MRCRv2 8-needle、NUB-1M三个基准上测试，27组实验中有26组TRACE AS STATE优于TRACE APPEND，显示方法的普遍性。
4. **丰富的消融实验排除替代解释**：通过Question First、Re2、Answer Feedback、Random Trace、Trace Only等控制条件，证明增益并非来自通用trace脚手架效果、首轮答案直接利用或文本近期性，而是源于问题特定的任务相关推理文本的前置。

## 方法详解
- **因果状态更新处理器定义**：处理器按输入顺序逐个读取信息单元$c_i$，仅基于当前记忆$s_{i-1}$和新单元$c_i$更新状态：$s_i = U(s_{i-1}, c_i)$，其中$U$为固定更新规则，状态空间$S$有限。
- **条件状态更新任务**：初始状态由任务条件$z \in S$决定而非固定，处理器设$s_0 = z$后对序列$C = (c_1, \ldots, c_n)$依次应用$U$，返回$s_n$。论文证明$[z, C]$顺序仅需$\lceil b \rceil$位（$b = \log_2 |S|$），而$[C, z]$顺序最坏需$\lceil b 2^b \rceil$位。
- **Reasoning Traces作为文本状态代理**：对同一问题运行模型$n_{tr}$次，每次获得推理轨迹$r_j$和可见答案$a_j$；通过固定序列化器$\pi$将轨迹拼接为$T = \pi(r_1, \ldots, r_{n_{tr}})$，保留源顺序并添加固定标签与分隔符（如`<trace_start>`/`<trace_end>`）。
- **TRACE AS STATE流程**：第一轮$M([x, q])$生成traces；第二轮使用$M([T, x, q])$，将序列化trace块置于长上下文$x$和问题$q$之前，使trace在重新处理上下文时作为条件状态可用。
- **TRACE APPEND对照**：第二轮使用$M([x, T, q])$，保持$x$在前、$T$在后，trace仅能影响后续推理和答案生成，无法改变已形成的上下文表征。
- **序列化细节**：每个trace截断至前50,000字符以避免超限；数据集特定的$p_{\mathcal{D}}$前缀提示模型将traces视为scratchpad hints并验证；答案格式指令（如`Final Answer: [node1, node2]`）附加于末尾。
- **理论附录证明**：对任意有限非空状态集$S$，构造输入字母表$\mathcal{C} = \{c_f \mid f \in S^S\}$及更新规则$U(s, c_f) = f(s)$，使得单步序列即可实现$|S|^{|S|}$种不同残差映射，达到最坏情况下界。

## 实验与结果
- **模型**：DeepSeek V4 Pro Preview（CSA+HCA+mHC混合注意力）、GLM-5.2（1M-token MoE + DSA+IC）、Qwen 3.7 Max（GDN+GA），均使用最高推理强度。
- **数据集**：GraphWalks 256K（图遍历，含BFS和Parents子任务）、MRCRv2 8-needle（256K/512K长上下文多 needle）、NUB-1M Season 2（400–700K token长篇小说阅读理解，20题×5次重复）。
- **评估指标**：GraphWalks用EM和set F1；MRCRv2用EM和SequenceMatcher ratio；NUB-1M用DeepSeek V4 Pro judge准确率。
- **主要结果**（Table 2）：
  - **GraphWalks Parents**：DeepSeek V4 Pro从首轮的29.2% EM / 43.0% F1提升至TRACE AS STATE的81.8% EM / 91.3% F1；GLM-5.2达到100.0% EM和F1。
  - **GraphWalks BFS**：DeepSeek V4 Pro从31.6% EM提升至58.8%；Qwen 3.7 Max从60.0%提升至70.4%。
  - **MRCRv2 256K**：DeepSeek V4 Pro从53.8% EM提升至88.7%；GLM-5.2从88.4%提升至61.2%（注：此处数值需核对，原文GLM-5.2 MRCRv2 256K EM为88.4%首轮回61.2% TRACE AS STATE，但F1从91.3%升至71.9%）。
  - **NUB-1M**：DeepSeek V4 Pro从60.0%准确率提升至73.0%。
  - **总体**：27组模型-任务-指标组合中，TRACE AS STATE在26组优于TRACE APPEND。
- **消融结果**（Table 3）：
  - Question First仅提升Parents但未显著改善BFS。
  - Re2优于首轮但远低于TRACE AS STATE。
  - Answer Feedback（仅放首轮回答）效果有限。
  - Random Trace表现差于首轮，排除格式化效应。
  - Trace Only（无长上下文）效果接近TRACE APPEND。
  - TRACE AS STATE超越Oracle@5（从5个首轮答案中选最优），说明反馈轮次带来实质增益。
- **trace数量消融**（Figure 2）：$n_{tr}$从1增至5，性能单调上升，TRACE AS STATE始终高于TRACE APPEND。
- **置信区间**（Table 5）：24组配对比较中20组95% CI完全在零以上，仅4组跨越零（DeepSeek V4 Pro MRCRv2 512K EM、Qwen 3.7 Max GraphWalks BFS F1、GLM-5.2 GraphWalks BFS EM和F1）。

## 相关工作脉络
- **长上下文架构与有效利用**：稀疏/压缩/混合注意力扩展了名义上下文窗口，但模型仍对证据位置、干扰项和跨输入信息组合操作敏感（Liu et al., 2024; Kuratov et al., 2024; Hsieh et al., 2024）。本文不修改架构，仅在推理阶段调整输入顺序。
- **因果推理中的顺序效应**：Chen et al. (2024) 指出前提顺序影响推理；Ok & Lee (2026) 发现多选提示的顺序差距并验证重复选项可缩小差距；CoRe通过重复完整上下文降低支撑文档顺序敏感性；Racing Thoughts将上下文定位错误归因于层间竞争。本文理论化并验证了"状态前置"原则。
- **架构级递归与迭代**：Geiping et al. (2025) 训练循环深度的语言模型；Saunshi et al. (2025) 研究共享权重Transformer块；LoopFormer训练可变循环轨迹；LLaDA等掩码扩散模型多步重建。这些方法需修改训练/推理基础设施，而TRACE AS STATE保持因果处理不变。
- **重新阅读与文本反馈**：Re2在单提示内重复问题作为重新阅读基线；The Markovian Thinker跨reset chunk携带有界文本历史；ReContext递归构建并回放证据池。本文与之不同在于：使用模型自身生成的推理轨迹而非简单重复问题，且显式比较前置vs后置。
- **推理轨迹作为状态**：Hao et al. (2026) 发现合成提示可影响后续输出；State over Tokens将增长推理前缀视为外化计算状态；causal mediation证据表明模型并不可靠地使用其中间步骤。本文接受trace可能不完美，但仍将其作为功能性状态代理。

## 局限性与未来方向
- **接口依赖**：当前实现需要访问原始reasoning traces或类似暴露的状态接口；仅提供最终答案的模型/API需不同接口。
- **推理延迟与成本增加**：额外轮次增加token消耗和延迟；前置状态可能降低多轮设置中的KV cache复用率。
- **评估范围有限**：仅测试三个因果Transformer模型和三个长上下文任务族，未涉及多轮agent任务；需更多模型、领域、上下文长度和交互设置验证泛化性。
- **trace质量依赖**：方法复用模型生成trace，若首轮trace严重错误可能误导第二轮；未来可探索trace选择、压缩或学习更好的文本状态接口。
- **训练框架未联合优化**：当前为固定模型下的推理期干预，未来可设计联合学习模型与反馈接口的训练框架。

## 研究启发与可借鉴点
- **推理期顺序优化作为通用扩展策略**：无需修改模型架构或重新训练，仅通过调整输入顺序即可显著提升长上下文推理性能，适合快速验证和资源受限场景。
- **条件状态更新的理论分析框架**：可将该抽象应用于其他因果序列处理场景（如流式处理、在线学习），分析条件出现时机对内存/计算复杂度的影响。
- **trace作为imperfect状态代理的实验设计**：接受中间表示的不完整性，通过对照实验（Random Trace、Answer Feedback等）排除替代解释，为后续研究提供严谨评估范式。
- **trace数量作为推理缩放参数**：$n_{tr}$单调增益表明可通过增加首轮采样数进一步提升性能，为test-time compute scaling提供新维度。
- **与团队方向的结合机会**：若团队关注长上下文推理、agent系统或多轮对话，可将TRACE AS STATE思想迁移至工具调用轨迹、记忆检索结果或代码执行日志的状态化前置处理。

## 关键术语表
- **Causal State Update Processor**：按输入顺序逐个读取信息单元、仅基于当前记忆和新单元更新状态的处理器，无法回溯或访问未来信息。
- **Conditional State Update Task**：初始状态由任务条件$z$决定的状态更新问题，条件出现时机显著影响处理器所需工作内存。
- **TRACE AS STATE**：本文提出的推理扩展方法，将首轮reasoning traces序列化后置于长上下文之前进行第二轮因果处理。
- **TRACE APPEND**：对照方法，将相同trace置于长上下文之后，仅影响后续推理和答案生成。
- **Reasoning Trace**：模型在生成可见答案前输出的中间推理文本，可携带活跃目标、搜索边界等任务状态信息。
- **Textual State Proxy**：将reasoning trace视为任务状态的不完美文本代理，接受其可能不完整或有误，但仍具功能性价值。
- **Serializer $\pi$**：固定拼接函数，将多个首轮推理轨迹按源顺序组合为序列化文本块$T$，添加固定标签和分隔符。
- **Residual Map**：信息序列$C$诱导的从条件空间$S$到状态空间$S$的映射$\phi_C(z) = t(C, z)$，用于刻画条件后置时的内存需求下界。

## 可复现要素
- **数据集**：GraphWalks（HuggingFace开源）、MRCRv2（HuggingFace开源）、NUB-1M（GitHub开源），均为公开基准。
- **代码/权重**：论文未提供开源代码仓库；使用官方API（阿里云百炼、DeepSeek、Bigmodel）访问模型。
- **关键超参**：首轮运行次数$n_{tr} = 5$；每个trace截断至50,000字符；DeepSeek V4 Pro和GLM-5.2使用max推理强度；Qwen 3.7 Max使用xhigh system prompt。
- **提示模板**：附录C详细给出数据集特定的$p_{\mathcal{D}}$前缀和答案格式指令，可直接复现。
- **模型版本**：DeepSeek V4 Pro使用2026-4-24版本（非2026-8-13更新版本）。
