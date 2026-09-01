---
title: "Shared-Circuits-for-Shared-Grammar-Tracing-Subject-Verb-Agre"
source: https://arxiv.org/pdf/2608.18545v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 21:54:05"
field: "多语言机制可解释性"
keywords: ["multilingual mechanistic interpretability", "subject-verb agreement", "activation patching", "cross-lingual circuit sharing", "morphosyntax", "minimal pair recovery", "attention routing"]
innovations: ["提出 target-logit 与 minimal-pair 两种恢复指标区分共享粒度的分层分析框架", "发现跨语言屈折回路共享在严格屈折对比恢复层面最强且由语法计算需求驱动", "通过英语桥梁案例证明共享程度动态依赖语境中是否需显式屈折计算"]
benchmarks: ["UniMorph", "Chinese Verb Semantic Feature Dataset", "BLOOM-7b1", "Gemma-2-9b", "Llama-2/3.2", "Mistral-7b", "Qwen2-7b"]
---

# 论文速读：Shared-Circuits-for-Shared-Grammar-Tracing-Subject-Verb-Agreement

## 一句话总结
本文通过激活补丁（activation patching）和注意力分析，在 29 种语言、5 个开源模型族上系统考察多语言 LLM 中主语-动词一致（subject-verb agreement）的内部计算回路是否在跨语言间共享，发现具有显式屈折变化的语言之间存在显著的回路重叠，且这种重叠在"恢复屈折对比"这一严格指标下最强；英语作为部分屈折语言恰好充当桥梁案例，验证了跨语言共享由语法操作本身的实现需求驱动而非语言身份决定。

## 研究问题与动机
1. **核心问题**：多语言 LLM 在处理同一形态句法现象（present-tense subject-verb agreement）时，是复用共享的内部回路还是依赖各自独立的语言特异性机制？
2. **已有研究的空白**：先前工作仅证明跨语言回路重叠"可以"发生，但缺乏系统回答"在何种条件下共享会增强或减弱"，尤其缺乏基于形态类型差异（有/无显式屈折）的对比设计。
3. **方法学不足**：现有跨语言可解释性研究多依赖探针（probing）或表征相似度，缺乏在受控最小对（minimal pairs）上的因果干预测量。
4. **实际意义**：若共享回路存在，则可解释性发现、因果干预与对齐方法可跨语言迁移；若各自独立，则单语言分析结论难以推广。

## 核心贡献（创新点）
1. **系统量化跨语言一致回路的共享程度**：首次在 29 种语言、5 个模型族上建立基于 activation patching 的跨语言回路相似度度量框架，区别于以往仅报告"是否存在"的定性结论。
2. **提出两种互补恢复指标区分共享粒度**：目标 logit 恢复（target-logit recovery）衡量更宽泛的语境适配形式预测，最小对恢复（minimal-pair recovery）严格衡量屈折对比的恢复；二者比较揭示共享在严格屈折层面最强、在宽泛预测层面较弱，这是对已有文献的本质推进。
3. **揭示英语作为"桥梁案例"的动态位移**：证明英语并非静态地接近或不接近强屈折语言，而是在需要显式一致计算的 3sg 语境下主动向屈折语言回路靠拢，将共享机制的解释从"语言身份"转向"任务计算需求"。
4. **建立共享头部的功能性一致性证据**：发现跨语言重叠 top-20 头部的注意力角色向量在两个指标下均保持极高相似度（≈0.926–0.928），证明跨语言重叠不仅是位置层面的巧合，更是功能角色（信息路由模式）的复用。

## 方法详解
- **数据构造**：从 UniMorph（33 种语言）和 Chinese Verb Semantic Feature Dataset（中文）提取动词不定式及现在时屈折形式，生成 4 对双向人称/数对比（1sg↔2sg、1sg↔3sg、1pl↔3pl、3sg↔3pl）。
- **提示模板**：统一结构"Conjugation of the verb [infinitive] in present tense: [pronoun] [conjugation]"，跨语言适配，保证位置对齐以支持 patching 分析；经 tokenization 过滤（保留仅末 token 不同的最小对）和行为过滤（剔除模型未产生目标形式的样本）后，最终保留 29 种语言。
- **Attention-output Patching**：在 corrupted prompt 的前向传播中，将第 ℓ 层第 h 个 attention head 的输出替换为 clean prompt 中对应位置的缓存激活，测量因果贡献。
- **两种恢复指标公式**：
  - **Target-logit recovery**：$M_{\mathrm{target}}(\ell,h) = \frac{1}{n}\sum_{i=1}^{n}\left([\mathbf{L}_{\mathrm{patch}}^{(\ell,h,i)}]_{y_{\mathrm{clean}}^{(i)}} - [\mathbf{L}_{\mathrm{corr}}^{(i)}]_{y_{\mathrm{clean}}^{(i)}}\right)$，衡量单一 logit 的提升，适用于有无屈折的语言。
  - **Minimal-pair recovery**：$M_{\mathrm{pair}}(\ell,h) = \frac{\Delta(\mathbf{L}_{\mathrm{patch}}^{(\ell,h)}) - \Delta_{\mathrm{corr}}}{\Delta_{\mathrm{clean}} - \Delta_{\mathrm{corr}} + \epsilon}$，其中 $\Delta^{(i)}(\mathbf{L}) = L_{y_{\mathrm{clean}}}^{(i)} - L_{y_{\mathrm{corr}}}^{(i)}$，严格衡量对屈折对比的恢复比例。
- **跨语言相似度量**：将每语言的 layer-head heatmap 展平为向量，计算同一模型内语言对的 Pearson 相关系数（使用绝对值），作为回路重叠的量化指标；同时在重叠 top-20 头部上比较注意力角色向量（pronoun/infix/other 三分区的注意力质量分布）。

## 实验与结果
- **模型覆盖**：5 个开源模型族共 6 个 checkpoint（BLOOM-7b1、Gemma-2-9b、Llama-2-7b、Llama-3.2-3b、Mistral-7b、Qwen2-7b）。
- **语言覆盖**：初始 34 种语言经筛选后保留 29 种（24 种有显式屈折、4 种无显式屈折、英语为中间案例）；每种语言-模型组合平均保留 1,509.59 条 prompt，每对取 50 条用于 patching。
- **主要结果 1（§5.1）**：在 minimal-pair 指标下，屈折语言间的 pairwise 相似度 Across all 方向均值达 **0.5876（SD=0.3204，n=11,622）**，显著为正，且在所有 4 对屈折对比和双方向 patching 下均保持正向；minimal-pair 指标的跨语言方差低于 target-logit 指标，说明严格屈折恢复的共享更集中。
- **主要结果 2（§5.2）**：target-logit 指标下，屈折-屈折比较的相似度明显高于屈折-非屈折比较，形成清晰的类型学聚类；英语在 3sg 屈折语境下比非屈折语境的相似度显著上升，所有 6 个模型均观察到一致方向的位移（图 12）。
- **主要结果 3（§5.3）**：共享头部的注意力路由模式在跨语言间高度一致——min-pair 指标下平均相似度 ≈ **0.928**，target-logit 指标下 ≈ **0.926**；非屈折语言的 target-logit 头部对 pronoun 的注意力更低（0.177 vs. 屈折语言的 0.288–0.291），而对其他 token 的注意力更高（0.762 vs. 0.685–0.689）。
- **稳健性分析**：跨模型大小（0.5B–176B）、instruction tuning（base vs. IT）、预训练进度（BLOOM 1k→300k steps）均验证主结论稳定；BLOOM 训练轨迹显示跨语言相似度随训练逐步上升（图 16）。

## 相关工作脉络
1. **Wendler et al. (2024)**：证明多语言模型的语义/事实知识表征跨语言共享；本文将"共享"的讨论从语义扩展到形态句法层面，并引入因果干预验证。
2. **Mueller et al. (2022)**：发现多语言模型中存在重叠的语法子空间和一致相关神经元；本文的推进在于使用受控 minimal pairs 和两种恢复指标区分不同粒度的共享。
3. **Ferrando & Costa-jussà (2024)**：对 subject-verb agreement 进行跨语言电路相似性案例研究；本文扩展至 29 种语言、5 个模型族，并从位置相似性推进到功能路由相似性。
4. **Zhang et al. (2025)**：发现多语言模型中既存在结构相似也存在差异；本文提供了解释"何时相似、何时不相似"的条件性框架（由显式屈折需求驱动）。
5. **Tang et al. (2024)**：报告语言特异性神经元的存在；本文与之一致地表明共享与特异是梯度分布而非二元对立，共享程度随计算需求的严格性而变化。
6. **Stanczak et al. (2022)**：发现同一神经元在不同语言中编码不同特征；本文通过 attention patching 的因果度量超越了相关性层面的探针分析。

## 局限性与未来方向
1. **仅分析 attention 通路**：因果分析聚焦于 attention-head 层的信息路由，下游 position-wise MLP 是否参与屈折信号的处理与放大未被检测。
2. **结构化提示的生态效度**：使用高度受控的零样本提示而非自然语境句，识别出的头部签名在真实文本中的泛化性需进一步验证。
3. **单一语法现象**：仅考察现在时主语-动词一致，未覆盖其他形态句法现象（如格标记、性一致、时态等），结论的普遍性有待扩展。
4. **仅分析成功样本**：过滤了模型产生错误屈折形式的样本，因此结论刻画的是"成功实现一致计算时的回路"而非模型的整体语言能力。
5. **未来方向**：扩展到更多语法现象、更多模型组件（MLP）、不同训练设置（SFT/RLHF 后），以及将发现应用于跨语言可解释性工具和对齐方法的可迁移性评估。

## 研究启发与可借鉴点
1. **两种恢复指标的分离设计**值得直接借鉴：将"预测正确目标形式"和"恢复形态对比"拆分为两个因果度量，可帮助我们更精细地刻画多语言模型中不同抽象层次的知识共享程度，适用于后续对格、数、性等特征的类似分析。
2. **利用类型学中间案例（如英语）作为动态检验**：将部分实现某语法操作的语言作为"条件性迁移"的自然实验，验证共享是否由计算需求驱动而非语言家族标签，是一种优雅的因果推理设计。
3. **跨语言重叠头部的注意力角色向量比较**为功能层面的共享证据提供了量化方法，可将此技术推广至其他跨语言机制研究（如指代消解、问答路由等）。
4. **BLOOM 训练轨迹分析（1k→300k steps）**展示了共享回路如何在预训练中逐步涌现，这种训练动力学视角可为多语言表示的学习过程研究提供模板。
5. **与团队方向结合机会**：可将本文的"共享-特异梯度框架"迁移到多语言代码理解、低资源翻译中的形态处理、以及多语言对齐干预的跨语言迁移性评估等方向。

## 关键术语表
**Activation Patching**：在 corrupted prompt 的前向传播中将指定 attention head 的输出替换为 clean prompt 中的对应激活，以因果方式测量该 head 对目标输出的贡献。
**Minimal Pair Recovery**：衡量 patching 对 clean 与 corrupted 两个对比 token 之间 logit 差值（contrast）的恢复比例，严格反映屈折对比的因果依赖。
**Target-Logit Recovery**：衡量 patching 对 clean 目标 token 的 logit 绝对提升量，适用于有无显式屈折的语言进行统一评估。
**Conjugating Language**：在测试的人称/数对比中存在显式动词屈折变化的语言（如西班牙语、俄语）。
**Non-conjugating Language**：在测试对比中无显式屈折变化的语言（如中文、土耳其语、瑞典语）。
**Cross-lingual Circuit Sharing**：多语言模型在处理同类型语法操作时复用相似层-头位置和注意力路由模式的程度。
**Attention Role Vector**：将 attention mass 按语义词元类别（pronoun/infinitive/other）分区后得到的三维度分布向量，用于刻画 head 的信息路由功能。

## 可复现要素
- **数据集**：UniMorph（Batsuren et al., 2022）公开；Chinese Verb Semantic Feature Dataset（Deng et al., 2024）公开；实验提示模板和语言清单见附录 A。
- **模型**：5 个开源模型族全部 Hugging Face 公开可下载（BLOOM、Gemma、Llama、Mistral、Qwen2），附录 A.2 列出具体 repo 名。
- **代码/权重**：论文未明确声明代码开源链接；权重为标准开源模型。
- **关键超参**：每对取 50 prompts 用于 patching；top-k=20 头部用于注意力角色分析；tokenization 过滤保留仅末 token 不同的样本；behavioral filtering 阈值：每种语言-模型组合至少保留 50 prompts。
- **评估指标**：Pearson 相关系数（主指标）、Spearman、cosine、Jaccard overlap（稳健性）；bootstrap 置信区间用于统计推断。
