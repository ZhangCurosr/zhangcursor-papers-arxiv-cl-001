---
title: "Quantifying-and-Mitigating-Korean-Jamo-Level-Typographical-V"
source: https://arxiv.org/pdf/2608.30229v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 16:27:52"
field: "多语言LLM鲁棒性与安全"
keywords: ["Korean typographical robustness", "jamo-level perturbation", "typo-aware chain-of-thought", "TACoT", "representational probing", "KMMLU", "language model vulnerability"]
innovations: ["提出5种韩语jamo层级扰动类型学并量化LLM脆弱性，揭示参数规模不带来鲁棒性增益", "发现typo-induced表征位移正交于答案错误方向，线性probe留一类型泛化AUROC达0.905–0.943", "提出TACoT探针路由框架，在节省约37%推理token的同时恢复CoT主要准确率增益"]
benchmarks: ["KMMLU", "HRM8K", "HAERAE-GK"]
---

# 论文速读：Quantifying-and-Mitigating-Korean-Jamo-Level-Typographical-V

## 一句话总结
本文系统量化了韩语特有**音节块内jamo层级排版错误**对大语言模型的脆弱性影响，发现参数规模与韩语专业化均无法抵御此类结构噪声；进一步揭示此类错误在模型内部表征中留下可检测的独立信号，并提出**TACoT（Typo-Aware Chain-of-Thought）**探针路由框架，以约37%的推理成本节省恢复CoT的主要准确率增益。

## 研究问题与动机
- 现有LLM鲁棒性基准以英语为中心，将typo建模为表面字符编辑，**无法捕捉韩语音节块内jamo（初声/中声/终声）层级的结构扰动**。
- 韩语键盘输入以jamo为单位逐个击键，自然typo发生在音节内部，可能产生两种失败模式：**有效形式歧义**（生成合法但语义变更的音节）或**暴露独立jamo**（破坏音节结构）；两者均超出子词tokenizer的容忍范围。
- 现有韩语GEC管线设计目标为恢复语法/拼写合法性，对有效形式ambiguity往往无动于衷，对重度扰动甚至会改写为语义偏移的流畅文本，**无法提供可靠的pre-correction prior**。
- 核心研究问题：① 现代LLM对真实jamo级扰动的脆弱程度如何？② 模型是否在内部分层表征中隐式编码此类结构噪声，能否借此实现低成本防御？

## 核心贡献（创新点）
1. **提出5种jamo层级扰动类型学**：基于韩语Dubeolsik键盘布局与音节块内部结构（初声替换、终声删除、jamo重复、空格删除、jamo换位），覆盖真实韩语输入错误的96.2%，形式化一类此前未被捕获的结构性排版脆弱性。
2. **建立首个受控的韩语jamo鲁棒性基准**：在KMMLU上对4款模型施加5×5强度梯度扰动，发现准确率单调下降且参数规模放大不带来鲁棒性增益，揭示该脆弱性具有结构性而非训练不足属性。
3. **发现并验证"typo方向"表征信号**：通过Fisher分离分数定位typo敏感层，并在表征空间中证明typo-induced shift正交于普通答案错误方向；训练线性logistic probe在留一typo类型设置下达到0.905–0.943 AUROC，泛化至未见扰动类型。
4. **提出TACoT探针引导的路由推理框架**：默认标准解码，仅在probe输出P(typo)超过阈值时触发CoT，在KMMLU与跨域HRM8K基准上恢复CoT主要增益的同时平均节省约37%输出token。

## 方法详解
### 4.2 Typo类型学
基于韩语音节块分解规则 `DECOMPOSE(s) = (c, v, f)`（初声c、中声v、终声f可选），定义5类扰动：
- **Jamo Substitution (T1)**：用键盘相邻键jamo替换c/v/f中第一个有邻接映射的分量，常产生合法但语义不同的音节。
- **Jongseong Deletion (T2)**：直接删除终声f，输入 `COMPOSE(c, v, ε)`，始终生成合法音节，难以被GEC捕获。
- **Jamo Repetition (T3)**：随机复制一个jamo分量，重组后在原音节后追加一个独立jamo，暴露原始jamo序列。
- **Space Deletion (T4)**：删除eojeol（韩语句间空格）使相邻词拼接，不影响音节内部结构。
- **Jamo Transposition (T5)**：交换c↔f（若有终声）或c↔v，破坏"初声-中声-终声"顺序，暴露raw jamo序列；对不变音节做2k过采样直至产生k次实际改变。

每个类型在5%–25%音节比例上独立施加，共生成 35,030 × 5 × 5 = 875,750 扰动样本。

### 6.1 Typo敏感层定位
对每层l计算Fisher分离分数：
$$J(l) = \frac{\|\mu_{\text{typo}}^{(l)} - \mu_{\text{clean}}^{(l)}\|_2^2}{\text{tr}(\Sigma_{\text{clean}}^{(l)}) + \text{tr}(\Sigma_{\text{typo}}^{(l)})}$$
选取各模型峰值层：EXAONE-2.4B→层10，EXAONE-7.8B→层9，A.X-Light→层9，Qwen3-4B→层18。

### 6.2 Typo方向与答案错误方向正交性检验
定义干净正确/错误样本均值隐状态 $\mu_{\text{correct}}, \mu_{\text{wrong}}$，构造：
- 答案错误方向：$d_{\text{wrong}} = \mu_{\text{wrong}} - \mu_{\text{correct}}$
- Typo方向：$d_{\text{typo}} = \mu_{\text{typo}} - \mu_{\text{correct}}$
将隐状态投影到二维基（第一轴$d_{\text{wrong}}$，第二轴$d_{\text{typo}}$正交于$d_{\text{wrong}}$的分量），可视化表明typo样本无论最终对错均沿typo方向偏移，证明信号不可还原为普通答错状态。

### 6.3 线性probe检测
在Fisher选定层最后token隐状态上训练L2正则logistic回归（C=1.0，特征标准化），训练集按clean/typo类别平衡采样（仅用非留一typo类型）。阈值选择采用Youden's J统计量最大化。留一typo类型AUROC达0.905–0.943。

### 7.1 TACoT路由框架
推断流程：
1. 将输入$x$送入模型，提取Fisher层最后token隐状态$h_l(x)$。
2. Probe估计$P(\text{typo} | h_l(x))$。
3. 路由决策：
$$\text{route}(x) = \begin{cases} \text{CoT}, & P(\text{typo}|h_l(x)) \geq \theta \\ \text{Standard}, & P(\text{typo}|h_l(x)) < \theta \end{cases}$$
CoT提示引入"题目可能含typo，请逐步分析"元认知引导；Standard仅输出单字母答案。

## 实验与结果
### 数据集与模型
- **KMMLU**（主基准）：35,030道韩语专家级四选一多选题，覆盖45个学科。
- **HRM8K**（泛化基准）：8,011道韩语自由形式数学推理题。
- **HAERAE-GK**（校准集）：176道通用知识题，用于Fisher层选择与probe训练，不与KMMLU/HRM8K重叠。
- 4款开源模型：EXAONE-3.5-2.4B/7.8B-Instruct（LG AI）、A.X-3.1-Light（SK Telecom）、Qwen3-4B-Instruct-2507（Alibaba）。

### 鲁棒性量化结果
- 所有模型在5种typo类型上准确率均随强度单调下降；jamo级扰动影响最大：EXAONE-7.8B在Jamo Transposition上较clean基线下降10.0%p，Qwen3-4B下降7.0%p。
- 规模放大不提升鲁棒性：EXAONE-7.8B在Transposition上的跌幅大于2.4B。
- Space Deletion是例外：所有模型在该类型下准确度接近clean基线，tokenizers能通过子词分割吸收空格缺失。

### 检测性能
- 留一typo类型probe AUROC（Table 2）：EXAONE-2.4B 0.924，EXAONE-7.8B 0.937，A.X-Light 0.943，Qwen3-4B 0.905；Jamo Transposition最易检测，Jongseong Deletion最难。
- 按强度分档（Table 9）：即使在最弱强度$l_1$（5%），AUROC仍达0.79–0.85，远高于随机；随强度单调上升。

### 缓解策略对比（KMMLU，Table 3）
- **CoT**：最稳定提升，A.X-Light clean基线49.5%→57.1%（+7.6p），所有typo类型提升>5.9p；但输出长度增至1024 tokens。
- **GEC/Meta-Cognition**：均未能稳定提升；GEC在Jamo Substitution上所有模型均下降，Meta-Cognition使Qwen3-4B全类型下降。
- **TACoT**：保留CoT主要增益，同时平均输出token减少约37%（EXAONE-2.4B -26%，A.X-Light -49%）；在Jamo Repetition/Transposition上效果最强。

### 泛化与路由有效性（HRM8K，Table 4 & Table 5）
- TACoT在HRM8K上同样优于Standard，CoT增益恢复幅度与KMMLU一致。
- 与rate-matched random router对比（仅看base-correct池）：TACoT在最高强度下领先random router达2.3p，证明增益来自probe路由而非单纯增多CoT调用。

## 相关工作脉络
1. **DeepWordBug / PromptRobust / PromptBench**（Gao et al., 2018; Zhu et al., 2024a,b）：研究英文prompt/文本层面的字符级对抗扰动；本文指出其范式无法捕捉韩语文本的内部音节结构扰动，需扩展至sub-character层级。
2. **多语言键盘typo鲁棒性评估**（Zhao et al., 2026）：评估多语种键盘typo但对韩语jamo内部结构缺乏细粒度建模；本文在其基础上引入5类jamo级扰动并提供结构化类型学。
3. **LLM对抗typo与CoT鲁棒性**（Gan et al., 2024）：证明 adversarial typo可破坏CoT推理轨迹；本文与其共同揭示typo脆弱性，但进一步提出基于内部表征信号的低成本路由防御，而非单纯暴露脆弱性。
4. **韩语GEC管线**（Yoon et al., 2023; Lee et al., 2021; Kim et al., 2024 KoGEC）：目标为语法/拼写纠正；本文系统验证KoGEC无法可靠处理valid-form ambiguity与exposed-jamo场景（可能无修改或语义偏移重写），确立pre-correction路线不可行。
5. **中文拼写纠错**（Cheng et al., 2020 SpellGCN; Sun et al., 2021 Chinese-BERT）：利用拼音/字形/部首相似性；本文将其与韩语jamo层级编辑做对照，突出不同书写系统的typo建模差异。
6. **表征探针对抗鲁棒性**：Fisher分离与线性probe方法源自传统表征分析；本文首次将其用于跨语言typo检测并用于inference-time路由决策，区别于仅做分析的探针工作。

## 局限性与未来方向
- 扰动为**自动生成**，虽覆盖96.2%真实键盘错误（基于AI-Hub 72.9万标注span验证），但仍可能与真实用户typo分布存在偏差；更广泛的领域、任务格式与自然typo分布待补充。
- **Probing与TACoT依赖内部隐藏状态访问**，仅适用于本地部署的open-weight模型，与API型LLM主流部署模式不兼容；论文仅对小-中规模模型（2.4B–7.8B）验证TACoT，更大模型（Gemini-3.1-Flash-Lite、Qwen3-235B）仅在Accuracy degradation层面评估。
- TACoT为**路由型防御**：检测typo并切换至CoT以提升准确性，但不直接修复或重对齐typo-induced表征偏移；未来需探索显式表征修复方法。
- 评测资源以韩语为主，跨语言可比性有限；其他黏着语/音节文字（如日语假名、泰语）的类似脆弱性有待验证。

## 研究启发与可借鉴点
1. **"书写系统感知"的扰动建模**：将typo类型学与底层键盘/书写结构（而非纯表面字符）挂钩，可在日语假名、泰语、阿拉伯语等书写系统中推广，构建多语种typo鲁棒性基准。
2. **Fisher分离+留一类型probe的检测范式**：不依赖额外标注，仅用clean/typo二元标签即可学习跨类型泛化的检测器；该方法可迁移至其他类型的input corruption（如OCR噪声、语音转写错误）检测。
3. **路由型推理调度（TACoT）**：以轻量probe作为gate在Standard/CoT间切换，兼顾准确率和推理成本；这种"选择性深度推理"范式可直接应用于其他需要防御特定输入污染的推理 pipeline。
4. **表征正交性分析**：通过$d_{\text{wrong}}$与$d_{\text{typo}}$的几何分解，证明typo信号独立于答案错误信号，为后续"解耦表征诊断"工作提供方法论参考。
5. **与GEC/Pre-correction的对比论证**：通过系统实验验证pre-correction在特定语言结构下失效，提示未来鲁棒性研究应优先分析correction prior的覆盖边界，而非直接套用GEC。

## 关键术语表
- **Jamo（자모）**：韩语音素书写单位，分为初声（onset）、中声（nucleus）、终声（coda），多个jamo组合构成一个音节块（서기）。
- **Valid-form ambiguity**：jamo扰动后生成另一个合法韩语音节，表面无误但语义改变，难以被GEC识别。
- **Exposed standalone jamo**：扰动破坏音节块结构，使独立jamo暴露在表面字符串中，tokenizer无法按合法音节切分。
- **KMMLU**：Korean Massive Multitask Language Understanding，35,030道韩语专家级四选一多选题，来自韩国真实考试。
- **Fisher separation score**：以类间均值距离除以类内散度追踪量，用于定位对typo最敏感的Transformer层。
- **TACoT（Typo-Aware Chain-of-Thought）**：基于内部表征probe的路由推理框架，仅在检测到typo时启用CoT。
- **HAERAE Bench**：韩语知识评测基准；其General Knowledge子集（HAERAE-GK，176题）用于本研究的probe校准。
- **Youden's J statistic**：灵敏度-特异度之和减1，用于优化二分类阈值，本文用于TACoT的routing threshold选择。

## 可复现要素
- **数据集**：KMMLU（公开）、HAERAE Bench（公开）、HRM8K（公开）；扰动生成代码已开源。
- **代码**：https://github.com/SJLee0311/korean-jamo-typo （论文声明开源）
- **权重**：EXAONE-3.5-2.4B/7.8B、A.X-3.1-Light、Qwen3-4B-Instruct-2507、KoGEC（Kim et al., 2024）均为公开模型；推理使用vLLM。
- **关键超参**：temperature=0，greedy decoding；non-CoT max_new_tokens=8，CoT max_new_tokens=1024；probe L2正则C=1.0，特征标准化；HAERAE-GK 80/20 train/val split；扰动强度5%/10%/15%/20%/25%。
- **Fisher选定层**：EXAONE-2.4B层10，EXAONE-7.8B层9，A.X-Light层9，Qwen3-4B层18。
- 更多实现细节见Appendix B（扰动生成）、Appendix D（prompt/inference）、Appendix G（probe训练）。
