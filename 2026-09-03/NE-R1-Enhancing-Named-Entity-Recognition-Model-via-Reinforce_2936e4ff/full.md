# NE-R1: Enhancing Named Entity Recognition Model via Reinforcement Learning

Meixuan Chen<sup>1</sup> \* <sup>‡</sup>, Hehan Li<sup>2</sup>\*, Ruizhi Zhao<sup>3</sup>, Xin Lu<sup>2</sup>, Peizhi Xu<sup>2</sup>, Liwei Qian<sup>2</sup>, Meifang Li<sup>2</sup>, Shuanglong Li<sup>2</sup>, Hanmeng Liu<sup>2</sup>, Xin Pei<sup>2</sup>, Yanbiao Ma<sup>4,5,6†</sup>

<sup>1</sup>Shenzhen Graduate School, Peking University <sup>2</sup>Baidu, Inc., Beijing, China <sup>3</sup>Shenzhen University, Shenzhen, China

<sup>4</sup>Gaoling School of Artificial Intelligence, Renmin University of China, Beijing, China <sup>5</sup>Beijing Key Laboratory of Research on Large Models and Intelligent Governance <sup>6</sup>Engineering Research Center of Next-Generation Intelligent Search and Recommendation, MOE

chenmeixuan26@stu.pku.edu.cn 2410145008@mails.szu.edu.cn ybma1998@ruc.edu.cn {lihehan,luxin06,xupeizhi,qianliwei,limeifang,lishuanglong,liuhanmeng}@baidu.com

## Abstract

Named Entity Recognition (NER) has achieved substantial progress since the advent of large language models (LLMs). Nevertheless, the recognition of long-tail and domain-specific entities remains challenging due to the deficiency in parametric knowledge. Retrievalaugmented generation (RAG) offers a promising remedy by injecting external knowledge, but it also introduces noise and unnecessary cost when dealing with familiar cases. In this paper, we propose NE-R1, a novel framework for adaptive retrieval-augmented NER. We design a “retrieval-on-demand” mechanism for NER. Then we integrate it into models by a two-stage training method: (1) multi-task instruction tuning initialization; (2) end-to-end RL optimization with CoT. To achieve reasonable selection between parameterized and external knowledge, we design a multi-dimensional reward considering both accuracy and retrieval benefit. NE-R1 achieves state-of-the-art performance on various benchmarks, with average improvements of 2.52 F1 points in in-domain evaluation and 1.18 F1 points in zero-shot crossdomain evaluation.

## 1 Introduction

Named Entity Recognition (NER) is a fundamental task in Natural Language Processing (NLP), aiming to identify entity spans and their types (Lample, 2016; Ma, 2016). In recent years, large language models (LLMs) have catalyzed the rapid advancement of Generative NER (GenNER) (Seow, 2025; Wang, 2023; Zhou, 2024), which reformulates NER as a conditional generation problem, and achieved remarkable progress in this area.

![](images/cc85cc0d0e4010c0aa4ec7134aa89b5b619b9fddc71d44ed135e6014703c05a3.jpg)  
Figure 1: Motivation and overview of NE-R1’s adaptive retrieval mechanism. (a) Always-on retrieval indiscriminately introduces redundant noise and degrades performance for familiar entities. (b) NE-R1 adaptively triggers retrieval only when genuinely necessary.

However, GenNER still suffers from the parametric knowledge deficiency of LLMs (Lu, 2022; Guo, 2025). This leads to hallucinations or omissions when encountering long-tail, domainspecific unfamiliar entities or ambiguous situations (Kang, 2025). To address this limitation, Retrieval-Augmented Generation (RAG) (Gao, 2023; Fan, 2024) has been introduced to inject external knowledge into NER models (Tan, 2023; Zhenwei, 2024), and further augmentation has been explored through retrieving similar examples (Li, 2024).

Despite the benefits brought by retrieval augmentation, existing methods predominantly adopt a “full retrieval” strategy—triggering retrieval indiscriminately regardless of input complexity. This paradigm engenders two critical issues: (1) Noise Introduction, where mandatory retrieval for familiar cases may inject irrelevant information that interferes with model decisions, as illustrated in Figure 1(a). (2) Unnecessary Latency Overhead, where retrieval for high-frequency entities is often redundant. Our experiments on the MIT-Movie dataset reveal that full retrieval incurs a 4.85× increase in inference latency compared to no retrieval, yet yields negligible performance gains. Consequently, it becomes paramount to realize an adaptive “retrieve-on-demand” mechanism and seamlessly integrate it into NER models.

In this paper, we propose NE-R1, a novel framework for adaptive retrieval-augmented NER. As shown in Figure 1(b), NE-R1 intelligently determines when to trigger retrieval based on input complexity. Inspired by the recent breakthroughs of DeepSeek-R1 (Guo, 2025) in reasoning tasks, we design a two-stage training procedure. The first stage is capability initialization through multi-task instruction tuning, which endows the model with three foundational capabilities: parametric inference, retrieval triggering, and evidence fusion; the second stage is end-to-end RL training, incorporating an explicit Chain-of-Thought mechanism and a multi-dimensional reward that comprehensively estimates prediction correctness and retrieval benefit to jointly enhance overall performance.

Extensive experiments show that NE-R1 outperforms the evaluated baselines, improving the average in-domain F1 score by 2.52 points over the strongest overall baseline, including a 3.30-point gain on GENIA. In zero-shot cross-domain evaluation, NE-R1 improves performance in four of the five CrossNER domains and achieves the best fivedomain average, exceeding the strongest baseline by 1.18 F1 points. On MIT-Movie, NE-R1 also reduces inference latency by approximately 55% compared with the always-retrieve baseline while achieving a higher F1 score. Our contributions are summarized as follows:

• We propose NE-R1, a novel framework that brings a “retrieve-on-demand” mechanism to NER, allowing models to retrieve only when their own knowledge falls short.

• We design a two-stage training method that first injects foundational capabilities via multi-task instruction tuning, then performs end-to-end RL optimization with Chain-of-Thought reasoning and a multi-dimensional reward.

• NE-R1 surpasses existing methods on both indomain and zero-shot cross-domain evaluations, demonstrating an favorable balance of performance, efficiency, and generalization.

## 2 Related Work

LLM-based NER. Recent LLM-based NER work primarily follows two main paradigms. The first is In-Context Learning (Dong, 2024), which directly invokes LLMs through prompts. Works in this direction focus on instruction optimization (Liu, 2023; Wang, 2025; Gonen, 2023), providing domain knowledge (Tong, 2025; Cocchieri, 2025; Huang, 2025), or introducing reflection mechanisms (Ashok, 2023; Xie, 2023). ICL example selection (Xie, 2024; Chen, 2025) is also explored.

The second paradigm fine-tunes pretrained LLMs with supervised training. Representative works employ unified instruction templates (Lu, 2022), incorporate Chain-of-Thought (Huang, 2026), or decompose NER into subtasks (Guo, 2025; Lu, 2024). For domain adaptation, methods leverage data augmentation (Huang, 2025; Bogdanov, 2024), inject guiding knowledge (Yang, 2025; Sainz, 2024), or denoise pseudo-labels (Ding, 2024).

Retrieval-Augmented Generation. RAG integrates external information to supplement knowledge and reduce hallucination (Zhao, 2026; Peng, 2025; Ram, 2023). Recent advances optimize retrieval methods (Yan, 2024; Gao, 2023; Jeong, 2024), perform document refinement (Trivedi, 2023; Liu, 2026), or design multi-round mechanisms (Shao, 2023). However, most RAG systems rely on static retrieval strategies that may not adapt to diverse query types.

Reinforcement Learning for LLMs. Reinforcement learning has been applied to LLMs(Ouyang, 2022), though early methods face tremendous training challenges (Schulman, 2017). Alternative approaches maximize reward by solving a classification problem(Rafailov, 2023) or optimize through advantage normalization (Hu, 2025). RL has been extended to machine translation (He, 2025; Feng, 2025), text-to-SQL (Unknown, 2025), and question-answering (Song, 2025; Zheng, 2025).

## 3 Methodology

In this section, we present the NE-R1 framework. As shown in Figure 2, it comprises two key components: (1) Multi-Task Capability Initialization (MTCI), which equips the model with capabilities in parametric inference, retrieval triggering, and evidence-fusion inference via multi-task instruction tuning (Section 3.1); and (2) End-to-End RL Optimization, which designs a Chainof-Thought (CoT)-guided reasoning paradigm and a multi-dimensional reward, and employs Group Relative Policy Optimization (GRPO) for end-toend optimization (Section 3.2).

![](images/bfcbaaf30730b590f55fc4d49bc2d489a549948d313e018ef0aea9c107698e18.jpg)  
Figure 2: Overview of the NE-R1 framework. The pipeline consists of two main stages: (1) Multi-Task Capability Initialization, where we construct three instruction tuning tasks via pass-rate-based data selection to equip the model with foundational capabilities for parametric inference, retrieval triggering, and fusion inference; and (2) End-to-End RL Optimization, which employs Group Relative Policy Optimization (GRPO) to fine-tune the model, leveraging Chain-of-Thought (CoT) guided reasoning and a multi-dimensional reward mechanism to achieve adaptive retrieval and precise entity extraction.

## 3.1 Multi-Task Capability Initialization

We aim to enable adaptive retrieval decisions for NER. However, uninitialized models cannot generate effective queries or integrate retrieved results, making direct reinforcement learning inefficient and unstable. We therefore apply multi-task instruction tuning to inject three foundational capabilities into Qwen2.5-7B, using supervised data distilled from a stronger teacher model, Qwen2.5-32B.

Task Design and Data Construction. We design three complementary instruction tuning tasks: parametric inference $( T _ { \mathrm { p a r a m } } )$ , retrieval triggering $( T _ { \mathrm { t r i g g e r } } )$ , and fusion inference $( T _ { \mathrm { r a g } } )$

To construct high-quality training data that balances signal quality with sufficient exploration potential, we employ a pass-rate-based data selection mechanism. For each training sample, we use Qwen2.5-32B to generate K=10 candidate responses and apply task-specific selection criteria. • Parametric Inference $( T _ { \mathrm { p a r a m } } )$ trains the model to extract entities using only the input context and its internal parametric knowledge. The target sequence contains an <answer> block with the predicted entities. For each training sample x, we estimate the non-retrieval pass rate as

$$
c _ { \mathrm { p a r a m } } ( \boldsymbol { x } ) = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \mathbb { I } ( \boldsymbol { \hat { y } } _ { \mathrm { p a r a m } , k } = \boldsymbol { y } ^ { * } ) ,\tag{1}
$$

where $\hat { y } _ { \mathrm { p a r a m } , k }$ is the k-th sampled output, $y ^ { * }$ is the gold entity annotation, and $\mathbb { I } ( \cdot )$ is the indicator function. Samples with reliable parametric predictions are retained:

$$
\mathcal { D } _ { \mathrm { p a r a m } } = \{ x \mid c _ { \mathrm { p a r a m } } ( x ) \geq \tau _ { \mathrm { p a r a m } } \} .\tag{2}
$$

• Retrieval Triggering $( T _ { \mathrm { t r i g g e r } } )$ trains the model to decide whether external knowledge is needed and, when necessary, to generate a rewritten search query. The supervised target contains a <think> fragment followed by a <search> block. The <think> fragment describes the source of uncertainty (e.g., entity ambiguity, rare mentions, or insufficient contextual evidence), while the <search> block contains an entity-centric rewritten query rather than the original sentence. Since no gold query exists for this task, we evaluate trigger quality indirectly via the downstream fusion pass rate (defined below) and retain trajectories whose retrieved evidence enables reliable fusion.

![](images/7fee880aff06a2d71101a8f03317daaffb357a7a921f6dd899deb427b106444f.jpg)  
Figure 3: Illustration of the three supervised tasks in Multi-Task Capability Initialization (MTCI). (a) Parametric Inference performs NER directly from the input context and the model’s parametric knowledge. (b) Retrieval Triggering determines whether external retrieval is necessary and, if needed, generates a query in <search> tags. (c) Fusion Inference integrates retrieved evidence to produce the final NER output while preserving spans from the original input.

• Fusion Inference $( T _ { \mathrm { r a g } } )$ trains the model to produce the final NER prediction after retrieved evidence is inserted via <information> tags. Given a query generated by $T _ { \mathrm { t r i g g e r } } .$ the retriever returns evidence, and the teacher samples K fusion outputs per training sample. We compute the retrievalaugmented pass rate as

$$
c _ { \mathrm { r a g } } ( \boldsymbol { x } ) = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \mathbb { I } ( \hat { y } _ { \mathrm { r a g } , k } = y ^ { * } ) ,\tag{3}
$$

where $\hat { y } _ { \mathrm { r a g } , k }$ is the k-th sampled fusion output. The fusion dataset is selected by

$$
\mathcal { D } _ { \mathrm { r a g } } = \{ x \mid c _ { \mathrm { r a g } } ( x ) \geq \tau _ { \mathrm { r a g } } \} .\tag{4}
$$

Here, τ<sub>param</sub>, $\tau _ { \mathrm { r a g } } ,$ and $\tau _ { \mathrm { t r i g g e r } }$ are task-specific selection thresholds. Beyond data filtering, the pass rates $c _ { \mathrm { p a r a m } }$ and $c _ { \mathrm { r a g } }$ also partition training samples into easy and hard groups, which are used to define the retrieval-benefit reward in Section 3.2.

Multi-Task Joint Fine-Tuning. As illustrated in Figure 3, we perform multi-task joint finetuning on the filtered samples to acquire the three capabilities. We consolidate the tasks into $\mathcal { T } ~ = ~ \{ T _ { \mathrm { p a r a m } } , T _ { \mathrm { t r i g g e r } } , T _ { \mathrm { r a g } } \}$ with distinct supervision: $T _ { \mathrm { p a r a m } }$ generates answers using internal knowledge; $T _ { \mathrm { t r i g g e r } }$ generates retrieval queries; and $T _ { \mathrm { r a g } }$ performs inference conditioned on retrieved information. The task-specific prompts are provided in Appendix A.4.

We jointly optimize over this mixed distribution by minimizing the weighted negative loglikelihood loss:

$$
\mathcal { L } = - \sum _ { ( x , y ) \in \mathcal { T } } \lambda _ { \mathrm { t y p e } } \sum _ { t = 1 } ^ { | y | } \log P _ { \theta } ( y _ { t } \mid y _ { < t } , x ) ,\tag{5}
$$

where $\lambda _ { \mathrm { t y p e } }$ denotes the task-specific balancing weight, x is the input context, y is the corresponding target output sequence, and |y| is its length.

## 3.2 End-to-end RL Optimization

MTCI equips the model with the abilities to reason, search, and fuse evidence, but it does not directly optimize the global trade-off between NER accuracy and retrieval cost. We therefore further optimize the initialized policy with rule-based rewards under a unified trajectory-level objective.

CoT-Guided Adaptive Retrieval. To enable dynamic selection between parametric and retrievalaugmented reasoning, we design a Chain-of-Thought (CoT) guided paradigm that formalizes NER as a "reasoning-then-action" branching process. Given input x, the generation process is:

$$
y = \left\{ { \begin{array} { l l } { \pi _ { \theta } ( y \mid x , c ) , } & { a ( c ) = \operatorname { d i r e c t } } \\ { \pi _ { \theta } ( y \mid x , c , { \mathcal { R } } ( q ) ) , } & { a ( c ) = \operatorname { s e a r c h } } \end{array} } \right.\tag{6}
$$

where c denotes the CoT reasoning fragment, $a ( c )$ is the decision action, q is the retrieval query, and $\mathcal { R } ( q )$ denotes retrieved knowledge. This comprises two steps:

• Introspective Planning. The model generates a <think> fragment c to analyze entity ambiguity, domain knowledge needs, and context sufficiency. With insufficient internal knowledge, it generates a <search> tag with a rewrite search query q; otherwise, it proceeds directly to answer generation.

• Evidence Fusion. When <search> is triggered, search engine $\mathcal { R }$ provides top-k documents $\mathcal { R } ( q )$ The model fuses input x with $\mathcal { R } ( q )$ then generates entity predictions y in the <answer> block.

Multi-Dimensional Reward. We define a reward function that jointly encourages valid trajectories, accurate entity extraction, and selective retrieval.

• Format Reward $r _ { \mathrm { f m t } }$ ensures that generated trajectories are parsable and verifiable:

$$
r _ { \mathrm { f m t } } ( y ) = ( \omega _ { c } + \omega _ { e } ) \mathbb { I } [ \mathcal { F } ( y ) ] - \omega _ { e } ,\tag{7}
$$

where $\mathcal { F } ( y )$ checks whether tags such as <search> and <answer> are complete and properly paired, and $\omega _ { c } , \omega _ { e } > 0$ are the reward and penalty weights, respectively.

• Accuracy Reward $r _ { \mathrm { a c c } }$ uses the entity-level Micro-F1 between the prediction $y$ and the gold annotation $y ^ { * }$ :

$$
r _ { \mathrm { a c c } } ( y , y ^ { * } ) = \mathrm { M i c r o - F 1 } ( y , y ^ { * } ) .\tag{8}
$$

• Retrieval Benefit Reward $r _ { \mathrm { b n f } }$ encourages retrieval on hard instances while discouraging unnecessary or harmful retrieval on easy instances:

$$
\begin{array} { r } { r _ { \mathrm { b n f } } ( x , y ) = \left\{ \begin{array} { l l } { + w _ { \mathrm { h a r d } } ^ { \mathrm { c o r r } } , } & { r _ { \mathrm { a c c } } > \tau \wedge x \in \mathcal { X } _ { \mathrm { h a r d } } , } \\ { + w _ { \mathrm { e a s y } } ^ { \mathrm { c o r r } } , } & { r _ { \mathrm { a c c } } > \tau \wedge x \in \mathcal { X } _ { \mathrm { e a s y } } , } \\ { - w _ { \mathrm { h a r d } } ^ { \mathrm { e r r } } , } & { r _ { \mathrm { a c c } } \leq \tau \wedge x \in \mathcal { X } _ { \mathrm { h a r d } } , } \\ { - w _ { \mathrm { e a s y } } ^ { \mathrm { e r r } } , } & { r _ { \mathrm { a c c } } \leq \tau \wedge x \in \mathcal { X } _ { \mathrm { e a s y } } . } \end{array} \right. } \end{array}\tag{9}
$$

We set $w _ { \mathrm { h a r d } } ^ { \mathrm { c o r r } } > w _ { \mathrm { e a s y } } ^ { \mathrm { c o r r } }$ and $w _ { \mathrm { h a r d } } ^ { \mathrm { e r r } } < w _ { \mathrm { e a s y } } ^ { \mathrm { e r r } }$ , so that successful retrieval is rewarded more on hard instances while erroneous retrieval is penalized more on easy instances. The subsets $\mathcal { X } _ { \mathrm { h a r d } }$ and $\mathcal { X } _ { \mathrm { e a s y } }$ are determined from the MTCI pass rates. The final reward is

$$
R = \alpha r _ { \mathrm { a c c } } + \gamma r _ { \mathrm { b n f } } + \lambda r _ { \mathrm { f m t } } ,\tag{10}
$$

where $\alpha , \gamma ,$ and λ control the weights of accuracy, retrieval benefit, and format validity, respectively.

RL Algorithm. We optimize the policy using Group Relative Policy Optimization (GRPO). Unlike actor–critic methods, GRPO estimates advantages from a group of sampled trajectories and requires no separate critic network. Since retrieval evidence is externally injected rather than generated by the policy, we follow R1-Searcher++ (Song, 2025) and mask environmental observation tokens when computing policy gradients:

$$
\mathbb { M } _ { t } = \left\{ { \begin{array} { l l } { 1 , } & { y _ { t } \in { \mathcal { A } } _ { \mathrm { g e n } } , } \\ { 0 , } & { y _ { t } \in { \mathcal { O } } _ { \mathrm { e n v } } , } \end{array} } \right.\tag{11}
$$

where $\boldsymbol { \mathcal { A } } _ { \mathrm { g e n } }$ denotes model-generated tokens and $\mathcal { O } _ { \mathrm { e n v } }$ denotes externally retrieved evidence. The masked GRPO objective is

$$
J ( \theta ) = \mathbb { E } \bigg [ \frac { 1 } { G } \sum _ { i = 1 } ^ { G } \mathcal { L } _ { i } ( \theta ) - \beta D _ { \mathrm { K L } } \bigg ] ,\tag{12}
$$

$$
\mathcal { L } _ { i } ( \theta ) = \frac { 1 } { \vert y _ { i } \vert } \sum _ { t = 1 } ^ { \vert y _ { i } \vert } \mathrm { M } _ { t } \mathrm { m i n } ( \rho _ { i , t } \hat { A } _ { i } , \bar { \rho } _ { i , t } \hat { A } _ { i } ) ,\tag{13}
$$

where $\bar { \rho } _ { i , t } = \mathrm { c l i p } ( \rho _ { i , t } , 1 - \epsilon , 1 + \epsilon )$ , G is the number of sampled trajectories, $\rho _ { i , t }$ is the importance sampling ratio, ${ \hat { A } } _ { i }$ is the group-relative advantage, and $\beta$ is the KL penalty coefficient.

## 4 Experiments

In this section, we systematically evaluate the proposed NE-R1 framework to address the following research questions:

• RQ1: Does NE-R1 outperform existing models on NER benchmarks?

• RQ2: Does NE-R1 generalize cross-domain in zero-shot?

• RQ3: What is the contribution of each NE-R1 component?

• RQ4: Does NE-R1 learn effective adaptive retrieval behavior?

## 4.1 Setup

Datasets and Metrics. We evaluate our model on standard benchmarks for both in-domain and cross-domain scenarios. For in-domain evaluation, we use OntoNotes 5.0 (Weischedel, 2013), MIT-Movie (Liu, 2013), MIT-Restaurant (Liu, 2013), and GENIA (Kim, 2003), covering general, shorttext, and specialized domains. For cross-domain evaluation (RQ2), following the standard protocol (Nandi, 2024), models are trained on CoNLL-03 and evaluated zero-shot on the five CrossNER domains. All results are evaluated using strict entitylevel Micro-F1. Detailed dataset statistics are provided in Appendix A.1.

<table><tr><td>Category Model</td><td></td><td>MIT-Movie MIT-Rest. OntoNotes</td><td></td><td>GENIA</td><td>Avg.</td></tr><tr><td rowspan="5">Few-shot LLMs</td><td>GPT-40</td><td>61.49 59.90</td><td>64.29</td><td>44.63</td><td>57.58</td></tr><tr><td>Llama-3.3-70B</td><td>60.76 53.62</td><td>59.98</td><td>41.91</td><td>54.07</td></tr><tr><td>Qwen3-32B</td><td>52.14 56.47</td><td>46.71</td><td>39.99</td><td>48.83</td></tr><tr><td>DeepSeek-R1-MoE</td><td>63.93 64.87</td><td>67.81</td><td>53.81</td><td>62.61</td></tr><tr><td>GPT-5</td><td>61.23 66.53</td><td>72.28</td><td>51.22</td><td>62.81</td></tr><tr><td rowspan="5">NER baselines</td><td>InstructUIE (Wang, 2023)</td><td>89.58</td><td>82.59 88.64</td><td>75.71</td><td>84.13</td></tr><tr><td>GLiNER-L (Zaratiana, 2024)</td><td>87.90 83.60</td><td>89.80</td><td>78.40</td><td>84.93</td></tr><tr><td>B2NER (Yang, 2025)</td><td>90.78 83.71</td><td>84.31</td><td>76.43</td><td>83.81</td></tr><tr><td>ReasoningNER (Huang, 2026)</td><td>89.60 81.40</td><td>89.10</td><td>78.30</td><td>84.60</td></tr><tr><td>UniNER-7B (Zhou, 2024)</td><td>90.17 82.35</td><td>89.91</td><td>77.54</td><td>85.00</td></tr><tr><td rowspan="2">Retrieval</td><td>Standard SFT (No Search)</td><td>85.24</td><td>80.68 83.47</td><td>75.92</td><td>80.83</td></tr><tr><td>RAG (Always Search)</td><td>85.56 81.72</td><td>84.38</td><td>77.23</td><td>82.22</td></tr><tr><td>NE-R1</td><td>NE-R1 (MTCI only)</td><td>87.94</td><td>83.46 85.12</td><td>77.77</td><td>83.57</td></tr><tr><td></td><td>NE-R1 (full)</td><td>91.05+0.27</td><td>85.44+1.73 91.89+1.98</td><td>81.70+3.30</td><td>87.52+2.52</td></tr></table>

Table 1: In-domain NER results. We report Micro-F1. Bold marks the best result in each column, underlining marks the strongest prior baseline, and the highlighted NE-R1 row additionally reports gains over the strongest prior result.

Baselines. We benchmark NE-R1 against general-purpose LLMs (GPT-4o, GPT-5, Llama-3-70B, Qwen3-32B, DeepSeek-R1) and domainspecific NER models (InstructUIE, UniNER, GLiNER, B<sup>2</sup>NER, ReasoningNER) for in-domain evaluation. For cross-domain evaluation, we follow (Nandi, 2024) and adopt their baselines including transfer-based, prompting-based, and retrieval-augmented approaches.

Implementation Details. We use Qwen-2.5-7B as the backbone pre-trained language model, Qwen-2.5-32B generates training contents for MTCI. All experiments are conducted on 8×NVIDIA A800 GPUs. In MTCI phase, we set learning rate to 1e-5 and batch size to 16. In RL phase, we set the rollout size to 5 and the learning rate to 1e-6. We employ a low-variance KL penalty (coefficient 1e-3) and a clip coefficient of 0.2. We use multi-source web search engines as external knowledge acquisition channels. For more detailed implementation details, refer to the appendix A.2.

## 4.2 Main Results (In-Domain Evaluation)

We first evaluate NE-R1 under the standard supervised setting. We compare the model against comparable (7B) and larger (70B+) baselines across four datasets characterized by diverse domain distributions and knowledge dependencies. As shown in Table 1, NE-R1 achieves superior performance on all benchmarks, consistently outperforming both equal-sized and larger models. Specifically, On MIT-Movie and MIT-Restaurant, it leads among 7B models with 91.05 and 85.44 F1 score, on the general-domain OntoNotes 5.0, it attains 91.89 F1 score, surpassing InstructUIE and UniNER-7B by over 2.0 points. On the terminology-dense GE-NIA, it achieves 81.70 F1 score, a significant 3 point improvement over GLiNER. These results demonstrate the effectiveness of adaptive retrieval across the four evaluated in-domain NER benchmarks.

## 4.3 Cross-Domain Evaluation

To address RQ2, we train NE-R1 on CoNLL-03 and evaluate it zero-shot on the five CrossNER domains (Table 2). Compared with IF-WRANER, NE-R1 improves Politics, Natural Science, Literature, and AI by 1.55, 0.52, 0.78, and 4.97 F1 points, respectively, but performs 1.92 points lower on Music. It nevertheless achieves the best five-domain average of 78.15, outperforming IF-WRANER by

<table><tr><td>Model</td><td>Politics</td><td>Natural Science</td><td>Music</td><td>Literature</td><td>AI</td><td>Avg.</td></tr><tr><td>BiLSTM-CRF (Lample, 2016)</td><td>56.60</td><td>49.97</td><td>44.79</td><td>43.03</td><td>43.56</td><td>47.59</td></tr><tr><td>Coach (Liu, 2020)</td><td>61.50</td><td>52.09</td><td>51.66</td><td>48.35</td><td>45.15</td><td>51.75</td></tr><tr><td>CROSS-DOMAIN LM (Jia, 2019)</td><td>68.44</td><td>64.31</td><td>63.56</td><td>59.59</td><td>53.70</td><td>61.92</td></tr><tr><td>FLAIR (Akbik, 2019)</td><td>69.54</td><td>64.71</td><td>65.60</td><td>61.35</td><td>52.48</td><td>62.73</td></tr><tr><td>BARTNER (Liu, 2021)</td><td>69.90</td><td>65.14</td><td>65.35</td><td>58.93</td><td>53.00</td><td>62.46</td></tr><tr><td>LIGHTNER (Chen, 2022)</td><td>72.78</td><td>66.74</td><td>72.28</td><td>65.17</td><td>35.82</td><td>62.56</td></tr><tr><td>LST-NER (Zheng, 2022)</td><td>73.25</td><td>70.07</td><td>76.83</td><td>70.76</td><td>63.28</td><td>70.84</td></tr><tr><td>LANER (Hu, 2022)</td><td>74.06</td><td>71.83</td><td>78.78</td><td>71.11</td><td>65.79</td><td>72.31</td></tr><tr><td>CP-NER (Chen, 2023)</td><td>74.25</td><td>75.82</td><td>79.10</td><td>72.17</td><td>67.95</td><td>73.86</td></tr><tr><td>GPT-NER (Wang, 2025)</td><td>74.71</td><td>70.77</td><td>78.30</td><td>62.18</td><td>66.07</td><td>70.41</td></tr><tr><td>PromptNER (GPT-3.5) (Ashok, 2023)</td><td>71.74</td><td>64.83</td><td>77.78</td><td>64.15</td><td>59.35</td><td>67.57</td></tr><tr><td>PromptNER (GPT-4) (Ashok, 2023)</td><td>78.61</td><td>72.59</td><td>84.26</td><td>74.44</td><td>64.83</td><td>74.95</td></tr><tr><td>RAG + GPT-4 (sentence embeddings)</td><td>78.20</td><td>73.52</td><td>83.61</td><td>71.32</td><td>66.91</td><td>74.71</td></tr><tr><td>RAG + GPT-4 (word embeddings)</td><td>78.63</td><td>73.95</td><td>84.25</td><td>74.68</td><td>68.19</td><td>75.94</td></tr><tr><td>IF-WRANER (Nandi, 2024)</td><td>79.80</td><td>75.31</td><td>85.43</td><td>75.52</td><td>68.81</td><td>76.97</td></tr><tr><td colspan="5">NE-R1 (ours) 81.35+1.55 75.83+0.52 83.51-1.92</td><td>76.30+0.78 73.78+4.97</td><td>78.15+1.18</td></tr></table>

Table 2: Zero-shot cross-domain NER results on CrossNER. We report Micro-F1. Bold marks the best result in each column, underlining marks the strongest prior baseline, and the highlighted NE-R1 row reports gains over IF-WRANER, the strongest overall baseline in average F1.

<table><tr><td>Variant</td><td>Movie Res. Onto.</td><td>Gen.</td><td>Avg.</td></tr><tr><td>Base model</td><td>51.50 56.53 44.24</td><td>39.96</td><td>48.06</td></tr><tr><td>w/o RL</td><td>87.94 83.46</td><td>85.12 77.77</td><td>83.57</td></tr><tr><td>w/o MTCI</td><td>71.92 67.84</td><td>74.63 67.58</td><td>70.49</td></tr><tr><td>w/o  $T _ { \mathrm { t r i g g e r } } \& T _ { \mathrm { r a g } }$ </td><td>84.77 80.06</td><td>82.76 75.34</td><td>80.73</td></tr><tr><td>w/o CoT</td><td>89.84 84.73</td><td>89.37 80.71</td><td>86.16</td></tr><tr><td>w/o  $r _ { \mathrm { b n f } }$  NE-R1 (full)</td><td>88.71 83.68 91.05 85.44</td><td>86.64 79.39 91.89 81.70</td><td>84.61 87.52</td></tr></table>

Table 3: Ablation study on different datasets. All results are micro-F1 score. Movie, Res., Onto., and Gen. denote MIT Movie, MIT Restaurant, OntoNotes, and GENIA datasets, respectively. “w/o” denotes removing a specific component from our full NE-R1 model.

## 1.18 F1 points.

The largest gain on AI suggests that retrieval is particularly useful for specialized and evolving technical entities. In contrast, the smaller gains on Natural Science and Literature and the decrease on Music indicate that retrieval is less effective for errors involving local context, entity boundaries, aliases, or dataset-specific label distinctions, and may introduce noise for ambiguous entity names. Overall, the benefit of adaptive retrieval varies across domains.

## 4.4 Ablation Studies

To dissect the efficacy of NE-R1 (RQ3), we performed ablations on training stages, initialization tasks, and core components (Table 3).

Training Stage Ablation. w/o MTCI removes the Multi-Task Capability Initialization stage; w/o end-to-end RL removes the RL Optimization stage. Removing MTCI prevents effective query generation and knowledge integration, while removing end-to-end RL causes notable performance drops, confirming both stages are essential.

Initialization Task Ablation. w/o $T _ { t r i g g e r }$ and $T _ { r a g }$ excludes Retrieval Triggering and Fusion Inference from initialization tasks. This causes a 6.8 point drop in F1 score (87.52→80.73), demonstrating that joint training on query generation and information fusion is crucial.

Reasoning and Reward Ablation. w/o CoT removes Chain-of-Thought reasoning; w/o $r _ { b n f }$ removes Retrieval Benefit Reward. Ablating r<sub>bnf</sub> proves more detrimental than removing CoT, indicating the benefit-aware reward is central to balancing retrieval costs.

## 4.5 Analysis of Adaptive Search Behavior

To address RQ4, we analyze the retrieval behavior of NE-R1 from three perspectives: domain-level retrieval frequency, the efficiency–performance tradeoff, and training dynamics. Together, these analyses examine whether retrieval decisions vary with knowledge demand rather than following a fixed always-retrieve or never-retrieve policy.

![](images/c502a7ba43fa5fbf83f9194d080cd41cd09970efde060ab08fab45e4727166a1.jpg)  
(a) Domain-wise performance and search behavior of NE-R1.

![](images/f314ceb2d894093a7df5d5852347c51b1cff1b900a9554f0b4bfa9395c8344ed.jpg)  
(b) Efficiency comparison on MIT-Movie.

Figure 4: Analysis of NE-R1’s adaptive retrieval behavior. (a) NE-R1 calibrates retrieval frequency based on domain knowledge density. (b) NE-R1 achieves the best trade-off between accuracy and inference cost on MIT-Movie.  
![](images/83b917a02ad7929f5bccc05b2361e6fe02829acbf89869bc7ced2efa3d466e5b.jpg)  
(a) Reward vs. Search Rate (Easy Samples)

![](images/cf20b3a10239377805eb868a46c5b888214b8993c1d806c99034f2b0cbc4c5a3.jpg)  
(b) Reward vs. Search Rate (Hard Samples)  
Figure 5: Reward and search rate dynamics under different sample difficulties. Note that the blue line represents Average Reward and the red line represents Search Rate.

Search Behavior Divergence. Figure 4(a) compares retrieval rates across domains with different knowledge requirements. The retrieval rate is 14.25% on OntoNotes 5.0, where many entities can be recognized from local context and parametric knowledge. It increases to 24.60% on MIT-Restaurant and 30.83% on MIT-Movie, whose short queries often contain ambiguous names with limited contextual information. On GENIA, the rate reaches 52.18%, reflecting the greater need for external knowledge when recognizing specialized biomedical terminology. This variation suggests that NE-R1 adjusts its retrieval frequency to the knowledge demands of different domains instead of applying a uniform retrieval policy.

Speed–Quality Trade-off. We compare NE-R1 with Standard SFT and Standard RAG (alwaysretrieve) on MIT-Movie, as shown in Figure 4(b). Standard SFT achieves 85.24 Micro-F1 with a relative latency of 1.00×, whereas Standard RAG achieves only a modest improvement to 85.56 Micro-F1 while increasing latency to 4.85×. In comparison, NE-R1 achieves the highest Micro-F1 score of 91.05 with a relative latency of 2.19×, corresponding to an approximately 55% reduction in inference latency relative to Standard RAG. These results show that adaptive retrieval avoids part of the overhead of indiscriminate retrieval while obtaining more useful external evidence.

Reward Evolution. Figure 5 illustrates the evolution of average reward and retrieval rate for easy and hard samples. For easy samples (Figure 5a), the retrieval rate decreases while the reward remains high, indicating that the model increasingly relies on parametric knowledge and avoids unnecessary retrieval. For hard samples (Figure 5b), the retrieval rate increases as training progresses, allocating more retrieval to inputs for which external evidence is more likely to be useful. The opposing trends show that training does not simply encourage a globally higher or lower retrieval rate. Instead, they support that NE-R1 learns a difficulty-aware retrieval strategy conditioned on the expected utility of external evidence.

## 4.6 Retrieval Leakage Audit

NE-R1 does not directly retrieve test answers or benchmark annotations. The model generates entity-centric queries around uncertain mentions only when external knowledge is needed, rather than submitting complete test sentences to search engines. The retrieved content provides entityrelated background knowledge for semantic disambiguation, while predicted entity spans and types remain constrained by the original input and the predefined label set.

We conduct an exhaustive leakage audit across all four in-domain benchmarks: MIT-Movie, MIT-Restaurant, OntoNotes 5.0, and GENIA. Their test sets contain 1,953, 1,521, 8,262, and 1,854 instances, respectively, covering 13,590 test instances in total. For every instance that triggers retrieval, we manually inspect the generated query and all of its top-3 retrieved snippets. The exhaustive audit covers 3,120 retrieval-triggered instances and 9,360 query–snippet pairs in total.

We examine whether the retrieved content contains annotation-related terms, BIO tags or other structured NER outputs, gold entity lists, complete test sentences, or links to benchmark pages and annotation files. We find no cases in which retrieval causes test-answer leakage.

## 5 Conclusion

In this paper, we introduced NE-R1, an adaptive retrieval-augmented framework for NER that implements a “retrieval-on-demand” mechanism. NE-R1 first acquires parametric inference, retrieval triggering, and evidence-fusion capabilities through multi-task instruction tuning, and then jointly optimizes NER predictions and retrieval decisions through end-to-end reinforcement learning. This design allows the model to rely on parametric knowledge for familiar inputs and retrieve external evidence for knowledge-intensive or ambiguous entities.

Across the four evaluated in-domain benchmarks, NE-R1 improves the average F1 score by 2.52 points over the strongest overall baseline. In zero-shot transfer from CoNLL-03 to CrossNER, it improves performance in four of the five domains and achieves the best five-domain average, exceeding the strongest baseline by 1.18 F1 points. Behavioral analysis further shows that retrieval varies across domains and sample difficulties, while on MIT-Movie, NE-R1 reduces inference latency by approximately 55% relative to always-retrieve RAG. Overall, these results demonstrate that NE-R1 consistently outperforms existing generative and retrieval-augmented NER methods with higher efficiency, while generalizing robustly to unseen entity types.

## Limitations

First, NE-R1 relies on live, multi-source Web search. The retrieved content and its ranking may change over time because of API updates, indexing changes, and regional variation. Consequently, exact reproduction with live search services cannot be guaranteed, even under the same retrieval configuration.

Second, our empirical conclusions are limited to the evaluated English NER benchmarks and the current multi-source Web-search setting. The experiments do not establish that NE-R1 generalizes to other languages, domains, or retrieval backends. Moreover, heterogeneous search engines may return multilingual, conflicting, or uneven-quality evidence, and we do not separately quantify the effects of these factors.

Finally, the current framework supports only single-turn retrieval. While iterative retrieval may provide additional evidence for obscure or highly ambiguous entities, it may also introduce additional latency, noise, and error propagation. The main results follow the standard strict entity-level Micro-F1 evaluation protocol. Because repeated-run results or test predictions are unavailable for many comparison systems, we cannot conduct a uniform run-level variance or paired significance analysis across all baselines.

## References

Akbik, A., Bergmann, T., Blythe, D., Rasul, K., Schweter, S., and Vollgraf, R.. 2019. FLAIR: An easy-to-use framework for state-of-the-art NLP. In Proceedings of the 2019 conference of the North

American chapter of the association for computational linguistics (demonstrations), pages 54–59.

Ashok, D. and Lipton, Z.. 2023. Promptner: Prompting for named entity recognition. arXiv preprint arXiv:2305.15444.

Bogdanov, S., Constantin, A., Bernard, T., Crabbé, B., and Bernard, E.. 2024. Nuner: Entity recognition encoder pre-training via llm-annotated data. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing, pages 11829– 11841.

Chen, Xiang; Li, Lei; Deng, Shumin; Tan, Chuanqi; Xu, Changliang; Huang, Fei, et al.. 2022. LightNER: A lightweight tuning paradigm for low-resource NER via pluggable prompting. In Proceedings of the 29th international conference on computational linguistics, pages 2374–2387.

Chen, Xiang; Li, Lei; Qiao, Shuofei; Zhang, Ningyu; Tan, Chuanqi; Jiang, Yong, et al.. 2023. One model for all domains: collaborative domain-prefix tuning for cross-domain NER. In Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence, pages 5030–5038.

Chen, Z., Shi, L., Wu, W., Zhou, Q., and Zhang, Y.. 2025. ALLabel: Three-stage Active Learning for LLM-based Entity Recognition using Demonstration Retrieval. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 25173–25187.

Cocchieri, A., Galindo, M., Frisoni, G., Moro, G., Sartori, C., and Tagliavini, G.. 2025. ZeroNER: Fueling Zero-Shot Named Entity Recognition via Entity Type Descriptions. In Findings of the Association for Computational Linguistics: ACL 2025, pages 15594– 15616.

Ding, Z., Wei, W., Qu, X., and Chen, D.. 2024. Improving pseudo labels with global-local denoising framework for cross-lingual named entity recognition. arXiv preprint arXiv:2406.01213.

Dong, Qingxiu; Li, Lei; Dai, Damai; Zheng, Ce; Ma, Jingyuan; Li, Rui, et al.. 2024. A Survey on Incontext Learning. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing, pages 1107–1128.

Fan, Wenqi; Ding, Yujuan; Ning, Liangbo; Wang, Shijie; Li, Hengyun; Yin, Dawei, et al.. 2024. A survey on rag meeting llms: Towards retrieval-augmented large language models. In Proceedings of the 30th ACM SIGKDD conference on knowledge discovery and data mining, pages 6491–6501.

Feng, Zhaopeng; Cao, Shaosheng; Ren, Jiahan; Su, Jiayuan; Chen, Ruizhe; Zhang, Yan, et al.. 2025. Mt-r1-zero: Advancing llm-based machine translation via r1-zero-like reinforcement learning. arXiv preprint arXiv:2504.10160.

Gao, L., Ma, X., Lin, J., and Callan, J.. 2023. Precise zero-shot dense retrieval without relevance labels. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1762–1777.

Gao, Yunfan; Xiong, Yun; Gao, Xinyu; Jia, Kangxiang; Pan, Jinliu; Bi, Yuxi, et al.. 2023. Retrieval-Augmented Generation for Large Language Models: A Survey. arXiv preprint arXiv:2312.10997.

Gonen, H., Iyer, S., Blevins, T., Smith, N., and Zettlemoyer, L.. 2023. Demystifying prompts in language models via perplexity estimation. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2023, pages 10136–10148.

Guo, Q., Dong, Y., Tian, L., Kang, Z., Zhang, Y., and Wang, S.. 2025. BANER: Boundary-aware LLMs for few-shot named entity recognition. In Proceedings of the 31st International Conference on Computational Linguistics, pages 10375–10389.

Guo, Daya; Yang, Dejian; Zhang, Haowei; Song, Junxiao; Wang, Peiyi; Zhu, Qihao, et al.. 2025. DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning. Nature, 645:633–638.

He, Minggui; Liu, Yilun; Tao, Shimin; Luo, Yuanchang; Zeng, Hongyong; Su, Chang, et al.. 2025. R1-T1: Fully Incentivizing Translation Capability in LLMs via Reasoning Learning.

Hu, J., Zhao, H., Guo, D., Wan, X., and Chang, T.. 2022. A label-aware autoregressive framework for cross-domain NER. In Findings ofthe Association for Computational Linguistics: NAACL 2022, pages 2222–2232.

Hu, J.. 2025. Reinforce++: A simple and efficient approach for aligning large language models. arXiv e-prints.

Huang, L., Liu, H., Gao, Q., Yu, J., Liu, G., and Chen, X.. 2025. Adversity-aware few-shot named entity recognition via augmentation learning. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 24132–24140.

Huang, S., Xu, B., Yu, Y., Li, C., and Lin, X.. 2025. Guidener: Annotation guidelines are better than examples for in-context named entity recognition. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 24159–24166.

Huang, H., Chen, Y., Huang, R., Lin, C., and Qin, Y.. 2026. A reasoning paradigm for named entity recognition. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 31140–31148.

Jeong, S., Baek, J., Cho, S., Hwang, S., and Park, J.. 2024. Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 7036–7050.

Jia, C., Liang, X., and Zhang, Y.. 2019. Cross-domain NER using cross-domain language modeling. In Proceedings of the 57th annual meeting of the associationfor computational linguistics, pages 2464–2474.

Kang, K., Wallace, E., Tomlin, C., Kumar, A., and Levine, S.. 2025. Unfamiliar finetuning examples control how language models hallucinate. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter ofthe Associationfor Compu tational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 3600–3612.

Kim, J., Ohta, T., Tateisi, Y., and Tsujii, J.. 2003. GE-NIA corpus—a semantically annotated corpus for bio-textmining. Bioinformatics, 19:i180–i182.

Lample, G., Ballesteros, M., Subramanian, S., Kawakami, K., and Dyer, C.. 2016. Neural architectures for named entity recognition. In Proceedings ofthe 2016 conference ofthe North American chapter ofthe associationfor computational linguistics: human language technologies, pages 260–270.

Li, Y., Deng, S., Shen, D., Tian, S., and Long, S.. 2024. WkNER: Enhancing named entity recognition with word segmentation constraints and kNN retrieval. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 17651–17663.

Liu, J., Pasupat, P., Cyphers, S., and Glass, J.. 2013. Asgard: A portable architecture for multilingual dialogue systems. In 2013 IEEE International Conference on Acoustics, Speech and Signal Processing, pages 8386–8390.

Liu, Z., Winata, G., Xu, P., and Fung, P.. 2020. Coach: A coarse-to-fine approach for cross-domain slot filling. In Proceedings of the 58th annual meeting of the associationfor computational linguistics, pages 19–25.

Liu, Zihan; Xu, Yan; Yu, Tiezheng; Dai, Wenliang; Ji, Ziwei; Cahyawijaya, Samuel, et al.. 2021. Crossner: Evaluating cross-domain named entity recognition. In Proceedings of the AAAI conference on artificial intelligence, pages 13452–13460.

Liu, P., Yuan, W., Fu, J., Jiang, Z., Hayashi, H., and Neubig, G.. 2023. Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language processing. ACM computing surveys, 55:1– 35.

Liu, S., Shang, Y., and Zhang, X.. 2026. Truthfulrag: Resolving factual-level conflicts in retrievalaugmented generation with knowledge graphs. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 32168–32176.

Lu, Yaojie; Liu, Qing; Dai, Dai; Xiao, Xinyan; Lin, Hongyu; Han, Xianpei, et al.. 2022. Unified structure generation for universal information extraction. In Proceedings of the 60th annual meeting of the

associationfor computational linguistics (volume 1: long papers), pages 5755–5772.

Lu, J., Yang, Z., Wang, Y., Liu, X., Mac Namee, B., and Huang, C.. 2024. PaDeLLM-NER: parallel decoding in large language models for named entity recognition. Advances in neural information processing systems, 37:117853–117880.

Ma, X. and Hovy, E.H.. 2016. End-to-end sequence labeling via bi-directional lstm-cnns-crf. In Proceedings ofthe 54th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 1064–1074.

Nandi, S. and Agrawal, N.. 2024. Improving few-shot cross-domain named entity recognition by instruction tuning a word-embedding based retrieval augmented large language model. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track, pages 686–696.

Ouyang, Long; Wu, Jeffrey; Jiang, Xu; Almeida, Diogo; Wainwright, Carroll; Mishkin, Pamela, et al.. 2022. Training Language Models to Follow Instructions with Human Feedback. In Advances in Neural Information Processing Systems, pages 27730–27744.

Peng, Boci; Zhu, Yun; Liu, Yongchao; Bo, Xiaohe; Shi, Haizhou; Hong, Chuntao, et al.. 2025. Graph retrieval-augmented generation: A survey. ACM Transactions on Information Systems, 44:1–52.

Unknown. 2025. Reasoning-SQL: Reinforcement Learning with SQL Tailored Partial Rewards for Reasoning-Enhanced Text-to-SQL.

Rafailov, R., Sharma, A., Mitchell, E., Manning, C., Ermon, S., and Finn, C.. 2023. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36:53728–53741.

Ram, Ori; Levine, Yoav; Dalmedigos, Itay; Muhlgay, Dor; Shashua, Amnon; Leyton-Brown, Kevin, et al.. 2023. In-context retrieval-augmented language models. Transactions of the Association for Computational Linguistics, 11:1316–1331.

Sainz, O., García-Ferrero, I., Agerri, R., Lacalle, O., Rigau, G., and Agirre, E.. 2024. Gollie: Annotation guidelines improve zero-shot information-extraction. In International Conference on Learning Representations, pages 47083–47107.

Sang, E. and De Meulder, F.. 2003. Introduction to the CoNLL-2003 shared task: Language-independent named entity recognition. In Proceedings of the seventh conference on Natural language learning at HLT-NAACL 2003, pages 142–147.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O.. 2017. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.

Seow, W., Chaturvedi, I., Hogarth, A., Mao, R., and Cambria, E.. 2025. A review of named entity recognition: from learning methods to modelling paradigms and tasks. Artificial Intelligence Review, 58:315.

Shao, Z., Gong, Y., Shen, Y., Huang, M., Duan, N., and Chen, W.. 2023. Enhancing retrievalaugmented large language models with iterative retrieval-generation synergy. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2023, pages 9248–9274.

Song, Huatong; Jiang, Jinhao; Tian, Wenqing; Chen, Zhipeng; Wu, Yuhuan; Zhao, Jiahao, et al.. 2025. R1- searcher++: Incentivizing the dynamic knowledge acquisition of llms via reinforcement learning. arXiv preprint arXiv:2505.17005.

Tan, Zeqi; Huang, Shen; Jia, Zixia; Cai, Jiong; Li, Yinghui; Lu, Weiming, et al.. 2023. DAMO-NLP at SemEval-2023 Task 2: A Unified Retrieval-Augmented System for Multilingual Named Entity Recognition. In Proceedings of the 17th International Workshop on Semantic Evaluation (SemEval-2023), pages 2014–2028.

Tong, Z., Ding, Z., and Wei, W.. 2025. Evoprompt: Evolving prompts for enhanced zero-shot named entity recognition with large language models. In Proceedings ofthe 31st international conference on computational linguistics, pages 5136–5153.

Trivedi, H., Balasubramanian, N., Khot, T., and Sabharwal, A.. 2023. Interleaving retrieval with chain-ofthought reasoning for knowledge-intensive multi-step questions. In Proceedings of the 61st annual meeting ofthe associationfor computational linguistics (volume 1: long papers), pages 10014–10037.

Wang, Xiao; Zhou, Weikang; Zu, Can; Xia, Han; Chen, Tianze; Zhang, Yuansen, et al.. 2023. InstructUIE: Multi-task Instruction Tuning for Unified Information Extraction.

Wang, Shuhe; Sun, Xiaofei; Li, Xiaoya; Ouyang, Rongbin; Wu, Fei; Zhang, Tianwei, et al.. 2025. Gpt-ner: Named entity recognition via large language models. In Findings of the association for computational linguistics: NAACL 2025, pages 4257–4275.

Wang, Z., Chen, H., Xu, G., and Ren, M.. 2025. A novel large-language-model-driven framework for named entity recognition. Information Processing & Management, 62:104054.

Weischedel, Ralph; Palmer, Martha; Marcus, Mitchell; Hovy, Eduard; Pradhan, Sameer; Ramshaw, Lance, et al.. 2013. OntoNotes Release 5.0 LDC2013T19. Web Download. Philadelphia: Linguistic Data Consortium.

Xie, T., Li, Q., Zhang, J., Zhang, Y., Liu, Z., and Wang, H.. 2023. Empirical study of zero-shot NER with ChatGPT. In Proceedings of the 2023 conference on empirical methods in natural language processing, pages 7935–7956.

Xie, T., Li, Q., Zhang, Y., Liu, Z., and Wang, H.. 2024. Self-improving for zero-shot named entity recognition with large language models. In Proceedings of the 2024 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 2: Short Papers), pages 583–593.

Yan, S., Gu, J., Zhu, Y., and Ling, Z.. 2024. Corrective Retrieval Augmented Generation. arXiv preprint arXiv:2401.15884.

Yang, Yuming; Zhao, Wantong; Huang, Caishuang; Ye, Junjie; Wang, Xiao; Zheng, Huiyuan, et al.. 2025. Beyond Boundaries: Learning a Universal Entity Taxonomy across Datasets and Languages for Open Named Entity Recognition. In Proceedings of the 31st International Conference on Computational Linguistics, pages 10902–10923.

Zaratiana, U., Tomeh, N., Holat, P., and Charnois, T.. 2024. GLiNER: Generalist model for named entity recognition using bidirectional transformer. In Proceedings of the 2024 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 5364–5376.

Zhao, Penghao; Zhang, Hailin; Yu, Qinhan; Wang, Zhengren; Geng, Yunteng; Fu, Fangcheng, et al.. 2026. Retrieval-augmented generation for aigenerated content: A survey. Data Science and Engineering.

Zheng, J., Chen, H., and Ma, Q.. 2022. Cross-domain named entity recognition via graph matching. In Findings ofthe Associationfor Computational Linguistics: ACL 2022, pages 2670–2680.

Zheng, Yuxiang; Fu, Dayuan; Hu, Xiangkun; Cai, Xiaojie; Ye, Lyumanshan; Lu, Pengrui, et al.. 2025. Deepresearcher: Scaling deep research via reinforcement learning in real-world environments. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 414–431.

Zhenwei, DAI; Luo, Chen; Li, Zhen; Tang, Xianfeng; Lu, Hanqing; Goutam, Rahul, et al.. 2024. RA-NER: Retrieval augmented NER for knowledge intensive named entity recognition. In The Second Tiny Papers Track at ICLR 2024.

Zhou, W., Zhang, S., Gu, Y., Chen, M., and Poon, H.. 2024. Universalner: Targeted distillation from large language models for open named entity recognition. In International conference on learning representations, pages 12276–12294.

## A Appendix

## A.1 Dataset Statistics

Table 4 presents the detailed statistics of the datasets used in our experiments, including the number of sentences in train/dev/test splits, the number of entity types, the average token length, and the average number of entities per sentence.

In-Domain Datasets. We utilize a diverse set of benchmarks to assess generalizability across different data distributions:

• Standard & General Domain: CoNLL-03 (Sang, 2003) and Ontonotes 5.0 (Weischedel, 2013) represent standard news and general text domains. Ontonotes 5.0 is the largest dataset in our evaluation, containing over 59k training samples with 18 fine-grained entity types.

• Specialized Domain: GENIA (Kim, 2003) focuses on the biomedical domain. As shown in Table 4, it exhibits a high density of entities (Avg. entities = 3.5), reflecting the knowledgeintensive nature of scientific texts.

• Short-Text Domain: MIT-Movie and MIT-Restaurant (Liu, 2013) consist of informal user queries. These datasets have significantly shorter sentence lengths (Avg. tokens ≈ 9–11) compared to standard news corpora, posing challenges for context-dependent entity recognition.

Cross-Domain Datasets. For the cross-domain evaluation scenarios (trained on CoNLL-03), we employ the CrossNER benchmark (Liu, 2021). This dataset encompasses five distinct domains: AI, Literature, Music, Politics, and Science. While our main experiment primarily follows a zero-shot setting, Table 4 lists the standard data splits provided by the benchmark, which are typically used for low-resource or few-shot adaptation studies.

## A.2 Implementation Details

Hyperparameters. In Table 5, we present detailed values of hyperparameters used in our study, categorized into the Multi-Task Capability Initialization (MTCI) phase and the Reinforcement Learning (RL) phase. All experiments were conducted on 8 NVIDIA A800 GPUs using Qwen-2.5- 7B as the backbone model.

In the SFT phase, we utilized the AdamW optimizer with a learning rate of 1e-5 and a cosine learning rate scheduler. The training was conducted for 5 epochs with a warmup ratio of 0.1. We set the per-device batch size to 16 with gradient accumulation steps set to 1, and enabled bfloat16 precision to optimize memory usage.

In the RL phase, the model was further optimized using Group Relative Policy Optimization (GRPO), without introducing a critic network. We lowered the learning rate to 1e-6 with a constant scheduler and a 1% warmup phase. The training spanned 500 steps with a data collection batch size of 512, and GRPO policy updates were performed on mini-batches of size 256. The clipping ratio was set to 0.2, and we incorporated a KL divergence penalty with a coefficient of 0.001, utilizing a low-variance estimator (low\_var\_kl). Regarding model configuration, we limited the maximum prompt length to 4096 tokens and the response length to 500 tokens, supporting up to 2 interaction turns. During the rollout phase, we set the sampling temperature to 0.95 and generated 5 candidate responses per prompt (N=5). For retrievalaugmented generation, the top-3 relevant documents were retrieved. Some implementation frameworks retain generic policy-optimization parameter names such as mini-batch or clip-ratio settings; in our experiments, the actual advantage estimator is GRPO.

Search Engines. In the implementation of the retrieval module, we utilize multi-source web search engines (as shown in the table 6) as the external knowledge acquisition channels, and dynamically incorporate the retrieval results into the reasoning process of the model through a unified search interface.

## A.3 Supplementary Experiments and Analyses

Matched-rate random trigger baseline. To further isolate the effect of the learned retrieval policy, we compare NE-R1 with random retrieval policies under different fixed search rates. As shown in Table 7, random triggering improves over naive retrieval decisions in some settings, but remains consistently inferior to the adaptive policy of NE-R1. This confirms that NE-R1 does not merely benefit from using a certain retrieval budget; instead, its learned trigger policy better identifies when external evidence is needed.

<table><tr><td>Dataset</td><td># train</td><td># dev</td><td># test</td><td># types</td><td>Avg. tokens</td><td>Avg. entities</td></tr><tr><td>Ontonotes(Weischedel, 2013)</td><td>59924</td><td>8528</td><td>8262</td><td>18</td><td>18</td><td>0.9</td></tr><tr><td>GENIA(Kim, 2003)</td><td>15023</td><td>1669</td><td>1854</td><td>5</td><td>43</td><td>3.5</td></tr><tr><td>MIT-Movie(Liu, 2013)</td><td>6816</td><td>1000</td><td>1953</td><td>12</td><td>11</td><td>2.0</td></tr><tr><td>MIT-Restaurant(Liu, 2013)</td><td>7660</td><td>1000</td><td>1521</td><td>8</td><td>9</td><td>1.7</td></tr><tr><td>CoNLL-03(Sang, 2003)</td><td>14041</td><td>3250</td><td>3453</td><td>3</td><td>25</td><td>1.9</td></tr><tr><td>CrossNER AI(Liu, 2021)</td><td>100</td><td>350</td><td>431</td><td>13</td><td>52</td><td>5.3</td></tr><tr><td>CrossNER Literature(Liu, 2021)</td><td>100</td><td>400</td><td>416</td><td>11</td><td>54</td><td>5.4</td></tr><tr><td>CrossNER Music(Liu, 2021)</td><td>100</td><td>380</td><td>465</td><td>12</td><td>57</td><td>6.5</td></tr><tr><td>CrossNER Politics(Liu, 2021)</td><td>199</td><td>540</td><td>650</td><td>8</td><td>61</td><td>6.5</td></tr><tr><td>CrossNER Science(Liu, 2021)</td><td>200</td><td>450</td><td>543</td><td>16</td><td>54</td><td>5.4</td></tr></table>

Table 4: Dataset statistics.

<table><tr><td>Hyperparameter MTCI Phase</td><td>Value</td></tr><tr><td>Learning Rate LR Scheduler Warmup Ratio Num Epochs Per-Device Batch Size Gradient Accumulation RL Phase</td><td>1e-5 Cosine 0.1 5 16 1</td></tr><tr><td>Optimizer Learning Rate Warmup Ratio Total Steps Data Collection Batch Size Clip Ratio KL Coefficient KL Type</td><td>AdamW 1e-6 0.01 500 512 0.2 0.001 low_var_kl</td></tr></table>

Table 5: Detailed hyperparameter configuration for both MTCI and RL phases.

<table><tr><td>Search Engine</td><td>URL</td></tr><tr><td>Baidu Search</td><td>https://www.baidu.com</td></tr><tr><td>Google Search</td><td>https://www.google.com</td></tr><tr><td>Bing Search</td><td>https://www.bing.com</td></tr><tr><td>Wikipedia</td><td>https://www.wikipedia.org</td></tr></table>

Table 6: Search engines used in the retrieval module. All search engines are accessed through a unified retrieval interface.

Discussion on retrieval leakage risk. A potential concern for retrieval-augmented evaluation on public NER benchmarks is that a search engine may return benchmark annotations or otherwise create an “open-book” setting. NE-R1 mitigates this risk in three ways. First, the emitted query is not the original test sentence; it is an entity-centric rewritten query produced inside the <search> tag, such as a request for background information about an ambiguous mention. Second, the retrieved content is inserted as natural-language evidence through <information> tags and does not provide BIO labels or dataset-specific NER annotations. Third, the always-retrieve baseline does not dominate NE-R1, suggesting that merely increasing retrieval exposure is insufficient; the benefit comes from deciding when the external evidence is useful. The zero-shot cross-domain evaluation further supports that the model relies on transferable retrieval and reasoning behavior rather than memorizing benchmark-specific labels.

## A.4 MIT-Movie Prompt Templates

This section presents the prompt templates used in our MTCI and evaluation pipelines, using MIT-Movie as a representative example. The same prompting scheme is applied to other datasets by adapting the task description and entity label set accordingly. Specifically, the task prompts for parametric inference, retrieval triggering, and fusion inference, together with the evaluation-time adaptive retrieval prompt, are shown in figures 6, 7, 8, and 9, respectively.

Parametric inference prompt. This prompt initializes direct NER extraction without retrieval. The examples emphasize preserving the original span text and returning a compact JSON annotation.

Retrieval triggering prompt. This prompt teaches the model to separate clear cases from ambiguous or knowledge-intensive ones, producing a search query only when external evidence is likely to help.

<table><tr><td>Method</td><td>Search Rate</td><td>MIT-Movie</td><td>MIT-Rest.</td><td>OntoNotes</td><td>GENIA</td><td>Avg.</td></tr><tr><td>NE-R1 + Random</td><td>20%</td><td>88.73</td><td>84.51</td><td>88.76</td><td>78.94</td><td>85.24</td></tr><tr><td>NE-R1 + Random</td><td>35%</td><td>89.61</td><td>84.19</td><td>87.83</td><td>79.87</td><td>85.38</td></tr><tr><td>NE-R1 + Random</td><td>50%</td><td>88.12</td><td>83.44</td><td>86.71</td><td>80.63</td><td>84.73</td></tr><tr><td>NE-R1 (Adaptive)</td><td>14–52%</td><td>91.05</td><td>85.44</td><td>91.89</td><td>81.70</td><td>87.52</td></tr></table>

Table 7: Comparison between random retrieval triggering and NE-R1’s adaptive retrieval policy. Random triggering uses fixed retrieval rates, while NE-R1 adaptively adjusts its search rate across datasets according to input difficulty and domain knowledge requirements. All results are Micro-F1 scores.

![](images/42b9aa1064ed849abe7e07c8946010774765635d449f9520d58e95271a871b82.jpg)  
Figure 6: MIT-Movie prompt for the parametric inference task. This task trains the model to perform NER directly without external retrieval.

Fusion inference prompt. This prompt trains evidence-aware extraction. It explicitly instructs the model to use retrieved snippets for disambiguation while keeping final entity spans grounded in the original query.

Evaluation prompt. The evaluation prompt unifies the direct and retrieval-augmented paths. The model may search when necessary, but the final answer must always be a valid JSON annotation inside the answer tag.

## A.5 Case Study

Successful adaptive retrieval behavior. Figure 10 visualizes the inference paths of NE-R1 across varying sample difficulties, demonstrating its capability for uncertainty-driven on-demand retrieval. In the ambiguous Case 1, the span "Crash" is polysemous (common noun vs. movie title). Recognizing that context alone is insufficient, the model flags uncertainty and triggers retrieval to resolve the ambiguity, correctly identifying it as a Title. Conversely, in the confident Case 2 ("Steven Spielberg"), the model detects a clear intent pattern and performs direct prediction without retrieval overhead. This comparison indicates that retrieval decisions are driven by introspective judgment rather than heuristics: the model seeks external knowledge only when internal evidence is lacking, bypassing retrieval for simple cases to balance efficiency and accuracy. This validates that NE-R1 has effectively learned an intuitive "retrieval-ondemand" strategy. The corresponding visualized trajectories are shown in Figure 10.

False-positive and false-negative retrieval cases. Beyond successful examples, we also inspect failure modes of the learned retrieval policy. Figure 11 reports one false-positive retrieval case, where the model retrieves despite sufficient local evidence, and one false-negative case, where the model fails to retrieve for an ambiguous or knowledge-intensive mention. These cases clarify the remaining limitations of on-demand retrieval.

![](images/9d4a6a3fbc891d0b234094ecb0fff778f33a3b06e96d1422da1d303d49b901d2.jpg)

Figure 7: MIT-Movie prompt for the retrieval triggering task. This task initializes the model to reason about retrieval necessity and generate search queries.  
![](images/da22d60c6a98b7c04f89ca966d8234e5136d1571a291865b1775bd57fa7da36b.jpg)

Figure 8: MIT-Movie prompt for the fusion inference task. Retrieved evidence is provided to help the model resolve ambiguous or knowledge-intensive mentions before producing the final NER output.  
![](images/28923ea648bcc244d202818edbca809a1dec0a5880d7d90d76d703ee05f4c45a.jpg)  
Figure 9: MIT-Movie evaluation prompt with adaptive retrieval. The model either triggers search and fuses retrieved information, or directly returns the final NER answer.

![](images/ce12eed8d4dab18147306c31ae2895e62cbc51dd471f2e33d3a6de906f074e70.jpg)  
Figure 10: Different handling strategies for simple samples and difficult samples in NE-R1

![](images/41164c25ffe201448c335802281af02078e3903092dce580da6057f96ce8baa3.jpg)  
Figure 11: Representative false-positive and false-negative retrieval cases on MIT-Movie. The false-positive case shows redundant retrieval on an easy query, while the false-negative case shows a missed retrieval opportunity for a long-tail movie title.