# JUDGESTEALER: Extracting LLM Judging Capabilities across Evaluation Protocols

Chen Chen<sup>1</sup>, Yaolin Chen<sup>2</sup>, Xuehan Sun<sup>2</sup>, Juan Lin<sup>2</sup> Xueluan Gong<sup>1∗</sup>, Yuhang Zheng<sup>1</sup>, Qian Wang<sup>2</sup>, Kwok-Yan Lam<sup>1</sup>

<sup>1</sup>Nanyang Technological University, <sup>2</sup>Wuhan University

<sup>1</sup>{chen.chen, xueluan.gong, yuhang.zheng, kwokyan.lam}@ntu.edu.sg <sup>2</sup>{2023302181047, 2023302181211, 2024302183015, qianwang}@whu.edu.cn

## Abstract

Large language model (LLM) judges are increasingly used across various evaluation scenarios, making their judgment capabilities valuable intellectual property. However, black-box access exposes these capabilities to model extraction attacks. Existing extraction methods do not specifically target LLM judges and provide limited support for multiple evaluation protocols under restricted query budgets. In this study, we propose JUDGESTEALER, the first query-efficient model extraction framework for replicating judging capabilities across pointwise scoring, pairwise comparison, and listwise ranking protocols. JUDGESTEALER exploits the strong cross-protocol agreement to acquire pointwise scores and transform them into pairwise and listwise supervisions without additional victim queries. To capture informative judge patterns and improve query efficiency, JUDGESTEALER dynamically selects pointwise inputs based on semantic diversity, predictive uncertainty, and potential judge biases. It further applies score smoothing and multi-protocol review to preserve the ordinal structure of scores and mitigate catastrophic forgetting during surrogate adaptation. Extensive experiments on state-of-the-art LLM-as-a-judge and reward models show that JUDGESTEALER consistently outperforms existing extraction baselines, achieving up to 73.3%, 87.0%, and 71.6% accuracy for pointwise, pairwise, and listwise evaluation, respectively. JUDGESTEALER also remains effective across different surrogate model scales, adaptation strategies, and reasoning settings. Moreover, JUDGESTEALER demonstrates robustness against representative extraction defenses.

## 1 Introduction

Large language models (LLMs) have increasingly been extended with specialized capabilities, such as coding and content evaluation, supporting their deployment across diverse real-world applications [23, 24]. Developing such capabilities requires substantial resources, which makes them valuable intellectual property. To protect such capabilities, LLM service providers often expose them through accessible black-box interfaces, such as APIs. However, black-box access also introduces fundamental security risks: an adversary may repeatedly query the service and reproduce its behavior in a locally controlled model [41]. Such functionality stealing enables adversaries to bypass the original service provider, potentially causing significant economic losses.

Among the diverse capabilities of LLMs, content evaluation has become an important and broadly reusable functionality, with applications across at least three domains. First, LLMas-a-Judge is widely adopted to evaluate open-ended model outputs, providing a scalable alternative to costly human annotation for model evaluation and benchmarking [26, 50]. Second, Reward Models supply feedback signals for LLM alignment methods such as reinforcement learning from human feedback (RLHF) [34]. Since these signals serve as supervision during optimization, their quality can substantially influence the behavior and performance of the resulting policy. Third, downstream safety and moderation applications, including guardrail systems [15] and toxicity evaluators<sup>1</sup>, also rely on model-based judgment to support automated safety evaluation. Therefore, replicating this judgment functionality transfers a valuable asset for the development and deployment of downstream LLM applications.

Recent studies have focused on model extraction attacks (MEAs), which use natural [33, 35] or synthetic queries [18,42] to replicate task-specific functionality [4,6,9,33,35] or model components [3, 8, 32, 43]. Despite substantial progress, existing studies exhibit several limitations, as summarized in Table 1. First, the MEA literature is dominated by attacks against conventional deep neural networks, with only a limited number of studies focusing on LLMs [2, 3, 27, 28]. Moreover, to the best of our knowledge, none specifically investigates the extraction of evaluation capabilities from LLM judges. Second, several approaches assume prior knowledge of the victim architecture [32] or require access to logits and internal states. These assumptions limit applicability to realistic API settings where only hard labels are available [4,6,8,10,18,43]. Third, prior evaluations primarily consider earlier proprietary APIs, relatively small-scale models, or simulated victims, leaving recent large-scale models and modern proprietary services insufficiently explored. Finally, the effectiveness of existing attacks often degrades under low query budgets, and their robustness to practical defenses has not been sufficiently demonstrated.

Table 1: A comprehensive comparison of state-of-the-art model extraction attacks.
<table><tr><td rowspan="2">Method</td><td colspan="2">Attack†</td><td colspan="3">Victim Model‡</td><td colspan="2">Agnosticism$</td><td colspan="2">Performance*</td><td colspan="3">Robustness¶</td></tr><tr><td>Data</td><td>Surface</td><td>Domain</td><td>Close-weight</td><td>Open-weight</td><td>Architecture</td><td>Logits</td><td>Low budget</td><td>Normal budget</td><td>OT</td><td>OP</td><td>AD</td></tr><tr><td>KnockoffNet [33]</td><td>Nat.</td><td>Func.</td><td>CNN</td><td></td><td>~138M</td><td>√</td><td>√</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Activethief [35]</td><td>Nat.</td><td>Func.</td><td>CNN/RNN</td><td></td><td>~1.18M</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Inversenet [9]</td><td>Syn.</td><td>Func.</td><td>CNN</td><td></td><td>~0.7M</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DataFree [42]</td><td>Syn.</td><td>Func.</td><td>CNN</td><td></td><td>~180M</td><td></td><td>×</td><td></td><td></td><td></td><td>×</td><td></td></tr><tr><td>Maze [18]</td><td>Syn.</td><td>Func.</td><td>CNN</td><td></td><td>~0.27M</td><td></td><td>×</td><td></td><td></td><td></td><td>×</td><td></td></tr><tr><td>D-DAE [6]</td><td>Nat.&amp;Syn.</td><td>Func.</td><td>CNN</td><td></td><td>~138M</td><td></td><td>×</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Augmenting [10]</td><td>Nat.&amp;Syn.</td><td>Func.</td><td>CNN</td><td></td><td>~138M</td><td></td><td>×</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Depth extraction [8]</td><td>Nat.&amp;Syn.</td><td>Comp.</td><td>CNN</td><td></td><td>~168M</td><td></td><td>X</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Hyperparameter [43]</td><td>Nat.&amp;Syn.</td><td>Comp.</td><td>SVM/DNN</td><td></td><td>~1.33M</td><td></td><td>×</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Killing [4]</td><td>Nat.</td><td>Func.</td><td>PLM</td><td></td><td>~340M</td><td></td><td>×</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Model extraction [13]</td><td>Nat.</td><td>Func.</td><td>PLM</td><td></td><td>~340M</td><td></td><td>× √</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Thieves [21]</td><td>Nat.&amp;Syn.</td><td>Func.</td><td>PLM</td><td></td><td>~340M</td><td>√</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LLM-FIN [32]</td><td>Nat.</td><td>Comp.</td><td>Edge LLM</td><td></td><td>~770M</td><td>×</td><td>×</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Extracting [27]</td><td>Nat.</td><td>Func.</td><td>LLM</td><td>GPT-3.5</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Model Leeching [2]</td><td>Nat.</td><td>Func.</td><td>LLM</td><td>GPT-3.5</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Lion [16]</td><td>Syn.</td><td>Func.</td><td>LLM</td><td>GPT-3.5</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LoRD [28]</td><td>Nat.</td><td>Func.</td><td>LLM</td><td>GPT-3.5/4/4o</td><td>70B</td><td></td><td>×</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Ours</td><td>Nat.&amp;Syn.</td><td>Func.</td><td>LLM</td><td>GPT-5.4/Sonnet-4.5</td><td>1.6T</td><td></td><td>√</td><td></td><td></td><td></td><td></td><td></td></tr></table>

† Attack. Data denotes query type. Nat. and Syn. are natural and synthetic data, respectively. Surface denotes the extraction target. Func. is functionality and Comp. is model-component. ‡ Victim Model. Domain denotes the model family of the victim. Close-weight and Open-weight denote the strongest proprietary and open-source victim evaluated in each work. § Agnosticism. Architecture indicates the attack is agnostic to the victim model architecture, and Logits denotes the attack is independent of victim logits or probability outputs. ∗ Performance. Low budget (∼300 queries), : Acc<sub>±1</sub> < 30%, : 30% ≤ Acc<sub>±1</sub> < 50%, : 50% ≤ Acc<sub>±1</sub>; Normal budget (∼2000 queries), : 30% ≤ Acc<sub>±1</sub> < 60%, : 60% ≤ Acc<sub>±1</sub>. ¶ Robustness. The robustness against defense methods. OT, OP, and AD denote ownership tracing, output perturbation, and anomaly detection, respectively. – denotes “not evaluated”.

Extracting judging capabilities from LLMs is particularly challenging. Specifically, LLM judgments can take different forms, which can be broadly categorized into three evaluation protocols: pointwise scoring, pairwise comparison, and listwise ranking, each providing a distinct form of supervision [22,45]. A straightforward strategy is to extract each protocol independently. However, this strategy makes it difficult to achieve consistently strong performance across protocols. More importantly, separately querying the victim for each protocol incurs substantial query costs, limiting extraction efficiency under restricted query budgets. To this end, we propose JUDGESTEALER, the first model extraction framework designed to replicate the judging capabilities of black-box LLMs across multiple evaluation protocols. JUDGESTEALER addresses the following key challenges:

## C1. How to jointly support multiple evaluation protocols?

JUDGESTEALER identifies high evaluation agreement across protocols, where judgments derived from pointwise scores closely align with direct pairwise comparison and listwise rankings. JUDGESTEALER exploits this observation to extract cross-protocol LLM judgments. Specifically, JUDGESTEALER first acquires fine-grained pointwise supervisions and then transforms the collected scores into pairwise comparisons and listwise rankings. The original and transformed data are jointly used to construct a surrogate model. This strategy supports multiple protocols within a unified framework while substantially improving query efficiency.

## C2. How to identify informative pointwise inputs?

Under a limited query budget, selecting informative inputs is also critical to extraction efficiency. JUDGESTEALER introduces a dynamic input-selection mechanism that jointly captures semantic diversity, predictive uncertainty, and potential judge biases. These signals are aggregated into a selection score, and candidates with the highest scores are prioritized for querying the victim model. The selection scores are computed based on its latest surrogate state, enabling more accurate estimation of input informativeness.

## C3. How to effectively adapt a multi-protocol surrogate?

Since supervision for different protocols becomes available progressively, sequential adaptation may lead to catastrophic forgetting of previously acquired judgment behavior. JUDGESTEALER addresses this issue with a review strategy that revisits earlier pointwise supervision while incorporating newly constructed pairwise and listwise data. In addition, it applies an adaptive smoothing mechanism over the pointwise score space to regularize surrogate predictions and preserves the relative structure among neighboring evaluation scores.

We conduct extensive experiments on both proprietary and open-source LLM-as-a-judge models, including GPT-5.4, Claude Sonnet 4.5, Qwen3-235B-A22B, and DeepSeek V4 Pro, as well as a dedicated reward model, UniRRM. Across two instruction-following datasets and different surrogate models, JUDGESTEALER outperforms existing baselines, achieving up to 73.3%, 87.0%, and 71.6% accuracy for pointwise, pairwise, and listwise evaluation, respectively. Further experiments demonstrate that JUDGESTEALER remains effective across different surrogate model scales, adaptation strategies, and reasoning settings. We also evaluate

JUDGESTEALER against representative defenses, including anomaly detection, anti-distillation perturbation, and ownership tracing. The results demonstrate its practical robustness against these defense settings.

To conclude, we make the following contributions:

• We present JUDGESTEALER, the first model extraction framework for replicating LLM judging capabilities across pointwise scoring, pairwise comparison, and listwise ranking under black-box access.

• We develop a query-efficient extraction pipeline that combines dynamic informative-input selection, cross-protocol transformation, score smoothing, and multi-protocol review to construct a unified surrogate judge.

• We conduct extensive experiments across proprietary and open-source LLM-as-a-judge and reward models. The results demonstrate strong extraction performance across evaluation protocols, model scales, and adaptation strategies, while remaining effective against representative defenses.

## 2 Background

## 2.1 LLM-Based Judging

LLM-based judgments can be obtained through either a prompted evaluator or a learned reward function. Although they produce judgments differently, both can provide pointwise scores, pairwise preferences, and listwise rankings.

LLM-as-a-Judge. LLM-as-a-Judge uses an LLM to assess candidate responses according to evaluation instructions given in the prompt [37]. The prompt typically includes the user query, one or more candidate responses, evaluation criteria, and optional context such as a reference answer. In pointwise evaluation, the judge assigns a score to a single response. In pairwise evaluation, it compares two responses and identifies the better one, possibly allowing a tie. In listwise evaluation, it orders a set of responses from best to worst. The judgment may consist of only the final score or preference, or may also include a textual explanation. Since the criteria are supplied at inference time, a prompted judge can be used across different evaluation tasks, although its outputs may remain sensitive to prompt wording, response order, and decoding randomness.

Reward Model. A reward model learns a scalar function $R _ { \Phi } ( q , a )$ from preference data, where $q$ is the query and a is a candidate response. Given a preferred response $a ^ { + }$ and a rejected response $a ^ { - }$ , a common training objective is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { R M } } = - \mathbb { E } _ { ( q , a ^ { + } , a ^ { - } ) } \log \sigma \sigma \rho ( R _ { \oplus } ( q , a ^ { + } ) - R _ { \oplus } ( q , a ^ { - } ) ) , } \end{array}\tag{1}
$$

which encourages the preferred response to receive a higher reward. Reward models are commonly used in RLHF to provide feedback for policy optimization [7, 34]. At inference time, the learned reward can be used directly for pointwise evaluation, while comparing or sorting the rewards produces pairwise preferences or listwise rankings. Unlike a prompted judge, a reward model usually provides no textual explanation, and its evaluation criteria are determined by its training data.

## 2.2 Model Extraction Attacks

Model extraction (a.k.a. model stealing) aims to obtain a surrogate model that approximates a victim model accessible only through an API [4, 10, 11, 13, 21]. Early modelextraction studies primarily targeted image classification models [6, 9, 35], with only limited works targeting LLMs. Existing model extraction against LLM can be divided into two categories, i.e., functionality extraction and parameter/architecture extraction.

Functionality Extraction. Functionality extraction aims to obtain a surrogate model $G _ { \phi }$ whose input-output behavior closely matches that of a black-box victim model F. For example, Birch et al. [2] proposed Model Leeching, a costeffective functionality extraction attack that distills a blackbox LLM’s task-specific behavior (e.g., ChatGPT-3.5-Turbo) into a smaller local model. The attacker crafts a task-specific prompt template to generate queries, collects victim responses to form an imitation dataset $\mathcal { D } _ { F }$ , and fine-tunes a compact model (e.g., RoBERTa-Large) on $\mathcal { D } _ { F }$ to match the victim on the target task. Following the query-and-distill paradigm, Li et al. [27] examined extracting specialized code abilities from black-box LLM APIs. They generate code-task queries under different schemes (zero-shot, in-context, and zero-shot CoT), apply response checks to filter low-quality outputs, and fine-tune medium-sized code backbones (e.g., CodeBERT/- CodeT5) on the collected prompt–output pairs to obtain an imitation model. Recently, Liang et al. [28] proposed LoRD for alignment-aware extraction of RLHF-aligned LLMs. Instead of MLE distillation that directly maximizes the likelihood of the victim’s exact responses, LoRD treats the victim output as a local guide, constructs preferred vs. non-preferred samples in its neighborhood, and optimizes a policy-gradient objective to maximize their probability gap, improving query efficiency and offering stronger resilience to output watermarks.

Component Extraction. In contrast to functionality extraction, component extraction aims to recover internal model components, such as weights or architectural designs. This threat is particularly salient for edge-deployed models (e.g., smartphones and IoT devices), where attackers may obtain physical access [30, 49].

Carlini et al. [3] proposed stealing the complete embedding projection layer of a transformer language model. The key idea is to exploit the fact that the last linear map from hidden states to vocabulary logits is low-rank. Nazari et al. [32] proposed LLM-FIN, an LLM fingerprinting attack that infers the architectural identity (i.e., model family) of an edgedeployed LLM. The attacker passively collects side-channel resource traces, primarily RAM/memory-usage patterns (and optionally CPU/GPU loads) from an edge device while the LLM runs (e.g., via system monitoring such as tegrastats on

NVIDIA Jetson), then uses a supervised time-series classifier trained offline on labeled traces from known model families to predict the architecture family of the unknown victim model.

## 2.3 Model Extraction Defenses

Output-perturbation defenses. There are few LLMextraction defenses based purely on output perturbation. Most LLM APIs return only free-form text (no full probability vectors), leaving limited room to perturb outputs without harming quality. Although not designed specifically for LLMs, ModelGuard [39] offers a principled blueprint for output-perturbation defenses. It formulates perturbation as a utility-constrained optimization problem and provides an information-theoretic defense that remains robust against adaptive extractors by reducing the recoverable information in the returned outputs. When an LLM API exposes auxiliary signals (e.g., top-k logprobs/logits or embeddings), this blueprint can be instantiated by perturbing these signals (e.g., clipping, randomized rounding, or calibrated noise) while keeping the primary text response unchanged.

Watermarking and ownership tracing. Watermarking mainly supports attribution rather than preventing imitation, where providers embed detectable signals into generated text or service outputs and later verify ownership from a suspected stolen model [20]. Pang et al. proposed ModelShield [36], a plug-and-play black-box watermarking defense for LMaaS. It uses self-watermarking by prepending a system instruction that prompts the victim LLM to insert watermark words with minimal quality impact. Ownership is verified by scoring outputs for watermark-word presence and applying hypothesis testing (e.g., a one-sided t-test with a p-value threshold). For Embedding-as-a-Service (EaaS), Wang et al. propose GuardEmb [44], which perturbs embeddings for inputs containing selected special tokens and trains a verifier to distinguish watermarked from clean embeddings while preserving utility. A stolen embedding model trained on such outputs inherits the watermark, enabling infringement verification through targeted queries.

Query Detection. Query-detection defenses monitor API traffic to identify suspicious extraction behaviors before the attacker collects enough input-output pairs [29]. Compared with output perturbation and watermarking, dedicated querydetection defenses for LLM extraction are still limited; most ideas are adapted from traditional ML model-extraction defenses. For example, PRADA [17] records the incoming query sequence for each client and, for every new query, measures its distance to previous queries that receive the same predicted label. Benign users are expected to submit natural queries whose distance distribution is approximately normal, whereas extraction attacks often generate synthetic or systematically selected queries that distort this distribution. PRADA therefore applies a normality test to the query-distance distribution and raises an alarm when the deviation is sufficiently large. SEAT [48] further learns a similarity encoder to detect extraction-like query sequences, especially when attackers submit many highly similar or structured queries.

## 3 Threat Model

Attacker Scenario. We consider an LLM service that provides proprietary content evaluation functionality. The victim may be a dedicated reward model or a general-purpose LLM whose judging capability can be invoked via user prompts. We assume the attacker is a malicious user of the service who repeatedly queries the victim to collect its response.

Attacker’s Capability. We consider a black-box setting, where the attacker submits queries to the victim and observes only the outputs through the service interface. The attacker has no access to the victim’s internal information, including its training data, architecture, model parameters, and output logits. We further assume that the attacker can train a local surrogate model and may use publicly available or synthetically generated data to construct candidate queries. We do not assume any knowledge of the victim model family or the data used to develop its evaluation capability.

Attacker’s Goal. A successful extraction attack on judging capability should satisfy three objectives.

• Attack Effectiveness. The extracted surrogate should faithfully reproduce the victim’s judging behavior across different inputs and evaluation protocols, including pointwise, pairwise, and listwise evaluation.

• Attack Efficiency. An effective attack should maximize the information obtained from each victim interaction and reduce the number of queries required to construct a surrogate model due to the costs and rate limits imposed by victims.

• Attack Stealthiness. The query process should resemble normal usage and avoid exposing recognizable extraction patterns, such as abnormal query distributions or excessive repetition, to the service provider.

## 4 Methodology

## 4.1 Overview

Intuition. In our model extraction setting, we consider an LLM-based judging system that supports multiple evaluation protocols, including pointwise scoring, pairwise comparison, and listwise ranking. While these settings are different in their input structures and output formats, we hypothesize that their judgments are guided by a shared underlying evaluation criterion for assessing response quality. We empirically examine this hypothesis across multiple victim models, datasets, and evaluation settings by transforming pointwise scores into pairwise and listwise judgments and measuring their agreement with direct victim outputs. As shown in Figure 1, the average agreement reaches 88.70%, 88.67%, and 94.95% for pointwise–pairwise, pointwise–listwise, and pairwise–listwise comparisons, respectively, demonstrating strong cross-protocol consistency across these capable LLM judges. Moreover, this consistency generally increases with judge capability, where weaker judges exhibit lower cross-protocol agreement, and stronger judges maintain substantially more consistent evaluation behavior, as shown in Figure 8. This trend suggests that such consistency may constitute an important component of robust judging capability. Importantly, this property is exploitable for our multi-protocol model extraction, particularly under limited query budgets. If different protocols largely reflect the same underlying evaluation criterion, querying the victim separately under every protocol becomes unnecessary. Motivated by this observation, JUDGESTEALER acquires informative pointwise supervision and transforms it into corresponding pairwise and listwise training signals without additional victim queries. Consequently, victim supervision can be reused across protocols, enabling joint improvement in multi-protocol extraction performance while substantially increasing query efficiency.

![](images/df3f291f6284b942c93b966b2ca2f8ac83530f8d01748589373020455133680f.jpg)  
(a) Agreement across protocols

![](images/1a72a9d3fd789ebad001558053992a1fddaec07058966abe68a543e97fc09d52.jpg)  
(b) Agreement across victim model and dataset  
Figure 1: Cross-protocol Agreement.

Workflow. Figure 2 presents the overall workflow of JUDGESTEALER, which comprises two stages. In Stage I, the attacker begins with a pool $\mathcal { G }$ of multi-response candidate instances and introduces a dynamic selection mechanism to identify informative instances based on the current state of the surrogate model $M _ { \Phi } ,$ , parameterized by φ. This mechanism jointly evaluates semantic diversity, predictive uncertainty, and potential judging bias. For the selected instance $G ,$ the attacker submits each query–response pair $( q , a _ { i } ) \in G$ to the victim under the pointwise protocol and collects the returned scores $y _ { i }$ as supervision. These labeled samples constitute the pointwise scoring dataset $S _ { : }$ , which is used to train an initial surrogate model $M _ { \Phi }$ . Stage II exploits cross-protocol agreement to extend the collected supervision without submitting additional victim queries. Specifically, the pointwise scores are transformed into the pairwise comparison dataset $c$ and the listwise ranking dataset ${ \mathcal { R } } .$ , respectively. These samples are used to surrogate model adaptation, and subsequently combined with the original pointwise data to construct a multi-protocol training set for surrogate model consolidation. To mitigate the influence of overconfident supervision, JUDGESTEALER further incorporates an adaptive smoothing mechanism using a Gaussian prior over the score space.

## 4.2 Stage I: Pointwise Score Extraction

JUDGESTEALER builds on the cross-protocol agreement of LLM judges, where supervision obtained under one evaluation protocol can be transformed into training signals for other protocols. Among the three protocols, pointwise scores provide the most fine-grained and versatile supervision. Specifically, given scores for multiple responses to the same query, pairwise comparison can be inferred by comparing their scores, while listwise rankings can be constructed by ordering the responses accordingly. However, the reverse transformation is ambiguous. We therefore adopt pointwise scoring as the primary querying protocol.

## 4.2.1 Sample Selection

JUDGESTEALER first collect a set of unlabeled candidate instances $\mathcal { G } ,$ where each instance $G \in { \mathcal { G } }$ consists of a user query $q$ and a set of corresponding responses $A = \{ a _ { i } \} _ { i = 1 } ^ { n }$ . Equivalently, each instance G can be represented as a collection of query-response pairs, i.e., $G = \{ ( q , a _ { i } ) \} _ { i = 1 } ^ { n } .$ To effectively identify informative instances under a limited query budget, we introduce an iterative selection mechanism. At iteration $t \in \{ 1 , 2 , . . . , T \}$ , JUDGESTEALER maintains a set of previously selected instances $\mathcal { G } _ { \leq t } \subset \mathcal { G }$ . We evaluate the remaining candidate $G \in { \mathcal { G } } \backslash { \mathcal { G } } _ { < t }$ using three signals, i.e., semantic diversity $D ( G )$ , predictive uncertainty $U ( G )$ , and potential judge biases $B ( G )$ . These signals are aggregated into a selection score $\Gamma ( G )$ , and the highest-scoring candidates are selected to form the current query batch $\mathcal { G } _ { t }$

Semantic Diversity. The selected instances should cover different regions of the judge’s input space rather than repeatedly querying semantically similar content. To this end, JUDGESTEALER adopts a distance-based selection method with density filtering. For each candidate instance $G ,$ we first compute its representation

$$
z _ { G } = { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } f _ { \mathrm { e m b } } ( q , a _ { i } ) , \quad ( q , a _ { i } ) \in G ,\tag{2}
$$

where $f _ { \mathrm { e m b } }$ denotes a text embedding model. Given $\scriptstyle { \mathcal { G } } _ { < t }$ , the diversity value is defined as:

$$
D ( G ) = \operatorname* { m i n } _ { H \in { \mathcal G } _ { < t } } d ( z _ { G } , z _ { H } ) ,\tag{3}
$$

where $d ( \cdot , \cdot )$ denotes cosine distance. A larger $D ( G )$ indicates that $G$ covers a semantic region that is insufficiently represented by the previously selected instances. However, this

![](images/5b18cddfed48661eedfb8bf158d2d89536e5e11730cb1a95ebbe9892ab498b98.jpg)  
Figure 2: Overview of JUDGESTEALER. In Stage I, JUDGESTEALER identifies informative instances, submits them to the victim under the pointwise protocol, and trains an initial surrogate model. In Stage II, the pointwise supervision is transformed into pairwise and listwise data, which are used to train the surrogate model. A smoothing mechanism is applied in both stages.

distance-only strategy may prioritize isolated outliers that are not representative of the underlying data distribution. We therefore introduce a local-density estimate:

$$
\rho ( G ) = \left( \frac { 1 } { k } \sum _ { H \in \mathcal { N } _ { k } ( G ) } d ( z _ { G } , z _ { H } ) + \varepsilon \right) ^ { - 1 } ,\tag{4}
$$

where $\mathcal { N } _ { k } ( \cdot )$ denotes k nearest neighbours of $G ,$ and ε is small constant for numerical stability. Candidates whose $\rho ( G )$ fall within the lowest 10% of the candidate pool are excluded from selection.

Predictive Uncertainty. Informative instances are expected to be at the decision boundary and exhibit high predictive uncertainty on the surrogate model $M _ { \Phi }$ . Given a sample $( q , a _ { i } ) \in G$ and score space ${ \mathcal { T } } ,$ , the surrogate model produces a probability distribution $\mathbf { p } _ { i } = M _ { \Phi } ( \cdot \mid \mathsf { P } _ { s } , q , a _ { i } )$ , where $\mathsf { P } _ { s }$ denotes the prompt for pointwise scoring. We quantify the uncertainty using the normalized entropy of $\mathbf { p } _ { i } \mathbf { \dot { \mathbf { \cdot } } }$

$$
U ( G ) = \frac { 1 } { n \log | \mathcal { Y } | } \sum _ { i = 1 } ^ { n } \mathrm { E n t } ( \mathbf { p } _ { i } ) .\tag{5}
$$

where Ent(·) denotes the Shannon Entropy<sup>2</sup>. Higher entropy indicates greater predictive uncertainty, which suggests the victim label is more likely to provide high-value supervision.

Judge Biases. JUDGESTEALER also prioritizes candidates that can expose systematic biases in the surrogate judge [50]. We examine two common forms of judge bias, i.e., verbosity bias and position bias. To assess verbosity bias, we maintain a collection of neutral and non-informative prefixes $\mathcal { W } .$ They are introduced to increase the response length without affecting its underlying quality. For a candidate $G ,$ we first select the prefix whose representation is most similar to $z _ { G }$

$$
w ^ { \star } = \arg \operatorname* { m a x } _ { w \in \mathcal { W } } \mathrm { C o s S i m } ( f _ { \mathrm { e m b } } ( w ) , z _ { G } ) ,\tag{6}
$$

where $\mathbf { C o s S i m } ( \cdot , \cdot )$ denotes cosine similarity. We then prepend $w ^ { \star }$ to each response $a _ { i }$ to obtain the extended response $a _ { i } ^ { + } = w ^ { \star } \oplus a _ { i } .$ , where ⊕ denotes text concatenation. The surrogate model produces output distribution for both the original response $\mathbf { p } _ { i } = M _ { \Phi } ( \cdot \mid \mathsf { P } _ { s } , q , a _ { i } )$ and the extended responses $\mathbf { p } _ { i } ^ { + } = M _ { \Phi } \bigl ( \cdot \mid \mathsf { P } _ { s } , q , a _ { i } ^ { + } \bigr )$ , which are subsequently used to quantify the sensitivity of G to verbosity

$$
B _ { \mathrm { v } } ( G ) = \frac { 1 } { n | \mathcal { N } | } \sum _ { i = 1 } ^ { n } \operatorname* { m a x } \left( 0 , \sum _ { y \in \mathcal { Y } } y \left( \mathbf { p } _ { i } ^ { + } ( y ) - \mathbf { p } _ { i } ( y ) \right) \right) .\tag{7}
$$

To assess position bias, we evaluate each response pair $a _ { j }$ and $a _ { k }$ under both orders. The surrogate model produces the corresponding output distributions as $\mathbf { p } _ { j , k } =$ $M _ { \Phi } ( \cdot \mid \mathsf { P } _ { c } , q , a _ { j } , a _ { k } )$ and $\mathbf { p } _ { k , j } = M _ { \Phi } \big ( \cdot \mid \mathsf { P } _ { c } , q , a _ { k } , a _ { j } \big )$ , where $\mathsf { P } _ { c }$ denotes the pairwise-comparison prompt. The position-bias score is defined as

$$
B _ { \mathrm { p } } ( G ) = { \frac { 1 } { n ( n - 1 ) } } \sum _ { \substack { j = 1 } } ^ { n } \sum _ { \substack { k = 1 } } ^ { n } D _ { \mathrm { K L } } ( \mathbf { p } _ { j , k }  \mathbf { p } _ { k , j } ) ,\tag{8}
$$

where $D _ { \mathrm { K L } } ( \cdot , \cdot )$ is the KL divergence between two distributions. These two signals are averaged into judge bias score.

$$
B ( G ) = \frac { B _ { \mathrm { v } } ( G ) + B _ { \mathrm { p } } ( G ) } { 2 } .\tag{9}
$$

Overall Selection Score. Finally, we obtain the overall score:

$$
\Gamma ( G ) = \lambda _ { 1 } D ( G ) + \lambda _ { 2 } U ( G ) + \lambda _ { 3 } B ( G ) ,\tag{10}
$$

where $\lambda _ { 1 } , \lambda _ { 2 } , \lambda _ { 3 } \geq 0$ control the contributions of the three components. To avoid the cost of evaluating all remaining candidates, at iteration t, JUDGESTEALER first samples a candidate subset ${ \widetilde { \mathcal { G } } } _ { t } \subset { \mathcal { G } } \backslash { \mathcal { G } } _ { < t }$ and then selects the K instances with the highest scores:

$$
G _ { t } = \mathrm { T o p K } \Gamma ( G ) .\tag{11}
$$

During the first few iterations, we skip the selection mecha nism and randomly sample instances to initialize $\scriptstyle { \mathcal { G } } _ { < t }$

## 4.2.2 Victim Scoring

Given the selected batch $\mathcal { G } _ { t }$ , JUDGESTEALER queries the victim judge and consumes the corresponding query budget. Specifically, for each $G \in G _ { t } ,$ we submit every query-response pair $( q , a _ { i } ) \in G$ under the pointwise evaluation protocol, and obtains the score returned by the victim model $M _ { \nu } \mathrm { : }$

$$
y _ { i } = M _ { \nu } ( \mathsf { P } _ { s } , q , a _ { i } ) , \quad y _ { i } \in \mathcal { Y } .\tag{12}
$$

When the API additionally provides a Chain-of-Thought (CoT) explanation, we retain it only as auxiliary metadata and use the scalar score as the primary supervision signal.

The resulting victim-labeled instance is represented as $G ^ { * } = \{ ( q , a _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n }$ , which yields n pointwise samples

$$
s _ { i } = ( q , a _ { i } , y _ { i } ) .\tag{13}
$$

These samples form the pointwise dataset $S _ { t }$ at iteration t and are accumulated into the overall dataset $S \gets S \cup S _ { t } . \ S$ will be subsequently used to construct pairwise and listwise supervision, and train the surrogate judge in Stage II.

## 4.2.3 Surrogate Training

JUDGESTEALER uses the pointwise dataset $S _ { t }$ obtained at iteration t to update the surrogate model, without performing cross-protocol transformation at this moment. We adopt this strategy for two main reasons. First, the selection mechanism relies on the surrogate’s predictive distribution. Therefore, updating the surrogate after each iteration ensures that subsequent selection decisions reflect the latest victim supervision. Second, transforming the currently available pointwise scores into pairwise and listwise labels would generate mul tiple highly correlated training samples from the same set of queries and responses. Introducing these samples during the early iterations may overemphasize a small number of instances and cause the surrogate to overfit to redundant su pervision. Accordingly, we first train the surrogate on the victim-labeled pointwise data as a stable warm-up stage before applying multi-protocol surrogate adaptation.

Moreover, scoring labels exhibit an inherent ordinal structure: scores closer to the victim-assigned label should generally receive higher probabilities than more distant ones. However, standard one-hot supervision does not encode this ordinal structure, which prevents the surrogate from preserving such a locality pattern among scores. To incorporate this prior, we smooth the hard one-hot target with a discrete Gaussian distribution centered at the victim-assigned score. Specifically, for a sample $( q , a , y ) \in S _ { t }$ , the target distribution is defined as

Algorithm 1: Stage I: Pointwise Score Extraction   
Input : Unlabeled candidate pool $\mathcal { G } ;$ victim $M _ { \nu } ;$ initial surrogate   
$M _ { \Phi } ;$ pointwise prompt $\mathbf { P } _ { s } ;$ batch size $K ;$ number of nearest   
neighbors k; selection weights $( \lambda _ { 1 } , \lambda _ { 2 } , \lambda _ { 3 } ) .$   
Output : Pointwise dataset S; updated surrogate judge $M _ { \Phi } .$   
1 Initialize $S  \emptyset$ and $G _ { < 1 } \gets 0$   
2 for $t = 1 , 2 , \dots$ while the remaining budget is sufficient do   
3 Sample a candidate subset $\tilde { g } _ { t } \subseteq \mathcal G \backslash \mathcal G _ { < t }$   
4 foreach $G \in \widetilde { { \mathcal G } } _ { t }$ do   
5 Compute the instance representation $z _ { G }$   
6 Estimate the local density $\rho ( G )$   
7 if G is not excluded by densityfiltering then   
8 Compute diversity $D ( G )$ (omit it when $G _ { < t } = \emptyset )$   
9 Compute predictive uncertainty $U ( G )$ using $M _ { \Phi }$   
10 Compute bias scores $B ( G )$   
11 $\Gamma ( G ) ^ { \bullet } - \lambda _ { 1 } D ( G ) + \lambda _ { 2 } { \dot { U } } ( { \dot { G } } ) + \lambda _ { 3 } B ( G )$   
12 ${ \mathcal { G } } _ { t } \gets \mathrm { T o p K } _ { G \in { \widetilde { \mathcal { G } } } _ { t } } \Gamma ( G )$   
13 Initialize $\mathbf { \boldsymbol { S } } _ { t } \gets \mathbf { \boldsymbol { \mathbb { 0 } } }$   
14 foreach $G = \{ ( q , a _ { i } ) \} _ { i = 1 } ^ { n } \in { \mathcal { G } } _ { t }$ do   
15 for i = 1 to n do   
16 $y _ { i } \gets M _ { \nu } ( \mathbf { P } _ { s } , q , a _ { i } )$   
17 $S _ { t } \gets S _ { t } \cup \{ ( q , a _ { i } , y _ { i } ) \}$   
18 $S \gets S \cup S _ { t }$   
19 Update $M _ { \Phi }$ on $S _ { t }$ by minimizing $\mathcal { L } _ { \mathrm { p o i n t } } ( \phi )$ with   
Gaussian-smoothed targets   
20 $\scriptstyle { { \mathcal { G } } < t + 1 }  { \mathcal { G } } < t \cup { \mathcal { G } } _ { t }$   
21 Update the remaining query budget   
22 return ${ \mathcal { S } } , M _ { \emptyset }$

$$
\begin{array} { r } { \widetilde { \mathbf { e } } _ { y } = ( 1 - \lambda ) \mathbf { e } _ { y } + \lambda \mathbf { g } _ { y } , } \end{array}\tag{14}
$$

where $\mathbf { e } _ { y }$ denotes the one-hot distribution that assigns probability 1 to y and 0 to all other scores. ${ \bf g } _ { y }$ denotes the probability mass function of a discrete Gaussian distribution over the score space centered at $y \left[ 1 9 \right] . \lambda \in \left[ 0 , 1 \right]$ controls the smoothing strength. In our setting, $\lambda$ is trainable, which enables an adaptive smoothing mechanism during training.

Accordingly, we optimize the surrogate model $M _ { \Phi }$ using the cross-entropy loss:

$$
\mathcal { L } _ { \mathrm { p o i n t } } ( \boldsymbol { \Phi } ) = \frac { 1 } { | \mathcal { S } _ { t } | } \sum _ { ( \boldsymbol { q } , \boldsymbol { a } , \boldsymbol { y } ) \in \mathcal { S } _ { t } } \mathrm { C E } ( \widetilde { \mathbf { e } } _ { \boldsymbol { y } } , \mathbf { p } ) ,\tag{15}
$$

where $\mathbf { p } = M _ { \Phi } \left( \cdot \mid \mathsf { P } _ { c } , q , a \right)$ denotes the score distribution predicted by the surrogate model. $\operatorname { C E } ( \cdot , \cdot )$ denotes the crossentropy between the target and predicted distributions.

## 4.3 Stage II: Multi-Protocol Extension

At this stage, JUDGESTEALER further extends the extracted judging capability across evaluation protocols. It first transforms the collected pointwise scores into corresponding pairwise comparisons and listwise rankings, and then integrates all three forms of supervision into a multi-protocol training dataset. Starting from the surrogate obtained in Stage I, JUDGESTEALER adapts the model using the transformed supervision and subsequently performs a consolidation procedure over the multi-protocol dataset. Importantly, this stage requires no additional queries to the victim model.

## 4.3.1 Cross-Protocol Transformation

Pairwise Sample Construction. Given a victim-labeled pointwise instance $G ^ { \star } = \{ ( q , a _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n }$ , JUDGESTEALER samples two responses $a _ { j }$ and $a _ { k } ,$ , where $1 \leq j < k \leq n ,$ and infer their pairwise comparison by comparing the corresponding pointwise scores:

$$
\begin{array} { r } { y _ { j , k } = \left\{ \begin{array} { l l } { a _ { j } \succ a _ { k } , } & { y _ { j } > y _ { k } , } \\ { a _ { j } \prec a _ { k } , } & { y _ { j } < y _ { k } , } \\ { a _ { k } = a _ { j } , } & { y _ { j } = y _ { k } , } \end{array} \right. } \end{array}\tag{16}
$$

where $\succ , \prec , \sim$ denote superiority, inferiority, and a tie, respectively. The resulting pairwise training sample is

$$
\boldsymbol { c } _ { j , \boldsymbol { k } } = \left( q _ { i } , a _ { j } , a _ { k } , y _ { j , \boldsymbol { k } } \right) .\tag{17}
$$

An victim-labeled instance with n responses yields at most ${ \binom { n } { 2 } } = n ( n - 1 ) / 2$ distinct response pairs. JUDGESTEALER samples $n _ { c }$ of these pairs from each instance and aggregates the resulting samples into the pairwise comparison dataset $c .$

Listwise Sample Construction. To construct listwise supervision, JUDGESTEALER draws a subset of $n _ { l }$ responses from each victim-labeled pointwise instance $G ^ { \star } = \{ ( q , a _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n } .$ Let $I = \{ j _ { 1 } , j _ { 2 } , . . . , j _ { n _ { l } } \}$ denote the indices of the sampled responses, where $j _ { * } \in \left\{ 1 , 2 , \ldots , n \right\}$ . The resulting subset is denoted by $\begin{array} { r } { G _ { I } ^ { \star } = \big \{ ( q , \dot { a } _ { j _ { r } } , y _ { j _ { r } } ) \big \} _ { r = 1 } ^ { n _ { l } } } \end{array}$ . JUDGESTEALER then ranks these responses in descending order according to their pointwise scores. Specifically, let π denote a permutation of the indices in I satisfying $y _ { \pi ( 1 ) } \geq y _ { \pi ( 2 ) } \geq \dots \geq y _ { \pi ( n _ { l } ) }$ . The corresponding listwise label is represented as

$$
y _ { I } = a _ { \pi ( 1 ) } \succeq a _ { \pi ( 2 ) } \succeq \cdots \succeq a _ { \pi ( n _ { l } ) } .\tag{18}
$$

where ≻ indicates that the preceding response is ranked no lower than the following response. If two responses receive the same pointwise score, they are assigned the same ranking position. The corresponding listwise training sample is

$$
r _ { I } = ( q , A _ { I } , y _ { I } ) .\tag{19}
$$

where $A _ { I } = \{ a _ { j _ { r } } \} _ { r = 1 } ^ { n _ { l } }$ . Each pointwise instance can yield at most $\binom { n } { n _ { l } }$ distinct listwise samples. JUDGESTEALER samples $n _ { r }$ of these combinations and collect them into the listwise ranking dataset $\mathcal { R }$

Algorithm 2: Stage II: Multi-Protocol Expansion   
Input : Victim-labeled pointwise dataset ${ \mathcal { S } } ;$ surrogate judge $M _ { \Phi } ;$   
number of sampled pairwise instances $n _ { c } ;$ number of   
responses per listwise instance $n _ { l } ;$ number of sampled   
listwise instances $n _ { r } .$   
Output : Pairwise dataset $c ;$ listwise dataset ${ \mathcal { R } } ;$ updated surrogate   
judge $M _ { \Phi } .$   
1 Initialize $c  \emptyset$ and $\mathcal { R }  \emptyset$   
2 foreach victim-labeled instance $G ^ { \star } = \{ ( q , a _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n }$ do   
3 Sample $n _ { c }$ distinct response pairs from $G ^ { \star }$   
4 foreach sampled response pair $( a _ { j } , a _ { k } )$ do   
$\left( a _ { j } \succ a _ { k } , \quad y _ { j } > y _ { k } , \right.$   
5 $y _ { j , k }  \{ \thinspace a _ { j } \prec a _ { k } , \quad y _ { j } < y _ { k } , $   
$\begin{array} { r } { \lfloor a _ { j } \sim a _ { k } , \quad y _ { j } = y _ { k } ; } \end{array}$   
6 $c _ { j , k } \gets ( q , a _ { j } , a _ { k } , y _ { j , k } )$   
7 $C \gets C \cup \{ c _ { j , k } \}$   
8 for $r = 1$ to $n _ { r }$ do   
9 Sample an index set $I = \{ j _ { 1 } , j _ { 2 } , \dotsc , j _ { n _ { l } } \} \subseteq \{ 1 , 2 , \dotsc , n \}$   
10 $A _ { I }  \{ a _ { j _ { 1 } } , a _ { j _ { 2 } } , . . . , a _ { j _ { n _ { l } } } \}$   
11 Find a permutation π such that $y _ { \pi ( 1 ) } \geq y _ { \pi ( 2 ) } \geq \dots \geq y _ { \pi ( n _ { l } ) }$   
12 $y _ { I }  a _ { \pi ( 1 ) } \succeq a _ { \pi ( 2 ) } \succeq \cdots \succeq a _ { \pi ( n _ { l } ) }$   
13 $r _ { I } \gets ( q , A _ { I } , y _ { I } )$   
14 $\mathcal { R }  \mathcal { R } \cup \{ r _ { I } \}$   
15 Update $M _ { \Phi }$ on $c$ by minimizing $\mathcal { L } _ { \mathrm { p a i r } } ( \boldsymbol { \Phi } )$   
16 Update $M _ { \Phi }$ on $\mathcal { R }$ by minimizing $\bar { \mathcal { L } } _ { \mathrm { l i s t } } ( \boldsymbol { \Phi } )$   
17 Construct the multi-protocol training dataset ${ \mathcal { T } } \gets S \cup C \cup { \mathcal { R } }$   
18 Update $M _ { \Phi }$ on T by minimizing $\scriptstyle { \mathcal { L } } _ { \mathrm { c o n } } ( \phi )$   
19 return ${ \mathcal { C } } , { \mathcal { R } } , M _ { \Phi }$

## 4.3.2 Progressive Multi-Protocol Training

After obtaining the pointwise dataset S in Stage I and constructing the pairwise and listwise datasets C and R through cross-protocol transformation, JUDGESTEALER progressively adapts the surrogate to all three evaluation protocols.

Pairwise and Listwise Adaptation. We adapt the surrogate to pairwise and listwise protocols. Specifically, the surrogate $M _ { \Phi }$ is fine-tuned on the two forms of supervision separately to ensure that both newly introduced protocols receive sufficient training. The corresponding objectives are

$$
\mathcal { L } _ { \mathrm { p a i r } } ( \boldsymbol { \Phi } ) = - \frac { 1 } { | C | } \sum _ { ( { q } , a _ { j } , a _ { k } , y _ { j , k } ) \in C } \log M _ { \emptyset } ( y _ { j , k } | \mathsf { P } _ { c } , q , a _ { j } , a _ { k } ) ,\tag{20}
$$

$$
\mathcal { L } _ { \mathrm { l i s t } } ( \boldsymbol { \Phi } ) = - \frac { 1 } { | \mathcal { R } | } \sum _ { ( { q } , A _ { I } , { y } _ { I } ) \in \mathcal { R } } \log M _ { \boldsymbol { \Phi } } ( { y } _ { I } \mid \mathsf { P } _ { r } , { q } , A _ { I } ) .\tag{21}
$$

$\mathsf { P } _ { c }$ and $\mathsf { P } _ { r }$ denote the pairwise-comparison and listwiseranking prompt, respectively.

Multi-Protocol Consolidation. Sequential adaptation may induce catastrophic forgetting [31], where subsequent adaptation on pairwise or listwise supervision can overwrite the pointwise evaluation behavior acquired in Stage I. To avoid this, we introduce a multi-protocol consolidation strategy. Specifically, we construct a unified training set ${ \mathcal { T } } = S \cup C \cup { \mathcal { R } }$ which is used to jointly fine-tune the surrogate. The consoli-

dation objective is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { c o n } } ( \boldsymbol { \Phi } ) = \alpha _ { \mathrm { p o i n t } } \mathcal { L } _ { \mathrm { p o i n t } } ( \boldsymbol { \Phi } ) + \alpha _ { \mathrm { p a i r } } \mathcal { L } _ { \mathrm { p a i r } } ( \boldsymbol { \Phi } ) + \alpha _ { \mathrm { l i s t } } \mathcal { L } _ { \mathrm { l i s t } } ( \boldsymbol { \Phi } ) . } \end{array}\tag{22}
$$

where α<sub>∗</sub> are relative sizes of the corresponding datasets, i.e., $\alpha _ { \mathrm { p o i n t } } = | \boldsymbol { S } | / | \mathcal { T } | , \alpha _ { \mathrm { p a i r } } = | \boldsymbol { C } | / | \mathcal { T } |$ , and $\alpha _ { \mathrm { l i s t } } = | \mathcal { R } | / | \mathcal { T } |$

## 5 Experiments

## 5.1 Experiment Setup

Datasets. We use two instruction-following datasets, i.e., Alpaca and GPT4All, to construct the candidate instances. Alpaca contains 52K instruction-response pairs generated using the self-instruct framework and covers a broad range of general-purpose tasks [40]. GPT4All contains 437K curated prompts collected from publicly available datasets with corresponding responses generated by GPT-3.5-Turbo [1]. We extract the instructions and prompts from these datasets as judging queries and sample outputs from 25 LLMs as candidate responses. The model list is shown in the Appendix.

Victim Model. We consider both proprietary and open-source models as victims in the LLM-as-a-judge setting. The proprietary victims include GPT-5.4<sup>3</sup> and Claude Sonnet 4.5<sup>4</sup>, while the open-source victims include Qwen3-235B-A22B [47] and DeepSeek V4 Pro [46]. For pointwise evaluation, these victims are prompted to provide an integer score from 1 to 10. For pairwise evaluation, they select the preferred response from two candidates, while for listwise evaluation, they rank a set of candidate responses. Additionally, we include UniRRM<sup>5</sup>, a dedicated reward model, as the victim. UniRRM produces a continuous score from 1 to 5 for pointwise evaluation, selects the preferred response for pairwise evaluation, and identifies the best response from a candidate set for listwise evaluation.

Surrogate Models. We use Llama-3.2-1B-Instruct [12] and Qwen3-1.7B [47] as the surrogate model in the main experiments. To study the effect of surrogate capacity and adaptation strategy, we further evaluate Qwen3 models at multiple scales: Qwen3-0.6B, 1.7B, 4B, 8B, 14B and 32B. For each scale, we consider both full fine-tuning and LoRA adaptation [14].

Baselines. We compare JUDGESTEALER with three extraction baselines, i.e., Vanilla, LoRD, and Lion. The Vanilla baseline randomly samples instances across all protocols, queries the victim for supervision, and trains the surrogate model on the victim-labeled instances. LoRD improves model extraction by using victim-model responses as implicit rewards for reinforcement-based surrogate training [28]. Lion employs adversarial knowledge distillation to iteratively identify and generate challenging instructions for improving surrogate model imitation [16]. Proxy-KD introduces an intermediate white-box proxy model aligned with the black-box victim and distills its soft output distributions into the surrogate [5].

Additional details of response generation models, victim models, evaluation metrics, protocol-specific prompts, and implementation settings are provided in the Appendix.

## 5.2 Experimental Results

## 5.2.1 LLM-as-a-Judge

We first evaluate JUDGESTEALER under the LLM-as-a-Judge setting, where the victim LLM is prompted to produce discrete evaluation outcomes in textual form.

Main Results. The results are reported in Table 2. Overall, JUDGESTEALER outperforms the baselines in 475 out of 480 metric-level comparisons, exhibiting consistently strong extraction performance across victims, surrogates, and datasets.

The advantage of JUDGESTEALER holds for both proprietary and open-source victim judges. For GPT-5, using Qwen3-1.7B as the surrogate on Alpaca, JUDGESTEALER outperform the strongest baseline from 0.4865 to 0.5905 at pointwise $A c c _ { \pm 1 }$ , from 0.6240 to 0.7707 pairwise Acc, and from 0.6300 to 0.6345 at listwise Acc. Similar improvements are observed on GPT4All, where pointwise MAE drops from 2.5989 to 1.6111, while pairwise Acc increases from 0.4838 to 0.7833. JUDGESTEALER also shows strong performance against Claude Sonnet 4.5. With Llama-3.2-1B-Instruct on GPT4All, pairwise accuracy increases 0.0992 (0.7118 vs 0.8110) while listwise MAE reduces 0.0511 (0.4729 vs 0.4218). This trend extends to open-source victims. Against Qwen3-235B-A22B, the extracted surrogate achieves 0.8578 pairwise Acc and 0.7100 listwise Acc on GPT4All with Qwen3-1.7B. Particularly large gains are observed for DeepSeek-V4-Pro: on GTP4All, pointwise Acc achieves a relative improvement of 93% (0.2800 vs 0.5400), and MAE 43% (2.3267 to 1.3267). These results demonstrate that JUDGESTEALER generalizes across victim judges with different model families, scales, and accessibility.

On the evaluation protocol dimensions, JUDGESTEALER demonstrates more balanced performance compared with the baselines. Under pointwise evaluation, JUDGESTEALER consistently achieves higher Acc, with an average improvement of 0.094 across all settings. Despite relying only on synthesized supervision for the pairwise and listwise protocols, JUDGESTEALER also outperforms the strongest baselines, with only a few marginal exceptions. Averaged across all settings, pairwise and listwise Acc improve by 0.064 and 0.024, respectively. These results indicate that JUDGESTEALER avoids over-optimizing for a single protocol, as observed in baselines such as Proxy-KD. Overall, JUDGESTEALER effectively transfers supervision among protocols, and collectively enhances their extraction performance.

Results across Surrogate Model Scale. Since the main experiments in Table 2 employ 1-2B surrogate models, we further discuss the effectiveness of JUDGESTEALER on larger surrogates. Specifically, we evaluate the Qwen3 family ranging from 0.6B to 32B parameters, as reported in Figure 3 and Table 8 (Appendix). Overall, increasing the surrogate model size generally enhances extraction performance on both Alpaca and GPT4All, with particularly pronounced improvement under pointwise and listwise evaluation. Pairwise also generally improves with model scale, although the gains become smaller once the surrogate reaches 4B parameters. This may be because pairwise evaluation provides relatively coarse supervision in the form of preferences between responses, making its performance easier to saturate at smaller model scales and leaving less room for improvement as model capacity increases. These results demonstrate that JUDGESTEALER effectively benefits from the capacity of larger surrogate models and scales well across a broad range of model sizes.

Table 2: Experimental results across different datasets, surrogate and victim models.
<table><tr><td rowspan="2">Victim Model</td><td rowspan="2">Dataset</td><td rowspan="2">Method</td><td colspan="6">Qwen3-1.7B</td><td colspan="6">Llama-3.2-1B</td></tr><tr><td colspan="3">Pointwise</td><td colspan="2">Pairwise Listwise</td><td colspan="2"></td><td colspan="2">Pointwise</td><td colspan="2">Pairwise Listwise</td></tr><tr><td>Alpaca</td><td>Vanilla Lion</td><td>Acc ↑ 0.2500 0.1110</td><td>Acc±1 ↑ 0.4865 0.3190</td><td>MAE↓ 2.4430 2.5889</td><td>Acc ↑ 0.5633 0.3740</td><td>Acc ↑ 0.6300 0.3590</td><td>MAE↓ 0.3052 0.4792</td><td>Acc ↑ 0.1895 0.2045</td><td>Acc±1 ↑ 0.4180 0.4595</td><td>MAE↓ 2.7800 3.1732</td><td>Acc ↑ 0.6482 0.3635</td><td>Acc ↑ 0.3855 0.0945</td><td>MAE↓ 0.5770 0.6941</td></tr><tr><td rowspan="2">GPT-5.4</td><td></td><td>LoRD Proxy-KD Ours Vanilla</td><td>0.1470 0.3005 0.3830 0.2078 0.0697</td><td>0.1700 0.4512 0.5905 0.3844</td><td>27 1.7755 2.5989</td><td>0.4680 0.6240 0.7707 0.4233</td><td>0.1530 0.3165 0.6345 0.6280</td><td>0.8775 0.5340 0.3000 0.3027</td><td>0.2005 0.2218 0.3080 0.2322</td><td>0.4590 0.2985 0.4825 0.4533</td><td>2.9910 2.5887 2.3850 2.4089</td><td>0.4340 0.4190 0.7362 0.7000</td><td>0.1430 0.1670 0.3875 0.3457</td><td>0.9133 0.7256 0.5510 0.5800</td></tr><tr><td>GPT4All</td><td>Lion LoRD Proxy-KD Ours Vanilla</td><td>0.0240 0.1512 0.3611 0.2530</td><td>0.2885 0.1540 0.3678 0.6100 0.5570</td><td>2.8582 5.3845 2.8675 1.6111 1.9260</td><td>0.4813 0.4685 0.4838 0.7833 0.7885</td><td>0.2075 0.2300 0.1320 0.6350 0.4960</td><td>0.5768 0.7582 0.6220 0.2950 0.4815</td><td>0.1538 0.1770 0.1535 0.3189 0.2440</td><td>0.3997 0.2000 0.4153 0.5700 0.4720</td><td>2.7769 2.8790 2.7212 1.9189 2.5030</td><td>0.3073 0.4930 0.5620 0.7579 0.7535</td><td>0.0610 0.1365 0.1605 0.4470 0.4850</td><td>0.8844 0.8957 0.8745 0.4633 0.4542</td></tr><tr><td rowspan="2">Claude Sonnet 4.5</td><td>Alpaca</td><td>Lion LoRD Proxy-KD Ours Vanilla</td><td>0.0735 0.1815 0.3040 0.3830 0.3756</td><td>0.4220 0.3980 0.5062 0.6000</td><td>2.3792 2.2255 1.7951 1.4770</td><td>0.3020 0.5165 0.6358 0.7623</td><td>0.2020 0.1295 0.3470 0.5360</td><td>0.5683 0.9123 0.4739 0.4615</td><td>0.2025 0.1295 0.2135 0.3130</td><td>0.3475 0.3265 0.3777 0.5280</td><td>2.2354 2.8115 2.3674 1.9110</td><td>0.5020 0.0190 0.4040 0.7405</td><td>0.1830 0.0690 0.1895 0.4855</td><td>0.7581 0.7960 0.7393 0.4537</td></tr><tr><td>GPT4All</td><td>Lion LoRD Proxy-KD Ours Vanilla</td><td>0.1345 0.1427 0.2407 0.4500 0.3400</td><td>0.5767 0.3865 0.2532 0.4705 0.6700 0.6200</td><td>1.6889 2.3028 4.1747 1.9183 1.2211</td><td>0.7720 0.3960 0.5728 0.4805 0.8128</td><td>0.6473 0.2945 0.1190 0.1500 0.6027</td><td>0.2767 0.6097 0.9107 0.5963 0.2971</td><td>0.3389 0.1800 0.1527 0.1893 0.3967</td><td>0.5444 0.2445 0.3990 0.4147 0.6067</td><td>1.9011 2.3857 2.8043 2.3211 1.4656</td><td>0.7118 0.4785 0.4958 0.5488 0.8110</td><td>0.4403 0.1540 0.1090 0.1650 0.4663</td><td>0.4729 0.8894 0.8230 0.8632 0.4218</td></tr><tr><td rowspan="2">Qwen3- 235B-22B</td><td>Alpaca</td><td>Lion LoRD Proxy-KD Ours</td><td>0.0425 0.2311 0.2957 0.3933</td><td>0.3860 0.5411 0.5361 0.6733</td><td>1.3133 2.0510 2.2600 1.8516 1.2733</td><td>0.8267 0.4185 0.5000 0.6674 0.8378</td><td>0.5933 0.1730 0.1833 0.3543 0.6233</td><td>0.3156 0.4774 0.8400 0.4352 0.2500</td><td>0.2133 0.1695 0.2133 0.2760 0.3533</td><td>0.5533 0.3385 0.5033 0.4508 0.5600</td><td>1.8667 2.0808 2.2522 1.8244 1.5800</td><td>0.7667 0.5035 0.5017 0.5888 0.8078</td><td>0.4567 0.1509 0.1033 0.1543 0.4933</td><td>0.4244 0.7483 0.8878 0.8147 0.4211</td></tr><tr><td>GPT4All</td><td>Vanilla Lion LoRD Proxy-KD Ours</td><td>0.3000 0.0820 0.2133 0.2082 0.3533</td><td>0.5733 0.3595 0.4733 0.4890 0.6400</td><td>1.7867 2.0085 2.6533 1.8928 1.5133</td><td>0.7944 0.4580 0.4717 0.4902 0.8578</td><td>0.7000 0.1220 0.1833 0.1636 0.7100</td><td>0.2233 0.5132 0.8211 0.5108 0.2222</td><td>0.1867 0.1270 0.1878 0.2034 0.2800</td><td>0.4267 0.2430 0.4267 0.4041 0.5467</td><td>2.5800 2.7759 3.0156 2.3590 1.6467</td><td>0.7589 0.5030 0.5017 0.5316 0.8422</td><td>0.5633 0.0745 0.1367 0.1643 0.5933</td><td>0.3400 0.7967 0.9156 0.8600 0.3089</td></tr><tr><td rowspan="2">DeepSeek V4 Pro</td><td>Alpaca</td><td>Vanilla Lion LoRD Proxy-KD Ours</td><td>0.3467 0.1080 0.1744 0.4261 0.5000</td><td>0.5867 0.4335 0.2778 0.5524 0.5933</td><td>2.0200 2.6245 4.4067 2.2419 1.8533</td><td>0.8256 0.3990 0.5617 0.6806 0.8500</td><td>0.5767 0.1560 0.0733 0.3150 0.6233</td><td>0.2856 0.4842 0.8867 0.4571 0.2811</td><td>0.2467 0.2120 0.1011 0.3185 0.4533</td><td>0.3533 0.3370 0.4789 0.4455 0.5867</td><td>3.2400 2.6687 3.4322 2.7900 2.0200</td><td>0.8267 0.5435 0.5617 0.6038 0.8356</td><td>0.4533 0.1505 0.1667 0.1570 0.5000</td><td>0.4322 0.8049 0.8967 0.8166 0.3811</td></tr><tr><td>GPT4All</td><td>Vanilla Lion LoRD Proxy-KD Ours</td><td>0.2800 0.1025 0.2178 0.3133 0.5400</td><td>0.5600 0.4180 0.4533 0.4665 0.7333</td><td>2.3267 2.5093 3.0778 2.2292 1.3267</td><td>0.8289 0.4680 0.5017 0.4435 0.8700</td><td>0.7000 0.1035 0.2067 0.1636 0.7167</td><td>0.2111 0.5505 0.7833 0.5284 0.2067</td><td>0.3333 0.1025 0.1533 0.2153 0.4533</td><td>0.5133 0.2565 0.4411 0.4003 0.6333</td><td>2.6000 2.6354 3.5467 2.5459 1.7467</td><td>0.7922 0.5075 0.5017 0.5480 0.8567</td><td>0.5467 0.0700 0.1900 0.1664 0.5700</td><td>0.3222 0.8374 0.8622 0.8618 0.3422</td></tr></table>

Results across Training Strategies. Since our main experiments adopt LoRA for surrogate adaptation, we further evaluate JUDGESTEALER under full fine-tuning, with the results reported in Figure 3 and Table 8 (Appendix). Overall, both adaptation strategies achieve comparable extraction performance across model scales and evaluation protocols, demonstrating that JUDGESTEALER generalizes well across different finetuning methods. Given its substantially lower computational cost while maintaining competitive extraction performance, we adopt LoRA in the remaining experiments.

Results across Reasoning Settings. We further evaluate JUDGESTEALER under the Chain-of-thought (CoT) setting, where the victim provides a reasoning trace before the final judgment. Specifically, the victim’s pointwise outputs, including both reasoning and scores, are used as pointwise supervision. For the other protocols, we use Qwen3-8B to generate corresponding reasoning traces based on the original input and transformed labels, which together form the supervision for pairwise and listwise protocols. We conduct experiments with GPT-5.4 as the victim and report the results in Table 3. JUDGESTEALER consistently outperforms Vanilla across all protocols. For example, with Qwen3-1.7B on Alpaca, JUDGESTEALER improves pointwise Acc<sub>±1</sub> by 0.0604, pairwise Acc by 0.0250, and listwise Acc by 0.0894 over Vanilla. As a result, JUDGESTEALER remains effective under CoT-based judgment settings.

Table 3: Evaluation results on Chain-of-thought (CoT) reasoning settings.
<table><tr><td rowspan="3">Dataset</td><td rowspan="3">Method</td><td colspan="5">Llama-3.2-1B</td><td rowspan="2">Qwen3-1.7B</td><td colspan="6"></td></tr><tr><td rowspan="2"></td><td colspan="2">Pointwise</td><td rowspan="2">Pairwise</td><td colspan="2">Listwise</td><td colspan="3">Pointwise</td><td colspan="2">Pairwise Listwise</td></tr><tr><td> $A c c \uparrow$   $A c c _ { \pm 1 } \uparrow$ </td><td>MAE↓</td><td>Acc ↑</td><td>Acc ↑</td><td>MAE↓</td><td> $A c c \uparrow$   $A c c _ { \pm 1 } \uparrow$ </td><td>MAE↓</td><td>Acc ↑</td><td>Acc ↑</td><td>MAE↓</td></tr><tr><td rowspan="2">Alpaca</td><td>Vanilla</td><td>0.2748</td><td>0.5461</td><td>2.0435</td><td>0.8291</td><td>0.6439</td><td>0.3089</td><td>0.3283</td><td>0.5839</td><td>1.6474</td><td>0.8520</td><td>0.6200</td><td>0.3422</td></tr><tr><td>Ours</td><td>0.3326</td><td>0.5670</td><td>1.6737</td><td>0.8620</td><td>0.6506</td><td>0.3033</td><td>0.3698</td><td>0.6443</td><td>1.5139</td><td>0.8770</td><td>0.7094</td><td>0.2422</td></tr><tr><td rowspan="2">GPT4All</td><td>Vanilla</td><td>0.2657</td><td>0.5409</td><td>2.0724</td><td>0.8326</td><td>0.6500</td><td>0.3033</td><td>0.2735</td><td>0.5120</td><td>2.3428</td><td>0.8754</td><td>0.7344</td><td>0.2176</td></tr><tr><td>Ours</td><td>0.3930</td><td>0.6669</td><td>1.4035</td><td>0.8796</td><td>0.7533</td><td>0.2056</td><td>0.3852</td><td>0.6669</td><td>1.3880</td><td>0.8857</td><td>0.7478</td><td>0.2057</td></tr></table>

![](images/18be6d7c0f33aa425412d0160c590e108333308a8a683ac3f4d896c3d7fe577a.jpg)  
Figure 3: Model scale and training strategy evaluation.

## 5.2.2 Reward Model

We further evaluate JUDGESTEALER on UniRRM, a dedicated reward model, to examine whether its extraction capability generalizes beyond LLM-as-a-judge. Table 4 demonstrates the results on Alpaca, where JUDGESTEALER consistently achieves the strongest extraction performance across pointwise, pairwise, and listwise protocols. With Qwen3-1.7B as the surrogate, JUDGESTEALER achieves a pointwise MAE of 0.6354, pairwise Acc of 0.8356, and listwise Acc@Top of 0.7967, outperforming the strongest baselines by 1.9%, 7.5%, and 5.2%, respectively. The advantage also extends to Llama-3.2-1B-Instruct, where the corresponding results reach 0.8609, 0.7655, and 0.7000. These results suggest that the underlying evaluation criteria exploited by our cross-protocol extraction also extend to dedicated reward models.

## 5.3 Ablation Study

We conduct an ablation study to evaluate the contribution of each component in JUDGESTEALER. The experiments use

Table 4: Evaluation results of reward models.
<table><tr><td>Surrogate Model</td><td>Method</td><td>MAE↓</td><td>Acc ↑</td><td>Acc@Top↑</td></tr><tr><td rowspan="5">Qwen3-1.7B</td><td>Vanilla</td><td>1.0567</td><td>0.7667</td><td>0.7400</td></tr><tr><td>Lion</td><td>0.6478</td><td>0.7770</td><td>0.7575</td></tr><tr><td>LoRD</td><td>0.7364</td><td>0.6180</td><td>0.5203</td></tr><tr><td>Proxy-KD</td><td>0.7506</td><td>0.5860</td><td>0.6495</td></tr><tr><td>Ours</td><td>0.6354</td><td>0.8356</td><td>0.7967</td></tr><tr><td rowspan="5">Llama-3.2-1B</td><td>Vanilla</td><td>1.0566</td><td>0.6622</td><td>0.6000</td></tr><tr><td>Lion</td><td>0.8959</td><td>0.6870</td><td>0.5055</td></tr><tr><td>LoRD</td><td>0.9426</td><td>0.6340</td><td>0.6923</td></tr><tr><td>Proxy-KD</td><td>0.8820</td><td>0.6175</td><td>0.4795</td></tr><tr><td>Ours</td><td>0.8609</td><td>0.7655</td><td>0.7000</td></tr></table>

Qwen3-1.7B as the surrogate and GPT-5.4 as the victim, with the results reported in Table 5 and Figure 4.

Impact of the Sample Selection Mechanism. We first examine the contribution of the three signals used for sample selection, i.e., semantic diversity D, predictive uncertainty U, and potential judge biases B. The ablation study includes all possible combinations of these signals. The full setting outperforms the other variants across datasets, evaluation protocols, and metrics (Table 5, Lines 1–7). Specifically, the three signals exhibit complementary effects, as removing any of them generally leads to performance degradation. For example, under the bias-only setting, listwise Acc decreases by 0.025 and 0.021 on Alpaca and GPT4All, respectively, compared with the full setting. Although some variants yield marginally higher pairwise and listwise performance on Alpaca (Line 4), such improvements do not generalize to GPT4All. Moreover, disabling all three signals results in more substantial degradation. Under the no-selection setting on GPT4All (Line 7), pairwise Acc drops from 0.783 to 0.740, while pointwise decreases $A c c _ { \pm 1 }$ from 0.610 to 0.582. This decline is likely due to the less informative and even redundant instances potentially introduced by random sampling, which reduces the utility of the victim supervision. Accordingly, the instance selection mechanism with D, U, and B components enables JUDGESTEALER to effectively deliver reliable performance.

Impact of the Adaptive Smoothing Mechanism. We further investigate the effect of the adaptive smoothing mechanism by comparing it with standard one-hot supervision, i.e., α = 0, and fixed smoothing strength ranging from 0.01 to 0.2. As shown in Table 5 (Lines 8–12), adaptive smoothing achieves stronger overall performance than the other settings. Specifically, removing smoothing entirely (Line 8) causes substantial degradation, including reductions in listwise Acc from 0.634 to 0.593 on Alpaca and from 0.635 to 0.602 on GPT4All. Meanwhile, fixed smoothing strengths partially alleviate this degradation, but the optimal strength varies considerably across metrics and datasets. For example, $( \mathbf { \alpha } \mathbf { \alpha } ) = 0 . 0 5 )$ produces the highest pairwise Acc 0.774 on Alpaca, while reduces pointwise $A c c _ { \pm } 1$ on GPT4All by 0.047, from 0.610 to 0.563 (Line 10). However, identifying an optimal value would require costly hyperparameter search. In contrast, the adaptive smoothing adopted by JUDGESTEALER avoids such tuning through dynamically adjusting the strength and provides more stable extraction performance.

Table 5: Ablation studies. JUDGESTEALER employs full setting with D✓ · U✓ · B✓, Adaptive Smoothing and Consolidation.
<table><tr><td rowspan="3" colspan="2">No. Setting</td><td colspan="6">Alpaca</td><td colspan="6">GPT4All</td></tr><tr><td colspan="3">Pointwise</td><td colspan="3">Pairwise Listwise</td><td colspan="3">Pointwise</td><td colspan="3">Pairwise Listwise</td></tr><tr><td>Acc ↑</td><td>Acc±1 ↑</td><td>MAE↓</td><td>Acc ↑</td><td>Acc ↑</td><td>MAE↓</td><td>Acc ↑</td><td>Acc±1 ↑</td><td>MAE↓</td><td>Acc ↑</td><td>Acc ↑ MAE↓</td></tr><tr><td>Full Setting (Ours)</td><td colspan="2">0.383</td><td>0.591</td><td>1.776</td><td>0.771</td><td>0.634</td><td>0.300</td><td>0.361</td><td>0.610</td><td>1.611</td><td>0.783</td><td>0.635</td><td>0.295</td></tr><tr><td colspan="12">Sample Selection Mechanism</td></tr><tr><td colspan="12">0.371(−.012) 0.587(−.004) 1.827(+.052) 0.766(−.005) 0.612(−.022) 0.309(+.009) 0.346(−.015) 0.586(−.024) 1.622(+.011) 0.783(+.000) 0.623(−.012) 0.303(+.008)</td></tr><tr><td></td><td> $\mathcal { \dot { D } } \mathcal { A } \cdot U \mathcal { A } \cdot B \times$   $D \check { \sqrt { \mathbf { \alpha } } } \cdot U \times \mathbf { \beta } \cdot B \check { \mathbf { \alpha } }$ </td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>23</td><td> $D \times \cdot U \sqrt { \phantom { - } \cdot B \sqrt { \phantom { - } } }$ </td><td></td><td></td><td></td><td></td><td>0.371(−.012) 0.575(−.015) 1.858(+.083) 0.764(−.006) 0.610(−.025) 0.313(+.013) 0.351(−.010) 0.581(−.029) 1.733(+.122) 0.783(+.000) 0.618(−.017) 0.303(+.008)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>0.368(−.015) 0.566(−.025) 1.861(+.085) 0.773(+.002) 0.636(+.002) 0.289(−.011) 0.356(−.005) 0.579(−.031) 1.746(+.135) 0.781(−.002) 0.628(−.007) 0.296(+.001)</td><td></td><td></td><td></td><td>0.371(−.012) 0.567(−.024) 1.824(+.049) 0.769(−.001) 0.626(−.008) 0.306(+.006) 0.350(−.011) 0.588(−.022) 1.612(+.001) 0.773(−.011) 0.608(−.027) 0.311(+.016)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>456  $D \times \cdot U \sqrt { \phantom { | } } \cdot B \times$ </td><td> $D \check { \sqrt { \mathrm { \mathbf { \theta } } \cdot U } } \times \cdot B \times$ </td><td>0.367(−.016) 0.582(−.009) 1.841(+.065) 0.772(+.003) 0.625(−.010) 0.303(+.004) 0.343(−.018) 0.567(−.043) 1.798(+.187) 0.780(−.003) 0.615(−.020) 0.303(+.008)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td> $D \times \cdot U \times \cdot B \sqrt { }$ </td><td></td><td></td><td></td><td>0.380(−.004) 0.584(−.007) 1.812(+.037) 0.774(+.004) 0.610(−.025) 0.317(+.017) 0.354(−.007) 0.568(−.042) 1.699(+.088) 0.782(−.001) 0.614(−.021) 0.302(+.007)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>7</td><td> $D \times \cdot U \times \cdot B \times$ </td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="11">0.374(−.009) 0.572(−.019) 1.835(+.059) 0.758(−.013) 0.630(−.004) 0.315(+.015) 0.349(−.012) 0.582(−.028) 1.686(+.074) 0.740(−.043) 0.634(−.001) 0.308(+.013)</td></tr><tr><td>Adaptive Smoothing Mechanism</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="11">8 Fixed α = 0.00  $0 . 3 6 5 \mathrm { ( s - a n s ) } 0 . 5 8 0 \mathrm { ( - a n ) } 1 . 8 7 1 \mathrm { ( s - a p s ) } 0 . 7 6 1 \mathrm { ( s - a p s ) } 0 . 5 9 3 \mathrm { ( s - a n s ) } 0 . 3 2 2 \mathrm { ( s + a n ) } 0 . 3 5 1 \mathrm { ( - a n ) } 0 . 5 5 9 \mathrm { ( - a n s ) } 1 . 7 5 3 \mathrm { ( s - a n ) } 0 . 7 7 6 \mathrm { ( s - a n ) } 0 . 6 6 0 2 \mathrm { ( - a n s ) } 0 . 3 1 4 \mathrm { ( s + a n s ) }$   $0 , 3 7 9 _ { \mathrm { c - a s s } } ~ 0 . 5 8 3 _ { \mathrm { c - a s s } } ~ 1 . 7 9 5 _ { \mathrm { c + } } { \mathrm { n 9 9 } } ~ 0 . 7 7 1 _ { \mathrm { c + } } { \mathrm { a s s } } ~ 0 . 6 0 0 _ { \mathrm { c - } } { \mathrm { a s s } } ~ 0 . 3 2 1 _ { \mathrm { c + } } { \mathrm { a s s } } ~ 0 . 3 5 6 _ { \mathrm { c - } } { \mathrm { a s s } } ~ 0 . 5 6 3 _ { \mathrm { c - } } { \mathrm { a s s } } ~ 1 . 6 9 7 _ { \mathrm { c + } } { \mathrm { a s s } } ~ 0 . 7 8 1 _ { \mathrm { c - } } { \mathrm { n 9 9 } } ~ 0 . 6 2 5 _ { \mathrm { c - } } { \mathrm { a s s } } ~ 0 . 2 9 6 _ { \mathrm { c + } } { \mathrm { a s s } } ,$ </td></tr><tr><td>9  $\mathrm { F i x e d } \propto = 0 . 0 1$ </td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>10  $\mathrm { F i x e d } \propto = 0 . 0 5$ </td><td>0.368(−.015) 0.583(−.008) 1.855(+.079) 0.774(+.004) 0.619(−.016) 0.305(+.005) 0.347(−.014) 0.563(−.047) 1.758(+.147) 0.776(−.007) 0.616(−.019) 0.301(+.006)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>11  $\mathrm { F i x e d } \propto = 0 . 1 0$  12  $\mathrm { F i x e d } \propto = 0 . 2 0$ </td><td colspan="10">0.369(−.013) 0.583(−.007) 1.853(+.077) 0.763(−.007) 0.600(−.035) 0.318(+.018) 0.352(−.009) 0.576(−.034) 1.733(+.122) 0.778(−.005) 0.620(−.015) 0.307(+.012)</td><td></td><td></td><td></td><td>0.364(−.018) 0.582(−.009) 1.881(+.105) 0.765(−.005) 0.605(−.029) 0.315(+.015) 0.360(−.001) 0.609(−.001) 1.679(+.068) 0.778(−.005) 0.591(−.044) 0.326(+.031)</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="10">Consolidation Mechanism</td><td colspan="3"></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="15">13 w/0 Consolidation 0.273(−.111) 0.496(−.094) 2.414(+.638) 0.761(−.010) 0.596(−.039) 0.328(+.028) 0.240(−.121) 0.504(−.106) 2.449(+.838) 0.772(−.011) 0.581(−.054) 0.337(+.042)</td></tr></table>

Impact of the Consolidation Mechanism. We examine the contribution of the multi-protocol consolidation mechanism by removing it after the sequential adaptation. As shown in Table 5 (Line 13), removing consolidation significantly lowers the judging performance, particularly under pointwise evaluation. On Alpaca, pointwise Acc decreases by 28.7%, while MAE increases by 35.9%. The degradation is even greater on GPT4All, where pointwise Acc drops by 33.5% and MAE increases by 52.0%. Under the pairwise evaluation, the performance also deteriorates, though the effect is relatively modest. Interestingly, despite being the final protocol during sequential training, listwise evaluation also benefits from consolidation. This may arise for two reasons. First, consolidation exposes the surrogate to additional training on the listwise data, further reinforcing the listwise judging behavior. Second, joint optimization over all three protocols may strengthen their shared underlying judging criteria and enhance cross-protocol generalization.

Impact of Query Budget. We finally investigate the impact of the query budget on extraction performance. The budget is defined as the percentage of candidate instances selected from the entire candidate pool for querying the victim, and we vary it from 0.5% to 10%. As shown in Figure 4, JUDGESTEALER already achieves strong performance under an extremely limited budget of 0.5%, with pointwise $A c c _ { \pm 1 }$ reaching approximately 50% and pairwise Acc exceeding 70%. Increasing the query budget further improves the performance on both Alpaca and GPT4All across all three evaluation protocols. Compared with pointwise results, pairwise and listwise performance benefits more from larger budgets. This may be because these protocols rely more on learning relative relation ships among responses, which requires supervision covering a broader range of response combinations. These results demonstrate that JUDGESTEALER scales effectively with available query resources while retaining strong extraction capability under highly restricted budgets.

We further conduct an ablation study for the cross-protocol transformation mechanism in the Appendix.

## 6 Robustness against Defense Methods

## 6.1 Anomaly Detection

We first evaluate JUDGESTEALER against anomaly detection strategies. Following the setting in [17], each incoming input is encoded into an embedding using a sentence encoder<sup>6</sup>, and selectively retained in a set of representative historical inputs. For each new input, the defender computes its Euclidean distance to the stored inputs and records the minimum distance to construct a distance distribution. The Shapiro-Wilk test [38] is applied to assess the normality W of this distribution. A user is flagged as suspicious when the normality statistic W falls below a threshold τ , which is determined by the benign input sequences.

![](images/651aa018c3014b2c80a081ea00b2b8ae362d41594ebbebf5da83e13c3a70aa6c.jpg)

![](images/67fcae6251a044e6ffca00a0fd728fa0d204423bbb945b89b76dde8e0bda1e88.jpg)

![](images/c9db102e411ee37f8811d31fa85c5a5885979ac382366fd2d2b3a14ddc75b263.jpg)  
Alpaca (Ours) GPT4All (Ours)

![](images/a6661fd69dd0bb09a71c727095675f1375ceaea5778a51c9117b40ba8cbc9746.jpg)

![](images/3856fbcb3c964fb95fbf59fdc0e38579ee572b6822aab0aa21980cafe76778fb.jpg)

![](images/e7f992f3597ff61d2ab9b0cd3b552fb8e328705dbf0405cf02ab6bd45d6b5b69.jpg)

Figure 4: Performance on different query budgets.  
Table 6: Evaluation against anti-distillation defense.
<table><tr><td>Model</td><td>Method</td><td colspan="3">Pointwise</td><td>Pairwise</td><td>Listwise</td></tr><tr><td></td><td></td><td>ACC</td><td> $A C C _ { \pm 1 }$ </td><td>MAE</td><td>ACC</td><td>ACC</td></tr><tr><td>Qwen3-1.7B</td><td>w/o defense</td><td>0.499</td><td>0.737</td><td>1.2165</td><td>0.803</td><td>0.538</td></tr><tr><td></td><td>w/ defense</td><td>0.498</td><td>0.730</td><td>1.2465</td><td>0.801</td><td>0.531</td></tr><tr><td>Llama-3.2-1B</td><td>w/o defense</td><td>0.362</td><td>0.692</td><td>1.7205</td><td>0.730</td><td>0.426</td></tr><tr><td></td><td>w/ defense</td><td>0.360</td><td>0.688</td><td>1.7450</td><td>0.723</td><td>0.411</td></tr></table>

The results of Llama-3.2-1B-Instruct and Qwen3-1.7B on Alpaca are reported in Table 10 (Appendix). For both models, the W values obtained from JUDGESTEALER queries are clearly above the threshold $\tau _ { W }$ , which indicates that JUDGESTEALER successfully evades the anomaly detector [38]. This is likely due to the query selection process, which jointly considers the query diversity D(G) and local density $\rho ( G )$ to avoid overly clustered inputs. Consequently, the distance distribution of JUDGESTEALER closely resembles that of benign user inputs, making the attack stealthier to detect.

## 6.2 Anti-Distillation Perturbation

We further evaluate the robustness of JUDGESTEALER against anti-distillation defenses. We implement a defense that modifies the victim model to perturb its output distribution [25]. Specifically, the defense introduces a fixed proxy student model to approximate a potential distillation attacker and adversarially fine-tune only the victim’s LM head while keeping the remaining parameters frozen. The training objective combines a supervised fine-tuning loss that preserves the victim’s task utility with an adversarial loss that maximizes the KL divergence between the output distributions of the victim and proxy student models.

The results in Table 6 show that JUDGESTEALER preserves the normal judging quality of the defended model. For Qwen3- 1.7B, the defended model slightly reduces pointwise, pairwise, and listwise Acc by only 0.001, 0.002, and 0.007. Similarly, a small impact is observed for Llama-3.2-1B-Instruct. These results indicate that the anti-distillation defense has limited influence on JUDGESTEALER.

## 6.3 Ownership Tracing

The ownership tracing defense provides post-hoc evidence of model extraction by embedding an invisible watermark into the victim’s outputs. We implement a statistical watermark based on a context-dependent vocabulary partitioning, which divides candidate tokens into a green list and non-green list and biases generation toward green tokens [20]. A surrogate distilled from these outputs may inherit this preference. The defender can then query a suspicious model and compute the Z-score to measure the green-token deviation for watermark detection. An average Z-score above 4 indicates the presence of watermarks [20].

We use the Qwen3-32B model as the victim model and inject the watermark into its outputs. As shown in Table 11 (appendix), the victim model exhibits a statistically detectable watermark, with an average Z-score of 6.917. In contrast, the extracted Qwen3-1.7B and Llama-3.2-1B-Instruct surrogates show no significant watermark inheritance, with Z-scores of −0.219 and −0.256, respectively. These values are only slightly higher than those obtained under the corresponding no-defense settings −1.017 and −1.108. Overall, the watermark transfers weakly to the extracted surrogates and does not yield a statistically detectable signal under our pipeline.

## 7 Conclusion

We present JUDGESTEALER, the first model extraction framework for replicating LLM judging capabilities across pointwise scoring, pairwise comparison, and listwise ranking under black-box access. JUDGESTEALER exploits the strong agreement across evaluation protocols by acquiring fine-grained pointwise supervision and transforming it into pairwise and listwise signals without additional victim queries. To improve extraction efficiency and performance, it further incorporates dynamic informative-instance selection, adaptive smoothing, and multi-protocol consolidation. Extensive experiments across proprietary and open-source LLM-as-a-Judge models and reward models demonstrate that JUDGESTEALER consistently outperforms existing extraction baselines across all three evaluation protocols. Its effectiveness is further preserved across different surrogate model scales, adaptation strategies, and reasoning settings, while remaining robust against representative extraction defenses. These findings reveal that LLM judging capabilities can be effectively replicated with limited black-box supervision and highlight the need for stronger protections for increasingly valuable modelbased evaluation services.

## Ethical Considerations

This work studies whether the judging capability of LLMbased evaluators can be extracted through black-box access. This is ethically important because such judges are used in evaluation, alignment, and safety moderation; if they can be copied, providers may lose intellectual property and downstream safeguards may be weakened. At the same time, studying this risk helps the community measure the vulnerability and design more effective defenses.

Stakeholders. The main stakeholders are model/API providers, attackers, and ordinary users. Model/API providers may suffer intellectual-property loss if proprietary judging capabilities are copied. Attackers may misuse extraction methods to build unauthorized surrogate judges or weaken safety and moderation services. Ordinary users may also be affected indirectly. If model/API providers respond to extraction risks with stricter access controls, heavier monitoring, output perturbation, or higher prices, benign users may experience reduced service quality, privacy concerns, or higher usage costs.

Experimental scope, data, and privacy. Our study is limited to black-box functionality extraction using standard victim interfaces. We only observe outputs exposed by the interface and do not attempt to recover parameters, hidden prompts, logits, training data, credentials, service logs, or infrastructure details. For proprietary judges, we report only aggregate measurements and do not bypass authentication, evade rate limits, or attack service infrastructure. The experiments use public instruction-following datasets, including Alpaca and GPT4All, to construct judging instances. We conduct no human-subject study and collect no private user data or personally identifiable information. Any derived artifacts will be checked for accidental identifiers before release.

Misuse mitigation. We will release artifacts sufficient to validate the paper’s claims, but not in a form that serves as a turnkey extraction tool for commercial judging services. The public artifact will support reproduction on open-source judges and reward models. We will exclude commercialservice credentials, raw proprietary query logs, providerspecific large-scale querying scripts, and surrogate checkpoints distilled from proprietary judges.

## Open Science

Artifact access. We will make the artifact repository publicly available upon acceptance.

Code. The repository will include the implementation of JUDGESTEALER, including query selection, pointwise extraction, adaptive smoothing, pairwise/listwise construction, multi-protocol training, and evaluation scripts. It will also include prompts, configuration files, random seeds, defenseevaluation scripts, and scripts for reproducing the main results.

Data. We will provide scripts to download and preprocess Alpaca and GPT4All and to construct the judging instances. When redistribution is allowed, we will include derived opensource labels and splits; otherwise, we will provide reconstruction scripts.

Omissions. We will not release commercial API credentials, proprietary model outputs that cannot be redistributed, or checkpoints imitating proprietary judges. Open-source experiments will be reproducible end-to-end; proprietary-victim experiments depend on service availability and terms.

## References

[1] Yuvanesh Anand, Zach Nussbaum, Adam Treat, Aaron Miller, Richard Guo, Benjamin Schmidt, Brandon Duderstadt, and Andriy Mulyar. Gpt4all: An ecosystem of open source compressed language models. In Proceedings of the 3rd Workshop for Natural Language Processing Open Source Software (NLP-OSS 2023), pages 59–64, 2023.

[2] Lewis Birch, William Hackett, Stefan Trawicki, Neeraj Suri, and Peter Garraghan. Model leeching: An extraction attack targeting llms. arXiv preprint arXiv:2309.10544, 2023.

[3] Nicholas Carlini, Daniel Paleka, Krishnamurthy Dj Dvijotham, Thomas Steinke, Jonathan Hayase, A Feder Cooper, Katherine Lee, Matthew Jagielski, Milad Nasr, Arthur Conmy, et al. Stealing part of a production language model. arXiv preprint arXiv:2403.06634, 2024.

[4] Chen Chen, Xuanli He, Lingjuan Lyu, and Fangzhao Wu. Killing one bird with two stones: Model extraction and attribute inference attacks against bert-based apis. arXiv preprint arXiv:2105.10909, 2021.

[5] Hongzhan Chen, Ruijun Chen, Yuqi Yi, Xiaojun Quan, Chenliang Li, Ming Yan, and Ji Zhang. Knowledge distillation of black-box large language models. arXiv preprint arXiv:2401.07013, 2024.

[6] Yanjiao Chen, Rui Guan, Xueluan Gong, Jianshuo Dong, and Meng Xue. D-dae: Defense-penetrating model extraction attacks. In IEEE Symposium on Security and Privacy, pages 382–399, 2023.

[7] Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. Advances in Neural Information Processing Systems, 30, 2017.

[8] Vasisht Duddu, Debasis Samanta, D Vijay Rao, and Valentina E Balas. Stealing neural networks via timing side channels. arXiv preprint arXiv:1812.11720, 2018.

[9] Xueluan Gong, Yanjiao Chen, Wenbin Yang, Guanghao Mei, and Qian Wang. Inversenet: Augmenting model extraction attacks with training data inversion. In International Joint Conference on Artificial Intelligence, pages 2439–2447, 2021.

[10] Xueluan Gong, Shuaike Li, Yanjiao Chen, Mingzhe Li, Rubin Wei, Qian Wang, and Kwok-Yan Lam. Augmenting model extraction attacks against disruption-based defenses. IEEE Transactions on Information Forensics and Security, 20:531–546, 2024.

[11] Xueluan Gong, Qian Wang, Yanjiao Chen, Wang Yang, and Xinchang Jiang. Model extraction attacks and defenses on cloud-based machine learning models. IEEE Communications Magazine, 58(12):83–89, 2021.

[12] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

[13] Xuanli He, Lingjuan Lyu, Qiongkai Xu, and Lichao Sun. Model extraction and adversarial transferability, your bert is vulnerable! In Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 2006–2012, 2021.

[14] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.

[15] Hakan Inan, Kartikeya Upasani, Jianfeng Chi, Rashi Rungta, Krithika Iyer, Yuning Mao, Michael Tontchev, Qing Hu, Brian Fuller, Davide Testuggine, et al. Llama guard: Llm-based input-output safeguard for human-ai conversations. arXiv preprint arXiv:2312.06674, 2023.

[16] Yuxin Jiang, Chunkit Chan, Mingyang Chen, and Wei Wang. Lion: Adversarial distillation of proprietary large language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 3134–3154, 2023.

[17] Mika Juuti, Sebastian Szyller, Samuel Marchal, and N Asokan. Prada: protecting against dnn model stealing attacks. In IEEE European Symposium on Security and Privacy (EuroS&P), pages 512–527, 2019.

[18] Sanjay Kariyappa, Atul Prakash, and Moinuddin K Qureshi. Maze: Data-free model stealing attack using zeroth-order gradient estimation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13814–13823, 2021.

[19] Adrienne W Kemp. Characterizations of a discrete normal distribution. Journal ofStatistical Planning and Inference, 63(2):223–229, 1997.

[20] John Kirchenbauer, Jonas Geiping, Yuxin Wen, Jonathan Katz, Ian Miers, and Tom Goldstein. A watermark for large language models. In International conference on machine learning, pages 17061–17084. PMLR, 2023.

[21] Kalpesh Krishna, Gaurav Singh Tomar, Ankur P Parikh, Nicolas Papernot, and Mohit Iyyer. Thieves on sesame street! model extraction of bert-based apis. arXiv preprint arXiv:1910.12366, 2019.

[22] H Li, Q Dong, J Chen, H Su, Y Zhou, Q Ai, Z Ye, and Y Liu. Llms-as-judges: a comprehensive survey on llm-based evaluation methods (2024). arXiv preprint arXiv:2412.05579.

[23] Jiawei Li, Yang Gao, Yizhe Yang, Yu Bai, Xiaofeng Zhou, Yinghao Li, Huashan Sun, Yuhang Liu, Xingpeng Si, Yuhao Ye, et al. Fundamental capabilities and applications of large language models: A survey. ACM Computing Surveys, 58(2):1–42, 2025.

[24] Jiawei Li, Yizhe Yang, Yu Bai, Xiaofeng Zhou, Yinghao Li, Huashan Sun, Yuhang Liu, Xingpeng Si, Yuhao Ye, Yixiao Wu, et al. Fundamental capabilities of large language models and their applications in domain scenarios: A survey. In Proceedings of the 62nd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 11116–11141, 2024.

[25] Pingzhi Li, Zhen Tan, Mohan Zhang, Huaizhi Qu, Huan Liu, and Tianlong Chen. Doge: Defensive output generation for llm protection against knowledge distillation. arXiv preprint arXiv:2505.19504, 2025.

[26] Xuechen Li, Tianyi Zhang, Yann Dubois, Rohan Taori, Ishaan Gulrajani, Carlos Guestrin, Percy Liang, and Tatsunori B Hashimoto. Alpacaeval: An automatic evaluator of instruction-following models, 2023.

[27] Zongjie Li, Chaozheng Wang, Pingchuan Ma, Chaowei Liu, Shuai Wang, Daoyuan Wu, Cuiyun Gao, and Yang Liu. On extracting specialized code abilities from large language models: A feasibility study. In IEEE/ACM 46th International Conference on Software Engineering, pages 1–13, 2024.

[28] Zi Liang, Qingqing Ye, Yanyun Wang, Sen Zhang, Yaxin Xiao, Ronghua Li, Jianliang Xu, and Haibo Hu. “yes, my lord.” guiding language model extraction with locality reinforced distillation. In Annual Meeting of the Associationfor Computational Linguistics, pages 1441–1465, 2025.

[29] Shuze Liu, Qianwen Guo, and Yushun Dong. An embarrassingly simple detector for model extraction attacks in large language model api traffic. arXiv preprint arXiv:2606.05725, 2026.

[30] Yupei Liu, Jinyuan Jia, Hongbin Liu, and Neil Zhenqiang Gong. Stolenencoder: Stealing pre-trained encoders in self-supervised learning. In ACM SIGSAC Conference on Computer and Communications Security, pages 2115–2128, 2022.

[31] Michael McCloskey and Neal J Cohen. Catastrophic interference in connectionist networks: The sequential learning problem. In Psychology oflearning and motivation, volume 24, pages 109–165. Elsevier, 1989.

[32] Najmeh Nazari, Furi Xiang, Chongzhou Fang, Hosein Mohammadi Makrani, Aditya Puri, Kartik Patwari, Hossein Sayadi, Setareh Rafatirad, Chen-Nee Chuah, and Houman Homayoun. Llm-fin: Large language models fingerprinting attack on edge devices. In 2024 25th International Symposium on Quality Electronic Design (ISQED), pages 1–6. IEEE, 2024.

[33] Tribhuvanesh Orekondy, Bernt Schiele, and Mario Fritz. Knockoff nets: Stealing functionality of black-box models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4954–4963, 2019.

[34] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730–27744, 2022.

[35] Soham Pal, Yash Gupta, Aditya Shukla, Aditya Kanade, Shirish Shevade, and Vinod Ganapathy. Activethief: Model extraction using active learning and unannotated public data. AAAI, 2020.

[36] Kaiyi Pang, Tao Qi, Chuhan Wu, Minhao Bai, Minghu Jiang, and Yongfeng Huang. Modelshield: Adaptive and

robust watermark against model extraction attack. IEEE Transactions on Information Forensics and Security, 20:1767–1782, 2025.

[37] Ravi Shanker Raju, Swayambhoo Jain, Bo Li, Jonathan Lingjie Li, and Urmish Thakker. Constructing domain-specific evaluation sets for llm-as-a-judge. In Workshop on Customizable NLP: Progress and Challenges in Customizing NLPfor a Domain, Application, Group, or Individual, pages 167–181, 2024.

[38] Samuel Sanford Shapiro and Martin B Wilk. An analysis of variance test for normality (complete samples). Biometrika, 52(3-4):591–611, 1965.

[39] Minxue Tang, Anna Dai, Louis DiValentin, Aolin Ding, Amin Hass, Neil Zhenqiang Gong, Yiran Chen, et al. {ModelGuard}:{Information-Theoretic} defense against model extraction attacks. In USENIX Security Symposium, pages 5305–5322, 2024.

[40] Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Stanford alpaca: An instruction-following llama model. https://github. com/tatsu-lab/stanford\_alpaca, 2023.

[41] Florian Tramèr, Fan Zhang, Ari Juels, Michael K Reiter, and Thomas Ristenpart. Stealing machine learning models via prediction APIs. In USENIX Security Symposium, pages 601–618. USENIX Association, 2016.

[42] Jean-Baptiste Truong, Pratyush Maini, Robert J Walls, and Nicolas Papernot. Data-free model extraction. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4771–4780, 2021.

[43] Binghui Wang and Neil Zhenqiang Gong. Stealing hyperparameters in machine learning. In IEEE Symposium on Security and Privacy, pages 36–52, 2018.

[44] Liaoyaqi Wang and Minhao Cheng. Guardemb: Dynamic watermark for safeguarding large language model embedding service against model stealing attack. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 7518–7534, 2024.

[45] Victor Wang, Michael JQ Zhang, and Eunsol Choi. Improving llm-as-a-judge inference with the judgment distribution, 2025. URL https://arxiv. org/abs/2503.03064.

[46] Anyi Xu, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, Chenchen Ling, et al. Deepseek-v4: Towards highly efficient million-token context intelligence. arXiv preprint arXiv:2606.19348, 2026.

[47] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

[48] Zhanyuan Zhang, Yizheng Chen, and David Wagner. Seat: Similarity encoder by adversarial training for detecting model extraction attack queries. In ACM Workshop on Artificial Intelligence and Security, pages 37–48, 2021.

[49] Kaixiang Zhao, Lincan Li, Kaize Ding, Neil Zhenqiang Gong, Yue Zhao, and Yushun Dong. A survey on model extraction attacks and defenses for large language models. In ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 6227–6236, 2025.

[50] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-asa-judge with mt-bench and chatbot arena. Advances in neural information processing systems, 36:46595– 46623, 2023.

## A Additional Experiment Setup

## A.1 Datasets

Query Construction. We construct the candidate instance pools using queries from Alpaca and GPT4All. For GPT-5.4 as the victim, we sample 18K queries from Alpaca and GPT4All, while for Claude Sonnet 4.5 as the victim, we sample 10K queries from each dataset. For all other victim– dataset combinations, we sample 2K queries.

Candidate Response Construction. To construct multiresponse judging instances, we employ a pool of 25 LLMs from diverse model families as response generators, as listed in Table 7. For each query, we randomly sample three models from this pool and obtain one response from each model, forming an instance $G = \{ ( q , a _ { i } ) \} _ { i = 1 } ^ { 3 }$ . We use the default decoding configuration provided by each model or API. These response generators are introduced to increase diversity in response content and quality while reducing reliance on any single model family.

Victim Judgment Collection Construction. To reduce the communication overhead and latency caused by querying remote victim APIs during experiments, we collect the victim judgments for all constructed instances in advance. During subsequent experiments, these pre-collected results are loaded to simulate victim interactions. This setup aims to ensure that all methods receive identical victim supervision while avoiding repeated API calls.

Table 7: Target LLM agents and the judge model.
<table><tr><td>LLMs</td><td>Versions</td><td>Availability</td><td>Access</td></tr><tr><td colspan="4">Response Generation Models</td></tr><tr><td>GPT</td><td>GPT-4.1, GPT-4o</td><td>Proprietary</td><td>OpenAI API</td></tr><tr><td>Claude</td><td>Sonnet 4.5</td><td>Proprietary</td><td>Anthropic API</td></tr><tr><td>Amazon Nova</td><td>Nova Micro, Lite, Nova 2 Lite</td><td>Proprietary</td><td>AWS Bedrock</td></tr><tr><td>Llama</td><td>Llama 3.1 Instruct 8B, 70B</td><td>Open-source</td><td>AWS Bedrock</td></tr><tr><td>Nvidia</td><td>Nemotron Nano 9B, 12B</td><td>Open-source</td><td>AWS Bedrock</td></tr><tr><td>Gemma</td><td>Gemma 3 4B, 12B, 27B, 3n-E4B</td><td>Open-source</td><td>Together AI</td></tr><tr><td>GPT-OSS</td><td>GPT-OSS-20B</td><td>Open-source</td><td>Together AI</td></tr><tr><td>Qwen</td><td>Qwen 3 0.6B, 1.7B, 4B,</td><td>Open-source</td><td>Together AI</td></tr><tr><td>Mistral</td><td>Qwen 2.5 7B, Qwen 2 1.5B Ministral 3B, 8B, Mistral 7B, Mistral 8x7B, Voxtral Mini 3B</td><td>Open-source</td><td>AWS Bedrock</td></tr><tr><td colspan="4">Victim Models</td></tr><tr><td>GPT</td><td>GPT-5.4</td><td>Proprietary</td><td>OpenAI API</td></tr><tr><td>Claude</td><td>Sonnet 4.5</td><td>Proprietary</td><td>Anthropic API</td></tr><tr><td>Qwen</td><td>Qwen3-235B-A22B</td><td>Open-source</td><td>Together AI</td></tr><tr><td>Deepseek</td><td>Deepseek-V4-Pro</td><td>Open-source</td><td>Together AI</td></tr><tr><td>UniRRM</td><td>UniRRM-8B</td><td>Open-source</td><td>Local Hosting</td></tr></table>

## A.2 Evaluation Metrics

## A.2.1 LLM-as-a-Judge Setting

In the LLM-as-a-Judge setting, we evaluate the performance of the extracted surrogate model using the following metrics. Pointwise Metrics. For pointwise evaluation, we report three metrics, i.e., Acc, Acc<sub>±1</sub>, and MAE. Given the pointwise test set $S _ { \mathrm { t e s t } } = \{ ( q ^ { ( i ) } , a ^ { ( i ) } , y ^ { ( i ) } ) \} _ { i = 1 } ^ { N }$ , Acc measures the proportion of instances for which the surrogate score $\hat { y } ^ { ( i ) }$ exactly reproduces the victim score $\boldsymbol { y } ^ { ( i ) }$

$$
A c c = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbb { I } \left( \hat { y } ^ { ( i ) } = y ^ { ( i ) } \right) ,\tag{23}
$$

where I(·) denotes the indicator function.

Since pointwise scores are ordinal, an exact match may be overly restrictive when the predicted score differs only slightly from the victim score. We therefore additionally report withinone accuracy $\left( A c c _ { \pm 1 } \right)$ , which regards a prediction as correct if its absolute deviation from the victim score is at most one:

$$
A c c _ { \pm 1 } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbb { I } \left( \left| \hat { y } ^ { ( i ) } - y ^ { ( i ) } \right| \leq 1 \right) .\tag{24}
$$

We further report Mean Absolute Error (MAE) to quantify the magnitude of score deviation:

$$
M A E = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left| \hat { y } ^ { ( i ) } - y ^ { ( i ) } \right| .\tag{25}
$$

Higher Acc and $A c c _ { \pm 1 }$ , and lower MAE, indicate stronger reproduction of the victim’s pointwise scoring behavior.

Pairwise Metric. For pairwise evaluation, we use Acc to measure preference agreement between the victim and surrogate models. Given the pairwise test set ${ \cal C } _ { \mathrm { t e s t } } =$ $\{ ( q ^ { ( i ) } , a _ { j } ^ { ( i ) } , a _ { k } ^ { ( i ) } , y _ { j , k } ^ { ( i ) } ) \} _ { i = 1 } ^ { N }$ , the preference agreement is:

$$
A c c = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbb { I } \left( \hat { y } _ { j , k } ^ { ( i ) } = y _ { j , k } ^ { ( i ) } \right) .\tag{26}
$$

Listwise Metrics. For listwise evaluation, we report Acc and R-MAE. Given a listwise test set $\mathcal { R } _ { \mathrm { t e s t } } = \{ ( q ^ { ( i ) } , A _ { I } ^ { ( i ) } , y _ { I } ^ { ( i ) } ) \} _ { i = 1 } ^ { N } ,$ let $r _ { I } ^ { ( i ) } ( a )$ and $\hat { r } _ { I } ^ { ( i ) } ( a )$ denote its ranking positions of a response $a \in A _ { I } ^ { ( i ) }$ assigned by the victim and surrogate models, respectively. Acc measures the exact match between the surrogate and victim rankings over all responses.

$$
A c c = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbb { I } \left( \forall a \in A _ { I } ^ { ( i ) } : \widehat { r } _ { I } ^ { ( i ) } ( a ) = r _ { I } ^ { ( i ) } ( a ) \right) .\tag{27}
$$

R-MAE measures the average absolute deviation in ranking positions on all responses:

$$
R \ – M A E = \frac { 1 } { N | I | } \sum _ { i = 1 } ^ { N } \sum _ { a \in A _ { I } ^ { ( i ) } } \left| \hat { r } _ { I } ^ { ( i ) } ( a ) - r _ { I } ^ { ( i ) } ( a ) \right| .\tag{28}
$$

Higher Acc and lower R-MAE indicate stronger agreement with the victim’s listwise ranking behavior.

## A.2.2 Reward Model Setting

Since reward models produce continuous scores under pointwise evaluation, the matching metrics Acc and $A c c _ { \pm 1 }$ are not applicable. We therefore only report MAE for the pointwise protocol. For pairwise evaluation, we follow the same setting as LLM-as-a-Judge and report Acc based on preference agreement. For listwise evaluation, the reward model selects the best response from the candidates by default. Accordingly, we follow this and report Acc@Top, which measures whether the surrogate identifies the same best response as the victim. Specifically, given a listwise test set $\mathcal { R } _ { \mathrm { t e s t } } = \{ ( q ^ { ( i ) } , A _ { I } ^ { ( i ) } , y _ { I } ^ { ( i ) } ) \} _ { i = 1 } ^ { N }$ , let $a _ { I } ^ { * ( i ) }$ and $\hat { a } _ { I } ^ { * ( i ) }$ denote the best responses selected by the victim and surrogate from $A _ { I } ^ { ( i ) }$ , respectively. We compute

$$
A c c @ T o p = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbb { I } \left( \hat { a } _ { I } ^ { * ( i ) } = a _ { I } ^ { * ( i ) } \right) .\tag{29}
$$

## B Evaluation Prompt Templates

We provide the evaluation prompt templates used in our experiments. The pointwise, pairwise and listwise prompt templates are presented in Figure 5, 6, and 7, respectively.

## C Implementation Details

Unless otherwise specified, we adapt the surrogate models using LoRA with 4-bit model loading. We employ AdamW optimizer with a learning rate of $1 \times 1 0 ^ { - 4 }$ , a cosine learning-rate schedule with 10% warmup, a per-device batch size of 1, gradient accumulation over 16 steps, and a maximum sequence length of 4096. We set the LoRA rank to 8, the scaling factor to 16, and the dropout rate to 0.05, and apply the adapters to the query, key, value, and output projection modules. For the full fine-tuning experiments, we reduce the learning rate to $1 \times 1 0 ^ { - 5 }$ while retaining the same batch configuration. In the sample-selection module, the weights for semantic diversity $\lambda _ { 1 } .$ , predictive uncertainty $\lambda _ { 2 } ,$ and judge bias $\lambda _ { 3 }$ are set to 1.0, 0.25, and 1.0, respectively. We initialize $\scriptstyle { \mathcal { G } } _ { < t }$ using random sampling with 80 instances and subsequently select $K = 2 0$ instances per iteration from a candidate subset of $| \widetilde { \cal G } _ { t } | = 1 0 0$ instances. Semantic embedding is computed using BAAI/bgesmall-en-v1 $. 5 ^ { 7 }$ with CLS pooling and L2-normalized representations. For adaptive score smoothing, we use a discrete Gaussian with $\sigma = 1 . 0$ , initialize the trainable smoothing coefficient at 0.10, and optimize it with a learning rate of $5 \times 1 0 ^ { - 6 }$ For the cross-protocol transformation, we set the sampling parameters to $n _ { c } = n _ { l } = n _ { r } = 3$ . Our experiments are conducted using Python 3.14.6 on a 384-core Intel(R) Xeon(R) 6972P CPU and NVIDIA H200 NVL PCIe GPU machine, running on Ubuntu 22.04.5 LTS.

![](images/87c2cc95b63590302bf717f51472c948882195e4ee59e1d936ab3a593040967a.jpg)  
Figure 5: Prompt template for pointwise evaluation.

## D Additional Experiment Results

Impact of Cross-Protocol Transformation. We further investigate whether allocating the entire query budget to a single evaluation protocol can provide better protocolspecific extraction performance. Specifically, we compare JUDGESTEALER with pointwise-only, pairwise-only, and listwise-only settings, where each setting randomly samples independent query–response instances and spends the full budget of 600 victim queries exclusively on the corresponding protocol. As shown in Table 9, JUDGESTEALER consistently outperforms the pointwise-only setting across all pointwise metrics on both datasets. Although pointwise-only can cover a more diverse set of inputs because it does not require multiple responses for each query, its supervision is restricted to a single protocol. In contrast, learning pairwise and listwise behaviors enables JUDGESTEALER to further reinforce the shared underlying evaluation criterion, which in turn improves pointwise judging. Similar cross-protocol benefits are observed for pairwise and listwise evaluation. On Alpaca, JUDGESTEALER remains competitive with pairwise-only, while on GPT4All it even achieves higher pairwise accuracy (0.783 vs. 0.753), despite never querying the victim under the pairwise protocol. Likewise, its listwise performance closely approaches listwise-only on Alpaca and slightly surpasses it on GPT4All. These results further support the existence of a shared evaluation criterion across protocols and demonstrate that JUDGESTEALER can effectively exploit this structure to jointly improve multiple judging capabilities, in some cases even outperforming protocol-specific training under the same query budget.

Table 8: Results of surrogate models across model scales and adaptation strategies.
<table><tr><td rowspan="3">Surrogate Model</td><td colspan="6">Alpaca</td><td colspan="6">GPT4All</td></tr><tr><td colspan="3">Pointwise</td><td>Pairwise</td><td colspan="2">Listwise</td><td colspan="3">Pointwise</td><td colspan="2">Pairwise Listwise</td></tr><tr><td>Acc ↑</td><td>Acc±1 ↑</td><td>MAE↓</td><td>Acc ↑</td><td>Acc ↑</td><td>MAE↓</td><td>Acc ↑</td><td>Acc±1 ↑</td><td>MAE↓</td><td>Acc ↑</td><td>Acc ↑ MAE↓</td></tr><tr><td>Qwen3-0.6B</td><td>0.3410</td><td>0.5235</td><td>2.2005</td><td>0.8016</td><td>0.5295</td><td>0.3911</td><td>0.2811</td><td>0.5122</td><td>2.3100</td><td>0.7761</td><td>0.5627</td><td>0.3557</td></tr><tr><td>Qwen3-0.6B LoRA</td><td>0.2995</td><td>0.4960</td><td>2.4135</td><td>0.8037</td><td>0.5250</td><td>0.4023</td><td>0.2744</td><td>0.5011</td><td>2.3400</td><td>0.7782</td><td>0.5530</td><td>0.3683</td></tr><tr><td>Qwen3-1.7B</td><td>0.3965</td><td>0.5690</td><td>1.8170</td><td>0.7810</td><td>0.6250</td><td>0.2956</td><td>0.3644</td><td>0.6010</td><td>1.6778</td><td>0.7717</td><td>0.6057</td><td>0.3087</td></tr><tr><td>Qwen3-1.7B LoRA</td><td>0.3830</td><td>0.5905</td><td>1.8200</td><td>0.7692</td><td>0.6105</td><td>0.3143</td><td>0.3389</td><td>0.5711</td><td>1.6767</td><td>0.7743</td><td>0.5890</td><td>0.3253</td></tr><tr><td>Qwen3-4B</td><td>0.4480</td><td>0.6650</td><td>1.3780</td><td>0.8826</td><td>0.6915</td><td>0.2328</td><td>0.4367</td><td>0.6433</td><td>1.4000</td><td>0.8736</td><td>0.7150</td><td>0.2090</td></tr><tr><td>Qwen3-4B LoRA</td><td>0.4335</td><td>0.6580</td><td>1.4035</td><td>0.8777</td><td>0.6875</td><td>0.2343</td><td>0.4144</td><td>0.6578</td><td>1.3944</td><td>0.8640</td><td>0.7157</td><td>0.2127</td></tr><tr><td>Qwen3-8B</td><td>0.4570</td><td>0.6875</td><td>1.2495</td><td>0.8835</td><td>0.7180</td><td>0.2108</td><td>0.4289</td><td>0.6833</td><td>1.2567</td><td>0.8711</td><td>0.7483</td><td>0.1723</td></tr><tr><td>Qwen3-8B LoRA</td><td>0.4540</td><td>0.6855</td><td>1.2845</td><td>0.8848</td><td>0.7135</td><td>0.2113</td><td>0.4233</td><td>0.6622</td><td>1.3022</td><td>0.8797</td><td>0.7493</td><td>0.1797</td></tr><tr><td>Qwen3-14B</td><td>0.4560</td><td>0.6540</td><td>1.3550</td><td>0.8700</td><td>0.7525</td><td>0.1815</td><td>0.4388</td><td>0.6622</td><td>1.1233</td><td>0.8074</td><td>0.7697</td><td>0.1657</td></tr><tr><td>Qwen3-14B LoRA</td><td>0.4990</td><td>0.7265</td><td>1.1150</td><td>0.8863</td><td>0.7410</td><td>0.1878</td><td>0.4755</td><td>0.7011</td><td>1.1233</td><td>0.8563</td><td>0.7677</td><td>0.1657</td></tr><tr><td>Qwen3-32B</td><td>0.4696</td><td>0.7350</td><td>1.1324</td><td>0.9070</td><td>0.7821</td><td>1.1589</td><td>0.4492</td><td>0.7423</td><td>0.9840</td><td>0.8620</td><td>0.8042</td><td>0.1382</td></tr><tr><td>Qwen3-32B LoRA</td><td>0.4980</td><td>0.7275</td><td>1.0850</td><td>0.9035</td><td>0.7770</td><td>0.1633</td><td>0.4844</td><td>0.7344</td><td>1.0090</td><td>0.8567</td><td>0.8007</td><td>0.1404</td></tr></table>

Table 9: Ablation study of cross-protocol transformation.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Setting</td><td colspan="3">Pointwise</td><td>Pairwise</td><td colspan="2">Listwise</td></tr><tr><td>Acc ↑</td><td> $A c c _ { \pm 1 } \uparrow$ </td><td>MAE↓</td><td>Acc ↑</td><td>Acc↑</td><td>MAE↓</td></tr><tr><td rowspan="4">Alpaca</td><td>Pointwise only</td><td>0.325</td><td>0.544</td><td>1.883</td><td>0.171</td><td>0.235</td><td>0.634</td></tr><tr><td>Pairwise only</td><td>0.025</td><td>0.289</td><td>3.139</td><td>0.832</td><td>0.374</td><td>0.471</td></tr><tr><td>Listwise only</td><td>0.123</td><td>0.401</td><td>2.629</td><td>0.637</td><td>0.696</td><td>0.240</td></tr><tr><td>JUDGESTEALER</td><td>0.383</td><td>0.591</td><td>1.776</td><td>0.771</td><td>0.634</td><td>0.300</td></tr><tr><td rowspan="4">GPT4All</td><td>Pointwise only</td><td>0.315</td><td>0.572</td><td>1.661</td><td>0.446</td><td>0.180</td><td>0.765</td></tr><tr><td>Pairwise only</td><td>0.146</td><td>0.343</td><td>3.267</td><td>0.753</td><td>0.342</td><td>0.558</td></tr><tr><td>Listwise only</td><td>0.067</td><td>0.301</td><td>3.192</td><td>0.647</td><td>0.633</td><td>0.296</td></tr><tr><td>JUDGESTEALER</td><td>0.361</td><td>0.610</td><td>1.611</td><td>0.783</td><td>0.635</td><td>0.295</td></tr></table>

More Cross-protocol Agreement. We further analyze crossprotocol agreement on less capable judges using models from the Qwen3 family, with the results reported in Figure 8. For smaller models, agreement across pointwise, pairwise, and listwise protocols remains relatively low, suggesting that a consistent underlying evaluation criterion is not yet clearly established. As model scale and judging capability increase, however, all three forms of cross-protocol agreement improve steadily, reaching nearly 90% for Qwen3-235B-A22B. This trend indicates a strong association between judging capability and cross-protocol agreement: more capable judges tend to exhibit more consistent evaluation behavior across different protocols. From a model-extraction perspective, this relationship further strengthens the practical threat considered in JUDGESTEALER. High-capability judges are typically more valuable extraction targets, yet their stronger cross-protocol agreement also provides greater opportunity to reuse supervision across protocols, thereby facilitating query-efficient extraction. Conversely, although weaker judges exhibit lower agreement and are therefore less vulnerable to such crossprotocol exploitation, their limited judging capability also makes them less attractive targets for model extraction.

Runtime Analysis. We further report the runtime breakdown of JUDGESTEALER to characterize its computational overhead. Using Alpaca with Qwen3-1.7B under a query budget of 600, the complete extraction pipeline requires 65 min 34 s. The sample selection mechanism takes 15 min 16 s (23.28%) of the total runtime, while Stage I pointwise training requires 5 min 05 s (7.76%). In Stage II, pairwise and listwise adaptation take 8 min 58 s (13.68%) and 9 min 01 s (13.74%), respectively. The final multi-protocol consolidation accounts for the largest portion of the runtime, requiring 27 min 15 s (41.54%).

Table 10: Evaluation against the anomaly detector.
<table><tr><td>Model</td><td>W</td><td>τW</td></tr><tr><td>Llama-3.2-1B</td><td>0.9370</td><td>0.9223</td></tr><tr><td>Qwen3-1.7B</td><td>0.9240</td><td>0.9223</td></tr></table>

More Detailed Results. We present the full results of surrogate models across model scales and adaptation strategies in Table 8. We present the results of JUDGESTEALER against anomaly detection and ownership tracing in Table 10 and 11, respectively.

![](images/5524afd5b82384ec520f58a815e15e2ba4db85e4ff8888e934b636323f7f0dd6.jpg)  
Figure 6: Prompt template for pairwise evaluation.

Table 11: Evaluation against ownership tracing defense.
<table><tr><td>Role</td><td>Model</td><td>Method</td><td>Avg. Z-score</td></tr><tr><td>Victim</td><td>Qwen3-32B</td><td>Watermarked</td><td>6.9173</td></tr><tr><td rowspan="2">Surrogate</td><td>Qwen3-1.7B</td><td>No defense With defense</td><td>-1.0170 -0.2191</td></tr><tr><td>Llama-3.2-1B</td><td>No defense With defense</td><td>-1.1081 -0.2559</td></tr></table>

![](images/8bf2692e052918f1f85d1d1348423173ace318e3bbde05e4c58d42112efdbe34.jpg)  
Figure 7: Prompt template for listwise evaluation.

![](images/0d606448120e7c64094c203727eba6a31bda4866f74f3a7a890d1c9c6ae7b4ee.jpg)  
Figure 8: Cross-protocol agreement on less capable judges.