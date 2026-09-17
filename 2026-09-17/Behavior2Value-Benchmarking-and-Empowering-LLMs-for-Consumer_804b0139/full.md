# Behavior2Value: Benchmarking and Empowering LLMs for Consumer Value Measurement from E-commerce Behaviors

Peixuan Hou<sup>1</sup>, Bin Chen<sup>1</sup>, Li He<sup>2</sup>, Jian Xu<sup>2</sup>, Bo Zheng<sup>2</sup>, Xiuli Ma<sup>1</sup>, Guojie Song 1∗

<sup>1</sup>State Key Laboratory of General Artificial Intelligence,

School of Intelligence Science and Technology, Peking University

<sup>2</sup>Alibaba Group

gjsong@pku.edu.cn

## Abstract

Human values are deep motivational orientations that shape human behaviors. In e-commerce, they reveal the stable drivers behind users’ purchase decisions. Compared with short-term interests, consumer values better explain how users evaluate products before purchase. However, consumer values are often implicit in complex and fragmented behavioral trajectories, leaving value measurement from e-commerce behaviors largely underexplored. To this end, we propose the Behavior-to-Value (B2V) task, which aims to identify consumer values from e-commerce behavioral trajectories. Centered on this task, we first construct the E-commerce Consumption Value Taxonomy (ECVT) and introduce B2V-Bench, the first B2V dataset and benchmark, based on anonymized Taobao behavioral logs. B2V-Bench consists of real-world purchase decision episodes, covering 25 types of purchase behaviors, along with corresponding consumer value orientations manifested in each episode. To improve consumer value measurement accuracy, we further present B2V-Verifier, a behavior-to-value measurement model based on Value Verification Tuning, which learns to assess whether behaviors provide suficient evidence for each value inference. Experiments show that B2V-Verifier outperforms strong LLM baselines, improving multi-label classification by 34%. The dataset and code will be publicly released upon acceptance.

## 1 Introduction

Human values are stable beliefs about desirable goals and principles (Rokeach 1973; Schwartz 1992). Unlike shortterm interests, they reflect deeper motivational orientations that shape how people interpret situations, compare alternatives, and ultimately take action. Therefore, studying values provides a foundation for explaining stable and transferable patterns in human behavior (Bardi and Schwartz 2003).

Values broadly influence human life and play a profound role in consumer behavior (Manolios, Hanjalic, and Liem 2019; Ge et al. 2020). In e-commerce, for example, riskaverse users may repeatedly inspect reviews and compare buyer feedback before purchase to reduce decision uncertainty, while brand-oriented users may prefer oficial stores and well-known brands. These value-driven behavioral patterns are crucial for e-commerce, as they reveal users’ enduring priorities, account for behavioral consistency across situations, and support more accurate and personalized ecommerce services (Zhang and Chen 2020). Nevertheless, consumer values are often implicit in complex and fragmented behavioral traces, leaving value measurement from e-commerce behaviors largely underexplored. To fill this gap, we propose the Behavior-to-Value (B2V) task, which aims to identify the consumer value tendencies manifested in an episode of e-commerce behavior, as illustrated in Figure 1. This task provides deeper support for user understanding, consumption behavior analysis and personalized services.

However, existing methods are limited for the B2V task. Traditional sequential models (Hidasi et al. 2015; Zhou et al. 2018; Sun et al. 2019) struggle to capture value orientations embedded in product attributes and behavioral actions. Large language models (LLMs) (Comanici et al. 2025; Singh et al. 2025; Liu et al. 2025; Yang et al. 2025) provide stronger semantic reasoning capabilities (Zhao et al. 2023), yet without task-specific data and explicit evidence constraints, they may still rely on superficial behavior–value associations and produce unreliable value judgments. Overall, B2V faces two key challenges: (1) Data scarcity. Existing ecommerce datasets (Jin et al. 2024; Ben-Shimon et al. 2015; Jin et al. 2023) lack consumer value annotations, while no ecommerce value taxonomy exists. Moreover, they cover only a few coarse-grained behavior types and treat isolated interactions as individual samples, obscuring users’ pre-purchase deliberation. (2) Value identification dificulty. Values are latent and cannot be inferred through a simple “behaviorto-value” mapping. Reliable value identification requires assessing whether behavioral evidence suficiently supports a value orientation; otherwise, models may over-attribute values from isolated cues while ignoring counter-evidence.

To tackle these challenges, we develop the E-commerce Consumption Value Taxonomy (ECVT), comprising 30 fine-grained value constructs tailored to e-commerce. It is grounded in real-world e-commerce reviews, refined with expert knowledge from established psychological value scales. Based on ECVT, we construct B2V-Bench, the first dataset and benchmark for the B2V task. B2V-Bench contains 5,440 real-world purchase-decision episodes derived from anonymized behavioral logs on Taobao, one of the world’s largest e-commerce platforms. Centered on a completed purchase, each episode contains the multi-step pre-purchase behaviors leading to that purchase, covering 25 behavior types such as queries, clicks, review inspection, detail-page browsing, favorites, and add-to-cart actions. For each episode,

![](images/67eb125633834c630d2ca731d19f2da3cd2ae56b3e6e44af93e8091fa1494426.jpg)  
Figure 1: Illustration of the proposed Behavior-to-Value task. B2V measures the consumer value tendencies reflected in a specific episode of e-commerce behavior, enabling more interpretable consumer behavior analysis and personalized services.

B2V-Bench provides a summary of the behavior sequence, value labels from our ECVT, and annotation rationales.

Building on this dataset, we propose B2V-Verifier, a behavior-to-value measurement model based on Value Verification Tuning. Rather than directly mapping behavioral trajectories to values, B2V-Verifier learns to verify whether a value label is suficiently supported by behavioral evidence, thereby producing more accurate and reliable value predictions. Extensive experiments show that B2V-Verifier consistently outperforms multiple advanced LLMs, with approximate gains of 34% in multi-label classification, 7.4% in label ranking, and 5.5% in primary-label identification.

In summary, the key contributions are as follows:

• We introduce the Behavior-to-Value task, which identifies consumer value orientations from e-commerce behavior trajectories and provides a new perspective for consumption behavior analysis and personalized services.

• The E-commerce Consumption Value Taxonomy is proposed to provide a label space for consumer values in e-commerce. Based on ECVT, we construct B2V-Bench, a dataset and benchmark comprising real-world e-commerce behavioral trajectories, each paired with its summary, value labels, and annotation rationales.

• We present B2V-Verifier as a behavior-to-value measurement model based on Value Verification Tuning, learning to assess whether behaviors provide suficient evidence for each value inference. Experiments show that B2V-Verifier outperforms strong baselines, with improvements of 34% in multi-label classification.

## 2 B2V-Bench: A New Dataset for B2V

Although existing e-commerce datasets are widely used for recommendation and behavior modeling, they are not well suited for the B2V task. As summarized in Table 1, they have three limitations: (1) no value-level labels, (2) limited behavior types, and (3) individual interactions as annotation units. These constraints make them insuficient for inferring consumption values from users’ decision processes.

To address this gap, we propose the E-commerce Consumption Value Taxonomy (ECVT), comprising 30 finegrained value constructs tailored to e-commerce. Based on ECVT, we introduce B2V-Bench, which contains 5,440 samples. Each sample is built around a real purchase-decision episode, which consists of multi-step pre-purchase behaviors leading to the same completed purchase. The sample further includes a summary of the behavioral trajectory, value labels, and rationale evidence supporting these labels. B2V-Bench captures richer behavioral evidence with 25 types of pre-purchase actions, including search, click, add-to-cart, coupon inspection, specification checking, review browsing, Q&A viewing and customer service interaction (see details in Appendix C). We annotate behavior sequences that capture a complete purchase decision process (341 actions on average), because values emerge from the overall process.

## 2.1 E-commerce Consumption Value Taxonomy

Constructing a B2V dataset requires an e-commercespecific value taxonomy. Classic consumption value theories (Sweeney and Soutar 2001; Holbrook 1999; Sheth, Newman, and Gross 1991) are too coarse-grained to capture the fine-grained motivations in modern online commerce. For example, the Theory of Consumption Values (TCV) (Sheth, Newman, and Gross 1991) may broadly treat review reading and coupon checking as functional value, while in ecommerce they reflect distinct concerns such as risk aversion and price sensitivity. E-commerce reviews provide a natural basis for taxonomy construction, as they capture users’ post-purchase evaluations in real consumption contexts. Accordingly, we construct ECVT by deriving value dimensions from large-scale reviews and refining them with established consumption value scales. As shown in Figure 2, ECVT comprises 30 fine-grained value constructs organized into the five dimensions of the Theory of Consumption Values (TCV).

<table><tr><td colspan="2">Type Dataset</td><td>Task</td><td></td><td>Behavior Type Annotation Unit</td><td>R</td><td>Value Labels</td><td>Source</td></tr><tr><td rowspan="10">EF-orce</td><td>Amazon Review (Keung et al. 2020)</td><td>Review prediction</td><td>1</td><td>per-item (1 step)</td><td>x</td><td>x</td><td>Amazon</td></tr><tr><td>Amazon-M2 (Jin et al. 2023)</td><td>Recommendation</td><td>1</td><td>per-session (avg. 4.2 steps)</td><td>x</td><td>x</td><td>Amazon</td></tr><tr><td>SessionIntentBench (Yang et al. 2026)</td><td>Intention-shift modeling</td><td>1</td><td>per-task (avg. 3.4 steps)</td><td>x</td><td>x</td><td>Amazon</td></tr><tr><td>Repeat Buyers (Liu et al. 2016)</td><td>Buyer prediction</td><td>4</td><td>per-click (1 step)</td><td>x</td><td>x</td><td>Tmall</td></tr><tr><td>Taobao (Zhu et al. 2018)</td><td>Recommendation</td><td>4</td><td>per-click (1 step)</td><td>x</td><td>x</td><td>Taobao</td></tr><tr><td>YOOCHOOSE (Ben-Shimon et al. 2015)</td><td>Purchase prediction</td><td>2</td><td>per-session (avg. 3.5 steps)</td><td>x</td><td>x</td><td>Retailer</td></tr><tr><td>Shopping MMLU (Jin et al. 2024)</td><td>Recommendation</td><td>3</td><td>per-item (1 step)</td><td>x</td><td>x</td><td>Amazon</td></tr><tr><td>MerRec (Li et al. 2024)</td><td>Multi-task</td><td>5</td><td>per-session (avg. 5.6 steps)</td><td>x</td><td>x</td><td>Mercari</td></tr><tr><td>OPeRA (Wang et al. 2026)</td><td>Behavior simulation</td><td>8</td><td>per-action (1 step)</td><td>x</td><td>x</td><td>Study</td></tr><tr><td>ValueBench (Ren et al. 2024)</td><td>Value probing (LLM)</td><td>x</td><td>per-item (1 step)</td><td>x</td><td>10, Schwartz</td><td>Synthetic</td></tr><tr><td colspan="2">Value PVQ (Schwartz et al. 2001)</td><td>Self-report PVQ</td><td>x</td><td>per-questionnaire</td><td>x</td><td>10, Schwartz</td><td>Survey</td></tr><tr><td colspan="2">Ours B2V-Bench</td><td>B2V</td><td>25</td><td>per-session (avg. 341 steps)</td><td>√</td><td>28, ECVT</td><td>Taobao</td></tr></table>

Table 1: Comparison of existing e-commerce and value datasets with B2V-Bench. Annotation Unit: the data instance being annotated with the average number of behavior steps per unit. R: whether annotations include explicit supporting reasons.

![](images/03acf8a7f31b261dc53d42c22cd81735221977af14a8e40402a9a5f7f65cd284.jpg)  
Figure 2: Overview of ECVT, comprising 30 fine-grained consumer value constructs across the five TCV dimensions.

For bottom-up value discovery, we use Amazon Reviews 2023 as empirical grounding to derive candidate value dimensions from real consumption contexts. We split each review into independent statements that express specific evaluations. Leveraging the semantic understanding and world knowledge of LLMs, we organize these statements into candidate value dimensions and describe each by its meaning, decision role, polarity, and broader value category. To provide theoretical grounding, we consolidate established consumption value scales from major frameworks (Sheth, Newman, and Gross 1991; Sweeney and Soutar 2001; Holbrook 1999; Blut et al. 2024; Fornell et al. 1996; Parasuraman, Zeithaml, and Malhotra 2005; Loiacono, Watson, and Goodhue 2007; Reimers and Gurevych 2019; Zheng et al. 2023). We then cluster semantically similar candidates and invite psychology experts to refine them with reference to these scales, removing redundancy and noise while preserving meaningful long-tail values, ultimately yielding 30 value constructs.

We also conduct a comprehensive psychometric validation of ECVT. The results in Appendix D show that ECVT is semantically coherent, clearly diferentiated, broadly applicable across product categories, and theoretically grounded, while capturing fine-grained values specific to e-commerce.

## 2.2 Behavior-to-Value Annotation

We obtained anonymized user behavior data from Taobao. Each behavior episode consists of a completed order and the user’s same-category interactions during the preceding day, capturing a coherent purchase goal and serving as the basic unit for value annotation. The key challenge is to infer latent consumption values from behaviors. Traditional self-report value questionnaires (Rokeach 1973; Schwartz 1992; Schwartz et al. 2001) provide a mature measurement paradigm but are costly to administer and unsuitable for large-scale behavioral data. We therefore design a four-stage annotation process: (1) behavior summarization condenses redundant raw logs into concise summaries; (2) a protocol is developed to define annotation rules and behavioral portrait items that describe the behaviors supporting each ECVT construct; (3) AutoPVQ, a multi-agent framework that adapts the measurement logic of the Portrait Values Questionnaire (PVQ) (Schwartz et al. 2001), matches the summaries against these behavioral portrait items to generate initial annotations; and (4) human experts verify and refine the results.

Behavior Summarization. Direct annotation over raw behavior logs is costly and error-prone, as many interactions are redundant and obscure key evidence underlying the purchase decision. We therefore compress each episode into a standardized behavioral summary that preserves critical decision signals, including product comparison, information verification, candidate switching, and final choice. To preserve the purchase-decision process, we compare multiple LLMs and prompt variants through blinded expert evaluation. GPT-5.4 achieves the highest overall quality and is therefore used to generate the behavioral representations for annotation. Full evaluation details are provided in Appendix G.

Value Construct Operationalization Protocol. To provide AutoPVQ with behavioral portrait items and clear annotation rules, we develop the Value Construct Operationalization Protocol (VCOP). VCOP specifies the behaviors that support each value and the criteria for distinguishing similar values, enabling reliable and theoretically grounded annotation.

For each value construct in ECVT, VCOP describes its core meaning, typical behavioral evidence, referred to as be-

Traditional Portrait Values Questionnaire (PVQ):

![](images/1d0c24a438d0b9f8668fc5dfa0b9dd699cd29bc7113f22c5eca90220662177b5.jpg)  
Figure 3: Framework of AutoPVQ. AutoPVQ adapts PVQ-style value measurement to e-commerce behavior annotation through behavior-item scoring, construct-level aggregation, and evidence-based label verification.

<table><tr><td>Metric</td><td>GPT-5.4</td><td>Claude-Sonnet</td><td>Qwen-3.5</td></tr><tr><td>Primary Agreement (%) ↑</td><td>86.7</td><td>82.9</td><td>79.6</td></tr><tr><td>Label-set Jaccard ↑</td><td>0.812</td><td>0.774</td><td>0.741</td></tr><tr><td>Exact Match (%) ↑</td><td>72.4</td><td>67.1</td><td>63.5</td></tr><tr><td>Cohen&#x27;s κ ↑</td><td>0.781</td><td>0.735</td><td>0.692</td></tr></table>

Table 2: Cross-model and human validation results of AutoPVQ on 1,000 behavior episodes.

havioral portrait items, possible counter-evidence, and distinctions from similar constructs. It is developed through a theory-driven and empirically refined process: initial guidelines are derived from established value scales, e-commerce review data, and consumer psychology literature, and then iteratively improved through pilot annotation and disagreement analysis. This process supports standardized and reliable annotation (see details in Appendix F).

AutoPVQ Annotation. PVQ measures values by assessing how closely a person matches portraits that describe valuerelated priorities. Following this logic, AutoPVQ compares each behavioral summary with the behavioral portrait items of each value construct. As shown in Figure 3, it decomposes annotation into specialized steps: portrait-item scoring, discriminative verification, feedback refinement, and result aggregation. Cross-agent verification and iterative refinement improve the reliability of the initial annotations.

Based on the behavior summary, the Portrait-Item Scorer first evaluates how well the episode matches the behavioral portrait items defined in VCOP. Because the value implications of e-commerce behaviors often depend on product attributes and category semantics, the scorer leverages the semantic knowledge of LLMs to contextualize observable evidence. For example, cues such as “portable” and “installation-free” may support the construct of convenience. Such reasoning is used only to interpret evidence, while the final judgment remains constrained by VCOP.

The Report Agent combines the support scores of individual behavioral portrait items into an overall score for each value construct. For each value construct v with associated VCOP items $I _ { v }$ , the raw score is computed as:

$$
\operatorname { R a w } ( v ) = { \frac { \sum _ { i \in I _ { v } } w _ { i } s _ { i } } { \sum _ { i \in I _ { v } } w _ { i } } } ,
$$

where $s _ { i } \in \{ 0 , 1 , 2 , 3 \}$ is the support score for each item and $w _ { i }$ is its diagnostic weight, reflecting evidence strength. To measure the relative salience of each value within an episode, we apply episode-level centering:

$$
\mathrm { S c o r e } ( v ) = \mathrm { R a w } ( v ) - \frac { 1 } { | V | } \sum _ { v ^ { \prime } \in V } \mathrm { R a w } ( v ^ { \prime } ) ,
$$

where V is the candidate value set. The top construct is assigned as the primary label if its score exceeds τ ; secondary labels are assigned when additional constructs exceed $\tau _ { 2 }$ and have independent evidence. The final output includes value labels, a quality score, and an evidence summary. To reduce over-inference and construct confusion, the Valid Agent verifies each candidate label against VCOP distinction rules and counter-evidence. Insuficiently supported labels are rescored by the Portrait-Item Scorer.

Expert Review and Correction. We randomly sample 1,000 behavior episodes as an expert-annotated reference set and invite three psychology experts to jointly annotate them from scratch and resolve disagreements through discussion to reach consensus. To select the backbone for AutoPVQ, we instantiate three advanced LLMs under the same multi-agent protocol and compare their outputs with expert annotations. As shown in Table 2, GPT-5.4 achieves the best overall alignment with expert judgments and is adopted as the backbone for full-scale AutoPVQ annotation. Human annotators further review and revise its outputs following VCOP guidelines. They assess label validity, evidence suficiency, and potential over-inference, and correct unsupported or ambiguous value labels to produce the final human-verified dataset.

![](images/931017c4ea60718d7cae5ebf23b1d9e70f737549ab96e29af854722639e1fd0a.jpg)

![](images/888bbb2689db13070b49ba7000744ad6b50c5c07a847fd42d31502e069fc7ce0.jpg)

![](images/e2f40c69f79cb54c42487769a12ad0e3b89b287e5c9eaad5fb4dfd0d0c6739a3.jpg)

Figure 4: Characteristics of B2V-Bench. (a) Label co-occurrence patterns among frequent value constructs. (b) Distribution of behavioral summary lengths. (c) Monthly data volume and TCV composition over time.  
![](images/98e5fef5b2b39e35f398704ef29f0c2b1b8a2b3159804068411b330980600711.jpg)  
Figure 5: Overview of Value Verification Tuning. Given a behavior summary and verbalized value descriptions, B2V-Verifier determines whether the observed behavior supports the value and predicts a three-way verification label.

## 2.3 Dataset Evaluation and Analysis

Dataset Split and Quality Control. B2V-Bench is collected from 13 monthly snapshots spanning March 2025 to March 2026, comprising 5,440 annotated episodes. A total of 4,352 episodes are used for training and 544 for validation, with labels generated by AutoPVQ and then reviewed and corrected by human annotators. The 544-episode test set is sampled from a pool of 1,000 episodes annotated from scratch by psychology experts, ensuring model-independent evaluation. Dataset Characteristics. B2V-Bench covers a broad and fine-grained label space of consumer values. It uses 28 value labels from ECVT, with an additional none tag for evidenceinsuficient episodes to avoid forced value attribution. After expert discussion, decision complexity and exploratory interest are excluded from the default label set because they are dificult to identify reliably from a single behavior episode. B2V-Bench exhibits a clear multi-label structure: 75.2% of episodes include at least one secondary value, indicating that purchase decisions often reflect multiple value orientations. Figure 4(a) shows clear asymmetric co-occurrence patterns among frequent value labels, such asfunctional eficacy with perceived risk and brand trust with perceived risk. These directional dependencies suggest that consumer values often appear in primary–secondary relations, supporting our multi-label formulation and the need for disambiguation rules. We further examine the evidence suficiency and temporal stability of B2V-Bench. Figure 4(b) shows that most behavioral summaries contain 500–1000 characters, providing suficient context for value inference. Figure 4(c) shows comparable session volumes and stable TCV composition across monthly snapshots, suggesting that B2V-Bench captures consistent patterns rather than seasonal artifacts.

## 3 Method: B2V-Verifier

Building on B2V-Bench, we develop B2V-Verifier by applying Value Verification Tuning (VVT) to Qwen3-8B. Unlike standard label classification, measuring consumption values from behavior requires context-dependent inference from indirect behavioral evidence, since identical surface behaviors may reflect diferent motivations. For example, a high-priced purchase does not necessarily imply low price sensitivity, nor does checking reviews always indicate risk aversion. So, B2V prediction should not only associate behaviors with value labels, but also determine whether the observed trajectory provides suficient evidence for a specific value inference.

Motivated by this observation, we design Value Verification Tuning (VVT), which reformulates B2V prediction as a value-wise verification task. As illustrated in Figure 5, for each value construct, we derive a behavioral value description from VCOP that defines the construct and specifies the concrete behaviors that support it. Given a behavior summary and a candidate value description, the model assesses whether the observed trajectory provides suficient evidence for that value. This formulation decomposes multi-label prediction into explicit evidence judgments for individual value constructs, rather than directly generating a likely label set. It therefore preserves the multi-label nature of B2V while requiring each predicted value to be independently justified by the behavior, reducing unsupported attribution.

Formally, given a behavior summary x and a value construct $v ,$ we derive a behavioral value description $d _ { v }$ that defines v and specifies the concrete behaviors that support it. The model takes $( x , d _ { v } )$ as input and predicts a verification label $y ,$ indicating how well the observed behavior supports

<table><tr><td rowspan="2">Model</td><td rowspan="2">Prompt |</td><td colspan="6">Multi-label Classification</td><td colspan="3">Label Ranking</td><td colspan="3">Label Identification</td></tr><tr><td>|Ma-F1 ↑</td><td>Mi-F1 ↑</td><td>W-F1↑</td><td>Prec. ↑</td><td>Rec. ↑</td><td>Ham. ↓</td><td>|N@1↑</td><td>N@2↑ N@3↑</td><td></td><td>Acc. ↑</td><td>T2 Acc. ↑</td><td>Avg. ↑</td></tr><tr><td rowspan="4">Claude-Haiku-4.5 Anthropic (2025)</td><td>P1</td><td>0.256</td><td>0.510</td><td>0.559</td><td>0.313</td><td>0.407</td><td>0.083</td><td>0.671</td><td>0.610</td><td>0.633</td><td>0.548</td><td>0.677</td><td>0.613</td></tr><tr><td>P2</td><td>0.286</td><td>0.576</td><td>0.600</td><td>0.315</td><td>0.393</td><td>0.068</td><td>0.714</td><td>0.672</td><td>0.672</td><td>0.574</td><td>0.743</td><td>0.659</td></tr><tr><td>P3</td><td>0.313</td><td>0.596</td><td>0.612</td><td>0.367</td><td>0.398</td><td>0.060</td><td>0.730</td><td>0.692</td><td>0.666</td><td>0.593</td><td>0.770</td><td>0.682</td></tr><tr><td>P4</td><td>0.275</td><td>0.624</td><td>0.622</td><td>0.301</td><td>0.355</td><td>0.053</td><td>0.757</td><td>0.727</td><td>0.679</td><td>0.604</td><td>0.777</td><td>0.691</td></tr><tr><td rowspan="4">DeepSeek-V3.2 Liu et al. (2025)</td><td>P1</td><td>0.266</td><td>0.568</td><td>0.605</td><td>0.300</td><td>0.433</td><td>0.073</td><td>0.717</td><td>0.676</td><td>0.688</td><td>0.569</td><td>0.732</td><td>0.651</td></tr><tr><td>P2</td><td>0.289</td><td>0.597</td><td>0.611</td><td>0.267</td><td>0.456</td><td>0.065</td><td>0.721</td><td>0.697</td><td>0.692</td><td>0.561</td><td>0.757</td><td>0.659</td></tr><tr><td>P3</td><td>0.317</td><td>0.619</td><td>0.645</td><td>0.318</td><td>0.442</td><td>0.059</td><td>0.757</td><td>0.711</td><td>0.701</td><td>0.595</td><td>0.757</td><td>0.676</td></tr><tr><td>P4</td><td>0.350</td><td>0.655†</td><td>0.659</td><td>0.380</td><td>0.422</td><td>0.050†</td><td>0.773</td><td>0.749†</td><td>0.711</td><td>0.613</td><td>0.781†</td><td>0.697†</td></tr><tr><td rowspan="4">Gemini-2.5 Comanici et al. (2025)</td><td>P1</td><td>0.276</td><td>0.582</td><td>0.570</td><td>0.350</td><td>0.353</td><td>0.067</td><td>0.773</td><td>0.691</td><td>0.691</td><td>0.563</td><td>0.695</td><td>0.629</td></tr><tr><td>P2</td><td>0.248</td><td>0.583</td><td>0.549</td><td>0.369</td><td>0.339</td><td>0.064</td><td>0.775</td><td>0.703</td><td>0.675</td><td>0.578</td><td>0.717</td><td>0.648</td></tr><tr><td>P3</td><td>0.263</td><td>0.596</td><td>0.568</td><td>0.350</td><td>0.355</td><td>0.060</td><td>0.781†</td><td>0.713</td><td>0.674</td><td>0.612</td><td>0.734</td><td>0.673</td></tr><tr><td>P4</td><td>0.242</td><td>0.604</td><td>0.572</td><td>0.305</td><td>0.301</td><td>0.054</td><td>0.775</td><td>0.726</td><td>0.657</td><td>0.615</td><td>0.751</td><td>0.683</td></tr><tr><td rowspan="3">GLM-5 Zeng et al. (2026)</td><td>P1 P2</td><td>0.308 0.335</td><td>0.585 0.609</td><td>0.612 0.628</td><td>0.297 0.354</td><td>0.433 0.415</td><td>0.066 0.058</td><td>0.708 0.704</td><td>0.679 0.673</td><td>0.676 0.659</td><td>0.569 0.578</td><td>0.704 0.714</td><td>0.637 0.646</td></tr><tr><td>P3</td><td>0.339</td><td>0.625</td><td>0.637</td><td>0.379</td><td>0.389</td><td>0.053</td><td>0.673</td><td>0.666</td><td>0.647</td><td>0.556</td><td>0.717</td><td>0.637</td></tr><tr><td>P4</td><td>0.358</td><td>0.608</td><td>0.622</td><td>0.417†</td><td>0.405</td><td>0.055</td><td>0.665</td><td>0.647</td><td>0.630</td><td>0.558</td><td>0.703</td><td>0.631</td></tr><tr><td rowspan="4">GPT-4o-mini OpenAI (2024)</td><td>P1 P2</td><td>0.248 0.253</td><td>0.514 0.504</td><td>0.525</td><td>0.297</td><td>0.316</td><td>0.083 0.084</td><td>0.710 0.721</td><td>0.642</td><td>0.641</td><td>0.517 0.526</td><td>0.638 0.645</td><td>0.578 0.586</td></tr><tr><td></td><td>0.248</td><td>0.503</td><td>0.521</td><td>0.303</td><td>0.298</td><td></td><td></td><td>0.642</td><td>0.635</td><td></td><td></td><td></td></tr><tr><td>P3 P4</td><td>0.220</td><td>0.569</td><td>0.519</td><td>0.350</td><td>0.298</td><td>0.085</td><td>0.717</td><td>0.644</td><td>0.635</td><td>0.543 0.532</td><td>0.658</td><td>0.601</td></tr><tr><td></td><td></td><td></td><td>0.577</td><td>0.305</td><td>0.273</td><td>0.069</td><td>0.691</td><td>0.657</td><td>0.659</td><td></td><td>0.704</td><td>0.618</td></tr><tr><td rowspan="4">GPT-5 Singh et al. (2025)</td><td>P1</td><td>0.314</td><td>0.556</td><td>0.613</td><td>0.284</td><td>0.447</td><td>0.075</td><td>0.599</td><td>0.623</td><td>0.651</td><td>0.476</td><td>0.701</td><td>0.589</td></tr><tr><td>P2</td><td>0.400</td><td>0.597</td><td>0.654</td><td>0.362</td><td>0.531†</td><td>0.067</td><td>0.686</td><td>0.687</td><td>0.705</td><td>0.548</td><td>0.749</td><td>0.649</td></tr><tr><td>P3</td><td>0.360</td><td>0.629</td><td>0.673†</td><td>0.342</td><td>0.467</td><td>0.062</td><td>0.690</td><td>0.697</td><td>0.725</td><td>0.548</td><td>0.775</td><td>0.662</td></tr><tr><td>P4</td><td>0.417†</td><td>0.628</td><td>0.666</td><td>0.400$</td><td>0.524</td><td>0.060</td><td>0.703</td><td>0.704</td><td>0.719</td><td>0.558</td><td>0.770</td><td>0.664</td></tr><tr><td rowspan="4">Kimi-K2.6 Moonshot AI (2026)</td><td>Pl</td><td>0.275</td><td>0.578</td><td>0.613</td><td>0.273</td><td>0.391</td><td>0.072</td><td>0.762</td><td>0.726</td><td>0.717</td><td>0.621</td><td>0.760</td><td>0.691</td></tr><tr><td>P2</td><td>0.325</td><td>0.585</td><td>0.598</td><td>0.360</td><td></td><td></td><td>0.779</td><td>0.729</td><td>0.723</td><td>0.632†</td><td>0.758</td><td>0.695</td></tr><tr><td>P3</td><td>0.307</td><td>0.593</td><td>0.620</td><td>0.374</td><td>0.432 0.398</td><td>0.070 0.069</td><td>0.757</td><td>0.724</td><td>0.726</td><td>0.604</td><td>0.764</td><td>0.684</td></tr><tr><td>P4</td><td>0.304</td><td>0.591</td><td>0.615</td><td>0.338</td><td>0.381</td><td>0.066</td><td>0.716</td><td>0.704</td><td>0.695</td><td>0.568</td><td>0.737</td><td>0.653</td></tr><tr><td rowspan="4">Qwen3-235B Yang et al. (2025)</td><td>P1</td><td>0.253</td><td>0.559</td><td>0.577</td><td>0.233</td><td>0.404</td><td>0.075</td><td>0.757</td><td>0.691</td><td>0.696 0.698</td><td>0.599 0.574</td><td>0.727 0.742</td><td>0.663 0.658</td></tr><tr><td>P2 P3</td><td>0.271 0.290</td><td>0.565 0.581</td><td>0.582 0.602</td><td>0.280 0.300</td><td>0.386 0.410</td><td>0.074 0.071</td><td>0.730 0.747</td><td>0.700 0.715</td></table>

Table 3: Performance comparison of mainstream LLM baselines and B2V-Verifier on B2V-Bench. P1–P4 denote label-only, +definition, +behavioral value description, and +few-shot prompting settings, respectively. The top three results are highlighted with dark, medium, and light gray backgrounds, respectively, and are marked with ⋆, †, and ‡. For B2V-Verifier, P3 corresponds to its standard input format used in VVT so we evaluate it under the same setting as training. More results are in Appendix B.

the candidate value:

## y ∈ {supported, unsupported, insuficient}.

Specifically, supported is assigned when the trajectory contains clear evidence for the candidate value. A prediction is considered unsupported if the value is irrelevant to, contradicted by, or inconsistent with the observed behavior. When the value appears plausible but the available evidence remains too weak or incomplete to justify the inference, the model outputs insuficient. This distinction is important because a behavior may appear consistent with a value without providing enough evidence to confirm it. Explicitly modeling insuficient evidence reduces uncertain or unsupported inferences, enabling more accurate value measurement.

We construct VVT training instances from B2V-Bench by pairing each behavior trajectory with descriptions of candidate values. Annotated value labels form supported instances. Labels proposed but rejected by the validation agent are assigned insuficient, as they represent plausible interpretations without adequate behavioral evidence. We additionally sample unrelated value constructs as unsupported instances to strengthen the model’s ability to reject irrelevant values. Together, these instances train the model to assess the evidential status of each construct rather than directly generate a label set. During inference, the model verifies each candidate value description against the behavior summary, and all constructs classified as supported constitute the final multi-label prediction. This verification-based formulation enables B2V-Verifier to make conservative, evidence-aware predictions and reduces unsupported value attribution.

<table><tr><td>Model Variant</td><td>Macro-F1</td><td>Micro-F1</td><td>Weighted-F1</td><td>NDCG@1</td><td>NDCG@3</td><td>Primary Acc.</td><td>Top-2 Acc.</td></tr><tr><td>B2V-Verifier (Full VVT)</td><td>0.615</td><td>0.815</td><td>0.812</td><td>0.824</td><td>0.798</td><td>0.652</td><td>0.842</td></tr><tr><td>w/o Verification Formulation</td><td>0.548</td><td>0.766</td><td>0.759</td><td>0.773</td><td>0.742</td><td>0.589</td><td>0.796</td></tr><tr><td>w/o Value Description</td><td>0.574</td><td>0.789</td><td>0.784</td><td>0.794</td><td>0.764</td><td>0.615</td><td>0.813</td></tr><tr><td>w/o Insufficient Modeling</td><td>0.563</td><td>0.781</td><td>0.776</td><td>0.785</td><td>0.755</td><td>0.603</td><td>0.807</td></tr><tr><td>w/o Unsupported Modeling</td><td>0.582</td><td>0.795</td><td>0.789</td><td>0.802</td><td>0.772</td><td>0.624</td><td>0.819</td></tr></table>

Table 4: Ablation study of B2V-Verifier. The full VVT framework consistently outperforms all ablations.

## 4 Experiments

We evaluate the B2V task on our newly constructed benchmark, B2V-Bench. Given a user’s pre-purchase behavioral trajectory and the corresponding item context, the task aims to predict the consumption value labels reflected in the behavioral episode. Formally, let $E = \{ a _ { 1 } , a _ { 2 } , . . . , a _ { T } \}$ denote a behavioral episode, where each action a<sub>t</sub> contains the action type, item information, and contextual semantics. The goal is to predict a set of value labels $Y \subseteq \nu _ { \mathrm { { } } }$ , where V denotes our E-commerce Consumption Value Taxonomy. Based on B2V-Bench, we benchmark traditional sequential models and mainstream LLMs and verify the efectiveness of our proposed B2V-Verifier for B2V measurement.

## 4.1 Experimental setup

Models. For sequential models, we train three representative architectures (Hidasi et al. 2015; Zhou et al. 2018; Sun et al. 2019) on the original structured action sequences, replacing their item-prediction heads with K-way consumption-value classifiers. For LLMs, we benchmark 10 advanced models spanning GPT, Qwen, DeepSeek, Gemini, Claude, GLM, and Kimi. They use the behavioral summaries validated during annotation, which retain decision-relevant evidence while reducing redundant interactions, and are evaluated under four prompting settings P1–P4 with increasing task guidance (see Appendix H). We train B2V-Verifier by supervised finetuning Qwen3-8B with VVT using LLaMA-Factory (Zheng et al. 2024) on four NVIDIA A40 GPUs (See Appendix I). Metrics. We evaluate models from three perspectives: multilabel classification, label ranking, and primary-label identification. Multi-label classification is measured using Macro-F1 (Ma-F1), Micro-F1 (Mi-F1), Weighted-F1 (W-F1), macro Precision (Prec.), macro Recall (Rec.), and Hamming Loss (Ham.) (Sokolova and Lapalme 2009; Schapire and Singer 2000; Tsoumakas and Katakis 2007), assessing label-level correctness, coverage, and robustness under label imbalance. To evaluate the quality of the ranking, we use NDCG@k (k = 1, 2, 3), denoted N @ k (Järvelin and Kekäläinen 2002). For primary-label identification, we report Primary Accuracy (Acc.), Top-2 Accuracy (T2 Acc.), and their average.

## 4.2 Main Results

Sequential models show limited efectiveness on B2V-Bench. The poor performance of GRU4Rec, BERT4Rec, and DIN (See Appendix B) suggests that B2V requires more than modeling user–item interaction sequences alone. This is because the task requires reasoning over product attributes and behavioral actions. As shown in Table 3, LLMs substantially outperform sequential models, highlighting the importance of semantic understanding in behavior-to-value inference. However, adding more information to the prompt does not lead to continued performance gains. More detailed descriptions of value construct generally clarify their semantics and reduce over-attribution, whereas few-shot examples do not consistently improve performance. For some models, demonstrations bias predictions toward familiar label combinations, improving primary-label accuracy but reducing secondary-value recall. Among the LLM baselines, GPT-5 exhibits strong classification performance, while DeepSeek-V3.2 under P4 achieves the most balanced overall results. Nevertheless, all LLM baselines remain clearly behind B2V-Verifier, indicating that general-purpose semantic reasoning alone is insuficient for reliable B2V measurement.

Our B2V-Verifier achieves the best performance across all metrics. The large improvement in Macro-F1 is important, showing that the B2V-Verifier can better recognize diverse and less frequent value constructs rather than relying on dominant value labels. Its superior ranking and primary-label identification performance further demonstrate the efectiveness of VVT for more accurate B2V measurement.

## 4.3 Ablation Study

To assess the contribution of VVT’s designs, we conduct an ablation study on B2V-Bench, comparing the full B2V-Verifier with four variants that remove the verification formulation, replace value descriptions with raw label names, or drop unsupported and insuficient modeling. All variants use the same training and evaluation settings. As shown in Table 4, the full VVT framework outperforms all ablations. Removing the verification formulation causes the largest drop, confirming the advantage of VVT over direct label prediction. The remaining drops further show that explicit value descriptions and fine-grained negative states help clarify construct boundaries and prevent unsupported value attribution.

## 5 Conclusion

In this work, we introduce the Behavior-to-Value (B2V) task, which aims to identify the consumer value tendencies manifested in an episode of e-commerce behavior. To support this task, we construct the E-commerce Consumption Value Taxonomy and introduce B2V-Bench, the first large-scale benchmark for measuring consumer values from e-commerce behaviors. We further present B2V-Verifier, a B2V measurement model based on Value Verification Tuning. Experiments show that it outperforms strong baselines, improving multi-label classification by 34%. This work lays a foundation for value-aware user modeling in e-commerce and opens promising directions for future research in valueaware recommendation and interpretable user profiling.

## References

Anthropic. 2025. Claude Haiku 4.5 System Card. Anthropic System Card.

Bardi, A.; and Schwartz, S. H. 2003. Values and behavior: Strength and structure of relations. Personality and social psychology bulletin, 29(10): 1207–1220.

Ben-Shimon, D.; Tsikinovsky, A.; Friedmann, M.; Shapira, B.; Rokach, L.; and Hoerle, J. 2015. RecSys Challenge 2015 and the YOOCHOOSE Dataset. In Proceedings of the 9th ACM Conference on Recommender Systems, 357–358. Association for Computing Machinery.

Blut, M.; Chaney, D.; Lunardo, R.; Mencarelli, R.; and Grewal, D. 2024. Customer perceived value: a comprehensive meta-analysis. Journal ofservice Research, 27(4): 501–524.

Comanici, G.; Bieber, E.; Schaekermann, M.; Pasupat, I.; Sachdeva, N.; Dhillon, I.; Blistein, M.; Ram, O.; Zhang, D.; Rosen, E.; et al. 2025. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261.

Fornell, C.; Johnson, M. D.; Anderson, E. W.; Cha, J.; and Bryant, B. E. 1996. The American customer satisfaction index: nature, purpose, and findings. Journal of marketing, 60(4): 7–18.

Ge, Y.; Xu, S.; Liu, S.; Fu, Z.; Sun, F.; and Zhang, Y. 2020. Learning personalized risk preferences for recommendation. In Proceedings ofthe 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval, 409–418.

Hidasi, B.; Karatzoglou, A.; Baltrunas, L.; and Tikk, D. 2015. Session-based recommendations with recurrent neural networks. arXiv preprint arXiv:1511.06939.

Holbrook, M. B. 1999. Consumer value: A framework for analysis and research. London and New York.

Järvelin, K.; and Kekäläinen, J. 2002. Cumulated gain-based evaluation of IR techniques. ACM Transactions on Information Systems (TOIS), 20(4): 422–446.

Jin, W.; Mao, H.; Li, Z.; Jiang, H.; Luo, C.; Wen, H.; Han, H.; Lu, H.; Wang, Z.; Li, R.; et al. 2023. Amazon-m2: A multilingual multi-locale shopping session dataset for recommendation and text generation. Advances in Neural Information Processing Systems, 36: 8006–8026.

Jin, Y.; Li, Z.; Zhang, C.; Cao, T.; Gao, Y.; Jayarao, P.; Li, M.; Liu, X.; Sarkhel, R.; Tang, X.; et al. 2024. Shopping mmlu: A massive multi-task online shopping benchmark for large language models. Advances in Neural Information Processing Systems, 37: 18062–18089.

Keung, P.; Lu, Y.; Szarvas, G.; and Smith, N. A. 2020. The Multilingual Amazon Reviews Corpus. arXiv:2010.02573.

Li, L.; Din, Z. A.; Tan, Z.; London, S.; Chen, T.; and Daptardar, A. 2024. MerRec: A Large-scale Multipurpose Mercari Dataset for Consumer-to-Consumer Recommendation Systems. arXiv:2402.14230.

Liu, A.; Mei, A.; Lin, B.; Xue, B.; Wang, B.; Xu, B.; Wu, B.; Zhang, B.; Lin, C.; Dong, C.; et al. 2025. Deepseek-v3.

2: Pushing the frontier of open large language models. arXiv preprint arXiv:2512.02556.

Liu, G.; Nguyen, T.-P.; Zhao, G.; Zha, W.; Yang, J.; Cao, J.; Wu, M.; Zhao, P.; and Chen, W. 2016. Repeat Buyer Prediction for E-Commerce. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 155–164. Association for Computing Machinery.

Loiacono, E. T.; Watson, R. T.; and Goodhue, D. L. 2007. WebQual: An instrument for consumer evaluation of web sites. International journal of electronic commerce, 11(3): 51–87.

Manolios, S.; Hanjalic, A.; and Liem, C. C. 2019. The influence of personal values on music taste: towards valuebased music recommendations. In Proceedings of the 13th ACM Conference on Recommender Systems, 501–505.

Moonshot AI. 2026. Kimi K2.6 Tech Blog: Advancing Open-Source Coding. Moonshot AI Technical Blog.

OpenAI. 2024. GPT-4o mini: Advancing Cost-Eficient In telligence. OpenAI Blog.

Parasuraman, A.; Zeithaml, V. A.; and Malhotra, A. 2005. ES-QUAL: A multiple-item scale for assessing electronic service quality. Journal ofservice research, 7(3): 213–233.

Reimers, N.; and Gurevych, I. 2019. Sentence-bert: Sentence embeddings using siamese bert-networks. In Proceedings of the 2019 conference on empirical methods in natural language processing and the 9th international joint conference on natural language processing (EMNLP-IJCNLP), 3982– 3992.

Ren, Y.; Ye, H.; Fang, H.; Zhang, X.; and Song, G. 2024. ValueBench: Towards comprehensively evaluating value orientations and understanding of large language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2015– 2040.

Rokeach, M. 1973. The nature ofhuman values. Free press. Schapire, R. E.; and Singer, Y. 2000. BoosTexter: A boostingbased system for text categorization. Machine learning, 39(2): 135–168.

Schwartz, S. H. 1992. Universals in the content and structure of values: Theoretical advances and empirical tests in 20 countries. In Advances in experimental social psychology, volume 25, 1–65. Elsevier.

Schwartz, S. H.; Melech, G.; Lehmann, A.; Burgess, S.; Harris, M.; and Owens, V. 2001. Extending the cross-cultural validity of the theory of basic human values with a diferent method of measurement. Journal of cross-cultural psychology, 32(5): 519–542.

Sheth, J. N.; Newman, B. I.; and Gross, B. L. 1991. Why we buy what we buy: A theory of consumption values. Journal ofbusiness research, 22(2): 159–170.

Singh, A.; Fry, A.; Perelman, A.; Tart, A.; Ganesh, A.; El-Kishky, A.; McLaughlin, A.; Low, A.; Ostrow, A.; Ananthram, A.; et al. 2025. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267.

Sokolova, M.; and Lapalme, G. 2009. A systematic analysis of performance measures for classification tasks. Information processing & management, 45(4): 427–437.

Sun, F.; Liu, J.; Wu, J.; Pei, C.; Lin, X.; Ou, W.; and Jiang, P. 2019. BERT4Rec: Sequential recommendation with bidirectional encoder representations from transformer. In Proceedings ofthe 28th ACM international conference on information and knowledge management, 1441–1450.

Sweeney, J. C.; and Soutar, G. N. 2001. Consumer perceived value: The development of a multiple item scale. Journal of retailing, 77(2): 203–220.

Tsoumakas, $\mathrm { G . ; }$ and Katakis, I. 2007. Multi-label classification: An overview. International Journal ofData Warehousing and Mining (IJDWM), 3(3): 1–13.

Wang, Z.; Lu, Y.; Li, W.; Amini, A.; Sun, B.; Bart, Y.; Lyu, W.; Gesi, J.; Wang, T.; Huang, J.; Su, Y.; Ehsan, U.; Alikhani, M.; Li, T. J.-J.; Chilton, L.; and Wang, D. 2026. OPeRA: A Dataset of Observation, Persona, Rationale, and Action for Evaluating LLMs on Human Online Shopping Behavior Simulation. arXiv:2506.05606.

Yang, A.; Li, A.; Yang, B.; Zhang, B.; Hui, B.; Zheng, B.; Yu, B.; Gao, C.; Huang, C.; Lv, C.; et al. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Yang, Y.; Wang, W.; Xu, B.; Fan, W.; Zong, Q.; Chan, C.; Deng, Z.; Liu, X.; Gao, Y.; Yu, C.; Luo, C.; Li, Y.; Li, Z.; Yin, Q.; Yin, B.; and Song, Y. 2026. SessionIntentBench: A Multitask Inter-session Intention-shift Modeling Benchmark for Ecommerce Customer Behavior Understanding. In Findings ofthe Associationfor Computational Linguistics: ACL 2026, 16748–16775. Association for Computational Linguistics.

Zeng, A.; et al. 2026. GLM-5: From Vibe Coding to Agentic Engineering. arXiv:2602.15763.

Zhang, Y.; and Chen, X. 2020. Explainable recommendation: A survey and new perspectives. Foundations and Trends® in Information Retrieval, 14(1): 1–101.

Zhao, W. X.; Zhou, K.; Li, J.; Tang, T.; Wang, X.; Hou, Y.; Min, Y.; Zhang, B.; Zhang, J.; Dong, Z.; et al. 2023. A survey of large language models. arXiv preprint arXiv:2303.18223, 1(2): 1–124.

Zheng, L.; Chiang, W.-L.; Sheng, Y.; Zhuang, S.; Wu, Z.; Zhuang, Y.; Lin, Z.; Li, Z.; Li, D.; Xing, E.; et al. 2023. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in neural informationprocessing systems, 36: 46595– 46623.

Zheng, Y.; Zhang, R.; Zhang, J.; Ye, Y.; and Luo, Z. 2024. LlamaFactory: Unified Eficient Fine-Tuning of 100+ Language Models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), 400–410. Bangkok, Thailand: Association for Computational Linguistics.

Zhou, G.; Zhu, X.; Song, C.; Fan, Y.; Zhu, H.; Ma, X.; Yan, Y.; Jin, J.; Li, H.; and Gai, K. 2018. Deep interest network for click-through rate prediction. In Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining, 1059–1068.

Zhu, H.; Li, X.; Zhang, P.; Li, G.; He, J.; Li, H.; and Gai, K. 2018. Learning Tree-based Deep Model for Recommender Systems. In Proceedings ofthe 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, KDD ’18, 1079–1088. ACM.