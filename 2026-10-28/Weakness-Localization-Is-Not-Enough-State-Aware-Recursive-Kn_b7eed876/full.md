# Weakness Localization Is Not Enough: State-Aware Recursive Knowledge Improvement for Large Language Models

Runze Jiang<sup>1,2</sup>, Yan Wang<sup>2</sup>, Boran Jin<sup>2,3</sup>, Bingyu Zhu<sup>2,†</sup>, Longtao Huang<sup>2</sup> <sup>1</sup>Peking University, <sup>2</sup>Alibaba Group <sup>3</sup>Delft University of Technology <sup>†</sup>Corresponding author

## Abstract

Knowledge improvement in large language models is shifting from predefined knowledge incorporation pipelines toward evaluation-guided updating. Existing methods use evaluation to localize model-specific knowledge weaknesses and guide targeted post-training. However, identifying what to improve does not necessarily provide sufficient information for deciding how to improve it. The same localized weakness can be compatible with distinct operational knowledge states: the relevant knowledge may be unavailable, available but difficult to access reliably, or accessible but ineffective for downstream use. We call this ambiguity knowledge-state aliasing. When compatible states favor different interventions, greater diagnostic resolution alone is insufficient for intervention selection. Moreover, model updates can change the underlying knowledge state before the observed weakness disappears, such that an intervention appropriate in one round may no longer address the bottleneck in the next. To address these challenges, we propose Recursive Knowledge Improvement (RKI), which formulates knowledge improvement as a sequential, state-aware intervention process. RKI organizes knowledge into graph-structured units that provide a shared basis for diagnostic evaluation and targeted intervention across rounds. At each round, RKI localizes knowledge weaknesses, estimates the model’s operational knowledge state from diagnostic evidence and intervention history, selects a corresponding intervention, and re-evaluates the updated model before determining the next intervention. On HOT-POTEVAL and SAFEINTERNALEVAL, RKI outperforms existing knowledge incorporation methods by 18.41 points on average. Further analyses show that knowledge states provide additional guidance for intervention selection beyond weakness localization and can change across model updates even when the observed weakness persists.

## 1 Introduction

Knowledge improvement aims to incorporate external knowledge into large language models so that it can be reliably accessed and used in downstream tasks. In Web settings, this knowledge is typically derived from Web corpora and organized into knowledge graphs, which support applications such as multi-hop question answering and knowledge-grounded agents. Conventional incorporation pipelines (Gururangan et al., 2020; Cheng et al., 2024b) typically follow a predefined procedure, where training data and update strategies are specified with limited consideration of the model’s current knowledge state.

Recent work has shifted toward evaluation-guided improvement, where evaluation extends beyond retrospective assessment to serve as a signal for prioritizing subsequent knowledge updates. Graph-Gen and SENATOR identify model-specific knowledge deficiencies and target post-training toward the corresponding knowledge regions (Chen et al., 2025; Wei et al., 2026). AutoDetect, EvalTree, and

![](images/59e707a5feba869b6beb2540c6845ab67ab64a44a6996912843249f78fc74bda.jpg)  
Figure 1: From weakness localization to stateaware knowledge improvement. RKI estimates operational knowledge states to guide interventions and re-estimates them after each model update.

CRAFT further localize weaknesses and guide targeted improvement through agent-based assessment, hierarchical capability profiling, and rubric-level evaluation, respectively (Cheng et al., 2024c; Zeng et al., 2025; Gupta et al., 2026). These methods identify what should be improved at increasingly fine levels of granularity. However, even a fine-grained diagnosis may leave the choice among training interventions underdetermined. This raises a distinct question: does localizing a weakness provide sufficient information to determine how it should be improved?

The information needed for this choice can lie in the model’s operational knowledge state, consistent with non-binary characterizations of LLM knowledge (Li et al., 2025b). The same localized weakness can reflect different states: the relevant knowledge may be unavailable, available but difficult to access reliably, or accessible but ineffective for understanding, reasoning, or application. We call the compatibility of the same observed weakness with distinct operational states knowledge-state aliasing. This ambiguity becomes consequential for knowledge improvement when these states favor different interventions: factual exposure, recall-focused training, and application-oriented examples can address different bottlenecks within the same knowledge region. In such cases, greater diagnostic resolution does not by itself ensure intervention sufficiency. When compatible states share no optimal intervention, a weakness-only policy cannot be optimal for all of them. Figure 1 illustrates this ambiguity and the state-aware intervention process that it motivates.

A separate challenge arises as knowledge states evolve across model updates. The same observed weakness may persist after an update even when the operational knowledge state underlying it has changed. For example, previously unavailable knowledge may become accessible but remain difficult to recall reliably, or become reliably recallable while remaining difficult to apply effectively. The appropriate intervention can therefore change before the observed weakness disappears. Continuing with the previous intervention may then fail to address the new bottleneck. This motivates re-evaluation and knowledgestate re-estimation before selecting the next intervention.

To address these issues, we propose Recursive Knowledge Improvement (RKI), which views knowledge improvement as a sequential, state-aware intervention process. RKI organizes knowledge into graph-structured units that align diagnostic evidence, target knowledge, and intervention records across rounds. At each round, diagnostic evaluation first localizes the model’s knowledge weaknesses, while diagnostic evidence and intervention history are used to estimate the model’s operational knowledgestate profile. The weakness profile identifies what needs improvement, while the state estimate provides additional information for deciding how each weakness should be addressed. RKI then applies the selected intervention, re-evaluates the updated model over the same knowledge units, and revises both the weakness profile and state estimate before selecting the next intervention. This closed-loop process allows RKI to adapt subsequent interventions as the model’s knowledge state evolves across rounds.

We evaluate RKI on the public multi-hop knowledge benchmark HOTPOTEVAL and the enterprise safety and compliance benchmark SAFEINTERNALEVAL. RKI achieves the best performance on both benchmarks, improving the average score across them by 18.41 points over the strongest baseline. More importantly, our experiments identify two phenomena underlying these gains. First, the same localized failure can correspond to different patterns of knowledge access, and conditioning intervention selection on the operational knowledge state further improves RKI by 5.17 points over Weakness Only under the same weakness information. Second, even when an application weakness persists after a model update, 57.1% of the corresponding knowledge units show a different recognition–recall pattern, showing that a persistent failure signal does not imply a stable underlying knowledge state. Together, these results support the two core design choices of RKI: using knowledge states to distinguish intervention needs under the same localized weakness, and re-estimating those states after model updates to guide subsequent interventions.

Our contributions are as follows:

• We identify knowledge-state aliasing and characterize when diagnostic resolution is insufficient for intervention selection.

• We propose Recursive Knowledge Improvement (RKI), which couples state-aware intervention selection with recursive knowledge-state re-estimation.

• We empirically show that operational knowledge states provide additional information for intervention selection beyond weakness localization, and that these states can change across model updates even when the observed weakness persists.

## 2 Related Work

## 2.1 Knowledge Incorporation for Large Language Models

Prior work has extensively explored incorporating external knowledge into large language models. Domain-specific models such as PMC-LLaMA, SaulLM, and HuatuoGPT-II acquire specialized knowledge through pre-training and instruction tuning (Gururangan et al., 2020; Wu et al., 2023; 2024; Colombo et al., 2024; Chen et al., 2023), while K-Adapter and knowledge-aligned domain shift tuning provide adapter-based and parameter-efficient adaptation, respectively (Wang et al., 2021; Kawamae, 2025). Other studies focus on constructing richer training data from a given knowledge source. Self-QA converts domain knowledge into instruction data (Zhang & Yang, 2023) and EntiGraph leverages entity relations to generate synthetic text for continued pre-training (Yang et al., 2025b). AdaptLLM and Instruction Pre-Training likewise derive training data from raw corpora (Cheng et al., 2024b;a). A complementary line edits parametric knowledge (Mitchell et al., 2021; Meng et al., 2022a;b; Zhang et al., 2024; Li et al., 2026) or post-processes black-box outputs (Song et al., 2025), rather than recursively selecting training interventions. Despite these advances, knowledge incorporation is often organized as a predefined pipeline in which data construction and training strategies are specified in advance, with limited use of evaluation feedback to adapt subsequent knowledge updates.

## 2.2 Evaluation-Guided Targeted Model Improvement

Recent work has begun to use model evaluation to guide subsequent updates, making observed weaknesses a signal for targeted improvement. GraphGen uses calibration uncertainty over finegrained knowledge graphs to identify knowledge gaps and prioritize synthetic data generation accordingly (Chen et al., 2025), while SENATOR combines structural entropy with MCTS to locate knowledge-deficient regions and generate targeted data for repair (Wei et al., 2026). Beyond knowledge-specific improvement, AutoDetect links agent-based weakness detection to targeted data augmentation (Cheng et al., 2024c), while EvalTree and CRAFT use hierarchical or rubric-level evaluation for targeted fine-tuning (Zeng et al., 2025; Gupta et al., 2026). More broadly, R-Select uses validation feedback for data selection (Gao et al., 2026), while the weakness-guided methods above determine what should be improved, often at increasingly fine levels of diagnostic granularity. However, greater diagnostic resolution does not necessarily imply intervention sufficiency: localizing a weakness may still leave the choice among candidate interventions underdetermined.

## 2.3 Knowledge States in Large Language Models

Knowledge-boundary research provides taxonomies and discovery methods (Li et al., 2025b; Wang & Lu, 2026). More generally, knowledge in large language models is not always well described by a binary distinction between “known” and “unknown”. Kazemnejad et al. (2023) distinguish knowledge acquisition from utilization, showing that acquired knowledge may still fail to support downstream tasks, while Gekhman et al. (2024) link fine-tuning on new knowledge to hallucinations. Gekhman et al. (2025) distinguish internally encoded knowledge from knowledge expressed in model outputs, while Calderon et al. (2026) separate missing knowledge from different levels of recall accessibility. Li et al. (2025a) further distinguish task-relevant knowledge retrieval from subsequent reasoning. These studies show that model knowledge cannot be adequately characterized by answer correctness alone: relevant knowledge may differ in whether it has been acquired, can be reliably accessed, and can be effectively used in downstream reasoning or application. However, prior work primarily studies these distinctions as properties to be measured or diagnosed, with limited attention to their role in intervention selection. RKI instead treats the operational knowledge state as intervention-relevant information: it uses the estimated state to distinguish among candidate interventions for the same localized weakness, and re-estimates that state after each model update to adapt subsequent intervention decisions.

## 3 Knowledge States for Knowledge Improvement

Given a domain knowledge base K and a model $M _ { t }$ at round t, we consider a set of knowledge units U that provides a common basis for diagnostic evaluation and intervention-data generation. Diagnostic evaluation produces observations over these knowledge units, which we collectively denote by $\bar { R _ { t } }$ . From $R _ { t } ,$ we derive a weakness profile

$$
W _ { t } = \Lambda ( R _ { t } ) ,\tag{1}
$$

where $W _ { t }$ identifies the knowledge units on which the model exhibits deficiencies and thus localizes what currently requires improvement. We next introduce the operational knowledge state to character-

![](images/a3fce413ac3736ceb403813b7003796589db4e50816fbed2a5ab576b7b474df2.jpg)  
Figure 2: Overview of RKI. RKI aligns diagnosis and intervention over graph-structured knowledge units and recursively adapts interventions to the model’s evolving operational knowledge state.

ize the knowledge conditions underlying these weaknesses and examine how knowledge interventions alter this state.

## 3.1 Operational Knowledge State

A weakness profile captures observable deficiencies revealed by diagnostic evaluation, but does not fully characterize the model’s current knowledge condition. The model’s operational knowledge state at round t, denoted by $Z _ { t } ,$ is a model-level profile estimated from the collective diagnostic evidence over U. It summarizes the knowledge conditions relevant to subsequent intervention, including whether the relevant knowledge is available, can be reliably accessed, and can be effectively used for downstream understanding, reasoning, or application.

Because similar observable behavior may arise from different underlying knowledge conditions, the operational knowledge state cannot be identified from a single evaluation outcome alone. We abstract the state-estimation process as

$$
Z _ { t } = \Phi ( R _ { t } , H _ { t } ) ,\tag{2}
$$

where $H _ { t }$ summarizes the history available up to round $t ,$ including prior knowledge exposure, diagnostic results, and interventions, and Φ denotes the state-estimation function.

Thus, W<sub>t</sub> and $Z _ { t }$ provide complementary information at different granularities. The weakness profile $W _ { t }$ localizes the knowledge units that currently require improvement, while the model-level operational knowledge state $Z _ { t }$ characterizes the model’s overall knowledge condition and provides additional information for deciding how these targets should be addressed.

## 3.2 Knowledge Intervention and State Transition

Let A<sub>t</sub> denote the set of knowledge interventions applied at round t, which together specify the knowledge to be updated and the corresponding update strategy for each intervention target. Applying A<sub>t</sub> transforms the current model as

$$
M _ { t + 1 } = \mathcal { F } ( M _ { t } , A _ { t } ) ,\tag{3}
$$

where F denotes the model-update procedure.

A knowledge intervention may alter the model’s operational knowledge state even when the observed weakness has not yet disappeared. For example, previously unavailable knowledge may become accessible after an update, while the limiting factor shifts from availability to reliable access, understanding,

or application. Each model update therefore induces a corresponding state transition:

$$
Z _ { t } \xrightarrow { A _ { t } } Z _ { t + 1 } .\tag{4}
$$

This transition makes knowledge improvement inherently sequential. An intervention appropriate for $Z _ { t }$ need not remain appropriate once the model reaches $\dot { Z } _ { t + 1 } ,$ even if the corresponding weakness remains observable. Continuing to rely on the previous state estimate after an update may therefore base subsequent interventions on stale information. Each intervention should consequently be followed by re-evaluation of the updated model and re-estimation of its current operational knowledge state.

## 3.3 Knowledge-State Aliasing

The preceding discussion raises a key question: does the weakness profile obtained from diagnostic evaluation contain sufficient information for intervention selection? When the same observed weakness is compatible with distinct operational knowledge states, conditioning on $W _ { t }$ alone obscures state differences that may affect the appropriate intervention. We refer to this ambiguity as knowledge-state aliasing.

For an observed weakness w, we define the set of compatible operational knowledge states as

$$
\Gamma ( w ) = \{ \Phi ( R , H ) | \Lambda ( R ) = w \} .\tag{5}
$$

Definition 1 (Knowledge-State Aliasing). For an observed weakness w, knowledge-state aliasing occurs if

$$
| \Gamma ( w ) | > 1 ,\tag{6}
$$

that is, if the same observed weakness is compatible with multiple distinct operational knowledge states.

Knowledge-state aliasing alone does not imply different intervention requirements. It becomes consequential for intervention selection when states compatible with the same weakness have disjoint sets of optimal interventions. Let A denote the intervention space, and let $Q ( z , a )$ denote the expected knowledge improvement from applying intervention $a \in { \mathcal { A } }$ in operational knowledge state z. The set of optimal interventions is

$$
{ \mathcal { A } } ^ { \star } ( z ) = \arg \operatorname* { m a x } _ { a \in { \mathcal { A } } } Q ( z , a ) .\tag{7}
$$

Proposition 2 (Insufficiency of Weakness-Only Intervention). Suppose an observed weakness w admits two compatible operational knowledge states $z , z ^ { \prime } \in \Gamma ( w )$ such that

$$
\mathcal { A } ^ { \star } ( z ) \cap \mathcal { A } ^ { \star } ( z ^ { \prime } ) = \emptyset .\tag{8}
$$

Then no intervention rule π conditioned only on w can be optimal for both z and $z ^ { \prime } .$

The result follows directly because a weakness-only rule assigns the same intervention to both states, while their optimal intervention sets are disjoint. It formalizes the gap between diagnostic resolution and intervention sufficiency: even a precisely localized weakness may remain insufficient for intervention selection when compatible operational knowledge states favor different interventions. Thus, the operational knowledge state provides intervention-relevant information beyond the weakness profile alone. A formal proof is provided in Appendix A.

## 4 Recursive Knowledge Improvement

As established in Section 3, a localized weakness may be insufficient for intervention selection, while the operational knowledge state underlying that weakness can change after model updating. We instantiate these observations in Recursive Knowledge Improvement (RKI), which alternates between diagnostic evaluation, state estimation, targeted intervention, and re-evaluation (Figure 2).

## 4.1 Framework Overview

Given a domain knowledge base K and a model $M _ { t }$ at round t, RKI first constructs a knowledge graph G from K and extracts a set of graph-structured knowledge units U. These units provide a common basis for diagnostic evaluation and intervention-data generation. Diagnostic evaluation produces observations $\check { R _ { t } }$ over the knowledge units, from which RKI derives a weakness profile $W _ { t }$ and, together with the intervention history ${ \cal H } _ { t } ,$ estimates a model-level operational knowledge state $Z _ { t }$ . The weakness profile determines which knowledge units are intervention targets, while the global state profile informs the intervention strategy applied to those targets. RKI then assigns interventions to the identified targets, updates the model, and repeats the same process on the updated model:

Table 1: Statistics of the datasets used in our experiments. The corpus serves as the knowledge source for graph construction, diagnostic evaluation, and knowledge incorporation, while the test set is used for closed-book evaluation.
<table><tr><td>Dataset</td><td>Domain</td><td>Language</td><td colspan="2">Samples</td><td colspan="2">Avg. Tokens</td><td colspan="2">Max. Tokens</td></tr><tr><td></td><td></td><td></td><td>Corpus</td><td>Test</td><td>Corpus</td><td>Test</td><td>Corpus</td><td>Test</td></tr><tr><td>HOTPOTEVAL</td><td>General</td><td>English</td><td>9,990</td><td>1,000</td><td>137.5</td><td>21.3</td><td>1,984</td><td>74</td></tr><tr><td>SAFEINTERNALEVAL</td><td>Security &amp; Compliance</td><td>Chinese</td><td>1,000</td><td>600</td><td>352.4</td><td>41.2</td><td>701</td><td>139</td></tr></table>

$$
\begin{array} { r l } & { ( M _ { t } , K )  R _ { t }  ( W _ { t } , Z _ { t } )  A _ { t } } \\ & { ~  M _ { t + 1 }  R _ { t + 1 }  ( W _ { t + 1 } , Z _ { t + 1 } )  \cdots . } \end{array}\tag{9}
$$

As summarized in Eq. (9), RKI does not treat a diagnosis as fixed across model updates. Each intervention is followed by re-evaluation, and subsequent intervention decisions are made from the updated weakness and state estimates.

## 4.2 Graph-Structured Knowledge Units

RKI organizes the knowledge in $\mathcal { G }$ into a collection of local graph structures

$$
{ \mathcal { U } } = \{ u _ { 1 } , u _ { 2 } , \ldots , u _ { N } \} ,\tag{10}
$$

which serve as the common knowledge objects for diagnosis, intervention, and re-evaluation. We instantiate $\mathcal { U }$ in Eq. (10) with the six types summarized in Table 5. The same unit can be used to construct diagnostic probes, generate intervention data, and re-evaluate the updated model, providing a shared address space that aligns diagnosis and intervention across rounds.

## 4.3 Operational Knowledge-State Estimation

RKI characterizes the operational knowledge state in terms of three aspects: availability, accessibility, and usability. To make these aspects operationally measurable, we instantiate them with six diagnostic dimensions: acquisition, recognition, recall, understanding, reasoning, and application, as summarized in Table 6.

Availability is estimated from acquisition-related diagnostic evidence, recognition and recall outcomes, together with the known intervention history that records which knowledge has been explicitly introduced in previous rounds. Accessibility is characterized primarily by recognition and recall, while usability is characterized by understanding, reasoning, and application. Together, these signals provide complementary evidence about the model’s current operational knowledge condition.

At round t, let $R _ { t }$ denote the diagnostic evidence collected over $u ,$ and let $H _ { t }$ summarize the history of the recursive improvement process up to round $t ,$ including known knowledge exposure, previous diagnostic results, state estimates, and intervention records. The operational knowledge-state profile is then estimated as

$$
\begin{array} { r l } & { Z _ { t } = \Phi ( R _ { t } , H _ { t } ) } \\ & { \quad = \left( z _ { t } ^ { \mathrm { a c q } } , z _ { t } ^ { \mathrm { r e c o g } } , z _ { t } ^ { \mathrm { r e c a l l } } , z _ { t } ^ { \mathrm { u n d e r } } , z _ { t } ^ { \mathrm { r e a s o n } } , z _ { t } ^ { \mathrm { a p p l y } } \right) . } \end{array}\tag{11}
$$

Each component in Eq. (11) summarizes the corresponding diagnostic dimension over the current knowledge space.

## 4.4 State-Aware Intervention and Recursive Re-estimation

Given $\left( W _ { t } , Z _ { t } \right)$ , RKI treats each knowledge unit identified by $W _ { t }$ as an intervention target. The target unit specifies which knowledge should be improved, while the model-level state $Z _ { t }$ informs the intervention strategy applied to that target. For each weak knowledge unit $u \in W _ { t }$ , we write the selected intervention as

$$
a _ { t } ( u ) = \Psi ( u , Z _ { t } ) , \qquad u \in W _ { t } ,\tag{12}
$$

and collect the resulting interventions as

$$
A _ { t } = \{ a _ { t } ( u ) \mid u \in W _ { t } \} .\tag{13}
$$

Table 2: Distribution of graph-structured knowledge units constructed from SAFEINTERNALEVAL.
<table><tr><td>Unit Type</td><td>Units</td><td>Proportion</td></tr><tr><td>Relation</td><td>14,439</td><td>44.45%</td></tr><tr><td>Ego</td><td>1,340</td><td>4.12%</td></tr><tr><td>Contrast</td><td>1,066</td><td>3.28%</td></tr><tr><td>Cycle</td><td>975</td><td>3.00%</td></tr><tr><td>Chain</td><td>13,494</td><td>41.54%</td></tr><tr><td>Tree</td><td>1,171</td><td>3.60%</td></tr><tr><td>Total</td><td>32,485</td><td>100.00%</td></tr></table>

Here Ψ denotes the intervention-selection procedure, which conditions the choice of strategy on the target unit and the model-level state.

We consider intervention families corresponding to different operational bottlenecks. Availabilityoriented interventions increase exposure to the target knowledge, for example through continued pre-training or structured knowledge exposure. Accessibility-oriented interventions emphasize reliable retrieval, for example through recall-focused supervision. Usability-oriented interventions provide examples that require the target knowledge to be understood, reasoned over, or applied. These intervention families define a set of candidate strategies and do not impose a fixed mapping from knowledge states to training methods. The state estimate is used to identify which type of intervention is more appropriate under the current model condition.

Applying the selected intervention produces

$$
M _ { t + 1 } = \mathcal { F } ( M _ { t } , A _ { t } ) ,\tag{14}
$$

After the update in Eq. (14), RKI re-evaluates $M _ { t + 1 }$ over the same knowledge units to obtain $R _ { t + 1 }$ and recompute $( W _ { t + 1 } , Z _ { t + 1 } )$ . The next intervention is therefore selected from the updated diagnosis. In this way, RKI adapts intervention decisions as the model’s operational knowledge state evolves across rounds.

## 5 Experiments

## 5.1 Experimental Setup

Benchmarks and Knowledge Sources. We evaluate RKI in two knowledge-intensive settings. For the public benchmark, we sample from HOTPOTEVAL released with GraphGen (Chen et al., 2025), retaining instances that can be aligned with their source corpus. HOTPOTEVAL is adapted from HotpotQA (Yang et al., 2018) and contains general-domain multi-hop questions over multiple supporting facts. It provides a source corpus for knowledge incorporation and a QA test set for closed-book evaluation. We also construct SAFEINTERNALEVAL from enterprise safety and compliance documents. Its evaluation questions are manually constructed and reviewed against the source documents, and are designed to avoid direct overlap with the knowledge units used for intervention-data generation. Dataset statistics are summarized in Table 1, with additional details and examples provided in Appendix E.3.

Implementation. We use Qwen3-4B-Instruct-2507 (Yang et al., 2025a) as the common backbone model and DeepSeek-V4-Pro (Xu et al., 2026) for knowledge graph construction, evaluation-instance generation, and data filtering. From each source corpus, we construct a knowledge graph and extract graphstructured knowledge units that support diagnosis, weakness localization, and intervention-data generation. Table 2 reports the 32,485 units extracted from SAFEINTERNALEVAL. These counts reflect structural knowledge units in the graph and do not directly correspond to the number of evaluation or training instances. We further filter the units by question-generation suitability and knowledge value before constructing diagnostic and knowledge-improvement data.

Comparison Methods. Following the baseline setting of GraphGen (Chen et al., 2025), we compare RKI with GENIE (Yehudai et al., 2024), LongForm (Köksal et al., 2024), WRAP (Maini et al., 2024), Self-QA (Zhang & Yang, 2023), EntiGraph (Yang et al., 2025b), and GraphGen. All methods use the same source corpora, backbone model, and training configurations for a consistent comparison. Training details are provided in Appendix E.1. We do not include SENATOR (Wei et al., 2026) in the main comparison because its deficiency detection assumes an external, curated domain knowledge graph, whereas our evaluation uses only the source corpus of each benchmark as the shared knowledge source. Applying SENATOR here would require replacing its external graph with an automatically constructed one, resulting in a substantially different experimental setting.

![](images/150885d0141ca5bcd2bb0473b7c0ba7d7913d229f3cb39776f9186ae36b70615.jpg)

![](images/aeb7ba680d81ea5f04faebba6b44fdba425add43378fa681a518e27a20782f36.jpg)  
Figure 3: Knowledge-state aliasing under the same localized failure signal. Left: recognition outcomes for recall-failure instances. Right: recognition and recall patterns for understanding-, reasoning-, and application-failure units.

Evaluation. For single-choice recognition probes, we rotate answer positions across four variants and control option lengths to reduce position and length biases. For fill-in-the-blank recall probes, we perform three independent random samples to reduce sampling variance. For open-ended evaluation, we use rubric-based LLM judgments as the primary evaluation score. We do not use lexical-overlap metrics such as token-F1 or ROUGE-F as primary measures because they mainly capture surface-form similarity and can be sensitive to response length and paraphrasing, which may not reliably reflect semantic correctness in open-ended knowledge evaluation. The full judging protocol is provided in Appendix E.4. We further validate the LLM judge against human annotations, with agreement results reported in Appendix E.5. Inference settings are provided in Appendix E.2.

## 5.2 Knowledge-State Aliasing

We first test whether the same localized weakness can correspond to different operational knowledge states. For instances sharing the same recall, understanding, reasoning, or application failure, we compare recognition and recall performance to examine whether a single weakness label conflates distinct patterns of knowledge access.

As shown in Figure 3, among instances with recall failure, 37.8% still answer the corresponding recognition probe correctly, while 62.2% also fail recognition. Similar heterogeneity appears for higher-order failures. Among units with application failure, 29.1% fail both recognition and recall on the measured associated facts, while 13.3% recall all of them. The corresponding proportions are 18.9% and 3.6% for understanding failure, and 36.6% and 5.3% for reasoning failure.

These results show that the same localized failure can correspond to different patterns of knowledge access. Recognition and recall evidence further separates operational knowledge-state profiles under the same failure signal, providing empirical support for knowledge-state aliasing. Here, complete recall refers only to the measured associated facts and does not imply recall of all knowledge required by the higher-order task.

## 5.3 Overall Knowledge Improvement

As shown in Table 3, RKI achieves the best performance on both knowledge-intensive benchmarks. On HOTPOTEVAL, RKI reaches an average score of 32.70, outperforming the strongest baseline, Self-QA, by 5.90 points. This result shows that RKI remains effective in a general-domain multi-hop setting where answering questions requires integrating multiple supporting facts.

The improvement is more pronounced on SAFEINTERNALEVAL, where RKI achieves 58.92, compared with 28.00 for Self-QA and 22.33 for GraphGen. Averaged across the two benchmarks, RKI obtains 45.81, which is 18.41 points above the strongest baseline average of 27.40. Overall, these results show that RKI consistently improves knowledge incorporation across both public and enterprise-internal settings. The analyses that follow further examine whether these gains arise from state-aware intervention selection and recursive knowledge-state re-estimation.

## 5.4 Recursive Knowledge-State Re-estimation

We further examine whether the operational knowledge state can change even when an observed weakness persists after a model update. We select knowledge units whose associated application questions remain failed both before and after the first intervention. As shown in Figure 4, 57.1% of these units show a different recognition–recall pattern after the intervention despite the persistent application weakness. In particular, the proportion of units that fully recall the measured associated facts increases from 12.2% to 40.9%.

Table 3: Performance of different knowledge incorporation methods. Bold marks the best result.
<table><tr><td>Method</td><td>HOTPOTEVAL</td><td>SAFEINTERNALEVAL Avg.</td></tr><tr><td>GENIE 23.00</td><td>9.75</td><td>16.38</td></tr><tr><td>LongForm</td><td>24.40</td><td>11.17 17.79</td></tr><tr><td>WRAP</td><td>22.90</td><td>9.75 16.33</td></tr><tr><td>Self-QA</td><td>26.80</td><td>28.00 27.40</td></tr><tr><td>EntiGraph</td><td>26.50</td><td>21.50 24.00</td></tr><tr><td>GraphGen</td><td>23.70</td><td>22.33 23.02</td></tr><tr><td>RKI (Ours)</td><td>32.70</td><td>58.92 45.81</td></tr></table>

![](images/0334c96c421803ad54b310cc5f191d1dc88549c1230ac0418fcf7eb685765b1d.jpg)  
Figure 4: Operational knowledge-state transitions under a persistent application weakness.

![](images/0e7d02fac61e2c33870decddd4a2a2ff72d4c115d7dfe22bbc598fa27caaa0f2.jpg)  
Figure 5: Evolution of the operational knowledge state and overall performance across recursive RKI updates.

These results show that a persistent observed weakness does not imply a stable operational knowledge state. Reusing the previous intervention based only on the unchanged failure signal may therefore rely on stale state information after the model has been updated. This directly motivates recursive knowledge-state re-estimation: after each intervention, RKI re-evaluates the updated model, collects new diagnostic evidence, and revises its estimate of the current operational knowledge state before selecting the next intervention.

We next examine how the operational knowledge-state profile evolves across RKI updates. Figure 5 reports the six diagnostic dimensions together with overall performance at each phase. Phase 0 denotes the base model, while the reference line marks the score of 50.50 obtained by direct training on the full intervention data. The overall score increases from 17.41 at Phase 0 to 31.50, 51.03, and 58.92 over successive updates. The final phase is 8.42 points above the full-data training reference, showing that recursive state-aware intervention yields stronger overall performance than direct full-data training in this setting.

By construction, acquisition reaches 1 after the first intervention because all target facts are included in the knowledge injection. This indicates complete exposure to the target knowledge and does not imply reliable recognition, recall, or use. The remaining dimensions therefore continue to evolve across subsequent updates. From Phase 2 to Phase 3, recall increases from 37.1 to 38.9, while understanding rises from 62.1 to 69.6 and application from 56.4 to 69.2. Thus, different dimensions of the operational knowledge state need not improve at the same rate. Together with the recognition–recall changes observed under persistent application weakness, these results support re-evaluating the model after each intervention, re-estimating its current operational knowledge state, and adapting subsequent interventions accordingly.

## 5.5 Ablation and Analysis

To isolate the contribution of state-aware intervention selection and recursive state re-estimation, we conduct an ablation study on SAFEINTERNALEVAL. Random Intervention, Fixed Access, Fixed Utilization, Weakness Only, and RKI follow the same four-phase update schedule. Fixed Access and Fixed Utilization apply the same intervention type to all targeted knowledge units, while Weakness Only selects interventions based on the localized weakness $W _ { t }$ without conditioning on the operational knowledge state $Z _ { t } .$ As an additional reference, Full-Data Training directly trains once on the complete intervention dataset without selective intervention assignment. Frozen State instead estimates the operational knowledge state once before the first intervention and reuses this estimate in all subsequent rounds, thereby removing recursive re-estimation while retaining state-aware intervention selection.

Table 4: Ablation study on intervention selection on SAFEINTERNALEVAL. Bold marks the best result.
<table><tr><td>Method</td><td>Average Score</td></tr><tr><td>Random Intervention</td><td>29.33</td></tr><tr><td>Fixed Access</td><td>23.17</td></tr><tr><td>Fixed Utilization</td><td>48.00</td></tr><tr><td>Full-Data Training</td><td>50.50</td></tr><tr><td>Weakness Only</td><td>53.75</td></tr><tr><td>Frozen State</td><td>39.00</td></tr><tr><td>RKI (Ours)</td><td>58.92</td></tr></table>

As shown in Table 4, Fixed Utilization (48.00) outperforms Fixed Access (23.17), consistent with the larger share of utilization-oriented items in SAFEINTERNALEVAL: improving access alone does not support understanding, reasoning, or application. Random Intervention reaches 29.33, remaining below the informed strategies.

Full-Data Training reaches 50.50, while Weakness Only reaches 53.75, exceeding both full-data training and the strongest fixed policy. Weakness localization therefore helps assign intervention data, and simply using all available data is insufficient.

Frozen State drops to 39.00, below both RKI and Weakness Only. Stale state estimates can therefore be less useful than ignoring the state, directly supporting recursive re-estimation after each update.

RKI achieves 58.92, 5.17 points above Weakness Only and 8.42 above Full-Data Training. Since both RKI and Weakness Only use the same localized weakness, this gain supports the additional value of state-aware intervention selection.

## 6 Conclusion

This work examines a limitation of evaluation-guided knowledge improvement: localizing a model’s knowledge weaknesses does not necessarily provide sufficient information for deciding how those weaknesses should be addressed. We characterize this ambiguity as knowledge-state aliasing, where the same observed weakness can correspond to different operational knowledge states, and show that weakness-only intervention selection can be insufficient when these states favor different interventions. Building on this observation, we propose Recursive Knowledge Improvement (RKI), which uses graphstructured knowledge units to align diagnosis, intervention, and re-evaluation, selects interventions based on the model’s current operational knowledge state, and re-estimates that state after each update.

On HOTPOTEVAL and SAFEINTERNALEVAL, RKI improves the average score across the two benchmarks by 18.41 points over the strongest baseline. Further analyses show that operational knowledge states provide information for intervention selection beyond weakness localization, and that these states can change after model updates even when the observed weakness persists. Taken together, these results suggest that knowledge improvement is better treated as a closed-loop process, with interventions continuously adapted to the model’s evolving knowledge state.

## 7 Limitation

This work has two limitations. First, the operational knowledge state is estimated from diagnostic evidence and intervention history. It is intended as an actionable representation for intervention decisions, not as a recovery of the model’s true latent knowledge state. The current state representation should therefore be viewed as a diagnostic abstraction for knowledge improvement, whose effectiveness depends on the chosen diagnostic dimensions and estimation procedure. Second, RKI currently performs state-aware intervention selection over a predefined set of candidate interventions and re-evaluates the model after each update to re-estimate its knowledge state, which introduces additional evaluation and training costs. Future work could explore more flexible state representations and intervention spaces, as well as more efficient recursive knowledge improvement while preserving intervention effectiveness.

## References

Nitay Calderon, Eyal Ben-David, Zorik Gekhman, Eran Ofek, and Gal Yona. Empty shelves or lost keys? recall is the bottleneck for parametric factuality. arXiv preprint arXiv:2602.14080, 2026.

Junying Chen, Xidong Wang, Ke Ji, Anningzhe Gao, Feng Jiang, Shunian Chen, Hongbo Zhang, Dingjie Song, Wenya Xie, Chuyi Kong, et al. Huatuogpt-ii, one-stage training for medical adaption of llms. arXiv preprint arXiv:2311.09774, 2023.

Zihong Chen, Wanli Jiang, Jinzhe Li, Zhonghang Yuan, Huanjun Kong, Wanli Ouyang, and Nanqing Dong. Graphgen: enhancing supervised fine-tuning for llms with knowledge-driven synthetic data generation. arXiv preprint arXiv:2505.20416, 2025.

Daixuan Cheng, Yuxian Gu, Shaohan Huang, Junyu Bi, Minlie Huang, and Furu Wei. Instruction pretraining: Language models are supervised multitask learners. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 2529–2550, 2024a.

Daixuan Cheng, Shaohan Huang, and Furu Wei. Adapting large language models via reading comprehension. In International Conference on Learning Representations, volume 2024, pp. 48624–48652, 2024b.

Jiale Cheng, Yida Lu, Xiaotao Gu, Pei Ke, Xiao Liu, Yuxiao Dong, Hongning Wang, Jie Tang, and Minlie Huang. Autodetect: Towards a unified framework for automated weakness detection in large language models. In Findings of the Association for Computational Linguistics: EMNLP 2024, pp. 6786–6803, 2024c.

Pierre Colombo, Telmo Pessoa Pires, Malik Boudiaf, Dominic Culver, Rui Melo, Caio Corro, Andre FT Martins, Fabrizio Esposito, Vera Lúcia Raposo, Sofia Morgado, et al. Saullm-7b: A pioneering large language model for law. arXiv preprint arXiv:2403.03883, 2024.

Xin Gao, Xiaoyang Wang, Yun Zhu, Zheng Liu, Conghui He, and Lijun Wu. R-select: A robust multimetric data selection approach for fine-tuning large language models. In Proceedings ofthe 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2, pp. 1262–1273, 2026.

Zorik Gekhman, Gal Yona, Roee Aharoni, Matan Eyal, Amir Feder, Roi Reichart, and Jonathan Herzig. Does fine-tuning llms on new knowledge encourage hallucinations? In Proceedings of the 2024 conference on empirical methods in natural language processing, pp. 7765–7784, 2024.

Zorik Gekhman, Eyal Ben David, Hadas Orgad, Eran Ofek, Yonatan Belinkov, Idan Szpektor, Jonathan Herzig, and Roi Reichart. Inside-out: Hidden factual knowledge in llms. arXiv preprint arXiv:2503.15299, 2025.

Vipul Gupta, Zihao Wang, Razvan-Gabriel Dumitru, MohammadHossein Rezaei, Aakash Sabharwal, and Yunzhong He. Craft: Clustering rubrics to diagnose weak llm capabilities and generate targeted fine-tuning data. arXiv preprint arXiv:2607.16122, 2026.

Suchin Gururangan, Ana Marasovi´c, Swabha Swayamdipta, Kyle Lo, Iz Beltagy, Doug Downey, and Noah A Smith. Don’t stop pretraining: Adapt language models to domains and tasks. In Proceedings of the 58th annual meeting of the association for computational linguistics, pp. 8342–8360, 2020.

Noriaki Kawamae. Knowledge-aligned domain shift tuning for efficient adaptation in large language models. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2, pp. 1128–1138, 2025.

Amirhossein Kazemnejad, Mehdi Rezagholizadeh, Prasanna Parthasarathi, and Sarath Chandar. Measuring the knowledge acquisition-utilization gap in pretrained language models. In Findings of the Associationfor Computational Linguistics: EMNLP 2023, pp. 4305–4319, 2023.

Abdullatif Köksal, Timo Schick, Anna Korhonen, and Hinrich Schütze. Longform: Effective instruction tuning with reverse instructions. In Findings of the Association for Computational Linguistics: EMNLP 2024, pp. 7056–7078, 2024.

Alan Li, Yixin Liu, Arpan Sarkar, Doug Downey, and Arman Cohan. Demystifying scientific problemsolving in llms by probing knowledge and reasoning. arXiv preprint arXiv:2508.19202, 2025a.

Moxin Li, Yong Zhao, Wenxuan Zhang, Shuaiyi Li, Wenya Xie, See Kiong Ng, Tat-Seng Chua, and Yang Deng. Knowledge boundary of large language models: A survey. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 5131–5157, 2025b.

Xiaopeng Li, Shasha Li, Xi Wang, Shezheng Song, Bin Ji, Shangwen Wang, Jun Ma, Xiaodong Liu, Mina Liu, and Jie Yu. Emsedit: Efficient multi-step meta-learning-based model editing. In Proceedings of the ACM Web Conference 2026, pp. 3847–3858, 2026.

Pratyush Maini, Skyler Seto, Richard Bai, David Grangier, Yizhe Zhang, and Navdeep Jaitly. Rephrasing the web: A recipe for compute and data-efficient language modeling. In Proceedings ofthe 62nd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 14044–14072, 2024.

Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing factual associations in gpt. Advances in neural information processing systems, 35:17359–17372, 2022a.

Kevin Meng, Arnab Sen Sharma, Alex Andonian, Yonatan Belinkov, and David Bau. Mass-editing memory in a transformer. arXiv preprint arXiv:2210.07229, 2022b.

Eric Mitchell, Charles Lin, Antoine Bosselut, Chelsea Finn, and Christopher D Manning. Fast model editing at scale. arXiv preprint arXiv:2110.11309, 2021.

Xiaoshuai Song, Zhengyang Wang, Keqing He, Guanting Dong, Yutao Mou, Jinxu Zhao, and Weiran Xu.

Assessing and post-processing black box large language models for knowledge editing. In Proceedings of the ACM on Web Conference 2025, pp. 1716–1732, 2025.

Ruize Wang, Duyu Tang, Nan Duan, Zhongyu Wei, Xuan-Jing Huang, Jianshu Ji, Guihong Cao, Daxin Jiang, and Ming Zhou. K-adapter: Infusing knowledge into pre-trained models with adapters. In Findings ofthe Associationfor Computational Linguistics: ACL-IJCNLP 2021, pp. 1405–1418, 2021.

Ziquan Wang and Zhongqi Lu. Knowledge boundary discovery for large language models. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 40, pp. 33791–33799, 2026.

Yifan Wei, Xiaoyan Yu, Tengfei Pan, Angsheng Li, and Li Du. Structural entropy guided agent for detecting and repairing knowledge deficiencies in llms. Advances in Neural Information Processing Systems, 38:3346–3372, 2026.

Chaoyi Wu, Weixiong Lin, Xiaoman Zhang, Ya Zhang, Weidi Xie, and Yanfeng Wang. Pmc-llama: toward building open-source language models for medicine. Journal of the American Medical Informatics Association, 31(9):1833–1843, 2024.

Shijie Wu, Ozan Irsoy, Steven Lu, Vadim Dabravolski, Mark Dredze, Sebastian Gehrmann, Prabhanjan Kambadur, David Rosenberg, and Gideon Mann. Bloomberggpt: A large language model for finance. arXiv preprint arXiv:2303.17564, 2023.

Anyi Xu, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, Chenchen Ling, et al. Deepseek-v4: Towards highly efficient million-token context intelligence. arXiv preprint arXiv:2606.19348, 2026.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025a.

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. In Proceedings of the 2018 conference on empirical methods in natural language processing, pp. 2369–2380, 2018.

Zitong Yang, Neil Band, Shuangping Li, Emmanuel Candes, and Tatsunori Hashimoto. Synthetic continued pretraining. In International Conference on Learning Representations, volume 2025, pp. 44379–44421, 2025b.

Asaf Yehudai, Boaz Carmeli, Yosi Mass, Ofir Arviv, Nathaniel Mills, Eyal Shnarch, and Leshem Choshen. Achieving human parity in content-grounded datasets generation. In International Conference on Learning Representations, volume 2024, pp. 38579–38594, 2024.

Zhiyuan Zeng, Yizhong Wang, Hannaneh Hajishirzi, and Pang Wei Koh. Evaltree: Profiling language model weaknesses via hierarchical capability trees. arXiv preprint arXiv:2503.08893, 2025.

Ningyu Zhang, Bozhong Tian, Siyuan Cheng, Xiaozhuan Liang, Yi Hu, Kouying Xue, Yanjie Gou, Xi Chen, and Huajun Chen. Instructedit: Instruction-based knowledge editing for large language models. arXiv preprint arXiv:2402.16123, 2024.

Xuanyu Zhang and Qing Yang. Self-qa: Unsupervised knowledge guided language model alignment. arXiv preprint arXiv:2305.11952, 2023.

Table 5: Graph-structured knowledge units in RKI and their functional roles.
<table><tr><td>Unit</td><td>Role</td></tr><tr><td>Relation</td><td>Represents a direct relational fact between two entities as the basic unit of relational</td></tr><tr><td>Contrast</td><td>knowledge. Organizes related facts or local structures along a shared comparison dimension.</td></tr><tr><td>Ego</td><td>Represents the local relational neighborhood centered on a focal entity.</td></tr><tr><td>Cycle</td><td>Represents a closed relational path formed by multiple entities and their relations.</td></tr><tr><td>Chain</td><td>Represents a multi-hop relational path connecting a sequence of entities.</td></tr><tr><td>Tree</td><td>Represents a branching relational structure extending from a shared local context.</td></tr></table>

Table 6: Diagnostic dimensions, operational aspects, and interpretations for knowledge-state estimation.
<table><tr><td>Dimension</td><td>Aspect</td><td>Interpretation</td></tr><tr><td>Acquisition</td><td>Availability</td><td>Whether the relevant knowledge shows evidence of having been incorporated into the model.</td></tr><tr><td>Recognition</td><td>Accessibility</td><td>Whether the model can identify the relevant knowledge when candidate information or relevant cues are provided.</td></tr><tr><td>Recall</td><td>Accessibility</td><td>Whether the model can reliably retrieve the relevant knowledge without explicit candidate cues.</td></tr><tr><td>Understanding</td><td>Usability</td><td>Whether the model can correctly interpret the relations and meaning expressed by the relevant knowledge.</td></tr><tr><td>Reasoning</td><td>Usability</td><td>Whether the model can combine and derive information from the relevant knowledge through reasoning.</td></tr><tr><td>Application</td><td>Usability</td><td>Whether the model can effectively use the relevant knowl- edge in downstream tasks or problem settings.</td></tr></table>

## A Proof of Proposition 2

Proof. Suppose, for contradiction, that there exists an intervention rule $\pi : \mathcal { W } \to \mathcal { A }$ conditioned only on the observed weakness that is optimal for both z and $z ^ { \prime } .$ Unlike the state-aware selector Ψ in Eq. (12), such a rule does not condition on the model-level state. Since $z , z ^ { \prime } \in \Gamma ( w )$ correspond to the same observed weakness w, the rule selects the same intervention

$$
a _ { w } = \pi ( w )\tag{15}
$$

for both states.

For π to be optimal in state $z ,$ we must have

$$
a _ { w } \in \mathcal { A } ^ { \star } ( z ) .\tag{16}
$$

Similarly, optimality in state $z ^ { \prime }$ requires

$$
a _ { w } \in \mathcal { A } ^ { \star } ( z ^ { \prime } ) .\tag{17}
$$

Therefore,

$$
a _ { w } \in \mathcal { A } ^ { \star } ( z ) \cap \mathcal { A } ^ { \star } ( z ^ { \prime } ) .\tag{18}
$$

However, by the assumption of Proposition 2,

$$
\mathcal { A } ^ { \star } ( z ) \cap \mathcal { A } ^ { \star } ( z ^ { \prime } ) = \emptyset ,\tag{19}
$$

which is a contradiction. Hence, no intervention rule conditioned only on w can be optimal for both z and z<sup>′</sup>. □

## B RKI Knowledge Units and Diagnostic Dimensions

We provide detailed definitions of the knowledge units and diagnostic dimensions introduced in Section 4.

## C Prompt Templates

Due to space constraints, we present only the knowledge graph extraction and relation fact extraction prompts in Figures 6 and 7. Our pipeline additionally uses prompts for iterative graph extraction,

![](images/2cb6b643fd088fc161965e76b9bc057782d1289149c139b3f7d10b065f0378b4.jpg)  
Figure 6: Knowledge graph extraction prompt.

relation fact extraction and filtering, diagnostic benchmark generation and filtering, and CPT- and SFTdata generation. The complete set of prompt templates will be made available in the accompanying code release.

The extraction process additionally uses continuation and termination prompts. The continuation prompt requests only entities and relations that remain missing from previous rounds, while the termination prompt determines whether another extraction round is necessary.

The relation fact extraction prompt first evaluates whether a relation is worth testing, then determines which querying directions admit a unique answer.

```erb
P2. Relation Fact Extraction
You are an expert constructing knowledge evaluation benchmarks, working through two
ordered steps: value judgment first, then direction judgment.
# Relation Pair
${relation_block}
# Source Material
${raw_context_block}
# Step 1: Value Judgment
Judge from the source material alone whether the relation between the two entities is
worth testing.
Set valuable to false if:
- the relation is unsupported or semantically indeterminate;
- it contains multiple independent propositions;
- it represents low-value metadata, transient states, or trivia;
- no direction can yield a question with a unique answer.
# Step 2: Direction Judgment
Only when valuable is true, determine the subject/object direction and whether each
direction admits a uniquely answerable question.
Choose exactly one:
- both
- forward
- reverse
- none
# Output Format
If the value judgment fails:
{"valuable": false, "valuable_reason": "..."}
Otherwise:
{"valuable": true,
"valuable_reason": "...
"subject": "...",
"object": "...",
"direction": "both/forward/reverse/none",
"direction_reason": "..."}
```  
Figure 7: Relation fact extraction prompt (P2, abridged). Value judgment precedes the assessment of valid querying directions.

![](images/caedebe5645b41e5f1100c48ab27fbd8e3feba6691ef08f3312a60f141eef953.jpg)  
Figure 8: Local structure of the knowledge graph constructed from the Hotpot corpus. (a) The complete two-hop neighborhood of Star Wars, with 130 entities and 162 edges. Outlined nodes are expanded in (b). (b) The induced subgraph of 15 selected entities and their 25 edges. Colors and markers denote entity types. All edges are undirected.

Table 7: Training configurations for CPT and SFT.
<table><tr><td>Parameter</td><td>CPT</td><td>SFT</td></tr><tr><td>Context Length</td><td>2,048</td><td>2,048</td></tr><tr><td>Global Batch Size</td><td>16</td><td>64</td></tr><tr><td>Epochs</td><td>1</td><td>1</td></tr><tr><td>Peak Learning Rate</td><td>5 × 10−6</td><td>2 × 10−5</td></tr><tr><td>LR Scheduler</td><td>Cosine</td><td>Linear</td></tr><tr><td>Warmup Ratio</td><td>0.05</td><td>0.03</td></tr><tr><td>Weight Decay</td><td>0.01</td><td>0.1</td></tr><tr><td>Optimizer</td><td>AdamW</td><td>AdamW</td></tr><tr><td>Gradient Clipping</td><td>1.0</td><td>1.0</td></tr><tr><td>Precision</td><td>BF16</td><td>BF16</td></tr></table>

## D Knowledge Graph Visualization

Figure 8 illustrates the local structure of the knowledge graph constructed from the Hotpot corpus. Panel (a) shows the complete two-hop neighborhood centered on Star Wars, containing 130 entities and 162 edges. Panel (b) provides a detailed view of 15 selected entities and all 25 edges among them. Node colors and shapes distinguish entity types, illustrating connections among works, people or fictional characters, and organizations.

## E Supplementary Experimental Details

## E.1 Training Configurations

We use separate training configurations for CPT and SFT, as summarized in Table 7.

## E.2 Inference Configurations

We use the same inference configuration across all models and benchmarks, with greedy decoding, a maximum input length of 5,120 tokens, a maximum generation length of 2,048 tokens, and thinking mode disabled.

Table 8: Representative source-corpus and evaluation examples from the public benchmarks.
<table><tr><td>Dataset</td><td>Corpus</td><td>Test</td></tr><tr><td>HOTPOTEVAL</td><td>Bolax is a genus of flowering plants in the Apiaceae, with four or five species, and is endemic to temperate South America. Jovibarba heuffelii, commonly known as Hen-and- chickens, is a plant species native to the Balkans and the</td><td>Question: Which plant has more species, Bolax or Jovibarba? Answer: Bolax.</td></tr></table>

Table 9: Judging rubrics across benchmarks. Each dimension is scored from 0 to 10, with scores of at least 6 considered passing. For the internal benchmark, understanding evaluates association coverage, while reasoning and application evaluate reasoning coverage.
<table><tr><td>Benchmark</td><td>Dimension</td><td>Evaluation Criterion</td></tr><tr><td rowspan="2">HOTPOTEVAL</td><td>final_fact</td><td>Whether the final answer states the same fact as the ref- erence answer.</td></tr><tr><td>completeness</td><td>Whether the answer covers the required constituents and qualifiers of the reference answer.</td></tr><tr><td rowspan="3">SAFEINTERNALEVAL Understanding</td><td>factual_accuracy</td><td>Whether every fact stated in the answer is correct; hallu- cinated facts lose points. How many of the required answer points the answer cov-</td></tr><tr><td>association_coverage</td><td>ers. Paraphrase and superordinate summaries count as coverage; a point mentioned with a wrong association</td></tr><tr><td>conclusion_correctness</td><td>does not. Whether the core judgment (overall positioning, the difference identified, or the misconception clarified) matches the reference, with the correct characterisation basis.</td></tr><tr><td rowspan="3">SAFEINTERNALEVAL Reasoning and Application</td><td>factual_accuracy</td><td>Whether every fact stated in the answer is correct; hallu- cinated facts lose points.</td></tr><tr><td>reasoning_coverage</td><td>How many of the required answer points the answer cov- ers. Coverage must be substantively correct: naming a point while stating its derivation wrongly does not count.</td></tr><tr><td>conclusion_correctness</td><td>Whether the final conclusion matches the reference, with the correct characterisation basis rather than one deriv- able from general knowledge alone.</td></tr></table>

## E.3 Benchmark Details and Examples

Each benchmark consists of a source corpus used for graph construction, diagnostic evaluation, and knowledge incorporation, together with a QA set for closed-book evaluation. For the public benchmark, we sample from the dataset released with GraphGen and retain test instances that can be aligned with their corresponding source texts.

HOTPOTEVAL is derived from HotpotQA (Yang et al., 2018), where answering a question requires combining evidence distributed across multiple supporting passages. We use the corresponding supporting texts as the source corpus and retain aligned questions for evaluation.

Table 8 provides a representative corpus–test pair from HOTPOTEVAL. We additionally evaluate on SAFEINTERNALEVAL, which is constructed from enterprise-internal safety and compliance documents. Its evaluation questions are manually constructed and reviewed against the source documents, and are designed to avoid direct overlap with the knowledge units used for intervention-data generation. Because the underlying documents contain proprietary information, we do not release or display examples from this benchmark.

## E.4 Evaluation and Judging Protocol

For open-ended evaluation, we use DeepSeek-V4-Pro as the LLM judge with temperature set to 0. Each evaluation dimension is scored on a 0–10 scale, with a score of at least 6 considered a pass. We use benchmark-specific evaluation dimensions, as summarized in Table 9. An instance passes the strict criterion only when all dimensions receive scores of at least 6.

Table 10: Agreement between LLM judgments and human annotations on 100 stratified instances per benchmark.
<table><tr><td>Benchmark</td><td>Pass Agreement</td><td>Cohen&#x27;s K</td></tr><tr><td>HOTPOTEVAL</td><td>98%</td><td>0.95</td></tr><tr><td>SAFEINTERNALEVAL</td><td>96%</td><td>0.89</td></tr></table>

## E.5 Agreement Between LLM Judgments and Human Annotations

To validate the LLM judge, we sample 100 instances from each benchmark, stratified by automated evaluation outcomes. Human annotators independently evaluate the sampled instances using the same rubric and pass threshold without access to the LLM judgments. An instance is considered to pass only when all dimensions receive scores of at least 6.

As shown in Table 10, LLM and human judgments achieve pass agreement of 98% and 96% on HOT-POTEVAL and SAFEINTERNALEVAL, with Cohen’s κ of 0.95 and 0.89, respectively. These results indicate consistently high agreement between automatic and human evaluation across both benchmarks.