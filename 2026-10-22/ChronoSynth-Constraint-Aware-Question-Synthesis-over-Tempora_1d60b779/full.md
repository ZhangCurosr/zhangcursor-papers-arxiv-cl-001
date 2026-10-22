# ChronoSynth: Constraint-Aware Question Synthesis over Temporal Knowledge Graphs

Feng Liang<sup>1</sup>, Weixin Zeng<sup>1</sup>, Runhao Zhao<sup>1</sup>, Mingjun Zhou<sup>1</sup>, Shaoshi Yang<sup>2</sup>, Xiang Zhao<sup>1</sup>\*

<sup>1</sup>National Key Laboratory of Big Data and Decision, National University of Defense Technology, China

<sup>2</sup>Beijing University of Posts and Telecommunications, China

fungloeng@gmail.com, {zengweixin13, runhaozhao, zhoumingjun24, xiangzhao}@nudt.edu.cn shaoshi.yang@bupt.edu.cn

## Abstract

Knowledge graph question generation (KGQG) generates natural-language questions from structured graph evidence and a target answer. When the input graph is temporal, a valid question must also preserve the answer field, time condition, reference event, and temporal order that determine the answer set. Existing KGQG methods and prompting baselines can produce fluent questions that alter these conditions, causing temporal semantic drift. This paper studies temporal KGQG as a constraint-preserving graph-to-question problem and proposes ChronoSynth, a memoryaugmented and constraint-aware generation framework. ChronoSynth first constructs a Temporal Intent State to make the intended question semantics explicit. It then retrieves compatible typed question patterns from an indexed pattern memory, adapts them to the input graph, and validates the generated question. Validation feedback guides revisions when a supported inconsistency is detected. Experiments on CRONQUESTIONS and MultiTQ demonstrate state-of-the-art generation perfor mance, with ablations supporting the contributions of its components.

Keywords: temporal knowledge graphs, question generation, synthetic data generation

## 1 Introduction

Knowledge graph question generation (KGQG) aims to generate natural-language questions from structured graph evidence, often conditioned on a local subgraph and a target answer (Chen et al., 2020). Recent database and graph-data research has advanced knowledge-graph querying and temporal graph processing (Arenas et al., 2021; Tang et al., 2025). KGQG supports the construction of question-answer pairs and evaluation benchmarks for graph-based question answering (Xiong et al.,

2022; Guo et al., 2022; Zhao et al., 2024b), and it also helps expand the coverage and diversity of questions over the same graph semantics (Ren et al., 2025).

Temporal knowledge graph question generation (TKGQG) extends KGQG from static triples to time-scoped facts. In a temporal knowledge graph (TKG), a fact may be valid at a timestamp, over an interval, or in relation to another event. This temporal structure changes what a generated question needs to express. Two questions involving the same entities and relation may have different answers because one asks “before 2014”, another asks “after 2014”, or one asks for the “last” event before a reference event. Changing the answer field, reference point, or temporal order can change the answer set even when the wording remains fluent, as illustrated in Figure 1. Temporal knowledge graph question answering (TKGQA) benchmarks such as CRONQUESTIONS and MultiTQ make this dependence explicit through questions over timestamps, intervals, and event ordering (Saxena et al., 2021; Chen et al., 2023). TKGQG must preserve the temporal condition that determines the answer. We refer to this requirement as temporal faithfulness.

Despite the importance of TKGQG, generating questions that preserve temporal constraints remains challenging. Closely related KGQG methods generate questions through graph encoding, pattern transfer, or reasoning and refinement with large language models (LLMs) (Chen et al., 2020, 2024a; Guo et al., 2022; Xiong et al., 2022; Liang et al., 2023; Guo et al., 2024; Zhao et al., 2024b; Ren et al., 2025). However, preserving entities and relations does not ensure that a question retains the temporal conditions governing its answer. As shown in Figure 1, prompting baselines can reverse a time condition or omit a reference event or order cue despite producing fluent questions. We call this temporal semantic drift: the generated question departs from the intended temporal query and may change its answer set.

![](images/30ccf4db1c5c975d6d53b22b11477cb4aba4cca4575c90f8dbe45defefaeaece.jpg)  
Figure 1: Motivating examples of temporal semantic drift. The prompting baselines are fluent but change the time condition, reference event, answer field, or order cue.

TKGQG requires addressing three challenges. First, the same temporal facts can support different questions, requiring the intended answer field and temporal rule to be identified from the input graph, constraints, and target answers. Second, similar relation patterns can express different temporal requirements, so pattern reuse must account for temporal compatibility. Third, generation can alter these requirements or disclose the answer, making output validation necessary even when the question is fluent.

This paper studies TKGQG as a constraintpreserving graph-to-question generation problem. Given a local temporal subgraph and an answer set, the goal is to generate one answerable question that preserves the temporal intent of the input instance. Our central idea is to make this intent explicit before question realization and use it to retrieve compatible typed question patterns. Under this view, temporal rules are semantic constraints rather than optional surface cues.

To address these challenges, we propose ChronoSynth, a memory-augmented and constraint-aware framework for TKGQG. It constructs a Temporal Intent State to specify the intended question, builds an indexed pattern memory from training questions, and retrieves patterns by temporal compatibility. Typed adaptation binds the patterns to the input graph, while validation and feedback refinement check the generated question. Experiments on CRONQUESTIONS and MultiTQ show improved generation quality across both datasets and generator backends.

We make the following contributions.

• We formulate TKGQG as a constraintpreserving graph-to-question problem and introduce a Temporal Intent State that makes the answer field, temporal rule, reference point, and graph context explicit.

• We propose ChronoSynth, which uses temporal intent to retrieve and adapt reusable question patterns, followed by constraint validation and feedback refinement.

• We evaluate ChronoSynth on CRONQUES-TIONS and MultiTQ, demonstrating stronger generation scores than the compared baselines and examining component contributions, temporal faithfulness, and synthesis efficiency.

## 2 Preliminaries

Temporal Knowledge Graph. A temporal knowledge graph is defined as

$$
\mathcal { G } = ( \mathcal { E } , \mathcal { R } , \mathcal { T } , \mathcal { F } ) , \quad \mathcal { F } \subseteq \mathcal { E } \times \mathcal { R } \times \mathcal { E } \times \mathcal { T } ,\tag{1}
$$

where E, R, and $\tau$ are the entity, relation, and time domains. Each fact $( s , r , o , \tau ) \in \mathcal { F }$ states that relation r holds between subject s and object o at timestamp or interval τ.

Temporal KG Question Generation. Given an instance $\boldsymbol { x } = ( G _ { x } , A _ { x } )$ , where $G _ { x } \subseteq { \mathcal { G } }$ is a local temporal subgraph and $A _ { x }$ is the target answer set, TKGQG generates a natural-language question $q _ { x }$ . The question should satisfy three requirements: fluency, as a well-formed question; answerability, with $A _ { x }$ derivable from $G _ { x } ;$ and temporal consistency, preserving the input’s time condition, reference point, temporal order, and answer field. Explicit constraints are supplied as part of the input graph representation. Temporal rules such as before, after, first, and last govern answer selection. They are semantic constraints: reversing a direction or dropping an extremum condition can change the answer set. The full rule inventory is given in the appendix.

Algorithm 1 ChronoSynth question generation   
Require: Graph $G _ { x } ,$ answers $A _ { x } ,$ , indexed memory $\mathcal { M } ,$   
generator $p _ { \theta } .$ , interpreter R   
Ensure: Generated question $q _ { x }$   
1: $S _ { x } \gets \mathrm { { B U I L D S T A T E } } \langle G _ { x } , A _ { x } \bar { ) }$   
2: Retrieve candidates from M using $S _ { x }$   
3: Rank candidates by Eq. 4   
4: Adapt top patterns to $G _ { x }$ and $S _ { x }$   
5: Filter drafts; supplement with default patterns   
6: Build context $P _ { x }$ from state, evidence, and patterns   
7: $\hat { q } _ { x }  p _ { \theta } ( P _ { x } )$   
8: q ← CHECKANDREPAIR $( \hat { q } _ { x } , S _ { x } , A _ { x } )$   
9: $q _ { x }  q _ { 0 }$   
10: Interpret $q _ { 0 }$ with $R$ and execute on $G _ { x }$   
11: if a supported inconsistency is found then   
12: Search for a correction preserving $A _ { x }$ and explicit con  
straints   
13: if a unique minimum-change correction exists then   
14: Express the correction as a revised question $q ^ { \prime }$   
15: Reinterpret $\cdot q ^ { \prime }$ and check the correction, answers, con  
straints, and format   
16: if all checks pass then   
17: $q _ { x }  q ^ { \prime }$   
end if   
end if   
end if   
18: return $q _ { x }$

## 3 Methodology

## 3.1 Overview

ChronoSynth addresses temporal semantic drift by making temporal intent explicit before question realization and using it to guide pattern reuse (Figure 2). It first constructs a Temporal Intent State from the input graph and answers. An indexed pattern memory supplies training questions and typed patterns with compatible temporal requirements. The generator then realizes a question from these patterns and the input evidence, followed by validation and feedback refinement. Memory construction is performed offline; the same index is reused across generation instances and backends. Algorithm 1 summarizes generation using this memory.

## 3.2 Temporal Intent State Construction

To select a compatible question pattern, ChronoSynth must distinguish the intended answer field, temporal rule, and reference point. We represent these choices and their graph context as

$$
S _ { x } = \langle a _ { x } , y _ { x } , \omega _ { x } , g _ { x } , e _ { x } , h _ { x } , c _ { x } , \phi _ { x } \rangle .\tag{2}
$$

Here $a _ { x }$ is the answer field, $y _ { x }$ the answer type, $\omega _ { x }$ the temporal rule, and $g _ { x }$ the time granularity. The remaining fields specify the focus edge $e _ { x } ,$ reference entity or event $h _ { x } .$ , comparison value $c _ { x } ,$ and graph context $\phi _ { x }$ . Together, they distinguish questions that share a relation but ask for different entities, times, or event orders.

The state is inferred from normalized graph records and supplied answers. Explicit constraint records take precedence in determining $\omega _ { x }$ . Answer strings and timestamp formats determine $y _ { x }$ and $g _ { x }$ . Matching the answers against edge sources, targets, and time fields identifies $a _ { x }$ and the focus edge. The matched temporal pattern then provides the reference point and graph context. Applying this construction to both training instances and new inputs makes their temporal requirements comparable. Field definitions and rule details are provided in the appendix.

## 3.3 Indexed Pattern Memory Construction

Typed pattern normalization. For each training instance z, ChronoSynth replaces instance-specific mentions in its reference question with typed placeholders. The resulting record is

$$
m _ { z } = ( q _ { z } , \pi _ { z } , k _ { z } , \eta _ { z } ) ,\tag{3}
$$

where $q _ { z }$ is the reference question, $\pi _ { z }$ the typed pattern, $k _ { z }$ its primary index key, and $\eta _ { z }$ its relation, topology, and surface metadata. Normalization retains the question structure while separating it from particular entities and dates. Anchor, comparisontime, source, and target placeholders preserve distinct semantic roles when the pattern is transferred to a new input.

Memory organization. Each record is indexed by fields derived from its Temporal Intent State. The primary key combines the dataset dialect, temporal rule, answer field and type, time granularity, and edge-count bucket. Complementary relationpattern and graph-shape indexes broaden the candidate pool beyond exact temporal matches; a bounded global pool supplies fallbacks when the combined pool is too small. This organization supports reuse across different entities while retaining the temporal distinctions needed for ranking. The complete index definitions are given in the appendix.

![](images/cbf3055ddb8532a5d5b6818568c1961b19b204baf052b7a9405bd322ab9afb5d.jpg)  
Figure 2: Overview of ChronoSynth. ChronoSynth constructs a Temporal Intent State, retrieves compatible typed question patterns from an indexed pattern memory, adapts them to the input graph, and validates and refines the generated question.

## 3.4 Memory-guided Question Generation

Rule-aware memory retrieval. Given $S _ { x } ,$ ChronoSynth collects candidates from the temporal bucket, relation-pattern, and graph-shape indexes, supplementing them from the fallback pool when necessary. It ranks each candidate $m _ { z }$ by a fixed compatibility score:

$$
\operatorname { s c o r e } ( m _ { z } , S _ { x } ) = \sum _ { j \in \mathcal { I } } \lambda _ { j } \kappa _ { j } ( m _ { z } , S _ { x } ) .\tag{4}
$$

Here $\mathcal { I }$ indexes compatibility features, $\kappa _ { j }$ their values, and $\lambda _ { j }$ their fixed weights. The features measure agreement in temporal rule, answer field and type, granularity, relation, dataset dialect, and graph shape, together with edge-count similarity, reference-token overlap, and reusable placeholders. The temporal rule receives the largest weight because relation-similar instances can require opposite temporal conditions. Relation and graph features provide additional ranking evidence. The weights are fixed before test evaluation. The expanded score and index settings are provided in the appendix.

Typed pattern adaptation. For each top-ranked pattern $\pi _ { z }$ , ChronoSynth constructs a substitution map $\mu _ { x }$ from the current state and graph:

$$
\tilde { q } _ { x , z } = \mathrm { A d a p t } ( \pi _ { z } , \mu _ { x } ) .\tag{5}
$$

The map binds anchors, comparison values, and focus entities or times to their respective roles. Keeping these roles distinct prevents relation similarity from making a reference entity interchangeable with an answer entity. ChronoSynth discards drafts with unresolved placeholders, literal answer disclosure, or duplicate wording. It retains at most four valid drafts and uses rule-specific default patterns when too few retrieved drafts remain.

Constraint-aware realization. The generation context $P _ { x }$ combines the Temporal Intent State, input graph and answers, retrieved memory records, adapted drafts, and generation constraints. A fixed language model produces the initial question:

$$
{ \hat { q } } _ { x } = \arg \operatorname* { m a x } _ { q } p _ { \theta } ( q \mid P _ { x } ) .\tag{6}
$$

Here θ denotes the fixed generator parameters. The state specifies the intended temporal choices, while the retrieved questions and adapted patterns provide examples of how to express them. The constraints require a single question without unresolved placeholders or answer disclosure and with the intended order cues, temporal rule, and answer field. Generation does not update model parameters or pattern memory.

## 3.5 Validation and Refinement

After realization, ChronoSynth cleans the output and checks question format, placeholders, answer leakage, order cues, temporal-rule compatibility, and the answer field. Failed checks trigger the original repair or draft fallback, producing the base question $q _ { 0 }$

The final stage checks the meaning of q against the input graph, target answers, and explicit temporal constraints. An interpreter maps the question to a temporal query program, which is executed on the graph. When a supported inconsistency is detected, the resulting feedback guides a targeted revision. A second interpretation must confirm that the revised wording expresses the selected correction and preserves the target answers and explicit constraints. If verification fails or no supported correction is available, the base question is retained. This stage refines individual questions without changing the state construction, retrieval procedure, or memory. Detailed indexing, validation checks, and complexity analysis are provided in the appendix.

## 4 Experimental Evaluation

We compare ChronoSynth with existing generation methods, examine its components through ablation, and assess question quality and utility through human evaluation and downstream QA. Additional analyses and detailed experimental settings are provided in the appendix.

## 4.1 Experimental Setting

Datasets. We evaluate ChronoSynth on the graphconverted test splits of CRONQUESTIONS and MultiTQ (Saxena et al., 2021; Chen et al., 2023). CRONQUESTIONS mainly contains timestampcentered temporal questions. MultiTQ contains more diverse temporal rules and larger local subgraphs. Dataset statistics and additional settings are provided in the appendix.

Baselines. We compare ChronoSynth with two categories of baselines: parametric trained generation and training-free inference-time generation. The parametric trained group includes graphto-sequence models and RoleAgentQG variants. The training-free group includes direct prompting, KQG-CoT prompting, and ChronoSynth (Chen et al., 2024a; Liang et al., 2023). In the trainingfree group, rows named by a backend alone denote direct prompting with that backend. KQG-CoT uses the same backend with reasoningstyle prompting, while ChronoSynth adds indexed pattern-memory retrieval, typed pattern adaptation, and constraint validation without updating generator parameters. For LLM-based comparisons, we keep the backend family aligned between ChronoSynth and its prompting counterparts.

Metrics. For the full baseline comparison, we report BLEU-1/2/3/4, ROUGE-L, and CIDEr. BLEU and ROUGE-L measure lexical overlap with reference questions, while CIDEr measures TF–IDFweighted n-gram similarity. We additionally assess temporal faithfulness and answerability through human evaluation and analyze generation performance across temporal rules, graph sizes, and answer types.

## 4.2 Main Results

Table 1 reports the full-test results on both datasets. It groups baselines by whether they update generation-model parameters. The trained group includes Graph2Seq-style and RoleAgentQG variants. The training-free group includes direct prompting, KQG-CoT, and ChronoSynth. Direct prompting and KQG-CoT use prompting only, whereas ChronoSynth additionally uses indexed pattern-memory retrieval, typed pattern adaptation, and constraint validation without updating generator parameters. Best values are bolded and secondbest values are underlined within each dataset. We organize the results as three observations.

Observation 1: ChronoSynth consistently improves temporal question generation quality. On CRONQUESTIONS, ChronoSynth reached BLEU-4 0.2971 and CIDEr 2.7491 with Qwen3.5-9B. The GPT-4o-mini configuration obtained BLEU-4 0.2956 and CIDEr 2.7487. By contrast, the best KQG-CoT BLEU-4 was 0.2149, and the best direct-prompting BLEU-4 was 0.1956. Compared with the strongest baseline for each metric, ChronoSynth improved BLEU-4 by 15.3% and CIDEr by 34.3%. The trained methods were competitive on some lexical metrics, but ChronoSynth ranked first and second across all six metrics with its two generator configurations.

Observation 2: the improvement is larger on temporally complex MultiTQ questions. On MultiTQ, ChronoSynth reached BLEU-4 0.3117, ROUGE-L 0.5280, and CIDEr 2.6509 with Qwen3.5-9B. With GPT-4o-mini, it obtained 0.3044, 0.5238, and 2.6197, respectively, compared with 0.1898, 0.4581, and 1.5053 for KQG-CoT. The best ChronoSynth scores improved BLEU-4 by 46.0% and CIDEr by 76.1% over the strongest baseline for each metric. These results indicate that indexed pattern memory is especially useful when the input contains richer temporal rules and larger local subgraphs.

Observation 3: the gains extend across generator backends. ChronoSynth improved over direct prompting and KQG-CoT with both Qwen3.5-9B and GPT-4o-mini. These configurations use the same temporal-intent representation, retrieval procedure, and validation policy, with GPT-4o-mini interpreting questions during feedback refinement. The consistent gains support using the framework with either generator.

<table><tr><td colspan="8">CRONQUESTIONS</td></tr><tr><td>Setting</td><td>Method</td><td>BLEU-1 BLEU-2 BLEU-3 BLEU-4 ROUGE-L</td><td></td><td></td><td></td><td></td><td>CIDEr</td></tr><tr><td rowspan="4">Trained</td><td>G2S-AE</td><td>0.5011</td><td>0.3873</td><td>0.3102</td><td>0.2577</td><td>0.4799</td><td>0.2480</td></tr><tr><td>G2S-AE-RL</td><td>0.3852</td><td>0.2628</td><td>0.1792</td><td>0.1261</td><td>0.3850</td><td>0.0625</td></tr><tr><td>RoleAgentQG-G2S</td><td>0.3743</td><td>0.2771</td><td>0.2123</td><td>0.1703</td><td>0.3686</td><td>0.1037</td></tr><tr><td>RoleAgentQG-G2S-AE-RL</td><td>0.3842</td><td>0.2861</td><td>0.2205</td><td>0.1778</td><td>0.3772</td><td>0.1110</td></tr><tr><td rowspan="5">Training-free</td><td>Qwen3.5-9B</td><td>0.4237</td><td>0.3122</td><td>0.2417</td><td>0.1952</td><td>0.4029</td><td>1.8729</td></tr><tr><td>GPT-4o-mini</td><td>0.4246</td><td>0.3124</td><td>0.2420</td><td>0.1956</td><td>0.4040</td><td>1.8745</td></tr><tr><td>KQG-CoT (Qwen3.5-9B)</td><td>0.3752</td><td>0.2731</td><td>0.2067</td><td>0.1635</td><td>0.3496</td><td>1.5976</td></tr><tr><td>KQG-CoT (GPT-4o-mini)</td><td>0.4665</td><td>0.3468</td><td>0.2667</td><td>0.2149</td><td>0.4290</td><td>2.0475</td></tr><tr><td>ChronoSynth (Qwen3.5-9B)</td><td>0.5691</td><td>0.4426</td><td>0.3559</td><td>0.2971</td><td>0.5063</td><td>2.7491</td></tr><tr><td></td><td>ChronoSynth (GPT-4o-mini)</td><td>0.5700</td><td>0.4425</td><td>0.3550</td><td>0.2956</td><td>0.5059</td><td>2.7487</td></tr><tr><td colspan="2">Relative improvement</td><td>+13.7%</td><td>+14.3%</td><td>+14.7%</td><td>+15.3%</td><td></td><td>+5.5% +34.3%</td></tr><tr><td colspan="8">MultiTQ</td></tr><tr><td colspan="8">BLEU-1 BLEU-2 BLEU-3 BLEU-4 ROUGE-L</td></tr><tr><td>Setting</td><td>Method</td><td></td><td></td><td></td><td></td><td></td><td>CIDEr</td></tr><tr><td rowspan="4">Trained</td><td>G2S-AE</td><td>0.1509</td><td>0.0670</td><td>0.0380</td><td>0.0234</td><td>0.1539</td><td>0.0381</td></tr><tr><td>G2S-AE-RL</td><td>0.4563</td><td>0.3510</td><td>0.2743</td><td>0.2135</td><td>0.4799</td><td>0.4079</td></tr><tr><td>RoleAgentQG-G2S</td><td>0.1205</td><td>0.0456</td><td>0.0210</td><td>0.0102</td><td>0.1224</td><td>0.0099</td></tr><tr><td>RoleAgentQG-G2S-AE-RL</td><td>0.1537</td><td>0.0534</td><td>0.0253</td><td>0.0130</td><td>0.1466</td><td>0.0131</td></tr><tr><td rowspan="5">Training-free</td><td>Qwen3.5-9B</td><td>0.3977</td><td>0.2683</td><td>0.1881</td><td>0.1345</td><td>0.4088</td><td>1.0367</td></tr><tr><td>GPT-4o-mini</td><td>0.3994</td><td>0.2698</td><td>0.1894</td><td>0.1356</td><td>0.4117</td><td>1.0459</td></tr><tr><td>KQG-CoT (Qwen3.5-9B)</td><td>0.4265</td><td>0.3098</td><td>0.2348</td><td>0.1803</td><td>0.4273</td><td>1.3283</td></tr><tr><td>KQG-CoT (GPT-4o-mini)</td><td>0.4436</td><td>0.3236</td><td>0.2460</td><td>0.1898</td><td>0.4581</td><td>1.5053</td></tr><tr><td>ChronoSynth (Qwen3.5-9B)</td><td>0.5747</td><td>0.4551</td><td>0.3739</td><td>0.3117</td><td>0.5280</td><td>2.6509</td></tr><tr><td>ChronoSynth (GPT-4o-mini)</td><td></td><td>0.5637</td><td>0.4457</td><td>0.3659</td><td>0.3044</td><td>0.5238</td><td>2.6197</td></tr><tr><td colspan="2">Relative improvement</td><td>+25.9%</td><td>+29.7%</td><td>+36.3%</td><td>+46.0%</td><td></td><td>+10.0% +76.1%</td></tr></table>

Table 1: Question-generation results on CRONQUESTIONS and MultiTQ. Bold and underlined values denote the best and second best. Relative improvements compare the best ChronoSynth score with the strongest baseline for each metric.

## 4.3 Ablation Study

We examine pattern memory, temporal retrieval, and feedback refinement using GPT-4o-mini on the same test splits as the main comparison (Figure 3). The variants remove pattern-memory retrieval, replace temporal retrieval with relation-only retrieval, or omit feedback refinement while retaining the original validation checks. The memory and retrieval variants retain feedback refinement. Detailed settings are provided in the appendix.

Pattern memory contributes the largest gain. Removing it lowers BLEU-4 from 0.2956 to 0.1881 on CRONQUESTIONS and from 0.3044 to 0.1543 on MultiTQ. ROUGE-L falls from 0.5059 to

0.3520 and from 0.5238 to 0.4047, respectively. Relation-only retrieval also underperforms the complete method: its BLEU-4 is 0.2818 on CRON-QUESTIONS and 0.2521 on MultiTQ. The larger decrease on MultiTQ supports selecting patterns by temporal compatibility rather than relation similarity alone.

Feedback refinement yields smaller, consistent improvements. Without it, BLEU-4 is 0.2947 on CRONQUESTIONS and 0.3037 on MultiTQ. The complete method leads all three variants across BLEU-1 through BLEU-4 and ROUGE-L. The gains indicate that refinement complements the larger contribution of temporal-intent-guided pattern retrieval.

## 4.4 Human Evaluation

We compare questions generated with GPT-4omini using direct prompting, KQG-CoT, retrievalaugmented prompting, and ChronoSynth. Three raters score fluency, temporal faithfulness, and answerability on a 1–5 scale with method identities hidden. Sampling and annotation details are provided in the appendix.

![](images/64bc58f604a310e1c8006171775e49168eddc4577409e476901c377dec62b471.jpg)  
Figure 3: Component ablations on CRONQUESTIONS and MultiTQ. F: complete ChronoSynth; W: w/o Refinement; R: Relation-only Retrieval; M: w/o Pattern Memory.

<table><tr><td colspan="4">Temporal</td></tr><tr><td>Method</td><td></td><td></td><td>Fluency faithfulness Answerability</td></tr><tr><td>CRONQUESTIONS</td><td></td><td></td><td></td></tr><tr><td>GPT-4o-mini</td><td>2.313</td><td>2.287</td><td>2.340</td></tr><tr><td>KQG-CoT</td><td>2.413</td><td>2.360</td><td>2.433</td></tr><tr><td>Retrieval prompting</td><td>2.380</td><td>2.360</td><td>2.373</td></tr><tr><td>ChronoSynth</td><td>2.987</td><td>2.973</td><td>2.967</td></tr><tr><td colspan="4">MultiTQ</td></tr><tr><td>GPT-4o-mini</td><td>2.313</td><td>2.420</td><td>2.453</td></tr><tr><td>KQG-CoT</td><td>2.353</td><td>2.360</td><td>2.400</td></tr><tr><td>Retrieval prompting</td><td>2.353</td><td>2.393</td><td>2.347</td></tr><tr><td>ChronoSynth</td><td>2.873</td><td>2.900</td><td>2.927</td></tr></table>

Table 2: Human evaluation on 50 matched inputs per dataset. Scores average three raters on a 1–5 scale; bold marks the highest mean.

Table 2 shows that ChronoSynth receives the highest mean rating in all three dimensions on both datasets. Its temporal-faithfulness scores reach 2.973 on CRONQUESTIONS and 2.900 on MultiTQ, compared with baseline maxima of 2.360 and 2.420. Higher answerability ratings further indicate that the generated questions preserve the conditions needed to recover the target answers. These findings complement the reference-based generation metrics.

<table><tr><td>Training data</td><td>CRON- QUESTIONS</td><td>MultiTQ</td></tr><tr><td>Gold Reference</td><td> $9 8 . 3 3 \pm 0 . 4 2 $ </td><td> $9 2 . 0 2 \pm 0 . 5 0$ </td></tr><tr><td>GPT-4o-mini</td><td> $9 2 . 3 6 \pm 2 . 8 8$ </td><td> $7 3 . 3 6 \pm 2 . 3 1$ </td></tr><tr><td>KQG-CoT</td><td> $9 5 . 1 5 \pm 1 . 2 2$ </td><td> $7 6 . 9 0 \pm 1 . 5 4$ </td></tr><tr><td>Vanilla RAG</td><td> $8 8 . 3 0 \pm 0 . 6 4$ </td><td> $7 7 . 7 1 \pm 2 . 0 7$ </td></tr><tr><td>ChronoSynth</td><td> ${ \bf 9 5 . 4 9 \pm 2 . 0 5 }$  </td><td> ${ \bf 8 6 . 3 7 \pm 0 . 1 0 }$ </td></tr></table>

Table 3: Downstream QA EM (%) with Qwen3.5-0.8B. Values are mean ± sample standard deviation across three seeds. Bold marks the best synthetic-data mean.

## 4.5 Downstream Question Answering

We evaluate whether generated questions provide useful supervision for temporal QA. Keeping training graphs and answer sets fixed, we fine-tune Qwen3.5-0.8B using questions from each method and evaluate it on original held-out questions. Synthetic-data conditions use generated questions only; Gold Reference uses the original training questions. Training and evaluation details are provided in the appendix.

ChronoSynth achieves the highest mean EM among synthetic-data conditions on both datasets (Table 3). It exceeds the strongest baseline by 0.34 points on CRONQUESTIONS and 8.66 points on MultiTQ. The CRONQUESTIONS difference is small relative to seed variation, while the larger MultiTQ gain supports the utility of preserving richer temporal conditions during synthesis. The remaining gaps to Gold Reference are 2.84 and 5.65 points, respectively.

## 5 Related Work

## 5.1 Knowledge Graph Question Generation

KGQG methods generated questions from structured graph inputs and a target answer. Early neural work encoded a local subgraph and decoded a question with copy mechanisms (Chen et al., 2024a). Later methods improved coverage with retrieval or diverse-subgraph modeling (Guo et al., 2022; Xiong et al., 2022). Recent LLM-based methods used prompting and reasoning-style generation (Liang et al., 2023; Guo et al., 2024; Zhao et al., 2024b), while R2DQG studied diverse realizations of the same semantics (Ren et al., 2025). Our setting is stricter because the generated question must preserve temporal rules, reference events or times, and answer fields. Broader question-generation work contributed answer-aware, pretrained, and controlled generation strategies (Du et al., 2017; Reddy et al., 2017; ElSahar et al., 2018; Vaswani et al., 2017; Sun et al., 2018; Song et al., 2018; Kim et al., 2019; Raffel et al., 2020; Fei et al., 2021, 2022; Bi et al., 2024). Related graph-to-text and KBQA settings connected KGQG to SPARQL verbalization, semantic parsing, and benchmarkcentered graph question construction (Lecorvé et al., 2022; Jia and Liang, 2016; Saxena et al., 2020; Jia et al., 2021).

## 5.2 Temporal Question Answering over Knowledge Graphs

Temporal KGQA provides the closest task context. CRONQUESTIONS organizes questions by temporal complexity (Saxena et al., 2021), and MultiTQ stresses multi-granularity temporal questions (Chen et al., 2023). Recent TKGQA work studies temporal retrieval, subgraph reasoning, and LLM-guided reasoning (Chen et al., 2024b; Qian et al., 2024; Gao et al., 2014; Chen et al., 2022; Agarwal et al., 2024; Zhang et al., 2024). These methods answer temporal questions over temporal graphs; our task works in the inverse direction and has to preserve temporal rules during generation. Related temporal KG research includes surveys, temporal QA variants, and temporal representation learning over dynamic graphs (Liang et al., 2024; Cai et al., 2024; Wang et al., 2024; Lin et al., 2023). This line of work motivates making temporal intent explicit rather than leaving it implicit in a linearized

graph.

## 5.3 Retrieval-guided and Constraint-aware Generation

Retrieval-augmented generation and selfrefinement used retrieved evidence or intermediate feedback to improve generation quality (Guu et al., 2020; Lewis et al., 2020; Yao et al., 2023; Asai et al., 2024). Structured-evidence prompting was also common in KG question answering (Baek et al., 2023; Zhao et al., 2024a). ChronoSynth differs in both objective and retrieval unit: retrieval supports question generation rather than answering, the retrieved objects are typed question patterns instead of passages, and validation checks answer leakage, placeholder resolution, order cues, temporal rules, and answer fields.

## 6 Conclusion

We have presented ChronoSynth, a constraintaware question generation framework for temporal knowledge graphs. By making temporal intent explicit through Temporal Intent State construction, indexed pattern-memory retrieval, typed pattern adaptation, and validation with feedback refinement, ChronoSynth has reduced temporal semantic drift during question realization. Experiments on CRONQUESTIONS and MultiTQ have shown consistent gains over parametric trained methods and training-free baselines across multiple backends. Ablation and grouped analyses have shown that pattern memory is the main source of improvement. The local efficiency study further demonstrates its quality–cost advantage over KQG-CoT. Future work will extend the temporal-rule inventory and strengthen temporal-consistency checking for broader query styles.

## Limitations

ChronoSynth relies on a predefined temporal-rule inventory and heuristic state construction, which may require adaptation to new graph schemas or temporal expressions. Pattern reuse depends on the coverage of the training memory, while verification depends on the question interpreter and does not guarantee semantic correctness. Our evaluation focuses on temporal question-generation benchmarks; performance in other domains and languages remains untested.

## Ethical Considerations

Generated questions may reproduce biases or factual errors in the source graphs and language models. They should therefore be reviewed before use in downstream datasets, particularly in sensitive applications. Reuse of the source datasets and models should follow their licenses and terms of use.

## References

Prerna Agarwal, Nishant Kumar, and Srikanta Bedathur. 2024. SymKGQA: Few-Shot Knowledge Graph Question Answering via Symbolic Program Generation and Execution. In Proceedings of the Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 10119– 10140.

Marcelo Arenas, Claudio Gutierrez, and Juan F. Sequeda. 2021. Querying in the Age of Graph Databases and Knowledge Graphs. In Proceedings of the ACM SIGMOD International Conference on Management ofData, pages 2821–2828.

Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2024. Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. In International Conference on Learning Representations.

Jinheon Baek, Alham Fikri Aji, and Amir Saffari. 2023. Knowledge-Augmented Language Model Prompting for Zero-Shot Knowledge Graph Question Answering. CoRR, abs/2306.04136.

Sheng Bi, Jianyu Liu, Zeyi Miao, and Qizhi Min. 2024. Difficulty-controllable question generation over knowledge graphs: A counterfactual reasoning approach. Information Processing and Management, page 103721.

Li Cai, Xin Mao, Yuhao Zhou, Zhaoguang Long, Changxu Wu, and Man Lan. 2024. A Survey on Temporal Knowledge Graph: Representation Learning and Applications. CoRR, abs/2403.04782.

Yu Chen, Lingfei Wu, and Mohammed J. Zaki. 2020. Reinforcement Learning Based Graph-to-Sequence Model for Natural Question Generation. In International Conference on Learning Representations.

Yu Chen, Lingfei Wu, and Mohammed J. Zaki. 2024a. Toward Subgraph-Guided Knowledge Graph Question Generation With Graph Neural Networks. IEEE Transactions on Neural Networks and Learning Systems, pages 12706–12717.

Ziyang Chen, Dongfang Li, Xiang Zhao, Baotian Hu, and Min Zhang. 2024b. Temporal Knowledge Question Answering via Abstract Reasoning Induction. In Proceedings of the Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 4872–4889.

Ziyang Chen, Jinzhi Liao, and Xiang Zhao. 2023. Multi-granularity Temporal Question Answering over Knowledge Graphs. In Proceedings ofthe Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 11378– 11392.

Ziyang Chen, Xiang Zhao, Jinzhi Liao, Xinyi Li, and Evangelos Kanoulas. 2022. Temporal knowledge graph question answering via subgraph reasoning. Knowledge-Based Systems, page 109134.

Xinya Du, Junru Shao, and Claire Cardie. 2017. Learning to Ask: Neural Question Generation for Reading Comprehension. In Proceedings ofthe Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1342–1352.

Hady ElSahar, Christophe Gravier, and Frédérique Laforest. 2018. Zero-Shot Question Generation from Knowledge Graphs for Unseen Predicates and Entity Types. In Proceedings of NAACL-HLT, pages 218–228.

Zichu Fei, Qi Zhang, and Yaqian Zhou. 2021. Iterative GNN-based Decoder for Question Generation. In Proceedings of the Conference on Empirical Methods in Natural Language Processing, pages 2573–2582.

Zichu Fei, Xin Zhou, Tao Gui, Qi Zhang, and Xuanjing Huang. 2022. LFKQG: A Controlled Generation Framework with Local Fine-tuning for Question Generation over Knowledge Bases. In Proceedings of the International Conference on Computational Linguistics, pages 6575–6585.

Yifu Gao, Linbo Qiao, Zhigang Kan, Zhihua Wen, Yongquan He, and Dongsheng Li. 2014. Two-stage Generative Question Answering on Temporal Knowledge Graph Using Large Language Models. In Findings ofthe Associationfor Computational Linguistics: ACL, pages 6719–6734.

Shasha Guo, Lizi Liao, Jing Zhang, Yanling Wang, Cuiping Li, and Hong Chen. 2024. SGSH: Stimulate Large Language Models with Skeleton Heuristics for Knowledge Base Question Generation. In Findings of the Association for Computational Linguistics: NAACL, pages 4613–4625.

Shasha Guo, Jing Zhang, Yanling Wang, Qianyi Zhang, Cuiping Li, and Hong Chen. 2022. DSM: Question Generation over Knowledge Base via Modeling Diverse Subgraphs with Meta-learner. In Proceedings of the Conference on Empirical Methods in Natural Language Processing, pages 4194–4207.

Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. 2020. Retrieval Augmented Language Model Pre-Training. In Proceedings of the International Conference on Machine Learning, pages 3929–3938.

Robin Jia and Percy Liang. 2016. Data Recombination for Neural Semantic Parsing. In Proceedings of the Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers).

Zhen Jia, Soumajit Pramanik, Rishiraj Saha Roy, and Gerhard Weikum. 2021. Complex Temporal Question Answering on Knowledge Graphs. In Proceedings of the ACM International Conference on Information and Knowledge Management, pages 792–802.

Yanghoon Kim, Hwanhee Lee, Joongbo Shin, and Kyomin Jung. 2019. Improving Neural Question Generation Using Answer Separation. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 6602–6609.

Gwénolé Lecorvé, Morgan Veyret, Quentin Brabant, and Lina Maria Rojas-Barahona. 2022. SPARQL-to-Text Question Generation for Knowledge-Based Conversational Applications. In Proceedings ofAACL-IJCNLP (Volume 1: Long Papers), pages 131–147.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. In Advances in Neural Information Processing Systems.

Ke Liang, Lingyuan Meng, Meng Liu, Yue Liu, Wenxuan Tu, Siwei Wang, Sihang Zhou, Xinwang Liu, Fuchun Sun, and Kunlun He. 2024. A Survey of Knowledge Graph Reasoning on Graph Types: Static, Dynamic, and Multi-Modal. IEEE Transactions on Pattern Analysis and Machine Intelligence, pages 9456–9478.

Yuanyuan Liang, Jianing Wang, Hanlun Zhu, Lei Wang, Weining Qian, and Yunshi Lan. 2023. Prompting Large Language Models with Chain-of-Thought for Few-Shot Knowledge Base Question Generation. In Proceedings ofthe Conference on Empirical Methods in Natural Language Processing, pages 4329–4343.

Xueyuan Lin, Haihong E, Chengjin Xu, Gengxian Zhou, Haoran Luo, Tianyi Hu, Fenglong Su, Ningyuan Li, and Mingzhi Sun. 2023. TFLEX: Temporal Feature-Logic Embedding Framework for Complex Reasoning over Temporal Knowledge Graph. In Advances in Neural Information Processing Systems.

Xinying Qian, Ying Zhang, Yu Zhao, Baohang Zhou, Xuhui Sui, Li Zhang, and Kehui Song. 2024. TimeR<sup>4</sup> : Time-aware Retrieval-Augmented Large Language Models for Temporal Knowledge Graph Question Answering. In Proceedings of the Conference on Empirical Methods in Natural Language Processing, pages 6942–6952.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2020. Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. Journal ofMachine Learning Research, pages 140:1–140:67.

Sathish Reddy, Dinesh Raghu, Mitesh M. Khapra, and Sachindra Joshi. 2017. Generating Natural Language Question-Answer Pairs from a Knowledge Graph

Using a RNN Based Question Generation Model. In Proceedings of EACL (Volume 1: Long Papers), pages 376–385.

Yimeng Ren, Yanhua Yu, Lizi Liao, Yuhu Shang, Kangkang Lu, and Mingliang Yan. 2025. R2DQG: A Quality Meets Diversity Framework for Question Generation over Knowledge Bases. In Proceedings of the International Joint Conference on Artificial Intelligence, pages 8231–8240.

Apoorv Saxena, Soumen Chakrabarti, and Partha P. Talukdar. 2021. Question Answering Over Temporal Knowledge Graphs. In Proceedings ofACL-IJCNLP (Volume 1: Long Papers), pages 6663–6676.

Apoorv Saxena, Aditay Tripathi, and Partha P. Talukdar. 2020. Improving Multi-hop Question Answering over Knowledge Graphs using Knowledge Base Embeddings. In Proceedings of the Annual Meeting of the Associationfor Computational Linguistics, pages 4498–4507.

Linfeng Song, Zhiguo Wang, Wael Hamza, Yue Zhang, and Daniel Gildea. 2018. Leveraging Context Information for Natural Question Generation. In Proceedings ofNAACL-HLT (Volume 2: Short Papers), pages 569–574.

Xingwu Sun, Jing Liu, Yajuan Lyu, Wei He, Yanjun Ma, and Shi Wang. 2018. Answer-focused and Positionaware Neural Question Generation. In Proceedings ofthe Conference on Empirical Methods in Natural Language Processing, pages 3930–3939.

Zhiyang Tang, Yanping Wu, Xiangjun Zai, Chen Chen, Xiaoyang Wang, and Ying Zhang. 2025. Efficient Temporal Simple Path Graph Generation. In IEEE International Conference on Data Engineering, pages 3589–3601.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is All you Need. In Advances in Neural Information Processing Systems, pages 5998–6008.

Jiapu Wang, Kai Sun, Linhao Luo, Wei Wei, Yongli Hu, Alan Wee-Chung Liew, Shirui Pan, and Baocai Yin. 2024. Large Language Models-guided Dynamic Adaptation for Temporal Knowledge Graph Reasoning. In Advances in Neural Information Processing Systems.

Guanming Xiong, Junwei Bao, Wen Zhao, Youzheng Wu, and Xiaodong He. 2022. AutoQGS: Auto-Prompt for Low-Resource Knowledge-based Question Generation from SPARQL. In Proceedings of the ACM International Conference on Information and Knowledge Management, pages 2250–2259.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R. Narasimhan, and Yuan Cao. 2023. ReAct: Synergizing Reasoning and Acting in Language Models. In International Conference on Learning Representations.

Yu Zhang, Kehai Chen, Xuefeng Bai, Zhao Kang, Quanjiang Guo, and Min Zhang. 2024. Questionguided Knowledge Graph Re-scoring and Injection for Knowledge Graph Question Answering. In Findings ofthe Associationfor Computational Linguistics: EMNLP, pages 8972–8985.

Ruilin Zhao, Feng Zhao, Long Wang, Xianzhi Wang, and Guandong Xu. 2024a. KG-CoT: Chain-of-Thought Prompting of Large Language Models over Knowledge Graphs for Knowledge-Aware Question Answering. In Proceedings ofthe International Joint Conference on Artificial Intelligence, pages 6642– 6650.

Runhao Zhao, Jiuyang Tang, Weixin Zeng, Ziyang Chen, and Xiang Zhao. 2024b. Zero-shot Knowledge Graph Question Generation via Multi-agent LLMs and Small Models Synthesis. In Proceedings ofthe ACM International Conference on Information and Knowledge Management, pages 3341–3351.

<table><tr><td>Temporal rule</td><td>Trigger condition</td><td>Semantics</td></tr><tr><td>plain</td><td>temporal comparison</td><td>entity answer, no explicit ask a non-extremum factual question</td></tr><tr><td>when</td><td>reference</td><td>time answer, no explicit ask for the event timestamp</td></tr><tr><td>first</td><td>liest time in its same- participant</td><td>focus edge has the ear- ask for the earliest event or</td></tr><tr><td>last</td><td>relation family time in its same-relation ticipant family</td><td>focus edge has the latest ask for the latest event or par-</td></tr><tr><td>before</td><td>later reference exists</td><td>ask for an answer before a ref- erence event</td></tr><tr><td>before_last</td><td>valid answer before it</td><td>later reference and latest ask for the last answer before a reference event</td></tr><tr><td>after</td><td>earlier reference exists</td><td>ask for an answer after a ref- erence event</td></tr><tr><td>after_first</td><td>liest valid answer after it reference event</td><td>earlier reference and ear- ask for the first answer after a</td></tr><tr><td>before_time</td><td>straint exists</td><td>explicit before-time con- ask under an explicit before- time constraint</td></tr><tr><td>after_time</td><td>straint exists</td><td>explicit after-time con- ask under an explicit after- time constraint</td></tr></table>

Table A1: Temporal-rule inventory.

## A Implementation Details

## A.1 Temporal Rules

Table A1 summarizes the deterministic state constructor. Explicit constraints take precedence; otherwise, the focus fact and neighboring facts with the same relation provide a temporal interpretation. When several questions are possible, this selects an interpretation rather than recovering a unique reference wording.

For inferred rules, the comparison family contains non-constraint facts with the focus relation and anchor entity. Explicit constraints take precedence. Otherwise, entity-answer inputs first check for a later reference fact, then an earlier reference fact; the nearest applicable fact supplies the reference. When neither relative rule applies and several temporal facts remain, the constructor tests whether the focus time is earliest or latest. These decisions use normalized temporal keys. The execution-based validator uses the interval semantics described below.

## A.2 State Fields and Memory Indexes

Table A2 lists the Temporal Intent State fields. Each training record stores its original question, normalized pattern, primary key, and auxiliary metadata. Typed placeholders include {ANCHOR}, {CMP\_ENTITY}, {CMP\_TIME}, {FOCUS\_SOURCE}, {FOCUS\_TARGET}, and {FOCUS\_TIME}.

The primary memory key is

$$
k _ { z } = ( d _ { z } , \omega _ { z } , a _ { z } , y _ { z } , g _ { z } , b _ { z } ) ,\tag{7}
$$

<table><tr><td>Field</td><td>Meaning</td><td>Main use</td></tr><tr><td>ax</td><td>answer field</td><td>decides whether the question asks for the source, target, or time field</td></tr><tr><td> $y _ { x }$ </td><td>answer type</td><td>constrains the wh-word and surface form controls temporal semantics such as</td></tr><tr><td> $\omega _ { x }$ </td><td>temporal rule</td><td>before, after, and last</td></tr><tr><td> $g _ { x }$ </td><td>time granularity</td><td>distinguishes year-, month-, and day-level realization</td></tr><tr><td> $e _ { x }$   $h _ { x }$ </td><td>focus edge reference en-</td><td>identifies the fact to verbalize provides the reference point for retrieval</td></tr><tr><td></td><td>tity/event</td><td>and drafting</td></tr><tr><td> $c _ { x }$ </td><td>comparison value</td><td>specifies the entity, event, or timestamp used for temporal comparison</td></tr><tr><td> $\phi _ { x }$ </td><td>graph text/shape</td><td>con- summarizes topology for memory- retrieval compatibility</td></tr></table>

Table A2: Fields in the Temporal Intent State.

where $d _ { z }$ is the dataset dialect, $\omega _ { z }$ is the temporal rule, $a _ { z }$ is the answer field, $y _ { z }$ is the answer type, $g _ { z }$ is the time granularity, and $b _ { z }$ is the edge-count bucket, defined as the edge count capped at six.

ChronoSynth stores each record in three complementary indexes: a rule-aware primary bucket index, a relation-pattern index, and a graph-shape index. A bounded global pool supplies fallback candidates. During retrieval, the primary index supplies exact temporal matches, while the relation and graph-shape indexes provide additional candidates for compatibility ranking. The fallback pool is used when the combined candidate pool is too small.

## A.3 Offline Memory Construction

The memory is built once from the training split. For each record with a reference question, we construct its Temporal Intent State, normalize entity and time mentions into typed placeholders, and store the question, pattern, and retrieval metadata. Records are inserted into temporal, relation, and graph-shape indexes and a bounded fallback pool. The same memory is reused by the online procedure in Algorithm 1; generation and feedback refinement do not modify it.

## A.4 Retrieval Features and Settings

Candidates are collected from the three indexes. If fewer than 24 candidates are available, the fallback pool supplies additional candidates. Each index bucket stores at most 192 records, and the fallback pool stores at most 512. Candidates are deduplicated by question ID and ranked using the following expansion of Eq. 4:

$$
\begin{array} { r l } & { \mathrm { s c o r e } ( m _ { z } , S _ { x } ) = 2 I _ { d } + 4 I _ { \omega } + 1 . 5 I _ { f } } \\ & { \qquad + 3 I _ { a } + 2 . 5 I _ { y } + 1 . 5 I _ { g } } \\ & { \qquad + 3 . 5 I _ { r } + 2 I _ { \phi } } \\ & { \qquad + \delta _ { \mathrm { e d g e } } + \delta _ { \mathrm { a n c h o r } } + \delta _ { \mathrm { p h } } . } \end{array}\tag{8}
$$

Here $I _ { d } , I _ { \omega } , I _ { a } , I _ { y } , I _ { g } , I _ { r } , I _ { \phi }$ indicate agreement in dataset dialect, temporal rule, answer field, answer type, granularity, relation pattern, and graph shape. $I _ { f }$ is one only when the temporal rules differ but share the same prefix before an underscore; it is zero for exact rule matches. Thus, exact and partial rule agreement are not counted together. The additional terms are

$$
\delta _ { \mathrm { e d g e } } = 1 / ( 1 + | b _ { z } - b _ { x } | ) ,\tag{9}
$$

$$
\delta _ { \mathrm { a n c h o r } } = 0 . 4 \operatorname* { m i n } ( | H _ { z } \cap H _ { x } | , 3 ) ,\tag{10}
$$

$$
\delta _ { \mathrm { p h } } = 0 . 2 \operatorname* { m i n } ( P _ { z } , 4 ) ,\tag{11}
$$

where b denotes the edge-count bucket, H contains normalized anchor tokens, and $P _ { z }$ counts typed placeholders. These are fixed implementation weights. Edge-count bucket and string-form question ID break score ties. Each prompt retains four memory records and at most four valid adapted drafts. Rule-specific drafts supplement insufficient valid adaptations.

## A.5 Validation and Feedback Refinement

Initial checks cover question format, unresolved placeholders, literal answer disclosure, temporal cues, and the requested answer field. Cleaning, repair, or draft fallback produces the base question. The interpreter then receives only this question and its local graph, without reference questions, the target answer list, or the inferred state. Its recovered program is executed and compared with the target answers and explicit input constraints.

Program representation. The program schema consists of match, filter, extremum, project, time\_boundary, and precision. These fields specify which facts to match, which temporal conditions to apply, and which answer field to return. Malformed programs, ambiguous references, and unsupported operations receive an unverified status.

Temporal semantics. Temporal values denote closed periods at their stated precision. Before requires the fact’s end to precede the reference period’s start; after requires its start to follow the reference period’s end. Explicit-date equality uses interval intersection. An event reference must resolve to a distinct period. First and last select the earliest and latest applicable endpoints and retain ties. Projection cannot return finer precision than the source provides. Open-ended and invalid intervals are unsupported.

Correction policy. Correction search can change filter direction, first/last selection, interval boundary, or projected time precision. It changes at most two supported temporal fields while keeping entity and relation bindings fixed. Feedback refinement selects a feasible program only if it uniquely minimizes the number of field changes. Executionchecked surface rules may also restore an explicit input date scope or revise first/last wording.

If no surface rule applies, the original generator proposes a rewrite. A second interpretation must recover a program equivalent to the selected target. Its execution must return the target answer set and satisfy supported explicit constraints, and the question must pass format and answer-disclosure checks. Failure of any check leaves the initial question unchanged. Excluding retries, feedback refinement uses at most two interpretations and one model rewrite per input.

Verification criterion. Entity bindings, temporal references, extrema, and interval-boundary cues must be grounded in the question. Equivalence is defined over the program schema: equal answers on one graph do not establish equivalent programs. Acceptance therefore requires both realization of the selected correction and consistency with the execution checks.

## A.6 Complexity and Reuse

For N training records, index construction takes $O ( N { \bar { c } } )$ time, where c¯ is the average stateconstruction and pattern-normalization cost. Storage is linear in the total size of retained records and postings. Ranking C candidates takes $O ( C \log C )$ time; adapting k selected patterns takes $O ( k L )$ where L bounds pattern length. These costs exclude language-model inference and program execution. The memory is built once and reused across inputs, while verification requires at most two interpretations and one model rewrite per question, excluding retries.

<table><tr><td>Dataset</td><td>Train</td><td>Dev</td><td>Test</td></tr><tr><td>CRONQUESTIONS</td><td>212,259</td><td>19,014</td><td>19,122</td></tr><tr><td>MultiTQ</td><td>257,687</td><td>39,128</td><td>36,318</td></tr></table>

Table A3: Numbers of graph-converted records.

## B Data and Experimental Settings

## B.1 Benchmark Preparation

We use the graph-converted splits of CRONQUES-TIONS and MultiTQ. The conversion constructs local graph evidence from the source QA records. Generation receives graph facts, target answers, and explicit constraints encoded in the input. Reference wording and source annotations are removed from the generation input. Reference questions are retained separately for evaluation.

## B.2 Generation and Scoring

GPT-4o-mini and Qwen3.5-9B generate and rewrite questions. Both main-table configurations use GPT-4o-mini for program interpretation; the separate efficiency experiment uses only local Qwen inference. GPT-4o-mini runs use temperature zero, a 900-token response limit, and up to 50 concurrent requests. Local Qwen generation is greedy with thinking disabled. Cached responses are reused only when the prompt and model configuration match.

Baseline scores come from the corresponding benchmark runs. RoleAgentQG-G2S denotes the non-RL G2S-AE variant, whose saved predictions are scored separately on each dataset. BLEU uses corpus-level n-gram precision and brevity penalty. ROUGE-L averages question-level scores with β = 1.2. Both use lowercased whitespace tokens. CIDEr uses case-preserving input and its native scale. All test records remain in evaluation, including unresolved outputs.

## B.3 Resources and Intended Use

We use the graph-converted CRONQUESTIONS and MultiTQ benchmarks for English temporal question generation. Each input contains local graph facts, target answers, and temporal constraints. The supplements provide data formats, split statistics, implementation, and evaluation records for research on question synthesis and temporal question answering. Table A4 identifies the main external resources.

The Qwen model cards specify Apache 2.0 for the two local models. Source datasets remain subject to their original distribution terms, and the supplements preserve source attribution and thirdparty notices. Model weights are not redistributed. These notices do not grant additional rights over source data or hosted models.

<table><tr><td>Resource</td><td>Use</td><td>Source</td></tr><tr><td>CRONQUESTIONS</td><td>temporal mark</td><td>bench- CronKGQA (Sax- ena et al., 2021)</td></tr><tr><td>MultiTQ</td><td>temporal mark</td><td>bench- MultiTQ (Chen et al., 2023)</td></tr><tr><td>Qwen3.5-9B Qwen3.5-0.8B</td><td>local generation downstream</td><td>Qwen model card Qwen model card</td></tr><tr><td>GPT-4o-mini</td><td>reader generation and in- OpenAI terpretation</td><td></td></tr><tr><td>GPT-4.1-mini</td><td>supplementary judge</td><td>OpenAI</td></tr><tr><td>BLEU, CIDEr</td><td>ROUGE-L, reference scoring</td><td>software supplement</td></tr></table>

Table A4: External resources and their experimental roles.

Software environment. Downstream reader experiments use PyTorch 2.5.1 and Transformers 5.12.1. Local generation uses vLLM 0.23.0 with Qwen3.5-9B. The software supplement provides dependency requirements, the model configuration, and the scoring implementation.

## C Ablation and Grouped Results

## C.1 Ablation Configurations

The complete method and three variants use the same complete test splits and GPT-4o-mini for generation, interpretation, and rewriting. Outputs are aligned by test position and question ID so that repeated IDs remain distinct. The w/o Refinement variant retains initial validation and omits feedback refinement. Relation-only Retrieval uses relationbased candidate selection and ranking instead of temporal compatibility. The w/o Pattern Memory variant omits retrieved examples but retains state construction and default drafts. The latter two variants retain feedback refinement.

## C.2 Performance across Input Groups

We regroup the saved final GPT-4o-mini outputs used in the main table and ablation figure. Groups use the temporal rule inferred from the generation input, its edge count, and the answer type. For each group, BLEU-4 is recomputed at corpus level and ROUGE-L is averaged over its questions. Tables A5 and A6 report every observed group; no new generation is performed. Positive differences favor the complete method over w/o Pattern Memory.

<table><tr><td>Group</td><td>Count ∆BLEU-4 ∆ROUGE-L</td><td></td></tr><tr><td>Temporal rule</td><td></td><td></td></tr><tr><td>after_first</td><td>1,494</td><td>+0.2253 +0.3688</td></tr><tr><td>before</td><td>27 +0.2256</td><td>+0.2837</td></tr><tr><td>before_last</td><td>1,907 +0.1861</td><td>+0.1670</td></tr><tr><td>first</td><td>1,030</td><td>+0.0928 +0.1244</td></tr><tr><td>last</td><td>879 +0.1095</td><td>+0.1355</td></tr><tr><td>plain</td><td>7,877 +0.1156</td><td>+0.1974</td></tr><tr><td>when</td><td>5,908 +0.0473</td><td>+0.0445</td></tr><tr><td>Edge count</td><td></td><td></td></tr><tr><td>1</td><td>12,376 +0.0739</td><td>+0.1324</td></tr><tr><td>2</td><td>6,745 +0.1632</td><td>+0.1934</td></tr><tr><td>3</td><td>1 +0.5157</td><td>+0.3049</td></tr><tr><td></td><td></td><td></td></tr><tr><td>Answer type</td><td>11,502</td><td></td></tr><tr><td>entity time</td><td>+0.1460 7,620 +0.0457</td><td>+0.2149 +0.0618</td></tr></table>

Table A5: Grouped gains on CRONQUESTIONS. Differences are complete ChronoSynth minus w/o Pattern Memory.

The complete method improves both metrics in every reported group. On CRONQUESTIONS, the BLEU-4 gain is 0.1460 for entity answers and 0.0457 for time answers. On MultiTQ, the corresponding gains are 0.1382 and 0.1708. Improvements therefore extend across answer types and graph sizes. Counts are reported because small groups, especially the single three-edge CRON-QUESTIONS record, do not support general conclusions on their own.

## D Human Evaluation Protocol

The main-text human evaluation uses matched inputs and a shared annotation rubric. Details of sampling, annotation, and aggregation follow.

Participants. Three volunteers were recruited through a fair process and received reasonable compensation for the annotation work. Participation was voluntary, and all volunteers consented to participate in the evaluation.

Annotation instructions. Raters assess each question using only the displayed graph facts, target answers, and expected temporal conditions. They do not consult reference questions, infer the generation method, or use outside information. Fluency, temporal faithfulness, and answerability are scored independently from 1 to 5 using Table A7. Answerability is assessed against the complete target answer set, and literal answer disclosure lowers the score. The interface accepts three successive scores per item and saves progress when the rater exits. The software supplement includes the English annotation guide.

<table><tr><td>Group</td><td>Count ∆BLEU-4 ∆ROUGE-L</td><td></td></tr><tr><td>Temporal rule</td><td></td><td></td></tr><tr><td>after</td><td>675</td><td>+0.1627 +0.1454</td></tr><tr><td>after_first</td><td>2,784 +0.1168</td><td>+0.1160</td></tr><tr><td>after_time</td><td>1,331</td><td>+0.1152 +0.0948</td></tr><tr><td>before_last</td><td>8,620</td><td>+0.0898 +0.0699</td></tr><tr><td>before_time</td><td>1,441</td><td>+0.1162 +0.0906</td></tr><tr><td>first</td><td>6,293</td><td>+0.2526 +0.1955</td></tr><tr><td>last</td><td>3,697</td><td>+0.2407 +0.1897</td></tr><tr><td>plain</td><td>4,890</td><td>+0.2165 +0.1451</td></tr><tr><td>when</td><td>6,587</td><td>+0.1100 +0.0610</td></tr><tr><td>Edge count</td><td></td><td></td></tr><tr><td>1</td><td>5,201 +0.2069</td><td>+0.1367</td></tr><tr><td>2</td><td>14,467 +0.1560</td><td>+0.1271</td></tr><tr><td>3</td><td>13,470 +0.1354</td><td>+0.1116</td></tr><tr><td>4</td><td>2,011 +0.1170</td><td>+0.0862</td></tr><tr><td>5</td><td>1,169 +0.0973</td><td>+0.0816</td></tr><tr><td>Answer type</td><td></td><td></td></tr><tr><td>entity</td><td>23,608 +0.1382</td><td>+0.1115</td></tr><tr><td></td><td></td><td></td></tr><tr><td>time</td><td>12,710 +0.1708</td><td>+0.1330</td></tr></table>

Table A6: Grouped gains on MultiTQ. Differences are complete ChronoSynth minus w/o Pattern Memory.

Privacy and use. Source entity names are retained because they are required to interpret the graph facts and generated questions. The anonymous supplements omit annotator names and private configuration information. Generated questions should be reviewed before use in sensitive applications.

Sampling and annotation. Using seed 20261005, we sampled 50 matched test inputs per dataset. Sampling preserves source-question-type proportions among inputs with available, uniquely aligned predictions from every method. Each input is paired with GPT-4o-mini questions from direct prompting, adapted KQG-CoT, retrieval-augmented prompting, and ChronoSynth.

Three raters independently assign scores from 1 to 5 for fluency, temporal faithfulness, and answerability. Each rater receives a separately randomized order with method labels hidden. Items display graph facts, target answers, and temporal conditions derived from source QA metadata, but no reference question. The displayed conditions are independent of the framework’s inferred state. Fluency concerns grammatical clarity, temporal faithfulness concerns preservation of the displayed condition, and answerability concerns support for the complete target answer set. Literal answer disclosure is penalized under answerability.

<table><tr><td>Score</td><td>Fluency</td><td>Temporal faithfulness</td><td>Answerability</td></tr><tr><td>5</td><td>Natural, clear, and grammatical</td><td>All required temporal conditions are preserved</td><td>Clear question with the correct complete target answer set</td></tr><tr><td>4</td><td>Minor language issues</td><td>Minor issue that does not materially change the temporal meaning</td><td>Minor ambiguity</td></tr><tr><td>3</td><td>Understandable but noticeably awkward</td><td>Temporal intent is recognizable but incomplete or ambiguous</td><td>Meaningful ambiguity</td></tr><tr><td>2</td><td>Major language problems</td><td>An important temporal component is wrong or missing</td><td>Poorly answerable or likely to yield another answer</td></tr><tr><td>1</td><td>Incoherent or invalid question</td><td>Core temporal meaning is reversed, lost, or replaced</td><td>Unanswerable, wrong target, or invalid question</td></tr></table>

Table A7: The annotation rubric used in human evaluation.

Retrieval comparator. The retrieval comparator uses archived predictions from a generic prompt with four training graph–question demonstrations. Relation-token Jaccard similarity ranks the examples, and question ID breaks ties. It uses no temporal state, typed binding, or feedback refinement. The downstream retrieval baseline is generated separately from its specified pool and tie-breaking rule. Human scores are aligned to the exact question text shown to the raters.

Each rater assesses the same 100 ChronoSynth questions and 300 baseline questions. For a given input, all methods share the displayed graph, answers, temporal condition, and rubric, allowing paired comparisons on the same evidence.

Aggregation and uncertainty. We first average the three ratings for each question, then average questions within each dataset. The overall score weights the three dimensions equally. Paired method differences use 10,000 bootstrap resamples of source inputs and their averaged ratings, with seed 20261005. These are pointwise 95% intervals without adjustment for multiple comparisons. ChronoSynth’s mean advantages range from 0.47 to 0.69 points. All 18 intervals have positive lower bounds, the smallest being 0.353.

## E Downstream Question Answering Protocol

Each condition uses 1,000 matched training inputs with fixed graphs and answer sets. Synthetic-data conditions use generated questions only; Gold Reference uses the original questions. All conditions share the reader, training budget, and evaluation protocol.

Synthesis pools. We sample the synthesis inputs and a disjoint memory pool of 2,000 training records with seed 20261004. Preparation excludes exact held-out question strings and IDs. Retrieval methods share the memory pool. KQG-CoT receives four structurally selected demonstrations and preserves complete answer sets.

Vanilla RAG selects four demonstrations by relation-token Jaccard similarity, breaking ties by memory position. Each demonstration contains a graph and its training question. A generic prompt adds the current graph and target answers and requests one answerable question without answer disclosure. It uses no temporal state, typed adaptation, validation, or feedback refinement. GPT-4o-mini generates one question per input at temperature 0 with a 900-token limit. Retries address failed calls or mismatched returned model identities, not output quality.

ChronoSynth uses GPT-4o-mini for generation, program interpretation, and rewriting, retaining unresolved outputs. Corpus audits check that source IDs, graphs, and answer labels match across methods. Shared training resources supply memory, and shared validation data supports checkpoint selection.

Reader optimization. All conditions fine-tune the text model of Qwen/Qwen3.5-0.8B with thinking disabled. The prompt contains only the question and graph; it excludes answer labels, templates, and specification fields. Prompt tokens are masked from the loss, and supervision consists of the complete answer set separated by |. We use float32 parameters and AdamW states with BF16 forward computation. The learning rate is $1 \times 1 0 ^ { - 5 }$ , weight decay 0.01, effective batch size 32, microbatch size 8, and gradient clipping 1.0. Learning rate decays linearly after 10% warmup. The trained reader is distinct from the feedback refinement interpreter.

Selection and decoding. Each condition is trained for twelve epochs with seeds 42, 43, and 44. We select the checkpoint with the highest EM on a shared set of 1,000 original validation records. Lower validation cross-entropy and then an earlier epoch resolve ties. Input and output limits are 1,024 and 128 tokens, with truncation disallowed. Test decoding is greedy and uses explicit chat and text termination tokens. Five conditions across two datasets and three seeds give 30 runs with equal example and optimizer-step budgets.

Scoring. Evaluation covers all 19,122 original CRONQUESTIONS and 36,318 MultiTQ test records. EM requires equality of complete normalized answer sets. F1 averages per-question F1 computed from answer-set precision and recall. Normalization changes case and whitespace while preserving aliases and date precision. We report means and sample standard deviations over three seeds. Every reader answers from supplied local graphs without global retrieval.

## F Supplementary Model-Based Assessment

A separate GPT-4.1-mini judge compares the complete method with w/o Pattern Memory on 1,000 matched inputs per dataset, stratified by sourcequestion type with seed 20261008. Both generators use GPT-4o-mini and retain feedback refinement. Method labels are hidden, A/B order is balanced, and each pair receives three independent calls. The judge sees graph evidence, target answers, temporal conditions, and the two questions, but not reference questions.

We average judgments per input and compute paired differences with 10,000 bootstrap resamples. Temporal-faithfulness and answerability gains are 0.474 [0.381, 0.570] and 0.586 [0.485, 0.690] on CRONQUESTIONS, respectively. On MultiTQ, the differences are −0.165 [−0.245, −0.086] and −0.159 [−0.243, −0.077]. Thus, memory improves reference-based generation metrics across both datasets, while this judge’s semantic ratings vary by dataset. This comparison uses different samples and a different evaluator from the human study; its scores are not interchangeable with human ratings.

## G Local Synthesis Efficiency

Local-model configuration. We run ChronoSynth, KQG-CoT, Vanilla RAG, and direct prompting on all 19,122 CRONQUES-TIONS and 36,318 MultiTQ test inputs. Each method uses Qwen3.5-9B on one H100 GPU with vLLM 0.23.0, BF16 weights, 40 concurrent workers, temperature 0, and a 900-token output limit per request. Thinking and prefix caching are disabled. For ChronoSynth, generation, program interpretation, and model-based rewriting all use the local model. This experiment measures the complete local pipeline; the main generation table retains its GPT-4o-mini interpretation configuration.

KQG-CoT uses 12 structure-clustered training demonstrations. Vanilla RAG retrieves four demonstrations from the complete training pool using relation-token Jaccard similarity; ties follow training-memory order. ChronoSynth uses the same training pool for its state-guided pattern memory and retains feedback refinement. All methods receive the same graph evidence and target answers. We retain every output, including five empty KQG-CoT predictions on MultiTQ, when computing reference metrics.

Measurement. Runtime covers prompt construction and all inference requests, including feedback refinement and retries, after preprocessing and a representative warmup batch. Model startup, data loading, memory construction, and retrieval-index preparation are excluded. Throughput divides the number of test inputs by this runtime. Tokens per question include input and output tokens from all requests. Each method has one complete measured run, so figures report measured values without repeated-run error bars. ChronoSynth’s preprocessing takes 28.13 seconds on CRONQUESTIONS and 43.57 seconds on MultiTQ.

Recorded inference cost. The eight complete local efficiency runs on CRONQUESTIONS and MultiTQ require approximately 4.68 H100 GPUhours in total. The four ChronoQG runs require approximately 1.64 H100 GPU-hours. Each run uses one GPU. These figures cover the measured inference stages, excluding model startup, preprocessing, and warmup; they do not include reader training, hosted-model calls, or development runs.

Quality and cost. Figure A1 compares generation quality and cost across the four methods. ChronoSynth achieves the highest scores on all six reference metrics in this experiment. Its BLEU-4 is 0.2993 on CRONQUESTIONS and 0.3141 on

![](images/e9eef004f0c290e23e0333910b6b4d3340e3b9a5348e31befeaa2f25de407dff.jpg)  
F: ChronoSynth C: KQG-CoT R: Vanilla RAG D: Direct  
Figure A1: Generation quality, throughput, and token use with local Qwen3.5-9B.

MultiTQ, compared with 0.2108 and 0.1811 for KQG-CoT. ChronoSynth completes synthesis in 26.19 and 53.17 minutes, whereas KQG-CoT requires 33.05 and 120.13 minutes. Corresponding throughput is 12.17 and 11.38 questions per second for ChronoSynth, and 9.64 and 5.04 for KQG-CoT.

Figure A2 shows the quality–token trade-off. ChronoSynth uses 1,316 and 1,481 tokens per question, compared with 1,916 and 3,653 for KQG-CoT. This includes an average of 2.12 and 2.18 model requests per question for ChronoSynth, versus one for each baseline. Direct prompting and Vanilla RAG require fewer tokens and less runtime, but produce lower BLEU-4. Thus, the efficiency advantage is relative to KQG-CoT, while the quality advantage extends to all three baselines evaluated here.

Output budget. Under the common 900-token output limit, KQG-CoT reaches the limit in 126 requests on CRONQUESTIONS and 1,415 on MultiTQ; ChronoSynth does so in zero and seven requests. These outputs remain in evaluation. The comparison therefore measures each method under the stated common decoding budget.

## H Additional Evaluation on ChronoQG

Dataset and split. ChronoQG-AB85 contains 15,186 temporal graph–question records from CronKG and EventKG, with single- and multipleconstraint questions. Each record supplies graph facts, temporal constraints, a target answer, and a natural-language reference question. As the release has no official training/test split, we use a fixed, approximately 80/20 split with seed 20261011. Records sharing a native source instance or an identical task input are assigned to the same partition, with stratification by source file and temporalconstraint codes. Table A8 reports the resulting 12,159 training and 3,027 test records.

<table><tr><td>Source subset</td><td>Train</td><td>Test</td></tr><tr><td>Cron-S</td><td>4,172</td><td>1,041</td></tr><tr><td>Cron-M</td><td>1,290</td><td>309</td></tr><tr><td>Event-S</td><td>4,955</td><td>1,236</td></tr><tr><td>Event-M</td><td>1,742</td><td>441</td></tr><tr><td>Total</td><td>12,1593,027</td><td></td></tr></table>

Table A8: ChronoQG data split. S/M denote single/multiple temporal constraints.

Adaptation and setting. All four methods receive the same graph facts, target answers, and native temporal conditions. The input adapter retains fact identifiers, entity/relation bindings, point/interval types, and time granularity. For ChronoSynth, native constraint codes and their conjunctions populate the Temporal Intent State and pattern-memory indexes. Retrieval weights, memory caps, and the generation procedure are unchanged. The validation adapter checks answerrole projections and grounded temporal relation pairs for supported native constraints; unresolved cases retain their initial questions.

![](images/a5b6d45c70ba11f4d739fe0e608dfbfe438223ba2062bca1b67935bb7439e720.jpg)

![](images/22162010bc9b7d92e7339956f9c0e238d7ece5996b302d6d4a32752f8cbd0717.jpg)

Figure A2: BLEU-4 versus total tokens per question. Arrows connect KQG-CoT to ChronoSynth.
<table><tr><td>Method</td><td>BLEU-1 BLEU-2 BLEU-3</td><td></td><td></td><td></td><td>BLEU-4 ROUGE-L</td><td>CIDEr</td></tr><tr><td>Direct Prompting</td><td>0.4710</td><td>0.3836</td><td>0.3220</td><td>0.2761</td><td>0.4625</td><td>2.3437</td></tr><tr><td>KQG-CoT</td><td>0.3987</td><td>0.3339</td><td>0.2892</td><td>0.2557</td><td>0.4339</td><td>2.6055</td></tr><tr><td>Vanilla RAG</td><td>0.5625</td><td>0.4832</td><td>0.4271</td><td>0.3834</td><td>0.57113.4138</td><td></td></tr><tr><td>ChronoSynth</td><td>0.6201</td><td>0.5495</td><td>0.4979</td><td>0.4562</td><td></td><td>0.6472 4.3373</td></tr></table>

Table A9: Question generation on ChronoQG with local Qwen3.5-9B. Bold marks the highest score.

Direct prompting and KQG-CoT reuse their generation recipes, with KQG-CoT using 12 structurally selected training demonstrations. Vanilla RAG retrieves four examples from the complete training partition by relation-token Jaccard similarity. ChronoSynth builds its pattern memory from the same partition. Every stage uses local Qwen3.5- 9B with vLLM 0.23.0, BF16 weights, one H100 per method, and 40 workers. Temperature is zero, the per-request output limit is 900 tokens, and thinking and prefix caching are disabled. Reference questions are used for training examples and offline scoring, and are excluded from test-generation inputs.

Generation results. Table A9 reports BLEU-1/2/3/4, ROUGE-L, and CIDEr using the same scoring implementation as the main experiments. Scores cover all 3,027 test records, including two empty KQG-CoT outputs. ChronoSynth leads the three baselines on all six reference-based metrics. Its BLEU-4 of 0.4562 exceeds Vanilla RAG’s 0.3834 by 19.0%, while ROUGE-L and CIDEr reach 0.6472 and 4.3373. These results extend the generation-quality gains to questions from two source graphs with richer explicit temporal constraints.

<table><tr><td>Method</td><td>Time</td><td>Tokens (min) /question</td><td>Questions /second</td></tr><tr><td>Direct Prompting</td><td>2.53</td><td>1,333</td><td>19.91</td></tr><tr><td>KQG-CoT</td><td>37.82</td><td>14,106</td><td>1.33</td></tr><tr><td>Vanilla RAG</td><td>6.49</td><td>3,682</td><td>7.77</td></tr><tr><td>ChronoSynth</td><td>51.65</td><td>5,012</td><td>0.98</td></tr></table>

Table A10: Measured inference cost on ChronoQG.

Inference cost. Table A10 reports one complete measured run per method. Runtime excludes model startup, memory preparation, and warmup; token counts include input and output tokens from every model request. ChronoSynth uses 5,012 tokens per question, compared with 14,106 for KQG-CoT, a reduction of 64.5%. Its additional interpretation and refinement requests yield an average of 2.36 calls per question and a runtime of 51.65 minutes. Direct prompting and Vanilla RAG are faster, while ChronoSynth obtains stronger reference-based generation scores.