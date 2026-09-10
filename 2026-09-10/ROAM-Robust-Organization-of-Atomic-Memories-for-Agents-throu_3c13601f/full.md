# ROAM: Robust Organization of Atomic Memories for Agents through Semantic Relations

Jianjie Zheng<sup>1∗</sup>, Peng Lai<sup>1</sup>\*, Sijie Cheng<sup>2,3</sup>, Jiehui Zhao<sup>4</sup>, Lei Yang<sup>4</sup>, Guanhua Chen<sup>1†</sup> <sup>1</sup>Southern University of Science and Technology, <sup>2</sup>Tsinghua University <sup>3</sup>RayNeo.AI, <sup>4</sup>Deepexi Technology Co. Ltd.

## Abstract

Long-term language-model agents rely on external memory across interactions. Atomic memories are particularly useful: their finegrained semantic boundaries enable precise retrieval and direct comparison between observations. Yet accumulating atoms inevitably become redundant, overlapping, or conflicting. Existing methods often ask an LLM manager to add, update, delete, or rewrite memories directly, coupling semantic interpretation, storage decisions, and content generation in one error-prone operation. We introduce ROAM, a relation-guided framework that uses atomicity for management while allowing richer answer-time representations. ROAM classifies incoming–stored atom pairs as independent, equivalent, directionally subsuming, or conflicting, then organizes observations into active Primary and supporting Evidence roles. Fusion subsequently combines complementary details and temporal changes into compact, potentially non-atomic views. Only Primary views are retrieved for answering, preventing redundant or outdated atoms from competing independently. Across models and evaluation settings, ROAM improves answer accuracy by up to 29.8 percentage points. Ablations show complementary benefits from different relations and consistent gains from fusion beyond role organization. Mechanism analysis further finds 15.6-point higher answer-critical source recall and an 11.5- point lower confounder-token share. ROAM remains robust across manager scales.

## 1 Introduction

Long-lived language-model agents must continuously accumulate user preferences, circumstances, and experiences across sessions while seamlessly revising that knowledge as real-world states change (Sun et al., 2026; Zhong et al., 2024).

![](images/6931223d93a1f53a162d06fcacb4f8f5bebcdf8b52f8bcf1be85272e6c8d0ebc.jpg)  
Figure 1: Motivation for ROAM. Existing methods ask the model to directly choose a memory operation for each old–new memory pair, making the task complex and the results unreliable. ROAM first identifies the semantic relations between memories, then uses them to guide memory organization and fusion, making memory management more controllable and accurate.

Because raw conversational histories rapidly exceed an agent’s working context limit, contemporary architectures maintain external memory stores to retrieve relevant historical subsets at query time (Zhong et al., 2024; Maharana et al., 2024; Tan et al., 2025). To maximize retrieval precision under strict context budgets, systems increasingly represent durable knowledge as fine-grained, atomic facts or structured relations (Modarressi et al., 2024; Gutiérrez et al., 2024; Chhikara et al., 2025; Shen et al., 2026). Compared to storing full dialogue transcripts, atomic representations preserve key salient information in significantly fewer tokens, provide precise addressable units, and enable targeted updating. Consequently, atomization has become the standard representational foundation for long-term memory systems.

However, atomization specifies only the granularity of memory—it does not govern how accumulating atomic facts should coexist over time as information evolves. Continuous streaming of atomic observations inevitably introduces semantic restatements, spatial or attribute refinements, and temporal state transitions. Merely appending every observation creates severe redundancy and ambiguity, while indiscriminate deletion or overwriting destroys the historical context needed to trace knowledge evolution. Crucially, uncurated, overlapping, and outdated records directly compete with answer-critical evidence for limited retrieval capacity, degrading downstream response quality (Liu et al., 2024; Amiraz et al., 2025). Existing approaches attempt to maintain growing memory stores through imperative, direct-operation mechanisms (e.g., predicting actions like ADD, UPDATE, or DELETE) (Chhikara et al., 2025). Yet, such paradigms force the model to infer underlying semantic relations, map them to physical storage mutations, and coordinate updates in a single, blackbox compound decision. Conflating semantic interpretation with physical storage execution makes operational consequences unstable and hinders historical traceability. Instead of relying on end-toend storage mutations, effective atomic memory management requires decoupling explicit semantic relation inference from deterministic state control.

We introduce ROAM (Robust Organization of Atomic Memories for Long-Lived Agents through Semantic Relations), a framework that factorizes this compound decision into explicit relation inference followed by deterministic policy execution. For each incoming observation, ROAM predicts one of five fine-grained semantic relations (IND, EQV, OSN, NSO, CON) against existing memories. A fixed policy then assigns memories into structural roles—promoting the most informative fact to Primary status while retaining supporting or superseded facts as Evidence. Finally, a fusion module synthesizes compatible details into a compact read view for each Primary entry, allowing downstream retrieval to search exclusively over Primary views. By replacing black-box mutations with relation-guided control, ROAM eliminates redundancy while preserving historical provenance for optimal query answering.

We evaluate ROAM in three complementary settings: a controlled intervention that varies semantic and temporal competition, the original longhorizon histories in LongMemEval (Wu et al., 2025), and post-change questions involving multiple entities in MEME-Post (Jung et al., 2026). Across three manager models, ROAM achieves the highest mean accuracy at every nonzero level of added competition and on full LongMemEval, while consistently outperforming model-based management baselines on MEME-Post. Relation and fusion ablations show that equivalence, directional containment, conflict, and fused representations provide complementary gains; retrieval diagnostics further link ROAM’s robustness to greater coverage of answer-critical memories. Together, these results show that explicit relation-guided organization helps long-term memory systems sustain answer quality as semantic and temporal competition increases.

## 2 Method

## 2.1 Problem Setup and Scope

In multi-turn, long-term agent interaction scenarios, the fundamental goal of a memory system is to answer downstream queries by delivering relevant history within a tight context budget B. To achieve this, the system operates through a structured three-stage pipeline consisting of Extraction, Management, and Retrieval:

$$
\begin{array} { r l } & { { \mathscr F } _ { t } = \mathrm { E x t r a c t } ( D _ { t } ) , } \\ & { { \mathscr S } _ { t } = \mathrm { M a n a g e } ( S _ { t - 1 } , { \mathscr F } _ { t } ) , } \\ & { { \mathscr K } _ { q } = \mathrm { R e t r i e v e } ( q , S _ { t } ; B ) . } \end{array}\tag{1}
$$

Specifically, given raw interaction $D _ { t }$ at turn t, the Extractor distills raw text into unstructured factual statements $\mathcal { F } _ { t } ;$ the Manager updates the structured memory state $S _ { t } .$ <sub>−1</sub> into $S _ { t }$ by integrating $\mathcal { F } _ { t } ;$ and the Retriever selects the most relevant subset ${ \mathcal { K } } _ { q } \subseteq$ $S _ { t }$ to satisfy downstream query $q$ under context budget B.

To meet the high-precision retrieval demands of downstream tasks, raw dialogue turns or coarse summaries are insufficient. This scenario naturally mandates representing extracted facts $\mathcal { F } _ { t }$ as finegrained, atomic observations:

$$
O _ { t } = ( a _ { t } , \tau _ { t } , c _ { t } ) ,\tag{2}
$$

where $a _ { t }$ represents a self-contained factual statement, $\tau _ { t }$ is its timestamp, and $c _ { t }$ denotes its source context. Breaking down complex interactions into standalone atomic units provides the necessary finegrained entry point for flexible memory operations.

However, the streaming influx of atomic observations creates severe challenges for the Management stage. Over time, newly incoming atomic entries frequently repeat known facts, introduce partial details, or invalidate past states. Without rigorous management, these uncurated observations rapidly accumulate, allowing redundant and outdated entries to crowd out valuable knowledge during downstream retrieval. This motivates us to optimize the intermediate Management stage, with the goal of consolidating incoming observations while preserving their historical context and provenance.

![](images/d08b096a61922621bca46137bf7462410b6f30a253a704a9c6e381936e31a644.jpg)  
Figure 2: Overview of the ROAM Framework. ROAM separates memory management from answer-time retrieval. Incoming atoms are compared with Primary atomic texts and assigned Primary or Evidence roles through five-way relation inference. Fusion constructs a read view for each surviving Primary; answer-time retrieval searches only these views, while Evidence remains preserved but inactive.

## Remark

In long-term agent interactions, memory management need not occur synchronously after every observation. Atomic observations can be consolidated asynchronously or periodically in the background, allowing the Management stage to prioritize memory compactness over minimal update latency.

## 2.2 ROAM: Semantic-Relation-Based Memory Management

Traditional memory management mechanisms commonly operate through imperative storage actions (e.g., ADD, UPDATE, DELETE). However, applying imperative mutations directly to incoming atomic observations creates a dilemma between accepting redundant storage or suffering permanent information loss from overwriting prior entries (Chhikara et al., 2025). This raises a critical question: How can we efficiently consolidate atomic observations to remove redundancy and track knowledge evolution without sacrificing historical provenance or context? To address this challenge, we propose ROAM, a relation-guided memory management framework that decouples semantic relation modeling from physical storage control. For any incoming observation, ROAM first infers its precise semantic relation against existing memories across five predefined categories. Based on the resolved relation, a deterministic policy reorganizes memories into Primary (active) and Evidence (supporting) entries, preserving historical provenance while preventing redundant accumulation. Finally, ROAM fuses compatible details into a unified read view for each Primary entry, allowing downstream retrieval to query over a compact, highly dense memory state within budget B. Figure 2 provides an overview of the proposed framework, while the detailed streaming procedure is formalized in Algorithm 1.

## 2.2.1 Semantic Relations between Atomic Memories

Let $\mathcal { P } _ { t - 1 }$ be the active atomic memories eligible for management-time retrieval. For an incoming observation $O _ { t } ,$ , ROAM retrieves candidate memories using their atomic text:

$$
\mathcal { C } _ { t } = \rho _ { k } ( a _ { t } , \mathcal { P } _ { t - 1 } ) ,\tag{3}
$$

where $\rho _ { k }$ returns the k most relevant Primary memories. A relation model then predicts the relation of each ordered pair $( a _ { t } , a _ { i } )$

$$
\boldsymbol { r } _ { t } ( i ) = \Phi ( \boldsymbol { a } _ { t } , \boldsymbol { a } _ { i } ) \in \mathcal { R }\tag{4}
$$

where $\mathcal { R } = \left\{ \mathrm { I N D , E Q V , O S N , N S O , C O N } \right\}$ , IND (independent) means that both observations can be true and neither entails the other; EQV (equivalent) means that they have the same truth conditions and mutually entail one another; OSN (old subsumed by new) means that the incoming observation strictly entails the old one and adds specificity; for example, “The user lives in Cambridge, Massachusetts” entails “The user lives in Massachusetts”; NSO (new subsumed by old) is the reverse entailment direction and adds no specificity; Finally, CON (conflict) means that, after entity, attribute, and temporal scope are aligned, the observations describe incompatible states. Explicit temporal qualifiers take precedence when resolving such conflicts.

## 2.2.2 Relation-Conditioned Memory Consolidation

The predicted relation determines a management action but never directly overwrites a source observation. ROAM stores each observation as an immutable record with a management role and an answer-time read view:

$$
M _ { i } = ( a _ { i } , v _ { i } , \tau _ { i } , c _ { i } , z _ { i } ) ,\tag{5}
$$

where $a _ { i }$ is atomic text, $v _ { i }$ is its current read view, and $z _ { i }$ is its role. Atomic text supports relation inference and provenance, whereas the read view provides a compact representation for answering. The two roles are

$$
\begin{array} { r l } & { \mathcal { P } _ { t } = \{ M _ { i } : z _ { i } = \mathrm { P R I M A R Y } \} , } \\ & { \mathcal { E } _ { t } = \{ M _ { i } : z _ { i } = \mathrm { E V I D E N C E } \} . } \end{array}\tag{6}
$$

Primary records remain eligible for subsequent management-time retrieval. Evidence records are retained for provenance but do not compete independently at answer time.

Because different candidates may receive different labels, ROAM selects the highest-priority relation:

$$
\mathrm { C O N } \succ \mathrm { O S N } \succ \mathrm { E Q V } \succ \mathrm { N S O } \succ \mathrm { I N D } .\tag{7}
$$

This ordering protects state distinctions first, favors newly introduced specificity next, and consolidates only redundant observations. If $r _ { t } ^ { * }$ is the selected relation, its matched candidates are

$$
\mathcal { H } _ { t } = \{ M _ { i } \in \mathcal { C } _ { t } : r _ { t } ( i ) = r _ { t } ^ { * } \} .\tag{8}
$$

Thus, EQV and NSO preserve the earliest anchor, whereas OSN and CON promote the incoming observation; IND creates a new Primary. For every non-independent update, let $M ^ { * }$ be the surviving Primary and let $\mathcal { D } _ { t }$ be the records assigned to Evidence. Sorting $\{ M ^ { * } \} \cup \mathcal { D } _ { t }$ by observation time as $( S _ { 1 } , \ldots , S _ { n } )$ , ROAM updates its read view by

Table 1: Relation-conditioned memory updates. Atomic text remains immutable. Role Update denotes the deterministic mapping.
<table><tr><td>Relation</td><td>Role update</td><td>View update</td></tr><tr><td>IND</td><td>Insert  $M _ { t }$  as a new Primary.</td><td> $v _ { t }  a _ { t } .$ </td></tr><tr><td>EQV</td><td>Earliest matched Pri- mary survives; others</td><td>Remove redun- dancy.</td></tr><tr><td>OSN</td><td>become Evidence.  $M _ { t }$  becomes Pri- mary; old matches</td><td>Preserve added specificity.</td></tr><tr><td>NSO</td><td>become Evidence. Earliest matched Pri- mary survives;  $M _ { t }$ </td><td>Preserve exist- ing information.</td></tr><tr><td>CON</td><td>becomes Evidence.  $M _ { t }$  becomes current Primary; old matches become Evidence.</td><td>Build temporal history.</td></tr></table>

$$
\begin{array} { r } { v _ { M ^ { \ast } }  \mathrm { F u s e } ( \mathrm { V i e w } ( S _ { 1 } ) , \dots , } \\ { \mathrm { V i e w } ( S _ { n } ) ; r _ { t } ^ { \ast } ) . \qquad } \end{array}\tag{9}
$$

Fusion changes only the read view: it removes redundancy for EQV, preserves compatible specificity for OSN/NSO, and records temporal state changes for CON. Answer-time retrieval uses Primary views, while subsequent management continues to use immutable atomic texts.

## 3 Experiments

Section 2 introduced ROAM, a relation conditioned memory management method. By analyzing the relations between incoming and stored memories, ROAM organizes, consolidates, and updates candidate memories to construct a compact, queryoriented collection of retrievable memory units. We refer to this collection as the active memory set: the memory units that are eligible for retrieval at answer time.

This section evaluates whether ROAM remains robust under retrieval-budget interference, whether its gains transfer to natural long-horizon and stateupdate tasks, and which relations and system components account for its gains.

## 3.1 Experimental Setup

## 3.1.1 Benchmarks

We use three complementary evaluation settings:

LongMemEval (Wu et al., 2025). We evaluate the unmodified 500-question LongMemEval benchmark to test transfer to natural long-horizon conversational histories. Memory histories, question difficulty, and candidate-memory distributions are all determined by the original dataset.

Algorithm 1 ROAM Atomic Memory Update   
Require: New record $M _ { t } = ( a _ { t } , a _ { t } , \tau _ { t } , c _ { t } , \cdot )$ and   
state $( \mathcal { P } , \mathcal { E } )$   
Ensure: Updated state and Primary views   
1: $\mathcal { C }  \rho ( a _ { t } , \{ a _ { i } : M _ { i } \in \mathcal { P } \} , k )$   
2: if ${ \mathcal { C } } = \emptyset$ then   
3: $r ^ { * }  \mathrm { I N D }$   
4: else   
5: $\mathcal { L }  \{ \Phi ( a _ { t } , a _ { i } ) : M _ { i } \in \mathcal { C } \}$   
6: $r ^ { * } \gets \mathrm { H i g h e s t P r i o r i t y } ( \mathcal { L } )$   
7: end if   
8: if $r ^ { * } = \mathrm { I N D }$ then   
9: Insert $M _ { t }$ as a new Primary   
10: else   
11: H ← Matches $( \mathcal { C } , r ^ { * } )$   
12: $( M ^ { \ast } , \mathcal { D } ) \gets$ RoleUpdat $\mathsf { \Omega } _ { \mathsf { r } ^ { * } } ^ { } ( \mathcal { H } , M _ { t } )$   
13: Update roles and sets: $M ^ { * }$ Primary, D Evi  
dence   
14: Sort $\{ M ^ { * } \} \cup \mathcal { D }$ by time and form V   
15: v ∗ ← FuseOrdered $\mathbf { \eta } . ( \mathbf { V } ; r ^ { * } )$   
16: end if

MEME-Post. MEME (Jung et al., 2026) involves multiple entities and values that may remain stable or change over time. We process preceding sessions chronologically and evaluate all 694 postchange questions using the final-state questionanswering formulation adopted throughout this study. We call the resulting unfiltered questionlevel accuracy MEME-Post; this setting tests transfer to multi-entity memory and state-update scenarios.

LongMemEval Controlled. LongMemEval and MEME test whether a complete memory system can answer questions from long conversational histories, but their fixed histories, difficulty, and candidate-memory distributions give end-to-end accuracy limited leverage for analyzing how systems organize redundant or unhelpful content and correctly resolve changes between old and new states. To address this, we construct LongMemEval Controlled from the 470 LongMemEval questions with usable gold-memory annotations. The core design holds the answer-critical memories fixed for each question and varies only the number of additional confounders a management policy must handle, evaluating at $N \in \{ 0 , 2 , 4 , 6 , 8 \}$ . This setting provides a precise diagnosis of whether a memorymanagement strategy can effectively suppress semantic interference and accurately retrieve critical memories under a constrained retrieval budget. The full construction protocol is given in Appendix F.

## 3.1.2 Baselines

We compare three memory-management policies spanning no active management, model-driven updates, and semantic consolidation:

▷ Append-all retains every candidate as an independently retrievable entry without active consolidation or deletion. Therefore, this method does not require a manager model.

▷ Mem0 (Chhikara et al., 2025) uses the manager to choose ADD, UPDATE, DELETE, or NONE for each incoming memory and retrieved history.

▷ EverMemOS-style (Hu et al., 2026a) applies semantic grouping and consolidation only to short factual memories, excluding the full system’s trace formation, profile building, and agentic recollection.

## 3.2 Implementation Details

The main experiments use Qwen3.5-9B (Qwen Team, 2026), Qwen3-8B (Yang et al., 2025), and Gemma-4-12B (Gemma Team et al., 2026) as manager models for cross-family comparisons. To analyze the effect of parameter scale, we additionally compare Gemma-4-E4B, Gemma-4-12B, and Gemma-4-26B-A4B within the same model family. Within each comparison, all managed methods use the same manager model. The relation, fusion, budget, and main-text retrieval analyses use Qwen3.5-9B; the supplementary retrieval breakdown additionally reports results using Gemma-4-12B.

Across all configurations, methods receive the same extracted candidate-memory stream and initial store. We use Qwen3-Embedding-0.6B (Zhang et al., 2025) for retrieval, Gemma-4- 26B-A4B for answer generation, and DeepSeek-V4- Flash (DeepSeek-AI et al., 2026) for evaluation. All methods operate under the same 256-token memory-context budget. For ROAM, the shared retriever embeds Primary atomic texts during memory management and Primary read views during answering. Consequently, performance differences primarily reflect how each method organizes memories and constructs retrieval units for answering.

Table 2 summarizes answer accuracy on Long-MemEval Controlled, full LongMemEval, and

Table 2: Answer accuracy (%; higher is better) with a 256-token memory budget. Because Append-all is managerindependent, its result is repeated across manager-model blocks. Boldface and underlining mark the highest and second-highest means, respectively, within each manager-model block and evaluation condition.
<table><tr><td rowspan="2">Manager model</td><td rowspan="2">Method</td><td colspan="5">LongMemEval Controlled</td><td rowspan="2">LongMemEval</td><td rowspan="2">MEME-Post</td></tr><tr><td> $N = 0$ </td><td> $N = 2$ </td><td> $N = 4$ </td><td> $N = 6$ </td><td> $N = 8$ </td></tr><tr><td rowspan="4">Gemma-4-12B</td><td>Append-all</td><td>87.0</td><td>77.3</td><td>47.0</td><td>25.4</td><td>24.4</td><td>57.1</td><td>32.5</td></tr><tr><td>Mem0</td><td>74.3</td><td>72.3</td><td>74.0</td><td>71.7</td><td>69.1</td><td>55.0</td><td>24.8</td></tr><tr><td>EverMemOS-style</td><td>86.2</td><td>74.5</td><td>48.5</td><td>38.3</td><td>36.0</td><td>54.8</td><td>32.4</td></tr><tr><td>ROAM (Ours)</td><td>87.2</td><td>81.5</td><td>78.3</td><td>76.2</td><td>71.3</td><td>63.0</td><td>38.9</td></tr><tr><td rowspan="4">Qwen3.5-9B</td><td>Append-all</td><td>87.0</td><td>77.3</td><td>47.0</td><td>25.4</td><td>24.4</td><td>57.1</td><td>32.5</td></tr><tr><td>Mem0</td><td>85.3</td><td>75.7</td><td>67.0</td><td>54.7</td><td>42.1</td><td>57.8</td><td>31.4</td></tr><tr><td>EverMemOS-style</td><td>86.0</td><td>74.0</td><td>47.2</td><td>37.2</td><td>36.0</td><td>55.8</td><td>24.9</td></tr><tr><td>ROAM (Ours)</td><td>84.9</td><td>78.1</td><td>78.3</td><td>76.6</td><td>71.9</td><td>59.8</td><td>33.1</td></tr><tr><td rowspan="4">Qwen3-8B</td><td>Append-all</td><td>87.0</td><td>77.3</td><td>47.0</td><td>25.4</td><td>24.4</td><td>57.1</td><td>32.5</td></tr><tr><td>Mem0</td><td>76.8</td><td>64.0</td><td>59.8</td><td>53.2</td><td>48.9</td><td>54.2</td><td>30.3</td></tr><tr><td>EverMemOS-style</td><td>86.2</td><td>73.2</td><td>48.3</td><td>39.6</td><td>37.0</td><td>54.4</td><td>23.6</td></tr><tr><td>ROAM (Ours)</td><td>84.7</td><td>77.9</td><td>74.9</td><td>68.1</td><td>64.3</td><td>58.6</td><td>31.6</td></tr></table>

MEME-Post for the three manager models.

## 3.3 Main Results

Robustness to memory interference and consistency across manager models. ROAM ranks first across all non-zero interference settings, and its advantage remains consistent across all three manager models. At N = 8, it surpasses the strongest baseline in each group by up to 29.8 percentage points, with substantially smaller degradation as interference increases. This indicates that the gains stem from the relation-conditioned organization policy itself rather than reliance on a specific manager model. In contrast, Append-all reaches 87.0% under N = 0 but plummets to 24.4% at N = 8, showing that retaining all candidates is effective only when the memory store is uncontested, whereas explicitly controlling the active memory set yields stable advantages under interference.

Generalization across evaluation settings. The advantage also transfers to natural scenarios. On the full LongMemEval, ROAM achieves the highest accuracy under all three manager models (63.0%, 59.8%, and 58.6%), exceeding the corresponding strongest baselines by 5.9, 2.0, and 1.5 percentage points. On MEME-Post, ROAM outperforms model-based baselines with every manager and ranks first overall with Gemma-4-12B and Qwen3.5-9B (38.9% and 33.1%). With Qwen3-8B, ROAM remains the strongest managed store at 31.6%. These results demonstrate that ROAM scales to long-horizon histories, multi-entity interactions, and post-update questions.

Table 3: Relation and fusion ablations with Qwen3.5-9B and a 256-token memory budget. Values are answer accuracy (%; higher is better). Append-all and ROAM reuse the corresponding results from Table 2; other rows use matched ablation runs. Boldface and underlining mark the highest and second-highest values, respectively, at each interference level.
<table><tr><td colspan="5">Management policy N = 0 N = 2 N = 4 N = 6  $N = 8$ </td></tr><tr><td>Append-all</td><td>87.0 77.3</td><td>47.0</td><td>25.4</td><td>24.4</td></tr><tr><td>CON-only</td><td>84.0</td><td>77.2 54.3</td><td>34.9</td><td>32.1</td></tr><tr><td>CON+EQV</td><td>84.0 78.3</td><td>72.8</td><td>66.6</td><td>62.3</td></tr><tr><td>ROAM w/o Fusion</td><td>79.1 77.9</td><td>75.5</td><td>73.6</td><td>70.9</td></tr><tr><td>ROAM</td><td>84.9</td><td>78.1 78.3</td><td>76.6</td><td>71.9</td></tr></table>

## 3.4 Ablation Studies

The main results establish that ROAM is most effective when the memory store contains many plausible candidates. We therefore examine which design choices in its relation-aware memory policy account for this robustness.

## 3.4.1 Which Relations Matter?

We compare three relation policies with progressively greater expressive power. CON-only handles contradiction relations. CON+EQV additionally handles equivalence relations. Full ROAM further models directional containment through the OSN and NSO relation types. Table 3 evaluates each policy across the interference sweep.

The overall trend is clear: as the policy covers more relation types, performance becomes more stable as interference increases. At N = 8, CON-only, CON+EQV, and ROAM achieve 32.1%, 62.3%, and 71.9%, respectively. From $ { \boldsymbol { N } } \quad = \quad 4$ onward, their ordering is consistent: $\mathbf { C O N - o n l y < C O N + E Q V < R O A M }$ . These results show that contradiction, equivalence, and directional-containment relations all contribute to memory management. Each captures a different form of interaction among memories, and together they enable ROAM to construct a more reliable active set when candidate memories compete for limited context.

## 3.4.2 Effect of Fusion

Relation ablations confirm that richer relation modeling improves the active set. To test whether the answer-time representation matters independently, we construct ROAM w/o Fusion: it keeps the same managed candidates, relation predictions, and Primary–Evidence assignments as ROAM, but retrieves the original Primary text instead of generating fused views. This isolates representation effects while holding active-set management fixed.

Table 3 suggests that Fusion improves accuracy at every interference level, so its benefit is consistent across all interference strengths rather than limited to a specific regime. Relation-conditioned management alone already provides a substantial gain, and fused views further increase accuracy. Because the two variants share identical relation predictions and Primary–Evidence assignments, the comparison shows that the mechanisms are complementary: management selects what to retain, while fusion determines how that information is organized and presented to the answer model.

## 3.5 Analysis

In this section, we want to address two questions. First, does ROAM remain effective across managermodel scales and memory-context budgets? Second, do its relation decisions improve the final context through the intended mechanism?

## 3.5.1 Manager-Model Scale Sensitivity

Because the manager model performs relation inference, its capacity may affect the quality of retrievaleligibility decisions. Figure 3 isolates this factor within the Gemma-4 family while holding all other components fixed. ROAM remains consistently strong across E4B, 12B, and 26B-A4B, ranking first at all displayed nonzero interference levels for all three scales; its advantage therefore does not depend on a particular manager-model size.

We further find that additional capacity becomes most useful as competition intensifies. At N=8,

![](images/d77b8cf30b5cfe62c3e6ef3074c5d9840a4dfd8cd1a45e8c0b1f5c4ce53575d3.jpg)

Figure 3: Manager-model scale sensitivity. The horizontal axis varies the Gemma-4 manager from E4B to 26B-A4B; the vertical axis reports answer accuracy. Panels show $N \in \{ 0 , 4 , 8 \}$ , with all other components fixed.  
![](images/b492c098509884ae1dcade61006b5069a01519d2bd085345d25dcbbd35e2ff83.jpg)  
Figure 4: Memory-context budget sensitivity with Qwen3.5-9B. The horizontal axis gives the memorycontext budget in tokens; the vertical axis reports answer accuracy . Panels show $N \in \{ 0 , 4 , 8 \}$

ROAM improves from 69.4% with E4B to 71.3% with 12B and 74.5% with 26B-A4B, a 5.1-point gain overall. By contrast, the three variants perform more similarly under low interference, where few candidates compete for the available context. This interaction is consistent with ROAM’s design: stronger relation inference matters most when the active memory set must preserve answer-relevant information under heavy competition.

## 3.5.2 Memory-Context Budget Sensitivity

The memory-context budget determines how selectively those decisions must allocate retrieval capacity. Figure 4 examines this complementary factor by fixing Qwen3.5-9B as the manager and varying the budget over 128, 256, and 512 tokens.

ROAM’s advantage is most pronounced when the context is constrained or interference is strong. At N = 8, it leads at all three budgets, reaching 42.8%, 71.9%, and 83.0% and outperforming the strongest baseline by 14.9, 29.8, and 4.1 points, respectively. Thus, even as the budget increases fourfold, ROAM retains its advantage in the most competitive setting. These results indicate that the gain comes from organizing limited context among plausible candidates rather than merely increasing the amount of available context.

## 3.5.3 Relation-Inference Quality

We next examine the internal control signal that supports the robustness of ROAM. ROAM formulates memory management as five-way relation inference and uses the predicted relation to assign Primary–Evidence roles and retrieval eligibility. Reliable relation inference is therefore a prerequisite for constructing the intended active memory set. We evaluate Gemma-4-26B-A4B-it on 843 memory pairs sampled from PersonaMem-v2 dialogues (Jiang et al., 2025). DeepSeek-V4-Pro provides initial labels for stratified sampling across the five relations, after which we manually verify every pair against ROAM’s definitions. This construction provides sufficient support for a class-balanced assessment of relation discriminability.

Table 4: Relation-inference performance (%) of Gemma-4-26B-A4B-it on 843 manually verified PersonaMem-v2 memory pairs. Stratified sampling supports balanced evaluation across the five relations.
<table><tr><td>Class</td><td>Precision</td><td>Recall</td><td>F1</td></tr><tr><td>IND</td><td>65.5</td><td>96.5</td><td>78.0</td></tr><tr><td>EQV</td><td>97.2</td><td>89.3</td><td>93.1</td></tr><tr><td>OSN</td><td>85.4</td><td>68.5</td><td>76.0</td></tr><tr><td>NSO</td><td>65.7</td><td>73.4</td><td>69.3</td></tr><tr><td>CON</td><td>98.0</td><td>84.9</td><td>91.0</td></tr><tr><td>Macro avg.</td><td>82.4</td><td>82.5</td><td>81.5</td></tr></table>

As Table 4 shows, Gemma-4-26B-A4B-it identifies equivalence and conflict particularly well, with F1 scores of 93.16% and 91.02%. It also distinguishes the two directional subsumption relations, achieving 76.08% for OSN, where the incoming memory is more specific, and 69.37% for NSO, where the stored memory is more specific. Overall, the results show that ROAM’s five-way relation space provides an explicit and measurable interface for memory management.

## 3.5.4 Retrieval Mechanism Analysis

Reliable relation predictions matter only if the resulting management decisions improve the context presented to the answer model. We therefore inspect the final 256-token context at N = 8 with Qwen3.5-9B. We report four complementary metrics. Any-source coverage asks whether the context represents at least one required gold source, while all-source coverage requires every gold source. Gold-source recall measures the fraction of required sources represented per query, and explicit confounder-token share measures the fraction of retained tokens uniquely attributable to confounders. Coverage propagates deduplicated provenance from atomic records to Primary read views that contribute tokens to the final context.

Table 5: retrieval diagnostics with Qwen3.5-9B at N = 8 under a 256-token memory-context budget. Goldsource metrics use deduplicated provenance; an original entry or its provenance-linked Primary read view covers the corresponding source. Explicit confounder tokens exclude mixed or unattributed tokens. Ever-s. denotes the adapted EverMemOS-style baseline.
<table><tr><td>Metric</td><td>Append</td><td>Mem0</td><td>Ever-s.</td><td>ROAM</td></tr><tr><td>Any-source cov. (%) ↑</td><td>36.6</td><td>81.3</td><td>35.5</td><td>95.1</td></tr><tr><td>All-source cov. (%) ↑</td><td>21.7</td><td>57.4</td><td>18.5</td><td>73.8</td></tr><tr><td>Gold-source recall (%) ↑</td><td>29.1</td><td>69.0</td><td>26.9</td><td>84.6</td></tr><tr><td>Confounder tokens (%) ↓</td><td>84.0</td><td>58.6</td><td>68.3</td><td>47.1</td></tr></table>

Table 5 shows that ROAM provides the strongest coverage on all three gold-source metrics. It represents at least one required source for 95.1% of queries and all required sources for 73.8%, with 84.6% gold-source recall.

All methods use almost the full budget, so the coverage gain does not come from exposing the answer model to more context. ROAM instead achieves higher coverage with 4.7 retrieved Primary read views on average, compared with 4.8– 5.1 units for the baselines. The mechanism is therefore selective context organization: ROAM allocates more of the same context budget to answercritical sources while reducing competition from confounders.

## 4 Conclusion

We introduced ROAM, a relation-guided framework for managing atomic factual memories under finite retrieval budgets. ROAM separates constrained semantic relation inference from deterministic state control, retaining immutable observations in auditable Primary–Evidence structures while exposing compact fused views for retrieval.

Across controlled interference, full Long-MemEval, and MEME-Post, ROAM consistently outperforms model-based memory-management baselines across three manager models. Ablations and retrieval diagnostics show that relation-guided control of the active memory set drives most of the gain by improving coverage of answer-critical evidence and reducing competition from redundant or superseded records, while fusion provides a complementary benefit. These results suggest that longterm agent memory should decouple source retention from retrieval eligibility, preserving historical evidence without requiring every observation to compete for limited answer-time context.

## Limitations

Our evaluation uses benchmark histories and controlled retrieval budgets, which enable measurable interference and consistent comparisons but cannot capture all deployments. Applications may differ in stream length, update frequency, context limits, and conversational complexity; gains should therefore be interpreted within the evaluated settings.

ROAM also operates on short, extracted factual units. This makes relation inference and provenance explicit, but leaves hierarchical, event-level, and richer structured memories unexplored. Extending the Primary–Evidence organization and active-set retrieval principles to these representations is an important direction for future work.

## References

Chen Amiraz, Florin Cuconasu, Simone Filice, and Zohar Karnin. 2025. The distracting effect: Understanding irrelevant passages in RAG. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 18228–18258, Vienna, Austria. Association for Computational Linguistics.

Pratyay Banerjee, Masud Moshtaghi, Shivashankar Subramanian, Amita Misra, and Ankit Chadha. 2026. APEX-MEM: Agentic semi-structured memory with temporal reasoning for long-term conversational AI. In Proceedings of the 64th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 16470–16489, San Diego, California, United States. Association for Computational Linguistics.

Tong Chen, Hongwei Wang, Sihao Chen, Wenhao Yu, Kaixin Ma, Xinran Zhao, Hongming Zhang, and Dong Yu. 2024. Dense X retrieval: What retrieval granularity should we use? In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 15159–15177, Miami, Florida, USA. Association for Computational Linguistics.

Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. 2025. Mem0: Building production-ready AI agents with scalable long-term memory. Preprint, arXiv:2504.19413.

DeepSeek-AI, Anyi Xu, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, Chenchen Ling, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chengyu Hou, Chenhao Xu, Chenze Shao, Chong Ruan, Conner Sun, and 300 others. 2026. DeepSeek-V4: Towards highly efficient million-token context intelligence. Preprint, arXiv:2606.19348.

Jizhan Fang, Xinle Deng, Haoming Xu, Ziyan Jiang, Yuqi Tang, Ziwen Xu, Shumin Deng, Yunzhi Yao, Mengru Wang, Shuofei Qiao, Huajun Chen, and Ningyu Zhang. 2026. LightMem: Lightweight and efficient memory-augmented generation. In International Conference on Learning Representations, volume 2026, pages 98706–98729.

Gemma Team, Sherif El Abd, Vaibhav Aggarwal, Robin Algayres, Alek Andreev, Olivier Bachem, Ian Ballantyne, Cormac Brick, Victor Carbune, Michelle˘ Casbon, Mayank Chaturvedi, Aditya Chawla, Victor Cotruta, Alice Coucke, Phil Culliton, Robert Dadashi, Lucas Dixon, Mohamed Elhawaty, Utku Evci, and 304 others. 2026. Gemma 4 technical report. Preprint, arXiv:2607.02770.

Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su. 2024. HippoRAG: Neurobiologically inspired long-term memory for large language models. In Advances in Neural Information Processing Systems, volume 37, pages 59532–59569. Curran Associates, Inc.

Kostas Hatalis, Despina Christou, Joshua Myers, Steven Jones, Keith Lambert, Adam Amos-Binks, Zohreh Dannenhauer, and Dustin Dannenhauer. 2024. Memory matters: The need to improve long-term memory in LLM-agents. In Proceedings ofthe AAAI Symposium Series, volume 2, pages 277–280.

Chuanrui Hu, Xingze Gao, Zuyi Zhou, Dannong Xu, Yi Bai, Xintong Li, Hui Zhang, Tong Li, Chong Zhang, Lidong Bing, and Yafeng Deng. 2026a. EverMemOS: A self-organizing memory operating system for structured long-horizon reasoning. Preprint, arXiv:2601.02163.

Yuanzhe Hu, Yu Wang, and Julian McAuley. 2026b. Evaluating memory in LLM agents via incremental multi-turn interactions. In International Conference on Learning Representations, volume 2026, pages 156259–156291.

Yupeng Huo, Yaxi Lu, Zhong Zhang, Haotian Chen, and Yankai Lin. 2026. AtomMem: Learnable dynamic agentic memory with atomic memory operation. Preprint, arXiv:2601.08323.

Bowen Jiang, Yuan Yuan, Maohao Shen, Zhuoqun Hao, Zhangchen Xu, Zichen Chen, Ziyi Liu, Anvesh Rao Vijjini, Jiashu He, Hanchao Yu, Radha Poovendran, Gregory Wornell, Lyle Ungar, Dan Roth, Sihao Chen, and Camillo Jose Taylor. 2025. PersonaMem-v2: Towards personalized intelligence via learning implicit user personas and agentic memory. Preprint, arXiv:2512.06688.

Seokwon Jung, Alexander Rubinstein, Arnas Uselis, Sangdoo Yun, and Seong Joon Oh. 2026. MEME: Multi-entity & evolving memory evaluation. Preprint, arXiv:2605.12477.

Jiazheng Kang, Mingming Ji, Zhe Zhao, and Ting Bai. 2025. Memory OS of AI agent. In Proceedings of the 2025 Conference on Empirical Methods in Natural

Language Processing, pages 25961–25970, Suzhou, China. Association for Computational Linguistics.

Chingkwun Lam, Jiaxin Li, Lingfei Zhang, and Kuo Zhao. 2026. Governing evolving memory in LLM agents: Risks, mechanisms, and the stability and safety governed memory (SSGM) framework. Preprint, arXiv:2603.11768.

Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2024. Lost in the middle: How language models use long contexts. Transactions ofthe Association for Computational Linguistics, 12:157–173.

Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang. 2024. Evaluating very long-term conversational memory of LLM agents. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 13851– 13870, Bangkok, Thailand. Association for Computational Linguistics.

Ali Modarressi, Ayyoob Imani, Mohsen Fayyaz, and Hinrich Schütze. 2024. RET-LLM: Towards a general read-write memory for large language models. Preprint, arXiv:2305.14322.

Kai Tzu-iunn Ong, Namyoung Kim, Minju Gwak, Hyungjoo Chae, Taeyoon Kwon, Yohan Jo, Seungwon Hwang, Dongha Lee, and Jinyoung Yeo. 2025. Towards lifelong dialogue agents via timeline-based memory management. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 8631–8661, Albuquerque, New Mexico. Association for Computational Linguistics.

Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, and Joseph E. Gonzalez. 2024. MemGPT: Towards LLMs as operating systems. Preprint, arXiv:2310.08560.

Vedant Patel. 2026. Supersede: Diagnosing and training the memory-update gap in LLM agents. Preprint, arXiv:2606.27472.

Qwen Team. 2026. Qwen3.5: Towards native multimodal agents.

Preston Rasmussen, Pavlo Paliychuk, Travis Beauvais, Jack Ryan, and Daniel Chalef. 2025. Zep: A temporal knowledge graph architecture for agent memory. Preprint, arXiv:2501.13956.

Zhanyu Shen, Sijie Cheng, Zhicheng Guo, Weiqin Wang, Yile Wang, and Hui Huang. 2026. AnchorMem: Anchored facts with associative contexts for building memory in large language models. In Findings ofthe Associationfor Computational Linguistics: ACL 2026, pages 34784–34798, San Diego, California, United States. Association for Computational Linguistics.

Haoran Sun, Zekun Zhang, and Shaoning Zeng. 2026. Preference-aware memory update for long-term LLM agents. In Findings of the Association for Computational Linguistics: ACL 2026, pages 783–793, San Diego, California, United States. Association for Computational Linguistics.

Zhen Tan, Jun Yan, I-Hung Hsu, Rujun Han, Zifeng Wang, Long Le, Yiwen Song, Yanfei Chen, Hamid Palangi, George Lee, Anand Rajan Iyer, Tianlong Chen, Huan Liu, Chen-Yu Lee, and Tomas Pfister. 2025. In prospect and retrospect: Reflective memory management for long-term personalized dialogue agents. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8416–8439, Vienna, Austria. Association for Computational Linguistics.

Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, and Dong Yu. 2025. LongMemEval: Benchmarking chat assistants on long-term interactive memory. In International Conference on Learning Representations, volume 2025, pages 86809– 86836.

Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang. 2025. A-Mem: Agentic memory for LLM agents. In Advances in Neural Information Processing Systems, volume 38, Main Conference, pages 17577–17604. Curran Associates, Inc.

Sikuan Yan, Xiufeng Yang, Zuchao Huang, Ercong Nie, Zifeng Ding, Zonggen Li, Xiaowen Ma, Jinhe Bi, Kristian Kersting, Jeff Z. Pan, Hinrich Schuetze, Volker Tresp, and Yunpu Ma. 2026. Memory-R1: Enhancing large language model agents to manage and utilize memories via reinforcement learning. In Proceedings of the 64th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 12805–12825, San Diego, California, United States. Association for Computational Linguistics.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41 others. 2025. Qwen3 technical report. Preprint, arXiv:2505.09388.

Guilin Zhang, Wei Jiang, Xiejiashan Wang, Aisha Behr, Kai Zhao, Jeffrey Friedman, Xu Chu, and Amine Anoun. 2026. Adaptive memory admission control for LLM agents. Preprint, arXiv:2603.04549.

Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren Zhou. 2025. Qwen3 Embedding: Advancing text embedding and reranking through foundation models. Preprint, arXiv:2506.05176.

Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and Yanlin Wang. 2024. MemoryBank: Enhancing large

language models with long-term memory. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 19724–19731.

## A Related Work

Memory representations. Long-term memory systems must first decide what unit of past interaction to store and retrieve. Early systems store conversation histories as text and retrieve relevant passages (Packer et al., 2024; Zhong et al., 2024). Yet vector retrieval over raw histories does not itself organize heterogeneous knowledge or determine how that knowledge should evolve (Hatalis et al., 2024). A complementary line therefore converts durable information into compact facts or relations. RET-LLM (Modarressi et al., 2024) maintains a read– write memory of triples, Dense X Retrieval (Chen et al., 2024) indexes atomic propositions, and HippoRAG (Gutiérrez et al., 2024) combines extracted relations with graph-based retrieval. Recent agentmemory systems likewise use atomic facts as retrieval units or enrich them with contextual structure (Shen et al., 2026; Huo et al., 2026). Across these systems, fine-grained units provide precise targets for retrieval, comparison, and updating. The same granularity, however, allows redundant, overlapping, and conflicting observations to accumulate as the store grows. ROAM assumes atomic facts have already been extracted and addresses how such observations should coexist, rather than proposing a new extractor, embedding model, or retriever.

Memory management. Once information is stored at fine granularity, a second problem is how to maintain the growing collection. One family delegates storage decisions directly to a language model. Mem0 (Chhikara et al., 2025) prompts the model to add, update, delete, or retain memories, while Memory-R1 (Yan et al., 2026) and Atom-Mem (Huo et al., 2026) learn policies over such operations. Another family emphasizes structural organization or consolidation. A-MEM (Xu et al., 2025) links memories and augments them with contextual attributes; Reflective Memory Management (RMM) (Tan et al., 2025) adaptively summarizes dialogue histories and refines retrieval; MemoryOS (Kang et al., 2025) coordinates hierarchical storage, updating, and retrieval; and Light-Mem (Fang et al., 2026) separates lightweight online processing from offline consolidation. Ever-MemOS (Hu et al., 2026a) further consolidates episodic MemCells into higher-level MemScenes.

These approaches position memory management between extraction and retrieval, but expose different control interfaces. Direct-operation policies couple semantic interpretation with storage mutation, whereas structural organizers generally do not derive an observation’s retrieval eligibility from an explicit semantic relation. ROAM instead uses that relation as the interface between model judgment and storage control. For each incoming–stored pair, it distinguishes independence, equivalence, conflict, and the two directions of subsumption; a fixed policy then assigns active Primary and supporting Evidence roles. Source observations remain unchanged, while a separate fusion stage constructs compact views for answer-time retrieval.

Evolving memory and evaluation. Organiza tion becomes harder when facts change over time, because a memory system must identify the current state without losing earlier evidence. THEANINE (Ong et al., 2025) retains old memories in temporal and causal timelines, Zep (Rasmussen et al., 2025) represents changing information in a temporal knowledge graph, and APEX-MEM (Banerjee et al., 2026) supports temporal reasoning over semi-structured conversational memory. Supersede (Patel, 2026) diagnoses fail ures when agents replace outdated values under a bounded memory budget. Related work also controls which observations enter the store and how stored entries are updated (Zhang et al., 2026; Lam et al., 2026). Together, these studies motivate temporal structure and explicit update control. ROAM handles temporal conflicts in the same relation space as restatements and specificity refinements, rather than invoking a separate update mechanism. Earlier observations remain available as Evidence, but only views associated with ac tive Primary records are retrieved for answering, separating historical provenance from retrieval eligibility. Evaluation has likewise moved beyond simple recall toward acquisition, updating, and temporal reasoning. LongMemEval (Wu et al., 2025), MEME (Jung et al., 2026), and MemoryAgentBench (Hu et al., 2026b) measure complementary aspects of these capabilities. We use Long MemEval and the post-change portion of MEME, and add a controlled intervention that holds answercritical memories fixed while varying semantic and temporal competition under a finite retrieval bud get.

## B LLM Usage Statement

## LLM Usage

LLMs were used both as components of the evaluated systems and as tools in the research workflow. In the research workflow, LLMs were used to generate the fixed pool of controlled confounders, provide initial labels for constructing the relation diagnostic, and serve as answer judges. All LLMgenerated labels were manually reviewed and corrected. The manager and answer models used in each experiment are specified in the experimental setup.

LLMs were also used solely to improve the language and readability of the manuscript. They did not contribute to the research questions, methodological design, experimental decisions, or interpretation of results. All LLM-assisted edits and research artifacts were reviewed and approved by the authors, who take full responsibility for the final manuscript, its claims, and the released artifacts.

## C Ethics Statement

This work studies personal factual memory in language-model agents. The experiments use existing benchmarks and involve neither new human interactions nor the collection of new personal data. Deployed memory systems may nevertheless expose sensitive information, retain outdated or incorrect claims, and influence later responses. ROAM’s Evidence records preserve source history and support auditability but are not a privacy or security mechanism. Deployments should therefore include informed consent, access control, secure storage, appropriate retention and deletion policies, and mechanisms for users to inspect and correct stored information. ROAM does not address these broader deployment requirements.

## D Evaluation Validation

Answer-judge agreement. We validate the automated answer judge through a stratified audit of 200 examples spanning all three benchmarks and all four compared memory-management methods. A human annotator independently assesses the semantic correctness of each candidate answer with respect to its question and reference answer. The resulting human judgments agree with the DeepSeek V4 Flash judge on 97.5% of the audited examples. This validation uses the same judge model and prompt as the main experiments.

Relation-label verification. The relation diagnostic reported in the main paper contains 843 memory pairs selected through stratified sampling to obtain approximately balanced coverage of IND, EQV, OSN, NSO, and CON. A human annotator reviews every pair against ROAM’s relation definitions before the diagnostic evaluation is performed.

## E Question-Type Results

Tables 6 and 7 report the question-type breakdowns for the full LongMemEval and MEME-Post evaluations. Append-all does not use a manager model; we therefore average its two recorded managerblock values for each question type and repeat the pooled result in both blocks. All values are rounded to one decimal place.

## F LongMemEval Controlled

## F.1 Construction and Confounder Audit

For each of the 470 LongMemEval questions with usable gold-memory annotations, we retain the answer-critical memories and the original non-gold background memories, then add a controlled number of additional memories, which we refer to as confounders. The answer-critical memories remain fixed across conditions; only the number and type of confounders vary. Let N denote the number of confounders in a question’s store. We evaluate $N \in \{ 0 , 2 , 4 , 6 , 8 \}$

We construct two types of confounders:

• Type I memories are semantically similar to the query but do not answer it: they provide neither the correct answer nor a verifiably incorrect alternative.

• Type II memories are used only for Knowledge Update questions. They state a valid but temporally obsolete value and conflict with the current one.

During memory ingestion, methods receive neither the current query, gold-memory annotations, confounder-type labels, nor the expected answer. Gemma-4-26B-A4B generates one fixed pool of eight confounders for each question. The same pool is then used by every method and manager, independently of their predictions or outputs.

All generated confounders pass an automated filtering stage before inclusion. For Type II confounders, the temporally obsolete value is always assigned an earlier timestamp than the corresponding current gold value. This preserves the intended chronological ordering and ensures that Type II examples represent genuine temporal supersession.

Table 6: Answer accuracy (%) by question type on the original LongMemEval benchmark. Temp., Multi., Update, User, Pref., and Asst. denote temporal reasoning, multi-session, knowledge update, single-session user, singlesession preference, and single-session assistant questions, respectively. Append-all is pooled across manager-model blocks. Boldface and underlining mark the highest and second-highest values, respectively, within each managermodel block and question type.
<table><tr><td>Manager model</td><td>Method</td><td>Temp.</td><td>Multi.</td><td>Update</td><td>User</td><td>Pref.</td><td>Asst.</td></tr><tr><td rowspan="4">Gemma 4 12B</td><td>Append-all</td><td>56.7</td><td>36.8</td><td>83.3</td><td>93.5</td><td>83.3</td><td>15.2</td></tr><tr><td>Mem0</td><td>55.9</td><td>37.2</td><td>69.4</td><td>90.6</td><td>86.7</td><td>14.3</td></tr><tr><td>EverMemOS-style</td><td>49.6</td><td>39.7</td><td>73.6</td><td>93.7</td><td>86.2</td><td>12.5</td></tr><tr><td>ROAM (ours)</td><td>63.8</td><td>43.8</td><td>87.5</td><td>95.3</td><td>90.0</td><td>14.3</td></tr><tr><td rowspan="4">Qwen3.5 9B</td><td>Append-all</td><td>56.7</td><td>36.8</td><td>83.3</td><td>93.5</td><td>83.3</td><td>15.2</td></tr><tr><td>Mem0</td><td>57.5</td><td>42.2</td><td>75.0</td><td>95.3</td><td>86.1</td><td>16.1</td></tr><tr><td>EverMemOS-style</td><td>51.2</td><td>37.2</td><td>76.4</td><td>92.2</td><td>83.3</td><td>10.7</td></tr><tr><td>ROAM (ours)</td><td>59.1</td><td>42.5</td><td>81.9</td><td>90.6</td><td>86.7</td><td>14.3</td></tr></table>

Table 7: Answer accuracy (%) by question type on MEME-Post. The benchmark’s question-type abbreviations are retained from the evaluation data. Append-all is pooled across manager-model blocks. Boldface and underlining mark the highest and second-highest values, respectively, within each manager-model block and question type.
<table><tr><td>Manager model</td><td>Method</td><td>Cas</td><td>Abs</td><td>Tr</td><td>Del</td><td>Agg</td><td>ER</td></tr><tr><td rowspan="3">Gemma 4 12B</td><td>Append-all</td><td>18.9</td><td>9.2</td><td>64.5</td><td>12.5</td><td>15.4</td><td>98.1</td></tr><tr><td>Mem0</td><td>26.2</td><td>8.5</td><td>4.5</td><td>3.7</td><td>14.2</td><td>97.3</td></tr><tr><td>EverMemOS-style ROAM (ours)</td><td>15.9 28.1</td><td>7.7 10.3</td><td>61.4 79.5</td><td>11.3 20.1</td><td>20.0 15.8</td><td>97.6 97.6</td></tr><tr><td rowspan="4">Qwen3.5 9B</td><td></td><td>18.9</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Append-all Mem0</td><td>13.4</td><td>9.2 9.2</td><td>64.5 16.6</td><td>12.5 7.4</td><td>15.4 19.3</td><td>98.1</td></tr><tr><td>EverMemOS-style</td><td>19.5</td><td>7.7</td><td>58.3</td><td>10.8</td><td>13.2</td><td>97.3 97.3</td></tr><tr><td>ROAM (ours)</td><td>23.8</td><td>8.5</td><td>60.9</td><td>16.6</td><td>14.7</td><td>97.6</td></tr></table>

The resulting controlled dataset contains 3,760 fixed confounders: 3,184 Type I memories for 398 non-update questions and 576 Type II memories for 72 Knowledge Update questions. Using Qwen3- Embedding-0.6B and the gold annotations, we compute the similarity of every confounder to the query and to each answer-critical memory. As Table 8 shows, most confounders are more similar to the query than all required gold memories. The controlled setting therefore requires systems to handle plausible candidate memories rather than merely filter obviously irrelevant ones.

## F.2 Retrieval Diagnostics

We compute retrieval diagnostics for Gemma 4 12B and Qwen3.5-9B at N = 8 over the same 470 questions and three runs used for controlled answer accuracy. Once a run’s final store, embeddings, and retrieval order are fixed, the diagnostic is deterministic and requires neither the answer model nor the judge.

Table 8: Audit of the fixed confounder set for the 470 controlled questions. Type I memories are semantic and non-answering; Type II memories contain temporally obsolete values. Similarity rows report candidate-level means. The final two rows give the percentage of confounders whose query similarity exceeds that of at least one or all required gold memories.

<table><tr><td>Statistic</td><td>Type I</td><td>Type II</td></tr><tr><td>Queries</td><td>398</td><td>72</td></tr><tr><td>Confounders</td><td>3,184</td><td>576</td></tr><tr><td>Confounder-query similarity</td><td>0.7149</td><td>0.7641</td></tr><tr><td>Gold-query similarity</td><td>0.6321</td><td>0.6926</td></tr><tr><td>Above at least one gold (%)</td><td>76.35</td><td>83.33</td></tr><tr><td>Above all gold (%)</td><td>62.19</td><td>73.09</td></tr></table>

Gold-source coverage. For each query q, let $G _ { q }$ contain stable keys for the required gold sources, and let $S _ { q }$ contain the source keys represented by retrieval units that contribute tokens to the final hard-truncated context. For ROAM, these units are the views attached to retrieved Primaries. We

Table 9: Gold-source recall (%) by confounder type at $N = 8$ . Type I contains semantic non-answering confounders for 398 queries; Type II contains temporally obsolete values for 72 Knowledge Update queries. Append-all is pooled across manager-model blocks.
<table><tr><td>Manager model</td><td>Method</td><td>Type I</td><td>Type II</td></tr><tr><td rowspan="3">Gemma 4 12B</td><td>Append-all</td><td>31.5</td><td>16.0</td></tr><tr><td>Mem0 EverMemOS-style</td><td>60.9 29.7</td><td>52.1 13.2</td></tr><tr><td>ROAM</td><td>85.9</td><td>79.9</td></tr><tr><td rowspan="3">Qwen3.5 9B</td><td>Append-all</td><td>31.5</td><td>16.0</td></tr><tr><td>Mem0</td><td>73.3</td><td>45.1</td></tr><tr><td>EverMemOS-style</td><td>29.6</td><td>11.8</td></tr><tr><td rowspan="2"></td><td>ROAM</td><td>83.5</td><td>91.0</td></tr><tr><td></td><td></td><td></td></tr></table>

compute

$$
\operatorname { A n y G o l d } ( q ) = \mathbb { I } [ | G _ { q } \cap S _ { q } | > 0 ] ,\tag{10}
$$

$$
\mathrm { A l l G o l d } ( q ) = \mathbb { I } [ G _ { q } \subseteq S _ { q } ] ,\tag{11}
$$

$$
\mathrm { G o l d R e c a l l } ( q ) = \frac { \vert G _ { q } \cap S _ { q } \vert } { \vert G _ { q } \vert } .\tag{12}
$$

Metrics are macro-averaged over queries after deduplicating source keys and retrieved matches. Append-all, EverMemOS-style, and ROAM use exact matches from gold text to ingestion records; ROAM propagates membership into fused views through fused\_member\_ids. Because Mem0 may rewrite text, matching first uses exact identity and then a SequenceMatcher ratio of at least 0.65. Unmapped sources remain in $G _ { q }$ and count as uncovered. These metrics quantify the representation of answer-critical sources in the final context.

Token attribution and truncation. The retrieved memory block is serialized and hardtruncated to 256 Qwen3-8B tokenizer tokens. A unit is counted only if it contributes retained tokens; for ROAM, this unit is a Primary view rather than its underlying atoms. Let $T _ { q }$ be the retained memory-context tokens and $C _ { q }$ the subset uniquely attributable to fixed confounder sources. We report the macro-average of $C _ { q } / T _ { q } ;$ ; tokens with mixed or unavailable attribution are excluded from the numerator. Formatting, system-prompt, question, and answer-instruction tokens are excluded from $T _ { q }$

Append-all does not use a manager model, so its retrieval diagnostics are shared across managermodel blocks.

Table 10: Mean memory-management overhead per episode at $N = 4 .$ . Call counts include failed requests.
<table><tr><td>Metric</td><td>Mem0</td><td>EverMemOS-style</td><td>ROAM</td></tr><tr><td>LLM calls</td><td>30.1</td><td>18.6</td><td>271.6</td></tr><tr><td>Input tokens after prefix reuse (k)</td><td>16.79</td><td>1.56</td><td>24.79</td></tr><tr><td>Output tokens (k)</td><td>11.46</td><td>0.85</td><td>1.90</td></tr><tr><td>Shared-prefix proportion (%)</td><td>69.5</td><td>60.5</td><td>94.6</td></tr></table>

## F.3 Memory-Management Token Consumption

We measure memory-management token consumption on LongMemEval Controlled at N = 4 and report averages over 470 episodes. All three methods use the same manager model and a common tokencounting convention. The measurements cover management calls for relation inference, memoryoperation decisions, and fusion, excluding fact extraction, answer generation, and answer evaluation. Input tokens are counted after prefix-cache reuse; the shared-prefix proportion is measured relative to the input token count before reuse.

ROAM decomposes memory management into fine-grained relation inference, deterministic state updates, and fusion when needed, resulting in more management calls. Nevertheless, it generates 1.90k output tokens per episode, 83.4% fewer than Mem0, consistent with its use of constrained relation-label outputs. Shared prefixes account for 94.6% of ROAM’s input tokens before reuse, indicating that repeated instructions constitute a substantial portion of its input volume. These results characterize ROAM’s design tradeoff: explicit relation management requires more fine-grained calls while keeping the generated output volume small.

## G Complete Prompts

The following boxes report the prompts used for baseline memory management, relation inference, fusion, answer generation, and evaluation. Template variables are shown in double brackets.

## Mem0 direct-operation prompt.

You are a smart memory manager which controls the memory of a system.   
You can perform four operations: (1) add into the memory, (2) update the memory, (3) delete   
from the memory, and (4) no change.   
Based on the above four operations, the memory will change.   
Compare newly retrieved facts with the existing memory. For each new fact, decide whether to:   
ADD: Add it to the memory as a new element   
1 UPDATE: Update an existing memory element   
一 DELETE: Delete an existing memory element   
NONE: Make no change (if the fact is already present or irrelevant)   
There are specific guidelines to select which operation to perform:   
1. \*\*Add\*\*: If the retrieved facts contain new information not present in the memory, then you   
have to add it by generating a new ID in the id field.   
- \*\*Example\*\*:   
- Old Memory:   
[   
{   
"id" : "0",   
"text" : "User is a software engineer"   
}   
]   
- Retrieved facts: ["Name is John"]   
- New Memory:   
{   
"memory" : [   
{   
"id" : "0",   
"text" : "User is a software engineer",   
"event" : "NONE"   
},<sub>{</sub>   
"id" : "1",   
"text" : "Name is John",   
"event" : "ADD"   
}   
]   
}   
2. \*\*Update\*\*: If the retrieved facts contain information that is already present in the   
memory but the information is totally different, then you have to update it.   
If the retrieved fact contains information that conveys the same thing as the elements present   
in the memory, then you have to keep the fact which has the most information.   
Example (a) -- if the memory contains "User likes to play cricket" and the retrieved fact is   
"Loves to play cricket with friends", then update the memory with the retrieved facts.   
Example (b) -- if the memory contains "Likes cheese pizza" and the retrieved fact is "Loves   
cheese pizza", then you do not need to update it because they convey the same information.   
If the direction is to update the memory, then you have to update it.   
Please keep in mind while updating you have to keep the same ID.   
Please note to return the IDs in the output from the input IDs only and do not generate any   
new ID.   
- \*\*Example\*\*:   
- Old Memory:   
[   
{   
"id" : "0",   
"text" : "I really like cheese pizza"   
},

{   
"id" : "1",   
"text" : "User is a software engineer"   
},<sub>{</sub>   
"id" : "2",   
"text" : "User likes to play cricket"   
}   
]   
- Retrieved facts: ["Loves chicken pizza", "Loves to play cricket with friends"]   
- New Memory:   
{   
"memory" : [   
{   
"id" : "0",   
"text" : "Loves cheese and chicken pizza",   
"event" : "UPDATE",   
"old\_memory" : "I really like cheese pizza"   
},<sub>{</sub>   
"id" : "1",   
"text" : "User is a software engineer",   
"event" : "NONE"   
},<sub>{</sub>   
"id" : "2",   
"text" : "Loves to play cricket with friends",   
"event" : "UPDATE",   
"old\_memory" : "User likes to play cricket"   
}   
]   
}   
3. \*\*Delete\*\*: If the retrieved facts contain information that contradicts the information   
present in the memory, then you have to delete it. Or if the direction is to delete the   
memory, then you have to delete it.   
Please note to return the IDs in the output from the input IDs only and do not generate any   
new ID.   
- \*\*Example\*\*:   
- Old Memory:   
[   
{   
"id" : "0",   
"text" : "Name is John"   
},<sub>{</sub>   
"id" : "1",   
"text" : "Loves cheese pizza"   
}   
]   
- Retrieved facts: ["Dislikes cheese pizza"]   
- New Memory:   
{   
"memory" : [   
{   
"id" : "0",   
"text" : "Name is John",   
"event" : "NONE"   
},<sub>{</sub>   
"id" : "1",   
"text" : "Loves cheese pizza",   
"event" : "DELETE"   
}   
<sup>]</sup><sub>}</sub>

4. \*\*No Change\*\*: If the retrieved facts contain information that is already present in the   
memory, then you do not need to make any changes.   
- \*\*Example\*\*:   
- Old Memory:   
[   
{   
"id" : "0",   
"text" : "Name is John"   
},<sub>{</sub>   
"id" : "1",   
"text" : "Loves cheese pizza"   
}   
]   
- Retrieved facts: ["Name is John"]   
- New Memory:   
{   
"memory" : [   
{   
"id" : "0",   
"text" : "Name is John",   
"event" : "NONE"   
},   
{   
"id" : "1",   
"text" : "Loves cheese pizza",   
"event" : "NONE"   
}   
]   
}   
[[ current\_memory\_part ]]   
The new retrieved facts are mentioned in the triple backticks. You have to analyze the new   
retrieved facts and determine whether these facts should be added, updated, or deleted in   
the memory.   
[[ response\_content ]]   
You must return your response in the following JSON structure only:   
{   
"memory" : [   
{   
"id" : "<ID of the memory>",   
"text" : "<Content of the memory>",   
"event" : "<Operation to be performed>",   
"old\_memory" : "<Old memory content>"   
},   
...   
]   
}   
Follow the instruction mentioned below:   
- Do not return anything from the custom few shot prompts provided above.   
- If the current memory is empty, then you have to add the new retrieved facts to the memory.   
- You should return the updated memory in only JSON format as shown below. The memory key   
should be the same if no changes are made.   
If there is an addition, generate a new key and add the new memory corresponding to it.   
- If there is a deletion, the memory key-value pair should be removed from the memory.   
- If there is an update, the ID key should remain the same and only the value needs to be   
updated.   
Do not return anything except the JSON format.

Prompt for relation labeling and ROAM’s relation inference. The same prompt is used both to obtain the initial relation labels and to perform relation inference within ROAM.

```markdown
# Task
Compare two facts about the SAME user.
- `OLD FACT`: an existing memory.
- `NEW FACT`: a newly extracted candidate fact.
## Label Set (Return Exactly One)
- `IND`: the facts can both be true, are only loosely related, or have no entailment relation.
`EQV`: the two facts are semantically equivalent (same truth conditions, mutual entailment).
`OSN`: `NEW FACT` strictly entails `OLD FACT`, so NEW is stronger, narrower, or more
specific.
`NSO`: `OLD FACT` strictly entails `NEW FACT`, so NEW is weaker, broader, or less specific.
- `CON`: the two facts cannot both be true -- incompatible values for the same single-valued
attribute of the same aligned entity.
## Decision Process
1. Identify the aligned entity and attribute in each fact. If they are about clearly different
entities or different attributes, choose `IND`.
2. If they share the same aligned entity and attribute with **incompatible values**, choose
`CON`.
3. If they entail each other (same truth conditions, reworded), choose `EQV`.
4. If only `NEW FACT` entails `OLD FACT` (strict entailment -- NEW being true forces OLD to be
true), choose `OSN`.
5. If only `OLD FACT` entails `NEW FACT` (strict entailment -- OLD being true forces NEW to be
true), choose `NSO`.
6. Otherwise, choose `IND`.
## What Counts as Each Label
### EQV
Same truth conditions -- each entails the other. They say the same thing, possibly reworded.
- Example: OLD "the user works at Google." / NEW "the user is employed by Google." -> `EQV`.
What does NOT count as `EQV`:
- One fact is strictly more specific than the other (that is `OSN`/`NSO`, not `EQV`).
- The facts conflict (`CON`) -> reject EQV.
- The facts are merely related or about the same topic but not truth-conditionally identical
-> reject EQV.
- Different aligned entity or attribute -> not EQV.
### OSN (new entails old)
`NEW FACT` strictly entails `OLD FACT`: if NEW is true, OLD must also be true. NEW adds
specificity to OLD (same aligned entity and attribute).
- Example: OLD "the user has a pet." / NEW "the user has a golden retriever named Max." ->
`OSN`.
- Example: OLD "the user has a daughter." / NEW "the user has a daughter named Emma." -> `OSN`.
- Geographic containment: OLD "the user lives in New York City." / NEW "the user lives in
Manhattan." -> `OSN` (Manhattan is within NYC).
What does NOT count as `OSN`:
The entailment runs the other way (OLD entails NEW) -- that is `NSO`.
- The two are equivalent (`EQV`) -- not `OSN`.
- Only loose association or plausible implication, not strict entailment -> `IND`.
- Different aligned entity or attribute -> `IND`.
- An added detail that does not actually entail the old fact (e.g., adding an unrelated
property) -> `IND`.
### NSO (old entails new)
`OLD FACT` strictly entails `NEW FACT`: if OLD is true, NEW must also be true. NEW is
weaker/broader (same aligned entity and attribute).
```

```markdown
Example: OLD "the user lives in Manhattan." / NEW "the user lives in New York City." ->
`NSO` (Manhattan is within NYC).
- Example: OLD "the user has a golden retriever." / NEW "the user has a dog." -> `NSO`.
- Geographic containment and clear taxonomic "is-a" relations are acceptable as entailment.
What does NOT count as `NSO`:
- The entailment runs the other way (NEW entails OLD) -- that is `OSN`.
- The two are equivalent (`EQV`) -- not `NSO`.
- Only loose association, not strict entailment: OLD "the user studies computer science." /
NEW "the user is a software engineer." -> `IND`.
- Working-in vs living-in, liking vs doing, and similar non-entailing pairs -> `IND`.
- Different aligned entity or attribute -> `IND`.
### CON (contradiction)
Incompatible values for the same **single-valued attribute** of the same aligned entity.
Single-valued attributes are those a person normally has only one of: residence, employer,
job title, marital status, favorite X, current phone/email, age, etc.
**Time-explainable conflicts still count as `CON`.** Even if the two facts could be reconciled
by a change over time (moving, switching jobs, changing preferences), treat them as `CON`
-- the newer fact supersedes the older one.
- Example: OLD "the user lives in Beijing." / NEW "the user lives in Shanghai." -> `CON`.
Example: OLD "the user works at Google." / NEW "the user works at Meta." -> `CON`.
Example: OLD "the user's favorite color is blue." / NEW "the user's favorite color is red."
-> `CON`.
What does NOT count as `CON`:
- Different entities: OLD "the user's brother lives in Beijing." / NEW "the user lives in
Shanghai." -> `IND`.
- Multi-valued facts where both can hold: OLD "the user has a son named Max." / NEW "the user
has a son named Leo." -> `IND` (could have two sons, no stated uniqueness).
- Facts that are simply independent with no shared single-valued attribute -> `IND`.
- One fact refines or generalizes the other without conflicting (that is `OSN`/`NSO`/`EQV`) ->
not `CON`.
### IND (independent)
All other cases: both facts can hold at once, no strict entailment relation, or no shared
aligned entity/attribute.
- Different entities: OLD "the user's brother lives in Boston." / NEW "the user lives in
Boston." -> `IND`.
Same entity, no entailment: OLD "the user likes jazz." / NEW "the user has a sister named
Sue." -> `IND`.
- Related but non-entailing: OLD "the user studies computer science." / NEW "the user is a
software engineer." -> `IND`.
- Preference vs behavior: OLD "the user likes Italian food." / NEW "the user often eats
pizza." -> `IND`.
- Different time/context avoiding contradiction: OLD "the user lived in Paris in 2020." / NEW
"the user lives in Berlin now." -> `IND`.
## Important Constraints
- Use `OSN` and `NSO` **only for strict entailment**, not weak association or plausible
implication. When in doubt, choose `IND`.
- The two facts must share the same aligned entity and attribute for `CON`, `OSN`, `NSO`, or
`EQV` to apply. If not, choose `IND`.
- For `CON`: the attribute must be single-valued. Multi-valued/accumulable facts (having
multiple children, knowing multiple languages) cannot contradict.
- Time-explainable changes (moves, job changes, preference shifts) are `CON`, not `IND`.
## Output Format
- Return a JSON object with exactly one key: `"relation"`.
- The value must be one of: `IND`, `EQV`, `NSO`, `OSN`, `CON`.
- Do not output extra keys, explanation, or prose.
```

## Fusion prompts.

```markdown
Merge the following into **one memory line**. **Minimal length. Zero information loss.**
`CURRENT MEMORY`: the accumulated answer memory so far (may already combine several facts).
`NEW FACT`: a newly confirmed fact that the classifier labeled **equivalent** (`EQV`) to a
fact already in the current memory (same truth conditions, just reworded).
Two equivalent statements can mean two different things. Decide which case applies:
1. **Same fact restated** -- a stable attribute or a single past event mentioned again (e.g.
"works at Google", "visited Rome in 2019"). State it **once**; do not duplicate phrasings
and do not invent a count.
2. **A repeatable event that recurred** -- an action that can naturally happen more than once
(e.g. buying snacks, visiting a friend, going to the gym). If `CURRENT MEMORY` and `NEW
FACT` describe such an event happening on **different occasions**, record it as **multiple
occurrences** with an exact count (e.g. "the user bought spicy snacks twice").
Use the occurrence times to decide:
`CURRENT MEMORY` time: [[ current_memory_time if current_memory_time else "unknown" ]]
- `NEW FACT` time: [[ new_fact_time if new_fact_time else "unknown" ]]
## Rules
1. **Value chains MUST be preserved.** If `CURRENT MEMORY` contains an arrow-separated history
like `v1 -> v2 -> v3 (current)`, copy the ENTIRE chain verbatim.
2. **Counts and numbers are always distinct information.** Never drop a number. Never drop a
count.
3. For stable attributes (employment, age, location, preferences, relationships), never count
-- always state once. Keep the version that has more specific detail. If both have
different but not conflicting details (e.g., one mentions the company, the other mentions
the role), merge both details.
4. **Same-attribute check with different values:** If `CURRENT MEMORY` and `NEW FACT` describe
the same attribute but with different specific values (even though they are labeled EQV),
preserve both: `previously <old_value>; now <new_value>`. This is a safety net -- the
classifier should have used CON, but if EQV was chosen, do not lose information.
5. Count repeatable events conservatively:
- Count as a separate occurrence **only** when the event is genuinely repeatable AND the
two times are clearly different occasions.
- If the times are the same, missing, or it is plausibly the **same** event re-mentioned,
do **not** increase the count.
- When `CURRENT MEMORY` already states a count and `NEW FACT` is the same repeatable event
at a **different** time, increment the count directly ("twice" -> "three times").
6. Keep any other unrelated factual content from `CURRENT MEMORY` intact. Do not invent facts.
## Examples
CURRENT: "the user bought spicy snacks" (time 2024-01-01)
NEW: "the user bought spicy snacks" (time 2024-01-08)
-> "the user bought spicy snacks twice"
CURRENT: "the user works at Google" (time 2024-01-01)
NEW: "the user is employed by Google" (time 2024-03-01)
-> "the user works at Google"
CURRENT: "The user's 5K PB: 27:45 -> 26:30 (current, as of 2023/07/30)"
NEW: "the user achieved a personal best 5K time of 26:30" (time 2023/07/30)
-> "The user's 5K PB: 27:45 -> 26:30 (current, as of 2023/07/30)"
CURRENT MEMORY:
[[ current_memory ]]
NEW FACT:
[[ new_fact ]]
```

Merged line:

## EverMemOS-style factual-memory management prompt.

You are a memory consolidation assistant. The following facts were extracted from the same   
conversation and are semantically related -- they belong to the same topic or subject.   
Merge them into \*\*one concise, comprehensive sentence or short paragraph\*\* that captures all   
the information without repetition.   
Rules:   
- Preserve every distinct piece of information.   
- Do not invent facts not present in the list.   
- Do not include bullets, headings, or preamble -- output only the merged memory text.   
- If the facts are already covered by a single fact, return that fact as-is.   
Facts to merge:   
[[ facts\_text ]]   
Merged memory:

## Answer-generation prompt.

You are a memory-augmented assistant. Use the retrieved memory units to provide accurate and   
context-aware answers to the user's questions.   
[[ context\_block ]]   
### Question Details   
{% if question\_time %}   
- Current Date: [[ question\_time ]]   
{% endif %}   
Question: [[ question ]]   
Please give a short answer.

## DeepSeek V4 Flash answer-judging prompt.

You are given a question, its ground-truth answer, and a model response. Judge if the model   
response is semantically correct. Be lenient for wording differences if the core meaning   
is correct.   
\*\*Question\*\*: [[ question ]]   
\*\*Ground-truth answer\*\*: [[ reference ]]   
\*\*Model response\*\*: [[ candidate ]]