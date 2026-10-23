# ColaBench: Evaluating Memory-Guided Knowledge Acquisition in Personal Agents

Wenjie Zhang wenjie.zhang@stu.jiangnan.edu.cn Jiangnan University, Wuxi, China

Xiaojun Wu   
Cehao Yang   
xwu647@connect.hkust-gz.edu.cn   
cyang289@connect.hkust-gz.edu.cn   
DataArc Tech Ltd., China   
IDEA Research, International Digital   
Economy Academy, Shenzhen, China   
The Hong Kong University of Science   
and Technology (Guangzhou),   
Guangzhou, China   
Zhichao Shi   
shizhichao22s@ict.ac.cn   
School of Advanced Interdisciplinary   
Sciences, UCAS, Beijing, China   
State Key Lab of AI Safety, Institute of   
Computing Technology, CAS, Beijing,   
China   
DataArc Tech Ltd., China   
IDEA Research, International Digital   
Economy Academy, Shenzhen, China Chengjin Xu xuchengjin@idea.edu.cn DataArc Tech Ltd., China   
IDEA Research, International Digital   
Economy Academy, Shenzhen, China Xuhui Jiang<sup>✉∗</sup> jiangxuhui@idea.edu.cn DataArc Tech Ltd., China   
IDEA Research, International Digital   
Economy Academy, Shenzhen, China

Xiaoning Song<sup>✉</sup> x.song@jiangnan.edu.cn Jiangnan University, Wuxi, China

## Abstract

Personal agents need to draw on long-term interaction history to retrieve and apply external knowledge suited to a user’s current request. Even an answer grounded in relevant documents can be unsuitable when it overlooks the user’s circumstances. This creates two connected challenges: using historical conditions to identify the knowledge needed for a new request, and applying that knowledge to the user’s situation. Existing benchmarks advance memory recall, personalized retrieval, and memory-guided action, while the joint evaluation of these two challenges remains underexplored. We introduce ColaBench, a synthetic benchmark in health and daily living, with 349 tasks across 10 Profiles and a knowledge base of 13,033 documents. Its Evidence Graph connects personal condi tions to their disclosures across public interaction sessions, records prior document exposure, and links answer requirements to personal and newly acquired document evidence. We also introduce MemKB, which uses state summaries to inform queries and rule application, and memory–document links to retrieve knowledge associated with the user’s history. MemKB achieves 86.25% Accuracy with an average of 4.07 interaction rounds per task. Task-level analysis finds larger gains on tasks integrating information across multiple historical Facts; error analysis further identifies unsup ported personal assumptions and rule misapplication as obstacles to using retrieved evidence. ColaBench provides traceable tasks

and an evaluation framework for studying how memory guides knowledge acquisition and rule application.

## CCS Concepts

• Information systems → Personalization; • Computing methodologies → Intelligent agents.

## Keywords

agents, long-term memory, knowledge acquisition, data synthesis, benchmarks

## 1 Introduction

Personal agents must draw on long-term interaction history to handle users’ evolving requests [10, 19]. Users often leave prior experiences, preferences, and constraints unstated [6], while the current task may require new external knowledge [8]. Consider a user asking whether they qualify for a caregiver respite grant after mentioning, in separate conversations, that they care for a relative with Alzheimer’s disease and live alone. The agent must find the grant’s eligibility rules and connect the co-residence requirement to the earlier disclosure of living alone (Table 1). Overlooking that connection can turn relevant information into unsuitable advice. Reliable personalization therefore depends on how history guides knowledge acquisition and how the acquired knowledge is applied to the user.

These tasks pose two connected challenges. Knowledge acquisition requires recovering the historical conditions that determine what to search for, even when the current request leaves them implicit. Rule application requires matching retrieved prerequisites, exceptions, and procedures to known personal circumstances while preserving what remains unknown. The process can move in both directions: history shapes the search, and a newly retrieved rule can make an earlier disclosure relevant. When memory and knowledge are retrieved separately, the agent must establish these connections through its queries and reasoning.

Existing benchmarks address important parts of this process. LoCoMo and LongMemEval provide a foundation for evaluating memory through cross-session question answering, temporal reasoning, and state updates [9, 16]. TREC iKAT brings user context, query reformulation, and external retrieval into the answering process [1]. Mem2ActBench and MemoryArena examine how memory guides subsequent tool use and interdependent actions [5, 15]. A remaining gap concerns the joint evaluation of how disclosed personal conditions guide the acquisition and application of knowledge not yet encountered in the interaction. This requires tasks that trace what the user has disclosed, what documents the agent has already encountered, and which personal and knowledge premises support each required conclusion. These dependencies make it possible to examine where the connection between history and knowledge succeeds or breaks down.

We introduce ColaBench, a benchmark built around these dependencies. Its Evidence Graph connects personal Facts to the public messages that disclose them, records document exposure during interaction, and links subsequent answer requirements to their personal and knowledge sources. Its document links guide knowledge exploration, and its disclosure and exposure records constrain staged, multi-agent task construction to disclosed personal conditions and supporting documents not previously exposed in the history. The resulting benchmark comprises 349 enhanced and validated Items across 10 Profiles and a knowledge base of 13,033 documents. The graph makes each task’s evidence dependencies explicit, supporting evaluation of knowledge acquisition and rule application and diagnosis of whether errors arise from missing personal conditions, missing knowledge, or their use in the answer.

We introduce MemKB, a memory-guided knowledge retrieval method that connects historical context to knowledge candidates. A user state summary brings dispersed personal conditions into query formulation and subsequent judgment. MemKB encodes memory text, establishes memory–document links, and fuses documents associated with retrieved memories with direct semantic and keyword retrieval results for the current query. The answering agent combines retrieved memories and document evidence to assess which rules apply to the user’s circumstances.

MemKB achieves 86.25% Accuracy, compared with 79.94% for Mem0+Hybrid using the same underlying memories, a gain of 6.30 percentage points, while reducing average interaction rounds from 4.44 to 4.07. The gain is larger on tasks requiring multiple historical Facts: 9.35 percentage points, compared with 4.96 on single-Fact tasks. Case and error analyses show how personal assumptions and rule misapplication can afect answers even when relevant evidence is available. These findings motivate evaluating memory through the knowledge it helps agents acquire and the user-specific judgments it supports.

Our contributions are as follows:

• Evidence Graph–driven benchmark synthesis. We develop a staged, multi-agent synthesis framework in which accumulated Fact disclosure and knowledge exposure constrain subsequent Item construction, producing the traceable ColaBench benchmark and its evaluation protocol.

• Memory-guided knowledge retrieval. We introduce MemKB, which combines state summaries and memory– document links to guide knowledge retrieval and support user-specific rule application.

• Empirical analysis of memory utility. Through taskstructure groupings, component ablations, and evidence traces, we characterize memory utility in knowledge acquisition and rule application: gains on tasks integrating multiple historical Facts and failures involving omitted personal conditions, unsupported assumptions, and rule misapplication after knowledge access.

## 2 Related Work

Long-Term Memory and Task Execution. Memory architectures organize interaction history for subsequent decisions. Generative Agents combine experience retrieval and reflection to support planning [11]; MemoryBank updates user memories over time [19], and MemGPT manages information across memory tiers [10]. Mem0 supports memory extraction, consolidation, and retrieval [2]. A-MEM builds evolving links between memory notes [17], while Graphiti, the graph engine underlying Zep, maintains temporal entity and fact relations [12].

LoCoMo generates long-term conversations from profiles and temporal event graphs, with human review, to evaluate question answering, event summarization, and multimodal dialogue generation [9]. LongMemEval covers cross-session reasoning, temporal information, and state updates, and analyzes memory systems at the indexing, retrieval, and reading stages [16]. Mem2ActBench examines how memory supports subsequent tool selection and argument specification [15], and MemoryArena evaluates experience reuse across interdependent multi-session tasks [5]. ColaBench studies subsequent actions through knowledge search, evidence reading, and rule application, tracing how recovered user state supports each required conclusion.

Personalized Conversational Search. LaMP studies retrieval of user-specific records for personalized classification and generation [14]. PersonaMem evaluates profile tracking and response selection as users’ circumstances evolve [6]. TREC iKAT is a closely related benchmark for personalized knowledge acquisition. Personal textual knowledge bases specify user backgrounds, and conversational tasks require systems to use this personal context. Evaluation covers passage relevance as well as the relevance, completeness, grounding, and naturalness of generated answers [1]. ColaBench records personal-state disclosure and knowledge exposure across simulated interactions. Its tasks trace evidence from public user statements through rule premises to required judgments, enabling evaluation of history-guided knowledge retrieval and rule application.

Evidence-Level Diagnosis. Retrieval-augmented generation uses external documents to generate answers [8]. HippoRAG combines knowledge graphs and Personalized PageRank to integrate evidence across passages for multi-hop question answering [7]. ALCE evalu ates answer correctness and the quality of supporting citations [4]. RAGChecker uses fine-grained claims to diagnose retrieval coverage and evidence use during generation, helping identify cases where relevant documents have been retrieved but required content is omitted [13]. We use necessary answer requirements as diagnos tic units and connect them to the sources of personal conditions and the applicability conditions of rules. This enables us to examine whether the required evidence was obtained and correctly applied.

Table 1: A ColaBench task (paraphrased).
<table><tr><td>Component</td><td>Description</td></tr><tr><td>Request</td><td>Am I eligible for an HFC caregiver respite care grant?</td></tr><tr><td>History</td><td>In a meal-preparation conversation, the user states that they live alone. In another exchange, they mention caring for a relative with Alzheimer&#x27;s disease.</td></tr><tr><td>Rule</td><td>The care recipient must live at home with the applicant.</td></tr><tr><td>Judgment</td><td>Living alone does not satisfy the co-residence require- ment; other undisclosed conditions remain unknown.</td></tr></table>

## 3 Evidence-Graph-Based Construction

ColaBench uses a Plan–Create–Review multi-agent framework with stage-specific roles and information access to ground synthetic personal backgrounds, interaction histories, and tasks in retrieved documents. An Evidence Graph $\mathcal { G } = ( V , E )$ begins with documents and their relationships and expands throughout construction. The graph incorporates Document, Profile, Fact, Topic, Turn, and Item nodes. Document links guide knowledge exploration, while Fact–Topic relations organize user states and interaction tasks. Records of fact disclosure and document exposure determine which personal conditions and new knowledge are available for later Items. Figure 1 shows the roles and information boundaries across document preparation, the three generation stages, and Item enhancement.

## 3.1 Task Definition

The evaluation tasks examine how an agent uses personal conditions from history to acquire and apply external knowledge. We denote the task setting by $x = ( q , { \mathcal { H } } , { \mathcal { K } } ; M )$ , where � is the current request, H is the preceding public interaction history, K is the frozen knowledge base, and M denotes the evaluation metrics defined in Section 5.2.

Personal conditions can guide knowledge search, and newly retrieved rules can prompt the agent to revisit history. Table 1 illustrates this interaction through a grant’s co-residence requirement.

## 3.2 Stage 0: Building the Document Layer

We collect public health information, drug labels, and open textbooks from screened public sources through APIs, XML files, and web pages. After text cleaning, the base corpus contains 11,069 documents. Each document forms an initial Document node in the Evidence Graph, retaining its text and source information. Appendix A.1 details the corpus sources and processing.

Starting from seed documents, an LLM retrieves and reads related documents to establish edges grounded in specific knowledge relationships, storing a description and weight for each relation. We further search, fetch, and filter web pages on relevant topics, incorporating the resulting documents into the corpus and graph. This document layer supplies knowledge and exploration links for Profile generation and Item construction. Web pages discovered during these stages are also added as Document nodes.

## 3.3 Stage I: Grounding Profiles and Facts

To construct internally consistent users with varied life circumstances, a Profile construction agent, the Director, plans new scenarios from the existing profile and consults retrieved documents to create Facts describing personal conditions and Topics specifying interaction tasks. An independent review agent checks consistency, relevance, and task completeness against the existing state and source material. Accepted content is added to the Evidence Graph. Documents ground the synthetic scenarios in relevant contexts and knowledge, while links among Profiles, Facts, Topics, and Documents support continued profile development across life domains and provide evidence for subsequent interactions.

## 3.4 Stage II: Growing Public Interaction Traces

Interactions between a user Simulator and an Assistant gradually disclose personal conditions as tasks unfold. The Simulator has access to the frozen Profile and Topic, whereas the Assistant responds using the public conversation and retrieved knowledge. The Simulator follows up or switches Topics in response to the Assistant’s actual replies, producing a continuous Trace driven by task progress. Accepted interactions enter the Evidence Graph as Turn nodes, recording Fact disclosures and document exposure to establish the history available for subsequent construction.

## 3.5 Stage III: Constructing Items

To construct an initial Item, the Builder starts from disclosed personal conditions and seeks new knowledge whose application requires those conditions, establishing the task and its answer evidence. An independent question-writing role then formulates a natural request from the public background and task intent, leaving the necessary personal context in history. A history-removal test checks whether the task can be completed without the public interaction history and whether an unmet answer requirement depends on missing historical conditions. Subsequent review assesses source support, knowledge novelty, and continuity with the preceding interaction. Accepted Items enter the Evidence Graph with links to the required Facts, historical Turns, and supporting Documents, providing a traceable account of how personal conditions afect the application of new knowledge. Construction details and evidence annotations appear in Appendix A.

## 3.6 Item Enhancement and Validation

An enhancement agent examines each initial Item’s historical conditions and knowledge rules to develop decision tasks that combine multiple personal conditions (Figure 1(e)). With the public interaction history fixed, it chooses one of two enhancement directions: reorganizing condition dependencies within the original evidence or replacing the supporting document with a retrieved candidate from the same Topic. It updates the question, reference answer, and answer requirements accordingly. An independent History Gate and evidence review assess history dependence, source sup port, and the combination of conditions. Appendix A.4 specifies the acceptance and validation procedures.

![](images/eef81da1f33e8f96b21efdd32db242112015d9f0fbc6566fd7c458ed7cb2eac3.jpg)  
Figure 1: Plan–Create–Review construction: (a) document collection; (b) grounded Profiles and Topics; (c) public interaction; (d) history-dependent Item construction; and (e) Item enhancement and independent validation. The growing Evidence Graph guides knowledge exploration and constrains generation and review through recorded Fact disclosures, document exposure, and supporting evidence.

## 3.7 Dataset Overview

ColaBench contains 10 Profiles, 349 Items, and 13,033 documents, as summarized in Table 2. A multi-Fact task cites at least two annotated Facts; a task citing multiple sessions draws its annotated historical evidence from at least two sessions. Appendix A.2 provides detailed structural statistics; Table 10 in Appendix C.3 illustrates tasks sharing personal backgrounds.

## 4 MemKB: Memory-Linked Retrieval

MemKB connects two decisions: which knowledge the user’s request requires and how the retrieved rules apply to their circumstances. MemKB supplies a user state summary before query generation and retrieves knowledge through explicit links between memory cards and documents. It then fuses these candidates with documents retrieved directly for the current query (Figure 2).

Table 2: Structural statistics of ColaBench.
<table><tr><td>Statistic</td><td>Value</td></tr><tr><td>Profiles / Sessions / Messages</td><td>10 / 21 / 1,198</td></tr><tr><td>Facts / Topics / Items</td><td>52 / 600 / 349</td></tr><tr><td>Base / Added web documents</td><td>11,069 / 1,964</td></tr><tr><td>Items with multiple Facts</td><td>107</td></tr><tr><td>Items citing multiple sessions</td><td>153</td></tr></table>

## 4.1 Memory Cards and Document Links

Given a public interaction history H and a frozen knowledge base K, we use Mem0 [2] to build personal memory incrementally and retain its write records. These records link each historical interaction to its surviving memory IDs, forming cards �<sub>�</sub> with source message IDs and normalized embeddings of their memory text. The document identifiers and start positions in the interaction’s public

![](images/172357a2839a2c031e17654ecf8806bbc6066d5c434266a2dee54023da18f577.jpg)  
Figure 2: MemKB construction and answering. (a) Public history yields state summaries and memory cards linked to historical Read documents and unexposed KB neighbors. (b) States inform the first query; round-robin link expansion and equal-weight RRF combine memory-linked, semantic, and keyword candidates. Search returns memories, states, and document snippets.

Read records allow the builder to recover short snippets from the frozen KB and establish document links.

For each card, the builder queries the full KB using memory text together with historical document titles and snippets. It retains top semantic candidates not previously exposed in the Profile’s history as ofline document links. These neighbors and historical Read documents form the card’s linked document set, connecting memory to both previously used and unexplored knowledge. Neigh bor discovery uses public interaction history and the full frozen KB, while reference-document annotations are withheld from the builder. Encoder settings and budgets appear in Appendix B.2.

A language model extracts a compact state bank � (H ) from the complete public interaction history, preserving who each condition concerns, its temporal context, unresolved conditions, and source message IDs. The complete � (H ) is added to the system prompt before the first model call, allowing personal conditions to inform the initial query. Each search returns the same complete state bank to support interpretation of retrieved evidence and assessment of rule applicability. Query-relevant cards add memory text, sourcemessage IDs, and knowledge links.

## 4.2 Memory-Linked Knowledge Retrieval

Hybrid retrieves documents directly from the KB through semantic and keyword channels. At retrieval step �, the agent uses the current request � and information acquired so far to generate a semantic query �<sub>�</sub> and a keyword set $w _ { t }$ . Vector similarity and term matching produce candidate lists $L _ { \mathrm { s e m } }$ and $L _ { \mathrm { k e y } }$ , respectively, whose rankings are then fused. In baselines with separate memory and knowledge retrieval, a dedicated tool returns personal memory, which influences knowledge retrieval through the queries formulated by the agent.

MemKB adds a third retrieval channel that transfers query relevance from memory cards to linked documents. The same �<sub>�</sub> retrieves cards by normalized inner product. Following card rank order, a round-robin procedure takes one document from each card per pass, skipping documents already in the candidate list, until the candidate budget is reached or all cards are exhausted. This produces $L _ { \mathrm { m e m } }$ . Distributing candidate slots across relevant cards allows knowledge associated with diferent historical conditions to enter the candidate set.

Reciprocal rank fusion (RRF) [3] combines the memory-linked candidates with Hybrid’s two direct retrieval channels, incorporating both associations with personal history and direct matches to the current query:

$$
s ( d ) = \sum _ { c \in \{ \mathrm { m e m , s e m , k e y } \} } \frac { 1 [ d \in L _ { c } ] } { \kappa + \mathrm { r a n k } _ { L _ { c } } ( d ) } .\tag{1}
$$

Here, $L _ { c }$ is the candidate list for channel $c ,$ � is a rank-smoothing constant, and ranks start at 1. Each document contributes at most once per channel, and all three channels receive equal weight. The fused ranking determines the returned documents, each accompanied by a relevant snippet, its document ID, and a line range.

A call to memkb\_search returns memory text and source identifiers from matched cards, �(H), and the fused document snippets.

## 4.3 From User State to Rule Application

The same answering agent compares the conditions specified by retrieved rules with the user’s state and memory records, continuing to search or call Read as needed. Its answering prompt includes a source-attribution constraint: distinguish the speaker, actor, and event status in memory records, and ground personal communications and decisions in the records that support them. When the source or event status is unspecified, the agent is instructed to preserve that uncertainty. This constraint guides the agent to retain the provenance of personal conditions when applying rules.

## 5 Experimental Setup

## 5.1 Methods and Experimental Settings

We pair Mem0, Graphiti, and A-MEM with a general-purpose retrieval-augmented generation component, Hybrid, and compare them with No Memory+Hybrid and MemKB. Hybrid combines semantic and keyword retrieval over the KB through RRF [3].

The Mem0+Hybrid baseline uses Mem0’s non-graph memory [2]. Graphiti+Hybrid retrieves facts about entities and their relations extracted from conversations [12], while A-MEM+Hybrid uses memory evolution and neighbor expansion [17]. These three methods build memory from the same public user and assistant messages. MemKB additionally uses public records of historical document access to build a memory-linked retrieval channel and supplies a state summary before the first query.

The main comparison uses all 349 tasks from 10 Profiles and the same frozen KB across methods. All methods can make tool calls for up to four rounds per task, followed by a final answering round. Knowledge search returns five documents by default, with a maximum of 20. Read returns at most 100 lines and 3,000 tokens per call.

All methods use DeepSeek v4.1 Flash for answering. GLM-5.3 Flash evaluates each answer in a separate call with low reasoning efort. Model parameters appear in Appendix B.2, and the shared evaluation protocol appears in Appendix D.1.

## 5.2 Evaluation Metrics

Accuracy measures whether an answer correctly completes the current task. A passing answer must address all explicitly requested subquestions and provide correct core conclusions and key actions, including the relevant personal conditions, applicability requirements, and limitations. The evaluator checks factual errors and omissions by asking whether correcting them would change the conclusion, action, timing, applicability, or safety. Errors and omissions that afect task completion lead to failure; incidental diferences are recorded separately. Grounded equivalent wording and implicit use of personal conditions are accepted, while unknown facts require the necessary conditional branches. Evaluation inputs combine the full public history, reference answer, requirements, historical and task-supporting documents, and each method’s actual Search and Read returns. Metric and input details appear in Appendix D.1.

Read Recall measures coverage of annotated new-support documents $G _ { i }$ by documents $D _ { i } ^ { \mathrm { r e a d } }$ for which Read returns nonempty text before the final answer. Each task has one reference document.

Table 3: Main comparison on 349 tasks. Acc. and Read Recall are percentages; rounds include the final answer.
<table><tr><td>Method</td><td>Acc. ↑</td><td>Read Recall ↑</td><td>Avg. Rounds ↓</td></tr><tr><td>No Memory+Hybrid [3]</td><td>6.59</td><td>37.25</td><td>4.85</td></tr><tr><td>Graphiti+Hybrid [12]</td><td>80.80</td><td>41.83</td><td>4.58</td></tr><tr><td>A-MEM+Hybrid [17]</td><td>81.38</td><td>39.26</td><td>4.42</td></tr><tr><td>Mem0+Hybrid [2]</td><td>79.94</td><td>44.70</td><td>4.44</td></tr><tr><td>MemKB (ours)</td><td>86.25</td><td>45.56</td><td>4.07</td></tr></table>

Table 4: Model-assessed answer-quality scores on 111 tasks from three Profiles. Each dimension is scored from 1 to 5.
<table><tr><td>Dimension</td><td>Mem0+Hybrid</td><td>MemKB</td><td>Gain</td></tr><tr><td>State Fidelity</td><td>4.523</td><td>4.595</td><td>+0.072</td></tr><tr><td>Condition Coverage</td><td>4.532</td><td>4.739</td><td>+0.207</td></tr><tr><td>Evidence Grounding</td><td>4.252</td><td>4.405</td><td>+0.153</td></tr><tr><td>Rule Application</td><td>4.396</td><td>4.604</td><td>+0.207</td></tr><tr><td>Answer Completeness</td><td>4.586</td><td>4.766</td><td>+0.180</td></tr></table>

Across � evaluated tasks,

$$
\mathrm { R e a d R e c a l l } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \frac { | G _ { i } \cap D _ { i } ^ { \mathrm { r e a d } } | } { | G _ { i } | } .\tag{2}
$$

Search snippets and information already present in memory can also support an answer directly, so we analyze document coverage alongside Accuracy.

Avg. Interaction Rounds is the mean number of agent rounds, including the final answering round.

We additionally assess five answer-quality dimensions. Each dimension receives a separate blinded GLM-5.3 Flash (low) call using the same reference inputs and candidate-visible evidence as Accuracy. Scores range from 1 to 5, with higher scores indicating better quality; Appendix D.2 defines the dimensions.

## 6 Results and Analysis

## 6.1 Answer Quality and Knowledge Access

On 349 tasks (Table 3), MemKB achieves 86.25% Accuracy, exceeding Mem0+Hybrid by 6.30 percentage points despite close Read Recall (45.56% versus 44.70%). This gap motivates examining how answers use personal conditions and retrieved evidence. No Memory+Hybrid achieves 6.59% Accuracy versus Graphiti+Hybrid’s 80.80%, illustrating the importance of recovering personal context. MemKB exceeds A-MEM+Hybrid by 4.87 percentage points and uses 0.35 and 0.37 fewer interaction rounds than A-MEM+Hybrid and Mem0+Hybrid, respectively.

Condition Coverage and Rule Application. Table 4 evaluates the five dimensions on 111 tasks from three Profiles. MemKB scores higher across all dimensions, with the largest gains in Condition Coverage and Rule Application (both +0.207). On the 16 tasks answered correctly by MemKB and incorrectly by Mem0+Hybrid, the mean Rule Application gain is 1.563 points. These results link memory use to identifying needed knowledge and applying it to the user’s circumstances.

(d) Composition sensitivity

Mem0+Hybrid MemKB

![](images/22d5154cfe52c7698040340906f3249e3a3dddad835f7469dae13a3e7a6a61b8.jpg)

![](images/588172b661a2d8b126b5b0c75c9e1457a48ae12977eb2c3f2a96448cc248c9c8.jpg)

![](images/b1a36e62fbbebd39a11627e9598074bdc4b87e276c6dab9b91d74452997ad654.jpg)

![](images/9c716a407b693e0d35d8a8764dab0e780a721caaabcc01ad507e6a7c22156d8c.jpg)  
Figure 3: Task structure and evidence access on 349 tasks. (a,b) Accuracy by Fact and history-pair counts. (c) Both exposed: both methods receive the reference document through search or Read; both read: through Read. (d) Paired gains under common composition weights with 95% Profile-cluster bootstrap intervals. pp: percentage points; req.: answer requirements.

## 6.2 Combining Personal Information

Personal Conditions and Their History Sources. We group tasks by the number of annotated Facts and cited user–assistant interaction pairs to examine tasks that combine more personal information. In Figure 3(a,b), the Accuracy gain is 4.96 percentage points on single-Fact tasks and 9.35 points on multi-Fact tasks. The gain is 5.33 points on tasks citing one or two interaction pairs, and 8.57 points on tasks citing at least three pairs. These groupings overlap: 91 multi-Fact tasks also cite at least three interaction pairs. MemKB gains more when tasks combine personal information.

Sensitivity to Task Composition. Stratifying by Profile and answerrequirement count and applying common weights to strata containing both Fact groups yields gains of 4.87 and 14.41 percentage points on 316 tasks (Figure 3(d)). Adding construction cohort yields gains of 5.86 and 12.24 points for single-Fact and multi-Fact tasks on 275 tasks. Appendix C.2 details the strata and bootstrap intervals. Larger observed gains across these analyses highlight the value of organizing multiple historical conditions for knowledge selection and rule application.

## 6.3 Shared Access and Answer Quality

We examine answer quality on subsets defined by both methods observed reference-document access (Figure 3(c)). The 136 tasks satisfying Both exposed are those where each method receives the reference document through search or a nonempty Read response. Accuracy is 86.03% for Mem0+Hybrid and 90.44% for MemKB. On the 108 tasks satisfying Both read, where each method receives document text through Read, Accuracy is 87.96% and 90.74%, respectively. Even with access to the same document, the visible passages, available personal conditions, and their application can difer. Together with the following cases, these answer gaps highlight the role of personal conditions in applying retrieved knowledge.

## 6.4 Case Studies: Rule Application

Table 5 contrasts recovery of personal history with a shared failure to preserve an unknown condition; Appendix C.4 provides source evidence.

Personal History Beyond the Query Topic. Despite reading the same contraindication document, Mem0+Hybrid’s two memory searches emphasize teaching content, while its answer claims no record of cancer treatment. MemKB’s state summary preserves the user’s cancer history; its answer distinguishes past from current treatment and keeps unknown treatment and medication details conditional.

Handling Unconfirmed Personal Conditions. Both methods correctly answer the naloxone-access question but assert opioid use to assess personal risk. MemKB’s state summary says “prescription pain medicine,” yet its answer assigns an opioid class. Prior discussion of a drug class does not establish personal use: applying a rule requires grounding known conditions and retaining conditional branches for unknown premises.

## 6.5 State Summaries and Document Links

We evaluate component ablations on 111 tasks from three Profiles. The ablations remove either the state summary or document links, while retaining the answering model, evaluation criteria, and interaction budget in Section 5. The ablation setup is detailed in Appendix B.3.

Removing state summaries and document links reduces Accuracy by 6.31 and 10.81 percentage points, respectively (Table 6).

Organizing Personal Conditions. Removing the state summary eliminates state information both before the first query and from search responses, while preserving query-relevant memory text. The first-search hit rate decreases from 36.94% to 22.52%, and average interaction rounds increase from 4.27 to 4.49. Organizing these conditions explicitly helps the agent locate relevant knowledge earlier and carry personal conditions into its subsequent judgments.

Connecting Memories to Knowledge. Removing document links preserves the state summary and returned memory text, with knowledge candidates supplied by semantic and keyword retrieval. Read Recall decreases from 42.34% to 36.94%, while the search hit rate over the full interaction increases from 46.85% to 61.26%. These divergent trends distinguish search exposure from subsequent read ing: with document links, MemKB reads the reference document on more tasks and completes more tasks correctly.

Table 5: Paired cases of personal context and rule application. Responses are paraphrased.
<table><tr><td>Task and decisive premise</td><td>Mem0+Hybrid</td><td>MemKB</td></tr><tr><td>Breastfeeding suitability. The user has a history of childhood cancer and works in maternal and child health. Applying the guidance requires distinguishing past from current treatment.</td><td>Retrieved memories emphasize teaching experience. The answer claims no record of cancer treatment, omitting personal history relevant to rule applicability. Verdict: incorrect.</td><td>The initial state summary preserves the cancer history. The answer distinguishes past from current treatment and keeps unknown treatment details conditional. Verdict: correct.</td></tr><tr><td>Keeping naloxone at home. The user takes a prescription pain medicine of unspecified class. A personal recommendation depends on opioid use and overdose risk.</td><td>Reads the relevant guidance, then asserts in the answer that the user takes an opioid. The risk discussion proceeds from this unconfirmed premise. Verdict: incorrect.</td><td>The state summary records a prescription pain medicine. The answer still asserts opioid use, leaving dose and concurrent medicines as the unresolved details. Verdict: incorrect.</td></tr></table>

Table 6: Ablations on 111 tasks from three Profiles; changes from MemKB appear in parentheses (pp or rounds).
<table><tr><td>Method</td><td></td><td>Acc. ↑ Read Recall ↑ Avg. Rounds ↓</td><td></td></tr><tr><td>MemKB</td><td>91.89</td><td>42.34</td><td>4.27</td></tr><tr><td>w/o States</td><td>85.59(-6.31)</td><td>40.54(-1.80)</td><td>4.49 (+0.22)</td></tr><tr><td>w/o Links</td><td>81.08 (-10.81)</td><td>36.94(-5.41)</td><td>4.41 (+0.14)</td></tr></table>

<table><tr><td colspan="2">Not visible</td><td>Partly visible</td><td>Visible</td><td>Total</td></tr><tr><td>Personal Assumptions</td><td>1</td><td>10</td><td>6</td><td>17 (35.4%)</td></tr><tr><td>Answer Omissions</td><td>7</td><td>5</td><td>1</td><td>13 (27.1%)</td></tr><tr><td>Rule Misapplication</td><td>2</td><td>3</td><td>7</td><td>12 (25.0%)</td></tr><tr><td>Context Underuse</td><td>0</td><td>3</td><td>3</td><td>6 (12.5%)</td></tr></table>

Figure 4: Primary error categories and model-assessed evidence visibility for 48 MemKB failures. Cells show counts; row totals include percentages. Other Issues has zero cases.

Together, the 111-task ablations support using organized personal conditions and memory-linked knowledge to guide acquisition and ground rule application.

## 6.6 Error Analysis

We analyze the 48 failed MemKB answers using GLM-5.3 Flash (low), with the complete evaluation references and actual answering traces. Figure 4 relates each answer’s primary error category to the visibility of its key evidence. Appendix D.3 defines the categories, primary-category selection, and evidence visibility.

Personal Assumptions is the largest category. Another Profile’s naloxone task illustrates an error during state extraction. The user confirms taking a prescription pain medicine, but the derived state assigns it to the opioid class. The answer then treats opioid-related guidance as personally applicable. This error begins in the memory representation; the case in Section 6.4 illustrates an unsupported inference during answering. Both show why the source and uncertainty of personal conditions matter when applying rules.

Answer Omissions and Rule Misapplication expose diferent failure points. A scalp-cooling answer omits a contraindication absent from its retrieved context, whereas a neuropathy answer recommends foot baths despite having read the relevant heat-risk guidance. The former reflects missing decision-relevant evidence; the latter reflects incorrect application of a visible rule to the user.

These errors show that memory-guided knowledge tasks require both decision-relevant evidence and a faithful account of the user’s conditions when applying it.

## 7 Discussion and Limitations

Method development and comparison use ten Profiles, with component ablations on three selected Profiles. Generalization to new users and the reliability of automated evaluation require independent data and human calibration. The frozen health and everydaylife corpus makes document versions, sources, and reading scope reproducible; generalization to other domains and live Web environments remains to be evaluated. Knowledge comes from real documents, while interactions encode synthetic personal states. Task answers have not been clinically validated.

## 8 Conclusion

ColaBench evaluates memory-guided knowledge acquisition and application through an Evidence Graph linking answer requirements to historical and document evidence. MemKB improves task accuracy with fewer interaction rounds, with ablations supporting the contributions of state summaries and document links. Together, they connect memory evaluation to the knowledge agents acquire and the user-specific judgments they make.

## References

[1] Mohammad Aliannejadi, Zahra Abbasiantaeb, Shubham Chatterjee, Jefery Dalton, and Leif Azzopardi. 2024. TREC iKAT 2023: A Test Collection for Evaluating Conversational and Interactive Knowledge Assistants. In Proceedings ofthe 47th International ACM SIGIR Conference on Research and Development in Information Retrieval. Association for Computing Machinery, New York, NY, USA, 11 pages. doi:10.1145/3626772.3657860

[2] Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. 2025. Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory. arXiv:2504.19413 https://arxiv.org/abs/2504.19413

[3] Gordon V. Cormack, Charles L. A. Clarke, and Stefan Büttcher. 2009. Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods. In Proceedings ofthe 32nd International ACM SIGIR Conference on Research

and Development in Information Retrieval. https://plg.uwaterloo.ca/\~gvcormac/ cormacksigir09-rrf.pdf

[4] Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi Chen. 2023. Enabling Large Language Models to Generate Text with Citations. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, Singapore, 6465–6488. doi:10.18653/v1/2023.emnlp main.398

[5] Zexue He, Yu Wang, Churan Zhi, Yuanzhe Hu, Tzu-Ping Chen, Lang Yin, Ze Chen, Tong Arthur Wu, Siru Ouyang, Zihan Wang, Jiaxin Pei, Julian McAuley, Yejin Choi, and Alex Pentland. 2026. Benchmarking Agent Memory in Interdependent Multi-Session Agentic Tasks. In Proceedings ofthe 43rd International Conference on Machine Learning (Proceedings ofMachine Learning Research, Vol. 306). PMLR, 41975–42005. https://proceedings.mlr.press/v306/he26am.html

[6] Bowen Jiang, Zhuoqun Hao, Young-Min Cho, Bryan Li, Yuan Yuan, Sihao Chen, Lyle Ungar, Camillo J. Taylor, and Dan Roth. 2025. Know Me, Respond to Me: Benchmarking LLMs for Dynamic User Profiling and Personalized Responses at Scale. In Second Conference on Language Modeling. https://openreview.net/ forum?id=6ox8XZGOqP

[7] Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su. 2024. HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models. In Advances in Neural Information Processing Systems, Vol. 37. doi:10.52202/079017-1902

[8] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. In Advances in Neural Information Processing Systems, Vol. 33. https://arxiv.org/abs/2005.1140

[9] Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Bar bieri, and Yuwei Fang. 2024. Evaluating Very Long-Term Conversational Memory of LLM Agents. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, Bangkok, Thailand, 13851–13870. doi:10.18653/v1/2024.acl-long.747

[10] Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, and Joseph E. Gonzalez. 2023. MemGPT: Towards LLMs as Operating Systems. arXiv:2310.08560 https://arxiv.org/abs/2310.08560

[11] Joon Sung Park, Joseph O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S. Bernstein. 2023. Generative Agents: Interactive Simulacra of Human Behavior. In Proceedings ofthe 36th Annual ACM Symposium on User Interface Software and Technology. Association for Computing Machinery, New York, NY, USA, Article 2, 22 pages. doi:10.1145/3586183.3606763

[12] Preston Rasmussen, Pavlo Paliychuk, Travis Beauvais, Jack Ryan, and Daniel Chalef. 2025. Zep: A Temporal Knowledge Graph Architecture for Agent Memory. arXiv:2501.13956 https://arxiv.org/abs/2501.13956

[13] Dongyu Ru, Lin Qiu, Xiangkun Hu, Tianhang Zhang, Peng Shi, Shuaichen Chang, Cheng Jiayang, Cunxiang Wang, Shichao Sun, Huanyu Li, Zizhao Zhang, Binjie Wang, Jiarong Jiang, Tong He, Zhiguo Wang, Pengfei Liu, Yue Zhang, and Zheng Zhang. 2024. RAGChecker: A Fine-grained Framework for Diagnosing Retrieval Augmented Generation. arXiv:2408.08067 https://arxiv.org/abs/2408.08067

[14] Alireza Salemi, Sheshera Mysore, Michael Bendersky, and Hamed Zamani. 2024. LaMP: When Large Language Models Meet Personalization. In Proceedings ofthe 62nd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, Bangkok, Thailand, 7370–7392. doi:10.18653/v1/2024.acl-long.399

[15] Yiting Shen, Kun Li, Wei Zhou, and Songlin Hu. 2026. Mem2ActBench: A Benchmark for Evaluating Long-Term Memory Utilization in Task-Oriented Autonomous Agents. In Proceedings ofthe 64th Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, San Diego, California, United States, 8173–8190. doi:10.18653/v1/2026.acl-long.370

[16] Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, and Dong Yu. 2025. LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory. In International Conference on Learning Representations. https: //openreview.net/forum?id=pZiyCaVuti

[17] Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang. 2025. A-Mem: Agentic Memory for LLM Agents. In Advances in Neural Information Processing Systems, Vol. 38. doi:10.52202/085713-0593

[18] Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren Zhou. 2025. Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models. arXiv:2506.05176 https://arxiv.org/abs/2506.05176

[19] Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and Yanlin Wang. 2024. Mem oryBank: Enhancing Large Language Models with Long-Term Memory. In Proceedings ofthe AAAI Conference on Artificial Intelligence, Vol. 38. 19724–19731. doi:10.1609/aaai.v38i17.29946

Table 7: Sources of the frozen knowledge base. Web additions include selected supporting material and related candidate documents.
<table><tr><td>Source</td><td>Documents</td></tr><tr><td>MedlinePlus</td><td>4,849</td></tr><tr><td>openFDA drug/label</td><td>2,803</td></tr><tr><td>National Cancer Institute (NCI)</td><td>2,371</td></tr><tr><td>NICHD Health Topics</td><td>818</td></tr><tr><td>MyHealthfinder</td><td>103</td></tr><tr><td>Open RN nursing textbooks</td><td>90</td></tr><tr><td>A Guide to Physical Activity</td><td>24</td></tr><tr><td>CDC health information</td><td>11</td></tr><tr><td>Base corpus</td><td>11,069</td></tr><tr><td>Web additions</td><td>1,964</td></tr><tr><td>Total corpus</td><td>13,033</td></tr></table>

Table 8: Distribution of annotation counts per task. Each cell gives the number of tasks, and each row sums to 349.
<table><tr><td>Count per task</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td></tr><tr><td>Facts</td><td>242</td><td>90</td><td>15</td><td>2</td><td>0</td><td>0</td></tr><tr><td>Cited turn pairs</td><td>22</td><td>222</td><td>79</td><td>22</td><td>2</td><td>2</td></tr><tr><td>Requirements</td><td>3</td><td>110</td><td>171</td><td>47</td><td>17</td><td>1</td></tr></table>

## A Dataset and Construction Protocol

## A.1 Corpus Sources

Table 7 summarizes the sources of the frozen knowledge base. We extract text from public APIs, XML files, web pages, and open textbooks, removing formatting markup and navigation content while preserving titles, body text, and source URLs. Documents are organized by entry, chapter, or drug-label field. Web additions are deduplicated by body text.

## A.2 Task Annotations

Table 8 describes the annotations of the 349 tasks. Single-Fact and multi-Fact tasks cite one Fact and at least two Facts, respectively. A Fact may contain several related personal conditions. Historical citations are counted as user–assistant interaction pairs, and cross-session statistics follow the definition in Section 3.7. The 1,015 answer requirements are the criteria used to evaluate Accuracy; their counts are also used in the task-composition analysis in Section 6.

## A.3 History and Evidence Construction

Public user statements establish personal conditions, and the current request follows all public sessions. The interaction histories retain tool-call arguments and document navigation information; the full text of documents used in history is available in the frozen knowledge base. Ofline memory construction uses the public interaction history and frozen knowledge base. During online execution, the agent receives the current request and obtains relevant information through the states, memories, and tool outputs supplied by each method. Private Facts, reference answers, answer requirements, and reference-document annotations are withheld from the methods.

Completed Topics are sampled without replacement using a fixed random seed, with one initial Item construction attempt per Topic. Tasks constructed from the frozen corpus use public interaction history and knowledge-base search and read tools. For web-supported tasks, the Builder uses the current Topic and the source interactions for its relevant Facts to search web pages and obtain supporting text. New supporting documents exclude documents already exposed in history and exact duplicate sources. Before reading a document, the Builder records the search event that revealed it. Document-identity checks and semantic validation together determine whether the required knowledge was already available in history.

Construction Models and Refinement. All construction roles, including multi-condition enhancement, use GLM-5.3 Flash through Zhipu Coding Plan at temperature 0. The Profile builder uses a percall output limit of 16,384 tokens. Question writing allows 8,192 tokens; the history-removal solver and final validator allow 24,576, and the requirement comparator allows 16,384. Separate prompts define each role’s task and evidence access, as shown in Figure 1.

## A.4 Acceptance and Evidence Mapping

The history-removal test gives a blind solver the current question and new supporting text. A comparator checks its answer against the reference answer and necessary requirements. A task is labeled answerable without history when all requirements are satisfied, and history-dependent when a requirement is unmet because the historical context is missing. Remaining cases are labeled as having insuficient evidence. Final validation checks whether the task naturally extends the interaction, whether personal conditions are grounded in their sources, whether historical knowledge is insufficient, whether new evidence supports the answer, and whether the answer adds information and satisfies safety requirements. It retains the history-dependence label from the preceding test.

We represent each answer requirement as $\boldsymbol { e } _ { j } = ( r _ { j } , U _ { j } , F _ { j } , K _ { j } , a _ { j } )$ linking the requirement $r _ { j }$ to verbatim user statements $U _ { j } ,$ personal conditions $F _ { j } { \mathrm { : } }$ , knowledge rules $K _ { j } ,$ , and the required judgment or action $a _ { j } .$ Personal evidence traces back to user messages, and knowledge evidence traces back to documents. Unknown premises are handled through conditional branches. Programmatic checks validate identifiers and reference integrity, while a language model assesses semantic support against the source text.

Item Refinement and Enhancement. Refinement jointly updates the question, reference answer, requirements, and evidence mapping while preserving public history and personal conditions. Rewording retains the supporting document. Multi-condition enhancement combines at least two disclosed conditions around the initial task, identifying the judgment afected by each, and may replace the document with a retrieved candidate from the same Topic. All drafts undergo the history-removal test and final validation of source support and answerability; enhancement additionally requires dependence on multiple conditions. Failed drafts are excluded, and revisions are re-evaluated under the same criteria.

## B Implementation and Experimental Settings

## B.1 Memory and Retrieval Implementations

Mem0 uses the non-graph memory path in the oficial SDK 2.0.20, with default extraction and update operations and local Qdrant storage. Mem0+Hybrid and MemKB share the resulting memories. For MemKB, an unresolved historical Read reference contributes no historical document link; memory records, valid historical links, and ofline-neighbor links are retained.

Graphiti uses SDK 0.31 with Neo4j, stores each complete interaction pair as an episode, and retrieves relational facts with its built-in RRF. A-MEM uses the authors’ implementation of note analysis, neighbor linking, memory evolution, and retrieval expansion. Each A-MEM memory response is capped at 5,120 tokens, including the serialized return structure. Notes are deduplicated and packed whole in their original retrieval order. All memory stores are built from the same 599 public interaction pairs, retaining message roles and sources.

The memory baselines expose separate memory\_search and kb\_search tools; MemKB uses memkb\_search. All methods share Read and the knowledge index, with ColaSearch’s term-based keyword retrieval.

## B.2 Model and Retrieval Configuration

Table 9 lists model and retrieval settings, including the Qwen3 encoder [18]. Ofline discovery retrieves 20 semantic candidates per card and excludes previously exposed documents. Each card lists retrieved neighbors followed by historical Read documents, preserving their respective orders; round-robin expansion deduplicates across cards.

All methods share the answering instructions for preserving actors, time, and unknown conditions and addressing each requested decision.

## B.3 Component Ablations

The component ablations cover all 111 retained tasks (40, 36, and 35) from three Profiles. The full MemKB configuration reuses answers and judgments from the main comparison. The ablated configurations use the same models, evaluation criteria, and interaction budgets.

The state-summary ablation removes the state lists supplied before the first query and in search results, retaining query-relevant memory. The document-link ablation clears each card’s linked document set while retaining memory vectors, the state summary, returned memory text, and the source-attribution constraint. Table 6 reports the results.

## C Supplementary Analysis and Cases

## C.1 Paired Estimation

The main comparison preserves the pairing between methods on all 349 tasks. We use a cluster bootstrap over the ten Profiles with 20,000 replicates and random seed 20261014. MemKB’s Accuracy gain over Mem0+Hybrid is 6.30 percentage points, with a 95% confidence interval of [3.19, 9.09].

Table 9: Configuration for the main comparison. Output limits apply per call, and the agent budget applies per task.
<table><tr><td>Component</td><td>Configuration</td></tr><tr><td>Answer model</td><td>DeepSeek v4.1 Flash; temperature 0; output limit 16,384 tokens; Agent budget 128,000 tokens. Mem0 and MemKB use OpenCode; No Memory, Graphiti, and A-MEM use SiliconFlow with OpenCode fallback.</td></tr><tr><td>Online interaction</td><td>Up to four tool rounds and one final-answer round; early completion allowed. A round is one model response and may contain multiple tool calls. Read: at most 100 lines and 3,000 tokens per call.</td></tr><tr><td>Accuracy evaluator</td><td>GLM-5.3 Flash; low reasoning effort; temperature 0; output limit 32,768 tokens. SiliconFlow and OpenCode, with identical model inputs and parameters.</td></tr><tr><td>Memory construction</td><td>Mem0 SDK extraction and update with GLM-5.3 Flash; Mem0 and MemKB share the resulting memories.</td></tr><tr><td>Embeddings</td><td>Qwen3-Embedding-0.6B for knowledge retrieval, Mem0, MemKB and Graphiti; all-MiniLM-L6-v2 for A-MEM memory.</td></tr><tr><td>MemKB state bank</td><td>GLM-5.3 Flash; output limit 16,384 tokens (32,768 for one Profile). Up to 30 states with subjects, time, unknown conditions and source-message IDs; provided before the first query and in search results</td></tr><tr><td>Memory retrieval</td><td>Mem0 returns 20 results by default, with a configurable count. Graphiti and A-MEM use 5 results or root notes by default, up to 20; A-MEM expands linked notes within its 5,120-token return budget.</td></tr><tr><td>MemKB candidates</td><td>Top 20 memory cards; round-robin expansion to at most 20 unique linked documents. Semantic and keyword channels each</td></tr><tr><td>Fusion and output</td><td>return at most 20 candidates. Equal-weight RRF with κ = 60; 5 documents by default, at most 20. Up to 20 deduplicated memories with source-card and message IDs.</td></tr></table>

## C.2 Task-Composition Sensitivity

Figure 3(d) stratifies tasks by Profile and the number of answer requirements, then adds construction cohort in a further adjustment. Each analysis retains strata containing both single-Fact and multi-Fact tasks. We calculate MemKB’s Accuracy gain over Mem0+Hybrid for each group within each stratum, then aggregate the gains using stratum task counts as weights. Both groups receive the same stratum weights. The two common-support sets contain 316 and 275 tasks, respectively. Construction cohort records the initial Item admission path: direct acceptance (155 tasks) or acceptance after revision and revalidation (194 tasks), as identified by the construction metadata.

Resampling preserves task-level pairings and recomputes stratum weights in each replicate, using the bootstrap settings above. The unadjusted gain diference between multi-Fact and single-Fact tasks is 4.39 percentage points, with a 95% confidence interval of [−5.52, 15.20]. Adjusting for Profile and requirement count yields a diference of 9.54 points, with an interval of [−2.21, 21.75]; adding construction cohort yields 6.39 points, with an interval of [−4.29, 18.67]. The figure’s error bars show confidence intervals for each group’s paired gain. Intervals for the between-group diference are calculated by subtracting the gains within each bootstrap replicate. These exploratory intervals are unadjusted for multiple comparisons.

## C.3 Shared Background across Tasks

Grouping tasks by Profile and the exact set ofselected Facts yields 98 groups across the 349 tasks. Of these, 60 groups involve at least two distinct reference documents and cover 311 tasks (89.1%). Table 10 presents three tasks that share the same personal background. Each question calls for diferent rules, whose application depends on the relative’s condition and the caregiver’s distance recorded in the history.

## C.4 Case Evidence

The HFC eligibility example in Table 1 draws on the user’s public statements about their caregiving relationship and living alone, together with the frozen Care Grants page. Caregiving role and co-residence each determine the applicability of the relevant rules.

Treatment History. For task item\_0a49978ebac6, user message tu\_0f0b8cae0621\_user states, “I’m a survivor ofchildhood cancer, and I still go for long-term follow-up care as an adult to keep an eye on how my treatment might be afecting me.” Both methods read the NICHD document kb\_4cd3cbc8ac; its opening 40 lines include the chemotherapy guidance and the rule for women “undergoing radiation therapy”. Mem0+Hybrid answers, “no record of HIV, HTLV, tuberculosis, cancer treatment, or radiation”. MemKB instead keeps treatment details conditional: “I don’t have your actual treatment details”.

Medication Class. For task item\_21c9bf907c11, message tu\_ 5415ad4050d9\_user states, “I take prescription pain medicine that my doctor prescribed.” Both methods read the CVS document kb\_ web\_73f82c20a758ffd8 in full (Read ofsets 0–18), including the condition “at increased risk for opioid overdose” and access “without an individual prescription”. MemKB writes, “You do take a prescribed opioid pain medicine”; Mem0+Hybrid writes, “You’re currently taking a prescription opioid pain medicine”.

Error-Analysis Evidence. In task item\_a6b3ffbe02ec, user message tu\_5add4071b6e7\_user states, “I take prescribed pain medicine for my cancer-related pain”. State state\_006 adds “(opioid)”, and the answer consequently treats opioidrelated naloxone guidance as applicable. The FDA document kb\_web\_cf4e473771cdc3d7 supplies the opioid-use condition; the historical statement leaves the medication class unspecified.

In the scalp-cooling task item\_0b65cfc2b717, document kb\_ web\_838a73aeafde37e1 states that cooling is not recommended for “migraine headaches from cold temperatures”. The returned snippet mentions cancer-related exclusions but omits this rule; the agent instead reads kb\_web\_a5e06a9cd52e5f04. In the neuropathy task item\_b58978a86ea7, the agent reads kb\_web\_f0f60132da209c71, whose heat-risk list includes “warm foot baths”, yet recommends a warm foot bath with temperature checks.

Table 10: Three tasks sharing the same personal background: the user cares for a relative with Alzheimer’s disease from a distance. Questions and answer points are paraphrased.
<table><tr><td>Current question</td><td>Knowledge to retrieve</td><td>History-informed judgment</td></tr><tr><td>Would my relative qualify for and be suited to a dementia-specific adult day program?</td><td>Target population and enrollment require- ments.</td><td>Alzheimer&#x27;s disease matches the target population; formal diagnostic documentation required for enrollment remains to be confirmed.</td></tr><tr><td>Does respite care apply, at home or during a short inpatient stay?</td><td>Hospice eligibility and respite-care ar- rangements.</td><td>The user is a family caregiver. If hospice enrollment is established, explain home and short-term inpatient options</td></tr><tr><td>How should my relative prepare for a winter storm or power outage?</td><td>Dementia-specific supplies, communica- tion and local assistance.</td><td>Adapt preparations to cognitive impairment and long-distance caregiv- ing; identify local helpers and agree on responsibilities.</td></tr></table>

## D Evaluation Protocol

## D.1 Metrics and Scoring Inputs

Accuracy follows Section 5.2: one call checks personal facts, knowledge claims, and task requirements, returning pass, fail, or needs\_adjudication. All main-comparison and ablation judgments resolve to pass or fail; Accuracy divides passes by all tasks in the comparison. The 48 MemKB errors in Section 6.6 are failed answers.

Read Recall follows Section 5.2, using unique reference documents with nonempty Read returns; search snippets, failed reads, and empty reads do not count. First-search and overall search hit rates measure reference-document presence in the first or any successful knowledge-search response, respectively. Their denominator includes all cohort tasks, counting tasks without successful searches as misses. Memory-only searches, failed calls, and internal candidates are excluded.

All methods share the task’s complete public history, reference answer, requirements, and fixed historical and new-support documents as evaluation references. The 1,015 requirements specify applicability conditions and unknown premises. Candidate inputs add the answer and actual Search snippets and Read text. Agent-visible evidence includes supplied states, memories, and Search/Read returns; internal candidates and evaluator-only material are excluded. Full construction, memory, and evaluation prompts and input templates will be provided on GitHub. Model parameters appear in Appendix B.2.

## D.2 Answer-Quality Dimensions

We evaluate the 222 Mem0+Hybrid and MemKB answers on the 111-task ablation cohort. State Fidelity checks whether represented personal conditions preserve the historical speaker, actor, timing, and uncertainty. Condition Coverage measures whether the personal premises needed for the current task are substantively reflected. Evidence Grounding assesses support for knowledge claims in the supplied documents and actual retrieved passages. Rule Application evaluates whether prerequisites, restrictions, exceptions, and timing are correctly applied to the user’s situation. Answer Completeness measures coverage of requested subquestions and necessary actions. Each call returns a score with candidate excerpts, source evidence, and a rationale.

## D.3 Error Categories and Evidence Visibility

The primary category identifies each failed answer’s decisive error; its Accuracy verdict is unchanged. Personal Assumptions covers treating unknown states, plans, advice, or another person’s information as established personal facts. Context Underuse covers failing to apply established user conditions to the current judgment. Rule Misapplication covers incorrect knowledge claims, rule interpretations, or applicability judgments. Answer Omissions covers unanswered necessary subquestions, actions, restrictions, or steps when no more direct personal-context error explains the omission. Other Issues covers errors outside these categories or unresolved attribution and source conflicts. Up to two secondary categories record other independently supported errors; Figure 4 reports primary categories.

Visibility concerns the personal premises and rules needed to resolve the errors, assessed against the agent’s question, states, memories, and Search and Read returns. Visible indicates that all required evidence was available; Partly visible, that some evidence was available but key content was missing; and Not visible, that the key evidence was absent. Evaluator-only material and unreturned candidates are excluded; for unsupported personal assertions, visibility concerns the correct condition or uncertainty. Uncertain is used when the records or source conflicts prevent a reliable judgment; no answer received this label.