# Event-Centric Memory with Query-Aware Graph Augmentation for Long-Term Conversational Agents

Yichen Liu, Chunfeng Yuan, Haowei Liu, Wenjuan Li, Zefeng Lin, Bing Li, Xu Chen, and Weiming Hu

Abstract—For persistent and personalized conversational agents, memory systems can enable them to remember, update, and reason over long histories by storing past interactions and retrieving relevant information. Existing memory systems typically follow two paradigms: flat-structured memory and graph-based memory. The former is lightweight but leaves event relations and state updates implicit, while the latter explicitly models memory structure but incurs additional construction cost and introduces irrelevant relations over long histories. To address these limitations, we propose QGMEM, a novel memory construction and activation framework motivated by human memory, in which experience is organized into events and query-relevant events are modeled by graph as working memory. QGMEM converts long dialogue histories into event-indexed atomic memory units that preserve individual experiences and consolidates related units into dynamic memory traces that retain state trajectories and current states. When a query arrives, hybrid memory retrieval gathers complementary candidate memories, and query-aware reranking activates the most relevant units as a compact working memory. To expose relational dependencies in the working memory and support conflict-aware reasoning, QGMEM organizes the working memory as a local graph, which is then encoded as a graph token and provided to the LLM together with the textual working memory to improve evidence utilization during answer generation. Experiments across six benchmarks validate the framework and show consistent gains in retrieval, multi-hop evidence composition, conflict resolution, and ultra-long dialogue reasoning with compact contexts and moderate inference cost.

Index Terms—Long-term memory, conversational agents, graphbased memory.

## I. INTRODUCTION

Maintaining continuity across long-term interaction between humans and agents remains a pivotal milestone towards personalized and autonomous artificial intelligence. Personal assistants, companion dialogue systems, and task-oriented agents are expected to remember user preferences, interpersonal relations, schedules, and later updates across many turns or sessions. Since such information cannot always fit into the model context [1], an agent needs memory mechanisms that preserve past interactions, track evolving states, identify relevant memory, and organize their relations for answer generation. Therefore, recent works [2]–[4] introduce long-term memory systems that store historical interactions as persistent external memory and retrieve relevant memory units when answering user queries.

Existing memory systems mainly follow two organizational paradigms, as illustrated in Figure 1(a) and (b). Flat-structured memory stores historical information as turns, sessions or summaries [3]–[5]. This paradigm is lightweight and easy to maintain, but relations among events and state changes are usually left implicit. Graph-based memory explicitly represents entities, events, and states as graph structures [6]–[8]. Such structures support relational reasoning, but constructing and retrieving graph structures over long histories introduces additional cost and even irrelevant relations. Consequently, existing long-term conversational memory systems struggle to simultaneously achieve lightweight persistent storage and explicit relational organization.

These limitations motivate a memory architecture that combines persistent storage with query-time relational organization: long-term memory preserves events and their evolving states, whereas working memory constructs a relational structure over query-relevant events. This view is consistent with classic research in cognitive psychology on event cognition, which suggests that people represent continuous experience as a sequence of events, update the current event in working memory, and retrieve relevant past events from long-term memory [9]– [11]. Motivated by this perspective, we propose Query-aware Graph-augmented Memory (QGMEM), which combines event centric memory storage with query-time relational organization, as illustrated in Figure 1(c). QGMEM first converts dialogue histories into event-indexed atomic memory units with traceable source content, each representing an independently interpretable event or state and indexed by explicit event attributes, including entities, locations, and factual time. Related units are then consolidated into dynamic memory traces that preserve state trajectories and current states, thereby making state updates explicit and distinguishing current facts from superseded ones. To improve the retrieval quality of answer-bearing memories, hybrid memory retrieval integrates lexical, semantic, and eventattribute relevance using reciprocal rank fusion (RRF) [12]. Query-aware reranking removes irrelevant and superseded candidates, yielding a compact working memory concentrated on the current query. To model relational dependencies within the activated working memory, QGMEM constructs a local memory graph from the facts of the selected memory units. Then, a relation-aware graph encoder is employed to encode the local memory graph, and the resulting graph-level representation is projected into the LLM embedding space. The graph token is supplied to the LLM together with the textual working memory, enabling multi-hop evidence composition based on both memory content and relational information.

![](images/35a22dccd7fe31096b7df5743bad2edcdefee0c023d131146bf1e8b49e84589d.jpg)  
Fig. 1. Comparison of flat-structured memory, graph-based memory, and QGMEM.

The resulting framework is designed to reconcile three objectives in long-term conversational memory: preserving verifiable historical evidence, exposing query-relevant relations, and maintaining a compact generation context. Figure 2 provides an empirical view of this balance by comparing QGMEM with retrieval-augmented generation (RAG) and representative flat-structured and graph-based memory systems in terms of retrieval quality, answer quality, and context efficiency. Flat-structured systems maintain compact contexts but exhibit lower retrieval or answer quality, whereas graphbased systems do not simultaneously attain high retrieval and answer quality. In contrast, QGMEM achieves an nDCG@10 of 82.50% and an LLM-judged score of 82.77% with 598.19 retrieved-context tokens. These results show that QGMEM improves the prioritization and utilization of answer-supporting memories while maintaining a compact context for generation.

We conduct main experiments on six long-term memory benchmarks. LoCoMo [13] and LongMemEval-S [14] evaluate accurate retrieval and multi-hop evidence composition, while Fact-SH and Fact-MH from MemoryAgentBench [15] evaluate conflict resolution. BEAM-100K and BEAM-500K [16] assess scalability over ultra-long dialogue histories. We further extend the BEAM evaluation to 1M and 10M tokens to examine performance under extreme context lengths. Overall, QGMEM consistently outperforms state-of-the-art systems. Further eval uations show that these gains remain consistent across two open-weight backbones, Qwen3-8B and Llama3.1-8B-Instruct, and one closed-source backbone, GPT-4o-mini [17]–[19]. Indepth ablation experiments confirm the effectiveness of the key components in QGMEM. Additional efficiency, scalability, hyperparameter sensitivity, and case analyses characterize the computational cost, robustness, and retrieval performance of QGMEM from complementary perspectives.

![](images/3a1124557ea3968c71f6c2fc97bfe215538ed5761b9a551a331fb8d0edc348f6.jpg)  
Fig. 2. Comparison of retrieval quality, LLM-judged answer score, and retrieved-context size on LoCoMo with Qwen3-8B. Colors distinguish the retrieval-augmented baseline, flat-structured memory, graph-based memory, and QGMEM. The horizontal axis denotes normalized discounted cumulative gain at rank 10 (nDCG@10), the vertical axis denotes the LLM-judged score, and bubble area denotes the average number of retrieved-context tokens. Higher axis values and smaller bubble areas indicate better overall performance.

The main contributions of this work are summarized as follows:

• We introduce an event-centric memory framework for persistent conversations that integrates fine-grained event storage, evolving-state consolidation, query-aware activation, and relational evidence organization within a unified memory lifecycle.

• We represent conversational experience using traceable atomic memory units and dynamic memory traces, preserving both historical states and currently valid information as the dialogue evolves.

• We organize query-relevant memories into a compact working memory and encode their local relational topology jointly with source-grounded textual evidence, supporting evidence composition without maintaining a graph over the complete dialogue history.

• We evaluate the framework across six benchmarks and multiple LLM backbones, examining answer quality, retrieval effectiveness, conflict-sensitive reasoning, longcontext scalability, and computational cost.

## II. RELATED WORK

## A. Persistent Conversational Memory Representation

A central design choice in conversational memory is how interaction histories are represented and organized for persistent storage [2], [8], [20], [21]. Flat-structured memory retains dialogue turns or transforms them into chunks, summaries, and facts. MemGPT manages prior interactions through external memory [4], while LongMem maintains a retrievable long-term memory cache [22]. MemoryBank and FACT organize summarized or extracted information [3], [5], whereas Mem0 and LightMem consolidate salient records to improve compactness and access efficiency [23], [24]. These representations are easy to maintain and can be supplied directly to an LLM, but event relations and state changes remain implicit.

Graph-based conversational memory, by contrast, material izes explicit associations at different granularities. A-MEM organizes memory notes into an evolving network through dynamically generated links and updates [25]. Associa constructs an event-centric memory graph and combines subgraph extraction with iterative deliberative recall [26]. Zep maintains a temporal knowledge graph that records the validity and history of relations [27]; SG-Mem connects sentences within chunked dialogue units [6]; and EMem-G organizes sessions, eventlike discourse units, and their arguments in a heterogeneous graph [7]. These approaches differ in graph granularity and edge semantics, while sharing the use of an explicit graph as a persistent memory index.

Recent systems further enrich persistent graphs with hierarchical, temporal, and provenance-aware organization. Li-CoMemory and GAM connect detailed events with higherlevel semantic structures [28], [29], while HyperMem uses hyperedges to capture higher-order associations among topics, episodes, and facts [30]. APEX-MEM adopts an append-only temporal property graph to preserve evolving facts [31], and MemORAI couples a provenance-enriched multi-relational graph with query-adaptive edge weighting [32]. Complementary structured-memory approaches such as StructMem preserve event-level bindings and periodically consolidate cross-event connections without imposing a persistent entity–relation graph [33]. A controlled analysis of long-term dialogue memory systems further shows that graph effectiveness depends jointly on memory-unit design, graph maintenance, seed activation, and graph expansion, making the surrounding retrieval pipeline important for interpreting gains from graph structure [34].

Persistent graphs expose relational structure throughout the memory lifecycle, but relation extraction, alignment, and maintenance operate over an expanding history. QGMEM places graph materialization after query-aware activation. Its persistent substrate consists of source-grounded atomic units and lightweight dynamic traces; a relational graph is instantiated only over the activated working memory. Dynamic traces preserve longitudinal state evolution, while the query-local graph organizes dependencies among the selected facts. This separation limits the scope of graph construction without discarding historical evidence or its provenance.

## B. Query-Time Memory Access

Effective use of persistent conversational memory depends on retrieving information that is relevant and sufficient for the current query without introducing extraneous context. Flatstructured memory systems generally rank independent records by lexical or semantic relevance. Lexical retrieval, commonly implemented with BM25 [35], preserves exact names, dates, and other discriminative expressions, but misses records expressed in substantially different terms. Dense retrieval captures semantic correspondence beyond exact terms [36]–[38] and is used by systems such as LongMem and Mem0 [22], [23]. However, descriptions of similar events may receive high semantic relevance despite referring to different entities, times, or states.

Retrieval over a persistent graph commonly begins with semantically matched seed nodes and then expands through stored edges [34]. Associa extracts evidence-rich subgraphs through prize-collecting Steiner tree optimization [26]; Ariadne-Mem discovers bridge paths between distributed evidence [39]; HyperMem performs coarse-to-fine retrieval over its topic, episode, and fact hierarchy [30]; and GAM follows cross-level links from relevant topics to archived event graphs [29]. Mem-ORAI applies query-conditioned edge weighting during graph propagation [32], while APEX-MEM uses a retrieval agent to resolve conflicting and evolving facts [31]. These strategies exploit stored topology to broaden evidence coverage, while their retrieval behavior remains coupled to the construction and maintenance of the persistent graph.

To address these limitations, QGMEM applies hybrid retrieval over event-indexed atomic memory units to improve the coverage of answer-supporting information while reducing the retrieval of events whose attributes do not match the query. Query-aware reranking then filters irrelevant and superseded candidates to form a compact working memory. A local graph constructed over this activated memory preserves explicit relational organization for evidence composition while avoiding the maintenance cost and query-irrelevant relations associated with a persistent global graph.

## C. Memory-Augmented Language Generation

Beyond selecting relevant records, memory-augmented generation determines how retrieved information is incorporated into the LLM. Most systems provide retrieved memories to the LLM as text: flat records are appended directly to the prompt, while graph-based memory is serialized as triples, paths, subgraphs, or summaries [27], [39]–[41]. Although broadly compatible with language models, serialization linearizes graph topology, requiring the LLM to reconstruct structural dependencies from tokens and increasing context consumption.

Recent studies on integrating graph structures with LLMs have shown that learned graph representations can preserve topology and provide relational signals directly during generation, without requiring the LLM to reconstruct these dependencies solely from serialized text [42], [43]. G-Retriever combines an encoded subgraph with its textual representation for graph question answering [44], while RAS and Relink construct query-specific evidence graphs for knowledge-intensive generation and GraphRAG [45], [46]. QGMEM builds on the general principle of query-conditioned graph augmentation but addresses persistent conversational memory, where facts may be introduced, revised, supplemented, or superseded over time. It therefore couples query-local relational organization with event-indexed atomic storage and dynamic memory traces, allowing generation to use selected evidence together with its retained state history.

![](images/28e1d551bc3cc05a4cf3123b7b70d0498d868b60d4a6bb1c576c8dc874633046.jpg)  
Fig. 3. Overall framework of QGMEM.

## III. METHOD

The overall framework of QGMEM is illustrated in Figure 3 and comprises three modules. First, event-centric memory construction converts the dialogue history into event-indexed atomic memory units and consolidates related units into dynamic memory traces, preserving verifiable evidence and evolving states. Second, query-aware working memory activation retrieves and reranks memories for the query q, forming a compact working memory that concentrates complete, current, and answer-relevant information. Finally, graph-augmented generation organizes the activated memories into a local graph and jointly provides its graph token and textual working memory to the LLM, enabling answer generation from both memory content and structural relations. In the following, we will introduce these three modules in detail, and analyze the generalization and efficiency of our proposed framework.

## A. Event-Centric Memory Construction

QGMEM represents a dialogue history as an event-indexed atomic memory set M and a dynamic trace set R. Each atomic unit represents a fine-grained event or state and retains its source dialogue, structured fact, and event and temporal attributes for traceable retrieval. Dynamic traces connect related units into state trajectories that distinguish current information from superseded records. Together, they form the long-term memory store $\boldsymbol { \mathcal { S } } = ( \mathcal { M } , \mathcal { R } )$ , supporting precise memory access and conflict-aware answering over evolving histories.

1) Atomic Memory Unit Construction: To construct atomic memories, QGMEM first scans each dialogue session with $N _ { W }$ overlapping sliding windows $\{ W _ { i } \} _ { i = 1 } ^ { N _ { W } }$ . Each window $W _ { i }$ provides a local dialogue context that contains newly processed turns together with limited preceding context for resolving pronouns and local references. Given $W _ { i } .$ , QGMEM decomposes the dialogue into independently retrievable memory units with traceable source content, each representing a single interpretable event or state. When one extracted candidate contains multiple independently retrievable relations, they are materialized as separate atomic units so that each finalized unit is anchored by one complete structured fact. Let $K _ { i }$ be the number of units extracted from $W _ { i } ,$ , and let $m _ { i , j }$ denote its j-th unit. The complete memory set is $\mathcal { M } = \{ m _ { i , j } \mid 1 \leq i \leq$ $N _ { W } , 1 \leq j \leq K _ { i } \}$ . For a generic unit $m \in \mathcal { M } , c _ { m }$ denotes its context-independent summary, $d _ { m }$ the corresponding source dialogue span, ${ \mathcal { E } } _ { m }$ and $\mathcal { P } _ { m }$ its entity and location sets, and $f _ { m }$ its structured fact. The factual and utterance times of the unit are denoted by $T _ { m } ^ { f }$ and $T _ { m } ^ { u }$ . The unit is represented as

$$
\begin{array} { r } { m = ( c _ { m } , d _ { m } , \mathcal { E } _ { m } , \mathcal { P } _ { m } , T _ { m } ^ { f } , T _ { m } ^ { u } , f _ { m } ) . } \end{array}\tag{1}
$$

Together, these fields preserve source traceability and expose explicit event and temporal attributes for trace consolidation and retrieval.

This representation is designed to make each memory unit independently retrievable while preserving the event and temporal context required for subsequent reasoning. It has two advantages: (1) Multi-perspective event matching. Explicit entity, location, and factual-time attributes characterize an event or state from complementary perspectives, helping distinguish answer-bearing memories from topically related events and improving their retrieval priority. (2) Dual-time representation. Separating factual time $T _ { m } ^ { f }$ from utterance time $T _ { m } ^ { u }$ makes temporal information directly comparable during retrieval. For example, if a user said on January 20, “I went to the hospital yesterday,” then $T _ { m } ^ { u }$ is January 20, while $T _ { m } ^ { f }$ is normalized to January 19. Resolving this distinction during memory construction supports temporal grounding and selection of the valid memory under time-sensitive or conflicting records.

2) Dynamic Memory Trace Consolidation: Atomic memory units preserve fine-grained evidence, but long-term conversations often contain state changes that cannot be captured by isolated units alone. For example, a user may first set a goal, later revise it, and finally ask about the previous goal. To make such update trajectories explicit, QGMEM consolidates related memory units into dynamic memory traces. Write the structured fact of unit m as $f _ { m } = \left( a _ { m } , p _ { m } , o _ { m } \right)$ , where $a _ { m } .$ $p _ { m }$ , and $o _ { m }$ are its subject, predicate, and object, respectively. With the normalized text encoder $E ( \cdot )$ , we first compute three pairwise similarities and then combine them into a relation score:

$$
\begin{array} { c } { { s _ { f } ^ { i j } = \cos \bigl ( E ( f _ { m _ { i } } ) , E ( f _ { m _ { j } } ) \bigr ) , } } \\ { { s _ { a } ^ { i j } = \cos \bigl ( E ( a _ { m _ { i } } ) , E ( a _ { m _ { j } } ) \bigr ) , } } \\ { { s _ { p } ^ { i j } = \cos \bigl ( E ( p _ { m _ { i } } ) , E ( p _ { m _ { j } } ) \bigr ) , } } \\ { { S _ { \mathrm { r e l } } ( m _ { i } , m _ { j } ) = \mathrm { m i n } \Bigl \{ s _ { f } ^ { i j } , s _ { a } ^ { i j } , s _ { p } ^ { i j } \Bigr \} . } } \end{array}\tag{2}
$$

Here, $s _ { a } ^ { i j }$ and $s _ { p } ^ { i j }$ ensure that two facts describe the same state holder and relation, while $s _ { f } ^ { i j }$ rejects otherwise unrelated assertions without requiring identical objects. We take their minimum because a false merge can corrupt an entire trajectory, whereas unmerged units remain independently retrievable. For each unit, we test its 20 nearest fact-level neighbors. A set of memory units forms a relation cluster only if the relation score between every pair of their structured facts is no smaller than $\tau _ { r } .$ . We set $\tau _ { r } = 0 . 8 0$ based on the sensitivity analysis in Figure 7.

For the h-th cluster, $\psi _ { h }$ denotes its representative subject– predicate relation. An LLM-based consolidator orders the facts chronologically, merges repeated or semantically equivalent states, and retains distinct state changes. It treats a later incompatible value as an update, a compatible refinement as a supplement, and mutually compatible values as coexisting states. The resulting trace is $r _ { h } = ( \psi _ { h } , H _ { h } , s _ { h } )$ , where $H _ { h }$ is the ordered state trajectory and $s _ { h }$ is its latest supported state:

$$
\boldsymbol { r } _ { h } = ( \psi _ { h } , H _ { h } , s _ { h } ) .\tag{3}
$$

If consolidation produces $N _ { R }$ traces, the resulting trace set is

$$
\mathcal { R } = \{ r _ { h } \} _ { h = 1 } ^ { N _ { R } } .\tag{4}
$$

At query time, each retrieved atomic unit remains linked to its dynamic trace. The trace supplies the state trajectory to the reranker and generator, while the latest member provides a current-state representative when several retrieved units describe different versions of the same relation. This separation preserves source-level granularity for verification and retrieval, while linking distant updates into a coherent trajectory for conflict resolution and multi-hop evidence composition.

## B. Query-Aware Working Memory Activation

QGMEM activates working memory by selecting a compact set of query-relevant memory units from the long-term memory store $\boldsymbol { \mathscr { S } } = ( \mathcal { M } , \mathcal { R } )$ . The procedure follows two successive steps: hybrid memory retrieval and query-aware reranking. Hybrid memory retrieval identifies complementary candidates through lexical, semantic, and event-attribute retrieval. Lexical retrieval preserves exact query cues, semantic retrieval identifies related expressions, and event-attribute retrieval distinguishes memories with similar content but different entities, locations, or temporal conditions. RRF [12] combines the three ranked lists according to rank positions, producing a unified candidate ordering without cross-channel score calibration. Query-aware reranking subsequently evaluates the fused candidates and selects the final working memory under a fixed budget, prioritizing memories that are relevant and consistent with the current state. Algorithm 1 summarizes the complete procedure.

1) Hybrid Memory Retrieval: Given a query q issued at time $t _ { q } ,$ QGMem retrieves candidate memory units $m \in \mathcal { M }$ through three complementary channels. Lexical retrieval, implemented with BM25 [35], identifies units containing exact query cues such as names, dates, and locations, whereas semantic retrieval identifies related units expressed through aliases or paraphrases using dense embeddings. For event-attribute retrieval, the parser $\phi ( q , t _ { q } )$ extracts the entity, location, and temporal constraint sets $\mathcal { C } _ { q } ^ { \mathrm { e n t } } , \mathcal { C } _ { q } ^ { \mathrm { l o c } }$ , and ${ \mathcal { C } } _ { q } ^ { \mathrm { t i m e } }$ . SemMatch evaluates the compatibility of entity and location constraints with ${ \mathcal { E } } _ { m }$ and $\mathcal { P } _ { m } .$ while TimeMatch evaluates the temporal constraint against $( T _ { m } ^ { f } , T _ { m } ^ { u } )$ . Writing $\mathcal { C } _ { q } ^ { x }$ for the corresponding constraint set, the available scores are averaged over nonempty constraints to obtain $S _ { \mathrm { E L T } }$

Each channel retains its n highest-scoring units and produces a ranked list $\mathcal { L } _ { b } , ~ \mathcal { L } _ { d } , ~ \mathrm { o r } ~ \mathcal { L } _ { e }$ . We use $\mathrm { T o p K } _ { x \in \mathcal { X } } ^ { b } s ( x )$ to denote the b highest-scoring elements of X under score s. RRF [12] assigns the fused score $S _ { \mathrm { R R F } } ( m , q )$ according to rank (m), the position of m in list ${ \mathcal { L } } .$ . Its rank-offset constant κ is fixed to 60. The n highest-scoring units after fusion form the reranking candidate pool $\mathcal { C } _ { 0 }$ . This fusion preserves complementary retrieval coverage without requiring score calibration, particularly when long histories contain many superficially similar memories.

2) Query-Aware Reranking: Candidates in $\mathcal { C } _ { 0 }$ still differ in content completeness and answer relevance because firststage scores consider each signal separately. QGMEM therefore constructs a reranking document for every candidate containing its context-independent fact summary, source dialogue, entities, locations, factual time, utterance time, and the state trajectory from its associated dynamic memory trace when available. Reranker $( q , m )$ , implemented with Qwen3-Reranker-8B, jointly scores the query and this structured document. Under the working-memory budget $k ,$ the highest-scoring candidates form the activated working memory $\mathcal { M } _ { q } .$ . Its ordering is used by both graph construction and answer generation. This two-stage activation preserves candidate-stage coverage while concentrating complete, current, and answer-relevant memories at the highest ranks. The bounded working context retains the evidence needed for generation with controlled input size, supporting retrieval quality, context efficiency, and stable access as the memory store grows.

## C. Graph-Augmented Generation

The graph token encodes relational topology, while temporal order and state validity remain in the dynamic traces and serialized working memory. These complementary inputs support source-grounded, relation-aware generation without constructing a graph over the full history.

1) Local Memory Graph Construction: Each structured fact $f _ { m } = ( u , \rho , v )$ in the activated working memory $\mathcal { M } _ { q }$ is represented by an entity node for the subject u, a value node for the object v, and a directed edge labeled by the predicate $\rho .$ The mappings $\nu _ { \mathrm { e n t } } ( \cdot )$ and $\nu _ { \mathrm { v a l } } ( \cdot )$ normalize entity and value labels, respectively, such that their repeated occurrences share the same node. The induced node and directed-edge sets are

$$
\begin{array} { r l } & { \mathcal { V } _ { q } = \bigcup _ { m \in \mathcal { M } _ { q } } \{ \nu _ { \mathrm { e n t } } ( u _ { m } ) , \nu _ { \mathrm { v a l } } ( v _ { m } ) \} , } \\ & { \mathcal { A } _ { q } = \left\{ ( \nu _ { \mathrm { e n t } } ( u _ { m } ) , \rho _ { m } , \nu _ { \mathrm { v a l } } ( v _ { m } ) ) \middle | \right. \left. \underset { f _ { m } = ( u _ { m } , \rho _ { m } , v _ { m } ) } { m \in \mathcal { M } _ { q } , } \right\} . } \end{array}\tag{5}
$$

The resulting local memory graph is $\mathcal { G } _ { q } = ( \mathcal { V } _ { q } , \mathcal { A } _ { q } )$ . It encodes the relational topology of the activated facts, supporting relationaware evidence composition. Time-aware retrieval and dynamic traces in the text branch preserve temporal order and state validity, complementing the graph topology during generation. Because $\mathcal { G } _ { q }$ is induced by the fixed-budget working memory $\mathcal { M } _ { q } ,$ its size remains bounded as the dialogue history grows.

2) Graph Encoding: QGMem initializes each node i and edge predicate $\rho _ { e }$ with the embeddings $\mathbf { x } _ { i }$ and $\mathbf { r } _ { e }$ produced by the fixed sentence encoder $f _ { \mathrm { e m b } }$ , which is shared with semantic retrieval.

To incorporate predicate information during message passing, the graph encoder applies L relation-aware TransformerConv layers [47] with H attention heads. For the incoming edge set $\mathcal { T } _ { q } ( i ) = \{ e = ( j , \rho _ { e } , i ) \in \mathcal { A } _ { q } \}$ , the node states are initialized by ${ \bf h } _ { i } ^ { ( 0 ) } = { \bf x } _ { i }$ . At layer ℓ and head $h ,$ , the relation-aware update is

$$
\begin{array} { r l } & { \eta _ { i e } ^ { \left( \ell , h \right) } = \frac { \left( \mathbf { W } _ { Q } ^ { \left( \ell , h \right) } \mathbf { h } _ { i } ^ { \left( \ell \right) } \right) ^ { \top } \left( \mathbf { W } _ { K } ^ { \left( \ell , h \right) } \mathbf { h } _ { j } ^ { \left( \ell \right) } + \mathbf { W } _ { E } ^ { \left( \ell , h \right) } \mathbf { r } _ { e } \right) } { \sqrt { d } h } , } \\ & { \boldsymbol { \alpha } _ { i e } ^ { \left( \ell , h \right) } = \frac { \exp \left( \eta _ { i e } ^ { \left( \ell , h \right) } \right) } { \sum _ { e ^ { \prime } \in \mathcal { Z } _ { q } \left( i \right) } \exp \left( \eta _ { i e ^ { \prime } } ^ { \left( \ell , h \right) } \right) } , } \\ & { \mathbf { m } _ { i } ^ { \left( \ell , h \right) } = \frac { \boldsymbol { \alpha } _ { i } ^ { \left( \ell , h \right) } } { { { e } = \left( j , { \rho } _ { e } , i \right) \in \mathcal { Z } _ { q } \left( i \right) } } \boldsymbol { \alpha } _ { i e } ^ { \left( \ell , h \right) } \left( \mathbf { W } _ { V } ^ { \left( \ell , h \right) } \mathbf { h } _ { j } ^ { \left( \ell \right) } + \mathbf { W } _ { E } ^ { \left( \ell , h \right) } \mathbf { r } _ { e } \right) , } \\ & { \mathbf { h } _ { i } ^ { \left( \ell + 1 \right) } = \Gamma _ { \ell } \left( \mathbf { W } _ { 0 } ^ { \left( \ell \right) } \mathbf { h } _ { i } ^ { \left( \ell \right) } + \mathbf { \Pi } _ { h = 1 } ^ { H } \mathbf { \Phi } _ { i } ^ { \left( \ell , h \right) } \right) . } \end{array}\tag{6}
$$

Here, $d _ { h }$ is the dimension of each attention head and all W terms and $\Gamma _ { \ell }$ are learned transformations. The attention weights jointly depend on neighboring node states and predicate embeddings. Their aggregation therefore propagates both memory content and typed relations into the updated node representations. After L layers, global mean pooling summarizes the node representations as $\mathbf { g } _ { q } ,$ and a two-layer nonlinear projector maps it to the memory graph token $\mathbf { z } _ { q }$ in the LLM embedding space:

Algorithm 1 Query-Aware Working Memory Activation   
Require: Memory set M; query q at $t _ { q } ;$ budgets n, k; RRF constant κ   
Ensure: Activated working memory $\bar { \mathcal { M } _ { q } }$   
1: $( \mathcal { C } _ { q } ^ { \mathrm { e n t } } , \mathcal { C } _ { q } ^ { \mathrm { l o c } } , \mathcal { C } _ { g } ^ { \mathrm { t i m e } } )  \phi ( q , \dot { t } _ { q } )$ ▷ query event constraints   
2: $\mathcal { L } _ { b } ^ { \bar { \mathbf { \alpha } } }  \underline { { \mathrm { T o p K } } } _ { m \in \mathcal { M } } ^ { \bar { \mathbf { \alpha } } }$ BM25(q, m) ▷ lexical retrieval   
3: $\mathcal { L } _ { d } \gets \mathrm { T o p K } _ { m \in \Lambda } ^ { n }$ Dense(q, m) ▷ semantic retrieval   
4: $\mathcal { L } _ { e } \gets \emptyset$   
5: $\mathbf { \widetilde { i f } } \mathbf { \mathcal { C } } _ { g } ^ { \mathrm { e n t } } \neq \mathbf { \emptyset } \lor \mathcal { C } _ { q _ { } } ^ { \mathrm { l o c } } \neq \mathbf { \emptyset } \lor \mathcal { C } _ { q } ^ { \mathrm { t i m e } } \neq \mathbf { \emptyset }$ then   
6: for m $\in \mathcal { M } ^ { \dagger }$ do   
7: $\mathbf { i f } \ C _ { q } ^ { \mathrm { e n t } } \neq \varnothing$ then   
8: s<sub>ent</sub>(m, q) ← SemMatch(C<sup>ent</sup>, E<sub>m</sub>)   
9: end if   
10: if $c _ { q } ^ { \mathrm { l o c } } \neq \emptyset$ then   
11: s (m, q) ← SemMatch $( \mathcal { C } _ { q } ^ { \mathrm { l o c } } , \mathcal { P } _ { m } )$   
12: end if   
13: i $\mathrm { \Delta } \mathfrak { f } \ : \mathcal { C } _ { q } ^ { \mathrm { t i m e } } \neq \varnothing$ then   
14: s<sub>time</sub>(m, q) ← TimeMatch(C<sup>time</sup><sub>q</sub> , (T<sup>f</sup><sub>m</sub>, T <sup>u</sup><sub>m</sub>))   
15: end if   
16: S<sub>ELT</sub>(m, q) ← mean<sub>x∈{ent,loc,time}:</sub> s<sub>x</sub>(m, q)   
C<sup>x</sup><sub>q</sub> ̸=∅   
17: end for   
18: $\mathcal { L } _ { e } \gets \mathrm { T o p K } _ { m \in \mathcal { M } } ^ { n } S _ { \mathrm { E L T } } ( m , q )$ ▷ event-attribute retrieval   
19: end if   
20: $\mathcal { U }  \mathcal { L } _ { b } \cup \mathcal { L } _ { d } \cup \mathcal { L } _ { e }$   
21: for m ∈ U do   
22: S<sub>RRF</sub>(m, q) ← X  κ + rank<sub>L</sub>(m)<sup>−1</sup>   
L∈{L<sub>b</sub>,L<sub>d</sub>,L<sub>e</sub>}   
m∈L   
23: end for   
24: C<sub>0</sub> ← TopK<sup>n</sup> S<sub>RRF</sub>(m, q)   
25: $\mathcal { M } _ { q }  \mathrm { T o p K } _ { m \in \mathcal { C } _ { 0 } } ^ { k }$ Reranker(q, m) ▷ working memory   
26: return $\mathcal { M } _ { q }$

$$
\begin{array} { r l } & { { \bf g } _ { q } = \frac { 1 } { | \mathcal { V } _ { q } | } \displaystyle \sum _ { i \in \mathcal { V } _ { q } } { \bf h } _ { i } ^ { ( L ) } , } \\ & { { \bf z } _ { q } = \mathbf { W } _ { 2 } \sigma ( \mathbf { W } _ { 1 } \mathbf { g } _ { q } + \mathbf { b } _ { 1 } ) + { \bf b } _ { 2 } . } \end{array}\tag{7}
$$

The resulting $\mathbf { z } _ { q }$ is supplied to the LLM. The graph encoder and projector parameters are collected in $\omega ,$ , with their configuration reported in Section B of the Appendix.

3) Graph-Text Fusion: The LLM text embedder converts the serialized working memory $C _ { q }$ into the textual token sequence $\mathbf { h } _ { t }$ . As illustrated in Figure 3, graph-text fusion concatenates the graph token $\mathbf { z } _ { q }$ with $\mathbf { h } _ { t }$ and supplies $[ { \bf z } _ { q } , { \bf h } _ { t } ]$ to the frozen LLM together with the query $q .$ For an answer sequence a, the resulting autoregressive distribution is $p _ { \theta , \omega } ( a _ { j } \mid \mathbf { a } _ { < j } , q , [ \mathbf { z } _ { q } , \mathbf { h } _ { t } ] )$ where θ and ω are the frozen LLM parameters and trainable graph-module parameters, respectively. This fusion preserves readable, source-grounded memory content while supplying graph-structural signals for relation-aware generation.

4) Training Objective: QGMEM first builds a persistent memory store through event-centric memory construction. Query-aware working memory activation and graph-augmented generation then transform query-relevant memories into textual and structural representations, which are supplied to an LLM for answer generation. These memory operations are decoupled from the LLM backbone model, allowing the framework to support different LLM backbone models through an appropriate generation interface. Accordingly, optimization is restricted to the parameters of the relation-aware graph encoder and the two-layer projector, whereas both the sentence encoder and the backbone LLM remain fixed throughout training. Given $N _ { D }$ graph-grounded training instances, the s-th instance comprises a query $q ^ { ( s ) }$ , serialized working memory $C _ { q } ^ { ( s ) }$ , local graph $\mathcal { G } _ { q } ^ { ( s ) }$ , and target sequence $\mathbf { a } ^ { ( s ) }$ . The graph branch produces $\mathbf { z } _ { q } ^ { ( s ) }$ , while the text branch produces $\mathbf { h } _ { t } ^ { ( s ) } ;$ ; their concatenation conditions the frozen LLM together with $q ^ { ( s ) }$ . The objective is evaluated over the target tokens, whose total number is $\begin{array} { r } { T _ { D } = \sum _ { s = 1 } ^ { N _ { D } } \left| \mathbf { a } ^ { ( s ) } \right| } \end{array}$ . The teacher-forced objective is

$$
\mathcal { T } ( \omega ) = - \frac { 1 } { T _ { D } } \sum _ { s = 1 } ^ { N _ { D } } \sum _ { j = 1 } ^ { | \mathbf { a } ^ { ( s ) } | }\tag{8}
$$

We minimize $\mathcal { I } ( \omega )$ in (8) with respect to $\omega .$ Consequently, answer supervision encourages the encoded graph representation to retain relational information that contributes to generation without altering the pretrained LLM. Detailed training settings are provided in Section B of the Appendix.

## D. Backbone Adaptability and Efficiency Analysis

1) Adaptability Across LLM Backbones: QGMEM supports different backbone models, enabling the complete framework to operate with both open-weight and proprietary LLMs without reliance on a specific model. For open-weight backbones, the projector maps the encoded local graph to a graph token in the embedding space of the target LLM, which is provided jointly with the textual working memory. For proprietary backbones accessed through an API, the embedding space is inaccessible; accordingly, the activated working memory and local graph are serialized as text. This interface-level adaptation preserves the memory construction, activation, and graph organization procedures across backbones while accommodating their respective access constraints.

2) Efficiency Analysis: Long-term memory construction is performed offline once, and the resulting atomic units, dynamic traces, and retrieval indexes are reused across queries. Atomic memory construction processes the dialogue through fixedsize overlapping windows. Therefore, the number of extraction operations grows linearly with the number of dialogue turns. These windows can be processed independently, and their construction cost is amortized over all subsequent queries. For dynamic trace consolidation, each atomic unit examines only its 20 nearest fact-level neighbors, avoiding exhaustive pairwise comparison over the complete memory store.

At query time, retrieval and reranking activate at most k atomic memory units. Since each activated unit contributes one primary structured fact, the induced local graph contains at most k directed edges and 2k nodes before node deduplication. Local graph materialization therefore requires O(k) time and space. With the best working-memory budget $k = 1 0$ , both graph construction and graph encoding operate on a bounded structure whose size does not grow with the complete dialogue history. This design confines explicit relational organization to query-relevant evidence while reusing the lightweight persistent memory across queries.

## IV. EXPERIMENTS

We evaluate QGMEM across complementary requirements for agent memory [48]: accurate retrieval, conflict resolution, ultra-long-history scalability, backbone adaptability, and efficiency. We further analyze retrieval quality, component contributions, hyperparameters, and a representative case.

## A. Experimental Setup

1) Benchmarks: For accurate retrieval, LoCoMo [13] tests evidence retrieval and composition across multi-session dialogues, while LongMemEval-S [14] tests persistent recall, reasoning, and updates. For conflict resolution, FactConsolidation-SH and FactConsolidation-MH from MemoryAgentBench [15] cover single- and multi-hop selection among conflicting facts. For ultra-long scalability, BEAM [16] provides histories of 100K, 500K, 1M, and 10M tokens.

2) Baselines: We compare QGMEM with seven representative systems. The retrieval-augmented baseline is RAG; the flat-structured memory baselines are Mem0 [23] and LightMem [24]; and the graph-based memory baselines are EMem-G [7], LiCoMemory [28], MR-Agent [49], and AriadneMem [39]. We use the same questions and backbone model for all available systems on each benchmark. Further implementation details are provided in Section D of the Appendix.

3) Evaluation Metrics: We use two answer-quality metrics. (1) LLM-judged Score. A judge evaluates whether each answer satisfies the reference or task rubric [50]; we repeat judging 10 times and report mean and standard deviation. For BEAM, we follow its official protocol, scoring nine nuggetbased abilities and Event Ordering with normalized Kendall’s $\tau _ { b } ,$ then macro-averaging the ten ability scores. (2) Answer Semantic Coverage. AlignScore-large [51] measures coverage of reference information; uncertainty is estimated with 10,000 bootstrap resamples.

## B. Quantitative Evaluation

1) Accurate Retrieval: Table I shows that QGMEM leads on LoCoMo and LongMemEval-S under both metrics. Its LLM judged scores of 82.77% and 62.14% exceed the strongest baselines by 8.30 and 5.72 points; semantic coverage reaches 56.31% and 51.31%, gains of 12.77 and 2.78 points. Agreement across the two evaluators indicates that event-indexed storage and query-aware activation prioritize complete answer-bearing memories across multi-session evidence.

2) Conflict Resolution: On Fact-SH and Fact-MH, QGMEM attains LLM-judged scores of 78.10% and 12.50%, surpassing the strongest baselines by 2.18 and 6.00 points. Fact-MH nearly doubles the strongest baseline, with semantic coverage improving by 5.69 points to 13.34%; Fact-SH coverage reaches 80.58%. The larger multi-hop gain supports the division of labor between dynamic traces and time-aware activation for state selection and the local graph for evidence composition.

TABLE I  
MAIN RESULTS WITH QWEN3-8B ACROSS SIX LONG-TERM MEMORY BENCHMARKS. BOLDFACE AND UNDERLINING INDICATE THE BEST AND SECOND-BEST PERFORMANCE RESPECTIVELY.
<table><tr><td rowspan="2">Method</td><td colspan="2">Accurate Retrieval</td><td colspan="2">Conflict Resolution</td><td colspan="2">Ultra-Long Scalability</td></tr><tr><td>LoCoMo  $N { = } I 5 4 0$ </td><td>LongMemEval-S N=500</td><td>Fact-SH N=400</td><td>Fact-MH N=400</td><td>BEAM-100K N=400</td><td>BEAM-500K N=700</td></tr><tr><td colspan="7">(a) LLM-judged Score (%)</td></tr><tr><td>Retrieval-Augmented Baseline</td><td> $5 7 . 9 3 { \pm } 0 . 0 6$ </td><td> $3 4 . 7 4 { \pm } 0 . 1 0$ </td><td> $3 7 . 8 2 { \pm } 0 . 2 4 $ </td><td> $3 . 9 3 \pm 0 . 1 7$ </td><td>29.04</td><td>28.00</td></tr><tr><td colspan="7">RAG</td></tr><tr><td>Flat-Structured Memory Baselines</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Mem0</td><td> $4 2 . 1 8 { \pm } 0 . 1 8$ </td><td> $2 5 . 9 6 { \pm } 0 . 0 8$ </td><td> $7 5 . 9 2 { \pm } 0 . 1 7 $ </td><td> $5 . 0 8 { \pm } 0 . 2 4$ </td><td>31.61</td><td>33.50</td></tr><tr><td>LightMem</td><td> $5 7 . 5 8 { \pm } 0 . 2 3 $ </td><td> $5 6 . 4 2 { \pm } 0 . 1 8 $ </td><td> $6 2 . 2 5 { \pm } 0 . 1 7$ </td><td> $\underline { { 6 . 5 0 \pm 0 . 0 0 } }$ </td><td>27.79</td><td>29.75</td></tr><tr><td colspan="7">Graph-Based Memory Baselines</td></tr><tr><td></td><td></td><td></td><td></td><td> $4 . 3 0 { \pm } 0 . 1 6$ </td><td></td><td></td></tr><tr><td>EMem-G LiCoMemory</td><td> $6 2 . 4 8 { \pm } 0 . 1 8$ </td><td> $4 9 . 0 0 { \pm } 0 . 1 6 $   $4 7 . 2 0 { \pm } 0 . 1 4$ </td><td> $3 6 . 8 5 { \pm } 1 . 1 3$   $2 2 . 2 5 { \pm } 0 . 1 9$ </td><td> $1 . 5 0 { \pm } 0 . 1 2 \ $ </td><td>16.14 13.34</td><td>16.13 13.03</td></tr><tr><td>MR-Agent</td><td> $7 4 . 4 7 { \pm } 0 . 1 0$   $\overline { { 4 3 . 4 6 { \pm 0 . 1 5 } } }$ </td><td> $2 3 . 2 0 { \pm } 0 . 1 2$ </td><td> $2 0 . 5 0 { \pm } 0 . 2 1 $ </td><td> $3 . 0 0 { \pm } 0 . 1 5 $ </td><td>12.89</td><td>16.85</td></tr><tr><td>AriadneMem</td><td> $5 1 . 4 2 { \pm } 0 . 1 0$ </td><td> $1 1 . 0 0 { \pm } 0 . 0 9 \ $ </td><td> $6 4 . 5 0 { \pm } 0 . 2 8 $ </td><td> $4 . 7 5 { \pm } 0 . 1 8$ </td><td>27.41</td><td>28.70</td></tr><tr><td>QGMEM (ours)</td><td> $\mathbf { 8 2 . 7 7 \pm 0 . 1 3 }$ </td><td> $\mathbf { 6 2 . 1 4 \pm 0 . 1 3 }$ </td><td> $\mathbf { 7 8 . 1 0 { \pm } 0 . 2 4 }$ </td><td> $\mathbf { 1 2 . 5 0 { \overset { . } { \bot } } 0 . 0 0 }$ </td><td>35.31</td><td>38.91</td></tr><tr><td colspan="7">(b) Answer Semantic Coverage (%)</td></tr><tr><td colspan="7">Retrieval-Augmented Baseline</td></tr><tr><td>RAG</td><td> $\underline { { 4 3 . 5 4 \pm 1 . 1 4 } }$ </td><td> $2 9 . 8 5 { \pm } 1 . 8 4 $ </td><td> $3 8 . 7 1 \pm 2 . 3 9$ </td><td> $5 . 0 0 { \scriptstyle \pm 0 . 9 5 }$ </td><td> $7 . 3 2 { \pm } 1 . 0 0 $ </td><td> $8 . 5 4 \pm 0 . 8 2$ </td></tr><tr><td>Flat-Structured Memory Baselines</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Mem0</td><td> $2 4 . 7 5 { \pm } 0 . 9 9$ </td><td> $2 6 . 5 4 \pm 1 . 6 4$ </td><td> $\mathbf { 8 1 . 7 0 { \pm } 1 . 8 7 }$ </td><td> $2 . 9 5 \pm 0 . 7 1$ </td><td> $8 . 8 2 \pm 1 . 1 4$ </td><td> $1 1 . 5 0 { \pm } 0 . 9 5 $ </td></tr><tr><td>LightMem</td><td> $3 5 . 4 5 { \pm } 1 . 1 3$ </td><td> $4 8 . 5 3 { \pm } 1 . 8 1 $ </td><td> $6 1 . 0 0 { \scriptstyle \pm 2 . 2 7 }$ </td><td> $7 . 6 5 { \pm } 1 . 1 6 $ </td><td> $8 . 8 8 { \pm } 1 . 1 5 $ </td><td> $\overline { { 9 . 5 0 { \pm } 0 . 8 6 } }$ </td></tr><tr><td colspan="7">Graph-Based Memory Baselines</td></tr><tr><td>EMem-G</td><td> $3 4 . 5 7 { \pm } 1 . 1 3$ </td><td> $1 8 . 3 7 { \pm } 1 . 6 0 $ </td><td> $4 0 . 0 9 { \pm } 2 . 4 1 $ </td><td> $7 . 5 7 { \pm } 1 . 2 1 $ </td><td> $5 . 8 6 \pm 0 . 8 1$ </td><td> $5 . 9 8 { \pm } 0 . 5 7 $ </td></tr><tr><td>LiCoMemory</td><td> $3 5 . 9 7 { \pm } 1 . 1 0 $ </td><td> $3 8 . 3 0 { \pm } 1 . 9 6 $ </td><td> $2 2 . 0 6 \pm 1 . 9 6$ </td><td> $2 . 3 3 { \pm } 0 . 5 7$ </td><td> $4 . 0 2 \pm 0 . 7 6$ </td><td> $4 . 2 0 { \pm } 0 . 5 7 \ $ </td></tr><tr><td>MR-Agent AriadneMem</td><td> $2 0 . 4 4 { \pm } 0 . 9 1$ </td><td> $2 1 . 0 3 { \pm } 1 . 7 0 $ </td><td> $2 0 . 3 9 { \pm } 1 . 8 8 $ </td><td> $3 . 7 9 { \pm } 0 . 8 2$ </td><td> $3 . 9 8 \pm 0 . 6 6$ </td><td> $5 . 6 1 \pm 0 . 5 9$ </td></tr><tr><td></td><td> $2 3 . 4 4 \pm 0 . 9 7$ </td><td> $6 . 9 3 \pm 0 . 9 3$ </td><td> $6 6 . 4 2 \pm 2 . 3 3$ </td><td> $5 . 7 1 \pm 1 . 0 2$ </td><td> $7 . 0 5 { \pm } 0 . 9 4$ </td><td> $1 0 . 0 7 { \scriptstyle \pm 0 . 9 0 }$ </td></tr><tr><td>QGMEM (ours)</td><td> $\mathbf { 5 6 . 3 1 \pm 1 . 0 8 }$ </td><td> $\mathbf { 5 1 . 3 1 \pm 1 . 9 1 }$ </td><td> $8 0 . 5 8 { \pm } 1 . 9 5 $ </td><td> $\mathbf { 1 3 . 3 4 \pm 1 . 6 3 }$ </td><td> $\mathbf { 1 6 . 1 5 \pm 1 . 5 7 }$ </td><td> $\mathbf { 1 6 . 9 5 \pm 1 . 2 0 }$ </td></tr></table>

3) Ultra-Long Dialogue History Scalability: Across both scales, QGMEM achieves overall scores of 35.31%/38.91% on BEAM-100K/500K, exceeding Mem0 by 3.70/5.41 points, and semantic coverage of 16.15%/16.95%, with gains of 7.27/5.45 points. Its nine-ability Nugget Scores are 36.81%/40.26%, with Event Ordering $\tau _ { b }$ of 21.77%/26.79%, showing that compact activation supports diverse memory abilities under long histories.

Figure 4 extends BEAM from 100K to 10M tokens; methods share questions and histories within each scale, while question sets differ across scales. QGMEM leads throughout, remains above 35% through 1M tokens, and scores 29.81% at 10M, exceeding LightMem and AriadneMem by 20.18 and 21.21 points. Event-indexed retrieval limits irrelevant-history interference, reranking selects a fixed-budget working memory, and the local graph grows with activated evidence rather than full history.

4) Retrieval Quality Analysis: Recall@10 measures goldevidence coverage, Precision@10 the concentration of relevant units, Sufficiency@10 whether the set jointly supports the answer, and nDCG@10 rank quality [52]. Table II shows that QGMEM combines 89.08% Recall@10 with the best Precision@10 (20.92%), Sufficiency@10 (97.11%), and nDCG@10 (82.50%). Precision and nDCG improve by 3.00 and 14.34 points, indicating that answer-supporting memories are both covered and concentrated at high ranks; the sufficiency score confirms coverage of combined multi-evidence requirements.

![](images/b4f68320af61f45170ee8775b6deb3e9cbf6257907c1101ae1e5d536899c5285.jpg)  
Fig. 4. Scalability on BEAM with Qwen3-8B. All methods use the same questions and histories within each scale, while the evaluation sets differ across scales.

5) Adaptability Across LLM Backbones: We compare Qwen3-8B, Llama3.1-8B-Instruct [18], and GPT-4o-mini [19] on four benchmarks. Figure 5 shows the highest average for every backbone: 64.58%, 56.00%, and 70.11%, exceeding the strongest corresponding baselines by 13.57, 5.31, and 13.80 points.

QGMEM leads on LoCoMo, LongMemEval-S, and BEAM-100K for all three backbones, and reaches 78.10%/81.25% on Fact-SH with Qwen3-8B/GPT-4o-mini. GPT-4o-mini receives a textual serialization because its embedding space is inaccessible; its gains show that event-indexed memory and query-aware organization transfer across graph-token and textonly interfaces.

6) Efficiency Analysis: Table III reports online LLM and retrieved-context tokens, end-to-end online latency, and amortized offline construction latency on LoCoMo. QGMEM uses 598.19 retrieved-context and 1208.79 total online tokens per QA, completes the full retrieval-to-generation pipeline in 5.94 seconds, and improves the LoCoMo score over EMem-G by 20.29 points. The 1.09-second offline value amortizes onetime construction of atomic units, traces, and indexes across all questions. Because the local graph is induced from fixedbudget working memory, longer histories enlarge the retrieval store without increasing the maximum graph supplied to the encoder or generator.

![](images/2d4bb824ea52b8fc0b777e9be7d5a95365fd4b1d8514a452cf420ad57f1a739e.jpg)

Fig. 5. Adaptability across LLM backbones on four benchmarks with Qwen3-8B, Llama3.1-8B-Instruct, and GPT-4o-mini.  
TABLE II  
THE EVIDENCE RETRIEVAL PERFORMANCE OF QGMEM COMPARED WITH REPRESENTATIVE BASELINES. HIGHER VALUES INDICATE BETTER PERFORMANCE FOR ALL METRICS.
<table><tr><td>Method</td><td>R@10</td><td>P@10</td><td>Suff.@10</td><td>nDCG@10</td></tr><tr><td>RAG</td><td>90.34</td><td>17.92</td><td>96.71</td><td>68.16</td></tr><tr><td>Mem0</td><td>82.77</td><td>16.42</td><td>86.71</td><td>48.13</td></tr><tr><td>LightMem</td><td>80.67</td><td>11.04</td><td>89.60</td><td>43.46</td></tr><tr><td>EMem-G</td><td>78.57</td><td>5.09</td><td>80.35</td><td>26.79</td></tr><tr><td>LiCoMemory</td><td>81.51</td><td>12.54</td><td>84.97</td><td>47.46</td></tr><tr><td>MR-Agent</td><td>55.88</td><td>3.99</td><td>45.09</td><td>28.73</td></tr><tr><td>AriadneMem</td><td>55.46</td><td>8.55</td><td>56.07</td><td>28.56</td></tr><tr><td>QGMEM (ours)</td><td>89.08</td><td>20.92</td><td>97.11</td><td>82.50</td></tr></table>

TABLE III

EFFICIENCY ON LOCOMO WITH QWEN3-8B. ONLINE LLM TOKENS INCLUDE ALL QUERY-TIME CALLS; OFFLINE LATENCY IS AMORTIZED PER QA.
<table><tr><td>Method</td><td colspan="2">Online</td><td>Ret. ctx.</td><td>Offline</td></tr><tr><td></td><td>tok./QA</td><td>Lat. (s/QA)</td><td>tok./QA</td><td>amort. s/QA</td></tr><tr><td>RAG</td><td>3430.01</td><td>3.84</td><td>3194.50</td><td>0.02</td></tr><tr><td>Mem0</td><td>879.68</td><td>3.49</td><td>656.12</td><td>5.07</td></tr><tr><td>LightMem</td><td>819.29</td><td>2.99</td><td>620.79</td><td>1.48</td></tr><tr><td>LiCoMemory</td><td>616.48</td><td>5.13</td><td>391.95</td><td>0.57</td></tr><tr><td>EMem-G</td><td>1505.78</td><td>7.18</td><td>563.35</td><td>0.24</td></tr><tr><td>QGMEM (ours)</td><td>1208.79</td><td>5.94</td><td>598.19</td><td>1.09</td></tr></table>

## C. Ablation and Analysis

1) Component ablation: Table IV reports five variants; Macro Avg. is the unweighted mean across six benchmarks. For memory construction, excluding atomic memory units produces the largest decrease in Macro Avg. at 7.43 points. The decreases are 14.12 points on LongMemEval-S, 8.03/7.75 on BEAM-100K/500K, and 5.42 on Fact-MH. Event-level retrieval granularity therefore affects both long-history retrieval and multi-fact composition. Dynamic memory traces contribute

2.08 points to Macro Avg., with larger effects on Fact-MH (2.89 points) and BEAM-500K (4.23 points), two settings that place greater demands on tracking distant state changes.

For memory retrieval, excluding the reranker decreases Macro Avg. by 5.23 points, the second-largest change in the table. The corresponding decreases are 8.95/9.58 points on LoCoMo/LongMemEval-S and 2.53/5.53 points on BEAM-100K/500K. Candidate fusion therefore requires query-specific prioritization before answer generation. Excluding hybrid retrieval has a smaller overall effect of 1.44 points, but decreases Fact-SH and BEAM-100K by 2.78 and 2.30 points, respectively. The three retrieval channels thus improve candidate coverage before reranking.

For graph-augmented generation, excluding the graph token decreases Macro Avg. by 3.95 points. The decreases on Fact-SH, LongMemEval-S, Fact-MH, and BEAM-500K are 7.81, 7.06, 3.40, and 4.91 points, respectively. The reductions across these benchmarks suggest that graph topology complements the temporal and state-validity information carried by dynamic traces and serialized working memory.

2) Qualitative analysis: Figure 6 illustrates three failures: Mem0 retrieves “a partner to dance with” only at Rank 9; LightMem retrieves it at Rank 8 but answers “a marathon”; and EMem-G selects a nearby motivation relation at Rank 4 and answers “a picture of progress.” These correspond to ranking, evidence-utilization, and relation-selection errors.

QGMEM preserves the comparison as a traceable atomic unit, reranks it to Rank 1, and connects Jon and Gina’s shared challenges to the comparison in the local graph. Text and graph evidence therefore agree on “dancing with a partner,” illustrating how memory granularity, selective activation, and local relational organization address the three failure points.

3) Hyperparameter sensitivity: Figure 7 varies one parameter at a time. For the retrieval budgets, accuracy increases with working-memory size up to K=10 and remains below this maximum at every larger value evaluated. The initial increase is consistent with improved evidence coverage, while the subsequent reduction indicates that additional units can introduce irrelevant or redundant evidence. Candidate-pool size has a different trend. Accuracy is highest at n=100, and larger pools provide no consistent improvement, indicating that the reranker has sufficient candidate diversity at this value.

For memory construction, w=6 achieves the highest accuracy by providing sufficient context for local references while retaining event-level separation. The neighboring settings remain competitive, although performance decreases when the window grows to ten turns. The macro-averaged result for relation consolidation is highest at $\tau { = } 0 . 8 0$ . Lower thresholds retain more weakly related facts, while higher thresholds remove useful cross-event connections. The observed values are consistent with a trade-off between relation coverage and relevance.

TABLE IV  
COMPONENT ABLATION OF QGMEM WITH QWEN3-8B ON EACH BENCHMARK. MACRO AVG. IS THE UNWEIGHTED MEAN ACROSS THE SIX BENCHMARKS, ASSIGNING EQUAL WEIGHT TO EACH BENCHMARK. HIGHER VALUES ARE BETTER; THE BEST AND SECOND-BEST RESULTS ARE BOLDED AND UNDERLINED RESPECTIVELY.
<table><tr><td>Setting</td><td>LoCoMo</td><td>LongMemEval-S</td><td>Fact-SH</td><td>Fact-MH</td><td>BEAM-100K</td><td>BEAM-500K</td><td>Macro Avg.</td></tr><tr><td>Full QGMEM</td><td>82.77</td><td>62.14</td><td>78.10</td><td>12.50</td><td>35.31</td><td>38.91</td><td>51.62</td></tr><tr><td colspan="8">Memory Construction Phase</td></tr><tr><td>w/o Dynamic Memory Traces</td><td>81.09</td><td>61.29</td><td>75.82</td><td>9.61</td><td>34.78</td><td>34.68</td><td> $4 9 . 5 5 ^ { \downarrow 2 . 0 8 }$ </td></tr><tr><td>w/o Atomic Memory Units</td><td>79.84</td><td>48.02</td><td>71.75</td><td>7.08</td><td>27.28</td><td>31.16</td><td> $4 4 . 1 9 \downarrow 7 . 4 3$ </td></tr><tr><td colspan="8">Memory Retrieval Phase</td></tr><tr><td>w/o Hybrid memory retrieval</td><td>81.93</td><td>61.58</td><td>75.32</td><td>11.90</td><td>33.01</td><td>37.34</td><td> $5 0 . 1 8 ^ { \downarrow 1 . 4 4 }$ </td></tr><tr><td>w/o Reranker</td><td>73.82</td><td>52.56</td><td>76.91</td><td>8.88</td><td>32.78</td><td>33.38</td><td> $\overline { { 4 6 . 3 9 } } \downarrow 5 . 2 3$ </td></tr><tr><td colspan="8">Graph-Augmented Generation Phase</td></tr><tr><td>w/o Graph Token</td><td>81.80</td><td>55.08</td><td>70.29</td><td>9.10</td><td>35.75</td><td>34.00</td><td> $4 7 . 6 7 ^ { \downarrow 3 . 9 5 }$ </td></tr></table>

<table><tr><td rowspan=1 colspan=2>Question: What did Jon and Gina compare their entrepreneurial journeys to?</td></tr><tr><td rowspan=1 colspan=1>Mem0</td><td rowspan=1 colspan=1>Key Memory: rank 9Jon and Gina find motivation in challenges, describing it as being like having a partner to dance with.Answer: Nothing is mentioned about comparing their entrepreneurial journeys. X</td></tr><tr><td rowspan=1 colspan=1>LightMem</td><td rowspan=1 colspan=1>Key Memory: rank 8Jon mentions that having someone like Gina to face challenges with is like having a partner to dance with.Answer: A marathon. X</td></tr><tr><td rowspan=1 colspan=1>Emem-G</td><td rowspan=1 colspan=1>Key Memory: rank 4Jon has been rehearsing and working on business plans ... but dancing has kept him going.Answer: A picture of progress. X</td></tr><tr><td rowspan=1 colspan=1>QGMem</td><td rowspan=1 colspan=1>Key Memory: rank 1Jon and Gina both face the same challenges .. it&#x27;s like having a partner to dance with.Answer: Dancing with a partner. </td></tr></table>

Fig. 6. A case study from LoCoMo. Mem0, LightMem, and EMem-G respectively illustrate ranking, evidence-utilization, and relation-selection failures, whereas QGMEM promotes the answer-bearing comparison to Rank 1 and produces the grounded answer.  
![](images/ea082771a4c9d0fb9a811ee20b92469d2e4867d22576ded85125878d90ff5fda.jpg)

![](images/875f8f9d68f2c7721c54990ec3c0c7a538aeab3289c88082d4bb3326623c43cd.jpg)

![](images/d6bac7b613e4c30135f5712f6f78ff3c18926a5af84211dd0dba560f6a106d5a.jpg)

![](images/8bc90b9ea7cea6e48aff597953a6bf01bc963e413932cc7b75b4d29545219014.jpg)  
Fig. 7. Hyperparameter sensitivity of QGMEM with Qwen3-8B. Panels (a) through (d) vary working memory size, candidate-pool size, construction-window size, and relation similarity threshold respectively. Orange markers indicate the maximum on each curve.

## V. CONCLUSION

QGMEM is an event-centric conversational memory framework with query-aware graph augmentation. Its atomic units and dynamic traces preserve grounded evidence and evolving states, while query-aware activation organizes relevant units for local graph reasoning. Across six benchmarks and contexts up to 10M tokens, QGMEM improves retrieval, multi-hop reasoning, conflict resolution, and scalability across diverse backbones with compact contexts.

## REFERENCES

[1] N. F. Liu, K. Lin, J. Hewitt, A. Paranjape, M. Bevilacqua, F. Petroni, and P. Liang, “Lost in the middle: How language models use long contexts,” Transactions of the Association for Computational Linguistics, vol. 12, pp. 157–173, 2024. [Online]. Available: https://aclanthology.org/2024.tacl-1.9/

[2] L. Wang, C. Ma, X. Feng, Z. Zhang, H. Yang, J. Zhang, Z. Chen, J. Tang, X. Chen, Y. Lin, W. X. Zhao, Z. Wei, and J. Wen, “A survey on large language model based autonomous agents,” Frontiers of Computer Science, vol. 18, p. 186345, 2024.

[3] W. Zhong, L. Guo, Q. Gao, H. Ye, and Y. Wang, “Memorybank: Enhancing large language models with long-term memory,” 2024. [Online]. Available: https://arxiv.org/abs/2305.10250

[4] C. Packer, S. Wooders, K. Lin, V. Fang, S. G. Patil, I. Stoica, and J. E. Gonzalez, “Memgpt: Towards llms as operating systems,” 2023. [Online]. Available: https://arxiv.org/abs/2310.08560

[5] J. Wang, S. Wang, Z. Xia, S. Hong, Y. Zhu, B. Liu, and C. Wu, “Fact: Examining the effectiveness of iterative context rewriting for multi-fact retrieval,” 2024. [Online]. Available: https://arxiv.org/abs/2410.21012

[6] Y. Wu, Y. Zhang, S. Liang, and Y. Liu, “Sgmem: Sentence graph memory for long-term conversational agents,” 2025. [Online]. Available: https://arxiv.org/abs/2509.21212

[7] S. Zhou and J. Han, “A simple yet strong baseline for long-term conversational memory of llm agents,” 2025. [Online]. Available: https://arxiv.org/abs/2511.17208

[8] C. Yang, C. Zhou, Y. Xiao, S. Dong, L. Zhuang, Y. Zhang, Z. Wang, Z. Hong, Z. Yuan, Z. Xiang, S. Chen, H. Zhou, Q. Zhang, N. Liu, J. Su, X. Wang, Y. Chang, and X. Huang, “Graph-based agent memory: Taxonomy, techniques, and applications,” 2026. [Online]. Available: https://arxiv.org/abs/2602.05665

[9] J. M. Zacks, N. K. Speer, K. M. Swallow, T. S. Braver, and J. R. Reynolds, “Event perception: A mind-brain perspective,” Psychological Bulletin, vol. 133, no. 2, pp. 273–293, 2007.

[10] G. A. Radvansky, “Across the event horizon,” Current Directions in Psychological Science, vol. 21, no. 4, pp. 269–272, 2012.

[11] G. A. Radvansky and J. M. Zacks, “Event boundaries in memory and cognition,” Current Opinion in Behavioral Sciences, vol. 17, pp. 133–140, 2017.

[12] G. V. Cormack, C. L. A. Clarke, and S. Büttcher, “Reciprocal rank fusion outperforms condorcet and individual rank learning methods,” in Proceedings of the 32nd International ACM SIGIR Conference on Research and Development in Information Retrieval, 2009, pp. 758–759.

[13] A. Maharana, D.-H. Lee, S. Tulyakov, M. Bansal, F. Barbieri, and Y. Fang, “Evaluating very long-term conversational memory of llm agents,” 2024. [Online]. Available: https://arxiv.org/abs/2402.17753

[14] D. Wu, H. Wang, W. Yu, Y. Zhang, K.-W. Chang, and D. Yu, “Longmemeval: Benchmarking chat assistants on long-term interactive memory,” 2025. [Online]. Available: https://arxiv.org/abs/2410.10813

[15] Y. Hu, Y. Wang, and J. McAuley, “Evaluating memory in llm agents via incremental multi-turn interactions,” 2025. [Online]. Available: https://arxiv.org/abs/2507.05257

[16] M. Tavakoli, A. Salemi, C. Ye, M. Abdalla, H. Zamani, and J. R. Mitchell, “Beyond a million tokens: Benchmarking and enhancing long-term memory in llms,” in International Conference on Learning Representations, 2026. [Online]. Available: https: //openreview.net/forum?id=y59hf5lrMn

[17] A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao et al., “Qwen3 technical report,” 2025. [Online]. Available: https://arxiv.org/abs/2505.09388

[18] A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur et al., “The llama 3 herd of models,” 2024. [Online]. Available: https://arxiv.org/abs/2407.21783

[19] OpenAI, “Gpt-4o mini: advancing cost-efficient intelligence,” 2024, accessed: 2026-05-20. [Online]. Available: https://openai.com/index/ gpt-4o-mini-advancing-cost-efficient-intelligence/

[20] Z. Zhang, Q. Dai, X. Bo, C. Ma, R. Li, X. Chen, J. Zhu, Z. Dong, and J.-R. Wen, “A survey on the memory mechanism of large language model-based agents,” ACM Transactions on Information Systems, vol. 43, no. 6, pp. 155:1–155:47, 2025.

[21] Y. Hu, S. Liu, Y. Yue, G. Zhang, B. Liu, F. Zhu, J. Lin, H. Guo, S. Dou, Z. Xi, S. Jin, J. Tan, Y. Yin, J. Liu, Z. Zhang, Z. Sun, Y. Zhu, H. Sun, B. Peng, Z. Cheng, X. Fan, J. Guo, X. Yu, Z. Zhou, Z. Hu, J. Huo, J. Wang, Y. Niu, Y. Wang, Z. Yin, X. Hu, Y. Liao, Q. Li, K. Wang, W. Zhou, Y. Liu, D. Cheng, Q. Zhang, T. Gui, S. Pan, Y. Zhang, P. Torr, Z. Dou, J.-R. Wen, X. Huang, Y.-G. Jiang, and

S. Yan, “Memory in the age of ai agents,” 2026. [Online]. Available: https://arxiv.org/abs/2512.13564

[22] W. Wang, L. Dong, H. Cheng, X. Liu, X. Yan, J. Gao, and F. Wei, “Augmenting language models with long-term memory,” 2023. [Online]. Available: https://arxiv.org/abs/2306.07174

[23] P. Chhikara, D. Khant, S. Aryan, T. Singh, and D. Yadav, “Mem0: Building production-ready ai agents with scalable long-term memory,” 2025. [Online]. Available: https://arxiv.org/abs/2504.19413

[24] J. Fang, X. Deng, H. Xu, Z. Jiang, Y. Tang, Z. Xu, S. Deng, Y. Yao, M. Wang, S. Qiao, H. Chen, and N. Zhang, “Lightmem: Lightweight and efficient memory-augmented generation,” 2026. [Online]. Available: https://arxiv.org/abs/2510.18866

[25] W. Xu, Z. Liang, K. Mei, H. Gao, J. Tan, and Y. Zhang, “Amem: Agentic memory for llm agents,” 2025. [Online]. Available: https://arxiv.org/abs/2502.12110

[26] Y. Zhang, W. Yuan, and Z. Jiang, “Bridging intuitive associations and deliberate recall: Empowering LLM personal assistant with graph-structured long-term memory,” in Findings of the Association for Computational Linguistics: ACL 2025. Association for Computational Linguistics, 2025, pp. 17 533–17 547. [Online]. Available: https: //aclanthology.org/2025.findings-acl.901/

[27] P. Rasmussen, P. Paliychuk, T. Beauvais, J. Ryan, and D. Chalef, “Zep: A temporal knowledge graph architecture for agent memory,” 2025. [Online]. Available: https://arxiv.org/abs/2501.13956

[28] Z. Huang, Z. Tian, Q. Guo, F. Zhang, Y. Zhou, D. Jiang, Z. Xie, and X. Zhou, “Licomemory: Lightweight and cognitive agentic memory for efficient long-term reasoning,” 2026. [Online]. Available: https://arxiv.org/abs/2511.01448

[29] Z. Wu, H. Zhang, F. Lin, W. Xu, X. Xu, Y. Chen, H. P. Zou, S. Chen, W. Zhang, X. Liu, P. S. Yu, and H. Wang, “Gam: Hierarchical graph-based agentic memory for llm agents,” 2026. [Online]. Available: https://arxiv.org/abs/2604.12285

[30] J. Yue, C. Hu, J. Sheng, Z. Zhou, W. Zhang, T. Liu, L. Guo, and Y. Deng, “Hypermem: Hypergraph memory for long-term conversations,” 2026. [Online]. Available: https://arxiv.org/abs/2604.08256

[31] P. Banerjee, M. Moshtaghi, S. Subramanian, A. Misra, and A. Chadha, “Apex-mem: Agentic semi-structured memory with temporal reasoning for long-term conversational ai,” 2026. [Online]. Available: https://arxiv.org/abs/2604.14362

[32] H. P. Van, N. M. Hieu, K. P. T. Tuan, N. L. Hai, L. N. Van, N. T. N. Diep, and T. Le, “MemORAI: Memory organization and retrieval via adaptive graph intelligence for LLM conversational agents,” in Findings of the Association for Computational Linguistics: ACL 2026. Association for Computational Linguistics, 2026, pp. 28 235–28 253. [Online]. Available: https://aclanthology.org/2026.findings-acl.1408/

[33] B. Xu, Y. Chen, J. Fang, R. Zhong, Y. Yao, Y. Zhu, L. Du, and S. Deng, “StructMem: Structured memory for long-horizon behavior in LLMs,” in Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers). Association for Computational Linguistics, 2026, pp. 122–146. [Online]. Available: https://aclanthology.org/2026.acl-short.12

[34] S. Hu, Y. Wei, J. Ran, X. Han, Z. Yao, H. Wang, R. Chen, and L. Zou, “Does memory need graphs? a unified framework and empirical analysis for long-term dialog memory,” in Proceedings ofthe 64th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, 2026, pp. 26 758–26 782. [Online]. Available: https://aclanthology.org/2026.acl-long.1232/

[35] S. E. Robertson and H. Zaragoza, “The probabilistic relevance framework: BM25 and beyond,” Foundations and Trends in Information Retrieval, vol. 3, no. 4, pp. 333–389, 2009.

[36] N. Reimers and I. Gurevych, “Sentence-BERT: Sentence embeddings using siamese BERT-networks,” in Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, 2019, pp. 3982–3992. [Online]. Available: https://aclanthology.org/D19-1410

[37] V. Karpukhin, B. Oguz, S. Min, P. Lewis, L. Wu, S. Edunov, D. Chen, and W. tau Yih, “Dense passage retrieval for open-domain question answering,” in Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, 2020, pp. 6769–6781. [Online]. Available: https://aclanthology.org/2020.emnlp-main.550

[38] G. Izacard, M. Caron, L. Hosseini, S. Riedel, P. Bojanowski, A. Joulin, and E. Grave, “Unsupervised dense information retrieval with contrastive learning,” Transactions on Machine Learning Research, 2022. [Online]. Available: https://openreview.net/forum?id=jKN1pXi7b0

[39] W. Zhu, X. Chen, Z. Wang, J. Wang, X. Dong, M. Huang, R. Cai, H. Sang, H. Wang, P. Qiu, Y. Deng, P. Tiwari, B. H. Rappazzo, and

Y. Wang, “Ariadnemem: Threading the maze of lifelong memory for llm agents,” 2026. [Online]. Available: https://arxiv.org/abs/2603.03290

[40] D. Edge, H. Trinh, N. Cheng, J. Bradley, A. Chao, A. Mody, S. Truitt, D. Metropolitansky, R. O. Ness, and J. Larson, “From local to global: A graph rag approach to query-focused summarization,” 2025. [Online]. Available: https://arxiv.org/abs/2404.16130

[41] B. J. Gutierrez, Y. Shu, Y. Gu, M. Yasunaga, and Y. Su, “HippoRAG: Neurobiologically inspired long-term memory for large language models,” in Advances in Neural Information Processing Systems, vol. 37, 2024. [Online]. Available: https://arxiv.org/abs/2405.14831

[42] J. Tang, Y. Yang, W. Wei, L. Shi, L. Su, S. Cheng, D. Yin, and C. Huang, “GraphGPT: Graph instruction tuning for large language models,” in Proceedings ofthe 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2024, pp. 491– 500.

[43] R. Chen, T. Zhao, A. K. Jaiswal, N. Shah, and Z. Wang, “LLaGA: Large language and graph assistant,” in Proceedings of the 41st International Conference on Machine Learning, vol. 235. PMLR, 2024, pp. 7809–7823. [Online]. Available: https://proceedings.mlr.press/v235/chen24bh.html

[44] X. He, Y. Tian, Y. Sun, N. V. Chawla, T. Laurent, Y. LeCun, X. Bresson, and B. Hooi, “G-retriever: Retrieval-augmented generation for textual graph understanding and question answering,” 2024. [Online]. Available: https://arxiv.org/abs/2402.07630

[45] P. Jiang, L. Cao, R. Zhu, M. Jiang, Y. Zhang, J. Shen, J. Sun, and J. Han, “Ras: Retrieval-and-structuring for knowledge-intensive llm generation,” 2026. [Online]. Available: https://arxiv.org/abs/2502.10996

[46] M. Huang, C. Bu, Y. He, X. Zhuo, and X. Wu, “Relink: Constructing query-driven evidence graph on-the-fly for graphrag,” 2026, accepted by AAAI 2026. [Online]. Available: https://arxiv.org/abs/2601.07192

[47] Y. Shi, Z. Huang, S. Feng, H. Zhong, W. Wang, and Y. Sun, “Masked label prediction: Unified message passing model for semi-supervised classification,” in Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, 2021, pp. 1548–1554.

[48] H. Tan, Z. Zhang, C. Ma, X. Chen, Q. Dai, and Z. Dong, “MemBench: Towards more comprehensive evaluation on the memory of LLM-based agents,” in Findings of the Association for Computational Linguistics: ACL 2025. Association for Computational Linguistics, 2025, pp. 19 336– 19 352.

[49] S. Ji, Y. Li, and B. Hooi, “Memory is reconstructed, not retrieved: Graph memory for llm agents,” 2026, accepted at ICML 2026. [Online]. Available: https://arxiv.org/abs/2606.06036

[50] L. Zheng, W.-L. Chiang, Y. Sheng, S. Zhuang, Z. Wu, Y. Zhuang, Z. Lin, Z. Li, D. Li, E. P. Xing, H. Zhang, J. E. Gonzalez, and I. Stoica, “Judging LLM-as-a-judge with MT-Bench and chatbot arena,” in Advances in Neural Information Processing Systems, vol. 36, 2023. [Online]. Available: https://arxiv.org/abs/2306.05685

[51] Y. Zha, Y. Yang, R. Li, and Z. Hu, “Alignscore: Evaluating factual consistency with a unified alignment function,” in Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2023, pp. 11 328–11 348. [Online]. Available: https://aclanthology.org/2023.acl-long.634

[52] K. Järvelin and J. Kekäläinen, “Cumulated gain-based evaluation of IR techniques,” ACM Transactions on Information Systems, vol. 20, no. 4, pp. 422–446, 2002.

[53] S. Xiao, Z. Liu, P. Zhang, N. Muennighoff, D. Lian, and J.-Y. Nie, “C-pack: Packed resources for general chinese embeddings,” 2024. [Online]. Available: https://arxiv.org/abs/2309.07597

[54] E. Pakhomov, E. Nijkamp, and C. Xiong, “Convomem benchmark: Why your first 150 conversations don’t need rag,” 2025. [Online]. Available: https://arxiv.org/abs/2511.10523

[55] Y. Zhang, M. Li, D. Long, X. Zhang, H. Lin, B. Yang et al., “Qwen3 embedding: Advancing text embedding and reranking through foundation models,” 2025. [Online]. Available: https://arxiv.org/abs/2506.05176

## APPENDIX

## I. IMPLEMENTATION DETAILS

## A. Implementation Settings

1) Memory construction: In the stage of memory construction, dialogue histories are first split at original session boundaries and then scanned within each session using fixed size sliding windows. Unless otherwise specified, we use a window size of 6 turns and an overlap of 2 turns, resulting in a step size of 4 turns. Overlap turns are explicitly marked in the extraction prompt and are used only as background context for resolving pronouns and local references; the extractor is instructed not to create memory units whose evidence lies only in the overlap. The extractor may generate multiple memory units from the same window when different facts, preferences, events, or state changes are independently retrievable. BGEbase-en-v1.5 [53] is then used to embed memory-unit labels and graph records for retrieval and graph construction.

2) Retrieval and generation: For all LLM-driven steps, we set the decoding temperature to 0. During retrieval, the candidate pool size is set to 100 after RRF fusion over lexical, semantic, and event-attribute retrieval results, and reranking selects the best-performing budget of 10 memory units as working memory. For entity and location matching in the eventattribute retriever, dense semantic matching uses a similarity threshold of 0.60. When memory evolution is enabled, relation clustering uses a relation similarity threshold of 0.80 and a minimum relation-cluster size of 2.

## B. Training Settings

Training data are constructed from ConvoMem [54], a conversational memory benchmark built from multi-turn dialogue histories. We draw examples from three deterministicanswer categories: user\_evidence, assistant\_facts\_evidence, and changing\_evidence, which respectively test user fact recall, assistant fact recall, and updated information tracking. These categories cover memory capabilities aligned with LoCoMo and LongMemEval-S, such as user/assistant information recall, temporal grounding, and knowledge update. We sample the three categories using a fixed 5:3:4 ratio, yielding 1,200 cases and 2,138 QA pairs.

For each case, we run the full QGMem pipeline and record the trajectory information, including the activated memory units, their serialized textual content, the local memory graph, the generated answer, and the reasoning path. We retain only cases with correct final answers, resulting in 1,634 judgecorrect graph-grounded QA examples. We split the retained examples into train/validation/test partitions, and checkpoint selection is based only on the validation split. No LoCoMo or LongMemEval-S examples are used in this part. The retained ConvoMem examples and their trajectory information are used as supervision for training the memory graph token. Node and edge labels are represented with 768-dimensional BGE-baseen-v1.5 embeddings. The target is a JSON sequence containing answer and reasoning\_path fields.

TABLE V  
TRAINING HYPERPARAMETERS FOR THE MEMORY GRAPH TOKEN.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Backbone LLM</td><td>Qwen3-8B, Llama3.1-8B-Instruct</td></tr><tr><td>LoRA</td><td>Disabled</td></tr><tr><td>Trainable modules</td><td>Graph encoder, projector</td></tr><tr><td>Graph encoder</td><td>4 TransformerConv layers</td></tr><tr><td>Hidden size</td><td>768</td></tr><tr><td>Attention heads</td><td>4</td></tr><tr><td>Pooling</td><td>Mean pooling</td></tr><tr><td>Projector</td><td>768 → 2048 → 4096</td></tr><tr><td>Batch size</td><td>4</td></tr><tr><td>Gradient accumulation</td><td>4</td></tr><tr><td>Learning rate</td><td> $1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Weight decay</td><td>0.05</td></tr><tr><td>Training epochs</td><td>10</td></tr><tr><td>Warmup epochs</td><td>1</td></tr></table>

## C. Evidence Retrieval Quality Evaluation Protocol

Since different memory systems use different memory granularities, we evaluate retrieval quality at the gold-evidence level rather than by exact text overlap. We use the full LoCoMo evaluation set under the Qwen3-8B setting and compare QGMEM with the complete LoCoMo baseline suite: RAG, Mem0 [23], LightMem [24], EMem-G [7], LiCoMemory [28], MR-Agent [49], and AriadneMem [39]. All systems use the full evaluation set and the same retrieval cutoff $\ell = 1 0$ . For each question, let ${ \mathcal { E } } ^ { \star }$ denote the gold-evidence set and $\hat { \mathcal { M } } _ { \ell }$ denote the top-ℓ retrieved memory units. A GPT-4o-mini judge identifies the covered subset Covered $( \widehat { \mathcal { M } } _ { \ell } , \mathcal { E } ^ { \star } ) \ \subseteq \mathcal { E } ^ { \star }$ , the relevant subset Relevant $( \widehat { \mathcal { M } } _ { \ell } , \mathcal { E } ^ { \star } ) \subseteq \widehat { \mathcal { M } } _ { \ell }$ , and whether the retrieved set contains sufficient answer-supporting evidence. We compute:

$$
{ \mathrm { R e c a l l } } @ \ell = { \frac { | { \mathrm { C o v e r e d } } ( { \widehat { \mathcal { M } } } _ { \ell } , \xi ^ { \star } ) | } { | { \mathcal { E } } ^ { \star } | } } ,\tag{9}
$$

$$
\mathrm { P r e c i s i o n @ } \ell = \frac { | \mathrm { R e l e v a n t } ( \widehat { \mathcal { M } } _ { \ell } , \mathcal { E } ^ { \star } ) | } { \ell } ,\tag{10}
$$

and Suficiency@ℓ = 1 if the retrieved set as a whole contains sufficient answer-supporting evidence to answer the question, and 0 otherwise. Unlike a conventional hit indicator, which is satisfied by any one relevant retrieval, sufficiency requires the combined retrieved evidence to cover the information needed for the answer. To measure ranking quality, let $y _ { i } \in \{ 0 , 1 \}$ indicate whether the memory unit at rank i is relevant, and let IDCG@ℓ denote the DCG of the ideal ordering for the same number of relevant retrieved memories. We compute

$$
\mathrm { n D C G @ } \ell = \frac { 1 } { \mathrm { I D C G @ } \ell } \sum _ { i = 1 } ^ { \ell } \frac { y _ { i } } { \log _ { 2 } ( i + 1 ) } ,\tag{11}
$$

Recall is micro-averaged over gold items. Precision@10 uses the fixed denominator 10N across N questions, treating unfilled ranks as non-relevant, while Sufficiency@10 and nDCG@10 are averaged over questions. All scores in the evidence-retrieval table of the main paper are reported as percentages. Gold evidence and judge labels are used only for evaluation and are unavailable to all retrieval systems.

TABLE VI  
ABLATION OF RETRIEVAL STEPS ON LOCOMO WITH QWEN3-8B.
<table><tr><td>Setting</td><td>Avg. Score</td></tr><tr><td>Full activation pipeline</td><td>82.77</td></tr><tr><td>w/o Hybrid memory retrieval</td><td>81.93</td></tr><tr><td>w/o Event-attribute retrieval</td><td>71.67</td></tr><tr><td>w/o reranking</td><td>73.82</td></tr></table>

## D. Experiment Details

1) Experimental Settings: To ensure comparability, all systems use the same answer format, LLM-as-judge setting, and number of retrieved entries under each benchmark. Semantic retrieval uses BGE-base-en-v1.5 [53], and reranking uses Qwen3-Reranker-8B [55].

## E. Additional Ablation Studies

1) Retrieval and Selection Ablation: Table VI ablates the retrieval and selection choices in working memory activation on LoCoMo with Qwen3-8B. Removing reranking leads to the largest drop, showing that semantic refinement after hybrid memory retrieval is important for selecting memory units containing answer-supporting evidence. Removing event-attribute retrieval also degrades performance, confirming that explicit entity, location, and time constraints help rank candidates matching the event attributes ahead of topically related but mismatched ones. Removing hybrid memory retrieval has a smaller but consistent effect, suggesting that the initial candidate pool improves recall before reranking.

## II. PROMPTS

## F. LLM-AS-JUDGE

## 1) LoCoMo Judge Prompt:

## LoCoMo Judge Prompt

<table><tr><td>You are an evaluator for long-term conversational question answering. Given a question, a reference answer, and a model prediction, determine whether the prediction correctly answers the question. A prediction should be judged as correct if it is semantically equivalent to the reference answer, even if the wording is different. Minor paraphrases are acceptable. A prediction should be judged as incorrect if it contradicts the reference answer, misses a key entity, event, location, time, or relation, gives an answer that is too vague, or contains unsupported information.</td></tr><tr><td>Question: {question} Reference answer: {reference_answer} Model prediction: {model_prediction } Return only a JSON object with two fields: {&quot;judgment&quot;: &quot;correct&quot; or &quot;incorrect&quot;, &quot;reason&quot;: &quot;brief</td></tr></table>

## 2) LongMemEval-S Judge Prompt:

## Single-session-user/assistant Multi-session

I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no.

Question: {question}

Correct Answer: {answer}

Model Response: {response}

Is the model response correct? Answer yes or no only.

## Temporal Reasoning Tasks

I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model’s response is still correct.   
Question: {question}   
Correct Answer: {answer}   
Model Response: {response}   
Is the model response correct? Answer yes or no only.

## Knowledge Update Tasks

I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.   
Question: {question}   
Correct Answer: {answer}   
Model Response: {response}   
Is the model response correct? Answer yes or no only.

## Single-session Preference Tasks

I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user’s personal information correctly.

Question: {question}

Rubric: {answer}

Model Response: {response}

Is the model response correct? Answer yes or no only.

## Abstention Tasks

I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.

Question: {question}

Explanation: {answer}

Model Response: {response}

Does the model correctly identify the question as unanswerable? Answer yes or no only.

1) Prompts for QGMem: The following templates follow the current implementation. Atomic memory construction has two LLM stages. The first stage segments an annotated sliding window and returns one source-grounded claim and one canonical triple per memory unit. The second stage receives those memory units in a batch and returns retrieval indexes keyed by the unchanged memory\_unit\_id. Event and activity terms are included in entities; the current implementation does not use a separate event-index field. The final-answer variable {serialized\_graph} contains the complete textual generation context: evidence clusters with labeled memory-unit text and trajectories, followed by labeled structured facts. For open-weight backbones, the learned graph token is injected at the embedding level and therefore has no textual prompt placeholder.

## Atomic Memory Unit Extraction Prompt

```ini
[System]
You are a long−conversation memory−unit extractor.
# Input
− MEMORY_CONTENT contains a fixed−size sliding window of
dialogue lines in this form:
NEW_IN_THIS_WINDOW [source_id] time: <timestamp> speaker:
<message>
OVERLAP_FROM_PREVIOUS_WINDOW [source_id] time:
<timestamp> speaker: <message>
OVERLAP_FROM_PREVIOUS_WINDOW lines were already
included in the previous sliding window. Do not extract memory
units from these turns; they serve only as background
information for resolving references and ambiguity for
NEW_IN_THIS_WINDOW turns.
− NEW_IN_THIS_WINDOW lines are new turns in the current sliding
window.
# Task
Segment the current sliding window into atomic, self−contained
memory units. Each memory unit should express exactly one
independently retrievable claim: one event, state, plan,
preference, or other information.
− If the same turn supports multiple independent claims, output
multiple memory units with the same start_turn/end_turn when
needed. A unit may cover multiple turns only when those turns
are needed to make the single claim complete.
## Evidence bound
− Each memory unit MUST include start_turn and end_turn chosen
only from source_ids shown in MEMORY_CONTENT.
## Memory−unit fact_text
− fact_text is compact answer evidence. It must be specific enough to
answer a later question without reading MEMORY_CONTENT.
− Resolve pronouns and speaker−relative references into the most
specific supported entity, role, object, or group. Prefer real names
when available; use ‘the user‘ or ‘the assistant‘ only when no
more specific identity is given.
## Triple extraction
− Output one ‘triple‘ for the core claim of each memory unit. It must
be directly supported by the unit and include subject, predicate,
and object.
# Output format
Return valid JSON only. If there is no retrievable factual content with
non−overlap evidence, return {"memory_units": []}.
"memory_units": [
1
"start_turn": "source id from MEMORY_CONTENT",
"end_turn": "source id from MEMORY_CONTENT",
"fact_text": "standalone reusable memory−unit text",
"triple": {
"subject": "...",
"predicate": "...",
"object": "..."
[User template]
Construct reusable memory units from this fixed sliding−window
dialogue segment.
Session id: {session_id}
Turn range: {start_turn}−{end_turn}
Utterance time: {utterance_time}
MEMORY_CONTENT:
{annotated_memory_content}
```

Output schema:   
{   
"memory\_units": [   
{   
"start\_turn":   
"end\_turn":   
"fact\_text":   
"triple": {   
"subject":   
"predicate":   
"object": "..."   
}   
}   
]   
}

## Memory-Unit Retrieval Index Prompt

```ini
[System]
You extract retrieval indexes for memory−unit candidates from one
dialogue segment.
Input contains a list of memory units. Each unit has memory_unit_id,
utterance_time, fact_text, and evidence_text.
For each memory unit:
1) Extract entities: explicit people, named entities, events, activities, or
concrete object/topic keywords needed to retrieve this unit and
distinguish it from other memories.
2) Extract locations: concrete places, venues, facilities, cities, countries,
or regions where the fact happened. Do not put events or
activities in locations unless the text names a physical venue.
3) Extract time: when the described fact happened. Each item is
{"start": "...", "end": "...", "granularity": "day|week|month|year"}.
Use [] for unknown or vague recency.
Time format:
− For an exact date, set start=end and use the matching granularity.
− For a relative but bounded time, such as "last weekend" or "two
weeks before DATE", output a conservative date range rather
than a guessed day.
Distinguish utterance_time and fact time. Resolve relative
expressions against the memory unit’s own utterance_time, and
do not copy utterance_time itself as the fact time.
− For vague times such as "recently", "lately", "a while ago", or "the
other day", output [].
Return valid JSON only:
"memory_units": [
{
"memory_unit_id": "...",
"entities": ["..."],
"locations": ["..."],
"time": [
{
"start": "YYYY−MM−DD|YYYY−MM|YYYY",
"end": "YYYY−MM−DD|YYYY−MM|YYYY",
"granularity": "day|week|month|year"
]
}
[User template]
Extract retrieval indexes for these memory units from one dialogue
segment.
Session id: {session_id}
Turn range: {start_turn}−{end_turn}
Segment utterance time: {utterance_time}
MEMORY_UNITS:
```

Dynamic Memory Trace Consolidation Prompt   
[System]   
You are responsible for turning candidate atomic memories into one   
concise memory trajectory. Several memory units may describe   
the same subject−relation, and later memories can update, refine,   
repeat, conflict with, or add to earlier memories. The input items   
are candidate members of one relation cluster, but some may be   
irrelevant or only loosely related.   
Write one chronological trajectory supported by the memory units:   
− keep only items that describe the same evolving subject−relation,   
− order the relevant states by memory time,   
− include both earlier history and the latest/current supported state.   
Use memory order, not world knowledge:   
− If time values are ‘serial−N‘, larger N means newer memory.   
− If a later memory contradicts an earlier one, describe the later   
memory as an update.   
Return only valid JSON:   
"trajectory": "compact chronological trajectory including earlier   
state(s) and the latest supported state"   
[User template]   
Write one memory trajectory for the following relation cluster.   
RELATION:   
{relation\_text}   
MEMORY\_UNITS are sorted from oldest to newest by memory time.   
Write the trajectory from earlier memory to the latest supported   
memory, including updates and non−conflicting additions.   
{members\_json}

```jsonl
− Do not put events or activities in ‘locations‘; keep them in ‘entities‘
when they constrain evidence.
# Time
− ‘time.fact‘ is for when the remembered fact happened, and
‘time.utterance‘ is only for when the conversation happened,
such as "in our chat on June 3" or "as mentioned last Tuesday".
− Normalize clear dates as date−only values: ‘YYYY−MM−DD‘ for a
day or ‘YYYY−MM−DD..YYYY−MM−DD‘ for an interval.
[User template]
Parse this query into retrieval−oriented intent.
Query:
{query}
Question time:
{question_time}
Use Question time only as the reference for explicit relative time
expressions in the query. Do not output it as a time constraint by
default.
Output schema:
"entities": ["explicit person, named entity, or topic/object constraint"],
"locations": ["concrete place or venue constraint stated in the query"],
"time": {
"fact": ["YYYY−MM−DD or YYYY−MM−DD..YYYY−MM−DD
for when the remembered fact happened"],
"utterance": ["YYYY−MM−DD or
YYYY−MM−DD..YYYY−MM−DD only when the query
explicitly constrains when the conversation happened"]
}
```

"memory\_unit\_id":   
"utterance\_time":   
"fact\_text":   
"evidence\_text":   
}   
]

## Query Intent Parsing Prompt

## [System]

You parse a question into retrieval constraints for a long−term memory store.

## # Background

− Your output is used to select candidate memories before answer generation.

− The extracted values are used for semantic retrieval, so they should be discriminative rather than general.

## # Task

− Read the question and extract retrieval constraints on three axes: ‘entities‘, ‘locations‘, and ‘time‘.

− If an axis has no useful constraint, return an empty list for that axis.

− Extract people, named objects, events, activities, roles, organizations, artifacts, and concrete topic words in ‘entities‘.

− Extract physical or named places, such as Paris, Pacific Northwest, school, gym, museum, beach, or library, in ‘locations‘.

− Do not put answer types in ‘locations‘. For example, "countries", "states", "places", or "where" describe what the answer should find.

## Final Answer Prompt

## [System]

You answer questions using only the provided memory context.

− QUESTION is the user’s question.

− QUESTION\_TIME, when provided, is the time the question was asked. Use it only to resolve relative time expressions in the question.

− MEMORY\_CONTEXT includes:

− Evidence Clusters: retrieved memories grouped by memory−evolution relation. A cluster includes a ‘trajectory‘ field describing earlier history and the latest state, together with labeled memory−unit text.

− Facts: labeled structured subject−predicate−object facts.

− Memory−unit ‘utterance\_time‘ is when the dialogue happened; if memory−unit times use ‘serial−N‘, larger N means newer evidence.

## # Task

− Produce the best supported answer to QUESTION.

− Ground the answer only in MEMORY\_CONTEXT.

− When a trajectory’s latest state conflicts with older content or older Facts from the same relation, use the trajectory’s latest state as the latest memory−supported evidence.

## # Output Format

Think step by step internally, then return valid JSON only:

"reasoning\_path": "brief chain of reasoning citing labels (MU#/F#) used",

Question time:

{question\_time}

Graph:

{serialized\_graph}

Return only valid JSON with keys reasoning\_path and answer, in that order.