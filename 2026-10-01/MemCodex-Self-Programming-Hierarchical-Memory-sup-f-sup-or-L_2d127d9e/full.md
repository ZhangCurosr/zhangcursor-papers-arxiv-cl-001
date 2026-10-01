# MemCodex: Self-Programming Hierarchical Memory <sup>f</sup>or Language Agents

Xiao<sub>q</sub>ian<sub>g</sub> Wan<sub>g</sub><sup>1,2</sup><sub>,</sub> Ban<sub>g</sub> Liu<sup>1,2</sup>

<sup>1</sup>Université de Montréal <sup>2</sup>Mila – Quebec AI Institute

Agent memory faces heterogeneous access needs: a single-hop question may require one piece of evidence, whereas a multi-hop question must combine evidence from multiple sources. Predefined memory workflows cannot adapt to these varying needs. Recent adaptive methods search or learn over memory components and their compositions, but the design space itself remains predefined. We introduce MemCodex, a self-evolving hierarchical memory system that organizes experience into executable memory programs for summaries, relational knowledge, reusable skills, and latent memory. Open-ended program evolution searches the open design space of layer programs by rewriting how each layer is constructed, indexed, retrieved, and routed, thereby adapting both within-layer implementations and cross-layer composition. At query time, reads traverse the hierarchy from coarse to fine and stop once suficient evidence is found, descending to the original history when needed. We further develop MemArena, a unified runtime that places heterogeneous data and memory systems behind a common interface. MemCodex improves average task success by 10.1% relative to the strongest adaptive-memory baseline, with 3.4× fewer context tokens and 2.1× faster inference.

{ Date: October 1, 2026 # Correspondence: bang.liu@umontreal.ca

Université m de Montréal

![](images/c0dfb9bb148a0e81aea9d975c7b6779c83889eaf776b56734f63a0cf0c88f8b2.jpg)

![](images/b5d8cefbcbe981024ff5f2e8eb5c9a995ca606ba30e044c58c3fe6d9431aed18.jpg)

![](images/4ef0522499849587189439060de6e5a882e4f8c6e45404ea345688d91d1d2222.jpg)  
Figure 1 Memory systems by memory organization and design search (left), and average test accuracy against latency per query (right).

## 1 Introduction

Language agents (Yao et al., 2022; Wang et al., 2024a; Liu et al., 2025; Ding et al., 2026) increasingly act over long horizons in web navigation (Zhou et al., 2023), software engineering (Jimenez et al., 2023; Hong et al., 2023) and computer use (Xie et al., 2024; Wang and Liu, 2025), and selfevolve from experience (Tao et al., 2024) through reflection on failures (Shinn et al., 2023), skill accumulation (Wang et al., 2023; Zhao et al., 2024; Ouyang et al., 2025; Shi et al., 2026) and code rewriting (Zhang et al., 2025b). Across sessions, much of this experience survives only in memory. Hand-designed pipelines organize it through rolling summaries in systems such as MemoryBank and MemGPT (Zhong et al., 2024; Packer et al., 2023), facts and entity graphs (Xu et al., 2025; Rasmussen et al., 2025; Edge et al., 2024), skill stores (Zhao et al., 2024; Ouyang et al., 2025; Wang et al., 2023), or memory written into parameters or latent state (Wang et al., 2024c; Behrouz et al., 2024; Li et al., 2025a; Wang et al., 2025a). Yet these pipelines face access heterogeneity, because the memory organization a query needs depends on the query, the record and the task (Xu et al., 2026; Lu et al., 2026; Huang et al., 2026). This heterogeneity appears across query types, since single-hop questions need raw turns (Han et al., 2026), multi-hop questions need cross-session links (Gutiérrez et al., 2024), temporal questions need dated events (Ge et al., 2025), and repeated procedures need skills (Wang et al., 2025c). Adaptive computation in reasoning shows the same pattern, where dynamic shortcuts spend depth only on the steps that need it (Wang et al., 2026b). A hand-designed pipeline hard-codes both which organizations exist and the rule that chooses among them, and a study of fourteen memory systems (Zhou et al., 2026) finds that no architecture dominates across tasks.

Adaptive memory addresses this rigidity in two main directions. One direction searches a predefined design space by selecting and composing modules for encoding, storage, retrieval, and management (Du et al., 2026; Lu et al., 2026; Zhang et al., 2025a). The other, exemplified by Mem-alpha, learns policies for when and how to write, update, and retrieve memory (Wang et al., 2025b, 2026a; Sun et al., 2026; Wang et al., 2026c). Despite their diferent forms, both directions must jointly determine how experience is organized into memory and how queries access it, since retrieval can only use structures created during writing. As shown in Figure 1, this coupling places memory systems along two axes: memory organization and design search. Among prior systems, only MemPro (Liu et al., 2026d) opens the design itself to program search, but it evolves a single monolithic program over flat memory, without a hierarchy whose components can evolve and be accessed independently.

We therefore introduce MemCodex, a hierarchical memory system whose layers are executable memory programs behind a fixed interface and runtime. Program evolution adapts both the write and read code of individual layers and their cross-layer composition, that is, which layers are active and how items and reads pass between them. Within this hierarchy, MemCodex organizes skill, graph, and summary above an append-only raw layer, with kv memory providing latent memory.

Inspired by the Darwin Gödel Machine (Zhang et al., 2025b) and HyperAgents (Zhang et al., 2026b), which evolve an agent by rewriting its own code, we apply program evolution to this memory hierarchy. At each round, a meta agent draws a parent from the archive and rewrites one method of one layer or toggles one layer. Restricting each child to one change makes the efect of that change attributable. An acceptance test then keeps changes that improve accuracy or reduce token use without degrading the other, and accuracy-only gains must additionally pass a one-sided McNemar test (McNemar, 1947). Before this test, validity checks K reject changes that break the memory constraints.

The fixed interface exposes five methods in each layer, spanning both memory writing and reading. On the write side, admit filters candidates, index keeps a bounded active set, and propose derives items for the layer above. On the read side, score ranks candidates and route decides where the search proceeds next. Reads move from coarse to fine layers, with route choosing Stop, Narrow, or Descend until suficient evidence is found or the search reaches the raw history. Because every stored item retains its exact source turns, the original evidence remains recoverable even after lossy summarization.

![](images/7881cb24669107c400bc438a0e54a7539b6354711fc7e5ad9f54c8b44f6615ba.jpg)  
Figure 2 MemCodex workflow. In the task loop (upper), MemArena writes the record and poses each query � to the task agent, which reads a context � from the layer hierarchy, stopping at the coarsest suficient layer or descending to source turns, and answers �. In the program evolution loop (lower), run once per round, a parent is sampled from the archive of architectures $g = ( \lambda , \psi )$ , meta agent $\rho$ rewrites one method of one layer program $\psi _ { \ell ; \ell }$ , and validity checks and the acceptance test decide whether the child enters the archive.

To evaluate these mechanisms systematically across diverse sources, interaction formats, and evaluation protocols, we further develop MemArena, a heterogeneous agent-source evaluation suite with a unified data interface, access interface, and metric suite covering benchmarks such as LoCoMo (Maharana et al., 2024), LongMemEval (LME) (Wu et al., 2025), and MemoryAgentBench (MAB) (Hu et al., 2025b). It separates the memory system from the task agent, as capability-level evaluation dissociates what a language model knows from how it reasons (Wang et al., 2024b). Under one protocol, MemArena spans seven sources, eleven baseline systems and two backbones, while each new source requires only one adapter. Under this evaluation, MemCodex gains 5.6 average accuracy points over the strongest baseline with Qwen3.5-27B (Qwen Team, 2026) and 3.6 with the API model GPT-5.4-mini, while reading 3.4× fewer tokens and answering 2.1× faster.

## 2 MemCodex: Hierarchical Memory as Evolving Programs

We model agent memory as a hierarchy of executable memory programs whose implementations and composition evolve with the task (Figure 2). A fixed five-method layer interface and runtime constrain how these programs interact while leaving their internal behavior open to evolution.

We model long-horizon memory as a task agent � answering questions about a record � that arrives as write chunks $R _ { 1 } , \ldots , R _ { n }$ of timestamped turns $\nu ,$ where each question $q$ is posed once the record up to its question time has been written. A memory system with architecture $g$ keeps a store M and serves � through a write operation and a read operation:

$$
M \gets \mathsf { w r i t e } _ { g } ( \mathcal { M } , R _ { i } ) , \qquad C = \mathsf { r e a d } _ { g } ( \mathcal { M } , q ) , \qquad a = \pi ( q , C ) .\tag{1}
$$

The write operation folds each chunk into the store, the read operation returns a context � of at most � tokens, and � answers from the question and that context. To preserve exact evidence despite this bounded context, the store contains an append-only raw layer $_ \textmd { ‰}$ of all turns and lossy derived items � above $\operatorname { i t } ,$ each of which retains its exact source turns src $\mathbf { \chi } ( x ) \subseteq \mathcal { V }$

We represent each architecture as $g = ( \lambda , \psi )$ with two parts. The layer programs $\psi$ implement the write and the read through five methods per layer. Specifically, the write operation uses admit, index and propose, while the read operation uses score and route. The layer mask � independently switches derived layers on or of. For the questions $\boldsymbol { Q }$ of a task, each query $q$ has gold turns $G ( q ) \subseteq { \mathcal { V } }$ The accuracy of $g$ is $\operatorname { a c c } ( g ) = \mathbb { E } _ { q \sim Q } { \bf 1 } [ \pi ( q , C _ { g } ( q ) )$ correct] and its cost is tok $/ { \sf q } ( g ) = \mathbb { E } _ { \boldsymbol { q } \sim Q } | C _ { g } ( \boldsymbol { q } ) |$ . Our objective is the most accurate architecture at no higher token cost than $g _ { 0 }$ . We first define the layer programs $\psi$ and the read and write operations they implement. We then define program evolution, which changes $g$ one method rewrite or one layer toggle at a time and keeps accepted architectures in an archive.

## 2.1 La<sub>y</sub>er Pro<sub>g</sub>rams $\psi$

Layers. Queries require diferent levels of abstraction, but every derived item must remain connected to the raw evidence from which it was produced. We therefore order the four text layers $\mathcal { L } \ =$ (skill, graph, summary, raw) from coarse to fine, as in RAPTOR (Sarthi et al., 2024), with store $M = { \mathcal { V } } \cup \bigcup _ { \ell \neq \mathrm { R A W } } M _ { \ell }$ , where $M _ { \ell }$ holds the derived items of layer ℓ. The raw layer keeps every raw turn $\upsilon \in ^ { \mathfrak { N } }$ unchanged so that the original evidence remains available. The summary layer condenses each closed segment of raw turns into a compact account of that segment. The graph layer stores timed assertions $x = \left( h , r , t , \tau \right)$ extracted from summaries, where entities ℎ and � are linked by relation � over the time � for which the relation holds, so that a newer assertion can supersede an older one about the same entities. The skill layer stores executable procedures distilled from assertions that record the same procedure in several sessions. We model a skill as a program rather than a text note because a procedure that runs can be checked against the sessions that demonstrate it. Each derived item � is built from inputs in(�) in the layer directly below, and its exact source turns src(�) resolve recursively,

![](images/54f221df85ffd9bc916e281a520fc38fce006b2cb1b788f873b8dc7886be09c2.jpg)  
Figure 3 Content and routing channels of MemCodex for a question about Caroline. $\mathsf { C } _ { \mathrm { t x t } }$ places summary � in �, omitting Paris, while $\mathsf { C } _ { \mathrm { r o u t e } }$ follows src(�) to the raw turn naming Paris.

$$
\begin{array} { r } { \operatorname { s r c } ( \nu ) = \{ \nu \} ( \nu \in \mathcal { V } ) , \qquad \operatorname { s r c } ( x ) = \bigcup _ { y \in \mathrm { i n } ( x ) } \operatorname { s r c } ( y ) ( x \in M \backslash \mathcal { V } ) . } \end{array}\tag{2}
$$

A read can therefore reach the raw turns underlying an assertion or a skill without traversing its derivation. The set src(�) records the exact derivation of an item rather than a minimal evidence set. In particular, a summary and its extracted assertions share the segment’s turns, which Narrow and the raw-layer read subsequently refine.

Layer interface. Local program evolution requires every method rewrite to remain within its target layer. We therefore expose each text layer through five methods with distinct functionalities. Within a layer, admit<sub>ℓ</sub> filters proposals, while index<sub>ℓ</sub> retains at most $K _ { \ell }$ items in the active index $\begin{array} { r } { \mathcal { I } _ { \ell } . } \end{array}$ . Evicted items remain in the store. The method score $\ell ( q , S )$ ranks the items � allowed by the search scope $F \subseteq \mathcal { V }$ and returns the top candidate $x ^ { \star }$ with confidence �. Across layers, propose sends derived items upward, while route $\cdot \ell \left( q , x ^ { \star } , c \right)$ returns Stop, Narrow or Descend. The runtime owns assemble, which packs � under �. In every architecture, turns are appended only to the raw layer and never changed. Every derived item names the exact raw turns used to build it. Narrow can only restrict the search scope to the top candidate’s source turns, $F \cap \operatorname { s r c } ( x ^ { \star } )$ . Every context must also fit within � tokens. These four memory constraints ensure that any read can trace a derived item back to raw turns, however lossy the item is.

Content channel and routing channel. A derived item can serve a query through two distinct channels, and the query-adaptive read uses the routing channel. The channel is an architecture setting rather than a per-query choice, so only the read depth adapts to each query. In the content channel $\mathsf { C } _ { \mathrm { t x t : } }$ , the text of the top-scoring items enters � in place of their evidence, as in summary, profile and skill memories (Packer et al., 2023; Chhikara et al., 2025; Zhao et al., 2024). In the routing channel $\mathsf { C } _ { \mathrm { r o u t e } }$ , a retrieval hit leads to its source turns, as HippoRAG passes node probability to source passages (Gutiérrez et al., 2024). In the Caroline example of Figure 3, the summary omits Paris, but routing through src(�) recovers the raw turn that names it. Evidence preservation, retaining every derived item’s exact source turns, keeps this route available.

Query-adaptive read. A fixed read depth can either skip useful abstractions or retrieve unnecessary detail. The $\boldsymbol { \Gamma } \boldsymbol { \Theta } \mathsf { a d } _ { g }$ operation in Equation (1) therefore visits layers from coarse to fine (Algorithm 1) and lets each route determine whether traversal should continue. Only Stop ends the read and adds the in-scope source turns $\operatorname { s r c } ( x ^ { \star } ) \cap F$ to � in temporal order through the routing channel. The Narrow action restricts the search scope to those source turns, while Descend preserves the current scope, and both actions continue to the next layer. At the raw layer, the � highest-scoring in-scope turns enter �. The routing channel therefore ensures that the task agent always answers from raw turns, even when the read stops at a skill whose source turns are the sessions that demonstrate the procedure. Each query consequently incurs only the read depth it needs.

Algorithm 1 QueryAdaptiveRead Algorithm 2 ProgramEvolution   
Input: query $q ,$ store M with an active index $\textstyle { \mathcal { T } } _ { \ell }$ per Input: initial architecture $g _ { 0 } ,$ , evolve questions $Q ,$   
layer, architecture $g$ with active layers $\mathcal { L } _ { g } \left( \lambda \right) =$ meta agent $\rho ,$ validity checks $\mathcal { K } ,$ rounds $T ,$   
$\mathbf { 1 } ) _ { : }$ , budget $B ,$ channel $\chi \in \{ \mathsf { C } _ { \mathrm { t x t } } , \mathsf { C } _ { \mathrm { r o u t e } } \} ,$ , and parents $P ,$ children per parent �   
$\mathrm { t o p } _ { k }$ the first � leaves by ${ \tt s c o r e } _ { \tt R A W }$ Output: archive A and the evolved architecture $g ^ { \star }$   
Output: context � 1: ${ \mathcal { A } } \gets \{ g _ { 0 } \} , \ E _ { g _ { 0 } } \gets \mathrm { E v a L U A T E } ( g _ { 0 } , Q )$   
1: $F  \mathcal { V }$ ⊲ every raw turn 2: for $r = 1 , \ldots , T$ do   
2: for $\ell \in \mathcal { L } _ { g } \ \backslash \ \left\{ \mathbf { R A W } \right\}$ , coarsest first do 3: for all $n \sim w , j \le H ,$ , in parallel do   
3: $S \gets \{ x \in { \mathcal { I } } _ { \ell } : \operatorname { s r c } ( x ) \cap F \neq \emptyset \}$ 4: (ℓ, �) ← next target of $\mathcal { D } ( g _ { n } )$   
4: $X \gets \mathsf { s c o r e } _ { \ell } ( q , S )$ 5: if $u =$ toggle then $g ^ { \prime } \gets g _ { n } ,$ flip $\lambda _ { \ell }$   
5: if $X = \varnothing$ then continue 6: else $g ^ { \prime } \gets \rho ( g _ { n } , \mathcal { D } ( g _ { n } ) , \ell , u , \mathcal { T } _ { n } )$   
6: $( x ^ { \star } , c ) \gets X _ { 1 }$ ⊲ top candidate 7: if some � $\in$ K fails on $g ^ { \prime }$ then   
7: $o \gets \mathrm { r o u t e } _ { \ell } ( q , x ^ { \star } , c )$ ⊲ searched method 8: ${ \mathcal { T } } _ { n } \gets { \mathcal { T } } _ { n } \cup \{ ( g ^ { \prime } , \kappa ) \} ;$ ; continue   
8: if $o = S$ top then 9: $E _ { g ^ { \prime } } \gets \mathrm { E v a L U A T E } ( g ^ { \prime } , Q )$   
9: if $\chi = \mathsf C _ { \mathrm { t x t } }$ then $Y \gets \mathrm { t e x t } ( X _ { 1 \dots 1 0 } )$ 10: if $\mathsf { A c c e p t } ( g _ { n } \to g ^ { \prime } )$ then add $g ^ { \prime }$ under �   
10: else $Y \gets \mathrm { s r c } ( x ^ { \star } ) \cap F$ in time order 11: else ${ \mathcal { T } } _ { n } \gets { \mathcal { T } } _ { n } \cup \{ ( g ^ { \prime } , E _ { g ^ { \prime } } ) \}$   
11: return assembl ${ \bf \dot { \theta } } _ { B } ( Y )$ 12: return A, $g ^ { \star } \gets \arg \operatorname* { m a x } _ { g \in \mathcal { A } } \arg \ O \{ g \}$   
12: else if $o = \mathrm { N A l }$ rrow then   
13: $F \gets F \cap { \mathrm { s r c } } ( x ^ { \star } )$   
14: return assembl ${ \mathsf { \Omega } } _ { B } ^ { } ( \tan { \mathsf { p } } _ { k } ( q , F ) )$ ⊲ Stop at raw

Write path. Derived layers depend on lower-layer content, so the wri $\mathsf { t e } _ { g }$ operation in Equation (1) processes each appended chunk from bottom to top through propose, admit and index. Each proposal to a layer combines the items newly admitted one layer below for the current chunk with the active items of that lower layer. The runtime records every item read to form the proposal as an input. At skill, admit accepts a procedure only when replay on every demonstration session reproduces the recorded outcome. The task agent’s backbone writes the textual proposals. In the initial architecture �<sub>0</sub>, index retains items according to the access-and-recency heat of MemoryOS (Kang et al., 2025), score uses cosine similarity and route stops or narrows above per-layer thresholds.

Latent memory. Some associations in the record have no representation in any text layer. We feed each write chunk into Metis-27B (Zhang et al., 2026d), a memory foundation model with stateful fast weights, and use the state of its memory blocks as kv memory, a fixed-size key-value memory. kv memory is write-only, exposes no score or route, is built on the write path, and is not evolved. Two interfaces provide memory to the task agent. The natural language (NL) interface places the text context � in the prompt and supports any agent. The KV interface injects kv memory into an open-weight agent’s memory attention. Because an API agent exposes neither weights nor activations, it uses only the NL interface and has no latent memory.

## 2.2 Evolvin<sub>g</sub> the Architecture <sub>�</sub>

Architecture. Joint evolution of layer behavior and composition requires each edit to remain local to one component. The layer mask $\lambda \in \{ 0 , 1 \} ^ { | \mathcal { L } | - 1 }$ covers the derived layers, while the raw layer remains active. Reads visit only the active layers. Writes run through every layer up to the highest active one so that every proposal retains its type and derivation, including when an intermediate layer is switched of for reading. The layer mask and cross-layer methods determine composition, while the within-layer methods determine implementation. Starting from $g _ { 0 } .$ , program evolution combines evaluator-guided program search (Hu et al., 2025a; Romera-Paredes et al., 2024; Novikov et al., 2025), the archive-based open-ended search of the Darwin Gödel Machine (Zhang et al., 2025b) and the meta agent of HyperAgents (Zhang et al., 2026b). Meta agent $\rho$ runs on the task agent’s backbone, so the agent programs its own memory. Attributable edits restrict every child to one method rewrite of one layer or one layer toggle, so each accepted change traces to one edit. The acceptance test then accepts only improvements that do not trade accuracy against token cost.

Archive. A single chain of edits cannot recover after reaching a dead end. Following MemPro (Liu et al., 2026d), we therefore store accepted architectures in the archive ${ \mathcal { A } } ,$ which forms a tree, while rejected children enter their parent’s rejection log J . Each node stores evolve-question accuracy $\mathsf { a c c } _ { Q . }$ , tokens per question and verdicts ${ \bf v } _ { g } \in \{ 0 , 1 \} ^ { | Q | }$ . A parent is drawn with probability proportional to

$$
w ( n ) = \sigma { \Big ( } { \frac { \operatorname { a c c } _ { Q } ( g _ { n } ) - { \overline { { \operatorname { a c c } } } } } { \hat { \sigma } } } { \Big ) } \cdot { \frac { 1 } { 1 + | \operatorname { c h } ( n ) | } } \cdot ( 1 + d _ { n } ) ,\tag{3}
$$

where � is logistic, acc and �ˆ are the archive accuracy mean and spread, ch(�) the children of � and $d _ { n }$ its normalized crowding distance on the accuracy-cost Pareto front (zero of it), so the weight favors accurate, rarely expanded nodes on the front. Any archived architecture can become a parent, and a child may contain any code that satisfies the layer interface. Program evolution therefore remains open-ended rather than following a single chain of edits.

Self-modification. Each local edit should address an observed failure. The parent’s diagnosis $\mathcal { D } ( g )$ therefore examines failed traces and refused or evicted items, then assigns each failure to the first method that loses a gold turn. Children target successive methods or layer toggles according to fault count, so siblings make distinct edits. For a method rewrite, meta agent $\rho$ reads the target layer’s code, diagnosis and rejection log, then returns a subclass that overrides only that method (Algorithm 2). Validity checks K reject any child that violates a memory constraint on fixture queries.

Acceptance test. Validity checks determine whether a child preserves the memory constraints, but they do not determine whether it trades accuracy for token cost. The acceptance test therefore compares child and parent on the same evolve questions with the task agent and judge frozen, accepting the child exactly when

$$
\mathsf { A c c e p t } ( g \to g ^ { \prime } ) \iff \Delta _ { \mathrm { t o k } } \leq 0 \wedge \Delta _ { \mathrm { a c c } } \geq 0 \wedge \big [ ( \Delta _ { \mathrm { a c c } } > 0 \wedge p < \alpha ) \vee \Delta _ { \mathrm { t o k } } < 0 \big ] ,\tag{4}
$$

where $\Delta _ { \mathrm { a c c } }$ and $\Delta _ { \mathrm { t o k } }$ are child-minus-parent changes in accuracy and tokens per question, $p$ is the one-sided exact McNemar �-value (McNemar, 1947) on discordant pairs of ${ \bf v } _ { g }$ and $\mathbf { v } _ { g ^ { \prime } }$ , and $\alpha = 0 . 0 5$ The evolved architecture $g ^ { \star }$ is the archived architecture with the highest evolve-split accuracy.

## 3 MemArena: A Hetero<sub>g</sub>eneous A<sub>g</sub>ent-Source Evaluation Suite

Memory systems are reported on sources with diferent evidence formats, time representations and access protocols, so their results are dificult to compare directly. MemArena addresses this problem by placing heterogeneous sources and memory systems behind one data interface, one access interface and one metric suite, which allows any memory module to be evaluated with any task agent under a fixed backbone and judge. It currently spans seven sources, eleven baseline systems and two backbones. We release the suite together with its adapters and wrappers.

![](images/5763ef53cffab639025c65a8d01e34cb937d27e01a57a1b804958dc8b504c1d5.jpg)  
Figure 4 The MemArena evaluation suite, with heterogeneous sources behind unified data and access interfaces and a metric suite.

Data interface. Source adapters map corpora with diferent evidence and time representations into an episode $\boldsymbol { e } = \left( \boldsymbol { R } , \boldsymbol { Q } _ { e } \right)$ (Figure 4). The record � orders sessions of timestamped, speaker-tagged turns, while each tuple $( q , a ^ { \star } , G ( q ) , y _ { q } , t _ { q } ) \in Q _ { e }$ specifies a question, reference answer, gold turns, question type and question time, posed after the record through $t _ { q }$ is written. Source adapters obtain $G ( q )$ from evidence annotations for LoCoMo and LongMemEval and from task construction for PersonaMind and MAB. They cover seven sources (Maharana et al., 2024; Wu et al., 2025; Hu et al., 2025b; Jiang et al., 2025; Li et al., 2025b; Du et al., 2024; Zheng et al., 2025), and adding a source requires only a source adapter. PersonaMind, our fourth benchmark, poses five preference or procedure questions for each of 300 PersonaMem-V2 and InMind records, yielding 1,500 questions.

Access interface. To separate memory quality from task-agent quality, MemArena wraps memory systems and task agents independently. A memory wrapper exposes write $( M , R _ { i } ) \ :  \ : M ^ { \prime }$ and $\mathsf { r e a d } ( M , q ) \to C .$ , where � may carry kv memory. An agent wrapper separately exposes answer $( q , C ) \to a$ for task agent �, using the NL interface for text and, for open-weight agents, the KV interface for kv memory. The harness writes each record into a fresh store in 20-turn chunks and poses each question after the last chunk written before $t _ { q } ,$ so stores are never shared across records or users and only � is shared. This separation allows any task agent to be evaluated with any memory module. Within each comparison, the harness fixes the backbone and judge, so systems difer only in memory.

Metric suite. MemArena measures both task success and memory cost. Accuracy acc is the judge verdict on � against $a ^ { \star }$ under the Mem0 rubric (Chhikara et al., 2025). Where gold turns are annotated, evidence recall is $\mathsf { R e c } = \mathbb { E } _ { q } [ | G ( q ) \cap C | / | G ( q ) | ]$ . Context cost tok/q counts prompt tokens per question, and inference time measures wall-clock latency from read to the last answer token.

## 4 Ex<sub>p</sub>eriments

Benchmarks and splits. We evaluate every system through MemArena on LoCoMo (Maharana et al., 2024), LongMemEval (S split) (Wu et al., 2025), MemoryAgentBench (MAB) (Hu et al., 2025b) and PersonaMind. Following MemEvolve (Zhang et al., 2025a) and MemPro (Liu et al., 2026d), which evolve on a held-out slice of each benchmark, we take 20 percent of each benchmark’s questions as the evolve split �. We stratify � by question category, such as single-hop, multi-hop, temporal and open-domain questions on LoCoMo, the five abilities of LongMemEval, the four competencies of MAB, and preference against procedure questions on PersonaMind. We never read the test split during search.

Table 1 Quantitative results of MemCodex with Qwen3.5-27B and GPT-5.4-mini as task agents. Bold marks the best per column, underline marks the strongest baseline, and parentheses give MemCodex’s margin over it. Tok/q counts thousands of prompt tokens per question, and LME, MAB and PMind abbreviate LongMemEval, MemoryAgentBench and PersonaMind.
<table><tr><td rowspan="2">Method</td><td colspan="5">Qwen3.5-27B</td><td colspan="5">GPT-5.4-mini</td><td rowspan="2">Tok/q</td></tr><tr><td>LoCoMo</td><td>LME</td><td></td><td>E MAB PMind</td><td>Avg.</td><td>LoCoMo</td><td>LME MAB PMind</td><td></td><td></td><td>Avg.</td></tr><tr><td>Full context</td><td>61.8</td><td>57.6</td><td>43.9</td><td>51.2</td><td>53.6</td><td>64.1</td><td>59.8</td><td>46.2</td><td>53.3</td><td>55.9</td><td>21.4</td></tr><tr><td>Dense RAG</td><td>48.9</td><td>55.1</td><td>40.7</td><td>45.8</td><td>47.6</td><td>51.2</td><td>57.3</td><td>43.0</td><td>48.1</td><td>49.9</td><td>3.6</td></tr><tr><td>MemGPT</td><td>55.4</td><td>50.9</td><td>39.8</td><td>47.1</td><td>48.3</td><td>58.0</td><td>52.9</td><td>41.9</td><td>49.5</td><td>50.6</td><td>8.9</td></tr><tr><td>A-Mem</td><td>54.6</td><td>53.2</td><td>41.5</td><td>47.9</td><td>49.3</td><td>56.8</td><td>55.6</td><td>43.4</td><td>50.1</td><td>51.5</td><td>5.1</td></tr><tr><td>Mem0</td><td>56.4</td><td>54.1</td><td>42.2</td><td>48.8</td><td>50.4</td><td>58.7</td><td>56.4</td><td>44.5</td><td>51.1</td><td>52.7</td><td>4.2</td></tr><tr><td>MemoryOS</td><td>58.0</td><td>54.8</td><td>43.6</td><td>50.3</td><td>51.7</td><td>60.3</td><td>57.0</td><td>45.9</td><td>52.6</td><td>54.0</td><td>6.3</td></tr><tr><td>HMO</td><td>58.7</td><td>55.9</td><td>44.1</td><td>51.3</td><td>52.5</td><td>60.9</td><td>58.3</td><td>46.2</td><td>53.4</td><td>54.7</td><td>7.4</td></tr><tr><td>Mem-α</td><td>59.3</td><td>56.8</td><td>45.7</td><td>52.5</td><td>53.6</td><td>61.5</td><td>58.9</td><td>47.9</td><td>54.6</td><td>55.7</td><td>5.8</td></tr><tr><td>AutoMem</td><td>60.1</td><td>57.3</td><td>46.0</td><td>52.9</td><td>54.1</td><td>62.4</td><td>59.6</td><td>48.3</td><td>55.2</td><td>56.4</td><td>6.1</td></tr><tr><td>MemEvolve</td><td>60.6</td><td>58.0</td><td>46.6</td><td>53.5</td><td>54.7</td><td>62.9</td><td>60.3</td><td>48.9</td><td>55.8</td><td>57.0</td><td>6.6</td></tr><tr><td>MemPro</td><td>61.2</td><td>58.4</td><td>47.1</td><td>54.0</td><td>55.2</td><td>63.5</td><td>60.7</td><td>49.4</td><td>56.2</td><td>57.5</td><td>6.8</td></tr><tr><td>Initial go</td><td>63.4</td><td>59.7</td><td>48.3</td><td>55.7</td><td>56.8</td><td>65.3</td><td>61.6</td><td>50.1</td><td>57.1</td><td>58.5</td><td></td></tr><tr><td>MemCodex</td><td>67.1</td><td></td><td>62.9 52.3</td><td>60.9</td><td>60.8(+5.6)</td><td>68.2</td><td>63.9</td><td>52.8</td><td>59.4</td><td>61.1(+3.6)</td><td>2.4 2.0</td></tr></table>

Baselines. Unstructured baselines are 32k-token full context and dense RAG with BGE (Xiao et al., 2024) over 512-token chunks under $B = 4 , 0 9 6$ . Hand-designed baselines are MemGPT (Packer et al., 2023), A-Mem (Xu et al., 2025), Mem0 (Chhikara et al., 2025), MemoryOS (Kang et al., 2025) and HMO (Liu et al., 2026c). Adaptive or evolved baselines are Mem-� (Wang et al., 2025b), AutoMem (Du et al., 2026), MemEvolve (Zhang et al., 2025a) and MemPro (Liu et al., 2026d). Search-based baselines share the evolve split, meta agent and 64-child budget.

Models and search. We evaluate the open-weight Qwen3.5-27B (Qwen Team, 2026) and the API model GPT-5.4-mini. Per run, one backbone serves every system as task agent �, writes derived items, and serves as meta agent � at temperature 0.2 with one call per child. Qwen3.5-27B reads text through the NL interface and Metis-27B (Zhang et al., 2026d) state kv memory through the KV interface, while GPT-5.4-mini uses only the NL interface. Specifically, Qwen3.5-27B takes kv memory from Metis-27B, an open-weight memory foundation model that inserts fast-weight memory blocks into a frozen Qwen3.5-27B and updates them by test-time training in the forward pass with a gated delta rule, as in Titans (Behrouz et al., 2024). For each record, we run Metis-27B, extract its memory blocks’ key-value state, and inject it through the KV interface. Because the two models share the frozen backbone, injection is native and requires no alignment. Without weight access, GPT-5.4-mini has no latent memory. GPT-4o-mini (Hurst et al., 2024) is the judge. Evolution uses � = 8 rounds, � = 4 parents and � = 2 children per parent. Search-based results average three seeds.

## 4.1 Main Results

MemCodex is the most accurate system on all four benchmarks and reads the fewest tokens. It averages 60.8 percent, 5.6 points above MemPro and 7.2 above full context, while reading 2.0k tokens per question against 6.8k and 21.4k (Table 1). It answers in 2.7 seconds, 2.1× faster than MemPro and 3.6× faster than full context (Figure 1, right). Seed standard deviations are at most 0.8 points. Margins over MemPro peak on PersonaMind (+6.9) and LoCoMo (+5.9), which we attribute to latent memory carrying recurring preferences and routes linking cross-session evidence (Figure 6, middle). The margin is smallest on LongMemEval (+4.5), whose records already isolate evidence.

Table 2 Ablations of MemCodex on hierarchy depth, evidence channel and evolution, with one change per row under the same search budget. Avg. averages the four benchmarks, Rec. denotes LoCoMo evidence recall, and parentheses give the change from MemCodex.
<table><tr><td>Method</td><td>Avg. (%)</td><td>Rec. (%)</td><td>Tok/q (k) Method</td><td></td><td>Avg. (%)</td><td>Rec. (%)</td><td>Tok/q (k)</td></tr><tr><td>MemCodex</td><td>60.8</td><td>88.4</td><td>2.0</td><td>Evolution</td><td></td><td></td><td></td></tr><tr><td>Hierarchy depth</td><td></td><td></td><td></td><td>No evolution</td><td>56.8 (-4.0) 82.6 (–5.8)</td><td></td><td>2.4</td></tr><tr><td>1 layer (raw)</td><td></td><td>54.1 (-6.7) 71.2 (-17.2)</td><td>1.8</td><td>Module selection</td><td>58.3 (-2.5) 84.9 (–3.5)</td><td></td><td>2.3</td></tr><tr><td>2 layers (+summary) 56.5 (-4.3) 78.9 (–9.5)</td><td></td><td></td><td>2.3</td><td>Layer rewrites</td><td>57.2 (–3.6) 80.9 (–7.5)</td><td></td><td>2.9</td></tr><tr><td>3 layers (+graph)</td><td></td><td>59.0 (-1.8)84.6 (-3.8)</td><td>2.1</td><td>Program rewrites</td><td>56.7 (-4.1) 79.8 (-8.6)</td><td></td><td>3.2</td></tr><tr><td>Evidence channel</td><td></td><td></td><td></td><td>No acceptance test 58.1 (–2.7) 84.7 (–3.7)</td><td></td><td></td><td>2.6</td></tr><tr><td>Content channel</td><td></td><td>55.4 (-5.4) 63.0 (–25.4)</td><td>3.9</td><td>No archive</td><td>59.2 (-1.6) 86.8 (–1.6)</td><td></td><td>2.1</td></tr><tr><td>Content + routing</td><td></td><td>59.6(-1.2) 88.7 (+0.3)</td><td>4.6</td><td></td><td></td><td></td><td></td></tr></table>

Evidence preservation puts �<sub>0</sub> ahead, and evolution supplies most of the margin. The initial architecture � averages 56.8 percent at 2.4k tokens, 1.6 points above MemPro, which we attribute to derived items routing reads to their exact source turns. Program evolution adds 4.0 points while reducing tokens by 17 percent. LoCoMo evidence recall rises from MemPro’s 79.3 percent to 88.4 percent (Figure 6, left). With GPT-5.4-mini, which evolves its own architecture through the NL interface alone, MemCodex leads MemPro by 3.6 points, matching the margin of the Qwen3.5-27B variant without kv memory, which averages 58.8 and leads MemPro by 3.6, so latent memory contributes 2.0 points under Qwen3.5-27B.

## 4.2 Ablation

Table 2 tests the two claims behind MemCodex, the hierarchy with routed evidence and per-method evolution, using variants evolved under the same budget. To test hierarchy depth, we fix the number of layers to one (the raw layer), two (adding summary), three (adding graph), or four (the full hierarchy with skill). To isolate the evidence channel, variants place derived-item text in � instead of source turns, or alongside them. The evolution variants retain four layers but change how their programs are obtained. No evolution keeps � . Module selection chooses each method among predefined implementations as AutoMem does. Layer rewrites change one layer program per child as the slot-wise evolver of MemEvolve does, while program rewrites change every layer per child as MemPro does. Finally, no acceptance test archives every child that passes the validity checks, and no archive follows one greedy chain. Read policies are compared under RQ5, and the variant without kv memory appears with the main results.

RQ1: Does the hierarchy matter? Each added layer improves accuracy while context remains near 2k tokens. One, two, three, and four layers score 54.1 (−6.7), 56.5 (−4.3), 59.0 (−1.8), and 60.8, while Tok/q remains between 1.8k and 2.3k. Because the context cost stays flat, we attribute the gains to reads that stop at the right level rather than to more context. graph contributes the largest gain, +2.5 points, which we attribute to its cross-session links.

Routing to source turns preserves evidence more eficiently than reading derived text. Against full MemCodex at 60.8 accuracy, 88.4 Rec. and 2.0k Tok/q, the content channel reaches 55.4 (−5.4) with 63.0 Rec. and 3.9k Tok/q. Content plus routing restores Rec. to 88.7 but reaches only 59.6 (−1.2) at 4.6k Tok/q. Derived text therefore more than doubles context when added beside source turns and still dilutes the evidence used for answering.

RQ2: Does open-ended per-method evolution beat the prior adaptive lines? Per-method evolution outperforms matched variants of the three prior search lines on the same four-layer hierarchy. MemCodex scores 60.8, compared with 56.8 (−4.0) for �<sub>0</sub>, 58.3 (−2.5) for module selection, 57.2 (−3.6)

![](images/07240fbfb41340e89e72d7bf7213cab07571c489b43021b847f50264abffbeb3.jpg)

![](images/9ff4c84ce972f6128abd30e3a9411bd0df3c7b137d0346e756e64eaaa29e7c24.jpg)  
Reads sto<sub>pp</sub>in<sub>g</sub> at la<sub>y</sub>er (%)

![](images/35d2bd5356ae1f2bd63e3e238bcd7dea2b2b8f871a4b1b09fadc3eae956b4adc.jpg)  
Figure 5 Evolution dynamics of MemCodex across three search seeds, as gain over round 0 on LoCoMo for the search-based systems (left), the fraction of test reads stopping at each layer under $g ^ { \star }$ by LoCoMo question type and by benchmark (middle), and LoCoMo test accuracy by question type under always-raw, stop-at-first-hit and adaptive reads (right). LME and PMind abbreviate LongMemEval and PersonaMind.

Table 3 Accepted edits along the chain from $g _ { 0 }$ to $g ^ { \star }$ on LoCoMo, median seed, with the round of acceptance, the rewritten method, evolve-split and test accuracy, and thousands of prompt tokens per test question.
<table><tr><td>Round Edit</td><td>Evolve Test Tok/q (k)</td></tr><tr><td>0 g0</td><td>63.6 63.4 2.31</td></tr><tr><td>1</td><td>SUMMARY.route 66.9 65.2 2.26</td></tr><tr><td>2 GRAPH.admit</td><td>68.2 65.9 2.21</td></tr><tr><td>3 RAW.score</td><td>68.5 66.1 2.13</td></tr><tr><td>5</td><td>SUMMARY.index 69.5 66.8 2.02</td></tr><tr><td>7 SKILL.route 8 GRAPH.propose</td><td>69.9 67.0 1.94 70.1 67.1 1.93</td></tr></table>

for layer rewrites, and 56.7 (−4.1) for program rewrites. Menus remain limited to implementations anticipated by the designer. Bundled rewrites remain near �<sub>0</sub>, which we attribute to several changes being harder to preserve and attribute at once, with validity checks rejecting 41 of 64 layer-rewrite children against 9 of 64 per-method children.

The acceptance test prevents evolve-split overfitting, and the archive beats a single greedy chain. Without the acceptance test, accuracy falls to 58.1 (−2.7) because noisy children enter the archive. Without the archive, a greedy chain reaches 59.2 (−1.6) because it cannot return to a useful ancestor after a dead end.

## 4.3 Evolution D<sub>y</sub>namics

Setting. These analyses use Qwen3.5-27B and three search seeds per benchmark, each with $T = 8$ rounds of eight children, or 64 children per search. Per-round test accuracy is computed after the search from archived checkpoints of the best evolve-split architecture and averaged over seeds. Retention divides the test-split gain of $g ^ { \star }$ over $g _ { 0 }$ by its evolve-split gain. Transfer evaluates one benchmark’s $g ^ { \star }$ unchanged on every test split.

RQ3: How does accuracy change across rounds? MemCodex gains more and keeps gaining, while the three search-based baselines gain less and flatten. MemCodex gains 3.7 points, from 63.4 to 67.1, including 1.8 in the first round from a route rewrite. MemPro gains 2.5 (58.7 to 61.2) through whole-program rewrites, mostly after round four. MemEvolve gains 1.7 (58.9 to 60.6) and flattens after round six. AutoMem gains 1.6 (58.5 to 60.1) and plateaus from round three. Per-method evolution keeps finding acceptable edits in later rounds because each child changes one method, so a small gain is attributable and passes the test.

Table 4 Test accuracy under record-disjoint splits with Qwen3.5-27B, mean of three seeds, with the best per column in bold and the second underlined, and the change of the average against Table 1 in parentheses.
<table><tr><td>Method</td><td>LoCoMo</td><td>LongMemEval</td><td>MAB</td><td>PersonaMind</td><td> $\operatorname { A v } { \mathbf { g } } .$ </td></tr><tr><td>AutoMem</td><td>59.6</td><td>57.3</td><td>45.5</td><td>52.2</td><td>53.7 (-0.4)</td></tr><tr><td>MemEvolve</td><td>60.0</td><td>58.0</td><td>46.1</td><td>52.9</td><td>54.3 (-0.4)</td></tr><tr><td>MemPro</td><td>60.7</td><td>58.4</td><td>46.6</td><td>53.5</td><td>54.8 (-0.4)</td></tr><tr><td>MemCodex (initial go)</td><td>63.0</td><td>59.7</td><td>47.9</td><td>55.1</td><td>56.4 (-0.4)</td></tr><tr><td>MemCodex</td><td>65.6</td><td>62.9</td><td>51.4</td><td>59.3</td><td>59.8 (-1.0)</td></tr></table>

Table 3 traces the accepted edits from $g _ { 0 }$ to $g ^ { \star }$ on LoCoMo. The first edit rewrites the route method of summary so that a stale summary cannot stop the read, raising evolve-split accuracy from 63.6 to 66.9. Later edits each add 0.2 to 1.3 evolve-split points, while Tok $/ \mathsf { q }$ falls with every accepted edit from 2.31k to 1.93k. The round-5 child descends from the round-3 node rather than the round-4 leader, because the archive retains every accepted architecture as a parent. Every accepted edit names one method of one layer.

RQ4: Do accepted gains hold on the test split? The acceptance test rejects improvements that primarily fit evolve-split noise. With the acceptance test, the test split retains 53 to 78 percent of the evolve-split gain, including 3.7 of 5.8 points on LoCoMo. By contrast, accepting every child that passes K admits 53 to 56 of 64 children and ends 1.3 to 2.0 points higher on the evolve split, but retains only 10 to 24 percent of its gain. Retention is lowest on LongMemEval, which has 100 evolve questions. Under record-disjoint splits that hold out whole records (Table 4), the margin over MemPro is 5.0 points and the evolution gain is 3.4, so at most 0.6 points are record-specific. The held-out records are 2 of the 10 LoCoMo conversations, 29 of the 146 MAB records and 60 of the 300 PersonaMind records, while LongMemEval already pairs each question with its own record. The evolve split takes 20 percent of the questions of the evolve records and the test split every question of the held-out records, so no question of a held-out record is seen at any stage. Every baseline and $g _ { 0 }$ lose 0.4 points against Table 1, whereas MemCodex loses 1.0.

RQ5: Does the read adapt to the question and the record? Adaptive reads match the best fixed depth on every question type at a fraction of the tokens. No fixed policy wins everywhere. Always raw leads on single-hop questions (74.3), and stop at first hit on the other three. The adaptive read scores 74.6, 57.4, 60.1 and 52.8 on single-hop, multi-hop, temporal and open-domain questions, matching or exceeding the best fixed policy on every type and leading by 4.6 points on multi-hop, while reading 1.9k tokens against 3.3k for always raw (Figure 5, right). Under $g ^ { \star }$ , 65 percent of single-hop reads reach the raw layer, whereas 72 percent of multi-hop reads stop above it. Across ten LoCoMo conversations, 37 to 61 percent of all reads and 58 to 83 percent of multi-hop reads stop above raw, so depth also varies with the record.

RQ6: Are evolved architectures task-specific? Each task evolves a distinct architecture, although useful structure transfers. Every search rewrites a route method, and three searches also rewrite a write-path method. Each native $g ^ { \star }$ beats the best transferred architecture by 1.5 to 2.8 points, while transferred architectures retain 0.6 to 2.4 points over $g _ { 0 } ,$ , which is about one third of the native gain (Figure 6, right). The LongMemEval architecture switches skill of and adds only 0.6 on MAB and 0.9 on PersonaMind, where 11 and 16 percent of reads stop at skill.

![](images/e2e644c3ba8299748ec78b1f59f6927b75e013293fd68877e71913ea1ae2c18f.jpg)

![](images/e148e0bc90ab747333a7ef9d1cb8bdc6a8a9884fb5c219299fdc3696143942c7.jpg)

![](images/2dda8b22ef43a0042fbff38f29745ca1b02db7162623304bcc1684dc0f60add8.jpg)  
Evaluated on  
Figure 6 Evidence recall, question-type accuracy and cross-benchmark transfer of MemCodex, as evidence recall on LoCoMo and LongMemEval for every system of Table 1 (left), LoCoMo test accuracy by question type for MemPro, the initial architecture $g _ { 0 }$ and the evolved architecture (middle), and test accuracy of each $g ^ { \star }$ on each test split, darker for larger gains over $g _ { 0 }$ and bold on the diagonal (right).

## 5 Related Work

Agent memory evolution. Agent memory has moved from hand-designed stores toward structures and operations that adapt to the task (Hu et al., 2025c; Wu et al., 2026b). The first line of work hand-designs architectures as paged tiers, fact and entity stores, hierarchical or graph indexes, and latent or parametric memory (Packer et al., 2023; Zhong et al., 2024; Chhikara et al., 2025; Xu et al., 2025; Rasmussen et al., 2025; Kang et al., 2025; Liu et al., 2026c; Sarthi et al., 2024; Gutiérrez et al., 2024; Wu et al., 2026c; Chen et al., 2026a; Hsu et al., 2026; Wang et al., 2024c; Behrouz et al., 2024; Li et al., 2025a; Zhang et al., 2026d; Wang et al., 2025a). MemGPT (Packer et al., 2023), for example, pages information between the context window and external storage following an operating-system model. Beyond hand design, a second line instead learns policies that decide what to write, update, consolidate, forget, or retrieve (Wang et al., 2025b; Sun et al., 2026; Wang et al., 2026c; Jiang et al., 2026; Zhang et al., 2026a; Li et al., 2026; Chen et al., 2026b; Lin et al., 2026b; Ye et al., 2026; Wang et al., 2026a). Mem-� (Wang et al., 2025b), for instance, trains its write and update operations with reinforcement learning. More recently, memory design itself has become a search problem (Du et al., 2026; Lu et al., 2026; Zhang et al., 2025a; Xiong et al., 2026; Liu et al., 2026b). Closest to our setting, MemPro (Liu et al., 2026d) evolves the entire construction and retrieval pipeline as one program over a version tree of runnable implementations. By contrast, MemCodex evolves the layer programs and hierarchy composition one method at a time, while every derived item keeps its exact source turns, so evidence remains recoverable throughout evolution.

Program search. More broadly, program search lets agents improve their own code or the harness around a model under empirical evaluation. One line automates agent and harness design (Hu et al., 2025a; Lee et al., 2026; Lin et al., 2026a; Zhang et al., 2026c; Wu et al., 2026a). ADAS (Hu et al., 2025a), for instance, lets a meta agent program new agents and archive its discoveries. Another line extends this approach to self-referential agents that rewrite their own code (Schmidhuber, 2007) and retain variants through empirical validation rather than the proof demanded by the Gödel machine (Zhang et al., 2025b, 2026b; Iacob et al., 2026; Liu et al., 2026a; Srikanth et al., 2026), while acceptance rules for such changes are studied directly (Shawn, 2026). The Darwin Gödel Machine (Zhang et al., 2025b) exemplifies this line by growing an archive of self-modifying coding agents validated on benchmarks. HyperAgents (Zhang et al., 2026b) further merge the task agent and meta agent into one editable program. By contrast, in MemCodex the search unit is one method of one memory layer behind a fixed interface, and a child enters the archive only after validity checks and an acceptance test.

## 6 Conclusion

We propose MemCodex, a self-evolving hierarchical memory system that organizes experience into executable layer programs for summaries, relations, skills and latent memory. It evolves each layer through attributable per-method edits behind fixed interfaces and keeps every derived item’s exact source turns. Reads traverse the hierarchy from coarse to fine and descend to source turns as needed. Through MemArena, it outperforms hand-designed and adaptive memory systems across benchmarks and backbones at lower cost. The experiments credit evidence preservation with the initial lead and per-method evolution with most of the rest, while the acceptance test carries evolve-split gains to the test split and evolved architectures remain task-specific yet transfer in part. Next, we will evolve the runtime and the layer set themselves under record-disjoint deployment.

## References

Ali Behrouz, Peilin Zhong, and Vahab Mirrokni. Titans: Learning to memorize at test time. arXiv preprint arXiv:2501.00663, 2024.

Han Chen, Zining Zhang, Wenqi Pei, Bingsheng He, Ming Wu, Jason Zeng, Michael Heinrich, Wei Wu, and Hongbao Zhang. MemForest: an eficient agent memory system with hierarchical temporal indexing. arXiv preprint arXiv:2605.23986, 2026a.

Xingyu Chen, Rui Wang, Zhaopeng Tu, and Liefeng Bo. AdaMem: learning what to remember with adaptive memory policies for personalized agents. arXiv preprint arXiv:2606.21144, 2026b.

Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. Mem0: Building productionready ai agents with scalable long-term memory. arXiv preprint arXiv:2504.19413, 2025.

Qianggang Ding, Xingyao Wang, Rui Feng, Zhibin Wang, Feixiang Wang, Kelong Mao, Hao Sun, Zhiyao Luo, Jiankai Tang, Lei Li, et al. Combodied agents: a new paradigm of human-centric agentic ai. arXiv preprint arXiv:2608.10915, 2026.

Lin Du, Jie Zhou, Yuxuan Cai, Kai Chen, Qin Chen, Xin Li, Bo Zhang, Wei Li, and Liang He. Automem: A text-gradient recursive self-improvement framework for automated memory architectures search. arXiv preprint arXiv:2608.14621, 2026.

Yiming Du, Hongru Wang, Zhengyi Zhao, Bin Liang, Baojun Wang, Wanjun Zhong, Zezhong Wang, and Kam-Fai Wong. PerLTQA: A personal long-term memory dataset for memory classification, retrieval, and fusion in question answering. In Kam-Fai Wong, Min Zhang, Ruifeng Xu, Jing Li, Zhongyu Wei, Lin Gui, Bin Liang, and Runcong Zhao, editors, Proceedings of the 10th SIGHAN Workshop on Chinese Language Processing (SIGHAN-10), pages 152–164, Bangkok, Thailand, August 2024. Association for Computational Linguistics. URL https://aclanthology.org/2024.sighan-1.18/.

Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan Larson. From local to global: A graph RAG approach to query-focused summarization. arXiv preprint arXiv:2404.16130, 2024.

Yubin Ge, Salvatore Romeo, Jason Cai, Raphael Shu, Yassine Benajiba, Monica Sunkara, and Yi Zhang. TReMu: Towards neuro-symbolic temporal reasoning for LLM-agents with memory in multi-session dialogues. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar, editors, Findings of the Associationfor Computational Linguistics: ACL 2025, pages 18974–18988, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-256-5. doi: 10.18653/v1/2025.findings-acl. 972. URL https://aclanthology.org/2025.findings-acl.972/.

Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su. Hipporag: Neurobiologically inspired long-term memory for large language models. In Advances in Neural Information Processing Systems, volume 37, 2024. arXiv:2405.14831.

Haoyu Han, Li Ma, Yu Wang, Harry Shomer, Yongjia Lei, Zhisheng Qi, Kai Guo, Zhigang Hua, Bo Long, Hui Liu, Charu C. Aggarwal, and Jiliang Tang. RAG vs. GraphRAG: A systematic evaluation and key insights. In Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2, pages 8966–8977, 2026. doi: 10.1145/3770855.3817575.

Sirui Hong, Mingchen Zhuge, Jiaqi Chen, Xiawu Zheng, Yuheng Cheng, Ceyao Zhang, Jinlin Wang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, Chenyu Ran, Lingfeng Xiao, Chenglin Wu, and Jürgen Schmidhuber. MetaGPT: Meta programming for a multi-agent collaborative framework. arXiv preprint arXiv:2308.00352, 2023.

Hao-Lun Hsu, Nikki Lijing Kuang, Boyi Liu, Zhewei Yao, and Yuxiong He. Organize then retrieve: Hierarchical memory navigation for eficient agents. arXiv preprint arXiv:2606.11680, 2026.

Shengran Hu, Cong Lu, and Jef Clune. Automated design of agentic systems. In International Conference on Learning Representations, 2025a.

Yuanzhe Hu, Yu Wang, and Julian McAuley. Evaluating memory in LLM agents via incremental multi-turn interactions. arXiv preprint arXiv:2507.05257, 2025b.

Yuyang Hu, Shichun Liu, Yanwei Yue, Guibin Zhang, Boyang Liu, Fangyi Zhu, Jiahang Lin, Honglin Guo, Shihan Dou, Zhiheng Xi, Senjie Jin, Jiejun Tan, Yanbin Yin, Jiongnan Liu, Zeyu Zhang, Zhongxiang Sun, Yutao Zhu, Hao Sun, Boci Peng, Zhenrong Cheng, Xuanbo Fan, Jiaxin Guo, Xinlei Yu, Zhenhong Zhou, Zewen Hu, Jiahao Huo, Junhao Wang, Yuwei Niu, Yu Wang, Zhenfei Yin, Xiaobin Hu, Yue Liao, Qiankun Li, Kun Wang, Wangchunshu Zhou, Yixin Liu, Dawei Cheng, Qi Zhang, Tao Gui, Shirui Pan, Yan Zhang, Philip Torr, Zhicheng Dou, Ji-Rong Wen, Xuanjing Huang, Yu-Gang Jiang, and Shuicheng Yan. Memory in the age of ai agents. arXiv preprint arXiv:2512.13564, 2025c.

Wei-Chieh Huang, Weizhi Zhang, Yuchen Wu, Yankai Chen, Eric Hanchen Jiang, Wooseong Yang, Yiwei Yang, Henry Peng Zou, Hanrong Zhang, Ying Nian Wu, Haolun Wu, Kai-Wei Chang, Philip S. Yu, Xue Liu, and Aylin Caliskan. Harness the memory: A holistic evaluation of memory substrates in memory agents. arXiv preprint arXiv:2608.15008, 2026.

Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.

Alex Iacob, Andrej Jovanović, William F. Shen, Daniel Burkhardt, Meghdad Kurmanji, Nurbek Tastan, Lorenzo Sani, Niccolò Alberto Elia Venanzi, Ambroise Odonnat, Zeyu Cao, Bill Marino, Xinchi Qiu, and Nicholas D. Lane. The red queen gödel machine: Co-Evolving agents and their evaluators. arXiv preprint arXiv:2606.26294, 2026.

Bowen Jiang, Zhuoqun Hao, Young-Min Cho, Bryan Li, Yuan Yuan, Sihao Chen, Lyle Ungar, Camillo J. Taylor, and Dan Roth. Know me, respond to me: Benchmarking LLMs for dynamic user profiling and personalized responses at scale. In Conference on Language Modeling (COLM), 2025. PersonaMem benchmark; arXiv:2504.14225.

Eric Hanchen Jiang, Zhi Zhang, Yuchen Wu, Levina Li, Dong Liu, Xiao Liang, Rui Sun, Yubei Li, Edward Sun, Haozheng Luo, Zhaolu Kang, Aylin Caliskan, Kai-Wei Chang, and Ying Nian Wu. Memory as a controlled process: Learned adaptive memory management for llm agents. arXiv preprint arXiv:2607.13591, 2026.

Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. Swe-bench: Can language models resolve real-world github issues? arXiv preprint arXiv:2310.06770, 2023.

Jiazheng Kang, Mingming Ji, Zhe Zhao, and Ting Bai. Memory OS of AI agent. arXiv preprint arXiv:2506.06326, 2025.

Yoonho Lee, Roshen Nair, Qizheng Zhang, Kangwook Lee, Omar Khattab, and Chelsea Finn. Meta-Harness: End-to-End optimization of model harnesses. arXiv preprint arXiv:2603.28052, 2026.

Wenzhi Li, Dong Nie, Rui Lan, Tongtong Lyu, Peiyao Wang, Lingzi Hong, Weihang Pan, Binbin Lin, Boyuan Pan, and Yao Hu. Dual-layer agentic memory with fast write routing and slow consolidation. arXiv preprint arXiv:2608.22215, 2026.

Zhiyu Li, Chenyang Xi, Chunyu Li, Ding Chen, Boyu Chen, Shichao Song, Simin Niu, Hanyu Wang, Jiawei Yang, Chen Tang, Qingchen Yu, Jihao Zhao, Yezhaohui Wang, Peng Liu, Zehao Lin, Pengyuan Wang, Jiahao Huo, Tianyi Chen, Kai Chen, Kehang Li, Zhen Tao, Huayi Lai, Hao Wu, Bo Tang, Zhengren Wang, Zhaoxin Fan, Ningyu Zhang, Linfeng Zhang, Junchi Yan, Mingchuan Yang, Tong Xu, Wei Xu, Huajun Chen, Haofen Wang, Hongkang Yang, Wentao Zhang, Zhi-Qin John Xu, Siheng Chen, and Feiyu Xiong. MemOS: A memory OS for AI system. arXiv preprint arXiv:2507.03724, 2025a.

Zizhen Li, Chuanhao Li, Yibin Wang, Qi Chen, Diping Song, Yukang Feng, Jianwen Sun, Jiaxin Ai, Fanrui Zhang, Mingzhu Sun, and Kaipeng Zhang. InMind: Evaluating LLMs in capturing and applying individual human reasoning styles. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng, editors, Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 5038– 5076, Suzhou, China, November 2025b. Association for Computational Linguistics. ISBN 979-8-89176-332-6. doi: 10.18653/v1/2025.emnlp-main.254. URL https://aclanthology.org/2025.emnlp-main.254/.

Jiahang Lin, Shichun Liu, Chengjun Pan, Lizhi Lin, Shihan Dou, Zhiheng Xi, Xuanjing Huang, Hang Yan, Zhenhua Han, Tao Gui, and Yu-Gang Jiang. Agentic harness engineering: Observability-Driven automatic evolution of Coding-Agent harnesses. arXiv preprint arXiv:2604.25850, 2026a.

Minhua Lin, Zhiwei Zhang, Hanqing Lu, Hui Liu, Xianfeng Tang, Qi He, Xiang Zhang, and Suhang Wang. MemMA: coordinating the memory cycle through Multi-Agent reasoning and In-Situ Self-Evolution. arXiv preprint arXiv:2603.18718, 2026b.

Bang Liu, Xinfeng Li, Jiayi Zhang, Jinlin Wang, Tanjin He, Sirui Hong, Hongzhang Liu, Shaokun Zhang, Kaitao Song, Kunlun Zhu, et al. Advances and challenges in foundation agents: From brain-inspired intelligence to evolutionary, collaborative, and safe systems. arXiv preprint arXiv:2504.01990, 2025.

Changzhi Liu, Yilun Liu, Sikuan Yan, Volker Tresp, and Yunpu Ma. Mendel gödel machine: Recursive Self-Improving coding agents via comparative evolution. arXiv preprint arXiv:2608.07645, 2026a.

Jiaqi Liu, Xinyu Ye, Peng Xia, Zeyu Zheng, Cihang Xie, Mingyu Ding, and Huaxiu Yao. EvolveMem:Self-Evolving memory architecture via AutoResearch for LLM agents. arXiv preprint arXiv:2605.13941, 2026b.

Junming Liu, Yifei Sun, Weihua Cheng, Haodong Lei, Yuqi Li, Yirong Chen, and Ding Wang. Hierarchical memory orchestration for personalized persistent agents. arXiv preprint arXiv:2604.01670, 2026c.

Qingshan Liu, Guoqing Wang, Wen Wu, Jingqi Huang, Xinqi Tao, Dejia Song, Jie Zhou, and Liang He. MemPro: Agentic memory systems as evolvable programs. arXiv preprint arXiv:2606.00619, 2026d.

Mingfei Lu, Mengjia Wu, Feng Liu, Jiawei Xu, Weikai Li, Haoyang Wang, Zhengdong Hu, Ying Ding, Yizhou Sun, Jie Lu, and Yi Zhang. Choosing how to remember: Adaptive memory structures for llm agents. arXiv preprint arXiv:2602.14038, 2026.

Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang. Evaluating very long-term conversational memory of LLM agents. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 13851–13870, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.747. URL https://aclanthology.org/2024.acl-long.747/.

Quinn McNemar. Note on the sampling error of the diference between correlated proportions or percentages. Psychometrika, 12(2):153–157, 1947.

Alexander Novikov, Ngân Vu, Marvin Eisenberger, Emilien Dupont, Po-Sen Huang, Adam Zsolt Wagner, Sergey˜ Shirobokov, Borislav Kozlovskii, Francisco J. R. Ruiz, Abbas Mehrabian, M. Pawan Kumar, Abigail See, Swarat Chaudhuri, George Holland, Alex Davies, Sebastian Nowozin, Pushmeet Kohli, and Matej Balog. AlphaEvolve: A coding agent for scientific and algorithmic discovery. arXiv preprint arXiv:2506.13131, 2025.

Siru Ouyang, Jun Yan, I-Hung Hsu, Yanfei Chen, Ke Jiang, Zifeng Wang, Rujun Han, Long T. Le, Samira Daruki, Xiangru Tang, Vishy Tirumalashetty, George Lee, Mahsan Rofouei, Hangfei Lin, Jiawei Han, Chen-Yu Lee, and Tomas Pfister. Reasoningbank: Scaling agent self-evolving with reasoning memory. arXiv preprint arXiv:2509.25140, 2025.

Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G Patil, Ion Stoica, and Joseph E Gonzalez. Memgpt: Towards llms as operating systems. arXiv preprint arXiv:2310.08560, 2023.

Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026. URL https://qwen.ai/blog?id= qwen3.5. Model card, https://huggingface.co/Qwen/Qwen3.5-27B.

Preston Rasmussen, Pavlo Paliychuk, Travis Beauvais, Jack Ryan, and Daniel Chalef. Zep: A temporal knowledge graph architecture for agent memory. arXiv preprint arXiv:2501.13956, 2025.

Bernardino Romera-Paredes, Mohammadamin Barekatain, Alexander Novikov, Matej Balog, M. Pawan Kumar, Emilien Dupont, Francisco J. R. Ruiz, Jordan S. Ellenberg, Pengming Wang, Omar Fawzi, Pushmeet Kohli,

and Alhussein Fawzi. Mathematical discoveries from program search with large language models. Nature, 625(7995):468–475, 2024. doi: 10.1038/s41586-023-06924-6.

Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and Christopher D. Manning. RAPTOR: Recursive abstractive processing for tree-organized retrieval. In The Twelfth International Conference on Learning Representations, 2024. arXiv:2401.18059.

Jürgen Schmidhuber. Gödel machines: Fully self-referential optimal universal self-improvers. In Ben Goertzel and Cassio Pennachin, editors, Artificial General Intelligence, pages 199–226. Springer, 2007. arXiv:cs/0309048.

Zayx Shawn. PACE: Anytime-Valid acceptance tests for Self-Evolving agents. arXiv preprint arXiv:2606.08106, 2026.

Haochen Shi, Xingdi Yuan, and Bang Liu. Evolving programmatic skill networks. arXiv preprint arXiv:2601.03509, 2026.

Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. Advances in neural information processing systems, 36:8634–8652, 2023.

Dhruv Srikanth, Bingchen Zhao, Dixing Xu, Yuxiang Wu, and Zhengyao Jiang. Recursive self-improvement of AI research agents. arXiv preprint arXiv:2609.26457, 2026.

Xiaolong Sun, Qichao Wang, Hangyu Li, and Liang Chen. Verifiable memory: Learning unified memory management with local and global verifiers for large language model agents. arXiv preprint arXiv:2608.03137, 2026.

Zhengwei Tao, Ting-En Lin, Xiancai Chen, Hangyu Li, Yuchuan Wu, Yongbin Li, Zhi Jin, Fei Huang, Dacheng Tao, and Jingren Zhou. A survey on self-evolution of large language models. arXiv preprint arXiv:2404.14387, 2024.

Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291, 2023.

Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, Wayne Xin Zhao, Zhewei Wei, and Ji-Rong Wen. A survey on large language model based autonomous agents. Frontiers of Computer Science, 18(6):186345, 2024a.

Xiaoqiang Wang and Bang Liu. Oscar: Operating system control via state-aware reasoning and re-planning. In International Conference on Learning Representations, volume 2025, pages 71417–71439, 2025.

Xiaoqiang Wang, Lingfei Wu, Tengfei Ma, and Bang Liu. FAC<sup>2</sup>E: Better understanding large language model capabilities by dissociating language and cognition. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen, editors, Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 13228–13243, Miami, Florida, USA, November 2024b. Association for Computational Linguistics. doi: 10.18653/v1/2024.emnlp-main.734. URL https://aclanthology.org/2024.emnlp-main.734/.

Xiaoqiang Wang, Suyuchen Wang, Yun Zhu, and Bang Liu. R<sup>3</sup>Mem: Bridging memory retention and retrieval via reversible compression. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar, editors, Findings of the Association for Computational Linguistics: ACL 2025, pages 4541–4557, Vienna, Austria, July 2025a. Association for Computational Linguistics. ISBN 979-8-89176-256-5. doi: 10.18653/v1/2025.findings-acl.235. URL https://aclanthology.org/2025.findings-acl.235/.

Xiaoqiang Wang, Chao Wang, Hadi Nekoei, Christopher Pal, Alexandre Lacoste, Spandana Gella, Bang Liu, and Perouz Taslakian. Mem-�: Adaptive memory through learning when and what to generate. arXiv preprint arXiv:2605.21463, 2026a.

Xiaoqiang Wang, Suyuchen Wang, Yun Zhu, and Bang Liu. System-1.5 reasoning: Traversal in language and latent spaces with dynamic shortcuts. Advances in Neural Information Processing Systems, 38:65328–65351, 2026b.

Yu Wang, Yifan Gao, Xiusi Chen, Haoming Jiang, Shiyang Li, Jingfeng Yang, Qingyu Yin, Zheng Li, Xian Li, Bing Yin, Jingbo Shang, and Julian McAuley. Memoryllm: towards self-updatable large language models. In Proceedings of the 41st International Conference on Machine Learning, ICML’24. JMLR.org, 2024c.

Yu Wang, Ryuichi Takanobu, Zhiqi Liang, Yuzhen Mao, Yuanzhe Hu, Julian McAuley, and Xiaojian Wu. Mem-�: Learning memory construction via reinforcement learning. arXiv preprint arXiv:2509.25911, 2025b.

Yubo Wang, Qiuyu Zhao, Zenghui Sun, Shichao Dong, Jinsong Lan, Xiaoyong Zhu, Haoyang Li, Bo Zheng, and Lei Chen. Cmi-mem: Toward generalizable long-term memory management via cmi-augmented reinforcement learning. arXiv preprint arXiv:2607.20553, 2026c.

Zora Zhiruo Wang, Jiayuan Mao, Daniel Fried, and Graham Neubig. Agent workflow memory. In International Conference on Machine Learning, pages 63897–63911. PMLR, 2025c.

Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, and Dong Yu. LongMemEval: Benchmarking chat assistants on long-term interactive memory. In The Thirteenth International Conference on Learning Representations, 2025. arXiv:2410.10813.

Siwei Wu, Jincheng Ren, Yizhi Li, Haau-Sing Li, Chengran Yang, Yuxuan Zhang, Weicheng Gu, Jian Yang, Riza Batista-Navarro, Chuanyi Zhang, Xianglong Liu, Ming Zhou, Bryan Dai, and Chenghua Lin. ModularRSI: modular and generalizable recursive harness Self-Improvement. arXiv preprint arXiv:2609.14857, 2026a.

Yanchen Wu, Tenghui Lin, Yingli Zhou, Fangyuan Zhang, Qintian Guo, Xun Zhou, Sibo Wang, Xilin Liu, Yuchi Ma, and Yixiang Fang. Memory in the LLM era: Modular architectures and strategies in a unified framework. arXiv preprint arXiv:2604.01707, 2026b.

Zhaofen Wu, Hanrong Zhang, Fulin Lin, Wujiang Xu, Xinran Xu, Yankai Chen, Henry Peng Zou, Shaowen Chen, Weizhi Zhang, Xue Liu, Philip S. Yu, and Hongwei Wang. GAM: hierarchical graph-based agentic memory for LLM agents. arXiv preprint arXiv:2604.12285, 2026c.

Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muennighof, Defu Lian, and Jian-Yun Nie. C-pack: Packed resources for general chinese embeddings. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 641–649, 2024. BGE embedding models; arXiv:2309.07597.

Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh Jing Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, Yitao Liu, Yiheng Xu, Shuyan Zhou, Silvio Savarese, Caiming Xiong, Victor Zhong, and Tao Yu. Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments. arXiv preprint arXiv:2404.07972, 2024.

Yiming Xiong, Shengran Hu, and Jef Clune. Learning to continually learn via meta-learning agentic memory designs. arXiv preprint arXiv:2602.07755, 2026.

Derong Xu, Yi Wen, Pengyue Jia, Yingyi Zhang, wenlin zhang, Yichao Wang, Huifeng Guo, Ruiming Tang, Xiangyu Zhao, Enhong Chen, and Tong Xu. From single to multi-granularity: Toward long-term memory association and selection of conversational agents. In The Fourteenth International Conference on Learning Representations, 2026. arXiv:2505.19549.

Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang. A-mem: Agentic memory for llm agents. arXiv preprint arXiv:2502.12110, 2025.

Shunyu Yao, Jefrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629, 2022.

Bowen Ye, Yongchao Xu, Zhijian Li, Xiang Yin, Junkai Ma, and Wenzhao Li. CoEvo-Mem: Co-Evolving retrieval policy and memory bank for LLM agents. arXiv preprint arXiv:2608.01739, 2026.

Guibin Zhang, Haotian Ren, Chong Zhan, Zhenhong Zhou, Junhao Wang, He Zhu, Wangchunshu Zhou, and Shuicheng Yan. Memevolve: Meta-evolution of agent memory systems. arXiv preprint arXiv:2512.18746, 2025a.

Haozhen Zhang, Quanyu Long, Jianzhu Bao, Tao Feng, Weizhi Zhang, Haodong Yue, and Wenya Wang. Memskill: Learning and evolving memory skills for self-evolving agents. arXiv preprint arXiv:2602.02474, 2026a.

Jenny Zhang, Shengran Hu, Cong Lu, Robert Lange, and Jef Clune. Darwin gödel machine: Open-ended evolution of self-improving agents. arXiv preprint arXiv:2505.22954, 2025b.

Jenny Zhang, Bingchen Zhao, Wannan Yang, Jakob Foerster, Jef Clune, Minqi Jiang, Sam Devlin, and Tatiana Shavrina. Hyperagents. arXiv preprint arXiv:2603.19461, 2026b.

Jiayi Zhang, Yongfeng Gu, Jianhao Ruan, Maojia Song, Yiran Peng, Zhiguang Han, Jinyu Xiang, Zhitao Wang, Caiyin Yang, Yixi Ouyang, Bang Liu, Chenglin Wu, and Yuyu Luo. Harnessing agentic evolution. arXiv preprint arXiv:2605.13821, 2026c.

Zeyu Zhang, Ziliang Guo, Yihang Sun, Xichong Zhang, Xixuan Hao, Zehao Lin, Yang Zhang, Xiaoyan Zhao, Tong Shen, Bo Tang, Zhi-Qin John Xu, Junchi Yan, Haofen Wang, Xu Chen, Feiyu Xiong, Zhiyu Li, and Tat-Seng Chua. Metis: Memory foundation model. arXiv preprint arXiv:2607.26760, 2026d.

Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu Lin, Yong-Jin Liu, and Gao Huang. Expel: Llm agents are experiential learners. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 19632–19642, 2024.

Junhao Zheng, Xidi Cai, Qiuke Li, Duzhen Zhang, ZhongZhi Li, Yingying Zhang, Le Song, and Qianli Ma. Lifelongagentbench: Evaluating llm agents as lifelong learners. arXiv preprint arXiv:2505.11942, 2025.

Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and Yanlin Wang. Memorybank: Enhancing large language models with long-term memory. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 19724–19731, 2024.

Shuyan Zhou, Frank F. Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, Uri Alon, and Graham Neubig. Webarena: A realistic web environment for building autonomous agents. arXiv preprint arXiv:2307.13854, 2023.

Wei Zhou, Xuanhe Zhou, Shaokun Han, Hongming Xu, Guoliang Li, Zhiyu Li, Feiyu Xiong, and Fan Wu. Are we ready for an agent-native memory system? arXiv preprint arXiv:2606.24775, 2026.