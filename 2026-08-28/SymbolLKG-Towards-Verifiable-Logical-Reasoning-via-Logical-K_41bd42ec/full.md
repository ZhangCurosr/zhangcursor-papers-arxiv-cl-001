# SymbolLKG: Towards Verifiable Logical Reasoning via Logical Knowledge Graph and Symbolic Solvers

Haizhao Fan Shanghai JiaoTong University

Yuchi Xiong Shanghai JiaoTong University

Xinping Guan Shanghai JiaoTong University

Jize Wang Shanghai JiaoTong University

Xinyi Le Shanghai JiaoTong University

## Abstract

Large Language Models (LLMs) have demonstrated remarkable proficiency in natural language understanding, yet they struggle with strict multi-step reasoning, frequently suffering from hallucinations and inconsistency. Existing solutions like Chain-of-Thought (CoT) lack rigorous verification mechanisms, while standard Retrieval-Augmented Generation (RAG) often misses the complex, structural dependencies inherent in logical tasks. To bridge this gap, we propose a Neuro-Symbolic architecture that integrates a Logical Knowledge Graph (LKG) with dynamic solver routing. Specifically, we introduce an ontology-based LKG that treats logical rules and constraints as first-class topological nodes, enabling explicit modeling of dependencies extracted from text. We further design a Logic Router to dynamically dispatch tasks to the optimal symbolic engine, which is supported by a topology-aware hybrid retrieval mechanism. Experimental results on logical reasoning benchmarks demonstrate that our framework significantly outperforms state-of-the-art prompting and RAG baselines, delivering higher accuracy and verifiable reasoning paths.

## 1 Introduction

Logical reasoning is the cornerstone of reliable artificial intelligence, particularly in high-stakes domains where precision is mandatory—healthcare [1], law [2], finance [3], and scientific discovery [4]. Driven by their natural language understanding capabilities, Large Language Models (LLMs) are increasingly deployed for complex tasks in these fields. However, the probabilistic nature of their “next-token prediction” architecture fundamentally limits the rigorous, multi-step logic required for strict reasoning scenarios.

Reasoning encompasses deductive, inductive, and abductive forms [5]; yet achieving trustworthy logical reasoning, characterized by reliability, interpretability, and adherence to formal rules, remains a fundamental challenge [5]. LLMs tend to generate plausible but incorrect steps and fail to maintain consistency across long contexts [6].

To address these issues, prompting methods such as Chain-of-Thought (CoT) [7] and Tree-of-Thoughts (ToT) [8] decompose complex problems, and Retrieval-Augmented Generation (RAG) [9] grounds responses in external data. However, these methods lack rigorous verification: CoT can propagate errors without symbolic grounding [10], and standard RAG misses the directional dependencies inherent in logical rules.

In contrast, Knowledge Graphs (KGs) provide structured models of entities and relationships [11, 12], supporting rigorous formal reasoning and interpretability [5]; but symbolic systems traditionally struggle with scalability on open-ended unstructured data. Integrating KGs with LLMs has thus emerged as a promising direction [13]: Think-on-Graph [14] and RoG [15] guide LLMs along graph paths; GraphRAG [16] structures retrieval from local contexts to global summarization; LightRAG [17] introduces a dual-level retrieval mechanism capturing both specific entities and conceptual themes; and KAG [18] pushes neuro-symbolic alignment in professional domains by coupling LLMs with a logical-form-guided engine. However, these approaches typically treat logical rules as unstructured text or implicit edge attributes rather than explicit, computable topological nodes. This representational gap limits the deterministic mapping required for rigorous symbolic verification, often forcing reliance on probabilistic text-to-code generation. Without distinct constraint nodes, analyzing the logical topology to dynamically route problems to the appropriate solver (e.g., Z3 vs. Prover9) also becomes significantly more challenging, restricting adaptability across diverse reasoning tasks.

Driven by these insights, we propose a neuro-symbolic framework that integrates Logical Knowledge Graphs (LKG) with external symbolic solvers, treating logical rules and constraints as first-class nodes. This combines the natural language understanding of LLMs for parsing problems into a structured graph with the deterministic power of external solvers for the actual reasoning.

Our main contributions are summarized as follows:

• SymbolLKG Framework: We propose SymbolLKG, a novel neuro-symbolic framework that grounds LLM reasoning via Logical Knowledge Graphs (LKG) and router-guided solvers. The pipeline parses natural language into an LKG, retrieves a topologically connected subgraph, and dynamically routes the problem to the optimal symbolic solver, bridging neural semantic flex-4. Formal Solver Oibility and symbolic deterministic precision.

![](images/ded55ce5f6a34a4ce230123b7c3bdb659df811311d17913b78afede49d2459af.jpg)  
Figure 1: Comparison of reasoning paradigms. The input highlights constraints in green, noise in grey, and premises in orange. The Direct LLM Pipeline (left) ignores this structure. It makes a “Closed World Assumption” (CWA) error and incorrectly guesses Jamie won. The SymbolLKG Pipeline (right) converts text into a Logical Knowledge Graph. It uses symbolic verification to strictly check the premises. The system correctly concludes the answer is “Unknown”.

• Topology-Aware Logical Knowledge Graph: We introduce a dynamic LKG architecture that treats Rules and Constraints as independent nodes, using a “Schema-on-Read” strategy to build Concept-Entity hierarchies from context. This topology enables hybrid retrieval combining vector search with graph traversal, refined via LLM pruning into a Logical Hull—a complete yet minimal subgraph of essential premises.

• Adaptive Logic Routing and Self-Refining Translation: We implement a Logic Router that analyzes problem structure to select the optimal symbolic tool (e.g., Z3 for math, Prover9 for logic), with a feedback loop where the LLM uses solver error traces to refine its generated code, improving reliability on complex logical tasks.

We evaluate our framework on standard logical reasoning and multi-hop QA datasets. Experimental results demonstrate that our approach significantly outperforms standard LLM prompting and baseline RAG methods, validating the efficacy of combining structured knowledge graphs with deterministic solvers for trustworthy reasoning.

## 2 Related Works

Large Language Models for Logical Reasoning. Large Language Models show strong performance in reasoning tasks. Techniques like CoT [7] and Zero-Shot CoT [19] are now standard for generating multi-step reasoning. To improve reliability, methods such as Self-Consistency [20] and ToT [8] explore multiple reasoning paths. Despite these advances, pure neural models still suffer from hallucinations and “unfaithful” explanations, meaning their generated reasons often do not match their actual decision process [21]. Furthermore, Transformers struggle with compositional generalization, failing on multi-hop deduction problems outside their training data [22, 23]. Self-Correction is also difficult because models rarely catch their own logical errors without external feedback [24]. Thi highlights the need for robust external verification mechanisms.

Neuro-Symbolic AI and Auto-formalization. Neuro-symbolic AI combines neural flexibility with symbolic rigor. A key approach is auto-formalization, where LLMs translate text into formal code [25, 10] or logic rules [26]. Systems like Logic-LM [27] and LINC [28] use external solvers to execute these rules. However, most existing systems use a static strategy, assigning one solver (e.g., Z3 or Prover9) to an entire dataset. This rigid method fails on complex queries mixing arithmetic and relational logic. As shown by [29], tool effectiveness depends heavily on the specific structure of each problem. Our SymbolLKG solves this with Dynamic Solver Routing. Instead of using dataset-level rules, our system acts as a controller, analyzing each query at runtime to select the best solver.

Retrieval-Augmented Generation for Structured Knowledge. Retrieval-augmented generation (RAG) reduces hallucinations by using external documents [9, 30]. Standard RAG relies on dense vector similarity to fetch relevant context. However, for logical reasoning tasks, this often leads to “semantic drift”—where the system retrieves semantically similar but logically irrelevant documents. While good for simple facts, standard retrieval fails at multi-hop reasoning, which requires following logical chains rather than just matching similar words. Recent works use Knowledge Graphs to solve this. Methods like Think-on-Graph [14], RoG [15], StructGPT [31] and GraphRAG [16] guide LLMs through graph structures. Others, like Selection-Inference [32], break reasoning into smaller steps. However, these methods usually treat KGs as static lists of entities and ignore complex logical rules. We propose a Logical Knowledge Graph (LKG) that treats rules and constraints as primary nodes. By combining vector search with Logical Hull Expansion, we ensure implicit axioms are retrieved for the symbolic solver, bridging the gap between semantic search and logical deduction.

## 3 Methodology

We propose the Symbolic-Cognitive Logical Knowledge Graph (SymbolLKG) framework, a neurosymbolic architecture (Figure 2) that maps unstructured text into verifiable reasoning paths. The workflow begins at the top with the Logical Knowledge Graph Construction phase, where an LLM parses the natural language corpus to extract entities, concepts, rules, and constraints into a structured LKG. The process flows into the Inference Pipeline, which first employs Hybrid Retrieval and Pruning to isolate a context-aware subgraph containing only relevant premises. Next, an Adaptive Logic Router analyzes the subgraph’s topology to pick the best solver, passing the problem to the Code Generation module for symbolic execution and self-refining verification.

![](images/0edd8618551336a37905edc369ff23c1542f6b5e8b083c30bff050b09b26e72c.jpg)  
Figure 2: The architecture of the SymbolLKG framework. The pipeline consists of two phases: Phase 1 (LKG Construction) transforms text into an LKG, establishing a dynamic Concept-Entity hierarchy alongside reified Rule and Constraint nodes. Phase 2 (Inference) employs a hybrid retrieval strategy (combining vector search with graph traversal) to extract a context-aware subgraph. Then a topology-aware router dispatch it to the optimal symbolic solver for verifiable execution.

## 3.1 Logical Knowledge Graph Construction

Phase 1 begins by processing the natural language corpus through an LLM-based extraction pipeline. As illustrated in the “LKG Construction” panel of Figure 2, we adopt an Open Information Extraction (OpenIE) paradigm to handle the diversity of natural language. We formally define the LKG as a directed, heterogeneous multigraph $G = \mathsf { \bar { ( } } V , E , \mathcal { A , T ) }$ , where V is the set of nodes, E is the set of edges, A is a set of attributes, and T represents the node types. The vertex set V is partitioned into four disjoint subsets, as shown in Table1.

First, we extract Concepts and Entities using a dynamic “Schema-on-Read” strategy. Unlike traditional graphs with rigid categories, our system generates Concept nodes (e.g., “Student”) and Entity nodes $( \mathrm { e . g . , \tilde { \mathrm { A l i c e ^ { 3 } ) } } }$ based on their specific roles in the text. This approach ensures that the graph captures the precise domain of discourse needed for logical reasoning.

Next, we identify logical dependencies and instantiate them as Rule and Constraint nodes. We treat logic as “first-class citizens” rather than simple text attributes. For example, a restriction like $\mathrm { G P A } > 3 . 8$ is extracted as a specific Constraint node, while an implication like “If A then $\mathbf { B } ^ { \ast }$ becomes a Rule node. These nodes are then connected to their respective entities via relation edges.

Finally, these components are assembled into the Logical Knowledge Graph (LKG). To ensure consistency, we assign a unique ID to each node using a hash function. This step merges variations like “The blue book” and “Blue Book” into a single node, preventing duplicates.

This design offers three key advantages. First, it makes logic semantically addressable. Rules get their own vector embeddings $\mathbf { h } _ { v } \in \mathbb { \mathbb { R } } ^ { d }$ , so the system can find “seating restrictions” directly via semantic search. Second, it separates the T-Box (logical rules) from the A-Box (assertions) [33]. This keeps the reasoning structure stable even when specific entity data changes. Third, the Concept node in our framework are extracted directly from the corpus based on their logical role in the specific problem context, rather than mapping to a universal ontology. They define the Domains of Discourse for logical quantifiers (e.g., ∀x ∈ Students), enabling the direct translation of natural language into verifiable First-Order Logic (FOL). This “Schema-on-Read” approach ensures that symbolic solver operate on precise, problem-relevant subsets rather than generic classes.

## 3.2 Hybrid Retrieval and Pruning

After the construction of the LKG, the inference pipeline begins by setting a precise entry point for the user query. As shown in the “Hybrid Retrieval” module (Figure 2), we first perform Query Entity Extraction to ground the search. The system parses the natural language query to find specific target entities (e.g., “Alice” or “Future Leaders Scholarship”) before any retrieval occurs.

These extracted entities, combined with the original query, drive the identification of Anchor Nodes. We employ a dense embedding model (BGE-M3 [34]) to encode the user query and calculate the cosine similarity between query embedding ${ \mathbf { v } } _ { q }$ and all node embeddings. Simultaneously, we perform exact matching against the names of the entities extracted in the previous step. This dual approach selects a set of anchors $S _ { \mathrm { a n c h o r } }$ defined as:

$$
\begin{array} { c } { S _ { \mathrm { a n c h o r } } = \left. v \in V \mid \cos ( \mathbf { v } _ { q } , \mathbf { h } _ { v } ) > \tau _ { s i m } \right. } \\ { \cup \left. v \in V _ { E } \mid \mathrm { e x a c t m a t c h } ( v . n a m e , q ) \right. } \end{array}\tag{1}
$$

This strategy captures both the explicit targets mentioned by the user and the abstract logical concepts relevant to the context.

However, semantic search alone is not enough for logic. A rule constraining an entity often lacks shared keywords with that entity. To fix this, we define the Logical Hull of the anchor set. We expand the context by traversing the graph’s [:MENTIONS], [:IS\_A] and [:APPLIES\_TO] edges. For each anchor $v \in S _ { \mathrm { a n c h o r } } .$ , we include its k-hop (determined by datasets) neighbors from the rule set $V _ { R }$ or constraint set $V _ { S }$

$$
\begin{array} { r l } & { S _ { \mathrm { t m p } } = \{ u \in V _ { R } \cup V _ { S } \mid \exists v \in S _ { \mathrm { a n c h o r } } , \mathrm { d i s t } ( u , v ) \le k \} } \\ & { S _ { \mathrm { h u l l } } = S _ { \mathrm { a n c h o r } } \cup S _ { \mathrm { t m p } } } \end{array}\tag{2}
$$

This step ensures Logical Completeness. If an entity is found, its related rules are automatically added. E.g., finding $\mathrm { \bar { \Psi } A l i c e ^ { , \bullet } }$ also fetches the “AllDifferent” constraint for her group.

Finally, the expanded set $S _ { \mathrm { h u l l } }$ can be large. To fit the solver’s context window, we use an LLMbased pruning module Ψ. This module checks if each node is needed to answer q and removes

<table><tr><td>Node Type</td><td>Symbol</td><td>Definition &amp; Categories (Codebase)</td><td>Key Attributes (Class Fields)</td><td>Role in Reasoning</td></tr><tr><td>Entity</td><td> $V _ { E }$ </td><td>Concrete instances implemented as Entity class. IDs are generated via SHA256 hash of lemmatized name and type.</td><td>name, entity_type, id (Canonical Hash), properties</td><td>Anchors for retrieval. Hash-based IDs ensure uniqueness across the graph.</td></tr><tr><td>Concept</td><td> $V _ { C }$ </td><td>Abstract categories implemented as Concept class.</td><td>name, labels</td><td>Simple taxonomy nodes supporting type grouping.</td></tr><tr><td>Rule</td><td> $V _ { R }$ </td><td>Logical implications implemented as Rule class.</td><td>expression, description, embedding</td><td>Stores logic formulas; description allows vector-based retrieval of rules.</td></tr><tr><td>Constraint</td><td> $V _ { S }$ </td><td>Restrictions on states, implemented as 4 subclasses: 1. Arithmetic (Math ops) 2. AllDifferent (Combinatorial) 3. Ordering (Spatial/Temporal relations like left_of, newer_than) 4. Generic (Constraints outside the above-listed categories)</td><td>raw_expression, constraint_type, Specific: relation, operation, value</td><td>Defines solution boundaries. Ordering handles both topological and chronological constraints via relation types.</td></tr></table>

Table 1: Node Taxonomy in the Symbolic-Cognitive Logical Knowledge Graph (SymbolLKG). The schema distinguishes between static entities, hierarchical concepts, deductive rules, and arithmetic constraints to facilitate dynamic solver routing.

“distractor” constraints. The result is $G _ { \mathrm { f i n a l } } = \Psi ( S _ { \mathrm { h u l l } } , q )$ , a compact, mathematically bounded context for reasoning.

## 3.3 Solver Selection via Logic Router

Given the pruned subgraph $G _ { \mathrm { f i n a l } } .$ , the framework next determines the best strategy to solve the logical problem. As depicted in the “Solver Selection” block of Figure 2, we introduce an Adaptive Logic Router to analyze the topology of the subgraph.

Real-world logical problems vary widely, so using a single solver is often inefficient. Research has shown that while Z3 [35] excels at arithmetic and combinatorial tasks, it is often less intuitive for pure first-order logic compared to Prover9 [29]. To optimize performance, we implement an Adaptive Logic Router. This meta-cognitive module $f _ { \mathrm { r o u t e } } : ( G _ { \mathrm { f i n a l } } , q ) \to \Omega$ analyzes the subgraph’s structure to select the best solver backend.

SMT Solver (Z3) Pathway: Used when $G _ { \mathrm { f i n a l } }$ has many $V _ { S }$ nodes (Constraints), especially for arithmetic or global uniqueness (e.g., CSPs). We leverage Z3’s DPLL(T) algorithm to efficiently prune large search spaces.

Automated Theorem Prover (Prover9) Pathway: Selected for strict FOL tasks or proof generation.   
This runs when the subgraph is mostly $V _ { R }$ nodes (Rules) with logical implications but no numbers.

Pyke Pathway: Used for direct relational queries (e.g., “Who is X’s father?”). Here, lightweight engines like Pyke are faster and more efficient than heavy solvers.

This adaptive approach ensures robustness. By categorizing the problem, we stop the LLM from trying unreliable arithmetic through token prediction—a known Transformer weakness. Instead, we force the system to delegate computation to symbolic engines designed for mathematical correctness.

## 3.4 Code Generation

Once the optimal solver is determined, the final phase translates the logical subgraph into executable symbolic code. The LLM acts as a compiler, mapping the nodes in $\breve { G } _ { \mathrm { f i n a l } }$ directly to solver-specific syntax. For instance, in the Z3 pathway, entities $e \in V _ { E }$ become variables $( x = \operatorname { I n t } ( x ) )$ ; concept nodes $V _ { C }$ set domains (solver.add $( 0 \leq x \leq 1 0 0 ) )$ ; and constraint nodes $v _ { c } \in V _ { S }$ become assertions (solver.add(Distinct(x, y, z))).

To ensure reliability, we use a self-refining execution loop. As shown in the bottom-right of Figure 2, the system captures any error messages during execution. It then feeds this feedback back to the model to refine the code. This cycle repeats until success or timeout.

This proposed methodology effectively transforms the stochastic nature of probabilistic text generation into the precision of deterministic computation. Consequently, the resulting output constitutes a rigorous formal proof or a logically valid assignment, delivering a level of interpretability and systemic trust that remains fundamentally unattainable through standard neural Chain-of-Thought prompting alone.

## 4 Experiments

## 4.1 Experimental Settings

Datasets To comprehensively evaluate the SymbolLKG framework, we utilize eight diverse benchmarks that span the spectrum from formal symbolic logic to open-domain multi-hop reasoning. We categorize them into two groups based on the evaluation focus:

Logical Reasoning Benchmarks These datasets validate translation precision and deductive validity. FOLIO [36] provides premises with FOL ground truths and complex logical structures. AR-LSAT [37] focuses on analytical reasoning and constraint satisfaction from law exams. ProofWriter [38] assesses rule-based reasoning over multi-hop chains and proof generation. LogicalDeduction [39] uses constraint satisfaction problems for abstract reasoning, while ProntoQA [40] evaluates transitive reasoning via synthetic multi-hop deductive chains.

Multi-hop Retrieval & QA Benchmarks These benchmarks assess the robustness of LKG Construction and Hybrid Retrieval. 2WikiMultiHopQA [41] requires information aggregation across multiple articles with reasoning chains up to 4 hops. HotpotQA [42] tests noise filtration and bridgeentity identification by requiring the system to find supporting facts among irrelevant distractors. Musique [43] challenges models with highly compositional questions designed to systematically minimize disconnected reasoning shortcuts.

Implementation Details Our SymbolLKG framework utilizes Llama-3.3-70B-Instruct as the backbone LLM for the extraction and translation modules due to its favorable balance between economic efficiency and inference speed. To ensure a fair comparison, we consistently applied Llama-3.3-70B-Instruct as the backbone model for our primary baseline methods, including IRCoT + HippoRAG. To minimize generation randomness and ensure reproducibility, we set the temperature to 0 for all LLM inference calls. For the symbolic solvers, we employ a hybrid strategy: Z3 is used for constraint satisfaction and arithmetic tasks, Prover9 handles strict first-order logic theorem proving, and Pyke is utilized as a lightweight inference engine for direct relational queries. The hybrid retrieval combines dense embeddings (using BGE-M3 [34]) with our topology-aware graph traversal.

## 4.2 Logical Reasoning Performance

Baselines We compare our framework against two representative settings: Standard LLM with CoT: Uses LLM with Chain-of-Thought prompting to establish a strong proprietary model baseline. Logic-LM: A neuro-symbolic framework that integrates LLMs with symbolic solvers, serving as a direct competitor in the symbolic reasoning domain.

<table><tr><td>DATASET</td><td>CoT</td><td>LOGIC-LM</td><td>OURS</td></tr><tr><td>FOLIO</td><td>70.58</td><td>78.92</td><td>71.39</td></tr><tr><td>AR-LSAT</td><td>35.06</td><td>43.04</td><td>57.85</td></tr><tr><td>LOGICAL DEDUCTION</td><td>75.25</td><td>87.63</td><td>90.81</td></tr><tr><td>PRONTOQA</td><td>98.79</td><td>83.20</td><td>100.00</td></tr><tr><td>PROOFWRITER</td><td>68.11</td><td>79.66</td><td>73.60</td></tr><tr><td>AVG.</td><td>69.56</td><td>74.49</td><td>78.73</td></tr></table>

Table 2: Logical Reasoning Accuracy (%) across five benchmarks. Methods compared include Standard CoT prompting and the neuro-symbolic baseline Logic-LM.

Results Table 2 summarizes the accuracy across five benchmarks. SymbolLKG achieves an average of 78.73%, outperforming Logic-LM (74.49%) and Standard CoT (69.56%). Two design choices drive its strongest cells: the typed constraint subclasses (Ordering / AllDifferent / Arithmetic) feed a structural signal to the Logic Router, deterministically selecting Z3 on AR-LSAT and yielding 57.85%; and the three-attempt self-refining loop, which echoes solver compiler errors back to the LLM, completes ProntoQA’s transitive Horn chains at 100.00%. SymbolLKG lags Logic-LM on FOLIO and ProofWriter because Logic-LM relies on dataset-specific prompts, whereas our generalist ontology often demotes free-form quantified premises to Generic rules, weakening the router signal and limiting the recovery scope of compiler-trace feedback.

## 4.3 Multi-hop Retrieval Performance

We evaluate the retrieval effectiveness of our system on two challenging multi-hop datasets: 2Wiki-MultiHopQA, HotpotQA and Musique. This experiment tests the system’s ability to identify and retrieve disjointed pieces of evidence required for multi-step reasoning.

Baselines We categorize our baselines into Single-step and Multi-step retrieval approaches to provide a comprehensive comparison: Single-step Retrieval: We include sparse retrieval (BM25 [44]), dense retrieval (Contriever [45], GTR [46], NativeRAG [9]), and advanced structure-aware methods (RAPTOR [47], Proposition [48], HippoRAG [49]). Multi-step Retrieval: We evaluate IRCoT [50] (Interleaving Retrieval with Chain-of-Thought) combined with four different retrievers: BM25, Contriever, NativeRAG, and HippoRAG. This setup allows us to compare our topology-aware retrieval against iterative feedback loops powered by different backbone retrievers.

Metrics We report Recall@k, defined as the proportion of ground-truth supporting documents found within the top-k retrieved candidates. Given the ground-truth supporting fact set $G _ { q }$ and the top-k retrieved set $R _ { q , k }$ for a query q, the recall is calculated as:

$$
\operatorname { R e c a l l } @ k = \frac { 1 } { | Q | } \sum _ { q \in Q } \frac { | R _ { q , k } \cap G _ { q } | } { | G _ { q } | }\tag{3}
$$

We use Recall@2 and Recall@5, which measure the percentage of queries where the top-2 and top-5 retrieved documents contain the complete set of ground-truth supporting facts.

<table><tr><td rowspan="2">TYPE</td><td rowspan="2">METHOD</td><td colspan="2">2WIKIMULTIHOPQA</td><td colspan="2">HOTPOTQA</td></tr><tr><td>RECALL@2</td><td>RECALL@5</td><td>RECALL@2</td><td>RECALL@5</td></tr><tr><td rowspan="7">SINGLE-STEP</td><td>BM25</td><td>51.8</td><td>61.9</td><td>55.4</td><td>72.2</td></tr><tr><td>CONTRIEVER</td><td>46.6</td><td>57.5</td><td>57.2</td><td>75.5</td></tr><tr><td>GTR</td><td>60.2</td><td>67.9</td><td>59.4</td><td>73.3</td></tr><tr><td>RAPTOR</td><td>46.3</td><td>53.8</td><td>58.1</td><td>71.2</td></tr><tr><td>PROPOSITION</td><td>56.4</td><td>63.1</td><td>58.7</td><td>71.1</td></tr><tr><td>NATIVERAG</td><td>59.2</td><td>68.2</td><td>64.7</td><td>79.3</td></tr><tr><td>HIPPORAG</td><td>70.7</td><td>89.1</td><td>60.5</td><td>77.7</td></tr><tr><td rowspan="4">MULTI-STEP</td><td>IRCoT + BM25</td><td>55.4</td><td>66.8</td><td>62.1</td><td>75.3</td></tr><tr><td>IRCOT + CONTRIEVER</td><td>51.6</td><td>63.8</td><td>65.9</td><td>81.6</td></tr><tr><td>IRCOT + NATIVERAG</td><td>64.1</td><td>74.4</td><td>67.9</td><td>82.0</td></tr><tr><td>IRCOT + HIPPORAG</td><td>75.3</td><td>93.4</td><td>67.0</td><td>83.0</td></tr><tr><td>OURS</td><td>SYMBOLLKG</td><td>79.4</td><td>88.2</td><td>68.5</td><td>84.1</td></tr></table>

Table 3: Retrieval Performance (Recall@2 and Recall@5) on 2WikiMultiHopQA and HotpotQA. Baselines are categorized into Single-step and Multi-step approaches. SymbolLKG leverages topology-aware traversal to optimize evidence retrieval.

Results Table 3 presents the retrieval results. SymbolLKG attains 79.4 / 88.2 on 2WikiMultiHopQA and 68.5 / 84.1 on HotpotQA. The Recall@2 lead is driven by hybrid anchoring: three independently thresholded vector queries (over Rule, Constraint and Entity indices) are unioned with a BM25 entity-name pass, so the second slot is reliably occupied either by an entity-string hit or by a rule whose raw\_text contains the bridge. The HotpotQA gain further reflects the Logical Hull: two-hop traversal along [:MENTIONS], [:APPLIES\_TO] and typed predicate edges surfaces distractor-buried supporting facts that share no surface tokens with the query. The single trail, R@5 on 2Wiki, is consistent with our single-pass anchor-and-prune strategy, which maximises top-rank precision but cannot re-introduce evidence rejected by the initial threshold, unlike IRCoT’s iterative re-querying.

## 4.4 Question Answering Performance

Finally, we assess the end-to-end question answering capabilities on 2WikiMultiHopQA and HotpotQA. This evaluation determines whether the retrieved context effectively supports the generation of accurate answers.

Baselines We benchmark against three primary RAG frameworks: NativeRAG: A standard retrieval-augmented generation approach using dense retrieval. GraphRAG [16]: A typical graphenhanced RAG framework with structured knowledge modeling. HippoRAG 2 [51]: A state-of-theart graph-based RAG method inspired by the hippocampal indexing theory.
<table><tr><td rowspan="2">Method</td><td colspan="2">2Wiki</td><td colspan="2">HotpotQA</td><td colspan="2">Musique</td><td colspan="2">Average</td></tr><tr><td>EM F1</td><td></td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td></tr><tr><td>NativeRAG</td><td>33.4</td><td>43.3</td><td>43.4</td><td>57.7</td><td>15.5</td><td>26.4</td><td>30.8</td><td>42.5</td></tr><tr><td>GraphRAG</td><td>51.4</td><td>58.6</td><td>55.2</td><td>68.6</td><td>27.3</td><td>38.5</td><td>44.6</td><td>55.2</td></tr><tr><td>IRCoT + HippoRAG</td><td>47.7</td><td>62.7</td><td>45.7</td><td>59.2</td><td>21.9</td><td>33.3</td><td>38.4</td><td>51.7</td></tr><tr><td>HippoRAG 2</td><td>65.0</td><td>71.0</td><td>62.7</td><td>75.5</td><td>37.2</td><td>48.6</td><td>55.0</td><td>65.0</td></tr><tr><td>SymbolLKG</td><td>70.2</td><td>74.9</td><td>73.8</td><td>81.5</td><td>48.6</td><td>59.4</td><td>64.2</td><td>71.9</td></tr></table>

Table 4: End-to-end QA Performance (EM and F1). We compare our SymbolLKG against baselines on 2WikiMultiHopQA, HotpotQA and Musique.

Metrics We utilize Exact Match (EM) and F1 Score. EM measures whether the predicted answer aˆ strictly matches the ground truth $a ^ { * }$ after normalization. F1 Score evaluates the token-level overlap. Let $T ( \cdot )$ denote the bag of tokens function; precision (P) and recall (R) are computed as:

$$
P = \frac { | T ( \hat { a } ) \cap T ( a ^ { * } ) | } { | T ( \hat { a } ) | } , \quad R = \frac { | T ( \hat { a } ) \cap T ( a ^ { * } ) | } { | T ( a ^ { * } ) | }\tag{4}
$$

The F1 score is the harmonic mean:

$$
\mathrm { F 1 } = 2 \cdot ( P \cdot R ) / ( P + R )\tag{5}
$$

Results Table 4 summarizes EM and F1 on the three multi-hop QA datasets. SymbolLKG attains 70.2 / 74.9 on 2Wiki, 73.8 / 81.5 on HotpotQA and 48.6 / 59.4 on MuSiQue, leading every prior baseline on every cell. The largest gain on MuSiQue reflects two complementary mechanisms: hash-based entity canonicalization merges paraphrased mentions of the same bridge entity (e.g., “The Beatles” and “the band Beatles”) into a single graph node, so a bridge cannot be silently fragmented across single-hop sub-facts; meanwhile, the Logical Hull expansion admits rules and constraints two hops away from a question anchor even when they share no surface tokens with the query. The +6.0 F1 on HotpotQA reflects the same mechanism on noise rather than on composition: the topologically bounded subgraph isolates anchor-connected facts among 8–10 distractor paragraphs and is appended alongside the raw context, acting as a high-precision prior over which titles the answer LLM should attend to. The narrower gap on 2Wiki indicates that on cleaner Wikipedia text, dense retrieval baselines already cover most evidence, and the LKG’s marginal contribution is mainly in entity disambiguation across same-surname or same-topic candidates.

## 4.5 Router Performance

Setup To isolate the contribution of the Logic Router, we sample 100 instances from each of the five logical benchmarks (N = 500 in total) and shuffle them into a single mixed evaluation stream. Each instance is processed end-to-end (LKG construction → hybrid retrieval → router decision) using Llama-3.3-70B-Instruct as the routing LLM. The ground-truth solver per dataset follows the design intent of the three engines: Z3 for AR-LSAT and LogicalDeduction (CSP / ordering / arithmetic); Prover9 for FOLIO (FOL with true/false/unknown) and ProofWriter, and Pyke for ProntoQA (Horn-clause forward chaining).

Results Table 5 reports the per-dataset routing accuracy. SymbolLKG’s router reaches an overall accuracy of 86.0%, more than 2.5× the random baseline. Routing is essentially perfect on the two

Z3-targeted datasets (100.0% on both AR-LSAT and LogicalDeduction), because the typed constraint subclasses appear directly in the retrieved subgraph and trigger the router’s first decision rule. Routing is also strong on the two Prover9-targeted datasets (85.0% on FOLIO, 92.0% on ProofWriter), where the three-valued cue “true, false, or uncertain/unknown” in the question text reliably steers the router to Prover9. The remaining error mass concentrates on ProntoQA (53.0%), where the dominant confusion is Pyke-vs-Prover9: ProntoQA’s “true or false” phrasing is closed-world but lexically close enough to a Prover9 trigger to occasionally mislead the router. This residual confusion is largely benign in practice, however, because both engines support Horn-clause forward chaining—an instance routed from Pyke to Prover9 in this regime is still solved correctly more often than not.

Table 5: Router classification accuracy (%) on a mixed evaluation set of 500 instances (100 per dataset). The ground-truth solver per dataset is determined by the design intent of each symbolic engine.
<table><tr><td>Dataset</td><td>Ground-Truth Solver</td><td>Random</td><td>SymbolLKG</td></tr><tr><td>AR-LSAT</td><td>Z3</td><td>33.3</td><td>100.0</td></tr><tr><td>LogicalDeduction</td><td>Z3</td><td>33.3</td><td>100.0</td></tr><tr><td>FOLIO</td><td>PROVER9</td><td>33.3</td><td>85.0</td></tr><tr><td>ProofWriter</td><td>PROVER9</td><td>33.3</td><td>92.0</td></tr><tr><td>ProntoQA</td><td>PYKE</td><td>33.3</td><td>53.0</td></tr><tr><td>Average</td><td>一</td><td>33.3</td><td>86.0</td></tr></table>

## 4.6 Computational Cost Analysis

On N = 100 samples per benchmark with Llama-3.3-70B-Instruct, the full SymbolLKG pipeline averages 122.2 s per logical reasoning problem and 166–407 s per multi-hop QA query, dominated by LLM API calls (> 99% of wall-clock time). For deployments with fixed corpora the LKG is built once offline and reused across queries, reducing inference-time latency to retrieval (< 0.1 s) plus a single answer-generation call (∼10 s). Full per-stage breakdowns, distributions, and theoretical complexity are provided in Appendix C.

## 5 Limitations & Conclusion

In this paper, we presented SymbolLKG, a neuro-symbolic framework that bridges the gap between the semantic flexibility of LLMs and the rigorous precision of symbolic solvers. By treating logical rules as topological nodes within a Logical Knowledge Graph and employing an adaptive router to dispatch tasks to the most appropriate engine (Z3, Prover9, or Pyke), our system significantly outperforms standard prompting and retrieval baselines on complex reasoning benchmarks. The core advantage of SymbolLKG lies in its verifiability: unlike the opaque generation process of Chain-of-Thought, our framework provides deterministic proof traces, ensuring that the reasoning process is both transparent and logically sound.

However, several limitations warrant further investigation. First is the translation bottleneck. Our framework operates on the premise that the LLM can accurately parse natural language into formal graph representations. While the self-refining loop mitigates syntax errors, semantic misinterpretations during the extraction phase can still propagate to the solver. Essentially, the symbolic engine guarantees validity (the structure of the argument) but not soundness (the truth of the premises) if the initial grounding is flawed. Second, the handling of ambiguity remains a challenge. Formal logic requires precise definitions, whereas natural language is inherently ambiguous. When the input text contains vague quantifiers or metaphorical expressions, the rigid schema of the LKG may struggle to capture the full nuance, potentially leading to oversimplification. Finally, the introduction of graph construction and external solvers incurs additional computational overhead. While more accurate, our pipeline is computationally heavier than direct single-pass generation. Future work will focus on optimizing the extraction latency and exploring end-to-end differentiable neuro-symbolic alignment to reduce the dependency on intermediate discrete translation.

## References

[1] Arun James Thirunavukarasu, Darren Shu Jeng Ting, Kabilan Elangovan, Laura Gutierrez, Ting Fang Tan, and Daniel Shu Wei Ting. Large language models in medicine. Nature medicine, 29(8):1930–1940, 2023.

[2] Daniel Martin Katz, Michael James Bommarito, Shang Gao, and Pablo Arredondo. Gpt-4 passes the bar exam. Philosophical Transactions ofthe Royal Society A, 382(2270):20230254, 2024.

[3] Shijie Wu, Ozan Irsoy, Steven Lu, Vadim Dabravolski, Mark Dredze, Sebastian Gehrmann, Prabhanjan Kambadur, David Rosenberg, and Gideon Mann. Bloomberggpt: A large language model for finance. arXiv preprint arXiv:2303.17564, 2023.

[4] Hanchen Wang, Tianfan Fu, Yuanqi Du, Wenhao Gao, Kexin Huang, Ziming Liu, Payal Chandak, Shengchao Liu, Peter Van Katwyk, Andreea Deac, et al. Scientific discovery in the age of artificial intelligence. Nature, 620(7972):47–60, 2023.

[5] Avinash Patil and Aryan Jadon. Advancing reasoning in large language models: Promising methods and approaches. arXiv preprint arXiv:2502.03671, 2025.

[6] Karthik Valmeekam, Matthew Marquez, Sarath Sreedharan, and Subbarao Kambhampati. On the planning abilities of large language models-a critical investigation. Advances in Neural Information Processing Systems, 36:75993–76005, 2023.

[7] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

[8] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. Advances in neural information processing systems, 36:11809–11822, 2023.

[9] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in neural information processing systems, 33:9459–9474, 2020.

[10] Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig. Pal: Program-aided language models. In International Conference on Machine Learning, pages 10764–10799. PMLR, 2023.

[11] Aidan Hogan, Eva Blomqvist, Michael Cochez, Claudia d’Amato, Gerard De Melo, Claudio Gutierrez, Sabrina Kirrane, José Emilio Labra Gayo, Roberto Navigli, Sebastian Neumaier, et al. Knowledge graphs. ACM Computing Surveys (Csur), 54(4):1–37, 2021.

[12] Shaoxiong Ji, Shirui Pan, Erik Cambria, Pekka Marttinen, and Philip S Yu. A survey on knowledge graphs: Representation, acquisition, and applications. IEEE transactions on neural networks and learning systems, 33(2):494–514, 2021.

[13] Shirui Pan, Linhao Luo, Yufei Wang, Chen Chen, Jiapu Wang, and Xindong Wu. Unifying large language models and knowledge graphs: A roadmap. IEEE Transactions on Knowledge and Data Engineering, 36(7):3580–3599, 2024.

[14] Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Saizhuo Wang, Chen Lin, Yeyun Gong, Lionel M Ni, Heung-Yeung Shum, and Jian Guo. Think-on-graph: Deep and responsible reasoning of large language model on knowledge graph. arXiv preprint arXiv:2307.07697, 2023.

[15] Linhao Luo, Yuan-Fang Li, Gholamreza Haffari, and Shirui Pan. Reasoning on graphs: Faithful and interpretable large language model reasoning. arXiv preprint arXiv:2310.01061, 2023.

[16] Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan Larson. From local to global: A graph rag approach to query-focused summarization. arXiv preprint arXiv:2404.16130, 2024.

[17] Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. Lightrag: Simple and fast retrieval-augmented generation. arXiv preprint arXiv:2410.05779, 2024.

[18] Lei Liang, Zhongpu Bo, Zhengke Gui, Zhongshu Zhu, Ling Zhong, Peilong Zhao, Mengshu Sun, Zhiqiang Zhang, Jun Zhou, Wenguang Chen, et al. Kag: Boosting llms in professional domains via knowledge augmented generation. In Companion Proceedings ofthe ACM on Web Conference 2025, pages 334–343, 2025.

[19] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. Advances in neural information processing systems, 35:22199–22213, 2022.

[20] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171, 2022.

[21] Miles Turpin, Julian Michael, Ethan Perez, and Samuel Bowman. Language models don’t always say what they think: Unfaithful explanations in chain-of-thought prompting. Advances in Neural Information Processing Systems, 36:74952–74965, 2023.

[22] Nouha Dziri, Ximing Lu, Melanie Sclar, Xiang Lorraine Li, Liwei Jiang, Bill Yuchen Lin, Sean Welleck, Peter West, Chandra Bhagavatula, Ronan Le Bras, et al. Faith and fate: Limits of transformers on compositionality. Advances in Neural Information Processing Systems, 36:70293–70332, 2023.

[23] Maxwell Nye, Anders Johan Andreassen, Guy Gur-Ari, Henryk Michalewski, Jacob Austin, David Bieber, David Dohan, Aitor Lewkowycz, Maarten Bosma, David Luan, et al. Show your work: Scratchpads for intermediate computation with language models. arXiv preprint arXiv:2105.05246, 2021.

[24] Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, Adams Wei Yu, Xinying Song, and Denny Zhou. Large language models cannot self-correct reasoning yet. arXiv preprint arXiv:2310.01798, 2023.

[25] Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W Cohen. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. arXiv preprint arXiv:2211.12588, 2022.

[26] Qing Lyu, Shreya Havaldar, Adam Stein, Li Zhang, Delip Rao, Eric Wong, Marianna Apidianaki, and Chris Callison-Burch. Faithful chain-of-thought reasoning. In The 13th International Joint Conference on Natural Language Processing and the 3rd Conference ofthe Asia-Pacific Chapter ofthe Associationfor Computational Linguistics (IJCNLP-AACL 2023), 2023.

[27] Liangming Pan, Alon Albalak, Xinyi Wang, and William Wang. Logic-lm: Empowering large language models with symbolic solvers for faithful logical reasoning. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 3806–3824, 2023.

[28] Theo Olausson, Alex Gu, Ben Lipkin, Cedegao Zhang, Armando Solar-Lezama, Joshua Tenenbaum, and Roger Levy. Linc: A neurosymbolic approach for logical reasoning by combining language models with first-order logic provers. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 5153–5176, 2023.

[29] Long Hei Matthew Lam, Ramya Keerthy Thatikonda, and Ehsan Shareghi. A closer look at logical reasoning with llms: The choice of tool matters. arXiv preprint arXiv:2406.00284, 2024.

[30] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang. Retrieval augmented language model pre-training. In International conference on machine learning, pages 3929–3938. PMLR, 2020.

[31] Jinhao Jiang, Kun Zhou, Zican Dong, Keming Ye, Wayne Xin Zhao, and Ji-Rong Wen. Structgpt: A general framework for large language model to reason over structured data. arXiv preprint arXiv:2305.09645, 2023.

[32] Antonia Creswell, Murray Shanahan, and Irina Higgins. Selection-inference: Exploiting large language models for interpretable logical reasoning. arXiv preprint arXiv:2205.09712, 2022.

[33] Franz Baader. The description logic handbook: Theory, implementation and applications. Cambridge university press, 2003.

[34] J. Chen, S. Xiao, P. Zhang, K. Luo, D. Lian, and Z. Liu. Bge m3-embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. arXiv preprint arXiv:2402.03216, 4(5), 2024.

[35] Leonardo De Moura and Nikolaj Bjørner. Z3: An efficient smt solver. In International conference on Tools and Algorithms for the Construction and Analysis of Systems, pages 337–340. Springer, 2008.

[36] Simeng Han, Hailey Schoelkopf, Yilun Zhao, Zhenting Qi, Martin Riddell, Wenfei Zhou, James Coady, David Peng, Yujie Qiao, Luke Benson, et al. Folio: Natural language reasoning with first-order logic. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 22017–22031, 2024.

[37] Wanjun Zhong, Siyuan Wang, Duyu Tang, Zenan Xu, Daya Guo, Yining Chen, Jiahai Wang, Jian Yin, Ming Zhou, and Nan Duan. Analytical reasoning of text. In Findings ofthe Association for Computational Linguistics: NAACL 2022, pages 2306–2319, 2022.

[38] Oyvind Tafjord, Bhavana Dalvi, and Peter Clark. Proofwriter: Generating implications, proofs, and abductive statements over natural language. In Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, pages 3621–3634, 2021.

[39] Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch, Adam R Brown, Adam Santoro, Aditya Gupta, Adrià Garriga-Alonso, et al. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. Transactions on machine learning research, 2023.

[40] Abulhair Saparov and He He. Language models are greedy reasoners: A systematic formal analysis of chain-of-thought. arXiv preprint arXiv:2210.01240, 2022.

[41] Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. Constructing a multi-hop qa dataset for comprehensive evaluation of reasoning steps. arXiv preprint arXiv:2011.01060, 2020.

[42] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. In Proceedings ofthe 2018 conference on empirical methods in natural language processing, pages 2369–2380, 2018.

[43] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Musique: Multihop questions via single-hop question composition. Transactions ofthe Associationfor Computational Linguistics, 10:539–554, 2022.

[44] Stephen E Robertson and Steve Walker. Some simple effective approximations to the 2-poisson model for probabilistic weighted retrieval. In SIGIR’94: Proceedings ofthe Seventeenth Annual International ACM-SIGIR Conference on Research and Development in Information Retrieval, organised by Dublin City University, pages 232–241. Springer, 1994.

[45] Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, and Edouard Grave. Unsupervised dense information retrieval with contrastive learning. arXiv preprint arXiv:2112.09118, 2021.

[46] Jianmo Ni, Chen Qu, Jing Lu, Zhuyun Dai, Gustavo Hernandez Abrego, Ji Ma, Vincent Zhao, Yi Luan, Keith Hall, Ming-Wei Chang, et al. Large dual encoders are generalizable retrievers. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 9844–9855, 2022.

[47] Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and Christopher D Manning. Raptor: Recursive abstractive processing for tree-organized retrieval. In The Twelfth International Conference on Learning Representations, 2024.

[48] Tong Chen, Hongwei Wang, Sihao Chen, Wenhao Yu, Kaixin Ma, Xinran Zhao, Hongming Zhang, and Dong Yu. Dense x retrieval: What retrieval granularity should we use? In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing, pages 15159–15177, 2024.

[49] Bernal Jimenez Gutierrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su. Hipporag: Neurobiologically inspired long-term memory for large language models. Advances in Neural Information Processing Systems, 37:59532–59569, 2024.

[50] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Interleaving retrieval with chain-of-thought reasoning for knowledge-intensive multi-step questions. In Proceedings of the 61st annual meeting of the association for computational linguistics (volume 1: long papers), pages 10014–10037, 2023.

[51] Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi, Sizhe Zhou, and Yu Su. From rag to memory: Non-parametric continual learning for large language models. arXiv preprint arXiv:2502.14802, 2025.

## A Prompt Templates

Appendix A provides the specific prompt templates utilized within the SymbolLKG framework to facilitate reproducibility and illustrate the detailed interaction mechanisms between the system and the Large Language Model.

## A.1 Prompt for LKG Construction

![](images/b73a6a18d01bdcd8d175781524ca17d5ebfe04cd78dd68e85cd149f665a6d66a.jpg)  
Figure 3: Prompt template for Logical Knowledge Graph Construction. This prompt instructs the LLM to extract structured logic from text based on a dynamic ontology. It defines specific schemas for Entities, Concepts, and logical relationships (Rules/Constraints), enforcing a strict JSON output format to ensure graph integrity.

![](images/b847a6d690d700235d0336b46cdd0a7132ac07e8cf2ca9fb66f55cc5f2f8f6e6.jpg)  
Figure 4: Prompt template for the Symbolic Solver (Z3) Code Generation. This prompt guides the model to translate the retrieved logical subgraph into an executable Python script utilizing the Z3 SMT solver. It enforces a structured coding process: declaring variables for entities, encoding facts as constraints, and formulating the specific query logic to verify the answer.

## A.3 Prompt for Pruner

Task Description: Your task is to identify which of the provided Graph Nodes are relevant to answering the specific Query.   
Criteria for Relevance:   
1. Direct Mention: The node contains entities or attributes explicitly mentioned in the query.   
2. Logical Dependency: The node represents a rule, constraint, or relationship that logically affects the entities in the query (e.g., "All men are mortal" is   
relevant to "Is Socrates mortal?").   
3. Contextual Isolation: STRICTLY EXCLUDE nodes from unrelated scenarios (e.g., rules about "fruit pricing" are irrelevant to a "family tree" query).   
Example:   
Query: "Determine the seating arrangement of Alice."   
Candidates: [{"ID": "node\_1", "Type": "Entity", "Content": "Alice"}, {"ID": "node\_2", "Type": "Rule", "Content": " If Person A sits next to Person B,   
they cannot be enemies ."},…]   
Relevant IDs: ["node\_1", "node\_2", "node\_4"]   
(Reasoning: 'Alice' and 'Bob' are people involved in seating. The 'Rule' affects seating logic. The 'Constraint' about a car is completely irrelevant.)   
Your Task:   
Query: "{query}"   
Candidates:   
{candidates\_text}   
Output Format:   
Return strictly a JSON list of the IDs of the relevant nodes.   
Do not explain. Return ONLY the JSON.  
Figure 5: Prompt template for the Logic Pruning Module. This prompt instructs the LLM to filter the expanded candidate set.

## A.4 Prompt for Router

![](images/fba1317b56b834bd3efc3767feb2a870ee4f79b9374d8c56bb0a891e1a23f44e.jpg)  
Figure 6: Prompt template for the Adaptive Logic Router. This prompt guides the model to classify the reasoning task by evaluating the retrieved constraints and rules. It explicitly defines the decision boundaries for solver selection: Z3 is chosen for arithmetic and constraint satisfaction problems (CSPs), Prover9 for first-order logi theorem proving, and Pyke for knowledge-based relational inference

## B Case Study

Appendix B presents a comprehensive case study derived from the AR-LSAT benchmark to demonstrate the end-to-end workflow of the SymbolLKG framework. It demonstrates the complete lifecycle of a logical reasoning task, detailing the intermediate outputs at each stage: from the initial extraction of the Logical Knowledge Graph (LKG) components (Entities, Concepts, Rules, and Constraints) to the dynamic inference process involving hybrid retrieval, logic routing, and the final generation of executable Z3 symbolic code.

![](images/1c43db6dac1cb82993da13f2471ed108f6a5c1f19379bd182e32859e4ccfd552.jpg)  
Figure 7: Case Study: Logical Knowledge Graph Extraction Result. The figure displays the structured JSON output extracted from an AR-LSAT logical reasoning problem. It demonstrates how the system identifies entities (e.g., Office, Computer), maps concepts, and formalizes complex constraints (e.g., “AllDifferent”, “Ordering”) and logical rules from the raw text context.

![](images/30e983213006c2e27f0abeaa5365ccdde960e0198db5524170eb5e493a43ad2c.jpg)  
Figure 8: Case Study: Inference Pipeline Log. This log trace illustrates the step-by-step execution flow: (1) Hybrid Search retrieves relevant rules regarding “Year” and “Office”; (2) Pruning filters candidates down to the essential logic; (3) The Logic Router correctly identifies the task as an arithmetic/ordering problem and selects the Z3 Solver; (4) The system generates and executes valid Z3 code to derive the final answer.

## C Time Complexity Analysis

## C.1 Empirical Latency

We profile per-sample wall-clock time on N = 100 instances per benchmark.

Multi-hop QA Table 6 reports the per-sample latency of the full SymbolLKG pipeline on the three multi-hop QA datasets. End-to-end latency ranges from 166.0 s on 2WikiMultiHopQA to 407.1 s on MuSiQue, with the bulk of the cost (159.2–395.8 s) spent on per-question LKG construction. The LKG itself triggers entity, rule, and constraint extraction over the question’s 8–10 paragraph context; the answer-generation call afterwards takes only 6.0–11.4 s.

Table 6: Per-sample wall-clock latency (seconds) on the multi-hop QA pipeline, averaged over N = 100 instances per benchmark.
<table><tr><td>Dataset</td><td>LKG</td><td>LLM</td><td>Total</td></tr><tr><td>2Wiki</td><td>159.2</td><td>6.79</td><td>165.96</td></tr><tr><td>HotpotQA</td><td>208.5</td><td>6.00</td><td>214.57</td></tr><tr><td>MuSiQue</td><td>395.8</td><td>11.35</td><td>407.14</td></tr></table>

Logical Reasoning Table 7 reports the per-sample latency of the logical reasoning pipeline (singlepath: LKG construction plus the routed-solver path including self-refining retries). End-to-end latency averages 122.2 s per problem (LKG construction 29.0 s, routed solver 93.2 s). ProofWriter and AR-LSAT exhibit the longest mean times (166.5 s and 135.6 s respectively), reflecting their longer Prover9 / Z3 programs and higher retry rates; ProntoQA is shortest (71.4 s) thanks to the lightweight Pyke forward-chaining path.

Table 7: Per-sample wall-clock latency (seconds) on the logical reasoning pipeline, averaged over 20 instances per dataset $( N = \mathrm { \bar { 1 0 0 } }$ in total). “Total” reflects single-path production deployment (LKG construction plus the routed-solver path with up to three self-refining retries).
<table><tr><td>Dataset</td><td>LKG</td><td>Solver</td><td>Total</td></tr><tr><td>AR-LSAT</td><td>33.1</td><td>102.5</td><td>135.6</td></tr><tr><td>LogicalDeduction</td><td>18.9</td><td>103.3</td><td>122.2</td></tr><tr><td>FOLIO</td><td>23.4</td><td>91.8</td><td>115.2</td></tr><tr><td>ProntoQA</td><td>34.3</td><td>37.1</td><td>71.4</td></tr><tr><td>ProofWriter</td><td>35.1</td><td>131.4</td><td>166.5</td></tr><tr><td>Average</td><td>29.0</td><td>93.2</td><td>122.2</td></tr></table>

## C.2 Discussion

The dominant cost is LLM API latency rather than graph operations: across both pipelines, the two recurring LLM calls (LKG extraction plus solver code generation or answer generation) jointly account for over 99% of wall-clock time, while retrieval and BFS combined remain below 0.1 s per query. Three properties of the framework limit the practical impact of this overhead. (i) For fixed corpora served at deployment time, the LKG can be built once offline and reused across all queries, eliminating the 155–494 s per-question construction cost that dominates the multi-hop QA numbers; under this amortisation, the inference-time cost reduces to retrieval (<0.1 s) plus a single answer-generation call (∼10 s). (ii) The self-refining loop is bounded at three retries, so worst-case solver latency is at most 3× the single-attempt cost. (iii) The routing and pruning calls operate on bounded-length prompts and can be served by a smaller, cheaper backbone without materially affecting accuracy. The remaining overhead reflects the cost of trading probabilistic single-pass inference for verifiable symbolic reasoning.