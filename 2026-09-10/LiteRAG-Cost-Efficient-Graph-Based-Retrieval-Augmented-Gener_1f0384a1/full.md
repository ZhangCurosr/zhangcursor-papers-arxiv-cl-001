# LiteRAG: Cost-Efficient Graph-Based Retrieval-Augmented Generation

Daniel Alejandro Coll Tejeda Pedro García López Daniel Barcelona-Pons

Universitat Rovira i Virgili

{danielalejandro.coll,pedro.garcia,daniel.barcelona}@urv.cat

## Abstract

Graph-based retrieval can improve multi-hop question answering, but existing approaches often incur high query-time costs and produce diffuse, oversized contexts that reduce generation efficiency. We present LiteRAG, a graph-based retrieval method that replaces expensive retrieval-time LLM control with query-conditioned algorithmic exploration and reasoning-chain context construction. On Dist-Comp, a benchmark for multi-hop retrieval over distributed-systems papers, LiteRAG attains the highest overall quality among the evaluated methods (0.798) while reducing perquery latency by over 100× and cost by over 99% relative to GraphRAG Global and DRIFT. On UltraDomain, it matches Linear-RAG on overall quality while using about 14× fewer tokens. An ablation study indicates that LiteRAG’s query-adaptive thresholding and community-aware hub penalization are the main drivers of its token-efficiency gains.

## 1 Introduction

Retrieval-Augmented Generation (RAG) has become a standard approach for grounding Large Language Models (LLMs) in external knowledge (Lewis et al., 2020; Fan et al., 2024). In its most common form, RAG retrieves text chunks by dense similarity and passes them to the generator as supporting context. This design is effective for many lookup-style queries, but it remains weak on questions that require combining evidence across documents, tracing relational dependencies, or synthesizing information over multiple hops (Gao et al., 2023; Jiang et al., 2023b).

Graph-based RAG methods address this limitation by representing corpora as structured graphs in which entities, relations, and communities are explicitly modeled (Edge et al., 2024; Guo et al., 2025; Huang et al., 2025a; Zhuang et al., 2026). However, many existing systems still rely heavily on the LLM at query time to traverse the graph, summarize communities, or aggregate evidence across retrieved regions. This dependence increases latency and cost, scales poorly with corpus size, and can produce broad contexts that force the generator to sift through substantial irrelevant or weakly relevant information (Liu et al., 2024).

For multi-hop question answering, the challenge is therefore not only to retrieve enough evidence, but to present it in a form that preserves a high density of query-relevant information. When retrieval returns loosely filtered text or broad summaries, the generator must still deduce how entities connect. Under realistic token budgets, this can dilute useful evidence and reduce the value of structured retrieval.

We therefore argue that graph retrieval for RAG should optimize both evidence selection and how retrieved evidence is prepared for the generator. We introduce LiteRAG, a method that removes the generation model from the retrieval loop. Rather than relying on sequential LLM calls for navigation, LiteRAG performs query-conditioned algorithmic graph exploration: it first selects semantic and lexical anchor nodes, then expands a bounded subgraph using dynamic thresholding and community-aware hub penalization.

The final stage addresses the same objective at the context level. Rather than passing large text collections or broad summaries to the generator, LiteRAG converts the retrieved subgraph into compact reasoning chains that express the retained relations directly in the prompt. This yields a more compact, query-focused context in which relevant connections are already explicit, preserving the structure needed for generation under restricted token budgets.

Our contributions are summarized as follows:

• Query-conditioned algorithmic retrieval: LiteRAG replaces fixed-hop or LLM-mediated traversal with algorithmic exploration based on dynamic thresholding and communityaware hub penalization.

• Reasoning-chain context construction: LiteRAG converts the retrieved subgraph into compact reasoning chains, producing a more information-dense context that remains effective within a budget of about 2,000 tokens.

• Quality-efficiency gains: On DistComp, LiteRAG attains the highest overall quality and lowest latency and cost among the evaluated methods; relative to the most expensive LLM-intensive GraphRAG configurations, it reduces per-query cost by over 99% and latency by over 100×.

To support reproducibility, the source code for LiteRAG, benchmark queries, configurations, and evaluation scripts will be released on GitHub.

## 2 Related Work

Recent work on Retrieval-Augmented Generation (RAG) (Fan et al., 2024) has shifted from flat passage retrieval toward structured augmentation to better support compositional multi-hop reasoning. Within this trend, graph-based methods explicitly encode entities and relations, enabling retrieval over relational structure rather than isolated text chunks.

Graph-based RAG methods can be grouped into two broad lines. A first line uses the graph to organize evidence but still relies heavily on the LLM at retrieval time. Microsoft GraphRAG (Edge et al., 2024) is the clearest example: LLMs are used both upstream to help construct graph representations and, more importantly, at query time to orchestrate traversal, summarize communities, and retrieve evidence from the graph. Related graphaware reasoning systems such as G-Retriever (He et al., 2024) and Think-on-Graph (Sun et al., 2024) likewise show the value of explicit relational structure. Across this line of work, the graph improves retrieval quality, but retrieval-time control remains closely coupled to the LLM.

A second line aims to make graph-based RAG more efficient. Among the most relevant systems for our setting, LightRAG (Guo et al., 2025) and HiRAG (Huang et al., 2025a) improve retrieval through dual-level or hierarchical organization, while related structured retrieval approaches such as RAPTOR (Sarthi et al., 2024) reorganize evidence before generation. More algorithmic methods such as HippoRAG (Gutiérrez et al., 2024), LinearRAG (Zhuang et al., 2026), KET-RAG (Huang et al., 2025b), and ROGRAG (Wang et al., 2025) reduce LLM dependence through graph scoring, indexing, and filtering, but they often rely on static graph statistics or fixed heuristics and give less direct attention to constructing a dense final context for the generator.

LiteRAG belongs to this second line, but differs from prior systems in both evidence collection and final context construction. Instead of relying on fixed neighborhoods, static graph statistics, or retrieval-time LLM decisions, it performs queryconditioned subgraph construction and then converts the retained evidence into compact reasoning chains. Whereas methods such as LightRAG, Hi-RAG, and LinearRAG primarily improve retrieval organization or graph scoring, LiteRAG couples adaptive evidence collection with explicit construction of a more information-dense final context. Relative to Microsoft GraphRAG and related LLMmediated systems, retrieval-time control is shifted away from the LLM.

## 3 LiteRAG Method

LiteRAG is a graph-based RAG method for costefficient retrieval over knowledge graphs. It replaces retrieval-time LLM control with queryconditioned algorithmic exploration and explicit context construction before final answer generation. As shown in Figure 1, the method has three stages: (1) query-conditioned anchor selection, which identifies promising graph entry points; (2) query-conditioned subgraph expansion, which builds a relevant subgraph from those anchors using semantic and structural signals; and (3) reasoningchain context construction, which converts the retrieved subgraph into a compact representation for the generator.

## 3.1 Phase 1: Query-Conditioned Anchor Selection

Given a query q and a knowledge graph $G \ =$ $( V , E )$ , Phase 1 selects an anchor set $A \subset V$ of semantically and lexically salient entities. Rather than scoring every node in V, LiteRAG retrieves a candidate pool $V _ { c } \subset V$ via top-K vector search, assigning each $v \in V _ { c }$ a composite score $S ( v )$

$$
\begin{array} { c } { S ( v ) = \alpha S _ { \mathrm { s e m } } ( q , v ) + \beta S _ { \mathrm { l e x } } ( q , v ) } \\ { + \gamma S _ { \mathrm { c o m } } ( q , C ( v ) ) } \end{array}\tag{1}
$$

![](images/f52c4d9b40adfcb0f4379b8308664d4922a751ea261c2e4c9b8c3b8e9c88ab3a.jpg)  
Figure 1: Overview of the LiteRAG method.

where $C ( v )$ denotes the community containing v, and $\alpha , \beta ,$ , and $\gamma$ are non-negative weights that balance the three signals. The components of the score are: $S _ { \mathrm { s e m } } ( q , v )$ is the semantic similarity between the query representation and node v, $S _ { \mathrm { l e x } } ( \boldsymbol { q } , \boldsymbol { v } )$ is a lexical matching score designed to preserve entities that may be crucial by name, such as proper nouns or domain-specific terms, even when semantic similarity alone is insufficient, and $S _ { \mathrm { c o m } } ( q , C ( v ) )$ is a community-level relevance signal that favors nodes in graph regions whose aggregate content is aligned with the query.

This formulation reflects the retrieval objective of the phase: maximize the chance that exploration starts from informative graph regions without overcommitting to a single notion of relevance. The semantic term captures conceptual alignment, the lexical term protects exact terminology, and the community term biases the search toward coherent topical neighborhoods.

After computing $S ( v )$ for all candidate nodes in $V _ { c } ,$ , LiteRAG constructs the anchor set through thresholding:

$$
A = \{ v \in V _ { c } \mid S ( v ) \geq \tau _ { \mathrm { a n c h o r } } \}\tag{2}
$$

where $\tau _ { \mathrm { a n c h o r } }$ is the anchor threshold. The resulting set A serves as the initial frontier for Phase 2, where LiteRAG expands a query-conditioned subgraph from these entry points.

## 3.2 Phase 2: Query-Conditioned Subgraph Expansion

Phase 2 takes the anchor set A from Phase 1 and the knowledge graph $G = ( V , E )$ , and returns a query-conditioned subgraph $G _ { q } ~ = ~ ( V _ { q } , E _ { q } )$ expanded from those anchors. Expansion proceeds from all anchors in parallel. Rather than delegating retrieval-time navigation to an LLM, LiteRAG treats graph traversal as an algorithmic search problem in which candidate expansions are accepted or pruned according to a query-conditioned relevance score.

Concretely, LiteRAG performs a parallel subgraph expansion initialized at the nodes in A. At each step, the current frontier contains nodes already admitted into the explored region. For a frontier node u and one of its neighbors v, LiteRAG computes a traversal relevance score that determines whether v should be incorporated into the explored subgraph and added to the next frontier. Expansion therefore grows adaptively around the anchors, and the final output is the subgraph induced by the retained nodes and traversed edges. Exploration stops when no frontier nodes satisfy the relevance criterion below, when the maximum hop depth $k _ { \mathrm { m a x } }$ is reached, or when the per-anchor expansion cap $N _ { \mathrm { m a x } }$ is exhausted.

## 3.2.1 Query-Adaptive Thresholding

The first component of the exploration rule is a query-conditioned threshold that controls how selective traversal should be for a given query. Instead of using a global fixed threshold, LiteRAG derives a dynamic threshold from the quality of the initial anchors:

$$
\tau _ { \mathrm { d y n } } ( q , A ) = \operatorname* { m i n } \biggl ( 1 . 0 , \tau _ { \mathrm { b a s e } } + \lambda \operatorname* { m a x } _ { a \in A } S ( a ) \biggr )\tag{3}
$$

where $\tau _ { \mathrm { b a s e } }$ is a minimum relevance floor, λ controls how strongly anchor evidence influences the threshold, and $S ( a )$ is the anchor score from Phase 1; the outer minimum caps $\tau _ { \mathrm { d y n } }$ at 1.0 because $S ( a )$ is an uncalibrated composite score rather than a strict probability. This formulation links exploration directly to the quality of the retrieval starting points: strong anchors induce a stricter traversal criterion, while weaker anchors allow broader exploration of potentially relevant graph regions.

## 3.2.2 Community-Aware Hub Penalization

The second component of the exploration rule addresses high-degree hubs, which can connect otherwise unrelated graph regions and cause the search to drift. To reduce this effect, LiteRAG penalizes candidate expansions through nodes whose structural centrality is more likely to reflect generic connectivity than query relevance.

For a frontier node $u ,$ a neighboring candidate node $v ,$ and hop depth k, LiteRAG defines the traversal relevance as:

$$
R ( q , u , v , k ) = S _ { \mathrm { s e m } } ( q , v ) \cdot d ^ { k } \cdot P _ { \mathrm { h u b } } ( u , v )\tag{4}
$$

where $S _ { \mathrm { s e m } } ( q , v )$ is the semantic similarity between the query and node $v ,$ d is a depth-decay factor, and $P _ { \mathrm { h u b } } ( u , v )$ is a hub penalty. A candidate expansion is retained when:

$$
R ( q , u , v , k ) \geq \tau _ { \mathrm { d y n } } ( q , A )\tag{5}
$$

The hub penalty is defined as:

$$
P _ { \mathrm { h u b } } ( u , v ) = \frac { 1 } { 1 + \delta \log ( 1 + \deg ( v ) ) \omega ( u , v ) }\tag{6}
$$

where $\deg ( v )$ is the degree of node $v , \delta$ controls the strength of the hub penalty, and $\omega ( u , v )$ modulates this penalty according to community alignment:

$$
\omega ( u , v ) = 1 - \mathbb { I } ( C ( u ) = C ( v ) ) \kappa\tag{7}
$$

Here, $\mathbb { I } ( \cdot )$ is the indicator function, $C ( u )$ and $C ( v )$ denote the communities of u and $v ,$ and $\kappa \in [ 0 , 1 ]$ controls how strongly within-community transitions are protected from hub penalization. As a result, high-degree nodes that remain within the same topical region are penalized less aggressively than hubs that bridge unrelated communities.

Together, the dynamic threshold and hub-aware penalty define the expansion rule used to grow the frontier from the anchors. The output of this phase is the retained query-conditioned subgraph $G _ { q } ,$ which is then passed to Phase 3 for ranking and context construction.

## 3.3 Phase 3: Reasoning-Chain Context Construction

Phase 3 converts the query-conditioned subgraph $G _ { q }$ into the final generation context. It aims to preserve relational evidence while expressing it in a compact form under strict token budgets.

To do so, LiteRAG ranks entities in $G _ { q }$ and retains relations incident to the top-ranked ones for inclusion in the final reasoning-chain context. For each explored entity $v \in V _ { q }$ , LiteRAG computes a consensus score $S _ { \mathrm { c o n s } } ( v )$ that combines four normalized signals:

$$
\begin{array} { r } { S _ { \mathrm { c o n s } } ( v ) = \rho _ { 1 } I _ { \mathrm { r a t e } } ( v ) + \rho _ { 2 } S _ { \mathrm { s e m } } ( q , v ) } \\ { + \rho _ { 3 } P _ { \mathrm { p r o x } } ( v ) + \rho _ { 4 } S _ { \mathrm { s t r } } ( v ) } \end{array}\tag{8}
$$

where $I _ { \mathrm { r a t e } } ( v )$ is the fraction of anchors whose expansions reach $v , S _ { \mathrm { s e m } } ( q , v )$ is the semantic alignment between v and the query, $P _ { \mathrm { p r o x } } ( v )$ is a topological proximity term that decays with the average hop distance from the anchors, and $S _ { \mathrm { s t r } } ( v )$ is a structural centrality score derived from graph measures such as PageRank and betweenness. To reward evidence supported by consistently relevant traversal paths, LiteRAG further rescales this score by the average traversal relevance accumulated when reaching v and by a small bonus when v is itself an initial anchor. Entities with the highest resulting scores are retained, and the relations connecting them are converted into a compact context representation that exposes relational structure directly to the generator.

The main representation used in this phase is a reasoning chain, which linearizes a graph relation into a short relational statement rather than a larger unstructured text block:

$$
\begin{array} { r l } & { \mathsf { E n t i t y \ A } \xrightarrow { \boldsymbol { I R e l a t i o n s h i p J } } \mathsf { E n t i t y \ B : \ ^ { \prime \prime } D e s c r i p t i o n \ o f } } \\ & { t h e \ s p e c i f i c \ i n t e r a c t i o n . . . ^ { \prime \prime } } \end{array}
$$

The final generation context combines these reasoning chains with minimal supporting information needed for interpretation, such as definitions of topranked entities or brief source-grounded evidence when available. In this way, LiteRAG converts the explored subgraph into a structurally explicit context that is substantially more compact than raw-text or community summaries.

This stage is related to context compression approaches (Jiang et al., 2023a), but its role here is more specific: rather than shortening arbitrary text, LiteRAG constructs a curated, information-dense relational context in which the relevant connections are already explicit.

Section 5.4 revisits this design choice and shows that the context-construction step reduces token consumption while preserving the explicit relational evidence needed for multi-hop reasoning.

## 4 Experimental Methodology

We evaluate LiteRAG on DistComp, a custom benchmark designed to test the query types and corpus-scaling behavior central to our setting, and on the Mix split of UltraDomain, a public benchmark evaluated under the same model setup across representative graph-based and denseretrieval baselines.

## 4.1 Datasets

DistComp. We construct DistComp from Distributed Computing papers published between 2020 and 2022 in nine major IEEE and ACM venues. The benchmark is designed to cover query types that are less well represented in standard public datasets but central to our setting, especially multi-hop synthesis and drift-style comparison, and to test how retrieval behavior changes as corpus size grows. We create five corpus sizes, $D =$ {40, 80, 160, 640, 1280}, with the largest split containing approximately 500,000 tokens, and annotate 160 expert-written queries in four categories: literal citation, local reasoning, global thematic synthesis, and drift-style multi-hop comparison across distant graph regions.

UltraDomain. We also evaluate on the Mix split of UltraDomain, a public multi-domain benchmark spanning agriculture, law, and healthcare.

## 4.2 Baselines

We compare LiteRAG against five baseline families: GraphRAG Basic as a dense-retrieval baseline over text chunks; Microsoft GraphRAG (Edge et al., 2024) in its Local, Global, and DRIFT modes; LightRAG (Guo et al., 2025) in Local, Global, Mix, and Hybrid modes; HiRAG (Huang et al., 2025a) in Hi, Local, Global, Bridge, and No-Bridge modes; and LinearRAG (Zhuang et al., 2026) in its default configuration.

## 4.3 Implementation Details

All methods use gemini-2.5-flash-lite for generation and gemini-embedding-001 for embeddings. LiteRAG hyperparameters were selected by grid search on a validation split and are reported in Appendix D. For graph construction, LiteRAG reuses Microsoft GraphRAG indexing (Edge et al., 2024) and differs from prior methods only in retrieval and context construction.

## 4.4 Evaluation Metrics

We report one composite quality metric and three efficiency measures. Overall quality is defined as

$$
\begin{array} { r l } { Q _ { \mathrm { t o t a l } } = 0 . 6 \cdot Q _ { \mathrm { j u d g e } } + 0 . 2 5 \cdot Q _ { \mathrm { s e m } } } & { { } } \\ { + 0 . 1 5 \cdot Q _ { \mathrm { l e x } } } & { { } } \end{array}\tag{9}
$$

where $Q _ { \mathrm { j u d g e } }$ is an LLM-as-a-judge score, $Q \mathrm { { s e m } }$ is embedding-based semantic similarity, and $Q _ { \mathrm { l e x } }$ is ROUGE-L. The weights were fixed on a heldout validation split to emphasize reasoning quality while retaining semantic and lexical grounding; disaggregated values and the final weighting are reported in Appendix B. For efficiency, we report end-to-end latency in seconds per query, total token consumption, and cost per query under a shared API pricing model (\$0.10 per 1M input tokens and \$0.40 per 1M output tokens).

## 5 Results and Empirical Analysis

This section reports LiteRAG’s results in terms of quality, latency, token usage, and cost.

## 5.1 Comparative Performance on DistComp and UltraDomain

Table 1 reports the main comparison on the largest DistComp setting, $D _ { 1 2 8 0 }$ , and the Mix split of Ultra-Domain. We report overall quality, $Q _ { \mathrm { t o t a l } }$ , together with the three efficiency metrics from Section 4.4.

On $D _ { 1 2 8 0 }$ , LiteRAG attains the highest overall quality $( Q _ { \mathrm { t o t a l } } = 0 . 7 9 8 )$ , followed by LinearRAG (0.784) and HiRAG Local (0.777). It also records the lowest latency (1.42s), token usage (2,291), and cost per query (\$0.0003). The efficiency difference is largest against LLM-intensive methods: GraphRAG Global requires 162.66s and 643k tokens per query, while GraphRAG DRIFT exceeds 142s and 6.4M tokens.

The comparison with LinearRAG is especially informative because both methods reduce retrievaltime LLM reliance, but they do so through different retrieval pipelines. LinearRAG relies on transformer-based query-time entity extraction and iterative graph-ranking stages, whereas LiteRAG uses bounded query-conditioned expansion and reasoning-chain context construction. On $D _ { 1 2 8 0 }$ , LiteRAG improves overall quality (0.798 vs. 0.784) while reducing latency from 8.01s to 1.42s, token usage from 3,077 to 2,291, and cost from \$0.0006 to \$0.0003 per query. This comparison suggests that LiteRAG can match or improve overall quality while passing a more efficient final context to the generator.

<table><tr><td rowspan="2">Method</td><td rowspan="2">Configuration</td><td colspan="4">DistComp  $D _ { 1 2 8 0 }$ </td><td colspan="4">UltraDomain Mix</td></tr><tr><td> $Q \mathrm { t o t a l }$  ←</td><td>Lat. (s) ↓</td><td>Tokens ↓</td><td> $\mathbf { C o s t } ( \$ 123,4$ </td><td> $Q \mathrm { t o t a l }$  ↑</td><td>Lat. (s) ↓</td><td>Tokens ↓</td><td>Cost ($) ↓</td></tr><tr><td>LiteRAG</td><td></td><td>0.798</td><td>1.42</td><td>2,291</td><td>0.0003</td><td>0.801</td><td>1.24</td><td>4,862</td><td>0.0006</td></tr><tr><td>GraphRAG</td><td>Basic (Naive)</td><td>0.775</td><td>1.96</td><td>4,599</td><td>0.0006</td><td>0.778</td><td>38.49</td><td>12,835</td><td>0.0014</td></tr><tr><td>GraphRAG</td><td>Local</td><td>0.773</td><td>4.27</td><td>7,866</td><td>0.0009</td><td>0.771</td><td>4.66</td><td>10,082</td><td>0.0011</td></tr><tr><td>GraphRAG</td><td>Global</td><td>0.714</td><td>162.66</td><td>643,909</td><td>0.0679</td><td>0.604</td><td>135.34</td><td>519,609</td><td>0.0533</td></tr><tr><td>GraphRAG</td><td>DRIFT</td><td>0.657</td><td>142.04</td><td>6,441,013</td><td>0.7901</td><td>0.572</td><td>207.66</td><td>33,160,141</td><td>3.9303</td></tr><tr><td>LightRAG</td><td>Local</td><td>0.762</td><td>3.17</td><td>16,083</td><td>0.0018</td><td>0.789</td><td>7.51</td><td>46,309</td><td>0.0048</td></tr><tr><td>LightRAG</td><td>Global</td><td>0.694</td><td>3.06</td><td>15,594</td><td>0.0017</td><td>0.747</td><td>8.45</td><td>38,204</td><td>0.0039</td></tr><tr><td>LightRAG</td><td>Mix</td><td>0.760</td><td>3.62</td><td>20,361</td><td>0.0022</td><td>0.796</td><td>7.85</td><td>88,449</td><td>0.0090</td></tr><tr><td>LightRAG</td><td>Hybrid</td><td>0.763</td><td>3.32</td><td>20,567</td><td>0.0022</td><td>0.791</td><td>7.59</td><td>69,480</td><td>0.0071</td></tr><tr><td>HiRAG</td><td>Hi</td><td>0.767</td><td>10.95</td><td>29,988</td><td>0.0031</td><td>0.783</td><td>6.15</td><td>28,200</td><td>0.0029</td></tr><tr><td>HiRAG</td><td>Local</td><td>0.777</td><td>8.03</td><td>14,296</td><td>0.0016</td><td>0.785</td><td>5.49</td><td>18,792</td><td>0.0020</td></tr><tr><td>HiRAG</td><td>Global</td><td>0.773</td><td>3.17</td><td>20,112</td><td>0.0021</td><td>0.790</td><td>6.54</td><td>20,177</td><td>0.0021</td></tr><tr><td>HiRAG</td><td>Bridge</td><td>0.760</td><td>2.68</td><td>16,631</td><td>0.0018</td><td>0.786</td><td>5.93</td><td>16,149</td><td>0.0017</td></tr><tr><td>HiRAG</td><td>No-Bridge</td><td>0.754</td><td>2.70</td><td>26,432</td><td>0.0028</td><td>0.782</td><td>6.13</td><td>29,549</td><td>0.0031</td></tr><tr><td>LinearRAG</td><td></td><td>0.784</td><td>8.01</td><td>3,077</td><td>0.0006</td><td>0.801</td><td>4.79</td><td>68,774</td><td>0.0071</td></tr></table>

Table 1: Main results comparing quality $( Q _ { \mathrm { t o t a l } } )$ , latency (seconds/query), token consumption, and cost (\$/query) across the DistComp $D _ { 1 2 8 0 }$ benchmark and the UltraDomain Mix split.

The UltraDomain results show a similar pattern. LiteRAG and LinearRAG attain the highest overall quality (0.801), followed by LightRAG Mix (0.796). LiteRAG also records the lowest latency (1.24s), token consumption (4,862), and cost (\$0.0006) among all evaluated methods.

While LinearRAG matches LiteRAG’s aggregate quality on UltraDomain, it requires 4.79 seconds, 68,774 tokens, and \$0.0071 per query. Relative to LinearRAG, LiteRAG achieves a 14× reduction in token volume, about a 92% reduction in cost, and a 74% reduction in latency. Other high-performing baselines incur even greater overhead; LightRAG Mix averages 7.85s and \$0.0090 per query, while GraphRAG DRIFT exceeds 200s and \$3.93.

Taken together, these results indicate that LiteRAG matches the quality of the strongest graphbased methods while using a much smaller execution budget. This is consistent with our central claim that a compact, structurally curated context can preserve answer quality without requiring the LLM to process large volumes of retrieved text.

## 5.2 Latency and Cost Across Corpus Scale

We next examine how efficiency changes as the DistComp corpus grows from 40 to 1280 documents. Figure 2 shows LiteRAG and the strongest baseline configuration from each method family, while Appendix A reports all configurations.

Latency Trends LiteRAG shows the lowest and most stable latency profile over the evaluated range, remaining between 1.33 and 1.74 seconds per query and ending at 1.42 seconds on $D _ { 1 2 8 0 }$ . The selected baselines are consistently slower: LightRAG Local ranges from 2.54 to 3.17 seconds, HiRAG Bridge from 2.02 to 3.22 seconds, GraphRAG Local from 3.53 to 4.27 seconds, and LinearRAG from 5.56 to 8.05 seconds. The gap is present at every corpus size and becomes much larger for the more expensive graph-based modes reported in the appendix.

![](images/1844892117295eac414b1433c2e55a0e4c980a9da6b5a9bf0d1eeedf1ade69fb.jpg)  
Figure 2: Latency-cost comparison across corpus sizes for LiteRAG and the best reported baseline configuration from each compared method family; labels indicate dataset size $( D _ { 4 0 }$ $D _ { 8 0 }$ $D _ { 1 6 0 }$ $D _ { 6 4 0 }$ , and $D _ { 1 2 8 0 } )$ .

The appendix makes this separation clearer. GraphRAG Global grows from 8.57s $( D _ { 4 0 } )$ to 162.66s $( D _ { 1 2 8 0 } )$ , and DRIFT remains above 139s throughout. Over the same settings, LiteRAG stays within a narrow band. The reported results therefore show that LiteRAG preserves low end-to-end latency as corpus size increases, whereas broader graph aggregation and agentic traversal incur much

larger runtime overhead.

Cost Trends The cost results follow the same pattern. LiteRAG remains between \$0.00028 and \$0.00030 per query across all corpus sizes. The selected baselines remain more expensive: Linear-RAG reaches \$0.0006 on $D _ { 1 2 8 0 }$ , LightRAG Local ranges from \$0.0014 to \$0.0018, HiRAG Bridge from \$0.0007 to \$0.0018, and GraphRAG Local from \$0.0008 to \$0.0012. The appendix again shows the largest increases in the LLM-intensive configurations, with GraphRAG Global rising from \$0.0031 to \$0.0679 and GraphRAG DRIFT remaining above \$0.6195 throughout.

Taken together, these results show that LiteRAG maintains the lowest reported latency-cost profile over the evaluated range. This is consistent with the design in Section 3: bounded candidate selection, thresholded query-conditioned expansion, and compact context construction rather than corpus-level summarization or agentic traversal. Section 5.3 examines the same pattern under an explicit fixedbudget constraint.

## 5.3 Quality Under a Fixed Token Budget

This experiment evaluates answer quality when each method is constrained to a context budget of about 2,000 tokens, comparable to LiteRAG’s default setting. For each baseline, the budgeted variant tightens the retrieval configuration so that less context is passed to the generator while keeping the shared generation model unchanged.

LiteRAG is unchanged because its pipeline already meets the budget. Table 2 therefore shows no quality loss for LiteRAG, whereas nearly all baselines decline. The largest drops are LightRAG Global (−16.1%), GraphRAG DRIFT (−13.3%), LightRAG Local (−9.3%), and LightRAG Mix (−8.7%). GraphRAG Basic and Local drop by 5.2% and 4.0%, while LinearRAG declines by 2.1%. HiRAG is the most resilient baseline family, but its strongest configurations still remain below LiteRAG’s budgeted score.

GraphRAG Global could not be evaluated because its Map-Reduce retrieval procedure does not expose a comparable hard token cap. GraphRAG DRIFT is also informative: even its most restrictive available configuration still uses 570,440 tokens, down from 6,441,013, and still loses 13.3% in quality. Some retrieval pipelines therefore cannot be reduced to a small prompt budget without either substantial degradation or budget violation.

<table><tr><td>Method</td><td>Default</td><td>Budgeted</td><td>∆(%)</td></tr><tr><td>GraphRAG</td><td></td><td></td><td></td></tr><tr><td>Basic</td><td>77.5%</td><td>72.3%</td><td>-5.2%</td></tr><tr><td>Local</td><td>77.3%</td><td>73.3%</td><td>-4.0%</td></tr><tr><td>Global</td><td>71.4%</td><td></td><td></td></tr><tr><td>DRIFT</td><td>65.7%</td><td>52.4%</td><td>-13.3%</td></tr><tr><td>LightRAG</td><td></td><td></td><td></td></tr><tr><td>Global</td><td>69.4%</td><td>53.3%</td><td>-16.1%</td></tr><tr><td>Hybrid</td><td>76.3%</td><td>71.9%</td><td>-4.4%</td></tr><tr><td>Local</td><td>76.2%</td><td>66.9%</td><td>-9.3%</td></tr><tr><td>Mix</td><td>76.0%</td><td>67.3%</td><td>-8.7%</td></tr><tr><td>HiRAG</td><td></td><td></td><td></td></tr><tr><td>Hi</td><td>76.7%</td><td>72.9%</td><td>-3.8%</td></tr><tr><td>Local</td><td>77.7%</td><td>75.7%</td><td>-2.0%</td></tr><tr><td>Global</td><td>77.3%</td><td>73.6%</td><td>-3.7%</td></tr><tr><td>Bridge</td><td>76.0%</td><td>74.8%</td><td>-1.2%</td></tr><tr><td>No-Bridge</td><td>75.4%</td><td>74.7%</td><td>-0.7%</td></tr><tr><td>LinearRAG</td><td>78.4%</td><td>76.3%</td><td>-2.1%</td></tr><tr><td>LiteRAG</td><td>79.8%</td><td>79.8%</td><td>0.0%</td></tr></table>

Table 2: Default and budgeted $Q _ { \mathrm { t o t a l } }$ under a fixed token budget of approximately 2,000 tokens; LiteRAG’s budgeted setting is identical to its default setting.

The contrast with LiteRAG is consistent with Phase 3. LiteRAG ranks the retrieved subgraph and constructs the final reasoning-chain context from the retained evidence, so the standard retrieval output already fits the target budget without an additional compression step.

Taken together, these results show that LiteRAG’s advantage is not only lower average token use, but also stronger quality retention under an explicit budget constraint. This complements the scalability analysis: the same bounded retrieval and context-construction stages that reduce latency and cost also preserve answer quality when context is tightly limited.

## 5.4 Ablation Study

To isolate the contribution of LiteRAG’s main design choices, we conduct an ablation study on 32 complex multi-hop queries. We systematically disable key pipeline components and measure the resulting changes in overall quality $( Q _ { \mathrm { t o t a l } } )$ , token consumption, and cost (Table 3).

Lexical and Community Signals (Phase 1): Removing the lexical and community signals from Phase 1 (Semantic-only anchors in Table 3) reduces overall quality from 0.798 to 0.761. This is consistent with the role of query-conditioned anchor selection: semantic similarity alone is less reliable for preserving exact entity terminology and for biasing retrieval toward relevant graph regions.

<table><tr><td>Configuration</td><td> $Q _ { \mathrm { t o t a l } }$  7</td><td>Tokens↓</td><td>Cost ($) ↓</td></tr><tr><td>LiteRAG (Full)</td><td>0.798</td><td>2,291</td><td>0.00030</td></tr><tr><td>Semantic-only anchors</td><td>0.761</td><td>2,103</td><td>0.00027</td></tr><tr><td>Fixed-hop expansion</td><td>0.787</td><td>7,979</td><td>0.00087</td></tr><tr><td>No hub penalty</td><td>0.789</td><td>4,522</td><td>0.00052</td></tr><tr><td>Raw subgraph context</td><td>0.780</td><td>3,622</td><td>0.00043</td></tr></table>

Table 3: Ablation study evaluating the impact of core architectural components on overall quality, token efficiency, and cost per query.

Query-Adaptive Thresholding (Phase 2): Replacing query-adaptive thresholding with a rigid fixed-hop expansion (Fixed-hop expansion in Table 3) leaves overall quality relatively stable (0.787), but increases average token consumption by nearly 250% (from 2,291 to 7,979) and roughly triples per-query cost. This pattern suggests that fixed-hop traversal pulls in substantially more lowyield neighboring context. Within this ablation, query-adaptive thresholding is therefore the main contributor to LiteRAG’s token-efficiency.

Community-Aware Hub Penalization (Phase 2): Removing community-aware hub penalization (No hub penalty in Table 3) increases token consumption to 4,522 while leaving overall quality similar (0.789). This suggests that, without the penalty, high-degree hubs mainly add structural noise and cost rather than useful evidence.

Reasoning-Chain Context Construction (Phase 3): Bypassing reasoning-chain context construction (Raw subgraph context in Table 3) and instead passing the retrieved subgraph to the LLM as raw lists of entities and relationships increases token consumption by nearly 60% (3,622 tokens) and lowers overall quality to 0.780. This result is consistent with the role of Phase 3: structuring retained graph evidence into a more compact and directly usable final context.

## 6 Conclusion

In this paper, we presented LiteRAG, a graph-based retrieval method for RAG built on a simple premise: for multi-hop generation, retrieval should not only find relevant evidence, but deliver it to the LLM in a compact, structurally explicit form. LiteRAG addresses this objective by replacing LLM-mediated navigation with query-conditioned algorithmic exploration and reasoning-chain context construction. In particular, the method combines lexical and community-aware anchor selection, query-adaptive thresholding, community-aware hub penalization, and compact reasoning-chain construction before final generation.

Across the reported experiments, this design yields a favorable quality-efficiency trade-off. On DistComp $D _ { 1 2 8 0 }$ , LiteRAG attains the highest reported $Q _ { \mathrm { t o t a l } }$ among the compared methods while also using the fewest tokens, the lowest latency, and the lowest per-query cost. The broader scaling results show a comparatively flat latency-cost profile over the evaluated corpus range, and the fixedbudget analysis suggests that LiteRAG’s advantage is not merely retrieving less context, but retrieving context with higher effective information density and remaining effective under a small budget. On UltraDomain, LiteRAG remains competitive in aggregate quality while using a substantially smaller retrieved context and markedly lower execution cost than the strongest comparator.

Taken together, these findings support the paper’s three main contributions. The comparative results establish the overall quality-efficiency gains; the fixed-budget and ablation results show that reasoning-chain context construction helps preserve overall quality under tight token budgets; and the ablation study further shows that lexical and community-aware anchor selection supports retrieval quality, while query-adaptive thresholding and community-aware hub penalization are the main drivers of token-efficiency. More broadly, separating graph exploration from LLM inference and treating context construction as a core part of retrieval yields a smaller, more information-dense final context for the generator. Future work can refine retrieval controls and test the same contextcuration principle across additional domains.

## Limitations

This study has some limitations. First, DistComp is a specialized benchmark centered on distributedsystems literature, so the absolute performance levels reported here should not be assumed to transfer unchanged to every domain; the UltraDomain results are intended to check that the main qualityefficiency pattern is not confined to that corpus. Second, LiteRAG focuses on retrieval and context construction rather than graph indexing, so the paper does not address the full end-to-end optimization problem when indexing cost is itself a primary concern. Third, all systems are evaluated under a shared generation and embedding setup to isolate retrieval behavior, which improves comparability but leaves cross-model robustness for future work. Finally, the scalability analysis is empirical rather than formal, and some baseline architectures do not expose controls that permit perfectly matched hard token budgets across methods. These limitations primarily affect scope and generality, rather than the paper’s central claim that LiteRAG delivers a strong quality-efficiency trade-off through queryconditioned graph exploration and compact context construction.

## References

Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva N. Mody, Steven Truitt, and Jonathan Larson. 2024. From local to global: A graph rag approach to query-focused summarization. ArXiv, abs/2404.16130.

Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin, Tat-Seng Chua, and Qing Li. 2024. A survey on rag meeting llms: Towards retrieval-augmented large language models. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, KDD ’24, pages 6491—-6501, New York, NY, USA. Association for Computing Machinery.

Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jin Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Qianyu Guo, Meng Wang, and Haofen Wang. 2023. Retrievalaugmented generation for large language models: A survey. ArXiv, abs/2312.10997.

Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. 2025. LightRAG: Simple and fast retrievalaugmented generation. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 10746–10761, Suzhou, China. Association for Computational Linguistics.

Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su. 2024. Hipporag: neurobiologically inspired long-term memory for large language models. In Proceedings ofthe 38th International Conference on Neural Information Processing Systems, NIPS ’24, Red Hook, NY, USA. Curran Associates Inc.

Xiaoxin He, Yijun Tian, Yifei Sun, Nitesh V Chawla, Thomas Laurent, Yann LeCun, Xavier Bresson, and Bryan Hooi. 2024. G-retriever: Retrieval-augmented generation for textual graph understanding and question answering. In The Thirty-eighth Annual Conference on Neural Information Processing Systems.

Haoyu Huang, Yongfeng Huang, Yang Junjie, Zhenyu Pan, Yongqiang Chen, Kaili Ma, Hongzhi Chen, and James Cheng. 2025a. Retrieval-augmented generation with hierarchical knowledge. In Findings of the Associationfor Computational Linguistics: EMNLP 2025, pages 6044–6060, Suzhou, China. Association for Computational Linguistics.

Yiqian Huang, Shiqi Zhang, and Xiaokui Xiao. 2025b. Ket-rag: A cost-efficient multi-granular indexing framework for graph-rag. In Proceedings ofthe 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2, KDD ’25, page 1003–1012, New York, NY, USA. Association for Computing Machinery.

Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. 2023a. LLMLingua: Compressing prompts for accelerated inference of large language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 13358–13376.

Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqiu Sun, Qian Liu, Jane Dwyer, Graham Neubig, Baolin Peng, and Dragomir Kalai. 2023b. Active retrieval augmented generation. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 7969–7992.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-augmented generation for knowledgeintensive nlp tasks. In Proceedings ofthe 34th International Conference on Neural Information Processing Systems, NIPS ’20, Red Hook, NY, USA. Curran Associates Inc.

Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2024. Lost in the middle: How language models use long contexts. Transactions ofthe Association for Computational Linguistics, 12:157–173.

Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and Christopher D. Manning. 2024. RAPTOR: Recursive abstractive processing for tree-organized retrieval. In The Twelfth International Conference on Learning Representations.

Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Saizhuo Wang, Chen Lin, Yeyun Gong, Heung-Yeung Shum, and Jian Guo. 2024. Think-on-graph: Deep and responsible reasoning of large language model on knowledge graph. In The Twelfth International Conference on Learning Representations.

Zhefan Wang, Huanjun Kong, Jie Ying, Wanli Ouyang, and Nanqing Dong. 2025. ROGRAG: A robustly optimized GraphRAG framework. In Proceedings of the 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 3: System Demonstrations), pages 604–613, Vienna, Austria. Association for Computational Linguistics.

Luyao Zhuang, Shengyuan Chen, Yilin Xiao, Huachi Zhou, Yujing Zhang, Hao Chen, Qinggang Zhang, and Xiao Huang. 2026. LinearRAG: Linear graph retrieval augmented generation on large-scale corpora. In The Fourteenth International Conference on Learning Representations.

## A Complete Scalability Results

Table 4 reports the complete scalability results for all evaluated query engines across all dataset sizes, expanding on the summary provided in Section 5.2. The table details average end-to-end latency, total token consumption, cost per query, and overall quality.
<table><tr><td>Engine</td><td>D</td><td>Lat. (s)</td><td>Tok.</td><td>Cost ($)</td><td> $Q _ { \mathrm { t o t a l } }$ </td></tr><tr><td rowspan="5">GraphRAG</td><td>40</td><td>2.13</td><td>4,417</td><td>0.0006</td><td>0.8067</td></tr><tr><td>80</td><td>1.62</td><td>4,509</td><td>0.0006</td><td>0.7938</td></tr><tr><td>160</td><td>1.56</td><td>4,354</td><td>0.0006</td><td>0.8213</td></tr><tr><td>640</td><td>1.84</td><td>4,539</td><td>0.0006</td><td>0.7906</td></tr><tr><td>1280</td><td>1.96</td><td>4,599</td><td>0.0006</td><td>0.7746</td></tr><tr><td rowspan="5">GraphRAG Local</td><td>40</td><td>3.53</td><td>6,482</td><td>0.0008</td><td>0.7547</td></tr><tr><td>80</td><td>3.69</td><td>9,331</td><td>0.0011</td><td>0.7818</td></tr><tr><td>160</td><td>3.69</td><td>8,739</td><td>0.0010</td><td>0.7152</td></tr><tr><td>640</td><td>4.10</td><td>10,630</td><td>0.0012</td><td>0.7610</td></tr><tr><td>1280</td><td>4.27</td><td>7,866</td><td>0.0009</td><td>0.7727</td></tr><tr><td rowspan="5">GraphRAG Global</td><td>40</td><td>8.57</td><td>26,633</td><td>0.0031</td><td>0.7077</td></tr><tr><td>80</td><td>19.16</td><td>53,208</td><td>0.0059</td><td>0.7498</td></tr><tr><td>160</td><td>30.63</td><td>93,364</td><td>0.0100</td><td>0.7156</td></tr><tr><td>640</td><td>92.68</td><td>348,525</td><td>0.0365</td><td>0.7088</td></tr><tr><td>1280</td><td>162.66</td><td>643,909</td><td>0.0679</td><td>0.7144</td></tr><tr><td rowspan="5">GraphRAG DRIFT</td><td>40</td><td>142.90</td><td>5,103,967</td><td>0.6470</td><td>0.6837</td></tr><tr><td>80</td><td>139.83</td><td>4,832,192</td><td>0.6195</td><td>0.5861</td></tr><tr><td>160</td><td>147.81</td><td>6,111,591</td><td>0.7502</td><td>0.6159</td></tr><tr><td>640</td><td>149.29</td><td>7,312,642</td><td>0.8777</td><td>0.4488</td></tr><tr><td>1280</td><td>142.04</td><td>6,441,013</td><td>0.7901</td><td>0.6574</td></tr><tr><td rowspan="5">LightRAG Global</td><td>40</td><td>3.00</td><td>13,178</td><td>0.0015</td><td>0.7932</td></tr><tr><td>80</td><td>3.08</td><td>13,624</td><td>0.0015</td><td>0.7657</td></tr><tr><td>160</td><td>2.50</td><td>13,859</td><td>0.0015</td><td>0.7803</td></tr><tr><td>640</td><td>3.09</td><td>15,557</td><td>0.0017</td><td>0.7524</td></tr><tr><td>1280</td><td>3.06</td><td>15,594</td><td>0.0017</td><td>0.6943</td></tr><tr><td rowspan="5">LightRAG Hybrid</td><td>40</td><td>3.52</td><td>17,246</td><td>0.0019</td><td>0.7939</td></tr><tr><td>80</td><td>3.28</td><td>18,491</td><td>0.0020</td><td>0.7988</td></tr><tr><td>160</td><td>3.34</td><td>18,465</td><td>0.0020</td><td>0.8103</td></tr><tr><td>640</td><td>3.63</td><td>20,256</td><td>0.0022</td><td>0.7877</td></tr><tr><td>1280</td><td>3.32</td><td>20,567</td><td>0.0022</td><td>0.7627</td></tr><tr><td rowspan="5">LightRAG Local</td><td>40</td><td>2.83</td><td>12,509</td><td>0.0014</td><td>0.7956</td></tr><tr><td>80</td><td>2.54</td><td>13,650</td><td>0.0015</td><td>0.7969</td></tr><tr><td>160</td><td>2.83</td><td>13,364</td><td>0.0015</td><td>0.8179</td></tr><tr><td>640</td><td>2.83</td><td>15,566</td><td>0.0017</td><td>0.8112</td></tr><tr><td>1280</td><td>3.17</td><td>16,083</td><td>0.0018</td><td>0.7622</td></tr><tr><td rowspan="5">LightRAG Mix</td><td>40</td><td>3.71</td><td>17,126</td><td>0.0019</td><td>0.8010</td></tr><tr><td>80</td><td>3.39</td><td>18,292</td><td>0.0020</td><td>0.7991</td></tr><tr><td>160</td><td>3.09</td><td>18,573</td><td>0.0020</td><td>0.8044</td></tr><tr><td>640</td><td>3.70</td><td>20,166</td><td>0.0022</td><td>0.7895</td></tr><tr><td>1280</td><td>3.62</td><td>20,361</td><td>0.0022</td><td>0.7596</td></tr></table>

<table><tr><td>Engine</td><td>D</td><td>Lat. (s)</td><td>Tok.</td><td>Cost ($)</td><td> $Q _ { \mathrm { t o t a l } }$ </td></tr><tr><td rowspan="5">HiRAG Hi</td><td>40 80</td><td>5.92</td><td>11,319</td><td>0.0013</td><td>0.7761</td></tr><tr><td></td><td>2.32</td><td>23,914</td><td>0.0025</td><td>0.7706</td></tr><tr><td>160</td><td>3.92</td><td>22,453</td><td>0.0024</td><td>0.8049</td></tr><tr><td>640</td><td>2.99</td><td>26,958</td><td>0.0028</td><td>0.7841</td></tr><tr><td>1280</td><td>10.95</td><td>29,988</td><td>0.0031</td><td>0.7668</td></tr><tr><td rowspan="5">HiRAG Local</td><td>40</td><td>2.26</td><td>6,640</td><td>0.0008</td><td>0.7680</td></tr><tr><td>80</td><td>1.96</td><td>8,833</td><td>0.0010</td><td>0.7860</td></tr><tr><td>160</td><td>8.77</td><td>8,029</td><td>0.0009</td><td>0.8115</td></tr><tr><td>640</td><td>8.24</td><td>10,834</td><td>0.0012</td><td>0.7924</td></tr><tr><td>1280</td><td>8.03</td><td>14,296</td><td>0.0016</td><td>0.7768</td></tr><tr><td rowspan="5">HiRAG Global</td><td>40</td><td>2.39</td><td>7,764</td><td>0.0009</td><td>0.7630</td></tr><tr><td>80</td><td>2.16</td><td>15,935</td><td>0.0017</td><td>0.8069</td></tr><tr><td>160</td><td>8.06</td><td>15,164</td><td>0.0016</td><td>0.8059</td></tr><tr><td>640</td><td>5.35</td><td>17,540</td><td>0.0019</td><td>0.7905</td></tr><tr><td>1280</td><td>3.17</td><td>20,112</td><td>0.0021</td><td>0.7731</td></tr><tr><td rowspan="5">HiRAG Bridge</td><td>40</td><td>2.02</td><td>5,584</td><td>0.0007</td><td>0.7210</td></tr><tr><td>80</td><td>2.33</td><td>11,362</td><td>0.0012</td><td>0.7780</td></tr><tr><td>160</td><td>3.22</td><td>10,035</td><td>0.0011</td><td>0.8047</td></tr><tr><td>640</td><td>2.89</td><td>13,245</td><td>0.0015</td><td>0.7812</td></tr><tr><td>1280</td><td>2.68</td><td>16,631</td><td>0.0018</td><td>0.7602</td></tr><tr><td rowspan="5">HiRAG No-Bridge</td><td>40</td><td>2.35</td><td>11,330</td><td>0.0013</td><td>0.7760</td></tr><tr><td>80</td><td>2.23</td><td>20,305</td><td>0.0022</td><td>0.7915</td></tr><tr><td>160</td><td>4.43</td><td>19,425</td><td>0.0021</td><td>0.8050</td></tr><tr><td>640</td><td>3.52</td><td>22,890</td><td>0.0024</td><td>0.7788</td></tr><tr><td>1280</td><td>2.70</td><td>26,432</td><td>0.0028</td><td>0.7541</td></tr><tr><td rowspan="5">LinearRAG</td><td>40</td><td>5.56</td><td>3,266</td><td>0.0007</td><td>0.8081</td></tr><tr><td>80</td><td>8.05</td><td>2,649</td><td>0.0005</td><td>0.7826</td></tr><tr><td>160</td><td>8.05</td><td>2,800</td><td>0.0006</td><td>0.8223</td></tr><tr><td>640</td><td>6.92</td><td>2,904</td><td>0.0006</td><td>0.7484</td></tr><tr><td>1280</td><td>8.01</td><td>3,077</td><td>0.0006</td><td>0.7841</td></tr><tr><td rowspan="5">LiteRAG</td><td>40</td><td>1.74</td><td>2,158</td><td>0.0003</td><td>0.8090</td></tr><tr><td>80</td><td>1.59</td><td>2,195</td><td>0.0003</td><td>0.8277</td></tr><tr><td>160</td><td>1.33</td><td>2,170</td><td>0.0003</td><td>0.8398</td></tr><tr><td>640</td><td>1.48</td><td>2,235</td><td>0.0003</td><td>0.8090</td></tr><tr><td>1280</td><td>1.42</td><td>2,291</td><td>0.0003</td><td>0.7980</td></tr></table>

Table 4: Complete scalability results for all query engines across all DistComp dataset sizes (D), detailing end-toend latency, token consumption, cost per query, and $Q _ { \mathrm { t o t a l } }$

## B Disaggregated Quality Metrics

This appendix reports the disaggregated components of the composite quality metric $Q _ { \mathrm { t o t a l } }$ for the top-performing configurations on the 1280-document corpus.

The composite metric is defined in Equation 9, with weights 0.6, 0.25, and 0.15 assigned to the judgebased, semantic, and lexical components, respectively (as introduced in Section 4.4). The weights were fixed on a held-out validation split after comparing candidate mixtures of judge-based, semantic, and lexical signals. The largest weight is assigned to $Q _ { \mathrm { j u d g e } }$ because it is the only component that directly evaluates multi-hop reasoning quality, factual correctness, and question relevance in an integrated way. In our implementation, $Q _ { \mathrm { j u d g e } }$ is normalized to the [0, 1] range from the aggregate scores of Correctness,

Completeness, and Relevance. $Q \mathrm { { s e m } }$ provides a softer semantic alignment signal, while $Q _ { \mathrm { l e x } }$ (ROUGE-L) acts as a stricter lexical grounding term that penalizes severe wording-level drift or unsupported terminology.

Table 5 displays the raw scores for each sub-metric. LiteRAG demonstrates consistently high performance across all three independent evaluators, suggesting that its efficiency (detailed in Section 5.1) does not come at the cost of semantic or factual degradation.

<table><tr><td>System (Mode)</td><td> $Q _ { \mathrm { j u d g e } }$ </td><td> $Q \mathrm { { s e m } }$ </td><td> $Q \mathrm { { _ { l e x } } }$ </td><td> $Q _ { \mathrm { t o t a l } }$ </td></tr><tr><td>LiteRAG (Default)</td><td>0.909</td><td>0.866</td><td>0.242</td><td>0.798</td></tr><tr><td>GraphRAG (Basic)</td><td>0.904</td><td>0.857</td><td>0.119</td><td>0.775</td></tr><tr><td>GraphRAG (Drift) GraphRAG (Global)</td><td>0.732</td><td>0.807</td><td>0.111</td><td>0.657</td></tr><tr><td></td><td>0.822</td><td>0.827</td><td>0.095</td><td>0.714</td></tr><tr><td>GraphRAG (Local)</td><td>0.912</td><td>0.837</td><td>0.108</td><td>0.773</td></tr><tr><td>LightRAG (Global)</td><td>0.774</td><td>0.827</td><td>0.155</td><td>0.694</td></tr><tr><td>LightRAG (Hybrid)</td><td>0.883</td><td>0.850</td><td>0.138</td><td>0.763</td></tr><tr><td>LightRAG (Local)</td><td>0.879 0.877</td><td>0.852</td><td>0.144</td><td>0.762</td></tr><tr><td>LightRAG (Mix) HiRAG (Bridge)</td><td>0.881</td><td>0.850</td><td>0.137</td><td>0.760</td></tr><tr><td>HiRAG (Global)</td><td>0.907</td><td>0.852</td><td>0.124</td><td>0.760</td></tr><tr><td></td><td></td><td>0.854</td><td>0.102</td><td>0.773</td></tr><tr><td>HiRAG (Hi)</td><td>0.894</td><td>0.852</td><td>0.117</td><td>0.767</td></tr><tr><td>HiRAG (Local)</td><td>0.912</td><td>0.852</td><td>0.113</td><td>0.777</td></tr><tr><td>HiRAG (No-Bridge) LinearRAG (Default)</td><td>0.875 0.851</td><td>0.849 0.875</td><td>0.110 0.365</td><td>0.754 0.784</td></tr></table>

Table 5: Disaggregated scores for LLM-as-a-judge $( Q _ { \mathrm { j u d g e } } )$ , semantic similarity $( Q _ { \mathrm { s e m } } )$ , lexical overlap $( Q _ { \mathrm { l e x } } ) ,$ and the composite metric $( Q _ { \mathrm { t o t a l } } )$ on the $D _ { 1 2 8 0 }$ benchmark.

## C Prompt Templates

To ensure reproducibility, we provide the exact prompt templates used for context assembly, final answer generation, and the LLM-as-a-judge evaluation. These templates were extracted directly from the system’s source code.

## C.1 LiteRAG Context Assembly and Generation Prompts

As described in Section 3.3, LiteRAG avoids dumping raw text by converting graph relationships into highly dense, structured sections. During the final response generation phase, the system prompt and user prompt are constructed as follows:

System Prompt   
You are a helpful assistant that answers questions based on provided knowledge graph context. Be   
accurate, cite specific entities when relevant, and acknowledge if information is incomplete.

```markdown
User Prompt
## Knowledge Graph Context
[Direct Evidence (Source Text)]
[Graph Context]: ...
[Graph Reasoning Chains (Connections)]
• **Entity A** is connected to **Entity B**
via *Description of the relationship*
...
[Entity Definitions]
**Entity A**: Description of Entity A...
## Question
{user_query}
## Answer
Based on the knowledge graph context above, provide a clear and accurate answer:
```

By pre-computing the logical connections in the Graph Reasoning Chains section, the prompt makes relationships explicit and reduces the need for the generator to infer them from dispersed passages, which lowers token consumption and the inferential burden on the generator.

## C.2 LLM-as-a-judge Evaluation Prompt

For the judge-based evaluation $( Q _ { \mathrm { j u d g e } } )$ , the independent evaluator utilizes a detailed, criteria-based prompt designed to enforce rigorous, step-by-step reasoning. The exact prompt used in the benchmarking framework is as follows:

Evaluation Prompt   
You are an impartial and scientific evaluator. Your task is to assess a generated answer against   
a ground truth reference, based on a specific question. Evaluate the answer based on the criteria   
of Correctness, Completeness, and Relevance.   
\*\*Evaluation Task:\*\*   
1. Analyze the Question: Understand what the user is asking for.   
- Question: "{question}"   
2. Analyze the Ground Truth: This is the reference for what a correct and complete answer should   
contain.   
- Ground Truth: "{ground\_truth}"   
3. Analyze the Generated Answer: This is the answer you must evaluate.   
- Generated Answer: "{prediction}"   
4. Perform a Step-by-Step Evaluation:   
- Correctness (0.0-1.0): Is the information in the Generated Answer factually accurate and   
consistent with the Ground Truth? Does it contradict the ground truth or introduce plausible but   
unsupported information (hallucinations)?   
- Completeness (0.0-1.0): Does the Generated Answer cover all the key information and essential   
points present in the Ground Truth?   
- Relevance (0.0-1.0): Does the Generated Answer directly address the user’s Question? Is the   
answer on-topic?   
5. Provide Scores and Reasoning: Based on your analysis, provide a score from 0.0 (terrible) to   
1.0 (perfect) for each criterion and write a detailed reasoning for your scores.   
Return your evaluation as JSON with this exact format:   
{   
"correctness": <float 0-1>,   
"completeness": <float 0-1>,   
"relevance": <float 0-1>,   
"reasoning": "<string>"   
}

## D LiteRAG Hyperparameter Configuration

This appendix reports the exact LiteRAG configuration used in all experiments. As noted in Section 4, hyperparameters were selected by grid search on a held-out validation split under the same model and graph-construction setup used for the reported results.

Table 6 instantiates the parameters introduced in the LiteRAG method (Section 3). For Phase 1 (Section 3.1), the lexical term $S _ { \mathrm { l e x } }$ is implemented as a combination of exact and fuzzy matching, so the table decomposes the single lexical weight $\beta$ in Equation 1 into its implementation-level components. For Phase 2 (Section 3.2), the reported runs used an explicit hop-depth budget $k _ { \operatorname* { m a x } } = 3$ together with a per-anchor expansion cap $N _ { \mathrm { m a x } } = 5 0$ , so exploration was governed by the dynamic thresholding rule under bounded traversal budgets.

These settings were used for every LiteRAG result reported in the main paper. Together, they reflect the intended retrieval bias of the method: preserve exact lexical anchors when necessary, expand conservatively from strong initial evidence, and rank Phase 3 (Section 3.3) entities primarily by repeated support across anchor-induced traversals rather than by generic graph centrality alone.

<table><tr><td>Phase</td><td>Parameter</td><td>Value</td><td>Description</td></tr><tr><td rowspan="7">Anchor</td><td>candidate_pool_size (K)</td><td>8</td><td>Number of top vector-retrieved candidates retained before an- chor scoring. This keeps Phase 1 local to a small semantically relevant pool rather than scoring the full graph.</td></tr><tr><td>min_anchor_score (τanchor)</td><td>0.30</td><td>Minimum composite anchor score required for a node to enter A. The selected value preserves multiple plausible entry</td></tr><tr><td>semantic_weight (α)</td><td>0.40</td><td>points without flooding Phase 2 with weak anchors. Contribution of semantic similarity to the Phase 1 anchor score.</td></tr><tr><td>lexical_weight (β)</td><td>0.45</td><td>Total contribution of lexical evidence to the Phase 1 anchor score. This relatively large weight protects exact terminology and named entities that may be poorly captured by embed-</td></tr><tr><td>keyword_exact_weight</td><td>0.30</td><td>dings alone. Exact-match component of  $S _ { \mathrm { l e x } } .$  This is the dominant share of β and rewards direct terminology overlap.</td></tr><tr><td>keyword_fuzzy_weight</td><td>0.15</td><td>Fuzzy-match component of  $S _ { \mathrm { l e x } } ,$  used to recover near matches and minor surface-form variations without overpow- ering exact lexical evidence.</td></tr><tr><td>community_weight (γ)</td><td>0.15</td><td>Contribution of community-level relevance to the Phase 1 anchor score. The smaller weight keeps community evidence supportive rather than dominant.</td></tr><tr><td rowspan="6">Subgraph Expansion</td><td>max_exploration_depth  $\left( k _ { \operatorname* { m a x } } \right)$ </td><td>3</td><td>Maximum hop distance from each anchor during Phase 2. This bounds query-time search radius and is the primary explicit exploration budget used in the reported runs.</td></tr><tr><td>max_expansions_per_anchor  $( N _ { \mathrm { m a x } } )$ </td><td>50</td><td>A per-anchor expansion cap of 50 nodes was used to prevent unbounded exploration from highly connected anchors.</td></tr><tr><td>min_relevance_threshold  $\left( \tau _ { \mathrm { b a s e } } \right)$ </td><td>0.25</td><td>Base floor in the dynamic threshold  $\tau _ { \mathrm { d y n } } ( q , A )$  . This prevents traversal from expanding on very weak evidence even when anchors are modest.</td></tr><tr><td>signal_amplification (λ)</td><td>0.25</td><td>Scaling factor that raises the traversal threshold when the initial anchors are strong. The selected value makes expansion more selective for well-grounded queries while still allowing</td></tr><tr><td>relevance_decay_factor (d)</td><td>0.70</td><td>broader search when anchor quality is weaker. Multiplicative decay applied per hop in  $R ( q , u , v , k )$  . This encourages shorter explanatory paths while still permitting multi-hop retrieval.</td></tr><tr><td>degree_influence (δ)</td><td>0.05</td><td>Strength of the hub penalty. The small value suppresses generic hubs without over-penalizing structurally important nodes.</td></tr><tr><td></td><td>community_cohesion (κ)</td><td>0.80</td><td>Within-community protection factor in  $\boldsymbol { \omega } ( u , v )$  . This sub- stantially relaxes hub penalization for transitions that remain inside the same topical region.</td></tr><tr><td rowspan="4">Consensus Ranking</td><td>intersection_weight (ρ₁)</td><td>0.30</td><td>Weight on  $I _ { \mathrm { r a t e } } ( v )$  , rewarding entities reached by multiple an- chor expansions. This is the largest single Phase 3 weight be- cause repeated recovery across paths is treated as the strongest</td></tr><tr><td>semantic_weight (ρ2)</td><td>0.25</td><td>indicator of relevance. Weight on semantic alignment in the final ranking. This keeps the retained context tightly tied to the user query after graph</td></tr><tr><td>proximity_weight (ρ3)</td><td>0.25</td><td>expansion. Weight on topological proximity to the anchor set. Matching the semantic term, it favors nearby evidence without exclud-</td></tr><tr><td>structural_weight (ρ₄)</td><td>0.20</td><td>ing informative multi-hop entities. Weight on structural centrality. This is the smallest Phase 3 weight so that generic graph prominence does not override</td></tr></table>

Table 6: Exact LiteRAG hyperparameters used in the reported experiments. The table instantiates the abstract parameters from Section 3 and records the implementation-level decomposition of the lexical anchor term.

## E Detailed Engine Configurations

This appendix provides the full configuration parameters for all retrieval engines evaluated in this study. To ensure readability and clear side-by-side comparison, we present the parameter configurations using standard academic tables. We provide both the standard settings used for the primary benchmarking (Table 1) and the budgeted settings used for the fixed-budget analysis (Section 5.3).

The default configurations used in this study were obtained from the standard parameters in the repositories of the different architectures.

## E.1 LiteRAG Configuration

The LiteRAG configuration and hyperparameters are defined in Appendix D. LiteRAG does not require a separate budgeted configuration because its algorithmic design naturally produces a context within the 2,000-token target budget.

## E.2 LinearRAG Configuration

The configuration for LinearRAG follows the suggested defaults for document-based retrieval as per its reference implementation.

<table><tr><td>Parameter</td><td>Standard</td><td>Budgeted</td></tr><tr><td>spacy_model</td><td>&quot;en_core_web_trf&quot;</td><td></td></tr><tr><td>max_workers</td><td>4</td><td>4</td></tr><tr><td>retrieval_top_k</td><td>5</td><td>3</td></tr><tr><td>max_iterations</td><td>3</td><td>2</td></tr><tr><td>top_k_sentence</td><td>1</td><td>1</td></tr><tr><td>passage_ratio</td><td>1.5</td><td>1.5</td></tr><tr><td>iteration_threshold</td><td>0.5</td><td>0.5</td></tr><tr><td>use_vectorized</td><td>false</td><td>false</td></tr></table>

Table 7: LinearRAG configuration comparison.

## E.3 LightRAG Configuration

To test LightRAG under a fixed budget, we reduced the max\_total\_tokens and associated entity/relation limits.

<table><tr><td>Parameter</td><td>Standard</td><td>Budgeted</td></tr><tr><td>top_k</td><td>60</td><td>60</td></tr><tr><td>max_entity_tokens</td><td>6000</td><td>600</td></tr><tr><td>max_relation_tokens</td><td>8000</td><td>600</td></tr><tr><td>max_total_tokens</td><td>30000</td><td>2000</td></tr><tr><td>enable_rerank</td><td>false</td><td>false</td></tr></table>

Table 8: LightRAG configuration comparison.

## E.4 HiRAG Configuration

For the budgeted version, we enforced limit\_tokens: true and capped the total context at 2,000 tokens.

## E.5 Microsoft GraphRAG Summary

The budgeted version capped max\_context\_tokens at 4,000 (the lowest functional setting for DRIFT) and reduced the basic search k.

## F Exploratory Multi-Entity Efficiency Comparison

As an exploratory analysis extending the findings in Section 5, this appendix reports a focused comparison between LiteRAG and LinearRAG on a small DistComp subset designed to probe multi-entity queries. The subset contains 20 queries, with 5 queries in each of four bins defined by the number of distinct entities or concepts explicitly mentioned in the question. The purpose of this analysis is focused: to examine whether the efficiency gap between the two methods remains visible as the number of entities in the query increases.

<table><tr><td>Parameter</td><td>Standard</td><td>Budgeted</td></tr><tr><td>top_k</td><td>20</td><td>20</td></tr><tr><td>top_m</td><td>10</td><td>10</td></tr><tr><td>max_token_text_unit</td><td>20000</td><td></td></tr><tr><td>max_token_local</td><td>20000</td><td></td></tr><tr><td>max_token_bridge</td><td>12500</td><td></td></tr><tr><td>max_token_report</td><td>12500</td><td></td></tr><tr><td>max_total_tokens</td><td></td><td>2000</td></tr><tr><td>limit_tokens</td><td>false</td><td>true</td></tr></table>

Table 9: HiRAG configuration comparison.
<table><tr><td>Mode / Parameter</td><td>Standard</td><td>Budgeted</td></tr><tr><td>Local Search max_context_tokens</td><td>12000</td><td>4000</td></tr><tr><td>Global Search</td><td></td><td>4000</td></tr><tr><td>max_context_tokens DRIFT Search</td><td>12000</td><td></td></tr><tr><td>n_depth</td><td>2</td><td></td></tr><tr><td>concurrency</td><td>32</td><td></td></tr><tr><td>loc_search_max_data_toks</td><td></td><td>4000</td></tr><tr><td>Basic Search</td><td></td><td></td></tr><tr><td>k</td><td></td><td>4</td></tr><tr><td>max_context_tokens</td><td></td><td>4000</td></tr></table>

Table 10: GraphRAG configuration comparison.

<table><tr><td>Ent.</td><td>L-Lat.</td><td>R-Lat.</td><td>L-Tok.</td><td>R-Tok.</td><td>L-Cost</td><td>R-Cost</td></tr><tr><td>1</td><td>1.02</td><td>2.92</td><td>2,009</td><td>2,365</td><td>0.00022</td><td>0.00033</td></tr><tr><td>2</td><td>1.11</td><td>2.59</td><td>2,120</td><td>3,077</td><td>0.00024</td><td>0.00044</td></tr><tr><td>3</td><td>1.00</td><td>2.12</td><td>1,881</td><td>2,475</td><td>0.00020</td><td>0.00034</td></tr><tr><td>4</td><td>1.09</td><td>3.77</td><td>1,961</td><td>2,833</td><td>0.00022</td><td>0.00048</td></tr></table>

Table 11: Efficiency comparison on the 20-query multi-entity subset. L denotes LiteRAG and R denotes LinearRAG. Latency is reported in seconds per query and cost in dollars per query.

Across all four bins in Table 11, LiteRAG remains faster, less expensive, and more token-efficient than LinearRAG. LiteRAG stays between 1.00 and 1.11 seconds and between 1,881 and 2,120 tokens, whereas LinearRAG ranges from 2.12 to 3.77 seconds and from 2,365 to 3,077 tokens. The quality differences on this subset are mixed in the lower-entity bins, while LiteRAG records the higher score on the 4-entity bin (0.81 vs. 0.74). Accordingly, the clearest conclusion from this subset is that LiteRAG preserves a more stable efficiency profile across query-complexity bins.

## G Example Queries by Entity Complexity

This appendix provides representative examples from the multi-entity subset summarized in Appendix F.   
The queries are grouped by the number of distinct entities that must be related to produce a correct answer.

## 1 Entity (Simple Lookup)

•“What specific hardware mechanism does ’Poseidon’ use to protect heap metadata?”

## 2 Entities (Relational Reasoning)

•“How does ’SparkFlow’ utilize ’Base Recalibrator’ in its design?”

•“In the context ofbinary hardening, what capability did ’RedFat’ demonstrate regarding ’Google Chrome’?”

## 3 Entities (Multi-hop Synthesis)

•“What is the relationship between ’HopsFS-CL’, ’HopsFS’, and ’HDFS’?”

•“How does ’Siren’ improve upon current ’Byzantine-robust’ aggregation rules in ’Federated Learning’?”

## 4 Entities (Comparative Analysis)

•“In the scalability evaluation of deep learning frameworks, which framework among ’Tensorflow’, ’Keras’, ’MXNet’, and ’PyTorch’ wasfound to be the most efficient?”

•“How much did ’EndGraph’ improve preprocessing performance compared to the group of ’LFGraph’, ’PowerLyra’, ’PowerGraph’, and ’D-Galois’?”