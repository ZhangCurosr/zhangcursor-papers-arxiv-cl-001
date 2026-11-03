# Scalaris: A Scalable Enterprise Memory Architecture for Conversational AI Assistants

Devarsh Gandhi Sarthak Pawar Hemprasad Badgujar Vaibhav Rathore

Siddharth Sureka Neha Bhargava

Motilal Oswal Financial Services Ltd

{devarsh.gandhi, sarthak.pawar, hemprasad.badgujar, rathore.vaibhav, siddharth.s, neha.bhargava}@motilaloswal.com

## Abstract

Enterprise conversational systems require memory to personalize interactions, retain informa tion across sessions, and maintain conversational coherence. At enterprise scale, how ever, memory must not only retrieve relevant information accurately but also provide it effi ciently to the answer-generation model. Exist ing memory systems face a trade-off between retrieval accuracy and context efficiency. Re trieving more context can improve recall but increases inference cost and latency. We present Scalaris, a scalable enterprise memory architec ture that addresses this trade-off through a uni fied long-term and short-term memory design. We propose that user interactions and assistant responses possess distinct characteristics and roles in a conversation: user interactions cap ture evolving knowledge about the user and provide value across sessions, whereas assis tant responses must be stored efficiently so that only information relevant to the current query is retrieved. We therefore treat them separately and support each with specialized representa tions and ingestion/retrieval pipelines. Scalaris reaches 92.2% accuracy on LongMemEval-S and 90.77% on LoCoMo using Qwen3-32B, state-of-the-art results on both benchmarks, while supplying the answer-generation model with approximately 10× fewer retrieved con text tokens than Hindsight, a leading memory system for conversational AI. When integrated with our proprietary closed-source AI assis tants, Scalaris improved overall response qual ity by 36%. Together, these results establish Scalaris as a cost-effective framework for build ing scalable, personalized conversational AI systems.

## 1 Introduction

The emergence of large language models has enabled organizations to deploy conversational AI systems across user-facing applications such as customer support and sales assistance (Yi et al., 2026).

Memory is essential to these systems for maintaining continuity across conversations and personalizing interactions (Wu et al., 2025b). However, simply providing an assistant with prior message history becomes limiting as conversations grow longer and span multiple sessions, since model performance degrades over long contexts (Liu et al., 2024). Existing memory frameworks have therefore explored mechanisms for selectively storing and retrieving information from past conversations (Zhang et al., 2025a), but they often face a trade-off between retrieval accuracy and context efficiency. Zep (Rasmussen et al., 2025), for example, retrieves relatively little context but achieves lower accuracy on long-term memory benchmarks, whereas Hindsight (Latimer et al., 2026) achieves substantially higher accuracy while retrieving considerably more. An effective enterprise memory system must therefore retrieve the information required to answer a query accurately without unnecessarily increasing the context supplied to the answer-generation model (Lewis et al., 2020; Shi et al., 2023; Chhikara et al., 2025).

Conversational memory must also operate across two different timescales. Information such as user preferences, interests, and circumstances may remain useful across sessions and must evolve as new interactions occur (Xu et al., 2025; Zhang et al., 2025b), creating the need for long-term memory. At the same time, an assistant must retain the immediate conversational state required to interpret follow-up questions, resolve references, and maintain coherence within an ongoing session (Packer et al., 2024; Yu et al., 2026), motivating short-term memory. A scalable conversational memory architecture must therefore support both persistent crosssession knowledge and immediate within-session context.

Existing systems typically place all conversational turns under a shared memory representation despite the different roles of user interactions and assistant responses. User interactions contain persistent facts, preferences, and evolving relationships that remain valuable across sessions; assistant responses are largely contextual and often obsolete outside their original dialogue. User interactions therefore benefit from relational representations that capture evolving user knowledge, while assistant responses are better stored as independently retrievable atomic units, so that only the relevant portion of a response is recalled rather than the whole.

![](images/52055d5052d341197c9b7ed8a357bdc37acc057cbd5bd4a00c0e408ba3a59399.jpg)  
Figure 1: Overview of Scalaris’s source-aware memory organization. User interactions are processed to maintain relevant facts, relevant entities, and user and session summaries in a temporal knowledge graph. Assistant responses are decomposed into independently retrievable atomic units and stored in a vector store. In parallel, both user interactions and assistant responses are preserved verbatim in a shared history store.

Our Proposal: We present Scalaris, a unified long-term and short-term memory architecture designed around this distinction. As illustrated in Figure 1, Scalaris maintains separate memory representations for user interactions and assistant responses. User interactions are organized in a temporal knowledge graph that maintains relevant facts, entities, and evolving user information, together with user and session summaries. Assistant responses are decomposed into independently retrievable atomic units and stored in a vector database, allowing only information relevant to the current query to be retrieved. In parallel, user interactions and assistant responses are preserved verbatim in a shared history store for recent conversational context. By providing specialized ingestion and retrieval pipelines for these different forms of memory, Scalaris aims to preserve the information needed for accurate recall while limiting unnecessary retrieved context.

Our main contributions are as follows:

(1) We present Scalaris, a unified long-term and short-term memory framework for building scalable, enterprise-ready conversational AI assistants.

(2) We propose that user interactions and assistant responses exhibit distinct characteristics that call for different memory representations, and we support each with a specialized ingestion and retrieval pipeline.

(3) We achieve state-of-the-art performance on both LongMemEval-S (92.2%) and LoCoMo (90.77%) using the relatively small Qwen3-32B model, while requiring approximately 10× fewer retrieved context tokens than Hindsight.

## 2 Related Work

Long-Term and Knowledge-Graph Memory Representations. Long-term memory frameworks let conversational agents retain information across sessions through varied representations. Knowledge-graph approaches such as Zep (Rasmussen et al., 2025) and Supermemory (Supermemory, 2025) capture entities and their evolving relationships over time, while Mem0 (Chhikara et al., 2025) combines graph components with selective consolidation. LangMem (LangChain, 2025) provides a general-purpose toolkit for extracting and consolidating memories. Unit-based and reflective models such as A-Mem (Xu et al., 2025) treat memory as interconnected units that update as new interactions occur; RGMem (Tian et al., 2026) de rives higher-level user profiles from episodic logs, and RMM (Tan et al., 2025) applies reflective man agement to long-term dialogue. Others target efficiency through different mechanisms: Simple-Mem (Liu et al., 2026) compresses conversations into compact semantic units, FraCom (Ke et al., 2025) composes fine-grained fragments at retrieval time, RecMem (Dai et al., 2026) optimizes memory construction cost, and Zero-Mem (Xiao et al., 2026) eliminates LLM calls during memory operations through deterministic memory management. Temporal Organization and Hierarchical Architectures. Operating across immediate and multisession contexts requires tracking temporal evolu tion and balancing short- against long-term stores. MemGPT (Packer et al., 2024) introduced hierarchical, OS-like memory management, while Age-Mem (Yu et al., 2026) learns actions across longand short-term tiers. To handle temporal shifts over long interaction spans, TiMem (Li et al., 2026), APEX-MEM (Banerjee et al., 2026), and THEA NINE (Ong et al., 2025) structure information around timelines and state updates. Complementary work explores content-based categorization: Hindsight (Latimer et al., 2026) organizes memory into world knowledge, agent experience, entity observations, and beliefs. This implicitly separates assistant-originated opinions from user observations, but the division follows from epistemic role rather than a deliberate partition by the origin of the information.

Partitioning User Interactions and Assistant Responses. Despite these developments, no existing system uses the functional distinction between user interactions and assistant responses as an explicit organizing principle. User inputs call for relational, graph-based representations that track changing preferences, facts, and context across sessions, whereas assistant responses are better decomposed into independently retrievable atomic units so that downstream models receive only the relevant excerpts. Scalaris closes this gap, unifying short- and long-term stores while applying a representation tailored to each source.

## 3 Methodology

## 3.1 System Architecture

Scalaris is a unified long-term and short-term memory architecture designed for conversational AI assistants. As illustrated in Figure 2, the architecture is partitioned into a long-term component tailored for cross-session recall and a short-term component dedicated to intra-session continuity.

The long-term component is built on Graphiti (Rasmussen et al., 2025; Zep AI, 2024), an open-source temporal knowledge graph framework, which we substantially modify for use as a memory layer. It consists of a user summary and Relevant user interactions. The user summary maintains an evolving, holistic profile of the user’s attributes, behavioral patterns, and focus areas. It is updated incrementally after each interaction and periodically rebuilt from historical logs to prevent semantic drift. Relevant user interactions extract three elements from user messages: Relevant Facts, which capture temporal semantic data; Relevant Entities, which represent previously discussed people, organizations, or concepts alongside their context; and Relevant Past Messages, which preserve the verbatim user interactions.

The short-term component consists of a session summary, the Last-N Exchanges, and Relevant Assistant Messages. The session summary compactly tracks the ongoing dialogue state, while the Last-N Exchanges provide the verbatim recent context required for immediate coreference resolution and intent tracking. For Relevant Assistant Messages, we leverage a vector database to store assistant responses as independently retrievable atomic units. The assistant’s outputs are fragmented into self-contained units, ensuring only informative segments are retrieved rather than full, verbose responses. We intentionally keep Relevant Assistant Messages in short-term memory for two reasons. First, assistant responses are often not useful across sessions, and enabling cross-session retrieval increases retrieved context with limited benefit. Second, in domains with rapidly changing information like finance, retrieving stale assistant responses increases the likelihood of hallucinations. This behavior is nonetheless configurable for applications that require assistant responses to persist across sessions. Figure 3 details the complete ingestion and retrieval pipeline.

## 3.2 Ingestion

Relevant user interactions. Each incoming user message is ingested as an episode from which entities, facts, and temporal information are extracted and linked to the memory graph. The memory uses a bi-temporal model that tracks both a fact’s validity interval and when it was learned or recorded, preserving historical changes. Each episode is processed as follows:

(a) Entity Extraction. Entities are extracted from each episode, and a concise summary of up to 250 characters is maintained for each entity to capture its conversational context.

(b) Entity Resolution. Extracted entities are matched against existing entities using a resolution pipeline that combines exact matching, lexical similarity, and embedding-based similarity. If a match is identified, the entity is linked to its existing graph representation; otherwise, a new representation is created.

(c) Fact Extraction. Atomic facts are extracted from the episode and represented as relationships between resolved entities, capturing the specific information conveyed by the user and its associated temporal context.

(d) Edge De-duplication. Newly extracted relationships are compared against existing edges to identify duplicate information, preventing repeated interactions from unnecessarily creating redundant relationships in the graph.

![](images/7c10a28203a097a3f32c6738645e9c34634e9ce4f23bd1c311a115369af80af6.jpg)  
User wants to compare the Civic with the Corolla  needs a detailed comparison focused on fuel economy, with trade-offs Action: Civic information is already in memory, so fetch Corolla data and give a detailed comparison.  
Figure 2: An illustrative instance of context retrieval in Scalaris. Long-term memory supplies the user’s preference for fuel economy together with prior Honda Civic interactions, while short-term memory supplies the Relevant Assistant Messages, session summary, and recent turns (only the latest exchange is shown, for visual clarity). Together these resolve the referent, let the agent determine that only Corolla information must be retrieved, and enable personalization by focusing on fuel economy and including trade-offs.

(e) Temporal Extraction. Temporal expressions associated with extracted facts are identified and normalized to determine the validity interval of each relationship, including explicit dates as well as relative expressions found in the conversation context.

(f) Edge Invalidation. When a newly extracted fact contradicts an existing relationship, the validity of the previous edge is terminated at the point when the new fact becomes valid, rather than deleting the historical edge, preserving the evolution of the user’s information over time.

For each extracted fact, the original user message is retained alongside its corresponding memory representation, preserving the source utterance for subsequent retrieval.

User and Session Summaries. User and session summaries are maintained in the same graph-based memory layer and follow a common update strategy. Both summaries are updated incrementally as new interactions are observed and periodically reconstructed from the underlying conversation history, with interactions provided in temporal order during reconstruction. Let $\sigma _ { t }$ denote the summary at time $t , x _ { t }$ the newly observed interaction at time $t ,$ and $H _ { t }$ the underlying user interaction history available at the reconstruction point:

$$
\sigma _ { t } = \left\{ \begin{array} { l l } { { \mathrm { R e b u i l d } ( H _ { t } ) , } } & { { \mathrm { a t ~ r e c o n s t r u c t i o n ~ i n t e r v a l s , } } } \\ { { \mathrm { U p d a t e } ( \sigma _ { t - 1 } , x _ { t } ) , } } & { { \mathrm { o t h e r w i s e . } } } \end{array} \right.\tag{1}
$$

Relevant Assistant Messages. Assistant responses are fragmented into atomic memory units to retain only informative content from each response and enable retrieval of only the relevant portions when needed. This is achieved through the following steps:

(a) Contextual Windowing. The conversation is divided into fixed-size windows of dialogue, with each window providing the surrounding context required to interpret the assistant response being processed.

(b) Information Filtering. Each dialogue window is evaluated for the information content of its assistant response, and windows containing only greetings, acknowledgements, or other low-value interactions are discarded to avoid storing uninformative content.

(c) Atomic Memory Construction. Informative assistant responses are decomposed into selfcontained atomic memory units with temporal awareness. During construction, references to preceding dialogue are resolved, and related information is combined when necessary to form a complete representation.

(d) Keyword and Entity Extraction. Relevant keywords and entities are extracted and stored alongside each memory unit as metadata.

![](images/94b174e03bc628597d16d4335a033a175e1b142f543a21eb949d88823a18f2be.jpg)  
Figure 3: Flow diagram of the ingestion and retrieval pipeline of Scalaris. The ingestion phase organizes user interactions as facts, entities, and summaries in the graph store, assistant responses as atomic units in the vector store, and stores each incoming user query and AI response verbatim in the history store. The retrieval pipeline begins by reformulating the incoming query, which is then used to retrieve information from the graph and vector stores. Summaries are fetched directly from the graph store, and the N most recent exchanges from the history store. All retrieved components are combined during context assembly and passed to the answering system.

Last-N Exchanges. All user-assistant messages are stored verbatim in a history store, preserving their exact wording and ordering.

## 3.3 Retrieval

Query Reformulation. Queries in ongoing conversations often depend on preceding dialogue through pronouns, ellipsis, or implicit references, making them incomplete retrieval keys in isolation. Each incoming query is therefore reformulated using recent conversation history before retrieval. Given the current query q<sub>t</sub> and recent conversation history $H _ { t } ^ { \mathrm { r e c e n t } }$ , a reformulation function produces a self-contained query

$$
q _ { t } ^ { * } = R ( q _ { t } , H _ { t } ^ { \mathrm { r e c e n t } } ) .\tag{2}
$$

The reformulation resolves references, restores omitted subjects, and makes implicit continuations explicit. If $q _ { t }$ is already self-contained, it is retained unchanged, such that $q _ { t } ^ { * } = q _ { t }$ . The resulting query $q _ { t } ^ { * }$ is used for all subsequent retrieval operations.

Relevant user interactions. Relevant user interactions are retrieved through a three-stage search, reranking, and context-construction pipeline.

(a) Search. The reformulated query is used to retrieve candidate facts and entities through semantic, lexical, and graph-traversal search. Semantic search uses dense embedding similarity, computed with Cohere’s embed-english-v3.0 model (Cohere, 2023) to identify facts and entities that are semantically related to the query. Lexical search uses BM25 (Robertson and Zaragoza, 2009) over fact text and entity names to recover exact terms, names, and rare expressions. Graph traversal complements these searches by starting from seed entities and exploring a bounded number of graph hops, retrieving related facts that may not be lexically or semantically similar to the query.

(b) Reranking. Candidate results are reranked using Reciprocal Rank Fusion (RRF) (Cormack et al., 2009), which combines independently ranked lists without requiring their scores to be directly comparable. For a candidate d, the fused scores for facts and entities are computed as:

$$
R _ { \mathrm { f a c t } } ( d ) = \sum _ { r \in \{ \mathrm { s e m , l e x , g r a p h } \} } \frac { 1 } { k + \operatorname { r a n k } _ { r } ( d ) } ,\tag{3}
$$

$$
R _ { \mathrm { e n t i t y } } ( d ) = \sum _ { r \in \{ \mathrm { s e m } , \mathrm { l e x } \} } \frac { 1 } { k + \mathrm { r a n k } _ { r } ( d ) } ,\tag{4}
$$

where d denotes a candidate result, r denotes a retrieval ranking, rank<sub>r</sub>(d) is the rank of d in ranking r, and k is the RRF smoothing constant, which we set to the default value of $k = 6 0$ . This stage improves precision by retaining the highest-ranked candidates across complementary retrieval mechanisms.

(c) Context Construction. The reranked results are converted into the context supplied to the answering system. Each relevant fact is represented with its descriptive text, while each relevant entity is represented by its name and concise summary. Relevant Past Messages can also be retrieved from the metadata of relevant edges to provide the assistant with the exact user utterances from which the corresponding knowledge was derived.

## Relevant Assistant Messages.

(a) Query Planning and Decomposition. Before retrieval begins, the reformulated query is passed to a single LLM call that performs all query-planning operations required by the subsequent retrieval stages. The LLM analyzes the information need and simultaneously generates one or more self-contained query phrasings for semantic retrieval, extracts keywords for lexical retrieval, and identifies attributes—including persons, locations, entities, and temporal expressions—for symbolic retrieval. Relative temporal expressions are resolved into absolute time ranges using the current date and time as a reference.

(b) Hybrid Retrieval. The three retrieval views execute concurrently, each using its corresponding query representation. Semantic retrieval ranks memory units by embedding similarity, lexical retrieval uses BM25 to identify exact or closely matching terms, and symbolic retrieval filters memory units using stored attributes such as persons, locations, entities, and temporal ranges:

$$
\begin{array} { r l } & { R _ { \mathrm { s e m } } = \mathrm { T o p K } _ { n _ { \mathrm { s e m } } } \Big \{ m _ { i } \ \Big | } \\ & { \qquad \quad \cos ( E ( q _ { \mathrm { s e m } } ) , E ( m _ { i } ) ) \geq \tau _ { \mathrm { s e m } } \Big \} , } \\ & { R _ { \mathrm { l e x } } = \mathrm { T o p K } _ { n _ { \mathrm { l e x } } } \{ m _ { i } \ | \ \mathrm { B M 2 5 } ( q _ { \mathrm { l e x } } , m _ { i } ) \geq \tau _ { \mathrm { l e x } } \} , } \\ & { R _ { \mathrm { s y m } } = \mathrm { T o p K } _ { n _ { \mathrm { s y m } } } \{ m _ { i } \ | \ \mathrm { M e t a } ( m _ { i } ) \ | = q _ { \mathrm { s y m } } \} . } \end{array}\tag{5}
$$

Here, $m _ { i }$ denotes a candidate memory unit; q<sub>sem</sub>, $q _ { \mathrm { l e x } } .$ , and $q _ { \mathrm { s y m } }$ denote the query representations generated for semantic, lexical, and symbolic retrieval, respectively; $E ( \cdot )$ denotes the embedding function; $n _ { \mathrm { s e m } } , n _ { \mathrm { l e x } }$ , and $n _ { \mathrm { s y m } }$ specify the number of candidates retained from each retrieval view; and $\tau _ { \mathrm { s e m } }$ and $\tau _ { \mathrm { l e x } }$ are the minimum similarity and BM25 score thresholds, respectively. $\mathrm { M e t a } ( m _ { i } ) \mid = q _ { \mathrm { s y m } }$ indicates that the stored metadata of $m _ { i }$ satisfies the constraints specified by $q _ { \mathrm { s y m } }$

Duplicate memory units returned by multiple views are removed using their unique identifiers, producing a single consolidated set of Relevant Assistant Messages.

Summaries and Last-N Exchanges. The user and session summaries are directly fetched from their corresponding graph nodes, while the most recent N user-assistant exchanges are fetched from the history store verbatim.

Context Assembly. Finally, the outputs of all memory components are concatenated and passed to the answering system.

## 4 Experiments and Results

## 4.1 Experimental Setup

## 4.1.1 Datasets

We evaluate Scalaris on two long-term conversational memory benchmarks: LongMemEval-S (Wu et al., 2025a), comprising 500 questions across six evaluation categories with approximately 115K input tokens per question, and LoCoMo (Maharana et al., 2024), comprising 1,986 questions across 10 multi-session conversations and five categories, with approximately 9K tokens per conversation. For LoCoMo, we follow the category labels and question distribution provided by the Hugging Face version of the dataset.

## 4.1.2 Evaluation Metrics

To assess both the effectiveness and practical viability of the memory systems, we measure accuracy (%), the primary metric for retrieval and reasoning success across all benchmark categories; retrieval efficiency, the average number of context tokens retrieved per query; and latency, the average retrieval time per query in seconds.

## 4.1.3 Implementation Details

For the end-to-end memory system, we consider Qwen3-32B (Qwen Team, 2025) and OSS-120B (OpenAI, 2025) as the underlying models. Following benchmark protocols, GPT-4o (OpenAI, 2024) is used as the answer generator and evaluator.

Because these benchmarks primarily evaluate long-term memory recall and reasoning rather than conversational continuity, we exclude the session summary and Last-N Exchanges during evaluation. We configure Relevant Assistant Messages for cross-session retrieval to align with the benchmarks’ requirements. All other components, including the proposed ingestion and retrieval pipelines, remain unchanged from the core architecture. The recommended hyperparameter ranges and sensitivity analysis are presented in the extended analysis in Appendix A.2. Baseline results in Tables 1 and 2 are taken from Latimer et al. (2025).

## 4.2 Results

## 4.2.1 Performance on LongMemEval-S

Table 1 compares Scalaris with existing memory systems across six categories of LongMemEval-S.

<table><tr><td>Type</td><td>Zep GPT-4o</td><td>Superm. GPT-5</td><td>Superm. Gemini-3</td><td>Hinds. OSS-120B</td><td>Hinds. Gemini-3</td><td>Scalaris Qwen3-32B</td><td>Scalaris OSS-120B</td></tr><tr><td>SSU</td><td>92.9</td><td>97.1</td><td>98.6</td><td>100.0</td><td>97.1</td><td>97.14</td><td>95.71</td></tr><tr><td>SSA</td><td>80.4</td><td>100.0</td><td>98.2</td><td>98.2</td><td>96.4</td><td>83.93</td><td>85.71</td></tr><tr><td>SSP</td><td>56.7</td><td>76.7</td><td>70.0</td><td>86.7</td><td>80.0</td><td>93.33</td><td>80.0</td></tr><tr><td>KU</td><td>83.3</td><td>87.2</td><td>89.7</td><td>92.3</td><td>94.9</td><td>89.74</td><td>89.74</td></tr><tr><td>TR</td><td>62.4</td><td>81.2</td><td>82.0</td><td>85.7</td><td>91.0</td><td>93.98</td><td>90.98</td></tr><tr><td>MS</td><td>57.9</td><td>75.2</td><td>76.7</td><td>81.2</td><td>87.2</td><td>92.48</td><td>92.48</td></tr><tr><td>Overall</td><td>71.2</td><td>84.6</td><td>85.2</td><td>89.0</td><td>91.4</td><td>92.2</td><td>90.6</td></tr></table>

Table 1: Performance comparison on LongMemEval-S (% Accuracy). SSU: single-session-user; SSA: singlesession-assistant; SSP: single-session-preference; KU: knowledge-update; TR: temporal-reasoning; MS: multisession. Superm.: Supermemory; Hinds.: Hindsight. (Bold: best; underline: second best)

Scalaris achieves the highest overall accuracy of 92.2%, outperforming all compared systems while utilizing an average of only 4,116 context tokens per query and maintaining a 0.94 s retrieval latency. It achieves state-of-the-art performance in three critical categories: SSP (93.33%), TR (93.98%), and MS (92.48%). Notably, this 92.2% overall accuracy is achieved with Qwen3-32B, a substantially smaller model than those used by the competing baselines. When paired with OSS-120B, Scalaris still outperforms most baseline configurations with 90.6% accuracy, including Hindsight with the same backbone (89.0%), while using 3,866 context tokens per query and 1.04 s latency. Failure cases are analyzed in Appendix A.3.

## 4.2.2 Performance on LoCoMo

As most baseline frameworks do not evaluate adversarial questions, Table 2 reports the four standard LoCoMo categories for direct comparison, with adversarial results detailed in the text.

Scalaris yields the highest overall accuracy of 90.77% among the compared systems and dominates the S-Hop (89.36%) and M-Hop (90.34%) categories. When the adversarial subset is included in the evaluation, Scalaris’s overall accuracy rises to 92.55%, driven by a 98.65% accuracy on adversarial questions. On this benchmark, Scalaris uses an average of 4,145 context tokens per query with a retrieval latency of 0.98 s.

<table><tr><td>Category</td><td>Zep</td><td>Mem0- Graph</td><td>Mem0</td><td>Lang- Mem</td><td>Hind- sight Gemini-3</td><td>Scalaris Qwen3-32B</td></tr><tr><td>S-Hop</td><td>74.11</td><td>65.71</td><td>67.13</td><td>62.23</td><td>86.17</td><td>89.36</td></tr><tr><td>M-Hop</td><td>66.04</td><td>47.19</td><td>51.15</td><td>47.92</td><td>70.83</td><td>90.34</td></tr><tr><td>O-Domain</td><td>67.71</td><td>75.71</td><td>72.93</td><td>71.12</td><td>95.12</td><td>93.10</td></tr><tr><td>Temporal</td><td>79.79</td><td>58.13</td><td>55.51</td><td>23.43</td><td>83.80</td><td>76.04</td></tr><tr><td>Overall</td><td>75.14</td><td>68.44</td><td>66.88</td><td>58.10</td><td>89.61</td><td>90.77</td></tr></table>

Table 2: Performance comparison on LoCoMo (% Accuracy). S-Hop: Single-Hop; M-Hop: Multi-Hop; O-Domain: Open Domain.

## 4.2.3 Accuracy–Efficiency Trade-off

Accuracy alone does not fully characterize the practical effectiveness of a memory system, as retrieved context length directly dictates enterprise inference costs. We evaluate this trade-off by comparing Scalaris with Hindsight (Table 3). Because Hindsight publicly reports detailed retrieval metrics (Vectorize, 2026), the figures in Table 3 utilize these publicly reported values for a strictly fair comparison (and thus differ slightly from baseline figures in Tables 1 and 2, which were taken from Latimer et al. (2025)).

<table><tr><td>Method</td><td>LME Acc. (%)</td><td>LME Tokens</td><td>LoCoMo Acc. LoCoMo (%)</td><td>Tokens</td></tr><tr><td>Hindsight</td><td>94.6</td><td>43,625</td><td>92.0</td><td>36,235</td></tr><tr><td>Scalaris</td><td>92.2</td><td>4,116</td><td>90.77</td><td>4,145</td></tr></table>

Table 3: Tokens vs. accuracy (LME = LongMemEval).

Scalaris achieves comparable accuracy to Hindsight while utilizing approximately 10× fewer retrieved context tokens on both benchmarks. Neither baseline distinguishes user interactions from assistant responses, and each sits at one end of the trade-off: Zep organizes all conversational content in a single knowledge graph, retrieving exceptionally lightweight contexts (averaging 1.6K tokens) at the cost of severe accuracy degradation (71.2% on LongMemEval) (Rasmussen et al., 2025), whereas Hindsight stores all content in a single PostgreSQLbacked memory bank (Latimer et al., 2025), securing high accuracy but demanding massive context windows. By representing each source according to its role, Scalaris attains comparable accuracy at a fraction of the context, validating our design choice to structurally differentiate user interactions from assistant responses.

## 4.3 Ablation Study

We conduct an ablation study to isolate the impact of our dual-representation memory strategy. All experiments use Qwen3-32B as the underlying model, with retrieval hyperparameters held constant across configurations. We compare four configurations by alternating the memory structures used for user interactions and assistant responses. The Graph configuration utilizes the ingestion and retrieval pipeline designed for Relevant user interactions, while the VectorDB configuration follows the pipeline for Relevant Assistant Messages. Table 4 reports the results across both benchmarks.

<table><tr><td colspan="2">Memory organization</td><td colspan="2">LongMemEval-S</td><td colspan="2">LoCoMo</td></tr><tr><td>User interactions</td><td>Assistant responses</td><td>Acc.</td><td>Avg. tokens</td><td>Acc.</td><td>token ss</td></tr><tr><td>Graph</td><td>Graph</td><td>76.2</td><td>5,532</td><td>73.5</td><td>5,729</td></tr><tr><td>VectorDB</td><td>Graph</td><td>79.3</td><td>6,267</td><td>77.2</td><td>6,183</td></tr><tr><td>VectorDB</td><td>VectorDB</td><td>86.3</td><td>8,729</td><td>88.5</td><td>8,325</td></tr><tr><td>Graph</td><td>VectorDB</td><td>92.2</td><td>4,116</td><td>90.77</td><td>4,145</td></tr></table>

Table 4: Ablation study of memory organization on LongMemEval-S and LoCoMo (% Accuracy). The last row is the proposed Scalaris organization.

The results demonstrate that memory organization plays a critical role in retrieval effectiveness and efficiency. Utilizing a graph-based structure for assistant responses yields poor accuracy despite retrieving more context, suggesting that relational graphs are not well suited for assistant responses. Conversely, using a VectorDB for both information streams provides substantially better accuracy, indicating that flat storage serves as a strong baseline. However, this homogeneous approach more than doubles the required context window while still underperforming the proposed architecture.

Ultimately, by organizing user interactions via a temporal graph and assistant responses as atomic units in a VectorDB, Scalaris achieves state-of-theart performance while requiring substantially fewer context tokens. These findings validate our core hypothesis: user and assistant messages exhibit distinct characteristics and require tailored memory representations. Additional retrieval-component ablations and hyperparameter sensitivity analyses are reported in Appendices A.1 and A.2.

## 4.4 Internal Evaluation

To validate Scalaris in a live production environment, we integrated the architecture into our proprietary, closed-source conversational AI assistants and evaluated its performance on a sample of 1,200 real user interactions. For each query, we generated two responses: one with memory (WM) utilizing all Scalaris components, and one without memory (WOM) relying solely on the Last-N Exchanges.

Using a blind evaluation protocol, GPT-4o scored each response on a 0–5 scale considering five dimensions: answer quality, personalization, clarity, actionability, and context consistency. Memory-enabled responses were preferred in 91.9% of cases, achieving a mean score of 4.33 compared to 3.18 for the WOM baseline (+36.2%). For the evaluation prompt and extended analysis, refer to the Appendices A.4 and A.5.

Notably, the performance gap widened significantly as dialogue length increased. For queries in the latter half of extended sessions (defined as sessions exceeding 20 turns, representing 433 queries in our dataset), WM responses averaged 4.62, while WOM responses dropped sharply to 2.40. This confirms Scalaris’s ability to maintain conversational coherence in long-horizon interactions.

## 5 Conclusion

We present Scalaris, a unified long-term and shortterm memory architecture designed to address the diverse memory requirements of conversational AI systems. Scalaris enables agents to personalize responses based on users’ preferences and interests, recall relevant information with temporal awareness, and maintain coherence across long and complex conversations. By appropriately organizing user interactions and assistant responses according to their characteristics, Scalaris achieves stateof-the-art performance on both LongMemEval-S and LoCoMo. Notably, these results are achieved with a relatively small Qwen3-32B model while requiring approximately 10× fewer retrieved context tokens than Hindsight with comparable accuracy. Internal evaluation further demonstrates significant improvements in response quality when Scalaris is integrated with conversational AI assistants, highlighting its potential as a practical and scalable framework for enterprise conversational AI systems.

## 6 Limitations

The current Scalaris architecture is restricted to declarative and episodic memory—such as user preferences, explicit facts, entities, and past interactions—and lacks dedicated mechanisms for modeling procedural knowledge like learned skills, workflows, or reusable action sequences. Furthermore, Scalaris is designed exclusively for text-based interfaces and has not been evaluated in voice-driven environments, which require strict real-time latency, continuous streaming, and low-overhead retrieval. Future Work: To address these constraints, future research will extend the framework to incorporate procedural memory capabilities while optimizing retrieval pipelines and streaming efficiency to support ultra-low-latency voice interactions.

## 7 Ethical Considerations

While Scalaris enhances personalization and continuity in conversational AI, persistent memory architectures introduce potential privacy and security risks. Long-term memory stores sensitive user information across sessions, making data protection critical. To mitigate these risks, production deployments must incorporate robust access controls, data anonymization, and user-driven mechanisms for inspecting, editing, or deleting stored memory entries (aligning with right-to-be-forgotten regulations). Furthermore, because long-term memory retains user preferences and traits, systems must safeguard against amplifying user biases or unintentionally storing personally identifiable information (PII).

Artifacts and Licenses. Scalaris builds on Graphiti (Rasmussen et al., 2025; Zep AI, 2024), released under the Apache-2.0 license. We evaluate on LongMemEval (Wu et al., 2025a) (MIT license) and LoCoMo (Maharana et al., 2024) (CC BY-NC 4.0), both used solely for non-commercial research evaluation in accordance with their licenses. We verified that these licenses permit our use and the planned release of our code, which will retain the original license and copyright notices; benchmark data will not be redistributed. No internal user data will be released.

## 8 Use of AI Assistants

AI-based writing tools were used exclusively for grammar correction, proofreading, and language polishing of author-drafted text. All core concepts, methodology, experiments, analyses, and final phrasing remain the original work of the authors, who take full responsibility for the content of this manuscript.

## References

Pratyay Banerjee, Masud Moshtaghi, Shivashankar Subramanian, Amita Misra, and Ankit Chadha. 2026. APEX-MEM: Agentic semi-structured memory with temporal reasoning for long-term conversational AI. In Proceedings of the 64th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 16470–16489, San Diego, California, United States. Association for Computational Linguistics.

Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. 2025. Mem0: Building production-ready AI agents with scalable long-term memory. In ECAI 2025: 28th European Conference on Artificial Intelligence, volume 413 of Frontiers in Artificial Intelligence and Applications, Bologna, Italy. IOS Press.

Cohere. 2023. Introducing embed v3. https:// cohere.com/blog/introducing-embed-v3. Embedding model (embed-english-v3.0).

Gordon V. Cormack, Charles L. A. Clarke, and Stefan Büttcher. 2009. Reciprocal rank fusion outperforms Condorcet and individual rank learning methods. In Proceedings of the 32nd International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 758–759. ACM.

Zijie Dai, Shiyuan Deng, Sheng Guan, Yizhou Tian, Xin Yao, Xiao Yan, and James Cheng. 2026. RecMem: Recurrence-based memory consolidation for efficient and effective long-running LLM agents. In Findings ofthe Associationfor Computational Linguistics: ACL 2026, pages 32353–32376, San Diego, California, United States. Association for Computational Linguistics.

Cai Ke, Yiming Du, Bin Liang, Yifan Xiang, Lin Gui, Zhongyang Li, Baojun Wang, Yue Yu, Hui Wang, Kam-Fai Wong, and Ruifeng Xu. 2025. Flexibly utilize memory for long-term conversation via a fragment-then-compose framework. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 21119–21136, Suzhou, China. Association for Computational Linguistics.

LangChain. 2025. LangMem: Long-term memory for AI agents. https://github.com/langchain-ai/ langmem. Open-source software.

Chris Latimer, Nicolò Boschi, Andrew Neeser, Chris Bartholomew, Gaurav Srivastava, Xuan Wang, and Naren Ramakrishnan. 2025. Hindsight is 20/20: Building agent memory that retains, recalls, and reflects. Preprint, arXiv:2512.12818.

Christopher Latimer, Nicolò Boschi, Andrew Neeser, Chris Bartholomew, Gaurav Srivastava, Xuan Wang, and Naren Ramakrishnan. 2026. Hindsight: Structured agent memory that retains, recalls, and reflects. In Proceedings of the 64th Annual Meeting of the Associationfor Computational Linguistics (Volume 3: System Demonstrations), pages 275–285, San Diego, California, United States. Association for Computational Linguistics.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-augmented generation for knowledgeintensive NLP tasks. In Advances in Neural Information Processing Systems (NeurIPS), volume 33, pages 9459–9474.

Kai Li, Xuanqing Yu, Ziyi Ni, Yi Zeng, Yao Xu, Zheqing Zhang, Xin Li, Jitao Sang, Xiaogang Duan, Xuelei Wang, Chengbao Liu, and Jie Tan. 2026. TiMem: Temporal-hierarchical memory consolidation for long-horizon conversational agents. In Findings of the Association for Computational Linguistics: ACL 2026, pages 21700–21720, San Diego, California, United States. Association for Computational Linguistics.

Jiaqi Liu, Yaofeng Su, Peng Xia, Siwei Han, Zeyu Zheng, Cihang Xie, Mingyu Ding, and Huaxiu Yao. 2026. SimpleMem: Efficient lifelong memory for LLM agents. In Proceedings of the 43rd International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research, Seoul, South Korea. PMLR.

Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2024. Lost in the middle: How language models use long contexts. Transactions of the Association for Computational Linguistics, 12:157–173.

Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang. 2024. Evaluating very long-term conversational memory of LLM agents. In Proceedings ofthe 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 13851– 13870, Bangkok, Thailand. Association for Computational Linguistics.

Kai Tzu-iunn Ong, Namyoung Kim, Minju Gwak, Hyungjoo Chae, Taeyoon Kwon, Yohan Jo, Seungwon Hwang, Dongha Lee, and Jinyoung Yeo. 2025. Towards lifelong dialogue agents via timeline-based memory management. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 8631–8661, Albuquerque, New Mexico. Association for Computational Linguistics.

OpenAI. 2024. GPT-4o system card. Preprint, arXiv:2410.21276.

OpenAI. 2025. gpt-oss-120b & gpt-oss-20b model card. Preprint, arXiv:2508.10925.

Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, and Joseph E. Gonzalez. 2024. MemGPT: Towards LLMs as operating systems. Preprint, arXiv:2310.08560.

Qwen Team. 2025. Qwen3 technical report. Preprint, arXiv:2505.09388.

Preston Rasmussen, Pavlo Paliychuk, Travis Beauvais, Jack Ryan, and Daniel Chalef. 2025. Zep: A temporal knowledge graph architecture for agent memory. Preprint, arXiv:2501.13956.

Stephen Robertson and Hugo Zaragoza. 2009. The probabilistic relevance framework: BM25 and beyond. Foundations and Trends in Information Retrieval, 3(4):333–389.

Freda Shi, Xinyun Chen, Kanishka Misra, Nathan Scales, David Dohan, Ed H. Chi, Nathanael Schärli, and Denny Zhou. 2023. Large language models can be easily distracted by irrelevant context. In Proceedings of the 40th International Conference on Machine Learning (ICML), volume 202 of PMLR, pages 31210–31227.

Supermemory. 2025. Supermemory: The memory API for the AI era. https://github.com/ supermemoryai/supermemory. Open-source software.

Zhen Tan, Jun Yan, I-Hung Hsu, Rujun Han, Zifeng Wang, Long Le, Yiwen Song, Yanfei Chen, Hamid Palangi, George Lee, Anand Rajan Iyer, Tianlong Chen, Huan Liu, Chen-Yu Lee, and Tomas Pfister. 2025. In prospect and retrospect: Reflective memory management for long-term personalized dialogue agents. In Proceedings of the 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 8416–8439, Vienna, Austria. Association for Computational Linguistics.

Ao Tian, Yunfeng Lu, Xinxin Fan, Changhao Wang, Lanzhi Zhou, Yeyao Zhang, and Yanfang Liu. 2026. RGMem: Renormalization group-inspired memory evolution for language agents. In Proceedings ofthe 43rd International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research, Seoul, South Korea. PMLR.

Vectorize. 2026. Agent memory benchmark: Hindsight benchmark results. https://benchmarks. hindsight.vectorize.io. Accessed: 2026-07-27.

Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, and Dong Yu. 2025a. LongMemEval: Benchmarking chat assistants on long-term interactive memory. In The Thirteenth International Conference on Learning Representations.

Yaxiong Wu, Sheng Liang, Chen Zhang, Yichao Wang, Yongyue Zhang, Huifeng Guo, Ruiming Tang, and

Yong Liu. 2025b. From human memory to AI memory: A survey on memory mechanisms in the era of LLMs. Preprint, arXiv:2504.15965.

Yilin Xiao, Zhehan Zhu, Yujing Zhang, Jin Chen, Zijin Hong, Luyao Zhuang, Qinggang Zhang, Shengyuan Chen, Xiaocao Ouyang, Lingfei Ren, and Xiao Huang. 2026. Zero-Mem: Zero-token memory operations for LLM agents. Preprint, arXiv:2607.29377.

Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang. 2025. A-Mem: Agentic memory for LLM agents. In Advances in Neural Information Processing Systems 38 (NeurIPS).

Zihao Yi, Jiarui Ouyang, Zhe Xu, Yuwen Liu, Tianhao Liao, Haohao Luo, and Ying Shen. 2026. A survey on recent advances in LLM-based multi-turn dialogue systems. ACM Computing Surveys, 58(6).

Yi Yu, Liuyi Yao, Yuexiang Xie, Qingquan Tan, Jiaqi Feng, Yaliang Li, and Libing Wu. 2026. Agentic memory: Learning unified long-term and shortterm memory management for large language model agents. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 21457–21483, San Diego, California, United States. Association for Computational Linguistics.

Zep AI. 2024. Graphiti: Build real-time knowledge graphs for AI agents. https://github.com/ getzep/graphiti. Open-source software.

Zeyu Zhang, Quanyu Dai, Xiaohe Bo, Chen Ma, Rui Li, Xu Chen, Jieming Zhu, Zhenhua Dong, and Ji-Rong Wen. 2025a. A survey on the memory mechanism of large language model-based agents. ACM Transactions on Information Systems, 43(6):1–47.

Zhehao Zhang, Ryan A. Rossi, Branislav Kveton, Yijia Shao, Diyi Yang, Hamed Zamani, Franck Dernoncourt, Joe Barrow, Tong Yu, Sungchul Kim, Ruiyi Zhang, Jiuxiang Gu, Tyler Derr, Hongjie Chen, Junda Wu, Xiang Chen, Zichao Wang, Subrata Mitra, Nedim Lipka, and 2 others. 2025b. Personalization of large language models: A survey. Transactions on Machine Learning Research.

## A Extended Analysis

To rigorously evaluate the individual contributions of Scalaris’s modules and assess its robustness to hyperparameter variations, we present an extended analysis encompassing fine-grained ablations, sensitivity experiments, and a qualitative error breakdown.

## A.1 Retrieval Component Analysis

To better understand the contribution of individual retrieval mechanisms, we evaluate the retrieval pipelines of Relevant Assistant Messages and Relevant User Interactions independently on the corresponding LongMemEval categories. Specifically, the assistant-response retrieval pipeline is evaluated only on the single-session-assistant category, while the user-interaction retrieval pipeline is evaluated only on the single-session-user category. These subsets isolate questions whose answers primarily depend on assistant responses or user interactions, respectively, allowing each retrieval component to be analyzed without interference from the other memory stream. Also, these questions require in formation distributed across multiple human-AI exchanges within one session, where omitting even a single relevant memory unit can lead to an incomplete or incorrect response. During each experiment, only the retrieval mechanism under evaluation is enabled while all other retrieval components within the same pipeline are disabled. Since graph traversal (BFS) requires an initial set of retrieved entities, it is evaluated only in combination with semantic retrieval or BM25 rather than as an independent retrieval mechanism. For reference, the complete retrieval pipeline achieves 83.93% accuracy on single-session-assistant and 97.14% on single-session-user.

The single-session-assistant (SSA) gap relative to some baselines in Table 1 reflects a deliberate design trade-off. The SSA subset contains 56 questions, of which Scalaris answers 47 correctly (83.93%). As noted above, these questions require combining details from several assistant responses within a session, so an answer is correct only if every relevant response is retrieved; missing even one yields an incomplete answer. This setting naturally favors systems that retrieve large portions of the conversation: Hindsight, for instance, retrieves roughly 10× more context than Scalaris (Table 3). Scalaris instead retrieves compact atomic units under a fixed retrieval budget, and among low-context

systems it remains stronger than Zep (83.93% vs. 80.4%).
<table><tr><td>Retrieval Method</td><td>k=5</td><td>k=10</td><td>k=15</td><td>k=20</td><td>k=25</td></tr><tr><td>Semantic Search</td><td>52.1</td><td>62.3</td><td>69.2</td><td>69.7</td><td>69.9</td></tr><tr><td>Lexical Search</td><td>44.1</td><td>49.2</td><td>54.7</td><td>55.2</td><td>55.3</td></tr><tr><td>Symbolic Search</td><td>36.7</td><td>37.7</td><td>37.8</td><td>37.8</td><td>37.8</td></tr></table>

Table 5: Retrieval analysis for Relevant Assistant Messages (Single-Session-Assistant). The optimal retrieval budget for each method (the elbow point beyond which gains become negligible) is shown in bold.

As shown in Table 5, semantic retrieval substantially outperforms lexical retrieval by capturing paraphrases and semantically related assistant responses that do not necessarily share surfacelevel vocabulary with the query. Lexical retrieval nevertheless remains valuable for recalling exact phrases, named entities, and numerical information, but its reliance on token overlap limits overall recall. Symbolic retrieval alone performs considerably worse because metadata filtering narrows the search space but cannot identify semantically related assistant responses, making it insufficient as a standalone retrieval strategy. Increasing the retrieval budget consistently improves performance for semantic and lexical retrieval until the optimal value, after which the gains become negligible despite retrieving additional memory units. Symbolic retrieval saturates even earlier because the number of metadata-matching candidates is inherently limited. These observations suggest that semantic, lexical, and symbolic retrieval capture complementary characteristics of assistant responses, motivating their integration in the final retrieval pipeline while avoiding unnecessary retrieval beyond the optimal budget.

<table><tr><td>Retrieval Method</td><td>k=5</td><td>k=10</td><td>k=15</td><td>k=20</td><td>k=25</td></tr><tr><td>Semantic Search</td><td>63.8</td><td>71.9</td><td>77.2</td><td>79.6</td><td>79.8</td></tr><tr><td>BM25</td><td>59.4</td><td>66.8</td><td>72.5</td><td>72.9</td><td>73.0</td></tr><tr><td>Semantic + BFS</td><td>74.6</td><td>80.9</td><td>84.2</td><td>85.8</td><td>85.9</td></tr><tr><td> $\mathbf { B M } 2 5 + \mathbf { B F S }$ </td><td>67.5</td><td>74.8</td><td>79.8</td><td>80.3</td><td>80.4</td></tr></table>

Table 6: Retrieval analysis for Relevant User Interactions (Single-Session-User). The optimal retrieval budget for each method (the elbow point beyond which gains become negligible) is shown in bold.

As reported in Table 6, semantic retrieval consistently outperforms BM25 because user queries often refer to previously discussed facts using different wording, allowing dense retrieval to identify semantically related interactions that lexical matching fails to capture. However, BM25 remains valuable for queries involving exact entity names, identifiers, or uncommon user-specific terminology, where lexical overlap provides a stronger retrieval signal than semantic similarity. Incorporating graph traversal improves both retrieval strategies by following the relationships of the initially retrieved entities and bringing together related facts that are distributed across multiple user interactions. The improvement is considerably larger when combined with semantic retrieval, as semantically retrieved entities provide more relevant starting points for graph traversal, whereas BM25 already retrieves highly specific matches that leave less additional information to be discovered through the graph. Similar to the assistantresponse pipeline, increasing the retrieval budget consistently improves performance until the optimal value, after which the gains become negligible despite retrieving additional facts. Taken together, these observations suggest that the three retrieval mechanisms capture different aspects of user memory, motivating their integration in the final retrieval pipeline.

## A.2 Hyperparameter Sensitivity

We further evaluate the sensitivity of Scalaris to its primary hyperparameters. During this analysis, the retrieval budget for each retrieval mechanism is fixed to its corresponding optimal value identified in Section A.1, allowing the effect of each hyperparameter to be analyzed independently. Table 7 summarizes the evaluated ranges and the corresponding recommended values.

<table><tr><td>Hyper- parameter</td><td>Range Evaluated</td><td>Optimal Range</td><td>Eval. Dataset</td></tr><tr><td>Semantic similarity threshold</td><td>0.3-0.7 (step 0.1)</td><td>0.4–0.6</td><td>LME</td></tr><tr><td>BM25 threshold</td><td>0.6-1.1 (step 0.1)</td><td>0.9-1.0</td><td>LME</td></tr><tr><td>Graph traversal depth</td><td>1–5 hops</td><td>3 hops</td><td>LME</td></tr><tr><td>Last-N Exchanges</td><td>1,3,5,7, 10</td><td>5</td><td>Internal Eval.</td></tr></table>

Table 7: Evaluated ranges and recommended values for Scalaris’s primary hyperparameters. LME: Long-MemEval.

Overall, Scalaris remains stable across a broad range of parameter values, indicating that its performance is not sensitive to minor variations in hyperparameter selection. We vary the semantic similarity threshold from 0.3 to 0.7 in increments of 0.1, observing consistently strong performance within the 0.4–0.6 range. Lower thresholds admit semantically weak matches, whereas higher thresholds filter relevant memories that differ in wording from the query. Similarly, we evaluate BM25 thresholds from 0.6 to 1.1, with performance remaining stable within the 0.9–1.0 range. Lower thresholds admit low-scoring lexical matches, whereas higher thresholds begin filtering lexical matches required for retrieval. Increasing the graph traversal depth improves retrieval up to three hops, after which additional hops primarily retrieve less relevant neighboring entities while increasing retrieval latency with negligible accuracy gains. Finally, evaluation on our internal benchmark shows that retaining the most recent five user-assistant exchanges provides sufficient conversational context for immediate coreference resolution, whereas larger context windows increase prompt length without improving response quality.

## A.3 Qualitative Error Analysis

To understand the limitations of Scalaris, we conducted a qualitative analysis of the remaining ∼8% of failure cases in LongMemEval. The majority of errors stem from three distinct patterns. First, ambiguous temporal expressions (e.g., “a couple of quarters ago”) occasionally fail to resolve accurately when the surrounding context lacks explicit anchor dates. Second, distributed evidence retrieval remains challenging when the information required to answer a query is scattered across multiple distant interactions, but the initial retrieval stage fails to surface all relevant memory units needed for complete reasoning. Finally, implicit fact negation poses a challenge; when a user implicitly changes a previously stated preference without directly contradicting it, the temporal edge invalidation fails to trigger, causing the system to retrieve outdated context. Addressing these implicit reasoning bottlenecks remains a critical area for future work.

## A.4 Internal Evaluation: Detailed Analysis

To complement the quantitative results presented in Section 4.4, we performed a qualitative analysis of 1,200 real user queries collected from our proprietary production finance assistants. Unlike benchmark datasets, these conversations represent genuine user interactions and therefore naturally contain incomplete follow-up questions, pronouns, ellipsis, and implicit references. Rather than restating the complete context in every turn, users progressively rely on the assistant’s conversational memory, making later queries increasingly ambiguous when viewed independently. A representative conversation from our evaluation is shown below:

1. Compare Tata Steel and Hindalco.

2. Which one looks better for long-term investment?

3. Compare it with Larsen & Toubro.

4. Which one is the weakest stock now?

5. Replace the weakest stock.

6. Which one looks safer?

7. What are the downside risks?

8. What are the upside drivers?

9. What is the expected timeline?

10. Summarize the key insights.

Although the first few questions explicitly identify the companies under discussion, the subsequent queries depend almost entirely on the preceding conversation. References such as “it”, “the weakest stock”, “the downside risks”, “the upside drivers”, and “the expected timeline” cannot be interpreted correctly without resolving the conversational context established in earlier turns. Similar interaction patterns were consistently observed throughout the internal evaluation, reflecting how users naturally communicate with conversational assistants rather than how benchmark datasets are typically constructed.

To understand how existing memory frameworks handle such conversations, we additionally evaluated the same production queries using Zep and Hindsight. A common limitation observed across both systems was their inability to consistently resolve conversational coreferences and implicit follow-up queries before retrieval. Retrieval was frequently performed using the surface form of the latest user query, causing relevant memories associated with previously discussed entities to be missed. In contrast, Scalaris first reformulates each conversationally dependent query into a self-contained retrieval query by resolving pronouns, omitted entities, and implicit continuations before initiating memory retrieval.

Despite this shared limitation, the two frameworks exhibited different behaviors. Zep was able to provide limited personalization through its maintained user summary, but often failed to recall previously generated assistant responses. Consequently, the downstream assistant repeatedly invoked external tools to retrieve information that had already been fetched earlier in the conversation, increasing unnecessary computation and response latency. Hindsight recalled prior assistant information more reliably by retrieving substantially larger portions of the conversation, thereby reducing repeated tool invocations. However, this came at the cost of retrieving significantly more context tokens, frequently overwhelming the answering model with redundant information. Furthermore, because Hindsight does not explicitly maintain an evolving user profile, it was generally less effective at personalizing responses according to long-term user preferences.

These observations suggest that an effective conversational memory system must adapt to the way users naturally communicate, rather than expecting them to formulate fully self-contained queries at every turn. Real-world conversations are inherently incremental, with users relying on shared conversational context through pronouns, omitted entities, and implicit references. Furthermore, accurate recall alone is insufficient for a practical memory system. It must retrieve only the information necessary to answer the current query, avoiding redundant or irrelevant context. This is particularly important in enterprise deployments, where excessive retrieval increases latency, inference costs, and can overwhelm the downstream language model, increasing the likelihood of hallucinations.

## A.5 Internal Evaluation Prompt

The exact scoring prompt supplied to the GPT-4o judge for the blind pairwise evaluation described in Section 4.4 is shown in Figure 4. Placeholders in braces are populated per query.

![](images/c358b047f4c607dafd497fee73a10ebdc47f91818182ff15443a0461313576c2.jpg)  
Figure 4: Evaluation prompt used by the GPT-4o judge to compare two AI assistant responses. The prompt provides the original user question, recent conversational history, and available memory context, followed by both candidate responses. The judge assesses answer quality, personalization, clarity, actionability, and consistency with the conversation before assigning a score from 0 to 5 to each response and producing a comparative verdict in a strictly specified JSON format.