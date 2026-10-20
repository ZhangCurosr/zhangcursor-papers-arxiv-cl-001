# RESIN: Multi-Chain Reasoning with Entity-Structured Indexing

Xuanyu Meng Jash Rajesh Parekh Jiawei Han University of Illinois Urbana-Champaign {xuanyum2,jashrp2,hanj}@illinois.edu

## Abstract

Multi-chain reasoning is a challenging task not only because the multi-chain evidence is dispersed, but also because different parts of reasoning may require distinct, interacting chains of support. Flat retrieval can flood a model with loosely related text, whereas early selection of a single reasoning path can discard evidence that becomes useful only after other facts are considered. We introduce RESIN (Reason ing with Entity Structured Indexing), which uses structure to locate evidence while keeping the original source as the basis for reasoning. RESIN decomposes a question into focused one-hop needs, retrieves a small set of source-grounded units for each hop, and places only this relevant evidence in a compact reasoning buffer. The buffer preserves alternative supports until final composition, making intermediate decisions explicit without forcing a brittle path. On the full HolisQA benchmark, RESIN achieves 82.92 context recall and 76.70 answer relevancy over all questions, with higher reported CR and AR across five domains and higher F1 in four domains than the published baselines. Ablations show that both typed evidence diversity and hop-wise decomposition are essential, supporting a clear conclusion: Precise evidence organization and de layed commitment make multi-chain reasoning effective and auditable. Our code is available at <sup>§</sup> github.com/RamonMeng/RESIN.

## 1 Introduction

Multi-chain reasoning requires a model to assemble several kinds of evidence chains whose roles are not interchangeable. One kind may identify an entity; another may catch a property or a state transition; and the third may conduct reasoning by following certain rules. The challenge is therefore not simply to find relevant text, but to assign evidence to the right inferential role while preserving entity identity, scope, condition, and direction.

Such distinct mechanisms make multi-chain reasoning a rich but challenging task. A passage that share many words with the question may not eventually contribute to the reasoning path, whereas a less obvious passage may later supply the bridge for effective multi-chain reasoning. A system that ranks evidence only against the complete question cannot handle such complexity. Thus a system that commits to the first locally plausible chain risks the opposite error: it may explain one part of the answer while losing a parallel chain needed for another.

Existing evidence-access strategies tend to encounter difficulties at handling such complexity. A single broad search compresses all information needs into one query, which can blur their roles, introduce distracting context, and impose the same granularity on every step (Liu et al., 2024; Chen et al., 2024; Jiang et al., 2024). Repeated search exposes intermediate information needs (Trivedi et al., 2023; Jiang et al., 2023; Asai et al., 2024), but untyped passages still leave entity and state bindings implicit. Graph-based systems make relations explicit (Edge et al., 2024; Jiménez Gutiérrez et al., 2025; Li et al., 2025; Guo et al., 2025), yet extraction and edge ranking are uncertain: pruning to one node, edge, or path can remove correct support before all chains are reconciled. This is especially risky in scientific text, where a relation may change with the condition, population, or state (Parekh et al., 2026).

The underlying mismatch is between the object that can be indexed in advance and the object required to answer a particular question. Evidence records should be reusable and query independent, whereas a valid chain is query specific: the same statement may be central for one question, a bridge for another, and irrelevant for a third. Precomputing every possible chain is neither tractable nor robust, because the number of combinations grows quickly and every extracted edge becomes an early commitment. Deferring all organization until generation avoids that early commitment but returns the model to a large, weakly structured context. What is needed is an intermediate representation that is fine grained enough to locate precise evidence, yet flexible enough to assemble different chains on demand.

![](images/1eedac1e9a1a24e622e12a7b51e0a2046730236641e56a7c96d07c1592561918.jpg)  
Figure 1: RESIN at a glance. Offline, entity-structured records serve as addresses into source-grounded evidence. Online, a dependency-aware plan creates focused one-hop needs; each retrieves a small candidate set, and the resulting evidence buffer remains available until final composition.

We propose a different division of labor. Structure should tell the system where to look; whereas source-grounded evidence should tell it what to believe. RESIN organizes evidence into typed, provenance-linked units that act as precise addresses. It then builds a small reasoning buffer for the current question rather than exposing the model to the full text or index. The buffer assigns a candidate set to each planned hop and retains alternatives until final composition. Thus, structure improves access without becoming a brittle, prematurely fixed reasoning path.

Figure 1 makes this separation concrete. The upper half is query independent: source passages are organized around their central entities, converted into compact reasoning units, and linked back to exact evidence. The lower half begins with the question: a planner expresses it as focused one-hop needs, retrieval fills a bounded candidate set for each need, and local answers expose what each set supports. The final composer sees the original question and the complete buffer. Dependency metadata records how the subquestions fit together, while the evidence itself remains available for global reconciliation.

This architecture offers three practical advantages. First, it bounds the model’s working context by the number of planned needs and the per-hop candidate budget, rather than by corpus size. Second, it makes failures inspectable: one can distinguish a missing indexed unit, a retrieval miss, an incorrect local interpretation, and a composition error. Third, it keeps competing evidence recoverable until the system has seen all parts of the question. These properties motivate our central hypothesis: multi-chain reasoning improves when evidence is localized early but selected late. The experiments test both sides of this claim by varying the unit representation, the decomposition strategy, and the number of candidates retained per hop.

Our contributions are:

• an entity-structured index that localizes composable, typed evidence while preserving exact source provenance;

• a hop-wise reasoning buffer that retrieves focused candidate sets and delays hard evidence selection until the complete question can be considered; and

• evaluation on HolisQA and two generaldomain QA benchmarks, with ablations showing when decomposition, evidence diversity, and retrieval budget matter.

## 2 Related Work

From passage retrieval to focused information needs. RAG combines parametric generation with external retrieval (Lewis et al., 2020), and dense passage retrieval provides a strong semantic interface for open-domain QA (Karpukhin et al., 2020). Work on Dense X Retrieval, LongRAG, and RAPTOR shows that retrieval granularity and hierarchy materially affect what evidence reaches the model (Chen et al., 2024; Jiang et al., 2024; Sarthi et al., 2024). These methods improve access to long corpora, but a retrieved passage collection still asks the generator to discover which part serves which reasoning step. RESIN instead starts from explicit information needs and gives each one a small, source-grounded evidence set.

Iterative retrieval and multi-step reasoning. Chain-of-thought and ReAct make intermediate decisions visible (Wei et al., 2022; Yao et al., 2023). IRCoT alternates retrieval with reasoning, FLARE retrieves during generation, and Self-RAG learns when to retrieve and critique (Trivedi et al., 2023; Jiang et al., 2023; Asai et al., 2024). Their central insight—reasoning should guide evidence access— also motivates RESIN. Our distinction is representational: planned hops search a typed index, and all retrieved alternatives remain in a bounded buffer for final reconciliation instead of being reduced immediately to a single intermediate statement.

Structured, entity-centered, and causal retrieval. GraphRAG, HippoRAG2, StructRAG, and LightRAG organize entities and relations to support global, associative, or structure-aware retrieval (Edge et al., 2024; Jiménez Gutiérrez et al., 2025; Li et al., 2025; Guo et al., 2025). SARG searches explicit relational chains (Parekh et al., 2025), while CausalRAG2 builds hierarchical causal graphs and applies top-down pruning for HolisQA (Wang et al., 2026). Such structure is valuable, but a selected path can conflate indexing with reasoning: an extraction or pruning error may remove evidence before the answer is composed. Entity structure has also proved useful in other settings: EnSI-RAG uses it to connect properties and relations to source passages (Meng et al., 2026b),

EnSIMem uses it to organize verifiable long-term agent memories (Meng et al., 2026a), and openschema discovery extracts entity–attribute–value structure without a fixed ontology (Xu et al., 2026). These applications illustrate the general utility of entities as stable access keys, but address different tasks. RESIN targets multi-chain reasoning by using typed structure as an address into evidence and retaining a small buffer of source-linked alternatives, rather than treating the structure as an irrevocable answer path.

## 3 RESIN

## 3.1 Design Goal and Running Example

Given corpus D and question q, RESIN uses entity structure to find the relevant original text, then reasons directly over that text. Offline, it builds searchable, source-linked records. Online, it plans focused searches, retrieves records for each need, recovers their source passages, and composes an answer from a small selected evidence buffer (Figure 1).

Consider an illustrative commute: Should I take the bus or metro to reach work within 40 minutes and spend at most \$10? Suppose route descriptions list the bus at \$4, with a 10-minute walk and a 35- minute ride; the metro costs \$8, with a 5-minute walk and a 25-minute ride. Assuming no waiting, the bus takes 45 minutes and the metro 30. Only the metro satisfies both constraints. Answering requires separate evidence chains for the two routes, keeping fares and travel times attached to the correct option.

## 3.2 Design Rationale

Give each answer component a search. The planner asks separately for the bus’s fare and journey times and for the metro’s corresponding values. Each need gets its own candidate set, so many matches about one route cannot consume the entire retrieval budget. The hops identify what must be found; they do not decide which route to recommend.

Keep facts attached to their meaning. A record binds a value to its entity, property, and qualifiers: the bus ride takes 35 minutes, not simply 35. This distinguishes ride time from total journey time and bus facts from metro facts. Its source pointer recovers the wording needed to check those distinctions.

<table><tr><td>Field group</td><td>Stored fields</td><td>Purpose and example</td></tr><tr><td>Identity</td><td>ID</td><td>kind; stable unit Distinguishes a fact from a rule or directed transition.</td></tr><tr><td></td><td>Core claim subject; entity Binds “bus fare is object</td><td>s  $\$ 4$  to the type; predicate; correct route and property.</td></tr><tr><td>Logical structure</td><td>conclusion;</td><td>conditions and Preserves conditions and di- rection, e.g., congestion in-</td></tr><tr><td>Qualifiers</td><td>source and target creasing ride time. states</td><td>modality; polar- Keeps a weekday travel-</td></tr><tr><td>Grounding</td><td>dence passage</td><td>ity; scope; confi- time estimate distinct from an unrestricted claim. evidence quote; Links every normalized and record back to the exact</td></tr></table>

Table 1: Functional schema of a reasoning unit. Optional fields are populated only when supported by the source.

Make the decision from original evidence. The final composer receives the original question and selected source passages, not generated hop answers. It calculates each journey’s total time and checks both constraints together. An early conclusion such as “the bus is cheaper” therefore need not determine the recommendation before arrival time is considered.

## 3.3 Source-Linked Evidence Organization

Offline processing groups contiguous sentences around an entity and evidence role, preserves their text and sentence IDs, and extracts minimal evidence-backed units. The bus’s ride-time statement becomes a searchable record linked to its route passage; unrelated service announcements need not enter that record. The record is an access handle, while the original passage remains the evidence.

Table 1 groups fields by function: claim identity, composition constraints, qualifiers, and provenance. Optional fields require source support.

Table 2 distinguishes six kinds of support. A route’s fare is an entity fact; a source-stated effect of congestion on ride time is a directed transition. Rules retain their conditions, and generic type properties remain separate from entity membership. Only source-supported fields are extracted: journey totals are computed at answer time, not inserted as facts during indexing.

The fields guide the LLM, not a symbolic proof checker: they distinguish evidence roles and expose direction, binding, and qualification for inspection.

## 3.4 The Index as an Evidence Address

The searchable store is

$$
\mathcal { U } = \{ ( \mathrm { i d } ( u ) , x _ { u } , \pi _ { u } ) : u \in U \} ,\tag{1}
$$

where $x _ { u }$ is normalized retrieval text and $\pi _ { u }$ links to the evidence quote, passage, and source sentences. Auxiliary entity, type, property, state, and relation views provide lookup handles. A sentence encoder caches embeddings of $x _ { u } ;$ the main online backend uses dense unit retrieval, not graph traversal.

In Equation 1, $x _ { u }$ makes “bus ride time” searchable, and $\pi _ { u }$ leads back to the route passage. This separates finding a relevant fact from deciding how to use it; bus and metro records need no preexisting connection in the index.

## 3.5 Planning and Need-Specific Retrieval

A planner maps q to an ordered plan $\textit { Q } =$ $( q _ { 1 } , \dots , q _ { H } )$ . Each hop specifies an evidence need, target and bridge terms, a reasoning type, and optional dependencies. Here, $q _ { 1 }$ asks for the bus’s fare, walk, and ride times; $q _ { 2 }$ asks for the metro’s. All hops are planned upfront and retrieve independently: the second does not use an answer generated for the first. Failed planning falls back to the original question as one hop.

For retrieval text $\tilde { q } _ { h }$ formed from the subquestion, target, and bridge terms, we select

$$
C _ { h } = \mathrm { T o p K } _ { u \in { U } } \cos ( E ( \tilde { q } _ { h } ) , E ( x _ { u } ) ) .\tag{2}
$$

Equation 2 retrieves bus records into $C _ { 1 }$ and metro records into $C _ { 2 }$ , with a separate top-k budget for each need. Retrieval returns evidence candidates, not route recommendations. The main setting permits at most four hops and $k = 1 0$ units per hop.

Multiple hops also increase total capacity; the decomposition ablation is not budget-matched. Upfront planning can miss a bridge entity not expressible from the question.

## 3.6 Buffer Construction and Final Composition

Provenance maps the retrieved units to original passages. Let $P ( q )$ be these passages, deduplicated and packed by alternating across hops at each candidate rank within a 16,000-character budget. The final evidence buffer is

$$
B ( q ) = \mathrm { S e l e c t S o u r c e s } ( q , P ( q ) ) ,\tag{3}
$$

Table 2: Reasoning-unit kinds and canonical representations. Colors mark entity, type, property, value, relation, state, and condition slots.
<table><tr><td>Unit kind</td><td>Representation</td><td>Why it is useful</td></tr><tr><td>ENTITY-FACT</td><td>entity Iproperty I value</td><td>Retrieves an attribute stated for one named entity.</td></tr><tr><td>TYPE-PROPERTY</td><td>type property value</td><td>Stores a generic property that may support type inheritance.</td></tr><tr><td>ENTITY-TYPE</td><td>entity l is-a type</td><td>Supplies explicit membership evidence before a type property is applied.</td></tr><tr><td>RELATION</td><td>entity or type relation entity or concept</td><td>Preserves a typed link that is neither causal nor conditional.</td></tr><tr><td>CAUSAL-EDGE</td><td>source state →target state</td><td>Keeps causal direction and state change explicit.</td></tr><tr><td>RULE</td><td>conditions →conclusion</td><td>Prevents a conclusion from being used when its antecedents are absent.</td></tr></table>

where the selector uses the question, article titles, and source snippets to choose up to two article groups and retain their packed passages. A failed selection falls back to question-term ranking of up to three groups. Selection uses neither gold answers nor generated hop answers.

The composer receives only q and B(q), not the query plan or intermediate answers. From both route passages, it computes 10 + 35 = 45 minutes for the bus and 5 + 25 = 30 for the metro. Both fares are within budget, but only the metro meets the deadline. The hops found the inputs; final composition makes the comparison. Auxiliary hop-solver outputs are still recorded by the implementation but are unused in this generation path. Evidence lost during extraction, retrieval, or source selection remains a limitation.

## 4 Experiments

## 4.1 Research Questions

We ask: (1) Does entity-structured reasoning improve evidence coverage and answer relevance? (2) Does the approach transfer beyond HolisQA? (3) Which parts of the evidence buffer matter? and (4) What costs accompany its more explicit reasoning process?

## 4.2 Datasets

We evaluate on the full HolisQA release (Wang et al., 2026), 500 queries from MS MARCO (Bajaj et al., 2016), and 426 queries from QASC (Khot et al., 2020). Together, the benchmarks cover scientific synthesis, web-search QA, and two-fact science composition. Appendix A.1 gives full benchmark details.

## 4.3 Baselines, Metrics, and Setup

We compare with the published CausalRAG2 suite (Wang et al., 2026). Baseline accuracy and efficiency entries are transcribed from that paper; only RESIN is measured in our environment. Appendix A.2 describes the baselines. On HolisQA, we report answer-overlap F1 alongside RAGAS answer relevancy (AR) and context recall (CR) (Es et al., 2024). F1 measures token overlap with the reference answer; AR measures whether an answer addresses the question, whereas CR asks whether retrieved passages cover claims in the reference answer. Appendix A.4 defines AR and CR. All scores are percentages; overall HolisQA AR and CR are query-weighted means across domains.

Our runs use RAGAS 0.4.3 with strictness 3, source-passage contexts, and Gemini-3.5-Flash-Lite as judge. GPT-5-nano performs passage construction, unit extraction, planning, hop solving, and composition; all-MiniLM-L6-v2 performs dense retrieval. The main setting allows at most four hops and retrieves k = 10 units per hop.

Reproduction scope. Full-scale CausalRAG2 indexing did not finish within a practical window in our environment. A Biology testing run was completed to verify pipeline execution, not full-dataset performance. We therefore retain published baseline rows. The scores obtained in this test closely match those reported in the CausalRAG2 paper, providing support for the comparability of our evaluation results.

## 4.4 Main Results

RESIN exceeds the transcribed baselines in reported CR and AR across all five HolisQA domains and in F1 across four domains (Table 3). Business F1 is 49.61, below CausalRAG2’s 51.51; Table 4 reports general-domain CR and AR. CausalRAG2 leads the baseline suite across all five HolisQA domains, making it the main reference for our architectural comparison. We therefore focus the following analysis on what RESIN adds beyond hierarchical causal retrieval. Evaluator differences and different general-domain query sets remain relevant when interpreting the reported score gaps.

Table 3: HolisQA results across five domains and all questions. F1 measures answer overlap; CR and AR denote context recall and answer relevancy. All scores are percentages. Baseline scores are transcribed from the published CausalRAG2 table; only the RESIN row is measured in our experiment. Highest and second-highest values are bold and underlined; cross-system evaluation caveats are discussed in the text.
<table><tr><td></td><td colspan="3">Medicine</td><td colspan="3">Computer Science</td><td colspan="3">Business</td><td colspan="3">Biology</td><td colspan="3">Psychology</td></tr><tr><td>Method</td><td>F1</td><td>CR</td><td>AR</td><td>F1</td><td>CR</td><td>AR</td><td>F1</td><td>CR</td><td>AR</td><td>F1</td><td>CR</td><td>AR</td><td>F1</td><td>CR AR</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>Naive baselines</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>NaiveGeneration</td><td>12.63</td><td>0.00</td><td>44.70</td><td>18.93</td><td>0.00 48.79</td><td>18.58</td><td>0.00</td><td>46.14</td><td>11.71</td><td>0.00</td><td>45.76</td><td>22.91</td><td></td><td>0.00 50.00</td></tr><tr><td>BM25</td><td>17.72</td><td>52.04</td><td>50.64</td><td>24.00 39.12</td><td>52.40</td><td>28.11</td><td>37.06</td><td>55.52</td><td>19.61</td><td>43.02</td><td>52.32</td><td>30.46</td><td>33.44</td><td>56.63</td></tr><tr><td>StandardRAG</td><td>26.87</td><td>61.08</td><td>56.24</td><td>28.87</td><td>49.44 57.10</td><td>47.57</td><td>46.79</td><td>67.42</td><td>28.31</td><td>42.69</td><td>57.58</td><td>37.19</td><td>52.21</td><td>59.85</td></tr><tr><td></td><td colspan="10">Graph-based RAG</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GraphRAG Global</td><td>17.13</td><td>54.56</td><td>48.19</td><td>23.75</td><td>37.65 53.17</td><td></td><td>23.62 25.01</td><td>48.12</td><td>20.67</td><td>40.90</td><td>52.41</td><td>31.09</td><td></td><td>34.2654.62</td></tr><tr><td>GraphRAG Local</td><td>19.03</td><td>56.07</td><td>49.52</td><td>25.10</td><td>39.9053.30</td><td>25.01</td><td>27.36</td><td>49.05</td><td>22.21</td><td>41.88</td><td>52.73</td><td>32.31</td><td>35.22</td><td>55.02</td></tr><tr><td>LightRAG</td><td>12.16</td><td>52.38</td><td>44.15</td><td>22.59</td><td>41.8651.62</td><td></td><td></td><td>29.98 34.22 54.50</td><td>17.70</td><td>41.24</td><td>50.32</td><td>33.63</td><td>45.54</td><td>56.42</td></tr><tr><td></td><td colspan="10">Structural / causal augmented</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>HippoRAG2 LeanRAG</td><td>21.12</td><td></td><td>57.5051.08</td><td>16.94</td><td>21.0547.29</td><td></td><td></td><td>21.10 18.3445.83</td><td>12.60</td><td>16.85</td><td>44.56</td><td>20.10</td><td>34.13</td><td>46.77</td></tr><tr><td>CausalRAG</td><td>34.25</td><td>60.43</td><td>56.60</td><td>30.51</td><td>57.61</td><td>55.45 48.30</td><td>59.29</td><td>60.35</td><td>33.82</td><td>58.43</td><td>56.10</td><td>42.85</td><td>57.46</td><td>58.65</td></tr><tr><td>CausalRAG2</td><td>31.12 36.45</td><td>58.90 69.91</td><td>58.77</td><td>30.98</td><td>54.10</td><td>57.54</td><td>45.20</td><td>44.55 66.10</td><td>33.50</td><td>51.20</td><td>58.90</td><td>42.80</td><td>55.60</td><td>61.90</td></tr><tr><td></td><td></td><td></td><td>60.65</td><td>31.60</td><td>60.94</td><td>58.34</td><td>51.51</td><td>67.34 68.76</td><td>34.80</td><td>61.97</td><td>59.99</td><td>44.42</td><td>60.87</td><td>63.53</td></tr><tr><td>RESIN</td><td>45.43</td><td>80.9575.43</td><td></td><td>50.96</td><td>86.27</td><td>75.09</td><td>49.61 84.97</td><td>79.05</td><td>47.21</td><td>84.86</td><td>75.86</td><td>46.48</td><td>77.95</td><td>77.93</td></tr></table>

Table 4: General-domain results. The RESIN row uses 500 MS MARCO queries and 426 QASC queries; baseline F1/CR/AR are transcribed from the published CausalRAG2 full-benchmark table. Highest and second-highest values are bold and underlined.
<table><tr><td></td><td colspan="3">MS MARCO</td><td colspan="3">QASC</td></tr><tr><td>Method</td><td>F1</td><td>CR</td><td>AR</td><td>F1</td><td>CR</td><td>AR</td></tr><tr><td colspan="7">Naive baselines</td></tr><tr><td>NaiveGeneration</td><td>5.28</td><td>0.00</td><td>15.06</td><td>2.69</td><td>0.00</td><td>13.74</td></tr><tr><td>BM25</td><td>6.97</td><td>45.78</td><td>20.33</td><td>2.49</td><td>6.12</td><td>13.17</td></tr><tr><td>StandardRAG</td><td>14.93</td><td>48.55</td><td>31.11</td><td>2.01</td><td>5.50</td><td>13.16</td></tr><tr><td colspan="7">Graph-based RAG</td></tr><tr><td>GraphRAG Global</td><td>9.41</td><td>3.65</td><td>13.08</td><td>0.68</td><td>3.38</td><td>3.56</td></tr><tr><td>GraphRAG Local</td><td>30.87</td><td>25.71</td><td>57.76</td><td>8.30</td><td>9.54</td><td>46.59</td></tr><tr><td>LightRAG</td><td>37.70</td><td>54.22</td><td>63.54</td><td>8.20</td><td>20.40</td><td>44.35</td></tr><tr><td colspan="7">Structural / causal augmented</td></tr><tr><td>HippoRAG2</td><td>23.35</td><td>45.45</td><td>55.18</td><td>14.73</td><td>4.38</td><td>49.94</td></tr><tr><td>LeanRAG</td><td>38.02</td><td>54.01</td><td>58.49</td><td>13.19</td><td>22.80</td><td>45.51</td></tr><tr><td>CausalRAG</td><td>27.66</td><td>39.38</td><td>46.03</td><td>7.65</td><td>46.86</td><td>35.56</td></tr><tr><td>CausalRAG2</td><td>38.40</td><td>60.48</td><td>66.02</td><td>13.35</td><td>70.80</td><td>49.40</td></tr><tr><td>RESIN</td><td>38.73</td><td>91.65</td><td>69.68</td><td>37.09</td><td>84.51</td><td>58.77</td></tr></table>

Evidence beyond a causal explanation. Causal-RAG2 uses hierarchical gates and causal refinement to recover coherent explanations. However, multi-chain questions can also require attributes, experimental observations, and comparative properties that do not form another step in the dominant causal chain. An evidence selection centered on that chain may retain a plausible explanation while omitting support for another requested component. RESIN makes these heterogeneous claims independently addressable through complementary unit kinds, so their retrieval does not require membership in one connected explanation.

Coverage of distinct information needs. A single question-level ranking can allocate much of its evidence budget to the most salient component. Several highly relevant items may then repeat the same mechanism instead of covering different parts of the answer. RESIN decomposes the question into focused needs and assigns each its own candidate set. This organization gives less prominent components a retrieval opportunity and reduces competition between different evidence roles. The CR drop when decomposition is removed, together with the losses from single-kind retrieval, supports the value of decomposed access and complementary representations.

Joint composition without premature exclusion. Evidence can also be lost after retrieval: an early selection may discard a claim whose importance becomes apparent only when another component is considered, or final synthesis may fail to use an available distinguishing fact. RESIN maps retrieved candidates back to their original passages and gives the final composer the original question and selected source evidence, without the query plan or generated hop answers. The composer can therefore consider explanatory, supporting, and comparative claims together in their original wording rather than inheriting intermediate answer summaries. This changes the reasoning interface from selecting a plausible explanation to assembling support for the question’s distinct requirements.

These differences explain why RESIN is particularly useful when an answer depends on heterogeneous, unevenly salient, or interacting evidence. Appendix B provides detailed case studies tracing these failure patterns and the corresponding evidence recovered by RESIN.

## 4.5 Ablation Studies

Ablations reuse the same HolisQA subset, preprocessing, generator, evaluator, and encoder. Each changes the named component; AR/CR aggregation excludes failed evaluator calls.

Per-hop candidate budget. Figure 2(a) varies k while holding the query plan and all other settings fixed. CR rises from 69.7 at k = 1 to 86.5 at k = 5 and 89.3 at k = 10. Increasing the budget further yields only small, non-monotonic changes: CR is 89.0 at k = 15 and 90.1 at k = 20. AR remains comparatively stable, reaching 77.0 at both k = 10 and k = 15, before declining to 75.7 at k = 20. Thus, k = 10 offers a practical balance between answer relevance, evidence coverage, and context size.

Table 5: Effect of hop-wise query decomposition on a fixed HolisQA subset. The full method uses k candidates per hop; the single-query variant has one set of k.
<table><tr><td>Variant</td><td>AR</td><td>CR</td></tr><tr><td>Full RESIN</td><td>77.03</td><td>89.30</td></tr><tr><td>w/o hop decomposition</td><td>70.04</td><td>42.33</td></tr></table>

Typed unit diversity. Figure 2(b) compares the complete mixed index with six variants that retain only one unit kind. The mixed index obtains 77.0 AR and 89.3 CR. Among the single-kind variants, ENTITY-FACT is strongest (74.0 AR and 77.9 CR), followed by CAUSAL-EDGE (61.1 AR and 65.9 CR). No individual representation matches the mixed index. The substantially lower scores of type properties, generic relations, and rules indicate that these representations alone provide insufficient evidence coverage. These results support treating the six unit kinds as complementary evidence representations rather than interchangeable schemas.

Hop decomposition. Table 5 replaces the ordered subquestions with a single retrieval over the original query. Removing decomposition reduces AR by 6.99 points and CR by 46.97 points. The larger CR loss supports decomposed evidence access, while the smaller AR loss suggests that an answer can remain topical despite missing support. Because the full method also allocates candidates across multiple hops, this experiment does not isolate query specificity from total evidence budget.

## 4.6 Efficiency

Unit extraction increases offline tokens; planning, hop solving, and composition add online latency. Appendix A.3 gives the cost table, with Causal-RAG2 efficiency left as unverified paper-reported values.

## 5 Discussion and Future Work

Long-context multi-chain reasoning. RESIN is architecturally suited to long contexts because its offline construction processes each passage independently. Moving from a short to a long corpus changes the number of indexed units, but not the amount of text handled by any single construction call. At inference, the complete corpus is never placed in the model context; only the top-k units for at most H information needs enter the reasoning buffer. The composer’s context therefore scales with O(Hk) rather than corpus length, allowing the same reasoning procedure to operate over short and long collections.

![](images/2819fa848c519ba99e074a9b3a807fc34d51190b36a03905559bbb604162255b.jpg)  
(a) Per-hop candidate budget

![](images/f14dc0883f4d7bb83b854588dbd60bfdbfec4d5333266cd1d526fce2f065216f.jpg)  
(b) Reasoning-unit representation  
Figure 2: Ablations on a fixed HolisQA subset. (a) CR improves rapidly with the number of retrieved units and then saturates, while AR peaks near k = 10. (b) Restricting retrieval to any single unit kind underperforms the complete mixed index.

Efficiency. Efficiency is a part that we can study further. Future systems could batch or distill extraction and planning, cache reusable plans and embeddings, update only passages whose source spans change, and choose the hop count and perhop candidate budget adaptively. These changes would preserve the bounded reasoning buffer while reducing both preprocessing tokens and online latency.

Scaling to very large corpora. Long-context tolerance at inference does not make complete corpus indexing free. For heterogeneous collections at Wikipedia scale, constructing a fine-grained index over every domain may be unnecessary. A promising extension is query-conditioned coarse indexing: first route a query to a small set of plausible topical shards—for example, technology rather than politics or history—and then construct or activate the entity-structured index only within those shards. When routing is uncertain, cross-domain fallback retrieval can expand the search scope, preventing efficiency gains from making potentially relevant evidence irrecoverable.

Data sources. All datasets used in our experiments were obtained from publicly available benchmark releases and publicly accessible repositories.

## 6 Conclusion

We presented RESIN, a multi-chain reasoning framework that uses entity-structured index to find evidence and a compact buffer to reason over it. Focused hop retrieval narrows attention, source links keep every unit auditable, and delayed commitment prevents a locally attractive candidate from prematurely fixing the global answer. Across scientific and general-domain QA, the results and ablations show that this combination improves evidence coverage while making the reasoning process easier to inspect. More broadly, separating evidence addressing from final reasoning provides a practical foundation for extending multi-chain QA to longer and more heterogeneous corpora.

## Limitations

A limitation of this study is that we do not empirically evaluate RESIN on long-context multi-chain reasoning. Existing long-context benchmarks primarily test multi-document question answering or fan-out evidence aggregation (Bai et al., 2024; Zhu et al., 2024; Wang et al., 2024); to our knowledge, they do not isolate the setting in which multiple distinct evidence chains must be preserved and reconciled within a single answer. We therefore lack an appropriate benchmark for directly validating the advantage under long-context multi-chain conditions. Our MS MARCO and QASC evaluations are also limited to 500 and 426 queries, respectively, rather than the complete evaluation splits. These subset results may not capture performance across the full range of questions; extending evaluation to the complete splits remains important when time and computational resources permit. In addition, the offline index is most suitable when one corpus serves many questions: its construction cost may not be well amortized for small or rapidly changing collections. Query planning, hop solving, and final composition also require several staged online calls. Finally, because RESIN is evaluated partly on scientific and medical questions, its outputs should support rather than replace expert judgment. Source grounding improves auditability but does not eliminate errors caused by incomplete corpora or mistakes in extraction, planning, and generation.

## References

Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2024. Self-RAG: Learning to retrieve, generate, and critique through self-reflection. In International Conference on Learning Representations.

Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. 2024. LongBench: A bilingual, multitask benchmark for long context understanding. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 3119–3137, Bangkok, Thailand. Association for Computational Linguistics.

Payal Bajaj, Daniel Campos, Nick Craswell, Li Deng, Jianfeng Gao, Xiaodong Liu, Rangan Majumder, Andrew McNamara, Bhaskar Mitra, Tri Nguyen, Mir Rosenberg, Xia Song, Alina Stoica, Saurabh Tiwary, and Tong Wang. 2016. MS MARCO: A human generated MAchine reading COmprehension dataset. arXiv preprint arXiv:1611.09268. Version 3.

Tong Chen, Hongwei Wang, Sihao Chen, Wenhao Yu, Kaixin Ma, Xinran Zhao, Hongming Zhang, and Dong Yu. 2024. Dense X retrieval: What retrieval granularity should we use? In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 15159–15177.

Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan Larson. 2024. From local to global: A Graph RAG approach to query-focused summarization. arXiv preprint arXiv:2404.16130. Version 2.

Shahul Es, Jithin James, Luis Espinosa Anke, and Steven Schockaert. 2024. RAGAs: Automated evaluation of retrieval augmented generation. In Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations, pages 150–158, St. Julians, Malta. Association for Computational Linguistics.

Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. 2025. LightRAG: Simple and fast retrievalaugmented generation. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 10746–10761, Suzhou, China. Association for Computational Linguistics.

Zhengbao Jiang, Frank F. Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, and Graham Neubig. 2023. Active retrieval augmented generation. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 7969–7992, Singapore. Association for Computational Linguistics.

Ziyan Jiang, Xueguang Ma, and Wenhu Chen. 2024. LongRAG: Enhancing retrieval-augmented generation with long-context LLMs. arXiv preprint arXiv:2406.15319. Version 3.

Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi, Sizhe Zhou, and Yu Su. 2025. From RAG to memory: Non-parametric continual learning for large language models. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 21497–21515. PMLR.

Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense passage retrieval for opendomain question answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, pages 6769–6781.

Tushar Khot, Peter Clark, Michal Guerquin, Peter Jansen, and Ashish Sabharwal. 2020. QASC: A dataset for question answering via sentence composition. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 8082–8090.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-augmented generation for knowledgeintensive NLP tasks. In Advances in Neural Information Processing Systems, volume 33, pages 9459– 9474.

Zhuoqun Li, Xuanang Chen, Haiyang Yu, Hongyu Lin, Yaojie Lu, Qiaoyu Tang, Fei Huang, Xianpei Han, Le Sun, and Yongbin Li. 2025. StructRAG: Boosting knowledge intensive reasoning of LLMs via inference-time hybrid information structurization. In International Conference on Learning Representations.

Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2024. Lost in the middle: How language models use long contexts. Transactions ofthe Association for Computational Linguistics, 12:157–173.

Xuanyu Meng, Xing Fan, Xinyi Fan, Chenlei Guo, Yixuan Xie, and Jiawei Han. 2026a. EnSIMem:

Entity-structured indexing for long-term agent memory. arXiv preprint arXiv:2609.27279. Version 2.

Xuanyu Meng, Jiashuo Sun, Jash Rajesh Parekh, and Jiawei Han. 2026b. EnSI-RAG: Entity-structureindexed retrieval-augmented generation for longdocument question answering. arXiv preprint arXiv:2608.21252. Version 1.

Jash Rajesh Parekh, Pengcheng Jiang, and Jiawei Han. 2025. Structure-augmented reasoning generation. arXiv preprint arXiv:2506.08364. Version 4.

Jash Rajesh Parekh, Wonbin Kweon, Joey Chan, Rezarta Islamaj, Robert Leaman, Pengcheng Jiang, Chih-Hsuan Wei, Zhizheng Wang, Zhiyong Lu, and Jiawei Han. 2026. Condition-gated reasoning for context-dependent biomedical question answering. In Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2, pages 11774–11785.

Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence embeddings using siamese BERTnetworks. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 3982–3992.

Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and Christopher D. Manning. 2024. RAPTOR: Recursive abstractive processing for tree-organized retrieval. In International Conference on Learning Representations.

Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. 2023. Interleaving retrieval with chain-of-thought reasoning for knowledgeintensive multi-step questions. In Proceedings ofthe 61st Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 10014–10037.

Minzheng Wang, Longze Chen, Fu Cheng, Shengyi Liao, Xinghua Zhang, Bingli Wu, Haiyang Yu, Nan Xu, Lei Zhang, Run Luo, Yunshui Li, Min Yang, Fei Huang, and Yongbin Li. 2024. Leave no document behind: Benchmarking long-context LLMs with extended multi-doc QA. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 5627–5646, Miami, Florida, USA. Association for Computational Linguistics.

Nengbo Wang, Tuo Liang, Vikash Singh, Chaoda Song, Van Yang, Yu Yin, Jing Ma, Jagdip Singh, and Vipin Chaudhary. 2026. CausalRAG2: Hierarchical causal knowledge graph design for RAG. In Proceedings of the 43rd International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research, pages 127971–127995. PMLR.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc V. Le, and Denny Zhou. 2022. Chain-of-thought prompting elicits reasoning in large language models. In

Advances in Neural Information Processing Systems, volume 35, pages 24824–24837.

Xueqiang Xu, Jinfeng Xiao, James Barry, Mohab Elkaref, Jiaru Zou, Pengcheng Jiang, Yunyi Zhang, Maxwell J Giammona, Geeth De Mel, and Jiawei Han. 2026. Zero-shot open-schema entity structure discovery. In Proceedings ofthe 19th Conference of the European Chapter ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 7547–7561, Rabat, Morocco. Association for Computational Linguistics.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2023. ReAct: Synergizing reasoning and acting in language models. In International Conference on Learning Representations.

Andrew Zhu, Alyssa Hwang, Liam Dugan, and Chris Callison-Burch. 2024. FanOutQA: A multi-hop, multi-document question answering benchmark for large language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 18–37, Bangkok, Thailand. Association for Computational Linguistics.

## A Experimental Protocol

This appendix provides the benchmark, baseline, efficiency, metric, and implementation details omitted from the main paper.

A.1 Benchmark Details
<table><tr><td colspan="2">Benchmark</td><td>Domain and source</td><td>Question and reasoning focus</td><td>Evidence setting</td></tr><tr><td colspan="2">HolisQA (Wang et al., 2026)</td><td>Scientific articles span- ning biology, business, computer science, medicine, and psychol- ogy</td><td>Open-ended holistic questions whose answers require integrating state- ments distributed across an article, including entity properties, relations, conditions, and causal processes</td><td>Gold answers and multiple supporting sentences are linked to the source ar- ticles. We evaluate the complete re- leased dataset and report results sepa- rately for its five domains.</td></tr><tr><td rowspan="2">MS (Bajaj 2016)</td><td>MARCO et al.,</td><td>Anonymized queries sampled from a produc- tion web-search system, paired with passages from retrieved web</td><td>Open-ended question answering in which a system must identify useful evidence among noisy, partially rel- evant search passages and produce a concise response</td><td>Candidate web passages accompany each query, and human annotators pro- vide reference answers grounded in passages judged to contain sufficient information.</td></tr><tr><td>(Khot et al., 2020)</td><td>documents Grade-school science questions paired with a large corpus of elemen- tary scientific facts</td><td>Multiple-choice questions are de- signed so that two complementary facts must be composed, commonly by connecting a concept in the ques- tion to the correct answer through an intermediate concept</td><td>Gold fact pairs expose the two-step ev- idence chain supporting each answer, allowing context recall to measure whether both complementary pieces of evidence were retrieved.</td></tr></table>

Table 6: Benchmarks used in the main evaluation and the distinct evidence-reasoning capabilities they test.

A.2 Baseline Details
<table><tr><td>Baseline</td><td>Family</td><td>Retrieval and reasoning mechanism</td></tr><tr><td>NaiveGeneration (Wang et al., 2026)</td><td>No retrieval</td><td>Generates an answer directly from the question, providing a retrieval-free reference for measuring the contribution of external evidence.</td></tr><tr><td>BM25 (Wang et al., 2026)</td><td>Sparse retrieval</td><td>Ranks passages by lexical term matching and inverse document frequency, then supplies the highest-ranked passages to the answer generator.</td></tr><tr><td>StandardRAG (Lewis et al., 2020)</td><td>Dense retrieval</td><td>Encodes the question and passages in a shared semantic space, retrieves the nearest passages, and conditions generation on their text.</td></tr><tr><td>GraphRAG Global (Edge et al., 2024)</td><td>Graph-based RAG</td><td>Aggregates community reports across a corpus-level entity graph to answer broad questions requiring global thematic evidence.</td></tr><tr><td>GraphRAG Local (Edge et al., 2024)</td><td>Graph-based RAG</td><td>Starts from query-matched entities and retrieves their local graph neigh- borhoods, associated relations, claims, and source text.</td></tr><tr><td>LightRAG (Guo et al., 2025)</td><td>Graph-based RAG</td><td>Builds a lightweight entity-relation graph and combines local entity re- trieval with global concept-level retrieval.</td></tr><tr><td>HippoRAG2 (Jiménez Gutiérrez et al., 2025)</td><td>Structural RAG</td><td>Treats the corpus as a non-parametric associative memory and propagates activation over linked concepts for multi-step retrieval.</td></tr><tr><td>LeanRAG (Wang et al., 2026)</td><td>Structural RAG</td><td>Organizes extracted knowledge into a lightweight hierarchical graph and retrieves evidence across semantically related graph regions.</td></tr><tr><td>CausalRAG (Wang et al., 2026)</td><td>Causal RAG</td><td>Extracts causal relations and uses them to augment evidence retrieval,</td></tr><tr><td>CausalRAG2 (Wang et al., 2026)</td><td>Causal RAG</td><td>without explicit hierarchical causal reasoning. Constructs hierarchical causal knowledge graphs and applies top-down causal pruning to select evidence for answer generation.</td></tr></table>

Table 7: Baselines transcribed from the CausalRAG2 evaluation suite. Their reported scores, efficiency values, and costs are taken from that work.

## A.3 Efficiency Details

Table 8: Cost and latency on HolisQA. Baseline entries are transcribed from the CausalRAG2 efficiency table and were not independently reproduced; only the RESIN row is measured in our environment. Construction values are normalized per 1K corpus tokens, online values per query, and evaluation cost is excluded.
<table><tr><td rowspan="2">Method</td><td colspan="3">Construction (per 1K corpus tokens)</td><td colspan="3">Online query (per query)</td></tr><tr><td>Tokens</td><td>Time (s)</td><td>Cost ($)</td><td>Tokens</td><td>Time (s)</td><td>Cost ($)</td></tr><tr><td>NaiveGeneration</td><td>0.00</td><td>0.000</td><td>0.00000</td><td>553.63</td><td>0.354</td><td>0.000078</td></tr><tr><td>BM25</td><td>0.00</td><td>0.000</td><td>0.00000</td><td>967.90</td><td>0.562</td><td>0.000110</td></tr><tr><td>StandardRAG</td><td>0.00</td><td>0.000</td><td>0.00000</td><td>1016.56</td><td>0.638</td><td>0.000120</td></tr><tr><td>GraphRAG Global</td><td>20241.49</td><td>19.694</td><td>0.00340</td><td>6566.85</td><td>1.850</td><td>0.000680</td></tr><tr><td>GraphRAG Local</td><td>20241.49</td><td>19.694</td><td>0.00340</td><td>4968.42</td><td>1.306</td><td>0.000510</td></tr><tr><td>LightRAG</td><td>11245.55</td><td>7.317</td><td>0.00150</td><td>1517.70</td><td>0.487</td><td>0.000180</td></tr><tr><td>HippoRAG2</td><td>5404.08</td><td>4.647</td><td>0.00092</td><td>3125.54</td><td>0.641</td><td>0.000320</td></tr><tr><td>LeanRAG</td><td>8893.11</td><td>5.589</td><td>0.00120</td><td>5031.30</td><td>0.731</td><td>0.000460</td></tr><tr><td>CausalRAG</td><td>9764.49</td><td>9.022</td><td>0.00140</td><td>5468.44</td><td>0.849</td><td>0.000520</td></tr><tr><td>CausalRAG2</td><td>9864.49</td><td>9.622</td><td>0.00150</td><td>5075.65</td><td>0.801</td><td>0.000490</td></tr><tr><td>RESIN</td><td>14328.85</td><td>6.880</td><td>0.00261</td><td>30333.35</td><td>2.601</td><td>0.00214</td></tr></table>

The offline token increase reflects two LLM stages that organize passages and extract source-grounded, typed units. Online cost is dominated by planning, per-hop solving, and final composition rather than dense retrieval. The values should not be read as a hardware-controlled head-to-head comparison: the baseline rows are paper-reported, and the released CausalRAG2 pipeline did not complete a full-scale local run in our environment.

## A.4 RAGAS Evaluation Metrics

We use RAGAS answer relevancy (AR) and context recall (CR) (Es et al., 2024). Both metrics are computed on a 0–1 scale and reported as percentages in the main tables.

Answer relevancy. AR measures whether the generated answer directly addresses the input question. RAGAS generates N synthetic questions g<sub>i</sub> from the answer and compares their embeddings with that of the original question $q \mathrm { : }$

$$
\mathrm { A R } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \cos ( E ( g _ { i } ) , E ( q ) ) .\tag{4}
$$

We use $N = 3 .$ , corresponding to strictness 3. A high score indicates that the response is focused and pertinent; incomplete, tangential, or unnecessarily verbose responses receive lower scores. AR does not by itself establish factual correctness: an incorrect answer can still be relevant if it directly addresses the question.

Context recall. CR measures whether retrieval recovers the information required by the reference answer. The evaluator decomposes the reference answer into claims and judges whether each claim is attributable to the retrieved source passages:

$$
\mathrm { C R } = { \frac { N _ { \mathrm { s u p p o r t e d } } } { N _ { \mathrm { r e f e r e n c e } } } } .\tag{5}
$$

Here, $N _ { \mathrm { s u p p o r t e d } }$ is the number of reference claims attributable to retrieved context, and $N _ { \mathrm { r e f e r e n c e } }$ is the total number of reference claims. A high CR therefore means that little required evidence was missed. It evaluates evidence coverage rather than response quality and does not penalize irrelevant retrieved passages. AR and CR are complementary: CR tests whether the system retrieved sufficient support, while AR tests whether the generated response remains aligned with the question.

## A.5 Implementation Details

Offline construction. Each source article is segmented into sentence-aligned units and passed to an LLM passage constructor that returns contiguous entity-centered spans. The constructor records the main entity, entity type, semantic focus, evidence role, and source sentence IDs. These output records carry an explicit semantic-passage marker, and the reasoning runner preserves every marked record as one extraction input. Its fixed-character chunker is invoked only for unmarked compatibility inputs and is not used for the reported semantic-passage runs. A second LLM pass extracts typed reasoning units from each passage. The normalizer assigns a stable unit ID, canonical predicate, aliases, scope, confidence, evidence quote, and provenance pointer. The passage and extraction prompts never receive question, answer, or gold-context fields.

Online reasoning. The query planner returns the reasoning mode, expected answer type, entity mentions, properties, constraints, and an ordered list of one-hop subquestions. For each hop, the retriever embeds the concatenation of the subquestion, target, and bridge terms, compares it with the cached embeddings of all reasoning units, and retains the top-k candidates. The main configuration uses k = 10; the budget ablation changes only this value. Each hop solver sees only its own subquestion, planner metadata, and candidate set; it does not receive earlier hop answers or gold evidence. Retrieved units are mapped back to their source passages, packed within the evidence budget, and selected for final composition. The final composer receives only the original question and selected source passages; local answers remain in the audit trace but are not passed to the composer. No separate verifier is invoked in this backend.

Models and reproducibility. The dense encoder is all-MiniLM-L6-v2 (Reimers and Gurevych, 2019). Reported RESIN runs explicitly select the hop-wise candidate backend, which skips the codebase’s legacy graph-construction path. Generation stages use GPT-5-nano, and evaluation uses RA-GAS 0.4.3 with Gemini-3.5-Flash-Lite. We serialize passages, units, auxiliary indexes, query plans, embedding caches, per-hop candidates, local answers, final answers, retrieved contexts, and evaluator inputs so that each result can be traced to its model and evidence state. The published Causal-RAG2 rows use the evaluator available to that work and are not presented as a same-runtime rerun.

We initially attempted to use Gemini-2.5-Flash-Lite, but the API reported that it was no longer available to new users, preventing access through our account; we therefore used Gemini-3.5-Flash-Lite. In a local rerun on a Biology subset of HolisQA, the resulting scores were close to those reported by CausalRAG2, providing a sanity check for our evaluation pipeline and supporting the use of its published results as reference comparisons, without establishing identical evaluation conditions.

Metrics. AR and CR use the RAGAS implementation and its configured judge model. HolisQA additionally reports token-level answer-overlap F1 against reference answers; it complements the judge-based metrics but is sensitive to wording. Sentence-ID recall is retained only as a provenance diagnostic rather than substituted for RAGAS context recall. Latency is measured from the first online model call through final composition, excluding one-time offline preprocessing unless explicitly stated.

## A.6 Algorithms

The following pseudocode summarizes the executed candidate-set backend. It is intentionally expressed in terms of serialized passages, reasoning units, and ordered hop records rather than graph nodes or edges. In Algorithm 2, RETRIEVAL-TEXT concatenates a subquestion with its target and bridge terms.

Algorithm 1 Offline Reasoning-Unit Construction   
Require: Corpus D; passage constructor P   
Require: Unit extractor X; normalizer N; encoder E   
Ensure: Unit store U; indexes I; embedding matrix M   
1: U ← ∅; I ← ∅   
2: for all documents d ∈ D do   
3: S ← SENTENCEUNITS(d)   
4: $P _ { d } \gets \mathcal { P } ( S )$   
5: for all passages $p \in P _ { d }$ do   
6: $R \dot {  } \dot { \mathcal { X } ( p ) }$   
7: for all raw units r ∈ R do   
8: u ← N (r, PROVENANCE(p))   
9: assign a stable unit identifier to u   
10: $U  U \cup \{ u \}$   
11: I ← UPDATETYPEDINDEXES(I, u)   
12: end for   
13: end for   
14: end for   
15: M ← E({EMBEDDINGTEXT(u) : u ∈ U})   
16: SERIALIZE(U, I, M)   
17: return U, I, M

Algorithm 2 Hop-Wise Candidate Retrieval   
Require: Question q; planner L; unit store U   
Require: Embeddings M; encoder E   
Require: Budget k; maximum hops H   
Ensure: Query plan P; ordered candidate trace T   
1: $P \gets \bar { \mathcal { L } } ( q , \mathbf { \bar { \phi } } , T \gets \emptyset ) ; T \gets \emptyset$   
2: for all s<sub>h</sub> ∈ P.SUBQUESTIONS in order do   
3: x<sub>h</sub> ← RETRIEVALTEXT(s<sub>h</sub>)   
4: $v _ { h }  E ( x _ { h } )$   
5: $z _ { h } \gets M v _ { h }$   
6: C<sub>h</sub> ← TOPK $( U , z _ { h } , k )$   
7: append $( s _ { h } , \dot { C } _ { h } , z _ { h } [ \dot { C } _ { h } ] )$ to T   
8: end for   
9: return P, T

Algorithm 3 Candidate-Set Answer Generation   
Require: Question q; query plan P; candidate trace T; prove  
nance π   
Require: Hop solver $\mathcal { L } _ { \mathrm { h o p } } ;$ source selector S; composer   
${ \mathcal { L } } _ { \mathrm { f i n a l } }$   
Ensure: Final answer a; auditable intermediate trace A   
1: A ← ∅   
2: for all $( s _ { h } , C _ { h } ) \in T$ do   
3: $a _ { h } \gets \mathcal { L } _ { \mathrm { h o p } } ( s _ { h } , C _ { h } )$   
4: append $\left( s _ { h } , C _ { h } , a _ { h } \right)$ to A ▷ Audit only   
5: end for   
6: R ← PACKSOURCEPASSAGES(T, π, 16000)   
7: J ← S(q, TITLESANDSNIPPETS(R))   
8: if J contains no valid article identifiers then   
9: J ← TOPTERMMATCHEDGROUPS(q, R, 3)   
10: end if   
11: B ← PASSAGESOFGROUPS(R, J)   
12: a ← L<sub>final</sub>(q, B)   
13: SERIALIZE $( \vec { P } , \vec { T } , A , B , a )$   
14: return a, A

## A.7 Prompt Templates

The boxes below reproduce the principal system prompts used by the released implementation. Runtime values such as the article text, question, query plan, and retrieved candidates replace the indicated placeholders. Gold answers and gold evidence are never included in construction, planning, retrieval, or generation prompts.

```jsonl
Passage Construction: System Prompt
You are the offline passage-construction module
of RESIN.
Construct query-independent passages from a
scientific article. Do not answer
any question and do not summarize the whole
article.
RESIN passage contract:
Each passage has one primary entity, mechanism,
intervention, finding,
method, outcome, or other evidence-bearing
topic.
Keep adjacent sentences together when they
jointly define the entity or
explain its properties, relations, mechanism,
or result.
Split when the main evidence center changes.
Passage lengths are semantic and variable. Do
not use a fixed number of
sentences, tokens, or characters as the
passage definition.
Return only contiguous ranges of the supplied
source sentences.
- Do not paraphrase, reorder, or add outside
knowledge.
Unit IDs are 1-based integers. Copy start_unit
and end_unit from the
provided unit labels. Valid IDs range from 1
to N.
Return JSON only in this form:
{"passages":[{"start_unit":1,"end_unit":3,"
main_entity":"...
"entity_type":"...","semantic_focus":"...","
evidence_role":"..."}]}
```

## Reasoning-Unit Extraction: System Prompt

You are the offline extractor for RESIN.   
Return JSON only: {"units":[...]}.   
RESIN represents multi-chain reasoning as typed   
states and directed   
transitions. Extract the smallest evidence  
backed units that can be composed.   
Each unit must have:   
kind: entity\_fact, type\_property, entity\_type,   
causal\_edge, relation, or rule;   
subject; subject\_kind: entity, type, variable,   
event, state, or unknown;   
predicate; object; conditions; conclusion;   
source\_state; target\_state;   
aliases; scope; modality; polarity; confidence;   
evidence\_quote.

Use these conventions:   
entity\_fact: an ordinary fact about a specific   
entity.   
type\_property: a generic property of a type,   
such as Humans are mortal.   
entity\_type: an explicit membership assertion,   
such as Socrates is a human.   
causal\_edge: a directed transition, such as   
enzyme deficiency causes substrate   
accumulation. Use source\_state and   
target\_state when the sentence expresses   
property-value states. Preserve the direction   
and any modality such as may,   
can, or is thought to.   
rule: only a conditional statement with non  
empty conditions and a conclusion,   
such as If people drink more water, health   
improves. Do not use rule for a   
simple one-edge causal fact.   
For a sentence containing multiple causal steps,   
emit multiple causal\_edge   
units, one per supported transition, while   
keeping the same evidence\_quote.   
Do not infer a transition merely because two   
entities occur in the same text.   
Do not merge a statue with the person it   
represents. Keep entity type,   
context, polarity, and uncertainty explicit.   
Every unit needs an exact short   
evidence\_quote.

The extraction user message contains only article metadata and the delimited article chunk: ARTICLE\_METADATA: {metadata} and ARTICLE\_CHUNK: «<{text}»>.

Query Planning: System Prompt   
You are the RESIN query planner. Do not answer   
the question.   
Decompose the question into an ordered list of   
minimal one-hop subquestions.   
Each subquestion must be answerable from a small   
set of evidence-backed   
reasoning units and should expose the bridge   
needed by the next hop. Do not   
write a multi-hop subquestion. Use at most the   
requested MAX\_HOPS.   
Return JSON only:   
{"reasoning\_mode":"entity\_join |   
type\_inheritance | conditional\_rule |   
causal\_chain | temporal | comparison | mixed",   
"answer\_type":"entity | boolean | date |   
explanation | unknown",   
"target\_description":"","entity\_mentions":[],"   
type\_hints":[],   
"properties":[],"constraints":[],"   
source\_constraints":[],   
"time\_constraints":[],"bridge\_terms":[],   
"subquestions":[{"hop":1,"question":"","target   
"depends\_on":[],"bridge\_terms":[],"   
reasoning\_type":""}]}

The corresponding user template is MAX\_HOPS: {max\_hops} followed by QUESTION: {query}.

Hop Solving: System Prompt   
You are a RESIN hop solver.   
Answer only the supplied one-hop subquestion   
using the candidate reasoning   
units for this hop. Do not use candidates from   
another hop and do not add   
outside knowledge. The answer should expose the   
entity, property, value, or   
bridge needed by later hops. If the candidates   
do not support the subquestion,   
return insufficient rather than guessing.   
Return JSON only:   
{"answer":"","status":"supported | insufficient   
| conflicted",   
"supporting\_unit\_ids":[],"reasoning\_step":"","   
confidence":0.0}

At runtime, the user payload contains the hop number, subquestion, target, dependency metadata, and that hop’s ranked candidate reasoning units. It contains neither gold evidence nor gold answers.

## Final Composition: System Prompt

Use only the retrieved source passages to answer   
the research question.   
Return JSON with answer first: {"answer":"...","   
status":"supported | insufficient",   
supporting\_unit\_ids":[]}.   
Form a self-contained answer of two to four   
short sentences, about 55-85 words. Cover   
each requested factor, mechanism, comparison   
, outcome, and condition found in the   
passages. Reuse the source's exact technical   
terms and short phrases for the central   
findings; avoid broad synonyms that erase   
the findings. Preserve polarity and   
uncertainty.   
Begin with the substantive answer. Do not   
describe the analysis, give background   
definitions, list unrelated possibilities,   
cite sources, or add a generic conclusion.   
If evidence is partial, give the supported   
facts without a long caveat. Never use a   
reference answer or outside knowledge.

The final user payload contains only the original question and selected source passages. It does not contain the query plan, candidate-unit records, or generated hop answers.

## B Case Studies

We present selected cases in which CausalRAG2 omits a required evidence component and RESIN recovers it. The comparison examines original source evidence, saved CausalRAG2 prompts, and the latest RESIN answers. Each table isolates the missing component and its architectural significance; it does not equate recovering that component with complete correctness of every additional statement.

Table 9: Kandelia: preserving molecular findings alongside the dominant physiological explanation.
<table><tr><td>Case component</td><td>Details</td></tr><tr><td>Question</td><td>What mechanisms explain low-temperature tolerance in the cold-tolerant Kandelia obovata population, and what molecular insights were obtained?</td></tr><tr><td>Required evidence</td><td>The answer needs antioxidant/metabolic adaptation, transcription-factor findings, and SNP/alternative-splicing evidence. A physiological explanation alone does not cover the requested molecular insights.</td></tr><tr><td>CausalRAG2 evidence</td><td>The exported recall context includes SNP and splicing information. The final generation prompt instead retains antioxidant activity, named transcription factors, and metabolic/signaling pathways, without the SNP/splicing findings.</td></tr><tr><td>CausalRAG2 response</td><td>The answer explains catalase-driven ROS management, WRKY/NAC/MYB/ERF expression, and enriched pathways, but does not report SNP or alternative-splicing findings. Missing molecular evidence.</td></tr><tr><td>Why the omission occurs</td><td>The recorded evidence assembly removes a distinct molecular evidence category before final generation. The answer then elaborates the retained antioxidant/transcriptional explanation rather than recovering the missing category. The trace locates this loss between broad recall and the final prompt, not at initial extraction.</td></tr><tr><td>RESIN organization</td><td>Typed units make molecular findings independently addressable. Decomposed needs can distinguish the physiological mechanism from molecular evidence; the latter need not depend on selection of the former&#x27;s causal chain.</td></tr><tr><td>RESIN source evidence</td><td>The latest mapped passages preserve both the transcription-factor findings and the statement that SNP diversity and alternative splicing may enhance protein function and contribute to cold tolerance.</td></tr><tr><td>RESIN response</td><td>The answer retains antioxidant/metabolic responses and the named transcription factors, and additionally reports potential SNP/splicing contributions to cold tolerance. Missing evidence category recovered. It does not further explain the proposed protein-function contribution.</td></tr><tr><td>Reference answer</td><td>The reference includes antioxidant and metabolic regulation, transcriptional/post-transcriptional responses, named TF families, and possible SNP/splicing contributions through enhanced protein function.</td></tr><tr><td>Key difference</td><td>Relevant evidence spans several molecular roles rather than one dominant mechanism. REsIN provides separate access to these roles and recovers their original passages for composition. The latest answer includes a category that is lost from CausalRAG2&#x27;s final evidence set.</td></tr></table>

Table 10: CD1c: retaining a comparative structural property during answer composition.
<table><tr><td>Case component</td><td>Details</td></tr><tr><td>Question</td><td>How does CD1c present lipids to TCRs, and how does this mechanism differ from other antigen-presenting molecules?</td></tr><tr><td>Required evidence</td><td>The mechanism combines simultaneous two-lipid loading with sideways presentation of bulky headgroups through a side portal. Two-lipid loading is a distinguishing property, not merely another step in the presentation pathway.</td></tr><tr><td>CausalRAG2 evidence</td><td>Its final generation prompt contains a fragment stating that two lipid antigens are presented from the top and side of the cleft. Its reranking draft also mentions two lipids. The required cardinality fact is therefore available before the final answer.</td></tr><tr><td>CausalRAG2 response</td><td>The answer describes lipid binding, sideways headgroups, the side portal, and TCR recognition, but omits simultaneous two-lipid loading. Missing distinguishing property.</td></tr><tr><td>Why the omission occurs</td><td>Available evidence is not fully carried into final synthesis: the answer explains how presentation works without preserving a central property needed to explain how it differs. This is a composition omission, not a failure to retrieve the two-lipid fact.</td></tr><tr><td>RESIN organization</td><td>An entity-specific property can be represented separately from the mechanistic transition. Need-specific retrieval locates the explanatory and comparative claims; source selection recovers original passages for direct final composition, without passing generated hop answers.</td></tr><tr><td>RESIN source evidence</td><td>The latest mapped passage explicitly states that two lipids, rather than one, bind in the CD1c cleft, and that bulky headgroups protrude laterally through a side portal while permitting TCR recognition.</td></tr><tr><td>RESIN response</td><td>The answer states that CD1c can bind two lipids simultaneously and explains sideways protrusion of bulky headgroups through the side portal. Distinguishing property recovered. Its additional peptide-MHC comparison is separate from this source-grounded finding.</td></tr><tr><td>Reference answer</td><td>The reference combines two-lipid binding, sideways orientation, headgroup protrusion, and TCR recognition to explain the distinctive presentation mechanism.</td></tr><tr><td>Key difference</td><td>Recovering a pathway does not automatically preserve the property that makes it distinctive. REsIN&#x27;s interface gives explanatory and comparative evidence separate roles and keeps them jointly available; the latest answer preserves both in this case.</td></tr></table>

Table 11: CKL12: separating a growth phenotype from evidence of BR regulation.
<table><tr><td>Case component</td><td>Details</td></tr><tr><td>Question</td><td>How does CKL12 contribute to plant growth, and what evidence suggests regulation by brassinosteroid (BR) signaling?</td></tr><tr><td>Required evidence</td><td>The answer requires a growth phenotype and evidence of regulation. Insertion mutants suggest a contribution to hypocotyl and primary-root growth; promoter binding and reduced transcript after transcription-factor knock-down support the BR connection.</td></tr><tr><td>CausalRAG2 evidence</td><td>Its final prompt contains a BR-regulation fragment, including promoter sites and knock-down results, but no hypocotyl or primary-root finding. The saved trace does not identify which earlier processing step first excluded the phenotype.</td></tr><tr><td>CausalRAG2 response</td><td>The answer says CKL12&#x27;s exact growth role is unspecified in the provided items, then describes BR regulation. Growth component unanswered.</td></tr><tr><td>Why the omission occurs</td><td>Regulation and phenotype are different evidence roles. Retaining the regulatory fragment does not supply the missing growth observation, so the final answer cannot connect the two requested parts using its provided evidence.</td></tr><tr><td>RESIN organization</td><td>Growth effects and regulatory support are directly searchable claims, not interchangeable descriptions of CKL12. Need-specific retrieval allocates candidate capacity to each role; the final buffer can preserve the source passages supporting both roles.</td></tr><tr><td>RESIN source evidence</td><td>The latest mapped passage contains the tentative hypocotyl/root finding, transcript-end observations, promoter binding sites, and the knock-down experiment. The phenotype is therefore represented in the source evidence associated with the latest run.</td></tr><tr><td>RESIN response</td><td>The answer names hypocotyl and primary-root growth and includes the BR-related promoter sites. Missing phenotype recovered. It still omits insertion/knock-down details and strengthens the source&#x27;s tentative growth wording.</td></tr><tr><td>Reference answer</td><td>The reference says CKL12 appears to contribute to growth; insertion findings suggest a role for the transcript&#x27;s 3&#x27; end, while promoter and knock-down results support downstream BR regulation.</td></tr><tr><td>Key difference</td><td>A regulator-centered evidence set can answer the signaling part while missing the phenotype. REsIN treats the two needs separately and retains their claims together. This case illustrates recovery of the missing growth component, not complete recovery of every supporting experiment.</td></tr></table>