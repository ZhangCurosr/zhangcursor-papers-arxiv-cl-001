# TEM: Temporal State Tracking Memory for Large Language Model Agents

Pengfeng Song<sup>1\*</sup>, Yunkai Zhuang<sup>1</sup> and Liye Shi<sup>1</sup>

<sup>1</sup>School of Information and Intelligent Engineering,Zhejiang Wanli University,Ningbo,315100,China.

\*Corresponding author(s). E-mail(s): sungkites@163.com;

## Abstract

An agent’s memory must remain useful as information changes across conversations. A new fact may replace an earlier value without making that value irrelevant to a later question. We present Temporal State Tracking Memory (TEM), a training-free framework for tracking these changes. TEM stores facts as versioned records linked to their sources. It retrieves relevant versions, places them in source order and labels historical records before passing the evidence to the answer model. Event memory and original dialogue turns provide details missing from structured facts. We evaluated TEM on 420 questions drawn from 600 sessions involving 60 personas. Across four open-weight models, key-entity hit rates reached 75.71– 97.62% overall and 96.67–100.00% on the historical-residence subset. Adding historical labels to otherwise identical, source-ordered facts increased overall hit rates by 17.14–55.95 percentage points. The evidence was compact: 94–118 tokens per answer, or 48.75–56.73% less than session-aware retrieval-augmented generation (RAG). A further evaluation on 1,540 LoCoMo questions tested hybrid memory under a fixed evidence budget. TEM exceeded session-aware RAG with both backbones; conservative score diferences were 7.92 and 10.58 percentage points. The results support preserving fact versions and exposing their temporal status to the generator when answering questions about repeatedly updated information.

Keywords: large language model agents; long-term memory; temporal state tracking; fact versions; retrieval-augmented generation

## 1 Introduction

A long-term agent accumulates records that may no longer describe its user. Residence, occupation, employer and preferences can change several times across sessions [1–3]. Earlier records still matter when a question refers to the past. The agent therefore needs to know which version applies to the question, including whether the latest record is relevant at all.

Suppose a user moves from Shanghai to Chengdu and then to Suzhou. Suzhou answers “Where does the user live now?” The question “Where did the user live before moving to Chengdu?” requires Shanghai. A memory that keeps only the latest address cannot answer both. Keeping all three addresses makes the historical answer available, but the model still has to identify the right one.

Persistent memory, action feedback and experience-based reflection have extended agents’ use of past interactions [4–6]. Work on hierarchical storage, memory updates, associative retrieval and temporal graphs has also improved access to this information [7–14]. Repeated updates pose a particular problem: several retrieved values may be equally relevant to a question yet refer to diferent states. We examine how the memory can convey these version relationships to the answering model.

TEM records cross-session changes as fact versions without retraining the backbone. New facts are appended, and the status of earlier records is updated from their source order. At query time, relevance scores select the evidence. TEM then orders the selected facts by source and labels historical versions. This separates evidence selection from the presentation of temporal information. Event memory and original turns supply details that the fact records do not capture.

Our contribution is a history-preserving update rule that assigns state over complete fact-version groups. The evidence construction procedure carries this status into retrieval output, so an older record stays historical when the latest version is not retrieved. We evaluate both components across multiple backbones and examine hybrid memory on public multi-session questions.

## 2 Related Work

## 2.1 Long-Term Agent Memory and Structured Organisation

External memory lets an agent read and write information outside its model parameters. Hierarchical management in MemGPT and the historical storage mechanisms studied in MemoryBank and long-term memory augmentation provide ways to retain information beyond the immediate context [7–9]. A-MEM, Mem0 and MemoryOS extend this work through memory associations, updates and hierarchical organisation [10–12]. Temporal information enters the memory itself in Zep’s knowledge graph [13], making it particularly relevant to state tracking. TEM examines how version status is carried from stored records into answer evidence. Memory processing cost is another concern; LightMem addresses it through staged operations [14].

Graph-based retrieval connects information spread across records. Associative retrieval is central to HippoRAG and its successor, while GraphRAG uses graph organisation for query-focused summarisation [15–17]. Beyond these retrieval structures,

MemoRAG uses global memory to guide evidence access, and MemOS considers knowledge access across the memory lifecycle [18, 19].

KG-RAG combines knowledge-graph retrieval and question decomposition for multihop question answering without additional training [20]. CTRN addresses complex time constraints through implicit temporal and relational representations [21]. MGIT models temporal dependencies at multiple granularities [22]. These graph-based questionanswering methods are relevant to TEM’s use of structured evidence. Our focus is the management of fact versions as conversations update a state, including how the resulting status information reaches the generator.

## 2.2 Long-Term Memory Evaluation and Retrieval Augmentation

LongMemEval and LoCoMo evaluate cross-session information use, fact updates and temporal reasoning [23, 24]. LongBench covers a broader set of long-context tasks [25]. Lost in the Middle shows that the position of relevant information can afect model performance [26]. For state tracking, finding a relevant passage does not settle which version answers the question. We study this choice in controlled update tasks and evaluate hybrid retrieval on public multi-session questions.

RAG and dense retrieval provide the basis for accessing external evidence [27, 28]. Self-RAG evaluates evidence, RAPTOR organises it hierarchically, and interleaved retrieval supports chain-of-thought reasoning [29–31]. LLMLingua, LongLLMLingua and RECOMP reduce the input passed to the generator [32–34]. TEM retains version order and state attributes alongside retrieved facts, allowing the generator to interpret the selected evidence in its temporal context.

## 3 Temporal State Tracking Memory

TEM stores each fact with its provenance and places successive values in a version group. Source order determines the status of each version. Retrieval then selects the facts to include in the answer evidence. Figure 1 gives the overall framework; Fig. 2 and Algorithm 1 detail the update and retrieval procedure.

## 3.1 Fact Versions with Provenance

A fact record contains the subject, relation, object, source session and source index. Records with the same subject and relation belong to one version group:

$$
f = ( s , r , o , \sigma , \tau ) , \qquad g ( f ) = ( \operatorname { n o r m } ( s ) , r )\tag{1}
$$

The source session identifies the original record. Its index determines the position of that record in the version sequence. We group records by subject and relation, normalising subject names to lowercase. Excluding the object from the key lets diferent values of a relation remain in the same group.

An update changes a record’s status but leaves its value and provenance intact. The rule assumes single-valued updates arriving in session order. Source indices record when information entered the memory; they do not establish when the underlying event took efect.

TEM: Temporal State Tracking Memory  
![](images/ee2ec739c998667e7df0fa4385e27b77c00346cf35846264c849dcb846a6c749.jpg)  
Fig. 1 TEM framework. Fact versions are linked to source sessions. Retrieved facts appear in source order, with historical labels on superseded versions. Dashed connections supply event and original-text evidence.

## 3.2 History-Preserving Incremental Updates

TEM appends facts from each new session to the existing version set without overwriting earlier values (Fig. 2a):

$$
F _ { t } = F _ { t - 1 } \cup \mathrm { A t t a c h S o u r c e } ( \Delta F _ { t } , \sigma _ { t } )\tag{2}
$$

A frozen model extracts facts from the new session. TEM checks their format and consistency with the source text before attaching the session provenance.

TEM determines status from the complete version group before selecting evidence. A version is historical if its source index is smaller than the maximum index in that subject–relation group:

$$
\begin{array} { r } { \tau ^ { * } ( g ) = \operatorname* { m a x } \{ \tau ( f ) : f \in F _ { g } \} , } \\ { h ( f ) = \mathbf { 1 } \big [ \tau ( f ) < \tau ^ { * } ( g ( f ) ) \big ] . } \end{array}\tag{3}
$$

The historical attribute is one for a superseded version and zero when no later source exists. Both the value and its provenance remain in memory. A missed retrieval of the latest version therefore cannot make an older record appear current.

The label identifies a record as historical; its source index places it among the earlier versions. All records at the maximum source index are treated as current. This rule leaves conflicting values within the same session unresolved.

## 3.3 Relevance Retrieval and Temporal Evidence

Retrieval selects facts by relevance without penalising older versions for their age. TEM then places the selected facts in source order (Fig. 2b). The query, source order and version status are available to the generator when it chooses an answer.

Candidate scores combine similarity from the shared vector retriever with a weight for the relation type:

$$
u ( f , q ) = \cos ( e ( f ) , e ( q ) ) \big ( 0 . 4 + 0 . 6 w ( f ) \big )\tag{4}
$$

The relation weights are 1.0 for residence and employer, 0.9 for occupation, 0.8 for possession, and 0.6 for likes and dislikes. We apply this scoring rule to all versions and retrieve at most 15, taking those with the highest scores.

TEM sorts the selected records by ascending source index, keeping the fact content and source session. Superseded versions receive a [historical] label. The latest version has no label. Evidence construction and answering follow:

$$
\begin{array} { r l } & { B _ { q } = \mathrm { R e n d e r } ( \mathrm { S o r t } _ { \tau } ( \mathrm { T o p K } ( F , q ) ) , \sigma , h ) , } \\ & { \quad a = \mathrm { L L M } ( q , B _ { q } ) . } \end{array}\tag{5}
$$

The retrieval budget counts fact records. Sorting and labelling leave their values unchanged. TEM does not use the question to exclude current or historical versions in advance; the generator chooses among the retrieved versions using the query and the supplied temporal information.

## 3.4 Update and Evidence Construction Algorithm

Algorithm 1 specifies version updates and retrieval, using frozen models for fact extraction and answer generation.

Algorithm 1 Fact-version updates and temporal evidence construction   
Input: stored versions F ; new session D and source σ; optional query q.   
Parameters: candidate count k; relation weights w.   
Output: updated F; evidence B and answer a if queried.   
1: ∆F ← ExtractAndValidate(D, F)   
2: F ← F ∪ AttachSource(∆F, σ)   
3: for each subject–relation group g ∈ Groups(F) do   
4: τ<sup>∗</sup>(g) ← max{τ(f) : f ∈ F<sub>g</sub>}   
5: for each version f ∈ F do   
6: h(f) ← 1[τ(f) < τ<sup>∗</sup>(g)]   
7: end for   
8: end for   
9: if no query then return F   
10: C ← TopK(F, q, k), scored by Eq. (4)   
11: C ← Sort (C), ascending source index   
12: B ← ∅   
13: for each candidate f ∈ C do   
14: b ← RenderFactAndSource(f)   
15: if h(f) = 1 then append [historical]   
16: B ← Concat(B, b)   
17: end for   
18: a ← FrozenLLM(q, B)   
19: return F, B, a

AttachSource links each fact to its source session and index.

![](images/46b4effd7fd56bcb187dc9c9d890206e9fa442eb8826f4898a6ad90f487c09b3.jpg)  
Fig. 2 Fact-version updates and evidence construction. a, Appending a new version updates the status of retained records. b, Retrieval selects relevant facts, which are then ordered by source and labelled if historical. Symbols v and s denote a fact version and its source session.

## 3.5 Event Memory and Original-Text Evidence

Some information does not fit a structured fact value, including event context and conditions on behaviour. TEM stores event summaries, free-text memories and original turns for this purpose. Source-session links connect these records to fact versions. Hybrid retrieval draws on both sources of evidence.

We use the fact-memory configuration TEM-Status for state-update experiments and TEM-Hybrid-State for public question answering. The hybrid configuration combines the evidence sources under a common budget. Its unmarked control uses the same candidates in the same order (Sections 4.1 and 4.6).

## 4 Experiments and Results

## 4.1 Experimental Setup

The state-update benchmark contains 60 personas, 600 sessions and 420 questions involving residence, occupation and drinks. There are 180 current-state questions, 180 other historical questions and 60 questions about residence immediately before the latest move. For a historical question, the reference answer is the most recent applicable fact before the target update. Running six memory configurations with four backbones produced 10,080 answers.

We compared TEM with naive RAG, session-aware RAG and full context. Sessionaware RAG keeps retrieved source turns in session order. The fact-memory comparison uses three configurations. Overwrite keeps only the latest version. Unmarked keeps all versions in source order, and TEM-Status adds historical labels to the same content without changing that order.

Experiments used an A100 40 GB GPU. Extraction and answering within a configuration used the same frozen backbone. We retrieved at most 15 candidates with all-mpnet-base-v2 and set temperature to zero and the random seed to 42. Llama and Qwen ran in BF16 with a 512-token answer limit; Qwen thinking was disabled. For gpt-oss, we used MXFP4, low reasoning efort and a 4,096-token answer limit. All backbones had an 8,192-token extraction output limit.

We measured state-update performance by key-entity hit rate, counting an answer as a hit if it contained the reference entity under case-insensitive matching. Public question answering used strict semantic scoring, which required the answer to satisfy both factual and temporal conditions.

## 4.2 Overall Question Answering

With fact content and source order held constant, TEM-Status improved overall key-entity hit rates by 17.14–55.95 percentage points over Unmarked across the four backbones (Table 1).

Table 1 Efect of historical labels on overall key-entity hit rate
<table><tr><td>Backbone</td><td>Unmarked (%)</td><td>TEM (%)</td><td>Gain (pp)</td></tr><tr><td>Llama-3.1-8B</td><td>19.76</td><td>75.71</td><td>+55.95</td></tr><tr><td>Qwen3.5-4B</td><td>48.10</td><td>97.62</td><td>+49.52</td></tr><tr><td>Qwen3.5-9B</td><td>74.52</td><td>91.67</td><td>+17.14</td></tr><tr><td>gpt-oss-20b</td><td>55.24</td><td>96.67</td><td>+41.43</td></tr></table>

Each configuration contains 420 questions. Hit rates are percentages; gains are percentage points (pp).

TEM-Status achieved overall key-entity hit rates of 75.71–97.62%, exceeding sessionaware RAG by 1.43–40.00 percentage points and full context by 2.14–30.24 points (Table 2).

Table 2 Overall key-entity hit rates across four backbones and six configurations (%)
<table><tr><td>Method</td><td>Llama-3.1-8B</td><td>Qwen3.5-4B</td><td>Qwen3.5-9B</td><td>gpt-oss-20b</td></tr><tr><td>Naive RAG</td><td>14.52</td><td>42.38</td><td>14.05</td><td>46.43</td></tr><tr><td>Session-aware RAG</td><td>56.67</td><td>57.62</td><td>79.76</td><td>95.24</td></tr><tr><td>Full context</td><td>49.76</td><td>95.48</td><td>87.38</td><td>66.43</td></tr><tr><td>Overwrite</td><td>43.33</td><td>29.76</td><td>41.43</td><td>34.76</td></tr><tr><td>Retain, unmarked</td><td>19.76</td><td>48.10</td><td>74.52</td><td>55.24</td></tr><tr><td>TEM-Status</td><td>75.71</td><td>97.62</td><td>91.67</td><td>96.67</td></tr></table>

Each entry covers 420 questions. Bold indicates the highest value for a backbone.

## 4.3 Historical States after Repeated Updates

Overwrite recovered few historical residence values, with hit rates of 0–8.33%. Keeping earlier versions without labels raised the rates to 41.67–81.67%. TEM-Status reached 96.67–100.00% after historical labels were added to the same evidence (Fig. 3a).

TEM-Status exceeded session-aware RAG by 10.00–88.33 percentage points and full context by 3.33–40.00 points (Table 3).

![](images/b094ba054532e4767946912e82c61aa183a17a234b0718d2ad571666fe4211a0.jpg)

![](images/d635e1ff9b1ebbf6460796a71b811335df89bcd620f374bc45fb962c1c860fa4.jpg)  
Fig. 3 Efects of version retention and historical labels. a, Historical-residence hit rates (60 questions). b, Overall gains over Unmarked (420 questions). pp denotes percentage points.

Table 3 Historical-residence retrieval after repeated updates (key-entity hit rate, %)
<table><tr><td>Method</td><td>Llama-3.1-8B</td><td>Qwen3.5-4B</td><td>Qwen3.5-9B</td><td>gpt-oss-20b</td></tr><tr><td>Naive RAG</td><td>0.00</td><td>0.00</td><td>0.00</td><td>5.00</td></tr><tr><td>Session-aware RAG</td><td>10.00</td><td>88.33</td><td>20.00</td><td>86.67</td></tr><tr><td>Full context</td><td>58.33</td><td>96.67</td><td>73.33</td><td>91.67</td></tr><tr><td>Overwrite</td><td>0.00</td><td>8.33</td><td>0.00</td><td>0.00</td></tr><tr><td>Retain, unmarked</td><td>41.67</td><td>65.00</td><td>81.67</td><td>76.67</td></tr><tr><td>TEM-Status</td><td>98.33</td><td>100.00</td><td>100.00</td><td>96.67</td></tr></table>

Each entry covers 60 questions. Bold indicates the highest value for a backbone.

## 4.4 Ablation of State Labels

Adding historical labels gave 235 more hits with Llama-3.1-8B, 208 with Qwen3.5-4B, 72 with Qwen3.5-9B and 174 with gpt-oss-20b over 420 questions (Table 4). These counts measure the net change in hits with fact content and source order held fixed.

Table 4 Overall hit counts with and without historical labels
<table><tr><td>Backbone</td><td>Unmarked hits</td><td>TEM hits</td><td>Net difference</td></tr><tr><td>Llama-3.1-8B</td><td>83</td><td>318</td><td>+235</td></tr><tr><td>Qwen3.5-4B</td><td>202</td><td>410</td><td>+208</td></tr><tr><td>Qwen3.5-9B</td><td>313</td><td>385</td><td>+72</td></tr><tr><td>gpt-oss-20b</td><td>232</td><td>406</td><td>+174</td></tr></table>

Each configuration contains 420 questions. Net gains are diferences in total hit counts.

The selected examples illustrate errors in identifying the time or relation requested by a question (Fig. 4). Under the control configurations, the same backbone sometimes used an old value as current, confused versions around an update or returned an unrelated relation. TEM recovered the requested value in each of these examples.

✓ correct answer × incorrect answer S: source session [...] omitted text  
![](images/dcf14f2d94bd3c425a6db174345980177f302370822ca42fbfac5214ade062a2.jpg)  
Fig. 4 Selected successful corrections across backbones. Arrows show source-session order; bold states give the reference values. Check marks and crosses indicate correct and incorrect answers. The control configuration is identified in each panel. Panel c marks omitted text with [. . . ].

Historical labels increased hit rates by 18.33–56.67 percentage points on the 60 residence questions (Table 5). The remaining 360 questions combine current-state and other historical queries. Gains of 16.94–55.83 points on this group show that the improvement extended beyond the historical-residence subset.

## 4.5 Answer-Evidence Length

TEM used 94.05–117.55 tokens of answer evidence on average, 48.75–56.73% less than session-aware RAG (Table 6). These lengths exclude memory construction and embedding computation.

TEM evidence ranged from 84 to 125 tokens across all queries (Fig. 5). Even the shortest inputs from session-aware RAG and full context were longer, at 202 and 165 tokens.

Table 5 Key-entity hit rates and label gains by query group
<table><tr><td>Backbone</td><td>Other queries Unmarked / TEM</td><td>Gain (pp)</td><td>Pre-move residence Unmarked / TEM</td><td>Gain (pp)</td></tr><tr><td>Llama-3.1-8B</td><td>16.11 / 71.94</td><td>+55.83</td><td>41.67 / 98.33</td><td>+56.67</td></tr><tr><td>Qwen3.5-4B</td><td>45.28 / 97.22</td><td>+51.94</td><td>65.00 / 100.00</td><td>+35.00</td></tr><tr><td>Qwen3.5-9B</td><td>73.33 / 90.28</td><td>+16.94</td><td>81.67 / 100.00</td><td>+18.33</td></tr><tr><td>gpt-oss-20b</td><td>51.67 / 96.67</td><td>+45.00</td><td>76.67 / 96.67</td><td>+20.00</td></tr></table>

Hit rates are percentages and gains are percentage points. Counts for other queries are obtained by subtracting the historical-residence subset from the overall counts.

![](images/73b51b3fbfdc7e7ee774fff8d958990d2e5e0a27aa13b0a879ae215865499649.jpg)  
Fig. 5 Answer-evidence length distributions, with 420 queries per configuration. Boxes show the interquartile range, centre lines the median, and whiskers the minimum and maximum.

Table 6 Mean evidence length during answering
<table><tr><td>Backbone</td><td>Session-aware RAG</td><td>Full context</td><td>TEM-Status</td><td>Saving (%)</td></tr><tr><td>Llama-3.1-8B</td><td>214.72</td><td>180.13</td><td>95.93</td><td>55.32</td></tr><tr><td>Qwen3.5-4B</td><td>229.36</td><td>199.52</td><td>99.25</td><td>56.73</td></tr><tr><td>Qwen3.5-9B</td><td>229.36</td><td>199.52</td><td>117.55</td><td>48.75</td></tr><tr><td>gpt-oss-20b</td><td>206.55</td><td>171.95</td><td>94.05</td><td>54.47</td></tr></table>

Evidence length is measured in tokens per question. Savings are relative to session-aware RAG.

## 4.6 Public Multi-Session Question Answering

We evaluated public question answering on 10 LoCoMo conversations containing 1,540 questions: 282 multi-hop, 321 temporal, 96 open-domain and 841 single-hop. Llama-3.1-8B and Qwen3.5-9B each ran four configurations, producing 12,320 answers in total. TEM-Hybrid-State, TEM-Hybrid-Unmarked and session-aware RAG shared a 1,024-token evidence limit. The two hybrid configurations used identical candidates and ordering. A long-context configuration provided a further reference. All memories were built from the source conversations only.

For each backbone, we kept the answer prompt, zero temperature, seed 42 and 1,024-token answer limit fixed across configurations. Qwen thinking was disabled. Inputs to the long-context reference were truncated at the 32,768-token model window; this afected 1,151 questions per backbone. We used gpt-oss-20b to score factual and temporal correctness against the reference answers and supporting turns.

Ambiguity in the reference information left 216 answers unresolved. We retained the full question count as the denominator and assigned these cases zero or one to obtain score bounds. Eleven truncated or empty answers counted as incorrect.

Under the same retrieval budget, Llama scored 27.53–29.22% with TEM and 17.92–19.61% with session-aware RAG (Table 7). Qwen scored 35.84–37.53% and 23.70– 25.26%, respectively. Even subtracting the RAG upper bound from the TEM lower bound left gaps of 7.92 and 10.58 percentage points.

Table 7 Strict semantic scores on LoCoMo for two backbones (%)
<table><tr><td>Backbone</td><td>Memory</td><td>Overall</td><td>Temporal</td><td>Multi-hop</td><td>Single-hop</td><td>Open- domain</td></tr><tr><td>Llama-3.1- 8B</td><td>TEM-Hybrid-State</td><td>27.53-29.22</td><td>14.02-17.76</td><td>17.73-18.09</td><td>37.57-38.88</td><td>13.54-15.62</td></tr><tr><td>Llama-3.1- 8B</td><td>TEM-Hybrid- Unmarked</td><td>27.40-28.96</td><td>14.33-17.45</td><td>14.89-15.25</td><td>38.17-39.48</td><td>13.54-15.62</td></tr><tr><td>Llama-3.1- 8B</td><td>Session-aware RAG</td><td>17.92-19.61</td><td>8.72-11.84</td><td>12.06-12.77</td><td>23.90-25.33</td><td>13.54-15.62</td></tr><tr><td>Llama-3.1- 8B</td><td>Long-context ref.</td><td>37.66-39.55</td><td>14.64-17.13</td><td>24.11-24.82</td><td>53.51-55.65</td><td>15.62-16.67</td></tr><tr><td></td><td>Qwen3.5-9B TEM-Hybrid-State</td><td>35.84-37.53</td><td>38.63-42.06</td><td>22.70-23.05</td><td>41.38-42.81</td><td>16.67-18.75</td></tr><tr><td>Qwen3.5-9B TEM-Hybrid-</td><td>Unmarked</td><td>35.97-37.66</td><td>38.32-41.43</td><td>25.18-25.89</td><td>40.78-42.21</td><td>17.71-19.79</td></tr><tr><td></td><td>Qwen3.5-9B Session-aware RAG</td><td>23.70-25.26</td><td>23.99-27.10</td><td>19.86-20.57</td><td>25.92–27.23</td><td>14.58-15.62</td></tr><tr><td></td><td>Qwen3.5-9B Long-context ref.</td><td>53.90-56.17</td><td>48.91-52.96</td><td>28.72-30.14</td><td>66.94–68.85</td><td>30.21-32.29</td></tr></table>

Each configuration contains 1,540 questions. Ranges are bounds obtained by assigning unresolved scores zero or one, not confidence intervals. The first three configurations have a 1,024-token evidence limit; the long-context reference has a 32,768-token window.

Historical labels made little diference to overall LoCoMo scores. Mean TEM evidence lengths were 1,014.6 tokens for Llama and 1,017.5 for Qwen, close to sessionaware RAG. The long-context reference scored higher with much larger inputs: 29,671.8 and 30,412.9 tokens, respectively. These public questions require event details and multi-hop associations as well as state information. The similar scores of the marked and unmarked variants suggest that hybrid evidence organisation accounts for most of the observed gain over RAG.

## 4.7 Results Reported for Existing Memory Systems

Published memory-system evaluations provide further context for these results [10–13]. Their generators and scoring metrics are recorded alongside the reported scores in Table 8. TEM was evaluated with open-weight generators and strict semantic scoring.

Table 8 Published results and evaluation settings of representative memory methods
<table><tr><td>Method</td><td>Generator</td><td>Benchmark</td><td>Metric</td><td>Overall</td><td>Temporal</td><td>Single-hop</td></tr><tr><td>A-MEM [10]</td><td>GPT-4o-mini</td><td>LoCoMo</td><td>F1</td><td></td><td>45.85</td><td>44.65</td></tr><tr><td>MemoryOS [12]</td><td>GPT-4o-mini</td><td>LoCoMo</td><td>F1</td><td></td><td>20.02</td><td>35.27</td></tr><tr><td>Mem0 [11]</td><td>GPT-4o-mini</td><td>LoCoMo</td><td>J</td><td>66.88±0.15</td><td>55.51±0.34</td><td>67.13±0.65</td></tr><tr><td>Mem0-graph [11]</td><td>GPT-4o-mini</td><td>LoCoMo</td><td>J</td><td>68.44±0.17</td><td>58.13±0.44</td><td>65.71±0.45</td></tr><tr><td>Zep [13]</td><td>GPT-4o-mini</td><td>LongMemEval-s</td><td>GPT-4o judge</td><td>63.80</td><td>54.10</td><td></td></tr><tr><td>Zep [13]</td><td>GPT-4o</td><td>LongMemEval-s</td><td>GPT-40</td><td>71.20</td><td>62.40</td><td></td></tr><tr><td>TEM-Hybrid</td><td>Llama-3.1-8B</td><td>LoCoMo</td><td>judge Strict judge</td><td>27.53-29.22</td><td>14.02-17.76</td><td>37.57-38.88</td></tr><tr><td>TEM-Hybrid</td><td>Qwen3.5-9B</td><td>LoCoMo</td><td>Strict judge</td><td>35.84-37.53</td><td>38.63-42.06</td><td>41.38-42.81</td></tr></table>

Scores are percentages. Generators, scoring rules and input budgets difer, so values are not a common ranking. A-MEM values are from v11, Table 1; MemoryOS values are from Table 2 of the published version. Mem0 reports standard deviations over 10 evaluations. TEM ranges bound unresolved scores. A dash indicates an unreported or inapplicable value.

## 5 Discussion

The state-update results separate two requirements of historical question answering. A memory must first keep earlier values accessible. Once those values are available, the generator still needs to identify which one applies. In the ablation, historical labels improved hit rates without changing the facts or their source order. Retrieval therefore did not account for the diference between the two configurations. The result supports treating version status as part of the evidence presented to the generator, even when it can in principle be inferred from the ordered records.

On LoCoMo, marked and unmarked hybrid memory obtained similar scores, and both exceeded session-aware RAG under the same evidence budget. The contrast with the state-update task helps define the role of the labels. They directly indicate whether a retained value has been superseded, but do not supply missing event details or cross-session associations. These are also needed for public dialogue questions. The hybrid-memory gain on LoCoMo is therefore better understood at the level of evidence organisation, without attributing it to historical labels alone.

TEM uses source-session order to assign state, assuming explicit single-valued updates. This is suficient for the controlled update setting considered here. A delayed report may place an earlier event in a later session, and contradictory values can occur within one session. Preferences may also remain valid simultaneously. Handling these cases requires event-validity times or explicit treatment of conflicting relations beyond the present ordering rule.

## 6 Conclusion

TEM couples history-preserving fact storage with evidence that makes version status explicit. The update rule operates on complete version groups, and retrieval passes source order and historical labels to the generator. Across four backbones, retaining and labelling versions improved historical information use in state-update tasks with compact fact evidence. Hybrid memory also exceeded session-aware RAG on public multi-session questions at the same evidence budget. Together, the results support managing fact versions throughout storage and answering when information changes across conversations.

## Statements and Declarations

## Funding

This research was supported by Zhejiang Provincial Natural Science Foundation of China under Grant No. LQN26F020016.

## Competing Interests

The authors declare that they have no competing interests.

## Data and Code Availability

The public LoCoMo dataset used in this study is available from its oficial source [24]. Illustrative code for the TEM mechanism and an overview of the main results are available at https://github.com/sungkites/TEM.

## References

[1] Zhang, Z., Dai, Q., Bo, X., et al.: A Survey on the Memory Mechanism of Large Language Model-based Agents. ACM Transactions on Information Systems 43(6), 1–47 (2025) https://doi.org/10.1145/3748302

[2] Hu, Y., Liu, S., Yue, Y., et al.: Memory in the Age of AI Agents. arXiv preprint arXiv:2512.13564 (2025)

[3] Wang, L., Ma, C., Feng, X., et al.: A survey on large language model based autonomous agents. Frontiers of Computer Science 18(6), 186345 (2024) https: //doi.org/10.1007/s11704-024-40231-1

[4] Park, J.S., O’Brien, J., Cai, C.J., et al.: Generative Agents: Interactive Simulacra of Human Behavior. In: Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology, pp. 1–22 (2023). https://doi.org/10.1145/ 3586183.3606763

[5] Yao, S., Zhao, J., Yu, D., et al.: ReAct: Synergizing Reasoning and Acting in Language Models. arXiv preprint arXiv:2210.03629 (2022)

[6] Shinn, N., Cassano, F., Berman, E., et al.: Reflexion: Language Agents with Verbal Reinforcement Learning. arXiv preprint arXiv:2303.11366 (2023)

[7] Packer, C., Wooders, S., Lin, K., et al.: MemGPT: Towards LLMs as Operating Systems. arXiv preprint arXiv:2310.08560 (2023)

[8] Zhong, W., Guo, L., Gao, Q., et al.: MemoryBank: Enhancing Large Language Models with Long-Term Memory. In: Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, pp. 19724–19731 (2024). https://doi.org/10.1609/ aaai.v38i17.29946

[9] Wang, W., Dong, L., Cheng, H., et al.: Augmenting Language Models with Long-Term Memory. In: Advances in Neural Information Processing Systems, vol. 36, pp. 74530–74543 (2023). https://doi.org/10.52202/075280-3259

[10] Xu, W., Liang, Z., Mei, K., et al.: A-MEM: Agentic Memory for LLM Agents. arXiv preprint arXiv:2502.12110 (2025)

[11] Chhikara, P., Khant, D., Aryan, S., et al.: Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory. arXiv preprint arXiv:2504.19413 (2025)

[12] Kang, J., Ji, M., Zhao, Z., et al.: Memory OS of AI Agent. In: Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 25961–25970 (2025). https://doi.org/10.18653/v1/2025.emnlp-main.1318

[13] Rasmussen, P., Paliychuk, P., Beauvais, T., et al.: Zep: A Temporal Knowledge Graph Architecture for Agent Memory. arXiv preprint arXiv:2501.13956 (2025)

[14] Fang, J., Deng, X., Xu, H., et al.: LightMem: Lightweight and Eficient Memory-Augmented Generation. arXiv preprint arXiv:2510.18866 (2025)

[15] Guti´errez, B., Shu, Y., Gu, Y., et al.: HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models. In: Advances in Neural Information Processing Systems, vol. 37, pp. 59532–59569 (2024). https://doi.org/10.52202/ 079017-1902

[16] Guti´errez, B.J., Shu, Y., Qi, W., et al.: From RAG to Memory: Non-Parametric Continual Learning for Large Language Models. arXiv preprint arXiv:2502.14802 (2025)

[17] Edge, D., Trinh, H., Cheng, N., et al.: From Local to Global: A Graph RAG Approach to Query-Focused Summarization. arXiv preprint arXiv:2404.16130 (2024)

[18] Qian, H., Liu, Z., Zhang, P., et al.: MemoRAG: Boosting Long Context Processing with Global Memory-Enhanced Retrieval Augmentation. arXiv preprint arXiv:2409.05591 (2024)

[19] Li, Z., Xi, C., Li, C., et al.: MemOS: A Memory OS for AI System. arXiv preprint arXiv:2507.03724 (2025)

[20] Linders, J., Tomczak, J.M.: Knowledge graph-extended retrieval augmented generation for question answering. Applied Intelligence 55(17), 1102 (2025) https://doi.org/10.1007/s10489-025-06885-5

[21] Jiao, S., Zhu, Z., Wu, W., et al.: An improving reasoning network for complex question answering over temporal knowledge graphs. Applied Intelligence 53(7), 8195–8208 (2023) https://doi.org/10.1007/s10489-022-03913-6

[22] Fang, J., Lu, L., Liu, X.: MGIT: Multi-Granularity implicit temporal framework for knowledge graph question answering. Applied Intelligence 56(12), 374 (2026) https://doi.org/10.1007/s10489-026-07404-w

[23] Wu, D., Wang, H., Yu, W., et al.: LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory. arXiv preprint arXiv:2410.10813 (2024)

[24] Maharana, A., Lee, D.-H., Tulyakov, S., et al.: Evaluating Very Long-Term Conversational Memory of LLM Agents. In: Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 13851–13870 (2024). https://doi.org/10.18653/v1/2024.acl-long.747

[25] Bai, Y., Lv, X., Zhang, J., et al.: LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding. In: Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 3119–3137 (2024). https://doi.org/10.18653/v1/2024.acl-long.172

[26] Liu, N.F., Lin, K., Hewitt, J., et al.: Lost in the Middle: How Language Models Use Long Contexts. Transactions of the Association for Computational Linguistics 12, 157–173 (2024) https://doi.org/10.1162/tacl a 00638

[27] Lewis, P., Perez, E., Piktus, A., et al.: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. arXiv preprint arXiv:2005.11401 (2020)

[28] Karpukhin, V., Oguz, B., Min, S., et al.: Dense Passage Retrieval for Open-Domain Question Answering. In: Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 6769–6781 (2020). https: //doi.org/10.18653/v1/2020.emnlp-main.550

[29] Asai, A., Wu, Z., Wang, Y., et al.: Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. arXiv preprint arXiv:2310.11511 (2023)

[30] Sarthi, P., Abdullah, S., Tuli, A., et al.: RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval. arXiv preprint arXiv:2401.18059 (2024)

[31] Trivedi, H., Balasubramanian, N., Khot, T., et al.: Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions. In: Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 10014–10037 (2023). https://doi.org/10.

[32] Jiang, H., Wu, Q., Lin, C.-Y., et al.: LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models. In: Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 13358– 13376 (2023). https://doi.org/10.18653/v1/2023.emnlp-main.825

[33] Jiang, H., Wu, Q., Luo, X., et al.: LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression. In: Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1658–1677 (2024). https://doi.org/10.18653/v1/2024.acl-long. 91

[34] Xu, F., Shi, W., Choi, E.: RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation. arXiv preprint arXiv:2310.04408 (2023)