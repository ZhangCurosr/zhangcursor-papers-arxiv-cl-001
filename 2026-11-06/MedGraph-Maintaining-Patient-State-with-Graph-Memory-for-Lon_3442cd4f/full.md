# MedGraph: Maintaining Patient State with Graph Memory for Long-Term Medical Dialogue

Donghua Zhang<sup>1†</sup> Zizheng Wang<sup>2\*</sup> Ze Dong<sup>1\*</sup> Yiding Feng<sup>3\*</sup> Siyu Qian<sup>4</sup> <sup>1</sup>University of Michigan <sup>2</sup>Georgetown University <sup>3</sup>Johns Hopkins University <sup>4</sup>Harvard University

## Abstract

LLM-based medical assistants that follow a patient across visits need a memory that keeps the patient’s clinical state correct as it changes. We examine the errors of five memory systems on longitudinal medical dialogue and find three recurring causes: the retrieved evidence gives an outdated state, includes another person’s health facts, or lacks a safety fact the answer depends on. We present MedGraph, a graph memory built for these three cases. Construction extracts dated facts from each session, attributes each fact to the person it describes, and links medication, diagnosis, and measurement records into chains that keep earlier states. Retrieval ranks the patient’s chains as units, reads each chain at the time the question asks about, and adds the original dialogue text; a patient profile lists every recorded safety fact with every question. On MedMemoryBench, Med-Graph has the highest overall accuracy with both a GPT and a Qwen backbone, 10.5 and 10.4 points above the strongest of five baselines, with efficient memory construction and answering in the GPT setting.

## 1 Introduction

LLM-based healthcare agents increasingly support care across multiple visits, including chronic disease management and medication guidance (Wang et al., 2026). They must track the patient’s current medications and reported allergies, since recommending a discontinued medication or overlooking an allergy can lead to unsafe advice. Reliable LLM agents therefore need long-term memory that keeps patient records accurate and up to date across visits.

General-purpose LLM memory systems include retrieval-based methods such as RAG (Lewis et al., 2020), graph-based methods such as GraphRAG (Edge et al., 2024), and agentic approaches such as

Mem0 (Chhikara et al., 2025). These systems differ in how they construct memory, including what they retain and how they organize or update it, and how they retrieve evidence for answering. For example, Mem0 extracts and updates conversational facts, while GraphRAG builds an entity-relation graph and community summaries. RAG retrieves similar passages, whereas GraphRAG uses graphderived summaries. Despite these advances, recent benchmarks report difficulties in reasoning across medical dialogue sessions (Hu et al., 2026; Zhang et al., 2026), and memory accuracy declines as distracting sessions accumulate (Wang et al., 2026).

To examine failures in memory construction and retrieval, we analyze errors from five representative memory systems. This analysis reveals three challenges in longitudinal medical dialogue (Figure 1).

(1) State validity. The same clinical item can have different values or states over time. Some memory systems, such as Mem0, overwrite or discard earlier records when new information arrives (Chhikara et al., 2025). The stored memory may then lack the records needed to answer historical questions. Other systems, such as MemOS, can retain a relevant record without marking it as current in the retrieved evidence. In 24 of the 150 sampled failures, the evidence contained only an outdated state or left the current state unclear. A medical memory must preserve past states and identify which state applies at the relevant time.

(2) Subject attribution. Longitudinal medical dialogue may contain health information about multiple subjects. Without clear attribution, clinical information about a subject other than the patient may be mistaken for the patient’s own. In 108 of the sampled failures, the evidence contained such information. Each clinical fact therefore needs an explicit link to the subject it describes.

(3) Safety prioritization. Safety information can constrain decisions across later visits. For example, a documented allergy may affect a subsequent prescription. In 18 of the sampled failures, we found that the evidence omitted a safety fact needed for the answer. The memory must therefore keep safety-critical information available for later clinical decisions.

![](images/8ff87f9caceff572821a8d3e393e8b2a903f25198942fde6e31875656e21071c.jpg)  
Figure 1: Three challenges in longitudinal medical memory. (a) We inspect 150 wrong answers from five memory systems, 30 per system, and the evidence supplied to the answering model. Each square represents one wrong answer, with colors indicating the observed challenges. Split squares indicate two challenges; gray squares indicate failures outside these categories. Categories overlap, so their counts do not sum to 150. (b) LightMem misses the patient’s September 10 walking-distance measurement, leading to an answer based on an older reading. (c) MemOS retrieves the father’s medication when the patient asks about their own symptoms. (d) REMem omits a recorded co-trimoxazole allergy when answering a later question about antibiotics.

Motivated by these findings, we propose Med-Graph, a multi-layer memory with raw dialogues (L1), a clinical fact graph (L2), and a patient profile (L3). During construction, we extract each session into typed fact nodes that record the subject and event date while preserving values and units. Supersedes edges connect successive records of the same clinical item, including medication changes and repeated measurements. Causal edges link related facts within a session. At retrieval time, we rank the patient’s chains as units, along with facts outside chains. Every answer also receives the patient profile so that important details, including safety constraints, remain available to the answering model.

We evaluate MedGraph on MedMemoryBench against five existing memory systems using two backbones. MedGraph achieves the highest overall accuracy with both backbones, exceeding the strongest baseline by 10.5 percentage points with GPT and 10.4 with Qwen.

Our contributions are as follows:

• We analyze failures in longitudinal medical dialogue and formulate three memory requirements: state validity, subject attribution, and safety prioritization.

• We introduce MedGraph, a multi-layer memory that links subject-attributed clinical facts into dated chains. Its patient profile accompanies every answer.

• We show consistent accuracy gains across two backbones, with efficient memory construction and answering in the GPT setting.

## 2 Related Work

Memory for LLM agents. External memory extends an agent’s context through evidence retrieval (Lewis et al., 2020), memory tiers (Packer et al., 2023), fact updates (Chhikara et al., 2025), and linked notes (Xu et al., 2025). MemOS unifies memory management (Li et al., 2025), MIRIX coordinates specialized components (Wang and Chen, 2025), and LightMem uses compression and offline consolidation to reduce costs (Fang et al., 2026).

Structured and temporal memory. GraphRAG uses community summaries (Edge et al., 2024), HippoRAG 2 retrieves passages and phrases through a graph (Gutiérrez et al., 2025), and Panini organizes memory around entities and events (Rajesh et al., 2026). REMem links episodic gists with time-scoped facts (Shu et al., 2026); Zep/Graphiti tracks validity intervals and invalidates conflicting relations while preserving history (Rasmussen et al., 2025). MedGraph uses type-specific clinical updates to distinguish state changes from repeated observations.

Longitudinal clinical memory. Clinical extraction identifies negation, temporality, and experiencer (Harkema et al., 2009), with benchmarks for medication extraction and temporal relations (Uzuner et al., 2010; Sun et al., 2013). LoCoMo and LongMemEval evaluate memory across sessions (Maharana et al., 2024; Wu et al., 2025); MedMemoryBench (Wang et al., 2026) focuses on longitudinal medical dialogue, including proxy consultations.

Our work brings together persistent memory, temporal state tracking, and clinical information extraction to maintain patient information across visits, with explicit subject attribution and safety facts available regardless of query relevance.

## 3 Problem Setting

We consider a medical assistant that follows one patient across many dialogue sessions. After each session, it updates a memory that it can use to answer questions about the history so far. The dialogue may also discuss a relative’s health or a general medical topic. If the patient describes their father’s medication, that medication must remain associated with the father.

The patient’s clinical state can change between visits. A drug may be started, adjusted, and later stopped, but an earlier dose may still be needed to answer a question about the past. Blood pressure readings also need their dates, since a new reading does not make an earlier measurement incorrect. The event date may differ from the session date when a patient reports a change that happened weeks earlier. The assistant must use the state that applies to the time asked about.

Some information matters long after it was first reported. An allergy mentioned early in the history may affect a later prescription, even if the question does not mention it. The memory must make such safety information available when it is needed for an answer.

Let $s _ { t }$ be the session received at step t, and let $\mathcal { H } _ { t } = ( s _ { 1 } , \ldots , s _ { t } )$ be the history available at that point. The system updates its memory as each session arrives:

$$
M _ { t } = \mathcal { U } ( M _ { t - 1 } , s _ { t } ) .\tag{1}
$$

Memory updates use only sessions observed so far, without access to future sessions or evaluation questions. When a question q arrives, the system retrieves evidence from its current memory, then generates an answer:

$$
E _ { t , q } = \mathcal { R } ( M _ { t } , q ) , \qquad \hat { a } _ { t , q } = \mathcal { G } ( q , E _ { t , q } ) .\tag{2}
$$

Here, U updates memory, R selects and assembles evidence, and G generates the answer.

We seek evidence that concerns the target patient, reflects the time asked about, and includes the safety information needed for the answer. We call these requirements subject attribution, state validity, and safety prioritization, respectively. In controlled comparisons, we hold the answer model G fixed to examine the effects of memory construction and retrieval.

To address these requirements, we introduce MedGraph, our memory system for tracking patient information across visits and selecting evidence to answer questions about the patient’s history.

## 4 MedGraph

MedGraph builds a clinical fact graph from each incoming session (Figure 2). Raw dialogue is retained in L1 as source evidence. L2 links extracted facts to the people they concern and the clinical items they describe, such as medications, diagnoses, or measured quantities. These items are represented as entity nodes. At retrieval time, the patient profile (L3) is rendered from the patient’s L2 records and included with every question.

Construction uses one extraction call per session. Retrieval selects graph records and adds their source text. The answer model, called the reader, receives this evidence and the profile, then answers in one LLM call.

## 4.1 Memory construction (L1 to L2)

Each fact stores its type τ , subject σ, clinical entity e (the item it describes), event date, and source session and turn. Its content is one sentence preserving reported numbers and units, with separate value, unit, and status fields when available. Condition facts describe diagnoses. Medication and condition facts have status labels; safety facts have no status field. Entities and persons have separate nodes, and facts about the same entity are linked through its entity node.

Extraction. Given a session, its date, and a fixed prompt (Appendix A), the extractor returns JSON with the session’s subject, clinical facts, and causal links between those facts. The links become causal edges in L2. Each fact is assigned to the person it describes, not necessarily the speaker. For example, a fact about the father’s medication is linked to the father’s node, even when reported by the patient. General health information stays in L1 only. Relative dates such as “two weeks ago” are calculated from the session date, which is also used when the event date is unknown. Entity names are standardized before matching to existing nodes (Appendix B).

(a) Construction from raw sessions to the fact graph  
![](images/05b852fc6129adbed9e119e5c74a3292dab6b465546e75227aac1c4b008cc592.jpg)  
Figure 2: MedGraph overview. (a) Construction: clinical facts are extracted from each session and linked to the person they describe in the fact graph (L2). Medication, condition, lab, and vital facts form chains by entity, subject, and type, following insertion order, with the patient’s and other people’s facts kept separate. (b) Retrieval: the system ranks the patient’s records, treating each chain as one unit and each fact outside a chain as an individual unit. For selected chains, it includes the matched fact, the state at the requested time, and the preceding record. It also adds causally linked facts, records from any explicitly requested date window, and original dialogue excerpts. The answer model receives the question, the assembled evidence, and the patient profile (L3), then generates an answer in one LLM call.

Chains and repeated mentions. Each chain groups facts with the same entity, subject, and type, using the key $( e , \sigma , \tau )$ . The patient and another person thus have separate chains for the same drug. Medication, condition, lab, and vital facts form chains; other types remain individual records (Table 1).

A new fact counts as a repeated mention if its status matches the chain tail and, after dates are removed, every Arabic numeral in its content already appears in the tail’s content. It updates the existing node’s last-mentioned date and mention count without creating a new node. The new wording remains in L1.

Chain updates. A fact that fails the repetition check is appended to its chain, with a supersession edge pointing to the previous tail. Chains follow insertion order, not event date, so the tail is the latest inserted record.

For medications and conditions, the previous record’s validity interval ends at the new record’s event date, or the session date if the event date is unavailable. Planned and active medication records share a chain, with plans labeled planned.

<table><tr><td>Fact type</td><td>Storage</td><td>Current record</td></tr><tr><td>medication, condition chain, insertion order chain tail</td><td></td><td></td></tr><tr><td>lab, vital</td><td>chain, all kept</td><td>latest by date</td></tr><tr><td>symptom, event</td><td>individual records</td><td>一</td></tr><tr><td>safety</td><td>append-only</td><td>all in profile</td></tr><tr><td>preference</td><td>individual records</td><td>一</td></tr></table>

Table 1: Storage and update policy by fact type.

Lab and vital records are retained as historical <sup>fi</sup>measurements, with the latest selected by event date rather than chain position. When a change includes a previous value, such as from 10 mg to 20 mg, that value is kept in the new fact’s content.

Indexing. We index each fact’s entity name, content, and available status using BM25 (Robertson and Zaragoza, 2009) and a dense index, retaining its subject label. L1 turn pairs are also indexed.

## 4.2 Patient profile (L3)

The profile is rendered from the patient’s L2 records at retrieval time. It includes every safety fact without truncation or retraction: allergies, contraindications, drug interactions, and surgical or bleeding history.

Diagnosis- and medication-chain tails and key preferences share a fixed budget. Stopped medications are listed separately from other medication tails, including plans. The profile begins with the latest available session date as the memory cutoff and is included with every question without a relevance check.

## 4.3 Retrieval

Every question uses the same retrieval pipeline. The reader receives the full question. Section 6.1 compares this pipeline with simpler retrieval controls.

Subject filtering and ranking. We rank only the patient’s facts using BM25 and dense similarity. Other people’s facts remain stored under their own person nodes. Each chain counts as one retrieval unit, while facts outside chains count individually. Reciprocal rank fusion (Cormack et al., 2009) combines the rankings, and the top eight units are retained.

Chain context. Each selected chain supplies up to three facts: the fact that triggered retrieval (Matched), the record that applies to the time asked about (Anchor), and the record the anchor supersedes (Relation). For medication and diagnosis chains, the anchor’s stored validity interval must cover the explicit question date; if no date is given, the chain tail is used. Lab and vital anchors are the latest measurements by event date up to the requested time.

Expansion. We add up to four causal neighbors of the matched facts. An explicit date containing a year activates the date channel, which adds patient records from that date window, along with medication and diagnosis records whose stored validity intervals cover the window’s start.

Rendering and answering. Each fact appears on one line with its date and available status. We mark superseded records and chain tails. For labs and vitals, the latest-measurement label is assigned using event dates. We include verbatim source windows for the top three matched facts. The profile and retrieved evidence fill the memory slot of the fixed answer prompt, and the reader answers in one LLM call. Appendix C gives the exact budgets and retrieval limits.

## 5 Experiments

## 5.1 Setup

Benchmark. We evaluate MedGraph on 967 questions from MedMemoryBench (Wang et al., 2026). Each history mixes about 100 patient sessions with 200 distractor sessions about general health or other people’s care. Sessions arrive in order; questions follow every ten patient sessions and concern the history available so far. The six types are entity exact match (EEM), multiple choice (MQ), temporal location accuracy (TLA), state update accuracy (SUA), inference generation (IG), and multi-hop clinical deduction (MCD). We report per-type accuracy and their unweighted average (Avg.).

Baselines. We use five systems from MedMemoryBench with their default implementations and settings. Embedding RAG retrieves the five most similar dialogue chunks, each up to 4,096 tokens. Mem0 (Chhikara et al., 2025) extracts short memories and adds, updates, or deletes them; MemOS (Li et al., 2025) stores extracted memories in a vector index; and LightMem (Fang et al., 2026) compresses topic segments into time-stamped entries. REMem (Shu et al., 2026) stores time-stamped gists and fact triples in a graph searched by an agent over several steps.

In the GPT setting, gpt-6-luna builds LLMbased memories and gpt-5.1 answers questions. In the Qwen setting, Qwen3-30B-A3B-Instruct-2507 (Yang et al., 2025) does both. Embedding RAG uses no construction LLM. All methods share the answer prompt, judge, and embedder (Appendix D).

## 5.2 Results

MedGraph leads overall, especially on TLA and IG. MedGraph has the highest Avg. and leads on most question types, with the largest gains over the best baseline on TLA and IG (Table 2). TLA requires evidence from specific times; IG depends on the patient’s conditions and medications. These gains come from dated facts, the date channel, and the profile (Section 6.1).

<table><tr><td>b</td><td colspan="2">EEM n = 54</td><td colspan="2">TLA n = 51</td><td colspan="2">All n = 105</td></tr><tr><td></td><td>when asked</td><td>after all sessions</td><td>when asked</td><td>after all sessions</td><td>when asked</td><td>after all sessions</td></tr><tr><td>Embedding RAG</td><td>59.3</td><td>55.6</td><td>43.1</td><td>23.5</td><td>51.4</td><td>40.0</td></tr><tr><td>Mem0</td><td>48.1</td><td>48.1</td><td>39.2</td><td>17.6</td><td>43.8</td><td>33.3</td></tr><tr><td>MemOS</td><td>51.9</td><td>48.1</td><td>45.1</td><td>25.5</td><td>48.6</td><td>37.1</td></tr><tr><td>LightMem</td><td>37.0</td><td>44.4</td><td>47.1</td><td>35.3</td><td>41.9</td><td>40.0</td></tr><tr><td>REMem ×</td><td>27.8</td><td>31.5</td><td>19.6</td><td>7.8</td><td>23.8</td><td>20.0</td></tr><tr><td>MedGraph </td><td>59.3</td><td>59.3</td><td>68.6</td><td>56.9</td><td>63.8</td><td>58.1</td></tr></table>

<table><tr><td>Avg. EEM MQ TLA SUA IG MCD</td></tr><tr><td>GPT backbone</td></tr><tr><td>Emb. RAG 41.4 50.5 69.8 46.3</td></tr><tr><td>Mem0 40.3 47.5 65.3 36.2</td></tr><tr><td>MemOS 39.1 46.061.8 50.3 61.0 LightMem 37.1 40.0 68.3 44.6 53.0</td></tr><tr><td>REMem 23.0 27.5 53.8 23.2 28.0</td></tr><tr><td>3.5 MedGraph 51.9 56.5 72.4 68.4 61.0 37.9</td></tr><tr><td>15.1 BM25 only 49.8 54.0 73.9 65.0 63.0 33.3</td></tr><tr><td>9.7 built facts, BM25 43.6 51.5 69.3 54.2 61.0 19.2 6.5 raw text, BM25 33.0 35.0 69.8 27.7 37.0 15.7 12.9</td></tr><tr><td>Qwen backbone</td></tr><tr><td>Emb. RAG 35.7 43.048.7 41.2 51.0 18.2 11.8</td></tr><tr><td>Mem0 26.4 35.0 47.2 18.6 51.0 4.5 2.2</td></tr><tr><td>MemOS 33.3 46.0 55.3 37.3 47.0 10.1 4.3</td></tr><tr><td>LightMem 17.6 22.0 42.7 14.7 24.0 1.0</td></tr><tr><td>1.1 REMem 16.3 14.5 45.7 18.6 19.0 0.0 0.0</td></tr><tr><td>MedGraph 46.1 58.5 52.8 68.9 69.0 19.7 7.5</td></tr><tr><td>BM25 only 46.2 58.5 53.3 69.5 70.0 17.2</td></tr><tr><td>8.6</td></tr><tr><td>built facts, BM2540.5 59.0 51.8 54.8 62.0 9.1 6.5 raw text, BM25 24.5 24.046.2 24.935.010.6 6.5</td></tr></table>

Table 2: Accuracy (%) with GPT (gpt-6-luna builds memory, gpt-5.1 answers) and Qwen (Qwen3-30B-A3B-Instruct-2507 does both); claude-sonnet-4-5 judges. Avg. is the mean across six types. Dark/light blue: best/second best among six methods; best scores are also bold. Indented BM25 controls (Section 6.1): BM25 only changes only MedGraph’s ranking; built facts, BM25 retrieves extracted facts without chain retrieval, the profile, or source windows; raw text, BM25 builds no memory and retrieves whole sessions with the benchmark’s BM25 implementation.

MedGraph also leads with an open model. With Qwen, MedGraph again has the highest Avg. (Table 2). All methods lose accuracy; MedGraph’s drop is similar to Embedding RAG’s, which builds memory without an LLM, and much smaller than Mem0’s and LightMem’s.

MedGraph leads at every checkpoint. Med-Graph has the highest Avg. at all ten checkpoints (Figure 3a). To separate history growth from changes in questions, we repeat the same early questions after the full history (Figure 3b). Embedding RAG, Mem0, and MemOS lose accuracy significantly; MedGraph’s drop is not significant, and it remains the most accurate (Appendix E).

## 6 Analysis

## 6.1 Memory construction and retrieval

Most sampled baseline failures come from missing or badly stored facts. We check sampled wrong answers of each system against its stored memory and retrieved evidence (Appendix E). In about two thirds of the baseline cases, the needed fact is not stored, stored incompletely, stored in a hard-to-find form, or stored with no owner (Figure 4). In most of MedGraph’s sampled failures, the needed fact is stored intact.

![](images/ba20044f6961ed8a0f47fef2614a021f2d2ca98295e946ec4129f5c29c852d44.jpg)

Figure 3: Accuracy as histories grow (GPT). (a) Avg. after every ten patient sessions, using new questions. (b) The same 105 early EEM and TLA questions at their original checkpoints and after the full history, with unchanged reference answers. LightMem uses one rebuilt memory for both measurements (Appendix E).  
![](images/3625e78764d25d435e7c0d80848601ec0f776c13ed0e966b789a1e33abe2a86e.jpg)  
Figure 4: Storage and indexing issues in sampled failures. Percentages use 31, 31, 32, 31, 38, and 25 retained cases in the plotted order. Numbers inside bars are counts. LightMem uses a rebuilt memory (Appendix E).

Fact construction accounts for more than half of the gain over raw text. To separate construction from retrieval, we rank raw sessions and Med-Graph’s extracted facts with BM25 alone (Table 2). With both backbones, these facts alone give more than half of the full method’s gain over raw sessions, and they score higher than raw sessions for every patient. Each fact is a short statement about one person and carries its date, so BM25 can match the record a question needs, while a raw session is long and covers many topics. Replacing MedGraph’s fused BM25 and dense ranking with BM25 alone, while keeping the rest of the retrieval, changes accuracy little, so the scoring function accounts for little of the lead.

<table><tr><td></td><td>State</td><td colspan="2">Subject</td><td colspan="2">Safety</td></tr><tr><td></td><td>Avg. Acc.</td><td>entries</td><td>Polluted Polluted questions Missing Acc.</td><td></td><td></td></tr><tr><td>Emb. RAG</td><td>41.4 49.0</td><td>12.2</td><td>22.8</td><td>38.2</td><td>44.0</td></tr><tr><td>Mem0</td><td>40.3 55.4</td><td>22.4</td><td>55.6</td><td>51.7</td><td>44.9</td></tr><tr><td>MemOS</td><td>39.1 53.4</td><td>17.1</td><td>40.4</td><td>59.6</td><td>39.7</td></tr><tr><td>LightMem</td><td>37.1 52.9</td><td>21.8</td><td>74.1</td><td>74.2</td><td>40.8</td></tr><tr><td>REMem</td><td>23.035.8</td><td>27.3</td><td>63.6</td><td>80.9</td><td>25.6</td></tr><tr><td>MedGraph</td><td>51.9 60.3</td><td>0.0</td><td>0.1</td><td>1.1</td><td>60.4</td></tr><tr><td>— subject filter 49.6 57.8 – profile</td><td></td><td>15.9</td><td>57.8</td><td>1.1</td><td>57.1</td></tr></table>

Table 3: The three challenges (%, GPT). State: accuracy on 204 changed-state questions (Appendix D). Subject: percentages of evidence entries about other people and questions with such evidence (lower is better; Appendix D). Safety: the missing-fact rate on 89 questions with identified required safety facts, and mean MQ and IG accuracy on 180 safety-dependent questions. Avg. follows Table 2. Bold: best among six systems. Bottom rows omit one component.

Retrieval helps most on TLA and IG. Compared with the facts ranked by BM25, the full retrieval (chain retrieval, the date channel, the profile, and source windows) gains most on IG and TLA with GPT (Table 2). TLA questions ask when an event was recorded or what was recorded on a given date. When a question names a date, the date channel adds the records of that day, and each chain is read at that date. IG questions depend on the patient’s own diagnoses, drugs, and allergies, which the profile gives with every question (Table 4).

## 6.2 The three challenges

State validity: MedGraph is the most accurate on changed-state questions. MedGraph answers questions about changed states more accurately than every baseline (Table 3); these are the 100 SUA questions and the 104 MQ and IG questions whose benchmark trap is a changed state or dose (Appendix D). Most of this gain comes from construction: ranked by BM25 alone, the extracted facts answer these questions significantly more often than raw sessions (Appendix D). Each record has a status and a date, so the reader can tell which state is current. In Table 4, LightMem retrieves the earlier ibuprofen use and the order to stop it as two unmarked entries.

Subject attribution: MedGraph keeps other people’s facts out of the evidence. The baselines often retrieve other people’s health facts, while Med-

![](images/bd61e23dfc9e47e501b5e98cc79756a87c2064717225280ea46d4c868bd99f51.jpg)  
Figure 5: Accuracy against cost (GPT backbone): (a, b) LLM calls and input tokens spent constructing memory; (c) seconds per question, the wall-clock time from the question to the answer string, including the method’s retrieval and the reader’s generation and excluding judging. The top right is best; dashed lines mark the Pareto frontier.

Graph almost never does (Table 3). Embedding RAG returns chunks of consultations about other people, and the textual memories have no subject label to filter on. During construction, MedGraph assigns all but 9 of the 11,039 facts from consultations about other people to someone other than the patient, and the subject filter keeps them out of the evidence. Without the filter, other people’s facts appear in the evidence of more than half of the questions, and accuracy drops. In Table 4, REMem retrieves another person’s records.

Safety prioritization: the profile supplies the safety facts. On questions whose answer depends on an allergy, a contraindication, or a drug interaction, the baselines often miss the needed safety fact, while MedGraph’s evidence almost always contains it (Table 3). The profile lists every safety fact with each question, so the reader sees it even when the question does not mention it. Without the profile, the fact is often missing, and accuracy drops. In Table 4, MemOS stores the ibuprofen warning but does not retrieve it.

## 6.3 Cost and efficiency

MedGraph is the only LLM-built memory on the accuracy–cost frontier. MedGraph makes one LLM call per session to build its memory, about as many as MemOS and half as many as Mem0 and REMem (Figure 5). It also has the shortest mean answering time: one retrieval pass and one LLM call per question, while REMem first searches with several LLM calls.

<table><tr><td colspan="3">Question (patient 4, IG): “Doctor, my knee pain is back. Can I take a painkiller again?&quot; Gold: Avoid ibuprofen, which the doctor stopped; with perindopril and spironolactone it risks kidney damage and high potassium.</td></tr><tr><td>Evidence</td><td></td><td>Answer</td></tr><tr><td>LightMem X</td><td>“Took ibuprofen 200 mg as needed for about a year&quot; (Jan 3) and “the doctor told the patient to stop ibuprofen&quot; (Jan 5), with no mark of which is current; none names perindopril or spironolactone.</td><td>“If the pain is very bad, a one-off 200 mg of ibuprofen is acceptable.&quot;</td></tr><tr><td>MemOS X</td><td>Knee-pain and exercise memories. Its memory holds the order to stop ibuprofen and the interaction warning, but neither is retrieved.</td><td>No drug record; an old painkiller “can be considered&quot;.</td></tr><tr><td>REMem X</td><td>General advice and another person&#x27;s records, e.g., “continue mecobalamin to support nerve recovery&quot;. Its stored interaction warning is not retrieved.</td><td>Asks whether the user is “Xiao Zhou, who used celecoxib&quot;.</td></tr><tr><td>MedGraph √</td><td>Current medications: perindopril, spironolactone, and valsartan, all “in use&quot; with dates. Safety: “[interaction] ibuprofen: with perindopril and spironolactone, adds kidney and potassium risk; do not keep taking  $\mathrm { i t } ^ { \dag }$  (2024-01-03).</td><td>&quot;With perindopril, valsartan, and spironolactone, do not take ibuprofen on your own.&quot;</td></tr></table>

Table 4: A question that needs the current state and a safety fact (translated and shortened). Red: misleading; blue: what the correct answer needs. More examples are in Appendix F.

## 7 Conclusion

We presented MedGraph, a graph memory that tracks clinical state changes, attributes facts to their subjects, and includes safety facts with every question. On MedMemoryBench, it leads with both GPT and Qwen by 10.5 and 10.4 points over the strongest baseline. Compared with retrieval over raw sessions, more than half of the gain comes from fact construction and the rest from retrieval. In the GPT setting, MedGraph builds memory with one extraction call per session, uses about half the construction input tokens of Mem0 and REMem, and has the shortest mean answering time. Future work could link causes across visits to support multi-hop questions.

## Limitations

We evaluate MedGraph on one Chinese benchmark of longitudinal medical dialogue; evaluating it in other languages and clinical settings is left to future work. MedGraph relies on an LLM to extract facts from each session, so the quality of its memory depends on the extractor. Causal links are drawn within a session only. Chains are ordered by when a fact is recorded, not by when the event happened, so a late report of an earlier event or a planned change becomes the chain tail. Safety facts are never removed, so an allergy that is later ruled out, or a warning about a drug the patient has since stopped, stays in the profile. Following the benchmark’s protocol, four question types are scored by an LLM judge.

## Ethics Statement

MedGraph is a research prototype and is not validated for clinical use. A memory that drops a current medication, keeps a stopped one, or attributes a relative’s diagnosis to the patient can lead to harmful advice; any deployment would need clinical review of what the memory shows. All experiments use the synthetic patients and dialogues of MedMemoryBench; no real patient data was used.

## Use of AI Assistants

AI assistants were used for language polishing and grammar correction. All text, code, and results were reviewed by the authors.

## References

Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. 2025. Mem0: Building production-ready AI agents with scalable long-term memory. arXiv preprint arXiv:2504.19413.

Gordon V. Cormack, Charles L. A. Clarke, and Stefan Büttcher. 2009. Reciprocal rank fusion outperforms Condorcet and individual rank learning methods. In Proceedings of the 32nd International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 758–759.

Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan Larson. 2024. From local to global: A Graph RAG approach to query-focused summarization. arXiv preprint arXiv:2404.16130.

Jizhan Fang, Xinle Deng, Haoming Xu, Ziyan Jiang, Yuqi Tang, Ziwen Xu, Shumin Deng, Yunzhi Yao, Mengru Wang, Shuofei Qiao, Huajun Chen, and

Ningyu Zhang. 2026. LightMem: Lightweight and efficient memory-augmented generation. In International Conference on Learning Representations (ICLR).

Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi, Sizhe Zhou, and Yu Su. 2025. From RAG to memory: Non-parametric continual learning for large language models. In Proceedings of the 42nd International Conference on Machine Learning (ICML).

Henk Harkema, John N. Dowling, Tyler Thornblade, and Wendy W. Chapman. 2009. ConText: An algorithm for determining negation, experiencer, and temporal status from clinical reports. Journal ofBiomedical Informatics, 42(5):839–851.

Hebin Hu, Renke Dai, Ah-Hwee Tan, and Yilin Kang. 2026. Synthesis and evaluation of long-term historyaware medical dialogue. In Proceedings ofthe 25th International Conference on Autonomous Agents and Multiagent Systems (AAMAS), pages 1710–1718.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-augmented generation for knowledgeintensive NLP tasks. In Advances in Neural Information Processing Systems (NeurIPS).

Zhiyu Li, Chenyang Xi, Chunyu Li, Ding Chen, Boyu Chen, Shichao Song, Simin Niu, Hanyu Wang, Jiawei Yang, Chen Tang, et al. 2025. MemOS: A memory OS for AI system. arXiv preprint arXiv:2507.03724.

Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang. 2024. Evaluating very long-term conversational memory of LLM agents. In Proceedings ofthe 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers).

Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, and Joseph E. Gonzalez. 2023. MemGPT: Towards LLMs as operating systems. arXiv preprint arXiv:2310.08560.

Shreyas Rajesh, Pavan S Holur, Mehmet Yigit Turali, Chenda Duan, and Vwani Roychowdhury. 2026. Panini: Continual learning in token space via structured memory. In Proceedings of the 43rd International Conference on Machine Learning (ICML).

Preston Rasmussen, Pavlo Paliychuk, Travis Beauvais, Jack Ryan, and Daniel Chalef. 2025. Zep: A temporal knowledge graph architecture for agent memory. arXiv preprint arXiv:2501.13956.

Stephen Robertson and Hugo Zaragoza. 2009. The probabilistic relevance framework: BM25 and beyond. Foundations and Trends in Information Retrieval, 3(4):333–389.

Yiheng Shu, Saisri Padmaja Jonnalagedda, Xiang Gao, Bernal Jiménez Gutiérrez, Weijian Qi, Kamalika Das, Huan Sun, and Yu Su. 2026. REMem: Reasoning with episodic memory in language agents. In International Conference on Learning Representations (ICLR).

Weiyi Sun, Anna Rumshisky, and Özlem Uzuner. 2013. Evaluating temporal relations in clinical text: 2012 i2b2 challenge. Journal of the American Medical Informatics Association, 20(5):806–813.

Özlem Uzuner, Imre Solti, and Eithon Cadag. 2010. Extracting medication information from clinical text. Journal ofthe American Medical Informatics Association, 17(5):514–518.

Yihao Wang, Haoran Xu, Renjie Gu, Yixuan Ye, Xinyi Chen, Xinyu Mu, Yuan Gao, Chunxiao Guo, Peng Wei, Jinjie Gu, Huan Li, Ke Chen, and Lidan Shou. 2026. MedMemoryBench: Benchmarking agent memory in personalized healthcare. In Advances in Neural Information Processing Systems (NeurIPS).

Yu Wang and Xi Chen. 2025. MIRIX: Multi-agent memory system for LLM-based agents. arXiv preprint arXiv:2507.07957.

Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, and Dong Yu. 2025. LongMemEval: Benchmarking chat assistants on long-term interactive memory. In International Conference on Learning Representations (ICLR).

Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muennighoff, Defu Lian, and Jian-Yun Nie. 2024. C-Pack: Packed resources for general Chinese embeddings. In Proceedings of the 47th International ACM SI-GIR Conference on Research and Development in Information Retrieval, pages 641–649.

Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang. 2025. A-Mem: Agentic memory for LLM agents. In Advances in Neural Information Processing Systems (NeurIPS).

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41 others. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Zeyu Zhang, Ziqing Wang, and Kaize Ding. 2026. Med-LoCoMo: A long-context multi-session medical dialogue benchmark for large language models. arXiv preprint arXiv:2607.22566.

## A Extraction prompt

The experiments use a Chinese prompt; its English translation is given first, followed by the original. The examples for subject attribution use placeholder names (disease A, drug B, metric C).

## Extraction prompt (English translation)

You are a medical dialogue information extractor.   
,→ Extract structured clinical facts from the   
,→ doctor-patient dialogue below and output strict   
,→ JSON.   
[Dialogue date] {date}   
[Dialogue]   
{dialogue}

0. [Decide first, then extract] Whose health is this   
,→ dialogue about? Check whom each piece of health   
,→ information in the patient's words is attached to:   
,→ "my mother / my father / my son / my daughter / my   
,→ spouse / my friend / she / he" + symptom /   
,→ medication / test / measurement -> the owner of   
,→ these health facts is THAT person, not the   
,→ speaker. (!) The speaker tag "Patient" does not   
,→ make the patient the subject of a fact: in a   
,→ family health consultation the patient only relays   
,→ the information, and the subject is   
,→ family:<relation>.

```csv
2. type (one per fact):
- medication_status: start / stop / dose change /
,→ planned / self-medication; status required: active
,→ | stopped | dose_changed | planned
- condition: diagnosis status; status required:
,→ suspected | confirmed | misdiagnosed | resolved |
,→ chronic | worsened | improved
- lab_result: lab, imaging, or antibody result; keep
,→ the exact value and unit in value
- vital: weight, blood pressure, home glucose, etc.;
,→ keep the exact value
- symptom: onset / worsening / relief
- safety: ★ safety-critical information, always
,→ extracted as its own fact: allergy,
,→ contraindication or interaction, prior surgery or
,→ bleeding (both surgical_history); safety_kind
,→ required: allergy | contraindication | interaction
,→ | surgical_history. Better too many than one
,→ missed.
- preference: lifestyle, schedule, diet, cost and
,→ care preferences (cost concerns, reluctance to
,→ take leave, adherence)
- event: other meaningful clinical or life events
```

```jsonl
[Output format] Output JSON only, no other text:
{"session_subject":
,→ "patient|family:<relation>|mixed|none",
"facts": [
{"type": "medication_status", "subject": "patient",
"entity": "empagliflozin",
"content": "Stopped empagliflozin as advised; had
,→ been taking it on their own for about 8 months",
"value": null, "unit": null, "status": "stopped",
"date": "2024-01-15", "is_state_change": true,
"replaces": "empagliflozin being taken",
,→ "safety_kind": null,
"causes": [1], "evidence_turn": 3, "confidence":
,→ 0.95},
{"type": "safety", "subject": "patient", "entity":
,→ "empagliflozin",
"content": "Empagliflozin with insulin markedly
,→ raises the risk of euglycemic ketoacidosis;
,→ contraindicated",
"value": null, "unit": null, "status": null,
"date": "2024-01-15", "is_state_change": false,
,→ "replaces": null,
"safety_kind": "contraindication",
"causes": [], "evidence_turn": 3, "confidence":
,→ 0.95}]}
```

## Extraction prompt (Chinese original)

## 你是一个医疗对话信息抽取器。请从下面这段医患对话中抽取结构化临床事实,输出严格的JSON。

## 【对话日期】<sub>{</sub>d<sub>ate}</sub>【对话内容】

## 【抽取要求】 【抽取要求】

<sub>0.</sub> 【第一步<sub>,</sub>先判断再抽取】这段对话在谈论【谁的健康】?  
逐条检查患者话里的健康信息挂在谁身上<sub>:</sub>出现"我妈/我爸/我  
儿子/我女儿/我爱人/我朋友/她/他"  
+症状/用药/检查/指标→ 这些健康事实的主人是【那个  
人】,不是说话人。  
(!) 说话人标签是"Patient"不代表事实主体是patient:  
家属健康咨询里,患者只是【转述者】,事实主体=  
f<sub>am</sub>il<sub>y:</sub>关系。  
<sub>1.</sub> 每条事实单独标注<sub>su</sub>bj<sub>ect</sub>(主体)。判据<sub>:</sub>【症状长在谁身上、  
药吃进谁嘴里、指标是谁的身体】<sub>:</sub>  
- "patient" = 该健康事实属于患者本人  
- "family:关系" = 该健康事实属于家人(如"family:父亲")  
· "我妈妈得了A病好几年,一直在用B药" → 病和用药都是  
妈妈的→ family:母亲  
· "丈夫住院让我压力大,我自己的C指标也变差了" → C指  
标是患者自己的身体→ patient  
(丈夫只是诱因<sub>,</sub>他本人没有健康事实被陈述)  
· 该判据与【谁说的】无关:医生说"你妻子确诊了A病" →  
病是妻子的→ family:妻子  
- "none" = 与任何具体个人无关的通用健康科普知识  
【微例】对话<sub>:</sub> "<sub>Pat</sub>i<sub>ent:</sub> 医生<sub>,</sub>我儿子<sub>15</sub>岁<sub>,</sub>得了<sub>A</sub>病<sub>,</sub>在  
用<sub>B</sub>药<sub>,</sub>他说症状还没好转。"  
正确: session\_subject="family:儿子"; 两条事实  
的<sub>su</sub>bj<sub>ect</sub> 都是"f<sub>am</sub>il<sub>y:</sub>儿子"  
(<sub>me</sub>di<sub>cat</sub>i<sub>on\_status B</sub>药<sub>act</sub>i<sub>ve; symptom</sub> 症状未好转)  
错误示范: subject 写成"patient" ← 说话人是患者,但病  
和用药都是儿子的。

2. type 取值(每条选一个):  
<sub>- me</sub>di<sub>cat</sub>i<sub>on\_status:</sub> 用药(开始/停用/调量/计划用/自行服  
用)<sub>,status</sub> 必填<sub>:</sub>  
active|stopped|dose\_changed|planned  
<sub>-</sub> <sub>con</sub>di<sub>t</sub>i<sub>on:</sub> 疾病诊断状态<sub>,status</sub> 必填<sub>:</sub> <sub>suspecte</sub>d|  
confirmed|misdiagnosed|resolved|chronic|worsened|improved  
<sub>-</sub> l<sub>a</sub>b<sub>\_resu</sub>l<sub>t:</sub> 检查结果(化验/影像/抗体)<sub>,va</sub>l<sub>ue</sub> 尽量保留  
原文精确数值和单位  
- vital: 生理指标(体重/血压/自测血糖等),value 保留精确  
数值  
<sub>- symptom:</sub> 症状(出现/加重/缓解)  
<sub>- sa</sub>f<sub>ety:</sub> ★安全关键信息<sub>,</sub>必须单独抽出<sub>:</sub>过敏  
史(<sub>a</sub>ll<sub>ergy</sub>)、用药禁忌/相互作用  
(<sub>contra</sub>i<sub>n</sub>di<sub>cat</sub>i<sub>on</sub>/i<sub>nteract</sub>i<sub>on</sub>)、既往手术或出血  
史(surgical\_history),  
safety\_kind 必填。宁多勿漏!  
<sub>-</sub> <sub>pre</sub>f<sub>erence:</sub> 生活方式/作息/饮食/经济与就医偏好(费用顾  
虑/怕请假/依从性)  
- event: 其他有意义的临床或生活事件  
<sub>3. ent</sub>i<sub>ty:</sub> 该事实的核心实体名(药名/疾病名/检查名/指标  
名)<sub>,</sub>没有则<sub>nu</sub>ll。  
药名用规范通用名(如"恩格列净")<sub>,</sub>不要带剂量。  
4. date: 事实发生日期YYYY-MM-DD;对话里说"两周前/上个月"要  
以对话日期为锚换算;  
长期既往史给年份如"2019-01-01";实在无法确定给null(默  
认=对话日期)。  
5. is\_state\_change: 该事实是否改变了此前已知状态(如停  
药/换药/诊断修订/症状缓解)。  
若是<sub>,rep</sub>l<sub>aces</sub> 写被取代的旧状态简述(如"恩格列净正在服  
用"),否则null。  
6. causes: 本次输出列表中,导致该事实的其他事实的下标数  
组(从0计,没有给[])。  
7. content: 一句话完整描述该事实,保留关键数值原  
文。confidence: 0\~1。  
8. evidence\_turn: 该事实主要出自对话的第几轮(从1计)。  
【输出格式】只输出<sub>JSON,</sub>不要任何其他文字<sub>:</sub>  
<sub>{</sub>"<sub>sess</sub>i<sub>on\_su</sub>bj<sub>ect</sub>"<sub>:</sub> "<sub>pat</sub>i<sub>ent</sub>|f<sub>am</sub>il<sub>y:</sub>关系|<sub>m</sub>i<sub>xe</sub>d|<sub>none</sub>"<sub>,</sub>  
"facts": [  
{"type": "medication\_status", "subject": "patient",  
"<sub>ent</sub>i<sub>ty</sub>"<sub>:</sub> "恩格列净"<sub>,</sub>  
"<sub>content</sub>"<sub>:</sub> "按医嘱停用恩格列净<sub>,</sub>此前自行连续服用  
约8个月",  
"value": null, "unit": null, "status": "stopped",  
"date": "2024-01-15", "is\_state\_change": true,  
"<sub>rep</sub>l<sub>aces</sub>"<sub>:</sub> "恩格列净正在服用"<sub>,</sub> "<sub>sa</sub>f<sub>ety\_</sub>ki<sub>n</sub>d"<sub>:</sub>  
null,  
"causes": [1], "evidence\_turn": 3, "confidence":  
0.95},  
{"type": "safety", "subject": "patient", "entity":  
"恩格列净"<sub>,</sub>  
"content": "恩格列净与胰岛素联用会显著增加正常血糖  
酮症酸中毒风险<sub>,</sub>医嘱禁用"<sub>,</sub>  
"value": null, "unit": null, "status": null,  
"date": "2024-01-15", "is\_state\_change": false,  
"replaces": null,  
"safety\_kind": "contraindication", "causes": [],  
"evidence\_turn": 3,  
"confidence": 0.95}  
]}

## B Entity resolution details

The surface name is first canonicalized through a seed dictionary of generic drug names and drug classes, then matched against existing entities of the same type: an exact match on the normalized name or a stored alias reuses the node; otherwise the node with the highest character-bigram overlap is reused only if the overlap is at least 0.75; otherwise a new entity is created. Class-level entries in the seed dictionary (a drug class listed with its members) merge members into the class.

## C Retrieval settings

Each ranked list (dense and BM25) is scanned until it yields 32 distinct units or 256 facts, and the top 8 units after reciprocal rank fusion are kept. Causal expansion adds at most 4 facts. Date-anchored retrieval adds at most 12 records in the window and at most 4 states in force at its start. The profile budget is about 1,600 characters, of which the safety part is never trimmed. Source windows are shown for the top 3 matched facts, and if fewer than 2 facts are retrieved, the top 4 raw turn pairs are added. The same settings are used for every question and every patient.

## D Baselines and evaluation details

Protocol. We use the first ten patients of Med-MemoryBench by ID and all 967 of their questions in the Chinese mixed setting. All methods run in the benchmark’s harness with the same reader (gpt-5.1, temperature 0.3), the same answer prompt, the same judge (claude-sonnet-4-5), and the same local embedder (bge-small-zh-v1.5; Xiao et al., 2024). Every LLM call that builds memory, by our extractor and by the baselines’ memory builders, uses gpt-6-luna. Memory is built online as sessions arrive, and questions are answered from memory alone. EEM is scored by string containment, MQ by option match, and the other four types by an LLM judge with the benchmark’s prompts.

Baseline settings. All baselines are the implementations shipped with MedMemoryBench and run through its harness, with their default configurations (Table 5). Apart from the REMem rebuild described below, we change only the model that builds memory, so that every method builds memory with the same model. Retrieval budgets are the defaults of each method and are not equalized. REMem’s prompts are in English, so it writes English paraphrases of the Chinese sessions, and its retrieval agent runs on the reader model. As released, the REMem adapter rebuilds REMem at every checkpoint from only the sessions since the previous checkpoint, so REMem would remember only the latest ten visits; we rebuild it from all sessions so far, reusing earlier extraction responses so that each session is still extracted once. In the Qwen setting (Table 2), Qwen3-30B-A3B-Instruct-2507 (FP8 weights, served with vLLM) replaces both gpt-6-luna as the memory builder and gpt-5.1 as the reader, at the same reading

temperature of 0.3; everything else is as in Table 5.
<table><tr><td>Method</td><td>Builder</td><td>Evidence per question</td></tr><tr><td>Embedding RAG</td><td>i none</td><td>top 5 chunks (≤4,096 tok.)</td></tr><tr><td>Mem0</td><td></td><td>gpt-6-luna top 5 memories</td></tr><tr><td>MemOS</td><td></td><td>gpt-6-1una top 5 memories</td></tr><tr><td>LightMem</td><td></td><td>gpt-6-luna top 20 memories</td></tr><tr><td>REMem</td><td></td><td>gpt-6-1una top 5, gist agent</td></tr><tr><td>MedGraph</td><td></td><td>gpt-6-1una 8 units, ≤4 causal, dates</td></tr></table>

Table 5: Baseline settings. The reader (gpt-5.1), answer prompt, judge (claude-sonnet-4-5), and embedder (bge-small-zh-v1.5) are shared by all methods.

Judging. TLA, SUA, and IG are judged with the benchmark’s prompts, and MCD with its nodelevel prompt. A judge reply that cannot be parsed counts as wrong, as in the benchmark; this affects 11 of the 568 judged answers of MedGraph and 11 to 25 for the baselines.

Statistics. Per-question tests are exact McNemar tests, and differences are computed before rounding. In the main setting, MedGraph is 10.5 points above embedding retrieval, the strongest baseline; against every baseline, it is higher on at least 9 of the 10 patients, and $p < 1 0 ^ { - 9 }$ . With the open model, it is 10.4 above embedding retrieval (9 of 10 patients) and more than 12 points above the other baselines, with $p < 1 0 ^ { - 9 }$ for every baseline. Removing the subject filter or the profile lowers Avg. significantly $( p = 0 . 0 2$ and $p = 0 . 0 0 8$ ; Table 3). With GPT, MedGraph’s extracted facts ranked by BM25 alone are 10.6 points above the raw sessions ranked by BM25 (10 of 10 patients; $p < 1 0 ^ { - 9 } )$ ; with Qwen, they are 16.0 above (10 of 10 patients). On the 204 changed-state questions with GPT, the same facts answer 55.4% correctly, against 42.6% for the raw sessions $( p < 0 . 0 0 1 )$ .

Tracing relatives’ facts (Table 3). Each benchmark session is the patient’s own consultation, a consultation about a relative or friend, or a general health question, and its header says which. We trace every retrieved item to the session it was written from. Embedding retrieval returns session text, which includes the header, and MedGraph’s facts and snippets record their session. Mem0 and MemOS log the items written from each session; a Mem0 item counts as a relative’s when any session that wrote or later updated it is a relative’s consultation. REMem links each stored gist to its source passage. LightMem stamps items from the undated sessions, which are exactly the non-patient ones, with the time of the run; we assign each such item to the most similar non-patient session by BM25. An item from a relative’s consultation counts only if it states a health fact of a particular person, not just general advice or the user’s worries; gpt-6-luna decides this for every such item.

Changed-state questions (Table 3). The 204 questions are all 100 SUA questions and the 104 MQ and IG questions whose benchmark trap type concerns a state or dose that has changed: temporal\_state (69 MQ), temporal\_change (22 IG), and dosage\_adjustment (13 IG). The set is defined only by the benchmark’s annotations.

Safety-dependent questions (Table 3). The 180 questions are the MQ and IG questions whose benchmark trap type is a contraindication, an allergy, or a drug interaction. In 97 of them the annotation names the safety item; we keep the 89 whose item is a single drug, drug class, or allergen to avoid, and drop eight whose item is not a drug or allergen. A safety fact counts as present if the drug or allergen, or a synonym from a fixed list, appears within 40 characters of a safety word (allergy, contraindication, avoid, interaction, and similar) in the reader’s context. For embedding retrieval we restore the full retrieved sessions, since its logs keep only the first 500 characters of each. REMem writes English memories, and its logs keep only the final gists, not the triples and earlier gists its agent also reads. For REMem we therefore rerun its retrieval agent on the memory stored at each question’s checkpoint, with the same models and settings, record every gist and triple it reads, and check each item for the safety fact (the drug or allergen, including English names and class members, stated with a safety relation).

Ablation runs. Each MedGraph variant rebuilds the memory from the same cached extraction outputs, in both model settings, so only the retrieval side differs. For the subject-filter ablation, a question whose rendered evidence is identical to that of the full model keeps the full model’s answer and judgment; the other 771 questions are answered and judged again. BM25 only keeps the full retrieval but ranks by BM25 alone. Builtfacts, BM25 keeps the same memory and the subject filter but shows only the top 18 facts ranked by BM25 (18 is the mean number of fact lines, 17.8, that the full retrieval shows per question), with no chain units or packs, date channel, causal expansion, profile, source windows, memory-cutoff line, or raw-text fallback. Raw text, BM25 is the benchmark’s BM25 retriever over whole raw sessions (jieba tokens, top 5, up to 10k tokens) and builds no memory.

## E Additional analysis

Failure analysis (Figure 1a). We sample 30 wrong answers from each baseline (150 in total). For each answer, we label its main cause (e.g., another person’s fact, an outdated state, a missing safety fact, a missing record) and add two marks: whether the evidence holds a health fact of someone other than the patient, and whether it holds two values or states of the same item of the patient with no date or wording that shows which is current. A failure involves state validity if it has the second mark or its main cause is an outdated state; subject attribution if it has the first mark or its main cause is another person’s fact; and safety prioritization if its main cause is a missing safety fact, or if it is one of the 89 questions with identified required safety facts (Appendix D) and its evidence lacks the needed fact. Of the 150, 24 involve state validity, 108 subject attribution, and 18 safety prioritization; 129 involve at least one.

Storage check (Figure 4). We sample 40 wrong answers from each baseline and from MedGraph with a fixed seed (240 in total) and check what each method stored for the needed fact, as of the question’s checkpoint, in this order: the gold answer or the judge is wrong; the fact is not stored; it is stored without its date, its current-state mark, or its exact value; it is stored in a hard-to-find form; it is stored with no owner, that is, it is missing from the evidence while the evidence holds other people’s facts that the memory keeps with no owner; or it is stored intact. A form counts as hard to find only with concrete evidence: the fact is inside a long entry that covers several other topics, or the stored wording lacks the key terms that the visit and the question share. All labels are assigned in one pass per method with the same written criteria. Figure 4 leaves out the answers whose gold answer or judge is wrong (9, 9, 8, 9, and 2 for the baselines in table order, and 15 for MedGraph), since they are not failures of the memory; MedGraph has the fewest wrong answers, so such cases make up a larger share of its sample. The memory of the LightMem run was not kept, so its answers are checked against the rebuilt memory described below, built from the same sessions with the same model.

Fixed early questions. For Figure 3, we take the 105 EEM and TLA questions asked after sessions 10–30 whose answer does not change over time, leaving out questions about the current or recent state. We answer them again with the same reader, prompt, and judge from each method’s memory after all sessions: MedGraph and embedding retrieval rebuild it from the same cached inputs, and Mem0, MemOS, and REMem load the memory they stored in the main run. LightMem’s store was not kept, so we rebuild it once with the same models and settings and answer each question twice from this rebuild, when its checkpoint is reached and after all sessions. Accuracy on these questions changes by −11.4 points for embedding retrieval (exact McNemar $p = 0 . 0 1 7 )$ , −10.5 for Mem0 $( p \ = \ 0 . 0 4 3 )$ −11.5 for MemOS $( p ~ = ~ 0 . 0 1 2 )$ −1.9 for LightMem $( p = 0 . 8 2 ; - 1 1 . 8$ on TLA and +7.4 on EEM), −3.8 for REMem $( p = 0 . 3 9 )$ , and −5.7 for MedGraph $( p = 0 . 1 5 )$

Where MedGraph fails. Of the 39 wrong stateupdate answers, only one comes from the state representation: a later session restated an old lab value, the extractor dated it as a new measurement, and the chain treated that record as current. Eight are not memory errors (four correct answers were misjudged, and four are disagreements with the reference answer), and the remaining 30 are retrieval or reading errors.

## F Examples

Tables 6–8 show three questions from the runs, one per property. Each table puts the evidence and answer of the baselines that fail next to those of MedGraph. Evidence and answers are translated from Chinese and shortened; omitted text is marked with . . . and the facts quoted for MedGraph are lines of its rendered evidence. Red marks evidence that misleads the reader; blue marks the evidence the correct answer needs.

<table><tr><td rowspan=1 colspan=4>Question (patient 9, inference): ‘Doctor, I am always tired in the afternoon. Is it the medicine? Should I stop it for now?&quot;Gold: do not stop; the afternoon fatigue was there before the diabetes drugs were started, so it is unlikely to be their sideeffect.</td></tr><tr><td rowspan=1 colspan=3>Embedding RAG, MemOS XEvidence. Embedding RAG retrieves three noisesessions—two general health consultations and one about thefather&#x27;s blood-pressure drugs—and none of the patient&#x27;s own.Three of MemOS&#x27;s five memories are about the father&#x27;sblood pressure, e.g., “if the father&#x27;s systolic pressure falls to</td><td rowspan=2 colspan=1>MedGraph √Evidence. Profile: “DPP-4 inhibitor: active, started2024-01-10&quot;; &quot;metformin: active, 1,500 mg/day on a newregimen for about two weeks&quot; (both as of 2024-01-10).Matched facts: “(2024-01-04) afternoon fatigue, like beingdrained, worst at 3–4 pm&quot;; “(2024-01-08) afternoon fatigueand poor concentration&quot;; “(2024-01-10) afternoon fatiguestill present&quot;. No fact about the father appears.Answer. The fatigue was reported before the drug change, soit is unlikely to be a side effect; do not stop on your own.</td></tr><tr><td rowspan=1 colspan=3>about 110 mmHg with dizziness, indapamide can be pausedfor a day&quot;.Answer (both discuss the father&#x27;s drugs). “.. . not likelycaused by your father&#x27;s current drugs (valsartan/amlodipine)...&quot; (Embedding RAG).</td><td rowspan=1 colspan=1>ndapamide can be paused</td></tr><tr><td rowspan=2 colspan=4>Why. The subject filter keeps the father&#x27;s sessions out and the profile gives the patient&#x27;s own diabetes drugs, so the answer isabout the right person&#x27;s drugs. The dated fatigue records in the evidence start on 2024-01-04, before the DPP-4 inhibitor isrecorded as added (2024-01-10). Mem0, LightMem, and REMem also answer wrongly; we show the two baselines whoseevidence is about the father.</td></tr><tr><td rowspan=1 colspan=1>about the</td></tr></table>

Table 6: Example A (whose fact).
<table><tr><td colspan="2">Question (patient 7, state update): “If the patient had reduced sertraline to 25 mg once a day on his own three months ago, what is his current actual dose?&quot; The records contain no such reduction. Gold: sertraline 50 mg every morning.</td></tr><tr><td rowspan="3">Embedding RAG, MemOS, LightMem, REMem X Evidence. The evidence of Embedding RAG and MemOS contains “.. . sertraline 50 mg every morning; I have kept taking it&quot;, but only inside a session text or a narrative memory, with nothing that marks it as the current state.</td><td>MedGraph √ Evidence.“[sertraline ·medication chain] [matched · current</td></tr><tr><td>state] sertraline [active]: started two years ago for anxiety and insomnia; still takes 50 mg every morning (recorded 2024-03-12)&quot;; safety alert: “sertraline with rasagiline: interaction risk; the dose should not be changed without the</td></tr><tr><td>doctor&quot;. Answer. The current dose is 50 mg every morning; the reduction in the question does not appear in the records. Why. The tail of the chain is shown as the current state, so the reader can tell a recorded state from a claim made in the</td></tr><tr><td colspan="2">question. Mem0, whose memory says “currently takes 50 mg&quot;, also answers correctly.</td></tr></table>

Table 7: Example B (which state is current).
<table><tr><td colspan="2">Question (patient 4, temporal location): “In the record of 2024-03-20, what was the patient&#x27;s blood pressure?&quot; Gold: 160/95 mmHg.</td></tr><tr><td>Embedding RAG, Mem0, MemOS, REMem X Evidence. Embedding RAG retrieves home readings from</td><td>MedGraph √ Evidence. Read at the question&#x27;s date, the heart-rate, oxygen,</td></tr><tr><td>other days: “mostly 145–155/88–92&quot;.</td><td>and ECG chains return records of that day, e.g., “(2024-03-20) heart rate 88, measured at the emergency visit&quot;. The blood-pressure chain returns its latest reading up to that date, marked as dated 2024-03-02 (home readings, mostly 145–155/88–92). The date channel adds the other records of that day, such as a negative troponin test. The source window of a matched record from that visit reads: “...I went to the emergency department . .. the blood pressure they measured was 160/95 ...&quot;.</td></tr><tr><td>Answer (Embedding RAG). &quot;No blood pressure dated 2024-03-20 is in the records.&quot; Why. The date in the question sets the time at which each chain is read, so the reader gets the measurements of that visit, and</td><td>Answer. 160/95 mmHg, measured at the emergency visit on 2024-03-20.</td></tr><tr><td colspan="2">every line has its date, so the home readings of 2024-03-02 are not taken for that day&#x27;s. The source window of a matched record from the same visit gives the reading. LightMem also answers correctly.</td></tr></table>

Table 8: Example C (which date).