# The Answer Path and the Grounding Instruction in LLM Question Answering over Knowledge Graphs

Arquimedes Canedo

Siemens Digital Industries Software

Princeton, NJ, USA

arquimedes.canedo@siemens.com

Abstract—A graph retrieval-augmented generation pipeline chooses which triples to put in the prompt, a syntax to write them in, an order to write them in, and a sentence telling the model what to do with them. We vary all four over six large language models and two knowledge-graph question answering benchmarks. Two of the four choices move the answer and the other two are flat.

The first is whether the answer path, the triples needed to reach the answer, is in the prompt at all. Holding the number of triples fixed and replacing every triple that is not on the chain with material from an unrelated entity changes answer accuracy by +0.003 F1, while removing the chain costs most of what the graph was worth. Retrieval budget belongs on recall, and precision in the range we can test buys nothing. There is no retriever in this study: subgraphs are constructed from gold SPARQL, so precision here names a property of the context we build rather than a setting on a system.

The second is the grounding instruction. With no facts in the prompt, telling a model to answer using only the provided facts drops F1 from 0.299 to 0.035, a factor of 8.63. That figure describes an evaluation with an empty context arm rather than a working pipeline, and an experiment that applies the instruction to its context arm but not to its no-context baseline manufactures a spurious finding that graph context hurts at depth. We found one in our own results and retract it. Because a wrong subgraph goes undetected by the model, the write path into the knowledge base is an attack surface with the reach of prompt injection and none of its visible signature.

Syntax, triple order and subgraph size produce no effect we can measure at multi-hop depth. The comparison that would price the grounding instruction against correct context turns out not to be measurable with a format-sensitive answer scorer, because the instruction determines the response format, and we report it as an open contrast rather than a number.

Index Terms—knowledge graphs, large language models, GraphRAG, retrieval precision, retrieval recall, grounding instructions, parametric knowledge, graph serialization, evaluation methodology, experimental artifacts

## I. INTRODUCTION

Graph retrieval-augmented generation (GraphRAG) pipelines retrieve subgraphs from a knowledge base, serialize them as text, and place them in the context window of a large language model (LLM) [1]. A pipeline therefore makes four choices: which triples to include, how to serialize them, how to order them, and what the system prompt tells the model to do with them. This paper evaluates all four using the same corpus, models, and scorer to identify where practitioner effort has measurable value.

The results support investment in the first and last choices. The intermediate choices produce no measurable effect under the conditions tested.

Terms. Parametric knowledge is the knowledge a model draws from its weights when no facts are supplied; parametric recall is the act of drawing on that knowledge. An arm is one condition in a comparison evaluated over the same questions as the other conditions. Most comparisons have two arms, while the subgraph-size experiment has three. In a no-context condition, the facts block in an otherwise identical prompt is replaced by “No facts are available,” isolating the presence of graph context. Matched prompts means that both conditions use the same system prompt, including the same grounding instruction.

The answer path contains the triples traversed by a question’s gold SPARQL query to reach its answer. We also refer to these as chain triples, matching the terminology used in our analysis code. A distractor triple is a genuine triple from the question’s entity neighbourhood that does not lie on the answer path. Answers are evaluated by token-level F1 against gold labels, defined as the harmonic mean of token precision and recall on a zero-to-one scale. Unless stated otherwise, intervals are ninety-five% bootstrap intervals over questions.

Two prompt regimes recur throughout the study. The strict prompt, common in research and production pipelines, tells the model to “answer using only the provided facts” and to quote the triples it used. The permissive prompt removes that restriction and allows the model to combine the supplied information with its existing knowledge.

Retrieval precision. An inexpensive retriever commonly returns the answer chain together with unrelated material. Returning only relevant material is more costly and motivates reranking and threshold tuning. The precision experiment holds each prompt at 50 triples, retains the chain in every condition, and replaces an increasing share of the remaining triples with material from an unrelated question’s subgraph. Across the full sweep, answer F1 changes by only +0.003. The null holds at every reasoning depth and among questions that the model could not answer without graph context. Removing the chain produces a large loss in accuracy. These results separate retrieval quality into two components with different consequences: preserving the answer path has substantial value, while improving the provenance of surrounding triples does not within the tested range.

The grounding instruction. With no facts available, the strict prompt scores F1 = 0.035, compared with 0.299 under the permissive prompt, for a difference of 0.264. Models follow the restriction even when the prompt supplies no facts. This result characterizes an evaluation with an emptycontext arm; it does not characterize a working pipeline whose retriever returns evidence. If the context arm receives the strict prompt while the no-context baseline receives a different prompt, the resulting contrast combines the effects of context and instruction while attributing both to the graph. An initial experiment built this way made context appear harmful at three and four hops, and that result is retracted here. When both arms use the strict prompt, context helps at one, two, and three hops. Its four-hop effect is non-negative but inconclusive.

What is flat. Once reasoning is required, serialization syntax, triple order, and subgraph size produce no measurable effects. Syntax appears to matter only for one-hop parsing, where the observed difference tracks whether the serializer provides a human-readable predicate name.

What we cannot measure. The remaining practical question is how much the strict instruction costs when the context is correct. The present apparatus cannot answer it. The strict prompt requires a structured evidence block, while the permissive prompt permits free-form answers. Prompt regime therefore determines response format, and the answer scorer is sensitive to that format. Changing the scorer’s fallback reverses the sign of the contrast, with both estimates rejecting the null. We consequently report this contrast as unmeasurable with the current apparatus. Four results in earlier drafts depended on it.

Table I lists the variables varied by the experiments. The design makes them orthogonal, allowing each to change while the others remain fixed. Reasoning depth is the exception because it is a property of the question, not an experimental setting. In this corpus, depth is also perfectly aliased with the source benchmark.

The paper makes four contributions based on 16 experiments comprising 30,841 trials across six models from three vendors.

1) For a retrieved subgraph, what matters is whether it contains the answer. With the chain present and the prompt fixed at 50 triples, reducing context precision from 1.00 to 0.51 changes answer F1 by +0.003 (95% CI [−0.019, +0.031], permutation p = 0.812). Removing the chain lowers F1 to 0.231 under the permissive prompt and 0.005 under the strict prompt. For a pipeline operating under a fixed budget, these results support prioritizing recall. The conclusion also holds when the analysis is restricted to questions the model could not answer unaided.

2) The nine-fold grounding figure applies to the nocontext condition and characterizes an evaluation, not a pipeline. The 8.63-fold suppression occurs in an empty-context arm. It is the largest effect measured in this study and is especially easy to misattribute to the graph. Any experiment that changes prompt regime together with context will assign the combined effect to context, as the retracted comparison did.

TABLE I  
VARIABLES VARIED IN THE STUDY AND THE CORRESPONDING SUMMARY FINDINGS. DEPTH IS MEASURED; ALL OTHER VARIABLES ARE SET EXPERIMENTALLY.
<table><tr><td>Variable</td><td>Levels</td><td>Effect</td></tr><tr><td>Retrieval precision</td><td>0–100% wrong</td><td>None at constant vol- ume; recall is the lever</td></tr><tr><td>Context</td><td>correct, wrong, none</td><td>Large when the an- swer is absent. Sup- pression itself is small and model-dependent</td></tr><tr><td>Prompt</td><td>strict, permissive</td><td>Largest lever with no context; not measur- able with it</td></tr><tr><td>Format</td><td>7 serializations</td><td>Parsing only, aliased with labelling</td></tr><tr><td>Triple order</td><td>5 orders</td><td>None</td></tr><tr><td>Subgraph size</td><td>0–200 triples</td><td>None</td></tr><tr><td>Model</td><td>6, in 2 tiers</td><td>Modulates format</td></tr><tr><td>Reasoning depth</td><td>1–4+ hops</td><td>Governs the rest, and is aliased with bench- mark</td></tr></table>

3) The raw comparison substantially overstates how much a wrong subgraph suppresses parametric recall. Replacing the correct subgraph with an unrelated one reduces F1 from 0.627 to 0.231, but almost all of this change reflects the loss of correct facts. Relative to a nocontext baseline, which isolates suppression, the effect is −0.068 (95% CI [−0.098, −0.039]). Moreover, 2 of the six models show no suppression. The performance that remains under a poisoned subgraph is parametric recall surviving the supplied context; a graph the model has never encountered offers no such reserve.

4) Four nulls and one unmeasurable contrast. Subgraph size, triple ordering, serialization and retrieval precision are not levers. After matching question sets, size has no effect from chain-only to two hundred triples, and ordering has no effect at either tested scale. The serialization spread decreases from 0.234 at one hop to 0.036 at three hops. Separately, the strict-versus-permissive contrast with correct context is not measurable with a formatsensitive answer scorer, because the prompt regime also determines the response format.

## II. RELATED WORK

## A. Retrieval Quality in Graph Question Answering

Dai et al. [2] report that multi-hop accuracy collapses beyond two hops and that irrelevant triples do not reduce accuracy. Our precision sweep provides a controlled test of the latter result by holding triple count fixed and changing only provenance, thereby separating relevance from context volume. Mavromatis and Karypis [3] adopt a different design, using a graph neural network to prune the retrieved subgraph before it reaches the model. Our findings provide no evidence that this pruning improves answer accuracy, although it may still reduce token use.

TABLE II  
SUMMARY OF THE 16 EXPERIMENTS, TOTALING 30,841 TRIALS ACROSS SIX MODELS AND THREE VENDORS. THE n COLUMN GIVES EACH EXPERIMENT’S DENOMINATOR; THE DENOMINATORS OVERLAP BECAUSE SEVERAL ARMS REUSE PHASE 1 AS THEIR CONTROL.
<table><tr><td>#</td><td>Experiment</td><td>Lever Tested</td><td>Key Result</td><td>n Verdict</td><td></td></tr><tr><td>1</td><td>Precision sweep</td><td>Wrong-provenance triples</td><td>∆=+0.003 at 0.51 precision</td><td>1,999</td><td>Precision is not a lever</td></tr><tr><td>2</td><td>Wrong context (strict)</td><td>Context attribution</td><td>wrong=0.005, correct=0.579</td><td>2,250</td><td>Falls to no-context floor</td></tr><tr><td>3</td><td>Wrong context (permissive)</td><td>Override, no strict instr.</td><td>wrong=0.231, correct=0.627</td><td>1,500</td><td>Override model-dependent</td></tr><tr><td>4</td><td>Baseline (permissive)</td><td>No-context, permissive</td><td>F1=0.299</td><td>750</td><td>Unfair comparison (retracted)</td></tr><tr><td>5</td><td>Matched-prompt baseline</td><td>No-context, matched</td><td>1-hop +0.809, 4-hop +0.147</td><td>750</td><td>Helps 1–3 hop; 4-hop n.s.</td></tr><tr><td>6</td><td>Second template</td><td>Rephrased strict prompt</td><td>No-ctx F1=0.019</td><td>1,500</td><td>Suppression confirmed</td></tr><tr><td>7</td><td>Format comparison</td><td>Serialization format</td><td>1-hop spread 0.234, 3-hop spread 0.036</td><td>5,250</td><td>1-hop only, labelling aliased</td></tr><tr><td>8</td><td>Ordering comparison</td><td>Triple order</td><td>∆=+0.006</td><td>3,000</td><td>No effect</td></tr><tr><td>9</td><td>Triple order × size</td><td>Ordering at 200 triples</td><td>∆200=-0.005</td><td>3,000</td><td>No effect at scale</td></tr><tr><td>10</td><td>Distractor density</td><td>Subgraph size</td><td>Matched: 0.752 / 0.737 / 0.730, all n.s.</td><td>1,842</td><td>No effect (selection artifact)</td></tr><tr><td>11</td><td>Evidence-first</td><td>Prompt order</td><td>Faithfulness +0.028, F1 —0.021</td><td>1,500</td><td>Grounding ≠ reasoning</td></tr><tr><td>12</td><td>Chain-of-thought</td><td>Step-by-step prompt</td><td>2-hop −0.020, 3-hop −0.043</td><td>1,500</td><td>Hurts at all depths</td></tr><tr><td>13</td><td>Dynamic oracle</td><td>Per-hop scoping</td><td>∆=-0.057 vs full-context</td><td>2,250</td><td>Worse: loses holistic view</td></tr><tr><td>14</td><td>Temperature 0.7</td><td>Sampling verification</td><td>Rankings preserved</td><td>3,750</td><td>Greedy results hold</td></tr><tr><td>15</td><td>SPARQL classification</td><td>Chain vs. fan-out</td><td>All 4 types present</td><td>43q</td><td>Structure varies within depth</td></tr><tr><td>16</td><td>Instruction self-handicap</td><td>Strict vs. permissive</td><td>Arms differ in response format</td><td>1,500</td><td>Not measurable with this scorer</td></tr></table>

## B. Graph Serialization for LLMs

Fatemi et al. [4] showed that graph encoding choice produces accuracy differences of 4.8–61.8% on synthetic structural tasks across 9 encodings and five PaLM models. Their encodings change naming and framing, including adjacency lists and sentences such as “X and Y are friends”; they do not compare formal Resource Description Framework (RDF) serialization syntax. Sui et al. [5] compared table formats including CSV, JSON, HTML, and Markdown for single-hop LLM comprehension and found format-dependent accuracy. Frey et al. [6] directly benchmarked LLM comprehension of RDF syntax. Perozzi et al. [7] instead represent structured data through learned soft prompts rather than textual serialization.

This study extends those lines of work to real knowledgegraph question answering with seven RDF-native formats. Its depth-stratified analysis also distinguishes parsing effects from reasoning effects. As in Fatemi et al., however, naming and syntax vary together in our design, leaving the two factors aliased.

## C. Knowledge Conflicts and Context Override

Wu et al. [8] found that LLMs override parametric knowledge with retrieved text roughly 60% of the time. Xie et al. [9] reported memorization ratios above 80% for popular entities. Zou et al. [10] corrupt a text corpus to steer retrievalaugmented generation (RAG) answers, while Zhao et al. [11] showed that adversarial knowledge-graph (KG) triples reduce

RoG’s F1 from 70% to 38%, a relative decline of 46%. Prior work uses either natural-language passages or individual adversarial triples; our experiment substitutes complete structured subgraphs.

Our result is not directly comparable to the figure reported by Wu et al. Their measure is an override rate, defined as the proportion of items for which the model followed retrieved text. Ours is a change in F1. Relative to a no-context baseline, the measured suppression is −0.068 F1, and 2 of the six models show no suppression. This evidence does not support treating suppression as an intrinsic property of structured context.

## D. Controlling Reliance on Context

A related line of work studies how strongly a model should rely on retrieved text relative to its parameters and treats that reliance as a control objective. Zhou et al. [12] showed that instruction phrasing alone changes context-faithfulness. Huang et al. [13] formulate the objective as situated faithfulness, in which context is trusted when warranted. Bi et al. pursue the same objective through alignment [14] and finegrained inference-time control [15]; Huang et al. [16] suppress knowledge-critical feed-forward layers. This literature supports calibrated reliance on context as an interior quantity worth tuning.

Our contribution to that work is methodological and negative. The endpoint prompts in our comparison produce different response formats, so an end-task scorer with unequal sensitivity to those formats cannot estimate the effect of moving between the endpoints. Section VIII documents this measurement failure.

In a controlled clinical setting, Mandarapu and Kunkunuru [17] report that knowledge-graph grounding helps only when the relevant knowledge is absent from training. Because our entities come from Wikidata and were almost certainly represented in pretraining, their result predicts a small context benefit in this setting. We measure a large benefit because the comparison also depends on the prompt: their baseline may use parametric knowledge, while ours is forbidden to do so.

## E. Multi-Hop Knowledge Graph Question Answering

Sun et al. [18] introduced Think-on-Graph, and Jiang et al. [19] introduced StructGPT. Both show that LLM-driven iterative graph exploration can succeed on multi-hop tasks by allowing the model to control which triples it receives. Shan and Luo [20] found that bounded path history outperforms full history in iterative knowledge graph question answering (KGQA). The dynamic-oracle experiment in Section VII extends this line by showing that even oracle-correct perhop scoping degrades performance when the decomposition is imposed externally. Nguyen et al. [21] evaluate chainof-thought directly on multi-hop knowledge-graph reasoning instead of relying only on end-task accuracy, and our chain-ofthought result is consistent with the pessimistic interpretation of that work.

## F. Capability Equalizer Effects

Canedo [22] showed that architecture-specification format acts as a capability equalizer for code-generation agents, with format effects tenfold larger for weaker models. We observe an analogous pattern, but only for parsing. The format F1 spread is 2.6× larger for mid-tier models than for frontier models. The effect disappears at multi-hop depths and remains subject to the same alias between syntax and predicate vocabulary disclosed in Section VII.

## III. EXPERIMENTAL SETUP

Figure 1 summarizes the experimental pipeline.

## A. Questions and Subgraphs

The corpus contains 125 Wikidata-grounded questions [23] from two benchmarks: 82 one- or two-hop questions from LC-QuAD 2.0 [24] and 43 questions of at least three hops from the tenth edition of Question Answering over Linked Data (QALD) [25]. A gold SPARQL query identifies the reasoning chain for each question. We use it to extract an oracle subgraph of 50 triples comprising the chain and distractors. The distractors are genuine Wikidata facts about nearby entities, so these subgraphs approximate plausible retriever output without using synthetic noise. The corpus includes 46 one-hop, 36 two-hop, 29 three-hop, 12 four-hop, and 2 five- or six-hop questions. The final 2 questions remain in all aggregate results but are excluded from hop-stratified tables because that group is too small to interpret.

Depth is completely aliased with benchmark in this corpus. LC-QuAD supplies every one- and two-hop question, while QALD supplies every three- and four-hop question. Specifically, all 46 one-hop and 36 two-hop questions come from LC-QuAD, and all 29 three-hop and 12 four-hop questions come from QALD. Consequently, no depth-stratified contrast in this paper can distinguish an effect of depth from a difference between the datasets. This qualification appears here because it applies to every depth-stratified result that follows.

![](images/cf6b84f84c2dce12f93d63c8f100c9db44907b4e621b6164b6ef7a4eaecee5de.jpg)  
Fig. 1. Experimental pipeline. The 125 questions have oracle subgraphs that are serialized in seven formats under five triple orders and sent to six LLMs. Responses are evaluated for answer F1 and evidence faithfulness.

The corpora also differ on several measured properties. LC-QuAD has 3.12 gold answers per question on average, compared with 7.70 for QALD. The answer is present in 76 of 82 LC-QuAD subgraphs and in 16 of 43 QALD subgraphs. With correct context, the models score 0.710 on LC-QuAD and 0.270 on QALD; without a graph, they score 0.231 and 0.427, respectively. Thus, on QALD, unaided performance exceeds performance with the graph. Depth, answer-set size, coverage, difficulty, and the direction of the context effect all covary under this design.

Accordingly, a result reported “at three-hop” should be read as a result “on QALD, at three-hop.” We retain depthstratified reporting because it matches the experimental design and conventions in prior work, but the evidence does not support recommending a depth threshold. Because depthstratified knowledge-graph question answering corpora are commonly assembled by pooling benchmarks with different depth profiles, we expect this alias may also affect results beyond this study.

Two additional corpus properties qualify the depth-stratified analyses. Gold-answer coverage within the extracted subgraph is incomplete and declines with depth, while hop count measures predicate count and not a verified sequential chain. Section §IX quantifies both properties.

## B. Models

The study uses six models from three vendors and two capability tiers: Claude Sonnet 4.6 and Haiku 4.5 from Anthropic, GPT-5 and GPT-5 Mini from OpenAI, and Gemini 2.5 Pro and

Gemini 2.5 Flash from Google. All calls use temperature = 0 through a unified API gateway serving the three providers.

## C. Serialization Formats and Triple Ordering

Each subgraph is represented in seven formats ranging from human-readable text to machine-oriented syntax: prose as natural-language sentences, tabular as pipe-delimited rows, Turtle [26], N-Triples [27], Cypher, JSON-LD [28], and RDF/XML [29]. The selection reflects formats practitioners are likely to have available, not a systematic sampling of a serialization design space. Four are standard RDF serializations emitted directly by W3C-compatible toolchains. Cypher represents the property-graph ecosystem outside the RDF stack, while prose and pipe-delimited tables are common choices when GraphRAG implementations provide their own serializers. Table III presents the same three triples in each format.

Serialization determines syntax but not necessarily sequence. For each format, we test five orders: entity-centric grouping by subject, path-centric placement of chain triples first, breadth-first search (BFS) from the question entity, a seeded random shuffle, and alphabetical lexicographic sorting. Prose, N-Triples, tabular, and Cypher preserve the supplied triple order. Turtle, JSON-LD, and RDF/XML regroup triples by subject, partially overriding the requested sequence. Prose therefore provides the cleanest test of ordering effects.

## D. Prompt Templates

![](images/cc87aa9745cac3a2905fadf8af0f5473197a78b5da9e5be623e462859014d5cd.jpg)

These boxes reproduce the prompts exactly. They differ along two dimensions: the strict prompt confines the answer to the supplied facts and requires a specific response format. Section VIII examines how the formatting difference interacts with the scorer.

## E. Two-Axis Evaluation

Each response receives separate scores for answer quality and evidence faithfulness.

Answer F1 compares the parsed response with the gold labels through token-level matching after lowercasing, article removal, date normalization, and alias resolution using a curated table. Multi-answer questions use optimal bipartite matching so that answer order does not affect the score.

Evidence faithfulness is the fraction of cited triples that match triples in the supplied subgraph. The evaluator recognizes six citation forms—arrow, dash-arrow, colon, parenthetical, comma-separated, and natural language—and fuzzymatches them to subgraph triples. The two-axis evaluation distinguishes a correct answer supported by valid evidence, which is consistent with reading and reasoning over the graph; a correct answer paired with fabricated evidence, which is consistent with parametric recall followed by confabulated citations; and an incorrect answer supported by valid evidence, where relevant triples were found but not composed into the gold answer.

Answer parsing uses a primary path and a fallback. The primary path extracts a line beginning with “Answer:”, as required by the strict prompt, while the fallback reads a free-form response. The prompt affects response format, and response format determines which parsing path runs. Section VIII describes the resulting measurement problem.

## IV. RETRIEVAL PRECISION DOES NOT MATTER

Retrieval can fail by omitting the answer chain or by returning that chain with additional material that is not on it. Although both failures are commonly summarized as retrieval quality and addressed with the same tuning controls, the results below show that they have different consequences.

## A. Design

We fix each subgraph at 50 triples and vary its composition while keeping the answer chain present. The other slots initially contain neighbours associated with the question. At each step of the sweep, a fraction of those triples is replaced with triples from another question’s subgraph, using the deterministic donor mapping also used in the wrongcontext experiments. The replaced fraction is 0, 25, 50, or 100% of the available slots. This lowers context precision from 1.00 to approximately 0.51.

Because every condition contains the same number of prompt triples, the sweep changes provenance without changing volume. Each condition includes every question and model, yielding 1,999 trials across 4 models and 125 questions, with prose serialization and entity-centric ordering.

The full-precision condition reproduces the standard correct-context condition through a second, independently implemented runner. Its F1 is within 0.002 of the Phase One prose average. Read that as reassurance rather than as a validation: the sweep covers 4 models and the Phase One average covers six, so the two figures are unpaired means over different model sets and the agreement could as easily be produced by the two absent models as by the runners agreeing. A validation would recompute Phase One on the same four models, paired by question, and we have not done it. Nothing in the sweep’s own contrast depends on this, since both of its arms come from the same runner.

TABLE III  
SEVEN REPRESENTATIONS OF THE SAME KNOWLEDGE, ILLUSTRATED WITH THREE TRIPLES ABOUT SUNGKYUNKWAN UNIVERSITY. MEAN LENGTH PERFIFTY-TRIPLE SUBGRAPH RANGES FROM 644 TOKENS FOR PROSE TO 2,383 FOR N-TRIPLES; RDF/XML USES 1,508, AND JSON-LD USES 1,642.LENGTH DOES NOT PREDICT ACCURACY BECAUSE N-TRIPLES IS BOTH THE LONGEST REPRESENTATION AND A TOP-TIER FORMAT. PROSE, TABULAR,TURTLE, N-TRIPLES, AND CYPHER INCLUDE PREDICATE NAMES, WHEREAS JSON-LD AND RDF/XML EXPOSE ONLY P-NUMBERS.  
Format Serialized Output (3 of 50 triples)   
Prose Sungkyunkwan University country South Korea. Sungkyunkwan University located in the administrative territorial entity   
Seoul. Sungkyunkwan University location Suwon.   
Tabular Subject | Predicate | Object   
Sungkyunkwan University | country | South Korea   
Sungkyunkwan University | located in admin. territorial entity | Seoul   
Sungkyunkwan University | location | Suwon   
Turtle wd:Q41085 rdfs:label "Sungkyunkwan University" ;   
wdt:P17 wd:Q884 ; # South Korea   
wdt:P131 wd:Q8684 ; # Seoul   
wdt:P276 wd:Q20714 . # Suwon   
N-Triples # Sungkyunkwan University -- country --> South Korea   
<http://.../Q41085> <http://.../P17> <http://.../Q884> .   
# Sungkyunkwan University -- location --> Seoul   
<http://.../Q41085> <http://.../P131> <http://.../Q8684> .   
Cypher CREATE (n0:Entity {name: ’Sungkyunkwan University’})   
CREATE (n1:Entity {name: ’South Korea’})   
CREATE (n0)-[:COUNTRY]->(n1)   
CREATE (n0)-[:LOCATED\_IN]->(n2)   
JSON-LD {"@id": "wd:Q41085", "rdfs:label": "Sungkyunkwan University",   
"wdt:P17": {"@id": "wd:Q884", "rdfs:label": "South Korea"},   
"wdt:P131": {"@id": "wd:Q8684", "rdfs:label": "Seoul"}}   
RDF/XML <rdf:Description rdf:about="wd:Q41085">   
<rdfs:label>Sungkyunkwan University</rdfs:label>   
<wdt:P17 rdf:resource="wd:Q884"/> <!-- South Korea -->   
<wdt:P131 rdf:resource="wd:Q8684"/> <!-- Seoul -->   
</rdf:Description>

All conditions use the strict prompt and therefore induce the same response format. The scorer issue described in Section VIII does not affect this comparison, and disabling the fallback parser leaves the results unchanged.

## B. Result

Answer accuracy does not measurably change across the sweep. Replacing every distractor slot with triples from an unrelated entity neighbourhood changes answer F1 by +0.003 (95% CI [−0.019, +0.031], permutation p = 0.812, 125 questions). Table IV reports all conditions. The null also holds within each reported depth: +0.030 at one hop, −0.021 at two hops, −0.007 at three hops, and −0.001 at four hops. Across the four models, individual deltas range from −0.025 to +0.014.

Two restricted analyses address cases that could make the pooled null uninformative.

The first restriction concerns answer coverage. Fixing the chain triples does not guarantee that the subgraph contains every answer: only 92 of 117 answerable questions have all gold answer labels in their subgraph. For questions whose context never contained the answer, changing the surrounding triples may have no answer to disrupt. Among the 92 fully covered questions, the delta remains +0.004 (95% CI [−0.027, +0.037], p = 0.799); it is −0.001 for the 33 questions without full coverage. The covered subset is not constrained by a floor, scoring 0.579 under full context precision.

The second restriction concerns parametric knowledge. If a model has memorized an answer, it may ignore degraded context, allowing parametric recall to conceal a genuine precision effect in the pooled result. The same null appears on the 48 questions for which unaided performance was zero and the model therefore had to use the supplied information: the delta is +0.015 (95% CI [−0.035, +0.066]). This subset approximates the knowledge conditions that would apply throughout a private graph.

Answer accuracy and evidence faithfulness are distinct axes that can move independently. Faithfulness is the only measured outcome that declines across this sweep, falling from 0.914 to 0.888 as context precision decreases. Models cite irrelevant triples somewhat more often while continuing to derive correct answers from the retained chain. An answer metric alone will not reveal this deterioration, so systems that expose evidence to users should evaluate faithfulness separately.

TABLE IV  
RETRIEVAL PRECISION AT CONSTANT VOLUME. EACH ROW CONTAINS THE SAME NUMBER OF TRIPLES AND DIFFERS ONLY IN THE FRACTION IMPORTED FROM ANOTHER QUESTION’S NEIGHBOURHOOD. ANSWER ACCURACY REMAINS FLAT, WHILE EVIDENCE FAITHFULNESS DECLINES SLIGHTLY.
<table><tr><td>Wrong triples</td><td>Context precision</td><td>Answer F1</td><td>Faithfulness</td></tr><tr><td>0%</td><td>1.00</td><td>0.579</td><td>0.914</td></tr><tr><td>25%</td><td>0.86</td><td>0.584</td><td>0.902</td></tr><tr><td>50%</td><td>0.73</td><td>0.583</td><td>0.893</td></tr><tr><td>100%</td><td>0.51</td><td>0.583</td><td>0.888</td></tr></table>

## C. Where the null stops

Together with Section VI, this result separates two components of retrieval quality. Removing the answer chain is catastrophic: wrong-context F1 falls to 0.231 with a permissive prompt and 0.005 with a strict prompt. Keeping the chain while filling the remaining prompt capacity with unrelated triples has no measured answer cost. For a pipeline operating under a fixed budget, the evidence therefore supports allocating that budget to retrieval recall.

This conclusion is bounded by the experimental construction. Every condition in the precision sweep contains the answer chain, so the result applies to retrievers that find the answer and return additional irrelevant material. It does not apply when retrieval omits the answer, which the wrong-context experiment shows is harmful. On this evidence, improving precision at the expense of recall offers no measured benefit and introduces substantial risk.

Two additional qualifications apply. The wrong triples for each question come from one donor question and are therefore more internally coherent than errors drawn from many unrelated sources; dispersed noise could behave differently. The maximum replacement fraction is also limited by chain length, which averages 23 of the 50 triples. Consequently, the 100% condition retains the complete chain and contains wrong material in only roughly half of the subgraph, so it is not a wholly incorrect context.

## V. THE GROUNDING INSTRUCTION SUPPRESSESPARAMETRIC RECALL

The strict prompt limits answers to the supplied facts, and models follow that instruction even when no facts are supplied.

With the facts block replaced by “No facts are available,” models under the strict prompt score $\mathrm { F 1 } ~ = ~ 0 . 0 3 5$ across 750 trials and all six models. On the same questions, the permissive prompt yields 0.299. The difference is a factor of 8.63 and a swing of 0.264 F1, the largest single effect measured in this study. The scorer’s fallback path rescues no trial in either arm, so disabling it leaves the contrast unchanged. A rephrased strict template (“Base your answer exclusively on the facts provided below”) produces slightly stronger suppression: no-context F1 is 0.019 over 1,500 trials, compared with Template A’s 0.035. Context benefit remains comparable across the templates (0.560 against 0.581, both on prose). The suppression therefore does not depend on the specific strict template.

## A. What the figure describes

This effect is specific to the no-context condition. It characterizes a model instructed to ground its answer in facts that are absent, a condition created by an evaluation but not by a working pipeline. It does not measure the cost of the instruction when a retriever supplies the correct subgraph. Section VIII examines that second contrast and concludes that the present apparatus cannot measure it.

The no-context result remains important because it can invalidate an evaluation whose arms use different prompts.

## B. The artifact

The initial comparison paired a correct subgraph under the strict prompt with a no-context baseline that omitted both the “use only the provided facts” restriction and the structured evidence format. In that comparison, context appeared beneficial at one and two hops but harmful at three hops $( \Delta \mathrm { F } 1 = - 0 . 0 8 2 )$ and four hops $( \Delta \mathrm { F 1 } = - 0 . 3 3 8 )$ . The result was interpreted as evidence that text-serialized subgraphs interfere with multihop reasoning.

That comparison changed context and prompt regime simultaneously, and the prompt has the larger effect. A permissive no-context baseline is not a fair comparator for a strict context arm. The result does not establish worse reasoning over the graph. The strict arm prohibited fallback to parametric knowledge, while the baseline allowed it.

When both arms use the strict prompt, context helps at one, two and three hops; at four hops the estimate is nonnegative but inconclusive (Table V). The benefit declines from $\Delta \mathrm { F } 1 = + 0 . 8 0 9$ at one hop to +0.147 at four hops without becoming negative. It is significant at one, two and three hops under an exact sign-flip permutation test, with $p < 0 . 0 0 1$ in each case. At four hops, $p = 0 . 1 0 9$ across 12 questions. Of those questions, 7 show no benefit, and dropping the two largest deltas reduces the mean from +0.147 to +0.028. Excluding the two models whose truncated trials were re-run gives +0.096 at $p \ = \ 0 . 3 2 8$ . The four-hop claim is limited to context not hurting; the evidence does not establish a measurable benefit. Although the bootstrap interval excludes zero, we do not rely on it because the percentile bootstrap is anti-conservative for a sample this small, with five exact zeros and two large positives.

The arms in Table V are not matched on format. The nocontext arm uses prose only, while the correct-context arm averages seven serializations. Section VII estimates the format spread at three hops as 0.036 F1, well within these deltas, so the format mismatch does not account for them.

Figure 2 presents the three baselines that separate the two changes. The separation between strict context and permissive no context at three and four hops was initially attributed to context interference. It instead measures the difference between instructed and unrestricted parametric recall.

TABLE V  
CONTEXT BENEFIT UNDER MATCHED PROMPTS, BOTH CONDITIONS STRICT. NO-CONTEXT IS THE MATCHED BASELINE (750 TRIALS), CORRECT-CONTEXT IS ALL OF PHASE 1 (5,250 TRIALS). ALL VALUES ARE ANSWER F1. 95% BOOTSTRAP CI IN BRACKETS; p IS AN EXACT SIGN-FLIP PERMUTATION TEST, WHICH WE RELY ON IN PREFERENCE TO THE INTERVAL AT 4-HOP.
<table><tr><td>Condition</td><td>1-hop</td><td>2-hop</td><td>3-hop</td><td>4-hop</td></tr><tr><td>Matched no-context</td><td>0.014</td><td>0.046</td><td>0.044</td><td>0.061</td></tr><tr><td>Correct context</td><td>0.823</td><td>0.567</td><td>0.311</td><td>0.208</td></tr><tr><td>∆</td><td>+0.809</td><td>+0.521</td><td>+0.268</td><td>+0.147</td></tr><tr><td>95% CI</td><td>[0.73,0.88]</td><td>[0.39,0.64]</td><td>[0.14,0.40]</td><td>[0.003,0.32]</td></tr><tr><td>perm. p</td><td>&lt;0.001</td><td>&lt;0.001</td><td>&lt;0.001</td><td>0.109</td></tr><tr><td>n questions</td><td>46</td><td>36</td><td>29</td><td>12</td></tr></table>

![](images/b3db489e95b2e246d9ecc47cbfa0d69911ba9306918f9452cce5db907f82685b.jpg)  
Fig. 2. Three baselines reveal the artifact. Strict prompt with no facts (F1 = 0.014–0.061); strict prompt with correct subgraph; and permissive prompt with no facts, which is unrestricted parametric recall. The original “context hurts” finding compared the strict-with-context series against the permissive-no-context one, conditions that differ in prompt regime and not just in context presence.

## C. Why the design is easy to build

The design follows from two individually plausible choices. A context arm receives the strict prompt to ensure that the model reads the graph. A no-context arm is then created by removing the facts, often along with the instruction to use them because that instruction appears inapplicable once the facts are gone. Each arm is coherent in isolation, but together they vary prompt regime and context presence at the same time. At depths where parametric knowledge is strong, any experiment with this construction will therefore make context appear harmful.

We treat the prevalence of this design in other work as a hypothesis, not a finding. Establishing prevalence would require coding published RAG evaluations according to whether their no-context baselines retain the prompt used by their context conditions. We have not conducted that review. The issue warrants attention because a results table does not show which of the two changes produced the difference.

TABLE VI  
CONTEXT OVERRIDE BY PROMPT REGIME (STRICT: 2,250 TRIALS; PERMISSIVE: 1,500 TRIALS; BOTH USE ALL 6 MODELS). ALL VALUES ARE ANSWER F1. THE NO-CONTEXT ROW IS THE MATCHED-PROMPT BASELINE OF SECTION V. 95% BOOTSTRAP CIS IN BRACKETS.
<table><tr><td>Condition</td><td>Strict prompt F1</td><td>Permissive prompt F1</td></tr><tr><td>Correct context</td><td>0.579 [0.510,0.658]</td><td>0.627 [0.561,0.702]</td></tr><tr><td>Wrong context</td><td>0.005 [0.000,0.012]</td><td>0.231 [0.177,0.274]</td></tr><tr><td>No context</td><td>0.035</td><td>not run</td></tr></table>

## D. The retraction is narrower than it first read

The matched comparison above uses the strict prompt in both arms. Using a permissive prompt for the context arm produces a different pattern: +0.699 at one hop and +0.234 at two hops, no measured effect at three hops (+0.061, p = 0.286), and a negative effect at four hops (−0.120, $p = 0 . 0 0 6 )$ . At four hops, a model free to use its own knowledge scores F1 = 0.545 without a graph and 0.425 with one.

That comparison is not dispositive for two reasons. Its baseline uses a third prompt: neither the strict nor the permissive template, but a plain instruction to “answer to the best of your knowledge.” The arms are therefore unmatched and reproduce the design defect examined in this section. The four-hop result also covers only 12 questions, of which 4 contain their answer. The retraction applies to the original comparison; it does not establish that context can never hurt. Under a permissive prompt, the question remains open and the observed sign is negative. Resolving it requires a permissive no-context arm that has not been run.

## VI. WRONG CONTEXT AND THE POISONED RETRIEVAL PATH

The precision sweep retains the answer chain. This experiment removes it by replacing each question’s correct subgraph with the subgraph of another question, selected through a deterministic offset in the question list. The substitution is tested under both prompt regimes.

Under the strict prompt, wrong-context F1 is 0.005, compared with 0.579 for the correct subgraph across 2,250 trials and every model. Under the permissive prompt, wrong-context F1 rises to 0.231 (95% CI [0.177, 0.274]), compared with 0.627 for correct context across 1,500 trials and every model. Table VI summarizes the prompt regimes, and Table VII reports the strict condition by hop count. Under the strict prompt, wrong-context performance falls to the no-context baseline, which is itself near zero.

A. The comparison that would isolate suppression, and the one we have

Comparing wrong context with correct context combines the loss of useful facts with any suppression of the model’s existing knowledge. Only the suppression component is override, so the relative drop of 63% between those arms does not measure it. The contrast that would isolate it compares wrong context with no context, holding the prompt fixed across both.

TABLE VII  
WRONG-CONTEXT OVERRIDE BY HOP COUNT, IN ANSWER F1 (STRICT PROMPT, ALL 6 MODELS, PROSE). THE NO-CONTEXT ROW IS THE MATCHED-PROMPT BASELINE. CORRECT-CONTEXT DIFFERS FROM TABLE V BECAUSE THIS EXPERIMENT IS PROSE-ONLY AND RUNS ITS OWN TRIALS.
<table><tr><td>Condition</td><td>1-hop</td><td>2-hop</td><td>3-hop</td><td>4-hop</td></tr><tr><td>No context</td><td>0.014</td><td>0.046</td><td>0.044</td><td>0.061</td></tr><tr><td>Wrong context</td><td>0.000</td><td>0.017</td><td>0.000</td><td>0.000</td></tr><tr><td>Correct context</td><td>0.875</td><td>0.559</td><td>0.324</td><td>0.215</td></tr></table>

We do not have that contrast. Our no-context arm uses a third prompt, neither strict nor permissive but a plain instruction to answer to the best of the model’s knowledge, while the wrong-context arm uses the permissive template that invites the model to combine supplied information with what it knows. The two arms therefore differ in prompt as well as in context, which is the same design this paper retracts a finding over in §V-B. The measured difference is −0.068 F1 (95% CI [−0.098, −0.039], p 0.000) over 125 questions, and the prompt change could account for part or all of it. Disabling the scorer’s fallback path leaves the sign and significance unchanged, which speaks to the parser and not to the prompt mismatch.

What the number bounds is the size of any suppression effect, not its value: whatever override contributes, it is small beside the 63% drop against correct context, which is the comparison that matters for a pipeline. Settling the value needs a permissive no-context arm, which we have not run.

Whatever that difference contains, it is not uniform across models. It inherits the prompt mismatch above, so the permodel figures are not clean suppression estimates either, but the disagreement in sign is harder to attribute to a prompt that is identical for every model. 2 of the six score higher with a wrong subgraph than with no subgraph: GPT-5 moves from 0.334 → 0.387, and GPT-5 Mini moves from 0.282 → 0.356. These changes have the opposite sign from override. Haiku scores 0.027 with a wrong subgraph, while Haiku and Gemini Flash lose almost everything. Because the models disagree even on the sign, the evidence does not support treating override as an intrinsic property of structured context.

## B. What this does and does not say about a poisoned graph

Treating the 0.231 score as resilience to a poisoned graph, or as a performance floor, would understate the exposure. Under the permissive prompt, wrong-context F1 is 0.551 when the model could already answer without a graph and 0.006 when it could not. The surviving performance comes from parametric recall despite the substituted context. On a graph the model has not memorised, that source of recall is unavailable by construction.

Substitution is not a poisoning experiment. Three properties limit the supported conclusion. First, the condition simultaneously removes the question’s evidence and adds unrelated evidence, so their effects cannot be separated. Second, the substituted triples come from an unrelated question and were not selected by an adversary or chosen to conflict semantically with the correct answer. Third, the strict condition also instructs the model to rely on the supplied facts. The experiment establishes that tolerance of benign distractors does not imply robustness to wholesale context substitution. It does not measure an attack.

What follows for the retrieval path. The write path into a knowledge base carries risks that are already guarded against on the prompt path. Models supplied with wrong triples do not signal that the context is wrong, and this failure requires no adversarial text in the prompt. Because the experiment included no adversary, we treat this as a property worth controlling for and do not claim a measured vulnerability. Establishing severity would require triples selected to conflict with the correct answer, an attacker model specifying who can write to the knowledge base, and a condition that substitutes evidence without also removing it. Prior work has begun this analysis for text corpora [10] and for individual adversarial triples [11]; this study does not extend those experiments.

The difference across prompt regimes should not be interpreted as a severity gradient. Wrong-context F1 is 0.005 under the strict prompt and 0.231 under the permissive prompt, but neither level measures resilience. The strict score reflects a model declining to answer, while the permissive score reflects answers drawn from memory. The between-prompt comparison also crosses the parser boundary described in §VIII. We therefore rely on the within-prompt contrasts, not on the difference between these levels.

## VII. THREE NULLS, AND FOUR SMALLER LEVERS

Serialization, triple ordering, and subgraph size have no measurable effect at multi-hop depth. Together with retrieval precision from Section IV, they constitute the four nulls in Contribution 4. Every comparison in this section holds the prompt fixed across arms and is therefore unaffected by the scorer problem in Section VIII.

## A. Serialization affects parsing, and the active ingredient is the predicate name

Phase 1 crossed all seven formats with six models and 125 questions in entity-centric triple order, for 5,250 trials. The formats form two tiers, visible as horizontal bands in Fig. 4. The top five—prose, tabular, Turtle, N-Triples, and Cypher— cluster between F1 = 0.580 and 0.589. JSON-LD (0.503) and RDF/XML (0.495) trail them by 0.08–0.09.

This separation is confined to parsing and disappears with reasoning depth. The F1 spread across formats is 0.234 at 1- hop and 0.036 at 3-hop (Figure 3). Once a question requires multi-hop chaining, the formats are not distinguishable on this evidence. The pooled spread is 0.093, but that quantity is the maximum minus the minimum across seven groups and consequently overstates the likely value of choosing a format.

Predicate vocabulary is aliased with syntax. Verbosity does not explain the tier split. N-Triples averages 2,383 tokens per subgraph, compared with 644 for prose. It is by far the most verbose format and remains in the top tier, while the more compact RDF/XML and JSON-LD formats occupy the bottom tier. The split instead aligns exactly with predicate naming. The 5 serializers that emit human-readable predicate labels do so through prose, table columns, Cypher relationship types, or N-Triples comments. The 2 serializers that emit only Wikidata P-numbers are JSON-LD and RDF/XML, precisely the two bottom-tier formats. Among the 5 label-bearing formats, the 1-hop spread is 0.021, compared with 0.234 across all seven.

The design cannot identify a vocabulary effect independently of syntax because the two are aliased across conditions. It establishes that models perform poorly when required to infer that wdt:P166 means “award received”. That effect is real, practically useful, and plausibly accounts for the entire 1-hop spread, but it is not an isolated effect of serialization syntax. Separating the two requires a 2 × 2 comparison of syntax and predicate labelling, including labelled JSON-LD and RDF/XML alongside prose and Cypher with bare Pnumbers. That experiment was not run. The format results therefore apply only to the serializers as implemented here.

![](images/f20006488658ddb0fea768ae232237cf0e2c096ad2b3880cb2acaad5b3ff5f74.jpg)  
Fig. 3. Format F1 spread, measured as max − min across seven formats, by hop count. The spread is concentrated at 1-hop, collapses once multi-hop reasoning is required, and remains flat thereafter.

Figure 4 shows the complete format-by-model comparison. The format spread is 0.160 for mid-tier models and 0.062 for frontier models, a ratio of 2.6. This greater sensitivity among weaker models is consistent with the capability equalizer effect reported in code generation [22]. Here, however, the sensitivity is limited to 1-hop; the two tiers converge at 3-hop.

Average evidence faithfulness across all format and model pairs is 0.85. Prose reaches 0.902, tabular reaches 0.907, and RDF/XML falls to 0.798. Its hallucination rate is 0.202, compared with 0.098 for prose. This ordering supports a parsing account for the markup formats, although answer accuracy and citation faithfulness remain distinct. Turtle has the highest answer F1 among the seven formats (0.589) and the lowest faithfulness (0.796). A format can therefore support correct answers while weakening the citations offered to justify them.

![](images/944b69c2cff76237ef440761a796bcbe1b25405127a1f898443f6a0c0396764a.jpg)  
Fig. 4. F1 by format and model in Phase 1. The two-tier pattern appears across all models. The three mid-tier models in the right columns are more sensitive to format, consistent with a capability equalizer effect.

The initial Cypher faithfulness result reflected a scoring defect, not a model property. Cypher scored 0.686, initially the lowest of the seven, because models cited node variables and SCREAMING SNAKE relationship types from the input while the evaluator matched label triples. Correct citations expressed in Cypher’s own notation consequently received zero credit. The defect affected 144 trials and no other format. Resolving the variables before matching raises Cypher to 0.839. Turtle’s deficit remains after the same check and is therefore treated as a real effect.

## B. Triple ordering

Path-centric ordering places the chain triples consecutively, while random ordering scatters them through the prompt. Across 3,000 trials—two orders × two formats × six models × 125 questions—the overall ∆F1 is +0.006. The cleaner comparison is prose, which preserves the requested input order; there, the difference is −0.003. Triple ordering has no measurable effect. A follow-up using 3,000 trials and two hundred triples, padded with cross-question distractors, also finds no interaction with scale $( \Delta _ { 2 0 0 } = - 0 . 0 0 5 , \Delta _ { 5 0 } = + 0 . 0 0 9 )$ The ordering null also holds at two hundred triples.

## C. Subgraph size, and a selection artifact

Subgraph retrieval commonly treats irrelevant triples as noise that should be filtered [2], [3]. The unmatched comparison appeared to contradict that assumption, but the apparent effect came from question selection.

Across 1,842 trials and six models, subgraph size varied among three conditions: chain-only, containing just the gold reasoning path; twenty-five triples; and 50 triples. Chainonly scored 0.591, below the twenty-five-triple condition at 0.737 and the fifty-triple condition at 0.649. Faithfulness also appeared to rise from 0.888 to 0.950. Taken at face value, these values imply that pruning to the reasoning chain is harmful, a result not reported in prior work.

TABLE VIII  
SUBGRAPH SIZE BEFORE AND AFTER MATCHING QUESTION SETS (1,842 TRIALS, SIX MODELS). THE UNMATCHED COMPARISON REFLECTS QUESTION SELECTION. AMONG THE 77 QUESTIONS PRESENT IN EVERY ARM, NO CONTRAST IS DISTINGUISHABLE FROM ZERO. THE $n _ { q }$ COLUMN GIVES THE NUMBER OF QUESTIONS SCORED IN EACH ARM. F1 AND FAITHFULNESS ARE TRIAL MEANS; BRACKETS GIVE 95% BOOTSTRAP CIS OVER QUESTIONS.
<table><tr><td colspan="3">Unmatched</td><td colspan="3">Matched (77q)</td></tr><tr><td>Condition</td><td> $n _ { q }$ </td><td>F1</td><td></td><td>F1</td><td>Faith.</td></tr><tr><td>Chain-only</td><td>125</td><td>0.591 [0.532,0.672]</td><td></td><td>0.752 [0.667,0.830]</td><td>0.935</td></tr><tr><td>25 triples</td><td>77</td><td>0.737 [0.635,0.821]</td><td></td><td>0.737 [0.635,0.821]</td><td>0.950</td></tr><tr><td>50 triples</td><td>105</td><td>0.649 [0.570,0.721]</td><td></td><td>0.730 [0.627,0.815]</td><td>0.954</td></tr></table>

The arms contained different questions. The runner skips a size condition whenever a question’s subgraph is already below the target, because padding would alter the intended comparison. That rule is reasonable within a question but invalidates an unmatched comparison across arms. The conditions ran on 125 / 77 / 105 questions. Because subgraph size correlates with reasoning depth, the resulting question sets also differ in difficulty. The twenty-five-triple arm contains no question above 3 hops and averages 1.49 hops, compared with 2.11 for chain-only. The apparent size effect was caused by an easier question set.

Restricting the analysis to the 77 questions shared by all three arms reverses the ordering and removes the effect (Table VIII; Fig. 5). With trials paired by question and bootstrap resampling over questions, every contrast crosses zero on both evaluation axes. Moving from chain-only to twenty-five triples changes F1 by −0.016 (95% CI [-0.075, 0.029]) and faithfulness by +0.016 ([-0.006, 0.037]). Excluding the two OpenAI models, whose truncation rates vary across arms, produces an almost flat sequence of 0.728 / 0.727 / 0.725. The larger experiment, matched by construction across all 125 questions, agrees: 50 → 200 triples changes F1 by +0.007 ([- 0.006, 0.018]).

The range is worth stating exactly, because it is narrower than a summary suggests. The smallest arm is chain-only, which retains the full gold path and removes the distractors around it; it is not an empty prompt, and the strict no-context results elsewhere show an empty prompt behaves nothing like it. The largest is two hundred triples, and it comes from a separate experiment over a different question set with crossquestion padding. There is no paired chain-only against two hundred contrast. What the two experiments jointly support is that with the answer path retained, adding distractors up to fifty changed nothing on the shared questions, and a further expansion from fifty to two hundred changed nothing either. Within that range, these results neither confirm nor refute the assumption that irrelevant triples are noise. They do show that subgraph size is not a measurable lever at this scale, consistent with both the ordering null above and the precision null in

Matched (same 77 questions) Unmatched (125 / 77 / 105)  
![](images/f841a3b73cfab7c899d509802c221857ecd3e390a08e1df0e46ff6bcc96bb220.jpg)  
Fig. 5. Subgraph size has no measurable effect after matching question sets. Orange shows each arm on the questions for which it ran, reproducing the selection artifact. Blue restricts every arm to the 77 shared questions; the resulting column is flat and every interval overlaps. The twenty-five-triple arm already defines the common set, so its two rows contain the same trials.

Section IV.

## D. Four further levers

Evidence-first prompting (1,500 trials, six models). Requiring the evidence block before the answer increased faithfulness by +0.028 and reduced F1 by −0.021. The models identified relevant triples when required to cite them first, but better evidence selection did not produce better answers. Improved grounding did not improve reasoning.

Chain-of-thought prompting (1,500 trials, six models). Adding “think step by step, follow connections one hop at a time” to the system prompt reduced performance at every hop count above 1-hop: ∆F1 = −0.020 at 2-hop, −0.043 at 3-hop, and −0.031 at 4-hop. Step-by-step instructions compounded errors and did not aid traversal. A two-model pilot had suggested a gain at 2-hop, but that pattern did not replicate across all six models.

Dynamic oracle scoping (2,250 trials, six models). Each multi-hop SPARQL query was decomposed into per-hop steps, and each step received only its oracle-correct triples plus the preceding step’s answer. Dynamic oracle scored F1 = 0.520, below both full-context at 0.578 and chain-only at 0.604. Externally imposed decomposition removed the holistic view needed by the model. StructGPT [19] instead allows the LLM to control which triples it explores, distinguishing modeldriven exploration from the externally imposed decomposition tested here.

Sampling. A verification run at temperature = 0.7 with five samples per trial (3,750 trials across all 125 questions) reproduces the main pattern. Context helps at 1-hop $( \Delta = + 0 . 7 2 6 )$ and wrong-context override persists (F1 = 0.015). Sampling preserves the condition rankings observed under greedy decoding.

Table II in §I reports every experiment in the study and its verdict.

## VIII. A CONTRAST WE CANNOT MEASURE

All preceding comparisons either hold the prompt fixed across arms or vary it while leaving both contexts empty. The remaining contrast changes the prompt while correct context is present. It asks the question most relevant to practice: what does the grounding instruction cost when retrieval succeeds?

The experiment was run, but its estimate is an artifact of the scorer. The source of that artifact prevents a defensible point estimate from being reported.

## A. One cause

The strict prompt requires a structured response beginning with “Answer: . . . ”. The permissive prompt imposes no response structure, and models generally answer it in prose. The answer parser has a primary path for the structured form and a fallback for free text. The primary path always fires in the strict arm. It often fails in the permissive arm, where the fallback then rescues the trial.

The fallback rescues 76 trials, but those rescues are distributed entirely on one side of the comparison. It adds +0.000 to the strict arm and +0.085 to the permissive arm. No strict arm in the study contains a rescued trial because the strict prompt requires the format expected by the primary parser. Every rescue occurs in a permissive arm with a subgraph in the prompt. Across the full corpus, the fallback changes the score by +0.003.

The independent variable therefore determines the response format, and the scorer responds to that format. Enabling the fallback credits permissive responses that the primary parser cannot read. Disabling it penalises the permissive arm for format instead of content. Neither version isolates the effect of the grounding instruction on accuracy.

## B. Four symptoms

Table IX distinguishes results that cross the strict/permissive prompt boundary with context present from results that do not. Both columns are computed from the same stored trials.

The upper block measures one quantity in four ways. With the fallback enabled, the overall grounding cost is +0.045 (95% CI [+0.007, +0.075], $p = 0 . 0 0 5$ , 125 questions), which rejects the null. With the fallback disabled, the same trials produce −0.039 (95% CI [−0.079, −0.007], $p ~ = ~ 0 . 0 4 1 )$ , which also rejects the null but has the opposite sign. Opposite significant estimates from the same trials, differing only in response parsing, cannot be resolved by collecting more data.

The next three rows divide the same contrast according to what each model could answer without a graph, using the permissive no-context arm as a per-question measure of parametric knowledge. With the fallback, the pooled values form a monotone gradient from −0.012 where the model knew nothing to +0.127 where it knew the answer. That pattern suggests that the instruction withholds parametric fallback and therefore costs accuracy only when such a fallback exists. The parser comparison does not support that mechanism. Under the primary parse, the bands become −0.090, −0.034, and +0.024. They remain ordered but shift enough that the band previously

## TABLE IX

DEPENDENCE ON THE ANSWER PARSER’S FALLBACK PATH. EVERY RESULT IN THE UPPER BLOCK COMPARES A STRICT ARM WITH A PERMISSIVE ARM WHILE CORRECT CONTEXT IS PRESENT, CAUSING THE ARMS TO PRODUCE DIFFERENT RESPONSE FORMATS. RESULTS IN THE LOWER BLOCK DO NOT CROSS THAT BOUNDARY WITH CONTEXT PRESENT, SO THE PARSER CHOICE DOES NOT ACT ON THEM. THE

WRONG-VERSUS-NO-CONTEXT ROW IS THE EXCEPTION WORTH NAMING: ITS ARMS DIFFER IN PROMPT AS WELL AS IN CONTEXT (§VI), WHICH IS A SEPARATE PROBLEM FROM THE PARSER AND IS NOT REPAIRED BY THE RIGHT-HAND COLUMN. VALUES ARE ANSWER F1. THE RIGHT COLUMN RECOMPUTES THE SAME TRIALS USING ONLY THE STRUCTURED PARSE;

PERMUTATION p APPEARS IN PARENTHESES. BOTH COLUMNS ARE COMPUTED FROM THE SAME STORED TRIALS.
<table><tr><td>Result</td><td>As reported</td><td>Primary parse only</td></tr><tr><td>Grounding cost, correct ctx</td><td>+0.045 (0.005)</td><td>–0.039 (0.041)</td></tr><tr><td>Cost where it knew nothing</td><td>–0.012 (0.521)</td><td>-0.090 (0.000)</td></tr><tr><td>Cost where it knew a little</td><td>+0.042 (0.126)</td><td>–0.034 (0.298)</td></tr><tr><td>Cost where it knew it</td><td>+0.127 (0.006)</td><td>+0.024 (0.632)</td></tr><tr><td>Suppression, no facts</td><td>0.264</td><td>0.264</td></tr><tr><td>Wrong ctx vs. no ctx</td><td>-0.068</td><td>-0.073</td></tr><tr><td>Format spread, pooled</td><td>0.093</td><td>0.093</td></tr><tr><td>Precision at constant volume</td><td>+0.003</td><td>+0.003</td></tr></table>

## TABLE X

THIS TABLE IS NOT A RESULT; IT SHOWS THE ARTIFACT’S SHAPE. IT COMPARES STRICT AND PERMISSIVE PROMPTS BY HOP COUNT, WITH BOTH ARMS USING PROSE AND CORRECT CONTEXT (750 TRIALS EACH). VALUES ARE ANSWER F1, AND POSITIVE ∆ MEANS THE STRICT PROMPT SCORED WORSE. THE SCORER CANNOT MEASURE THIS CONTRAST FOR

THE REASON GIVEN IN THIS SECTION. BRACKETS CONTAIN 95% BOOTSTRAP CIS FOR ∆, RESAMPLING QUESTIONS.
<table><tr><td>Prompt</td><td>1-hop</td><td>2-hop</td><td>3-hop</td><td>4-hop</td></tr><tr><td>Strict</td><td>0.883</td><td>0.556</td><td>0.321</td><td>0.226</td></tr><tr><td>Permissive</td><td>0.871</td><td>0.541</td><td>0.455</td><td>0.425</td></tr><tr><td>∆</td><td>-0.012</td><td>-0.015</td><td>+0.133</td><td>+0.199</td></tr><tr><td>95% CI</td><td>[-0.030,0.003]</td><td>[-0.069,0.037]</td><td>[0.051,0.216]</td><td>[0.078,0.343]</td></tr><tr><td>perm. p</td><td>0.127</td><td>0.609</td><td>0.004</td><td>0.020</td></tr></table>

interpreted as free rejects the null $( p \ : = \ : 0 . 0 0 0 )$ , while the band previously interpreted as expensive does not $( p = 0 . 6 3 2 )$ . Because the substantive interpretation reverses with the parser, the ordering alone does not identify a mechanism.

Two additional analyses use the same defective contrast. The depth-specific strict-versus-permissive handicap in Table X compares strict and permissive prompts with correct context in both arms. Restricting that comparison to questions whose subgraph contains the answer does the same and yields +0.211 at 3-hop over 11 questions. Both are retained here only to show the shape of the parser-dependent artifact.

## C. Two problems that are not the parser

The covered-set restriction has another defect that would remain even after repairing the parser. Its coverage filter checks whether the gold answer string appears anywhere in the subgraph, not whether the answer is reachable from the question entity along the gold chain. Of the 11 questions in that subset, 3 fail the stricter reachability test. Those same questions produce the three largest strict-prompt costs in the set. The direction follows from the filter: when an answer is present but unreachable, the strict prompt cannot reach it from the supplied facts, while the permissive prompt can use parametric knowledge. The filter therefore selects questions that penalise the strict arm. Among the 8 questions that pass the reachability test, the contrast is +0.114, with an interval spanning zero (95% CI [−0.022, +0.271], $p = 0 . 2 5 0 )$ . At 4- hop, the covered subset contains 4 questions at $p = 0 . 2 5 0$ and supports no conclusion.

The strict prompt also bundles two interventions. It restricts the model to the supplied facts and fixes the response format. The evidence-first result shows that response format alone changes F1. Even a format-insensitive scorer would therefore estimate the combined effect of these instructions, not the cost of the grounding sentence by itself.

## D. What we would need

A valid comparison requires three changes, none of which was tested here. First, the scorer must be validated on held-out responses to read both output formats equally well. Second, the permissive prompt must request the same structured output as the strict prompt, leaving the grounding restriction as the only difference between arms. Third, $\textbf { a } 2 \times 2$ comparison of the restriction and the evidence block must separate their effects. Until those conditions are met, Table IX exhausts what this apparatus supports. A point estimate in either direction measures the parser, not the grounding instruction.

## E. A note on the corpus split

One part of the gradient remains descriptive at the corpus level, although it does not support a mechanism. The two source corpora behave differently on this contrast. Within QALD, the strict-permissive gap is large and rejects the null, ranging from +0.091 $\left( p = 0 . 0 3 8 \right) \mathrm { t o } + 0 . 3 0 6 \left( p = 0 . 0 0 1 \right)$ . Within LC-QuAD, the gap is absent in every band: −0.017, −0.011, and −0.007, with every interval spanning zero. Table XI reports the split.

LC-QuAD scores 0.710 with context, compared with 0.270 for QALD, so the data cannot distinguish a genuine absence of the contrast from a ceiling effect. The parser problem also applies to both corpus columns. The supported conclusion is limited but consequential: the pooled column combines two populations, and even a valid measurement of this contrast would not yield a single value for the corpus.

Five implications follow, ordered by their expected importance to a pipeline. Each rests on the section cited with it.

• Changing the provenance of triples around the answer has no measured benefit, while losing the answer path has a large cost. With the chain present, replacing every surrounding triple with material from an unrelated entity changes F1 by +0.003. The corresponding changes are +0.004 on questions whose subgraph demonstrably contains the answer and +0.015 on questions the model could not answer unaided. Removing the chain reduces F1 to 0.231 under a permissive prompt and 0.005 under a strict prompt. Sections IV and VI report these two results.

TABLE XI  
THIS TABLE IS NOT A RESULT; IT DISPLAYS THE ARTIFACT BY SOURCE CORPUS AND BY WHAT THE MODEL COULD ANSWER WITHOUT A GRAPH. VALUES ARE DIFFERENCES IN ANSWER F1, WITH POSITIVE VALUES INDICATING THAT THE STRICT PROMPT SCORED WORSE. BANDS ARE DEFINED BY THE PERMISSIVE NO-CONTEXT SCORE AVERAGED PER QUESTION, AND PARENTHESISED VALUES ARE QUESTION COUNTS. EVERY CELL DEPENDS ON THE PARSER CHOICE DESCRIBED IN THIS SECTION.
<table><tr><td>Unaided score (not a result)</td><td>LC-QuAD</td><td>QALD</td><td>Pooled</td></tr><tr><td>F1 = 0</td><td>–0.017 (42)</td><td>+0.028 (6)</td><td>-0.012</td></tr><tr><td> $0 < \mathrm { F } 1 \le 0 . 5$ </td><td>−0.011 (20)</td><td>+0.091 (22)</td><td>+0.042</td></tr><tr><td> $\mathrm { F 1 } > 0 . 5$ </td><td>−0.007 (20)</td><td>+0.306 (15)</td><td>+0.127</td></tr><tr><td>All questions</td><td>0.710 ctx F1</td><td>0.270 ctx F1</td><td>+0.045</td></tr></table>

The evidence for composition is more complete than the evidence for recall. Composition was varied from 1.00 context precision to 0.51, with a flat result across the gradient. Recall was observed only at the endpoints because the wrong-context condition removes all evidence for the question. The study therefore establishes a large difference between a complete answer path and no answer path, but it does not measure intermediate recall levels. Pipeline settings are one step removed from these measurements. The experiments varied the number of triples reaching the prompt and the fraction belonging to the question; they did not build an index, run similarity search, or test a retriever. The supported interpretation is that retriever output with the measured properties loses nothing, not that any particular retrieval setting has been validated. Raising k and declining to prune correspond most directly to the tested conditions. Widening the hop radius and lowering a similarity threshold are inferential extensions because they are useful only if they exchange precision for recall along the same axis. Within the tested range, that exchange costs tokens but no measured accuracy, up to two hundred triples and a context that is half wrong.

This asymmetry supports biasing retrieval toward recall instead of tuning for observable precision. Production recall is unavailable as a direct signal because identifying the triples on the answer chain requires solving the question. Optimizing the metric that can be observed would therefore optimize the term that had no measured effect here.

• Guard the write path into the knowledge base. Models do not detect a wrong subgraph, so the retrieval path needs controls comparable to those applied to the prompt path. Relative to no context, average suppression is −0.068 F1, and 2 of the six models show none. The effect is therefore smaller and less uniform than the raw correctversus-wrong comparison implies. The aggregate also masks the greater exposure on an unmemorised graph, where wrong-context F1 is 0.006.

• Match the prompt across arms in every evaluation you run. This is the least expensive control in the paper, and omitting it produced an apparently publishable but

TABLE XII

artifactual finding. If the context arm says “use only the provided facts”, the no-context arm must carry the same instruction. Section V measures the consequence of that difference. Section VIII adds a second requirement: before comparing prompt regimes, verify that the scorer reads both response formats equally well.

• Include human-readable names for predicates. The apparent one-hop format gap tracks whether a serializer emits the predicate label or only its identifier. The spread among the 5 formats that include labels is 0.021, compared with 0.234 across all seven. Because syntax and vocabulary are aliased in this design, the actionable intervention is to add labels; the evidence does not support changing syntax. A label lookup is also cheaper than replacing a serializer.

• Stop tuning four choices that produced no measurable benefit. Subgraph size is flat from chain-only to two hundred triples, triple ordering is flat at every tested scale, serialization has no measurable effect at multi-hop depth, and retrieval precision is flat at constant volume. Give the model the complete subgraph: per-hop scoping scores 0.520 even when every step receives oracle-correct triples, compared with 0.578 for full context. The evidence provides no support for withholding context and some evidence that doing so is harmful.

The study does not support a recommendation about the grounding instruction itself. Section VIII explains why. The instruction is close to free when the model has no parametric answer to suppress. When retrieval returns nothing, it separates 0.035 from 0.299. Its cost when retrieval succeeds cannot be measured by this apparatus.

## IX. LIMITATIONS

The oracle subgraphs do not guarantee oracle coverage. They are called oracle subgraphs because they are constructed from gold SPARQL instead of retrieved. That construction does not ensure that the extracted neighbourhood retains the answer. Coverage was checked by testing whether every gold answer label appeared anywhere in each question’s subgraph. Full coverage declines with depth: 46 of 46 at one-hop, 30 of 36 at two-hop, 11 of 23 at three-hop, and 4 of 11 at four-hop. Another 7 subgraphs contain none of their gold answers. The totals exclude 7 aggregate questions whose answers are counts and therefore could not appear in a subgraph.

Because depth and benchmark are aliased, coverage is more plainly described as a corpus property: 76 of 82 for LC-QuAD and 16 of 43 for QALD. The depth curve reflects those corpus-level values through the hop split. These cases are extraction failures, not evidence that the questions themselves are difficult. For example, the four-hop subgraph for “Where is the poet Alexander Pope buried?” contains triples about Alexander von Humboldt, Jenson Button and Alex Salmond because entity linking selected the wrong Alexander. The subgraphs were not repaired because all trials used them as extracted.

SPARQL STRUCTURAL CLASSIFICATION OF ALL 43 QUESTIONS AT THREE OR MORE HOPS, DERIVED FROM THE GOLD QUERIES BY THE CLASSIFIER DESCRIBED ABOVE. NO TYPE CONTRADICTS THE OTHERS, ALTHOUGH TWO ARE TOO SMALL TO TEST. STRUCTURE VARIES WITHIN A HOP COUNT, SO THIS PROXY MEASURES PREDICATE COUNT AND NOT VERIFIED REASONING DEPTH. EACH EXAMPLE GIVES THE SHAPE FOUND IN THE QUESTION SET, WITH BOUND ENTITIES OMITTED AND REPEATED SUBJECTS WRITTEN IN FULL. FAN-OUT DENOTES THREE OR MORE PREDICATES THAT CONSTRAIN ONE NODE.
<table><tr><td>Type</td><td>Count</td><td>Example Pattern</td></tr><tr><td>Sequential chain</td><td>11</td><td>?a P1 ?b . ?b P2 ?c</td></tr><tr><td>Transitive closure</td><td>17</td><td>?x P1/P279* ?y</td></tr><tr><td>Fan-out</td><td>10</td><td>?x P1 ?a . ?x P2 ?b. ?x P3 ?c</td></tr><tr><td>Mixed</td><td>5</td><td>?x P1 ?a . ?x P2 ?b. FILTER(...)</td></tr></table>

Hop count is the number of predicates in the SPARQL string, not verified sequential chain depth. Every question at three or more hops was therefore classified from the structure of its gold SPARQL. The classifier expands property paths and repeated-subject shorthand. It gives priority to transitive paths, classifies three or more predicates on one node as fanout, assigns queries with a residual filter to mixed, and treats the remainder as chains. Table XII reports all 43 of the 43 questions in this group. The structural classes are unevenly represented, which limits comparisons among them. Context benefit is positive for every class: +0.284 for sequential chains, +0.242 for transitive closures, +0.170 for fan-outs and +0.129 for mixed patterns. However, only 2 of 4 reject at 0.05, and the smallest class contains 5 questions. No structural class points in the opposite direction, but that does not establish that the results hold across classes.

Depth cannot be separated from benchmark. All oneand two-hop questions come from LC-QuAD, while all threeand four-hop questions come from QALD. Every depthstratified result is therefore also a between-corpus result, and this design cannot identify which difference produced it. The corpora vary simultaneously in answer-set size, coverage, difficulty and the sign of the context effect. This is the paper’s most consequential limitation because depth organizes most of the reported results. Separation would require both benchmarks to supply questions at every depth, which neither does.

The scorer is format-sensitive and was not validated across formats. Section VIII gives the consequence for one contrast. More generally, an end-task metric computed by a parser is comparable across conditions only if the parser is equally valid for each response format. Any independent variable that changes response format therefore requires validation on both arms before comparison. This parser was validated on neither format; the defect became visible only after the fallback was disabled and the analysis rerun.

Training-data contamination. All 125 questions concern Wikidata entities likely to have appeared in pretraining data, while graph retrieval is also deployed over graphs that have not. The permissive no-context arm measures the relevant parametric knowledge per question and is used to stratify both the precision null and the wrong-context result. It remains a proxy. An unaided score of zero establishes only that the model could not answer that question; it does not establish that the entity was absent from training. A difficult question about a famous entity falls into the same band. A genuine test requires a private graph or post-cutoff facts, neither of which was used here.

Coverage falls with depth. Full gold coverage holds for 92 of 117 answerable questions. The rate falls from 100.00% at one-hop to 36.36% at four-hop. Coverage and depth consequently move together in every depth-stratified result. The deepest questions are therefore the least reliable both because coverage is lowest and because the group is small.

Sample size at high hops. The four-hop group contains only 12 questions, and 7 show no context benefit. Its findings are directionally consistent with the three-hop results but underpowered, so no significance is claimed. The percentile bootstrap excludes zero, while the exact permutation test gives p = 0.109. No multiplicity correction is applied across the roughly forty intervals in the paper, and individual p values should be interpreted accordingly.

Oracle subgraph construction. Chain triples are identified from gold SPARQL queries and are not retrieved. The study contains no retriever, so “retrieval precision” describes a property of constructed context, not a setting on an implemented system. Real retrievers may omit chain triples in ways the sweep does not represent, and experiments with genuine retrieval may produce different results.

The precision sweep covers four models, not the full set. The strongest practical recommendation rests on 1,999 trials over 4 models from the six-model set, using only prose serialization and entity-centric order. The other models were not run, and the null is not claimed for them. Each condition also draws its wrong triples from a single donor question. Those triples are consequently more coherent than the mistakes of a real retriever may be.

Plain text is the only tested channel. Every condition supplies the subgraph as plain text in the prompt, as current GraphRAG pipelines do. The results do not cover graph context introduced during tokenization, represented by learned graph tokens, or passed through an encoder as embeddings. Because the observed format effects are parsing effects, a channel that bypasses text parsing might eliminate them or move them into the encoder. That channel remains to be tested.

No iterative baseline. Every condition injects a fixed subgraph. Systems in which the model chooses subsequent retrievals [18], [19] use a different and stronger multi-hop design. No such system is implemented here, so the static findings cannot be claimed to transfer to it.

Three of the six models are moving targets. All models were accessed through one gateway, and half of the identifiers are mutable aliases instead of pinned snapshots. The API-reported resolutions on the twenty-sixth of August, two thousand twenty-six, were gpt-5-2025-08-

07, gpt-5-mini-2025-08-07, claude-haiku-4-5- 20251001, claude-sonnet-4-6, gemini-2.5-pro, gemini-2.5-flash. The latter three cannot be pinned retrospectively, so exact replication of their results is not guaranteed.

Two output-token configurations. A total of 971 trials from the two OpenAI models returned no visible text and were rerun with a higher token cap, leaving those models represented by two configurations. These reasoning models count reasoning tokens among reported completion tokens, and the original 2,048 cap was exhausted before an answer appeared. The reruns used 16,384. Trials that had already returned text were not rerun because the cap had not bound for them and decoding was greedy. Blank responses correlate with question difficulty, so this repair is not missing-at-random and preferentially upgrades the hardest questions. Four-hop results are therefore also reported with an OpenAI-excluded sensitivity analysis.

Hop-count measurement. Hop count records predicate count in the SPARQL string, not verified sequential chain depth. Classification of all 43 high-hop questions into chains, transitive closures, fan-outs and mixed patterns finds no class that contradicts the others. The measure nonetheless remains a proxy and reflects benchmark authors’ choices about expressing relations as property paths or explicit chains.

Two prompt regimes. The experiments cover two regimes and one rephrased strict template. They do not characterize the grounding instructions between those conditions. The strict prompt also combines a restriction with a response format, and those components were not varied independently.

## X. CONCLUSION

Across 16 experiments and 30,841 trials, two of the four subgraph-related choices in a GraphRAG pipeline change the answer. They occur at the beginning and end of the pipeline.

The first is whether the answer chain reaches the prompt. Once it does, changing whether the remaining context belongs to the question is worth +0.003 F1 at constant volume. That null holds at every tested depth and among questions the model could not answer unaided. Retrieval quality therefore has two components with sharply different measured value, and only recall warrants an accuracy budget. The intervening choices produce no measurable benefit at multi-hop depth. Serialization spread falls from 0.234 at one-hop to 0.036 at three-hop, and the one-hop gap tracks the presence of a human-readable predicate name, not syntax alone. Triple order has no effect at either tested scale, and subgraph size is flat from chain-only to two hundred triples after matching the question sets.

The second consequential choice is the grounding instruction, which is also where the measurement limitations arise. With no facts in the prompt, the instruction suppresses parametric recall by a factor of 8.63. This is the largest effect in the study, but it characterizes an evaluation with empty context and not a functioning retrieval pipeline. Applying the instruction to a context arm while omitting it from the baseline produces an artifactual finding that graph context hurts at depth. With prompts matched across arms, context helps at one-, two- and three-hop and remains inconclusive at four-hop.

The practically important contrast is the cost of that instruction when correct context is available, and this apparatus cannot measure it. Prompt regime determines response format, the answer scorer handles those formats differently, and changing the parser reverses the estimated effect while rejecting the null in both directions. Resolving the contrast requires a scorer validated across both formats and a permissive prompt that requests the same output structure as the strict prompt.

## REFERENCES

[1] D. Edge et al., “From local to global: A graph RAG approach to query-focused summarization,” arXiv preprint arXiv:2404.16130, 2024. [Online]. Available: https://arxiv.org/abs/2404.16130

[2] X. Dai, Y. Hua, T. Wu, Y. Sheng, Q. Ji, and G. Qi, “Large language models can better understand knowledge graphs than we thought,” arXiv preprint arXiv:2402.11541, 2024. [Online]. Available: https://arxiv.org/abs/2402.11541

[3] C. Mavromatis and G. Karypis, “GNN-RAG: Graph neural retrieval for large language model reasoning,” arXiv preprint arXiv:2405.20139, 2024. [Online]. Available: https://arxiv.org/abs/2405.20139

[4] B. Fatemi, J. Halcrow, and B. Perozzi, “Talk like a graph: Encoding graphs for large language models,” in International Conference on Learning Representations (ICLR), 2024. [Online]. Available: https://arxiv.org/abs/2310.04560

[5] Y. Sui, M. Zhou, M. Zhou, S. Han, and D. Zhang, “Table meets LLM: Can large language models understand structured table data? a benchmark and empirical study,” in ACM International Conference on Web Search and Data Mining (WSDM), 2024. [Online]. Available: https://arxiv.org/abs/2305.13062

[6] J. Frey et al., “Benchmarking the abilities of large language models for RDF knowledge graph creation and comprehension: How well do LLMs speak Turtle?” in DL4KG Workshop at ISWC, 2023. [Online]. Available: https://arxiv.org/abs/2309.17122

[7] B. Perozzi, B. Fatemi et al., “Let your graph do the talking: Encoding structured data for LLMs,” arXiv preprint arXiv:2402.05862, 2024. [Online]. Available: https://arxiv.org/abs/2402.05862

[8] K. Wu, E. Wu, and J. Zou, “ClashEval: Quantifying the tugof-war between an LLM’s internal prior and external evidence,” in Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track, 2024. [Online]. Available: https: //arxiv.org/abs/2404.10198

[9] J. Xie, K. Zhang, J. Chen, R. Lou, and Y. Su, “Adaptive chameleon or stubborn sloth: Revealing the behavior of large language models in knowledge conflicts,” in International Conference on Learning Representations (ICLR), Spotlight, 2024. [Online]. Available: https://arxiv.org/abs/2305.13300

[10] W. Zou, R. Geng, B. Wang, and J. Jia, “PoisonedRAG: Knowledge corruption attacks to retrieval-augmented generation of large language models,” in USENIX Security Symposium, 2025. [Online]. Available: https://arxiv.org/abs/2402.07867

[11] T. Zhao et al., “RAG safety: Exploring knowledge poisoning attacks to retrieval-augmented generation,” arXiv preprint arXiv:2507.08862, 2025. [Online]. Available: https://arxiv.org/abs/2507.08862

[12] W. Zhou, S. Zhang, H. Poon, and M. Chen, “Context-faithful prompting for large language models,” in Findings of the Association for Computational Linguistics: EMNLP, 2023. [Online]. Available: https://arxiv.org/abs/2303.11315

[13] Y. Huang, S. Chen, H. Cai, and B. Dhingra, “To trust or not to trust? enhancing large language models’ situated faithfulness to external contexts,” in International Conference on Learning Representations (ICLR), 2025. [Online]. Available: https://arxiv.org/abs/2410.14675

[14] B. Bi, S. Huang, Y. Wang et al., “Context-DPO: Aligning language models for context-faithfulness,” arXiv preprint arXiv:2412.15280, 2024. [Online]. Available: https://arxiv.org/abs/2412.15280

[15] B. Bi, S. Liu, Y. Wang, Y. Xu, L. Mei, J. Fang, and X. Cheng, “Parameters vs. context: Fine-grained control of knowledge reliance in language models,” arXiv preprint arXiv:2503.15888, 2025. [Online]. Available: https://arxiv.org/abs/2503.15888

[16] P. Huang, Z. Liu, Y. Yan et al., “ParamMute: Suppressing knowledgecritical FFNs for faithful retrieval-augmented generation,” in Advances in Neural Information Processing Systems (NeurIPS), 2025. [Online]. Available: https://arxiv.org/abs/2502.15543

[17] M. Mandarapu and S. Kunkunuru, “Knowledge-graph grounding helps LLMs only for out-of-training knowledge: A controlled study on clinical question answering,” arXiv preprint arXiv:2606.22419, 2026. [Online]. Available: https://arxiv.org/abs/2606.22419

[18] J. Sun, C. Xu, L. Tang, S. Wang, C. Lin, Y. Gong, L. M. Ni, H.-Y. Shum, and J. Guo, “Think-on-graph: Deep and responsible reasoning of large language model on knowledge graph,” in International Conference on Learning Representations (ICLR), 2024. [Online]. Available: https://arxiv.org/abs/2307.07697

[19] J. Jiang, K. Zhou, Z. Dong, K. Ye, W. X. Zhao, and J.-R. Wen, “StructGPT: A general framework for large language model to reason over structured data,” in Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP), 2023. [Online]. Available: https://arxiv.org/abs/2305.09645

[20] X. Shan and Y. Luo, “Bounded path context: A controlled study of visible path history in LLM-based knowledge graph question answering,” arXiv preprint arXiv:2605.26645, 2026, code: https://github.com/AndyShan11/Bounded-Path-Context. [Online]. Available: https://arxiv.org/abs/2605.26645

[21] M.-V. Nguyen et al., “Direct evaluation of chain-of-thought in multi-hop reasoning with knowledge graphs,” in Findings of the Association for Computational Linguistics: ACL, 2024. [Online]. Available: https://arxiv.org/abs/2402.11199

[22] A. Canedo, “Architecture as capability equalizer for coding agents,” 2026. [Online]. Available: https://github.com/arquicanedo/ architecture-as-equalizer

[23] D. Vrandeciˇ c and M. Kr´ otzsch, “Wikidata: A free collaborative knowl-¨ edgebase,” Communications of the ACM, vol. 57, no. 10, pp. 78–85, 2014.

[24] M. Dubey, D. Banerjee, A. Abdelkawi, and J. Lehmann, “LC-QuAD 2.0: A large dataset for complex question answering over Wikidata and DBpedia,” in International Semantic Web Conference (ISWC), 2019. [Online]. Available: https://jens-lehmann.org/files/2019/iswc lcquad2. pdf

[25] R. Usbeck et al., “QALD-10 — the 10th challenge on question answering over linked data,” in Semantic Web Challenges (SemWebEval), 2023.

[26] D. Beckett, T. Berners-Lee, E. Prud’hommeaux, and G. Carothers, “RDF 1.1 Turtle,” W3C, Recommendation, 2014. [Online]. Available: https://www.w3.org/TR/turtle/

[27] G. Carothers and A. Seaborne, “RDF 1.1 N-Triples,” W3C, Recommendation, 2014. [Online]. Available: https://www.w3.org/TR/ n-triples/

[28] M. Sporny, G. Kellogg, and M. Lanthaler, “JSON-LD 1.0,” W3C, Recommendation, 2014. [Online]. Available: https://www.w3.org/TR/ json-ld/

[29] D. Beckett, “RDF/XML syntax specification (revised),” W3C, Recommendation, 2004. [Online]. Available: https://www.w3.org/TR/ rdf-syntax-grammar/