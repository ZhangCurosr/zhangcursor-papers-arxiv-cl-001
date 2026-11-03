# Every Clause Counts: Instruction-Following Reranking by Multiplying Small, Answerable Questions

S. M. Mahbub Uddin Ahmed United International University Dhaka, Bangladesh sahmed221068@bscse.uiu.ac.bd

## Abstract

People often attach instructions to their search queries, asking for pages written for beginners, documents in a certain language or official sources only. Most rerankers were trained to judge topical relevance and treat such instructions as a few extra words. On the InfoSearch benchmark, even GPT-4o used as a list-wise reranker reaches an average WISE (Weighted Instruction Sensitivity Evaluation) of only 33.5. We present DMReranker (Decision-Model Reranker), a zero-shot reranker that splits an instructed query into a topic and a short list of typed clauses. A small judge model answers one narrow question per clause, and the answer probabilities are multiplied into a single score. Nothing is trained, and the configuration was frozen before the test runs. Reranking the same top-100 candidates as the published baselines, DMReranker reaches a macro WISE of 50.9 over all six InfoSearch dimensions while keeping original-mode nDCG@10 at 82.9. It trails GPT-4o on the stricter SICR (Strict Instruction Compliance Ratio) metric, 19.2 against 32.1. We show that this gap follows from how a product of probabilities behaves when SICR compares raw scores across runs. Computed from ranks, our SICR is 67.0. On the same judge, splitting the instruction raises WISE from 32.9 to 47.6 over a single combined question, while raw-score SICR falls. Scoring all six dimensions cost about \$10 at list price.

## 1 Introduction

A search query used to be a handful of keywords. Today it is often a sentence with conditions attached. A developer asks how to tune indexes in a database and adds that only the official documentation is useful. A student wants an explanation of photosynthesis that a twelve-year-old could follow. A reader wants news coverage in Spanish. In each case the user cares about two things at once,

Abdullah Al Musaddiq Islamic University of Technology Gazipur, Bangladesh musaddiq@iut-dhaka.edu

![](images/b63cf350528a523b179d80c6ebe81ad8225faa21a8cffe69a9b792d818f07c14.jpg)  
Figure 1: A relevance reranker puts the expert forum post first, because it is slightly more on topic. DMReranker asks one question per clause ofthe instruction and multiplies the answers, so the official manual for non-experts moves to the top. Each colour links a clause of the instruction to the question that checks it. Probabilities are illustrative. The bars show average WISE on InfoSearch, measuredfor DMReranker and as published for the others.

whether a document is about the right topic and whether it has the right properties. A document can be perfectly on topic and still be useless because it is a forum post, written for experts, or in the wrong language.

Rerankers are the natural place to handle such conditions, because they read the query and each candidate document together. Yet most rerankers were built to estimate topical relevance. Crossencoders such as monoBERT and monoT5 are trained on relevance labels (Nogueira and Cho, 2019; Nogueira et al., 2020), and large language model (LLM) rerankers prompted or distilled for ranking inherit the same goal (Sun et al., 2023; Pradeep et al., 2023a,b). Recent benchmarks make the cost of this visible. FollowIR showed that retrieval models largely ignore instructions that change what counts as relevant (Weller et al., 2025a). InfoSearch went further and tested six document-level conditions, namely audience, keyword, format, language, length and source (Zhou et al., 2025), and scored them with two metrics, WISE (Weighted Instruction Sensitivity Evaluation) and SICR (Strict Instruction Compliance Ratio). On InfoSearch, the strongest reranker reported, GPT-4o used list-wise, reaches an average WISE of 33.5, and the best open model reaches 14.5. Several rerankers score below zero, which means that adding the instruction moved the ranking in the wrong direction.

We hypothesize that a large part of the problem is the form of the question that the model is asked. A list-wise reranker must weigh topic and conditions together and produce a single ordering, with no explicit place for a condition to veto a document. Our approach breaks the decision into pieces that we expect a model to answer more reliably. We first turn the instruction into a topic and a set of typed clauses. Then, for each candidate, we ask a judge model a series of narrow questions. Is this document relevant to the topic? Which audience is it written for? Is it official documentation? Each answer comes back as a probability. Finally we multiply the probabilities. A document that fails any clause receives a low score, no matter how relevant it is, and a document that satisfies every clause keeps its relevance score almost intact. Figure 1 illustrates the idea on one query, and Figure 2 shows the full architecture.

This design has three practical benefits. It needs no training data, since each question is one that a general instruction-tuned judge already answers well. Document attributes do not depend on the query, so we profile each document once and reuse the profile for every query that touches it. Finally, every score can be traced back to the individual answers, so we can study what each component contributes by recombining saved answers, without calling a model again.

We describe DMReranker (Decision-Model Reranker), which parses an instruction into typed clauses and combines per-clause probabilities with a fixed product rule (Section 3). On InfoSearch it gives large gains in WISE and p-MRR over the published baselines on the same candidate pool, with no loss in plain relevance (Section 5). Section 6 reports offline ablations, a comparison with one combined question to the same judge, the reason for its lower SICR, and the cost of the run.

## 2 Related Work

Instruction following in retrieval. Early work on instructions in retrieval used them to describe the task, such as “retrieve a passage that answers this question” (Asai et al., 2023; Su et al., 2023). Later benchmarks asked whether models follow instructions that change the meaning of relevance for a single query. FollowIR measures this with p-MRR, which compares rankings before and after the instruction is changed (Weller et al., 2025a). INSTRUCTIR attaches user-specific instructions to queries (Oh et al., 2024). Promptriever trains a bi-encoder retriever to respond to prompts (Weller et al., 2025b). InfoSearch, which we use, isolates six document-level conditions and evaluates each in an original, an instructed and a reversely instructed mode (Zhou et al., 2025). Its two metrics, WISE and SICR, reward a model only when the ranking moves in the direction the instruction asks for.

LLM rerankers. LLMs can rerank point-wise, by scoring one document at a time, pairwise, or list-wise, by ordering a window of documents (Sun et al., 2023; Qin et al., 2024). Setwise prompting reduces the calls of pairwise ranking (Zhuang et al., 2024b), and Rank1 trains a reranker to reason before it answers (Weller et al., 2025c). RankVicuna and RankZephyr distil list-wise ranking into open 7B models (Pradeep et al., 2023a,b). Point-wise rerankers that read out label probabilities, rather than a single yes or no, give finer and better calibrated scores (Zhuang et al., 2024a). Our relevance question follows this idea with a four-level scale. The difference is that we do not ask one model to fold the conditions into its relevance judgement. Each condition receives its own question and its own probability.

Decomposing the decision. Splitting a complex task into simpler steps is a common way to make LLM behaviour more reliable, both for reasoning (Zhou et al., 2023; Khot et al., 2023) and for evaluation, where a long text is checked one atomic fact at a time (Min et al., 2023). In retrieval, the closest analogue is filtering by metadata, which works when documents carry clean fields such as language or date. Web documents rarely do. Our document profile plays the role of these missing fields, filled in by a model and kept as probability distributions rather than hard labels, so that an uncertain attribute lowers a score without removing the document outright. Multiplying beliefs about separate conditions is old, as in the #and operator of inference networks (Turtle and Croft, 1991), soft conjunction in extended Boolean retrieval (Salton et al., 1983) and products of experts (Hinton, 2002). What is new here is how the factors are obtained.

![](images/174be1d0c2488fdbd8f308b4ae67a44aa4cc6afb5e4e97d6d5d20cf06b0034da.jpg)  
Figure 2: Architecture ofDMReranker. (1) An LLM parses the instructed query into a topic and typed clauses, and a second pass removes clauses it cannot support. (2) For each query and candidate, a small judge model answers a four-level relevance question and one yes/no question per remaining rule. (3) The same judge profiles every document once over sixfixed attributes. When its confidence on an attribute that a rule needs is below 0.3, a larger judge answers again. Plain code multiplies the probabilities into the final score s(d). Blue boxes call a model, the grey box does not.

## 3 Method

DMReranker takes an instructed query q and a list of candidate documents and returns a score s(d) for each candidate. It works in three steps. Only the first two call a model.

## 3.1 Parsing the instruction

A parser model (Gemma-4-26B-A4B, a mixtureof-experts model with about 4B active parameters (Google DeepMind, 2026)) reads the query and returns a structured policy p in JSON. The policy has four parts. The topic is one sentence that states the information need with every condition removed. Attribute rules restrict a document attribute to a set of allowed values, for example audience ∈ {general}. Other rules cover conditions that do not fit an attribute. Each is a yes/no question about the document, together with the answer that keeps it, for example “Does the document mention FooDB 3?” with keep-if yes. Metadata rules cover dates and versions, such as “published after 2023” or “most recent first”.

A second call to the same model checks the parse. It sees the query and the parsed rules and votes to keep or drop each rule, and it may rewrite the topic. It cannot add rules or change their content, so it can only make the policy more conservative. This step exists because a parser that invents a rule does more damage than one that misses a rule. An invented gate can push the only relevant document out of the top ten, while a missed rule leaves the ranking no worse than a plain relevance reranker.

## 3.2 Asking the judge

The judge is Clef-flash, a 9B model served on Cloudflare Workers AI that answers typed questions about a text and returns a probability distribution over the possible answers (Cloudflare, 2026). It supports three question types that we use, a yes/no question, a choice among named options, and a score on an ordered scale.

Document profile. For each candidate document d we ask six choice questions in one request, one per attribute: source, format, audience, length, language and status. Each attribute has a fixed set of options. The answer is a distribution P(o | d)

over the options of that attribute, together with a confidence value. These questions do not mention the query, so the profile of a document is computed once and shared by every query whose candidate list contains it. We wrote the attribute list with the six InfoSearch conditions in mind, and status was added for versioned documentation, so the profile fits this benchmark more closely than a list written for a new domain would. Its language options, for instance, are only English, Chinese and other. If a rule depends on an attribute whose confidence is below $\tau = 0 . 3$ , the larger Clef model (27B) answers the profile again for that document, and its distribution replaces only the low-confidence attributes.

Relevance. We ask the judge how well d addresses the parsed topic, on four levels from “does not address $\mathrm { i t } ^ { \prime \prime }$ to “fully answers it”. We use the probability that the level is at least 2,

$$
P _ { \mathrm { u s e } } ( d ) = P ( \ell \geq 2 \mid d , \mathrm { t o p i c } ) .\tag{1}
$$

Asking about the parsed topic rather than the raw query matters. The raw query still contains the instruction, and a judge that sees “for non-experts” in the relevance question may fold the condition into its relevance estimate, which would count the condition twice.

Other rules. Each other rule r becomes one yes/no question. If $P ( \mathrm { y e s } )$ is the judge’s answer, the keep probability is $P _ { \mathrm { k e e p } } ( d , r ) = P ( \mathrm { y e s } )$ when the rule keeps on yes, and 1− $P ( \mathrm { y e s } )$ when it keeps on no.

Metadata. Date and version rules become yes/no questions in the same way, and “most recent” becomes a pairwise recency factor between candidates. We write the product of these factors as $m ( d )$ . InfoSearch does not grade documents by date, so $m ( d ) = 1$ for most queries. A few instructions still ask for “the latest” or “current” information, which the parser turns into a recency rule in 46 of the 4,791 query-mode rows (43 on Source). We kept these rules and measure their effect in Section 6.1.

## 3.3 Combining the answers

For an attribute rule a with allowed set $A _ { a }$ , the probability that d satisfies it is the mass the profile puts on the allowed options, $P _ { \mathrm { a l l o w e d } } ( d , a ) =$

$\textstyle \sum _ { o \in A _ { a } } P ( o \mid d )$ . The final score is

$$
\begin{array} { l } { { \displaystyle s ( d ) = P _ { \mathrm { u s e } } ( d ) \cdot \prod _ { r } P _ { \mathrm { k e e p } } ( d , r ) } } \\ { { \displaystyle ~ \cdot ~ m ( d ) \cdot \prod _ { a } P _ { \mathrm { a l l o w e d } } ( d , a ) . } } \end{array}\tag{2}
$$

Equation 2 treats the clauses as independent conditions that must all hold, so the score reads as the probability that the document is useful and satisfies every clause. There are no weights, thresholds or learned parameters. A document that clearly fails one clause gets a factor near zero and drops out of the top of the ranking. A document that clearly passes every clause keeps a score close to its relevance probability. Figure 1 shows an example.

Every question, answer distribution and factor is written to a trace file. Since Equation 2 is plain arithmetic over these saved values, we can drop or replace any factor afterwards and recompute the rankings without a single new model call. We use this for the ablations in Section 6.1.

## 4 Experimental Setup

Benchmark. InfoSearch has 600 core queries, 100 per dimension, and 6,392 documents (Zhou et al., 2025). Each core query comes with instructed versions that add a condition, and reversely instructed versions that ask for the opposite. The paper reports 1,598 instructed queries. The public release contains 1,597, because one Keyword query is missing, and we evaluate on the release as is. We report all six dimensions, Audience (210 instructed queries), Keyword (287), Format (300), Language (200), Length (300) and Source (300).

Candidates. Following the InfoSearch reranking protocol, every system reranks the published top-100 candidates from e5-mistral (Wang et al., 2024), in all three modes. Because the candidates and relevance labels are identical, the reranker rows in our tables are same-pool comparisons. BM25 and e5-mistral rows are full-corpus retrieval results and are shown for reference only.

Metrics. We use the official InfoSearch evaluation code for WISE, SICR and p-MRR, and pytrec\_eval for nDCG@10 (Järvelin and Kekäläinen, 2002). WISE rewards a system when documents that satisfy the condition move up under the instruction and move down under the reverse instruction, with more weight on changes near the top of the ranking. SICR is stricter. It counts a query as followed only when the condition-satisfying document gains in the instructed mode and loses in the reversed mode. p-MRR, taken from FollowIR, measures the change in reciprocal rank between modes (Weller et al., 2025a). For each dimension we compute 95% confidence intervals for our scores with a bootstrap over instructed queries (2,000 resamples) (Efron and Tibshirani, 1993; Smucker et al., 2007).

Baselines. We did not rerun any baseline. All baseline numbers come from Tables 1 and 2 of the InfoSearch paper, which uses the same candidates, labels and evaluation code (Zhou et al., 2025). They cover point-wise rerankers (Mistral-7B-Instruct (Jiang et al., 2023), Llama-3.1-8B (Grattafiori et al., 2024) and FollowIR-7B (Weller et al., 2025a)) and list-wise rerankers (Mistral-7B-Instruct, Zephyr-7B-β (Tunstall et al., 2023), RankVicuna (Pradeep et al., 2023a), RankZephyr (Pradeep et al., 2023b) and GPT-4o (OpenAI, 2024)). InfoSearch reports p-MRR and nDCG@10 only as six-dimension averages, so we compare those as averages.

Implementation. The parser (@cf/google/gemma-4-26b-a4b-it) and both judges (@cf/cloudflare/clef-flash and @cf/cloudflare/clef) are called through Cloudflare Workers AI. All prompts, question wordings, the confidence threshold and the combination rule were fixed before the test runs (Appendix A gives their full text), and we record a hash of this configuration with every result. During development we tried the pipeline on 60 instructed queries each from Format and Source, which are also in the test set, and lowered τ from 0.5 to 0.3 to save cost. Without these 120 queries, WISE is 43.8 on Format and 67.3 on Source, against 45.3 and 65.2 on all queries. Responses are cached by request, so a rerun of the full evaluation makes no new calls and returns identical numbers.

## 5 Results

Instruction following. Table 1 gives the main comparison. Averaged over the six dimensions, DMReranker reaches a WISE of 50.9. GPT-4o reaches 33.5 and RankZephyr, the strongest open reranker by WISE, reaches 14.5. The 95% bootstrap intervals for our WISE are [44.2, 59.6] on Audience, [52.5, 63.9] on Keyword, [38.1, 51.7] on Format, [38.0, 55.2] on Language, [31.6, 44.0] on Length and [59.3, 71.1] on Source. The baselines are published only as averages, so we cannot test the differences, and the comparisons below are between point estimates.

The average hides a clear pattern across dimensions, which Figure 3 makes easy to see. The largest gains are on Audience (52.1 against 7.4 for GPT-4o), Format (45.3 against 21.9), Length (37.9 against 10.2) and Source (65.2 against 45.2). These are conditions on what kind of document something is, which most rerankers barely react to. Our document profile asks about them directly. On Keyword and Language, GPT-4o is ahead (63.0 against 58.3, and 53.1 against 46.7). Both are conditions that a strong LLM can check by reading the text, and GPT-4o has seen the whole list of candidates at once.

SICR. On SICR the picture is different. DMReranker is ahead of GPT-4o only on Audience and Format, and its average of 19.2 is below GPT-4o (32.1) and the list-wise Mistral-7B (22.0). Source, our best dimension by WISE, is one of our worst by SICR (10.0). Section 6.3 explains why.

Relevance is preserved. Table 2 shows that following instructions does not cost plain ranking quality. In the original mode, where there is no instruction, our nDCG@10 averages 82.9, close to GPT-4o (83.8) and above every open reranker in the InfoSearch table. In the instructed mode our nDCG@10 of 78.3 is above GPT-4o (74.2). Our p-MRR of 52.7 is far above the published values, which top out at 15.0 for GPT-4o.

## 6 Analysis

## 6.1 Which component does the work

Because Equation 2 is computed from saved answers, we can remove a factor and recompute the rankings without new model calls. Table 3 shows four variants, and Figure 4 in Appendix B breaks WISE down by dimension. With relevance only, WISE falls to about zero on every dimension (−4.0 on average), which confirms that the gains come from the clause factors and not from a better relevance model. Since the relevance question already uses the parsed topic, this variant is a strong relevance reranker that simply ignores the conditions.

The two kinds of clause split the work in an interpretable way. Removing the attribute gates collapses Audience (−1.2), Language (−9.1) and Source (−0.9), halves Format (20.6) and Length (17.6), but leaves Keyword unchanged at 58.3. Removing the yes/no rules does the opposite. Keyword drops to 4.1, while Audience and Language barely move and Source loses 4.7 points. On Keyword the parser keeps a required term in the topic, where the relevance question checks it, and turns an excluded term into a yes/no rule (286 of 287 reversed queries). Most other conditions go to attributes. Format and Length use both, which is why they lose ground under either removal.

<table><tr><td></td><td>Aud.</td><td>Key.</td><td>Form.</td><td>Lang.</td><td>Len.</td><td>Src.</td><td>Avg.</td></tr><tr><td>WISE</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>BM25†</td><td>-3.0</td><td>-42.1</td><td>-2.8</td><td>-7.2</td><td>-7.5</td><td>2.0</td><td>-12.0</td></tr><tr><td>e5-mistral†</td><td>-7.3</td><td>-44.5</td><td>-19.9</td><td>0.1</td><td>-13.4</td><td>-13.0</td><td>-16.3</td></tr><tr><td>Mistral-7B (point-wise)</td><td>-8.9</td><td>34.5</td><td>-9.3</td><td>21.4</td><td>-12.6</td><td>-0.3</td><td>4.1</td></tr><tr><td>Llama-3.1-8B (point-wise)</td><td>-6.2</td><td>38.7</td><td>-9.5</td><td>29.0</td><td>-5.9</td><td>40.2</td><td>14.4</td></tr><tr><td>FollowIR-7B (point-wise)</td><td>-2.3</td><td>47.7</td><td>-2.3</td><td>20.9</td><td>-2.6</td><td>18.8</td><td>13.4</td></tr><tr><td>Mistral-7B (list-wise)</td><td>-6.3</td><td>46.0</td><td>-6.6</td><td>7.6</td><td>-1.9</td><td>10.0</td><td>8.1</td></tr><tr><td>Zephyr-7B-β (list-wise)</td><td>-2.7</td><td>14.1</td><td>-13.9</td><td>-6.9</td><td>-5.7</td><td>-3.9</td><td>-3.2</td></tr><tr><td>RankVicuna (list-wise)</td><td>-2.5</td><td>-8.5</td><td>-9.8</td><td>-11.8</td><td>-4.3</td><td>-2.2</td><td>-6.5</td></tr><tr><td>RankZephyr (list-wise)</td><td>7.4</td><td>53.9</td><td>7.8</td><td>10.6</td><td>7.8</td><td>-0.3</td><td>14.5</td></tr><tr><td>GPT-4o (list-wise) DMReranker (ours)</td><td>7.4</td><td>63.0</td><td>21.9 45.3</td><td>53.1</td><td>10.2</td><td>45.2</td><td>33.5</td></tr><tr><td></td><td>52.1</td><td>58.3</td><td></td><td>46.7</td><td>37.9</td><td>65.2</td><td>50.9</td></tr><tr><td>SICR</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>BM25†</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>e5-mistral†</td><td>2.9</td><td>0.0</td><td>0.0</td><td>10.5</td><td>0.0</td><td>3.3</td><td>2.8</td></tr><tr><td>Mistral-7B (point-wise)</td><td>1.4</td><td>28.6</td><td>2.7</td><td>6.5</td><td>4.7</td><td>4.7</td><td>8.1</td></tr><tr><td>Llama-3.1-8B (point-wise)</td><td>6.2</td><td>38.3</td><td>10.0</td><td>22.0</td><td>2.7</td><td>36.7</td><td>19.3</td></tr><tr><td>FollowIR-7B (point-wise)</td><td>3.3</td><td>27.2</td><td>7.0</td><td>19.5</td><td>1.7</td><td>16.3</td><td>12.5</td></tr><tr><td>Mistral-7B (list-wise)</td><td>10.5</td><td>59.2</td><td>9.7</td><td>23.0</td><td>8.0</td><td>21.7</td><td>22.0</td></tr><tr><td>Zephyr-7B-β (list-wise)</td><td>1.0</td><td>27.5</td><td>8.0</td><td>10.5</td><td>2.0</td><td>3.0</td><td>8.7</td></tr><tr><td>RankVicuna (list-wise)</td><td>5.2</td><td>10.5</td><td>3.3</td><td>4.5</td><td>2.3</td><td>8.0</td><td>5.6</td></tr><tr><td>RankZephyr (list-wise)</td><td>4.3</td><td>42.5</td><td>1.0</td><td>5.5</td><td>4.3</td><td>5.3</td><td>10.5</td></tr><tr><td>GPT-4o (list-wise)</td><td>15.2</td><td>60.6</td><td>11.3</td><td>55.5</td><td>10.3</td><td>39.7</td><td>32.1</td></tr><tr><td>DMReranker (ours)</td><td>19.5</td><td>43.6</td><td>14.0</td><td>20.5</td><td>7.7</td><td>10.0</td><td>19.2</td></tr></table>

Table 1: WISE and SICR per dimension on InfoSearch. All rerankers reorder the same e5-mistral top-100 candidates. Baseline values arefrom Zhou et al. (2025). “Avg.” is the six-dimension average. Bold marks the best same-pool reranker in each column. †Full-corpus retrieval, not a same-pool comparison.

<table><tr><td rowspan="2"></td><td colspan="3">nDCG@10</td></tr><tr><td>p-MRR</td><td>Ori.</td><td>Ins. Rev.</td></tr><tr><td>BM25†</td><td>7.0</td><td>47.5</td><td>39.1 38.5</td></tr><tr><td>e5-mistral†</td><td>4.0</td><td>78.3 64.3</td><td>66.0</td></tr><tr><td>FollowIR-7B (pw)</td><td>4.1</td><td>72.4 66.3</td><td>65.5</td></tr><tr><td>Llama-3.1-8B (pw)</td><td>11.5</td><td>74.8</td><td>66.8 65.4</td></tr><tr><td>RankZephyr (lw)</td><td>0.7</td><td>75.0</td><td>63.5 64.7</td></tr><tr><td>GPT-4o (1w)</td><td>15.0</td><td>83.8 74.2</td><td>74.2</td></tr><tr><td>DMReranker (ours)</td><td>52.7</td><td>82.9</td><td>78.3 73.8</td></tr></table>

Table 2: p-MRR and nDCG@10 in the original, instructed and reversed modes. All values are sixdimension averages, as reported by InfoSearch. pw: point-wise, lw: list-wise. †Full-corpus retrieval.

We also checked the parse and the profile directly. The parser added no rule to any of the 1,597 original-mode queries. Against each document that the instruction makes irrelevant, the clause factors rank the document that meets it higher in

83.2% of pairs in the instructed mode and 90.0% in the reversed mode. Instructed Keyword is lowest (53.3%), since its term stays in the topic, and the other dimensions range from 85.1% (Length) to 99.0% (Language).

The metadata factor is the one component that does not help. Removing it leaves five dimensions within 0.5 WISE points, but on Source it raises WISE from 65.2 to 67.6 and p-MRR from 74.8 to 81.7. The recency rules on Source come from instructions that ask for the latest information, so the parser read them correctly. InfoSearch, however, labels Source documents by the kind of source and not by date, and reordering candidates by date adds noise to that judgement. The factor was also the most expensive part of the run (Section 6.4). We report the frozen system as it was run and do not remove the factor after the fact.

## 6.2 Clause questions against one combined question

The ablations show that the clause factors carry the gains, but they do not show that splitting the instruction is better than asking about it directly. Relevance only ignores the instruction, and the published baselines use other models. We therefore built a single-question baseline on the same judge.

![](images/309351c84837dd41267e1df9024cd86af1450a3bf79c92201d5c32bfb8c50682.jpg)

Figure 3: Per-dimension WISE (top) and SICR (bottom) on InfoSearchfor DMReranker and three cited rerankers. DMReranker leads by a wide margin on Audience, Format, Length and Source WISE. GPT-4o leads on Keyword and Language WISE and on SICR overall. Baseline values arefrom Zhou et al. (2025).
<table><tr><td>Variant</td><td>WISE</td><td>SICR</td><td>p-MRR</td><td>nDCG@10 Ins.</td><td>nDCG@10 Rev.</td></tr><tr><td>VO Full system</td><td>50.9</td><td>19.2</td><td>52.7</td><td>78.3</td><td>73.8</td></tr><tr><td>V1 Relevance only</td><td>-4.0</td><td>4.2</td><td>1.6</td><td>64.8</td><td>71.0</td></tr><tr><td>V2 No metadata factor</td><td>51.3</td><td>19.4</td><td>53.9</td><td>78.5</td><td>73.9</td></tr><tr><td>V3 No attribute gates</td><td>14.2</td><td>15.0</td><td>2.2</td><td>65.0</td><td>70.7</td></tr><tr><td>V4 No yes/no rules</td><td>34.9</td><td>12.3</td><td>52.3</td><td>78.0</td><td>74.6</td></tr><tr><td>V5 Mean instead of product (post hoc)</td><td>47.4</td><td>33.0</td><td>57.2</td><td>77.2</td><td>73.5</td></tr></table>

Table 3: Offline ablations, averaged over the six dimensions. Each variant recombines the saved answers ofthefull system, so no new model calls were made. V5 was tried after seeing test results and is not part ofthe system.

For each query and candidate, Clef-flash answers one yes/no question, whether the document is useful for the query including its instruction, and the probability of yes is the score. There is no parse, no profile and no product, and the question is yes/no rather than our four-level scale.

To keep the cost small we ran this baseline on a subset of 50 instructed queries per dimension, 300 in total, drawn with a fixed random seed. The subset was fixed and recorded before any baseline call was made. Both systems are scored on exactly these queries in all three modes, and we compare them with a paired bootstrap over queries (2,000 resamples). The baseline needed 85,170 judge calls and cost \$1.22 at list price.

Table 4 gives the result. Splitting the instruction raises WISE from 32.9 to 47.6 and p-MRR from 20.0 to 53.0, and both intervals exclude zero. DMReranker also has better nDCG@10 in all three modes, but in the original mode, which has no clauses, the gain comes from the four-level question and the parsed topic, not from splitting. On SICR the result reverses, and the single question is clearly ahead (37.0 against 20.3). Per dimension, the WISE gain is large on Audience (+46.9) and Format (+25.6) and within noise elsewhere, with the single question ahead on Source by 10.5 points. With 50 queries per dimension these intervals are wide, so we treat the macro difference as the main result.

<table><tr><td>Metric</td><td>Single</td><td>Ours</td><td>Diff.</td><td>95% CI</td></tr><tr><td>WISE</td><td>32.9</td><td>47.6</td><td>+14.7</td><td>[7.3, 22.2]</td></tr><tr><td>SICR</td><td>37.0</td><td>20.3</td><td>-16.7</td><td>[-22.7, -10.3]</td></tr><tr><td>p-MRR</td><td>20.0</td><td>53.0</td><td>+33.0</td><td>[28.4, 38.1]</td></tr><tr><td>nDCG@10 Ori.</td><td>77.9</td><td>81.6</td><td>+3.7</td><td>[1.4, 6.0]</td></tr><tr><td>nDCG@10 Ins.</td><td>72.5</td><td>76.7</td><td>+4.1</td><td>[0.8, 7.5]</td></tr><tr><td>nDCG@10 Rev.</td><td>66.8</td><td>74.0</td><td>+7.2</td><td>[3.9, 10.4]</td></tr></table>

Table 4: DMReranker against a single combined question to the same judge (Clef-flash), on afixed random subset of300 instructed queries, 50 per dimension. “Single” is the single question and “Ours” is DMReranker. Differences are DMReranker minus the single question, with paired bootstrap 95% intervals.

## 6.3 Why SICR lags

SICR asks whether the score of a document that satisfies the condition rises when the instruction is added. In our model every clause factor lies between 0 and 1, so adding a clause can never raise a score by itself. A matching document only gains if its relevance probability goes up, and that rarely happens, because the parser strips the condition from the topic and the topic is therefore nearly the same in both modes. Attribute gates help the matching document by pushing the other documents down, which is exactly what WISE and p-MRR measure, but its own score usually stays the same or falls a little. On Language, 159 of the 200 instructed queries have an SICR of zero, and in most of them the instructed-mode score of the matching document is slightly lower than its original-mode score.

Two further results support this reading. The single question asks about the instruction directly, so a matching document can gain score, and its SICR is much higher than ours (Section 6.2). The post-hoc variant V5 (Table 3) replaces the product with a mean, which lets a passed clause add to the score. SICR rises from 19.2 to 33.0 while WISE falls from 50.9 to 47.4. V5 was chosen after seeing test results, so we report it only as evidence about the metric.

The official code compares the raw scores of the matching document across the three runs. InfoSearch does not say how scores were assigned to the output of list-wise rerankers. If they come from ranks, SICR reduces to its rank conditions, which a gate that pushes other documents down does meet. Scoring our runs by rank raises our SICR from 19.2 to 67.0, and on the 300-query subset it reverses the comparison with the single question (67.0 against 52.0). SICR values are thus comparable only between systems whose scores are formed the same way.

## 6.4 Cost

Scoring all six dimensions took 695,393 Clef-flash calls and about 274 million input tokens, which comes to about \$10.4 at the Workers AI list price, or about two thirds of a cent per instructed query.

Appendix C breaks this down by dimension (Table 7).

We did not run GPT-4o, but counting the tokens of the actual candidates under the sliding-window setup of InfoSearch and pricing them at the GPT-4o list price gives about \$486 for the same 4,791 rankings, roughly 47 times our cost (Appendix C). The two prices come from different providers, so the ratio is only a rough guide.

Most of our cost comes from one component. A “most recent” rule compares every pair of the 100 candidates in both orders, 9,900 calls per query. The 43 Source rows with such a rule asked 425,700 pair questions, 415,800 of them unique because two rows share their topic and candidates. That is 60% of all judge calls, and Section 6.1 shows they did not improve the ranking.

## 7 Conclusion

We showed that a reranker can follow documentlevel instructions well without any training, by turning the instruction into small questions that a judge model can answer and multiplying the answers. The clause questions carry the gains, and on the same judge, splitting the instruction raises WISE from 32.9 to 47.6 over one combined question, although raw-score SICR falls because the product rule cannot raise a score. On InfoSearch, this simple design scores well above the published GPT-4o result on average WISE and on four of the six dimensions, keeps plain relevance intact, and costs under one cent per instructed query. We think the same pattern (parse once, profile once, ask narrow questions and combine in code) is worth trying wherever the conditions users state can be named in advance as document attributes.

Fine-tuning has improved instruction following in retrieval models (Weller et al., 2025a,b; Zhou et al., 2025), and training our judge on attribute and rule questions, for example with labels from a larger model, could sharpen its answers.

## Limitations

The single-question comparison covers a random subset of 300 instructed queries, not the full benchmark, so its per-dimension results are noisy. Apart from that baseline, our baseline numbers are cited from the InfoSearch paper and not reproduced. The candidate pools, labels and evaluation code are the same, but we cannot rule out small differences in evaluation details. The public release has one Keyword query fewer than the paper, so our Keyword average is over 287 queries rather than 288.

The system depends on hosted models. We pin model identifiers and cache every response, which makes our numbers reproducible from the cache, but a provider may update a hosted model in place. The judge has open weights, and a local rerun would remove this dependence.

The parser can add rules that the benchmark does not grade. On InfoSearch it added a recency rule to 46 query-mode rows. Each of these matched wording in the instruction, so none was a parse error, but InfoSearch does not grade recency, and on Source the factor lowered scores slightly while adding most of the cost. A parser that invents a rule could do more harm, since a wrong gate can remove a relevant document. The self-check step reduces this risk without removing it.

We evaluate on a single benchmark. InfoSearch conditions map cleanly onto document attributes, which suits our design, and our attribute list was written with them in mind. A new domain needs its own list, and we have not measured how much a list written without the benchmark in view would lose. Instructions that redefine relevance in subtler ways, as in FollowIR, may need more yes/no rules and fewer attribute gates. A first attempt to reproduce the FollowIR candidate pool did not match the published first-stage numbers, so we leave FollowIR to future work rather than report numbers on a pool we cannot verify. Finally, the method needs one judge call per candidate and question. From an empty cache a ranking needed 145 judge calls on average, or 58 without the Source recency pairs, where GPT-4o needs 9 list-wise windows. The calls are small and independent, so they can run in parallel and latency depends more on the provider’s concurrency than on their number. We did not measure latency.

## References

Akari Asai et al. 2023. Task-aware retrieval with instructions. In Findings of the Association for Computational Linguistics: ACL 2023. ArXiv:2211.09260.

Cloudflare. 2026. Clef-flash and clef model cards, Workers AI. https://developers.cloudflare. com/workers-ai/models/clef-flash/ and https://developers.cloudflare.com/ workers-ai/models/clef/. Accessed 2026- 11-02.

Bradley Efron and Robert J. Tibshirani. 1993. An Introduction to the Bootstrap. Chapman and Hall.

Google DeepMind. 2026. Gemma 4 model card. https://ai.google.dev/gemma/docs/ core/model\_card\_4. Accessed 2026-11-02.

Aaron Grattafiori et al. 2024. The Llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Geoffrey E. Hinton. 2002. Training products of experts by minimizing contrastive divergence. Neural Computation, 14(8):1771–1800.

Kalervo Järvelin and Jaana Kekäläinen. 2002. Cumulated gain-based evaluation of IR techniques. ACM Transactions on Information Systems, 20(4):422– 446.

Albert Q. Jiang et al. 2023. Mistral 7B. arXiv preprint arXiv:2310.06825.

Tushar Khot et al. 2023. Decomposed prompting: A modular approach for solving complex tasks. In International Conference on Learning Representations (ICLR). ArXiv:2210.02406.

Sewon Min et al. 2023. FActScore: Fine-grained atomic evaluation of factual precision in long form text generation. In Proceedings of EMNLP. ArXiv:2305.14251.

Rodrigo Nogueira and Kyunghyun Cho. 2019. Passage re-ranking with BERT. arXiv preprint arXiv:1901.04085.

Rodrigo Nogueira, Zhiying Jiang, and Jimmy Lin. 2020. Document ranking with a pretrained sequence-tosequence model. In Findings of the Association for Computational Linguistics: EMNLP 2020.

Hanseok Oh et al. 2024. INSTRUCTIR: A benchmark for instruction following of information retrieval models. arXiv preprint arXiv:2402.14334.

OpenAI. 2024. GPT-4o system card. arXiv preprint arXiv:2410.21276.

Ronak Pradeep, Sahel Sharifymoghaddam, and Jimmy Lin. 2023a. RankVicuna: Zero-shot listwise document reranking with open-source large language models. arXiv preprint arXiv:2309.15088.

Ronak Pradeep, Sahel Sharifymoghaddam, and Jimmy Lin. 2023b. RankZephyr: Effective and robust zeroshot listwise reranking is a breeze! arXiv preprint arXiv:2312.02724.

Zhen Qin et al. 2024. Large language models are effective text rankers with pairwise ranking prompting. In Findings of the Association for Computational Linguistics: NAACL 2024. ArXiv:2306.17563.

Gerard Salton, Edward A. Fox, and Harry Wu. 1983. Extended Boolean information retrieval. Communications ofthe ACM, 26(11):1022–1036.

Mark D. Smucker, James Allan, and Ben Carterette. 2007. A comparison of statistical significance tests for information retrieval evaluation. In Proceedings of CIKM.

Hongjin Su et al. 2023. One embedder, any task: Instruction-finetuned text embeddings. In Findings of the Associationfor Computational Linguistics: ACL 2023. ArXiv:2212.09741.

Weiwei Sun et al. 2023. Is ChatGPT good at search? investigating large language models as re-ranking agents. In Proceedings of EMNLP. ArXiv:2304.09542.

Lewis Tunstall et al. 2023. Zephyr: Direct distillation of LM alignment. arXiv preprint arXiv:2310.16944.

Howard Turtle and W. Bruce Croft. 1991. Evaluation of an inference network-based retrieval model. ACM Transactions on Information Systems, 9(3):187–222.

Liang Wang et al. 2024. Improving text embeddings with large language models. In Proceedings of ACL. ArXiv:2401.00368.

Orion Weller et al. 2025a. FollowIR: Evaluating and teaching information retrieval models to follow instructions. In Proceedings of NAACL. ArXiv:2403.15246.

Orion Weller et al. 2025b. Promptriever: Instructiontrained retrievers can be prompted like language models. In International Conference on Learning Representations (ICLR). ArXiv:2409.11136.

Orion Weller et al. 2025c. Rank1: Test-time compute for reranking in information retrieval. In Proceedings ofCOLM. ArXiv:2502.18418.

Denny Zhou et al. 2023. Least-to-most prompting enables complex reasoning in large language models. In International Conference on Learning Representations (ICLR). ArXiv:2205.10625.

Jianqun Zhou et al. 2025. Beyond content relevance: Evaluating instruction following in retrieval models. In International Conference on Learning Representations (ICLR). ArXiv:2410.23841.

Honglei Zhuang et al. 2024a. Beyond yes and no: Improving zero-shot LLM rankers via scoring finegrained relevance labels. In Proceedings of NAACL. ArXiv:2310.14122.

Shengyao Zhuang et al. 2024b. A setwise approach for effective and highly efficient zero-shot ranking with large language models. In Proceedings of SIGIR. ArXiv:2310.09497.

## A Prompts and Questions

This appendix gives the exact text of every prompt and question in the frozen configuration. Placeholders in braces are filled at run time.

Parser. Figure 5 shows the parse prompt and Figure 6 the self-check prompt. Both go to Gemma-4- 26B-A4B with the system message “You convert search requests into structured filters. You answer with JSON only.” The parse call allows 700 output tokens and the self-check 400. The first attempt uses temperature 0. If the reply is not valid JSON of the expected shape, the call is repeated at temperature 0.4, at most three times in total. If the self-check fails all three times, the parse is kept unchanged. The {taxonomy} placeholder lists the six attributes with their option names (Table 5), one line per attribute. An empty instruction is shown as “(none)”.

Judge. Clef-flash does not take a free-form prompt. Each request carries the document text and a set of typed questions, and the model returns a probability for every answer. Table 6 lists the questions we ask. The profile questions are sent as one request per document, always in the same order. All yes/no rule questions for a document and query are also sent as one request.

Recency. For a “most recent” rule the judge sees two documents, A and B, each cut to its first 80 words, and answers the choice question in Table 6. Each pair is asked in both orders, and the two answers are averaged to cancel position bias, which gives $P ( e \succ d )$ , the probability that e is more recent than d. The factor for d is $\Pi _ { e \neq d } \big ( 1 - P ( e \succ d ) P _ { \mathrm { u s e } } ( e ) \big )$ , the probability that no other useful candidate is more recent.

## B Per-Dimension Ablations

Figure 4 shows the WISE of each ablation variant per dimension. The averages are in Table 3, and Section 6.1 discusses the pattern.

## C Cost Details

Table 7 lists the judge calls and their cost per dimension. Calls are the unique requests a dimension needs from an empty cache, priced at \$0.038 per million input tokens. The parser added 4,791 calls, one parse and one self-check per query-mode row. These and the calls to the larger Clef model are not priced in the table.

![](images/d3fdc2ef5fd80dc17876059850bf5990404c9858131ab7867a10cd4b3b7fec1e.jpg)  
Figure 4: WISE per dimension when one component is removed. Attribute gates carry Audience, Language and Source. Yes/no rules carry Keyword. Format and Length need both. Relevance alone does notfollow instructions on any dimension.

GPT-4o estimate. We did not run GPT-4o. InfoSearch follows the sliding-window setup of RankZephyr (Pradeep et al., 2023b; Sun et al., 2023), which ranks the top 100 candidates in 9 windows of 20 with a stride of 10. For each of the 4,791 rankings, we counted the tokens of the actual candidate texts with the GPT-4o tokenizer (o200k\_base), added 150 prompt tokens per window for the template and the query, and assumed 110 output tokens per window for the ranked list of identifiers. Windows overlap, so each ranking reads 180 candidate slots, which we priced at the mean candidate length of that ranking. This gives 43,119 calls, 175.5 million input tokens and 4.7 million output tokens. At the GPT-4o list price of \$2.50 per million input and \$10 per million output tokens, the total is about \$486.

Where the cost goes. Apart from the recency pairs, cost follows document length more than query count. Since profiles are computed once per document, a new query adds only its relevance and rule questions. A single choice question over the whole list, or sorting on an extracted date, would remove nearly all of the extra cost on Source.

```jsonl
Search request: {query}
Instructions / requirements from the user: {instruction}
Documents have these attributes (attribute: allowed option names):
{taxonomy}
Documents may also carry metadata: a last-modified date and a version.
Convert the request into JSON:
- "topic": one sentence describing the information the user needs. Do NOT include any rule: no source, format, audience, length,
, language, status, date, recency or version.
- "metadata_rules": date, recency and version requirements ONLY, using these forms:
{"type": "date_on_or_after", "date": "YYYY-MM-DD"}, {"type": "date_before", "date": "YYYY-MM-DD"},
{"type": "most_recent"} (user wants only the newest / most up-to-date information),
{"type": "version_equals", "version": "<exact version string>"}
"attribute_rules": requirements or exclusions about the attributes above that the user EXPLICITLY mentions, with the list of
, option names that are ALLOWED. Example: "not a blog" -> {"attribute": "source", "allowed": ["official_docs", "forum_qa",
, "news", "academic", "other"]}.
"other_rules": any other requirement or exclusion, as a simple positive yes/no question plus "keep_if" ("yes" or "no").
, Example: "ignore documents about the flu" -> {"question": "Does the document discuss the flu?", "keep_if": "no"}.
- Only encode requirements the user explicitly stated; never invent rules. Express each requirement exactly once
, (dates/versions/recency only in metadata_rules).
Use empty lists when there are no such rules.
Return only JSON: {"topic": "...", "metadata_rules": [...], "attribute_rules": [...], "other_rules": [...]}
```

## Figure 5: Parse prompt for Gemma-4-26B-A4B.

Search request: {query}   
Instructions / requirements from the user: {instruction}   
A parser turned this into a topic and a numbered list of rules:   
Topic: {topic}   
Rules:   
{rules}   
Only the line "Instructions / requirements from the user" contains instructions.   
For each rule answer true (keep) or false (remove):   
- true if the instructions ask for it. The parser may have reworded it, mapped it to option names or written it as   
a yes/no question; that still counts as asked for. Examples: instructions "Only use the most recent information"   
-> {"type": "most\_recent"} is true; instructions "Only consider official documents" -> {"attribute": "source",   
"allowed": ["official\_docs"]} is true; instructions "Do not use forums or blogs" -> {"attribute": "source",   
"excluded": ["forum\_qa", "blog"]} is true.   
- false if the instructions do not ask for it: e.g. the instructions are (none), or a source rule when the   
instructions only name a version.   
- false if it repeats a requirement already expressed by an earlier rule in the list.   
When in doubt, answer true.   
Also rewrite the topic so it describes only the information need, without any version, date, recency, source,   
format, audience, length or language requirement.   
Return only JSON: {"topic": "...", "keep": [true/false for rule 1, rule 2, ...]} with exactly {n} entries in "keep".  
Figure 6: Self-check prompt. {rules} is a numbered list ofthe parsed rules in JSON. An attribute rule that allows most options is shown by the options it excludes, so “not a blog” reads as "excluded": ["blog"]. This changes only the display, not the rule.

<table><tr><td>Attribute</td><td>Option</td><td>Description given to the judge</td></tr><tr><td>source</td><td>official_docs</td><td>Official documentation, manual or reference published by the project, vendor or organisation</td></tr><tr><td></td><td>blog</td><td>A blog post (personal or company blog)</td></tr><tr><td></td><td>forum_qa</td><td>A forum thread or Q&amp;A site post, e.g. Stack Overflow, Reddit, Quora</td></tr><tr><td></td><td>news</td><td>A news article from a news outlet or newspaper</td></tr><tr><td></td><td>academic</td><td>A scientific paper, study or academic text</td></tr><tr><td></td><td>other</td><td>Anything else, e.g. encyclopedia, product page, generic web page</td></tr><tr><td>format</td><td>prose</td><td>Mainly explanatory prose</td></tr><tr><td></td><td>code</td><td>Mainly a code snippet with little or no prose</td></tr><tr><td></td><td>qa_post</td><td>A question followed by one or more answers</td></tr><tr><td></td><td>list_table</td><td>Mainly a list, table or bullet points</td></tr><tr><td>audience</td><td>expert</td><td>Written for experts or specialists, using technical language</td></tr><tr><td></td><td>general</td><td>Written for a general, non-expert audience</td></tr><tr><td>length</td><td>sentence</td><td>A single sentence or very short snippet</td></tr><tr><td></td><td>paragraph</td><td>One paragraph</td></tr><tr><td></td><td>article</td><td>Several paragraphs or a full article</td></tr><tr><td>language</td><td>english</td><td>Written in English</td></tr><tr><td></td><td>chinese</td><td>Written in Chinese</td></tr><tr><td></td><td>other</td><td>Written in another language</td></tr><tr><td></td><td>current</td><td>Describes current, recommended practice</td></tr><tr><td>status</td><td></td><td></td></tr><tr><td></td><td>deprecated unclear</td><td>Describes something marked deprecated, removed, superseded or outdated Gives no indication either way</td></tr></table>

Table 5: The six profile attributes and their options. The parser sees the option names, and the judge sees the descriptions.

<table><tr><td>Step</td><td>Type</td><td>Question and answer options</td></tr><tr><td>Profile (per attribute)</td><td>choice</td><td>&quot;Which best describes the document&#x27;s {attribute}?&quot; Options: the de- scriptions in Table 5.</td></tr><tr><td>Relevance</td><td>score</td><td>“How completely does this document answer this need: &quot;{topic}&quot;?&quot; Levels 0–3: Does not address it; Mentions the subject but gives no usable information; Partially answers it; Fully answers it.</td></tr><tr><td>Other rule</td><td>yes/no</td><td>The question written by the parser, e.g. “Does the document discuss the flu?&quot;</td></tr><tr><td>Date rule</td><td>yes/no</td><td>“According to this document&#x27;s information, was it last modified or published on or after {date }?&quot; (or &quot;before {date} ?&quot;)</td></tr><tr><td>Version rule</td><td>yes/no</td><td>“Is this document about version {version}?&quot;</td></tr><tr><td>Recency pair</td><td>choice</td><td>“Which of the two documents contains the more recent, up-to-date information about: &quot;{topic} &quot;?&quot; Options: Document A, Document B.</td></tr></table>

Table 6: Questions sent to the Clef-flash judge. Dates and versions are passed exactly as the parser wrote them.

<table><tr><td>Dimension</td><td>Calls</td><td>Tok. (M) USD</td><td>¢/query</td></tr><tr><td>Audience</td><td>48,743</td><td>24.6 0.93</td><td>0.44</td></tr><tr><td>Keyword</td><td>30,695</td><td>6.8 0.26</td><td>0.09</td></tr><tr><td>Format</td><td>45,025</td><td>17.6 0.67</td><td>0.22</td></tr><tr><td>Language</td><td>43,877</td><td>14.9 0.56</td><td>0.28</td></tr><tr><td>Length</td><td>59,166</td><td>14.5 0.55</td><td>0.18</td></tr><tr><td>Source</td><td>467,887</td><td>196.1 7.45</td><td>2.48</td></tr><tr><td>recency pairs</td><td>415,800</td><td></td><td></td></tr><tr><td>Total</td><td>695,393</td><td>274.4 10.42</td><td>0.65</td></tr><tr><td>GPT-4o list-wise, estimated, not run by us</td><td></td><td></td><td></td></tr><tr><td>Total</td><td>43,119</td><td>180.2</td><td>486 30.4</td></tr></table>

Table 7: Judge calls, input tokens and cost at listprice. ¢/query covers the three rankings ofone instructed query. Length and Source tokens are reconstructed after an interrupted run (about 15% accuracy). GPT-4o tokens include 4.7M output tokens.