# Scoring Both Directions: LLMs realize the MRS they cannot reliably parse

Soham Dan Scale AI soham.dan@scale.com

## Abstract

The English Resource Grammar (ERG) is a hand-written computational grammar of En glish. Given a sentence, its processor, ACE, produces a formal meaning representation called Minimal Recursion Semantics (MRS): a graph of the sentence’s predicates and their arguments. The grammar is bidirectional and can also turn an MRS back into an English sentence. Hajdik et al. (2019) used the ERG’s treebank to build a benchmark for that generation task, MRS to text, and trained sequence-to-sequence models to solve it. The parsing task, text to MRS, can be tested on the same sentences. We reconstruct their 10K-sentence test split, and score two large language models, Claude Sonnet 4.5 and Claude Opus 5, in both directions against their trained systems and against ACE, with no task-specific training. Given an MRS and three examples, Opus writes the sentence at 76.3 BLEU, ten points above their system trained on 72k pairs (66.1 BLEU), and compa rable to their system trained on a million extra pairs (77.2 BLEU). Sonnet scores 65.7 BLEU, and letting it choose among ACE’s own candidate sentences lifts it to 69.6, while a pooled judge that keeps Opus’s own sentence among the candidates adds 0.6 points (77.0 BLEU). In the parsing direction, however, the models fall far behind ACE: asked for the MRS of the same sentences, they reach 57.2 (Sonnet) and 65.5 (Opus) $\mathrm { F _ { 1 } }$ on the graph’s predicates and arguments against 91.0 for ACE, and exact-match the gold on about 1% of sentences. We charac terize the failure modes for the parsing tasks, and conclude that a generation score alone does not show that models understand formal seman tic representations.

## 1 Introduction

The English Resource Grammar (ERG; Flickinger, 2000) is a hand-written, broad-coverage grammar of English. Its processor, ACE (Crysmann and

![](images/9e594e590388db74da2fb785f411096d2729a6ac4e6657c6cc8c04b87bdebcbc.jpg)  
Figure 1: An example sentence where all three systems realize the reference exactly and only ACE parses it exactly. Given the gold MRS (linearized for the LLMs), ACE, Sonnet and Opus all write the reference sentence. Given the sentence, only ACE recovers the MRS. Both LLMs split the particle verb send back, which the ERG treats as one predicate, and use predicates for how that the ERG does not have. Red marks predicates that differ from gold. The last column is the spans-off EDM $\mathrm { F _ { 1 } }$

Packard, 2012), parses a sentence into Minimal Recursion Semantics (MRS; Copestake et al., 2005), a graph whose nodes are the sentence’s predicates and whose edges are their argument roles, with quantifier scope left underspecified, and runs backwards to generate from a given MRS the sentences it represents. The Redwoods treebank (Oepen et al., 2004) pairs sentences from newswire, Wikipedia, tourism prose, fiction and dialogue with their handselected gold MRS (about ten thousand in its test partition). Hajdik et al. (2019) turned this treebank into a benchmark for the generation direction, MRS to text: each gold MRS linearized (as Dependency MRS, Copestake, 2009, in PENMAN form), trained a sequence-to-sequence model on 72k pairs, using BLEU to score against the original sentence. Their systems reach 66.11 BLEU on the 10K test split, and 77.17 with about a million extra silver pairs parsed automatically from newswire. Because the grammar is bidirectional, the same split poses the reverse task, text to MRS, on the same sentences, with ACE as a reference in both directions. A prompted language model needs no task-specific training for either, so the same system can be scored both ways on identical items. We use Claude Sonnet 4.5 and Claude Opus 5 with three exemplars each, and ask three research questions.

Does a prompted model reach the trained realizer? Yes, and Opus beats it (Table 1): Opus 5 scores 76.34 BLEU, 10 points above the 72k-pair trained seq2seq system, and on par with the millionpair silver-data trained system. Sonnet 4.5 obtains 65.65 BLEU matching the trained 72k-pair system. These public sentences predate both models’ training cutoffs and we run a substitution control to test memorization which shows relatively small memorization effects(§3). Parsed back through the grammar, the models’ sentences recover the input MRS at 85.4 (Sonnet) and 89.8 (Opus) spans-off EDM F<sub>1</sub> against 91.0 for the gold sentences themselves, so high BLEU here reflects preserved meaning, not surface overlap alone.

Does the grammar help? It helps the weaker model, Sonnet: ACE generates candidates for 79% of the items, its top-10 best list holds realizations its ranker does not surface, and Sonnet as a fluency judge over that list (using Sonnet’s own sentence where ACE abstains) reaches 69.64 BLEU on the test split, four points over Sonnet alone (Table 1). For Opus, we use a pooled judge that keeps Opus’s own sentence among the candidates and this shows a gain of about 0.6 BLEU, which shows that the grammar helps both models with diminishing returns for the stronger model.

Can the models parse back the MRS from the sentence? Not reliably. When we ask for the MRS of the same sentences, Sonnet reaches only 57.2 span-free EDM F<sub>1</sub> and Opus 65.5 compared to 91.0 for ACE’s parser, matching the gold analysis exactly on 0.6% and 1.4% of items against 50.5% for ACE (Table 3). The outputs are structureformatted correctly and almost always decode as MRS but what fails is the grammar’s conventions, the predicate inventory and the argument frames (§4).

Limits of LLMs as AMR analysts have been reported (Ettinger et al., 2023) and the producing/understanding gap mentioned in West et al. (2024), ERG lets us measure it on the same benchmark with a grammar as the reference both ways.

## 2 Setup

Benchmark. We take the ERG-1214 gold profiles and run Hajdik et al.’s released conversion unchanged, so inputs and references match their files exactly yielding 10,201 test items over 37 Redwoods profiles.

Three tasks. Q1, realization: linearized MRS → sentence over the full test split. Q2, reranking: Sonnet, as a fluency judge, chooses among ACE’s 10-best sentences, the model’s own prediction filling in where ACE produces nothing (hybrid) and a pooled variant that adds the model’s own sentence to the list. Q3, analysis: sentence → SimpleMRS on a fixed-seed 1,000-item sample stratified on the newswire data (whose gold MRS passes through the evaluator unchanged) are scored.

Models and prompts. Claude Sonnet 4.5 (20250929) and Claude Opus 5 with three fixed exemplars from outside the scored items. Sonnet ran with greedy decoding and a 3,000-token budget, and Opus 5 runs with default thinking.

The grammar. ACE 0.9.31 with ERG 1214, as realizer (n-best 10, 10 s per item) and parser (top-1). ACE is in-sample: the gold MRSs are ERG-1214 analyses, and its ranking model was trained on Redwoods, including these profiles.

Metrics. Realization is scored by SacreBLEU (exponential smoothing) (Post, 2018) against the original sentence after their post-processing. Parsing is scored on the EDS reduction of each MRS (Oepen and Lønning, 2006) with three metrics (App. A). EDM F<sub>1</sub> (Dridan and Oepen, 2011) matches predicates and argument triples, either anchored to their character spans or with spans removed so that nodes are keyed by predicate name alone. Exact match requires the predicted and gold graphs to be isomorphic once spans are ignored. Smatch (Cai and Knight, 2013) matches triples under the node alignment that maximizes agreement, so it is invariant to variable names.

## 3 Realization

Q1: LLMs against the trained realizer. Table 1 shows both LLMs with three prompted exemplars, evaluated on the same linearized MRS test set as the trained systems from Hajdik et al. (2019). Sonnet 4.5 matches the trained 72k-pair (gold-only) system and Opus 5 is 8-10 points above it essentially tied with the million-pair (gold+silver) trained system. We also compare this to ACE which realizes 79.2% of the split at 61.81 BLEU.

<table><tr><td>System</td><td>BLEU</td></tr><tr><td>trained systems (Hajdik et al., 2019)</td><td></td></tr><tr><td>gold-only (72k pairs)</td><td>66.11</td></tr><tr><td>gold+silver (~1M pairs)</td><td>77.17</td></tr><tr><td>grammar</td><td></td></tr><tr><td>ACE 1214, generable (79.2%)</td><td>61.81</td></tr><tr><td>prompted LLMs with three exemplars</td><td></td></tr><tr><td>Sonnet 4.5</td><td>65.65</td></tr><tr><td>Opus 5</td><td>76.34</td></tr><tr><td>model + grammar (Q2)</td><td></td></tr><tr><td>Sonnet + ACE, judge hybrid</td><td>69.64</td></tr><tr><td>Opus + ACE, pooled judge</td><td>76.95</td></tr></table>

Table 1: Q1 realization on Hajdik et al.’s split and the Q2 hybrids (Sonnet as judge throughout: its pick where ACE generates, the model’s prediction elsewhere).
<table><tr><td>items ACE generates for (n=7,865)</td><td>BLEU</td></tr><tr><td>ACE 1-best (its own order)</td><td>61.82</td></tr><tr><td>shortest candidate</td><td>69.01</td></tr><tr><td>Sonnet as fluency judge over ACE&#x27;s n-best reference selector over ACE&#x27;s n-bestº</td><td>71.25 74.84</td></tr><tr><td>Sonnet 4.5 alone</td><td></td></tr><tr><td>Opus 5 alone</td><td>67.02</td></tr><tr><td></td><td>77.50</td></tr><tr><td>reference selector, ACE&#x27;s n-best ∪ Opus Sonnet judge, ACE&#x27;s n-best ∪ Opus</td><td>83.07 78.36</td></tr></table>

Table 2: Q2, where ACE generates. <sup>c</sup>Per item the highest sentence-BLEU candidate which is a ceiling. Sonnet scores 62.56 and Opus 73.72 on the subset that ACE abstains (2,067 items) on.

Is high BLEU faithful? BLEU measures overlap with the reference, not preservation of the input, so we parse each system’s sentence back with ACE and score the parse against the input MRS: its parse of the gold sentence itself scores 91.0 spans-off EDM F on the 1K items (92.6 on the 762 ACE generates for), exact on 50.5%. Sonnet’s sentences score 85.4 (exact 28.6%; 96.1% parse), Opus’s 89.8 (38.9%; 96.7%); ACE’s own realization and the judge’s pick score 94.5 and 94.3 on the 762. Opus’s sentences sit within 1.2 points of the gold sentence’s own reparse, Sonnet’s within 5.6: the realizations largely preserve the input’s predicates and arguments (App. A).

Q2: the grammar as a candidate generator. Here we focus on the examples ACE generates successfully on, to see how much the grammar helps in a hybrid system. We see that on this subset the top-10 often has the correct sentence but the ACE ranking is poor (Table 2): ACE ranks its realizations with the Redwoods parse-selection model (redwoods.mem, the only model its configuration loads) but it has no ranker trained for realization (Velldal and Oepen, 2005, 2006). Its first choice scores 61.82, the shortest candidate 69.01, Sonnet as a fluency judge 71.25 (+9.4, on this subset). The judge’s pick also beats Sonnet’s own generation by 4.2 points, and the hybrid scores 69.64 on the full split, four points over Sonnet alone.

Adding Opus’s own sentence to ACE’s candidate pool raises the reference-selected score (corpus BLEU over the per-item best pick) to 83.07. The pooled judge (using Sonnet for fluency) reaches 78.36 on these items and 76.95 on the full split, essentially tied with the gold+silver system (Hajdik et al., 2019). The judge (Sonnet) picks Opus’s sentence on 66.9% of items and an ACE candidate on the remaining 33.1%, close to the 65.2% of the reference selector, so the judge is not simply preferring model text.

Memorization It does not by itself explain the realization results. A verbatim-continuation probe shows that Sonnet has seen some of the public text: it recalls 33% of one essay’s sentences and 0% on three control profiles, but recall is unrelated to realization quality. A second control swaps one content noun for another in both the MRS and the reference (140 pairs). BLEU is 3.17 points higher on the original pairs than on the substituted ones, the direction memorization would predict, the LLMs are still on par with the trained systems.

## 4 Parsing

Q3: the same models on the reverse task. Both models write MRS that decodes, but rarely the gold MRS (Table 3). ACE’s top parse matches gold exactly on 50.5% of sentences, but Opus does so on only 1.4% and Sonnet on 0.6%. On spans-off EDM the models trail ACE by 26 (Opus) and 34 (Sonnet) points.

The calibration rows show how to read these scores. Gold scores 100 on every metric. Gold with every predicate name junked (dummy word replacement) still scores 79.2 Smatch, level with Opus and above Sonnet, because Smatch rewards graph structure even when every name is wrong; a constant MRS still earns 33.1. Spans-off EDM and exact match fall to zero on both.

<table><tr><td></td><td colspan="4">whole graph</td><td colspan="2">relation F1</td><td colspan="2">recall</td></tr><tr><td></td><td>EDMsf</td><td>EDMsa</td><td>exact</td><td>Smatch</td><td>lab. unlab.</td><td></td><td>names edges</td><td></td></tr><tr><td>ACE 1214</td><td>91.0</td><td>88.3</td><td>50.5</td><td>94.3</td><td>90.5</td><td>91.1</td><td>92.8</td><td>94.8</td></tr><tr><td>Opus 5</td><td>65.5</td><td>43.0</td><td>1.4</td><td></td><td>79.073.6</td><td>76.1</td><td>73.9</td><td>84.0</td></tr><tr><td>Sonnet 4.5</td><td>57.2</td><td>36.9</td><td>0.6</td><td></td><td>77.666.5</td><td>69.7</td><td>65.5</td><td>74.1</td></tr><tr><td colspan="9">Calibration inputs</td></tr><tr><td>gold</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td></tr><tr><td>names junked</td><td>0.0</td><td>79.4</td><td>0.0</td><td>79.2 99.7</td><td></td><td>99.6</td><td>0.0</td><td>0.0</td></tr><tr><td>constant</td><td>5.6</td><td>0.2</td><td>0.1</td><td>33.1 35.4</td><td></td><td>39.7</td><td>11.2</td><td>12.5</td></tr></table>

Table 3: Parsing the same 998 sentences (text → SimpleMRS; LLMs three-shot, ACE top-1 parse). Outputs decode as SimpleMRS for 97.1% (ACE, parse within resource limits), 96.3% (Opus) and 99.4% (Sonnet) of sentences; failures score 0. Whole graph: EDM $\mathrm { F _ { 1 } }$ with spans off (sf) and span-anchored (sa); exact graph match ignoring spans; Smatch $\mathrm { F _ { 1 } }$ over all triples. Relation $F _ { 1 } \mathbf { : }$ Smatch on edges only, with role labels (lab.) or ignoring them (unlab., attachment only). Recall: share of gold predicates named exactly (names), and share of gold edges recovered between correctly named nodes (edges). Calibration inputs are scored against gold: gold itself, gold with every predicate name replaced by a nonsense string, and one fixed MRS for every sentence.

The failure is neither format, since 96–99% of outputs decode, nor total since, counting only which predicate lemmas appear (\_find $\_ \mathbb { 1 }$ and \_find\_v\_mental both count as find; $\mathrm { F _ { 1 } }$ over the multiset, ignoring edges), the models score 78 (Sonnet) and 84 (Opus). It lies in the grammar’s conventions, at two levels. First, names: the models name only 65.48% (Sonnet) and 74% (Opus) of gold predicates exactly, against 93% for ACE, and the misses are plausible but incorrect (\_find\_v\_1 for \_find\_v\_mental, compound omitted). Second, attachment: between correctly named nodes they recover 74% and 84% of gold edges, against 95% for ACE. These are wrong attachments, not wrong role labels. Ignoring labels raises relation $\mathrm { F _ { 1 } }$ only from 66.5 to 69.7 (Sonnet) and from 73.6 to 76.1 (Opus), far below ACE’s 91.1.

## 5 Conclusion

We ran Hajdik et al.’s benchmark in both directions on the same sentences: from meaning representation to text, and from text back to meaning representation. The two directions give very different answers.

Going from MRS to text, prompted LLMs are strong. With three examples in the prompt, Sonnet matches the system trained on 72k gold pairs, and Opus beats it by about ten BLEU points and is level with the system that also used silver data. The sentences they write mostly keep the meaning of their input. Letting the models choose among the grammar’s own candidate sentences helps Sonnet by four points and Opus slightly.

Going from text to MRS, the same models fail. They almost never produce the exact gold MRS. They mostly choose the right words but not the grammar’s exact predicate names or the way it attaches arguments.

So a model can turn an MRS into a good sentence without being able to write that MRS itself. A high generation score should not be read as evidence that the model knows the representation.

## Limitations

We evaluate two Anthropic models under one prompting setup with three examples. Stronger setups, such as retrieval or constrained decoding, might narrow the parsing gap. Adding examples and enabling extended thinking did not close it. Our aim is to measure what prompted models can do on their own and combining them with constrained decoding is future work.

## References

Shu Cai and Kevin Knight. 2013. Smatch: an evaluation metric for semantic feature structures. In Proceedings of the 51st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 748–752.

Ann Copestake. 2009. Slacker semantics: Why superficiality, dependency and avoidance of commitment can be the right way to go. In Proceedings ofthe 12th Conference of the European Chapter of the Association for Computational Linguistics (EACL), pages 1–9.

Ann Copestake, Dan Flickinger, Ivan A. Sag, and Carl Pollard. 2005. Minimal recursion semantics: An introduction. Research on Language and Computation, 3(2–3):281–332.

Berthold Crysmann and Woodley Packard. 2012. Towards efficient HPSG generation for German, a nonconfigurational language. In Proceedings of COL-ING 2012, pages 695–710, Mumbai, India.

Rebecca Dridan and Stephan Oepen. 2011. Parser evaluation using elementary dependency matching. In Proceedings ofthe 12th International Conference on Parsing Technologies, pages 225–230, Dublin, Ireland.

Allyson Ettinger, Jena D. Hwang, Valentina Pyatkin, Chandra Bhagavatula, and Yejin Choi. 2023. “you are an expert linguistic annotator”: Limits of LLMs

as analyzers of Abstract Meaning Representation. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 8250–8263, Singapore.

Dan Flickinger. 2000. On building a more efficient grammar by exploiting types. Natural Language Engineering, 6(1):15–28.

Valerie Hajdik, Jan Buys, Michael W. Goodman, and Emily M. Bender. 2019. Neural text generation from rich semantic representations. In Proceedings ofthe 2019 Conference of the North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies, pages 2259–2266, Minneapolis, Minnesota. Association for Computational Linguistics.

Stephan Oepen, Dan Flickinger, Kristina Toutanova, and Christopher D. Manning. 2004. LinGO Redwoods: A rich and dynamic treebank for HPSG. Research on Language and Computation, 2(4):575–596.

Stephan Oepen and Jan Tore Lønning. 2006. Discriminant-based MRS banking. In Proceedings of the Fifth International Conference on Language Resources and Evaluation (LREC), Genoa, Italy.

Matt Post. 2018. A call for clarity in reporting BLEU scores. In Proceedings of the Third Conference on Machine Translation: Research Papers, pages 186– 191, Brussels, Belgium. Association for Computational Linguistics.

Erik Velldal and Stephan Oepen. 2005. Maximum entropy models for realization ranking. In Proceedings ofthe 10th Machine Translation Summit (MT Summit ${ \dot { X } } ) ,$ , pages 109–116, Phuket, Thailand.

Erik Velldal and Stephan Oepen. 2006. Statistical ranking in tactical generation. In Proceedings ofthe 2006 Conference on Empirical Methods in Natural Language Processing, pages 517–525, Sydney, Australia.

Peter West, Ximing Lu, Nouha Dziri, Faeze Brahman, Linjie Li, Jena D. Hwang, Liwei Jiang, Jillian Fisher, Abhilasha Ravichander, Khyathi Chandu, Benjamin Newman, Pang Wei Koh, Allyson Ettinger, and Yejin Choi. 2024. The generative AI paradox: “what it can create, it may not understand”. In Proceedings of the Twelfth International Conference on Learning Representations (ICLR).

## A Appendix

Data. Hajdik et al. (2019)’s linearizer fails on 266 of the 10,201 test items (blank lines in their released source file). We score the remaining 9,935 minus the three held-out exemplars, 9,932 items. ACE generates for 7,865 of these.

Decoding. Sonnet and the judge use temperature 0 and a 3,000-token limit. Opus 5 does not accept a temperature and runs at its default thinking.

Round-trip check. ACE parses each system’s sentence (top-1), and we score the parse against the input MRS with spans-off EDM. Sentences ACE cannot parse score zero. As a reference, the same procedure applied to the gold sentence gives 91.0 on the 1000 items.

Memorization controls. The verbatim probe gives Sonnet the opening of a sentence from the test documents and checks whether it continues it word for word. The substitution control replaces one content noun with a different noun of the same length in both the MRS and the reference and compares BLEU on original and substituted pairs.

Parsing metrics. We read gold and predicted SimpleMRS with PyDelphin 1.11 and convert both to EDS graphs. An output that cannot be read scores zero on every metric.

EDM splits a graph into small facts: which predicates it contains, which arguments link them, and which features (tense, number, and so on) they carry. It then counts how many of the gold facts the prediction also contains, reporting precision, recall and F<sub>1</sub>. In the span-anchored version, a predicted node matches a gold node only if both cover the same stretch of the sentence. In the spans-off version, nodes match by predicate name alone.

Exact match asks whether the predicted graph is the gold graph with only the variable labels changed, ignoring spans. We check this with a standard graph-matching search, which finished on every item.

Smatch is the usual AMR metric, applied to the EDS graphs. It finds the pairing of predicted and gold nodes that makes the most facts agree, then scores those facts. It searches for this pairing at random, so repeated runs can differ by about 0.3.

Predicate-lemma overlap ignores edges and sense labels. \_find\_v\_1 and \_find\_v\_mental both count as find, and we compute $\mathrm { F _ { 1 } }$ between the two lists of lemmas. Edge recall pairs each gold node with a predicted node that has exactly the same predicate name. It then asks what share of the gold edges between paired nodes also appear, with the same label, in the prediction.

BLEU. We use SacreBLEU 2.6 with its default settings (13a tokenization, exponential smoothing, one reference).

Choosing among candidates. The reference selector picks, for each item, the candidate closest to the reference sentence by BLEU. It looks at the answer, so it only shows how good the candidate list could be; no real system can use it. Picking the best sentence one at a time does not quite maximize corpus BLEU, so we also report a corpus-BLEU selector. It starts from the same picks and keeps swapping candidates as long as corpus BLEU goes up.

Prompts. Each prompt has a system message, then three worked examples, then the test item.

• Realization. “You are an expert in the English Resource Grammar (ERG) and Minimal Recursion Semantics (MRS). You are given a linearized Dependency MRS graph. Write the English sentence it represents. Tokens like named0, named1, month0 are placeholders for named entities: copy them through verbatim in your output. Output only the sentence, with no explanation, no quotes and no markup.” Examples are Graph:/Sentence: pairs.

• Parsing. “You are an expert linguistic annotator producing Minimal Recursion Semantics in the DELPH-IN SimpleMRS format used by the English Resource Grammar. Output ONLY the MRS.” Examples are Sentence:/MRS: pairs.

• Judge. “You are a careful editor of written English. You will be shown numbered candidate sentences that are all intended to express the same meaning. Choose the single most fluent, natural and well-formed English sentence. Output only its number.” The user message lists the candidates and ends with “Answer with the number only.”