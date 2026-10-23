# Mostly Reading: What Open-Book Scores Measure in Document-Level LoRA Adapters

Martino M. L. Pulici<sup>1,2</sup>, Cuong Xuan Chu<sup>1</sup>, Evgeny Kharlamov<sup>1</sup>, Volker Tresp<sup>2,3</sup>

<sup>1</sup>Bosch Center for Artificial Intelligence, Germany <sup>2</sup>LMU Munich, Germany   
<sup>3</sup>Munich Center for Machine Learning, Germany Correspondence: martino.pulici@de.bosch.com

## Abstract

Single-document LoRA adapters have been proposed as units of knowledge to be merged and retrieved; yet open-book scoring, with the document in the prompt, can credit a document’s own adapter (its specialist) for reading that document as readily as for holding it. Training one adapter for each of 150 Wikipedia articles on Llama-3.1-8B-Instruct, we find that open-book scores mostly measure a reading behaviour every adapter shares: a random adapter trained on another document recovers 82 % of the specialist’s open-book advantage over the base model but only 2 % of its advantage on a cloze probe that asks for the same document’s facts without providing the source text, a share we cannot distinguish from zero. Content does reach the weights, since the specialists beat the base model in cloze recall by 0.094, and the geometry of their updates follows the documents they read; yet, in a test of 30 queries, adapters chosen by weight-space proximity or text embedding are not distinguishable from random ones on either probe, although that test cannot exclude an advantage of about four fifths of what a query’s own adapter gains over a random one. At 16 documents or more, correctly scaled merges recover 87 % to 89 % of the specialists’ open-book advantage but only 12 % to 16 % of their cloze advantage, with intervals that run from below zero to at most 32 %, against 94 % for joint training. We also report four silent errors, each of which produced a clean, wrong result.

## 1 Introduction

Low-rank adaptation (LoRA; Hu et al., 2022) makes fine-tuning cheap enough that adapters can be imagined as a storage medium, one per document, kept in a bank and merged (Ilharco et al., 2023; Yadav et al., 2023) or retrieved (Su et al., 2025; Back et al., 2026) as a task requires. What such a bank is worth depends on what its adapters hold; yet scoring them open-book (OB), with the document in the prompt as in the variant of parametric retrieval-augmented generation (RAG) that Tang et al. (2025b) find strongest, cannot tell an adapter that holds the document’s content from one that has learned to extract answers from whatever text it is given: we find that it mostly measures the second. To separate the two, on Llama-3.1- 8B-Instruct (Llama Team, 2024), we change which adapter answers a fixed document in place of its own specialist, and we employ a cloze probe that states each held-out fact as a sentence with one blank — without the document and in a surface form that no adapter was trained on (section 2).

Holding each document’s training questions fixed and varying the text trained beside them (section 3.1), we find that an adapter trained on the questions without the text comes second of five training conditions when scored OB, yet on cloze it recalls 0.132 less of the document than the base model does; a cloze gain clear of zero appears exactly where an adapter read the scored document’s full text, and the specialists’ gain over the base model is 0.094. The weights record the same variable: distances between updates sort adapters by domain (section 3.2) and, when an adapter’s questions and text come from different documents, follow the text rather than the questions (section 3.3).

OB scores follow something else. An adapter applied to another article of its own domain recovers 88 % of the OB advantage that the specialist holds over the base model (section 4.1), and pooled over collections of 16 documents or more, a random adapter from outside the collection recovers 82 % of the specialist’s OB advantage and 2 % of its cloze advantage, with a confidence interval (CI) that includes zero (section 4.2). In a bank that never holds the query’s own adapter, an adapter chosen by weight-space proximity is not distinguishable from a random one on either probe, nor is one chosen by text embedding, although with 30 queries the cloze CI still admits an advantage of about four fifths of what the query’s own adapter gains over a random one (section 4.3).

Composition shows the same split (section 5): at four documents or more, merges keep 0.93 to 1.01 of the specialists’ OB score against 0.88 to 0.97 for random, but at 16 documents or more they recover 12 % to 16 % of the cloze advantage, each CI including zero, where joint training recovers 94 %. OB scores thus overstate how much of a document’s content survives merging or substitution.

Each of four silent errors produced clean results we were prepared to report (section 6): (a) a training-length default that reversed a comparison between training conditions; (b) a repetition penalty left in our evaluation code; (c) a scaling error in our merge, which looked like a cost of merging; and (d) an evaluation precision unlike the one the adapters were trained on.

## 2 Experimental setup

Corpus and training. We use 150 Wikipedia articles in English (Wikimedia Foundation, 2026), 25 from each of biology, computer science, economics, history, law, and medicine. For each article Claude Sonnet 4.6 (Anthropic, 2026) generated 12 question-answering (QA) pairs, each validated by string match against the source text; eight are used for training, and four (three factual questions and one true/false item) are held out for evaluation. The base model is Llama-3.1-8B-Instruct (Llama Team, 2024; Meta Llama, 2024a), loaded in 4-bit NF4 with bf16 computation as in QLoRA (Dettmers et al., 2023); adapters of rank 8 on all seven projection matrices train for 500 steps with next-token cross-entropy on every token of the training string (instruction, document, question, and answer), capped at 2048 tokens, which fits every training example. The geometry of section 3 is replicated with the same hyper-parameters on three other models; section A.1 gives the rest of the configuration.

Training-text conditions. Each document’s QA pairs are held fixed, while the text trained beside them (generated by Claude Sonnet 4.6 from the prompts of section B) varies, resulting in five conditions: (a) the natural article; (b) a style-normalized rewrite into neutral prose; (c) a synthetic fill-in-the-blank template that keeps six extracted facts; (d) no context, which omits the document text; and (e) swapped context, including another article’s prose beside this article’s questions. When no condition is explicitly named, the adapter is trained on the natural article.

Evaluation. Every evaluation loads the base model in 4-bit NF4 and applies each adapter as a LoRA module. OB evaluation supplies the document with the training-time instruction template (section B.1) without a chat template, at a window of 2048 tokens that cuts no prompt, while closedbook (CB) evaluation withholds it; answers are generated greedily, up to 128 new tokens and with no repetition penalty, and scored by Stanford Question Answering Dataset (SQuAD) token-level $F _ { 1 }$ (Rajpurkar et al., 2016) over all four held-out items; section A.2 checks the metric against answer length. Semantic distance between documents is one minus the cosine similarity of their all-mpnet-base-v2 sentence embeddings (Reimers and Gurevych, 2019; Sentence Transformers, 2021).

The cloze probe. The cloze probe rewrites each held-out QA pair as a declarative sentence with one blank that the answer fills (section B.2) and presents it without the document, so that filling it requires the fact from the weights in a surface form that no adapter was trained on. The three factual items are scored by recall of the reference fill (an empty prediction scoring zero), and the true/false item is reported beside them. Unlike the QA evaluations the cloze harness wraps each prompt in the model’s chat template, which no adapter saw in training but which enters every score and every reference alike; rerun in the training format on 35 documents, the chat wrapper raises the specialists’ cloze gain over the base model by +0.057, a change whose CI, [−0.069, +0.183], includes zero (section A.3).

Confidence intervals. CIs are 95 % percentile intervals from 10 000 bootstrap resamples, paired across methods; the resampled unit is the document, except for composition contrasts (for which it is the collection), retrieval (the query), and transfer (the source–target pair).

Composition protocols. For a collection of N documents we compare five ways of supplying one adapter to answer all of the questions, the first three of which are merging methods: (a) LERP, a linear interpolation of the stored factors at square-root weight forming $( N ^ { - 1 / 2 } \textstyle \sum _ { i } \mathbf { B } _ { i } ) ( N ^ { - 1 / 2 } \textstyle \sum _ { j } \mathbf { A } _ { j } )$ in which each adapter’s own update is weighted $1 / N$ (section 5.2); (b) update merging, which averages the updates $\mathbf { B } _ { i } \mathbf { A } _ { i }$ and projects the average back to rank 8 by truncated singular value decomposition (SVD; section E.1); (c) trim, elect sign, and merge (TIES; Yadav et al., 2023), applied to the A and B factors separately (section E.1); (d) joint training, which fits one adapter to the union of the collection’s training examples for the same 500 steps and should be viewed as a retraining reference rather than a bound; and (e) random, applying one adapter trained on a document outside the collection and averaging its scores over five independent draws because single values vary widely (section D.1). Retention is a method’s score on a document’s held-out items divided by the score of the document’s own specialist adapter. Collections are drawn at $N \in \{ 2 , 4 , 8 , 1 6 , 2 5 , 3 2 \}$ under a within-domain regime and a cross-domain one spread over all six domains, five per cell with seeds shared across methods so that every comparison is paired, 56 collections in all; within-domain at $N = 2 5$ uses the six domains themselves, so in that setting every random adapter necessarily comes from another domain.

## 3 Content and geometry

## 3.1 Training text and encoding

Table 1 scores the training-text conditions of section 2, which fix each document’s QA pairs and vary only the text trained beside them; cloze figures are paired differences from the base model over the 150 documents, since the probe’s absolute scale is not comparable to that of the QA evaluations, and section C.1 gives every CI. For the OB evaluation the conditions read different texts, listed in section C.1; swapped context scored on its source, in particular, reads the donor document with the source’s questions, so its context holds none of the answers.

CB $F _ { 1 }$ and cloze recall rank the conditions in nearly opposite orders. The three conditions that lead CB are those that read least of the scored document’s content (swapped context scored on its source at 0.456, synthetic at 0.443, and no context at 0.426, against 0.399 for the natural adapter and 0.320 for the base model), yet on cloze the swapped and synthetic adapters are not distinguishable from the base model, and the no-context adapters fall below it by $0 . 1 3 2 \ : ( \mathrm { C I } _ { 9 5 \% } = [ 0 . 0 8 7 , 0 . 1 7 6 ] )$ , further than the natural adapter rises above it (+0.094). A cloze gain clear of zero appears exactly where an adapter read the scored document’s full content:

<table><tr><td rowspan="2">Condition</td><td colspan="2">Mean  $F _ { 1 }$ </td><td rowspan="2"> $\Delta$  cloze recall</td></tr><tr><td>OB</td><td>CB</td></tr><tr><td>Base</td><td>0.504</td><td>0.320</td><td></td></tr><tr><td>Natural</td><td>0.822</td><td>0.399</td><td>+0.094</td></tr><tr><td>Style-normalized</td><td>0.708</td><td>0.419</td><td>+0.097</td></tr><tr><td>Synthetic</td><td>0.649</td><td>0.443</td><td>+0.034</td></tr><tr><td>No context</td><td>0.773</td><td>0.426</td><td>-0.132</td></tr><tr><td>Swapped (source)</td><td>0.424</td><td>0.456</td><td>-0.028</td></tr><tr><td>Swapped (donor)</td><td>0.686</td><td>0.428</td><td>+0.056</td></tr></table>

Table 1: Token $F _ { 1 }$ over all 150 documents (OB and CB) and cloze recall on each document’s three factual items as a paired difference from the base model. All CIs exclude zero except for synthetic and swapped context on source; each OB condition reads the text listed in section C.1, with swapped context on source reading its donor’s text (which holds none of its answers).

the natural adapter, its style-normalized rewrite, and the swapped adapter scored on the document it read. Natural and style-normalized are basically equal (−0.003 with $\mathrm { C I _ { 9 5 \% } = [ - 0 . 0 4 2 , + 0 . 0 3 7 ] ) }$ whereas the style-normalized adapters exceed the synthetic ones by 0.063 with a CI clear of zero; the synthetic condition sees different facts during training and evaluation, so its missing gain may reflect coverage as much as format. We therefore read CB scores as rewarding answer-shaped output in the format an adapter trained on, most readily supplied by the conditions without usable prose, while cloze, which states each fact in a surface form that no condition trained on, is the probe that tracks content.

OB scores follow neither ordering: the nocontext adapter, which read no document text at all, comes second of the five conditions at 0.773, $0 . 0 4 9 \left( \mathrm { C I } _ { 9 5 \% } = [ 0 . 0 2 5 , 0 . 0 7 4 ] \right)$ behind the natural adapter and ahead of both rewrites, and every neighbouring pair in that column differs by a paired CI that excludes zero. Therefore, an adapter can score close to a specialist OB while recalling less of the document than the unadapted model does.

## 3.2 Orthogonality and semantic structure

Merging acts on the weight updates themselves, whose cosine (averaged over all 224 layer–module combinations) has a mean of 0.0066 across the 11 175 pairs, every pair lying between 0.004 and 0.013. Two adapters driven by identical gradients through independently initialized A matrices would give about $r / d _ { \mathrm { i n } } \ : ( \mathrm { i . e . , 0 . 0 0 1 8 }$ averaged over the 224 combinations), so a small cosine cannot by itself show that adapters hold different things; the observed mean is nearly four times that reference, which indicates that the A matrices move toward input directions the adapters share (section C.3), although for merging the updates stay nearly orthogonal layer by layer.

Pairwise Frobenius distances between adapter updates correlate with sentence-embedding distances between the corresponding documents on every architecture we train. Before these runs we had fixed a Spearman $\rho$ of 0.250 as the threshold a single layer would need to be selected; no single layer reached it, whereas with every layer aggregated three of the four architectures exceed it, with Llama-3.1-8B-Instruct falling just short at 0.242. The correlation concentrates in the final third of each network: on Llama-3.1-8B-Instruct selecting the eight highest-correlating layers on half the adapters and evaluating them on the other half gives a held-out $\rho$ of $0 . 2 9 6 \ ( \mathrm { C I } _ { 9 5 \% } = \ [ 0 . 2 0 4 , 0 . 3 5 6 ] )$ against 0.318 when the same adapters both select and score the layers. Over all layers the fraction of an adapter’s six nearest neighbours sharing its domain is 0.708 against a chance rate of 0.161, and clustering separates sub-domains that the six labels do not encode (section C.4 gives every architecture, the selection procedure, and the clusters): if proximity in this space fails to pick out adapters that carry a document’s content, an uninformative geometry is not the likely explanation.

## 3.3 Context swap

Because the document an adapter reads and the questions it answers come from the same source in every condition but one, the correlation above cannot say which of the two is recorded by the geometry: the swapped-context condition separates them. The geometry follows the donor, correlating with its semantics at $\rho = + 0 . 2 1 7$ against +0.033 for the source, and not because the swapped adapters fail to move, since their updates travel as far as the natural condition’s (mean pairwise distance of 21.09 against 21.10). Cloze recall points the same way: the swapped adapters exceed the base model by 0.056 on the document they read, whereas on their source they are not distinguishable from it. Paired on the 99 distinct donor documents, a swapped adapter reaches an OB $F _ { 1 }$ of 0.685, against 0.812 for the document’s natural adapter and 0.503 for the base model: the swapped adapter recovers $5 9 \% ( \mathrm { C I } _ { 9 5 \% } = [ 4 8 \% , 6 9 \% ] )$ of the natural adapter’s advantage, with question supervision supplying the rest (section C.2).

## 4 Behaviour and content

OB scoring supplies the document in the prompt, as does the variant of parametric RAG that also places the retrieved passages in context, which Tang et al. (2025b) find the strongest: an adapter can therefore be credited for reading a supplied text well as readily as for holding what it read, and we separate the two by fixing the scored document and changing which adapter answers it. We express each result as a share of the specialists’ advantage over the base model on the same documents: $( \bar { s } - \bar { s } _ { b } ) / ( \bar { s } _ { s } - \bar { s } _ { b } )$ where $\mathit { \Pi } _ { \overline { { { S } } } , \ \overline { { { S } } } _ { S } , }$ and $\bar { s } _ { b }$ are the mean scores of the method, of the documents’ own specialists, and of the base model respectively. We give shares as percentages and retentions, which divide scores rather than advantages, as decimals — this way the two kinds of ratio cannot be mistaken for one another.

## 4.1 Transfer to unseen documents

When applied OB to a different article from its own domain, over 120 source–target pairs, an adapter raises the base model’s score on the target by $+ 0 . 2 7 3 ~ \ : \ : ( \mathrm { C I } _ { 9 5 } \% = [ + 0 . 2 4 5 , + 0 . 3 0 1 ] )$ while each specialist adds $+ 0 . 3 0 9 \mathrm { \ : \ : ( C I _ { 9 5 \% } \ : \ : = \ : \alepharpoonright }$ $[ + 0 . 2 8 3 , + 0 . 3 3 5 ] )$ on its own article. A foreign adapter therefore recovers 88 % $( \mathrm { C I _ { 9 5 \% } } =$ $[ 7 7 \% , 1 0 1 \% ] )$ of the specialist’s advantage; the CI reaches 100 %, so the part of that advantage which does not transfer — the only part OB scoring could attribute to the document itself— cannot be distinguished from zero. No-context adapters, trained on the target’s questions without its text, gain +0.242 $( \mathrm { C I _ { 9 5 \% } } = [ + 0 . 2 1 4 , + 0 . 2 7 0 ] )$ or $78 \% ( \mathrm { C I } _ { 9 5 \% } = [ 6 8 \% , 9 0 \% ] )$ of the specialists’ advantage. What an adapter carries to another document is therefore mostly a disposition to extract answers from whatever context it is given, which applies to any document; section 4.2 asks how much of it survives once the document is withheld.

## 4.2 Random adapter decomposition

Pooled over the 26 collections of the composition grid that hold 16 documents or more (595 document rows), the random adapter recovers 82 % $( \mathrm { C I _ { 9 5 \% } } = [ 7 7 \% , 8 7 \% ] )$ of the specialists’ OB advantage over the base model and $2 \% ( \mathrm { C I _ { 9 5 \% } }$ = $[ - 1 9 \% , 1 7 \% ] )$ of their cloze advantage, a share not distinguishable from zero (fig. 1; by regime in section D.1).

The cloze share carries a wider CI because its margin is smaller: over these rows the specialists lead the base model by +0.330 OB and by +0.105 in cloze recall, where the base model already reaches 0.81 of their score, against 0.61 of their OB score over the 150 documents; both shares therefore divide what adapters add to a model that already does much of the task. The probe is not blind to recovery: joint training, scored on 25 of these collections, recovers 94 % $( \mathrm { C I } _ { 9 5 \% } = [ 7 0 \% , 1 3 9 \% ] )$ of the cloze advantage and 99 % $( \mathrm { C I _ { 9 5 \% } } = [ 9 3 \% , 1 0 6 \% ] )$ of the OB one (section 5.4).

## 4.3 Retrieval by weight-space proximity

Since an adapter’s weights record which document it read (section 3.3), we ask whether a bank of 120 specialists, searched by weight-space proximity, yields an adapter that carries the content of each of 30 other documents (five per domain) whose own adapters the bank never holds. For each query we compare four settings: (a) the bank adapter nearest to the query’s own using the all-layer Frobenius distance; (b) the one whose document lies nearest in sentence-embedding distance; (c) five averaged random adapters from the whole bank; and (d) five averaged ones from the query’s domain. Before computing any result, we fix the primary contrast: weight-space selection against random adapters from the whole bank on the cloze factual items. The chosen criterion is that weight space counts as better only if that contrast’s CI lay above zero.

An adapter chosen by weight-space proximity is not distinguishable from a random one on either probe (table 2), and neither is one chosen by text embedding: on cloze the primary contrast is $+ 0 . 0 2 0 ~ \ : \mathrm { { ( C I _ { 9 5 \% } ~ = ~ } ~ [ - 0 . 0 3 8 , + 0 . 0 7 5 ] ) }$ and text embedding against random gives −0.001 $( \mathrm { C I _ { 9 5 \% } } = [ - 0 . 0 6 0 , + 0 . 0 6 1 ] )$ , while OB the same weight-space selections score −0.024 (CI<sub>95 %</sub> = $[ - 0 . 0 6 1 , + 0 . 0 1 3 ] )$ against random. These CIs are wide by design: on cloze the query’s own adapter scores 0.091 above a random one (0.507 against 0.415), and the primary contrast’s CI reaches +0.075, about four fifths of that margin, so the run finds no advantage for weight-space selection but cannot rule out one nearly as large as that of the query’s own adapter. Over these 30 documents even the own adapter is not distinguishable from the base model on cloze (+0.085), although over all 150 its gain is clear of zero; a share of that margin would have an undetermined denominator, so we report retrieval as differences in recall, and fig. 1 carries no retrieval point. With one adapter selected, every arm that uses a foreign adapter lies within 0.03 of the base model on cloze, yet recovers 70 % to 80 % of the own adapter’s OB advantage over it. Weight space places its choice in the query’s domain for 26/30 queries, where random choice would give 5/30, since each domain holds a sixth of the bank: geometry that sorts adapters by domain did not, in our bank, pick out adapters that help with a document’s content. Section D.2 gives the criterion’s timeline and all 24 declared contrasts.

<table><tr><td>Adapter applied</td><td>OB  $F _ { 1 }$ </td><td>Cloze recall</td></tr><tr><td>None</td><td>0.510</td><td>0.422</td></tr><tr><td>Weight space</td><td>0.726</td><td>0.435</td></tr><tr><td>Text embedding</td><td>0.757</td><td>0.414</td></tr><tr><td>Random (bank)</td><td>0.750</td><td>0.415</td></tr><tr><td>Random (domain)</td><td>0.757</td><td>0.395</td></tr><tr><td>Query&#x27;s own</td><td>0.818</td><td>0.507</td></tr></table>

Table 2: OB token $F _ { 1 }$ on all four held-out items and cloze recall on the three factual items. One adapter is selected for each of 30 query documents from a bank of 120 that never holds the query’s own. The random rows average five draws, from the whole bank and from the query’s domain respectively.

## 5 Composition

Since OB scores mostly credit a behaviour that every adapter shares, we score the composition grid of section 2 on both probes.

## 5.1 Merges on open book

At rank 8 the three merges keep between 0.93 and 1.01 of the specialists’ OB score over the nine cells <sub>with</sub> <sub>N</sub> <sub>≥</sub> <sub>4</sub> <sub>(table</sub> <sub>3):</sub> √<sub>LERP</sub> <sub>preserves</sub> <sub>0.95</sub> <sub>to</sub> 1.01, update merging 0.93 to 0.99, and TIES 0.94 to 1.00— against 0.88 to 0.97 for random. Paired over the collections of each cell, $\surd \mathrm { L }$ ERP lies above random in all nine cells, by 0.014 to 0.117, with the CIs excluding zero in six; update merging’s CIs exclude zero in six cells as well, and those of TIES in seven (section E.2). We do not test the merges against parity with the specialists (update merging lies below it in all nine cells), because a retention near one says little: the reference is the random adapter, which already recovers about four fifths of the specialists’ OB advantage over the base model (section 4.2).

![](images/eff6428f35ca4669c2d800afd7c84a505d001547bcf09d5bced25ae0c8dcc007.jpg)  
Figure 1: Share of the specialists’ advantage over the base model that each method recovers, OB on all four held-out items against cloze on the factual items, at $N \geq 1 6 .$ . Each point is a ratio of means with its 95 % CI from 10 000 bootstrap resamples over documents; the dotted lines mark full recovery on each probe, and the dashed diagonal indicates equal recovery on both. The shares of no context and swapped context are divided by the natural adapter’s advantage over the base model on the documents each is scored on — all 150 for no context, and its 99 donor documents for swapped context; the no-context adapter’s cloze CI, [−276 %, −74 %], runs below the axis.

<table><tr><td>Reg.</td><td>N</td><td> $\scriptstyle { \sqrt { \mathrm { L E R P } } }$  UM</td><td>TIES</td><td>Rand.</td></tr><tr><td rowspan="4">Cross</td><td>4 0.996</td><td>0.961</td><td>0.980</td><td>0.878</td></tr><tr><td>8 1.011</td><td>0.994</td><td>1.004</td><td>0.963</td></tr><tr><td>16 0.948</td><td>0.932</td><td>0.942</td><td>0.933</td></tr><tr><td>25 0.978 32 0.954</td><td>0.961 0.960</td><td>0.973 0.968</td><td>0.918 0.941</td></tr><tr><td rowspan="4">With.</td><td>4</td><td>0.982 0.948</td><td>0.990</td><td>0.917</td></tr><tr><td>8</td><td>0.986 0.976</td><td>0.985</td><td>0.944</td></tr><tr><td>16</td><td>0.981 0.989</td><td>0.972</td><td>0.963</td></tr><tr><td>25 0.996</td><td>0.983</td><td>0.987</td><td>0.968</td></tr></table>

Table 3: OB retention at rank 8 in the nine cells with $N \geq 4$ . Each document’s $F _ { 1 }$ under the method is divided by that of its own specialist, averaged over documents, and then over the cell’s sets; section E.2 gives the scaling error, the controls, and every paired CI. ‘Reg. stands for regime, ‘UM’ for update merging, ‘rand.’ for random, and ‘with.’ for within; random averages five out-of-collection adapters per set.

## 5.2 Scale and cross terms

Scaling each summed factor by $N ^ { - 1 / 2 }$ , as $\surd$ ERP does, gives $\begin{array} { r } { N ^ { - 1 } \sum _ { i } \mathbf { B } _ { i } \mathbf { A } _ { i } + N ^ { - 1 } \sum _ { i \neq j } \mathbf { B } _ { i } \mathbf { A } _ { j } } \end{array}$ (section E.1): so each adapter’s own update is weighted by $1 / N$ , as in update merging, together with $N ( N - 1 )$ cross terms that pair one adapter’s output directions with another’s input directions and belong to no trained adapter. Averaging each factor at $1 / N$ instead, as our first LERP runs did through the error described in section 6, divides both sums by a further N, which against $\surd$ ERP lowers OB retention by 0.176 to 0.348, every CI excluding zero.

Update merging matches $\surd \mathrm { L E R P }$ in own-term weight and parameter count, truncating where $\surd \mathrm { L E R P }$ adds the cross terms: OB it scores between 0.035 below $\surd \mathrm { L E R P }$ and 0.008 above $\mathbf { i t } ,$ and the four of nine CIs that exclude zero all favour ${ \sqrt { \operatorname { L E R P } } } ;$ on cloze, pooled over the 56 collections, $\surd \mathrm { L E R P }$ scores $+ 0 . 0 0 9 \mathrm { \ : \ : ( C I _ { 9 5 \% } \ : \ : = \ : }$ $[ - 0 . 0 2 0 , + 0 . 0 3 8 ] )$ above update merging, three of the four cells whose CIs exclude zero favouring √<sub>LERP and one favouring update merging. The</sub> gap that Tang et al. (2025a) attribute to the cross terms also carries the $1 / N ^ { 2 }$ own-term scale of their factor merge (section E.2); at matched own-term weight the cross terms cost about ten points on the eight vision tasks to which Zheng et al. (2025) apply them, against at most 0.035 of OB retention on our documents. Truncating to rank 8 costs at most 0.034 of retention, TIES differs from update merging by at most 0.041 in either direction, and a tuned merging coefficient adds nothing over $1 / N$ (section E.2).

## 5.3 Joint training

Pooled over the collections of 16 documents or more, joint training exceeds LERP, update merging, and TIES by 0.038 to 0.045 of the specialists’ OB score and by 0.172 to 0.183 of their cloze recall, no CI’s lower end falling below +0.023 OB or +0.114 on cloze, although per cell the ordering is less uniform (section E.3). At cross-domain with $N = 3 2$ joint training is not distinguishable from parity with the specialists, while its lead over random excludes zero: a 16-fold training budget changes retention by less than 0.02 (section E.3).

## 5.4 Merges on cloze

At 16 documents or more the merges recover 12 % to 16 % of the specialists’ cloze advantage over the base model, each CI including zero, against $8 7 \%$ to 89 % of their OB advantage (section 4.2). Paired contrasts place them between the random adapter and joint training (fig. 2): pooled over the 56 collections, they exceed random on cloze retention by 0.043 to 0.052, each CI excluding zero, about a third of joint training’s 0.150 $( \mathrm { C I _ { 9 5 } \% = }$ [0.087, 0.212]), and at $N \ge 1 6$ alone $1 1 / 1 5$ contrasts with random include zero, against one of joint training’s five. As aggregate ratios at these sizes the merges keep 0.80 to 0.87 of the specialists’ cloze recall and a random adapter 0.78 to 0.83 (close to the base model’s 0.81 over the same rows), while joint training keeps 0.955 to 1.015 and, since the CI on its share reaches 100 % from below, is not distinguishable from full recovery.

## 6 Four silent errors

Each of these errors produced a clean result we were prepared to report, and none stopped a run or raised an error.

A training-length default. In TRL 1.1.0 and 1.2.0 (von Werra et al., 2020), under which our adapters trained, SFTConfig.max\_length defaults to 1024 tokens and truncation keeps the start of an over-long string; since our template places the document before the question and answer, in 44 % of documents no training example kept any token of its answer. Under that default adapters trained on a document’s questions without its text scored 0.168 $( \mathrm { C I _ { 9 5 \% } } = [ 0 . 1 2 0 , 0 . 2 1 5 ] )$ above adapters trained with it OB, whereas with a 2048-token window they score 0.049 $( \mathrm { C I _ { 9 5 \% } } = \ : [ 0 . 0 2 5 , 0 . 0 7 4 ] )$ below them, and a domain’s change in this gap— which mirrors what its natural adapters lost — falls as its truncation rate rises (Spearman $\rho = - 0 . 9 4 3$ $p = 0 . 0 1 7 ;$ Pearson $r = - 0 . 9 8 2 , p = 0 . 0 0 3 )$ under each of five definitions of a truncated document (section F.1).

A decoding setting in our own code. Until we removed it, our generation function applied a repetition penalty of 1.1, which evaluations read from earlier caches kept, so penalized and unpenalized scores sat side by side in our result files; we took the resulting spread for irreproducibility in quantized decoding, whereas under one configuration all 150 scores repeat exactly, and penalized denominators had placed update merging’s best cell in section 5.1 above parity, at 1.006 against the 0.994 of table 3.

A scaling error in our own merge. Our LERP implementation multiplied both stored factors by $1 / N$ before forming their product, so each adapter’s own update was scaled to $1 / N ^ { 2 }$ of its trained magnitude rather than $1 / N$ . Because the error grows with N, it looked like a mechanism: retention fell smoothly from 0.82 of the specialists’ score at four documents to 0.62 at 32 across domains (section E.2), a curve we read as the cost of <sub>merging in factor space until</sub> √<sub>LERP showed that</sub> scale accounts for the whole decline (section 5.2); the merge of PEFT 0.18.1 (Mangrulkar et al., 2022), which our runs used, forms $\sqrt \mathrm { L E R P } \mathrm { s }$ update exactly, so the error was ours (section F.2).

A mismatch in evaluation precision. The adapters were trained on a 4-bit base, on which every cloze number in the main text evaluates them unmerged, whereas our first cloze runs had merged each adapter into a bf16 base. Making the footing uniform lowers the specialists’ cloze gain over the base model by 0.045 $( \mathrm { C I _ { 9 5 \% } } = [ 0 . 0 1 1 , 0 . 0 8 0 ] )$ and the synthetic condition’s by 0.059, which is then no longer clear of zero; it levels natural training with its style-normalized rewrite and changes whether the CI excludes zero for eight of the composition grid’s 22 contrasts with random (section F.3).

## 7 Related work

Document adapters and knowledge in weights. Parametric RAG (Su et al., 2025) trains one adapter per passage and, as re-implemented by Tang et al. (2025b), sums the updates of those retrieved for a question; re-scored by answer presence, it does not consistently beat retrieval in context, and a generic question-answering adapter trained on 400 pooled examples, used as a control, leads the authors to conclude that its gains are mostly document-specific— a conclusion we reach, with a control drawn from the same population as the specialists (section 4.2), only on the probe that withholds the document. Back et al. (2026) find routing among per-document adapters to be the main bottleneck, and Rakotonirina et al. (2026), asking questions without the document, find that merged per-document adapters degrade as more documents are retrieved while the merges they compare lie within about one point of each other at five documents — which is consistent with our cloze grid (section 5.4) and with how little the operator matters once scale is right (section 5.2). The reading behaviour of section 4 fits the account of Zhou et al. (2023), in which instruction tuning mostly shapes how capability acquired in pre-training is expressed, and our training conditions (section 3.1) are a document-level case of the finding of Allen-Zhu and Li (2024) that whether a fact can be extracted depends on how it is presented in training (further work in section G).

![](images/db841d452a967bc2e02ccfc2f817e8d179f15327d2d78d2f1d9ff1c9edcfa154.jpg)  
Figure 2: Advantage over the random adapter in cloze retention (mean of ratios). Values are shown with 95 % CIs, reported in the five cells with $N \geq 1 6$ and pooled over all 56 collections; section E.4 gives every cell.

Merging and its scale. Task arithmetic (Ilharco et al., 2023) and TIES-merging (Yadav et al., 2023) act on the full update, whereas a LoRA adapter stores two factors: averaging them instead of their product changes the scale at which each adapter’s own update enters as well as adding the cross terms that Prabhakar et al. (2025) note and Zheng et al. (2025) isolate (section 5.2).

Adapter selection. LoraHub (Huang et al., 2024) weights adapters on a few target examples, and AdapterFusion (Pfeiffer et al., 2021) learns attention over several: both presuppose that which adapter is applied matters, a presupposition that at document granularity our cloze probe supports and OB scores barely do (section 4.2).

## 8 Conclusion

On Llama-3.1-8B-Instruct, OB scoring mostly credits a reading behaviour every adapter shares: a random adapter recovers about four fifths of the specialists’ OB advantage over the base model and 2 % of their cloze advantage, a share not distinguishable from zero. At 16 documents or more, merges that recover 87 % to 89 % of the specialists OB advantage recover 12 % to 16 % of their cloze advantage, each CI including zero, against 94 % for joint training: so OB scores overstate how much of a document’s content survives merging or substitution. Content does reach the weights, whose geometry records what each adapter read; yet in our bank that geometry did not pick out adapters that help with another document’s content on either probe, although with thirty queries the cloze CI still admits an advantage of about four fifths of what a document’s own adapter gains over a random one.

Three recommendations follow: (a) score what an adapter holds with a probe that withholds the document, beside the OB one; (b) measure against a random adapter averaged over several draws; and (c) evaluate specialists, base model, and composed adapters under one precision and decoding configuration, with one set of denominators per table. For a collection of up to 32 documents known in advance, joint training keeps what merges lose, at the cost of retraining whenever the collection changes.

## Limitations

Every result except the geometry rests on one base model, Llama-3.1-8B-Instruct, adapted at one rank and under one training schedule on 150 encyclopedic articles in six domains, so we cannot say how the shares of section 4 move with model size, adapter capacity, or genre; the effect of the traininglength default on OB scores was likewise measured on this model alone. That schedule passes over each specialist’s eight training strings about a thousand times, with the loss over the whole string, most of whose tokens are the article; such a loss would lead one to expect that the geometry follows the text an adapter read (section 3.3), and every adapter learns the shared template from the same repetition. Composition was tested on collections fixed in advance and no larger than 32 documents, where joint training is not yet distinguishable from parity with the specialists (section 5.3), so both the size at which a rank-8 joint adapter stops keeping pace and the incremental case, in which composition would have to undercut retraining on each new document, remain untested.

Each document contributes three factual items and one true/false item, enough to estimate means over documents and collections but too few to characterize a single document; this is why the retrieval run cannot resolve content, since over its 30 queries even the specialist adapter is not distinguishable from the base model on cloze (+0.085 with $\mathrm { C I _ { 9 5 \% } } = [ - 0 . 0 1 1 , + 0 . 1 8 9 ]$ ; section 4.3), and an advantage for weight-space selection of up to about four fifths of the own adapter’s margin over a random one remains possible. The two probes also score different items, because cloze shares use the factual items alone, whereas OB shares include the true/false item, on which the specialists score below the base model (section A.2); since the items are generated by one model and validated only by string match against the source, a question that admits more than one answer, or a correct fill phrased differently, would go unrecognized, an error we did not measure.

Wikipedia articles have probably been seen in pre-training, a criticism Rakotonirina et al. (2026) raise against evaluating parametric retrieval on such text, and our base model already reaches 0.81 of the specialists’ cloze recall over the composition grid’s rows (section 4.2); every cloze result is stated against that base model, so what we measure is how adaptation changes the availability of facts the model may already hold, and whether a random adapter recovers as little of the cloze advantage on unseen documents is untested. All our adapters also share one training format, the instruction template in which OB evaluation presents the document (section 2), so the four fifths that a random adapter recovers measure a behaviour our training instilled in every adapter alike; a bank of adapters trained in different formats might share less of it. The base model, moreover, is scored OB in that raw template rather than in its own chat template, so part of what every adapter adds may be adaptation to the format rather than to reading.

The retrieval bank never holds the query document’s own adapter, so that result bears on substituting one document’s adapter for another’s and says nothing about banks that hold every document’s adapter, as parametric RAG does. Our account of why the cross terms of section 5.2 cost little here and about ten points across eight vision tasks (Zheng et al., 2025)— that updates sharing learned input directions let the cross terms re-add a common component — is a hypothesis we have not tested; on cloze, moreover, the merges keep so little of the specialists’ advantage at 16 documents or more that a cost of the cross terms would have little room to show.

Two checks rest on less evidence than the main results: the footing of the archived 4-bit composition numerators is inferred from the runner’s code and supported by 64 of the 66 evaluations they share with the retrieval run agreeing at six decimals (section F.4), and the check on the chat wrapper rests on 35 documents, whose CI on the change, [−0.069, +0.183], includes a change larger than the gain it checks (section A.3). The retrieval contrast is the only comparison whose decision criterion we fixed before computing it, and the other CIs carry no correction for how many we computed, so a CI that only just excludes zero carries little weight by itself. A cell’s CI in the composition grid, moreover, is resampled over only five collections, which share documents within and across cells, so it rests on few distinct resamples and will undercover; we therefore take the pooled contrasts as the evidence and the counts of cells whose CIs exclude zero as description.

## References

Zeyuan Allen-Zhu and Yuanzhi Li. 2024. Physics of language models: Part 3.1, knowledge storage and extraction. In Proceedings ofthe 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 1067–1077.

Jason Ansel, Edward Yang, Horace He, Natalia Gimelshein, Animesh Jain, Michael Voznesensky, Bin Bao, Peter Bell, David Berard, Evgeni Burovski, Geeta Chauhan, Anjali Chourdia, Will Constable, Alban Desmaison, Zachary DeVito, Elias Ellison, Will Feng, Jiong Gong, Michael Gschwind, and 30 others. 2024. PyTorch 2: Faster machine learning through dynamic Python bytecode transformation and graph compilation. In Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2, pages 929–947.

Anthropic. 2026. Introducing Claude Sonnet 4.6. Accessed 22 Oct. 2026.

Seungju Back, Dongwoo Lee, Naun Kang, Taehee Lee, S. K. Hong, Youngjune Gwon, and Sungjin Ahn. 2026. Understanding LoRA as knowledge memory: An empirical analysis. In Proceedings of the 43rd International Conference on Machine Learning.

Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. 2023. QLoRA: Efficient finetuning of quantized LLMs. Advances in Neural Information Processing Systems, 36:10088–10115.

Zorik Gekhman, Gal Yona, Roee Aharoni, Matan Eyal, Amir Feder, Roi Reichart, and Jonathan Herzig. 2024. Does fine-tuning LLMs on new knowledge encourage hallucinations? In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing, pages 7765–7784.

Sylvain Gugger, Lysandre Debut, Thomas Wolf, Philipp Schmid, Zachary Mueller, Sourab Mangrulkar, Marc Sun, and Benjamin Bossan. 2022. Accelerate: Training and inference at scale made simple, efficient and adaptable. Accessed 22 Oct. 2026.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-rank adaptation of large language models. In Proceedings of the 10th International Conference on Learning Representations.

Chengsong Huang, Qian Liu, Bill Yuchen Lin, Tianyu Pang, Chao Du, and Min Lin. 2024. LoraHub: Efficient cross-task generalization via dynamic LoRA composition. In Proceedings of the 1st Conference on Language Modeling.

Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Ludwig Schmidt, Hannaneh Hajishirzi, and Ali Farhadi. 2023. Editing models with task arithmetic. In Proceedings ofthe 11th International Conference on Learning Representations.

Llama Team. 2024. The Llama 3 herd of models. Computing Research Repository, arXiv:2407.21783.

Ilya Loshchilov and Frank Hutter. 2019. Decoupled weight decay regularization. In Proceedings of the 7th International Conference on Learning Representations.

Sourab Mangrulkar, Sylvain Gugger, Lysandre Debut, Younes Belkada, Sayak Paul, Benjamin Bossan, and Marian Tietz. 2022. PEFT: State-of-the-art parameter-efficient fine-tuning methods. Accessed 22 Oct. 2026.

Meta Llama. 2024a. Llama-3.1-8B-Instruct. Accessed 22 Oct. 2026.

Meta Llama. 2024b. Llama-3.2-3B-Instruct. Accessed 22 Oct. 2026.

Aniello Panariello, Daniel Marczak, Simone Magistri, Angelo Porrello, Bartłomiej Twardowski, Andrew Bagdanov, Simone Calderara, and Joost van de Weijer. 2025. Accurate and efficient low-rank model merging in core space. Advances in Neural Information Processing Systems, 38:61793–61825.

Jonas Pfeiffer, Aishwarya Kamath, Andreas Rücklé, Kyunghyun Cho, and Iryna Gurevych. 2021. AdapterFusion: Non-destructive task composition for transfer learning. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, pages 487–503.

Akshara Prabhakar, Yuanzhi Li, Karthik Narasimhan, Sham Kakade, Eran Malach, and Samy Jelassi. 2025. LoRA soups: Merging LoRAs for practical skill composition tasks. In Proceedings of the 31st International Conference on Computational Linguistics: Industry Track, pages 644–655.

Martino M. L. Pulici, Cuong Xuan Chu, Evgeny Kharlamov, and Volker Tresp. 2026. A storage–retrieval gap in parametric knowledge graph memory. In Proceedings of the 3rd Scaling Knowledge Graphs for Industry Workshop.

Qwen. 2026a. Qwen3.5-2B. Accessed 22 Oct. 2026.

Qwen. 2026b. Qwen3.5-9B. Accessed 22 Oct. 2026.

QwenTeam. 2026. Qwen3.5: Towards native multimodal agents. Accessed 22 Oct. 2026.

Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. SQuAD: 100,000+ questions for machine comprehension of text. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pages 2383–2392.

Nathanaël Carraz Rakotonirina, Momchil Hardalov, Gonzalo Iglesias, and Adrià de Gispert. 2026. Where should a document live: Context, representations, or parameters? Computing Research Repository, arXiv:2609.17346.

Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence embeddings using siamese BERTnetworks. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, pages 3982–3992.

Sentence Transformers. 2021. all-mpnet-base-v2. Accessed 22 Oct. 2026.

George Stoica, Pratik Ramesh, Boglarka Ecsedi, Leshem Choshen, and Judy Hoffman. 2025. Model merging with SVD to tie the knots. In Proceedings of the 13th International Conference on Learning Representations, pages 4501–4519.

Weihang Su, Yichen Tang, Qingyao Ai, Junxi Yan, Changyue Wang, Hongning Wang, Ziyi Ye, Yujia Zhou, and Yiqun Liu. 2025. Parametric retrieval augmented generation. In Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 1240–1250.

Dennis Tang, Prateek Yadav, Yi-Lin Sung, Jaehong Yoon, and Mohit Bansal. 2025a. LoRA merging with SVD: Understanding interference and preserving performance. In Proceedings of the 2nd Workshop on Reliable and Responsible Foundation Models.

Minghao Tang, Shiyu Ni, Jingtong Wu, Zengxin Han, and Keping Bi. 2025b. Understanding parametric knowledge injection in retrieval-augmented generation. Computing Research Repository, arXiv:2510.12668.

Leandro von Werra, Younes Belkada, Lewis Tunstall, Edward Beeching, Tristan Thrush, Nathan Lambert, Shengyi Huang, Kashif Rasul, and Quentin Galloué- dec. 2020. TRL: Transformers reinforcement learning. Accessed 22 Oct. 2026.

Wikimedia Foundation. 2026. Wikipedia, the free encyclopedia. Accessed 22 Oct. 2026.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Perric Cistac, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. 2020. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 38–45.

Prateek Yadav, Derek Tam, Leshem Choshen, Colin A. Raffel, and Mohit Bansal. 2023. TIES-merging: Resolving interference when merging models. Advances in Neural Information Processing Systems, 36:7093–7115.

Le Yu, Bowen Yu, Haiyang Yu, Fei Huang, and Yongbin Li. 2024. Language models are Super Mario: Absorbing abilities from homologous models as a free lunch. In Proceedings ofthe 41st International Conference on Machine Learning, volume 235 of

Proceedings of Machine Learning Research, pages 57755–57775.

Ziyu Zhao, Tao Shen, Didi Zhu, Zexi Li, Jing Su, Xuwu Wang, and Fei Wu. 2025. Merging LoRAs like playing LEGO: Pushing the modularity of LoRA to extremes through rank-wise clustering. In Proceedings of the 13th International Conference on Learning Representations.

Shenghe Zheng, Hongzhi Wang, Chenyu Huang, Xiaohui Wang, Tao Chen, Jiayuan Fan, Shuyue Hu, and Peng Ye. 2025. Decouple and orthogonalize: A datafree framework for LoRA merging. Computing Research Repository, arXiv:2505.15875.

Chunting Zhou, Pengfei Liu, Puxin Xu, Srinivasan Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, Susan Zhang, Gargi Ghosh, Mike Lewis, Luke Zettlemoyer, and Omer Levy. 2023. LIMA: Less is more for alignment. Advances in Neural Information Processing Systems, 36:55006–55021.

## A Setup details

## A.1 Training and software

Adapters use rank 8, $\alpha = 1 6$ and dropout 0.05 on all seven projection matrices, and train for 500 steps with AdamW (Loshchilov and Hutter, 2019) at a learning rate of $2 \times 1 0 ^ { - 4 }$ under a cosine schedule with 50 warmup steps and an effective batch size of 16, with the language-modelling loss taken over the whole training string. Sequences are capped at 2048 tokens, which fits every training example, for every adapter we report except the no-context adapters, which trained under the library’s default cap of 1024 tokens; their training strings carry no article, so that cap does not bear on them, and section 6 describes what the same default did to natural adapters trained before we set the cap ourselves. Our runs used PyTorch 2.11.0 (Ansel et al., 2024), Transformers 5.5.3 (Wolf et al., 2020), PEFT 0.18.1 (Mangrulkar et al., 2022), bitsandbytes 0.49.2 (Dettmers et al., 2023), and Accelerate 1.13.0 (Gugger et al., 2022). Adapters trained under TRL 1.2.0 (von Werra et al., 2020), except the natural adapters trained at the 1024- token window, the Qwen3.5-2B adapters, and the no-context adapters; these trained under TRL 1.1.0, whose defaults and truncation code are the same. The geometry of section 3 is replicated with the same hyper-parameters on Llama-3.2-3B-Instruct (Meta Llama, 2024b) and on two models from the Qwen3.5 family (QwenTeam, 2026), Qwen3.5-2B (Qwen, 2026a) and Qwen3.5-9B (Qwen, 2026b); every other experiment uses Llama-3.1-8B-Instruct (Meta Llama, 2024a). On Llama-3.1-8B-Instruct each adapter has 20 971 520 trainable parameters, about 0.26 % of the base model’s 8.0 billion; every run was a single-GPU job on a heterogeneous cluster pool of NVIDIA A100, H100, H200, and B200 GPUs, except the cloze evaluations on the 4-bit footing, which all ran on A100s (section F.4), and we estimate the total budget of all our experiments, superseded and preliminary runs included, at a few hundred GPU hours.

## A.2 Held-out items and the metric

Every document’s held-out set holds three factual questions and one true/false item, and the true/- false item does not behave like the others: OB the base model answers it correctly for 0.987 of documents and the natural adapter for 0.933, so on all four items it narrows the specialists’ advantage rather than adding to it, while the swapped-context adapter, scored on its source document with its donor’s text in context, answers it correctly for only 0.413, and CB every training condition answers it less often than the base model (0.520 to 0.673 against 0.747). The cloze probe therefore scores the factual items alone, with the true/false item reported beside them; OB scores include it, because the composition and transfer runs keep no per-item OB scores, which leaves every OB share in this paper on all four items.

Token-level $F _ { 1 }$ penalizes answers that are correct but longer than the reference, so a composed adapter that grew more verbose would lose retention without losing knowledge, and we checked for this on a pilot grid of thirty collections, five in each of six cells with $N \leq 8$ . Joint and specialist answers there average 10.4 and 11.1 words after the metric’s normalization, and retention computed from recall alone, which added tokens cannot lower, exceeds $F _ { 1 }$ -based retention by 0.002 for joint training and 0.005 for TIES, pooled over the documents of all six cells; since neither the sign nor the size of these differences points to a verbosity artefact, we keep $F _ { 1 }$ as the metric.

## A.3 Cloze harness

Unlike the QA evaluations, the cloze harness wraps each prompt in the model’s chat template, which adds the template’s default system header and, through the tokenizer, a second beginning-ofsequence token; the wrapper is the same for every arm, the specialist and base-model references included. Its length caps barely bind: prompts run from 60 to 108 tokens against a cap of 512, and of the 900 cloze generations of the specialists and the base model, 8 reach the limit of 64 new tokens, all of them the specialists’, which can have lowered the specialists’ mean recall by at most $8 / 4 5 0 = 0 . 0 1 8 ,$ since only a generation that was cut can lose its fill. Because the adapters trained without a chat template, we reran 35 documents with the trainingformat prompt instead: under it the specialists return an empty answer on 14 of their 105 cloze items and the base model on none, which the records keep only as decoded strings, so what the specialists emitted in their place cannot be recovered, and the wrapper does not measurably change the specialists’ cloze gain over the base model: chat wrapper minus training format, the gain changes by $+ 0 . 0 5 7 \left( \mathrm { C I } _ { 9 5 \% } = \left[ - 0 . 0 6 9 , + 0 . 1 8 3 \right] \right)$ with empty answers scored zero and by $+ 0 . 0 0 4 ~ ( \mathrm { C I } _ { 9 5 \% } ~ =$ $[ - 0 . 1 4 7 , + 0 . 1 5 1 ] )$ on the items both prompts answer, so the subset cannot exclude a change about the size of the gain in either direction. Those documents are the 30 retrieval queries of section 4.3 and five chosen because their specialists recalled none of the factual items on the earlier footing, so their means are not the collection’s.

## B Prompts

Unicode punctuation is transliterated in the monospace rendering below; the released prompt files carry the original characters.

## B.1 Instruction template

The instruction template was fixed and used across all documents, both for training and inference, as

You are a factual   
,→ question-answering assistant.   
,→ Answer the question below   
,→ using only the information   
,→ provided. Be concise and   
,→ accurate.   
Context: {context}   
Question: {question}   
Answer:

## B.2 Generation of question-answering pairs

All QA pairs, probe items, and document variants were generated with Claude Sonnet 4.6 (Anthropic,

,→ sentence.

2026) at 0.3 temperature with a 2048-token generation limit, using the prompts below. Each generation was validated by string match against the source document and regenerated on failure.

Base pairs. The base QA pairs are generated using the system prompt

You are an expert at generating

,→ factual question-answer pairs

,→ for training language models.

,→ Generate high-quality QA pairs

,→ strictly based on the provided

,→ text.

## followed by the user prompt

Read the following text carefully

,→ and generate exactly {n\_total}

,→ question-answer pairs of the

,→ following types:

\- {n\_factual} single-hop

,→ factual questions (direct

,→ facts from the text)

\- {n\_tf} true/false questions

,→ (statement + answer "True" or

,→ "False")

\- {n\_multihop} multi-hop

,→ question (requires connecting

,→ two facts from the text)

## Rules:

1. Questions must be answerable

,→ solely from the provided text.

2. Answers should be concise (1-3

,→ sentences maximum).

3. Avoid trivially easy questions

,→ (e.g. "What is the title?").

4. Do NOT ask the same question

,→ twice.

5. Vary question phrasing and

,→ structure.

6. Return a valid JSON array with

,→ this exact schema:   
[   
{{   
"question": "..."   
"answer": ".

"type": "factual" |   
,→ "true\_false" | "multi\_hop"   
}},   
]   
Return ONLY the JSON array,   
,→ nothing else.   
Text:   
{text}

## Cloze variants. The cloze variants are generated using the system prompt

You are a precise QA rewriter.

,→ Given a question-answer pair,

,→ fill-in-the-blank cloze

## Rules:

\- The cloze sentence must be a

,→ complete declarative sentence

,→ with exactly one blank (\_\_\_).

\- The blank fills in the key fact

,→ stated in the answer.

\- The answer to the blank is the

,→ shortest phrase that makes the

,→ sentence factually correct,

,→ matching or closely

,→ paraphrasing the original

,→ answer.

\- Do NOT include any explanation.

,→ Output ONLY a JSON object with

,→ keys "cloze" and "fill".

question: "What is the boiling

,→ point of water at sea level?"

answer: "The boiling point of

,→ water at sea level is 100

```csv
,→ degrees Celsius."
```

{"cloze": "The boiling point of

,→ "fill": "100 degrees Celsius"}

```csv
,→ names, dates, quantities,
```

Rewrite the following article   
,→ text in neutral expository   
,→ prose:   
{text}

You are an expert technical   
,→ editor. Your task is to   
,→ rewrite encyclopedia articles   
,→ into neutral, consistent   
,→ expository prose. You must:   
1. Preserve ALL factual content:   
,→ names, dates, quantities,   
,→ causal relationships,   
,→ definitions, and examples.   
2. Write in plain declarative   
,→ sentences. Avoid rhetorical   
,→ questions, vivid metaphors,   
,→ and stylistic flourishes.   
3. Remove Wikipedia-specific   
,→ patterns: lead-sentence   
,→ formulas ("X is a Y that..."),   
,→ "As of [year]", "According   
,→ to...", citation markers like   
,→ [1], navbox fragments, and   
,→ disambiguation notices.   
4. Keep approximately the same   
,→ length as the input (within ±   
,→ 20%).   
5. Do not introduce any claims   
,→ not present in the original.   
6. Do not add section headers or   
,→ bullet points — continuous   
,→ prose only.   
7. Use consistent grammatical   
,→ register across the entire   
,→ text (simple present tense for   
,→ general facts, simple past for   
,→ historical events).   
Return only the rewritten text,   
,→ nothing else.

followed by the user prompt

## B.3 Document variants

Style-normalized. The style-normalized documents are generated using the system prompt

followed by the user prompt

## Synthetic. The synthetic documents are generated using the system prompt

You are an information extraction   
,→ assistant. Given an   
,→ encyclopedia article, extract   
,→ its key content and render it   
,→ as a structured   
,→ fill-in-the-blank template.   
,→ You must:

```json
Return ONLY a JSON object with
,→ this exact schema:
{
"domain": "<string>",
"topic": "<string>",
"definition": "<one-sentence
,→ string>",
"key_facts": ["<fact 1>",
,→ "<fact 2>", "<fact 3>", "<fact
,→ 4>", "<fact 5>", "<fact 6>"]
```

![](images/73e52c5f8282ddd3255d0b3be75c381b5810bfa508ca96ecd6037be36eaca80a.jpg)  
followed by the user prompt

Extract the structured template   
,→ from the following article:   
{text}

## C Content and geometry

## C.1 Training-text conditions

Table 4 repeats the scores of section 3.1 with every CI. OB, the conditions read different texts: the base model and the natural and no-context adapters read the source document, the two rewrites read their own text, and the swapped adapters read the donor document, scored either with their source’s questions, for which no answer is in the context, or with the donor’s.

## C.2 The context swap on its donor documents

Pairing at the level of the 99 distinct donor documents shifts both means slightly from the corpuswide figures of table 1. On that pairing the swapped adapter reaches 84 % of the natural adapter’s OB score on the document it read (section 3.3). On cloze it gains $+ 0 . 0 5 1 \ ( \mathrm { C I } _ { 9 5 \% } \ =$ $\left[ + 0 . 0 0 3 , + 0 . 0 9 9 \right] )$ over the base model on these donors, and the natural adapter leads it by 0.039 $( \mathrm { C I } _ { 9 5 \% } = [ 0 . 0 0 1 , 0 . 0 7 7 ] )$ , a CI that includes zero once the true/false item is also scored, so we give that margin little weight.

## C.3 Update geometry

A LoRA adapter adds to each weight matrix $\mathbf { W } _ { 0 } \in \mathbb { R } ^ { d _ { \mathrm { o u t } } \times d _ { \mathrm { i n } } }$ the update $\Delta \mathbf { W } = \mathbf { B } \mathbf { A }$ , with $\mathbf { B } \in \mathbb { R } ^ { d _ { \mathrm { o u t } } \times r }$ and $\mathbf { A } \in \mathbb { R } ^ { r \times d _ { \mathrm { i n } } }$ , and our geometry compares updates rather than factors. For each layer–module pair we compute the truncated SVD $\Delta { \mathbf W } = { \mathbf U } \mathrm { d i a g } ( { \pmb \sigma } ) { \mathbf V } ^ { \top }$ from QR factorizations of B and $\mathbf { A } ^ { \top }$ , without forming $\Delta \mathbf { W }$ , and normalize the update to unit Frobenius norm by dividing σ by its Euclidean norm, which gives $\tilde { \pmb { \sigma } }$

Distance. For adapters i and $j$ at layer–module pair $\ell ,$ the normalized Frobenius distance is

$$
\begin{array} { r l } & { { d _ { \ell } ( i , j ) } = \left\| \Delta \tilde { \mathbf { W } } _ { i } ^ { \ell } - \Delta \tilde { \mathbf { W } } _ { j } ^ { \ell } \right\| _ { F } } \\ & { \quad \quad \quad = \sqrt { 2 - 2 \left. \Delta \tilde { \mathbf { W } } _ { i } ^ { \ell } , \Delta \tilde { \mathbf { W } } _ { j } ^ { \ell } \right. _ { F } } } \end{array}
$$

whose inner product the SVD factors give as $\tilde { \pmb { \sigma } } _ { i } ^ { \top } ( \mathbf { U } _ { i } ^ { \top } \mathbf { U } _ { j } \odot \mathbf { \bar { V } } _ { i } ^ { \top } \mathbf { V } _ { j } ) \tilde { \pmb { \sigma } } _ { j }$ , with ⊙ the element-wise product of two $r \ \times \ r$ matrices. Over a set L of layer–module pairs the distance is $d ( i , j ) =$ $\begin{array} { r } { \big ( \sum _ { \ell \in \mathcal { L } } d _ { \ell } ( i , j ) ^ { 2 } \big ) ^ { 1 / 2 } } \end{array}$ , and since each layer is normalized before aggregation, layers with larger updates do not dominate it.

Subspace overlap. The distance compares whole updates; to ask instead whether two adapters occupy the same directions, we compare the column spaces of their left singular vectors,

$$
o _ { \ell } ( i , j ) = \frac { \left\| \mathbf { U } _ { i } ^ { \ell ^ { \top } } \mathbf { U } _ { j } ^ { \ell } \right\| _ { F } } { \sqrt { r } }
$$

which equals 1 when the two rank-r subspaces coincide and 0 when they are orthogonal, since U has orthonormal columns and $\Vert \mathbf { U } _ { i } ^ { \top } \mathbf { U } _ { j } \Vert _ { F } ^ { 2 } =$ $\textstyle \sum _ { k } \cos ^ { 2 } \theta _ { k }$ for the principal angles $\theta _ { k }$ between the two subspaces. We report the mean of $o _ { \ell }$ over all 224 layer–module combinations.

Two subspaces drawn uniformly at random give an expected overlap of ${ \sqrt { r / d } } .$ , which is not a single number here because grouped-query attention gives the projections different output dimensions: 0.044 for the query, output, and down projections at $d \ : = \ : 4 0 9 6$ , 0.088 for the key and value projections at d = 1024, and 0.024 for the gate and up projections at $d = 1 4 3 3 6$ . Weighting these by the 224 layer–module combinations gives a reference of 0.051, which simulation over 10 000 random frames per shape reproduces to within 1 %. The measured 0.155 is therefore about three times the random floor.

Update cosine. The alignment of the full updates follows from the distances already computed, since for unit-normalized perturbations cos $( \Delta \mathbf { W } _ { i } , \Delta \mathbf { W } _ { j } ) = 1 - d _ { \ell } ( i , j ) ^ { 2 } / 2$ at each layer, and averaging over the 224 combinations gives the mean cosine of section 3. The two statistics answer different questions: the subspace overlap asks whether adapters recruit the same directions, and the cosine asks whether the resulting updates point the same way.

<table><tr><td>Condition</td><td>Open-book  $F _ { 1 }$ </td><td>Closed-book  $F _ { 1 }$ </td><td>∆ cloze recall</td></tr><tr><td>Base model</td><td>0.504 [0.485, 0.522]</td><td>0.320 [0.303, 0.337]</td><td></td></tr><tr><td>Natural</td><td>0.822 [0.803, 0.841]</td><td>0.399 [0.374, 0.424]</td><td>+0.094[+0.056, +0.134]</td></tr><tr><td>Style-normalized</td><td>0.708 [0.685, 0.731]</td><td>0.419 [0.394, 0.444]</td><td>+0.097[+0.054, +0.141]</td></tr><tr><td>Synthetic</td><td>0.649 [0.621, 0.677]</td><td>0.443 [0.416, 0.470]</td><td>+0.034 [-0.014, +0.083]</td></tr><tr><td>No context</td><td>0.773 [0.748, 0.796]</td><td>0.426 [0.396, 0.455]</td><td>-0.132 [-0.176, -0.087]</td></tr><tr><td>Swapped, on source</td><td>0.424 [0.394, 0.455]</td><td>0.456 [0.427, 0.484]</td><td> $- 0 . 0 2 8 \left[ - 0 . 0 5 8 , + 0 . 0 0 3 \right]$ </td></tr><tr><td>Swapped, on donor</td><td>0.686 [0.656, 0.715]</td><td>0.428 [0.401, 0.454]</td><td> $+ 0 . 0 5 6 \left[ + 0 . 0 1 7 , + 0 . 0 9 7 \right]$ </td></tr></table>

Table 4: The training-text conditions with every CI: OB and CB token $F _ { 1 }$ on all four held-out items and cloze recall on the three factual items as a paired difference from the base model (on the donor document for the donor row). Values are computed over all 150 documents, with 95 % percentile bootstrap CIs over documents from 10 000 resamples. OB and CB use the 4-bit base with greedy decoding, a 2048-token window, no repetition penalty and the training template; cloze uses the 4-bit base with each adapter unmerged and the chat wrapper of section A.3.

A reference for the cosine. The cosine also has a reference set by the parametrization of LoRA rather than by chance. Suppose two rank-r adapters on a module with input dimension $d _ { \mathrm { i n } }$ receive the same gradient G with respect to the full weight, and their lora\_A matrices, initialized independently at random, stay near their initial values. Under plain gradient steps B then accumulates in proportion to $\mathbf { G A } ^ { \top }$ , so each update is proportional to $\mathbf { G A } ^ { \top } \mathbf { A }$ , and $\mathbf { A } ^ { \top } \mathbf { A }$ is, up to scale, close to a projection P onto a random r-dimensional subspace of the input space with $\mathbb { E } [ \mathbf { P } ] \ : = \ : ( r / d _ { \mathrm { i n } } ) \mathbb { I }$ . The expected inner product of the two updates is then about $( r / d _ { \mathrm { i n } } ) ^ { 2 } \bar { \| } \mathbf { G } \| _ { F } ^ { 2 }$ against a squared norm of about $( r / d _ { \mathrm { i n } } ) \lVert \mathbf { G } \rVert _ { F } ^ { 2 }$ for each, a cosine of about $r / d _ { \mathrm { i n } } .$ : 0.0020 for the six projections whose input dimension is 4096, 0.0006 for the down projection at 14 336, and 0.0018 averaged over the 224 combinations. A mean above this reference requires the lora\_A matrices of different adapters to have become correlated, that is, to have moved toward shared input directions.

Spread of the cosine. Across the 11 175 pairs the mean cosine of 0.0066 has a standard deviation of 0.0012. Independent random rank-8 updates of the trained shapes give a cosine of zero with a standard deviation of about $2 \times 1 0 ^ { - 5 }$ , and the smallest observed pair lies 192 of those standard deviations above zero. We normalize each layer before averaging, so the mean cosine is a statement about the typical layer rather than about the whole update: a global cosine would weight layers by their magnitude, and a few high-norm modules could in principle carry alignment that this average conceals.

Overlap and semantic similarity. Subspace overlap correlates with semantic similarity between the training documents at $r = 0 . 1 2 2$ . The 11 175 pairs come from only 150 adapters and are therefore not independent, so we assess significance by permuting document labels across adapters rather than across pairs; the observed correlation sits 4.4 standard deviations above that null, at $p < 0 . 0 0 1$ over 10 000 permutations. The correlation accounts for roughly 1.5 % of variance, so adapters trained on closely related documents remain very nearly as orthogonal as adapters trained on unrelated ones.

## C.4 Geometry across architectures

Distances can be aggregated over every layer– module combination or over a selected subset. We had fixed a threshold of $\rho \geq 0 . 2 5 0$ before these runs, intending to select layers clearing it; no single layer does, the strongest reaching 0.241, so we take the highest-correlating layers instead, eight on the base model and between seven and twelve on the others, and call that set late, since on every architecture it falls in the final third of the network. Llama-3.1-8B’s shortfall of 0.008 under full aggregation reflects where the threshold was set rather than any measured uncertainty.

Selecting layers by their correlation and then reporting that correlation uses the same data twice. The held-out $\rho$ of section 3 comes from splitting the adapters, choosing layers on one half and evaluating on the other over 200 stratified splits; against 0.318 in sample its optimism is 0.022, and six of the eight layers are chosen in over 70 % of splits. Neighbourhood purity over the selected layers is more sensitive to the selection, falling from 0.746 in sample to 0.616 held out, so the main text reports purity over all layers, where nothing was selected, with the adapter excluded from its own neighbourhood.

Grouping the 150 adapters into six clusters recovers domain labels at an normalized mutual information (NMI) of 0.511 at full aggregation and 0.587 over the selected layers. The assignments that disagree with the labels are structured rather than scattered: the economics adapters split into a macroeconomic group, holding central banking, fiscal and monetary policy, inflation and the national accounts, and a group holding game theory, market structure and welfare, while the history adapters separate political and military events from social and economic history.

Table 5 gives the correlations for all four architectures before and after the truncation fix, together with the clustering metrics and the degree of overlap between the layer sets selected in each case. The correction barely moves the correlation on any architecture: under full aggregation it drops by at most 0.019, and over the late layers it moves by at most 0.021 in either direction, and the layers carrying it are largely the same ones, which is why we treat the geometry result as independent of the training-window error rather than as another of its casualties.

## D Behaviour

## D.1 The random-adapter baseline

Pooled over the collections of 16 documents or more, the random adapter’s OB share (section 4.2) is $0 . 7 9 \left( \mathrm { C I } _ { 9 5 \% } = [ 0 . 7 3 , 0 . 8 5 ] \right)$ ) across domains and $0 . 8 7 \ : ( \mathrm { C I } _ { 9 5 \% } = [ 0 . 8 1 , 0 . 9 3 ] )$ within one, a figure that includes $N = 2 5$ , where every random adapter necessarily comes from another domain.

The random baseline averages five independently drawn out-of-collection adapters per collection, and the averaging is needed: individual draws range from 0.57 to 1.19 of the specialists’ score, and at cross-domain $N \ = \ 4$ a single draw returned an adapter at parity with the specialists, against which that cell’s contrast with TIES would no longer have excluded zero. Table 6 separates the two sources of spread, between the five draws made for a collection and between the collections themselves.

## D.2 Retrieval by weight-space proximity

The criterion and the five-draw analysis of section 4.3 were fixed before the primary contrast was computed: the commit in our repository that declares them fixed is dated 22 September 2026, and the centred arm described below was added the next day, after a two-document pilot on the earlier bf16 footing whose output was discarded without its scores being viewed, and before the full run.

Two further declared contrasts on cloze at top-1 complete those of section 4.3: weight space against text embedding $+ 0 . 0 2 1 \ : ( \mathrm { C I } _ { 9 5 \% } =$ $[ - 0 . 0 3 6 , + 0 . 0 7 7 ] )$ , and weight space against a random adapter from the query’s own domain +0.040 $( \mathrm { C I _ { 9 5 \% } } = [ - 0 . 0 2 0 , + 0 . 0 9 8 ] )$ .

The reference rows fit section 4.2, a reading we offer as consistent with the data rather than as a prediction we made. At top-1 the arms that use a foreign adapter lie between −0.027 and +0.016 of the base model on cloze. A bank without the query’s own adapter can help OB only through the behaviour every adapter carries, so selection has little to act on there, and on cloze only through content that other documents share, which no method found: weight space and text embedding place their choice in the query’s domain for 26 and 29 of 30 queries, yet a random adapter from the query’s domain scores no higher on cloze than one from any domain (0.395 against 0.415 at top-1), a comparison we did not declare and report only as description.

Of the 24 contrasts, two exclude zero, and they are one contrast under two overlapping scorings: at top-3, where the three highest-ranked adapters are merged with $\surd \mathrm { I }$ ERP (section 5.2), weight space exceeds text embedding on cloze $\begin{array} { l l l } { \mathbf { b y } } & { + 0 . 0 6 6 \ \left( \mathrm { C I _ { 9 5 \% } } ^ { } = } & { \left[ + 0 . 0 0 9 , + 0 . 1 2 3 \right] \right) } \end{array}$ on the factual items and by $+ 0 . 0 4 9 \mathrm { ( C I _ { 9 5 \% } = }$ $[ + 0 . 0 0 6 , + 0 . 0 9 3 ] )$ with the true/false item, because the text-embedding merge falls below both the random merge (0.377 against 0.403) and the base model while the weight-space merge is not distinguishable from random (+0.040 with $\mathrm { C I _ { 9 5 } \% = }$ $[ - 0 . 0 1 7 , + 0 . 0 9 9 ] )$ ; we build nothing on it. The centred variant of weight-space proximity, added on the strength of a diagnostic of hubs that uses no scores and kept outside the decision criterion, is not distinguishable from random (+0.022 with $\mathrm { C I _ { 9 5 \% } } = \ [ - 0 . 0 3 9 , + 0 . 0 8 5 ] )$ or from uncentred weight space (+0.003 with $\mathrm { { C I } _ { 9 5 } \% = }$ $[ - 0 . 0 5 0 , + 0 . 0 5 4 ] )$ . Chosen for each query with hindsight, the best bank adapter reaches 0.953 OB, but it is chosen on the items on which it is then scored, so it marks an upper bound rather than a target. To ask whether a selection rule could approach it, we split each document’s held-out items, chose the best adapter on one part and scored that adapter on the other. Choosing on two factual items and scoring on the third together with the true/false item, the chosen adapter beats a random one by 0.026 and 0.025 in the two such splits we ran and is also the best on the scoring part for 0.27 and 0.20 of the 30 documents; choosing on one factual item together with the true/false item, which the bank’s adapters answer correctly in 0.968 of the 3600 query–adapter cells, selects on little more than noise, and the chosen adapter is then the best on the scoring part for none of the documents. Restricted to the factual items, choosing on two and scoring on the third, averaged over the three such splits, the chosen adapter gains 0.018 over random selection and is the best for 0.18 of the documents, so the ceiling marks no target that a selection rule has been shown to reach.

<table><tr><td rowspan="2">Architecture</td><td colspan="2">ρ(all layers)</td><td colspan="2">(late layers)  $\rho$ </td><td>NMI</td><td colspan="2">k-NN purity</td><td colspan="2">Late layers</td></tr><tr><td>1024</td><td>2048</td><td>1024</td><td>2048</td><td>Late</td><td>Late</td><td>All</td><td>2048</td><td>Kept</td></tr><tr><td>Llama-3.1-8B</td><td>0.261</td><td>0.242</td><td>0.338</td><td>0.318</td><td>0.587</td><td>0.746</td><td>0.708</td><td>8</td><td>4/4</td></tr><tr><td>Llama-3.2-3B</td><td>0.434</td><td>0.427</td><td>0.421</td><td>0.434</td><td>0.648</td><td>0.894</td><td>0.908</td><td>7</td><td>5/8</td></tr><tr><td>Qwen3.5-2B</td><td>0.516</td><td>0.507</td><td>0.518</td><td>0.540</td><td>0.781</td><td>0.896</td><td>0.884</td><td>12</td><td>8/8</td></tr><tr><td>Qwen3.5-9B</td><td>0.341</td><td>0.333</td><td>0.414</td><td>0.417</td><td>0.609</td><td>0.876</td><td>0.823</td><td>10</td><td>7/11</td></tr></table>

Table 5: Geometry before and after the truncation fix. Spearman correlations are given at both training windows; NMI and k-NN purity are reported for the corrected adapters only, NMI over the selected late layers and purity both over those layers and over all layers; purity is computed as the fraction of an adapter’s six nearest neighbours sharing its domain, for which chance is 0.161; the last two columns give the number of late layers selected after the fix and, of those selected before it, how many are selected again.

<table><tr><td rowspan="2">Regime</td><td rowspan="2">N</td><td rowspan="2">Mean</td><td colspan="2">Standard deviation</td><td colspan="2">Draws</td></tr><tr><td>Within</td><td>Between</td><td>Min</td><td>Max</td></tr><tr><td rowspan="5">Within</td><td>2</td><td>0.930</td><td>0.09</td><td>0.13</td><td>0.57</td><td>1.19</td></tr><tr><td>4</td><td>0.917</td><td>0.09</td><td>0.04</td><td>0.70</td><td>1.08</td></tr><tr><td>8</td><td>0.944</td><td>0.06</td><td>0.07</td><td>0.77</td><td>1.12</td></tr><tr><td>16</td><td>0.963</td><td>0.04</td><td>0.05</td><td>0.81</td><td>1.07</td></tr><tr><td>25</td><td>0.968</td><td>0.04</td><td>0.02</td><td>0.84</td><td>1.04</td></tr><tr><td rowspan="5">Cross</td><td>2</td><td>0.987</td><td>0.06</td><td>0.05</td><td>0.83</td><td>1.18</td></tr><tr><td>4</td><td>0.878</td><td>0.06</td><td>0.08</td><td>0.68</td><td>1.02</td></tr><tr><td>8</td><td>0.963</td><td>0.05</td><td>0.05</td><td>0.80</td><td>1.10</td></tr><tr><td>16</td><td>0.933</td><td>0.04</td><td>0.05</td><td>0.83</td><td>1.07</td></tr><tr><td>25</td><td>0.918</td><td>0.05</td><td>0.00</td><td>0.72</td><td>1.01</td></tr><tr><td></td><td>32</td><td>0.941</td><td>0.03</td><td>0.04</td><td>0.83</td><td>1.06</td></tr></table>

Table 6: Mean and spread of the random baseline’s OB retention. ‘Within’ is the mean standard deviation across the five draws made for a collection, ‘between’ is the standard deviation of the collection means, and the last two columns the most extreme single draws.

## E Composition

## E.1 Merge operators and scale

Each merge acts independently on each layer– module pair and returns an adapter of the original rank; update merging averages the products $\mathbf { B } _ { i } \mathbf { A } _ { i }$ whereas $\surd \mathrm { L E R P }$ and TIES act on the stored factors, and the two choices differ in the weight at which each adapter’s own update enters the result.

Averaging the updates. Let $\begin{array} { r l } { \Delta \mathbf { W } _ { \Delta } } & { { } = } \end{array}$ $N ^ { - 1 } \sum _ { i } { \bf B } _ { i } { \bf A } _ { i }$ be the update merge before truncation, and consider one document whose specialist it replaces. That document’s own update survives at $1 / N$ of its trained magnitude, whereas the other $N - 1$ updates, being close to orthogonal, sum to a norm about $\sqrt { N - 1 }$ times that of a single update and so contribute about $\sqrt { N - 1 } / N ;$ ; the wanted component therefore stands to the rest in a ratio of roughly

$$
\frac { 1 / N } { \sqrt { N - 1 } / N } = \frac { 1 } { \sqrt { N - 1 } }
$$

which falls from 1.00 at $N = 2$ to 0.20 at $N =$ 25. Orthogonality sets this rate, since updates that opposed one another would grow more slowly in sum, while aligned ones would approach $N - 1$ times a single update. On OB this dilution costs little, update merging keeping between 0.93 and 0.99 of the specialists’ score over the nine cells with $N \geq 4$ (section 5.1), although on cloze it keeps little of their advantage over the base model (section 5.4).

Returning to rank r. Stored exactly, the update merge has rank up to $N r$ , since

$$
\sum _ { i } \mathbf { B } _ { i } \mathbf { A } _ { i } = \left[ \mathbf { B } _ { 1 } \quad \mathbf { B } _ { 2 } \quad \cdots \quad \mathbf { B } _ { N } \right] { \left[ \begin{array} { l } { \mathbf { A } _ { 1 } } \\ { \mathbf { A } _ { 2 } } \\ { \vdots } \\ { \mathbf { A } _ { N } } \end{array} \right] }
$$

To compare it with the factor merges at equal size we take the truncated SVD of the averaged update, U<sub>r</sub> diag $( \pmb { \sigma } _ { r } ) \mathbf { V } _ { r } ^ { \top }$ , and refactor it as $\mathbf { B ^ { \prime } } =$ U diag $( \pmb { \sigma } _ { r } ) ^ { 1 / 2 }$ and $\mathbf A ^ { \prime } = \mathrm { d i a g } ( \pmb { \sigma } _ { r } ) ^ { 1 / 2 } \mathbf V _ { r } ^ { \top }$ , the best rank-r approximation in Frobenius norm. The fraction of squared singular values it retains falls from 0.72 at $N = 4$ to 0.38 at $N = 3 2$ , while OB retention falls by at most 0.034 against the full-rank average (section E.2).

Averaging the factors. Summing each factor over the collection and scaling each sum by $N ^ { - 1 / 2 }$ as $\surd \mathrm { L E R P }$ does, gives

$$
\begin{array} { r } { \left( \frac { 1 } { \sqrt { N } } \sum _ { i } \mathbf { B } _ { i } \right) \left( \frac { 1 } { \sqrt { N } } \sum _ { j } \mathbf { A } _ { j } \right) } \\ { \frac { 1 } { N } \sum _ { i } \mathbf { B } _ { i } \mathbf { A } _ { i } + \frac { 1 } { N } \sum _ { i \neq j } \mathbf { B } _ { i } \mathbf { A } _ { j } } \end{array}
$$

so each adapter’s own update enters at $1 / N ,$ as in update merging, beside $N ( N - 1 )$ cross terms that pair one adapter’s output directions with another’s input directions, correspond to no trained adapter, and are not constrained by the near-orthogonality of the updates, since that concerns the products $\mathbf { B } _ { i } \mathbf { A } _ { i }$ and says nothing of $\mathbf { B } _ { i } \mathbf { A } _ { j }$ . Averaging each factor at $1 / N$ instead, as our first runs did through the error of section $^ { 6 , }$ divides both sums by a further $N ,$ so each own update arrives at $1 / N ^ { 2 }$ of its trained magnitude, a sixty-fourth rather than an eighth at $N = 8 ;$ comparing $\surd \mathrm { I }$ ERP with update merging therefore separates the cost of the cross terms from that of scale (section 5.2).

TIES. TIES (Yadav et al., 2023) applies its three steps to A and to B separately: it trims each factor to its largest-magnitude entries, elects one sign per parameter by summed magnitude

$$
\gamma _ { p } = \mathrm { s g n } \left( \sum _ { i = 1 } ^ { N } \hat { w } _ { i , p } \right)
$$

where $\hat { w } _ { i , p }$ is the trimmed value of factor entry $p$ in adapter $i ,$ and averages only the entries that agree

<table><tr><td>Contrast</td><td>Factual</td><td>With true/false</td></tr><tr><td></td><td colspan="2">Top 1</td></tr><tr><td>Weight space — random (bank)</td><td>+0.020 [-0.038, +0.075]</td><td>+0.028 [-0.028, +0.086]</td></tr><tr><td>Weight space — text embedding</td><td>+0.021 [-0.036, +0.077]</td><td>+0.007[-0.041, +0.056]</td></tr><tr><td>Text embedding — random (bank)</td><td>-0.001[-0.060, +0.061]</td><td>+0.021[-0.025, +0.071]</td></tr><tr><td>Weight space — random (domain)</td><td>+0.040 [-0.020, +0.098]</td><td>+0.040 [-0.018, +0.099]</td></tr><tr><td>Centred weight space — random (bank)</td><td>+0.022[-0.039, +0.085]</td><td> $+ 0 . 0 2 2 \ [ - 0 . 0 3 2 , + 0 . 0 7 9 ]$ </td></tr><tr><td>Centred – uncentred weight space</td><td>+0.003 [-0.050, +0.054]</td><td>-0.006 [-0.066, +0.043]</td></tr><tr><td></td><td colspan="2">Top 3</td></tr><tr><td>Weight space — random (bank)</td><td>+0.040 [-0.017, +0.099]</td><td></td></tr><tr><td>Weight space — text embedding</td><td>+0.066 [+0.009, +0.123]</td><td> $+ 0 . 0 2 5 \left[ - 0 . 0 1 6 , + 0 . 0 7 0 \right]$ </td></tr><tr><td>Text embedding – random (bank)</td><td></td><td> $+ 0 . 0 4 9 \left[ + 0 . 0 0 6 , + 0 . 0 9 3 \right]$ </td></tr><tr><td>Weight space — random (domain)</td><td>-0.026 [-0.070, +0.014]</td><td> $- 0 . 0 2 4 \left[ - 0 . 0 6 4 , + 0 . 0 1 3 \right]$ </td></tr><tr><td></td><td>+0.033[-0.013, +0.080]</td><td> $+ 0 . 0 2 6 [ - 0 . 0 0 6 , + 0 . 0 6 1 ]$ </td></tr><tr><td>Centred weight space — random (bank)</td><td>+0.020[-0.028, +0.071]</td><td> $+ 0 . 0 1 0 \left[ - 0 . 0 3 0 , + 0 . 0 5 2 \right]$ </td></tr><tr><td>Centred – uncentred weight space</td><td>-0.020 [-0.072, +0.029]</td><td>-0.015 [-0.054, +0.022]</td></tr></table>

Table 7: The 24 declared cloze contrasts of the retrieval run, paired over the 30 query documents, with 95 % CIs from 10 000 resamples; each query’s adapter is selected from a bank of 120 that never holds the query’s own. ‘Top 1’ applies the selected adapter and ‘top 3’ merges the three highest-ranked with ${ \sqrt { \operatorname { L E R P } } } ;$ ‘factual’ scores the three factual items and ‘with true/false’ adds the fourth; the random arms average five draws. The primary contrast is weight space against a random adapter from the bank at top 1 on the factual items, and its decision criterion was fixed before it was computed; the centred contrasts were added afterwards, outside that criterion.

with the elected sign

$$
\mathbf { \Delta } \mathbf { W } _ { \mathrm { T I E S } , p } = \frac { 1 } { | S _ { p } | } \sum _ { i \in S _ { p } } \hat { w } _ { i , p }
$$

where $S _ { p } = \{ i : \mathrm { s g n } ( \hat { w } _ { i , p } ) = \gamma _ { p } \}$ . Because sign election does not commute with the product, electing over A and B is not electing over BA; a TIES merge in update space would require materializing each product and refactoring the result, which we do not attempt.

## E.2 Merges at matched rank

Against random (section 5.1), update merging’s contrast at cross-domain N = 16 is the only point estimate below zero, by less than 0.001, and its CI excludes zero in six cells of nine. Against the same average kept at full rank Nr, which is equivalent to loading the whole collection, the rank-8 projection of update merging loses between 0.002 and 0.034 of retention, with six CIs of nine excluding zero, even though at N = 32 its eight directions carry only 0.38 of the summed update’s squared singular values (0.72 at N = 4); whatever the model draws on from these adapters lies in few directions. A merging coefficient tuned on half of each collection’s documents and scored on the other half does worse than 1/N in four cells, the same in three and better in two, by between −0.061 and +0.019. At matched rank TIES lies above update merging in eight cells of nine (sign test $p = 0 . 0 3 9 )$ , and three CIs exclude zero, two favouring TIES, by 0.020 and 0.009, and one update merging, by 0.017; the largest difference, 0.041 at within-domain N = 4, has a CI that includes zero. Table 8 gives every method’s retention in every cell, the scaling error and joint training included, and tables 9 and 10 the paired contrasts.

Prior measurements of the cross terms. Tang et al. (2025a) measure the gap between merging the factors and merging the updates of rank-16 adapters on eight vision tasks and attribute it to the cross terms; under averaging, though, their factor merge also enters each adapter’s own update at $1 / N ^ { 2 }$ rather than $1 / N ,$ and their control, which adds noise matched to the cross terms’ mean and standard deviation to update merging, keeps update merging’s own-term scale, so the gap they report joins scale to cross terms. Zheng et al. (2025) write their separate merge with one coefficient on the own terms and the cross terms alike, which as writ-<sub>ten</sub> <sub>isolates</sub> <sub>the</sub> <sub>cross</sub> <sub>terms</sub> <sub>as</sub> √<sub>LERP</sub> <sub>does,</sub> <sub>and</sub> find that they cost about ten points across eight vision tasks, against at most 0.035 of OB retention across our documents; KnOTS (Stoica et al., 2025) argues that cross terms can cost performance and merges only full updates, but its merged update is not limited in rank and so offers no matchedbudget comparison. Our updates share input directions learned in training (section 3), so a cross term $\mathbf { B } _ { i } \mathbf { A } _ { j }$ may partly re-add what every update carries, which would make the cost of cross terms depend on how alike the merged adapters are; we offer this as a hypothesis we have not tested.

<table><tr><td>Regime</td><td> $N$ </td><td>Scaling error</td><td> $\scriptstyle { \sqrt { \mathrm { L E R P } } }$ </td><td>Rank-8 update</td><td>TIES</td><td>Random</td><td>Joint</td><td>Full-rank update</td><td>Spectral share</td></tr><tr><td rowspan="5">Cross</td><td>4</td><td>0.820</td><td>0.996</td><td>0.961</td><td>0.980</td><td>0.878</td><td>1.032</td><td>0.995</td><td>0.719</td></tr><tr><td>8</td><td>0.713</td><td>1.011</td><td>0.994</td><td>1.004</td><td>0.963</td><td>1.027</td><td>1.013</td><td>0.577</td></tr><tr><td>16</td><td>0.638</td><td>0.948</td><td>0.932</td><td>0.942</td><td>0.933</td><td>0.994</td><td>0.960</td><td>0.441</td></tr><tr><td>25</td><td>0.639</td><td>0.978</td><td>0.961</td><td>0.973</td><td>0.918</td><td>1.009</td><td>0.979</td><td>0.390</td></tr><tr><td>32</td><td>0.622</td><td>0.954</td><td>0.960</td><td>0.968</td><td>0.941</td><td>1.023</td><td>0.974</td><td>0.379</td></tr><tr><td rowspan="4">Within</td><td>4</td><td>0.744</td><td>0.982</td><td>0.948</td><td>0.990</td><td>0.917</td><td>1.020</td><td>0.977</td><td>0.716</td></tr><tr><td>8</td><td>0.677</td><td>0.986</td><td>0.976</td><td>0.985</td><td>0.944</td><td>0.988</td><td>0.979</td><td>0.591</td></tr><tr><td>16</td><td>0.654</td><td>0.981</td><td>0.989</td><td>0.972</td><td>0.963</td><td>1.015</td><td>0.991</td><td>0.451</td></tr><tr><td>25</td><td>0.648</td><td>0.996</td><td>0.983</td><td>0.987</td><td>0.968</td><td>1.016</td><td>0.997</td><td>0.404</td></tr></table>

Table 8: OB retention at matched rank in the nine cells with $N \geq 4 .$ Each document’s $F _ { 1 }$ under the method is divided by that of its own specialist, every column by the same specialists, averaged over documents and then over the cell’s sets (five per cell, six at within-domain $N = 2 5 )$ . The scaling error is the factor average with both factors scaled by $1 / N ;$ ; the random experiment averages five out-of-collection adapters per set, which at within-domain $N = 2 5$ necessarily come from other domains; joint training is a retraining reference; full-rank update keeps the averaged update at rank 8N, a ceiling rather than a merge; the spectral share is the fraction of the averaged update’s squared singular values in its top 8 directions. Joint training at cross-domain $N = 2 5$ averages four of the five sets, the fifth having no OB evaluation of its joint adapter.

<table><tr><td>Regime</td><td>N</td><td> $\sqrt { \mathrm { L E R P - s c a l i n g \ e r r o r } }$ </td><td> $\mathrm { R a n k - 8 \ u p d a t e - \sqrt { L E R P } }$ </td><td>Rank-8 — full-rank update</td></tr><tr><td rowspan="5">Cross</td><td>4</td><td> $+ 0 . 1 7 6 \ [ + 0 . 1 2 6 , + 0 . 2 4 0 ]$ </td><td> $- 0 . 0 3 5 \left[ - 0 . 0 6 9 , - 0 . 0 0 5 \right]$ </td><td> $- 0 . 0 3 4 \ : [ - 0 . 0 7 4 , - 0 . 0 0 4 ]$ </td></tr><tr><td>8</td><td> $+ 0 . 2 9 7 \left[ + 0 . 2 6 5 , + 0 . 3 2 6 \right]$ </td><td> $- 0 . 0 1 6 \left[ - 0 . 0 2 4 , - 0 . 0 0 7 \right]$ </td><td> $- 0 . 0 1 9 \left[ - 0 . 0 3 0 , - 0 . 0 1 0 \right]$ </td></tr><tr><td>16</td><td> $+ 0 . 3 1 0 [ + 0 . 2 8 3 , + 0 . 3 3 6 ]$ </td><td> $- 0 . 0 1 5 \left[ - 0 . 0 4 6 , + 0 . 0 0 8 \right]$ </td><td> $- 0 . 0 2 7 \left[ - 0 . 0 4 2 , - 0 . 0 1 3 \right]$ </td></tr><tr><td>25</td><td> $+ 0 . 3 3 8 [ + 0 . 3 2 2 , + 0 . 3 5 8 ]$ </td><td> $- 0 . 0 1 6 \left[ - 0 . 0 3 6 , - 0 . 0 0 1 \right]$ </td><td> $- 0 . 0 1 8 \left[ - 0 . 0 3 2 , - 0 . 0 0 5 \right]$ </td></tr><tr><td>32</td><td> $+ 0 . 3 3 3 [ + 0 . 3 0 1 , + 0 . 3 6 4 ]$ </td><td> $+ 0 . 0 0 5 \left[ - 0 . 0 1 3 , + 0 . 0 2 3 \right]$ </td><td> $- 0 . 0 1 4 \left[ - 0 . 0 2 0 , - 0 . 0 1 0 \right]$ </td></tr><tr><td rowspan="4">Within</td><td>4</td><td> $+ 0 . 2 3 8 \left[ + 0 . 1 7 4 , + 0 . 3 0 7 \right]$ </td><td> $- 0 . 0 3 3 \left[ - 0 . 0 7 8 , + 0 . 0 1 1 \right]$ </td><td> $- 0 . 0 2 9 \left[ - 0 . 0 7 9 , + 0 . 0 1 1 \right]$ </td></tr><tr><td>8</td><td> $+ 0 . 3 0 9 [ + 0 . 2 5 2 , + 0 . 3 6 0 ]$ </td><td> $- 0 . 0 0 9 \left[ - 0 . 0 2 3 , + 0 . 0 0 4 \right]$ </td><td> $- 0 . 0 0 2 \left[ - 0 . 0 2 2 , + 0 . 0 1 7 \right]$ </td></tr><tr><td>16</td><td> $+ 0 . 3 2 7 \left[ + 0 . 2 8 9 , + 0 . 3 7 0 \right]$ </td><td>+0.008[-0.010, +0.032]</td><td>-0.002[-0.008, +0.006]</td></tr><tr><td>25</td><td> $+ 0 . 3 4 8 \ [ + 0 . 3 2 2 , + 0 . 3 7 5 ]$ </td><td>-0.013[-0.018, -0.009]</td><td>-0.015[-0.021, -0.007]</td></tr><tr><td>Regime</td><td> N</td><td> $\sqrt { \mathrm { L E R P - r a n d o m } }$ </td><td>Rank-8 update — random</td><td>TIES — rank-8 update</td></tr><tr><td rowspan="5">Cross</td><td>4</td><td> $+ 0 . 1 1 7 \left[ + 0 . 0 6 1 , + 0 . 2 0 5 \right]$ </td><td> $+ 0 . 0 8 2 \ : [ + 0 . 0 2 9 , + 0 . 1 3 9 ]$ </td><td> $+ 0 . 0 2 0 \left[ + 0 . 0 0 3 , + 0 . 0 3 7 \right]$ </td></tr><tr><td>8</td><td> $+ 0 . 0 4 7 \left[ + 0 . 0 1 8 , + 0 . 0 7 2 \right]$ </td><td> $+ 0 . 0 3 1 \left[ + 0 . 0 0 9 , + 0 . 0 5 3 \right]$ </td><td> $+ 0 . 0 0 9 \left[ + 0 . 0 0 1 , + 0 . 0 1 6 \right]$ </td></tr><tr><td>16</td><td> $+ 0 . 0 1 5 \left[ - 0 . 0 0 2 , + 0 . 0 4 5 \right]$ </td><td> $- 0 . 0 0 0 \left[ - 0 . 0 0 8 , + 0 . 0 1 1 \right]$ </td><td> $+ 0 . 0 1 0 \left[ - 0 . 0 0 7 , + 0 . 0 2 1 \right]$ </td></tr><tr><td>25</td><td> $+ 0 . 0 6 0 \left[ + 0 . 0 3 3 , + 0 . 0 8 3 \right]$ </td><td> $+ 0 . 0 4 3 \ : [ + 0 . 0 1 7 , + 0 . 0 6 7 ]$ </td><td> $+ 0 . 0 1 1 \left[ - 0 . 0 0 2 , + 0 . 0 2 7 \right]$ </td></tr><tr><td>32</td><td> $+ 0 . 0 1 4 \left[ - 0 . 0 0 0 , + 0 . 0 2 9 \right]$ </td><td> $+ 0 . 0 1 9 \left[ + 0 . 0 1 4 , + 0 . 0 2 5 \right]$ </td><td> $+ 0 . 0 0 8 \left[ - 0 . 0 0 6 , + 0 . 0 3 0 \right]$ </td></tr><tr><td rowspan="4">Within</td><td>4</td><td> $+ 0 . 0 6 5 \ [ + 0 . 0 1 5 , + 0 . 1 3 6 ]$ </td><td> $+ 0 . 0 3 1 \ [ - 0 . 0 1 5 , + 0 . 0 7 7 ]$ </td><td> $+ 0 . 0 4 1 \left[ - 0 . 0 0 6 , + 0 . 1 2 1 \right]$ </td></tr><tr><td>8</td><td> $+ 0 . 0 4 1 \ [ + 0 . 0 2 4 , + 0 . 0 5 8 ]$ </td><td> $+ 0 . 0 3 2 \ : [ + 0 . 0 2 7 , + 0 . 0 3 6 ]$ </td><td> $+ 0 . 0 0 9 \left[ - 0 . 0 0 2 , + 0 . 0 2 0 \right]$ </td></tr><tr><td>16</td><td> $+ 0 . 0 1 8 \left[ - 0 . 0 0 7 , + 0 . 0 5 2 \right]$ </td><td> $+ 0 . 0 2 6 \left[ - 0 . 0 0 2 , + 0 . 0 5 8 \right]$ </td><td> $- 0 . 0 1 7 \left[ - 0 . 0 3 2 , - 0 . 0 0 3 \right]$ </td></tr><tr><td>25</td><td> $+ 0 . 0 2 8 \left[ + 0 . 0 2 1 , + 0 . 0 3 7 \right]$ </td><td>+0.015[+0.008, +0.021]</td><td> $+ 0 . 0 0 4 \left[ - 0 . 0 0 7 , + 0 . 0 1 6 \right]$ </td></tr></table>

Table 9: Paired OB contrasts that isolate scale, cross terms, and truncation. All values are shown with 95 % CIs, computed from 10 000 resamples over sets. LERP minus the scaling error isolates scale, rank-8 update minus LERP isolates cross terms at equal own-term weight, and rank-8 update minus full-rank update isolates the cost of truncation. Each contrast is computed from unrounded retentions, so it can differ in its last digit from the difference of the displayed values.

Table 10: Paired OB contrasts of the correctly scaled merges with the random adapter and of TIES with the rank-8 update. Values are shown with 95 % CIs from 10 000 resamples over sets. Each contrast is computed from unrounded retentions, so it can differ in its last digit from the difference of the displayed values.

## E.3 Joint training and the full grid

Per cell, joint training’s lead over the merges (section 5.3) is less uniform than pooled: OB, its CI includes zero against $\surd \mathrm { L E R P }$ and TIES at crossdomain $N = 2 5$ <sub>,</sub> <sub>against</sub> √<sub>LERP</sub> <sub>at</sub> <sub>within-domain</sub> $N \ = \ 2 5$ and against update merging at withindomain N = 16 (table 11), as it does in the within-domain cells at N = 4 and N = 8, and at within-domain $N \ = \ 2$ joint training sits below every merge on both probes. At cross-domain $N = 3 2$ , where 49 of the 150 documents appear in more than one collection, joint training is not distinguishable from parity with the specialists, while its lead over random excludes zero under every resampling scheme we tried, over collections and over documents.

Training budget. Each joint adapter trains for the same 500 steps as each specialist it replaces, so at N documents it receives a factor of N less gradient per document. Repeating N = 16 with the budget scaled linearly to 8000 steps, so that each document receives the gradient it received as a specialist, retention at the matched budget minus that at the scaled one is $+ 0 . 0 1 7 \ : ( \mathrm { C I } _ { 9 5 \% } =$ $[ - 0 . 0 1 0 , + 0 . 0 4 9 ] )$ within domains (1.015 against 0.998) and +0.006 $( \mathrm { C I } _ { 9 5 \% } = [ - 0 . 0 1 1 , + 0 . 0 1 8 ] )$ across them (0.994 against 0.988); neither CI excludes zero, which places joint training’s flat retention on capacity rather than on undertraining.

The economics deficit. Economics is the one domain whose N = 25 joint adapter falls below parity, at 0.940, and its deficit is concentrated rather than diffuse: two documents, price elasticity of demand (0.562) and fiscal policy (0.603), lose most, and without them the domain’s mean rises to 0.971, while their specialist scores repeat exactly under the corrected configuration, so they are not artefacts of the denominator; we report the deficit without a mechanism.

Retention by collection size. Figure 3 plots retention against collection size for every method; √<sub>LERP, update merging and the scaling error are</sub> drawn from N = 4, the range of the matchedrank comparison, since they were not run OB at N = 2, and the within-domain regime stops at N = 25 because each domain holds exactly that many documents.

## E.4 Cloze composition grid

At $N \ge 1 6$ the merges’ cloze contrasts with random run from −0.037 to +0.063, with LERP excluding zero in three cells, two above the random adapter and one (cross-domain N = 32) below it, update merging in one and TIES in none, while joint training leads random by 0.164 to 0.237, four CIs of five excluding zero. At within-domain $N = 2$ joint training sits below the random adapter on cloze (−0.242 with $\mathrm { C I _ { 9 5 \% } } = [ - 0 . 4 5 0 , - 0 . 0 3 3 ] )$ , as it sits below every merge there on both probes (section E.3); we report this without an explanation. Table 12 gives both aggregations of retention for every cell and method, and table 13 each method’s advantage over random; $\surd \mathrm { L E R P }$ and update merging were run at $N = 2$ on cloze only. The seven documents whose specialist recalls none of the factual items, and which the mean of ratios therefore leaves out, are adenosine triphosphate, algorithm, gradient descent, sorting algorithm, microeconomics, the Industrial Revolution and judicial review. The random, TIES and joint numerators come from an earlier run whose result files do not record the base model’s precision; its code loads the base in 4-bit, and 64 of the 66 evaluations it shares with the retrieval run agree to six decimals, so we treat it as on the same footing.

<table><tr><td></td><td>N</td><td> $\mathrm { J o i n t - \sqrt { L E R P } }$ </td><td>Joint — rank-8 update</td><td>Joint – TIES</td></tr><tr><td rowspan="3">Cross</td><td>16</td><td> $+ 0 . 0 4 6 [ + 0 . 0 2 1 , + 0 . 0 7 1 ]$ </td><td> $+ 0 . 0 6 2 [ + 0 . 0 2 3 , + 0 . 0 8 7 ]$ </td><td> $+ 0 . 0 5 2 \left[ + 0 . 0 0 7 , + 0 . 0 8 4 \right]$ </td></tr><tr><td>25</td><td> $+ 0 . 0 2 0 \left[ - 0 . 0 0 6 , + 0 . 0 4 6 \right]$ </td><td> $+ 0 . 0 4 3 \ [ + 0 . 0 1 1 , + 0 . 0 7 4 ]$ </td><td> $+ 0 . 0 2 9 \left[ - 0 . 0 0 7 , + 0 . 0 6 5 \right]$ </td></tr><tr><td>32</td><td> $+ 0 . 0 6 8 \left[ + 0 . 0 4 9 , + 0 . 0 8 8 \right]$ </td><td> $+ 0 . 0 6 3 \ : [ + 0 . 0 5 1 , + 0 . 0 7 2 ]$ </td><td> $+ 0 . 0 5 5 \left[ + 0 . 0 3 2 , + 0 . 0 7 5 \right]$ </td></tr><tr><td rowspan="2">Within</td><td>16</td><td> $+ 0 . 0 3 4 \left[ + 0 . 0 0 0 , + 0 . 0 8 4 \right]$ </td><td> $+ 0 . 0 2 6 \left[ - 0 . 0 0 1 , + 0 . 0 5 6 \right]$ </td><td> $+ 0 . 0 4 3 \left[ + 0 . 0 1 0 , + 0 . 0 8 0 \right]$ </td></tr><tr><td>25</td><td> $+ 0 . 0 2 0 [ - 0 . 0 0 5 , + 0 . 0 4 1 ]$ </td><td> $+ 0 . 0 3 3 \left[ + 0 . 0 0 9 , + 0 . 0 5 1 \right]$ </td><td> $+ 0 . 0 2 9 \left[ + 0 . 0 1 1 , + 0 . 0 4 3 \right]$ </td></tr><tr><td>Pooled</td><td>一</td><td> $\begin{array} { r l } { - } & { { } + 0 . 0 3 8 \left[ + 0 . 0 2 3 , + 0 . 0 5 3 \right] } \end{array}$ </td><td> $+ 0 . 0 4 5 \ [ + 0 . 0 3 2 , + 0 . 0 5 8 ]$ </td><td> $+ 0 . 0 4 2 \ : [ + 0 . 0 2 7 , + 0 . 0 5 6 ]$ </td></tr></table>

Table 11: Paired contrasts in retention of joint training against each rank-8 merge at $N \ge 1 6 , 0 8$ . Values are computed at set level, with 95 % CIs from 10 000 resamples. Joint training at cross-domain $N = 2 5$ averages four of the five sets, the fifth having no OB evaluation of its joint adapter. Each contrast is computed from unrounded retentions, so it can differ in its last digit from the difference of the displayed values.

![](images/5079a5dbe5607b6dfee395749fb4030c28c15403fbc876ca0319965e70541c56.jpg)  
Figure 3: OB retention against collection size. CIs are omitted for legibility, so the paired contrasts of sections E.2 and E.3, rather than the figure, carry the claims about significance; the 1.0 marks parity with the per-document specialist each method replaces. Scaling error is the factor average at $1 / N ^ { 2 }$ ; joint training at cross-domain $N = 2 5$ averages four of the five sets, the fifth having no OB evaluation of its joint adapter.

## E.5 Pairwise interference

Merging pairs of adapters with TIES, over 200 pairs stratified into four equal-quantile bins of semantic distance between the two training documents, changes accuracy by $- 0 . 0 1 1 , \ - 0 . 0 0 7 .$ +0.001 and −0.016 from the closest bin to the most distant, whose mean distances are 0.736, 0.861, 0.917 and 0.977. Neither distance anticipates the change: its correlation with semantic distance and with weight-space distance stays within $| r | \leq 0 . 0 1 6$ , and between weight-space distance and the change Spearman $\rho ~ = ~ 0 . 0 0 0$ with p = 0.999 under a permutation of adapter labels, which respects the dependence between pairs that share an adapter. Within-domain pairs lose more than cross-domain ones (−0.026 against −0.005), the opposite of what a distance that anticipated interference would give. We report no factor-averaging merge here, since the pairwise LERP merges of these runs carried the scaling error described in section 6.

<table><tr><td rowspan="2">Regime</td><td rowspan="2">N</td><td colspan="2">Random</td><td colspan="2"> $\scriptstyle { \sqrt { \mathrm { L E R P } } }$ </td><td colspan="2">Rank-8 update</td><td colspan="2">TIES</td><td colspan="2">Joint</td></tr><tr><td>MR</td><td>AR</td><td>MR</td><td>AR</td><td>MR</td><td>AR</td><td>MR</td><td>AR</td><td>MR</td><td>AR</td></tr><tr><td rowspan="6">Cross</td><td>2</td><td>0.916</td><td>0.876</td><td>1.014</td><td>1.005</td><td>1.086</td><td>1.056</td><td>1.089</td><td>1.065</td><td>1.103</td><td>1.066</td></tr><tr><td>4</td><td>1.017</td><td>0.901</td><td>1.154</td><td>1.044</td><td>1.104</td><td>0.987</td><td>1.142</td><td>1.044</td><td>1.197</td><td>1.139</td></tr><tr><td>8</td><td>0.966</td><td>0.873</td><td>0.975</td><td>0.919</td><td>0.930</td><td>0.847</td><td>0.916</td><td>0.853</td><td>1.195</td><td>1.085</td></tr><tr><td>16</td><td>0.893</td><td>0.811</td><td>0.943</td><td>0.849</td><td>0.934</td><td>0.853</td><td>0.924</td><td>0.842</td><td>1.057</td><td>0.966</td></tr><tr><td>25</td><td>0.896</td><td>0.818</td><td>0.909</td><td>0.826</td><td>0.904</td><td>0.826</td><td>0.905</td><td>0.827</td><td>1.103</td><td>0.993</td></tr><tr><td>32</td><td>0.923</td><td>0.817</td><td>0.886</td><td>0.796</td><td>0.929</td><td>0.831</td><td>0.920</td><td>0.819</td><td>1.112</td><td>1.000</td></tr><tr><td rowspan="5">Within</td><td>2</td><td>1.057</td><td>1.012</td><td>1.148</td><td>1.118</td><td>1.214</td><td>1.133</td><td>1.164</td><td>1.089</td><td>0.815</td><td>0.818</td></tr><tr><td>4</td><td>0.886</td><td>0.873</td><td>0.975</td><td>0.956</td><td>0.880</td><td>0.877</td><td>0.977</td><td>0.958</td><td>1.040</td><td>0.994</td></tr><tr><td>8</td><td>0.977</td><td>0.868</td><td>0.997</td><td>0.943</td><td>1.000</td><td>0.925</td><td>0.949</td><td>0.903</td><td>1.115</td><td>1.017</td></tr><tr><td>16</td><td>0.860</td><td>0.782</td><td>0.901</td><td>0.826</td><td>0.866</td><td>0.801</td><td>0.922</td><td>0.847</td><td>1.052</td><td>0.955</td></tr><tr><td>25</td><td>0.895</td><td>0.827</td><td>0.953</td><td>0.872</td><td>0.915</td><td>0.849</td><td>0.930</td><td>0.864</td><td>1.131</td><td>1.015</td></tr><tr><td>Pooled</td><td></td><td>0.934</td><td>0.827</td><td>0.986</td><td>0.857</td><td>0.977</td><td>0.851</td><td>0.984</td><td>0.858</td><td>1.085</td><td>1.000</td></tr></table>

Table 12: Cloze retention relative to each document’s own specialist, on the 4-bit footing. $\mathbf { \hat { M } R } ^ { \prime }$ is the mean over documents of the ratio of recalls, without the seven documents whose specialist recalls none of the factual items; ‘AR’ is the ratio of mean recalls, with them; the pooled AR covers 735 rows. The random, TIES, and joint values come from an earlier run, whose footing is described in section E.4.

## F Silent errors

## F.1 Truncation by domain

Under the training-length default of section 6, the share of documents in which none of the eight training examples kept any token of its answer, counted with the training tokenizer, runs from 4 % in law to 80 % in biology, 44 % overall (‘all answers’ row of table 14). Table 15 gives each domain’s gap between no-context and natural adapters at both training windows: the sign changes in four of the six domains, since law is negative at both windows and medicine sits at +0.001 $( \mathrm { C I _ { 9 5 \% } } = [ - 0 . 0 6 4 , + 0 . 0 6 3 ] )$ at 2048, and of the five domains negative at 2048 only economics and law exclude zero. The correlation that section 6 reports holds under each of the five definitions of table 14, with Spearman $\rho$ between −0.943 and −0.986 and Pearson r between −0.960 and −0.982. Evaluation had a window of its own, which in some of our earlier runs was 512 tokens and cut every OB document to about 330 of its 871 to 1257 tokens; at the 2048 tokens used throughout, no prompt is cut.

## F.2 The merge scaling error and prior work on scale

At $N = 3 2$ the error of section 6 entered each adapter’s own update at one part in 1024 of its trained magnitude rather than one in 32. The same arithmetic sat in two further helpers, which formed the pairwise LERP merges of our interference runs and the top-3 merges of an earlier OB retrieval run, and we report neither. In PEFT 0.18.1 (Mangrulkar et al., 2022), the linear combination multiplies both factors of each adapter by the square root of its weight times its scaling $\alpha / r$ and gives the merged adapter a scaling of one, so with weights $1 / N$ and a shared $\alpha / r$ it forms exactly the update of $\scriptstyle { \sqrt { \mathrm { L E R P } } } ,$ while concatenation keeps each adapter’s own update at its weight and adds no cross terms; neither reproduces our scaling. Back et al. (2026) find that concatenating three unscaled document adapters collapses under an N-fold perturbation and that rescaling each factor by $1 / \sqrt { N }$ recovers most of the gap to linear averaging, LoRA-LEGO (Zhao et al., 2025) rescales merged adapters for the decay in norm that averaging causes, and LoRA Soups (Prabhakar et al., 2025) writes linear merging with its coefficient on both factors.

<table><tr><td>Regime</td><td>N</td><td>√LERP - random</td><td>Rank-8 update — random</td><td></td><td>TIES — random</td><td>Joint — random</td></tr><tr><td rowspan="6">Cross</td><td>2</td><td>+0.099 [-0.022, +0.219]</td><td>+0.170 [+0.102, +0.246]</td><td></td><td>+0.174[+0.076, +0.273]</td><td>+0.187[-0.020, +0.409]</td></tr><tr><td>4</td><td>+0.137[+0.000, +0.302]</td><td>+0.087[-0.050, +0.277]</td><td></td><td>+0.125 [-0.032, +0.324]</td><td>+0.180 [-0.064, +0.412]</td></tr><tr><td>8</td><td>+0.009[-0.107, +0.107]</td><td>-0.035 [-0.134, +0.053]</td><td></td><td>-0.050 [-0.108, +0.004]</td><td>+0.229 [+0.060, +0.372]</td></tr><tr><td>16</td><td>+0.050 [+0.018, +0.092]</td><td>+0.041 [+0.021, +0.056]</td><td></td><td>+0.031[-0.010, +0.068]</td><td>+0.164[-0.008, +0.308]</td></tr><tr><td>25</td><td>+0.013[-0.031, +0.059]</td><td>+0.008 [-0.033, +0.061]</td><td></td><td>+0.009[-0.011, +0.027]</td><td>+0.207[+0.128, +0.307]</td></tr><tr><td>32</td><td>-0.037 [-0.075, -0.005]</td><td>+0.006 [-0.056, +0.079]</td><td></td><td>-0.003 [-0.020, +0.013]</td><td>+0.189 [+0.107, +0.287]</td></tr><tr><td rowspan="5">Within</td><td>2</td><td>+0.091 [-0.122, +0.271]</td><td>+0.158 [-0.067, +0.382]</td><td></td><td>+0.108 [-0.024, +0.240]</td><td>-0.242 [-0.450, -0.033]</td></tr><tr><td>4</td><td>+0.089 [-0.058, +0.276]</td><td>-0.006 [-0.117, +0.104]</td><td></td><td>+0.091 [-0.049, +0.280]</td><td>+0.154[-0.106, +0.448]</td></tr><tr><td>8</td><td>+0.020 [-0.132, +0.133]</td><td>+0.022[-0.118, +0.119]</td><td></td><td>-0.029 [-0.175, +0.092]</td><td>+0.138[-0.018, +0.270]</td></tr><tr><td>16</td><td>+0.042[-0.046, +0.123]</td><td>+0.006 [-0.083, +0.085]</td><td></td><td>+0.063 [-0.054, +0.171]</td><td>+0.192[+0.107, +0.307]</td></tr><tr><td>25</td><td>+0.059 [+0.023, +0.092]</td><td>+0.021[-0.038, +0.069]</td><td></td><td>+0.035 [-0.021, +0.089]</td><td>+0.237[+0.123, +0.360]</td></tr><tr><td>Pooled, 56</td><td></td><td>+0.052 [+0.016, +0.089]</td><td>+0.043 [+0.006, +0.081]</td><td></td><td>+0.050 [+0.015, +0.088]</td><td>+0.150 [+0.087, +0.212]</td></tr></table>

<sub>ntage</sub> <sub>over</sub> <sub>the</sub> <sub>random</sub> <sub>adapter</sub> <sub>in</sub> <sub>cloze</sub> <sub>retention.</sub> <sub>All</sub> <sub>valu</sub><sup>es</sup> <sup>are</sup> <sup>mean</sup> <sup>of</sup> <sup>ratios,</sup> <sup>paired</sup> <sup>over</sup> <sup>collections,</sup> <sup>with95</sup> <sup>%CIs</sup> <sub>from</sub> <sub>unrounded</sub> <sub>retentions,</sub> <sub>so</sub> <sub>it</sub> <sup>can</sup> <sup>differ</sup> <sup>in</sup> <sup>its</sup> <sup>last</sup> <sup>digit</sup> <sup>from</sup> <sup>the</sup> <sup>difference</sup> <sup>of</sup> <sup>t</sup>

<table><tr><td rowspan="2">Definition</td><td colspan="7">Truncated documents (%)</td><td colspan="2">Spearman</td><td colspan="2">Pearson</td></tr><tr><td>L</td><td>E</td><td>CS</td><td>H</td><td>M</td><td>B</td><td>All</td><td>ρ</td><td>p</td><td>r</td><td>p</td></tr><tr><td>First answer</td><td>8</td><td>24</td><td>44</td><td>56</td><td>64</td><td>80</td><td>46.0</td><td>-0.943</td><td>0.017*</td><td>-0.976</td><td>0.003**</td></tr><tr><td>Any string</td><td>40</td><td>48</td><td>68</td><td>84</td><td>88</td><td>96</td><td>70.7</td><td>-0.943</td><td>0.017*</td><td>-0.960</td><td>0.003**</td></tr><tr><td>Any answer</td><td>12</td><td>28</td><td>48</td><td>72</td><td>72</td><td>84</td><td>52.7</td><td>-0.986</td><td>0.006**</td><td>-0.963</td><td>0.003**</td></tr><tr><td>All answers</td><td>4</td><td>24</td><td>40</td><td>56</td><td>60</td><td>80</td><td>44.0</td><td>-0.943</td><td> $0 . 0 1 7 ^ { * }$ </td><td>-0.982</td><td>0.003**</td></tr><tr><td>Mean share</td><td>6</td><td>24</td><td>44</td><td>58</td><td>64</td><td>82</td><td>46.6</td><td>-0.943</td><td> $0 . 0 1 7 ^ { * }$ </td><td>-0.980</td><td>0.003**</td></tr></table>

Table 14: Truncation at the 1024-token default under five definitions, counted with the training tokenizer, and each definition’s correlation across the six domains with the change in the gap (table 15). A document counts as truncated when: (a) the answer of its first training example lies wholly beyond the window; (b) any of its training strings exceeds the window; (c) at least one of its answers lies wholly beyond it; (d) all eight do, the definition of section $6 ;$ or (e) is the mean share of the eight answers that do. ‘L’ stands for law, ‘E’ for economics, ‘CS’ for computer science, ‘H’ for history, ‘M’ for medicine, and $ { \mathbf { \ell } } ^ { 6 }  { \mathbf { B } } ^ { \prime }$ for biology. The p-values are computed over the 720 orderings; $^ { * } p < 0 . 0 5 , ^ { * * } p < 0 . 0 1 , ^ { * * * } p < 0 . 0 0 1$

<table><tr><td rowspan="2">Domain</td><td colspan="2">Gap</td><td rowspan="2">Change</td></tr><tr><td>at 1024</td><td>at 2048</td></tr><tr><td>Law</td><td>-0.017[-0.105, +0.073]</td><td>-0.078[-0.137, -0.017]</td><td>-0.061[-0.122, -0.005]</td></tr><tr><td>Economics</td><td>+0.042 [−0.054, +0.144]</td><td>-0.071 [-0.130, -0.013]</td><td>-0.112[-0.201, -0.031]</td></tr><tr><td>C. science</td><td>+0.148 [+0.010, +0.275]</td><td>-0.062 [-0.134, +0.002]</td><td>-0.210[-0.302, −0.121]</td></tr><tr><td>History</td><td>+0.241 [+0.137, +0.345]</td><td>-0.035 [-0.072, +0.004]</td><td>-0.276[-0.384, -0.164]</td></tr><tr><td>Medicine</td><td>+0.251 [+0.139, +0.357]</td><td>+0.001 [-0.064, +0.063]</td><td>-0.249 [-0.336, -0.163]</td></tr><tr><td>Biology</td><td>+0.344 [+0.254, +0.435]</td><td>-0.049 [-0.113, +0.011]</td><td>-0.394[-0.484, -0.300]</td></tr><tr><td>All</td><td>+0.168 [+0.120, +0.215]</td><td>-0.049 [-0.074, -0.025]</td><td>-0.217[-0.257, -0.177]</td></tr></table>

Table 15: OB $F _ { 1 }$ of the no-context adapters minus that of the natural adapters trained at each window. Rows are ordered by truncation rate, on one evaluation configuration (2048-token window, no repetition penalty), with paired bootstrap 95 % CIs over documents. The change is the gap at 2048 minus the gap at 1024, which equals, with its sign reversed, what the natural adapters lost to truncation; ‘c. science’ stands for computer science.

## F.3 Evaluation footing

Every table in this paper takes its denominators from one set of runs, scored under the configuration of its numerators. The specialist denominators of our first cloze composition grid, taken from the archived bf16 runs, are 7 % higher on average than those on the 4-bit footing (0.584 against 0.545). Of the 0.045 by which the uniform footing lowers the specialists’ cloze gain over the base model, almost all lies on the specialists’ side: they move by $- 0 . 0 3 9 \left( \mathrm { C I _ { 9 5 \% } } = \left[ - 0 . 0 7 0 , - 0 . 0 1 0 \right] \right)$ and the base model by $+ 0 . 0 0 6 ~ \mathrm { ( C I _ { 9 5 \% } = [ - 0 . 0 1 6 , + 0 . 0 2 8 ] ) }$ The footing’s shifts in the composition grid come through the mean of ratios, since on the aggregate ratio it moves no composition CI across zero, and the no-context and style-normalized gains move by amounts not distinguishable from zero. The archived bf16 evaluation did not itself repeat exactly: two runs of the base model under the same code, configuration, and items agree on about five generations in six and on every generation in only a third of the documents, although their mean factual recall barely differs, and the records establish no cause; the 4-bit runs, by contrast, agree on every generation where they overlap (section F.4).

## F.4 What repeats exactly

Where the runs on the 4-bit cloze footing overlap, across five runs and four versions of the runner, they agree string for string on all 1393 generations compared (518 distinct generations, most of them compared in more than one pair of runs), including the retrieval run’s cells for each query’s own adapter, which reproduce the specialist denominators although that run loads and merges other adapters on the same base between them, as the 150 OB specialist scores repeat under one configuration. That is the standard against which a reproduction can be judged, and its scope is narrower than determinism in general: every one of these runs used one cluster queue, the two archived bf16 base-model cloze runs, from the same code and items, agree on only 883 of 1050 generations for reasons their records do not establish, and the archived 4-bit composition numerators match the cells they share with the retrieval run on 64 of 66 evaluations at the six decimals the archive stores.

## G Further related work

The closest work in framing is that of Back et al. (2026), who train adapters per document and per chunk, route among them by text embedding and by token-level routers, merge the three highest-ranked, and score answers with a language-model judge, both CB and with retrieved text in context. They measure how much a module stores and how routing and merging lose it, whereas we ask what the standard OB probe measures and whether the geometry of the weights indexes content; their retrieval matches query text to documents, ours adapters to adapters. With the ground-truth modules merged by TIES, their performance falls as the number merged grows from one to five, which is consistent with our cloze grid (section 5.4), although probe, merge operator and metric all differ.

Two re-evaluations of parametric RAG add details that section 7 omits. Tang et al. (2025b) find that token $F _ { 1 }$ had rewarded the short, template-like answers of the parametric variant, and Rakotonirina et al. (2026), whose merged per-document adapters collapse with a single distractor document, find four merges within about one point of each other at five documents: averaging, concatenation, TIES, and DARE (Yu et al., 2024), which drops a random fraction of each update’s entries and rescales the rest.

Gekhman et al. (2024) find that fine-tuning acquires new factual knowledge slowly and that doing so can increase hallucination. Per-entity adapters trained on a knowledge graph store facts that, CB, only the entity’s own adapter recovers, and neither text embedding nor weight-space distance finds that adapter in a bank that holds it (Pulici et al., 2026); our bank never holds the query’s own adapter and asks instead whether a neighbour’s can stand in, which on cloze it cannot detectably do (section 4.3), and on OB every adapter stands in about equally well (section 4.1).

Document adapters are close to the most favourable case for the interference that merging methods address, since their updates are nearly orthogonal layer by layer (section 3), and neither semantic nor weight-space distance anticipates what a TIES merge of two adapters costs (section E.5). KnOTS (Stoica et al., 2025) and Core Space merging (Panariello et al., 2025) merge full updates into an adapter whose rank may exceed that of its parts, so neither offers a matched-rank comparison for our merges, and section F.2 sets the $1 / N ^ { 2 }$ scale of factor averaging against the rescaling of Back et al. (2026) and LoRA-LEGO (Zhao et al., 2025).