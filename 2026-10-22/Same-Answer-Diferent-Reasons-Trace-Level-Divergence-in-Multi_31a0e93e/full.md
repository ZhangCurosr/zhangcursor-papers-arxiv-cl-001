# Same Answer, Diferent Reasons: Trace-Level Divergence in Multilingual Large Language Models

Rui Sun<sup>1,†</sup> Peizhao Mei<sup>1,\*</sup> Xiwei Cheng<sup>1,\*</sup> Kaizhuo Chen<sup>1,\*</sup> Taoran Li<sup>2</sup> Zhiyuan Yu<sup>2</sup> Varun Chandrasekaran<sup>1</sup>

<sup>1</sup>University of Illinois Urbana-Champaign <sup>2</sup>Texas A&M University

## Abstract

Multilingual large language models are increasingly deployed in settings where visible reasoning traces support verification, education, and safety auditing, yet standard evaluations typically measure only finalanswer agreement or holistic response similarity. As a result, they establish outcome consistency without determining whether users across languages receive comparable supporting evidence. We introduce a trace-level divergence framework that compares visible reasoning traces as structured sequences, decomposing agreement into structural similarity, order consistency, and holistic similarity, with pair-specific calibration to enable comparison across languages. Across three reasoning benchmarks, seven languages, and six frontier LLMs, the natural baseline—embedding each whole response and comparing the two by cosine— systematically overstates agreement: 52.1%– 67.3% of examples in the top quartile of holistic similarity are not in the top quartile of structural similarity. This dissocia tion is stable across sentence encoders and step segmentations, and blind bilingual annotation studies in two English-anchored language pairs show that human judgments align with structural similarity rather than with the holistic baseline. A same-language control shows that the level of divergence is language-specific: two English traces for paraphrases of the same question agree more closely than an English trace and its counterparts in other languages. Our results show that answer-level and holistic evaluations can substantially overestimate the consistency of multilingual explanatory artifacts, motivating trace-level evaluation for reliable multilingual deployment.

## 1 Introduction

Large language models (LLMs) are increasingly deployed in settings where they produce not only final answers but also visible reasoning traces intended to justify those answers. These traces are consumed by users, educators, auditors, and process-supervision systems to understand, verify, and improve model behavior (Wei et al., 2022). As multilingual LLMs become widely deployed, an implicit expectation follows: semantically equivalent questions posed in diferent languages should produce not only compatible answers but also broadly comparable supporting explanations. In this paper, we study the visible chain of thought emitted by the model, not its hidden internal computation. A chain of thought need not faithfully expose the model’s internal reasoning process (Turpin et al., 2023; Lanham et al., 2023; Chen et al., 2025), and we make no claim that it does. Rather, we study the explanatory artifact that downstream users, verifiers, and chain-of-thought monitors directly observe and act upon; our claims concern what that artifact contains in each language, not whether it reflects the computation behind it.

Existing multilingual evaluations largely measure outcome consistency: whether translated prompts produce the same or correct final answer (Shi et al., 2022; Xuan et al., 2025). More recent approaches compare complete responses using holistic semantic similarity, but still collapse an entire reasoning trace into a single representation. As a result, two multilingual responses may be judged equivalent even when they expose diferent intermediate claims, assumptions, problem decompositions, or execution order. Some of these diferences may simply reflect natural linguistic variation; others may represent substantively diferent supporting evidence. Existing evaluations cannot distinguish between them. We therefore argue that multilingual systems should also be evaluated for evidentiary consistency: whether users receive comparable supporting evidence for the same conclusion regardless of the language they ask in.

To address this problem, we introduce a trace-level divergence framework that compares multilingual reasoning traces as structured sequences rather than as single holistic responses. Our framework aligns step-structured chains of thought using multilingual sentence embeddings and decomposes cross-lingual trace consistency into three complementary metrics: structural similarity S, measuring preservation of reasoning content; order consistency τ , measuring preservation of reasoning order; and holistic similarity H, a whole-response baseline. Because multilingual sentence embeddings exhibit language-pair-dependent similarity floors and ceilings, a pair-specific calibration rescales sentence similarities onto a readable scale within each language pair before aggregation.

We evaluate the proposed framework on three reasoning benchmarks, seven languages, and six frontier LLMs spanning four model families. We find that whole-answer similarity systematically overstates agreement between multilingual reasoning traces. Across all model–dataset evaluations, holistic similarity remains consistently high $( H = 0 . 7 7 0 – 0 . 8 5 7 )$ while structural similarity is substantially lower $( S = 0 . 5 0 2 – 0 . 6 2 9 )$ , with order consistency lower still $( \tau = 0 . 3 0 7 – 0 . 4 7 0 )$ , even when both language outputs produce the same verified-correct answer. Rank-based analyses reach the same conclusion: within each model–dataset evaluation and language pair, 52.1%–67.3% of examples in the top quartile of holistic similarity are not also in the top quartile of structural similarity. Finally, blind bilingual annotation studies in two English-anchored language pairs show that human judgments of reasoning-trace equivalence align with structural similarity rather than holistic similarity, indicating that wholeresponse embeddings systematically overestimate agreement in the explanatory artifacts multilingual users actually observe.

Further analyses locate the divergence. In a same-language control, where each question is paraphrased in English and run through the identical pipeline, two English traces agree more closely than an English trace and its counterparts in other languages in every cell tested, although the holistic–structural dissociation itself also appears within English. Within a trace, alignment is weakest at the step stating the shared answer, and divergence grows on harder items. Throughout, we read low structural or order similarity as divergence, not failure.

Our contributions are threefold. First, we formulate multilingual evaluation as the problem of measuring the consistency of visible reasoning traces, complementing traditional outcomebased evaluation. Second, we introduce a calibrated trace-level framework that decomposes agreement into content preservation, execution order, and holistic similarity. Third, through a comprehensive empirical study, we show that answer-level and holistic evaluations systematically overstate the consistency of multilingual explanatory artifacts, motivating tracelevel evaluation for reliable multilingual deployment.

## 2 Problem Formulation

We study whether multilingual LLMs produce consistent visible reasoning traces when given semantically equivalent prompts in diferent languages. Our goal is to evaluate two complementary notions of consistency: (i) whether multilingual prompts produce the same final answer, and (ii) whether they produce comparable supporting explanations. The former captures outcome consistency; the latter captures trace consistency. This section formalizes both notions and introduces the calibration required to compare multilingual reasoning traces across language pairs.

Setup. Let $f _ { \theta }$ denote a fixed LLM and let $\mathcal { Q } =$ $\{ q _ { i } \} _ { i = 1 } ^ { N }$ be a set of reasoning problems specified at the level of semantic intent. For each $q _ { i } .$ we collect $K$ semantic-preserving translations $\mathcal { L } _ { i } = \{ \ell _ { i , 1 } , \ldots , \ell _ { i , K } \}$ and obtain responses

$$
y _ { i , k } = f _ { \theta } ( \ell _ { i , k } ) .
$$

Each response is parsed into a final answer $\alpha _ { i , k }$ and a step-structured visible reasoning trace

$$
\boldsymbol { r } _ { i , k } = ( s _ { i , k , 1 } , \ldots , s _ { i , k , t _ { i , k } } ) ,
$$

where $t _ { i , k }$ denotes the number of reasoning steps. Each step is itself a sequence of sen-

tences,

$$
s _ { i , k , t } = ( u _ { i , k , t , 1 } , \dots , u _ { i , k , t , n _ { i , k , t } } ) ,
$$

where $n _ { i , k , t }$ is the number of sentences in step t. Let $\alpha _ { i } ^ { \star }$ denote the reference answer.

## 2.1 Outcome and Trace Consistency

We formalize multilingual consistency at two complementary levels. Outcome consistency asks whether translated prompts produce the same final answer. Trace consistency asks whether they produce comparable visible reasoning traces. These notions are complementary: multilingual responses may reach the same answer while exposing diferent supporting claims, decompositions, or execution order. 1. Outcome Consistency. Following prior multilingual evaluation work (Qi et al., 2023; Shi et al., 2022), we define

$$
\begin{array} { l } { \displaystyle \Delta _ { \mathrm { a c c } } ( q _ { i } ) = \operatorname* { m a x } _ { k } \mathbf { 1 } [ \alpha _ { i , k } \neq \alpha _ { i } ^ { \star } ] } \\ { \displaystyle - \operatorname* { m i n } _ { k } \mathbf { 1 } [ \alpha _ { i , k } \neq \alpha _ { i } ^ { \star } ] \ \in \ \{ 0 , 1 \} . } \end{array}\tag{1}
$$

Here, $\Delta _ { \mathrm { a c c } } = 0$ if all language variants are uniformly correct or uniformly incorrect, and $\Delta _ { \mathrm { a c c } } = 1$ otherwise. While this captures multilingual agreement at the level of outcomes, it provides no information about whether the accompanying visible reasoning traces remain consistent across languages.

2. Trace Divergence. To quantify divergence between visible reasoning traces, we define

$$
D _ { \mathrm { t r a c e } } ( q _ { i } ) = \frac { 2 } { K ( K - 1 ) } \sum _ { 1 \leq k < k ^ { \prime } \leq K } d ( r _ { i , k } , r _ { i , k ^ { \prime } } ) .\tag{2}
$$

Rather than a single scalar, we represent trace divergence using three complementary dimensions,

$$
\begin{array} { c } { { d ( r _ { a } , r _ { b } ) = \Big ( 1 - S ( r _ { a } , r _ { b } ) , ~ \frac { 1 - \tau ( r _ { a } , r _ { b } ) } { 2 } , } } \\ { { 1 - H ( r _ { a } , r _ { b } ) \Big ) . } } \end{array}\tag{3}
$$

Thus, Eq. (2) yields a three-dimensional expectation over all language pairs. The three components capture complementary aspects of multilingual trace consistency:

• Structural similarity $S ( \cdot , \cdot ) \in [ 0 , 1 ] ;$ : preservation of step-level reasoning content across traces (§ 3.3).

• Order consistency $\tau ( \cdot , \cdot ) \in [ - 1 , 1 ]$ : preservation of the ordering of corresponding reasoning steps (§ 3.4).

• Whole-answer cosine $H ( \cdot , \cdot ) \ \in \ [ - 1 , 1 ]$ : a holistic embedding-based baseline that compares complete responses without localizing where divergence occurs (formally defined in Appendix B.6).

Raw multilingual cosines carry language-pairspecific floors and ceilings, so the sentence similarities behind S and τ are calibrated per language pair before aggregation (§ 3.2, $\mathrm { A p \mathrm { - } }$ pendix A).

## 3 Method

Given two visible reasoning traces produced for semantically equivalent prompts in diferent languages, our framework measures whether they preserve the same step-level content and ordering. The pipeline first parses each response into steps (§ 3.1), calibrates multilingual sentence similarities for each language pair (§ 3.2), and constructs a step-pair similarity matrix through two-level greedy alignment (§ 3.3). This matrix yields the structural similarity score S and the order-consistency score τ (§ 3.4). We compute the holistic baseline H separately over complete responses (Appendix B.6).

## 3.1 Step Parsing

We segment each response using line-initial step headers. The parser first matches the prompted English pattern Step $n \mathrm { : } ,$ followed by its equivalents in the seven evaluated languages, a plain numbered-list pattern, and finally paragraph boundaries. The same ordered rule set is applied to every language, and parsing terminates at the final-answer line. Each non-empty step is further divided into sentences using a Unicodeaware regular expression. No LLM is used during parsing. Appendix B.1 provides the complete parsing hierarchy and validation details.

## 3.2 Per-Pair Calibration

Raw multilingual cosine similarities $c _ { a , b } ( u , v ) =$ cos $( \phi ( u ) , \phi ( v ) )$ , for a multilingual sentence encoder $\phi ,$ exhibit language-pair-specific floors and ceilings. For each language pair $( a , b )$ we therefore estimate an unrelated-sentence floor $\widehat { \mathrm { H o o r } } _ { a , b } .$ , the mean cosine between sentences drawn from traces of diferent questions, and a self-translation ceiling ${ \widehat { \mathrm { c e i l i n g } } } _ { a , b } ,$ a high percentile of the cosine between trace sentences and their LLM translations into the other language (estimators and constants in Appendix B.2). Each raw sentence cosine is then rescaled as

$$
\widetilde { c } _ { a , b } ( \boldsymbol { u } , \boldsymbol { v } ) = \mathrm { c l i p } \left( \frac { c _ { a , b } ( \boldsymbol { u } , \boldsymbol { v } ) - \widehat { \mathrm { H o o r } } _ { a , b } } { \widehat { \mathrm { c e i l i n g } } _ { a , b } - \widehat { \mathrm { H o o r } } _ { a , b } } , 0 , 1 \right)\tag{4}
$$

Calibration aligns the empirical endpoints of each language pair, so that within a pair 0 marks the unrelated-sentence floor and 1 the self-translation ceiling; it does not assume that intermediate calibrated values are semantically identical across pairs. Within a pair it is a fixed monotone map, so it cannot reorder instances inside a pair, and the within-pair rank comparisons that carry our results are unaffected by it (§ 5.2). Appendix B.2 gives the estimators, the motivation for the constants, and translator-independence checks.

## 3.3 Two-Level Greedy Alignment

A trace is an ordered sequence of steps, and each step may contain multiple sentences. We therefore align traces hierarchically: an inner alignment compares sentences to score each step pair, and an outer alignment aggregates step-pair scores into a trace-level structural similarity.

Inner sentence alignment. For steps $s _ { a , i }$ and $s _ { b , j }$ containing $n _ { a , i }$ and ${ n } _ { b , j }$ sentences, respectively, let $\widetilde { \mathbf { C } } \in [ 0 , 1 ] ^ { n _ { a , i } \times n _ { b , j } }$ contain the calibrated sentence similarities from Eq. (4). We define the precision-style score $p _ { i , j } ^ { \mathrm { i n } } ~ =$ $\frac { 1 } { n _ { a , i } } \sum _ { m = 1 } ^ { n _ { a , i } }$ ma $\mathfrak { c } _ { n } \widetilde { \mathbf { C } } _ { m , n } .$ the recall-style score $\begin{array} { r } { r _ { i , j } ^ { \mathrm { i n } } = \frac { 1 } { n _ { b , j } } \sum _ { n = 1 } ^ { n _ { b , j } } } \end{array}$ max ${ \bf \Omega } _ { m } \widetilde { \mathbf { C } } _ { m , n } .$ , and the step-pair similarity $\sigma _ { i , j } = F _ { 1 } ( p _ { i , j } ^ { \mathrm { i n } } , r _ { i , j } ^ { \mathrm { i n } } )$ . This BERTScorestyle greedy alignment (Zhang et al., 2020) is high only when the sentences in each step are covered by the other.

Outer step alignment. Collect the step-pair similarities into $\mathbf { A } \in [ 0 , 1 ] ^ { t _ { a } \times t _ { b } }$ , where $\mathbf { A } _ { i , j } =$ $\sigma _ { i , j }$ . The structural similarity is

$$
S ( r _ { a } , r _ { b } ) = F _ { 1 } \left( \frac { 1 } { t _ { a } } \sum _ { i } \operatorname* { m a x } _ { j } \mathbf { A } _ { i , j } , \mathbf { \Omega } \frac { 1 } { t _ { b } } \sum _ { j } \operatorname* { m a x } _ { i } \mathbf { A } _ { i , j } \right) .\tag{5}
$$

The bidirectional maxima permit many-to-one correspondences, allowing traces to express the same content at diferent step granularities. Because S ignores the locations of matched steps, it measures content preservation independently of order.

## 3.4 Soft Kendall-τ Order Consistency

Structural similarity does not account for the positions of matched steps. We therefore derive an order-consistency score from the same steppair matrix A.

Soft assignment. For the directed comparison $a  b ,$ we sharpen the step similarities and normalize each source-step row:

$$
\begin{array} { r l } & { \widetilde { \mathbf { A } } _ { i , j } = \mathbf { A } _ { i , j } ^ { 3 } , } \\ & { P _ { i , j } ^ { a  b } = \{ \begin{array} { l l } { \displaystyle \frac { \widetilde { \mathbf { A } } _ { i , j } } { \sum _ { j ^ { \prime } } \widetilde { \mathbf { A } } _ { i , j ^ { \prime } } } , } & { \sum _ { j ^ { \prime } } \widetilde { \mathbf { A } } _ { i , j ^ { \prime } } > 0 , } \\ { \displaystyle \frac { 1 } { t _ { b } } , } & { \mathrm { o t h e r w i s e } . } \end{array}  } \end{array}\tag{6}
$$

(7)

Thus, $P _ { i , j } ^ { a  b }$ represents the soft correspondence between source step i and target step $j .$ . Sharpening concentrates mass on stronger matches without imposing a hard one-to-one assignment. Appendix D.8 reports sensitivity to the sharpening choice.

Order mass. For each ordered source-step pair $i ~ < ~ k$ , we accumulate probability mass according to whether their matched target positions are concordant, discordant, or tied:

$$
\begin{array} { c } { { M _ { r } ^ { a  b } = \displaystyle \frac { 1 } { \binom { t _ { a } } { 2 } } \sum _ { i < k } \sum _ { ( j , \ell ) \in \Omega _ { r } } { \cal P } _ { i , j } ^ { a  b } { \cal P } _ { k , \ell } ^ { a  b } , } } \\ { { r \in \{ \mathrm { c o n } , \mathrm { d i s } , \mathrm { t i e } \} , } } \end{array}\tag{8}
$$

where $\Omega _ { \mathrm { c o n } } = \{ ( j , \ell ) : j < \ell \} , \Omega _ { \mathrm { d i s } } = \{ ( j , \ell )$ $j > \ell \}$ , and $\Omega _ { \mathrm { t i e } } = \{ ( j , \ell ) : j = \ell \}$ . The directed order score is

$$
\tau ^ { a  b } = \frac { M _ { \mathrm { c o n } } ^ { a  b } - M _ { \mathrm { d i s } } ^ { a  b } } { M _ { \mathrm { c o n } } ^ { a  b } + M _ { \mathrm { d i s } } ^ { a  b } + M _ { \mathrm { t i e } } ^ { a  b } } .\tag{9}
$$

Because this score is directional, we symmetrize it:

$$
\tau ( r _ { a } , r _ { b } ) = \frac { 1 } { 2 } ( \tau ^ { a  b } + \tau ^ { b  a } ) ,\tag{10}
$$

using A for $a  b$ and $\mathbf { A } ^ { \top }$ for $b  a$ . The score is undefined when either trace has fewer than two steps, when either trace is empty, or when either directed denominator is zero. We interpret τ jointly with S: high τ with low S indicates that the matched content is ordered consistently but structurally incomplete.

## 4 Experimental Setup

We evaluate on three multilingual reasoning benchmarks: MMLU-ProX (Xuan et al., 2025), StrategyQA (Geva et al., 2021), and SciBench (Wang et al., 2024). Each run uses seven languages (Arabic, English, Spanish, Hindi, Russian, Vietnamese, and Chinese), yielding 21 unordered language pairs per model–dataset evaluation. The benchmark suite spans distinct reasoning regimes: MMLU-ProX tests multiple-choice academic reasoning, where answer options and domain knowledge constrain the response; StrategyQA tests yes/no commonsense reasoning, where premise interpretation and implicit world knowledge are central; and SciBench tests quantitative STEM reasoning, where traces often combine formula selection, unit conversion, and numerical computation. This spread lets us distinguish dataset efects from model-specific behavior under fixed metric definitions.

We evaluate six models from four families: Claude Opus 4.6, Claude Haiku 4.5, DeepSeek-V3.2, GPT-OSS-120B, Qwen3-235B, and Qwen3-32B, sampling 500 problems per dataset–model run. Trace metrics are reported only after the answer- and trace-validity cascade; one-language-correct cases contribute only to $\Delta _ { \mathrm { a c c } }$ (Appendix C.1).

Metrics follow § 3, using LaBSE (Feng et al., 2022) for step scores, BGE-M3 (Chen et al., 2024) for H, and soft Kendall-τ sharpening power 3. Table 1 reports problem-level 95% bootstrap confidence intervals (2,000 resamples), resampling each question’s 21 language pairs jointly. Appendices C.3, C.4, D.8, and D.5 provide filtering details, audits, and ablations.

## 5 Results

Our evaluation asks whether final-answer agreement implies trace agreement across languages (§ 5.1), whether whole-answer similarity can substitute for step-level alignment (§ 5.2), whether structural, order, and holistic similarity provide complementary signals (§ 5.3), whether the divergence is specific to crossing a language (§ 5.4), where in a trace it appears and what a low score signals (§ 5.5), and whether the resulting metrics are robust and agree with human judgment (§ 5.6).

Appendix D provides the supporting result details: pair-level tables (Appendix D.2), cross-dataset and cross-model patterns (Appendix D.3), qualitative verified traces (Appendix E.1), and rejected-sample audits outside the primary verified-correct set (Appendix E.2).

## 5.1 Correct Answers Still Hide Trace Divergence

We first test whether agreement on the final answer implies agreement in the emitted reasoning trace by restricting to language pairs where both answers are verified correct. If multilingual reasoning were invariant, such pairs would exhibit high structural similarity S, high order consistency τ , and only a small gap between the holistic score H and the step-level score S.

However, across the evaluations in Table 1, H remains consistently high (0.770–0.857), while S is substantially lower (0.502–0.629) and τ lower still (0.307–0.470). The same pattern holds on MMLU-ProX: across six 500-sample evaluations (21 language pairs), S=0.513–0.569, $\tau { = } 0 . 3 4 8 { - } 0 . 4 6 9$ , and $H { = } 0 . 7 8 7 { - } 0 . 8 0 0$

Taken together, these results contradict the invariance hypothesis. Even when answers agree, reasoning traces diverge: responses remain globally similar as full strings (high H) while difering in step content (S) and, more markedly, in step ordering (τ ), particularly on MMLU-ProX.

## 5.2 Holistic Similarity Overstates Consistency

We test whether a whole-answer embedding baseline can substitute for step-level alignment by comparing H against S. Because H uses BGE-M3 (8K context) while S uses LaBSE for trace-length reasons (Appendix B.6), we treat H−S as a diagnostic contrast and rely on a scale-free rank comparison. If a holistic embedding captures cross-lingual reasoning structure, high H should coincide with high S within the same evaluation and language pair.

In practice, Table 1 shows aggregate H−S gaps of 0.152 to 0.327 across the 21 languagepair evaluations; we do not interpret these absolute magnitudes as calibrated distances given the encoder asymmetry. Figure 1 provides the scale-free rank comparison (computed within each (evaluation, pair, encoder) cell, hence invariant to per-pair monotone rescaling): within each evaluation and language pair, 52.1%–67.3% of top-quartile H examples fall outside the top quartile of S, and a nontrivial fraction land in the bottom structural quartile. Appendix D.4 reports the full rank table and representative raw point-cloud panels.

Table 1: Aggregate divergence analyses. Q is the per-evaluation question count after the trace-validity cascade (Appendix C.1); P is retained pair–question instances; “pairs” is the number of language pairs. S (structural similarity), τ (soft Kendall-τ), and H (whole-answer BGE-M3 cosine) are macro-averaged over Q; brackets are half-widths of problem-level percentile bootstrap 95% CIs (2,000 resamples). H−S is a diagnostic gap, not a calibrated distance. Per-language outcome accuracies (Fig. 2) are computed on the full 500-problem inference set before this filter and are not bounded by Q.
<table><tr><td>Dataset</td><td>Model</td><td># samples</td><td>Q</td><td>P</td><td># pairs</td><td></td><td>S</td><td></td><td>T</td><td></td><td></td><td>H</td><td>H-S</td></tr><tr><td>MMLU-PROX</td><td>Claude Opus 4.6</td><td>500</td><td>431</td><td>8,470</td><td></td><td>0.569</td><td>[±0.007]</td><td></td><td>0.414 [ [±0.014]</td><td></td><td>0.800 [±0.004]</td><td></td><td>0.231</td></tr><tr><td>MMLU-PROX</td><td>Claude Haiku 4.5</td><td>500</td><td>321</td><td>5,459</td><td></td><td>0.514</td><td>[±0.009]</td><td></td><td>0.393</td><td>[±0.016]</td><td>0.787</td><td>[±0.004]</td><td>0.273</td></tr><tr><td>MMLU-PROX</td><td>DeepSeek-V3.2</td><td>500</td><td>394</td><td>7,179</td><td>21 21</td><td>0.537</td><td>[±0.007]</td><td></td><td>0.356 [±0.014]</td><td></td><td>0.794 [±0.004]</td><td></td><td>0.257</td></tr><tr><td>MMLU-PROX</td><td>GPT-OSS-120B</td><td>500</td><td>372</td><td>6,552</td><td></td><td>0.542</td><td>[±0.008]</td><td></td><td>0.469 [±0.016]</td><td></td><td>0.791 [±0.004]</td><td></td><td>0.249</td></tr><tr><td>MMLU-PROX</td><td>Qwen3-235B</td><td>500</td><td>370</td><td>6,656</td><td></td><td>0.540</td><td>[±0.006]</td><td></td><td>0.348 [±0.015]</td><td></td><td>0.793 [±0.004]</td><td></td><td>0.253</td></tr><tr><td>MMLU-PROX</td><td>Qwen3-32B</td><td>500</td><td>295</td><td>4,714</td><td></td><td>0.513</td><td>[±0.008]</td><td></td><td>0.412 [±0.015]</td><td></td><td>0.791 [±0.005]</td><td></td><td>0.278</td></tr><tr><td>SCIBENCH</td><td>Claude Opus 4.6</td><td>500</td><td>362</td><td>7,062</td><td></td><td>0.611</td><td>[±0.006]</td><td></td><td>0.447 [±0.013]</td><td></td><td>0.780 [±0.004]</td><td></td><td>0.169</td></tr><tr><td>SCIBENCH</td><td>Claude Haiku 4.5</td><td>500</td><td>296</td><td>4,739</td><td></td><td>0.572</td><td>[±0.009]</td><td></td><td>0.470 [±0.016]</td><td></td><td>0.770 [±0.004]</td><td></td><td>0.197</td></tr><tr><td>ScIBENCH</td><td>DeepSeek-V3.2</td><td>500</td><td>357</td><td>5,977</td><td></td><td>0.502</td><td>[±0.009]</td><td></td><td>0.326 [±0.012]</td><td></td><td>0.785 [±0.004]</td><td></td><td>0.283</td></tr><tr><td>SCIBENCH</td><td>GPT-OSS-120B</td><td>500</td><td>324</td><td>5,435</td><td></td><td>0.592</td><td>[±0.011]</td><td></td><td>0.321 [±0.015]</td><td></td><td>0.796 [±0.004]</td><td></td><td>0.204</td></tr><tr><td>SCIBENCH</td><td>Qwen3-235B</td><td>500</td><td>336</td><td>5,686</td><td></td><td>0.522</td><td>[±0.009]</td><td></td><td>0.339 [±0.013]</td><td></td><td>0.783 [±0.004]</td><td></td><td>0.261</td></tr><tr><td>SCIBENCH</td><td>Qwen3-32B</td><td>500</td><td>268</td><td>3,874</td><td></td><td>0.629</td><td>[±0.012]</td><td></td><td>0.332 [±0.015]</td><td></td><td>0.781 [±0.005]</td><td></td><td>0.152</td></tr><tr><td>STRATEGYQA</td><td>Claude Opus 4.6</td><td>500</td><td>451</td><td>9,039</td><td></td><td>0.598</td><td>[±0.007]</td><td></td><td>0.418 [±0.012]</td><td></td><td>0.818 [±0.004]</td><td></td><td>0.220</td></tr><tr><td>STRATEGYQA</td><td>Claude Haiku 4.5</td><td>500</td><td>379</td><td>6,857</td><td></td><td>0.530</td><td>[±0.008]</td><td></td><td>0.400</td><td>[±0.013]</td><td>0.857 [±0.004]</td><td></td><td>0.327</td></tr><tr><td>STRATEGYQA</td><td>DeepSeek-V3.2</td><td>500</td><td>414</td><td>7,813</td><td>21</td><td>0.560</td><td>[±0.006]</td><td></td><td>0.324 [±0.010]</td><td></td><td>0.795 [±0.004]</td><td></td><td>0.235</td></tr><tr><td>STRATEGYQA</td><td>GPT-OSS-120B</td><td>500</td><td>357</td><td>6,751</td><td>21</td><td>0.541</td><td>[±0.007]</td><td></td><td>0.380 [±0.013]</td><td></td><td>0.816 [±0.004]</td><td></td><td>0.275</td></tr><tr><td>STRATEGYQA</td><td>Qwen3-235B</td><td>500 500</td><td>402</td><td>7,591</td><td>21 21</td><td>0.560</td><td>[±0.007]</td><td></td><td>0.355 [±0.012] [±0.012]</td><td></td><td>0.801 [±0.004] [±0.005]</td><td></td><td>0.240</td></tr><tr><td>STRATEGYQA</td><td>Qwen3-32B</td><td></td><td>301</td><td>5,177</td><td></td><td>0.515</td><td>[±0.008]</td><td></td><td>0.307</td><td></td><td>0.793</td><td></td><td>0.278</td></tr></table>

![](images/fdadd070d1e69775b1deaf9354d3371d095b2159a46c3559ad2d2339a1264b82.jpg)  
Figure 1: Rank-based holistic-to-structural mismatch across all 18 model–dataset evaluations. Each cell reports the share of pair–question instances in the top quartile of whole-answer cosine H that are not also in the top quartile of structural score S, with quartiles computed within the same evaluation and language pair. Appendix D.4 reports the per-dataset breakdown, including the stricter share of top-H instances falling in the bottom structural quartile.

This comparison is unafected by the per-pair calibration: quartile thresholds are computed within each language pair, where calibration is a single monotone map, and recomputing the statistic on the uncalibrated scale reproduces it to four decimal places on all three Claude Opus 4.6 evaluations (Appendix B.2.4). The reported mismatch is therefore a property of the traces rather than of the calibration constants.

These results do not support replacing steplevel alignment with a holistic baseline. While H serves as a coarse semantic signal, it is structurally under-informative, failing to detect missing, merged, or reordered reasoning steps: a consistency check built on whole-response cosine returns a high score on precisely the pairs where step content and order have moved.

## 5.3 Structural Coverage and Order Are Non-Redundant

We finally test whether the decomposition into S, τ , and H is necessary by examining whether the three metrics capture distinct variation. If a single trace-level score were suficient, then S, τ , and H would be strongly correlated after controlling for the remaining metric.

In practice, Table 2 shows that τ and H are nearly uncorrelated, while S has only moderate independent association with both. The same separation holds at the aggregate level: evaluations with similar S can exhibit markedly diferent order consistency, and large H−S gaps or top-H mismatches do not align with τ . Appendix D.3 provides a complementary crossdataset check (Claude Opus 4.6): languagepair rankings of H replicate strongly across datasets $\left( \rho = 0 . 7 8 – 0 . 9 8 \right)$ , τ rankings replicate moderately $( \rho = 0 . 5 2 – 0 . 7 6 )$ , while S is more task-sensitive $( \rho = 0 . 1 1 \mathrm { - } 0 . 5 5 )$

Table 2: Pairwise Pearson and partial correlations among the three metrics at the pair–question level for Claude Opus 4.6. For each metric pair, the first value is the Pearson correlation and the second is the partial correlation after controlling for the remaining metric.
<table><tr><td>Dataset</td><td>n</td><td> $S { - } \tau$  S-H</td><td> $\tau { - } H$ </td></tr><tr><td>MMLU-PROX</td><td>8,470 .29 /.29</td><td>.29 /.29</td><td> $. 0 5 / - . 0 4$ </td></tr><tr><td>STRATEGYQA</td><td></td><td></td><td>9,039 .30 / .31 .33 / .33 .04/ −.07</td></tr><tr><td>SCIBENCH</td><td></td><td></td><td>7,062 .17/.17 .29 /.29 .02 / −.03</td></tr></table>

These results reject the single-score hypothesis. The three metrics capture complementary aspects of reasoning: $S$ reflects step coverage, τ captures execution order, and H measures only holistic semantic similarity. The cross-dataset pattern further highlights this distinction: holistic similarity is relatively stable across tasks, order behaviour is partly language-pair dependent, and structural coverage is most sensitive to how reasoning is elicited. Collapsing these axes would obscure both task- and modelspecific forms of trace divergence.

## 5.4 Divergence Is Larger Across Languages Than Within One

Every comparison so far crosses a language boundary, so on its own it cannot separate divergence caused by changing language from ordinary variation in generated reasoning. As a same-language control, we rewrite each question as a meaning-preserving English paraphrase and run it through the identical pipeline as a condition $\mathrm { e n } _ { \mathrm { p a r a } }$ that receives the same appended instruction as a translated condition, for GPT-OSS-120B and DeepSeek-V3.2 on all three benchmarks, about 100 questions each (Appendix H). Pairing within question, we compare $S ( \mathrm { e n , e n _ { p a r a } } )$ with the same question’s mean S over its English-anchored cross-lingual pairs. Two English traces agree more closely in all six dataset–model cells and in 272 of 337 questions (Table 3; pooled $\Delta S = + 0 . 0 9 0$ 95% CI $[ + 0 . 0 7 7 , + 0 . 1 0 3 ] , p < 1 0 ^ { - 3 3 } )$ , and order consistency moves the same way in every cell. For this cross-pair comparison of levels, calibration favours the same-language pair by at most 0.024; subtracting that bias cell by cell leaves $\Delta S$ between +0.046 and +0.124, above thresholds fixed before the run.

Table 3: Same-language control (GPT-OSS: GPT-OSS-120B; DeepSeek: DeepSeek-V3.2). ∆S is $S ( \mathrm { e n , e n _ { p a r a } } )$ minus the same question’s mean S over its six English-anchored cross-lingual pairs; brackets are half-widths of 95% t-intervals. “#>0” counts questions with $\Delta S > 0 .$
<table><tr><td>Dataset</td><td>Model</td><td>n</td><td>∆S</td><td>#&gt;0</td></tr><tr><td>MMLU-PROX</td><td>GPT-OSS</td><td>59+0.120</td><td>[±0.029]</td><td>49</td></tr><tr><td>MMLU-PROX</td><td>DeepSeek</td><td>68+0.071</td><td>[±0.024]</td><td>56</td></tr><tr><td>SCIBENCH</td><td>GPT-OSS</td><td>41+0.136</td><td>[±0.048]</td><td>36</td></tr><tr><td>SCIBENCH</td><td>DeepSeek</td><td>47+0.083</td><td>[±0.037]</td><td>39</td></tr><tr><td>STRATEGYQA</td><td>GPT-OSS</td><td>57+0.041</td><td>[±0.029]</td><td>36</td></tr><tr><td>STRATEGYQA</td><td>DeepSeek</td><td>65+0.101</td><td>[±0.028]</td><td>56</td></tr><tr><td>Pooled</td><td></td><td></td><td>337 +0.090 [±0.013]</td><td>272</td></tr></table>

The rank dissociation of § 5.2, by contrast, persists within English: 41.2%–60.0% of topquartile-H instances are not top-quartile-S between two English traces, against 50.5%–64.6% across languages in the same run. Holistic similarity thus overstates step-level agreement whether or not the traces share a language, a property of the metrics; what is specific to crossing a language is the level of divergence.

## 5.5 Where Divergence Appears and What It Signals

Position. Scoring each English step by its best match in the other trace (English-anchored pairs, all 18 evaluations; Appendix I.1), alignment is weakest at the end: the final step is less similar than the middle steps by 0.042 (95% CI [0.038, 0.046]; lower in 16 of 18 evaluations), and the gap widens to 0.058 when both traces have the same number of steps, so it is not an unmatched extra step. Every pair here reaches the same correct answer; the step stating it is the least aligned. Where traces first part depends on the task, in an order stable across thresholds: SciBench and MMLU-ProX traces often part at the setup step, StrategyQA traces mid-trace. The profile has the same shape in all six target languages, which difer only in level.

Dificulty and correctness. Divergence is larger on harder items: indexing dificulty by the number k of the seven languages that answer correctly, mean S over correct EN–ZH and EN–ES pairs rises from 0.526 at k = 2 to 0.579 at $k = 7 ~ ( \rho = 0 . 1 0 , p < 1 0 ^ { - 2 5 }$ ; Appendix I.2). Scoring the pairs the main analyses exclude, among answer-agreeing pairs the 225 that share a wrong answer have lower S than the 11,176 that share the correct one (Clif’s $\delta = + 0 . 2 9 3$ 95% CI $[ + 0 . 2 1 7 , + 0 . 3 6 6 ] )$ , while H does not separate them $( \delta = - 0 . 0 2 7 , [ - 0 . 1 1 4 , + 0 . 0 5 5 ] )$ Against correct pairs of equal k, however, the S contrast falls to $+ 0 . 0 7 3 \ ( [ - 0 . 0 6 1 , + 0 . 2 0 9 ] )$ : low S marks hard items, on which shared errors concentrate, rather than errors as such.

Reading a low score. Low S or τ is divergence, not failure: two correct traces can legitimately difer, and the step matrix shows where. It still helps allocate review. Answer agreement cannot flag answer-agreeing pairs that share a wrong answer; ranking such pairs by S within each evaluation and language pair, the least similar quarter contains 44.0% of these errors, against 28.9% for H and 25% at random (paired diference +15.1 points, [+6.2, +23.9]). At a 2% base rate this prioritizes a fixed review budget; it is not an error detector.

## 5.6 Robustness and Human Corroboration

The analyses above read low S as genuine tracelevel divergence. We check that this reading does not depend on the choice of sentence encoder or on step segmentation, and that it is corroborated by human judgment. Protocols, sampling, and full statistics are in Appendices D.6, F, and G.

Encoder robustness. Recomputing S and τ under two alternative sentence encoders—BGE-M3 (568M parameters), the encoder H already uses, so that H and S share an encoder, and a distilled MiniLM (118M)—leaves the rankbased finding intact: across three encoders and three datasets, every evaluation falls between 50.1% and 65.5% of top-quartile-H examples not being top-quartile-S, closely tracking the 52.1%–67.3% range reported above. Absolute calibrated levels are less stable, with the $H { - } S$ gap widening under one encoder and narrowing under the other. We therefore interpret S and τ by rank within a fixed encoder rather than by absolute level, and report them that way throughout. Because the framework reads only the emitted traces and a sentence encoder, its requirements are modest: it needs no access to weights or activations—two of the six evaluated models are closed-weight—and the 118M encoder runs on CPU.

Table 4: Spearman correlations between humanjudged reasoning equivalence and each metric $( N =$ 60 per study). Partial correlations control for the other metric. Brackets are bootstrap 95% CIs; they were not computed for the EN–ZH study. Ordinal Krippendorf’s α is 0.41 (EN–ZH) and 0.58 (EN– ES).
<table><tr><td rowspan="2">Quantity</td><td colspan="2">EN-ZH</td><td colspan="2">EN-ES</td></tr><tr><td>ρ</td><td>p</td><td>ρ [95% CI]</td><td>p</td></tr><tr><td>ρ(equiv, S)</td><td>.371</td><td>.004</td><td>.44 [.23, .62]</td><td>&lt;.001</td></tr><tr><td> $\rho ( \mathrm { e q u i v } , \tau )$ </td><td>.421</td><td>&lt;.001</td><td>.38 [.16, .57]</td><td>.001</td></tr><tr><td>ρ(equiv, H)</td><td>-.066</td><td>n.s.</td><td>.03 [−.21, .26]</td><td>.80</td></tr><tr><td>ρ(equiv, S | H)</td><td>.391</td><td>.002</td><td>.45 [.24, .63]</td><td>&lt;.001</td></tr><tr><td>ρ(equiv, H | S)</td><td>-.147</td><td>n.s.</td><td>−.08 [−.31, .16]</td><td>.51</td></tr></table>

Step segmentation. Restricting each evaluation to pairs whose traces have equal step counts lowers the statistic by 1.0–2.9 points on the three Claude Opus 4.6 evaluations and leaves all 18 evaluations between 49.5% and 71.0% (Appendix D.6); the parser recovers steps from 99.8%–100% of traces in every language.

Human corroboration. To check that the divergence S measures is one a reader would recognize, bilingual annotators rated verifiedcorrect Claude Opus 4.6 trace pairs on a threepoint reasoning-equivalence scale, blind to all metric values, in two language pairs (English– Chinese and English–Spanish, $N = 6 0$ each, sampled so that H and S both vary over their full ranges). Table 4 reports the correlations: equivalence judgments track S and τ in both pairs and are unrelated to H, and the separation survives controlling each metric for the other. The quantity we test directly is the diference between S and H as predictors of the same judgments. Williams’ test for dependent correlations gives $\Delta \rho = 0 . 4 4 , t ( 5 7 ) = 2 . 8 0$ 2 $p = 0 . 0 0 7$ for English–Chinese and $\Delta \rho = 0 . 4 1$ $t ( 5 7 ) = 2 . 8 1 , p = 0 . 0 0 7$ for English–Spanish, and a Fisher-z test finds no diference between the two language pairs $( p = 0 . 6 5 )$ even though Spanish is typologically the closer of the two to English. Asked to name the type of difference, annotators point to missing reasoning content, granularity, and step order rather than wording—the profile a content-sensitive measure should produce, and the opposite of what a verbalization detector would. A blind LLM judge given both traces and the same rubric tracks the human consensus less closely than S does, and never applies the Substantially different label.

These studies are small, and we present them as corroboration of the automated results rather than as independent evidence that S is superior to alternatives. Their role is to check that the metric responds to something human readers also see. They cover one model’s traces and two English-anchored pairs, and we do not claim the agreement extends beyond them.

## 6 Related Work

Multilingual reasoning evaluation. Chainof-thought prompting elicits step-by-step reasoning (Wei et al., 2022), but multilingual evaluations largely stop at outputs. Shi et al. (2022) measure multilingual CoT accuracy and Barua et al. (2025) extend this to long-form reasoning, while benchmarks such as MMLU-ProX (Xuan et al., 2025) and metrics like RankC (Qi et al., 2023) assess consistency at the level of final answers. Qi et al. (2025) instead control the language a model thinks in and measure the resulting accuracy cost, which is complementary along a distinct axis: they vary the trace language and observe correctness, whereas we hold correctness fixed and measure how far the traces diverge. None of these localizes divergence within a trace.

Trace-level reasoning evaluation. ROSCOE (Golovneva et al., 2023) scores reasoning chains at step level in monolingual settings, and we extend that perspective to the cross-lingual case, where similarity must be calibrated per language pair. ROSCOE is not an available baseline here: it scores a single chain against a reference to judge that chain’s quality, whereas we compare two equally valid traces whose quality is held fixed by the validity cascade, and its monolingual encoders would have to be replaced by a multilingual one—reducing the comparison to the uncalibrated step cosines our calibration exists to correct (Appendix C.4). Concurrent work (Zhao et al., 2026) studies cross-lingual transfer of rationales, but without step- or order-level alignment.

Faithfulness of the visible trace. A separate line of work asks whether an emitted chain of thought reflects the computation that produced the answer: models can rationalize a predetermined answer (Turpin et al., 2023; Lanham et al., 2023; Chen et al., 2025), and consequential computation can leave no interpretable trace in the output tokens at all (Baherwani et al., 2026). Our measurement is orthogonal to this question: we compare two emitted traces and make no claim that either exhausts internal computation. The visible trace is nonetheless what verification, education, and chain-ofthought monitoring pipelines consume (Korbak et al., 2025), so our claim concerns a monitor’s input, not its faithfulness: whatever a monitor extracts from a trace, it extracts from a diferent artifact in each language. This needs only the emitted text, so the framework also applies to closed-weight models.

Multilingual embeddings and alignment. We use LaBSE (Feng et al., 2022) for sentence embeddings and BGE-M3 (Chen et al., 2024) for holistic similarity. Our alignment extends embedding-based matching (Zhang et al., 2020) to sentence- and step-level structure, and uses a soft Kendall-τ inspired by continuous relaxations of permutations (Mena et al., 2018; Grover et al., 2019). Appendix B.8 gives a detailed comparison with prior step-level metrics.

## 7 Conclusion

We show that agreement on final answers does not imply agreement in the reasoning traces a model emits across languages: holistic similarity can remain high even when step-level structure and order diverge. The consequence is at the level of the artifact a user reads: for the same correct answer, the supporting evidence a reader is given can depend on the language they asked in—steps another reader never sees, or the same steps in a diferent order—more than on how the question is worded within one language. It appears in every one of the 21 language pairs we evaluate, including the typologically closest. By decomposing trace consistency into structural similarity S, order consistency τ, and holistic similarity H, our framework reveals complementary axes that single-score evaluations obscure and locates where traces part. The rank-based finding is stable across sentence encoders and step segmentations, and small blind annotation studies on one model’s traces in two English-anchored language pairs are consistent with it. Because the dissociation also appears between two English traces, the decomposition extends to process-level consistency beyond cross-lingual settings, including across paraphrases, prompts, and decoding strategies.

## Limitations

Visible traces, not internal computation. A chain of thought need not be a faithful record of the computation that produced the answer (Turpin et al., 2023; Lanham et al., 2023; Chen et al., 2025), and we link the divergence we measure to no internal quantity. The visible trace is what verification, education, and auditing pipelines consume, and that is the scope of our claims. Activation-level analysis would address faithfulness directly, but it would exclude closed-weight models, including two of the six we evaluate.

Ranks, not levels. What survives an encoder change is rank structure rather than absolute scale (Appendix G), and per-pair calibration aligns only the endpoints of the similarity scale. We therefore compare S and τ by rank within a fixed encoder and read absolute cross-pair values as descriptive only. The one cross-pair comparison of levels we draw, the same-language control of § 5.4, is paired within question and its calibration exposure is measured, but it remains a level comparison. Ceiling estimates additionally rely on LLM translation, whose co-dependence with the evaluated model we bound in Appendix D.1.

Scope of the evidence. The human validation covers two language pairs, both including English, with 60 verified-correct trace pairs each, all from one model; agreement is moderate, which attenuates rather than inflates the reported correlations. We do not establish that the metrics track human judgment for other models or for pairs without English. The same-language control covers two of the six models, about 100 questions per benchmark, and one paraphrase per question, and it does not resample an identical prompt at non-zero temperature, which would isolate decoding variation; its efect is unanimous in direction but varies about threefold in size across cells. Step boundaries are verified automatically and by a manual spot check too small to bound the residual error rate (Appendix B.1.1).

Divergence and outcomes. The association between low S and shared wrong answers rests on 225 pairs from two language pairs and is largely carried by item dificulty (§ 5.5). We draw no causal conclusion from it, and we do not show that divergence impairs verification, comprehension, or safety; that would require a study with human readers.

## Ethical Considerations

Data and annotation. Questions come from public reasoning benchmarks and the traces are model-generated, so no personal or sensitive data is involved. Annotators were bilingual speakers informed of the purpose of the study, and no personal data was collected from them; Appendix F.7 reports recruitment and consent in full.

Artifact release. We will release the evaluation code under a permissive open-source license. It builds on publicly released models and benchmarks—LaBSE, BGE-M3, and paraphrase-multilingual-MiniLM-L12-v2,

together with MMLU-ProX, StrategyQA, and SciBench—and our use and redistribution are consistent with their respective licenses. We redistribute no benchmark data and no annotation labels.

Intended use. The framework diagnoses whether a system reasons the same way in different languages; it does not certify reasoning quality. Scores are relative within a fixed encoder and language pair and should not be used to rank languages, to gate access to a service, or as a training target—traces can be made to agree without any improvement in the underlying reasoning. Because all pairs in the main analyses are verified-correct, a low score there marks a diferent route to the same answer, not a wrong one; its association with shared errors supports prioritizing human review, not detecting errors.

Fairness. Equal accuracy can mask unequal traces, so users who read a model’s explanation may receive systematically diferent support depending on the language they use.

## References

Vatsal Baherwani, Tom Goldstein, and Ashwinee Panda. 2026. Not all LLM reasoning is visible in the chain-of-thought. Preprint, arXiv:2607.22925.

Josh Barua, Seun Eisape, Kayo Yin, and Alane Suhr. 2025. Long chain-of-thought reasoning across languages. arXiv preprint arXiv:2508.14828.

Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. 2024. BGE M3- embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. arXiv preprint arXiv:2402.03216.

Yanda Chen, Joe Benton, Ansh Radhakrishnan, Jonathan Uesato, Carson Denison, John Schulman, Arushi Somani, Peter Hase, Misha Wagner, Fabien Roger, et al. 2025. Reasoning models don’t always say what they think. Preprint, arXiv:2505.05410.

Fangxiaoyu Feng, Yinfei Yang, Daniel Cer, Naveen Arivazhagan, and Wei Wang. 2022. Languageagnostic BERT sentence embedding. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (ACL).

Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot, Dan Roth, and Jonathan Berant. 2021. Did Aristotle use a laptop? A question answering benchmark with implicit reasoning strategies. Transactions of the Association for Computational Linguistics (TACL), 9:346–361.

Olga Golovneva, Moya Chen, Spencer Pof, Martin Corredor, Luke Zettlemoyer, Maryam Fazel-Zarandi, and Asli Celikyilmaz. 2023. ROSCOE: A suite of metrics for scoring step-by-step reasoning. In International Conference on Learning Representations (ICLR).

Aditya Grover, Eric Wang, Aaron Zweig, and Stefano Ermon. 2019. Stochastic optimization of sorting networks via continuous relaxations. In International Conference on Learning Representations (ICLR).

Tomek Korbak, Mikita Balesni, Elizabeth Barnes, Yoshua Bengio, Joe Benton, Joseph Bloom, Mark Chen, Alan Cooney, Allan Dafoe, Anca Dragan, et al. 2025. Chain of thought monitorability: A new and fragile opportunity for AI safety. Preprint, arXiv:2507.11473.

Tamera Lanham, Anna Chen, Ansh Radhakrishnan, Benoit Steiner, Carson Denison, Danny Hernandez, Dustin Li, Esin Durmus, Evan Hubinger, Jackson Kernion, et al. 2023. Measuring faithfulness in chain-of-thought reasoning. Preprint, arXiv:2307.13702.

Gonzalo Mena, David Belanger, Scott Linderman, and Jasper Snoek. 2018. Learning latent permutations with Gumbel-Sinkhorn networks. In International Conference on Learning Representations (ICLR).

Jirui Qi, Raquel Fernández, and Arianna Bisazza. 2023. Cross-lingual consistency of factual knowledge in multilingual language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP).

Jirui Qi, Philipp Mondorf, Gabriele Sarti, Barbara Plank, and Arianna Bisazza. 2025. When models reason in your language: Controlling thinking trace language comes at the cost of accuracy. In Findings of the Association for Computational Linguistics: EMNLP 2025.

Freda Shi, Mirac Suzgun, Markus Freitag, Xuezhi Wang, Suraj Srivats, Soroush Vosoughi, Hyung Won Chung, Yi Tay, Sebastian Ruder, Denny Zhou, Dipanjan Das, and Jason Wei. 2022. Language models are multilingual chain-of-thought reasoners. arXiv preprint arXiv:2210.03057.

Miles Turpin, Julian Michael, Ethan Perez, and Samuel R. Bowman. 2023. Language models don’t always say what they think: Unfaithful explanations in chain-of-thought prompting. Preprint, arXiv:2305.04388.

Xiaoxuan Wang, Ziniu Hu, Pan Lu, Yanqiao Zhu, Jieyu Zhang, Satyen Subramaniam, Arjun R. Loomba, Shichang Zhang, Yizhou Sun, and Wei Wang. 2024. SciBench: Evaluating college-level scientific problem-solving abilities of large language models. In Proceedings of the 41st International Conference on Machine Learning (ICML), volume 235 of Proceedings of Machine Learning Research, pages 50622–50649.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. 2022. Chainof-thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems (NeurIPS).

Weihao Xuan, Rui Yang, Heli Qi, Qingcheng Zeng, Yunze Xiao, Aosong Feng, Dairui Liu, Yun Xing, Junjue Wang, Fan Gao, et al. 2025. Mmlu-prox: A multilingual benchmark for advanced large language model evaluation. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 1513–1532.

Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi. 2020. BERTScore: Evaluating text generation with BERT. In International Conference on Learning Representations (ICLR).

Raoyuan Zhao, Yihong Liu, Hinrich Schütze, and Michael A. Hedderich. 2026. A comprehensive evaluation of multilingual chain-ofthought reasoning: Performance, consistency, and faithfulness across languages. Preprint, arXiv:2510.09555.

## A The Necessity of

## Per-Language-Pair Calibration

The trace-divergence measure $D _ { \mathrm { t r a c e } }$ relies on cross-lingual sentence similarities that must be comparable across language pairs. In practice, raw cosine similarities violate this requirement because they conflate semantic relatedness with language-pair-specific properties of the embedding space. These diferences manifest as pairdependent similarity floors and ceilings (reported in Appendix C.4), making raw cosine scores unsuitable for direct comparison. We therefore normalize sentence similarities using pair-specific calibration before constructing the trace-level metrics.

Formally, the structural and order metrics are built from the sentence-level cosine

$$
\begin{array} { c l c r } { { c _ { a , b } ( u , u ^ { \prime } ) = \cos ( \phi ( u ) , \phi ( u ^ { \prime } ) ) , } } \\ { { u \in \mathcal { U } _ { a } , \ u ^ { \prime } \in \mathcal { U } _ { b } , } } \end{array}
$$

where $\phi$ is a multilingual sentence encoder and $\mathcal { U } _ { a }$ denotes the space of sentences appearing in language-a reasoning traces.

Raw cosine similarity is not directly comparable across language pairs because both the expected similarity between unrelated sentences (the floor) and the attainable similarity between semantically equivalent sentences (the ceiling) vary across languages. These shifts arise from anisotropy and uneven cross-lingual alignment in multilingual embedding spaces (Feng et al., 2022; Chen et al., 2024). We define

$$
\begin{array} { r l } & { \mathrm { ~ f l o o r } _ { a , b } = \mathbb E [ c _ { a , b } ( \boldsymbol { u } , \boldsymbol { u } ^ { \prime } ) \ | \ \boldsymbol { u } \perp \boldsymbol { u } ^ { \prime } ] , } \\ & { \mathrm { c e i l i n g } _ { a , b } = \mathbb E [ c _ { a , b } ( \boldsymbol { u } , T _ { a  b } ( \boldsymbol { u } ) ) ] , } \end{array}
$$

where $u \perp u ^ { \prime }$ denotes semantic independence and $T _ { a  b } : \mathcal { U } _ { a }  \mathcal { U } _ { b }$ is a semantics-preserving translator.

The calibrated similarity is then

$$
\widetilde { c } _ { a , b } ( \boldsymbol { u } , \boldsymbol { u } ^ { \prime } ) = \frac { c _ { a , b } ( \boldsymbol { u } , \boldsymbol { u } ^ { \prime } ) - \mathrm { f o o r } _ { a , b } } { \mathrm { c e i l i n g } _ { a , b } - \mathrm { f o o r } _ { a , b } } \in [ 0 , 1 ] ,
$$

After calibration, sentence similarities depend primarily on semantic relatedness rather than language-pair-specific embedding geometry, enabling meaningful comparison and aggregation across multilingual reasoning traces.

## B Supplementary Method Details

## B.1 Step-Parsing Details

Each response is segmented using the following ordered fallback hierarchy:

1. the prompted English line-initial pattern Step n:;

2. line-initial equivalents of the step marker in the seven evaluated languages;

3. a plain line-initial numbered-list pattern; and

4. paragraph boundaries.

Parsing terminates when the final-answer marker is encountered. Empty blocks are discarded, and each remaining step is divided into sentences using a Unicode-aware regular expression. The same ordered rule set is applied in every language, and the parser records which tier was used for each response. No LLM or language-specific semantic parser is invoked.

We distinguish two notions of parser quality. Coverage asks whether a response yields a non-empty sequence of steps. Boundary $f _ { - }$ delity asks whether the extracted step count agrees with the explicit step markers in the raw output. The corresponding diagnostics by language, model, and dataset, together with a manual spot check of the recovered boundaries, are reported in Appendix B.1.1.

## B.1.1 Parser Validation

The automatic marker-count comparison evaluates whether the implementation faithfully recovers the boundaries explicitly emitted by the model. It does not establish that those modelprovided boundaries constitute the uniquely correct semantic segmentation of the trace. We therefore also read the parsed output manually, inspecting roughly 20 randomly drawn traces per model–dataset cell against the corresponding raw responses. In every trace inspected, the recovered boundaries matched the segmentation a reader would impose, and we found no case of a step being merged or split against the evident structure of the trace. This audit is small: manual boundary inspection is costly, and we did not extend it to a sample large enough to bound the residual error rate. We report it as a spot check supporting the automatic diagnostics rather than as an independent measurement of boundary fidelity.

## B.2 Calibration Details and Robustness

## B.2.1 Floor and Ceiling Estimators

The calibration interval of §3.2 is estimated independently for each language pair (a, b).

Let $\mathcal { U } _ { a }$ denote the sentences appearing in language-a traces, and let $q ( u )$ identify the source question from which sentence u was extracted. We estimate the unrelated-sentence floor using pairs sampled from distinct questions:

$$
\widehat { \mathrm { f l o o r } } _ { a , b } = \frac { 1 } { | \mathcal { P } _ { a , b } | } \sum _ { ( u , u ^ { \prime } ) \in \mathcal { P } _ { a , b } } c _ { a , b } ( u , u ^ { \prime } ) ,\tag{11}
$$

where

$$
c _ { a , b } ( u , u ^ { \prime } ) = \cos ( \phi ( u ) , \phi ( u ^ { \prime } ) )
$$

and ϕ is a multilingual sentence encoder.

To estimate the attainable similarity between semantically equivalent sentences, we sample $\smash { \ v { V } _ { a } } \subseteq \boldsymbol { U } _ { a }$ and $\mathcal { V } _ { b } \subseteq \mathcal { U } _ { b }$ from traces whose final answers are verified as correct. Let $T _ { a  b }$ and $T _ { b  a }$ denote an LLM translator in the two directions. We pool the resulting self-translation similarities:

$$
\begin{array} { r l } & { \mathcal { C } _ { a , b } ^ { \mathrm { t r a n s } } = \{ c _ { a , b } ( \boldsymbol { u } , T _ { a  b } ( \boldsymbol { u } ) ) : \boldsymbol { u } \in \mathcal { V } _ { a } \} } \\ & { \quad \quad \cup \big \{ c _ { a , b } ( T _ { b  a } ( \boldsymbol { v } ) , \boldsymbol { v } ) : \boldsymbol { v } \in \mathcal { V } _ { b } \} . } \end{array}\tag{12}
$$

We define the empirical ceiling as

$$
\begin{array} { r } { \widehat { \mathrm { c e i l i n g } } _ { a , b } = \mathrm { c l i p } ( \operatorname { p e r c } _ { 9 0 } ( \mathcal { C } _ { a , b } ^ { \mathrm { t r a n s } } ) , \ } \\ { \widehat { \mathrm { H o o r } } _ { a , b } + \epsilon _ { \mathrm { c a l } } , \ c _ { \mathrm { m a x } } ) , } \end{array}\tag{13}
$$

where $\epsilon _ { \mathrm { c a l } } = 0 . 1 0$ and $c _ { \operatorname* { m a x } } = 0 . 9 8$ . Each raw sentence cosine is then rescaled by Eq. (4).

## B.2.2 Motivation for the Calibration Constants

The empirical ceiling in Eq. (13) uses the 90th percentile of bidirectional self-translation similarities rather than the maximum. The percentile reduces sensitivity to occasional nearduplicate or unusually literal translations that would otherwise compress the calibrated dynamic range.

The minimum margin

$$
\epsilon _ { \mathrm { c a l } } = 0 . 1 0
$$

prevents a nearly degenerate calibration interval when the estimated floor and ceiling are too close. The ceiling cap

$$
c _ { \operatorname* { m a x } } = 0 . 9 8
$$

prevents isolated near-identity translations from setting an unrealistically high endpoint.

The floor and ceiling are estimated independently for each language pair. All remaining constants—the ceiling percentile, ϵ<sub>cal</sub>, c<sub>max</sub>, and the sharpening power used in Eq. (6)—were fixed before evaluating the reported trace-level results and were held identical across all model– dataset evaluations. They were not selected against S, τ, H, or the top-quartile rank statistic.

## B.2.3 What Calibration Does and Does Not Establish

The floor anchors the calibrated scale at an empirical neutral level between unrelated content. The ceiling anchors it at an empirical estimate of the encoder’s attainable similarity between semantically equivalent content for the same language pair. Calibration therefore removes pair-dependent drift at these two endpoints.

This procedure does not establish that an intermediate calibrated value has exactly the same semantic interpretation across diferent language pairs. Translation quality and embedding geometry may vary nonlinearly between the estimated endpoints. Accordingly, pooled absolute values should be interpreted descriptively.

The paper’s central comparison is less demanding: examples are ranked within a fixed model–dataset–language-pair cell. For a fixed embedding space, any strictly monotone pairspecific rescaling preserves this ordering except where clipping creates ties. The rank-based comparison therefore does not require intermediate calibrated values to be globally semantically identical.

## B.2.4 Rank Invariance of the Headline Statistic

The rank comparison reported in §5.2 is the share of pair–question instances in the top quartile of H that do not fall in the top quartile of S, with both thresholds taken within a single evaluation and language pair. Within such a cell the calibration of Eq. (4) is a fixed afine rescaling followed by clipping to [0, 1]. It is monotone, so it preserves the ordering of instances in the cell and leaves the quantile thresholds at the same instances; the greedy alignment of §3.3 selects matches by argmax and is likewise invariant. The only way calibration could move the statistic is by clipping distinct values onto a shared endpoint and creating ties. H is an uncalibrated cosine, so any movement would fall entirely on S.

The residual tie efect is nil in practice. Recomputing the statistic on the uncalibrated scale—recovered by inverting the stored perpair floor and ceiling—reproduces the calibrated value on all three Claude Opus 4.6 evaluations, with every diference below 10<sup>−4</sup> (Table 5). All three reproduced values lie inside the 52.1%–67.3% range reported in §5.2.

Table 5: Headline rank statistic recomputed without calibration. “top- $H ^ { \dprime }$ is the number of instances in the top quartile of H within their language pair; the reported value is the share of those not in the top quartile of S.
<table><tr><td>Dataset</td><td>P</td><td>top-H</td><td>calibrated</td><td> $\Delta$ </td></tr><tr><td>STRATEGYQA</td><td>9,039</td><td>2,267</td><td>0.5554</td><td>0.0000</td></tr><tr><td>SCIBENCH</td><td>7,062</td><td>1,772</td><td>0.6552</td><td>0.0000</td></tr><tr><td>MMLU-PROX</td><td>8,470</td><td>2,125</td><td>0.5906</td><td>0.0000</td></tr></table>

Pooling the quartile thresholds across the 21 language pairs of an evaluation, rather than computing them within each pair, defines a diferent statistic: it folds between-pair diferences in level into a comparison that is otherwise made within a pair. Under that definition the calibrated and uncalibrated values difer by +1.0, −0.4, and −1.5 percentage points on StrategyQA, SciBench, and MMLU-ProX respectively—small, and inconsistent in sign.

Calibration is therefore not what produces the reported divergence. What it provides is a readable scale: within a language pair, 0 marks the unrelated-pair floor and 1 the attainable self-translation ceiling, and S and H are placed on a common footing so that the $H { - } S$ gap is a well-defined quantity at all. Both are statements about levels, and the rank comparison the paper relies on does not depend on either.

## B.2.5 Ceiling-Percentile Sensitivity

We assess the stability of the estimated ceilings by varying the self-translation percentile while holding the sampled translation pool fixed. This analysis measures sensitivity of the endpoint estimator rather than sensitivity to a change in encoder.

## B.2.6 Translator Independence

The ceiling translator can, in some evaluations, belong to the same broad model family as the model whose traces are being analyzed. We test whether this creates a family-specific calibration artifact by re-estimating the languagepair ceilings with an independent-family translator and recomputing the metrics; per-pair values and rank correlations are reported in Appendix D.1. The experiment tests the dependence of the estimated calibration endpoints on the translation model. It does not imply that machine translation is equally reliable for every language, particularly for lower-resource languages.

## B.3 Sharpening Sensitivity

The soft assignment sharpens step-pair similarities as $\widetilde { \mathbf { A } } _ { i , j } \ = \ \mathbf { A } _ { i , j } ^ { p }$ with $p = 3$ , concentrating each source step’s assignment mass on its strongest candidate matches while retaining uncertainty over multiple candidates. We compare this default against a power family, $p ~ \in ~ \{ 1 , 2 , 3 , 5 , 1 0 \}$ , and a softmaxtemperature family, $P _ { i , j } \propto \exp ( \mathbf { A } _ { i , j } / T )$ with $T \in \{ 0 . 0 5 , 0 . 1 , 0 . 2 , 0 . 5 , 1 . 0 \}$ . Because diferent schemes distribute probability mass diferently, their absolute τ values need not coincide, so robustness is assessed through preservation of rankings across evaluation cells. Full results are reported in Appendix D.8.

## B.4 Alternative No-Tie Order Score

The main paper reports the tie-aware order score from Eq. (9). For completeness, we also define a variant that conditions only on concordant and discordant mass:

$$
\tau _ { \mathrm { n o - t i e } } ^ { a  b } = \frac { M _ { \mathrm { c o n } } ^ { a  b } - M _ { \mathrm { d i s } } ^ { a  b } } { M _ { \mathrm { c o n } } ^ { a  b } + M _ { \mathrm { d i s } } ^ { a  b } } .\tag{14}
$$

The corresponding symmetrized score is

$$
\tau _ { \mathrm { n o - t i e } } ( r _ { a } , r _ { b } ) = \frac { 1 } { 2 } ( \tau _ { \mathrm { n o - t i e } } ^ { a  b } + \tau _ { \mathrm { n o - t i e } } ^ { b  a } ) .\tag{15}
$$

The no-tie score measures relative ordering conditional on the two source steps being assigned to distinct target positions. The tieaware score additionally penalizes concentration of multiple source steps on the same target step and is therefore used as the primary metric.

Both directed variants are undefined when their denominators are zero. A symmetrized score is reported as undefined if either direction is undefined.

## B.5 Controlled Metric Checks

We use controlled perturbations to verify that the metrics respond to their intended dimensions. Removing a reasoning step should reduce structural similarity, because one trace no longer covers all content in the other. Rewriting a step without altering its semantic content should preserve structural similarity, subject to the multilingual encoder’s robustness. Dividing one reasoning move into multiple steps, or combining adjacent moves, should be tolerated by the many-to-one correspondences in the structural score. Permuting otherwise unchanged steps should leave structural similarity relatively stable while reducing order consistency. Appendix D.7 reports S, τ, and H for each perturbation family. These checks establish dimension-specific metric behavior under known transformations; they complement, but do not replace, validation on naturally generated multilingual traces.

## B.6 Whole-Answer Holistic Similarity

The holistic baseline embeds each complete visible reasoning trace as a single sequence. Let $\psi$ denote the whole-response encoder. We define

$$
H ( r _ { a } , r _ { b } ) = \cos ( \psi ( r _ { a } ) , \psi ( r _ { b } ) ) .\tag{16}
$$

Unlike $S ,$ the holistic baseline does not align individual steps or identify where two traces difer. Unlike $\tau ,$ it does not explicitly measure preservation of step order. It serves as a comparison against the common practice of representing an entire response with a single embedding.

Why BGE-M3 for H and LaBSE for S. Step-level structural similarity S uses LaBSE (Feng et al., 2022) because it is the standard cross-lingual sentence encoder and individual reasoning steps comfortably fit its 256-token context. The whole-trace baseline H instead uses BGE-M3 (Chen et al., 2024) because LaBSE’s 256-token limit truncates a substantial fraction of our reasoning traces (especially on SciBench and StrategyQA, where a single verified trace frequently exceeds 1,000 tokens), and in preliminary tests the resulting LaBSE-on-truncated-trace cosines were uninformative as a holistic signal. BGE-M3’s 8K-token window covers the entire trace without truncation in our evaluations, which is the property a holistic baseline is meant to provide. Adopting LaBSE on both sides would therefore not isolate the aggregation efect from $H { - } S ;$ it would replace the encoder asymmetry with a truncation artifact whose magnitude depends on dataset-specific trace-length distributions. We accordingly do not interpret the absolute magnitude of $H { - } S$ in Table 1 as a calibrated distance, and rely on the scale-free rank comparison in Figure 1 for the qualitative claim that holistic similarity overstates structural consistency.

## B.7 Interpretation of Structural and Order Scores

The two primary trace-level metrics capture complementary properties.

High S, high τ . The traces contain similar step-level content in a similar order.

High S, low $\tau .$ . The traces contain similar reasoning moves but arrange them diferently.

Low S, high τ. The subset of content that can be aligned appears in a consistent order, but substantial content is absent or unmatched.

Low S, low τ. The traces difer both in step-level content and in the ordering of their matched components.

Neither metric determines whether a diference is harmful. They identify and decompose observable diferences in the visible explanatory artifacts, which can then be inspected by human or automated evaluators.

## B.8 Relation to Prior Step-Level Metrics

We compare against prior step-level metrics to clarify which parts of the framework are reused and which are specific to cross-lingual reasoning. The method recomposes BERTScorestyle greedy matching (Zhang et al., 2020), ROSCOE-style step alignment (Golovneva et al., 2023), multilingual sentence embeddings (Feng et al., 2022), and soft ordering ideas from continuous sorting (Mena et al., 2018; Grover et al., 2019).

The main diference is that our alignment is calibrated per language pair before aggregation. Prior step metrics typically use raw within-language cosine scores, while crosslingual traces require a pair-specific unrelatedsentence floor and self-translation ceiling so that a score for, for example, en-es has the same interpretation as a score for hi-zh.

The second diference is that we report structure and order separately. A single alignment score can hide whether two traces contain diferent reasoning moves or contain similar moves in a diferent sequence; S and τ are kept separate so these two kinds of divergence remain visible.

## C Supplementary Experimental Details

This appendix collects the experimental details that support the three main results without expanding Section 5. We place the dataset, model, verification, accuracy-audit, and calibration materials next to the discussion they support.

## C.1 Datasets, Models, and Inference

All reported evaluations use the same seven languages: Arabic (ar), English (en), Spanish (es), Hindi (hi), Russian (ru), Vietnamese (vi), and Chinese (zh). This gives ${ \binom { 7 } { 2 } } \ = \ 2 1$ unordered language pairs for each model–dataset run. We sample 500 problems per run from each benchmark.

MMLU-ProX (Xuan et al., 2025) supplies parallel multiple-choice questions across academic and professional domains. We verify traces by extracting the final option and comparing it with the dataset answer key; ambiguous outputs are checked by an answerconsistency judge that sees the question, reference answer, and model final answer, but not any trace-similarity metric. StrategyQA (Geva et al., 2021) provides yes/no commonsense questions requiring implicit multistep reasoning; we verify the final yes/no answer against the dataset label, with the same answer-consistency judge used for ambiguous outputs. SciBench (Wang et al., 2024) contains open-ended STEM problems with fixed numeric or symbolic answers; the verification step checks whether the model final answer is equivalent to the reference answer before the trace is used for divergence analysis.

The model suite comprises Claude Opus 4.6, Claude Haiku 4.5, DeepSeek-V3.2, GPT-OSS-120B, Qwen3-235B, and Qwen3-32B. All reported runs use greedy decoding.

Retention cascade. For each 500-problem dataset–model run, we first verify each language-specific final answer against the reference. For every language whose answer is correct on a given question, an LLM trace judge then inspects the reasoning trace and emits VALID or INVALID. If any answer-correct trace for that question is judged INVALID, the question is dropped from trace aggregation, even if other answer-correct languages would otherwise form valid pairs. The surviving pair–question instances enter S, τ, and $H ;$ one-languagecorrect cases are reserved for $\Delta _ { \mathrm { a c c } }$ Table 1 reports the retained question counts Q and pair–question counts P after this cascade. Metric settings. Step-level metrics use LaBSE (Feng et al., 2022), the holistic baseline H uses BGE-M3 (Chen et al., 2024), unrelatedpair floor sampling draws up to $1 0 ^ { 4 }$ sentence pairs per language pair, the self-translation ceiling samples up to 500 sentences per direction, and the soft Kendall-τ sharpening power is 3. Appendix D.8 shows that the cross-evaluation τ¯ ranking is preserved with Spearman $\geq 0 . 8 5$ across other powers and a softmax-temperature baseline. Aggregate means in Table 1 use 95% percentile bootstrap intervals with 2,000 problem-level resamples, jointly resampling the 21 language-pair observations for each question. Compute. All LLM trace generation and ceiling-translation calls use hosted APIs (AWS Bedrock for Claude, DeepSeek, GPT-OSS, and Qwen endpoints); we therefore report no local GPU hours for inference. Sentence-level embedding, per-pair calibration, and (S, τ, H) computation for the full 18 model–dataset evaluations run on a single NVIDIA A100 and take roughly 25 hours end-to-end, dominated by LaBSE and BGE-M3 encoding of the retained traces.

## C.2 Prompt Templates

Because the prompt elicits the step structure on which the parser and both step-level metrics depend, we give the templates in full. All inference uses a single fixed template family across every model, dataset, and language; no prompt is tuned per language. Decoding is greedy (temperature 0) with an identical token budget and retry policy across all runs.

Shared output-format block. Appended to every system prompt:

System prompt: shared output-format block   
OUTPUT FORMAT REQUIREMENTS:   
- Write the reasoning as a numbered   
sequence of steps.   
- Every reasoning line must begin with an   
English step header in exactly this   
format: Step 1:, Step 2:, Step 3:, ...   
- Keep the step labels exactly as   
\`Step N:\` even when the reasoning   
content is in another language.   
- Put only one logical step under each   
step header; multi-line steps are   
allowed when needed.   
- End with exactly one final-answer line   
in this format:   
Final Answer: [your answer here]   
- Do not wrap the final answer in a   
sentence.   
- Do not omit the step headers.

Task-specific system prompts. The preamble difers only in role and emphasis. MMLU-ProX: “You are an expert problem solver. Answer the following multiple-choice question,” followed by four instructions—analyze the options systematically; explain your reasoning step by step; end with the single best option letter; follow the output format exactly—then the block above, then a note that the final answer must be a single option label such as A, B, C, or D. SciBench: “You are an expert mathematician and scientist. Solve the given problem carefully,” followed by five instructions—show your reasoning step by step; perform calculations explicitly and do not skip important intermediate steps; if the problem has sub-parts, solve them in a clean logical order; be concise but complete; follow the output format exactly—then the block above. StrategyQA: “You are an expert at commonsense and multi-hop reasoning. Answer the given yes/no question,” followed by six instructions—identify the implicit sub-questions that must be resolved; establish the relevant facts for each, one sub-question per step; combine the established facts explicitly into a single conclusion; rely on general world knowledge; be concise but complete; follow the output format exactly—then the block above, then a note that the final answer must be exactly Yes or No, written in English, with no additional words.

User message   
MMLU-ProX:   
Question:\n{question}\n\n   
Options:\n{A. ...}\n{B. ...}   
SciBench:   
Problem (unit: {unit}):\n{question}   
StrategyQA:   
Question:\n{question}

Language-specific instructions. English receives none. Every non-English language appends exactly one sentence to the system prompt:

Language-specific instruction (non-English)   
IMPORTANT: Write the content of every   
reasoning step and the final answer in   
{Language}. Keep the step headers exactly   
as \`Step 1:\`, \`Step 2:\`, etc., and keep   
\`Final Answer:\` in English.

where {Language} is the English language name. For SciBench items already supplied in the target language, the wording instead notes that the problem is already in that language and asks for the reasoning content and final answer in it. For StrategyQA, the sentence requests the reasoning content in the target language with the answer token as the English word Yes or No, which restores rule-based answer checking in all seven languages. This constraint touches only the final-answer line, at which the step parser terminates, so S and τ are computed on identical step content either way.

Efect on cross-lingual comparison. Prompting shapes granularity and parser compliance, and the design controls for this in the way that matters here: the segmentation cue is an English literal held constant across all seven languages. Step boundaries are therefore never recovered by language-specific heuristics, and diferences in S or τ across languages cannot arise from difering header conventions. The parser audit in Appendix D.5 bears this out, with efectively complete coverage in every language.

## C.3 Outcome-Level Accuracy Audit

Figure 2 reports per-language final-answer accuracy for the same model–dataset evaluations summarized in Table 1. This audit contextualizes the correctness filter used before trace-level aggregation: accuracy varies across languages and datasets, but the main divergence metrics are computed only after both compared traces pass answer verification.

## C.4 Calibration Intervals

We report calibration ranges to show how much the raw multilingual cosine scale varies before pair-specific normalization. Table 6 summarises the sentence-level intervals used in the main analyses: StrategyQA has the largest floor spread, while SciBench and MMLU-ProX often hit the ceiling cap.

![](images/e57b0cffb6ab604f7fceed964a0b28fd4eeb1659c9d72b79c76aa43599fd04dc.jpg)  
Figure 2: Outcome-level accuracy heatmap across the multilingual evaluations. Within each panel, columns are ar, en, es, hi, ru, vi, zh, mean, and range. Cell values are computed over the full 500-problem inference set for each (model, dataset) pair, prior to the trace-validity filter that defines Q in Table 1. The figure audits final-answer performance, while Section 5 focuses on trace divergence after correctness filtering.

Table 6: Sentence-level calibration intervals. Ranges are over language pairs in the corresponding evaluation. Ceiling values are capped at $c _ { \operatorname* { m a x } } = 0 . 9 8$ , so datasets whose self-translation similarities reach the cap can have zero or near-zero ceiling spread.
<table><tr><td>Evaluation</td><td>floor range</td><td>floor spread</td><td>ceiling range</td><td>ceiling spread</td></tr><tr><td>MMLU-PRoX, Claude Opus 4.6</td><td>0.187-0.331</td><td>0.144</td><td>0.960–0.980</td><td>0.020</td></tr><tr><td>STRATEGYQA, Claude Opus 4.6</td><td>0.157-0.381</td><td>0.224</td><td>0.912-0.968</td><td>0.056</td></tr><tr><td>SciBENCH, Claude Opus 4.6</td><td>0.292–0.352</td><td>0.059</td><td>0.980–0.980</td><td>0.000</td></tr></table>

## D Additional Results

## D.1 Translator-Independence Probe

Methodological concern. The default ceiling estimation in §3.2 uses an LLM translator from the same family (Claude Opus 4.6) as several of the evaluated models. A strong translator whose idiolect matches the evaluated model could inflate ceilings on exactly those pairs where the model is already consistent, partially undoing the calibration. We probe this co-dependence by re-estimating the per-pair sentence ceiling with an independent translator from a diferent model family and recomputing the divergence metrics on the same retained pair–question instances.

Setup. The probe uses StrategyQA on Claude Opus 4.6 $\begin{array} { r l r } { ( n } & { { } = } & { 4 5 1 } \end{array}$ retained verified-correct questions over 21 language pairs). The independent translator is Mistral-Large-3-675B-Instruct. The unrelated-pair sentence floor is shared between the two analyses (the floor does not depend on the translator); only the ceiling estimation changes. All other settings (LaBSE step embeddings, BGE-M3 holistic embedding, 90thpercentile self-translation ceiling, $\epsilon _ { \mathrm { c a l } } = 0 . 1 0 $ $c _ { \operatorname* { m a x } } = 0 . 9 8$ , sharpening power 3) are identical. Result. Per-pair rankings of the three metrics are stable under the alternative ceiling. Across

21 language pairs, the Spearman rank correlations between the original (Claude-Opus ceiling) and the independent-translator (Mistral-Large-3 ceiling) estimates are $\rho _ { C } ~ = ~ + 0 . 9 2$ on the ceiling itself, $\rho _ { \bar { S } } ~ = ~ + 0 . 8 9$ on mean structural similarity, $\rho _ { \bar { \tau } } ~ = ~ + 0 . 8 7$ on mean soft Kendall-τ, and $\rho _ { \bar { H } } = + 0 . 9 9$ on the holistic baseline (sanity: H does not use the ceiling and should be near-invariant). Pooled aggregate means change by $\Delta \bar { S } = + 0 . 0 0 2$ and $\Delta \bar { \tau } = - 0 . 0 0 3$ . Per-pair values are reported in Table 7.

Table 7: Translator-independence probe on StrategyQA Claude Opus 4.6 $( n = 4 5 1 , 2 1 \ \mathrm { p a i r s } )$ . C is the per-pair sentence ceiling. ∆S<sup>¯</sup> and ∆τ¯ are the (Mistral − Opus) shifts in mean structural similarity and mean soft Kendall-τ with-tie. H does not depend on the ceiling and is unchanged.
<table><tr><td>pair</td><td> $C _ { \mathrm { O p u s } }$ </td><td> $C _ { \mathrm { M i s t } }$ </td><td> $\Delta C$ </td><td>∆S</td><td> $\Delta \bar { \tau }$ </td></tr><tr><td>ar-en</td><td>0.943</td><td>0.941</td><td>-0.002</td><td>+0.001</td><td>-0.013</td></tr><tr><td>ar-es</td><td>0.960</td><td>0.957</td><td>-0.003</td><td>+0.002</td><td>+0.003</td></tr><tr><td>ar-hi</td><td>0.935</td><td>0.934</td><td>-0.001</td><td>+0.002</td><td>+0.004</td></tr><tr><td>ar-ru</td><td>0.943</td><td>0.935</td><td>-0.008</td><td>+0.006</td><td>-0.005</td></tr><tr><td>ar-vi</td><td>0.922</td><td>0.923</td><td>+0.001</td><td>-0.000</td><td>-0.005</td></tr><tr><td>ar-zh</td><td>0.950</td><td>0.947</td><td>-0.003</td><td>+0.004</td><td>-0.008</td></tr><tr><td>en-es</td><td>0.962</td><td>0.962</td><td>-0.000</td><td>+0.000</td><td>+0.017</td></tr><tr><td>en-hi</td><td>0.929</td><td>0.931</td><td>+0.003</td><td>-0.001</td><td>+0.012</td></tr><tr><td>en-ru</td><td>0.934</td><td>0.941</td><td>+0.008</td><td>-0.005</td><td>+0.007</td></tr><tr><td>en-vi</td><td>0.936</td><td>0.936</td><td>+0.001</td><td>-0.001</td><td>+0.011</td></tr><tr><td>en-zh</td><td>0.933</td><td>0.941</td><td>+0.007</td><td>-0.006</td><td>+0.009</td></tr><tr><td>es-hi</td><td>0.952</td><td>0.958</td><td>+0.006</td><td>+0.002</td><td>-0.007</td></tr><tr><td>es-ru</td><td>0.968</td><td>0.970</td><td>+0.002</td><td>-0.003</td><td>-0.005</td></tr><tr><td>es-vi</td><td>0.962</td><td>0.958</td><td>-0.004</td><td>+0.002</td><td>-0.007</td></tr><tr><td>es-zh</td><td>0.945</td><td>0.941</td><td>-0.004</td><td>+0.003</td><td>-0.006</td></tr><tr><td>hi-ru</td><td>0.963</td><td>0.955</td><td>-0.008</td><td>+0.019</td><td>-0.032</td></tr><tr><td>hi-vi</td><td>0.968</td><td>0.963</td><td>-0.005</td><td>+0.023</td><td>-0.036</td></tr><tr><td>hi-zh</td><td>0.912</td><td>0.925</td><td>+0.013</td><td>-0.005</td><td>-0.021</td></tr><tr><td>ru-vi</td><td>0.960</td><td>0.959</td><td>-0.001</td><td>+0.001</td><td>-0.001</td></tr><tr><td>ru-zh</td><td>0.926</td><td>0.917</td><td>-0.009</td><td>+0.004</td><td>+0.007</td></tr><tr><td>vi-zh</td><td>0.924</td><td>0.933</td><td>+0.009</td><td>-0.013</td><td>+0.012</td></tr><tr><td>mean</td><td>0.944</td><td>0.944</td><td>+0.000</td><td>+0.002</td><td>-0.003</td></tr></table>

Interpretation. Per-pair ceiling changes are bounded by $| \Delta C | \le \ 0 . 0 1 3$ and pooled-mean shifts are within ±0.003 for both S<sup>¯</sup> and τ¯. The largest individual shifts on the downstream metrics fall on Hindi-involving pairs: the top three $| \Delta \bar { \tau } |$ shifts are hi-vi (0.036), hi-ru (0.032), and hi-zh (0.021), and the top two $| \Delta \bar { S } |$ shifts are hi-vi (0.023) and hi-ru (0.019); on the ceiling itself, |∆C| is largest for hi-zh (+0.013), with ru-zh, vi-zh, ar-ru, en-ru, and hi-ru clustered between 0.008 and 0.009, so $| \Delta C |$ alone does not single out Hindi. Together these patterns suggest that ceiling estimation for Hindi-side directions is the most translatordependent of the seven languages tested, with the bidirectional soft Kendall-τ amplifying this sensitivity relative to the calibrated structural score. Relative pair rankings nonetheless remain stable $( \rho _ { \bar { S } } = + 0 . 8 9 , \rho _ { \bar { \tau } } = + 0 . 8 7 )$ . The headline conclusion—H systematically exceeds $\bar { S }$ on verified-correct trace pairs—does not depend on the choice of ceiling translator, and the per-pair calibration findings are therefore not an artefact of using a Claude-family translator on Claude-family traces.

## D.2 Pair-Level Result Tables

We keep pair-level values in the appendix so the main results can focus on aggregate hypotheses. Table 8 gives the Claude Opus 4.6 pair-level views for all three evaluated datasets.

The pair-level rankings show that languagepair dificulty is not fixed across tasks. English– Spanish and Spanish–Russian are consistently among the strongest pairs, but Hindi- and Chinese-involving pairs shift more substantially across datasets, reinforcing the main claim that trace divergence reflects model– dataset–language interactions rather than a single language-pair ordering.

## D.3 Cross-Dataset and Cross-Model Patterns

We compare shared language pairs across datasets to check whether the aggregate conclusion depends on a single corpus. Table 9 shows that H rankings replicate strongly across datasets, τ rankings replicate moderately, and S is more task-sensitive, as expected for a metric afected by how each dataset encourages step granularity.

Beyond rank replication, the datasets difer in the type of divergence they expose. Averaging over the six tested models, MMLU-ProX has the lowest mean structural similarity $( \bar { S } = 0 . 5 3 6 )$ but the highest mean order consistency of the three $( \bar { \tau } = 0 . 3 9 9 )$ , suggesting that multiple-choice academic questions often preserve a broad procedural order while still varying the content or granularity of intermediate steps. SciBench has the highest mean structural similarity $( \bar { S } ~ = ~ 0 . 5 7 1 )$ but lower mean order consistency $( \bar { \tau } \ : = \ : 0 . 3 7 3 )$ , consistent with quantitative STEM solutions that reuse similar formulas and quantities while allowing languages to reorder setup, substitution, conversion, and rounding. StrategyQA sits between them structurally $( \bar { S } \ : = \ : 0 . 5 5 1 )$ but has the highest holistic similarity $( \bar { H } = 0 . 8 1 3 )$ and the largest mean holistic-to-structural gap $( \overline { { H - S } } = 0 . 2 6 3 )$ , indicating that yes/no commonsense answers can sound semantically similar even when premise interpretation or evidence selection changes inside the trace.

Table 8: Claude Opus 4.6 pair-level divergence across datasets. Each dataset block is independently sorted by structural similarity S in descending order. N is the number of problems on which both languages produced an answer-verified response and both reasoning traces were marked VALID by the LLM trace-judge in Appendix C.1 (the same cascade used for Q in Table 1); it is therefore typically smaller than the count of problems on which both languages are merely answer-correct. $\Delta _ { \mathrm { a c c } }$ reports pair-level one-language-correct disagreement for MMLU-ProX.
<table><tr><td rowspan=1 colspan=16>MMLU-PROX                      STRATEGYQA                    SCIBENCH</td></tr><tr><td rowspan=1 colspan=11>pair  N $\Delta _ { \mathrm { a c c } }$      S    T    Hpair   N    S     T    H</td><td rowspan=1 colspan=5>pair  N    S    T    H</td></tr><tr><td rowspan=1 colspan=3>es-ru 4084.2%</td><td rowspan=1 colspan=3>0.6100.4510.819</td><td rowspan=1 colspan=1>es-ru</td><td rowspan=1 colspan=2>4300.625</td><td rowspan=1 colspan=2>0.4620.836</td><td rowspan=1 colspan=5>en-es3380.6690.4920.863</td></tr><tr><td rowspan=1 colspan=2>en-es 411</td><td rowspan=1 colspan=1>3.8%</td><td rowspan=1 colspan=1>0.604</td><td rowspan=1 colspan=1>0.449</td><td rowspan=1 colspan=1>0.859</td><td rowspan=1 colspan=1>ar-hi</td><td rowspan=1 colspan=2>4340.622</td><td rowspan=1 colspan=2>0.4190.798</td><td rowspan=1 colspan=1>es-ru</td><td rowspan=1 colspan=3>3360.6450.483</td><td rowspan=1 colspan=1>0.812</td></tr><tr><td rowspan=1 colspan=2>ru-vi 409</td><td rowspan=1 colspan=1>4.8%</td><td rowspan=1 colspan=1>0.585</td><td rowspan=1 colspan=1>0.429</td><td rowspan=1 colspan=1>0.799</td><td rowspan=1 colspan=1>ar-zh</td><td rowspan=1 colspan=2>4300.618</td><td rowspan=1 colspan=1>0.420</td><td rowspan=1 colspan=1>0.775</td><td rowspan=1 colspan=1>es-vi</td><td rowspan=1 colspan=3>3360.6380.471</td><td rowspan=1 colspan=1>0.784</td></tr><tr><td rowspan=1 colspan=1>ar-es</td><td rowspan=1 colspan=1>409</td><td rowspan=1 colspan=1>3.2%</td><td rowspan=1 colspan=1>0.583</td><td rowspan=1 colspan=1>0.423</td><td rowspan=1 colspan=1>0.789</td><td rowspan=1 colspan=1>vi-zh</td><td rowspan=1 colspan=2>4310.614</td><td rowspan=1 colspan=1>0.410</td><td rowspan=1 colspan=1>0.813</td><td rowspan=1 colspan=1>ru-vi</td><td rowspan=1 colspan=2>3340.625</td><td rowspan=1 colspan=1>0.464</td><td rowspan=1 colspan=1>0.778</td></tr><tr><td rowspan=1 colspan=1>en-zh</td><td rowspan=1 colspan=1>400</td><td rowspan=1 colspan=1>6.6%</td><td rowspan=1 colspan=1>0.583</td><td rowspan=1 colspan=1>0.416</td><td rowspan=1 colspan=1>0.813</td><td rowspan=1 colspan=1>en-zh</td><td rowspan=1 colspan=2>4310.613</td><td rowspan=1 colspan=1>0.407</td><td rowspan=1 colspan=1>0.820</td><td rowspan=1 colspan=1>en-vi</td><td rowspan=1 colspan=1>335</td><td rowspan=1 colspan=1>0.621</td><td rowspan=1 colspan=1>0.441</td><td rowspan=1 colspan=1>0.801</td></tr><tr><td rowspan=1 colspan=1>es-zh</td><td rowspan=1 colspan=1>401</td><td rowspan=1 colspan=1>4.4%</td><td rowspan=1 colspan=1>0.583</td><td rowspan=1 colspan=1>0.425</td><td rowspan=1 colspan=1>0.787</td><td rowspan=1 colspan=1>es-zh</td><td rowspan=1 colspan=1>427</td><td rowspan=1 colspan=1>0.611</td><td rowspan=1 colspan=1>0.409</td><td rowspan=1 colspan=1>0.798</td><td rowspan=1 colspan=1>en-ru</td><td rowspan=1 colspan=1>340</td><td rowspan=1 colspan=1>0.620</td><td rowspan=1 colspan=1>0.456</td><td rowspan=1 colspan=1>0.821</td></tr><tr><td rowspan=1 colspan=1>es-vi</td><td rowspan=1 colspan=1>406</td><td rowspan=1 colspan=1>5.4%</td><td rowspan=1 colspan=1>0.581</td><td rowspan=1 colspan=1>0.422</td><td rowspan=1 colspan=1>0.803</td><td rowspan=1 colspan=1>ar-es</td><td rowspan=1 colspan=1>429</td><td rowspan=1 colspan=1>0.611</td><td rowspan=1 colspan=1>0.444</td><td rowspan=1 colspan=1>0.792</td><td rowspan=1 colspan=1>es-hi</td><td rowspan=1 colspan=1>340</td><td rowspan=1 colspan=1>0.610</td><td rowspan=1 colspan=1>0.440</td><td rowspan=1 colspan=1>0.767</td></tr><tr><td rowspan=1 colspan=1>ar-hi</td><td rowspan=1 colspan=1>399</td><td rowspan=1 colspan=1>5.6%</td><td rowspan=1 colspan=1>0.578</td><td rowspan=1 colspan=1>0.416</td><td rowspan=1 colspan=1>0.777</td><td rowspan=1 colspan=1>ru-zh</td><td rowspan=1 colspan=1>431</td><td rowspan=1 colspan=1>0.611</td><td rowspan=1 colspan=1>0.416</td><td rowspan=1 colspan=1>0.801</td><td rowspan=1 colspan=1>vi-zh</td><td rowspan=1 colspan=1>328</td><td rowspan=1 colspan=1>0.609</td><td rowspan=1 colspan=1>0.446</td><td rowspan=1 colspan=1>0.778</td></tr><tr><td rowspan=1 colspan=1>es-hi</td><td rowspan=1 colspan=1>397</td><td rowspan=1 colspan=1>7.2%</td><td rowspan=1 colspan=1>0.572</td><td rowspan=1 colspan=1>0.408</td><td rowspan=1 colspan=1>0.788</td><td rowspan=1 colspan=1>hi-zh</td><td rowspan=1 colspan=1>431</td><td rowspan=1 colspan=1>0.609</td><td rowspan=1 colspan=1>0.411</td><td rowspan=1 colspan=1>0.792</td><td rowspan=1 colspan=1>ar-es</td><td rowspan=1 colspan=1>339</td><td rowspan=1 colspan=1>0.608</td><td rowspan=1 colspan=1>0.451</td><td rowspan=1 colspan=1>0.777</td></tr><tr><td rowspan=1 colspan=1>ar-en</td><td rowspan=1 colspan=1>411</td><td rowspan=1 colspan=1>3.4%</td><td rowspan=1 colspan=1>0.570</td><td rowspan=1 colspan=1>0.417</td><td rowspan=1 colspan=1>0.794</td><td rowspan=1 colspan=1>es-vi</td><td rowspan=1 colspan=1>430</td><td rowspan=1 colspan=1>0.609</td><td rowspan=1 colspan=1>0.436</td><td rowspan=1 colspan=1>0.846</td><td rowspan=1 colspan=1>es-zh</td><td rowspan=1 colspan=1>334</td><td rowspan=1 colspan=1>0.607</td><td rowspan=1 colspan=1>0.450</td><td rowspan=1 colspan=1>0.761</td></tr><tr><td rowspan=1 colspan=1>ar-ru</td><td rowspan=1 colspan=1>407</td><td rowspan=1 colspan=1>4.6%</td><td rowspan=1 colspan=1>0.570</td><td rowspan=1 colspan=1>0.425</td><td rowspan=1 colspan=1>0.792</td><td rowspan=1 colspan=1>ru-vi</td><td rowspan=1 colspan=1>431</td><td rowspan=1 colspan=1>0.608</td><td rowspan=1 colspan=1>0.428</td><td rowspan=1 colspan=1>0.821</td><td rowspan=1 colspan=1>en-zh</td><td rowspan=1 colspan=1>332</td><td rowspan=1 colspan=1>0.606</td><td rowspan=1 colspan=1>0.429</td><td rowspan=1 colspan=1>0.783</td></tr><tr><td rowspan=1 colspan=1>ar-zh</td><td rowspan=1 colspan=1>398</td><td rowspan=1 colspan=1>5.6%</td><td rowspan=1 colspan=1>0.569</td><td rowspan=1 colspan=1>0.400</td><td rowspan=1 colspan=1>0.777</td><td rowspan=1 colspan=1>ar-ru</td><td rowspan=1 colspan=1>432</td><td rowspan=1 colspan=1>0.605</td><td rowspan=1 colspan=1>0.436</td><td rowspan=1 colspan=1>0.802</td><td rowspan=1 colspan=1>hi-vi</td><td rowspan=1 colspan=1>336</td><td rowspan=1 colspan=1>0.606</td><td rowspan=1 colspan=1>0.436</td><td rowspan=1 colspan=1>0.754</td></tr><tr><td rowspan=1 colspan=1>en-ru</td><td rowspan=1 colspan=1>411</td><td rowspan=1 colspan=1>4.4%</td><td rowspan=1 colspan=1>0.569</td><td rowspan=1 colspan=1>0.444</td><td rowspan=1 colspan=1>0.828</td><td rowspan=1 colspan=1>en-es</td><td rowspan=1 colspan=1>429</td><td rowspan=1 colspan=1>0.602</td><td rowspan=1 colspan=1>0.428</td><td rowspan=1 colspan=1>0.882</td><td rowspan=1 colspan=1>ar-hi</td><td rowspan=1 colspan=1>342</td><td rowspan=1 colspan=1>0.602</td><td rowspan=1 colspan=1>0.439</td><td rowspan=1 colspan=1>0.752</td></tr><tr><td rowspan=1 colspan=1>hi-vi</td><td rowspan=1 colspan=1>402</td><td rowspan=1 colspan=1>5.4%</td><td rowspan=1 colspan=1>0.568</td><td rowspan=1 colspan=1>0.399</td><td rowspan=1 colspan=1>0.782</td><td rowspan=1 colspan=1>en-vi</td><td rowspan=1 colspan=1>427</td><td rowspan=1 colspan=1>0.601</td><td rowspan=1 colspan=1>0.440</td><td rowspan=1 colspan=1>0.868</td><td rowspan=1 colspan=1>ar-vi</td><td rowspan=1 colspan=1>337</td><td rowspan=1 colspan=1>0.602</td><td rowspan=1 colspan=1>0.450</td><td rowspan=1 colspan=1>0.763</td></tr><tr><td rowspan=1 colspan=1>hi-ru</td><td rowspan=1 colspan=1>400</td><td rowspan=1 colspan=1>6.6%</td><td rowspan=1 colspan=1>0.566</td><td rowspan=1 colspan=1>0.408</td><td rowspan=1 colspan=1>0.788</td><td rowspan=1 colspan=1>en-ru</td><td rowspan=1 colspan=1>431</td><td rowspan=1 colspan=1>0.593</td><td rowspan=1 colspan=1>0.431</td><td rowspan=1 colspan=1>0.843</td><td rowspan=1 colspan=1>en-hi</td><td rowspan=1 colspan=1>341</td><td rowspan=1 colspan=1>0.598</td><td rowspan=1 colspan=1>0.420</td><td rowspan=1 colspan=1>0.788</td></tr><tr><td rowspan=1 colspan=1>ar-vi</td><td rowspan=1 colspan=1>405</td><td rowspan=1 colspan=1>5.0%</td><td rowspan=1 colspan=1>0.565</td><td rowspan=1 colspan=1>0.414</td><td rowspan=1 colspan=1>0.785</td><td rowspan=1 colspan=1>ar-vi</td><td rowspan=1 colspan=1>430</td><td rowspan=1 colspan=1>0.592</td><td rowspan=1 colspan=1>0.421</td><td rowspan=1 colspan=1>0.790</td><td rowspan=1 colspan=1>ar-en</td><td rowspan=1 colspan=1>338</td><td rowspan=1 colspan=1>0.597</td><td rowspan=1 colspan=1>0.427</td><td rowspan=1 colspan=1>0.783</td></tr><tr><td rowspan=1 colspan=1>ru-zh</td><td rowspan=1 colspan=1>397</td><td rowspan=1 colspan=1>7.0%</td><td rowspan=1 colspan=1>0.560</td><td rowspan=1 colspan=1>0.429</td><td rowspan=1 colspan=1>0.794</td><td rowspan=1 colspan=1>es-hi</td><td rowspan=1 colspan=2>4300.589</td><td rowspan=1 colspan=1>0.421</td><td rowspan=1 colspan=1>0.823</td><td rowspan=1 colspan=1>hi-ru</td><td rowspan=1 colspan=1>339</td><td rowspan=1 colspan=1>0.595</td><td rowspan=1 colspan=1>0.432</td><td rowspan=1 colspan=1>0.761</td></tr><tr><td rowspan=1 colspan=1>vi-zh</td><td rowspan=1 colspan=1>400</td><td rowspan=1 colspan=1>5.8%</td><td rowspan=1 colspan=1>0.555</td><td rowspan=1 colspan=1>0.423</td><td rowspan=1 colspan=1>0.801</td><td rowspan=1 colspan=1>ar-en</td><td rowspan=1 colspan=1>431</td><td rowspan=1 colspan=1>0.577</td><td rowspan=1 colspan=1>0.398</td><td rowspan=1 colspan=1>0.799</td><td rowspan=1 colspan=1>ru-zh</td><td rowspan=1 colspan=1>331</td><td rowspan=1 colspan=1>0.594</td><td rowspan=1 colspan=1>0.444</td><td rowspan=1 colspan=1>0.770</td></tr><tr><td rowspan=1 colspan=1>en-vi</td><td rowspan=1 colspan=1>408</td><td rowspan=1 colspan=1>5.2%</td><td rowspan=1 colspan=1>0.548</td><td rowspan=1 colspan=1>0.421</td><td rowspan=1 colspan=1>0.820</td><td rowspan=1 colspan=1>hi-ru</td><td rowspan=1 colspan=1>431</td><td rowspan=1 colspan=1>0.564</td><td rowspan=1 colspan=1>0.414</td><td rowspan=1 colspan=1>0.822</td><td rowspan=1 colspan=1>ar-zh</td><td rowspan=1 colspan=1>334</td><td rowspan=1 colspan=1>0.594</td><td rowspan=1 colspan=1>0.417</td><td rowspan=1 colspan=1>0.751</td></tr><tr><td rowspan=1 colspan=1>en-hi</td><td rowspan=1 colspan=1>399</td><td rowspan=1 colspan=1>7.0%</td><td rowspan=1 colspan=1>0.546</td><td rowspan=1 colspan=1>0.406</td><td rowspan=1 colspan=1>0.811</td><td rowspan=1 colspan=1>en-hi</td><td rowspan=1 colspan=1>431</td><td rowspan=1 colspan=1>0.561</td><td rowspan=1 colspan=1>0.377</td><td rowspan=1 colspan=1>0.844</td><td rowspan=1 colspan=1>hi-zh</td><td rowspan=1 colspan=1>334</td><td rowspan=1 colspan=1>0.589</td><td rowspan=1 colspan=1>0.425</td><td rowspan=1 colspan=1>0.744</td></tr><tr><td rowspan=1 colspan=1>hi-zh</td><td rowspan=1 colspan=1>392</td><td rowspan=1 colspan=1>7.2%</td><td rowspan=1 colspan=1>0.546</td><td rowspan=1 colspan=1>0.398</td><td rowspan=1 colspan=1>0.772</td><td rowspan=1 colspan=1>hi-vi</td><td rowspan=1 colspan=1>432</td><td rowspan=1 colspan=1>0.545</td><td rowspan=1 colspan=1>0.400</td><td rowspan=1 colspan=1>0.817</td><td rowspan=1 colspan=1>ar-ru</td><td rowspan=1 colspan=1>338</td><td rowspan=1 colspan=1>0.587</td><td rowspan=1 colspan=1>0.427</td><td rowspan=1 colspan=1>0.777</td></tr></table>

Table 9: Cross-dataset replication on the 21 shared language pairs for Claude Opus 4.6. Each cell shows the Spearman rank correlation across language-pair means with a two-sided p-value.
<table><tr><td>metric</td><td>MMLU-PROX/STRATEGYQA</td><td>MMLU-PROX/SCIBENCH</td><td>STRATEGYQA/SCIBENCH</td></tr><tr><td>S</td><td>0.37 (p=0.095)</td><td>0.55 (p=0.010)</td><td>0.11 (p=0.642)</td></tr><tr><td>T</td><td>0.52 (p=0.015)</td><td>0.76 (p&lt;0.001)</td><td>0.62 (p=0.002)</td></tr><tr><td>H</td><td>0.80 (p&lt;0.001)</td><td>0.98 (p&lt;0.001)</td><td>0.78 (p&lt;0.001)</td></tr></table>

Model comparisons within each dataset show that these are not purely dataset-level efects. On $\mathrm { M M L U - P R O X } .$ Claude Opus 4.6 has the strongest structural alignment $( S ~ = ~ 0 . 5 6 9 )$ , while GPT-OSS-120B has the strongest order consistency $( \tau = 0 . 4 6 9 )$ ; on SciBench, Qwen3-32B reaches the highest structural score $( S = 0 . 6 2 9 )$ but much lower order consistency $( \tau = 0 . 3 3 2 )$ , whereas Claude Haiku 4.5 has the highest order score $( \tau = 0 . 4 7 0 )$ . On Strate-$\mathrm { G Y Q A } .$ Claude Opus 4.6 leads both S and τ, while Claude Haiku 4.5 has the highest holistic score $\left( H = 0 . 8 5 7 \right)$ but also the largest diagnostic gap $( H - S = 0 . 3 2 7 )$ ). Model quality under final-answer or whole-output similarity therefore does not translate monotonically into steplevel consistency, and the same model family can shift its relative behaviour across datasets.

## D.4 Rank-Based Holistic–Structural Comparison

Because S is computed from pair-calibrated sentence similarities while H is an uncalibrated whole-answer cosine, we supplement the raw H-versus-S plots with a rank-based comparison that uses only the within-evaluation, withinlanguage-pair distributions. This avoids treating $H - S$ as a calibrated metric distance.

Figure 3 gives representative raw point clouds for the same diagnostic. The panels show that high whole-output similarity can coexist with much lower step-level structural alignment, motivating the rank-based summary in Table 10.

Across the evaluations in Table 10, the mean within-pair Spearman correlation ranges from 0.21 to 0.45. Among examples in the top quartile of whole-answer cosine, a weighted average of 59.9% are not also in the top structural quartile, and 12.4% fall into the bottom structural quartile. Thus, high holistic similarity often fails to identify the examples that are most structurally aligned under the step-level metric, even when the comparison is made entirely in rank space.

## D.5 Parser Coverage

We measure the success rate of the regex-based step parser of §3.1 on the StrategyQA Claude Opus 4.6 traces from the post-judge verified question set (|verified|=458 questions, equal to 500 minus the 42 rejected-sample questions of §E.2 and seven greater than Q=451 in Table 1 because Q additionally requires that at least two answer-correct languages survive the trace-judge per question). The parser is run on the trace produced for every language on each verified question regardless of whether that language is itself answer-correct, giving 458 parser invocations per language; per-language answer accuracies on the underlying 500-problem inference set are reported separately in Figure 2.

Coverage—the fraction of those 458 traces yielding at least one parsed step—is 100.0% for English, Spanish, Hindi, Russian, Vietnamese, and Chinese, and 99.8% for Arabic, where exactly one trace produced no steps. No language required the generic numbered-list fallback in place of the language-specific Step n: / Paso n: / Chinese-character header: the fallback rate is 0.0% in all seven languages. Parser failure is therefore not a material source of the divergence patterns reported in Section 4. Coverage records only whether a trace parses; Appendix B.1.1 reports the marker-agreement diagnostic and the manual boundary spot check that address where the recovered boundaries fall.

## D.6 Step-Count-Matched Rank Statistic

The many-to-one correspondences in S tolerate a diference in step granularity, but they do not make S blind to it (Appendix D.7: merging steps lowers S). If the headline rank statistic were driven by one language splitting the same reasoning into more steps than another, it would weaken on pairs whose traces have the same number of steps. Table 11 recomputes it on exactly those pairs, with quartile thresholds recomputed within each language pair on the restricted subset, so that the restricted value is the same statistic on a smaller population.

The “all” column reproduces Table 10. Restricting to equal step counts lowers the statistic in 11 of the 18 evaluations and raises it in $7 ;$ the change averages 1.8 points in magnitude, with a net shift of −0.4 points, and every matched value lies between 49.5% and 71.0%. The largest change, +5.5 points on SciBench with DeepSeek-V3.2, falls on the evaluation where the fewest pairs have matched step counts (16.6%). Mean S rises slightly on matched pairs, by 0.003–0.023, consistent with the sanity check that a granularity mismatch costs some structural similarity. Stepgranularity mismatch therefore does not account for the dissociation. Part of what remains is arguably the construct rather than a confound: if a model systematically emits coarser steps in one language, that is a diference a reader of that trace experiences.

Table 10: Rank-based comparison of whole-answer cosine H and structural score S. Percentile thresholds are computed within each model–dataset evaluation and language pair, so the table does not interpret raw $H - S$ as a calibrated distance. $\rho _ { \mathrm { p a i r } }$ is the mean within-pair Spearman correlation. top-H not top-S is the share of instances in the top quartile of H that are not also in the top quartile of S; top-H bottom-S is the stricter share that falls in the bottom structural quartile.
<table><tr><td>Dataset</td><td>Model</td><td> $P$ </td><td>pairs</td><td> $\rho _ { \mathrm { p a i r } }$ </td><td>top-H not top-S</td><td>top-H bottom-S</td></tr><tr><td>MMLU-PROX</td><td>Claude Haiku 4.5</td><td>5,459</td><td>21</td><td>0.23</td><td>66.0%</td><td>13.8%</td></tr><tr><td>MMLU-PROX</td><td>Claude Opus 4.6</td><td>8,470</td><td>21</td><td>0.29</td><td>59.1%</td><td>12.8%</td></tr><tr><td>MMLU-PROX</td><td>DeepSeek-V3.2</td><td>7,179</td><td>21</td><td>0.30</td><td>60.6%</td><td>12.9%</td></tr><tr><td>MMLU-PROX</td><td>GPT-OSS-120B</td><td>6,552</td><td>21</td><td>0.42</td><td>53.9%</td><td>7.5%</td></tr><tr><td>MMLU-PROX</td><td>Qwen3-235B</td><td>6,656</td><td>21</td><td>0.30</td><td>60.6%</td><td>13.3%</td></tr><tr><td>MMLU-PROX</td><td>Qwen3-32B</td><td>4,714</td><td>21</td><td>0.30</td><td>59.9%</td><td>14.5%</td></tr><tr><td>SCIBENCH</td><td>Claude Haiku 4.5</td><td>4,739</td><td>21</td><td>0.24</td><td>63.3%</td><td>15.2%</td></tr><tr><td>SCIBENCH</td><td>Claude Opus 4.6</td><td>7,062</td><td>21</td><td>0.24</td><td>65.5%</td><td>15.0%</td></tr><tr><td>SCIBENCH</td><td>DeepSeek-V3.2</td><td>5,977</td><td>21</td><td>0.24</td><td>65.5%</td><td>14.7%</td></tr><tr><td>SCIBENCH</td><td>GPT-OSS-120B</td><td>5,435</td><td>21</td><td>0.24</td><td>63.7%</td><td>16.2%</td></tr><tr><td>SCIBENCH</td><td>Qwen3-235B</td><td>5,686</td><td>21</td><td>0.21</td><td>67.3%</td><td>16.4%</td></tr><tr><td>SCIBENCH</td><td>Qwen3-32B</td><td>3,874</td><td>21</td><td>0.25</td><td>61.5%</td><td>17.1%</td></tr><tr><td>STRATEGYQA</td><td>Claude Haiku 4.5</td><td>6,857</td><td>21</td><td>0.45</td><td>52.1%</td><td>7.8%</td></tr><tr><td>STRATEGYQA</td><td>Claude Opus 4.6</td><td>9,039</td><td>21</td><td>0.39</td><td>55.5%</td><td>9.3%</td></tr><tr><td>STRATEGYQA</td><td>DeepSeek-V3.2</td><td>7,813</td><td>21</td><td>0.36</td><td>56.8%</td><td>10.7%</td></tr><tr><td>STRATEGYQA</td><td>GPT-OSS-120B</td><td>6,751</td><td>21</td><td>0.29</td><td>60.2%</td><td>13.2%</td></tr><tr><td>STRATEGYQA</td><td>Qwen3-235B</td><td>7,591</td><td>21</td><td>0.39</td><td>55.8%</td><td>9.5%</td></tr><tr><td>STRATEGYQA</td><td>Qwen3-32B</td><td>5,177</td><td>21</td><td>0.38</td><td>57.1%</td><td>9.6%</td></tr></table>

Table 11: Headline rank statistic (share of top-quartile-H instances not in the top quartile of $S ,$ quartiles within each evaluation and language pair) on all pair–question instances and on those whose two traces have equal step counts. $\Delta$ is matched minus all, in percentage points, computed before rounding. Mean S is pooled over pair–question instances and therefore difers slightly from the question-macro-averaged values of Table 1.
<table><tr><td colspan="3"></td><td colspan="2">top-H not top-S</td><td colspan="2"></td><td colspan="2">mean S</td></tr><tr><td>Dataset</td><td>Model</td><td>P</td><td>all</td><td>matched</td><td>∆</td><td>% matched</td><td>all</td><td>matched</td></tr><tr><td>MMLU-PROX</td><td>Claude Haiku 4.5</td><td>5,459</td><td>66.0%</td><td>68.1%</td><td>+2.0</td><td>37.9%</td><td>0.520</td><td>0.540</td></tr><tr><td>MMLU-PROX</td><td>Claude Opus 4.6</td><td>8,470</td><td>59.1%</td><td>56.2%</td><td>-2.9</td><td>38.0%</td><td>0.572</td><td>0.595</td></tr><tr><td>MMLU-PROX</td><td>DeepSeek-V3.2</td><td>7,179</td><td>60.6%</td><td>58.0%</td><td>-2.6</td><td>22.5%</td><td>0.542</td><td>0.565</td></tr><tr><td>MMLU-PROX</td><td>GPT-OSS-120B</td><td>6,552</td><td>53.9%</td><td>52.7%</td><td>-1.2</td><td>41.0%</td><td>0.548</td><td>0.563</td></tr><tr><td>MMLU-PROX</td><td>Qwen3-235B</td><td>6,656</td><td>60.6%</td><td>61.5%</td><td>+0.9</td><td>20.2%</td><td>0.544</td><td>0.566</td></tr><tr><td>MMLU-PROX</td><td>Qwen3-32B</td><td>4,714</td><td>59.9%</td><td>59.2%</td><td>-0.7</td><td>37.3%</td><td>0.521</td><td>0.537</td></tr><tr><td>SCIBENCH</td><td>Claude Haiku 4.5</td><td>4,739</td><td>63.3%</td><td>65.7%</td><td>+2.4</td><td>34.8%</td><td>0.576</td><td>0.591</td></tr><tr><td>SCIBENCH</td><td>Claude Opus 4.6</td><td>7,062</td><td>65.5%</td><td>62.8%</td><td>-2.8</td><td>36.8%</td><td>0.611</td><td>0.621</td></tr><tr><td>SCIBENCH</td><td>DeepSeek-V3.2</td><td>5,977</td><td>65.5%</td><td>71.0%</td><td>+5.5</td><td>16.6%</td><td>0.506</td><td>0.518</td></tr><tr><td>SCIBENCH</td><td>GPT-OSS-120B</td><td>5,435</td><td>63.7%</td><td>61.9%</td><td>-1.9</td><td>31.6%</td><td>0.591</td><td>0.600</td></tr><tr><td>SCIBENCH</td><td>Qwen3-235B</td><td>5,686</td><td>67.3%</td><td>67.2%</td><td>-0.1</td><td>20.3%</td><td>0.526</td><td>0.534</td></tr><tr><td>SCIBENCH</td><td>Qwen3-32B</td><td>3,874</td><td>61.5%</td><td>57.8%</td><td>-3.7</td><td>28.7%</td><td>0.631</td><td>0.634</td></tr><tr><td>STRATEGYQA</td><td>Claude Haiku 4.5</td><td>6,857</td><td>52.1%</td><td>49.5%</td><td>-2.6</td><td>56.8%</td><td>0.533</td><td>0.547</td></tr><tr><td>STRATEGYQA</td><td>Claude Opus 4.6</td><td>9,039</td><td>55.5%</td><td>54.6%</td><td>-1.0</td><td>51.9%</td><td>0.599</td><td>0.611</td></tr><tr><td>STRATEGYQA</td><td>DeepSeek-V3.2</td><td>7,813</td><td>56.8%</td><td>57.8%</td><td>+1.0</td><td>35.5%</td><td>0.563</td><td>0.569</td></tr><tr><td>STRATEGYQA</td><td>GPT-OSS-120B</td><td>6,751</td><td>60.2%</td><td>60.4%</td><td>+0.2</td><td>47.2%</td><td>0.542</td><td>0.553</td></tr><tr><td>STRATEGYQA</td><td>Qwen3-235B</td><td>7,591</td><td>55.8%</td><td>55.9%</td><td>+0.1</td><td>46.7%</td><td>0.562</td><td>0.576</td></tr><tr><td>STRATEGYQA</td><td>Qwen3-32B</td><td>5,177</td><td>57.1%</td><td>56.8%</td><td>-0.2</td><td>46.3%</td><td>0.521</td><td>0.529</td></tr></table>

![](images/96cffde366dc10a02141fdadd2d643e1603a69f4ed255387d2b05f3aeedae9f1.jpg)  
(a) MMLU-ProX, Claude Haiku 4.5

![](images/494fa1ba65d5d2f56bb28f2c157dbfad41f1e0ffb034ffcf8a71fd44f6fd2cca.jpg)  
(b) MMLU-ProX, Qwen3-235B

![](images/e60fefe03e8b87aad3e65745a4ea46d4abc01070422a30cc8c42e7cf202eecb9.jpg)  
(c) StrategyQA, Claude Opus 4.6

![](images/4adba536467b1e1f2dad7a3fbcd227cf108aa5c66c8970c3c8f37942fcb3ba22.jpg)  
(d) StrategyQA, Qwen3-32B

![](images/5bc7b661e56ef8ceae77451554a5957a6f30c50ff52c37f177c3fdc72929f6bb.jpg)  
(e) SciBench, DeepSeek-V3.2

![](images/05c1a59e05b992f7ca8743419f12e39e2a1cd0796bf7b4934bb02786a670b21e.jpg)  
(f) SciBench, Qwen3-235B  
Figure 3: Representative pair–question point clouds for whole-answer cosine H against structural score S. These plots audit the rank-based result in Figure 1: high whole-output similarity commonly coexists with substantially lower step-level structural alignment.

## D.7 Sanity-Check Validation

We use controlled English–Chinese trace perturbations to verify that the metrics respond to the intended kinds of reasoning change. The source trace asks whether Aristotle could have used a laptop; the Chinese-side trace is either a direct translation or a controlled perturbation that reorders, merges, removes, abstracts, or replaces reasoning steps. Table 12 combines the metric outcomes with representative perturbation descriptions.

The sanity check confirms that the decomposition separates content coverage from order. Reversing the same reasoning keeps H and S high but drives τ negative, while abstraction and unrelated substitutions keep nonzero holistic similarity but sharply reduce S.

## D.8 Sharpening Ablation: Power-Law vs Softmax-Temperature

The soft assignment in §3.4 sharpens the calibrated outer step matrix A element-wise by a fixed power before row L<sub>1</sub>-normalization (default exponent p=3). To check that this choice is not driving the reported τ values, we re-aggregate soft Kendall-τ-with-tie on the same per-question step-similarity matrices stored in our step4\_divergence artifacts (so no re-encoding or re-inference is required), comparing two sharpening families: (i) power-law $\begin{array} { r c l } { \tilde { A } _ { i , j } } & { = } & { \operatorname* { m a x } ( A _ { i , j } , 0 ) ^ { p } } \end{array}$ for $\begin{array} { r l r } { p } & { { } \in } & { \{ 1 , 2 , 3 , 5 , 1 0 \} } \end{array}$ and (ii) softmaxtemperature $\tilde { A } _ { i , j } \ \propto$ exp(max $( A _ { i , j } , 0 ) / T )$ for $T \in \{ 0 . 0 5 , 0 . 1 , 0 . 2 , 0 . 5 , 1 . 0 \}$ . Both families collapse to the row-stochastic soft-assignment matrix P used by the soft Kendall-τ computation in §3.4.

Table 12: Combined sanity-check results and illustrative perturbations for the en–zh validation set. H is whole-answer cosine, S is structural similarity, and τ is soft Kendall-τ order consistency.
<table><tr><td>Case</td><td>H</td><td>S</td><td>T</td><td>Chinese-side perturbation</td></tr><tr><td>base (aligned reasoning)</td><td>0.851</td><td>0.827</td><td>0.960</td><td>Same five reasoning moves in the same order, trans- lated directly.</td></tr><tr><td>reversed order steps</td><td>0.834</td><td>0.857</td><td>-0.964</td><td>Same five moves but in reversed order.</td></tr><tr><td>step merge (coarser steps)</td><td>0.826</td><td>0.723</td><td>0.891</td><td>The first two premises are merged, and feasibility plus conclusion are expressed in fewer coarser steps.</td></tr><tr><td>step skip (missing steps)</td><td>0.848</td><td>0.746</td><td>0.884</td><td>One intermediate historical comparison is omitted while the final answer is preserved.</td></tr><tr><td>abstraction (semantic drift)</td><td>0.779</td><td>0.366</td><td>0.643</td><td>The trace keeps the broad topic but replaces con- crete date comparisons with higher-level historical</td></tr><tr><td>granularity mismatch (un- 0.825 even split)</td><td></td><td>0.631</td><td>0.839</td><td>claims. The same content is split into uneven step units across the two traces.</td></tr><tr><td>low-entropy granularity</td><td>0.811</td><td>0.433</td><td>0.000</td><td>Multiple source steps collapse onto a single broad target step.</td></tr><tr><td>unrelated (semantic failure) 0.346 0.035</td><td></td><td></td><td>0.495</td><td>A different corporate-governance reasoning trace is substituted while preserving a final  $^ { 6 } \mathrm { n o } ^ { \mathfrak { n } }$  answer string.</td></tr></table>

Table 13: Sharpening ablation. All three columns are computed against the default power $p { = } 3$ across the 18 (model, dataset) evaluations: Spearman and Pearson rank/linear correlation of $\bar { \tau } ^ { \mathrm { w i t h - t i e } }$ , and its mean absolute gap. Both families preserve the cross-evaluation ranking very closely; absolute scale shifts are expected because each scheme distributes probability mass diferently.

<table><tr><td>scheme</td><td>Spearman</td><td>Pearson</td><td>mean  $| \Delta \bar { \tau } |$ </td></tr><tr><td>power p=1</td><td>0.887</td><td>0.857</td><td>0.226</td></tr><tr><td>power  $p { = } 2$ </td><td>0.994</td><td>0.978</td><td>0.098</td></tr><tr><td>power  $p { = } 3$ </td><td>1.000</td><td>1.000</td><td>0.000</td></tr><tr><td>power  $p { = } 5$ </td><td>0.977</td><td>0.980</td><td>0.122</td></tr><tr><td>power p=10</td><td>0.948</td><td>0.933</td><td>0.235</td></tr><tr><td>softmax  $T { = } 0 . 0 5$ </td><td>0.926</td><td>0.898</td><td>0.220</td></tr><tr><td>softmax  $T { = } 0 . 1$ </td><td>0.940</td><td>0.928</td><td>0.076</td></tr><tr><td>softmax  $T { = } 0 . 2$ </td><td>0.920</td><td>0.916</td><td>0.117</td></tr><tr><td>softmax  $T { = } 0 . 5$ </td><td>0.897</td><td>0.875</td><td>0.281</td></tr><tr><td>softmax  $T { = } 1 . 0$ </td><td>0.855</td><td>0.849</td><td>0.334</td></tr></table>

Across all sharpening choices in both families, the cross-evaluation ranking of (model, dataset) pairs by mean τ¯<sup>with-tie</sup> is preserved with Spearman $\ge 0 . 8 5$ (and $\geq 0 . 9 2$ for every alternative within mean-absolute-tau distance $\leq 0 . 1 2$ of $\scriptstyle { p = 3 } )$ , and the closest softmax setting $( T \mathrm { = } 0 . 1 )$ matches the absolute scale of $p { = } 3$ to within mean $| \Delta \bar { \tau } | = 0 . 0 7 6$ . Within the power family, $p { = } 2$ and $p { = } 5$ have Spearman $\geq 0 . 9 8$ relative to $\scriptstyle p = 3 .$ . The qualitative conclusions in §5.1 and Table 1—which depend on the relative ordering of evaluations rather than absolute $\bar { \tau }$ values— are therefore robust to the sharpening scheme. We retain power-law $p { = } 3$ as the reported default because (a) it is temperature-free, (b) it has no near-zero divisor for empty rows, and (c) it produces P that is well behaved for the small step counts $( t \leq 1 5 )$ typical in our verified traces.

Table 14: StrategyQA trace excerpt for question f15b0aaa611d1d554186. The traces reach the same answer, but decompose reasoning diferently.  
English trace Spanish trace   
Step 1. Consider the physi- Step 1. Analicemos la   
cal properties of sandals. San- pregunta desde el punto de   
dals are open-toed, minimally vista de la seguridad y la   
insulated footwear that leave salud. Usar sandalias en la   
significant portions of the feet nieve implica exponer los pies   
exposed to the environment. a temperaturas muy bajas,   
humedad y superficies resbal  
adizas.   
Step 2. Consider the con- Step 2. Riesgos principales   
ditions associated with snow. de usar sandalias en la nieve:   
Snow implies temperatures at hipotermia localizada y con  
or below $0 ^ { \circ } \mathrm { C } ,$ along with gelación; resbalones y caí-   
wet conditions from snow and das; entumecimiento; enfer  
slush. medades.   
Step 3. Evaluate the risks. Step 3. Comparación con el   
Exposed feet in snow can calzado adecuado: las botas   
rapidly lose heat, leading to de invierno aíslan, mantienen   
frostbite; sandals also lack los pies secos y proporcionan   
traction suitable for icy or tracción; las sandalias carecen   
snowy surfaces. de estas características.   
Step 4. Consider any mitigat- Step 4. Conclusión: no, no   
ing factors. Thick socks with es seguro usar sandalias en la   
sandals might marginally de- nieve. Se recomienda calzado   
lay heat loss but do not ad- cerrado, impermeable y con   
equately protect against the buena tracción.   
hazards.   
Step 5. Wearing sandals   
in snow is not safe due to   
frostbite, hypothermia of the   
feet, loss of sensation, and in  
creased likelihood of slipping.

## E Qualitative Examples and Case Studies

## E.1 Qualitative Trace Examples

We include qualitative examples showing what numerical gaps correspond to in actual traces.

Verified StrategyQA example. On StrategyQA question (“Is it safe to wear sandals in snow?”), English and Spanish agree and reach high whole-trace similarity(H=0.875), but step alignment is weak(S=0.338, τ=0.009).

Verified SciBench example. On SciBench question 69, the model must recover the original pressure of an ideal gas after isothermal compression reduces its volume from 6.85 to $4 . 6 5 \mathrm { d m ^ { 3 } }$ , with final pressure 5.04 bar. The Hindi and Russian traces are both verified correct and reach $P _ { 1 } ~ \approx ~ 3 . 3 8$ atm. They preserve the same high-level order (τ=0.832) but have low structural similarity (S=0.269), because one trace folds some setup into formula-bearing steps while the other separates data listing, volume recovery, Boyle’s law, unit conversion, and rounding.

Table 15: Concrete SciBench verified trace excerpt for question 69. Non-English traces are shown as English translations of the Hindi and Russian reasoning steps.  
Hindi trace (translated) Russian trace (translated)   
Step 1. Find the initial vol- Step 1. List the given data:   
ume: since compression re- $\Delta V { \mathrm { \ ' = ~ 2 . 2 0 d m ^ { 3 } , ~ \bar { \cal P } _ { 2 } ~ = ~ 5 . } }$ 04   
duces volume, $\begin{array} { r l r } { \bar { V _ { 1 } } } & { { } = } & { V _ { 2 } \ + } \end{array}$ bar, $V _ { 2 } = 4 . 6 5 \mathrm { d m } ^ { 3 } ,$ and the   
$2 . 2 0 = 4 . 6 5 + 2 . 2 0 = 6 . 8 5 \mathrm { d m } ^ { 3 } .$ process is isothermal.   
Step 2. Apply Boyle’s law for Step 2. Recover the initial   
an isothermal ideal-gas pro- volume: $V _ { 1 } ~ = ~ V _ { 2 } + { \Delta V } ~ =$   
cess: $P _ { 1 } V _ { 1 } = P _ { 2 } V _ { 2 } .$ 4.65 + 2.20 = 6.85 dm<sup>3</sup>.   
Step 3. Compute P in bar: Step $\mathbf { 3 . \qquad A p p l y }$ Boyle’s   
$P _ { 1 } \ \stackrel { \bullet } { = } \ ( 5 . 0 4 \ \stackrel { \bullet } { \times } \ 4 . 6 5 ) \bar { / } 6 . 8 5 \ =$ law and compute in bar:   
3.4213 bar. $P _ { 1 } = P _ { 2 } V _ { 2 } / \bar { V _ { 1 } } = ( 5 . 0 4 \times$   
$4 . 6 5 ) / 6 . 8 5 = \dot { 3 } . 4 2 1 2$ bar.   
Step 4. Convert bar to atm: Step 4. Convert pressure   
$P _ { 1 } \overset { \bullet } { = } 3 . 4 2 1 3 / 1 . 0 1 3 2 5 = 3 . 3 7 7$ units: 1 atm = 1.01325 bar, so   
atm. $P _ { 1 } = 3 . 4 2 1 2 / 1 . 0 1 3 2 5 = 3 . 3 7 6$   
atm.   
Step 5. Express with ap- Step 5. Round to three signif  
propriate significant figures: icant figures: P ≈ 3.38 atm.   
P ≈ 3.38 atm.

Verified MMLU-ProX example. In a retained MMLU-ProX seed-dormancy case, English and Chinese both select the correct option J. Whole-output similarity remains high (H=0.857), but structural similarity is much lower (S=0.384), because English evaluates the answer choices almost immediately while Chinese first states the biological definition and the environmental-cue mechanism before excluding alternatives.

Table 16: Concrete MMLU-ProX verified trace excerpt for question 222. The Chinese trace is shown in English translation.

![](images/34f0dab7cebc800c60f4329107dc431db0ffe265f9325beb5832e0998b7a15d8.jpg)

## E.2 Rejected-Sample Case Studies

The main trace-divergence results use only language pairs for which both answers are verified correct. As a qualitative diagnostic, we also evaluate examples rejected from the primary verified set with the same trace-divergence metrics. This analysis is not a primary metric, because rejected examples can conflate answer error, answer inconsistency, and reasoning divergence. It is nevertheless useful for inspecting whether failures are shared across languages or arise from language-specific interpretations.

Compared with the corresponding verified traces, rejected traces have lower structural similarity and order consistency, while wholeoutput similarity remains high. For Claude Opus 4.6 on StrategyQA (42 rejected questions), the rejected set has S=0.563 and τ=0.341, compared with S=0.598 and τ=0.418 on the verified set. On MMLU-ProX (65 rejected), the rejected set similarly drops from S=0.569 to S=0.525 and from τ=0.414 to τ=0.354. For DeepSeek-V3.2 on StrategyQA (69 rejected), rejected examples have S=0.537 and τ=0.270, compared with verified values of S=0.560 and τ=0.324. In all three cases H remains close to 0.80 (0.810, 0.807, and 0.803 respectively), so holistic similarity alone does not distinguish verified reasoning agreement from rejected-sample failure modes.

Shared wrong reasoning. On the StrategyQA question “Is the Foreign and Commonwealth Ofice a European political agency?”, the dataset answer is yes, but Claude Opus 4.6 answers no in all seven languages. The traces are highly similar (S=0.756, H=0.887): each language interprets “European political agency” as an EU or pan-European institution and rejects the label because the Foreign and Commonwealth Ofice is a UK government department. This case shows that high cross-lingual trace similarity can reflect a shared error rather than reliable reasoning. DeepSeek-V3.2 exhibits the same final-answer failure on this question, with all seven languages answering no and S=0.700.

Table 17: Rejected StrategyQA trace excerpt for the Foreign and Commonwealth Ofice question. Both traces share the same wrong institutional interpretation.
<table><tr><td>English trace</td><td>Spanish trace</td></tr><tr><td>Step 1. Identify the FCO as a department of the UK government responsi- ble for British foreign af- fairs. Step 2. Define a Euro- pean political agency as an EU or pan-European institution such as the Eu-</td><td>Step 1. Identifica la FCO como un ministerio del go- bierno del Reino Unido re- sponsable de la política exterior y la Common- wealth. Step 2. Determina que una agencia política euro- pea sería una agencia de la Unión Europea o de una</td></tr><tr><td>ropean Commission or Eu- ropol. Step 3. Compare the FCO against that defini- tion and conclude it is a national British institu-</td><td>organización paneuropea. Step 3. Aunque el Reino Unido está geográ- ficamente en Europa, la FCO es un ministerio na-</td></tr><tr><td>tion, not an EU or pan- European agency. Final. No.</td><td>cional británico, no una institución supranacional europea. Final. No.</td></tr></table>

Divergent rejected reasoning. On the StrategyQA question “Can a lifeboat rescue people in the Hooke Sea?”, the dataset answer is no. DeepSeek-V3.2 splits across languages: Arabic, English, and Vietnamese answer yes, while Spanish, Hindi, Russian, and Chinese answer no. The structural score is low (S=0.449, τ=0.221), with particularly weak pairs such as English–Hindi (S=0.268) and English–Russian (S=0.333). The yes-answer traces treat Hooke Sea as a hypothetical ordinary sea; the noanswer traces notice that the named sea is fictional or wordplay. This case illustrates a diferent rejected-sample mode: the model does not merely share one wrong belief across languages, but changes the interpretation of the premise.

Table 18: Rejected StrategyQA trace excerpt for the Hooke Sea question. The English trace answers yes by assuming an ordinary sea, while the Hindi trace answers no by treating the location as fictional. The Hindi trace is shown in English translation.  
Table 19: Rejected MMLU-ProX trace excerpt for question 141. The Chinese trace is shown in English translation.
<table><tr><td colspan="3">The Hindi trace is shown in English translation.</td><td colspan="2">English trace</td><td colspan="2">Spanish trace Chinese trace (translated)</td></tr><tr><td>English trace</td><td>Hindi trace lated)</td><td>(trans-</td><td>Rate assump- tion. Uses</td><td>Rate assump- tion. Starts</td><td>tion.</td><td>Rate assump- Starts</td></tr><tr><td>Step 1. Treat &quot;Hooke Sea&quot; as a fictional or hy- pothetical sea named after</td><td>Step 1. Treat Hooke Sea as a fictional sea related to Hooke&#x27;s law.</td><td></td><td>20-payment life $26.47 and ordi- nary life $24.44 per $1,000, then</td><td>with $25.04 and $22.86 per $1,000,</td><td>$24.26, then switches much</td><td>with $26.40 and then to a wider</td></tr><tr><td>Robert Hooke. Step 2. In the absence of special constraints, as- sume it is a body of water</td><td>Step 2. Since the sea is fictional, there are no real people there to rescue.</td><td></td><td>tries nearby textbook rates. Calculation.</td><td>tries alternate rates. Calculation.</td><td>Calculation.</td><td>“common text- book&quot; rate gap.</td></tr><tr><td>similar to ordinary seas. Step 3. Lifeboats are de- signed to rescue people in seas or oceans if they can</td><td>Step 3. A lifeboat is for rescuing real people, but a fictional sea contains no</td><td></td><td>Computes dif- ferences around $30.45-$30.60 and maps this to option A. date that maps</td><td>Computes $32.70, $32.10, and then $32.55 candi-</td><td>Reverse- engineers option $2.1033 per</td><td>I: $31.55/15 =</td></tr></table>

Rejected MMLU-ProX answer split. On MMLU-ProX question 141, Claude Opus 4.6 is asked to compare annual premiums for a \$15,000 insurance policy under a 20-payment life policy and an ordinary life policy paid up at age 65. The gold option is B, but none of the inspected English, Spanish, and Chinese traces is correct: English selects A, Spanish selects D, and Chinese selects I. The failure is not a single shared wrong answer. Each trace follows the same broad tabular-rate procedure–choose premium rates per \$1,000, multiply by 15, and match the nearest option– but the languages assume diferent rate-table values, causing diferent numerical diferences and therefore diferent final options. This case adds a multiple-choice rejected-sample mode to the StrategyQA examples: translation preserves the task format, yet the model’s unstated lookup assumptions drift across languages.

## F Human Validation of the Trace-Divergence Metrics

This appendix gives the protocol, sampling procedure, and full statistics behind §5.6.

## F.1 Protocol

Task. Each item presents an English source question, the final answer (both languages reached the same verified-correct answer), and two reasoning traces labeled Trace A and Trace B. Annotators judge whether the traces reach the answer by the same reasoning. Three fields are recorded per item:

• equivalence, the primary variable: Same / Partially diferent / Substantially diferent, coded ordinally as $2 / 1 / 0$

• diff\_\*, a multi-label diference taxonomy: wording, granularity, missing content, different strategy, diferent order.

• material, binary: would the diference change what a reviewer concludes when using the trace to verify the answer? A binary item yields a more stable per-stratum rate at this sample size than a Likert scale.

Blinding. Annotators never see S, τ , H, or any stratum assignment; scores live only in a hidden key. The assignment of English and non-English traces to Trace A and Trace B is randomized per item to remove positional bias. Every item is labeled independently by at least two annotators, with a shared calibration subset labeled by all four. Annotators do not confer during labeling.

Annotators. Four bilingual annotators per language pair, disjoint across the two studies. §F.5 reports a shared-anchor control establishing that the two groups do not difer in labeling threshold.

## F.2 Sampling

Both studies draw from Claude Opus 4.6 outputs pooled over MMLU-ProX, SciBench, and StrategyQA, restricted to pairs where both languages are answer-correct and both traces parse into at least two steps, with duplicate question text removed. The EN–ES eligible pool contains n = 1,178 pairs.

Items are sampled crossed on H and S: both metrics are split at pool-relative terciles and items are drawn from every cell of the resulting 3 × 3 grid, with a round-robin over datasets inside each cell. This matters for interpretation. Stratifying on S alone while holding H high would suppress H’s variance by construction, so a null correlation for H would be partly guaranteed by the design rather than measured. Sampling crossed avoids that: both metrics span their natural ranges, and any asymmetry between them is a property of the metrics rather than of the sample.

Table 20 confirms the design achieved this. Sample standard deviations closely match the pool for both metrics in both studies, and the sample correlation between H and S reproduces the pool value (0.25 against a pool 0.254 for EN–ES), so the design induces no dependence between the two axes.

Table 20: Range-restriction check. A ratio near 1 indicates the metric was sampled over its natural spread, so its correlation is not attenuated by the design.
<table><tr><td>Study</td><td>Metric</td><td>sample sd</td><td>pool sd</td><td>ratio</td></tr><tr><td>EN-ZH</td><td>S</td><td>0.103</td><td>0.102</td><td>1.02</td></tr><tr><td>EN-ZH</td><td>H</td><td>0.048</td><td>0.053</td><td>0.90</td></tr><tr><td>EN-ES</td><td>S</td><td>0.109</td><td>0.105</td><td>1.04</td></tr><tr><td>EN-ES</td><td>H</td><td>0.042</td><td>0.047</td><td>0.89</td></tr></table>

## F.3 Correlational Results

Table 4 in §5.6 reports the correlations themselves. This section gives the tests behind them. Testing the diference, not two separate nulls. Reporting that S correlates while H does not is two independent tests, and invites the objection that the null on H merely reflects low power. The claim is a diference between dependent correlations sharing a criterion variable, so we test it as one. Williams’ test on ρ(equiv, S) against ρ(equiv, H) gives $\Delta \rho ~ = ~ 0 . 4 1$ bootstrap 95% CI [0.12, 0.68], t(57) = 2.81, $ { p } \ : = \ : \ : 0 . 0 0 7$ for EN–ES, and ∆ρ = 0.437, t(57) = 2.80, p = 0.0069 for EN– ZH. The dissociation is significant in both pairs at efectively the same strength.

Parametric form. An ordinal logistic regression of equivalence on standardized S and H (EN–ES) gives $\beta _ { z ( S ) } = 0 . 8 6$ (SE 0.25, OR 2.36, $p = 0 . 0 0 0 6 )$ and $\dot { \beta } _ { z ( H ) } = - 0 . 0 9 \ \mathrm { ( S E ~ 0 . 2 4 , ~ O R }$ 0.91, p = 0.71). Likelihood-ratio tests dropping each term give $\chi ^ { 2 } ( 1 ) = 1 3 . 1$ $p = 0 . 0 0 0 3$ for

S and $\chi ^ { 2 } ( 1 ) = 0 . 1 4 , p = 0 . 7 1$ for H. A onestandard-deviation drop in S multiplies the odds of a lower human equivalence rating by roughly 2.4.

Generalization across language pairs. Pooling both studies gives $\rho ( \mathrm { e q u i v } , S ) = 0 . 4 0$ $p < 0 . 0 0 1 , N = 1 2 0$ . A Fisher-z homogeneity test between pairs gives $z = 0 . 4 6 , p = 0 . 6 5 \colon$ the two language pairs do not difer. English– Spanish is the more demanding test of the two, since Spanish is typologically closer to English and its traces should diverge less.

## F.4 Comparison with an LLM-as-Judge Baseline

A natural alternative to a structural metric is to ask a strong model directly whether two traces reason the same way. We ran that baseline on the 60 English–Chinese pairs of the initial annotation round, giving the judge (GPT-5.6) the identical rubric our annotators used, both traces, and no similarity scores.

The structural metric predicts human-judged equivalence more closely than the judge does. On these items $\rho ( { \mathrm { e q u i v } } , S ) = 0 . 4 1 9 \ ( p = 0 . 0 0 4 )$ and $\rho ( \mathrm { e q u i v } , \tau ) = 0 . 4 1 3 \ ( p = 0 . 0 0 5 )$ , while the judge reaches $\rho = 0 . 2 3 5 ~ ( p = 0 . 1 2 , \mathrm { n . s . } )$ and the holistic baseline $\rho = 0 . 1 2 9 \ \mathrm { ( n . s . ) }$ . The judge also agrees with the human consensus less than the annotators agree with one another (pairwise agreement 0.576 against 0.633), and entering it as a fifth coder halves ordinal Krippendorf’s α, from 0.462 to 0.237: relative to the consensus it behaves as an outlier rather than as a substitute for it. Its errors are systematic rather than noisy—exact agreement with the human majority is 0.422 but within-one agreement is 0.956, so it is nearly always adjacent on the ordinal scale.

The failure mode is the one the framework targets. The judge never applied the Substantially diferent label at any point, and its mean rating does not vary monotonically across S strata. On high-H / low-S pairs it rates as Same or Partially diferent items that annotators unanimously call substantially diferent— for instance a multiple-choice item answered correctly in both languages where one trace eliminates every option exhaustively and the other identifies the answer directly $( S = 0 . 4 6$ $H = 0 . 8 0$ ; human Substantially diferent, judge Same).

We state the strength of this comparison precisely: the confidence intervals for S and for the judge overlap, so the defensible claim is that S is at least as faithful to human judgment as the judge and, unlike the judge, statistically significant on this sample. Beyond faithfulness, S is deterministic, localizes divergence to specific steps, and separates content from order at no per-call cost.

## F.5 Supporting Analyses

Monotonicity. Mean human equivalence rises monotonically across S terciles and is flat across H terciles. For EN–ES the means are 0.92/1.28/1.65 by S (Jonckheere–Terpstra $\begin{array} { r l r } { p } & { { } < } & { 0 . 0 0 1 ) } \end{array}$ against 1.31/1.24/1.30 by H $( p = 0 . 7 2 )$ . EN–ZH shows the same pattern: 1.069/1.324/1.420 by S, and 1.433/1.162/1.300 by H.

Detection framing. Treating a majority Substantially diferent label as the positive class, the area under the ROC curve is 0.81 [0.68, 0.92] for S, 0.76 [0.61, 0.88] for τ, and 0.54 [0.36, 0.71] for H; a DeLong test of S against H gives $p = 0 . 0 1 2$ . We present this as further evidence for the dissociation rather than as a deployable screening threshold: the positive class holds 11 items in EN–ES, and the interval is correspondingly wide.

Diference types. In EN–ES the diff\_wording rate is flat across S terciles (0.21/0.19/0.17) while the content tags (diff\_missing, diff\_granularity, diff\_order) fall by a factor of three to four from low to high S. In EN–ZH the low-S stratum is dominated by missing reasoning content (0.477), granularity (0.295), and order (0.205), with wording both the smallest category (0.114) and the only one declining monotonically across S (0.114/0.071/0.034). A metric tracking surface verbalization would produce the opposite profile in both studies.

Agreement and attenuation. Ordinal Krippendorf’s α is 0.411 for EN–ZH and 0.58 for EN–ES. We use Krippendorf’s α as the headline agreement statistic because it is the only standard option that simultaneously accommodates a small sample, a variable number of coders per item, and ordinal data; Cohen’s κ requires exactly two fixed raters and Fleiss’ κ a fixed rater count and nominal data. Because the human rating is itself a noisy instrument, the correlations above are attenuated rather than inflated by this unreliability. Correcting with the composite reliability of each study’s rating mean gives 0.44 → 0.49 for EN–ES (composite reliability 0.81) and 0.371 → 0.47 for EN–ZH (0.61). The two pairs agree more closely once measurement error is accounted for. We report the observed values as the headline throughout.

Annotator-group control. Because the two studies use disjoint annotator groups, a diference between language pairs could in principle reflect the annotators rather than the languages. On 12 shared English–English anchor items labeled by both groups, mean equivalence is 1.36 for the EN–ZH group and 1.31 for the EN–ES group (diference 0.05, [−0.19, 0.29]), Same rates are 34% and 36% (Mann–Whitney $p = 0 . 6 8 )$ , and α on the anchors is 0.61 and 0.59. The groups are indistinguishable on shared items, so the EN–ES result reflects the language pair rather than a group-specific labeling threshold.

Orientation control. Comparing items where the non-English trace appeared as Trace A against those where it appeared as Trace B gives mean equivalence 1.29 against 1.28 (Mann– Whitney $p = 0 . 8 8 )$ : no presentation-order effect.

Outcome variance. Overall mean equivalence is 1.29 (sd 0.474) for EN–ZH and 1.28 (sd 0.47) for EN–ES, so the diference between studies is not an artifact of difering outcome spread.

## F.6 Annotation Guide

We reproduce the instructions given to annotators. The guide opens by stating that no similarity score of any kind will be shown and that only the reasoning is to be judged.

What is being labeled. Each item gives an English source question, the final answer (both languages reached the same verified-correct answer), and two reasoning traces labeled Trace A and Trace B, one English and one non-English, with the side assignment randomized per item. Annotators read both and decide whether the traces reach the answer by the same reasoning, recording three fields per item with fixed allowed values.

Field 1: equivalence (the primary label, exactly one value).

• Same — the two traces use the same reasoning: same key claims, same logical path. Pure wording or translation diferences still count as Same.

• Partially diferent — largely the same reasoning, but with a real diference in one part: a step present on one side is missing or merged on the other, or one subargument difers, while the overall path is recognizably the same.

• Substantially diferent — the traces reason diferently: diferent strategy, diferent decomposition, diferent premises or evidence, or diferent logical order of the core steps. A reader could not treat one as a faithful stand-in for the other.

Field 2: diff\_\* (diference type; mark each that applies, skipped entirely when equivalence is Same). diff\_wording: only phrasing or translation difers. diff\_granularity: same content, split into more, fewer, or uneven steps. diff\_missing: a reasoning step or claim present on one side is absent on the other. diff\_strategy: a genuinely diferent solution approach or argument. diff\_order: the same core steps occur in a diferent order. diff\_other: with a free-text description.

Field 3: material (binary). Would the diference change what a human reviewer concludes when using the trace to verify the answer or understand how it was reached? If the traces are the same or difer only cosmetically, No. If a reviewer auditing one trace would reach a materially diferent understanding of why the answer holds than from the other, Yes.

Rules of thumb. The guide instructs annotators to translate mentally and compare logic rather than surface tokens, noting that two traces in diferent languages will never be wordidentical and that this alone is not a diference; to disregard correctness, since both answers are already verified correct, and to judge only whether the reasoning is the same rather than whether one trace is better; to pick the more diferent label when torn between two and add a note; and not to discuss items with other annotators while labeling, so that judgments remain independent.

Language check. Annotators are asked to flag any trace written in the wrong language— one that should be in Spanish but is actually in English, or the reverse—with a fixed token in the notes field, then label the item as normal. This is requested particularly of the EN–ES annotators, because English and Spanish share the Latin alphabet and a reversion that an automatic script-level check cannot see is immediately obvious to a bilingual reader.

Calibration. All annotators for a study first label a shared calibration subset. After those are complete the group meets briefly to compare judgments and settle the wording of edge cases, and the remaining items are then labeled independently. Annotators returning for the EN–ZH top-up batch were instructed that the labels and their definitions were unchanged and to apply exactly the standard they had used previously, without re-reading their earlier files to match specific judgments, since the two batches are pooled in analysis.

## F.7 Annotator Recruitment and Consent

Annotators were unpaid bilingual volunteers from the authors’ academic and professional networks, based in Asia and North America, all adults with professional or native proficiency in both languages of the pair they annotated. Participation was voluntary and could be stopped at any time; annotators were told the purpose of the study and how their labels would be used, and consented on that basis. The burden was under thirty minutes per annotator for English– Chinese and under ninety minutes for English– Spanish. No personal data was collected, and none is retained in any released artifact. We did not seek ethics-board review: the study collects no personal or sensitive data, involves no deception or intervention, and asks adult volunteers only to compare model-generated text derived from public benchmarks—a minimal-risk protocol under our institutional framework.

## G Encoder Sensitivity

Methodological concern. S and τ are computed from sentence embeddings, so a natural question is how much of the reported H−S contrast is a property of reasoning traces and how much is a property of the encoders. Two distinct worries sit inside this. First, H and S are produced by diferent encoders in the main experiments (BGE-M3 and LaBSE respectively), so the gap between them could be an artifact of that asymmetry. Second, the framework might require a strong encoder to work at all, which would limit its usefulness.

Setup. We recompute S and τ from scratch under two alternative step encoders, on three Claude Opus 4.6 evaluations at full retained size (StrategyQA n = 451, MMLU-ProX n = 431, SciBench n = 362), across all 21 language pairs. The per-pair floor and ceiling are re-derived for each encoder rather than reused, since calibration constants estimated for one embedding space do not transfer. The alternatives are chosen to probe the two worries separately:

• BGE-M3 (568M parameters, 1024 dimensions, retrieval-contrastive objective) is the encoder H already uses, so running S on it removes the H–S encoder asymmetry entirely. This is the only sound direction for that swap: placing LaBSE on H would truncate whole traces at its 256-token limit, whereas individual steps fit comfortably in both encoders.

• paraphrase-multilingual-MiniLM-L12-v2 (118M parameters, 384 dimensions, paraphrase-distillation objective) is a fourto five-fold reduction in scale from LaBSE (471M) and a third training objective. Its 128-token limit is not binding here, because the encoding unit is sentences within steps rather than whole steps: median sentence length is 56 characters in English and 28 in Chinese, with 99th percentiles of 237 and 124.

Validation of the reimplementation. Before comparing encoders we re-ran the identical pipeline with LaBSE, the encoder used in the main experiments, and recovered the cached values (Spearman 0.959 on S, mean |∆S|= 0.0044, top-quartile mismatch unchanged). Movement under an alternative encoder is therefore attributable to the encoder and not to the reimplementation.

Table 21: Encoder sensitivity on three Claude Opus 4.6 evaluations. H is held fixed (the BGE-M3 whole-answer cosine) in every column, so all movement is in S. “Mismatch” is the percentage of top-quartile-H examples that are not top-quartile-$S . ~ \rho ( S )$ and $\rho ( \tau )$ are Spearman rank correlations of per-question scores against the LaBSE baseline, so they are undefined for LaBSE itself.
<table><tr><td>Dataset</td><td colspan="3">Quantity LaBSE BGE-M3 MiniLM</td></tr><tr><td rowspan="6">STRATEGYQA</td><td>H-S</td><td>0.220</td><td>0.232</td><td>0.111</td></tr><tr><td>Mismatch</td><td>55.5%</td><td>50.1%</td><td>58.7%</td></tr><tr><td> $\rho ( S )$ </td><td></td><td>0.803</td><td>0.696</td></tr><tr><td> $\rho ( \tau )$ </td><td></td><td>0.876</td><td>0.795</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td> $H { - } S$ </td><td>0.231 59.1%</td><td>0.244 52.2%</td><td>0.134 57.9%</td></tr><tr><td rowspan="3">MMLU-PROX</td><td>Mismatch</td><td></td><td></td><td></td></tr><tr><td> $\rho ( S )$ </td><td></td><td>0.839</td><td>0.842</td></tr><tr><td> $\rho ( \tau )$ </td><td></td><td>0.838</td><td>0.826</td></tr><tr><td rowspan="4">SCIBENCH</td><td> $H { - } S$ </td><td>0.169</td><td>0.198</td><td>0.131</td></tr><tr><td>Mismatch</td><td>65.5%</td><td>60.9%</td><td>62.1%</td></tr><tr><td> $\rho ( S )$ </td><td></td><td>0.876</td><td>0.913</td></tr><tr><td> $\rho ( \tau )$ </td><td></td><td>0.665</td><td>0.820</td></tr><tr><td rowspan="2">Mean</td><td> $H { - } S$ </td><td>0.207</td><td>0.225</td><td>0.125</td></tr><tr><td> $\operatorname { M i s m a t c h }$ </td><td>60.0%</td><td>54.4%</td><td>59.6%</td></tr></table>

Result: the rank-based finding is stable. Across all three encoders and three datasets, every evaluation falls between 50.1% and 65.5% of top-quartile-H examples not being top-quartile-S, closely tracking the 52.1%–67.3% range in Table 1. This holds when H and S are produced by the same encoder, which disposes of the asymmetry worry, and it holds on a distilled 118M model that runs on CPU, which disposes of the second: the dissociation does not require a strong encoder.

Result: absolute levels are not stable. The H−S gap moves in both directions widening under BGE-M3 $( 0 . 2 0 7  0 . 2 2 5$ on average) and narrowing under MiniLM $( 0 . 2 0 7 $ 0.125), the latter because MiniLM assigns systematically higher calibrated step similarities. H is identical in every column, so this movement is entirely in S. We therefore treat absolute calibrated values of S and τ as encoderrelative and interpret them by rank within a fixed encoder, which is how they are reported throughout the paper.

Rank preservation and trace length. Perquestion S rankings are substantially preserved across encoders, and preservation rises monotonically with trace length under both alternatives $( \mathrm { B G E  – M 3 : } 0 . 8 0 3 < 0 . 8 3 9 < 0 . 8 7 6 ;$ MiniLM: $0 . 6 9 6 ~ < ~ 0 . 8 4 2 ~ < ~ 0 . 9 1 3$ , ordered from the shortest-trace dataset to the longest). Short traces yield small step-similarity matrices, which are inherently more sensitive to which encoder scores them. The weakest cell in the study is StrategyQA under MiniLM at $\rho ( S ) = 0 . 6 9 6$

Order consistency. Which of the two metrics is more encoder-sensitive is itself encoderdependent. Under BGE-M3, τ is the less stable of the two and falls to $\rho = 0 . 6 6 5$ on SciBench while S holds at 0.876; under MiniLM the ordering reverses, with τ the more uniform $( 0 . 7 9 5 / 0 . 8 2 6 / 0 . 8 2 0 )$ and S the more variable (0.696/0.842/0.913). We therefore state the limitation without attaching a mechanism to it: absolute τ values should be compared within an encoder rather than across encoders.

Calibration behavior. How much work the per-pair calibration does is also encoder-specific, and does not track encoder strength. The crosslanguage-pair drift in the unrelated-pair floor that calibration exists to correct spans 0.157– 0.381 under LaBSE on StrategyQA, but only 0.008–0.022 under BGE-M3 and 0.046–0.055 under MiniLM. On a well-aligned embedding space the floor correction is nearly inert, while the ceiling frequently saturates at the cap (all 21 SciBench ceilings pin at 0.98 under both alternatives). The magnitude of the calibration correction is thus a property of the embedding space rather than of the encoder’s capacity. Scope. One model family, three datasets at full retained size, three encoders. We do not test whether the pattern holds on other model families or below 118M parameters.

## H Same-Language Control

This appendix gives the design and full results of the same-language control summarized in §5.4.

## H.1 Design

The paraphrase condition. Every reported S and τ in the main evaluation compares traces across a language boundary, so the level of divergence alone cannot distinguish an efect of changing language from ordinary variation in generated reasoning. The control adds an eighth condition, $\mathrm { e n } _ { \mathrm { p a r a } } .$ each English question is rewritten as a meaning-preserving English paraphrase and answered under the same system prompt, decoding (greedy, temperature 0, no reasoning mode), parser, retention cascade, and metric settings as every other condition. $\mathrm { e n } _ { \mathrm { p a r a } }$ receives the same appended language instruction as a translated condition, naming English, so that the only uncontrolled diference between the pairs $( \mathrm { { e n } , \mathrm { { e n _ { p a r a } } ) } }$ and $( \mathrm { e n } , \ell )$ is the language of the output. The paraphrases are produced by Claude Opus 4.6 with the prompt below, which asks for as little surface overlap with the original as the language allows: a paraphraser left to itself restates a question nearly verbatim, which would collapse the condition onto the source and measure nothing.

![](images/a8bf83a07d936e4e6de96f72358256db9b56aa9e916238c681c7624110940c96.jpg)

Models, questions, and conditions. The control runs on GPT-OSS-120B and DeepSeek-V3.2 across all three benchmarks. For each benchmark it uses a seeded random subset of 100 questions from the main 500-problem sample. On MMLU-ProX and SciBench, the seven original conditions reuse the main-run responses and only $\mathrm { e n } _ { \mathrm { p a r a } }$ is generated. On StrategyQA, all eight conditions are generated within this run, with each question translated into the six target languages by Claude Opus 4.6 before inference. Between 41 and 68 questions per cell retain both $( \mathrm { { e n } , \mathrm { { e n _ { p a r a } } ) } }$ and at least one English-anchored cross-lingual pair after the retention cascade, 337 in total.

Calibration of the same-language pair. Floors are estimated within this run for all 28 language pairs by the procedure of Eq. (11); cross-lingual ceilings are taken from the main runs. A same-language ceiling cannot be estimated from translation, because translating a sentence into its own language is the identity map, which would peg the ceiling at $c _ { \mathrm { m a x } }$ and deflate every calibrated score in the cell. The $( \mathrm { { e n } , \mathrm { { e n _ { p a r a } } ) } }$ ceiling is therefore estimated from sentence-level paraphrases, the same-language analogue of a translation, over 50 sentences per direction:

Same-language ceiling: system prompt   
You are a careful sentence rewriter.   
Rewrite the given sentence in the same   
language using different wording,   
preserving its meaning, technical terms,   
symbols, and numbers exactly.   
Change word choice and sentence structure   
where the language allows it; do not   
return the sentence unchanged.   
Do not add, omit, explain, or reformat   
anything.   
Return only the rewritten sentence.

Comparison and reporting rule. For each question, ∆S is $S ( \mathrm { e n , e n _ { p a r a } } )$ minus the mean of $S ( \mathrm { e n } , \ell )$ over the English-anchored crosslingual pairs available for that question. Pairing within question matters here because question identity carries far more of the variance in S than language pair does. The comparison is against English-anchored pairs because those are the pairs that share the English trace with $( \mathrm { { e n } , \mathrm { { e n _ { p a r a } } ) } }$ . The comparison set, the perquestion pairing, and the reporting of the result whichever way it fell were fixed before the run, as were decision thresholds for MMLU-ProX (0.030) and SciBench (0.015), set from the size of the floor-induced bias measured on stored artifacts before any paraphrase was generated: an advantage for $( \mathrm { { e n } , \mathrm { { e n _ { p a r a } } ) } }$ below the threshold would have been read as no detectable diference. StrategyQA was added to the run after the thresholds were set and has none.

## H.2 Results

Table 22 gives the full per-cell results. Two English traces agree more closely than an English trace and its cross-lingual counterparts in every cell, and order consistency is higher within language in every cell. The efect clears its threshold in all four cells that have one. Its size varies about threefold across cells, from +0.041 to +0.136, so the direction is consistent but the magnitude should not be quoted as a single constant. The weakest cell, StrategyQA with GPT-OSS-120B, is also the only one in which (en, en<sub>para</sub>) does not rank first.

Calibration exposure. Per-pair calibration cannot afect within-pair rankings, but this control compares levels across pairs, which it can. To measure that exposure we pass a single fixed raw cosine, 0.60, through each pair’s calibration map and take the diference between the calibrated value for $( \mathrm { { e n } , \mathrm { { e n _ { p a r a } } ) } }$ and the mean calibrated value over the Englishanchored pairs. A positive diference means calibration alone favours the same-language pair. Table 23 reports it. The bias is largest in the two DeepSeek-V3.2 cells, where the same-language floor falls below every Englishanchored floor, and it reaches at most 29% of the observed efect. Subtracting it cell by cell leaves ∆S between +0.046 and +0.124, above threshold wherever a threshold was set.

The dissociation within language. The rank statistic of §5.2 is invariant to any perpair monotone rescaling, so it is immune to the exposure above. Table 24 reports it for $( \mathrm { { e n } , \mathrm { { e n _ { p a r a } } ) } }$ beside its mean over the 21 crosslingual pairs of the same run. The cross-lingual values land in or near the 52.1%–67.3% range of the main evaluation. Within language the statistic is lower in five of six cells, but it stays between 41.2% and 60.0%: two English traces whose whole responses are most alike are, more often than not, not among the most alike step by step. Both S and H rise in the paraphrase condition. The within-language statistic rests on a single language pair of 41–68 instances per cell, so it is coarser than the cross-lingual one. Scope. The control covers two of the six models; the model behind most of the detailed analyses, Claude Opus 4.6, is not among them, so the control sits beside those results rather than under them. Each question has one paraphrase, so variation between paraphrases is not measured. Resampling an identical prompt at non-zero temperature, which would isolate pure decoding variation from sensitivity to rewording, is not part of the control.

Table 22: Same-language control, full results. ∆S is the per-question diference defined in Appendix H.1, with its 95% t-interval and a one-sample t-test; $\# > 0$ counts questions with $\Delta S > 0$ . τ columns give mean tie-aware soft Kendall-τ for $( \mathrm { { e n } , \mathrm { { e n _ { p a r a } } ) } }$ and for the English-anchored cross-lingual pairs. “Rank” is the position of $( \mathrm { { e n } , \mathrm { { e n _ { p a r a } } ) } }$ by mean S among the 22 pairs that do not involve a second paraphrase condition.
<table><tr><td>Dataset</td><td>Model</td><td>n</td><td>∆S</td><td>95% CI</td><td></td><td>t</td><td>p</td><td> $\# > 0$ </td><td> $\tau _ { \mathrm { w i t h i n } }$ </td><td> $\tau _ { \mathrm { c r o s s } }$ </td><td>Rank</td></tr><tr><td>MMLU-PROX</td><td>GPT-OSS-120B</td><td></td><td>59+0.120</td><td>[+0.091, +0.149]</td><td></td><td>8.32</td><td> $1 . 8 \times 1 0 ^ { - 1 1 }$ </td><td>49</td><td>0.599</td><td>0.500</td><td>1/22</td></tr><tr><td>MMLU-PROX</td><td>DeepSeek-V3.2</td><td></td><td>68+0.071</td><td>[+0.047, +0.096]</td><td></td><td>5.79</td><td> $2 . 1 \times 1 0 ^ { - 7 }$ </td><td>56</td><td>0.416</td><td>0.389</td><td>1/22</td></tr><tr><td>SCIBENCH</td><td>GPT-OSS-120B</td><td></td><td>41+0.136</td><td>[+0.088, +0.184]</td><td></td><td>5.76</td><td> $1 . 0 \times 1 0 ^ { - 6 }$ </td><td>36</td><td>0.441</td><td>0.353</td><td>1/22</td></tr><tr><td>SCIBENCH</td><td>DeepSeek-V3.2</td><td></td><td>47+0.083</td><td>[+0.046, +0.120]</td><td></td><td>4.50</td><td> $4 . 6 \times 1 0 ^ { - 5 }$ </td><td>39</td><td>0.404</td><td>0.357</td><td>1/22</td></tr><tr><td>STRATEGYQA</td><td>GPT-OSS-120B</td><td></td><td>57+0.041</td><td>[+0.012, +0.070]</td><td></td><td>2.80</td><td> $7 . 1 \times 1 0 ^ { - 3 }$ </td><td>36</td><td>0.438</td><td>0.401</td><td>4/22</td></tr><tr><td>STRATEGYQA</td><td>DeepSeek-V3.2</td><td></td><td>65+0.101</td><td>[+0.073, +0.128]</td><td></td><td>7.31</td><td> $5 . 3 \times 1 0 ^ { - 1 0 }$ </td><td>56</td><td>0.335</td><td>0.267</td><td>1/22</td></tr><tr><td>Pooled</td><td></td><td>337</td><td>+0.090</td><td> $[ + 0 . 0 7 7 , + 0 . 1 0 3 ]$ </td><td></td><td>13.75</td><td> $1 . 9 \times 1 0 ^ { - 3 4 }$ </td><td>272</td><td></td><td></td><td></td></tr></table>

Table 23: Calibration exposure of the same-language comparison. “Floor” is the $( \mathrm { { e n } , \mathrm { { e n _ { p a r a } } ) } }$ floor, beside the range of the six English-anchored floors. “Bias” is the calibrated advantage a fixed raw cosine of 0.60 receives in the same-language pair, “share” is |bias|/∆S, and “corrected” is $\Delta S -$ bias, compared with the decision threshold where one was set.
<table><tr><td>Dataset</td><td>Model</td><td>Floor</td><td>en-anch. floors</td><td>Bias</td><td>Share</td><td>Corrected</td><td>Thr.</td></tr><tr><td>MMLU-PROX</td><td>GPT-OSS-120B</td><td>0.209</td><td>0.191-0.253</td><td>+0.0042</td><td>3%</td><td>+0.116</td><td>0.030</td></tr><tr><td>MMLU-PROX</td><td>DeepSeek-V3.2</td><td>0.195</td><td>0.203-0.259</td><td>+0.0139</td><td>20%</td><td>+0.057</td><td>0.030</td></tr><tr><td>SCIBENCH</td><td>GPT-OSS-120B</td><td>0.373</td><td>0.368-0.397</td><td>+0.0117</td><td>9%</td><td>+0.124</td><td>0.015</td></tr><tr><td>SCIBENCH</td><td>DeepSeek-V3.2</td><td>0.317</td><td>0.327-0.358</td><td>+0.0238</td><td>29%</td><td>+0.059</td><td>0.015</td></tr><tr><td>STRATEGYQA</td><td>GPT-OSS-120B</td><td>0.200</td><td>0.182-0.268</td><td>-0.0055</td><td>13%</td><td>+0.046</td><td></td></tr><tr><td>STRATEGYQA</td><td>DeepSeek-V3.2</td><td>0.174</td><td>0.164–0.229</td><td>+0.0010</td><td>1%</td><td>+0.100</td><td></td></tr></table>

Table 24: The dissociation within and across languages. “Top-H not $\mathrm { t o p } { - } S ^ { \dagger }$ is the share of top-quartile-H instances not in the top quartile of S, quartiles within each pair; “cross-lingual” averages over the 21 cross lingual pairs of the same run. Mean S and H compare $( \mathrm { { e n } , \mathrm { { e n _ { p a r a } } ) } }$ (“within”) with the English-anchored cross-lingual pairs (“cross”).
<table><tr><td></td><td></td><td colspan="2">Top-H not top-S</td><td colspan="2">Mean S</td><td colspan="2">Mean H</td></tr><tr><td>Dataset</td><td>Model</td><td>within</td><td>cross-lingual</td><td>within</td><td>cross</td><td>within</td><td>cross</td></tr><tr><td>MMLU-PROX</td><td>GPT-OSS-120B</td><td>43.8%</td><td>54.8%</td><td>0.658</td><td>0.534</td><td>0.922</td><td>0.804</td></tr><tr><td>MMLU-PROX</td><td>DeepSeek-V3.2</td><td>52.9%</td><td>64.6%</td><td>0.609</td><td>0.543</td><td>0.923</td><td>0.813</td></tr><tr><td>SCIBENCH</td><td>GPT-OSS-120B</td><td>45.5%</td><td>63.8%</td><td>0.695</td><td>0.562</td><td>0.929</td><td>0.814</td></tr><tr><td>SCIBENCH</td><td>DeepSeek-V3.2</td><td>41.7%</td><td>62.4%</td><td>0.591</td><td>0.509</td><td>0.917</td><td>0.811</td></tr><tr><td>STRATEGYQA</td><td>GPT-OSS-120B</td><td>60.0%</td><td>50.5%</td><td>0.615</td><td>0.575</td><td>0.911</td><td>0.803</td></tr><tr><td>STRATEGYQA</td><td>DeepSeek-V3.2</td><td>41.2%</td><td>54.0%</td><td>0.608</td><td>0.507</td><td>0.897</td><td>0.790</td></tr></table>

## I Where Divergence Appears and What It Signals

This appendix gives the analyses behind §5.5. Both reuse the stored traces and step-pair matrices of the main runs; neither requires new inference.

## I.1 Positional Structure

Method. For every scored pair with English on one side we orient the calibrated step-pair matrix A so that its rows are the English steps, and score each English step by its best match anywhere in the other trace, max<sub>j</sub> $\mathbf { A } _ { i , j }$ , the quantity whose mean is the precision term of Eq. (5). Pairs with fewer than four English steps are dropped so that the first, middle, and last steps are distinct. The analysis covers all 18 evaluations: 31,277 English-anchored pairs, 192,243 English steps, and 5,630 questions. Confidence intervals are percentile bootstrap intervals over questions (2,000 resamples), since one question contributes six language pairs and each pair several steps. Because the best match carries no monotonicity constraint, “position” is position in the English trace: the fifth English step may match the second step of the other trace, and order is what τ measures. Every pair in this analysis reaches the same verified-correct answer in both languages.

Profile. Table 25 bins English steps by relative position into quintiles. Alignment declines toward the end of the trace in every benchmark, though by diferent routes: StrategyQA drops in mid-trace, SciBench rises through the middle before a sharp terminal drop, and MMLU-ProX declines gently throughout. All three end near 0.49.

Table 25: Mean best-match calibrated similarity of English steps by quintile of relative position in the English trace.
<table><tr><td></td><td> $\mathrm { q 1 }$ </td><td> ${ \bf q } 2$ </td><td> $\mathrm { q 3 }$ </td><td> $\mathbf { q } 4$ </td><td> $\mathbf { q } 5$ </td></tr><tr><td>All</td><td>0.556</td><td>0.561</td><td>0.551</td><td>0.531</td><td>0.492</td></tr><tr><td>STRATEGYQA</td><td>0.589</td><td>0.575</td><td>0.526</td><td>0.487</td><td>0.483</td></tr><tr><td>SCIBENCH</td><td>0.546</td><td>0.583</td><td>0.600</td><td>0.588</td><td>0.494</td></tr><tr><td>MMLU-PROX</td><td>0.539</td><td>0.529</td><td>0.528</td><td>0.515</td><td>0.497</td></tr></table>

The concluding step. Table 26 compares the first, the last, and the interior steps. The last step is the least aligned in every benchmark. The obvious mechanical reading is that one trace has more steps and its last step has no counterpart; if so, restricting to the 11,334 pairs with equal step counts should shrink the gap. It widens instead, in every benchmark. Per evaluation, the last-minus-middle gap is negative in 16 of 18, ranging from −0.137 (SciBench, DeepSeek-V3.2) upward; the two exceptions, MMLU-ProX with Qwen3-235B (+0.013) and StrategyQA with GPT-OSS-120B (+0.003), sit at zero rather than reversing.

Step length. Step length accounts for part of the terminal gap, by an amount that depends on the benchmark. Within terciles of English step length (taken within each evaluation), the last-minus-middle gap is −0.093, −0.048, and −0.021 from shortest to longest on StrategyQA, and −0.041, −0.013, and −0.009 on MMLU-ProX, where the longest tercile’s interval $\left( \left[ - 0 . 0 2 0 , + 0 . 0 0 4 \right] \right)$ includes zero. On SciBench it is flat (−0.067, −0.066, −0.065). The terminal gap is therefore not independent of step length.

Where traces part. We take as the branch point the first English step whose best match falls below a threshold. The threshold is arbitrary, so Table 27 reports three values. The levels move substantially with the threshold, but the ordering of the benchmarks does not: SciBench branches earliest and most often at the first step, StrategyQA latest, and MMLU-ProX between, at every threshold. We therefore interpret only the ordering.

By language. Table 28 gives the profile for each target language. The shape is the same in all six, a flat opening and a terminal decline, with the decline from the first to the last quintile confined to 0.052–0.078 and every interval clear of zero. The level difers by language, highest for Spanish and lowest overall for Hindi; we attach no mechanism to the ordering.

Scope. English-anchored pairs only, so the analysis says nothing about pairs between two non-English languages. It is descriptive: it locates where the emitted traces difer and licenses no causal or predictive conclusion.

## I.2 Dificulty and Correctness Outcomes

Scoring the unscored pairs. The main analyses score a language pair only when both languages answer correctly, so they cannot say what divergence looks like when correctness difers. The traces for the other outcomes are stored regardless of correctness, so for EN–ZH and EN–ES, across all six models and three benchmarks, we computed S, H, and τ for every pair with the pipeline’s own modules and each pair’s stored floor and ceiling, with no new inference and no re-calibration. On pairs scored by both routes the recomputation reproduces the stored values exactly. All groups are drawn from questions that passed the tracevalidity cascade, so trace-quality filtering is uniform across groups. The τ in this analysis is the no-tie variant of Appendix B.4; it is rank-correlated with the tie-aware score at Spearman 0.988 on a check set of 2,341 of these pairs, and we report only its rank-based efect sizes.

Table 26: First, interior (“middle”), and last English steps: mean best-match similarity and the last-minusmiddle gap with its 95% CI, over all pairs and over pairs with equal step counts (“eq.”).
<table><tr><td></td><td></td><td>first</td><td>middle</td><td>last</td><td>last-middle</td><td>95% CI</td></tr><tr><td>All</td><td>all</td><td>0.563</td><td>0.546</td><td>0.504</td><td>-0.042</td><td>[−0.046, -0.038]</td></tr><tr><td></td><td>eq.</td><td>0.591</td><td>0.573</td><td>0.515</td><td>-0.058</td><td>[−0.063, -0.053]</td></tr><tr><td>STRATEGYQA</td><td>all</td><td>0.596</td><td>0.534</td><td>0.494</td><td>-0.040</td><td>[−0.046, −0.034]</td></tr><tr><td></td><td>eq.</td><td>0.622</td><td>0.549</td><td>0.498</td><td>-0.051</td><td>[−0.057, −0.045]</td></tr><tr><td>SCIBENCH</td><td>all</td><td>0.536</td><td>0.580</td><td>0.513</td><td>-0.067</td><td>[−0.076, −0.059]</td></tr><tr><td></td><td>eq.</td><td>0.558</td><td>0.618</td><td>0.541</td><td>-0.077</td><td>[−0.089, -0.065]</td></tr><tr><td>MMLU-PROX</td><td>alī</td><td>0.551</td><td>0.522</td><td>0.508</td><td>-0.014</td><td>[−0.022, −0.008]</td></tr><tr><td></td><td>eq.</td><td>0.571</td><td>0.561</td><td>0.521</td><td>-0.040</td><td>[−0.049, -0.031]</td></tr></table>

Table 27: Branch points: mean relative position of the first English step whose best match falls below the threshold, and the share of branched pairs that branch at the first step.
<table><tr><td></td><td colspan="3">mean position</td><td colspan="3">share at first step</td></tr><tr><td>threshold</td><td>0.4</td><td>0.5</td><td>0.6</td><td>0.4</td><td>0.5</td><td>0.6</td></tr><tr><td>SCIBENCH</td><td>0.315</td><td>0.207</td><td>0.129</td><td>0.445</td><td>0.573</td><td>0.698</td></tr><tr><td>MMLU-PROX</td><td>0.383</td><td>0.253</td><td>0.141</td><td>0.274</td><td>0.451</td><td>0.654</td></tr><tr><td>STRATEGYQA</td><td>0.495</td><td>0.368</td><td>0.230</td><td>0.234</td><td>0.348</td><td>0.532</td></tr></table>

Table 28: Profile by target language (Englishanchored pairs): mean best-match similarity by quintile, and the q1−q5 decline with its 95% CI.
<table><tr><td></td><td> ${ \bf q } 1$ </td><td> $_ { \mathrm { q 2 } }$ </td><td>q3</td><td>q4</td><td> $\mathbf { q } 5$ </td><td> ${ \bf q 1 - q 5 }$ </td><td>[95% CI]</td></tr><tr><td>es</td><td>0.592</td><td>0.590</td><td>0.580</td><td>0.559</td><td>0.520</td><td>0.072</td><td>[0.065, 0.078]</td></tr><tr><td>zh</td><td>0.563</td><td>0.574</td><td>0.565</td><td>0.548</td><td>0.510</td><td>0.053</td><td>[0.047, 0.058]</td></tr><tr><td>ar</td><td>0.554</td><td>0.561</td><td>0.553</td><td>0.533</td><td>0.494</td><td>0.059</td><td>[0.053, 0.065]</td></tr><tr><td>ru</td><td>0.552</td><td>0.557</td><td>0.547</td><td>0.523</td><td>0.483</td><td>0.068</td><td>[0.063, 0.074]</td></tr><tr><td>vi</td><td>0.550</td><td>0.549</td><td>0.536</td><td>0.512</td><td>0.472</td><td>0.078</td><td>[0.072, 0.084]</td></tr><tr><td>hi</td><td>0.525</td><td>0.534</td><td>0.527</td><td>0.508</td><td>0.474</td><td></td><td>0.052 [0.046, 0.057]</td></tr></table>

Efect sizes. Table 30 reports Clif’s δ against the both-correct reference (positive: the reference is more similar). The contrast with pairs that share a wrong answer is the informative one, because there both languages give the same answer and difer only in whether it is right, so nothing about the answers themselves can produce a divergence gap. When exactly one language is correct, both S and H drop, as traces ending on diferent answers must difer somewhere; that contrast is partly mechanical and we report it for completeness. Per benchmark, the shared-wrong-answer contrast for S is +0.328 on StrategyQA (n = 118), +0.184 on SciBench (n = 34), and +0.289 on MMLU-ProX $( n = 7 3 )$ ; for H it is +0.046, +0.007, and −0.030.

Table 29: Correctness-outcome groups among EN– ZH and EN–ES pairs, with median S and H.
<table><tr><td>Group</td><td>n</td><td>median S</td><td>median H</td></tr><tr><td>Both correct (reference)</td><td>11,176</td><td>0.570</td><td>0.837</td></tr><tr><td>Both wrong, same answer</td><td>225</td><td>0.511</td><td>0.834</td></tr><tr><td>Both wrong, different answers</td><td>137</td><td>0.528</td><td>0.816</td></tr><tr><td>Exactly one correct</td><td>1,318</td><td>0.512</td><td>0.814</td></tr></table>

Dificulty. Item dificulty is the obvious common cause: hard questions plausibly produce both more variable reasoning and more errors. We index dificulty by k, the number of the seven languages that answer a question correctly. Among both-correct pairs, mean S rises with k, from 0.526 at k = 2 through 0.537, 0.536, 0.545, and 0.553 to 0.579 at k = 7 (Spearman $\rho = 0 . 0 9 9$ $p = 8 \times 1 0 ^ { - 2 6 }$ $n = 1 1 , 1 7 6 )$ . The two groups also have very diferent dificulty profiles: 85% of both-correct pairs come from questions all seven languages solve, whereas every shared-wrong-answer pair comes from a question with $k \leq 5$ (106, 59, 38, and 22 pairs at $k = 2 , \ldots , 5 )$ We therefore standardize the contrast to the dificulty distribution of the shared-wrong-answer pairs: within each stratum we compute δ against both-correct pairs of the same k and average with weights proportional to the number of shared-wrong-answer pairs, which is the comparison a design matched on dificulty makes. Matched on $k ,$ the $S$ contrast falls from +0.293 to +0.073, with an interval that includes zero, and matched on dataset and k to +0.016 (Table 30); H stays within noise of zero throughout. The adjustment is conservative: by construction a shared-wrong-answer pair has both English and the target language wrong, so $k \leq 5 ,$ while a both-correct pair has $k \geq 2$ , and conditioning on k removes part of the very correctness signal under test. The adjusted value is a lower bound rather than an unbiased estimate of a dificulty-free efect. Even so, the unadjusted contrast cannot be read as evidence that S tracks correctness beyond dificulty. What the data support is that low S marks hard items, on which shared errors concentrate.

Table 30: Clif’s δ against both-correct pairs, with 95% percentile bootstrap intervals over questions (2,000 resamples). “Matched” rows standardize the shared-wrong-answer contrast to that group’s dificulty distribution. The one-correct contrast is shown without intervals; it is partly mechanical.
<table><tr><td>Contrast</td><td colspan="3">S</td><td colspan="3">H</td><td colspan="3"> $\tau _ { \mathrm { n o - t i e } }$ </td></tr><tr><td>Same wrong answer</td><td>+0.293</td><td> $[ + 0 . 2 1 7 , + 0 . 3 6 6 ]$ </td><td></td><td>-0.027</td><td> $[ - 0 . 1 1 4 , + 0 . 0 5 5 ]$ </td><td></td><td>+0.082</td><td> $[ - 0 . 0 0 4 , + 0 . 1 6 9 ]$ </td><td></td></tr><tr><td>matched on k</td><td> $+ 0 . 0 7 3$ </td><td> $\left[ - 0 . 0 6 1 , \ : + 0 . 2 0 9 \right]$ </td><td></td><td>-0.107</td><td> $\left[ - 0 . 2 3 4 , + 0 . 0 1 9 \right]$ </td><td></td><td>-0.073 [</td><td></td><td> $- 0 . 2 0 2 , + 0 . 0 5 9 \ ]$ </td></tr><tr><td>matched on dataset, k</td><td>+0.016</td><td> $[ - 0 . 1 6 3 , + 0 . 2 0 2 ]$ </td><td></td><td>-0.040</td><td> $[ - 0 . 2 0 2 , + 0 . 1 5 6 ]$ </td><td></td><td>-0.084 </td><td> $- 0 . 2 6 3 , + 0 . 0 9 8 \ ]$ </td><td></td></tr><tr><td>Exactly one correct</td><td></td><td>+0.261</td><td></td><td></td><td>+0.198</td><td></td><td></td><td>+0.228</td><td></td></tr></table>

Scope. The shared-wrong-answer group is small and unusual: questions on which English and the target language both fail while at least

Review prioritization. Among the 11,401 pairs whose two languages give the same answer, 225 (1.97%) are wrong, and no answerconsistency check can surface any of them. We rank pairs within their own evaluation and language pair, so that diferences in mean similarity between cells do not stand in for the ordering a reviewer would use, and review the least similar first. Table 31 reports the share of the 225 found at a 25% review budget. Intervals are percentile bootstrap over questions with the ranking recomputed inside each resample, and S and H are evaluated on the same resamples so that their diference has a paired interval. This is the same association as above, expressed as a quantity a reviewer can act on, not independent evidence. At a base rate under 2%, precision within the reviewed quarter stays low, so the result concerns where to spend a fixed review budget and does not license a screening threshold.

Table 31: Shared wrong answers found when the least similar 25% of answer-agreeing pairs are reviewed.
<table><tr><td>Ranking</td><td>found at 25% budget</td></tr><tr><td>structural S</td><td>44.0% [35.9, 51.3]</td></tr><tr><td>holistic H</td><td>28.9% [21.7, 35.1]</td></tr><tr><td>random</td><td>25.0%</td></tr><tr><td>paired difference  $S - H$ </td><td>+15.1 points  $[ + 6 . 2 , + 2 3 . 9 ]$ </td></tr></table>

two other languages succeed. The analysis is limited to two English-anchored language pairs and to questions that passed the validity cascade, and it is correlational throughout. τ is the least stable of the three metrics here, as under encoder swaps (Appendix G): its shared-wronganswer contrast is positive on StrategyQA (+0.102) and MMLU-ProX (+0.122) but negative on SciBench (−0.095), so we do not rely on it.