# SITAR: A SITUATIONAL DISTORTION METRIC FOR AUTOMATIC SPEECH RECOGNITION EVALUATION

Mingzhe Lu<sup>1,2</sup>, Jinhao Liu<sup>1,2</sup>, Yunpeng Li<sup>1,2,\*</sup>, Hang Su<sup>3,\*</sup>, Lichun Fan<sup>3</sup>, Jian Luan<sup>3</sup>

<sup>1</sup>Institute of Information Engineering, Chinese Academy of Sciences, Beijing, China <sup>2</sup>School of Cyber Security, University of Chinese Academy of Sciences, Beijing, China <sup>3</sup>Xiaomi Inc.

## ABSTRACT

In automatic speech recognition (ASR), transcription errors with similar lexical or semantic discrepancies can have markedly different consequences for the situation conveyed by speech. Capturing these differences requires not merely contextual assessment, but an understanding of what the spoken discourse describes, establishes, or seeks to bring about. However, relying on LLM judges introduces computational cost and scoring variability. We introduce SITAR, a situational distortion metric for ASR, realized through a comprehensive end-to-end construction framework. Specifically, this framework aggregates situational severity and cumulative extent into a unified score via a bounded recursive formulation. Using targets derived from this formulation with limited offline LLM support, we train SITAR as a lightweight BERT-family regressor that scores reference–hypothesis pairs in a single forward pass, entirely free of any prohibitive runtime costs. Recognizing that introducing a new metric demands rigorous justification, we formalize a principled validation protocol spanning both extrinsic validity, assessed at macro and micro levels, and intrinsic scorer reliability. The former, evaluated across diverse datasets and models, demonstrates that SITAR reliably tracks broad capability scaling while providing orthogonal diagnostic utility, whereas the latter, examined on a purpose-built controlled benchmark, confirms the operational soundness of the scoring instrument itself. Through systematic validation, we demonstrate that SITAR holds strong potential as a practical, lightweight complement to existing ASR metrics. We will release the full recipes (code, model weights, dataset, etc.) after the internal review to support community research.

## 1 INTRODUCTION

Automatic speech recognition (ASR) is pivotal to modern human-computer interaction, yet even small transcription errors can cause downstream failures in conversational voice assistants (Harvill et al., 2024). Beyond architectural improvements, meaningful progress fundamentally hinges on evaluation metrics that faithfully reflect transcription quality (Aksenova et al., 2021). Although¨ modern LLM-based ASR metrics align more closely with human judgments (Parulekar & Jyothi, 2025), the prompt-dependent scoring variability (Hong et al., 2026) and prohibitive computational costs of frequent LLM calls (Huang et al., 2026) hinder their widespread adoption as standard metrics. This presents a threefold imperative for next-generation metrics: to effectively harness the understanding capabilities of modern LLMs, to operate independently of them at inference time, and to deliver deterministic scores that make any two transcription errors comparable on a unified scale. Beyond these operational criteria, and aiming for closer alignment with human judgments, we present a novel target property for transcription evaluation, termed situational distortion.

Across the evolution of ASR metrics, the conceptualization of transcription fidelity has broadened from verbatim correspondence to semantic consistency and, more recently, alignment with human judgments, driving a methodological shift from edit-based error counting toward embedding-based metrics and large language model evaluators. Despite providing a well-defined and deterministic error metric, WER remains indifferent to the meaning carried by language. Subsequent metrics address this limitation at differing levels of granularity, with NE-WER prioritizing micro-level entity components (Garofolo et al., 1999) and SemDist assessing macro-level semantic representations (Kim et al., 2021). However, a gap persists between semantic proximity and alignment with human judgments. For example, a single-token shift from “depart at fifteen” to “depart at fifty” leaves almost all of the wording unchanged but alters the specified departure time. AER therefore probes comprehension fidelity through task-driven question answering.

Yet, AER (Pulikodan et al., 2025) introduces fresh dilemmas. Although question answering exposes errors invisible to semantic metrics, AER lacks a calibrated scale for comparing transcriptions that lead to failures on entirely different questions and requires repeated LLM queries. LASER (Parulekar & Jyothi, 2025) also explores fine-tuning a smaller LLM to reduce inference costs. Its rubric assigns fixed penalties to word-level error categories and normalizes their sum by reference length. Although these categories distinguish meaning-preserving from meaning-changing errors, they do not explicitly grade their situational consequences. More fundamentally, although these metrics introduce fine-grained distinctions within their respective basic units, their assessment of the entire reference–hypothesis pair remains a statistical aggregation of locally defined judgments rather than a unified measure of overall transcript distortion.

To develop modern ASR metrics aligned with human judgments, constantly scrambling to address these disparate dilemmas through piecemeal fixes only traps us in treating isolated symptoms. This recognition urges us instead to step back and re-examine the structural foundations of transcription evaluation. Laying these foundations requires treating evaluation as a unified measurement system rather than a collection of heuristics. Specifically, its design must answer four core questions: what property is being measured, against what reference deviations are defined, on what scale their values become comparable, and through what instrument those values are obtained.

Thus, we present SITAR, an ASR metric that quantifies reference-grounded situational distortion on a common scale and is realized as a lightweight, deterministic scorer. Constructing SITAR proceeds through these four elements as a coherent sequence, with each establishing the basis for the next. Specifically, we define the target property as the distortion that a hypothesis introduces into the situation conveyed by the spoken source utterance. We establish the reference condition by treating the reference as the gold textual realization of that utterance and taking h = r as zero distortion, with our proposed Constrained Gap Delineation (CGD) identifying paired gaps as elementary deviation units. These units are first assessed relative to their own reference, yielding reference-specific deviation levels that are then projected onto a common scale according to how they distort the situation conveyed by the spoken source utterance, that is, how they change what that utterance describes, establishes, or seeks to bring about. A bounded recursive formulation subsequently integrates situational severity with cumulative extent into scalar targets. Finally, these targets, constructed offline with LLM-assisted judgments, are used to train the operational instrument, namely SITAR itself, a lightweight BERT-family regressor requiring no LLM calls at scoring time.

The contributions of this paper can be summarized as follows:

• We introduce SITAR, a lightweight, deterministic ASR metric that quantifies referencegrounded situational distortion in a single forward pass, eliminating any LLM involvement during practical use while achieving closer alignment with human judgments.

• We formulate a principled, measurement-grounded methodology to construct SITAR, establishing an explicit reference anchor, CGD-based deviation delineation, a common scale for reference-specific deviation levels, and a bounded recursive formulation that unifies situational severity with cumulative extent.

• We comprehensively validate SITAR through broad cross-dataset and cross-model evaluations, complemented by a purpose-built controlled benchmark, demonstrating its capacity to track model capability trends, reveal fine-grained diagnostic distinctions, and maintain within-reference severity ordering and cross-reference score consistency.

## 2 RELATED WORK

With recent developments in ASR evaluation, we consider it informative to examine literature not only methodologically, regarding how transcription quality is computationally measured and operationalized, but also conceptually, regarding how transcription fidelity is understood and defined.

Methodological Evolution. At the lexical level, Word Error Rate (WER) remains the de facto standard (Morris et al., 2004) despite recognized limitations (Aksenova et al., 2021), with Character¨ Error Rate (CER) as its character-level counterpart (K et al., 2025). Both rely on optimal alignment under uniform edit penalties. However, minimum-edit-distance alignment can be linguistically implausible for splits, merges (Kuhn et al., 2024), or homophonic spans (Ruiz & Federico, 2015), motivating relaxed alignment constraints (Borgholt et al., 2026), while uniform penalties overlook error severity, prompting weighted variants based on task relevance (Nanjo et al., 2005) or human perception (Mishra et al., 2011). Yet these edit-based paradigms remain tied to surface-form discrepancies. Shifting toward semantic preservation, representational metrics compare sentence-level embeddings, as in SemDist (Kim et al., 2021), while Aligned Semantic Distance (Rugayan et al., 2022) introduces token-level alignment to expose local discrepancies. Representational similarity cannot fully replace lexical signals. BERTScore and WER jointly fit expert ratings of disordered speech better than either alone (Tobin et al., 2022), supporting hybrid metrics that blend semantic similarity with edit penalties, including $H _ { \mathrm { e v a l } }$ (Sasindran et al., 2023) and SeMaScore (Sasindran et al., 2024). Phonetic extensions further target intelligibility (Phukon et al., 2025). Still, representational similarity alone does not determine practical consequences across diverse contexts, prompting domain-specific weighting for clinically relevant errors (Shor et al., 2023) and driving the turn toward the LLM-based paradigm. Enabled by large-scale pretraining (Kaplan et al., 2020; Hoffmann et al., 2022) and in-context learning (Brown et al., 2020), LLM evaluators directly assess error severity, semantic disruption, and acceptable variation, as in LASER (Parulekar & Jyothi, 2025). They also support intent-aware probes, with AER (Pulikodan et al., 2025) using questionanswering accuracy as a proxy for fidelity and S<sup>2</sup>ER (Jiang et al., 2026) evaluating the preservation of core intent and meaning-bearing content. Nevertheless, prompt-based scoring introduces instability across rubric perturbations (Hong et al., 2026) and repeated runs (Haldar & Hockenmaier, 2025), while prohibitive inference latency and computational cost restrict scalability, motivating selective invocation of LLM judges (Huang et al., 2026).

Conceptual Perspectives. Under the surface view, fidelity is defined through strict verbatim correspondence, yet lexical discrepancies do not consistently translate into downstream comprehension failures. Lower slot-understanding error can coexist with higher WER (Wang et al., 2003), and errors involving key actions or entities can prevent a voice assistant from fulfilling a request even when other lexical errors leave the intended action intact (Harvill et al., 2024). In practical decision audits, WER poorly tracks how effectively humans extract decisions from transcripts (Favre et al., 2013). The semantic view therefore assesses whether asserted or implied content is preserved, drawing on semantic textual similarity (Agirre et al., 2012) and natural language inference (Bowman et al., 2015). However, equivalent propositional content can function differently across discourse contexts or conversational positions. In clinical dialogue, standard surface and semantic metrics correlate poorly with clinician-assigned risk labels (Ellis et al., 2026). This motivates our situational perspective, grounded in cognitive situation models where comprehenders integrate events and entities into an evolving representation of the described state of affairs (Zwaan & Radvansky, 1998), as well as communicative grounding through which interlocutors establish mutual understanding sufficient for their current purposes (Clark & Brennan, 1991). Bridging these cognitive and interactive views, we define the situation relevant to ASR evaluation in terms of what the spoken source utterance describes, establishes, or seeks to bring about, and evaluate transcription errors by the degree to which they distort this situation.

## 3 METHODOLOGY

Let $r \in \mathcal { R }$ denote the reference transcript of the raw audio and $h \in \mathcal H$ the ASR hypothesis. Figure 1 exemplifies the operational workflow of SITAR on a concrete instance, with each procedural stage detailed below.

## 3.1 REFERENCE CONDITION

The primary prerequisite for reliable scoring is establishing an unambiguous, well-bounded unit of comparison across a reference r and a hypothesis h. Its boundaries must respect the informational structure of the source utterance, rendering word-level Levenshtein alignment ill-suited for defin ing such units (Ruiz & Federico, 2015; Borgholt et al., 2026). While LLMs can delineate such units, costly inference limits scalability and variable outputs undermine reproducibility. We therefore introduce Constrained Gap Delineation (CGD), a deterministic algorithm that locates text discrepancies using shared text as anchors and consolidates adjacent differences into paired gaps $g _ { i }$ . Each gap pairs an offset-defined character span from r with one from $h ,$ either of which may be empty, and serves as an elementary unit for subsequent situational scoring. We begin by normalizing r and h into word sequences $\tilde { r }$ and $\tilde { h } ,$ , respectively, through lowercasing and punctuation removal, while retaining mappings from normalized tokens to their original character spans. In what follows, we elaborate on the algorithmic details of CGD.

![](images/20af9221e090244a72ac32bb7fdef720e6b3aaf9a95973b0c1ba50d6c76cf77a.jpg)  
Figure 1: Overview of SITAR construction: CGD extracts paired gaps from transcripts, which are annotated for situational severity and aggregated into supervision targets to train the LLM-free scorer.

Decomposition. To identify anchors, we recursively process paired regions $( u , v )$ , beginning with $( \tilde { r } , \tilde { h } )$ . Within each region, we set $n _ { u } = | u |$ and $n _ { v } = | \boldsymbol { v } |$ and identify maximal common word blocks extension using a match-length dynamic program: $L _ { p , q }$ denotes the number of consecutive matches starting at $u _ { p }$ and $v _ { q } ,$ with $L _ { p , q } = 1 + L _ { p + 1 , q + 1 }$ if $u _ { p } = v _ { q }$ and 0 otherwise, for $0 \le p < n _ { u }$ and $0 \leq q < n _ { v }$ . The boundary values are $L _ { n _ { u } , q } = 0$ for $0 \leq q \leq n _ { v }$ and $L _ { p , n _ { v } } = 0$ for $0 \leq p \leq n _ { u }$ Requiring left-maximality with a minimum block length $\ell _ { \mathrm { m i n } }$ gives the candidate set

$$
\begin{array} { r } { \mathcal { B } = \{ ( p , q , L _ { p , q } ) : L _ { p , q } \geq \ell _ { \mathrm { m i n } } , p = 0 \vee q = 0 \vee u _ { p - 1 } \neq v _ { q - 1 } \} . } \end{array}\tag{1}
$$

We set $\ell _ { \mathrm { m i n } } = 2$ to suppress spurious isolated matches, especially on function words such as the and $t o ,$ and recompute $\bar { B }$ with $\ell _ { \mathrm { m i n } } = 1$ if it is empty. If B remains empty, no anchor is selected for the current region. Otherwise, we sort the candidates by $( p , q , - \ell )$ into indexed triples $( p _ { i } , q _ { i } , \ell _ { i } )$ . A second monotonic-chain dynamic program then selects an ordered, nonoverlapping chain with the greatest cumulative matched length. Let $f _ { i }$ denote the maximum cumulative matched length of such a chain ending at candidate i:

$$
f _ { i } = \ell _ { i } + \operatorname* { m a x } \left( \{ f _ { j } : j < i , \ : p _ { j } + \ell _ { j } \leq p _ { i } , \ : q _ { j } + \ell _ { j } \leq q _ { i } \} \cup \{ 0 \} \right) .\tag{2}
$$

The optimal chain is recovered by backtracking from the earliest index in arg max $f _ { i } ,$ , with predecessor ties likewise resolved in favor of the earliest candidate in the sort order.

We then filter the matching word-position pairs $( p , q )$ in the selected chain using forward and backward edit-distance dynamic programs. Let $E _ { a , b } ^ { + }$ and $E _ { a , b } ^ { - }$ denote the resulting unit-cost prefix and suffix edit distances for $( u [ : \ a ] , v [ : \ b ] )$ and $( u [ a \ : ] , v [ b \ : ] )$ , respectively. Both use unit costs for insertion and deletion and substitution cost $\delta _ { a , b } = \mathbf { 1 } [ u _ { a } \neq v _ { b } ]$ ], with boundary values $E _ { a , 0 } ^ { + } = a$ $E _ { 0 , b } ^ { + } = b , E _ { a , n _ { v } } ^ { - } = n _ { u } - a$ , and $E _ { n _ { u } , b } ^ { - } = n _ { v } - b$ . Because a matching pair satisfies $u _ { p } = v _ { q }$ , it lies on a minimum-cost edit path if and only if

$$
E _ { p , q } ^ { + } + E _ { p + 1 , q + 1 } ^ { - } = E _ { n _ { u } , n _ { v } } ^ { + } .\tag{3}
$$

Retained matching pairs contiguous in both sequences form anchor blocks. We recursively repeat this selection and filtering in the paired subregions before, between, and after these blocks. Recursion terminates in subregions with no retained matches; terminal subregions with at least one nonempty side form the ordered atomic gaps $g _ { t } ^ { \mathrm { a t o m } } = ( \tilde { r } [ I _ { t } ^ { r } ] , \tilde { h } [ I _ { t } ^ { h } ] )$ , where $I _ { t } ^ { r }$ and $I _ { t } ^ { h }$ are zerobased word intervals in $\tilde { r }$ and ${ \tilde { h } } ,$ respectively.

Consolidation. Atomic gaps can over-segment a coherent local deviation when opportunistic matches split it into multiple fragments. CGD therefore complements lexical matching with linguistic constraints, consolidating adjacent gaps into elementary deviation units. Specifically, given atomic gaps $g _ { 0 } ^ { \mathrm { a t o m } } , \ldots , g _ { M - 1 } ^ { \mathrm { a t o m } }$ , we identify those with at most three tokens per side as candidate fragments, and examine whether adjacent gaps can merge with them into an elementary deviation unit. Such consolidation is permitted only when intervening matches do not cross informational boundaries and the resulting envelope spans at most one predicate per side. For example, matching the shared word search splits a coherent local deviation into two atomic gaps, which CGD consolidates:

$$
\begin{array} { r l } & { \underbrace { g _ { i } ^ { \mathrm { a t o m } } } _ { r : \mathrm { O k a y , s o } \Gamma ^ { \mathrm { m } } \mathrm { m j u s t } } \underbrace { g _ { i } ^ { \mathrm { a t o m } } } _ { \mathrm { e l e m e n t a r y d e v i a t i o n } } \underbrace { g _ { i + 1 } ^ { \mathrm { a t o m } } } _ { \mathrm { t h e \ a p p } } \underbrace { } _ { \mathrm { s o \ I \ c a n s e \ t h a t \ w e \ d o \ h a v e \ a \ f i i g h t . } } _ { \mathrm { c h e r m e n t a r y d e v i a t i o n } } } \\ & { \underbrace { g _ { i + 1 } ^ { \mathrm { a t o m } } } _ { h : \mathrm { O k a y , \ s o \Gamma ^ { \mathrm { m } } j u s i t } } \underbrace { g _ { i + 1 } ^ { \mathrm { a t o m } } } _ { \mathrm { e l e m e n t a r y d e v i a t i o n } } \underbrace { g _ { i + 1 } ^ { \mathrm { a t o m } } } _ { \mathrm { e l e a n \ s e \ t h a t \ w e \ d o \ h a v e \ a \ f i i g h t . } } } \end{array}
$$

Let $A _ { i }$ denote the endpoint indices $j \in \{ i , \dotsc , M - 1 \}$ for which $g _ { i } ^ { \mathrm { a t o m } } , \ldots , g _ { i } ^ { \mathrm { a t o m } }$ forms an admissible group, where singleton feasibility guarantees $i \in \mathcal A _ { i }$ . CGD then computes a minimumcardinality partition of the atomic-gap sequence by dynamic programming over suffixes. Let $d _ { i }$ denote the minimum number of groups covering $g _ { i } ^ { \mathrm { a i o m } } , \ldots , g _ { M - 1 } ^ { \mathrm { a t o m } }$ . With $d _ { M } = 0$ , the recurrence is

$$
d _ { i } : = 1 + \operatorname* { m i n } _ { j \in { \cal A } _ { i } } d _ { j + 1 } , \qquad i = M - 1 , \ldots , 0 .\tag{4}
$$

The optimal partition is decoded forward from $i = 0$ , with ties broken in favor of the largest endpoint. Retained character-offset mappings then project each consolidated group back to the original transcripts, yielding the ordered paired gaps $g _ { 0 } , \ldots , g _ { m - 1 }$

For an original transcript pair $( r , h )$ , let $g _ { 0 } , \ldots , g _ { m - 1 }$ denote the paired gaps identified by CGD. For each $g _ { i } ,$ , we construct a complete single-gap hypothesis $\it { h ^ { ( i ) } }$ . It retains the hypothesis-side content of $g _ { i }$ , while all other gaps are restored with their reference-side content. This produces m single-gap pairs, $( r , h ^ { ( 0 ) } ) , \ldots , ( r , h ^ { ( m - 1 ) } )$ , all sharing the same reference.

## 3.2 COMMON SCALE

The distortion caused by each $g _ { i }$ reflects how strongly $\it { { h ^ { ( i ) } } }$ alters the information conveyed by $^ { r } \cdot$ When $\it { h ^ { ( i ) } }$ induces less distortion than $h ^ { ( j ) }$ , we denote $g _ { i } \prec ^ { ( r ) } g _ { j }$ . When their impacts are approximately equal, we denote $g _ { i } \approx ^ { ( r ) } g _ { j }$ . Anchoring this relation at the zero-deviation baseline $g _ { \emptyset }$ yields a reference-specific ordering of the schematic form:

$$
g _ { \emptyset } \prec ^ { ( r ) } \dots \prec ^ { ( r ) } g _ { i } \prec ^ { ( r ) } \dots \prec ^ { ( r ) } g _ { j } \approx ^ { ( r ) } g _ { k } \prec ^ { ( r ) } \dots , \qquad \mathrm { w h e r e ~ } g _ { \emptyset }  \big ( h ^ { ( \emptyset ) } \equiv r \big ) .\tag{5}
$$

However, a general metric must make transcription deviations comparable across hypotheses produced by any ASR model for any reference. Although existing metrics, exemplified by lexical edit distance and representational similarity, yield common numerical scales, their implicit priors impose an intrinsic ceiling on evaluation, restricting each metric to a partial view of the situation conveyed by the reference. Any common scale necessarily abstracts, but where that abstraction stops determines the highest level of change the scale can register, as in WER, whose defining advantage in lexical fidelity serves as the very barrier blinding it to higher-level semantic consequences.

We seek comparability not by impoverishing the object of measurement, but through a shared structure capable of preserving effects across distinct levels. Results-chain frameworks<sup>1</sup> provide such a structure by organizing results into linked tiers and treating the links among them as integral to evaluation. Specifically, they distinguish three progressively broader levels, namely changes confined to the immediate result, changes that alter subsequent outcomes, and changes that lead to higher-level consequences, whether intended or unintended. When it comes to a spoken utterance, its effects may arise through what it describes, establishes, or seeks to bring about.

Across these functions, we argue that transcription deviations can be distinguished by whether they remain confined to expression, alter uptake without changing the intended outcome, or affect the ultimate outcome. These distinctions lead us to define three foundational consequence tiers, formalizing an evaluation perspective we term situational distortion. At ${ \mathcal { T } } _ { 0 } .$ , the situation is preserved despite differences in expression. $\mathrm { A t } \ \mathcal { T } _ { 1 }$ , uptake changes while the intended outcome is preserved. $\mathrm { A t } \ \bar { \mathcal { T } } _ { 2 }$ , the ultimate outcome changes, including consequences for decisions and execution. These tiers are deliberately coarse-grained to reduce ambiguity in severity assignment. Finer resolution is subsequently introduced through directional boundary cases and tier-wise gap counts, yielding a composite score that captures both the severity and extent of distortion.

We assign nominal scores 0, 1, 2 to $\mathcal { T } _ { 0 } , \mathcal { T } _ { 1 } , \mathcal { T } _ { 2 } ,$ respectively. Between adjacent tiers $\mathcal { T } _ { i }$ and $\mathcal { T } _ { i + 1 }$ , we distinguish two directional boundary cases: $\mathcal T _ { i } \to \mathcal T _ { i + 1 }$ and $\mathcal T _ { i + 1 } \to \mathcal T _ { i }$ . Here, the starting tier gives the gap’s nominal assignment, while trisecting the unit interval gives a $1 / 3$ severity adjustment in the arrow’s direction. For a transcript pair with at least one gap, the highest severity position among its gaps defines the base score $\beta \colon$

$$
\beta \in \left\{ 0 , 0 + 1 / 3 , 1 - 1 / 3 , 1 , 1 + 1 / 3 , 2 - 1 / 3 , 2 \right\} .\tag{6}
$$

Each $\beta$ anchors a nonoverlapping interval $[ \beta , \beta { + } 1 / 3 )$ reserved for refinement by cumulative extent, yielding an overall scale of $[ \bar { 0 } , 2 ^ { - } + 1 / 3 )$

To capture cumulative extent, let $c _ { k }$ count gaps assigned to nominal tier $\mathcal { T } _ { k }$ , for $k \in \{ 0 , 1 , 2 \}$ . Here, $k _ { \mathrm { m a x } } = \operatorname* { m a x } \{ k : c _ { k } > 0 \}$ identifies the nominal tier of the most severe gap, whereas $\beta$ records its exact severity position, including any directional boundary adjustment. Quantity refinement counts gaps by nominal tier irrespective of boundary adjustments, to avoid unnecessary combinatorial complexity. Since $\beta$ already accounts for one gap in $\mathcal { T } _ { k _ { \mathrm { m a x } } }$ , refinement begins with the second gap at that tier and includes all gaps at lower tiers, yielding counts $\bar { c } _ { k } = c _ { k } - \mathbf 1 [ k = k _ { \operatorname* { m a x } } ]$ . Within each basescore interval, this refinement preserves tier priority: of two distinct count configurations, the one with more gaps at the highest tier where they differ receives the higher score. For a fixed parameter $0 < \rho < 1$ controlling saturation, we define the score as

$$
S ( \boldsymbol r , h ) = \beta + ( 1 / 3 ) \mathcal Q _ { \rho } \big ( \bar { c } _ { 0 } , \ldots , \bar { c } _ { k _ { \operatorname* { m a x } } } \big ) .\tag{7}
$$

Here, $\mathcal { Q } _ { \rho }$ aggregates counts recursively from lower to higher tiers, with $x _ { j }$ denoting the nonnegative integer refinement count at tier $j \colon$

$$
\begin{array} { r l r } { \mathcal { Q } _ { \rho } ( \emptyset ) = 0 , } & { { } } & { \mathcal { Q } _ { \rho } ( x _ { 0 } , \ldots , x _ { k } ) = 1 - \rho ^ { x _ { k } + \mathcal { Q } _ { \rho } ( x _ { 0 } , \ldots , x _ { k - 1 } ) } . } \end{array}\tag{8}
$$

In particular, $\mathcal { Q } _ { \rho } ( x _ { 0 } ) = 1 - \rho ^ { x _ { 0 } }$ . For finite counts, $0 \leq \mathcal { Q } _ { \rho } < 1$ , ensuring that $S ( r , h ) \in [ \beta , \beta + 1 / 3 )$ This bound, combined with integer counts and monotone recursive updates, preserves tier priority while allowing lower-tier gaps to refine the score. A notable case is benign speech disfluencies, where multiple $\mathcal { T } _ { 0 }$ gaps yield a positive but saturating refinement, yet preserve the situation. $\mathsf { A p - }$ pendix A.1 provides the parameter derivation, proofs, and worked examples.

## 3.3 OPERATIONAL INSTRUMENT: SITAR

In practice, we first rank the single-gap pairs $( r , h ^ { ( i ) } )$ to establish a reference-specific ordering of the form shown in Eq. 5, and subsequently assign each a severity tier independently. Decoupling these steps provides a self-consistency check within each round, and any contradiction between the ranking and tier assignments in any round leads to rejection of the entire transcript pair. Specifically, both the ranking and tier annotations are generated by DeepSeek-V4-Pro across three independent rounds to mitigate stochasticity.

At this stage, each single-gap hypothesis $\it { h ^ { ( i ) } }$ receives three candidate tier assignments generated by DeepSeek, $\{ t _ { \mathrm { D S } } ^ { ( i , k ) } \} _ { k = 1 } ^ { 3 }$ . For each $\it { { h ^ { ( i ) } } }$ , we obtain the majority tier $t _ { \mathrm { m a j } } ^ { ( i ) }$ across the three rounds; if any hypothesis in a transcript pair lacks such a majority, the entire pair is discarded. Qualified pairs then undergo quality review by a four-agent committee using GPT-5.6-Sol-Max in ultra mode, which assigns a reviewed tier $t _ { \mathrm { r e v } } ^ { ( i ) }$ to each hypothesis. We apply a second quality filter, discarding the pair if any constituent hypothesis violates the consistency condition

$$
\begin{array} { r l r } { | t _ { \mathrm { m a j } } ^ { ( i ) } - t _ { \mathrm { r e v } } ^ { ( i ) } | \le 1 } & { { } \wedge } & { \left( t _ { \mathrm { d i s } } ^ { ( i ) } \mathrm { ~ i s ~ u n d e f i n e d ~ } \vee ~ | t _ { \mathrm { m a j } } ^ { ( i ) } - t _ { \mathrm { d i s } } ^ { ( i ) } | \le 1 \right) , } \end{array}\tag{9}
$$

where $t _ { \mathrm { d i s } } ^ { ( i ) }$ denotes the unique dissenting DeepSeek assignment if one exists.

For each retained hypothesis, $t _ { \mathrm { r e v } } ^ { ( i ) }$ sets the nominal score, while the sign of $t _ { \mathrm { m a j } } ^ { ( i ) } - t _ { \mathrm { r e v } } ^ { ( i ) }$ determines the directional boundary adjustment. The maximum adjusted position within each transcript pair defines its base score $\beta .$ After excluding one hypothesis that attains this maximum, we count the remaining hypotheses by their GPT-reviewed tiers to obtain the refinement counts $\bar { c } _ { k } .$ . Substituting $\beta$ and these refinement counts into Eq. 7 yields the deterministic supervision target $S ( r , h )$

For scalable, LLM-free evaluation with low latency, we amortize this multi-stage pipeline into a lightweight regressor $f _ { \theta } ( r , h ) \in \mathbb { R }$ . Specifically, we implement $f _ { \theta }$ using DeBERTa-v3-large (He et al., 2023) and optimize it on reference-hypothesis pairs using exclusively the scalar supervision targets $S ( r , h )$ via three complementary objectives:

$$
\mathcal { L } ( \theta ) = \lambda _ { \mathrm { m s e } } \mathcal { L } _ { \mathrm { m s e } } + \lambda _ { \mathrm { b o u n d a r y } } \mathcal { L } _ { \mathrm { b o u n d a r y } } + \lambda _ { \mathrm { o r d i n a l } } \mathcal { L } _ { \mathrm { o r d i n a l } } .\tag{10}
$$

In this joint objective, $\mathcal { L } _ { \mathrm { m s e } }$ regresses toward the target scores, while $\mathcal { L } _ { \mathrm { b o u n d a r y } }$ penalizes predictions outside the score interval containing the target. Additionally, $\mathcal { L } _ { \mathrm { o r d i n a l } }$ (Niu et al., 2016) supervises an auxiliary head to predict whether the target score reaches each predefined threshold. During inference, $f _ { \theta }$ evaluates any reference-hypothesis pair in a single forward pass, and we clip and normalize the raw prediction to [0, 1] such that lower values indicate less situational distortion:

$$
\widehat { S } ( r , h ) = \frac { \mathrm { c l i p } \left( f _ { \theta } ( r , h ) , 0 , 7 / 3 \right) } { 7 / 3 } .\tag{11}
$$

## 4 EXPERIMENTS

To justify a new evaluation metric alongside established baselines, we evaluate SITAR along two complementary axes of extrinsic validity. At the macro level, the metric must faithfully track broad developmental progress with systematically decreasing distortion. At the micro level, it must offer orthogonal diagnostic insights, differentiating nuanced error behaviors that conventional metrics treat identically. Finally, we evaluate the intrinsic reliability of the trained SITAR regressor itself.

## 4.1 BENCHMARK SETUP

SITAR-specific. The supervision corpus originates primarily from AppTek Beck et al. (2026), with transcripts generated by four ASR models. Through quality filtering, FLEURS error augmentation, and selective gap-correction synthesis on multi-gap instances, we construct a supervision dataset of 12,347 pairs. Data construction and scorer training are detailed in Appendices A.2 and A.3, respectively.

Cross-domain. We evaluate SITAR across domains and model families using the FLEURS English test split (Conneau et al., 2023), LibriSpeech (Panayotov et al., 2015), and VoxPopuli English (Wang et al., 2021), with five Whisper variants (Radford et al., 2023) ranging from tiny to large alongside the base and large variants of wav2vec 2.0 (Baevski et al., 2020). Among these datasets and models, only a portion of the FLEURS training split and selected erroneous Whisper-large-v3 transcriptions contributed to supervision; LibriSpeech, VoxPopuli, and outputs from the other evaluated models were excluded. We compare against five baselines: WER and Named Entity WER, abbreviated as NE-WER (Garofolo et al., 1999), for lexical overlap; BERTScore<sub>F1</sub> (Zhang et al., 2020) and SemDist (Kim et al., 2021) for semantic similarity; and AER for task-driven comprehension. For AER, we use the scores reported by Pulikodan et al. (2025).

## 4.2 EXTRINSIC METRIC VALIDITY

Macro-Level. Intuitively, as an ASR system generates more accurate transcriptions, evaluation metrics should broadly reflect this improvement. Model scaling serves as a natural testbed, leading us to evaluate Whisper across varying sizes. As capacity expands from tiny to large, transcription performance tends to improve, as reflected by existing metrics such as WER, SemDist, and AER in Table 1. SITAR follows the same trend across all three benchmarks and both Whisper and wav2vec2. These evaluations include LibriSpeech and VoxPopuli, which were absent from its supervision data, alongside Whisper variants from tiny to medium and wav2vec2, whose outputs were not used for supervision. For example, on VoxPopuli, SITAR decreases from 0.4084 for wav2vec2-base to 0.3577 for wav2vec2-large.

Table 1: Comparison of ASR evaluation metrics across datasets.
<table><tr><td>Model Name</td><td>Dataset</td><td>WER</td><td>NE-WER</td><td>BERTScoreF1</td><td>SemDist</td><td>AER</td><td>SITAR</td></tr><tr><td rowspan="6">whisper-tiny whisper-base whisper-small whispèr-medium</td><td rowspan="6">fleurs_en</td><td>0.1446</td><td>0.6500</td><td>0.9437</td><td>0.0711</td><td>0.2478</td><td>0.3872</td></tr><tr><td>0.1150</td><td>0.5500</td><td>0.9516</td><td>0.0558</td><td>0.2015</td><td>0.2999</td></tr><tr><td>0.0711</td><td>0.4250</td><td>0.9585</td><td>0.0457</td><td>0.1586</td><td>0.2113</td></tr><tr><td>0.0671</td><td>0.3167</td><td>0.9610</td><td>0.0421</td><td>0.1508</td><td>0.1590</td></tr><tr><td>0.0465</td><td>0.3000</td><td>0.9621</td><td>0.0410</td><td>0.1477</td><td>0.1388</td></tr><tr><td>0.2016</td><td>0.7750</td><td>0.8709</td><td>0.0855</td><td>0.2393</td><td>0.3670</td></tr><tr><td rowspan="6">wav2vec2-large whisper-tiny whisper-base</td><td rowspan="6">librispeech</td><td>0.1719</td><td>0.7917</td><td>0.8752</td><td>0.0651</td><td>0.2301</td><td>0.3247</td></tr><tr><td>0.0816</td><td>0.2348</td><td>0.8890</td><td>0.0828</td><td>0.3009</td><td>0.2257</td></tr><tr><td>0.0558</td><td>0.1927</td><td>0.8912</td><td>0.0776</td><td>0.2897</td><td>0.1631</td></tr><tr><td>0.0397</td><td>0.1569</td><td>0.8927</td><td>0.0738</td><td>0.2600</td><td>0.1047</td></tr><tr><td>0.0324</td><td>0.1356</td><td>0.8927</td><td>0.0729</td><td>0.2637</td><td>0.0818</td></tr><tr><td>0.0233</td><td>0.1199</td><td>0.8924</td><td>0.0672</td><td>0.2540</td><td>0.0634</td></tr><tr><td rowspan="2">wav2vec2-base wav2vec2-large</td><td rowspan="6"></td><td>0.0356 0.0290</td><td>0.1205 0.1136</td><td>0.9902 0.9918</td><td>0.0183</td><td>0.2696 0.2643</td><td>0.1585</td></tr><tr><td>0.1387</td><td></td><td></td><td>0.0149</td><td></td><td>0.1344</td></tr><tr><td rowspan="6">whisper-tiny whisper-base whisper-small whisper-medium voxpopuli whisper-large</td><td>0.1149</td><td>0.3580 0.2716</td><td>0.9583</td><td>0.0482</td><td>0.4723</td><td>0.3157</td></tr><tr><td>0.1025</td><td></td><td>0.9628</td><td>0.0397</td><td>0.4505</td><td>0.2532</td></tr><tr><td></td><td>0.1481</td><td>0.9660</td><td>0.0346</td><td>0.4567</td><td>0.1984</td></tr><tr><td>0.0920</td><td>0.1605</td><td>0.9675 0.9651</td><td>0.0312</td><td>0.4739</td><td>0.1701</td></tr><tr><td>0.1084</td><td>0.1605</td><td></td><td>0.0406</td><td>0.4345</td><td>0.1684</td></tr><tr><td>0.2247 0.1876</td><td>0.7407 0.6914</td><td>0.8754 0.8799</td><td>0.1334 0.1017</td><td>0.4387 0.4366</td><td>0.3577</td><td>0.4084</td></tr></table>

Micro-Level. Beyond macro-level trends, a new metric should provide diagnostic utility orthogonal to established baselines. Conventional metrics often occupy narrow ranges for modern ASR systems, with WER and SemDist below 0.1 and BERTScoreF1 around 0.9–0.99. SITAR scores span 0.06 to 0.40, showing greater dispersion similar to NE-WER and AER. Table 1 further reveals rank inversions across model families. On LibriSpeech, SITAR favors whisper-small (0.1047) over wav2vec2- large (0.1344), reversing the ordering under WER, NE-WER, BERTScoreF1, and SemDist. On FLEURS, whisper-tiny leads wav2vec2-large under WER, NE-WER, and BERTScore , but receives a higher SITAR score (0.3872 versus 0.3247). Comprehension-centric AER agrees with SITAR in both cases, as does SemDist on FLEURS. However, AER’s uncached evaluation requires four test-time LLM calls per transcript pair (Pulikodan et al., 2025), whereas SITAR requires none. Appendix B provides further comparative analysis.

## 4.3 INTRINSIC RELIABILITY

We next evaluate the intrinsic quality of SITAR using 330 controlled transcript pairs in $\mathcal { D } _ { \mathrm { c t r l } } .$ , comprising 11 hypothesis variants for each of 30 English references across finance, culture and sports, public affairs, services, and general communication. The construction profiles are listed in Table 2. Appendix C provides a complete construction example, annotation instructions, and analysis details, including the prediction-based selection of the 30 references from 35 candidates.

Validating the Common Scale. A common scale should map comparable error structures to consistent severity regions across contexts. Grouping by the planned tier and gap-count profiles, Table 2 summarizes normalized score distributions across references in $\mathcal { D } _ { \mathrm { c t r l } }$ . These groups retain their construction assignments; the subsequent ranking analysis uses final human-adjudicated labels, including the correction detailed in Appendix C.4. Expression-only profiles $\mathcal { T } _ { 0 }$ and $2 \mathcal { T } _ { 0 }$ receive consistently low scores across references, with medians below 0.10 and interquartile ranges near 0.04. For the $\mathcal { T } _ { 1 }$ and $\mathcal { T } _ { 2 }$ profiles, distribution centers shift upward, with greater dispersion and partially overlapping central intervals. Profiles containing both tiers have medians above 0.70 and narrower interquartile ranges, indicating higher score positions with less dispersion. Within a fixed planned $\mathcal { T } _ { 1 }$ and $\mathcal { T } _ { 2 }$ composition, adding $\mathcal { T } _ { 0 }$ gaps produces small changes in location and spread. For example, adding two $\mathcal { T } _ { 0 }$ gaps to $\mathcal { T } _ { 2 } + \mathcal { T } _ { 1 }$ increases the mean by 0.0030 and standard deviation by 0.0082, consistent with expression-level differences providing modest refinements to principal severity position.

Table 2: Cross-reference distributions of normalized SITAR scores.
<table><tr><td>Construction profile</td><td>Mean</td><td>Median</td><td>Standard deviation</td><td>Central 50%</td><td>IQR</td></tr><tr><td> $\mathcal { T } _ { 0 }$ </td><td>0.0813</td><td>0.0763</td><td>0.0491</td><td>[0.0575, 0.0986]</td><td>0.0411</td></tr><tr><td> $2 \mathcal { T } _ { 0 }$ </td><td>0.1268</td><td>0.0946</td><td>0.0791</td><td>[0.0848, 0.1269]</td><td>0.0420</td></tr><tr><td> $\mathcal { T } _ { 1 }$ </td><td>0.5175</td><td>0.5131</td><td>0.1634</td><td>[0.3907, 0.6414]</td><td>0.2506</td></tr><tr><td> $\mathcal { T } _ { 1 } + \mathcal { T } _ { 0 }$ </td><td>0.5181</td><td>0.5032</td><td>0.1629</td><td>[0.3873, 0.6512]</td><td>0.2639</td></tr><tr><td> $\mathcal { T } _ { 1 } + 2 \mathcal { T } _ { 0 }$ </td><td>0.5324</td><td>0.5268</td><td>0.1672</td><td>[0.3921, 0.6740]</td><td>0.2819</td></tr><tr><td> $\mathcal { T } _ { 2 }$ </td><td>0.6612</td><td>0.6807</td><td>0.1344</td><td>[0.5711, 0.7602]</td><td>0.1891</td></tr><tr><td> $\mathcal { T } _ { 2 } + \mathcal { T } _ { 0 }$ </td><td>0.6664</td><td>0.6807</td><td>0.1343</td><td>[0.5889, 0.7684]</td><td>0.1796</td></tr><tr><td> $\mathcal { T } _ { 2 } + 2 \mathcal { T } _ { 0 }$ </td><td>0.6718</td><td>0.6933</td><td>0.1404</td><td>[0.5747, 0.7766]</td><td>0.2019</td></tr><tr><td> $\mathcal { T } _ { 2 } + \mathcal { T } _ { 1 }$ </td><td>0.6974</td><td>0.7334</td><td>0.1282</td><td>[0.6570, 0.7784]</td><td>0.1214</td></tr><tr><td> $\mathcal { T } _ { 2 } + \mathcal { T } _ { 1 } + \mathcal { T } _ { 0 }$ </td><td>0.6988</td><td>0.7346</td><td>0.1305</td><td>[0.6448, 0.7828]</td><td>0.1380</td></tr><tr><td> $\mathcal { T } _ { 2 } + \mathcal { T } _ { 1 } + 2 \mathcal { T } _ { 0 }$ </td><td>0.7004</td><td>0.7409</td><td>0.1364</td><td>[0.6536, 0.7902]</td><td>0.1365</td></tr></table>

Assessing Human Consistency. Holding each reference fixed, we assess whether predicted scores align with human severity judgments. For this evaluation, two PhD students independently annotated gap severity tiers, yielding the inter-annotator agreement summarized in Table 3. Agreement reaches 90.00% at the gap level and 81.21% at the transcript-pair level, with all kappa coefficients exceeding 0.83, and most disagreements occur between adjacent tiers, in line with weighted κ increasing beyond the unweighted baseline. With human annotations directly adopted upon agreement

Table 3: Inter-annotator agreement. Gap-level agreement compares severity labels for each $g _ { i } ,$ while pair-level agreement requires matching labels for all gaps associated with h.  
Table 4: Within-reference ranking agreement using final labels. N counts strict-order comparisons; $\operatorname { A g r } .$ and Spearman $\rho _ { s }$ are averaged over references.
<table><tr><td>Measure</td><td>Score</td></tr><tr><td> $\kappa$ </td><td>0.8396</td></tr><tr><td> $\kappa _ { w } ^ { \mathrm { l i n } }$ </td><td>0.8675</td></tr><tr><td>uad  $\kappa _ { w } ^ { \mathrm { q } \mathfrak { l } }$ </td><td>0.8924</td></tr><tr><td>Gap-Level</td><td>90.00%</td></tr><tr><td>Pair-Level</td><td>81.21%</td></tr></table>

<table><tr><td>Domain</td><td>N</td><td>Agr. (%)</td><td>Mean  $\rho _ { s }$ </td></tr><tr><td>Culture</td><td>495</td><td>88.48</td><td>0.8758</td></tr><tr><td>Finance</td><td>220</td><td>80.45</td><td>0.7614</td></tr><tr><td>Services</td><td>385</td><td>80.00</td><td>0.7078</td></tr><tr><td>Civic</td><td>275</td><td>83.64</td><td>0.8182</td></tr><tr><td>General</td><td>272</td><td>85.16</td><td>0.8062</td></tr></table>

and resolved via majority voting with initial model predictions in case of disagreement, we obtain 1,647 strict-order comparisons. SITAR achieves a mean pairwise agreement of 84.07% and a mean $\rho _ { s }$ of 0.8001. The median per-reference agreement of 85.45% is close to the overall result, indicating similar ordering agreement for a typical reference. As shown in Table 4, all five evaluated domains attain mean pairwise agreement of at least 80.00% and mean $\rho _ { s }$ above 0.70. These results complement the cross-reference analysis by showing that most expected severity relations are also retained within individual references.

## 5 CONCLUSION

We introduced SITAR, a lightweight, deterministic metric for reference-grounded ASR situational distortion. Trained via CGD targets, a shared severity scale, and bounded recursive aggregation, it enables fast single-pass evaluation without runtime LLM calls. Across three English benchmarks and a controlled test set, SITAR demonstrates strong diagnostic value, severity ordering, and score consistency. Future work will scale data volume and multilingual coverage to study its scaling behavior.

## REFERENCES

Eneko Agirre, Daniel Cer, Mona Diab, and Aitor Gonzalez-Agirre. SemEval-2012 task 6: A pilot on semantic textual similarity. In Eneko Agirre, Johan Bos, Mona Diab, Suresh Manandhar, Yuval Marton, and Deniz Yuret (eds.), \*SEM 2012: The First Joint Conference on Lexical and Computational Semantics – Volume 1: Proceedings of the main conference and the shared task, and Volume 2: Proceedings of the Sixth International Workshop on Semantic Evaluation (SemEval 2012), pp. 385–393, Montreal, Canada, 7-8 June 2012. Association for Computational Linguis-´ tics. URL https://aclanthology.org/S12-1051/.

Alena Aks¨ enova, Daan van Esch, James Flynn, and Pavel Golik. How might we create better bench-¨ marks for speech recognition? In Kenneth Church, Mark Liberman, and Valia Kordoni (eds.), Proceedings ofthe 1st Workshop on Benchmarking: Past, Present and Future, pp. 22–34, Online, August 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.bppf-1.4. URL https://aclanthology.org/2021.bppf-1.4/.

Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, and Michael Auli. wav2vec 2.0: A framework for self-supervised learning of speech representations. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.), Advances in Neural Information Processing Systems, volume 33, pp. 12449–12460. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper\_files/paper/2020/ file/92d1e1eb1cd6f9fba3227870bb6d7f07-Paper.pdf.

Eugen Beck, Sarah Beranek, Uma Moothiringote, Daniel Mann, Wilfried Michel, Katie Nguyen, and Taylor Tragemann. AppTek Call-Center Dialogues: A Multi-Accent Long-Form Benchmark for English ASR. In Interspeech 2026, pp. 7179–7183, 2026. doi: 10.21437/Interspeech. 2026-2047.

Lasse Borgholt, Jakob Havtorn, Christian Ige, Lars Maaløe, and Zheng-Hua Tan. A text-to-text alignment algorithm for better evaluation of modern speech recognition systems. In ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 3476–3480, 2026. doi: 10.1109/ICASSP55912.2026.11463731.

Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning. A large annotated corpus for learning natural language inference. In Llu´ıs Marquez, Chris Callison-Burch, and\` Jian Su (eds.), Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, pp. 632–642, Lisbon, Portugal, September 2015. Association for Computational Linguistics. doi: 10.18653/v1/D15-1075. URL https://aclanthology.org/D15-1075/.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.), Advances in Neural Information Processing Systems, volume 33, pp. 1877–1901. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper\_files/paper/2020/ file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf.

H. H. Clark and S. E. Brennan. Grounding in communication. Perspectives on Socially Shared Cognition., pp. 127–149, 1991. doi: 10.1037/10096-006.

Alexis Conneau, Min Ma, Simran Khanuja, Yu Zhang, Vera Axelrod, Siddharth Dalmia, Jason Riesa, Clara Rivera, and Ankur Bapna. Fleurs: Few-shot learning evaluation of universal representations of speech. In 2022 IEEE Spoken Language Technology Workshop (SLT), pp. 798–805, 2023. doi: 10.1109/SLT54892.2023.10023141.

Zachary Ellis, Jared Joselowitz, Yash Deo, Yajie He, Anna Kalygina, Aisling Higham, Mana Rahimzadeh, Yan Jia, Ibrahim Habli, and Ernest Lim. WER is unaware: Assessing how ASR errors distort clinical understanding in patient facing dialogue. In Giuseppe Riccardi, Seyed Mahed Mousavi, Maria Ines Torres, Koichiro Yoshino, Zoraida Callejas, Shammur Absar Chowdhury, Yun-Nung Chen, Frederic Bechet, Joakim Gustafson, Geraldine Damnati, Alex´ Papangelis, Luis Fernando D’Haro, John Mendonc¸a, Raffaella Bernardi, Dilek Hakkani-Tur, Giuseppe ”Pino” Di Fabbrizio, Tatsuya Kawahara, Firoj Alam, Gokhan Tur, and Michael Johnston (eds.), Proceedings ofthe 16th International Workshop on Spoken Dialogue System Technology, pp. 391–417, Trento, Italy, February 2026. Association for Computational Linguistics. URL https://aclanthology.org/2026.iwsds-1.39/.

Benoit Favre, Kyla Cheung, Siavash Kazemian, Adam Lee, Yang Liu, Cosmin Munteanu, Ani Nenkova, Dennis Ochei, Gerald Penn, Stephen Tratz, Clare Voss, and Frauke Zeller. Automatic human utility evaluation of ASR systems: does WER really predict performance? In Interspeech 2013, pp. 3463–3467, 2013. doi: 10.21437/Interspeech.2013-610.

John S. Garofolo, Ellen M. Voorhees, Cedric G. P. Auzanne, and Vincent M. Stanford. Spoken Document Retrieval:1998 evaluation and investigation of new metrics. In Workshop on Accessing Information in Spoken Audio, pp. 1–7, 1999.

Rajarshi Haldar and Julia Hockenmaier. Rating roulette: Self-inconsistency in LLM-as-a-judge frameworks. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng (eds.), Findings ofthe Associationfor Computational Linguistics: EMNLP 2025, pp. 24986– 25004, Suzhou, China, November 2025. Association for Computational Linguistics. ISBN 979-8- 89176-335-7. doi: 10.18653/v1/2025.findings-emnlp.1361. URL https://aclanthology. org/2025.findings-emnlp.1361/.

John Harvill, Rinat Khaziev, Scarlett Li, Randy Cogill, Lidan Wang, Gopinath Chennupati, and Hari Thadakamalla. Significant asr error detection for conversational voice assistants. In ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 11606–11610, 2024. doi: 10.1109/ICASSP48485.2024.10448230.

Pengcheng He, Jianfeng Gao, and Weizhu Chen. Debertav3: Improving deberta using electra-style pre-training with gradient-disentangled embedding sharing. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. URL https://openreview.net/forum?id=sE7-XhLxHA.

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Thomas Hennigan, Eric Noland, Katherine Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Oriol Vinyals, Jack Rae, and Laurent Sifre.´ An empirical analysis of compute-optimal large language model training. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (eds.), Advances in Neural Information Processing Systems, volume 35, pp. 30016–30030. Curran Associates, Inc., 2022. doi: 10.52202/ 068431-2176. URL https://proceedings.neurips.cc/paper\_files/paper/ 2022/file/c1e2faff6f588870935f114ebe04a3e5-Paper-Conference.pdf.

Yihan Hong, Huaiyuan Yao, Bolin Shen, Wanpeng Xu, Hua Wei, and Yushun Dong. From rubrics to reliable scores: Evidence-grounded text evaluation with llm judges, 2026. URL https: //arxiv.org/abs/2601.08654.

Tzu-Heng Huang, Shengqi Qiu, and Frederic Sala. Codifying the judge: Scalable evaluation via program distillation, 2026. URL https://arxiv.org/abs/2607.22561.

Zixuan Jiang, Yanqiao Zhu, Peng Wang, Qinyuan Chen, Xinjian Zhao, Xipeng Qiu, Wupeng Wang, Zhifu Gao, Xiangang Li, Kai Yu, and Xie Chen. Towards human-like interactive speech recognition with agentic correction and semantic evaluation, 2026. URL https://arxiv.org/ abs/2605.29430.

Thennal D K, Jesin James, Deepa Padmini Gopinath, and Muhammed Ashraf K. Advocating character error rate for multilingual ASR evaluation. In Luis Chiruzzo, Alan Ritter, and Lu Wang (eds.), Findings of the Association for Computational Linguistics: NAACL 2025, pp.

4941–4950, Albuquerque, New Mexico, April 2025. Association for Computational Linguistics. ISBN 979-8-89176-195-7. doi: 10.18653/v1/2025.findings-naacl.277. URL https: //aclanthology.org/2025.findings-naacl.277/.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. CoRR, abs/2001.08361, 2020. URL https://arxiv.org/abs/2001.08361.

Suyoun Kim, Abhinav Arora, Duc Le, Ching-Feng Yeh, Christian Fuegen, Ozlem Kalinli, and Michael L. Seltzer. Semantic Distance: A New Metric for ASR Performance Analysis Towards Spoken Language Understanding. In Interspeech 2021, pp. 1977–1981, 2021. doi: 10.21437/Interspeech.2021-1929.

Korbinian Kuhn, Verena Kersken, and Gottfried Zimmermann. Beyond Levenshtein: Leveraging Multiple Algorithms for Robust Word Error Rate Computations And Granular Error Classifica tions. In Interspeech 2024, pp. 4543–4547, 2024. doi: 10.21437/Interspeech.2024-32.

Taniya Mishra, Andrej Ljolje, and Mazin Gilbert. Predicting human perceived accuracy of ASR systems. In Interspeech 2011, pp. 1945–1948, 2011. doi: 10.21437/Interspeech.2011-364.

Andrew Cameron Morris, Viktoria Maier, and Phil Green. From WER and RIL to MER and WIL: improved evaluation measures for connected speech recognition. In Interspeech 2004, pp. 2765– 2768, 2004. doi: 10.21437/Interspeech.2004-668.

Hiroaki Nanjo, Teruhisa Misu, and Tatsuya Kawahara. Minimum Bayes-risk decoding considering word significance for information retrieval system. In Interspeech 2005, pp. 561–564, 2005. doi: 10.21437/Interspeech.2005-341.

Zhenxing Niu, Mo Zhou, Le Wang, Xinbo Gao, and Gang Hua. Ordinal regression with multiple output cnn for age estimation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2016.

Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur. Librispeech: An asr corpus based on public domain audio books. In 2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 5206–5210, 2015. doi: 10.1109/ICASSP.2015. 7178964.

Amruta Parulekar and Preethi Jyothi. LASER: An LLM-based ASR scoring and evaluation rubric. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng (eds.), Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 24762–24771, Suzhou, China, November 2025. Association for Computational Linguistics. ISBN 979-8-89176-332-6. doi: 10.18653/v1/2025.emnlp-main.1257. URL https: //aclanthology.org/2025.emnlp-main.1257/.

Bornali Phukon, Xiuwen Zheng, and Mark Hasegawa-Johnson. Aligning ASR Evaluation with Human and LLM Judgments: Intelligibility Metrics Using Phonetic, Semantic, and NLI Approaches. In Interspeech 2025, pp. 5708–5712, 2025. doi: 10.21437/Interspeech.2025-891.

Sujith Pulikodan, Sahapthan K, Prasanta Kumar Ghosh, Visruth Sanka, and Nihar Desai. An approach to measuring the performance of Automatic Speech Recognition(ASR) models in the context of Large Language Model(LLM) powered applications. In Interspeech 2025, pp. 5718–5722, 2025. doi: 10.21437/Interspeech.2025-2488.

Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine Mcleavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (eds.), Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pp. 28492–28518. PMLR, 23–29 Jul 2023. URL https: //proceedings.mlr.press/v202/radford23a.html.

Janine Rugayan, Torbjørn Svendsen, and Giampiero Salvi. Semantically Meaningful Metrics for Norwegian ASR Systems. In Interspeech 2022, pp. 2283–2287, 2022. doi: 10.21437/Interspeech. 2022-817.

Nicholas Ruiz and Marcello Federico. Phonetically-oriented word error alignment for speech recognition error analysis in speech translation. In 2015 IEEE Workshop on Automatic Speech Recognition and Understanding (ASRU), pp. 296–302, 2015. doi: 10.1109/ASRU.2015.7404808.

Zitha Sasindran, Harsha Yelchuri, T. V. Prabhakar, and Supreeth Rao. Heval: A new hybrid evaluation metric for automatic speech recognition tasks. In 2023 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU), pp. 1–7, 2023. doi: 10.1109/ASRU57964.2023.10389717.

Zitha Sasindran, Harsha Yelchuri, and T. V. Prabhakar. SeMaScore: A new evaluation metric for automatic speech recognition tasks. In Interspeech 2024, pp. 4558–4562, 2024. doi: 10.21437/ Interspeech.2024-2033.

Joel Shor, Ruyue Agnes Bi, Subhashini Venugopalan, Steven Ibara, Roman Goldenberg, and Ehud Rivlin. Clinical BERTScore: An improved measure of automatic speech recognition performance in clinical settings. In Tristan Naumann, Asma Ben Abacha, Steven Bethard, Kirk Roberts, and Anna Rumshisky (eds.), Proceedings ofthe 5th Clinical Natural Language Processing Workshop, pp. 1–7, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/ v1/2023.clinicalnlp-1.1. URL https://aclanthology.org/2023.clinicalnlp-1. 1/.

Jimmy Tobin, Qisheng Li, Subhashini Venugopalan, Katie Seaver, Richard Cave, and Katrin Tomanek. Assessing ASR Model Quality on Disordered Speech using BERTScore. In 1st Workshop on Speechfor Social Good (S4SG), pp. 26–30, 2022. doi: 10.21437/S4SG.2022-6.

Changhan Wang, Morgane Riviere, Ann Lee, Anne Wu, Chaitanya Talnikar, Daniel Haziza, Mary Williamson, Juan Pino, and Emmanuel Dupoux. VoxPopuli: A large-scale multilingual speech corpus for representation learning, semi-supervised learning and interpretation. In Chengqing Zong, Fei Xia, Wenjie Li, and Roberto Navigli (eds.), Proceedings of the 59th Annual Meeting ofthe Associationfor Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pp. 993–1003, Online, August 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.acl-long.80. URL https: //aclanthology.org/2021.acl-long.80/.

Ye-Yi Wang, A. Acero, and C. Chelba. Is word error rate a good indicator for spoken language understanding accuracy. In 2003 IEEE Workshop on Automatic Speech Recognition and Understanding (IEEE Cat. No.03EX721), pp. 577–582, 2003. doi: 10.1109/ASRU.2003.1318504.

Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi. Bertscore: Evaluating text generation with BERT. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net, 2020. URL https://openreview.net/forum?id=SkeHuCVFDr.

R. A. Zwaan and G. A. Radvansky. Situation models in language comprehension and memory. Psychological Bulletin, 123:162–185, 1998. doi: 10.1037/0033-2909.123.2.162.

## A SITAR CONSTRUCTION DETAILS

## A.1 SCORE CONSTRUCTION AND PROPERTIES

Equation 7 combines a base score $\beta$ with a bounded refinement determined by gap counts. The base score records the most severe adjusted gap position, and the refinement records the remaining gaps according to their nominal tiers. This section gives the complete calculation, derives the parameter choice, and establishes the score’s bounds, tier priority, and diminishing response to increasing counts.

## A.1.1 BASE SCORE AND REFINEMENT COUNTS

For a retained transcript pair, let $t _ { \mathrm { r e v } } ^ { ( i ) }$ and $t _ { \mathrm { m a j } } ^ { ( i ) }$ denote the reviewed and majority tiers of gap $g _ { i } ,$ respectively. Following the annotation procedure in the main text, its adjusted severity position is

$$
b _ { i } = t _ { \mathrm { r e v } } ^ { ( i ) } + \frac { 1 } { 3 } \mathrm { s g n } \Bigl ( t _ { \mathrm { m a j } } ^ { ( i ) } - t _ { \mathrm { r e v } } ^ { ( i ) } \Bigr ) , \qquad \beta = \operatorname* { m a x } _ { i } b _ { i } ,\tag{12}
$$

where $\operatorname { s g n } ( 0 ) = 0$ . Agreement leaves the nominal score unchanged, while disagreement shifts it by $1 / 3$ toward the majority tier. The reviewed tier remains the nominal assignment used for counting. The consistency filter in Eq. 9 restricts retained majority and reviewed assignments to the same or adjacent tiers.

Explicitly, the possible $( t _ { \mathrm { r e v } } ^ { ( i ) } , t _ { \mathrm { m a j } } ^ { ( i ) } )$ pairs are $( 0 , 0 ) , ( 0 , 1 ) , ( 1 , 0 ) , ( 1 , 1 ) , ( 1 , 2 ) , ( 2 , 1 )$ , and (2, 2), producing adjusted positions $0 , \dot { 1 / 3 } , 2 / 3 , 1 , 4 / 3 , 5 / 3 .$ , and 2, respectively. These adjustments are computed for each gap before taking the maximum. A pair’s base score therefore depends on the most severe adjusted gap, rather than an average of gap positions. The majority tier is obtained from the three annotations of that same gap; pairs without a per-gap majority have already been excluded by the annotation procedure.

We count each gap once using its reviewed tier,

$$
c _ { k } = \sum _ { i } \mathbf { 1 } \left[ t _ { \mathrm { r e v } } ^ { ( i ) } = k \right] , \qquad k _ { \mathrm { m a x } } = \operatorname* { m a x } \{ k : c _ { k } > 0 \} , \qquad { \bar { c } } _ { k } = c _ { k } - \mathbf { 1 } [ k = k _ { \mathrm { m a x } } ] .\tag{13}
$$

The gap that determines $\beta$ belongs to $\mathcal { T } _ { k _ { \mathrm { m a x } } }$ , because even the lowest adjusted position at a higher nominal tier exceeds the highest position at the tier below it. Subtracting one at this tier prevents counting that gap again in the quantity refinement. All lower-tier gaps remain included. If several gaps attain the same maximum position, excluding any one of them gives the same refinement counts. The three original annotation rounds determine the majority assignments; they do not count as three separate observations of each gap.

Table 5 lists the seven base positions and their score intervals. A pair with a single gap receives its base score directly. If no gaps are present, we define $S ( r , h ) = 0 $ separately, without taking a maximum over an empty set or entering the recursion.

Table 5: Base positions and score intervals for finite gap counts. The highest nominal tier is determined by the reviewed assignments.
<table><tr><td>Highest nominal tier</td><td>Base  $\beta$ </td><td>Score interval</td></tr><tr><td> $\mathcal { T } _ { 0 }$ </td><td>0 1/3</td><td>[0,1/3) [1/3, 2/3)</td></tr><tr><td>T1</td><td>2/3 1 4/3</td><td>[2/3,1) [1, 4/3) [4/3, 5/3)</td></tr><tr><td>T2</td><td>5/3 2</td><td>[5/3,2) [2, 7/3)</td></tr></table>

## A.1.2 RECURSIVE CALCULATION

For explicit computation of Eq. 8, initialize $u _ { - 1 } = 0$ and process every nominal tier up to $k _ { \mathrm { m a x } }$ in ascending order,

$$
u _ { k } = 1 - \rho ^ { { \bar { c } } _ { k } + u _ { k - 1 } } , \qquad k = 0 , \dots , k _ { \mathrm { m a x } } , \qquad S ( r , h ) = \beta + \frac { u _ { k _ { \mathrm { m a x } } } } { 3 } .\tag{14}
$$

An intermediate tier with zero gaps is still evaluated. Its update becomes $u _ { k } = 1 - \rho ^ { u _ { k - 1 } }$ , preserving the tier through which lower-level contributions pass. Skipping such a tier would change the score.

For convenience, define $U _ { \rho } ( c , u ) = 1 - \rho ^ { c + u }$ and write $R = u _ { k _ { \mathrm { m a x } } }$ for the final refinement. The three possible highest tiers give

$$
R = \left\{ \begin{array} { l l } { U _ { \rho } ( c _ { 0 } - 1 , 0 ) , } & { k _ { \mathrm { m a x } } = 0 , } \\ { U _ { \rho } ( c _ { 1 } - 1 , U _ { \rho } ( c _ { 0 } , 0 ) ) , } & { k _ { \mathrm { m a x } } = 1 , } \\ { U _ { \rho } ( c _ { 2 } - 1 , U _ { \rho } ( c _ { 1 } , U _ { \rho } ( c _ { 0 } , 0 ) ) ) , } & { k _ { \mathrm { m a x } } = 2 . } \end{array} \right.\tag{15}
$$

Only the highest occupied tier has its count reduced by one. A missing lower tier contributes zero new gaps but still applies its recursive update. Since $U _ { \rho } \mathbf { \bar { ( } 0 , 0 ) } = 0$ , leading empty tiers do not change an otherwise single-tier calculation. All updates use unrounded intermediate values; the numerical examples below are rounded only for presentation.

## A.1.3 PARAMETER CHOICE

Refinement width. Suppose the score is written as $S = \beta + \lambda R$ . Adjacent base positions are separated by $1 / 3 ,$ , while the recursive refinement satisfies $0 \leq R < 1$ and approaches one as the highest-tier count increases. Any $0 < \lambda \leq 1 / 3$ preserves the interval boundaries. We use $\lambda = 1 / 3$ to make the full interval available for quantity refinement. A smaller value would leave part of each interval unused even in the infinite-count limit, and a larger value would allow sufficiently many gaps to cross the next base position. The selected width is thus fixed by the base-score grid rather than tuned during regressor training.

Saturation rate. To set $\rho ,$ consider n gaps all assigned to the same nominal tier. After the first gap is accounted for by $\beta ,$ the refinement simplifies to

$$
R _ { n } = 1 - \rho ^ { n - 1 } , \qquad S _ { n } = \beta + { \frac { 1 - \rho ^ { n - 1 } } { 3 } } .\tag{16}
$$

We choose seven same-tier gaps as a reference count and require them to occupy 90% of the refinement interval. Thus,

$$
R _ { 7 } = 0 . 9 \quad \Longrightarrow \quad \rho ^ { 6 } = 0 . 1 \quad \Longrightarrow \quad \rho = 1 0 ^ { - 1 / 6 } \approx 0 . 6 8 1 2 9 2 .\tag{17}
$$

More generally, a reference count $N _ { \mathrm { r e f } } > 1$ and utilization $0 < \eta < 1$ give $\rho = ( 1 - \eta ) ^ { 1 / ( N _ { \mathrm { r e f } } - 1 ) }$ Seven is a calibration point, not a cap on the number of gaps. Table 6 shows how the refinement continues to grow beyond this count.

Table 6: Quantity refinement for n gaps at the same nominal tier, with $\rho = 1 0 ^ { - 1 / 6 }$ . Values are rounded to six decimal places.
<table><tr><td>Gaps n</td><td>Refinement  $R _ { n }$ </td><td>Score increment  $R _ { n } / 3$ </td></tr><tr><td>1</td><td>0.000000</td><td>0.000000</td></tr><tr><td>2</td><td>0.318708</td><td>0.106236</td></tr><tr><td>3</td><td>0.535841</td><td>0.178614</td></tr><tr><td>4</td><td>0.683772</td><td>0.227924</td></tr><tr><td>5</td><td>0.784557</td><td>0.261519</td></tr><tr><td>6</td><td>0.853220</td><td>0.284407</td></tr><tr><td>7</td><td>0.900000</td><td>0.300000</td></tr><tr><td>8</td><td>0.931871</td><td>0.310624</td></tr><tr><td>9</td><td>0.953584</td><td>0.317861</td></tr><tr><td>10</td><td>0.968377</td><td>0.322792</td></tr><tr><td>11</td><td>0.978456</td><td>0.326152</td></tr></table>

## A.1.4 PROPERTIES OF THE SCORE

Bounded refinement. For $0 < \rho < 1$ , a finite nonnegative integer c, and $0 \leq u < 1$ , the update $1 - \rho ^ { c + u }$ lies in [0, 1). Starting from $u _ { - 1 } = 0$ , induction therefore gives $0 \leq u _ { k } < 1$ at every tier. Consequently,

$$
\beta \leq S ( r , h ) < \beta + { \frac { 1 } { 3 } } \leq { \frac { 7 } { 3 } } .\tag{18}
$$

The base intervals are disjoint. Combining the intervals associated with each highest nominal tier gives $[ 0 , 2 / 3 )$ for $\mathcal { T } _ { 0 } , [ 2 / 3 , 5 / 3 )$ for $\mathcal { T } _ { 1 }$ , and $[ 5 / 3 , 7 / 3 )$ for $\mathcal { T } _ { 2 }$ . Quantity refinement therefore cannot move a pair into a higher base interval or nominal tier.

Priority of higher tiers. The update $U _ { \rho } ( c , u )$ is strictly increasing in both arguments. More strongly, for every $0 \leq u < 1$

$$
U _ { \rho } ( c + 1 , 0 ) - U _ { \rho } ( c , u ) = \rho ^ { c } \left( \rho ^ { u } - \rho \right) > 0 .\tag{19}
$$

An additional gap at the current tier thus outweighs any finite configuration at lower tiers. For two count configurations with the same $\beta ,$ consider the highest tier at which their counts differ. At that tier, the configuration with the larger integer count has the larger recursive result, even if its lowertier contribution is zero and the other configuration’s lower-tier contribution is arbitrarily close to one. Counts at all higher tiers agree, so their strictly increasing updates preserve this ordering. The refinement therefore orders distinct count configurations lexicographically from the highest tier downward. At a fixed highest nominal tier, this also means that distinct count configurations cannot have identical refinements in exact arithmetic.

Increasing counts with diminishing increments. Fix $\beta$ and $k _ { \operatorname* { m a x } } .$ . To study sensitivity to a count, extend the update to nonnegative real-valued counts and let $a = - \log \rho > 0$ . Its first and second derivatives with respect to the current count are

$$
\frac { \partial U _ { \rho } ( c , u ) } { \partial c } = a \rho ^ { c + u } > 0 , \qquad \frac { \partial ^ { 2 } U _ { \rho } ( c , u ) } { \partial c ^ { 2 } } = - a ^ { 2 } \rho ^ { c + u } < 0 .\tag{20}
$$

The same derivatives apply with respect to u. Holding all other counts fixed, each subsequent tier applies another strictly increasing, concave function. Their composition is therefore strictly increasing and concave in the count being varied. Restricting back to integer counts, adding a gap at any tier up to $k _ { \mathrm { m a x } }$ increases the score, while successive additions at that same tier have decreasing effects. In the single-tier case, Eq. 16 gives the increment explicitly,

$$
S _ { n + 1 } - S _ { n } = { \frac { ( 1 - \rho ) \rho ^ { n - 1 } } { 3 } } > 0 .\tag{21}
$$

Each successive increment is multiplied by $\rho ,$ so the score approaches $\beta + 1 / 3$ without reaching it at any finite count. For benign speech disfluencies assigned to $\mathcal { T } _ { 0 }$ , this refinement records their accumulation while retaining the situation-preserving nominal tier.

A bound for a fixed total count. Let $\begin{array} { r } { n = \sum _ { k } c _ { k } \ge 1 } \end{array}$ and let $\begin{array} { r } { L = \sum _ { k < k _ { \mathrm { m a x } } } c _ { k } } \end{array}$ count all lowertier gaps. If $L \ = \ 0 .$ , then $R = 1 - \rho ^ { n - 1 }$ . If $L \ge 1$ , the lower-tier recursive result satisfies $u _ { k _ { \operatorname* { m a x } } - 1 } < 1 \leq L$ , giving

$$
R = 1 - \rho ^ { c _ { k _ { \operatorname* { m a x } } } - 1 + u _ { k _ { \operatorname* { m a x } } - 1 } } < 1 - \rho ^ { c _ { k _ { \operatorname* { m a x } } } - 1 + L } = 1 - \rho ^ { n - 1 } .\tag{22}
$$

Thus $R \leq 1 - \rho ^ { n - 1 }$ , with equality when all gaps occupy the highest tier. For at most $N$ gaps, the distance between the largest attainable score in one base interval and the next base position is at least $\rho ^ { N - 1 } / 3$ . With $N = 7$ , this gives $R \leq 0 . 9 , S \leq 2 . 3$ , and a separation of at least $1 / 3 0$ . The universal bound $S < 7 / 3$ remains valid when the count exceeds seven.

## A.1.5 WORKED EXAMPLES

A complete mixed-tier calculation. Consider the illustrative annotations in Table 7. The reviewed assignments contain two $\mathcal { T } _ { 0 }$ gaps, one $\mathcal { T } _ { 1 }$ gap, and two $\mathcal { T } _ { 2 }$ gaps. For $g _ { 4 } .$ the majority tier is one but the reviewed tier is two, so its adjusted position is $5 / 3$ . Gap $g _ { 3 }$ has adjusted position two and sets $\beta = 2$ . The counts are $( c _ { 0 } , c _ { 1 } , c _ { 2 } ) = ( 2 , 1 , 2 )$ , and excluding one highest-tier gap gives $( \bar { c } _ { 0 } , \bar { c } _ { 1 } , \bar { c } _ { 2 } ) = ( 2 , 1 , 1 )$

Table 7: Illustrative gap annotations for the mixed-tier calculation. The three original labels determine the majority tier, and the reviewed tier determines the nominal count.
<table><tr><td>Gap</td><td>Original labels</td><td>Majority tier</td><td>Reviewed tier</td><td>Position  $b _ { i }$ </td></tr><tr><td>go</td><td>(0,0,0)</td><td>0</td><td>0</td><td>0</td></tr><tr><td>g1</td><td>(0, 0, 1)</td><td>0</td><td>0</td><td>0</td></tr><tr><td>g2</td><td>(1, 1, 0)</td><td>1</td><td>1</td><td>1</td></tr><tr><td>g3</td><td>(2, 2, 1)</td><td>2</td><td>2</td><td>2</td></tr><tr><td>g4</td><td>(1, 1, 2)</td><td>1</td><td>2</td><td>5/3</td></tr></table>

Applying Eq. 14 gives

$$
\begin{array} { r } { u _ { 0 } = 1 - \rho ^ { 2 } \approx 0 . 5 3 5 8 4 1 , ~ } \\ { u _ { 1 } = 1 - \rho ^ { 1 + u _ { 0 } } \approx 0 . 4 4 5 3 4 0 , ~ } \\ { u _ { 2 } = 1 - \rho ^ { 1 + u _ { 1 } } \approx 0 . 4 2 5 7 3 8 , ~ } \\ { S ( r , h ) = 2 + u _ { 2 } / 3 \approx 2 . 1 4 1 9 1 3 . ~ } \end{array}\tag{23}
$$

If the original labels of $g _ { 3 }$ are instead (1, 1, 2) while its reviewed tier remains two, its adjusted position becomes $5 / 3$ . Both highest-tier gaps then have that position, giving $\beta = 5 / 3$ . The reviewed counts and quantity refinement remain unchanged, and the final score becomes 1.808579. The base positions differ by $\mathrm { 1 / 3 }$ , and the final scores retain exactly this difference.

Higher-tier count versus lower-tier accumulation. For a direct comparison of tier priority, fix $\beta = 1$ . One $\mathcal { T } _ { 1 }$ gap with six $\mathcal { T } _ { 0 }$ gaps receives

$$
S _ { A } = 1 + { \frac { 1 - \rho ^ { 1 - \rho ^ { 6 } } } { 3 } } \approx 1 . 0 9 7 3 5 1 ,
$$

whereas two $\mathcal { T } _ { 1 }$ gaps with no lower-tier gaps receive

$$
S _ { B } = 1 + \frac { 1 - \rho } { 3 } \approx 1 . 1 0 6 2 3 6 .
$$

Thus $S _ { A } < S _ { B }$ , as required by $\operatorname { E q } .$ . 19.

An empty intermediate tier. For $( c _ { 0 } , c _ { 1 } , c _ { 2 } ) = ( 2 , 0 , 1 )$ and $\beta = 2$ , the refinement counts are (2, 0, 0). The successive values are

$$
u _ { 0 } = 1 - \rho ^ { 2 } \approx 0 . 5 3 5 8 4 1 , \qquad u _ { 1 } = 1 - \rho ^ { u _ { 0 } } \approx 0 . 1 8 5 8 7 1 , \qquad u _ { 2 } = 1 - \rho ^ { u _ { 1 } } \approx 0 . 0 6 8 8 4 6 .
$$

The resulting score is 2.022949. Skipping the empty $\mathcal { T } _ { 1 }$ tier would instead give 2.061957, incorrectly treating the $\bar { \mathcal { T } } _ { 0 }$ contribution as if it were immediately below $\mathcal { T } _ { 2 }$

Situation-preserving disfluencies. When all reviewed gaps belong to $\mathcal { T } _ { 0 }$ and their majority assignments agree, $\beta = 0$ . One such gap receives score zero, two receive 0.106236, and seven receive 0.300000. Further gaps increase the score toward $1 / 3$ while remaining strictly below it. These positive refinements record the accumulation of expression-level differences, including benign speech disfluencies, without changing the nominal assessment that the situation is preserved. A zero score is therefore compatible with either no gap or a single unadjusted $\mathcal { T } _ { 0 }$ gap.

## A.1.6 FINITE-COUNT VERIFICATION

We enumerate all nonempty count configurations with at most seven gaps,

$$
\sum _ { n = 1 } ^ { 7 } { \binom { n + 2 } { 2 } } = 1 1 9 .
$$

These comprise 7 configurations with highest tier ${ \mathcal { T } } _ { 0 } .$ 28 with $\mathcal { T } _ { 1 }$ , and 84 with $\mathcal { T } _ { 2 }$ . All satisfy $0 \leq$ $R \le 0 . 9$ . Sorting the configurations within each highest tier by $( c _ { k _ { \operatorname* { m a x } } } , \ldots , c _ { 0 } )$ produces strictly increasing refinements, with 7, 28, and 84 distinct values, respectively. The largest refinement is 0.9, attained by seven gaps at the same tier. This finite enumeration checks the calculations at the chosen calibration count; the bounds and ordering for arbitrary finite counts follow from the preceding proofs. Distinctness refers to the unrounded mathematical scores, since rounding can merge nearby values.

## A.2 TRAINING DATA CONSTRUCTION

We primarily draw on AppTek Call-Center Dialogues (Beck et al., 2026), a newly collected conversational corpus with professional verbatim transcripts that preserve disfluencies, repetitions, and self-repairs. Its 1,746 single-channel recordings are segmented into 31,354 short clips spanning 14 accent configurations and 16 service domains. To extend this conversational coverage beyond customer-service interactions, we supplement the clips with 3,643 English recordings from FLEURS (Conneau et al., 2023), which provides read speech based on Wikipedia passages. Together, the two sources yield 34,997 speech segments at 16 kHz, with their audio and referencetranscript length distributions summarized in Table 8.

This speech pool supplies the reference–hypothesis pairs used to construct supervision for SITAR. We transcribe the segments using Qwen3-ASR-1.7B, Granite-Speech-4.1, Whisper-large-v3, and

Total: 6,038

Table 8: Statistics of candidate audio and reference transcripts across AppTek and FLEURS datasets.
<table><tr><td>Type</td><td>Source</td><td>N</td><td>Mean</td><td>P25</td><td>P50</td><td>P75</td><td>P95</td></tr><tr><td>Audio</td><td>AppTek_Clip FLEURS_en</td><td>31,354 3,643</td><td>10.77 s 10.19 s</td><td>3.14 s 7.68 s</td><td>7.66 s 9.66 s</td><td>15.90 s 11.88 s</td><td>31.79 s 16.90 s</td></tr><tr><td>Transcript</td><td>AppTek_Clip FLEURS_en</td><td>31,354 3,643</td><td>24.26 21.21</td><td>8 16</td><td>18 20</td><td>34 25</td><td>70 34</td></tr></table>

Table 9: Selected reference–hypothesis pairs by source, locale, and ASR system.
<table><tr><td>Source</td><td>Locale</td><td>Qwen3</td><td>Granite</td><td>Whisper</td><td>Parakeet</td><td>Total</td></tr><tr><td rowspan="11">AppTek_Clip</td><td>en-AU</td><td>188</td><td>190</td><td>167</td><td>155</td><td>700</td></tr><tr><td>en-CA</td><td>151</td><td>157</td><td>150</td><td>187</td><td>645</td></tr><tr><td>en-CN</td><td>185</td><td>151</td><td>160</td><td>177</td><td>673</td></tr><tr><td>en-GB</td><td>162</td><td>167</td><td>177</td><td>152</td><td>658</td></tr><tr><td>en-IE</td><td>184</td><td>165</td><td>171</td><td>151</td><td>671</td></tr><tr><td>en-MX en-SG</td><td>176</td><td>164</td><td>167</td><td>155</td><td>662</td></tr><tr><td>en-US_General</td><td>164</td><td>158</td><td>159</td><td>163</td><td>644</td></tr><tr><td></td><td>178</td><td>156</td><td>163</td><td>164</td><td>661</td></tr><tr><td>en-ZA</td><td>187</td><td>184</td><td>171</td><td>182</td><td>724</td></tr><tr><td>AppTek subtotal</td><td>1,575</td><td>1,492</td><td>1,485</td><td>1,486</td><td>6,038</td></tr><tr><td>FLEURS_en en_us</td><td>500</td><td>500</td><td>500</td><td>500</td><td>2,000</td></tr><tr><td colspan="2">Total</td><td>2,075</td><td>1,992</td><td>1,985</td><td>1,986</td><td>8,038</td></tr></table>

Parakeet-v3, pairing each output with the corresponding reference. Applying multiple recognizers to the same speech allows a shared reference to be associated with different naturally occurring transcription discrepancies, with each recognizer’s output forming a separate candidate pair. From these candidates, we select 8,038 pairs for annotation, restricting audio duration to 10–33 s and balancing recognizer contributions. For AppTek, sampling jointly balances ASR systems and accents, yielding 6,038 pairs across nine accents while covering all 16 service domains. This conversational subset is complemented by 2,000 FLEURS pairs, split equally across the four recognizers. Table 9 shows candidate distributions across sources, recognizers, and accents, while Figure 2 details domain coverage for AppTek.

![](images/6974ffd509142bfba142665be0f90b990f23aa18b9c24f3dbcbf2abc963d7682.jpg)  
Figure 2: Domain distribution of the 6,038 selected AppTek pairs.

![](images/676220b02f135813b6fc86c7d25b86f0addccaf4dbb910eafcef6c5d7fb093e1.jpg)  
Figure 3: Distribution of unique training pairs across supervision score bins (width 0.1). Hatched and solid blue regions denote original and added pairs, respectively. Values above bars show total counts; labels inside (‘+’ prefixed) indicate added counts.

We annotate the selected pairs using DeepSeek-V4-Pro across three independent rounds. We crafted a dedicated prompt to guide the model, ensuring strict adherence to a multilevel rubric grounded in the OECD results-chain framework. This initial rubric preserves finer distinctions among levels of consequence than the three-tier scheme ultimately adopted by SITAR. In practice, however, the highest consequence levels are represented by very few examples. These low-frequency categories carry greater surprisal in an information-theoretic sense and, in our annotations, correspond to more severe situational consequences. Their scarcity nevertheless provides insufficient support for reliable analysis and model training at this granularity. We therefore consolidate the higher consequence tiers into T<sub>2</sub>, yielding the final three-tier scheme used throughout the paper. The annotations subsequently undergo review by a committee of four agents using GPT-5.6-Sol-Max in ultra mode. The committee reassesses each gap’s severity in the context of its reference and supplies the reviewed tier assignments used to construct the scalar supervision targets.

We split the annotated pairs into training, validation, and test sets using proportions of 75%, 10%, and 15%, yielding 6,436, 856, and 1,292 pairs, respectively. These pairs come from errors that existing ASR systems happen to produce, so we cannot directly control their types or frequencies. We need severe-error examples to train the metric, but ASR systems are optimized to reduce transcription errors. Collecting such examples therefore requires searching through a large number of transcriptions, and some score intervals remain sparsely represented in the resulting training set.

We augment the training data by constructing variants of pairs containing at least three gaps. For each pair, we selectively correct a subset of the gaps while preserving the highest severity level, then recompute the supervision score from the remaining gaps. We preferentially select variants whose scores fall in sparsely populated intervals. This adds 3,763 unique pairs, increasing the training set from 6,436 to 10,199 pairs. Together with the validation and test sets, the dataset contains 12,347 unique pairs.

Figure 3 shows the original and added training pairs in score intervals of width 0.1. For intervals that remain sparse after augmentation, we upsample the training pairs by duplicating each pair once.

Although annotation starts from three severity tiers with nominal scores of 0, 1, and 2, directional boundary adjustments and the bounded recursive aggregation of gap counts yield finer score distinctions. Different combinations, such as $2 \mathcal { T } _ { 2 } + \mathcal { T } _ { 1 }$ and $\check { \mathcal { T } _ { 2 } } + 3 \mathcal { T } _ { 1 } + \dot { \mathcal { T } _ { 0 } }$ , can therefore receive different scores despite sharing the same highest severity tier.

![](images/5a2b00108524bd5456f8e886a03e590d9d7d79a5e8a462985f855bc2e3239c77.jpg)  
Figure 4: Normalized supervision scores across all 12,347 pairs. The upper panel shows the sample count at each score, and the lower panel marks the positions of the 246 distinct values. Distinct values are counted by rounding raw scores to four decimal places before normalization. Dashed lines mark the normalized base-score interval boundaries.

We further examine the supervision scores across all 12,347 pairs. When rounded to four decimal places, the raw scores contain 246 distinct values within the theoretical range of [0, 7/3). Figure 4 shows their distribution after dividing the raw scores by 7/3 to normalize them to [0, 1). On this normalized scale, the observed values range from 0 to 0.9929, with a median spacing of 0.0023 between adjacent distinct values. These values provide a fine-grained, pseudo-continuous supervision scale that captures differences in both error severity and gap composition.

## A.3 SCORER CONSTRUCTION

This section details the implementation of the SITAR scorer and the experiments supporting its configuration. The objective is to learn the scalar supervision targets described in Section 3 from a reference–hypothesis pair. Accordingly, the regression experiments below assess how well the scorer reproduces these targets; the metric-validity evaluations in the main text examine its behavior against complementary criteria.

## A.3.1 ARCHITECTURE AND OPTIMIZATION

We jointly encode the reference and hypothesis using microsoft/deberta-v3-large. The tokenizer constructs a sentence-pair input, with dynamic padding and truncation to a maximum of 512 tokens, including special tokens. The final configuration uses the original transcripts without explicit gap markers. Last-layer token representations are passed through a single bidirectional LSTM, whose hidden size in each direction is half the encoder hidden size. We then average its outputs over non-padding positions to obtain a fixed-dimensional pair representation. Sequences are packed before the LSTM so that padding does not enter the recurrent computation.

The pooled representation is passed through dropout with probability 0.1 and two separate linear heads: a scalar regression head and an auxiliary ordinal head. Only the scalar head supplies the SITAR prediction. All encoder, pooling, and head parameters are optimized jointly in the final model. Table 10 lists the training configuration.

Table 10: Configuration of the final SITAR scorer. Learning rates distinguish the pretrained encoder from the LSTM and prediction heads.
<table><tr><td>Component</td><td>Setting</td></tr><tr><td>Encoder</td><td>DeBERTa-v3-large; full fine-tuning</td></tr><tr><td>Input and maximum length</td><td>Original reference-hypothesis pair; 512 tokens</td></tr><tr><td>Pooling</td><td>One-layer bidirectional LSTM, followed by masked mean</td></tr><tr><td>Prediction heads</td><td>Linear scalar head and six-logit ordinal head</td></tr><tr><td>Head dropout Loss weights  $( \lambda _ { \mathrm { m s e } } , \lambda _ { \mathrm { b o u n d a r y } } , \lambda _ { \mathrm { o r d i n a l } } )$ </td><td>0.1 (1.0, 0.2, 0.5)</td></tr><tr><td>Ordinal thresholds</td><td> $\{ 1 / 3 , 2 / 3 , 1 , \stackrel { \cdot } { 4 } / 3 , 5 / 3 , 2 \}$ </td></tr><tr><td>Boundary refinement</td><td> $\rho = 1 0 ^ { - 1 / 6 } ; q = 2$ </td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Encoder / pooling and head learning rates</td><td> $6 \times 1 0 ^ { - 6 } / 1 0 ^ { - 4 }$ </td></tr><tr><td>Learning-rate schedule</td><td>5% linear warmup, then cosine decay</td></tr><tr><td>Batch size / gradient accumulation</td><td>8 / 4 steps</td></tr><tr><td>Maximum epochs / early-stopping patience</td><td>10 / 3 validation checks</td></tr><tr><td>Gradient clipping / precision</td><td>Global norm 1.0 / FP32</td></tr><tr><td>Checkpoint criterion</td><td>Lowest validation MSE within each run</td></tr><tr><td>Configuration and final seed selection</td><td>Validation MSE</td></tr><tr><td>Random seeds for final retraining</td><td>42, 43, 44, 45, 46</td></tr></table>

AdamW uses separate learning rates for encoder parameters and newly trained pooling and head parameters. Biases, normalization parameters, and other one-dimensional parameters are excluded from weight decay. Linear warmup occupies the first 5% of planned optimizer updates; cosine decay subsequently reduces the learning rate to 10% of its peak at the final planned update. Validation MSE controls checkpoint retention and early stopping. All experiments use the same data partition, comprising 10,199 unique training pairs, 856 validation pairs, and 1,292 test pairs, as described in Appendix A.2.

## A.3.2 TRAINING OBJECTIVE

Let $y _ { n }$ be the scalar target of pair $( r _ { n } , h _ { n } ) , s _ { n } = f _ { \theta } ( r _ { n } , h _ { n } )$ its regression output, and $z _ { n j }$ the ordinal logit associated with threshold $\tau _ { j }$ . For a minibatch of N pairs, the final objective is

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { m s e } } + 0 . 2 \mathcal { L } _ { \mathrm { b o u n d a r y } } + 0 . 5 \mathcal { L } _ { \mathrm { o r d i n a l } } .\tag{24}
$$

The three terms supervise numerical accuracy, interval membership, and cumulative threshold prediction, respectively:

$$
\begin{array} { c } { \displaystyle \mathcal { L } _ { \mathrm { m s e } } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } ( s _ { n } - y _ { n } ) ^ { 2 } , } \\ { \displaystyle \mathcal { L } _ { \mathrm { b o u n d a r y } } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \left( [ \ell _ { n } - s _ { n } ] _ { + } ^ { 2 } + [ s _ { n } - u _ { n } ] _ { + } ^ { 2 } \right) , } \\ { \displaystyle \mathcal { L } _ { \mathrm { o r d i n a l } } = \frac { 1 } { N J } \sum _ { n = 1 } ^ { N } \sum _ { j = 1 } ^ { J } \mathrm { B C E W i t h L o g i t s } ( z _ { n j } , \mathbf { 1 } [ y _ { n } \geq \tau _ { j } ] ) , } \end{array}\tag{25}
$$

where $[ x ] _ { + } = \operatorname* { m a x } ( x , 0 ) , J = 6 ,$ , and $( \tau _ { 1 } , \dots , \tau _ { J } ) = ( 1 / 3 , 2 / 3 , 1 , 4 / 3 , 5 / 3 , 2 )$ . The ordinal targets are derived entirely from $y _ { n }$ , requiring no additional annotations. All thresholds and positive classes receive unit weight in the final configuration.

The bounds $\ell _ { n }$ and $u _ { n }$ are the adjacent points of a fixed boundary grid containing $y _ { n }$ . To construct this grid, let $( a _ { 0 } , \dots , a _ { J + 1 } ) = ( 0 , \tau _ { 1 } , \dots , \tau _ { J } , 7 / 3 )$ . Each coarse interval is subdivided using

$$
{ \mathcal { B } } _ { q } = \bigcup _ { j = 0 } ^ { J } \left\{ a _ { j } + ( 1 - \rho ^ { k } ) ( a _ { j + 1 } - a _ { j } ) : k = 0 , \ldots , q - 1 \right\} \cup \{ 7 / 3 \} .\tag{26}
$$

We use $\rho = 1 0 ^ { - 1 / 6 }$ and $q = 2 ,$ , giving 14 fine intervals. Intervals are left-closed and right-open, except that the last includes $7 / 3 ;$ comparisons use a tolerance of $1 0 ^ { - 6 }$ for rounded labels. Boundary loss is zero for predictions within the target interval and grows quadratically outside it. This grid is a training constraint on the scalar prediction; constructing it does not require gap counts at inference time.

## A.3.3 DESIGN ABLATIONS

We examine pooling, auxiliary losses, input representation, and encoder adaptation while retaining the same data partition. Checkpoints within each run, model configurations, and the final random seed are selected using validation MSE. The regression results in this and the following subsections are reported on the 1,292-pair test split against the constructed scalar targets, using raw, unclipped predictions. Lower MSE and MAE indicate smaller numerical errors, while higher Pearson correlation r and Spearman correlation $\rho _ { s }$ indicate closer agreement with the targets. Multi-seed results are means and sample standard deviations.

Pooling. We first compare five pooling methods under MSE-only training with seed 42. In addition to the last-layer first-token representation (CLS) and masked mean, we consider concatenation and a learned weighted mixture of the mean-pooled last four layers, as well as LSTM pooling. We select LSTM pooling by validation MSE for the subsequent regression experiments. On the test split, Table 11 shows that LSTM pooling achieves the lowest MSE and MAE, whereas CLS yields the highest Spearman correlation.

Table 11: Pooling comparison with MSE-only training and seed 42. Bold values indicate the best result in each column.
<table><tr><td>Pooling</td><td>MSE↓</td><td>MAE↓</td><td>Pearson  $r \uparrow$ </td><td>Spearman  $\rho _ { s } \uparrow$ </td></tr><tr><td>CLS</td><td>0.138204</td><td>0.261133</td><td>0.810322</td><td>0.813424</td></tr><tr><td>Mean</td><td>0.143248</td><td>0.277111</td><td>0.795377</td><td>0.793949</td></tr><tr><td>Last-four-layer concatenation</td><td>0.150199</td><td>0.275274</td><td>0.794139</td><td>0.789104</td></tr><tr><td>Learned layer mixture</td><td>0.147843</td><td>0.267151</td><td>0.798982</td><td>0.796896</td></tr><tr><td>LSTM</td><td>0.129790</td><td>0.255395</td><td>0.812736</td><td>0.807285</td></tr></table>

Auxiliary losses. An initial comparison of MSE alone and its combinations with boundary and ordinal losses under CLS and LSTM pooling yielded a lowest MSE of 0.128095 for LSTM with

Table 12: Loss-weight comparison with LSTM pooling and full encoder fine-tuning. Results are mean ± sample standard deviation over five seeds; $\lambda _ { \mathrm { m s e } } = 1$ throughout. Bold marks the best means.
<table><tr><td> $\lambda _ { \mathrm { b o u n d a r y } }$ </td><td> $\lambda _ { \mathrm { o r d i n a l } }$ </td><td>MSE↓</td><td> $\mathrm { S p e a r m a n } \rho _ { s } \uparrow$ </td></tr><tr><td>0.0</td><td>0.0</td><td> $0 . 1 3 4 5 6 5 \pm 0 . 0 0 4 4 2 7$ </td><td> $0 . 8 0 2 0 4 6 \pm 0 . 0 0 2 0 3 4$ </td></tr><tr><td>0.0</td><td>0.1</td><td> $0 . 1 3 1 8 3 0 \pm 0 . 0 0 1 9 9 0$ </td><td> $0 . 8 0 2 6 7 7 \pm 0 . 0 0 2 4 0 7$ </td></tr><tr><td>0.0</td><td>0.9</td><td> $0 . 1 3 1 2 9 3 \pm 0 . 0 0 1 8 6 5$ </td><td> $0 . 8 0 8 1 9 3 \pm 0 . 0 0 3 2 6 2$ </td></tr><tr><td>0.3</td><td>0.0</td><td> $0 . 1 3 3 4 3 4 \pm 0 . 0 0 2 8 0 2$ </td><td> $0 . 8 0 3 9 7 1 \pm 0 . 0 0 5 7 2 5$ </td></tr><tr><td>0.3</td><td>0.9</td><td> $0 . 1 3 0 4 7 5 \pm 0 . 0 0 2 7 8 1$ </td><td> $0 . 8 0 6 0 2 9 \pm 0 . 0 0 6 5 5 0$ </td></tr><tr><td>0.2</td><td>0.5</td><td> $\mathbf { 0 . 1 2 7 0 4 0 } \pm 0 . 0 0 4 4 6 6$ </td><td> $\mathbf { 0 . 8 0 8 9 2 9 } \pm 0 . 0 0 5 0 6 5$ </td></tr><tr><td>0.4</td><td>0.1</td><td> $0 . 1 3 0 0 9 2 \pm 0 . 0 0 3 1 0 0$ </td><td> $0 . 8 0 4 0 5 8 \pm 0 . 0 0 2 7 1 7$ </td></tr><tr><td>0.4</td><td>0.8</td><td> $0 . 1 3 0 2 3 8 \pm 0 . 0 0 3 9 1 7$ </td><td> $0 . 8 0 7 4 2 6 \pm 0 . 0 0 3 0 3 0$ </td></tr></table>

boundary loss. We then searched a $. 9 \times 9$ grid with $\lambda _ { \mathrm { b o u n d a r y } } , \lambda _ { \mathrm { o r d i n a l } } \in \{ 0 . 1 , 0 . 2 , \ldots , 0 . 9 \}$ , fixing $\lambda _ { \mathrm { m s e } } = 1$ and seed 42. Neither weight exhibited a monotonic relationship with performance: the lowest MSE occurred at (0.3, 0.9), whereas the highest Spearman correlation occurred at (0.2, 0.3).  
To examine sensitivity to initialization, we evaluated eight loss configurations using seeds 42–46 (Table 12). Validation MSE is used to select $( \lambda _ { \mathrm { b o u n d a r y } } , \bar { \lambda } _ { \mathrm { o r d i n a l } } ) = ( \bar { 0 . 2 } , 0 . 5 )$ for the remaining ablations and final retraining. On the test split, this combination gives the lowest mean MSE, reducing it from 0.134565 to 0.127040 relative to MSE-only training, a relative reduction of 5.59%. It also gives the highest mean Spearman correlation among these eight configurations.

Table 13: Input and encoder-adaptation ablations. Each row reports five-seed mean ± sample standard deviation with LSTM pooling and loss weights (1, 0.2, 0.5). The original-input and full-finetuning rows reuse the same baseline runs.
<table><tr><td>Configuration</td><td>MSE↓</td><td>Spearman  $\rho _ { s } \mathrm { ~ \ } \mathrm { \uparrow }$ </td></tr><tr><td colspan="3">Input representation; encoder fully trainable</td></tr><tr><td>Original reference-hypothesis pair</td><td> $\mathbf { 0 . 1 2 7 0 4 0 } \pm 0 . 0 0 4 4 6 6$ </td><td> $\mathbf { 0 . 8 0 8 9 2 9 } \pm 0 . 0 0 5 0 6 5$ </td></tr><tr><td>Inline gap markers</td><td> $0 . 1 3 6 6 9 8 \pm 0 . 0 0 2 3 1 3$ </td><td> $0 . 8 0 2 9 4 5 \pm 0 . 0 0 3 3 3 6$ </td></tr><tr><td>Appended gap spans</td><td> $0 . 1 4 0 3 5 6 \pm 0 . 0 0 5 4 6 1$ </td><td> $0 . 7 9 8 8 0 1 \pm 0 . 0 0 3 9 6 3$ </td></tr><tr><td colspan="3">Encoder adaptation; original transcript input</td></tr><tr><td>Frozen encoder</td><td> $0 . 1 \dot { 6 } 1 9 1 5 \pm 0 . 0 0 2 6 6 0$ </td><td> $0 . 7 6 4 5 2 5 \pm 0 . 0 0 4 7 9 5$ </td></tr><tr><td>Last 4 layers</td><td> $0 . 1 4 1 6 1 4 \pm 0 . 0 0 1 6 9 0$ </td><td> $0 . 7 9 4 6 4 1 \pm 0 . 0 0 3 4 3 2$ </td></tr><tr><td>Last 8 layers</td><td> $0 . 1 3 8 3 9 0 \pm 0 . 0 0 4 8 5 5$ </td><td> $0 . 8 0 3 4 1 2 \pm 0 . 0 0 2 4 4 1$ </td></tr><tr><td>Last 12 layers</td><td> $0 . 1 3 7 5 1 3 \pm 0 . 0 0 4 6 7 5$ </td><td> $0 . 7 9 9 9 4 8 \pm 0 . 0 0 4 6 1 7$ </td></tr><tr><td>Last 16 layers</td><td> $0 . 1 3 6 0 4 4 \pm 0 . 0 0 3 7 9 6$ </td><td> $0 . 8 0 3 1 1 5 \pm 0 . 0 0 2 5 1 8$ </td></tr><tr><td>Last 20 layers</td><td> $0 . 1 3 4 4 2 2 \pm 0 . 0 0 2 9 8 9$ </td><td> $0 . 8 0 2 0 7 3 \pm 0 . 0 0 5 1 2 0$ </td></tr><tr><td>Last 24 layers</td><td> $0 . 1 2 8 2 2 3 \pm 0 . 0 0 3 7 5 8$ </td><td> $0 . 8 0 7 6 4 4 \pm 0 . 0 0 1 4 8 7$ </td></tr><tr><td>Full fine-tuning</td><td> $\mathbf { 0 . 1 2 7 0 4 0 } \pm 0 . 0 0 4 4 6 6$ </td><td> $\mathbf { 0 . 8 0 8 9 2 9 } \pm 0 . 0 0 5 0 6 5$ </td></tr></table>

Input representation and encoder adaptation. We compare the original transcript pair with two ways of exposing CGD gaps to the encoder: inserting markers around gaps at their original positions, or appending marked gap spans to each transcript. All three use LSTM pooling and the same loss weights. Validation MSE is used to select the original transcript pair as input, while CGD is used during offline target construction. As shown in Table 13, neither form of explicit gap input improves the five-seed mean test MSE or Spearman correlation.

We also compare full encoder fine-tuning with a frozen encoder and with training only its last K Transformer layers. The pooling and prediction heads remain trainable in every condition. In creasing the number of trainable encoder layers generally reduces MSE, with $K = 2 4$ approaching full fine-tuning. The K-layer policy keeps embeddings and other encoder parameters outside the selected layers frozen, so it differs from full fine-tuning even when all Transformer layers are train able. Full fine-tuning is selected using validation MSE and also gives the lowest mean test MSE in this comparison.

## A.3.4 HYPERPARAMETER SENSITIVITY

Learning rates. We vary the encoder learning rate over $\{ 3 \times 1 0 ^ { - 6 } , 6 \times 1 0 ^ { - 6 } , 1 0 ^ { - 5 } , 2 \times 1 0 ^ { - 5 } \}$ and the pooling/head learning rate over $\{ 5 \times 1 0 ^ { - 5 } , 1 0 ^ { - 4 } , 2 \times 1 0 ^ { - 4 } , 4 \times 1 0 ^ { - 4 } \}$ , using seeds 42 and 43. Validation MSE is used to select $( 6 \times 1 0 ^ { - 6 } , 1 0 ^ { - 4 } )$ for the final configuration. Table 14 reports mean test MSE for all 16 combinations. The selected pair gives the lowest mean test MSE, whereas $( 6 \times 1 0 ^ { - 6 } , 2 \times 1 0 ^ { - 4 } )$ gives the highest mean test Spearman correlation (0.812878 versus 0.811107). The latter also yields the lowest single-run test MSE in the search (0.121435).

Table 14: Learning-rate sensitivity: mean MSE over seeds 42 and 43. Columns specify the learning rate for pooling and prediction heads. The bold entry is the lowest mean.
<table><tr><td>Encoder learning rate</td><td> $5 \times 1 0 ^ { - 5 }$ </td><td> $1 0 ^ { - 4 }$ </td><td> $2 \times 1 0 ^ { - 4 }$ </td><td> $4 \times 1 0 ^ { - 4 }$ </td></tr><tr><td> $3 \times 1 0 ^ { - 6 }$ </td><td>0.134568</td><td>0.133172</td><td>0.133939</td><td>0.134907</td></tr><tr><td> $6 \times 1 0 ^ { - 6 }$ </td><td>0.128956</td><td>0.123872</td><td>0.124715</td><td>0.129092</td></tr><tr><td> $1 0 ^ { - 5 }$ </td><td>0.127830</td><td>0.130582</td><td>0.128878</td><td>0.128369</td></tr><tr><td> $2 \times 1 0 ^ { - 5 }$ </td><td>0.136164</td><td>0.138593</td><td>0.135775</td><td>0.132319</td></tr></table>

Ordinal thresholds and boundary resolution. The ordinal thresholds also determine the coarse intervals in Eq. 26. We compare the six-threshold configuration with a two-threshold alternative, $\{ 2 / 3 , 4 / 3 \}$ , and vary the boundary resolution $q \in \{ 2 , 3 , 4 \}$ . We select six thresholds with $q =$ 2 using validation MSE. Table 15 shows that this configuration also gives the lowest mean test MSE. Increasing q does not consistently improve regression; the highest mean Spearman correlation, obtained with six thresholds and $q = 4 ,$ exceeds the $q = 2$ result by only 0.000079.

Table 15: Ordinal-threshold and boundary-resolution comparison. Results are mean ± sample standard deviation over seeds 42 and 43. The six-threshold, $q = 2$ row reuses the learning-rate baseline.
<table><tr><td>Ordinal thresholds</td><td>q</td><td>MSE↓</td><td> $\mathrm { S p e a r m a n } \rho _ { s } \uparrow$ </td></tr><tr><td>Six thresholds</td><td>2</td><td> $\mathbf { 0 . 1 2 3 8 7 2 } \pm 0 . 0 0 2 9 9 0$ </td><td> $0 . 8 1 1 1 0 7 \pm 0 . 0 0 6 7 0 1$ </td></tr><tr><td>Six thresholds</td><td>3</td><td> $0 . 1 3 1 2 9 1 \pm 0 . 0 0 0 1 6 1$ </td><td> $0 . 8 0 5 5 5 3 \pm 0 . 0 0 1 1 8 3$ </td></tr><tr><td>Six thresholds</td><td>4</td><td> $0 . 1 2 9 1 7 6 \pm 0 . 0 0 3 4 1 8$ </td><td> $\mathbf { 0 . 8 1 1 1 8 6 \pm 0 . 0 0 2 7 6 2 }$ </td></tr><tr><td>Two thresholds</td><td>2</td><td> $0 . 1 3 4 3 0 9 \pm 0 . 0 0 2 9 4 6$ </td><td> $0 . 8 0 0 7 7 7 \pm 0 . 0 0 9 8 7 5$ </td></tr><tr><td>Two thresholds</td><td>3</td><td> $0 . 1 3 1 7 2 6 \pm 0 . 0 0 0 9 2 1$ </td><td> $0 . 8 0 2 8 2 3 \pm 0 . 0 0 8 8 5 7$ </td></tr><tr><td>Two thresholds</td><td>4</td><td> $0 . 1 3 3 6 4 2 \pm 0 . 0 0 4 8 7 5$ </td><td> $0 . 8 0 4 6 3 6 \pm 0 . 0 0 8 8 3 7$ </td></tr></table>

Ordinal reweighting. Finally, we examine whether weighting ordinal thresholds or positive classes improves the scorer. In threshold order, we test four normalized weight configurations: uniform $( 1 , \ldots , 1 )$ , low-emphasis $( 1 . 5 , 1 . 3 , \ldots , 0 . 5 )$ , high-emphasis $( 0 . 5 , 0 . 7 , \ldots , 1 . 5 )$ , and middleemphasis $( 0 . 6 , 1 , 1 . 4 , 1 . 4 , 1 , 0 . 6 )$ . These are crossed with four positive-class weight vectors:

$$
\pmb { p } _ { 0 } = ( 1 , 1 , 1 , 1 , 1 , 1 ) ,
$$

$$
\pmb { p } _ { 1 } = ( 0 . 8 2 3 , 1 . 0 3 7 , 1 . 2 1 2 , 1 . 2 9 1 , 1 . 4 5 0 , 2 . 0 1 6 ) ,
$$

$$
\pmb { p } _ { 2 } = ( 0 . 6 7 7 , 1 . 0 7 5 , 1 . 4 6 9 , 1 . 6 6 7 , 2 . 1 0 3 , 4 . 0 6 5 ) ,
$$

$$
\pmb { p } _ { 3 } = ( 0 . 4 5 8 , 1 . 1 5 5 , 2 . 1 5 9 , 2 . 7 8 0 , 4 . 4 2 2 , 1 6 . 5 2 4 ) .
$$

Each combination uses five seeds. The final scorer retains unit weights based on validation MSE. Table 16 reports the test results, where $\mathbf { \mathit { p } } _ { 2 }$ with high-threshold emphasis reduces mean MSE from 0.127040 to 0.126756, a relative change of 0.22%. The highest mean Spearman correlation instead occurs with $\mathbf { p } _ { 3 }$ and uniform threshold weights (0.811850).

## A.3.5 FINAL SCORER AND EVALUATION PROTOCOL

We retrain the fixed configuration in Table 10 with five random seeds. Within each run, we retain the checkpoint with the lowest validation MSE, then select the final checkpoint across runs using the same validation criterion. This selects seed 46 for all external ASR evaluations reported in Table 1.

Table 16: Ordinal reweighting: five-seed mean MSE. Columns indicate threshold-loss weighting; rows indicate positive-class weighting. The bold entry is the lowest mean.
<table><tr><td>Positive-class weights</td><td>Uniform</td><td>Low</td><td>High</td><td>Middle</td></tr><tr><td>po</td><td>0.127040</td><td>0.131188</td><td>0.127591</td><td>0.132335</td></tr><tr><td>p1</td><td>0.128455</td><td>0.128698</td><td>0.127074</td><td>0.127770</td></tr><tr><td>p2</td><td>0.128300</td><td>0.131183</td><td>0.126756</td><td>0.127376</td></tr><tr><td>p3</td><td>0.128768</td><td>0.129774</td><td>0.128308</td><td>0.127996</td></tr></table>

Table 17 reports the test results for all five runs, separately from the development runs used in the preceding ablations. The resulting mean MSE is $0 . { \overset { \cdot } { 1 } } 2 7 6 9 { \overset { \cdot } { 3 } } \pm 0 . 0 0 2 3 1 8$ , and mean Spearman correlation is $0 . 8 0 8 0 8 0 \pm 0 . 0 0 4 3 6 8$ . The final scorer uses the fixed configuration rather than combining the best individual outcomes from different hyperparameter searches.

Table 17: Independent retraining of the final scorer configuration. All results use the same 1,292- pair test split and raw predictions. Bold values identify the best result in each column.
<table><tr><td>Seed</td><td>MSE↓</td><td>MAE↓</td><td>Pearson  $r \uparrow$ </td><td>Spearman  $\rho _ { s } \mathrm { ~ \ } \mathrm { \uparrow }$ </td></tr><tr><td>42</td><td>0.126302</td><td>0.254662</td><td>0.818037</td><td>0.807938</td></tr><tr><td>43</td><td>0.129235</td><td>0.255753</td><td>0.811441</td><td>0.804729</td></tr><tr><td>44</td><td>0.130549</td><td>0.250619</td><td>0.810363</td><td>0.804272</td></tr><tr><td>45</td><td>0.127689</td><td>0.244722</td><td>0.819069</td><td>0.808274</td></tr><tr><td>46</td><td>0.124689</td><td>0.245155</td><td>0.821666</td><td>0.815188</td></tr></table>

For external evaluation, we freeze the scorer and apply it to seven ASR systems on 647 FLEURS English utterances, 2,611 LibriSpeech test-clean utterances, and 1,830 VoxPopuli English utterances per system. The resulting 21 system–dataset combinations contain 35,616 transcript pairs in total. Each pair is scored independently in a single forward pass. We apply the clipping and normalization in Eq. 11 to each prediction before taking the arithmetic mean within a system–dataset combination. These normalized means are the SITAR values reported in the main comparison table. The external datasets do not carry the constructed scalar targets used for scorer training, so the external analysis examines ASR-system behavior rather than reporting target-regression error.

## B DETAILED COMPARISON WITH AER

AER and SITAR both assess consequences of transcription errors beyond lexical mismatch. AER measures the fraction of reference-derived questions whose answers differ when conditioned on the reference and the ASR hypothesis (Pulikodan et al., 2025). SITAR instead assigns a situational distortion score directly to the transcript pair. We compare their system-level behavior using the 21 entries in Table 1. The AER entries come from the original paper’s evaluation subsets, while the SITAR entries use the data described in Appendix A.3.5. The analysis focuses on model ordering, numerical separation between system means, and the computation required at scoring time.

## B.1 AGREEMENT IN MODEL TRENDS

Both metrics assign a lower score to whisper-large than to whisper-tiny on all three datasets. They also favor wav2vec2-large over wav2vec2-base in every dataset, and both identify whisper-large as the best of the seven evaluated systems throughout. These shared preferences show that SITAR retains broad model-scaling trends also captured by the comprehension-based evaluation.

Agreement is strongest on FLEURS, where the two metrics produce exactly the same ordering of all seven systems. On LibriSpeech, 20 of the 21 pairwise orderings agree. The only reversal concerns whisper-small and whisper-medium. AER slightly favors whisper-small, with scores of 0.2600 and 0.2637, whereas SITAR favors whisper-medium, with scores of 0.1047 and 0.0818. The corresponding Spearman correlations between the seven system scores are 1.0000 on FLEURS and 0.9643 on LibriSpeech, as summarized in Table 18.

The ordering differs more substantially on VoxPopuli. Across the five Whisper sizes, AER takes values 0.4723, 0.4505, 0.4567, 0.4739, and 0.4345, while SITAR decreases at every step from 0.3157 to 0.1684. Cross-family preferences also differ. For example, AER favors wav2vec2-large over whisper-base, with scores of 0.4366 and 0.4505, while SITAR favors whisper-base, with scores of 0.2532 and 0.3577. Consequently, the seven-system Spearman correlation on VoxPopuli is −0.1071. The shared preference for the largest model within each family therefore coexists with different finer-grained rankings on this dataset.

Table 18: Score separation and rank agreement derived from the seven systems per dataset in Table 1. A span is the maximum score minus the minimum score. The span ratio is SITAR divided by AER, and $\rho _ { s }$ is their Spearman rank correlation.
<table><tr><td>Dataset</td><td>AER span</td><td>SITAR span</td><td>Span ratio</td><td> $\rho _ { s }$ </td></tr><tr><td>FLEURS</td><td>0.1001</td><td>0.2484</td><td>2.48</td><td>1.0000</td></tr><tr><td>LibriSpeech</td><td>0.0469</td><td>0.1623</td><td>3.46</td><td>0.9643</td></tr><tr><td>VoxPopuli</td><td>0.0394</td><td>0.2400</td><td>6.09</td><td>-0.1071</td></tr></table>

## B.2 SEPARATION BETWEEN SYSTEM SCORES

Both metrics are reported on a [0, 1] scale, with lower values indicating less distortion. Within this range, SITAR uses a wider span to distinguish the evaluated systems on every dataset. On FLEURS, the seven AER values lie between 0.1477 and 0.2478, giving a span of 0.1001. The corresponding SITAR values range from 0.1388 to 0.3872, giving a span of 0.2484. The span is therefore 2.48 times larger while preserving exactly the same model ranking. This is the clearest case in Table 1 of agreement in ordering accompanied by greater numerical separation.

The difference in span is larger on LibriSpeech and VoxPopuli. On LibriSpeech, AER occupies the interval from 0.2540 to 0.3009, whereas SITAR occupies 0.0634 to 0.2257, giving spans of 0.0469 and 0.1623. On VoxPopuli, all AER values are concentrated between 0.4345 and 0.4739, a span of 0.0394, while SITAR spans 0.1684 to 0.4084, a width of 0.2400. These produce the respective span ratios of 3.46 and 6.09 in Table 18. Here, separation refers specifically to distances between reported system means on each metric’s scale.

Several individual comparisons illustrate what this difference looks like in practice. On FLEURS, whisper-medium and whisper-large differ by only 0.0031 under AER, compared with 0.0202 under SITAR. On LibriSpeech, both metrics favor whisper-small over wav2vec2-large, but the difference is 0.0043 under AER and 0.0297 under SITAR. For wav2vec2-base and wav2vec2-large, the gaps are respectively 0.0092 versus 0.0423 on FLEURS, 0.0053 versus 0.0241 on LibriSpeech, and 0.0021 versus 0.0507 on VoxPopuli. Thus, several improvements that occupy a narrow numerical interval under AER are more visibly separated by SITAR.

The amount of separation nevertheless depends on the model pair. For whisper-medium and whisper-large on VoxPopuli, SITAR gives nearly equal scores of 0.1701 and 0.1684, whereas AER gives 0.4739 and 0.4345. Together with the ordering differences above, this shows that the wider overall spans coexist with selective compression of individual contrasts. SITAR’s numerical separation reflects its own severity-based scoring rule rather than a uniform expansion of the AER values.

## B.3 COMPUTATION AT SCORING TIME

AER’s published workflow comprises question generation, separate answer generation from the reference and hypothesis, and comparison of the two answer sets (Pulikodan et al., 2025). Evaluating an uncached pair therefore involves four LLM calls. The reported setup uses Gemini 1.5 Flash for questions, GPT-4o-mini for both answer sets, and GPT-4o for judging. Questions and reference answers could be cached across hypotheses, but hypothesis answering and answer comparison would still require LLM inference.

For SITAR, LLM annotation and review are confined to offline supervision construction. Once trained, the DeBERTa-based scorer takes the reference and hypothesis together and returns a scalar in one forward pass. Scoring requires no question generation, answer generation, or LLM judgment.

Evaluating another ASR system or checkpoint therefore adds only direct scorer passes over its transcript pairs. This removes repeated generative inference from routine evaluation while retaining the model-level trends and score separation described above.

## C CONTROLLED DATASET CONSTRUCTION AND HUMAN ANNOTATION

This appendix describes the controlled dataset $\mathcal { D } _ { \mathrm { c t r l } }$ used in Section 4.3. We explain how fixed edits produce comparable hypothesis variants, illustrate the construction with a complete example from the dataset, and detail the human annotation and subsequent analyses.

## C.1 DATASET COMPOSITION AND CONTROLLED CONSTRUCTION

The dataset contains 30 English references, each paired with 11 hypothesis variants, yielding 330 distinct reference–hypothesis pairs. The references cover five domains, with the distribution shown in Table 19. Twenty-five references come from curated ASR examples, retaining two semantic edits from their ASR hypotheses. The remaining five references are designed scenarios with two specified semantic edits. Each reference also receives two synthetic expression-level changes, consisting of an immediate word repetition and a filled pause. Thus, the construction uses 120 fixed edits, comprising 50 ASR-derived semantic edits, 10 designed semantic edits, and 60 synthetic disfluencies.

Table 19: Composition of the controlled dataset. Each reference has four candidate gaps for annotation and 11 hypothesis variants for evaluation. Culture includes cultural and sports content.
<table><tr><td>Domain</td><td>References</td><td>Candidate gaps</td><td>Transcript pairs</td></tr><tr><td>Culture</td><td>9</td><td>36</td><td>99</td></tr><tr><td>Finance</td><td>4</td><td>16</td><td>44</td></tr><tr><td>Services</td><td>7</td><td>28</td><td>77</td></tr><tr><td>Public affairs</td><td>5</td><td>20</td><td>55</td></tr><tr><td>General communication</td><td>5</td><td>20</td><td>55</td></tr><tr><td>Total</td><td>30</td><td>120</td><td>330</td></tr></table>

At construction time, the four edits for each reference have provisional assignments of two $\mathcal { T } _ { 0 }$ gaps, one $\mathcal { T } _ { 1 }$ gap, and one $\mathcal { T } _ { 2 }$ gap. These assignments specify the planned experimental combinations and are subsequently checked by human annotators. Starting from the unchanged reference, we apply selected subsets of the four edits to obtain the 11 profiles in Table 21. Every profile therefore occurs once for each reference. A profile requiring one $\mathcal { T } _ { 0 }$ edit always uses the earlier of the two edits in reference character order; a profile requiring two uses both. The $\mathcal { T } _ { 1 }$ and $\mathcal { T } _ { 2 }$ edits likewise remain identical whenever they recur across profiles.

This construction holds the reference, surrounding context, and wording of each retained edit fixed. For example, the difference between a $\mathcal { T } _ { 2 }$ variant and a $\mathcal { T } _ { 2 } + \mathcal { T } _ { 1 }$ variant is the addition of the same predefined $\mathcal { T } _ { 1 }$ edit, while the $\mathcal { T } _ { 2 }$ edit remains unchanged. Comparing variants within a reference therefore isolates specified changes in error composition. Comparing the same construction profile across references examines how the scorer responds to that composition in different contexts.

The edits are applied at recorded, nonoverlapping character spans. We verify that reversing all retained edits reconstructs the reference exactly, that the four-edit variant reproduces its parent hypothesis, and that all 330 reference–hypothesis pairs are unique. Gap identities are inherited from these controlled edits. The counts in this experiment refer to the recorded edit units, without rerunning the full CGD segmentation on every generated variant. The 30 references were retained from a pool of 35 using the prediction-based selection procedure described in Appendix C.6.

## C.2 A COMPLETE CONSTRUCTION EXAMPLE

Reference 27 concerns approval of a draft before sending it to a client. Its English reference is reproduced below.

I haven’t approved the draft yet. The figures look right, but I still need to check the conclusion. Do not send it to the client until I approve it. Maya can keep working on the appendix in the meantime.

Table 20 gives its four fixed edits. The two $\mathcal { T } _ { 0 }$ edits introduce disfluencies, repeating $^ { 6 6 } \mathrm { I } ^ { \prime }$ at the beginning and inserting “uh,” before “meantime.” Neither changes the approval status or the instruction about sending the draft. The $\mathcal { T } _ { 1 }$ edit changes the assessment of the figures from “The figures look right” to “Most of the figures look right.” This changes what the listener understands about the figures, while preserving both the pending approval and the prohibition on sending the draft. The $\breve { \tau } _ { 2 }$ edit reverses that prohibition, allowing the draft to be sent before approval. Both annotators agreed on all four assignments in this example.

Table 20: The four fixed edits in reference 27. The symbol ∅ denotes an insertion. $G _ { 0 }$ is inserted at the beginning of the reference, and $G _ { 3 }$ immediately before “meantime.” Gap IDs indicate positions, independently of severity.
<table><tr><td>Gap</td><td>Reference span</td><td>Hypothesis span</td><td>Tier</td></tr><tr><td> $G _ { 0 }$ </td><td>∅</td><td>I</td><td> $\mathcal { T } _ { 0 }$ </td></tr><tr><td> $G _ { 1 }$ </td><td>The figures</td><td>Most of the figures</td><td> $\mathcal { T } _ { 1 }$ </td></tr><tr><td> $G _ { 2 }$ </td><td>Do not send it to the client until</td><td>You can send it to the client before I ap-</td><td> $\mathcal { T } _ { 2 }$ </td></tr><tr><td> $G _ { 3 }$ </td><td>I approve it. ∅</td><td>prove it. uh,</td><td> $\mathcal { T } _ { 0 }$ </td></tr></table>

The 11 variants are obtained by retaining the subsets in Table 21. The gap IDs in this table refer to the four parent edits, so that the same ID always identifies the same textual change. The suffixes a–k identify the construction profiles used throughout the analysis. The example’s predicted scores are included to show the actual output for each resulting pair.

Table 21: All 11 variants of reference 27. Each hypothesis applies exactly the listed edits to the reference. The final column reports normalized SITAR predictions, rounded to four decimal places. These construction profiles are used for all 30 references.
<table><tr><td>Variant</td><td>Construction profile</td><td>Retained edits</td><td>SITAR</td></tr><tr><td>a</td><td> $\mathcal { T } _ { 0 }$ </td><td> $G _ { 0 }$ </td><td>0.0274</td></tr><tr><td>b</td><td> $2 \mathcal { T } _ { 0 }$ </td><td> $G _ { 0 } , G _ { 3 }$ </td><td>0.0564</td></tr><tr><td> $c$ </td><td> $\mathcal { T } _ { 1 }$ </td><td> $G _ { 1 }$ </td><td>0.3204</td></tr><tr><td> $d$ </td><td> $\mathcal { T } _ { 1 } + \mathcal { T } _ { 0 }$ </td><td> $G _ { 0 } , G _ { 1 }$ </td><td>0.3356</td></tr><tr><td> $e$ </td><td> $\mathcal { T } _ { 1 } + 2 \mathcal { T } _ { 0 }$ </td><td> $G _ { 0 } , G _ { 1 } , G _ { 3 }$ </td><td>0.3774</td></tr><tr><td> $f$ </td><td> $\mathcal { T } _ { 2 }$ </td><td> $G _ { 2 }$ </td><td>0.7573</td></tr><tr><td> $g$ </td><td> $\mathcal { T } _ { 2 } + \mathcal { T } _ { 0 }$ </td><td> $G _ { 0 } , G _ { 2 }$ </td><td>0.7751</td></tr><tr><td> $h$ </td><td> $\mathcal { T } _ { 2 } + 2 \mathcal { T } _ { 0 }$ </td><td> $G _ { 0 } , G _ { 2 } , G _ { 3 }$ </td><td>0.7952</td></tr><tr><td> $i$ </td><td> $\mathcal { T } _ { 2 } + \mathcal { T } _ { 1 }$ </td><td> $G _ { 1 } , G _ { 2 }$ </td><td>0.7678</td></tr><tr><td> $j$ </td><td> $\mathcal { T } _ { 2 } + \mathcal { T } _ { 1 } + \mathcal { T } _ { 0 }$ </td><td> $G _ { 0 } , G _ { 1 } , G _ { 2 }$ </td><td>0.7824</td></tr><tr><td> $k$ </td><td> $\mathcal { T } _ { 2 } + \mathcal { T } _ { 1 } + 2 \mathcal { T } _ { 0 }$ </td><td> $G _ { 0 } , G _ { 1 } , G _ { 2 } , G _ { 3 }$ </td><td>0.7934</td></tr></table>

For variant $c ,$ only $G _ { 1 }$ is applied. Its complete hypothesis is

I haven’t approved the draft yet. Most of the figures look right, but I still need to check the conclusion. Do not send it to the client until I approve it. Maya can keep working on the appendix in the meantime.

For variant $f ,$ only $G _ { 2 }$ is applied. The figure assessment returns to the reference wording, and the permission to send the draft changes instead.

I haven’t approved the draft yet. The figures look right, but I still need to check the conclusion. You can send it to the client before I approve it. Maya can keep working on the appendix in the meantime.

For variant k, all four edits are retained.

I I haven’t approved the draft yet. Most of the figures look right, but I still need to check the conclusion. You can send it to the client before I approve it. Maya can keep working on the appendix in the uh, meantime.

The reference and each selected change remain verbatim across these variants. Thus, moving from $f$ to i adds the same change in the figure assessment shown in c, and moving from i to k adds the two specified disfluencies. The variants form a controlled set because their differences are known and reusable at fixed positions within the same situation.

## C.3 ANNOTATION RULES AND PROCEDURE

Two PhD students independently annotated the 120 candidate gaps. Each received 30 records containing the complete English reference, the hypothesis with all four changes, and a list of the four candidate gaps. Each candidate identifies its changed span within the full reference context and has an initially empty level field. The annotators filled this field with T0, T1, or T2, preserving the record IDs, gap IDs, and transcript text. The task assigns nominal tiers only; it does not ask annotators to supply numerical distortion scores or directional boundary adjustments.

The instructions require the annotator to first read the complete reference and establish what it describes, communicates, or seeks to accomplish. Each gap is then judged through its single-gap hypothesis, with the other three edits restored to the reference. In the preceding example, judging $G _ { 1 }$ means comparing the reference with variant c. The permission reversal in $G _ { 2 }$ is absent from that comparison, so its consequences do not determine the label of $G _ { 1 }$

The three annotation rules are as follows.

1. $\mathcal { T } _ { 0 }$ applies when expression changes while the conveyed situation, understanding, and communicative purpose remain intact. A filler or immediate repetition is assigned to this tier when it introduces a disfluency without changing the relevant information or instructions.

2. $\mathcal { T } _ { 1 }$ applies when the listener’s understanding of some information changes but the intended outcome remains preserved. The annotator identifies both the information that changes and the purpose or requirement that survives. In reference 27, the assessment of the figures changes while approval remains pending and sending remains prohibited.

3. $\mathcal { T } _ { 2 }$ applies when the change affects the intended outcome, including what judgment should be made, what action should be executed, or what result is expected. In reference 27, the altered instruction permits an action that the reference explicitly prohibits before approval.

The instructions explicitly discourage assigning severity from the number of changed words, the mere presence of a lexical error, or the fluency of the hypothesis. A short edit may change an execution requirement, whereas a longer expression change may preserve the situation. The same lexical replacement may also receive different labels in different reference contexts. Annotators are not required to reproduce the planned allocation of two ${ \mathcal { T } } _ { 0 } ,$ one $\mathcal { T } _ { 1 } .$ and one $\mathcal { T } _ { 2 }$ gap. If a gap cannot be judged, the instructions allow its label to remain empty with the record and gap IDs flagged for review. Both returned annotations contained valid labels for all 120 gaps.

The annotation unit is the individual parent gap. Once its tier is assigned, that label is carried into every hypothesis variant containing the same edit. Annotators therefore make 120 judgments each, and these judgments supply labels for the 330 derived pairs. Before computing agreement, we align the two returned files by reference and gap IDs and verify that their reference text, hypothesis text, and marked contexts match. When mapping labels to derived variants, we match the reference and the full marked edit context, allowing the local gap numbering to change.

## C.4 ANNOTATION AGREEMENT AND FINAL LABELS

Agreement before adjudication. The two annotators agree on 108 of the 120 gaps, giving 90.00% exact agreement. Table 22 shows their full label counts. The 12 disagreements occur across nine references. Seven concern the $\mathcal { T } _ { 1 } { - } \mathcal { T } _ { 2 }$ boundary, three concern $\mathcal { T } _ { 0 } { - } \mathcal { T } _ { 1 }$ , and two concern $\mathcal { T } _ { 0 } \mathrm { - } \mathcal { T } _ { 2 }$ . Thus,

10 of the 12 disagreements involve adjacent tiers. The largest group concerns whether an information change preserves or alters the intended outcome.

Table 22: Human annotation confusion matrix before adjudication. Rows give annotator A’s labels and columns give annotator B’s labels for the same 120 gaps.
<table><tr><td></td><td>B  $\mathcal { T } _ { 0 }$ </td><td>B T1</td><td>B T2</td><td>Total</td></tr><tr><td>A  $\mathcal { T } _ { 0 }$ </td><td>58</td><td>0</td><td>0</td><td>58</td></tr><tr><td>A  $\mathcal { T } _ { 1 }$ </td><td>3</td><td>24</td><td>3</td><td>30</td></tr><tr><td>A T2</td><td>2</td><td>4</td><td>26</td><td>32</td></tr><tr><td>Total</td><td>63</td><td>28</td><td>29</td><td>120</td></tr></table>

Cohen’s κ is 0.8396. Treating the tiers as ordinal categories gives linearly weighted $\kappa = 0 . 8 6 7 5$ and quadratically weighted $\kappa = 0 . 8 9 2 4$ , using disagreement weights proportional to the absolute and squared tier distances, respectively. Their increase over unweighted κ reflects the predominance of adjacent-tier disagreements. These statistics use the two independent human label sets before any resolution with model labels.

At the transcript-pair level, exact agreement requires the two annotators to assign the same label to every retained gap in a variant. This holds for 268 of 330 pairs, or 81.21%, as reported in Table 3. One disputed parent gap appears in several variants, and a variant with multiple gaps must satisfy all corresponding label matches. Consequently, the lower pair-level agreement is a stricter summary of the same 120 judgments. If agreement is instead defined only by the counts of each tier, 274 pairs agree, or 83.03%. The six additional matches exchange the $\mathcal { T } _ { 1 }$ and $\mathcal { T } _ { 2 }$ assignments of two gaps while retaining the same tier counts. The main text uses the stricter gap-identity criterion.

Resolution of disagreements. Whenever the two human labels agree, their label is adopted. Otherwise, the initial model-assigned tier provides a third vote. It agrees with annotator A on five of the 12 disputed gaps and with annotator B on seven, resolving all disagreements without a three-way tie. This third vote is a categorical gap label from construction, distinct from the continuous SITAR prediction evaluated in Section 4.3. The final labels contain 60 $\mathcal { T } _ { 0 } .$ , 31 T<sub>1</sub>, and 29 $\mathcal { T } _ { 2 }$ gaps.

The final labels differ from the provisional construction assignments at one gap in reference 13. Both human annotators assign $\mathcal { T } _ { 1 }$ to a gap initially assigned $\bar { \mathcal { T } } _ { 2 } .$ , so the human agreement overrides that initial label. All six variants containing the gap inherit this correction. The underlying text and predicted scores stay the same. Consequently, the 11 construction-profile groups in Table 2 each retain 30 pairs, while their profile names record the original construction assignments. The within-reference ranking analysis uses the final adjudicated labels of every retained gap.

## C.5 RANKING PROTOCOL AND DETAILED RESULTS

For a hypothesis h belonging to reference $r ,$ let $c _ { k } ( r , h )$ be the number of retained gaps whose final label is $\mathcal { T } _ { k }$ . Its expected severity order is defined by the count vector

$$
\mathbf { v } ( r , h ) = \big ( c _ { 2 } ( r , h ) , c _ { 1 } ( r , h ) , c _ { 0 } ( r , h ) \big ) .
$$

We compare these vectors lexicographically, prioritizing $\mathcal { T } _ { 2 }$ counts, followed by $\mathcal { T } _ { 1 }$ counts and then $\mathcal { T } _ { 0 }$ counts. Equal vectors receive equal expected ranks. This evaluates the tier-priority and countrefinement behavior of the scorer using the final nominal labels. The expected ordering is derived from those labels under the stated aggregation rule; annotators do not separately rank all 330 hypotheses.

The predicted score is normalized as

$$
\widehat { s } ( r , h ) = \frac { \operatorname* { m i n } \{ 7 / 3 , \operatorname* { m a x } \{ 0 , f _ { \theta } ( r , h ) \} \} } { 7 / 3 } .
$$

All 330 raw predictions fall within the clipping bounds. For each reference, we compare all $\left( ^ { 1 1 } _ { 2 } \right) =$ 55 pairs of variants. If their count vectors differ, agreement requires the more severe variant to have a strictly higher predicted score. A predicted tie does not satisfy this strict ordering. Pairs with equal

expected count vectors are excluded from pairwise agreement, while Spearman’s $\rho _ { s }$ includes all 11 variants using average ranks for ties.

The correction in reference 13 changes its expected order to

$$
a < b < c = f < d = g < e = h < i < j < k .
$$

Its three expected ties, $( c , f ) , ( d , g )$ , and $( e , h )$ , reduce the number of strict comparisons for that reference from 55 to 52. The other 29 references each contribute 55, giving $2 9 \times \bar { 5 } 5 + 5 2 = 1 { , } 6 4 7$ comparisons. If $K _ { r }$ of the M<sub>r</sub> eligible comparisons agree for reference r, the main-text agreement is

$$
\mathrm { A g r . } = \frac { 1 } { 3 0 } \sum _ { r = 1 } ^ { 3 0 } \frac { K _ { r } } { M _ { r } } .
$$

Spearman correlations are also computed separately for each reference and then averaged. Domain results use the same averaging rule over references in that domain

Overall and domain results. The mean per-reference agreement is 84.07%, with a median of 85.45% and a mean Spearman correlation of 0.8001. In total, 1,385 of the 1,647 strict comparisons agree. Pooling all comparisons directly gives 84.09%; its small difference from the reported 84.07% follows from the unequal denominators of 52 and 55. Across domains, mean agreement ranges from 80.00% for services to 88.48% for culture, with mean $\rho _ { s }$ ranging from 0.7078 to 0.8758, as detailed in Table 4. These results summarize consistency across 30 reference groups, whose variants share text and edits.

Where ordering succeeds and fails. To locate the remaining ordering errors, Table 23 groups each strict comparison by the highest tier at which its two count vectors differ. $\textrm { A } \mathcal { T } _ { 1 }$ comparison therefore has identical $\mathcal { T } _ { 2 }$ counts but different $\mathcal { T } _ { 1 }$ counts. $\mathrm { ~ A ~ } \mathcal { T } _ { 0 }$ comparison has identical counts at both higher tiers and differs only in disfluency count. Each comparison belongs to exactly one group. The percentages in this table pool comparisons within each group.

Table 23: Ordering agreement by the highest tier with a count difference. Counts use final adjudicated labels. Agreement is the fraction of correct strict comparisons within each row.
<table><tr><td>Distinguishing tier</td><td>Comparisons</td><td>Correct</td><td>Agreement</td></tr><tr><td> $\mathcal { T } _ { 2 }$ </td><td>870</td><td>793</td><td>91.15%</td></tr><tr><td> $\mathcal { T } _ { 1 }$ </td><td>471</td><td>393</td><td>83.44%</td></tr><tr><td> $\mathcal { T } _ { 0 }$ </td><td>306</td><td>199</td><td>65.03%</td></tr><tr><td>Total</td><td>1,647</td><td>1,385</td><td>84.09%</td></tr></table>

Agreement is highest when the variants differ in $\mathcal { T } _ { 2 }$ count and lowest when only $\mathcal { T } _ { 0 }$ count differs. Of the 262 incorrect strict comparisons, 107 concern only the number of disfluencies. The scorer thus reproduces the larger severity distinctions more consistently than the smaller refinements from expression-level accumulation. Reference 27 illustrates this distinction. Its pure $\mathcal { T } _ { 0 } , \ \mathcal { T } _ { 1 }$ , and $\mathcal { T } _ { 2 }$ variants receive scores of 0.0274, 0.3204, and 0.7573. However, variant h scores 0.7952, above $i ,$ $j ,$ and $k ,$ despite the latter retaining the additional $\mathcal { T } _ { 1 }$ edit and having higher expected ranks. This example shows both the separation of basic tiers and a failure to preserve their full compositional ordering.

## C.6 REFERENCE SELECTION AND SCOPE OF THE ANALYSIS

The final 30 references were retained from 35 candidate references after inspecting the existing SITAR predictions. The recorded selection criterion maximizes the mean-score separation between the $\mathcal { T } _ { 2 }$ construction family $( f , g , h )$ and the $\mathcal { T } _ { 1 }$ construction family $( c , d , e )$ , standardized by the square root of the mean of the six within-profile sample variances. Five references were excluded through this prediction-based selection. Their exclusion was not established independently as an annotation-quality correction.

Accordingly, the score distributions and ranking results describe this selected diagnostic dataset. In particular, the separation of the $\mathcal { T } _ { 1 }$ and $\mathcal { T } _ { 2 }$ profile families is influenced by the selection criterion and is not an independent estimate for randomly sampled references. Human agreement is measured separately on the retained gaps before adjudication, while final ranking targets incorporate the initial model label only when the two humans disagree. The controlled edits make individual score changes inspectable, and the reported counts specify the evidence available at the reference, gap, and derivedpair levels.

## D CASE STUDY

We examine two types of transcription variation: deletion of a negation at different informational positions, and accumulation of expression-level disfluencies. Tables below report BERTScore (higher is better), WER (lower is better), and SITAR (Ours; lower indicates less distortion). Complete references and hypotheses accompany the scores. Highlighted words identify the relevant negations or altered expressions; deleted words are omitted from the hypotheses.

## D.1 NEGATION DELETION IN DIFFERENT CONTEXTS

Each hypothesis deletes one occurrence of not. N0 changes the stated reason for a flight choice while preserving the booking request; N1 changes the explicit flight choice and creates a conflict with the request to book the earlier flight. N2 reverses lunch availability while leaving meeting attendance intact; N3 reverses meeting attendance itself. These examples use separate references, reproduced in full below.

<table><tr><td>Index</td><td>Change</td><td>BERTScoreF1</td><td>WER</td><td>Ours</td></tr><tr><td>NO (flight)</td><td>Reason reversed</td><td>0.9866</td><td>0.0526</td><td>0.6815</td></tr><tr><td>N1 (flight)</td><td>Core flight choice reversed</td><td>0.9668</td><td>0.0526</td><td>0.7114</td></tr><tr><td>N2 (meeting)</td><td>Secondary availability reversed</td><td>0.9514</td><td>0.0526</td><td>0.8006</td></tr><tr><td>N3 (meeting)</td><td>Meeting attendance reversed</td><td>0.9534</td><td>0.0526</td><td>0.8293</td></tr></table>

WER assigns the same value to all four examples. Ours assigns a higher score to the core-event change within each scenario: N0 < N1 and N2 < N3. BERTScore agrees with this ordering for the flight pair, but reverses it for the meeting pair. Thus, the meeting examples illustrate how a similar surface edit can have different situational consequences that lexical error counts and representation similarity do not necessarily distinguish. The Ours values in this table are normalized using Eq. 11.

N0: Reason reversed.

Ref: I chose the 8:15 flight because I am not worried about the price. Please book the 8:15 refundable ticket.

Hyp: I chose the 8:15 flight because I am worried about the price. Please book the 8:15 refundable ticket.

N1: Core flight choice reversed.

Ref: The 8:15 flight is available. I have decided not to take the 9:40 flight. Please book the earlier one.   
Hyp: The 8:15 flight is available. I have decided to take the 9:40 flight. Please book the earlier one.

N2: Secondary availability reversed.

Ref: The meeting is on Thursday at three. Maya is not available for lunch, but she will attend the meeting.   
Hyp: The meeting is on Thursday at three. Maya is available for lunch, but she will attend the meeting.

N3: Meeting attendance reversed.

Ref: The meeting is on Thursday at three. Maya is available for lunch, but she will not attend the meeting.   
Hyp: The meeting is on Thursday at three. Maya is available for lunch, but she will attend the meeting.

## D.2 ACCUMULATING EXPRESSION-LEVEL REDUNDANCY

F0–F4 share one reference and progressively introduce inflectional variation, fillers, repetitions, a lengthened spelling, and punctuation changes. The show time and the question about reserving seats remain unchanged. The table summarizes the cumulative edits; the complete texts retain the original punctuation and wording.

<table><tr><td>Index</td><td>Cumulative change</td><td>BERTScoreF1</td><td>WER</td><td>Ours</td></tr><tr><td>FO</td><td>Inflection: starts → start</td><td>0.9783</td><td>0.0345</td><td>0.112</td></tr><tr><td>F1</td><td>FO + just repetition + comma deletion</td><td>0.9404</td><td>0.1034</td><td>0.281</td></tr><tr><td>F2</td><td>F1 + filler Um</td><td>0.9428</td><td>0.1724</td><td>0.317</td></tr><tr><td>F3</td><td>F2 + I repetition + reaaally + comma deletion</td><td>0.8976</td><td>0.2414</td><td>0.330</td></tr><tr><td>F4</td><td>F3 + filler uh + the repetition + further punctuation removal</td><td>0.8611</td><td>0.3793</td><td>0.321</td></tr></table>

WER rises from 0.0345 to 0.3793, while BERTScore decreases overall from 0.9783 to 0.8611. Ours initially rises from 0.112 to 0.330 and then changes slightly to 0.321 despite the additional surface edits in F4. This pattern illustrates a limited response to accumulating expression-level variation, with a small final decrease rather than strict monotonicity.

## Shared reference

Well, I think the show starts at eight, and it sounds really good. I just want to know whether we need to reserve seats before we get on board.

## Hyp-F0

Well, I think the show start at eight, and it sounds really good. I just want to know whether we need to reserve seats before we get on board.

## Hyp-F1

Well, I think the show start at eight and it sounds really good. I just just want to know whether we need to reserve seats before we get on board.

## Hyp-F2

Um, well, I think the show start at eight and it sounds really good. I just just want to know whether we need to reserve seats before we get on board.

## Hyp-F3

Um, well I I think the show start at eight and it sounds reaaally good. I just just want to know whether we need to reserve seats before we get on board.

## Hyp-F4

Um uh well I I think the the show start at eight and it sounds reaaally good I just just want to know whether we need to reserve seats before we get on board