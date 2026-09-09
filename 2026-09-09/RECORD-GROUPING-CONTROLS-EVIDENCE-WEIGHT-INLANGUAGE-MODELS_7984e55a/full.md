# RECORD GROUPING CONTROLS EVIDENCE WEIGHT INLANGUAGE MODELS

Zhongxuan Liu Sicheng Zhou Hongzhi Wang<sup>∗</sup> Faculty of Computing, Harbin Institute of Technology lazrix@163.com ylnfq 2021@qq.com wangzh@hit.edu.cn

## ABSTRACT

Retrieved records are presentation units; a supplied partition determines which records enter a language model as one evidential contribution. We characterize the invariant group–content state that removes within-group copies while retaining complementary canonical content, show that equal group counts can encode different evidence states, and derive a sharp content-aware partition-error bound. Given a supplied partition, our pre-generation representation deduplicates and aggregates content within groups and bounds each group’s contribution. Across 104,402 trials and 6 public checkpoints, a central natural-text intervention finds that content-fixed false splits add 10.27–32.66 percentage points and false merges remove 9.13–31.79 points; a matched six-slot control retains the positive direction in all 16 cells. In a new 48-item controlled campaign panel, changing the supplied partition produces measurable, checkpoint-dependent decision shifts across all four models, and the balanced mirror design exposes substantial order interactions. Together, the theory and experiments establish the supplied partition as a controllable pre-generation representation variable and characterize its checkpoint-dependent behavioral effects.

## 1 INTRODUCTION

Retrieval-augmented systems consume packaging records whose multiplicity can diverge from the number of underlying evidence sources. A coordinated campaign can render one informationgenerating process across many pages; raw retrieval serialization then exposes each rendering as a separate model-facing contribution opportunity. This matters for answer engines, search agents, product recommendation, and any system exposed to coordinated content placement because every choice of chunking, syndication handling, and record emission implicitly chooses an evidenceweighting rule.

Prior work has established that frequency, repeated arguments, source labels, metadata, and document diversity can steer language-model decisions (Jin et al., 2024; Wan et al., 2024; Chiang & Lee, 2024; Schuster et al., 2026; Naphade, 2026; Ross et al., 2026). We take this behavioral susceptibility as the starting point. The systems question is: which state preserves the content of each supplied evidence unit while removing within-unit copies? Once an upstream partition specifies which records share a dependence unit, the representation should preserve complementary content while allocating one bounded contribution per group. This guarantee resides in the model-facing state and holds independently of behavioral tendencies, prompt instructions, and trained preferences.

Group count alone leaves an essential choice unresolved. Four distinct evidence elements can be grouped as {{1, 2}, {3, 4}} or {{1, 3}, {2, 4}}: both have two groups of size two, yet combine different facts within each unit. We prove that even bounded group aggregation can distinguish these states. This motivates retaining group membership together with content.

We group records before generation. An upstream process supplies the partition. Model-facing membership labels are opaque: equality encodes membership, while authenticated identity or credibility metadata, when available, live in explicit group signatures. Within each group, the representation removes exact copies, aggregates complementary passages, and assigns one bounded contribution. Figure 1 summarizes the introductory logic from repeated records and shared roots to the supplied partition and its two principal error modes.

![](images/c791d9b794502722627c171a63213e640806497ebe6a76127e47d2703a619ac3.jpg)  
Figure 1: Retrieved records are presentation units and may share claim-relative evidence roots. Raw serialization allocates weight record by record; a supplied partition enables within-group aggregation and one bounded contribution per group. False splits and false merges are the two central partition errors.

Table 1 provides the central causal test. Its full-content contrast adds complementary passages within one supplied group; its split and merge contrasts hold the ordered 160 evidence words fixed while changing the emitted partition. The matched six-slot control fixes every repeated non-content field and tokenizer length. The experiments condition on supplied operational grouping keys and isolate the downstream record partition.

Our contributions are threefold.

• We give a pre-generation grouping representation that combines a supplied partition, withingroup content aggregation, and one bounded contribution per group. Exact copies map to the same model-facing state; explicit signatures carry authenticated identity metadata.

• We characterize invariant and faithful group–content representations, establish equal-count separation, and derive a sharp content-aware error bound. Claim-relative roots and the exact text-only recovery boundary specify the upstream provenance problem.

• We provide a three-contrast causal decomposition and a matched six-slot control that fixes object count, headers, identifiers, separators, and tokenizer length. A new controlled panel extends the intervention from exact or partial copies to four cross-genre renderings of one campaign root and a four-independent-root control, revealing checkpoint-specific reversals.

Figure 2 connects the representation to the theory, controlled interventions, and empirical findings developed in the remainder of the paper.

## 2 RELATED WORK

Context conflicts and source preferences. Language models arbitrate inconsistently between parametric memory, contextual claims, and user-supplied specifications (Longpre et al., 2021; Zhou et al., 2023; 2024). Evidence frequency, rationale framing, metadata, institutional labels, and source authority further alter this balance (Jin et al., 2024; Sun et al., 2026; Wan et al., 2024; Chiang & Lee, 2024; Ge et al., 2025; Li et al., 2025; Liao, 2026). These studies establish that contextual frequency and source cues are behaviorally active. Our question begins at the next layer: how an external representation makes replication invariance a property of the model-facing state with enforcement independent of checkpoint responses.

![](images/6d8d4cf6464253df16468b9a33f7b8d53e4298a2b4e9ef8ec78d23b8daec1c02.jpg)  
Figure 2: From retrieved records to model-facing influence. A supplied claim-relative partition maps retrieved records into a group–content state before a fixed generator. The paper characterizes this representation theoretically, tests it with content-fixed and matched controls, and measures how record boundaries reallocate model-facing influence across checkpoints.

Redundancy, diversity, and grouped evidence. GroupQA shows that paraphrased documents supporting one argument can outweigh distinct support and documents order effects and unfaithful explanations (Naphade, 2026). Whose Facts Win? shows that repetition can reverse credibility preferences across 13 open-weight models and combines teacher–student LoRA distillation with a credibility-aware prompt to induce approximate repetition invariance (Schuster et al., 2026). In a benign fictional-QA setting, Ross et al. (2026) find little correctness gain from duplicates or paraphrases and a large gain from diverse documents. These works define the closest behavioral and learned-mitigation frontier. Song et al. (2026) attach canonical identities to observations and maintain mergeable aggregation states outside long-context generation. We study the representation itself: supplied source-dependence keys induce a group–content state before generation, and content-fixed split/merge interventions with matched six-slot serialization identify the effect of record partition on evidence weight.

Adversarial retrieval and source grouping. SearchGEO and generative-engine optimization supply the adversarial application: coordinated web publication can manipulate the evidence retrieved and endorsed by answer systems (Chen et al., 2026; Aggarwal et al., 2024). Rahadi (2026) proposes provenance-graph estimation, effective independent evidence count and confidence-inflation diagnos tics, and copy-cluster-discounted answer aggregation. Its aggregation uses membership information as well as counts. Our complementary question concerns the state supplied before generation: which invariances it enforces, which within-group content it preserves, and how partition error distorts it. Equal-count separation below concerns scalar count summaries; the controlled interventions establish checkpoint-dependent responses to record boundaries. Adversarial tool outputs expose the same problem for agents (Zhan et al., 2026), while instruction-hierarchy training improves prioritization of privileged over untrusted inputs (Wallace et al., 2024). The characterization uses standard quotient factorization, with the evidence-specific content and error semantics made explicit.

## 3 PARTITIONED EVIDENCE BEFORE GENERATION

## 3.1 CLAIM-RELATIVE EVIDENCE ROOTS

For a target claim c and fixed observed record set I, let $\rho _ { c } ( r ) \in \mathcal { S } _ { c }$ denote the information-generating root from which record r derives its claim-relevant evidence. Each $r \in I$ is a claim-evidence atom with exactly one claim-relative root. A physical document drawing on multiple roots enters I only after atomization into such units or assignment to an explicitly defined composite root. Here “independent” denotes distinct roots under this provenance relation; statistical dependence among their observations remains admissible. The corresponding oracle partition is

$$
P _ { c } ^ { \star } = \big \{ \{ r \in I : \rho _ { c } ( r ) = s \} : s \in \rho _ { c } ( I ) \big \} .\tag{1}
$$

When admissible worlds vary below, $P _ { c } ^ { \star } ( w )$ denotes the partition obtained in world w; elsewhere we suppress w. Its block count is the epistemic multiplicity of the retrieved evidence for c; the number of returned records is its presentation multiplicity. The distinction is claim-relative: two articles can share a measurement for one claim while carrying independently produced evidence for another.

An upstream dependence-discovery process estimates $P _ { c } ^ { \star }$ as $\widehat { P } _ { c }$ . The downstream mechanism accepts either partition as an input. Opaque labels encode membership; authenticated signatures carry provenance and credibility. Under $P _ { c } ^ { \star }$ , another rendering from an existing root may enrich canonical content inside that block while the group-level summand count stays fixed.

For a fixed claim and record count, let $Z _ { c } ( w )$ be the complete visible text in an admissible world w. Universal deterministic text-only recovery exists exactly when $Z _ { c } ( w _ { 0 } ) ~ = ~ Z _ { c } ( w _ { 1 } )$ implies $P _ { c } ^ { \star } ( w _ { 0 } ) = P _ { c } ^ { \star } ( w _ { 1 } )$ (Proposition 6).

Equivalently, the oracle partition is constant on every observational fiber of $Z _ { c }$ . Similarity models provide useful evidence under distributional assumptions; universal exact recovery is available precisely when worlds with identical visible text share the same oracle partition. Appendix B.1 proves the characterization.

## 3.2 INVARIANT AND FAITHFUL GROUP–CONTENT STATE

The representation retains the canonical content of every supplied group while removing repeated occurrences within that group.

Let K be an infinite set of opaque membership labels and let $\mathcal { Z } _ { \mathrm { e v } }$ contain canonical evidence elements. A label’s spelling carries no evidential meaning; equality and inequality encode group membership. An upstream process supplies the partition, and authenticated provenance, when available, can justify it; the model-facing label remains local bookkeeping. A finite grouped configuration is $\begin{array} { r } { \bar { \boldsymbol { x } } = ( ( k _ { i } , z _ { i } ) ) _ { i = 1 } ^ { n } \in \mathsf { C f g } = \bar { \boldsymbol { ( } \boldsymbol { K } \times \mathcal { Z } _ { \mathrm { e v } } ) ^ { * } } } \end{array}$ . Let $K _ { x }$ be its set of occupied labels and, for each $k \in K _ { x }$ let $V _ { x } ( k ) = \{ z _ { i } : k _ { i } = k \}$ . The canonical evidence quotient is the finite multiset

$$
q _ { \mathrm { e v } } ( x ) = \sum _ { k \in K _ { x } } \delta _ { V _ { x } ( k ) } \in \mathsf { Q u o t } = \mathsf { M } _ { \mathrm { f i n } } ( \mathrm { F i n } ^ { + } ( \mathcal { Z } _ { \mathrm { e v } } ) ) .\tag{2}
$$

The outer object is a multiset, so distinct groups with identical content remain distinct; each inner object is a set, so another canonical copy inside one group disappears. Declare $x \sim y$ when $q _ { \mathrm { e v } } ( x ) = q _ { \mathrm { e v } } ( y )$ . This relation removes record order, bijective renaming of opaque occupied labels, and within-group canonical copies while preserving complementary elements and every nontrivial split or merge as distinct quotient states. Appendix B gives the signature-augmented version that carries authenticated identity or credibility as an evidence-bearing coordinate.

The representation has three nuisance invariances: record permutation, bijective renaming of occupied opaque labels, and within-group insertion or deletion of an exact canonical duplicate while one occurrence remains. Faithfulness asks that all distinctions between the resulting group–content states remain recoverable.

Theorem 1 (Invariant and faithful evidence representations). For any set Y and representation $\Psi : { \mathsf { C f g } } \to Y$ , the following are equivalent: (i) Ψ has the three nuisance invariances above; (ii) $\Psi ( x ) = \Psi ( y )$ whenever $x \sim y ;$ and (iii) there is a unique ${ \bar { \Psi } } : \mathsf { Q u o t } \to Y$ such that

$$
\Psi = \bar { \Psi } \circ q _ { \mathrm { e v } } .\tag{3}
$$

Among these invariant representations, preserving every relation-invariant observable is equivalent to injectivity of Ψ<sup>¯</sup> , and to recoverability of q<sub>ev</sub> from Ψ. Thus faithful invariant states are precisely injective recodings of $q _ { \mathrm { e v } }$

The three operations generate exactly the fibers of $q _ { \mathrm { e v } } \mathbf { \cdot }$ match groups with equal content sets, rename their labels, delete duplicate occurrences, and permute records. Standard quotient factorization then gives the characterization; choosing $q _ { \mathrm { e v } }$ itself as the observable gives recoverability. The minimality is relative to preserving all nuisance-invariant information under a fixed canonicalizer. A specified downstream task can use a coarser state. Appendix B supplies the full proofs and the downstream-map formulation.

The group-additive construction is one factorized realization. Let U be an ambient record universe, let $z _ { \mathrm { e v } } : U  \mathcal { Z } _ { \mathrm { e v } }$ be one fixed canonicalizer, let $( \mathcal { X } , \Vert \cdot \Vert )$ be a real normed vector space, and let a map finite subsets of $\mathcal { Z } _ { \mathrm { e v } }$ into X. For a finite $I \subseteq U$ and a partition P of I, define

$$
E _ { I } ( P ) = \sum _ { G \in P } a ( \{ z _ { \mathrm { e v } } ( i ) : i \in G \} ) .\tag{4}
$$

Writing $\begin{array} { r } { q _ { P } = \sum _ { G \in P } \delta _ { \{ z _ { \mathrm { e v } } ( i ) : i \in G \} } } \end{array}$ and $\begin{array} { r } { \Phi _ { a } ( m ) = \sum _ { S } m ( S ) a ( S ) } \end{array}$ gives $E _ { I } ( P ) = \Phi _ { a } ( q _ { P } )$ . Hence $E _ { I }$ always factors through the quotient; quotient faithfulness holds exactly when the aggregator separates the reachable quotient states, as characterized in Appendix B.

Proposition 2 (Equal-count separation). Let $z _ { 1 } , z _ { 2 } , z _ { 3 } , z _ { 4 }$ be distinct canonical elements attached to fourfixed records, and set

$$
P _ { A } = \{ \{ 1 , 2 \} , \{ 3 , 4 \} \} , \qquad P _ { B } = \{ \{ 1 , 3 \} , \{ 2 , 4 \} \} .
$$

The record content, group count, and block-size multiset agree, but $q _ { P _ { A } } \neq q _ { P _ { B } }$ . For every $B > 0$ there is afixed scalar aggregator with $| a ( S ) | \leq B$ for which $E _ { I } ( P _ { A } ) = 2 B$ and $E _ { I } ( P _ { B } ) \stackrel { \cdot } { = } - 2 B$

For the witness, assign +B to $\{ z _ { 1 } , z _ { 2 } \}$ and $\{ z _ { 3 } , z _ { 4 } \} , - B \mathrm { t o } \{ z _ { 1 } , z _ { 3 } \}$ and $\{ z _ { 2 } , z _ { 4 } \}$ , and zero elsewhere. Thus the same global content and count can support distinct bounded evidence states. Count records how many units exist; the group–content state also records which facts they combine. This is a representation-level separation; the model experiments below measure content-fixed split/merge responses.

Proposition 3 (Exact-replication invariance). If a new record $i ^ { \prime } \in U \setminus I$ is added to the block containing i and $z _ { \mathrm { e v } } ( i ^ { \prime } ) = z _ { \mathrm { e v } } ( i )$ , then the evidence state in Equation (4) is unchanged.

The construction uses a supplied grouping key, a fixed canonicalizer, and an extensional set-valued group input; its proof and the corresponding grouping-error bound are in Appendix B. A canonicalizer determines which copies share an element, while complementary passages remain separate elements within one group. More generally, let $J \subseteq U \setminus I$ be a finite set of new records that joins an existing block $G \in { \bar { P } }$ , and set $P ^ { \prime } = ( P \setminus \{ G \} ) \cup \{ G \cup J \}$ . Their entire effect is

$$
E _ { I \cup J } ( P ^ { \prime } ) - E _ { I } ( P ) = a ( A _ { G } \cup A _ { J } ) - a ( A _ { G } ) ,\tag{5}
$$

where $A _ { G } = \{ z _ { \mathrm { e v } } ( i ) : i \in G \}$ and $A _ { J } = \{ z _ { \mathrm { e v } } ( j ) : j \in J \}$ . When $P = P _ { c } ^ { \star }$ and every record in J shares the claim-relative root of G, this is same-root proliferation. The number of outer contributions stays fixed; the state changes only through genuinely new canonical content inside the root.

## 3.3 FROM PARTITION ERROR TO DECISION STABILITY

For two partitions $P , Q$ of the same records, define the content-aware discrepancy

$$
\Delta _ { q } ( P , Q ) = \| q _ { P } - q _ { Q } \| _ { 1 } = \sum _ { A } | q _ { P } ( A ) - q _ { Q } ( A ) | .
$$

It cancels matching group–content sets even when their record identities differ. With a fixed canonicalizer and $\| a ( A ) \| \leq B$ , Proposition 12 gives

$$
\begin{array} { r } { \| E _ { I } ( P ) - E _ { I } ( Q ) \| \le B \Delta _ { q } ( P , Q ) \le B H ( P , Q ) , } \end{array}\tag{6}
$$

where H counts the blocks in changed overlap components (Appendix B). The first bound is exact in the worst case over bounded scalar aggregators for each fixed pair $P , Q$ . Equal-count reassignment can attain 4B, as the preceding witness shows. For an L-Lipschitz decision margin m, its sign is stable whenever $| m ( E _ { I } ( P ) ) | > L B \Delta _ { q } ( P , Q )$ . Taking $P = P _ { c } ^ { \star }$ and $Q = \widehat { P } _ { c }$ connects upstream partition error to representation distortion and then to decision stability. The numerical constants belong to a specified representation and downstream map; the language-model experiments measure behavioral effects separately.

The natural-text experiment uses externally supplied grouping metadata. HUMAN keys denote crowd-annotated evidence units; WEB copies share canonical URLs, and its Four-Unit records satisfy frozen domain, URL, and text-hash separation criteria. These are operational units, while authenticated identity and credibility remain explicit signature coordinates.

## 4 EXPERIMENTAL PROGRAM

## 4.1 PANELS, SCORING, AND STATISTICS

The natural audit analyzes 101 PERSPECTRUM claims in 49 dependence components (Chen et al., 2019) and 138 ConflictingQA questions in 135 components (Wan et al., 2024; Chiang & Lee, 2024) as separate domains. The grouping panel uses 66 HUMAN claims passing a frozen document filter and all 138 WEB questions; generation uses 40 components per domain.

Candidate-choice audits score two complete, length-matched assistant answers with native termination sequences by total conditional log likelihood. Exact ties score zero in the primary analysis and one half in sensitivity analysis. Controlled generation separately scores the parsed leading Yes/No over all outputs.

For trial $i ,$ let $x _ { i }$ be the rendered prompt, let $a _ { i }$ and $b _ { i }$ be the attack-side and opposite complete candidates, and let $u _ { i , c , 1 : m _ { i c } }$ be candidate c’s scoring suffix, including the checkpoint’s native termination sequence. We compute

$$
\begin{array} { l } { \displaystyle \ell _ { i } ( c ) = \sum _ { t = 1 } ^ { m _ { i c } } \log p _ { \theta } ( u _ { i , c , t } \mid x _ { i } , u _ { i , c , < t } ) , } \\ { \displaystyle p _ { \mathrm { a t t a c k } , i } = \frac { \exp \ell _ { i } ( a _ { i } ) } { \exp \ell _ { i } ( a _ { i } ) + \exp \ell _ { i } ( b _ { i } ) } = \frac { 1 } { 1 + \exp ( \ell _ { i } ( b _ { i } ) - \ell _ { i } ( a _ { i } ) ) } . } \end{array}\tag{7}
$$

Thus $p _ { \mathrm { a t t a c k } }$ is the likelihood share over the two frozen candidates.

We average mirrored assignments and orders within item before each contrast. Fixed-seed 10,000- replicate bootstraps resample items or natural-text dependence components; estimates remain checkpoint- and domain-specific. Appendix C.4–C.5 gives full specifications.

## 4.2 CENTRAL GROUPING INTERVENTION AND BREADTH STUDIES

For the one-group partial-copy panel, four non-overlapping 40-word windows come from one frozen document. For the four-group panel, four windows come from four operationally distinct supplied units. We hold the supplied keys fixed, inject false splits or false merges downstream, aggregate content within resulting groups, and emit one evidence record per estimated group. The main comparisons hold the ordered 160 attack words fixed. Across Qwen3-8B and Qwen3-4B (Yang et al., 2025), Phi-4-mini (Microsoft, 2025), and Mistral-7B (Jiang et al., 2023), this experiment contributes 39,168 trials. A further 13,056-trial control uses six JSON slots in both arms, retains R1–R6 and every header and separator, matches final character, byte, and tokenizer lengths, and varies whether the fixed word stream occupies one or four nonempty content-bearing slots.

Controlled campaign and independent-root panel. We construct 48 fictional product pairs. For each attack side, four 40-word cross-genre records deterministically restate one frozen three-fact campaign brief and share one oracle root; a second family contains four 40-word records tied to independently specified laboratory, panel, endurance, and service-data roots. Every visible product claim is bound to an enumerated atomic fact before inference. Hidden root, author, and syntheticdomain fields never enter the prompt. In both families, the Phase 11 six-slot renderer compares [160, 0, 0, 0] with [40, 40, 40, 40] while fixing the ordered word stream, six objects, R1–R6 fields, side sequence, final characters, UTF-8 bytes, and model-specific token length. Two attack sides and two order mirrors yield 3,072 choice trials over the same four checkpoints.

Table 1: Central causal decomposition of content retention and record partition, in percentage points with 95% dependence-component bootstrap intervals. Full-content gain compares the 160-word one-group record with its 40-word endpoint. The split and merge contrasts hold the ordered 160 attack words fixed; positive values denote influence gained by splitting one supplied group or influence lost by merging four supplied groups. The columns identify complementary-content retention, false-split inflation, and false-merge suppression. Cells report claims/dependence components: HUMAN 66/32 and WEB 138/135.
<table><tr><td>Checkpoint Domain</td><td>Full-content gain</td><td>One supplied unit: split</td><td></td><td>Four supplied units: merge loss</td></tr><tr><td></td><td></td><td></td><td></td><td>Qwen3-8B HUMAN (66/32) +13.98[+8.65, +19.82] +26.38[+20.51, +32.36] +25.17[+18.26, +33.11]</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Qwen3-8B WEB (138/135) +14.30[+11.24, +17.51] +25.23[+21.43, +29.19] +24.65[+20.55, +28.77] Qwen3-4BHUMAN (66/32) +29.82[+25.39, +34.45] +20.61[+15.38, +25.68] +31.79[+27.05, +38.06]</td><td></td><td></td><td></td></tr><tr><td></td><td>Qwen3-4B WEB (138/135) +15.35[+11.80, +18.82] +24.43[+20.99, +28.01] +31.56[+27.46, +35.81]</td><td></td><td></td><td></td></tr><tr><td>Phi-4-mini HUMAN (66/32)</td><td>+4.96[+2.49, +8.14]+14.46[+9.81, +19.94]</td><td></td><td></td><td>+9.13[+6.02, +13.08]</td></tr><tr><td>Phi-4-mini WEB (138/135)</td><td>+2.91[+1.07, +4.80]+10.27[+7.64, +13.23]</td><td></td><td></td><td>+10.10[+7.65, +12.76]</td></tr><tr><td></td><td>Mistral-7B HUMAN (66/32) +18.93[+14.27, +24.16] +32.66[+28.59, +37.74] +31.15[+27.18, +35.64]</td><td></td><td></td><td></td></tr><tr><td></td><td>Mistral-7B WEB (138/135) +18.47[+14.99, +22.08] +25.21[+21.99, +28.49] +26.69[+23.00, +30.51]</td><td></td><td></td><td></td></tr></table>

Supporting audits cover authority cues, explicit relation/count stages, repetition dose, natural-text copies, added Qwen3-14B-AWQ and Phi-4 checkpoints (Yang et al., 2025; Abdin et al., 2024), quantization, and controlled generation. Together with the central grouping, matched-serialization, and provenance panels, the program contains 104,402 trials across 6 unique checkpoints. Appendix E gives complete designs, counts, and lineage.

## 5 RESULTS

## 5.1 GROUPING SEPARATES CONTENT RETENTION FROM PARTITION EFFECTS

Table 1 is the paper’s central causal-identification test. It holds the supplied operational keys fixed, intervenes on downstream aggregation and rendering, and decomposes evidence weighting into three interpretable contrasts.

First, the full-content arm exact-deduplicates and concatenates four non-overlapping windows from one supplied unit into a single 160-word group record. Relative to the one-window endpoint, the additional complementary content raises attack-side probability by 2.91–29.82 points, with all eight confidence intervals above zero. One-group normalization therefore combines complementary within-unit content retention with multiplicity control.

Second, falsely splitting that same supplied unit into four records raises attack-side probability by 10.27–32.66 points. Third, falsely merging four supplied units into one record suppresses their influence by 9.13–31.79 points. Every interval is positive. In both contrasts, the evidence words and their order are identical across arms; after the supplied keys are fixed, the intervention changes the record partition and its serialization. Together, the three contrasts disentangle complementary-content retention, multiplicity inflation from false splits, and influence suppression from false merges.

The original renderer fixes the instruction, query, claim, anchor content, chat adapter, candidates, and scoring suffix. Its four-record implementation introduces three additional JSON objects with headers, separators, and ordinal identifiers, corresponding to 43–63 additional input tokens across the four tokenizers and an evidence-token difference bounded by five. The matched six-slot control fixes the full serialization skeleton: both arms contain the same six objects, R1–R6 identifiers, side labels, headers, separators, template text, ordered evidence word stream, final characters and bytes, and tokenizer input length; the content spans one or four attack-side slots. Its effects are positive in 16/16 cells and 12 confidence intervals lie entirely above zero, ranging from +0.63 to +13.33 points. The matched design identifies content-bearing record placement under equal object, header, and token counts. Outcome-blind tail whitespace maintains exact tokenizer length. Appendix C.5 gives the full accounting and model-specific estimates.

Table 2: Controlled campaign and independent-root interventions. Values are percentage-point changes in $p _ { \mathrm { a t t a c k } }$ with pointwise 95% dependence-component bootstrap intervals. Same-root split is four content-bearing records minus one grouped record; independent-root merge loss is four records minus one falsely merged record. Positive values are the preregistered directions; checkpoints are reported without pooling.
<table><tr><td>Model</td><td>Same-root split</td><td>Independent-root merge loss</td></tr><tr><td>Qwen3-8B</td><td> $+ 1 8 . 4 6 [ + 1 5 . 5 5 , + 2 1 . 4 1 ]$ </td><td> $- 4 . 2 5 [ - 6 . 8 6 , - 1 . 8 7 ]$ </td></tr><tr><td>Qwen3-4B</td><td> $+ 9 . 6 \dot { 6 } [ + 6 . 5 9 , + 1 2 . 5 4 \ ]$ </td><td> $+ 5 . 3 9 \bar { [ + 3 . 1 2 , + 7 . 9 4 \bar { ] } }$ </td></tr><tr><td>Phi-4-mini</td><td> $+ 2 7 . 4 4 [ \dot { + } 2 4 . 6 4 , + 3 0 . 2 5 ]$ </td><td> $+ 1 5 . 7 3 [ + \dot { 1 } 2 . 8 7 , + 1 8 . 7 0 \dot { ] }$ </td></tr><tr><td>Mistral-7B</td><td> $- 4 . { \dot { 4 } } 4 [ - 7 . 3 5 , - 1 . 6 4 ]$ </td><td> $+ 3 . { \dot { 0 } } 2 [ + 1 . 6 3 , + 4 . 6 1 ]$ </td></tr></table>

A post-hoc multiplicity sensitivity applies Bonferroni familywise 95% bootstrap intervals to the 24 Table 1 cells. Positive lower bounds remain for all 16 content-fixed split/merge effects and seven of eight content-retention effects; only Phi-4-mini/WEB content retention crosses zero by 0.02 points. The matched six-slot family retains 11 of 16 positive lower bounds after the same correction (Appendix D.4).

## 5.2 CONTROLLED GEO RENDERINGS EXPOSE CHECKPOINT-DEPENDENT PARTITION EFFECTS

Table 2 extends the content-fixed six-slot intervention to cross-genre records from one campaign root and to four independently specified roots. The same-root split effect is positive for Qwen3-8B, Qwen3-4B, and Phi-4-mini but negative for Mistral-7B. Thus coordinated proliferation can gain model-facing influence without exact copying, while the Mistral reversal establishes a checkpointdependent behavioral sign.

The independent-root merge loss is positive for Qwen3-4B, Phi-4-mini, and Mistral-7B but reverses for Qwen3-8B. The balanced average is a boundary-sensitivity stress test: mirror decomposition shows large order interactions (for Qwen3-8B, the independent-root contrast is +41.43 points in original order and −49.92 in reverse order). Record-boundary placement is therefore behaviorally active, with its sign determined jointly by checkpoint, content family, and presentation. The representation guarantee fixes one bounded group-level contribution for each supplied group; checkpoint-specific behavioral directions remain empirical.

Appendix Table 6 reports a paired post-hoc order decomposition of these same predictions for every checkpoint and root family. It estimates the order interaction within each item before resampling, preserving the two attack-side mirrors. The balanced effects in Table 2 are recovered exactly by averaging the two orders.

## 5.3 RECORD COPIES REWEIGHT CANDIDATE AND GENERATED DECISIONS

Across six checkpoints, four copies of one supplied unit shift attack-side candidate decisions by 9.24–51.98 points; every model–domain interval excludes replication invariance in the observed direction (Figure 5 and Appendix Tables 12 and 7).

The original HUMAN and WEB ranges are 14.11–49.01 and 9.24–37.14 points. Added 14B checkpoints reach 30.62–51.98 points. In controlled answer-plus-one-sentence generation, raw copying moves the leading answer by 15.62–20.62 points across three checkpoints, with positive lower bounds and 2,880/2,880 valid outputs. Appendix C.5 reports prompt-rule heterogeneity, threshold sensitivity, and 956 byte-identical grouping-recovery checks per compiled grid.

## 6 DISCUSSION

Partition is an evidence-accounting decision. The same claim can reach a generator as one record, several copied records, several complementary passages, or several records carrying one supplied key. Table 1 shows that this packaging actively allocates evidential influence: complementary content survives one-group aggregation, false splits inflate influence, and false merges suppress it. The matched six-slot control preserves the predicted direction in every cell after equalizing serialization counts and tokenizer length. Table 2 shows that cross-genre boundary placement remains active while its sign can reverse across checkpoints. A RAG pipeline therefore chooses an evidence-weighting rule whenever it chooses how to chunk, duplicate, group, and serialize retrieval results. Evidence independence tracks claim-relevant information-generating roots; surface diversity and URL count describe presentation, and fixed-model sensitivity to serialization is measured separately.

Dependence-aware weighting belongs before generation. A source-aware system separates four responsibilities: discover an upstream dependence partition, group records, aggregate complementary within-group content, and normalize group-level weight. The supplied blocks can encode exact record identity, article lineage, or an application-defined claim-relative evidence root. Model-facing membership labels remain opaque, while explicit group signatures carry authenticated identity and credibility. On the controlled panel, exact hash and MinHash miss every cross-genre same-root grouping, whereas a fixed-threshold sentence-embedding baseline (Reimers & Gurevych, 2019) separates both provenance structures; Appendix Table 4 reports the full partition audit. The deliberately structured templates provide a controlled-panel sanity check for semantic grouping. Proposition 6 places universal exact recovery at observational-fiber constancy. Once a partition is supplied, aggregation and per-group normalization enforce the chosen accounting rule independently of a checkpoint’s behavioral sign.

What information must survive grouping. Theorem 1 distinguishes invariance from faithful content retention: an invariant summary preserves the full group–content state exactly when its recoding is injective. Proposition 2 exhibits the information discarded by a scalar group count even with identical records and equal block sizes. The sharper discrepancy $\Delta _ { q }$ makes the same distinction in error propagation, charging the unmatched group–content mass. Together these results specify what the supplied partition controls before generation. The observed sign reversals motivate checkpoint-specific measurement of the response to that state.

## 7 CONCLUSION

A supplied partition specifies which records contribute together before generation. Its group–content state preserves distinctions beyond record count and group count, while removing within-group canonical copies. Across exact copies, complementary passages, content-fixed split/merge interventions, and controlled campaign renderings, the emitted partition changes model-facing influence. The campaign panel exhibits checkpoint-dependent sign reversals. The formal guarantee belongs to the external representation: each supplied group receives one bounded group-level contribution while complementary canonical content remains available within that group. Under oracle grouping, each group corresponds to one claim-relative root; text, metadata, and authenticated provenance signals inform the estimated partition, explicit signatures preserve evidence-bearing identity, and Equation (6) converts partition errors into a representation bound.

## AI USE STATEMENT

None.

ETHICS STATEMENT

The study analyzes public, previously released datasets, including prior crowd annotations in the HUMAN domain, and introduces zero new human-subject interactions. All analyzed data are public. The main misuse risk is content manipulation informed by the controlled repetition protocol. The presentation centers defensive evaluation and separates oracle source metadata from editable text labels.

## REPRODUCIBILITY STATEMENT

The anonymous artifact combines the verified stored-output layer with the controlled GEO extension and recomputes packaged Phase 6 and Phase 8–12 results. Its Phase 12 layer contains all 3,072 choices, predictions, grouping assignments, five paper outputs, and the independent audit. A clean extraction verifies 284 manifest items and recomputes legacy and Phase 12 statistics with zero failures; the maximum Phase 12 numerical difference is $\dot { 5 } . 3 3 \times \dot { 1 } 0 ^ { - 1 5 }$ . Phase 5 and Phase 7 execution counts remain in their audited run records. Fresh forward execution uses the named public checkpoints and licensed datasets.

The accompanying analysis scripts reproduce the additional post-hoc order decomposition from these stored Phase 12 predictions and check the preservation of the original result files.

## REFERENCES

Marah Abdin et al. Phi-4 technical report. arXiv preprint arXiv:2412.08905, 2024. URL https: //arxiv.org/abs/2412.08905.

Pranjal Aggarwal, Vishvak Murahari, Tanmay Rajpurohit, Ashwin Kalyan, Karthik Narasimhan, and Ameet Deshpande. GEO: Generative engine optimization. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pp. 5–16, 2024. doi: 10.1145/3637528.3671900. URL https://doi.org/10.1145/3637528.3671900.

Sihao Chen, Daniel Khashabi, Wenpeng Yin, Chris Callison-Burch, and Dan Roth. Seeing things from a different angle: Discovering diverse perspectives about claims. In Proceedings of the 2019 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 542–557, 2019. doi: 10.18653/v1/N19-1053. URL https://aclanthology.org/N19-1053/.

Yimeng Chen, Zhe Ren, Firas Laakom, Yu Li, Dandan Guo, and Jurgen Schmidhuber. How¨ much can we trust LLM search agents? measuring endorsement vulnerability to web content manipulation. arXiv preprint arXiv:2606.16821, 2026. doi: 10.48550/arXiv.2606.16821. URL https://arxiv.org/abs/2606.16821.

Cheng-Han Chiang and Hung-yi Lee. Do metadata and appearance of the retrieved webpages affect LLM’s reasoning in retrieval-augmented generation? In Proceedings of the 7th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP, pp. 389–406, 2024. doi: 10.18653/v1/2024.blackboxnlp-1.24. URL https://aclanthology.org/2024. blackboxnlp-1.24/.

Ziyu Ge, Yuhao Wu, Daniel Wai Kit Chin, Roy Ka-Wei Lee, and Rui Cao. Resolving conflicting evidence in automated fact-checking: A study on retrieval-augmented LLMs. arXiv preprint arXiv:2505.17762, 2025. URL https://arxiv.org/abs/2505.17762.

Albert Q. Jiang et al. Mistral 7B. arXiv preprint arXiv:2310.06825, 2023. URL https://arxiv. org/abs/2310.06825.

Zhuoran Jin, Pengfei Cao, Yubo Chen, Kang Liu, Xiaojian Jiang, Jiexin Xu, Li Qiuxia, and Jun Zhao. Tug-of-war between knowledge: Exploring and resolving knowledge conflicts in retrievalaugmented language models. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pp. 16867–16878, 2024. URL https://aclanthology.org/2024.lrec-main.1466/.

Yuxuan Li, Xinwei Guo, Jiashi Gao, Guanhua Chen, Xiangyu Zhao, Jiaxin Zhang, Quanying Liu, Haiyan Wu, Xin Yao, and Xuetao Wei. LLMs trust humans more, that’s a problem! unveiling and mitigating the authority bias in retrieval-augmented generation. In Proceedings ofthe 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 28844–28858, 2025. doi: 10.18653/v1/2025.acl-long.1400. URL https://aclanthology. org/2025.acl-long.1400/.

Junchi Liao. Auditing provenance sensitivity in LLM agent action selection. arXiv preprint arXiv:2607.20827, 2026. URL https://arxiv.org/abs/2607.20827.

Shayne Longpre, Kartik Perisetla, Anthony Chen, Nikhil Ramesh, Chris DuBois, and Sameer Singh. Entity-based knowledge conflicts in question answering. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 7052–7063, 2021. doi: 10.18653/v1/ 2021.emnlp-main.565. URL https://aclanthology.org/2021.emnlp-main.565/.

Microsoft. Phi-4-Mini technical report: Compact yet powerful multimodal language models via mixture-of-LoRAs. arXiv preprint arXiv:2503.01743, 2025. URL https://arxiv.org/ abs/2503.01743.

Atharv Naphade. Rational synthesizers or heuristic followers? analyzing LLMs in RAG-based question-answering. In Findings of the Association for Computational Linguistics: ACL 2026, pp. 40293–40311, 2026. doi: 10.18653/v1/2026.findings-acl.2003. URL https: //aclanthology.org/2026.findings-acl.2003/.

Irwan Rahadi. Counting copies as evidence: Confidence inflation from dependent evidence in retrieval-augmented generation (RAG), August 2026. URL https://doi.org/10.5281/ zenodo.21923648. Position paper and preprint.

Nils Reimers and Iryna Gurevych. Sentence-BERT: Sentence embeddings using Siamese BERTnetworks. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, pp. 3982–3992. Association for Computational Linguistics, 2019. doi: 10.18653/v1/D19-1410. URL https://aclanthology.org/D19-1410/.

Jonathan J. Ross, Bevan Koopman, Anton van der Vegt, and Guido Zuccon. How retriever redundancy and diversity impact RAG effectiveness. arXiv preprint arXiv:2608.13956, 2026. URL https: //arxiv.org/abs/2608.13956.

Jakob Schuster, Vagrant Gautam, and Katja Markert. Whose facts win? LLM source preferences under knowledge conflicts. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 29430–29459, 2026. doi: 10.18653/v1/ 2026.acl-long.1357. URL https://aclanthology.org/2026.acl-long.1357/.

Dachuan Song, Junyu Yin, Zechen Hu, and Xuan Wang. Mergeable model-side aggregation states for long-context language models. arXiv preprint arXiv:2607.26448, 2026. URL https://arxiv. org/abs/2607.26448.

Kaiser Sun, Fan Bai, and Mark Dredze. Task matters: Knowledge requirements shape LLM responses to context–memory conflict. In Findings of the Association for Computational Linguistics: ACL 2026, pp. 4154–4176, 2026. doi: 10.18653/v1/2026.findings-acl.202. URL https://aclanthology.org/2026.findings-acl.202/.

Eric Wallace, Kai Xiao, Reimar Leike, Lilian Weng, Johannes Heidecke, and Alex Beutel. The instruction hierarchy: Training LLMs to prioritize privileged instructions. arXiv preprint arXiv:2404.13208, 2024. URL https://arxiv.org/abs/2404.13208.

Alexander Wan, Eric Wallace, and Dan Klein. What evidence do language models find convincing? In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 7468–7484, 2024. doi: 10.18653/v1/2024.acl-long.403. URL https://aclanthology.org/2024.acl-long.403/.

An Yang et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. URL https: //arxiv.org/abs/2505.09388.

Zhonghao Zhan, Huichi Zhou, Zhenhao Li, Peiyuan Jing, Krinos Li, and Hamed Haddadi. How adversarial environments mislead agentic AI? In Findings ofthe Associationfor Computational Linguistics: ACL 2026, pp. 10264–10280, 2026. doi: 10.18653/v1/2026.findings-acl.499. URL https://aclanthology.org/2026.findings-acl.499/.

Sizhe Zhou, Sha Li, Yu Meng, Yizhu Jiao, Heng Ji, and Jiawei Han. Establishing knowledge preference in language models. arXiv preprint arXiv:2407.13048, 2024. URL https://arxiv. org/abs/2407.13048.

Wenxuan Zhou, Sheng Zhang, Hoifung Poon, and Muhao Chen. Context-faithful prompting for large language models. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2023, pp. 14544–14556, 2023. doi: 10.18653/v1/2023.findings-emnlp.968. URL https:// aclanthology.org/2023.findings-emnlp.968/.

## A SUPPORTING TWO-STAGE AUDIT THEORY

## A.1 FINITE-STATE TWO-STAGE IDENTITY

For item i, let balanced direction $S _ { i } \in \{ - 1 , + 1 \}$ determine which abstract value wins the suppliedgroup count. Let $Z _ { i } ^ { \mathrm { p a r s e } } \in { \mathcal { Z } }$ be an explicit parse-stage output and $\mu _ { i } ( z )$ the mean response of a separate use-stage call when a program supplies state z. Define

$$
\pi _ { i } ( z ) = { \frac { 1 } { 2 } } \left[ \operatorname* { P r } ( Z _ { i } ^ { \mathrm { p a r s e } } = z \mid S _ { i } = + 1 , i ) - \operatorname* { P r } ( Z _ { i } ^ { \mathrm { p a r s e } } = z \mid S _ { i } = - 1 , i ) \right] .
$$

Theorem 4 (Finite-state two-stage identity). If the executed two-stage program satisfies $\mathbb { E } [ Y _ { i } ^ { \mathrm { c a s } }$ $Z _ { i } ^ { \mathrm { p a r s e } } = z , \dot { S } _ { i } = s , i ] = \mu _ { i } ( z )$ , then its half contrast is

$$
\begin{array} { l } { { r _ { i } ^ { \mathrm { { c a s } } } = \displaystyle \frac { \mathbb { E } [ Y _ { i } ^ { \mathrm { { c a s } } } \mid S _ { i } = + 1 , i ] - \mathbb { E } [ Y _ { i } ^ { \mathrm { { c a s } } } \mid S _ { i } = - 1 , i ] } { 2 } } } \\ { { \displaystyle \ } } \\ { { \displaystyle \ = \sum _ { z \in \mathcal { Z } } \mu _ { i } ( z ) \pi _ { i } ( z ) } . } \end{array}\tag{8}
$$

ProofofTheorem 4. For each $s \in \{ - 1 , + 1 \}$ , the law of total expectation and the explicit two-stage condition give

$$
\mathbb { E } [ Y _ { i } ^ { \mathrm { c a s } } \mid S _ { i } = s , i ] = \sum _ { z \in \mathcal { Z } } \mu _ { i } ( z ) \operatorname* { P r } ( Z _ { i } ^ { \mathrm { p a r s e } } = z \mid S _ { i } = s , i ) .
$$

Subtracting the two directions and dividing by two gives Equation (8). The derivation accommodates dependent stages and direction-asymmetric errors. □

For a binary parse state, any use response has the form $\mu _ { i } ( z ) = a _ { i } + u _ { i } z$ . If

$$
p _ { i } = { \frac { \mathbb { E } [ Z _ { i } ^ { \mathrm { p a r s e } } \mid S _ { i } = + 1 , i ] - \mathbb { E } [ Z _ { i } ^ { \mathrm { p a r s e } } \mid S _ { i } = - 1 , i ] } { 2 } } ,
$$

then $r _ { i } ^ { \mathrm { c a s } } = p _ { i } u _ { i }$ . Across items, $\mathbb { E } [ p _ { i } u _ { i } ] = \mathbb { E } [ p _ { i } ] \mathbb { E } [ u _ { i } ] + \operatorname { C o v } ( p _ { i } , u _ { i } )$ . Exact reconstruction from pooled parse and use means therefore requires control of the item-level covariance.

## A.2 TWO-WORLD TRANSPORT LOWER BOUND

Proposition 5 (Prompt-transport lower bound). For responses bounded in $[ - L , L ]$ under arbitrary relationships between auxiliary and end-to-end prompts, two end-to-end worlds can have identical auxiliary distributions while their end-to-end effects are opposite. Every predictor determined by the auxiliary distribution then has worst-case absolute error at least L.

ProofofProposition 5. Fix any auxiliary-task distribution and any predictor $\widehat { R }$ measurable with respect to it. Construct two compatible end-to-end worlds with the same auxiliary observations:

$$
\mathcal { W } _ { + } : Y ^ { \mathrm { e 2 e } } = L S , \qquad \mathcal { W } _ { - } : Y ^ { \mathrm { e 2 e } } = - L S .
$$

Their end-to-end effects are $+ L$ and $- L .$ . The predictor has the same value in both worlds, while

$$
2 L \leq | \widehat { R } - L | + | \widehat { R } + L | .
$$

At least one error is at least $L .$

## A.3 COMMON-STATE THREE-DEFECT BOUND

Let $Z ^ { \mathrm { e 2 e } }$ be a reference end-to-end state on the same finite state space, let $h _ { i } ( z )$ be its reference downstream response, and define

$$
R _ { \mathrm { e 2 e } } = \mathbb { E } [ S _ { i } Y _ { i } ^ { \mathrm { e 2 e } } ] , \qquad R _ { \mathrm { c a s } } = \mathbb { E } [ S _ { i } \mu _ { i } ( Z _ { i } ^ { \mathrm { p a r s e } } ) ] .
$$

Introduce

$$
\begin{array} { r l r } & { } & { \eta _ { \mathrm { d i r } } = \left| \mathbb { E } [ S _ { i } \{ Y _ { i } ^ { \mathrm { e 2 e } } - h _ { i } ( Z _ { i } ^ { \mathrm { e 2 e } } ) \} ] \right| , } \\ & { } & { \eta _ { \mathrm { u s e } } = \mathbb { E } \left[ | h _ { i } ( Z _ { i } ^ { \mathrm { e 2 e } } ) - \mu _ { i } ( Z _ { i } ^ { \mathrm { e 2 e } } ) | \right] , } \\ & { } & { \eta _ { \mathrm { p a r s e } } = \mathbb { E } \left[ | \mu _ { i } ( Z _ { i } ^ { \mathrm { e 2 e } } ) - \mu _ { i } ( Z _ { i } ^ { \mathrm { p a r s e } } ) | \right] . } \end{array}
$$

Adding and subtracting the two intermediate responses and applying the triangle inequality gives

$$
| R _ { \mathrm { e 2 e } } - R _ { \mathrm { c a s } } | \leq \eta _ { \mathrm { d i r } } + \eta _ { \mathrm { u s e } } + \eta _ { \mathrm { p a r s e } } .
$$

On a common reference state space, the transport gap decomposes into the three displayed defects. If direct mismatch is zero, use mismatch is uniformly at most $\epsilon _ { u } .$ , and the parse states disagree with probability at most $\epsilon _ { p } ,$ with responses in $[ - L , L ]$ , then

$$
| R _ { \mathrm { e 2 e } } - R _ { \mathrm { c a s } } | \leq \epsilon _ { u } + 2 L \epsilon _ { p } .
$$

## B CANONICAL EVIDENCE QUOTIENT AND EXTERNAL GROUPING GUARANTEES

## B.1 TEXT-ONLY PROVENANCE RECOVERY BOUNDARY

Proposition 6 (Exact boundary for deterministic text-only provenance recovery). Fix a claim c and an integer $n \geq 2 .$ Let $\Pi _ { n }$ be the set ofpartitions $o f [ n ]$ and let $\mathcal { W } _ { c , n }$ be the class of admissible worlds with n observed records. Let $Z _ { c } : \bar { \mathcal { W } } _ { c , n } \to \mathcal { Z } ^ { \bar { n } }$ map each world to its complete model-visible text observation, and let $P _ { c } ^ { \star } : \mathcal { W } _ { c , n }  \Pi _ { n }$ map each world to its oracle claim-relative provenance partition. There exists a deterministic text-only rule $\Gamma _ { c } : \mathcal { Z } ^ { n } \to \Pi _ { n }$ satisfying $\Gamma _ { c } ( Z _ { c } ( w ) ) = P _ { c } ^ { \star } ( w )$ for every $w \in \mathcal { W } _ { c , n } \ i j$ and only if,for all $w _ { 0 } , w _ { 1 } \in \mathcal { W } _ { c , n } ,$

$$
Z _ { c } ( w _ { 0 } ) = Z _ { c } ( w _ { 1 } ) \quad \Longrightarrow \quad P _ { c } ^ { \star } ( w _ { 0 } ) = P _ { c } ^ { \star } ( w _ { 1 } ) .
$$

Proof of Proposition 6. For necessity, suppose a universally exact rule $\Gamma _ { c }$ exists. If $Z _ { c } ( w _ { 0 } ) =$ $Z _ { c } ( w _ { 1 } )$ , then functionality gives $\Gamma _ { c } ( \bar { Z } _ { c } ( w _ { 0 } ) ) = \Gamma _ { c } ( Z _ { c } ( w _ { 1 } ) )$ , while exactness identifies the two sides with $P _ { c } ^ { \star } ( w _ { 0 } )$ and $P _ { c } ^ { \star } ( w _ { 1 } )$ . Hence $P _ { c } ^ { \star } ( w _ { 0 } ) = P _ { c } ^ { \star } ( w _ { 1 } )$

For sufficiency, suppose $P _ { c } ^ { \star }$ is constant on each observational fiber of $Z _ { c }$ . For every $z \in Z _ { c } ( \mathcal { W } _ { c , n } )$ define $\Gamma _ { c } ( z )$ as the unique oracle-partition value attained on the fiber $Z _ { c } ^ { - 1 } ( \{ z \} )$ . Fiber constancy makes this definition well-defined. Since $\Pi _ { n }$ is nonempty, extend $\Gamma _ { c }$ arbitrarily outside $Z _ { c } ( \mathcal { W } _ { c , n } )$ Then $\Gamma _ { c } ( Z _ { c } ( w ) ) = P _ { c } ^ { \star } ( w )$ for every $w \in \mathcal { W } _ { c , n }$

## B.2 OPERATIONAL QUOTIENT AND FACTORIZATION

Let $\mathrm { C o n t } = \mathrm { F i n } ^ { + } ( \mathcal { Z } _ { \mathrm { e v } } )$ be the nonempty finite canonical-content sets and let

$$
\mathsf { Q u o t } = \mathsf { M } _ { \mathrm { f i n } } ( \mathsf { C o n t } ) = \{ m : \mathsf { C o n t } \to \mathbb { N } _ { 0 } : \operatorname { s u p p } ( m ) \mathrm { ~ i s ~ f i n i t e } \} .
$$

For $x \ = \ ( ( k _ { i } , z _ { i } ) ) _ { i = 1 } ^ { n } \ \in \ \mathsf { C f g }$ , write $K _ { x }$ for its occupied labels and $V _ { x } ( k ) = \{ z _ { i } : k _ { i } = k \}$ Equation (2) maps x to the outer multiset of these inner sets. Label equality encodes membership; label spelling is removed by the quotient.

Identity-sensitive representations use an explicit authenticated signature coordinate. Let $\Sigma$ be a nonempty space of evidence-bearing group signatures and define

$$
\begin{array} { c } { { \mathsf { C f g } _ { \Sigma } = \{ ( x , s _ { x } ) : x \in \mathsf { C f g } , s _ { x } : K _ { x } \to \Sigma \} , \qquad \mathsf { Q u o t } _ { \Sigma } = \mathsf { M } _ { \mathrm { f i n } } ( \Sigma \times \mathsf { C o n t } ) , } } \\ { { q _ { \mathrm { e v } } ^ { \Sigma } ( x , s _ { x } ) = \displaystyle \sum _ { k \in K _ { x } } \delta _ { ( s _ { x } ( k ) , V _ { x } ( k ) ) } . } } \end{array}\tag{9}
$$

Declare $( x , s _ { x } ) \sim _ { \Sigma } ( y , s _ { y } )$ exactly when $q _ { \mathrm { e v } } ^ { \Sigma } ( x , s _ { x } ) = q _ { \mathrm { e v } } ^ { \Sigma } ( y , s _ { y } )$ . A bijective label renaming $\beta : K _ { x } \stackrel { } { \to } K ^ { \prime }$ sends $( ( k _ { i } , \bar { z _ { i } } ) _ { i } , s _ { x } )$ to $( ( \beta ( k _ { i } ) , \ddot { z } _ { i } ) _ { i } , s ^ { \prime } )$ , where $s ^ { \prime } ( \beta ( \boldsymbol { k } ) ) = s _ { x } ( \boldsymbol { k } )$ . Thus label spelling is removed while each signature–content pair is retained. Because K is infinite, $q _ { \mathrm { e v } } ^ { \Sigma }$ is surjective. Consequently, for every $\mathbf { \bar { \Psi } } _ { \Sigma } : \mathsf { C f g } _ { \Sigma } \to \mathbf { \bar { \Psi } }$ , invariance on ${ \sim } _ { \Sigma }$ classes is equivalent to a unique factorization $\Psi _ { \Sigma } = \bar { \Psi } _ { \Sigma } \circ q _ { \mathrm { e v } } ^ { \Sigma }$ . The sufficiency, faithfulness, and prompt-visible statements below have the same typed analogues. For $a : \Sigma \times \mathsf { C o n t } \to \mathcal { X }$ , the additive analogue is

$$
E _ { a } ^ { \Sigma } ( x , s _ { x } ) = \sum _ { k \in K _ { x } } a ( s _ { x } ( k ) , V _ { x } ( k ) ) , \qquad \Phi _ { a } ^ { \Sigma } ( m ) = \sum _ { ( \sigma , A ) } m ( \sigma , A ) a ( \sigma , A ) ,
$$

and its faithfulness criterion is finite integer-linear independence over $( \sigma , A ) \in \Sigma \times { \mathsf { C o n t } }$

Proposition 7 (Operational quotient). The map $q _ { \mathrm { e v } } : { \mathsf { C f g } } \to$ Quot is surjective. Equality of its outputs is exactly the equivalence relation generated by record permutation, bijective renaming of occupied opaque labels, and insertion or deletion ofa within-group canonical duplicate while one occurrence remains. A nontrivial split of one occupied group into $r \geq 2$ nonempty groups, or a nontrivial merge of $\mathbf { \dot { r } } \geq 2$ occupied groups, changes the quotient state even when the union ofvisible canonical content is unchanged.

Proof. Each declared nuisance transformation leaves the outer multiset of inner content sets unchanged. Conversely, if $q _ { \mathrm { e v } } ( x ) = q _ { \mathrm { e v } } ( y )$ , equality of finite multisets gives a bijection between occupied labels whose matched groups have identical content sets. Rename labels by that bijection, delete duplicate occurrences inside each matched group, and permute the remaining records; the reduced configurations coincide. Reversing duplicate deletions recovers the originals, so the stated moves generate the whole equivalence relation.

For surjectivity, write any $m \in \mathsf { Q u o t }$ as $\begin{array} { r } { m = \sum _ { j = 1 } ^ { r } \delta _ { A _ { j } } } \end{array}$ . Choose r distinct labels and list one pair $( k _ { j } , z )$ for each $z \in A _ { i } ;$ the resulting configuration maps to m. The empty configuration maps to the empty multiset. Finally define $\begin{array} { r } { N ( \bar { m } ) = \sum _ { A } m ( A ) } \end{array}$ . A split changes N by $r - 1$ and a merge by $1 - r ;$ both operations therefore change the quotient state. □

ProofofTheorem 1. Proposition 7 identifies the three-operation invariance with constancy on quotient fibers. If $\Psi = \bar { \Psi } \circ q _ { \mathrm { e v } }$ and $x \sim y .$ , equality of the quotient states gives $\Psi ( x ) = \Psi ( y )$ ). Conversely, suppose Ψ is invariant. For $m \in { \mathsf { Q } }$ uot choose any $x _ { m }$ with $q _ { \mathrm { e v } } ( x _ { m } ) = m$ and set $\bar { \Psi } ( m ) = \Psi ( x _ { m } )$ Invariance makes this definition independent of the representative, and $\Psi = \bar { \Psi } \circ q _ { \mathrm { e v } }$ . Surjectivity of $q _ { \mathrm { e v } }$ makes Ψ<sup>¯</sup> unique.

If Ψ is invariant, every composition $h \circ \Psi$ is invariant. Conversely, quantifying over all downstream maps includes the identity map on $Y ;$ equivalently, binary maps can separate any two unequal points of $Y .$ . For a specified downstream family, point separation on $\bar { \Psi } ( \mathsf { C f g } )$ supplies the converse. The final faithfulness and recoverability equivalences follow from Proposition 8 below, whose proof uses only the factorization just established. □

## B.3 FAITHFULNESS, MINIMAL SUFFICIENCY, AND ADDITIVE REALIZATIONS

An enforcing representation $\Theta : { \mathsf { C f g } } \to W$ is quotient-faithful when $\Theta ( x ) = \Theta ( y )$ if and only if $x \sim y .$ . It is universally sufficient when, for every set $D ,$ every relation-invariant observable $f : { \mathsf { C f g } } \to D$ factors through Θ. Together, quotient faithfulness and universal sufficiency characterize exact preservation of relation-invariant information.

Proposition 8 (Minimal sufficiency and faithful recoding). The quotient $q _ { \mathrm { e v } }$ is universally sufficient. IfΘ is universally sufficient, then $q _ { \mathrm { e v } } = \rho \circ \Theta$ for some $\rho : W \to$ Quot. $\bar { I f } \Theta = \bar { \Theta } \circ q _ { \mathrm { e v } }$ is enforcing, then thefollowing are equivalent: Θ is universally sufficient; Θ is quotient-faithful; and Θ<sup>¯</sup> is injective on Quot.

Proof. Theorem 1 factors every invariant f through $q _ { \mathrm { e v } }$ , proving its universal sufficiency. If Θ is universally sufficient, apply its definition to the invariant observable $f = q _ { \mathrm { e v } }$ to obtain $q _ { \mathrm { e v } } = \rho \circ \Theta$ Hence $\Theta ( \overset { \cdot } { x } ) = \Theta ( y )$ implies $q _ { \mathrm { e v } } ( x ) = q _ { \mathrm { e v } } ( y )$ . Enforcement supplies the reverse implication, proving faithfulness.

For an enforcing $\Theta = \bar { \Theta } \circ q _ { \mathrm { e v } }$ , faithfulness implies injectivity of Θ<sup>¯</sup> by choosing representatives of any two quotient states. Conversely, injectivity of Θ<sup>¯</sup> makes equality of Θ equivalent to equality of $q _ { \mathrm { e v } }$ . The inverse of Θ<sup>¯</sup> on the reachable image recovers $q _ { \mathrm { e v } }$ , so every invariant observable factors through Θ; extend that factor arbitrarily outside the reachable image. □

Let $a : { \mathsf { C o n t } } \to \mathcal { X }$ be a deterministic group aggregator and define

$$
E _ { a } ( x ) = \sum _ { k \in K _ { x } } a ( V _ { x } ( k ) ) , \qquad \Phi _ { a } ( m ) = \sum _ { A \in \mathsf { C o n t } } m ( A ) a ( A ) .\tag{10}
$$

Proposition 9 (Additive faithfulness criterion). Every additive realization satisfies $E _ { a } = \Phi _ { a } \circ q _ { \mathrm { e v } }$ and therefore enforces the declared nuisance invariance. It is quotient-faithful if and only if the family $\{ a ( A ) : { \dot { A } } \in \mathsf { C o n t } \}$ isfinitely integer-linearly independent:

$$
\sum _ { A \in \mathsf { C o n t } } c ( A ) a ( A ) = 0 , \qquad c \in \mathbb { Z } ^ { ( \mathsf { C o n t } ) } \quad \Longrightarrow \quad c = 0 ,
$$

where the parenthesized exponent denotes finite support. A faithful realization always exists by taking $\mathcal { X } = \mathbb { R } ^ { ( \mathrm { { C o n t } } ) }$ and $a ( A ) = e _ { A }$ , the canonical coordinate vector.

Proof. Grouping equal outer multiset atoms in Equation (10) gives $E _ { a } ( x ) = \Phi _ { a } ( q _ { \mathrm { e v } } ( x ) )$ . If a nonzero integer relation exists, decompose $c = c ^ { + } - c ^ { - }$ into distinct nonnegative finite multisets. Then $\Phi _ { a } ( c ^ { + } ) = \Phi _ { a } ( c ^ { - } )$ , and surjectivity supplies two distinct quotient states that collide under $E _ { a }$ . Conversely, any collision $\bar { \Phi _ { a } ( m ) } = \bar { \Phi _ { a } ( m ^ { \prime } ) }$ for m $\neq m ^ { \prime }$ yields the nonzero integer relation $c = m - m ^ { \prime }$ . The canonical coordinate vectors are integer-linearly independent and expose every multiset coefficient, so they give a faithful realization. □

Corollary 10 (Prompt-visible enforcement criterion). Let $R : { \mathsf { C f g } } \to { \mathcal { P } }$ be the actual model-visible renderer. Uniform invariance $o f h \circ R$ for every binary downstream map $h : \mathcal { P }  \{ 0 , 1 \}$ holds ifand only $i f R = \bar { R } \circ q _ { \mathrm { e v } }$ . Therefore, $i f x \sim y$ but $R ( x ) \neq R ( y )$ , a deterministic binary downstream rule exists that distinguishes the two rendered prompts.

Proof. Apply Theorem 1 to R. Any pair of distinct rendered prompts is separated by a binary map, establishing necessity of the factorization criterion. □

The corollary characterizes uniform invariance over all binary downstream maps; fixed-model response is measured empirically. The default quotient carries content and group multiplicity, while Equation (9) additionally preserves authenticated identity, credibility, or reliability metadata. Opaque membership labels encode membership in both versions.

## B.4 CONCRETE ADDITIVE CONSTRUCTION AND GROUPING-ERROR BOUND

Let record r support target $T$ or competitor $C ,$ and let $g ( r )$ denote its supplied group. With group sets $G _ { T }$ and $G _ { C }$ , a unit-weight reference margin is

$$
D = b + | G _ { T } | - | G _ { C } | ,\tag{11}
$$

where b is the pre-context preference for T. Copying a record while preserving the group set leaves D unchanged. An exact-replication-invariant representation preserves the evidence state when another canonical copy is added to its existing group. The design separates evidential multiplicity from within-group content aggregation: unique content is aggregated into a bounded group representation, and each group contributes once. We realize this property both with an exact stable-representative construction and with exact deduplication followed by lossless within-group concatenation. Our mirrored topology gives the favored side two groups and the other side one group, then reverses that assignment while holding the records fixed.

We now give the complete guarantees for the external representation in Equation (4). The ambient record universe $U .$ , canonicalizer $z _ { \mathrm { e v } } : U  \mathcal { Z } _ { \mathrm { e v } }$ , and deterministic aggregator a are fixed across all compared partitions, and a receives the extensional set of canonical group elements.

ProofofProposition 3. Let $G \in P$ be the block containing $i ,$ set $I ^ { \prime } = I \cup \{ i ^ { \prime } \}$ , and define

$$
P ^ { \prime } = ( P \setminus \{ G \} ) \cup \{ G \cup \{ i ^ { \prime } \} \} .
$$

Because $z _ { \mathrm { e v } } ( i ^ { \prime } ) = z _ { \mathrm { e v } } ( i )$

$$
\{ z _ { \mathrm { e v } } ( j ) : j \in G \cup \{ i ^ { \prime } \} \} = \{ z _ { \mathrm { e v } } ( j ) : j \in G \} .
$$

The changed block therefore supplies exactly the same set to the fixed aggregator a, while every other block is unchanged. The two finite sums agree term by term, so $E _ { I ^ { \prime } } ( \breve { P ^ { \prime } } ) = E _ { I } ( \breve { P } )$ □

For two partitions $P$ and $Q$ of the same finite record set I, build a bipartite overlap graph whose vertices are their blocks and whose edges join blocks with nonempty record intersection. Exclude exactly those connected components containing one P-block and one Q-block with identical record sets. Let $\mathcal { C } _ { \mathrm { e r r } }$ denote the remaining components; for $c \in { \mathcal { C } } _ { \mathrm { e r r } }$ , let $p _ { c }$ and $q _ { c }$ be its numbers of $P -$ and Q-blocks, and define

$$
H ( P , Q ) = \sum _ { c \in \mathcal { C } _ { \mathrm { e r r } } } ( p _ { c } + q _ { c } ) .
$$

Proposition 11 (Partition-error bound). Suppose $B \geq 0 a n d \| a ( S ) \| \leq B$ for every semantic set that occurs under P or Q. Then

$$
\| E _ { I } ( P ) - E _ { I } ( Q ) \| \leq B H ( P , Q ) .
$$

Proof. Every excluded component contributes the same record block under both partitions. The same $z _ { \mathrm { e v } }$ and a therefore produce identical vectors, which cancel. In an erroneous component $c ,$ write $P _ { c }$ and $Q _ { c }$ for its blocks. The triangle inequality and the assumed bound give

$$
\begin{array} { r } { \displaystyle \left\| \displaystyle \sum _ { G \in { \cal P } _ { c } } a ( \{ z _ { \mathrm { e v } } ( i ) : i \in G \} ) - \displaystyle \sum _ { G \in { \cal Q } _ { c } } a ( \{ z _ { \mathrm { e v } } ( i ) : i \in G \} ) \right\| \le \displaystyle \sum _ { G \in { \cal P } _ { c } } \| a ( \{ z _ { \mathrm { e v } } ( i ) : i \in G \} ) \| } \\ { + \displaystyle \sum _ { G \in { \cal Q } _ { c } } \| a ( \{ z _ { \mathrm { e v } } ( i ) : i \in G \} ) \| } \\ { \le B ( p _ { c } + q _ { c } ) . } \end{array}
$$

The erroneous components are disjoint. Summing their differences and applying the triangle inequality once more yields $\| \bar { E } _ { I } ( P ) - E _ { I } ( \bar { Q } ) \| \le B H ( \bar { P , Q } )$ □

The coefficient one is worst-case sharp over this admissible class. For any $B > 0 ,$ , take $U = I =$ $\{ 1 , 2 \} , \mathcal { X } = \mathbb { R }$ , and $\mathcal { Z } _ { \mathrm { e v } } = \{ z _ { 1 } , z _ { 2 } \}$ with $z _ { 1 } \neq z _ { 2 }$ , where $z _ { \mathrm { e v } } ( 1 ) = z _ { 1 }$ and $z _ { \mathrm { e v } } ( 2 ) = z _ { 2 }$ . Define the fixed aggregator on every subset of $\mathcal { Z } _ { \mathrm { e v } }$ by

$$
a ( \emptyset ) = 0 , \qquad a ( \{ z _ { 1 } \} ) = a ( \{ z _ { 2 } \} ) = B , \qquad a ( \{ z _ { 1 } , z _ { 2 } \} ) = - B .
$$

For $P = \{ \{ 1 \} , \{ 2 \} \}$ and $Q \ = \ \{ \{ 1 , 2 \} \}$ , the overlap graph has one erroneous component with $H ( P , Q ) = 3$ , while $E _ { I } ( P ) = \mathrm { { 2 } } \dot { B }$ and $E _ { I } ( Q ) = - B$ . Hence $\| E _ { I } ( P ) - E _ { I } ( Q ) \| = 3 B =$ $B \dot { H } ( P , \dot { Q } )$ . This witness establishes worst-case sharpness over the allowed class.

Proposition 12 (Sharp content-aware partition-error bound). Fix a finite record set I, its canonicalizer, and partitions $P , Q o f I .$ For a fixed aggregator a with $\| a ( A ) \| \leq B$ on every occurring content set,

$$
\begin{array} { r } { \| E _ { I } ( P ) - E _ { I } ( Q ) \| \le B \Delta _ { q } ( P , Q ) \le B H ( P , Q ) . } \end{array}
$$

For eachfixed pair $P , Q$ and $B \geq 0 ,$

$$
\operatorname* { s u p } _ { a \colon | a ( A ) | \leq B } \left| \sum _ { A } ( q _ { P } ( A ) - q _ { Q } ( A ) ) a ( A ) \right| = B \Delta _ { q } ( P , Q ) ,
$$

where the supremum is over scalar aggregators on finite nonempty canonical-content sets. An L-Lipschitz downstream margin therefore obeys $| m ( \hat { E _ { I } } ( P ) ) - m ( \hat { E _ { I } } ( Q ) ) | \le L B \Delta _ { q } ( P , Q )$ , and its nonzero sign is stable $i f | m ( \bar { E } _ { I } ( P ) ) | \dot { > } L B \dot { \Delta } _ { q } ( \dot { P } , \dot { Q } )$ ).

Proof. Put $c _ { A } = q _ { P } ( A ) - q _ { Q } ( A )$ . The fixed canonicalizer and fixed aggregator give

$$
E _ { I } ( P ) - E _ { I } ( Q ) = \sum _ { A } c _ { A } a ( A ) .
$$

The triangle inequality bounds the norm by $B \sum _ { A } | c _ { A } |$ . Identical record blocks cancel between the two partitions. The total numbers of remaining blocks on the two sides sum to $H ( P , Q )$ ; mapping these blocks to their content sets can cancel additional mass, so $\Sigma _ { A } \left| c _ { A } \right| \leq H ( P , Q )$ . For sharpness, choose the single fixed scalar function $a ( A ) = B \operatorname { s g n } ( c _ { A } )$ , with $\mathrm { s g n } ( 0 ) = 0$ and zero on absent content sets. It attains $B \sum _ { A } | c _ { A } |$ This is a worst-case choice for the given pair; a particular implemented aggregator can have a smaller distortion. Lipschitz continuity and the strict margin argument prove the final two claims. □

The two discrepancies distinguish record-level and canonical-content-level changes. For example, let $z _ { 1 } = z _ { 2 } = u$ and $z _ { 3 } = z _ { 4 } = v$ with $u \ne v$ , and compare $P = \{ \{ 1 , 3 \} , \{ \bar { 2 } , 4 \} \}$ with $Q =$ $\{ \{ 1 , 4 \} , \{ 2 , 3 \} \}$ . Their overlap graph has $H ( P , Q ) = 4$ , but both quotient states equal $2 \delta _ { \{ u , v \} }$ , so $\Delta _ { q } ( P , Q ) = 0$ and every fixed set-based aggregator gives identical states. Conversely, the fourdistinct-element witness of Proposition 2 has $H = \Delta _ { q } = 4$ and attains distortion $4 B ,$ , despite zero change in group count. More generally, $\Delta _ { q }$ is a pseudometric on partitions and becomes an $\ell _ { 1 }$ metric on their distinct reachable quotient states.

Corollary 13 (Downstream decision stability). Let $L \geq 0$ and let $m : \mathcal { X } $ R be L-Lipschitz. Then

$$
| m ( E _ { I } ( P ) ) - m ( E _ { I } ( Q ) ) | \leq L B H ( P , Q ) .
$$

For a binary decision boundary at $m = 0 ,$ , the nonzero sign is unchanged whenever

$$
| m ( E _ { I } ( P ) ) | > L B H ( P , Q ) .
$$

Proof. The first claim follows by applying Lipschitz continuity to Proposition 11. Under the displayed strict condition, the largest admissible change in the margin is smaller than its distance from zero, so the sign remains unchanged. The displayed strict inequality provides a uniform two-sided guarantee: the perturbation remains strictly inside the reference margin. At equality the perturbed margin can reach zero, so a fixed tie convention covers a single reference sign. □

These guarantees compose with an externally specified representation pipeline whose provenance signals, supplied partition, and canonicalized content provide the upstream inputs. Authenticated signatures can carry verified identity without changing the role of opaque membership labels. External caps on the group aggregator instantiate B, and an independently verified Lipschitz bound instantiates L. Behavioral curves test grouping effects; system-level checks provide the numerical stability constants.

## C DETAILED EXPERIMENTAL SPECIFICATIONS

## C.1 CUE AND TOPOLOGY FAMILIES

The decision audit has five prompt families. Marker swaps which value is labeled verified. Graph swaps anonymous record-to-node edges. Provenance uses the same graph with nodes explicitly described as origins. Rule adds a same-origin-counts-once instruction. Full crosses marker correctness and topology correctness. Within a paired family, candidate strings, claims, record order, value-table order, literal offsets, and the full tokenizer multiset are fixed.

## C.2 DIRECT PIPELINE MODES

Edge asks whether two specified records share a root. Count asks for the number of distinct roots supporting a deterministically selected abstract value. Winner asks which value has more distinct roots. Opaque oracle supplies the 2:1 count summary after removing the graph. Semantic oracle supplies the same summary alongside the original fact task. End-to-end supplies graph, rule, and fact task together. All modes use direct semantic candidates.

## C.3 REPETITION DOSES

At every dose, two records supporting the factual target arise from two distinct roots. The competitor has one root, whose record is repeated D times. Raw hides root IDs; Prompt preserves all records while exposing the root mapping and deduplication rule. External retains one representative per oracle root and removes root instructions from the final prompt. This stable-representative operator certifies multiplicity control; the broader systems recommendation aggregates complementary within-root content under one bounded evidential contribution. Forward and reverse record orders are both evaluated. The External and Raw $D = 1$ serialized inputs are identical for all 282 item-presentationorder pairs in each checkpoint.

## C.4 NATURAL-TEXT SUPPLIED UNITS

Phase 8 freezes two domains before model inference. HUMAN EVIDENCE contains 101 PERSPEC-TRUM claims grouped into 49 dependence components. Each side supplies four different crowdannotated stance-perspective clusters, matched one-to-one to side-exclusive evidence-unit IDs and unique normalized and visible-text hashes. Each ID indexes one crowd-annotated stance-perspective evidence unit. WEB CONSENSUS contains 138 ConflictingQA questions in 135 dependence components, with frozen two-model-consensus stance groupings. Each selected eight-record panel has operationally separated registrable domains, canonical URLs, normalized-text hashes, and visible-text hashes across both sides. A WEB unit is a supplied canonical-URL key under these frozen criteria. Publisher identity and ownership enter through explicit authenticated signature coordinates. The two domains are analyzed separately with domain-specific component bootstraps.

Text is normalized with Unicode NFKC, removal of control, format, and private-use characters, and whitespace collapse while preserving ASCII punctuation. Eligible evidence has at least 400 normalized characters. The model-visible field is a 400-character head–tail excerpt comprising the first 200 characters, the literal marker [...] , and the last 193. Every Base prompt contains three evidence records; Copy, Prompt, and Four-Unit contain six. Copy and Four-Unit match record count and 2,400 visible evidence characters; Phase 11 additionally matches final character, byte, and tokenizer length exactly.

For each claim and attack side, a frozen hash selects two opposite-side anchors and one dose anchor. BASE contains these three units. COPY repeats the dose anchor four times. PROMPT preserves the same six evidence texts and record count as Copy while showing short opaque aliases and a same-unit-counts-once rule. FOUR-UNIT retains the same anchors and dose anchor, replacing three copies with three other attack-side units. The machine artifacts retain the frozen condition key DISTINCT. Support and undermine attacks and original and reverse record orders yield 16 trials per claim. Opaque aliases are fixed by canonical role before reversal; reversal preserves every field and changes the record sequence.

Candidates are the direct semantic strings Yes and No; scoring starts from each tokenizer’s native assistant state and termination sequence. The primary outcome is strict selection of the manipulated side, with an exact likelihood tie scored zero. We average two attack sides and two orders within claim before forming contrasts. Each domain uses its own frozen dependence-component bootstrap stream for every contrast and metric; a component joins claims that reuse a supplied key or final visible text. The point estimand remains claim-weighted. Ten thousand replicates use master seed 20260828 with deterministic domain-specific offsets (20260828 for WEB and 20270828 for HUMAN); auxiliary sensitivities use separately frozen offsets, and ordinary item bootstrap provides sensitivity estimates.

Copy–Base tests replication invariance: its 90% interval must lie entirely inside [−5, +5] points. Four-Unit–Base, Four-Unit–Copy, and Copy–Prompt each require a point estimate of at least +5 points and a 95% lower bound above zero. These gates are evaluated separately for every checkpoint and domain. For each checkpoint, 956 byte-for-byte checks show that stable collapse of Copy by its hidden unit key reproduces the corresponding Base prompt. Consequently, upstream recovery is numerically identical to Copy–Base and represents the same evidence contrast.

The 5-point operational margin corresponds to approximately one changed choice per 20 constrained decisions. The stricter 10-point Phase 6 marker criterion was likewise frozen before treatment inference. The two gate types answer different questions: an equivalence band assesses whether an effect is small enough to treat as negligible, whereas a positive-effect gate assesses whether an estimate reaches a stated magnitude with direction supported by its interval. Appendix Table 14 reports a 2.5/5/7.5/10-point sensitivity sweep around the frozen gates. Copy–Base lies outside equivalence in all eight model–domain cells at every margin; model-specific estimates and intervals are primary, and pass counts summarize gate-level magnitudes.

## C.5 EXTENDED CHECKPOINT AND GROUPING SPECIFICATIONS

Added checkpoint grid. The complete Phase 8 grid is repeated with the same prompts, candidates, estimands, and bootstrap streams on Qwen3-14B-AWQ and Phi-4. Qwen3-14B uses the publisher’s AWQ 4-bit checkpoint; Phi-4 uses dynamic NF4 inference. The Qwen3-8B NF4 rerun is counted as a quantization-control inference configuration under the existing Qwen3-8B checkpoint identity.

Each inference configuration completes 3,824 trials, for 11,472 new candidate-choice forward trials. The main paper reports the two added 14B checkpoints; the control remains bound in the machine snapshot and trial accounting.

Controlled answer-plus-one-sentence generation. The generation panel freezes 40 HUMAN and 40 WEB items, one per Phase 8 dependence component, before inference. For each item it crosses BASE, raw four-copy, and prompt-rule conditions with two attack sides and two record orders, yielding 960 trials per checkpoint. Qwen3-8B runs in bfloat16, Qwen3-14B uses its official AWQ checkpoint, and Phi-4 uses dynamic NF4 inference. Greedy decoding requests a leading Yes or No followed by one concise sentence, with at most 48 new tokens. All-trial leading-answer selection is the primary endpoint; invalid outputs receive zero attack selection. The valid-output analysis is a frozen sensitivity. The explanatory sentence is retained in the output, and scoring uses deterministic leading-answer parsing.

For each domain and checkpoint, we average the two attack sides and two orders within item, then bootstrap the 40 frozen dependence components 10,000 times. All three formal runs contain 960/960 predictions and zero invalid outputs, for 2,880 controlled-generation trials.

False splits and false merges after fixing supplied keys. This audit conditions on the supplied Phase 8 grouping keys and injects partition error before record emission. Eligibility requires both sides of a claim to contain a frozen document of at least 160 whitespace-delimited words. The resulting HUMAN panel has 66 claims in 32 dependence components; the WEB panel retains 138 questions in 135 components. This filter is model-independent and fixed before treatment inference.

For the one-group panel, the attack material consists of four non-overlapping 40-word windows from one frozen document. For the four-group panel, four 40-word windows come from four supplied operational units. Estimated keys emit one, two, or four records after within-group exact-text deduplication and a fixed per-group content budget. The richer PARTIAL-G1-FULL-CONTENT endpoint concatenates all four unique windows into one 160-word record while keeping estimated group count at one; subtracting PARTIAL-G1 measures the full-window inclusion gain relative to the one-window 40-word endpoint. The main content-fixed contrasts retain all 160 attack words in the one-record condition: PARTIAL-G4 minus PARTIAL-G1-FULL-CONTENT measures false splitting of one supplied document, and DIST-G4 minus DIST-G1-FULL-CONTENT measures the loss from merging four supplied operational units. In each pair, the ordered attack words are identical and record placement implements the grouping intervention.

The primary outcome is the attack-side likelihood share in Equation (7). Secondary endpoints are strict selection, half-tie selection, and likelihood margin. Two attack sides and two orders are averaged within item, and 10,000 bootstraps resample supplied-key or final-text connected components while retaining all member claims. Twelve conditions produce 9,792 trials per checkpoint and 39,168 across Qwen3-8B, Qwen3-4B, Phi-4-mini, and Mistral-7B. Conditioning on the supplied operational keys, the experiment identifies the causal effect of downstream record placement.

Matched six-slot serialization control. Phase 9 fixes evidence words while jointly varying partition rendering and its deterministic serialization. Its four-record arm contains three additional JSON objects, ordinal record IDs, side and evidence headers, and object separators. The serialized difference is 147–153 UTF-8 bytes and 43–63 tokenizer input tokens across the four checkpoints, while the evidence-token difference is bounded by five. Table 1 therefore estimates the total effect of operational record placement under fixed evidence words.

Phase 11 removes these count and length differences. Both arms use six JSON objects with the same R1–R6 identifiers, side labels, headers, separators, instruction, query, claim, anchor content, and candidate format. The same ordered 160-word attack stream is placed in four attack-side slots as either [160, 0, 0, 0] or [40, 40, 40, 40] words; the empty slots and their headers remain model-visible in both arms. A deterministic outcome-blind search adds up to four whitespace characters after the final JSON brace so that each tokenizer-specific pair has exactly the same final character count, UTF-8 byte count, and input-token count while parsing to the same JSON records. Model-visible, parse-inert whitespace supplies the exact length match. The identified treatment is the placement of a fixed word stream across one versus four nonempty content-bearing record slots within a matched six-slot skeleton.

Each checkpoint completes 3,264 choice trials, yielding 13,056 trials and 26,112 physical candidate forward calls with zero failures. Two sides and two orders are averaged within item; 10,000 replicates resample the same dependence components used by the parent audit. All 16 model–domain–contrast point estimates have the predicted positive sign, and 12 intervals lie entirely above zero. The matched control identifies a partition-boundary effect beyond extra objects, repeated headers, identifiers, separators, and total tokenizer length.

Table 3: Phase 11 matched-serialization control. Effects are percentage-point changes in $p _ { \mathrm { a t t a c k } }$ with 95% dependence-component bootstrap intervals. The intervention is matched six-slot serialization; the same ordered evidence word stream is placed across one versus four nonempty content-bearing record slots. Model-visible, parse-inert pair padding equalizes final character, byte, and tokenizer lengths. The estimand is operational content placement across record slots within the matched skeleton.
<table><tr><td>Model</td><td>Domain</td><td>One supplied unit: split Four supplied units: merge loss</td><td></td></tr><tr><td>Qwen3-8B</td><td>WEB (138/135)</td><td>+8.27 [+5.73, +10.82]</td><td>+8.00 [+5.18, +10.88]</td></tr><tr><td>Qwen3-8B</td><td>HUMAN (66/32)</td><td>+5.92 [+2.39, +8.87]</td><td>+11.27 [+7.37, +15.96]</td></tr><tr><td>Qwen3-4B</td><td>WEB (138/135)</td><td>+9.95 [+6.90, +13.04]</td><td>+13.33 [+10.29, +16.43]</td></tr><tr><td>Qwen3-4B</td><td>HUMAN (66/32)</td><td>+1.24 [-2.48, +5.69]</td><td>+7.47 [+3.57, +12.23]</td></tr><tr><td>Phi-4-mini</td><td>WEB (138/135)</td><td>+1.24 [+0.38, +2.29]</td><td>+1.25 [+0.40, +2.39]</td></tr><tr><td>Phi-4-mini</td><td>HUMAN (66/32)</td><td>+0.63 [-0.02, +1.65]</td><td>+0.99 [+0.01, +2.44]</td></tr><tr><td>Mistral-7B</td><td>WEB (138/135)</td><td>+3.22 [+1.11, +5.27]</td><td>+1.46 [-0.40, +3.34]</td></tr><tr><td>Mistral-7B</td><td>HUMAN (66/32)</td><td>+6.46 [+3.65, +8.84]</td><td>+2.33 [-0.65, +5.73]</td></tr></table>

Controlled GEO proliferation panel. The frozen panel contains 48 fictional product pairs. For each product and attack side, the same-root family uses four exactly 40-word genre-varied records bound to one three-fact campaign brief and one oracle root. The independent-root family uses four exactly 40-word records bound respectively to laboratory, user-panel, endurance, and service-data facts generated by four oracle roots. The source validator confirms that every visible claim belongs to its enumerated atomic-fact set, each attack stream contains exactly 160 ordered words, and hidden root, author, and synthetic-domain fields are absent from the prompt. Both structures use the Phase 11 six-slot renderer and compare [160, 0, 0, 0] with [40, 40, 40, 40] under exact character, UTF-8 byte, and model-token length matching.

Each checkpoint completes 768/768 choices with zero failures, yielding 3,072 choices and 6,144 candidate forward calls. Every condition–attack-side–order cell contains 48 trials. The primary item-first estimand averages the two attack sides and two orders before taking the contrast; 10,000 fixed-seed bootstrap replicates resample the 48 items. A descriptive mirror decomposition reveals substantial presentation interactions. For Qwen3-8B, the independent-root contrast is +41.43 points in original order and −49.92 in reverse order. The main table therefore reports balanced mirrored effects for each checkpoint and preserves both reversals whose pointwise 95% intervals exclude zero.

Grouping baselines and downstream distortion. We evaluate four grouping rules on 192 withinitem partition problems containing 768 records. Oracle root identifiers and single-link cosine clustering with all-MiniLM-L6-v2 at threshold 0.80 recover every partition. Normalized exact hashing and 128-permutation, three-word-shingle MinHash at threshold 0.80 emit all four same-root records as singletons while preserving all four independent roots. On the 96 same-root scopes, these lexical methods have pairwise recall and F1 equal to zero, 576 false-split pairs, adjusted Rand index zero, B-cubed F1 0.40, and total H = 480. Table 4 reports partition accuracy, and Table 5 maps every predicted partition to the corresponding executed intervention arm. Thresholds were fixed before result inspection. The embedding result establishes separability for these controlled templates; Proposition 6 characterizes the observational-collision boundary for universal exact recovery.

Table 4: Grouping baselines on the controlled provenance panel. Exact partition rates are over 96 attack-side scopes per provenance structure. H is the total partition-error quantity from Proposition 11.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Same-root exact</td><td rowspan="2">Independent-root</td><td rowspan="2">Total H</td></tr><tr><td>exact</td></tr><tr><td>Oracle root</td><td>100%</td><td>100%</td><td>0</td></tr><tr><td>Exact normalized hash</td><td>0%</td><td>100%</td><td>480</td></tr><tr><td>MinHash</td><td>0%</td><td>100%</td><td>480</td></tr><tr><td>Sentence embedding</td><td>100%</td><td>100%</td><td>0</td></tr></table>

Table 5: Downstream grouping distortion on same-root campaign scopes. Each entry is $| p _ { \mathrm { a t t a c k } } ( \widehat { P } ) -$ $p _ { \mathrm { a t t a c k } } ( P ^ { \star } ) |$ in percentage points. The predicted partitions coincide exactly with executed matchedintervention arms. Independent-root distortion is zero for all three methods because each preserves the four oracle roots.
<table><tr><td>Method</td><td>Qwen3-8B</td><td>Qwen3-4B</td><td>Phi-4-mini</td><td>Mistral-7B</td></tr><tr><td>Exact normalized hash</td><td>18.46</td><td>9.66</td><td>27.44</td><td>4.44</td></tr><tr><td>MinHash</td><td>18.46</td><td>9.66</td><td>27.44</td><td>4.44</td></tr><tr><td>Sentence embedding</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr></table>

## D ADDITIONAL RESULTS

## D.1 ORDER DECOMPOSITION OF CONTROLLED PARTITION EFFECTS

Table 6: Post-hoc order decomposition of the existing controlled GEO predictions. Both families use four content-bearing records minus one grouped record. Entries are percentage points with pointwise 95% bootstrap intervals over 48 item-components after averaging the two attack sides. Interaction is original minus reverse, estimated with paired resampling. Each checkpoint and family is reported separately; the intervals are descriptive. Values smaller than 0.005 points in magnitude round to 0.00.
<table><tr><td>Model</td><td>Roots</td><td>Original order</td><td>Reverse order</td><td>Order interaction</td></tr><tr><td>Qwen3-8B</td><td>3 Same root</td><td> $+ 1 6 . 8 3 [ + 1 2 . 3 7 , + 2 1 . 5 4 ]$ </td><td> $+ 2 0 . 0 9 [ + 1 5 . 9 1 , + 2 4 . 4 6 ]$ </td><td> $- 3 . 2 6 [ - 9 . 9 1 , + 3 . 3 7 ]$ </td></tr><tr><td></td><td></td><td>Qwen3-8B Independent +41.43[+36.15, +46.22]</td><td> $- 4 9 . 9 2 { \bar { [ - 4 9 . 9 5 , - 4 9 . 8 8 ] } }$ </td><td> $+ 9 1 . 3 6 [ + \dot { 8 } 6 . 0 7 , + 9 6 . 1 4 \dot { ] }$ </td></tr><tr><td>Qwen3-4B Same root</td><td></td><td>-13.97[-19.26, -9.39]</td><td>+33.28[+29.53, +36.85]</td><td> $- 4 7 . 2 6 [ - 5 3 . 9 3 , - 4 1 . 2 1 ]$ </td></tr><tr><td></td><td>Qwen3-4B Independent</td><td>+10.98[+6.43, +15.92]</td><td>-0.19[−0.28, −0.12]</td><td> $+ 1 1 . 1 7 [ + 6 . 6 3 , + 1 6 . 1 0 ]$ </td></tr><tr><td>Phi-4-mini Same root</td><td></td><td>+17.95[+14.19, +21.99]+36.93[+33.56, +39.98]</td><td></td><td> $- 1 8 . 9 8 [ \dot { - } 2 3 . 4 2 , - 1 4 . 3 2 \dot { ] }$ </td></tr><tr><td></td><td>Phi-4-mini Independent</td><td> $+ 1 9 . 1 5 [ + 1 5 . 0 9 , + 2 3 . 2 9 ]$ </td><td> $+ 1 2 . 3 1 [ + 9 . 1 3 , + 1 5 . 7 9 ]$ </td><td> $+ 6 . 8 4 [ + 2 . 3 3 , + 1 1 . 3 5 ]$ </td></tr><tr><td>Mistral-7B Same root</td><td></td><td> $- 8 . 8 \dot { 8 } [ - 1 4 . 6 4 , - 3 . 2 9 \dot { ] }$ </td><td>0.00[0.00, 0.00]</td><td> $- 8 . 8 8 \bar { [ - 1 4 . 6 4 , - 3 . 2 9 \bar { ] } }$ </td></tr><tr><td></td><td>Mistral-7B Independent</td><td> $+ 6 . 0 \dot { 3 } [ + 3 . 2 7 , + 9 . 2 8 \dot { ] }$ </td><td> $+ 0 . 0 1 [ + 0 . 0 1 , + 0 . 0 1 ]$ </td><td> $+ 6 . 0 \dot { 3 } [ + 3 . 2 6 , + 9 . 2 7 ]$ </td></tr></table>

Table 6 reuses all 3,072 stored Phase 12 choices. Each model has 48 items, two root families, two attack sides, two order mirrors, and two partition arms. For each item, family, and order, we subtract the one-group probability from the four-group probability and then average the two attack sides. The original-minus-reverse contrast is formed within the same item. We resample the 48 distinct item-components 10,000 times with seed 20260907, keeping every paired contrast together, and report percentile intervals. These are post-hoc descriptive intervals; the primary balanced estimates and their original intervals remain as reported in Table 2. The decomposition makes the presentation dependence of each checkpoint visible alongside its balanced effect.

## D.2 CHECKPOINT AND GENERATION BREADTH

Table 7: Breadth checks at added 14B checkpoints and in controlled generation, in percentage points with 95% dependence-component bootstrap intervals. The generation endpoint is the parsed leading Yes/No answer.
<table><tr><td>Checkpoint</td><td>Domain</td><td>Copy-Base</td><td>Copy-Prompt</td></tr><tr><td colspan="4">Added 14B checkpoints, complete-candidate likelihood</td></tr><tr><td>Qwen3-14B official AWQ 4-bit HUMAN</td><td></td><td> $+ 5 1 . 4 9 [ + 4 5 . 7 1 , + 5 7 . 4 3 ]$ </td><td>+4.70[+0.24, +9.23]</td></tr><tr><td>Qwen3-14B official AWQ 4-bit WEB</td><td></td><td> $+ 3 0 . 6 2 \bar { [ + 2 5 . 9 0 , + 3 5 . 4 0 \bar { ] } }$ </td><td>+0.91[-2.14, +3.93]</td></tr><tr><td>Phi-4 14B dynamic NF4</td><td>HUMAN</td><td>+51.98[+45.41, +58.82]</td><td>+9.16[+5.46, +12.99]</td></tr><tr><td>Phi-4 14B dynamic NF4</td><td>WEB</td><td>+43.30[+38.13, +48.52]</td><td>+1.63[−1.48, +4.86]</td></tr><tr><td colspan="4">Controlled generation, leading-answer endpoint</td></tr><tr><td>Checkpoint</td><td></td><td>Leading Copy-Base</td><td></td></tr><tr><td>Qwen3-8B</td><td>HUMAN</td><td>+20.62[+15.00,+26.25]</td><td></td></tr><tr><td>Qwen3-8B</td><td>WEB</td><td>+15.62[+9.38, +22.50]</td><td></td></tr><tr><td>Qwen3-14B-AWQ</td><td>HUMAN</td><td>+16.25[+9.38, +23.12]</td><td></td></tr><tr><td>Qwen3-14B-AWQ</td><td>WEB</td><td>+16.88[+10.62, +23.12]</td><td></td></tr><tr><td>Phi-4</td><td>HUMAN</td><td>+20.00[+13.75, +26.88]</td><td></td></tr><tr><td>Phi-4</td><td>WEB</td><td>+20.00[+13.12,+27.50]</td><td></td></tr></table>

## D.3 GROUPING-KEY ROBUSTNESS CURVES

![](images/0c6bfee3b9fc68e971ed24a4c82f827ec87f300e6e7f6ab04093ebec79f2494c.jpg)

![](images/9a9a15e069d185913b472307961b912653a6e048dcc5974639bdf4fdb29ef1f7.jpg)

![](images/50ea239a3ce0a34377f20f7612f4d24f6dba987a6d48cd60b56bee9909858016.jpg)

![](images/eb078177a0b28a1f5a316516991478de4dfbd38fd7b8044bb6ebeb0f072ab7a5.jpg)  
Figure 3: Fixed-budget supplied-key robustness curves. The partial-copy panels split non-overlapping 40-word windows from one frozen document into one, two, or four emitted records. The four-unit panels merge four operationally distinct supplied units into one, two, or four emitted records. The curves jointly vary grouping and the fixed per-group content budget; the content-fixed contrasts in Table 1 provide the primary comparison. HUMAN contains 66 claims in 32 dependence components; WEB contains 138 claims in 135 components. Each domain and checkpoint is plotted separately.

The dose curves vary emitted group count together with the fixed 40-word budget per group and show system behavior as an estimated partition moves between one, two, and four groups. Table 1 reports both the one-group full-window inclusion gain and the content-fixed comparisons in which the ordered 160 attack words remain constant while record placement changes.

## D.4 MULTIPLICITY-ADJUSTED BOOTSTRAP SENSITIVITY

Table 8: Bonferroni familywise bootstrap sensitivity. Counts show cells whose two-sided interval has a positive lower bound. Adjusted 95% intervals use percentile $\alpha / ( 2 m )$ tails with m = 24 for every Table 1 row and m = 16 for the matched six-slot row.
<table><tr><td>Reported subset</td><td>Cells</td><td>Ordinary 95%</td><td>Familywise 95%</td></tr><tr><td>Table 1: full-content gain</td><td>8</td><td>8/8</td><td>7/8</td></tr><tr><td>Table 1: content-fixed split/merge</td><td>16</td><td>16/16</td><td>16/16</td></tr><tr><td>Table 1: all contrasts</td><td>24</td><td>24/24</td><td>23/24</td></tr><tr><td>Matched six-slot control</td><td>16</td><td>12/16</td><td>11/16</td></tr></table>

We apply Bonferroni-adjusted percentile intervals within two design-distinct result families using 10,000 dependence-component bootstrap replicates regenerated from the frozen seeds. For the 24 Table 1 cells, simultaneous two-sided 95% intervals retain positive lower bounds in 23 cells: all 16 content-fixed split/merge effects and seven of eight full-content gains. The Phi-4-mini/WEB full-content gain is +2.91 points with adjusted interval [−0.02, +5.95]. Directional familywise lower bounds are positive in all 24 cells. For the 16 matched six-slot effects, 11 simultaneous two-sided intervals retain positive lower bounds. These counts summarize multiplicity sensitivity; the model–domain estimates remain the reported effects.

## D.5 OBSERVABLE PIPELINE STAGES

Table 9: Equal-domain strict accuracy (%) for observable pipeline stages. Each estimate averages mirrored directions and three presentations within item.
<table><tr><td>Checkpoint</td><td>Edge</td><td>Count</td><td>Winner</td><td>Opaque 2:1</td><td>Semantic 2:1</td><td>End-to-end</td></tr><tr><td>Qwen3-8B</td><td>58.85</td><td>47.10</td><td>51.04</td><td>100.00</td><td>54.30</td><td>49.64</td></tr><tr><td>Qwen3-4B</td><td>54.26</td><td>48.93</td><td>48.94</td><td>98.93</td><td>53.26</td><td>49.64</td></tr><tr><td>Phi-4-mini</td><td>50.00</td><td>49.61</td><td>51.77</td><td>74.85</td><td>57.23</td><td>49.28</td></tr></table>

## D.6 PHASE 6 BEHAVIORAL GATES

Across the frozen Marker, M–P, and Rescue gates, all four checkpoints remain below threshold. The provenance-equivalence gate is satisfied by Qwen3-8B, Qwen3-4B, and Mistral; Phi lies outside the equivalence band. The Qwen3-8B tie leaves every gate unchanged under the half-tie sensitivity analysis.

Table 10: Complete-candidate effects on the equal-domain 47-item panel. Likelihood contrasts use 95% item-bootstrap intervals. Accuracy effects are percentage points; provenance uses a 90% interval for the frozen equivalence test and the others use 95% intervals.
<table><tr><td colspan="2">Checkpoint</td><td colspan="2">Marker LL</td><td colspan="2">Prov. LL</td><td colspan="2">M-PLL</td></tr><tr><td>Qwen3-8B</td><td></td><td>1.618[1.350, 1.876]</td><td></td><td>-0.349[-0.531, -0.170]</td><td></td><td></td><td>1.967[1.590, 2.321]</td><td></td></tr><tr><td>Qwen3-4B</td><td></td><td>2.175[1.343, 3.180]</td><td></td><td></td><td></td><td>-0.243[−0.448, -0.045]</td><td></td><td>2.418[1.553, 3.435]</td></tr><tr><td></td><td>Phi-4-mini Mistral-7B</td><td></td><td>1.102[0.628, 1.598] 2.586[2.086, 3.115]</td><td></td><td>0.306[0.025, 0.602] -0.213[−0.301, -0.127]</td><td></td><td></td><td>0.796[0.248, 1.320] 2.799[2.279, 3.347]</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Checkpoint</td><td></td><td>Marker acc.</td><td></td><td>Prov. acc.</td><td></td><td></td><td>M-P acc.</td><td>Rescue acc.</td></tr><tr><td>Qwen3-8B</td><td></td><td>2.87[0.69, 5.74]</td><td></td><td>-0.72[-2.17,0.00]</td><td></td><td></td><td>3.59[0.69, 7.25]</td><td>0.00[0.00, 0.00]</td></tr><tr><td>Qwen3-4B</td><td></td><td>2.90[0.00, 6.52]</td><td></td><td>0.00[−1.45, 1.45]</td><td></td><td>2.90[0.00, 6.52]</td><td></td><td>0.00[0.00, 0.00]</td></tr><tr><td>Phi-4-mini</td><td></td><td>3.47[-2.90, 9.78]</td><td></td><td>2.93[-1.36, 7.31]</td><td></td><td>0.54[−7.94, 8.42]</td><td></td><td>0.66[-3.62, 5.62]</td></tr><tr><td>Mistral-7B</td><td></td><td>3.59[0.72, 6.52]</td><td></td><td>0.00[0.00, 0.00]</td><td></td><td>3.59[0.72, 6.49]</td><td></td><td>-1.45[—3.62, 0.00]</td></tr></table>

## D.7 PHASE 7 PRIMARY CONTRASTS

Table 11: Same-root repetition contrasts in percentage points, equal-domain item means with 95% bootstrap intervals. A pass requires an estimate of at least 5 points and an interval lower bound above zero.
<table><tr><td>Checkpoint</td><td>Raw D1-D8</td><td>Prompt-Raw D8</td><td>External-Raw D8</td><td>External-Prompt D8</td></tr><tr><td>Qwen3-8B</td><td> $1 . 4 2 [ - 0 . 0 3 , 3 . 2 0 ]$ </td><td> $0 . 3 5 [ - 2 . 4 8 , 3 . 5 6 ]$ </td><td> $1 . 4 2 [ - 0 . 0 3 , 3 . 1 9 ]$ </td><td> $1 . 0 7 [ - 1 . 4 5 , 3 . 6 1 ]$ </td></tr><tr><td>Qwen3-4B</td><td> $4 . { \dot { 3 5 } } [ 1 . 0 9 , 8 . 7 0 { \dot { ] } }$ </td><td> $0 . 0 0 \bar { [ - 1 . 8 1 , 1 . 8 1 \bar { ] } }$ </td><td>4.35[1.09, 8.70]</td><td> $4 . { \dot { 3 } } 5 [ 0 . 7 2 , 8 . 7 0 { \dot { ] } }$ </td></tr><tr><td>Phi-4-mini</td><td>27.02[21.60, 32.43]</td><td> $- 7 . 0 2 [ - \dot { 1 } 2 . 9 4 , - 0 . 9 5 ]$ </td><td>27.02[21.63, 32.44]</td><td> $3 4 . 0 4 [ 2 \dot { 7 } . 9 4 , 4 0 . 0 7 \dot { ] }$ </td></tr></table>

Phi satisfies the attack, external-recovery, and external-advantage gates. The prompt-recovery pass count is 0/3. Because External equals Raw D = 1, external recovery and attack reversal are the same numerical contrast.

The Phi attack remains positive in both domains (30.43 and 23.61 points), all three presentations (38.41, 24.18, and 18.48 points), and both record orders (33.06 and 20.98 points). Across its six presentation-by-order cells, the magnitude spans 4.35 to 48.64 points. Qwen3-4B yields a 4.35-point aggregate change on KNOWN items and five of 47 items.

![](images/b554626915b96efb988ba5bb2adf9f7da528a782ee09c4c324935b31a6ea9d6d.jpg)  
Figure 4: Strict factual choice in the controlled 47-item panel under repeated records from one wrong root. Raw prompts hide shared roots; Prompt prompts show root IDs and a deduplication rule. The green external point is the oracle-folded input and is exactly the Raw D = 1 input. The vertical axis starts at 40%; points are equal-domain item means and paired-contrast intervals appear in Table 11.

## D.8 PHASE 8 GATE PATTERN AND TOKEN SENSITIVITY

![](images/74417d97e9b5580065744ab708c5afc45de2a6d4d49af18bf341571542fa4ff0.jpg)  
Figure 5: Record multiplicity changes complete-candidate decisions in both natural-evidence domains. Copy–Base uses a 90% interval and the shaded ±5-point invariance band; other contrasts use 95% intervals.

Table 12: Phase 8 strict attack-side selection effects in percentage points. HUMAN uses annotated evidence units. WEB uses supplied canonical-URL units under the frozen operational separation criteria; a two-model consensus fixes stance labels. Copy–Base uses a 90% dependence-component bootstrap interval for the frozen ±5-point invariance gate; the other columns use 95% intervals. Bold cells satisfy their prespecified gate. Every domain and checkpoint is analyzed separately.
<table><tr><td>Checkpoint</td><td>Copy-Base</td><td>Four-Unit-Base</td><td>Four-Unit-Copy</td><td>Copy-Prompt</td></tr><tr><td colspan="5">HUMAN: annotated supplied evidence units</td></tr><tr><td>Qwen3-8B</td><td> $+ 4 1 . 8 3 [ + 3 6 . 7 3 , + 4 7 . 3 0 ]$ </td><td> $+ 4 4 . 0 6 [ + 3 8 . 1 1 , + 5 0 . 5 7 ]$ </td><td> $+ 2 . 2 3 [ - 0 . 7 4 , + 5 . 2 9 ]$ </td><td> $- 3 . 4 7 [ - 5 . 1 8 , - 1 . 7 2 ]$ </td></tr><tr><td>Qwen3-4B</td><td> $+ 4 3 . 3 2 \bar { [ + 3 9 . 2 4 , + 4 7 . 5 8 ] }$ </td><td> $+ 4 8 . 2 7 [ + 4 2 . 9 3 , + 5 3 . 0 4 ]$ </td><td> $+ 4 . 9 5 [ + 1 . 7 4 , + 7 . 8 0 ]$ </td><td> $+ 5 . 6 9 \dot { [ + 2 . 1 3 , + 9 . 3 1 ] }$ </td></tr><tr><td>Phi-4-mini</td><td> $+ 1 4 . 1 1 [ + 1 0 . 8 2 , + 1 7 . 9 3 ]$ </td><td> $+ 1 4 . 8 5 [ + 1 1 . 2 5 , + 1 9 . 0 2 ]$ </td><td> $+ 0 . 7 4 \bar { [ - 1 . 0 2 , + 2 . 3 9 ] }$ </td><td> $+ 1 . 2 4 \dot { [ + 0 . 2 3 , + 2 . 6 7 ] }$ </td></tr><tr><td>Mistral-7B</td><td> $+ 4 9 . 0 1 \bar { [ + 4 4 . 4 8 , + 5 3 . 6 8 ] }$ </td><td> $+ 6 0 . 8 9 [ + 5 4 . 5 2 , + 6 7 . 0 7 ]$ </td><td> $+ 1 1 . 8 8 [ \dot { + } 8 . 4 3 , + 1 5 . 2 3 ]$ </td><td> $+ 1 0 . 6 4 [ \dot { + } 6 . 9 9 , + 1 4 . 1 8 ]$ </td></tr><tr><td colspan="5">WEB: operational supplied URL units; frozen stance consensus</td></tr><tr><td>Qwen3-8B</td><td> $+ 3 1 . 8 8 [ + 2 8 . 2 6 , + 3 5 . 5 1 ]$ </td><td> $+ 4 0 . 7 6 [ + 3 5 . 4 3 , + 4 6 . 1 7 ]$ </td><td> $+ 8 . 8 8 [ + 4 . 9 6 , + 1 2 . 8 6 ]$ </td><td> $- 2 . 5 4 [ - 5 . 4 0 , + 0 . 1 8 ]$ </td></tr><tr><td>Qwen3-4B</td><td> $+ 3 4 . 0 6 [ + 3 0 . 7 0 , + 3 7 . 4 1 ]$ </td><td> $+ 4 2 . 2 1 [ + 3 7 . 2 3 , + 4 7 . 0 8 ]$ </td><td> $+ 8 . 1 5 [ + 4 . 3 5 , + 1 2 . 0 4 ]$ </td><td> $- 3 . 9 9 \dot { [ - 6 . 8 7 , - 1 . 0 9 ] }$ </td></tr><tr><td>Phi-4-mini</td><td> $+ 9 . 2 4 [ + 6 . 9 3 , + 1 1 . 6 9 ]$ </td><td> $+ 9 . 6 0 [ + 6 . 8 3 , + 1 2 . 5 9 ]$ </td><td> $+ 0 . 3 6 [ - 0 . 5 4 , + 1 . 2 7 ]$ </td><td> $+ 0 . 0 0 [ - 0 . 5 4 , + 0 . 5 4 ]$ </td></tr><tr><td>Mistral-7B</td><td> $+ 3 7 . 1 4 [ + 3 3 . 3 3 , + 4 0 . 7 8 ]$ </td><td> $+ 5 3 . 4 4 [ + 4 8 . 3 3 , + 5 8 . 4 5 ]$ </td><td> $+ 1 6 . 3 0 [ + \bar { 1 } 2 . 4 1 , + 2 0 . 2 9 \bar { ] }$ </td><td> $+ 1 0 . 3 3 [ \dot { + } 7 . 0 6 , + 1 3 . 5 9 \dot { ] }$ </td></tr></table>

Positive values mean more influence from the manipulated side. Certified upstream collapse reproduces BASE byte-for-byte; its recovery is the same paired contrast as Copy–Base.

Table 13: Near-token-balanced sensitivity for Four-Unit–Copy. The subset was frozen from tokenizer counts before model inference. Intervals use the same within-domain dependence-component resampling rule.
<table><tr><td>Domain</td><td>Checkpoint</td><td>n</td><td>Selectivity [95% CI]</td></tr><tr><td>HUMAN</td><td>Qwen3-8B</td><td>57</td><td> $+ 0 . 4 4 [ - 3 . 1 9 , + 4 . 1 7 ]$ </td></tr><tr><td rowspan="4"></td><td>Qwen3-4B</td><td>57</td><td> $+ 6 . 1 4 [ + 1 . 9 6 , + 9 . 6 9 ]$ </td></tr><tr><td>Phi-4-mini</td><td>72</td><td> $+ 0 . 6 9 \bar { [ - 1 . 7 2 , + 2 . 8 7 ] }$ </td></tr><tr><td>Mistral-7B</td><td>50</td><td> $+ 1 1 . 0 0 [ \dot { + } 5 . 7 7 , + 1 5 . 6 9 \dot { ] }$ </td></tr><tr><td>Qwen3-8B</td><td>79</td><td> $+ 9 . 4 9 [ + 4 . 1 7 , + 1 4 . 8 7 ]$ </td></tr><tr><td rowspan="4">WEB</td><td>Qwen3-4B</td><td>79</td><td> $+ 8 . 8 6 \bar { [ + 3 . 1 2 , + 1 4 . 7 4 \bar { ] } }$ </td></tr><tr><td>Phi-4-mini</td><td>89</td><td> $- 0 . 2 \dot { 8 } [ - 1 . 3 7 , + 0 . 5 7 \dot { ] }$ </td></tr><tr><td>Mistral-7B</td><td>57</td><td></td></tr><tr><td></td><td></td><td> $+ 1 4 . 4 7 [ \dot { + } 7 . 8 9 , + 2 0 . 6 1 \dot { ] }$ </td></tr></table>

Appendix Table 12 contains all strict selection contrasts. Across eight separately evaluated model– domain cells, replication invariance is 0/8, Four-Unit–Base sensitivity is 8/8, Four-Unit–Copy selectivity is 4/8, and prompt recovery is 3/8. Model-specific effects and intervals remain the inferential unit; these counts summarize the eight frozen decisions.

The near-token-balanced sensitivity subset uses the pre-inference criterion of a mean absolute Copy–Four-Unit input-token difference bounded by 20 over the four attack-side/order cells. The prespecified gates apply to the full analysis. Full strict, half-tie, attack-probability, likelihood-margin, item-bootstrap, and component-bootstrap outputs remain in the four analysis files; the independent audit recomputes the primary contrasts through a separate implementation.

## D.9 THRESHOLD SENSITIVITY

Table 14: Threshold sensitivity. Cells show passes/eligible model–domain cells. Equivalence uses a 90% CI within $[ - \delta , + \delta ]$ ; positive effects require an estimate of at least δ and a 95% lower bound above zero. The sweep reports the frozen gates across four operational margins.
<table><tr><td>Phase</td><td>Contrast</td><td>2.5 pp</td><td>5 pp</td><td>7.5 pp</td><td>10 pp</td></tr><tr><td>Phase6</td><td>Marker positive effect</td><td>2/4</td><td>0/4</td><td>0/4</td><td>0/4</td></tr><tr><td>Phase6</td><td>Provenance equivalence</td><td>3/4</td><td>3/4</td><td>4/4</td><td>4/4</td></tr><tr><td>Phase6</td><td>Marker-provenance positive effect</td><td>2/4</td><td>0/4</td><td>0/4</td><td>0/4</td></tr><tr><td>Phase6</td><td>Topology-rescue positive effect</td><td>0/4</td><td>0/4</td><td>0/4</td><td>0/4</td></tr><tr><td>Phase8</td><td>Copy-Base equivalence</td><td>0/8</td><td>0/8</td><td>0/8</td><td>0/8</td></tr><tr><td>Phase8</td><td>Four-Unit-Base positive effect</td><td>8/8</td><td>8/8</td><td>8/8</td><td>7/8</td></tr><tr><td>Phase8</td><td>Four-Unit-Copy positive effect</td><td>5/8</td><td>4/8</td><td>4/8</td><td>2/8</td></tr><tr><td>Phase8</td><td>Copy-Prompt positive effect</td><td>3/8</td><td>3/8</td><td>2/8</td><td>2/8</td></tr></table>

Copy–Base lies outside equivalence in all eight cells at every tested margin, so the invariance result is stable across the 2.5–10-point sweep. Four-Unit–Base remains positive and material in eight of eight cells through 7.5 points and seven of eight at 10 points. Selectivity changes from 5/8 at 2.5 points to 2/8 at 10 points, while prompt recovery changes from 3/8 to 2/8. Model-specific effects and intervals are primary; gate counts provide an operational digest.

## D.10 CONTROLLED-GENERATION DIAGNOSTICS

Table 15: Diagnostics for controlled answer-plus-one-sentence generation. Each model–domain row contains 480 generated trials: 40 items by three conditions, two attack sides, and two orders. Invalid is computed over those 480 trials. Agreement uses the same matched trials; all are generation-valid and candidate-untied. The parsed leading answer is the scored endpoint.
<table><tr><td>Checkpoint</td><td>Domain</td><td>Invalid (%)</td><td>Choice agreement (%)</td></tr><tr><td>Qwen3-8B</td><td>HUMAN</td><td>0.00</td><td>65.62</td></tr><tr><td>Qwen3-8B</td><td>WEB</td><td>0.00</td><td>70.83</td></tr><tr><td>Qwen3-14B-AWQ</td><td>HUMAN</td><td>0.00</td><td>63.75</td></tr><tr><td>Qwen3-14B-AWQ</td><td>WEB</td><td>0.00</td><td>74.17</td></tr><tr><td>Phi-4</td><td>HUMAN</td><td>0.00</td><td>65.62</td></tr><tr><td>Phi-4</td><td>WEB</td><td>0.00</td><td>77.08</td></tr></table>

Every formal output has a valid leading answer, making all-trial and valid-output estimates identical. Agreement quantifies consistency between the candidate and generated interfaces, which expose different decision rules while their Copy–Base effects share direction.

## E RUN LINEAGE AND REPRODUCIBILITY

Inference precision is frozen per checkpoint: the original runs use bfloat16, Qwen3-14B uses its official AWQ checkpoint, Phi-4 uses dynamic NF4, and the Qwen3-8B quantization control uses dynamic NF4. Candidate scoring is deterministic, generation is greedy, and each experiment freezes its bootstrap seed before treatment inference. Model-specific chat templates and native assistant termination sequences are fixed during compilation.

Phase 5 completes 1,692/1,692 trials per checkpoint for Qwen3-8B, Qwen3-4B, and Phi. Phase 6 completes 1,692/1,692 per checkpoint for those three and Mistral. Phase 7 completes 2,538/2,538 per checkpoint for the two Qwen models and Phi. Phase 8 completes 3,824/3,824 per checkpoint for all four original models. The extended candidate grid adds 3,824/3,824 trials in each of three inference configurations; controlled generation adds 960/960 in each of three; grouping stress adds 9,792/9,792 in each of four; matched serialization adds 3,264/3,264 in each of four; the controlled GEO panel adds 768/768 in each of four. These totals sum to 104,402 formal evaluation trials across six unique checkpoint identities. Quantized reruns contribute executed trials under their existing checkpoint identities. Phase 11’s 13,056 choices require 26,112 physical candidate forward calls, and Phase 12’s 3,072 choices require 6,144 calls. This implementation accounting is reported separately from the paper’s decision-trial total. The anonymous ZIP’s per-trial output layer spans Phase 6 and Phase 8–12; Phase 5 and Phase 7 are represented by independent internal result audits.

Independent scripts recheck prompt reconstruction, candidate likelihood arithmetic or leading-answer parsing, typed labels, item aggregation, dependence components, and every reported bootstrap statistic. The grouping, added-checkpoint, controlled-generation, matched-serialization, and controlled-GEO audits all have pass status and zero errors. The Phase 12 audit verifies four 768-trial model runs, exact key coverage, and item-first 10,000-replicate bootstrap estimates recomputed from raw predictions. Audit recomputation independently verifies stored outputs and every derived statistic; model inference is represented by the executed forward trials counted above.

The anonymous result-recomputation artifacts use portable paths and identity-scrubbed metadata. The Phase 9–12 tables, auxiliary curve, generated macros, and JSON snapshots are produced after their independent empirical audits pass status, error-count, and trial-accounting checks.

The anonymous result-recomputation artifact retains the verified Phase 6 and Phase 8–11 layers and adds the complete Phase 12 result-verification extension. The ZIP is 64,750,897 bytes; its 302 entries sit under one clean root and contain 285 files. The top manifest covers all 284 non-manifest files. In a fresh extraction, the combined verifier and standard-library recomputation scripts return PASS with zero failures, including all 3,072 Phase 12 predictions, grouping assignments, five rebuilt paper outputs, and an independently recomputed maximum numerical difference of $5 . 3 3 \times 1 0 ^ { - 1 5 }$ . Recursive scans across the 285 text and manifest files report zero host-path, identity, device-identifier, modelweight, cache, bytecode, or nested-ZIP findings. The artifact supports stored-output recomputation; fresh forward execution uses public checkpoints and licensed source texts.