# Structured Claim-Level Discourse Representations for Dense Health Narratives

Farnoushsadat Nilizadeh<sup>1</sup>\* Elham Pourabbas Vafa<sup>2∗</sup> Shirin Nilizadeh<sup>2</sup> Eduard Dragut<sup>1</sup>

<sup>1</sup>Temple University <sup>2</sup>University of Texas at Arlington

{farnoushsadat.nilizadeh, edragut}@temple.edu, {elham.pourabbasvafa, shirin.nilizadeh}@uta.edu

## Abstract

Health discourse in social media videos often contains densely entangled claims spanning multiple thematic aspects, stances, evidential frames, and rhetorical functions within short conversational spans. Existing approaches largely rely on coarse topic-level, sentimentbased, or stance-oriented representations that do not adequately capture this structure. Our analysis identifies an average of 13.22 atomic claims per minute, motivating richer claim level discourse representations. We introduce a structured framework for claim-level discourse analysis in dense health narratives. Our frame work models discourse through tuples linking atomic claims with thematic aspects, stance, and multidimensional pragmatic discourse attributes. To support this setting, we construct a benchmark spanning four health domains with 1,191 manually annotated claims from 60 videos. Using this framework, we evaluate automated structured discourse analysis under different discourse context settings. Results show that current LLMs achieve strong performance on thematic categorization and stance prediction, but struggle with high-dimensional pragmatic profiling. We also find that different discourse tasks benefit from different forms of contextual reasoning, suggesting that future systems may require task decomposition and specialized inference strategies.

## 1 Introduction

Short-form social media videos are major sources of health information, yet health-related discourse in these environments is highly informal, rhetorically heterogeneous, and difficult to analyze computationally (Duan et al., 2025; Fong et al., 2024; Han et al., 2024; Yeung et al., 2025; Basch et al., 2023; Javaid et al., 2024). Existing computational approaches largely rely on coarse topic-level classification, sentiment analysis, or stance detection (Augenstein et al., 2016; Hosseinia et al.,

2020), providing limited support for modeling the internal structure of dense health narratives. In practice, a single conversational segment may contain multiple atomic claims spanning different thematic aspects, conflicting stances, evidential frames, uncertainties, behavioral recommendations, and rhetorical functions. Our analysis identifies an average of 13.22 atomic claims per minute across health-related narratives, revealing a level of information density that conventional discourse representations struggle to capture. We refer to this phenomenon as multi-aspect entanglement: the interleaving of multiple claims, aspects, and communicative intents within short conversational spans.

Prior work in fine-grained sentiment analysis, including aspect-based sentiment analysis (Dragut et al., 2010; Schneider and Dragut, 2015) and aspect-based sentiment intensity analysis (Dragut and Fellbaum, 2014; Wang and Dragut, 2024), demonstrated the value of decomposing opinions into aspect-specific representations rather than assigning document-level labels (Cai et al., 2021; Su et al., 2024; Bai et al., 2024). Dense health narratives, however, involve intertwined claims and pragmatic discourse phenomena that extend beyond sentiment-oriented representations. We argue that dense health narratives require claim-level representations that jointly model atomic claims, thematic aspects, stance, and pragmatic discourse structure. To address this problem, we represent each narrative as a collection of (c, a, s, ⃗τ ) tuples linking an atomic claim c with its thematic aspect a, stance s, and multidimensional typology ⃗τ. This decomposition enables fine-grained computational analysis of dense conversational discourse.

As illustrated in Figure 1, a single short-form health narrative may contain medical eligibility criteria, health outcomes, anecdotal evidence, and persuasive framing within the conversational span. Our representation separates these intertwined components into distinct claim-level structures while

<table><tr><td colspan="2">↓ Aspect-Conditioned Claim Extraction (V → {(c, a, s, τ) } ⁿ) ↓</td><td colspan="2"></td></tr><tr><td>#</td><td>Atomic Claim (c) Aspect (a)</td><td>Stance</td><td>Typology Vector (τ)</td></tr><tr><td>1</td><td>TRT isn&#x27;t for everyone.</td><td>Clinical Diagnosis &amp; Patient Eligibility</td><td>Neu [Non, FaC, Gen, Abs, Tml, Eva]</td></tr><tr><td>2 for TRT.</td><td>You need proper blood tests to qualify Eligibility</td><td>Clinical Diagnosis &amp; Patient Neu</td><td>[Con, Pol, Gen, Abs, Tml, Des]</td></tr><tr><td>3 mood.</td><td>Qualifying for TRT can lead to a better</td><td>Mental &amp; Cognitive Impact</td><td>Pos [Cau, FaC, Gen, Hed, Tml, Des]</td></tr><tr><td>4 blood pressure.</td><td>Taking TRT incorrectly can lead to high Biomarkers</td><td>Medical Health &amp; Neg</td><td>[Cau, FaC, Gen, Hed, Tml, Des]</td></tr><tr><td>5</td><td>Joe&#x27;s been on TRT since his 40s.</td><td>Clinical Diagnosis &amp; Patient Eligibility</td><td>Neu [Non, FaL, Ane, Att, Cur, Nar]</td></tr><tr><td>6 ically.</td><td>Joe says TRT helps him stay strong phys-</td><td>Muscle, Performance &amp; Aesthetics</td><td>Pos [Cau, FaC, Ane, Att, Cur, Nar]</td></tr></table>

Typology Legend (⃗τ): Logic (Non-logical, Conditional, Causal); Verifiability (FaC:Factual(Clinical/Biological), FaL:Factual(Logistical/Contextual), Policy(Prescriptive)); Evidence (General Knowledge/Common Sense, Anecdotal/Personal); Certainty (Absolute/Imperative, Hedged(Probabilistic) , Attributed/Distanced); Temporality (Tml: Timeless, Current/Past); Focus (Evaluative, Descriptive/Definitional, Narrative/Sequential).

Figure 1: An illustration of our formal task: decomposing high-density narratives into structured tuples. This segment demonstrates multi-aspect entanglement and shifting rhetorical profiles, moving from prescriptive medical policy (Row 2) to hedged physiological outcomes (Rows 3–4) and attributed personal narrative (Row 6).

preserving their pragmatic and rhetorical properties. By moving beyond coarse topic- and sentimentlevel labels, the proposed framework enables finegrained modeling of dense health discourse.

To study this problem, we introduce the first benchmark for structured claim-level discourse analysis of dense health narratives. The benchmark spans four health domains: GLP-1 weight-loss medications, Testosterone replacement therapy (TRT), Collagen supplementation, and Intermittent fasting. Constructing the benchmark required extensive expert reconciliation to address the rhetorical ambiguity, implicit framing, and conversational variability characteristic of online health discourse.

Using this benchmark, we evaluate automated structured discourse analysis under varying contextual and inference settings. Our experiments show that LLM-based systems achieve strong performance on thematic and stance classification while degrading substantially on pragmatic discourse modeling. We further find that different discourse tasks require different forms of contextual reasoning: aspect classification benefits from localized semantic context, whereas pragmatic discourse modeling relies more on sequential conversational structure.

Our contributions are as follows:

• We introduce the task of structured claim-level discourse analysis for dense health narratives.

• We propose a multidimensional representation linking atomic claims with thematic aspects, stance, and pragmatic attributes.

• We construct a benchmark spanning four health domains with 1,191 manually annotated claims from 60 videos.

• We evaluate automated structured discourse analysis and study how different contextual settings affect discourse prediction tasks.

## 2 Related Work

Prior work on claim mining and argument analysis has progressively evolved from surface-level claim detection toward richer structured representations (Yang et al., 2020; Habernal and Gurevych, 2017; Lawrence and Reed, 2019; Lippi et al., 2015; Lippi and Torroni, 2016). More recent approaches jointly model claims with stance, argumentative structure, semantic aspects, and rhetorical context rather than treating claims as isolated propositions (Trautmann et al., 2020; Hosseinia et al., 2021; Jo et al., 2020; Plenz et al., 2025; Guo and Vosoughi, 2024a,b). Parallel work in discourse analysis and evidentiality modeling has further introduced pragmatic dimensions such as certainty, evidential grounding, communicative intent, and logical structure into discourse representations (Daxenberger et al., 2017; Boland et al., 2022; Schaefer et al., 2023; Dhole et al., 2025; Visser et al., 2020).

Collectively, these works highlight the importance of modeling semantic, argumentative, and pragmatic structure jointly. However, existing benchmarks typically study these dimensions in isolation or within comparatively constrained discourse settings. Our work instead focuses on dense health narratives where multiple intertwined claims, aspects, evidential strategies, and rhetorical intents coexist within short conversational spans. Appendix 10.1 gives additional related discussion.

## 3 Structured Discourse Representation

In this section, we introduce a claim-level representation framework for dense discourse environments that jointly models atomic claims, thematic aspects, stance, and pragmatic discourse structure. Structured representations play a central role in NLP, including semantic role labeling, frame semantics, discourse parsing, argument mining, and aspectbased sentiment analysis (Lippi and Torroni, 2016; Augenstein et al., 2016). Existing formulations, however, provide limited support for jointly modeling multiple semantic and pragmatic dimensions at the claim level. We therefore represent dense discourse through structured tuples:

$$
\mathcal { T } _ { V } = \{ ( c , a , s , \vec { \phi } ) _ { i } \} _ { i = 1 } ^ { n } , \mathrm { w h e r e }
$$

• c denotes an atomic claim,

$a \in { \mathcal { A } }$ denotes the thematic aspect associated with the claim,

$s \in \{ + , - , 0 \}$ denotes the claim stance,

$\dot { \phi } = ( \phi _ { 1 } , \ldots , \phi _ { 6 } )$ denotes an ordered pragmatic discourse profile, where each component $\phi _ { j } ~ \in ~ { \mathcal { D } } _ { j }$ is drawn from a dimensionspecific categorical domain.

The proposed framework consists of two complementary semantic layers. The first captures domaindependent thematic structure, represented through aspects and stance. The second captures domainstable pragmatic discourse structure, represented through rhetorical and communicative dimensions. This distinction is intuitive in dense discourse environments. For example, Ozempic-related narratives involve aspects such as appetite suppression, gastrointestinal side effects, insurance access, and weight-loss effectiveness, whereas Testosterone Replacement Therapy (TRT) narratives emphasize hormonal regulation, fertility, libido, and physical performance. In contrast, rhetorical structures such as anecdotal evidence, causal reasoning, speculative framing, uncertainty, and persuasive recommendations recur consistently across both domains.

The proposed framework is a flexible claim-level representation rather than a fixed schema. Though developed for health-related social media discourse, the framework may generalize to other domains involving complex narrative structure, including financial advice, political commentary, scientific debate, and legal discourse. We leave broader crossdomain evaluation to future work.

## 3.1 Atomic Claim Decomposition

Given a narrative or transcript segment $V ,$ , our first objective is to decompose the discourse into a set of atomic claims $V \to \{ c _ { i } \} _ { i = 1 } ^ { n }$ , where each $c _ { i }$ denotes a self-contained propositional unit expressing a single coherent assertion. This decomposition separates compound conversational structures into individually analyzable claim units. For example, the statement: “Ozempic helped me lose weight, but it made me constantly nauseous,” contains at least two distinct claims involving different aspects and opposing stances. Treating such discourse as a single unit obscures the distinction between therapeutic benefits and adverse effects.

To address this problem, we decompose complex statements into minimal propositional units while preserving their semantic interpretation. We resolve ambiguous references following principles of anaphora grounding (Lee et al., 2017), replacing pronouns and context-dependent expressions with explicit referents when necessary. The resulting atomic claims form the basic units for downstream analysis.

## 3.2 Thematic Aspect Modeling

Each atomic claim is associated with a thematic aspect $a \in { \mathcal { A } }$ representing the primary semantic category addressed by the claim. Aspect assignments operate at the claim level, allowing different propositions within the same narrative segment to be mapped to distinct semantic categories. The aspect inventory A is inherently domain-dependent and reflects the semantic structure of a particular narrative environment. Consequently, aspects are modeled as flexible semantic categories rather than fixed universal labels.

Our formulation is conceptually related to aspectbased sentiment analysis, where opinions are decomposed according to thematic targets rather than treated as document-level judgments (Wang et al., 2026). Dense narrative settings, however, require modeling substantially richer semantic and pragmatic structure.

## 3.3 Pragmatic Discourse Instantiation

While aspects capture thematic content, they provide limited support for modeling rhetorical and pragmatic discourse structure. To address this limitation, we associate each claim with a multidimensional typology vector:

$$
\Vec { \phi } = [ l _ { 1 } , l _ { 2 } , \ldots , l _ { 6 } ]
$$

where each dimension characterizes a distinct discourse property.

The proposed typology is synthesized from prior work in discourse analysis, argumentation theory, evidentiality, stance modeling, uncertainty detection, and pragmatic classification (Lippi and Torroni, 2016; Augenstein et al., 2016). The difficulty of defining stable fine-grained intent taxonomies has also been observed in other structured languageanalysis settings (Lan et al., 2025). During development, we evaluated a broader inventory of candidate dimensions and retained only those that consistently appeared across multiple discourse domains and could be reliably annotated under expert reconciliation. This selection reflects a balance between expressiveness and annotatability rather than a claim that the six dimensions are exhaustive of all possible pragmatic phenomena in health discourse. The resulting framework should be viewed as an extensible discourse representation rather than a closed label schema.

Table 1 summarizes the proposed typology, with detailed definitions provided in Table 10 in Appendix 10.2. The typology models six complementary dimensions: Logical Relationship: the structural relationship expressed by the claim, e.g., causal, conditional, or comparative reasoning. Verifiability and Intent: whether the claim expresses factual, policy-oriented, subjective, or speculative content. Evidence Basis: the implied source of authority underlying the claim, e.g., anecdotal experience, scientific evidence, expert opinion, or general knowledge. Certainty: the degree of confidence or hedging expressed by the speaker. Temporality: the temporal orientation of the claim, e.g., timeless principle, current experience, or future prediction. Claim Focus: the primary communicative role of the statement, e.g., explanatory, descriptive, evaluative, or narrative. The proposed dimensions enable fine-grained modeling of rhetorical and pragmatic structure at the claim level.

For example, in Figure 1, the statement “Joe says TRT helps him stay strong physically” is represented as an atomic claim with thematic aspect Muscle, Performance & Aesthetics, positive stance, and a pragmatic profile consisting of causal reasoning, factual(clinical) verifiability, anecdotal evidence, attributed certainty, past temporality, and narrative focus. In tuple form:

Table 1: Six-dimensional pragmatic discourse typology.
<table><tr><td>Dimension</td><td>Representative Labels</td></tr><tr><td>Logical Relationship</td><td>Causal, Comparative, Conditional</td></tr><tr><td>Verifiability &amp; Intent</td><td>Factual, Subjective, Prescriptive</td></tr><tr><td>Evidence Basis</td><td>Scientific, Anecdotal, Commercial</td></tr><tr><td>Certainty</td><td>Absolute, Hedged, Attributed</td></tr><tr><td>Temporality</td><td>Past, Predictive, Timeless</td></tr><tr><td>Claim Focus</td><td>Explanatory, Narrative, Promotional</td></tr></table>

$$
\begin{array} { r } { ( c , a , s , \vec { \phi } ) = ( \tt T R T \ h e l p s \ h i m \ s t a y \ s t r o n g \ p h y s i c a l l y , } \\ { \tt M u s c l e / P e r f o r m a n c e , P o s i t i v e , } \\ { ( c a u s a l , F a c t u a l ( C l i n i c a l ) , A n e c d o t a l , } \\ { \tt A t t r i b u t e d , P a s t , N a r r a t i v e ) ) \quad \tt } \end{array}
$$

To illustrate why simpler structures such as aspect-plus-sentiment or stance-only labels are insufficient for dense health narratives, consider the paired claims from the TRT domain in Table 2. While both share the identical thematic aspect (Fertility & HPTA) and the same negative stance (Neg), they diverge fundamentally across every pragmatic axis: A traditional aspect-plus-sentiment or stanceonly representation completely collapses these critical rhetorical variations, treating both assertions as identical. Our 6-axis typology is what preserves these vital distinctions, which are necessary for downstream tasks like misinformation detection and evidence quality assessment.

A broader qualitative analysis of multi-aspect entanglement and intra-speaker rhetorical trajectories across domains is provided in Appendix 10.3.

## 4 Benchmark Construction

We now describe the benchmark construction process for the proposed framework.

## 4.1 Domain Selection and Video Collection

We selected four health domains: GLP-1 weightloss medications, Testosterone Replacement Therapy (TRT), Collagen supplementation, and Intermittent fasting. These domains capture diverse forms of online health discourse, including personal narratives, lifestyle advice, clinical explanations, commercial promotion, and policy-oriented discussion.

Videos were collected using the YouTube Data API v3 (Google, 2026). All searches were constrained to a fixed collection window spanning January 1, 2024 through August 30, 2025. Initial seed queries, e.g., “Ozempic journey” and “TRT side effects,” were manually audited for relevance and expanded into broader topic-specific query sets. Using these queries, we collected an initial pool of approximately 2,000 unique video IDs per topic.

Table 2: Qualitative comparison of two atomic claims from the TRT domain sharing identical thematic aspect and stance labels yet diverging across all six pragmatic discourse dimensions, illustrating the limitation of coarse aspect-plus-sentiment or stance-only baselines.
<table><tr><td>Claim</td><td>Aspect</td><td></td><td>Stance Logical</td><td>Verif.</td><td>Evidence</td><td>Certainty</td><td>Temp.</td><td>Focus</td></tr><tr><td>On average, testicular function drops approximately 50–75%.</td><td>Fertility HPTA</td><td>&amp; Neg</td><td>Causal</td><td>Factual (Clin.)</td><td>Sci. (Stat.)</td><td>Absolute</td><td>Timeless</td><td>Descriptive</td></tr><tr><td>The speaker acknowledges that 50–75% decrease is dramatic.</td><td>Fertility HPTA</td><td>&amp; Neg</td><td></td><td>Non_Log Value (Subj.)</td><td>Anecdotal</td><td></td><td>Attributed Current / Evaluative Past</td><td></td></tr></table>

## 4.2 Corpus Filtering and Sampling

To move from the high-recall pool to a highprecision evaluation set, we implemented a multistage selection process. First, we extracted a randomized subset of 100 candidate videos per topic to establish a manageable working pool for intensive processing. These candidates were processed through the multimedia pipeline described in Sec tion 4.3 to generate denoised transcripts. Next, an automated triage layer was applied to the transcripts using an LLM-based screening prompt (Appendix 10.4). This stage filtered videos according to three criteria: (i) English-language content, (ii) presence of meaningful propositional speech, and (iii) thematic relevance to the target health domain. The filtering stage reduced the candidate pool to approximately 50 high-quality videos per topic. From this pool, we performed a final purposive selection of 15 videos per topic (N = 60 total). The selection process additionally prioritized shorter, claimdense videos to maximize rhetorical diversity and avoid over-representation of a single discourse type. Across the 60 selected videos, individual durations range from 15 seconds to 6 minutes 50 seconds (mean = 90.1 s, SD = 83.7 s), reflecting a corpus of short-to-medium-length content. Consequently, the final corpus includes a rich cross-section of informational, promotional, prescriptive, and narrative categories. The distribution of these discourse types across the 60 videos is presented in Table 3.

## 4.3 Transcript Processing

Our multimedia pipeline processed the videos through a series of automated transcription and stabilization layers. Audio streams were extracted using yt-dlp (yt-dlp Contributors, 2026) and transcribed using OpenAI’s Whisper model (Radford et al., 2023). yt-dlp handles network retrieval and audio extraction from the YouTube platform, while Whisper performs the speech-to-text transcription on the resulting audio stream.

Table 3: Distribution of video discourse categories across the curated Ground Truth datasets (N = 60).
<table><tr><td>Topic</td><td>Informa- tional</td><td>Promo- tional</td><td>Prescrip- tive</td><td>Narrative</td></tr><tr><td>Ozempic</td><td>4</td><td>1</td><td>4</td><td>6</td></tr><tr><td>TRT</td><td>7</td><td>3</td><td>3</td><td>2</td></tr><tr><td>Collagen</td><td>5</td><td>7</td><td>2</td><td>1</td></tr><tr><td>Fasting</td><td>5</td><td>1</td><td>7</td><td>2</td></tr><tr><td>Total</td><td>21</td><td>12</td><td>16</td><td>11</td></tr></table>

The raw automatic speech recognition (ASR) output frequently contained malformed medical terminology, fragmented sentence boundaries, and missing punctuation (Appendix 10.5). To address these issues, we introduced an Orthographic Denoising and Stabilization layer using Gemini 2.5 Flash (Comanici et al., 2025) (Appendix 10.6) as a post-processing step over the raw Whisper transcripts. The denoising stage performed two complementary corrections: (i) spelling normalization for ASR-induced errors in drug names, biomarkers, and domain-specific medical terminology, and (ii) orthographic stabilization correcting fragmented sentence boundaries, capitalization, and missing punctuation. The prompt enforced conservative editing constraints that prohibited content deletion, summarization, or insertion, preserving the speaker’s original lexical content and propositional meaning. Examples in Appendix 10.5 illustrate consistent recovery of degraded terminology and sentence structure.

## 4.4 Gold Standard Annotation

To construct the gold-standard benchmark, two annotators with backgrounds in Natural Language Processing (NLP) and computational discourse analysis carried out a multi-stage annotation protocol over the evaluation set. The resulting benchmark required around 400 person-hours of annotation and reconciliation effort (Table 16 in $\mathsf { A p - }$ pendix 10.7).

Table 4: Descriptive statistics of the curated Ground Truth (GT) datasets $( N = 6 0 )$ .
<table><tr><td>Topic</td><td>Videos</td><td>Duration Stmts</td><td>Claims</td><td>s Density 单</td></tr><tr><td>Ozempic</td><td>15</td><td>35m 54s 166</td><td>292</td><td>8.13</td></tr><tr><td>TRT</td><td>15</td><td>19m 12s 185</td><td>331</td><td>17.24</td></tr><tr><td>Collagen</td><td>15</td><td>15m 28s 126</td><td>237</td><td>15.32</td></tr><tr><td>Fasting</td><td>15</td><td>19m 30s 175</td><td>331</td><td>16.97</td></tr><tr><td>Total</td><td>60</td><td>90m 4s 652</td><td>1,191</td><td>13.22</td></tr></table>

<sup>∗</sup>Density measured as Atomic Claims per minute of video content.

The annotation process consists of four stages. First, annotators identify discourse segments relevant to the target health domain. Second, extracted statements are decomposed into atomic claims using the principles described in Section 3. Third, annotators assign thematic aspects and stance labels to each claim. Finally, each claim is annotated across the six-dimensional pragmatic typology introduced in Section 3. The process requires substantial reconciliation due to implicit causality, anecdotal framing, speculative language, and overlapping communicative intent.

Table 4 summarizes descriptive statistics of the resulting benchmark. Across the four domains, the corpus contains 1,191 atomic claims extracted from approximately 90 minutes of discourse, corresponding to an average density of 13.22 atomic claims per minute. We further observe substantial variation in claim density across domains. While narrative-oriented Ozempic videos show lower density, content in the TRT, Fasting, and Collagen domains is highly condensed, peaking at 17.24.

## 4.5 Annotation Reliability

To validate benchmark robustness, we compute reliability metrics across all annotation layers. Because open-ended segment extraction lacks clearly defined true negatives, we evaluate agreement for the initial statement extraction and atomic decomposition stages using pairwise $F _ { 1 }$ -scores and macro-averaged agreement measures rather than chance-corrected coefficients, following prior work on argument mining and span-based discourse annotation (Mochales and Moens, 2011; Habernal and Gurevych, 2017; Artstein and Poesio, 2008).

For statement extraction, annotators achieved a mean pairwise $F _ { 1 }$ of 0.93 across the 60-video corpus. For atomic decomposition, the overall macro-averaged agreement reached 92%, indicating strong consistency in identifying claim boundaries despite the density and rhetorical complexity of the discourse. For the multi-axis classification tasks, inter-annotator agreement was evaluated using both Cohen’s κ and raw percentage agreement (Cohen, 1960; Artstein and Poesio, 2008). As summarized in Table 5 (see Table 17 for details), aspect assignment achieved a mean raw agreement of 85% (mean $\kappa = 0 . 8 3 )$ , reflecting the stability of the induced aspect taxonomies across domains.

Table 5: Inter-annotator agreement across layers.
<table><tr><td>Annotation Layer</td><td>Mean κ / Agreement</td></tr><tr><td>Statement Extraction</td><td>Pairwise  $\overline { { F _ { 1 } = 0 . 9 3 } }$ </td></tr><tr><td>Atomic Decomposition</td><td> $\mathrm { A g r e e m e n t } = 0 . 9 2$ </td></tr><tr><td>Aspect Assignment</td><td> $\kappa = 0 . 8 3 / \mathrm { A g r e e m e n t } = 0 . 8 5$ </td></tr><tr><td>Stance Classification</td><td> $\kappa = 0 . 8 0 / \mathrm { A g r e e m e n t } = 0 . 8 7$ </td></tr><tr><td>6-Axis Typology</td><td> $\kappa = 0 . 7 6 / \mathrm { A g r e e m e n t } = 0 . 8 8$ </td></tr></table>

Agreement across the six-dimensional typology remained consistently substantial. Logical Relationship achieved the highest structural alignment (Mean raw agreement of 90.4%; Mean $\kappa = 0 . 8 5 )$ while Certainty reached 92.9% raw agreement. The comparatively lower agreement observed for Claim Focus (Mean $\kappa = 0 . 6 8 )$ reflects the inherent difficulty of distinguishing explanatory, descriptive, and narrative intent in spontaneous conversational discourse. All remaining annotation conflicts were resolved through structured reconciliation sessions following standard adjudication practices in discourse annotation (Artstein and Poesio, 2008).

## 5 Structured Discourse Prediction

We now investigate whether large language models (LLMs) can recover structured discourse tuples $( c , a , s , { \vec { \phi } } )$ from dense health narratives.

## 5.1 Task Formulation

Given an atomic claim c and a contextual setting Γ, the objective is to predict its thematic aspect a, stance s, and pragmatic discourse profile $\vec { \phi } \colon$

$$
f _ { \theta } ( c , \Gamma )  ( a , s , \vec { \phi } ) ,
$$

where $f _ { \theta }$ denotes an LLM-based structured discourse prediction function.

We investigate three research questions:

RQ1: Can LLMs reliably recover structured discourse tuples from dense health narratives? RQ2: How do different forms of discourse context affect discourse prediction tasks?

RQ3: Which discourse dimensions generalize most consistently across domains?

We evaluated six inference settings that varied the amount and structure of surrounding discourse available to the model: Atomic Claim, where the model receives only the isolated target claim with no surrounding context; Batch, where claims are processed in fixed-size chronological windows of eight claims per API call, following transcript order (these windows reflect narrative co-occurrence rather than semantic retrieval); Local ±2, where only the two claims immediately preceding and following the target claim are provided; Narrative, where a higher-level narrative-arc summary of the surrounding discourse is provided; Transcript, where the full video transcript is provided as context; and Summary, where a compressed summary of the transcript is provided in place of the raw text. All results in Sections 6.1 and 6.2 use Gemini 2.5 Flash unless noted.

Our central hypothesis is that different discourse tasks require different forms of contextual reasoning. In particular, aspect assignment depends on localized semantic contrasts, whereas pragmatic discourse interpretation relies on broader conversational structure.

## 5.2 Aspect Taxonomy Instantiation

Because thematic aspects are domain-dependent, we adopt an inductive aspect construction strategy rather than imposing a fixed predefined schema. Following prior work on iterative discourse schema development and aspect induction (Jo et al., 2020; Guo and Vosoughi, 2024b; Plenz et al., 2025), we analyze representative pilot transcripts from each topic domain and use LLM-assisted claim decomposition on these transcripts to surface recurring claim targets within the domain, without constraining the output to a fixed label set. Candidate categories arising from this process are then refined through expert reconciliation, in which annotators merge, discard, or sharpen category boundaries to arrive at a finalized taxonomy for each domain. The induced taxonomies vary substantially across domains. For example, Ozempic discourse emphasizes appetite suppression and gastrointestinal side effects, whereas TRT discussions focus more on hormonal regulation, fertility, and physical performance. To support consistency across videos, aspect categories are defined at the domain level rather than tailored to individual narratives.

The finalized taxonomies are further validated using boundary cases containing semantically ambiguous or overlapping claims. Across the 1,191 annotated claims, only five could not be mapped cleanly to an existing aspect category, indicating broad coverage of the induced semantic categories. Appendix 10.8 gives the aspect taxonomies for all four domains in Table 18.

## 5.3 Automated Labeling Setup

To scale annotation beyond the manually curated benchmark, we use an automated labeling framework based on Gemini 2.5 Flash. All generation and evaluation pipelines use deterministic greedy decoding $( \tau = 0 . 0 )$ and structured JSON outputs. The prompt templates for both aspect/stance classification and pragmatic typology prediction are in Appendix 10.9; domain-invariant axis definitions follow Table 10, while domain-specific aspect taxonomies follow Table 18. We use two separate prompts for these tasks. The first prompt handles joint aspect and stance prediction, receiving the domain-specific aspect schema together with category definitions and representative keywords. The second prompt handles the six-dimensional pragmatic typology. To improve robustness under ambiguous discourse conditions, both prompts require a brief intermediate justification before final predictions. For pragmatic discourse classification, the model additionally receives surrounding transcript context to help resolve ambiguity across pragmatic dimensions.

Table 6: Core pipeline constants.
<table><tr><td>Constant</td><td>Value</td></tr><tr><td>Backbone model (primary) Decoding strategy Temperature (primary runs) Output format Batch window size Local context window Few-shot examples/domain Robustness temperatures Robustness trials/setting</td><td>Gemini 2.5 Flash Greedy τ = 0.0 Structured JSON 8 claims/call ±2 claims 5  $\tau \in \{ 0 . 0 , 0 . 2 , 0 . 5 , 0 . 8 \}$  3 (12 runs total)</td></tr></table>

Our objective is not to optimize specialized discourse architectures, but to establish a robust benchmark and evaluation setting for structured discourse analysis. More advanced inference frameworks, including agentic and multi-stage reasoning systems, remain important directions for future work. Table 6 summarizes the core pipeline constants used across all primary evaluations.

## 6 Experiments

We evaluate the framework in three settings: (1) aspect and stance classification, (2) multidimensional pragmatic profiling using the 6-axis typology, and (3) cross-model generalization across proprietary and open-weight LLMs. All evaluations utilize deterministic greedy decoding $( \tau = 0 . 0 )$ with fixed prompts and structured JSON outputs.

## 6.1 Aspect and Stance Classification

We first evaluate thematic aspect assignment and stance classification under different contextual settings. We benchmarked five contextual configurations ranging from isolated atomic claims to full transcripts, where the Batch setting processes fixedsize chronological windows of 8 claims per API call based on narrative order, and the Local context setting provides the two immediately neighboring claims around a single target claim. Results are summarized in Table 7. Batch-based inference achieved the strongest overall performance, outperforming both isolated claim classification and broader contextual settings. In contrast, fulltranscript inference consistently reduced performance, suggesting that excessive conversational context introduces semantic noise.

We further evaluated prompt refinement strategies using the batch configuration. Adding category definitions and representative keywords substantially improved aspect assignment performance, while lightweight chain-of-thought prompting produced additional gains for stance classification. The final configuration (Exp. Batch + CoT) achieved 79.23% aspect accuracy $( \kappa = 0 . 7 7 )$ and 91.55% stance accuracy $( \kappa = 0 . 8 7 )$ on the Ozempic dataset. These findings suggest that thematic discourse tasks benefit from localized semantic context and lightweight reasoning support, whereas excessive discourse context may dilute the model’s focus on the target claim.

## 6.2 Pragmatic Typology Classification

Next, we evaluate automated classification of the six-dimensional pragmatic discourse typology . Unlike thematic aspect labeling, pragmatic profiling requires modeling rhetorical and structural properties such as evidential framing, temporality, and communicative intent.

We evaluate multiple discourse context settings ranging from isolated claims to full transcripts and compressed summaries. Results are shown on the left side of Table 8. In contrast to thematic aspect assignment, broader contextual windows do not consistently improve performance. Full-transcript and narrative-level contexts reduce performance, particularly for structurally sensitive dimensions such as Evidence Basis and Claim Focus. Instead, the isolated claim setting (Atomic) emerged as the strongest baseline context (72.70%, κ = 0.57).

Table 7: Context and prompting ablations for aspect and stance classification using Gemini 2.5 Flash. Each cell reports Acc. / κ.
<table><tr><td>Configuration of Experiment</td><td>Aspect</td><td>Stance</td></tr><tr><td>Atomic Claim Batch</td><td>69.71 / .66 73.72 / .70</td><td>85.04 / .77 90.51 / .85</td></tr><tr><td>Local (±2) Narrative Transcript</td><td>68.98 / .65 71.90 /.68 69.34 / .65</td><td>85.04 / .76 85.04 / .76 83.58 / .74</td></tr><tr><td></td><td></td><td></td></tr><tr><td>Batch + Defs. &amp; Keywords  $\mathbf { B a t c h + C o T }$ </td><td>79.16/ .77 79.23 / .77 91.55 / .87</td><td>89.05 / .83</td></tr></table>

We refined prompting using heuristic discourse constraints from annotation reconciliation sessions acting as a proof of concept for rule-augmented pragmatic classification. Under this configuration, the local-context setting paired with these heuristics achieved the strongest overall performance, reaching 86.44% mean accuracy $( \kappa = 0 . 7 6 )$

Finally, we evaluated explicit chain-of-thought (CoT) prompting with the optimized typology classifier, as shown in the right-hand portion of Table 8. Unlike aspect classification, CoT prompting substantially degraded performance across most pragmatic dimensions, particularly Claim Focus and Evidence Basis, dropping the mean accuracy to 73.12% ( see Appendix 10.10 for a qualitative error analysis of these CoT failure modes). Table 21 in Appendix 10.11 outlines the full heuristic optimization across all other context windows.

These results show a distinction between semantic categorization and pragmatic discourse modeling. While lightweight reasoning improved thematic classification, explicit CoT prompting harmed high-dimensional pragmatic profiling. This suggests that pragmatic interpretation relies more on implicit rhetorical cues and localized conversational structure than on explicit reasoning chains.

To evaluate stability, we reran the TRT pipeline across temperatures $\tau \in \{ 0 . 0 , 0 . 2 , 0 . 5 , 0 . 8 \}$ (3 trials each), demonstrating robust empirical consistency across runs alongside minor, task-specific temperature sensitivities (Appendix 10.12).

Table 8: Impact of discourse context and chain-of-thought (CoT) prompting on 6-axis typology classification using Gemini 2.5 Flash. The left portion of the table evaluates different contextual settings ranging from isolated atomic claims to broader conversational contexts. The right portion reports the effect of heuristic prompting and explicit CoT reasoning on the optimized pragmatic discourse classifier. Each cell reports Accuracy and Cohen’s κ.
<table><tr><td rowspan="2">Axis</td><td colspan="2">Atomic</td><td colspan="2">Batch</td><td colspan="2">Local ±2</td><td colspan="2">Narr.</td><td colspan="2">Trans.</td><td colspan="2">Summ.</td><td colspan="2">Heuristic</td><td colspan="2">+ CoT</td></tr><tr><td>Acc.</td><td>κ</td><td>Acc.</td><td>κ</td><td>Acc.</td><td>κ</td><td>Acc.</td><td>κ</td><td>Acc.</td><td>κ</td><td>Acc.</td><td>κ</td><td>Acc.</td><td>κ</td><td>Acc.</td><td>κ</td></tr><tr><td>I. Logical Rel.</td><td>55.91</td><td>0.31</td><td>57.71</td><td>0.38</td><td>55.56</td><td>0.34</td><td>54.84</td><td>0.33</td><td>57.71</td><td>0.33</td><td>57.35</td><td>0.33</td><td>84.23</td><td>0.74</td><td>80.29</td><td>0.68</td></tr><tr><td>II. Verifiability</td><td>82.80</td><td>0.69</td><td>81.72</td><td>0.68</td><td>84.59</td><td>0.73</td><td>84.95</td><td>0.74</td><td>81.00</td><td>0.66</td><td>82.08</td><td>0.69</td><td>86.38</td><td>0.76</td><td>82.08</td><td>0.69</td></tr><tr><td>III. Evidence</td><td>79.57</td><td>0.68</td><td>62.37</td><td>0.50</td><td>77.78</td><td>0.65</td><td>58.78</td><td>0.46</td><td>44.44</td><td>0.31</td><td>53.76</td><td>0.40</td><td>84.23</td><td>0.74</td><td>61.65</td><td>0.48</td></tr><tr><td>IV. Certainty</td><td>93.91</td><td>0.83</td><td>92.47</td><td>0.80</td><td>91.40</td><td>0.77</td><td>90.32</td><td>0.75</td><td>86.38</td><td>0.66</td><td>92.11</td><td>0.79</td><td>90.32</td><td>0.75</td><td>91.04</td><td>0.76</td></tr><tr><td>V. Temporality</td><td>71.33</td><td>0.52</td><td>63.80</td><td>0.42</td><td>62.72</td><td>0.41</td><td>66.67</td><td>0.46</td><td>60.93</td><td>0.37</td><td>68.82</td><td>0.49</td><td>91.40</td><td>0.84</td><td>74.19</td><td>0.57</td></tr><tr><td>VI. Claim Focus</td><td>52.69</td><td>0.35</td><td>53.05</td><td>0.38</td><td>53.76</td><td>0.37</td><td>51.61</td><td>0.36</td><td>46.24</td><td>0.26</td><td>41.22</td><td>0.26</td><td>82.08</td><td>0.74</td><td>49.46</td><td>0.34</td></tr><tr><td>Mean</td><td>72.70</td><td>0.57</td><td>68.52</td><td>0.53</td><td>70.97</td><td>0.54</td><td>67.86</td><td>0.52</td><td>62.78</td><td>0.43</td><td>65.89</td><td>0.49</td><td>86.44</td><td>0.76</td><td>73.12</td><td>0.59</td></tr></table>

## 6.3 Cross-Model Generalization

To evaluate the robustness of the framework across backbone models, we instantiated the identical inference pipeline using three additional models: Llama 3.3 (Grattafiori et al., 2024), Qwen 3.5 35B (Qwen Team, 2026), and Gpt-oss 20B (Agarwal et al., 2025). All models were evaluated under identical prompting and contextual settings.

Table 9 displays Gemini 2.5 Flash and the top open-source follower, Qwen 3.5 35B; four-model results are in Appendix 10.13 (Table 23) Across all domains, Gemini 2.5 Flash achieved the strongest overall performance, particularly on the 6-axis typology task. However, open-weight models remained competitive across aspect and stance classification, with Qwen 3.5 35B consistently approaching proprietary-model performance.

The largest performance divergence emerged in the pragmatic typology task, where smaller openweight models degraded substantially under the high-dimensional classification setting. This pattern aligns with our ablation findings and suggests that structured pragmatic discourse analysis imposes greater reasoning and instruction-following demands than thematic categorization alone.

Table 9: Cross-model eval. Each cell reports Acc./κ.
<table><tr><td>Model</td><td>Domain</td><td>Aspect</td><td>Stance</td><td>6-Axis</td></tr><tr><td>Gemini 2.5 Flash</td><td>Ozemp. TRT Collag. Fast.</td><td>79.23 / .77 89.12 / .88 86.44 / .85 77.64 / .75</td><td>91.55 / .87 92.15 / .88 85.59 / .77 83.38 / .73</td><td>86.15 / .76 88.62 / .77 83.97 / .72 82.23 / .59</td></tr><tr><td>Qwen 3.5 35B</td><td>Ozemp. TRT Collag. Fast.</td><td>77.54 /.75 79.00 / .77 81.36 / .79 78.25 / .76</td><td>85.26 / .78 83.30 / .74 78.39 / .65 86.71 / .79</td><td>81.58 / .70 82.05 / .62 81.43 / .68 80.72 / .58</td></tr></table>

These results suggest that pragmatic discourse

modeling remains substantially more difficult than thematic categorization and stance prediction. Performance degrades under broader contextual settings and high-dimensional pragmatic inference, indicating important limitations in current monolithic prompting approaches. The divergent behavior observed across tasks further suggests that future systems may benefit from task decomposition and specialized inference strategies.

## 7 Conclusion

We introduced the task of structured claim-level discourse analysis for dense health narratives and presented a benchmark for modeling thematic, stance, and pragmatic discourse structure in conversational health content. These domain choices and taxonomies reflect the health-discourse setting studied here and are not intended as a fixed or universal schema. Beyond this specific setting, our experiments showed that different discourse tasks require substantially different forms of contextual reasoning. While current LLMs perform strongly on thematic categorization and stance prediction, performance degrades considerably for high-dimensional pragmatic discourse profiling. These findings suggest that future discourse systems may benefit from task decomposition and specialized inference strategies rather than relying on a single monolithic prompting configuration. While developed for health discourse, the underlying representation is not inherently domain-specific and may extend to other dense narrative settings, such as financial advice, political commentary, or legal discourse; we leave full cross-domain validation to future work. We hope this benchmark encourages further research on structured discourse understanding in complex narrative environments.

## 8 Limitations

While our framework and benchmark establish a rigorous foundation for fine-grained claim modeling, several limitations must be acknowledged.

Our study focuses on English-language health discourse collected from YouTube short-form videos within four health domains. Although these domains capture diverse forms of online health communication, the benchmark does not cover the full range of medical topics, platforms, or cultural communication styles present in broader social media ecosystems. In addition, the benchmark uses transcripts as the primary unit of analysis, abstracting away multimodal dimensions of the original videos such as visual demonstrations, on-screen text, and creator affect. In health communication settings, these non-verbal signals can shape claim interpretation, evidential framing, and persuasive intent in ways that transcript-only representations cannot capture. Future work may benefit from integrating visual and paralinguistic signals into the proposed discourse framework. Our best-performing pragmatic typology results rely on heuristic discourse constraints manually derived from annotation reconciliation sessions(Section 6.2). These heuristics serve as a proof-of-concept that rule-augmented prompting can substantially improve high-dimensional pragmatic classification, but they were constructed specifically for this benchmark’s domains and typology, and their transferability to new domains or label schemas has not been tested. We leave the development of learnable, data-driven constraints that could generalize automatically across domains to future work.

The proposed representation framework focuses on claim-level semantic and pragmatic structure within single-speaker narrative discourse rather than multi-speaker conversational interaction. Consequently, the benchmark does not explicitly model dialogue phenomena such as turn-taking, speaker coordination, or cross-speaker stance dynamics.

Ethics Statement This benchmark is constructed from publicly available YouTube videos collected via the YouTube Data API v3 for noncommercial research purposes. No private user data was collected. The dataset contains healthrelated claims sourced from social media that may be medically inaccurate, speculative, or commercially motivated; we do not endorse any claims present in the corpus. Models trained or evaluated on this benchmark should not be deployed in clinical or public health settings without appropriate expert oversight. GPT-5 was used solely for grammar and proofreading assistance on the manuscript. All research uses of large language models are described in Sections 4 and 5.

## 9 Acknowledgments

This work was supported in part by the U.S. NSF under awards 2107213, 2026513, and 2309318. We thank Jacob Meltzer for their valuable effort in data preparation and annotation.

## References

Sandhini Agarwal, Lama Ahmad, Jason Ai, Sam Altman, Andy Applebaum, Edwin Arbus, Rahul K Arora, Yu Bai, Bowen Baker, Haiming Bao, and 1 others. 2025. gpt-oss-120b & gpt-oss-20b model card. arXiv preprint arXiv:2508.10925.

Ron Artstein and Massimo Poesio. 2008. Survey article: Inter-coder agreement for computational linguistics. Computational linguistics, 34(4):555–596.

Isabelle Augenstein, Tim Rocktäschel, Andreas Vlachos, and Kalina Bontcheva. 2016. Stance detection with bidirectional conditional encoding. In EMNLP, pages 876–885.

Yinhao Bai, Zhixin Han, Yuhua Zhao, Hang Gao, Zhuowei Zhang, Xunzhi Wang, and Mengting Hu. 2024. Is compound aspect-based sentiment analysis addressed by LLMs? In Findings of EMNLP, pages 7836–7861.

Corey H Basch, Sandhya Narayanan, Hao Tang, Joseph Fera, and Charles E Basch. 2023. Descriptive analysis of tiktok videos posted under the hashtag# ozempic. Journal ofMedicine, Surgery, and Public Health, 1:100013.

Katarina Boland, Pavlos Fafalios, Andon Tchechmedjiev, Stefan Dietze, and Konstantin Todorov. 2022. Beyond facts–a survey and conceptualisation of claims in online discourse analysis. Semantic Web, 13(5):793–827.

Hongjie Cai, Rui Xia, and Jianfei Yu. 2021. Aspectcategory-opinion-sentiment quadruple extraction with implicit aspects and opinions. In ACL-IJCNLP, pages 340–350.

Jacob Cohen. 1960. A coefficient of agreement for nominal scales. Educational and psychological measurement, 20(1):37–46.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, and 1 others. 2025. Gemini 2.5: Pushing the frontier with

advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261.

Johannes Daxenberger, Steffen Eger, Ivan Habernal, Christian Stab, and Iryna Gurevych. 2017. What is the essence of a claim? cross-domain claim identification. In Proceedings ofthe 2017 Conference on Empirical Methods in Natural Language Processing, pages 2055–2066.

Gianluca Detommaso, Martin Bertran, Riccardo Fogliato, and Aaron Roth. 2024. Multicalibration for confidence scoring in llms. arXiv preprint arXiv:2404.04689.

Kaustubh Dhole, Kai Shu, and Eugene Agichtein. 2025. Conqret: A new benchmark for fine-grained automatic evaluation of retrieval augmented computational argumentation. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 5687–5713.

Eduard Dragut and Christiane Fellbaum. 2014. The role of adverbs in sentiment analysis. In Frame Semantics in NLP, pages 38–41.

Eduard Dragut, Yunyao Li, Lucian Popa, and Slobodan Vucetic. 2021. Data science with human in the loop. In KDD, pages 4123–4124.

Eduard C. Dragut, Clement T. Yu, A. Prasad Sistla, and Weiyi Meng. 2010. Construction of a sentimental word dictionary. In CIKM, pages 1761–1764.

Zhijie Duan, Kai Wei, Zhaoqian Xue, Jiayan Zhou, Shu Yang, Siyuan Ma, Jin Jin, and Lingyao Li. 2025. Crowdsourcing-based knowledge graph construction for drug side effects using large language models with an application on semaglutide. In AMIA Annual Symposium Proceedings, volume 2024, page 332.

Arduin Findeis, Floris Weers, Guoli Yin, Ke Ye, Ruoming Pang, and Tom Gunter. 2025. Can external validation tools improve annotation quality for llm-as-ajudge? In Proceedings ofthe 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15997–16020.

Seraphina Fong, Alessandro Carollo, Lambros Lazuras, Ornella Corazza, and Gianluca Esposito. 2024. Ozempic (glucagon-like peptide 1 receptor agonist) in social media posts: unveiling user perspectives through reddit topic modeling. Emerging Trends in Drugs, Addictions, and Health, 4:100157.

Google. 2026. Youtube Data API (v3). Accessed: May 2026.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, and 1 others. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Xiaobo Guo and Soroush Vosoughi. 2024a. Disordereddabs: A benchmark for dynamic aspect-based summarization in disordered texts. In Findings of the Associationfor Computational Linguistics: EMNLP 2024, pages 416–431.

Xiaobo Guo and Soroush Vosoughi. 2024b. Modabs: Multi-objective learning for dynamic aspect-based summarization. In Findings of the Association for Computational Linguistics: ACL 2024, pages 2814– 2827.

Ivan Habernal and Iryna Gurevych. 2017. Argumentation mining in user-generated web discourse. Computational linguistics, 43(1):125–179.

Sabrina H Han, Rachel Safeek, Kyle Ockerman, Nhan Trieu, Patricia Mars, Audrey Klenke, Heather Furnas, and Sarah Sorice-Virk. 2024. Public interest in the off-label use of glucagon-like peptide 1 agonists (ozempic) for cosmetic weight loss: a google trends analysis. Aesthetic Surgery Journal, 44(1):60–67.

Marjan Hosseinia, Eduard Dragut, Dainis Boumber, and Arjun Mukherjee. 2021. On the usefulness of personality traits in opinion-oriented tasks. In RANLP, pages 547–556.

Marjan Hosseinia, Eduard Dragut, and Arjun Mukherjee. 2020. Stance prediction for contemporary issues: Data and experiments. In SocialNLP, pages 32–40.

A Javaid, S Baviriseaty, R Javaid, A Zirikly, H Kukreja, CH Kim, and 1 others. 2024. Trends in glucagon-like peptide-1 receptor agonist social media posts using artificial intelligence. jacc adv. 2024; 3 (9): 101182.

Yohan Jo, Elijah Mayfield, Chris Reed, and Eduard Hovy. 2020. Machine-aided annotation for finegrained proposition types in argumentation. In Proceedings of the Twelfth Language Resources and Evaluation Conference, pages 1008–1018.

Jaehun Jung, Faeze Brahman, and Yejin Choi. 2024. Trust or escalate: Llm judges with provable guarantees for human agreement. arXiv preprint arXiv:2407.18370.

Hannah Kim, Kushan Mitra, Rafael Li Chen, Sajjadur Rahman, and Dan Zhang. 2024. Meganno+: A human-llm collaborative annotation system. In Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations, pages 168–176.

Fangping Lan, Abdullah Aljebreen, and Eduard Dragut. 2025. UniT: One document, many revisions, too many edit intention taxonomies. In Findings of ACL, pages 23005–23024.

John Lawrence and Chris Reed. 2019. Argument mining: A survey. Computational linguistics, 45(4):765– 818.

Kenton Lee, Luheng He, Mike Lewis, and Luke Zettlemoyer. 2017. End-to-end neural coreference resolution. In Proceedings of the 2017 conference on empirical methods in natural language processing, pages 188–197.

Minzhi Li, Taiwei Shi, Caleb Ziems, Min-Yen Kan, Nancy Chen, Zhengyuan Liu, and Diyi Yang. 2023. Coannotating: Uncertainty-guided work allocation between human and large language models for data annotation. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 1487–1505.

Marco Lippi and Paolo Torroni. 2016. Argument mining from speech: Detecting claims in political debates. In Proceedings of the AAAI conference on artificial intelligence, volume 30.

Marco Lippi, Paolo Torroni, and 1 others. 2015. Context-independent claim detection for argument mining. In IJCAI, volume 15, pages 185–191. Buenos Aires.

Yuxuan Liu, Tianchi Yang, Shaohan Huang, Zihan Zhang, Haizhen Huang, Furu Wei, Weiwei Deng, Feng Sun, and Qi Zhang. 2024. Calibrating llmbased evaluator. In Proceedings of the 2024 joint international conference on computational linguistics, language resources and evaluation (lrec-coling 2024), pages 2638–2656.

Aleksandr Meister, Matvei Novikov, Nikolay Karpov, Evelina Bakhturina, Vitaly Lavrukhin, and Boris Ginsburg. 2023. Librispeech-pc: Benchmark for evaluation of punctuation and capitalization capabilities of end-to-end asr models. In 2023 IEEE automatic speech recognition and understanding workshop (ASRU), pages 1–7. IEEE.

Raquel Mochales and Marie-Francine Moens. 2011. Argumentation mining. Artificial intelligence and law, 19(1):1–22.

Arbi Haza Nasution and Aytug Onan. 2024. Chatgpt˘ label: Comparing the quality of human-generated and llm-generated annotations in low-resource language nlp tasks. Ieee Access, 12:71876–71900.

Nick Pangakis and Sam Wolken. 2025. Keeping humans in the loop: human-centered automated annotation with generative ai. In Proceedings of the International AAAI Conference on Web and Social Media, volume 19, pages 1471–1492.

Moritz Plenz, Philipp Heinisch, Janosch Gehring, Philipp Cimiano, and Anette Frank. 2025. From argumentation to deliberation: Perspectivized stance vectors for fine-grained (dis) agreement analysis. In Findings ofthe Associationfor Computational Linguistics: NAACL 2025, pages 1525–1553.

Fatima Quddos, Zachary Hubshman, Allison Tegge, Daniel Sane, Erin Marti, Anita S Kablinger, Kirstin M Gatchalian, Amber L Kelly, Alexandra G DiFeliceantonio, and Warren K Bickel. 2023.

Semaglutide and tirzepatide reduce alcohol consumption in individuals with obesity. Scientific Reports, 13(1):20998.

Qwen Team. 2026. Qwen3.5: Towards native multimodal agents.

Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. 2023. Robust speech recognition via large-scale weak supervision. In International conference on machine learning, pages 28492–28518. PMLR.

Robin Schaefer, René Knaebel, and Manfred Stede. 2023. Towards fine-grained argumentation strategy analysis in persuasive essays. In Proceedings ofthe 10th workshop on argument mining, pages 76–88.

Andrew Schneider and Eduard Dragut. 2015. Towards debugging sentiment lexicons. In ACL, pages 1024– 1034.

Joel Shor, Ruyue Agnes Bi, Subhashini Venugopalan, Steven Ibara, Roman Goldenberg, and Ehud Rivlin. 2023. Clinical bertscore: an improved measure of automatic speech recognition performance in clinical settings. In Proceedings ofthe 5th Clinical Natural Language Processing Workshop, pages 1–7.

Guixin Su, Mingmin Wu, Zhongqiang Huang, Yongcheng Zhang, Tongguan Wang, Yuxue Hu, and Ying Sha. 2024. Refine, align, and aggregate: Multiview linguistic features enhancement for aspect sentiment triplet extraction. In Findings ACL, pages 3212–3228.

Leila Tavakoli and Hamed Zamani. 2025. Reliable annotations with less effort: Evaluating llm-human collaboration in search clarifications. In Proceedings of the 2025 International ACM SIGIR Conference on Innovative Concepts and Theories in Information Retrieval (ICTIR), pages 92–102.

Dietrich Trautmann, Johannes Daxenberger, Christian Stab, Hinrich Schütze, and Iryna Gurevych. 2020. Fine-grained argument unit recognition and classification. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 9048–9056.

Jacky Visser, John Lawrence, Chris Reed, Jean Wagemans, and Douglas Walton. 2020. Annotating argument schemes. In Argumentation through languages and cultures, pages 101–139. Springer.

Lei Wang and Eduard Dragut. 2024. The overlooked repetitive lengthening form in sentiment analysis. In Findings ofEMNLP, pages 16225–16238.

Lei Wang, Min Huang, and Eduard Dragut. 2026. Danceha: A multi-agent framework for documentlevel aspect-based sentiment analysis. AAAI, 40(35):29714–29722.

Xinru Wang, Hannah Kim, Sajjadur Rahman, Kushan Mitra, and Zhengjie Miao. 2024. Human-llm collaborative annotation through effective verification of llm

labels. In Proceedings of the 2024 CHI conference on humanfactors in computing systems, pages 1–21.

Fan Yang, Eduard Dragut, and Arjun Mukherjee. 2020. Claim verification under positive unlabeled learning. In ASONAM, pages 143–150.

Andy Wai Kan Yeung, Fabian Peter Hammerle, Sybille Behrens, Maima Matin, Michel-Edwar Mickael, Olena Litvinova, Emil D Parvanov, Maria Kletecka-Pulker, and Atanas G Atanasov. 2025. Online information about side effects and safety concerns of semaglutide: mixed methods study of youtube videos. JMIR infodemiology, 5(1):e59767.

yt-dlp Contributors. 2026. yt-dlp.

Shanshan Zhang, Lihong He, Eduard Dragut, and Slobodan Vucetic. 2019. How to invest my time: Lessons from human-in-the-loop entity extraction. In KDD, pages 2305–2313.

## 10 Appendix

## 10.1 Related Work

Human–LLM Hybrid Annotation: Recent work increasingly treats large language models as collaborative annotation agents operating within humansupervised workflows rather than fully autonomous labeling systems (Li et al., 2023; Wang et al., 2024; Kim et al., 2024; Pangakis and Wolken, 2025; Jung et al., 2024; Nasution and Onan, 2024; Zhang et al., 2019). These studies show that LLMs can substantially accelerate annotation and benchmark construction while still requiring calibration, verification, and expert reconciliation to maintain reliability (Dragut et al., 2021; Findeis et al., 2025; Detommaso et al., 2024; Liu et al., 2024; Tavakoli and Zamani, 2025). Our work follows this broader hybrid annotation paradigm but targets substantially richer narrative environments by jointly modeling atomic claims, thematic aspects, stance, and multidimensional pragmatic structure.

Health Discourse on Social Media: Social media platforms increasingly shape public health communication through decentralized peer, influencer, and consumer-generated narratives rather than traditional clinical authorities (Javaid et al., 2024; Fong et al., 2024). This shift is especially pronounced on video-centric platforms such as YouTube and TikTok, where users share personal experiences, medication advice, side-effects, and persuasive health narratives at scale (Yeung et al., 2025; Basch et al., 2023).—environments that studies show amplify anecdotal evidence, off-label medication discourse, and incomplete or misleading health interpretations (Javaid et al., 2024; Han et al., 2024; Quddos et al., 2023). Existing computational approaches to online health discourse rely largely on coarse topic modeling or sentiment analysis (Javaid et al., 2024; Fong et al., 2024). However, dense health narratives often interleave causal claims, uncertainty, experiences, and persuasive rhetoric within short conversational spans, making flat, document-level representations insufficient for fine-grained analysis. These limitations motivate structured representations capable of modeling claim-level semantic and pragmatic structure (Duan et al., 2025).

## 10.2 Multi-Dimensional Health Claim Taxonomy

This section details the six-dimensional pragmatic typology used to analyze rhetorical and communicative features in dense health narratives. Table 10 provides the formalized definitions, categorical labels, and structural scopes for each of the orthogonal axes.

## 10.3 Extended Qualitative Analysis of Multi-Aspect Entanglement and Rhetorical Trajectories

To complement the paired-claim comparison presented in the main text, this subsection provides broader qualitative evidence demonstrating why both domain-specific aspect taxonomies and multiaxis pragmatic profiling are required to capture the complexity of health discourse.

Multi-Aspect Entanglement in Dense Narratives. As discussed in Section 1, health discourse exhibits high claim density (averaging 13.22 claims per minute), wherein speakers rapidly interleave assertions across multiple distinct health dimensions with opposing stances. Coarser representations, such as document-level sentiment or broad topic categorization, completely collapse these distinctions. Table 11 illustrates this phenomenon using a representative transcript span from the TRT domain, showing how a single speaker simultaneously moves across muscle performance, adverse side effects, sexual health, severe medical risks, and clinical eligibility within seconds.

Intra-Speaker Rhetorical Strategy Trajectories. Beyond aspect-level assignment, capturing pragmatic variation across multiple axes reveals structural shifts in a speaker’s rhetorical strategy that remain invisible under flat sentiment or stance frameworks. To demonstrate this, Table 12 tracks six consecutive claims made by a single physician discussing cardiovascular risk under predominantly the same thematic aspect (Severe Risks & Long-Term Safety).

Table 10: The Multi-Dimensional Health Claim Taxonomy (6-Axis Typology).
<table><tr><td>Axis &amp; Definition</td><td>Labels</td></tr><tr><td>I. Logical Relationship The structural mechanism connecting concepts within the assertion.</td><td>(1) Causal • (2) Correlational • (3) Comparative • (4) Conditional • (5) None_Logical</td></tr><tr><td>II. Verifiability &amp; Intent The nature of the statement regarding biological reality or logistical context.</td><td>(1) Factual(Clinical/Biological) • (2) Factual(Logistical/Contextual) • (3) Value(Subjective) • (4) Policy(Prescriptive) • (5) Speculative/Rumor</td></tr><tr><td>III. Evidence Basis The implied source of authority or support used to back the claim.</td><td>(1) Scientific(Statistical) • (2) Scientific(Expert/Authority) • (3) Scien- tific(General/Vague) • (4) External Reference(Media/Cultural) • (5) Anecdotal/Per- sonal • (6) Visual/Observable • (7) General Knowledge/Common Sense • (8) Com- mercial</td></tr><tr><td>IV. Certainty The speaker&#x27;s confidence level and the use of linguistic hedges.</td><td>(1) Absolute/Imperative • (2) Hedged(Probabilistic) • (3) Attributed/Distanced</td></tr><tr><td>V. Temporality The time orientation or universal principle of the assertion.</td><td>(1) Current/Past • (2) Counterfactual • (3) Future/Predictive • (4) Sequential • (5) Timeless</td></tr><tr><td>VI. Claim Focus The primary rhetorical intent and structural purpose of the claim.</td><td>(1) Explanatory/Argumentative • (2) Descriptive/Definitional • (3) Narrative/Se- quential • (4) Evaluative • (5) Promotional / Čommercial</td></tr></table>

Table 11: Representative transcript excerpt demonstrating multi-aspect entanglement, where a single conversational span interleaves multiple health dimensions and opposing stances.
<table><tr><td rowspan=1 colspan=1>Claim (Transcript Excerpt)</td><td rowspan=1 colspan=1>Thematic Aspect</td><td rowspan=1 colspan=1>Stance</td></tr><tr><td rowspan=1 colspan=1>&quot;you&#x27;ll look better, you&#x27;ll feel better, and you&#x27;ll put muscle on faster&#x27;</td><td rowspan=1 colspan=1>Muscle, Performance &amp; Aesthetics</td><td rowspan=1 colspan=1>Positive</td></tr><tr><td rowspan=1 colspan=1>&quot;will f**k your hormonal system up indefinitely if you do it too long&#x27;</td><td rowspan=1 colspan=1>Side Effects &amp; Adverse Reactions</td><td rowspan=1 colspan=1>Negative</td></tr><tr><td rowspan=1 colspan=1>“they have a lot of problems with getting erections&#x27;</td><td rowspan=1 colspan=1>Sexual Health &amp; Libido</td><td rowspan=1 colspan=1>Negative</td></tr><tr><td rowspan=1 colspan=1>they have liver problems&#x27;</td><td rowspan=1 colspan=1>Medical Health &amp; Biomarkers</td><td rowspan=1 colspan=1>Negative</td></tr><tr><td rowspan=1 colspan=1>“Many of them die from heart attacks early&quot;</td><td rowspan=1 colspan=1>Severe Risks &amp; Long-Term Safety</td><td rowspan=1 colspan=1>Negative</td></tr><tr><td rowspan=1 colspan=1>&quot;I still got washboard abs, I can still do single-arm chin-ups&quot;</td><td rowspan=1 colspan=1>Muscle, Performance &amp; Aesthetics</td><td rowspan=1 colspan=1>Positive</td></tr><tr><td rowspan=1 colspan=1>&quot;the only place for it is when you&#x27;ve got four doctors in balance, you&#x27;vedetoxified your body&quot;</td><td rowspan=1 colspan=1>Clinical Diagnosis &amp; Patient Eligi-bility</td><td rowspan=1 colspan=1>Neutral</td></tr></table>

The pragmatic profile shifts dynamically across the narrative: the speaker opens with a categorical framing statement, transitions to public concern, cites published expert literature, summarizes a trend across studies, describes their own research methodology, and finally concludes with a general statement. This complete rhetorical arc, from initial framing through evidence accumulation to final assertion, proves the necessity of decomposing discourse into multi-dimensional pragmatic axes.

## 10.4 Transcript Triage and Filtering Prompt

This prompt template was used for initial screening of video transcripts harvested from the YouTube Data API. This triage phase ensures the relevance and linguistic consistency of the candidate pool before purposive sampling. Placeholders {TOPIC} and {transcript\_text} are populated at runtime.

Transcript Triage and Filtering Prompt Template   
You are an AI assistant performing an initial triage on   
the transcript of a YouTube video. Your task is to   
quickly determine if the transcript is usable and   
relevant to the topic of interest.   
Topic of Interest: {TOPIC}   
Transcript:   
{transcript\_text}   
### INSTRUCTIONS:   
Respond ONLY with a valid JSON object containing three   
keys: 1. has\_meaningful\_speech (boolean): true if the   
transcript has coherent sentences. 2. is\_english   
(boolean): true if the primary language is English. 3.   
is\_relevant (boolean): true if the transcript discusses   
the Topic of Interest, with a brief relevance\_reason   
(string).   
Response Format   
{   
"has\_meaningful\_speech": true,   
"is\_english": true,   
"is\_relevant": true,   
"relevance\_reason":   
"Explain briefly why the content is on-topic"   
}

Table 12: Intra-speaker rhetorical strategy trajectory across six claims by a single physician predominantly under the same thematic aspect (Severe Risks), illustrating how the 6-axis typology captures shifts in evidential framing, certainty, and logical structure that stance alone cannot represent.
<table><tr><td>Claim</td><td>Aspect</td><td>Stance</td><td>Logical</td><td>Verif.</td><td>Evidence</td><td>Certainty</td><td>Temp.</td><td>Focus</td></tr><tr><td>“Cardiovascular risk is a subject of Severe Risks controversy regarding TRT.&quot;</td><td></td><td>Neg</td><td>Non_Log</td><td>Factual (Clin.)</td><td>Sci. (Gen./Vague)</td><td>Absolute</td><td>Current/Past</td><td>Descriptive</td></tr><tr><td>“There was significant concern that Severe Risks testosterone may cause a heart attack.&quot;</td><td></td><td>Neg</td><td>Causal</td><td>Factual (Clin.)</td><td>General edge</td><td>Knowl- Hedged</td><td>Current/Past</td><td>Descriptive</td></tr><tr><td>“Men with lower testosterone were sig- Severe Risks nificantly more likely to die earlier&quot; (citing Molly Shores, 2006).</td><td></td><td>Neu</td><td>Comparative</td><td>Factual (Clin.)</td><td>Sci. (Expert/Auth.) Attributed</td><td></td><td>Current/Past</td><td>Descriptive</td></tr><tr><td>“Men with lower testosterone exhib- Severe Risks ited a general trend of higher death rates.&quot;</td><td></td><td>Pos</td><td></td><td>Correlational Factual (Clin.)</td><td>Sci. (Gen./Vague)</td><td>Hedged</td><td>Current/Past</td><td>Descriptive</td></tr><tr><td>“We found over 200 articles address- Medical Health ing testosterone and cardiovascular dis- ease.&quot;</td><td></td><td>Neu</td><td>Non_Log</td><td>Factual (Log.)</td><td>Anecdotal/Personal Attributed</td><td></td><td>Current/Past</td><td>Narrative</td></tr><tr><td>“Low testosterone is a risk factor for Severe Risks cardiovascular events.&quot;</td><td></td><td>Neg</td><td></td><td>Correlational Factual (Clin.)</td><td>Sci. (Gen./Vague)</td><td>Absolute</td><td>Timeless</td><td>Descriptive</td></tr></table>

## 10.5 Orthographic Denoising Evaluation

We evaluated all 60 videos using two complementary metrics adapted from prior ASR evaluationquality assessment work: Medical Term Accuracy (MTA) (Shor et al., 2023) and Punctuation Quality (PQ) (Meister et al., 2023). Both metrics were computed on the raw Whisper transcripts and the transcripts produced after the output and Orthographic Denoising and Stabilization stage to enable direct comparison. As shown in Table 13, denoising improved average Medical Term Accuracy (MTA) from 0.97 to 1.0 and average PQ from 0.95 to 0.98 across the full corpus. The normalization stage proved critical for a non-trivial subset of transcripts. Five videos exhibited raw MTA at or below 0.67, all five recovered to 1.0 following denoising. Similarly, five videos exhibited raw PQ below 0.9, with three falling below 0.40, recovering to a mean of 0.94 post-denoising, with the most severely degraded transcript improving from 0.22 to 0.97. Tables 14 and 15 present representative corrections.

Table 13: Medical Term Accuracy and Punctuation Quality scores for raw Whisper ASR output and postdenoising transcripts (N = 60).
<table><tr><td>Med. Acc. Domain (Raw)</td><td>Med. Acc. (Den.)</td><td>∆ Med.</td><td>(Raw)</td><td>Punct. Punct. (Den.)</td><td>∆ Punct.</td></tr><tr><td>Ozempic</td><td>0.889</td><td></td><td>0.894</td><td>0.983</td><td>+0.089</td></tr><tr><td>TRT 1.000</td><td>1.000 1.000</td><td>+0.111 +0.000</td><td>0.958</td><td>0.975</td><td>+0.017</td></tr><tr><td>Collagen 1.000</td><td>1.000</td><td>+0.000</td><td>0.969</td><td>0.967</td><td>-0.002</td></tr><tr><td>Fasting 0.965</td><td>1.000</td><td>+0.035</td><td>0.971</td><td>0.973</td><td>+0.002</td></tr><tr><td>Overall</td><td>0.963</td><td>1.000 +0.037</td><td>0.948</td><td>0.975</td><td>+0.027</td></tr></table>

Table 14: Denoising example from the Ozempic domain
<table><tr><td colspan="2">Raw Whisper Output</td><td colspan="3">Denoised Output</td></tr><tr><td>...heard about the buzz ...heard about the buzz</td><td></td><td></td><td></td><td></td></tr><tr><td>around a Zempic Loss.</td><td>Weight around</td><td>loss.</td><td>Ozempic</td><td>weight</td></tr><tr><td>... the weightall he</td><td>seems ...the</td><td></td><td>weight</td><td>always</td></tr><tr><td>to come back.</td><td></td><td>seems to come back.</td><td></td><td></td></tr><tr><td>... that a Zempic</td><td>might ...that</td><td></td><td>Ozempic</td><td>might</td></tr><tr><td>seem like a quick fix ..</td><td></td><td>seem like a quick fix . .</td><td></td><td></td></tr><tr><td>... beyond</td><td></td><td>just ... beyond</td><td></td><td>just</td></tr><tr><td>temperate to temperate</td><td></td><td>temporary to temporary</td><td></td><td></td></tr><tr><td>discomfort.</td><td></td><td>discomfort.</td><td></td><td></td></tr></table>

## 10.6 Orthographic Denoising and Stabilization

The following prompt template was used with Gemini 2.5 Flash to stabilize raw Whisper ASR output.

Orthographic Denoising Prompt Template   
You are an expert-level text editor and proofreader   
specializing in medical transcripts. The following text   
is an AI-generated transcript from Whisper. It likely   
contains spelling errors and incorrect punctuation.   
Topic of Interest: {TOPIC}   
Transcript:   
{transcript\_text}   
### INSTRUCTIONS:   
1. Correct Spelling: Fix spelling errors. Pay special   
attention to metabolic, physiological, and   
topic-specific terms. (Example keywords provided for   
{TOPIC} included: [e.g., Autophagy, Insulin Sensitivity,   
Glycogen, Bioavailability, etc.])   
2. Fix Punctuation: Fix capitalization, sentence breaks,   
and punctuation to create clean, readable text.   
3. Conservative Editing: Do NOT summarize. Do NOT delete   
content. Do NOT add new content. Keep the text as close   
to the original meaning as possible, just cleaned up.   
Response Format   
Respond ONLY with a valid JSON object containing a   
single key:   
{   
"denoised\_transcript":   
"The corrected and punctuated text here..."   
}

Table 15: Denoising example from the Fasting domain
<table><tr><td colspan="2">Raw Whisper Output</td><td colspan="2">Denoised Output</td></tr><tr><td>... something</td><td>called</td><td>... something</td><td>called</td></tr><tr><td>a topogy , which is ...</td><td></td><td>autophagy,</td><td> which is ...</td></tr><tr><td>Now what</td><td>the studies</td><td>Now,</td><td>what the studies</td></tr><tr><td>show is that fasting trig-</td><td></td><td>show</td><td>is that fasting-</td></tr><tr><td>gered a topogy</td><td>in the brain</td><td>triggered</td><td>autophagy</td></tr><tr><td>So really</td><td>great for brain...</td><td>So,</td><td>really great for brain...</td></tr><tr><td>to</td><td></td><td>reduce ... to reduce</td><td>body-wide in-</td></tr><tr><td>the body-wide</td><td>inflam-</td><td>flammation.</td><td></td></tr><tr><td>mation.</td><td></td><td></td><td></td></tr><tr><td>So try</td><td>maybe fasting for</td><td>So,</td><td>try maybe fasting for</td></tr><tr><td>24 hours.</td><td></td><td>24 hours.</td><td></td></tr></table>

## 10.7 Annotation Labor and Complexity

Table 16 summarizes the approximate expert effort required to construct the gold-standard benchmark. The reported estimates include both independent annotation and collaborative reconciliation phases across the full multidimensional labeling workflow.

Table 16: Expert labor intensity per video (Cumulative Person-Minutes).
<table><tr><td>Annotation Phase</td><td>Avg. Person-Mins</td></tr><tr><td>Independent Phase (For 2 Annotators)</td><td></td></tr><tr><td>Statement Extraction Atomic Decomposition</td><td>10–20 min</td></tr><tr><td>Multidimensional Labeling</td><td>30–50 min 60–90 min</td></tr><tr><td>Consensus Phase</td><td></td></tr><tr><td>Consensus &amp; Dispute Resolution</td><td>25–40 min</td></tr><tr><td>Total Expert Time per Video</td><td>125–200 min</td></tr></table>

## 10.8 Aspect Taxonomy Instantiation

Table 18 presents the canonical aspect taxonomies induced across all four health domains.

## 10.9 Structured Discourse Prediction Prompts

We used two prompt templates for automated structured discourse prediction: one for joint aspect and stance classification, and one for six-dimensional pragmatic typology classification. Both prompts share a fixed instruction structure across all four domains; domain-specific content (aspect taxonomy, definitions, keywords, and few-shot examples) is populated via placeholders at runtime. The aspect taxonomy used to populate {ASPECT\_TAXONOMY} is provided in full in Table 18.

## Prompt 1: Aspect and Stance Classification.

```jsonl
Aspect & Stance Classification Prompt Template
You are a senior health communication researcher.
Analyze the following list of “Target Claims” about the
Topic of Interest.
Topic of Interest: {TOPIC}
### TAXONOMY: {N} MASTER ASPECTS
{ASPECT_TAXONOMY}
[Domain-specific aspect labels, definitions, and
keywords. See Table 18 for all four domains.]
### STANCE DEFINITIONS
• Positive: The claim describes a benefit, a success, a
favorable outcome, or an improvement.
• Negative: The claim describes a harm, a failure, a
side effect, a risk, or an unfavorable outcome.
• Neutral: The claim is a factual observation, a
description of medical protocol, or a statement of
intent/action without a positive or negative judgment.
### TARGET CLAIMS LIST:
{CLAIMS_LIST_JSON}
### INSTRUCTIONS: 1. For each claim, first provide a
1-sentence Reasoning identifying the core subject and
context. 2. Based on that reasoning, assign exactly ONE
Aspect Label and ONE Stance. 3. Respond ONLY with a
valid JSON object containing a list named
“classified_claims”.
Response Format
{
"classified_claims": [
{
"claim_id": 0,
"reasoning": "Explain the medical or
logistical context",
"aspect_label": "Label Name",
"stance": "Sentiment"
}
]
}
```

## Prompt 2: Six-Dimensional Pragmatic Typology Classification.

6-Axis Typology Classification Prompt Template   
You are a senior health communication researcher.   
Analyze the “Target Claim” provided about {TOPIC}.   
To help you understand the intent, I have provided the   
Surrounding Context (the 2 claims immediately before   
and after this claim in the video).   
IMPORTANT GENERAL RULES   
• Always pick the single best-fitting label for each   
axis (no multi-labeling).   
• Use the Surrounding Context to clarify ambiguity, but   
label ONLY the “Target Claim.”   
• If a statement contains multiple parts, label the   
main claim (the core assertion).   
• Be consistent across items: similar phrasing should   
yield similar labels.   
[Axis definitions for I–VI as specified in Table 10]   
### GOLD STANDARD FEW-SHOT EXAMPLES   
{FEW\_SHOT\_EXAMPLES}   
[5 manually annotated claims drawn from the target   
domain. Examples vary per domain to reflect   
domain-specific vocabulary]   
### SURROUNDING CONTEXT:   
{SURROUNDING\_CONTEXT}   
>>> TARGET CLAIM TO ANALYZE: “{TARGET\_CLAIM}” <   
Response Format   
{   
"Logical": "Label",   
"Verifiability": "Label",   
"Evidence Basis": "Label",   
"Certainty": "Label",   
"Temporality": "Label",   
"Claim Focus": "Label"   
}  
The axis definitions and label inventories supplied

Table 17: Inter-annotator agreement metrics across hierarchical pipeline layers.
<table><tr><td>Annotation Layer / Axis</td><td>Metric</td><td>Ozempic</td><td>TRT</td><td>Collagen</td><td>Fasting</td><td>Mean</td></tr><tr><td>Statement Extraction</td><td>Pairwise  $\overline { { F _ { 1 } } }$ </td><td>0.78</td><td>0.99</td><td>0.95</td><td>0.98</td><td>0.93</td></tr><tr><td>Atomic Decomposition</td><td>Raw Agreement</td><td>0.89</td><td>0.93</td><td>0.92</td><td>0.95</td><td>0.92</td></tr><tr><td>Aspect Assignment</td><td>Cohen&#x27;s κ Raw Agreement</td><td>0.89 0.90</td><td>0.81 0.83</td><td>0.88 0.89</td><td>0.75 0.78</td><td>0.83 0.85</td></tr><tr><td rowspan="2">Stance Classification</td><td>Cohen&#x27;s κ</td><td>0.65</td><td>0.87</td><td>0.84</td><td>0.83</td><td>0.80</td></tr><tr><td>Raw Agreement</td><td>0.77</td><td>0.92</td><td>0.90</td><td>0.89</td><td>0.87</td></tr><tr><td>6-Axis Typology Dimensions</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Axis I: Logical Relationship</td><td>Cohen&#x27;s κ</td><td>0.83</td><td>0.86</td><td>0.88</td><td>0.82</td><td>0.85</td></tr><tr><td rowspan="2">Axis II: Verifiability &amp; Intent</td><td>Raw Agreement Cohen&#x27;s κ</td><td>0.90</td><td>0.91</td><td>0.92</td><td>0.89</td><td>0.90</td></tr><tr><td>Raw Agreement</td><td>0.63</td><td>0.74</td><td>0.82</td><td>0.81</td><td>0.75</td></tr><tr><td rowspan="2">Axis III: Evidence Basis</td><td>Cohen&#x27;s κ</td><td>0.79</td><td>0.85</td><td>0.90</td><td>0.89</td><td>0.86</td></tr><tr><td>Raw Agreement</td><td>0.64</td><td>0.71 0.87</td><td>0.90</td><td>0.85</td><td>0.77</td></tr><tr><td rowspan="2">Axis IV: Certainty</td><td>Cohen&#x27;s κ</td><td>0.76 0.79</td><td>0.68</td><td>0.92</td><td>0.95</td><td>0.88</td></tr><tr><td>Raw Agreement</td><td></td><td>0.89</td><td>0.80</td><td>0.88</td><td>0.79</td></tr><tr><td rowspan="2">Axis V: Temporality</td><td>Cohen&#x27;s κ</td><td>0.92 0.77</td><td></td><td>0.94</td><td>0.97</td><td>0.93</td></tr><tr><td>Raw Agreement</td><td></td><td>0.65</td><td>0.88</td><td>0.67</td><td>0.74</td></tr><tr><td rowspan="2">Axis VI: Claim Focus</td><td>Cohen&#x27;s κ</td><td>0.87</td><td>0.86</td><td>0.95</td><td>0.90</td><td>0.89</td></tr><tr><td>Raw Agreement</td><td>0.82</td><td>0.56</td><td>0.81</td><td>0.53</td><td>0.68</td></tr><tr><td rowspan="2">6-Axis Typology (Mean)</td><td>Cohen&#x27;s κ</td><td>0.86</td><td>0.75</td><td>0.87</td><td>0.71</td><td>0.80</td></tr><tr><td>Raw Agreement</td><td>0.75 0.85</td><td>0.70 0.85</td><td>0.85 0.92</td><td>0.76 0.88</td><td>0.76 0.88</td></tr></table>

to Prompt 2 are identical across all four domains and correspond directly to the six-dimensional typology defined in Table 10.

## 10.10 Qualitative Error Analysis of Chain-of-Thought Prompting

To better understand why a straightforward, singlepass Chain-of-Thought (CoT) prompting underperforms on high-dimensional pragmatic profiling compared to heuristic constraints (Section 6.2), we performed a qualitative error analysis comparing predictions where heuristic prompting successfully matched the ground truth but CoT failed. This analysis reveals three distinct behavioral failure modes in standard CoT prompting:

1. Holistic Interference from Joint Prediction: Our CoT implementation utilized a lightweight reasoning constraint designed to capture speaker intent across all six pragmatic axes simultaneously. This holistic approach caused structural interference: the generated reasoning would often optimize for one axis (e.g., Claim Focus) while completely misaligning with another (e.g., Logical Relationship).

2. Semantic Over-Analysis over Structural Mapping: The CoT prompt frequently caused the model to over-analyze conversational nuances rather than enforce schema constraints. As shown in Table 19, for the claim “ozempic is the ideal weight loss solution for diabetics,” the model generates an accurate semantic explanation of the speaker’s intent in its reasoning block (positioning it as the “best possible option”), but completely fails to map that understanding to the correct categorical schema label, defaulting to None\_logical instead of Comparative.

3. Linguistic Triggers and Rationalization Traps: CoT proved highly vulnerable to modal verbs and ambiguous phrasing. For claims like “we can use ozempicfor weight loss,” the model got tripped up by surface-level linguistic triggers like the modal verb “can,” causing it to ignore the pragmatic context and absolute reality of the health narrative, incorrectly driving an absolute assertion into a Hedged classification. Furthermore, on borderline claims (e.g., “GLP active definitely worth the money”), the model demonstrated “rationalization flip-flopping”: because text generation is autoregressive, once the model randomly generated its first few reasoning tokens in a certain direction, it forced itself to write a highly convincing post-hoc rationalization for an incorrect label (fluctuating between Anecdotal and General Knowledge).

Table 18: Canonical aspect taxonomies induced across all four health domains.
<table><tr><td>Topic</td><td>Aspects</td></tr><tr><td></td><td>Ozempic (1) Weight Loss Effectiveness • (2) Medical Health Benefits (Non-Weight) • (3) Appetite &amp; Satiety • (4) Gas- trointestinal &amp; Acute Side Effects • (5) Mental &amp; Emotional Impact • (6) Aesthetic &amp; Physical Transformation • (7) Financial &amp; Insurance • (8) Logistics &amp; Supply Chain • (9) Medical Regimen &amp; Dosing • (10) Dietary &amp; Lifestyle Changes • (11) Long-term Safety &amp; Risks • (12) Social Stigma &amp; Perception • (13) Product Comparisons &amp; Alternatives</td></tr><tr><td>TRT</td><td>(1) Muscle, Performance &amp; Aesthetics • (2) Mental &amp; Cognitive Impact • (3) Sexual Health &amp; Libido • (4) Energy &amp; Vitality • (5) Medical Health &amp; Biomarkers • (6) Side Effects &amp; Adverse Reactions • (7) Fertility &amp; HPTA Shutdown • (8) Dosing Schedule &amp; Protocol Strategy • (9) Medication Form &amp; Delivery Tools • (10) Social Stigma &amp; Perception • (11) Financial, Market &amp; Access • (12) Severe Risks &amp; Long-Term Safety • (13) Lifestyle &amp; Holistic Integration • (14) Clinical Diagnosis &amp; Patient Eligibility</td></tr><tr><td>Collagen</td><td>(1) Ingredients &amp; Formulation • (2) Sensory &amp; Usability • (3) Quality &amp; Safety • (4) Skin Health &amp; Anti-Aging • (5) Joint &amp; Bone Mobility • (6) Hair &amp; Nail Fortification • (7) Muscle Growth &amp; Body Composition • (8) Digestion &amp; Satiety • (9) General Wellbeing • (10) Scientific Validation • (11) Price &amp; Value • (12) Brand Reputation &amp; Market</td></tr><tr><td>Fasting</td><td>(1) Weight Loss &amp; Body Composition • (2) Metabolism &amp; Energy Processing • (3) Cellular Health &amp; Longevity • (4) Hormonal Balance &amp; Endocrine Health • (5) Appetite, Hunger &amp; Cravings • (6) Mental Health &amp; Cognitive Function • (7) Physical Energy &amp; Vitality • (8) Immunity &amp; Inflammation • (9) Organ &amp; Cardiovascular Health • (10) Fasting Protocols &amp; Timing Strategy • (11) Nutritional Intake &amp; Food Choices • (12) Lifestyle Convenience &amp; Flexibility • (13) Medical Guidance, Safety &amp; Scientific Evidence • (14) Evolutionary &amp; Psychological Responses • (15) Side Effects &amp; Adverse Reactions • (16) Sleep &amp; Recovery • (17) Social Perception &amp; Public Awareness • (18) Market, Industry &amp; Financial Access • (19) Adjunct Fitness &amp; Holistic Habits</td></tr></table>

Table 19: Representative qualitative error cases illustrating behavioral failure modes of Chain-of-Thought (CoT) prompting on high-dimensional pragmatic discourse classification using Gemini 2.5 Flash.
<table><tr><td rowspan=1 colspan=1>Target Claim</td><td rowspan=1 colspan=1>Dimension</td><td rowspan=1 colspan=1>Gold Label</td><td rowspan=1 colspan=1>CoT Prediction</td><td rowspan=1 colspan=1>Observed Failure Mechanism</td></tr><tr><td rowspan=1 colspan=1>&quot;Ozempic is the idealweight loss solution fordiabetics.&quot;</td><td rowspan=1 colspan=1>Logical Relation-ship</td><td rowspan=1 colspan=1>Comparative</td><td rowspan=1 colspan=1>None_Logical</td><td rowspan=1 colspan=1>Semantic Over-Analysis: Accuratelyexplains the superlative intent in rea-soning, but fails to map to the categori-cal schema label.</td></tr><tr><td rowspan=1 colspan=1>&quot;We can use ozempic forweight loss.&quot;</td><td rowspan=1 colspan=1>Certainty</td><td rowspan=1 colspan=1>Absolute / Impera-tive</td><td rowspan=1 colspan=1>Hedged (Probabilis-tic)</td><td rowspan=1 colspan=1>Surface Linguistic Triggers: Trippedup by the modal verb “can&quot;, misinter-preting an absolute statement as a prob-abilistic hedge.</td></tr><tr><td rowspan=1 colspan=1>&quot;GLP active definitelyworth the money.&quot;</td><td rowspan=1 colspan=1>Evidence Basis</td><td rowspan=1 colspan=1>General Knowledge/ Common Sense</td><td rowspan=1 colspan=1>Anecdotal    Per-sonal</td><td rowspan=1 colspan=1>Rationalization Flip-Flopping: Au-toregressive generation traps the modelinto writing post-hoc justifications thatfluctuate across runs.</td></tr></table>

Table 20: Comparison between original submitted evaluation results and the new clean execution baseline on the TRT domain dataset using Gemini 2.5 Flash.
<table><tr><td>Evaluation Task</td><td>Original Submitted (TRT)</td><td>New Baseline (Mean ± SD)</td><td>Variance</td></tr><tr><td>Thematic Aspect Labeling</td><td>89.12%</td><td> $\overline { { 9 2 . 8 2 \% \pm 1 . 2 3 \% } }$ </td><td> $\overline { { \Delta + 3 . 7 0 \% } }$ </td></tr><tr><td>Core Stance Detection</td><td>92.15%</td><td> $9 3 . 9 2 \% \pm 1 . 1 6 \%$ </td><td> $\Delta + 1 . 7 7 \%$ </td></tr><tr><td>6-Axis Typology Mean</td><td>88.62%</td><td> $9 3 . 4 1 \% \pm 0 . 1 5 \%$ </td><td> $\Delta + 4 . 7 9 \%$ </td></tr></table>

Table 21: Phase II Optimization: Impact of Heuristic Refinement across Contextual Windows.
<table><tr><td></td><td colspan="2">Atomic</td><td colspan="2">Batch</td><td colspan="2">Local ±2</td><td colspan="2">Narr.</td><td colspan="2">Trans.</td><td colspan="2">Summ.</td></tr><tr><td>Axis</td><td>Acc.</td><td>κ</td><td>Acc.</td><td>κ</td><td>Acc.</td><td>κ</td><td>Acc.</td><td>K</td><td>Acc.</td><td>κ</td><td>Acc.</td><td>κ</td></tr><tr><td>I. Logical Rel.</td><td>84.59</td><td>0.7383</td><td>85.30</td><td>0.7524</td><td>84.23</td><td>0.7366</td><td>84.59</td><td>0.7423</td><td>82.44</td><td>0.7012</td><td>85.30</td><td>0.7507</td></tr><tr><td>II. Verifiability</td><td>84.95</td><td>0.7277</td><td>84.59</td><td>0.7303</td><td>86.38</td><td>0.7609</td><td>86.38</td><td>0.7572</td><td>84.59</td><td>0.7209</td><td>86.38</td><td>0.7593</td></tr><tr><td>III. Evidence</td><td>82.44</td><td>0.7063</td><td>85.66</td><td>0.7616</td><td>84.23</td><td>0.7391</td><td>70.61</td><td>0.5680</td><td>69.89</td><td>0.5575</td><td>77.06</td><td>0.6493</td></tr><tr><td>IV. Certainty</td><td>91.04</td><td>0.7688</td><td>91.76</td><td>0.7713</td><td>90.32</td><td>0.7533</td><td>89.96</td><td>0.7339</td><td>92.83</td><td>0.8101</td><td>93.55</td><td>0.8221</td></tr><tr><td>V. Temporality</td><td>89.96</td><td>0.8161</td><td>90.32</td><td>0.8283</td><td>91.40</td><td>0.8443</td><td>90.32</td><td>0.8252</td><td>89.96</td><td>0.8181</td><td>90.68</td><td>0.8301</td></tr><tr><td>VI. Claim Focus</td><td>59.86</td><td>0.4918</td><td>60.93</td><td>0.5073</td><td>82.08</td><td>0.7442</td><td>62.37</td><td>0.5184</td><td>57.71</td><td>0.4655</td><td>56.27</td><td>0.4638</td></tr><tr><td>Mean</td><td>82.14 0.7082</td><td></td><td>83.09</td><td>0.7252</td><td>86.44</td><td>0.7631</td><td>80.70</td><td>0.6908</td><td>79.57</td><td>0.6789</td><td>81.54</td><td>0.7125</td></tr></table>

Table 22: Full multi-temperature robustness breakdown across 12 execution runs (3 trials per temperature setting) on the TRT domain dataset using Gemini 2.5 Flash.
<table><tr><td>Task / Dimension</td><td> $\mathbf { T i e r } \mathbf { A } \left( \tau = 0 . 0 \right)$ </td><td> $\mathbf { T i e r } \mathbf { B } \left( \tau = 0 . 2 \right)$ </td><td> $\mathbf { T i e r } \mathbf { C } \left( \tau = 0 . 5 \right)$ </td><td> $\mathbf { T i e r } \mathbf { D } \left( \tau = 0 . 8 \right)$ </td></tr><tr><td>Aspect</td><td> $9 2 . 8 2 \% \pm 1 . 2 3 \%$ </td><td> $9 2 . 9 2 \% \pm 0 . 2 6 \%$ </td><td> $9 1 . 6 8 \% \pm 1 . 5 2 \%$ </td><td> $9 1 . 4 9 \% \pm 0 . 5 3 \%$ </td></tr><tr><td>Stance</td><td> $9 3 . 9 2 \% \pm 1 . 1 6 \%$ </td><td> $9 4 . 1 8 \% \pm 0 . 1 1 \%$ </td><td> $9 3 . 0 6 \% \pm 1 . 4 9 \%$ </td><td> $9 3 . 6 2 \% \pm 0 . 4 5 \%$ </td></tr><tr><td>Logical Relationship</td><td> $9 5 . 2 3 \% \pm 0 . 0 6 \%$ </td><td> $9 5 . 1 8 \% \pm 0 . 1 4 \%$ </td><td> $9 5 . 3 5 \% \pm 0 . 2 0 \%$ </td><td> $9 5 . 1 6 \% \pm 0 . 2 0 \%$ </td></tr><tr><td>Verifiability &amp; Intent</td><td> $9 2 . 5 4 \% \pm 0 . 1 3 \%$ </td><td> $9 2 . 9 6 \% \pm 0 . 0 9 \%$ </td><td> $9 2 . 6 1 \% \pm 0 . 1 4 \%$ </td><td> $9 2 . 7 3 \% \pm 0 . 2 0 \%$ </td></tr><tr><td>Evidence Basis</td><td> $9 3 . 6 4 \% \pm 0 . 2 3 \%$ </td><td> $9 3 . 6 7 \% \pm 0 . 0 9 \%$ </td><td> $9 4 . 0 6 \% \pm 0 . 0 9 \%$ </td><td> $9 4 . 2 5 \% \pm 0 . 2 6 \%$ </td></tr><tr><td>Certainty</td><td> $9 4 . 3 0 \% \pm 0 . 0 9 \%$ </td><td> $9 4 . 5 3 \% \pm 0 . 1 5 \%$ </td><td> $9 4 . 5 5 \% \pm 0 . 0 7 \%$ </td><td> $9 4 . 5 3 \% \pm 0 . 1 0 \%$ </td></tr><tr><td>Temporality</td><td> $9 3 . 4 5 \% \pm 0 . 1 3 \%$ </td><td> $9 3 . 5 0 \% \pm 0 . 2 2 \%$ </td><td> $9 3 . 3 8 \% \pm 0 . 1 7 \%$ </td><td> $9 3 . 6 7 \% \pm 0 . 2 4 \%$ </td></tr><tr><td>Claim Focus</td><td> $9 1 . 3 0 \% \pm 0 . 2 5 \%$ </td><td> $9 1 . 3 7 \% \pm 0 . 2 5 \%$ </td><td> $9 1 . 0 5 \% \pm 0 . 0 3 \%$ </td><td> $9 1 . 4 9 \% \pm 0 . 1 7 \%$ </td></tr></table>

## 10.11 Phase II Optimization: Heuristic Refinement across Contextual Windows

To understand the interaction between heuristic injection and discourse context, we expanded our optimization analysis by evaluating the rule-based lexical heuristics across all six candidate context windows. In these ablation experiments, the exact same linguistic heuristics—governing structural transitions, explicit lexical anchors, and grammatical patterns—were appended to each respective context prompt. The results validate that while the introduction of rule-based heuristics uniformly elevates accuracy baselines across all configurations compared to raw context alone, their effectiveness is highly sensitive to the size of the discourse window. Most notably, Axis VI (Claim Focus) displays a critical architectural dependency: under every other contextual setting (such as isolated claims, full transcripts, or summaries), the accuracy of Claim Focus remains severely bottle-necked, dropping as low as 56.27%. However, when these linguistic guidelines are paired precisely with the short-range Local +/- 2 window, the model successfully anchors the lexical cues. This dynamic drives the mean performance to its study-wide peak of 86.44% accuracy $( \kappa = 0 . 7 6 )$

## 10.12 Multi-Temperature Robustness Analysis and Stability Breakdown

To evaluate the empirical stability and robustness of our structured discourse pipeline under stochastic decoding, we conducted a multi-temperature evaluation on the TRT domain dataset. We executed the complete pipeline across four decoding temperatures $( \tau \in \{ 0 . 0 , 0 . 2 , 0 . 5 , 0 . 8 \} )$ ) with three independent trials per setting, totaling 12 execution runs.

Analysis of Results and Stability: As detailed in Table 22, the standard deviations across all 12 runs are exceptionally low across all evaluation dimensions. This high level of internal consistency demonstrates that our framework and rulebased heuristics produce stable, reproducible outputs rather than random generational artifacts, confirming the overall robustness of the structured prompting approach across varying decoding temperatures.

Baseline Comparison: When comparing these repeated-trial figures to the primary evaluation results in the main text, minor baseline performance variations are observable (e.g., a 3.70% increase in Aspect Labeling, a 1.77% increase in Stance Detection, and a 4.79% increase in the 6-Axis Typology mean at Temperature 0.0, as shown in Table 20). These minor numerical differences stem from standard API execution environment updates and downstream library evolutions over time. Crucially, the relative performance tiers and core findings across tasks remain entirely consistent.

## 10.13 Cross-Model Performance

Table 23 reports complete results for all four models across all four domains under the identical B2-CoT pipeline configuration. While Gemini 2.5 Flash achieves the strongest overall performance, particularly on the 6-axis typology task, open-weight alternatives remain highly competitive. Specifically, Qwen 3.5 35B consistently approaches proprietary-model performance on aspect and stance classification across multiple thematic domains. In contrast, Llama 3.3 and Gptoss 20B exhibit more substantial performance degradation on the pragmatic typology task, highlighting a clear capability gap in handling highdimensional rhetorical profiling. As evidenced by these results, structured pragmatic discourse analysis imposes significantly greater reasoning and instruction-following demands than traditional thematic categorization, causing smaller or less specialized open-weight models to struggle under complex multi-dimensional inference settings.

Table 23: Cross-model performance comparison on Aspect labeling, Stance detection, and 6-Axis Typology classification under the identical B2-CoT pipeline configuration.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Domain</td><td colspan="2">Aspect Labeling</td><td colspan="2">Stance Detection</td><td colspan="2">6-Axis Typology (Mean)</td></tr><tr><td>Acc. (%)</td><td>κ</td><td>Acc. (%)</td><td>κ</td><td>Acc. (%)</td><td>κ</td></tr><tr><td rowspan="4">Gemini 2.5 Flash</td><td>Ozempic</td><td>79.23</td><td>0.7663</td><td>91.55</td><td>0.8716</td><td>86.15</td><td>0.7583</td></tr><tr><td>TRT</td><td>89.12</td><td>0.8813</td><td>92.15</td><td>0.8770</td><td>88.62</td><td>0.7676</td></tr><tr><td>Collagen</td><td>86.44</td><td>0.8475</td><td>85.59</td><td>0.7663</td><td>83.97</td><td>0.7249</td></tr><tr><td>Fasting</td><td>77.64</td><td>0.7512</td><td>83.38</td><td>0.7326</td><td>82.23</td><td>0.5865</td></tr><tr><td rowspan="4">Llama 3.3</td><td>Ozempic</td><td>76.49</td><td>0.7364</td><td>83.86</td><td>0.7531</td><td>75.56</td><td>0.6095</td></tr><tr><td>TRT</td><td>64.77</td><td>0.6147</td><td>82.56</td><td>0.7369</td><td>73.90</td><td>0.4880</td></tr><tr><td>Collagen</td><td>74.15</td><td>0.7105</td><td>75.85</td><td>0.5972</td><td>71.96</td><td>0.5244</td></tr><tr><td>Fasting</td><td>67.37</td><td>0.6376</td><td>84.59</td><td>0.7536</td><td>69.79</td><td>0.4077</td></tr><tr><td rowspan="4">Qwen 3.5 35B</td><td>Ozempic</td><td>77.54</td><td>0.7474</td><td>85.26</td><td>0.7761</td><td>81.58</td><td>0.6983</td></tr><tr><td>TRT</td><td>79.00</td><td>0.7708</td><td>83.30</td><td>0.7384</td><td>82.05</td><td>0.6221</td></tr><tr><td>Collagen</td><td>81.36</td><td>0.7896</td><td>78.39</td><td>0.6480</td><td>81.43</td><td>0.6815</td></tr><tr><td>Fasting</td><td>78.25</td><td>0.7579</td><td>86.71</td><td>0.7869</td><td>80.72</td><td>0.5847</td></tr><tr><td rowspan="4">Gpt-oss 20B</td><td>Ozempic</td><td>78.60</td><td>0.7582</td><td>81.40</td><td>0.7171</td><td>73.04</td><td>0.5894</td></tr><tr><td>TRT</td><td>71.17</td><td>0.6839</td><td>82.56</td><td>0.7319</td><td>75.33</td><td>0.5262</td></tr><tr><td>Collagen</td><td>71.61</td><td>0.6766</td><td>79.66</td><td>0.6629</td><td>75.85</td><td>0.5907</td></tr><tr><td>Fasting</td><td>70.69</td><td>0.6716</td><td>87.31</td><td>0.7971</td><td>76.08</td><td>0.5181</td></tr></table>