# MiniVer-V: Identifying Minimal Suficient Evidence for Short Video Verification

Leran Chen The University of Sydney lche0711@uni.sydney.edu.au

Lingnan Kong The University of Sydney lkon0802@uni.sydney.edu.au

Zile Cai The University of Sydney zcai0293@uni.sydney.edu.au

## Abstract

A core challenge in short-video fact-checking is identifying which evidence is suficient to support a verification conclusion. Existing approaches either give the verifier all available evidence, introducing noise, or select evidence by topical relevance, which conflates relatedness with suficiency. We identify evidential suficiency as the selection criterion: whether a subset of evidence is adequate to support a confident verdict without redundancy. We introduce MiniVer-V, a benchmark of 195 short videos with three-way verdict annotations (supported, refuted, insuficient) and 5,510 multimodal evidence units spanning visual keyframes, speech transcripts, and web-retrieved external sources. We propose a two-layer verification framework that separates claim–video consistency, assessed from internal evidence, from factual verdict determination, which additionally requires external corroboration. On top of it, a suficiency-driven greedy search assembles evidence until a suficiency threshold is met and outputs insuficient when the candidate pool is exhausted, rather than forcing a verdict. With Claude Sonnet 4, the method reaches a Macro-F1 of 0.510 using 4.5 evidence units on average (16% of the full evidence set), statistically indistinguishable from the full-evidence baseline (0.518 with 27.7 units), while significantly improving recognition of insuficient cases over the same search without abstention. The eficiency result replicates with GPT-5.5 and holds only partially with an open-weight Qwen2.5-72B verifier. Ablations show that external evidence is indispensable for factual determination, while internal video evidence grounds the verdict in claim–video consistency. These findings suggest that evidence-eficient verification is achievable, and that explicit abstention is needed when evidence is genuinely inadequate.

Keywords: short video verification, multimodal misinformation, evidential suficiency, evidence selection, multimodal evidence reasoning, abstention, benchmark

## 1 Introduction

Short-video platforms have become a major channel for news consumption and a breeding ground for misinformation [3, 7]. Unlike textual claims or static image–text posts [10], a short video weaves together visual scenes, spoken narration, overlaid text, and contextual metadata into a tightly coupled multimodal package that is persuasive, rapidly consumable, and easy to redistribute [3]. Verifying the factual claims embedded in such videos requires reasoning over heterogeneous evidence— yet not all evidence contributes equally. Consider a video that accurately depicts a real flood: the footage is genuine, the narration matches the visuals, and the claim appears credible. Yet the claim misattributes the footage to a diferent city and a diferent year. Visual keyframes confirm the event occurred but cannot establish where or when; only an external news report from the claimed date and location can confirm or deny the attribution. The critical question is not which evidence is relevant, but which evidence is suficient.

Existing fact-checking and multimodal verification systems typically either aggregate all available evidence or rank evidence by relevance and select a fixed-budget subset [7, 12, 14]. Both paradigms implicitly assume that more—or more relevant—evidence leads to more accurate verdicts. We identify three limitations of this assumption. First, relevance and suficiency are fundamentally diferent properties: a set of highly relevant units may collectively fail to establish the key factual link needed for verification, while a moderately relevant unit may provide the missing piece. Second, evidence beyond what is needed introduces redundancy and noise that can impair reasoning quality [11, 17]. Third, and most critically, systems without an explicit abstention mechanism tend to issue verdicts even when the evidence is insuficient [1]—a failure mode especially dangerous in misinformation detection, where a confident but wrong verdict can be more harmful than abstaining. Despite these risks, to our knowledge no existing multimodal video verification benchmark explicitly evaluates whether a model’s chosen evidence is truly suficient and free of redundancy.

In this paper, we formalize minimal suficient evidence set identification as a new task for short video verification. Given a claim and a pool of multimodal evidence units derived from the video and from external sources, the model must select a compact cross-modal evidence subset suficient to support a confident verdict—and explicitly abstain when no such subset exists. This formulation difers from conventional evidence ranking in two key respects: suficiency is a first-class objective rather than an implicit by-product of relevance, and evidence selection is jointly constrained by suficiency and compactness—the selector prefers subsets whose elements are not redundant with one another.

To study this task, we construct MiniVer-V, a benchmark of 195 short videos with three-way verdict annotations (supported, refuted, insuficient) and 5,510 multimodal evidence units spanning visual keyframe descriptions (V), speech transcripts (A), and web-retrieved external sources (E). We propose a two-layer verification framework: the first layer assesses claim–video consistency using internal evidence (V and A) alone; the second incorporates external evidence to reach a factual verdict. We evaluate five evidence selection strategies of increasing sophistication—from full-evidence and fixed-budget top-k baselines, through model-driven chain-of-thought selection, to our proposed method: a suficiency-driven greedy search with explicit abstention (B5) that dynamically determines how much evidence is needed per sample and outputs insuficient when the candidate pool is exhausted without reaching the suficiency threshold.

Experiments with Claude Sonnet 4 show that B5 achieves a Macro-F1 of 0.510 using 4.5 evidence units on average (16% of the full evidence set), recovering 98.5% of full-evidence performance. Compared with the greedy variant without abstention, B5 nearly doubles recall on insuficient cases. The eficiency result replicates with GPT-5.5 and holds only partially with the open-weight Qwen2.5-72B. Ablations show that external evidence is indispensable for factual determination, while internal evidence remains necessary for claim–video grounding. This work makes three contributions:

• We introduce minimal suficient evidence set identification as a task for short video verification, highlighting the underexplored gap between evidence relevance and evidence suficiency.

• We construct MiniVer-V, a benchmark of 195 short videos with 5,510 multimodal evidence units that enables controlled experiments on evidence selection strategies for video verification.

• We show that suficiency-aware selection with explicit abstention recovers 98.5% of fullevidence performance using only 16% of the evidence while improving recognition of genuinely insuficient cases.

## 2 Related Work

## 2.1 Multimodal Misinformation Detection

Recent work has advanced short-video misinformation detection through cross-modal benchmarks and video–text verification datasets [3, 7, 15, 16]. In the broader fact-checking literature, FEVER, MOCHEG, and AVeriTeC address text-based, multimodal, and web-grounded settings respectively [9, 12, 14], while DEFAME dynamically plans which multimodal tools to invoke and how much evidence to retrieve [2].

These works evaluate final verdict accuracy, and most of them provide evidence as a fixed input or retrieve it as a single package. To our knowledge, none treats evidence selection—how much evidence is enough and which subsets sufice—as a primary evaluation objective. MiniVer-V addresses this gap by providing unit-level evidence decomposition and evaluating selection strategies under controlled conditions.

## 2.2 Evidence Retrieval and Selection

Automated fact-checking is commonly framed as a pipeline of claim detection, evidence retrieval, and verdict prediction, with the reliability of retrieved evidence identified as a central open challenge [4]. Structured reasoning over retrieved evidence [5], claim decomposition into sub-tasks [6], and improved retrieval strategies [17] have all yielded substantial gains. However, these approaches focus on evidence relevance—retrieving evidence topically related to the claim. Relevance and suficiency are distinct: a set of highly relevant units may still fail to establish the critical factual link needed for verification. Atanasova et al. [1] formalize fact-checking with insuficient evidence and show that standard models often fail to recognize when evidence is insuficient. Our work extends this insight to the multimodal video setting by treating suficiency as a first-class selection criterion.

## 2.3 Selective Prediction and Abstention

Selective prediction—abstaining when confidence is low or evidence is inadequate—has a long history in machine learning, with a recent survey framing LLM abstention from query, model, and human-value perspectives [13]. In fact verification, the need for abstention arises naturally: FEVER and AVeriTeC both include “not enough evidence” verdict categories. Yet most verification systems lack an explicit mechanism to predict these labels reliably, defaulting to a forced choice regardless of evidence quality. Our method addresses this by incorporating a suficiency threshold: when the greedy search exhausts the candidate pool without reaching suficiency, the system outputs insuficient rather than forcing a potentially unreliable verdict, treating abstention as a procedura outcome of evidence insuficiency rather than a classification label to be learned.

## 3 MiniVer-V Benchmark

## 3.1 Task Setting and Benchmark Motivation

Task Definition. Given a short video paired with a textual claim about its content, the task is to determine a factual verdict—supported, refuted, or insuficient—by selecting and reasoning over a subset of multimodal evidence units derived from the video itself and from external sources.

Unlike binary fact-checking formulations that assign claims a true-or-false label, we adopt a three-way verdict scheme. The insuficient category captures an epistemic state common in real-world verification: the available evidence may be genuinely inadequate to determine whether a

claim is true or false. In our benchmark, 30.3% of samples (59 out of 195) carry this label, showing that evidence insuficiency is a substantial and practically relevant case rather than a marginal exception.

Motivation. MiniVer-V is designed for controlled study of evidence suficiency in short-video verification. By decomposing each sample into modality-labeled evidence units, it enables comparison of evidence selection strategies under a shared verification setting.

## 3.2 Data Collection

We curate 195 short videos through two complementary collection strategies: (i) tracing videos referenced in published fact-checking reports back to their original social media posts, and (ii) independently browsing social media platforms to identify videos whose factual claims warrant verification. Candidates are filtered to remove duplicates, videos shorter than 10 seconds, and clips with severely degraded visual quality. We further exclude videos whose claims are purely subjective or for which no verifiable evidence—either internal to the video or externally retrievable—could reasonably be obtained. All claims are written in Chinese. The videos are predominantly Chineselanguage; 40 of 195 (20.5%) contain mainly English speech. Videos come primarily from Douyin (112; 57.4%) and Xiaohongshu (64; 32.8%), with the remainder from X, Bilibili, Weibo, and news outlets (Appendix E.2). Topics span international politics, public health, science, consumer safety, and natural disasters.

For each video, an annotator writes a textual claim capturing its main factual assertion and assigns a verdict based on the video content and web-retrieved external evidence, using the original caption only as auxiliary context. A second team member reviews a random subset of samples to assess annotation consistency, yielding Cohen’s κ = 0.82.

Each sample is assigned one of three verdict labels: supported (the claim is substantiated by the available evidence), refuted (the claim is contradicted by the available evidence), or insuficient (the evidence does not permit a confident determination). The distribution across the 195 samples is shown in Table 1.

Table 1: Verdict distribution in MiniVer-V.
<table><tr><td>Verdict</td><td>Count</td><td>Proportion</td></tr><tr><td>Supported</td><td>97</td><td>49.7%</td></tr><tr><td>Refuted</td><td>39</td><td>20.0%</td></tr><tr><td>Insufficient</td><td>59</td><td>30.3%</td></tr><tr><td>Total</td><td>195</td><td>100%</td></tr></table>

The label distribution is imbalanced, with nearly half of the claims labeled as supported and only one-fifth as refuted. We retain the observed distribution rather than artificially balancing it, as it better reflects real-world verification conditions.

## 3.3 Multimodal Evidence Extraction

Each video is processed into discrete, modality-labeled evidence units, all represented in text form so that heterogeneous evidence can be handled under a unified interface. As a result, visual and audio signals are consumed through model-generated descriptions and transcripts rather than raw inputs. The three evidence modalities and the treatment of metadata are described below.

Visual keyframes (V). Keyframes are extracted by uniform temporal sampling at 5-second intervals, yielding up to 8 frames per video. Each sampled frame is processed by a vision-language model (Claude Sonnet 4), which generates a structured textual description including the scene, salient persons and objects, transcription of on-screen text, and visual cues relevant to fact verification. The resulting V-type unit is this textual description rather than the raw image. The description prompt asks for at most 150 characters to keep visual evidence concise and comparable across samples. This process produces 1,207 units across all samples (mean: 6.2 per sample).

ASR transcripts (A). Spoken content is transcribed using an automatic speech recognition model (Whisper base) with automatic language detection. We use the timestamped transcript segments returned by the model as the A-type evidence units. This yields 2,847 segments in total (mean: 14.6 per sample), making ASR the largest single modality by volume, as spoken content varies substantially in length across videos.

External web evidence (E). External evidence is retrieved via web search. For each claim, two rounds of queries are issued: (1) the original claim text, and (2) the claim augmented with languagematched fact-checking keywords (e.g., “fact check” or “debunk” in English, and corresponding terms in Chinese). Each round returns up to 5 results, which are then deduplicated by URL. Each retrieved result is represented as a structured E-type unit containing a title, content snippet, and source URL. This yields 1,456 units in total (mean: 7.5 per sample).

Metadata exclusion. Each video also carries a metadata unit containing the video title and platform-level contextual information. However, because the textual claim for each sample is written by the annotator after watching the video (Section 3.2), the video title frequently overlaps with or closely paraphrases the claim itself. Including such metadata as evidence would introduce circular reasoning—the verifier would efectively compare the claim against a near-duplicate of itself. We therefore exclude metadata units from the verification evidence pool. All subsequent statistics, evidence counts, and experiments refer to the three-modality pool (V, A, E) after metadata exclusion.

The complete evidence composition is summarized in Table 2.

Table 2: Evidence unit composition in MiniVer-V (metadata excluded).
<table><tr><td>Modality</td><td>Code</td><td>Total Units</td><td>Avg / Sample</td><td>Proportion</td></tr><tr><td>Visual keyframes</td><td>V</td><td>1,207</td><td>6.2</td><td>21.9%</td></tr><tr><td>ASR transcripts</td><td>A</td><td>2,847</td><td>14.6</td><td>51.7%</td></tr><tr><td>External search</td><td>E</td><td>1,456</td><td>7.5</td><td>26.4%</td></tr><tr><td>Total</td><td></td><td>5,510</td><td>28.3</td><td>100%</td></tr></table>

The number of evidence units per sample ranges from 9 to 91, with a median of 26. This unit-level decomposition enables fine-grained control over evidence selection, allowing systematic comparison of strategies that vary in both the amount and composition of evidence presented to the verifier.

## 3.4 Dataset Statistics and Comparison

Evidence counts vary substantially across samples (9–91 units; mean 28.3, median 26), with the right-skew driven mainly by ASR length. Internal evidence (V+A) accounts for 73.6% of all units, making modality imbalance a practical challenge for evidence selection.

Table 3 compares MiniVer-V with representative fact-checking benchmarks across key design dimensions.

Table 3: Comparison of MiniVer-V with representative fact-checking benchmarks. Selection focus: whether evidence selection is the primary object of evaluation. <sup>∗</sup>FakeSV: fake and real videos used for binary detection (5,538 videos including debunking videos).
<table><tr><td>Benchmark</td><td>Media</td><td>Samples</td><td>Lang</td><td>Labels</td><td>Evidence (granularity)</td><td>Selection focus</td></tr><tr><td>FEVER</td><td>Text</td><td>~185K</td><td>EN</td><td>3-way</td><td>Text sentences</td><td>No</td></tr><tr><td>AVeriTeC</td><td>Text</td><td>~4.6K</td><td>EN</td><td>4-way</td><td>QA pairs + URLs</td><td>No</td></tr><tr><td>MOCHEG</td><td>Multimodal web</td><td>~15.6K</td><td>EN</td><td>3-way</td><td>Paragraphs + images</td><td>No</td></tr><tr><td>FakeSV</td><td>Short video</td><td>~3.7K*</td><td>ZH</td><td>Binary</td><td>V + A + M (per sample)</td><td>No</td></tr><tr><td>MiniVer-V</td><td>Short video</td><td>195</td><td>ZH/EN</td><td>3-way</td><td>V + A + E (per unit)</td><td>Yes</td></tr></table>

MiniVer-V combines design elements that appear separately in prior benchmarks—three-way labels, short-video content, and web evidence—but difers in making unit-level evidence selection the primary object of study. Although smaller in sample count than existing benchmarks, MiniVer-V is intended as an evaluation benchmark for controlled evidence-selection experiments rather than a large-scale training resource [8]. Here, the key experimental variation lies in the selectable evidence units and the subsets induced per sample.

## 3.5 Evidence Quality Control and Annotation

A random 20% of samples undergo manual inspection to verify extraction quality; units that fail are re-extracted or corrected. Each evidence unit is additionally assigned an automatically generated relation label (support, refute, or irrelevant) with respect to the claim, produced by the backbone model with a 20% human audit. These relation labels are not used as input to any verification strategy; they serve as metadata for post-hoc analysis.

## 3.6 Human Annotation of Reference Evidence Sets

To validate automated evidence selection, three graduate-student annotators independently select compact evidence subsets they consider suficient for the ground-truth verdict. Annotators are shown the claim, ground-truth verdict, and all evidence units grouped by modality. We obtain annotations for all 195 samples. For evaluation, we construct a single reference set for each sample by retaining evidence units selected by at least two annotators. Quantitative comparison with B5 is reported in Section 5.5.

## 4 Method

## 4.1 Overview

Given a claim c and its associated set of evidence units $\mathcal { E } = \{ e _ { 1 } , e _ { 2 } , \ldots , e _ { n } \}$ extracted in Section 3.3, the verification task proceeds in three stages: (1) relevance scoring and candidate pool construction, (2) evidence selection, and (3) two-layer verdict prediction. The output is a selected subset ${ \mathcal { S } } \subseteq { \mathcal { E } }$ together with a factual verdict. Figure 1 illustrates the overall architecture.

We instantiate the evidence selection stage with five strategies that form a progression of increasing sophistication. B1 (Full Evidence) uses all available units without selection, serving as the full-evidence reference. B2 (Top-K) selects a fixed number of highest-scoring units, testing whether relevance ranking alone yields a suficient subset. B3 (CoT Self-Selection) asks the model to select a subset via chain-of-thought reasoning. B4 (Greedy-Suficient) introduces a suficiency-aware search loop that dynamically determines how many units are needed per sample, followed by redundancy pruning. B5 (Greedy + Abstention), our proposed method, extends B4 by outputting insuficient when the candidate pool is exhausted without reaching suficiency, rather than forcing a verdict. This progression—from no selection, through fixed-budget and model-driven selection, to suficiency-driven dynamic selection with explicit abstention—is designed to isolate the contribution of each design choice.

![](images/bad7fd033302b96a31561e4192ef63734e170ef672fda0cd6bb3cac56f985403.jpg)  
Figure 1: Overview of the MiniVer-V verification framework. Given a claim and a multimodal evidence pool (visual keyframes, speech transcripts, and web-retrieved external sources), five evidence selection strategies of increasing sophistication are evaluated. B5, our proposed method, performs suficiency-driven greedy search with explicit abstention, feeding the selected compact evidence subset into a two-layer verifier that disentangles claim–video consistency (Layer 1) from factual verdict determination (Layer 2).

All five strategies share the same backbone model, prompt family, and two-layer verifier; they difer only in how evidence is selected, with B5 additionally introducing an abstention gate.

## 4.2 Relevance Scoring and Candidate Pool Construction

For strategies that rely on explicit evidence ranking (B2, B4, and B5), we first score each evidence unit for relevance to the claim. Given a claim c and the full evidence set E, all units are scored jointly in a single pass: the model assigns each unit $e _ { i }$ a pointwise relevance score $r _ { i } \in [ 0 , 1 ]$ , reflecting how directly the unit relates to verifying the claim. The same scoring criterion is applied uniformly across modalities, so that visual descriptions, ASR segments, and external search results are compared on a common scale. No post-hoc normalization is applied. Units are then sorted by score in descending order to produce a ranked list shared by B2, B4, and B5. Scores are used strictly for ranking and candidate selection; they do not enter the verification stage as features or weights.

For B4 and B5, the ranked list is further refined into a balanced candidate pool of at most 10 units, a budget chosen to balance evidence coverage against prompt length. To prevent single-modality dominance—particularly by ASR segments, which account for over half of all units (Section 3.4)—the pool is constructed to include at least 2 internal units (V or A) and at least 3 external units (E), with remaining slots filled by the highest-scoring units regardless of modality. The higher minimum for external units reflects the finding that external evidence is critical for factual verdict determination (Section 5.3), while internal evidence primarily supports claim–video consistency assessment. This constraint ensures that the suficiency search operates over both internal and external evidence types.

## 4.3 Evidence Selection Strategies

## 4.3.1 Baseline Strategies

B1 (Full Evidence) presents all available units to the verifier without selection, serving as the full-evidence reference. External units are capped at 8 to limit prompt length. B2 (Top-K) selects the k highest-scoring units from the ranked list and passes them directly to the verifier; we evaluate $k = 3 , 5 , 7$ to trace the relationship between evidence budget and verification performance. B3 (CoT Self-Selection) provides all evidence units and instructs the model to analyze each unit’s relationship to the claim, select a subset it deems suficient, and produce a verdict based on that subset—all within a single inference pass.

## 4.3.2 Suficiency-Driven Selection

The preceding strategies either use all evidence or select by relevance score, but none explicitly models whether the selected subset is suficient for a confident verdict. We define suficiency operationally: a selected evidence subset S is treated as suficient when the verifier judges it adequate to support a verdict with confidence above a predefined threshold τ. We set $\tau = 0 . 8$ as a conservative threshold.

B4 (Greedy-Suficient) operates in three phases on the balanced candidate pool (Section 4.2):

Phase 1: Greedy Addition. Starting from an empty set, units are added one at a time in descending relevance order. After each addition, a suficiency check queries the model: given the current subset and the claim, is the evidence suficient to reach a confident verdict? If the model returns suficient with confidence $\geq \tau$ , addition stops. If the entire candidate pool is exhausted without reaching suficiency, all candidates proceed to the next phase.

Phase 2: Redundancy Pruning. Each unit in the selected set is tested for removal: if the remaining units still satisfy the suficiency criterion, the unit is permanently discarded as redundant. This phase yields a compact suficient subset under the greedy pruning procedure—no single remaining unit can be removed without falling below the suficiency threshold, though the result is not guaranteed to be globally minimal.

Phase 3: Final Verification. The pruned subset is passed to the two-layer verification framework (Section 4.4) to produce the final verdict.

B5 (Greedy + Abstention) follows the same procedure as B4, with one key modification: if Phase 1 exhausts the candidate pool without reaching suficiency, the system outputs insuficient as its verdict and skips Phases 2 and 3. This mechanism allows the system to explicitly acknowledge when the available evidence is inadequate, rather than forcing a verdict from an insuficient evidence base.

## 4.4 Two-Layer Verification

All strategies produce their final verdict through a shared two-layer verification framework.

Layer 1: Internal-Evidence Grounding. Using only the selected internal evidence units (V and A), the model assesses whether the claim is compatible with the video-derived evidence representation. The output is one of three labels: consistent, inconsistent, or unclear. Because both the claim and the internal evidence are derived from the same underlying video (Section 3.2), this layer is not intended as a strong discriminative module on its own; rather, it encourages the verifier to anchor its reasoning in video-internal content before consulting external sources, and provides an interpretable intermediate signal about internal alignment.

Layer 2: Factual Verdict. Using the full selected subset S, including external evidence (E units), the model determines the factual status of the claim. The output is supported, refuted, or insuficient, accompanied by a confidence score and reasoning. The final verdict is always determined by this layer.

The two layers are executed jointly in a single inference call. Decoupling internal grounding from factual verification reflects the observation that internal compatibility does not imply factual truth—a video may faithfully depict content described by the claim while the underlying event is fabricated or misattributed—and that external evidence is necessary to bridge this gap.

One exception to this flow arises in B5: when the selection stage fails to reach suficiency (Section 4.3), the system outputs insuficient directly, bypassing the two-layer verification entirely.

## 5 Experiments

## 5.1 Experimental Setup

Model and configuration. All five strategies use the same backbone model—Claude Sonnet 4 (claude-sonnet-4-20250514, Anthropic)—for relevance scoring, suficiency checking, and verdict prediction. The temperature is set to 0.1 to reduce output variance, and the maximum generation length is 4,096 tokens. No strategy-specific tuning is applied: prompts difer only in the evidence subset provided and the task instruction (scoring, suficiency check, or verification). In all prompts, each evidence unit is truncated to its first 150 characters (100 characters for relevance scoring), for every modality and strategy. Section 5.6 repeats the pipeline with two further verifiers.

Evaluation metrics. We report five metrics. Macro-F1 is the primary metric, computed as the unweighted average of per-class F1 scores across the three verdict categories (supported, refuted, insuficient). Because the label distribution is imbalanced (Section 3.2), Macro-F1 gives equal weight to each category regardless of its sample count, penalizing methods that sacrifice minorityclass performance for majority-class gains. Accuracy is the fraction of correctly classified samples. Abstention Precision is the precision of the insuficient prediction: among all samples predicted as insuficient, the fraction that are truly insuficient. Abstention Recall is the corresponding recall: among truly insuficient samples, the fraction correctly predicted. Overconfidence Rate is the complement of Abstention Recall (1 − Abstention Recall), measuring the proportion of truly insuficient samples that receive a non-insuficient verdict—a proxy for the system’s tendency to fail to abstain when evidence is genuinely inadequate.

We additionally report average evidence units per sample for each strategy, as evidence eficiency is a central concern of this work.

## 5.2 Main Results

Table 4 presents the main results across all five strategies.

We assess diferences with a paired bootstrap over the 195 videos (2,000 resamples); full results, including the cross-model backbones, are in Appendix C.3.

Suficiency-driven selection with abstention matches full-evidence performance at a fraction of the cost. B5, our proposed method, achieves a Macro-F1 of 0.510 using an average of 4.5 evidence units—approximately 16% of the full evidence set—recovering 98.5% of the fullevidence baseline B1’s performance (0.518 with 27.7 units). The diference between B5 and B1 is not statistically significant (∆Macro-F1 = −0.008, 95% CI [−0.08, +0.06]): B1’s slight advantage comes at over 6× the evidence cost. Among eficient methods (B2 through B5), B5 attains the highest Macro-F1, ahead of B2–B4 by 0.04 to 0.06. These margins are consistent in sign, but at n=195 none of them is statistically significant.

Table 4: Main results (195 samples, Claude Sonnet 4). The lower block applies confidence-threshold abstention to three base strategies: any committed verdict with confidence below $\tau = 0 . 8$ (the threshold B4 and B5 use for the suficiency check) is converted to insuficient. Bold marks the highest Macro-F1 among eficient methods (≤10 units).
<table><tr><td>Method</td><td>Macro-F1</td><td>Accuracy</td><td>Avg Units</td><td>Abst. Prec.</td><td>Abst. Recall</td><td>OverConf.</td></tr><tr><td>B1 Full</td><td>0.518</td><td>56.9%</td><td>27.7</td><td>0.500</td><td>0.254</td><td>0.746</td></tr><tr><td>B2 Top-3</td><td>0.466</td><td>49.7%</td><td>3.0</td><td>0.378</td><td>0.525</td><td>0.475</td></tr><tr><td>B2 Top-5</td><td>0.459</td><td>50.3%</td><td>5.0</td><td>0.370</td><td>0.339</td><td>0.661</td></tr><tr><td>B2 Top-7</td><td>0.460</td><td>51.3%</td><td>7.0</td><td>0.350</td><td>0.237</td><td>0.763</td></tr><tr><td>B3 CoT</td><td>0.453</td><td>57.4%</td><td>9.4</td><td>1.000</td><td>0.051</td><td>0.949</td></tr><tr><td>B4 Greedy</td><td>0.469</td><td>52.3%</td><td>4.1</td><td>0.333</td><td>0.203</td><td>0.797</td></tr><tr><td>B5 Abstention</td><td>0.510</td><td>54.4%</td><td>4.5</td><td>0.377</td><td>0.390</td><td>0.610</td></tr><tr><td colspan="7">Confidence-threshold abstention (τ = 0.8)</td></tr><tr><td>B1  $\mathrm { F u l l } + \tau$ </td><td>0.525</td><td>57.4%</td><td>27.7</td><td>0.516</td><td>0.271</td><td>0.729</td></tr><tr><td> $\mathrm { B 2 \ T o p { - } 5 + \Delta \tau }$ </td><td>0.464</td><td>50.8%</td><td>5.0</td><td>0.382</td><td>0.356</td><td>0.644</td></tr><tr><td>B4  ${ \mathrm { G r e e d y } } + \tau$ </td><td>0.476</td><td>52.8%</td><td>4.1</td><td>0.351</td><td>0.220</td><td>0.780</td></tr></table>

Fixed-budget relevance selection ofers no principled budget. The B2 Top-K variants cluster below 0.47 Macro-F1 for every k (Top-3 0.466, Top-5 0.459, Top-7 0.460), while their abstention behavior shifts sharply with k (Abstention Recall $0 . 5 2 5  0 . 2 3 7 )$ , leaving no principled way to choose k. We attribute this to relevance to the claim not guaranteeing suficiency for verification—a highly relevant unit may duplicate information already present in the selected set, while a moderately relevant unit may provide the missing factual link needed for a verdict. B5 eliminates the need to pre-specify k by dynamically determining how much evidence is enough on a per-sample basis.

CoT self-selection exhibits systematic overcommitment. B3 achieves the highest accuracy among the base strategies (57.4%) but the lowest Macro-F1 (0.453), exposing a strong tendency toward unjustified non-abstention. B3 predicts insuficient for only 3 out of 195 samples, yielding an Overconfidence Rate of 94.9%—the highest among all methods. It correctly classifies 84/97 supported cases (86.6%) but only 3/59 insuficient cases (5.1%). When the model self-selects evidence and produces a verdict in a single pass, it consistently overestimates the adequacy of its evidence, committing to a verdict even when the evidence is genuinely inadequate. This finding underscores the need for an explicit abstention mechanism rather than leaving abstention to the model’s own judgment during verdict generation.

Explicit abstention improves recognition of insuficient cases. Comparing B4 and B5 isolates the contribution of the abstention gate. B5 improves Macro-F1 from 0.469 to 0.510 (+0.041, not significant at n=195), with the gain concentrated in the insuficient class: insuficient-class F1 rises from 0.253 to 0.383 (+0.131, 95% CI [+0.03, +0.23], p=0.010), and Abstention Recall nearly doubles, from 0.203 to 0.390 (p<0.001). Without abstention, B4 is forced to commit to a verdict even when the candidate pool fails to reach suficiency, producing overconfident errors on dificult samples. The abstention mechanism converts these potential misclassifications into calibrated insuficient outputs.

Confidence thresholding is a competitive alternative abstention mechanism. A simpler way to abstain is to threshold the verifier’s self-reported confidence. With $\tau = 0 . 8$ , the threshold B4 and B5 already use for the suficiency check, thresholding raises Macro-F1 for every base strategy (lower block of Table 4): B1 to 0.525, B2 Top-5 to 0.464, and B4 to 0.476. B5 is statistically indistinguishable from thresholded full evidence (∆Macro-F1 = −0.015, 95% CI $[ - 0 . 0 9 , + 0 . 0 5 ] )$ ) while using 16% of the evidence, and its insuficient-class F1 is significantly higher than that of thresholded B4 $( + 0 . 1 1 3 , p { = } 0 . 0 2 8 )$ . Tuning τ by 5-fold cross-validation does not change this picture: B1 + τ reaches $0 . 5 3 4 \pm 0 . 0 1 4$ and $\mathrm { B 4 } + \tau \ 0 . 4 7 1 \pm 0 . 0 1 3$ over 20 random splits, and neither difers significantly from B5. Explicit abstention therefore matters more than its specific form—both the suficiency gate and a confidence threshold counter the overcommitment of B3 and B4. What distinguishes B5 is that it reaches full-evidence performance with a small evidence subset that is itself judged suficient for the verdict, so each abstention is tied to an explicit, inspectable failure to reach suficiency.

Per-class breakdown. B2 Top-3 achieves the highest insuficient accuracy (52.5%) but at the cost of much lower supported and refuted accuracy (54.6% / 33.3%). B5 reaches 39.0% on insuficient cases while maintaining stronger supported and refuted accuracy (68.0% / 43.6%), yielding the most balanced profile.

## 5.3 Ablation Studies

To isolate the contribution of individual components and evidence sources, we conduct five ablation experiments (Table 5). The abstention gate is ablated by comparing B5 with B4, which is identical except that it never abstains. The evidence-source and pruning ablations each remove one component from B4. We run them without the abstention gate because the gate outputs insuficient whenever the candidate pool is exhausted: with a weakened evidence pool, it would turn evidence loss into abstention by construction, and the measured efect would reflect the gate rather than the verifier’s verdicts. ∆F1 is therefore reported relative to B5 for the abstention ablation and relative to B4 for all others, with 95% confidence intervals from a paired bootstrap over the 195 videos (2,000 resamples).

Table 5: Ablation study results. The abstention gate is ablated by comparing B5 with B4; all other components are ablated on B4 (the selection pipeline without the abstention gate), and their ∆F1 is relative to B4. Brackets give 95% paired-bootstrap confidence intervals.
<table><tr><td>Setting</td><td>Macro-F1</td><td>Acc.</td><td>Units</td><td></td><td>∆F1 [95% CI]</td></tr><tr><td>B5 (proposed) w/o Abstention (= B4)</td><td>0.510 0.469</td><td>54.4% 52.3%</td><td>4.5 4.1</td><td></td><td>-0.041 [−0.10, +0.02]</td></tr><tr><td>Component ablations on B4 (∆F1 relative to B4)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>0.273</td><td>32.3%</td><td></td><td></td><td></td></tr><tr><td>w/o External</td><td></td><td></td><td>7.3</td><td></td><td>-0.196 [−0.28, -0.11]</td></tr><tr><td> $\mathrm { w / o \ V + A \ ( E { \cdot } o n l y ) }$ </td><td>0.484</td><td>53.3%</td><td>3.5</td><td></td><td>+0.016 [−0.05, +0.08]</td></tr><tr><td>w/o ASR</td><td>0.465</td><td>51.3%</td><td>4.6</td><td></td><td>−0.004 [−0.07, +0.06]</td></tr><tr><td>w/o Pruning</td><td>0.477</td><td>52.8%</td><td>5.4</td><td></td><td>+0.008 [−0.04, +0.05]</td></tr></table>

Abstention improves insuficient-case recognition. Removing the abstention gate lowers Macro-F1 from 0.510 to 0.469. This overall diference is not statistically significant at n=195, but the efect is concentrated where the gate is designed to act: insuficient-class F1 and Abstention Recall both drop significantly (Section 5.2).

External evidence is indispensable $( \Delta { \bf F 1 } = - { \bf 0 . 1 9 6 } )$ . Removing external web-retrieved evidence causes the most severe performance collapse across all experiments, and the only significant one among the component ablations. Macro-F1 drops from 0.469 to 0.273, and the verifier returns insuficient for 153 of 195 samples (78.5%) even though B4 has no abstention gate. The Layer 1 outputs show why: with internal evidence alone, the verifier judges 134 claims (68.7%) consistent with the video, yet still cannot determine whether they are true. Average evidence consumption rises from 4.1 to 7.3 units, as the search frequently exhausts the candidate pool without reaching the suficiency threshold. Claim–video consistency is therefore no substitute for external corroboration, which is the single most critical component for factual verdict determination in short-video verification.

Internal evidence grounds the verdict rather than determining it $\begin{array} { r } { ( \Delta \mathbf { F 1 } = + \mathbf { 0 . 0 1 6 } . } \end{array}$ n.s.). Removing visual keyframes and ASR transcripts (E-only) leaves Macro-F1 statistically unchanged (0.484 vs. 0.469), indicating that external evidence carries the dominant signal for verdict determination. Verdict accuracy, however, does not capture the role of internal evidence: in the E-only setting, all 195 samples receive unclear as their Layer 1 consistency assessment, whereas with internal evidence available, B4 reaches a definite consistency judgment for 96 samples (49.2%). Without internal evidence, the verdict is reached with no link to the video’s content. Internal evidence is thus what makes the two-layer framework interpretable: it anchors the verifier’s reasoning in video-derived content, which is a prerequisite for recognizing videos that are internally consistent with their claims yet factually misleading.

Removing ASR transcripts alone likewise leaves Macro-F1 unchanged $( \Delta \mathrm { F 1 } = - 0 . 0 0 4 , \mathrm { n . s . } )$ . ASR segments help establish what the video claims but rarely provide independent factual corroboration, so their contribution lies in internal grounding rather than factual resolution.

Pruning improves eficiency without loss of accuracy $( \Delta \mathbf { F 1 } = + \mathbf { 0 . 0 0 8 } , \mathbf { n . s . } )$ . Skipping the redundancy pruning phase leaves Macro-F1 statistically unchanged (0.477 vs. 0.469) but increases average evidence from 4.1 to 5.4 units. Pruning thus removes 24% of the selected units while preserving verdict quality: the discarded units are redundant, since the remaining subset still satisfies the suficiency criterion. This is precisely the property that minimal suficient evidence identification targets—a compact subset that supports the same verdict as a larger one.

## 5.4 Evidence Eficiency Analysis

A central claim of this work is that a small, suficiency-aware evidence subset can recover most of the performance achieved by full evidence. Figure 2 illustrates this trade-of across all strategies.

B5 operates at a favorable point on the eficiency–performance frontier: it uses 4.5 units (16% of B1’s 27.7) while recovering 98.5% of B1’s Macro-F1. By contrast, B2 Top-7 uses 7.0 units (25%) but achieves only 88.8% of B1’s performance, and B3 uses 9.4 units (34%) while achieving only 87.5%. The suficiency-driven approach dynamically allocates evidence per sample—easy cases are resolved with 1–2 units, while genuinely dificult cases consume the full candidate pool—rather than applying a uniform budget across all samples.

The ablation results reinforce this picture: removing external evidence forces the pipeline to consume 7.3 units on average while achieving only 52.7% of B1’s Macro-F1, demonstrating that evidence quantity cannot compensate for the absence of the right evidence type. Conversely, removing pruning increases evidence by 31% (4.1 → 5.4) with no significant change in Macro-F1, confirming that the units removed by pruning are redundant: compactness is obtained without sacrificing accuracy.

## 5.5 Human Validation of Evidence Selection

The preceding sections evaluate evidence selection through verdict accuracy, but a natural question remains: does B5 select the right evidence? We address this by comparing B5’s automatically selected evidence sets against the human annotations described in Section 3.6. Table 6 summarizes the comparison across all 195 samples. The high level of overlap suggests that B5’s suficiency-driven selection is broadly consistent with human judgment, though the two are not identical because multiple compact evidence subsets can support the same verdict.

![](images/3fd095abe175fb762759f1b905000132e9d2902e58dcbe632417d5cc2d92994a.jpg)  
Figure 2: Macro-F1 vs. average evidence units per sample across all strategies. B5 (star) achieves near-full-evidence performance (98.5% of B1) using only 16% of the evidence. Hollow markers apply confidence-threshold abstention $( \tau = 0 . 8 )$ to B1, B2 Top-5, and B4. The dashed line marks B1’s full-evidence Macro-F1.

Table 6: Comparison of B5’s evidence selection against human annotations (195 samples).
<table><tr><td>Metric</td><td>Value</td></tr><tr><td>Avg Jaccard similarity Avg Precision (B5 ∩ human / B5)</td><td>0.882</td></tr><tr><td>Avg Recall (B5 ∩ human / human)</td><td>0.899</td></tr><tr><td></td><td>0.970</td></tr><tr><td>Exact match rate</td><td>71.8%</td></tr><tr><td>Human  $\subseteq { \mathrm { B 5 } }$  rate</td><td>92.8%</td></tr><tr><td> $\operatorname { A v g }$  human set size</td><td>3.74</td></tr><tr><td> $\operatorname { A v g }$  B5 set size</td><td>4.46</td></tr></table>

B5 shows substantial overlap with human annotations. The average Jaccard similarity between B5’s selected sets and human annotations is 0.882, with 71.8% of samples achieving an exact match. B5’s recall with respect to the human annotations is 0.970, indicating that it recovers most of the evidence units identified by human annotators, while its precision of 0.899 suggests that the automated method retains only a small number of additional units. The size diference is modest: B5 selects 4.46 units on average versus 3.74 for human annotators.

Human annotations also highlight the importance of external evidence. The modality composition of human-annotated sets is 51.6% external (E), 32.5% ASR (A), and 15.9% visual (V). This distribution closely mirrors the ablation findings in Section 5.3: external evidence is the most frequently selected modality in both automated and human selections, reinforcing its indispensable role in factual verdict determination.

## 5.6 Cross-Model Generalization

To test whether the findings depend on the verifier, we rerun the full pipeline—relevance scoring, suficiency search, and two-layer verification—with GPT-5.5 (closed-weight, diferent provider) and Qwen2.5-72B-Instruct (open-weight) on the identical frozen evidence pool: the same 5,510 units, including the Claude-generated V descriptions. Because the verifier now difers from the description model, this also tests whether the results rely on one model both producing and consuming the visual descriptions. Qwen covers 193 of the 195 videos; the provider’s content-safety filter declined two samples about disasters and violent incidents. Table 7 summarizes the results; full results and significance tests are in Appendices C.2 and C.3.

Table 7: Cross-model results on the identical frozen evidence pool. B5/B1: share of B1’s Macro-F1 recovered by B5. Units: B5’s average evidence units (share of B1’s). Ins-F1: insuficient-class F1.
<table><tr><td>Verifier</td><td>B1 F1</td><td>B5 F1</td><td>B5/B1</td><td>Units</td><td>Ins-F1 B4→B5</td></tr><tr><td>Claude Sonnet 4</td><td>0.518</td><td>0.510</td><td>98.5%</td><td>4.5 (16%)</td><td>0.253→0.383</td></tr><tr><td>GPT-5.5</td><td>0.586</td><td>0.559</td><td>95.3%</td><td>4.9 (18%)</td><td>0.407→0.508</td></tr><tr><td>Qwen2.5-72B</td><td>0.580</td><td>0.489</td><td>84.3%</td><td>2.4 (9%)</td><td>0.413→0.423</td></tr></table>

Full-evidence Macro-F1 lies in the same band for all three verifiers (0.52–0.59), so the absolute level tracks verifier capability rather than a property of one model. The eficiency result replicates on GPT-5.5: B5 recovers 95.3% of B1’s Macro-F1 with 18% of the evidence, with no significant diference from B1, and again improves insuficient-class F1 over B4 (+0.100, p=0.025). On Qwen, B5 recovers 84.3% of B1’s Macro-F1 with 9% of the evidence but is significantly below both B1 and Top-5, and its abstention gain is small. Qwen is a coarse suficiency judge: it ends with a single evidence unit on 60% of samples, compared with 28% for GPT-5.5 and 17% for Claude, so its small evidence sets reflect premature stopping rather than genuine minimality. Suficiency-driven selection therefore transfers across providers, but its benefit depends on the quality of the verifier’s suficiency judgments.

## 5.7 Robustness and Error Analysis

Textualized vs. raw visual evidence. Because V-units are text descriptions, we test whether textualization discards verification-relevant information. On a stratified subset of 60 videos (20 per verdict class), we rerun B1 with the same verifier (Claude Sonnet 4), replacing each V-unit with its keyframe image and leaving ASR and external evidence unchanged, so that only the form of the visual channel difers. Raw keyframes do not improve verification: Macro-F1 is 0.561 with frames and 0.585 with descriptions (∆Macro-F1 = −0.024, 95% CI [−0.14, +0.09], p=0.69), and the two settings agree on 78.3% of verdicts. Within the current claim-verification setting, the concise descriptions thus preserve most of the visual information the verifier uses. This control isolates information loss in the visual channel; it does not cover reasoning over motion or temporal dynamics, which our text-based selection does not model (Section 6.4).

Speech language and evidence quantity. We classify a video as English-speech when CJK characters make up at most 10% of its ASR transcript. B5 performs similarly on Chinese-speech videos (Macro-F1 0.510, n=155) and English-speech videos (0.500, n=40), as does B1 (0.509 and 0.540); the English group is small, so these figures are indicative only. Insuficiency is also not an artifact of evidence quantity: the average number of available units is nearly identical across verdict classes (28.1 supported, 28.3 refuted, 28.5 insuficient).

Error analysis. B5 misclassifies 89 of the 195 videos, and most errors involve the insuficient class. The largest group is missed abstention (36): insuficient cases that receive a committed verdict, 30 of them supported, after B5 judges a small set (3.6 units on average) suficient. Over-abstention on supported claims accounts for 23 errors and refuted claims labeled insuficient for 15; in 22 of these 38 cases, the search exhausts the candidate pool without reaching suficiency, and in the rest the final verifier itself returns insuficient. Only 15 errors confuse supported and refuted. Refuted-to-insuficient errors include claims that mix real and fabricated elements, which require synthesizing several partially contradicting units into a refutation (case study in Appendix D). The dominant failure is therefore misjudged suficiency rather than polarity: the verifier accepts inadequate evidence too early on some samples and fails to find decisive evidence within the pool on others.

## 6 Discussion

## 6.1 The Asymmetric Roles of Internal and External Evidence

Our ablation results reveal a striking asymmetry in how diferent evidence modalities contribute to verification. Removing external evidence collapses Macro-F1 by 0.196, the largest single-component efect, whereas removing internal evidence (V+A) leaves it statistically unchanged. At first glance, this suggests that internal evidence is nearly dispensable, but this interpretation is misleading.

In the E-only setting, every sample receives unclear as its Layer 1 consistency assessment. The system can still reach correct verdicts in many cases, but it does so without any grounding in the video’s actual content—it efectively performs text-based fact-checking using web-retrieved evidence against a textual claim, bypassing the video entirely. The two-layer framework therefore degenerates into a single-layer system, losing the intermediate signal that connects the verdict to the video.

Internal evidence thus serves a qualitatively diferent function from external evidence: it does not primarily determine what the verdict is, but rather why the verdict is connected to the video. Visual keyframes and speech transcripts anchor the verifier’s reasoning in video-derived content, in principle helping the system remain grounded in cases where a video is internally consistent with its claim yet factually misleading—such as repurposed footage or out-of-context clips. This division of labor validates the two-layer design: Layer 1 (internal grounding) and Layer 2 (factual determination) address complementary aspects of video verification that cannot be collapsed into a single step without losing interpretability.

## 6.2 External Evidence Dominance and Claim-Based Retrieval

Claim-based retrieval is a standard component of evidence-based fact-checking [9], not a methodological shortcoming: verifying a video’s factual assertion against external sources is precisely what fact-checking requires. However, the E-only ablation shows that removing internal evidence eliminates all consistency signal, confirming that internal evidence provides grounding that external retrieval cannot supply. Future work could explore video-grounded retrieval—using keyframe content or ASR segments to generate supplementary queries—to reduce dependence on the claim as the sole retrieval anchor.

## 6.3 The Necessity of Explicit Abstention

Our results consistently show that systems without explicit abstention are overconfident on genuinely insuficient cases. B5 mitigates this failure mode by converting unresolved selections into insuficient outputs rather than forcing a verdict. A confidence threshold on the verifier’s output achieves a comparable efect (Section 5.2), which suggests that the essential requirement is an explicit abstention decision rather than a particular mechanism; the suficiency gate additionally grounds each abstention in a specific evidence set that was checked and found inadequate. This is especially important in high-stakes verification settings, where calibrated uncertainty is preferable to confident error.

## 6.4 Limitations

Sample size. MiniVer-V contains 195 videos with 5,510 evidence units. The unit-level decomposition supports controlled comparisons of selection strategies within the same videos, but the number of videos limits statistical power: diferences of a few Macro-F1 points between strategies are not significant at this size (Section 5.2). It also limits the diversity of topics, platforms, claim types, and evidence configurations. Scaling MiniVer-V is a priority for future work.

Verifier and description models. We evaluate three verifiers (Claude Sonnet 4, GPT-5.5, and Qwen2.5-72B), but all V descriptions are generated by Claude Sonnet 4; testing other description models remains future work. The benefit of suficiency-driven selection also depends on the verifier’s suficiency judgments, as the weaker results with Qwen show (Section 5.6).

Textualized evidence. All evidence is consumed as text. The raw-keyframe control (Section 5.7) suggests little information loss for claim verification, but the framework does not model motion or temporal dynamics, and MiniVer-V evaluates evidence selection over textualized units rather than full video-level multimodal reasoning.

Retrieval of existing fact-checks. External evidence is retrieved with the claim text, and the second query round adds fact-checking keywords. The results can therefore include fact-checking articles about the same claim, which may state the verdict directly. We did not filter such articles or restrict retrieval to dates before a claim was checked, so the dominant role of external evidence (Section 5.3) may partly reflect access to existing fact-checks rather than independent corroboration. Filtering fact-checking sources or applying temporal cutofs is a natural next step.

Consistency evaluation. Because claims in MiniVer-V are written by annotators based on the video content, most samples are expected to be internally compatible with the video representation. As a result, the benchmark provides limited coverage of naturally inconsistent claim–video pairs (e.g., repurposed footage paired with fabricated claims), which limits evaluation of Layer 1’s discriminative power. Introducing adversarial or synthetic inconsistent samples would provide a more rigorous test of the internal grounding layer.

Language coverage. All claims are written in Chinese, and only 40 videos contain mainly English speech. Performance on other languages and in cross-lingual verification settings remains untested.

## 7 Conclusion

We introduced minimal suficient evidence set identification as a new task for short-video verification and presented MiniVer-V, a benchmark designed to study evidence suficiency rather than evidence relevance alone. Across five evidence selection strategies, our suficiency-driven method with explicit abstention recovers 98.5% of full-evidence performance using only 16% of the evidence, while improving recognition of genuinely insuficient cases. Human annotation further shows substantial overlap between B5’s selected evidence sets and human annotations (Jaccard 0.882). These findings indicate that evidence-eficient, calibrated short-video verification depends not on presenting more evidence, but on selecting enough evidence, grounding predictions in video-internal content, and abstaining when that threshold is not met. Future work includes scaling MiniVer-V to a larger and more diverse sample set, testing description models beyond Claude, and exploring video-grounded retrieval strategies.

## Ethics and Data Availability

Ethics statement. The videos in MiniVer-V were collected from publicly available posts on social media platforms and are used solely for non-commercial research. All annotations—claims, verdict labels, and reference evidence sets—were produced by the three authors; no crowdsourcing platform was used. Because the dataset contains news and public-afairs content, we reviewed the samples to ensure that no personally identifiable information is disclosed beyond what is already public in the original posts. The benchmark is intended to support fact-checking research, not to target individuals or communities.

Data availability. We plan to release the MiniVer-V evidence units, verdict annotations, human-annotated reference evidence sets, and the code for evidence extraction, selection, and evaluation. Raw video files will not be redistributed; we will provide source references for obtaining the videos, subject to each platform’s terms of service.

## References

[1] Pepa Atanasova, Jakob Grue Simonsen, Christina Lioma, and Isabelle Augenstein. Fact checking with insuficient evidence. Transactions of the Association for Computational Linguistics, 10: 746–763, 2022.

[2] Tobias Braun, Mark Rothermel, Marcus Rohrbach, and Anna Rohrbach. DEFAME: Dynamic evidence-based FAct-checking with multimodal experts. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 5383–5417. PMLR, 2025.

[3] Yuyan Bu, Qiang Sheng, Juan Cao, Peng Qi, Danding Wang, and Jintao Li. FakingRecipe: Detecting fake news on short video platforms from the perspective of creative process. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 1351–1360. ACM, 2024. doi: 10.1145/3664647.3680663.

[4] Zhijiang Guo, Michael Schlichtkrull, and Andreas Vlachos. A survey on automated fact-checking. Transactions of the Association for Computational Linguistics, 10:178–206, 2022.

[5] Zhenghao Liu, Chenyan Xiong, Maosong Sun, and Zhiyuan Liu. Fine-grained fact verification with kernel graph attention network. In Proceedings of the Annual Meeting of the Association for Computational Linguistics, pages 7342–7351, 2020.

[6] Liangming Pan, Xiaobao Wu, Xinyuan Lu, Anh Tuan Luu, William Yang Wang, Min-Yen Kan, and Preslav Nakov. Fact-checking complex claims with program-guided reasoning. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 6981–7004. Association for Computational Linguistics, 2023. doi: 10.18653/v1/2023.acl-long.386.

[7] Peng Qi, Yuyan Bu, Juan Cao, Wei Ji, Ruihao Shui, Junbin Xiao, Danding Wang, and Tat-Seng Chua. FakeSV: A multimodal benchmark with rich social context for fake news detection

on short video platforms. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 14444–14452, 2023. doi: 10.1609/aaai.v37i12.26689.

[8] Jonathan Roberts, Kai Han, Neil Houlsby, and Samuel Albanie. SciFIBench: Benchmarking large multimodal models for scientific figure interpretation. In Advances in Neural Information Processing Systems, Datasets and Benchmarks Track, volume 37, pages 18695–18728. Curran Associates, Inc., 2024.

[9] Michael Schlichtkrull, Zhijiang Guo, and Andreas Vlachos. AVeriTeC: A dataset for real-world claim verification with evidence from the web. In Advances in Neural Information Processing Systems, Datasets and Benchmarks Track, volume 36, pages 65128–65167. Curran Associates, Inc., 2023.

[10] Rui Shao, Tianxing Wu, and Ziwei Liu. Detecting and grounding multi-modal media manipulation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6904–6913, 2023. doi: 10.1109/CVPR52729.2023.00667.

[11] Freda Shi, Xinyun Chen, Kanishka Misra, Nathan Scales, David Dohan, Ed H. Chi, Nathanael Schärli, and Denny Zhou. Large language models can be easily distracted by irrelevant context. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 31210–31227. PMLR, 2023.

[12] James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal. FEVER: a large-scale dataset for fact extraction and VERification. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 809–819, New Orleans, Louisiana, 2018. Association for Computational Linguistics. doi: 10.18653/v1/N18-1074.

[13] Bingbing Wen, Jihan Yao, Shangbin Feng, Chenjun Xu, Yulia Tsvetkov, Bill Howe, and Lucy Lu Wang. Know your limits: A survey of abstention in large language models. Transactions of the Association for Computational Linguistics, 13:529–556, 2025.

[14] Barry Menglong Yao, Aditya Shah, Lichao Sun, Jin-Hee Cho, and Lifu Huang. End-to-end multimodal fact-checking and explanation generation: A challenging dataset and models. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 2733–2743. ACM, 2023. doi: 10.1145/3539618.3591879.

[15] Zhi Zeng, Yifei Yang, Jiaying Wu, Xulang Zhang, Xiangzheng Kong, Herun Wan, Zihan Ma, and Minnan Luo. From manipulation to mistrust: Explaining diverse micro-video misinformation for robust debunking in the wild. In Proceedings of the ACM Web Conference 2026, pages 7621–7632. ACM, 2026. doi: 10.1145/3774904.3792721.

[16] Fanrui Zhang, Dian Li, Qiang Zhang, Jun Chen, Gang Liu, Junxiong Lin, Jiahong Yan, Jiawei Liu, and Zheng-Jun Zha. Fact-R1: Towards explainable video misinformation detection with deep reasoning. In Advances in Neural Information Processing Systems, volume 38. Curran Associates, Inc., 2025. doi: 10.52202/085713-5438.

[17] Liwen Zheng, Chaozhuo Li, Xi Zhang, Yu-Ming Shang, Feiran Huang, and Haoran Jia. Evidence retrieval is almost all you need for fact verification. In Findings of the Association for Computational Linguistics: ACL 2024, pages 9274–9281. Association for Computational Linguistics, 2024. doi: 10.18653/v1/2024.findings-acl.551.

## A Prompt Templates

All prompts follow the same instruction format and were implemented in Chinese, matching the dominant language of the dataset. We present English translations of the four core prompt templates used across all five baseline strategies.

## A.1 Two-Layer Verification Prompt

This prompt is shared by all five strategies (B1–B5) for final verdict prediction. The {claim} and {evidence\_text} placeholders are filled at runtime.

You are a professional fact-checker. Please make a two-layer judgment based on the following evidence.

Claim: {claim}

Evidence: {evidence\_text}

Task: Please perform two layers of analysis:

Layer 1: Claim–Video Consistency. Using only internal video evidence (V-type visual descriptions + A-type speech transcripts), determine whether the claim is consistent with the video content: “consistent” (matches), “inconsistent” (contradicts), or “unclear” (cannot judge from internal evidence alone).

Layer 2: Factual Verdict. Using all evidence (including E-type external search results), determine the factual status of the claim: “supported” (evidence supports the claim), “refuted” (evidence disproves the claim), or “insuficient” (evidence is inadequate for a reliable judgment).

Note: Video consistency does not imply truth (the video may be repurposed footage); video inconsistency does not imply falsehood (the claim may be loosely worded but factually correct). External evidence can overturn preliminary judgments based solely on video content.

Output in JSON format:

{"consistency": "...", "verdict": "...",

```jsonl
"confidence": 0.0–1.0, "reasoning": "..."}
```

## A.2 Relevance Scoring Prompt

Used by B2 (Top-K), B4, and B5 to score all evidence units for relevance in a single batch call.

Given a claim and multiple evidence units, rate each unit’s relevance to the claim (0–1).

Claim: {claim}

Evidence list: {evidence\_list}

Output in JSON format: {"scores": [{"unit\_id": "...", "score": 0.0–1.0}]}

Output JSON only, no other content.

## A.3 Chain-of-Thought Selection Prompt

Used by B3 (CoT Self-Selection). The model analyzes, selects, and verifies in a single pass.

You are a fact-checking expert. Given a claim and candidate evidence, please:

1. Analyze each evidence unit’s relationship to the claim

2. Select a subset you consider suficient for a verdict (considering both internal and external evidence)

3. Based on the selected evidence, provide consistency and verdict judgments

Claim: {claim}

Candidate evidence: {evidence\_text}

Evidence type key: V-type (video\_segment): visual keyframe descriptions; A-type (asr): speech transcripts; E-type (external): web-retrieved evidence (Google Search)

```jsonl
Output in JSON format:
{"selected_units": [...], "consistency": "...", "verdict": "...",
"confidence": 0.0–1.0, "reasoning": "..."}
```

## A.4 Suficiency Check Prompt

Used by B4 and B5 during the greedy addition and pruning phases to assess whether the current evidence subset is suficient.

Given a claim and a set of evidence, determine whether this evidence set is suficient to support a verification conclusion.

Claim: {claim}

Evidence: {evidence\_text}

“Suficient” means: based solely on this evidence, a rational fact-checker could confidently reach a supported or refuted conclusion. Note: Suficiency requires considering both internal video evidence (V-type visual + A-type speech for consistency) and external evidence (E-type for factual verification).

Output in JSON format:   
{"is\_sufficient": true/false, "consistency": "...",   
"verdict\_if\_sufficient": "supported/refuted/null", "confidence": 0.0–1.0,   
"missing\_info": "if insufficient, what information is missing"}

## B Annotation Guidelines and Procedure

## B.1 Three-Layer Annotation Scheme

Our annotation follows a three-layer scheme, each targeting a diferent aspect of evidence quality: Layer 1 (L1): Evidence Unit Segmentation Quality. A random 20% of processed samples are inspected to verify that (i) ASR transcript segmentation boundaries are reasonable, (ii) keyframe sampling captures distinct visual content, and (iii) external search results are non-degenerate (i.e., not empty or irrelevant). Units that fail inspection are re-extracted or manually corrected.

Layer 2 (L2): Evidence–Claim Relation Labels. Each evidence unit is labeled with its relation to the claim: support (provides evidence for the claim), refute (provides evidence against the claim), or irrelevant (does not bear on the claim’s truth value). This layer is initially populated by the backbone VLM (Claude Sonnet 4) and audited by a human reviewer on a random 20% subset. These relation labels serve as analysis metadata and are not used as input to any verification strategy.

Layer 3 (L3): Minimal Suficient Evidence Set Annotation. Three graduate-student annotators independently select the smallest subset of evidence units they consider suficient to reach the ground-truth verdict. Annotators are shown the claim, ground-truth verdict, and all evidence units grouped by modality. The reference set for each sample is constructed by retaining units selected by at least two of the three annotators (majority vote).

## B.2 Annotation Quality

Inter-annotator agreement. For the ground-truth verdict labels (Section 3.2), a second reviewer independently annotates a random subset, yielding Cohen’s $\kappa = 0 . 8 2$ , indicating substantial agreement.

L3 annotation statistics. Across all 195 samples, the mean reference minimal set size is 3.74 units (median: 3.0, range: 1–10, σ=2.54). The distribution of minimal set sizes is shown in Table 8.

Table 8: Distribution of human-annotated minimal suficient evidence set sizes.
<table><tr><td>Set Size</td><td>Count</td><td>Proportion</td><td>Cumulative</td></tr><tr><td>1</td><td>32</td><td>16.4%</td><td>16.4%</td></tr><tr><td>2</td><td>47</td><td>24.1%</td><td>40.5%</td></tr><tr><td>3</td><td>31</td><td>15.9%</td><td>56.4%</td></tr><tr><td>4</td><td>31</td><td>15.9%</td><td>72.3%</td></tr><tr><td>5</td><td>18</td><td>9.2%</td><td>81.5%</td></tr><tr><td>6</td><td>9</td><td>4.6%</td><td>86.2%</td></tr><tr><td>7</td><td>6</td><td>3.1%</td><td>89.2%</td></tr><tr><td>8</td><td>4</td><td>2.1%</td><td>91.3%</td></tr><tr><td>9</td><td>2</td><td>1.0%</td><td>92.3%</td></tr><tr><td>10</td><td>15</td><td>7.7%</td><td>100%</td></tr></table>

Over 72% of samples require 4 or fewer evidence units, confirming that minimal suficient evidence sets are typically small relative to the full evidence pool (mean 28.3 units).

Minimal set size by verdict class. Supported claims require the fewest evidence units on average (3.25), followed by refuted (3.69) and insuficient (4.58). The larger sets for insuficient cases reflect the need for annotators to demonstrate that no compact subset sufices, often requiring evidence from multiple modalities to establish that the claim cannot be resolved.

## C Complete Experimental Results

## C.1 Per-Class F1 Scores

Table 9 presents the per-class precision, recall, and F1 for all strategies, complementing the Macro-F1 reported in Table 4.

Key observations. (1) B3 achieves the highest precision on insuficient cases (1.000) because it predicts “insuficient” for only 3 samples—all of which happen to be correct—but its recall of 0.051 renders this precision meaningless in practice. Meanwhile, B3 achieves the highest supported recall (0.866) by aggressively committing to verdicts. (2) B5 has the second-highest insuficient-class F1 (0.383, after B2 Top-3) while staying within 0.05 of the best supported F1 and within 0.07 of the best refuted F1. The methods with higher supported and refuted F1 (B1 and B3) are much weaker on the insuficient class (0.337 and 0.097). (3) The refuted class is the most challenging across all methods, with the best F1 being 0.556 (B3), reflecting the dificulty of locating specific contradictory evidence. Refuted precision is consistently low across methods (0.37–0.55), indicating that when systems predict “refuted,” they are frequently wrong.

Table 9: Per-class precision, recall, and F1 across all strategies (195 samples). Bold marks the best value in each column.
<table><tr><td rowspan="2">Method</td><td colspan="3">Supported</td><td colspan="3">Refuted</td><td colspan="3">Insufficient</td></tr><tr><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td></tr><tr><td>B1 Full</td><td>0.632</td><td>0.742</td><td>0.682</td><td>0.471</td><td>0.615</td><td>0.533</td><td>0.500</td><td>0.254</td><td>0.337</td></tr><tr><td>B2 Top-3</td><td>0.679</td><td>0.546</td><td>0.606</td><td>0.371</td><td>0.333</td><td>0.351</td><td>0.378</td><td>0.525</td><td>0.440</td></tr><tr><td>B2 Top-5</td><td>0.606</td><td>0.649</td><td>0.627</td><td>0.405</td><td>0.385</td><td>0.395</td><td>0.370</td><td>0.339</td><td>0.354</td></tr><tr><td>B2 Top-7</td><td>0.615</td><td>0.691</td><td>0.650</td><td>0.413</td><td>0.487</td><td>0.447</td><td>0.350</td><td>0.237</td><td>0.283</td></tr><tr><td>B3 CoT</td><td>0.596</td><td>0.866</td><td>0.706</td><td>0.490</td><td>0.641</td><td>0.556</td><td>1.000</td><td>0.051</td><td>0.097</td></tr><tr><td>B4 Greedy</td><td>0.605</td><td>0.711</td><td>0.654</td><td>0.467</td><td>0.538</td><td>0.500</td><td>0.333</td><td>0.203</td><td>0.253</td></tr><tr><td>B5 Abstention</td><td>0.641</td><td>0.680</td><td>0.660</td><td>0.548</td><td>0.436</td><td>0.486</td><td>0.377</td><td>0.390</td><td>0.383</td></tr></table>

## C.2 Cross-Model Generalization

To verify that our findings are not idiosyncratic to a single backbone, we re-ran the entire verification pipeline (relevance scoring → suficiency search → two-layer verdict) with two additional verifier models on the identical, frozen evidence pool: the same 5,510 cached units (Claude-written V descriptions, Whisper ASR, retrieved E snippets), with the same unit IDs and contents. Only the verifier model changes; the inputs do not. We test GPT-5.5 (a diferent provider/architecture, closed-weight) and Qwen2.5-72B-Instruct (open-weight). Because the descriptions remain Claudegenerated, the verifier ̸= the description model, which also tests whether the results depend on the same model both describing and verifying. Table 10 reports the full B1–B5 results for all three backbones.

Key observations. (1) Full-evidence performance is consistent across backbones (B1 Macro-F1 0.52–0.59); the two alternate verifiers are if anything slightly stronger than Claude, confirming that the absolute level tracks verifier capability rather than a quirk of one model. (2) The minimal-suficient-subset efect replicates on GPT-5.5 and only partially on Qwen: B5 recovers 95.3% (GPT-5.5) and 84.3% (Qwen) of full-evidence Macro-F1 while using only 9–18% of the evidence units, mirroring Claude’s 98.5%. The gap to B1 is not significant on GPT-5.5 but is significant on Qwen (Section C.3). (3) The abstention gain scales with verifier capability. On the two stronger verifiers (Claude, GPT-5.5), B5 clearly improves the insuficient class over B4 (Ins-F1 +0.131 and +0.100; GPT-5.5 abstention recall 0.373→0.559). The open-weight Qwen is a coarser suficiency judge—it ends with a single evidence unit on 60% of samples (vs. 28% for GPT-5.5)—so its selected sets are smaller but its abstention margin is correspondingly thinner. The selection framework thus transfers across providers, while its benefit depends on the quality of the verifier’s suficiency judgments.

## C.3 Statistical Significance and Confidence-Threshold Baselines

All comparisons in this section use a paired bootstrap over videos (2,000 resamples, seed 42) of the diference B5 minus the compared method, for Macro-F1 and for insuficient-class F1; positive values favor B5. In addition to the strategies of Section 4.3, we evaluate confidence-threshold abstention $( ^ { 6 6 } + \tau ^ { 5 7 } )$ : any committed verdict (supported or refuted) whose self-reported confidence is below τ is converted to insuficient. The main variant uses τ = 0.8, the same threshold B4 and B5 use for the suficiency check, and involves no tuning. As a tuned reference, we also select τ from {0, 0.05, . . . , 0.95} by 5-fold cross-validation (maximizing Macro-F1 on the training folds and applying it to the held-out fold), repeated over 20 random splits. Table 11 reports the results for all three backbones.

Table 10: Cross-model generalization: full results for three verifier backbones on the identical frozen evidence pool. Claude Sonnet 4 (n=195) and GPT-5.5 (n=195) cover all samples; for Qwen2.5-72B, a few samples whose claims concern disaster and violent-incident reporting were refused by the DashScope provider’s content-safety filter and excluded (n=193 for B1/B2; n=192 for B4/B5). These are provider-side content refusals unrelated to the selection method. “Units” = average selected evidence units; “Ins-F1” = insuficient-class F1. Bold marks the best value per column within each backbone. The eficiency efect (B5 recovers most of B1’s Macro-F1 with a fraction of the evidence) holds on GPT-5.5 and only partially on Qwen2.5-72B; the abstention gain (B5 > B4 on Ins-F1) is clear on Claude Sonnet 4 and GPT-5.5 but small on Qwen2.5-72B.
<table><tr><td>Backbone</td><td>Method</td><td>Macro-F1</td><td>Accuracy</td><td>Units</td><td>Ins-F1</td></tr><tr><td rowspan="4">Claude Sonnet 4</td><td>B1 Full</td><td>0.518</td><td>0.569</td><td>27.7</td><td>0.337</td></tr><tr><td>B2 Top-5</td><td>0.459</td><td>0.503</td><td>5.0</td><td>0.354</td></tr><tr><td>B4 Greedy</td><td>0.469</td><td>0.523</td><td>4.1</td><td>0.253</td></tr><tr><td>B5 Abstention</td><td>0.510</td><td>0.544</td><td>4.5</td><td>0.383</td></tr><tr><td rowspan="4">GPT-5.5 (closed)</td><td>B1 Full</td><td>0.586</td><td>0.651</td><td>27.7</td><td>0.464</td></tr><tr><td>B2 Top-5</td><td>0.551</td><td>0.605</td><td>5.0</td><td>0.404</td></tr><tr><td>B4 Greedy</td><td>0.547</td><td>0.615</td><td>4.7</td><td>0.407</td></tr><tr><td>B5 Abstention</td><td>0.559</td><td>0.626</td><td>4.9</td><td>0.508</td></tr><tr><td rowspan="4">Qwen2.5-72B (open)</td><td>B1 Full</td><td>0.580</td><td>0.658</td><td>27.9</td><td>0.386</td></tr><tr><td>B2 Top-5</td><td>0.577</td><td>0.622</td><td>5.0</td><td>0.459</td></tr><tr><td>B4 Greedy</td><td>0.526</td><td>0.573</td><td>2.3</td><td>0.413</td></tr><tr><td>B5 Abstention</td><td>0.489</td><td>0.573</td><td>2.4</td><td>0.423</td></tr></table>

Table 11: Paired-bootstrap comparisons of B5 against every other method, on all three verifier backbones. $\Delta = \mathrm { B 5 }$ minus method (positive favors B5), with 95% confidence intervals; <sup>∗</sup> marks $p < 0 . 0 5 . \ ^ { \ast } + \tau ^ { \ast }$ rows apply confidence-threshold abstention with $\tau = 0 . 8 ;$ the last column gives the Macro-F1 of the same base method with a cross-validated τ (mean ± s.d. over 20 random 5-fold splits). For Qwen2.5-72B, the comparison uses the 192 samples on which all four strategies returned a verdict, so B1 and Top-5 difer slightly from Table 10 (n=193).
<table><tr><td>Method</td><td>Macro-F1</td><td>Units</td><td></td><td>∆Macro-F1 [95% CI]</td><td></td><td>∆Ins-F1 [95% CI]</td><td></td><td>CV-τ Macro-F1</td></tr><tr><td colspan="7">Claude Sonnet 4 (n=195)</td><td></td><td></td></tr><tr><td>B1 Full</td><td>0.518</td><td>27.7</td><td></td><td>-0.008 [−0.08, +0.06]</td><td></td><td></td><td>+0.046 [−0.08, +0.18]</td><td></td></tr><tr><td>B2 Top-3</td><td>0.466</td><td>3.0</td><td></td><td>+0.044 [−0.03, +0.12]</td><td></td><td></td><td>−0.056 [−0.18, +0.07]</td><td></td></tr><tr><td>B2 Top-5</td><td>0.459</td><td>5.0</td><td></td><td>+0.051 [−0.02, +0.13]</td><td></td><td></td><td>+0.029 [−0.11, +0.17]</td><td></td></tr><tr><td>B2 Top-7</td><td>0.460</td><td>7.0</td><td></td><td>+0.050 [−0.02, +0.12]</td><td></td><td></td><td>+0.101 [−0.02, +0.22]</td><td></td></tr><tr><td>B3 CoT</td><td>0.453</td><td>9.4</td><td></td><td>+0.057 [−0.02, +0.13]</td><td></td><td></td><td>+0.287 [+0.16, +0.41]*</td><td></td></tr><tr><td>B4 Greedy</td><td>0.469</td><td>4.1</td><td></td><td></td><td>+0.041 [−0.02, +0.10]</td><td></td><td>+0.131 [+0.03, +0.23]*</td><td></td></tr><tr><td>B1 Full + τ</td><td>0.525</td><td>27.7</td><td></td><td>-0.015 [−0.09, +0.05]</td><td></td><td></td><td>+0.028 [−0.10, +0.15]</td><td> $0 . 5 3 4 \pm 0 . 0 1 4$ </td></tr><tr><td> $\mathrm { B 2 \ T o p { - } 5 + \Delta \tau }$ </td><td>0.464</td><td>5.0</td><td></td><td>+0.045 [−0.03, +0.12]</td><td></td><td></td><td>+0.015 [−0.12, +0.15]</td><td> $0 . 4 5 2 \pm 0 . 0 0 9$ </td></tr><tr><td> ${ \mathrm { B 4 ~ G r e e d y } } + \tau$ </td><td>0.476</td><td>4.1</td><td></td><td>+0.034 [−0.03, +0.09]</td><td></td><td></td><td>+0.113 [+0.01, +0.22]*</td><td> $0 . 4 7 1 \pm 0 . 0 1 3$ </td></tr><tr><td>B5 (reference)</td><td>0.510</td><td>4.5</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2">GPT-5.5 (n=195)</td><td colspan="7"></td></tr><tr><td>B1 Full</td><td>0.586</td><td>27.7</td><td></td><td>−0.027 [−0.10, +0.04]</td><td></td><td></td><td>+0.043 [−0.05, +0.14]</td><td></td></tr><tr><td>B2 Top-5</td><td>0.551</td><td>5.0</td><td></td><td>+0.008 [−0.05, +0.06]</td><td></td><td></td><td>+0.104 [+0.02, +0.20]*</td><td></td></tr><tr><td>B4 Greedy</td><td>0.547</td><td>4.7</td><td></td><td>+0.012 [−0.04, +0.06]</td><td></td><td></td><td>+0.100 [+0.01, +0.19]*</td><td></td></tr><tr><td>B1 Full + τ</td><td>0.577</td><td>27.7</td><td>-0.018</td><td>[−0.08, +0.04]</td><td></td><td></td><td>+0.043 [−0.06, +0.13]</td><td> $0 . 5 9 2 \pm 0 . 0 1 0$ </td></tr><tr><td> $\mathrm { B 2 \ T o p { - } 5 + \Delta \tau }$ </td><td>0.554</td><td>5.0</td><td></td><td>+0.005 [−0.05, +0.06]</td><td></td><td></td><td>+0.032 [−0.05, +0.12]</td><td> $0 . 5 2 5 \pm 0 . 0 1 1$ </td></tr><tr><td>B4 Greedy + τ</td><td>0.527</td><td>4.7</td><td></td><td>+0.032 [−0.03, +0.09]</td><td></td><td></td><td>+0.054 [−0.03, +0.13]</td><td></td></tr><tr><td>B5 (reference)</td><td>0.559</td><td>4.9</td><td></td><td></td><td></td><td></td><td></td><td> $0 . 5 6 4 \pm 0 . 0 0 7$ </td></tr><tr><td>Qwen2.5-72B (n=192)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="9"></td></tr><tr><td>B1 Full</td><td>0.575</td><td>27.9</td><td></td><td>-0.086 [−0.17, -0.00]*</td><td></td><td></td><td>+0.052 [−0.09, +0.20]</td><td></td></tr><tr><td>B2 Top-5</td><td>0.574</td><td>5.0</td><td></td><td></td><td>-0.085 [−0.16, -0.01]*</td><td></td><td>-0.026 [−0.14, +0.09]</td><td></td></tr><tr><td>B4 Greedy</td><td>0.526</td><td>2.3</td><td></td><td>−0.037 [−0.09, +0.02]</td><td></td><td></td><td>+0.010 [−0.04, +0.06]</td><td></td></tr><tr><td>B1  $\mathrm { F u l l } + \tau$ </td><td>0.590</td><td>27.9</td><td></td><td></td><td>-0.101 [−0.19, -0.01]*</td><td></td><td>+0.013 [−0.13, +0.17]</td><td> $0 . 5 6 6 \pm 0 . 0 2 0$ </td></tr><tr><td>B2  $\mathrm { T o p } { - } 5 + \tau$ </td><td>0.574</td><td>5.0</td><td></td><td></td><td>-0.085 [−0.16, −0.01]*</td><td></td><td>−0.022 [−0.13, +0.09]</td><td> $0 . 5 6 3 \pm 0 . 0 1 3$ </td></tr><tr><td>B4 Greedy + τ</td><td>0.526</td><td>2.3</td><td></td><td>−0.037 [−0.09, +0.02]</td><td></td><td></td><td>+0.010 [−0.04, +0.06]</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td> $0 . 5 0 9 \pm 0 . 0 0 7$ </td></tr><tr><td>B5 (reference)</td><td>0.489</td><td>2.4</td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Key observations. (1) B5 matches full evidence on the two stronger verifiers. On Claude Sonnet 4 and GPT-5.5, no Macro-F1 diference between B5 and either B1 or thresholded B1 is significant, although B5 uses 16–18% of the evidence. On Qwen2.5-72B, B5 is significantly below B1 and Top-5, consistent with Section C.2: Qwen’s coarse suficiency judgments stop the search too early (2.4 units on average). (2) Margins among eficient methods are not significant at n=195. On Claude, B5’s Macro-F1 exceeds every eficient strategy (Top-K, CoT, B4, and their thresholded variants) by 0.03–0.06, but none of these margins is statistically significant; on GPT-5.5 the eficient methods lie within 0.04 of each other. (3) Abstention gains concentrate in the insuficient class. B5’s significant advantages are in insuficient-class F1: over B3 and B4 on Claude, over Top-5 and B4 on GPT-5.5, and over thresholded B4 on Claude (+0.113). On Claude and GPT-5.5, thresholding lifts B1, Top-5, and B4 toward B5 on this class, which is why we regard explicit abstention, rather than any particular abstention mechanism, as the key ingredient (Section 5.2). (4) Tuning τ does not change the conclusions. Median cross-validated thresholds lie between 0.75 and 0.85 for B1 and B4 on Claude and GPT-5.5, close to the fixed τ = 0.8. On the first cross-validation split, B5 does not difer significantly from tuned B1 (Claude: −0.011, p=0.78; GPT-5.5: −0.040, p=0.25) or tuned B4 (Claude: +0.021, p=0.55; GPT-5.5: −0.007, p=0.79).

## C.4 Confusion Matrices

We present confusion matrices for B5 (proposed) and B1 (full-evidence baseline).

Table 12: Confusion matrix for B5 (Greedy + Abstention).
<table><tr><td>True \ Pred</td><td>Supported</td><td>Refuted</td><td>Insufficient</td></tr><tr><td>Supported (97)</td><td>66</td><td>8</td><td>23</td></tr><tr><td>Refuted (39)</td><td>7</td><td>17</td><td>15</td></tr><tr><td>Insufficient (59)</td><td>30</td><td>6</td><td>23</td></tr></table>

Table 13: Confusion matrix for B1 (Full Evidence).
<table><tr><td>True \ Pred</td><td>Supported</td><td>Refuted</td><td>Insufficient</td></tr><tr><td>Supported (97)</td><td>72</td><td>16</td><td>9</td></tr><tr><td>Refuted (39)</td><td>9</td><td>24</td><td>6</td></tr><tr><td>Insufficient (59)</td><td>33</td><td>11</td><td>15</td></tr></table>

Analysis. B1 correctly identifies more supported cases (72 vs. 66) and more refuted cases (24 vs. 17) due to its access to all evidence, but performs worse on insuficient cases (15 vs. 23). The key diference is in the insuficient row: B1 misclassifies 33 insuficient samples as supported (overconfident), while B5 correctly identifies 23 insuficient cases. B5’s abstention mechanism converts potential overconfident misclassifications into calibrated “insuficient” predictions, at the cost of more supported samples being conservatively labeled as insuficient (23 vs. 9).

## C.5 Consistency Layer Distribution

Table 14 reports the distribution of Layer 1 (consistency) outputs across strategies, showing how each method assesses claim–video alignment.

B3 produces zero “unclear” assessments and the most “inconsistent” assessments (27.7%), reflecting its tendency to overcommit to strong judgments in both layers. B2 Top-5 produces a high “unclear” rate (35.4%) because fixed-budget selection may exclude internal evidence in favor of higher-scoring external units, depriving Layer 1 of the visual and speech signals needed for consistency assessment.

## C.6 Ablation: Per-Class Results

Table 15 expands the ablation study with per-class F1 scores. As in Section 5.3, the abstention gate is ablated by comparing B5 with B4, and all other components are ablated on B4; the reference for those rows is therefore B4.

Table 14: Distribution of Layer 1 (consistency) assessments across strategies.
<table><tr><td>Method</td><td>Consistent</td><td>Inconsistent</td><td>Unclear</td></tr><tr><td>B1 Full</td><td>168 (86.2%)</td><td>18 (9.2%)</td><td>9 (4.6%)</td></tr><tr><td>B2 Top-5</td><td>117 (60.0%)</td><td>9 (4.6%)</td><td>69 (35.4%)</td></tr><tr><td>B3 CoT</td><td>141 (72.3%)</td><td>54 (27.7%)</td><td>0 (0.0%)</td></tr><tr><td>B4 Greedy</td><td>83 (42.6%)</td><td>13 (6.7%)</td><td>99 (50.8%)†</td></tr><tr><td>B5 Abstention</td><td>79 (40.5%)</td><td>14 (7.2%)</td><td>102 (52.3%)†</td></tr></table>

<sup>†</sup>Most “unclear” labels for B4 and B5 arise because the pruned subset contains only external units, leaving Layer 1 no internal evidence to assess (68 of 99 for B4; 67 of 102 for B5). For the 39 samples on which B5 abstains, the label comes from the last suficiency check, because these samples skip the two-layer verification.

Table 15: Per-class F1 for ablation variants. Rows below the rule are ablations of B4 (the selection pipeline without the abstention gate) and should be compared with the B4 row.
<table><tr><td>Setting</td><td>F1-Sup</td><td>F1-Ref</td><td>F1-Ins</td><td>Macro-F1</td></tr><tr><td>B5 (proposed)</td><td>0.660</td><td>0.486</td><td>0.383</td><td>0.510</td></tr><tr><td>B4 (= B5 w/o Abstention)</td><td>0.654</td><td>0.500</td><td>0.253</td><td>0.469</td></tr><tr><td>B4 w/o External</td><td>0.096</td><td>0.270</td><td>0.453</td><td>0.273</td></tr><tr><td>B4 w/o V+A (E-only)</td><td>0.660</td><td>0.453</td><td>0.340</td><td>0.484</td></tr><tr><td>B4 w/o ASR</td><td>0.638</td><td>0.494</td><td>0.263</td><td>0.465</td></tr><tr><td>B4 w/o Pruning</td><td>0.660</td><td>0.506</td><td>0.265</td><td>0.477</td></tr></table>

Key observations. (1) The abstention gate acts almost entirely on the insuficient class: compared with B4, B5 raises insuficient F1 from 0.253 to 0.383 while supported and refuted F1 change by less than 0.015. (2) Removing external evidence causes the most dramatic degradation: relative to B4, supported F1 collapses from 0.654 to 0.096 and refuted F1 from 0.500 to 0.270, while insuficient F1 rises to 0.453 because the verifier returns insuficient for most samples even without an abstention gate—consistent with the finding in Section 5.3 that external evidence is indispensable for factual determination. (3) Without internal evidence (E-only), Macro-F1 stays close to B4 (0.484 vs. 0.469), with per-class shifts in both directions (supported 0.660 vs. 0.654, refuted 0.453 vs. 0.500, insuficient 0.340 vs. 0.253), but 100% of samples receive “unclear” in Layer 1, confirming that V+A are necessary for interpretable consistency assessment. Supported F1 is unchanged even without internal evidence, indicating that external sources carry the dominant signal for verifying supported claims. (4) Removing ASR alone or skipping pruning changes no class by more than 0.02 relative to B4. Pruning therefore makes the evidence set more compact (4.1 vs. 5.4 units on average) without measurable cost to any verdict class.

## D Case Study

We present two cases illustrating B5’s strengths and limitations.

## D.1 Success Case: manual\_0034

• Claim: A Sydney tech entrepreneur’s adopted dog, after failed conventional treatment, had its tumor halved by an anti-cancer drug created using ChatGPT and AlphaFold.

• Ground truth: Supported

• Total evidence units: 74 (metadata excluded)

B5 selects 4 units out of 74 (5.4%) and correctly outputs “supported” (conf. 0.85). The selected evidence set:

• A2 (ASR): Garbled transcription mentioning “ChatGPT and AlphaFold made a dog’s anticancer drug”—weak but topically anchoring.

• E1 (External, MIT Technology Review China): Confirms Paul used ChatGPT to brainstorm treatments and designed an mRNA cancer vaccine for his dog.

• E4 (External, yzwb.net): Independent confirmation from Paul Conyngham; tumor shrank significantly.

• E5 (External, Zhihu): Confirms ChatGPT + AlphaFold pipeline; DNA sequencing cost only \$3,000.

Human-annotated minimal set: {A2, E1, E4, E5}—exactly the same 4 units, yielding 100% overlap (Jaccard = 1.0).

Comparison across strategies:

• B1 (Full): correct, used all 74 units (100%)

• B3 (CoT): correct, self-selected 21 units (28.4%)

• B4 (Greedy): correct, used 4 units (5.4%)

• B5 (Abstention): correct, used 4 units (5.4%)

This case illustrates one of B5’s strengths: the selected evidence coincides with the set independently identified by human annotators as minimally suficient, achieving a 94.6% evidence reduction without loss of verdict accuracy. The selected set combines one weak internal signal (garbled ASR) with three independent external confirmations, consistent with the two-layer framework’s design where external evidence drives factual determination.

## D.2 Failure Case: manual\_0059

• Claim: US company “Newkins” launches the world’s first custom baby service—\$43K to screen embryos for IQ, appearance, and lifespan across 900+ traits; tested on 120K embryos; used by Musk and Thiel.

• Ground truth: Refuted

• Total evidence units: 32

B5 selects 10 units (the full candidate pool) but outputs “insuficient” instead of “refuted” (conf. 0.3).

The human-annotated minimal set contains 5 units {V2, V7, A1, A3, E5}. B5’s selected set of 10 units is a superset of the human minimal set (5/5 covered), yet the system fails to synthesize them into a refutation.

B1 (Full Evidence) correctly identifies “refuted” (conf. 0.85) by cross-referencing multiple evidence pieces: (1) the real company is “Nucleus Genomics,” not “Newkins”; (2) it screens only ∼12 diseases, not 900+ traits; (3) actual cost is \$2,500 per embryo, not \$43K total; (4) no evidence that Musk or Thiel used the service; (5) the technology is a commercialization of existing PGT-P, not “the world’s first.”

Comparison across strategies:

• B1 (Full): correct (refuted), used all 32 units

• B3 (CoT): correct (refuted), self-selected evidence

• B4 (Greedy): incorrect (insuficient), used 10 units

• B5 (Abstention): incorrect (insuficient), used 10 units

This case suggests that B5’s failure may stem less from evidence coverage than from the dificulty of synthesizing multiple partially contradictory pieces into a confident refutation. The claim mixes real elements (a real company, real technology) with fabrications and exaggerations, requiring crossreferencing across multiple evidence pieces to identify the pattern of distortion. B5’s incremental greedy process evaluates suficiency after each addition, but refutation of such complex claims may require holistic reasoning over the full evidence constellation—something the full-evidence baseline B1 achieves by processing all evidence jointly.

## E Dataset Statistics

## E.1 Topic Distribution

Table 16 shows the distribution of videos across topic categories, annotated via keyword-based classification.

Table 16: Topic distribution of MiniVer-V samples.
<table><tr><td>Topic</td><td>Count</td><td>Proportion</td></tr><tr><td>International Politics &amp; Military</td><td>45</td><td>23.1%</td></tr><tr><td>Domestic News &amp; Society</td><td>30</td><td>15.4%</td></tr><tr><td>Health &amp; Medicine</td><td>29</td><td>14.9%</td></tr><tr><td>Science &amp; Technology</td><td>24</td><td>12.3%</td></tr><tr><td>Natural Disaster &amp; Accident</td><td>22</td><td>11.3%</td></tr><tr><td>Food Safety &amp; Consumer</td><td>14</td><td>7.2%</td></tr><tr><td>Sports &amp; Entertainment</td><td>13</td><td>6.7%</td></tr><tr><td>Daily Life &amp; Myth-busting</td><td>12</td><td>6.2%</td></tr><tr><td>Other</td><td>6</td><td>3.1%</td></tr><tr><td>Total</td><td>195</td><td>100%</td></tr></table>

The dataset spans a broad range of topics, with international politics being the most represented (23.1%), reflecting the prevalence of geopolitical misinformation on Chinese social media platforms during the data collection period. Health & Medicine (14.9%) and Science & Technology (12.3%) are also well-represented, consistent with prior findings that these domains are frequent targets of online misinformation.

## E.2 Platform Distribution

Table 17 shows the source platform of each video. Most videos come from the two largest Chinese short-video platforms, Douyin and Xiaohongshu; the remainder come from X, Bilibili, Weibo, and the video channels of news outlets. All claims are written in Chinese; 40 videos (20.5%) contain mainly English speech, defined as CJK characters making up at most 10% of the ASR transcript.

Table 17: Source platform distribution of MiniVer-V samples.
<table><tr><td>Platform</td><td>Count</td><td>Proportion</td></tr><tr><td>Douyin</td><td>112</td><td>57.4%</td></tr><tr><td>Xiaohongshu</td><td>64</td><td>32.8%</td></tr><tr><td>X (Twitter)</td><td>8</td><td>4.1%</td></tr><tr><td>Bilibili</td><td>3</td><td>1.5%</td></tr><tr><td>Weibo</td><td>3</td><td>1.5%</td></tr><tr><td>News outlets (CNN, BBC, Xinhua, Southern Daily)</td><td>5</td><td>2.6%</td></tr><tr><td>Total</td><td>195</td><td>100%</td></tr></table>

## E.3 Evidence Unit Distribution

Table 18 presents detailed per-modality statistics after metadata exclusion.

Table 18: Evidence unit statistics by modality (metadata excluded).
<table><tr><td>Modality</td><td>Total</td><td>Mean/Sample</td><td>Proportion</td><td>Source</td></tr><tr><td>Visual (V)</td><td>1,207</td><td>6.2</td><td>21.9%</td><td>Claude Sonnet 4</td></tr><tr><td>ASR (A)</td><td>2,847</td><td>14.6</td><td>51.7%</td><td>Whisper (base)</td></tr><tr><td>External (E)</td><td>1,456</td><td>7.5</td><td>26.4%</td><td>Web Search</td></tr><tr><td>Total</td><td>5,510</td><td>28.3</td><td>100%</td><td></td></tr></table>

## E.4 Modality Composition in Human-Annotated Minimal Sets

Table 19 shows the modality composition of the human-annotated minimal suficient evidence sets (L3 annotation), providing insight into which evidence types humans consider most critical for verification.

Table 19: Modality composition in human-annotated minimal suficient sets.
<table><tr><td>Modality</td><td>Units Selected</td><td>Proportion</td><td>Contrast with Pool</td></tr><tr><td>Visual (V)</td><td>116</td><td>15.9%</td><td>21.9% in pool</td></tr><tr><td>ASR (A)</td><td>237</td><td>32.5%</td><td>51.7% in pool</td></tr><tr><td>External (E)</td><td>376</td><td>51.6%</td><td>26.4% in pool</td></tr><tr><td>Total</td><td>729</td><td>100%</td><td></td></tr></table>

External evidence is overrepresented in minimal sets (51.6% selected vs. 26.4% in pool), consistent with its critical role in factual determination as shown by the ablation results in Section 5.3.

Conversely, ASR is underrepresented (32.5% vs. 51.7%), suggesting that while speech content is voluminous, much of it is redundant for verification purposes.

## E.5 Evidence Eficiency by Strategy

Table 20 provides a detailed breakdown of evidence usage across correctly and incorrectly classified samples.

Table 20: Evidence eficiency: average units used for correct vs. incorrect predictions.
<table><tr><td>Method</td><td>Avg (All)</td><td> $\operatorname { A v g }$  (Correct)</td><td>Avg (Incorrect)</td></tr><tr><td>B1 Full</td><td>27.7</td><td>27.7</td><td>27.7</td></tr><tr><td>B2 Top-3</td><td>3.0</td><td>3.0</td><td>3.0</td></tr><tr><td>B2 Top-5</td><td>5.0</td><td>5.0</td><td>5.0</td></tr><tr><td>B2 Top-7</td><td>7.0</td><td>7.0</td><td>7.0</td></tr><tr><td>B3 CoT</td><td>9.4</td><td>9.2</td><td>9.6</td></tr><tr><td>B4 Greedy</td><td>4.1</td><td>3.5</td><td>4.8</td></tr><tr><td>B5 Abstention</td><td>4.5</td><td>3.8</td><td>5.2</td></tr></table>

For B4 and B5, correctly classified samples use substantially fewer evidence units on average (3.5/3.8) than incorrectly classified ones (4.8/5.2). This gap suggests that “easy” samples reach suficiency quickly with a small evidence set, while “hard” samples accumulate more evidence without reaching the suficiency threshold—the model exhausts a larger portion of the candidate pool on samples it ultimately gets wrong. For B1 and B2, evidence counts are fixed by design and thus identical across correct and incorrect predictions.

## F Additional Metric Definitions

For completeness, we provide formal definitions for the metrics used or discussed in our analysis, including several evidence-selection diagnostics not reported in the main text.

## F.1 Verdict Macro-F1

The primary evaluation metric. Let C = {supported, refuted, insuficient} be the set of verdict classes. For each class $c \in { \mathcal { C } }$ , we compute precision $P _ { c } ,$ recall $R _ { c } .$ , and F1 score $F _ { 1 , c } \colon$

$$
P _ { c } = { \frac { T P _ { c } } { T P _ { c } + F P _ { c } } }\tag{1}
$$

$$
R _ { c } = { \frac { T P _ { c } } { T P _ { c } + F N _ { c } } }\tag{2}
$$

$$
F _ { 1 , c } = { \frac { 2 \cdot P _ { c } \cdot R _ { c } } { P _ { c } + R _ { c } } }\tag{3}
$$

where $T P _ { c } , F P _ { c } ,$ and $F N _ { c }$ are the true positives, false positives, and false negatives for class $^ { c , }$ respectively. The Macro-F1 is the unweighted average:

$$
\mathrm { M a c r o - F 1 } = { \frac { 1 } { | { \mathcal { C } } | } } \sum _ { c \in { \mathcal { C } } } F _ { 1 , c }\tag{4}
$$

Macro-F1 treats all classes equally regardless of sample count, penalizing methods that sacrifice minority-class performance $( \mathrm { e . g . }$ , refuted or insuficient) for majority-class gains.

## F.2 Consistency Accuracy

Measures the accuracy of the Layer 1 consistency assessment against human annotations:

$$
\mathrm { C o n s A c c } = \frac { | \{ i : \hat { y } _ { i } ^ { \mathrm { c o n s } } = y _ { i } ^ { \mathrm { c o n s } } \} | } { N }\tag{5}
$$

where $\hat { y } _ { i } ^ { \mathrm { c o n s } }$ and $y _ { i } ^ { \mathrm { { c o n s } } }$ are the predicted and gold consistency labels for sample i, and N is the number of samples with gold consistency annotations. This metric would require gold consistency annotations and is included here for completeness; it is not reported in the current study.

## F.3 Suficiency Accuracy

Measures whether the model’s selected evidence covers the human-annotated minimal suficient set. For samples with gold verdict ̸= insuficient:

$$
\mathrm { S u f A c c } = \frac { | \{ i : S _ { i } ^ { * } \subseteq \hat { S } _ { i } \} | } { N _ { \mathrm { n o n - i n s u f } } }\tag{6}
$$

where $\boldsymbol { S } _ { i } ^ { * }$ is the human-annotated minimal set and $\hat { S } _ { i }$ is the model’s selected evidence. Suficiency accuracy checks that the model’s selection is a superset of the human reference.

## F.4 Minimality Rate

Evaluates how compact the model’s selections are relative to the human reference:

$$
\mathrm { M i n P r e c } _ { i } = \frac { | \hat { S _ { i } } \cap S _ { i } ^ { * } | } { | \hat { S _ { i } } | } , \quad \mathrm { S i z e R a t i o } _ { i } = \frac { | \hat { S _ { i } } | } { | S _ { i } ^ { * } | }\tag{7}
$$

Minimality Precision measures the proportion of model-selected units that appear in the human reference (higher is better). Size Ratio indicates how many times larger the model’s selection is compared to the human reference (closer to 1.0 is bet $\mathrm { t e r } ; > 1$ indicates redundancy).

## F.5 Evidence Eficiency

The average number of evidence units used per sample:

$$
\mathrm { A v g U n i t s } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } | \hat { \mathcal { S } } _ { i } |\tag{8}
$$

We also report this metric separately for correctly and incorrectly classified samples to analyze whether evidence quantity correlates with prediction quality.

## F.6 Abstention Quality

Evaluated through three sub-metrics focused on the “insuficient” prediction:

$$
{ \mathrm { A b s t . ~ P r e c i s i o n } } = { \frac { T P _ { \mathrm { i n s } } } { T P _ { \mathrm { i n s } } + F P _ { \mathrm { i n s } } } }\tag{9}
$$

$$
{ \mathrm { A b s t . ~ R e c a l l } } = { \frac { T P _ { \mathrm { i n s } } } { T P _ { \mathrm { i n s } } + F N _ { \mathrm { i n s } } } }\tag{10}
$$

$$
\mathrm { O v e r C o n f } = 1 - \mathrm { A b s t . } \ \mathrm { R e c a l l } = { \frac { F N _ { \mathrm { i n s } } } { T P _ { \mathrm { i n s } } + F N _ { \mathrm { i n s } } } }\tag{11}
$$

where $T P _ { \mathrm { i n s } }$ counts samples correctly predicted as insuficient, $F P _ { \mathrm { i n s } }$ counts non-insuficient samples incorrectly predicted as insuficient, and $F N _ { \mathrm { i n s } }$ counts truly insuficient samples predicted as noninsuficient.

The Overconfidence Rate specifically measures the system’s tendency to fail to abstain when evidence is genuinely inadequate—a particularly dangerous failure mode in high-stakes fact-checking applications.

## F.7 Evidence Faithfulness

Measures whether the model’s verdict is grounded in evidence that overlaps with the humanannotated reference:

$$
{ \mathrm { F a i t h f u l n e s s } } = { \frac { | \{ i : { \hat { S _ { i } } } \cap S _ { i } ^ { * } \neq \emptyset , \ { \hat { y _ { i } } } \neq { \mathrm { i n s u f . } } \} | } { | \{ i : { \hat { y _ { i } } } \neq { \mathrm { i n s u f . } } \} | } }\tag{12}
$$

A verdict is considered “faithful” if the model’s selected evidence set contains at least one unit from the human-annotated minimal set. Low faithfulness indicates that the model reaches correct verdicts through evidence not considered critical by human annotators—a potential interpretability concern.

## G Model Configuration

For completeness, we list the full model configuration used across all experiments.

<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Main experiments</td><td></td></tr><tr><td>Backbone model</td><td>Claude Sonnet 4</td></tr><tr><td>Model ID</td><td>claude-sonnet-4-20250514</td></tr><tr><td>Provider</td><td>Anthropic</td></tr><tr><td>Temperature</td><td>0.1</td></tr><tr><td>Max generation tokens</td><td>4,096</td></tr><tr><td>ASR model</td><td>Whisper (base)</td></tr><tr><td>Visual description model</td><td>Claude Sonnet 4</td></tr><tr><td>Search API</td><td>SerpAPI (Google Search)</td></tr><tr><td>Candidate pool size (B4/B5)</td><td>10</td></tr><tr><td>Sufficiency threshold τ</td><td>0.8</td></tr><tr><td>Min. internal units in pool</td><td>2</td></tr><tr><td>Min. external units in pool</td><td>3</td></tr><tr><td>Max. external units (B1)</td><td>8</td></tr><tr><td>Unit truncation in prompts</td><td>150 characters per unit (100 for relevance scoring)</td></tr><tr><td>Cross-model verifiers (Section 5.6)</td><td></td></tr><tr><td>GPT-5.5</td><td>OpenAI API, model gpt-5.5; default temperature (reasoning model); max. 4,096 completion tokens</td></tr><tr><td>Qwen2.5-72B-Instruct</td><td>DashScope API (OpenAI-compatible), model qwen2.5-72b-instruct; temperature 0.1; max.</td></tr><tr><td>Scoring and sufficiency checks Same model as the verifier Evidence pool</td><td>4,096 tokens Identical frozen pool (Claude-generated V descrip-</td></tr><tr><td>Raw-keyframe control (Section 5.7)</td><td>tions)</td></tr><tr><td>Verifier</td><td>Claude Sonnet 4 (B1)</td></tr><tr><td>Visual input</td><td>Keyframe images (JPEG) in place of V descrip-</td></tr><tr><td></td><td>tions</td></tr><tr><td>Subset</td><td>60 videos, 20 per verdict class</td></tr></table>