# ReviewMap: Mapping Evaluative Acts and Automation Boundaries in Peer-Review Comments

Kihyun Na<sup>1</sup>\* and Injung Kim<sup>2†</sup> <sup>1</sup>Department of Neurosurgery, University of Alabama at Birmingham Birmingham, AL 35294, USA <sup>2</sup>Department of Artificial Intelligence, Computer and Electrical Engineering Handong Global University, Pohang 37554, Republic of Korea khna@uab.edu, ijkim@handong.edu

## Abstract

Peer-review comments check what a paper does and judge why it matters, but no existing scheme distinguishes these two evaluative acts. We argue that making these acts explicit and measuring how reliably LLMs identify them is a prerequisite for responsible LLM-assisted review analysis: humans need to inspect review judgments and their grounds. This defines a more honest mission for NLP in peer review than full automation. We present ReviewMap, a framework that classifies each comment as assessment of claims and scholarly standards or calibration of contribution value and venue fit, with finer-grained attributes applied condition ally by act type. Applied to 500 ICLR 2026 papers through multi-model LLM consensus, the act layer proves reliably automatable under the operational guideline (inter-model κ ≥ 0.75), supported by five-human validation on 470 development comments, while the attribute layer requires human oversight, defining a measured automation boundary. At the corpus level, 17% of reviewer–paper instances contain no panelcoded calibration, and the panel-coded calibration share is nearly 3× as high in Strengths as in Weaknesses. This boundary informs three affordances: a review guideline that operationalizes the assessment-then-calibration workflow for reviewers, a per-comment rebuttal strategy scaffold for authors, and a discussion-phase dashboard for area chairs.

## 1 Introduction

Peer review requires evaluating a submission from multiple angles: whether claims are supported by adequate evidence, whether the methodology meets the standards of the field, and whether the work constitutes a contribution significant enough for the venue. These dimensions are often recorded through separate criterion scores but ultimately inform an overall recommendation. Yet they differ in kind. Checking whether an experiment lacks a proper baseline is a different evaluative act from declaring that the contribution is too incremental. The first assesses the paper against its claims and scholarly standards; the second judges the contribution’s value and venue fit. Both involve judgment; their evaluative targets and criteria differ. We call these two acts assessment and calibration. Venue reviewer guidelines already encode a related separation at the scoring-criterion level: ACL Rolling Review separates soundness, excitement, and overall recommendation; ICML separates soundness from significance; and NeurIPS distinguishes quality from significance. ICLR, the venue our corpus draws from, instead uses a contribution criterion that can combine novelty, value, and venue-level significance. Across venues, however, the free-text comments interleave these evaluative acts without marking them as structurally distinct.

![](images/9056ea20cc2f5d35622d62caf3cc111ee9829f5f1b37d608aa1b540bea72d852.jpg)  
Figure 1: The ReviewMap ecosystem: reviewers check assessments and ground value judgments; authors choose responses; area chairs inspect panel judgments and disagreements. These proposed affordances require users to inspect source comments and confirm attributebased suggestions (§6).

The growing volume of submissions has motivated NLP-for-peer-review research (Kuznetsov et al., 2024) and raised concerns about the sustainability of the AI-venue peer-review process at scale (Kim et al., 2025). Recent work has begun deploying LLMs in this process, from detecting LLMmodified reviews (Liang et al., 2024) to providing automated feedback to reviewers at scale (Thakkar et al., 2026) and simulating peer review dynamics with LLM agents (Jin et al., 2024). However, these efforts operate without a clear framework for determining which parts of review analysis LLMs can reliably perform and which require human judgment. Baumann et al. (2026) argue that deploying LLMs in peer review without rigorous evaluation of their reliability risks undermining the process. Yet no existing scheme provides the measurement instrument such evaluation would require.

Prior schemes classify comments by topic, pragmatic action, or quality attributes; none distinguishes the evaluative act itself. A comment about novelty can assess a priority claim—“The approach was previously proposed by X et al.”—or calibrate contribution value—“The novelty is insufficient for this venue.” Topic and evaluative act are orthogonal: the shared topic does not determine whether a response must establish what is new or justify why it matters.

This decomposition matters because the two acts call for different justifications. Assessments ask whether claims or standards are satisfied; calibrations ask why the contribution matters. A response to either act can use evidence and argument, but an additional ablation answers a significance judgment only when its result bears on the contribution’s value. Making the act explicit helps authors address the relevant criterion and reviewers explain how assessments support an overall recommendation. Without explicit calibration, that link remains implicit; without explicit grounding, its basis is harder to engage with.

Our position. Operationalizing this distinction and measuring how reliably LLMs classify it is a prerequisite for responsible LLM-assisted analysis of review judgments. The aim is to make review judgments and uncertainty in their coding inspectable by humans. Three contributions support this position: (1) An act-based framework. ReviewMap operationalizes assessment and calibration at the comment level, with act-conditional attributes. (2) A measured automation boundary.

Three-model consensus over 500 ICLR 2026 papers (17,654 comments), supported by five-human development-set validation, establishes reliable act classification under the protocol and exposes lower attribute agreement. (3) Design implications across roles. We articulate design implications for a reviewer guideline, an author scaffold, and an AC dashboard using this distinction and measured boundary (Figure 1).

## 2 Related Work

Table 1 situates ReviewMap against prior peerreview comment analysis schemes, grouped by what they classify; we walk through each tradition below.

Aspect classification. PeerRead (Kang et al., 2018) introduced review-level aspect scores; ASAP-Review (Yuan et al., 2022) extended this to span-level aspect and polarity annotation; Lu et al. (2025) proposed a bottom-up, LLM-driven taxonomy with 16 coarse and 39 fine-grained aspect categories. These schemes classify what a comment is about. ReviewMap classifies what a comment does.

Pragmatic action. AMPERE (Hua et al., 2019) introduced proposition-level labels for review argumentation. DISAPERE (Kennard et al., 2022) synthesized prior schemes into sentence-level properties including Review-Action, where “Evaluative” is a single category. NLPeer (Dycke et al., 2023) contributed a multi-domain ethically licensed corpus with pragmatic tags. These schemes classify which discourse role a comment plays (evaluate, request, recap, . . . ); DISAPERE is our closest structural precedent. ReviewMap operates one layer deeper: we decompose DISAPERE’s single “Evaluative” role into assessment and calibration, and replace human annotation with multi-model LLM consensus for scalability.

Review quality attributes. SubstanReview (Guo et al., 2023) measures claim substantiation in 550 reviews, reporting a declining substantiation trend across NLP conferences over time. PolitePEER (Bharti et al., 2024) annotates peer-review comments by politeness intensity across five levels. LazyReview (Purkayastha et al., 2025) derives an 18-category annotation scheme from ARR guidelines and validates it through a controlled experiment. These schemes assess review quality without distinguishing assessment from calibration.

<table><tr><td>Paper</td><td>Venue</td><td>Unit</td><td>Classifies</td><td>LLM annot.</td><td>Audience</td></tr><tr><td>PeerRead (2018)</td><td>NAACL &#x27;18</td><td>Review</td><td>Aspect scores</td><td>No</td><td></td></tr><tr><td>ASAP-Review (2022)</td><td>JAIR &#x27;22</td><td>Text segment</td><td>Aspect labels</td><td>No</td><td>R</td></tr><tr><td>Lu et al. (2025)</td><td>EMNLP F’25</td><td>Sentence</td><td>Aspect (coarse/fine)</td><td>Single</td><td>R</td></tr><tr><td>AMPERE (2019)</td><td>NAACL &#x27;19</td><td>Text segment</td><td>Proposition type</td><td>No</td><td>一</td></tr><tr><td>DISAPERE (2022)</td><td>NAACL &#x27;22</td><td>Sentence</td><td>Review-Action (+attributes)</td><td>No</td><td>C</td></tr><tr><td>NLPeer (2023)</td><td>ACL&#x27;23</td><td>Sentence</td><td>Pragmatic tag</td><td>No</td><td>R</td></tr><tr><td>SubstanReview (2023)</td><td>EMNLP F’23</td><td>Text segment</td><td>Substantiation</td><td>No</td><td>R/C</td></tr><tr><td>PolitePEER (2024)</td><td>LRE&#x27;24</td><td>Sentence</td><td>Politeness (5 levels)</td><td>No</td><td>R/C</td></tr><tr><td>LazyReview (2025)</td><td>ACL ’25</td><td>Text segment</td><td>Lazy thinking</td><td>Single</td><td>R</td></tr><tr><td>Kuznetsov et al. (2026)</td><td>EMNLP’26</td><td>Text segment / review</td><td>Quality metrics</td><td>Single</td><td>A/C</td></tr><tr><td>ReviewMap (ours)</td><td></td><td>Text segment</td><td>Evaluative act (ASM/CAL)</td><td>Multi</td><td>R/A/C</td></tr></table>

Table 1: Selected review-text tasks; prior schemes do not distinguish assessment from calibration. Text segments cover study-specific spans, propositions, items, and comments, with differing boundaries (LazyReview: 1–5 sentences). LLM annotation counts generative models assigning the listed labels/scores, excluding models used only for preprocessing or performance comparison; Multi denotes consensus. Audience identifies roles explicitly addressed in task or use-case design: R = reviewer, A = author, C = chairs/editors; – = unspecified.

Kuznetsov et al. (2026) score weaknesses and related review items for actionability and grounding alongside review-level quality. Amirshahi et al. (2026) find LLM-based review metrics sensitive to meaning-preserving rewrites. ReviewMap first separates evaluative acts, then applies act-specific attributes.

Multi-model consensus. Single-model LLM annotation is effective but contested in reliability (Gilardi et al., 2023; Bavaresco et al., 2025). Verga et al. (2024) show that a panel of diverse LLMs outperforms any single judge for generation evaluation. Courvoisier et al. (2026) apply multi-model agreement to classification and extraction tasks, introducing a withheld-judgment protocol for cases without consensus. Wang et al. (2023) provide the theoretical basis for self-consistency sampling. We extend panel consensus to multi-layer subjective annotation; reliability stratifies by layer.

## 3 The ReviewMap Framework

ReviewMap separates two evaluative acts: assessment of claims and scholarly standards, and calibration of contribution value and venue fit. We choose this distinction because whether a claim is supported or a standard met and whether a contribution is valuable ask different evaluative questions. Level 1 classifies the act (§3.1); conditional Level 2 attributes expose the assessment’s criteria (P1/P2) or the calibration’s stated grounds (§3.2), so readers can inspect the basis of each judgment.

## 3.1 Level 1: Evaluative Act (ASM / CAL)

Each atomic comment in our corpus is assigned exactly one of three act codes (Table 2, top). The boundary between ASM and CAL is the locus of value commitment: ASM evaluates support for claims and conformity with scholarly standards, while CAL judges the contribution's novelty, significance, or venue fit. Both can be positive or negative and involve interpretation. Questions are classified by their evaluative function; mixed comments receive their dominant act. FLAG is the fallback when neither act can be assigned under the guideline (Appendix A).

## 3.2 Level 2: Act-conditional attributes

The second layer attaches binary attributes conditioned on the Level 1 act (Table 2, bottom). For ASM, two attributes record what the assessment evaluates against: a specific claim made by the paper (claim\_relevant, P1) and/or a field-wide standard or convention (standard\_relevant, P2). For CAL, a single attribute records whether the reviewer’s value judgment is anchored in explicit evidence (grounded), following the substantiation literature (Guo et al., 2023). FLAG comments carry no Level 2 attributes.

P1 and P2 are orthogonal dimensions: a comment can score yes on both when a specific claim also conflicts with a field standard. P2 and grounded target adjacent but orthogonal axes: P2 is a targeting property (does the comment invoke a shared community standard?), while grounded is a justification property (does the comment supply explicit evidence for a personal judgment?).

<table><tr><td>Code At- tribute</td><td>Definition</td><td>Example</td></tr><tr><td colspan="2">LEVEL 1 — EVALUATIVE ACT Evaluates claims,</td><td>Ablations don&#x27;t re-</td></tr><tr><td>ASM (assessment) CAL</td><td>methods, or schol- arly standards. Commits to a judg-</td><td>port variance The empirical gains are marginal</td></tr><tr><td>(calibration) FLAG</td><td>ment about the pa- per&#x27;s significance. ASM/CAL cannot be assigned under the guideline.</td><td>See major weak- nesses above</td></tr><tr><td colspan="3">LEVEL 2 — ACT-CONDITIONAL ATTRIBUTES</td></tr><tr><td>claim_relevant (P1, for ASM) standard_ relevant</td><td>Evaluates against a paper-specific claim.</td><td>Paper claims A, but Table shows B.</td></tr><tr><td rowspan="2">(P2, for ASM) grounded (for CAL)</td><td>Evaluates against a field-wide stan- dard.</td><td>Ablations should report variance.</td></tr><tr><td>Value judgment an- chored in explicit evidence.</td><td>Novelty is limited; the derivation resembles prior work.</td></tr></table>

Table 2: The ReviewMap framework. Level 1 classifies each comment by its evaluative act; Level 2 attaches binary attributes conditioned on the Level 1 act. Full operational definitions in Appendix A; corpus distribution and reliability in §5.

From a comment to a coding. Two positive corpus comments illustrate the distinction. “The claims are supported with theoretical justification” assesses claims: all three model labels are ASM, P1=yes, P2=no. “The idea is very simple but impactful” judges contribution value: all three are CAL, grounded=no, with no explicit support for “impactful.” Thus act differs from sentiment. Labels are aggregated first across runs, then across models (§4.3); Appendix B links source identifiers, inputs, and output format.

## 4 Methodology

## 4.1 Data collection

We sample 500 ICLR 2026 submissions from Open-Review (Soergel et al., 2013), stratified by 21 primary areas crossed with accept/reject decision (seed 42; 38.8% accepted; at least three accepted and three rejected papers per area). Sampling across areas examines the framework’s coverage beyond a single subfield.

Each review is segmented into comments by a rule-based parser that uses list markers and paragraph boundaries within the venue’s Strengths /

Weaknesses / Questions sections. The artifact preserves the batch-specific preprocessing profiles. The 500 papers yield 17,654 comments across 1,955 reviewer–paper instances. This is the unit of annotation; everything downstream (§5, §6) operates over these comments rather than over full reviews.

Separately, before the main run, we compiled an additional development set (12 papers, 470 comments). The dev set supports the open-weight, frontier-LLM, and human comparisons in §4.4.

## 4.2 Annotation design

We developed a single annotation guideline applied uniformly across all models (see Appendix A for the full text and worked examples; §3 summarizes the operational definitions). Each prompt joins a single paper’s segmented comments with the guideline; the LLM returns one structured row per comment with the Level-1 code and the applicable Level-2 attributes.

Annotation inputs. Each per-paper prompt contains the title, abstract, keywords, reviewer identifiers, scores and confidence, section headers, comment identifiers and text, together with the shared guideline. The wrapper extracts claims from the supplied abstract; the full paper is not supplied. Human dev-set reading sheets also include scores and confidence. We evaluate the annotation pipeline under this input condition.

Inference. We use vLLM (Kwon et al., 2023) at temperature 0.6 for self-consistency sampling (Wang et al., 2023). The 3-model × 3-run pipeline on the 500-paper main set consumes approximately 350 GPU-hours across two hardware configurations; the released artifact contains the archived prompts, parsed model outputs, and aggregation code (Appendix C; hardware accounting is in Table 8).

## 4.3 Multi-model consensus design

We select three open-weight LLMs from different groups and architecture families: Qwen 3.5 (Qwen Team, 2026), Gemma 4 (Google Deep-Mind, 2026b), and Nemotron 3-Super (NVIDIA, 2026) (Table 3).

The annotation pipeline runs each model three times and resolves to a per-model coding via intramodel voting (Stage 1). The three per-model codings are then resolved to a final consensus via intermodel voting (Stage 2). This exposes run-to-run variation within models and disagreement between model families (Figure 2).

<table><tr><td>Model</td><td>Group</td><td>Active params</td></tr><tr><td>Qwen 3.5</td><td>Alibaba</td><td>27 B (dense)</td></tr><tr><td>Gemma 4</td><td>Google</td><td>31 B (dense)</td></tr><tr><td>Nemotron 3-Super</td><td>NVIDIA</td><td>12 B / 120 B (MoE)</td></tr></table>

Table 3: The three open-weight LLMs used for the corpus annotation, spanning three research groups.

Act and attribute voting. At each stage, the act is the mode of available codes. Ties retain the first available run (Stage 1) or the first model in the fixed Qwen–Nemotron–Gemma order (Stage 2), with the vote counts preserved. Missing outputs are excluded, not treated as FLAG. For each applicable Level-2 field, only voters whose act matches the selected act contribute a yes/no vote, and at least two agreeing votes are required at each stage. Thus a single valid attribute cannot determine a model or final label. A tie or fewer than two supporting votes yields unresolved; fields outside the selected act are not\_applicable. These distinct states replace the raw output’s ambiguous NA. Table 6 distinguishes three supporting votes, two with dissent, two without a third valid vote, and unresolved attribute fields. Appendix D gives full conventions and the retained joint-schema exceptions.

Unlike prior schemes that train a dedicated classifier on labeled examples (Kennard et al., 2022; Purkayastha et al., 2025), ReviewMap uses LLMs directly as annotators against a written guideline, eliminating the train/test split and making schema iteration possible without retraining.

## 4.4 Pipeline validation on the dev set

We evaluate dev-set act labels against human and frontier-model references before interpreting the main-corpus analysis.

Five-human reference. An ML-expert author and four additional ML graduate annotators each code all 470 comments under the same guideline, without model outputs, one another’s labels, or adjudication. This comparison evaluates Level 1 only. We use the human mode as a comparison reference, not an error-free gold standard; the single 2–2–1 tie follows ASM > CAL > FLAG. Human agreement is Fleiss’ κ = .551 (mean pairwise Cohen’s κ = .555, range .468–.649; raw agreement 88.1%). All five agree on 352/470 comments; 102 of the remaining 118 involve ASM/CAL-only disagreement.

<table><tr><td>Human-reference comparison</td><td>Open-weight</td><td>Frontier</td></tr><tr><td>All 470: Cohen&#x27;s κ</td><td>.633</td><td>.686</td></tr><tr><td>raw agreement</td><td>90.0%</td><td>91.5%</td></tr><tr><td>Human-unanimous (n = 352)</td><td>96.6%</td><td>97.2%</td></tr><tr><td>Human-contested (n = 118)</td><td>70.3%</td><td>74.6%</td></tr></table>

Table 4: Level-1 dev validation against the five-human mode. All five annotators label all 470 items; subset rows report raw agreement for human-unanimous and contested items.

Frontier-LLM anchor. We additionally collect a three-model frontier consensus via manual chat-UI sessions: GPT-5.5 (ChatGPT) (OpenAI, 2026), Gemini 3.1 Pro (Google DeepMind, 2026a), and Claude Opus 4.6 (Anthropic, 2026), each run three times per paper (108 sessions). Their Level-1 labels use the same two-stage mode procedure, with GPT– Gemini–Opus as the panel tie order. Appendix D reports the dev output audit and individual-model comparisons.

Validation result. Both panels agree substantially with the five-human reference under the Cohen’s κ (Cohen, 1960) interpretation of Landis and Koch (1977): .633 for open-weight and .686 for frontier models (Table 4). Frontier–open-weight agreement is .836. Across five individual annotators and both panels, pairwise κ ranges from .505 to .658 (mean .581), overlapping the human–human range. Agreement approaches 97% on humanunanimous items and decreases on contested items, locating the remaining ambiguity. CAL precision/recall is .560/.824 for the open-weight panel and .613/.902 for the frontier panel; both predict 75 CALs versus 51 in the human reference. These results support act coding under the evaluated protocol. Open-weight panel agreement with the human reference is 95.8% on model-unanimous items and 56.5% otherwise (Table 15), motivating visible dissent for inspection.

## 4.5 Scaling to the 500-paper main set

We apply the three-model pipeline to all 500 papers. All 17,654 expected comments receive a consensus act. The final applicable-attribute coverage is 96.26% for P1, 95.95% for P2, and 91.42% for grounding; 1,115 comments retain at least one unresolved applicable field.

Only generated rows whose (paper, reviewer, comment\_id) occurs in the expected segmented corpus enter aggregation. Duplicate keys retain the last row, matching the saved-output loader; missing and extra keys are audited. Raw generation errors and deliberate aggregation abstentions are reported separately in §5.2.

![](images/d98430ddbda3aaf70210f80d30003796233da4e3112e0cafd9102a6ba7fc73c9.jpg)  
Output: act + vote counts; applicable attributes = yes / no / unresolved; other fields = not applicable.

Figure 2: The two-stage voting pipeline. The main corpus contains 17,654 comments from 500 papers. The same act-conditioned attribute rule applies at both stages. Act ties follow fixed orders and remain visible in vote metadata; unresolved attributes identify insufficient support for human inspection.
<table><tr><td>Metric</td><td>Value</td></tr><tr><td>Intra-Fleiss κ (Q/G/N)</td><td>0.84 / 0.96 / 0.78</td></tr><tr><td>Inter-Cohen κ (QG / QN / GN)</td><td>0.77 / 0.77 / 0.75</td></tr><tr><td>Act distribution % (ASM/CAL/FLAG)</td><td>80.5 / 17.4 / 2.2</td></tr><tr><td>Consensus split % (unan./2-of-3/no-maj)</td><td>88.4 / 11.3 / 0.3</td></tr></table>

Table 5: Act-layer reliability across the three openweight models (main set, n = 17,654 comments). Both intra- and inter-model agreement are Substantial or higher. Intra-model n is 17,654/17,653/17,642 for Q/G/N; inter-model n = 17,654 throughout.

Agreement measurement. Intra-model agreement uses Fleiss’ κ over three runs; inter-model agreement uses pairwise Cohen’s κ over model labels. Compared raters must all emit valid acts; attribute coefficients additionally require the same applicable act and binary values. Appendix D reports n, act-eligible coverage, observed agreement, and matched-item sensitivity. These coefficients measure label consistency under the annotation protocol.

## 5 Results

## 5.1 What ReviewMap reveals

Act-layer classification is reliable. The Level-1 act code is reliably classified by all three openweight LLMs (Table 5). Intra-model 3-run selfconsistency κ is Substantial-to-Almost-Perfect for every model; inter-model pairwise κ is uniformly Substantial across all three model pairs. Distributions converge tightly—ASM 80.5% / CAL 17.4% / FLAG 2.2%—and the three-way consensus achieves 88.4% unanimous, 11.3% 2-of-3, 0.3% no-majority.

![](images/058b3c11f2fc97e2fdbf71c491aede4b605f2c373a014c1f50145dcad2f86bdf.jpg)  
Figure 3: Agreement across acts and attributes. Intramodel points use Fleiss’ κ over three runs; inter-model points use Cohen’s κ over pairs of model labels. Mean inter-model agreement decreases across act, P1, P2, and grounding. Attribute samples are conditional on matching applicable acts and binary labels; denominators and observed agreement appear in Appendix D.

Agreement is lower for attributes. The Level-2 attributes (P1 / P2 / grounded) tell a different story. Mean inter-model κ decreases from .764 at the act layer to .577, .464, and .328, respectively (Figure 3). The trend varies across model pairs: Nemotron–Gemma rises from .407 on P2 to .480 on grounding. Intra-model Level-2 κ remains Almost Perfect for Gemma (.86–.89), is Moderateto-Substantial for Qwen (.55–.63), and Moderate for Nemotron (.46–.59). For grounding, withinmodel agreement exceeds between-model agreement in every pair even on matched items (Appendix D), consistent with model-specific decision thresholds. The attribute layer therefore structures response options that require human confirmation (§6.4). Three-model unanimity is about 60% for P1/P2 and 37.8% for grounding, while unresolved attribute fields increase from 3.7–4.0% to 8.6% (Table 6).

Explicit calibration is sparse in review text. The panel classifies 17.4% of comments as CAL (explicit value commitments), and the remainder assess claims or scholarly standards, or fall outside these two acts. At the review level, 17.0% of reviewer–paper instances contain no panel-coded CAL; the median reviewer issues a single CAL per paper (mean 1.57; 81.4% issue ≤ 2). Across 21 primary areas (11–54 papers each), comment-level CAL shares range from 12.4% to 21.7% (Table 16), describing variation within ICLR rather than crossvenue generalization.

<table><tr><td>Field</td><td>n</td><td>3/3</td><td>2/3</td><td>2/2</td><td>Unres.</td></tr><tr><td>P1 (ASM)</td><td>14,203</td><td>59.46</td><td>27.49</td><td>9.31</td><td>3.74</td></tr><tr><td>P2 (ASM)</td><td>14,203</td><td>60.87</td><td>27.42</td><td>7.66</td><td>4.05</td></tr><tr><td>Grounded (CAL)</td><td>3,067</td><td>37.82</td><td>35.31</td><td>18.29</td><td>8.58</td></tr></table>

Table 6: Final attribute support tiers (% of applicable n). 3/3 = three agreeing valid votes; 2/3 = two agreeing with one dissent; 2/2 = two agreeing with no third valid, same-act vote. Unres. = tie or insufficient valid support.

<table><tr><td>Observation</td><td>Value</td></tr><tr><td>Act distribution (ASM / CAL / FLAG) Zero-CAL reviewer-paper instances Per-reviewer CAL (median / mean)</td><td>80.5 / 17.4 / 2.2 % 333/1,955 (17.0%) 1/1.57</td></tr><tr><td>reviewers with ≤ 2 CALs</td><td>81.4%</td></tr><tr><td>ASM/CAL/FLAG share by section (%)</td><td></td></tr><tr><td>Strengths Weaknesses</td><td>65.1 / 34.5 / 0.4</td></tr><tr><td>Questions</td><td>85.5 / 12.2 / 2.2 89.2 / 6.9 / 3.9</td></tr></table>

Table 7: Corpus-level findings from final three-model consensus acts. Section shares use 5,269 Strengths, 7,390 Weaknesses, and 4,995 Questions comments; review statistics use 1,955 reviewer–paper instances.

Strengths are calibration-heavy; Weaknesses emphasize assessment. Where CAL does appear, it concentrates in Strengths: 34.5% of Strengths comments are CAL versus 12.2% in Weaknesses and only 6.9% in Questions—a nearly 3× gap between Strengths and Weaknesses (Table 7). Section headers alone do not identify evaluative acts: even Strengths sections are 65.1% ASM (most “strengths” assess claims or scholarly standards), and Weaknesses sections are 85.5% ASM (reviewers tend to assess a weakness against those criteria rather than explicitly judge its bearing on the contribution’s value). These panel codings suggest that reviewers express value judgments more often when complimenting than when critiquing. On dev, the human mode labels 1/189 Weaknesses comments as CAL versus 14/189 for the open-weight panel; both show higher CAL shares in Strengths, while the magnitude of the corpus contrast remains panel-dependent (Appendix D).

These four observations motivate §6: the act layer is reliably automatable but the attribute layer requires care, and calibration is both sparse and section-asymmetric. Authors, reviewers, and area chairs all need a view of where value commitments live, not just how many comments a section contains—the basis for the audience-specific affordances.

## 5.2 Per-model failure analysis

Table 17 audits missing/extra IDs, duplicates, and applicable attributes emitted as NA. Across runs, Qwen omits P2 more than P1 (313 versus 84 applicable NAs); Gemma shows the reverse (1,454 P1 versus 520 P2). Generation failures are distinct from the two-vote rule’s unresolved outputs.

Aggregation covers every expected act. Of all comments, 16,155/17,654 (91.51%) are non-FLAG with all applicable attributes resolved. Independent attribute-field voting leaves 40/14,203 ASM comments with P1=P2=no, violating the guideline’s at-least-one-yes constraint. The released labels retain these joint-schema exceptions alongside resolved-field coverage.

## 6 Design Implications

The measured boundary in §5 connects three inspection tasks: reviewers examine their evaluative targets and grounds, authors identify what responses must justify, and area chairs trace what remains disputed. The sketches retain model votes and require human confirmation of attributederived suggestions. Appendix G follows a concern from contested act coding to a manually interpreted response, connecting these role-specific questions.

## 6.1 For reviewers:

## assessment-then-calibration workflow

The framework provides a structural self-check: does the review explicitly evaluate contribution value alongside claims and scholarly standards? In our corpus, 17% of reviewer–paper instances contain no consensus-coded calibration in the sampled Strengths, Weaknesses, and Questions comments. This invites inspection of the explicit links between assessments and value judgments. Our practical reviewer guideline (Appendix E) maps assessment to Claim–Evidence Alignment and Scholarly Standard Assessment, and calibration to ground your judgment. The corpus norms in §5.1 (median 1 calibration per review; 81.4% of reviewers issue ≤ 2) provide descriptive context for that inspection.

## 6.2 For authors: what a response must justify

The distinction identifies what a response must justify. Assessment comments call for addressing claim support or scholarly standards; calibration comments call for explaining the contribution’s significance or contextualizing its value. Either response may combine evidence and argument. The coding routes authors toward five candidate strategies (LEVERAGE, DEFEND, CONCEDE, AC-CEPT, CLARIFY); the author selects and develops the response. Appendix F specifies the mapping and quantifies its conditional attribute dependence and unresolved cases. This is a prospective decision-support proposal; large-scale validation against author-side rebuttal corpora, complementing author-input-driven response generation (Ruan and Gurevych, 2026), remains future work.

## 6.3 For area chairs: discussion-phase panel view

The same distinction lets area chairs inspect how a panel assesses the paper and judges its contribution, beyond its raw scores. Wang et al. (2026) weight conflicting opinions by support and review quality to generate meta-reviews; our proposed act-based view supports human inspection. An area chair can see which reviews explicitly calibrate contribution value and which focus on claims or standards, where calibrations cluster across the panel, and which comments the LLM panel itself disagreed on (the 11.3% with 2-of-3 consensus warrant closer reading). A discussion-phase dashboard in Appendix G traces two concerns in a polarized four-reviewer panel (2/4/6/8). The two-stage aggregation marks one concern as grounded and the other as unresolved; both remain available for the AC to inspect alongside manually linked author responses.

## 6.4 Responsible use

Each affordance above relies on the act layer, where LLM consensus is reliable under the measured conditions. Attribute-level outputs are advisory and require human confirmation. Any user-facing implementation should display the source comment, individual model votes, and agreement tier; mark 2-of-3 acts as contested; and treat three-way act disagreements as unresolved despite the corpus’s deterministic tie-break. Unresolved attributes remain unresolved, and every attribute-derived menu or filter requires confirmation, including resolved values with dissenting votes. Users must be able to override codings. These sketches do not generate reviews, score reviewers, rank papers, or recommend decisions. Making the measured boundary visible is the contribution: stakeholders can inspect and revise the classifications supporting their judgments.

Alternative views. Criterion scores separate soundness and significance but do not identify the judgments or grounds expressed in individual comments. Aspect and pragmatic labels can complement act coding: the same topic or request may concern claim support or contribution value, requiring different justifications. Better models or guidelines may reduce disagreement; the measured boundary is conditional. Agreement about which act a comment expresses does not require agreement about the paper’s merit. Our position is that NLP support should expose evaluative targets, grounds, and coding uncertainty for human deliberation.

## 7 Conclusion

ReviewMap makes assessment of claims and scholarly standards and calibration of contribution value explicit at the comment level. Across 500 papers and 17,654 comments, its codings reveal sparse explicit calibration (17% zero-CAL reviewer–paper instances) and section asymmetry (a nearly 3× higher CAL share in Strengths than Weaknesses). The measured reliability boundary and five-human validation support using act classification to structure inspection, while finer attributes remain advisory. The reviewer guideline, author scaffold, and AC dashboard translate that boundary across roles. We argue that exposing what LLMs can reliably structure and where human judgment must remain explicit defines a more honest mission for NLP in peer review than full automation, complementing the rigor call of Baumann et al. (2026).

## Limitations

Our empirical analysis covers English reviews from ICLR 2026, with Level-1 human validation on 470 development comments. The 17% zero-CAL finding (§5) concerns explicit calibration in the sampled review sections; it provides no direct evidence about reviewers’ unstated reasoning or the validity of their recommendations. CAL overprediction relative to the dev human reference may affect both comment-level prevalence and zero-CAL counts; the dev comparison cannot calibrate main-corpus estimates. Generalization to other venues remains to be evaluated. The Yes–Yes–Yes donation protocol in NLPeer (Dycke et al., 2023) and crossvenue corpora such as Re2 (Zhang et al., 2025) offer precedents for extending this analysis with appropriate data access.

The five-human study supports act classification under the operational guideline, with interpretive disagreement on 118 of 470 comments. Human-majority labels therefore provide an imperfect reference, and agreement on the unanimous subset should be interpreted alongside the full-set results. Human validation is limited to Level 1. Attribute agreement is conditional on shared acts and available binary values, with different samples and prevalences across fields (§5). The observed agreement differences are specific to this protocol. Annotation accuracy, intrinsic subjectivity, and shared model bias require separate evidence. The guideline’s traceable-evidence criterion and the wrapper’s reference-to-assessment wording differ for grounding (Appendix B); their contribution to disagreement remains untested. Sensitivity to alternative guidelines, a concern in LLM annotation (Casanova et al., 2026), also remains to be evaluated.

The three affordances illustrate potential uses of the framework, including a worked dashboard case (Appendix G); their effects on human decisions require further study. Rule-based segmentation can yield fragmentary comments, and the attributes capture different aspects of review text: P1/P2 identify assessment targets, while grounding identifies stated support for value judgments. These annotations describe review structure rather than verify the correctness of an assessment or the merit of a paper. Their use in decision support retains the human confirmation described in §6.4.

## Ethical considerations

The source papers and reviews are publicly available ICLR 2026 submissions on OpenReview, following prior work on public peer-review corpora (Kennard et al., 2022; D’Arcy et al., 2024; Dycke et al., 2023). The five-human study labels these existing comments at Level 1, with anonymized annotator identifiers. The four additional annotators were ML graduate students recruited through personal contacts in neighboring laboratories. Participation was voluntary and unpaid, with acknowledgment promised upon acceptance. They agreed to research use and public sharing of their annotations and received the annotation guideline without additional task training. No institutional ethics review was sought for this study. The released artifact retains paper identifiers and the venue’s public pseudonymous reviewer identifiers (e.g., Reviewer\_NydX) unchanged, matching the DIS-APERE / ARIES convention; we do not perform additional pseudonymization and do not attempt to de-anonymize reviewers. The case study in Appendix G quotes short illustrative fragments and uses short forms of the venue-assigned reviewer pseudonyms (e.g., LnWH) for readability; full review and rebuttal threads remain accessible at the OpenReview source URLs.

Risks and intended-use scope. ReviewMap is intentionally a classification framework, not a generation system: the released code does not synthesize review or rebuttal text. We share the position of Dycke et al. (2023) and Baumann et al. (2026) that automatic review generation is a category of NLP application where benefits are unclear and risks to academic quality assurance are real. The reviewerside, author-side, and area-chair-side applications in §6 are diagnostic decision-support layers; they do not write text on behalf of users. The released corpus is not intended for reviewer profiling, for automated decision-making in peer-review venues, or for training general-purpose language models.

## Acknowledgments

We thank the four additional annotators for their voluntary contributions to the development-set validation.

AI assistance. The authors used ChatGPT (OpenAI) to polish and rephrase their notes and drafts; some passages were drafted from those notes. AI assistants also supported literature discovery and source checks. Building on the authors’ analysis code, Codex (OpenAI) assisted with analysis and reproduction code and numerical checks. Claude (Anthropic) provided separate checks by rerunning reproduction code and comparing reported results, tables, and cross-references against the artifacts. The authors developed the research questions, framework, and annotation guideline, made the final methodological and interpretive decisions, and reviewed and take responsibility for all content.

## References

Shakiba Amirshahi, Sajad Ebrahimi, Hai Son Le, Negar Arabzadeh, and Ebrahim Bagheri. 2026. Judging a review by its cover: A reliability analysis of LLM-based peer review evaluation metrics. Preprint, arXiv:2609.23264. Accepted at CIKM 2026.

Anthropic. 2026. Claude Opus 4.6 System Card.

Joachim Baumann, Jiaxin Pei, Sanmi Koyejo, and Dirk Hovy. 2026. Position: Stop automating peer review without rigorous evaluation. In Proceedings of the 43rd International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research, pages 169349–169380. PMLR.

Anna Bavaresco, Raffaella Bernardi, Leonardo Bertolazzi, Desmond Elliott, Raquel Fernández, Albert Gatt, Esam Ghaleb, Mario Giulianelli, Michael Hanna, Alexander Koller, Andre Martins, Philipp Mondorf, Vera Neplenbroek, Sandro Pezzelle, Barbara Plank, David Schlangen, Alessandro Suglia, Aditya K Surikuchi, Ece Takmaz, and Alberto Testoni. 2025. LLMs instead of human judges? a large scale empirical study across 20 NLP evaluation tasks. In Proceedings ofthe 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 2: Short Papers), pages 238–255, Vienna, Austria. Association for Computational Linguistics.

Prabhat Kumar Bharti, Meith Navlakha, Mayank Agarwal, and Asif Ekbal. 2024. PolitePEER: does peer review hurt? A dataset to gauge politeness intensity in the peer reviews. Language Resources and Evaluation, 58(4):1291–1313.

Etienne Casanova, Rafal Kocielnik, and R. Michael Alvarez. 2026. On the limits of LLM adaptability: Impact of model-internalized priors on annotation task performance. In Proceedings ofthe 43rd International Conference on Machine Learning, volume 306 of Proceedings ofMachine Learning Research, pages 11976–12000. PMLR.

Jacob Cohen. 1960. A coefficient of agreement for nominal scales. Educational and Psychological Measurement, 20(1):37–46.

Corinna Cortes and Neil D. Lawrence. 2021. Inconsistency in Conference Peer Review: Revisiting the 2014 NeurIPS Experiment. arXiv preprint arXiv:2109.09774.

Delphine S. Courvoisier, Diana Buitrago-Garcia, Clé- ment P. Buclin, Nils Bürgisser, Michele Iudici, and Denis Mongin. 2026. Beyond human gold standards: A multimodel framework for automated abstract classification and information extraction. Research Synthesis Methods, 17(2):365–377.

Mike D’Arcy, Alexis Ross, Erin Bransom, Bailey Kuehl, Jonathan Bragg, Tom Hope, and Doug Downey. 2024. ARIES: A corpus of scientific paper edits made in response to peer reviews. In Proceedings of the 62nd Annual Meeting of the Association for Computational

Linguistics (Volume 1: Long Papers), pages 6985– 7001, Bangkok, Thailand. Association for Computational Linguistics.

Nils Dycke, Ilia Kuznetsov, and Iryna Gurevych. 2023. NLPeer: A unified resource for the computational study of peer review. In Proceedings ofthe 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5049– 5073, Toronto, Canada. Association for Computational Linguistics.

Fabrizio Gilardi, Meysam Alizadeh, and Maël Kubli. 2023. ChatGPT outperforms crowd workers for text-annotation tasks. Proceedings of the National Academy ofSciences, 120(30):e2305016120.

Google DeepMind. 2026a. Gemini 3.1 Pro Model Card.

Google DeepMind. 2026b. Gemma 4 Model Card.

Yanzhu Guo, Guokan Shang, Virgile Rennard, Michalis Vazirgiannis, and Chloé Clavel. 2023. Automatic analysis of substantiation in scientific peer reviews. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 10198–10216, Singapore. Association for Computational Linguistics.

Xinyu Hua, Mitko Nikolov, Nikhil Badugu, and Lu Wang. 2019. Argument mining for understanding peer reviews. In Proceedings ofthe 2019 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 2131–2137, Minneapolis, Minnesota. Association for Computational Linguistics.

Yiqiao Jin, Qinlin Zhao, Yiyang Wang, Hao Chen, Kaijie Zhu, Yijia Xiao, and Jindong Wang. 2024. AgentReview: Exploring peer review dynamics with LLM agents. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 1208–1226, Miami, Florida, USA. Association for Computational Linguistics.

Dongyeop Kang, Waleed Ammar, Bhavana Dalvi, Madeleine van Zuylen, Sebastian Kohlmeier, Eduard Hovy, and Roy Schwartz. 2018. A dataset of peer reviews (PeerRead): Collection, insights and NLP applications. In Proceedings ofthe 2018 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 1647–1661, New Orleans, Louisiana. Association for Computational Linguistics.

Neha Nayak Kennard, Tim O’Gorman, Rajarshi Das, Akshay Sharma, Chhandak Bagchi, Matthew Clinton, Pranay Kumar Yelugam, Hamed Zamani, and Andrew McCallum. 2022. DISAPERE: A dataset for discourse structure in peer review discussions. In Proceedings ofthe 2022 Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 1234–1249, Seattle, United States. Association for Computational Linguistics.

Jaeho Kim, Yunseok Lee, and Seulki Lee. 2025. Position: The AI Conference Peer Review Crisis Demands Author Feedback and Reviewer Rewards. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, pages 81634–81651. PMLR.

Ilia Kuznetsov, Osama Mohammed Afzal, Koen Dercksen, Nils Dycke, Alexander Goldberg, Tom Hope, Dirk Hovy, Jonathan K. Kummerfeld, Anne Lauscher, Kevin Leyton-Brown, Sheng Lu, Mausam, Margot Mieskes, Aurélie Névéol, Danish Pruthi, Lizhen Qu, Roy Schwartz, Noah A. Smith, Thamar Solorio, and 5 others. 2024. What Can Natural Language Processing Do for Peer Review? arXiv preprint arXiv:2405.06563.

Ilia Kuznetsov, Rohan Nayak, Alla Rozovskaya, and Iryna Gurevych. 2026. Is peer review really in decline? Analyzing review quality across venues and time. Preprint, arXiv:2601.15172. To appear in EMNLP 2026, main conference.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient Memory Management for Large Language Model Serving with PagedAttention. In Proceedings ofthe 29th Symposium on Operating Systems Principles, pages 611–626.

J. Richard Landis and Gary G. Koch. 1977. The Measurement of Observer Agreement for Categorical Data. Biometrics, 33(1):159–174.

Weixin Liang, Zachary Izzo, Yaohui Zhang, Haley Lepp, Hancheng Cao, Xuandong Zhao, Lingjiao Chen, Haotian Ye, Sheng Liu, Zhi Huang, Daniel Mcfarland, and James Y. Zou. 2024. Monitoring AI-modified content at scale: A case study on the impact of Chat-GPT on AI conference peer reviews. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 29575–29620. PMLR.

Sheng Lu, Ilia Kuznetsov, and Iryna Gurevych. 2025. Identifying aspects in peer reviews. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2025, pages 6145–6167, Suzhou, China. Association for Computational Linguistics.

NVIDIA. 2026. Nemotron 3 Super: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning. arXiv preprint arXiv:2604.12374.

OpenAI. 2026. GPT-5.5 System Card.

Sukannya Purkayastha, Zhuang Li, Anne Lauscher, Lizhen Qu, and Iryna Gurevych. 2025. LazyReview: A dataset for uncovering lazy thinking in NLP peer reviews. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3280–3308, Vienna, Austria. Association for Computational Linguistics.

Qwen Team. 2026. Qwen3.5-27B Model Card.

Qian Ruan and Iryna Gurevych. 2026. Author-in-theloop response generation and evaluation: Integrating author expertise and intent in responses to peer review. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 12390–12418, San Diego, California, United States. Association for Computational Linguistics.

David Soergel, Adam Saunders, and Andrew McCallum. 2013. Open scholarship and peer review: A time for experimentation. ICML 2013 Peer Review Workshop.

Nitya Thakkar, Mert Yuksekgonul, Jake Silberg, Animesh Garg, Nanyun Peng, Fei Sha, Rose Yu, Carl Vondrick, and James Zou. 2026. A large-scale randomized study of large language model feedback in peer review. Nature Machine Intelligence, 8(3):326– 336.

Pat Verga, Sebastian Hofstatter, Sophia Althammer, Yixuan Su, Aleksandra Piktus, Arkady Arkhangorodsky, Minjie Xu, Naomi White, and Patrick Lewis. 2024. Replacing Judges with Juries: Evaluating LLM Generations with a Panel of Diverse Models. arXiv preprint arXiv:2404.18796.

Xinzhe Wang, Fei Tao, Jiang Xie, Hong Yu, and Ye Wang. 2026. When evidence conflicts: Reliability-aware meta-review generation. Preprint, arXiv:2609.24028. Accepted to Findings of EMNLP 2026.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2023. Self-Consistency Improves Chain of Thought Reasoning in Language Models. In International Conference on Learning Representations.

Sihong Wu, Yiling Ma, Yilun Zhao, Tiansheng Hu, Owen Jiang, Manasi Patwardhan, and Arman Cohan. 2026. RbtAct: Rebuttal as supervision for actionable review feedback generation. In Findings of the Association for Computational Linguistics: ACL 2026, pages 33965–33992, San Diego, California, United States. Association for Computational Linguistics.

Weizhe Yuan, Pengfei Liu, and Graham Neubig. 2022. Can We Automate Scientific Reviewing? Journal of Artificial Intelligence Research, 75:171–212.

Daoze Zhang, Zhijian Bao, Sihang Du, Zhiyi Zhao, Kuangling Zhang, Dezheng Bao, and Yang Yang. 2025. Re2: A Consistency-ensured Dataset for Fullstage Peer Review and Multi-turn Rebuttal Discussions. arXiv preprint arXiv:2505.07920.

## A Annotation Guideline for ReviewMap

This appendix reproduces the annotation guideline used in the experiments (§4.2, §5). It accompanies the system and output-format instructions in Appendix B. The guideline refers generally to reading the abstract, introduction, and conclusion; the experimental prompts supply the title, abstract, keywords, and review data, with claims extracted from the abstract. The term evaluative function below denotes the evaluative acts discussed in the main text.

## Annotation Guideline (released artifact)

## A.1 Purpose

This guideline provides instructions for classifying peer-review comments from ML/CS venues by the evaluativefunction each comment performs: Assessment (ASM) or Calibration (CAL).

• ASM evaluates what the paper does or how it does it — whether claims are supported and standards are met.

• CAL evaluates why the work matters — the contribution’s novelty, significance, importance, or venue fit.

## A.2 Output format

Each comment receives the following fields:

<table><tr><td>Field</td><td>Values</td><td>When</td></tr><tr><td>code</td><td>ASM CAL</td><td>/ Always</td></tr><tr><td>claim_relevant</td><td>yes / no</td><td>ASM only</td></tr><tr><td>standard_relevant</td><td>yes / no</td><td>ASM only</td></tr><tr><td>grounded</td><td>yes / no</td><td>CAL only</td></tr><tr><td>note</td><td>free text</td><td>When deliberating be- tween codes, or when recording grounding evi-</td></tr></table>

Coding unit. Each individual comment as delimited by the reviewer under Strengths (S1, S2, . . . ), Weaknesses (W1, W2, . . . ), or Questions (Q1, Q2, . . . ). Code only official reviews.

## A.3 Identifying claims (required before coding)

Before coding any comments, read the paper’s abstract, introduction, and conclusion and list the main claims. A claim is any assertion the paper explicitly makes — empirical results, scope, design properties, or performance characteristics.

Examples of claims:

• “achieves state-of-the-art performance on X”

• “our method is model-agnostic and can be applied to any architecture”

This claim list is your reference for determining claim\_relevant below.

## A.4 Category definitions

## A.4.1 Assessment (ASM)

A comment is ASM when it evaluates what the paper does or how it does it — whether claims are supported, methods are sound, standards are met, or presentation is adequate. ASM examines the paper’s content against its own assertions or against accepted scholarly practices.

To recognize ASM, ask two questions:

• Does this comment evaluate whether a specific claim is supported by evidence? (paper vs. its own assertions)

• Does this comment evaluate whether the paper meets a field-wide standard? (paper vs. accepted practices)

If either answer is yes, the comment is ASM. ASM encompasses both positive and negative assessments:

• “The analysis clearly demonstrates the contribution of each proposed component.” — evaluates what the paper did (claim support).

• “The paper claims real-time performance but the reported latency excludes preprocessing time.” — evaluates what the paper claimed vs. showed.

• “The paper is well-written and easy to follow.” — evaluates how the paper presents its work.

• “The experimental setup is not described in sufficient detail for reproduction.” evaluates what is missing (standard).

## A.4.2 Calibration (CAL)

A comment is CAL when it evaluates why the work matters — positioning the contribution in the field by judging its novelty, significance, importance, or venue fit.

CAL encompasses both positive and negative calibrations:

• “This is the first computational work to link X and Y.” — positions the work as novel; the point is why it matters, not what was done.

• “The novelty is limited — similar approaches have been widely explored.” judges why the work does not matter (novelty).

• “This will change how the field thinks about X.” — judges why the work matters (significance).

• “The contribution is incremental.” judges why the work does not matter (significance).

## A.5 Distinguishing ASM vs. CAL

Use the note field to record any secondary dimension you considered. Interrogative form does not determine the label — questions should be coded by the evaluative function they perform.

<table><tr><td colspan="2">Comment Code Reasoning</td></tr><tr><td>&quot;This is an invalu- CAL Mixed: ultimately able contribution with careful detail and rigor&quot;</td><td>about why the work matters (invaluable), not about the detail it- self</td></tr><tr><td>pressive but the com- parison with prior work is unfair&quot;</td><td>“The results are im- ASM Mixed: ultimately about what the paper does (unfair compar- ison), not about im- pressiveness</td></tr></table>

## A.6 ASM attributes

For each comment coded ASM, record two attributes:

claim\_relevant. Does the comment evaluate the support for a specific claim listed above? (yes / no). This checks the paper against its own assertions — did the paper do what it said?

standard\_relevant. Does the comment evaluate whether the paper meets a field-wide standard that applies regardless of what the paper claims? (yes / no). This checks the paper against accepted scholarly practices — did the paper do what is expected?

Both can be yes simultaneously. At least one must be yes; if neither applies, reconsider whether the comment is CAL.

## A.7 CAL attribute

For each comment coded CAL, record:

grounded. Does the review text contain traceable evidence supporting the value judgment? (yes / no). If yes, briefly note the evidence in the note field.

Traceable evidence includes the reviewer’s own assessment of the paper, a specific reference to the paper’s content, or a comparison with other work. Identifying the feature being evaluated does not by itself count as evidence.

• Grounded: “The improvement is marginal — only X% over the compared method.” (evidence: inline, X% figure)

• Ungrounded: “This is a strong contribution to the field.”

## A.8 Uncertain cases (FLAG)

If you cannot determine whether a comment is ASM or CAL, output FLAG as the code.

## B Annotation Prompt and Worked Examples

The template below specifies the system instructions, input fields, claim-extraction task, and output format. The guideline placeholder is filled with the text in Appendix A. Each prompt groups a paper’s review comments under reviewer score and confidence headers. Paper content is limited to the title, abstract, and keywords, and claims are extracted from the abstract. For grounding, the wrapper refers to a specific assessment; the accompanying guideline gives the broader traceable-evidence examples. The template preserves both formulations used in the experiment. Appendix A’s act table lists ASM/CAL; its fallback rule and the wrapper additionally permit FLAG when the act cannot be determined (Figure 4).

The ASM example in §3 is paper 6wA4qpyyU9, reviewer CW52, comment S2; the CAL example is paper ZfCCwJ4Wcs, reviewer jvoP, comment S2. Quotations omit list markers and terminal punctuation. Their final labels are ASM/P1=yes/P2=no and CAL/grounded=no, respectively, with three-model agreement on each applicable field. The output format assigns one row to each expected comment; aggregated records also include vote counts and unresolved states.

Human reading materials. The archived human sheets contain the title, abstract, sectioned review comments, and reviewer scores and confidence, without full-paper attachments. The four additional annotators’ cover note specifies Level 1 judgments from the abstract and review context. Their sheets omit the keywords, abstract claim-extraction task, and Level 2 fields present for the author anchor and model prompts. All use the shared guideline. Although it refers to the introduction and conclusion, these were not supplied; external consultation was not recorded.

Populated input/output example. The following excerpt pairs the CAL example above with its actual Qwen run 1 output. We omit the system message, paper context, other comments, generated preamble and claims, and other output rows. The note is from this individual response, not a consensus explanation. Full populated requests and parsed run outputs accompany the artifact.

![](images/c87c76990835e9b457fa06b4b68db9f1b06fe0693497b7bde8bfa8dddacb30fc.jpg)

The final consensus agrees on CAL/grounded=no; raw NA becomes not\_applicable in the aggregate schema.

## C Reproducibility

Implementation. The three open-weight models were served with vLLM (BF16), with temperature 0.6. The provenance record lists output limits of Qwen 16K, Nemotron 20K, and Gemma 4K; these are not a complete history of the requests that produced the main annotations. Recovery records document 14K retries for 57 Qwen, 43 Nemotron, and 4 Gemma paper–runs, followed by 20K retries for 10 of those Nemotron paper–runs. Complete initial per-request logs are unavailable. Available server logs record top-p = 0.95 for all three, topk = 20 for Qwen, and 64 for Gemma; other sampling settings followed the served model configurations where documented. No hyperparameter search was performed. Archived prompts, parsed outputs, and analysis code support deterministic reproduction of aggregation, not identical fresh inference.

Hardware and compute budget. Annotation used two hardware configurations: a workstation with dual NVIDIA RTX Pro 6000 Blackwell GPUs (96 GB each), with Qwen and Gemma on one GPU each, and a server with dual NVIDIA B200 GPUs (192 GB each), with Nemotron at tensor-parallel TP=2. Table 8 reports GPU-hours, including both GPUs for Nemotron. The 3-model × 3-run × 500- paper annotation required approximately 352 GPUhours, or 42.4 GPU-minutes per paper across nine model passes. Qwen timings were measured from per-run logs. Gemma and Nemotron timings were estimated from truncation-retry samples at the production output limits because complete timing logs were unavailable.

![](images/20a63dc06a5f660e24b8bfd561a2a70afea043f00f40eadbf96c68b06a1c269b.jpg)  
Figure 4: Annotation prompt template. Placeholders are filled from the released inputs and annotation guideline.

Sampling and reproduction. Corpus sampling (§4.1) uses SEED=42. Reproduction starts from the released, fixed segmented comments. Aggregation deterministically maps the archived stochastic outputs to the reported labels, using fixed voter orders for act ties and explicit attribute abstention. The artifact includes both the run outputs and aggregated labels; fresh inference may produce different annotations.

Data and analysis reproduction. Data/main/aggregation/ contains final labels and summary statistics; Data/main/runs/ contains the three parsed runs per model. The corpus and all 512 main/dev prompts are accompanied by an input specification and checksums. Data/dev/validation/ contains anonymized H1–H5 act labels, dev model outputs, and the annotation protocol; H1 denotes the author anchor. Original final act labels are retained in Data/reference/ to check preservation under reaggregation. Per-model aggregate files are regenerated from the archived runs.

The artifact README provides one command for deterministic aggregation, development-set validation, and the corpus and sensitivity analyses. These calculations run without a GPU or network using Python 3.10 or newer and its standard library. The software also includes prompt reconstruction and optional plot rendering with matplotlib. A checksum manifest covers the released files.

## D Agreement and Validation Details

Aggregation conventions. All main-set calculations use the 17,654 segmented-comment identifiers. For duplicate identifiers, the last row in inputfile order is retained; extra identifiers are excluded and missing identifiers are recorded. Stage 1 takes the mode of observed run labels. After duplicate resolution, every retained expected row has a valid act. Stage 2 takes the mode of available valid model labels. Ties follow the first available run in order 1–3 or the model order Qwen–Nemotron–Gemma. Items with no available act remain null; all items in the main set have a model and final act label. Vote counts are retained for the 49 three-way panel ties.

For a field conditional on ASM or CAL, a voter participates only when its act matches the selected act and its field value is yes or no. At least two matching binary votes are required independently at each stage. Consequently, a Stage-1 unresolved field supplies no binary vote at Stage 2. Unresolved records either a binary tie or insufficient valid votes; not applicable denotes a field outside the selected act. Neither state is counted as no. The released records preserve the support counts and reasons.

The guideline requires at least one ASM field to be yes, but independent field voting can yield $\mathrm { P 1 } { = } \mathrm { P } 2 { = } \mathrm { n o } .$ . This occurs for 126, 79, and 1 comments in Qwen, Gemma, and Nemotron, respectively, and 40 in the final consensus. These labels are retained and reported as violations of the joint constraint, separately from unresolved fields.

Conditional agreement. Table 9 distinguishes within-model Fleiss’ κ (three raw runs) from between-model Cohen’s κ (two emitted model labels). For acts, $N _ { \mathrm { e l i g } } = 1 7 , 6 5 4$ and n contains only complete valid ratings. For attributes, $N _ { \mathrm { e l i g } }$ contains items whose compared acts all match the applicable act; n further requires complete binary field values. Coverage is $n / N _ { \mathrm { e l i g } }$ , and observed agreement is pairwise agreement (averaged over rater pairs for Fleiss). The artifact also reports n/17,654, undefined-coefficient reasons, and threemodel Fleiss coefficients. These are measures of label consistency under the annotation protocol.

Matched-item comparison. Within- and between-model coefficients need not use identical items in Table 9. Table 10 therefore restricts each model-pair/field comparison to items with all six valid raw ratings and both valid model labels, with every act applicable for attributes. On grounding, each model’s intra-κ exceeds the pair’s inter-κ on the same items. This ordering is not universal for P1/P2 and does not identify a causal source of model disagreement.

Table 11 reports final field counts, support tiers, and resolved coverage.

Output quality. Table 12 breaks down Table 17 by run, each with 17,654 expected comments. Extra identifiers are counted once per run; duplicates count additional rows. Applicable-field NAs are counted after retaining the last duplicate row. All retained expected rows have valid acts. P1=P2=no outputs are reported separately as violations of the ASM constraint. Unresolved aggregation results are recorded separately from generation errors.

For example, Qwen run 3 adds Q2 for paper 32QQlzm9ft, reviewer cpQL, whose input IDs are

<table><tr><td>Model (BF16)</td><td>Hardware</td><td>GPUs</td><td>Per single pass (GPU-min)</td><td>Per paper (3 runs) (GPU-min)</td><td>Full pipeline (GPU-h)</td></tr><tr><td>Qwen 3.5–27B</td><td>RTX Pro 6000 (96 GB)</td><td>1</td><td>8.5</td><td>25.5</td><td>212.3</td></tr><tr><td>Gemma 4-31B</td><td>RTX Pro 6000 (96 GB)</td><td>1</td><td>0.7</td><td>2.0</td><td>16.3</td></tr><tr><td>Nemotron 3-Super 120B</td><td>NVIDIA B200 (192 GB)</td><td>2</td><td>5.0</td><td>14.9</td><td>124.0</td></tr><tr><td>Total (all 3 models)</td><td>two hardware configurations</td><td></td><td>14.2</td><td>42.4</td><td>≈352</td></tr></table>

Table 8: Compute budget for the main-set annotation. Per single pass = one paper through one model in one SC run; per paper (3 runs) = one paper through all three SC runs of one model; full pipeline = the entire 3-model × 3-run × 500-paper annotation that produced the released artifact. Nemotron’s TP=2 contributes two GPU-hours per elapsed hour.
<table><tr><td>Model / pair</td><td>Field</td><td>κ</td><td>n</td><td> $N _ { \mathrm { e l i g } }$ </td><td>Coverage (%)</td><td>Agreement</td></tr><tr><td colspan="7">Intra-model Fleiss&#x27; κ over three raw runs</td></tr><tr><td>Qwen</td><td>Act</td><td>0.842</td><td>17,654</td><td>17,654</td><td>100.00</td><td>0.950</td></tr><tr><td>Qwen</td><td>P1</td><td>0.630</td><td>13,532</td><td>13,613</td><td>99.40</td><td>0.815</td></tr><tr><td>Qwen</td><td>P2</td><td>0.563</td><td>13,350</td><td>13,613</td><td>98.07</td><td>0.827</td></tr><tr><td>Qwen</td><td>Grounded</td><td>0.546</td><td>2,607</td><td>2,609</td><td>99.92</td><td>0.905</td></tr><tr><td>Gemma</td><td>Act</td><td>0.959</td><td>17,653</td><td>17,654</td><td>99.99</td><td>0.987</td></tr><tr><td>Gemma</td><td>P1</td><td>0.891</td><td>13,156</td><td>13,945</td><td>94.34</td><td>0.946</td></tr><tr><td>Gemma</td><td>P2</td><td>0.860</td><td>13,652</td><td>13,945</td><td>97.90</td><td>0.936</td></tr><tr><td>Gemma</td><td>Grounded</td><td>0.864</td><td>2,939</td><td>2,946</td><td>99.76</td><td>0.934</td></tr><tr><td>Nemotron</td><td>Act</td><td>0.780</td><td>17,642</td><td>17,654</td><td>99.93</td><td>0.921</td></tr><tr><td>Nemotron</td><td>P1</td><td>0.581</td><td>12,589</td><td>12,686</td><td>99.24</td><td>0.791</td></tr><tr><td>Nemotron</td><td>P2</td><td>0.456</td><td>12,643</td><td>12,686</td><td>99.66</td><td>0.821</td></tr><tr><td>Nemotron</td><td>Grounded</td><td>0.587</td><td>2,509</td><td>2,509</td><td>100.00</td><td>0.794</td></tr><tr><td colspan="7">Inter-model Cohen&#x27;s κ over emitted model labels</td></tr><tr><td>Q-N</td><td>Act</td><td>0.767</td><td>17,654</td><td>17,654</td><td>100.00</td><td>0.923</td></tr><tr><td>Q-N</td><td>P1</td><td>0.588</td><td>13,107</td><td>13,435</td><td>97.56</td><td>0.794</td></tr><tr><td>Q-N</td><td>P2</td><td>0.481</td><td>13,005</td><td>13,435</td><td>96.80</td><td>0.821</td></tr><tr><td>Q-N</td><td>Grounded</td><td>0.212</td><td>2,512</td><td>2,625</td><td>95.70</td><td>0.615</td></tr><tr><td>Q-G</td><td>Act</td><td>0.772</td><td>17,654</td><td>17,654</td><td>100.00</td><td>0.927</td></tr><tr><td>Q-G</td><td>P1</td><td>0.583</td><td>13,050</td><td>13,577</td><td>96.12</td><td>0.791</td></tr><tr><td>Q-G</td><td>P2</td><td>0.504</td><td>13,210</td><td>13,577</td><td>97.30</td><td>0.789</td></tr><tr><td>Q-G</td><td>Grounded</td><td>0.293</td><td>2,521</td><td>2,557</td><td>98.59</td><td>0.705</td></tr><tr><td>N-G</td><td>Act</td><td>0.753</td><td>17,654</td><td>17,654</td><td>100.00</td><td>0.916</td></tr><tr><td>N-G</td><td>P1</td><td>0.560</td><td>12,600</td><td>13,274</td><td>94.92</td><td>0.780</td></tr><tr><td>N-G</td><td>P2</td><td>0.407</td><td>12,888</td><td>13,274</td><td>97.09</td><td>0.762</td></tr><tr><td>N-G</td><td>Grounded</td><td>0.480</td><td>2,440</td><td>2,529</td><td>96.48</td><td>0.739</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 9: Main-set agreement with contributing and act-eligible denominators. Q, G, and N denote Qwen, Gemma, and Nemotron. Each attribute comparison conditions on its own applicable act and complete yes/no ratings.

S1, W1–W3, and Q1. The extra row (ASM; P1=yes) is excluded because its identifier is outside the input set. Input Q1 already contains two questions; this illustrates an extra identifier, not evidence of invented review content.

Five-human and model validation. Each of the five annotators labels all 470 dev comments (2,350 labels). The human-reference distribution is ASM 406, CAL 51, FLAG 13. All five agree on 352 items (74.9%), and at least four agree on 428 (91.1%). Of the 118 contested items, 102 involve only ASM/CAL disagreement. The human tie rule resolves one item; the individual annotations are retained without adjudication.

Table 13 compares each model’s run-aggregated labels and the two panels with the five-human reference. Table 14 reports the corresponding confusion matrices. Using the author anchor alone as reference yields Cohen’s $\kappa ~ = ~ . 6 2 0$ with the open-weight panel and .653 with the frontier panel. Across individual humans, the open-weight comparisons range from .505 to .620 (mean .569), and frontier comparisons from .518 to .658 (mean .593). Pooling these ten comparisons gives .505– .658 (mean .581), overlapping the ten human– human comparisons (.468–.649, mean .555), without establishing statistical equivalence.

For context, DISAPERE reports Cohen’s κ = .447–.605 across six sentence-annotation tasks (Kennard et al., 2022). Its labels and units differ from ours, so these values do not define a performance ceiling. Inconsistency in review scores (Cortes and Lawrence, 2021) concerns judgments of paper merit, distinct from agreement on the act expressed in a comment.

<table><tr><td>Pair A-B</td><td>Field</td><td>Matched n</td><td>Intra A</td><td>Intra B</td><td>Inter A-B</td></tr><tr><td>Q-N</td><td>Act</td><td>17,642</td><td>0.842</td><td>0.780</td><td>0.768</td></tr><tr><td>Q-N</td><td>P1</td><td>12,038</td><td>0.639</td><td>0.578</td><td>0.580</td></tr><tr><td>Q-N</td><td>P2</td><td>11,925</td><td>0.563</td><td>0.454</td><td>0.475</td></tr><tr><td>Q-N</td><td>Grounded</td><td>2,117</td><td>0.554</td><td>0.567</td><td>0.208</td></tr><tr><td>Q-G</td><td>Act</td><td>17,653</td><td>0.842</td><td>0.959</td><td>0.772</td></tr><tr><td>Q-G</td><td>P1</td><td>12,266</td><td>0.629</td><td>0.890</td><td>0.582</td></tr><tr><td>Q-G</td><td>P2</td><td>12,563</td><td>0.562</td><td>0.857</td><td>0.497</td></tr><tr><td>Q-G</td><td>Grounded</td><td>2,257</td><td>0.547</td><td>0.867</td><td>0.280</td></tr><tr><td>N-G</td><td>Act</td><td>17,641</td><td>0.780</td><td>0.959</td><td>0.753</td></tr><tr><td>N-G</td><td>P1</td><td>11,476</td><td>0.579</td><td>0.893</td><td>0.556</td></tr><tr><td>N-G</td><td>P2</td><td>12,016</td><td>0.455</td><td>0.860</td><td>0.407</td></tr><tr><td>N-G</td><td>Grounded</td><td>2,159</td><td>0.578</td><td>0.866</td><td>0.475</td></tr></table>

Table 10: Matched-item sensitivity: intra-model Fleiss’ κ and inter-model Cohen’s κ on identical items within each row. The artifact includes observed agreement for each coefficient.
<table><tr><td>Field</td><td>Applicable</td><td>Yes</td><td>No</td><td>Unresolved</td><td>3/3</td><td>2/3</td><td>2/2</td><td>Coverage (%)</td></tr><tr><td>P1</td><td>14,203</td><td>6,451</td><td>7,221</td><td>531</td><td>8,445</td><td>3,905</td><td>1,322</td><td>96.26</td></tr><tr><td>P2</td><td>14,203</td><td>10,223</td><td>3,405</td><td>575</td><td>8,646</td><td>3,894</td><td>1,088</td><td>95.95</td></tr><tr><td>Grounded</td><td>3,067</td><td>1,933</td><td>871</td><td>263</td><td>1,160</td><td>1,083</td><td>561</td><td>91.42</td></tr></table>

Table 11: Final attribute counts. Coverage is the proportion of applicable fields resolved as yes or no. Unresolved P1/P2/grounding fields comprise 385/395/183 binary ties and 146/180/80 cases with insufficient support, respectively.

Section-specific CAL disagreements. The fivehuman mode and open-weight panel label respectively 50/150 and 55/150 Strengths comments as CAL, 1/189 and 14/189 Weaknesses comments, and 0/131 and 6/131 Questions comments. Relative to the human reference, the 33 CAL false positives comprise 14 Strengths, 13 Weaknesses, and 6 Questions; all nine false negatives occur in Strengths. Fourteen of 47 reviewer–paper instances contain no human-mode CAL, versus seven under the panel, with five shared. The human reference also shows higher CAL prevalence in Strengths than Weaknesses, but only one human-mode CAL occurs in Weaknesses. This small development sample does not justify correcting the main-corpus prevalence or effect size.

Model disagreement as an inspection cue. Table 15 groups the same 470 comments by agreement among each panel’s three run-voted model labels. The non-unanimous subsets include two openweight and one frontier three-way ties, retained under the documented panel tie-break. This is a posthoc descriptive analysis of 12 development papers, with different class compositions across subsets and an imperfect human reference. For the openweight panel, the 69 non-unanimous items contain 30 of 47 human-reference mismatches; 17 remain among unanimous items. The association motivates displaying dissent for closer reading, without treating unanimity as a reason to bypass inspection. It does not measure effects on human decisions or review time.

Dev aggregation. All six models supply three act votes per comment except Gemma, whose runs 1 and 3 each omit two comments; these use the available run 2 label. Two Nemotron run ties follow the first-available-run rule. The open-weight and frontier panels have two and one three-way ties, respectively, resolved by their specified model orders. Reaggregation from archived runs reproduces all model and panel acts. Human validation covers Level 1 only.

<table><tr><td>Model / run</td><td>Extra</td><td>Missing</td><td>Dup.</td><td>P1 NA / applicable</td><td>P2 NA / applicable</td><td>Grounded NA / applicable</td><td>Both-no</td></tr><tr><td>Qwen / 1</td><td>0</td><td>0</td><td>60</td><td>34 / 14,289</td><td>182 / 14,289</td><td>0 /3,113</td><td>175</td></tr><tr><td>Qwen / 2</td><td>0</td><td>0</td><td>50</td><td>48 / 14,292</td><td>94 / 14,292</td><td>0/ 3,147</td><td>166</td></tr><tr><td>Qwen / 3</td><td>1</td><td>0</td><td>4</td><td>2 / 14,248</td><td>37 / 14,248</td><td>2/ 3,185</td><td>134</td></tr><tr><td>Gemma / 1</td><td>44</td><td>1</td><td>0</td><td>418 / 14,119</td><td>166 / 14,119</td><td>4/3,110</td><td>60</td></tr><tr><td>Gemma / 2</td><td>52</td><td>0</td><td>6</td><td>628 / 14,082</td><td>188 / 14,082</td><td>6 /3,107</td><td>96</td></tr><tr><td>Gemma / 3</td><td>49</td><td>0</td><td>8</td><td>408 / 14,109</td><td>166 / 14,109</td><td>1/3,079</td><td>103</td></tr><tr><td>Nemotron / 1</td><td>9</td><td>3</td><td>0</td><td>37 / 13,773</td><td>3 / 13,773</td><td>1/3,243</td><td>0</td></tr><tr><td>Nemotron / 2</td><td>8</td><td>0</td><td>18</td><td>70 /13,723</td><td>39 / 13,723</td><td>0 /3,288</td><td>0</td></tr><tr><td>Nemotron / 3</td><td>14</td><td>10</td><td>0</td><td>15 / 13,749</td><td>4 /13,749</td><td>0 /3,309</td><td>2</td></tr></table>

Table 12: Per-run main-set output audit. Applicable denominators use each retained raw row’s act, so they differ from final-consensus denominators.

<table><tr><td>Model / panel</td><td>κ</td><td>Agreement (%)</td><td>Macro F1</td><td>CAL count</td><td>CAL precision</td><td>CAL recall</td><td>CAL F1</td></tr><tr><td>Qwen</td><td>0.616</td><td>89.57</td><td>0.665</td><td>80</td><td>0.575</td><td>0.902</td><td>0.702</td></tr><tr><td>Gemma</td><td>0.630</td><td>89.57</td><td>0.830</td><td>76</td><td>0.526</td><td>0.784</td><td>0.630</td></tr><tr><td>Nemotron</td><td>0.602</td><td>88.94</td><td>0.751</td><td>74</td><td>0.527</td><td>0.765</td><td>0.624</td></tr><tr><td>GPT-5.5</td><td>0.660</td><td>90.64</td><td>0.752</td><td>79</td><td>0.582</td><td>0.902</td><td>0.708</td></tr><tr><td>Gemini 3.1 Pro</td><td>0.690</td><td>91.70</td><td>0.787</td><td>73</td><td>0.630</td><td>0.902</td><td>0.742</td></tr><tr><td>Opus 4.6</td><td>0.645</td><td>90.85</td><td>0.751</td><td>68</td><td>0.618</td><td>0.824</td><td>0.706</td></tr><tr><td>Open-weight panel</td><td>0.633</td><td>90.00</td><td>0.779</td><td>75</td><td>0.560</td><td>0.824</td><td>0.667</td></tr><tr><td>Frontier panel</td><td>0.686</td><td>91.49</td><td>0.783</td><td>75</td><td>0.613</td><td>0.902</td><td>0.730</td></tr></table>

Table 13: Dev comparisons against the five-human mode (n = 470 throughout; human-reference CAL count = 51). A panel does not necessarily outperform every constituent model.

<table><tr><td>Panel</td><td>Human reference</td><td>ASM</td><td>CAL</td><td>FLAG</td></tr><tr><td>Open-weight</td><td>ASM</td><td>373</td><td>32</td><td>1</td></tr><tr><td>Open-weight</td><td>CAL</td><td>9</td><td>42</td><td>0</td></tr><tr><td>Open-weight</td><td>FLAG</td><td>4</td><td>1</td><td>8</td></tr><tr><td>Frontier</td><td>ASM</td><td>377</td><td>28</td><td>1</td></tr><tr><td>Frontier</td><td>CAL</td><td>5</td><td>46</td><td>0</td></tr><tr><td>Frontier</td><td>FLAG</td><td>5</td><td>1</td><td>7</td></tr></table>

Table 14: Dev panel confusion matrices. Rows are the five-human mode; columns are panel predictions.

<table><tr><td>Model votes</td><td>n</td><td></td><td>Matches Agreement (%)</td></tr><tr><td colspan="4">Open-weight panel</td></tr><tr><td>Unanimous</td><td>401</td><td>384</td><td>95.8</td></tr><tr><td>Non-unanimous</td><td>69</td><td>39</td><td>56.5</td></tr><tr><td colspan="4">Frontier panel</td></tr><tr><td>Unanimous</td><td>430</td><td>407</td><td>94.7</td></tr><tr><td>Non-unanimous</td><td>40</td><td>23</td><td>57.5</td></tr></table>

Table 15: Dev panel agreement with the five-human mode, grouped by agreement among three per-model run-voted acts. Non-unanimous includes 2-of-3 and three-way ties; panel labels retain the specified tie-break. This groups by model votes, unlike the human-defined subsets in Table 4.

<table><tr><td>Primary area</td><td>Papers</td><td>CAL/all comments</td><td>CAL (%)</td></tr><tr><td>causal reasoning</td><td>12</td><td>58/467</td><td>12.4</td></tr><tr><td>learning on graphs and other geometries &amp; topologies</td><td>17</td><td>81/630</td><td>12.9</td></tr><tr><td>probabilistic methods (Bayesian methods, variational inference, sampling, UQ,</td><td>16</td><td>84/547</td><td>15.4</td></tr><tr><td>etc.) generative models</td><td>37</td><td>203/1,256</td><td>16.2</td></tr><tr><td>applications to robotics, autonomy, planning</td><td>17</td><td>99/601</td><td>16.5</td></tr><tr><td>transfer learning, meta learning, and lifelong learning</td><td>18</td><td>104/631</td><td>16.5</td></tr><tr><td>infrastructure, software libraries, hardware, systems, etc.</td><td>11</td><td>53/321</td><td>16.5</td></tr><tr><td>foundation or frontier models, including LLMs</td><td>54</td><td>304/1,833</td><td>16.6</td></tr><tr><td>applications to computer vision, audio, language, and other modalities</td><td>48</td><td>258/1,546</td><td>16.7</td></tr><tr><td>applications to physical sciences (physics, chemistry, biology, etc.)</td><td>23</td><td>141/835</td><td>16.9</td></tr><tr><td>alignment, fairness, safety, privacy, and societal considerations</td><td>36</td><td>217/1,270</td><td>17.1</td></tr><tr><td>other topics in machine learning (i.e., none of the above)</td><td>21</td><td>134/780</td><td>17.2</td></tr><tr><td>reinforcement learning</td><td>27</td><td>172/987</td><td>17.4</td></tr><tr><td>learning on time series and dynamical systems</td><td>16</td><td>114/630</td><td>18.1</td></tr><tr><td>optimization</td><td>21</td><td>148/817</td><td>18.1</td></tr><tr><td>applications to neuroscience &amp; cognitive science</td><td>14</td><td>94/513</td><td>18.3</td></tr><tr><td>datasets and benchmarks</td><td>33</td><td>230/1,202</td><td>19.1</td></tr><tr><td>neurosymbolic &amp; hybrid AI systems (physics-informed, logic &amp; formal rea- soning, etc.)</td><td>12</td><td>86/430</td><td>20.0</td></tr><tr><td>interpretability and explainable AI</td><td>21</td><td>157/783</td><td>20.1</td></tr><tr><td>unsupervised, self-supervised, semi-supervised, and supervised representation learning</td><td>29</td><td>223/1,082</td><td>20.6</td></tr><tr><td>learning theory</td><td>17</td><td>107/493</td><td>21.7</td></tr><tr><td>Total</td><td>500</td><td>3,067/17,654</td><td>17.4</td></tr></table>

Table 16: CAL shares across all 21 primary areas in the main corpus under the score/confidence-provided condition and two-stage aggregation. Comments are grouped by their paper’s recorded primary area; each denominator includes all original comments in that area, including FLAG. Rows are ordered by CAL share. These finite-sample descriptive rates characterize variation within ICLR 2026; they are not uncertainty bounds or evidence of cross-venue portability.

<table><tr><td>Audit count</td><td>Qwen</td><td>Gemma</td><td>Nemotron</td></tr><tr><td>Raw outputs: counts summed over three runs</td><td></td><td></td><td></td></tr><tr><td>Extra unique IDs (outside expected set)</td><td>1</td><td>145</td><td>31</td></tr><tr><td>Missing expected IDs</td><td>0</td><td>1</td><td>13</td></tr><tr><td>Duplicate rows</td><td>114</td><td>14</td><td>18</td></tr><tr><td>NA in applicable P1 / P2 / grounded fields</td><td>84 / 313 / 2</td><td>1,454 / 520 / 11</td><td>122 / 46 / 1</td></tr><tr><td colspan="4">Stage-1 aggregation: unresolved attribute fields in emitted model labels</td></tr><tr><td>P1 / P2 / grounded</td><td>130 / 274 / 53</td><td>460 / 185 / 16</td><td>293 / 295 / 152</td></tr></table>

Table 17: Raw-output quality and aggregation coverage have different units. Raw counts are summed across 3 × 17,654 expected comment–run opportunities per model (an extra ID can recur across runs); raw attribute-field counts follow last-row duplicate resolution. Stage-1 unresolved counts refer to one emitted label per expected comment and are deliberate abstentions, not raw-generation errors. Appendix D gives per-run and coverage denominators.

## E ReviewMap as Reviewer Guideline

This appendix adapts the released Practical Reviewer Guideline (§6.1), which originated in peer-review teaching and informed the annotation scheme. It organizes human reviewing around assessment and calibration: both involve judgment, directed at different evaluative targets. Appendix A presents the annotation instrument used in the experiments.

## Practical Reviewer Guideline

## E.1 Before you write: reading the paper

Anchor the review in the paper’s stated aims, claims, and evidence. Before writing, identify the following:

• What does the paper claim? Read abstract, introduction, and conclusion carefully. List the specific claims the authors make.

• What evidence does the paper provide? For each claim, identify the corresponding evidence: experiments, proofs, ablations, visualizations, or theoretical arguments.

• What is the closest prior work? Identify the most similar existing work. Understanding how the paper positions itself relative to this work is often the key to evaluating its contribution.

Use notes on these three points as the basis for the review.

## E.2 The review process: a step-by-step workflow

The workflow below recommends independent reading and judgment. Use AI assistance only where permitted by the venue’s current policy, including its confidentiality requirements.

## Phase 1 — Preparation (AI assistance where permitted).

1. Read the abstract and decide whether to accept the review. Decline if outside your area of competence.

2. Build your reference landscape. List cited related works; where venue policy permits, use AI to search for additional recent works.

3. Familiarize yourself with key references, optionally using AI assistance where permitted.

4. Prepare comparison points: what each related work does differently, where claims overlap, how the proposed method should be positioned.

## Phase 2 — Reading the paper independently.

5. Read the paper yourself, with your prepared comparison points. Independent reading helps reviewers assess support for the paper’s claims and scholarly standards, then judge the contribution’s significance.

## Phase 3 — Post-reading verification.

6. Check for additional missing references. Your understanding after reading the paper allows precise queries.

## Phase 4 — Writing and finalizing.

7. Write the review in your own words.

8. Where the venue permits it, optionally use AI to check grammar, clarity, and tone. Retain responsibility for the review’s judgments and recommendation.

9. Submit.

## E.3 Structure of a review

A well-structured review serves authors, the area chair, and other reviewers:

• Summary (1 paragraph). What the paper does in your own words: problem, approach, key results.

• Strengths (2–5 points). Specific strengths tied to concrete aspects. General: “The experiments are thorough.” Specific: “The ablation in Table 3 isolates the contribution of each component across three datasets.”

• Weaknesses, separated as major (affecting validity of core claims) vs. minor (presentation, secondary). Be specific about why each issue matters.

• Questions to authors. Ask specific questions that the authors can address.

• Suggestions for improvement (optional but recommended).

• Minor issues (typos, formatting; keep brief).

## E.4 Two core principles for evaluating a paper

The framework organizes evaluation into two stages:

1. Assessment (this section): assess what the paper did and how it did it, using Principles 1 and 2.

2. Calibration (§E.5): build on these assessments to evaluate novelty and significance and formulate an overall recommendation.

Identifying the claims and assessing their support provides a basis for evaluating novelty and significance. P1/P2 assessments address a paper’s claims and shared scholarly standards; calibration addresses the contribution’s value and venue fit. Reviewers can examine the evidence and criteria behind either judgment while disagreeing about their adequacy or significance.

## E.4.1 Principle 1: Claim–Evidence Alignment

“Does this paper sufficiently support what it claims?”

Evaluate the evidence in relation to the authors’ specific claims. Steps: (1) identify each claim (empirical, scope, design, or compliance assertions); (2) find the corresponding evidence; (3) assess the gap. Common patterns range from well-supported (acknowledge as strength), through overclaimed (point out the specific mismatch) and incomplete evidence (request the specific missing experiment), to no evidence (flag as major concern) and evidence contradicting the claim (point out precisely with references). Match the severity of the concern to the evidential gap.

## E.4.2 Principle 2: Scholarly Standard Assessment

“Does this paper meet the standards that apply regardless ofwhat it claims?”

Scholarly standards apply beyond a paper’s explicit claims. Key areas: reproducibility; ethical considerations; fair treatment of prior work (citation accuracy, comparison fairness); accurate acknowledgment of limitations; writing quality. A paper may support its claims while falling short of these standards.

## E.4.3 How the two principles work together

In practice, most comments fall under one principle or the other; some fall under both. Examples: “Claims state-of-the-art performance but does not compare with Method X” → Principle 1 (claim without evidence) and Principle 2 (unfair treatment of prior work); “No code or hyperparameters provided” → Principle 2 (scholarly standard); “Claims robustness across diverse domains but evaluates only on one narrow dataset” → Principle 1. The principles guide evaluation; explicit labels for individual comments are optional.

## E.5 Calibrating your judgment

After completing P1/P2 assessment, evaluate the contribution’s novelty and significance. Ground each judgment by citing the P1 or P2 assessment that supports it.

Novelty. “Is this contribution new relative to existing knowledge?” Specific: “This is, to my knowledge, the first work to combine X with Y.” Supported by a citation: “Method Z [Ref 12] already addresses the same problem with a similar approach, which limits the novelty of this submission.” Unsupported: “The novelty is limited.”

Significance. “Does this contribution meaningfully advance the field or community?” Novelty and significance can differ: a first solution to a problem may have limited significance, whereas a large-scale validation of a known technique may be significant. Significance includes venue-relative impact.

Confidence. State a confidence level that reflects your expertise and the evidential basis of the assessment.

Severity of language. Match the strength of your language to the strength of your evidence. A mathematical proof of an error supports direct statements; a suspicion supports a question or request for clarification; a personal preference should be identified as such.

Overall recommendation. The recommendation shouldfollowfrom P1/P2 assessments and novelty/significance judgments. If it appears inconsistent with the stated assessments (no major weaknesses but a rejection recommendation; identified fatal flaws but an acceptance recommendation), explain the reasoning or revisit the assessment.

## E.6 Common pitfalls in reviewing

• Misidentifying the contribution type. Evaluate a task-definition paper according to its aims and claims, rather than applying the expectations for a methods paper.

• Equating simplicity with limited novelty. Consider the potential long-term contribution of novel or interdisciplinary work. Before describing an approach as “too simple,” check whether the specific combination has been tried before.

• Making general criticisms without supporting evidence. “This work is not new” requires a reference to prior work. “This work is not enough” requires an explanation of the additional contribution needed and why. “There is no [X] experiment” requires a link to a specific claim or a relevant scholarly standard.

• Omitting strengths from a critical review. Identify the paper’s strengths alongside its weaknesses, regardless of the overall recommendation.

• Treating unfamiliarity as evidence of error. Consult relevant sources or state that the aspect is outside your expertise.

• Requesting experiments without a clear rationale. Replace a general request such as “more experiments” with a rationale: “an ablation comparing X vs. Y would clarify whether the improvement comes from component A or B.”

• Using dismissive language. State the substantive concern directly and professionally: “This is trivial” → “The technical contribution is primarily in system design”; “The authors failed to” → “The paper does not currently include”.

## E.7 Pre-submission checklist

• Comprehension: does my summary accurately reflect what the paper does?

• Balance: have I identified both strengths and weaknesses?

• Specificity: is every weakness tied to a specific section, table, equation, or claim?

• Proportionality: does the stated severity of each concern reflect its implications for the paper?

• Actionability: would the authors know how to improve their paper from my review?

• Grounded value judgments: when I state a novelty or significance judgment, have I cited the P1 or P2 assessment that supports it?

• Assessment before recommendation: did I complete P1/P2 before forming my overall recommendation?

• Tone: would I be comfortable receiving this review for my own paper?

• Consistency: does my rating match the severity of the issues I raised?

• Confidence: does my confidence score accurately reflect my expertise on this topic?

## F Author-Side Strategy Scaffold

This appendix expands §6.2 with strategy definitions, the coding-to-menu mapping, venuetemplate adaptation, and a comparison with RbtAct’s independently derived rebuttal-action taxonomy. The proposed scaffold supports authors’ response planning; large-scale validation with authorside rebuttal corpora remains future work.

## F.1 Five response strategies

We define five candidate strategies for authors planning a response and compare their scope with RbtAct’s rebuttal-action taxonomy below:

• LEVERAGE — cites the comment as thirdparty support in the rebuttal opening.

• DEFEND — contests a critique using citations, data, or precedent.

• CONCEDE — accepts a substantive critique and offers a concrete remedy.

• ACCEPT — commits to a surface-level revision, such as correcting a typo or formatting, without an epistemic concession.

• CLARIFY — answers directly with paper content or asks the reviewer for clarification.

ACCEPT and CONCEDE both involve agreement but differ in depth: ACCEPT addresses presentation without revising the model or claims, while CON-CEDE entails a substantive update to the paper or its scope.

## F.2 Mapping codings to strategy menus

The (act, attribute) coding provides up to five candidate strategies per comment (Table 18); the author chooses the response. For ASM with both P1 and P2 marked yes, the menu is the union of the two rows. Unresolved attributes leave the corresponding menu branch inactive; ASM cases with both attributes marked no require direct reading. Attribute-derived options remain subject to human confirmation.

In the proposed interface, FLAG and three-way act disagreements prompt direct reading, and unresolved attributes retain their status. Act unanimity (88.4%) and 2-of-3 agreement (11.3%) appear as distinct vote states alongside the source comment. Authors confirm attribute-dependent options and can revise the coding or choose a response outside the menu.

<table><tr><td>Coding</td><td>Strategy menu</td></tr><tr><td>ASM, P1=yes</td><td>LEVERAGE · DEFEND · CON- CEDE·CLARIFY</td></tr><tr><td>ASM, P2=yes</td><td>LEVERAGE · DEFEND · CON- CEDE·CLARIFY·ACCEPT</td></tr><tr><td>CAL</td><td>LEVERAGE · DEFEND · CON- CEDE</td></tr><tr><td>FLAG</td><td>(no scaffold — author reads di- rectly)</td></tr></table>

Table 18: Proposed default mapping from codings to response strategies. The union of applicable rows yields at most five options. Unresolved attributes leave the corresponding row inactive; the author confirms attributedependent options and chooses the response.

How much of the mapping depends on attributes? Under the guideline’s valid ASM combinations (P1 or P2=yes), four strategy memberships are invariant; only ACCEPT depends on P2. CAL and FLAG each fix all five inclusion/exclusion decisions. Thus, for the observed act distribution, $( 5 \times 1 7 , 6 5 4 - 1 4 , 2 0 3 ) / ( 5 \times 1 7 , 6 5 4 )$ = 74,067/88,270 comment–strategy pairs (83.9%) are structurally act-determined under that constraint. This counts inclusion/exclusion decisions for comment–strategy pairs. At the comment level, the act alone fixes the full menu for 3,451/17,654 comments (19.5%, CAL or FLAG), including the empty FLAG menu.

The aggregated attributes do not always satisfy that constraint: 327/14,203 ASM comments (2.3%) have neither P1 nor P2 marked yes (287 involve unresolved fields; 40 have P1=P2=no). No ASM branch activates for these comments pending direct reading. Another 305 have P1=yes and P2=unresolved: four options are available, while ACCEPT remains pending. These counts describe the proposed mapping under the two-stage aggregation; attribute-derived options require human confirmation.

Relationship to DISAPERE rebuttal acts. DIS-APERE’s existing labels (Kennard et al., 2022) can describe realized responses to these choices: Concede criticism corresponds to CONCEDE, Answer to CLARIFY, and dispute labels such as Reject criticism or Mitigate criticism to DEFEND. Its completed/promised-task labels may realize either ACCEPT or CONCEDE, depending on whether the change is surface-level or substantive; Accept praise overlaps with, but does not entail, reusing praise as LEVERAGE. The menu thus groups familiar response acts as choices available before

drafting.

## F.3 Venue-template adaptation

Venue-specific section templates can help narrow the candidate menu: ICLR’s Strengths/Weaknesses/Questions template provides section cues (Strengths → LEVERAGE; Weaknesses → DE-FEND/CONCEDE/ACCEPT; Questions → CLAR-IFY); ARR’s heterogeneous Comments/Suggestions/Typos sections require the coding for routing; CVPR-style rebuttals without sections rely entirely on the coding. The coding-to-menu mapping remains the venue-invariant core. Authors write the rebuttal, while the scaffold offers a paper-level overview of response options (e.g., “7 LEVERAGE candidates, 11 DEFEND/CONCEDE pending, 18 CLARIFY pending”).

## F.4 Comparison with RbtAct’s response taxonomy

RbtAct (Wu et al., 2026) classifies observed rebuttal actions, while ReviewMap proposes candidate strategies before a response is written. The taxonomies partly overlap: DEFEND resembles defense without change, and revision-oriented responses can occur under CONCEDE or ACCEPT. RbtAct distinguishes completed revisions, specific revision plans, and vague commitments to revise; this concerns the status and specificity of a revision, rather than ReviewMap’s distinction between substantive concession and presentation changes. Our menu also includes using positive comments as support (LEVERAGE). These correspondences provide a conceptual comparison of the two taxonomies.

## G Area-Chair Dashboard Example

Figure 5 illustrates the proposed Area-Chair Dashboard using one paper from the main corpus. The reviewer profile summarizes consensus codings, while manually annotated response links, strategies, and resolution notes illustrate how an AC could inspect the ensuing discussion.

From an evaluative target to a response. Consider LnWH/W1, “Trivial target properties.” The act vote is CAL 2, ASM 1. The linked response identifies the main targets as adsorption capacity and separation selectivity and points to Table 1 of the paper. We interpret this response as CLAR-IFY, outside the default CAL menu: it addresses the premise of the value judgment by clarifying what is evaluated. Although the CAL menu in Appendix F lists LEVERAGE/DEFEND/CONCEDE, the menu remains advisory. A reviewer can inspect whether the criticism names the intended target and grounds; an author can clarify that target before defending its value; an AC can assess which part of the concern the response addresses. This retrospective, manually linked example illustrates how the proposed views organize these questions for human judgment.

<table><tr><td colspan="9">ReviewMap — discussion-phase panel view</td></tr><tr><td colspan="7">Paper: v2oYZJ7Exo Reviewer profile: counts from consensus codings</td><td colspan="2">Recorded scores: 2 / 4 / 6 / 8</td></tr><tr><td colspan="9">Reviewer Score ASM</td></tr><tr><td rowspan="7">8 6</td><td rowspan="7"></td><td rowspan="7">CAL</td><td rowspan="7">FLAG</td><td colspan="3"></td><td colspan="3">CAL grounding</td></tr><tr><td></td><td></td><td>yes</td><td>no</td><td>unresolved</td></tr><tr><td>G88g</td><td>3</td><td>0</td><td>0</td><td>2</td><td>1</td></tr><tr><td>4Vi5</td><td>8 4</td><td>1</td><td>0</td><td>0</td><td>1</td><td></td></tr><tr><td>DEPu</td><td>12</td><td>0</td><td>0</td><td></td><td>0 0</td><td>0 0</td></tr><tr><td>4 LnWH 2</td><td>8</td><td>4</td><td>0</td><td></td><td>1 2</td><td>1</td></tr><tr><td colspan="7">Inspect the unresolved cases. LnWH/W7 and G88g/S2 each have one yes and one no grounding vote among two eligible CAL voters. Both retain an unresolved grounding label. LnWH/W1 has two yes votes among two eligible voters; its act remains contested (CAL 2, ASM 1).</td></tr><tr><td colspan="10">Manually linked concern-response examples Comment / Review excerpt Linked author response (paraphrase) grounding</td></tr><tr><td>LnWH/W1: yes (2/2 eligible)</td><td colspan="2">“Trivial target properties&quot;</td><td colspan="4">R1: the main targets are adsorption capac- ity and separation selectivity, with non-</td><td></td><td>tion note CLARIFY (outside default</td><td>menu);</td></tr><tr><td>LnWH/W7: un- 1 no)</td><td colspan="2">“Given the weak empirical resolved (1 yes, gains, it&#x27;s unclear whether</td><td colspan="4">trivial properties in Table 1. R5: the authors report monotonic improve- ments, over 30% MAE reduction, and</td><td></td><td colspan="2">scope clarified DEFEND; partial</td></tr><tr><td></td><td colspan="2">warranted&quot; Resolution notes reflect our interpretation of how each response addresses the concern and do not establish reviewer</td><td colspan="3">this added complexity is about 30% additional computational com- plexity.</td><td colspan="2"></td><td colspan="2"></td></tr><tr><td colspan="8">acceptance. Excerpts are abridged: W7 also includes a benchmark-choice concern. Full comments and responses remain available for inspection, including concerns with unresolved grounding. Discussion context: the authors publicly disavowed an unrelated third-party comment. This manually annotated discussion event appears alongside the official-review act labels.</td></tr></table>

Figure 5: AC dashboard mockup for paper v2oYZJ7Exo (§6.3). Scores record the post-discussion snapshot in the corpus; comment counts, act labels, grounding states, and vote counts summarize the consensus codings. Concern selection, response matching, strategy badges, resolution notes, and discussion context are manually annotated for this case study. Source evidence and uncertain labels remain visible to support the AC’s judgment.

## H Sensitivity to Score and Confidence Headers

This appendix reports a supplementary inputcondition comparison. The main analysis, preceding empirical tables and figures, and human and frontier-model development validation use the score/confidence-provided condition. The main compute estimate in Appendix C excludes these additional runs.

Comparison protocol. We compare the main condition with separate runs in which reviewer score and confidence headers were jointly removed. The 500 papers, 17,654 comment identifiers, guideline, and remaining prompt content are unchanged. Both conditions use three runs per model and the same two-stage aggregation rules. The removal condition uses a 16,000-token generation limit for each model; the main condition includes varying budgets and targeted retries (Appendix C). Generation settings and recovery histories were therefore not fully matched.

Act labels and corpus patterns. Final acts agree on 17,211/17,654 comments (97.49%); 443 change. Of these, 164 move from ASM to CAL and 202 from CAL to ASM; the remaining 77 involve FLAG. Overall agreement is weighted by the large ASM share. CAL remains sparse and concentrated in Strengths (Table 19). Both conditions have 333 zero-CAL reviewer–paper instances, but only 300 are shared: 33 leave and 33 enter this group. The zero-CAL sets therefore overlap on 300/333 instances.

<table><tr><td>Measure</td><td>Main</td><td>Removed</td></tr><tr><td>CAL comments (%)</td><td>17.37</td><td>17.27</td></tr><tr><td>Zero-CAL reviews (%)</td><td>17.03</td><td>17.03</td></tr><tr><td>Strengths/Weaknesses CAL ratio</td><td>2.819</td><td>2.726</td></tr><tr><td>Mean inter-model κ: Act</td><td>.764</td><td>.757</td></tr><tr><td>P1</td><td>.577</td><td>.578</td></tr><tr><td>P2</td><td>.464</td><td>.450</td></tr><tr><td>Grounded</td><td>.328</td><td>.351</td></tr></table>

Table 19: Main condition versus score/confidence headers removed. Denominators are 17,654 comments and 1,955 review instances. Agreement values are means of three pairwise Cohen’s κ coefficients.

Agreement across annotation layers. The ordering of mean inter-model agreement persists. Attribute coefficients require matching applicable acts and available binary labels; samples differ by field and condition. Restricting each pair–field comparison to items eligible in both conditions also preserves this ordering. Removal-condition act coefficients range from .745 to .773.

Final attribute-state agreement is 87.38% for P1 and 87.76% for P2 on the 14,007 comments retaining ASM in both conditions, and 86.74% for grounding on the 2,865 retaining CAL. These states include yes, no, and unresolved. Each rate uses its stated classes and conditional sample.

Within-condition variation. For a comparison at equal aggregation depth, we form three panels per condition, each using one run from each model. Mean act disagreement is 3.20% across the three within-main panel pairs, 3.64% across the three within-removal pairs, and 3.46% across the nine between-condition pairs. These three-rating panels differ from the nine-rating, two-stage final consensus above. The between-condition differences are descriptively similar in scale to within-condition variation. The panel pairs share runs and items, so these values provide a descriptive baseline.

Interpretation and limits. The reported corpus patterns persist across the evaluated conditions, while individual labels and attribute states change. The comparison measures joint header-removal sensitivity across differing generation settings; the removal condition has no separate human validation. Review text can itself convey evaluative stance even without numeric headers. The supplementary artifact provides saved outputs, aggregation and comparison code, and a missing/duplicaterow sensitivity check; excluding all 260 comments with any missing, duplicate, or invalid act among the 18 raw ratings gives 97.52% final-act agreement on 17,394 comments.