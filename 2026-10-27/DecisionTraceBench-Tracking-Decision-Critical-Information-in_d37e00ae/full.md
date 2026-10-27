# DecisionTraceBench: Tracking Decision-Critical Information in Multi-Party Spoken Conversations

Wenchao Du<sup>1,2</sup> Fangxin Shang<sup>1</sup> Long Wu<sup>1</sup> Runze Cui<sup>1</sup> Qing Yang<sup>1</sup> Yehui Yang<sup>1,</sup>∗

<sup>1</sup>AI Lab, Qifu Technology, Beijing, China

<sup>2</sup>University of Chinese Academy of Sciences

Corresponding author: yangyehuisw@126.com

## Abstract

In high-stakes domains such as finance, critical information evolves through multi-party interaction. Recovering these changes and their sources is fundamental to reliable speech understanding. Existing speech benchmarks focus on transcription, local understanding, and turn-level state prediction, leaving joint decision evolution across speakers insufficiently evaluated. We introduce Decision-TraceBench, a speech state-tracking benchmark for Decision-Critical Information State Tracking (DCIST). It jointly evaluates decision event sequences, speaker attribution, scope, and temporal evidence, using trajectorylevel F1 to measure complete event-chain recovery. Our dataset comprises Chinese financial dialogues involving two to four speakers, with 100.39 hours of synthetic audio and 8.61 hours of human-performed recordings. On a fixed cohort of human-performed recordings, five cascaded systems and two direct-audio models achieve DCIST scores of 5.73% to 21.35%. Ablations with speaker-labeled reference scripts and no temporal localization requirements yield a maximum score of only 38.72%. Audio perturbation experiments further show that overall transcription error rates do not adequately reflect complete trajectory recovery. These findings reveal limitations in jointly understanding information provenance and state evolution, providing a benchmark for evaluating decision-process understanding and decision traceability in high-stakes speech applications.

## 1 Introduction

In financial decision-making, a conversational record must establish not only what was said, but how a proposal was revised, suspended, resumed, and resolved, and who was responsible for each effective change. Identical terminal values can arise through different processes. The same proposition can also have different force when expressed by a participant making a request, an analyst verifying evidence, or an authorized decision-maker. Recovering this history is essential to a traceable record.

Existing benchmarks evaluate broad audio understanding, speaker-attributed reasoning, and turn-level dialogue state tracking (Wang et al., 2025; Si et al., 2023; Kwon et al., 2025; Lee et al., 2026a). Text-based resources also model successive state changes (Dalvi et al., 2018; Tandon et al., 2020). Yet local correctness does not establish trajectory correctness: a model may recover the terminal amount while omitting a suspension and resumption, or recognize every amount while attributing another participant’s suggestion to an authorized decision-maker. Full-trajectory recovery requires jointly identifying effective events, their relations, their sources, and their evidence.

We introduce DecisionTraceBench, a benchmark for Decision-Critical Information State Tracking (DCIST), which makes the complete decision-critical event chain the prediction target (Figure 1). Completeness is defined by the minimum events and relations required by the annotation protocol, rather than by every possible segmentation of speech. Evidence-supported refinements are allowed, but cannot compensate for missing required events. We instantiate the framework in financial dialogues using controlled synthetic speech, human-recorded scripted conversations, and paired variants that change speaker assignments or confirmation relations.

Our experiments address three questions: Can current systems recover full trajectories? Does this remain difficult when reference text and speaker labels are available? Do aggregate transcription errors reflect changes in trajectory quality? Our contributions are:

• A full-trajectory speech state tracking framework with a structured representation, minimum required chains, and a joint trajectorylevel scoring protocol.

![](images/4c113cc940d005fb548afe5badb31f7eb42a1346767861f85977c24bef74cba9.jpg)  
Figure 1: Full-trajectory speech state tracking. Systems recover an event chain from audio without manual transcripts or gold speaker labels. Evaluation jointly checks required-event coverage, attributes and relations, and temporal evidence. Start, Pause, Progress, and Finalize correspond to start, hold, advance, and settle.

• A Chinese financial dialogue dataset spanning two to four speakers, with synthetic and human-recorded speech, event provenance and temporal evidence, and paired variants.

• A systematic evaluation of cascaded and direct-audio systems, reference-script controls, and targeted acoustic perturbations, exposing process-recovery failures not captured by transcription quality alone.

## 2 Related Work

Audio understanding and financial speech. AudioBench (Wang et al., 2025) provides multitask audio evaluation; MMAU (Sakshi et al., 2025) covers auditory perception and reasoning; SpeakerSleuth (Lee et al., 2026b) tests speaker consistency across dialogue turns. MultiFinBen (Peng et al., 2026a) spans multilingual, multimodal financial applications, while FinAudio (Cao et al., 2025) targets financial transcription and summarization. These resources support content understanding. DecisionTraceBench further evaluates when information changes effectively, who causes the change, and within which scope it holds.

Attribution, events, and dialogue states. M3- SLU (Kwon et al., 2025) and HEAR (Lee et al., 2026a) evaluate speaker-attributed reasoning. SpeechEE (Wang et al., 2024) extracts spoken events and arguments. SpokenWOZ (Si et al.,

2023) evaluates turn-level slot-state maintenance, with related work studying synthetic speech and audio-based verification (Lee et al., 2023; Lee and Pfister, 2026). DecisionTraceBench connects event actions and relations to speaker identity, scope, and time, evaluating their joint correctness over an entire required chain.

State evolution and memory. MultiWOZ (Budzianowski et al., 2018) annotates dialogue belief states; ProPara (Dalvi et al., 2018) tracks entity states in procedural text; OpenPI (Tandon et al., 2020) represents stepwise entity–attribute changes. The MEET consensus framework (Esfandiari-Baiat and Edlund, 2026) preserves incremental propositions and decision histories with source and temporal metadata. We operationalize full decision-record recovery as a raw-audio prediction benchmark with structured gold trajectories and a joint metric. VoxMem (Xiao et al., 2026) evaluates cross-session audio memory through information retrieval and temporal reasoning questions. Its retrieval target is question-conditioned; our target is the complete event chain itself. Appendix B contrasts these output requirements.

## 3 Full-Trajectory State Tracking

## 3.1 Representation and Task

Given an unsegmented multi-speaker recording and a task-wide public schema, a system predicts a set of trajectories. A trajectory is uniquely identified by an entity–field pair and contains temporally ordered decision cycles. Each cycle contains effective events recording an action, value, speaker, scope, predecessor, and evidence interval. Scope specifies the statement’s authority and business effect, distinguishing, for example, application presentation, material verification, technical review, and formal approval. The schema provides 193 entity–field combinations, 64 scope labels, and permitted value structures. It does not reveal the fields, event counts, roles, or answers present in an individual sample.

An event requires an observable substantive change. Ordinary questions and unchanged repetitions do not automatically create nodes. Turns and events are not one-to-one: a turn can affect multiple fields, while the interpretation of one event can depend on preceding turns. Values record the information available at that point; a numeric candidate is not necessarily an approved result.

## 3.2 Events, Cycles, and Evidence

<table><tr><td>Action</td><td>Trigger</td><td>State after</td></tr><tr><td>start</td><td>First valid candidate or explicit active initiation</td><td></td></tr><tr><td>advance</td><td>Substantive progress or resump- active tion</td><td></td></tr><tr><td>hold</td><td>Actual suspension due to an ex- held plicit obstacle</td><td></td></tr><tr><td>settle</td><td>Explicit, source-supported settled scoped conclusion</td><td></td></tr></table>

Table 1: Event actions and resulting states.

Table 1 defines four actions. The first observed event may be hold or settle; no unobserved start is invented. A successive hold is added only when the target value or a tracked critical condition materially changes. A revision that leaves processing suspended is hold, not advance. Settlement closes a cycle. An explicit reopening, revocation, or replacement decision starts another, which may consist of a single supported settle event.

Predecessor edges connect adjacent events within a cycle. Event and cycle identifiers may be renamed consistently. Speakers are numbered by first occurrence and evaluated without an optimal speaker permutation. Evidence intervals locate the principal triggering utterance; preceding context may justify its interpretation, but later conclusions cannot retroactively establish an earlier state.

## 3.3 Matching and DCIST

Required-chain matching. A prediction must cover every required event through an orderpreserving, one-to-one alignment on the same tracked field. Matching checks cycle boundaries and order, action, value, speaker, scope, settlement result, predecessor relations, and time. One predicted node cannot satisfy two required nodes. The terminal state is derived from the last event rather than independently scored. Top-level explanatory text is not scored.

Expression equivalence. Free-text values admit many semantically equivalent expressions that cannot be exhaustively enumerated in the gold annotations. Strict surface matching may therefore reject a correct value because of a minor wording difference; under full-trajectory evaluation, this single mismatch can invalidate an otherwise correct trajectory. This motivates evaluating answer meaning rather than lexical overlap alone (Bulian et al., 2022). We therefore allow contextdependent semantic correspondences while preserving the model’s freedom to express values. Astra first assesses whether field referents, scope meanings, and non-numeric value expressions convey the same information in the prediction, gold record, and relevant context. To reduce reliance on a single model’s judgment, we introduce crossmodel review (Verga et al., 2024): Kimi-K3 examines the initial decision and rationale and challenges specific contradictions with the evidence or task rules. Candidates on which Astra and Kimi-K3 disagree are reviewed by a human, who makes the final equivalent or non-equivalent judgment. Approved mappings are sample-local and cannot supply missing facts, merge distinct propositions, or relax quantitative, action, speaker, authority, temporal, or structural constraints. Deterministic code performs trajectory matching and score aggregation using the frozen judgments.

Time and additional content. A predicted interval P must have positive duration and satisfy $| P \cap G | / | P | \geq 0 . 8 0$ , where G is the coarse gold interval. Independently reviewed, supported extra nodes are exempt from penalties. Reviewed extras whose evidence or rule boundary remains unresolved are also exempt; unreviewed candidates are not. Unsupported insertions, mislabeled required events, structural violations, and omissions remain errors. Exempt nodes are removed only in the scoring view, with predecessor edges contracted; this cannot repair an invalid edge or remove an extra cycle. Figure 2 illustrates the distinction. Appendix E gives the full adjudication procedure and examples.

Same device-verification trajectory: identical final value does not guarantee a correct event chain  
![](images/3063114f875b49a7eaac8a4d77cb6404822c52a9687eaedd5767a30dfdf92f48.jpg)  
In this example, all other attributes, time evidence, and predecessor relations are correct.  
Figure 2: Minimum required chains and additional nodes. S1 handles documentation and S2 performs technical verification. Omitting a required suspension or resumption fails the trajectory despite the correct terminal value. Supported refinements are exempt; an invalid insertion fails the trajectory. All other attributes, evidence intervals, and predecessor relations are assumed correct.

For recording i, let $G _ { i }$ and $P _ { i }$ be the gold and predicted trajectory counts, $L _ { i }$ the exempt wholly new trajectories, and $T P _ { i }$ the maximum one-toone trajectory matching count:

$$
F _ { 1 , i } = \frac { 2 T P _ { i } } { G _ { i } + P _ { i } - L _ { i } } ,\tag{1}
$$

$$
\mathrm { D C I S T } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } F _ { 1 , i } .\tag{2}
$$

Legitimate new trajectories are excluded from the predicted denominator but never add true positives. Exempt content cannot offset a requiredevent error. We separately report the proportion of unique exempt nodes among all parseable predicted nodes. Complete but malformed answers count as output errors; cohort and completion accounting are specified in Appendix C.4. Reference-script experiments disable time checks but retain other structural requirements.

## 4 Dataset

We first design gold decision trajectories, then generate colloquial dialogue scripts conditioned on these trajectories, verify script–trajectory consistency, and finally produce audio with aligned event evidence. Synthetic speech enables controlled variation; human-recorded scripted conversations test the same task with natural human voices. Table 2 summarizes the corpus. Synthetic dialogues cover 14 financial scenarios and human recordings cover 10, including credit due diligence, supplychain finance, drawdown checks, audit adjustments, and insurance claims.

<table><tr><td>Subset</td><td>Audio</td><td>Hours</td><td>Tracks / events</td></tr><tr><td>Synthetic</td><td>2,246</td><td>100.39</td><td>5,319 / 18,227</td></tr><tr><td>Human</td><td>26</td><td>8.61</td><td>308 / 562</td></tr><tr><td>Total</td><td>2,272</td><td>109.01</td><td>5,627 / 18,789</td></tr></table>

Table 2: Dataset size. Hours are playback duration; totals use unrounded values. A/B variants count as separate recordings.

Synthetic recordings last 95.47–272.39 seconds; human recordings last 8.26–54.40 minutes. Human collection involved 22 participants. The estimated transcript-character rate, including pauses in the duration denominator, is 229.53 characters/minute (recording median: 230.77).

The synthetic subset contains 790 A/B pairs and 666 standalone scripts; the human subset contains 10 pairs and six standalone scripts. Paired variants change speakers, role assignments, or confirmation relations, with coherence checks on references and subsequent responses. They form 1,456 and 16 dialogue families, respectively; shared scripts and A/B variants remain in the same family. All recordings form fixed research evaluation cohorts, without a training/development partition. Across 58,396 synthetic–human script pairs, no normalized full-text duplicate was found; maximum character-five-gram Jaccard overlap is 0.50%. Appendix C documents cohort membership, lexical audits, generation, consent, and alignment. Both subsets are scored with monetary equivalence, expression mappings, and minimumchain matching.

## 5 Experiments

## 5.1 Setup

Systems. For human recordings, we evaluate FunASR (Gao et al., 2023) with Astra, Kimi-K3, or Qwen3.8-Max, and VibeVoice-ASR (Peng et al., 2026b) with Kimi-K3 or Qwen3.8-Max. On synthetic speech, each frontend is paired with DeepSeek V4 Flash or Qwen3.8-Max. VibeVoice produces text, timestamps, and speaker labels. Neither receives manual transcripts, role answers, or sample-specific gold records. API text models use temperature 0, a 32,768-token output budget, and requests to disable thinking. Astra uses independent per-sample sessions with no prior sample or research-discussion history.

We additionally evaluate Gemini-4-Argon (Google DeepMind, 2026b) and Gemini-3.8- Flash (Google DeepMind, 2026a) directly on full-length human audio, without supplying a transcript, speaker count, role labels, or gold annotations. Both use the same audio-task query and time-grounded scoring protocol, with independent per-recording requests. Appendix C.4 documents input encoding, request settings, and completion accounting.

Reference-script experiments provide all 26 scripts with or without speaker labels to Astra, Kimi-K3, DeepSeek V4 Flash, and Qwen3.8-Max, yielding 208 predictions. Turn order and text are unchanged; no additional cast list or speaker count is provided. Both conditions omit temporal verification.

Coverage and statistics. The original humandata experiments comprise 939 predictions across 41 model/input conditions. The main audio comparison extends the original five-pipeline cohort with two direct-audio systems, fixing the same 25 recordings, 293 gold trajectories, and 16 dialogue families for all seven systems. Each FunASR pipeline completes 26 recordings; each VibeVoice pipeline completes 25. Gemini retains the fixed denominator of 25. Other diagnostics retain their original systems and paired cohorts. All reported comparisons use frozen predictions and the same revised gold set.

DCIST is the sole primary metric. Terminalvalue recall, A/B joint success, and failure categories are diagnostic. Perturbation experiments additionally report CER, fixed-tokenization WER, and concatenated minimum-permutation WER (cpWER), all relative to reading scripts. These may include performance deviations from the script and are not error rates against independently verified verbatim speech. Speaker identity remains strict in DCIST.

Reported human-data confidence intervals use 20,000 dialogue-family bootstrap samples, resampling A/B variants together while retaining recording-macro averaging. Two-sided paired sign-permutation tests flip differences at the family level, enumerating up to 2<sup>16</sup> configurations. Holm correction jointly covers all 21 pairwise main-system comparisons; label ablations and B2 comparisons remain separate families of four and two tests. The threshold is 0.05. Appendix F reports the full methods and results.

## 5.2 Full-Trajectory Recovery

<table><tr><td>Input pipeline</td><td>Trajectory model</td><td>N DCIST</td><td>TP</td></tr><tr><td>FunASR FunASR</td><td>Astra Kimi-K3</td><td>25 25</td><td>21.35 82 9.80 38</td></tr><tr><td>FunASR VibeVoice</td><td>Qwen3.8-Max Kimi-K3</td><td>25 25</td><td>5.75 18 39</td></tr><tr><td>VibeVoice</td><td></td><td>25</td><td>11.09</td></tr><tr><td></td><td>Qwen3.8-Max</td><td></td><td>5.73 15</td></tr><tr><td>Direct audio</td><td></td><td>25</td><td></td></tr><tr><td>Direct audio</td><td>Gemini-4-Argon Gemini-3.8-Flash</td><td>25</td><td>18.82 72 6.59 19</td></tr></table>

Table 3: Full-trajectory recovery (DCIST, %) on a fixed cohort of 25 unperturbed human recordings, 16 dialogue families, and 293 gold trajectories. TP counts complete matches. VibeVoice uses native full-length transcription; Gemini receives full-length audio directly. All systems use the same minimum-chain protocol and time threshold. Appendix C.4 specifies the direct-audio configuration; Appendix F reports all 21 Holm-corrected comparisons and the retained cascade intervals.

The seven systems score 5.73–21.35% DCIST (Table 3). FunASR+Astra has the highest numerical score, recovering 82 of 293 complete trajectories. Direct-audio Gemini-4-Argon reaches 18.82% with 72 matches, while Gemini-3.8-Flash reaches 6.59% with 19. Kimi-K3 recovers 39 with VibeVoice and 38 with FunASR; Qwen recovers 18 with FunASR and 15 with VibeVoice. Since DCIST macro-averages recording-level F1, total matches alone do not determine ranking. Direct audio access therefore does not by itself ensure full-trajectory recovery.

The 95% family-clustered interval for FunASR+Astra is [9.70, 34.62]. Across all 21 Holmcorrected comparisons, it exceeds same-frontend Kimi-K3 and Qwen by 11.55 and 15.59 percentage points (pp), respectively (p = 0.0498 and 0.0371), and Gemini-3.8-Flash by 14.75 pp $( p \ \mathrm { ~  ~ { ~ = ~ } ~ } 0 . 0 2 0 5 )$ Gemini-4-Argon exceeds FunASR+Qwen and Gemini-3.8-Flash (p = 0.0205 and 0.0371). The Astra–Argon difference of 2.53 pp is not significant $( p = 0 . 7 0 3 1 )$ ; no other pair passes the corrected 0.05 threshold. This does not establish equivalence between systems.

Multi-event recovery is especially difficult. The cohort contains 131 one-node, 103 twonode, and 59 three-or-more-node gold trajectories. FunASR+Astra recalls 42.75%, 17.48%, and 13.56%, respectively; each Kimi pipeline matches only 1/59 in the last group. Speaker-count and duration strata are not monotonic, with scenario and chain complexity varying jointly (Appendix G).

## 5.3 Synthetic Speech

<table><tr><td>ASR</td><td>Reasoner</td><td>DCIST</td><td>No time</td></tr><tr><td>FunASR</td><td>DeepSeek</td><td>32.71</td><td>51.98</td></tr><tr><td>FunASR</td><td>Qwen</td><td>28.33</td><td>46.47</td></tr><tr><td>VibeVoice</td><td>DeepSeek</td><td>22.61</td><td>74.59</td></tr><tr><td>VibeVoice</td><td>Qwen</td><td>21.64</td><td>67.70</td></tr></table>

Table 4: Synthetic results on 2,224 shared recordings (5,269 gold trajectories). Scores are percentages. The no-time column disables temporal checks only.

Four systems complete 8,957 outputs and share 2,224 recordings with 5,269 gold trajectories. Table 4 applies the same monetary-equivalence, semantic-mapping, and minimum-chain rules. With time checks, FunASR+DeepSeek scores 32.71%, followed by FunASR+Qwen (28.33%), VibeVoice+DeepSeek (22.61%), and VibeVoice+Qwen (21.64%). Without time, VibeVoice+DeepSeek reaches 74.59% and FunASR+DeepSeek scores 51.98%. These results provide empirical evidence of task solvability: on short synthetic dialogues, systems can recover correct complete event chains from automatic transcripts without reference scripts. In particular, the higher untimed DCIST demonstrates successful joint recovery of content, attribution, and process structure, while timed evaluation additionally requires grounding those events in audio evidence. Appendix C.7 gives full coverage and scoring decomposition.

## 5.4 Reference Text and Speaker Labels

<table><tr><td>Model Labeled</td><td>TP Unlabeled</td><td>TP ∆ (pp)</td></tr><tr><td>Astra</td><td>38.72 133</td><td>28.99 100 +9.73</td></tr><tr><td>Kimi-K3</td><td>22.72 78</td><td>23.58 77 -0.86</td></tr><tr><td>DeepSeek V4 Flash</td><td>14.04 41</td><td>7.31 18 +6.73</td></tr><tr><td>Qwen3.8-Max</td><td>18.89 55</td><td>13.79 40 +5.10</td></tr></table>

Table 5: Reference-script results: 26 scripts, two input conditions, four models, and 208 predictions. Scores are percentages; ∆ is labeled minus unlabeled in percentage points. Both conditions disable time checks but retain strict speaker matching. TP counts complete trajectories.

With labeled scripts and no time requirement, the highest DCIST remains 38.72% (Astra, 133/308 complete matches; Table 5). Thus event selection, state interpretation, and relation recovery remain difficult even when designed wording and attribution are supplied. Removing labels lowers Astra, DeepSeek, and Qwen by 9.73, 6.73, and 5.10 pp, respectively; Kimi increases by 0.86 pp. Label benefits are significant for Astra and DeepSeek (both corrected $p = 0 . 0 1 9 )$ , but not Qwen or Kimi (0.285 and 0.766).

## 5.5 Terminal Values and Paired Dialogues

We project each frozen prediction onto its terminal value after removing exempt refinements in the scoring view. Gold fields with null terminal values are excluded. This diagnostic retains field/value matching but not intermediate events, speaker, scope, or time; a terminal value can remain tentative or held rather than settled.

Across 266 eligible fields in the common audio cohort, terminal-value recall is 32.71– 62.41%, versus 5.26–30.45% full-chain recall. FunASR+Astra recovers 166 terminal values, but 85 of those trajectories are incomplete or incorrect (51.20%). The corresponding proportion for other pipelines is 71.53–85.26%. With labeled scripts, Astra recovers 193 of 280 terminal values, of which 63 lack a correct full chain (32.64%). Correct terminal content therefore does not establish process recovery. These are recall diagnostics, not DCIST F1; all denominators and intervals appear in Appendix F.

A/B joint success requires both variants of a shared entity–field pair to match their respective full gold trajectories. Audio systems share nine complete families and 87 field pairs; scripts contain ten families and 102 pairs. FunASR+Astra jointly recovers 21/87 (24.14%), versus 41/102 (40.20%) with labeled scripts. On the subset whose gold structure or attributes change after ignoring identifiers and evidence times, these counts are only 3/35 and 14/40. Other audio pipelines jointly recover 1.15–14.94% of all paired fields. Role assignments and context can change together, so this is paired robustness, not a voice-only causal intervention.

## 5.6 Scoring Sensitivity

Kimi-K3 reviewed 2,366 deduplicated semantic candidates from the 939 human-data predictions. Keeping outputs, gold records, and other rules fixed, adopting all second-round objections and decisions on formerly unresolved candidates changes DCIST by at most 1.81 pp across all

41 conditions. The maximum occurs for Astra with unlabeled scripts: 28.99% to 30.80%, with complete matches increasing from 100 to 107. With labeled scripts, Astra increases from 38.72% to 40.50% (+1.78 pp): ten trajectories become matches and two cease to match, affecting 12/308 outcomes (3.90%) with a net gain of eight. This indicates that the primary score is relatively stable under the judgment differences observed in sequential review. A single human reviewer subsequently conducted an independent, trajectoryby-trajectory review of all changed match outcomes. Per-trajectory score changes and corresponding adjudication records are retained for inspection. Temporal-threshold and speaker-ID sensitivity analyses appear in Appendix G.

## 5.7 Targeted Acoustic Perturbations

We fix the trajectory model to Kimi-K3 and compare B2 keyword perturbation against unperturbed recordings using the same API route, prompt, and output budget. B2 selects 3,153 anchors for amounts, units, negation, confirmation, state expressions, and financial details, covering all 562 gold events. Selection uses original automatic word positions and coarse gold windows. Masks are frozen before downstream evaluation; approximately 4 dB noise changes 8.20% of fullrecording samples, preserving duration and all samples outside the masks.

<table><tr><td>Condition CER WER cpWER DCIST</td></tr><tr><td>FunASR (N = 26)</td></tr><tr><td>Original 17.12 21.69 31.18 9.68 B2 18.77 23.68 33.15 7.64</td></tr><tr><td>VibeVoice (N = 22)</td></tr><tr><td>Original 9.01 11.10 13.29 9.66</td></tr><tr><td>B2 10.41 12.57 14.21 6.34</td></tr></table>

Table 6: Same-route Kimi-K3 comparison with B2 keyword noise. All metrics are percentages. FunASR and VibeVoice share 26 and 22 completed recordings across conditions, respectively. Within each frontend, the API route, prompt, and output budget are held constant. Transcription metrics use the reading scripts as references.

In Table 6, FunASR’s script-referenced WER rises by 1.99 pp and DCIST falls by 2.04 pp, from 9.68% to 7.64%. VibeVoice’s WER rises by 1.47 pp, while DCIST falls by 3.32 pp, from 9.66% to 6.34%. Its smaller aggregate word-error increase accompanies a larger full-trajectory loss. The VibeVoice DCIST difference has a 95% interval of [−5.51, −0.96] and corrected p = 0.039; FunASR’s interval is [−8.15, 2.40], p = 0.625. Only the former provides statistically detectable degradation under this test.

Historical A1 and A2 perturbations target speech outside gold evidence while protecting identity, authority, and critical premises, changing 7.32% and 21.62% of samples. Their nonmonotonic results and automatic-speaker controls are reported separately in Appendix D. Evidenceexternal speech is not necessarily semantically irrelevant. Differences in routes, transcription budgets, and noise duration preclude a direct causal comparison of A and B2 locations. Automatic word boundaries and retained redundant evidence also limit attribution of the B2 effect to individual words.

## 5.8 Failure Analysis

The five cascaded systems produce 1,465 goldtrajectory evaluation observations: 192 matches and 1,273 failures. Required-chain mismatch accounts for 807 failures (63.39%), missing field candidates for 340 (26.71%), and whole-output format errors for 112 (8.80%). Thirteen predictions cover the required chain but contain invalid insertions; one violates trajectory structure. These are mutually exclusive matching barriers, not independent judgments of the model’s underlying error. A missing candidate can also reflect field granularity, rather than absence of all corresponding information. This error census concerns the five cascades, not the direct-audio extension. Appendix H provides a detailed discussion of Astra’s error distributions under both reference-script conditions, including category definitions and counts over all unmatched trajectories.

Exempt extras constitute 10.50–26.37% of parseable nodes for the audio pipelines and 17.58– 27.28% for labeled scripts. This measures output granularity, not hallucination rate, and does not increase true positives. Taken together, terminalvalue projections, paired variants, reference-text controls, and acoustic perturbations show why full-record correctness must be evaluated directly rather than inferred from a local extraction or transcription score.

## 6 Conclusion

We introduce DecisionTraceBench, a speechbased benchmark for event-chain tracking that jointly evaluates decision evolution across speakers, information attribution, scope, and temporal evidence. It provides the first quantitative assessment in speech understanding of how correct terminal values can mask errors in reconstructing the underlying event history. The benchmark offers a process-oriented evaluation of speech understanding for settings requiring accurate event reconstruction from audio, including financial operations, accident investigations, and critical meetings.

Our experiments show that accurately recovering complete event histories remains challenging for current models. The highest DCIST score on human recordings is 21.35%. For the bestperforming system, FunASR+Astra, 51.20% of the correctly recovered terminal values still correspond to incomplete or incorrect trajectories. Acoustic perturbation experiments further show that overall transcription error rates do not adequately reflect the recovery of decision-critical information: a higher overall transcription error rate does not necessarily imply a lower DCIST score. Complete decision-process recovery must therefore be evaluated directly, rather than inferred from transcription accuracy or terminal-value correctness.

We hope that DCIST will support the development of next-generation speech agents.

## Limitations

Representativeness. The human subset consists of 26 scripted Chinese recordings, with limited coverage of spontaneous meetings or multilingual practice. A/B variants share content, so the number of independent examples is smaller than the recording count.

Representation and scoring. Minimum required chains do not enumerate every valid event. Once the required chain is correctly recovered, additional accurate, fine-grained events do not increase the primary score, limiting the metric’s ability to distinguish such incremental capabilities.

Temporal localization. The coarse-window overlap rule tests evidence placement more directly than word-boundary precision. The small score changes under 50–90% predicted-window overlap thresholds do not imply accurate boundaries. Requiring coverage of half the gold window sharply reduces scores, showing that placement and boundary coverage are different targets. Fine-grained temporal evaluation requires finer annotation.

Comparability and generalization. Native fulllength processing, chunked inference, and auxiliary speaker alignment involve different computational and information conditions. Incomplete coverage can also affect comparisons. Familyclustered intervals quantify variation within the collected dialogue set, not repeated-generation variance or participant-independent population variation. Both subsets are fixed research evaluation cohorts. Low cross-source lexical overlap does not establish semantic-template independence. Validation in other record-sensitive settings requires independent data and experiments.

## Ethical Considerations

All human-recording participants provided informed consent, documented through signed forms and supplementary voice records, covering recording, foundational research, model evaluation, and public release. Attribution preferences are recorded separately. Dialogues are fictional and do not request real client or personal financial information. Participants were informed about transcription, annotation, segmentation, mixing, and noise augmentation, and that their voices may remain identifiable.

Public release follows each participant’s permissions and uses anonymous speaker identifiers. Human recordings authorized for public release are distributed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0); commercial use requires separate permission. Identity-to-recording mappings are not public, and acknowledgments follow individual preferences. Participants may withdraw permission for future use; affected recordings will be excluded from subsequent official versions, although previously distributed third-party copies cannot be guaranteed to be recalled.

Speech synthesis uses Alibaba’s open-source CosyVoice 2 model (Du et al., 2024). Referencevoice resources include public AISHELL corpora (Bu et al., 2017; Shi et al., 2021) and authorprovided recordings authorized for research synthesis; the 100.39-hour batch reported here is linked to AISHELL-1/3 entries. Third-party audio and derivatives are managed under their respective source licenses, separately from the humanrecording release. Volunteer recording consent is not treated as voice-cloning authorization. The benchmark evaluates research systems rather than establishing their suitability for operational lending or other high-stakes decisions.

AI assistance. Generative models assisted script construction, semantic adjudication, code preparation, and manuscript drafting, translation, and language editing. The model roles relevant to the experiments are described in the methods and appendices. Responsibility for the methods, results, citations, code, and claims remains with the authors.

## Acknowledgments

We thank Xudong Cai, Han Cui, Huifang Du, Jiajing Gong, Hongjin Hu, Tong Hu, Jia Kang, Yifei Sun, Xiaoyu Yan, Kai Yang, and Yuting Zhang for their contributions to the humanperformed recordings in DecisionTraceBench. We also thank the participants who requested anonymous acknowledgment for their time and contributions to the recording sessions.

## References

Arcada Labs. 2026. Audio Arena: Multi-Turn Speechto-Speech Evaluation Benchmarks.

Hui Bu, Jiayu Du, Xingyu Na, Bengu Wu, and Hao Zheng. 2017. AISHELL-1: An Open-Source Mandarin Speech Corpus and a Speech Recognition Baseline. In 2017 20th Conference of the Oriental Chapter of the International Coordinating Committee on Speech Databases and Speech I/O Systems and Assessment (O-COCOSDA), pages 1–5. IEEE.

Paweł Budzianowski, Tsung-Hsien Wen, Bo-Hsiang Tseng, Iñigo Casanueva, Stefan Ultes, Osman Ramadan, and Milica Gašic. 2018.´ MultiWOZ - A Large-Scale Multi-Domain Wizard-of-Oz Dataset for Task-Oriented Dialogue Modelling. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 5016–5026. Association for Computational Linguistics.

Jannis Bulian, Christian Buck, Wojciech Gajewski, Benjamin Börschinger, and Tal Schuster. 2022. Tomayto, tomahto. beyond token-level answer equivalence for question answering evaluation. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 291–305. Association for Computational Linguistics.

Yupeng Cao, Haohang Li, Yangyang Yu, Shashidhar Reddy Javaji, Yueru He, Jimin Huang, Qianqian Xie, Fabrizio Dimino, Xiao-yang Liu, K. P. Subbalakshmi, Meikang Qiu, Sophia Ananiadou, and Jian-Yun Nie. 2025. FinAudio: A Benchmark for Audio Large Language Models in Financial Applications. arXiv preprint arXiv:2503.20990.

Bhavana Dalvi, Lifu Huang, Niket Tandon, Wen-tau Yih, and Peter Clark. 2018. Tracking state changes in procedural text: a challenge dataset and models for process paragraph comprehension. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 1595–1604. Association for Computational Linguistics.

Zhihao Du, Yuxuan Wang, Qian Chen, Xian Shi, Xiang Lv, Tianyu Zhao, Zhifu Gao, Yexin Yang, Changfeng Gao, Hui Wang, Fan Yu, Huadai Liu, Zhengyan Sheng, Yue Gu, Chong Deng, Wen Wang, Shiliang Zhang, Zhijie Yan, and Jingren Zhou. 2024. CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models. arXiv preprint arXiv:2412.10117.

Ghazaleh Esfandiari-Baiat and Jens Edlund. 2026. Tracing consensus formation in meetings: Annotation and incremental decision modelling in the MEET corpus. In Proceedings of the 22nd Joint ACL - ISO Workshop on Interoperable Semantic Annotation and Representation (ISA-22) @ LREC 2026, pages 62–67, Palma, Mallorca (Spain). ELRA Language Resources Association (ELRA).

Zhifu Gao, Zerui Li, Jiaming Wang, Haoneng Luo, Xian Shi, Mengzhe Chen, Yabin Li, Lingyun Zuo, Zhihao Du, and Shiliang Zhang. 2023. FunASR: A Fundamental End-to-End Speech Recognition Toolkit. In Interspeech 2023, pages 1593–1597.

Google DeepMind. 2026a. Gemini 3.8 Flash Model Card. Accessed October 22, 2026.

Google DeepMind. 2026b. Gemini 4 Argon Model Card. Accessed October 22, 2026.

Yejin Kwon, Taewoo Kang, Hyunsoo Yoon, and Changouk Kim. 2025. M3-SLU: Evaluating Speaker-Attributed Reasoning in Multimodal Large Language Models. arXiv preprint arXiv:2510.19358.

Chunggi Lee and Hanspeter Pfister. 2026. AVERT: Audio-Verified Adjudication for Spoken Dialogue State Tracking. arXiv preprint arXiv:2609.01828.

Dongwook Lee, Sangkwon Park, Eunwoo Song, Che Hyun Lee, Youngho Cho, Junho Kim, June Young Yi, Heeseung Kim, and Sungroh Yoon. 2026a. HEAR Who Said What: Unlocking Speaker-Attributed Reasoning via Counterfactual Voice Grounding. arXiv preprint arXiv:2608.29120.

Jihyun Lee, Yejin Jeon, Wonjun Lee, Yunsu Kim, and Gary Geunbae Lee. 2023. Exploring the Viability

of Synthetic Audio Data for Audio-Based Dialogue State Tracking. In 2023 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU), pages 1–8. IEEE.

Jonggeun Lee, Junseong Pyo, Gyuhyeon Seo, and Yohan Jo. 2026b. SpeakerSleuth: Can large audiolanguage models judge speaker consistency across multi-turn dialogues? In Proceedings ofthe 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 20612– 20636, San Diego, California, United States. Association for Computational Linguistics.

Xueqing Peng, Lingfei Qian, Yan Wang, Ruoyu Xiang, Yueru He, Yang Ren, Mingyang Jiang, Vincent Jim Zhang, Yuqing Guo, Jeff Zhao, Huan He, Yi Han, Yun Feng, Yuechen Jiang, Yupeng Cao, Haohang Li, Yangyang Yu, Xiaoyu Wang, Penglei Gao, and 28 others. 2026a. MultiFinBen: Benchmarking large language models for multilingual and multimodal financial application. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 16934– 16963, San Diego, California, United States. Association for Computational Linguistics.

Zhiliang Peng, Jianwei Yu, Yaoyao Chang, Zilong Wang, Li Dong, Yingbo Hao, Yujie Tu, Chenyu Yang, Wenhui Wang, Songchen Xu, Yutao Sun, Hangbo Bao, Weijiang Xu, Yi Zhu, Zehua Wang, Ting Song, Yan Xia, Zewen Chi, Shaohan Huang, and 5 others. 2026b. VIBEVOICE-ASR Technical Report. arXiv preprint arXiv:2601.18184.

S Sakshi, Utkarsh Tyagi, Sonal Kumar, Ashish Seth, Ramaneswaran Selvakumar, Oriol Nieto, Ramani Duraiswami, Sreyan Ghosh, and Dinesh Manocha. 2025. MMAU: A Massive Multi-Task Audio Understanding and Reasoning Benchmark. In International Conference on Learning Representations.

Yao Shi, Hui Bu, Xin Xu, Shaoji Zhang, and Ming Li. 2021. AISHELL-3: A Multi-Speaker Mandarin TTS Corpus. In Interspeech 2021, pages 2756– 2760.

Shuzheng Si, Wentao Ma, Haoyu Gao, Yuchuan Wu, Ting-En Lin, Yinpei Dai, Hangyu Li, Rui Yan, Fei Huang, and Yongbin Li. 2023. SpokenWOZ: A Large-Scale Speech-Text Benchmark for Spoken Task-Oriented Dialogue Agents. In Advances in Neural Information Processing Systems, volume 36, pages 39088–39118. Curran Associates, Inc.

Niket Tandon, Keisuke Sakaguchi, Bhavana Dalvi, Dheeraj Rajagopal, Peter Clark, Michal Guerquin, Kyle Richardson, and Eduard Hovy. 2020. A dataset for tracking entities in open domain procedural text. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, pages 6408–6417. Association for Computational Linguistics.

Pat Verga, Sebastian Hofstätter, Sophia Althammer, Yixuan Su, Aleksandra Piktus, Arkady Arkhangorodsky, Minjie Xu, Naomi White, and Patrick Lewis. 2024. Replacing judges with juries: Evaluating LLM generations with a panel of diverse models. arXiv preprint arXiv:2404.18796.

Bin Wang, Meishan Zhang, Hao Fei, Yu Zhao, Bobo Li, Shengqiong Wu, Wei Ji, and Min Zhang. 2024. SpeechEE: A Novel Benchmark for Speech Event Extraction. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 10449– 10458. ACM.

Bin Wang, Xunlong Zou, Geyu Lin, Shuo Sun, Zhuohan Liu, Wenyu Zhang, Zhengyuan Liu, AiTi Aw, and Nancy F. Chen. 2025. AudioBench: A universal benchmark for audio large language models. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 4297–4316, Albuquerque, New Mexico. Association for Computational Linguistics.

Yang Xiao, Vidhyasaharan Sethu, Eun-Jung Holden, and Ting Dang. 2026. VoxMem: Benchmarking Multimodal Memory in Large Audio Language Models. arXiv preprint arXiv:2609.32607.

Utterance (English translation) Event   
S1: Start checking this equipment at CNY start: 3.0M   
3 million.   
S2: An invoice is missing. Pause this; we hold: 3.0M   
cannot finalize it yet.   
S2: The invoice and serial numbers now advance: 2.9M   
check out. Depreciation gives 2.9 million.   
Resume verification.   
S2: Yes, record the equipment at 2.9 mil- settle: 2.9M   
lion.  
Table 7: A required chain in an established technicalverification context. M denotes million CNY.

## A Worked Scoring Examples

## A.1 A Four-Event Required Chain

Table 7 gives a constructed example, not an observed dataset result. S1 handles materials and S2 performs technical verification. S2’s technical conclusion does not approve a loan. Speakers are numbered in first-occurrence order. The value at an event is the current candidate or recorded result, not necessarily a settled value.

The following is one trajectory with one cycle and four events. Intervals illustrate millisecond evidence fields. JSON Unicode escapes preserve the original Chinese monetary units: \u4e07 denotes ten thousand; the additional \u5143 denotes yuan. Thus number 300 with the first unit is CNY 3 million. No gold or prediction value is altered by the display encoding.

```jsonl
{"tracks": [{
"entity_id": "collateral_equipment",
"field_id":
"technical_recognized_value",
"derived_final_state": "settled",
"cycles": [{"cycle_id": "C1",
"events": [
{"event_id":"E1", "action":"start",
"speaker_id":"S1",
"scope_id":"material_verification",
"value":{"number":"300","unit":"\u
4e07"},
"source_event_id":null,
"start_ms":10000, "end_ms":15000},
{"event_id":"E2", "action":"hold",
"speaker_id":"S2",
"scope_id":"technical_review",
"value":{"number":"300","unit":"\u
4e07"},
"source_event_id":"E1",
"start_ms":18000, "end_ms":24000},
{"event_id":"E3",
"action":"advance",
"speaker_id":"S2",
"scope_id":"technical_review",
"value":{"number":"290","unit":"\u
4e07"},
"source_event_id":"E2",
"start_ms":55000, "end_ms":63000},
```

```json
{"event_id":"E4", "action":"settle",
"speaker_id":"S2",
"scope_id":"technical_review",
"value":{"number":"290","unit":"\u
4e07"},
"result":"accepted",
"source_event_id":"E3",
"start_ms":68000, "end_ms":72000}
]}]
}]}
```

E2 inherits the 3-million candidate; it is not an accepted technical value. Renaming E1–E4 to P1–P4 while consistently renaming references preserves the structure. Omitting an event, changing the speaker, or substituting a different scope does not.

## A.2 Correct Terminal Value, Incorrect Chain

The following prediction contains only the first and final events. Its terminal amount is correct and its own predecessor edge is well formed, but it omits the required suspension and resumption.

```jsonl
{"tracks": [{
"entity_id": "collateral_equipment",
"field_id":
"technical_recognized_value",
"derived_final_state": "settled",
"cycles": [{"cycle_id": "C1",
"events": [
{"event_id":"P1", "action":"start",
"speaker_id":"S1",
"scope_id":"material_verification",
"value":{"number":"300","unit":"\u
4e07\u5143"},
"source_event_id":null,
"start_ms":10000, "end_ms":15000},
{"event_id":"P2", "action":"settle",
"speaker_id":"S2",
"scope_id":"technical_review",
"value":{"number":"290","unit":"\u
4e07"},
"result":"accepted",
"source_event_id":"P1",
"start_ms":68000, "end_ms":72000}
]}]
}]}
```

Here G = P = 1, L = 0, and TP = 0, so F1 is zero. If a recording instead contains two gold trajectories and two predictions, with the other trajectory fully correct, then TP = 1 and F1 is 0.50. Node count alone is not the criterion: a longer prediction must be checked for legitimate refinements, whereas a missing required node always prevents a match.

For the gold interval [68000,72000] ms, prediction [67500,72500] overlaps by 4000 ms, or 0.80 of predicted duration, and passes. Prediction [66000,74000] overlaps by 0.50 and fails. Passing the time rule does not substitute for correct event attributes or relations.

## A.3 Additional Nodes and Trajectories

Legitimate refinement. Between E2 and E3, suppose S2 states at [30000,34000] ms: “The serial number also disagrees with the list. That must be resolved too; 3 million is still the candidate, and verification remains paused.” This new critical obstacle supports another hold node X. Its predecessor is E2, and E3 points to X. After adjudication, only the scoring view contracts E2–X–E3 into E2– E3. All required nodes are covered, F1 is 1, and one of five predicted nodes is exempt (20%).

Invalid insertion. Insert the same hold node, but let its evidence merely repeat: “The same invoice is still missing. Keep waiting.” Neither the target value nor a critical condition changes. The insertion is invalid even though all four required nodes are present: $T P = 0$ and F1 is zero. No extra node is exempt. Deleting this error to award full credit is not permitted.

A wholly new supported trajectory. Instead, retain the four-node chain and add S1’s statement at [76000,80000] ms: “We are also applying for a five-million loan; submit that amount for now.” It supports a separate loan\_request.application\_amount trajectory with one start, scope application\_presentation, a fivemillion value, and a null predecessor. This field is absent from the example gold. If all output content is supported, then G = 1, P = 2, L = 1, and TP = 1: F1 remains 1, not greater. Its node contributes to the extra proportion (one of five). An additional unsupported approval would prevent whole-trajectory exemption, although the supported start could still count separately in the auxiliary node statistic. These examples construct predictions for illustration; scoring never edits model answers to create them.

## B Relation to Existing Tasks

Full-trajectory recovery evaluates states together with the effective events that produce them. Financial fields instantiate the representation; the structural target is not intrinsically financial.

## C Data and Implementation

## C.1 Distribution and Scenarios

Synthetic scenarios comprise working-capital review (235 recordings), equipment valuation (209), guarantee review (208), use-of-funds investigation (200), drawdown-condition verification (194), repayment reconciliation (190), supply-chain financing (180), invoice financing (172), project drawdowns (170), certificate-of-deposit pledges (167), receivables assignment (91), syndicated participation intentions (91), mortgage negotiation (76), and export proceeds (63).

The human subset contains three recordings each for corporate credit due diligence, equipmentrenewal loan due diligence, resale-home mortgages and transaction funding, post-approval drawdown conditions, working-capital credit committees, and property-insurance claims. Audit adjustments, supply-chain financing, financial leasing/sale-and-leaseback, and syndicated loan share coordination each contribute two.

## C.2 Voice Sources, Evidence, and Speaking Rate

The 100.39-hour synthetic batch uses 193 sourcespeaker entries: 94 from AISHELL-1 (Bu et al., 2017) and 99 from AISHELL-3 (Shi et al., 2021). Reference provenance includes the source corpus, speaker identifier, file location, and hash. These are source-corpus entries, not a cross-corpus deduplication of physical identities. Turn-level synthesis and concatenation provide exact assembly positions. All 3,538 human-script turns have start/end times; trajectory events reference corresponding coarse windows on the original recording timeline.

The authors attended and supervised the human recording sessions, checking decision-critical content and the corresponding speaker roles against the recording scripts. This supervision ensured that the recorded key information was consistent with the events underlying the trajectory gold. Aligned scripts provide temporal locations for event evidence on the original recording timeline; trajectory annotation relies on supervised collection and event-level evidence rather than exhaustive word-level annotation. Candidate adjudication preserves differences between actual model input and reference scripts. Collection dates or scripted settings cannot establish relative years in an utterance, and scripted values cannot replace values in the actual transcription.

Human collection involved 22 participants, distinct from the two-to-four role labels in each script. Unperturbed FunASR transcripts contain 118,620 Chinese characters, Latin letters, and digits over 31,008.216 seconds. After excluding punctuation and whitespace but retaining repetitions and fillers, the overall rate is 229.53 characters/minute. Perrecording mean, median, and range are 231.53, 230.77, and 198.46–275.48. Duration includes pauses; these are automatic transcript-character estimates rather than articulation rates.

<table><tr><td>Task family</td><td>Existing evaluation or representation</td><td>Full-trajectory extension</td></tr><tr><td>MultiWOZ, ProPara, OpenPI (Budzianowski et al., 2018; Dalvi et al.. 2018; Tandon et al., 2020)</td><td>Dialogue belief states or entity changes in pro- cedural text</td><td>Joint recovery of spoken evolution, source identity, scope, and acoustic evidence</td></tr><tr><td>SpokenWOZ and spoken DST (Si et al., 2023; Lee et al., 2023; Lee and Pfister, 2026)</td><td>Turn-level slot updates, including synthetic-1 speech and audio-verification studies</td><td>Event actions, authority scope, and time evi- dence evaluated over the full chain</td></tr><tr><td>MEET consensus model (Esfandiari-Baiat and Ed- and time metadata lund, 2026)</td><td>Incremental decision history with provenance</td><td>A unified raw-audio prediction benchmark with joint trajectory scoring</td></tr><tr><td>M3-SLU / HEAR (Kwon et al., 2025; Lee et al., voice-swap controls 2026a)</td><td>Speaker binding, provenance questions, and</td><td>Attribution as part of effective decision events and their scoped consequences</td></tr><tr><td>2024) Audio Arena (Arcada Labs,</td><td>SpeechEE (Wang et al., Spoken event types, triggers, and arguments</td><td>The continuous suspension, resumption, and resolution of a proposition</td></tr><tr><td>2026) VoxMem (Xiao et al., 2026)</td><td>Correction, memory, and state maintenance in interactive speech</td><td>Traceable record recovery from an existing multi-speaker conversation</td></tr><tr><td></td><td>Question-conditioned cross-session retrieval and temporal reasoning</td><td>The complete event chain, rather than question-selected information, as the output</td></tr></table>

Table 8: Structured comparison of prediction targets.
<table><tr><td>Statistic</td><td>Synthetic</td><td>Human-recorded</td></tr><tr><td>Scenario categories</td><td>14</td><td>10</td></tr><tr><td>Two / three / four speakers</td><td>448 / 1,357 / 441</td><td>5 / 14 / 7</td></tr><tr><td>A/B pairs (recordings)</td><td>790 (1,580)</td><td>10 (20)</td></tr><tr><td>Standalone scripts</td><td>666</td><td>6</td></tr><tr><td>Dialogue families</td><td>1,456</td><td>16</td></tr><tr><td>Minimum / median / maximum duration (s)</td><td>95.47 / 154.50 / 272.39</td><td>495.50 / 997.53 / 3,264.00</td></tr><tr><td>Mean duration (s)</td><td>160.92</td><td>1,192.62</td></tr></table>

Table 9: Recording distribution. Speaker counts refer to script roles. Both A/B variants are counted as audio, but grouped into one family.

## C.3 Generation and Inference

The recorded script-experiment request identifiers are kimi-k3, qwen3.8-max, deepseek-v4-flash, and gpt-6-astra. The speech frontend methods are described by Gao et al. (2023); Peng et al. (2026b).

Construction proceeds through gold-trajectory design, gold-conditioned colloquial script generation, multi-model cross-review, rule checks, recording production, and time alignment. All 1,456 admitted synthetic script groups have both GLM and Qwen review records; 14 also have Kimi-K3 review, totaling 2,926 archived admission reviews. Hash-linked audio, text, gold records, and timing metadata support provenance.

Human gold review covers all 308 trajectories and 3,538 scripted turns, checking textual support, substantive state changes, and settlement evidence.

## C.4 Direct-Audio Gemini Evaluation

The requested and returned service identifiers are gemini-4-argon and gemini-3.8-flash. Each independent request includes one complete MP3 recording and the frozen audio-task query, without a transcript, cast list, speaker count, gold labels, or previous conversation. Temperature is 0, streaming is disabled, and provider-default thinking is retained. The initial output allowance is 15,360 tokens. Flash receives one 32,768-token retry per originally truncated request for R0005, R0006, and R0011, using the same complete audio and query; these are fresh requests rather than concatenated answers. Existing completed answers are retained regardless of their correctness.

Eight recordings are encoded as full-length mono 48-kbps MP3 copies to meet request-body limits; both systems receive identical audio bytes for these recordings. No recording is cropped or accelerated. The remaining inputs retain their original MP3 bytes. Input hashes, conversion receipts, request settings, and unmodified responses are retained for reproduction. These encoding and budget differences are part of the direct-audio configuration, not matched-compute ablations against the cascades.

<table><tr><td>Stage</td><td>Model / implementation</td><td>Configuration</td></tr><tr><td>Script generation</td><td>DeepSeek V4 Flash</td><td>Temperature 0.45; generates colloquial scripts conditioned on predesigned gold trajectories</td></tr><tr><td>Cross-review</td><td>GLM5.3-Flash; Qwen3.8-Max</td><td>Temperature 0.05; respective output budgets 18,000 and 9,000 tokens</td></tr><tr><td>Speech synthesis</td><td>CosyVoice2-0.5B (Du et al., 2024)</td><td>24 kHz, mono, 16-bit; speed 1.0; half precision; turn-level synthesis</td></tr><tr><td>Full-length transcription</td><td>FunASR</td><td>Punctuation-based segmentation; processing batch configured for 300 s; no supplied speaker count</td></tr><tr><td>Native transcription</td><td>VibeVoice-ASR</td><td>Non-sampling decoding, SDPA; 32,768-token budget for syn- thetic, original human, and B2; 65,536 for historical A1</td></tr><tr><td>Chunked transcription</td><td>VibeVoice with FunASR speaker alignment</td><td>Continuous non-overlapping 300-s and 60-s variants; 16,384 tokens per chunk; uncertain identities retained</td></tr><tr><td>API trajectory recovery</td><td>Kimi-K3; Qwen3.8-Max; DeepSeek V4 Flash</td><td>Temperature 0; 32,768-token budget; requests disable think- ing; actual responses and usage retained</td></tr><tr><td>Independent-session recovery</td><td>Astra (gpt-6-astra)</td><td>One sample per session with no other conversation history</td></tr></table>

Table 10: Generation and inference settings. Different frontends, chunk lengths, and auxiliary inputs define distinct conditions.

The main comparison fixes the existing 25- recording cohort and excludes R0024 for every system. Each direct-audio recording contributes one F1 value; an absent complete prediction contributes zero without changing N = 25. Argon returns 25 complete answers; Flash returns 24, including one format-invalid answer, and has one absent answer. These completion counts are distinct from semantic error categories. Earlier cascade, synthetic, and perturbation analyses retain their reported completed-output cohorts.

The same monetary-equivalence, field/scope/value review, additional-node adjudication, speaker, structure, and 80% temporaloverlap rules apply. Astra’s candidate judgments are checked by Kimi-K3 in 48 independent peroutput review batches covering 1,471 decisions: 451 expression candidates, 878 node candidates, and 142 whole-trajectory candidates. Eight objections are examined against the frozen rules; the finalized review leaves all trajectory-match outcomes and both main DCIST scores unchanged. This is sequential review with access to the first judgments, not blind inter-model agreement. The earlier 939-prediction sensitivity analysis remains a separate experiment.

## C.5 Pairs, Evaluation Cohorts, and Gold Revisions

Counterfactual pairs vary speakers, responsibilities, or confirmation relations around the same critical information. Pair review checks references and later responses; residual contextual differences are retained rather than represented as a pure voice swap. Families group shared scripts and A/B variants. The cohort manifest assigns all 2,246 synthetic recordings and all 26 human recordings to fixed research evaluation cohorts, with no training or development partition. Our evaluation assesses existing systems rather than train-to-test generalization; family identifiers preserve paired dependence in analyses.

A lexical overlap audit concatenates spoken text in script order, removes punctuation, whitespace, and speaker labels, retains Chinese characters, ASCII letters and digits, and lowercases letters. Across 315 cross-family human-recording pairs, there are no normalized whole-script duplicates. Character-five-gram set Jaccard overlap has median 0.08%, 90th percentile 0.41%, and maximum 1.65%. No identical complete turn of at least 12 characters is shared across human families. Within-family A/B overlap is intentional.

Across all 58,396 synthetic–human script pairs, normalized whole-script duplicates and identical complete turns of at least 12 characters are both absent. Character-five-gram Jaccard overlap has median 0.018%, 90th percentile 0.077%, and maximum 0.497%. No normalized whole-script duplicate occurs between distinct synthetic families. These lexical checks do not establish semantictemplate independence, speaker independence, or absence of pretraining contamination.

The project repository is https://github .com/QFIN-tech/DecisionTraceBench. Appendix A provides constructed scoring examples. The complete audio dataset and gold annotations will be publicly released after acceptance.

The current human gold version corrects unsupported monetary values and a repetition by an unauthorized participant previously treated as suspension. Three files, four trajectories, and six events are affected; total trajectory and event counts are unchanged. Triggering turns, speakers, conclusions, and time windows are synchronized; other disputed boundaries remain unchanged. Replacing only these records leaves DCIST unchanged for all 939 predictions, because other required constraints still prevent full matches.

## C.6 Noise Verification

A1, A2, and B2 each contain 26 full-length copies. Samplewise checks confirm unchanged samples outside masks and unchanged protected identity, authority, premise, and gold regions for A conditions. A2 changes 21.62% overall; 25 recordings exceed 20%, while one reaches 11.58% because its safe evidence-external region is insufficient. B2 changes 40,659,118 samples (8.20%), with all 26 passing input-correspondence checks. FunASR completes 26 B2 transcriptions; VibeVoice completes 22. The four missing VibeVoice samples are not measured zeros.

## C.7 Synthetic Evaluation

The four systems complete 2,232, 2,245, 2,239, and 2,241 recordings (99.38%, 99.96%, 99.69%, and 99.78% of 2,246). The shared cohort contains 2,224 recordings and 5,269 gold trajectories; full-coverage scores appear in Table 11. Existing predictions and gold remain frozen. Actions, speaker identity, scope, numerical content, cycles, and predecessor relations remain required. Primary semantic review comprises 18,518 candidate decisions. Review closure covers every candidate capable of changing DCIST; a candidate can be skipped only when code establishes score invariance, never by marking an unreviewed item equivalent or unresolved. This procedure determines the main score but does not classify every superfluous node, so we do not report a complete synthetic extra-node ratio.

Required events use frozen synthesis timestamps. Extra-event semantic and structural decisions are shared across the timed and untimed conditions; timed scoring additionally checks the principal triggering turn against synthesis metadata at the 80% overlap threshold. Removing time removes that check only. The scoring implementation passes 24 regression tests and the final evidence/timing audits.

Table 12 separates successive scoring changes on fixed outputs. Under temporal constraints, monetary normalization adds 0.02–0.28 pp, expression mappings add 11.21–21.35 pp, and extranode adjudication adds 0.05–0.81 pp. These increments depend on the order of operations, not on improved model predictions. The large expression component highlights the sensitivity of literal field/scope/value matching; it is distinct from the 1.81-pp second-model sensitivity bound measured on human experiments.

## D Acoustic Perturbations

B2 uses 3,153 anchors covering 562 gold events: 1,146 numeric/unit/term/date anchors, 925 negation/state/confirmation anchors, and 1,082 financial-detail anchors. Each is expanded by 160 ms on both sides, with overlapping intervals merged. Boundaries are automatic, not manually verified words. Redundant evidence outside masks remains audible; no mask is selected or changed to obtain a desired downstream error.

## D.1 Historical Evidence-External Noise

In Table 14, FunASR’s script-referenced WER increases from 21.69% to 24.17% (A1) and 29.51% (A2), while DCIST is 12.41%, 7.91%, and 10.19%, respectively. Larger global deviation in A2 does not yield greater trajectory loss: the decline is 2.22 pp versus 4.50 pp for A1. On 23 shared VibeVoice recordings, WER increases from 11.13% to 12.34% but DCIST increases from 10.10% to 11.22%. These non-monotonic patterns motivate separating recognition changes from downstream variation through repeated inference and event-level analysis. They are not a controlled causal contrast with B2.

## D.2 Automatic Speaker Control

Replacing noisy automatic attribution with the original recording’s automatic segmentation increases Kimi’s score by 1.82 pp for FunASR A1 and 1.24 pp for VibeVoice A1, but decreases FunASR A2 by 1.25 pp. These controls do not consistently eliminate trajectory error. They use automatic identities, not human gold speakers, and are not substitutes for diarization error evaluation. None of the three corrected tests is significant (Table 19).

<table><tr><td>ASR</td><td>Reasoner</td><td>Completed / total</td><td>Format errors</td><td>DCIST</td><td>No time</td></tr><tr><td>FunASR</td><td>DeepSeek</td><td>2232/2246</td><td>66</td><td>32.65</td><td>51.88</td></tr><tr><td>FunASR</td><td>Qwen</td><td>2245/2246</td><td>95</td><td>28.17</td><td>46.25</td></tr><tr><td>VibeVoice</td><td>DeepSeek</td><td>2239/2246</td><td>53</td><td>22.60</td><td>74.44</td></tr><tr><td>VibeVoice</td><td>Qwen</td><td>2241/2246</td><td>65</td><td>21.66</td><td>67.57</td></tr></table>

Table 11: Synthetic results over all completed outputs per system, with coverage reported separately. DeepSeek denotes DeepSeek V4 Flash and Qwen denotes Qwen3.8-Max. Complete malformed answers remain scored; incomplete outputs are not imputed as zeros.
<table><tr><td>ASR</td><td>Reasoner</td><td>Time</td><td>Legacy</td><td>+ Amount</td><td>+ Semantics</td><td>+ Extras</td></tr><tr><td>FunASR</td><td>DeepSeek</td><td>Yes</td><td>15.42</td><td>15.62</td><td>31.90</td><td>32.71</td></tr><tr><td>FunASR</td><td>DeepSeek</td><td>No</td><td>24.16</td><td>24.45</td><td>50.55</td><td>51.98</td></tr><tr><td>FunASR</td><td>Qwen</td><td>Yes</td><td>6.47</td><td>6.75</td><td>28.10</td><td>28.33</td></tr><tr><td>FunASR</td><td>Qwen</td><td>No</td><td>10.38</td><td>10.92</td><td>46.08</td><td>46.47</td></tr><tr><td>VibeVoice</td><td>DeepSeek</td><td>Yes</td><td>10.95</td><td>10.97</td><td>22.18</td><td>22.61</td></tr><tr><td>VibeVoice</td><td>DeepSeek</td><td>No</td><td>36.18</td><td>36.21</td><td>72.96</td><td>74.59</td></tr><tr><td>VibeVoice</td><td>Qwen</td><td>Yes</td><td>5.97</td><td>6.00</td><td>21.59</td><td>21.64</td></tr><tr><td>VibeVoice</td><td>Qwen</td><td>No</td><td>17.65</td><td>17.80</td><td>67.41</td><td>67.70</td></tr></table>

Table 12: Sequential scoring decomposition on the same 2,224 recordings, keeping predictions and gold fixed. Columns cumulatively introduce monetary equivalence, reviewed expression mappings, and extra-node adjudication; the last column is the reported score. Increments depend on this order and are not independent causal effects.

## E Scoring Protocol

## E.1 Required Attributes and Relations

Matching jointly checks the tracked field, cycles, and event attributes. Cycle count, boundaries, and order must agree; an extra cycle cannot be deleted as an exempt node. Within each cycle, every required event must be covered one-to-one in order. Action, speaker, and settlement result must agree. Event and cycle identifiers can be consistently renamed, but speaker permutation and content editing cannot repair a prediction. The derived final state and top-level reason are not separately scored; content nested inside a value belongs to that value.

Numbers are output as Arabic-digit strings with spoken units. Monetary equivalence uses exact decimal conversion within an explicit currency and unit scale; it neither guesses missing magnitudes nor applies exchange rates or general duration conversions. Composite values must preserve all critical components, with unordered object keys. A node that expresses a required event incorrectly is a mislabeled substitution, not a legitimate extra; exemption cannot bypass requiredevent constraints.

## E.2 Sequential Semantic Review

First stage. Astra judges whether gold and predicted expressions are equivalent in context, saving evidence and rationale; insufficient evidence remains unresolved. Field mappings require the same subject, attribute, and business proposition. Scope mappings require the same authority conditions, confirmation level, and business effect. Nonnumeric values allow synonyms, cross-language terms, abbreviations, or lossless structural representations. Equal numbers alone do not justify merging fields or scopes. Amounts, durations, negation, time ranges, and necessary conditions cannot be deleted or supplied.

Second stage. We simplify this stage to a directed check of explicit first-round contradictions. Primary scoring uses the frozen judgments finalized after human review of disagreements between Astra and Kimi-K3. A separate sensitivity analysis evaluates the hypothetical adoption of all secondmodel objections and decisions on formerly unresolved candidates. Each of 2,366 deduplicated candidates from 939 human-data predictions is separately submitted to Kimi-K3 with the original prediction, gold, relevant events and reference turns, and Astra’s judgment and reason. Kimi must identify a specific first-round error, applicable rule, and locatable evidence before contradicting it. Another permissible interpretation or naming preference is not an objection. For formerly unresolved candidates, it returns a binary decision when supported, otherwise identifies missing context. Because it sees the first opinion, this is sequential verification, not blind independent agreement.

<table><tr><td>Condition</td><td>Region</td><td>Mask duration</td><td>Changed samples</td></tr><tr><td>Original</td><td>No added noise</td><td></td><td>Reference</td></tr><tr><td>A1</td><td>Outside protected gold evidence</td><td>2,270.24 s</td><td>7.32% overall</td></tr><tr><td>A2</td><td>Expanded evidence-external speech</td><td>6,705.04 s</td><td>21.62%; one 11.58% exception</td></tr><tr><td>B2</td><td>Critical short words</td><td>2,541.69 s</td><td>8.20% overall</td></tr></table>

Table 13: Frozen perturbation conditions, each with 26 full-length recordings and approximately 4 dB noise. Changed-sample proportions are measured against original waveforms. The other 25 A2 recordings range from 21.01% to 22.89%.
<table><tr><td>Frontend</td><td>Condition</td><td>N</td><td>CER (%)</td><td>WER (%)</td><td>cpWER (%)</td><td>DCIST (%)</td></tr><tr><td>FunASR</td><td>Original</td><td>26</td><td>17.12</td><td>21.69</td><td>31.18</td><td>12.41</td></tr><tr><td>FunASR</td><td>A1</td><td>26</td><td>19.20</td><td>24.17</td><td>34.59</td><td>7.91</td></tr><tr><td>FunASR</td><td>A2</td><td>26</td><td>23.67</td><td>29.51</td><td>43.00</td><td>10.19</td></tr><tr><td>VibeVoice</td><td>Original</td><td>23</td><td>9.10</td><td>11.13</td><td>13.33</td><td>10.10</td></tr><tr><td>VibeVoice</td><td>A1</td><td>23</td><td>10.09</td><td>12.34</td><td>14.23</td><td>11.22</td></tr></table>

Table 14: Historical Kimi-K3 evidence-external noise experiments: 26 common FunASR recordings and 23 common VibeVoice recordings. Transcription metrics aggregate corpus-level errors against reading scripts. These routes differ from the B2 control.

Frozen scoring and scope. Records retain candidate IDs, both judgments, rules, and evidence positions. Disagreements between the two models are resolved through human review before the judgments used for primary scoring are frozen. Confirmed equivalent expressions participate normally in required-event matching, not as exempt extras. Mappings apply only to their sample and candidate; they are not global aliases and cannot change speaker, action, time, cycle, or source-edge constraints. Deterministic code uses the finalized expression judgments for structure checks, maximum one-to-one matching, and aggregation.

Coverage and sensitivity. All 2,366 candidates received second-stage responses: 2,108 did not meet the clear-error objection threshold, 154 received contrary judgments, 103 formerly unresolved candidates received decisions, and one remained unresolved at the second-model stage. Contrary judgments comprise 129 non-equivalentto-equivalent and 25 equivalent-to-non-equivalent decisions. These counts summarize judgments at the second-model review stage.

The all-adopted sensitivity condition applies the 154 objections and 103 decisions while freezing gold, predictions, structure rules, and extra-node judgments. It measures the effect of the second model’s proposals, not confirmed first-round error. Across 41 conditions, the maximum absolute DCIST change is 1.81 pp. Astra’s unlabeledscript score rises from 28.99% to 30.80%, with seven trajectories becoming correct (100 to 107 matches). Its labeled-script score rises from 38.72% to 40.50%: ten new matches and two lost matches, 12 changed outcomes out of 308 (3.90%), and 133 to 141 total matches. These changes indicate relative stability within this sequential review while leaving the full-recovery challenge intact.

Human review and traceability. Candidates on which Astra and Kimi-K3 disagree are reviewed by a human, who makes the final equivalent or non-equivalent judgment. A single human reviewer also independently examined every trajectory whose match outcome changed. Prediction and gold coordinates, both model judgments, and score-change records are retained for each case to support traceable inspection.

## E.3 Additional-Content Adjudication

After expression matching, code identifies remaining unmatched nodes and absent-gold fields. Astra reviews them against actual model input, context, reference script, and frozen task rules. Supported, permitted refinements are exempt. Reviewed candidates with a specific unresolved evidence or rule boundary are also exempt; unreviewed candidates are not. Repetition without change, ordinary questions, cross-field reuse of a conclusion, and unsupported or rule-violating insertions are not exempt. A mentioned number alone does not justify an event: action, field, value, speaker, scope, time, and relations all matter. Any erroneous insertion invalidates its trajectory.

<table><tr><td>Kimi-K3 condition</td><td>N</td><td>Automatic (%)</td><td>Fixed (%)</td><td>Difference</td></tr><tr><td>FunASR A1</td><td>26</td><td>7.91</td><td>9.73</td><td>+1.82</td></tr><tr><td>FunASR A2</td><td>26</td><td>10.19</td><td>8.94</td><td>-1.25</td></tr><tr><td>VibeVoice A1</td><td>23</td><td>11.22</td><td>12.46</td><td>+1.24</td></tr></table>

Table 15: Automatic-speaker control. Fixed attribution uses speaker predictions from the unperturbed recording, not human gold identities. Differences are percentage points.

For a gold-corresponding trajectory, exempt nodes are removed only from the scoring view and predecessor paths are contracted. This cannot fix unsupported edges, cycle misalignment, substituted required events, or missing required events. For a wholly new independent field, all predicted nodes and relations are reviewed without inventing unannotated later events. If all output content is valid, or reviewed evidence remains unresolved without a definite error, the new trajectory is exempt from the denominator but not counted as a true positive. Any invalid node or relation prevents whole-trajectory exemption. Renaming an existing gold event to a related field is not new independent information.

Whole-trajectory records and expanded node records are deduplicated by original coordinates; rejecting a whole trajectory does not classify every constituent node as wrong. Actual cascade input takes precedence for what the model observed. A coarse script cannot resolve a discrepancy requiring listening and cannot repair the transcript or answer. Such evidence remains unresolved. Prediction and adjudication sessions are separate, but an Astra prediction and its Astra review still use the same model. The protocol is model-assisted semantic/evidence adjudication followed by deterministic scoring, not entirely mechanical scoring.

## E.4 Aggregation and Verification

Predicted evidence uses $| P \cap G | / | P | \geq 0 . 8 0 .$ with positive predicted duration. Every required event must pass. After exemptions and edge contraction, maximum one-to-one trajectory matching supplies $T P _ { i }$ for the main equations. Global micro-F1 is supplementary; actions have no extra weights. Complete malformed answers are output errors. Cohort-specific completion accounting follows Appendix C.4 for direct audio and the reported completed-output cohorts for the earlier experiments.

Let U contain all parseable predicted nodes and E the union of individually exempt nodes and nodes in wholly exempt new trajectories. The auxiliary ratio is

$$
R _ { \mathrm { { e x t r a } } } = | E | / | U | .\tag{3}
$$

Unique coordinates include input condition, recording, trajectory, cycle, and event position. A three-node new trajectory contributes three nodes, not one. Legitimate extras can remain in E even if another error invalidates their trajectory. This ratio is neither a score bonus nor normalized by gold-node count. The human-data extranode ratio uses completed adjudication coverage. Synthetic review closes all score-relevant candidates through explicit judgments or verified scoreinvariance checks; it does not provide an exhaustive extra-node classification.

Frozen reproduction tests cover money equivalence, the 80% time rule, identity/actions, edges, cycles, exemptions, and one-to-one matching. The manuscript’s scoring snapshot passed 40 regression tests and self-matching for all 26 gold files with and without time. Sequential-review sensitivity reproduction recovered all 939 baseline results, checked matching counts and F1 arithmetic, and verified frozen-input and judgment hashes. Review records preserve original positions, evidence, rule references, and reasons.

## F Statistical Methods and Full Results

## F.1 Clustered Intervals and Paired Tests

There are 16 human dialogue families: each A/B pair is one cluster, and each independent long script is one. For a comparison with K families, resample K families with replacement, include all available recordings of each selected family, and recompute recording-macro DCIST. We use 20,000 draws and percentile 2.5/97.5 limits. This preserves recording weights instead of first averaging families equally. The main 25-recording cohort still contains 16 families; paired A/B diagnostics require both variants.

The main-system tests use the fixed 25- recording cohort and its completion accounting in Appendix C.4; earlier paired ablations retain their jointly completed cohorts. Differences for every recording within a family receive the same sign, with all $\bar { 2 } ^ { K }$ assignments enumerated. Twosided $p$ is the fraction with an absolute recordingmacro difference at least the observed value. All main comparisons contain 16 families and enumerate 65,536 assignments. Holm correction jointly covers the 21 contrasts between seven main systems. Four label contrasts, two B2 contrasts, three historical A contrasts, and three attribution controls remain separate test families. Intervals are pointwise, not simultaneous; significance uses corrected $p .$

For sorted raw values $p _ { ( 1 ) } \leq \dots \leq p _ { ( 2 1 ) }$ , the adjusted value at rank j is min $\{ 1 , \operatorname* { m a x } _ { k \leq j } ( 2 2 -$ $k ) p _ { ( k ) } \}$ }. Correction uses unrounded values. Before extending the family, all ten original cascade raw and adjusted $p$ values were exactly reproduced. All 21 new tail counts were verified by an independent enumeration. The two original significant cascade contrasts remain significant under the larger family; three contrasts involving directaudio models also pass 0.05 (Table 16).

The base seed is 20261013; name hashes derive reproducible independent streams for statistical quantities. All predictions, gold, mappings, and extra judgments stay fixed. Differences are computed before rounding; subtracting displayed scores can therefore differ by 0.01 pp. Intervals reflect sampled dialogue-family variation, not repeated generation or independent humanparticipant sampling.

Historical FunASR A1 declines by 4.50 pp with corrected $p = 0 . 0 4 7 $ ; FunASR A2 and VibeVoice A1 do not meet 0.05. None of the fixed automaticspeaker controls meets that threshold. Therefore the numerical speaker-control changes alone do not establish a segmentation effect.

## F.2 Terminal-Value Projection

The final required event in the final cycle supplies each non-null gold terminal value. There are 280 such fields across all 26 recordings and 266 in the main cohort; 28 null-valued fields are excluded from the all-recording diagnostic. Field and value comparisons retain exact labels, frozen sample-local mappings, and monetary conversion; no new equivalences are adjudicated for this projection.

Predictions retain their original event order. Required matches remain, exempt refinements are removed in the scoring view, and the remaining terminal value is projected. Field assignment is oneto-one, prioritizing existing full-chain matches; one prediction cannot hit multiple gold fields. The output-format gate remains, but the diagnostic does not additionally require intermediate events, action, speaker, scope, or time. It is a projection of the same answer, not a separately prompted model task.

For eligible gold fields D, terminal-value hits $V .$ and full-chain hits $T$ , we report $| V | / | D | , | T | / | D |$ and $| V \setminus T | / | V |$ . All observations satisfy $T \subseteq V$ Frozen records retain denominators, field coordinates, projected prediction nodes, and mapping IDs.

## F.3 A/B Joint Recovery

Pair variants by family metadata and match shared entity–field keys. For shared field pairs $Q .$

$$
J _ { A B } = \frac { 1 } { | Q | } \sum _ { q \in Q } { \bf 1 } [ q _ { A } \mathrm { m a t c h e s } \land q _ { B } \mathrm { m a t c h e s } ] .\tag{4}
$$

Each variant must satisfy its own full gold trajectory; the two gold records need not be identical. Fields unique to one variant remain in DCIST but not this diagnostic. One audio family lacks a completed variant, leaving nine pairs/87 shared fields; scripts retain ten/102.

The changed subset compares cycles, actions, speakers, scopes, values, settlement results, and predecessor positions, normalizing explicit monetary units and ignoring renamable IDs, reasons, and evidence times. This yields 35 audio and 40 script field pairs. It adds no semantic judgment. Intervals resample paired families and aggregate field-pair counts.

<table><tr><td>Comparison (first minus second)</td><td>Difference (pp)</td><td>Raw p</td><td>PHolm</td></tr><tr><td>F+Astra − F+Kimi</td><td>+11.55</td><td>0.0029</td><td>0.0498</td></tr><tr><td>F+Astra − F+Qwen</td><td>+15.59</td><td>0.0020</td><td>0.0371</td></tr><tr><td>F+Astra − V+Kimi</td><td>+10.26</td><td>0.0078</td><td>0.1172</td></tr><tr><td>F+Astra − V+Qwen</td><td>+15.62</td><td>0.0137</td><td>0.1777</td></tr><tr><td>F+Astra – G-Argon</td><td>+2.53</td><td>0.0703</td><td>0.7031</td></tr><tr><td>F+Astra − G-Flash</td><td>+14.75</td><td>0.0010</td><td>0.0205</td></tr><tr><td>F+Kimi - F+Qwen</td><td>+4.04</td><td>0.3906</td><td>1.0000</td></tr><tr><td>F+Kimi − V+Kimi</td><td>-1.29</td><td>0.5049</td><td>1.0000</td></tr><tr><td>F+Kimi - V+Qwen</td><td>+4.07</td><td>0.6211</td><td>1.0000</td></tr><tr><td>F+Kimi – G-Argon</td><td>-9.02</td><td>0.0068</td><td>0.1094</td></tr><tr><td>F+Kimi — G-Flash</td><td>+3.20</td><td>0.5234</td><td>1.0000</td></tr><tr><td>F+Qwen - V+Kimi</td><td>-5.33</td><td>0.0547</td><td>0.6016</td></tr><tr><td>F+Qwen - V+Qwen</td><td>+0.03</td><td>0.9961</td><td>1.0000</td></tr><tr><td>F+Qwen – G-Argon</td><td>-13.07</td><td>0.0010</td><td>0.0205</td></tr><tr><td> $\mathrm { F + Q w e n - G \mathrm { - } F l a s h }$ </td><td>-0.84</td><td>0.5312</td><td>1.0000</td></tr><tr><td>V+Kimi − V+Qwen</td><td>+5.36</td><td>0.2266</td><td>1.0000</td></tr><tr><td>V+Kimi − G-Argon</td><td>-7.73</td><td>0.0088</td><td>0.1230</td></tr><tr><td>V+Kimi – G-Flash</td><td>+4.49</td><td>0.1641</td><td>1.0000</td></tr><tr><td> $\mathrm { V + Q w e n - G { \mathrm { - } } A r g o n }$ </td><td>-13.09</td><td>0.0195</td><td>0.2344</td></tr><tr><td> $\mathrm { V + Q w e n - G \mathrm { - } F l a s h }$ </td><td>-0.87</td><td>0.7070</td><td>1.0000</td></tr><tr><td> $\mathrm { G } { \cdot } \mathrm { A r g o n } - \mathrm { G } { \cdot } \mathrm { F l a s h }$ </td><td>+12.23</td><td>0.0020</td><td>0.0371</td></tr></table>

Table 16: All 21 pairwise main-system comparisons on the fixed 25-recording cohort in 16 families. F/V denote FunASR/VibeVoice; G-Argon/G-Flash denote Gemini-4-Argon/Gemini-3.8-Flash. Two-sided exact family-level sign-permutation tests enumerate 65,536 assignments. Holm correction covers all 21 contrasts jointly; bold adjusted values pass 0.05 before rounding. Differences are recording-macro DCIST percentage points.

## G Stratification and Local Diagnostics

## G.1 Speakers, Duration, and Chain Length

Speaker-count strata contain five two-speaker, thirteen three-speaker, and seven four-speaker recordings. Duration strata contain nine recordings below 15 minutes, twelve from 15 to below 30, and four of at least 30 minutes. Each dimension partitions the same 25-recording cohort and macro-averages F1 within strata. Scenario, length, and speaker count are not independently controlled; four long recordings cannot establish a stable length-degradation law. Chain length counts required gold events across all cycles, not predicted refinements. Recall is complete matches divided by gold trajectories in that bin; unmatched predicted fields are not arbitrarily allocated to a length bin. One-node recall spans 7.63–42.75%, while three-or-more-node recall spans 1.69–13.56% across systems.

## G.2 Extra Nodes and Micro-Averaging

The exemption union deduplicates node coordinates and includes supported or reviewedunresolved extras. Nodes selected as required matches are removed from that union; a formerly new trajectory matched to gold is no longer wholly exempt. All nodes in parseable output form the denominator, even if that output fails a schema check.

Unparseable output contributes no invented node count but is still scored by the format rule. Micro-F1 is $2 \textstyle \sum _ { i } T P _ { i } / ( \sum _ { i } G _ { i } + \sum _ { i } ( P _ { i } - L _ { i } ) )$ , weighting recordings with more trajectories more heavily than recording-macro DCIST. The extra proportion describes granularity, not hallucination, and cannot cancel omissions.

## G.3 Time Windows and Speaker Identifiers

Each diagnostic changes one mechanical requirement while holding predictions, gold, semantic mappings, and extra judgments fixed. No previously unmapped expression becomes equivalent automatically. Moving alignment anchors introduced no blocked, newly unreviewed extra candidates in these time diagnostics. Since mappings were established under the 80% protocol, the notime diagnostic is not a newly adjudicated upper bound.

Valid predicted windows have median lengths of 5.06–10.20 seconds. Of 2,673 windows, only seven are at most one second. Within fully matched trajectories, median predicted/gold window-length ratios range from 15.54% to 48.56%. Requiring coverage of at least half the gold window, in addition to the official 80% predicted-window criterion, reduces scores to 0.00–3.99%. Coarse evidence windows often exceed the cited phrase; this added constraint measures boundary coverage rather than simply correct contextual location.

<table><tr><td>System / contrast</td><td>Estimate (% / pp)</td><td>Pointwise 95% CI</td></tr><tr><td>F+Astra</td><td>21.35</td><td>[9.70, 34.62]</td></tr><tr><td>F+Kimi</td><td>9.80</td><td>[3.08, 19.52]</td></tr><tr><td>F+Qwen</td><td>5.75</td><td>[1.16, 11.19]</td></tr><tr><td>V+Kimi</td><td>11.09</td><td>[4.46, 19.17]</td></tr><tr><td>V+Qwen</td><td>5.73</td><td>[2.30, 9.46]</td></tr><tr><td>G-Argon</td><td>18.82</td><td>[8.55, 30.71]</td></tr><tr><td>G-Flash</td><td>6.59</td><td>[1.60, 12.77]</td></tr><tr><td> $\mathrm { F + A s t r a - F + K i m i }$ </td><td>+11.55</td><td>[5.12, 18.69]</td></tr><tr><td> $\mathrm { F + A s t r a - F + Q w e n }$ </td><td>+15.59</td><td>[6.84, 25.79]</td></tr><tr><td> $\mathrm { F + A s t r a \mathrm { ~ - ~ } V + K i m i }$ </td><td>+10.26</td><td>[3.98, 17.46]</td></tr><tr><td> $\mathrm { F + A s t r a - V + Q w e n }$ </td><td>+15.62</td><td>[4.60, 29.29]</td></tr><tr><td>F+Astra – G-Argon</td><td>+2.53</td><td>[0.32, 4.81]</td></tr><tr><td>F+Astra – G-Flash</td><td>+14.75</td><td>[5.80, 25.49]</td></tr><tr><td>F+Kimi - F+Qwen</td><td>+4.04</td><td>[-2.91, 11.87]</td></tr><tr><td>F+Kimi − V+Kimi</td><td>-1.29</td><td>[-4.99, 2.19]</td></tr><tr><td>F+Kimi − V+Qwen</td><td>+4.07</td><td>[-3.45, 15.04]</td></tr><tr><td>F+Kimi – G-Argon</td><td>-9.02</td><td>[-15.47, -3.64]</td></tr><tr><td>F+Kimi – G-Flash</td><td>+3.20</td><td>[-4.84, 12.18]</td></tr><tr><td> $\mathrm { F + Q w e n - V + K i m i }$ </td><td>-5.33</td><td>[-11.24, -0.60]</td></tr><tr><td>F+Qwen - V+Qwen</td><td>+0.03</td><td>[-4.00, 4.36]</td></tr><tr><td> $\mathrm { F + Q w e n - G { \mathrm { - } } A r g o n }$ </td><td>-13.07</td><td>[-22.00, -5.56]</td></tr><tr><td> $\mathrm { F + Q w e n - G \mathrm { - } F l a s h }$ </td><td>-0.84</td><td>[-3.28, 1.37]</td></tr><tr><td> $\mathrm { \Delta V + K i m i - V + Q w e n }$ </td><td>+5.36</td><td>[-1.15, 14.45]</td></tr><tr><td> $\mathrm { V + K i m i - G { \mathrm { - } } A r g o n }$ </td><td>-7.73</td><td>[-13.60, -2.77]</td></tr><tr><td> $\mathrm { V + K i m i - G { \mathrm { - } } F l a s h }$ </td><td>+4.49</td><td>[-0.74, 10.88]</td></tr><tr><td> $\mathrm { V + Q w e n - G { \mathrm { - } } A r g o n }$ </td><td>-13.09</td><td>[-25.28, -3.51]</td></tr><tr><td> $\mathrm { V + Q w e n - G \mathrm { - } F l a s h }$ </td><td>-0.87</td><td>[-5.38, 3.30]</td></tr><tr><td> $\mathrm { G } { \cdot } \mathrm { A r g o n } - \mathrm { G } { \cdot } \mathrm { F l a s h }$ </td><td>+12.23</td><td>[4.70, 21.40]</td></tr></table>

Table 17: Family-clustered 95% confidence intervals for all seven main systems and their 21 pairwise differences on the fixed 25-recording cohort (16 families; 20,000 bootstrap draws). F/V denote FunASR/VibeVoice; G-Argon/G-Flash denote Gemini-4-Argon/Gemini-3.8-Flash. System estimates are percentages; differences and their intervals are percentage points. These are pointwise, not simultaneous, intervals. Main-system significance is determined by the 21-contrast Holm correction in Table 16, not by interval exclusion of zero.

The identifier diagnostic enumerates one globally consistent, one-to-one mapping per recording; it neither merges speakers nor changes them independently per event. It enumerates only when the numbers of distinct gold/predicted event speakers agree and are at most six, otherwise retaining original IDs. All other constraints remain. One FunASR+Astra recording and four FunASR+Kimi recordings improve; the other three systems have no improvement. This diagnoses numbering convention, not diarization accuracy.

## G.4 Local Evidence and Trajectory Changes

The same-route Kimi comparison joins conditions by recording and gold-trajectory coordinate, not prediction order. It covers 26 FunASR recordings/308 trajectories and 22 VibeVoice recordings/255 trajectories. We retain lost and gained matches as well as stable outcomes, rather than selecting only degradations.

Local diagnostics reuse frozen B2 anchors linked by event ID and require the anchor core to overlap the revised gold window. An absence signal means a normalized anchor form occurs in original local context but not B2 context; normalization covers explicit numeric spelling, punctuation, and unit forms. FunASR provides native word times, whereas VibeVoice uses overlapping segment text. All 26 FunASR lost trajectories and 13 of 16 VibeVoice lost trajectories contain such a signal. It also occurs in eleven and one gained trajectories, respectively. It is therefore neither a sufficient cause nor a manually verified ASR error rate.

The wrong amount and fee appear directly in both noisy input and settlement predictions; the third case omits a term within a composite value. Other context, segmentation, and downstream inference can change concurrently, and both recovered and lost matches occur. These examples locate specific observed errors, not a claim that every score difference is caused by one word.

<table><tr><td>Model</td><td>Labeled 95% CI</td><td>Unlabeled 95% CI</td><td>Difference 95% CI</td><td>PHolm</td></tr><tr><td>Astra</td><td>[26.06, 52.53]</td><td>[16.77, 42.44]</td><td>[4.18, 15.14]</td><td>0.019</td></tr><tr><td>Kimi-K3</td><td>[13.44, 34.36]</td><td>[16.21, 31.47]</td><td>[-5.90, 4.24]</td><td>0.766</td></tr><tr><td>Qwen3.8-Max</td><td>[16.36, 21.23]</td><td>[7.42, 20.30]</td><td>[-1.11, 11.20]</td><td>0.285</td></tr><tr><td>DeepSeek V4 Flash</td><td>[9.31, 19.02]</td><td>[3.25, 11.60]</td><td>[3.05, 10.56]</td><td>0.019</td></tr></table>

Table 18: Family-clustered intervals and two-sided paired tests for 26 scripts in 16 families. Differences are labeled minus unlabeled, in percentage points. Holm correction jointly covers the four within-model label contrasts.

<table><tr><td>Kimi-K3 comparison</td><td>N</td><td>Families</td><td>Difference</td><td>95% CI</td><td>PHolm</td></tr><tr><td>FunASR: B2 – Original</td><td>26</td><td>16</td><td>-2.04</td><td>[-8.15, 2.40]</td><td>0.625</td></tr><tr><td>VibeVoice: B2 – Original</td><td>22</td><td>16</td><td>-3.32</td><td>[-5.51, -0.96]</td><td>0.039</td></tr><tr><td>FunASR: A1 – Original</td><td>26</td><td>16</td><td>-4.50</td><td>[-9.08, -0.81]</td><td>0.047</td></tr><tr><td>FunASR: A2 – Original</td><td>26</td><td>16</td><td>-2.22</td><td>[-6.05, 0.85]</td><td>0.562</td></tr><tr><td>VibeVoice: A1 – Original</td><td>23</td><td>15</td><td>+1.13</td><td>[-1.71, 4.72]</td><td>0.609</td></tr><tr><td>FunASR A1: Fixed – Automatic</td><td>26</td><td>16</td><td>+1.82</td><td>[-0.11, 4.32]</td><td>0.469</td></tr><tr><td>FunASR A2: Fixed – Automatic</td><td>26</td><td>16</td><td>-1.25</td><td>[-3.34, 0.00]</td><td>0.625</td></tr><tr><td>VibeVoice A1: Fixed – Automatic</td><td>23</td><td>15</td><td>+1.24</td><td>[-0.48, 3.31]</td><td>0.625</td></tr></table>

Table 19: Paired acoustic and automatic-speaker controls. Holm correction is separate for two B2 comparisons, three historical A comparisons, and three speaker controls. Only jointly completed recordings enter each contrast. Differences and intervals are percentage points. Fixed speaker labels are still automatic.

## H Astra Error Distributions on Reference Scripts

Table 30 reports Astra’s error distributions under the two reference-script conditions. The analysis covers all originally unmatched gold trajectories: 175 with speaker labels and 208 without speaker labels, out of 308 gold trajectories across 26 scripts in each condition. Counts refer to trajectories involving each error type, rather than the number of erroneous nodes.

Content and scope errors are the two most frequent categories in both conditions. Speaker errors affect 38 unmatched trajectories (21.71%) with labels and 72 (34.62%) without labels. Missing required nodes and erroneous extra nodes also occur in both conditions, whereas a cycle-count mismatch appears in only one labeled-script trajectory. Since these categories can co-occur and the two failure sets differ, the percentages describe the composition of each set, not mutually exclusive causes or a paired causal decomposition of the score difference.

<table><tr><td>System / script input</td><td>Fields</td><td>Value recall (%)</td><td>Chain recall (%)</td><td>Value-only hits (%)</td></tr><tr><td>FunASR + Astra</td><td>266</td><td>62.41</td><td>30.45</td><td>51.20</td></tr><tr><td>FunASR + Kimi-K3</td><td>266</td><td>52.63</td><td>14.29</td><td>72.86</td></tr><tr><td>FunASR + Qwen3.8-Max</td><td>266</td><td>32.71</td><td>6.77</td><td>79.31</td></tr><tr><td>VibeVoice + Kimi-K3</td><td>266</td><td>51.50</td><td>14.66</td><td>71.53</td></tr><tr><td>VibeVoice + Qwen3.8-Max</td><td>266</td><td>35.71</td><td>5.26</td><td>85.26</td></tr><tr><td>Astra (labeled)</td><td>280</td><td>68.93</td><td>46.43</td><td>32.64</td></tr><tr><td>Kimi-K3 (labeled)</td><td>280</td><td>51.07</td><td>27.86</td><td>45.45</td></tr><tr><td>Qwen3.8-Max (labeled)</td><td>280</td><td>51.79</td><td>19.29</td><td>62.76</td></tr><tr><td>DeepSeek V4 Flash (labeled)</td><td>280</td><td>45.00</td><td>14.64</td><td>67.46</td></tr><tr><td>Astra (unlabeled)</td><td>280</td><td>66.07</td><td>34.64</td><td>47.57</td></tr><tr><td>Kimi-K3 (unlabeled)</td><td>280</td><td>58.57</td><td>27.14</td><td>53.66</td></tr><tr><td>Qwen3.8-Max (unlabeled)</td><td>280</td><td>42.86</td><td>14.29</td><td>66.67</td></tr><tr><td>DeepSeek V4 Flash (unlabeled)</td><td>280</td><td>28.57</td><td>6.43</td><td>77.50</td></tr></table>

Table 20: Terminal-value projections versus full trajectories from the same predictions. Both recall denominators are gold fields with non-null terminal values: 266 fields for the common 25 audio recordings, 280 for all 26 scripts. The last column divides value-correct but chain-incorrect fields by all value-correct fields. Full-chain recall is not macro-F1 DCIST.

<table><tr><td>System / script input</td><td>Terminal-value recall 95% CI</td><td>Full-chain recall 95% CI</td></tr><tr><td>FunASR + Astra</td><td>[47.77, 74.80]</td><td>[13.08, 45.08]</td></tr><tr><td>FunASR + Kimi-K3</td><td>[37.96, 64.91]</td><td>[4.09, 25.82]</td></tr><tr><td>FunASR + Qwen3.8-Max</td><td>[20.93, 44.24]</td><td>[1.90, 11.30]</td></tr><tr><td>VibeVoice + Kimi-K3</td><td>[32.14, 66.99]</td><td>[5.29, 25.25]</td></tr><tr><td>VibeVoice + Qwen3.8-Max</td><td>[25.16, 45.87]</td><td>[2.17, 8.66]</td></tr><tr><td>Astra (labeled)</td><td>[55.00, 79.80]</td><td>[30.70, 60.36]</td></tr><tr><td>Kimi-K3 (labeled)</td><td>[31.17, 67.59]</td><td>[13.66, 43.20]</td></tr><tr><td>Qwen3.8-Max (labeled)</td><td>[43.03, 60.11]</td><td>[15.84, 23.20]</td></tr><tr><td>DeepSeek V4 Flash (labeled)</td><td>[27.82, 60.12]</td><td>[9.09, 19.93]</td></tr><tr><td>Astra (unlabeled)</td><td>[54.55, 75.30]</td><td>[20.92, 49.50]</td></tr><tr><td>Kimi-K3 (unlabeled)</td><td>[45.00, 69.37]</td><td>[16.05, 38.46]</td></tr><tr><td>Qwen3.8-Max (unlabeled)</td><td>[31.06, 52.90]</td><td>[8.11, 22.28]</td></tr><tr><td>DeepSeek V4 Flash (unlabeled)</td><td>[17.36, 40.45]</td><td>[2.72, 11.71]</td></tr></table>

Table 21: Family-clustered 95% confidence intervals for terminal-value and full-chain recall on the same non-nullterminal-value subsets as Table 20.
<table><tr><td>System / script input</td><td>Field pairs</td><td>Both correct</td><td>Joint (%)</td><td>95% CI</td><td>Changed subset</td></tr><tr><td>FunASR + Astra</td><td>87</td><td>21</td><td>24.14</td><td>[0.00, 41.18]</td><td>3/35</td></tr><tr><td>FunASR + Kimi-K3</td><td>87</td><td>13</td><td>14.94</td><td>[0.00, 29.47]</td><td>0/35</td></tr><tr><td>FunASR + Qwen3.8-Max</td><td>87</td><td>3</td><td>3.45</td><td>[0.00, 7.81]</td><td>1/35</td></tr><tr><td>Vibe Voice + Kimi-K3</td><td>87</td><td>5</td><td>5.75</td><td>[0.00, 13.13]</td><td>0/35</td></tr><tr><td>VibeVoice + Qwen3.8-Max</td><td>87</td><td>1</td><td>1.15</td><td>[0.00, 4.00]</td><td>0/35</td></tr><tr><td>Astra (labeled)</td><td>102</td><td>41</td><td>40.20</td><td>[13.89, 61.07]</td><td>14/40</td></tr><tr><td>Kimi-K3 (labeled)</td><td>102</td><td>23</td><td>22.55</td><td>[7.87, 37.17]</td><td>2/40</td></tr><tr><td>Qwen3.8-Max (labeled)</td><td>102</td><td>12</td><td>11.76</td><td>[6.15, 17.58]</td><td>0/40</td></tr><tr><td>DeepSeek V4 Flash (labeled)</td><td>102</td><td>14</td><td>13.73</td><td>[7.14, 19.61]</td><td>2/40</td></tr><tr><td>Astra (unlabeled)</td><td>102</td><td>26</td><td>25.49</td><td>[6.15, 43.59]</td><td>4/40</td></tr><tr><td>Kimi-K3 (unlabeled)</td><td>102</td><td>20</td><td>19.61</td><td>[8.14, 31.30]</td><td>1/40</td></tr><tr><td>Qwen3.8-Max (unlabeled)</td><td>102</td><td>12</td><td>11.76</td><td>[5.48, 18.31]</td><td>0/40</td></tr><tr><td>DeepSeek V4 Flash (unlabeled)</td><td>102</td><td>6</td><td>5.88</td><td>[1.31, 13.25]</td><td>0/40</td></tr></table>

Table 22: A/B joint recovery. Audio systems share nine complete families and 87 field pairs; scripts cover ten families and 102 pairs. A pair succeeds only if both trajectories match. The changed subset has differing gold structure or attributes after ignoring evidence times and renamable IDs (35 and 40 pairs). Intervals resample paired families; fields unique to one variant are excluded.

<table><tr><td>Stratum</td><td>Audio</td><td>Tracks</td><td>F+A</td><td>F+K</td><td>F+Q</td><td>V+K</td><td>V+Q</td></tr><tr><td>2 speakers</td><td>5</td><td>40</td><td>20.00</td><td>6.98</td><td>12.00</td><td>12.68</td><td>10.68</td></tr><tr><td>3 speakers</td><td>13</td><td>163</td><td>20.24</td><td>5.68</td><td>3.64</td><td>8.77</td><td>4.99</td></tr><tr><td>4 speakers</td><td>7</td><td>90</td><td>24.36</td><td>19.46</td><td>5.21</td><td>14.25</td><td>3.57</td></tr><tr><td>&lt; 15 min</td><td>9</td><td>68</td><td>5.84</td><td>1.39</td><td>2.22</td><td>3.02</td><td>5.25</td></tr><tr><td>15-&lt; 30 min</td><td>12</td><td>163</td><td>34.88</td><td>18.21</td><td>9.25</td><td>19.25</td><td>6.98</td></tr><tr><td>≥ 30 min</td><td>4</td><td>62</td><td>15.62</td><td>3.49</td><td>3.23</td><td>4.76</td><td>3.04</td></tr></table>

Table 23: DCIST (%) by speaker count and duration on the same 25 recordings. F/V denote FunASR/VibeVoice; A/K/Q denote Astra/Kimi-K3/Qwen3.8-Max. Each stratum macro-averages recording F1. Counts refer to script roles and full audio duration; the middle duration bin excludes 30 minutes.

<table><tr><td>Nodes</td><td>Tracks</td><td>F+A</td><td>F+K</td><td>F+Q</td><td>V+K</td><td>V+Q</td></tr><tr><td>1</td><td>131</td><td>56/131 (42.75)</td><td>31/131 (23.66)</td><td>15/131 (11.45)</td><td>29/131 (22.14)</td><td>10/131 (7.63)</td></tr><tr><td>2</td><td>103</td><td>18/103 (17.48)</td><td>6/103 (5.83)</td><td>1/103 (0.97)</td><td>9/103 (8.74)</td><td>2/103 (1.94)</td></tr><tr><td>3+</td><td>59</td><td>8/59 (13.56)</td><td>1/59 (1.69)</td><td>2/59 (3.39)</td><td>1/59 (1.69)</td><td>3/59 (5.08)</td></tr></table>

Table 24: Full-chain recall by required gold-node count: hits/trajectories (%). F/V denote FunASR/VibeVoice; A/K/Q denote Astra/Kimi-K3/Qwen3.8-Max. These are gold-trajectory recall rates, not stratified DCIST.

<table><tr><td>System / script input</td><td>N</td><td>Macro F1</td><td>Micro F1</td><td>All nodes</td><td>Exempt nodes</td><td>Extra (%)</td></tr><tr><td>FunASR + Astra</td><td>25</td><td>21.35</td><td>27.06</td><td>732</td><td>193</td><td>26.37</td></tr><tr><td>FunASR + Kimi-K3</td><td>25</td><td>9.80</td><td>13.36</td><td>483</td><td>73</td><td>15.11</td></tr><tr><td>FunASR + Qwen</td><td>25</td><td>5.75</td><td>6.36</td><td>476</td><td>50</td><td>10.50</td></tr><tr><td>VibeVoice + Kimi-K3</td><td>25</td><td>11.09</td><td>13.66</td><td>493</td><td>76</td><td>15.42</td></tr><tr><td>VibeVoice + Qwen</td><td>25</td><td>5.73</td><td>5.23</td><td>491</td><td>66</td><td>13.44</td></tr><tr><td>Astra / labeled</td><td>26</td><td>38.72</td><td>43.18</td><td>744</td><td>203</td><td>27.28</td></tr><tr><td>Kimi-K3 / labeled</td><td>26</td><td>22.72</td><td>26.71</td><td>495</td><td>89</td><td>17.98</td></tr><tr><td>Qwen / labeled</td><td>26</td><td>18.89</td><td>18.77</td><td>524</td><td>114</td><td>21.76</td></tr><tr><td>DeepSeek / labeled</td><td>26</td><td>14.04</td><td>13.99</td><td>512</td><td>90</td><td>17.58</td></tr><tr><td>Astra / unlabeled</td><td>26</td><td>28.99</td><td>32.21</td><td>755</td><td>198</td><td>26.23</td></tr><tr><td>Kimi-K3 / unlabeled</td><td>26</td><td>23.58</td><td>25.75</td><td>554</td><td>108</td><td>19.49</td></tr><tr><td>Qwen / unlabeled</td><td>26</td><td>13.79</td><td>13.05</td><td>531</td><td>101</td><td>19.02</td></tr><tr><td>DeepSeek / unlabeled</td><td>26</td><td>7.31</td><td>6.01</td><td>478</td><td>85</td><td>17.78</td></tr></table>

Table 25: Auxiliary metrics for the common audio cohort (25 recordings) and scripts (26 per condition). Macro F1 is DCIST. F1 and extra-node proportions are percentages. Node denominators include parseable outputs; no nodes are invented for unparseable answers, which remain subject to the format-error rule.

<table><tr><td>Field / evidence</td><td>Original input and prediction</td><td>B2 input and prediction</td><td>Located mismatch</td></tr><tr><td>Financing amount; R0023; 674.97–680.23 s</td><td>“Set financing at 8.4 million&quot;; 9.0M start, 8.4M settlement</td><td>“Set financing at 2.4 million&quot;; 9.0M start, 2.4M settlement</td><td>Wrong settlement value; other required attributes, edges, and time pass</td></tr><tr><td>Service fee; R0023; 713.93–720.30 s</td><td>“Set the fee rate at 0.8%”; 1% start, 0.8% settlement</td><td>“Set the fee rate at 1.8%”; 1% start, 1.8% settlement</td><td>Wrong settlement value de- spite matching remaining re- quirements</td></tr><tr><td>Requested amount and 267.55 s</td><td>“Five million, six months&quot;; term; R0013; 263.43- prediction contains both</td><td>Corrupted phrase correspond- ing to “five million user ap- ply for him&quot;; prediction re- tains 5M, omits six months</td><td>Missing term in a composite value; correct amount alone is insufficient</td></tr></table>

Table 26: Traceable Kimi-K3 examples, translated into English for presentation. The first two use VibeVoice, the third FunASR. Times are intervals cited by the original predictions. Each original trajectory matches and its B2 counterpart fails. Actual input errors are preserved rather than replaced with scripted text. M denotes million CNY.

<table><tr><td>System</td><td>50%</td><td>80%</td><td>90%</td><td>No time</td><td>Bidirectional</td><td>ID permutation</td></tr><tr><td> $\mathrm { F u n A S R + A s t r a }$ </td><td>21.74</td><td>21.35</td><td>20.59</td><td>23.43</td><td>1.24</td><td>21.77</td></tr><tr><td> $\mathrm { F u n A S R } + \mathrm { K i m i - K } 3$ </td><td>9.80</td><td>9.80</td><td>9.80</td><td>10.27</td><td>0.00</td><td>11.65</td></tr><tr><td> $\mathrm { F u n A S R } + \mathrm { Q w e n }$ </td><td>5.95</td><td>5.75</td><td>5.75</td><td>5.95</td><td>0.00</td><td>5.75</td></tr><tr><td> $\mathrm { V i b e V o i c e } + \mathrm { K i m i } { - } \mathrm { K } 3$ </td><td>11.09</td><td>11.09</td><td>11.09</td><td>12.14</td><td>3.99</td><td>11.09</td></tr><tr><td> ${ \mathrm { V i b e V o i c e } } + { \mathrm { Q w e n } }$ </td><td>5.73</td><td>5.73</td><td>5.73</td><td>5.73</td><td>0.21</td><td>5.73</td></tr></table>

Table 27: DCIST (%) under frozen-prediction diagnostics. The first three thresholds divide overlap by predicted duration. Bidirectional coverage additionally requires 50% of the gold window at the 80% threshold. ID permutation is globally consistent per recording. Changes are tested separately, retaining semantic and extra-node judgments; 80% is the official score.

<table><tr><td>System</td><td>Valid windows</td><td>At most 1 s</td><td>Median (s)</td><td>Matched nodes</td><td>Median ratio (%)</td></tr><tr><td> $\mathrm { F u n A S R } + \mathrm { A s t r a }$ </td><td>732</td><td>3</td><td>6.73</td><td>118</td><td>25.36</td></tr><tr><td> $\mathrm { F u n A S R } + \mathrm { K i m i - K } 3$ </td><td>483</td><td>2</td><td>5.10</td><td>46</td><td>15.54</td></tr><tr><td>Fur  $\mathrm { \Delta \mathrm { A S R } } + \mathrm { Q w e n }$ </td><td>476</td><td>1</td><td>5.06</td><td>23</td><td>21.54</td></tr><tr><td> $\mathrm { V i b e V o i c e } + \mathrm { K i m i } { - } \mathrm { K } 3$ </td><td>493</td><td>0</td><td>10.20</td><td>50</td><td>48.56</td></tr><tr><td> ${ \mathrm { V i b e V o i c e } } + { \mathrm { Q w e n } }$ </td><td>489</td><td>1</td><td>10.04</td><td>23</td><td>29.51</td></tr></table>

Table 28: Predicted evidence-window distribution. The duration ratio is predicted/gold window length (%) for required nodes in fully matched trajectories. Valid windows have non-negative starts and positive duration.

<table><tr><td>Frontend</td><td>Audio</td><td>Tracks</td><td>Retained</td><td>Lost</td><td>Gained</td><td>Neither</td><td>Lost with absent anchor</td></tr><tr><td>FunASR</td><td>26</td><td>308</td><td>13</td><td>26</td><td>13</td><td>256</td><td>26/26</td></tr><tr><td>VibeVoice</td><td>22</td><td>255</td><td>12</td><td>16</td><td>7</td><td>220</td><td>13/16</td></tr></table>

Table 29: Paired Kimi-K3 trajectory outcomes for the same-route baseline and B2. Lost means baseline matched and B2 did not; gained is the converse. The final column counts lost trajectories with at least one linked normalized anchor form present before but absent after noise, not proven single-word causes.

<table><tr><td>Error type</td><td>Criterion</td><td>With speaker labels</td><td>Without speaker labels</td></tr><tr><td>Action error</td><td>start / advance / hold / settle differs from GT</td><td>39 (22.29%)</td><td>45 (21.63%)</td></tr><tr><td>Content error</td><td>A number, term, state, or other content in value differs from GT</td><td>94 (53.71%)</td><td>100 (48.08%)</td></tr><tr><td>Speaker error</td><td>speaker_id differs from GT</td><td>38 (21.71%)</td><td>72 (34.62%)</td></tr><tr><td>Scope error</td><td>scope_id differs from GT</td><td>70 (40.00%)</td><td>92 (44.23%)</td></tr><tr><td>Settlement-result error</td><td>result is missing, extra, or has an incorrect value</td><td>18 (10.29%)</td><td>28 (13.46%)</td></tr><tr><td>Missing node</td><td>A required GT event node is not recovered</td><td>33 (18.86%)</td><td>42 (20.19%)</td></tr><tr><td>Erroneous extra node</td><td>An extra output node was judged incorrect in the existing adjudication</td><td>32 (18.29%)</td><td>43 (20.67%)</td></tr><tr><td>Cycle-segmentation error</td><td>The number of decision cycles differs from GT</td><td>1 (0.57%)</td><td>0 (0.00%)</td></tr><tr><td>Unaligned field / missing entire trajectory</td><td>No corresponding predicted trajectory is found using the current fields and existing mappings</td><td>12 (6.86%)</td><td>10 (4.81%)</td></tr></table>

Table 30: Astra’s trajectory error distributions under the two reference-script conditions. Entries are the number (percentage) of trajectories involving each error, among 175 unmatched trajectories with speaker labels and 208 without speaker labels.

Notes. Each input condition contains 26 scripts and 308 GT trajectories. GT is the reference, and the analysis retains monetary equivalence, existing synonym mappings, and extra-node exemption rules; exempt nodes are not counted as errors. One trajectory may involve multiple error types, so percentages can sum to more than 100%. Classification first preserves correctly matched nodes, then performs ordered alignment of the unmatched nodes between them. This table covers the originally un matched trajectories, not the residual trajectories after hypothetically correcting five attribute categories.