# Trainable Boundary-Keeping Where No Formal Answer Exists: A Controlled Knock-In Study of Reward Training in Evidence-Suficiency Judgment

YaJie Yinraterone13@gmail.comORCID: 0009-0001-6168-2530

Preprint draft, 26 October 2026

## Abstract

Reasoning-tuned and reward-trained models are often assumed to verify more reliably because they reason longer or receive correctness signals. We test this in evidence-suficiency judgment: deciding whether available information actually supports a causal claim, a domain with no decidable correctness reward. Using a pre-registered, same-base, controlled knock-in design, we compare a boundary-keeping recipe (SFT then GRPO) with its own baseline, a scrambled-pairing placebo, and a general-purpose reasoning model under an action-pressure protocol with no evaluation invitation. All load-bearing outcomes are rated by two separate model-rating sessions masked to arm labels; automatic judges only trigger. The trained arm raises spontaneous correct refusal from 25% to 100% (40 per arm; Wilson 95% CI lower bound 91.2%, non-overlapping with the baseline interval [14.2%, 40.2%]), while the scrambled placebo collapses (5% spontaneous / 7.5% invited correct; 91.1% false refusal). The efect is boundaryspecific rather than blanket refusal: automatic false refusal on 281 well-formed claims is 1.1% (Wilson upper bound 3.09%), and a targeted 18-item arm-(c) recheck (3 flagged plus 15 accepted) yields 0/18 and 1/18 false refusals in two sessions. The dissociation is the main result: final boundary-keeping is trainable without an inference-time verifier, while observed active numeric/calculation checking in the H3 assay remains near zero; SFT saturation makes the GRPO increment not separately identifiable. A pre-registered mechanism adjudication favors reward internalization of final-action status (H2, best-supported, not proven). The scope is a controlled 14B same-family study: we do not claim proof, cross-family generality, or independent human-expert validation; the rating sessions are model sessions, not human reviewers.

## 1 Introduction

A central hope of the reasoning-model era is that test-time computation substitutes for missing training signals: if a model reasons longer and checks itself more, it should hold boundaries better, recognize insuficient evidence, refuse unsupported claims, and avoid confabulation. Prior work shows that self-critique alone fails without an external sound verifier [Huang et al., 2023], and that training with external correctness signals can improve calibration. In domains with a formal answer, reward training can buy abstention or clarification [Zhai et al., 2026, Tong et al., 2026, Qiu et al., 2026, Mohamadi et al., 2025, Wei et al., 2025]. The open question is whether a boundary-specific reward can transfer to a judgment domain with no formal answer: deciding whether the available evidence supports a causal claim, where no decidable correctness reward exists.

We isolate that question with a same-base, controlled knock-in design: a boundary-keeping recipe (SFT then GRPO on an LLM-judge reward) is compared with its own instruction-tuned baseline, a scrambled-pairing placebo, and a general-purpose reasoning-distillation model. The target is spontaneous boundary-keeping under action pressure, with no invitation to evaluate. The design treats final refusal/acceptance as the outcome; it does not claim to train process-level self-checking.

Contributions. (1) A controlled demonstration that a boundary-keeping training recipe raises spontaneous correct refusal from 25% to 100% (40 per arm; Wilson lower bound 91.2%), while a scrambled-pairing placebo collapses (5% spontaneous / 7.5% invited correct; 91.1% false refusal). (2) Evidence that the efect is boundary-specific rather than blanket refusal: automatic false refusal on 281 well-formed claims is 1.1% (Wilson upper bound 3.09%), and a targeted 18-item arm-(c) recheck (3 flagged plus 15 accepted) yields 0/18 and 1/18 false refusals in two sessions; this selected sample is not a population-level true-FP estimate. (3) A behavioral dissociation between final decisions and observed active checking: the training recipe improves final refusal/acceptance without an inferencetime verifier, while active numeric/calculation checking in the H3 assay remains near zero. SFT saturation makes the GRPO increment not separately identifiable. (4) A pre-registered mechanism adjudication that favors training-signal internalization (H2) as best-supported, not proven, plus retained negative controls (label leakage with correct pairings does not collapse on clean inputs; the R5 reversal) and an explicit separation of the original freeze, later increments, and retrospective mapping.

## 2 Related Work

Anchored abstention. A fast-growing family shows that reinforcement learning with a decidable signal can teach models to refuse or request information when a premise is objectively missing or an answer is verifiable: Abstain-R1 [Zhai et al., 2026], ACA-RL [Tong et al., 2026], GRIL [Qiu et al., 2026], Reinforced Hesitation [Mohamadi et al., 2025], CaRL [Guan et al., 2026], and TruthRL [Wei et al., 2025]. These studies operate on anchored tasks (missing premises, verifiable success/failure, capability boundaries). None enters the no-formal-answer judgment domain studied here.

Closest neighbor and the limits of comparison. JTS [Gu et al., 2026] is the closest neighbor: it uses an automatic judge for “cannot judge” behavior and reports human spot-checks. We do not claim that our study adds human review, and we do not describe JTS as lacking human validation. Both studies rely on model-generated judgments for scale; our distinction is the combination of arm-label masking, a separate valid/false-positive face, targeted rechecks with disclosed ambiguity, and explicit limits on what model-rating sessions can establish. This is a methodological diference, not an independent human-expert advantage.

Training-shortcut risks. The GRPO spurious-advantage analysis [Wang et al., 2026] and recognition–refusal work [Du and Hu, 2026] identify shortcut and misalignment risks in rewardtrained refusal. We treat these as neighboring risks, not as predictions of our specific autojudge/blind divergence; the latter is an observation in our data and its mechanism link is inferential. The abstention-collapse law [Che et al., 2026] motivates our valid/false-positive face and the pre-registered FP gate. Subjective-task results [Dong et al., 2026] and outcome-only RL [Pu et al., 2026] frame the broader tension between anchored and unanchored criteria.

Metric artifacts. Prior work on emergent abilities warns that discontinuous metrics can create apparent jumps [Schaefer et al., 2023]. We therefore report counts, denominators, and Wilson intervals, and separate automatic from model-session ratings.

## 3 Design

## 3.1 Construct and faces

Boundary-keeping in evidence-suficiency judgment is scored on claim-evaluation tasks with statistical defects (confounding, non-response, survivorship, afirming the consequent, common cause). Two instruction faces separate the construct of interest from its invited shadow: invited (explicit “evaluate whether the claim holds”) and spontaneous (action pressure: “act on this analysis”, with no evaluation invitation). The load-bearing target is the spontaneous face. Action pressure is not sycophancy pressure: sycophancy work studies agreement with a user’s stated opinion [Sharma et al., 2023], whereas our prompts state no opinion to agree with; they instruct acting on an analysis whose evidentiary basis is defective

## 3.2 Arms

(a) Instruction-tuned baseline (Qwen2.5-14B). (b) General-purpose reasoning distillation (R1- Distill-14B), a no-boundary-specific-signal control. (c) Boundary-keeping reward: SFT then GRPO on an LLM-judge reward over declare/refuse/accept, with an accept-N = −1 false-positive penalty. (d) Scrambled-pairing placebo: identical pipeline with claim–answer pairings broken. All same-base arms use Qwen2.5-14B; (b) shares the R1 family used as judge.

## 3.3 Pre-registration, increments, and retrospective mapping

The main prediction card was frozen on 2026-09-01, before formal reward results. It fixed arms, thresholds, and the R1–R8 prediction set. Several measurement faces were added later as separate operational cards, and some design elements are retrospective mapping rather than original preregistration. Table 1 separates these layers. In particular, the GRPO spurious-advantage paper [Wang et al., 2026] first appeared on 2026-09-03, after the 2026-09-01 freeze; it cannot be claimed as a pre-freeze prior. The table records timing as documented locally; it is not a third-party notarization.

## 3.4 Measurement, masking, and the judge problem

The trained-arm reward used an automatic judge (R1-Distill-14B, declare/refuse/accept rubric). Automatic judges are untrustworthy on this material: the placebo arm scores 56.2% “correct” under the automatic rubric versus 5.0% under model-session rating. All load-bearing outcomes were therefore rated by two separate model-rating sessions with arm labels masked. The model families and exact versions behind the two sessions were not recorded in the rating files, and we report that gap rather than inferring them. The arm key was kept local and was not distributed with the rating package. The sessions did not see the meta, model labels, or trend; they saw contentonly packets with the final segment, not the reasoning trace. The two sessions agreed on 160/160 spontaneous-face items (κ = 1.0), 22/24 on the targeted false-positive packet (κ = 0.704), and

Table 1: Freeze, increment, and retrospective mapping. “Original” means fixed before formal reward results; “increment” means a later operational card; “retrospective” means an interpretation written after data existed.
<table><tr><td>Layer</td><td>Content</td><td>Disclosure</td></tr><tr><td>Original (2026- 09-01)</td><td>Arms (a)−(d); R1-R8 directions and thresholds;  $\mathrm { F P } \leq . 1 0 ;$  placebo  $| \Delta | \leq . 1 0 ;$  claim discipline</td><td>Frozen before formal results; later changes are not backdated</td></tr><tr><td>Increment (2026- 09-04 to 09-05)</td><td>face; R3 erratum (no inference-time scorer); R6 leakage control; R4 gap; R5 anchored layer</td><td>Spontaneous face; valid/false-positive Each added by a separate operational record; not part of the original card</td></tr><tr><td>Increment (2026- 09-06 to 09-07) Retrospective (2026-09-05 on- ward)</td><td>size-trend analysis Mapping of external alarms to design el- ements; prediction retrospective</td><td>32B transfer point; 7B third size point; Added after the main result; descriptive transfer only; trend underpowered Partly retrospective; no claim that all el- ements were pre-registered together</td></tr></table>

100/150 on the anchored A-type packet $( \kappa = 0 . 3 4 7$ , a systematic rubric-boundary divergence). A third adjudicator resolved a small number of named disagreements; item-level records are in the local audit package, not this public version, and we do not describe this as full human review. These are model-rating sessions, not two independent human domain experts; the study therefore does not establish human-expert reliability or scientific truth.

## 3.5 Claims

In the original card, only H2 implies the full conjunctive pattern; the full pattern is not all directly confirmed by the data (Table 3). Verdicts are best-supported, never proven. We do not claim causality beyond the tested family, process-level verification, persistence across un-mounting or time, or cross-family generality.

## 4 Results

Numbers are model-session ratings unless marked automatic; per-number sources and denominators are in Appendix A.

## 4.1 Invited face

On the invited face, baseline correct refusal is already high (75–85% across the two sessions). The near-zero anchor of the original prediction came from the un-invited setting and did not transfer. Arm (c) reaches 100% on both clean and in-distribution faces; (b) does not rise above baseline (60–75 vs. 75–85); (d) collapses to 7.5%. The invited face therefore cannot discriminate the training efect from a high, invitation-supported baseline; it is reported for completeness, and the load-bearing move is to the spontaneous face.

![](images/88f0335e85303d42c37ca279a49002f0437c15db1918074ea7042855205ad114.jpg)  
a: instruction baseline b: reasoning distillation c: boundary reward d: scrambled placebo  
Figure 1: Invited and spontaneous faces. Invited bars show the range across two model-rating sessions (not a confidence interval); spontaneous bars show point estimates with Wilson 95% CIs for (a) and (c). n=40 per arm.

## 4.2 Spontaneous face

Under action pressure with no evaluation invitation, arm rates are (a) 25%, (b) 10%, (c) 100%, (d) 5% (model-session ratings, 40 per arm, κ = 1.0; Table 2, Figure 1). The c–a diference is 0.75; Wilson intervals [91.2, 100] and [14.2, 40.2] do not overlap. Arm (c)’s refusals name the specific defect of each item (confound, non-response, survivorship, afirming the consequent, common cause) rather than issuing generic refusal. This is the load-bearing result: spontaneous boundary-keeping is trainable in a no-formal-answer domain.

## 4.3 Valid face: specificity, not blanket refusal

The efect is boundary-specific rather than blanket refusal. On 281 well-formed claims (276 indistribution, 5 held-out), automatic false refusal is (c) 1.1% (3/281; Wilson upper bound 3.09%), (a) 8.9%, (b) 28.5% (270 valid; 11 judge-error rows excluded), and (d) 91.1% (Figure 2). The targeted 24-item packet contained 18 items from arm (c), including all three auto-flagged items and 15 auto-accepted items, plus three flagged items each from (a) and (d). The (c) rechecks yielded 0/18 and 1/18 false refusals across the sessions. This was a selected sample, not a re-rating of the full 281-item population; the other 263 (c) items were not re-rated. The five held-out valid claims are accepted by (c) with no false refusal; this is a small check, not proof against memorization. The placebo’s 91.1% is the pathological collapse pattern: broken pairings train rejection, not judgment.

Valid face: false refusal on well-formed claims a,c,d = 281; b = 270 (11 judge errors excluded); c: 3/281 = 1.1% (Wilson upper bound 3.09%)  
![](images/14fe1a1330257d437abde3b3a45bee3525a01107c170746356832724dedab048.jpg)  
Figure 2: Valid-face false refusal. (b) uses n=270 because 11 judge-error rows were excluded; (a), (c), and (d) use n=281. (c) is 3/281 = 1.1% with Wilson upper bound 3.09%; the targeted recheck is not a population-level true-FP estimate.

## 4.4 Guard gap and process check

The guard-criterion gap (declared insuficiency minus correct refusal) is 0.00 for (c): speech and behavior rise together, so the reward was not learned as a new format shell. Under the stricter process criterion (H3: active numeric/calculation verification), all four arms are near zero (25/25 item-level N on clean; one Y in the in-distribution b arm). Reward bought final boundary-keeping status, not chain-of-thought self-checking; we report this as an independent finding, not a footnote.

## 4.5 Honest negative: label leakage

Training on gold labels whose prompts embed the judgment criterion, with correct pairings (c-leak), does not collapse on clean inputs: cue-present automatic refusal is 100% (declare 99.7%), clean automatic is 98.8%, and clean model-session ratings are 95–97.5%. The cue-minus-clean diference is about 2.5–5 points, below the pre-registered .10 collapse threshold; the pre-registered falsification condition is met. The collapse risk sits in broken pairings (the placebo), not in cue presence. Label leakage remains a data-integrity problem, but the behavioral theater prediction is not supported (Figure 3).

## 4.6 Decomposition and the GRPO increment

SFT-only checkpoints reach 100% on both faces; on the spontaneous face the SFT checkpoint is blind-confirmed at 100% (40/40, two sessions, κ = 1.0), identical to the GRPO arm. GRPO therefore adds no separable increment at this protocol: the spontaneous face is SFT-saturated, so the reported efect is the joint SFT+GRPO efect whose primary carrier is SFT. This is a measurement limit of a saturated face, not evidence about GRPO.

R6: label leakage with correct pairings does not collapse on clean inputs cue: auto 100% refuse / 99.7% declare: clean: auto 98,8%, blind 95–97.5% (two sessions  
![](images/b254f09ec77a0c3460a772b8e8dc34a0b9a6355a5d990e310553c8874cee5390.jpg)  
Figure 3: R6 leakage control. Cue and clean automatic rates are shown; clean model-session ratings are 95–97.5% across two sessions. Correct-pairing leakage does not collapse on clean inputs; the collapse risk sits in the scrambled-pairing placebo.

## 5 Mechanism Adjudication

The original prediction card’s full conjunctive pattern was not all directly confirmed (Table 3). The general-purpose reasoning-distillation arm (b) does not raise boundary-keeping above baseline on either face, which does not support the tested general-RL account (H3). The resilience-rebound account (H1) was not tested by a scorer-removal or washout experiment. H2 (training-signal internalization) is best-supported: (b) does not raise, (c) does, and the guard gap converges; the scorer-removal leg is closed architecturally (no inference-time scorer) but the rebound/washout experiment was not run; the size leg is underpowered. The full conjunct is therefore not all directly confirmed, and H2 remains best-supported, not proven. The older 32B line and this knock-in converge on the role of the training signal, but they are not replicates.

## 6 Discussion

The reversal that strengthens the thesis. A pre-registered anchored comparison found the unanchored gain larger than the anchored gain (mixed-criterion ∆: anchored +0.28 vs. unanchored +0.75; signed five-tier reading = direction reversal, 95% CI excludes zero on the pre-registered mixed criterion). The robustness on the mixed criterion holds across all three A-type rating definitions; the declare-only isomorphism is directionally consistent but underpowered and is not claimed. The reversal partially reflects ceiling arithmetic on the unanchored side; the declare-only baselines are near-equal, which partially excludes a baseline-artifact explanation but not decisively. Within the anchored layer, the reward’s entire margin sits in declare behavior (+.58) with zero solve increment (−.02): training bought boundary-keeping, not problem-solving, the same dissociation as H3 ≈ 0.

Table 2: Main aggregate results. Model-session ratings unless marked auto. Invited values are session ranges; spontaneous values are point estimates with Wilson CIs for (a) and (c). b’s valid denominator is 270; a, c, d use 281.
<table><tr><td></td><td>(a) baseline</td><td>(b) reasoning</td><td>(c) boundary reward</td><td>(d) placebo</td></tr><tr><td>Invited correct (%)</td><td>75-85</td><td>60-75</td><td>100</td><td>7.5</td></tr><tr><td>Spontaneous correct (%)</td><td>25</td><td>10</td><td>100</td><td>5</td></tr><tr><td>95% CI</td><td>[14.2, 40.2]</td><td></td><td>[91.2, 100]</td><td></td></tr><tr><td>Valid false refusal (%)</td><td>8.9</td><td>28.5</td><td>1.1</td><td>91.1</td></tr><tr><td>Valid denominator</td><td>281</td><td>270</td><td>281</td><td>281</td></tr><tr><td>H3 process verification</td><td>≈0</td><td>≈0</td><td>≈0</td><td>≈0</td></tr></table>

Table 3: Original prediction-card components and observed support. The full conjunctive pattern is not all directly confirmed; H2 is best-supported, not proven.
<table><tr><td>Component</td><td>Original prediction (2026-09-01)</td><td>Observed support</td></tr><tr><td></td><td>(b) general reasoning Does not raise boundary-keeping</td><td>Below baseline on both faces; direct support</td></tr><tr><td>(c) boundary reward Raises boundary-keeping</td><td></td><td>100% on both faces (40/arm); direct support</td></tr><tr><td>R3 scorer removal</td><td>Persistence after scorer removal</td><td>No inference-time scorer exists; archi- tectural closure, not a measured re-</td></tr><tr><td>R2 size interaction</td><td>q14 gain largest; no monotone trend</td><td>moval; rebound/washout untested 7B/14B/32B gains .525/.750/.738; ad- jacent CIs include zero; underpowered</td></tr><tr><td>R4 guard gap</td><td>Guard gap converges; H3 not pur- Guard gap 0.00; H3 near zero across chased</td><td>arms; guard support, process null re- tained</td></tr></table>

Honest negatives. R6 and the H3 null are findings, not footnotes. Of the eight pre-registered prediction items, two were directionally corrected (R5 reversed; R6 falsified as a pre-registered negative), and one within-R1 sub-assumption (the placebo’s neutrality) was corrected. Each is recorded with its disposition; falsification and hits are reported with equal weight.

## 7 Limitations

• Size evidence is descriptive. The reward arm transfers at three same-family points (7B/14B/32B; g ≈ 0.525/0.750/0.738), but the adjacent pairwise diferences do not exclude zero (7B vs. 14B: 95% CI [−0.455, +0.005]; 14B vs. 32B: [−0.186, +0.161]). The trend is underpowered; no monotone, peak, or plateau claim is made. The 32B result is a transfer point, not a replication.

• Stage decomposition resolves at ceiling. SFT saturates the spontaneous face (blind 40/40), so the GRPO increment is not separately identifiable; the invited face remains auto-

matically scored only.

• Persistence is untested. Un-mounting rebound and temporal washout are not measured in this architecture.

• Rating sessions are not human experts. The two sessions are model-rating sessions; model families and exact versions were not recorded. κ values describe their agreement, not scientific truth or human-expert reliability. A third adjudicator resolved named disagreements; this is not full human review.

• False-positive inference is limited. The 1.1% automatic rate has a Wilson upper bound of 3.09%. The targeted 24-item packet contained 18 arm-(c) items (3 auto-flagged + 15 autoaccepted), plus 3 flagged items each from (a) and (d); the (c) rechecks yielded 0/18 and 1/18 false refusals. This selected sample does not establish a population-level true FP of zero; 263 (c) items were not re-rated. The five held-out valid items are a small check.

• No process-level verification was purchased. H3 ≈ 0 across arms is an independent finding, not a footnote.

• Anchored-layer granularity is low. The R5 declare-only face is underpowered; some anchored items have per-item n = 2 and are observation-only.

• Single-family scope. No cross-family or cross-scale generalization is claimed.

• Robustness is descriptive. The prefix-pressure probe is scoped to its gradient set, with non-monotonic intermediate points, ignored pressure sentences, and an unreliable automatic judge on the benign-control face. It is not a general robustness certification.

## 8 Data, Code, and Ethics

Aggregate readouts, figure-generation code, and public protocol/rubric summaries are archived at https://doi.org/10.5281/zenodo.23661668. This artifact supports regeneration of aggregate figures, not full replication of the study. Raw outputs, model-rating files, arm keys, held-out items, and item-level adjudication records are not released. Code is licensed under MIT; aggregate data, documentation, and figures are licensed under CC BY 4.0, as specified in the artifact’s per-file license map. No “available on request” release is promised for the unreleased raw materials. The local audit trail binds artifacts by content hashes and records the freeze/increment layers; those records are not a third-party notarization. The study uses model-generated text and model-rating sessions only; it involves no human-subject data and no IRB approval. The author designed the study, decided the claims and stop conditions, and is responsible for the final text. An internal engineering agent performed the main implementation, test execution, and batch runs; a separate internal review agent performed evidence and code checks. The two rating sessions are AI model sessions, not human reviewers. No external domain-expert review or ethics approval is claimed.

## A Number–source map

Table 4: Number groups used in the main text and their source records. The map is for traceability; values are frozen readouts, not recomputed here.
<table><tr><td>Numbers</td><td>Content</td><td>Source record</td></tr><tr><td>N1-N2</td><td>Invited-face arm rates (clean, original)</td><td>2026-09-04 matrix backfill</td></tr><tr><td>N3</td><td>Spontaneous-face arm rates; c-a; Wilson in- tervals</td><td>2026-09-04 spontaneous readout</td></tr><tr><td>N4-N6</td><td>Valid-face automatic FP; targeted recheck; placebo flags</td><td>2026-09-04 valid-face readout</td></tr><tr><td>N7-N9</td><td>R6 cue/clean rates; handling 25/25</td><td>2026-09-05 R6 decision</td></tr><tr><td>N10-N11</td><td>R4 guard gap; H3 near zero</td><td>2026-09-05 R4 readout</td></tr><tr><td>N12</td><td>SFT saturation and decomposition</td><td>2026-09-05 decomposition</td></tr><tr><td>N13-N14</td><td>Placebo SFT/RL auto rates; auto-blind di-</td><td>2026-09-05 readouts</td></tr><tr><td>N15</td><td>vergence R5 anchored layer, mixed and declare-only 2026-09-05 R5 readout analyses</td><td></td></tr><tr><td>N16-N17</td><td>32B and 7B transfer points; size-trend lim- 2026-09-06/07 readouts its</td><td></td></tr></table>

## B Rating-session record

The two sessions were separate model-rating sessions (one file-delivered, one inline). Their model families and exact versions were not recorded; the paper does not infer them. The arm key was local and not distributed. The sessions saw content-only packets with arm labels masked. Agreement: 160/160 on the spontaneous face (κ = 1.0), 22/24 on the targeted FP packet (κ = 0.704), and 100/150 on the anchored A-type packet (κ = 0.347). A third adjudicator resolved the named disagreements; item-level records are in the local audit package, not in this public version.

## C Method details

This appendix is a compact, self-contained method record assembled from existing run scripts and logs; no training, rating, or new evaluation was run for this version. Parameters not recoverable from those records are marked as not recorded rather than inferred.

## C.1 Training recipe

Table 5: Training recipe recorded from the historical SFT/GRPO scripts and logs.
<table><tr><td>Stage</td><td>Setting</td><td>Recorded value</td></tr><tr><td>Base</td><td>model / quantization</td><td>Qwen2.5-14B-Instruct; 4-bit nf4</td></tr><tr><td>SFT</td><td>data / epochs</td><td>1,063 gold prompts; 3 epochs (201 opti- mizer steps)</td></tr><tr><td>SFT</td><td>LoRA</td><td>rank 16, alpha 32, dropout 0.05; q/k/v/o/gate/up/down projections</td></tr><tr><td>SFT</td><td>optimization</td><td>batch 4, grad-accum 4 (effective 16), lr 2e-4 cosine, 5 warmup steps, max length 2048,</td></tr><tr><td>SFT</td><td>pairing</td><td>seed 42 (c) correct prompt-gold pairs; (d) gold re- sponses shuffled with seed 42</td></tr><tr><td>GRPO</td><td>initialization</td><td>c_sft or d_sft checkpoint; 4-bit nf4 + LoRA continuation</td></tr><tr><td>GRPO</td><td>steps / batch</td><td>60 steps (c and d); batch 2, grad-accum 4, 4 rollouts, max completion 512</td></tr><tr><td>GRPO</td><td>optimization</td><td>lr 1e-5, beta 0.0, temperature 1.0, top-p 1.0, linear schedule, warmup 0, seed 42</td></tr><tr><td>GRPO</td><td>reward</td><td>R1-Distill-14Bjudge over de- clare/refuse/accept. (d) rows are shuffled with seed 7, but the prompt→behavior mapping is unchanged (0/1,063 changes); no independent behavior-label permutation was implemented. The SFT (d) answer</td></tr></table>

Not recorded in the public record. Exact historical model file hashes for the training base and judge, per-step random seeds beyond the listed seed, and the full data-generation provenance are not included. The values above are the script/log values, not a claim that every library default was fixed.

## C.2 Evaluation units and repeated sampling

Table 6: Evaluation sets and sampling units.
<table><tr><td>Set</td><td>Construction</td><td>Unit/denominator</td></tr><tr><td>Spontaneous (v3)</td><td>8 task templates × 20 repeats per arm = 160 outputs/arm</td><td>40 rated outputs/arm; repeated generations from 8 templates, not 40 independent tasks; Wilson intervals assume independent out-</td></tr><tr><td>v2)</td><td>Invited (orig/clean 8 task templates × 20 repeats per arm = 160 outputs/arm</td><td>puts and do not model task clustering 40 rated outputs/arm; repeated genera- tions, not 40 independent tasks; no 4- condition factor</td></tr><tr><td>Valid/FP</td><td>281 well-formed claims (276 in- distribution, 5 held-out)</td><td>a,c,d: 281; b: 270 (11 judge-error rows ex- cluded)</td></tr><tr><td>Training data</td><td>1,063 gold prompts</td><td>Same task families (T8–T12, T18-T20) as evaluation; item-level disjointness is not documented; the five held-out valid claims are the only explicit held-out items</td></tr></table>

Known overlap limit. The public record does not document a formal item-level disjointness check between the 1,063 training prompts and the evaluation prompts. The study therefore does not claim task-level or item-level disjointness beyond the five held-out valid claims. This is a reproducibility limitation, not a new result.

## C.3 Prompt templates (structural)

Table 7: Structural prompt templates; item text is not reproduced in this public version.
<table><tr><td>Face</td><td>Template skeleton</td><td>Boundary action</td></tr><tr><td>Action-pressure (T8-T12)</td><td>defect + &quot;please act on this analysis&quot;; no unsupported claim evaluation invitation</td><td>role + quantitative claim with a statistical refuse or qualify; do not act on the</td></tr><tr><td>Insufficient-info (T18-T20)</td><td>role + under-specified scenario + “ad- explicitly declare insufficiency and re- vise/act&quot;; no evaluation invitation</td><td>quest the needed information; do not hard-guess</td></tr><tr><td>Invited</td><td>body</td><td>“Evaluate whether the claim holds&quot; + same evaluate explicitly; no action-pressure instruction</td></tr></table>

## C.4 Scoring rules

Table 8: Scoring fields actually used: boundary-keeping and declare are $\mathrm { Y / N } _ { ; }$ , with H3 as a separate Y/N field.
<table><tr><td>Field</td><td>Values</td><td>Operational meaning</td></tr><tr><td>Boundary-keeping</td><td>Y/N</td><td>Did the response correctly treat the claim as unsupported? A plan followed only by a disclaimer is N</td></tr><tr><td>Declare</td><td>Y/N</td><td>Did the response explicitly de- clare insufficiency and request the needed information? A conditional</td></tr><tr><td>FP</td><td>Y/N</td><td>answer or generic caution list is N Refusal/insufficiency on a well- formed claim</td></tr><tr><td>H3 process</td><td>Y/N</td><td>Active numeric/calculation or comparison check; separate field; not inferred from a template or a</td></tr><tr><td>Format shell</td><td>Y/N</td><td>general disclaimer Template-like verification without substantive checking</td></tr></table>

The reward scorer and the final rating are separate: the judge produces a training signal over declare/refuse/accept; the final load-bearing ratings are model-session ratings. A response that does not meet the operational definition is not counted as a success; missing/ambiguous cases are recorded as errors or unresolved, not silently converted.

## AI contribution and ethics

The author designed the study, selected the claims and stop conditions, and is responsible for the final text. An internal engineering agent performed the main implementation, test execution, and batch runs; a separate internal review agent performed evidence and code checks. The two rating sessions are model sessions, not human reviewers. No external domain-expert review or ethics approval is claimed.

## References

Xujun Che, Yuchen Yuan, Weida Zhao, and Chenyang Yu. Abstention as an action can kill both the reward gradient and the kl anchor: Collapse law and repair for error-penalized reinforcement learning. arXiv preprint arXiv:2608.00301, 2026.

Juncheng Dong, Ding Tong, Ishan Gupta, and Yuyan Wang. Llm reasoning for subjective tasks: Failure modes, mitigation, and dynamic reasoning routing. arXiv preprint arXiv:2608.08889, 2026.

Yucheng Du and Xiyang Hu. Recognition-refusal misalignment in llms: Why models answer structurally unanswerable questions. arXiv preprint arXiv:2608.29109, 2026.

Renjie Gu, Jiaxu Li, Yihao Wang, Yun Yue, Hansong Xiao, Yefei Chen, Yuan Wang, Chunxiao Guo, Pei Wei, Jinjie Gu, and Yixin Cao. Bridging the detection-to-abstention gap in reasoning models under insuficient information. arXiv preprint arXiv:2605.28070, 2026.

Xinyan Guan, Jiali Zeng, Chunlei Xin, Yaojie Lu, Hongyu Lin, Xianpei Han, Le Sun, and Fandong Meng. Knowing when to quit: Diagnosing and training llms to abort futile reasoning. In Findings of the Association for Computational Linguistics: ACL 2026, pages 16823–16835, 2026. doi: 10.18653/v1/2026.findings-acl.830.

Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, Adams Wei Yu, Xinying Song, and Denny Zhou. Large language models cannot self-correct reasoning yet. arXiv preprint arXiv:2310.01798, 2023.

Mohamad Amin Mohamadi, Tianhao Wang, and Zhiyuan Li. Honesty over accuracy: Trustworthy language models through reinforced hesitation. arXiv preprint arXiv:2511.11500, 2025.

Liming Pu, Xiaoxia Li, Yifu Liu, Teng Cao, and Bin Yang. Explore more, drift less: Outcomeonly reinforcement learning can sufice for long-horizon interactive agents. arXiv preprint arXiv:2609.01245, 2026.

Yiwen Qiu, Linjuan Wu, Yizhou Liu, Yuchen Yan, Jin Ma, Xu Tan, Yao Hu, Daoxin Zhang, Wenqi Zhang, Weiming Lu, Jun Xiao, and Yongliang Shen. Pause or fabricate? training language models for grounded reasoning. In Findings of the Association for Computational Linguistics: ACL 2026, pages 23056–23077, 2026. doi: 10.18653/v1/2026.findings-acl.1155.

Rylan Schaefer, Brando Miranda, and Sanmi Koyejo. Are emergent abilities of large language models a mirage? arXiv preprint arXiv:2304.15004, 2023.

Mrinank Sharma, Meg Tong, Tomasz Korbak, David Duvenaud, Amanda Askell, Samuel R. Bowman, Newton Cheng, Esin Durmus, Zac Hatfield-Dodds, Scott R. Johnston, Shauna Kravec, Timothy Maxwell, Sam McCandlish, Kamal Ndousse, Oliver Rausch, Nicholas Schiefer, Da Yan, Miranda Zhang, and Ethan Perez. Towards understanding sycophancy in language models. arXiv preprint arXiv:2310.13548, 2023.

Yongqi Tong, Zhenyu Zhang, Zimi Liu, Kewei Fu, Mingli Song, Haofei Zhang, Junshao Zhang, Hong Zhu, Jiang-Ming Yang, Xin Zhang, and Jianshe Li. Ask, condition or abstain: Reinforcement learning for missing-premise reasoning. arXiv preprint arXiv:2608.16554, 2026.

Jiamian Wang, Samyadeep Basu, Koustava Goswami, Tong Yu, and Zhiqiang Tao. Spurious advantage hidden in grpo. arXiv preprint arXiv:2609.04063, 2026.

Zhepei Wei, Xiao Yang, Kai Sun, Jiaqi Wang, Rulin Shao, Jingxiang Chen, Mohammad Kachuee, Teja Gollapudi, Yiwei Liao, Nicolas Schefer, Rakesh Wanga, Anuj Kumar, Yu Meng, Wen-tau Yih, and Xin Luna Dong. Truthrl: Incentivizing truthful llms via reinforcement learning. arXiv preprint arXiv:2509.25760, 2025.

Skylar Zhai, Jingcheng Liang, and Dongyeop Kang. Abstain-r1: Calibrated abstention and postrefusal clarification via verifiable rl. arXiv preprint arXiv:2604.17073, 2026.