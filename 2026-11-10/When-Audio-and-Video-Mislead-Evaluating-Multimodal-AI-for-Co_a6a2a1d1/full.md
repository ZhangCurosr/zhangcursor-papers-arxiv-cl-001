# When Audio and Video Mislead: Evaluating Multimodal AI for Conversational Emotion Recognition

Youngseok Kim, Minjae Kim, Eunsoo Lee, and Jihie Kim<sup>⋆</sup>

Dongguk University, Seoul, Republic of Korea hhon0213@dgu.ac.kr, kim0729@dgu.ac.kr dmstn7432@dgu.ac.kr, jihie.kim@dgu.edu

Abstract. Multimodal Emotion Recognition in Conversation (MERC) combines lexical, acoustic, visual and conversational cues, but accepting audio and video does not establish their efective or robust use. We audit of-the-shelf Omni models while holding the target transcript and prior textual context fixed and adding, removing or replacing target audio and video. In the primary evaluation of Qwen2.5-Omni-7B, Baichuan-Omni-1.5 and MiniCPM-o 2.6 on MELD and the full five-session six-class IEMOCAP set, T+A improves Accuracy in all six model–dataset pairs, with four gains significant after false discovery rate (FDR) correction. Across three source mappings, unrelated A/V (RandA/V) lowers Accuracy relative to Full in all 18 comparisons; all 18 remain significant under dialogue-cluster inference and BH-FDR correction, both on commonvalid pairs and when invalid/OOM outcomes are treated as incorrect. Comparisons with modality-omission baselines further distinguish replacement penalties from mismatch-specific harm. A Qwen/MELD case study finds significantly greater gold-vs-best-candidate margin recovery in Late than Early or Middle under decision-query audio knockout, relative to an edge-count-matched Video-Sham control, using Holm-corrected tests. These results distinguish genuine-modality benefit from sensitivity and vulnerability to misleading input.

Keywords: Conversational emotion recognition · Omni models · Multimodal reasoning · Modality reliance · Audio-visual intervention · Mechanistic analysis

## 1 Introduction

Human emotion in conversation is conveyed through language, prosody, voice quality, facial expression and dialogue context. Multimodal Emotion Recognition in Conversation (MERC) combines these signals, commonly using MELD and IEMOCAP [1,14]. Recent systems model modality-specific evidence and conversational structure [9,15], while Omni models can accept raw or near-raw text, audio and video within an integrated model [10,17]. Yet accepting a modality does not establish its efective use: a model may ignore audio, react strongly without improving accuracy, or let misleading media override an otherwise correct decision.

Existing MERC and Omni-model evaluations demonstrate unequal modality contributions and vulnerability to conflict [2,6,12,16]. Most compare naturally aligned inputs or modality ablations. Ablation measures whether a channel contributes useful information; replacement asks whether unrelated but plausible in-domain content can redirect the same prediction. These questions require matched comparisons with the explicitly supplied target transcript and preceding textual context held fixed. A loss relative to correctly matched multimodal input can reflect removal of useful evidence, whereas a loss relative to modality omission establishes additional mismatch-specific harm.

We introduce a controlled reliance audit with seven conditions: Text, Text+Audio (T+A), Text+Video (T+V), Full, Random Audio (RandA), Random Video (RandV), and Random A/V (RandA/V). Random media come from another dialogue in the same dataset under label-agnostic mappings shared across models. We distinguish utility, the benefit of genuine target modalities; sensitivity, changes in predictions after replacement; and interference, assessed against both matched Full input and omission baselines. Evaluating three source mappings tests whether the findings depend on particular source–target pairings. A separate Qwen/MELD intervention study examines decision-query connections to mismatched audio. Extended methods and complete statistical results are provided in the appendix.

Our contributions are threefold:

1. A matched seven-condition audit that varies target audio and video while preserving the transcript and prior textual dialogue context.

2. Behavioral evidence separating useful-modality contribution, replacement sensitivity and mismatch-specific harm across three Omni models, two datasets and three source mappings.

3. A mechanistic confirmation in new Qwen RandA-Harm cases: decision-query Audio KO yields greater gold-vs-best-candidate margin recovery in Late than Early or Middle blocks relative to Video Sham.

The primary evaluation finds positive T+A Accuracy changes in all six model– dataset pairs, with four significant after FDR correction. Across mappings, all 18 RandA/V-minus-Full comparisons are negative and significant under both common-valid and failed-as-incorrect policies. Efects relative to omission vary across settings. This distinction prevents modality sensitivity from being mistaken for either reliable integration or uniformly harmful mismatch.

## 2 Related Work

Conversational emotion recognition. Speaker-state, memory and graph-based architectures establish the importance of utterance evidence and dialogue structure [4,7,8,11]. Later work examines modality/context disentanglement, afective representations and modality selection [9,15,2]. Such task-specific systems are usually evaluated on naturally aligned inputs. Their ablations quantify useful contributions but do not directly measure how unrelated media changes the same decision with transcript and context fixed.

Omni models and modality conflict. OmniVox evaluates Omni-LLMs across modality and context configurations [12]. CA-MER and EmoMM expose modality biases when emotion-bearing evidence conflicts or is missing [6,16]. Our audit complements these studies through label-agnostic, within-dataset cross-dialogue replacement and paired comparisons with both Full and omission baselines. It targets reliance on the complete audio/video channels rather than isolating acoustic prosody or estimating deployment failure rates.

Mechanistic multimodal analysis. NOTICE studies controlled cross-modal corruption, VisiPruner analyzes layer-wise interaction, and EmoMM steers attention under modality conflict [5,3,16]. Attention patterns can nominate intervention sites but do not themselves establish causal importance. We therefore separate exploratory localization from confirmatory decision-query audio-edge masking in new RandA-Harm cases. The resulting evidence concerns the tested intervention sites and cohort; it does not identify where the original prediction error first arises.

## 3 Controlled Modality-Intervention Framework

## 3.1 Task Formulation

For a target utterance $u _ { i }$ with gold emotion $y _ { i } \in \mathcal { V }$ , let $T _ { i } , A _ { i }$ and $V _ { i }$ denote its transcript, audio and video. The textual context $C _ { i }$ contains speaker information and at most ten preceding utterances; future turns and gold labels are excluded. Full input to an Omni model $f _ { \theta }$ yields

$$
\hat { y } _ { i } ^ { F } = f _ { \theta } ( C _ { i } , T _ { i } , A _ { i } , V _ { i } ) .\tag{1}
$$

We hold $C _ { i }$ and the explicit transcript $T _ { i }$ fixed while varying modality availability or source content.

## 3.2 Evaluation Data and Omni Models

MELD’s oficial test split contains 2,610 utterances and seven classes: anger, disgust, fear, joy, neutral, sadness and surprise [14]. For IEMOCAP, we retain all 7,380 utterances labeled angry, excited, frustrated, happy, neutral or sad across 151 dialogues in Sessions 1–5 [1]. We neither merge these labels nor retain other target annotations.

We evaluate Qwen2.5-Omni-7B, Baichuan-Omni-1.5 and MiniCPM-o 2.6 of the-shelf without task-specific fine-tuning [17,10,13]. Within each model, prompts preserve the context, transcript, label order and output format. Availability instructions adapt to the supplied modalities; Full and all replacement conditions share the same audiovisual instructions. Decoding and frame-sampling settings remain fixed within each model. All runs use batch size 1 and at most 160 generated tokens. Qwen and Baichuan use greedy decoding; MiniCPM uses deterministic decoding with internal beam size 3.

For Baichuan and MiniCPM on MELD, every video-bearing condition uses a common silent-MP4 re-encoding procedure with separate mono 16 kHz WAV audio. Prepared-media hashes are checked across conditions. MiniCPM retains its duration-based A/V packing, so audio duration can change the supplied frame count. Baichuan processes decoded visual frames without converting embedded MP4 audio into audio tokens. Appendix H documents checkpoints, prompts, media handling and available runtime records.

## 3.3 Seven Matched Input Conditions

Table 1 defines all conditions. A fixed mapping $\pi ( i )$ selects the unrelated source: RandA substitutes $A _ { \pi ( i ) }$ , RandV substitutes $V _ { \pi ( i ) }$ , and RandA/V substitutes both. Full is the matched-input reference; Text is the baseline for genuine-modality utility.

Table 1. Seven matched input conditions. Text denotes the fixed target transcript plus preceding textual dialogue context.
<table><tr><td></td><td>Condition Text/Context</td><td>Audio</td><td>Video</td></tr><tr><td>Text</td><td>Yes</td><td></td><td></td></tr><tr><td>T+A</td><td>Yes</td><td>Correct</td><td></td></tr><tr><td>T+V</td><td>Yes</td><td></td><td>Correct</td></tr><tr><td>Full</td><td>Yes</td><td>Correct</td><td>Correct</td></tr><tr><td>RandA</td><td>Yes</td><td> $A _ { \pi ( i ) }$ </td><td>Correct</td></tr><tr><td>RandV</td><td>Yes</td><td>Correct</td><td> $V _ { \pi ( i ) }$ </td></tr><tr><td>RandA/V</td><td>Yes</td><td> $A _ { \pi ( i ) }$ </td><td> $V _ { \pi ( i ) }$ </td></tr></table>

## 3.4 Random-Modality Construction and Leakage Control

Mappings are generated before inference, fixed throughout execution and shared across models and replacement conditions. Sources must come from another dialogue in the same dataset; IEMOCAP sources must additionally share the target session. Source selection is label-agnostic and may include annotations outside the six IEMOCAP target classes. The primary mapping uses seed 42; seeds 43 and 44 provide independent additional mappings under the same restrictions. RandA/V always draws audio and video from the same source utterance, preserving their mutual correspondence while breaking alignment with the target transcript/context.

This is an in-domain stress test, not synthetic noise or an estimate of realworld mismatch frequency. Complete-speech replacement changes lexical content available through audio as well as paralinguistic cues. The explicit transcript stays fixed, but the intervention does not isolate prosody.

## 3.5 Utility, Sensitivity, and Interference

We report Accuracy as the primary metric and Weighted F1 as an additional metric, supplemented by Macro-F1 and Balanced Accuracy. Genuine-modality utility compares T+A, T+V or Full with Text. Sensitivity is the Prediction Change Rate (PCR) between Full and replacement condition R:

$$
\mathrm { P C R } ( R ) = \frac { 1 } { | \mathcal V _ { R } | } \sum _ { i \in \mathcal V _ { R } } \mathrm { I } [ \hat { y } _ { i } ^ { F } \neq \hat { y } _ { i } ^ { R } ] ,\tag{2}
$$

where $\gamma _ { R }$ contains targets with valid outputs in both conditions and I is the indicator function. PCR counts any decision change, regardless of correctness.

Interference has two baselines. Rand-minus-Full measures a replacement penalty. RandA-minus-T+V, RandV-minus-T+A and RandA/V-minus-Text test mismatch-specific harm beyond omission. A modality can be sensitive to replacement without producing negative efects relative to omission. Figure 1 summarizes the audit. Appendix A gives full Benefit/Harm definitions and denominators.

![](images/4e1334ce92c11604c21a7908fe58a2467a0e5b3424f148fccf1444da822a1137.jpg)  
Fig. 1. Controlled reliance audit. Target transcript, prior textual context and speaker information remain fixed. Genuine inputs measure utility; replacements measure sensitivity and efects relative to Full or omission baselines. Emotion-level and mechanistic analyses provide follow-up diagnostics.

## 3.6 Secondary Analyses and Mechanistic Case Study

Exploratory analyses examine emotion-level heterogeneity and source-label compatibility after predictions are fixed. Same sources share the target gold label; Conflict sources do not. Both are cross-dialogue replacements. Conditional Harm is the fraction of Full-correct targets that become incorrect after replacement. Source-conflict Accuracy retains all source-annotated targets, scoring failed outputs as incorrect; conditional Harm uses common-valid Full/replacement pairs. Valence summaries retain all Full-correct targets and count failed replacement outputs as incorrect. Label-standardized Accuracy and conditional Harm assess associations with conflict, rather than identifying a causal emotion mechanism. IEMOCAP sources annotated exactly xxx after normalization (the undefined source annotation) are excluded only from these compatibility analyses. Full definitions and exploratory methods appear in Appendices A–B.

The mechanistic study tests Qwen/MELD RandA-Harm cases, correct under Full but incorrect under RandA. Stable-Correct cases serve as a control. A discovery set of 50 gold-label-matched samples per group nominates fixed blocks. The new confirmatory RandA-Harm cohort is dialogue-disjoint from RandA-Harm discovery: 112 nominal cases yield 106 causal-eligible samples from 76 dialogues after prespecified parity and sham-feasibility checks.

Audio knockout (Audio KO) masks all attention logits from the single decode query predicting the first emotion token after Emotion: to audio-key positions, before softmax, in Early L0–8, Middle L9–18 or Late L19–27. L19–23 is a secondary candidate subset. Prefill activations are unchanged. Video Sham matches block span and masked-edge count, but not attention mass or functional importance.

Teacher-forced candidate-continuation log-likelihood $\ell _ { i } ( y )$ defines the primary margin

$$
m _ { i } = \ell _ { i } ( y _ { i } ) - \operatorname* { m a x } _ { y \neq y _ { i } } \ell _ { i } ( y ) .\tag{3}
$$

This is a gold-vs-best-candidate log-likelihood margin, not a normalized class probability. The primary intervention outcome is the Audio-KO-minus-Video-Sham change in $m _ { i } ;$ a positive efect favors the gold label. Appendices A and G detail scoring, cohort checks and control analyses.

## 3.7 Statistical Testing and Reproducibility

Utterances share speakers and context within dialogues. We therefore retain utterance-weighted paired Accuracy efects while resampling and sign-flipping entire dialogues. Primary seed-42 CIs and p values use 1,000 dialogue-cluster bootstrap and sign-flip resamples. BH-FDR is applied separately to 18 utility comparisons, 18 omission comparisons and six RandA/V-minus-Full comparisons. Throughout, q denotes the adjusted p value within the specified family.

The three-mapping analysis uses 10,000 resamples and separate 18-test replacement and 54-test omission BH families under each failed-output policy. Primary failed-output sensitivity uses 10,000 resamples with separate 18-test utility, 18-test omission and six-test replacement families. Repeated targets across mappings are not pooled as independent observations; mapping means/ranges are descriptive. Full efects, CIs and family definitions are provided in Appendices C, I–K.

Aggregate Accuracy, Weighted F1, Macro-F1 and Balanced Accuracy are computed over all expected targets. Invalid/OOM outcomes contribute no true positives and remain in gold-class support. Primary paired efects and PCR use common-valid pairs. Consequently, aggregate diferences and paired efects can difer. We also test all-target paired efects with failures scored as incorrect. Final outcomes, exclusions and unchanged-input OOM retries are retained, not silently dropped (Appendix K).

Exploratory source-conflict analyses standardize target-label distributions. Broad comparisons use 1,000 bootstrap/within-label permutation resamples and endpoint-specific 18-test BH families; three focused follow-ups additionally use dialogue-cluster bootstrap and permutations within dialogue/target-label strata, with endpoint-specific three-test BH families.

Mechanistic margin CIs use 10,000 dialogue-cluster bootstrap resamples. The prespecified tests of Late versus zero, Late minus Early and Late minus Middle use 100,000 cluster sign flips and Holm correction. Secondary block, rescue and control comparisons use their stated BH families. The accompanying materials include code, complete prompts, six mapping manifests, per-target predictions, runtime records and invalid/OOM logs.

## 4 Results

## 4.1 Genuine Modalities Provide Unequal Utility

The primary evaluation uses source-mapping seed 42. Table 2 reports all seven conditions, and Table 3 reports key common-valid paired efects with their separate FDR families. T+A increases aggregate Accuracy in all six model– dataset pairs; four paired gains remain significant after FDR correction. For Baichuan/MELD, the paired audio gain is +4.10 pp (95% CI [2.38, 5.70], q = .0045); for MiniCPM/IEMOCAP it is +3.55 pp ([2.51, 4.59], q = .0045). The two remaining gains are positive but not FDR-significant.

Table 2. Primary behavioral results (seed 42; %). Acc.=Accuracy; WF1=Weighted F1; RandA=Random Audio; RandV=Random Video; RandA/V=Random A/V.  
(a) MELD
<table><tr><td>Model</td><td>Metric</td><td>Text</td><td>T+A</td><td>T+V</td><td>Full</td><td>RandA</td><td>RandV</td><td>RandA/V</td></tr><tr><td>Qwen</td><td>Acc.</td><td>61.88</td><td>63.22</td><td>61.00</td><td>62.84</td><td>59.00</td><td>62.34</td><td>58.16</td></tr><tr><td>Qwen</td><td>WF1</td><td>60.22</td><td>61.10</td><td>58.39</td><td>60.17</td><td>55.41</td><td>59.23</td><td>54.02</td></tr><tr><td>Baichuan</td><td>Acc.</td><td>36.63</td><td>40.73</td><td>36.90</td><td>36.63</td><td>34.71</td><td>33.14</td><td>32.95</td></tr><tr><td>Baichuan</td><td>WF1</td><td>35.60</td><td>39.89</td><td>35.95</td><td>34.96</td><td>33.65</td><td>30.91</td><td>31.80</td></tr><tr><td>MiniCPM</td><td>Acc.</td><td>43.10</td><td>43.41</td><td>41.88</td><td>44.48</td><td>43.30</td><td>44.56</td><td>41.61</td></tr><tr><td>MiniCPM</td><td>WF1</td><td>43.37</td><td>43.45</td><td>42.10</td><td>44.70</td><td>43.69</td><td>44.83</td><td>41.64</td></tr></table>

(b) IEMOCAP
<table><tr><td>Model</td><td>Metric</td><td>Text</td><td>T+A</td><td>T+V</td><td>Full</td><td>RandA</td><td>RandV</td><td>RandA/V</td></tr><tr><td>Qwen</td><td>Acc.</td><td>49.82</td><td>51.41</td><td>48.21</td><td>50.42</td><td>46.59</td><td>49.66</td><td>45.42</td></tr><tr><td>Qwen</td><td>WF1</td><td>47.16</td><td>48.98</td><td>45.54</td><td>47.95</td><td>43.49</td><td>47.35</td><td>42.42</td></tr><tr><td>Baichuan</td><td>Acc.</td><td>46.34</td><td>50.46</td><td>45.53</td><td>49.20</td><td>45.11</td><td>47.21</td><td>43.02</td></tr><tr><td>Baichuan</td><td>WF1</td><td>45.20</td><td>49.79</td><td>44.04</td><td>48.02</td><td>43.74</td><td>46.20</td><td>41.71</td></tr><tr><td>MiniCPM</td><td>Acc.</td><td>45.07</td><td>48.63</td><td>46.02</td><td>47.76</td><td>46.78</td><td>47.43</td><td>46.04</td></tr><tr><td>MiniCPM</td><td>WF1</td><td>42.97</td><td>46.72</td><td>43.96</td><td>45.49</td><td>44.38</td><td>45.09</td><td>43.92</td></tr></table>

Additional modalities are not monotonically beneficial. Baichuan/MELD rises from 36.63% with Text to 40.73% with T+A, while Full returns to 36.63%. T+A exceeds Full in five of six pairs, whereas T+V exceeds Text in only two. Utility also depends on the metric: Qwen/MELD T+A improves Accuracy and Weighted F1, but Macro-F1 falls from .462 to .460 and Balanced Accuracy from .440 to .424. Better overall Accuracy therefore does not imply uniformly better emotion-class performance. Appendices D and I provide all metrics and paired comparisons.

## 4.2 Sensitivity, Utility, and Mismatch Vulnerability Diverge

Figure 2 contrasts genuine utility, replacement penalties and PCR. Modest net gains can coexist with substantial sensitivity. For Qwen, Full-minus-Text paired efects are +1.00 pp on MELD (95% CI [−0.41, 2.39], cluster p = .189) and +0.59 pp on IEMOCAP ([−0.58, 1.76], p = .356). Yet RandA changes 15.12% of common-valid MELD predictions, and RandA/V changes 18.78%. Baichuan/MELD RandA/V changes 41.80%. PCR also counts wrong-to-right and wrong-to-diferent-wrong transitions, so it can greatly exceed the net Accuracy diference.

![](images/beffbdc0b9f214d72b2f6e86d0da9400aec9ddf11a67ede63b1fd0e1e8a1f1e5.jpg)  
Fig. 2. Behavioral separation using primary seed-42 predictions. Blue, orange diagonally hatched and green cross-hatched bars denote Audio, Video and A/V. (a) Accuracy change versus Text. (b) Replacement penalty versus Full. (c) PCR on common-valid pairs. Panels (a)–(b) use aggregate Accuracy diferences; Table 3 gives paired efects and omission comparisons.

RandA/V has a significant negative paired efect relative to Full in all six primary comparisons (Table 3). This does not establish that every replacement is harmful: MiniCPM/MELD RandV attains 44.56% Accuracy versus 44.48% for Full. The omission comparisons distinguish replacement penalties from additional mismatch-specific harm. Baichuan/MELD RandV is 7.55 pp below T+A on common-valid pairs (q = .0022), and Qwen/MELD RandA/V is 3.72 pp below Text (q = .0022).

Conversely, MiniCPM/IEMOCAP RandA/V exceeds Text by +1.56 pp (q = .0054), despite a negative RandA/V-minus-Full efect. This positive omissionrelative finding is sensitive to failed-output handling. A loss relative to Full therefore need not imply additional harm beyond simply omitting the replaced modalities.

Together, these results show that a modality can provide little net benefit when correctly matched to the target yet still substantially influence predictions when its content is replaced. Utility and vulnerability to misleading input are therefore distinct properties of multimodal behavior. Appendix B, Table S2 summarizes Full–Text, RandA–Full, and RandA/V–Full aggregate Accuracy diferences and PCR values for RandA and RandA/V.

Table 3. Primary common-valid paired Accuracy efects (pp), 95% dialogue-cluster CIs and BH-FDR-adjusted q values. N is the number of paired valid targets. The utility and omission families include all three relevant conditions; panels show the key audio and joint-replacement contrasts. Positive omission efects are not harm.  
(a) T+A–Text: genuine audio utility (18-test family)
<table><tr><td>Dataset</td><td>Model</td><td>Effect</td><td>95% CI</td><td>q</td><td>N</td></tr><tr><td rowspan="3">MELD</td><td>Qwen</td><td>+1.30</td><td>[-0.14, +2.66]</td><td>.1169</td><td>2606</td></tr><tr><td>Baichuan</td><td>+4.10</td><td>[+2.38, +5.70]</td><td>.0045</td><td>2610</td></tr><tr><td>MiniCPM</td><td>+0.27</td><td>[-1.16, +1.59]</td><td>.8177</td><td>2603</td></tr><tr><td rowspan="3">IEMOCAP</td><td>Qwen</td><td>+1.65</td><td>[+0.56, +2.68]</td><td>.0072</td><td>7335</td></tr><tr><td>Baichuan</td><td>+4.15</td><td>[+2.19, +6.18]</td><td>.0045</td><td>7373</td></tr><tr><td>MiniCPM</td><td>+3.55</td><td>[+2.51, +4.59]</td><td>.0045</td><td>7370</td></tr></table>

(b) RandA/V–Full: replacement penalty (six-test family)
<table><tr><td>Dataset</td><td>Model</td><td>Effect</td><td>95% CI</td><td>q</td><td>N</td></tr><tr><td rowspan="2">MELD</td><td>Qwen</td><td>-4.69</td><td>[-6.16, -3.29]</td><td>.0012</td><td>2604</td></tr><tr><td>Baichuan MiniCPM</td><td>-3.72 -2.81</td><td>[-5.70, -1.80] [-4.12, -1.46]</td><td>.0012 .0012</td><td>2605 2598</td></tr><tr><td rowspan="2">IEMOCAP</td><td>Qwen</td><td>-5.00</td><td>[-6.02, -3.98]</td><td>.0012</td><td>7344</td></tr><tr><td>Baichuan</td><td>-6.21</td><td>[-7.97, -4.32]</td><td>.0012</td><td>7365</td></tr><tr><td></td><td>MiniCPM</td><td>-1.24</td><td>[-2.07, -0.49]</td><td>.0070</td><td>7260</td></tr></table>

(c) RandA/V–Text: mismatch versus omission (18-test family)
<table><tr><td>Dataset</td><td>Model</td><td>Effect</td><td>95% CI</td><td>q</td><td>N</td></tr><tr><td rowspan="2">MELD</td><td>Qwen</td><td>-3.72</td><td>[-5.00, -2.33]</td><td>.0022</td><td>2605</td></tr><tr><td>Baichuan MiniCPM</td><td>-3.68 -1.39</td><td>[-5.57, -1.91] [-2.91, +0.04]</td><td>.0022 .0820</td><td>2608</td></tr><tr><td rowspan="3">IEMOCAP</td><td>Qwen</td><td>-4.39</td><td>[-5.69, -3.07]</td><td>.0022</td><td>2595 7340</td></tr><tr><td>Baichuan</td><td>-3.27</td><td>[-5.42, -1.26]</td><td>.0054</td><td>7373</td></tr><tr><td>MiniCPM</td><td>+1.56</td><td>[+0.65, +2.49]</td><td>.0054</td><td>7261</td></tr></table>

## 4.3 Exploratory Emotion and Source-Conflict Analyses

Emotion-level efects do not follow a universal valence pattern. Class-balanced negative-class Full Accuracy is lower than positive-class Accuracy for all models on MELD but higher on IEMOCAP. Negative targets nevertheless show greater RandA PCR and conditional Harm in five of six pairs. We treat these as descriptive heterogeneity rather than a general claim that one valence is harder (Appendices B and F).

Source-label conflict is associated with additional harm in several exploratory settings after target-label standardization. For Qwen/MELD RandA, Conflictminus-Same adjusted Accuracy is −7.59 pp and conditional Harm is +8.26 pp, with focused dialogue-cluster $q = . 0 1 7 0$ and .0030, respectively. This setting, Qwen/MELD RandA/V and Baichuan/IEMOCAP RandA/V remain significant for both endpoints in the broader 18-comparison families. These post-hoc label associations do not isolate a causal emotion mechanism. Appendices B and E report the focused and complete analyses.

## 4.4 Mechanistic Case Study: RandA in Qwen

Attention patterns nominate candidate blocks. However, attention diferences between RandA-Harm and Stable-Correct do not survive multiple-testing correction. Direct intervention supplies the confirmatory evidence. Figure 3 reports the causal-eligible confirmatory cohort (N = 106, 76 dialogues). Audio-KO-minus-Video-Sham margin recovery is near zero in Early, increases in Middle and is largest in Late: +0.1722 nats (95% CI [0.1120, 0.2393]). All three prespecified contrasts pass Holm correction, including Late versus Middle $( p _ { \mathrm { H o l m } } = . 0 0 3 8 2 )$

(a) Prediction rescue (Audio KO vs. Video Sham)  
![](images/17cc7d5b96fdda030a36fea5e643ac9ec8b921a64afbaab5e931046912bbc6ec.jpg)  
Secondary BH-FDR tests; confirmatory causal-eligible N=106

(b) Gold-vs-best-candidate margin recovery (Audio KO − Video Sham)  
![](images/360ac9c6d06dcdc2368ba01054c5a92b7dd522951ff44ff6653824d5e245ef7e.jpg)  
Bars: utterance-weighted means; error bars: 95% dialogue-cluster bootstrap CIs L19–23 is a prespecified secondary candidate subset  
Fig. 3. Qwen/MELD RandA-Harm confirmation (N = 106, 76 dialogues). (a) Prediction rescue under Audio KO and Video Sham; q values are secondary BH-FDR tests. (b) Utterance-weighted Audio-KO-minus-Video-Sham margin recovery with 95% dialoguecluster bootstrap CIs. Brackets show prespecified Holm-corrected contrasts; L19–23 is a secondary candidate subset.

Prediction rescue is secondary: negligible in Early but significant in Middle, Late and the candidate subset (Late: 9.43%). Later Stable-Correct margin shifts are positive, but RandA-Harm recovery is significantly greater in Middle and Late under the tested masking controls. The intervention recovers some failures without locating their origin; full scores and controls are in Appendix G.

## 4.5 Robustness Across Mappings and Failed-Output Policies

Across seeds 42, 43 and 44, all 18 model–dataset–mapping RandA/V-minus-Full comparisons are negative and FDR-significant under common-valid dialoguecluster inference and when invalid/OOM outcomes are retained as incorrect. These are separate mapping-specific tests, not pooled repeated targets (Appendix J).

In the common-valid 54-test omission family, 32 comparisons have significant negative efects and seven significant positive efects. MiniCPM/IEMOCAP RandA/V-minus-Text is positive across all mappings (+1.19 to +1.62 pp), but only seed 43 remains significant when failures are scored as incorrect (q = .04976). Replacement penalties are robust across the tested mappings; omission-relative interpretations can depend on both model and failed-output policy. Appendix K gives all failure counts and policy comparisons.

## 5 Discussion

Evaluating reliance. The audit distinguishes useful contribution, replacement sensitivity and harm beyond omission. Their joint reporting matters because Full is not uniformly better than T+A and omission-relative efects can difer in sign. For human–AI interaction, a model that benefits from genuine audio may still change a correct decision under unrelated input. Matched-input accuracy would miss this vulnerability, while PCR alone would conflate harmful changes with corrections. Reliability assessments should therefore include both matched-input gains and replacement-versus-omission efects. These measurements motivate future consistency checks and modality-selection strategies; testing their efectiveness and estimating deployment failure rates require separate studies.

Interpreting the intervention. The confirmation on new RandA-Harm cases shows greater decision-query margin recovery in Late than Early or Middle. Margin recovery is significantly greater for RandA-Harm than for Stable-Correct in Middle and Late under the tested masking controls. These findings concern the tested intervention sites and do not establish where the original error originates or generalize the internal dynamics beyond Qwen/MELD.

Limitations. The audit tests target-level A/V interventions in MERC on two datasets and three open-source Omni models; broader tasks, architectures and commercial systems remain future work. Three mappings establish robustness over the tested pairings rather than precisely estimating all mapping variability. Complete-speech replacement changes lexical and paralinguistic information. Source-label conflict is post-hoc and label-based, and Video Sham controls edge count without matching attention mass or functional importance. These qualifications bound the scope of the conclusions.

## 6 Conclusion

We presented a matched modality-intervention audit separating utility, sensitivity and mismatch-specific interference in conversational emotion recognition. Genuine audio improves Accuracy in all six model–dataset pairs, with four FDR-significant gains, while video efects are less consistent. RandA/V lowers Accuracy relative to Full in all 18 mapping comparisons under both failed-output policies, whereas omission-baseline efects vary. A Qwen/MELD case study confirms greater Late decision-query audio-edge margin recovery than Early or Middle relative to Video Sham. Multimodal evaluation should jointly measure useful contribution and behavior under misleading input.

Data and supplementary material. The appendix contains the extended methods and complete statistical tables. Reproducibility records are available at https:// doi.org/10.5281/zenodo.23970406. The main results and their primary testing families are reported above.

Acknowledgments. This research was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) under the Artificial Intelligence Convergence Innovation Human Resources Development (IITP-2026-RS-2023-00254592) , and by ANCHOR through the Seoul ANCHOR Center, funded by the Ministry of Education (MOE) and the Seoul Metropolitan Government (2026-ANCHOR-01-007-05)

Disclosure of Interests. The authors have no competing interests to declare that are relevant to the content of this article.

## References

1. Busso, C., Bulut, M., Lee, C.C., Kazemzadeh, A., Mower, E., Kim, S., Chang, J.N., Lee, S., Narayanan, S.S.: IEMOCAP: Interactive emotional dyadic motion capture database. Language Resources and Evaluation 42(4), 335–359 (2008). https://doi.org/10.1007/s10579-008-9076-6

2. Chen, Y., Yang, S., Xiang, Y., Song, R., Huang, Y., Yu, Z.: GRPO-guided modality selection enhanced LoRA-tuned LLMs for multimodal emotion recognition. In: Findings of the Association for Computational Linguistics: EMNLP 2025. pp. 19458–19471 (2025). https://doi.org/10.18653/v1/2025.findings-emnlp.1059

3. Fan, Y., Zhao, A., Fu, J., Tong, J., Su, H., Pan, Y., Zhang, W., Shen, X.: VisiPruner: Decoding discontinuous cross-modal dynamics for eficient multimodal LLMs. In: EMNLP 2025. pp. 18885–18902 (2025). https://doi.org/10.18653/v1/2025. emnlp-main.955

4. Ghosal, D., Majumder, N., Poria, S., Chhaya, N., Gelbukh, A.: DialogueGCN: A graph convolutional neural network for emotion recognition in conversation. In: EMNLP-IJCNLP 2019. pp. 154–164 (2019). https://doi.org/10.18653/v1/ D19-1015

5. Golovanevsky, M., Rudman, W., Palit, V., Eickhof, C., Singh, R.: What do VLMs NOTICE? A mechanistic interpretability pipeline for Gaussian-noise-free textimage corruption and evaluation. In: NAACL 2025. pp. 11462–11482 (2025). https: //doi.org/10.18653/v1/2025.naacl-long.571

6. Han, Z., Zhu, B., Xu, Y., Song, P., Yang, X.: Benchmarking and bridging emotion conflicts for multimodal emotion reasoning. In: 33rd ACM International Conference on Multimedia. pp. 5528–5537 (2025). https://doi.org/10.1145/3746027. 3754856

7. Hazarika, D., Poria, S., Zadeh, A., Cambria, E., Morency, L.P., Zimmermann, R.: Conversational memory network for emotion recognition in dyadic dialogue videos. In: NAACL-HLT 2018, Volume 1. pp. 2122–2132 (2018). https://doi.org/ 10.18653/v1/N18-1193

8. Hu, J., Liu, Y., Zhao, J., Jin, Q.: MMGCN: Multimodal fusion via deep graph convolution network for emotion recognition in conversation. In: ACL-IJCNLP 2021. pp. 5666–5675 (2021). https://doi.org/10.18653/v1/2021.acl-long.440

9. Li, B., Fei, H., Liao, L., Zhao, Y., Teng, C., Chua, T.S., Ji, D., Li, F.: Revisiting disentanglement and fusion on modality and context in conversational multimodal emotion recognition. In: 31st ACM International Conference on Multimedia. pp. 5923–5934 (2023). https://doi.org/10.1145/3581783.3612053

10. Li, Y., Liu, J., Zhang, T., Chen, S., Li, T., Li, Z., Liu, L., Ming, L., Dong, G., Pan, D., et al.: Baichuan-Omni-1.5 technical report. arXiv:2501.15368 (2025).

11. Majumder, N., Poria, S., Hazarika, D., Mihalcea, R., Gelbukh, A., Cambria, E.: DialogueRNN: An attentive RNN for emotion detection in conversations. In: AAAI. vol. 33, pp. 6818–6825 (2019). https://doi.org/10.1609/aaai.v33i01.33016818

12. Murzaku, J., Rambow, O.: OmniVox: Zero-shot emotion recognition with Omni LLMs. arXiv:2503.21480 (2025).

13. OpenBMB: MiniCPM-o 2.6 model card. Hugging Face (2025). https://huggingface.co/openbmb/MiniCPM-o-2\_6

14. Poria, S., Hazarika, D., Majumder, N., Naik, G., Cambria, E., Mihalcea, R.: MELD: A multimodal multi-party dataset for emotion recognition in conversations. In: ACL 2019. pp. 527–536 (2019). https://doi.org/10.18653/v1/P19-1050

15. Seong, E., Lee, H., Kim, D., Kim, C., Chae, D.K.: Emotion-wheel-guided audio referred text representation for multimodal emotion recognition in conversation. In: ACL 2026. pp. 40392–40403 (2026). https://doi.org/10.18653/v1/2026. acl-long.1875

16. Sun, Y., Zhang, Y., Gu, H., Chen, N., She, D., Yao, X., Gao, Y., Jin, Z.: EmoMM: Benchmarking and steering MLLM for multimodal emotion recognition under conflict and missingness. In: Findings of ACL 2026. pp. 20351–20371 (2026). https: //doi.org/10.18653/v1/2026.findings-acl.1018

17. Xu, J., Guo, Z., He, J., Hu, H., He, T., Bai, S., Chen, K., Wang, J., Fan, Y., Dang, K., Zhang, B., Wang, X., Chu, Y., Lin, J.: Qwen2.5-Omni technical report. arXiv:2503.20215 (2025).

## Appendix

This appendix uses the task notation of the main text and provides the extended definitions, exploratory analyses, complete statistics and runtime/failure documentation. The primary seed-42 and additional-mapping results use the same reference predictions. The accompanying machine-readable files distinguish the 1,000-resample primary families from the 10,000-resample mapping and failed-output families; their adjusted p values are not interchangeable.

## A Extended Definitions and Analysis Methods

## A.1 Definitions and Sample-Level Outcomes

We report Accuracy as the primary metric and Weighted F1 as an additional metric for comparability with prior MERC evaluations. Aggregate metrics are supplemented with paired sample-level measurements. For a genuine multimodal condition $M \in \{ T A , T V , F \}$ , Benefit and Harm relative to Text are

$$
B _ { i } ^ { M } = \operatorname { I } [ \hat { y } _ { i } ^ { T } \neq y _ { i } \land \hat { y } _ { i } ^ { M } = y _ { i } ] ,\tag{S1}
$$

$$
H _ { i } ^ { M } = \mathrm { I } [ \hat { y } _ { i } ^ { T } = y _ { i } \wedge \hat { y } _ { i } ^ { M } \neq y _ { i } ] .\tag{S2}
$$

Benefit measures correction of a text-only error, whereas Harm measures cases in which added target audio/video input overrides an already correct textual decision. When aggregated, Benefit is computed over Text-incorrect samples and Harm over Text-correct samples, so the two rates have diferent denominators. Benefit and Harm are computed only over samples with valid predictions under both Text and the corresponding multimodal condition. We operationalize sensitivity using Prediction Change Rate (PCR) between Full and a random-modality condition R:

$$
\mathrm { P C R } ( R ) = \frac { 1 } { | \mathcal V _ { R } | } \sum _ { i \in \mathcal V _ { R } } \mathrm { I } [ \hat { y } _ { i } ^ { F } \neq \hat { y } _ { i } ^ { R } ] ,\tag{S3}
$$

where $\gamma _ { R }$ contains samples with valid predictions under both conditions. PCR measures whether the decision changes, irrespective of whether the change is correct. The indicator function I[E] equals 1 if event E holds and 0 otherwise. Figure 1 summarizes the overall controlled modality-intervention framework and the follow-up analyses.

We assess interference at two complementary levels. Rand-versus-Full comparisons quantify the replacement penalty: the performance change when matched evidence is replaced by unrelated in-domain media. The stricter mismatch-specific interference comparison asks whether adding unrelated media is more harmful than omitting that modality altogether: RandA is compared with T+V, RandV with T+A, and RandA/V with Text. A negative mismatch-specific efect therefore indicates active harm beyond modality absence. This distinction is central to the audit: a model can be highly sensitive to replacement while showing either weak, strong, or even reversed active interference relative to omission. Table S1 summarizes these operational distinctions.

Table S1. Operational distinction among utility, sensitivity, and interference. Negative interference efects indicate harm relative to the stated baseline.
<table><tr><td>Concept</td><td>Primary metric</td><td>Interpretation</td></tr><tr><td>Utility</td><td>∆Acc vs. Text</td><td>Benefit from genuine A/V</td></tr><tr><td>Sensitivity</td><td>PCR</td><td>Prediction changes after replacement</td></tr><tr><td>Interference</td><td>∆Acc: Rand-Full</td><td>Replacement penalty</td></tr><tr><td></td><td>∆Acc: Rand-omission</td><td>Mismatch-specific harm</td></tr></table>

## A.2 Exploratory and Mechanistic Methods

We use two secondary behavioral analyses to characterize how aggregate modality efects vary across emotions and to examine what factors are associated with greater mismatch harm. First, we examine emotion-level heterogeneity to determine whether the aggregate modality efects are broadly shared across classes or concentrated in particular emotions. For MELD, Positive contains joy and Negative contains anger, disgust, fear, and sadness; for IEMOCAP, Positive contains happy and excited, and Negative contains angry, frustrated, and sad. Neutral classes, and MELD surprise, are excluded from the valence comparison. Because class frequencies difer, accuracy is macro-averaged across emotion classes within each group. This analysis is used to characterize how modality efects vary across emotions and broad valence groups, rather than to test whether positive or negative emotions are generally more dificult to recognize.

Second, we test source-label compatibility to distinguish harm caused by crossdialogue replacement itself from additional harm associated with emotion-label conflict. After all predictions are fixed, a random source is labeled Same when its dataset gold emotion matches the target gold emotion and Conflict otherwise. For IEMOCAP, random-source cases carrying the undefined source annotation xxx after normalization are excluded from the source-label compatibility analysis only. All other source labels are retained; labels that do not match the target gold label are treated as Conflict. These exclusions do not alter the behavioral random-replacement analyses. This is a label-level compatibility test: the source label does not guarantee that every acoustic or visual cue in the raw media expresses that emotion. Both groups still use media from another dialogue. To reduce target-label imbalance, group accuracy is standardized to the pooled target-emotion distribution. Conditional Harm is the proportion of samples that are correct under Full but become incorrect under the mismatch condition within each compatibility group. Because this analysis is post-hoc, the three comparisons highlighted in the main text are exploratory follow-ups. They survive the focused dialogue-cluster reanalysis and remain significant for both endpoints in the broader label-adjusted 18-comparison sensitivity analysis; all 18 results are reported in Section E.

We additionally conduct a mechanistic case study on Qwen2.5-Omni-7B using MELD to test whether the behavioral efect of misleading audio is reflected in a targeted internal intervention. We use Qwen/MELD as a focused mechanistic case study because the behavioral analysis yields a well-defined cohort of RandAinduced failure cases suitable for targeted intervention. RandA-Harm samples are correct under Full but incorrect under RandA, whereas Stable-Correct samples remain correct. An initial discovery set of 50 samples per group, matched by gold emotion, is used only to nominate fixed layer blocks. The primary confirmatory cohort contains 112 new RandA-Harm samples from dialogues disjoint from RandA-Harm discovery; prespecified parity and sham-feasibility checks leave 106 causal-eligible samples from 76 dialogues. The remaining new cases are used only in robustness summaries. Stable-Correct serves as a control for determining whether the intervention efect is specific to RandA-induced failures rather than a generic consequence of masking.

Layer-wise attention is used only to nominate candidate regions for the direct intervention. For RandA-Harm inputs, the audio-key positions correspond to the mismatched RandA audio supplied in the behavioral condition. Audio knockout (Audio KO) acts on the single decode query used to predict the first emotionlabel token immediately after the generated Emotion: prefix. The tested blocks are Early (L0–8), Middle (L9–18), Late (L19–27), and the secondary L19–23 candidate subset. At the selected layers, all attention logits from this query to audio-key positions are masked before softmax; prefill activations are left unchanged. Video Sham matches the layer span and masked-edge count of Audio KO. It does not match the removed attention mass or the functional importance of the masked connections.

Candidate labels are scored by teacher-forced sequence log-likelihood. For a candidate continuation $y = ( y _ { 1 } , \dots , y _ { T } )$ , let $x _ { i }$ denote the conditioned multimodal input and $h _ { i }$ the generated prefix through Emotion:. We compute

$$
\ell _ { i } ( y ) = \sum _ { t = 1 } ^ { T } \log p _ { \theta } ( y _ { t } \mid x _ { i } , h _ { i } , y _ { < t } ) ,
$$

and the primary score endpoint is

$$
m _ { i } = \ell _ { i } ( y _ { i } ) - \operatorname* { m a x } _ { y \neq y _ { i } } \ell _ { i } ( y ) .\tag{S4}
$$

Thus, $m _ { i }$ is a gold-vs-best-candidate log-likelihood margin rather than a normalized seven-class probability margin. Under the tokenizer used for the MELD intervention analysis, all seven label continuations are single tokens, although the implementation scores the full continuation and therefore also supports multitoken candidates. A positive Audio-KO-minus-Video-Sham diference means that masking the targeted audio-to-decision edges moves this candidate-label margin toward the gold emotion relative to the matched control.

## A.3 Complete Statistical Protocols

Utterances within a dialogue share speakers, conversational context, and recording conditions, so their predictions can be dependent. We therefore use dialogue as the resampling unit while retaining an utterance-weighted paired Accuracy efect. In the primary seed-42 behavioral analyses, confidence intervals use 1,000 dialogue-cluster bootstrap resamples and p values use 1,000 dialogue-level sign-flip resamples. Genuine utility comparisons (T+A vs. Text, T+V vs. Text, Full vs. Text) form a separate 18-test BH-FDR family across three conditions, three models, and two datasets. Throughout the paper, q denotes the Benjamini–Hochberg FDR-adjusted p value within the stated testing family. Mismatch-specific interference comparisons (RandA vs. T+V, RandV vs. T+A, RandA/V vs. Text) form another separate 18-test BH-FDR family. For the six primary RandA/V-vs.-Full replacement-penalty robustness comparisons, BH-FDR is applied across the six model–dataset tests; all six remain significant (Section C). Exact McNemar tests are treated only as unclustered auxiliary analyses.

The additional-mapping analysis uses 10,000 dialogue-cluster bootstrap and sign-flip resamples. Under each failed-output policy, BH-FDR is applied separately to 18 RandA/V-minus-Full comparisons and 54 omission-baseline comparisons across three mappings, three models and two datasets. Efects are reported separately by mapping; their means and ranges are descriptive, and repeated targets across mappings are not pooled as independent observations. The failed-output sensitivity analysis of the primary predictions also uses 10,000 resamples, retaining separate families of 18 utility, 18 omission and six replacement comparisons. Reference predictions and runtime details for the three-mapping analysis are described in Section J.

For IEMOCAP source-conflict analyses, exact xxx source labels are excluded before standardization and resampling. For the label-adjusted source-conflict analysis, target-label distributions are standardized to a pooled common distribution. The broad 18-comparison analysis uses 1,000 bootstrap and 1,000 within-label permutation resamples, with separate 18-test BH-FDR families for adjusted Accuracy and Conditional Harm. The three main follow-ups additionally use dialogue-cluster bootstrap and permutation within strata jointly defined by dialogue ID and target gold label, with BH-FDR over the three focused comparisons. Full details and all 18 results are in Sections E and H.

For the mechanistic confirmation, the primary endpoint is the utteranceweighted Audio KO minus Video Sham change in gold-vs-best margin. Confidence intervals use 10,000 dialogue-cluster bootstrap resamples, and primary p values use 100,000 dialogue-cluster sign-flip resamples. Three prespecified primary tests— Late vs. zero, Late minus Early, and Late minus Middle—are corrected with Holm’s method. Secondary block-level, rescue, and control comparisons use BH-FDR within their stated families.

Aggregate Accuracy, Weighted F1, Macro-F1 and Balanced Accuracy are computed over all expected targets, retaining invalid outputs and out-of-memory (OOM) failures as incorrect predictions. PCR and the primary paired Accuracy estimates use only samples with valid predictions under both compared conditions.

Diferences of condition-wise aggregate accuracies (Table S2) can therefore difer from common-valid paired efects (Sections C and I). We additionally repeat the key paired comparisons on all expected targets, treating failed outcomes as incorrect. These policies estimate diferent quantities and can change efect sizes and significance. Condition-wise failures, exclusions and policy comparisons are reported in Section K. In the post-hoc conflict analysis, standardized Accuracy treats invalid/OOM outcomes as incorrect, whereas conditional Harm excludes invalid outcomes, using common-valid Full/mismatch pairs. Decoding is deterministic but model-specific. Qwen and Baichuan use deterministic greedy decoding, whereas MiniCPM uses deterministic decoding with an internal beam size of 3. All runs use batch size 1 and a maximum generation length of 160 tokens. Model checkpoints, prompt specifications, media handling, recorded runtime versions, and unrecorded fields are detailed in Section H. In particular, Baichuan processes video inputs as decoded visual frames only; embedded MP4 audio is not converted into audio tokens.

## B Additional Behavioral Summaries

## B.1 Aggregate Efects and Prediction Changes

The following table reports condition-wise aggregate Accuracy diferences and common-valid PCR. Aggregate diferences need not equal paired common-valid Accuracy efects because their denominators difer.

Table S2. Summary of multimodal efects in the primary evaluation (source-mapping seed 42). ∆Accuracy is in percentage points (pp), and PCR in percent. Full–Text reflects genuine-input utility; RandA–Full and RandA/V–Full are replacement penalties relative to matched Full input. Entries are diferences of condition-wise aggregate accuracies, not common-valid paired estimates. Mismatch-specific interference is analyzed in Section I.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Model</td><td colspan="3">∆ Accuracy (pp)</td><td colspan="2">PCR (%)</td></tr><tr><td></td><td>Full-Text RandA-Full RandA/V-Full</td><td></td><td>RandA</td><td>RandA/V</td></tr><tr><td rowspan="3">MELD</td><td>Qwen</td><td>0.96</td><td>-3.83</td><td>-4.67</td><td>15.12</td><td>18.78</td></tr><tr><td>Baichuan</td><td>0.00</td><td>-1.92</td><td>-3.68</td><td>24.34</td><td>41.80</td></tr><tr><td>MiniCPM</td><td>1.38</td><td>-1.19</td><td>-2.87</td><td>13.98</td><td>18.59</td></tr><tr><td rowspan="3">IEMOCAP</td><td>Qwen</td><td>0.60</td><td>-3.83</td><td>-5.00</td><td>15.11</td><td>18.06</td></tr><tr><td>Baichuan</td><td>2.86</td><td>-4.09</td><td>-6.18</td><td>30.18</td><td>38.98</td></tr><tr><td>MiniCPM</td><td>2.70</td><td>-0.99</td><td>-1.72</td><td>12.48</td><td>13.13</td></tr></table>

## B.2 Secondary Analysis: Emotion-Level Heterogeneity

The efect of modality interventions varies across emotions and is not explained by a consistent positive-versus-negative pattern. Under class-balanced Full accuracy, negative classes perform worse than the positive group on MELD for all three models, whereas the direction reverses on IEMOCAP. Large diferences also appear within the same valence group: for Qwen on IEMOCAP, Full Accuracy is 65.71% for happy but only 8.93% for excited. At the same time, negative targets show higher RandA PCR and conditional harm in five of six model–dataset pairs.

These results show that broad valence groups alone do not explain which emotions are dificult or vulnerable to modality replacement. We therefore treat the valence comparison as a secondary descriptive analysis of emotion-level variation rather than as evidence that positive or negative emotions are generally more dificult to recognize. Full valence-group results are provided in Section F.

## B.3 Source-Label Conflict Is Associated with Greater Mismatch Harm

A mismatched modality may be harmful simply because it comes from another dialogue. We therefore ask whether mismatch harm is greater when the emotion label of the replacement source conflicts with the target label. This is a post-hoc, exploratory analysis rather than a prespecified confirmatory analysis. Table S3 reports the three focused comparisons that remain significant after target-label standardization, dialogue-cluster inference, and BH-FDR correction. These same three settings also remain significant for both endpoints in the broader labeladjusted 18-comparison BH families; the full analysis is reported in Section E.

For Qwen on MELD, conflicting RandA reduces adjusted accuracy by 7.59 pp relative to Same and increases conditional harm by 8.26 pp. The pattern persists under RandA/V. For Baichuan on IEMOCAP, conflicting RandA/V reduces adjusted Accuracy by 10.61 pp and increases conditional Harm by 12.11 pp. These results indicate that mismatch harm is not explained solely by replacing target media with content from another dialogue; in some settings, source-target label conflict is associated with greater harm.

Table S3. Focused dialogue-cluster robustness for the three exploratory source-label conflict comparisons. Efects are Conflict–Same. Positive conditional harm means that conflicting mismatches more often break a prediction that was correct under Full.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Dataset</td><td rowspan="2"></td><td colspan="2">Condition Adjusted Accuracy</td><td colspan="2">Conditional Harm</td></tr><tr><td>∆(pp)</td><td>q</td><td>∆(pp)</td><td>q</td></tr><tr><td>Qwen</td><td>MELD</td><td>RandA</td><td>-7.59</td><td>.0170</td><td>8.26</td><td>.0030</td></tr><tr><td>Qwen</td><td>MELD</td><td>RandA/V</td><td>-8.06</td><td>.0105</td><td>9.11</td><td>.0015</td></tr><tr><td>Baichuan</td><td>IEMOCAP</td><td>RandA/V</td><td>-10.61</td><td>.0030</td><td>12.11</td><td>.0015</td></tr></table>

## C Main RandA/V Replacement-Penalty Robustness

Table S4 reports the primary seed-42 dialogue-cluster analysis for the six comparisons between RandA/V and Full. The additional-mapping analysis is reported separately in Section J. Efects and confidence intervals use the same utteranceweighted estimand. BH-FDR correction is applied across the six model–dataset comparisons. The efect direction is negative in all six pairs, and all six remain statistically significant after accounting for dialogue dependence and BH-FDR correction.

Table S4. Dialogue-cluster robustness for RandA/V replacement penalty relative to Full Accuracy. CIs use 1,000 dialogue-cluster bootstrap resamples; p values use 1,000 dialogue-level sign-flip resamples; q values are BH-FDR adjusted across the six comparisons.  
(a) MELD
<table><tr><td>Model</td><td>∆Acc (pp)</td><td>95% CI</td><td>q</td><td>N / Dlg.</td></tr><tr><td>Qwen</td><td>-4.69</td><td>[-6.16,-3.29]</td><td>.0012</td><td>2604  /  280</td></tr><tr><td>Baichuan</td><td>-3.72</td><td>[-5.70,-1.80]</td><td>.0012</td><td>2605 / 280</td></tr><tr><td>MiniCPM</td><td>-2.81</td><td>[-4.12,-1.46]</td><td>.0012</td><td>2598 / 280</td></tr></table>

(b) IEMOCAP
<table><tr><td>Model</td><td>∆Acc (pp)</td><td>95% CI</td><td>q</td><td>N / Dlg.</td></tr><tr><td>Qwen</td><td>-5.00</td><td>[-6.02,-3.98]</td><td>.0012</td><td>7344 /  151</td></tr><tr><td>Baichuan</td><td>-6.21</td><td>[-7.97,-4.32]</td><td>.0012</td><td>7365 /  151</td></tr><tr><td>MiniCPM</td><td>-1.24</td><td>[-2.07,-0.49]</td><td>.0070</td><td>7260 / 151</td></tr></table>

## D Behavioral Metric Robustness and Benefit/Harm

Table S5 reports Macro-F1 and Balanced Accuracy from the same stored predic tions used in the main behavioral audit. These metrics are robustness diagnostics rather than additional primary endpoints.

Table S5. Macro-F1 and Balanced Accuracy in the primary evaluation (source-mapping seed 42). Both metrics use all expected targets, retaining invalid and OOM outcomes as incorrect predictions. MF1=Macro-F1; BA=Balanced Accuracy (0–1 scale).  
(a) MELD
<table><tr><td>Model</td><td>Metric</td><td>Text</td><td>T+A</td><td>T+V</td><td>Full</td><td>RandA</td><td></td><td>RandV RandA/V</td></tr><tr><td>Qwen</td><td>MF1</td><td>.462</td><td>.460</td><td>.435</td><td>.449</td><td>.389</td><td>.447</td><td>.378</td></tr><tr><td>Qwen</td><td>BA</td><td>.440</td><td>.424</td><td>.398</td><td>.408</td><td>.356</td><td>.406</td><td>.346</td></tr><tr><td>Baichuan</td><td>MF1</td><td>.314</td><td>.352</td><td>.283</td><td>.296</td><td>.263</td><td>.269</td><td>.246</td></tr><tr><td>Baichuan</td><td>BA</td><td>.365</td><td>.418</td><td>.325</td><td>.345</td><td>.308</td><td>.319</td><td>.291</td></tr><tr><td>MiniCPM</td><td>MF1</td><td>.367</td><td>.389</td><td>.363</td><td>.389</td><td>.382</td><td>.390</td><td>.361</td></tr><tr><td>MiniCPM</td><td>BA</td><td>.427</td><td>.455</td><td>.416</td><td>.446</td><td>.436</td><td>.441</td><td>.414</td></tr></table>

(b) IEMOCAP
<table><tr><td>Model</td><td>Metric</td><td>Text</td><td>T+A</td><td>T+V</td><td>Full</td><td></td><td></td><td>RandA RandV RandA/V</td></tr><tr><td>Qwen</td><td>MF1</td><td>.450</td><td>.471</td><td>.439</td><td>.464</td><td>.420</td><td>.458</td><td>.406</td></tr><tr><td>Qwen</td><td>BA</td><td>.475</td><td>.500</td><td>.467</td><td>.494</td><td>.450</td><td>.482</td><td>.431</td></tr><tr><td>Baichuan</td><td>MF1</td><td>.441</td><td>.489</td><td>.443</td><td>.481</td><td>.440</td><td>.460</td><td>.418</td></tr><tr><td>Baichuan</td><td>BA</td><td>.448</td><td>.512</td><td>.471</td><td>.520</td><td>.470</td><td>.491</td><td>.444</td></tr><tr><td>MiniCPM</td><td>MF1</td><td>.406</td><td>.443</td><td>.417</td><td>.432</td><td>.422</td><td>.426</td><td>.416</td></tr><tr><td>MiniCPM</td><td>BA</td><td>.436</td><td>.469</td><td>.453</td><td>.467</td><td>.457</td><td>.459</td><td>.449</td></tr></table>

Table S6 reports genuine-modality Benefit and Harm relative to Text. Benefit is the fraction of Text-incorrect samples corrected by the added genuine modality; Harm is the fraction of Text-correct samples made incorrect.

Table S6. Genuine-modality Benefit and Harm relative to Text. Each entry shows percentage and numerator/denominator. Rates are computed over common-valid Text/condition pairs.  
(a) MELD
<table><tr><td>Model</td><td>Condition</td><td>Benefit %  $( n / d )$ </td><td>Harm %  $( n / d )$ </td></tr><tr><td rowspan="3">Qwen</td><td>T+A</td><td>17.05 (169/991)</td><td>8.36 (135/1615)</td></tr><tr><td>T+V</td><td>13.71 (136/992)</td><td>9.80 (158/1613)</td></tr><tr><td>Full</td><td>19.56 (194/992)</td><td>10.42 (168/1613)</td></tr><tr><td rowspan="3">Baichuan</td><td>T+A</td><td>18.38 (304/1654)</td><td>20.61 (197/956)</td></tr><tr><td>T+V</td><td>18.29 (302/1651)</td><td>30.71 (293/954)</td></tr><tr><td>Full</td><td>18.63 (308/1653)</td><td>32.08 (306/954)</td></tr><tr><td rowspan="3">MiniCPM</td><td>T+A</td><td>9.68 (143/1478)</td><td>12.09 (136/1125)</td></tr><tr><td>T+V</td><td>9.60 (141/1469)</td><td>15.09 (169/1120)</td></tr><tr><td>Full</td><td>11.30 (167/1478)</td><td>11.67 (131/1123)</td></tr></table>

(b) IEMOCAP
<table><tr><td>Model</td><td>Condition</td><td>Benefit %  $( n / d )$ </td><td>Harm %  $( n / d )$ </td></tr><tr><td rowspan="3">Qwen</td><td>T+A</td><td>14.96 (548/3663)</td><td>11.63 (427/3672)</td></tr><tr><td>T+V</td><td>10.51 (385/3663)</td><td>13.67 (502/3673)</td></tr><tr><td>Full</td><td>15.91 (583/3665)</td><td>14.69 (540/3675)</td></tr><tr><td rowspan="3">Baichuan</td><td>T+A</td><td>31.83 (1259/3955)</td><td>27.88 (953/3418)</td></tr><tr><td>T+V</td><td>29.59 (1171/3957)</td><td>35.92 (1227/3416)</td></tr><tr><td>Full</td><td>34.12 (1350/3957)</td><td>33.21 (1134/3415)</td></tr><tr><td rowspan="3">MiniCPM</td><td>T+A</td><td>14.49 (586/4044)</td><td>9.74 (324/3326)</td></tr><tr><td>T+V</td><td>11.08 (445/4016)</td><td>10.96 (363/3312)</td></tr><tr><td>Full</td><td>13.24 (535/4040)</td><td>10.03 (333/3321)</td></tr></table>

## E Full Source-Label Conflict Sensitivity Analysis

The broad post-hoc sensitivity analysis evaluates all 18 combinations of model, dataset, and mismatch condition after target-label standardization. For IEMO-CAP, source cases labeled exactly xxx are excluded before Same/Conflict grouping; all other source labels are retained. To preserve readability, MELD and IEMO-CAP are reported as two panels. The q columns use separate 18-test BH-FDR families for adjusted Accuracy and Conditional Harm. Table S7 reports the complete 18 label-adjusted source-conflict comparisons.

Table S7. All 18 label-adjusted source-conflict comparisons. Efects are Conflict–Same after target-label standardization.  
(a) MELD
<table><tr><td rowspan="2">Model</td><td rowspan="2">Condition</td><td colspan="3">Adjusted Accuracy</td><td colspan="3">Conditional Harm</td></tr><tr><td>Effect (pp)</td><td>95% CI</td><td>q</td><td>Effect (pp)</td><td>95% CI</td><td>q</td></tr><tr><td rowspan="3">Qwen</td><td>RandA</td><td>-7.59</td><td>[-12.17,-3.26]</td><td>.0090</td><td>+8.26</td><td>[+5.21,+11.37]</td><td>.0036</td></tr><tr><td>RandV</td><td>-2.49</td><td>[-7.05,+2.17]</td><td>.3480</td><td>+0.92</td><td>[-2.34,+3.77]</td><td>.5485</td></tr><tr><td>RandA/V</td><td>-8.06</td><td>[-12.19,-3.80]</td><td>.0108</td><td>+9.11</td><td>[+5.39,+12.95]</td><td>.0036</td></tr><tr><td rowspan="3">Baichuan</td><td>RandA</td><td>-2.23</td><td>[-6.51,+1.69]</td><td>.3480</td><td>-2.44</td><td>[-9.59,+4.76]</td><td>.5204</td></tr><tr><td>RandV</td><td>-2.96</td><td>[-6.87,+1.10]</td><td>.2479</td><td>+2.19</td><td>[-4.45,+8.99]</td><td>.5485</td></tr><tr><td>RandA/V</td><td>-4.82</td><td>[-8.98,-1.04]</td><td>.0877</td><td>+4.08</td><td>[-3.16,+10.76]</td><td>.4177</td></tr><tr><td rowspan="3">MiniCPM</td><td>RandA</td><td>-5.35</td><td>[-10.06,-0.63]</td><td>.0877</td><td>+1.98</td><td>[-2.94,+6.03]</td><td>.5204</td></tr><tr><td>RandV</td><td>-4.81</td><td>[-9.29,-0.24]</td><td>.1239</td><td>+2.07</td><td>[-1.38,+5.23]</td><td>.4855</td></tr><tr><td>RandA/V</td><td>-4.16</td><td>[-8.66,+0.49]</td><td>.1780</td><td>+4.67</td><td>[+0.32,+8.92]</td><td>.1782</td></tr></table>

(b) IEMOCAP
<table><tr><td rowspan="2">Model</td><td rowspan="2">Condition</td><td colspan="3">Adjusted Accuracy</td><td colspan="3">Conditional Harm</td></tr><tr><td>Effect (pp)</td><td>95% CI</td><td>q</td><td>Effect (pp)</td><td>95% CI</td><td>q</td></tr><tr><td rowspan="3">Qwen</td><td>RandA</td><td>-3.70</td><td>[-6.86,-0.75]</td><td>.0629</td><td>+5.76</td><td>[+2.65,+8.80]</td><td>.0051</td></tr><tr><td>RandV</td><td>-2.22</td><td>[-5.14,+0.84]</td><td>.2434</td><td>+3.00</td><td>[+1.03,+4.91]</td><td>.0427</td></tr><tr><td>RandA/V</td><td>-5.62</td><td>[-8.55,-2.57]</td><td>.0090</td><td>+8.03</td><td>[+5.22,+11.01]</td><td>.0036</td></tr><tr><td rowspan="4">Baichuan</td><td>RandA</td><td>-6.59</td><td>[-9.81,-3.27]</td><td>.0090</td><td>+8.19</td><td>[+4.03,+12.32]</td><td>.0036</td></tr><tr><td>RandV</td><td>-2.33</td><td>[-5.57,+0.76]</td><td>.2434</td><td>+4.19</td><td>[+0.61,+7.43]</td><td>.0899</td></tr><tr><td>RandA/V</td><td>-10.61</td><td>[-13.93,-7.40]</td><td>.0090</td><td>+12.11</td><td>[+7.65,+16.37]</td><td>.0036</td></tr><tr><td>RandA</td><td>-1.56</td><td>[-4.49,+1.64]</td><td>.3480</td><td>+3.28</td><td>[+0.18,+6.19]</td><td>.0659</td></tr><tr><td rowspan="2">MiniCPM</td><td>RandV</td><td>+0.61</td><td>[-2.70,+3.75]</td><td>.7193</td><td>+1.98</td><td>[-0.61,+4.17]</td><td>.2652</td></tr><tr><td>RandA/V</td><td>-2.46</td><td>[-5.77,+0.84]</td><td>.2191</td><td>+5.36</td><td>[+2.52,+7.73]</td><td>.0051</td></tr></table>

## F Supplementary Emotion-Level Results

Table S8 provides the complete valence-group summary used for the descriptive analysis in Sec. 4.3. Full accuracy and Audio Utility are class-balanced within each valence group; PCR and Conditional Harm are sample-level rates within the group. PCR uses common-valid Full/RandA pairs; Conditional Harm uses all Full-correct targets, counting invalid/OOM RandA outcomes as incorrect. These values are descriptive and are not used to claim a universal positive-versusnegative diference.

Table S8. Emotion-level descriptive analysis by valence group. Full Accuracy and Audio Utility are class-balanced within each group; PCR and Conditional Harm are sample-level rates. Audio Utility is T+A minus Text.  
(a) MELD
<table><tr><td>Model</td><td>Group</td><td>Full Acc. (%)</td><td>Audio Utility (pp)</td><td>RandA PCR (%)</td><td>Cond. Harm (%)</td></tr><tr><td rowspan="2">Qwen</td><td>Positive</td><td>41.04</td><td>+5.72</td><td>21.45</td><td>30.91</td></tr><tr><td>Negative</td><td>29.06</td><td>-5.25</td><td>20.06</td><td>21.76</td></tr><tr><td rowspan="2">Baichuan</td><td>Positive</td><td>86.07</td><td>-1.74</td><td>10.75</td><td>7.80</td></tr><tr><td>Negative</td><td>17.26</td><td>+4.20</td><td>29.49</td><td>35.17</td></tr><tr><td rowspan="2">MiniCPM</td><td>Positive</td><td>81.09</td><td>+2.24</td><td>7.48</td><td>4.91</td></tr><tr><td>Negative</td><td>30.24</td><td>+4.77</td><td>16.92</td><td>14.53</td></tr></table>

(b) IEMOCAP
<table><tr><td>Model</td><td>Group</td><td>Full Acc. (%)</td><td>Audio Utility (pp)</td><td>(%)</td><td>RandA PCR Cond. Harm (%)</td></tr><tr><td rowspan="2">Qwen</td><td>Positive</td><td>37.32</td><td>+4.08</td><td>14.71</td><td>17.98</td></tr><tr><td>Negative</td><td>46.52</td><td>+0.59</td><td>17.68</td><td>20.38</td></tr><tr><td rowspan="2">Baichuan</td><td>Positive</td><td>47.60</td><td>+0.91</td><td>28.70</td><td>22.95</td></tr><tr><td>Negative</td><td>53.76</td><td>+8.46</td><td>30.99</td><td>27.31</td></tr><tr><td rowspan="2">MiniCPM</td><td>Positive</td><td>41.65</td><td>+0.47</td><td>11.42</td><td>8.94</td></tr><tr><td>Negative</td><td>46.68</td><td>+7.69</td><td>12.25</td><td>9.59</td></tr></table>

## G Direct Audio-Edge Intervention Statistics

The main mechanistic confirmation uses the primary RandA-Harm cohort, which excludes RandA-Harm discovery dialogues (112 nominal samples; 106 causaleligible samples from 76 dialogues). The pooled RandA-Harm causal-eligible robustness set contains 193 samples. Tables S9–S11 report the confirmatory score-level efects, prespecified primary tests, and Stable-Correct control results, respectively.

Table S9. Confirmatory score-level direct audio-edge intervention results (N = 106). Efects are utterance-weighted Audio KO minus Video Sham diferences; CIs are 95% dialogue-cluster bootstrap intervals and q values are secondary BH-FDR adjusted.  
(a) Gold-label score recovery
<table><tr><td>Block</td><td>Score effect (nats)</td><td>95% CI</td><td>q</td></tr><tr><td>Early L0–8</td><td>0.0062</td><td> $\left[ - 0 . 0 0 0 3 , + 0 . 0 1 3 1 \right]$ </td><td>.0771</td></tr><tr><td>Middle L9–18</td><td>0.0690</td><td> $\left[ + 0 . 0 3 8 7 , + 0 . 1 0 5 4 \right]$ </td><td>1.5e-5</td></tr><tr><td>Late L19–27</td><td>0.1189</td><td> $\left[ + 0 . 0 8 0 4 , + 0 . 1 6 2 3 \right]$ </td><td>1.5e-5</td></tr><tr><td>L19–23 (candidate)</td><td>0.1062</td><td> $\left[ + 0 . 0 6 9 6 , + 0 . 1 4 8 0 \right]$ </td><td>1.5e-5</td></tr></table>

(b) Gold-vs-best-candidate margin recovery
<table><tr><td>Block</td><td>Margin effect (nats)</td><td>95% CI</td><td>q</td></tr><tr><td>Early L0-8</td><td>0.0065</td><td> $\left[ - 0 . 0 0 4 2 , + 0 . 0 1 7 8 \right]$ </td><td>.2298</td></tr><tr><td>Middle L9–18</td><td>0.1014</td><td> $[ + 0 . 0 5 7 3 , + 0 . 1 5 4 6 ]$ </td><td>1.5e-5</td></tr><tr><td>Late L19–27</td><td>0.1722</td><td> $\left[ + 0 . 1 1 2 0 , + 0 . 2 3 9 3 \right]$ </td><td>1.5e-5</td></tr><tr><td>L19–23 (candidate)</td><td>0.1533</td><td> $\left[ + 0 . 0 9 6 4 , + 0 . 2 1 8 7 \right]$ </td><td>1.5e-5</td></tr></table>

Table S10. Prespecified primary confirmatory tests on the gold-vs-best-candidate margin. p values use dialogue-cluster sign-flip tests and Holm correction.
<table><tr><td>Primary test Effect (nats)</td><td></td><td>Raw p Holm p</td></tr><tr><td>Late vs. 0</td><td>+0.1722</td><td>1.0e-5 3.0e-5</td></tr><tr><td> $_ \mathrm { L a t e \mathrm { ~ - ~ } E a r l y }$ </td><td>+0.1657</td><td>1.0e-5 3.0e-5</td></tr><tr><td> $\mathrm { L a t e \mathrm { ~ - ~ } M i d d l e }$ </td><td>+0.0708</td><td>.00382 .00382</td></tr></table>

Table S11. Stable-Correct confirmatory control. The first efect is Audio KO minus Video Sham within Stable-Correct; the second is the RandA-Harm minus Stable-Correct diference in margin recovery. Efects are in nats; q values are BH-FDR adjusted.
<table><tr><td>Block</td><td colspan="2">Stable-Correct</td><td colspan="3">RandA-Harm - Stable</td></tr><tr><td></td><td>Effect</td><td>q</td><td>Contrast</td><td>95% CI</td><td>q</td></tr><tr><td>Early</td><td>+0.0050</td><td>.2951</td><td>+0.0015</td><td>[-0.0128,+0.0162]</td><td>.8235</td></tr><tr><td>Middle</td><td>+0.0067</td><td>.5131</td><td>+0.0947</td><td>[+0.0446,+0.1506]</td><td>.00092</td></tr><tr><td>Late</td><td>+0.0441</td><td>.0396</td><td>+0.1281</td><td>[+0.0492,+0.2091]</td><td>.00414</td></tr><tr><td>L19–23 (candidate)</td><td>+0.0720</td><td>.00036</td><td>+0.0813</td><td>[+0.0065,+0.1575]</td><td>.05096</td></tr></table>

Stable-Correct confirmatory prediction harm is 0/112 for both Audio KO and Video Sham in all four blocks. A smaller positive score shift appears in later Stable-Correct blocks, but RandA-Harm shows significantly larger efects in Middle and Late. Across the expansion manifest, 3,128 of 3,184 nominal causal pairs are eligible: 52 exclusions arise from 13 canonical–master prediction mismatches propagated across four blocks, and four from one condition in which a matched Video Sham cannot be constructed. No OOM, runtime, or score errors occur.

The original confirmatory split excludes RandA-Harm discovery dialogues. As a post-hoc sensitivity check, additionally excluding dialogues represented in Stable-Correct discovery leaves 84 eligible cases from 60 dialogues. Late margin recovery is +0.1801 nats (95% dialogue-cluster CI [0.1058, 0.2614]); all three sensitivity contrasts remain positive and significant after Holm correction across the three tests (Late versus zero: $p _ { \mathrm { H o l m } } = . 0 0 0 0 3 $ Late minus Early: .00012; Late minus Middle: .02805). This stricter check supplements the original prespecified analysis; its full results are supplied in strict\_discovery\_sensitivity.csv.

## H Reproducibility and Resampling Details

Table S12 summarizes the recorded model/runtime and media-processing setup.

Table S12. Recorded model/runtime and media-processing setup. All runs use batch size 1 and a maximum generation length of 160 tokens.
<table><tr><td>Model</td><td>Checkpoint / recorded runtime</td><td>Decoding and media handling</td></tr><tr><td>Qwen</td><td>PyTorch 2.11.0+cu128</td><td>Sampling disabled; 2 fps; min 4/max Qwen/Qwen2.5-Omni-7B; 768 frames; explicit mono 16 kHz WAV; video-path embedded audio disabled. Sampling disabled; explicit audio and</td></tr><tr><td>Baichuan</td><td>baichuan-inc/Baichuan- Omni-1d5; PyTorch 2.4.0+cu124</td><td>video-frame paths processed separately. The video processor reads visual frames only; embedded MP4 audio is not converted to audio tokens.</td></tr><tr><td>MiniCPM 2_6;</td><td>openbmb/MiniCPM-o- PyTorch 2.3.1+cu121</td><td>Sampling disabled; internal beam size 3; bf16/SDPA; video opened with audio disabled and separate mono 16 kHz audio chunks.</td></tr></table>

The system prompt is: “You are an emotion analysis expert. Please infer emotion labels based on the given video/audio.” The user prompt presents up to ten preceding turns in chronological order together with the target transcript, forbids future context, provides the dataset label set, and requests an Emotion: line followed by a brief Reason: line. The templates share a common structure, with modality-availability instructions adapted to the supplied inputs. Full and all replacement conditions use the same audiovisual instructions; complete templates are included in the accompanying reproducibility files. MELD label order is anger, disgust, sadness, joy, neutral, surprise, fear; IEMOCAP label order is angry, happy, sad, neutral, excited, frustrated. The Qwen/MELD behavioral evaluation uses a pinned checkpoint, Transformers 4.57.6 and SDPA attention; its configuration and hashes accompany the reproducibility records. Pinned checkpoint and runtime records also accompany the primary Baichuan and MiniCPM MELD evaluations; corresponding version records remain incomplete for the primary IEMOCAP runs.

The primary random-source construction uses seed 42; the additional analysis evaluates seeds 43 and 44 under the same cross-dialogue, label-agnostic rules, sharing each mapping across models and replacement conditions. IEMOCAP source selection remains restricted to the target session.

Reference predictions and runtime details for the three-mapping analysis are documented in Section J. For Baichuan and MiniCPM on MELD, all videobearing conditions use a common silent-MP4 re-encoding procedure, with audio supplied separately as mono 16 kHz WAV. Prepared-media hashes are verified across conditions. MiniCPM retains its duration-based audio–video packing; diferences in audio duration can therefore change the supplied frame count.

Resampling families for the primary analyses are defined in Section 3.7. The primary seed-42 replacement analysis uses 1,000 dialogue-cluster bootstrap and 1,000 sign-flip resamples; the broad label-adjusted conflict analysis uses 1,000 bootstrap plus 1,000 within-label permutations; and the mechanistic confirmation uses 10,000 dialogue-cluster bootstrap resamples and 100,000 sign flips, with Holm correction for its three prespecified primary tests. The additional-mapping and failed-output sensitivity analyses use separate 10,000-resample protocols and the testing families specified in Section 3.7 and Sections J and K.

The accompanying machine-readable tables contain all seed-specific efects, confidence intervals, adjusted p values and failure counts. The accompanying reproducibility archive provides inference and analysis code, complete prompt templates, seed-specific replacement manifests and hashes, per-target predictions, invalid/OOM and retry records, environment metadata, and mechanistic cohort, intervention and score records. Raw MELD and IEMOCAP media, dataset transcripts, free-form model responses and model weights are not redistributed; records refer to dataset identifiers.

## I Paired Utility and Mismatch-Specific Interference

Tables S13 and S14 report the primary seed-42 cluster-aware paired analyses used to distinguish useful contribution from active mismatch harm. Utility and mismatch-specific interference are corrected in separate 18-test BH-FDR families. The tables use the same primary seed-42 predictions as the main behavioral results; additional mappings and failed-output analyses are reported separately in Sections J and K.

Table S13. Cluster-aware genuine-modality utility. Efects are condition minus Text Accuracy (pp). The q column uses one 18-test BH-FDR family.  
(a) MELD
<table><tr><td>Model</td><td>Comparison</td><td>∆Acc</td><td>95% CI</td><td>p</td><td>q</td><td>N</td></tr><tr><td rowspan="3">Qwen</td><td>T+A-Text</td><td>+1.30</td><td>[-0.14,+2.66]</td><td>.0649</td><td>.1169</td><td>2606</td></tr><tr><td>T+V-Text</td><td>-0.84</td><td>[-1.96,+0.34]</td><td>.1918</td><td>.2656</td><td>2605</td></tr><tr><td>Full-Text</td><td>+1.00</td><td>[-0.41,+2.39]</td><td>.1888</td><td>.2656</td><td>2605</td></tr><tr><td rowspan="3">Baichuan</td><td>T+A-Text</td><td>+4.10</td><td>[+2.38,+5.70]</td><td>.0010</td><td>.0045</td><td>2610</td></tr><tr><td>T+V-Text</td><td>+0.35</td><td>[-1.62,+2.47]</td><td>.7722</td><td>.8177</td><td>2605</td></tr><tr><td>Full–Text</td><td>+0.08</td><td>[-1.92,+2.12]</td><td>.9720</td><td>.9720</td><td>2607</td></tr><tr><td rowspan="3">MiniCPM</td><td>T+A-Text</td><td>+0.27</td><td>[-1.16,+1.59]</td><td>.7652</td><td>.8177</td><td>2603</td></tr><tr><td>T+V-Text</td><td>-1.08</td><td>[-2.43,+0.31]</td><td>.1349</td><td>.2207</td><td>2589</td></tr><tr><td>Full–Text</td><td>+1.38</td><td>[-0.04,+2.75]</td><td>.0500</td><td>.0999</td><td>2601</td></tr></table>

(b) IEMOCAP
<table><tr><td>Model</td><td>Comparison</td><td>∆Acc</td><td>95% CI</td><td>p</td><td>q</td><td>N</td></tr><tr><td rowspan="3">Qwen</td><td>T+A-Text</td><td>+1.65</td><td>[+0.56,+2.68]</td><td>.0020</td><td>.0072</td><td>7335</td></tr><tr><td>T+V-Text</td><td>-1.59</td><td>[-2.78,-0.39]</td><td>.0110</td><td>.0330</td><td>7336</td></tr><tr><td>Full-Text</td><td>+0.59</td><td>[-0.58,+1.76]</td><td>.3556</td><td>.4573</td><td>7340</td></tr><tr><td rowspan="3">Baichuan</td><td>T+A-Text</td><td>+4.15</td><td>[+2.19,+6.18]</td><td>.0010</td><td>.0045</td><td>7373</td></tr><tr><td>T+V-Text</td><td>-0.76</td><td>[-3.18,+1.51]</td><td>.5275</td><td>.6330</td><td>7373</td></tr><tr><td>Full-Text</td><td>+2.93</td><td>[+0.50,+5.28]</td><td>.0170</td><td>.0437</td><td>7372</td></tr><tr><td rowspan="3">MiniCPM</td><td>T+A-Text</td><td>+3.55</td><td>[+2.51,+4.59]</td><td>.0010</td><td>.0045</td><td>7370</td></tr><tr><td>T+V-Text</td><td>+1.12</td><td>[+0.13,+2.07]</td><td>.0250</td><td>.0562</td><td>7328</td></tr><tr><td>Full-Text</td><td>+2.74</td><td>[+1.86,+3.60]</td><td>.0010</td><td>.0045</td><td>7361</td></tr></table>

Table S14. Cluster-aware mismatch-specific interference. Efects are Random minus the corresponding omission baseline Accuracy (pp): RandA–T+V, RandV–T+A, and RandA/V–Text. Negative efects indicate active harm beyond modality omission. The q column uses a separate 18-test BH-FDR family.  
(a) MELD
<table><tr><td>Model</td><td>Comparison</td><td>∆Acc</td><td>95% CI</td><td>p</td><td>q</td><td>N</td></tr><tr><td rowspan="3">Qwen</td><td>RandA-T+V</td><td>-2.00</td><td>[-2.99,-0.99]</td><td>.0010</td><td>.0022</td><td>2606</td></tr><tr><td>RandV-T+A</td><td>-0.88</td><td>[-2.04,+0.16]</td><td>.1309</td><td>.1386</td><td>2605</td></tr><tr><td>RandA/V-Text</td><td>-3.72</td><td>[-5.00,-2.33]</td><td>.0010</td><td>.0022</td><td>2605</td></tr><tr><td rowspan="3">Baichuan</td><td>RandA-T+V</td><td>-2.19</td><td>[-3.48,-0.89]</td><td>.0040</td><td>.0065</td><td>2605</td></tr><tr><td>RandV-T+A</td><td>-7.55</td><td>[-9.34,-5.68]</td><td>.0010</td><td>.0022</td><td>2608</td></tr><tr><td>RandA/V-Text</td><td>-3.68</td><td>[-5.57,-1.91]</td><td>.0010</td><td>.0022</td><td>2608</td></tr><tr><td rowspan="3">MiniCPM</td><td>RandA-T+V</td><td>+1.31</td><td>[+0.34,+2.37]</td><td>.0150</td><td>.0207</td><td>2591</td></tr><tr><td>RandV-T+A</td><td>+1.23</td><td>[+0.04,+2.34]</td><td>.0480</td><td>.0599</td><td>2604</td></tr><tr><td>RandA/V-Text</td><td>-1.39</td><td>[-2.91,+0.04]</td><td>.0729</td><td>.0820</td><td>2595</td></tr></table>

(b) IEMOCAP
<table><tr><td>Model</td><td>Comparison</td><td>∆Acc</td><td>95% CI</td><td>p</td><td>q</td><td>N</td></tr><tr><td rowspan="3">Qwen</td><td>RandA-T+V</td><td>-1.66</td><td>[-2.30,-1.03]</td><td>.0010</td><td>.0022</td><td>7344</td></tr><tr><td>RandV-T+A</td><td>-1.79</td><td>[-2.52,-1.04]</td><td>.0010</td><td>.0022</td><td>7337</td></tr><tr><td>RandA/V-Text</td><td>-4.39</td><td>[-5.69,-3.07]</td><td>.0010</td><td>.0022</td><td>7340</td></tr><tr><td rowspan="3">Baichuan</td><td>RandA-T+V</td><td>-0.42</td><td>[-1.38,+0.63]</td><td>.4396</td><td>.4396</td><td>7373</td></tr><tr><td>RandV-T+A</td><td>-3.14</td><td>[-4.44,-1.77]</td><td>.0010</td><td>.0022</td><td>7351</td></tr><tr><td>RandA/V-Text</td><td>-3.27</td><td>[-5.42,-1.26]</td><td>.0030</td><td>.0054</td><td>7373</td></tr><tr><td rowspan="3">MiniCPM</td><td>RandA-T+V</td><td>+0.81</td><td>[+0.04,+1.64]</td><td>.0500</td><td>.0599</td><td>7276</td></tr><tr><td>RandV-T+A</td><td>-1.09</td><td>[-1.86,-0.28]</td><td>.0120</td><td>.0180</td><td>7354</td></tr><tr><td>RandA/V-Text</td><td>+1.56</td><td>[+0.65,+2.49]</td><td>.0030</td><td>.0054</td><td>7261</td></tr></table>

## J Additional-Mapping Robustness

This section reports behavioral robustness across source-mapping seeds 42, 43 and 44. The primary seed-42 predictions are shared with the main behavioral evaluation; the reference configuration is described in Section J.1. Efects are reported separately by mapping, with dialogue-cluster inference. Under each failed-output policy, BH-FDR is applied separately to 18 replacement and 54 omission comparisons, using 10,000 bootstrap and sign-flip resamples. Repeated targets across mappings are not pooled as independent observations.

Table S15. Three-mapping RandA/V-minus-Full Accuracy efects on common-valid pairs. Efects and confidence intervals are in pp; q uses the separate 18-test BH-FDR family. Primary reference predictions are described in Section J.1.
<table><tr><td>Model</td><td>Seed</td><td>Effect</td><td>95% CI</td><td>q</td><td>N</td></tr><tr><td colspan="6">MELD</td></tr><tr><td rowspan="3">Qwen</td><td>42</td><td>-4.69</td><td>[-6.09,-3.26]</td><td>.00018</td><td>2604</td></tr><tr><td>43</td><td>-4.46</td><td>[-5.87,-3.04]</td><td>.00018</td><td>2603</td></tr><tr><td>44</td><td>-4.76</td><td>[-6.25,-3.25]</td><td>.00018</td><td>2604</td></tr><tr><td rowspan="3">Baichuan</td><td>42</td><td>-3.72</td><td>[-5.70,-1.77]</td><td>.00055</td><td>2605</td></tr><tr><td>43</td><td>-3.22</td><td>[-5.05,-1.43]</td><td>.00096</td><td>2605</td></tr><tr><td>44</td><td>-3.15</td><td>[-4.92,-1.37]</td><td>.00090</td><td>2604</td></tr><tr><td rowspan="3">MiniCPM</td><td>42</td><td>-2.81</td><td>[-4.22,-1.44]</td><td>.00045</td><td>2598</td></tr><tr><td>43</td><td>-2.27</td><td>[-3.49,-1.06]</td><td>.00045</td><td>2596</td></tr><tr><td>44</td><td>-2.24</td><td>[-3.56,-0.93]</td><td>.00101</td><td>2595</td></tr><tr><td colspan="6">IEMOCAP</td></tr><tr><td rowspan="3">Qwen</td><td>42</td><td>-5.00</td><td>[-5.97,-4.07]</td><td>.00018</td><td>7344</td></tr><tr><td>43</td><td>-5.04</td><td>[-5.94,-4.16]</td><td>.00018</td><td>7341</td></tr><tr><td>44</td><td>-4.81</td><td>[-5.81,-3.83]</td><td>.00018</td><td>7346</td></tr><tr><td rowspan="3">Baichuan</td><td>42</td><td>-6.21</td><td>[-7.91,-4.45]</td><td>.00018</td><td>7365</td></tr><tr><td>43</td><td>-6.03</td><td>[-7.82,-4.19]</td><td>.00018</td><td>7368</td></tr><tr><td>44</td><td>-6.11</td><td>[-7.85,-4.37]</td><td>.00018</td><td>7368</td></tr><tr><td rowspan="3">MiniCPM</td><td>42</td><td>-1.24</td><td>[-2.03,-0.44]</td><td>.00233</td><td>7260</td></tr><tr><td>43</td><td>-1.17</td><td>[-1.93,-0.41]</td><td>.00260</td><td>7271</td></tr><tr><td>44</td><td>-1.55</td><td>[-2.30,-0.78]</td><td>.00018</td><td>7289</td></tr></table>

All 18 efects are negative and have q < .05 under both failed-output policies. CIs are unadjusted percentile-bootstrap intervals; significance is based on BH-FDR-adjusted sign-flip tests.

Table S16. Omission-baseline efects across all three mappings on common-valid pairs. Min/max efects are in pp. Significant negative and positive counts use $q < . 0 5$ in the separate 54-test BH-FDR family.
<table><tr><td>Model</td><td>Comparison</td><td>Min. Max.</td><td></td><td>Sig. neg. Sig. pos.</td></tr><tr><td colspan="5">MELD</td></tr><tr><td>Qwen</td><td>RandA-T+V RandV-T+A RandA/V-Text</td><td>-2.00 -0.88 -3.80</td><td>-1.69 -0.54 -3.53</td><td>3/3 0/3 0/3 0/3 3/3 0/3</td></tr><tr><td>Baichuan</td><td>RandA-T+V RandV-T+A RandA/V-Text</td><td>-2.19 -7.55 -3.68</td><td>-1.19 -5.60 -3.03</td><td>2/3 0/3 3/3 0/3 3/3 0/3</td></tr><tr><td>MiniCPM</td><td>RandA-T+V RandV-T+A RandA/V-Text</td><td>+1.12 -0.38 -1.39</td><td>+2.28 +1.23 -0.89</td><td>0/3 3/3 0/3 0/3 0/3 0/3</td></tr><tr><td colspan="5">IEMOCAP</td></tr><tr><td>Qwen</td><td>RandA-T+V RandV-T+A RandA/V-Text</td><td>-1.99 -1.92 -4.46</td><td>-1.42 3/3 -1.79 3/3 -4.21 3/3 0/3</td><td>0/3 0/3 0/3</td></tr><tr><td>Baichuan</td><td>RandA-T+V RandV-T+A RandA/V-Text</td><td>-1.00 -3.14 -3.27</td><td>-0.42 -2.66 -3.10</td><td>0/3 3/3 0/3 3/3 0/3</td></tr><tr><td>MiniCPM</td><td>RandA-T+V RandV-T+A RandA/V-Text</td><td>+0.63 +0.91 -1.19 -0.96 +1.19 +1.62</td><td>0/3 3/3 0/3</td><td>1/3 0/3 3/3</td></tr></table>

Full seed-specific efects, confidence intervals, sample sizes and q values for both policies are provided in the accompanying machine-readable tables. Positive efects are not counted as harm.

## J.1 Evaluation References and Runtime Details

Within each model, the three-mapping analysis uses the primary Text, T+A, T+V and Full reference predictions. The primary MELD evaluations and additionalmapping runs use the recorded model-specific inference configurations and mediaprocessing procedures.

Table S17. Recorded runtime versions for the primary MELD evaluations and additional-mapping runs.
<table><tr><td>Model</td><td>PyTorch</td><td>Transformers</td></tr><tr><td>Qwen</td><td>2.11.0+cu128</td><td>4.57.6</td></tr><tr><td>Baichuan</td><td>2.4.0+cu124</td><td>4.45.0</td></tr><tr><td>MiniCPM</td><td>2.3.1</td><td>4.44.2</td></tr></table>

Model identifiers, mapping hashes, script hashes and runtime records accompany the machine-readable results. The additional mappings evaluate behavioral robustness; mechanistic results are from the Qwen/MELD cohort described in Section 4.4.

## K Failed-Output Sensitivity

We distinguish common-valid estimates from all-target estimates retaining invalid and OOM outcomes as incorrect. The first estimates performance conditional on both outputs being valid; the second also reflects failures to produce a valid label. Efects are utterance-weighted and inference is clustered by dialogue. The accompanying CSV tables report both policies’ efects, confidence intervals, p, q, sample sizes and exclusions.

For the primary seed-42 predictions, the 10,000-resample sensitivity analysis uses families of 18 utility, 18 omission and six replacement comparisons. The primary tables in Sections C and I use 1,000 resamples with the same underlying predictions. For the three-mapping analysis, families have 18 replacement and 54 omission tests under each policy. BH-FDR adjustment is performed separately within each testing family and failed-output policy. The primary MELD and additional-mapping runs permit one OOM retry with unchanged inputs; final outcomes and retry records are retained rather than deleting failed targets.

Table S18. Primary seed-42 MELD failed-output counts. Every condition has 2,610 expected targets. Invalid and OOM are disjoint final outcomes; other failed-outcome counts are zero.
<table><tr><td>Model</td><td>Condition</td><td>Valid</td><td>Invalid OOM</td></tr><tr><td rowspan="7">Qwen</td><td>Text</td><td>2607</td><td>3</td></tr><tr><td>T+A</td><td>2607</td><td>0 3 0</td></tr><tr><td>T+V</td><td>2606</td><td>2 2</td></tr><tr><td>Full</td><td>2606</td><td>2 2</td></tr><tr><td>RandA</td><td>2606</td><td>2 2</td></tr><tr><td>RandV</td><td>2606</td><td>2 2</td></tr><tr><td>RandA/V</td><td>2606</td><td>2 2</td></tr><tr><td>Text T+A</td><td>2610</td><td>0</td><td>0</td></tr><tr><td>Baichuan</td><td></td><td>2610</td><td>0 0</td></tr><tr><td>T+V</td><td>2605</td><td>0</td><td>5</td></tr><tr><td>Full</td><td>2607</td><td>0</td><td>3</td></tr><tr><td>RandA</td><td>2605</td><td>0</td><td>5</td></tr><tr><td>RandV</td><td>2608</td><td>0</td><td>2</td></tr><tr><td>RandA/V</td><td>2608</td><td>0</td><td>2</td></tr><tr><td rowspan="8">MiniCPM</td><td></td><td></td><td></td><td></td></tr><tr><td>Text T+A</td><td>2604</td><td>6</td><td>0 0</td></tr><tr><td>T+V</td><td>2608 2595</td><td>2 13</td><td>2</td></tr><tr><td>Full</td><td>2607</td><td>1</td><td>2</td></tr><tr><td>RandA</td><td>2604</td><td>2</td><td>4</td></tr><tr><td>RandV</td><td>2606</td><td>0</td><td>4</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>RandA/V</td><td>2601</td><td>7</td><td>2</td></tr></table>

Table S19. Primary seed-42 IEMOCAP failed-output counts. Every condition has 7,380 expected targets. Invalid and OOM are disjoint final outcomes; other failed-outcome counts are zero.
<table><tr><td>Model</td><td>Condition</td><td>Valid</td><td>Invalid OOM</td></tr><tr><td rowspan="7">Qwen</td><td>Text</td><td>7347</td><td>33</td></tr><tr><td>T+A</td><td>7343 37</td><td>0 0</td></tr><tr><td>T+V</td><td>7349 31</td><td>0</td></tr><tr><td>Full</td><td>7355</td><td>25 0</td></tr><tr><td>RandA</td><td>7356 24</td><td>0</td></tr><tr><td>RandV</td><td>7346</td><td>34 0</td></tr><tr><td>RandA/V</td><td>7354 26</td><td>0</td></tr><tr><td>Text</td><td>7380</td><td>0</td><td>0</td></tr><tr><td rowspan="6">Baichuan</td><td>T+A</td><td>7373</td><td>7</td><td>0</td></tr><tr><td>T+V</td><td>7373</td><td>0</td><td>7</td></tr><tr><td>Full</td><td>7372</td><td>0</td><td>8</td></tr><tr><td>RandA</td><td>7373</td><td>0</td><td>7</td></tr><tr><td>RandV</td><td>7358</td><td>2</td><td>20</td></tr><tr><td>RandA/V</td><td>7373</td><td>0</td><td>7</td></tr><tr><td rowspan="8">MiniCPM</td><td>Text</td><td></td><td></td><td></td></tr><tr><td>T+A</td><td>7371</td><td>9</td><td>0 0</td></tr><tr><td>T+V</td><td>7376 7336</td><td>4 44</td><td>0</td></tr><tr><td>Full</td><td>7369</td><td>11</td><td>0</td></tr><tr><td>RandA</td><td>7320</td><td>60</td><td>0</td></tr><tr><td>RandV</td><td></td><td></td><td>0</td></tr><tr><td></td><td>7357</td><td>23</td><td></td></tr><tr><td>RandA/V</td><td>7270</td><td>110</td><td>0</td></tr></table>

Table S20. Additional-mapping failed-output counts. Invalid and OOM are disjoint final outcomes; other failed-outcome counts are zero. OOM counts denote failures remaining after the prespecified single identical-input retry.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Condition</td><td colspan="3">Seed 43</td><td colspan="3">Seed 44</td></tr><tr><td>Valid</td><td>Invalid</td><td>OOM</td><td>Valid</td><td>Invalid</td><td>OOM</td></tr><tr><td></td><td colspan="7">MELD (N = 2, 610 per condition and seed)</td></tr><tr><td rowspan="3">Qwen</td><td>RandA</td><td>2606</td><td>2</td><td>2</td><td>2606</td><td>2</td><td>2</td></tr><tr><td>RandV</td><td>2606</td><td>2</td><td>2</td><td>2606</td><td>2</td><td>2</td></tr><tr><td>RandA/V</td><td>2607</td><td>1</td><td>2</td><td>2607</td><td>1</td><td>2</td></tr><tr><td rowspan="3">Baichuan</td><td>RandA</td><td>2604</td><td>0</td><td>6</td><td>2604</td><td>0</td><td>6</td></tr><tr><td>RandV</td><td>2605</td><td>0</td><td>5</td><td>2606</td><td>1</td><td>3</td></tr><tr><td>RandA/V</td><td>2608</td><td>0</td><td>2</td><td>2607</td><td>0</td><td>3</td></tr><tr><td rowspan="3">MiniCPM</td><td>RandA</td><td>2600</td><td>6</td><td>4</td><td>2601</td><td>5</td><td>4</td></tr><tr><td>RandV</td><td>2605</td><td>1</td><td>4</td><td>2606</td><td>0</td><td>4</td></tr><tr><td>RandA/V</td><td>2599</td><td>9</td><td>2</td><td>2598</td><td>10</td><td>2</td></tr><tr><td colspan="7">IEMOCAP (N = 7, 380 per condition and seed)</td><td></td></tr><tr><td rowspan="3">Qwen</td><td>RandA</td><td>7360</td><td>20</td><td>0</td><td>7360</td><td>20</td><td>0</td></tr><tr><td>RandV</td><td>7344</td><td>36</td><td>0</td><td>7345</td><td>35</td><td>0</td></tr><tr><td>RandA/V</td><td>7352</td><td>28</td><td>0</td><td>7357</td><td>23</td><td>0</td></tr><tr><td rowspan="3">Baichuan</td><td>RandA</td><td>7374</td><td>0</td><td>6</td><td>7374</td><td>0</td><td>6</td></tr><tr><td>RandV</td><td>7372</td><td>4</td><td>4</td><td>7375</td><td>1</td><td>4</td></tr><tr><td>RandA/V</td><td>7376</td><td>0</td><td>4</td><td>7376</td><td>0</td><td>4</td></tr><tr><td rowspan="3">MiniCPM</td><td>RandA</td><td>7318</td><td>62</td><td>0</td><td>7330</td><td>50</td><td>0</td></tr><tr><td>RandV</td><td>7375</td><td>5</td><td>0</td><td>7368</td><td>12</td><td>0</td></tr><tr><td>RandA/V</td><td>7280</td><td>100</td><td>0</td><td>7299</td><td>81</td><td>0</td></tr></table>

Across the 179,820 additional predictions, 179,216 are valid, 519 are invalid and 85 remain OOM. Recovered retries are retained in the per-target retry records.

## K.1 Findings Sensitive to the Failed-Output Policy

Table S21 shows the main boundary cases. All 18 replacement efects are negative under both policies. All 18 are significant both on common-valid pairs and on all targets. Statistical significance is assessed using the specified FDR-adjusted sign-flip tests.

Table S21. MiniCPM policy-sensitive omission comparisons from the three-mapping analysis. Efects are in pp; q uses the separate 54-test omission family within each policy. Excluded is the number omitted from common-valid estimation.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2"></td><td rowspan="2">Seed Comparison</td><td colspan="2">Common-valid</td><td colspan="2">All-target</td><td rowspan="2">Excluded</td></tr><tr><td>Effect</td><td>q</td><td>Effect</td><td>q</td></tr><tr><td>IEMOCAP</td><td>42</td><td>RandA/V-Text</td><td>+1.56</td><td>.00389</td><td>+0.98</td><td>.07573</td><td>119</td></tr><tr><td>IEMOCAP</td><td>43</td><td>RandA/V-Text +1.62</td><td></td><td>.00282</td><td>+1.06</td><td>.04976</td><td>108</td></tr><tr><td>IEMOCAP</td><td>44</td><td>RandA/V-Text +1.19</td><td></td><td>.02640</td><td>+0.79</td><td>.13302</td><td>89</td></tr></table>

In the 54-test common-valid omission family, 43 efects are negative and 11 positive; 32 negative and seven positive efects have $q < . 0 5$ . The all-target family has 43 negative, ten positive and one zero efect; 32 negative and five positive efects have $q < . 0 5$ . MiniCPM/IEMOCAP RandA/V-minus-Text is positive and significant for all three common-valid comparisons, but only seed 43 remains significant on all targets (q = .04976).

For MiniCPM on MELD, the primary seed-42 comparison of RandV relative to T+A has common-valid q = .04962 and all-target q = .07139 within its 18-test omission family. This is distinct from the three-mapping 54-test result; the corresponding 1,000-resample estimate in Section I has q = .0599.

## K.2 Metric Denominators

Accuracy and Weighted F1 in Table 2, and Macro-F1 and Balanced Accuracy in Table S5, are computed over all expected targets for both datasets. Invalid and OOM outcomes contribute no true positives and remain in gold-class support. The supplementary metric-policy table reports values under both all-target and valid-only denominators.