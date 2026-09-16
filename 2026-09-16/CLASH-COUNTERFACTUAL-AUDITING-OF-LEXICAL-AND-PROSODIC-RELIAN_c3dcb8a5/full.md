# CLASH: COUNTERFACTUAL AUDITING OF LEXICAL AND PROSODIC RELIANCE IN SPOKEN SARCASM DETECTION

Qiyang Sun<sup>1,∗</sup> Xudong Li<sup>2,∗</sup> Yupei Li<sup>1,3</sup> Jiabin Xue<sup>4</sup>

Yuhang Dai<sup>5</sup> Jiaming Li<sup>6</sup> Bjorn W. Schuller¨ <sup>1,3</sup>

<sup>1</sup>Imperial College London <sup>2</sup>New York University <sup>3</sup>Technical University of Munich <sup>4</sup>Tencent Inc. <sup>5</sup>Wuhan University <sup>6</sup>Nankai University

## ABSTRACT

Spoken sarcasm detectors may exploit lexical content, prosody, or their interaction, yet conventional evaluation cannot reveal which cues drive their predictions. We introduce CLASH (Controlled Lexical-Acoustic Separation Harness), a bilingual counterfactual diagnostic framework that evaluates each utterance under original, lexical-preserving, prosody-preserving, and approximately neutralised conditions. We evaluate handcrafted acoustic-feature systems, self-supervised learning (SSL) probes, and large audio language models (LALMs) on CMMA and MUStARD. For targetonly Qwen3-Omni, lexical-preserving speech retains a 0.135–0.148 AUROC advantage over prosody-preserving speech after duration balancing, with cluster-bootstrap intervals above zero; alternative lexical resynthesis preserves this advantage. Acoustic interventions shift scores without consistently improving discrimination or changing binary predictions under the evaluated conditions. Context and interaction estimates vary across corpora. These findings distinguish acoustic sensitivity from sarcasm discrimination while exposing duration, identity, and transformation effects.

Index Terms— Spoken sarcasm detection, counterfactual analysis, cue attribution, acoustic sensitivity, spoken language evaluation

## 1. INTRODUCTION

As a task in affective computing [1], sarcasm detection requires recognising intended meanings that differ from, and often oppose, the literal content of an utterance [2]. In speech, this intention may be conveyed through lexical choice, discourse context, vocal delivery, or a combination of these sources [3]. Acoustic studies associate sarcasm with variation in fundamental frequency, intensity, voice quality, and speaking rate [4, 5]. Such associations do not imply that prosody provides a stable or sufficient decision signal. In spontaneous speech, acoustic realisations of irony vary considerably across utterances and speakers [6]. Human judgements also change with discourse context [7], and prosodic marking can weaken when lexical semantics already make sarcastic intent salient [8]. A correct prediction alone therefore does not reveal which cues the detector uses. We address this question by selectively altering lexical content and prosodic delivery and examining the resulting changes in model scores, sarcasm discrimination, and binary predictions.

Automatic spoken sarcasm detection progresses from classifiers based on prosodic, spectral, and contextual descriptors [9] to transfer learning with learned acoustic representations [10]. Multimodal corpora such as MUStARD [11], and CMMA [12] further enable the study of sarcasm in English and Mandarin conversations. Most subsequent work pursues higher predictive performance through feature fusion or incongruity modelling [13]. This direction is useful, but aggregate improvements obtained from audio do not establish that a system uses prosody. Learned speech embeddings can encode lexical and prosodic information within the same representation. Comparing classifiers with and without these embeddings measures the aggregate contribution of speech features, but does not isolate lexical, prosodic, or interaction effects. Likewise, combining information sources does not establish whether their interaction contributes to prediction [14].

Recent diagnostic studies begin to expose this distinction. Prosodically neutral speech synthesis is used to reveal linguistic sensitivity in speech emotion recognition systems [15]. LISTEN evaluates lexical and acoustic reliance in audio language models and reports substantial lexical dominance in emotion recognition [16]. For sarcasm, CHARM combines prompt calibration with acoustic evidence to improve detection [17]. Recent work identifies prosodic heuristics through modality comparisons and targeted pitch and pause manipulations [18]. We complement this analysis with a paired factorial design that introduces a neutralised reference and distinguishes score shifts, their label-discriminative AUROC, and binary predictions. This reveals acoustic responses hidden by unchanged decisions. Duration balancing, within-identity comparisons, and alternative lexical resynthesis further establish which intervention contrasts remain robust. These diagnostics address questions left open by the reported heuristic analysis: whether score changes carry label information and whether observed advantages survive duration adjustment and alternative resynthesis, extending paired contrastive evaluation [19].

We therefore introduce CLASH, the Controlled Lexical-Acoustic Separation Harness, a counterfactual diagnostic framework for spoken sarcasm detection. We evaluate handcrafted acoustic-feature systems, self-supervised learning (SSL) probes, and large audio language models (LALMs) on MUStARD and CMMA, representing English and Mandarin, respectively. Context comparisons, alternative text-to-speech resynthesis, and duration balancing assess robustness. For target-only Qwen3-Omni, lexical-preserving speech retains a 0.135–0.148 AUROC advantage over prosody-preserving speech after duration balancing; this advantage also persists under alternative resynthesis.

Our contributions are threefold. First, we introduce an utterancepaired factorial protocol for auditing lexical and prosodic reliance under fixed detector parameters. Second, we demonstrate that acoustic score shifts can coexist with weak label discrimination and unchanged binary predictions. Third, we examine attribution robustness through duration balancing, within-identity comparisons, cluster-bootstrap inference, and alternative lexical resynthesis. The CLASH implementation is available at https: //github.com/glam-imperial/clash.

![](images/51944894ee21d2ff167922a0a35eb2dd47317aa3bc8ed62582dabc1fac323bb4.jpg)  
Fig. 1. Overview of CLASH. Utterances from MUStARD and CMMA undergo paired interventions to form original (O), lexical-preserving (L), prosody-preserving (P), and approximately neutralised (F) conditions. The same inputs are evaluated using handcrafted acoustic-feature classifiers, SSL probes, and speech LLMs. Contrasts between conditions characterise lexical and prosodic effects and their interaction. The two corpora are evaluated separately.

## 2. METHODOLOGY

## 2.1. Paired Speech Interventions

CLASH combines paired speech interventions with complementary measures of score sensitivity, sarcasm discrimination, and binary predictions (Fig. 1). Each utterance has four conditions: Original (O), Lexical-preserving (L), Prosody-preserving (P), and Approximately neutralised (F). Detector parameters remain fixed across conditions, so within-detector contrasts measure responses to input interventions; cross-detector comparisons additionally reflect representation and readout differences. The original label measures discrimination retained after transformation, without assuming that perceived sarcasm remains unchanged.

All detector inputs use 16 kHz mono audio. For condition L, voiced-frame F0 is set to the utterance median in English. In Mandarin, deviations from the utterance median log-F0 are scaled by 0.5, reducing pitch variation while retaining lexical tone-contour shapes. Non-silent frames are normalised to their median RMS energy, with timing and pauses unchanged. Thus, L attenuates pitch and intensity variation without eliminating rhythmic cues. Condition P uses WORLD resynthesis [20], replacing the original time-varying spectral envelope with a fixed carrier envelope while retaining F0 and aperiodicity parameters. This replacement disrupts segmental spectral cues to reduce lexical intelligibility. Condition F combines these operations; retained timing and processing effects preclude an information-free baseline.

To test whether the lexical-preservation advantage depends on WORLD-based processing, we construct an alternative L condition, denoted L , by synthesising each utterance’s reference transcript with Qwen3-TTS-12Hz-1.7B-CustomVoice [21], using a fixed voice and neutral-delivery instructions. This supplementary condition leaves the original O/L/P/F design unchanged. Whisper-based word/character error rates, F0 and energy correlations, and ECAPA-

TDNN [22] similarities assess intelligibility, acoustic preservation, and speaker changes across the generated conditions.

## 2.2. Detector Evaluation

We compare handcrafted acoustic-feature systems, SSL probes based on WavLM [23] and wav2vec 2.0 [24], and LALMs. Main estimates use 3,960 CMMA and 690 MUStARD utterances; additional cross-model comparisons use a common held-out subset of 781 and 163 utterances, respectively. These evaluations remain separate. The additional comparison includes WavLM-Base-Plus, wav2vec 2.0- Base, Qwen2.5-Omni-7B, Qwen2-Audio-7B, and Voxtral-Mini-3B [25].

We compare handcrafted and learned acoustic representations under the same classifier and training protocol. The handcrafted baseline uses 88-dimensional eGeMAPSv02 functionals extracted with openSMILE [26], providing explicit prosodic, spectral, and voice-quality descriptors. The learned baseline uses frozen WavLM-Large [23] (microsoft/wavlm-large), with temporal mean pooling of the final hidden layer. This comparison examines whether intervention responses depend on the representation, without assuming that either system selectively encodes lexical or prosodic information. Both systems use standardised features and L2-regularised logistic regression trained on original CMMA training speech only; three-fold training-set cross-validation selects regularisation by AU-ROC. MUStARD therefore assesses cross-lingual, cross-corpus transfer for these probes. Auxiliary audits compare logistic regression, linear regression, XGBoost, and LightGBM heads and examine train–test feature separation. The main LALM is Qwen3- Omni-30B-A3B-Instruct [27], evaluated without fine-tuning using greedy decoding and the fixed instruction below.

## Fixed Detection Prompt

Determine whether the speaker is sarcastic. Answer with exactly one word: Yes or No.

The context-aware setting prepends preceding dialogue text to the target-only input. Context remains identical across speech conditions, so context-aware F retains contextual evidence.

## 2.3. Diagnostic Contrasts and Robustness

Let s<sup>c</sup> denote the continuous score for utterance i under condition $c \in \{ O , L , P , F , L _ { \mathrm { T T S } } \}$ , and let $A _ { c }$ denote its condition-level AU-ROC against the original sarcasm labels. Scores use logistic regression decision values or LALM affirmative-versus-negative token log-probability masses. Paired differences, $d _ { i } = s _ { i } ^ { P } - \bar { s _ { i } ^ { F } }$ , measure score sensitivity. The AUROC of $d _ { i }$ assesses whether these changes distinguish sarcastic from non-sarcastic utterances; it differs from $A _ { P } - A _ { F } ,$ , which compares discrimination under the two conditions. Macro-F1 and positive-prediction rates describe binary predictions at a fixed score threshold of zero. We summarise standardised score sensitivity within each context setting as $d _ { z } = \overline { { d } } / \operatorname { s d } ( d )$

Principal contrasts are $A _ { L } - A _ { P }$ and $A _ { L _ { \mathrm { T T S } } } - A _ { P } ;$ supplementary contrasts are

$$
\Delta _ { L } = A _ { L } - A _ { F } , \quad \Delta _ { P } = A _ { P } - A _ { F } ,\tag{1}
$$

$$
I = A _ { O } - A _ { L } - A _ { P } + A _ { F } .\tag{2}
$$

These contrasts quantify discrimination under the implemented transformations for each detector. Although $\Delta _ { L } - \Delta _ { P } = A _ { L } - A _ { P }$ removes dependence on $A _ { F } ,$ off-target differences between L and $P$ remain uncontrolled. The interaction I describes AUROC nonadditivity, not a specific incongruity mechanism.

For context comparisons, we define $D = \Delta _ { P , \mathrm { t a r g e t } } - \Delta _ { P , \mathrm { c o n t e x t } } .$ Within-identity AUROC restricts positive–negative comparisons to the same conversation on CMMA or speaker on MUStARD. A separate processing-sensitivity check uses the detector scores to distinguish O from ${ \dot { F } } ,$ , reporting max $( A , 1 - A )$ to allow either score orientation.

To assess sensitivity to duration, we partition each corpus into five quantile bins of $\log ( 1 + t _ { i } )$ , where $t _ { i }$ denotes the original duration of utterance i. Within each bin, weighting equalises the total contribution of the two classes:

$$
w _ { i } = \frac { \operatorname* { m i n } \left\{ n _ { b ( i ) , 0 } , n _ { b ( i ) , 1 } \right\} } { n _ { b ( i ) , y _ { i } } } ,\tag{3}
$$

where $b ( i )$ denotes the bin containing utterance $i , y _ { i } \in \{ 0 , 1 \}$ is its sarcasm label, and $n _ { b , y }$ is the number of utterances with label $_ y$ in bin $b .$ The same weights apply to all speech conditions when computing weighted AUROC. A ten-bin analysis assesses sensitivity to binning. We estimate 95% percentile confidence intervals using 2,000 cluster-bootstrap replicates [28], retaining all paired speech conditions and context settings and recomputing the contrasts and weights within each replicate. Bin boundaries remain fixed. Resampling uses conversation clusters for CMMA and 21 speaker clusters for MUStARD, where reliable conversation identifiers are unavailable.

## 3. EXPERIMENTS AND RESULTS

## 3.1. Cross-Model Intervention Profiles

Table 1 reports AUROC and Macro-F1 on CMMA and MUStARD, whose sarcasm prevalence is 11.4% and 50.0%, respectively.

Qwen3-Omni achieves the highest original-speech AUROC among the systems in Table 1, and its lexical-preserving condition retains substantial discrimination. The unadjusted $A _ { L } - A _ { P }$ contrast is 0.139 on CMMA and 0.299 on MUStARD. The acoustic and SSL probes show different patterns. eGeMAPS provides weak discrimination across conditions, whereas WavLM-Large performs better on original speech but exhibits little separation between $L$ and $P ,$ particularly on MUStARD. Thus, stronger original-speech performance does not consistently correspond to a larger prosody-preservation advantage.

![](images/cb1947fb4f3d3e2c7a5479f367881482712dfd879ed501b7ba637d98b4c1b3d5.jpg)  
Fig. 2. Qwen3-Omni context comparison. Left: target-only minus context-aware standardised $P - F$ score shifts. Right: $D =$ $\Delta _ { P , \mathrm { t a r g e t } } - \Delta _ { P , \mathrm { c o n t e x t } }$ . Within-identity comparisons use CMMA conversations or MUStARD speakers. Bars show marginal 95% paired cluster-bootstrap intervals; dashed lines mark zero. The axes measure different quantities.

These representation baselines expose limitations of probebased attribution. Across four classifier heads, original-speech CMMA AUROC ranges from 0.456 to 0.503 for eGeMAPS and from 0.552 to 0.610 for WavLM-Large. Train–test domain classification yields 0.810/0.848 AUROC, indicating distributional separation. Their detector scores distinguish O from F at 0.824/0.818 AU-ROC. This separation can reflect intended cue changes or processing artefacts; it does not identify their relative contributions. These profiles therefore characterise the tested representations, readouts, and transfer settings rather than general properties of model families.

Table 2 extends the detector comparison on a common CMMA subset $( n \ = \ 7 8 1 )$ . All six systems yield positive $A _ { L } - A _ { P }$ contrasts, ranging from 0.025 to 0.058 for acoustic probes and from 0.071 to 0.219 for LALMs. However, negative $\Delta _ { P }$ values for the probes show that a relative lexical-preservation advantage need not imply strong absolute discrimination. These unadjusted CMMA results extend the contrast direction beyond Qwen3-Omni; the full robustness assessment remains model-specific.

## 3.2. Score Sensitivity and Binary Predictions

Paired score diagnostics distinguish score sensitivity, label discrimination, and binary predictions. Under target-only Qwen3-Omni, P increases log-odds relative to F for 74.8% of CMMA utterances and 82.5% of MUStARD utterances, yet both conditions produce exclusively non-sarcastic predictions. The paired differences yield AU-ROC values of 0.566 and 0.524, respectively. WavLM-Large shows a similar dissociation on CMMA, with a mean shift of 0.567 but a difference-based AUROC of 0.502; on MUStARD, the corresponding AUROC is 0.596. The same paired inputs therefore yield different conclusions at the score, discrimination, and decision levels. Acoustic responsiveness alone is consequently an insufficient criterion for evaluating effective prosodic cue use in the tested sarcasm detectors.

## 3.3. Duration and Lexical Resynthesis

Because the interventions retain temporal structure, we examine whether duration provides a residual label-related cue that affects the intervention contrasts. On MUStARD, duration alone yields an AUROC of 0.660, and sarcastic utterances average 5.77 s compared with 4.45 s for non-sarcastic utterances. Qwen3-Omni scores on unintelligible speech correlate negatively with duration $( \rho = - 0 . 7 2 )$ Longer sarcastic utterances can therefore receive lower scores after lexical removal, contributing to the below-chance ordering in $P$ and $F .$

Table 1. Performance on CMMA and MUStARD, unadjusted for duration. Cells report AUROC / Macro-F1; bold marks the highest value for each metric within each row. The final column reports the AUROC contrast $A _ { L } - A _ { P }$ . L provides an alternative to WORLD-based lexical preservation. LR denotes logistic regression. Preceding dialogue text remains available across all conditions in context-aware evaluation
<table><tr><td>System</td><td>Corpus</td><td>0</td><td>L</td><td> $P$ </td><td> $F$ </td><td> $L _ { \mathrm { T T S } }$ </td><td> $A _ { L } - A _ { P }$ </td></tr><tr><td colspan="8">Target-only evaluation</td></tr><tr><td> $\mathrm { e G e M A P S + L R }$ </td><td>CMMA</td><td>.466 / .471</td><td>.511 / .474</td><td>.513 / .502</td><td>.521 / .489</td><td>.531 / .496</td><td>-.002</td></tr><tr><td> $\mathrm { e G e M A P S + L R }$ </td><td>MUStARD</td><td>.554 / .423</td><td>.543 / .390</td><td>.522 / .464</td><td>.456 / .445</td><td>.499 / .336</td><td>+.021</td></tr><tr><td> $\mathrm { W a v L M - L a r g e + L R }$ </td><td>CMMA</td><td>.584 / .490</td><td>.521 / .472</td><td>.499 / .470</td><td>.502 / .470</td><td>.523 / .472</td><td>+.022</td></tr><tr><td> $\mathrm { W a v L M - L a r g e + L R }$ </td><td>MUStARD</td><td>.612 / .370</td><td>.571 / .361</td><td>.568 / .332</td><td>.442 / .333</td><td>.524 / .333</td><td>+.003</td></tr><tr><td>Qwen3-Omni</td><td>CMMA</td><td>.734 / .602</td><td>.688 / .594</td><td>.549 / .470</td><td>.471 / .470</td><td>.695 / .557</td><td>+.139</td></tr><tr><td>Qwen3-Omni</td><td>MUStARD</td><td>.775 / .690</td><td>.669 / .619</td><td>.370 / .333</td><td>.343 / .333</td><td>.687 / .625</td><td>+.299</td></tr><tr><td colspan="8">Context-aware evaluation</td></tr><tr><td>Qwen3-Omni</td><td>CMMA</td><td>.774 / .615</td><td>.756 / .610</td><td>.682 / .565</td><td>.673 / .571</td><td>.766 / .558</td><td>+.074</td></tr><tr><td>Qwen3-Omni</td><td>MUStARD</td><td>.780 / .658</td><td>.682 / .583</td><td>.499 / .488</td><td>.486 / .477</td><td>.707 / .634</td><td>+.183</td></tr></table>

Table 2. Additional model comparisons on the common CMMA held-out subset. Here, $\Delta _ { L } = A _ { L } - A _ { F }$ and $\Delta _ { P } = A _ { P } - A _ { F }$ . The final column gives $\Delta _ { L } - \Delta _ { P } = A _ { L } - A _ { P }$ . Estimates are unadjusted and reported to three decimal places.
<table><tr><td>System</td><td> $\Delta _ { L }$   $\Delta _ { P }$   $A _ { L } - A _ { P }$ </td></tr><tr><td> $\mathrm { e G e M A P S + L R }$ </td><td>-.027 -.085 .058</td></tr><tr><td>WavLM-Base-Plus -.001</td><td>-.048 .047</td></tr><tr><td>wav2vec 2.0-Base .001</td><td>-.024 .025</td></tr><tr><td>Qwen2.5-Omni-7B .258</td><td>.039 .219</td></tr><tr><td>Qwen2-Audio-7B .056</td><td>-.033 .089</td></tr><tr><td>Voxtral-Mini-3B .050</td><td>-.021 .071</td></tr></table>

Table 3. Duration-balanced AUROC contrasts for target-only Qwen3-Omni. Intervals are marginal 95% cluster-bootstrap confidence intervals, using conversations for CMMA and speakers for MUStARD.
<table><tr><td>Corpus</td><td>Contrast</td><td>Estimate</td><td>95% CI</td></tr><tr><td rowspan="2">CMMA</td><td> $A _ { L } - A _ { P }$ </td><td>.148</td><td>[.103, .193]</td></tr><tr><td> $A _ { L _ { \mathrm { T T S } } } - A _ { P }$ </td><td>.157</td><td>[.113, .202]</td></tr><tr><td rowspan="2">MUStARD</td><td> $A _ { L } - A _ { P }$ </td><td>.135</td><td>[.094, .210]</td></tr><tr><td> $A _ { L _ { \mathrm { T T S } } } - A _ { P }$ </td><td>.173</td><td>[.118, .276]</td></tr></table>

After duration balancing, $A _ { F }$ increases from 0.343 to 0.447 on MUStARD, while $A _ { L } - A _ { . }$ decreases from 0.299 to 0.135. Its direction remains positive. CMMA changes less, with the corresponding contrast increasing from 0.139 to 0.148. Table 3 shows that both duration-balanced contrasts have cluster-bootstrap intervals above zero.

Target-only Qwen3-Omni retains positive duration-balanced lexical-preservation contrasts under WORLD and TTS, with intervals above zero (Table 3). This agreement concerns AUROC; binary decisions differ. On CMMA, TTS raises positive predictions from 14.4% to 30.8%, while Macro-F1 falls from 0.594 to 0.557. Robustness is model-specific: on MUStARD, eGeMAPS and WavLM-Large yield $A _ { L _ { \mathrm { T T S } } } ~ - ~ A _ { P } ~ = ~ - 0 . 0 2 3$ and −0.044, respectively. Alternative resynthesis supports the Qwen3-Omni contrast without establishing equivalent transformation effects across the generated speech conditions.

Relative to O, WORLD-based L increases CMMA CER/MUStARD

WER by 19.9/11.0 percentage points; $L _ { \mathrm { T T S } }$ reduces them by 30.4/15.4 points. Two experts each inspect all four $O / L / P / F$ versions of 50 utterances per corpus and judge the inspected generated samples to have acceptable perceptual quality. These checks support transformation quality without establishing perfect cue separation.

## 3.4. Context Effects across Corpora

For Qwen3-Omni, we examine whether removing dialogue context increases the discriminative benefit of P relative to F. Target-only inputs yield larger standardised $P - F$ score shifts than contextaware inputs: $d _ { z }$ is 0.77 versus 0.44 on CMMA and 1.02 versus 0.80 on MUStARD. This sensitivity does not establish greater prosodic discrimination (Fig. 2). On CMMA, D is 0.069 [0.036, 0.100] before adjustment and 0.067 [0.033, 0.100] after duration balancing, but 0.037 [−0.018, 0.089] within conversations. The within-speaker MUStARD estimate is −0.012 [−0.038, 0.034]. Both withinidentity intervals include zero, providing insufficient evidence for a consistent compensatory benefit from removing context.

The additional value of original delivery also varies. Withinconversation analysis reduces the target-only $A _ { O } - A _ { L }$ contrast on CMMA from 0.046 to 0.006, whereas the within-speaker MUStARD contrast remains 0.090, compared with 0.106 overall. After duration balancing, I is 0.092 on MUStARD (95% CI [0.014, 0.146]) and −0.030 on CMMA. The recurring lexical-preservation advantage therefore coexists with corpus-dependent context and joint-condition effects. Since language and corpus vary together, these observations do not isolate cultural mechanisms.

## 4. CONCLUSION

We introduced CLASH, a paired diagnostic framework for examining lexical and prosodic cue reliance in spoken sarcasm detection. Across CMMA and MUStARD, the systems exhibited response patterns, and acoustic interventions shifted model scores without consistently improving sarcasm discrimination or changing binary predictions. For target-only Qwen3-Omni, the lexical-preservation advantage persisted after duration balancing and under independent lexical resynthesis, whereas context and interaction effects varied across corpora. These findings distinguished acoustic sensitivity from discrimination and showed why audio-based performance gains alone provided insufficient evidence of prosodic cue use. Imperfect cue separation, probe transfer, and corpus-specific structure limited the scope of the attribution.

## 5. REFERENCES

[1] Qiyang Sun, Yupei Li, Emran Alturki, et al., “Towards friendly ai: A comprehensive review and new perspectives on human-ai alignment,” AI and Ethics, vol. 6, no. 2, pp. 193, 2026.

[2] Firman Santosa, Ema Utami, Kusrini Kusrini, et al., “Sarcasm detection in the era of ai: A systematic review of techniques, datasets, performance, and future outlook,” in 2025 International Conference on Computer, Control, Informatics and its Applications (IC3INA). IEEE, 2025, pp. 19–24.

[3] Zhu Li, Yuqing Zhang, Xiyuan Gao, et al., “Modeling sarcastic speech: Semantic and prosodic cues in a speech synthesis framework,” arXiv preprint arXiv:2510.07096, 2025.

[4] Patricia Rockwell, “Lower, slower, louder: Vocal cues of sarcasm,” Journal of Psycholinguistic research, vol. 29, no. 5, pp. 483–495, 2000.

[5] Gina M Caucci, Roger J Kreuz, and Eugene H Buder, “What’s a little sarcasm between friends: Exploring the sarcastic tone of voice,” Journal of Language and Social Psychology, vol. 43, no. 4, pp. 486–502, 2024.

[6] Gregory A Bryant, “Prosodic contrasts in ironic speech,” Discourse Processes, vol. 47, no. 7, pp. 545–566, 2010.

[7] Hyewon Jang, Moritz Jakob, and Diego Frassinelli, “Context vs. human disagreement in sarcasm detection,” in Proceedings of the 4th Workshop on Figurative Language Processing (FigLang 2024), 2024, pp. 1–7.

[8] Zhu Li, Xiyuan Gao, Yuqing Zhang, et al., “A functional tradeoff between prosodic and semantic cues in conveying sarcasm,” arXiv preprint arXiv:2408.14892, 2024.

[9] Xiyuan Gao, Shekhar Nayak, and Matt Coler, “Spoken in jest, detected in earnest: A systematic review of sarcasm recognition-multimodal fusion, challenges, and future prospects,” IEEE Transactions on Affective Computing, 2025.

[10] Xiyuan Gao, Shekhar Nayak, and Matt Coler, “Deep cnn-based inductive transfer learning for sarcasm detection in speech.,” in INTERSPEECH, 2022, pp. 2323–2327.

[11] Santiago Castro, Devamanyu Hazarika, Ver’onica P’erez-Rosas, et al., “Towards multimodal sarcasm detection (an obviously perfect paper),” in Proceedings ofthe 57th annual meeting of the association for computational linguistics, 2019, pp. 4619–4629.

[12] Yazhou Zhang, Yang Yu, Qing Guo, et al., “Cmma: benchmarking multi-affection detection in chinese multi-modal conversations,” Advances in Neural Information Processing Systems, vol. 36, pp. 18794–18805, 2023.

[13] Hongliang Pan, Zheng Lin, Peng Fu, et al., “Modeling intra and inter-modality incongruity for multi-modal sarcasm detection,” in Findings of the Association for Computational Linguistics: EMNLP 2020, 2020, pp. 1383–1392.

[14] Jack Hessel and Lillian Lee, “Does my multimodal model learn cross-modal interactions? it’s harder to tell than you might think!,” in Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2020, pp. 861–877.

[15] Andreas Triantafyllopoulos, Johannes Wagner, Hagen Wierstorf, et al., “Probing speech emotion recognition transformers for linguistic knowledge,” arXiv preprint arXiv:2204.00400, 2022.

[16] Jingyi Chen, Zhimeng Guo, Jiyun Chun, et al., “Do audio llms really listen, or just transcribe? measuring lexical vs. acoustic emotion cues reliance,” in Proceedings ofthe 19th Conference ofthe European Chapter ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), 2026, pp. 5848–5877.

[17] Qiyang Sun, Yi Chang, Yupei Li, et al., “Charm: Charge calibration and acoustic rescue for llm-based multimodal sarcasm detection,” arXiv preprint arXiv:2607.11102, 2026.

[18] Yongjian Chen, Pengfei Wei, Yiqun Sun, et al., “When models hear what they expect: Diagnosing prosodic heuristics in multimodal sarcasm detection,” arXiv preprint arXiv:2608.30204, 2026.

[19] Matt Gardner, Yoav Artzi, Victoria Basmov, et al., “Evaluating models’ local decision boundaries via contrast sets,” in Findings ofthe Associationfor Computational Linguistics: EMNLP 2020, 2020, pp. 1307–1323.

[20] Masanori Morise, Fumiya Yokomori, and Kenji Ozawa, “World: A vocoder-based high-quality speech synthesis system for real-time applications,” IEICE TRANSACTIONS on Information and Systems, vol. 99, no. 7, pp. 1877–1884, 2016.

[21] Hangrui Hu, Xinfa Zhu, Ting He, et al., “Qwen3-tts technical report,” arXiv preprint arXiv:2601.15621, 2026.

[22] Brecht Desplanques, Jenthe Thienpondt, and Kris Demuynck, “Ecapa-tdnn: Emphasized channel attention, propagation and aggregation in tdnn based speaker verification,” arXiv preprint arXiv:2005.07143, 2020.

[23] Sanyuan Chen, Chengyi Wang, Zhengyang Chen, et al., “Wavlm: Large-scale self-supervised pre-training for full stack speech processing,” IEEE Journal of Selected Topics in Signal Processing, vol. 16, no. 6, pp. 1505–1518, 2022.

[24] Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, et al., “wav2vec 2.0: A framework for self-supervised learning of speech representations,” Advances in neural information processing systems, vol. 33, pp. 12449–12460, 2020.

[25] Alexander H Liu, Andy Ehrenberg, Andy Lo, et al., “Voxtral,” arXiv preprint arXiv:2507.13264, 2025.

[26] Florian Eyben, Klaus R Scherer, Bjorn W Schuller, Johan¨ Sundberg, Elisabeth Andre, Carlos Busso, Laurence Y Dev-´ illers, Julien Epps, Petri Laukka, Shrikanth S Narayanan, et al., “The geneva minimalistic acoustic parameter set (gemaps) for voice research and affective computing,” IEEE transactions on affective computing, vol. 7, no. 2, pp. 190–202, 2015.

[27] Jin Xu, Zhifang Guo, Hangrui Hu, et al., “Qwen3-omni technical report,” arXiv preprint arXiv:2509.17765, 2025.

[28] Christopher A Field and Alan H Welsh, “Bootstrapping clustered data,” Journal of the Royal Statistical Society Series B: Statistical Methodology, vol. 69, no. 3, pp. 369–390, 2007.