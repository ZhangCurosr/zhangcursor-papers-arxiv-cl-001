# EmoHarness: Boosting Multimodal Emotion Understanding with Meta-Harness Evolving

Fenghai Li<sup>1</sup>, Bingxuan Li<sup>1</sup>, Jiawen Zhang<sup>2</sup>, Yifan Shen<sup>1</sup>, Jiateng Liu<sup>1</sup>, Yutong Zhang<sup>3</sup>, Yue Guo<sup>1</sup>

1<sub>University</sub> <sub>of</sub> <sub>Illinois</sub> <sub>Urbana-Champaign</sub> 2<sub>University</sub> <sub>of</sub> <sub>California,</sub> <sub>Davis</sub> 3<sub>Stanford</sub> <sub>University</sub>( ) : <sup></sup> https://emo-harness.github.io

## Abstract

Small multimodal models can misjudge emotion when words, voice, and visual cues conflict. We present EmoHarness, which runs such a model, the frozen student, inside a harness: the prompts, settings, and verification code around its calls. A human-written starting harness, the scaffold, numbers the student’s visual<sup>×</sup> observations, transcript, and prosody descriptions as citable evidence and has the student weigh candidate emotions against it. In Meta-Harness Evolving (MHE), a stronger LLM, the Builder, reads the student’s errors and execution traces offline and edits the harness; static checks and validation gates decide which edits are kept, and the Builder never sees evaluation items. On 5,507 MME-Emotion items, EmoHarness raises task-macro Recognition by 7.6–13.4 points over a single video-only call for all four students and both Builders. Transcripts and prosody descriptions supply most of this boost (+5.38 to +10.71); the scaffold changes the audio-augmented call by −0.79 to +2.54, partly through extra multi-label predictions, and evolved edits add at most +0.95, although no evolved increment survives Holm correction. The harness’s emotion context also transfers to IntentQA, with small gains for three of four students, none of which survives correction. EmoHarness thus boosts small multi-+ 9 modal students chiefly by supplying and organizing audio-derived evidence.

## 1 Introduction

When someone says “That’s great,” a pause or+ 5 a tense expression may reflect sarcasm rather than satisfaction. Understanding emotion requires interpreting words, voice, and visual behavior+ 1 0 together, particularly when these cues conflict (Bagher Zadeh et al., 2018; Shen et al., 2026). Mis-<sub>reading them can have consequences beyond an</sub>0 4 8 1 2 <sup>+</sup>Figure 1: EmoHarness at a glance. (a) A selected heldout clip (Gemma-4-E4B; Figure 4a): posture suggests neutral, the words negative. Video, Audio (video plus transcript and prosody text), and Shared (the humanwritten scaffold) miss the reference; both MHE harnesses answer negative. (b) Offline, a Builder LLM (GPT-5.6 Luna or GPT-6.1 Sol) edits the harness from 0 <sub>.the frozen student’s failures and execution traces; checks</sub> and validation gates promote edits over four generations. (c) Task-macro Recognition contrasts (VL: Qwen3-VL; Q3.5: Qwen3.5-4B).

![](images/809d599470a4c12c7f47a6810247478d9f84f3ba82ce415bdc3a13ad82440f84.jpg)

incorrect label: a system may misunderstand a per-0 1 2<sup>son’s</sup> <sup>intent,</sup> <sup>respond</sup> <sup>inappropriately</sup> <sup>to</sup> <sup>a</sup> <sup>social</sup> interaction, or overlook a patient’s indirectly expressed fear or discomfort (Li et al., 2023; Wilf et al., 2023; Han et al., 2026). For smaller multimodal models, transcripts and prosody descriptions supplement visual observations with evidence about what is said and how it is delivered (Cao et al., 2024; Li et al., 2026d,b). But interpreting these sources together remains challenging, especially when they suggest different emotions. We therefore ask whether another AI can help a small multimodal student use this evidence more effectively without updating the student’s weights.

Existing emotion models adapt weights through instruction tuning or reinforcement learning (Cheng et al., 2024b; Zhao et al., 2025). An alternative is to adapt frozen systems through modelgenerated instructions and program optimization (Zhou et al., 2023; Opsahl-Ong et al., 2024; Hu et al., 2025). We build on this approach but direct adaptation toward a student’s recurring emotionlabel failures and cross-modal conflicts.

We introduce EmoHarness (Figure 1), an AIfor-AI (AI4AI) framework that adapts emotion inference through Meta-Harness Evolving (MHE). An offline Builder examines emotion-label confusions, cross-modal conflicts, and execution traces, and then proposes reusable edits to prompts, execution settings, and restricted verification code. The frozen student organizes visual observations, transcripts, and prosody descriptions into indexed textual evidence, proposes alternative emotional interpretations, and reviews their support and contradictions. Checks and validation select a harness that is frozen before evaluation. If no edit passes, the human-written scaffold is retained. The Builder does not answer evaluation questions.

To assess the contribution of Builder-guided adaptation, we compare the evolved harness (MHE) with its initial human-written version (Shared) and with single-call prediction from video supplemented with transcripts and prosody descriptions (Audio). Comparing Audio with single-call video-only prediction (Video) measures the effect of adding this audio-derived evidence. We also examine transfer to task types excluded from harness construction and test whether emotion context generated without access to downstream questions helps intent question answering.

Across four students and two Builders, EmoHarness improves task-macro Recognition on 5,507 MME-Emotion items by 7.6–13.4 points over Video. Most of this improvement comes from transcripts and prosody descriptions: Audio outperforms Video by 5.38–10.71 points, whereas

Builder-guided adaptation produces much smaller changes relative to Shared (−0.07 to +0.95 points). Adding the harness’s emotion context also yields small observed gains for three of four students on IntentQA.

In summary, we introduce EmoHarness, a failure-driven harness construction framework, which boosts frozen multimodal students on emotion recognition and transfers its emotion context to intent question answering. In addition, we conduct a systematic analysis of where the boost comes from and what limits harness evolution. We believe our work paves the way for more adaptive and generalizable multimodal systems that can systematically learn from failures, improve without updating model parameters, and transfer acquired capabilities across related tasks.

## 2 Related Work

Multimodal emotion understanding. Emotion-LLaMA and R1-Omni adapt model weights through instruction tuning and reinforcement learning, respectively (Cheng et al., 2024b; Zhao et al., 2025); EmoHarness instead adapts the inference program around frozen weights. We evaluate on MME-Emotion (Zhang et al., 2026) and reuse emotion context on IntentQA (Li et al., 2023). Following the multi-dimensional evaluation of HEMM (Liang et al., 2024), we separate recognizing an emotion from producing useful downstream context.

Optimizing prompts, programs, and harnesses. APE, OPRO, and ProTeGi search instructions with model-generated proposals or feedback (Zhou et al., 2023; Yang et al., 2024; Pryzant et al., 2023); DSPy and MIPRO optimize multi-stage programs (Khattab et al., 2024; Opsahl-Ong et al., 2024); TextGrad and Trace learn from textual feedback and execution traces (Yuksekgonul et al., 2025; Cheng et al., 2024a; Xu et al., 2026); and ADAS and AFlow search code-defined agents (Hu et al., 2025; Zhang et al., 2025). Meta-Harness exposes prior code, scores, and traces to a proposer (Lee et al., 2026), automated harness adaptation can target the failures of smaller models (Yang et al., 2026), and multi-agent and self-evolving systems learn from feedback (Li et al., 2025, 2026c,a). MHE applies this line of work to multimodal emotion recognition: a student’s emotion confusions and cross-modal conflicts drive reusable edits, a separate validation split gates every promotion, and the evolved harness is frozen and reused, unlike programs generated for each request (Gupta and Kembhavi, 2023).

Feedback and procedural adaptation. Self-Refine revises single outputs and Reflexion carries feedback across trials (Madaan et al., 2023; Shinn et al., 2023); EmoHarness separates the student’s per-instance review from the Builder’s offline edits. Self-correction can fail (Huang et al., 2024), agent errors can cascade across steps (Zhu et al., 2025), and explanations need not reflect decisions (Turpin et al., 2023). We therefore score harnesses only on items the Builder never saw, report validation and evaluation gains separately, and count student calls (Kapoor et al., 2025).

## 3 EmoHarness: Meta-Harness Evolving for Emotion Understanding

A harness is a bundle of prompts, settings files, and verification code. Figure 2 shows how MHE evolves one offline and how the student runs it.

## 3.1 Task, Roles, and Fixed Boundaries

Given a video $v ,$ a request $q ,$ and allowed labels $\mathcal { V } _ { q }$ , a harness H invokes the frozen student $M _ { \theta }$ and returns labels $\hat { Y }$ , an explanation $z ,$ and a trace $\tau { : }$

$$
( \hat { Y } , z , \tau ) = H ( M _ { \theta } ; v , P ( v ) , q , \mathcal { Y } _ { q } ) .\tag{1}
$$

Fixed preprocessing $P ( v )$ supplies a transcript, prosodic descriptors, and a frame cache. MHE changes H but never $\theta ,$ and within H the perception tools, evidence and output schemas, and output controller are also fixed; the Builder may rewrite prompts, settings (e.g., transcript presentation), and restricted verification code.

## 3.2 Evidence-Indexed Emotion Inference

Audio preprocessing uses Whisper large-v3 (via faster-whisper) and openSMILE eGeMAPSv02 (Radford et al., 2023; Eyben et al., 2010, 2016); reference-normalized acoustic features become descriptions such as high pitch, never emotion labels. The human-written scaffold $H _ { 0 }$ (Shared) uses four student calls. Observe extracts facial, bodily, scene, and temporal cues from the video, decoded at 2 fps; deterministic assembly indexes them $( v _ { 1 } , v _ { 2 } , \ldots )$ along with transcript sentences and prosody phrases $( a _ { 1 } , a _ { 2 } , \ldots )$ , and later calls see only this textual record. Propose lists candidate labels with confidences and with identifiers of supporting and contradicting evidence; deterministic code checks the identifiers against the record and then ranks the candidates. Review re-examines the ranked candidates, and its output is verified the same way. A fixed controller then selects at most three labels from the vocabulary. It prefers candidates whose valid support exceeds their contradiction and fills any remaining slots from the ranking, so it always returns an answer, even one without valid support. Explain then writes an explanation that cannot change the labels. Each malformed call may be retried once, under a cap of six calls (Appendix B).

## 3.3 Meta-Harness Evolving

MHE evolves the harness over $T = 4$ generations. In each, the Builder proposes candidate edits, and gates either promote one candidate or keep the current harness. MHE inherits from Meta-Harness (Lee et al., 2026) a proposer that reads prior harness code and execution traces. It differs in three respects. First, the Builder sees the failures of one frozen student, clustered by gold–prediction confusion and by cross-modal conflict; a deterministic check flags a conflict when the student’s visual and audio-derived citations favor different labels. Second, gates on a separate validation split decide promotion, and the Builder sees neither validation scores nor gate decisions, only the resulting harness. Third, each run keeps one current harness, the incumbent, giving a chain $H _ { 0 } \to \cdots \to H _ { T }$

Variation. The incumbent runs on a construction split $\mathcal { D } _ { b } .$ , whose failures the Builder may read. Given the incumbent’s files, intermediate records, clustered failures, and earlier proposals with check and reviewer feedback, the Builder proposes up to four candidate harnesses, each an edit aimed at one failure cluster; evaluation data are never exposed.

Selection. Candidates must pass file-scope, schema, loadability, and static-code checks, as well as a proposal reviewer that rejects, for example, edits that force a label regardless of evidence or add unsupported settings; these are filters, not guarantees. Survivors run on the validation split $\mathcal { D } _ { v }$ Let $r _ { i } ( H ) \in [ 0 , 1 ]$ be per-item Recognition (Section 4.2) and $S _ { t }$ the validation items solved $( r _ { i } = 1 )$ by any incumbent so far. The regression rate is the fraction of $S _ { t }$ that a candidate no longer solves:

$$
\rho _ { t } ( H ) = \frac { | S _ { t } \setminus \operatorname { S o l v e d } ( H ; \mathcal { D } _ { v } ) | } { | S _ { t } | } .\tag{2}
$$

![](images/21979b3cf8835afe8f1577716523fbe81a27c7cb7c547b6e03e060cb3c42f892.jpg)  
Figure 2: How Meta-Harness Evolving builds a harness and how the frozen student runs it. (a) Offline, the Builder turns the incumbent’s construction failures and traces into at most four candidate edits per generation; static checks, a proposal reviewer, and the gates of Eq. 3 promote the best candidate or retain the incumbent, and after four generations the harness is frozen. (b) At inference the frozen student makes four calls (at most six with retries) around deterministic verification and a fixed controller. The example is the fine-grained sentiment (FG-SA) clip of Appendix Figure 11a: the evidence and the quoted output come from the GPT-5.6 Luna harness on Qwen3-VL-4B; the Propose rows, the confusions in (a), and c1–c4 are illustrative.

A candidate H replaces the incumbent $H _ { t }$ only if

$$
\begin{array} { r l r } & { } & { \widehat { R } _ { v } ( H ) > \widehat { R } _ { v } ( H _ { t } ) , \qquad \rho _ { t } ( H ) \le \delta , ~ } \\ & { } & { \bar { c } ( H ) \le B , \qquad \widehat { Q } _ { v } ( H ) \ge \widehat { Q } _ { v } ( H _ { t } ) - \epsilon , ~ } \end{array}\tag{3}
$$

where $\widehat { R } _ { v }$ is the mean validation Recognition, c <sup>.</sup>the mean number of student calls, and $\widehat { Q } _ { v }$ a reasoning proxy used only for selection: the average share of non-final reasoning steps that a judge marks correct, on a fixed 200-item validation subset (Appendix B). We use $\delta = 0 . 0 5 , B = 4 . 5$ , and $\epsilon = 0 . 0 2$ . The best passing candidate, ranked by validation Recognition and then by fewer calls, be-<sup>.</sup>comes $H _ { t + 1 } ;$ otherwise $H _ { t + 1 } = H _ { t }$ . After T generations the evolved harness $H ^ { \star } = H _ { T }$ is frozen; if nothing was promoted, $H ^ { \star }$ is the Shared scaffold itself. Algorithm 1 (Appendix B) summarizes the loop.

## 4 Experimental Setup

We ask what contributes to recognition (RQ1), whether evolved edits transfer to held-out task types (RQ2), and whether the harness’s emotion context helps intent question answering (RQ3).

## 4.1 Students, Builders, and Data

Models. We serve Qwen3-VL-8B, Qwen3-VL-4B (Bai et al., 2025), Qwen3.5-4B (Qwen Team, 2026), and Gemma-4-E4B (Gemma Team, 2026) with vLLM (Kwon et al., 2023) at temperature zero, with thinking disabled for Qwen3.5 and Gemma. Audio reaches every student only as transcript and prosody text. Each student’s harness is evolved separately by GPT-5.6 Luna and by GPT-6.1 Sol; each Builder also acts as its own proposal reviewer, and GPT-5.6 Luna scores the reasoning proxy. Each student–Builder pair has one four-generation run. A third Builder configuration, which changes no conclusion, is described in Appendix A.

Data. Construction and validation use 399 and 394 MME-Emotion items (Zhang et al., 2026) from three task types (ER-Lab, ER-Wild: lab and in-thewild emotion recognition; SA: sentiment analysis). Evaluation uses 5,507 further items, namely the benchmark minus every item whose media overlap construction, validation, or a reserved 129-item set. The evaluation set contains 2,963 items in the seen types and 2,544 in five held-out types (noisy, fine-grained, and multi-label emotion recognition: Noise-ER, FG-ER, ML-ER; fine-grained sentiment: FG-SA; intent recognition: IR). Heldout refers to task type, not to an untouched test set: 986 evaluation items, which we cannot identify, were inspected during development. Inspection could have influenced any human design choice, most plausibly the scaffold, but not the Builder’s edits directly: the Builder never saw evaluation items (Appendix A).

## 4.2 Conditions and Metrics

Video prompts the student once with the video, and Audio adds transcript and prosody text to that call. Shared runs the four-call scaffold $H _ { 0 }$ , and MHE runs the frozen evolved harness $H ^ { \star }$ . Recognition, computed locally without an LLM judge, scores the first normalized prediction for single-label requests and the fraction of gold labels recovered for multi-label requests, a recall-like score that ignores extra labels. Audio−Video measures added evidence, and MHE−Shared measures what evolution adds. Shared−Audio mixes four changes: a multi-step procedure, four calls instead of one, later calls that see only a text record of the video, and more predicted labels, which the recall-like score rewards. Task-macro Recognition averages the eight task means:

$$
R _ { \mathrm { m a c r o } } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \frac { 1 } { | \mathscr { D } _ { k } | } \sum _ { i \in \mathscr { D } _ { k } } r _ { i } .\tag{4}
$$

Differences are in percentage points (pp). Intervals are pointwise 95% task-stratified paired bootstrap intervals (10,000 resamples), and p-values are twosided bootstrap values. The eight MHE−Shared tests form one Holm family (Holm, 1979).

## 4.3 Downstream Reuse of Emotion Context

We use the IntentQA test split (2,134 questions on 567 videos). Each evolved harness writes one emotion context per video (predicted emotion, alternatives, and a four-step analysis) using the ER-Wild instruction and label set, without seeing any question. Without a harness, the student answers each question given the cached frames (Frames), the frames with transcript and prosody (Audio), or the Audio inputs plus the emotion context (Audio+Emotion); answers are capped at 64 tokens (1,024 with a final-answer parser for Qwen3.5-4B), and unparsed replies count as wrong. Intervals use 10,000 paired bootstrap resamples clustered by video, with Holm correction over eight tests.

(a) Video → Audio → Shared, per student  
![](images/e768896e912788a63ef3a6bc7f71d28f7d5a2852624664118bc7e4af65669ea9.jpg)

(b) What MHE adds, zoomed: change vs. Shared  
![](images/4e2f5b3fef1e3c7d083d5de8f777a1925e2ea2d2cc37b16ef59f4bb40671dddd.jpg)  
Figure 3: Recognition paths per student. (a) Seenversus held-out-type macro Recognition (%; Table 1): arrows run Video (open circle) → Audio (amber diamond) → Shared (gray); amber numbers beside each name: Audio−Video macro gain (pp); dashed lines: equal overall Macro. (b) MHE−Shared, zoomed (origin = Shared; ticks 10× finer than in (a)); violet circle: GPT-5.6 Luna; blue square: GPT-6.1 Sol; changes below 0.1 points have no arrow; labels and dashed lines give the macro change. <sup>†</sup> Harness equals Shared. Short names and rounding as in Table 2.

## 5 Results

## 5.1 RQ1: Where Do the Gains Come From?

Added evidence dominates. Transcript and prosody text give every student its largest gain (Table 1, Figure 3a; Appendix Table 4b): Audio−Video ranges from +5.38 (Qwen3.5-4B) to +10.71 points (Gemma-4-E4B), and every interval excludes zero. Shared−Audio is +2.23 (Qwen3.5-4B) and +2.54 (Gemma-4-E4B), whose intervals exclude zero, and +1.32 (Qwen3-VL-8B) and −0.79 (Qwen3-VL-4B), whose intervals include zero. The Qwen3.5-4B and Gemma-4- E4B gains depend partly on recall-like scoring of FG-ER and ML-ER, where Shared predicts more labels than Audio. Scoring these two tasks by set F1 lowers Shared−Audio to +1.10 (Qwen3.5- 4B), +1.00 (Gemma-4-E4B), −0.25 (Qwen3-VL-8B), and −1.41 (Qwen3-VL-4B); under F1, every Qwen3-VL harness, scaffold or evolved, scores below single-call Audio (Appendix C). Evolved harnesses exceed Video by 7.61–13.36 points and differ from single-call Audio by −0.54 to +2.65.

<table><tr><td rowspan="3" colspan="2"></td><td colspan="3">Seen task types</td><td colspan="5">Held-out task types</td><td colspan="3">Summary</td><td rowspan="3"></td></tr><tr><td rowspan="2">ER- Lab</td><td rowspan="2">ER- Wild</td><td rowspan="2">SA</td><td rowspan="2">Noise- ER</td><td rowspan="2">FG- ER</td><td rowspan="2">ML- ER</td><td>FG-</td><td rowspan="2"></td><td colspan="2">Held-</td><td rowspan="2">95% CI</td></tr><tr><td>SA</td><td>IR</td><td>out</td><td>Macro</td></tr><tr><td>Video</td><td></td><td>8.5</td><td>36.2</td><td>41.4</td><td>56.4</td><td>23.6</td><td>38.8</td><td>13.2</td><td>21.5</td><td>28.7</td><td>30.7</td><td>29.95</td><td>28.8–31.1</td></tr><tr><td>Audio</td><td></td><td>23.2</td><td>38.0</td><td>63.3</td><td>65.0</td><td>19.2</td><td>41.7</td><td>30.8</td><td>32.0</td><td>41.5</td><td>37.7</td><td>39.15</td><td>37.8–40.5</td></tr><tr><td>we--8B Shared</td><td></td><td>28.8</td><td>35.5</td><td>59.9</td><td>56.6</td><td>26.1</td><td>48.9</td><td>26.7</td><td>41.2</td><td>41.4</td><td>39.9</td><td>40.47</td><td>39.0–41.9</td></tr><tr><td></td><td>MHE GPT-5.6 Luna</td><td>29.4</td><td>35.0</td><td>60.6</td><td>57.2</td><td>26.2</td><td>48.7</td><td>27.8</td><td>40.6</td><td>41.7</td><td>40.1</td><td>40.68</td><td>39.2–42.1</td></tr><tr><td></td><td>MHE GPT-6.1 Sol</td><td>28.2</td><td>35.7</td><td>60.5</td><td>56.4</td><td>26.4</td><td>49.1</td><td>26.7</td><td>41.2</td><td>41.5</td><td>40.0</td><td>40.52</td><td>39.1-41.9</td></tr><tr><td>Ow--4B Video</td><td></td><td>8.5</td><td>34.7</td><td>42.6</td><td>51.8</td><td>21.9</td><td>36.9</td><td>11.1</td><td>27.6</td><td>28.6</td><td>29.9</td><td>29.39</td><td>28.2–30.6</td></tr><tr><td>Audio</td><td></td><td>18.6</td><td>37.0</td><td>63.8</td><td>61.4</td><td>20.8</td><td>39.4</td><td>35.1</td><td>31.0</td><td>39.8</td><td>37.5</td><td>38.39</td><td>37.0–39.7</td></tr><tr><td>Shared</td><td></td><td>22.6</td><td>35.6</td><td>65.6</td><td>57.4</td><td>17.7</td><td>45.4</td><td>20.8</td><td>35.7</td><td>41.3</td><td>35.4</td><td>37.60</td><td>36.2–39.0</td></tr><tr><td></td><td>MHE GPT-5.6 Luna</td><td>23.7</td><td>35.7</td><td>65.7</td><td>58.6</td><td>18.4</td><td>46.4</td><td>21.9</td><td>37.9</td><td>41.7</td><td>36.7</td><td>38.56</td><td>37.2–39.9</td></tr><tr><td></td><td>MHE GPT-6.1 Sol</td><td>23.2</td><td>35.9</td><td>65.8</td><td>57.8</td><td>18.1</td><td>45.5</td><td>20.8</td><td>35.7</td><td>41.6</td><td>35.6</td><td>37.85</td><td>36.5–39.2</td></tr><tr><td></td><td>Video</td><td>14.1</td><td>38.1</td><td>38.3</td><td>46.8</td><td>20.7</td><td>46.7</td><td>6.1</td><td>26.2</td><td>30.2</td><td>29.3</td><td>29.62</td><td>28.4–30.8</td></tr><tr><td>Owe-4B</td><td>Audio</td><td>20.3</td><td>38.2</td><td>56.1</td><td>51.2</td><td>20.8</td><td>39.2</td><td>21.7</td><td>32.5</td><td>38.2</td><td>33.1</td><td>34.99</td><td>33.7-36.4</td></tr><tr><td>Shared</td><td></td><td>29.9</td><td>33.4</td><td>60.6</td><td>45.6</td><td>23.8</td><td>45.7</td><td>24.9</td><td>33.9</td><td>41.3</td><td>34.8</td><td>37.23</td><td>35.8–38.6</td></tr><tr><td></td><td>MHE GPT-5.6 Luna†</td><td>29.9</td><td>33.4</td><td>60.6</td><td>45.6</td><td>23.8</td><td>45.7</td><td>24.9</td><td>33.9</td><td>41.3</td><td>34.8</td><td>37.23</td><td>35.8–38.6</td></tr><tr><td></td><td>MHE GPT-6.1 Sol</td><td>29.9</td><td>33.6</td><td>60.6</td><td>45.6</td><td>23.8</td><td>45.7</td><td>24.9</td><td>33.9</td><td>41.4</td><td>34.8</td><td>37.25</td><td>35.8–38.6</td></tr><tr><td>Video</td><td></td><td>9.0</td><td>30.8</td><td>39.8</td><td>41.0</td><td>25.1</td><td>36.0</td><td>9.1</td><td>25.4</td><td>26.6</td><td>27.3</td><td>27.03</td><td>25.8–28.2</td></tr><tr><td>Geam--4B</td><td>Audio</td><td>27.1</td><td>33.4</td><td>61.2</td><td>52.4</td><td>20.2</td><td>40.4</td><td>32.8</td><td>34.5</td><td>40.6</td><td>36.1</td><td>37.74</td><td>36.3–39.2</td></tr><tr><td>Shared</td><td></td><td>29.4</td><td>31.0</td><td>64.9</td><td>47.4</td><td>26.0</td><td>47.2</td><td>38.8</td><td>37.5</td><td>41.7</td><td>39.4</td><td>40.28</td><td>38.8–41.7</td></tr><tr><td></td><td>MHE GPT-5.6 Luna</td><td>29.4</td><td>31.5</td><td>64.5</td><td>47.2</td><td>26.0</td><td>47.6</td><td>38.6</td><td>36.9</td><td>41.8</td><td>39.3</td><td>40.21</td><td>38.8–41.7</td></tr><tr><td></td><td>MHE GPT-6.1 Sol</td><td>29.4</td><td>31.1</td><td>64.2</td><td>46.6</td><td>25.9</td><td>48.5</td><td>39.7</td><td>37.7</td><td>41.6</td><td>39.7</td><td>40.39</td><td>38.9-41.9</td></tr></table>

Table 1: Main results on MME-Emotion. Recognition (%) per task type and task-macro averages (Seen, Held-out, Macro) on 5,507 items, for each student under Video and Audio (one call each), the Shared scaffold, and the harnesses evolved by Meta-Harness Evolving (MHE) with each Builder (about four calls each). Seen task types occur in construction; held-out types do not. FG-ER and ML-ER use recall-like Recognition. Shaded rows: MHE; bold: best per student and column (ties unmarked). 95% CI: task-stratified bootstrap interval of the macro score (10,000 resamples). <sup>†</sup> No candidate passed the gates, so the harness equals Shared. No MHE−Shared difference survives Holm correction over eight tests (smallest adjusted p = .192). Differences reported in the text and figures use unrounded scores and can differ by 0.01 (0.1 in one-decimal columns) from differences of the displayed values.

What Meta-Harness Evolving adds. Across the eight student–Builder pairs, MHE−Shared ranges from −0.07 to +0.95 points (Figure 3b). The largest increment, GPT-5.6 Luna on Qwen3-VL-4B (+0.95, 95% CI [+0.12, +1.82]), does not survive Holm correction (adjusted p = .192), and neither does any other contrast (Appendix C). GPT-6.1

Sol’s harnesses score 0.03 to 0.25 points above Shared for every student; only the Qwen3-VL-4B interval excludes zero (lower bound +0.01).

Validation gains shrink on evaluation. Across four runs per Builder, the gates promote six GPT-5.6 Luna edits and five GPT-6.1 Sol edits, raising validation Recognition by up to 2.54 points (Appendix Figure 6). Scored like validation (flat item mean) on the 2,963 seen-type evaluation items, the seven promoted harnesses change by −0.20 to +0.30 points; the gain of GPT-5.6 Luna’s Qwen3- VL-8B harness falls from 2.54 to 0.10 points.

## 5.2 RQ2: Transfer to Held-Out Task Types

With one exception, held-out changes are as small as seen-type changes. MHE−Shared ranges from −0.16 to +0.47 points on the seen types and, for seven pairs, from −0.14 to +0.28 on the five heldout types (Table 1, Figure 3b; Appendix Figure 8). The exception is again GPT-5.6 Luna on Qwen3- VL-4B (+1.24), which lifts held-out Recognition to 36.65, still below single-call Audio (37.55).

<table><tr><td></td><td>VL-8B</td><td>VL-4B</td><td>Q3.5</td><td>Gemma</td></tr><tr><td>Frames</td><td>70.99</td><td>67.48</td><td>79.90</td><td>57.31</td></tr><tr><td>Audio</td><td>72.63</td><td>68.42</td><td>77.84</td><td>60.59</td></tr><tr><td>+ Emotion MHE Luna</td><td>73.10</td><td>70.29</td><td>76.34†</td><td>61.62</td></tr><tr><td>+ Emotion MHE Sol</td><td>72.87</td><td>69.92</td><td>76.43</td><td>61.95</td></tr></table>

Table 2: Downstream QA accuracy (%). Inputs as in Section 4.3; 2,134 IntentQA test questions. Shaded rows add MHE emotion context; bold: best per student (ties unmarked). Luna: GPT-5.6 Luna; Sol: GPT-6.1 Sol. VL-8B/VL-4B: Qwen3-VL-8B/4B; Q3.5: Qwen3.5-4B; Gemma: Gemma-4-E4B. No Audio+Emotion−Audio contrast survives Holm correction over eight tests (smallest adjusted $p = . 0 8 5 ;$ Appendix Table 5). Differences reported in the text and figures use unrounded scores and can differ by 0.01 from differences of the displayed values. <sup>†</sup> Retained scaffold: this context is generated by Shared.

With seen-type gains this small, the held-out results say little about transfer beyond this pair.

## 5.3 Case Study: What an Evolved Edit Changes

Figure 4 follows two Gemma-4-E4B items. In (a), Shared rates a mildly negative remark as strong negative because a confidence score, not an observed cue, ranks that label first. With the GPT-5.6 Luna edit, the review call treats the composed posture as evidence for a neutral reading, and the harness answers the reference negative instead. In (b), Video answers the correct Neutral, but once the transcript of a billing complaint is added, Audio, Shared, and both MHE harnesses answer Frustration. The GPT-5.6 Luna harness keeps Frustration even though its added rule asks it to treat neutral as a serious alternative: its explanation never names Neutral and still lists crossed arms that no sampled frame shows. Over all 5,507 items, this harness changes Shared’s label set on 169 items: 46 improve, 45 worsen, and 78 score the same (more cases in Appendix F).

## 5.4 RQ3: Does Emotion Context Help Intent QA?

On IntentQA, Audio−Frames ranges from −2.06 to +3.28 points (Table 2). Adding the emotion context raises accuracy in all six pairs of Qwen3-VL-8B, Qwen3-VL-4B, and Gemma-4-E4B (+0.23 to +1.87 points; Figure 5, Appendix Figure 10); for Qwen3-VL-4B both pointwise intervals exclude zero (+1.87 [+0.47, +3.27]; +1.50 [+0.05, +2.97]). Qwen3.5-4B loses 1.50 and 1.41 points. Each context changes the correctness of 248 to 306 of the 2,134 answers, so each net change is a small difference between many gains and losses. None of the eight contrasts survives Holm correction (smallest adjusted $p = . 0 8 5 \mathrm { ; }$ ; Appendix Table 5), and the mean change is positive in every official question type, none significantly (Appendix Table 6). We have no length-matched control (Appendix G), so longer inputs, rather than emotion content, could account for these changes. Scaffold contexts exist only for Qwen3.5-4B, whose GPT-5.6 Luna harness retained the scaffold. Its GPT-6.1 Sol context matches the scaffold’s for 531 of 567 videos. No experiment therefore compares a scaffold context with a substantively evolved one, so these changes cannot be attributed to evolution.

## 6 Discussion and Conclusion

When does MHE help? The largest gain, GPT-5.6 Luna on Qwen3-VL-4B, occurs for the only student whose scaffold scores below Audio. Its edits add 0.95 points, mostly on held-out types (+1.24 versus +0.47 on seen types). They change the label set on 1,189 items (21.6%): 281 improve, 233 worsen, and 675 score the same. Three findings suggest why the other gains are small. First, each promotion picks the best of up to four candidates on 394 items, so the winner’s validation gain tends to overstate its gain on new items (Section 5.1). The Recognition gate also requires no margin: three of the eleven promotions rest on a single validation item (0.25 points). Second, the regression gate blocked every large validation gain for both Builders: all twelve candidates that gained at least eight validation items (2.03 points) exceeded the 5% regression budget. Nine were for Qwen3.5-4B, whose prompt edits change the correctness of 56 construction items on average, against 23–27 for the other students. Third, the other evolved harnesses are conservative: each changes the label set on fewer than 4% of evaluation items. The regression gate protects solved cases at the cost of broad edits; whether a looser gate generalizes better needs a separate test set.

Edits are instructions the student may not follow. In Figure 4a the added rule moves the trace toward a neutral alternative. In Figure 4b the rule written for the Neutral→Frustration confusion is ignored: Neutral is never named. These selected cases suggest that the student’s instruction following, not only the Builder’s proposals, can limit what edits achieve. Future gates could check that traces apply edits and that citations match the media.

![](images/24062bf7683ddf9d430c3167b5aececdcc53fbf829a1d7c6b65137a27f8c8d58.jpg)  
Figure 4: Case study: MHE versus baselines on Gemma-4-E4B. The input is drawn once: instruction, transcript, and five of our seven sampled frames per clip (the student saw 2 fps), subtitled with the words spoken within 0.12 s of each frame (every condition except Video receives the transcript); lettered boxes mark the regions that cards cite. Each card gives one condition’s answer (green: correct; red: wrong), and the tags between cards name what each condition adds. Note glyphs: green check (supported); red cross (misread, unsupported, or wrong); amber note (audio-derived); gray dash (limitation). The last MHE note gives the answer of the GPT-6.1 Sol harness; [vi]/[aj]: evidence IDs. The edit panel quotes the rule GPT-5.6 Luna added to the review prompt, with verbatim Shared and MHE excerpts. Cases are selected, not representative.

Emotion context transfers modestly to intent QA. The gains for three students are small, and none survives Holm correction (Table 2). Qwen3.5- 4B, the one student that loses, also loses 2.06 points when only transcript and prosody are added. Its loss may therefore come from added text rather than from emotion content; this student also uses a different answer format (Section 4.3). Conditioning the context on the question could strengthen transfer but would reduce reuse across questions.

Conclusion. EmoHarness boosts Recognition by 7.6–13.4 points over Video for every student. Most of the boost comes from audio-derived evidence (Audio−Video: 5.38–10.71 points). Edits evolved by MHE add at most 0.95 points over Shared, and none survives Holm correction. The harness’s emotion context transfers to IntentQA with small, nonsignificant gains for three of four students.

## Limitations

Evidence base. The media-overlap filter that defines the evaluation split was applied once and has not been independently re-verified. Part of the evaluation set was inspected during development, so -held-out results describe unseen task types rather than an untouched test set. The 129 reserved examples remain unevaluated. We score Recognition against gold labels without the benchmark’s LLM judge, so our values may differ from published MME-Emotion results, and we do not report its Reasoning and Chain-of-Thought scores.

![](images/f9221bf00a0042ec04575aadebf9e085e1b8f9dad10ea040cd6b5fbf6f402acb.jpg)  
Figure 5: Emotion context transfers to IntentQA. Accuracy change (pp) from adding the harness’s emotion context to Audio. Bars: mean over 4 students × 2 Builders (green: gain; red: loss); dots: the 8 individual changes. Rows: all questions, then the official types (temporal: TN+TP). The mean change is positive for every type; none survives Holm correction (Appendix Table 6).

Construction. Each student–Builder pair has one construction trajectory. The two Builders were run about three weeks apart, and the later runs used a slightly extended diagnostic packet. Four generations do not imply matched proposal budgets or compute (Appendix D). The same model acts as Builder and proposal reviewer, and GPT-5.6 Luna also scores the reasoning proxy for both Builders’ candidates. The gate values are fixed design choices that we did not vary. Fixed-harness intervals do not capture construction variance, and nonsignificant corrected tests do not establish equivalence. We do not compare against existing optimizers (e.g., MIPRO, TextGrad, or Meta-Harness itself). Prompt-only, evidence-separated, and compute-matched controls are absent, as are Shared-generated contexts except for Qwen3.5-4B, whose GPT-5.6 Luna harness retained the scaffold.

Measurement. Visual information can be lost when observations are converted to text. Acoustic descriptions depend on reference statistics, speakers, languages, and recording conditions, and the lack of diarization limits multi-speaker interpretation. MME-Emotion clips include English and Mandarin Chinese speech, and we do not analyze results by language. The three-label cap constrains recall, while recall-like Recognition does not penalize excess labels and favors the scaffold on multilabel tasks (Appendix C). We do not report Recognition separately on items with cross-modal conflicts, so whether edits help where they are aimed is untested. Valid citations do not establish grounding or faithful explanations, fallback logic can complete unsupported answers, and static checks and model reviewers do not guarantee correct verification code. The 64-token answer cap leaves some downstream replies unparsed, and Qwen3.5-4B instead answers under a 1,024-token cap with a finalanswer parser, so answer formatting differs across students.

## Ethical Considerations

Benchmark emotion labels are interpretations of behavior, not direct measurements of private mental states or clinical conditions. This work does not validate use in diagnosis, employment screening, surveillance, or educational assessment. Cultural, linguistic, and demographic variation requires separate analysis with suitable annotations. Dataset access and redistribution must follow source terms. Videos, transcripts, and construction diagnostics may identify individuals; external model processing and released traces must respect applicable permissions, and the frames reproduced in our casestudy figures come from public research datasets, with on-screen names removed. We use MME-Emotion (MIT license) and IntentQA (whose repository states no license) for non-commercial research evaluation only; source videos remain subject to their original terms. The student models are released under Apache 2.0, and Whisper and faster-whisper (MIT), openSMILE (audEERING research license), and vLLM (Apache 2.0) are used under their terms. We release no new data.

## References

AmirAli Bagher Zadeh, Paul Pu Liang, Soujanya Poria, Erik Cambria, and Louis-Philippe Morency. 2018. Multimodal language analysis in the wild: CMU-MOSEI dataset and interpretable dynamic fusion graph. In Proceedings of the 56th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 2236–2246. Association for Computational Linguistics.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, et al. 2025. Qwen3-VL technical report. arXiv preprint arXiv:2511.21631.

Xu Cao, Yifan Shen, Bolin Lai, Wenqian Ye, Yunsheng Ma, Joerg Heintz, Jintai Chen, Meihuan Huang, Jianguo Cao, Aidong Zhang, et al. 2024. What is the vi-

sual cognition gap between humans and multimodal llms? arXiv preprint arXiv:2406.10424.

Ching-An Cheng, Allen Nie, and Adith Swaminathan. 2024a. Trace is the next AutoDiff: Generative optimization with rich feedback, execution traces, and LLMs. In Advances in Neural Information Processing Systems, volume 37, pages 71596–71642.

Zebang Cheng, Zhi-Qi Cheng, Jun-Yan He, Jingdong Sun, Kai Wang, Yuxiang Lin, Zheng Lian, Xiaojiang Peng, and Alexander G. Hauptmann. 2024b. Emotion-LLaMA: Multimodal emotion recognition and reasoning with instruction tuning. In Advances in Neural Information Processing Systems, volume 37, pages 110805–110853.

Florian Eyben, Klaus R. Scherer, Björn W. Schuller, Johan Sundberg, Elisabeth André, Carlos Busso, Laurence Y. Devillers, Julien Epps, Petri Laukka, Shrikanth S. Narayanan, and Khiet P. Truong. 2016. The Geneva minimalistic acoustic parameter set (GeMAPS) for voice research and affective computing. IEEE Transactions on Affective Computing, 7(2):190–202.

Florian Eyben, Martin Wöllmer, and Björn Schuller. 2010. openSMILE: The Munich versatile and fast open-source audio feature extractor. In Proceedings of the 18th ACM International Conference on Multimedia, pages 1459–1462.

Gemma Team. 2026. Gemma 4 technical report. arXiv preprint arXiv:2607.02770.

Tanmay Gupta and Aniruddha Kembhavi. 2023. Visual programming: Compositional visual reasoning without training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14953–14962.

Yikun Han, Joey Chan, Jingyuan Chen, Mengting Ai, Simo Du, and Yue Guo. 2026. MedConceal: A benchmark for clinical hidden-concern reasoning under partial observability. In Conference on Language Modeling (COLM).

Sture Holm. 1979. A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics, 6(2):65–70.

Shengran Hu, Cong Lu, and Jeff Clune. 2025. Automated design of agentic systems. In The Thirteenth International Conference on Learning Representations.

Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, Adams Wei Yu, Xinying Song, and Denny Zhou. 2024. Large language models cannot self-correct reasoning yet. In The Twelfth International Conference on Learning Representations.

Sayash Kapoor, Benedikt Stroebl, Zachary S. Siegel, Nitya Nadgir, and Arvind Narayanan. 2025. AI agents that matter. Transactions on Machine Learning Research.

Omar Khattab, Arnav Singhvi, Paridhi Maheshwari, Zhiyuan Zhang, Keshav Santhanam, Sri Vardhamanan, Saiful Haq, Ashutosh Sharma, Thomas T. Joshi, Hanna Moazam, Heather Miller, Matei Zaharia, and Christopher Potts. 2024. DSPy: Compiling declarative language model calls into stateof-the-art pipelines. In The Twelfth International Conference on Learning Representations.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient memory management for large language model serving with PagedAttention. In Proceedings ofthe 29th Symposium on Operating Systems Principles, pages 611–626.

Yoonho Lee, Roshen Nair, Qizheng Zhang, Kangwook Lee, Omar Khattab, and Chelsea Finn. 2026. Meta-Harness: End-to-end optimization of model harnesses. In Conference on Language Modeling (COLM).

Bingxuan Li, Yiming Cui, Yicheng He, Yiwei Wang, Shu Zhang, Longyin Wen, and Yulei Niu. 2026a. Echofoley: Event-centric hierarchical control for video grounded creative sound generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 27229–27238.

Bingxuan Li, Yining Hong, Cheng Qian, Hyeonjeong Ha, Jiateng Liu, Zhenhailong Wang, Yue Guo, Yunzhu Li, and Heng Ji. 2026b. Trimming the long-tail of visual world modeling evaluation. arXiv preprint arXiv:2606.24256.

Bingxuan Li, Jeonghwan Kim, Cheng Qian, Xiusi Chen, Eitan Anzenberg, Niran Kundapur, and Heng Ji. 2026c. PEARL: Self-evolving assistant for time management with reinforcement learning. In Proceedings ofthe 64th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 33522–33539, San Diego, California, United States. Association for Computational Linguistics.

Bingxuan Li, Yiwei Wang, Jiuxiang Gu, Kai-Wei Chang, and Nanyun Peng. 2025. METAL: A multi-agent framework for chart generation with test-time scaling. In Proceedings of the 63rd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 30054–30069.

Boyi Li, Yifan Shen, Yuanzhe Liu, Yifan Xu, Jiateng Liu, Xinzhuo Li, Zhengyuan Li, Jingyuan Zhu, Yunhan Zhong, Fangzhou Lan, et al. 2026d. Toward cognitive supersensing in multimodal large language model. arXiv preprint arXiv:2602.01541.

Jiapeng Li, Ping Wei, Wenjuan Han, and Lifeng Fan. 2023. IntentQA: Context-aware video intent reasoning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11963– 11974.

Paul Pu Liang, Akshay Goindani, Talha Chafekar, Leena Mathur, Haofei Yu, Ruslan Salakhutdinov, and Louis-Philippe Morency. 2024. HEMM: Holistic evaluation of multimodal foundation models. In Advances in Neural Information Processing Systems, volume 37, pages 42899–42940.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. 2023. Self-Refine: Iterative refinement with self-feedback. In Advances in Neural Information Processing Systems, volume 36, pages 46534–46594.

Krista Opsahl-Ong, Michael J Ryan, Josh Purtell, David Broman, Christopher Potts, Matei Zaharia, and Omar Khattab. 2024. Optimizing instructions and demonstrations for multi-stage language model programs. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 9340–9366. Association for Computational Linguistics.

Reid Pryzant, Dan Iter, Jerry Li, Yin Tat Lee, Chenguang Zhu, and Michael Zeng. 2023. Automatic prompt optimization with “gradient descent” and beam search. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 7957–7968. Association for Computational Linguistics.

Qwen Team. 2026. Qwen3.5: Towards native multimodal agents. Blog post.

Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. 2023. Robust speech recognition via large-scale weak supervision. In Proceedings ofthe 40th International Conference on Machine Learning, volume 202 of Proceedings ofMachine Learning Research, pages 28492–28518. PMLR.

Yifan Shen, Pei Tian, Xinzhuo Li, Bowen Fang, Shujun Xia, Bingxuan Li, Ana Jojic, Wenming Ye, Xu Cao, James Matthew Rehg, et al. 2026. Cogniroute: Learning to route social evidence in omni-modal models. arXiv preprint arXiv:2606.20970.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. 2023. Reflexion: Language agents with verbal reinforcement learning. In Advances in Neural Information Processing Systems, volume 36, pages 8634–8652.

Miles Turpin, Julian Michael, Ethan Perez, and Samuel R. Bowman. 2023. Language models don’t always say what they think: Unfaithful explanations in chain-of-thought prompting. In Advances in Neural Information Processing Systems, volume 36, pages 74952–74965.

Alex Wilf, Leena Mathur, Sheryl Mathew, Claire Ko, Youssouf Kebe, Paul Pu Liang, and Louis-Philippe

Morency. 2023. Social-IQ 2.0 challenge: Benchmarking multimodal social understanding. GitHub repository.

Yifan Xu, Yixuan Li, Xinzhuo Li, Yixin Gu, Yifan Shen, Lijun Yu, and Haohan Wang. 2026. Steve: Stabilizing textual gradient-based prompt optimization via error-driven refinement and regularized verification. arXiv preprint arXiv:2609.23716.

Chengrun Yang, Xuezhi Wang, Yifeng Lu, Hanxiao Liu, Quoc V. Le, Denny Zhou, and Xinyun Chen. 2024. Large language models as optimizers. In The Twelfth International Conference on Learning Representations.

Chenyang Yang, Xinran Zhao, Tongshuang Wu, and Christian Kästner. 2026. Better harnesses, smaller models: Building 90% cheaper agents via automated harness adaptation. arXiv preprint arXiv:2607.08938.

Mert Yuksekgonul, Federico Bianchi, Joseph Boen, Sheng Liu, Pan Lu, Zhi Huang, Carlos Guestrin, and James Zou. 2025. Optimizing generative AI by backpropagating language model feedback. Nature, 639(8055):609–616.

Fan Zhang, Zebang Cheng, Chong Deng, Haoxuan Li, Zheng Lian, Qian Chen, Huadai Liu, Wen Wang, Yi-Fan Zhang, Renrui Zhang, Ziyu Guo, Zhihong Zhu, Hao Wu, Haixin Wang, Yefeng Zheng, Xiaojiang Peng, Xian Wu, Kun Wang, Xiangang Li, and 2 others. 2026. MME-Emotion: A holistic evaluation benchmark for emotional intelligence in multimodal large language models. In The Fourteenth International Conference on Learning Representations.

Jiayi Zhang, Jinyu Xiang, Zhaoyang Yu, Fengwei Teng, Xionghui Chen, Jiaqi Chen, Mingchen Zhuge, Xin Cheng, Sirui Hong, Jinlin Wang, Bingnan Zheng, Bang Liu, Yuyu Luo, and Chenglin Wu. 2025. AFlow: Automating agentic workflow generation. In The Thirteenth International Conference on Learning Representations.

Jiaxing Zhao, Xihan Wei, and Liefeng Bo. 2025. R1- Omni: Explainable omni-multimodal emotion recognition with reinforcement learning. arXiv preprint arXiv:2503.05379.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. In Advances in Neural Information Processing Systems, volume 36, pages 46595–46623.

Yongchao Zhou, Andrei Ioan Muresanu, Ziwen Han, Keiran Paster, Silviu Pitis, Harris Chan, and Jimmy Ba. 2023. Large language models are human-level prompt engineers. In The Eleventh International Conference on Learning Representations.

Kunlun Zhu, Zijia Liu, Bingxuan Li, Muxin Tian, Yingxuan Yang, Jiaxun Zhang, Pengrui Han, Qipeng Xie, Fuyang Cui, Weijia Zhang, et al. 2025. Where LLM agents fail and how they can learn from failures. arXiv preprint arXiv:2509.25370.

## A Implementation Details

Scoring and statistics. All recognition means and QA accuracies are computed from item-level predictions. MHE−Shared and downstream intervals use paired bootstraps (task-stratified for recognition, clustered by video for QA); Audio−Video, Shared−Audio, and per-condition intervals use the same task-stratified procedure. All bootstraps draw 10,000 resamples.

Evaluation split. The evaluation set (5,507 items on 5,083 media) excludes 70 items whose media overlap construction, validation, or the 129 reserved examples. It contains 986 items that were inspected during development and 4,521 that were not; because we cannot identify which items were inspected, results excluding them cannot be reported. Inspection could have influenced any human design choice, most plausibly the Shared scaffold. It did not inform the Builder’s edits directly, because the Builder never saw evaluation items.

Construction runs. Each harness comes from one four-generation run. The GPT-6.1 Sol runs were made about three weeks after the GPT-5.6 Luna runs, and their diagnostic packet lists one additional tool (about 58 tokens); first-generation inputs are otherwise identical across Builders. Each run’s validation trajectory follows the incumbent after each generation’s promotion decision. Table 3 lists the construction records, Figure 6 traces every generation, and Figure 7 contrasts validation and evaluation gains. A third Builder configuration, also run on all four students, is not analyzed further: one of its 24 validation-scored proposals was promoted (for Qwen3-VL-4B, worth +2.03 validation points but +0.04 task-macro points on evaluation), and including it leaves every conclusion unchanged (12-test families; smallest adjusted $p \mathrm { : }$ MHE−Shared .288, IntentQA .127).

Serving and editable interfaces. Native video is decoded at 2 fps with at most 32 frames; auxiliary frame-cache sampling is a separate path, and the Shared scaffold attaches no auxiliary images or face crops to later stages. Emotion-context generation uses a 16,384-token context window and downstream answering uses 32,768. Prompt and configuration edits target existing prompt and YAML files, and full-bundle search also permits restricted verification Python files; perception tools, schemas, and the output controller are outside that scope.

<table><tr><td colspan="3"></td><td colspan="2">Dv Rec. (%)</td><td colspan="2">Seen</td></tr><tr><td>Student Scored</td><td></td><td>Prom.</td><td>Start</td><td>Final</td><td>eval.</td><td>USD</td></tr><tr><td>VL-8B</td><td>14</td><td>2</td><td>44.92</td><td>47.46</td><td>+0.10</td><td>0.41</td></tr><tr><td>VL-4B</td><td>10</td><td>2</td><td>45.94</td><td>47.97</td><td>+0.20</td><td>0.49</td></tr><tr><td>GP -na Q3.5</td><td>10</td><td>0</td><td>41.12</td><td>41.12</td><td>0.00†</td><td>0.23</td></tr><tr><td>Gemma</td><td>12</td><td>2</td><td>42.89</td><td>43.65</td><td>+0.10</td><td>0.40</td></tr><tr><td>VL-8B GP6-oI</td><td>12</td><td>1</td><td>44.92</td><td>45.94</td><td>+0.30</td><td>1.51</td></tr><tr><td>VL-4B</td><td>11</td><td>1</td><td>45.94</td><td>46.19</td><td>+0.30</td><td>1.49</td></tr><tr><td>Q3.5</td><td>11</td><td>1</td><td>41.12</td><td>41.88</td><td>+0.10</td><td>1.53</td></tr><tr><td>Gemma</td><td>14</td><td>2</td><td>42.89</td><td>43.91</td><td>-0.20</td><td>1.65</td></tr></table>

Table 3: MHE construction records. Validation Recognition is a flat item mean over 394 items, distinct from evaluation task-macro Recognition. Every run has four generations. Scored: proposals scored on $\mathcal { D } _ { v }$ ; Prom.: promotions. Proposals screened out by static checks or the proposal reviewer before validation: 5 (GPT-5.6 Luna) and 9 (GPT-6.1 Sol). USD covers logged Builder, reviewer, and proxy-judge calls, excluding local GPU work. Seen eval.: gain over Shared as a flat item mean on the 2,963 seen-type evaluation items, comparable to Final−Start. VL-8B/VL-4B: Qwen3-VL-8B/4B; Q3.5: Qwen3.5-4B; Gemma: Gemma-4-E4B. <sup>†</sup> Retained scaffold with identical labels.

![](images/e69509c475dc579b74bf7c77d63830ff7e28c4bd7cbb4104efb0e9ac7f46ee74.jpg)  
Figure 6: MHE trajectories on the validation set. Incumbent validation Recognition (item mean over 394 items) per generation for each Builder; markers are candidates scored on validation (filled: promoted; open: rejected by the gates); × marks proposals screened out by the checks or the reviewer, drawn at the panel floor and never scored. S: Shared scaffold; diamond: run with no promotion.

## B Execution and Construction Details

This appendix expands on the execution and construction rules in Section 3; it defines no additional evaluated conditions.

![](images/3df320101262ba61132b13dc394081fa6f0097c7d569c01b193689da5c806787.jpg)

Figure 7: Validation gains overstate evaluation gains. Gain of each evolved harness over Shared on validation (item mean over 394 items) and on evaluation (task macro over 5,507 items). Connected points describe the same harness under two summaries, not two estimates of one quantity.  
Meta-Harness Evolving   
Input: student M<sub>θ</sub>, Shared scaffold H<sub>0</sub>, construction   
data $\mathcal { D } _ { b } ,$ validation data D<sub>v</sub>, generations T.   
$H \gets H _ { 0 } ;$ initialize the proposal history and the solved   
set S from H.   
For $t = 1 , \dots , T \colon$   
1. Run H on $\mathcal { D } _ { b } ;$ cluster failures by gold–prediction   
confusion and cross-modal conflict.   
2. Variation: the Builder proposes up to four can  
didate harnesses, editing prompts, settings, and   
restricted verification code.   
3. Reject candidates that fail the static checks or the   
proposal reviewer.   
4. Selection: score survivors on D<sub>v</sub> (computing the   
reasoning proxy only for those passing the cheaper   
gates) and apply Eq. 3.   
5. Promote the best passing candidate or retain H;   
update S and the history.   
Output: the frozen evolved harness $H ^ { \star } = H$ , evaluated   
only after construction.  
Algorithm 1: Meta-Harness Evolving. The Builder proposes edits to the harness, and checks and validation gates decide which survive. The Builder never answers evaluation items.

## B.1 Perception and Evidence Assembly

Whisper produces word-timestamped transcripts. Acoustic features are verbalized relative to reference means and standard deviations: an absolute standardized deviation of at least one yields a descriptor. Missing features and zero-variance reference features are omitted, and the descriptions neither identify speakers nor label emotion.

Native-video decoding and auxiliary framecache sampling are separate paths (Appendix A). Deterministic assembly assigns identifiers v1, . . . to visual observations and a1, . . . to transcript sentences and prosody phrases. The request target is stored as t1 but cannot be cited as observed decision evidence. Each hypothesis has a label, a model-generated confidence, supporting identifiers, and contradicting identifiers. Valid identifiers do not establish semantic support.

## B.2 Output and Failure Handling

The controller enforces the requested vocabulary and cardinality, with a three-label cap for multilabel outputs. A candidate normally needs a valid supporting identifier and more support than contradiction. If too few pass, remaining slots are filled from available rankings and labels. This is completion logic, not an evidence guarantee.

The Explain call receives the record and verified interpretations and writes context, facial and bodily cues, vocal and verbal cues, and a cross-modal synthesis with the strongest alternative. The controller sanitizes citations and appends a decision section. With upstream outputs fixed, the explanation cannot change the labels, but it can change the downstream context. One retry per malformed stage is permitted under the shared six-call cap; once the cap is reached, the remaining stages are skipped and the output is completed from available information.

## B.3 Edit Spaces and Selection History

All runs search the full edit space (existing prompts, YAML settings, and restricted verification files) directly from Shared. Each proposal supplies complete replacements for the allowed files and names its target construction-failure cluster. The solved set S<sub>t</sub> (Eq. 2) grows with each promotion. The proposal history contains proposals, construction effects, and check and reviewer feedback; validation scores and gate decisions are withheld, although validation still decides which incumbent supplies the next diagnostics.

MHE obtains reasoning-proxy scores for the initial harness and for candidates that pass the cheaper recognition, regression, and call-budget gates before applying Eq. 3. Static checks and the proposal reviewer constrain candidates but cannot guarantee their safety or semantic correctness.

(a) GPT-5.6 Luna (b) GPT-6.1 Sol  
![](images/1beb0cc51a6f9a51aaa69c0cb819dd09e75abbe73cfd785987c4a002963d9c95.jpg)  
Figure 8: Per-task MHE−Shared changes. Differences (pp) for each Builder. Both panels share one zero-centered color scale, and the horizontal rule separates the three seen task types from the five held-out types. Exact scores are in Table 1.

Reasoning proxy used for selection. We evaluate $\widehat { Q } _ { v }$ on a fixed 200-item sample from $\mathcal { D } _ { v }$ . The judge receives each response, up to ten cached frames, and gold labels; the benchmark’s judgeside audio-clue field is left empty. Its rubric gives binary scores to non-final reasoning steps, marking clear contradictions with the supplied cues as incorrect. The per-item proxy is the mean of these non-final scores, or zero if none exist; $\widehat { Q } _ { v }$ averages the per-item values. The final decision-step score is excluded. Following the LLM-as-judge protocol of Zheng et al. (2023), the judge is GPT-5.6 Luna. The proxy does not measure official reasoning performance or explanation faithfulness.

## C Detailed Recognition Results

Figure 9 and Table 4 report the Audio−Video and Shared−Audio contrasts and the eight MHE−Shared contrasts, the latter with Holmadjusted probabilities; a pointwise interval that excludes zero can coexist with a nonsignificant corrected test. Per-task Recognition for all five conditions appears in Table 1, and Figure 8 visualizes the per-task MHE−Shared changes. Task-level values are descriptive and not separately corrected.

Metric interpretation. ER-Lab, ER-Wild, and SA are the seen types (177, 1,491, and 1,295 items); Noise-ER, FG-ER, ML-ER, FG-SA, and IR are held out (500, 500, 590, 461, and 493 items). FG-ER and ML-ER use recall-like multi-label Recognition, which should not be read as precision, exact match, or balanced F1. This choice favors the scaffold: on these two tasks Shared predicts 2.2–2.9 labels per item against 1.5–2.0 for Audio. Scoring FG-ER and ML-ER by set F1 instead changes Shared−Audio to −0.25, −1.41, +1.10, and +1.00 points (VL-8B, VL-4B, Q3.5, Gemma), while Audio−Video (+6.05 to +11.17) and MHE−Shared (at most +0.80) barely move. Under set F1, every evolved harness for the two Qwen3-VL students scores below single-call Audio (−0.05 to −1.16 points), so for them the four-call harness adds nothing over the audio-derived evidence.

(a) Meta-Harness Evolving: MHE−Shared
<table><tr><td>Builder</td><td></td><td>∆</td><td>95% CI</td><td>pHolm</td></tr><tr><td>VL-8B</td><td>GPT-5.6 Luna GPT-6.1 Sol</td><td>+0.21 +0.06</td><td> $\cdot 0 . 1 8 \mathrm { t o } + 0 . 5 9$   $- 0 . 2 0 \mathrm { t o } + 0 . 2 9$ </td><td>1.000 1.000</td></tr><tr><td>VL-4B</td><td>GPT-5.6 Luna GPT-6.1 Sol</td><td>+0.95 +0.25</td><td> $+ 0 . 1 2 \ \mathrm { t o } + 1 . 8 2$   $+ 0 . 0 1 \ \mathrm { t o } + 0 . 4 9$ </td><td>.192 .262</td></tr><tr><td>Q3.5</td><td>GPT-5.6 Luna GPT-6.1 Sol</td><td> $0 . 0 0 ^ { \dag }$   $+ 0 . 0 3$ </td><td>0.00 to 0.00  $- 0 . 1 0 \mathrm { ~ t o ~ } { + 0 . 1 6 }$ </td><td>1.000 1.000</td></tr><tr><td>Gemma</td><td>GPT-5.6 Luna GPT-6.1 Sol</td><td> $- 0 . 0 7$   $+ 0 . 1 1$ </td><td> $- 0 . 3 3 \mathrm { t o } + 0 . 2 0$   $- 0 . 1 9 \mathrm { t o } + 0 . 4 2$ </td><td>1.000 1.000</td></tr></table>

<table><tr><td colspan="4">(b) Added evidence and scaffold</td></tr><tr><td colspan="3">Contrast</td><td>95% CI</td></tr><tr><td> $\mathrm { V I } { - } 8 \mathrm { B }$ </td><td>Audio—Video Shared—Audio</td><td>+9.20 +1.32</td><td> $+ 7 . 7 6 \mathrm { t o } + 1 0 . 6 3$   $- 0 . 2 8 \mathrm { t o } + 2 . 9 2$ </td></tr><tr><td>VL-4B</td><td>Audio-Video Shared—Audio</td><td>+9.00 -0.79</td><td> $+ 7 . 6 3 \mathrm { t o } + 1 0 . 3 9$   $- 2 . 2 5 \mathrm { t o } + 0 . 6 1$ </td></tr><tr><td> $\mathbf { Q } 3 . 5$ </td><td>Audio—Video Shared—Audio</td><td> $+ 5 . 3 8$   $+ 2 . 2 3$ </td><td>+4.02 to +6.76  $+ 0 . 7 3 \mathrm { t o } + 3 . 7 0$ </td></tr><tr><td>Gemma</td><td>Audio—Video Shared—Audio</td><td>+10.71 +2.54</td><td> $+ 9 . 1 6 \mathrm { t o } + 1 2 . 2 9$   $+ 1 . 1 2 \ \mathrm { t o \ + 3 . 9 5 }$ </td></tr></table>

Table 4: Recognition contrasts with intervals. Taskmacro Recognition differences (pp) with pointwise 95% intervals. (a) MHE−Shared, with task-stratified paired bootstrap intervals (10,000 resamples) and Holmadjusted $p .$ No difference survives Holm correction over eight tests (smallest adjusted $p = . 1 9 2 )$ . <sup>†</sup> Retained scaffold with identical labels. (b) Audio−Video and Shared−Audio, with intervals from the same procedure (pointwise; not part of a Holm family). VL-8B/VL-4B: Qwen3-VL-8B/4B; Q3.5: Qwen3.5-4B; Gemma: Gemma-4-E4B.

## D Resource Accounting

Table 3 reports the API charges logged for each construction run, including prompt-cache discounts.

(a) Added evidence and scaffold  
![](images/8f17821e74a08ab3cc6ea4eea9c831744fc65189ba30a26ef1be349a5b1ca2bd.jpg)

![](images/7f9252a2d89dcaa9cecd3fadfbdd46c96e5ee5396edda2a10daaccfe69db4ff2.jpg)  
Figure 9: Decomposing recognition gains. Taskmacro contrasts (pp) on 5,507 MME-Emotion items. (a) Audio-derived text adds 5.4–10.7 points, and the Shared scaffold changes Audio by between −0.8 and +2.5. (b) MHE adds at most +0.95 over Shared (note the narrower axis), and no contrast survives Holm correction over eight tests. Whiskers are pointwise 95% paired bootstrap intervals; the diamond marks the run that retained the scaffold.

They cover the Builder, the proposal reviewer, and the proxy judge, and exclude local GPU work and fixed preprocessing, so they are not a matched endto-end efficiency comparison. Each construction run cost \$0.23–0.49 with GPT-5.6 Luna and \$1.49– 1.65 with GPT-6.1 Sol. All student inference ran locally with vLLM on two NVIDIA A100 80GB GPUs; we did not record GPU-hours, so we report call volume instead. The eight construction runs issued 9,824–25,312 student calls each (162,007 in total; 292M input and 30M output tokens), MME-Emotion evaluation issued 309,226 student calls across the five reported conditions, and the downstream study generated 11,848 emotion contexts and answered 78,448 questions. Inference with Shared or an evolved harness uses about four student calls per item instead of one. For V videos and Q questions, reusing emotion context costs

$$
\begin{array} { r l } & { C _ { \mathrm { r e u s e } } = C _ { \mathrm { b u i l d } } + V ( C _ { \mathrm { p e r c e p t i o n } } + C _ { \mathrm { e m o t i o n } } ) } \\ & { ~ + Q C _ { \mathrm { Q A + c o n t e x t } } . } \end{array}\tag{5}
$$

Reuse amortizes context generation across questions but repeats context tokens in every answer; this accounting does not establish a cost advantage.

<table><tr><td>Student</td><td>Builder</td><td>∆</td><td>95% CI</td><td>pHolm</td></tr><tr><td rowspan="2">VL-8B</td><td>GPT-5.6 Luna</td><td>+0.47</td><td> $- 1 . 1 5 \mathrm { t o } + 2 . 0 5$ </td><td>1.000</td></tr><tr><td>GPT-6.1 Sol</td><td>+0.23</td><td> $- 1 . 3 3 \mathrm { t o } + 1 . 7 7$ </td><td>1.000</td></tr><tr><td rowspan="2">VL-4B</td><td>GPT-5.6 Luna</td><td></td><td> $+ 1 . 8 7 + 0 . 4 7 \mathrm { t o } + 3 . 2 7$ </td><td>.085</td></tr><tr><td>GPT-6.1 Sol</td><td></td><td> $+ 1 . 5 0 \mathrm { ~ \ } + 0 . 0 5 \mathrm { t o \ } + 2 . 9 7$ </td><td>.344</td></tr><tr><td rowspan="2">Q3.5</td><td>GPT-5.6 Luna</td><td></td><td> $- 1 . 5 0 ^ { \dagger } \ - 3 . 0 5 \ \mathrm { t o } + 0 . 0 9$ </td><td>.395</td></tr><tr><td>GPT-6.1 Sol</td><td>-1.41</td><td> $- 2 . 9 1 \ \mathrm { t o } \ + 0 . 1 5$ </td><td>.426</td></tr><tr><td rowspan="2">Gemma</td><td>GPT-5.6 Luna</td><td></td><td> $+ 1 . 0 3 - 0 . 6 3 \mathrm { t o } + 2 . 6 6$ </td><td>.695</td></tr><tr><td>GPT-6.1 Sol</td><td></td><td> $+ 1 . 3 6 - 0 . 2 4 \mathrm { t o } + 2 . 9 6$ </td><td>.426</td></tr></table>

Table 5: Effect of emotion context on downstream QA. Audio+Emotion−Audio accuracy (pp) on the 2,134 IntentQA test questions, with pointwise 95% intervals from a paired bootstrap clustered by video (10,000 resamples). No contrast survives Holm correction over eight tests (smallest adjusted $p = . 0 8 5 )$ . <sup>†</sup> Retained scaffold: this context is generated by Shared.

![](images/54d297a4c082b943a69069e7110e9058d799ff3bfc3d2aca6ab75faac134c794.jpg)  
Figure 10: Emotion context added to Audio on IntentQA. Accuracy change (pp); whiskers: pointwise 95% paired intervals clustered by video. No contrast survives Holm correction (eight tests). Frames and audio text are identical in both conditions; the gray diamond (Qwen3.5-4B, GPT-5.6 Luna) is the retained Shared scaffold’s context.

## E Downstream Protocol and Diagnostics

The eight primary Audio+Emotion−Audio contrasts in Table 5 have pointwise intervals and probabilities Holm-adjusted over the eight tests; none survives correction. The context contains the predicted emotion, its confidence, alternatives, and analyses of setting, facial and bodily cues, vocal and verbal cues, and cross-modal interpretation; it is generated without any downstream question.

## F Additional Case Studies

Figure 11 shows two further selected MME-Emotion cases in the layout of Figure 4. In (a), for Qwen3-VL-4B, a clip pairs a smile with the words “And I hate to say that”; the GPT-5.6 Luna harness reaches the reference negative, whereas the GPT-6.1 Sol harness does not. In (b), the Qwen3- VL-8B harness evolved by GPT-5.6 Luna switches

<table><tr><td></td><td colspan="2">VL-8B</td><td colspan="2">VL-4B</td><td colspan="2">Q3.5</td><td colspan="2">Gemma</td></tr><tr><td>Stratum</td><td>Luna</td><td>Sol</td><td>Luna</td><td>Sol</td><td>Luna</td><td>Sol</td><td>Luna</td><td>Sol</td></tr><tr><td>Why (CW) n = 1,250 How (CH)</td><td>+1.04 −1.0 to +3.1 -2.23</td><td>+0.48 −1.5 to +2.4 -2.51</td><td>+1.36 −0.6 to +3.3 +2.51</td><td>+1.04 −0.9 to +2.9 +3.90</td><td>-1.68 −3.6 to +0.2 -0.84</td><td>-1.76 -3.6 to +0.2 -0.56</td><td>+0.96 −1.1 to +3.2 +0.84</td><td>+1.44 −0.6 to +3.6 +1.11</td></tr></table>

Table 6: Emotion context by IntentQA question type. Audio+Emotion−Audio accuracy (pp) within exploratory strata given by the official types of the 2,134 test questions, never by outcomes: causal-why (CW), causal-how (CH), and temporal (TN+TP, next and previous); every stratum is reported. Gray: pointwise 95% interval from a paired bootstrap clustered by video (10,000 resamples); n: questions in the stratum. No contrast survives Holm correction within its stratum (8 tests: 4 students × 2 Builders; smallest adjusted p = .278). Luna: GPT-5.6 Luna; Sol: GPT-6.1 Sol. VL-8B/VL-4B: Qwen3-VL-8B/4B; Q3.5: Qwen3.5-4B; Gemma: Gemma-4-E4B.

![](images/6c2e59d49475649213e98c51ab53379571a1c5da5b27f5340afb024561eb61b9.jpg)  
<sub>-</sub>Figure 11: Additional cases. Selected, not representative; letters, note glyphs, and subtitles as in Figure 4 (no headline tags or edit panels); merged cards report two conditions that agree. (a) Qwen3-VL-4B, FG-SA: the GPT-5.6 Luna harness reaches the reference negative; GPT-6.1 Sol’s does not. (b) Qwen3-VL-8B, ER-Lab: the correct Excited fits the evolved ranking code, while the written explanation favors Happiness.

Shared’s Happiness to the correct Excited, while its written explanation still leans toward Happiness. The switch is consistent with its evolved ranking code, which boosts audio-supported alternatives, although its intermediate ranking is unavailable; explanations need not track the decision rule.

## G Controls for Future Work

The controls below would sharpen the comparisons in this paper.

Evidence and computation. Transcript-only and prosody-only conditions would separate the sources of the audio gain. A one-call structured prediction using the same tools and controller would test decomposition against a direct call that uses the same output contract. Repeated direct samples with a fixed aggregation rule would test the value of additional calls. Token and call counts must be measured rather than presumed matched. Giving the decision stages access to the media would measure how much information is lost when observations are converted to text.

Construction and context. Comparing promptor configuration-only search with full-bundle search requires matched splits and budgets. Evaluating the twelve candidates that the regression gate rejected despite gains of at least eight validation items, or rerunning selection with a looser δ, would test whether the gate rather than the proposals limits MHE. Independent construction repetitions are necessary to assess search variability. Sharedgenerated contexts for every student are needed to test the downstream increment from evolution. Length-matched visual captions and non-emotion summaries would separate emotion-specific utility from generic additional text. The reserved examples require a separately frozen evaluation, and explanation quality requires its own measurement.