# Sequential Trajectories and Simultaneous Blending: Multi-Emotion Modeling for Instruction-Following TTS

Yan Zhou, Yun Hong, Yang Feng<sup>∗</sup>

Key Laboratory of Intelligent Information Processing, Institute of Computing Technology, Chinese Academy of Sciences (ICT/CAS)

State Key Laboratory of AI Safety, Institute of Computing Technology, Chinese Academy of Sciences University of Chinese Academy of Sciences, Beijing, China zhouyan23z@ict.ac.cn, fengyang@ict.ac.cn

## Abstract

Natural-language instructions enable flexible control of synthesized speech, yet emotional TTS systems primarily model a single utterance-level afect, leaving multi-emotion control underexplored. We study two complementary multi-emotion TTS tasks: emotion trajectory, which spans several ordered affective stages, and emotion blending, in which multiple emotions coexist throughout an utterance. These tasks expose a supervision mismatch: supervised fine-tuning (SFT) does not explicitly evaluate emotion features, while single-emotion rewards provide neither structure-aware feedback for trajectory completion nor pair-aware feedback for blending. We introduce HybridEmo, a post-training framework that initializes both tasks with SFT and then aligns the speech-token policy through Group Relative Policy Optimization using a sampleaware hybrid reward. For trajectory samples, segment-aligned consistency combines average and weakest-stage evidence to preserve the correctness and completeness of prescribed stages. For blending samples, a GMM-based reward combines frame-level support from the union of target-emotion anchors in an ofline emotion space with an utterance-level weakertarget margin. Both branches share an ASR reward and are routed within a unified policy. On MultiEmo-Test, HybridEmo significantly improves trajectory correctness and blending intensity, without a noticeable degradation in speaker similarity. Human evaluation prefers HybridEmo to CosyVoice 3 and EmoVoice-0.5B, with nearly balanced preferences against Qwen3-TTS.<sup>1</sup>

## Introduction

Natural-language instructions enable flexible control of synthesized speech, but current emotional text-to-speech (TTS) systems largely model a single utterance-level emotion (Du et al. 2025; Yang et al. 2025; Chen et al. 2026). However, real-world utterances may contain multi-emotion patterns which can be categorized into two forms: sequential evolution and simultaneous blending. On these grounds, we study multi-emotion control in TTS: given the text, an emotion instruction and a reference voice, the task is to generate intelligible speech following the requested emotion pattern while retaining the reference timbre. The two forms of the multi-emotion control are defined as follows (illustrated in

Figure 1): Emotion Trajectory arranges multiple emotions sequentially within an utterance, whereas Emotion Blending expresses multiple emotions concurrently. Both forms require modeling relationships among emotions beyond a single utterance-level label. Therefore, training a TTS model with multi-emotion control requires endowing it with the ability to capture the precise emotional patterns within speech.

However, conventional token-level supervised fine-tuning (SFT) alone is inadequate for multi-emotion control. Multiemotion TTS requires both stable speech generation and structured emotional-pattern realization. Learning both from large-scale triplets of input text, emotion instructions, and multi-emotion target speech is costly, while cross-entropy fits target speech-token sequences without explicitly assessing the prescribed sequential or concurrent pattern. SFT therefore provides a necessary initialization but insuficient direct supervision for precise multi-emotion control.

Reinforcement learning (RL) ofers a complementary solution by explicitly evaluating emotional patterns in generated speech. Starting from an SFT-initialized model, RL can optimize the speech-token policy by scoring its generated waveforms, without requiring a paired target waveform for each condition. This makes RL a better match for multi-emotion instructions that admit diverse valid realizations, while allowing emotion optimization to be combined with content-preservation objectives. However, existing speech RL objectives cannot adequately evaluate the requested multi-emotion patterns. Recent methods optimize global properties such as intelligibility and speaker similarity (Sun et al. 2025; Liu et al. 2025), or apply GRPO to flexible style control (Chen et al. 2026), but do not explicitly evaluate relationships among multiple emotions. Moreover, emotion recognizers typically provide utterance-level scores for individual classes (Ma et al. 2024). For trajectories, such scores cannot localize emotion changes or detect missing and misordered emotions; for blending, they cannot measure whether multiple targets coexist. The key challenge is therefore to construct task-matched rewards for sequential and concurrent emotional patterns.

To meet this challenge, we propose HybridEmo, a twostage post-training framework that initializes multi-emotion generation through SFT and then aligns the speech-token policy through Group Relative Policy Optimization (GRPO) with a sample-aware hybrid reward. For trajectories, a segment-aligned consistency reward combines mean and weakest-stage emotion evidence to assess both the overall correctness and the completion of prescribed emotional stages. For blending, a GMM-based mixture-density reward combines graded frame-level target-pair compatibility with an utterance-level weaker-target margin, encouraging both emotions to coexist while preventing single-target dominance. A sample-aware router applies the appropriate emotion reward to each sample type, while a shared ASR reward preserves linguistic content during emotion optimization.

![](images/0dc342d3471ef94f3247e1902479eb9e5552d54ae3d325692a2e59f62d1dc4f6.jpg)  
Figure 1: Multi-emotion control through sequential emotion trajectories and concurrent emotion blending.

We construct MultiEmo-Test to evaluate emotion trajectory and blending tasks. HybridEmo significantly improves trajectory correctness and blending-oriented perceptual scores without a noticeable degradation in speaker similarity. Human evaluation also favors HybridEmo over CosyVoice 3 and EmoVoice-0.5B, with a near-balanced preference compared with Qwen3-TTS.

Our main contributions are as follows:

• We formulate multi-emotion control through emotion trajectories and blending, and introduce the MultiEmo-Test evaluation set.

• We propose HybridEmo, a two-stage post-training framework that combines supervised initialization with sampleaware GRPO, routing task-matched emotion rewards alongside shared content feedback.

## Background

## Discrete Speech-Token TTS

Discrete speech tokens provide a common interface between speech waveforms and language-model-based generation. VALL-E (Wang et al. 2023) casts zero-shot TTS as conditional language modeling over neural-codec codes, while SPEAR-TTS (Kharitonov et al. 2023) factoarizes generation into text-to-semantic and semantic-to-acoustic stages to exploit audio-only data. The CosyVoice series (Du et al. 2024, 2025) combines an autoregressive speech-token large language model (LLM) with a flow-matching acoustic generator for controllable synthesis and zero-shot voice cloning. Spark-TTS (Wang et al. 2025) proposes BiCodec to encode linguistic content and speaker attributes in a decoupled single stream, while Qwen3-TTS (Hu et al. 2026) develops complementary tokenizers for high-fidelity and streaming synthesis.

## Instruction-Based and Emotional TTS

Natural-language prompting replaces fixed style labels with a more expressive way of control. PromptTTS (Guo et al. 2023) conditions TTS on textual descriptions of style, and PromptTTS 2 (Leng et al. 2024) adds a variation network to model vocal factors underspecified by text. InstructTTS (Yang et al. 2024) maps free-form style prompts into a discrete acoustic latent space while disentangling style, speaker, and content. ControlSpeech (Ji et al. 2025) combines content, style, and speech prompts in a decoupled codec space for simultaneous speaker and style control. Specializing toward afective expression, EmoVoice (Yang et al. 2025) uses an LLM to interpret fine-grained emotion descriptions and predicts phoneme and audio tokens in parallel for content consistency. Collectively, these methods broaden controllability from closed attribute inventories to free-form descriptions, but largely treat style or emotion as an utterance-level condition rather than explicitly modeling sequential emotion trajectories or simultaneous emotion blending.

## Reinforcement Learning for Speech Generation

Reinforcement learning (RL) complements token-level supervision with sequence-level preference or reward signals that better reflect perceptual and task-specific speech quality. Direct Preference Optimization (DPO) (Rafailov et al. 2023) learns directly from preference pairs without fitting an explicit reward, and Emo-DPO (Gao et al. 2025) adapts it to emotional TTS by constructing preferences that sharpen distinctions among target emotions. Group Relative Policy Optimization (GRPO) (Shao et al. 2024) instead estimates relative advantages from groups of sampled outputs without a critic. F5R-TTS (Sun et al. 2025) applies GRPO to flow-matching TTS with intelligibility and speaker-similarity rewards, while another GRPO-based approach (Liu et al. 2025) optimizes an LLM-based TTS model with ASR rewards. CosyVoice 3 (Du et al. 2025) introduces diferentiable reward optimization for speech-token RL, and FlexiVoice (Chen et al. 2026) progressively combines multimodal DPO with multi-objective and instruction-focused GRPO. These methods, however, largely formulate rewards over global features, providing limited structure-aware supervision for multi-stage emotion trajectory or blended target pairs.

## Methodology

## Task Definition

Given text $x ,$ a natural-language emotion instruction $c _ { e } ,$ and a timbre reference utterance $y _ { \mathrm { r e f } } ,$ a conditional TTS model $F _ { \theta }$ generates

$$
{ \hat { y } } = F _ { \theta } ( x , c _ { e } , y _ { \mathrm { r e f } } ) .\tag{1}
$$

The output $\hat { y }$ should preserve the linguistic content of x and the timbre of $y _ { \mathrm { r e f } }$ while realizing the emotional pattern specified by $c _ { e }$

We consider two task types. An emotion trajectory condition contains an ordered sequence

$$
\mathcal { T } = \{ ( x _ { k } , e _ { k } ) \} _ { k = 1 } ^ { K } , \qquad x = x _ { 1 } \oplus \cdot \cdot \cdot \oplus x _ { K } ,\tag{2}
$$

where $x _ { k }$ is a text span and $e _ { k }$ is its target emotion. The generated speech should express these emotions in the prescribed order. An emotion blending condition instead contains an unordered pair $B = \{ A , B \}$ of distinct non-neutral emotions that should coexist throughout the utterance; it has no ordered emotion-tagged spans. These structures require order-aware trajectory control and target-pair-aware blending control, respectively.

## Two-Stage Multi-Emotion Post-Training

HybridEmo builds on CosyVoice $3 ^ { 2 }$ (Du et al. 2025), which combines an autoregressive speech-token LLM with a downstream flow-matching acoustic generator. Across both SFT and RL, we optimize only the LLM and keep the speech tokenizer and acoustic generator frozen. The LLM is conditioned only on the input text and emotion instruction; the timbre reference is used only by the acoustic generator for waveform decoding.

Supervised multi-emotion initialization. We first apply supervised fine-tuning (SFT) to trajectory and blending demonstrations so that the model acquires an initial ability to generate both multi-emotion patterns. For a target waveform $y ^ { * }$ , the frozen tokenizer produces target speech tokens $z ^ { * }$ We optimize the token-level cross-entropy objective

$$
\mathcal { L } _ { \mathrm { S F T } } = - \sum _ { t = 1 } ^ { | z ^ { * } | } \log p _ { \theta } \left( z _ { t } ^ { * } \mid z _ { < t } ^ { * } , x , c _ { e } \right) .\tag{3}
$$

This stage learns a conditional speech-token distribution from real multi-emotion speech before reward-based alignment.

Reinforcement learning alignment. Starting from the SFT checkpoint, we treat the autoregressive speech-token LLM as a policy $\pi _ { \boldsymbol { \theta } } . \mathbf { A }$ trajectory RL condition provides $( x , c _ { e } , \tau )$ , whereas a blending condition provides $( x , c _ { e } , B )$ neither contains a target waveform. The policy samples a group of speech-token rollouts, which the frozen acoustic generator decodes into waveforms conditioned on $y _ { \mathrm { r e f } } \colon$

$$
\begin{array} { r l } & { z ^ { ( i ) } \sim \pi _ { \boldsymbol { \theta } } ( \cdot \mid x , c _ { e } ) , } \\ & { \hat { y } ^ { ( i ) } = \mathcal { D } ( z ^ { ( i ) } ; y _ { \mathrm { r e f } } ) , \qquad i = 1 , \dots , G , } \end{array}\tag{4}
$$

where the fixed acoustic generator D renders each rollout as speech. The sample-aware reward below scores each waveform, and GRPO (Shao et al. 2024) computes group-relative advantages and updates the policy under a KL constraint. Figure 2 illustrates this alignment stage.

## Sample-Aware Hybrid Reward

Multi-emotion samples require diferent evidence according to their task structure. Let s ∈ {trajectory, blending} denote the sample type. HybridEmo shares an ASR reward $R _ { \mathrm { a s r } }$ across both types and routes one task-specific emotion reward:

$$
\begin{array} { r l } & { R ( \hat { y } , s ) = w _ { \mathrm { a s r } } R _ { \mathrm { a s r } } } \\ & { \phantom { = } + \left\{ \begin{array} { l l } { w _ { \mathrm { c o n s i } } R _ { \mathrm { c o n s i } } , } & { s = \mathrm { t r a j e c t o r y } , } \\ { w _ { \mathrm { g m m } } R _ { \mathrm { g m m } } , } & { s = \mathrm { b l e n d i n g } . } \end{array} \right. } \end{array}\tag{5}
$$

Here $R _ { \mathrm { c o n s i } }$ evaluates ordered trajectory completion, whereas $R _ { \mathrm { g m m } }$ evaluates compatibility with a blended target pair. The inapplicable emotion component is masked, and invalid or degenerate token sequences receive zero reward.

Shared ASR Reward Following the ASR-based content reward in CosyVoice 2 (Du et al. 2024), we explicitly protect intelligibility during emotion optimization. We transcribe $\hat { y }$ with SenseVoice<sup>3</sup> (An et al. 2024) and normalize the reference and recognized text with the same English text normalizer Norm(·). Let $x _ { \mathrm { a s r } } = \operatorname { A S R } ( \hat { y } )$ , and let $\epsilon _ { \mathrm { w e r } }$ denote the WER between Norm(x) and $\operatorname { N o r m } ( x _ { \mathrm { a s r } } )$ . We compute

$$
R _ { \mathrm { a s r } } = \mathrm { c l i p } _ { [ 0 , 1 ] } ( 1 - \mathrm { t a n h } ( 3 \epsilon _ { \mathrm { w e r } } ) ) .\tag{6}
$$

This bounded transformation gives high reward to contentfaithful speech during emotion optimization.

Trajectory-Aligned Emotion Consistency A trajectory condition provides the ordered emotion-tagged spans $\mathcal { T } =$ $\{ ( \boldsymbol { x } _ { k } , \boldsymbol { e } _ { k } ) \} _ { k = 1 } ^ { K }$ but no target waveform or timestamps. Because a global emotion score cannot localize an incorrect stage or identify missing and misordered stages, we approximate span boundaries in the generated waveform from text length. Let $\ell _ { k } = \operatorname* { m a x } ( | x _ { k } | , 1 )$ and $D$ be the duration of ${ \hat { y } } .$ Stage k occupies

$$
I _ { k } = \left[ D \frac { \sum _ { j = 1 } ^ { k - 1 } \ell _ { j } } { \sum _ { j = 1 } ^ { K } \ell _ { j } } , D \frac { \sum _ { j = 1 } ^ { k } \ell _ { j } } { \sum _ { j = 1 } ^ { K } \ell _ { j } } \right) .\tag{7}
$$

For each segment, a speech emotion recognition model, emotion2vec+ large<sup>4</sup> (Ma et al. 2024) yields the target posterior $p _ { k } ~ = ~ P ( e _ { k } ~ | ~ \hat { y } _ { k } )$ . Let $\begin{array} { r } { \bar { p } \ = \ K ^ { - 1 } \sum _ { k = 1 } ^ { K } p _ { k } } \end{array}$ and p<sub>min</sub> = min<sub>k</sub> $p _ { k }$ denote the mean and weakest-stage scores. The trajectory consistency reward is

$$
R _ { \mathrm { c o n s i } } = \beta \bar { p } + ( 1 - \beta ) p _ { \mathrm { m i n } } .\tag{8}
$$

Here $\beta \in [ 0 , 1 ]$ balances the mean and weakest-stage scores, preventing a high average from hiding a failed stage. For a single span, $R _ { \mathrm { c o n s i } }$ reduces to utterance-level target-emotion consistency; for multiple spans, it protects the weakest stage when evaluating trajectory completion. This timestamp-free alignment assumes that text length roughly tracks speaking duration.

![](images/163742bca582ccb99967066c9cc96a2fd8d999c74766160eafabc556e791bf79.jpg)  
Figure 2: HybridEmo GRPO alignment. A shared ASR reward preserves linguistic content, while the task router selects either trajectory consistency or GMM-based mixture-density feedback before GRPO updates the TTS policy.

GMM-Based Mixture-Density Reward A blending condition provides the unordered target pair $B = \{ A , B \}$ , without emotion-tagged spans or a target waveform. Because an utterance-level single-emotion posterior cannot represent this concurrent target, we construct ofline frame-level GMM anchors and use their mixture-density support as graded target-pair compatibility feedback, regularized with a weaker-target margin.

Ofline anchor construction. We use the base CosyVoice 3 model to synthesize single-emotion speech from instructions sampled from the SFT source corpus. We retain samples whose target-emotion confidence from emotion2vec+ large is at least τ, trim boundary frames, and extract frame-level emotion features. From these labeled features, we learn an LDA projector W, remove outliers with Isolation Forest (Liu, Ting, and Zhou 2008), and fit an emotion-specific GMM in the projected space:

$$
p _ { e } ( u ) = \sum _ { m = 1 } ^ { M } \omega _ { e , m } \mathcal { N } ( u ; \mu _ { e , m } , \Sigma _ { e , m } ) .\tag{9}
$$

Here $\omega _ { e , m } \geq 0$ and $\begin{array} { r } { \sum _ { m = 1 } ^ { M } \omega _ { e , m } = 1 } \end{array}$ . We freeze W and the per-emotion density anchors during RL.

Online mixture-density scoring. For a blending candidate, we obtain frame features $( h _ { 1 } , \ldots , h _ { T } )$ ) and project them with the same frozen mapper, $u _ { t } = W h _ { t } .$ . Given target emotions A and $B , p _ { A } ( u _ { t } )$ and $p _ { B } ( u _ { t } )$ are their per-frame densities. We define an unnormalized target-pair union score in log space using $\mathrm { L S E } ( a , b ) = \log ( \exp a + \exp b )$

$$
\ell _ { \mathrm { m i x } } ( u _ { t } ) = \mathrm { L S E } ( \log p _ { A } ( u _ { t } ) , \log p _ { B } ( u _ { t } ) ) .\tag{10}
$$

This score is high for frames that lie in regions supported by target anchors. To discourage utterance-level dominance by one target, we compute the pair-normalized contribution of each emotion and its weaker utterance-level share:

$$
\begin{array} { c } { \displaystyle q _ { t , e } = \frac { p _ { e } ( { \boldsymbol u } _ { t } ) } { p _ { A } ( { \boldsymbol u } _ { t } ) + p _ { B } ( { \boldsymbol u } _ { t } ) } , \quad e \in \{ A , B \} , } \\ { \displaystyle m = \operatorname* { m i n } _ { e \in \{ A , B \} } \frac { 1 } { T } \sum _ { t = 1 } ^ { T } q _ { t , e } . } \end{array}\tag{11}
$$

We then regularize the utterance-level union score with a normalized weaker-target margin:

$$
\begin{array} { l } { \displaystyle { s _ { \mathrm { u n i o n } } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \ell _ { \mathrm { m i x } } ( u _ { t } ) , } } \\ { \displaystyle { s _ { \mathrm { r a w } } = s _ { \mathrm { u n i o n } } - \alpha \operatorname* { m a x } \left( 0 , 1 - \frac { m } { \rho } \right) . } } \end{array}\tag{12}
$$

Here $\rho \in ( 0 , 0 . 5 ]$ sets the minimum desired weaker targetanchor contribution, and $\alpha \geq 0$ controls the maximum penalty. The weaker-target margin becomes inactive once $m \geq \rho ,$ , avoiding a strict equal-density constraint. Because the raw score is scale-sensitive, we standardize it using a running mean and scale:

$$
\tilde { s } = \frac { s _ { \mathrm { r a w } } - \mu _ { \mathrm { r u n } } } { \operatorname* { m a x } ( \sigma _ { \mathrm { r u n } } , \varepsilon ) } .\tag{13}
$$

Here $\varepsilon$ is a small constant for numerical stability. We then map the standardized score to a bounded reward:

$$
R _ { \mathrm { g m m } } = \mathrm { s i g m o i d } ( \tilde { s } ) .\tag{14}
$$

<table><tr><td>Dataset</td><td>Statistic</td><td>Traj. 1</td><td>Blend.</td><td>Total</td></tr><tr><td>SFT corpus</td><td>Utterances</td><td>119,981</td><td>4,115</td><td>124,096</td></tr><tr><td></td><td>Duration (h)</td><td>372.4</td><td>11.2</td><td>383.6</td></tr><tr><td>MultiEmo-RL</td><td>Samples</td><td>12,000</td><td>2,400</td><td>14,400</td></tr><tr><td>MultiEmo-Test Samples</td><td></td><td>600</td><td>120</td><td>720</td></tr></table>

Table 1: Statistics of the SFT corpus, MultiEmo-RL, and MultiEmo-Test. Audio duration applies only to SFT; the latter two contain text–instruction samples.

After warm-up, $\mu _ { \mathrm { r u n } }$ and $\sigma _ { \mathrm { r u n } }$ are updated with exponential moving averages of the raw score and its absolute deviation, respectively. The resulting reward combines graded union support with a weaker-target margin while accommodating multimodal variation within each emotion. Building on the blending distribution learned during SFT, GRPO uses this dense feedback to refine outputs toward regions supported by the requested emotion pair.

## Experiments

## Datasets

Table 1 summarizes the data used for the two training stages and evaluation. Our SFT corpus is constructed from the English portion of the in-the-wild Emilia dataset<sup>5</sup> (He et al. 2024). We use Qwen3-Omni-30B-A3B-Captioner<sup>6</sup> (Xu et al. 2025) to describe acoustic and paralinguistic characteristics, and then use MiniMax-M2.5<sup>7</sup> (MiniMax 2026) to screen emotional speech, assign the 7-class emotion taxonomy (angry, disgusted, fearful, happy, sad, surprised, and neutral), identify trajectory or blending structures, and generate natural-language instructions. Trajectory examples contain 1 to 3 labeled text spans, whereas each blending example contains exactly 2 distinct non-neutral emotions.

We construct MultiEmo-RL for reinforcement learning. It contains 12,000 emotion trajectory and 2,400 emotion blending text–instruction pairs generated from diverse metadata with MiniMax-M2.5. Each condition consists of a target text, a natural-language instruction, and its emotion structure, without a target waveform. Trajectory examples additionally include an emotion-tagged transcript only for constructing the alignment reward. The blending portion covers 12 unordered emotion pairs.

We further construct MultiEmo-Test with 720 conditions. Its trajectory portion contains 200 examples for each of the 1- , 2-, and 3-stage trajectory settings, while its blending portion contains 120 examples over the same 12 emotion pairs. Each condition is paired with a reference voice from the English Seed-TTS evaluation set<sup>8</sup> (Anastassiou et al. 2024); the same reference is provided to every system that supports timbre conditioning. We programmatically verify that MultiEmo-RL and MultiEmo-Test contain no duplicate samples.

Figure 3: Pairwise preferences from human listeners on MultiEmo-Test from HybridEmo’s perspective.  
![](images/908efcc29327ed46beae0cf6963191ec1c76f36628a20deb5bf7945959eae0ff.jpg)

## Implementation Details

We initialize HybridEmo from the 0.5B CosyVoice 3 checkpoint (Du et al. 2025) and perform SFT for 5 epochs using Adam (Kingma and Ba 2015), with a learning rate of $2 \times 1 0 ^ { - 6 }$ and an efective global batch size of 64.

We then train the model for 1 GRPO epoch on MultiEmo-RL using verl<sup>9</sup> (Sheng et al. 2025). For each input, the policy samples n = 8 speech-token rollouts at temperature 0.6. For waveform decoding and reward evaluation, the frozen acoustic generator uses a randomly sampled LibriSpeech ut terance<sup>10</sup> (Panayotov et al. 2015) as the voice prompt. GRPO uses a learning rate of 10<sup>−6</sup>, a batch size of 64, and KL regularization against the SFT reference policy. Both stages run on 4 NVIDIA H800 GPUs. We set $\beta = 0 . 7 , w _ { \mathrm { a s r } } = 0 . 5 .$ $w _ { \mathrm { c o n s i } } = 0 . 5 , w _ { \mathrm { g m m } } = 0 . 3 , \alpha = 0 . 0 1 , \mathrm { a n d } \rho = 0 . 0 5 .$

For ofline anchor construction, we use $\tau \ = \ 0 . 8 ,$ trim δ = 5% from each utterance boundary, project frame features to d = 6 dimensions with LDA, and remove $\rho _ { \mathrm { o u t } } = 8 \%$ of the frames using Isolation Forest. Each emotion is modeled by a full-covariance GMM with M = 3 components.

## Baselines

We compare HybridEmo with the 0.5B CosyVoice 3 (Du et al. 2025) model and representative external systems: EmoVoice (Yang et al. 2025) at 0.5B and 1.5B scales, and Qwen3-TTS-12Hz-1.7B-VoiceDesign (Hu et al. 2026). All systems synthesize the same target texts. CosyVoice 3, HybridEmo, and EmoVoice receive the same natural-language emotion instruction and reference speech.

Because its tested interface does notjointly accept a speech prompt and a text instruction, Qwen3-TTS uses a textual emotion instruction with a designated voice. Reference speech can introduce acoustic style priors that afect emotion control (Chen et al. 2026); thus, this setting is not strictly matched to the speech-prompted systems, and we treat Qwen3-TTS as an external capability reference.

<table><tr><td></td><td colspan="4">Correctness ↑</td><td colspan="3">Naturalness ↑</td><td></td><td></td><td></td></tr><tr><td>Model</td><td>1E</td><td>2E</td><td>3E</td><td>1-3E Avg.</td><td>2E</td><td>3E</td><td>2-3E Avg.</td><td>WER (%) ↓ UTMOS ↑ SIM ↑</td><td></td><td></td></tr><tr><td>EmoVoice-0.5B</td><td>3.70</td><td>2.94</td><td>3.16</td><td>3.27</td><td>2.26</td><td>2.57</td><td>2.41</td><td>4.29</td><td>3.22</td><td>0.56</td></tr><tr><td>EmoVoice-1.5B</td><td>3.74</td><td>2.95</td><td>3.13</td><td>3.28</td><td>2.29</td><td>2.49</td><td>2.39</td><td>7.94</td><td>3.12</td><td>0.56</td></tr><tr><td>Qwen3-TTS*</td><td>4.02</td><td>2.99</td><td>3.30</td><td>3.44</td><td>2.41</td><td>2.73</td><td>2.57</td><td>2.69</td><td>3.28</td><td>一</td></tr><tr><td>CosyVoice 3</td><td>3.68</td><td>2.95</td><td>3.09</td><td>3.24</td><td>2.30</td><td>2.49</td><td>2.40</td><td>1.91</td><td>3.20</td><td>0.69</td></tr><tr><td>HybridEmo</td><td>3.78</td><td>3.01</td><td>3.21</td><td>3.33</td><td>2.40</td><td>2.60</td><td>2.50</td><td>1.87</td><td>3.21</td><td>0.66</td></tr></table>

Table 2: Automatic evaluation results for the trajectory family on MultiEmo-Test. 1–3E Avg. is the macro correctness over the 1-, 2-, and 3-stage settings; 2–3E Avg. is the macro naturalness over the 2- and 3-stage settings because naturalness is defined only for multi-stage trajectories. SIM stands for speaker similarity. Qwen3-TTS<sup>∗</sup> uses a designated voice without a speech prompt.

<table><tr><td colspan="4">Model Int. ↑ Nat. ↑ WER (%) ↓ UTMOS ↑ SIM ↑</td></tr><tr><td>EmoVoice-0.5B 3.57</td><td>3.13</td><td>1.89</td><td>3.13 0.56</td></tr><tr><td>EmoVoice-1.5B</td><td>33.55 3.13</td><td>4.15 3.06</td><td>0.56</td></tr><tr><td>Qwen3-TTS*</td><td>3.71 3.29</td><td>0.64 3.39</td><td>一</td></tr><tr><td>CosyVoice 3</td><td>3.47 3.06</td><td>0.30</td><td>3.18 0.70</td></tr><tr><td>HybridEmo</td><td>3.71 3.24</td><td>0.37</td><td>3.17 0.64</td></tr></table>

Table 3: Automatic blending results on MultiEmo-Test. Int. and Nat. stand for intensity and naturalness. Qwen3-TTS<sup>∗</sup> uses a designated voice without a speech prompt.

## Evaluation Protocol

We score emotion control with task-specific 1–5 rubrics using Qwen3-Omni-30B-A3B-Instruct<sup>11</sup> (Xu et al. 2025). For trajectories, Emotion Correctness evaluates target expression for 1-stage samples and ordered completion for 2- and 3-stage samples, while Emotion Naturalness evaluates within-stage expression and transition coherence for the latter two settings. We report per-setting scores with macro averages over 1–3 stages for correctness and 2–3 stages for naturalness.

For blending, Emotion Intensity measures the overall perceptual strength of the requested two-emotion blend, while Emotion Naturalness measures whether the two emotional qualities coexist naturally rather than appearing sequentially or collapsing to one emotion. We evaluate them jointly as task-aligned perceptual criteria.

We report corpus-level WER (%) computed with whisper-large-v3<sup>12</sup> (Radford et al. 2023), applying the same English normalization to hypotheses and references. We assess acoustic quality with UTMOSv2<sup>13</sup> (Baba et al. 2024), which predicts a mean opinion score, and speaker similarity as the cosine similarity between ERes2Net<sup>14</sup> (Chen et al. 2023) embeddings of the reference and generated speech. Speaker similarity is unavailable for Qwen3-TTS because it uses no reference speech.

## Results and Analysis

## Automatic Evaluation Results

HybridEmo significantly raises trajectory macro correctness from 3.24 to 3.33 over CosyVoice 3, with consistent gains at every length: 3.68 to 3.78 for 1E, 2.95 to 3.01 for 2E, and 3.09 to 3.21 for 3E. It also raises macro naturalness from 2.40 to 2.50, keeps WER below 2% at 1.87%, and increases UTMOS from 3.20 to 3.21, without a noticeable degradation in speaker similarity.

For blending, HybridEmo significantly raises Intensity from 3.47 to 3.71 and Naturalness from 3.06 to 3.24 over CosyVoice 3. It exceeds both EmoVoice variants on these perceptual metrics, matches Qwen3-TTS in intensity, and trails it by only 0.05 in naturalness. HybridEmo also keeps WER below 2% at 0.37% and UTMOS within 0.01 of CosyVoice 3, without a noticeable degradation in speaker similarity.

## Human Preference Evaluation

Four trained listeners with synthetic-speech evaluation experience conducted pairwise comparisons for the three model pairs on MultiEmo-Test. Each listener assigned a win, tie, or loss from HybridEmo’s perspective to each of 20 randomly sampled conditions per pair (70% trajectory, 30% blending), yielding 80 judgments per pair. Listeners considered emotion correctness, expressiveness, naturalness, and intelligibility.

Figure 3 shows that HybridEmo receives more wins than losses against CosyVoice 3 and EmoVoice-0.5B, with win rates exceeding loss rates by 32 percentage points in both comparisons. Against Qwen3-TTS, HybridEmo obtains 35% wins, 29% ties, and 36% losses, yielding an essentially balanced preference, although the comparison is afected by difering timbre conditions. The trend is consistent with the automatic evaluation: HybridEmo surpasses CosyVoice 3 and EmoVoice and achieves a comparable preference split against Qwen3-TTS.

## Ablation Study

We next separate the efects of SFT initialization and structure-specific reinforcement learning. Table 4 uses Direct Hybrid-GRPO for hybrid-reward GRPO initialized directly from CosyVoice 3, and Trajectory-GRPO and Blending-GRPO for SFT-initialized variants optimized only on the corresponding type. HybridEmo uses the same SFT initialization followed by joint sample-aware hybrid GRPO. To isolate the efect of the weaker-target margin, we also evaluate a HybridEmo variant without this term in the reward.

<table><tr><td colspan="3">(a) Emotion trajectory</td></tr><tr><td colspan="3">Variant</td></tr><tr><td>CosyVoice 3</td><td>Corr. ↑ Nat. ↑ 3.24 2.40</td><td>WER (%) ↓ 1.91</td></tr><tr><td>SFT</td><td></td><td>2.06</td></tr><tr><td>Direct Hybrid-GRPO</td><td>3.28 2.47 3.30 2.47</td><td>2.08</td></tr><tr><td>Trajectory-GRPO</td><td>3.35 2.53</td><td>1.85</td></tr><tr><td>Blending-GRPO</td><td>3.26 2.44</td><td>1.82</td></tr><tr><td>HybridEmo</td><td>3.33 2.50</td><td>1.87</td></tr><tr><td>(b) Emotion blending</td><td></td><td></td></tr><tr><td colspan="3"></td></tr><tr><td>Variant</td><td>Inten. ↑ Nat. ↑</td><td>WER (%) ↓</td></tr><tr><td>CosyVoice 3</td><td>3.47 3.06</td><td>0.30</td></tr><tr><td>SFT</td><td>3.50 3.03</td><td>0.50</td></tr><tr><td>Direct Hybrid-GRPO</td><td>3.50 3.03</td><td>0.71</td></tr><tr><td>Trajectory-GRPO</td><td>3.53 3.05</td><td>0.30</td></tr><tr><td>Blending-GRPO</td><td>3.63 3.17</td><td>0.37</td></tr><tr><td>HybridEmo</td><td>3.71 3.24</td><td>0.37</td></tr><tr><td>w/o margin</td><td>3.66 3.05</td><td>0.32</td></tr></table>

Table 4: Ablation results on MultiEmo-Test. Trajectory Corr. averages 1E–3E and trajectory Nat. averages 2E–3E. Inten./Nat. denote blending intensity/naturalness; w/o margin removes the weaker-target margin.

SFT alone gives limited gains, while Direct Hybrid-GRPO improves some emotion scores but raises WER and remains below HybridEmo, supporting the role of supervised initialization before RL. Specialized variants are task-dependent: Trajectory-GRPO slightly exceeds joint HybridEmo on trajectory correctness and naturalness, whereas Blending-GRPO improves blending-oriented perceptual scores over SFT. Joint HybridEmo achieves the highest blending intensity and naturalness while remaining competitive on trajectory control.

Removing the weaker-target margin reduces Intensity from 3.71 to 3.66 and Naturalness from 3.24 to 3.05, lowering their mean from 3.48 to 3.36. These results suggest that limiting single-anchor dominance improves both aspects of blending quality.

## Representation and Reward-Space Analysis

Speech-token reconstruction. We examine how much emotion-related information is retained after discretetoken reconstruction. We encode and reconstruct the full RAVDESS speech corpus<sup>15</sup> (Livingstone and Russo 2018), which consists of studio-recorded emotional utterances, with the CosyVoice 3 speech tokenizer and decoder. For each utterance, we extract emotion2vec+ large (Ma et al. 2024) embeddings from the original and reconstructed waveforms and compute their paired cosine similarity. The mean similarity of 0.9441 shows that reconstruction retains highly similar emotion representations in the emotion2vec+ large space.

![](images/8968733cde20a66dced9829186220abcd7301f9f4ad8390b926c7c3d3c7c3988.jpg)  
Figure 4: 2D LDA projection of three-component GMM emotion anchors. Crosses mark component means, and ellipses show 95% contours; scoring uses the full 6D space.

Geometry of the GMM emotion anchors. Figure 4 visualizes the ofline anchors fitted to filtered single-emotion utterances synthesized by CosyVoice 3. The projected distributions form distinct but partially overlapping regions, while multiple full-covariance components capture intra-emotion modes and anisotropic variation. These anchors provide continuous densities for mixture-density scoring. This geometry suggests that the anchors remain discriminative while allowing blended realizations to receive support from both target distributions.

## Conclusion

In this paper, we formulate emotion control through emotion trajectories and blending. We propose HybridEmo, which combines supervised multi-emotion initialization with sample-aware hybrid-reward GRPO, routing task-matched rewards for trajectory and blending samples alongside shared ASR feedback. On our constructed MultiEmo-Test, HybridEmo significantly improves emotion correctness over CosyVoice 3 at every trajectory length and both blendingoriented perceptual scores, while matching Qwen3-TTS in blending intensity. It keeps WER below 2% and UTMOS within 0.01 of CosyVoice 3 in both settings, without a noticeable degradation in speaker similarity. Human preferences further favor HybridEmo over CosyVoice 3 and EmoVoice-0.5B. The ablation study further demonstrates the complementary value of the two reward routes. The margin ablation also shows that the weaker-target margin, designed to limit single-anchor dominance, improves blending intensity and naturalness. Overall, these results show that matching reinforcement signals to task structure improves instructionaligned perceptual outcomes across trajectory and blending settings, ofering a practical direction for multi-emotion TTS.

## References

An, K.; Chen, Q.; Deng, C.; Du, Z.; Gao, C.; Gao, Z.; Gu, Y.; He, T.; Hu, H.; Hu, K.; Ji, S.; Li, Y.; Li, Z.; Lu, H.; Luo, H.; Lv, X.; Ma, B.; Ma, Z.; Ni, C.; Song, C.; Shi, J.; Shi, X.; Wang, H.; Wang, W.; Wang, Y.; Xiao, Z.; Yan, Z.; Yang, Y.; Zhang, B.; Zhang, Q.; Zhang, S.; Zhao, N.; and Zheng, S. 2024. FunAudioLLM: Voice Understanding and Generation Foundation Models for Natural Interaction Between Humans and LLMs. arXiv:2407.04051.

Anastassiou, P.; et al. 2024. Seed-TTS: A Family of High-Quality Versatile Speech Generation Models. arXiv:2406.02430.

Baba, K.; Nakata, W.; Saito, Y.; and Saruwatari, H. 2024. The T05 System for the VoiceMOS Challenge 2024: Transfer Learning from Deep Image Classifier to Naturalness MOS Prediction of High-Quality Synthetic Speech. In 2024 IEEE Spoken Language Technology Workshop (SLT), 818–824.

Chen, D.; Zhang, X.; Wang, Y.; Dai, K.; Ma, L.; and Wu, Z. 2026. FlexiVoice: Enabling Flexible Style Control in Zero-Shot TTS with Natural Language Instructions. arXiv:2601.04656.

Chen, Y.; Zheng, S.; Wang, H.; Cheng, L.; Chen, Q.; and Qi, J. 2023. An Enhanced Res2Net with Local and Global Feature Fusion for Speaker Verification. In Interspeech 2023, 2228–2232.

Du, Z.; Gao, C.; Wang, Y.; Yu, F.; Zhao, T.; Wang, H.; Lv, X.; Wang, H.; Ni, C.; Shi, X.; An, K.; Yang, G.; Li, Y.; Chen, Y.; Gao, Z.; Chen, Q.; Gu, Y.; Chen, M.; Chen, Y.; Zhang, S.; Wang, W.; and Ye, J. 2025. CosyVoice 3: Towards In-the-Wild Speech Generation via Scaling-Up and Post-Training. arXiv:2505.17589.

Du, Z.; Wang, Y.; Chen, Q.; Shi, X.; Lv, X.; Zhao, T.; Gao, Z.; Yang, Y.; Gao, C.; Wang, H.; Yu, F.; Liu, H.; Sheng, Z.; Gu, Y.; Deng, C.; Wang, W.; Zhang, S.; Yan, Z.; and Zhou, J. 2024. CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models. arXiv:2412.10117.

Gao, X.; Zhang, C.; Chen, Y.; Zhang, H.; and Chen, N. F. 2025. Emo-DPO: Controllable Emotional Speech Synthesis through Direct Preference Optimization. In 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 1–5.

Guo, Z.; Leng, Y.; Wu, Y.; Zhao, S.; and Tan, X. 2023. PromptTTS: Controllable Text-to-Speech with Text Descriptions. In 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 1–5.

He, H.; Shang, Z.; Wang, C.; Li, X.; Gu, Y.; Hua, H.; Liu, L.; Yang, C.; Li, J.; Shi, P.; Wang, Y.; Chen, K.; Zhang, P.; and Wu, Z. 2024. Emilia: An Extensive, Multilingual, and Diverse Speech Dataset for Large-Scale Speech Generation. In 2024 IEEE Spoken Language Technology Workshop (SLT), 885–890.

Hu, H.; Zhu, X.; He, T.; Guo, D.; Zhang, B.; Wang, X.; Guo, Z.; Jiang, Z.; Hao, H.; Guo, Z.; Zhang, X.; Zhang, P.; Yang, B.; Xu, J.; Zhou, J.; and Lin, J. 2026. Qwen3-TTS Technical Report. arXiv:2601.15621.

Ji, S.; Chen, Q.; Wang, W.; Zuo, J.; Fang, M.; Jiang, Z.; Huang, H.; Wang, Z.; Cheng, X.; Zheng, S.; and Zhao, Z. 2025. ControlSpeech: Towards Simultaneous and Independent Zero-Shot Speaker Cloning and Zero-Shot Language Style Control. In Proceedings of the 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), 6966–6981. Association for Computational Linguistics.

Kharitonov, E.; Vincent, D.; Borsos, Z.; Marinier, R.; Girgin, S.; Pietquin, O.; Sharifi, M.; Tagliasacchi, M.; and Zeghidour, N. 2023. Speak, Read and Prompt: High-Fidelity Textto-Speech with Minimal Supervision. Transactions of the Associationfor Computational Linguistics, 11: 1703–1718.

Kingma, D. P.; and Ba, J. 2015. Adam: A Method for Stochastic Optimization. In 3rd International Conference on Learning Representations.

Leng, Y.; Guo, Z.; Shen, K.; Ju, Z.; Tan, X.; Liu, Y.; Liu, Y.; Yang, D.; Zhang, L.; Song, K.; He, L.; Li, X.-Y.; Zhao, S.; Qin, T.; and Bian, J. 2024. PromptTTS 2: Describing and Generating Voices with Text Prompt. In The Twelfth International Conference on Learning Representations.

Liu, C.; Hu, Y.-J.; Gao, Y.-Y.; Zhang, S.-L.; and Ling, Z.- H. 2025. Group Relative Policy Optimization for Text-to-Speech with Large Language Models. arXiv:2509.18798.

Liu, F. T.; Ting, K. M.; and Zhou, Z.-H. 2008. Isolation Forest. In 2008 Eighth IEEE International Conference on Data Mining, 413–422. IEEE.

Livingstone, S. R.; and Russo, F. A. 2018. The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A Dynamic, Multimodal Set of Facial and Vocal Expressions in North American English. PLOS ONE, 13(5): e0196391.

Ma, Z.; Zheng, Z.; Ye, J.; Li, J.; Gao, Z.; Zhang, S.; and Chen, X. 2024. emotion2vec: Self-Supervised Pre-Training for Speech Emotion Representation. In Findings of the Associationfor Computational Linguistics: ACL 2024, 15747– 15760. Association for Computational Linguistics.

MiniMax. 2026. MiniMax-M2.5. Hugging Face model card, https://huggingface.co/MiniMaxAI/MiniMax-M2.5. Accessed July 21, 2026.

Panayotov, V.; Chen, G.; Povey, D.; and Khudanpur, S. 2015. LibriSpeech: An ASR Corpus Based on Public Domain Audio Books. In 2015 IEEEInternational Conference onAcoustics, Speech and Signal Processing, 5206–5210.

Radford, A.; Kim, J. W.; Xu, T.; Brockman, G.; McLeavey, C.; and Sutskever, I. 2023. Robust Speech Recognition via Large-Scale Weak Supervision. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, 28492–28518. PMLR.

Rafailov, R.; Sharma, A.; Mitchell, E.; Manning, C. D.; Ermon, S.; and Finn, C. 2023. Direct Preference Optimization: Your Language Model Is Secretly a Reward Model. In Advances in Neural Information Processing Systems, volume 36, 53728–53741.

Shao, Z.; Wang, P.; Zhu, Q.; Xu, R.; Song, J.; Bi, X.; Zhang, H.; Zhang, M.; Li, Y. K.; Wu, Y.; and Guo, D. 2024.

DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv:2402.03300.

Sheng, G.; Zhang, C.; Ye, Z.; Wu, X.; Zhang, W.; Zhang, R.; Peng, Y.; Lin, H.; and Wu, C. 2025. HybridFlow: A Flexible and Eficient RLHF Framework. In Proceedings of the Twentieth European Conference on Computer Systems, 1279–1297.

Sun, X.; Xiao, R.; Mo, J.; Wu, B.; Yu, Q.; and Wang, B. 2025. F5R-TTS: Improving Flow-Matching Based Text-to-Speech with Group Relative Policy Optimization. arXiv:2504.02407.

Wang, C.; Chen, S.; Wu, Y.; Zhang, Z.; Zhou, L.; Liu, S.; Chen, Z.; Liu, Y.; Wang, H.; Li, J.; He, L.; Zhao, S.; and Wei, F. 2023. Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers. arXiv:2301.02111.

Wang, X.; Jiang, M.; Ma, Z.; Zhang, Z.; Liu, S.; Li, L.; Liang, Z.; Zheng, Q.; Wang, R.; Feng, X.; Bian, W.; Ye, Z.;

Cheng, S.; Yuan, R.; Zhao, Z.; Zhu, X.; Pan, J.; Xue, L.; Zhu, P.; Chen, Y.; Li, Z.; Chen, X.; Xie, L.; Guo, Y.; and Xue, W. 2025. Spark-TTS: An Eficient LLM-Based Textto-Speech Model with Single-Stream Decoupled Speech Tokens. arXiv:2503.01710.

Xu, J.; et al. 2025. Qwen3-Omni Technical Report. arXiv:2509.17765.

Yang, D.; Liu, S.; Huang, R.; Weng, C.; and Meng, H. 2024. InstructTTS: Modelling Expressive TTS in Discrete Latent Space with Natural Language Style Prompt. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 32: 2913–2925.

Yang, G.; Yang, C.; Chen, Q.; Ma, Z.; Chen, W.; Wang, W.; Wang, T.; Yang, Y.; Niu, Z.; Liu, W.; Yu, F.; Du, Z.; Gao, Z.; Zhang, S.; and Chen, X. 2025. EmoVoice: LLM-Based Emotional Text-to-Speech Model with Freestyle Text Prompting. In Proceedings of the 33rd ACM International Conference on Multimedia. Association for Computing Machinery.