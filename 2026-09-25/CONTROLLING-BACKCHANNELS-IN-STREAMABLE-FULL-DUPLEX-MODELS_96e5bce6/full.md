# CONTROLLING BACKCHANNELS IN STREAMABLE FULL-DUPLEX MODELS

Maike Zufle ¨ <sup>1</sup>, Peter Polak´ <sup>2,3</sup>,

Sefik Emre Eskimez <sup>4</sup>, Jan Niehues <sup>1</sup>, Peter Bell <sup>5</sup>, Ondrej Klejch ˇ <sup>5</sup>

<sup>1</sup>Karlsruhe Institute of Technology, Germany <sup>2</sup>AppTek, Germany <sup>3</sup> Charles University, Czech Republic <sup>4</sup>Sesame AI, USA <sup>5</sup>University of Edinburgh, UK maike.zuefle@kit.edu

## ABSTRACT

Backchannels, brief acknowledgements like “uh-huh” produced while the other party may still be talking, are central to natural conversation, but full-duplex spoken dialogue models rarely model them explicitly. We introduce a lightweight backchannel head that predicts, from a full-duplex model’s own hidden states, when a backchannel should begin. Once this probability crosses a tunable threshold, a backchannel is force-decoded. Attached to both a 7B (PersonaPlex) and a 1B (F-Actor) model, it generalizes across scale. Probing confirms the hidden states anticipate real human timing, and generation evaluation shows more frequent, better-timed backchannels. Human raters judge the resulting backchannels on par with real ones.

Index Terms— full-duplex, backchannel, turn-taking

## 1. INTRODUCTION

Full-duplex spoken dialogue models [1, 2, 3, 4] can listen and speak simultaneously, making them well suited for generating backchannels, i.e., short acknowledgements, like “uh-huh”, a listener produces while the other party may still be talking.

Backchannel modelling received attention before the advent of full-duplex modelling, with earlier systems using explicit prediction heads to generate backchannels and improve rapport [5, 6]. Voice Activity Projection (VAP) models predict backchanneling frame-by-frame as an external classifier in a cascaded pipeline [7, 8]. Endpointing targets the more general problem of turn boundaries [9]. Complementary work studies the lexico-prosodic meaning of backchannels [10].

In full-duplex dialogue modelling, recent models produce backchannels as a byproduct of training on conversational data [3, 4]. Reinforcement learning can further shape this behaviour into more natural backchanneling [11]. Though effective, this is expensive and yields an implicit policy with no controllable signal for when or why backchannels occur. Closer to explicit control, one line of work first pretrains a full-duplex model and then adds VAP heads for turn-taking [12], while another probes a full-duplex model’s representations for turn-taking without pretraining [13].

Other existing evaluations of full-duplex backchanneling focus on whether models backchannel at all [2, 4], and not whether those backchannels are appropriate. Therefore, it is critical to evaluate backchannel placement against real human conversational timing, and to train a model that explicitly predicts when a backchannel should occur.

We address this gap with a lightweight, model-independent backchannel head. It predicts, at each timestep, the probability that a backchannel should begin, and when this probability crosses a threshold, we force-decode a backchannel token into the text stream, which in turn drives the corresponding audio output. This is in the spirit of frame-wise VAP-based backchannel predictors [7], but attached directly to a full-duplex model’s own hidden states rather than run as an external classifier. We attach this head to two full-duplex models at different scales, the 7B PersonaPlex [3] and the 1B F-Actor [4], to show that the method generalizes. Since F-Actor does not support streaming inference, we additionally adapt it into a streaming model to enable this comparison.

We evaluate the resulting models along three complementary axes. First, we measure whether the model backchannels more often than the unmodified baseline. Second, we probe the models’ hidden states to test whether they predict backchannel onset where real speakers actually backchanneled in human conversations, using human-annotated timing as ground truth. Third, we run a human evaluation in which participants rate how appropriate the backchannels in generated dialogues sound, finding that they are rated on par with human backchannels.

Our contributions are as follows:

• A model-independent method for controlling backchannel timing, which we show generalizes across two architectures and two model scales.

• A human-aligned evaluation of backchannel behaviour, combining a probing analysis against real human timing data with a human appropriateness study.

• A streaming version of F-Actor to validate our method on models of multiple sizes.

We release the code and models openly.<sup>1</sup>

## 2. CONTROLLABLE BACKCHANNELING

This section presents our proposed model-independent backchannel head. Our design fulfils two objectives: (1) At each frame, the model predicts whether the current point in the interlocutor’s speech is an appropriate moment for a backchannel, i.e., a moment at which a human would produce one. (2) The system must provide an independent control mechanism to adjust how frequently backchannels are emitted without degrading the appropriateness of their placement.

## 2.1. Backchannel Head

To predict backchannel timing without altering the backbone’s language modelling capabilities, we attach a lightweight MLP head directly to its hidden states. Let $\mathbf { h } _ { t } \in \mathbb { R } ^ { d }$ denote the hidden state of the backbone (the temporal transformer) at timestep t. The head maps $\mathbf { h } _ { t }$ to a probability via a two-layer MLP with a sigmoid output $\sigma ( \cdot )$

$$
p _ { t } = \sigma ( \mathbf { W } _ { 2 } \operatorname { G E L U } ( \mathbf { W } _ { 1 } \mathbf { h } _ { t } + \mathbf { b } _ { 1 } ) + b _ { 2 } ) ,\tag{1}
$$

where $p _ { t } \in [ 0 , 1 ]$ is the predicted probability of a backchannel onset, i.e., the first frame of a backchannel, at $t + 1$ , and $b _ { 2 }$ is initialized to the empirical log-odds of the class prior.

To handle extreme class imbalance (onsets account for ∼1% of conversational frames), the head is trained using Focal Loss [14] evaluated only over eligible frames $\mathcal { T } _ { \mathrm { v a l i d } }$ where the partner is speaking and the agent is silent. All frames during agent speech or mutual silence are excluded from $\mathcal { T } _ { \mathrm { v a l i d } }$ . Let $y _ { t } \in \{ 0 , 1 \}$ denote the frame-level target backchannel onset label at timestep t + 1 and let $\tilde { p } _ { t } = y _ { t } p _ { t } + ( 1 - y _ { t } ) ( 1 - p _ { t } )$ denote the probability assigned to the target class using the predicted probabilities $p _ { t }$ . The Focal Loss $\mathcal { L } _ { \mathrm { h e a d } }$ is then computed as:

$$
\mathcal { L } _ { \mathrm { h e a d } } = - \frac { 1 } { | \mathcal { T } _ { \mathrm { v a l i d } } | } \sum _ { t \in \mathcal { T } _ { \mathrm { v a l i d } } } \alpha _ { t } ( 1 - \tilde { p } _ { t } ) ^ { \gamma } \log ( \tilde { p } _ { t } ) ,\tag{2}
$$

where $\alpha _ { t } = \alpha y _ { t } + ( 1 - \alpha ) ( 1 - y _ { t } )$ balances class frequencies with hyperparameters $\alpha = 0 . 9$ and $\gamma = 2 . 0$

## 2.2. Thresholding and Conditioned Text Generation

During inference, we decouple placement detection from token emission via a threshold $\tau \in [ 0 , 1 ] ;$ : a backchannel is triggered whenever $p _ { t } \geq \tau$ , giving control over backchanneling frequency independently of placement quality.

Upon triggering at timestep t, we force-decode the special [BC] token into the model’s text stream, replacing the new-word marker [EPAD]. The model then continues autoregressive generation to produce the backchannel’s lexical form (e.g., “yeah”, “uh-huh”, “right”), conditioned on the injected token and surrounding context.

Critically, we mask the loss on the [BC] token itself, so the backbone never learns to emit it spontaneously, leaving initiation entirely to the thresholded head, while retaining standard cross-entropy loss on the subsequent transcript tokens that realize the backchannel’s content.

## 3. EXPERIMENTAL SETUP

## 3.1. Models

We evaluate our approach on two full-duplex models. We train PersonaPlex (7B) [3] with the backchannel mechanism from Section 2 on a single H100 80GB GPU for approximately 10 hours. The backbone is fine-tuned during training using LoRA [15], while the depth transformer is kept frozen.

In addition, we adapt F-Actor (1B) [4]. Backchanneling requires listening and speaking in real time, since backchannels overlap with the interlocutor’s speech. F-Actor, however, does not support streaming inference, as it relies on NanoCodec [16]. We therefore replace NanoCodec with the causal Mimi codec [2] and, following [17], add a depth transformer pretrained from CSM 1B (frozen) to model Mimi’s codebooks. Speaker conditioning uses ECAPA-TDNN [18] embeddings. The temporal transformer backbone is Llama 3.2 1B Instruct [19], with the BC head from Section 2.1 attached. Unlike PersonaPlex, we train it only to predict the system channel’s inner monologue and Mimi codes. Training takes 13 hours on four A100 80GB GPUs.

Models with the backchannel head carry the suffix -BC.

## 3.2. Data

Training. We use Fisher [20], which contains around 2000h of English conversations, restoring the 8kHz audio into 24kHz with Sidon [21]. Each conversation is transcribed with Parakeet TDT 0.6B V3 [22], and split into 90s chunks at the nearest utterance end. To condition the model to reliably produce backchannels upon receiving the [BC] token, we augment these dialogues by inserting additional backchannels from other occurrences at locations where the agent is silent for at least 1.5 seconds, and the interlocutor keeps speaking for at least 1s, constraints chosen after pilot experiments.

Evaluation. We use TurnBench [23], human-human conversations with backchannel annotations from three annotators per speaker (majority vote). We restrict to the twelve conversations where all three segmented the same turns: 137 minutes, 574 backchannels out of 1,888 total utterances. Each conversation is split into 60s windows (plus 10s prior context), snapped to the nearest silence gap and capped at 120s, and evaluated with leave-one-conversation-out crossvalidation. Since TurnBench gives only utterance-level timestamps, we obtain word-level ones via Parakeet TDT 0.6B V3 [22] on each turn’s own audio and transcript.

![](images/7446e29b1032e345298ec5f429824d49b286d4e3e0e4a740262cfd94dbc83575.jpg)  
Fig. 1. Correct backchannel timing F1 score per layer for predicting backchannel onset from frozen hidden states on TurnBench, for base and backchannel-head-trained (BC) models. A prediction only counts as correct if it lands at the exact annotated frame.

![](images/4e7601f359e13eff18234530e06f9ee46f7a8db819ef145bbc21ad50b0107ab1.jpg)  
Fig. 2. Predicted backchannel frequency per layer, for base and backchannel-head-trained (BC) models on TurnBench.

## 4. EVALUATION

## 4.1. Probing

We test whether a model anticipates when a backchannel is appropriate by force-decoding annotated human conversations and training a probe to predict from the hidden state at frame t whether a backchannel onset occurs at t + 1. We probe both the base models and their counterparts fine-tuned with the backchannel head (-BC): since the backbone is finetuned jointly with the head (Section 3), training can change the hidden states, and the comparison tests whether it makes upcoming onsets easier to predict. A separate probe, rather than the head itself, lets us measure the base models, which have no head, in the same way.

Method. For each conversation, we force-decode the ground-truth audio of both speakers and the word-aligned text stream, silencing the real backchannels so that predictions can also be tested during and after them, and extract the hidden states of every temporal-transformer layer. Probing all layers ensures a fair comparison with the base models, which may encode backchannel timing best at an intermediate layer, and shows whether training makes the signal available at the last layer, which the head reads at inference. For each layer, we fit an MLP probe with class-balanced weighting on TurnBench and evaluate it on held-out conversations.

Evaluation. Using the probe, we count how often $p ( y _ { t } =$ $1 \ | \ h _ { t } )$ crosses a fixed threshold and compare this rate to the human rate. We separately score whether the predicted timing itself is correct against the human annotations, reporting F1.

## 4.2. Evaluating the Generated Backchannels

We test whether the trained head produces correctly timed backchannels once it actually drives generation, and whether doing so comes at the cost of the model’s general full-duplex conversational abilities. We evaluate our BC-head-augmented models on Full-Duplex-Bench v1 [24] alongside their base models, isolating the effect of the backchannel head. The benchmark reports backchannel-specific metrics alongside metrics for pause handling, turn-taking, and interruption. We use three thresholds: normal matches the human TurnBench rate; low/max yield fewer/more backchannels.

## 4.3. Human Evaluation

In addition to the automatic evaluation, we assess whether human listeners judge the models’ backchannels as appropriately placed. We randomly select 35 TurnBench samples of 10–20 seconds, each containing one backchannel, and forcedecode them with the PersonaPlex-BC model, letting it predict the backchannel. As a baseline, we also silence the real backchannels and reinsert them at random points. Each sample thus has three versions, human, random, and model, each rated 1–5 for appropriate backchanneling by four annotators (12 in total) via the Pearmut platform [25].

## 5. RESULTS

We evaluate the backchannel-head mechanism in three stages: probing its hidden states, generation, and human evaluation.

<table><tr><td rowspan=1 colspan=10>Pause (Synth.)  Pause (Candor)         Backchannel         Smooth Turn Taking   User InterruptionModel                TOR↓          TOR↓      TOR↓ Freq ↑ JSD↓  TOR↑  Latency ↓  TOR↑ Latency ↓</td></tr><tr><td rowspan=1 colspan=1>Moshi</td><td rowspan=1 colspan=1>0.445</td><td rowspan=1 colspan=1>0.528</td><td rowspan=1 colspan=1>0.255</td><td rowspan=1 colspan=2>0.074   0.824</td><td rowspan=1 colspan=1>0.739</td><td rowspan=1 colspan=1>0.162</td><td rowspan=1 colspan=1>0.920</td><td rowspan=1 colspan=1>1.377</td></tr><tr><td rowspan=1 colspan=1>+ RL [11]</td><td rowspan=1 colspan=1>0.226</td><td rowspan=1 colspan=1>0.417</td><td rowspan=1 colspan=1>0.091</td><td rowspan=1 colspan=1>0.095</td><td rowspan=1 colspan=1>0.789</td><td rowspan=1 colspan=1>0.966</td><td rowspan=1 colspan=1>0.121</td><td rowspan=1 colspan=1>1.000</td><td rowspan=1 colspan=1>0.461</td></tr><tr><td rowspan=3 colspan=1>PersonaPlex</td><td rowspan=3 colspan=1>0.482</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=2 colspan=1>0.444</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=1>0.182</td><td rowspan=1 colspan=1>0.046</td><td rowspan=1 colspan=1>0.841</td><td rowspan=1 colspan=1>0.958</td><td rowspan=1 colspan=1>0.219</td><td rowspan=1 colspan=1>0.940</td><td rowspan=1 colspan=1>0.271</td></tr><tr><td rowspan=1 colspan=1>+ RL [11]</td><td rowspan=1 colspan=1>0.328</td><td rowspan=1 colspan=1>0.361</td><td rowspan=1 colspan=1>0.127</td><td rowspan=1 colspan=1>0.122</td><td rowspan=1 colspan=1>0.783</td><td rowspan=1 colspan=1>0.950</td><td rowspan=1 colspan=1>0.079</td><td rowspan=1 colspan=1>1.000</td><td rowspan=1 colspan=1>0.187</td></tr><tr><td rowspan=1 colspan=1>+ BC low (ours)</td><td rowspan=1 colspan=1>0.569</td><td rowspan=1 colspan=1>0.620</td><td rowspan=1 colspan=1>0.291</td><td rowspan=1 colspan=1>0.081</td><td rowspan=1 colspan=1>0.780</td><td rowspan=1 colspan=1>0.958</td><td rowspan=1 colspan=1>0.098</td><td rowspan=1 colspan=1>0.935</td><td rowspan=1 colspan=1>0.294</td></tr><tr><td rowspan=1 colspan=1>+ BC normal (ours)</td><td rowspan=1 colspan=1>0.577</td><td rowspan=1 colspan=1>0.630</td><td rowspan=1 colspan=1>0.236</td><td rowspan=1 colspan=1>0.088</td><td rowspan=1 colspan=1>0.778</td><td rowspan=1 colspan=1>0.958</td><td rowspan=1 colspan=1>0.098</td><td rowspan=1 colspan=1>0.935</td><td rowspan=1 colspan=1>0.295</td></tr><tr><td rowspan=1 colspan=1>+ BC max (ours)</td><td rowspan=1 colspan=1>0.766</td><td rowspan=1 colspan=1>0.773</td><td rowspan=1 colspan=1>0.218</td><td rowspan=1 colspan=1>0.259</td><td rowspan=1 colspan=1>0.672</td><td rowspan=1 colspan=1>0.983</td><td rowspan=1 colspan=1>0.015</td><td rowspan=1 colspan=1>0.945</td><td rowspan=1 colspan=1>0.181</td></tr><tr><td rowspan=1 colspan=1>F-Actor</td><td rowspan=1 colspan=1>0.409</td><td rowspan=1 colspan=1>0.296</td><td rowspan=1 colspan=1>0.673</td><td rowspan=1 colspan=1>0.102</td><td rowspan=1 colspan=1>0.743</td><td rowspan=1 colspan=1>0.513</td><td rowspan=1 colspan=1>0.675</td><td rowspan=1 colspan=1>0.785</td><td rowspan=1 colspan=1>2.008</td></tr><tr><td rowspan=1 colspan=1>+ BC low (ours)</td><td rowspan=1 colspan=1>0.533</td><td rowspan=1 colspan=1>0.463</td><td rowspan=1 colspan=1>0.600</td><td rowspan=1 colspan=1>0.118</td><td rowspan=1 colspan=1>0.721</td><td rowspan=1 colspan=1>0.731</td><td rowspan=1 colspan=1>0.124</td><td rowspan=1 colspan=1>0.925</td><td rowspan=1 colspan=1>1.443</td></tr><tr><td rowspan=1 colspan=1>+ BC normal (ours)</td><td rowspan=1 colspan=1>0.343</td><td rowspan=1 colspan=1>0.296</td><td rowspan=1 colspan=1>0.618</td><td rowspan=1 colspan=1>0.122</td><td rowspan=1 colspan=1>0.724</td><td rowspan=1 colspan=1>0.504</td><td rowspan=1 colspan=1>0.905</td><td rowspan=1 colspan=1>0.845</td><td rowspan=1 colspan=1>1.613</td></tr><tr><td rowspan=1 colspan=1>+ BC max (ours)</td><td rowspan=1 colspan=1>0.650</td><td rowspan=1 colspan=1>0.569</td><td rowspan=1 colspan=1>0.600</td><td rowspan=1 colspan=2>0.151   0.709</td><td rowspan=1 colspan=1>0.765</td><td rowspan=1 colspan=1>0.187</td><td rowspan=1 colspan=1>0.920</td><td rowspan=1 colspan=1>1.507</td></tr></table>

Table 1. Comparison of Moshi, PersonaPlex, F-Actor, and our BC variants on Full-Duplex-Bench v1.

## 5.1. Probing

Backchannel Frequency. Figure 2 shows the probe’s predicted backchannel frequency per layer against the human rate. PersonaPlex-BC reaches this human rate when probing the last layer at a well-calibrated threshold around 0.5. F-Actor-BC likewise improves over its own base model at the last layer, but to a smaller degree. This suggests that the backchannel-head mechanism successfully shifts the model’s hidden states toward human-like backchannel frequency.

Onset Timing Accuracy. F1 measures whether the probe’s predicted onset lands at the correct moment rather than merely how often it fires. Figure 1 shows PersonaPlex-BC with substantially higher F1 than base PersonaPlex across nearly all layers, while F-Actor-BC’s large gain over base F-Actor at middle layers narrows to a small improvement by the last layer. F-Actor(-BC) also performs best at low decision thresholds. A tolerance of two frames (±0.16s) around the annotated onset improves both models’ scores.

## 5.2. Evaluating the Generated Backchannels

We now evaluate the backchannels the model generates.

Full-Duplex-Bench. Table 1 compares the base models against our BC-augmented versions and against RL-based backchannel shaping [11]. We find that BC backchannels more often and with better timing (lower JSD) than the base models. A lower threshold produces more backchannels, as expected (max vs. low). Compared to RL, PersonaPlex-$\mathrm { B C } _ { \mathrm { m a x } }$ achieves higher backchannel frequency, better timing (JSD) and faster turn-taking, but higher take-over rates (TOR), especially on pauses. F-Actor’s pause TOR at the normal setting, in contrast, improves over its base model and is among the best in the table.

Surface Forms. Beyond deciding when to backchannel, the model also has to choose what to say. We test this in two ways: via free generation and via generation constrained to the training data’s own token paths. Both produce similar lexical distributions, for example roughly doubling the human share of yeah. Despite this lexical similarity, listening to the audio, we find the constrained version sounds more natural.

![](images/3a329dfd849530d7431a2078e973d27606b609ff8dd408dd8200a3ee8e593f08.jpg)  
Fig. 3. Results of the human evaluation.

## 5.3. Human Evaluation

We ask human evaluators to judge backchannel appropriateness (Figure 3): Human recordings score 4.34/5, PersonaPlex-BC 4.37/5, and the random baseline 3.77/5, with both human and PersonaPlex-BC rated significantly higher than random $( p < 0 . 0 0 0 1 )$ but not significantly different from each other $( p \ = \ 0 . 4 7 )$ . Inter-annotator agreement (Krippendorff’s α, interval metric) is 0.42, moderate and consistent with the inherent subjectivity of naturalness judgments.

## 6. DISCUSSION AND CONCLUSION

We show that backchannel timing can be learned as a lightweight, pluggable signal, rather than left to emerge implicitly from conversational training. Across two architectures, this explicit control improves backchannel frequency and timing, and in a human evaluation its backchannels are judged on par with human ones. We release our models and code to make this control mechanism easy to build upon.

## 7. ACKNOWLEDGMENTS

This work was supported by JSALT 2026 at Johns Hopkins University with funds from NSF CCRI Grant No. 2120435, Google DeepMind, JHU HLTCOE, JHU AI2AI and ACL, and has received funding from the European Union’s Horizon research and innovation programme under grant agreement No 101135798, project Meetween (My Personal AI Mediator for Virtual MEETtings BetWEEN People).

Generative AI tools were used to assist with editing and grammar checking of the manuscript, as well as for coding and plotting. All scientific content, analyses, and conclusions were developed and verified by the authors.

## 8. REFERENCES

[1] K. Hu, E. Hosseini-Asl, C. Chen et al., “Efficient and Direct Duplex Modeling for Speech-to-Speech Language Model,” in Proc. Interspeech, 2025.

[2] A. Defossez, L. Mazar´ e, M. Orsini, A. Royer, P. P´ erez,´ H. Jegou, E. Grave, and N. Zeghidour, “Moshi: a´ speech-text foundation model for real-time dialogue,” arXiv:2410.00037, 2024.

[3] R. Roy, J. Raiman, S. Lee, T.-D. Ene, R. Kirby, S. Kim, J. Kim, and B. Catanzaro, “PersonaPlex: Voice and role control for full duplex conversational speech models,” in Proc. ICASSP, 2026.

[4] M. Zufle, O. Klejch, N. Sanders, J. Niehues, A. Birch,¨ and T. K. Lam, “F-Actor: Controllable conversational behavior in full-duplex models,” in ACL Findings, 2026.

[5] D. Lala, P. Milhorat, K. Inoue, M. Ishida, K. Takanashi, and T. Kawahara, “Attentive listening system with backchanneling, response generation and flexible turntaking,” in Proc. SIGDial, 2017.

[6] R. Ruede, M. Muller, S. St ¨ uker, and A. Waibel, “Yeah,¨ right, uh-huh: a deep learning backchannel predictor,” in Proc. IWSDS, 2018.

[7] K. Inoue, D. Lala, G. Skantze, and T. Kawahara, “Yeah, un, oh: Continuous and real-time backchannel prediction with fine-tuning of voice activity projection,” in Proc. NAACL-HLT, 2025.

[8] K. Inoue, M. Elmers, Y. Fu, Z. H. Pang, T. Mori, D. Lala, K. Ochi, and T. Kawahara, “Multilingual and continuous backchannel prediction: A cross-lingual study,” in Proc. IWSDS, 2026.

[9] S. Udupa, S. Watanabe, P. Schwarz, and J. Cernocky, “Endpoint anticipation for low-latency spoken dialogue,” in Proc. Interspeech, 2026.

[10] L. Qian and G. Skantze, “Aligning backchannel and dialogue context representations via contrastive LLM finetuning,” in Proc. ACL, 2026.

[11] A. Ohashi, N. Zeghidour, A. Defossez, and´

E. Kharitonov, “Multi-faceted interactivity alignment in full-duplex speech models,” arXiv:2606.11167, 2026.

[12] S. Rajaa, “DualTurn: Learning turn-taking from dual-channel generative speech pretraining,” arXiv:2603.08216, 2026.

[13] P. Riera, P. Brusco, C. Kuo, M. Sancinetti, and S. Branavan, “Synchronization and turn-taking in full-duplex speech dialogue models,” arXiv:2605.20356, 2026.

[14] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar, “Focal loss for dense object detection,” in Proceedings of the IEEE International Conference on Computer Vision (ICCV), Oct 2017.

[15] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “LoRA: Low-rank adaptation of large language models,” arXiv:2106.09685, 2021.

[16] E. Casanova, P. Neekhara, R. Langman, S. Hussain, S. Ghosh, X. Yang, A. Jukic, J. Li, and B. Ginsburg, “NanoCodec: Towards High-Quality Ultra Fast Speech LLM Inference,” in Proc. Interspeech, 2025.

[17] N. Torgashov, G. E. Henter, and G. Skantze, “VoXtream2: Full-stream TTS with dynamic speaking rate control,” arXiv:2603.13518, 2026.

[18] B. Desplanques, J. Thienpondt, and K. Demuynck, “ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification,” in Proc. Interspeech, 2020.

[19] A. Grattafiori, A. Dubey, A. Jauhri et al., “The Llama 3 herd of models,” arXiv:2407.21783, 2024.

[20] C. Cieri, D. Miller, and K. Walker, “The Fisher corpus: A resource for the next generations of speech-to-text,” in LREC, vol. 4, 2004, pp. 69–71.

[21] W. Nakata, Y. Saito, Y. Ueda, and H. Saruwatari, “Sidon: Fast and robust open-source multilingual speech restoration for large-scale dataset cleansing,” in Proc. ICASSP, 2026.

[22] M. Sekoyan, N. R. Koluguri, N. Tadevosyan, P. Zelasko, T. Bartley, N. Karpov, J. Balam, and B. Ginsburg, “Canary-1B-v2 & Parakeet-TDT-0.6B-v3: Efficient and high-performance models for multilingual ASR and AST,” arXiv:2509.14128, 2025.

[23] F. Jiang, R. Sanabria, S. Deshmukh et al., “TurnBench: A multi-domain benchmark for turn-taking dynamics in spoken dialogue,” arXiv:2608.25218, 2026.

[24] G.-T. Lin, J. Lian, T. Li, Q. Wang, G. Anumanchipalli, A. H. Liu, and H.-y. Lee, “Full-Duplex-Bench: A benchmark to evaluate full-duplex spoken dialogue models on turn-taking capabilities,” in Proc. ASRU Workshop, 2025.

[25] V. Zouhar and T. Kocmi, “Pearmut: Human evaluation of translation made trivial,” arXiv:2601.02933, 2026.