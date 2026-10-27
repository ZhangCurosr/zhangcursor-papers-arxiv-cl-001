# WHEN THE PERSONA CONTRADICTS THE LINE: DIAGNOSING PLAN-TO-SYNTHESIS FAILURES IN CHAIN-OF-THOUGHT TTS

Joonyong Park, Jerry Li

Spellbrush, CA, USA

jyjoon97@gmail.com, jerry@sizigistudios.com

## ABSTRACT

Chain-of-thought TTS systems rely on language-model planners to translate persona and context into vocal delivery, but it is unclear where this intent is lost when the requested affect conflicts with lexical sentiment. We introduce a bilingual Japanese–English persona– line conflict benchmark in which each item pairs one line with congruent and conflicting personas and pre-declares the expected acoustic direction. We probe this direction in the planner-generated delivery phrase, the planner-written emotion vector, the QwenEmotiongenerated vector, and the rendered waveform. Planner outputs are usually direction-correct, while downstream agreement can be substantially lower. Bypassing QwenEmotion provides only modest gains over the full planner-controlled Base. Supervising QwenEmotion raises the direction rate to 75% in Japanese and 84% in English. A learned text-to-latent interface reaches the highest English rate of 89% and the highest Japanese arousal rate of 88%. These results show that learned repair of the plan-to-synthesis interface can recover counter-textual delivery, although its effect differs across languages and acoustic axes.

Index Terms— expressive TTS, chain-of-thought, persona TTS, emotion recognition

## 1. INTRODUCTION

Instruction-conditioned text-to-speech (TTS) adjusts its delivery to natural-language descriptions of speaker, situation or tone [1, 2]. To achieve this, recent systems add a planning step in which a language model scripts how a line should be spoken before any audio is generated [3]. When the requested affect agrees with the words, such systems follow instructions with high perceived quality [4, 1].

However, dramatic acting frequently requires the exact opposite: a character might offer congratulations while deeply grieving, or speak cheerful words while utterly exhausted. We refer to this as a persona–line conflict, where the required vocal delivery must contradict the underlying sentiment of the written text. In these cases, current instruction-following synthesizers typically fail, defaulting to the text’s inherent emotion far more often than they follow the prompt [2, 5]. Recent evaluations acknowledge this failure [2], but they rely on broad categorical labels or overall ratings. Because they only judge the final audio output, they cannot pinpoint where the system breaks down: did the planner choose the wrong delivery, did the emotion-control interface fail to preserve it, or did the synthesizer fail to realize the intended acoustics? Without probing these intermediate representations, researchers cannot determine which component should be modified, and a reported improvement may be difficult to localize.

To answer this question, we introduce a test set in Japanese and English that isolates persona–line conflicts. Each item pairs one spoken line with a congruent persona, under which the line reads naturally, and a conflicting persona that requires an opposing affect. The item also declares an acoustic target (a, s), specifying whether arousal or valence should increase or decrease between the two renditions. Because this target is determined before synthesis, the same declared direction can be used to evaluate both intermediate representations and the rendered speech. Declaring an acoustic axis also avoids requiring each persona to correspond to a single emotion category. We score the generated speech with a predictor read on the declared axis [6] and validate its pairwise direction judgments on parallel human recordings [7, 8].

We apply this protocol to a planner-controlled IndexTTS-2.5 pipeline. Given the persona and spoken line, the external planner produces a free-form delivery phrase and an eight-dimensional emotion vector. In the stock route, QwenEmotion [9] independently converts the delivery phrase into another eight-dimensional emotion vector, which is mapped through a speaker-matched emotion prototype bank to the conditioning representation used for synthesis. We probe agreement with the declared direction at four points: the delivery phrase, the planner-written emotion vector, the QwenEmotiongenerated emotion vector, and the waveform produced by the stock emotion route with planner-side duration control disabled.

We then evaluate three modifications of this emotion-control interface while keeping the synthesis backbone fixed. First, we bypass QwenEmotion and pass the planner-written emotion vector directly to the existing prototype bank without additional training. Second, we fine-tune QwenEmotion using render-selected supervision. Third, we replace both QwenEmotion and the prototype bank with a learned text-to-latent network that predicts a continuous emotion-conditioning shift directly from the planner-generated delivery phrase.

Our contributions are (i) a bilingual persona–line conflict benchmark with a pre-declared acoustic target and reference-free scoring protocol; (ii) a representation-level diagnosis that probes direction agreement in the delivery phrase, planner-written emotion vector, QwenEmotion-generated emotion vector, and rendered waveform; and (iii) a controlled comparison of bypassed, supervised, and learned emotion-control interfaces while keeping the synthesis backbone fixed.

Related work. Benchmarks of instruction-following TTS do not declare an acoustic target in advance. As a result, they cannot locate a breakdown between the plan, the control parameters and the waveform [3, 2, 5]. Speech emotion recognizers can rely on the sentiment of the spoken words [6, 10]. This lexical bias is problematic for evaluating speech whose intended affect conflicts with the text. Parallel recordings of the same sentence under different emotions have been used in studies of mismatched perceptual channels [11] and in constructing preference pairs [2]. Such recordings have not been used to validate a metric for counter-textual direction [7, 8]. Natural-language style prompts can be mapped to style embeddings by prompt encoders trained jointly with the synthesizer [12, 13]. Other work aligns text and speech style in a shared representation space [14]. Emotion can also be controlled through categorical vectors and reference or exemplar embeddings [4, 15]. Our learned interface instead predicts the pretrained synthesizer’s own emotion latent from text using targets extracted from human acted speech. The synthesizer itself remains fixed.

Table 1. Example test items. The target is fixed before any audio exists.
<table><tr><td>Line</td><td> $P _ { \mathrm { c o n g } }$ </td><td> $P _ { \mathrm { c o n f } }$ </td><td>(a, s)</td></tr><tr><td>“美味  $L \cup \overrightarrow { C } \overrightarrow { 9 } \overrightarrow { 4 }$  (It&#x27;s delicious.) “You missed the tram by about ten sec-</td><td>a child praising the meal a parent cooked a helpful stranger on a rainy platform, to a com-</td><td>a wedding guest eating food they cannot enjoy a weary parent in the dark kitchen, as the exhausted</td><td>val –</td></tr><tr><td> $\mathrm { o n d s } ? \mathrm { \Omega }$ </td><td>muter who arrived too late</td><td>teenager comes home long after curfew</td><td>ar -</td></tr></table>

## 2. METHOD

## 2.1. Conflict benchmark and scorer

Each item comprises a line ℓ, a congruent persona $P _ { \mathrm { c o n g } } ,$ a conflicting persona $P _ { \mathrm { c o n f } } .$ , and a declared target (a, s). The target specifies an axis $a \in$ {arousal, valence} and a direction $s \in \{ + , - \}$ $P _ { \mathrm { c o n g } }$ elicits a natural reading of the line, whereas $P _ { \mathrm { c o n f } }$ places the same words in a context requiring an opposing affect. A conflicting rendition is direction-correct when it moves along a in direction s relative to the congruent rendition.

An LLM authors each item in two steps, first proposing a scene and then introducing a conflicting twist. A planner-independent jury retains an item only when at least four of five seeded votes agree with its declared direction. A fresh five-vote audit of 75 items reproduced the declared target on 96% of items. The authors then reviewed the retained items and vetoed ambiguous cases. Each language contains 150 test items, with 56 arousal and 94 valence items. Japanese speech is rendered with Tsukuyomi-chan<sup>1</sup> and Amitaro<sup>2</sup>, and English speech with the EmoV-DB speakers Bea and Sam [8]. All training lines and training voices used in the experiments are disjoint from the test set.

We score rendered speech with a wav2vec2-based acoustic predictor, which maps an utterance to continuous arousal, dominance and valence in [0, 1] from the audio alone[6]. The regressor is trained on dimensional arousal, dominance and valence annotations from MSP-Podcast [16]. Let m<sub>a</sub>(·) denote the output of a regressor on the declared axis a. An item is direction-correct when the sign of $\Delta = m _ { a } ( x _ { \mathrm { c o n f } } ) - m _ { a } ( x _ { \mathrm { c o n g } } )$ equals the declared direction s.

Because the regressor is trained only on MSP-Podcast, we validate it on out-of-domain parallel human recordings. We use samespeaker, same-sentence recordings in different emotional states from Voice-Statistics<sup>3</sup>, the Amitaro ITA recordings, EmoV-DB [8], and ESD [7]. Across 24,334 arousal pairs, the regressor recovers the labelled direction on 99.6%; across 15,601 valence pairs, the corresponding rate is 83.1%.

The direction rate for each condition is the per-item majority over three plan seeds. We compare conditions with two-sided exact sign tests over discordant items. All reported renders satisfy a Whisper-derived CER [17] of at most 0.15 for Japanese and 0.25 for English. UTMOS [18] remains within 0.14 of the base in Japanese and above the base in English.

## 2.2. Pipeline and representation probe

Fig. 1 explains the whole pipeline. We use IndexTTS-2.5 [4] as a fixed synthesis backbone and place an external LLM planner Gemini 3.5 Flash-Lite in front of it. For each persona–line pair, the planner produces a free-form delivery phrase, an eightdimensional emotion vector, a scalar strength α, and a duration factor. The spoken line is passed separately to the synthesizer model as the transcript. In the stock emotion route, the delivery phrase is converted by QwenEmotion[9] into an eight-dimensional emotion vector, which is mapped through the speaker-matched prototype bank to the emotion-conditioning representation used for synthesis. The planner-predicted duration factor is supplied directly to the synthesizer and is shared across the main interface conditions.

We probe agreement with the declared direction at four points in the emotion-control path. The delivery phrase is scored by the planner-independent jury used for benchmark construction. The planner-written and QwenEmotion-generated emotion vectors are projected onto the declared axis using a fixed sign map over the eight emotion categories.<sup>4</sup> The waveform is scored by the acoustic contrast ∆ defined in Sec. 2.1. The duration factor is not included in this representation probe because it is a separate prosodic control supplied directly to the synthesizer. Accordingly, for the waveform probe, we fix the duration factor to 1.0 to isolate the stock emotion-control path.

## 2.3. Interface treatments

All treatments keep the IndexTTS-2.5 synthesis backbone fixed and differ only in how the planner output is converted into emotion conditioning. For each persona condition, the planner predicts a duration factor in [0.7, 1.4] from the situation and delivery phrase. Unless otherwise stated, all main conditions use this planner-predicted duration factor. In the w/o duration ablation, the same emotioncontrol route is used while the duration factor is fixed to 1.0. In the w/o conditioning ablation, we replace the planner-dependent emotion conditioning with the neutral conditioning used by the synthesis backbone while retaining the planner-predicted duration factor. The w/o duration & conditioning ablation removes both controls.

Bypassing QwenEmotion. The structured route bypasses QwenEmotion and passes the planner-written eight-dimensional emotion vector and scalar strength α directly to the same speakermatched prototype bank. The prototype bank, planner-side duration control, and synthesis backbone remain unchanged.

Supervising QwenEmotion. We fine-tune only QwenEmotion, whose input is the planner-generated delivery phrase and whose output is the eight-dimensional emotion vector used by the stock route. We initially construct 800 training items per language using Gemini 3.5 Flash-Lite, each specifying a line, situation, delivery phrase, and declared affective axis and direction. Training voices are disjoint from the test voices: three Voice-Statistics speakers and one JVNV speaker for Japanese, and two EmoV-DB

![](images/749552522795f1806e540133ae64158c49d2cb9c668469b8a4e530367104d7c3.jpg)  
Fig. 1. Stock IndexTTS-2.5 emotion-control route and the three interface interventions evaluated in this work.

speakers for English.

For each item, we render one neutral reference and five candidate (v, α) controls with the synthesis backbone fixed. Candidate vectors are constructed from emotion categories consistent with the declared direction. Each candidate is evaluated by its acoustic contrast against the neutral rendition on the declared axis, subject to CER ≤ 0.15 for Japanese and ≤ 0.25 for English. We additionally reject candidates that induce an excessive off-axis change inconsistent with the delivery phrase’s LLM-labelled valence, or whose dominant emotion category contradicts that valence.

Among the remaining candidates, we select the lowest-α candidate whose contrast exceeds the pilot threshold of 0.15 for arousal or 0.30 for valence. If no candidate exceeds the threshold, we select the candidate with the largest contrast in the declared direction; items for which no candidate moves in the required direction are discarded. This procedure yields 569 Japanese and 524 English training pairs, of which 293 and 342, respectively, exceed the corresponding threshold. The supervision target is the selected emotion vector scaled by α in QwenEmotion’s native output format, since the stock text route does not expose a separate strength output. We fine-tune QwenEmotion with LoRA [19] using rank 16, scaling 32, dropout 0.05, four epochs, a learning rate of $\bar { 1 0 } ^ { - 4 }$ , and batch size 8.

Learning a text-to-latent interface. We train a model that maps a natural-language delivery description directly to a continuous 1,280-dimensional emotion-conditioning representation. The training targets are constructed from 30,083 acted clips of 27 speakers disjoint from the test voices: 14,345 English clips (ESD [7], 9 speakers; EmoV-DB [8], 2 speakers), 14,000 Mandarin clips (ESD, 10 speakers), and 1,738 Japanese clips (JVNV [20], 3 speakers; Voice-Statistics, 3 speakers).

For each emotional recording, we pair it with the same speaker’s neutral recording of the same sentence and pass both through the IndexTTS-2.5 audio-prompt pathway. Let $\hat { A } ( x ) \ \in \ \mathbb { R } ^ { 1 2 8 0 }$ denote the extracted emotion-conditioning representation. For an emotional recording $x _ { i } ^ { \mathrm { { e m o } } }$ and its matched neutral recording $x _ { i } ^ { \mathrm { n e u } }$ , we define the target shift as

$$
\Delta \mathbf { z } _ { i } = A ( x _ { i } ^ { \mathrm { e m o } } ) - A ( x _ { i } ^ { \mathrm { n e u } } ) .\tag{1}
$$

Because the paired recordings share the speaker and linguistic content, this difference is used as an audio-derived target for the change associated with emotional delivery.

Table 2. Agreement with the declared direction at four probe points over 150 items per language. Each rate is the per-item majority over three plan seeds. Phrase–vector rows report item counts.
<table><tr><td>Probe point</td><td>JP</td><td>EN</td></tr><tr><td>Delivery phrase (LLM jury)</td><td>84%</td><td>98%</td></tr><tr><td>Planner-written emotion vector (sign map)</td><td>86%</td><td>89%</td></tr><tr><td>agreement with delivery phrase</td><td>83%</td><td>89%</td></tr><tr><td>phrase-only / vector-only correct (items)</td><td>11 / 14</td><td>15 / 1</td></tr><tr><td>QwenEmotion vector (sign map)</td><td>46%</td><td>83%</td></tr><tr><td>arousal / valence</td><td>61 / 37%</td><td>82 / 84%</td></tr><tr><td>Waveform (stock, w/o duration)</td><td>55%</td><td>66%</td></tr></table>

Each target shift is paired with a free-form delivery description $d _ { i }$ generated from the source corpus’s emotion annotation using Gemini 3.5 Flash-Lite. The description is generated without access to the recording or its extracted latent representation. A frozen multilingual E5-large encoder [21] maps $d _ { i }$ to a 1,024- dimensional text embedding, and a two-layer MLP $f _ { \theta }$ predicts the corresponding 1,280-dimensional emotion-conditioning shift:

$$
\begin{array} { r } { \hat { \Delta } \mathbf { { z } } _ { i } = f _ { \theta } \left( E \left( d _ { i } \right) \right) , \quad \quad E \left( d _ { i } \right) \in \mathbb { R } ^ { 1 0 2 4 } , \quad \hat { \Delta } \mathbf { { z } } _ { i } \in \mathbb { R } ^ { 1 2 8 0 } . } \end{array}\tag{2}
$$

The MLP is trained to match the audio-derived target shift using mean-squared error and cosine loss, while the E5 encoder and synthesis backbone remain fixed.

At inference, the planner-generated delivery phrase is encoded by E5 and mapped by the trained MLP directly to an emotionconditioning shift. The predicted shift is added to the neutral emotion-conditioning representation of the target speaker to obtain the conditioning representation used for synthesis. This bypasses both QwenEmotion’s category representation and the subsequent prototype-bank mapping.

Listening test. Ten listeners per language perform a blinded matching test on 30 items drawn at random from the test set, each rendered under three conditions of Table 3: the base, the supervised module and the learned interface, all with the planner’s duration. In every trial the listener hears the two renditions of one item and assigns each to its situation; the condition is hidden, and trial order and the positions of audio and text are randomised per listener. The 90 trials per language are split into two lists of 45, each taken by five listeners, so every item receives five judgments per condition. An item counts as correct for a condition when at least three of its five judgments are. Conditions are tested against chance by two-sided exact sign tests over items, and we report the item-level agreement between the listener majority and the predictor’s verdict on the served renditions.

Table 3. Direction rate on the 150-item test sets, three plan seeds, per-item majority. Every condition receives the planner’s duration decision unless marked w/o duration. Ar./Val. split the rate by declared axis. For the three treatments p is a two-sided exact sign test against the base of the same language; for an ablation it is against the row it ablates.
<table><tr><td></td><td colspan="4">Japanese</td><td colspan="4">English</td></tr><tr><td>Condition</td><td>Dir.</td><td>Ar.</td><td>Val.</td><td>p</td><td>Dir.</td><td>Ar.</td><td>Val.</td><td>p</td></tr><tr><td>Base: stock module</td><td>65%</td><td>70%</td><td>62%</td><td>1</td><td>74%</td><td>77%</td><td>72%</td><td>1</td></tr><tr><td>Bypassed module</td><td>68%</td><td>80%</td><td>61%</td><td>.58</td><td>79%</td><td>80%</td><td>79%</td><td>.26</td></tr><tr><td>Supervised module (SFT)</td><td>75%</td><td>77%</td><td>74%</td><td>.033</td><td>84%</td><td>91%</td><td>80%</td><td>.036</td></tr><tr><td>Learned interface</td><td>66%</td><td>88%</td><td>53%</td><td>.60</td><td>89% 95%</td><td></td><td></td><td>85%.008</td></tr><tr><td>Ablations</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Base w/o duration</td><td>55% 59% 52% .014</td><td></td><td></td><td></td><td>66%</td><td>66%</td><td></td><td>66%.008</td></tr><tr><td>Base w/o conditioning</td><td>61%</td><td>64%</td><td>60%</td><td>.61</td><td>74%</td><td>79%</td><td>71%</td><td>1.0</td></tr><tr><td>Base w/o dur. &amp; cond.</td><td>53% 54% 53% .040</td><td></td><td></td><td></td><td>69%</td><td>64%</td><td>72%</td><td>.39</td></tr></table>

## 3. EXPERIMENTS AND RESULTS

## 3.1. Where direction agreement breaks down

We probe the declared direction in the planner-generated delivery phrase, the planner-written emotion vector, the QwenEmotiongenerated emotion vector, and the waveform produced by the stock emotion route with planner-side duration control disabled. Table 2 reports the agreement at each probe point independently.

The delivery phrase agrees with the declared direction on 84% of Japanese items and 98% of English items. The planner-written emotion vector reaches 86% and 89%, respectively. The two planner outputs agree with each other on 83% of Japanese items and 89% of English items. Their disagreements differ across languages. In Japanese, the delivery phrase alone is correct on 11 items and the planner-written vector alone on 14, with no significant asymmetry $( p = . 6 9 )$ . In English, the delivery phrase alone is correct on 15 items, compared with one item for the vector alone $( p = . 0 0 0 5 2 )$ .

The QwenEmotion-generated vector shows substantially lower agreement in Japanese than either planner representation. Given the planner-generated delivery phrase, the QwenEmotion vector agrees with the declared direction on 46% of Japanese items and 83% of English items. The Japanese rates are 61% for arousal and 37% for valence, while the corresponding English rates are 82% and 84%. With the stock emotion route and planner-side duration control disabled, the final waveform is direction-correct on 55% of Japanese items and 66% of English items. Because the probe points use representationspecific readouts, these rates should not be interpreted as conditional survival probabilities from one stage to the next.

## 3.2. Can the interface be repaired?

Table 3 compares three modifications of the emotion-control interface under the same planner-side duration control.

Bypassing QwenEmotion. Passing the planner-written emotion vector and α directly to the prototype bank reaches 68% in Japanese and 79% in English. Neither change is significant relative to Base $( p = . 5 8$ and .26). Thus, bypassing the stock mapping alone provides only modest evidence of improvement.

Table 4. Listening test on 30 random items per language, five judgments per item and condition. Judg.: share of correct judgments; Items: items (of 30) whose listener majority is correct, with a twosided sign test against chance; Agr.: items (of 30) on which the listener majority agrees with the predictor’s verdict.
<table><tr><td></td><td colspan="4">Japanese</td><td colspan="4">English</td></tr><tr><td>Condition</td><td>Judg.</td><td>Items</td><td>p</td><td> $\operatorname { A g r } .$ </td><td>Judg.</td><td>Items</td><td>p</td><td> $\operatorname { A g r } .$ </td></tr><tr><td>Base</td><td>64%</td><td>19</td><td>.20</td><td>20</td><td>62%</td><td>19</td><td>.20</td><td>23</td></tr><tr><td>Supervised (SFT)</td><td>67%</td><td>20</td><td>.10</td><td>21</td><td>69%</td><td>22</td><td>.016</td><td>23</td></tr><tr><td>Learned interface</td><td>66%</td><td>20</td><td>.10</td><td>20</td><td>63%</td><td>20</td><td>.10</td><td>23</td></tr></table>

Supervising QwenEmotion. We next retain the stock interface but explicitly supervise its text-to-vector mapping. SFT raises the direction rate to 75% in Japanese and 84% in English, significantly above Base in both languages $\left( p = . 0 3 3 \right.$ and .036).

Learning a text-to-latent interface. Finally, we replace both QwenEmotion and the prototype bank with a learned text-to-latent mapping. The learned route reaches 89% overall in English $( p =$ .008 against Base), the highest observed English rate, with 95% arousal and 85% valence. In Japanese, its overall rate is 66% $( p =$ .60), but arousal reaches 88%, the highest observed Japanese arousal rate. The effect of the learned interface is therefore strongly axis- and language-dependent.

Ablations. Removing planner-side duration control from Base reduces the direction rate from 65% to 55% in Japanese $( p = . 0 1 4 )$ and from 74% to 66% in English $( p = . 0 0 8 )$ . Removing plannerside emotion conditioning alone has a smaller effect, yielding 61% in Japanese $( p = . 6 1 )$ and 74% in English $( p = 1 . 0 )$ . Removing both duration and conditioning further lowers the Japanese rate to 53% $( p = . 0 4 0 )$ ), while the English rate becomes 69% $( p = . 3 9 )$ These results indicate that planner-side duration control provides the clearest independent contribution to the full Base, while the effect of emotion conditioning is weaker in the aggregate metric and may depend on the acoustic axis.

## 3.3. Human validation

Table 4 gives the result. Judgment-level assignment accuracy ranges from 64–67% in Japanese and 62–69% in English. At the item-majority level, the supervised module in English exceeds chance (22/30, $p = . 0 1 6 )$ , while the other conditions yield 19–20 correct items out of 30 $( p = . 1 0 \ – . 2 0 )$ . Listener majorities agree with the predictor on $6 1 / 9 0 \ ( 6 8 \% )$ Japanese and 69/90 (77%) English item–condition pairs. The test therefore provides descriptive evidence that the intended contrasts are perceptible and that the predictor often tracks listener judgments, but does not establish differences among the treatments, whose comparison rests on Table 3.

## 4. CONCLUSION

We introduced a bilingual persona–line conflict benchmark, and used it to probe the delivery phrase, emotion vector, and rendered speech while comparing bypassed, supervised, and learned emotioncontrol interfaces. Our study is limited to one planner–synthesizer pipeline, two affective axes, and a reference-free acoustic scorer whose cross-lingual transfer remains imperfectly characterized. Future work should move beyond coarse emotion categories and intermediate delivery descriptions by developing richer emotion captioners and evaluators, and by training models that reason directly from persona and context to continuous acting representations.

## 5. ACKNOWLEDGMENTS

Ethics statement. The listening study involved non-invasive perceptual judgments of speech. Participants were recruited through personal contacts, and were informed that their responses would be used for research and reported only in aggregate.

Use of Generative AI Disclosure. Claude (Anthropic) was used to assist with language revision of the manuscript and experimental code development. All AI-generated outputs were reviewed and verified by the authors.

## 6. REFERENCES

[1] Zhihao Du et al., “CosyVoice 2: Scalable streaming speech synthesis with large language models,” arXiv preprint arXiv:2412.10117, 2024.

[2] Dekun Chen, Xueyao Zhang, Yuancheng Wang, Kenan Dai, Li Ma, and Zhizheng Wu, “FlexiVoice: Enabling flexible style control in zero-shot TTS with natural language instructions,” in Proc. ICLR, 2026.

[3] Yue Wang, Ruotian Ma, Xingyu Chen, Zhengliang Shi, Morunliu Yang, Wanshun Chen, Huang Liu, Jiadi Yao, Xin He, Qu Yang, Qingxuan Jiang, Fanghua Ye, Juntao Li, Zhaopeng Tu, Xiaolong Li, Liefeng Bo, and Min Zhang, “BatonVoice: An operationalist framework for enhancing controllable speech synthesis with linguistic intelligence from LLMs,” in Proc. ACL (Volume 1: Long Papers), 2026, pp. 46683–46697.

[4] Siyi Zhou, Yiquan Zhou, Yi He, Xun Zhou, Jinchao Wang, Wei Deng, and Jingchen Shu, “IndexTTS2: A breakthrough in emotionally expressive and duration-controlled auto-regressive zero-shot text-to-speech,” in Proc. AAAI, 2026, vol. 40, pp. 35139–35148.

[5] Huakang Chen, Jingbin Hu, Liumeng Xue, Qirui Zhan, Wenhao Li, et al., “MINT-Bench: A comprehensive multilingual benchmark for instruction-following text-to-speech,” arXiv preprint arXiv:2604.17958, 2026.

[6] Johannes Wagner, Andreas Triantafyllopoulos, Hagen Wierstorf, Maximilian Schmitt, Felix Burkhardt, Florian Eyben, and Bjorn W. Schuller, “Dawn of the transformer era in speech¨ emotion recognition: Closing the valence gap,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 9, pp. 10745–10759, 2023.

[7] Kun Zhou, Berrak Sisman, Rui Liu, and Haizhou Li, “Seen and unseen emotional style transfer for voice conversion with a new emotional speech dataset,” in Proc. ICASSP, 2021, pp. 920–924.

[8] Adaeze Adigwe, Noe Tits, Kevin El Haddad, Sarah Ostadab-´ bas, and Thierry Dutoit, “The emotional voices database: Towards controlling the emotion dimension in voice generation systems,” arXiv preprint arXiv:1806.09514, 2018.

[9] Dawei Huang, Qing Li, Chuan Yan, Zebang Cheng, Zihao Han, Yurong Huang, Xiang Li, Bin Li, Xiaohui Wang, Zheng Lian, Zhi-Qi Cheng, and Xiaojiang Peng, “Emotion-Qwen: A unified framework for emotion and vision understanding,” arXiv preprint arXiv:2505.06685, 2025.

[10] Andreas Triantafyllopoulos, Johannes Wagner, Hagen Wierstorf, Maximilian Schmitt, Uwe Reichel, Florian Eyben, Felix Burkhardt, and Bjorn W. Schuller, “Probing speech emotion¨ recognition transformers for linguistic knowledge,” in Proc. Interspeech, 2022, pp. 146–150.

[11] Rachel L. C. Mitchell, “How does the brain mediate interpretation of incongruent auditory emotions? The neural response to prosody in the presence of conflicting lexico-semantic cues,” European Journal ofNeuroscience, vol. 24, no. 12, pp. 3611– 3618, 2006.

[12] Zhifang Guo, Yichong Leng, Yihan Wu, Sheng Zhao, and Xu Tan, “PromptTTS: Controllable text-to-speech with text descriptions,” in Proc. ICASSP, 2023, pp. 1–5.

[13] Dongchao Yang, Songxiang Liu, Rongjie Huang, Chao Weng, and Helen Meng, “InstructTTS: Modelling expressive TTS in discrete latent space with natural language style prompt,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 32, pp. 2913–2925, 2024.

[14] Guanghou Liu, Yongmao Zhang, Yi Lei, Yunlin Chen, Rui Wang, Lei Xie, and Zhifei Li, “PromptStyle: Controllable style transfer for text-to-speech with natural language descriptions,” in Proc. Interspeech, 2023, pp. 4888–4892.

[15] Deok-Hyeon Cho, Hyung-Seok Oh, Seung-Bin Kim, Sang-Hoon Lee, and Seong-Whan Lee, “EmoSphere-TTS: Emotional style and intensity modeling via spherical emotion vector for controllable emotional text-to-speech,” in Proc. Interspeech, 2024, pp. 1810–1814.

[16] Carlos Busso, Reza Lotfian, Kusha Sridhar, Ali N. Salman, Wei-Cheng Lin, Lucas Goncalves, Srinivas Parthasarathy, Abinay Reddy Naini, Seong-Gyun Leem, Luz Martinez-Lucas, Huang-Cheng Chou, and Pravin Mote, “The MSP-Podcast corpus,” IEEE Transactions on Affective Computing, vol. 17, no. 3, pp. 3065–3083, 2026.

[17] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever, “Robust speech recognition via large-scale weak supervision,” in Proc. ICML, 2023, vol. 202 of PMLR, pp. 28492–28518.

[18] Takaaki Saeki, Detai Xin, Wataru Nakata, Tomoki Koriyama, Shinnosuke Takamichi, and Hiroshi Saruwatari, “UTMOS: UTokyo-SaruLab system for VoiceMOS challenge 2022,” in Proc. Interspeech, 2022, pp. 4521–4525.

[19] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen, “LoRA: Low-rank adaptation of large language models,” in Proc. ICLR, 2022.

[20] Detai Xin, Junfeng Jiang, Shinnosuke Takamichi, Yuki Saito, Akiko Aizawa, and Hiroshi Saruwatari, “JVNV: A corpus of Japanese emotional speech with verbal content and nonverbal expressions,” IEEE Access, vol. 12, pp. 19752–19764, 2024.

[21] Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and Furu Wei, “Multilingual E5 text embeddings: A technical report,” arXiv preprint arXiv:2402.05672, 2024.