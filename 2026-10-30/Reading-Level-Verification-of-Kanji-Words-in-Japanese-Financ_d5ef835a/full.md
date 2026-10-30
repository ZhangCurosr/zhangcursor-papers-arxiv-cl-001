# Reading-Level Verification of Kanji Words in Japanese Financial Speech Synthesis

Cheng ZHANG

Best AI Co., Ltd., Tokyo, Japan

Email: bestaijapan@gmail.com

Abstract—Few-shot voice cloning makes it cheap to produce daily Japanese financial commentary with several synthetic speakers, but kanji words with more than one reading, such as 市 場 (shijō vs. ichiba) or 大 勢 (taisei vs. ōze), are still misread. A common quality gate transcribes each synthesized sentence with automatic speech recognition (ASR), converts both the input and the transcript to kana, and regenerates on mismatch. We show that this gate is unreliable in both directions: when the audio contains a misreading, the ASR language model frequently writes the expected kanji back, hiding the error, and when the audio is correct, ordinary recognition errors trigger needless regeneration. We propose a reading-level check that scores the expected reading and its confusable alternatives with a connectionist temporal classification (CTC) hiragana recognizer and flags a word when an alternative explains the audio better. On 540 utterances from three voice-cloned production voices covering 30 financial terms, with misreadings injected under controlled conditions, the transcript-based gate detected 78.9% of misreadings at a 44.4% false-alarm rate, whereas the proposed check detected 91.7% at 5.0% (AUC 0.990). On natural kanji input, it caught a front-end error that the transcript-based gate missed for one voice, and reduced regeneration requests from 38.9% to 6.1% of sentences.

Index Terms—speech synthesis, kanji reading, pronunciation verification, CTC, Japanese, financial content

## I. INTRODUCTION

Synthetic voices are increasingly used to produce daily market commentary for video platforms and podcasts, often as a dialogue between two characters. Few-shot voice cloning systems built on neural codec or semantic-token language models [1], [2] allow a small team to maintain several distinct voices and to generate a ten-minute episode in a few minutes. In this setting the main quality problem is no longer naturalness but correctness. Listeners of financial content are sensitive to terminology: a voice that reads 市場 (market) as ichiba (marketplace) or 大勢 (overall trend) as ōze (a crowd) sounds amateurish and damages the credibility of the channel.

Japanese is particularly prone to such errors because many kanji words have several readings whose choice depends on domain and context. Financial text adds a vocabulary of jargon with idiosyncratic readings (上値 uwane, 後場 goba, 手仕 舞い tejimai) and of numeral expressions whose euphonic changes are easy to get wrong (十分 juppun “ten minutes” vs. jūbun “enough”). Misreadings come from two sources: the text front-end may predict the wrong reading, and the acoustic model, fine-tuned on limited speaker data, may deviate from the phonemes it was given. In our production system we have observed both, e.g., 千円 rendered as sennen in all of nine candidate generations although the front-end reading was correct.

A practical and widely used safeguard is an ASR-based gate: synthesize each sentence, transcribe it with a robust recognizer such as Whisper [8], compare the transcript with the input, and regenerate on mismatch. Comparing kana readings rather than surface strings avoids false alarms from orthographic variants (上 げ/揚 げ). Our own pipeline uses exactly this design. However, the gate relies on the recognizer’s text output, and a strong recognizer is also a strong language model. When the audio says ichiba in a sentence about stocks, the most probable transcript is still 市 場, which the morphological analyzer converts back to the expected shijō; the gate passes and the error reaches the listener. Conversely, when the audio is correct, recognition errors on rare jargon change the kana string and cause needless regeneration.

This paper makes three contributions:

We quantify, under controlled error injection, how often the transcript-based kana gate misses misreadings and how often it raises false alarms on correct audio, using three production voices and 30 financial terms.

We propose a reading-level check that compares CTC likelihoods of the expected reading and its confusable alternatives, independent of the recognizer’s choice of kanji.

We show that the check also detects a real front-end misreading in natural generation and sharply reduces unnecessary regeneration.

## II. RELATED WORK

Reading prediction. Japanese TTS front-ends predict readings with a morphological analyzer and dictionary, as in Open JTalk [4] and its Python binding pyopenjtalk [5], optionally with user dictionaries. Neural approaches to heteronym and polyphone disambiguation, e.g., g2pW for Mandarin [6], improve accuracy on general text. All of these act before synthesis and cannot observe errors introduced by the acoustic model, nor verify that a domain term was spoken as intended.

Neural TTS and voice cloning. End-to-end models such as VITS [7] synthesize speech directly from phonemes, while recent few-shot systems treat TTS as language modeling: VALL-E [1] predicts neural codec tokens and SPEAR-TTS [2] predicts self-supervised semantic tokens autoregressively from text, before a second stage generates the waveform. Such twostage systems build custom voices from small amounts of speech, but their autoregressive stage can deviate from the input phonemes, and for Japanese they still depend on a frontend for readings.

![](images/d39ef183d28dec5c8c397f7a209a2980eaed01b243a4009da100b277acb0a1ff.jpg)  
Fig. 1. Generate–verify–regenerate loop. Stage 1 is the deployed transcriptbased gate; Stage 2 is the proposed reading-level check.

ASR for TTS evaluation. ASR transcripts are routinely used to measure the intelligibility of synthesized speech via character or word error rate, and large models such as Whisper [8] and WhisperX [9] are popular for this purpose. Such metrics measure what the recognizer writes, which for logographic scripts is only loosely tied to what was pronounced.

Pronunciation assessment. Scoring the likelihood of an expected phone sequence against alternatives is the basis of goodness-of-pronunciation measures in computer-assisted language learning [10]. We apply the same idea to synthesized speech, using a CTC [11] recognizer with kana outputs and a small set of confusable readings per word.

## III. SYSTEM

## A. Generation Pipeline

Fig. 1 shows the production loop. Scripts are written as dialogue; each line carries a speaker and a speaking-style tag. Lines of the same speaker are synthesized together to avoid repeated model loading and are concatenated in script order with 0.3 s pauses between sentences and 0.6 s at speaker or paragraph changes. The voices are built on an open-source two-stage voice-cloning framework of the type described in Section II: pyopenjtalk converts the text to phonemes, an autoregressive Transformer predicts self-supervised semantic tokens from the phonemes, and a VITS-based decoder [7] generates the waveform. Each voice is fine-tuned with LoRA [3] on approximately 30 minutes of Japanese speech from a single speaker and conditioned on a 3–10 s reference clip per style.

Before synthesis, a substitution table rewrites expressions that have been misread in the past into kana (e.g., 円安 え $\therefore \overrightarrow { o } \overrightarrow { s } , \overrightarrow { F } \overrightarrow { H }  \overrightarrow { t } \overrightarrow { h } \overrightarrow { e } h )$ . Entries are applied longest-match first, may be restricted to particular voices, and numeral-initial entries are skipped when preceded by a digit. The original text is kept for subtitles and verification. The table is reactive by design: an entry is added only after an error has been heard, which is why automatic detection matters.

## B. Stage 1: Transcript-Based Kana Check

Each synthesized sentence is transcribed by WhisperX (large-v3) [9]. Both the input text and the transcript are converted to katakana readings with pyopenjtalk, all nonkana symbols are removed, and the sentence passes only if the two strings are identical. Orthographic variants therefore count as matches. This stage reliably catches skipped, repeated or hallucinated phrases, which are typical failure modes of autoregressive TTS, and misreadings that lead the recognizer to a diferent word.

## C. Stage 2: Reading-Level Check

Stage 2 targets words with known confusable readings. For each such word w in a sentence, let $r ^ { * }$ be the expected reading, taken from a domain lexicon, and $\mathcal { A } ( w )$ a small set of alternative readings that a front-end or acoustic model could plausibly produce (in this study, one alternative per word; Table I). The sentence is split as $x = p w q .$ , and the hiragana readings $\rho ( p )$ and $\rho ( q )$ of the surrounding text are taken from the front-end. For a candidate reading $r ,$ the sentence-level label sequence is $y ( r ) = \rho ( p ) r \rho ( q )$ , and its score is the CTC log-likelihood

$$
s ( r ) = \log P _ { \mathrm { C T C } } ( y ( r ) \mid \mathbf { a } ) ,\tag{1}
$$

where a is the audio and $P _ { \mathrm { C T C } }$ is computed by a wav2vec 2.0 XLSR-53 model [12] fine-tuned on Common Voice Japanese [13] with hiragana targets [14]. The word is flagged if

$$
m ( w ) = \operatorname* { m a x } _ { r \in A ( w ) } s ( r ) - s ( r ^ { * } ) > \theta ,\tag{2}
$$

and the sentence fails if any word is flagged. Because $y ( r )$ and $y ( r ^ { * } )$ difer only in the target span, context errors of the recognizer afect both scores similarly and largely cancel in the margin $m ( w )$ . The decision depends on acoustic evidence for each reading rather than on which kanji a language model would write. The cost is one forward pass of the CTC model per sentence plus one CTC loss evaluation per candidate.

## D. Building the Reading Lexicon

Stage 2 needs, for each word to be verified, its expected reading and a few confusable alternatives. We build this lexicon from three sources. First, every entry added to the substitution table records an observed misreading, which becomes an alternative for that word. Second, domain glossaries of market terms provide the expected financial readings, which often difer from the general-domain default (e.g., 上値 uwane, 後場 goba). Third, alternatives can be enumerated from dictionary readings of the constituent kanji and from the front-end’s own output when it disagrees with the glossary; this third step is not used in the present study, where one alternative per word was chosen manually. Because the check only scores readings and does not need audio examples, a new term can be covered by adding one line to the lexicon.

The expected reading must not be taken from the same frontend that drives the synthesizer: if the front-end is wrong, synthesizer and verifier agree on the wrong reading. Section V-F shows a real instance of this failure.

## E. Regeneration Policy

A sentence failing either stage is regenerated with a new random seed up to K times $( K = 9$ in production: three rounds of three candidates), and the first candidate passing both stages

TABLE I  
FINANCIAL TERMS, EXPECTED READINGS AND INJECTED MISREADINGS
<table><tr><td>Word</td><td>Expected</td><td>Injected</td><td>| Word</td><td>Expected</td><td>Injected</td></tr><tr><td>市場</td><td> $L \vec { \subset } \mathtt { k } \breve { \circ }$ </td><td>いちば</td><td>決算</td><td> $\downarrow  \eth \uparrow$ </td><td> $\downarrow + \supset \ z ^ { \circ } / \downarrow$ </td></tr><tr><td>大手</td><td>おおて</td><td>おおで</td><td>增益</td><td>ぞうえき</td><td>ぞうやく</td></tr><tr><td>上值</td><td>うわね</td><td>じょうね</td><td>千円</td><td>せんえん</td><td>せんねん</td></tr><tr><td>下值</td><td>したね</td><td>しもね</td><td>十分</td><td> $c \textcircled { 1 } 0 \textcircled { 2 } 3 0 \%$ </td><td> $c \textcircled { 1 } 5 . 3 \times h$ </td></tr><tr><td>大引け</td><td>おおびけ</td><td>おおひけ</td><td>人気</td><td>にんき</td><td>ひとけ</td></tr><tr><td>相場</td><td>そうば</td><td>あいば</td><td>大勢</td><td>たいせい</td><td>おおぜい</td></tr><tr><td>銘柄</td><td>めいがら</td><td>めいへい</td><td>一时</td><td> $\hphantom { 0 } 0 \hphantom { 0 } \forall \hphantom { 0 } \forall \hphantom { 0 }$ </td><td> $\alpha \geq \gamma \notin$ </td></tr><tr><td>手仕舞い</td><td>てじまい</td><td>てしまい</td><td>後場</td><td>ごば</td><td>あとば</td></tr><tr><td>円安</td><td>えんやす</td><td>えんあん</td><td>前場</td><td>ぜんば</td><td>まえば</td></tr><tr><td>円高</td><td>えんだか</td><td>えんこう</td><td>空売り</td><td>からうり</td><td>くううり</td></tr><tr><td>為替</td><td>かわせ</td><td>いがえ</td><td>利食い</td><td>りぐい</td><td>りしょく</td></tr><tr><td>先物</td><td>さきもの</td><td>せんもつ</td><td>日経平均</td><td></td><td>にっけいへいきんにちけいへいきん</td></tr><tr><td>出来高</td><td>できだか</td><td>できたか</td><td>動向</td><td> $z ^ { * } 5 z 5$ </td><td> $\varepsilon > \ 5 \Re$ </td></tr><tr><td>值幅</td><td>ねはば</td><td>ちはば</td><td>下落</td><td>げらく</td><td>からく</td></tr><tr><td>押し目</td><td>おしめ</td><td>おしもく</td><td>寄り付き</td><td> $\pounds \} \}$ </td><td> $\downarrow \ 1  \eth$ </td></tr></table>

is kept. If none passes, the best-scoring candidate is used and the flagged word is reported, so that the operator can add an entry to the substitution table.

## IV. EXPERIMENTAL SETUP

## A. Voices and Test Set

We used three production voices: two female (F1, F2) and one male (M1), fine-tuned from the framework’s pretrained models with LoRA ranks 64, 32 and 32, respectively. Each voice was conditioned on its default reference clip.

We selected 30 terms frequent in Japanese market commentary, each with one financial reading and one plausible misreading (Table I). The misreadings fall into three types: (a) reading choice, where a diferent on- or kun-reading of the same characters is used (21 terms, e.g., 市場 ichiba, 前 場 maeba, 相場 aiba); (b) sound change, where the correct morphemes are used but voicing or gemination is wrong (7 terms, e.g., 大手 ōde, 寄り付き yoridzuki, 手仕舞い teshimai, 日経平均 nichikei); and (c) numeral euphony (2 terms: 十分 jūbun, 千円 sennen). Type (a) changes several morae, whereas type (b) often changes a single phonetic feature. For each term we wrote two natural sentences in which context determines the financial reading, giving 60 sentences with a mean duration of 4.1 s when synthesized.

## B. Conditions and Ground Truth

Each sentence was synthesized by each voice under three input conditions with a fixed seed: K, the target word replaced by its expected reading in hiragana; W, the target word replaced by the injected misreading in hiragana; and N, the original kanji text. This yields $6 0 \times 3 \times 3 = 5 4 0$ utterances. Because the front-end converts hiragana to phonemes deterministically, K and W utterances are requested to contain the correct and the wrong reading, respectively, which provides ground truth without manual labeling. The 360 K and W utterances form the controlled set; the 180 N utterances measure behavior on natural input. In all conditions the verifier is given the original kanji sentence, as in production.

TABLE II  
DETECTION ON THE CONTROLLED SET (3 VOICES, 180 W AND 180 K UTTERANCES). BRACKETS: WILSON 95% CI.
<table><tr><td>Method</td><td>Recall (W)</td><td>False alarm (K)</td><td>Precision</td></tr><tr><td>Stage 1: Whisper + kana</td><td>78.9 [72.4, 84.2]</td><td>44.4 [37.4, 51.7]</td><td>64.0</td></tr><tr><td>Greedy hiragana CTC</td><td>100 [97.9, 100]</td><td>48.3 [41.1, 55.6]</td><td>67.4</td></tr><tr><td>Proposed  $( \theta = 0 )$ </td><td>91.7 [86.7, 94.9]</td><td> ${ \bf 5 . 0 } \ [ 2 . 7 , 9 . 2 ]$ </td><td>94.8</td></tr></table>

![](images/2b27003fcf684e474e227a1b7e6beab6ae13d76bdf6342117176ef97f88f6e8a.jpg)  
Fig. 2. Distribution of the margin m(w) in (2) for utterances with the correct (K) and the injected wrong (W) reading. Dashed line: $\theta = 0 .$

## C. Compared Methods

We compare (i) the deployed Stage 1 gate; (ii) a direct phonetic transcription baseline, which decodes the hiragana CTC model greedily and flags the word if the expected reading does not occur in the output; and (iii) the proposed Stage 2 check with $\theta = 0 ,$ . We report recall on W utterances, falsealarm rate on K utterances, precision, and the area under the ROC curve (AUC) of the margin $m ( w )$ , with Wilson 95% confidence intervals and exact McNemar tests for paired comparisons.

## V. RESULTS

## A. Controlled Set

Table II summarizes the results. The deployed gate missed one in five injected misreadings and, more strikingly, rejected 44.4% of utterances that contained the correct reading. Greedy phonetic transcription detected every misreading, but its output is noisy (e.g., ぼうひけ for おおびけ) and it rejected 48.3% of correct utterances, which makes it unusable as a gate. The proposed check detected 91.7% of misreadings with a 5.0% false-alarm rate. Compared with Stage 1, it detected 35 misreadings that Stage 1 missed while missing 12 that Stage 1 caught $( p = 0 . 0 0 1 )$ , and it raised 3 false alarms that Stage 1 did not versus 74 in the opposite direction $( p < 1 0 ^ { - 1 7 } )$ .

Fig. 2 shows that the margin separates the two classes well (median 26.9 for K and +23.1 for W; AUC 0.990). The threshold ofers a simple operating trade-of (Fig. 3 and Table III): $\theta = 5$ gives 90.6% recall at 1.7% false alarms, and $\theta ~ = ~ - 1 0$ gives 98.9% recall at 15.0% false alarms. Every operating point of the proposed check between these two dominates both baselines, which lie far to the right of the curve. Results were consistent across voices (Table IV); AUC ranged from 0.983 to 0.998, with the best separation for the male voice.

![](images/7219c0bda7dca23092af7bff1cd86aaeffdb855efb1d0a2b827099f05eee4e53.jpg)  
Fig. 3. ROC curve of the proposed margin on the controlled set, with the operating points of the two baselines.

TABLE III  
EFFECT OF THE THRESHOLD θ (CONTROLLED SET, 3 VOICES)
<table><tr><td>θ</td><td>Recall (W)</td><td>False alarm (K)</td></tr><tr><td>-10</td><td>98.9% (178/180)</td><td>15.0% (27/180)</td></tr><tr><td>-5</td><td>93.9% (169/180)</td><td>8.3% (15/180)</td></tr><tr><td>0</td><td>91.7% (165/180)</td><td>5.0% (9/180)</td></tr><tr><td>5</td><td>90.6% (163/180)</td><td>1.7% (3/180)</td></tr><tr><td>10</td><td>86.7% (156/180)</td><td>0.6% (1/180)</td></tr></table>

## B. Why the Transcript-Based Gate Fails

Of the 38 injected misreadings that passed Stage 1, Whisper wrote the target kanji word in 36. Table V lists typical cases for voice F1: the audio clearly contains the wrong reading, yet the transcript is the correctly written sentence, so its kana reading matches the input and the gate passes. The efect is strongest for the most common terms, where the language-model prior is strongest: Stage 1 detected misreadings of 相場 in only 1 of 6 utterances and of 市場, 後場 and 日経平均 in 2 of 6 each, whereas the proposed check detected 6, 5, 6 and 6 of 6 (Table VI).

Stage 1 false alarms on correct audio were caused by recognition errors on jargon and homophones that change the kana string: 上値 transcribed as お金, 手仕舞い as 手締い, 大引け as 大引き, 円高 as 縁どこ. In production, each such false alarm costs a regeneration and, when all candidates fail, leaves a correct sentence marked as suspicious.

## C. Failure Cases of the Proposed Check

The proposed check missed all six misreadings of 寄り付き (yoritsuki vs. yoridzuki). The two readings difer only in the voicing of one mora, and the CTC model’s outputs for this region were unstable (e.g., yarisuki, yoritsuki) in both conditions. Excluding this term, recall was 94.8% (165/174) at the same false-alarm count. The remaining misses (円安, 空売り, 大勢, 市場, 大手, 人気; one or two utterances each) had small negative margins in utterances where the recognizer’s output was degraded overall. The nine false alarms concentrated on 大引け and 手仕舞い, terms whose correct reading contains a voiced mora that the recognizer frequently heard as unvoiced. These cases suggest that minimal voicing contrasts call for a stronger acoustic model or a word-specific threshold.

TABLE IV  
PER-VOICE RECALL / FALSE-ALARM RATE (%) AND AUC OF THE MARGIN
<table><tr><td>Voice</td><td>Stage 1</td><td>Proposed (θ = 0)</td><td>AUC</td></tr><tr><td>F1 (female)</td><td>81.7 / 50.0</td><td>93.3 / 6.7</td><td>0.988</td></tr><tr><td>F2 (female)</td><td>80.0 / 45.0</td><td>86.7 / 5.0</td><td>0.983</td></tr><tr><td>M1 (male)</td><td>75.0 / 38.3</td><td>95.0 / 3.3</td><td>0.998</td></tr></table>

TABLE V

INJECTED MISREADINGS HIDDEN BY THE RECOGNIZER (VOICE F1). THE TRANSCRIPT RESTORES THE EXPECTED KANJI, SO STAGE 1 PASSES.
<table><tr><td>TTS input (injected reading)</td><td>Whisper transcript</td><td>m(w)</td></tr><tr><td>今週の東京いちばは、…</td><td>今週の東京市場は…</td><td>+19.3</td></tr><tr><td>金利の上昇が、あいばの重しに…</td><td>金利の上昇が相場の重しに…+28.7</td><td></td></tr><tr><td>取引開始からじゅうぶん後に、…</td><td>取引開始から十分後に、…</td><td>+32.1</td></tr><tr><td>あとばは日銀の発表待ちで、…</td><td>後場は日銀の発表待ちで、…+24.7</td><td></td></tr><tr><td>にちけいへいきんは、三日ぶりに…</td><td>日経平均は3日ぶりに…</td><td>+16.4</td></tr><tr><td>原油価格のからくが、…</td><td>原油価格の下落が…</td><td>+22.9</td></tr></table>

Table VII confirms this pattern by misreading type. For reading-choice errors, which change several morae, the proposed check detected 93.7% with a single false alarm in 126 correct utterances, while Stage 1 produced 61 false alarms. Sound-change errors were harder for both methods: the proposed check detected 83.3% (35/42), six of the seven misses being 寄り付き, and 7 of its 9 false alarms fell into this type. Numeral-euphony errors were detected in all 12 utterances.

## D. Combining the Two Stages

In production the two stages are applied in sequence, so a sentence is rejected if either fails. With the deployed exactmatch Stage 1, the combination detects 177/180 misreadings but inherits Stage 1’s false alarms (83/180 on K). Stage 1 is still needed for omissions, insertions and garbled speech, which Stage 2 does not target, but it need not require an exact kana match. Relaxing Stage 1 to reject only sentences whose kana similarity to the input (diflib ratio) falls below 0.8 leaves just 2 Stage 1 rejections among the 180 correct K utterances; combined with Stage 2 this yields 165/180 recall at 11/180 false alarms, and 12/180 regeneration requests on natural input. We therefore recommend a lenient Stage 1 for gross errors and Stage 2 for readings. Our test set contains no omissions, so the appropriate Stage 1 threshold for catching them remains to be validated.

## E. Computational Cost

Stage 2 requires one forward pass of the 300M-parameter CTC model per sentence and one CTC loss per candidate reading. On one NVIDIA H100 NVL GPU, scoring took 17.6 ms per sentence on average (median 14.9 ms) for utterances averaging 4.1 s, a real-time factor of 0.004, which is negligible compared with synthesis and Whisper transcription.

TABLE VI  
DETECTIONS PER TERM (OUT OF 6 UTTERANCES: 3 VOICES 2 SENTENCES). W: MISREADINGS DETECTED; K: FALSE ALARMS.
<table><tr><td>Term</td><td>Stage 1 W</td><td>K</td><td>Proposed W</td><td>K</td><td>Term</td><td>Stage 1 W</td><td>K</td><td>W</td><td> $\mathrm { P r o p o s e d }$  K</td></tr><tr><td>市場</td><td>2</td><td>3</td><td>5</td><td>0</td><td>決算</td><td>5</td><td>1</td><td>6</td><td>0</td></tr><tr><td>大手</td><td>3</td><td>1</td><td>5</td><td>0</td><td>增益</td><td>6</td><td>3</td><td>6</td><td>0</td></tr><tr><td>上值</td><td>6</td><td>3</td><td>6</td><td>0</td><td>千円</td><td>6</td><td>1</td><td>6</td><td>0</td></tr><tr><td>下值</td><td>6</td><td>1</td><td>6</td><td>0</td><td>十分</td><td>3</td><td>1</td><td>6</td><td>1</td></tr><tr><td>大引け</td><td>3</td><td>2</td><td>6</td><td>3</td><td>人気</td><td>3</td><td>4</td><td>5</td><td>0</td></tr><tr><td>相場</td><td>1</td><td>0</td><td>6</td><td>0</td><td>大勢</td><td>5</td><td>6</td><td>4</td><td>0</td></tr><tr><td>銘柄</td><td>6</td><td>4</td><td>6</td><td>0</td><td>一時</td><td>6</td><td>0</td><td>6</td><td>0</td></tr><tr><td>手仕舞い</td><td>6</td><td>6</td><td>6</td><td>3</td><td>後場</td><td>2</td><td>4</td><td>6</td><td>0</td></tr><tr><td>円安</td><td>5</td><td>2</td><td>4</td><td>0</td><td>前場</td><td>5</td><td>4</td><td>6</td><td>0</td></tr><tr><td>円高</td><td>6</td><td>2</td><td>6</td><td>1</td><td>空売り</td><td>5</td><td>3</td><td>4</td><td>0</td></tr><tr><td>為替</td><td>6</td><td>3</td><td>6</td><td>0</td><td>利食い</td><td>6</td><td>3</td><td>6</td><td>0</td></tr><tr><td>先物</td><td>6</td><td>5</td><td>6</td><td>0</td><td>日経平均</td><td>2</td><td>0</td><td>6</td><td>1</td></tr><tr><td>出来高</td><td>6</td><td>3</td><td>6</td><td>0</td><td>動向</td><td>6</td><td>2</td><td>6</td><td>0</td></tr><tr><td>值幅</td><td>6</td><td>6</td><td>6</td><td>0</td><td>下落</td><td>4</td><td>1</td><td>6</td><td>0</td></tr><tr><td>押し目</td><td>6</td><td>2</td><td>6</td><td>0</td><td>寄り付き</td><td>4</td><td>4</td><td>0</td><td>0</td></tr></table>

TABLE VII

RECALL / FALSE ALARMS BY MISREADING TYPE (CONTROLLED SET)
<table><tr><td>Type (terms, utterances per class)</td><td>Stage 1</td><td>Proposed</td></tr><tr><td>(a) Reading choice (21, 126)</td><td>104/126, 61 FA</td><td>118/126, 1 FA</td></tr><tr><td>(b) Sound change (7, 42)</td><td>29/42, 17 FA</td><td>35/42, 7 FA</td></tr><tr><td>(c) Numeral euphony (2, 12)</td><td>9/12, 2 FA</td><td>12/12, 1 FA</td></tr></table>

## F. Natural Input

With the original kanji input, the front-end produced the expected reading for 29 of 30 terms; it read 大勢 as ōze (a crowd) in both of its sentences, so all six N utterances of 大勢 contained a real misreading originating in the frontend. Stage 1 flagged 4 of the 6: for M1 both utterances passed because Whisper transcribed the audio as 大勢, which pyopenjtalk again read as $\bar { o } z e ,$ , matching the (equally wrong) expected reading derived from the input. The proposed check, whose expected reading comes from the domain lexicon rather than the front-end, flagged 5 of 6. This illustrates a second blind spot: a verifier that derives its expectation from the same front-end as the synthesizer cannot detect front-end errors.

On the remaining 174 natural utterances, Stage 1 requested regeneration for 66 (37.9%) and the proposed check for 6 (3.4%); over all 180 natural utterances the rates were 38.9% and 6.1%. Stage 1’s rate varied strongly with the voice (F1 50.0%, F2 41.7%, M1 25.0%), reflecting how well Whisper recognized each voice rather than how many misreadings it produced; the proposed check’s rate was 10.0%, 5.0% and 3.3%, respectively. The six natural utterances flagged only by the proposed check (大引け 2, 手仕舞い 3, $^ { \neg } \mathbb { H } _ { \overline { { \cdot } } \overline { { \ v { * } } } } ^ { \pm } \times 1 )$ were verified by a native listener, and all six contained the correct reading. They are therefore false alarms, and they involve the same voiced morae (bi in ōbike, ji in tejimai) that caused false alarms in the controlled set. In total, 5 of the proposed check’s 11 alarms on natural input were genuine front-end misreadings and 6 were false alarms, a false-alarm rate of 3.4% (6/174) on utterances without front-end errors, consistent with the 5.0% measured on the controlled set. Stage 1 raised 66 alarms on the same 174 utterances; we did not verify all of them by listening, but the front-end readings of these utterances were correct, 4 of them are among the six utterances verified as correct above, and the remaining 62 had negative margins, so most of them are likely needless regenerations.

![](images/d40e91d5991b596493bee8116048d4da237828b2a2c0e12580154d6ae74ce57c.jpg)  
Fig. 4. Margin $m ( w )$ per voice and condition (K: correct reading injected, W: wrong reading injected, N: natural kanji input). Natural utterances behave like K except for the front-end misreadings of 大勢, which appear as positive outliers.

## VI. DISCUSSION AND LIMITATIONS

Practical guidelines. From these results we derive three recommendations for production pipelines. (1) Do not use exact kana match of an ASR transcript as the only correctness gate for Japanese: it both hides misreadings of common terms and rejects many correct sentences. Use it leniently, for gross errors only. (2) Verify readings of known ambiguous terms with likelihood comparison of explicit candidates, with the expected reading taken from a domain lexicon rather than from the synthesizer’s front-end. (3) Choose θ according to the cost of errors: for published content where a misreading is costly, a negative threshold (e.g., θ = 5, 93.9% recall at 8.3% false alarms) is afordable because a false alarm only costs one regeneration of a short sentence. Fig. 4 also shows that natural utterances score like the K condition, so the controlled falsealarm rates are representative of production.

Limitations. The controlled set relies on injected misreadings, which are spoken fluently by the TTS model and may be easier to detect than natural acoustic deviations that are partially articulated. The ground truth assumes that the acoustic model followed the requested phonemes; the Kcondition false alarms show that this is mostly but not always true, so a small part of the reported errors may be label noise. The proposed check requires a lexicon of terms with their expected and confusable readings. In our use case this lexicon grows naturally from the substitution table and from domain glossaries, but it does not cover unseen words; Stage 1 remains necessary for omissions and insertions, and the two stages are complementary. Finally, the study uses one TTS architecture, one CTC model trained on read speech, and 60 sentences; pitch-accent errors, which are also noticeable to native listeners, are not addressed.

## VII. CONCLUSION

We showed that ASR-based verification of synthesized Japanese, even at the kana level, is unreliable for kanji reading errors: the recognizer’s language model hides misreadings of common financial terms, recognition errors on jargon cause frequent false alarms, and expectations derived from the synthesizer’s own front-end cannot reveal front-end errors. A reading-level check that compares CTC likelihoods of the expected and confusable readings detected 91.7% of injected misreadings at a 5.0% false-alarm rate across three voices, caught a real front-end misreading, and reduced regeneration requests on natural input from 38.9% to 6.1%. Future work includes automatic generation of confusable readings, wordspecific thresholds for voicing contrasts, pitch-accent verification, and extension to Mandarin polyphonic characters.

## REFERENCES

[1] C. Wang et al., “Neural codec language models are zero-shot text to speech synthesizers,” arXiv:2301.02111, 2023.

[2] E. Kharitonov et al., “Speak, read and prompt: High-fidelity text-tospeech with minimal supervision,” Trans. Assoc. Comput. Linguist., vol. 11, pp. 1703–1718, 2023.

[3] E. J. Hu et al., “LoRA: Low-rank adaptation of large language models,” in Proc. ICLR, 2022.

[4] Open JTalk: HMM-based Japanese text-to-speech system. [Online]. Available: https://open-jtalk.sourceforge.net/

[5] R. Yamamoto, “pyopenjtalk: A Python wrapper for OpenJTalk,” GitHub repository. [Online]. Available: https://github.com/r9y9/pyopenjtalk

[6] Y.-C. Chen, Y.-C. Chang, Y.-C. Chang, and Y.-R. Yeh, “g2pW: A conditional weighted softmax BERT for polyphone disambiguation in Mandarin,” in Proc. Interspeech, 2022, pp. 1926–1930.

[7] J. Kim, J. Kong, and J. Son, “Conditional variational autoencoder with adversarial learning for end-to-end text-to-speech,” in Proc. ICML, PMLR vol. 139, 2021, pp. 5530–5540.

[8] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, “Robust speech recognition via large-scale weak supervision,” in Proc. ICML, PMLR vol. 202, 2023, pp. 28492–28518.

[9] M. Bain, J. Huh, T. Han, and A. Zisserman, “WhisperX: Time-accurate speech transcription of long-form audio,” in Proc. Interspeech, 2023, pp. 4489–4493.

[10] S. M. Witt and S. J. Young, “Phone-level pronunciation scoring and assessment for interactive language learning,” Speech Communication, vol. 30, no. 2, pp. 95–108, 2000.

[11] A. Graves, S. Fernández, F. Gomez, and J. Schmidhuber, “Connectionist temporal classification: Labelling unsegmented sequence data with recurrent neural networks,” in Proc. ICML, 2006, pp. 369–376.

[12] A. Conneau, A. Baevski, R. Collobert, A. Mohamed, and M. Auli, “Unsupervised cross-lingual representation learning for speech recognition,” in Proc. Interspeech, 2021, pp. 2426–2430.

[13] R. Ardila et al., “Common Voice: A massively-multilingual speech corpus,” in Proc. LREC, 2020, pp. 4218–4222.

[14] vumichien, “wav2vec2-large-xlsr-japanese-hiragana,” Hugging Face model, 2022, doi:10.57967/hf/0344. [Online]. Available: https:// huggingface.co/vumichien/wav2vec2-large-xlsr-japanese-hiragana