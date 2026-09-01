# Context-Aware Interleaved Batching for WhisperX

Carlos Bain<sup>∗</sup>, Max Bain<sup>†</sup>

University of Oxford<sup>†</sup>

![](images/76a4e6d65187a5aa1d07018f50996e79263f5d0702b94095a148dedeb659db2a.jpg)  
Figure 1: Context-Aware Interleaved Batching for speech transcription. Long-form audio is first segmented into chunks (≤30s) via Voice Activity Detection (VAD). These chunks are partitioned into contiguous streams and batched by drawing one chunk per stream. To preserve text context during batched transcription, each chunk is conditioned on a FIFO rolling text buffer generated from its stream’s preceding chunks. Finally, to resolve the cold-start problem, thefirst batch is re-transcribed using thefinal text of chronologically preceding streams as context.

## Abstract

While WhisperX accelerates speech transcription via intraaudio batching, it isolates audio segments, losing the historical context needed for coherent punctuation and terminology transcription. Conversely, standard Whisper retains context sequentially but suffers from slow inference and hallucination loops. To achieve the best of both worlds, we propose Context-Aware Interleaved Batching. By using VAD-derived segment boundaries, our algorithm stabilizes Whisper’s text conditioning, allowing us to safely maintain continuous historical context across batched audio segments. As demonstrated on longform audio benchmarks, this approach reduces Word Error Rate (WER) and improves proper noun transcription, all while maintaining high-throughput inference speeds.

## 1. Introduction

Large-scale speech recognition models, such as OpenAI’s Whisper [1], achieve high accuracy across diverse domains. However, deploying these models on long-form audio is computationally expensive. Standard Whisper processes audio sequentially, preventing batched inference. Furthermore, while Whisper includes a condition on previous text feature to maintain historical context, it is often unstable. As demonstrated in our evaluations, enabling this feature in standard Whisper can actually degrade overall Word Error Rate (WER), a known symptom of the model hallucinating or propagating errors across sequential forward passes [2]. In addition, recent work [3] further highlights the risks of autoregressive context in long-form transcription, showing that an erroneous context can propagate across successive decoding steps and trigger hallucinations.

To resolve this, WhisperX [4] introduced a Voice Activity Detection (VAD) based pipeline. By segmenting audio at natural silences rather than arbitrary fixed-length windows, WhisperX offers significant inference speedups via within-audio batched transcription, improves overall transcription accuracy, and adds highly accurate word-level timestamps via forced alignment with a CTC model [5]. Despite these advantages, batched processing inherently isolates each audio chunk. This prevents the model from conditioning on previous text, stripping it of its autoregressive context capabilities. On longform audio, this loss of historical context degrades performance on continuous conversational threads, technical language, and proper nouns. Prior works in end-to-end ASR have long established that such context, via contextual biasing, is essential for accurately transcribing rare and domain-specific terms [6, 7, 8].

Currently, no transcription method simultaneously achieves high parallelization, reliable word-level timestamps, and stable autoregressive conditioning. We address this with Context-Aware Interleaved Batching. By propagating historical text across parallel batch boundaries, our method maintains continuous context without sacrificing inference speed. Crucially, decoupling audio chunk boundaries from Whisper’s internal timestamp decoding prevents hallucination loops, stabilizing the model’s conditioning on previous text and improving accuracy.

Our contributions are threefold: (i) a novel method enabling historical text conditioning within fully parallelized transcription frameworks; (ii) a new, open-source medical long-form benchmark, evaluated alongside targeted Proper Noun (Pn) and punctuation metrics; and (iii) empirical demonstration that our method improves accuracy across all metrics and integrates into any batched, segmentation-based transcription pipeline.

## 2. Method

In this section, we describe the parallel speech transcription baseline introduced by WhisperX [4], followed by our proposed Context-Aware Interleaved Batching method.

## 2.1. Overview of Baseline Architectures

We first review the autoregressive conditioning mechanism of the standard Whisper model [1]. Whisper operates as an encoder-decoder Transformer. During standard sequential inference, the audio is processed in fixed windows up to 30 seconds. When transcribing a given window i, the model relies on two distinct attention mechanisms to generate text:

1. Cross-Attention: The encoder processes the audio features of window i and passes the representations to the decoder.

2. Self-Attention (Conditioning): Because Whisper processes these audio windows chronologically, the decoded text tokens from the previous window i − 1 can be provided as a conditioning prefix to the decoder for window i.

To determine the boundaries of the input audio chunks for subsequent passes, standard Whisper autoregressively decodes specialized timestamp tokens. By predicting the end time of the current speech segment, the model dictates how far to shift the 30-second audio window for the next forward pass. However, this internal timestamp decoding is prone to errors and hallucination, frequently causing the model to predict incorrect audio boundaries, which leads to erroneous window shifting, looping, and severe text hallucinations. Furthermore, the condition on previous text feature acts as a First-In, First-Out (FIFO) rolling buffer, retaining up to 224 previously decoded text tokens as a historical prefix for the next chunk. While designed to maintain conversational continuity, feeding a 224-token context derived from erroneous timestamp boundaries often traps the model in a negative feedback loop, propagating hallucinations forward.

WhisperX circumvents this fragile timestamp decoding by relying on a robust Voice Activity Detection (VAD) pipeline. The long-form audio is first segmented by a lightweight VAD model [9] at natural silences, rather than relying on the Whisper model’s internal text-based timestamps. This results in a temporally ordered sequence of N variable-length chunks, denoted as $\mathbf { s } = \left[ s _ { 0 } , s _ { 1 } , \ldots , s _ { N - 1 } \right]$ . These individual chunks (each up to 30 seconds) are then grouped into batches and transcribed in parallel, enabling significant inference speedups, as well as improved word-level timestamps and overall transcription accuracy. However, batched processing isolates each chunk. If chunks $s _ { i - 1 }$ and $s _ { i }$ are processed simultaneously in the same batch, the text tokens of $s _ { i - 1 }$ cannot be passed as a prefix to $s _ { i } .$ Without this text conditioning, the model loses contextual awareness across chunk boundaries. This frequently results in textual inconsistencies between segments, particularly regarding punctuation, capitalization, and the transcription of proper nouns.

## 2.2. Context-Aware Interleaved Batching

To retain both the autoregressive text-conditioning of standard Whisper and the throughput of parallel transcription, we propose Context-Aware Interleaved Batching. Given the VADsegmented sequence s and a target hardware batch size $B ,$ we partition the sequence into B contiguous streams of length $M \bar { = } \lceil N / B \rceil$ (padding the final stream with empty audio if N is not divisible by B). The b-th stream, for $b \in \{ 0 , 1 , \ldots , B -$ 1}, is defined as:

$$
\mathbf { S } _ { b } = \left[ s _ { b \cdot M } , s _ { b \cdot M + 1 } , . . . , s _ { b \cdot M + M - 1 } \right]\tag{1}
$$

Transcription proceeds over M sequential batch iterations. At step $m \in \{ 0 , 1 , \ldots , M - 1 \}$ , we construct batch $B _ { m }$ by extracting the m-th chunk from every stream:

$$
B _ { m } = \{ { \bf S } _ { b } [ m ] \ | \ b \in \{ 0 , 1 , \ldots , B - 1 \} \}\tag{2}
$$

Because chunk $\mathbf { S } _ { b } [ m ]$ immediately follows $\mathbf { S } _ { b } [ m \mathrm { ~ - ~ } 1 ]$ in the original audio, the decoded text from batch $\boldsymbol { B } _ { m - 1 }$ is appended to a stream-specific FIFO rolling context buffer (capped at 224 tokens). This rolling buffer then serves as the conditioning prefix for the corresponding chunk in batch $\boldsymbol { B _ { m } }$ . This enables standard Whisper’s autoregressive context passing across the audio while safely operating on isolated VAD boundaries, all while processing B chunks in parallel.

Cold-Start Resolution. At the start of each stream $( m =$ 0), the preceding chronological chunk is $\mathbf { S } _ { b - 1 } [ M - 1 ]$ , which belongs to the final batch $_ { B _ { M - 1 } }$ . Since $B _ { M - 1 }$ is not yet transcribed when evaluating $\scriptstyle B _ { 0 }$ , the initial batch $\scriptstyle B _ { 0 }$ is processed without text conditioning. Once all M batches are transcribed, $\scriptstyle B _ { 0 }$ is re-transcribed using the newly generated text from $B _ { M - 1 }$ as the context prefix, ensuring consistent context boundaries across all streams.

1 def batched\_transcription(audio, batch\_size,   
max\_ctx=224):   
2 11 1   
3 audio - input audio signal   
4 batch\_size - number of streams (B)   
5 max\_ctx - maximum number of text tokens for   
conditioning   
6 returns:   
7 segs - chronological array of transcript   
segments   
8   
9 # Extract speech chunks <= 30s   
10 chunks = extract\_vad\_chunks(audio)   
11   
12 # Split into B sequential streams   
13 streams = distribute\_contiguous(chunks,   
batch\_size)   
14   
15 # Group across streams for batching   
16 batches = interleave\_roundrobin(streams)   
17   
18 # Initialise empty rolling ctx per stream   
19 ctx = [[] for \_ in range(batch\_size)]   
20 segs = []   
21   
22 for b in batches:   
23 out = batch\_transcribe(b, ctx)   
24 segs.append(out)   
25   
26 # update rolling context   
27 for sdx in range(len(out)):   
28 # Extract token IDs to feed into   
29 # the next forward pass   
30 ctx[sdx] += out[sdx][’tokens’]   
31 # Truncate to the max ctx window   
32 ctx[sdx] = ctx[sdx][-max\_ctx:]   
33   
34 # Cold start resolution: stream i gets stream   
i-1th ctx.   
35 ctx\_wrap = [None] + ctx[:-1]   
36 segs[0] = batch\_transcribe(batches[0],   
ctx\_wrap)   
37   
38 # Flatten batches and sort chronologically   
39 return sort\_by\_time(segs)  
Listing 1: Pseudocode for Context-Aware Interleaved Batching for WhisperX transcription.

## 3. Results

## 3.1. Evaluation Methodology

## 3.1.1. Evaluation Datasets

Earnings-21 [10]. To evaluate our method, we use two benchmarks. First, the widespread Earnings-21 corpus, a challenging dataset consisting of long-form, unscripted financial earnings calls from the year 2020. This corpus is particularly well-suited for benchmarking real-world transcription systems due to its high density of complex financial terminology, proper nouns, and continuous speech segments. Crucially, the Earnings-21 dataset provides strict ground truth punctuation markers, enabling us to evaluate punctuation accuracy.

MedicalLessons. Second, we introduce a custom corpus of medical terminology lessons: MedicalLessons - a purpose-built dataset consisting of educational lecture recordings covering foundational and anatomical medical vocabulary. The corpus contains a structured series of lessons covering topics such as medical prefixes and suffixes, anatomical terminology, sourced from YouTube. Ground-truth transcripts for all six recordings were manually verified to ensure correctness. This is released as an open source dataset<sup>\*</sup>.

## 3.1.2. Metrics

WER. To measure transcription accuracy we use the standard word error rate metric (WER).

F1. For punctuation we use F1; punctuation alignment between reference and hypothesis transcriptions is computed using a Levenshtein distance-based algorithm to isolate and evaluate grammatical pauses and structural errors independently of the normalized text. All metrics are computed at the transcript level rather than as averages over individual segments. For WER, all predicted segments within a file are concatenated into a single hypothesis string and all ground-truth segments are concatenated into a single reference string; WER is then computed once on these full-length strings using standard Levenshtein-based alignment.

Both of these metrics of course have limitations in their exact matching nature, and lack of “semantic” measurement. Standard ASR metrics have not been shown to measure proper nouns, technical language and punctuation properly [11]. Therefore we additionally introduce LLM-based metrics.

LLM Punctuation Score. To calculate punctuation accuracy based on semantic meaning rather than rigid exact matches, we introduce the LLM Punctuation Score (LPS), similar to the LLM as judge metric shown in [11]. LPS ensures that stylistic choices, such as using a dash instead of a comma or a period instead of a clause-separating comma are not penalised. To compute the LPS, ground truth and predicted transcripts are stripped of casing and punctuation for word-level alignment using the jiwer library, the punctuation is separately tagged. These fully punctuated texts are divided into segments of exactly 120 GT words. Each paired segment is then passed to GPT-5.1 to evaluate the semantic accuracy of the predicted punctuation on a 1-to-3 scale. The prompt instructs the model to ignore word-level transcription errors and penalise only structural divergences that distort the core meaning. The final

LPS for a predicted transcript is the mean of these individual segment scores. Given a transcript divided into $N$ segments, where each segment i receives a score $S _ { i } ~ \in ~ \{ 1 , 2 , 3 \}$ , the overall score is computed as $\begin{array} { r } { \mathrm { L P S } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } S _ { i } } \end{array}$

Proper Noun Score (Pn). For measuring adherence to proper nouns and domain-specific technical terms , we propose Proper Noun Score (Pn). Each ground-truth word is labeled as a proper noun or technical term by an LLM-based classifier. Word-level alignment between the ground-truth and candidate transcript is then used to identify substitutions and deletions. The Pn Score measures the proportion of proper nouns that were correctly transcribed:

$$
\mathrm { P n } = { \frac { P - M } { P } } \times 1 0 0\tag{3}
$$

where $P$ is the total number of proper nouns in the ground truth and M is the number of those that were incorrectly transcribed (substituted or deleted). A score of 100 indicates perfect proper-noun accuracy. This metric is motivated by the limitations of conventional WER for capturing the significance of transcription errors. Semantic-WER [12] addresses this limitation by assigning greater importance to semantically significant errors rather than treating all word errors equally. More recently, Baneras-Roux et al. [13] investigate generative LLMs˜ as ASR evaluators, showing that LLM-based evaluation can achieve substantially higher agreement with human judgments than WER when grading between competing transcription hypotheses. To better evaluate transcriptions, our Pn Score targets domain-specific technical terms and proper nouns, a superior way of measuring transcription quality.

## 3.1.3. Implementation Details

All evaluations were conducted using the openai/whisper-large-v2 as the base transcription model, executing on a single Nvidia T4 GPU. The default batch size of 16 was used on both WhisperX variants for the earnings data set. For the medical dataset, a batch size of 4 was used. Moving on, across both datasets compute type float16 was applied to all variants. Decoding used beam search (width 5, temperature 0.0), falling back to best-of-5 sampling across temperatures 0.2-1.0. Default settings were used for both WhisperX and Whisper from their respective repositories. Finally, to calculate WER, the jiwer Python package was used.

## 3.2. Results and Discussion

In Table 1, we present the evaluation of our proposed Context-Aware Interleaved Batching (WhisperX+IB) against standard WhisperX and the sequential openai/whisper baseline on the Earnings-21 corpus. The results reveal that our method effectively bridges the performance gap caused by isolated batch processing. Although the sequential openai/whisper model with preceding text achieves a high F1 score (67.91) and a leading LPS of (2.9), it suffers from a high WER (12.0) and lacks parallelisation, operating at a 1.0× baseline speed. By re-enabling autoregressive context across chunk boundaries, WhisperX+IB matches the top LPS of (2.9), improves transcription accuracy and delivers an 8.4× speedup. Despite a slight drop in latency from the cold-start resolution pass, our method remains competitive with standard WhisperX (11.8×) while providing improved transcription structure and accuracy.

Table 1: Evaluation ofperformance metrics (reported as percentages) for transcription variants. Comparison of WER, Pn Score, F1 and Speed on 15 transcripts from a long-form speech corpus. The arrows (↑/↓) indicate the preferred direction for each metric. Best performing values are highlighted in bold.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Condition on Prev. Text</td><td colspan="5"></td></tr><tr><td>WER↓</td><td>Pn Score ↑</td><td>F1↑</td><td>LPS ↑</td><td>Speed ↑</td></tr><tr><td>WhisperX+IB (ours)</td><td>√</td><td>8.2</td><td>79.3</td><td>67.4</td><td>2.9</td><td>8.4×</td></tr><tr><td>WhisperX [4]</td><td>X</td><td>8.4</td><td>76.5</td><td>67.2</td><td>2.8</td><td>11.8×</td></tr><tr><td rowspan="2">openai/whisper [1]</td><td>V</td><td>12.0</td><td>72.0</td><td>67.9</td><td>2.9</td><td>1.0×</td></tr><tr><td>X</td><td>11.9</td><td>66.9</td><td>66.7</td><td>2.8</td><td>1.0×</td></tr></table>

We can observe from the Earnings-21 results that openai/whisper outperforms our method in punctuation quality. WhisperX sets without timestamps=True for pure transcription (and therefore so does WhisperX+IB). The Whisper decoder is trained on two target formats controlled by a single conditioning token, one with interleaved timestamp tokens, and one without citeradford2022. We hypothesise that the timestamp-conditioned mode learns a tighter association between full stops and the tagged segment boundaries. Therefore, decoding without them may worsen the full stop placement and explain Whisper’s higher F1 despite trailing on WER and Pn Score.

Table 2: Evaluation oftranscription variants on medical YouTube lectures (reported as percentages). Comparison of WER and Pn Score, evaluated on a manually verified ground-truth dataset ofmedical terminology lesson transcripts. The arrows (↑/↓) indicate the preferred direction for each metric. Best performing values are highlighted in bold. Batch size 4 settingsfor WhisperX and WhisperX+IB.
<table><tr><td>Variant</td><td>WER↓</td><td>Pn Score ↑</td></tr><tr><td>WhisperX+IB (ours)</td><td>3.3</td><td>84.7</td></tr><tr><td>WhisperX</td><td>3.5</td><td>83.6</td></tr></table>

In Table 2, we report results on the custom MedicalLessons dataset to evaluate the models’ ability to handle complex, domain-specific terminology. We see that Interleaved Batching again outperforms the standard WhisperX approach. By successfully propagating preceding text context to subsequent batches, WhisperX+IB reduces the overall WER from 3.5% to 3.3%. More importantly, our proposed method yields a notable improvement in the Proper Noun (PN) Score, increasing it from 83.6% to 84.7%. This indicates that supplying contiguous textual context significantly aids the model in correctly predicting challenging medical vocabulary and technical terms that span across VAD-segmented chunk boundaries, a capability that is otherwise lost during isolated batched inference.

Table 3 illustrates the crucial role of context injection in improving transcription accuracy. When operating in isolation, the baseline WhisperX model hallucinates the phrase “post-2007.” By contrast, the WhisperX+IB variant successfully leverages preceding dialogue to contextually ground its output, eliminating the hallucination and accurately transcribing the phrase “upholstery fabric segment.”

## 3.3. Limitations

While Interleaved Batching is highly effective for long-form audio, an inherent limitation arises when the total number of VAD-segmented chunks (N) is less than or equal to the batch

Table 3: Qualitative comparison demonstrating the impact of context injection. Without access to historical context, the isolated WhisperX model hallucinates the phrase “post-2007”. In contrast, the WhisperX+IB variant successfully leverages the preceding context to accurately transcribe “upholsteryfabric segment”.

## Preceding Context

“For the upholsteryfabric segment, salesfor the third quarter were \$35 million down 5.7% over the prior year, which prior year was a very strong quarter sales performance...”

## WhisperX

“Return on capitalfor the trailing 12-month periodfor the post-2007 continues to be impressive, coming in at 54%. The home accessory segment, which includes our e-commerce...”

## WhisperX+IB (With preceding context)

“Return on capitalfor the trailing 12-month periodfor the upholstery fabric segment continues to be impressive, coming in at 54%. The home accessory segment, which includes our e-commerce...”

size (B). In such cases, transcription completes in a single parallel pass, precluding the propagation of preceding text context. Restoring this context requires artificially reducing the batch size, potentially down to B = 1, which trades parallel throughput for contextual accuracy. Importantly, however, even in this fully sequential configuration (B = 1), our approach maintains a significant accuracy advantage over standard Whisper. By decoupling autoregressive text conditioning from fragile timestamp decoding, our method avoids hallucination loops and preserves state-of-the-art WER and punctuation and proper noun accuracy.

## 4. Conclusion

In this paper, we address a major challenge in high-speed audio transcription: the trade-off between processing speed and accuracy. While modern systems achieve high throughput by breaking audio into separate chunks for parallel processing, this approach isolates the segments from one another, cutting off the continuous context and degrading transcription quality. Our proposed method successfully bridges this gap. The results demonstrate how WhisperX+IB reduces errors both grammatically and punctually. Ultimately, this framework proves that inference speed does not have to come at the expense of accuracy, offering a scalable solution for high-performance, contextaware speech transcription.

## 5. References

[1] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, “Robust speech recognition via largescale weak supervision,” 2022. [Online]. Available: https: //arxiv.org/abs/2212.04356

[2] Y. Peng, J. Tian, B. Yan, D. Berrebbi, X. Chang, X. Li, J. Shi, S. Arora, W. Chen, R. Sharma, W. Zhang, Y. Sudo, M. Shakeel, J. weon Jung, S. Maiti, and S. Watanabe, “Reproducing whisperstyle training using an open-source toolkit and publicly available data,” 2023. [Online]. Available: https://arxiv.org/abs/2309.13876

[3] H. Ahn, J. Chae, Y. Park, and K. Shim, “Whisper-cd: Accurate long-form speech recognition using multi-negative contrastive decoding,” 2026. [Online]. Available: https://arxiv.org/abs/2603. 06193

[4] M. Bain, J. Huh, T. Han, and A. Zisserman, “Whisperx: Time-accurate speech transcription of long-form audio,” 2023. [Online]. Available: https://arxiv.org/abs/2303.00747

[5] L. Kurzinger, D. Winkelbauer, L. Li, T. Watzel, and G. Rigoll,¨ “Ctc-segmentation of large corpora for german end-to-end speech recognition,” in International Conference on Speech and Computer. Springer, 2020, pp. 267–278.

[6] I. Williams, A. Kannan, P. Aleksic, D. Rybach, and T. Sainath, “Contextual speech recognition in end-to-end neural network systems using beam search,” in Proc. Interspeech 2018, 2018, pp. 2227–2231.

[7] M. Jain, G. Keren, J. Mahadeokar, and Y. Saraf, “Contextual rnn-t for open domain asr,” in Interspeech, 2020. [Online]. Available: https://api.semanticscholar.org/CorpusID:219401479

[8] F.-J. Chang, J. Liu, M. H. Radfar, A. Mouchtaris, M. Omologo, A. Rastrow, and S. Kunzmann, “Context-aware transformer transducer for speech recognition,” 2021 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU), pp. 503– 510, 2021. [Online]. Available: https://api.semanticscholar.org/ CorpusID:243832733

[9] H. Bredin, R. Yin, J. M. Coria, G. Gelly, P. Korshunov, M. Lavechin, D. Fustes, H. Titeux, W. Bouaziz, and M.-P. Gill, “Pyannote. audio: neural building blocks for speaker diarization,” in ICASSP 2020-2020 IEEE International conference on acoustics, speech and signal processing (ICASSP). IEEE, 2020, pp. 7124–7128.

[10] M. Del Rio, N. Delworth, R. Westerman, M. Huang, N. Bhandari, J. Palakapilly, Q. McNamara, J. Dong, P. Zelasko, and M. Jett<sup>˙</sup> e,´ “Earnings-21: A practical benchmark for asr in the wild,” in Interspeech 2021. ISCA, Aug. 2021, pp. 3465–3469. [Online]. Available: http://dx.doi.org/10.21437/Interspeech.2021-1915

[11] A. Parulekar and P. Jyothi, “Laser: An llm-based asr scoring and evaluation rubric,” 2025. [Online]. Available: https://arxiv.org/abs/2510.07437

[12] S. Roy, “Semantic-wer: A unified metric for the evaluation of asr transcript for end usability,” 2021. [Online]. Available: https://arxiv.org/abs/2106.02016

[13] T. Baneras-Roux, S. Kumar, D. Khalil, S. Burdisso, P. Motlicek,˜ S. Liu, M. Rouvier, J. Wottawa, and R. Dufour, “Evaluation of automatic speech recognition using generative large language models,” 2026. [Online]. Available: https://arxiv.org/abs/2604. 21928