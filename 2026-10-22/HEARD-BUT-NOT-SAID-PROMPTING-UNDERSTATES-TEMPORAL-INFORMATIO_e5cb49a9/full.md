# HEARD BUT NOT SAID: PROMPTING UNDERSTATES TEMPORAL INFORMATION OF MUSICAL MOMENTS IN FROZEN LARGE AUDIO-LANGUAGE MODELS

Eunsik Shin

Music and Audio Research Group (MARG) Department of Intelligence and Information, Seoul National University esshin@snu.ac.kr

## ABSTRACT

Large audio-language models (LALMs) understand music, yet struggle to locate a described moment in time. We ask whether this reflects missing temporal information or a bottleneck in how the time span is extracted. Using a paired-query probe built from Slakh2100 stem activations, we attach a state query (e.g., “only piano and bass are playing”) and a change query (e.g., “the guitar drops out, leaving piano and bass”) to the same span and vary how much preceding audio the model is shown. Across frozen LALMs, three ways of extracting the span ask different questions: full-track prompting asking whether the model can say when; windowed yes/no scoring asking whether it verifies the moment when handed the segment, and whether it needs what came before; attention asking whether it can find the moment on its own. State queries score above baseline without context, while change queries improve with preceding context in most short-segment models; 30 s padded encoders show no comparable recovery. Prompting and attention expose almost none of this. Prompting failures thus understate what LALMs can verify, though not what they can locate, by an amount set by query type and encoder.

Index Terms— Audio-language models, music temporal grounding, training-free probing, music information retrieval

## 1. INTRODUCTION

Temporal grounding—returning the time span in a long signal that matches a natural-language description—has been studied across several domains. In video, work has progressed from early proposalbased methods [1, 2] to DETR-style span predictors [3, 4]. The audio counterpart is more recent: language-based audio moment retrieval [5] and frame-wise language-audio alignment [6] mostly target environmental sound events, leaving music comparatively unexplored. Temporal grounding in music is particularly challenging because a moment often shares instrumentation, tempo, and other properties with the rest of the track, making localization depend on both local evidence and comparison with surrounding or preceding context.

Large audio-language models (LALMs), which couple an audio encoder to an instruction-following language model [7, 8, 9, 10, 11, 12], can be natural candidates for music temporal grounding. Recent work has begun to test this in other domains: activation steering shows that general-audio LALMs carry temporal signal that prompting does not surface [13]; a trained head on internal representations localizes efficiently [14]; attention analysis finds layer-specific audio attention during timestamp generation [15] and a small set of audiospecialist heads indicates when a model actually draws on its audio input [16]. Together, these suggest that prompting may understate temporal information available within LALMs, echoing findings in language models that direct probability measurements can reveal capabilities missed by metalinguistic prompting [17, 18, 19].

On music, however, LALMs have so far been examined mainly through prompting [20], while existing probes use coarse event choices [13] or single-instrument notes [21], leaving it unclear whether models fail to identify what is playing or to detect change relative to preceding context, and how these abilities depend on encoder design. We address this with a controlled, training-free probe. From Slakh2100 instrument stem activity we build spans that can each be described as a state query judged from the moment alone or a change query that requires comparison with what came before. Both share the audio, position, and duration, so the positional baseline is identical by construction and any difference isolates the temporal computation the query demands. We then hold model, audio, and query fixed and vary how the time span is extracted—prompting for timestamps over the full track, scoring a yes/no judgment window by window under controlled preceding contexts, or reading attention over audio tokens—across seven LALMs whose encoders segment time differently. All models verify local states above chance when given the relevant segment, while change queries benefit from preceding context mainly in models with short-segment encoders; 30 s padded encoders do not benefit from added context, while their state scores decline. Meanwhile, full-track prompting and attention expose little of this signal.

This study introduces a paired-query probe that controls audio, position, duration, and chance while manipulating preceding context, and evaluates three training-free ways of extracting temporal span information from frozen LALMs over full-length multiinstrument music. The probe set, prompts, and code will be released upon publication.

## 2. RELATED WORK

Audio-language models. Audio-language models pair a pretrained audio encoder with a language model through projection and instruction tuning. Pengi [7], LTU [8], and SALMONN [9] established audio question answering; Qwen2-Audio [10] and Audio Flamingo 3 [12] scaled it with Whisper-style encoders [22], while Qwen2.5-Omni and Qwen3-Omni [11, 23] use encoders that process short fixed segments; other models like Kimi-Audio [24] and Step-Audio [25] are general-purpose, while Music Flamingo [26] and MOSS-Music [27] target music.

Temporal grounding in audio. AM-DETR defined language-based audio moment retrieval [5]; FLAM learns frame-wise languageaudio alignment [6]; SpotSound adds explicit timestamp grounding to an LALM through training [28]. Closest to us, Lin and Lee [13] revealed latent temporal signal that prompting does not surface, but in a three-event setting (chance 27.7%) without a context manipulation or encoder comparison. Bhattacharya et al. benchmark LALM temporal reasoning with confidence estimates [29], and Kulkarni et al. trace failures on environmental-sound multiple-choice questions to under-use of audio and mitigate them by attention scaling [30]; both use multiple-choice tasks on environmental sounds (the latter intervening on attention), whereas we score continuous spans in music and compare extraction methods without intervening. Reusing audio-LM internals with a trained head is efficient [14] but cannot distinguish information already present from information the head learns; our probe adds no parameters.

![](images/7b1499753da24d74988cd884a641e454dfddc0a8d8b70fc03d9544178ec4e2d6.jpg)  
Fig. 1. (a) Stem activity read from Slakh2100 MIDI. A sparse texture (only piano and bass) that begins right after a stem departure (guitar) becomes one target span with two queries sharing the same audio, so chance is identical. (b) In Exp. 2 the model sees a window that ends with the judged 2 s segment and begins C seconds earlier; at C=0 the departure lies outside the window and cannot be verified.

Temporal grounding in music. MusTBench benchmarks temporal reasoning in music LLMs and improves it by training [20]; concurrent work grounds piano MIDI at the note level [21]; music grounding by short video localizes a music moment for a video query [31]. We leave models frozen and use text queries over multi-instrument full-length music in which moments share global context.

## 3. PAIRED PROBE AND SPAN EXTRACTION

## 3.1. Probe construction

Slakh2100 [32] renders multi-instrument MIDI into audio with one stem per instrument, so we know exactly which instruments sound at every moment (Fig. 1a). We look for a sparse texture—a stretch of at least 3 s in which no more than four stems play, in a combination that occupies less than a fifth of the track—and for an instrument departure, where a stem that has been playing for at least 6 s falls silent for at least 3 s. When a sparse texture begins right after a departure, the same span can be described in two ways: a state query saying what is playing (“only piano and bass are playing”), and a change query saying what just left (“the guitar drops out, leaving piano and bass”). The change query contains the full state description plus the departure.

Pairing is what makes the two conditions comparable: both queries point to the same audio, position, and duration, so chance is the same by construction. This matters because the two kinds of moments are not distributed alike—sparse textures cluster heavily at track openings, where always predicting the start already reaches $\operatorname { R } @ 1 \ @ 0 . 5 = 0 . 1 9$ , whereas departures are spread across the track (0.01), while the evaluated models also show a preference for early positions. Pairing also reduces this positional bias, because spans near the track opening cannot follow a departure and are therefore excluded.

While the stem metadata provides precise event boundaries, we additionally filter for audibility in the rendered audio: an active stem must exceed −6 dB best-band SNR against the stem-subtracted mixture and remain within 20 dB of the track median. Fade-outs are removed and spans are capped at 30 s. After removing 221 duplicated tracks, this yields 713 pairs on 503 tracks. We hold out 20% (143 pairs, stratified by position and template) as a dev split used only to fix two post-processing parameters and select the attention variant; the remaining 570 are evaluated. Queries use 13 coarse instrument classes, are paraphrased by an LLM with both members of a pair rewritten together, and pass checks for positional or duration words, length leakage, matched length distributions, and pair integrity (change ⊇ state for all 713).

## 3.2. Span extraction experiments

Every experiment uses the same frozen model, audio, and query; only the way the span is extracted differs, each method asking a different question.

Exp. 1 — full-track prompting: can the model say when? The full track and the query are given with a fixed prompt asking for start and end seconds, and the generated text is parsed.

Exp. 2 — windowed yes/no scoring: does the model verify the moment when handed the segment, and does it need to hear what came before? We never ask for a time. We slide a window along the track and, at each position t, ask whether the final 2 s of the window match the query; instead of letting the model answer, we read the next-token logits and take $s ( t ) = \mathrm { l o g i t } [ \mathrm { y e s } ] - \mathrm { l o g i t } [ \mathrm { n o } ]$ The model scores candidate segments rather than predicting their timestamps. The window $[ t - C , t + 2 ]$ always ends with the judged 2 s segment and begins C seconds earlier (Fig. 1b). The preceding context $C \in \{ 0 , 2 , 4 , 8 \}$ s is the one variable we manipulate. With C=0, the model hears only the judged segment, so the departure in a change query cannot be verified—only its residual state content can. As C gets longer, the departure enters the window. Whether change queries recover with C thus tests directly whether the model compares the segment with what preceded it. Scores are z-normalized per (track, query) across windows, removing each query’s overall tendency to say yes. A question-free variant that instead scores the log-probability of the query text is examined in Sec. 5.3.

Exp. 3 — full-track attention: without any window scaffolding, does the modelfind the moment on its own? Exp. 3 asks whether the model can identify the relevant moment internally when given the full track, even if it cannot verbalize the timestamp. We read attention from the final prompt token to the audio tokens, averaging over the last third of layers and all heads. Layer, source position, and normalization were fixed in advance, and only models with a positive dev in-span attention excess (CI excluding 0) were evaluated.

From score curves to spans. Exp. 2 and Exp. 3 produce temporal score curves rather than span boundaries. For standard evaluation, we convert each curve to a span by expanding from its maximum while the score remains within a tolerance δ, with a minimum duration; both parameters are fixed on the dev set. Because this heuristic can penalize short targets even when their location is correct, we also report an oracle-duration diagnostic that fixes the predicted duration to the ground-truth duration and evaluates localization alone.

## 4. EXPERIMENTAL SETUP

Models. We evaluate seven open-weight LALMs: Qwen2-Audio-7B [10], Qwen2.5-Omni-3B [11], Qwen3-Omni-30B-A3B [23], MOSS-Audio-4B [33], MOSS-Music-8B [27], Audio Flamingo 3 [12], and Music Flamingo [26]. Selection required open weights, a known audio-token-to-time mapping, and full-track input in one pass (Qwen2-Audio’s 8k context precludes Exp. 1 and 3), and within these constraints varied the encoder’s segment length. In four models, encoder self-attention is confined to a short window of the input: 2 s blocks in Qwen2.5-Omni, and 1 s (Qwen3-Omni) and 4 s (MOSS) windows, which we set through the encoders’ inference-window parameters. The other three use Whisper-style encoders that pad every input to 30 s with absolute positional embeddings inherited from Whisper’s 30 s training windows [22], so a long track is encoded as a sequence of independent 30 s frames. The two MOSS models share an encoder and an input pipeline that interleaves time-marker tokens with the audio tokens; MOSS-Music adds music-specific continual pretraining and instruction tuning.

Data. All experiments use the same $5 7 0$ evaluation pairs (1,140 queries); Exp. 2 uses a 1 s hop for each C. Padded-encoder models score the truncated windows at a track’s very start spuriously high (47–65% of argmaxes in the first 8 s at $C { = } 8 .$ , vs. 5–18% otherwise), so windows whose judged segment begins at $t < 8 \mathrm { s }$ are excluded for all models and $C ,$ chosen after inspecting the argmax distribution of the padded models. No evaluation span begins there.

Metrics and chance. We report R@1@0.3 for the standard spans, where the minimum-duration constraint can limit IoU, and R@1@0.5 for oracle-duration spans, where this boundary effect is removed. We use a best constant span baseline rather than random span—the single normalized center and length that maximizes R@1 over the evaluation set, i.e., what a model achieves by always guessing the most common location (0.147 at IoU 0.3, 0.098 at 0.5, and 0.100 for oracle-duration).

Controls. A high score could arise without the model actually matching this query to this audio: it might respond to the track alone, regardless of query; it might produce a peak for any query, because normalized scores are relative; or it might answer from position without listening. One control rules out each. Query shuffle scores a track with the curve of a different query from the same track—if performance survives, the model reacts to the track, not the query. Cross-track uses a query that describes nothing in this track— performance should fall to chance and the raw yes/no score should drop, showing that the score means something in absolute terms. Audio shuffle permutes the track in 2 s pieces before Exp. 1 and 3—if the answer does not change, it came from a positional prior, not the audio (Exp. 2 scores windows independently, so it does not apply).

Statistics. To test whether differences across query types and context conditions are reliable, all comparisons are paired on the same span: McNemar’s test on discordant pairs for R@1, with a bootstrap CI that resamples tracks rather than pairs, since pairs on one track are not independent. Comparisons across C and state–change gaps are Holm-corrected, and whether a single cell exceeds chance uses the same track-resampled 95% CI.

## 5. RESULTS

## 5.1. State queries can be verified

Handed the segment alone $( C { = } 0 )$ , every model places state queries above the baseline through windowed scoring under both standard and oracle-duration evaluation (Table 1, Fig. 2). In every model the state query also beats the change query on the very same spans (Holm $p \ < \ 0 . 0 3 )$ Four models also place change queries above chance under oracle duration at $C { = } 0 ,$ consistent with their embedded state description, but none reaches the state level; under standard post-processing, none has a CI over chance. The controls confirm a genuine query–audio match: a shuffled or cross-track query drops performance to near zero, and a cross-track query’s raw yes/no score is significantly lower than the true query’s in all but three of the results.

## 5.2. Change queries need preceding context

Whether a change query becomes answerable once the model hears what came before differs by model, and the difference follows the encoder. In three of the four short-segment models, change queries rise with C (Fig. 2, left): Qwen2.5-Omni, MOSS-Audio, and above all Qwen3-Omni, whose change queries go from near chance at $C { = } 0$ to the level of its state queries at $C { = } 8$ (within-condition $p =$ $2 \times 1 0 ^ { - 1 9 }$ ; gap n.s., Holm $p ~ = ~ 0 . 9 2 )$ The gap also closes for MOSS-Audio $( p = 0 . 8 0 )$ and shrinks to a third for Qwen2.5-Omni $( p = 0 . 0 0 3 )$ . Although state scores decline mildly with C in two of the three, change improves significantly in all three, and these are the only models whose $C = 8$ change scores clear chance at IoU 0.3. Qwen3-Omni shows the strongest recovery; its larger scale is consistent with this pattern, although scale is confounded with encoder design. MOSS-Music is the exception among short-segment models with its change score rising only slightly and non-significantly $( p = 0 . 0 9 )$ , while its state score declines, accounting for most of the gap reduction. Because MOSS-Music shares its encoder with MOSS-Audio, this divergence shows that the encoder does not fully determine context use: training can suppress a recovery that the encoder permits.

The three remaining models share Whisper-style encoders that pad every input to 30 s and, despite different language models and training, behave alike (Fig. 2, right): change queries show no meaningful recovery, state queries degrade toward chance as $C$ grows (Audio Flamingo 3 loses half its state score), and at IoU 0.3 all three fall below chance for both conditions at $C { = } 8 .$ (State–change gap at $C { = } 8 \colon$ Holm $p = 0 . 5 4 , 0 . 9 0 , 0 . 0 1 3 .$ ) One possible explanation is that padding hides where the audio ends: at $C { = } 0$ the audio is just the judged 2 s, but as $C$ grows the model must isolate those 2 s inside a 30 s frame, so its judgment drifts toward the whole window. This shared pattern across differently trained models suggests an encoderrelated effect, although the evidence remains correlational and may not be causal.

## 5.3. Prompting and attention struggle to expose the signal

Prompting. Asked for timestamps directly, all models evaluated in Exp. 1 are at or below chance, and on the same pairs windowed scoring is far better. The prompted answers are largely independent of the audio: most models return a constant that depends only on the query’s shape (a track-opening span for state, a mid-track one for change), and for four models most answers survive audio shuffling unchanged. The exception is MOSS-Music: its answers do change under shuffling and their centers land inside the true span four times more often than random, but its spans are about one second long, so IoU still fails.

Scoring variants. Two variants of Exp. 2 clarify what the logit scoring preserves. Verbalizing each window-level judgment as yes/no sharply degrades performance—halving state and pushing change below chance—by discarding the ranking carried by the yes–no logit margin. Removing the judgment prompt and scoring only the query also stays at chance across models, so the useful signal lies in the graded margin of an explicit but ungenerated judgment.

![](images/1df65088458ef556b0098e0a60ad43255ffd2067f7c9ed9d2289120f13be7030.jpg)

Fig. 2. Exp. 2, oracle-duration R@1@0.5 versus preceding context C for state (blue) and change (red) queries on the 570 evaluation pairs. Change queries recover with context in three of the four short-segment models (left); the three 30 s padded models (right) degrade as C grows regardless of training.
<table><tr><td rowspan="3">Model</td><td rowspan="3">Encoder segment</td><td colspan="4">Exp. 2, standard R@1@0.3</td><td colspan="4">Exp. 2, oracle-duration R@1@0.5</td><td colspan="2">Exp. 1 R@1@0.5</td><td>Exp. 3</td></tr><tr><td colspan="2">State</td><td colspan="2">Change</td><td colspan="2">State</td><td colspan="2">Change</td><td colspan="2"></td><td>attn. excess</td></tr><tr><td>C=0</td><td>C=8</td><td>C=0</td><td>Č=8</td><td>C=0</td><td>C=8</td><td>C=0</td><td>C=8</td><td>state</td><td>change</td><td>state / change</td></tr><tr><td>Qwen2.5-Omni-3B</td><td>Whisper-v3 (block attn.)</td><td>0.221‡</td><td>0.240</td><td>0.105</td><td>0.219‡</td><td>0.279</td><td>0.219</td><td>0.119</td><td>0.165‡</td><td>0.011</td><td>0.011</td><td>+0.041 /+0.020*</td></tr><tr><td>Qwen3-Omni-30B-A3B</td><td>AuT</td><td>0.247</td><td>0.298‡</td><td>0.102</td><td>0.312</td><td>0.242</td><td>0.265</td><td>0.118</td><td>0.268</td><td>0.042</td><td>0.025</td><td>+0.044 /+0.062*</td></tr><tr><td>MOSS-Audio-4B</td><td>MOSS Audio Encoder †</td><td>0.232</td><td>0.237‡</td><td>0.163</td><td>0.233</td><td>0.256</td><td>0.204‡</td><td>0.160‡</td><td>0.211‡</td><td>0.030</td><td>0.019</td><td>n.s.</td></tr><tr><td>MOSS-Music-8B</td><td>MOSS Audio Encoder †</td><td>0.223</td><td>0.184</td><td>0.139</td><td>0.170</td><td>0.209</td><td>0.142</td><td>0.160</td><td>0.168‡</td><td>0.072</td><td>0.042</td><td>n.s.</td></tr><tr><td>Qwen2-Audio-7B</td><td>Whisper-v3 (30 s pad.)</td><td>0.181‡</td><td>0.125</td><td>0.096</td><td>0.111</td><td>0.165</td><td>0.096</td><td>0.075</td><td>0.088</td><td></td><td></td><td></td></tr><tr><td>Audio Flamingo 3 (7B)</td><td>AF-Whisper (30 s pad.)</td><td>0.230‡</td><td>0.142</td><td>0.147</td><td>0.107</td><td>0.226</td><td>0.107</td><td>0.156</td><td>0.104</td><td>0.007</td><td>0.004</td><td>n.s.</td></tr><tr><td>Music Flamingo (7B)</td><td>AF-Whisper (30 s pad.)</td><td>0.214</td><td>0.135</td><td>0.140</td><td>0.114</td><td>0.232</td><td>0.135</td><td>0.184‡</td><td>0.095</td><td>0.012</td><td>0.009</td><td>n.s.</td></tr><tr><td>Chance (best constant span)</td><td></td><td>0.147</td><td>0.147</td><td>0.147</td><td>0.147</td><td>0.100</td><td>0.100</td><td>0.100</td><td>0.100</td><td>0.098</td><td>0.098</td><td>0</td></tr></table>

Table 1. All results on the 570 evaluation pairs. Exp. 2: R@1@0.3 with standard post-processing and R@1@0.5 with oracle duration at C=0/8; Exp. 1: R@1@0.5; Exp. 3: span-specific in-span attention excess (z-units). <sup>‡</sup>95% track-bootstrap CI lower bound exceeds chance (2,000 resamples, uncorrected; no Exp. 1 cell qualifies). <sup>∗</sup>CI excludes 0 for both conditions; n.s.: CI includes 0; <sup>†</sup>Input pipeline interleaves 2 s time markers.

Attention. We measure the in-span attention excess—mean normalized attention inside the true span minus outside, in z-units— and its span-specific version, the excess for the target query minus that for another query on the same track. Only the two Qwen-Omni models show significant span-specific excess, but even there span localization remains below chance. Under audio shuffling the attention follows the content to its new location rather than staying at the original position, indicating local acoustic matching rather than reliable track-level localization. This is consistent with Exp. 2: verifying a segment that is explicitly presented does not imply that the model can locate the same segment within the full track. The model may hear the segment when handed it, yet fail to find where it was heard; Exp. 2 supplies the search, whereas Exp. 3 asks the model to perform it.

## 6. DISCUSSION AND LIMITATIONS

Windowed scoring reveals a consistent distinction between being able to verify a candidate segment and being able to locate it in a full track. Across models, local states are recoverable when the relevant audio is explicitly presented, while verifying the change itself benefits from preceding context. Full-track prompting and attention expose much less of this signal, suggesting that temporal grounding failures in LALMs are not explained solely by a simple absence of temporal information, but also by how that information is accessed and searched over time.

The context results further suggest that this access is shaped, though not fully determined, by the audio encoder. Short-segment encoders are generally able to benefit from additional context, whereas others show a shared degradation pattern despite differences in language model and training. Music-specific fine-tuning does not consistently alter this behavior, which suggests that encoder-side temporal organization may matter alongside model scale and domain specialization.

These findings should be interpreted within the scope of the probe. Slakh2100 provides precise stem activity, but it is MIDIrendered and our queries are restricted to instrumentation, so state and change capture only which instruments are present or disappear, not other temporally meaningful musical properties such as harmony, dynamics, rhythm, or vocal structure. Extending the probe to real recordings with vocals, for example using MUSDB18, would therefore test whether the same patterns hold under less controlled but more realistic acoustic conditions.

The paired design makes the comparisons well controlled, but the absolute scores remain modest across methods. This leaves substantial room for work that targets music temporal grounding more directly, including richer query types, more realistic recordings, and methods that bridge the gap between local verification and full-track localization.

## 7. CONCLUSION

We asked whether frozen large audio-language models fail at music temporal grounding because temporal information is absent, or because standard prompting does not expose it. Using a trainingfree paired probe built from Slakh2100 and evaluated on seven openweight LALMs, we find that models can verify local musical states and, in several short-segment models, context-dependent changes even when full-track prompting and attention fail to localize them reliably. These results suggest a distinction between verifying that a candidate segment matches a query and locating that segment within a full track. Prompted timestamp failure therefore does not fully reflect the temporal information accessible to these models, while reliable full-track localization remains limited.

## 8. REFERENCES

[1] Jiyang Gao, Chen Sun, Zhenheng Yang, and Ram Nevatia, “TALL: Temporal activity localization via language query,” in Proc. IEEE ICCV, 2017.

[2] Lisa Anne Hendricks, Oliver Wang, Eli Shechtman, Josef Sivic, Trevor Darrell, and Bryan Russell, “Localizing moments in video with natural language,” in Proc. IEEE ICCV, 2017.

[3] Jie Lei, Tamara L. Berg, and Mohit Bansal, “QVHighlights: Detecting moments and highlights in videos via natural language queries,” in Proc. NeurIPS, 2021.

[4] WonJun Moon, Sangeek Hyun, SangUk Park, Dongchan Park, and Jae-Pil Heo, “Query-dependent video representation for moment retrieval and highlight detection,” in Proc. IEEE/CVF CVPR, 2023.

[5] Hokuto Munakata, Taichi Nishimura, Shota Nakada, and Tatsuya Komatsu, “Language-based audio moment retrieval,” in Proc. IEEE ICASSP, 2025.

[6] Yusong Wu, Christos Tsirigotis, Ke Chen, Cheng-Zhi Anna Huang, Aaron Courville, Oriol Nieto, Prem Seetharaman, and Justin Salamon, “FLAM: Frame-wise language-audio modeling,” in Proc. ICML, 2025.

[7] Soham Deshmukh, Benjamin Elizalde, Rita Singh, and Huaming Wang, “Pengi: An audio language model for audio tasks,” in Proc. NeurIPS, 2023.

[8] Yuan Gong, Hongyin Luo, Alexander H. Liu, Leonid Karlinsky, and James Glass, “Listen, think, and understand,” in Proc. ICLR, 2024.

[9] Changli Tang, Wenyi Yu, Guangzhi Sun, Xianzhao Chen, Tian Tan, Wei Li, Lu Lu, Zejun Ma, and Chao Zhang, “SALMONN: Towards generic hearing abilities for large language models,” in Proc. ICLR, 2024.

[10] Yunfei Chu, Jin Xu, Qian Yang, et al., “Qwen2-Audio technical report,” arXiv preprint arXiv:2407.10759, 2024.

[11] Jin Xu, Zhifang Guo, Jinzheng He, et al., “Qwen2.5-Omni technical report,” arXiv preprint arXiv:2503.20215, 2025.

[12] Arushi Goel, Sreyan Ghosh, Jaehyeon Kim, et al., “Audio flamingo 3: Advancing audio intelligence with fully open large audio language models,” in Proc. NeurIPS, 2025.

[13] Tsung-En Lin and Hung-Yi Lee, “Steering where to listen: Instruction-based activation steering redirects temporal attention in large audio-language models,” arXiv preprint arXiv:2606.11400, 2026.

[14] Joseph An, Phillip Keung, Jiaqi Wang, Orevaoghene Ahia, and Noah A. Smith, “Encode once, decode never: Reusing audio LM internals for efficient temporal localization,” arXiv preprint arXiv:2602.10230, 2026.

[15] Jihoon Jeong, Pooneh Mousavi, Mirco Ravanelli, and Cem Subakan, “Listen first, then answer: Timestamp-grounded speech reasoning,” arXiv preprint arXiv:2603.19468, 2026.

[16] Neta Glazer, Lenny Aharon, and Ethan Fetaya, “Are audiolanguage models listening? audio-specialist heads for adaptive audio steering,” arXiv preprint arXiv:2603.06854, 2026.

[17] Jennifer Hu and Roger Levy, “Prompting is not a substitute for probability measurements in large language models,” in Proc. EMNLP, 2023.

[18] Zorik Gekhman, Eyal Ben David, Hadas Orgad, Eran Ofek, Yonatan Belinkov, Idan Szpektor, Jonathan Herzig, and Roi Reichart, “Inside-out: Hidden factual knowledge in LLMs,” arXiv preprint arXiv:2503.15299, 2025.

[19] Hadas Orgad, Michael Toker, Zorik Gekhman, Roi Reichart, Idan Szpektor, Hadas Kotek, and Yonatan Belinkov, “LLMs know more than they show: On the intrinsic representation of LLM hallucinations,” in Proc. ICLR, 2025.

[20] Daeyong Kwon, Qiyu Wu, Shinobu Kuriya, Junghyun Koo, Shuyang Cui, Zhi Zhong, Wei-Hsiang Liao, Hiromi Wakaki, and Yuki Mitsufuji, “MusTBench: Benchmarking and advancing temporal grounding in music LLMs,” arXiv preprint arXiv:2605.29300, 2026.

[21] Kun Fang, Ziyu Wang, and Ichiro Fujinaga, “What are you listening to? temporal music grounding for audio-to-text large language models,” arXiv preprint arXiv:2608.29480, 2026.

[22] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever, “Robust speech recognition via large-scale weak supervision,” in Proc. ICML, 2023.

[23] Jin Xu, Zhifang Guo, Hangrui Hu, et al., “Qwen3-Omni technical report,” arXiv preprint arXiv:2509.17765, 2025.

[24] Kimi Team, “Kimi-Audio technical report,” arXiv preprint arXiv:2504.18425, 2025.

[25] Ailin Huang et al., “Step-Audio: Unified understanding and generation in intelligent speech interaction,” arXiv preprint arXiv:2502.11946, 2025.

[26] Sreyan Ghosh, Arushi Goel, Lasha Koroshinadze, et al., “Music flamingo: Scaling music understanding in audio language models,” in Proc. ICLR, 2026.

[27] OpenMOSS, “MOSS-Music: A music-specialized audio language model,” https://github.com/OpenMOSS/ MOSS-Music, 2026.

[28] Luoyi Sun, Xiao Zhou, Zeqian Li, Ya Zhang, Yanfeng Wang, and Weidi Xie, “SpotSound: Enhancing large audio-language models with fine-grained temporal grounding,” in Proc. ACM Multimedia, 2026.

[29] Debarpan Bhattacharya, Apoorva Kulkarni, and Sriram Ganapathy, “Benchmarking and confidence evaluation of LALMs for temporal reasoning,” in Proc. Interspeech, 2025, pp. 2068– 2072.

[30] Apoorva Kulkarni, Kaousheik Jayakumar, Sreyan Ghosh, Sarah Wiegreffe, Dinesh Manocha, and Ramani Duraiswami, “A closer look at failure modes in temporal understanding of large audio-language models,” in Proc. Interspeech, 2026.

[31] Zijie Xin, Minquan Wang, Jingyu Liu, Quan Chen, Ye Ma, Peng Jiang, and Xirong Li, “Music grounding by short video,” in Proc. IEEE/CVF ICCV, 2025, pp. 22285–22293.

[32] Ethan Manilow, Gordon Wichern, Prem Seetharaman, and Jonathan Le Roux, “Cutting music source separation some Slakh: A dataset to study the impact of training data quality and quantity,” in Proc. IEEE WASPAA, 2019.

[33] Chen Yang et al., “MOSS-Audio technical report,” arXiv preprint arXiv:2606.01802, 2026.