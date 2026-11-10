# Text Grouping Changes Completion-Time Comparisons in Long-Form Simultaneous Speech Translation

Hieu Minh Truong and Dung D. Le

VinUniversity, Vietnam {hieu.tm2,dung.ld}@vinuni.edu.vn

## Abstract

Does grouping translation updates before speech synthesis change the measured completion gap between two translators? We compare StreamAtt with its fixed-audio-history baseline on ten English–German talks. For each translation, we keep the intended text and its availability times fixed and vary how updates are grouped for synthesis. Four-second grouping increases StreamAtt’s mean completion lead from 106.6 to 229.1 seconds with XTTS-v2, but reduces it from 169.5 to 146.5 seconds with Piper. Both produce shorter audio; different changes in playback idle account for the oppo site effects. These directions recur on a separately selected twenty-talk panel. In continuing CosyVoice 3 sessions, grouping instead leaves paired audio unchanged and delays completion on nineteen of twenty blocks. A speech-timing comparison therefore needs to specify how text enters synthesis, not just which synthesizer is used. Recognition and automatic word timing describe the outputs, but do not establish translation accuracy or intelligibility.

## 1 Introduction

A speech translator can produce text early and still finish speaking late. Simultaneous translation is commonly evaluated by relating text-emission delay to translation quality (Ma et al., 2019, 2020; Papi et al., 2022, 2023, 2024). Spoken output adds synthesis and serial playback: committed text must become audio, and that audio must be played in order. Work on simultaneous speech-to-speech translation therefore also studies initial waiting, synthesized duration, and output completion (Zheng et al., 2020; Liu et al., 2022; Kano et al., 2023).

Using the same text-to-speech (TTS) model for two translators does not make their synthesis inputs comparable. One translator may emit many short updates, while another emits fewer, longer ones. The system can synthesize each update separately or group several updates first. Grouping changes both text context and request boundaries; the resulting audio can differ in duration even when the intended words do not change.

Related work. Liu et al. (2022) study the tradeoff between initial waiting and synthesized duration. Related evaluation work examines sensitivity to segmentation, computation accounting, and aggregation (Papi et al., 2025; Xu et al., 2025; Polák et al., 2026). Long-form speech-output evaluation uses recognition and alignment to examine generated speech (Xue et al., 2026), while systems differ in how translated text enters synthesis (Djanibekov et al., 2026). Our question is not whether grouping can shorten one system’s playback. It is whether grouping changes the completion gap between two translators.

We compare StreamAtt’s attention-based audiohistory selection with its fixed-audio-history baseline, using the same translation checkpoint (Papi et al., 2024). We save each strategy’s German text updates and availability times, then replay each stream under different grouping rules. Text stays fixed across grouping conditions within a strategy, not across the two strategies. This control separates the grouping intervention from changes to the saved translation stream.

Four-second grouping widens StreamAtt’s mean completion lead with XTTS-v2 but narrows it with Piper. A separate twenty-talk panel reproduces both directions. Duration and playback idle explain where the completion changes occur, but do not isolate an architectural cause. The contribution is this controlled comparison and its implication for evaluation: reporting a TTS model without its textgrouping rule leaves part of the timing comparison unspecified.

<table><tr><td>TTS</td><td>Translator</td><td>Wait (s)</td><td>First playback</td><td>Audio D</td><td>Idle I</td><td>Offset L</td><td>ASR edits (%)</td></tr><tr><td>XTTS-v2</td><td>StreamAtt</td><td>0</td><td>17.44</td><td>1080.36</td><td>19.21</td><td>468.01</td><td>32.89</td></tr><tr><td></td><td>StreamAtt</td><td>2</td><td>19.82</td><td>834.02</td><td>30.05</td><td>232.52</td><td>26.72</td></tr><tr><td></td><td>StreamAtt</td><td>4</td><td>22.42</td><td>723.36</td><td>41.24</td><td>133.04</td><td>22.45</td></tr><tr><td></td><td>Baseline</td><td>0</td><td>17.44</td><td>1159.30</td><td>46.85</td><td>574.59</td><td>30.90</td></tr><tr><td></td><td>Baseline</td><td>2</td><td>19.81</td><td>989.54</td><td>64.74</td><td>422.72</td><td>24.87</td></tr><tr><td></td><td>Baseline</td><td>4</td><td>22.34</td><td>919.94</td><td>73.75</td><td>362.13</td><td>24.16</td></tr><tr><td>Piper</td><td>StreamAtt</td><td>0</td><td>16.85</td><td>562.51</td><td>90.14</td><td>21.09</td><td>16.45</td></tr><tr><td></td><td>StreamAtt</td><td>4</td><td>21.09</td><td>488.80</td><td>150.89</td><td>8.13</td><td>13.94</td></tr><tr><td></td><td>Baseline</td><td>0</td><td>16.86</td><td>692.98</td><td>129.14</td><td>190.56</td><td>17.66</td></tr><tr><td></td><td>Baseline</td><td>4</td><td>21.09</td><td>642.35</td><td>143.80</td><td>154.58</td><td>16.15</td></tr></table>

Table 1: Exploratory full-talk results. Means give each of the ten talks equal weight; all 100 outputs are complete. Times are seconds, and mean source duration is 631.56 seconds. The ASR column compares recognized with intended words, retaining punctuation. Normalized error components appear in Appendix D. Compare grouping waits within a TTS configuration: hardware, voice, and frontend differ between backends.

## 2 Study Design

Saved streams and controls. We separate translation from synthesis. For each talk and strategy, the saved stream contains updates $( x _ { i } , a _ { i } )$ : newly committed text and its availability time. A grouping rule combines updates into synthesis input; an input-limit splitter may divide that input into several requests. Within a backend, changing the rule preserves the stream, voice, inference settings, and playback rule. Audio duration, synthesis cost, and playback timing are outcomes. Across backends, request text is shared at matched timer settings, but voice, frontend, and hardware differ.

Talks and translators. The exploratory panel contains ten English–German MuST-C talks (Di Gangi et al., 2019), totaling 6,315.6 seconds. They were selected by metadata and inspected before the grouping study; all are retained. A second panel contains twenty additional talks selected before new model outputs, with both histories and immediate/four-second synthesis under both backends. We keep the panels separate.

Both translators use the released StreamAtt checkpoint (Papi et al., 2023, 2024). StreamAtt (A) retains source audio using attention; the baseline (F) removes audio according to discarded texthistory words. On the exploratory panel, StreamAtt emits 4,064 updates containing 14,430 whitespace words; the baseline emits 2,924 updates containing 19,823 words. Both have a median update length of three words. The baseline’s 37.4% greater text length is not evidence of better or worse translation. Settings and source lists are in Appendices A–B.

Grouping. Immediate dispatch (w = 0) sends each update separately. With grouping wait w, the oldest pending update starts a deadline at $a _ { i } + w .$ Updates available by that deadline are joined with one space; later arrivals do not reset the timer. Translation end releases any pending text early. A busy synthesizer delays execution, not group membership, so w is not a bound on total latency. We also test punctuation- and word-count rules with a two-second cap (Appendix C).

Synthesis and playback. XTTS-v2 uses Ana Florence (Casanova et al., 2024); Piper uses the German voice de\_DE-thorsten-high (Kim et al., 2021; OHF-Voice, n.d.; Piper Voice Contributors, n.d.). The original comparison has sixty XTTS outputs at waits 0, 2, and 4 seconds and forty Piper outputs at 0 and 4 seconds. Four seconds was chosen before synthesis; two seconds was a shorter check, not a score-selected setting.

Each request runs separately. Readiness includes measured preparation, inference, and transfer; loading, warmup, logging, and file storage are excluded. Playback is serial at normal speed, without trimming in the primary analysis. We replay saved translation times rather than rerun translation alongside synthesis. XTTS runs on a GPU and Piper on a CPU; this is not a matched-hardware speed comparison.

Continuing synthesis. A separate CosyVoice 3 experiment compares immediate and four-second text feeds within one continuing session per block (Du et al., 2025; FunAudioLLM, n.d.). Twenty paired blocks yield forty sessions. State resets between blocks, not feeds. This tests the feed schedule within a block, not full-talk translator ordering (Appendix F).

<table><tr><td>Panel</td><td>TTS</td><td>w (s)</td><td>Gap δ(0)</td><td>Gap δ(w)</td><td>Change η(w)</td><td>95% interval for η</td></tr><tr><td rowspan="3">Exploratory (10)</td><td>XTTS-v2</td><td>2</td><td>-106.58</td><td>-190.21</td><td>-83.63</td><td>[−124.22, -44.66]</td></tr><tr><td>XTTS-v2</td><td>4</td><td>-106.58</td><td>-229.08</td><td>-122.50</td><td>[-180.03, -69.92]</td></tr><tr><td>Piper</td><td>4</td><td>-169.46</td><td>-146.46</td><td>+23.01</td><td>[7.36, 40.99]</td></tr><tr><td rowspan="2">Confirmation (20)</td><td>XTTS-v2</td><td>4</td><td>-118.63</td><td>-224.92</td><td>-106.29</td><td>[-148.74, -67.53]</td></tr><tr><td>Piper</td><td>4</td><td>-118.23</td><td>-84.12</td><td>+34.11</td><td>[17.52, 55.17]</td></tr></table>

Table 2: Grouping changes the between-translator completion gap. Times are seconds; gaps and changes are rounded separately. Negative δ means StreamAtt finishes earlier. Negative η widens its mean lead; positive η narrows it in these comparisons. Intervals resample paired talks within each panel. The mean translator ordering does not reverse.

## 3 Measurements

Completion and its components. Primary timing uses all 100 exploratory outputs and all 160 confirmation outputs, without recognition or alignment filtering. Additional policy and seed comparisons are reported separately. Time zero is source start. An audio part $j$ becomes ready at $r _ { j } .$ , has sample-derived duration $d _ { j }$ , and finishes playback at

$$
C _ { j } = \operatorname* { m a x } ( r _ { j } , C _ { j - 1 } ) + d _ { j } , \qquad C _ { 0 } = 0 .\tag{1}
$$

For m parts, let $\begin{array} { r l r } { D } & { { } = } & { \sum _ { j } d _ { j } } \end{array}$ and $I \quad =$ $\begin{array} { r } { \sum _ { j } \operatorname* { m a x } ( 0 , r _ { j } - C _ { j - 1 } ) } \end{array}$ . With source duration S, the final offset is

$$
L = C _ { m } - S = D + I - S .\tag{2}
$$

Thus $L$ measures completion relative to source end. Silence inside generated audio belongs to $D ;$ time when playback waits for audio, including the initial wait, belongs to I (Liu et al., 2022). Shorter audio need not reduce L if later readiness creates additional idle.

For talk s, we measure the completion gap and its change under grouping:

$$
\delta _ { s } ( w ) = L _ { s } ( \mathsf { A } , w ) - L _ { s } ( \mathsf { F } , w ) ,\tag{3}
$$

$$
\eta _ { s } ( w ) = \delta _ { s } ( w ) - \delta _ { s } ( 0 ) .\tag{4}
$$

Negative δ favors StreamAtt. We calculate each backend separately, giving talks equal weight. Descriptive 95% intervals use 10,000 paired talkbootstrap draws (Koehn, 2004), retaining all conditions of each talk together. They describe source variation within the sampled panel, not new synthesis realizations.

Content and word timing. Whisper-large-v3- turbo transcribes the original 100 outputs (Radford et al., 2022; OpenAI, n.d.); Qwen3-ASR independently transcribes their immediate/four-second conditions (Shi et al., 2026). Word edit rates compare recognition with intended synthesis text. We report the punctuation-retaining rates in Table 1 and normalized error components in Appendix D. These are not source-reference translation scores.

![](images/9b08cb92192a31de477f5d1d50cf7f1e9abc112f38fa7b8a4eff0a09b9002f86.jpg)  
Figure 1: Four-second grouping effects on individual exploratory talks. Negative η shifts the completion gap toward StreamAtt; positive η shifts it toward the baseline. Each backend uses the same ten talks.

For XTTS, a separate forced aligner estimates when recognized intended words finish playing (Qwen Team, 2026). The comparison uses the same unambiguous, validly timed occurrences across all three waits. It covers only a selected subset and does not determine eligibility for completion measurements (Appendix E).

## 4 Results

The completion gap changes in opposite directions. On the exploratory panel, four-second grouping increases StreamAtt’s mean lead from 106.58 to 229.08 seconds with XTTS, but reduces it from 169.46 to 146.46 seconds with Piper (Table 2). The XTTS effect is negative on nine of ten talks; the Piper effect is positive on seven (Figure 1). StreamAtt leads on eight XTTS talks without grouping and all ten with grouping; it leads on all ten Piper talks at both waits. The mean ordering

does not reverse.

Separate confirmation. On the twenty-talk panel, grouping widens StreamAtt’s mean lead from 118.63 to 224.92 seconds with XTTS (η(4) = −106.29) and narrows it from 118.23 to 84.12 seconds with Piper (η(4) = +34.11). The exploratory directions recur on 17/20 and 19/20 talks, respectively. Appendix B gives the source list, Piper’s individual exceptions, and the duration–idle decomposition.

Why shorter audio can narrow a lead. Grouping shortens StreamAtt’s XTTS audio by 357.00 seconds while adding 22.03 seconds of idle. Most of that saving therefore advances completion. With Piper, StreamAtt saves more audio than the baseline—73.71 versus 50.63 seconds—but adds much more idle: 60.75 versus 14.66 seconds. It finishes 12.96 seconds earlier, while the baseline finishes 35.97 seconds earlier. The baseline gains more at the endpoint, so StreamAtt’s lead narrows. This accounting explains where the seconds go, not an architectural cause. On baseline talk 805, changing XTTS grouping from two to four seconds saves 33.408 seconds of audio and adds exactly the same idle: completion is unchanged.

Other grouping rules and synthesis runs. The punctuation- and word-count rules retain the foursecond effect’s direction relative to immediate dispatch. Among non-immediate rules, the mean completion-gap span is 47.34 seconds for XTTS and 13.57 seconds for Piper. A punctuation rule differs from a timer with the same two-second cap by +8.47 and −1.91 seconds, respectively. Paired intervals are [−0.83, 18.97] and [−3.18, −0.72] seconds; these direct comparisons were chosen after inspecting the aggregate results. Across three XTTS seeds, η(4) ranges from −127.6 to −116.8 seconds. Removing request-edge silence reduces the effects’ magnitudes but preserves their signs (Appendix C). Across the original run and two additional Piper realizations, the mean interaction ranges from +23.01 to +23.99 seconds; all three means are positive (Appendix C.4).

Starting earlier and finishing earlier differ. Four-second grouping delays mean first playback by about five seconds with XTTS and four seconds with Piper. It does not advance every endpoint. Figure 2 shows the separate change in readyaudio queue. In continuing CosyVoice sessions, paired waveform samples are identical: grouping instead delays mean completion by 1.49 seconds for StreamAtt blocks and 2.19 seconds for baseline blocks. Nineteen pairs finish later; one finishes 0.038 seconds earlier. Backend, voice, and block context also differ, so this does not isolate persistence alone.

Content behind the timing changes. Both recognizers report lower mean total error under grouping, but more XTTS deletion mismatches. Whisper’s StreamAtt deletions rise from 612 to 1,313; its insertions fall from 1,954 to 360. These are recognition errors, not verified acoustic omissions. Detected non-speech accounts for only 38.75–43.43% of StreamAtt’s XTTS duration reduction. Without human judgments, we cannot determine how much of the shorter audio reflects fewer repetitions, faster speech, or omitted content. The baseline also produces 37.4% more intended words; this difference does not establish greater source coverage or translation accuracy.

## 5 Conclusion

Changing how fixed translation updates enter synthesis changes the measured completion gap between translators. The direction differs between XTTS and Piper and recurs on a separate source panel. Evaluations should report grouping and end-of-stream release rules, actual request sizes, audio duration, playback idle, and first playback alongside final completion. These timing differences should not be treated as translation-quality gains: the automatic diagnostics do not establish preserved content.

## Limitations

The study covers English–German talks, two audiohistory strategies from one checkpoint, and one voice per backend. The twenty-talk confirmation does not establish generalization across languages, domains, or voices. XTTS and Piper differ in frontend, implementation, hardware, and voice, preventing architecture-only attribution. Bufferedpolicy comparisons were selected after inspecting aggregate means; their exploratory intervals do not establish equivalence.

Intended text is fixed within each stream, not across translators, while grouping changes both context and request boundaries. Production replay excludes feedback and device contention. Edgesilence replay holds readiness fixed, and CosyVoice uses restarted short blocks rather than continuous talks.

Without human listening or bilingual adequacy judgments, recognition and alignment remain indirect diagnostics. XTTS timing excludes 7.52% of alignment units; its validity rule was revised after a timestamp failure without human calibration. The unverified common-word subset covers roughly half of intended words. Source-reference quality and semantic delay remain unreported due to unverified reference coverage and acoustic offsets. For example, talk 805 has annotations ending at 386.66 seconds, but audio continues to 594.55 seconds; the remainder is not assumed silent.

## Ethical Considerations

Earlier completion does not necessarily imply accurate or complete translation. Our experiments use existing research recordings and publicly released synthesis voices. No new participant recordings or human evaluations were collected. All datasets, models, and voices remain subject to their respective usage conditions.

Use of AI tools. The authors used generative AI tools only for code debugging and minor language editing and to improve readability. These tools were not used to generate any scientific content, experimental results, data analyses, or conclusions.

## Acknowledgments

This research was supported by the VinUniversity Cross-College Research Grant (Grant ID: VUNI.2324.CC06).

## References

Edresson Casanova, Kelly Davis, Eren Gölge, Görkem Göknar, Iulian Gulea, Logan Hart, Aya Aljafari, Joshua Meyer, Reuben Morais, Samuel Olayemi, and Julian Weber. 2024. Xtts: a massively multilingual zero-shot text-to-speech model. Preprint, arXiv:2406.04904.

Mattia A. Di Gangi, Roldano Cattoni, Luisa Bentivogli, Matteo Negri, and Marco Turchi. 2019. MuST-C: a Multilingual Speech Translation Corpus. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 2012–2017, Minneapolis, Minnesota. Association for Computational Linguistics.

Amirbek Djanibekov, Luisa Bentivogli, Matteo Negri, and Sara Papi. 2026. Simulu: Training-free policy for

long-form simultaneous speech-to-speech translation. Preprint, arXiv:2603.16924.

Zhihao Du, Changfeng Gao, Yuxuan Wang, Fan Yu, Tianyu Zhao, Hao Wang, Xiang Lv, Hui Wang, Chongjia Ni, Xian Shi, Keyu An, Guanrou Yang, Yabin Li, Yanni Chen, Zhifu Gao, Qian Chen, Yue Gu, Mengzhe Chen, Yafeng Chen, and 3 others. 2025. Cosyvoice 3: Towards in-the-wild speech generation via scaling-up and post-training. Preprint, arXiv:2505.17589.

FunAudioLLM. n.d. Fun-CosyVoice3-0.5B-2512. Model card and base checkpoint. Revision used: 29e01c4e.

Yasumasa Kano, Katsuhito Sudoh, and Satoshi Nakamura. 2023. Average token delay: A latency metric for simultaneous translation. Preprint, arXiv:2211.13173.

Jaehyeon Kim, Jungil Kong, and Juhee Son. 2021. Conditional variational autoencoder with adversarial learning for end-to-end text-to-speech. Preprint, arXiv:2106.06103.

Philipp Koehn. 2004. Statistical significance tests for machine translation evaluation. In Proceedings of the 2004 Conference on Empirical Methods in Natural Language Processing, pages 388–395, Barcelona, Spain. Association for Computational Linguistics.

Danni Liu, Changhan Wang, Hongyu Gong, Xutai Ma, Yun Tang, and Juan Pino. 2022. From start to finish: Latency reduction strategies for incremental speech synthesis in simultaneous speech-to-speech translation. Preprint, arXiv:2110.08214.

Mingbo Ma, Liang Huang, Hao Xiong, Renjie Zheng, Kaibo Liu, Baigong Zheng, Chuanqiang Zhang, Zhongjun He, Hairong Liu, Xing Li, Hua Wu, and Haifeng Wang. 2019. STACL: Simultaneous translation with implicit anticipation and controllable latency using prefix-to-prefix framework. In Proceedings ofthe 57th Annual Meeting ofthe Associationfor Computational Linguistics, pages 3025–3036, Florence, Italy. Association for Computational Linguistics.

Xutai Ma, Mohammad Javad Dousti, Changhan Wang, Jiatao Gu, and Juan Pino. 2020. SIMULEVAL: An evaluation toolkit for simultaneous translation. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 144–150, Online. Association for Computational Linguistics.

OHF-Voice. n.d. Piper: Local neural text-to-speech engine. Experiment uses package version 1.4.2; full identity retained in the evidence record.

OpenAI. n.d. Whisper large-v3-turbo. Model card. The checkpoint-specific documentation supplements the Whisper paper.

Myle Ott, Sergey Edunov, Alexei Baevski, Angela Fan, Sam Gross, Nathan Ng, David Grangier, and Michael Auli. 2019. fairseq: A fast, extensible toolkit for sequence modeling. Preprint, arXiv:1904.01038.

Sara Papi, Marco Gaido, Matteo Negri, and Luisa Bentivogli. 2024. StreamAtt: Direct streaming speech-totext translation with attention-based audio history selection. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 3692–3707, Bangkok, Thailand. Association for Computational Linguistics.

Sara Papi, Marco Gaido, Matteo Negri, and Marco Turchi. 2022. Over-generation cannot be rewarded: Length-adaptive average lagging for simultaneous speech translation. In Proceedings ofthe Third Workshop on Automatic Simultaneous Translation, pages 12–17, Online. Association for Computational Linguistics.

Sara Papi, Peter Polák, Dominik Machácek, and Ondˇ ˇrej Bojar. 2025. How “real” is your real-time simultaneous speech-to-text translation system? Transactions of the Association for Computational Linguistics, 13:281–313.

Sara Papi, Marco Turchi, and Matteo Negri. 2023. AlignAtt: Using Attention-based Audio-Translation Alignments as a Guide for Simultaneous Speech Translation. In Interspeech 2023, pages 3974–3978.

Piper Voice Contributors. n.d. de\_DE-thorsten-high: German voice model card. Pinned voice assets and model-card bytes retained in the experiment.

Peter Polák, Sara Papi, Luisa Bentivogli, and Ondˇrej Bojar. 2026. Better late than never: Meta-evaluation of latency metrics for simultaneous speech-to-text translation. Preprint, arXiv:2509.17349.

Qwen Team. 2026. Qwen3-ForcedAligner-0.6B: Model card and inference software. Standalone forcedalignment model. Documentation accessed 6 October 2026.

Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. 2022. Robust speech recognition via large-scale weak supervision. Preprint, arXiv:2212.04356.

Xian Shi, Xiong Wang, Zhifang Guo, Yongqi Wang, Pei Zhang, Xinyu Zhang, Zishan Guo, Hongkun Hao, Yu Xi, Baosong Yang, Jin Xu, Jingren Zhou, and Junyang Lin. 2026. Qwen3-asr technical report. Preprint, arXiv:2601.21337.

Silero Team. 2024. Silero VAD: Pre-trained enterprisegrade voice activity detector, number detector and language classifier. Software repository.

Xi Xu, Wenda Xu, Siqi Ouyang, and Lei Li. 2025. CA\*: Addressing evaluation pitfalls in computation-aware latency for simultaneous speech translation. In Findings ofthe Associationfor Computational Linguistics: NAACL 2025, pages 7077–7082, Albuquerque, New Mexico. Association for Computational Linguistics.

Yulin Xue, Siqi Ouyang, and Lei Li. 2026. A practical evaluation method for long-form simultaneous speech-to-speech translation. In Proceedings ofthe 23rd International Conference on Spoken Language Translation (IWSLT 2026), pages 32–39, San Diego, USA (in-person and online). Association for Computational Linguistics.

Renjie Zheng, Mingbo Ma, Baigong Zheng, Kaibo Liu, Jiahong Yuan, Kenneth Church, and Liang Huang. 2020. Fluent and low-latency simultaneous speechto-speech translation with self-adaptive training. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2020, pages 3928–3937, Online. Association for Computational Linguistics.

## A Data and Reproduction Details

The exploratory panel contains two development, two tst-COMMON, and six tst-HE talks (Table 3). The same ten sources provide the original 100 fulltalk outputs and twenty CosyVoice blocks. History variants, grouping settings, and blocks are not additional independent sources. Selection used metadata and prior-exposure exclusions; the grouping study followed inspection of the resulting translations.

Translation. Both strategies use one-second source packets, a commitment setting that withholds a suffix aligned to the most recent f = 6 encoder frames, attention-layer index 3, and twenty words of textual history. The baseline removes 280 ms of source audio for each discarded history word; it does not reset translation at fixed time intervals. The FBK-fairseq implementation (Ott et al., 2019; Papi et al., 2023, 2024) uses main checkout 4dde0ea4 and the authors’ fixed-audio-history class from 248135e0. The twenty exploratory streams contain 6,988 updates and 34,253 whitespace words. Their original availability times are replayed, so upstream computation is not charged again.

XTTS. XTTS-v2 uses the stock Ana Florence voice. The original run sets seed 5242 before each request, with temperature 0.75, length penalty 1.0, repetition penalty 10, top-k 50, top-p 0.85, one beam, sampling enabled, and speed 1.0. The internal sentence splitter is disabled. An external splitter preserves text while respecting 253 characters and the 402-token limit. Inference runs on a partitioned NVIDIA H100 (3g.40gb, 40 GB).

Thirty original full-talk outputs were reused and thirty newly synthesized. Reusing compatible requests preserves their samples and measured synthesis time; that time is included in every replay that uses the request. The reused and new outputs were collected in separate runs. Six predetermined requests reproduced their waveforms on three invocations with the same seed; service-time standard deviations were 0.010–0.056 seconds. This requestlevel repeatability check is distinct from the fulltalk, different-seed comparison in Appendix C.

<table><tr><td>Talk</td><td>Split</td><td>Audio (s)</td></tr><tr><td>805</td><td>Dev</td><td>594.547</td></tr><tr><td>1102</td><td>Common</td><td>421.396</td></tr><tr><td>792</td><td>Dev</td><td>545.901</td></tr><tr><td>2017</td><td>HE</td><td>556.536</td></tr><tr><td>2007</td><td>HE</td><td>551.335</td></tr><tr><td>1939</td><td>HE</td><td>626.521</td></tr><tr><td>2102</td><td>HE</td><td>726.251</td></tr><tr><td>1381</td><td>Common</td><td>731.057</td></tr><tr><td>1997</td><td>HE</td><td>853.264</td></tr><tr><td>1922</td><td>HE</td><td>708.789</td></tr></table>

Table 3: The ten exploratory MuST-C source recordings. Common and HE denote tst-COMMON and tst-HE. Audio duration is not a manually verified speech-duration measurement.

Piper. Piper 1.4.2 uses the single-speaker 22,050- Hz de\_DE-thorsten-high voice (OHF-Voice, n.d.; Piper Voice Contributors, n.d.). ONNX runs on the CPU with four intra-operation threads, one inter-operation thread, and sequential execution. Length scale is 1.0, noise scale 0.667, noisewidth scale 0.8, normalization is enabled, and volume is 1.0. The original immediate/four-second panel contains 9,377 requests and 10,759 internal sentence outputs. Seventeen punctuation-only requests generate no audio but still incur service cost.

Every Piper request is synthesized afresh. Attempted seed resets did not reproduce sample counts or waveform samples, so audio reuse is disabled. The original full-talk panels contain one realization per condition; Appendix C.4 reports the separate repetition comparison. Four exploratory conditions for talk 792 were collected earlier than the remaining thirty-six, under unchanged settings; the fixed counterbalanced source/condition order is retained.

Timing and uncertainty. Request readiness includes frontend processing, inference, and output transfer. Loading, warmup, logging, and file storage are excluded. Playback starts a part only after both its readiness and the preceding part’s completion. Full-talk endpoint intervals use 10,000 paired talk-bootstrap draws, seed 1032026, and the recorded order-statistic convention. All conditions of a talk remain together; the exploratory and confirmation panels are not pooled. The intervals are descriptive and are not adjusted for multiple comparisons. Word-timing intervals use their separately recorded convention (Appendix E).

Records. The numerical supplement contains translation-release schedules, request and audiopart clocks, completion flags, scoring inputs, software/model revisions, and source-level results. It reconstructs comparisons from saved records; it does not distribute private waveforms or model weights. Incomplete exploratory configurations and stopping records are kept separately.

## B Source-Level Results and Confirmation

Tables 4 and 5 give every exploratory source’s completion contrasts. The XTTS table retains the twosecond condition and the individual ordering reversals. Piper’s table includes the duration and idle components rather than reporting only a mean interaction.

## B.1 Confirmation panel

Twenty additional English–German MuST-C talks were selected before new model outputs using seed 20261011. The intended split allocation of four development, four tst-COMMON, and twelve tst-HE talks was changed before synthesis to four, ten, and six, because only six eligible tst-HE talks remained after excluding the exploratory sources. All selected talks were retained. The panel totals 4.55 hours and spans 251.2–1772.1 seconds. Forty complete translation streams were frozen before synthesis. Both histories and immediate/four-second conditions under both backends give 160 full outputs.

With XTTS, the mean gap changes from −118.63 to −224.92 seconds: η(4) = −106.29, with interval [−148.74, −67.53]. Seventeen of twenty interactions are negative. The duration component is −101.77 seconds and the idle component is −4.52 seconds, so most of the interaction comes from the difference in audio-duration savings (Table 7). With Piper, the gap changes from −118.23 to −84.12 seconds: η(4) = +34.11, with interval [17.52, 55.17]. Nineteen interactions are positive; the median is +18.58 seconds. Both backends retain a mean completion advantage for StreamAtt.

Table 6 preserves every Piper confirmation source, including the two talks that favor the baseline at both waits. Talk 1131 has a particularly large positive completion gap; it is retained rather than excluded from the mean. The table describes the timing exception, not its translation or acoustic cause.

<table><tr><td>Talk</td><td>δ(0)</td><td>δ(2)</td><td>δ(4)</td><td>η(2)</td><td>η(4)</td></tr><tr><td>805</td><td>-100.752</td><td>-175.063</td><td>-175.050</td><td>-74.310</td><td>-74.298</td></tr><tr><td>1102</td><td>-73.747</td><td>-188.574</td><td>-233.912</td><td>-114.827</td><td>-160.165</td></tr><tr><td>792</td><td>+76.843</td><td>-66.650</td><td>-118.590</td><td>-143.493</td><td>-195.433</td></tr><tr><td>2017</td><td>-189.004</td><td>-211.120</td><td>-238.417</td><td>-22.116</td><td>-49.413</td></tr><tr><td>2007</td><td>-87.943</td><td>-90.048</td><td>-121.696</td><td>-2.105</td><td>-33.753</td></tr><tr><td>1939</td><td>-147.164</td><td>-247.972</td><td>-286.235</td><td>-100.808</td><td>-139.071</td></tr><tr><td>2102</td><td>-179.145</td><td>-293.291</td><td>-326.248</td><td>-114.146</td><td>-147.103</td></tr><tr><td>1381</td><td>-174.269</td><td>-165.853</td><td>-162.750</td><td>+8.416</td><td>+11.519</td></tr><tr><td>1997</td><td>+64.410</td><td>-142.197</td><td>-247.929</td><td>-206.607</td><td>-312.339</td></tr><tr><td>1922</td><td>-255.040</td><td>-321.316</td><td>-380.021</td><td>-66.276</td><td>-124.981</td></tr></table>

Table 4: All completion gaps and their changes under batching, in seconds. Differences are StreamAtt minus baseline. Negative gaps favor StreamAtt. Gaps and changes are rounded separately.
<table><tr><td>Talk</td><td>δ(0)</td><td>δ(4)</td><td> $\eta ( 4 )$ </td><td>Duration part</td><td>Idle part</td><td>XTTS η(4)</td></tr><tr><td>792</td><td>-30.77</td><td>-30.69</td><td>+0.08</td><td>-30.02</td><td>+30.11</td><td>-195.43</td></tr><tr><td>805</td><td>-168.54</td><td>-169.29</td><td>-0.75</td><td>-15.59</td><td>+14.84</td><td>-74.30</td></tr><tr><td>1102</td><td>-236.94</td><td>-237.01</td><td>-0.07</td><td>-34.96</td><td>+34.88</td><td>-160.17</td></tr><tr><td>1381</td><td>-115.51</td><td>-107.62</td><td>+7.89</td><td>-6.16</td><td>+14.06</td><td>+11.52</td></tr><tr><td>1922</td><td>-232.68</td><td>-172.73</td><td>+59.95</td><td>-12.54</td><td>+72.49</td><td>-124.98</td></tr><tr><td>1939</td><td>-197.30</td><td>-166.47</td><td>+30.83</td><td>-32.37</td><td>+63.20</td><td>-139.07</td></tr><tr><td>1997</td><td>-268.48</td><td>-268.62</td><td>-0.14</td><td>-56.25</td><td>+56.11</td><td>-312.34</td></tr><tr><td>2007</td><td>-71.65</td><td>-65.46</td><td>+6.19</td><td>-20.68</td><td>+26.86</td><td>-33.75</td></tr><tr><td>2017</td><td>-153.70</td><td>-92.73</td><td>+60.97</td><td>-3.13</td><td>+64.11</td><td>-49.41</td></tr><tr><td>2102</td><td>-219.07</td><td>-153.94</td><td>+65.12</td><td>-19.09</td><td>+84.21</td><td>-147.10</td></tr></table>

Table 5: All source-level Piper contrasts, in seconds. δ is StreamAtt minus baseline completion; $\eta = \delta ( 4 ) - \delta ( 0 )$ The duration and idle parts sum to η up to rounding. Positive η narrows StreamAtt’s lead. The last column preserves the XTTS contrast on the same source rather than pooling the backends.

## C Grouping Rules and Timing Sensitivities

## C.1 Practical grouping rules

All rules act on the same saved streams and preserve the text sequence. The two- and four-second timers start with the oldest pending update. The punctuation rule sends a group once it contains at least three whitespace words and ends in a period, question mark, or exclamation mark (optionally followed by a closing quote or bracket), or after two seconds. The word-count rule sends a group after eight accumulated whitespace words or two seconds. Updates available at the trigger join the group; later arrivals do not reset its timer. Translation end releases any remaining text. Input-limit splitting follows grouping and preserves the joined string.

Every non-immediate rule in Table 8 retains the sign found with the four-second timer relative to immediate dispatch. That does not mean the rules have equal effects. The range of mean gaps among the four non-immediate rules is 47.34 seconds for XTTS and 13.57 seconds for Piper.

To compare two buffered policies directly, subtract their mean completion gaps. Punctuation minus the two-second timer gives +8.47 seconds for XTTS and −1.91 seconds for Piper. Thus differences remain without using immediate dispatch as either member of the contrast. These point contrasts are calculated from the reported rounded means. Paired source-level comparisons are reported in Appendix C.2.

## C.2 Direct buffered-policy comparisons

The direct comparison uses the punctuation rule and a timer with the same two-second cap. Group membership, input-limit splitting, and dispatch times are reconstructed from the saved streams. A punctuation trigger accounts for 3.39% of StreamAtt groups and 2.38% of baseline groups; end-oftranslation release accounts for 0.52% and 0.63%. The remaining groups reach the timer.

We resample paired talks, retaining both histories and both policies in each draw. These direct contrasts were selected after inspecting the aggregate policy means, so the analysis is exploratory.

<table><tr><td>Talk</td><td>Split</td><td>Source (s)</td><td>δ(0)</td><td>δ(4)</td><td>η(4)</td><td>Duration part</td><td>Idle part</td></tr><tr><td>783</td><td>Dev</td><td>413.5</td><td>-72.32</td><td>-33.38</td><td>+38.94</td><td>-0.03</td><td>+38.98</td></tr><tr><td>785</td><td>Dev</td><td>1055.9</td><td>-236.45</td><td>-173.84</td><td>+62.61</td><td>-17.95</td><td>+80.56</td></tr><tr><td>769</td><td>Dev</td><td>1318.3</td><td>-308.70</td><td>-254.07</td><td>+54.64</td><td>-87.89</td><td>+142.53</td></tr><tr><td>767</td><td>Dev</td><td>1772.1</td><td>-396.07</td><td>-208.27</td><td>+187.80</td><td>-32.35</td><td>+220.15</td></tr><tr><td>1375</td><td>Common</td><td>974.3</td><td>-312.88</td><td>-252.52</td><td>+60.36</td><td>-28.89</td><td>+89.24</td></tr><tr><td>1161</td><td>Common</td><td>324.8</td><td>-70.06</td><td>-56.06</td><td>+14.00</td><td>-18.55</td><td>+32.55</td></tr><tr><td>1160</td><td>Common</td><td>471.2</td><td>-79.22</td><td>-67.61</td><td>+11.61</td><td>-0.67</td><td>+12.28</td></tr><tr><td>1165</td><td>Common</td><td>553.6</td><td>-108.91</td><td>-98.22</td><td>+10.69</td><td>-12.52</td><td>+23.21</td></tr><tr><td>1378</td><td>Common</td><td>1421.0</td><td>-199.40</td><td>-179.89</td><td>+19.51</td><td>+18.12</td><td>+1.39</td></tr><tr><td>1171</td><td>Common</td><td>251.2</td><td>-18.27</td><td>-17.69</td><td>+0.58</td><td>-5.20</td><td>+5.78</td></tr><tr><td>1346</td><td>Common</td><td>1150.2</td><td>-99.70</td><td>-53.98</td><td>+45.73</td><td>-31.76</td><td>+77.49</td></tr><tr><td>1353</td><td>Common</td><td>381.8</td><td>+49.34</td><td>+77.30</td><td>+27.96</td><td>-3.53</td><td>+31.49</td></tr><tr><td>1144</td><td>Common</td><td>335.8</td><td>-57.95</td><td>-45.54</td><td>+12.41</td><td>-6.93</td><td>+19.34</td></tr><tr><td>1131</td><td>Common</td><td>1041.4</td><td>+1289.27</td><td>+1268.34</td><td>-20.94</td><td>-32.84</td><td>+11.91</td></tr><tr><td>2183</td><td>HE</td><td>1089.1</td><td>-780.99</td><td>-780.83</td><td>+0.16</td><td>-103.10</td><td>+103.26</td></tr><tr><td>2045</td><td>HE</td><td>1117.3</td><td>-443.71</td><td>-355.86</td><td>+87.84</td><td>-17.45</td><td>+105.29</td></tr><tr><td>1932</td><td>HE</td><td>562.2</td><td>-93.46</td><td>-65.97</td><td>+27.50</td><td>-7.04</td><td>+34.53</td></tr><tr><td>2024</td><td>HE</td><td>553.8</td><td>-44.98</td><td>-27.33</td><td>+17.65</td><td>-22.33</td><td>+39.98</td></tr><tr><td>1954</td><td>HE</td><td>996.3</td><td>-311.95</td><td>-298.30</td><td>+13.65</td><td>-70.25</td><td>+83.90</td></tr><tr><td>1961</td><td>HE</td><td>599.2</td><td>-68.11</td><td>-58.62</td><td>+9.49</td><td>-42.30</td><td>+51.79</td></tr><tr><td>Mean</td><td></td><td></td><td>-118.23</td><td>-84.12</td><td>+34.11</td><td>-26.17</td><td>+60.28</td></tr></table>

Table 6: The twenty-talk confirmation panel and all Piper source-level contrasts. Common and HE denote tst-COMMON and tst-HE. All quantities except talk ID and split are seconds. Negative δ favors StreamAtt; positive η shifts the gap toward the baseline. Components sum to η up to rounding.

<table><tr><td>TTS</td><td>Duration part</td><td>Idle part</td><td>η(4)</td></tr><tr><td>XTTS-v2</td><td>-101.77</td><td>-4.52</td><td>-106.29</td></tr><tr><td>Piper</td><td>-26.17</td><td>+60.28</td><td>+34.11</td></tr></table>

Table 7: Confirmation-set mean interaction components, in seconds. Duration and idle account for different parts of the opposite-signed effects.

The XTTS paired interval is [−0.83, 18.97] seconds, with a positive difference on 6/10 talks; the Piper interval is [−3.18, −0.72], with 2/10 positive talks. Most groups reach the common timer cap, so this contrast compares two similar dispatch rules. It does not establish that all reasonable buffered policies are interchangeable.

## C.3 XTTS synthesis seeds

The full-talk comparison uses seeds 5242, 5243, and 5244 on the ten exploratory talks, both histories, and immediate/four-second grouping. The original seed-5242 outputs are retained; all nonseed settings remain fixed. The three seed-level mean effects are negative, ranging from −127.6 to −116.8 seconds. The original effect is −122.50 seconds. The seed is reset before each synthesis request; seed-level records are retained in the numerical supplement. This checks the sign and scale of the observed XTTS interaction, not every possi-

ble synthesis realization.

## C.4 Piper realization variability

The comparison combines forty archived Piper outputs with eighty newly synthesized outputs: the ten exploratory talks, both histories, and immediate/four-second grouping in each of three realizations. Voice, inference settings, and CPU model are unchanged. The two new runs use counterbalanced condition orders; the original run is retained as a historical realization rather than treated as counterbalanced. Realization identifiers do not imply controlled or independent random seeds. All completed outcomes are retained.

The mean interaction is +23.01 seconds in the original run and +23.71 and +23.99 seconds in the two new runs. The respective numbers of talks with positive interactions are 7/10, 8/10, and 9/10. The two new runs have 95% paired-talk bootstrap intervals of [7.97, 41.65] and [8.86, 40.94] seconds. The average narrowing direction therefore recurs in both repetitions.

For each talk, we compute the sample standard deviation of its interaction across the three realizations. The median of these standard deviations is 0.96 seconds and the maximum is 6.37 seconds. Talk 1381 has the largest range, from +7.89 to +20.16 seconds. Some near-zero source effects change sign across runs. These repetitions describe variability on the same ten talks; they are not additional independent sources or evidence of equivalence between realizations.

<table><tr><td>TTS</td><td>Grouping rule</td><td> $\delta ( 0 )$ </td><td> $\delta ( p )$ </td><td> $\eta ( p )$ </td></tr><tr><td rowspan="4">XTTS-v2</td><td>2-second timer</td><td>-106.58</td><td>-190.21</td><td>-83.63</td></tr><tr><td>Punctuation / 2-second cap</td><td>-106.58</td><td>-181.74</td><td>-75.16</td></tr><tr><td>8 words / 2-second cap</td><td>-106.58</td><td>-188.35</td><td>-81.77</td></tr><tr><td>4-second timer</td><td>-106.58</td><td>-229.08</td><td>-122.50</td></tr><tr><td rowspan="4">Piper</td><td>2-second timer</td><td>-169.46</td><td>-158.12</td><td>+11.34</td></tr><tr><td>Punctuation / 2-second cap</td><td>-169.46</td><td>-160.03</td><td>+9.43</td></tr><tr><td>8 words / 2-second cap</td><td>-169.46</td><td>-154.67</td><td>+14.79</td></tr><tr><td>4-second timer</td><td>-169.46</td><td>-146.46</td><td>+23.01</td></tr></table>

Table 8: Exploratory mean completion gaps under four grouping rules, in seconds. Immediate dispatch is the reference for $\eta ( p ) = \delta ( p ) - \delta ( 0 )$ . Comparisons between buffered rules use differences of $\delta ( p )$ instead.

<table><tr><td>TTS</td><td>Mean (s)</td><td></td><td>95% interval Positive talks</td></tr><tr><td>XTTS</td><td>+8.47</td><td>[-0.83, 18.97]</td><td>6/10</td></tr><tr><td>Piper</td><td>-1.91</td><td>[−3.18, −0.72]</td><td>2/10</td></tr></table>

Table 9: Punctuation-minus-two-second-timer completion-gap contrasts. Point estimates follow from the policy means in Table 8. Intervals and direction counts use paired source contrasts.

## C.5 Request-edge silence

The edge-silence replay keeps each request’s measured readiness and changes only its sample duration. Silero probabilities (Silero Team, 2024) are evaluated at thresholds 0.3, 0.5, and 0.7. Leading and trailing material outside a 100-ms speech guard is removed; interior pauses and request order are unchanged. Playback is recomputed from the modified durations.

Across thresholds, the interaction ranges from −96.37 to −87.58 seconds for XTTS and from +16.94 to +19.72 seconds for Piper (Table 10). At threshold 0.5, the effects are −91.76 and +18.43 seconds, respectively. Removing the detected request edges reduces the magnitude but preserves the negative XTTS and positive Piper directions. Because readiness is held fixed, this is a retrospective sensitivity analysis, not a timed online trimming implementation or a content-preservation test.

## C.6 Piper readiness checks

In the original panel, Piper grouping shortens all twenty waveforms, but advances completion on only eight StreamAtt and seven baseline talks. The mean interaction is +23.01 seconds, median +7.04, and range [−0.75, 65.12]. The duration and idle components are −23.08 and +46.09 seconds.

<table><tr><td>Threshold</td><td>XTTS η(4)</td><td>Piper η(4)</td></tr><tr><td>0.3</td><td>-96.37</td><td>+19.72</td></tr><tr><td>0.5</td><td>-91.76</td><td>+18.43</td></tr><tr><td>0.7</td><td>-87.58</td><td>+16.94</td></tr></table>

Table 10: Mean grouping effects after request-edge silence removal, in seconds. The primary analysis uses untrimmed audio.

A sentence-release replay makes each internal sentence available at its recorded readiness, instead of waiting for the whole request. A zero-service replay removes downstream computation time. Both retain the same waveform samples, request grouping, and original text times. Their interactions are +23.22 and +23.50 seconds, respectively. Neither completed-request buffering nor measured CPU service alone explains the positive sign. Sentencelevel audio release is not a different rule for grouping text before synthesis.

The source-paired difference between Piper’s and XTTS’s four-second interactions is +145.51 seconds, with interval [90.83, 198.50]. Nine sources are positive; talk 1381 is the exception. This comparison includes voice, frontend, implementation, and device differences.

## C.7 Duration savings and the final endpoint

For nonnegative readiness times, repeated substitution in Equation 1 gives

$$
C _ { m } = \operatorname* { m a x } _ { 1 \leq k \leq m } \left( r _ { k } + \sum _ { j = k } ^ { m } d _ { j } \right) .\tag{5}
$$

Each candidate combines one readiness event with the audio that follows it. A maximizing k identifies the event that limits completion. Duration saved before that event can become idle rather than advance the endpoint. The baseline talk-805 example in Section 4 shows this cancellation with the reported durations. The identity describes playback accounting; it does not predict audio duration from text or isolate an acoustic mechanism.

<table><tr><td>Update statistic</td><td>StreamAtt</td><td>Baseline</td></tr><tr><td>Number of updates</td><td>4064</td><td>2924</td></tr><tr><td>Whitespace words</td><td>14430</td><td>19823</td></tr><tr><td>Median words</td><td>3</td><td>3</td></tr><tr><td>90th-percentile words</td><td>6</td><td>10</td></tr><tr><td>At most two words (%)</td><td>43.36</td><td>40.97</td></tr><tr><td>Final punctuation (%)</td><td>1.77</td><td>1.44</td></tr></table>

Table 11: Text-update lengths and punctuation, pooled over ten talks per system. An update can be split into several synthesis requests. Terminal punctuation does not label a prosodic boundary.
<table><tr><td>History</td><td>0 s</td><td>2 s</td><td>4 s</td></tr><tr><td>StreamAtt</td><td>4,068</td><td>1,894</td><td>1,149</td></tr><tr><td>Baseline</td><td>3,064</td><td>1,586</td><td>1,096</td></tr></table>

Table 12: Actual XTTS request occurrences across ten recordings. Reused requests count once per use; these are not counts of unique model invocations. Input-limit splitting can create more than one request per update.

## D Request Structure and Content Diagnostics

## D.1 Updates and requests

Whitespace-word counts include standalone punctuation and music symbols. An update is not necessarily one synthesis request: input-limit splitting can create several requests. Table 11 describes the text streams; Table 12 counts actual XTTS request occurrences after splitting. Reused requests are counted and timed each time their audio enters a replay. A text-assembly check recovers the stored canonical text in all sixty streams across the exploratory and confirmation panels. Packet-toupdate agreement is also verified for all forty confirmation streams; the corresponding check is unavailable for the twenty exploratory streams. These checks concern stream assembly, not translation adequacy, and do not explain the baseline’s greater intended-text length.

Four-second grouping removes 2,919 StreamAtt requests and 1,968 baseline requests. Total waveform savings divided by removed requests are 1.223 and 1.216 seconds, respectively; the zeroto-two-second ratios are 1.133 and 1.149. These are descriptive ratios, not a fixed cost per boundary: context, punctuation, request length, and restarts change together.

## D.2 Recognition and scoring

Whisper receives 16-kHz audio, a German transcription prefix, and no intended text. Four prefix positions leave up to 444 new decoder positions. Capacity-limited fifteen-second windows are split into 7.5-second halves and, when needed, 3.75- second quarters. Transcripts are not selected by reference score or completed from intended text.

The Table 1 word metric lowercases whitespace tokens but retains punctuation. The normalized breakdown lowercases each token, retains Unicode letters/numbers and ASCII apostrophes, and discards tokens that become empty. Counts are pooled for the error breakdown, whereas mean rates give talks equal weight. Neither statistic measures adequacy against the English source.

Normalized Whisper mean word edit rates decrease from 23.05% to 14.87% for StreamAtt– XTTS and from 22.38% to 17.73% for baseline– XTTS. Piper’s rates decrease from 7.65% to 6.60% and from 9.92% to 9.23%. However, XTTS deletions increase on nine StreamAtt talks and six baseline talks. Normalized total error worsens on StreamAtt talk 1102 and baseline talk 2007. Table 13 retains these adverse error components; a deletion mismatch is not a verified omitted word.

Full-output Qwen recognition. Qwen3-ASR-0.6B (Shi et al., 2026) independently transcribes the exploratory immediate/four-second outputs without the intended text. Audio preparation, German language specification, tokenization, and scoring normalization are fixed across conditions. Table 14 reports mean word edit rates and pooled deletion counts. Like Whisper, Qwen reports lower mean total error in all four backend–history comparisons but higher XTTS deletion counts. Agreement between recognizers reduces dependence on one recognizer; it does not establish what listeners hear.

Selected-clip comparison. An earlier inputselected XTTS sample contains forty immediate/grouped pairs, with 15–45 lexical words per block and matching complete request spans. Selection seed 20261007 gives eighty clip IDs covering seventy-three distinct stored waveforms, including seven identical-audio pairs. Whisper and Qwen independently transcribe disjoint 7.5-second windows, with German specified and no intended text; all 160 clip–recognizer results complete. Both report fewer insertions in this sample, while StreamAtt deletions decrease and baseline deletions increase slightly. Table 15 gives the pooled error counts. Different selection and recognition context mean this clip result does not replace the full-output deletion increase.

<table><tr><td>Backend</td><td>History</td><td>Intended words</td><td>Substitutions</td><td>Insertions</td><td>Deletions</td></tr><tr><td>XTTS-v2</td><td>StreamAtt</td><td>14,336</td><td> $7 6 5  4 4 6$ </td><td> $1 , 9 5 4  3 6 0$ </td><td> $6 1 2  1 , 3 1 3$ </td></tr><tr><td></td><td>Baseline</td><td>19,766</td><td> $9 6 8 \to 6 0 2$ </td><td> $1 , 4 5 2  4 5 8$ </td><td> $2 , 0 3 8  2 , 4 4 1$ </td></tr><tr><td>Piper</td><td>StreamAtt</td><td>14,336</td><td> $7 8 4  6 4 0$ </td><td> $1 5 4 \to 1 4 4$ </td><td> $1 6 3  1 6 8$ </td></tr><tr><td></td><td>Baseline</td><td>19,766</td><td> $1 , 1 0 8  1 , 0 0 9$ </td><td> $2 3 1  2 5 5$ </td><td> $5 0 6  4 3 2$ </td></tr></table>

Table 13: Pooled punctuation-normalized edits across all ten recordings, immediate → four-second grouping. Intended words are fixed within a history across waits and backends. Counts are pooled for the error breakdown; mean percentages elsewhere weight recordings equally. A deletion is an ASR mismatch, not a confirmed acoustic omission.

<table><tr><td>TTS</td><td>Translator</td><td> $\mathrm { A S R \ e d i t s \ ( \% ) }$ </td><td>Deletions</td></tr><tr><td></td><td>XTTS StreamAtt</td><td> $3 0 . 7 0  2 1 . 6 0$ </td><td> $7 3 8 \to 1 , 1 8 6$ </td></tr><tr><td rowspan="3">Piper</td><td>Baseline</td><td> $2 9 . 4 0  2 3 . 2 0$ </td><td> $1 , 9 8 4  2 , 2 6 8$ </td></tr><tr><td>StreamAtt</td><td> $1 5 . 8 0 \to 1 3 . 2 0$ </td><td> $1 8 1  1 9 0$ </td></tr><tr><td>Baseline</td><td> $1 7 . 1 0  1 5 . 5 0$ </td><td> $4 8 7  4 5 1$ </td></tr></table>

Table 14: Full-output Qwen recognition, immediate → four-second grouping. Rates weight talks equally; deletion counts are pooled. These are recognition mismatches, not verified acoustic omissions.
<table><tr><td>History</td><td>Recognizer</td><td>Insertions</td><td>Deletions</td></tr><tr><td>StreamAtt</td><td>Whisper</td><td> $1 3 0  2 5$ </td><td> $4  2$ </td></tr><tr><td>StreamAtt</td><td>Qwen ASR</td><td> $1 5 9  1 5$ </td><td> $2  0$ </td></tr><tr><td>Baseline</td><td>Whisper</td><td> $5 1  2 0$ </td><td> $7  9$ </td></tr><tr><td>Baseline</td><td>Qwen ASR</td><td> $5 8 \to 1 8$ </td><td> $1  2$ </td></tr></table>

Table 15: Selected-clip edit counts, immediate → foursecond grouping. Each row pools twenty clips per wait from ten recordings. Recognition context, windowing, and selected text coverage differ from the full-recording diagnostic; these counts do not replace its deletion increase.

## D.3 Speech activity and word duration

Silero speech probabilities (Silero Team, 2024) are calculated on consecutive 512-sample frames at 16 kHz, resetting state between waveforms. Thresholds 0.3, 0.5, and 0.7 use the same probabilities without padding or gap merging and retain the actual final-frame duration. The complete analysis includes immediate/four-second StreamAtt–XTTS outputs for all ten talks; other coverage is partial and is not imputed.

Detected non-speech explains 38.75–43.43% of the waveform reduction. Detector-classified speech does not establish preserved linguistic content.

<table><tr><td>Threshold</td><td>Non-speech (s)</td><td>Speech (s)</td><td>Share (%)</td></tr><tr><td>0.3</td><td>138.35</td><td>218.65</td><td>38.75</td></tr><tr><td>0.5</td><td>146.18</td><td>210.82</td><td>40.95</td></tr><tr><td>0.7</td><td>155.04</td><td>201.96</td><td>43.43</td></tr></table>

Table 16: Mean waveform reduction under four-second XTTS grouping for StreamAtt, partitioned by speech activity. Times are seconds; all ten immediate/grouped pairs are included. The final column divides summed non-speech reduction by summed total reduction, not a mean of per-talk percentages. Detector-classified speech does not establish preserved linguistic content.

Repetition, omission, speaking rate, and detector or recognition error may contribute to the remaining reduction.

A separate word-duration analysis uses unambiguous matches to identical intended occurrences, excluding invalid, ambiguous, and request/windowcrossing intervals. Duration is compact-audio end minus start, not playback delay. Request-final words that become request-interior shorten by a mean 0.132 seconds for StreamAtt (1,637 occurrences) and 0.131 seconds for the baseline (1,113). Words interior in both conditions shorten by 0.044 seconds (5,219) and 0.017 seconds (8,843). Means give each talk equal weight; all ten talks contribute. Overlapping intervals are counted once. These estimates are consistent with boundary lengthening, but matching and automatic timestamps do not identify a causal boundary effect or explain all speech-classified audio. The aligner’s 80-ms bins are resolution, not measured accuracy.

## E Automatic Word-Timing Assessment

This supporting analysis uses the original sixty XTTS outputs. It asks when recognizable intended words finish playing after their text becomes available, not when the source first supports their meaning or when a listener understands them. It does not filter any full-talk completion result.

## E.1 Alignment and playback coordinates

Qwen3-ForcedAligner-0.6B (Qwen Team, 2026) aligns each retained Whisper window to that same transcript. It is not a second recognizer. Inference uses German, 16-kHz audio, bfloat16, scaled dotproduct attention, and batches of four. Recorded versions are qwen-asr 0.0.6, PyTorch 2.9.1, Transformers 4.57.6, and NumPy 2.2.6. All 3,860 windows are processed, with 3,845 distinct prediction inputs. Alignment is offline and does not enter the production clock.

The compact waveform concatenates generated audio but omits playback idle. Let u be a predicted compact-audio coordinate, $q _ { j }$ the compact start of part j, and $p _ { j } = \operatorname* { m a x } ( r _ { j } , C _ { j - 1 } )$ its playback start. A coordinate in part j maps to

$$
t _ { \mathrm { p l a y } } ( u ) = p _ { j } + ( u - q _ { j } ) .\tag{6}
$$

The recognition-window offset is added first. At a part join, word ends belong to the preceding part and word starts to the following part. Crossing intervals use their mapped endpoints. This retains later playback gaps rather than adding only the firstplayback delay. Mapping uses 12,857 request parts and 90,859 retained timing units.

## E.2 Matching and added delay

The denominator counts original whitespace words, including punctuation-only tokens. Matching units retain letters, numbers, and apostrophes, omit punctuation, and ignore case while preserving original character offsets. Punctuation-only tokens have no timed unit. Each intended word inherits the latest recorded availability of its characters.

A unit-cost edit alignment compares intended and recognized units. An occurrence is accepted only when every optimal edit path assigns the same exact recognized match and none leaves it unmatched. Timestamps never resolve a repeatedword ambiguity. Every unit in an accepted word needs an unambiguous match and a valid time; its end is the latest matched-unit end. Substitutions, omissions, ambiguous matches, and invalid times remain unknown, not zero delay.

Let $M _ { s h }$ contain the intended occurrences accepted at all three waits for talk s and history h. If $e _ { s h i } ( w )$ is the predicted playback end and $a _ { s h i }$ the fixed text availability, added delay is $\ell _ { s h i } ( w ) = e _ { s h i } ( w ) - a _ { s h i }$ . We compare within-

talk quantiles:

$$
\begin{array} { r l } & { \Delta _ { s h } ^ { ( q ) } ( w ) = Q _ { q } \big ( \{ \ell _ { s h i } ( w ) : i \in M _ { s h } \} \big ) } \\ & { \qquad - Q _ { q } \big ( \{ \ell _ { s h i } ( 0 ) : i \in M _ { s h } \} \big ) , } \end{array}\tag{7}
$$

for $q \in \{ 0 . 5 , 0 . 9 \}$ , then average over talks. These are differences of quantiles, not quantiles of paired word differences. Occurrences are matched across waits within a history, not between the translators different texts.

## E.3 Coverage and uncertainty

The common subset contains 8,508 of 14,430 StreamAtt words (58.96%) and 10,554 of 19,823 baseline words (53.24%). Equal-talk mean coverage is 58.51% and 53.91%, respectively. Per-talk coverage and changes are retained in the numerical supplement. The same subset improves comparability across waits but may not represent excluded words; its coverage gap is not an omission rate.

Table 17 uses 10,000 paired talk-bootstrap draws, seed 20261005, and NumPy’s default linear quantiles. All three outputs and the common subset remain together. Intervals do not include recognizer or timestamp error. The numerical supplement retains the original analysis by text-availability interval, including words released after source end; no final speech tail was discarded.

## E.4 Invalid timestamps and sensitivity

After an initial timestamp failure, the rule was changed to retain isolated invalid units as unknown rather than reject the whole panel. This postobservation change was not independently calibrated. Invalid times were not clipped or regenerated to obtain usable values.

Of 98,243 recognized units, 90,859 retain times and 7,384 (7.52%) remain unknown. Overlapping flags identify 6,313 units changed by the aligner’s nonmonotonic-timestamp postprocessor, 4,395 zero-duration units, 716 out-of-window units, and 64 duplicates. These counts are not disjoint. A single-unit window with an unknown time remains in coverage. Unknown units and the larger common-word coverage gap have different denominators; the latter also excludes words without unambiguous recognition at every wait. The accepted transcripts and timestamps have not been independently checked by listeners. Matching an intended word does not establish its source faithfulness or confirm that it was spoken.

A post-hoc conservative check excludes every common word whose recognition window has any flagged timestamp in any output. It retains only 701 StreamAtt words (4.86%) and 752 baseline words (3.79%). Mean within-talk median changes are −133.31/ − 195.28 seconds for StreamAtt and $- 8 7 . 1 7 / - 1 2 4 . 2 7$ for the baseline at two/four seconds. Lower-median counts remain 10/10 and 9/10. The small retained subset and unchanged signs do not validate timestamp accuracy.

<table><tr><td>History</td><td>Wait (s)</td><td>∆ median</td><td>95% interval</td><td>∆p90</td><td>95% interval</td><td>Lower median</td></tr><tr><td>StreamAtt</td><td>2</td><td>-119.03</td><td>[-145.18, -95.48]</td><td>-219.00</td><td>[-261.47,-177.58]</td><td>10/10</td></tr><tr><td>StreamAtt</td><td>4</td><td>-173.56</td><td>[-207.73, -140.76]</td><td>-314.88</td><td>[-377.34, -252.99]</td><td>10/10</td></tr><tr><td>Baseline</td><td>2</td><td>-72.70</td><td>[-106.75, -41.46]</td><td>-145.13</td><td>[-204.97, -88.56]</td><td>9/10</td></tr><tr><td>Baseline</td><td>4</td><td>-105.36</td><td>[-158.36, -57.25]</td><td>-203.06</td><td>[-289.41, -120.88]</td><td>9/10</td></tr></table>

Table 17: Automatically estimated added delay on the common three-setting word subset. Entries are equal-talk means of within-talk quantile changes from separate requests; all times and intervals are seconds. Negative values mean less delay. The final column counts talks with a lower median, not individual words. Every row uses all ten talks. Intervals resample talks and do not include recognition or timestamp error.

## F Continuing Text-Input Synthesis

Twenty blocks were selected before this experiment, one from each exploratory talk–history stream. They contain 64–151 normalized lexical words spanning 14.43–32.66 seconds of original text availability. Each block has an immediate-feed and a four-second-feed condition. Both start with fresh synthesis state and an empty playback queue, then preserve state across feeds. The grouped condition uses the oldest-update deadline and releases pending text at block end. Blocks are not sourcealigned between histories; only conditions on the same block are compared.

Configuration and clock. Fun-CosyVoice3- 0.5B-2512 uses model revision 29e01c4e and inference code 074ca6dc (Du et al., 2025; FunAudioLLM, n.d.). A timed Python text generator enters the native bistream interface. The fixed reference voice and prompt are identified in the numerical supplement. Settings are FP32, seed 5242 per session, speed 1.0, initial 25-token acoustic hop, and disabled frontend normalization. Native text/acoustic postprocessing is unchanged; independently synthesized fragments are not substituted for continuing generation.

Controls with two feeds fifteen seconds apart verify output before the final feed, retained state, natural completion, and parity of instrumented and uninstrumented generation. All forty sessions reach native end of sequence and drain fully. The 180- second wall-time and 240-second emitted-audio limits do not censor a session. These checks establish execution behavior, not listening quality.

<table><tr><td>History</td><td>∆ first</td><td>∆D</td><td>∆B</td><td>Later end</td></tr><tr><td>StreamAtt</td><td>+3.261</td><td>0</td><td>+1.486</td><td>9/10</td></tr><tr><td>Baseline</td><td>+3.066</td><td>0</td><td>+2.194</td><td>10/10</td></tr></table>

Table 18: Native CosyVoice paired-block changes, foursecond minus immediate feeding. Means are seconds over ten blocks per row. Each pair has the same waveform samples; all forty sessions complete. B is completion after the block’s final original text update. These within-block contrasts are not full-talk translator rankings.

Readiness is measured from the first original block update and includes feed waits and the instrumented host loop; loading and prompt preparation precede that clock. Completion relative to the last original text update is

$$
B _ { b } = C _ { \mathrm { l a s t } , b } - a _ { \mathrm { l a s t } , b } = D _ { b } + I _ { b } - a _ { \mathrm { l a s t } , b } .\tag{8}
$$

This is not a full-talk source-end offset.

Paired outcomes. Each pair has identical flattened text-token sequences, committed speech units, recognized strings, and waveform samples. A check of the archived token records confirms sequence equality in all twenty pairs, although the serialized batch dictionaries differ in every pair. The supplement retains the sample-identity checks and part/feed traces. Normalized mean edit rates are unchanged at 4.84% for StreamAtt blocks and 13.63% for baseline blocks.

Grouping changes $B _ { b }$ by a mean +1.486 seconds for StreamAtt blocks (nine later; range [−0.038, 3.722]) and +2.194 for baseline blocks (ten later; range [0.084, 3.830]). Source-bootstrap intervals are [0.849, 2.193] and [1.416, 2.961], using 10,000 draws, seed 1032026, and the endpoint order-statistic convention. They exclude fresh-seed and execution-time variability. The one 38-ms earlier endpoint is not treated as a reliable speedup.

With unchanged duration, idle accounts for the endpoint differences. The large XTTS shortening is absent in this continuing-session block protocol. Backend, voice prompt, frontend, and context also differ, so persistence alone is not isolated; these blocks do not establish full-talk performance.

## G Ready-Audio Queue during Source Arrival

Final completion does not describe the queue during source arrival. For positive-duration part $j .$ , let $r _ { j } , p _ { j }$ , and $C _ { j }$ be readiness, playback start, and playback end. At source-start time $t ,$ ready but unplayed audio duration is

$$
Q ( t ) = \sum _ { j : r _ { j } \leq t } \operatorname* { m a x } \{ 0 , C _ { j } - \operatorname* { m a x } ( p _ { j } , t ) \} .\tag{9}
$$

This includes the unplayed remainder of a currently playing part, but excludes audio not yet generated. A low queue is therefore not evidence of prompt or faithful translation.

For source duration $S _ { s }$ and progress $u \in [ 0 , 1 ]$ we compare the same source-clock time:

$$
\Delta Q _ { s } ( u ) = Q _ { s , 4 } ( u S _ { s } ) - Q _ { s , 0 } ( u S _ { s } ) .\tag{10}
$$

Figure 2 shows equal-talk means on a 101-point source-progress grid. Bands are the 25th–75th source percentiles with linear quantile interpolation, not confidence intervals. Queue values are reconstructed from readiness jumps and playback start/end points, not interpolated between sparse source observations. No smoothing is used. The figure stops at source end; full post-source draining remains in the completion results and archived queue paths.

XTTS · ten paired talks · bands: source IQR  
![](images/b41dfc836880fb8cd101c4c18bbce2961736df5319ad3170e11ab6645a1153a0.jpg)

![](images/7c58d04eb7f17eb3bb71e752dca54f680130c5aee35c00e5d22ea6dfc75147b3.jpg)  
Figure 2: Change in ready-audio queue during source arrival. Curves show mean paired changes under four-second rather than immediate grouping; bands show the interquartile range across ten talks. Negative values mean less already-generated audio waiting to play, not earlier word delivery. The horizontal axis is source progress; the panels use visibly different vertical scales. Post-source tails remain in the completion tables.