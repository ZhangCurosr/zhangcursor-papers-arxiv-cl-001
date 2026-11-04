# ReListen: Training-Free Evidence Localization in Hour-Long Audio with Frozen Audio Language Models

Ming-Hao Hsu<sup>1,†</sup> Shujie Liu<sup>3,∗</sup> Yuxuan Hu<sup>2</sup> Yuxiang Wang<sup>1</sup> Yingda Shen<sup>1</sup> Jinyu Li<sup>2</sup> Zhizheng Wu<sup>1,∗</sup>

## Abstract

Audio language models are meant to answer questions from the audio itself, but an hourlong recording rarely fits in their context, and the answer often depends on a few seconds of it. Transcribing the recording first makes it fit but discards sounds and voices, while systems that return to the audio must be trained for this behavior. We propose ReListen, a trainingfree method in which a frozen audio language model finds its own evidence before answering. In a Jev-style scan, the model scores each 20- second window by the probability that it helps answer the question, and it then answers from the highest-scoring minute. This minute overlaps the annotated evidence far more often than transcript retrieval, for 88.5% of VoiceGiraffe questions. Without any change, the same scan also grounds events on LAT-Bench as well as a model trained for temporal grounding. Answering from it matches or exceeds hearing the whole recording with a small fraction of the audio, and on VoiceGiraffe it exceeds a transcript of the whole recording by 9.2 points. The selected evidence is also worth hearing rather than reading. Given the same excerpts, reading their transcript matches listening on speech content, but listening is clearly more accurate on questions about sounds and voices. Moreover, ReListen answers faster than every transcriptbased system, and its memory stays constant even for recordings too long to hear whole on one GPU. Therefore, a frozen audio language model can search hour-long recordings by itself, without a transcript or additional training.

## 1 Introduction

Audio language models answer open questions about speech and sound in clips of seconds to a few minutes (Chu et al., 2024; Tang et al., 2024; Gong et al., 2024; Ghosh et al., 2025; Team, 2025). However, meetings, lectures, broadcasts and podcasts last an hour or more. Benchmarks built from such recordings report that accuracy falls as duration grows (Ahia et al., 2025; He et al., 2025; Luo et al., 2026; Huang et al., 2026; Ye et al., 2026). Indeed, many models cannot take an hour of audio as input. Those that can must still find a few relevant seconds.

Question. How many distinct speakers are heard during the pineapple-pizza and cold-pizza debate?  
![](images/dd1907ba4276840f7beb58fbba351be6bc13ae64a2ff74d5e109af3f9937a488.jpg)  
Figure 1: ReListen on a test question about an 84- minute recording. All scores and answers come from the actual run. The frozen audio language model scores each 20-second window by the log-odds of a yes when asked whether the window helps to answer the question. The three best windows overlap the annotated evidence, and the answer from them is correct. From three evenly spaced windows, the same model answers 4, which is wrong.

A common solution places a second system in front of the audio model (Chen et al., 2025; Min et al., 2025; Someki et al., 2026). This system is either a speech recognizer whose transcript is searched or read, or a retriever over audio embeddings. A transcript is particularly effective, since it compresses an hour of speech into text that fits the model’s context. However, the audio model then answers as a text reader. The transcript also discards sounds, voices and timing, which many questions ask about and which audio language models are built to hear. Other systems therefore keep the audio and train the model to return to selected parts of it. They refine a global timeline with audio crops (Shao et al., 2026) or request segments during reasoning (Wu et al., 2026). These systems must be fine-tuned for this behavior, and the training must be repeated for each new audio model. Therefore, a useful solution should search the audio itself, but without a separate front end or additional training.

In this paper, we show that a frozen audio language model can perform this search itself. Our key idea is to reduce the search to many small decisions that the model can already make, namely whether a short excerpt helps to answer the question. We therefore propose ReListen, which uses one frozen audio language model both to search and to answer (Figure 1). ReListen splits the recording into short windows and scores each window by the model’s confidence that it helps to answer the question. The same model then listens again to the highest-scoring minute and answers. This scoring is a Jev-style scan (TypeSafe AI, 2026): each window receives a yes/no decision read as a probability rather than generated text. The scan examines the whole recording, while the answer uses only one minute of audio. No component is trained.

We select the configuration on a development half of LongAudioSpan (Huang et al., 2026). We then evaluate it unchanged on the test half, Voice-Giraffe (Ye et al., 2026) and LAT-Bench (Shao et al., 2026). First, the scan finds evidence. The selected minute overlaps the annotated evidence for 88.5% of VoiceGiraffe questions, compared with 58.8% for transcript retrieval. On LAT-Bench, its mean temporal IoU is comparable to that of the trained LAT-Audio model (47.7 vs. 46.3). Second, the selected evidence improves answers. At the same one-minute budget, ReListen exceeds uniformly sampled audio by 8.5 and 21.3 points on LongAudioSpan and VoiceGiraffe. On VoiceGiraffe, it is the most accurate audio input and exceeds the reader’s transcript of the whole recording by 9.2 points. On LongAudioSpan, it is on par with hearing the whole recording. On VoiceGiraffe sound questions, listening to the selected excerpts also exceeds reading their transcript by 8.6 points. Third, ReListen is efficient, since it needs no training, no transcript and only one minute of audio for the answer. On recordings of about an hour, it answers a question seven to eight times faster than the reader’s transcript pipeline. Its memory also stays constant with duration, and its independent windows divide across GPUs. Taken together, a frozen audio language model can find its own evidence in hour-long recordings, and answering from this evidence is both accurate and efficient.

## 2 Related Work

Answering questions about long recordings requires deciding which part of the audio to process.

Long-Audio Understanding. Prior systems fit long recordings into an audio language model in three ways. They prune audio tokens (Guo et al., 2025; Lin et al., 2025), retrieve over an index of audio, transcripts or events (Chen et al., 2025; Min et al., 2025; Someki et al., 2026), or read text descriptions of short windows (Huang et al., 2026; Ye et al., 2026). These approaches make hour-long inputs tractable and are effective when the question concerns what is said. However, their selection relies on a separate front end, such as a recognizer, an embedding model or a captioner, which decides what the answering model receives. In contrast, we let the answering model decide what to hear.

Answer Probabilities as Scores. Reading the probability of a yes as a relevance score is common in ranking and evaluation. Text rerankers score query-document pairs this way (Nogueira et al., 2020), VQAScore scores image-text alignment (Lin et al., 2024), and recent decision models such as Jev (TypeSafe AI, 2026) answer typed questions with probabilities instead of text. Closest to our scan is the localization probe of video question answering (Yu et al., 2023), which asks the same kind of yes/no question about video frames. Its localizer, however, is a separately trained module on clips of 12 to 76 seconds. In contrast, our Jev-style scan is the frozen answering model itself, applied to recordings in which one minute is at most a tenth of the audio. Yes/no probes of audio language models also hallucinate presence (Kuan et al., 2024) and under-report recurring events (Dai et al., 2026). Therefore, for occurrence questions, we further calibrate these scores (Guo et al., 2017).

Re-Listening Audio Models. Recent audio models show that returning to the audio improves temporal understanding and reasoning. LAT-Audio (Shao et al., 2026) first constructs a global timeline and then refines it with audio crops. Echo (Wu et al., 2026) trains Qwen2.5-Omni to request audio segments by generated timestamps during its reasoning. It is trained by supervised fine-tuning on clips of about ten seconds, followed by reinforcement learning. Speaker-Reasoner (Lin et al., 2026) learns multi-turn temporal interaction for timestamped, speaker-attributed transcription and extends it with a speaker-aware cache. These systems learn when and where to listen, but this behavior must be trained for each model, and the times they use are generated by the model. In contrast, ReListen trains nothing and needs neither a transcript nor generated timestamps. The times of its excerpts are simply the positions of the windows that its scan selects. We compare with these systems in Section 5 and Appendix D.

![](images/c7bc32859a78d575512ff918bb7f2af1afbe4df26f2946c0c9f67329eca9811b.jpg)  
Figure 2: Architecture of ReListen. One frozen reader scores every window in a Jev-style scan (1). It reads each yes/no decision from one forward pass as a probability, and it then answers from the selected audio (3). The native-question path selects by relevance without training (2a). The supervised occurrence extension uses trained acoustic detectors and a posterior over calibrated candidates (2b).

## 3 ReListen

Given a long recording and a question about it, the task is to answer the question. For a grounding query, the task is instead to predict the interval that the query describes. We call the frozen audio language model that answers the reader. Native questions concern the natural content of a recording. Occurrence questions instead ask about the k-th occurrence of an event, such as a sound planted in the recording. Existing systems let a separate front end decide what the reader hears. In contrast, ReListen lets the frozen reader make this decision itself (Figure 2). The reader scores every short window of the recording by whether the window helps to answer the question. It then answers from the highest-scoring windows. The Jev-style scan and native-question answering use no training. We also study a supervised occurrence extension with acoustic detectors and calibration on annotated development recordings (Section 3.3).

## 3.1 Jev-Style Scan by the Reader

We first cut the recording into windows $w _ { 1 } , \ldots , w _ { n }$ of length τ , with a hop $\delta < \tau$ between consecutive windows $( \tau = 2 0$ and $\delta { = } 1 0$ seconds in our experiments). For each window $w _ { i }$ , the reader then receives the audio and a yes/no prompt c built from the question (Appendix B). For a native question, the prompt states the question, with its answer options when there are any, and asks whether the excerpt contains information that helps answer it. For a question about an event $e ,$ it instead asks whether the excerpt contains e. We score each window without generating a yes/no sentence. Using the next-token distribution $p _ { \theta }$ at the first answer position, we write this score as

$$
\ell _ { i } = \log p _ { \boldsymbol \theta } ( { \mathrm { y e s ~ } } \vert \ w _ { i } , c ) - \log p _ { \boldsymbol \theta } ( { \mathrm { n o ~ } } \vert \ w _ { i } , c ) ,\tag{1}
$$

where each probability sums over the forms of the answer word in each prompt language. A positive score means that yes is more likely than no. Higher scores rank the windows for selection. We call this a Jev-style scan. Like the Jev decision model (Type-Safe AI, 2026), it answers a typed yes/no question with a probability instead of text. All windows of a recording share the prompt and the length, so they form batches without padding. Each window costs one forward pass. The cost of the scan therefore grows linearly with duration at constant memory. The window passes are independent, so they can also run in parallel on several GPUs.

## 3.2 Answering Native Questions

For a native question, ReListen takes the windows in decreasing order of $\ell _ { i }$ . It keeps a window only if it does not overlap any window already kept. It stops after b windows, that is, bτ seconds of audio (b=3, or 60 seconds, in our experiments). A larger budget suits questions whose evidence spreads across the recording (Section 5.6). The reader then answers the original question once from these excerpts in chronological order. Adjacent excerpts are merged, and each is labelled with its start and end time in the recording, since many options name times. Answering one question thus costs n window passes and one answer from bτ seconds of audio, without any transcription.

Event Intervals. For a grounding query, ReListen outputs one interval instead of excerpts. This expanded region starts from the highest-scoring window and extends it left and right through consecutive windows with positive scores. Its duration is therefore set by the scores, not by the budget bτ.

## 3.3 Extension to Occurrence Questions

However, selection by relevance has an inherent limit. A question about the k-th occurrence of an event is answered correctly only if every earlier occurrence is found and no spurious one is counted before it. Yet all occurrences match the event’s description equally well. For these questions, we therefore study a supervised extension. Some event types, such as changes of speed and volume, are not described by what the audio contains. On development recordings, the scan’s scores indeed carry little information about them (Appendix K). Therefore, two small acoustic detectors, trained on synthetic edits of the benchmark’s types, propose candidate event times. The highest-scoring windows of the scan add further candidates. A logistic model fitted on annotated development recordings then assigns each candidate $j$ a probability $p _ { j }$ of being a real event. A natural rule keeps the candidates above a threshold and takes the k-th in time. However, a single false alarm or miss before the target then shifts the answer by one occurrence. Instead, the extension estimates for each candidate the probability that it is real and that exactly k − 1 earlier candidates are real. With the candidates sorted by time and treated as independent, this probability is

$$
\begin{array} { r } { \pi _ { j } ^ { ( k ) } = p _ { j } \cdot \mathrm { P r } \left[ \sum _ { i < j } z _ { i } = k - 1 \right] , } \end{array}\tag{2}
$$

with $z _ { i } \sim \mathrm { B e r n o u l l i } ( p _ { i } )$ . The second factor is a Poisson-binomial probability (Hong, 2013), computed for all $j$ by dynamic programming. The extension then selects the candidate with the largest $\pi _ { j } ^ { ( k ) }$ . The reader then answers the remaining questions of the chain from one minute of audio around this candidate (Appendix K).

## 4 Experimental Setup

Our experiments test whether the reader’s scan finds the evidence for a question, and whether answering from the selected audio improves accuracy at a fixed audio budget. The same reader answers from every audio and text input, so differences reflect only what it receives.

## 4.1 Benchmarks

We evaluate on three benchmarks of long English and Mandarin recordings (Figure 3).

LongAudioSpan. LongAudioSpan (Huang et al., 2026) contains 360 recordings of 10 to 164 minutes, with three native questions written from each unedited recording. We evaluate its two multiplechoice modes. The accuracy mode asks the 1,080 native questions with four options each. The chain mode asks three linked questions about an edited copy of the recording with one to three planted edits of one type. An edit is an inserted sound event, a stretch replaced by silence or masked by noise, or a change of playback speed or volume. Q1 names the edited event and asks how many occurrences there are and where the k-th one is, and its four options cross two counts with two times. Q2 then asks what was said at that point, and Q3 asks how that relates to the rest of the recording. A chain scores 0, 33.3, 66.7 or 100 for zero to three consecutive correct answers from Q1, and random guessing gives 10.9 in expectation. We split the recordings into 180 development and 180 test recordings, with 540 native questions and 180 occurrence chains per half. This split is stratified by edit type, language and duration, and the test recordings average 48.3 minutes. We choose the scan’s use of answer options and its number of windows on development recordings. The same recordings choose the detector recipes, candidate thresholds and calibrators of the occurrence extension. We then evaluate the fixed configuration once on the test half in each mode.

![](images/a5959067408c170dbdcae9e010e5775b9a4a084853be2f248efa2f9825e77d92.jpg)  
Figure 3: The three benchmarks, with real questions. Each bar is a whole recording drawn to scale, and the mark on it is the annotated evidence. Questions are shortened only with ellipses, and Appendix A gives full examples.

VoiceGiraffe. VoiceGiraffe (Ye et al., 2026) contains 123 recordings of 35 to 79 minutes, and we evaluate 1,499 of its 1,500 multiple-choice questions. We call its Audio Event and Paralinguistic families, which ask about non-speech sounds and voice properties, the sound families (549 questions). The other four concern the content (950 questions).

LAT-Bench. LAT-Bench (Shao et al., 2026) provides 1,002 temporal grounding queries about 247 recordings of up to 30 minutes. Each query describes a moment by its speech, music and sound effects, and a system predicts its start and end times. We report the mean temporal IoU (mIoU) and the recall at IoU thresholds of 0.3, 0.5 and 0.7, averaged equally over English and Chinese.

## 4.2 Reader and Metrics

Reader. We use Qwen3-Omni-30B-A3B-Instruct (Team, 2025) as the reader, with frozen weights and greedy decoding. The reader receives 16 kHz waveforms, and each selected excerpt is cropped from the waveform before the reader’s audio encoder. VoiceGiraffe and LAT-Bench use the LongAudioSpan prompts, windows and budgets without change (Appendix C).

Metrics. The evidence hit rate is the percentage of annotated questions for which any selected excerpt overlaps any annotated evidence range, even partially. We use the annotated support ranges on LongAudioSpan and the evidence times on VoiceGiraffe. For question answering, we report multiple-choice accuracy. A time-option question has a clock time in at least three of its four options. Intervals are 95% bootstrap confidence intervals that resample recordings, since questions about the same recording are not independent.

Efficiency. Since the scan depends on the question, we time one question per recording. The timed questions come from 22 test LongAudioSpan recordings, 10 VoiceGiraffe recordings and 10 LAT-Bench recordings. Each time runs from decoding the audio file to the answer (Appendix C).

## 4.3 Baselines

Selection baselines. These baselines give the reader the same 60 seconds of audio as ReListen but choose it differently. Uniform sampling takes 20-second windows centred at one sixth, one half and five sixths of the recording. Our learned audio retriever is a scorer trained on frozen reader features, which selects eight short regions. BM25 (Robertson and Zaragoza, 2009) and BGE-M3 (Chen et al., 2024) rank 30-second transcript chunks by word matching and dense-embedding similarity, respectively. Both use the question and its options as the query and select the audio of the top two chunks.

Input baselines. These baselines change what the same reader receives. No audio supplies the question and its answer options alone. Full recording supplies the audio up to an 80-minute cap, which covers every VoiceGiraffe recording. Its time marks variant precedes each consecutive 30-second segment with its start time as text. Whisper transcript and reader transcript supply the text of the whole recording, which Whisper large-v3 (Radford et al., 2023) or the reader transcribes in consecutive 30-second excerpts. They carry no time marks and no audio, so they measure answering from the words alone. Finally, annotated evidence serves as a reference rather than a baseline. It supplies the evidence intervals, padded to at least 20 seconds each and capped at 120 seconds in total.

Prior systems. We run the released LAT-Audio model (Shao et al., 2026) with its official tool policy on the same LAT-Bench queries as ReListen. Echo (Wu et al., 2026) reasons over the audio itself, but its trained checkpoint is unavailable. Our Echostyle system therefore gives the frozen reader its whole timestamped transcript and lets it request audio while reasoning. Speaker-Reasoner (Lin et al., 2026) transcribes each recording with its released Stage 2 model, and the reader answers from this timestamped transcript (Appendix D). We call the transcript inputs and these two systems textmediated, since they convert the whole recording to text before any question is answered.

![](images/02a8d5db79637cb90b2feca7c82e8e32243b237a8fe16be98f58bd4bd8b8428e.jpg)  
Figure 4: Accuracy against the audio given to the reader. Multiple-choice accuracy on test LongAudioSpan and on VoiceGiraffe. The full recording is placed at its mean duration up to 80 minutes, and the annotated evidence at its mean length.

## 5 Results

## 5.1 Experimental Results

We report multiple-choice accuracy and cost per question on LongAudioSpan and VoiceGiraffe in Table 1. Appendices E and G report results by dimension, duration and question family.

Overall Task Performance. With the same 60- second budget, our ReListen outperforms uniform excerpts by 8.5 points on LongAudioSpan and 21.3 points on VoiceGiraffe. Compared with the full recording, which gives the reader 44 to 54 times more audio, ReListen performs competitively on LongAudioSpan and outperforms it by 10.6 points on VoiceGiraffe (Figure 4). In summary, ReListen achieves the best or competitive accuracy among audio inputs, which verifies that a frozen reader can select the evidence it needs to answer.

Comparison with Transcripts. On VoiceGiraffe, ReListen outperforms the reader’s transcript of the whole recording by 9.2 points. The Echo-style and Speaker-Reasoner systems, which read timestamped transcripts, also perform below ReListen (Appendix D). On LongAudioSpan, whose questions mostly concern what is said, the transcript performs better on questions without time options.

![](images/a6c3abe7d72b25a6059901604a38bcef31e5877558a7a8e9a1f9a61e9a29a2a1.jpg)  
Figure 5: Accuracy against seconds per question on VoiceGiraffe. Each point gives a method’s mean time over the timed questions and its test accuracy (Table 1). Filled marks use one GPU, and open marks divide the scan over four GPUs with the same selection. The horizontal axis is logarithmic, and every ReListen point lies left of all text-mediated systems.

## 5.2 Inference Efficiency

Table 1 also reports the mean seconds per question, and Figure 5 plots accuracy against them on VoiceGiraffe.

Comparison with Text-Mediated Systems. On LongAudioSpan, ReListen answers a question in 17.3 seconds on one GPU. In contrast, the textmediated systems take 27.6 to 1124 seconds, since they first transcribe the whole recording. On both benchmarks, ReListen is seven to eight times faster than the reader’s transcript pipeline. Therefore, ReListen is faster than every text-mediated system and, on VoiceGiraffe, also more accurate.

Comparison with Trained Grounding. The trained LAT-Audio model grounds a LAT-Bench query in 173 seconds on average. In contrast, the scan grounds the same queries in 3.1 seconds without any training, at a comparable mIoU (Table 3).

Comparison with the Full Recording. On one GPU, the full recording is faster than ReListen-Fast only for recordings under about 30 minutes. Its time grows faster than linearly with duration, and it exceeds one 80 GB GPU beyond about 62 minutes. In contrast, the time of the scan grows linearly, and its memory stays below 68 GiB at every duration.

Parallel Scan and ReListen-Fast. Since the reader scores every window independently, the scan divides across GPUs. On the two benchmarks, four

<table><tr><td></td><td></td><td></td><td colspan="2">LongAudioSpan</td><td colspan="4">VoiceGiraffe</td></tr><tr><td>Method</td><td>Training</td><td>GiB</td><td>S</td><td>Acc.</td><td>S</td><td>All</td><td>Sound</td><td>Content</td></tr><tr><td colspan="9">Text-mediated</td></tr><tr><td>Reader transcript</td><td>x</td><td>71.9</td><td>119</td><td>59.6</td><td>135</td><td>52.8</td><td>50.8</td><td>54.0</td></tr><tr><td>Whisper transcript</td><td>x</td><td>68.4</td><td>27.6</td><td>58.1</td><td>28.4</td><td></td><td></td><td></td></tr><tr><td>Echo-style (Wu et al., 2026)</td><td>x</td><td>72.8</td><td>157</td><td>58.1</td><td>169</td><td>58.1</td><td>51.0</td><td>62.2</td></tr><tr><td>Speaker-Reasoner (Lin et al., 2026)</td><td>√</td><td>78.1</td><td>1124</td><td>53.0</td><td>1048</td><td>55.7</td><td>52.6</td><td>57.5</td></tr><tr><td colspan="9">Audio input</td></tr><tr><td>No audio</td><td>x</td><td></td><td>一</td><td>43.5</td><td></td><td>34.5</td><td>37.3</td><td>32.8</td></tr><tr><td>Uniform excerpts</td><td>x</td><td></td><td></td><td>45.0</td><td></td><td>40.7</td><td>48.5</td><td>36.2</td></tr><tr><td>Full recording</td><td>x</td><td>78.2</td><td>11.4</td><td>54.6</td><td>11.6</td><td>51.4</td><td>51.4</td><td>51.5</td></tr><tr><td>ReListen (ours)</td><td>x</td><td>67.8</td><td>17.3</td><td>53.5</td><td>17.0</td><td>62.0</td><td>58.7</td><td>64.0</td></tr><tr><td>four GPUs</td><td></td><td>67.8</td><td>5.6</td><td></td><td>5.7</td><td></td><td></td><td></td></tr><tr><td>ReListen-Fast (ours)</td><td>X</td><td>67.8</td><td>10.0</td><td>53.7</td><td>9.0</td><td>60.9</td><td>57.7</td><td>62.7</td></tr><tr><td>four GPUs</td><td></td><td>67.8</td><td>3.7</td><td></td><td>3.6</td><td></td><td></td><td></td></tr></table>

Table 1: Accuracy and cost per question. All rows use the same frozen reader to answer, and we report accuracy (%) on test LongAudioSpan and on VoiceGiraffe. Seconds (s) are means over the timed questions. GiB is the peak memory on any one GPU, including the reader’s 59 GiB of weights, and Training marks a model fine-tuned for the task. Uniform excerpts and ReListen give the reader 60 seconds of audio, and the full recording carries time marks. Rows use one GPU unless marked four GPUs, which reproduce the one-GPU selections and accuracy. The full recording uses two GPUs beyond about 62 minutes.

<table><tr><td></td><td>LAS</td><td colspan="2">VoiceGiraffe</td></tr><tr><td>Method</td><td>All</td><td>All</td><td>Sound</td></tr><tr><td>Uniform sampling</td><td>18.7</td><td>11.1</td><td>4.4</td></tr><tr><td>Learned audio retriever</td><td>32.6</td><td>16.5</td><td>5.8</td></tr><tr><td>BM25</td><td>45.2</td><td>58.8</td><td>45.6</td></tr><tr><td>BGE-M3</td><td>47.8</td><td>52.1</td><td>37.6</td></tr><tr><td>ReListen (ours)</td><td>69.2</td><td>88.5</td><td>76.5</td></tr></table>

Table 2: Evidence localization with a 60-second budget. The hit rate (%) is the fraction of questions for which any selected excerpt overlaps the annotated evidence. LAS is test LongAudioSpan.

GPUs make ReListen 3.1 and 3.0 times faster than one GPU. It is then faster than the full recording on 29 of the 32 timed questions. As an efficiency setting, ReListen-Fast uses a hop of 20 seconds, so that the scan hears each second once. Compared with ReListen, its accuracy changes by +0.2 points on LongAudioSpan and −1.1 on VoiceGiraffe, and both 95% intervals include zero. We keep the 10- second hop, chosen on development recordings, as the main configuration. In summary, ReListen is faster than every text-mediated system at constant memory, which verifies that searching the audio itself is efficient.

## 5.3 Evidence Localization

This section investigates whether the scan selects the evidence that a question needs. To this end, we measure how often each 60-second selection overlaps the annotated evidence (Table 2).

<table><tr><td>Method</td><td>mIoU</td><td>R@0.3 R@0.5 R@0.7</td><td></td><td></td></tr><tr><td>Uniform, centre window BM25, best 30-s chunk Full recording</td><td>2.9 34.2 12.3</td><td>3.8 56.6 17.9</td><td>1.9 29.4 10.0</td><td>0.8 11.1 5.1</td></tr><tr><td>with time marks Gemini-2.5-Pro†</td><td>39.3 42.8</td><td>61.9 63.3</td><td>37.7 51.3</td><td>15.7 26.9</td></tr><tr><td>LAT-Audio ReListen (ours)</td><td>46.3 47.7</td><td>62.0 77.2</td><td>48.4 50.0</td><td>32.5 19.2</td></tr></table>

Table 3: Event grounding on LAT-Bench. We report mean IoU and recall at IoU 0.3, 0.5 and 0.7 (%). ReListen returns the expanded region of its scan. Appendix G gives ablations. <sup>†</sup>Reported by Shao et al. (2026).

Comparison with Retrieval. The scan hits the evidence for 69.2% of LongAudioSpan and 88.5% of VoiceGiraffe questions, whereas transcript retrieval hits it for at most 47.8% and 58.8%. The gap is largest on sound questions, where transcript retrieval falls to 45.6% and the scan keeps 76.5%, since sounds and voices leave no words to match.

Event Grounding. On LAT-Bench, the expanded region of the scan reaches 47.7 mIoU without any change, which is comparable to the trained LAT-Audio model on the same queries (Table 3). Its recall at an IoU of 0.7 is lower, so the scan finds the right region with less precise boundaries.

## 5.4 Listening versus Reading

This section further investigates whether the selected evidence is worth hearing rather than reading. To this end, we let the reader read its own transcript of exactly the excerpts it would hear (Table 4). On content questions, listening and reading differ by about two points or less, for both the scan’s excerpts and the annotated evidence. Therefore, the lead of a whole transcript on LongAudioSpan comes from covering the recording rather than from the text. On VoiceGiraffe sound questions, however, listening outperforms reading by 8.6 points on the scan’s excerpts and by 19.5 points on the annotated evidence. This verifies our motivation to answer from the audio itself, since a transcript discards the sounds and voices that these questions ask about.

<table><tr><td rowspan="2">Excerpt source</td><td colspan="2">LAS</td><td>VG content</td><td>VG sound</td></tr><tr><td></td><td></td><td>Audio Text Audio Text Audio Text</td><td></td></tr><tr><td>ReListen, 60 s</td><td>60.5</td><td>59.4 64.0</td><td>65.2</td><td>58.7 50.1</td></tr><tr><td>Annotated evidence</td><td>70.9</td><td>70.1 78.3</td><td>76.3</td><td>67.9 48.4</td></tr></table>

Table 4: Listening versus reading the same excerpts. The reader hears the excerpts (Audio) or reads its own transcript of them (Text), and we report accuracy (%). LAS uses test LongAudioSpan questions without time options, with audio sliced from whole-recording encoder outputs.

## 5.5 Upper Bound with Annotated Evidence

To analyze the remaining errors, we replace the selected minute by the annotated evidence. The reader then answers 72.7% on LongAudioSpan and 74.5% on VoiceGiraffe, 19.2 and 12.5 points above ReListen (Tables 5 and 8). On questions without time options, the annotated evidence also outperforms the whole transcript, 73.0% against 65.7% on LongAudioSpan and 69.7% against 55.7% on VoiceGiraffe. On LongAudioSpan, when the scan misses the evidence, ReListen answers 38.7%, the same as without audio, and these misses account for half of the gap. When the scan hits the evidence, more audio does not help, since one, three and ten minutes yield about 64% on questions without time options (Appendix J). These results suggest that more precise selection, rather than more audio, would further improve ReListen.

## 5.6 Questions about the Whole Recording

However, more audio does help when the evidence spreads across a recording. On 61 test questions of LongAudioSpan, the annotated evidence extends over at least half of the recording. There, the selected minute answers 55.7%, against 73.8% for the full recording. Ten minutes from the same scan reach 72.1%, although a gap remains on recordings of 30 to 60 minutes (Appendix F). Beyond ten minutes, the budget can grow up to the whole recording that the full-recording baseline hears. On VoiceGiraffe, ReListen outperforms the full recording on the 31 such questions, 54.8% against 35.5%. For questions about the k-th occurrence of an event, the occurrence extension instead weighs candidates from the whole recording (Section 3.3). In summary, the same scan with a larger budget closes most of the gap to the full recording, which verifies that the budget rather than the scan limits ReListen on these questions.

## 5.7 Occurrence Questions

Occurrence chains ask for the k-th occurrence of an event, and every occurrence matches the event’s description equally well. Hence, selection by relevance alone cannot tell the occurrences apart, and the Whisper transcript reaches 10.2, below random guessing at 10.9 (Table 10). Even the full recording with time marks reaches only 17.2. Our supervised extension reaches a chain score of 60.7 and outperforms verifying retrieved candidates one by one, which reaches 37.4. Most of this gain comes from the detectors, since the scan alone reaches 35.9 and the detectors alone 52.0. A thresholded count instead of the occurrence estimate lowers the score to 55.9 (Appendix M). For coughs in meetings, which have no detector, the scan alone places the k-th cough within 30 seconds for 37.0% of the questions, against 3.8% for a random time (Appendix N).

## 6 Conclusion

A frozen audio language model can locate the evidence for a question in a long recording without training a selector. Its selected minute overlaps the annotated evidence for 69.2% of test LongAudioSpan and 88.5% of VoiceGiraffe questions. Answering from it matches or exceeds hearing the whole recording with 44 to 54 times less audio. On VoiceGiraffe the selected minute exceeds a transcript of the whole recording. On sound questions, selected audio also exceeds a transcript of the same excerpts. It also answers faster than every transcript-based system, with constant memory and a scan that divides across GPUs. The remaining gap to the annotated evidence calls for selection that isolates the needed sound more precisely.

## Limitations

Our evidence comes from one reader and three benchmarks of mostly English and Mandarin speech. Transfer to other model families, music, sound scenes or other languages is therefore untested. Our cost comparison also assumes one question per recording, whereas with several questions a transcript can be reused while the scan reruns for each question. ReListen also uses one budget for every question, and choosing the budget for each question is untested. Questions that need more audio than the reader can hear at once, such as summaries of very long recordings, lie beyond this work.

## Ethical Considerations

Evidence localization can help users search meetings, calls and other long recordings, but can also expose statements from private conversations without the speakers’ consent, so applications should respect consent and access restrictions for both recordings and their transcripts. All three benchmarks are research releases of human speech, VoiceGiraffe waveforms were obtained under its gated access agreement, and we redistribute no recordings. Some questions concern age, gender or emotion inferred from a voice, which we report as benchmark categories rather than proposed applications, and since the systems still make substantial errors, their outputs should not be used for decisions about individuals.

## References

Orevaoghene Ahia, Martijn Bartelds, Kabir Ahuja, Hila Gonen, Valentin Hofmann, Siddhant Arora, Shuyue Stella Li, Vishal Puttagunta, Mofetoluwa Adeyemi, Charishma Buchireddy, Ben Walls, Noah Bennett, Shinji Watanabe, Noah A. Smith, Yulia Tsvetkov, and Sachin Kumar. 2025. BLAB: brutally long audio bench. CoRR, abs/2505.03054.

Jean Carletta, Simone Ashby, Sebastien Bourban, Mike Flynn, Maël Guillemot, Thomas Hain, Jaroslav Kadlec, Vasilis Karaiskos, Wessel Kraaij, Melissa Kronenthal, Guillaume Lathoud, Mike Lincoln, Agnes Lisowska, Iain McCowan, Wilfried M. Post, Dennis Reidsma, and Pierre Wellner. 2005. The AMI meeting corpus: A pre-announcement. In MLMI, volume 3869 of Lecture Notes in Computer Science, pages 28–39. Springer.

Jianlyu Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. 2024. M3- embedding: Multi-linguality, multi-functionality,

multi-granularity text embeddings through selfknowledge distillation. In ACL (Findings), volume ACL 2024 of Findings of ACL, pages 2318–2335. Association for Computational Linguistics.

Yifu Chen, Shengpeng Ji, Haoxiao Wang, Ziqing Wang, Siyu Chen, Jinzheng He, Jin Xu, and Zhou Zhao. 2025. Wavrag: Audio-integrated retrieval augmented generation for spoken dialogue models. In ACL (1), pages 12505–12523. Association for Computational Linguistics.

Yunfei Chu, Jin Xu, Qian Yang, Haojie Wei, Xipin Wei, Zhifang Guo, Yichong Leng, Yuanjun Lv, Jinzheng He, Junyang Lin, Chang Zhou, and Jingren Zhou. 2024. Qwen2-audio technical report. CoRR, abs/2407.10759.

Yuhang Dai, Xin Shu, Zengxi Li, Lei Xie, Xiangang Li, and Jianwei Yu. 2026. Tag-bench: Benchmarking temporal audio grounding in large audio language models. Preprint, arXiv:2609.01542.

Yihui Fu, Luyao Cheng, Shubo Lv, Yukai Jv, Yuxiang Kong, Zhuo Chen, Yanxin Hu, Lei Xie, Jian Wu, Hui Bu, Xin Xu, Jun Du, and Jingdong Chen. 2021. AISHELL-4: an open source dataset for speech enhancement, separation, recognition and speaker diarization in conference scenario. In Interspeech, pages 3665–3669. ISCA.

Sreyan Ghosh, Arushi Goel, Jaehyeon Kim, Sonal Kumar, Zhifeng Kong, Sang-gil Lee, Chao-Han Huck Yang, Ramani Duraiswami, Dinesh Manocha, Rafael Valle, and Bryan Catanzaro. 2025. Audio flamingo 3: Advancing audio intelligence with fully open large audio language models. In NeurIPS.

Yuan Gong, Hongyin Luo, Alexander H. Liu, Leonid Karlinsky, and James R. Glass. 2024. Listen, think, and understand. In ICLR. OpenReview.net.

Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. 2017. On calibration of modern neural networks. In ICML, volume 70 of Proceedings of Machine Learning Research, pages 1321–1330. PMLR.

Shoutao Guo, Shaolei Zhang, Qingkai Fang, Zhengrui Ma, Min Zhang, and Yang Feng. 2025. Fastlongspeech: Enhancing large speech-language models for efficient long-speech processing. In NeurIPS.

Peize He, Zichen Wen, Yubo Wang, Yuxuan Wang, Xiaoqian Liu, Jiajie Huang, Zehui Lei, Zhuangcheng Gu, Xiangqi Jin, Jiabing Yang, Kai Li, Zhifei Liu, Weijia Li, Cunxiang Wang, Conghui He, and Linfeng Zhang. 2025. Audiomarathon: A comprehensive benchmark for long-context audio understanding and efficiency in audio llms. CoRR, abs/2510.07293.

François Hernandez, Vincent Nguyen, Sahar Ghannay, Natalia A. Tomashenko, and Yannick Estève. 2018. TED-LIUM 3: Twice as much data and corpus repartition for experiments on speaker adaptation. In SPECOM, volume 11096 of Lecture Notes in Computer Science, pages 198–208. Springer.

Yili Hong. 2013. On computing the distribution function for the poisson binomial distribution. Comput. Stat. Data Anal., 59:41–51.

Wen Huang, Yunfei Chu, Meng Gao, Haolin He, and Jin Xu. 2026. Audiospan: Spanning the duration and depth of audio comprehension. CoRR, abs/2608.26431.

Chun-Yi Kuan, Wei-Ping Huang, and Hung-yi Lee. 2024. Understanding sounds, missing the questions: The challenge of object hallucination in large audiolanguage models. In INTERSPEECH. ISCA.

Yueqian Lin, Yuzhe Fu, Jingyang Zhang, Yudong Liu, Jianyi Zhang, Jingwei Sun, Hai Helen Li, and Yiran Chen. 2025. Speechprune: Context-aware token pruning for speech information retrieval. In ICME, pages 1–6. IEEE.

Zhennan Lin, Shuai Wang, Zhaokai Sun, Pengyuan Xie, Chuan Xie, Jie Liu, Qiang Zhang, and Lei Xie. 2026. Speaker-reasoner: Scaling interaction turns and reasoning patterns for timestamped speaker-attributed ASR. Preprint, arXiv:2604.03074.

Zhiqiu Lin, Deepak Pathak, Baiqi Li, Jiayao Li, Xide Xia, Graham Neubig, Pengchuan Zhang, and Deva Ramanan. 2024. Evaluating text-to-visual generation with image-to-text generation. In ECCV (9), volume 15067 of Lecture Notes in Computer Science, pages 366–384. Springer.

Kaiwen Luo, Liang Lin, Yibo Zhang, Moayad Aloqaily, Dexian Wang, Zhenhong Zhou, Junwei Zhang, Kun Wang, Li Sun, and Qingsong Wen. 2026. Chronosaudio: A comprehensive long-audio benchmark for evaluating audio-large language models. CoRR, abs/2601.04876.

Do June Min, Karel Mundnich, Andy Lapastora, Erfan Soltanmohammadi, Srikanth Ronanki, and Kyu J. Han. 2025. Speech retrieval-augmented generation without automatic speech recognition. In ICASSP, pages 1–5. IEEE.

Rodrigo Nogueira, Zhiying Jiang, Ronak Pradeep, and Jimmy Lin. 2020. Document ranking with a pretrained sequence-to-sequence model. In EMNLP (Findings), volume EMNLP 2020 of Findings of ACL, pages 708–718. Association for Computational Linguistics.

Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur. 2015. Librispeech: An ASR corpus based on public domain audio books. In ICASSP, pages 5206–5210. IEEE.

Karol J. Piczak. 2015. ESC: dataset for environmental sound classification. In ACM Multimedia, pages 1015–1018. ACM.

Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. 2023. Robust speech recognition via large-scale weak supervision. In ICML, volume 202 of Proceedings of

Machine Learning Research, pages 28492–28518. PMLR.

Stephen E. Robertson and Hugo Zaragoza. 2009. The probabilistic relevance framework: BM25 and beyond. Found. Trends Inf. Retr., 3(4):333–389.

Mingchen Shao, Hang Su, Wenjie Tian, Bingshen Mu, Zhennan Lin, Lichun Fan, Zhenbo Luo, Jian Luan, and Lei Xie. 2026. Listening with time: Precise temporal awareness for long-form audio understanding. Preprint, arXiv:2604.22245.

Masao Someki, Chien-Yu Huang, Siddhant Arora, Samuele Cornell, Markus Müller, Nathan Susanj, Rupak Vignesh Swaminathan, Grant P. Strimel, Jing Liu, and Shinji Watanabe. 2026. Planrag-audio: Planning and retrieval augmented generation for long-form audio understanding. In ACL (Findings), pages 26167– 26183. Association for Computational Linguistics.

Changli Tang, Wenyi Yu, Guangzhi Sun, Xianzhao Chen, Tian Tan, Wei Li, Lu Lu, Zejun Ma, and Chao Zhang. 2024. SALMONN: towards generic hearing abilities for large language models. In ICLR. Open-Review.net.

Qwen Team. 2025. Qwen3-omni technical report. CoRR, abs/2509.17765.

TypeSafe AI. 2026. Jev: A system one decision model. https://techcrunch.com/2026/09/18/ a-new-kind-of-ai-model-from-a-cha tgpt-inventor-is-thrilling-develop ers/. Accessed 2026-10-23.

Daiqing Wu, Xuan Zhang, Dongbao Yang, Jiashu Yao, Longfei Chen, Qingsong Liu, Sicheng Zhao, Can Ma, Yangyang Kang, and Yu Zhou. 2026. Echo: Towards advanced audio comprehension via audio-interleaved reasoning. In The Fourteenth International Conference on Learning Representations.

Yusong Wu, Ke Chen, Tianyu Zhang, Yuchen Hui, Taylor Berg-Kirkpatrick, and Shlomo Dubnov. 2023. Large-scale contrastive language-audio pretraining with feature fusion and keyword-to-caption augmentation. In ICASSP, pages 1–5. IEEE.

Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He, Shuai Bai, Keqin Chen, Jialin Wang, Yang Fan, Kai Dang, Bin Zhang, Xiong Wang, Yunfei Chu, and Junyang Lin. 2025. Qwen2.5-omni technical report. CoRR, abs/2503.20215.

Jashin Ye, Dongxiao Wang, Yixuan Ye, Sashuai Zhou, Weihuang Lin, Mingyang Han, Kunpeng Wang, Zeyu Yuan, Boyu Li, Haoxiang Shi, Jingchen Shu, Jun Song, and Bo Zheng. 2026. Voicegiraffe: A benchmark for extreme long-context audio-language understanding. CoRR, abs/2605.27976.

Shoubin Yu, Jaemin Cho, Prateek Yadav, and Mohit Bansal. 2023. Self-chained image-language model for video localization and question answering. In NeurIPS.

## A Benchmark Examples

We show examples from the three benchmarks. The correct option is in bold, and the evidence is the annotated time range.

## A.1 LongAudioSpan

LongAudioSpan assigns each native question one of three levels, perception, understanding and reasoning, and one recording receives one question per level. The first three boxes show one question of each level in the accuracy mode. Note that a perception question can already require searching a long recording: in the first example, the 11-second evidence lies 17 minutes into a 73-minute recording.

Accuracy mode, perception (recognition). 72.6-minute   
recording.   
During the brief pause to fix the internet connection, which   
non-speech sound is clearly audible alongside the speaker’s   
voice?   
(A) Heavy breathing from exertion (B) A loud digital static   
burst (C) Rapid mouse clicks and keyboard taps (D)   
Distant traffic noise outside   
Evidence: 17:33–17:44.

Accuracy mode, understanding (tracking). 65.1-minute   
recording.   
At what point does the primary geographic focus of the   
festival broadcast shift from Colorado to Australia?   
(A) 05:13–05:45 (B) 34:41–36:29 (C) 32:52–34:41   
(D) 44:45–48:35   
Evidence: 34:41–36:29.

Accuracy mode, reasoning (relational). 10.5-minute   
recording.   
What unstated connection do the commentators draw be  
tween the Bulgarian pair’s earlier pedigree and their later   
play?   
(A) It gives them resilience, but not enough attacking   
variety   
(B) It proves they are coasting on reputation alone   
(C) It makes them the sharper pair at the net   
(D) It shows their coaching is driving the late tactical   
changes   
Evidence: 05:37–05:53 and 10:00–10:25.

Occurrence chains are asked about an edited copy of a recording. In the example below, the same 6.5-second clip of bird song is inserted twice, at 09:36 and at 30:09 of a 40.8-minute recording. Q1 asks for the count and the position of one occurrence. In contrast, Q2 and Q3 refer to that occurrence without stating its time, so they can be answered only after it is found.

Occurrence chain, inserted sound event. 40.8-minute edited recording.

Q1. This audio contains bird singing. How many instances are there in total, and approximately where does the 1st one occur?

(A) 2 in total, the 1st near 09:39 (B) 2 in total, the 1st near 30:12

(C) 1 in total, the 1st near 09:39 (D) 1 in total, the 1st near 30:12

Q2. This audio contains bird singing. At the location of the first occurrence, what is being discussed?

(A) A male host rapidly lists specific Breeze Copilot capabilities like CRM object updates and engagement summaries.

(B) A female narrator defines the Commerce Analytics Suite as a tool for streamlining sales billing and payments.

(C) A discussion focuses on editing video clip transcripts to improve the workflow for creating short content.

(D) Speakers reflect on breaking down silos to create a unified view across the entire customer journey.

Q3. This audio contains bird singing. How does the content at the location of the first occurrence relate to other parts of the audio?

(A) The content here serves as a thematic callback to the subsequent demonstration of using Claude for project management tasks.

(B) The content here provides a comparative example to the later discussion on external AI models integrating via MCP. (C) The content here acts as a verification check that contradicts the earlier announcement of the Commerce Analytics Suite availability.

(D) The content here forms a feature expansion relation to the later section on collaborative workflow building with Copilot.

The other edit types use the same three templates. For example, a Q1 about a speed edit reads “This audio contains sudden playback speedup. How many instances are there in total, and approximately where does the 2nd one occur?”

## A.2 VoiceGiraffe

VoiceGiraffe groups its questions into six families. The two sound families ask about non-speech sounds and about voice properties, and the Paralinguistic family covers emotion, pitch, timbre, age and gender. In contrast, the four content families ask about what is said. They differ in their options: time ranges for Temporal Localization, orderings of numbered events for Causal Alignment, and counts or outcomes across a long stretch for Event Tracking. Many questions first describe a scene in words and then ask about it, so the description serves to find the evidence and the question tests what is heard there.

(A) [2514, 2544] (B) [1289, 1315] (C) [2452, 2482] (D) [1338, 1366]

## Audio Event. 52.7-minute recording.

During the calm corner advice between rounds, as the coaches instruct their fighter to breathe, “catch some air,” and prevent his opponent from getting comfortable, what distinct background sound faintly intrudes upon the intimate broadcast audio?

(A) A low-volume, rhythmic electronic arena music beat

(B) The distant, metallic clanking of a ringside warning bell

(C) A faint, intermittent public address system announcement

(D) An organized, rhythmic crowd chant of the challenger’s name

Evidence: 14:16–14:45.

## Paralinguistic (emotion). 60.3-minute recording.

During the plot twist reveal about the successful audition, how does the friend’s vocal delivery of the seemingly supportive phrase, “Well, that’s good. Because you didn’t. And I’m incredibly happy for you,” reveal his true emotional subtext?

(A) He employs a warm, sincere tone, expressing genuine forgiveness and immediate relief after the cruel prank.

(B) He employs a loud, sarcastic tone, executing a reverse fake-out to playfully retaliate against his friend.

(C) He employs a breathless, panicked tone, revealing lingering anxiety over the initially misunderstood audition message.

(D) He employs a soft, mournful tone, indicating deep internal resentment despite the outwardly supportive spoken words.

Evidence: 57:58–58:00.

## Semantic Content. 59.9-minute recording.

During the mock game show challenge immediately following the playful corporate recruiting dispute, what specific physical impairments does the distressed competitor explicitly report experiencing while attempting the task? (A) Triggering electrical buzzer penalties and dropping the extracted target components into the restricted cavity.

(B) Losing grip stability on the required hand tools and slipping repeatedly on accidentally spilled chemical fluids. (C) Swallowing stray paint and failing to operate the interface buttons due to the bulkiness of protective gloves. (D) Experiencing severe physical dehydration and suffering visual hallucinations due to prolonged bodily exhaustion. Evidence: 24:24–24:47.

## Temporal Localization. 68.0-minute recording. Options are in seconds.

During the broader discussion analyzing how algorithmic social media captures the mind and isolates users from nuanced conversation, at what point do the speakers specifically draw a parallel between phone app engineering and slot-machine psychology?

Evidence: 22:18–22:46.

## Causal Alignment. 67.4-minute recording.

During the chaotic overtime phase on the second map, how does FURIA successfully establish numerical superiority and completely sweep the defensive site?

[1] A chaotic overtime scramble ensues, characterized by deep flank hunts.

[2] Vitality safely preserves their expensive sniper weaponry by retreating off-site.

[3] FURIA utilizes a deceptive smoke grenade fake to bait out the defenders.

[4] FURIA executes a lightning-fast, highly aggressive pathing maneuver.

[5] Vitality’s superstar players fail to firmly lock down the defense.

[6] FURIA establishes numerical superiority and ruthlessly sweeps the site.

(A) [4]→[5]→[1]→[6] (B) [5]→[3]→[4]→[6]

(C) [5]→[1]→[4]→[6] (D) [2]→[1]→[4]→[6]

Evidence: 41:05–42:38.

## Event Tracking. 47.0-minute recording.

In the early portion of the audio, during the characters’ extended debate over video game consoles and entertainment center space, how many specific past consumer devices are explicitly named as purchasing blunders or deep regrets? (A) 4 devices (B) 1 device (C) 2 devices (D) 3 devices Evidence: 01:54–04:14.

## A.3 LAT-Bench

Finally, a LAT-Bench query describes one moment by its speech, its speaking style and the surrounding sounds, and asks for its interval without options. The prediction is compared with the annotated interval by intersection over union.

## Temporal grounding query. 2.7-minute recording.

Please listen to the audio carefully and locate the first instance where the male speaker, in a guiding tone, provides instructions on how to handle the net immediately after the distinct sound of it splashing into the water. His speech begins with praise and then directs the woman to let the net sink before pulling it. Please strictly output the result in the format of [Start Time - End Time]. Do not output any extra explanatory text.

Annotated interval: 01:09–01:26.

## B Prompts

Jev-Style Scan. English: “Question: {question}”, the four options on separate lines, a blank line, and “Does this audio excerpt contain information that helps answer the question? Answer yes or no.” The question-only variant omits the options. The Mandarin prompt is a direct translation with a one-character answer.

Presence. English: “Does this audio excerpt contain {event}? Answer yes or no.” The Mandarin prompt is its direct translation and asks for a onecharacter answer. Here, the event phrase is the noun phrase that follows “This audio contains” in the Q1 stem. The score sums the probabilities of “yes”, “Yes” and the Mandarin affirmative, and of “no”, “No” and the two Mandarin negatives.

Routing. “A question about a long recording asks about this event: {event}. Which category describes the event best?” The eight options are: the playback speed suddenly becomes faster, the playback speed suddenly becomes slower, the volume suddenly becomes louder, the volume suddenly becomes quieter, a stretch of silence where the audio is missing, a stretch of noise that makes the speech unintelligible, a particular sound or sound effect that can be heard, and none of these. The answer is then the option letter with the largest logit.

Answering. The reader receives the selected excerpts as separate audio segments, together with the original prompt of the benchmark. One line per segment states its start and end as minutes and seconds of the original recording. For occurrence chains there is one 60-second segment. Native multiple-choice questions use three 20-second windows, with adjacent windows merged.

## C Further Details for Reproduction

Baselines. The learned retriever and candidate verification are our own baselines. The retriever presents eight regions of 4 to 8 seconds with their times in seconds. Candidate verification asks the reader a yes/no question about 16 candidates of the learned scorer. It answers Q1 with the option that matches its number of verified candidates and the time of its k-th verified candidate. For Q2 and Q3, it hears again at that candidate. The option rule chooses Q1 without listening. It takes the earlier option time when k = 1 and the later one otherwise, together with the smaller of the two option counts. For Q2 and Q3, the reader then hears the 60-second excerpt centred on the chosen time.

Evaluation Details. We rebuild the edited LongAudioSpan recordings with the official preparation script. On VoiceGiraffe, the time labels of the excerpts are in seconds, since its options give times in seconds. On LAT-Bench, the scan prompt has no options, since its queries have none. For grounding, the expanded positive-score region extends from the start of the first included window to the end of the last, as defined in Section 3.2. BM25 uses $k _ { 1 } = 1 . 5$ and $b = 0 . 7 5$ . The calibrators are logistic regressions with an L2 penalty of 1. The development numbers of the occurrence path use five-fold cross-fitting over recordings. Intervals come from

2,000 bootstrap resamples of recordings. Paired comparisons count the questions that only one of two systems answers correctly and apply a twosided sign test over questions or chains. The full recording with time marks is given as consecutive 30-second audio segments. Each segment is preceded by a text item with its start time, and one sentence before them explains the format. One 80 GB device holds at most about 62 minutes of audio in one input. The full-recording runs therefore divide the reader’s decoder layers between two devices at a layer boundary. The VoiceGiraffe results use the native 16 kHz waveforms of the benchmark, and answers without an option letter take the option with the highest log-probability. On VoiceGiraffe, the scan scores 16 windows per batch, whereas it scores 32 on LongAudioSpan and LAT-Bench. In BF16, the score of a window varies slightly with the shape of its batch. With 32 windows per batch, the VoiceGiraffe scan selects at least one different window for 952 of the 1,499 questions. ReListen then reaches 62.6% instead of 62.0%, a difference of 0.5 points with a 95% interval from −0.7 to 1.7. The LongAudioSpan matched-content comparison instead hears slices of the reader’s encoding of each whole recording, placed where the excerpt would stand. On all native questions, ReListen answers 54.7% from such slices and 55.1% from waveforms, a difference that is not significant.

Cost. All methods run on A100 80 GB GPUs. We exclude model loading and one warm-up question, and we synchronize the GPU at the start and end of each stage. Each text-mediated time includes transcribing the whole recording. All weights are in BF16, and the scan scores 32 windows per batch. When the full recording exceeds one GPU, the reader is divided over two GPUs, as in all fullrecording runs. The reader and Whisper run on one local GPU. The parallel scan, the two-GPU full recording, the vLLM transcription of Speaker-Reasoner and the official LAT-Audio runner run on cluster nodes with the same GPU type. On LongAudioSpan, a scan makes one reader call per window, 295 on average. On one A100 80 GB GPU, the reader scores about 24 windows per second. Its peak memory stays below 68 GiB at every duration, including 59 GiB of weights. On the 22 timed LongAudioSpan recordings, 61 minutes on average, the scan takes 15.0 seconds and the 60- second answer 0.4 seconds. The reader writes its transcript of these recordings in 116 seconds on average, and Whisper large-v3 in 24.7 seconds. The full recording fits one GPU up to 55 minutes on LongAudioSpan and 61.6 minutes on VoiceGiraffe, and it runs out of memory on every longer timed recording. Its forward pass takes 0.13 seconds per minute of audio at 10 minutes and 0.20 seconds at 62 minutes. On one GPU, ReListen-Fast is faster than the full recording on 12 of the 13 timed recordings between 30 and 62 minutes and slower on all 7 recordings under 30 minutes. Beyond 62 min utes, ReListen-Fast on one GPU is faster than the full recording on two GPUs on 10 of the 12 timed recordings. The two exceptions last over 2.5 hours, of which the full recording hears only the first 80 minutes. ReListen-Fast halves the scan and lowers its hit rate from 68.5% to 64.2% on development recordings (Section 5.2). Each GPU of the parallel scan scores whole batches of 32 windows, so short recordings give some GPUs fewer batches or none and gain less from four GPUs. The three timed questions on which the four-GPU scan is slower than the full recording are on 10-minute recordings, whose batches fill only two of the four GPUs. On every timed question, the four-GPU scan reproduces the one-GPU scores and selections for both hops. On one GPU, the cluster nodes take 5 to 8% longer than the local GPU, so the four-GPU speedups in Section 5.2 understate those within one node. LAT-Audio runs one of the ten timed LAT-Bench queries, on a 29.7-minute recording, over two GPUs. On LAT-Bench, the peak memory per GPU is 62.8 GiB for LAT-Audio and 64.8 GiB for the scan.

## D Comparisons with Closely Related Systems

We compare ReListen with the three closest relistening systems on shared questions and scoring.

LAT-Audio. We run the released LAT-Audio checkpoint (Shao et al., 2026) with its official runner on the same 1,002 queries as ReListen. Each metric is computed within a language, and the two languages are weighted equally. The released model reaches a macro average of 46.3 mIoU, with seven failed queries counted as zero. The authors report 48.6 under their original protocol, which averages over parsed outputs of 1,006 queries.

Echo-Style Frozen Adaptation. Echo (Wu et al., 2026) learns audio-interleaved reasoning with supervised fine-tuning and reinforcement learning on

Qwen2.5-Omni-7B. Its trained checkpoint is unavailable, so we pair its interface with the frozen Qwen3-Omni reader. The reader starts from its own whole, timestamped transcript and may request waveform excerpts while it reasons. We allow eight turns and at most 60 seconds of audio in total with greedy decoding. A paired control uses the same prompt and transcript but cannot receive audio. On VoiceGiraffe, the adaptation receives audio for only 22 of 1,499 questions, 413 seconds in total. Most of its requests repeat the placeholder times of the format instruction. Its 58.1% accuracy differs from the control’s 58.2% on three questions. On the 540 test LongAudioSpan multiple-choice questions, it receives audio for 13 questions, 400 seconds in total. It reaches 58.1% against 58.3% for the control, a difference of one question. Thus, with a frozen reader, the Echo interface rarely returns to the audio, and its accuracy is that of its transcript.

Speaker-Reasoner Transcript Frontend. We use the released bilingual Stage 2 checkpoint of Speaker-Reasoner (Lin et al., 2026) with its official prompts. This checkpoint lacks the Stage 3 cache that tracks speakers across chunks. We therefore transcribe each recording in 45-second chunks with speaker labels local to each chunk. Chunks that still fail after the official retries remain gaps in the transcript, 212 of 9,103 on VoiceGiraffe. Questions over gaps stay in the denominators. The frozen reader answers from this transcript with a 60,000- token input budget. On VoiceGiraffe it answers 835 of 1,499 questions correctly. On LongAudioSpan, the transcripts of 21 test recordings exceed the budget and are cut. The reader answers 286 of the 540 test multiple-choice questions correctly (53.0%).

## E Native Questions in Detail

We give the full native-question results behind Section 5.1, by option form, recording length and question dimension. The duration tiers are short (S, below 30 minutes), medium (M, 30–60 minutes) and long (L, above 60 minutes).

The Full Recording. The full recording gives the reader the first 80 minutes of every recording, as the benchmark’s authors do. The audio is one item, or 30-second segments with time marks. Over all 1,080 native questions, the reader answers 54.8% without time marks, where the benchmark’s authors report 51.5%. In the three duration tiers, it answers

<table><tr><td rowspan="2">Method</td><td rowspan="2">Audio</td><td colspan="8">Test</td><td rowspan="2">Development</td><td colspan="2"></td></tr><tr><td>All</td><td>Content</td><td>Time</td><td>S</td><td>M</td><td>L</td><td>Hit</td><td>All</td><td>Content</td><td>Time</td></tr><tr><td colspan="10">Text-mediated</td><td colspan="3"></td></tr><tr><td>Whisper transcript</td><td>none</td><td>58.1</td><td>67.5</td><td>35.4</td><td>58.2</td><td>58.7</td><td>57.5</td><td></td><td></td><td>54.8</td><td>65.8</td><td>30.1</td></tr><tr><td>Whisper text + ReListen</td><td>60 s</td><td>59.8</td><td>63.1</td><td>51.9</td><td>61.6</td><td>60.8</td><td>56.9</td><td>69.2</td><td></td><td>58.9</td><td>65.0</td><td>45.2</td></tr><tr><td colspan="10">Audio input</td><td colspan="3"></td></tr><tr><td>No audio</td><td>none</td><td>43.5</td><td>45.3</td><td>39.2</td><td>42.4</td><td>42.3</td><td>46.0</td><td></td><td></td><td>44.8</td><td>46.0</td><td>42.2</td></tr><tr><td>Uniform</td><td>60 s</td><td>45.0</td><td>49.0</td><td>35.4</td><td>46.9</td><td>42.9</td><td>45.4</td><td>18.7</td><td></td><td>44.1</td><td>49.7</td><td>31.3</td></tr><tr><td>Learned retrieval</td><td>60 s</td><td>44.6</td><td>50.3</td><td>31.0</td><td>43.5</td><td>44.4</td><td>46.0</td><td>32.6</td><td></td><td>45.7</td><td>50.8</td><td>34.3</td></tr><tr><td>BM25-selected audio</td><td>60 s</td><td>52.4</td><td>55.5</td><td>44.9</td><td>53.7</td><td>53.4</td><td>50.0</td><td></td><td>45.2</td><td>48.1</td><td>50.3</td><td>43.4</td></tr><tr><td>Full recording, time marks</td><td>≤80 min</td><td>54.6</td><td>59.9</td><td>41.8</td><td>62.1</td><td>57.1</td><td>44.3</td><td></td><td>97.1</td><td>58.0</td><td>62.8</td><td>47.0</td></tr><tr><td>ReListen, more audio</td><td>600 s</td><td>56.5</td><td>64.1</td><td>38.0</td><td>61.0</td><td>55.6</td><td></td><td>52.9</td><td>94.7</td><td>54.8</td><td>64.2</td><td>33.7</td></tr><tr><td>ReListen</td><td>60 s</td><td>53.5</td><td>57.3</td><td>44.3</td><td>55.4</td><td>55.6</td><td></td><td>49.4</td><td>69.2</td><td>56.7</td><td>60.4</td><td>48.2</td></tr><tr><td colspan="10">Reference</td><td colspan="3"></td></tr><tr><td>Annotated evidence</td><td>≤120 s</td><td>72.7</td><td>73.0</td><td>72.0</td><td></td><td>69.9</td><td>73.8</td><td>74.3</td><td>100.0</td><td>72.9</td><td>71.1</td><td>77.0</td></tr></table>

Table 5: Native multiple-choice results on LongAudioSpan. We report accuracy (%) by option type and recording length, and the evidence hit rate (%) of the audio heard. Whisper text + ReListen adds the selected minute to the Whisper transcript. Annotated evidence is scored on the 487 test questions with annotated support ranges.

58.1%, 56.7% and 49.7%. With time marks, it answers 56.3%. On the test half the two yield 54.4% and 54.6%, and 58.0% with time marks on development recordings. The chain score over all items is 16.1 without time marks, where the authors report 17.6, and 17.7 with them. With time marks, the full recording and ReListen do not differ significantly on the test half, 54.6% against 53.5%. Their order depends on recording length. On recordings under 30 minutes, the full recording leads, 62.1% against 55.4%. Beyond an hour, where recordings longer than 80 minutes are cut, ReListen scores higher, 49.4% against 44.3%. Neither difference is significant. The development half repeats both orders.

Results by Dimension. Table 6 splits the test questions by dimension. One selected minute helps most on recognition, tracking, interpretation and relational reasoning, by 17 to 21 points over no audio. However, it does not help on localization, where the options are clock times. Nevertheless, the oracle is higher there, since its excerpts are labelled with the times that the options name. Without time marks, the transcript falls to 34.6% on tracking, where most options are clock times, against 48.1% for ReListen. Hence, the benefit of the selected minute depends on the form of the options. On counting, the selected minute also falls below no audio, 36.5% against 48.1%. Counting needs every occurrence of an event, which selection by relevance does not guarantee (Section 3.3). Ten minutes raise counting to 46.2%. The full recording reaches 51.9%. The occurrence extension addresses this limit for planted events, but we do not apply it to native counting questions.

Transcript Systems. Whisper large-v3 transcribes each recording in consecutive 30-second chunks with greedy decoding and the language of the recording forced. Chunks whose text compresses by a factor above 2.4 are treated as repetition loops and dropped, 0.6% of all chunks. The transcript given to the reader joins the chunk texts in order, without time marks. The longest resulting input has 33,701 tokens, within the reader’s context, so no transcript is truncated. For BM25, the query is the question with its options, and the ranking is computed within the recording. Tokens are lower-cased words for English, and character unigrams and bigrams for Mandarin after conversion to simplified characters.

## F Questions about the Whole Recording

We give the details behind Section 5.6. The extent of a question’s evidence is the time from the start of its first annotated support range to the end of its last, as a share of the recording. On LongAudioSpan, 61 test and 72 development questions have an extent of at least one half. Their annotated evidence lasts 170 seconds at the median. However, 16.5% of them annotate more than ten minutes of evidence, against 3.7% of all annotated native questions. Table 7 gives their accuracy by budget and by duration tier.

On each half, the selected minute wins 4 of these questions against the full recording and loses 15, a significant difference. In contrast, ten minutes from the same scan win 3 and lose 4 on the test half, a difference that is not significant. On the development half they win 4 and lose 6, which is also not significant. The effect of ten minutes depends on the duration tier. On short recordings, ten minutes are at least a third of the audio and match the full recording. On medium recordings, they still trail it, 61.5% against 71.8%. On long recordings, where ten minutes are at most a sixth of the audio, they reach 75.8% against 72.7%. On the other 479 test questions, the selected minute and the full recording perform alike, 53.2% against 52.2%. On VoiceGiraffe, 31 questions have an extent of at least one half. ReListen answers 54.8% of them, against 35.5% for the full recording with time marks.

<table><tr><td></td><td></td><td></td><td colspan="5">Audio input</td><td colspan="2">Reference</td></tr><tr><td>Dimension</td><td>n</td><td>Time options</td><td>No audio</td><td>Uniform</td><td>ReListen</td><td>ReListen 600 s</td><td>Full recording</td><td>Transcript Oracle</td><td></td></tr><tr><td colspan="10">Perception</td></tr><tr><td>Recognition</td><td>48</td><td>0</td><td>58.3</td><td>56.3</td><td>75.0</td><td>77.1</td><td>68.8</td><td>72.9</td><td>79.5</td></tr><tr><td>Localization</td><td>80</td><td>78</td><td>43.8</td><td>40.0</td><td>40.0</td><td>31.3</td><td>38.8</td><td>41.2</td><td>73.2</td></tr><tr><td>Counting</td><td>52</td><td>0</td><td>48.1</td><td>34.6</td><td>36.5</td><td>46.2</td><td>51.9</td><td>48.1</td><td>55.3</td></tr><tr><td colspan="10">Understanding</td></tr><tr><td>Tracking</td><td>81</td><td>69</td><td>29.6</td><td>29.6</td><td>48.1</td><td>48.1</td><td>51.9</td><td>34.6</td><td>70.7</td></tr><tr><td>Interpretation</td><td>58</td><td>1</td><td>50.0</td><td>56.9</td><td>69.0</td><td>72.4</td><td>72.4</td><td>69.0</td><td>71.9</td></tr><tr><td>Summarization</td><td>41</td><td>7</td><td>51.2</td><td>48.8</td><td>58.5</td><td>65.9</td><td>53.7</td><td>78.0</td><td>64.7</td></tr><tr><td colspan="10">Reasoning</td></tr><tr><td>Causal</td><td>57</td><td>0</td><td>45.6</td><td>49.1</td><td>59.6</td><td>75.4</td><td>64.9</td><td>73.7</td><td>88.9</td></tr><tr><td>Logical</td><td>61</td><td>3</td><td>50.8</td><td>59.0</td><td>59.0</td><td>54.1</td><td>54.1</td><td>68.9</td><td>72.9</td></tr><tr><td>Relational</td><td>62</td><td>0</td><td>25.8</td><td>40.3</td><td>46.8</td><td>56.5</td><td>45.2</td><td>59.7</td><td>71.7</td></tr></table>

Table 6: Native accuracy by question dimension. We report test accuracy (%), with n questions per dimension and the number of them whose options are clock times. The full recording carries time marks, the transcript is from Whisper, and the oracle gives the reader the annotated support ranges.

<table><tr><td rowspan="2">Input</td><td colspan="2">Half</td><td colspan="2">Duration tier</td></tr><tr><td>Test Dev (61)</td><td>S (72) (61)</td><td>M (39)</td><td>L (33)</td></tr><tr><td>No audio</td><td>49.2 55.6</td><td>49.2</td><td>61.5</td><td>48.5</td></tr><tr><td>ReListen, 60 s</td><td>55.7 62.5</td><td>62.3</td><td>56.4</td><td>57.6</td></tr><tr><td>ReListen, 180 s</td><td>67.2 68.1</td><td>68.9</td><td>64.1</td><td>69.7</td></tr><tr><td>ReListen, 300 s</td><td>68.9 69.4</td><td>72.1</td><td>64.1</td><td>69.7</td></tr><tr><td>ReListen, 600 s</td><td>72.1 75.0</td><td>80.3</td><td>61.5</td><td>75.8</td></tr><tr><td>Full recording</td><td>73.8 77.8</td><td>80.3</td><td>71.8</td><td>72.7</td></tr><tr><td>Whisper transcript</td><td>68.9 79.2</td><td>75.4</td><td>71.8</td><td>75.8</td></tr><tr><td>Reader transcript</td><td>72.1 75.0</td><td>75.4</td><td>76.9</td><td>66.7</td></tr><tr><td>Annotated evidence</td><td>72.1</td><td>70.8 68.9</td><td>74.4</td><td>72.7</td></tr></table>

Table 7: Questions about the whole recording. We report native accuracy (%) on LongAudioSpan questions whose annotated evidence extends over at least half of the recording. Parentheses give the number of questions, and the duration tiers pool both halves. The full recording carries time marks, and the transcripts carry none.

## G VoiceGiraffe and LAT-Bench in Detail

We give the per-family VoiceGiraffe results and the LAT-Bench protocol behind Sections 5.1 and 5.3. We exclude one malformed Paralinguistic question with an extra empty option. The sound families are Audio Event (200) and Paralinguistic (349), and the content families are Semantic Content, Temporal Localization, Causal Alignment and Event Tracking. The options of Temporal Localization are time ranges. Evidence time ranges exist for 1,498 questions, 548 of them sound questions.

Table 8 gives the native-waveform VoiceGiraffe results by question family and language. We use 1,499 of the 1,500 questions, excluding one Paralinguistic item with a fifth, empty option. The Paralinguistic questions ask about the age, gender, emotion, pitch and timbre of a voice. For them, the benchmark’s authors report 50.6% for the same reader when it reads captions of 30-second windows. From native VoiceGiraffe waveforms, the full recording answers Temporal Localization questions at 33.2% without time marks and 48.0% with them. Without time marks, the reader’s transcript answers 38.4% of these questions. The selected minute with its scan-derived time labels reaches 77.6%, with or without the transcript. The recordings last 35 to 79 minutes, all within the 80 minutes that the full recording hears. ReListen exceeds the full recording with time marks, 62.0% against 51.4%.

On LAT-Bench, the uniform baseline is one 20- second window at the centre of the recording. The transcript baseline is the best 30-second chunk of a Whisper transcript under BM25. The expanded positive-score region reaches a mean IoU of 47.2 on the Chinese queries and 48.2 on the English ones. We also give the reader the whole recording and ask for the span, as the benchmark does. We parse the first two times of its answer as the span. In a second variant, the start time of every 30-second segment is given as text. Thus, on both benchmarks, the scan’s labelled windows exceed the full recording on temporal questions, even when the full recording carries time marks (Table 3).

<table><tr><td>Method</td><td>All</td><td>Aud. Ev.</td><td>Paraling.</td><td>Semantic</td><td>Temp. Loc.</td><td>Causal</td><td>Ev. Track.</td><td>En</td><td>Zh</td></tr><tr><td colspan="10">Text-mediated</td></tr><tr><td>Transcript by the reader</td><td>52.8</td><td>46.0</td><td>53.6</td><td>85.5</td><td>38.4</td><td>46.0</td><td>52.4</td><td>51.1</td><td>54.6</td></tr><tr><td>Transcript +ReListen, 60 s</td><td>63.0</td><td>55.5</td><td>61.0</td><td>86.0</td><td>77.6</td><td>49.6</td><td>52.4</td><td>60.5</td><td>65.6</td></tr><tr><td colspan="10">Audio input</td></tr><tr><td>No audio</td><td>34.5</td><td>35.5</td><td>38.4</td><td>32.0</td><td>34.0</td><td>38.0</td><td>27.2</td><td>33.9</td><td>35.1</td></tr><tr><td>Uniform, 60 s</td><td>40.7</td><td>46.0</td><td>49.9</td><td>38.5</td><td>36.4</td><td>45.2</td><td>25.2</td><td>40.9</td><td>40.5</td></tr><tr><td>Full recording</td><td>49.6</td><td>49.0</td><td>53.6</td><td>70.0</td><td>33.2</td><td>48.4</td><td>45.6</td><td>46.9</td><td>52.2</td></tr><tr><td>Full recording, time marks</td><td>51.4</td><td>49.5</td><td>52.4</td><td>67.0</td><td>48.0</td><td>46.8</td><td>47.2</td><td>48.7</td><td>54.2</td></tr><tr><td>ReListen, 60 s</td><td>62.0</td><td>54.5</td><td>61.0</td><td>87.5</td><td>78.0</td><td>47.2</td><td>48.0</td><td>59.7</td><td>64.4</td></tr><tr><td colspan="10">Reference</td></tr><tr><td>Annotated evidence</td><td>74.5</td><td>70.4</td><td>66.5</td><td>98.0</td><td>98.4</td><td>69.2</td><td>51.6</td><td></td><td>74.6 74.4</td></tr><tr><td colspan="10">Evidence hit rate</td></tr><tr><td>Uniform, 60 s</td><td>11.1</td><td>6.0</td><td>3.4</td><td>6.5</td><td>5.2</td><td>17.2</td><td>29.6</td><td>9.9</td><td>12.4</td></tr><tr><td>BM25, 60 s</td><td>58.8</td><td>41.7</td><td>47.9</td><td>63.0</td><td>66.0</td><td>68.4</td><td>67.6</td><td>57.0</td><td>60.6</td></tr><tr><td>Dense retrieval, 60 s</td><td>52.1</td><td>30.7</td><td>41.5</td><td>60.0</td><td>62.0</td><td>56.8</td><td>62.8</td><td>51.3</td><td>52.9</td></tr><tr><td>ReListen scan, 60 s</td><td>88.5</td><td>72.4</td><td>78.8</td><td>98.0</td><td>98.0</td><td>93.6</td><td>92.4</td><td>89.7</td><td>87.2</td></tr></table>

Table 8: VoiceGiraffe results by question family and language. We report accuracy (%) on all 1,499 questions and the evidence hit rate (%) on the 1,498 with annotated evidence. Transcript + ReListen adds the selected minute to the reader’s transcript. BM25 and dense retrieval rank chunks of the same transcript.

<table><tr><td>Variant</td><td>mIoU R@0.3 R@0.5 R@0.7</td><td></td></tr><tr><td>Best window only</td><td>41.8 67.3</td><td>35.7 13.8</td></tr><tr><td>Qwen2.5-Omni-7B reader</td><td>46.2 75.6</td><td>45.2 17.3</td></tr><tr><td>ReListen</td><td>77.2</td><td>50.0 19.2</td></tr></table>

Table 9: Grounding ablations on LAT-Bench. The first row returns only the highest-scoring window instead of the expanded region. The second row replaces the reader, Qwen3-Omni-30B-A3B, with Qwen2.5- Omni-7B. We report mean IoU and recall at IoU thresholds 0.3, 0.5 and 0.7 (%).

Table 9 gives two ablations of the grounding rule. The expanded region raises the mIoU of the best window by 5.9 points. With the smaller Qwen2.5- Omni-7B reader, the scan still reaches 46.2 mIoU.

## H Robustness of the Jev-Style Scan

We test whether the scan depends on the reader, and whether other locators can replace it.

A Second Reader. We repeated the multiplechoice experiment with Qwen2.5-Omni-7B (Xu et al., 2025) as scanner and answerer. It uses the configuration of the main reader. This reader takes at most five minutes of audio per input, so the full recording is not an option for it. Its scan places a support range in the selected 60 seconds for 61.2% of the annotated test questions. It answers 51.9% of the test questions, against 43.7% from uniform windows and 39.6% without audio. On development recordings the three accuracies are 48.9%, 43.0% and 39.3%. On the test half, the gain over uniform windows is 6.3 points with content options. With time options, it is 12.7 points. On LAT-Bench, the scan of this reader uses the same prompt and windows. Its best window overlaps the annotated span for 94.1% of the queries and reaches a mean IoU of 41.6. The expanded positive-score region reaches 46.2, with recalls of 75.6%, 45.2% and 17.3% at the three thresholds. By comparison, the main reader reaches 41.8 and 47.7.

A Contrastive Audio-Text Model as Scanner. We replaced the reader’s scan by CLAP (Wu et al., 2023), a zero-shot audio-text model, on the 92 English test recordings. On native questions, CLAP places a support range in the selected 60 seconds for 33.9% of the English test questions. The reader’s scan does so for 68.5%. On occurrence chains, however, the two scanners perform alike. Time accuracy is 50.0% with either evidence alone. When each is fused with the detectors, it is 77.2% with CLAP against 80.4% with the reader’s scan. Planted sounds and masks are therefore distinct enough for a small contrastive model. Questions about the content of natural recordings, however, need the reader’s understanding of the question.

A Dense Text Retriever. BGE-M3 (Chen et al., 2024), a multilingual dense retriever, ranks the same 30-second transcript chunks as BM25 for the same query. Its two best chunks overlap a support range for 47.8% of the annotated test questions and 47.5% of the development questions. BM25 reaches 45.2% and 46.5%, and the scan 69.2% and 68.5%. On native-waveform VoiceGiraffe, the same query and top-two-chunk policy uses the reader’s transcript from all 123 recordings. BM25 hits annotated evidence on 881 of 1,498 questions (58.8%) and BGE-M3 on 780 (52.1%), compared with 1,325 (88.5%) for the audio scan. On the sound families their hit rates are 45.6% and 37.6%, and on the content families 66.4% and 60.4%. BGE-M3 uses normalized CLS embeddings, cosine ranking and a 1,024-token encoding limit. Across these checks, no alternative locator matches the reader’s audio scan, in line with Section 5.3.

## I Scores along Recordings

Figure 6 shows the scan of a native question from a development recording, and Figure 7 shows the reader’s presence scores and the detector outputs for two chains. On the test question of Figure 1, the reader also answers 4 without audio and from the full recording, and 3 from the Whisper transcript.

![](images/f539f3354cb8553d58adcdda7a27498d395f60fe0020fad96bd60baf34571207.jpg)  
Figure 6: Scan scores along a recording. The scan of an 87-minute development recording for the question “According to the lecture, what is the direct mechanical cause of the lens flattening when shifting focus to a distant object?” Circles mark the three windows that ReListen selects, and the grey band marks the support range that the benchmark annotates for the question.

## J Reading and Listening

We detail how reading and listening compare when the reader receives its own transcript, the same excerpts as text, or the annotated evidence.

The Reader as Its Own Recognizer. The frozen reader transcribed every native recording in consecutive 30-second excerpts, one generation per excerpt and independent of any question. It then read its own transcript as text, with the prompt and the format of the Whisper transcript. It answers 59.6% of the test multiple-choice questions and 57.0% of the development questions. The Whisper transcript yields 58.1% and 54.8%, a difference that is not significant. The two transcripts are level on questions with content options, 67.2% against 66.7% over both halves. However, the reader’s transcript is ahead on questions with time options, 37.7% against 32.7%. The lead of a transcript on native questions therefore needs no second system.

Reading and Listening at Matched Content. We use two selections, the frozen scan’s three windows and the annotated evidence. For each question and selection, the reader transcribes exactly the selected excerpts. It then answers from that text without audio. An excerpt is transcribed in equal pieces of at most 30 seconds, and the text uses the same time label as the audio input. A second text input contains notes with the words and a description of voices and other sounds. The LongAudioSpan control uses the encoder once on each whole recording and selects slices from its outputs. The main VoiceGiraffe control instead crops the native waveform before encoding, consistently across the audio, transcript and notes inputs.

On LongAudioSpan questions without time options, listening and reading the scan’s windows yield 60.5% and 59.4% on the test half. On the development half, they yield 59.9% and 59.1%. From annotated evidence the corresponding values are 70.9% and 70.1%, and 71.7% and 71.7%. Neither difference on the test half is significant. With time options, text is ahead from the scan’s windows, 51.9% against 42.0% over both halves. It is also ahead from annotated evidence, 79.0% against 71.5%. The scan-window difference is 5.7 points on the test half and 13.8 on the development half.

On native-waveform VoiceGiraffe, audio and matched transcripts yield 62.0% and 59.6% from the scan’s excerpts. From annotated evidence, they yield 74.5% and 66.1%. On content questions the audio-minus-transcript differences are −1.2 points for the scan and 2.0 points for annotated evidence. On sound questions the differences are 8.6 and 19.5 points, and both are significant. For annotated Audio Event questions, audio reaches 70.4% against 39.7% for the transcript. For Paralinguistic questions, it reaches 66.5% against 53.3%. Notes reach 57.7% on sound questions from the scan’s excerpts and 58.2% from annotated evidence, compared with audio at 58.7% and 67.9%. Over all questions, scan notes reach 62.0%, almost identical to scan audio. These results distinguish losses in ordinary speech transcripts from the information that richer text descriptions can preserve.

![](images/66a7282bd5587c8aa43639f2949b7edb4b69657834bc822859dde5c6b2d1a70a.jpg)

![](images/e147e68268e8a3ace596a93bdb65ff33a9c293aa24f4d156e897dd6713403973.jpg)  
Figure 7: Reader and detector evidence on planted events. On two development recordings, the reader’s presence score rises at each inserted sound but stays flat at each volume edit, whereas the acoustic detectors respond to both. Grey bars and triangles mark the true occurrences, and the circle marks the occurrence that ReListen selected.

With Whisper text on LongAudioSpan, the two best BM25 chunks yield 51.5% as text and 52.4% as audio on the test half. Whisper chunks overlapping the scan’s windows provide a superset of the selected audio content. They yield 57.6% against 53.5% for audio on the test half.

The Gap to Annotated Evidence by Scan Hit. On the test half of LongAudioSpan, the questions that the scan misses account for 9.9 of the 19.7 points between ReListen and the annotated evidence. On the development half, they account for 8.9 of 16.3. The questions that the scan hits account for 9.9 and 7.5. The misses thus account for about half of the gap. However, more audio does not close the gap on the hits. Consider the 484 questions without time options whose support range the selected minute overlaps. On them, the reader answers 64.3%, 63.0% and 64.7% from 60, 180 and 600 seconds over both halves (62.0%, 60.7% and 63.2% on the test half). These inputs hold 38%, 55% and 74% of the annotated seconds, and the evidence alone yields 72.9%. On the 203 such questions that the minute misses, however, the three budgets yield 45.3%, 52.2% and 59.1%. Here, ten minutes win 40 questions against the minute and lose 12. The evidence yields 70.0% on them. Over all questions without time options, ten minutes yield 64.2% over both halves, against 67.2% for the reader’s transcript and 65.1% for ten minutes of BM25 text. On these questions, coverage therefore closes most of the transcript’s lead. By comparison, the reader’s transcript leads the selected minute on the test half by 10.0 points on the 382 questions without time options. On the 158 questions with time options, the minute leads by 3.2 points.

![](images/8d0fc507522b8b55a40e46eb93265be2030bc53901bac67435b72539a7890edb.jpg)  
Figure 8: Accuracy against the evidence heard. Native multiple-choice questions with content options on both halves of LongAudioSpan, with the scan at 60, 180 and 600 seconds pooled and grouped by the share of the annotated support seconds that the selected audio holds.

Figure 8 relates accuracy to the share of the annotated support seconds that the selected audio holds. Without any annotated second the reader answers 44.6% of these questions, with up to a quarter of them 60.0%, and with more 63.6% to 68.8%. Fully covered ranges yield only 60.1%. They are short, 13 seconds at the median against 78 seconds for partly covered ones, and come from a different mix of question types. Hence, the largest step is from hearing none of the evidence to hearing part of it. More of the evidence adds little. The same holds within a question. From 60 to 600 seconds, among the questions that the minute misses and ten minutes cover by at least half, 26 become correct and 6 become wrong. However, among the questions that the minute already hits and whose coverage rises by at least half, 15 become correct and 12 become wrong. Therefore, the remaining gap on the hits does not come from too little evidence in the input. It points instead to the reader’s use of the evidence it hears.

## K Occurrence Extension in Detail

We specify the candidates, the detectors, the calibration and the posterior of the occurrence extension in Section 3.3.

## K.1 Candidates

The candidates are the highest-scoring windows of the scan together with the regions flagged by two small acoustic detectors. The detectors are convolutional networks with under one million parameters each that read the waveform. The reader’s scores separate inserted sounds and masks from other windows, with areas under the ROC curve of 0.93 to 0.98 on development recordings. However, they carry little information about changes of speed (0.79) and volume (0.57), so we add the detectors. We train the detectors on unrelated recordings augmented with the benchmark’s public edit operators (Appendix K.2). For each of four event groups, we fit a logistic model on development recordings to predict whether a candidate matches an annotated edit. This calibration combines reader and detector features into a presence probability $p _ { j } \left( { \mathsf { A p - } } \right.$ pendix K.3).

## K.2 Acoustic Detectors

Inputs and Architecture. The speed and volume detector has 0.63M parameters. It reads eight band energies, the broadband energy and an onsetstrength channel, and predicts volume increase, volume decrease, speed-up and slow-down. The content-edit detector has 0.71M parameters and reads 40 log-mel bands. It is trained on all seven edit classes and supplies the probabilities of inserted sound, silence mask and noise mask. Each is used as an ensemble of three seeds, 4.0M parameters in total. Four strided convolutions with kernel sizes 7, 5, 5 and 4 reduce 100 frames per second to one. The intermediate channels are 64, 96 and 128 for the 40-band input and 48, 64 and 96 for the 10-channel input. Eight residual blocks of width 128 follow. Each has a dilated convolution of kernel size 3, batch normalization and a point-wise convolution, with dilations 1, 2, 4, 8, 16, 32, 64 and 1 across the blocks. A point-wise convolution then gives one logit per class and second.

Training Data. The detectors’ classes are the benchmark’s edit types. Their training edits are produced with the benchmark’s public preparation code on unrelated recordings. These are meetings from AMI (Carletta et al., 2005) and AISHELL-4 (Fu et al., 2021), read speech from LibriSpeech (Panayotov et al., 2015), the 30 podcast episodes in the full Rev.AI collection<sup>1</sup> and talks from TED-LIUM (Hernandez et al., 2018). Each training clip is a crop of 8 to 15 minutes with zero to four edits whose parameters are drawn from broad ranges. These are gains of 5 to 14 dB, tempo factors of 0.33 to 0.67 and 1.5 to 3, inserted sounds at 3 to 14 dB above the local level, and masks of 5 to 35 seconds. Inserted sounds come from ESC-50 (Piczak, 2015) and from procedurally generated tones, chirps and pulse trains.

Two choices make the detectors transfer to the benchmark recordings. The first is background diversification: dynamic-range compression, noise gating to digital silence, additive noise, low-pass filtering, silence removal and peak normalization. Digital silence occurs in many benchmark recordings and in almost no meeting recording. The second is negative examples that resemble edits: a segment replaced by another recording at a different level, and a level change with slow ramps. Level features are expressed relative to the 90th percentile of the clip and clamped 70 dB below it. This keeps the operating point fixed under a global gain or digital silence. Inference runs on 400-second crops with 200-second hop, matching the training length.

Training. The loss is binary cross-entropy with a positive weight of 3, summed over all valid secondclass pairs of a batch and divided by their number. Seconds that an edit covers only partly are excluded. We train with AdamW, a one-cycle schedule with peak learning rate $2 \times 1 0 ^ { - 3 }$ , weight decay 0.01, batches of 32 crops of 400 seconds and gradient clipping at 1. The speed and volume detector trains for 4,000 steps on 5,000 clips (927 hours of rendered audio). The content-edit detector trains for 6,000 steps on 6,000 clips (1,121 hours). The clips are repeated crops of the source recordings. A random global offset is added to the level channels of each crop. Checkpoints are selected by the loss on synthetic clips built from held-out sessions, speakers and sound effects.

Synthetic Edits. Volume edits multiply a region of 6 to 35 seconds by a gain of 5 to 14 dB in either direction and clip at full scale. Speed edits apply the atempo filter of FFmpeg with a factor of 1.5 to 3 or 0.33 to 0.67 to a region whose output lasts 3 to 70 seconds. Inserted sounds last 0.8 to 9 seconds and are mixed 3 to 14 dB above the level of the region they cover. Silence masks set 5 to 35 seconds to zero. Noise masks replace 5 to 35 seconds by a copy that is low-passed at 200 to 600 Hz and attenuated by 3 to 12 dB. Pink, white or brown noise is added, with fades of 0.2 to 1 second. Sixty percent of the clips use one edit type for all regions and one repeated sound effect, as the benchmark does, whereas the rest mix types. Forty-five percent of the backgrounds are podcasts and talks, twenty percent read speech and the rest meetings.

## K.3 Candidate Features and Calibration

The reader classifies the event phrase, without audio, into an edit class, which selects the detector output to read. Candidates are then the union of two sets. The first is the 24 windows with the highest scores $\ell _ { i }$ whose centres are at least 30 seconds apart. The second is the regions where the routed detector probability exceeds 0.2. A reader peak within 15 seconds of a detector region is merged into it. However, for speed and volume events the candidates are the detector regions alone, since the reader’s scores carry little information about them. A direct question to the reader about loudness does no better, with AUCs of 0.56 to 0.66 on development recordings. Each candidate has a feature vector with reader features and detector features. The reader features are the score ℓ of the best window covering the candidate, the larger score of its two neighbours, and the score relative to the 90th percentile and to the maximum of the recording. The detector features are the mean probability in the region and the region length capped at 8 seconds. Volume candidates add the smaller of the two level steps at the region boundaries. The logistic models are fitted on the development recordings, with cross-fitting for the development numbers. Appendix M measures how many annotated recordings they need.

## K.4 Occurrence Posterior

Let the candidates be sorted by time, $t _ { 1 } < \cdots <$ $t _ { m }$ , with presence probabilities $p _ { 1 } , \ldots , p _ { m }$ that we treat as independent. For candidate $j$ to be the k-th occurrence, it must be real and exactly $k - 1$ earlier candidates must be real. Thus its probability is

$$
\begin{array} { r } { \pi _ { j } ^ { ( k ) } = p _ { j } \cdot \mathrm { P r } \left[ \sum _ { i < j } z _ { i } = k - 1 \right] , } \end{array}\tag{3}
$$

with $z _ { i } \sim$ Bernoulli $( p _ { i } )$ . Here the second factor is a Poisson-binomial probability (Hong, 2013). One pass of dynamic programming computes it for all $j$ and also gives the distribution of the count $N = \textstyle \sum _ { j } z _ { j }$ . The extension then selects $t ^ { \star }$ as the time of the candidate with the largest $\pi _ { j } ^ { ( k ) }$ . By comparison, verification pipelines keep the candidates judged present and take the k-th in time. One weak false alarm or miss before the target then shifts the answer. For Q2 and Q3, the reader hears one 60-second window centred on $t ^ { \star }$ and answers the original question. For Q1, each option names a count n and a time t. ReListen scores an option by $\begin{array} { r } { \operatorname* { P r } [ N = n ] \cdot \sum _ { j : | t _ { j } - t | \le 4 5 \mathrm { s } } \pi _ { j } ^ { ( k ) } } \end{array}$

## L Occurrence Chains: Both Halves and Edit Types

Table 10 gives the chain results of Section 5.7 on both halves. Table 11 separates the test results by edit type, and the development half shows the same pattern. In the first group of Table 10 the reader answers Q1 itself, except for candidate verification and ReListen, which answer it from their enumeration. Reader’s Q1 choice hears again at the time of the option that the reader selected for Q1. Time accuracy is the fraction of chains for which the selected time lies within 30 seconds of the midpoint of the k-th edited region. On the chains ReListen exceeds the option rule by 25 points. On all 180 test chains, the full recording reaches a chain score of 16.5, and 17.2 with time marks, against 60.7 for ReListen.

Remaining Errors. On the test half, 34 of 180 chains receive a wrong time. Of these, 17 have inserted sounds, 9 volume changes, 3 each silence masks and speed changes, and 2 noise masks. In 19 of them ReListen selects a false alarm, in 13 it selects a different true occurrence, and in 2 no candidate lies at the target. The dominant failure is thus an ordinal error among detected candidates. Q3 reaches 56.1% at the oracle time. Hence, a large share of the remaining chain errors comes from reasoning about content beyond the window, the same limit that the native questions show.

<table><tr><td rowspan="2">Method</td><td colspan="4">Test</td><td colspan="4">Development</td></tr><tr><td>Q1</td><td>Q2</td><td>Q3</td><td>Chain</td><td>Q1</td><td>Q2</td><td>Q3</td><td>Chain</td></tr><tr><td colspan="9">Locating without the Q1 options</td></tr><tr><td>No audio</td><td>16.1</td><td>39.4</td><td>29.4</td><td>8.3</td><td>25.6</td><td>32.2</td><td>28.3</td><td>13.7</td></tr><tr><td>Uniform 60 s</td><td>22.8</td><td>28.9</td><td>34.4</td><td>10.0</td><td>33.3</td><td>28.9</td><td>27.8</td><td>15.0</td></tr><tr><td>Learned retrieval</td><td>18.3</td><td>39.4</td><td>32.8</td><td>9.6</td><td>27.2</td><td>42.8</td><td>27.2</td><td>13.5</td></tr><tr><td>Whisper transcript</td><td>22.8</td><td>29.4</td><td>32.8</td><td>10.2</td><td>31.1</td><td>23.3</td><td>26.1</td><td>13.7</td></tr><tr><td>Candidate verification</td><td>55.0</td><td>51.7</td><td>36.7</td><td>37.4</td><td>53.3</td><td>53.3</td><td>31.1</td><td>36.9</td></tr><tr><td>ReListen</td><td>78.3</td><td>75.6</td><td>50.6</td><td>60.7</td><td>75.6</td><td>72.8</td><td>45.6</td><td>56.1</td></tr><tr><td colspan="9">Using the Q1 options</td></tr><tr><td>Option rule, no listening to locate</td><td>47.2</td><td>72.2</td><td>47.8</td><td>35.7</td><td>43.9</td><td>71.1</td><td>41.7</td><td>33.9</td></tr><tr><td>Reader&#x27;s Q1 choice</td><td>18.3</td><td>58.9</td><td>42.2</td><td>12.0</td><td>27.2</td><td>61.7</td><td>41.1</td><td>18.7</td></tr><tr><td>ReListen, chain-consistent</td><td>78.3</td><td>81.7</td><td>53.9</td><td>62.8</td><td>75.6</td><td>80.6</td><td>47.8</td><td>59.3</td></tr><tr><td colspan="9">Reference</td></tr><tr><td>Oracle time, smaller count</td><td>67.2</td><td>88.9</td><td>56.1</td><td>52.2</td><td>63.3</td><td>87.8</td><td>51.1</td><td>48.9</td></tr></table>

Table 10: Occurrence chains on both halves. We report accuracy (%) on each question and the chain score on 180 chains per half. Every audio selection totals 60 seconds, and the transcript is from Whisper. The chain-consistent row answers Q2 and Q3 at the time that ReListen chose for Q1.

<table><tr><td></td><td>Ins. Noise (60) (31)</td><td>Sil. (29)</td><td></td><td>Speed Vol. (26) (34)</td></tr><tr><td colspan="5">Time accuracy</td></tr><tr><td>Reader evidence only Detector evidence only</td><td>61.7 35.0</td><td>90.3 82.8 87.1 82.8</td><td>3.8 88.5</td><td>2.9 73.5</td></tr><tr><td>ReListen</td><td>71.7</td><td>93.5</td><td>89.7 88.5</td><td>73.5</td></tr><tr><td colspan="3">Q2 accuracy</td><td></td><td></td></tr><tr><td>Candidate verification</td><td>56.7</td><td>77.4 51.7</td><td>38.5</td><td>29.4</td></tr><tr><td>ReListen</td><td>73.3</td><td>80.6</td><td>69.0 80.8</td><td>76.5</td></tr><tr><td>Oracle time (reference)</td><td>95.0</td><td>87.1</td><td>72.4 88.5</td><td>94.1</td></tr></table>

Table 11: Occurrence chains by edit type. We report time accuracy and Q2 accuracy (%) on the test half. Parentheses give the number of chains of each edit type.

## M Ablation of the Occurrence Path

Table 12 removes one element of the occurrence path at a time. The acoustic detectors matter most: without them time accuracy falls by 31 points on the test half. By comparison, without the reader scan it falls by 14 points.

Calibration and Its Labels. Calibration is the third largest factor. A fusion without benchmark annotation takes the noisy-OR of the reader’s raw yes-probability and the detector probability. It reaches 60.0% time accuracy, 21 points below the calibrated system. However, the calibrators need few annotated recordings. Fitted on 12 development recordings drawn at random, they reach 72.8 ± 3.2% time accuracy on the test half over 20 draws, and 78.5 ± 1.6% on 96 recordings. Fitted on the test half and applied to the development half, they yield 81.7%, against 80.6% with cross-fitting.

<table><tr><td colspan="2"></td><td colspan="2">Test Dev</td></tr><tr><td>Variant</td><td>Windows Time</td><td>Q1 Time</td><td>Q1</td></tr><tr><td>Removing one component</td></tr><tr><td>Reader evidence only 289</td><td>50.6 51.1 54.4 53.3</td><td></td><td>71.1</td></tr><tr><td>Detector evidence only 289</td><td>66.7</td><td>71.1 72.2</td><td>59.4</td></tr><tr><td>No calibration labels Hard ordinal rule</td><td>289 289</td><td>60.0 58.3 80.0 72.8</td><td>59.4 79.4 71.7</td></tr><tr><td>Single detector seed</td><td>289 75.7</td><td>71.9</td><td>81.3 72.2</td></tr><tr><td colspan="4">Scoring fewer windows</td></tr><tr><td>20 s hop</td><td>145 92</td><td>81.1 77.8 79.4</td><td>80.6 75.6</td></tr><tr><td>Retriever-pruned scan</td><td>48 76.7</td><td>76.1</td><td>81.1 76.1 75.6</td></tr><tr><td>Retriever-pruned scan Retriever-pruned scan</td><td>16 74.4</td><td>73.3 70.6</td><td>78.9 77.2 73.3</td></tr><tr><td>reader only, hard rule</td><td>16</td><td>45.0 43.3</td><td>49.4 48.9</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>ReListen</td><td>289</td><td>81.1 78.3</td><td>80.6 75.6</td></tr></table>

Table 12: Ablation of the occurrence path. We report time accuracy and Q1 accuracy (%). Windows is the mean number of windows that the reader scores per test recording. The detector-only row falls back to the reader’s scan when the detectors propose no region. The single-seed row averages three seeds.

The Ordinal Posterior. Replacing the posterior by the hard rule leaves time accuracy within one point but lowers Q1 by 4 to 6 points on each half. Pooled over both halves the posterior wins Q1 on 38 chains and loses on 21, a significant difference. This gain comes from scoring each Q1 option by its count and its location together. The posterior’s most probable count alone is about as accurate as the count of the hard rule, 66.1% against 68.3% on

the test half.

Ensemble and Cost. A single seed yields 81.3% time accuracy on development recordings and 75.7% on the test half. Doubling the hop to 20 seconds halves the reader’s work and changes Q1 on five test chains, three for the worse and two for the better. Restricting the scan to the windows ranked highest by the learned retriever trades accuracy for cost. With 16 windows per recording, time accuracy on the test half is 74.4% with the detectors. It is 45.0% in the configuration that corresponds to candidate verification.

## N Occurrence Path beyond Planted Edits

Unedited Recordings. The detectors could in principle respond to natural events in the recordings and not to the edits. To check this, we ran the detector branch of the configuration on the unedited version of every recording with the event of its chain. The expected number of occurrences is 0.52 per development recording and 0.56 per test recording on the unedited versions. On the edited versions it is 1.76 and 1.80, where the true mean is 1.94. Similarly, a candidate with a calibrated probability above one half exists in 17% and 15% of the unedited recordings and in 80% and 83% of the edited ones.

A Natural Event. The AMI meeting corpus (Carletta et al., 2005) annotates coughs. We built 119 questions over 24.5 hours about the k-th cough for the 49 meetings with one to four occurrences. Coughs closer than 30 seconds count as one occurrence. We then ran the reader’s presence scan with the phrase “a person coughing”, without detectors. The window-level AUC of the scan is 0.88. With the hard rule on decoded answers, the selected time is within 30 seconds of the k-th cough for 37.0% of the questions, against 3.8% for a random time. The count is exact for 26.5% of the meetings. By comparison, the calibrators fitted on the benchmark’s inserted sounds and the posterior yield 36.1% and 26.5%. Calibrators cross-fitted on the meetings yield 37.8% and 24.5%. The scan thus finds a natural sound far above chance, whereas calibration and the posterior add nothing measurable at this size.