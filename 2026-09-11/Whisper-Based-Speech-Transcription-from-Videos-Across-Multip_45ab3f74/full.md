# Whisper-Based Speech Transcription from Videos Across Multiple Languages for Cross-Cultural Understanding

Michael Picheny

NYU Courant Institute School of Mathematics, Computing, and Data Science

NYU

New York, USA

map22@nyu.edu

Abstract—Cross-cultural understanding has become increasingly important in today’s highly connected, cross-national world. The success of LLM-based technologies is now driving the development of automated tools to aid understanding for nonnative people trying to succeed in cross-cultural environments. Building such automated tools is often done by leveraging in-thewild text, audio, and video data. This paper presents techniques for improving speech recognition-based transcript creation in multiple languages from videos to better train these automated tools. The focus is on processes and speech tools that can easily be used by cross-cultural tool builders without requiring deep speech processing expertise. Using publicly available videos from YouTube and Whisper-based tools, average transcription error rate across seven languages (Spanish, Japanese, Korean, Mandarin, Turkish, Russian, and Hebrew) of 30% are observed. With a modest amount of fine-tuning data, the average error rate can be reduced to 20% making such output much more usable for downstream processing. Speech and metadata associated with these videos that can be used by the community to further refine these experiments are released as well.

Index Terms—speech recognition, whisper, video transcription, cross-cultural understanding

## I. INTRODUCTION

Cross-cultural understanding has become increasingly important in today’s highly connected, cross-national world. The success of LLM-based technologies is now driving the development of automated tools to aid understanding for nonnative people trying to succeed in cross-cultural environments. Building such automated tools is often done by leveraging analyses of multimodal data, including text, audio, and video data [1]–[5].

One way to leverage audio data for such purposes is to convert the audio data into text data and then utilize text-based LLM-based tools to extract relevant cultural information (see [6] for a survey of such tools). The degree to which audio data can be leveraged depends on the performance of the extraction process. Deep learning has greatly improved speech recognition performance over the past several years. Opensource leaderboard results [7] might suggest to the casual user of speech recognition that word error rates are now significantly below 10% across a variety of tasks, and therefore not impact research on cultural understanding. However, these tasks tend to contain prompted and/or stylized speech from professional speakers and are recorded in relatively benign environments. For cultural understanding, the focus is on information extraction from naturally produced speech ”inthe-wild”. Extracting accurate transcripts from such realistic audio data can still be challenging, especially for languages that are not as data-rich as English. Accuracy can be improved by building custom systems leveraging thousands of hours of data, but such approaches are not often feasible for researchers interested in cross-cultural understanding. Such researchers do not tend to be speech experts and would prefer easy, inexpensive methodologies for extracting transcripts from audio. Finally, because cultural markers are low-density with respect to hours of audio, thousands of hours of audio need to be analyzed to produce an adequate amount of data to create downstream cultural understanding models, making processing time and cost an important practical issue.

Given the increasing interest in research in developing automatic tools for cross-cultural understanding, and that an important pre-processing step is the extraction of audio transcriptions from large amounts of multimodal, multilingual data, this study was initiated with the following three goals. The first was to estimate open source speech transcription accuracy for data relevant to cross-cultural understanding across a variety of languages. The second was to outline a process that non-speech researchers with limited resources could realistically use to collect data and build usable speech transcription systems for cultural understanding. The third goal was to share representative speech data with the community to spur new research to obtain further performance improvements.

Given ease-of-use, processing time, and cost constraints Whisper-based [8] speech transcription generation was utilized. A flavor of Whisper developed at the Univerity of Oxford called WhisperX [9] seemed particularly well suited for this purpose. Whisper itself is a high-performing open source speech recognition system with good multilingual performance and fine tuning capabilities. It has an easy to use API. WhisperX incorporates a number of significant speedups to basic Whisper processing and also is capable of speaker diarization (important for conversation analysis). It also has an easy to use API.

The rest of this paper describes the detailed methodology developed that others could potentially copy and use to extract transcripts in large quantities from long-form speech. Section II-A describes the choice in recognizers, Section II-B, the choice of languages, Section II-C, the processing pipeline, Section II-D, the selection of videos, Section III, out-of-thebox and fine tuning experiments, Section IV, discussion of results including usability implications, and Section V, an overview of the data and metadata release.

## II. DATA PROCESSING

## A. Recognizer Choice

Whisper-based processing was chosen because of an easy to use API and excellent out-of-box performance. Whisper was trained on 680,000 hours of audio. 563,000 hours are English, and 117,000 hours cover 96 other languages [8]. At one point in time, 2000 hours of speech was viewed as an enormous amount of training data. 13 of the 96 languages contain at least 2000 hours of audio, so good out-of-box Whisper performance is not unexpected for these languages. Unfortunately, no details are given about the sources of the data.

Both Whisper and a derivative (”WhisperX”) were evaluated. WhisperX was developed at the University of Oxford. It not only performs Speech Recognition but also performs speaker diarization by incorporating an open-source modular package for diarization (Pyannote [10], [11]). Pyannote comes with a set of pretrained models for English but can be trained from scratch for other languages as well. For research on cultural understanding, speaker diarization is a key feature as it is needed for identification of speaker turns in a conversation. WhisperX also contains a sped-up version of Whisper. The speedup is essential for processing the thousands of videos needed to extract enough information to train downstream models for cultural understanding. All ou experiments utilized the ”large-v2” Whisper model ( [9] claims WhisperX is 10x faster than Whisper for the large-v2 model). For some languages, the ”large-v3” models might have had improved performance, but Github discussions (e.g., [12]) suggested that large-v3 might be more prone to hallucinations for noisy data.

## B. Language Data

The choice of languages to study was inspired by the languages selected for the DARPA Computational Cultural Understanding program (CCU) [13] (Mandarin, Spanish, Korean, Japanese, Russian, and Turkish). In CCU, speech recognition was needed to extract transcripts from audio and video data. Since speech recognition in itself was not a CCU focus no manual transcripts were supplied. It was therefore not possible to evaluate speech recognition performance.

TABLE I  
SIZE OF WHISPER TRAINING DATA FOR THE SELECTED LANGUAGES.
<table><tr><td>Language</td><td>Hours</td></tr><tr><td>Mandarin</td><td>23446</td></tr><tr><td>Spanish</td><td>11000</td></tr><tr><td>Russian</td><td>9761</td></tr><tr><td>Japanese</td><td>7064</td></tr><tr><td>Korean</td><td>7793</td></tr><tr><td>Turkish</td><td>4333</td></tr><tr><td>Hebrew</td><td>688</td></tr></table>

The 6 CCU languages were each represented by over 4000 hours of speech (Table I) in Whisper training data, so good out-of-box performance seemed to be a reasonable expectation. Hebrew was added as a challenge language because its coverage in Whisper relative to the other six languages is significantly less (688 hours). This is large enough to expect decent performance, but its behavior relative to the other languages before and after fine-tuning might be different. In addition, because it was familiar to the author, it made it easier to debug the various tools and pipelines needed to process the data.

## C. Processing Pipeline

A significant fraction of the CCU data came from YouTube. This suggested the following data collection process. The YouTube platform processes HTTP GET requests where state or filter criteria are passed via query parameters [14] and search parameters [15] within the Uniform Resource Identifier (URI). One can use these parameters to specify search keywords, and also locate videos that match particular criteria, such as restricting videos to the current year, month, week, or day, videos with subtitles, and videos provided under a Creative Commons license. Third parties have developed easyto-use command-line interfaces to access YouTube data. ytdlp [16] leverages the above mechanism to search for videos, return content, and help interpret the returned content in a user friendly fashion. The Jtubespeech repository [17] is a set of tools built on top of yt-dlp that make it easy to collect large amounts of Youtube-based speech data for specific languages.

The supplied CCU videos from Youtube (provided via YouTube video ids) were filtered using the Jtubespeech and ytdlp tools to select those videos with manually created subtitles and provided with a Creative Commons license. Three of the languages (Mandarin, Korean and Turkish) had little or no videos that satisfied these criteria. For these videos, and for Hebrew, processing was started from scratch. The time period for crawling for new videos was restricted to the previous year (mid 2024- mid 2025) except for Mandarin, where too many videos were being produced, so the time period was restricted to April, 2025 (Note the ”large-v2” Whisper model was released in early 2023).

TABLE II  
CONSOLIDATED LANGUAGE METADATA (DURATION IN HOURS)
<table><tr><td>Size</td><td>Type</td><td>Stat</td><td>Man.</td><td>Spa.</td><td>Rus.</td><td>Jap.</td><td>Kor.</td><td>Tur.</td><td>Heb.</td></tr><tr><td>All</td><td>Train</td><td>Num Videos</td><td>127</td><td>20</td><td>186</td><td>82</td><td>247</td><td>740</td><td>135</td></tr><tr><td>All</td><td>Train</td><td>Num Segs</td><td>57819</td><td>9063</td><td>72979</td><td>10465</td><td>48319</td><td>205018</td><td>61210</td></tr><tr><td>All</td><td>Train</td><td>Duration</td><td>50.35</td><td>8.45</td><td>117.30</td><td>31.01</td><td>53.58</td><td>207.22</td><td>63.88</td></tr><tr><td>All</td><td>Dev</td><td>Num Videos</td><td>10</td><td>3</td><td>6</td><td>8</td><td>13</td><td>29</td><td>8</td></tr><tr><td>All</td><td>Dev</td><td>Num Segs</td><td>3738</td><td>381</td><td>1142</td><td>1007</td><td>1679</td><td>8207</td><td>5266</td></tr><tr><td>All</td><td>Dev</td><td>Duration</td><td>3.37</td><td>0.31</td><td>3.57</td><td>2.18</td><td>1.68</td><td>8.36</td><td>5.18</td></tr><tr><td>All</td><td>Test</td><td>Num Videos</td><td>21</td><td>10</td><td>12</td><td>11</td><td>23</td><td>58</td><td>13</td></tr><tr><td>All</td><td>Test</td><td>Num Segs</td><td>7572</td><td>1713</td><td>7464</td><td>1354</td><td>3391</td><td>15317</td><td>6758</td></tr><tr><td>All</td><td>Test</td><td>Duration</td><td>7.65</td><td>1.58</td><td>8.23</td><td>5.72</td><td>4.21</td><td>15.10</td><td>7.44</td></tr><tr><td>Small</td><td>Train</td><td>Num Segs</td><td>5782</td><td>9063</td><td>6999</td><td>3488</td><td>8053</td><td>10250</td><td>6121</td></tr><tr><td>Small</td><td>Train</td><td>Duration</td><td>5.01</td><td>8.45</td><td>11.04</td><td>10.35</td><td>8.98</td><td>10.34</td><td>6.32</td></tr><tr><td>Small</td><td>Dev</td><td>Num Segs</td><td>373</td><td>381</td><td>1142</td><td>336</td><td>560</td><td>820</td><td>526</td></tr><tr><td>Small</td><td>Dev</td><td>Duration</td><td>0.32</td><td>0.31</td><td>3.57</td><td>0.74</td><td>0.57</td><td>0.83</td><td>0.52</td></tr><tr><td>Small</td><td>Test</td><td>Num Segs</td><td>757</td><td>1713</td><td>746</td><td>451</td><td>1130</td><td>1531</td><td>675</td></tr><tr><td>Small</td><td>Test</td><td>Duration</td><td>0.75</td><td>1.58</td><td>0.83</td><td>1.88</td><td>1.40</td><td>1.51</td><td>0.74</td></tr></table>

Here is an outline of the processing steps. Details are provided as part of Goal 2 - to provide a process for non-speech experts to build transcription systems. All the processing was performed using the Jtubespeech github repository tools. The repository documentation is good and the tools easy to use. For those languages that are CCU based the first two steps were unnecessary. The associated Jtubespech tool for each step is provided in boldface.

1) Extract a list of words created from the titles stored in a Wikimedia index dump file, a file with the titles of recent Wikipedia articles for different languages (index dump format described in [18]) (make search word).

2) Use this list of words to search for videos that contain subtitles and are creative commons licensed and are restricted to a specific time period (obtain search word)

3) Extract information about whether or not the subtitles for a video are automatic or manually produced. (retrieve subtitle exists)

4) Download the videos with manual subtitles along with the subtitles. Extract the audio and change the sampling rate to be 16 KHz mono-channel audio. (download video)

The Jtubespeech tools were modified so that they could process data in multiple parallel batches from one master list of video ids. The tool in step 1 was modified to limit titles to those that only contained words for the language in question (the titles often contained English). Note that this required language dependent processing, typically restricting the numeric values of characters in a UTF-8 representation to lie in specific ranges appropriate for a specific language. The original search parameters (found in the util.py tool in the make query url function) were modified to restrict the time and the license type, see [15] for the values of the search parameters and the options.

The above process produced captions for each video as a vtt file [19], a format suitable for closed-captioning the videos. A vtt file contains the text, the beginning, and the end time of the caption to be displayed. They tend to be just a few seconds long and abut each other closely in time. The time marks in short segments were frequently inaccurate for the purpose of marking audio word and/or phrase/sentence boundaries. To ameliorate this problem, the following procedure was used.

1) Merge consecutive segments from the vtt file till either 20 seconds of audio is accumulated or a measurable silence gap (.1 sec was chosen) between segments is found. In vtt files, adjacent segments abut in time unless there was some significant silence between the segments. This process produced either long segments or silencedelimited segments, both of which would allow for better alignments to be computed.

2) For each new merged segment, use one of the supplied language dependent phonetic alignment models provided by WhisperX to determine more accurate begin and end times for the words in each merged segment.

3) Resegment each merged segment using the modified word times from step 2 at silences >= .5 sec producing a new set of segments.

While far from perfect, this process appeared to produce better time alignments than those provided in the original vtt files.

## D. Video Selection Process

The new segments produced via the above process were then transcribed using WhisperX and scored using the NIST scoring toolkit SCTK [20] for computing word and character error rates. The new segment transcripts were used as references. This produces an overall Word Error Rate for each video transcript. Videos with a WER > 50% were discarded; too high a WER threshold would result in data whose transcripts were likely to be of poor quality. Videos shorter than 5 minutes and longer than two hours were also discarded. The former hopefully would improve the chance of selecting videos with the potential of containing interesting cultural markers. The latter was a practical constraint, as longer videos could not be easily processed without significant code changes. The set of associated audio files were divided into separate training, development, and test files, with the bulk of the data for each language assigned for training. The fractional split of the videos across each language varied because the total number of available video files varied, with some languages having relatively few files. The average duration per file across languages varied as well. A ”small” version of the data was also created with about 10% of the data to streamline finetuning experiments given limited computational resources.

The breakdown across languages are shown in Table II. There is significant variability across the languages in the amount of data that was obtained. When possible the original CCU data was used when there were enough files tagged as having Creative Commons licenses (10 hours minimum) (Spanish, Russian, Japanese), otherwise (for Mandarin, Korean, Turkish, Hebrew) Youtube was crawled for more data as described above.

A significant assumption is that the new crawled videos are as suitable for cultural understanding purposes as the original curated CCU data. By limiting the video durations to be greater than 5 minutes the hope was to obtain videos containing interactions and some amount of actual cultural markers. Note that CCU involved processing literally thousands of videos per language to ensure enough cultural markers were obtained to train good models and allow for in-depth evaluations, as cultural markers tend to be low density with rspect to the amount of audio transcribed.

## III. EXPERIMENTS

Three sets of experiments were run on all seven languages. The first set of experiments tested recognition performance with Whisper and WhisperX ”out of the box”. The second set of experiments fine-tuned Whisper and WhisperX using the ”small” set of training data (Table II-D). The third set of experiments was similar to the second except that fine-tuning was performed on the ”all” (complete) set of training data (Table II-D). For the purpose of rapid experiment turnover the ”small” set of test data was used for recognition, and for fine tuning, the ”small” set of development data was used across languages. Word error rates for all languages were computed except for Mandarin and Japanese, where character error rates were used. All fine tuning and inference was performed using Pytorch 2.7.1 and the torch-based Seq2Seq tools using whatever gpus were available on the available high performance compute cluster, (usually A100s, sometimes RTX6000s).

## A. Out-of-Box performance

Figure 1 presents out of box performance across languages for Whisper and WhisperX. WERs tend to lie between 25%- 30%. There is a slight edge in performance for WhisperX; it is also considerably faster than Whisper itself. Mandarin error rate is substantially lower than the other languages, probably because of the use of character error rate (CER) as a metric. However, Japanese error rates, also computed using CER, seem high, with the bulk of the errors arising from a high deletion error rate. No obvious difference is seen in overall WER for CCU vs non-CCU data, though the balance of errors across substitutions, deletions, and insertions across the two data sources are different. Similarly, no obvious difference is seen for Hebrew, with less Whisper training data, and the other languages.

SI Word Error Rate Breakdown by Language  
![](images/2833e53425117a77a59fdd8f25e735bfd175c5d12c6dd2ab6389984cbaf11f3d.jpg)  
Fig. 1. Out-of-the box performance for Whisper vs WhisperX

## B. Performance after Fine-Tuning

The next set of experiments focused on fine tuning. Initial experiments suggested that simply unfreezing all the parameters in the model and running two epochs of training on top of the base models with a learning rate of 1e-5 and weight decay of .005 was as good, if not better, than anything else tried. However, no attempt was made to perform an in-depth search to find the optimal (single) processing recipe across languages.

Figure 2 presents out of box performance averaged across language. The first bar is Whisper performance out of the box (29.7% WER). The second bar is the result of fine-tuning using the ”small” amount of data setting found in Table II for each language. Performance after fine-tuning is worse than outof-box performance! Perhaps this occurred because of poor parameter tuning, or not using more sophisticated adaptation schemes such as LORA [21]. However, it can be seen that the main reason for the WER increase was a significant increase in the number of insertions. The insertion increase was almost always due to an increase in hallucinations. One simple technique used to reduce hallucinations is to reduce the maximum number of tokens that can be generated by explicitly specifying generation config.max new tokens, one of the arguments to the Seq2Seq trainer code. A setting of 64 resulted in output transcript truncation. A setting of 256 seemed to generate the best results given the maximum length of a test segment was limited to 20 seconds. The third bar presents the result of this setting for Whisper based decoding (26.0% WER). As can be seen, some improvement with fine tuning now results relative to out-of-box performance (29.7%).

For WhisperX, the corresponding WER number to the Whisper results (26.0%) is the fourth bar (24.4%). No attempt was made to tune any of the WhisperX parameters. Note WhisperX was less prone to hallucinations. This is perhaps due to a much better VAD used in WhisperX relative to Whisper, resulting in elimination of more low-level noise, and shorter audio segments to recognize, both of which tend to reduce the amount of hallucination.

The third set of experiments fine-tune on all the training data. Only results for WhisperX are presented (Whisper results were slightly worse). The final result is shown in the fifth bar, achieving another decrease in average WER (21.7%).

Statistical significance was checked for the above results by applying paired bootstrap tests [22] to all 10 pairwise condition comparisons, aggregated across all 7 languages. All comparisons were significant (p < .001) except for Whisper out-of-box performance vs. Whisper optimized fine tuning performance (p = .012). This suggests that the ”small” amount of data used for fine tuning - 10 hours or less per language - was just too small to generate large enough improvements to be considered significant at the desired threshold.

No substantial difference in the performance of Hebrew relative to the other languages before and after fine tuning was observed even though it was represented by much less training data in Whisper than the other six languages. It was also noted that performance for the four languages harvested from YouTube was substantially better than the three languages whose was usable (in terms of having Creative Commons licenses) from the CCU program. This is perhaps attributable to the careful screening process used by LDC in data selection to ensure the data was appropriate for cultural understanding purposes, potentially resulting in more complex data.

![](images/3713006fba6daa534df8fec1c0c8114cf9c1f29b75e78381738f0d3adfcc9b33.jpg)  
Fig. 2. Performance after Fine-Tuning

## IV. DISCUSSION

The above results demonstrate that the average word error rate for transcription of difficult speech using state-of-theart open source speech tools and systems and relatively light (compared to industrial strength) computing resources can be lowered by almost 33% relative from 30% to close to 20%. The maximum amount of data used for any one language was 200 hours of speech (Turkish). Larger gains could probably be observed by fine tuning with significantly more training data, computing and people resources.

TABLE III  
WORD ERROR RATE DEGRADATION TOLERANCES ESTIMATED FOR THREE NLP COMPONENTS (FROM [23])
<table><tr><td>Metric</td><td>WER Tolerance (NTP)</td></tr><tr><td>Summarization</td><td>7%-30%</td></tr><tr><td>Q&amp;A</td><td>5%-34%</td></tr><tr><td>Dialog Act Classification</td><td>44%-71%</td></tr></table>

Is this level of performance ( 20% WER) adequate for accurate extraction of cultural markers? It is difficult to answer this question directly. Unlike speech recognition tools such as Whisper and WhisperX, there are no controlled data bases and open source code for extraction of spoken cultural markers that would permit straightforward evaluations as a function of speech recognition error rates.

One (admittedly, crude) proxy for a full evaluation is the evaluation of typical NLP components for spoken language processing as a function of Word Error Rate. Such studies are rare (and multilingual studies basically nonexistent). In one such recent study, synthetic speech corrupted by varying levels of noise was used to evaluate the impact of WER degradation on 3 different NLP components [23].

The components examined were: Summarization, Question Answering (Q& A) and Dialog Act Classification (DAC). Several measures of sensitivity to transcription quality were examined, but most germane to this discussion is the metric they called the noise toleration point (NTP). The NTP is the increase in WER that generates a ”statistically significant” degradation in task metric relative to baseline no-noise performance. Here ”statistical significance” is defined as a performance degradation by one standard deviation as estimated from metric score statistics. Each component was implemented with four different models corresponding to four different LLMs producing four different NTP values per component.

Table III contains the range of NTP values for each component. It can be seen that DAC is the least sensitive to increases in WER, with a range of 44%-71% across models. Insofar as DAC is analogous to topic spotting, this does not come as a complete surprise; it has been known since the early 1990s [24] that good topic spotting could be done at high word error rate levels. Summarization and Q&A seem more sensitive to model quality; some models start degrading even for very low word error rates while others seem more robust, at least up WER ranges of 20%-30%.

It would therefore appear that out of the box Whisper or WhisperX performance for the languages investigated here (with word error rates ∼30%) might somewhat degrade NLP component performance, but that fine tuning with as little as

50-100 hours of data might lower WERs to 20%- the point of producing results close to what would be achieved with perfect transcriptions. Of course, this is highly speculative given that underlying NLP component performance may vary across languages and that the effects of degradations on multiple NLP components may have a multiplicative effect on performance of cultural marker performance extraction systems built upon such components.

It should be noted that true word error rates are not being measured, just deviations from closed captions of indeterminate quality that were provided without documentation. However, the WERs do decrease substantially after both finetuning and inference parameter tuning. This suggests that the transcripts are not completely inaccurate. Nevertheless, it certainly would have been desirable to have gold-standard transcripts generated for the test data for these datasets.

## V. DATA RELEASE

One of the goals of this work was to create a set of processes, data and tools that would make it easier for nonspeech-experts to generate speech transcripts for the purpose of identifying cultural markers. The original plan was to release all the data described in this paper, as the data selection process using the YouTube API (Section II-C) specified to filter on the basis of the presence of a Creative Commons license. Unfortunately, in the data release process, it was found that although the filter in the YouTube API was set to select only Creative Commons data, there was no actual license information in the video metadata for about half the data, and some fraction of the videos were no longer available (Table IV.

TABLE IV  
CREATIVE COMMONS LICENSE PRESENT IN VIDEO METADATA
<table><tr><td>Language</td><td>Original</td><td>Present</td><td>Absent</td><td>Video Vanished</td></tr><tr><td>Mandarin</td><td>158</td><td>130</td><td>6</td><td>22</td></tr><tr><td>Spanish</td><td>33</td><td>0</td><td>33</td><td>0</td></tr><tr><td>Russian</td><td>204</td><td>1</td><td>199</td><td>4</td></tr><tr><td>Japanese</td><td>101</td><td>1</td><td>100</td><td>0</td></tr><tr><td>Korean</td><td>283</td><td>211</td><td>1</td><td>71</td></tr><tr><td>Turkish</td><td>827</td><td>441</td><td>14</td><td>372</td></tr><tr><td>Hebrew</td><td>156</td><td>148</td><td>0</td><td>8</td></tr></table>

Therefore, to err on the side of caution for now, only a subset of the data can be released. Three of the languages (Spanish, Russian, and Japanese) had little or no license information explicitly set in the video metadata. The release [25] contains both the speech data, and a set of of tsv files breaking the data into train, test, and dev subsets. The speech data is provided after the segmentation described in Section II-C was performed, along with the transcriptions of all the speech segments.

Table V summarizes the statistics on the released data. Error rates are presented before and after fine-tuning using the Whisper large-v2 model with the optimized configurations described in Section III. Even though the amounts of training data are less than the amounts in the original configurations, significant improvements after fine-tuning are still observed for all languages.

TABLE V  
STATISTICS ON THE RELEASED DATA
<table><tr><td>Metric</td><td>Korean</td><td>Mandarin</td><td>Hebrew</td><td>Turkish</td></tr><tr><td>Train Segments</td><td>41865</td><td>25930</td><td>59128</td><td>102615</td></tr><tr><td>Train Hours</td><td>43.48</td><td>30.42</td><td>61.38</td><td>105.07</td></tr><tr><td>Dev Segments</td><td>1113</td><td>1402</td><td>5255</td><td>7285</td></tr><tr><td>Dev Hours</td><td>1.09</td><td>2.88</td><td>5.16</td><td>7.74</td></tr><tr><td>Test Segments</td><td>3027</td><td>5250</td><td>6558</td><td>13847</td></tr><tr><td>Test Hours</td><td>3.44</td><td>6.42</td><td>7.28</td><td>13.61</td></tr><tr><td>Error % (No-Train)</td><td>32.1</td><td>21.4</td><td>27.8</td><td>29.4</td></tr><tr><td>Error % (Train)</td><td>23.4</td><td>10.2</td><td>22.1</td><td>18.2</td></tr></table>

## VI. SUMMARY

A methodology for researchers interested in cross-cultural understanding to generate new data for input into training systems for cross-cultural understanding and other natural language tasks was described. The methodology proposed does not require deep speech recognition expertise and leverages open source data and tools. It was shown that even with relatively small amounts of fine tuning data (30-200 hours), one could obtain substantial improvements in speech recognition performance using state of the art transcription tools such as Whisper and WhisperX when manual transcriptions are available. In addition, 260 hours of training, dev, and test data were released to the community in Mandarin, Korean, Turkish, and Hebrew that can be used for future research by the community to improve speech recognition performance for cultural understanding research. Future work in this area might include experimenting with more sophisticated fine tuning methodologies, or perhaps ensembling multiple multilingual speech recognizers to achieve further improvements in performance. In addition, investigating the actual impact of WER on cross-cultural information extraction would be fascinating when such automated tools become available to the broader community.

## ACKNOWLEDGMENT

The author would like to thank Profs. He He and Kyunghyun Cho of NYU for their advice and support and the following NYU students who were part of the CCU program. They created and ran the processing pipeline that extracted speech transcriptions for the many evaluations in the actual program: Nikhil Verma, Sukrit Rao, Jash Rathod, and Sriphani Bellamkonda.

AI assistance (ChatGPT 4.3) was used to help create properly formatted Tables II and V and Figures 1 and 2. In addition, it was consulted for for suggestions on how to reduce memory footprint and improve i/o performance for the original training code. Claude Sonnet 4.6 suggested the use of bootstrap methods for the significance tests, produced code and ran the tests described in the Results section.

## REFERENCES

[1] O. Li, M. Subramanian, A. Saakyan, S. C.-W. Wang, and S. Muresan, “Normdial: A comparable bilingual synthetic dialogue dataset for modeling social norm adherence and violation,” in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP). Singapore: Association for Computational Linguistics, 2023, pp. 15 732–15 744. [Online]. Available: https://aclanthology.org/2023.emnlp-main.974/

[2] Y. R. Fung, T. Chakrabarty, H. Guo, O. Rambow, S. Muresan, and H. Ji, “Normsage: Multi-lingual multi-cultural norm discovery from conversations on-the-fly,” in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing. Singapore: Association for Computational Linguistics, 2023, pp. 15 217–15 230. [Online]. Available: https://aclanthology.org/2023.emnlp-main.941

[3] S. C. Wang, O. Li, S. Muresan et al., “Normgenesis: A benchmark for generating and repairing social norm violations in dialogue,” in Proceedings of the Annual Meeting of the Association for Computational Linguistics, 2024.

[4] Y. Yuan, K. Tang, J. Shen, M. Zhang, and C. Wang, “Measuring social norms of large language models,” in Findings of the 2024 Conference of the Association for Computational Linguistics (NAACL). Association for Computational Linguistics, 2024, pp. 650–699, nAACL 2024 – Findings. [Online]. Available: https://doi.org/10.18653/v1/2024.findingsnaacl.43

[5] P. Sahu, A. Som, A. Divakaran, and D. Vergyri, “Minds: A crosscultural dialogue corpus for social norm classification and adherence detection,” in Findings of the 14th International Joint Conference on Natural Language Processing (IJCNLP) & 4th Asia-Pacific ACL. Mumbai, India: The Asian Federation of Natural Language Processing and ACL, 2025, pp. 2039–2052. [Online]. Available: https://aclanthology.org/2025.findings-ijcnlp.128/

[6] S. Pawar, J. Park, J. Jin, A. Arora, J. Myung, S. Yadav, F. G. Haznitrama, I. Song, A. Oh, and I. Augenstein, “Survey of cultural awareness in language models: Text and beyond,” Computational Linguistics, vol. 51, no. 3, pp. 907–1004, 2025.

[7] V. Srivastav, S. Zheng, E. Bezzam, E. Le Bihan, N. Koluguri, P. Zelasko,<sup>˙</sup> S. Majumdar, A. Moumen, and S. Gandhi, “Open asr leaderboard: Towards reproducible and transparent multilingual and long-form speech recognition evaluation,” arXiv e-prints, pp. arXiv–2510, 2025.

[8] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, “Robust speech recognition via large-scale weak supervision,” in International conference on machine learning. PMLR, 2023, pp. 28 492–28 518.

[9] M. Bain, J. Huh, T. Han, and A. Zisserman, “WhisperX: Time-Accurate Speech Transcription of Long-Form Audio,” in Interspeech 2023, 2023, pp. 4489–4493.

[10] A. Plaquet and H. Bredin, “Powerset multi-class cross entropy loss for neural speaker diarization,” in Proc. INTERSPEECH 2023, 2023.

[11] H. Bredin, “pyannote.audio 2.1 speaker diarization pipeline: principle, benchmark, and recipe,” in Proc. INTERSPEECH 2023, 2023.

[12] GitHub Community. (2024) Differences in large-v1, v2 v3 models? #338. Accessed: March 17, 2026. [Online]. Available: https://github.com/jhj0517/Whisper-WebUI/issues/338

[13] DARPA, “Ccu: Computational cultural understanding,” 2021, accessed: March 17, 2026. [Online]. Available: https://www.darpa.mil/research/programs/computational-culturalunderstanding

[14] “Youtube data api,” 2026, accessed: May 13, 2026. [Online]. Available: https://developers.google.com/youtube/v3/docs/search/list

[15] “Paginating, sorting, and filtering with the youtube api,” 2026, accessed: May 13, 2026. [Online]. Available: https://serpapi.com/blog/youtubesp-filters-paginating-sorting-and-filtering-with-the-youtube-api/

[16] “Yt-dlp a feature-rich command-line audio/video downloader,” 2020, accessed 2026-03-17. [Online]. Available: https://github.com/yt-dlp/ytdlp

[17] S. Takamichi, L. Kurzinger, T. Saeki, S. Shiota, and S. Watanabe,¨ “Jtubespeech: corpus of japanese speech collected from youtube for speech recognition and speaker verification,” 2021. [Online]. Available: https://arxiv.org/abs/2112.09323

[18] “How to read a wikipedia dump,” 2020, accessed: March 18, 2026. [Online]. Available: https://data-and-the-world.onrender.com/posts/readwikipedia-dump/

[19] “Web video text tracks format (webvtt),” 2025, accessed: March 18, 2026. [Online]. Available: https://developer.mozilla.org/en-US/docs/Web/API/WebVTT API/Web Video Text Tracks Format

[20] “Sctk, the nist scoring toolkit,” 2021, accessed 2026-03-17. [Online]. Available: https://github.com/usnistgov/SCTK

[21] Z. Song, J. Zhuo, Y. Yang, Z. Ma, S. Zhang, and X. Chen, “LoRA-Whisper: Parameter-Efficient and Extensible Multilingual ASR,” in Interspeech 2024, 2024, pp. 3934–3938.

[22] M. Bisani and H. Ney, “Bootstrap estimates for confidence intervals in ASR performance evaluation,” in Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), vol. 1, 2004, pp. 409–412.

[23] O. Shapira, S. Chazan, and A. D. N. Cohen, “Measuring the effect of transcription noise on downstream language understanding tasks,” in Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), W. Che, J. Nabende, E. Shutova, and M. T. Pilehvar, Eds. Vienna, Austria: Association for Computational Linguistics, Jul. 2025, pp. 29 978–30 004. [Online]. Available: https://aclanthology.org/2025.acl-long.1449/

[24] B. Peskin, L. Gillick, Y. Ito, S. Lowe, R. Roth, F. Scattone, J. Baker, J. Baker, J. Bridle, M. Hunt, and J. Orloff, “Topic and speaker identification via large vocabulary continuous speech recognition,” in Proceedings of the Workshop on Human Language Technology, ser. HLT ’93. USA: Association for Computational Linguistics, 1993, p. 119–124. [Online]. Available: https://doi.org/10.3115/1075671.1075697

[25] M. Picheny, “ccu-hf-data (revision 38103b7),” 2026. [Online]. Available: https://huggingface.co/datasets/picheny/ccu-hf-data