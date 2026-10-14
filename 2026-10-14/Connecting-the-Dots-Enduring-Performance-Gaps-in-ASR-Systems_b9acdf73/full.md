# Connecting the Dots: Enduring Performance Gaps in ASR Systems Between Standard and Minority English Varieties

Alexander Metzger<sup>1∗</sup>, Aruna Srivastava<sup>1∗</sup> Ruslan Mukhamedvaleev<sup>1</sup> Eunjung Yeo<sup>2</sup> Syed Ishtiaque Ahmed<sup>3</sup> Nina Markl<sup>4</sup> Sachin Kumar<sup>5</sup> Farhan Samir<sup>3</sup>

<sup>1</sup>Koel Labs LLC <sup>2</sup>University of Texas at Austin <sup>3</sup>University of Toronto <sup>4</sup>University of Essex <sup>5</sup>The Ohio State University

Abstract—Automatic Speech Recognition (ASR) system evaluations have consistently revealed performance disparities along lines of race, gender, socioeconomic status, and language variety. While these disparities are well-documented, they are often (implicitly or explicitly) treated as issues that subsequent, larger, and more universal models will correct. We challenge this assumption of steady progress in reducing accent-related performance disparities, conducting a longitudinal evaluation of 16 prominent systems released between 2021-2025 across 8 datasets representing diverse English accent varieties. We find that the performance disparity between Mainstream US English (MUSE) and minority accents is stubborn, not reliably declining as a function of time. Specifically, while systems are reliably performant on MUSE, minority accents are subject to risks of large performance degradations between generations of systems, sometimes worsening by 65% absolute Word Error Rate. We argue that standard ASR benchmarks enact an implicit language policy privileging Mainstream US English varieties. As such, these performance disparities reflect systemic failures of language technology development infrastructure that disproportionately affect underrepresented speaker populations.<sup>1</sup>

Index Terms—ASR, fairness, evaluation, sociolinguistics

## I. INTRODUCTION

Automatic Speech Recognition (ASR) systems are among one of the oldest, formative pursuits in computational linguistics research [1]. Concerns about the limited robustness of these systems, owing to the relative homogeneity of their training datasets, have been prominent for decades [2]. A wide array of studies have reported on performance disparities among important demographic constructs. At different points in time, English ASR systems have been shown to be worse for children and the elderly [2], [3]; for Black speakers compared to white speakers in the US [4], [5]; for speakers from lower socioeconomic backgrounds [6]; and for speakers of stigmatized varieties of English [7], [8], among others. These studies are situated in particular temporal and technological contexts. As such, they may be interpreted as merely temporary shortcomings, or glitches [9] that will be attenuated by future research and development.

Indeed by now there are a number of targeted interventions for resolving these shortcomings [10], but there is no guarantee that these practices will be adopted in training the next generations of pretrained speech models. Instead, the more standard approach in training new generations of models is to train on larger datasets with larger models [11], speculating that their larger scale will also entail improved representation of minority accents. It is then natural to wonder whether the performance gap between hegemonic Mainstream US English and minority accent varieties may simply naturally dissipate over time. The optimism undergirding scaling heuristics is not entirely unfounded, optimizing large models on ever-larger big data archives has proven a useful heuristic when it comes to optimizing predictive performance on standard benchmarks [12].

At the same time, it is unclear whether scaling heuristics would help in making predictions for inputs that are scarcely represented in standard evaluation harnesses, as we argue to be the case for minority accents in speech processing evaluations. That is, increased representation on some apparent axes (number of hours, number of languages, number of tasks), does not entail comparable improvements on other less-visible axes (minority accent representation). It then behooves us to investigate whether performance disparities are actually dissipating over time, as models grow larger [13], are trained on more tasks [14] as well as datasets with a longer overall duration and more languages [15]. Taking a longitudinal view towards analyzing performance disparities is important, as it can inform researchers, developers, and policymakers as to whether the scaling modus operandi in machine-learning can be expected to naturally erode performance disparities until they are negligible, or whether we require more targeted interventions for obtaining minority accent representation.

In this work, we perform this much-needed longitudinal analysis, evaluating 16 prominent models on datasets that are designed to capture a wide range of varieties of English. Crucially, we select datasets that represent both standard accents, specifically Mainstream US English and Standard Southern British English, or MUSE and SSBE respectively, as well as minority accents, enabling us to assess the performance gap. We show that the gap has endured over the course of these years, with even some of the latest generations of models exhibiting a gap of 10% absolute WER, even for relatively simple scripted sentences, and even worse (20%) for unscripted ones. We also observe that high quality performance on MUSE is durably prioritized as a feature in the development of new systems, with systems consistently achieving high-quality transcripts that are often close to 0% WER rates. By comparison, minority accent representation can deteriorate between generations of systems, sometimes by as much as 65% of WER; as such, the performance gap not only endures but can increase with the release of newer generations of systems. Our findings call for greater transparency of speech processing systems that are deployed in downstream applications where speakers of marginalized English varieties would come into sensitive contact with them.

## II. ASR EVALUATION INFRASTRUCTURE

To understand why accent-based disparities reflect systematic trends, rather than merely independent and episodic glitches, we will first consider current benchmarking practices in ASR development. ML benchmark datasets have been theorized as a form of infrastructure [16], [17], in many senses of the concept [18]. Most importantly for our purposes, they undergird the development of new speech-processing models; and the human judgments that went into the data curation process [19] become more invisible as they become naturalized as standard benchmark datasets [17]. For ASR benchmarks, these are socio-linguistic judgments about the type of speech that is considered “typical” and thus important to model, and the types that can be made invisible and subsequently ignored. In effect, this process of formalization into a benchmark dataset can be interpreted as a type of language policy [20], specifying the spoken English varieties and contexts that are most appropriate. Many Global Englishes are foregone in the curation of standard ASR benchmarks.

As such, we argue that prominent evaluation benchmarks effectuate an Anglocentric language policy. While there are no firm prescriptions (e.g., by publishers, regulators or professional organisations) regarding benchmark selection, there can nevertheless be discernible norms within the field. As observed in [20], “the absence of an official policy...often serves only to reinforce the power and hegemony of prestige varieties, and marginalize others.” We can thus analyze regularities in benchmark selection, and interpret these as a de facto language policy [20].

## A. Benchmark Selection

Concretely, we study these regularities in benchmark usage by first selecting a range of 16 prominent ASR models over the last 4 years, shown in Table I. We selected models from 2021 onwards, as this period is when large-scale pre-training emerged as the dominant paradigm in machine learning [28].<sup>2</sup> We selected organization that are notable for prominent speech-processing technologies: Mozilla, Microsoft, OpenAI, Meta, CMU, Alibaba, NVIDIA, and Google. We list the benchmarks used to evaluate the models’ English ASR capabilities, drawn from their respective technical reports, which are also cited in full detail in Table III in the Appendix. In the case of composite benchmarks like SUPERB [29], OpenASR [30], and, SpeechStew [31], we list all of the constituent datasets individually rather than the aggregated benchmark. In Table IV in the Appendix, we list the subset of benchmarks that are explicitly concerned with multi-accent representation. The concern is not whether pluralistic benchmarks exist at all, but rather that they are peripheral – they are sometimes used in evaluations of speech AI systems, and often not.

TABLE I: Models evaluated in our diachronic analysis. Parentheses indicate additional versions from the same model family. The last column is the proportion of evaluation datasets used by the model authors for benchmarking that are explicitly concerned with multi-accent representation. The names of the datasets are included in Table III in the Appendix.
<table><tr><td>Model</td><td>Release</td><td>Org.</td><td>Multi-Accent Evaluation Datasets (ASR)</td></tr><tr><td>Deepspeech [21] (m1)</td><td>2021-10</td><td>Mozilla</td><td>2/6</td></tr><tr><td>WavLM [22] (m2)</td><td>2021-12</td><td>Microsoft</td><td>0/1</td></tr><tr><td>whisper [15] (m3, (m4, m8))</td><td>2022-09</td><td>OpenAI</td><td>4/14</td></tr><tr><td>MMS [23] (m5)</td><td>2023-05</td><td>Meta</td><td>0/5</td></tr><tr><td>OWSM [24] (m6(m7, m10, m12))</td><td>2023-09</td><td>CMU</td><td>1/8</td></tr><tr><td>SeamlessM4T [25] (m9)</td><td>2023-12</td><td>Meta</td><td>0/1</td></tr><tr><td>Qwen-Audio [14] (m₁1(m13))</td><td>2024-08</td><td>Alibaba</td><td>1/3</td></tr><tr><td>Canary-Qwen [13] (m₁4)</td><td>2025-07</td><td>NVIDIA</td><td>3/9</td></tr><tr><td>Chirp-3 [26] (m15)</td><td>2025-10</td><td>Google</td><td>1/9</td></tr><tr><td>Omnilingual [27] (m16)</td><td>2025-11</td><td>Meta</td><td>0/5</td></tr></table>

Consider that out of the 61 model evaluations conducted across the technical reports enumerated in Table I, only 12 of them include some consideration of accent diversity. Of the 24 distinct evaluation benchmark families (counting Englishlanguage subsets within their parent families), 7 have pluricentric considerations of English varieties. Three of the systems (WavLM, MMS, Omnilingual) include no explicit considerations of accent pluralism. It almost goes without saying that Mainstream US English is never excluded from the evaluation infrastructure. Moreover, ASR performance on this variety is often assessed in multiple contexts, from spontaneous vs. read speech (e.g., Switchboard vs. Librispeech); to individual speakers vs. multiple (TedLIUM vs. CasualConversations). By comparison, if accent pluralism is considered in the evaluation, it tends to be in a more limited range of contexts and registers. For example, the OWSM and Qwen 2.5 models evaluates performance on CommonVoice, only assessing for read-speech of neutral-register Wikipedia sentences. Whisper [15] and Canary [13] diverge from this, evaluating on a variety of registers and contexts.

## B. Discussion

Other benchmarks also sought to increase representational diversity, but along other axes than L2-English speaker representation. We discuss two of these major axes, number of hours and multilinguality, while noting that progress along them is separate from improvements in diverse accent representation.

a) Other representational axes — multilinguality: One prominent axis has been multilinguality; for example, Common

Voice, Fleurs. This is a reasonable, well-motivated axis to pursue, since a prominent critique of NLP as a discipline has been its focus on English datasets emerging from central Anglophone countries [32], a critique that applies for audio datasets as well [33]. In collecting datasets that contain a larger number of distinct languages, however, these works typically adopt the normative stance that languages are geographically and culturally homogenous artifacts [34]. Often this stance is implicit, but can be ascertained from the categorization of the speech data into different language bins, with no discussion of the particular varieties as well as limited sociodemographic information [33], e.g., in Multilingual Librispeech [35]. Sometimes, this stance is explicit, for example the Fleurs dataset explicitly recruits “native” speakers in constructing its multilingual speech evaluation datasets: “We collected three recordings by three different native speakers” [36]. Inspection of the English speech in this dataset indicates a Mainstream US English variety. Thus, the construction of “linguistic diversity” is complex and value-laden, and the representation of World Englishes may even be better in a monolingual dataset than one that contends to serve as evaluation infrastructure for “universal” [36] speech models.

There are multilingual datasets that have more pluralistic representations of English, for example, CommonVoice [37] and VoxPopuli [38]. However, the most common accent representations in CommonVoice is by far Mainstream US English, based on audits of v7 and v13 [39], [40]. VoxPopuli contains 15 European accent varieties, but is a minority of the total transcribed corpus (29/1,800 hours). How these corpora are used matters. For example, MMS evaluates on CommonVoice over the whole corpus [23]. Since minority accent varieties are a small proportion of the English subset of the corpus, and necessarily an even smaller subset of the entire corpus, the average word error rate would be rendered ineffective in tracking robustness to accent variation, particularly variation outside of Mainstream US English. As recognized in [41], operationalizing performance as “correctness across individual predictions” is a value-laden modus operandi, and reflects Anglocentric language policy due to the skewed constructions of these datasets. By comparison, the Whisper model presents their evaluation targeted on the English partitions of both VoxPopuli and Whisper, thus more effectively tracking accent robustness [15]. For this reason, we distinguish between CommonVoice and CommonVoice-EN, depending on whether the subset was evaluated specifically or whether it was the entire set. We draw the same contrast for VoxPopuli and Voxpopuli-EN.

b) Other representational axes — total duration.: Other evaluation benchmarks have sought to increase diversity through increasing the total duration of the entire benchmark dataset, implementing the aphorism “there is no data like more data”, attributed to Robert Mercer during his tenure at IBM’s Continuous Speech Recognition team [1], as a data collection policy. The CALLHOME [42], SwitchBoard [43], and WSJ [44] corpora were funded by the US Department of Defense to advance speech recognition research from “specific database inquiry tasks, characterized by medium vocabularies” [44] towards open-ended, multi-speaker dictation. These were characterized by increases in number of speakers, number of words in transcripts, and recording durations. For example, CALLHOME emphasized “18.3 hours of transcribed spontaneous speech, comprising about 230,000 words” [45]. In a similar vein, we have “WSJ corpus will provide DARPA its first general-purpose English, large vocabulary, natural language, high perplexity, corpus containing significant quantities of both speech data (400 hrs.) and text data (47M words).” The focus on quantification and scale as a stand-in for diversity continues to be pervasive in ML data collection [19], [46], like TedLium [47] and Librispeech [48], both describing their datasets first and foremost in terms of scale (“a total of 774 talks, representing 118 hours of speech”; “contains 1000 hours of speech”), rather than other potentially salient characteristics, like speakers’ demographic attributes. As previously observed by [19], a foundational assumption in targeting large volumes of data is that such “unconstrained” data collection methods will naturally represent diversity on sociodemographic axes [19]. In these works, the disparities between the availabilities of different sociodemographic speaker groups is rarely acknowledged let alone interrogated. This inevitably leads to the proliferation of speech benchmarks that have skewed representations of Mainstream English varieties [39].

c) Implications of sporadic multi-accent representation.: Benchmarks serve a dual purpose in ML. Most commonly, they serve as evidence of improved performance, understood as the most common goal in ML [41]. But there are other commonplace values, including parameter and data efficiency, transparency, and theoretical analyzability [41]. In pursuing primary goals that are not standard benchmark dominance, benchmarks serve a role that is analogous to regression tests in software development – preventing against performance degradations.

Indeed, we observe benchmarks assuming this regression prevention role in some of these systems, that have primary goals other than achieving state of the art ASR performance. For example, Qwen-2 expands the set of speech-processing tasks that the model can complete [14]. The OWSM system was meant to serve as an open-data, open-weights (near-)replication of OpenAI’s Whisper [24], the latter being trained on a dataset of 680, 000 hours of speech recordings [15]. The MMS system was aimed at expanding the set of languages that ASR models can transcribe [23]. The models are often not the state-of-the art on some benchmarks, yet this is not understood as a failure, but rather evidence that the model did not regress too much along other performance axes of their primary one; for example, both the MMS and OWSM reports are transparent that Whisper outperforms them on some standard benchmarks. However, we see that performance degradations for standard varieties of English are thoroughly tested, while minority varieties are not. We thus test the following hypothesis (H1): The performance gap between standard and minority varieties will not be systematically diminished over time, as minority English varieties are more likely to be subjected to significant performance degradations in subsequent generations of systems.

## III. DATASETS

To select datasets for the evaluation, we compiled a list of multi-accent datasets, specifically those with speakers for whom English is a second language (L2-English speakers). We aim for datasets that are not considered standard benchmarks in the ASR community; none of the ones in our evaluation were used to by the development teams for the systems listed in Table I. We built our evaluation corpora list through a combination of scholarly search engines and consulting with sociophoneticians. This aggregation phase left us with 39 potential datasets. However, many of these were unsuitable for the purposes of our analysis. Several datasets lacked accent annotations for their recordings which is unsuitable for a comparative analysis between performance on standard and minority accents. In the same vein, other datasets did not have recordings from L1-speakers, precluding a comparative analysis. Other datasets required requests for access that are still pending. For the remaining datasets, we downloaded and listened to random samples, excluding ones that had an overrepresentation of noisy, unintelligible recordings. Ultimately, we used 4 multi-accent corpora for our evaluation.

Specifically, we use the Speech Accent Archive [49, 12 English varieties]; the L2-Arctic dataset [50, 6 varieties]; the OpenSLR83 dataset [51, 6 varieties]; and the ALLSTAR dataset [52, 22 varieties]. L2-Arctic contains both scripted and unscripted portions, and we treat these as separate datasets for our analysis. Similarly, ALLSTAR is partitioned into 4 subsets, comprising recordings from speakers reading sentences from the novel The Little Prince (ALLSTAR-LPP); the Declaration of Human Rights (ALLSTAR-DHR); and two lists from the Hearing in Noise Test [53] (ALLSTAR-HT1; ALLSTAR-HT2). We treat these pre-existing partitions as separate datasets for our analysis. We describe these datasets further in our supplementary appendix.

## IV. EXPERIMENTS AND RESULTS

Here, we consider the series of models from Table I ordered by their release date, with $m _ { 1 } = { \tt D e e p s p e e c h , . . . , m _ { n } } =$ Omnilingual. We then compute model Word Error Rates (WERs) on every accent variety recorded in every dataset we use for our evaluation. We then study the evolution of performance on minority accent varieties, with Mainstream US English (MUSE) as a control accent in each dataset.

Our longitudinal study entails comparing models that were trained by different research teams, who are using different training and evaluation datasets and procedures. That is they were developed in heterogeneous situational contexts. Our aim is not to argue for particular training algorithms or dataset compositions over others, but rather a sociotechnical analysis characterize broader dynamics in the representation of minority accents. As such, comparing models developed in varying contexts is reasonable, especially considering that ML system development has long been fueled by a singular focus on “single defined performance number” [54]; in the case of speech recognition, the WER. The type of heterogeneous longitudinal comparison we are conducting has also been performed in prior work [55].

## A. Longitudinal trends

We assess the extent to which the performance gap between standard accents (MUSE and SSBE) and minority accents has declined over the time period from $m _ { 1 }$ to $m _ { n }$ . For a given dataset, we represent this performance gap as the WER difference between these standard accents and the accent variety with the highest average WER. We note that focusing on worst-case performance is a common practice in the ML fairness literature [56]; see also [41] for a critical discussion of the pitfalls of average-case metrics.

We highlight both of these accents for every dataset in Figure 1. The top panel of each figure shows the WERs across each accent in the dataset, while the bottom panel shows the evolution of the performance gap between MUSE and the selected minority accent across $m _ { 1 } , \ldots , m _ { n }$ . We followed standard practice in ASR evaluation and applied a standard text normalization module prior to computing WER [30].

a) Large average performance gaps across datasets.: As shown in Figure 1, we find that sizable performance gaps between standard accents and minority ones are present in every dataset that we evaluated across the time period represented by $m _ { 1 }$ to $m _ { n } .$ , with an average performance gap ranging from 9% absolute WER, for SSBE compared to Irish English in the OpenSLR83 dataset, to 26% for MUSE compared to Vietnamese English in the Little Prince section of the ALLSTAR corpus.

b) Are the performance gaps declining over time?: To assess whether the performance gap is declining systematically over time, we computed the spearman correlation between (ρ) between the model-release dates and the standard-minority WER differences that constitute the performance gaps. We only find statistically significant evidence of the performance gap being ameliorated during this time period for two datasets, specifically for OpenSLR-83 and L2-Arctic with correlation values of $\rho = - 0 . 7 5$ and $\rho = - 0 . 6 0$ (both $p < 0 . 0 5 )$ . For L2-Arctic, however, even the model $( m _ { 1 3 } )$ that performs the best at the selected minority accent, Vietnamese English, still exhibits a sizeable performance gap of 10% WER. Meanwhile, for OpenSLR-83, while the performance gap has generally diminished over the full time period, the most recent releases we evaluated from $m _ { 1 3 }$ to $m _ { 1 6 }$ have bucked this trend, instead exacerbating the gap at each of these releases. In fact, the most recent evaluated model $( m _ { 1 6 } )$ exhibits a performance gap that matches the historical average performance gap of 9.2% WER. For other datasets, evidence of the performance gap declining is not statistically significant. That is to say, there is overall scant evidence of the performance gap diminishing over time; instead of consistently declining, the trends in the performance gap size are highly non-monotonic.

L2-Arctic - Vietnamese (highest WER) vs USA (standard): WER over time  
![](images/ec19df0cb0bc306dddf3ae75803270b2ab94456b24a2b5818f91f120ddf1a5b8.jpg)  
L2-Arctic Spontaneous - Korean (highest WER) vs USA (standard): WER over time

![](images/b9816cef4f49ba18f269939c98e9d41b0c26eb1c2693f42a2b7f208310259ded.jpg)  
Openslr83 - Irish English (highest WER) vs Southern English (standard): WER over time

![](images/379bd75083b80ae150ab639df38aafb5be215696a978850677744ed64eb6946f.jpg)

![](images/0656917f4d02d8ccc46bc5a7045aa224b7c5f2484517f3e030ddf94d572e57d4.jpg)  
SAA - Korean (highest WER) vs English (standard): WER over time

![](images/bbc8dbfc2dac41e7c5b3fdf05ee55ee23817ad4e3544e7cfcb30f71d2a3f8d68.jpg)

![](images/4d4cd7516dd42c35658dcd1e613b57acb6fcf95652e5d4b9b661589e30aa4c80.jpg)

![](images/fa3a98d9d092d79aa9615fdd97327dc9e826d322b0887094724cc3fc936d92e7.jpg)

ALLSTAR HT1 | RUN Vs ENG  
![](images/44006a6c7f7b2bc8f3647a82f1ce43d36283feee635e30141ef004a39c196239.jpg)  
ALLSTAR HT2 | RUN Vs ENG

![](images/e64d936547dfa0a2b76153c62ca8f2eca4c80e4c82c54407e8a78f5d6fcf0d82.jpg)

![](images/0b7c461a5fc278b9e991b1b4473fbf94784272735e4214e3fb751bcc250b845f.jpg)

![](images/fe82cc8ed315e2a5cf644f087a1c1cb2baf2e8597b227a6e8880325282278cef.jpg)

ALLSTAR DHR | CCT Vs ENG  
![](images/718f04127e02ded55aba037e3cf99a4e08a31f80c3a326bdd29193e7c1ce9a79.jpg)

![](images/9b5e10b768716b4009d3b33c2608129a157fe3eb58ed1855eea7c8488ae64fb6.jpg)

![](images/645171af013dc5bffb15ecfc63349e73e29b3a8e882c4da57bf2ae7064b22a9a.jpg)

ALLSTAR\_LPP | VIE vs ENG  
![](images/84eddddc1d2880abff07529243feb97600a8ef8b263c8b4f37292c45c741360e.jpg)

![](images/93b699b07cd305889db1d183853fdc44a3bee451ff933e54be048e27b3e1a250.jpg)  
Fig. 1: Longitudinal system performances on 8 datasets. Top panels: Word Error Rates (WERs) for each accent variety. Bottom panels: WER difference between Mainstream US English (MUSE) or Standard Southern British English (SSBE) and the minority accent variety with the highest average WER in the dataset. Bottom panels also show the average performance gap over the time period from the earliest evaluated model (2021-10) to the latest one (2025-11), as well as the correlation between the temporal ordering of the models and the size of the performance gap between the two accent varieties in each evaluation.

TABLE II: Mean Degradation (MD) and MD rank.
<table><tr><td>Dataset</td><td>Background</td><td>MD</td><td>Rank</td></tr><tr><td>L2-Arctic</td><td>Vietnamese En. MUSE</td><td>11.4%* 3.9%</td><td>5/7 1/7</td></tr><tr><td>L2-Arctic-Spontaneous</td><td>Korean En. MUSE</td><td>9.8%* 7.5%</td><td>3/7 2/7</td></tr><tr><td>Speech Accent Archive</td><td>Korean En. MUSE</td><td>11.7% 11.6%</td><td>4/12 3/12</td></tr><tr><td>OpenSLR-83</td><td>Irish En. Southern En.</td><td>16.5% 8.4%</td><td>6/6 5/6</td></tr><tr><td>ALLSTAR-LPP</td><td>Vietnamese En.</td><td>3.6%*</td><td>7/22</td></tr><tr><td>ALLSTAR-DHR</td><td>MUSE Cantonese En.</td><td>2.5% 3.8%*</td><td>2/22 17/21</td></tr><tr><td>ALLSTAR-HT1</td><td>MUSE Nkore En.</td><td>1.2% 11.8%*</td><td>2/21 22/22</td></tr><tr><td>ALLSTAR-HT2</td><td>MUSE Nkore En.</td><td>2.3% 14.1%*</td><td>2/22 22/22</td></tr></table>

## B. Large degradations across successive systems

Here we look into the degradations that are the source of this non-monotonicity. By degradation we mean a system at time i that is worse than a system at time $j ,$ where $i > j .$ . ML system development is a messy, loosely collaborative procedure, carried out by a large number of research teams in diverse settings [54]. It is natural to expect some degradations in this noisy development process, especially since we explicitly selected evaluation datasets that were not in the evaluation suites of the original development teams (see Table I). Even the MUSE variety has some degradations in these datasets, as can be observed for Whisper-V1 in the L2-Arctic evaluation (Figure 1; top-left panel). But here, in alignment with our hypothesis (H1) we demonstrate experimentally that the regressions are much larger for minority accent varieties than for MUSE. To do so we can consider the set of degradations for an accent variety:

$$
{ \mathbb D } = \{ d ( m _ { j } \to m _ { i } ) \mid i \in [ n ] , j \in [ i ] , d ( m _ { j } \to m _ { i } ) > 0 \}
$$

where $d ( m _ { j } \to m _ { i } ) = \mathbf { W E R } ( m _ { i } ) - \mathbf { W E R } ( m _ { j } )$ . We can then compute the average magnitude, or the Mean Degradation (MD).

In Table II, we show the Mean Degradation for both the standard and the selected accent varieties. In all cases, we find the minority accent variety has a higher MD score, indicating performance degradations will be more impactful for these varieties. We test for statistical significance using a bootstrap percentile test [57]: re-sampling the WER distributions for both varieties, re-computing their MD scores, and checking whether the minority accent variety has a higher MD than MUSE. We repeat this procedure with $B = 1 0 0 0$ bootstrap samples. We find that the procedure is statistically significant for all datasets except OpenSLR-83 and Speech Accent Archive (SAA). For OpenSLR-83, we find Irish English (minority) has similar regression patterns to Southern British English (standard).

For SAA, the similar MD scores are explained by a large increase in error rates with $m _ { 3 } = { \mathrm { w h i s p e r - v 1 } }$ for MUSE. whisper-v1 also performed poorly on Korean English, but since prior models $( m _ { 1 } , \ m _ { 2 } )$ also struggled with Korean English, the regression $( d ( m _ { 2 }  m _ { 3 } )$ or $d ( m _ { 1 }  m _ { 3 } ) )$ ) was not nearly as pronounced.

a) Ordinal rankings of accent varieties.: In each dataset, we rank the varieties in ascending order of MD. MUSE ranks 1/7 for L2-Arctic, 2/7 for L2-Arctic Spontaneous, 3/12 for Speech Accent Archive, 2/22 for ALLSTAR-LPP, 2/21 for ALLSTAR-DHR, and 2/22 for both ALLSTAR-HT1 and ALLSTAR-HT2. In OpenSLR-83, the Southern English control ranks 5/6 and Irish English ranks 6/6. In every dataset, the selected minority accent variety has a higher MD rank than its standard-accent control. These results support greater average severity of performance degradations for the selected minority varieties.

b) Large degradations across successive systems.: The degradations between successive iterations of systems for minority accent varieties can be dramatic. We highlight the largest degradation for the selected minority accent variety in Figure 1. This is especially notable with $m _ { 5 } ~ = ~ \mathrm { { \cal M M S } } ,$ where $d ( m _ { 4 }  m _ { 5 } ) = 1 8 \%$ (Speech Accent Archive), as well as $d ( m _ { 4 } { \phantom { - } }  m _ { 5 } { \phantom { - } } ) = 4 5 \% ( \tt R L L S T A R { - } \mathrm { H T 1 } )$ $\begin{array} { c c c c c } { { d ( m _ { 4 } } } & { {  } } & { { m _ { 5 } ) } } & { { = } } & { { 6 5 \% } } \end{array}$ (ALLSTAR-HT2), $d ( m _ { 4 } \textrm { ~ } $ $m _ { 5 } ) ~ = ~ 7 \%$ (ALLSTAR-LPP), and $d ( m _ { 4 } \  \ m _ { 5 } ) \ = \ 9 \%$ (ALLSTAR-DHR). We find that the $m _ { 1 6 } = { \tt O m n i l i n g u a l }$ model, the most recent model we assessed, also degraded on some varieties: $d ( m _ { 1 4 } \to m _ { 1 6 } ) = 1 5 . 4 \%$ on L2-Arctic Spontaneous; $d ( m _ { 1 5 }  m _ { 1 6 } ) = 5 . 7 \%$ on OpenSLR-83. In all cases, the degradations for the MUSE variety are considerably smaller. $m _ { 1 4 } = \mathtt { C a n a r y }$ is recognized as the top-performing multilingual ASR model on the OpenASR leaderboard, boasting a 5% error rate on the OpenASR evaluation corpus. However, these error rates nearly double for Cantonese and Nkore-influenced accents in the ALLSTAR corpus $( d ( m _ { 3 }  m _ { 1 4 } ) = 6 . 7 \%$ and $d ( m _ { 4 } \to m _ { 1 4 } ) = 1 0 . 1 \% )$

## V. CONCLUSION

A common fixture in the development process of novel ASR systems is evaluation on standard benchmarks. Among these standard benchmarks, Mainstream US English accents are always represented, often disproportionately so. This conventionalized cultural practice ensures that novel ASR systems exhibit stable performance on this prestige variety. But other World Englishes are often marginalized in the computational operationalization of English, especially those in the Global Majority [39]. We study the effects of this systemic exclusion on a longitudinal basis, investigating performance on marginalized English accent varieties on a number of notable models across a critical four-year period (2021-2025), using sociolinguistically varied speech datasets that are outside of the standard benchmarking milieu. We find that while performance can be improved on these varieties, the selected minority accent varieties exhibit higher mean degradation than their standard-accent controls.

[1] X. Li, “Divination engines: A media history of text prediction,” Ph.D. dissertation, New York University, 2017.

[2] M. Benzeghiba et al., “Automatic speech recognition and speech variability: A review,” Speech communication, vol. 49, no. 10-11, pp. 763– 786, 2007, publisher: Elsevier.

[3] R. Vipperla, S. Renals, and J. Frankel, “Longitudinal study of ASR performance on ageing voices,” in Interspeech, 2008, pp. 2550–2553.

[4] A. Koenecke et al., “Racial disparities in automated speech recognition,” Proceedings of the National Academy of Sciences, vol. 117, no. 14, pp. 7684–7689, Apr. 2020. [Online]. Available: https://pnas.org/doi/full/10.1073/pnas.1915768117

[5] L.-F. Lai and N. R. Holliday, “Exploring Sources of Racial Bias in Automatic Speech Recognition through the Lens of Rhythmic Variation.” in Interspeech, vol. 2023, 2023, pp. 1284–1288.

[6] A. Cercas Curry et al., “Classist Tools: Social Class Correlates with Performance in NLP,” in Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), L.-W. Ku, A. Martins, and V. Srikumar, Eds. Bangkok, Thailand: Association for Computational Linguistics, Aug. 2024, pp. 12 643– 12 655. [Online]. Available: https://aclanthology.org/2024.acl-long.682/

[7] N. Markl, “Language variation and algorithmic bias: understanding algorithmic bias in British English automatic speech recognition,” in Proceedings of the 2022 ACM Conference on Fairness, Accountability, and Transparency, 2022, pp. 521–534.

[8] C. Graham and N. Roll, “Evaluating OpenAI’s Whisper ASR: Performance analysis across diverse accents and speaker traits,” JASA Express Letters, vol. 4, no. 2, 2024, publisher: AIP Publishing.

[9] R. Benjamin, “Race after technology,” in Social Theory Re-Wired. Routledge, 2023, pp. 405–415.

[10] J. Nakatumba-Nabende et al., “A systematic literature review on bias evaluation and mitigation in automatic speech recognition models for low-resource african languages,” ACM Computing Surveys, vol. 58, no. 4, pp. 1–24, 2025.

[11] J. Kaplan et al., “Scaling laws for neural language models,” arXiv preprint arXiv:2001.08361, 2020.

[12] B. Recht, “The actuary’s final word on algorithmic decision making,” arXiv preprint arXiv:2509.04546, 2025.

[13] K. C. Puvvada et al., “Less is More: Accurate Speech Recognition &amp; Translation without Web-Scale Data,” 2024, version Number: 1. [Online]. Available: https://arxiv.org/abs/2406.19674

[14] Y. Chu et al., “Qwen2-Audio Technical Report,” 2024, version Number: 1. [Online]. Available: https://arxiv.org/abs/2407.10759

[15] A. Radford et al., “Robust Speech Recognition via Large-Scale Weak Supervision,” 2022, version Number: 1. [Online]. Available: https://arxiv.org/abs/2212.04356

[16] B. Hutchinson et al., “Towards accountability for machine learning datasets: Practices from software engineering and infrastructure,” in Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency, ser. FAccT ’21. New York, NY, USA: Association for Computing Machinery, 2021, p. 560–575. [Online]. Available: https://doi.org/10.1145/3442188.3445918

[17] R. Denton et al., “Bringing the people back in: Contesting benchmark machine learning datasets,” arXiv preprint arXiv:2007.07399, 2020.

[18] G. C. Bowker and S. L. Star, Sorting things out: Classification and its consequences. MIT press, 2000.

[19] M. K. Scheuerman, A. Hanna, and R. Denton, “Do datasets have politics? disciplinary values in computer vision dataset development,” Proceedings of the ACM on Human-Computer Interaction, vol. 5, no. CSCW2, pp. 1–37, 2021.

[20] N. Markl and S. J. McNulty, “Language technology practitioners as language managers: arbitrating data bias and predictive bias in ASR,” in Proceedings of the Thirteenth Language Resources and Evaluation Conference, N. Calzolari et al., Eds. Marseille, France: European Language Resources Association, Jun. 2022, pp. 6328–6339. [Online]. Available: https://aclanthology.org/2022.lrec-1.680/

[21] D. Amodei et al., “Deep speech 2: End-to-end speech recognition in english and mandarin,” in International conference on machine learning. PMLR, 2016, pp. 173–182.

[22] S. Chen et al., “WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing,” 2021, publisher: arXiv Version Number: 5. [Online]. Available: https://arxiv.org/abs/2110.13900

[23] V. Pratap et al., “Scaling Speech Technology to 1,000+ Languages,” 2023, version Number: 1. [Online]. Available: https://arxiv.org/abs/ 2305.13516

[24] Y. Peng et al., “Reproducing Whisper-Style Training Using an Open-Source Toolkit and Publicly Available Data,” 2023, version Number: 3. [Online]. Available: https://arxiv.org/abs/2309.13876

[25] L. Barrault et al., “Seamless: Multilingual expressive and streaming speech translation,” arXiv preprint arXiv:2312.05187, 2023.

[26] Y. Zhang et al., “Google USM: Scaling Automatic Speech Recognition Beyond 100 Languages,” 2023, version Number: 3. [Online]. Available: https://arxiv.org/abs/2303.01037

[27] Omnilingual ASR team et al., “Omnilingual ASR: Open-Source Multilingual Speech Recognition for 1600+ Languages,” 2025, version Number: 1. [Online]. Available: https://arxiv.org/abs/2511.09690

[28] S. Gururangan et al., “Don’t stop pretraining: Adapt language models to domains and tasks,” in Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, D. Jurafsky et al., Eds. Online: Association for Computational Linguistics, Jul. 2020, pp. 8342– 8360. [Online]. Available: https://aclanthology.org/2020.acl-main.740/

[29] S. wen Yang et al., “Superb: Speech processing universal performance benchmark,” 2021. [Online]. Available: https://arxiv.org/abs/2105.01051

[30] V. Srivastav et al., “Open asr leaderboard: Towards reproducible and transparent multilingual and long-form speech recognition evaluation,” 2025. [Online]. Available: https://arxiv.org/abs/2510.06961

[31] W. Chan et al., “Speechstew: Simply mix all available speech recognition data to train one large neural network,” arXiv preprint arXiv:2104.02133, 2021.

[32] L. Lucy et al., “AboutMe: Using self-descriptions in webpages to document the effects of English pretraining data filters,” in Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), L.-W. Ku, A. Martins, and V. Srikumar, Eds. Bangkok, Thailand: Association for Computational Linguistics, Aug. 2024, pp. 7393–7420. [Online]. Available: https://aclanthology.org/2024.acl-long.400/

[33] W. Agnew et al., “Sound check: Auditing recent audio dataset practices,” in Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society, vol. 8, no. 1, 2025, pp. 26–40.

[34] B. Schneider, “Methodological nationalism in Linguistics,” Language Sciences, vol. 76, p. 101169, Nov. 2019. [Online]. Available: https://linkinghub.elsevier.com/retrieve/pii/S0388000117303480

[35] V. Pratap et al., “Mls: A large-scale multilingual dataset for speech research,” arXiv preprint arXiv:2012.03411, 2020.

[36] A. Conneau et al., “FLEURS: FEW-Shot Learning Evaluation of Universal Representations of Speech,” in Proceedings of SLT, Doha, Qatar, Jan. 2023, pp. 798–805. [Online]. Available: https: //ieeexplore.ieee.org/document/10023141/

[37] R. Ardila et al., “Common voice: A massively-multilingual speech corpus,” in Proceedings LREC, 2020, pp. 4211–4215.

[38] C. Wang et al., “VoxPopuli: A large-scale multilingual speech corpus for representation learning, semi-supervised learning and interpretation,” in Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), C. Zong et al., Eds. Online: Association for Computational Linguistics, Aug. 2021, pp. 993–1003. [Online]. Available: https://aclanthology.org/2021.acl-long.80/

[39] N. Markl, “Mind the data gap(s): Investigating power in speech and language datasets,” in Proceedings of the Second Workshop on Language Technology for Equality, Diversity and Inclusion, B. R. Chakravarthi et al., Eds. Dublin, Ireland: Association for Computational Linguistics, May 2022, pp. 1–12. [Online]. Available: https://aclanthology.org/2022.ltedi-1.1/

[40] K. Reid and E. T. Williams, “Common voice and accent choice: data contributors self-describe their spoken accents in diverse ways,” in Proceedings of the 3rd ACM Conference on Equity and Access in Algorithms, Mechanisms, and Optimization, ser. EAAMO ’23. New York, NY, USA: Association for Computing Machinery, 2023. [Online]. Available: https://doi.org/10.1145/3617694.3623258

[41] A. Birhane et al., “The Values Encoded in Machine Learning Research,” in 2022 ACM Conference on Fairness Accountability and Transparency. Seoul Republic of Korea: ACM, Jun. 2022, pp. 173–184. [Online]. Available: https://dl.acm.org/doi/10.1145/3531146.3533083

[42] Linguistic Data Consortium, “CABank English CallHome Corpus,” 2013. [Online]. Available: https://ca.talkbank.org/access/CallHome/eng. html

[43] D. Graff, A. Canavan, and G. Zipperlen, “Switchboard-2 Phase I (LDC98S75),” 1998, place: Philadelphia Published: Web Download. [Online]. Available: https://catalog.ldc.upenn.edu/LDC98S75

[44] D. B. Paul and J. M. Baker, “The Design for the Wall Street Journalbased CSR Corpus,” in Speech and Natural Language: Proceedings of a Workshop Held at Harriman, New York, February 23-26, 1992, 1992. [Online]. Available: https://aclanthology.org/H92-1073/

[45] A. Canavan, D. Graff, and G. Zipperlen, “CALLHOME American English Speech (LDC97S42),” 1997, iSBN: 1-58563-111-6 Place: Philadelphia Published: Web Download. [Online]. Available: https: //catalog.ldc.upenn.edu/LDC97S42

[46] E. M. Bender et al., “On the dangers of stochastic parrots: Can language models be too big?” in Proceedings of the 2021 ACM conference on fairness, accountability, and transparency, 2021, pp. 610–623.

[47] A. Rousseau, P. Deleglise, and Y. Est´ eve, “TED-LIUM: an automatic\` speech recognition dedicated corpus,” in Proceedings of LREC, N. Calzolari et al., Eds. Istanbul, Turkey: European Language Resources Association (ELRA), May 2012, pp. 125–129. [Online]. Available: https://aclanthology.org/L12-1405/

[48] V. Panayotov et al., “Librispeech: An ASR corpus based on public domain audio books,” in Proceedings of ICASSP, Apr. 2015, pp. 5206–5210, iSSN: 2379-190X. [Online]. Available: https: //ieeexplore.ieee.org/document/7178964

[49] S. Weinberger, “Speech accent archive,” George Mason University, 2015.

[50] G. Zhao et al., “L2-arctic: A non-native english speech corpus,” 2018.

[51] I. Demirsahin et al., “Open-source Multi-speaker Corpora of the English Accents in the British Isles,” in Proceedings of LREC. Marseille, France: European Language Resources Association (ELRA), May 2020, pp. 6532–6541. [Online]. Available: https://www.aclweb. org/anthology/2020.lrec-1.804

[52] A. R. Bradlow, “Allsstar: Archive of l1 and l2 scripted and spontaneous transcripts and recordings,” https://speechbox.linguistics.northwestern. edu/allsstar, n.d., accessed: 2025.

[53] M. Nilsson, S. D. Soli, and J. A. Sullivan, “Development of the hearing in noise test for the measurement of speech reception thresholds in quiet and in noise,” The journal of the acoustical society of America, vol. 95, no. 2, pp. 1085–1099, 1994.

[54] D. Donoho, “Data science at the singularity,” arXiv preprint arXiv:2310.00865, 2023.

[55] S. Ott et al., “Mapping global dynamics of benchmark creation and saturation in artificial intelligence,” Nature Communications, vol. 13, no. 1, p. 6793, 2022.

[56] S. Sagawa et al., “Distributionally robust neural networks for group shifts: On the importance of regularization for worst-case generalization,” arXiv preprint arXiv:1911.08731, 2019.

[57] B. Efron and R. J. Tibshirani, An introduction to the bootstrap. Chapman and Hall/CRC, 1994.

[58] Voxforge.org, “Free speech recognition (linux, windows and mac) - voxforge.org,” http://www.voxforge.org/, accessed 1/1/2026.

[59] J. Barker et al., “The Fifth ’CHiME’ Speech Separation and Recognition Challenge: Dataset, Task and Baselines,” in Interspeech, 2018, pp. 1561– 1565.

[60] T. Kendall and C. Farrington, “The corpus of regional african american language,” Version, vol. 6, p. 1, 2018.

[61] J. Carletta et al., “The ami meeting corpus: a pre-announcement.” Berlin, Heidelberg: Springer-Verlag, 2005. [Online]. Available: https: //doi.org/10.1007/11677482 3

[62] J. Meyer et al., “Artie bias corpus: An open dataset for detecting demographic bias in speech applications,” in Proceedings of the Twelfth Language Resources and Evaluation Conference, N. Calzolari et al., Eds. Marseille, France: European Language Resources Association, May 2020, pp. 6462–6468. [Online]. Available: https: //aclanthology.org/2020.lrec-1.796/

[63] M. Del Rio et al., “Earnings-21: A practical benchmark for asr in the wild,” arXiv preprint arXiv:2104.11348, 2021.

[64] M. D. Rio et al., “Earnings-22: A practical benchmark for accents in the wild,” 2022. [Online]. Available: https://arxiv.org/abs/2203.15591

[65] G. Chen et al., “Gigaspeech: An evolving, multi-domain asr corpus with 10,000 hours of transcribed audio,” in Interspeech,

ser. interspeech. ISCA, Aug. 2021. [Online]. Available: http: //dx.doi.org/10.21437/Interspeech.2021-1965

[66] P. K. O’Neill et al., “Spgispeech: 5,000 hours of transcribed financial audio for fully formatted end-to-end speech recognition,” arXiv preprint arXiv:2104.02014, 2021.

[67] D. Snyder, G. Chen, and D. Povey, “Musan: A music, speech, and noise corpus,” 2015. [Online]. Available: https://arxiv.org/abs/1510.08484

[68] B. Porgali et al., “Casual conversations v2: Designing a large consentdriven dataset for ai fairness,” arXiv preprint arXiv:2211.05809, 2022.

[69] D. Graff et al., “1996 English Broadcast News Speech (HUB4),” LDC97S44. Web Download. Philadelphia: Linguistic Data Consortium, 1997. [Online]. Available: https://catalog.ldc.upenn.edu/LDC97S44

[70] Facebook Research, “facebookresearch/voxpopuli: A large-scale multilingual speech corpus for representation learning, semi-supervised learning and interpretation,” https://github.com/facebookresearch/voxpopuli, 2021, gitHub repository (archived).

[71] A. DiChristofano et al., “Global performance disparities between english-language accents in automatic speech recognition,” arXiv preprint arXiv:2208.01157, 2022.

[72] M. P. Y. Chan et al., “Training and typological bias in asr performance for world englishes.” in Interspeech, 2022, pp. 1273–1277.

[73] J. Kominek and A. Black, “The cmu arctic speech databases,” SSW5- 2004, 01 2004.

[74] T. M. Derwing et al., “The relationship between l1 fluency and l2 fluency development,” Studies in Second Language Acquisition, vol. 31, no. 4, p. 533–557, 2009.

## A. Benchmark Details

The specific benchmark datasets used by each model in our analysis are listed in Table III.

In Table IV, we list the subset of benchmarks that are explicitly concerned with multi-accent representation, as demonstrated by pull quotes from the respective papers describing the benchmark.

## B. Dataset Descriptions

a) Speech Accent Archive: Speech Accent Archive [49] is an English speech corpus containing both non-standard and standard speakers reading the same passage: “Please call Stella. Ask her to bring these things with her from the store: Six spoons offresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob. We also need a small plastic snake and a big toy frog for the kids. She can scoop these things into three red bags, and we will go meet her Wednesday at the train station.”

It has previously been used for assessing performance disparities in ASR systems across different Global English varieties [8], [49], [71], [72]. The dataset controls for the noise level and the distance to the microphone. Each subject is given one minute to read over the passage and ask questions about unfamiliar words before recording. However, this was done by many different researchers, so the quality and procedure does differ between samples. As of writing, the corpus contains 3038 samples across 392 standard languages. The corpus contains a smaller subset of high quality expert-annotated samples. This constitutes approximately 10 hours of speech from 1238 speakers across 831 birthplaces and 190 standard languages. Of these speakers, 181 are listed as standard English speakers. We only kept accents where 20+ speakers had contributed a recording. Each sample has median duration of 26 seconds (8 second std.), varying by speaking rate.

b) L2 ARCTIC: L2 ARCTIC [50] contains about 1 hour of scripted English from each of 24 non-standard speakers, utilizing source texts originally derived from Project Gutenberg [73]. The speakers are evenly divided, 2 male and 2 female, between standard Arabic, Hindi, Vietnamese, Spanish, Korean, and Mandarin. There is also an unscripted subset, the “suitcase” subset from 22 of the speakers where they were prompted with the suitcase story [74] and asked to explain what was happening in the pictures. About 1 hour of the scripted subset is expert-annotated (3,599 samples, 150 utterances for each speaker). The entire suitcase subset is expert-annotated. Additionally, we record 1 sample of L1 North American English speech by one of the authors using the suitcase prompt, given that L2 Arctic does not contain an unscripted sample spoken by a standard English speaker.

c) OpenSLR83: OpenSLR83 is a dataset of transcribed speech featuring English speakers from various dialectal regions across the UK and Ireland. The original dataset comprises 17,877 utterances across nine dialect-gender combinations: Irish English, Midlands English, Northern English, Scottish English, Southern English, and Welsh English.

d) ALLSSTAR: ALLSSTAR Corpus [52] is a multilingual speech dataset containing paired recordings from each participant in both their first language (L1), and in English as a second language (L2). The dataset features 102 L2 English speakers representing 21 standard language backgrounds, along with 26 standard English speakers.

The corpus comprises both read and spontaneous speech (e.g., picture descriptions and question–answer sessions); we focused on the read-speech portion for this dataset. The read materials include short sentences and paragraphs, with each speaker producing 120 unique sentences and 3 unique paragraphs. Sentence recordings were collected in two sessions, each captured as a single continuous audio file. The paragraph topics were: Little Prince (LPP), Declaration of Human Rights (DHR), and The North Wind and the Sun (NWS). In total, the dataset includes 129 recordings from L1 English speakers and 570 from L2 English speakers. The final analyzed subset comprised 10,753 short sentences (2,862 L1; 7891 L2), 2,548 LPP recordings (727 L1; 1,821 L2), and 1,869 DHR paragraphs (514 L1; 1,355 L2). Recording duration varied by material type, with short sentences averaging 1.68 seconds, LPP averaging 4.16 seconds, and DHR paragraphs averaging 6.04 seconds.

## C. Preprocessing Details

Here we provide additional details for filtering out some samples from the evaluation datasets we applied in our analysis.

a) Speech Accent Archive.: Each recording is published as a single wave file. The expert annotated subset comes with phonetic and text transcriptions in a Praat TextGrid file. We keep only the expert annotated subset. We remove untranscribed sections of the audio labeled as “chatter”, “selfintroduction”, “self-talk”, “experimenter comments”, “background speech”, “noise”, “bird song”, “cough” and any variant mispellings thereof. Furthermore, we standardize numbers to text form (“2” → “two”), fix typos, and clip audio into consistent duration segments of 50 phonemes each.

b) L2-Arctic.: Each utterance is published as a single wave file with a corresponding TextGrid file containing phonetic and text transcriptions. Except for the expert-annotated subset, these are machine generated. We keep only the high quality, expert-annotated subsets. In this subset, each scripted utterance is about 3.6 seconds of speech. We clip the suitcase utterances to be approximately the same length. As part of the clipping process, we remove audio that hasn’t been annotated, e.g., the interviewer speaking or the subject mumbling.

c) OpenSLR83.: We remove utterances containing fewer than 4 words. To maintain balanced representation across dialects and genders, we select 100 samples per dialect with equal gender distribution (50 female, 50 male samples per dialect). For Irish English, which contains only male speakers, we select 50 samples. This yields a balanced subset for analysis.

d) ALLSTAR.: Each recording is accompanied by Praat TextGrid annotations at the sentence, word, and phoneme levels. For the present analysis, full-session audio files were segmented into individual sentence-level utterances based on these annotations, and NWS paragraph recordings were excluded due to the absence of utterance-level time alignments. Instances in which the recordings did not correspond to the reference text were automatically filtered out by removing utterances for which all ASR models produced a word error rate greater than 100.

TABLE III: Models evaluated in our diachronic analysis. Parentheses indicate additional versions from the same model family. Datasets that explicitly evaluate accent variety within English are underlined.
<table><tr><td>Model</td><td>Release date</td><td>Organization</td><td>Paper</td><td>Evaluation Datasets (ASR)</td></tr><tr><td>Deepspeech (m1)</td><td>2021-10</td><td>Mozilla</td><td>[21]</td><td>WSJ [44],Librispeech [48],VoxForge [58],CHiME [59],</td></tr><tr><td>WavLM (m2)</td><td>2021-12</td><td>Microsoft</td><td>[22]</td><td>CommonVoice-EN [37],MultilinguaI Librispeech [35] Librispeech [48]</td></tr><tr><td>whisper (m3, (m4, m8))</td><td>2022-09</td><td>OpenAI</td><td>[15]</td><td>Librispeech [48],WSJ [44],Switchboard [43],CORAAL [60], TedLium [47],CallHome [45],VoxPopuli En [38], Fleurs [36],AMI Meeting corpus [61],CommonVoice [37], CHiME</td></tr><tr><td>MMS (m5)</td><td>2023-05</td><td>Meta</td><td>[23]</td><td>[59],Artie Bias Corpus [62],Earnings21 [63],Earnings22 [64] Fleurs [36],CommonVoice [37],VoxPopuli [38],Multilingual</td></tr><tr><td>OWSM (m6(m7, m10, m12))</td><td>2023-09</td><td>CMU</td><td>[24]</td><td>Librispeech [35],MMS-lab (bible) [23] CommonVoice-EN [37],Fleurs [36],Librispeech [48], TedLium [47],VoxPopuli [38],WSJ [44],Switchboard [43], Multilingual</td></tr><tr><td>SeamlessM4T (m9)</td><td>2023-12</td><td>Meta</td><td>[25]</td><td>Librispeech [35] Fleurs [36]</td></tr><tr><td>Qwen-Audio (m11(m13)) Canary-Qwen (m14)</td><td>2024-08</td><td>Alibaba</td><td>[14]</td><td>Librispeech [48],CommonVoice-EN [37],Fleurs [36] AMI Meeting corpus [61],Earnings22 [64],Gigaspeech [65],</td></tr><tr><td></td><td>2025-07</td><td>NVIDIA</td><td>[13]</td><td>SPGIspeech [66],Tedlium [47],Voxpopuli [38],MUSAN [67], CasualConversations Dataset [68],Multilingual Librispeech [35]</td></tr><tr><td>Chirp-3 (m15)</td><td>2025-10</td><td>Google</td><td>[26]</td><td>AMI Meeting Corpus [61],Broadcast News [69],Common Voice [37],LibriSpeech [48],Switchboard [43],Tedlium [47],WSJ, Fleurs</td></tr><tr><td>Omnilingual (m16)</td><td>2025-11</td><td>Meta</td><td>[27]</td><td>[36],Youtube captions MMS-Lab (bible) [23], Fleurs [36],MLS [35],CommonVoice [37], AllASR [27]</td></tr></table>

TABLE IV: Datasets that are explictly concerned with minority accent representation.
<table><tr><td>VoxForge</td><td>“Our source for accented speech is the publicly available VoxForge (http://www.voxforge.org) dataset, which has clean speech read</td></tr><tr><td></td><td>from speakers with many different accents.&quot; [21] “The Artie Bias Corpus contains information on 3 gender classes, 17 English accents, and 8 age ranges.&quot; [62]</td></tr><tr><td>Artie VoxPopuli-En</td><td>&quot;VoxPopuli provides 29 hours of transcribed speech data of non-native English intended for research in ASR for accented speech&quot;</td></tr><tr><td></td><td>[70]</td></tr><tr><td>Earnings22 SPGISpeech</td><td>“Our attention focused on aggregation of accented public English-language earnings calls from global companies.&quot; [64] “There are roughly 50,000 speakers in SPGISpeech, drawn from corporate officers and analysts appearing in English earnings calls</td></tr><tr><td>CommonVoice-EN</td><td>and spanning a broad cross-section of L1 and L2 accents&quot; [66] “The 2021 release of Common Voice English (7.0) contains 2,015 hours of (validated) speech submitted by over 75,000 speakers</td></tr><tr><td>AMI Meeting Corpus</td><td>some of whom opted to provide some information about their gender and accent&quot; [39] “The meetings were recorded in English using three different rooms with different acoustic properties, and include mostly non-native speakers&quot; [61]</td></tr></table>

## D. Accent and Demographic Information

See Table V.

## E. System Degradation Examples

In Table VI, we list examples of other pairs of models, one pair for each dataset we evaluated on the minority accent we highlighted in Figure 1. For the selected pair of models $m _ { i }$ and $m _ { j } .$ , we compute the average degradation across the entire dataset, listed in the second column of Table VI, and select a random example from the top quartile of the degradation distribution across the dataset.

TABLE V: Summary of evaluation datasets. Duration is given in seconds per sample as the interquartile range (IQR). All datasets contain scripted speech except L2-Arctic Spontaneous.
<table><tr><td>Dataset</td><td>Duration  $( \mathbf { s } ,$  IQR)</td><td>Samples</td><td>Accents</td><td>Speaker Demographics</td></tr><tr><td>Speech Accent Archive</td><td>5.4-7.2</td><td>1,238</td><td>nld, spa, ita, eng, fra, kor, tur, ara, por, deu, zho, rus</td><td>Mean age 33.31 (SD 13.57); started English at 10.53 (SD 8.49); 55.7% male.</td></tr><tr><td>L2-Arctic</td><td>2.8-4.4</td><td>3,599</td><td>vie, cmn, hin, spa, ara, kor</td><td>Iowa State University students; age 22–43, mean 29 (SD 6.9).</td></tr><tr><td>L2-Arctic Spontaneous</td><td>6.2-7.3</td><td>22</td><td>vie, cmn, hin, spa, ara, kor</td><td>Same as L2-Arctic.</td></tr><tr><td>OpenSLR83</td><td>4.6-7.7</td><td>17,877</td><td>Irish, Midlands, North- ern, Scottish, Southern, Welsh</td><td>Volunteers 21+; recruited by Google London and Cardiff University.</td></tr><tr><td>ALLSTAR</td><td>1.5-2.4</td><td>42,398</td><td>pbr, cct, kor, guj, vie, ctw, run, hin, fra, rus, gis, csp, gre, ind, spa, heb, tur, ger, far, shs,</td><td>Northwestern University students; age 18–41, mean 23.52.</td></tr></table>

TABLE VI: Illustration of degradation from earlier model $m _ { i }$ to later model $m _ { j }$ across all of our evaluation datasets.
<table><tr><td>Dataset (Accent)</td><td> $\bf \bar { d } _ { ( m _ { i } \to m _ { j } ) }$ </td><td>Prediction  $( m _ { i } )$ </td><td>Prediction  $( m _ { j } )$ </td><td>Transcript</td></tr><tr><td>L2-Arctic (Vietnamese)</td><td>12.5%</td><td>Whisper Large v2 (2022-12): With- out a doubt, some of them have dinner engagements.</td><td>MMS 1B All (2023-05): wit dou a doubt some of them hapdiner engagements</td><td>without a doubt some of them have dinner engage- ments</td></tr><tr><td>L2-Arctic Spontaneous (Korean)</td><td>15.4%</td><td>Canary (2025-07): Okay, um I can see skyscrapers tall buildings and I think</td><td>Omnilingual 7B (2025-11): i can see skyscripers tall buildings and i think</td><td>ok uh i can see skyscrap- ers tall buildings and i think</td></tr><tr><td>OpenSLR-83 (Irish)</td><td>5.7%</td><td>Google Chirp 3 (2025-10): Your app facilitates or promotes content that includes gratuitous violence or dan- gerous activities.</td><td>Omnilingual 7B (2025-11): your off facilitates or promotes con- tent that includes gratuitous fly- lence or dangerous activities</td><td>Your app facilitates or promotes content that in- cludes gratuitous violence or dangerous activities</td></tr><tr><td>SAA (Korean)</td><td>18.3%</td><td>Whisper Large v2 (2022-12): Please call Stella. Ask her to bring these things with her from the store.</td><td>MMS 1B All (2023-05): please costella escr to bring these things with her from the store</td><td>please call stella Ask her to bring these things with her from the store.</td></tr><tr><td>ALLSTAR-HT1 (Nkore)</td><td>6.7%</td><td>Whisper Large v1 (2022-09): the salt shaker is empty</td><td>Canary (2025-07): the sword sheikah is empty</td><td>the salt shaker is empty</td></tr><tr><td>ALLSTAR-HT2 (Nkore)</td><td>3.1%</td><td>Whisper Large v1 (2022-09): he is washing his face with soap</td><td>Omnilingual (2025-11): he is washing his face with sore</td><td>he is washing his face with soap</td></tr><tr><td>ALLSTAR-LPP (Nkore)</td><td>10.1%</td><td>Whisper Large v2 (2022-12): draw me a sheep</td><td>Canary (2025-07): turn me a say</td><td>draw me a sheep</td></tr><tr><td>ALLSTAR-DHR (Cantonese)</td><td>6.1%</td><td>Whisper Large v3 (2023-11): every- one has the right to a nationality</td><td>OWSM 3.1 (2024-01): every one has to waittt snrt</td><td>everyone has the right to a nationality</td></tr></table>