# Can LLMs Follow the Pulse of a Crisis? Evaluating Crisis Sentiment in Bangladesh’s July Uprising

Md. Samiul Alim<sup>1,</sup>∗ Mahir Shahriar Tamim<sup>1,</sup>∗ Tanvir Ahmed Khan<sup>1</sup> Sharjil Khan<sup>1</sup> Rafia Ferdous Duti<sup>1</sup> Shahriyar Zaman Ridoy<sup>1</sup> Mohammad Ali Moni<sup>2,</sup>†

<sup>1</sup>North South University, Dhaka, Bangladesh <sup>2</sup>Charles Sturt University, Australia

samiul.alim01@northsouth.edu, mahir.tamim@northsouth.edu, khan.tanvir01@northsouth.edu, sharjil.khan@northsouth.edu, rafia.duti@northsouth.edu, shahriyar.zaman01@gmail.com, mmoni@csu.edu.au

![](images/da001e8510ce11143c7efba4b9c4a7a7cf351a0823c8b08c5369f6618f3df1fd.jpg)  
Figure 1: Grafiti from the July 2024 student-led uprising in Bangladesh, illustrating public expressions of resistance, solidarity, generational identity, and political sentiment captured on urban walls.

## Abstract

Crisis sentiment analysis is especially challenging for low-resource languages such as Bangla, where language, context, and public reaction shift rapidly. We introduce UNREST-SENT200K, a Bangla crisis sentiment dataset with ≈200K Facebook and YouTube comments from the July–August 2024 Bangladesh uprising. The dataset covers five event-aligned phases, from early escalation and internet blackout to regime transition and a later flood crisis. Each comment is linked to its parent post, enabling evaluation with and without discourse context. All comments are annotated through a fully human process involving 14 native Bangla-speaking annotators and senior validation, achieving substantial agreement (κ = 0.73, α = 0.71) and 94.2% blind-audit agreement. We benchmark fine-tuned encoders, prompted LLMs, and LoRA-tuned LLMs. Results show that parent-post context consistently improves performance, while temporal shift across phases causes large performance drops. Strong LLMs perform well, but still struggle with sarcasm, implicit political references, and phase-dependent meaning. UNREST-SENT200K provides a benchmark for studying context-aware and temporally robust sentiment analysis in low-resource crisis discourse.

UNRESTSENT200K is available at https:// sami0055.github.io/UNRESTSENT200K/

## 1 Introduction

During a political crisis, public language can change almost overnight. A short social-media comment that appears positive at one moment may sound sarcastic, fearful, or critical after a major event. Its meaning may also depend on the news post to which it responds. Yet most sentiment benchmarks treat language as stable over time, evaluate each utterance in isolation, and focus mainly on English (Socher et al., 2013; Maas et al., 2011; Go et al., 2009). These assumptions are especially fragile in crisis settings, where sentiment systems are increasingly used to support situational awareness, journalism, and policy decisions (Tufekci, 2017; STEINERT-THRELKELD, 2017). The central problem is therefore not only a lack of data. It is also a lack of evaluation settings that show whether models can follow a rapidly changing public conversation.

The July–August 2024 Bangladesh uprising (Rana et al., 2026) ofers a rare opportunity to study this problem (Figure 1). In only seven weeks, public discussion on Facebook and YouTube moved through five distinct phases: early mobilisation, a nationwide internet blackout, regime collapse, political transition, and an overlapping flood crisis. The language, platforms, and broad user population remained largely the same, but the social and political context changed sharply. This creates a natural stress test: can a model trained during one phase still understand sentiment in the next, and can it interpret a comment without seeing the post that prompted it?

For Bangla, these questions have been dificult to answer. Existing resources such as SentNoB, SentiGOLD, and BnSentMix (Islam et al., 2021, 2023; Alam et al., 2025) have made important progress, but they are mostly static, decontextualised, and focused on consumer or general socialmedia text. They do not divide a single unfolding event into meaningful time periods, link comments to their parent posts, or support controlled comparisons across phases. As a result, they cannot reveal how quickly model performance changes as an event develops. To fill this gap, we introduce UNRESTSENT200K, the largest Bangla sentiment dataset to date, with approximately 200K comments (2.8× SentiGOLD), as summarised in Table 1. Every comment is timestamped, assigned to one of five event-aligned phases, and paired with its parent post. The labels were produced entirely by 14 native Bangla-speaking undergraduate annotators who had first-hand exposure to the events. Our five-stage Pilot Label Annotation (PiLA) framework includes calibration, double annotation, senior adjudication, and an independent blind audit. It achieves substantial agreement (Cohen’s κ = 0.73 and Krippendorf’s α = 0.71) and 94.2% agreement in the blind audit.

Our experiments tell a consistent story. We evaluate four fine-tuned encoders (BanglaBERT, mBERT, XLM-R, and BanglaElectra), 15 zeroand few-shot LLMs, and two LoRA-tuned LLMs. Giving an encoder the parent post improves performance by 7–11 percentage points, showing that context is often essential rather than optional. However, context alone does not solve the problem: training on earlier phases and testing on later ones causes drops of 19–28 F1 points. Sentiment changes occur in sharp, event-linked phases rather than along a smooth trend. GPT-4o-mini, the strongest prompted model, reaches approximately 76% few-shot accuracy but shares several failure modes with the encoders. LoRA-tuned Gemma-4B performs best overall at 78.4% accuracy, yet even this result remains below 80%. Together, these findings show that larger models and stronger prompting help, but do not remove the need for contextual grounding and temporal adaptation.

## Contributions.

• A large Bangla crisis benchmark. We introduce UNRESTSENT200K, containing approximately 200K comments from the July–August 2024 Bangladesh uprising.

• A context- and time-aware design. Each comment is linked to its parent post and an event phase, enabling direct tests of contextual understanding and temporal robustness.

• Careful human annotation. PiLA provides a five-stage process with native Bangla-speaking annotators, senior adjudication, and an independent blind audit.

• Evidence across model families. Experiments with fine-tuned encoders, prompted LLMs, and LoRA-tuned LLMs show clear gains from parent-post context and large losses under temporal shift.

## 2 Related Work

Sentiment benchmarks and the low-resource gap. Sentiment analysis is driven by large static English benchmarks (Socher et al., 2013; Maas et al., 2011; McAuley et al., 2015; Go et al., 2009; Hu and Liu, 2004). In Bangla, SentNoB (Islam et al., 2021) introduced noisy multi-domain social-media comments (15K), SentiGOLD (Islam et al., 2023) scaled to 70K with fine-grained labels, and BnSentMix (Alam et al., 2025) targeted code-mixed text. Pretrained Bangla encoders (Bhattacharjee et al., 2022) and recent taskspecific resources (Kundu et al., 2026; Nayeem et al., 2026; Islam and Hossen, 2025; Hasan et al., 2024; Talukder, 2025; Rashid et al., 2024) have raised the ceiling, but all are static, decontextualised, and consumer-oriented; none is anchored to a single event whose internal phases can be used to probe within-episode shift (Hossen et al., 2025; Olsen et al., 2024).

Temporal and contextual robustness. Prior work has studied concept drift and temporal generalisation in social-media classification (Tufekci, 2017; STEINERT-THRELKELD, 2017), but often across years, topics, or domains, where temporal change is confounded with demographic and topical shifts. In contrast, UNRESTSENT200K isolates temporal shift within a single seven-week crisis, while keeping language, platform, and population largely fixed. It also operationalises contextual grounding by pairing each comment with its parent post, enabling direct comment-only versus post+comment evaluation.

Table 1: Comparison of Bangla social-media datasets across sentiment, hate-speech, and ofensive-language benchmarks.
<table><tr><td>Dataset</td><td>Size</td><td>Task</td><td>Domain</td></tr><tr><td>SentNoB (Islam et al., 2021)</td><td>15K</td><td>Sent.</td><td>Multi</td></tr><tr><td>SentiGOLD (Islam et al., 2023)</td><td>70K</td><td>Sent.</td><td>Multi</td></tr><tr><td>BnSentMix (Alam et al., 2025)</td><td>20K</td><td>Sent.</td><td>Code-mix</td></tr><tr><td>BD-SHS (Romim et al., 2022)</td><td>50K</td><td>HS</td><td>Social</td></tr><tr><td>Bengali Tweets (Das et al., 2022)</td><td>10K</td><td>Hate</td><td>Code-mix</td></tr><tr><td>TB-OLID (Raihan et al., 2023)</td><td>5K</td><td>Offensive</td><td>Transliterated</td></tr><tr><td>BanTH (Haider et al., 2025)</td><td>37K</td><td>HS</td><td>Transliterated</td></tr><tr><td>BIDWESH (Fayaz et al., 2025)</td><td>9K</td><td>HS</td><td>Dialectal</td></tr><tr><td>BanglaMultiHate (Hasan et al., 2026)</td><td>50K</td><td>HS</td><td>YouTube</td></tr><tr><td>UNRESTSENT200K (Ours)</td><td>≈200K</td><td>Sent.</td><td>Crisis</td></tr></table>

Human annotation for sensitive discourse. Politically sensitive crisis discourse requires annotators with strong linguistic and cultural grounding. Following best practices in corpus annotation (Pustejovsky, 2012; Sabou et al., 2014; Mukta et al., 2021), we use a fully human five-stage PiLA protocol with guideline piloting, double annotation, senior adjudication, and blind audit. The process involved 14 native-Bangla annotators with first-hand familiarity with the events and achieved substantial reliability: κ = 0.73, α = 0.71, and 94.2% blind-audit agreement. Thus, UNRESTSENT200K contributes not only a benchmark, but also a replicable annotation protocol for low-resource, politically sensitive NLP.

## 3 Dataset

We construct UNRESTSENT200K, a Bangla sentiment dataset based on public Facebook and YouTube discussions about the July–August 2024 Bangladesh uprising (Rana et al., 2026). The dataset contains ≈200K annotated comments. Each comment is linked to its parent post, which allows us to evaluate sentiment classification in both comment-only and post+comment settings. Each comment is assigned one of three sentiment labels: POSITIVE, NEUTRAL, or NEGATIVE. For each instance, we retain the comment text, parent post, timestamp, platform, and event phase.

## 3.1 Data Collection

We collected public posts and comments from Facebook and YouTube. The posts were selected from verified news pages, public-interest groups, and high-engagement discussion threads related to the July–August 2024 uprising. We used the Apify platform<sup>1</sup>, the Facebook Graph API, and the YouTube Data API v3 for data collection. The collection period covers July 5 to August 31, 2024. We divide this period into five phases: P1 Pre-Escalation (July 5–15), P2 Crisis and Blackout (July 16–August 4), P3 Post-Blackout (August 5– 10), P4 Post-Revolution (August 11–20), and P5 Flood Crisis (August 21–31). These phases are used for temporal analysis and split construction. They are not used as sentiment labels. In total, we collected more than 2,000+ source posts and their comment threads. For each comment, we keep the corresponding parent post so that the dataset can support both comment-only and post+comment sentiment classification.

## 3.2 Preprocessing

We applied several preprocessing steps before annotation. First, we normalized the text using Unicode NFC normalization and removed irregular whitespace. URLs and user mentions were replaced with special tokens. Emojis were kept because they often carry sentiment in social media comments. We removed comments with fewer than three tokens, comments consisting mainly of non-Bangla text, and obvious spam entries. We also removed near-duplicate comments within the same thread using a Jaccard similarity threshold of 0.85. After preprocessing, the final dataset contains ≈200K comments.

## 3.3 Annotation Guidelines

We prepared annotation guidelines for three-way sentiment classification. Annotators were asked to assign one label to each comment.

Positive. A comment is labeled positive if it expresses support, approval, hope, praise, relief, or solidarity toward the event, actor, or situation discussed in the parent post.

Neutral. A comment is labeled neutral if it is factual, unclear, mixed, question-like, or does not express a clear positive or negative sentiment.

Negative. A comment is labeled negative if it expresses criticism, anger, frustration, sadness, fear, ridicule, sarcasm, blame, or disappointment. Annotators were instructed to use the parent post as context, but the label was assigned only to the comment. The guidelines included examples for dificult cases, including sarcasm, rhetorical questions, religious expressions, political references, code-switched profanity, and emoji-based sentiment. This follows standard practice in corpus annotation, where guidelines are refined through pilot annotation and disagreement analysis (Pustejovsky, 2012; Sabou et al., 2014).

![](images/6e69bd9859ef0d9b39c7f5f41cad534c6754a216da2d7626d433727e358c8e1a.jpg)  
Figure 2: Overview of the UNRESTSENT200K construction and evaluation framework. (1) Data acquisition and contextual grounding: public Bangla crisis discourse is collected from Facebook and YouTube, comments are linked to their parent posts and metadata, and utterance-only and context-aware inputs are constructed. (2) PiLA annotation and quality control: a five-stage, fully human workflow covers guideline development, annotator calibration, context-aware double annotation, senior adjudication, and an independent blind audit. (3) Experimental probing axes: the benchmark evaluates temporal robustness across event phases and the efects of contextual grounding and supervised fine-tuning across encoder and LLM families.

## 3.4 Manual Annotation

The annotation was carried out by 14 native Bangla-speaking undergraduate annotators from Bangladesh. All annotators were familiar with the July–August 2024 events and completed pilot training before the main annotation stage. A separate group of 10 senior validators supervised the process. Each comment was annotated independently by two annotators. If both annotators selected the same label, that label was accepted. If they disagreed, the comment was reviewed by a senior validator. Dificult cases were resolved through adju-রেdication based on the annotation guidelines. The annotation process had five stages: pilot guideline development, annotator training and qualification, মে য়ে দেdouble annotation, senior adjudication, and blind audit. The full process took approximately eight months.

## 3.5 Annotation Agreement

We measured annotation reliability using Cohen’s κ and Krippendorf’s α. The annotation process achieved a pairwise Cohen’s κ of 0.73 and Krippendorf’s α of 0.71, indicating substantial agreement. We also conducted a blind audit on a stratified subset of the dataset. The audit achieved 94.2% exactlabel agreement with the final gold labels. These results show that the annotation process was reliable despite the context-dependent nature of the comments.

## 3.6 Dataset Split

We split the dataset into training, validation, and test sets using a 70/15/15 ratio. The split was stratified by both sentiment label and event phase so that each partition preserves the overall class and temporal distribution. The training set contains 66,382 negative, 29,249 neutral, and 43,950 positive examples. The validation set contains 14,225 negative, 6,268 neutral, and 9,419 positive examples. The test set contains 14,223 negative, 6,267 neutral, and 9,416 positive examples. The full phasewise distribution is shown in Table 2. Overall, the dataset contains 94,830 negative (47.6%), 62,785 positive (31.5%), and 41,784 neutral (21.0%) comments.

Table 2: Stratified 70/15/15 train/validation/test split of UNRESTSENT200K by event-aligned phase and sentiment class. Splits are jointly stratified on phase and sentiment to preserve the class and temporal distribution within each partition.
<table><tr><td>Period</td><td></td><td colspan="3">Train (70%)</td><td colspan="3">Validation (15%)</td><td colspan="3">Test (15%)</td><td></td></tr><tr><td>Phase</td><td></td><td>Neg</td><td>Neu</td><td>Pos</td><td>Neg</td><td>Neu</td><td>Pos</td><td>Neg</td><td>Neu</td><td>Pos</td><td>Total</td></tr><tr><td>P1</td><td>Pre-Escalation of Revolution (Jul 5–15)</td><td>2,281</td><td>853</td><td>1,576</td><td>489</td><td>183</td><td>338</td><td>488</td><td>182</td><td>337</td><td>6,727</td></tr><tr><td>P2</td><td>Crisis &amp; Blackout (Jul 16–Aug 4)</td><td>9,180</td><td>4,700</td><td>9,698</td><td>1,967</td><td>1,007</td><td>2,078</td><td>1,968</td><td>1,008</td><td>2,078</td><td>33,684</td></tr><tr><td>P3</td><td>Post-Blackout (Aug 5–10)</td><td>14,153</td><td>7,482</td><td>10,704</td><td>3,033</td><td>1,603</td><td>2,294</td><td>3,032</td><td>1,603</td><td>2,293</td><td>46,197</td></tr><tr><td>P4</td><td>Post-Revolution (Aug 11–20)</td><td>23,664</td><td>10,532</td><td>14,479</td><td>5,071</td><td>2,257</td><td>3,103</td><td>5,070</td><td>2,257</td><td>3,103</td><td>69,536</td></tr><tr><td>P5</td><td>Flood Crisis (Aug 21–31)</td><td>17,104</td><td>5,682</td><td>7,493</td><td>3,665</td><td>1,218</td><td>1,606</td><td>3,665</td><td>1,217</td><td>1,605</td><td>43,255</td></tr><tr><td colspan="2">Overall</td><td>66,382</td><td>29,249</td><td>43,950</td><td>14,225</td><td>6,268</td><td>9,419</td><td>14,223</td><td>6,267</td><td>9,416</td><td>199,399</td></tr></table>

## 4 Experiments and Analysis

The July–August 2024 Bangladesh uprising was not a static event (Rana et al., 2026). Public discussion changed as the crisis moved from early protest activity to internet blackout, regime collapse, political transition, and later the flood crisis. This makes UNRESTSENT200K useful for studying more than standard sentiment classification. It allows us to ask whether sentiment models can handle three problems that often appear in real crisis discourse: context dependence, temporal change, and abrupt shifts in public reaction. We organize the experiments around these problems. First, we evaluate recent LLMs to see how far prompting alone can go on Bangla crisis sentiment. Second, we test whether supervised adaptation with midscale LLMs improves over prompting. Third, we measure the efect of adding the parent post as context. Finally, we evaluate temporal robustness and examine whether sentiment changes gradually or through phase-level shifts. We evaluate four finetuned encoder models: BanglaBERT (Bhattacharjee et al., 2022), BanglaElectra, mBERT (Devlin et al., 2019), and XLM RoBERTa (Conneau et al., 2020). We also evaluate 15 LLMs under zero-shot and few-shot prompting. Across experiments, we report accuracy, macro-F1, and chance-corrected metrics such as Cohen’s κ and MCC where applicable.

## 4.1 Prompted LLMs

Motivation. We first ask whether recent LLMs can classify sentiment in UNRESTSENT200K without task-specific training. This is an important baseline because LLMs are often used directly on new events, especially when labeled data is limited.

Models and prompting setup. We evaluate open-source models from the Gemma, Qwen, and LLaMA families, along with proprietary models including GPT-4o-mini, GPT-4.1-mini, and GPT-4.1-nano. Each model is tested with zero-shot and few-shot prompts written in Bangla. In the fewshot setting, the prompt includes seven examples covering positive, negative, neutral, sarcastic, and politically implicit comments. The full prompt templates are shown in Appendix B.1.

Input format. Each test input is given in the same format: [post] [SEP] [comment]. This is the same input format used in the post+comment encoder setting, making the LLM and encoder results comparable.

Main results. Table 3 shows that the strongest prompted model is GPT-4o-mini in the few-shot setting. It reaches 76.0% accuracy, 75.0 macro-F1, and κ = 0.55. This is slightly above the strongest post+comment encoder, BanglaBERT, which reaches 74.72% accuracy and 0.7167 macro-F1. However, the improvement is small. This suggests that strong LLMs can benefit from the provided context, but prompting alone does not fully solve the task.

Efect of few-shot examples. Few-shot examples help the stronger models more than the smaller ones. For GPT-4o-mini, κ improves from 0.47 to 0.55. For GPT-4.1-mini, it improves from 0.34 to 0.40. In contrast, several smaller open-source models show little improvement from few-shot examples. This suggests that examples are useful only when the model already has enough Bangla language ability and event-level reasoning capacity to use them.

Error patterns. A manual inspection of errors shows three recurring patterns. First, models often miss sarcasm, especially when a comment appears positive on the surface but is negative in context. Second, neutral comments are often pushed toward negative labels. Third, politically charged comments become harder when their interpretation depends on the phase of the uprising. These errors are also seen in the encoder models, suggesting that they are properties of the task rather than failures of one model family.

Table 3: Zero-shot and few-shot performance (%) of LLMs with expert prompting (Bangla-native) on UnRest-Sent200K. Best result per section is bolded and underlined.
<table><tr><td rowspan="2">Model</td><td colspan="4">Zero-Shot</td><td colspan="4">Few-Shot</td></tr><tr><td>Acc</td><td>F1</td><td>MCC</td><td>κ</td><td>Acc</td><td>F1</td><td>MCC</td><td>K</td></tr><tr><td colspan="9">Proprietary Models</td></tr><tr><td>GPT-4o-mini</td><td>71.0</td><td>69.0</td><td>50.0</td><td>47.0</td><td>76.0</td><td>75.0</td><td>58.0</td><td>55.0</td></tr><tr><td>gpt-4.1-mini</td><td>73.0</td><td>71.0</td><td>36.0</td><td>34.0</td><td>73.0</td><td>71.0</td><td>45.0</td><td>40.0</td></tr><tr><td>gpt-4.1-nano</td><td>58.0</td><td>56.0</td><td>22.0</td><td>13.0</td><td>69.0</td><td>61.0</td><td>31.0</td><td>29.0</td></tr><tr><td colspan="9">Open-Source Models (≤3B)</td></tr><tr><td>gemma4-2b gemma3-1b</td><td>61.0</td><td>59.0</td><td>40.0</td><td>38.0 15.0</td><td>64.0 52.0</td><td>63.0 47.0</td><td>45.0 26.0</td><td>43.0 22.0</td></tr><tr><td>qwen2.5-3b</td><td>43.0 49.0</td><td>37.0 44.0</td><td>18.0 25.0</td><td>20.0</td><td>55.0</td><td>52.0</td><td>31.0</td><td>28.0</td></tr><tr><td>qwen2.5-1.5b</td><td>46.0</td><td>36.0</td><td>20.0</td><td>11.0</td><td>49.0</td><td>40.0</td><td>23.0</td><td>15.0</td></tr><tr><td>llama-3.2-3b</td><td>43.0</td><td>29.0</td><td>15.0</td><td>5.0</td><td>45.0</td><td>33.0</td><td>17.0</td><td>8.0</td></tr><tr><td>qwen2.5-0.5b</td><td>38.0</td><td>22.0</td><td>0.2</td><td>0.0</td><td>39.0</td><td>23.0</td><td>2.0</td><td>0.4</td></tr><tr><td>Open-Source Models (&gt;3B)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="9">gemma4-4b</td></tr><tr><td>gemma3-4b</td><td>52.0</td><td>49.0</td><td>16.0</td><td>15.0</td><td>66.2</td><td>65.0</td><td>48.7</td><td>46.3</td></tr><tr><td></td><td>49.0</td><td>40.0</td><td>27.0</td><td>16.0</td><td>59.0</td><td>54.0</td><td>39.0</td><td>33.0</td></tr><tr><td>qwen3-4b</td><td>56.0</td><td>52.0</td><td>35.0</td><td>30.0</td><td>58.0</td><td>56.0</td><td>38.0</td><td>34.0</td></tr><tr><td>qwen3-4b-instruct</td><td>56.0</td><td>52.0</td><td>35.0</td><td>30.0</td><td>59.0</td><td>57.0</td><td>23.0</td><td>14.0</td></tr><tr><td>qwen2.5-7b</td><td>47.0</td><td>45.0</td><td>23.0</td><td>21.0</td><td>61.0</td><td>60.0</td><td>42.0</td><td>39.0</td></tr><tr><td>1lama-3.1-8b</td><td>53.0</td><td>52.0</td><td>37.0</td><td>11.0</td><td>58.0</td><td>55.0</td><td>37.0</td><td>33.0</td></tr><tr><td>LoRA SFT</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="9"></td></tr><tr><td>Fine-Tuned Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Gemma-4B</td><td>78.4</td><td>77.6</td><td>65.3</td><td>63.1</td><td>一</td><td>一</td><td>一</td><td>一</td></tr><tr><td>Llama-3.1-8B</td><td>76.8</td><td>75.9</td><td>62.4</td><td>60.2</td><td></td><td></td><td>一</td><td></td></tr></table>

## 4.2 Supervised Adaptation with Mid-Scale LLMs

Prompting provides a useful starting point, but UNRESTSENT200K also allows us to test whether in-domain supervised training improves performance. We therefore fine-tune two mid-scale LLMs, Gemma-4B and LLaMA-3.1-8B, using LoRA. Both models are trained with r=64, α=32, and learning rate $3 \times 1 0 ^ { - 4 }$ . The fine-tuned models outperform all prompted models in Table 3.

Gemma-4B achieves 78.4% accuracy, 77.6 macro-F1, MCC 65.3, and κ 63.1. LLaMA-3.1- 8B achieves 76.8% accuracy and 75.9 macro-F1. Gemma-4B also improves over GPT-4o-mini by 2.4 percentage points in accuracy and over grounded BanglaBERT by 3.7 points. These results show that supervised adaptation is still important for low-resource crisis sentiment analysis. The smaller Gemma-4B model also performs better than the larger LLaMA-3.1-8B model, suggesting that the choice of backbone and fine-tuning recipe matters more than parameter count alone. At the same time, the best model remains below 80% accuracy, leaving room for future work on crisis-aware and context-aware sentiment models.

## 4.3 Qualitative Efect of Fine-Tuning

Table 6 illustrates the three failure patterns flagged in §4.1 sarcasm, neutral/positive collapse to negative, and politically charged surface cues and shows the same (post, comment) pairs being recovered after LoRA-SFT on Gemma-4B. The pattern is qualitative: SFT exposure to the gold PiLA labels lets the model pick up domain-specific pragmatic cues that the zero-shot model misses despite seeing identical input.

Table 4: Context ablation on UNRESTSENT200K. Fine-tuned encoder performance when the model receives only the comment versus the parent post concatenated with the comment via [SEP]. Corpus, splits, optimiser, schedule, and hyperparameters are held fixed; the only variable is the input. Adding the parent post yields a consistent +7– 11pp accuracy gain across every architecture. Best per column bolded and underlined.
<table><tr><td colspan="5">Comment-only input</td><td colspan="4">Post + Comment input [post] [SEP] [comment]</td></tr><tr><td>Model</td><td>Acc. (%)</td><td>[comment] Prec.</td><td>Rec.</td><td>F1</td><td>Acc. (%)</td><td>Prec.</td><td>Rec.</td><td>F1</td></tr><tr><td>BanglaBERT</td><td>64.19</td><td>0.6195</td><td>0.6158</td><td>0.6170</td><td>74.72</td><td>0.7245</td><td>0.7116</td><td>0.7167</td></tr><tr><td>mBERT</td><td>61.06</td><td>0.5869</td><td>0.5881</td><td>0.5874</td><td>69.90</td><td>0.6705</td><td>0.6790</td><td>0.6731</td></tr><tr><td>BanglaElectra</td><td>62.02</td><td>0.6193</td><td>0.6202</td><td>0.6159</td><td>68.98</td><td>0.6637</td><td>0.6705</td><td>0.6649</td></tr><tr><td>XLM-RoBERTa-Base</td><td>63.41</td><td>0.6287</td><td>0.6312</td><td>0.6299</td><td>72.96</td><td>0.7289</td><td>0.7296</td><td>0.7012</td></tr></table>

Table 5: Temporal generalization performance of supervised fine-tuned (SFT) transformer models under distribution shift on UnRestSent200K. Models are trained on comments collected between July 5 and August 5, 2024, and evaluated on two chronologically non-overlapping future test partitions: Test-A (August 6–20) and Test-B (August 21–31). Results highlight the degradation of model performance under evolving crisis discourse and temporal non-stationarity.
<table><tr><td>Model</td><td></td><td>Test Acc. (%)</td><td>Prec.</td><td>Rec.</td><td>F1 Score</td></tr><tr><td rowspan="2">BanglaBERT</td><td>A</td><td>53.70</td><td>0.5204</td><td>0.5236</td><td>0.5217</td></tr><tr><td>B</td><td>54.83</td><td>0.5027</td><td>0.5222</td><td>0.5064</td></tr><tr><td rowspan="2">BanglaElectra</td><td>A</td><td>40.11</td><td>0.4195</td><td>0.4275</td><td>0.4006</td></tr><tr><td>B</td><td>41.18</td><td>0.4164</td><td>0.4343</td><td>0.3942</td></tr><tr><td rowspan="2">mBERT</td><td>A</td><td>40.96</td><td>0.4234</td><td>0.4380</td><td>0.4111</td></tr><tr><td>B</td><td>41.24</td><td>0.4191</td><td>0.4354</td><td>0.3948</td></tr><tr><td rowspan="2">XLM-RoBERTa</td><td>A</td><td>47.35</td><td>0.4708</td><td>0.4784</td><td>0.4695</td></tr><tr><td>B</td><td>47.28</td><td>0.4480</td><td>0.4664</td><td>0.4436</td></tr></table>

## 4.4 Efect of Parent-Post Context

Many comments in UNRESTSENT200K are dificult to interpret without their parent post, as shown in Table 6. This is common in social media discussions during political events. A short comment may look factual or neutral by itself, but become clearly positive or negative once the surrounding news post is known. For example, the comment “We know who is doing this.” is hard to label in isolation. When paired with a post about a family being forced from their home, the same comment becomes a negative reaction. This shows why the parent post is not extra information; in many cases, it is part of the meaning of the comment. We test this directly using a paired ablation. In the comment-only setting, the encoder receives only the comment. In the post+comment setting, the encoder receives the parent post and comment separated by [SEP]. All other training settings are kept fixed. The results in Table 4 show consistent gains from context. BanglaBERT improves from 64.19% to 74.72% accuracy. XLM-RoBERTa improves by 9.55 points, mBERT by 8.84 points, and BanglaElectra by 6.96 points. The gain appears across all encoder families. This result has a direct implication for future dataset design. In crisis discourse, comment-only sentiment classification can remove information needed for the correct label. For this reason, UNRESTSENT200K supports both comment-only and post+comment evaluation, allowing future work to measure how well models use discourse context.

## 4.5 Temporal Robustness

We next evaluate whether models trained on earlier phases of the uprising generalize to later phases. This setting reflects a realistic use case: during an unfolding crisis, models may be trained on available data and then applied as the event continues to change. We train the encoder models on comments from July 5 to August 5, 2024. We then test them on two later windows: Test-A, covering August 6–20, and Test-B, covering August 21–31. These windows do not overlap with the training period. The results in Table 5 show a large performance drop. BanglaBERT remains the best model, reaching 54.83% accuracy on Test-B, but this is far below its in-distribution post+comment accuracy of 74.72%. Other encoders also degrade under the time-based split. Overall, the models lose between 19 and 28 macro-F1 points compared with their in-distribution results. This drop is important because the language, platforms, and broad event remain the same. What changes is the stage of the crisis. After the blackout and regime collapse, public discussion shifts toward new actors, new concerns, and new emotional frames. Appendix A.8 further shows measurable lexical change across phases using Jensen–Shannon divergence. These results show why random splits are not enough for crisis sentiment analysis. A model that performs well on a random test set may still fail when the event enters a new phase. UNRESTSENT200K therefore provides a useful setting for studying temporal robustness in low-resource NLP.

Table 6: Qualitative comparison of zero-shot and supervised predictions. Both settings use Gemma-4B; LoRA SFT denotes the same backbone fine-tuned on UNRESTSENT200K $( r { = } 6 4 , \alpha { = } 3 2$ , learning rate $3 \times 1 0 ^ { - 4 } )$ . ✓ and ✗ indicate agreement and disagreement with the PiLA gold label, respectively. Examples represent the three failure modes discussed in §4.1.
<table><tr><td>#</td><td>Post-Comment Pair</td><td>Gold</td><td>Zero-shot</td><td>LoRA SFT</td></tr><tr><td>1</td><td>Post可利800 ：2 &quot;The peon who worked at my house now owns 4 billion taka: Prime Minister.&quot;</td><td>Negative</td><td>Positive X</td><td>Negative√</td></tr><tr><td colspan="5">Comment可!“Ah, what a matter of pride!&quot; Sarcasm. The ostensibly positive expression “&quot;pride&quot; becomes negative under the corruption-related headline. Zero-shot</td></tr><tr><td colspan="2">follows the literal wording, whereas LoRA SFT captures the intended sarcasm. 2 POt , -A 2</td><td>Negative Neutral X Negative√</td><td></td><td></td></tr><tr><td>disrupted.&quot;</td><td>&quot;Internet service is slow nationwide; access to Facebook and Messenger is also Comment“This is Hasina&#x27;s doing.”</td><td></td><td></td><td></td></tr><tr><td colspan="5">Pragmatic negativity. Zero-shot treats the comment as a neutral factual attribution, whereas LoRA SFT recognizes its accusatory and negative pragmatic force.</td></tr><tr><td colspan="2">3 Post可 何T ， 8 &quot;13 dead in floods so far; 4.5 million affected.&quot;</td><td></td><td>Positive Negative X Positive √</td><td></td></tr><tr><td>everyone.&quot;</td><td>Comment可可“May Allah protect</td><td></td><td></td><td></td></tr><tr><td colspan="5">Surface-cue confusion. Disaster-related terms in the post mislead the zero-shot model, while LoRA SFT correctly identifies the prayer and expression of solidarity as positive.</td></tr></table>

## 4.6 Sentiment Change Across Phases

The temporal experiment shows that performance drops over time. We then ask whether this change is gradual or concentrated around specific moments in the uprising. To measure this, we compute Cohen’s d for all pairwise comparisons between the five event phases. The full results are shown in Table 7. Seven of the ten comparisons show negligible efect sizes. The strongest changes are centered around the Post-Blackout phase. The comparison between Crisis and Blackout and Post-Blackout has a medium efect size $( d = - 0 . 5 5 6 )$ , and the comparison between Post-Blackout and Flood Crisis also has a medium efect size $( d = 0 . 5 6 7 )$ . This suggests that sentiment did not change smoothly across the full period. Instead, the largest shift occurred around the moment when internet access returned, the regime collapsed, and people began reacting to a new political situation. Sentiment then shifted again as the flood crisis became the dominant concern. For future research, this makes UNRESTSENT200K useful for studying change-point-aware sentiment models. Models for crisis monitoring may need to detect when the discourse has entered a new phase, rather than assuming that sentiment changes slowly over time.

## 4.7 Findings

Our experiments yield five main findings.

Prompting alone does not resolve crisis sentiment. Few-shot GPT-4o-mini achieves the strongest prompted result at 76.0% accuracy, only slightly exceeding grounded BanglaBERT at 74.72%. Its remaining errors in sarcasm, implicit attribution, and context-dependent language show that strong prompting cannot fully address the task. Supervised adaptation produces the best performance. LoRA-tuned Gemma-4B achieves the highest overall accuracy of 78.4%, outperforming both prompted LLMs and fine-tuned encoders. As illustrated in Table 6, in-domain supervision helps recover pragmatic meanings that the zeroshot model misses, including sarcastic criticism, political accusation, and expressions of solidarity. Parent-post context is essential. Including the parent post improves all encoders by approximately 7–11 accuracy points. This consistent gain shows that the post is often part of the sentiment signal rather than optional background information.

Table 7: Pairwise sentiment-shift comparisons between the five event phases of the 2024 Bangladesh uprising using Cohen’s d efect size. Larger absolute values indicate stronger changes in sentiment distribution between phases, with the strongest shifts observed around the immediate post-blackout period.
<table><tr><td>Period 1</td><td>Period 2</td><td>d</td><td>Effect</td></tr><tr><td>Pre-Escalation</td><td>Crisis/Blackout</td><td>0.138</td><td>Negligible</td></tr><tr><td>Pre-Escalation</td><td>Post-Blackout</td><td>-0.411</td><td>Small</td></tr><tr><td>Pre-Escalation</td><td>Post-Revolution</td><td>-0.034</td><td>Negligible</td></tr><tr><td>Pre-Escalation</td><td>Flood Crisis</td><td>0.144</td><td>Negligible</td></tr><tr><td>Crisis/Blackout</td><td>Post-Blackout</td><td>-0.556</td><td>Medium</td></tr><tr><td>Crisis/Blackout</td><td>Post-Revolution</td><td>-0.174</td><td>Negligible</td></tr><tr><td>Crisis/Blackout</td><td>Flood Crisis</td><td>0.003</td><td>Negligible</td></tr><tr><td>Post-Blackout</td><td>Post-Revolution</td><td>0.381</td><td>Small</td></tr><tr><td>Post-Blackout</td><td>Flood Crisis</td><td>0.567</td><td>Medium</td></tr><tr><td>Post-Revolution</td><td>Flood Crisis</td><td>0.178</td><td>Negligible</td></tr></table>

Temporal shift substantially reduces performance. Models trained on earlier phases lose approximately 19–28 macro-F1 points when evaluated on later phases. Thus, random splits can overestimate real-world reliability by mixing examples from diferent stages of an evolving crisis.

Sentiment changes around major events. Most phase comparisons show negligible diferences, but medium shifts occur around the Post-Blackout period (d = −0.556) and the later Flood Crisis (d = 0.567). This suggests that crisis sentiment changes through event-linked transitions rather than a smooth temporal trend.

Together, these findings show that reliable crisis sentiment analysis requires both discourse context and temporal adaptation. UNRESTSENT200K supports the study of these challenges through parent-linked comments and event-aligned evaluation phases.

## 5 Discussion

Our results show that crisis sentiment analysis depends strongly on both context and time. Models perform much better when the parent post is included, with consistent gains of 7–11 percentage points across encoder models. At the same time, performance drops substantially when models are trained on earlier phases of the uprising and tested on later phases. This shows that standard random splits can overestimate model performance in an unfolding crisis. The results also show that strong LLMs do not remove these challenges. GPT-4o-mini performs well under fewshot prompting, but it only slightly improves over the best post+comment encoder and remains below the best fine-tuned LLM. Its errors also overlap with those of smaller models, especially on sarcasm, implicit political references, and comments whose meaning depends on the event phase. These findings suggest that future sentiment benchmarks, especially for crisis settings, should include temporal splits and discourse context. For low-resource languages such as Bangla, this is particularly important because models may appear reliable under standard evaluation while failing when the public conversation changes. UNRESTSENT200K provides a setting for studying these issues in a controlled way.

## 6 Conclusion

We introduce UNRESTSENT200K, a Bangla crisis sentiment dataset containing ≈200K comments from Facebook and YouTube discussions around the July–August 2024 Bangladesh uprising. Each comment is linked to its parent post and assigned one of three sentiment labels: POSITIVE, NEUTRAL, or NEGATIVE. The dataset is annotated through a fully human process under the PiLA framework. Experiments with fine-tuned encoders, prompted LLMs, and LoRA-tuned LLMs show three main results. First, parent-post context consistently improves sentiment classification. Second, temporal shift across phases of the uprising leads to large performance drops. Third, supervised adaptation gives the strongest results, but the benchmark remains challenging. Overall, UNRESTSENT200K supports research on context-aware and temporally robust sentiment analysis in low-resource crisis discourse.

## Limitations

UNRESTSENT200K has several limitations. First, it covers only Facebook and YouTube. These platforms were important sources of public discussion during the uprising, but they do not represent all Bangla speakers or all forms of political communication. Other sources, such as X/Twitter, Telegram, private messaging, and ofline discourse, are not included. Second, the dataset uses three sentiment labels. This makes large-scale annotation reliable, but it does not capture finer distinctions such as anger, fear, hope, grief, or stance. Future work can extend the label schema to include emotion and stance labels. Third, the dataset is based on one event in Bangladesh. The results should therefore not be treated as universal claims about all crises or all low-resource languages. Cross-event and cross-language studies are needed to test how well the findings generalize. Finally, the models evaluated in this paper were not specifically designed for temporal adaptation or change-point detection. Developing models that can use context and adapt to new crisis phases is an important direction for future work.

## Reproducibility Statement

To support reproducibility, the release package will include the processed and de-identified UNRESTSENT200K corpus; fixed training, validation, and test splits; parent-post context; timestamps; platform and event-phase metadata; annotation guidelines; prompt templates; preprocessing and evaluation scripts; and the configurations used for the encoder and LoRA experiments. Dataset statistics and phase boundaries are reported in Section 3, while detailed preprocessing, annotation, temporal-analysis, topic-modeling, and prompting procedures are provided in the remainder of this appendix.

AI Usage Disclosure Large language models were used as experimental systems to generate benchmark predictions for UNRESTSENT200K under zero-shot and few-shot prompting and supervised fine-tuning. AI-assisted tools were also used for limited non-experimental support, including language editing, figure ideation, and visualization assistance. All AI-assisted material was manually reviewed and verified by the authors, who retain full responsibility for the paper’s content, results, and interpretations.

## References

Sadia Alam, Md Farhan Ishmam, Navid Hasin Alvee, Md Shahnewaz Siddique, Md Azam Hossain, and Abu Raihan Mostofa Kamal. 2025. BnSentMix: A diverse Bengali-English code-mixed dataset for sentiment analysis. In Proceedings of the First Workshop on Language Models for Low-Resource Languages. Association for Computational Linguistics.

Abhik Bhattacharjee, Tahmid Hasan, Wasi Ahmad, Kazi Samin Mubasshir, Md Saiful Islam, Anindya Iqbal, M. Sohel Rahman, and Rifat Shahriyar. 2022. BanglaBERT: Language model pretraining and benchmarks for low-resource language understanding evaluation in Bangla. In Findings of the Association for Computational Linguistics: NAACL 2022, Seattle, United States. Association for Computational Linguistics.

Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. 2020. Unsupervised cross-lingual representation learning at scale. In Proceedings ofthe 58th Annual Meeting ofthe Association for Computational Linguistics. Association for Computational Linguistics.

Mithun Das, Somnath Banerjee, Punyajoy Saha, and Animesh Mukherjee. 2022. Hate speech and ofensive language detection in Bengali. In Proceedings of the 2nd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 12th International Joint Conference on Natural Language Processing (Volume 1: Long Papers). Association for Computational Linguistics.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings ofthe 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), Minneapolis, Minnesota. Association for Computational Linguistics.

Azizul Hakim Fayaz, MD Uddin, Rayhan Uddin Bhuiyan, Zakia Sultana, Md Samiul Islam, Bidyarthi Paul, Tashreef Muhammad, and Shahriar Manzoor. 2025. Bidwesh: A bangla regional based hate speech detection dataset. arXiv preprint arXiv:2507.16183.

Alec Go, Richa Bhayani, and Lei Huang. 2009. Twitter sentiment classification using distant supervision. CS224Nproject report, Stanford, 1(12):2009.

Maarten Grootendorst. 2022. Bertopic: Neural topic modeling with a class-based tf-idf procedure. arXiv preprint arXiv:2203.05794.

Fabiha Haider, Fariha Tanjim Shifat, Md Farhan Ishmam, Md Sakib Ul Rahman Sourove, Deeparghya Dutta Barua, Md Fahim, and Md Farhad Alam Bhuiyan. 2025. BanTH: A

multi-label hate speech detection dataset for transliterated Bangla. In Findings of the Association for Computational Linguistics: NAACL 2025, Albuquerque, New Mexico. Association for Computational Linguistics.

Mahmudul Hasan, Md Rashedul Ghani, and KM Azharul Hasan. 2024. Aspect based sentiment analysis datasets for bangla text. Data in Brief, 57:111107.

Md Arid Hasan, Firoj Alam, Md Fahad Hossain, Usman Naseem, and Syed Ishtiaque Ahmed. 2026. LLMbased multi-task Bangla hate speech detection: Type, severity, and target. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), San Diego, California, United States. Association for Computational Linguistics.

Md. Sabbir Hossen, Md. Saiduzzaman, and Pabon Shaha. 2025. Social media sentiments analysis on the july revolution in bangladesh: A hybrid transformer based machine learning approach. In 2025 17th International Conference on Electronics, Computers and Artificial Intelligence (ECAI), pages 1–8.

Minqing Hu and Bing Liu. 2004. Mining and summarizing customer reviews. In Proceedings of the Tenth ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD ’04, page 168–177, New York, NY, USA. Association for Computing Machinery.

Ariful Islam and Md Rifat Hossen. 2025. Banglasentnet: A hybrid deep learning framework for multiaspect sentiment analysis in bangla e-commerce reviews. In Data Science, AI and Applications. Springer Nature Switzerland.

Khondoker Ittehadul Islam, Sudipta Kar, Md Saiful Islam, and Mohammad Ruhul Amin. 2021. SentNoB: A dataset for analysing sentiment on noisy Bangla texts. In Findings of the Association for Computational Linguistics: EMNLP 2021. Association for Computational Linguistics.

Md. Ekramul Islam, Labib Chowdhury, Faisal Ahamed Khan, Shazzad Hossain, Md Sourave Hossain, Mohammad Mamun Or Rashid, Nabeel Mohammed, and Mohammad Ruhul Amin. 2023. Sentigold: A large bangla gold standard multi-domain sentiment analysis dataset and its evaluation. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, KDD ’23, page 4207– 4218. Association for Computing Machinery.

Swastika Kundu, Autoshi Ibrahim, Mithila Rahman, and Tanvir Ahmed. 2026. Anubhuti: a comprehensive corpus for sentiment analysis in bangla regional languages. In 2026 IEEE 2nd International Conference on Quantum Photonics, Artificial Intelligence & Networking (QPAIN), pages 1–6. IEEE.

Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts.

2011. Learning word vectors for sentiment analysis. In Proceedings ofthe 49th Annual Meeting ofthe Association for Computational Linguistics: Human Language Technologies, Portland, Oregon, USA. Association for Computational Linguistics.

Julian McAuley, Christopher Targett, Qinfeng Shi, and Anton van den Hengel. 2015. Image-based recommendations on styles and substitutes. In Proceedings of the 38th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’15. Association for Computing Machinery.

Md. Saddam Hossain Mukta, Md. Adnanul Islam, Faisal Ahamed Khan, Afjal Hossain, Shuvanon Razik, Shazzad Hossain, and Jalal Mahmud. 2021. A comprehensive guideline for bengali sentiment annotation. ACM Trans. Asian Low-Resour. Lang. Inf. Process., 21(2).

Md. Darun Nayeem, Zarin Rafa, Tasnuva Tasnim Nova, Yasin Rahman, Abdul Mumeet Pathan, and Md. Masudul Islam. 2026. Banglamuse: A multimodal bangla sentiment dataset of text–audio pairs for speech and sentiment analysis. Data in Brief, 65:112458.

Helene Olsen, Étienne Simon, Erik Velldal, and Lilja Øvrelid. 2024. Socio-political events of conflict and unrest: A survey of available datasets. In Proceedings of the 7th Workshop on Challenges and Applications of Automated Extraction of Socio-political Eventsfrom Text (CASE 2024), St. Julians, Malta. Association for Computational Linguistics.

James Pustejovsky. 2012. The role of linguistic models and language annotation in feature selection for machine learning. In Proceedings of the Sixth Linguistic Annotation Workshop, page 1, Jeju, Republic of Korea. Association for Computational Linguistics.

Md Nishat Raihan, Umma Tanmoy, Anika Binte Islam, Kai North, Tharindu Ranasinghe, Antonios Anastasopoulos, and Marcos Zampieri. 2023. Ofensive language identification in transliterated and codemixed Bangla. In Proceedings of the First Workshop on Bangla Language Processing (BLP-2023), Singapore. Association for Computational Linguistics.

Md. Sohel Rana, Shadia Sharmin, Mohammad Bin Amin, and Judit Oláh. 2026. From revolt to revolution: exploring motivating factors of bangladesh’s july 2024 uprising. Research in Globalization, 12:100341.

Mohammad Rifat Ahmmad Rashid, Kazi Ferdous Hasan, Rakibul Hasan, Aritra Das, Mithila Sultana, and Mahamudul Hasan. 2024. A comprehensive dataset for sentiment and emotion classification from bangladesh e-commerce reviews. Data in Brief, 53:110052.

Nauros Romim, Mosahed Ahmed, Md Saiful Islam, Arnab Sen Sharma, Hriteshwar Talukder, and Mohammad Ruhul Amin. 2022. BD-SHS: A benchmark dataset for learning to detect online Bangla hate

speech in diferent social contexts. In Proceedings of the Thirteenth Language Resources and Evaluation Conference, Marseille, France. European Language Resources Association.

Marta Sabou, Kalina Bontcheva, Leon Derczynski, and Arno Scharl. 2014. Corpus annotation through crowdsourcing: Towards best practice guidelines. In Proceedings of the Ninth International Conference on Language Resources and Evaluation (LREC’14), Reykjavik, Iceland. European Language Resources Association (ELRA).

Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D. Manning, Andrew Ng, and Christopher Potts. 2013. Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing, Seattle, Washington, USA. Association for Computational Linguistics.

ZACHARY C. STEINERT-THRELKELD. 2017. Spontaneous collective action: Peripheral mobilization during the arab spring. American Political Science Review, 111(2):379–403.

Md. Ashraful Islam Talukder. 2025. Bangla and banglish e-commerce reviews dataset for aspect-based sentiment analysis. Mendeley Data.

Zeynep Tufekci. 2017. Twitter and tear gas: The power and fragility of networked protest. Yale University Press.

## A Appendix

## A.1 Reproducibility Statement

To support reproducibility, the release package will include the processed and de-identified UN-RESTSENT200K corpus; fixed training, validation, and test splits; parent-post context; timestamps; platform and event-phase metadata; annotation guidelines; prompt templates; preprocessing and evaluation scripts; and the configurations used for the encoder and LoRA experiments. Dataset statistics and phase boundaries are reported in Section 3, while detailed preprocessing, annotation, temporal-analysis, topic-modeling, and prompting procedures are provided in the remainder of this appendix.

## A.2 AI Usage Disclosure

Large language models were used as experimental systems to generate benchmark predictions for UNRESTSENT200K under zero-shot and fewshot prompting and supervised fine-tuning. AIassisted tools were also used for limited nonexperimental support, including language editing, figure ideation, and visualization assistance. All AI-assisted material was manually reviewed and verified by the authors, who retain full responsibility for the paper’s content, results, and interpretations.

## A.3 Preprocessing Details

This section describes the preprocessing steps used before annotation, as summarized in Section 3.2.

## A.3.1 Normalization

• Unicode normalization: Text is normalized to NFC form to reduce inconsistent Bangla character encodings.

• Whitespace cleanup: Extra spaces, tabs, and irregular line breaks are removed.

• Entity masking: URLs and user mentions are replaced with [URL] and [USER].

• Emoji retention: Emojis are kept because they often express sentiment in social media comments.

## A.3.2 Filtering

We remove comments with fewer than three tokens after normalization. We also remove comments that consist mainly of non-Bangla text or obvious spam. To keep the released corpus primarily Bangla, comments with substantial English or Bangla–English transliteration are excluded. Figure 3 reports the resulting sentiment distribution across word-length bins of the retained corpus: short comments (≤10 words) skew positive, while neutral expression increases with comment length and negative sentiment stays relatively flat across lengths.

## A.3.3 Deduplication

Near-duplicate comments within the same thread are removed using a Jaccard similarity threshold of 0.85. Duplicates across diferent posts are retained when they appear as part of separate public discussions.

## A.3.4 Post-Level Framing Labels

In addition to comment-level sentiment labels, each source post is assigned a coarse framing label:

$$
\mathcal { Y } ^ { ( p ) } = \{ \mathrm { H o p E , D E S P A I R , O U T R A G E } \}\tag{1}
$$

Table 8: Table showing topic modeling statistics for each period. $C _ { v }$ denotes the topic coherence score, computed using normalized pointwise mutual information (NPMI) and ranging from 0 to 1, where higher values indicate greater semantic coherence. $E \mathcal { f } .$ represents the efective number of topics, calculated as exp(H), where $H = - \textstyle \sum _ { i } p _ { i }$ log $p _ { i }$ is the Shannon entropy of the topic distribution. This quantity reflects the number of equally prominent topics that would yield the same level of topic diversity.
<table><tr><td>Period</td><td>Docs</td><td>Topics</td><td> $\mathbf { C } _ { v }$ </td><td>Eff.</td></tr><tr><td>P1: Pre-Escalation</td><td>6,727</td><td>12</td><td>0.405</td><td>1.7</td></tr><tr><td>P2: Crisis &amp; Blackout</td><td>33,684</td><td>29</td><td>0.781</td><td>1.8</td></tr><tr><td>P3: Post-Blackout</td><td>46,197</td><td>18</td><td>0.384</td><td>1.9</td></tr><tr><td>P4: Post-Revolution</td><td>69,536</td><td>25</td><td>0.420</td><td>2.0</td></tr><tr><td>P5: Flood Crisis</td><td>43,255</td><td>25</td><td>0.422</td><td>1.9</td></tr><tr><td>Global</td><td>199,399</td><td>29</td><td>0.375</td><td>一</td></tr></table>

These labels describe the dominant framing of the parent post. HOPE captures optimistic or changeoriented framing, DESPAIR captures grief or uncertainty, and OUTRAGE captures anger, blame, or calls for accountability. These post-level labels are used as annotation context and metadata; the main prediction task remains comment-level sentiment classification.

## A.4 PiLA: Annotation Protocol Details

This appendix provides additional details about the Pilot Label Annotation (PiLA) framework shown in Figure 2. PiLA is a fully human annotation process. No labels are assigned by automated systems. The full annotation efort took approximately eight months and involved 14 primary annotators and 10 senior validators.

## A.4.1 Annotator and Validator Pool

The primary annotation team consists of 14 undergraduate annotators from Bangladesh. All annotators are native Bangla speakers and were familiar with the July–August 2024 events through Bangla news and social media. The validator team consists of 10 senior annotators who did not overlap with the primary annotation pool. All participants provided informed consent, and the study protocol was reviewed by the host institution.

## A.4.2 Guideline Document

The annotation guideline defines the three sentiment labels: POSITIVE, NEUTRAL, and NEGA-TIVE. It also provides rules and examples for common ambiguous cases, including sarcasm, rhetorical questions, religious expressions, political references, code-switched profanity, and emoji-based sentiment. Annotators were instructed to use the parent post as context, but not to copy the post’s sentiment into the comment label. The label was assigned to the comment only. The guideline was refined during the pilot stage and then fixed before the main annotation stage.

## A.4.3 Qualification and Quality Checks

Before the main annotation stage, annotators completed a qualification round. Only annotators who reached Cohen’s $\kappa \ge 0 . 6 5$ were admitted to the main task. All 14 annotators met this threshold, with scores ranging from 0.66 to 0.81. During annotation, gold-check examples were inserted at a 2% rate. Annotators whose agreement dropped below the required threshold were paused, retrained on the relevant ambiguity cases, and re-qualified before continuing. Labels from afected batches were reviewed and re-annotated where necessary.

## A.4.4 Adjudication

Each comment was independently labeled by two annotators. If both annotators agreed, their shared label was accepted. If they disagreed, the comment was reviewed by a senior validator. Dificult cases were escalated to a small validator panel. For adjudicated comments, we record the two initial labels, annotator confidence scores, validator decision, and final gold label. These logs allow later analysis of disagreement patterns and label reliability.

## A.5 Quality Metrics Details

## A.5.1 Pairwise Inter-Annotator Agreement

We report Cohen’s $\kappa$ for agreement between the two primary annotators assigned to each comment. The overall pairwise score is $\kappa = 0 . 7 3$ , indicating substantial agreement. We compute:

$$
\kappa = { \frac { p _ { o } - p _ { e } } { 1 - p _ { e } } } ,\tag{2}
$$

where $p _ { o }$ is observed agreement and $p _ { e }$ is expected agreement under the annotators’ marginal label distributions. Per-phase κ ranges from 0.69 to 0.78.

## A.5.2 Multi-Rater Reliability

We also report Krippendorf’s α to measure reliability across the full annotator pool. The overall score is $\alpha = 0 . 7 1$ , showing that the labels are consistent beyond individual annotator pairs.

## A.5.3 Annotator Confidence

Annotators provided a confidence score in [0, 1] for each label. The mean confidence score is $\bar { c } = 0 . 8 2$ with standard deviation 0.14. In total, 87.3% of labels have confidence scores of at least 0.7. These scores are included as metadata so that users can filter examples by confidence if needed.

## A.5.4 Pre-Adjudication Agreement

The two primary annotators agreed directly on 78.4% of comments. The remaining 21.6% were sent to senior validators for adjudication. Among these, 3.1% required review by a three-validator panel.

## A.5.5 Independent Blind Audit

As a final quality check, two senior validators who were not involved in adjudication re-labeled a stratified random sample of 2,000 comments, with 400 comments from each phase. They were blind to the final gold labels. The audit achieved 94.2% exactlabel agreement and 96.8% agreement within one step on the sentiment scale. Most disagreements involved rhetorical questions and emoji-mediated sentiment, which were also identified as dificult cases during guideline development.

## A.6 Efect of Comment Length

We analyze sentiment distribution across wordlevel comment length intervals to examine whether afective polarity varies with comment verbosity. As shown in Figure 3, shorter comments are more frequently positive, with the positive share decreasing from 50% in 1–10 word comments to 20% in comments longer than 50 words. In contrast, neutral sentiment increases steadily with length, suggesting that longer comments tend to provide explanation, context, or factual discussion rather than direct afective reactions. Negative sentiment remains comparatively stable across all length ranges. This indicates that comment length is associated with sentiment composition, and that models may need to account for verbosity-driven shifts in crisis discourse.

## A.7 Quantifying Temporal Non-Stationarity

We measure lexical change across event phases using three signals: Jensen–Shannon (JS) divergence over unigram distributions, Jaccard similarity of top words, and Shannon entropy. The results are shown in Figure 4.

![](images/35b650d807788dd17e29d5b094d52fa8c7c7d1b912d2745f5c0a40921584f6e0.jpg)  
Figure 3: Sentiment distribution across sentence length intervals (word-wise). Short sentences (1–10 words) are dominated by positive sentiment, while the proportion of neutral sentiment increases with sentence length. Negative sentiment remains relatively stable across all length ranges.

JS divergence. JS divergence values range from 0.3713 to 0.3832 across phase pairs, showing that word distributions change over time.

Jaccard similarity. Top-word Jaccard similarity ranges from 0.4235 to 0.4652. This means that less than half of the most frequent terms are shared across phase pairs, indicating vocabulary turnover as the event develops.

Entropy. Shannon entropy increases from 7.5989 to 7.7872, suggesting that the discourse becomes more lexically diverse in later phases. Together, these results show that the corpus changes measurably across event phases. This supports the temporal-shift results in Table 5: performance drops are associated with changes in the data distribution, not only with model limitations. The corresponding sentiment-level shift, quantified using Cohen’s d on the polarity distributions of all $( \bar { \boldsymbol { \mathrm { 2 } } } ) = 1 0$ pairwise phase pairs, is reported in Table $7 ;$ seven of ten contrasts are negligible, with the strongest efects clustered around the postblackout transition mirroring the lexical-level drift pattern above.

![](images/548d9609beef61e987d3795f6f49efa5433e02708d43fad861ed5dfb372baa54.jpg)

![](images/73bb9113bab45bd5e74b878b2997dcae984630e725f84fbd157e89fea84455a0.jpg)

![](images/cd89ef3e123623af750245c143e9176ac13eea9667c913ec1e7559698075554a.jpg)  
Figure 4: Temporal drift across three chronological phases of Bangla text. Divergence, vocabulary turnover, and rising entropy together reveal substantial nonstationarity and increasing lexical diversity over time.

## A.8 Topic Modeling: Extended Methodology and Results

## Detailed Methodology

To understand how the online conversation changed during the uprising, we follow a simple three-step process. First, a multilingual sentence transformer represents each comment so that comments with similar meanings are placed close together. Second, UMAP reduces these representations to a smaller space. Third, HDBSCAN groups nearby comments into topics. We also use a custom list of 619 Bangla stopwords, including 158 discourse markers, 151 platform-specific artifacts, and 6 special tokens. Removing this noise helps the model focus on meaningful themes. Table 8 reports the topic statistics for each event phase.

We choose the model settings to keep the topics detailed but still easy to interpret. For the global model, UMAP uses n\_components=5, n\_- neighbors=20, and min\_dist=0.1. HDBSCAN uses min\_cluster\_size=20, min\_samples=5, and cluster\_selection\_epsilon=0.1. These lenient settings prevent the crisis discussion from being divided into too many small topics. CountVectorizer uses one- to three-word phrases, with min\_df=10 and max\_df=0.9, to retain useful Bangla phrases while removing terms that are too rare or too common.

The five event phases contain diferent numbers of comments, so one clustering setting would not suit every phase. We therefore scale min\_- cluster\_size from 25 in the smallest phase to 100 in the largest. If HDBSCAN marks a comment as an outlier (topic = −1), we assign it to the nearest topic centroid using cosine similarity. This gives every comment a topic. As a final check, we vary min\_cluster\_size by ±20%. Topic coherence changes by less than 0.05 (∆C<sub>v</sub> < 0.05), showing that the main findings remain stable under reasonable parameter changes.

## Extended Results

Attention rises and falls in sharp bursts. The story begins with an online discussion that does not grow smoothly. Instead, public attention moves in sudden waves as events unfold. Seventy-five percent of the topics have a coeficient of variation above $1 . 0 ,$ meaning that their activity changes greatly relative to their average level. The largest burst appears on August 9, three days after the collapse, when Topic T1 reaches 1,063 posts in a single day. A topic remains active for 44.8 days on average. The dominant topic, T0, changes momentum 15 times and has a volatility value of 4,269.78. Together, these patterns show how quickly attention moved from one concern to another during the crisis.

The themes change, but they remain connected. Although attention is highly volatile, the discussion does not repeatedly start from zero. Across 100 topic transformations, the mean semantic similarity is 0.93. In other words, themes usually evolve from earlier themes rather than disappear and reappear in unrelated forms. For example, student-protest discussion (ছাতৰ্লীেগর, আেন্দালন) develops into narratives about state violence $( \sqrt [ 3 ] { 1 0 } ) ^ { 5 }$ েতালা) with a similarity of 0.902. Confrontation rhetoric later becomes celebratory discourse (জিড়ত, হািসমাখা) with a similarity of 0.954, while its volume grows by 462% after the collapse.

Negative sentiment shapes most of the conversation. The final part of the story concerns emotion. Only Topic T8 (িসদ্ধান্ত, সংস্কার—decisions and reforms) has a positive polarity (+0.246). The most negative topics, T15 (−0.486) and T6 (−0.294), also carry a large share of the discussion. The topics are closely connected: we find 38 significant co-occurrence relationships with $r \ > \ 0 . 5$ The strongest link is between decision-making and public-reaction topics (T3 ↔ T1, r = 0.988), reflecting the rapid cycle of action and reaction common in crisis communication (Tufekci, 2017).

Taken together, the results tell a consistent story. Discussion stays centered on a small set of core grievances; these themes become more focused under stress, attract attention in sudden bursts, and continue to evolve across event phases. At the same time, sentiment remains mostly negative. Crisis discourse is therefore not simply a collection of disconnected conversations. It has a connected but rapidly changing structure, which calls for analysis methods designed specifically for crisis settings.

## A.9 Failure Mode Discovery: Topic-Level Sentiment Instability

Overall performance scores hide an important part of the story: a model may appear stable on average while becoming unreliable for a particular topic. We therefore return to the 25 BERTopic topics and follow each one across the five event phases. For every topic, we measure how much its sentiment polarity changes using the standard deviation (σ). This topic-level view complements the lexical drift shown in Figure 4 and reveals that sentiment instability is concentrated in specific themes.

Diferent topics follow diferent paths. Six topics are highly volatile $( \sigma > 3 0 )$ , 14 have medium volatility $( 1 5 ~ \leq ~ \sigma ~ \leq ~ 3 0 )$ , and 5 remain stable $( \sigma ~ < ~ 1 5 )$ . Topic T22 (েকাটা—quota reform) provides the clearest example. Its polarity changes from +57.1% to −33.3% between P3 and P4, giving $\sigma = 4 5 . 7 6$ and a range of 90.48 (Figure 5). Humanitarian topics, in contrast, remain much steadier, with $\sigma \approx 1 2 – 1 5$ . The key lesson is that temporal shift does not afect every topic in the same way.

Two turning points create the greatest risk. Fifteen topics show a dramatic sentiment flip $( \Delta \mathbf { \theta } >$ 40%), but most of the largest changes occur at two moments in the uprising. The first is the P2→P3 transition, after the internet blackout, when information returns and people begin to interpret what happened. Four topics change by more than 60% at this point. The second is the P3→P4 transition, during the regime collapse, when celebration and uncertainty appear together. Five topics reverse by more than $45 \%$ . For example, Topic T11 (সমসয্া— problems) moves from −15.0% to −45.6% during P2→P3 $( \Delta = 6 4 . 2 \% )$ . A classifier trained only on P2 would therefore underestimate P3 negativity for this topic by more than 30 percentage points.

This pattern explains where models are likely to fail. Politically charged themes, such as quota reform and league politics, are three times more volatile than humanitarian concerns. A single fixed model may therefore work well for stable topics but lose reliability when the meaning and sentiment of a political topic change. One practical response is topic-aware temporal recalibration: the model could lower or adjust its confidence when it detects a historically volatile topic. Future work could also prioritize these topics during temporal fine-tuning. Both strategies follow the same principle—adapt the model where the conversation changes most.

## A.10 Failure Mode Discovery through Topic Modeling

An overall accuracy score tells us how often a model is wrong, but it does not tell us whether the errors follow a common pattern. To find that pattern, we collect all 2,789 misclassified comments and apply BERTopic to the error set alone. Using a smaller configuration suited to this subset (min\_- cluster\_size=15 and one- to two-word phrases), the model finds 33 semantically coherent error topics with $C _ { v } ~ = ~ 0 . 9 4 9 9$ . We then follow the evidence from class-level errors to topic concentration, confidence, and complete sentiment reversals.

![](images/b523dd2e0744759e0c118265c2cac74a2de05e78fea594e90ad44a08f3457502.jpg)  
Figure 5: How topic-level sentiment changes across the five phases (P1–P5) of the July Revolution. A hollow circle shows a topic’s initial polarity, and a filled circle shows its final polarity; larger filled circles represent more documents. Arrows and percentage labels show the direction and size of each change. T22 (েকাটা) is highly volatile, moving from +57.1% to −33.3% $( \sigma = 4 5 . 7 6 )$ , while flood-relief topics remain comparatively stable $( \sigma \approx 1 2 – 1 5 )$ . The largest reversals occur after the internet blackout (P2→P3) and during the regime collapse (P3→P4), showing that sentiment shift depends strongly on both topic and time.

The main problem begins with neutral comments. The model classifies negative comments with 83.71% accuracy and positive comments with 63.10% accuracy, but neutral accuracy falls to 42.54%. Thus, 57.46% of neutral comments are misclassified. The largest single error type is neutral→negative: it occurs 895 times and accounts for 32.1% of all errors. This is the first sign that the model systematically reads ambiguous language as negative.

The errors gather around a small set of themes. They are not spread evenly across the 33 topics. Topic 0 alone contains 676 errors, or 24.2% of the full error set. Among its 322 neutral errors, 261 (81.1%) are predicted as negative. Large error clusters also appear around political discussion (ৈবষময্িবেরাধী আেন্দালেনর) and education policy (আেগর িশক্ষাকৰ্েম). These examples suggest that

Table 9: The ten topics containing the most classification errors. Topic 0 contains 676 errors (24.2% of all errors), and 57.46% of neutral comments are misclassified across the full error set. The mean confidence on incorrect predictions is 0.977, showing that the model is often highly confident even when it is wrong.
<table><tr><td>Topic</td><td>Errors</td><td>Conf.</td><td>Neu→Neg</td><td>Neu→Pos</td><td>Neg↔Pos</td></tr><tr><td>TO:  P</td><td>676</td><td>0.968</td><td>261</td><td>61</td><td>104</td></tr><tr><td>T1: </td><td>369</td><td>0.971</td><td>186</td><td>20</td><td>58</td></tr><tr><td>T2: </td><td>134</td><td>0.966</td><td>29</td><td>5</td><td>23</td></tr><tr><td>T3: 可 </td><td>110</td><td>0.968</td><td>40</td><td>18</td><td>29</td></tr><tr><td>T4: 种可 </td><td>92</td><td>0.970</td><td>14</td><td>13</td><td>38</td></tr><tr><td>T5: 可</td><td>89</td><td>1.000</td><td>0</td><td>85</td><td>4</td></tr><tr><td>T6: </td><td>75</td><td>0.980</td><td>40</td><td>7</td><td>10</td></tr><tr><td>T7:  </td><td>75</td><td>0.982</td><td>19</td><td>9</td><td>17</td></tr><tr><td>T8: 间 可</td><td>72</td><td>0.992</td><td>10</td><td>16</td><td>27</td></tr><tr><td>T9:</td><td>70</td><td>0.977</td><td>10</td><td>11</td><td>27</td></tr><tr><td>All Topics (33)</td><td>2,789</td><td>0.977</td><td>895</td><td>356</td><td>531</td></tr></table>

BanglaBERT often reacts to words associated with conflict or criticism but misses the neutral pragmatic meaning of the full comment. LoRA-SFT on a mid-scale LLM recovers several such contextdependent cases, as illustrated in Table 6.

The table confirms a consistent negative bias. As Table 9 shows, the model misclassifies 1,251 of the 2,177 neutral comments (57.46%). It predicts negative for 895 of them and positive for 356, a ratio of about 2.5:1. The same direction of bias appears in T1, T3, and T6, where neutral→negative errors clearly outnumber neutral→positive errors. The problem therefore extends beyond one topic: ambiguous Bangla discourse is repeatedly pulled toward the negative class.

High confidence makes these errors harder to detect. The model’s mean confidence across all incorrect predictions is 0.977. Topic 5 is the clearest warning: the confidence is 1.000 even though 85 neutral comments are predicted as positive. The model is therefore not simply uncertain about difficult examples; it is often confidently wrong. A standard probability threshold may fail to catch these cases, which motivates confidence calibration that also considers the topic.

Some mistakes reverse the sentiment completely. In 531 cases, or 19.0% of all errors, the prediction flips directly between negative and positive. Topic 0 contains 104 such flips and Topic 1 contains 58; both focus on ৈবষময্িবেরাধী আেন্দালেনর, where political framing can change the sentiment of otherwise similar language. Topic 4 (িশক্ষাথর্ীেদর সব্প্নেক—students’ dream) has an especially high flip rate: 38 of its 92 errors, or 41.3%. The full error story is therefore not only about neutral comments. BanglaBERT also struggles to distinguish hopeful and critical framings when they discuss the same political or educational issue.

## A.11 Structural Diagnosis: Where Do Models Fail?

The previous section showed what the model’s errors look like. We now ask why those errors gather in particular places. For this purpose, BERTopic (Grootendorst, 2022) gives us two connected views. The first follows the topic structure of the full corpus over time; the second shows where the model’s errors sit within that structure. The pipeline—multilingual sentence-transformer embeddings, UMAP, HDBSCAN, phase-specific parameters, and a 619-term Bangla stopword list— is described in Appendix A.8. Table 8 summarizes the results by phase. Two structural patterns explain much of the model behavior.

First, one political theme dominates the conversation. The global model finds 29 topics $( C _ { v } \ =$ 0.375, silhouette = 0.402), but T0 (লীগ, সরকার, েকাটা—league, government, quota) contains 56.3% of all documents $( n = 1 0 5 , 2 2 6 )$ . It is also highly stable over time $\mathrm { { ( C V = \ 0 . 0 1 4 ) } }$ and has a negative polarity (−0.224). The efective number of topics based on Shannon entropy is only 1.7–2.0 across the five phases. Thus, the discussion may look diverse on the surface, but most attention remains concentrated around one central political grievance. We call this pattern discourse hegemony.

This dominance creates a risk for sentiment models. Because T0 supplies such a large share of the training data, a model can learn to connect its political vocabulary with negative sentiment. It may then apply that negative tendency to neutral comments or to comments from other topics. This provides a structural explanation for the neutral→negative bias found in Appendix A.10.

Second, the blackout makes the discussion more focused, not more random. Topic coherence is not determined by the number of comments. P2, the Crisis & Blackout phase, contains 33,684 comments (16.9% of the corpus) and produces 29 topics, yet it has the highest coherence by a wide margin $( C _ { v } = 0 . 7 8 1 )$ ). Coherence in every other phase ranges only from 0.384 to 0.422. The largest phase, P4 (Post-Revolution), contains 69,536 comments (34.9%) but reaches only $C _ { v } = 0 . 4 2 0$ . The smallest phase, P1 (Pre-Escalation), contains 6,727 comments (3.4%) and reaches $C _ { v } = 0 . 4 0 5$

We call this result the crisis coherence paradox. During the blackout, limited information and immediate danger push people toward a small set of urgent concerns, making the discussion more semantically focused. After the crisis, attention spreads across more issues and coherence falls. Extreme conditions therefore crystallize the conversation instead of turning it into noise.

Together, these patterns explain where models fail. Errors cluster around the dominant topic and closely related political themes rather than appearing uniformly across the corpus. Politically charged topics, including quota reform and party politics, show three times the sentiment volatility of humanitarian topics, and 33% of all errors fall within T0’s thematic orbit. Temporal robustness should therefore be measured separately for each topic. In the same way, confidence recalibration should use topic-level volatility instead of relying only on one aggregate confidence score.

## A.12 Prompting Techniques

After identifying where sentiment models struggle, we test whether LLMs can solve the task from instructions and examples alone. We compare two prompting settings: zero-shot and few-shot. The model parameters remain frozen in both settings, so the only diference is the information supplied in the prompt. This design separates in-context reasoning from any benefit that could come from taskspecific fine-tuning.

Zero-shot prompting. We begin with the harder setting: the model receives no labeled examples. The prompt contains only a short expertrole instruction (for example, “You are an expert in Bengali political sentiment analysis”), the parent post, the target comment, the allowed labels {positive, neutral, negative}, and an instruction to return one label. The resulting score shows how well the model can understand crisis-era Bangla using its pretrained knowledge alone. Success therefore depends on its Bangla vocabulary, cultural and political understanding, and ability to follow instructions.

Few-shot prompting. Next, we give the same model seven labeled examples from the training split of UNRESTSENT200K. Each example contains a post, a comment, and the correct label. Together, the examples cover all three sentiment classes, both supportive and critical views of crisis events, and dificult language such as sarcasm, rhetorical questions, emoji descriptions, Bangla–English code-mixing, and indirect political references. The model then labels a held-out post–comment pair under the same one-label rule. Comparing this result with zero-shot performance shows how much seven relevant examples can help without changing the model itself.

Decoding and output format. Finally, we keep the decision process identical across both settings. We use greedy decoding with temperature = 0 and require exactly one label from {positive, neutral, negative}. If a response does not match an allowed label, we count it as an error and do not ask the model again. The reported results therefore measure the reliability of the complete prompting process, including instruction following, rather than the quality of predictions after manual correction or extra parsing. The complete prompt templates appear in §B.1.

## A.13 Expert Prompting with LLMs

Table 3 reports the zero-shot and few-shot performance of open-source and proprietary LLMs using expert prompting with Bangla-native instructions on the UnRestSent200K benchmark.

## B Ethics Statement

All data used in this work were collected from publicly accessible Facebook and YouTube content in accordance with platform terms of service. Personally identifiable information, including usernames and profile links, was removed during preprocessing.

## B.1 Sample Prompts

## িফউ-শট পৰ্ম্পট: েসিন্টেমন্ট অয্ােনােটশন (Few-Shot: Sentiment Annotation — Bengali)

System: তুিম বাংলা ভাষার রাজৈনিতক sentiment analysis িবেশষজ্ঞ। েতামােক একিট সংবাদ েপাস্ট/িশেরানাম এবং েসই েপােস্টর উপর করা একিট মন্তবয্ েদওয়া হেব। েতামার কাজ হেলা িনধর্ারণ করা েয মন্তবয্িট উিল্িখত েপাস্ট, ঘটনা, বা পৰ্সেঙ্গর পৰ্িত কী ধরেনর sentiment পৰ্কাশ করেছ।

User: িনেচ িকছু উদাহরণ েদওয়া আেছ। উদাহরণগুেলা অনুসরণ কের েশেষ েদওয়া পৰ্শ্নিটর উত্র দাও।

িনেদর্শনা:

• post শুধুমাতৰ্ contextual information িহেসেব বয্বহার করেব।

• sentiment িনধর্ারেণর সময় sarcasm, rhetorical tone, slang, emoji description (েযমন: \`কান্া জিড়ত হািসমুখ পৰ্িতিকৰ্য়া'), এবং mixed Bangla-English expression িবেবচনা করেব।

• রাজৈনিতক মতাদশর্ বা বয্িক্তগত অবস্থান নয়, শুধুমাতৰ্ comment-এর sentiment িবচার করেব।

• যিদ comment সমথর্ন, পৰ্শংসা, আশাবাদ, বা ইিতবাচক পৰ্িতিকৰ্য়া পৰ্কাশ কের → positive

• যিদ comment সমােলাচনা, হতাশা, রাগ, বয্ঙ্গ, আকৰ্মণ, বা েনিতবাচক পৰ্িতিকৰ্য়া পৰ্কাশ কের → negative

• যিদ comment তথয্িভিত্ক, পৰ্শ্নধমর্ী, দব্য্থর্ক, বা স্পষ্ট ইিতবাচক/েনিতবাচক sentiment না পৰ্কাশ কের → neutral

## উদাহরণসমূহ:

## উদাহরণ ১

েপাস্ট: েকাটািবেরাধী আেন্দালন : ঢাকা িবশব্িবদয্ালেয় িমিছল শুরু, মধুর কয্ানিটেন জেড়া হেয়েছ ছাতৰ্লীগ

মন্তবয্: সাবধান, আবার রক্ত ঝরােব।

উত্র: negative

উদাহরণ ২

েপাস্ট: অপরােধ সম্পৃক্ততায় যুবদল--ছাতৰ্দেলর ১৫ জনেক বিহষ্কার

মন্তবয্: নাটক কম কেরা িপও, জনগণ সব বুেঝ।

উত্র: negative

উদাহরণ ৩

েপাস্ট: আজ বৃহস্পিতবার েদেশর পৰ্ায় সব েমাবাইল অপােরটেরর ইন্টারেনট ধীরগিতেত চলেছ।

মন্তবয্: শ্লা টাউট! কান্া জিড়ত হািসমুখ পৰ্িতিকৰ্য়া

উত্র: negative

উদাহরণ ৪

েপাস্ট: েকাটািবেরাধী আেন্দালন : ঢাকা িবশব্িবদয্ালেয় িমিছল শুরু, মধুর কয্ানিটেন জেড়া হেয়েছ ছাতৰ্লীগ

মন্তবয্: ছাতৰ্লীগও েকাটা আেন্দালেন অংশ িনেয়েছ? ভাবনা মুখ পৰ্িতিকৰ্য়া

উত্র: neutral

উদাহরণ ৫

েপাস্ট: অপরােধ সম্পৃক্ততায় যুবদল--ছাতৰ্দেলর ১৫ জনেক বিহষ্কার

মন্তবয্: যােদর পদ েনই তােদর িক শািস্ত হেব?

উত্র: neutral

উদাহরণ ৬

েপাস্ট: েকাটািবেরাধী আেন্দালন : ঢাকা িবশব্িবদয্ালেয় িমিছল শুরু, মধুর কয্ানিটেন জেড়া হেয়েছ ছাতৰ্লীগ

মন্তবয্: েকাটা িসেস্টম িনপাত যাক। েমধাবীরা চাকির পাক।

উত্র: positive

উদাহরণ ৭

েপাস্ট: অপরােধ সম্পৃক্ততায় যুবদল--ছাতৰ্দেলর ১৫ জনেক বিহষ্কার

মন্তবয্: সিঠক সমেয়র সিঠক িসদ্ধান্ত ধনয্বাদ

উত্র: positive

এখন িনেচর উদাহরণিটর sentiment িনধর্ারণ কেরা:

েপাস্ট: {post}

মন্তবয্: {comment}

শুধুমাতৰ্ িনেচর একিট label পৰ্দান কেরা: positive, neutral, negative

## Few-Shot: Sentiment Annotation — English

System: You are an expert in political sentiment analysis for the Bangla language. You will be given a news post or headline and a comment made on that post. Your task is to determine what kind of sentiment the comment expresses toward the mentioned post, event, or context.

User: Some examples are given below. Follow the examples and answer the final question.

## Instructions:

• Use the post only as contextual information.

• While determining sentiment, consider sarcasm, rhetorical tone, slang, emoji descriptions such as “crying laughing face reaction”, and mixed Bangla-English expressions.

• Judge only the sentiment of the comment, not political ideology or personal position.

• If the comment expresses support, praise, optimism, or a positive reaction, label it positive.

• If the comment expresses criticism, frustration, anger, sarcasm, attack, or a negative reaction, label it negative.

• If the comment is factual, question-like, ambiguous, or does not express a clear positive or negative sentiment, label it neutral.

## Examples:

## Example 1

Post: Anti-quota movement: A procession has started at Dhaka University, and Chhatra League has gathered at Madhur Canteen.

Comment: Be careful, they will shed blood again.

Answer: negative

## Example 2

Post: Fifteen members of Jubo Dal and Chhatra Dal expelled for involvement in crimes.

Comment: Stop the drama, people understand everything.

Answer: negative

## Example 3

Post: Today, Thursday, internet service is slow across almost all mobile operators in the country.

Comment: Damn fraud! Crying laughing face reaction.

Answer: negative

## Example 4

Post: Anti-quota movement: A procession has started at Dhaka University, and Chhatra League has gathered at Madhur Canteen.

Comment: Has Chhatra League also joined the quota movement? Thinking face reaction.

Answer: neutral

## Example 5

Post: Fifteen members of Jubo Dal and Chhatra Dal expelled for involvement in crimes.

Comment: Will those who do not hold any position also be punished?

Answer: neutral

## Example 6

Post: Anti-quota movement: A procession has started at Dhaka University, and Chhatra League has gathered at Madhur Canteen.

Comment: Down with the quota system. Let meritorious students get jobs.

Answer: positive

## Example 7

Post: Fifteen members of Jubo Dal and Chhatra Dal expelled for involvement in crimes.

Comment: The right decision at the right time. Thank you.

Answer: positive

## Now determine the sentiment of the following example:

Post: {post}

Comment: {comment}

Provide only one of the following labels: positive, neutral, or negative.

## Zero-Shot Prompt: Political Expert (Bengali)

তুিম একজন বাংলা রাজৈনিতক sentiment analysis িবেশষজ্ঞ। িনেচর বাংলা েপাস্ট এবং মন্তবয্িটর sentiment িনধর্ারণ কেরা।

েপাস্ট: {post}

মন্তবয্: {comment}

শুধুমাতৰ্ একিট েলেবল দাও: positive, neutral, negative

## Zero-Shot: Political Expert — English

You are an expert in Bengali political sentiment analysis. Determine the sentiment of the following Bengali post and comment.

Post: {post}

Comment: {comment}

Respond with only one label: positive, neutral, or negative.