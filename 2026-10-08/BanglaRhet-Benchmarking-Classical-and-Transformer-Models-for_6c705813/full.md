# BanglaRhet: Benchmarking Classical and Transformer Models for Rhetorical and Persuasion Detection in Bangla Political Speech

1<sup>st</sup> Rohit Kumar Sen

Department of Computer Science and Engineering

North East University Bangladesh

Sylhet, Bangladesh

rohit.k.sen.neub@gmail.com

2<sup>nd</sup> Anik Chowdhury

Department of Computer Science and Engineering

North East University Bangladesh

Sylhet, Bangladesh

anikchowdhuryraj.neub@gmail.com

Abstract—Political discourse often uses rhetorical and persuasive language to frame narratives, influence public opinion, and mobilize audiences. While Bangla natural language processing has made progress in sentiment analysis, opinion mining, and pretrained language modeling, systematic benchmarking of transformer models for fine-grained rhetorical and persuasion technique detection in Bangla political speech remains largely underexplored. This paper presents a benchmark study of transformer-based models for detecting rhetorical form and persuasive intent in Bangla political discourse. Using BanglaRhet, a manually annotated corpus of 30,289 Bangla political speech segments collected from publicly available political news sources and speech-related articles, we formulate two supervised singlelabel classification tasks: rhetorical technique detection and persuasion technique detection. The rhetorical task captures stylistic forms such as contrast, repetition, exaggeration, metaphor, and rhetorical questions, while the persuasion task captures communicative functions such as blame assignment, call to action, unity call, moral appeal, emotional appeal, and logical appeal. We evaluate four transformer-based models, BanglaBERT, Bangla-BERT-Base, SahajBERT, and XLM-RoBERTa-Base, against classical TF-IDF baselines. BanglaBERT achieves the highest performance, with 65.40% macro-F1 for rhetorical technique detection and 66.46% macro-F1 for persuasion technique detection, outperforming the best tuned classical baseline by 19.2 macro-F1 points for rhetorical technique detection and 13.8 macro-F1 points for persuasion technique detection. Class-level analysis indicates that model errors are mainly associated with semantic overlap among labels, figurative language, and class imbalance. The results provide initial benchmark baselines for Bangla rhetorical and persuasion-aware political discourse analysis and highlight the need for future work on context-aware and multi-label modeling.

Index Terms—Bangla NLP, political discourse analysis, rhetorical technique detection, persuasion technique detection, transformer models, low-resource NLP

## I. Introduction

Political discourse, including speeches, debates, and news commentary, is a powerful medium for shaping public opinion, framing ideological positions, and influencing social behavior [1], [2]. Political texts often rely on rhetorical and persuasive strategies to strengthen arguments, assign responsibility, appeal to values, and mobilize audiences [1], [3]. Automated analysis of these strategies can support research in political science, the social sciences, computational linguistics, and natural language processing (NLP).

In this study, we distinguish between two complementary aspects of political language: rhetorical techniques, which describe how a statement is expressed, and persuasion techniques, which describe the communicative purpose of the statement. Detecting these aspects is challenging because political language often contains figurative expressions, emotionally charged wording, implicit intent, and overlapping categories. As a result, models must capture stylistic and pragmatic cues beyond surface-level meaning.

Despite recent progress in Bangla NLP, existing Bangla political text classification studies have mainly focused on sentiment analysis, opinion mining, polarity detection, and related tasks [4], [5]. Although useful, these tasks primarily capture broad attitudes or opinions and do not directly model the rhetorical structure or persuasive function of political statements. Fine-grained, technique-level taxonomies exist for related English tasks such as propaganda technique detection [6], but no equivalent resource exists for Bangla. Consequently, benchmark evaluation for fine-grained rhetorical and persuasion-aware analysis in Bangla political discourse remains largely underexplored.

To address this gap, this paper presents a transformerbased benchmark study using BanglaRhet, a manually annotated corpus of 30,289 Bangla political speech segments collected from publicly available political news sources and speech-related articles [5], [7]. We evaluate Bangla-specific transformer models, including BanglaBERT [8], SahajBERT [9], and Bangla-BERT-Base [10], along with the multilingual baseline XLM-RoBERTa-Base [11].

The main contributions of this work are: (i) a benchmark framework for fine-grained rhetorical and persuasion technique detection tasks in Bangla political discourse; (ii) an evaluation of Bangla-specific and multilingual transformer models using 30,289 manually annotated political speech segments; (iii) an empirical comparison against classical TF-IDF-based baselines confirming the necessity of transformerbased representations for this task; and (iv) class-level analysis of challenges caused by overlapping label semantics, figurative usage, and class imbalance.

## II. Background and Related Work

A. Rhetorical and Persuasive Language in Political Discourse Political communication often relies on rhetorical and persuasive strategies to influence audience perception, frame ideological positions, and strengthen arguments. Rhetorical techniques describe how political language is expressed, while persuasion techniques describe the communicative intent or strategic function of a statement [1], [2]. Based on this distinction, this study models rhetorical categories such as contrast, repetition, metaphor, exaggeration, and rhetorical questions, and persuasion categories such as blame assignment, moral appeal, logical appeal, emotional appeal, and call to action.

## B. Political Text Analysis and Persuasion Detection

Computational political text analysis has been studied through sentiment analysis, ideology detection, stance detection, argument mining, and persuasion detection. Prior work has analyzed political sentiment and public opinion from news texts, social media posts, and online comments [12], but such polarity scores cannot distinguish, for example, a metaphordriven appeal from a direct call to action expressed with the same sentiment. Transformer-based models have also been applied to political communication tasks, such as PoliBERT for classifying political social media messages [13], though this targets short social-media posts using coarse functional categories rather than technique-level rhetorical form.

Persuasion-related studies include the CORPS corpus, which used political speeches and audience applause as a weak, effect-level proxy for persuasiveness rather than a technique-level label [14], and traditional persuasiveness prediction methods that output a single scalar score rather than a categorical taxonomy [15]. Recent persuasion detection work in advertising, online campaigns, and social media adopts finer-grained technique labels [16], but remains limited to English, non-political domains. Closest to our setting is SemEval-2020 Task 11, which introduced a 14-category taxonomy of propaganda techniques (e.g., loaded language, repetition, exaggeration, appeal to fear) for English news, with transformerbased models performing best [6]. This confirms that finegrained technique taxonomies are learnable, but no equivalent exists for Bangla, and none of the above jointly model stylistic form and communicative function as we do.

## C. Bangla Political NLP and Existing Resources

Bangla political NLP research has mainly focused on sentiment analysis, opinion mining, polarity detection, ideology, and stance-related tasks. The Motamot dataset provides Bangla political news texts annotated for binary sentiment and supports comparative evaluation of Bangla NLP models [5]; sentiment, however, is a downstream byproduct of rhetorical strategy, so two segments can share sentiment while using entirely different techniques. Other studies analyze political sentiment from YouTube and news comments [4], targeting audience reaction rather than the speaker’s own discourse, and political ideology or stance prediction has been explored with classical machine learning methods that model leaning rather than rhetorical device [17]. Bengali transnational political discourse resources such as BTPD [18] hand-curate discussion from online Q&A and social platforms and characterize it through unsupervised topic modeling; the resource is not annotated for rhetorical or persuasive technique and targets informal community discussion rather than formal political speech.

Overall, prior Bangla political NLP operates almost exclusively at the sentiment or stance level, while technique-level rhetorical and persuasion analysis, established for English propaganda detection [6], has no Bangla counterpart. BanglaRhet addresses this by introducing (i) a Bangla-specific taxonomy spanning nine rhetorical and seven persuasion techniques, (ii) a manually annotated benchmark of 30,289 political speech segments, to our knowledge the first of its kind for Bangla, and (iii) an evaluation of whether Bangla-specific pretraining offers a measurable advantage over multilingual pretraining on this fine-grained task.

## III. Benchmark Dataset and Task Formulation

## A. Data Collection and Segment Extraction

BanglaRhet was built from publicly available Bangla political news and speech-related articles. We used 1,890 links from the existing Motamot dataset [5], supplemented with links from Prothom Alo [7] and Manab Zamin [19] covering 2014– 2025. In total, 15,032 articles were successfully processed into usable segments after removing inaccessible, duplicate, malformed, or non-political pages.

Articles were manually reviewed to identify political speech segments: a segment was included if it contained a direct quotation or reported statement attributed to a political figure or institution, by name or by role or position (e.g., a local MP), as characterized by the source outlet; where the speaker’s identity could not be determined, the segment was retained and marked unknown\_speaker. Journalistic analysis, background context, and commentary from non-political actors were excluded, except where retained to construct the None class. Extraction was performed collaboratively, with segments periodically crosschecked and ambiguous cases resolved through discussion. Passages exceeding the 128-token input limit were condensed, with both authors manually verifying that rhetorical and persuasive content was preserved.

## B. Task Formulation, Labels, and Annotation

We formulate two independent classification tasks per segment x: rhetorical technique $y _ { r }$ and persuasion technique $y _ { p } ,$ each assigned its dominant label; multi-label modeling is left for future work. The rhetorical task has nine labels (Contrast, Emotional Language, Repetition, Exaggeration, Anecdote, Rhetorical Question, Metaphor, Parallelism, and None); the persuasion task has seven (Blame Assignment, Call to Action, Unity Call, Emotional Appeal, Moral Appeal, Logical Appeal, and None). Table I gives the operational definition of each label. When a segment plausibly exhibited more than one technique, annotators assigned the single dominant label, i.e., the one most central to the segment’s overall effect, with disagreements resolved by discussion.

TABLE I Operational label definitions.
<table><tr><td colspan="2">Rhetorical Techniques</td></tr><tr><td>Contrast</td><td>Opposing ideas presented to emphasize dif- ference.</td></tr><tr><td>Emotional</td><td>Emotionally charged wording to influence</td></tr><tr><td>Lang. Repetition</td><td>feeling. Repeated words/phrases to reinforce a mes-</td></tr><tr><td>Exaggeration</td><td>sage. Deliberate overstatement for rhetorical im-</td></tr><tr><td>Anecdote</td><td>pact. Concrete example or personal experience</td></tr><tr><td>Rhet. Question</td><td>as support. Question posed to provoke reflection, not</td></tr><tr><td>Metaphor</td><td>an answer. Figurative comparison for im-</td></tr><tr><td>Parallelism</td><td>agery/interpretation. Repeated grammatical structure for empha-</td></tr><tr><td>None</td><td>sis. No predefined rhetorical technique present.</td></tr><tr><td colspan="2">Persuasion Techniques</td></tr><tr><td>ment</td><td>Blame Assign- Attributing fault to an opposing party.</td></tr><tr><td>Call to Action</td><td>Urging the audience toward a specific ac- tion.</td></tr><tr><td>Unity Call Emotional  $\mathrm { A p \mathrm { - } }$ </td><td>Promoting solidarity or collective identity. Persuasion via emotional reaction over</td></tr><tr><td>peal Moral Appeal Logical Appeal</td><td>logic. Appeal to ethics, justice, or moral duty. Persuasion via facts, evidence, or reason-</td></tr><tr><td>None</td><td>ing. No predefined persuasion technique present.</td></tr></table>

A total of 30,289 segments were manually annotated by two authors with prior study of political discourse. Both jointly annotated a pilot sample to converge on shared label boundaries, particularly for conceptually related pairs (e.g., Emotional Language vs. Emotional Appeal), continuing until disagreement stabilized before annotating the full corpus. A segment was labeled None when it showed no discernible instance of any technique, or, less commonly, when it contained borderline language not clearly satisfying any single label’s definition. Inter-annotator reliability, measured on a random 500-segment subset before consensus resolution, showed moderate-to-substantial agreement: Cohen’s κ = 0.67 for rhetorical technique and 0.71 for persuasion technique.

## C. Benchmark Statistics and Imbalance

BanglaRhet contains 30,289 segments drawn from 15,032 sources, averaging 48.34 words per segment, with 9 rhetorical and 7 persuasion labels (each including None). Figure 1 shows both label distributions. The benchmark is naturally imbalanced: Contrast, Emotional Language, and Repetition dominate the rhetorical dimension while Parallelism is rare; Blame Assignment, Call to Action, and Unity Call dominate the persuasion dimension. This imbalance motivates the macro-F1 metric and the class-balancing strategy described in Section IV. The rhetorical distribution also shows a sharp discontinuity: consecutive classes differ by at most 1.4x, but Parallelism (137 instances) is over 15 times rarer than the next smallest class, Metaphor (2,128), yielding only about 14 test instances under our split, too few for stable macro-F1; we therefore excluded it rather than oversampling (risk of near duplicate memorization) or merging it with Repetition, a structurally distinct category in our taxonomy (Table I).

![](images/8c2b1e9a932d6ef717a2d5b897e6c64407f0cdf669937c3ee164537d2af301c9.jpg)  
(a) Rhetorical techniques

![](images/080d7744508c7f4807c9fc31cc42cac9501a7cbf2b2e613ae809a8c42fbecf7b.jpg)  
(b) Persuasion techniques  
Fig. 1. Label distributions in BanglaRhet: (a) rhetorical techniques, (b) persuasion techniques.

## IV. Experimental Setup

The objective of the experiments is to establish benchmark results for rhetorical and persuasion detection tasks in Bangla political speech. Rather than proposing a new model architecture, this work evaluates whether widely used Bangla-specific and multilingual transformer models can learn fine-grained rhetorical and persuasive patterns under a fixed experimental setting.

## A. Task Setup

The processed benchmark was split 80%–10%–10% into training, validation, and test sets, stratified on the rhetorical labels; persuasion labels were encoded separately over the same splits. Segments were tokenized with each model’s tokenizer, padded or truncated to 128 tokens.

## B. Class Balancing Strategy

The original benchmark is naturally imbalanced, with some rhetorical and persuasion techniques occurring much more frequently than others. Critically, this imbalance is not merely per-class but joint: Blame Assignment, Call to Action, and Unity Call together account for 67.2% of all segments, and they also dominate the co-occurrence counts of the most frequent rhetorical labels (e.g., Contrast co-occurs with Blame Assignment 2,923 times, versus at most 824 times with any other persuasion label). Since uniform per-class downsampling cannot correct this pairwise skew, we instead manually inspected the full rhetorical–persuasion co-occurrence table and reduced the ten most overrepresented pairs, as shown in Table II. Parallelism was excluded separately from the experimental benchmark for the reasons discussed in Section III-C.

These reductions were applied through iterative manual inspection of the co-occurrence table rather than a fixed formula, which we acknowledge as a limitation in Section VI; however, the ablation in Table IV confirms this step was not merely cosmetic, as training without these reductions yields consistently lower macro-F1 across all four evaluated models.

After removing Parallelism and applying controlled downsampling, the processed experimental benchmark contained 22,565 samples, while the full annotated BanglaRhet corpus contained 30,289 segments. This processed benchmark was used for the train–validation–test split. During training, classweighted cross-entropy loss was applied for both tasks to further reduce majority-class bias.

TABLE II  
Controlled downsampling rules applied to overrepresented co-occurrence pairs across the rhetorical and persuasion tasks.
<table><tr><td>Rhetorical Label Contrast</td><td>Persuasion Label</td><td>Removed</td></tr><tr><td>Emotional Language Emotional Language Exaggeration Repetition Repetition None None None Anecdote</td><td>Blame Assignment Blame Assignment Call to Action Blame Assignment Blame Assignment Call to Action Call to Action Blame Assignment Unity Call Unity Call</td><td>2000 1571 447 537 375 1100 300 685 300 272</td></tr></table>

## C. Baseline Models

We evaluate four pretrained transformer models commonly used in Bangla NLP research: Bangla-BERT-Base [10], SahajBERT [9], BanglaBERT [8], and XLM-RoBERTa-Base [11]. The first three are Bangla-specific pretrained models, while XLM-RoBERTa-Base is a multilingual cross-lingual baseline, allowing us to examine whether Bangla-specific pretraining offers advantages over multilingual pretraining for this task; all models were fine-tuned using the standard sequence classification head from the Hugging Face Transformers library. To justify this choice empirically, we also trained two classical baselines, TF-IDF with Logistic Regression and TF-IDF with a linear SVM, with hyperparameters (regularization strength, vocabulary size, n-gram range) selected via grid search on the validation set; as shown in Table III, both are outperformed by every transformer model even after tuning, confirming that this task needs representations beyond bag-ofwords features.

## D. Training Configuration and Evaluation

All models were trained for ten epochs using the AdamW optimizer with a learning rate of ${ \mathrm { 2 } } \times 1 0 ^ { - 5 } { \mathrm { . } }$ , a batch size of 16, and a maximum sequence length of 128 tokens. A single random seed (42) was fixed throughout for data splitting, controlled downsampling, and model training, ensuring a consistent, reproducible comparison across models. The best-performing checkpoint was selected based on validation macro-F1, with class-weighted cross-entropy loss (weights inversely proportional to class frequency) applied for both tasks, which can raise macro-F1 relative to accuracy by improving minority-class performance. Other settings followed the default Hugging Face Trainer configuration; extensive hyperparameter tuning was not performed, as the goal is reproducible benchmark baselines rather than maximizing architecture-specific performance. Models are evaluated on the held-out test set using accuracy and macro-F1, with macro-F1 emphasized given the benchmark’s imbalance. The dataset, annotation guidelines, preprocessing scripts, and experimental code will be released in a public repository upon acceptance.

## V. Results and Analysis

## A. Benchmark Results

Table III reports test-set performance for all evaluated models, including the tuned TF-IDF baselines. Both classical baselines are outperformed by every transformer model on both tasks. BanglaBERT achieves the highest performance overall, with 65.40% macro-F1 for rhetorical detection and 66.46% macro-F1 for persuasion detection, a gain of 19.2 and 13.8 points respectively over the best tuned classical baseline (Table III). These results suggest that BanglaBERT’s Bangla-specific pretraining is effective for capturing stylistic and pragmatic cues in Bangla political discourse, although the moderate macro-F1 scores indicate that both tasks remain challenging.

To examine the effect of controlled downsampling, we conducted an ablation experiment without the downsampling rules while keeping the same label set, random seed, classweighted loss, and training configuration. The rare Parallelism class was excluded in both settings to ensure a consistent label space. As shown in Table IV, controlled downsampling improves macro-F1 across all models for both tasks, with the largest single gain observed for BanglaBERT on rhetorical detection.

## B. Effect of Source-Level Splitting

Since segment-level splitting can allow segments from the same source article to appear in both training and test sets, we additionally evaluated BanglaBERT under a source-grouped split, using GroupShuffleSplit on the article source to guarantee no article contributes segments to more than one split. Under the original segment-level split, 61.23% of test segments and 59.35% of validation segments shared a source article with the training set. Table V compares BanglaBERT’s performance under both splits.

Despite substantial source overlap under the original split, macro-F1 drops by only 1.02 and 1.08 points for rhetorical and persuasion detection respectively under the leakage-free source-level split. This modest degradation suggests the model is not substantially reliant on memorized source-specific phrasing or speaker style, and that the benchmark results in Table III are not meaningfully inflated by segment-level splitting.

## C. Confusion Matrices and Per-Class Analysis

To examine class-level behavior, we analyze the confusion matrices of the best-performing model, BanglaBERT. Figure 2 shows the confusion matrices for rhetorical and persuasion detection.

For rhetorical detection, many errors occur among stylistically related categories. One notable confusion appears between Emotional Language and Repetition, where 46 Emotional Language instances are predicted as Repetition. For example, a test segment, শহীদেদর স্īŚপ্ন এখেনা পূরণ হয় িন... আমােদর একিত্রত হেয় কাজ করেত হেব (“The martyrs’ dream has still not been fulfilled... we must come together and work”), was labeled Emotional Language but predicted Repetition: the model latches onto the lexical repetition of unityrelated phrases, even though annotators judged the segment’s dominant effect to be emotional appeal to sacrifice rather than structural reinforcement. Additional confusion occurs between Repetition and Contrast, and between Exaggeration and Emotional Language. These patterns indicate that rhetorical devices in political speech often overlap in surface form and expressive function.

TABLE III  
Benchmark performance on the test set.
<table><tr><td rowspan="2">Model</td><td colspan="2">Rhetorical</td><td colspan="2">Persuasion</td></tr><tr><td>Acc</td><td>Macro-F1</td><td>Acc</td><td>Macro-F1</td></tr><tr><td>TF-IDF + LR</td><td>0.4599</td><td>0.4587</td><td>0.5321</td><td>0.5270</td></tr><tr><td>TF-IDF + SVM</td><td>0.4692</td><td>0.4624</td><td>0.5241</td><td>0.5114</td></tr><tr><td>Bangla-BERT-Base</td><td>0.5237</td><td>0.5285</td><td>0.5733</td><td>0.5720</td></tr><tr><td>SahajBERT</td><td>0.5853</td><td>0.5974</td><td>0.6185</td><td>0.6227</td></tr><tr><td>BanglaBERT</td><td>0.6460</td><td>0.6540</td><td>0.6619</td><td>0.6646</td></tr><tr><td>XLM-RoBERTa-Base</td><td>0.6043</td><td>0.6110</td><td>0.6349</td><td>0.6405</td></tr></table>

TABLE IV  
Effect of controlled downsampling on macro-F1.
<table><tr><td>Model</td><td>Rhetorical No DS</td><td>Persuasion DS</td></tr><tr><td rowspan="3">Bangla-BERT-Base SahajBERT</td><td>DS 0.5238 0.5285</td><td>No DS 0.5500 0.5720</td></tr><tr><td>0.5821 0.5974</td><td>0.6086 0.6227</td></tr><tr><td>0.6214 0.6540</td><td>0.6523 0.6646</td></tr><tr><td>BanglaBERT XLM-RoBERTa-Base</td><td>0.6005 0.6110</td><td>0.6259 0.6405</td></tr></table>

TABLE V  
BanglaBERT macro-F1 under segment-level vs. source-level splitting.
<table><tr><td>Split</td><td>Rhetorical</td><td>Persuasion</td></tr><tr><td>Segment-level (original)</td><td>0.6540</td><td>0.6646</td></tr><tr><td>Source-level (grouped)</td><td>0.6438</td><td>0.6538</td></tr></table>

For persuasion detection, Unity Call and None are classified more reliably, while Blame Assignment, Moral Appeal, and Emotional Appeal show more confusion; Moral Appeal and Emotional Appeal are frequently predicted as Blame Assignment, since the explicit naming of a rival party or a specific accusation gives the model a strong blame-oriented surface cue that overrides the underlying moral or emotional framing.

Table VI reports per-class F1 scores computed from the BanglaBERT confusion matrices. The results show stronger performance for categories with clearer lexical or functional cues, such as None, Rhetorical Question, and Unity Call. In contrast, lower F1 scores are observed for semantically overlapping or more difficult categories such as Emotional Language, Blame Assignment, Moral Appeal, and Logical Appeal. This suggests that model performance is affected not only by class frequency but also by conceptual overlap among fine-grained political language categories.

![](images/3098e0c68ec6bade19a9e432d4ad00ac16e53bc10c0c3d171d3a34d3832551db.jpg)  
(a) Rhetorical

![](images/08febf7fd1f34f8a8880ac7e22113b18f776519c3eef61e304c4328ed54dbe31.jpg)  
(b) Persuasion  
Fig. 2. Confusion matrices of BanglaBERT: (a) rhetorical technique detection, (b) persuasion technique detection.

TABLE VI  
Per-class F1 scores of BanglaBERT.
<table><tr><td rowspan=1 colspan=3>Rhetorical Label</td><td rowspan=1 colspan=1>F1</td></tr><tr><td rowspan=2 colspan=3>ContrastEmotional LanguageExaggerationMetaphor</td><td rowspan=1 colspan=1>0.5900.5510.597</td></tr><tr><td rowspan=2 colspan=1></td><td rowspan=1 colspan=1>0.656</td></tr><tr><td rowspan=3 colspan=3>NoneRepetitionRhetorical Question</td><td rowspan=1 colspan=1>0.794</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>0.646</td></tr><tr><td rowspan=1 colspan=1>0.752</td></tr><tr><td rowspan=1 colspan=3>Anecdote</td><td rowspan=1 colspan=1>0.666</td></tr></table>

<table><tr><td>Persuasion Label</td><td>F1 0.564</td></tr><tr><td>Blame Assignment Call to Action Emotional Appeal Logical Appeal Moral Appeal None Unity Call</td><td>0.681 0.625 0.586 0.576 0.840 0.813</td></tr></table>

## D. Discussion

BanglaBERT achieves the highest performance on both tasks, suggesting that its Bangla-specific pretraining is effective for this fine-grained classification setting and that bag-of-words representations are insufficient for capturing rhetorical and persuasive cues in Bangla political speech. XLM-RoBERTa-Base performs competitively but remains below BanglaBERT, indicating that multilingual representations do not fully capture Bangla-specific rhetorical and political phrasing.

Controlled downsampling improves macro-F1 across all evaluated models for both tasks, with the largest gain observed for BanglaBERT on rhetorical detection (+3.26 points); the effect is more mixed for the remaining models, where persuasion detection sometimes benefits more, possibly because Call to Action, Unity Call, and None carry clearer lexical or functional cues than the rhetorical categories.

For context, BanglaBERT reaches 88.10% accuracy on Motamot’s binary political sentiment task [5], a considerably coarser problem than the 9-way and 7-way distinctions modeled here. A closer comparison is SemEval-2020 Task 11, a similarly fine-grained propaganda technique classification task (14 techniques) for English news, where the best of 32 systems reached 62.1% micro-F1 and several BERT-based systems reported macro-F1 of 0.32–0.45 under comparable imbalance [6]. Our 65.40% and 66.46% macro-F1 are broadly in line with, and in some respects ahead of, this closest related benchmark, suggesting the difficulty observed here is typical of technique-level classification generally.

Confusion matrices show model errors are strongly tied to semantic overlap: for rhetorical detection, confusion among Emotional Language, Repetition, and Exaggeration suggests expressive political language often combines intensity, repetition, and emotional wording, while for persuasion detection, confusion among Blame Assignment, Moral Appeal, and Emotional Appeal suggests statements often combine blame, emotion, and value-based argumentation within the same segment.

## VI. Limitations and Ethical Considerations

## A. Limitations

Several limitations remain. First, rhetorical and persuasion annotation is inherently subjective, especially for semantically overlapping categories such as Emotional Language vs. Exaggeration and Emotional Appeal vs. Moral Appeal; the Cohen’s kappa scores of 0.67 and 0.71 indicate moderate to substantial agreement, but some ambiguity remains unavoidable in finegrained political discourse annotation. Second, despite controlled downsampling and class-weighted loss, the benchmark remains imbalanced and minority or overlapping categories are harder to classify, as reflected in the moderate macro-F1 and per-class scores. The downsampling itself was applied through manual inspection of the co-occurrence table rather than an automated, parameterized rule; we release the exact resulting counts and data for reproducibility, but future work should formalize this step. The proportion of segments requiring manual condensation was not logged during annotation and cannot be reconstructed retrospectively. Third, the data are temporally and contextually bounded: collected articles span 2014–2025 and are mainly drawn from Bangladeshi political news sources, so performance may vary across time periods, speakers, or Bangla-speaking regions. Finally, both tasks are formulated as single-label classification using the dominant label per segment, though political statements may contain multiple rhetorical or persuasive strategies, making multi-label modeling an important direction for future work.

## B. Ethical Considerations

All data were collected from publicly accessible news articles and speeches. The benchmark analyzes linguistic and rhetorical patterns only, not the truthfulness or political validity of any statement, and does not endorse or rank any political individual or group. It should be used only for research and education, not political profiling or voter manipulation; all extraction and annotation were performed manually by the authors.

## VII. Conclusion

This paper presents a transformer-based benchmark for Bangla rhetorical and persuasion detection tasks using BanglaRhet, a manually annotated corpus of 30,289 political speech segments. We evaluated Bangla-specific and multilingual transformer models, including Bangla-BERT-Base, SahajBERT, BanglaBERT, and XLM-RoBERTa-Base. BanglaBERT achieved the highest performance, with 65.40% macro-F1 for rhetorical detection and 66.46% macro-F1 for persuasion detection. All transformer models outperformed tuned classical TF-IDF baselines by a wide margin, confirming that this task requires representations beyond bag-of-words features.

The ablation results show that controlled downsampling improves macro-F1 across all evaluated models, most notably for BanglaBERT on rhetorical detection. A source-grouped reevaluation further confirms these results are not substantially inflated by source-level leakage, with macro-F1 dropping by only about one point under a fully source-disjoint split. Confusion matrices and per-class analyses further show that errors are concentrated among semantically related categories, particularly where emotional, moral, exaggerative, and blameoriented language overlap. Overall, the results establish initial benchmark baselines and indicate that analysis of rhetorical and persuasion-aware political discourse in Bangla remains challenging. Future work will focus on broader Banglaspeaking political contexts, improved minority-class modeling, source-level evaluation, and context-aware or multi-label approaches.

## References

[1] T. Sultonov, “The use of rhetorical devices in political discourse,” International Journal of Artificial Intelligence, vol. 1, no. 3, pp. 1354– 1358, 2025.

[2] A.-M. F. Z. Salih, “The persuasive language strategies and techniques in political discourse: Differences and similarities between the united states of america and the middle east region in the language of political speeches,” Al-Mansour Journal, vol. 41, no. 1, pp. 236–262, 2024.

[3] S. Afzal and A. Hassan, “Using rhetorical and persuasive techniques: A political discourse analysis of the victory speech by imran khan,” Journal of Communication and Cultural Trends, vol. 3, no. 2, pp. 105– 121, 2021.

[4] M. Arman, A. Roy, M. A. Rusho, M. R. Riyan, M. J. Hossain, N. N. Hasan, N. M. Dipu, M. A. A. Chowdhury, and M. K. Jahan, “Evaluating transformer-based models for bengali political sentiment analysis,” in 5th International Conference on Electrical, Computer and Energy Technologies (ICECET 2025), 2025.

[5] F. T. J. Faria, M. B. Moin, R. I. Mumu, M. M. A. Abir, A. N. Alfy, and M. S. Alam, “Motamot: A dataset for revealing the supremacy of large language models over transformer models in bengali political sentiment analysis,” in 2024 IEEE Region 10 Symposium (TENSYMP). IEEE, 2024, pp. 1–8.

[6] G. Da San Martino, A. Barrón-Cedeño, H. Wachsmuth, R. Petrov, and P. Nakov, “SemEval-2020 task 11: Detection of propaganda techniques in news articles,” in Proceedings of the Fourteenth Workshop on Semantic Evaluation. International Committee for Computational Linguistics, 2020, pp. 1377–1414.

[7] Prothom Alo, “Prothom alo website,” https://www.prothomalo.com.

[8] A. Bhattacharjee, T. Hasan, W. U. Ahmad, K. S. Mubasshir, M. S. Islam, A. Iqbal, M. S. Rahman, and R. Shahriyar, “Banglabert: Language model pretraining and benchmarks for low-resource language understanding evaluation in bangla,” in Findings of the Association for Computational Linguistics: NAACL 2022, 2022, pp. 1318–1327.

[9] A. Bhattacharjee and K. S. Hasan, “Sahajbert: A pretrained transformerbased model for bangla language understanding,” arXiv preprint arXiv:2101.03837, 2021.

[10] S. Sarker, “Banglabert: Bengali mask language model for bengali language understanding,” https://huggingface.co/sagorsarker/bangla-bertbase, 2020.

[11] A. Conneau et al., “Unsupervised cross-lingual representation learning at scale,” arXiv preprint arXiv:1911.02116, 2019.

[12] M. S. Hossain, M. R. Islam, B. R. Riskhan, M. M. H. HASAN, and R. I. ISLAM, “Political sentiment analysis using natural language processing on social media,” International Journal of Applied Methods in Electronics and Computers, vol. 12, no. 4, pp. 81–89, 2024.

[13] S. Gupta, S. Bolden, J. Kachhadia, A. Korsunska, and J. Stromer-Galley, “Polibert: Classifying political social media messages with bert,” in Social, cultural and behavioral modeling (SBP-BRIMS 2020) conference. Washington, DC, 2020.

[14] M. Guerini, C. Strapparava, and O. Stock, “Corps: A corpus of tagged political speeches for persuasive communication processing,” Journal of Information Technology & Politics, vol. 5, no. 1, pp. 19–32, 2008.

[15] C. Strapparava, M. Guerini, O. Stock et al., “Predicting persuasiveness in political discourses.” in LREC, 2010.

[16] E. Meguellati, S. Civelli, P. Bernardelle, S. Sadiq, I. King, and G. Demartini, “Towards detecting persuasion on social media: From model development to insights on persuasion strategies,” arXiv preprint arXiv:2503.13844, 2025.

[17] Z. Tasnim, S. Ahmed, A. Rahman, J. F. Sorna, and M. Rahman, “Political ideology prediction from bengali text using word embedding models,” in 2021 international conference on emerging smart computing and informatics (ESCI). IEEE, 2021, pp. 724–727.

[18] D. Das, S. I. Ahmed, and S. Guha, “Btpd: A multilingual handcurated dataset of bengali transnational political discourse across online communities,” in Companion Publication of the 2025 Conference on Computer-Supported Cooperative Work and Social Computing, 2025, pp. 188–193.

[19] Manab Zamin, “Manab zamin website,” https://www.mzamin.com/.