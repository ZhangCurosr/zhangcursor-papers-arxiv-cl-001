# Detecting GPT-Assisted Writing Using Interpretable Stylometric Features

Rajesh Kumar Bucknell University rajesh.kumar@bucknell.edu

Nabeel Siddiqui Susquehanna University siddiqui@susqu.edu

Alexander Fuchsberger Bucknell University af033@bucknell.edu

## Abstract

Distinguishing GPT-assisted from independently authored student writing has become a critical challenge in academia. This paper evaluates the discriminative capability of interpretable stylometric features extracted solelyfrom submitted text. Using data from 90 participants who wrote both independently and with ChatGPT assistance, we evaluate eight machine learning classifiers while keeping data from the same participant together during validation. On the held-out test set, Random Forest achieved an ROC-AUC of0.870 and an F1-score of 0.842, with False Positive and False Negative rates of 22.2% and 11.1%, respectively. SHAP analysis shows that lexical and grammatical characteristics drive the resulting predictions. The findings suggest that transparent, text-intrinsic features provide measurable signal for detecting GPT-assisted writing.

Keywords: AI text detection, stylometry, interpretable machine learning

## 1. Introduction

Large Language Models (LLMs) have changed how students produce written assignments. While they can provide personalized tutoring, on-demand explanations, brainstorming assistance, and writing support, they also raise concerns regarding authorship and academic integrity (Kundu et al., 2024; Mehta et al., 2026; Roh et al., 2025; Sabzalieva & Valentini, 2023). Recent evidence indicates that student AI use in education has become routine rather than exceptional. In the Lumina Foundation–Gallup 2026 State of Higher Education Study, 57% of currently enrolled U.S. college students reported using AI in their coursework at least weekly, including about one in five who reported using it daily. More than half reported using AI daily or weekly to edit or improve their writing, and 36% reported using it daily or weekly to write papers (Gallup, 2026). As GPT-assisted writing becomes increasingly integrated into educational workflows, institutions require reliable methods for distinguishing independently authored and GPT-assisted student work.

This need has led to rapid growth in AI text detection research. Existing approaches include statistical detection and visualization tools such as GLTR (Gehrmann et al., 2019), zero-shot detection methods such as DetectGPT (Mitchell et al., 2023), watermarking approaches that embed detectable signals during text generation (Kirchenbauer et al., 2023), adversarially trained detectors such as RADAR (Hu et al., 2023), and fine-grained detection systems including LLM-DetectAIve (Abassy et al., 2024). Recent studies have also explored behavioral approaches based on keystroke dynamics (Kundu et al., 2024; Mehta et al., 2026; Roh et al., 2025). Although these techniques have reported reasonable performance, recent studies have questioned detector robustness under recursive paraphrasing (Sadasivan et al., 2023) and stylistic or text-complexity variation (Doughman et al., 2024).

A key challenge is detecting GPT-assisted writing while also providing an additional layer of evidence that instructors can examine during academic integrity investigations. Misconduct decisions may affect grades and disciplinary actions. Such decisions require explanations that instructors and review committees can examine, not opaque prediction scores from proprietary models whose internal evidence cannot be inspected. The explainable artificial intelligence literature frames explanations as a way to make model behavior understandable to humans, examine predictive models, and assess properties such as reliability, fairness, usability, and trust (Biecek & Burzykowski, 2021; Doshi-Velez & Kim, 2017; Ribeiro et al., 2016). Detection therefore needs approaches that characterize GPT-assisted writing through observable linguistic evidence rather than latent neural representations.

Stylometry provides a natural foundation for such an approach. Rather than modeling semantic content, stylometric analysis measures writing through characteristics such as lexical diversity, vocabulary richness, grammatical composition, and sentence structure. Stylometry has a long history in authorship attribution, including the work of Mosteller et al. (1964) and subsequent extensions through stylistic distance measures (Burrows, 2002), authorship verification methods (Koppel & Schler, 2004), and function-word classification approaches (Koppel et al., 2002). Computational linguistics and stylometry have also studied measures of lexical richness and vocabulary diversity (McCarthy & Jarvis, 2010; Tweedie & Baayen, 1998).

Despite extensive research on AI text detection, prior work has given relatively little attention to whether a compact set of interpretable stylometric features extracted from the final submitted text alone could provide sufficient evidence for detection. This study investigates that question, focusing on text-intrinsic evidence that can be applied retrospectively without access to the writing process. The following are the main contributions:

• We investigate whether a set of interpretable stylometric features extracted from submitted text can distinguish GPT-assisted writing from independently authored student writing.

• We propose a sliding window-based protocol (250-word windows with a 125-word stride) to reduce the impact of document length on the extracted features, along with disjoint sets of users across the training and testing partitions to prevent data leakage and evaluate performance on unseen users.

• We systematically evaluate eight machine learning classifiers using ROC-AUC, F1, False Positive, and False Negative rates. Window-level probabilities are aggregated into document-level decisions using a pre-specified median-probability rule.

• We examine the interpretability of Random Forest, which ranked first during training-data validation, using SHAP analysis to identify the stylometric features that contribute most strongly to its predictions.

The remainder of this paper is organized as follows. Section 2 reviews prior research on AI-generated text detection and stylometric analysis. Section 3 describes the dataset, stylometric feature extraction process, machine learning models, and experimental methodology. Section 4 presents the empirical results, statistical analyses, and SHAP-based interpretation of Random Forest. Section 5 discusses ethical considerations and limitations. Finally, Section 6 summarizes the main findings and outlines directions for future research.

## 2. Related Work

Existing detection approaches primarily infer authorship from statistical regularities learned by language models or introduced during text generation. DetectGPT, for example, exploits the observation that machine-generated text tends to occupy regions of negative probability curvature, enabling zero-shot detection without additional training (Mitchell et al., 2023). Watermarking methods alter the generation process by embedding detectable signatures into token selection (Kirchenbauer et al., 2023). Although these approaches offer reasonable evidence, they rely on proprietary models, access to generation probabilities, or high-dimensional neural representations that often give limited insight into the evidence supporting individual predictions. Recent studies have further reported reduced robustness under recursive paraphrasing (Sadasivan et al., 2023) and stylistic or text-complexity variation (Doughman et al., 2024), as well as systematic biases against non-native English writers (Liang et al., 2023), which raise concerns about detector use in high-stakes educational settings.

Parallel research has shifted focus from the submitted document to the writing process. Keystroke dynamics captures observable behavioral signals, including typing speed, pauses, revisions, and temporal writing patterns, that can differ between independently authored and GPT-assisted writing (Kundu et al., 2024; Mehta et al., 2026; Roh et al., 2025). However, behavioral approaches require instrumented writing environments and therefore do not apply retrospectively to submitted assignments or existing document collections.

Stylometry offers a complementary text-intrinsic perspective based on the hypothesis that writers exhibit lexical and grammatical preferences that remain sufficiently consistent to characterize writing style. Authorship studies show that surface-level lexical and syntactic features carry enough signal to distinguish writers (Burrows, 2002; Koppel & Schler, 2004; Koppel et al., 2002; Mosteller et al., 1964). Lexical diversity measures and vocabulary richness metrics capture aspects of writing style while remaining directly observable and statistically interpretable (McCarthy & Jarvis, 2010; Tweedie & Baayen, 1998). We also include entropy as an information-theoretic summary of token-frequency dispersion (Shannon, 1948). Unlike proprietary prediction scores or latent neural embeddings, these features correspond to explicit linguistic characteristics that can be independently examined by educators and academic integrity investigators. While GPT can imitate human writing style, detectable differences may remain (Jemama & Kumar, 2025). Another closely related study Safi (2025) used a transparent classification scheme based on word choices and function words to distinguish student responses from ChatGPT responses, and found that lightly edited or style-mimicking AI text remained detectable. We extend this line of work with evaluation on unseen participants, SHAP-based feature-level explanations, and explicit confidence intervals on error rates.

![](images/25c36dba51581461fac54e6f946f304cb26d9179ae6216a9f756d16e3a1fc0d8.jpg)  
Figure 1: Overview of the proposed detection framework. Stylometric features are extracted from fixed-length sliding windows, with disjoint sets of users used for training and testing. Hyperparameters are selected through cross-validation on the training set, and window-level predictions are aggregated using the median to obtain the final document-level classification.

## 3. Methodology

Figure 1 describes the detection framework we implemented and evaluated. We first extract the features listed in Table 1 using a sliding window-based protocol. We then create two disjoint sets of users, one for training the detectors and the other for evaluating their performance. Each detector produces a probability score for each window (sample), and the window-level probabilities are aggregated using a median-probability rule to obtain a document-level score. The document-level scores are then used to classify each document as human-authored or GPT-assisted. The performance of each detector on the test data is evaluated using F1, ROC-AUC, False Positive Rate, and False Negative Rate. Finally, we use SHAP to examine the stylometric features contributing to the predictions of Random Forest.

## 3.1. Dataset

We use the IIITD-BU (Paraphrased) dataset described by Mehta et al. (2026). The dataset was collected to examine student writing under independent and GPT-assisted conditions using keystroke dynamics (Mehta et al., 2026). Participants were primarily students enrolled in a course on Large Language Models at IIIT Delhi, India. They responded to two questions about large language models: one focused on explaining how LLMs work and their applications, while the other asked participants to analyze their strengths and weaknesses in educational settings and propose an improvement.

Participants completed the tasks in two sessions. In the first session, they answered the questions independently using their own knowledge and reasoning. In the second session, they were allowed to use LLMs such as ChatGPT for assistance but were explicitly instructed to paraphrase the generated content before typing their responses. Copy-and-paste functionality was disabled, as were auto-correct, grammar tools, and browser extensions. Thus, in this study, “GPT-assisted” refers specifically to responses written by paraphrasing GPT-generated content rather than to the broader range of possible GPT-assisted writing practices.

Table 1: Stylometric representation used for GPT-assisted writing detection. Each writing sample is represented by nine interpretable lexical, syntactic, and sentence-level features that quantify vocabulary diversity, vocabulary richness, grammatical composition, and sentence-length variation.
<table><tr><td>Feature</td><td>Name</td><td>Computation</td><td>Description</td></tr><tr><td>TTR</td><td>Type-token ratio</td><td>|V|/N</td><td>Proportion of unique tokens.</td></tr><tr><td>Hapax Ratio</td><td>Hapax legomena ratio</td><td> $\dot { H } / \dot { N }$ </td><td>Proportion of tokens occurring exactly once.</td></tr><tr><td>Word</td><td></td><td> $- \sum _ { i = 1 } ^ { | V | } p _ { i } \log _ { 2 } ( p _ { i } )$ </td><td>Shannon entropy of the token frequency distribution.</td></tr><tr><td>Entropy</td><td>Word entropy</td><td></td><td></td></tr><tr><td>NSR NOUN</td><td>Non-stopword ratio Noun ratio</td><td> $S / N$   $\# N O U N / N _ { P O S }$ </td><td>Proportion of non-stopword tokens. Proportion of tokens tagged as nouns.</td></tr><tr><td>Ratio VERB</td><td></td><td></td><td></td></tr><tr><td>Ratio ADJ</td><td>Verb ratio</td><td> $\# V E R B / N _ { P O S }$ </td><td>Proportion of tokens tagged as verbs.</td></tr><tr><td>Ratio ADV</td><td>Adjective ratio</td><td> $\# A D J / N _ { P O S }$ </td><td>Proportion of tokens tagged as adjectives.</td></tr><tr><td>Ratio</td><td>Adverb ratio</td><td> $\# A D V / N _ { P O S }$ </td><td>Proportion of tokens tagged as adverbs.</td></tr><tr><td>SLV</td><td>Sentence-length variability</td><td> $\sqrt { \frac { \sum _ { i = 1 } ^ { m } ( L _ { i } - \bar { L } ) ^ { 2 } } { m - 1 } }$ </td><td>Standard deviation of sentence lengths.</td></tr></table>

N = total number of tokens; |V| = number of unique tokens; H = number of hapax legomena; p = relative frequency of token i; S = number of non-stopword tokens; $N _ { P O S } = \mathrm { t o t a l }$ POS-tagged tokens; m = number of sentences; $L _ { i }$ = length of sentence i in tokens; L<sup>¯</sup> = mean sentence length.

Although the original study collected keystroke dynamics and interaction traces, we focus exclusively on the final submitted text. We make this choice deliberately because our objective is to determine whether interpretable stylometric features extracted directly from submitted text contain sufficient signal to distinguish the two writing conditions without behavioral information, which may be unavailable in retrospective academic integrity investigations.

Each participant completed two prompts in each session. We concatenate the two responses from the same session to form a single response for each participant under each writing condition. The resulting dataset contains 180 writing samples from 90 participants, with each participant contributing one independently authored response and one GPT-assisted response. The dataset is therefore balanced across the two writing conditions.

Independently authored responses contained an average of 620 whitespace-delimited words (SD = 198), compared with 515 words (SD = 155) for GPT-assisted responses. The corresponding mean sentence counts were 26.5 and 24.7, while mean sentence lengths were 27.2 and 27.3 words, respectively. Because differences in document length could influence some stylometric features, particularly lexical-diversity measures, Section 4 explicitly examines these differences and whether document length alone can explain the observed classification performance.

Because every participant appears in both conditions, a naive random split could place a participant’s independently authored and ChatGPT-assisted texts on opposite sides of the split, allowing author-specific writing style to leak between training and evaluation and potentially inflating performance. To prevent this, we use disjoint sets of users for training and evaluation, with all texts from a given participant kept in the same set. The 90 participants were partitioned into 72 training participants (144 samples) and 18 held-out participants (36 samples: 18 independently authored and 18 ChatGPT-assisted), with no participant shared between the two sets.

## 3.2. Preprocessing and feature engineering

We extracted the final text from each response and removed residual keyboard-event tokens from the original keystroke-logging instrumentation, including Control and CapsLock. We split each cleaned document into whitespace-delimited words and generated windows using a pre-specified length of 250 words and a stride of 125 words, resulting in 50% overlap between adjacent windows. We right-aligned the final window to the end of the document so that no text was discarded, and no artificial padding was added. Documents shorter than 250 words contributed one full-document window.

This procedure produced 722 windows, including 397 independently authored windows and 325 GPT-assisted windows. The 72 training participants contributed 595 windows, while the 18 held-out participants contributed 127 windows from 36 documents, including 74 independently authored and 53 GPT-assisted windows. Windowing increases the number of text segments available to the classifiers, but these windows are not additional independent participants. Therefore, we kept all windows from a given participant within the same training, validation, or test set throughout the analysis.

We tokenized each window using quanteda with punctuation removed and obtained part-of-speech annotations using the udpipe English EWT model (english-ewt-ud-2.5). The Non-stopword Ratio used the 175-word Snowball English stop list. Stop-list matching was case-folded, while Type-Token Ratio, Hapax Ratio, and Word Entropy preserved case. We segmented sentences at sentence-final periods for Sentence-Length Variability. We computed the nine features independently for each window and centered and scaled them using parameters estimated from the training data only. All analyses used R 4.4.3 with fastml (0.7.8), quanteda (4.3.1), and udpipe (0.8.16). A fixed random seed (123) was used for the user split, cross-validation folds, and Bayesian tuning.

Each window received a predicted probability of GPT assistance. We aggregated these probabilities using the median to obtain a document-level score. Using a pre-specified threshold of 0.50, we classified a document as GPT-assisted when its median probability was at least 0.50. The median reduces the influence of an unusually high or low window while retaining the probability information that would be lost with hard majority voting.

## 3.3. Feature analysis

We analyzed the nine stylometric features to examine how they differ between independently authored and GPT-assisted writing. Because overlapping windows from the same document are not statistically independent, for this analysis we first summarized each feature by its median across the windows of each document. We then used paired t-tests to compare each feature between the two writing conditions across the 90 participants. We also computed point-biserial correlations (r) to measure the association between each feature and the writing condition.

Table 2: Mean document-level median feature values for independently authored and GPT-assisted writing across 90 participants. Paired t-tests compare the two writing conditions, and positive r indicates higher values in GPT-assisted writing.
<table><tr><td>Feature</td><td>Human</td><td>AI</td><td>t</td><td>p</td><td>r</td></tr><tr><td>TTR</td><td>0.60</td><td>0.65</td><td>6.66</td><td>.001</td><td>0.43</td></tr><tr><td>Hapax Ratio</td><td>0.44</td><td>0.51</td><td>7.19</td><td>.001</td><td>0.46</td></tr><tr><td>Word Entropy</td><td>6.74</td><td>6.94</td><td>7.98</td><td>.001</td><td>0.42</td></tr><tr><td>SLV</td><td>14.67</td><td>13.07</td><td>-2.22</td><td>.03</td><td>-0.13</td></tr><tr><td>NSR</td><td>0.59</td><td>0.64</td><td>8.80</td><td>.001</td><td>0.50</td></tr><tr><td>NOUN Ratio</td><td>0.24</td><td>0.27</td><td>8.37</td><td>.001</td><td>0.47</td></tr><tr><td>VERB Ratio</td><td>0.13</td><td>0.14</td><td>2.52</td><td>.01</td><td>0.18</td></tr><tr><td>ADJ Ratio</td><td>0.08</td><td>0.08</td><td>2.09</td><td>.04</td><td>0.15</td></tr><tr><td>ADV Ratio</td><td>0.05</td><td>0.04</td><td>-6.21</td><td>.001</td><td>-0.34</td></tr></table>

## 3.4. Classifier training and validation

We evaluated Logistic Regression, Linear Discriminant Analysis, Quadratic Discriminant Analysis, Naive Bayes, Random Forest, linear Support Vector Machine, k-Nearest Neighbors, and a multilayer perceptron. These classifiers were selected because they represent different learning paradigms, including linear and discriminant models, probabilistic methods, instance-based learning, ensemble learning, margin-based classification, and neural networks. They are also commonly used in text classification and related stylometric classification tasks. Evaluating this diverse set of classifiers allows us to examine whether the proposed stylometric features are useful across different modeling approaches rather than being specific to a single classifier.

Hyperparameter tuning was performed using only the training partition through Bayesian optimization with 30 iterations over the default fastml search space for each classifier. During validation, all windows and both writing conditions from a given participant were kept in the same fold. We compared the eight classifiers using mean validation ROC-AUC under repeated 10-fold validation and an additional 5-by-5 nested-validation check. Random Forest ranked first under both validation procedures. The final Random Forest used three candidate features at each split, 500 trees, and a minimum node size of 10.

After tuning, each classifier was fit to the complete training partition and evaluated once on the held-out test set. The held-out test set was not used for hyperparameter tuning or model selection.

## 3.5. Evaluation metrics

We treat GPT-assisted writing as the positive class and independently authored writing as the negative class. We use ROC-AUC and F1-score as the principal performance measures. ROC-AUC measures how well the document-level probabilities rank GPT-assisted texts above independently authored texts across different thresholds. F1 summarizes the balance between precision and recall for detecting GPT-assisted writing.

We also report the False Positive Rate (FPR) and False Negative Rate (FNR). A false positive occurs when independently authored writing is incorrectly classified as GPT-assisted, while a false negative occurs when GPT-assisted writing is incorrectly classified as independently authored. We report both errors as counts, rates, and 95% Wilson confidence intervals.

## 3.6. Interpretability framework

We used SHapley Additive exPlanations (SHAP) to examine how each feature contributes to the Random Forest predictions. Random Forest was selected for the interpretability analysis based on its performance during training-data validation rather than its performance on the held-out test set. Global SHAP importance summarizes the mean absolute contribution of each feature across the held-out windows. The beeswarm plot shows the direction and magnitude of each window-level contribution together with the corresponding feature value.

Because the final document score is the median of its window probabilities, for the local examples we explain the window with a probability closest to the document median. These examples therefore show how the features contribute to the prediction for a representative window of the document rather than providing an explanation of the entire multi-window aggregation.

## 4. Results and discussion

## 4.1. Detection performance

Figure 2 shows the cross-validation results used to compare the eight classifiers on the training data. Random Forest achieved the highest mean validation ROC-AUC (0.838), followed closely by Naive Bayes, QDA, LDA, Logistic Regression, and linear SVM. The error bars show substantial overlap, indicating similar validation performance among these classifiers. Random Forest also ranked first in the additional nested-validation check and was therefore selected for the subsequent interpretability analysis.

![](images/e552f9ad8894ce3942fb241bbab9139337852246ce64036cd6c371443962775f.jpg)  
Figure 2: Comparison of the eight classifiers using repeated 10-fold cross-validation on the training data, with all data from the same participant kept in the same fold. Points show mean validation ROC-AUC, and error bars show one standard deviation.

The classifier comparison and hyperparameter tuning used only the training data, and the held-out test set played no role in these decisions. Final evaluation was performed on 36 documents from 18 held-out participants, using the median of the window-level probabilities to obtain a score for each document. Random Forest achieved a holdout ROC-AUC of 0.870 (95% CI: 0.747–0.981) and an F1-score of 0.842 (95% CI: 0.737–0.944). Confidence intervals were estimated using participant-level bootstrap resampling.

Table 3: Random Forest performance on 36 held-out documents from 18 unseen participants. GPT-assisted writing is treated as the positive class.
<table><tr><td>Measure</td><td>Estimate</td><td>Count</td><td>95% CI</td></tr><tr><td>ROC-AUC</td><td>0.870</td><td>一</td><td>0.747-0.981</td></tr><tr><td>F1</td><td>0.842</td><td>一</td><td>0.737–0.944</td></tr><tr><td>False Positive</td><td>22.2%</td><td>4/18</td><td>9.0–45.2%</td></tr><tr><td>False Negative</td><td>11.1%</td><td>2/18</td><td>3.1–32.8%</td></tr></table>

Table 4: Document-level confusion matrix for Random Forest on the 36 held-out documents using the median window-probability rule.
<table><tr><td></td><td>Predicted Human Predicted GPT</td><td></td></tr><tr><td>Actual Human</td><td>14</td><td>4</td></tr><tr><td>Actual GPT</td><td>2</td><td>16</td></tr></table>

With GPT-assisted writing treated as the positive class, four independently authored documents were incorrectly classified as GPT-assisted, resulting in a

False Positive Rate of $4 / 1 8 ~ = ~ 2 2 . 2 \%$ (95% Wilson CI: 9.0%–45.2%). Two GPT-assisted documents were incorrectly classified as independently authored, resulting in a False Negative Rate of $2 / 1 8 \ = \ 1 1 . 1 \%$ (95% Wilson CI: 3.1%–32.8%). The false positives are particularly important in an academic integrity setting because they represent independently authored work incorrectly flagged as GPT-assisted. Both confidence intervals are wide, reflecting the small held-out sample, and these results do not support using the model as standalone evidence for academic misconduct.

As a post-hoc sensitivity analysis, we conducted additional experiments using five train-test ratios ranging from 60/40 to 80/20. As shown in Table 5, the mean False Positive Rate ranged from 12.4% to 15.8%, while the mean False Negative Rate ranged from 23.7% to 24.8%. Although the mean error rates remained relatively stable across the evaluated train-test ratios, the class-specific estimates differed from those obtained on the original held-out test set. This variation highlights the uncertainty associated with evaluation on a single small test set.

Table 5: Sensitivity of error rates to train-test partitioning. Values are means with 2.5th–97.5th percentile ranges across splits.
<table><tr><td>Train/Test</td><td>FPR</td><td>FNR</td></tr><tr><td>60/40</td><td>.158 [.083–.244]</td><td>.243 [.139–.389]</td></tr><tr><td>65/35</td><td>.152 [.062–.274]</td><td>.242 [.125–.399]</td></tr><tr><td>70/30</td><td>.142 [.037–.288]</td><td>.237 [.082–.370]</td></tr><tr><td>75/25</td><td>.124 [.045–.262]</td><td>.248 [.091–.409]</td></tr><tr><td>80/20</td><td>.143 [.000–.278]</td><td>.244 [.056–.487]</td></tr></table>

## 4.2. Error analysis

The median-probability rule misclassified six of the 36 held-out documents. Four independently authored documents received median GPT-assisted probabilities ranging from 0.636 to 0.920 and were therefore false positives. Two GPT-assisted documents were false negatives. One was close to the decision threshold, with a median GPT-assisted probability of 0.495, while the other received a substantially lower probability of 0.151. These errors show that the median aggregation reduces the influence of unusually high or low window probabilities but does not eliminate overlap between independently authored and GPT-assisted writing. The results further support using the model as a decision-support tool requiring contextual review rather than as standalone evidence of academic misconduct.

## 4.3. Length and feature controls

Our sliding-window design standardizes the amount of text used to compute the stylometric features while also producing multiple samples from each document. Each document was divided into 250-word windows with a 125-word stride, and the nine features were computed separately for each window. Thus, for most samples, features such as Type-Token Ratio and Hapax Ratio are computed from the same number of words rather than from complete documents of different lengths. Of the 722 windows, 508 contained 250 words. The remaining 214 came from documents shorter than 250 words and were retained without padding.

We nevertheless examined document length because the original documents differed between the two writing conditions. Independently authored documents contained more words than GPT-assisted documents (paired $t ( 8 9 ) ~ = ~ 6 . 4 6 , ~ p ~ < ~ . 0 0 1$ ; Welch $p \ < \ . 0 0 1 )$ with independently authored documents being longer in 72 of the 90 pairs. In contrast, sentence count and mean sentence length did not differ significantly between the two conditions (paired $p = . 0 5 1$ and $p =$ .97, respectively).

As an additional control, a classifier using only the total document word count achieved a cross-validation ROC-AUC of 0.68, showing that document length itself contains some information about the writing condition. However, total document word count is not used as a feature in our main classifier, and the fixed-word windowing design largely prevents the stylometric features from directly capturing differences in overall document length. Replacing Type-Token Ratio and Hapax Ratio with moving-average TTR produced an ROC-AUC close to that of the full model. Feature-group ablations further showed that lexical features carried most of the discriminative signal, followed by part-of-speech features, while Sentence-Length Variability alone performed near chance.

## 4.4. Feature differences

Table 2 shows significant differences in several stylometric features between independently authored and GPT-assisted writing. GPT-assisted documents have higher TTR, Hapax Ratio, Word Entropy, NSR, Noun Ratio, Verb Ratio, and Adjective Ratio, while independently authored documents have higher Adverb Ratio and Sentence-Length Variability. NSR and Noun Ratio show the strongest positive associations with GPT-assisted writing, while Adverb Ratio shows the strongest negative association. For these statistical comparisons, each feature was summarized by its median across the windows of each document, preventing overlapping windows from inflating the inferential sample size.

## 4.5. SHAP analysis

Figure 3 shows the global Random Forest SHAP importance. Hapax Ratio is the most influential feature, followed by NSR, Noun Ratio, Adverb Ratio, and TTR. Verb Ratio, Sentence-Length Variability, and Word Entropy have smaller contributions, while Adjective Ratio contributes least. Figure 4 shows that the direction and magnitude of these contributions vary across windows, reflecting the non-linear relationships learned by Random Forest.

![](images/9a0c1954cdb2cbe1b2c354e1f7898fd7fd9b39e2e1c853d573eaee1d5d0269c4.jpg)  
Figure 3: Global Random Forest SHAP importance across held-out windows. Higher mean absolute SHAP values indicate greater influence on the predicted probability of GPT-assisted writing.

For the GPT-assisted example, the selected window has a predicted probability of 0.947, with NSR and Hapax Ratio making the largest positive contributions. For the independently authored example, the selected window has a predicted probability of 0.077, with Hapax Ratio, Noun Ratio, NSR, and TTR shifting the prediction toward independently authored writing. These examples show how observable stylometric features contribute to individual window predictions. They do not explain the entire multi-window aggregation or establish that a prediction is valid evidence of misconduct.

## 5. Ethical considerations and limitations

GPT-assisted writing detection may influence grades and disciplinary actions. Detection systems should therefore be used as decision-support tools rather than definitive evidence of misconduct. A prediction should initiate contextual review and should not replace an instructor’s assessment, student testimony, or institutional due process.

![](images/09b20aa619293593eeae2797e48d5a39a88a0b078426916cc1501da1cd244cba.jpg)  
Figure 4: Random Forest SHAP summary across held-out windows. Horizontal position shows the contribution to the predicted probability of GPT-assisted writing, and color indicates the corresponding feature value.

![](images/67b32e88e9c8a3eae70ceae2f8a56f8c263a580b7be571be689e9c2bf2f8dfd5.jpg)  
Figure 5: SHAP explanation for a representative window of a correctly classified GPT-assisted document. The selected window has a predicted probability closest to the document median.

![](images/640856bec58d04addfa8b02a2c63655911032cc4159586c850444ff62bc4743b.jpg)  
Figure 6: SHAP explanation for a representative window of a correctly classified independently authored document. The selected window has a predicted probability closest to the document median.

False positives are particularly important because they represent independently authored work incorrectly classified as GPT-assisted. In our held-out evaluation, the False Positive Rate was 22.2% (4/18), with a 95% confidence interval of 9.0%–45.2%. The wide interval reflects the small test set and does not support automated use. SHAP can explain which features contributed to a prediction, but it does not establish that the prediction is reliable or fair.

Several limitations bound our findings. The study uses 90 participants from one course, institution, topic area, and GPT-assistance setting. In this dataset, participants used GPT-generated content and were instructed to paraphrase it before typing their responses. Other forms of AI assistance, such as brainstorming or editing, were not evaluated. The results also do not establish performance across other assignments, disciplines, institutions, languages, student populations, or LLMs. Because all participants completed the independent session first, the writing condition may also be affected by session order. External validation is therefore necessary.

Finally, we did not evaluate performance across demographic or language background groups and therefore cannot draw conclusions about fairness across these groups. Writing styles and GPT outputs may also change over time, and the stylometric features may be deliberately altered. Evaluation on new populations and writing conditions is needed before institutional use.

## 6. Conclusion and future work

This study evaluated an interpretable stylometric approach for distinguishing GPT-assisted and independently authored student writing. Documents were divided into overlapping 250-word windows, and all data from the same participant were kept in the same training, validation, or test set. Eight classifiers were compared using the training data, with Random Forest achieving the highest validation ROC-AUC. Window-level probabilities were then combined using their median to obtain the final document-level prediction.

On 36 held-out documents, Random Forest achieved an ROC-AUC of 0.870 and an F1-score of 0.842. Four independently authored documents were incorrectly classified as GPT-assisted, while two GPT-assisted documents were incorrectly classified as independently authored. SHAP identified Hapax Ratio, NSR, Noun Ratio, Adverb Ratio, and TTR as the most influential features. These results show that interpretable stylometric features provide useful signal for distinguishing the two writing conditions. However, the small test set, observed errors, and wide confidence intervals do not support using the model as standalone evidence of academic misconduct.

Future work should evaluate the approach on other institutions, writing tasks, student populations, and forms of AI assistance. It should also examine performance across different groups and determine how much text is needed for reliable classification. Future studies can also evaluate other ways of combining window predictions and combine stylometric features with behavioral information when writing-process data are available.

## References

Abassy, M., Elozeiri, K., Aziz, A., et al. (2024). LLM-DetectAive: A tool for fine-grained machine-generated text detection. Proceedings of EMNLP 2024 Demo Track. https://doi.org/ 10.18653/V1/2024.EMNLP-DEMO.35

Biecek, P., & Burzykowski, T. (2021). Explanatory model analysis: Explore, explain and examine predictive models. CRC Press.

Burrows, J. (2002). ∆: A measure of stylistic difference and a guide to likely authorship. Literary and Linguistic Computing, 17(3), 267–287.

Doshi-Velez, F., & Kim, B. (2017). Towards a rigorous science of interpretable machine learning. arXiv preprint arXiv:1702.08608.

Doughman, J., Afzal, O. M., Toyin, H. O., Shehata, S., Nakov, P., & Talat, Z. (2024). Exploring the limitations of detecting machine-generated text. Proceedings ofCOLING 2025. https://doi. org/10.48550/arXiv.2406.11073

Gallup. (2026, April). AI Is Routine for College Students, Despite Campus Limits [Accessed: 2026-06-11]. https : / / news . gallup . com / poll / 704090 / routine - college - students - despite - campus-limits.aspx

Gehrmann, S., Strobelt, H., & Rush, A. M. (2019). GLTR: Statistical detection and visualization of generated text. Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics: System Demonstrations, 111–116. https://doi.org/10. 18653/v1/P19-3019

Hu, X., Chen, P.-Y., & Ho, T.-Y. (2023). RADAR: Robust AI-text detection via adversarial learning. Advances in Neural Information Processing Systems 36 (NeurIPS 2023), 36, 15077–15095. https : / / doi . org / 10 . 52202 / 075280-0662

Jemama, R., & Kumar, R. (2025). How well do llms imitate human writing style? 2025 IEEE 16th Annual Ubiquitous Computing, Electronics & Mobile Communication Conference (UEMCON). https : / / doi . org / 10 . 1109 / UEMCON67449.2025.11267719

Kirchenbauer, J., Geiping, J., Wen, Y., Katz, J., Miers, I., & Goldstein, T. (2023). A watermark for large language models. Proceedings of the 40th International Conference on Machine Learning, 202, 17061–17084.

Koppel, M., Argamon, S., & Shimoni, A. R. (2002). Automatically categorizing written texts by author gender. Literary and Linguistic Computing, 17(4), 401–412.

Koppel, M., & Schler, J. (2004). Authorship verification as a one-class classification problem. Proceedings ofICML 2004, 489–495.

Kundu, D., Mehta, A., Kumar, R., Lal, N., Anand, A., Singh, A., & Shah, R. R. (2024). Keystroke dynamics against academic dishonesty in the age of llms. 2024 IEEE International Joint Conference on Biometrics (IJCB), 1–10. https: //doi.org/10.1109/IJCB62174.2024.10744461

Liang, W., Yuksekgonul, M., Mao, Y., Wu, E., & Zou, J. (2023). GPT detectors are biased against non-native English writers. Patterns,

4(7), 100779. https://doi.org/10.1016/j.patter. 2023.100779

McCarthy, P. M., & Jarvis, S. (2010). MTLD, vocd-D, and HD-D: A validation study of sophisticated approaches to lexical diversity assessment. Behavior Research Methods, 42(2), 381–392.

Mehta, A., Kumar, R., Singla, A., Bisht, K., Singla, Y. K., & Shah, R. R. (2026). Detecting llm-assisted academic dishonesty using keystroke dynamics. IEEE Transactions on Biometrics, Behavior, and Identity Science, 1–1. https://doi.org/10.1109/TBIOM.2026. 3683979

Mitchell, E., Lee, Y., Khazatsky, A., Manning, C. D., & Finn, C. (2023). DetectGPT: Zero-shot machine-generated text detection using probability curvature. Proceedings of the 40th International Conference on Machine Learning, 202, 24950–24962.

Mosteller, F., Wallace, D. L., & Nerbonne, J. A. (1964). Inference and disputed authorship: The Federalist. Addison-Wesley.

Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). “why should I trust you?”: Explaining the predictions of any classifier. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 1135–1144.

Roh, D. H., Kumar, R., & Ngo, A. (2025). Llm-assisted cheating detection in korean language via keystrokes. 2025 IEEE IJCB. https://doi.org/ 10.1109/IJCB65343.2025.11411145

Sabzalieva, E., & Valentini, A. (2023). Chatgpt and artificial intelligence in higher education: Quick start guide. UNESCO.

Sadasivan, V. S., Kumar, A., Balachandran, S., Wang, W., & Feizi, S. (2023). Can AI-generated text be reliably detected? arXiv preprint arXiv:2303.11156.

Safi, R. (2025). Detecting plagiarism in the age of generative AI: An exploratory experiment. Communications of the Association for Information Systems, 56, 594–612. https : //doi.org/10.17705/1CAIS.05624

Shannon, C. E. (1948). A mathematical theory of communication. Bell System Technical Journal, 27(3), 379–423.

Tweedie, F. J., & Baayen, R. H. (1998). How variable may a constant be? measures of lexical richness in perspective. Computers and the Humanities, 32(5), 323–352.