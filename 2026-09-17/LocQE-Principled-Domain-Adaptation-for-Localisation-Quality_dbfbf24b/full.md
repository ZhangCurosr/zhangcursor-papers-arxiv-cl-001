# LocQE: Principled Domain Adaptation for Localisation Quality Estimation by Leveraging Post-Edits

Kathy Hämmerl<sup>1,2,</sup>\* and Gabriel Bretschner<sup>3</sup> and Joern Wuebker<sup>3</sup> <sup>1</sup>Technical University of Munich, <sup>2</sup>Munich Center for Machine Learning, <sup>3</sup>LILT Correspondence: k.haemmerl@tum.de

## Abstract

Learned quality estimation (QE) models such as COMETKiwi are widespread and work well for general machine translation evaluation. However, they are known to struggle on unseen domains, limiting their performance in a real-world localisation context. We show that they are insensitive to some important factors in localisation, such as whether numbers are translated accurately, or even whether the correct number of spaces and punctuation are preserved in a translation. Further, a key capability for optimisation of machine translation is the ability of QE models to accurately rank different translations of a single segment, which suffers significantly from the domain transfer. In the absence of large-scale direct assessment data, we propose principled finetuning approaches to reduce the domain gap with even small amounts of post-editing data. Using a multi-task fine-tuning approach and a simple tokeniser intervention, we create a QE model which proves markedly better at distinguishing preferred post-edits from rejected initial translations in a localisation context. We show that preferences and artificial continuous scores stabilise each other, and argue that to calibrate metrics both in terms of their absolute scores and comparisons between translation of the same source, both types of signal are needed.

## 1 Introduction

Learned neural machine translation metrics are known to struggle in unseen domains (Zouhar et al., 2024). Both reference-based metrics and quality estimation models such as COMETKiwi (Rei et al., 2022), have been trained primarily on news and wiki text. In real-life commercial localisation workflows, the domains vary from spec sheets to user interface labels to legal text (e.g., Buschbeck and

![](images/46ec9755c481a4fba96991ecdfb8b79a5bf7f9d486abbb1dc8dd09404fe03406.jpg)  
Figure 1: Comparison of GEMBA<sub>ESA</sub>-QE, erence accuracy on public data (WMT24) is much higher than on real localisation data (here: internal post-edit data), or on long-tail issues important to localisation (LocCheck). Domain adaptation reduces the gap significantly, but to solve for long-tail issues, data augmentation is needed.

Exel, 2020; Lin et al., 2022). Additionally, localisation workflows require the consistent use of dedicated terminology, and following specific style guidelines.

As illustrated in Figure 1, quality estimation (QE) is highly vulnerable to this domain shift, with $\mathrm { G E M B A _ { E S A } – Q E }$ and COMETKiwi<sup>22</sup><sub>DA</sub> both unable to consistently identify the preferred translation on localisation data, while performing well on WMT data.

At the same time, QE is highly relevant for localisation workflows. Firstly, due to the nature of post-editing, any available references are frequently similar to the hypotheses, biasing reference-based metrics towards the originally used machine translation model. Secondly, effective QE is needed for finetuning workflows such as direct quality optimisation (DQO; Uhlig et al., 2025). In this context, an especially important capability is segment-level ranking of different translations of the same source, so that QE can reliably pick preferred translations.

Not only is there a marked drop in performance compared to WMT data (Figure 1), but we also empirically observe that 1) Translators and reviewers react strongly to specific issues that the models either cannot see at all—such as the incorrect use of non-breaking spaces—, or even reward, such as hallucinating a word to fill a cloze; and 2) For long numbers, incorrect translations are not consistently distinguished and caught. Examples are shown in Table 1. These are mostly long-tail issues, meaning they occur rarely in training or test data, but since human translators often consider them trivially wrong, they may erode trust if not caught.

To address the domain shift, we aim to adapt a QE model in the absence of large-scale human score data by leveraging post-edits. In addition to this adaptation training, we analyse QE model performance on the aforementioned long-tail issues, and propose adapting the input tokeniser so that models can actually “see” some of these issues.

Although other high-quality QE models have been proposed since its release, COMETKiwi<sup>22</sup><sub>DA</sub><sup>1</sup> represents a very good tradeoff between strong performance and a lightweight model. It is both cheap to run and efficient to fine-tune for quick iteration. It has also been found to outperform LLM-based metrics at segment-level ranking (Moghe et al., base model for experimentation in this work.

Our key contributions are:

1. We formulate a lightweight, data-efficient approach to domain adaptation for QE models without human direct assessments, using postedits and simple heuristics as our training signal. Tokeniser adaptation allows the model to learn specific relevant patterns.

2. We describe a new challenge set for localisation-specific issues (LocCheck), and a new human-annotated localisation dataset (LocHD).

3. We demonstrate improvements on the two newly-proposed datasets, as well as for preference accuracy on natural target-domain data.

Our code for regenerating LocCheck, and the LocHD data, are released on GitHub.<sup>2</sup>

## 2 Related Work

## 2.1 COMETKiwi models

The COMET framework was initially proposed in Rei et al. (2020). In it, multilingual encoders are fine-tuned to predict normalised human direct assessments of machine translations. The first COMET models required a reference, but later iterations expanded to QE with COMET-QE (Rei et al., 2021) and COMETKiwi (Rei et al., 2022), as well as error span prediction with xCOMET models (Guerreiro et al., 2024).

## 2.2 Domain Adaptation for Metrics

Similarly to any fine-tuned language model, the COMET models can struggle when applied to unseen domains, as Zouhar et al. (2024) show for reference-based neural models. Sharami et al. (2023) also discuss this phenomenon for quality estimation, and fine-tune a model pre-trained on outof-domain data with progressively more specific target-domain data. They produce target-domain augmented data by inferencing an MT model and scoring its hypothesis against the existing reference translation via TER (Snover et al., 2006), and evaluate their QE model on correlation with HTER (“Human-targeted Translation Edit Rate”, referring to the edit rate between a hypothesis and a human post-edit given that hypothesis). The present work instead evaluates both against preferences derived from post-edits and ESA scores (Error Span Annotation, Kocmi et al., 2024b) from professional annotators. We use two types of training signal derived from these post-edits: continuous scores and preference pairs.

Another related approach is “poor man’s quality estimation” (Zouhar et al., 2023). They suggest pre-training quality estimation with large-scale metric estimation (i.e., predicting the scores of a reference-based metric) before fine-tuning with small amounts of human labels. In the present work, we instead use metric estimation as a continuous training signal for domain adaptation.

MS-COMET (Kocmi et al., 2022) is a complete re-training of COMET and COMET-QE with much larger training corpora of 2 M and 3.5 M humanannotated segments, respectively. They cover over 110 languages and 15 domains, using only annotations from professional translators. In contrast to the original COMET models, they use raw direct assessment scores, despite the risk of scores becoming incomparable across languages. Our approach requires much less data, and we keep z-score normalisation intact.

<table><tr><td></td><td>COMETKiwi22 :22</td></tr><tr><td>These cargo pants are the easiest to dress when you are attending an outing!</td><td></td></tr><tr><td>+ Ce pantalon cargo est le plus facile à habiller lorsque vous participez à une sortie[NBSP]! — Ce pantalon cargo est le plus facile à habiller lorsque vous participez à une sortie !</td><td>84.83 84.83</td></tr><tr><td>Since they caused the problem, maybe they can do something to it too.</td><td></td></tr><tr><td>+ Nachdem sie das Problem verursacht haben, können sie vielleicht auch etwas tun, um es zu</td><td>78.89</td></tr><tr><td>— Nachdem sie das Problem verursacht haben, können sie vielleicht auch etwas tun, um es zu lösen.</td><td>84.20</td></tr><tr><td>Can you confirm that the input 7777777 has a number with seven digits?</td><td></td></tr><tr><td>+ Können Sie bestätigen, dass die Eingabe 7777777 eine Zahl mit sieben Ziffern hat?</td><td></td></tr><tr><td>— Können Sie bestätigen, dass die Eingabe 777777 eine Zahl mit sieben Ziffern hat?</td><td>85.91 85.57</td></tr></table>

Table 1: Exemplars of localisation-specific issues that COMETKiwi misses. The model cannot see non-breaking spaces (“[NBSP]”), may prefer a grammatical sentence with a hallucination over the correct translation of a cloze (“\_\_ \_”), and is not sensitive enough to errors in long numbers.

Schmidt et al. (2026) analyse system-level and segment-level accuracy of metrics across domains, showing that while human annotators exhibit independent noise around a “true” rating, metrics tend to exhibit correlated errors. They aim to disentangle domain shift from noise in human labels, and demonstrate metric biases on out-of-domain data.

## 2.3 Contrastive Training

Contrastive training has been proven effective for representation learning in the past, such as in InfoXLM (Chi et al., 2021) or mSimCSE (Wang et al., 2022). Preference tuning of LLMs has enjoyed great popularity in recent years, from RLHF (Christiano et al., 2017; Ouyang et al., 2022) and direct preference optimisation (DPO; Rafailov et al., 2023), to specialised workflows such as DQO (Uhlig et al., 2025). Like the present work, Berger et al. (2024) discuss using machine translated segments and their corresponding post-edits as preference pairs, though they apply them to DPO on LLMs for machine translation.

The original COMET paper includes a rankingbased metric version, which creates a score from the Euclidean distance of the hypothesis to both the source and reference, learned via Triplet Margin Loss (Rei et al., 2020). However, this method relies on a reference, unlike our approach, and it produces scores directly from model representations rather than using a regression head. It is also not regularised, meaning that the model tends to distribute scores unevenly. The COMET-Rank model was dropped in later iterations of COMET.

Recently, Proietti et al. (2026) proposed a relative quality estimation framework that takes in a source segment and two candidate translations, thus focusing on segment-level ranking accuracy. However, their system cannot return absolute scores, making the ratings more difficult to interpret.

The most conceptually similar work to ours is Tan and Monz (2025), who use raw human scores as inputs to induce preference pairs with a variable margin, then train using a Bradley-Terry objective. Similarly to us, they are motivated by relatively poor metric performance at a segment level. Compared to our work, they train larger models using a reference and WMT-domain data, whereas we focus on domain adaptation and do not have access to an independent reference or continuous human ratings. By leveraging post-edits into both preferences and continuous scores, using a multi-task training setup, we avoid the need for post-hoc calibration of output scores.

## 2.4 Metric Challenge Sets

ACES and SPAN-ACES (Amrhein et al., 2023; Moghe et al., 2025) are metric challenge sets intended to profile strengths and weaknesses of machine translation metrics on different subcategories of translation accuracy issues. They do not cover surface issues other than some relatively easy punctuation errors. ACES covers many language pairs, though with a very small number of examples for some. Metrics are scored on whether a “good” translation is ranked higher than a translation with the error under examination. We use a similar design for our LocCheck dataset, but focus on localisation-specific issues such as those discussed in the introduction.

## 3 Datasets

In this section, we summarise the datasets used in this study. Table 2 lists the dataset sizes.

<table><tr><td>Corpus</td><td>train</td><td>dev</td><td>test</td></tr><tr><td>ACED corpus</td><td>17,455</td><td>7,884</td><td>9,922</td></tr><tr><td>ACED (prefs)</td><td>2,179</td><td>1,056</td><td>959</td></tr><tr><td>Internal</td><td>10,000</td><td>1,000</td><td>Table 11</td></tr><tr><td>LocHD</td><td>1</td><td></td><td>3,000</td></tr><tr><td>WMT24 (prefs)</td><td>一</td><td></td><td>5,718</td></tr><tr><td>ACES</td><td>一</td><td></td><td>36,476</td></tr></table>

Table 2: Dataset sizes, in number of rows, used in our experiments. For details on the per-language distribution and number of test segments we create from internal data, see Appendix Table 11.

Internal Localisation Data. Internally, we have access to a large collection of localisation-specific data across various client domains and language pairs, of which a large percentage has been postedited by humans. The data includes, for example, technical documentation, user interface text, lists of features, and other specialised content. There is a history of edits for each segment, but there are no at-scale direct assessments.

In order to adapt QE models to the localisation domain, we process the post-edits in two different ways, producing both artificial continuous scores and preference pairs. To obtain preference pairs, we take machine translated segments as the negative sample, and the post-edited version provided by a human reviewer as the positive sample, provided an edit was made. We have no independent reference in this setting. To create artificial scores, we treat the human-edited segment as a reference and use a surface metric (chrF++) to provide a score for the machine-translated segment. Additionally, we use heuristics to identify major errors that resulted in small chrF changes, and reduce the score in such cases. We then apply z-score normalisation, treating each client project as one annotator, and rescale the scores to the range [0,1]. Further data processing details can be found in Appendix A. Additionally, in Appendix A.4, we describe a small validation study for both types of internal data.

Despite the potential subjectivity of post-edits (cf. Popovic´, 2021), they are the best humancreated quality signal we have. Therefore, we use preference accuracy as the main evaluation metric on this data.

ACED Localisation Data. The ACED corpus is a public collection of English-to-German localisation data, initially for Translation Error Correction (Lin et al., 2022). For each segment, it includes a source, target, and perturbed target, where the perturbed target represents the initial hypothesis and the target is the post-edited segment. The data is spread across three sub-datasets with different domains: ASICS (marketing copy for an activewear company), EMERSON (industrial product listings for a manufacturer), and DIGITALOCEAN (software engineering tutorials).

We use the predefined train, dev, and test splits, but merge them across sub-datasets for a larger data size. We apply the same transformations as for our internal data to obtain preference pairs and continuous scores. Note that the number of preference pairs is especially limited, since not all segments in ACED received a post-edit. Table 2 lists the merged data sizes for ACED.

Localisation Data with Human Scores (LocHD). For evaluation purposes, we construct a Localisation Human Data corpus. We translate 300 segments from English into ten target languages<sup>3</sup> using a selection of different commercial models, then collect Error Span Annotations (Kocmi et al., 2024b) from professional translators, who were paid standard rates. We refer to this corpus as LocHD from here on.

The source data originates from a real client project, and includes primarily user interface text and help documentation. Since this data does not contain personally identifiable information in the translation segments, we only pseudonymise the annotators involved before release. There were between three and five annotators involved in the project per target language. Annotations were reviewed on a spot-checking basis, but independent overlapping annotations were not collected. LocHD contains hypotheses but no references.

We compute z-scores per annotator and evaluate QE models on a global correlation (Kendall’s τ) with the z-scores. This data is not suitable to transform into preference pairs because we only have one scored translation per segment, so we only use it to compute correlations.

LocCheck. Motivated by the observations stated in the introduction and exemplified in Table 1, we create a localisation-specific challenge set addressing a specific set of potential issues. We think of this dataset as a kind of CheckList (Ribeiro et al.,

![](images/d26f73546ee6b77ed140619e1dd3a6c00f67b0c49451189551e49b97684820fe.jpg)  
Figure 2: Margin Ranking Loss applied to a COMETKiwi model. Figure adapted from Rei et al. (2020, 2022).

2020), covering behaviours that may appear simple on the surface but could get lost while optimising for other skills. Specific points we address here include the translation of long number sequences, the preservation of leading or trailing whitespaces, the correct use of non-breaking spaces in French, the translation of all-caps sentences, and more.

We create both an internal and a public data split: For the public split, we augment data from WMT25 and BOUQuET (Omnilingual MT Team et al., 2025), whereas the internal split is sourced from real samples in our internal data. We create an internal split because these specific issues are rarer in public MT datasets. We include eleven target languages in LocCheck, all translating from English: Arabic, German, Spanish, Finnish, French, Hindi, Japanese, Polish, Russian, Thai, and Chinese. Our public split is missing Finnish, Polish and Thai as target languages because they were not present in WMT25 or BOUQuET.

Each row consists of a source, a reference, a positive sample, and a negative sample which is identical to the positive sample except for the variable under study. We ensure that the reference differs from the positive and negative samples. Appendix E.1 gives a detailed description of each heuristic, and shows the distribution of samples in both splits. The code for the public test set is released for reproducibility.

Training data augmentation. By definition, many phenomena in LocCheck are long-tail issues: Although they are rated as important by translators, they occur rarely in real-world data. Therefore, we create augmented training data to ensure the model sees these patterns. We apply the same transformations that were used to create LocCheck to a portion of our internal data, yielding 5k pairs.

WMT Data. COMETKiwi (Rei et al., 2022) was originally trained with normalised direct assessments from the WMT years 2017-20 (Bojar et al., 2017, 2018; Barrault et al., 2019, 2020). We experimented with mixing in data from those years during training to mitigate forgetting, and test on preference pairs derived from WMT24 (Kocmi et al., 2024a) to validate the general performance of our finetuned metrics. See Appendix A.3 for how we process this data.

## 4 Metric Adaptation

## 4.1 Multi-Task Training

Rather than fine-tuning solely on preference pairs or continuous scores, we implement a multi-task training regime. We randomly mix samples from both datasets—continuous or contrastive—in every single batch. The respective losses for each type of data are described in this section. We compute the appropriate loss only for the corresponding samples, making the final batch loss a mixture of both losses. We balance the influence of each loss by varying the data distribution, empirically selecting a ratio of 2:1 in favour of continuous scores (cf. ablation in Appendix D).

Contrastive training with preference pairs. For training with preference pairs, we use the Margin Ranking Loss (MRL; Herbrich et al., 2000) to encourage the model to score the preferred hypothesis above the rejected one:

$$
\begin{array} { r } { \begin{array} { c } { \mathrm { M R L } ( x _ { p o s } , x _ { n e g } ) = } \\ { \mathrm { m a x } ( 0 , - ( x _ { p o s } - x _ { n e g } ) + m ) } \end{array} } \end{array}\tag{1}
$$

where $x _ { p o s } , x _ { n e g }$ are the scores of the preferred and rejected inputs, respectively, and the margin m is a hyperparameter. Figure 2 shows the loss schematic for MRL applied to COMETKiwi.

Regression on continuous scores. With the artificial continuous scores, we use the standard meansquared error (MSE) loss. A single score is computed for the hypothesis and compared against the target score described in § 3.

## 4.2 Tokeniser Adjustments

Some of the LocCheck rules involve special characters such as non-breaking spaces, which $\mathrm { C O M E T K i w i _ { D A } ^ { 2 2 } }$ ignores due to its tokeniser. The process of tokenisation involves multiple steps, all of which influence the outcome (cf. Schmidt et al., 2024): Normalisation, pretokenisation, tokeniser inference, and post-processing. We propose adjusting the pretokenisation step of the encoder model’s tokeniser, adding new tokens for special whitespace and other control characters. These are targeted interventions which somewhat change the distribution of tokenised texts seen by the model, but leave the majority of segments unchanged. Specifically, we keep trailing and multiple spaces, and add missing tokens which enable the model to “see” the most relevant special characters. Details of the tokeniser modifications are in Appendix B.

## 4.3 Models

Main system. We finetune $\mathrm { C O M E T K i w i _ { D A } ^ { 2 2 } }$ with the updated tokeniser using the multi-task loss on internal data. We use 10k samples of MSE data and 5k preference pairs for MRL. For comparison, we show runs with each loss individually, using a total of 15k samples respectively. Finally, we add the 5k samples of augmented preference pairs for a LocQE version with data augmentation. Since these augmented data are minimal pairs exemplifying the rules in LocCheck, they only provide an explicit signal to the model in cases where the baseline ranking is wrong, and none otherwise.

Public data analog. Similarly, we finetune models on the ACED data, using all available preference pairs and inferred continuous scores for the respective losses. Due to the limited training set size in ACED, we also added 5.7k WMT data samples to both of these training sets, keeping all data for the multi-task training here.

Hyperparameters. The margin m for the MRL loss is set to 0.02 (cf. ablation in Appendix C). We unfreeze the entire encoder, including the embedding layer, from the beginning of finetuning, in order to train the added token embeddings. Instead of freezing the encoder, we add learning rate warmup to stabilise training, and increase both the effective batch size and the learning rates. We use 50 warmup steps, an encoder learning rate of 1e-5, and a feedforward layer learning rate of 1e-4. The effective batch size is 512. We apply early stopping with a checkpoint every epoch, training for up to five epochs. The results show individual runs with a fixed random seed.

GEMBA Baseline. We additionally compute reference-free $\mathrm { G E M B A _ { E S A } – Q E }$ (Kocmi and Federmann, 2023; Kocmi et al., 2024b) with GPT-4.1 as the backbone. GEMBA has been shown to yield good system-level pairwise accuracy but perform relatively poorly at the segment level.

## 5 Domain Adaptation Results

## 5.1 Effective domain adaptation leveraging a small number of post-edits

Figure 3 shows the accuracy of the QE models at selecting the preferred translation over the rejected on the respective target test set. The baseline $\mathrm { C O M E T K i w i _ { D A } ^ { 2 2 } }$ struggles to identify preferred segments in our internal data, while it performs better initially on the ACED corpus. $\mathrm { G E M B A _ { E S A } – Q E }$ performs even worse at this task. Both the contrastive and the multi-task training consistently improve preference accuracy on the target corpus, adding between 6.1 and 13.8 percentage points. MSE fine-tuning worsens preference accuracy on internal data, while for ACED, the addition of original training data seems to reduce forgetting and improve overall performance.

In Appendix Table 10, we show performance of the fine-tuned models on WMT24 preference accuracy and ACES score as a proxy of general capability. We find that WMT24 preference accuracy stays within a relatively small range regardless of fine-tuning objective, though MRL-only with internal data degrades it slightly and multi-task training improves it somewhat. ACES score collapses with MRL-only training on internal data, and suffers almost regardless of the setting. When training with MRL on ACED plus original training data, ACES score is relatively unaffected. With the exception of MRL-only on internal data, the fine-tuned models also still outperform $\mathrm { G E M B A _ { E S A } – Q E }$ on ACES. We surmise that including original training data helps to stabilise general model capabilities, but so does the multi-task training setup. Our analysis in § 5.2 sheds further light on this observation.

![](images/44d816c8e633782218acd579a515bea1513c7073ac63fcd221b6fc4f612917f7.jpg)

![](images/65ab6ff6cdd2d6b87492af99e4d57f56b4fa44686e651f5f8660c3d05415cf32.jpg)  
Figure 3: Performance on the respective target test set, in terms of preference accuracy. Left: Internal data. Right: ACED corpus. Per-language-pair performance for the internal data is listed in Appendix Table 12.

![](images/cf386e9aaaa4a5feba2340b9d5d0f677c962fcb1235bf37813d4ed0706bc8948.jpg)  
Figure 4: Overall performance on LocHD, in terms of global Kendall’s τ. Fine-tuning was done on internal data. Per-language-pair performance is listed in $\mathsf { A p - }$ pendix Table 14.

Meanwhile, Figure 4 shows global Kendall’s $\tau$ on the LocHD corpus. $\mathrm { C O M E T K i w i _ { D A } ^ { 2 2 } }$ initially performs much worse at this than $\mathrm { G E M B A _ { E S A } – Q E }$ improving somewhat with our internal data training. Notably, MSE training improves global Kendall the most, while MRL-only training further worsens performance on global Kendall despite the matching domain. Multitask training strikes a balance between the two, yielding an overall improvement on both preference accuracy and global Kendall. Next, we analyse why this is the case.

## 5.2 Score distribution analysis

We observe that global correlation and segmentlevel accuracy conflict, with models that perform well by one of these measures tending to do worse on the other. This is in keeping with analysis by DiIanni and Deutsch (2025), who emphasise the difference between Global and Segment-Wise correlations: Global correlations try to place translations on an overall distribution of human scores, while segment-wise correlations or accuracy compare translations given the same source. Importantly, performance on one of these aspects does not translate directly to the other—for instance, human score noise in a regression setting (cf. Schmidt et al., 2026) might obscure direct comparisons. We argue that both are relevant in a localisation context, but segment-wise performance is more welldefined in our setting.

The two losses, MSE and MRL, intuitively represent a global and a segment-wise objective. When training purely on continuous scores, the model learns from isolated scores given to translations of many different sources, and can merely approximate comparisons between different translations of the same source. By contrast, when training purely on preference pairs, the model learns to compare translation quality given the same source, but has no anchor for the absolute score value.

To visualise the issue, we plot scores on the internal LocCheck split before and after fine-tuning. Figure 5 (left) shows MRL by itself: Since many preference pairs need to be reversed compared to the base model’s judgements—remember that baseline performance on internal preference data was just above guessing level—, the model learns to fulfill the loss by compressing the output scores to a band roughly twice the size of the margin.

![](images/f40a0c5800e72c8195833df17421a56744967a4b8dc91c3b181c87618ae86980.jpg)

![](images/f46acb92ca33c136fc235b9e508dc46f461542fe661fc0a50643ed4fd7d313a8.jpg)  
Figure 5: Scatter plots contrasting fine-tuned vs. baseline COMETKiwi scores on internal LocCheck split. Left: Model fine-tuned only with MRL; Right: Multi-task training.

This collapsed score distribution then makes it very difficult for the model to preserve ranking across different sources, affecting global Kendall.

Regularising MRL. Since our preference-pair data does not contain human scores, or held-out references with which to compute artificial scores, we initially attempt to regularise MRL against COMETKiwi<sup>22</sup><sub>DA</sub>’s original score. The equation and ablation study are described in Appendix C. This regulariser partially mitigated score collapse. However, due to the large number of samples where the baseline is wrong, MRL and the regulariser frequently conflict, producing worse overall outcomes with a stronger regularisation weight.

In our main multi-task training setup, we instead process the data in two separate ways, inducing a chrF++-based score to regress towards for MSE, and creating preference pairs for MRL. Figure 5 (right) shows the resulting score distribution on LocCheck: Scores remain much more spread out across the range, while still producing better separation between the positive and negative samples.

## 6 LocCheck Results

Table 3 shows results on LocCheck, both the public performs poorly on both splits, well below guessing on the public split. We performed additional checks to rule out corrupt data, then tested our fine-tuned models on both splits. We ablate our tokeniser adjustments and fine-tuning approaches separately from each other. In Figure 1 we show the macro average over both splits.

<table><tr><td>Model</td><td>Internal</td><td>Public</td></tr><tr><td>baseline</td><td>45.1</td><td>30.1</td></tr><tr><td>+ new tokeniser</td><td>49.5</td><td>35.8</td></tr><tr><td>+ MSE</td><td></td><td></td></tr><tr><td>base tokeniser</td><td>56.3</td><td>40.5</td></tr><tr><td>new tokeniser</td><td>61.6</td><td>41.4</td></tr><tr><td>+ MRL</td><td></td><td></td></tr><tr><td>base tokeniser</td><td>41.9</td><td>50.3</td></tr><tr><td>new tokeniser</td><td>54.0</td><td>54.2</td></tr><tr><td>+ MSE + MRL</td><td></td><td></td></tr><tr><td>new tokeniser</td><td>64.1</td><td>46.3</td></tr><tr><td>+ data augmentation</td><td>89.6</td><td>86.1</td></tr></table>

Table 3: Effect of the tokeniser and data augmentations on preference accuracy (%) over LocCheck. Internal is our in-house challenge set; Public is the same set of heuristics applied to WMT25 system outputs and the BOUQuET test split. Best result per column in bold.

Tokeniser adaptation helps with learning specific patterns. Results on LocCheck demonstrate a large improvement (4-10 percentage points in most cases) simply due to the tokenizer adjustments: Replacing the tokeniser allows the model to ‘see’ leading and trailing spaces for the first time, for instance. The All Caps rule is learned better with the new tokeniser, while the NBSP rule can only be learned correctly with the new tokeniser. Appendix E.2 offers more detail on how the individual categories respond to the new tokeniser, as well as to the different fine-tuning approaches. However, the overall results are not satisfactory, even with multi-task fine-tuning: While accuracy on the internal split has reached 64%, accuracy on the public split is still below guessing level.

LocCheck needs data augmentation. As expected, adding this augmented data allows the finetuned model to rank the majority of samples correctly, now approaching 90% on both data splits. However, preference accuracy on the main test set drops somewhat across language pairs, compared to the regular multi-task training (see Appendix Table 12). This reflects the shift in the training distribution, and indicates that the mentioned “rules” may in fact not be consistently applied by the posteditors. This result suggests that although QE models can be efficiently taught to respect explicit editing rules, heuristics may equally remain a useful tool in the localisation pipeline.

## 7 Conclusions

Real-life localisation data is significantly different from public machine translation and metric benchmarks, introducing a domain shift that affects QE models strongly. To help with the evaluation of QE models for a localisation context, we have proposed two new datasets: LocCheck, focused on localisation-specific long-tail issues, and LocHD, a dataset of realistic translations annotated with ESA scores from professional translators.

To bridge the domain gap in the absence of traditional metric training data, we have demonstrated a data processing approach to leverage in-domain post-edits into two types of signal: Between-source regression of continuous scores and within-source ordering of preference pairs. Both approaches individually have their limitations, and indeed, initial results illustrate the trade-off between global between-source correlation and within-source preference accuracy. To address this trade-off, we combine both losses in a multi-task training approach, where they stabilise each other and yield better overall results. Our recommendation for future adaptation and/or training of metrics, therefore, is to consistently incorporate training signals at both the global and the segment level.

Using these two signals enables us to perform data-efficient domain adaptation without requiring human scores. Specifically, we adapted COMETKiwi<sup>22</sup><sub>DA</sub> to localisation data with training sets of only 10k-20k samples, less than 1% of its original training data. Along with targeted tokeniser interventions, our fine-tuning approaches yield LocQE models more suited to localisation data. To contend with long-tail issues described by translators, we find that targeted data augmentation is required.

Future Work. The application of modified QE models to MT preference optimisation training, such as DQO, could yield deeper insights into their behaviour. We leave this exploration to future work.

## Limitations

We acknowledge several limitations of this work. Primarily, some of the data for our main experiments is internal and cannot be published. However, we have made an effort to provide comparable experiments on similar, previously published data, showing qualitatively similar results, to share code for reproducibility, and to describe the internal data in detail.

The post-edits used are potentially noisy, particularly in the sense that different stylistic preferences may apply for different client projects. Our analysis in Appendix A.4 gives a further sense of the subjectivity of edits, indicating that not every preference pair is likely to yield a helpful signal. Nevertheless, our results show promising results for domain adaptation even with this limitation.

Additionally, we have demonstrated our approach on only one type of quality estimation model. This was done for efficiency and cost reasons, as COMETKiwi is a lightweight model that makes it easier to inference and fine-tune. Tuning a significantly larger model such as MetricX-24 would have been out of scope for this project.

## AI Assistant Information

Coding assistants were used in implementing the data processing, fine-tuning approaches, and creating the plots. The outputs were thoroughly checked and tested for purpose. No LLMs were used in the writing of this article.

## Acknowledgments

The data processing, tokeniser adaptation, and finetuning setups were implemented by KH during an internship at LILT, advised by JW. GB provided significant help in running further ablations and combining the two losses into the multi-task finetuning. KH wrote the paper draft with input from both co-authors.

Thank you to Kaden Uhlig for helpful discussions during the development of this project. We also thank the anonymous reviewers for their insights and feedback.

This work was co-funded by the European Union (ERC, EPICAL, 101141712). Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council. Neither the European Union nor the granting authority can be held responsible for them.

## References

Chantal Amrhein, Nikita Moghe, and Liane Guillou. 2023. ACES: Translation accuracy challenge sets at WMT 2023. In Proceedings of the Eighth Conference on Machine Translation, pages 695–712, Singapore. Association for Computational Linguistics.

Loïc Barrault, Magdalena Biesialska, Ondˇrej Bojar, Marta R. Costa-jussà, Christian Federmann, Yvette Graham, Roman Grundkiewicz, Barry Haddow, Matthias Huck, Eric Joanis, Tom Kocmi, Philipp Koehn, Chi-kiu Lo, Nikola Ljubešic, Christof´ Monz, Makoto Morishita, Masaaki Nagata, Toshiaki Nakazawa, Santanu Pal, and 2 others. 2020. Findings of the 2020 conference on machine translation (WMT20). In Proceedings of the Fifth Conference on Machine Translation, pages 1–55, Online. Association for Computational Linguistics.

Loïc Barrault, Ondˇrej Bojar, Marta R. Costa-jussà, Christian Federmann, Mark Fishel, Yvette Graham, Barry Haddow, Matthias Huck, Philipp Koehn, Shervin Malmasi, Christof Monz, Mathias Müller, Santanu Pal, Matt Post, and Marcos Zampieri. 2019. Findings of the 2019 conference on machine translation (WMT19). In Proceedings ofthe Fourth Conference on Machine Translation (Volume 2: Shared Task Papers, Day 1), pages 1–61, Florence, Italy. Association for Computational Linguistics.

Nathaniel Berger, Miriam Exel, Matthias Huck, and Stefan Riezler. 2024. Post-edits are preferences too. In Proceedings of the Ninth Conference on Machine Translation, pages 1289–1300, Miami, Florida, USA. Association for Computational Linguistics.

Ondˇrej Bojar, Rajen Chatterjee, Christian Federmann, Yvette Graham, Barry Haddow, Shujian Huang, Matthias Huck, Philipp Koehn, Qun Liu, Varvara Logacheva, Christof Monz, Matteo Negri, Matt Post, Raphael Rubino, Lucia Specia, and Marco Turchi. 2017. Findings of the 2017 conference on machine translation (WMT17). In Proceedings of the Second Conference on Machine Translation, pages 169–214, Copenhagen, Denmark. Association for Computational Linguistics.

Ondˇrej Bojar, Christian Federmann, Mark Fishel, Yvette Graham, Barry Haddow, Matthias Huck, Philipp

Koehn, and Christof Monz. 2018. Findings of the 2018 conference on machine translation (WMT18). In Proceedings ofthe Third Conference on Machine Translation: Shared Task Papers, pages 272–303, Belgium, Brussels. Association for Computational Linguistics.

Bianka Buschbeck and Miriam Exel. 2020. A parallel evaluation data set of software documentation with document structure annotation. In Proceedings of the 7th Workshop on Asian Translation, pages 160– 169, Suzhou, China. Association for Computational Linguistics.

Zewen Chi, Li Dong, Furu Wei, Nan Yang, Saksham Singhal, Wenhui Wang, Xia Song, Xian-Ling Mao, Heyan Huang, and Ming Zhou. 2021. InfoXLM: An information-theoretic framework for cross-lingual language model pre-training. In Proceedings of the 2021 Conference ofthe North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies, pages 3576–3588, Online. Association for Computational Linguistics.

Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. 2017. Deep reinforcement learning from human preferences. In Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc.

Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. 2020. Unsupervised cross-lingual representation learning at scale. In Proceedings of the 58th Annual Meeting of the Associationfor Computational Linguistics, pages 8440– 8451, Online. Association for Computational Linguistics.

Colten DiIanni and Daniel Deutsch. 2025. Don’t sweat the small stuff: Segment-level meta-evaluation based on pairwise difference correlation. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 25062–25070, Suzhou, China. Association for Computational Linguistics.

Markus Freitag, George Foster, David Grangier, Viresh Ratnakar, Qijun Tan, and Wolfgang Macherey. 2021. Experts, errors, and context: A large-scale study of human evaluation for machine translation. Transactions ofthe Associationfor Computational Linguistics, 9:1460–1474.

Nuno M. Guerreiro, Ricardo Rei, Daan van Stigt, Luisa Coheur, Pierre Colombo, and André F. T. Martins. 2024. xCOMET: Transparent machine translation evaluation through fine-grained error detection. Transactions of the Association for Computational Linguistics, 12:979–995.

Ralf Herbrich, Thore Graepel, and Klaus Obermayer. 2000. Large margin bank boundaries for ordinal regression. In Advances in Large-Margin Classifiers, pages 115–132. MIT Press.

John Hewitt. 2021. Initializing new word embeddings for pretrained language models.

Tom Kocmi, Eleftherios Avramidis, Rachel Bawden, Ondˇrej Bojar, Anton Dvorkovich, Christian Federmann, Mark Fishel, Markus Freitag, Thamme Gowda, Roman Grundkiewicz, Barry Haddow, Marzena Karpinska, Philipp Koehn, Benjamin Marie, Christof Monz, Kenton Murray, Masaaki Nagata, Martin Popel, Maja Popovic, and 3 others. 2024a.´ Findings of the WMT24 general machine translation shared task: The LLM era is here but MT is not solved yet. In Proceedings ofthe Ninth Conference on Machine Translation, pages 1–46, Miami, Florida, USA. Association for Computational Linguistics.

Tom Kocmi and Christian Federmann. 2023. Large language models are state-of-the-art evaluators of translation quality. In Proceedings of the 24th Annual Conference ofthe European Associationfor Machine Translation, pages 193–203, Tampere, Finland. European Association for Machine Translation.

Tom Kocmi, Hitokazu Matsushita, and Christian Federmann. 2022. MS-COMET: More and better human judgements improve metric performance. In Proceedings of the Seventh Conference on Machine Translation (WMT), pages 541–548, Abu Dhabi, United Arab Emirates (Hybrid). Association for Computational Linguistics.

Tom Kocmi, Vilém Zouhar, Eleftherios Avramidis, Roman Grundkiewicz, Marzena Karpinska, Maja Popovic, Mrinmaya Sachan, and Mariya Shmatova.´ 2024b. Error span annotation: A balanced approach for human evaluation of machine translation. In Proceedings of the Ninth Conference on Machine Translation, pages 1440–1453, Miami, Florida, USA. Association for Computational Linguistics.

Tom Kocmi, Vilém Zouhar, Christian Federmann, and Matt Post. 2024c. Navigating the metrics maze: Reconciling score magnitudes and accuracies. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 1999–2014, Bangkok, Thailand. Association for Computational Linguistics.

Taku Kudo and John Richardson. 2018. SentencePiece: A simple and language independent subword tokenizer and detokenizer for neural text processing. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 66–71, Brussels, Belgium. Association for Computational Linguistics.

Jessy Lin, Geza Kovacs, Aditya Shastry, Joern Wuebker, and John DeNero. 2022. Automatic correction of human translations. In Proceedings ofthe 2022 Conference of the North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies, pages 494–507, Seattle, United States. Association for Computational Linguistics.

Nikita Moghe, Arnisa Fazla, Chantal Amrhein, Tom Kocmi, Mark Steedman, Alexandra Birch, Rico Sennrich, and Liane Guillou. 2025. Machine translation meta evaluation through translation accuracy challenge sets. Computational Linguistics, 51(1):73–137.

Omnilingual MT Team, Pierre Andrews, Mikel Artetxe, Mariano Coria Meglioli, Marta R. Costa-jussà, Joe Chuang, David Dale, Cynthia Gao, Jean Maillard, Alex Mourachko, Christophe Ropers, Safiyyah Saleem, Eduardo Sánchez, Ioannis Tsiamas, Arina Turkatenko, Albert Ventayol-Boada, and Shireen Yates. 2025. BOUQuET: Dataset, Benchmark and Open initiative for Universal Quality Evaluation in Translation. Preprint, arXiv:2502.04314.

Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. 2022. Training language models to follow instructions with human feedback. Preprint, arXiv:2203.02155.

Maja Popovic. 2021.´ Agree to disagree: Analysis of inter-annotator disagreements in human evaluation of machine translation output. In Proceedings of the 25th Conference on Computational Natural Language Learning, pages 234–243, Online. Association for Computational Linguistics.

Lorenzo Proietti, Roman Grundkiewicz, and Matt Post. 2026. PEAR: Pairwise evaluation for automatic relative scoring in machine translation. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 42189–42207, San Diego, California, United States. Association for Computational Linguistics.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. 2023. Direct preference optimization: Your language model is secretly a reward model. In Advances in Neural Information Processing Systems, volume 36, pages 53728–53741. Curran Associates, Inc.

Ricardo Rei, Ana C Farinha, Chrysoula Zerva, Daan van Stigt, Craig Stewart, Pedro Ramos, Taisiya Glushkova, André F. T. Martins, and Alon Lavie. 2021. Are references really needed? unbabel-IST 2021 submission for the metrics shared task. In Proceedings ofthe Sixth Conference on Machine Translation, pages 1030–1040, Online. Association for Computational Linguistics.

Ricardo Rei, Craig Stewart, Ana C Farinha, and Alon Lavie. 2020. COMET: A neural framework for MT evaluation. In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 2685–2702, Online. Association for Computational Linguistics.

Ricardo Rei, Marcos Treviso, Nuno M. Guerreiro, Chrysoula Zerva, Ana C Farinha, Christine Maroti,

José G. C. de Souza, Taisiya Glushkova, Duarte Alves, Luisa Coheur, Alon Lavie, and André F. T. Martins. 2022. CometKiwi: IST-unbabel 2022 submission for the quality estimation shared task. In Proceedings ofthe Seventh Conference on Machine Translation (WMT), pages 634–645, Abu Dhabi, United Arab Emirates (Hybrid). Association for Computational Linguistics.

Marco Tulio Ribeiro, Tongshuang Wu, Carlos Guestrin, and Sameer Singh. 2020. Beyond accuracy: Behavioral testing of NLP models with CheckList. In Proceedings ofthe 58th Annual Meeting ofthe Associationfor Computational Linguistics, pages 4902– 4912, Online. Association for Computational Linguistics.

Craig W Schmidt, Varshini Reddy, Haoran Zhang, Alec Alameddine, Omri Uzan, Yuval Pinter, and Chris Tanner. 2024. Tokenization is more than compression. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 678–702, Miami, Florida, USA. Association for Computational Linguistics.

Finn Schmidt, Jan Philip Wahle, Terry Ruas, and Bela Gipp. 2026. Who watches the watchmen? humans disagree with translation metrics on unseen domains. In Findings ofthe Associationfor Computational Linguistics: ACL 2026, pages 22822–22841, San Diego, California, United States. Association for Computational Linguistics.

Javad Pourmostafa Roshan Sharami, Dimitar Shterionov, Frédéric Blain, Eva Vanmassenhove, Mirella De Sisto, Chris Emmery, and Pieter Spronck. 2023. Tailoring domain adaptation for machine translation quality estimation. In Proceedings ofthe 24th Annual Conference ofthe European Associationfor Machine Translation, pages 9–20, Tampere, Finland. European Association for Machine Translation.

Matthew Snover, Bonnie Dorr, Rich Schwartz, Linnea Micciulla, and John Makhoul. 2006. A study of translation edit rate with targeted human annotation. In Proceedings ofthe 7th Conference ofthe Association for Machine Translation in the Americas: Technical Papers, pages 223–231, Cambridge, Massachusetts, USA. Association for Machine Translation in the Americas.

Shaomu Tan and Christof Monz. 2025. ReMedy: Learning machine translation evaluation from human preferences with reward modeling. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 4370–4387, Suzhou, China. Association for Computational Linguistics.

Kaden Uhlig, Joern Wuebker, Raphael Reinauer, and John Denero. 2025. Cross-lingual human-preference alignment for neural machine translation with direct quality optimization. In Proceedings of the Tenth Conference on Machine Translation, pages 31–51, Suzhou, China. Association for Computational Linguistics.

Yaushian Wang, Ashley Wu, and Graham Neubig. 2022. English contrastive learning can learn universal crosslingual sentence embeddings. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 9122–9133, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Vilém Zouhar, Shehzaad Dhuliawala, Wangchunshu Zhou, Nico Daheim, Tom Kocmi, Yuchen Eleanor Jiang, and Mrinmaya Sachan. 2023. Poor man’s quality estimation: Predicting reference-based MT metrics without the reference. In Proceedings ofthe 17th Conference ofthe European Chapter ofthe Associationfor Computational Linguistics, pages 1311– 1325, Dubrovnik, Croatia. Association for Computational Linguistics.

Vilém Zouhar, Shuoyang Ding, Anna Currey, Tatyana Badeka, Jenyuan Wang, and Brian Thompson. 2024. Fine-tuned machine translation metrics struggle in unseen domains. In Proceedings ofthe 62nd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 488–500, Bangkok, Thailand. Association for Computational Linguistics.

## A Data Processing Details

We filter our internal data substantially before processing it. For the initial filtering, we use these heuristics:

• Only use documents with any human reviews

• Only use language pairs with > 5, 000 segments available after document filtering

• Only using segments with > 5 but < 425 characters (around the $9 9 ^ { t h }$ percentile in length)

We then process data into preference pairs on one hand and continuous scores on the other hand, as described below. After applying both transformations separately, we sample train, dev, and test splits from both types of data, ensuring that each individual client project is assigned to only one split across both datasets to prevent overlap.

## A.1 Preference Pairs from Post-Edits

To create the preference pairs from our post-edit data, we filter for segments where the final translation was provided or approved by a human reviewer, treating this as the preferred translation. As negative samples, we use previous machine-translated versions of the segment, where they differ from the final version. Table 11 lists absolute test set sizes, and relative train and development set sizes across all language pairs used. For the ACED corpus, we always use the post-edited segment as the preferred, provided it is different from the machine translated segment. Because significant portions of the ACED data were not edited, this unfortunately limits the number of available preference pairs from this corpus (cf. Table 2).

## A.2 Continuous Scores from Post-Edits

Using HTER (human-targeted translation edit rate), as part of the training signal for metrics is an established practice (e.g., Sharami et al., 2023). That said, TER is somewhat unstable on a segment level, especially when segments are short—a common occurrence in our data. Thus, we select chrF++ as our surface metric for measuring post-edit distance. Additionally, we use heuristics to identify numbers, URLs, and “DNTs” (Do Not Translate) which the reviewer had to edit. Since these are semantically major errors which may result only in a small edit distance, we subtract an additional 20 chrF points for these cases (equivalent to 5 out of a maximum score of 25 in MQM; see Freitag et al., 2021). Finally, we calculate z-scores from these artificial scores, and rescale them to between [0,1].

<table><tr><td>System</td><td>MT</td><td>AI-PE</td><td>Human</td></tr><tr><td>MT</td><td></td><td>0.26</td><td>0.14</td></tr><tr><td>AI-PE</td><td>0.43</td><td></td><td>0.23</td></tr><tr><td>Human</td><td>0.61</td><td>0.52</td><td></td></tr></table>

Table 4: Pairwise system comparison results from an internal study, where each cell indicates the win rate of the system listed in the row when compared against the system listed in the corresponding column. MT refers to machine translation, AI-PE to automatic post-editing of MT output, and Human to final human translation.
<table><tr><td colspan="2">Metric ρ T</td></tr><tr><td> $\mathrm { C O M E T _ { D A } ^ { 2 2 } }$ </td><td>0.404 0.282</td></tr><tr><td> $\mathrm { C O M E T K i w i _ { D A } ^ { 2 2 } }$  0.371</td><td>0.253</td></tr><tr><td>BLEU 0.166</td><td>0.115</td></tr><tr><td>chrF++ 0.227</td><td>0.158</td></tr><tr><td>TER -0.195</td><td>-0.137</td></tr></table>

Table 5: Segment-level Spearman R (ρ) and Kendall’s Tau (τ) correlations between ESA scores and automatic evaluation metrics in the internal study. BLEU, chrF++, and TER are computed using their segment-level variants. Only MT and AI-PE annotations are included, with the final human translation serving as the reference for reference-based metrics.

## A.3 WMT Data Processing

To use the WMT data in both our training losses, we need both continuous scores and preference pairs. As continuous scores, we use z-scores derived from the DA or ESA annotations, rescaled to between [0,1]. To transform the data into preference pairs, we compare the scores of all submissions for a given segment. If the z-scores of two candidates are more than one standard deviation apart, we use them as a preference pair. The raw scores act as a sanity check: We never use a submission as the rejected translation if its raw score is 98 or higher. We subsample the WMT24 preference accuracy test set to one preference pair per source, in order to keep the test set size manageable.

## A.4 Quality Validation of Internal Data

To provide a clearer sense of our internal data, we annotate a sample (n = 984 annotations across 317 source segments in 53 documents) of English-German hypotheses in an ESA setting (Kocmi et al.,

2024b). The annotators in this small validation study are German-speaking NLP practitioners, and the data is sampled independently from LocHD. The data sample is drawn randomly from our internal data, but with the criterion that there are three hypotheses present: A machine translation, an AI post-edit, and a final human post-edit. Wherever possible, we show document context.

Table 4 shows how often each type of hypothesis wins against the other hypotheses. That is, the human translation won over the MT system 61% of the time while the MT system was rated preferable over the human translation 14% of the time. The rest are ties. These win rates show how often our annotators agreed with the the edits made by the AI-PE or the human post-edit, effectively giving us a sense of how often edits are subjective.

Table 5 reports segment-level Spearman’s rank correlation $( \rho )$ and Kendall’s Tau (τ) between human judgements and automatic evaluation metrics. Annotations corresponding to the final human postedit are excluded and instead used as a reference, resulting in a total of 644 annotations. Overall, we observe slightly stronger correlations than those reported by (Rei et al., 2022). While neural metrics show very similar correlation values, surfacelevel metrics show substantially higher correlations compared to prior work. This can be attributed to the bias introduced by the post-editing setting in which the data was generated. Later edits tend to be close to the original version, which is measured by BLEU, TER and chrF++.

## B Details of Tokeniser Modifications

The pretrained encoder used by COMETKiwi<sup>22</sup> :22 is based on XLM-R (Conneau et al., 2020), specifically, InfoXLM (Chi et al., 2021). Thus, it uses the XLM-R tokeniser, with a vocabulary of ca. 250k subword tokens, trained with SentencePiece (Kudo and Richardson, 2018) and implemented for inference in the Huggingface tokenizers library.

By default, the XLM-R tokeniser splits on any whitespace, stripping away trailing and multiple spaces. It also replaces spaces with the “metaspace” character. Since we are only inferencing the tokeniser, it has already learned exclusively tokens that do not cross whitespace boundaries. Rather than implement a new whitespace pretokeniser, we can simply remove this processing step to preserve trailing and multiple spaces without significantly changing the subword token sequence, as long as we keep the metaspace replacement.

Regarding the special space and control characters, the tokenizers library allows us to add tokens which bypass the Unicode normalisation step. We add the non-breaking space (NBSP) in the narrow and regular form, the zero-width nonjoiner (ZWNJ), the zero-width space (ZWSP) and control characters such as the left-to-right (LTR) and right-to-left (RTL) marker as additional tokens. In the model’s embedding layer, the new tokens are initialised according to the AvgEmb procedure shown by Hewitt (2021).

## C MRL Regulariser and Margin Ablation

Margin Ranking Loss comes with the margin hyperparameter m. Rather than set it only empirically, we test a range of margins in this section.

Additionally, we test a simple regularisation technique: MRL on its own cannot provide a signal about appropriate absolute values. However, computing a continuous score for both the positive and negative sample would require a held-out reference, which we do not have access to for our internal data. Therefore, we compute the original model’s score on both the positive and negative sample, and use these scores as a regularisation target.

Using mean-squared error (MSE) between the predicted score $x _ { i }$ and the original score $y _ { i } .$ , as a regulariser, the full loss equation becomes:

$$
\begin{array} { r l } & { \mathcal { L } = \mathbf { M } \mathbf { R } \mathbf { L } ( x _ { p o s } , x _ { n e g } ) } \\ & { \quad + \lambda _ { r } ( \mathbf { M } \mathbf { S } \mathbf { E } ( x _ { p o s } , y _ { p o s } ) + \mathbf { M } \mathbf { S } \mathbf { E } ( x _ { n e g } , y _ { n e g } ) ) } \end{array}\tag{2}
$$

We ablate the hyperparameters m and λ, testing margins $m \in \{ 0 . 0 , 0 . 0 1 , 0 . 0 2 , 0 . 0 5 , 0 . 1 \}$ , and regulariser weights $\lambda \in \{ 0 . 0 , 0 . 0 1 , 0 . 0 5 , 0 . 1 , 0 . 5 , 1 \}$ Figure 6 shows the resulting pareto curve over two metrics: Segment-wise preference accuracy and global Kendall’s τ, both measured on our internal test set.

starts out with poor preference accuracy on our internal data, this regulariser frequently conflicts with MRL. Tuning the strength of the regularisation loss via λ proves insufficient for balancing the two: The majority of points on the pareto curve has a $\lambda = 0 . 0$ . Therefore, we discard regularisation to the original scores, and instead focus on multi-task training.

We select a margin of 0.02, which sits on the pareto curve, and corresponds to the common semantics of COMET-type metrics, where a difference of 0.02 already corresponds to a meaningful difference in quality (cf. Kocmi et al., 2024c).

Performance on internal data  
![](images/05b8dc19de68ac376bc78d278b3a1fd2823b6a9b3253ca5716037c1c0cefd7e9.jpg)  
Figure 6: Pareto curve of Kendall’s τ vs. preference accuracy achieved by contrastive training with different settings for m and λ.

## D Data Mix Ablation

In the multi-task setup, we balance the influence of the two losses by varying the composition of the training data. This leaves open two choices: the share of original training data (WMT2017- 20) mixed into the training data, and the ratio between continuous scores (MSE) and preference pairs (MRL). Table 6 shows our ablation experiment for both choices.

First, we vary the share of internal data from 0% to 100%, applying the same mix to the MSE and the MRL side (10k samples each). We originally mixed in WMT data to mitigate forgetting (cf. § 3). In the multi-task setting, however, preference accuracy improves with a growing share of internal data across all preference-based test sets, including WMT24 itself. Only ACES and the global correlations benefit from added WMT data, mirroring the trade-off between segment-wise and global performance discussed in the main text. We decide to train our final models on internal data only.

Then, we vary the ratio of MSE to MRL samples on this internal-only mix, from pure MRL to pure MSE at a matched total budget of 15k samples. The two extremes recover the single-loss failure modes: Pure MRL yields the highest preference accuracy on the internal test set, but collapses the global correlations and ACES performance, while pure MSE shows the opposite behaviour. Between the extremes, the measures trade off smoothly. We select the 2:1 ratio, which achieves the best WMT24 accuracy and strong LocCheck performance, and retains most of the global-correlation gains of MSEonly training, at a small cost in preference accuracy on the internal test set.

## E LocCheck Details

## E.1 Composition

We include eleven language pairs in LocCheck, translating from English into the following target languages: Arabic, German, Spanish, Finnish, French, Hindi, Japanese, Polish, Russian, Thai, and Chinese. These target languages cover both very high-resource and more mid-resource language pairs, seven different scripts, and as many language families. We apply the following heuristics to find and create positive and negative samples:

• Copy URLs. Although URLs may be localised, an MT model cannot know the localised URL and so should be expected to copy the source URL if no other instruction is given. We sample translations where a URL was changed during the post-editing process. We ensure that the reference and positive sample contain the source URL, then place the changed URL in the negative sample.

• Copy DNTs. Do Not Translate items are elements within the segment, usually marked by a specific syntax, that should be copied to the correct place in the target segment, and must not be translated. We sample segments where a DNT item was changed during the post-editing process, analogously to URLs.

• Long Numbers. If a number is mistranslated, the metric should catch this and consider it a major error, even if the number is long. We sample translations organically containing long digit sequences, and check that the intermediate translation matches the reference. Then, we randomly remove, add, or swap one or more digits, and place the resulting digit sequence within the negative sample.

• Leading/Trailing Spaces. Translators are careful to leave leading and trailing spaces in a segment unchanged, but the default tokenisation of COMETKiwi models drops these entirely. We sample translations with leading or trailing spaces, then randomly add, remove, or replace whitespace characters from these leading or trailing space sequences. The whitespace characters we sample from for addition or replacement are the regular space, tab, NBSP, ZWSP, and ZWNJ.

• Extra Period. If a longer segment does not end in a period, language models still sometimes prefer a translation ending in a period, but translators do not. We sample segments with a minimum of ten source words that do not end in a period, then add the appropriate sentence-ending symbol for each script to the negative sample.

• All Caps. If the source segment is written in all-caps, a translation in all-caps should be preferred to a translation in sentence case. We sample segments in sentence case and use the initial translation as the negative sample, then upper-case the source, positive sample, and reference. This criterion is only applicable in Latin and Cyrillic script.

<table><tr><td></td><td colspan="2">LocCheck</td><td colspan="2">Internal Data</td><td>LocHD Kendall</td><td>WMT24 Pref Acc.</td><td>ACES</td></tr><tr><td>Model Baseline</td><td>Internal 45.1</td><td>Public 30.1</td><td>Pref Acc. 53.1</td><td>Pearson 0.173</td><td>0.107</td><td>66.2</td><td>Score 17.94</td></tr><tr><td colspan="8">Internal/WMT data mix (MSE 10k + MRL 10k)</td></tr><tr><td colspan="8"></td></tr><tr><td>WMT only 25% internal</td><td>52.7 58.9</td><td>36.7</td><td>52.7 56.2</td><td>0.201 0.222</td><td>0.120 0.138</td><td>66.2 66.9</td><td>17.40 14.16</td></tr><tr><td>50% internal</td><td>61.0</td><td>40.8</td><td>55.2</td><td>0.220</td><td>0.135</td><td>66.9</td><td>15.08</td></tr><tr><td>75% internal</td><td></td><td></td><td>58.4</td><td>0.225</td><td>0.124</td><td>66.8</td><td></td></tr><tr><td>Internal only</td><td>61.9 62.1</td><td>44.4</td><td>60.7</td><td>0.194</td><td>0.131</td><td>67.5</td><td>13.14 11.25</td></tr><tr><td colspan="8">MSE:MRL supervision ratio (internal data only)</td></tr><tr><td>0:1 (MRL only, 15k)</td><td></td><td>54.2</td><td>66.9</td><td>0.113</td><td>0.044</td><td>63.9</td><td></td></tr><tr><td>1:4 (2.5k + 10k)</td><td>54.0 61.0</td><td>44.5</td><td>61.2</td><td>0.179</td><td>0.110</td><td>67.0</td><td>3.67 11.20</td></tr><tr><td>1:2 (5k + 10k)</td><td>61.1</td><td>46.7</td><td>61.4</td><td>0.178</td><td>0.112</td><td>67.5</td><td>11.50</td></tr><tr><td>1:1 (10k + 10k)</td><td>62.1</td><td>44.4</td><td>60.7</td><td>0.194</td><td>0.131</td><td>67.5</td><td>11.25</td></tr><tr><td>2:1 (10k + 5k, ours)</td><td>64.1</td><td>46.3</td><td>59.6</td><td>0.207</td><td>0.135</td><td>67.6</td><td>11.97</td></tr><tr><td>4:1 (10k + 2.5k)</td><td>65.4</td><td>45.9</td><td>56.5</td><td>0.218</td><td>0.127</td><td>67.1</td><td>13.25</td></tr><tr><td>1:0 (MSE only, 15k)</td><td>61.6</td><td>41.4</td><td>51.1</td><td>0.234</td><td>0.145</td><td>65.5</td><td>14.58</td></tr></table>

Table 6: Ablation of training data mix. We use varying ratios of a fixed pool of data. We conclude that the multi-task training sees limited benefit from replaying WMT data, and settle on a 2:1 ratio of the two losses.
<table><tr><td></td><td>en-ar</td><td>en-de</td><td>en-es</td><td>en-fi</td><td>en-fr</td><td>en-hi</td><td>en-ja</td><td>en-pl</td><td>en-ru</td><td>en-th</td><td>en-zh-Hans</td><td>all</td></tr><tr><td>Copy URLs</td><td></td><td>8</td><td>6</td><td>一</td><td>12</td><td>一</td><td>18</td><td>2</td><td>1</td><td>一</td><td>5</td><td>52</td></tr><tr><td>Copy DNTs</td><td></td><td>一</td><td>一</td><td></td><td>一</td><td>一</td><td>一</td><td>1</td><td>4</td><td>1</td><td></td><td>5</td></tr><tr><td>Long Numbers</td><td>50</td><td>50</td><td>50</td><td>50</td><td>50</td><td>50</td><td>50</td><td>50</td><td>50</td><td>34</td><td>50</td><td>534</td></tr><tr><td>Lead./Trail. Sp.</td><td>50</td><td>50</td><td>50</td><td>50</td><td>50</td><td>50</td><td>50</td><td>50</td><td>50</td><td>50</td><td>50</td><td>550</td></tr><tr><td>Extra Period</td><td>50</td><td>50</td><td>50</td><td>50</td><td>50</td><td>50</td><td>50</td><td>50</td><td>50</td><td>一</td><td>50</td><td>500</td></tr><tr><td>All Caps</td><td>一</td><td>50</td><td>50</td><td>50</td><td>50</td><td>一</td><td>一</td><td>50</td><td>50</td><td>一</td><td></td><td>300</td></tr><tr><td>Spanish ¿ &amp;</td><td></td><td>一</td><td>50</td><td>一</td><td>一</td><td></td><td></td><td>一</td><td>一</td><td></td><td></td><td>50</td></tr><tr><td>French NBSPs</td><td></td><td></td><td></td><td>一</td><td>50</td><td>一</td><td></td><td>1</td><td>一</td><td></td><td></td><td>50</td></tr><tr><td>Unicode Spaces</td><td>50</td><td>50</td><td>50</td><td>50</td><td>50</td><td>39</td><td>50</td><td>50</td><td>50</td><td>34</td><td>50</td><td>523</td></tr><tr><td>Totals</td><td>200</td><td>258</td><td>306</td><td>250</td><td>312</td><td>189</td><td>218</td><td>253</td><td>255</td><td>118</td><td>205</td><td>2,564</td></tr></table>

Table 7: Distribution of samples in the internal LocCheck split.

• Spanish ¿ and ¡. We want to verify that the sentence-initial exclamation mark and question mark in Spanish are treated correctly by the metrics. We sample Spanish sentences containing these characters and remove them to create the negative sample.

• French NBSPs. French has specific rules about where non-breaking spaces should be used, but COMETKiwi models cannot differentiate special space characters. We sample French segments where a NBSP occurs with punctuation, and randomly either remove the NBSP or replace it with a regular whitespace character.

• Unicode Spaces. Some segments contain special Unicode whitespace and control characters such as NBSP, narrow NBSP, zero-width space, zero-width non-joiner (ZWNJ), zerowidth joiner (ZWJ), ideographic space, and the LTR/RTL marks. These are stripped by the original tokenizer but often need to be kept in the translation. We find segments where the reference contains any of these characters. We use the initial translation as a positive example if it contains the characters as well; otherwise, we use the reference. We produce the negative example by either dropping the special character or replacing it with a regular space.

Table 7 shows the composition of the Localisation CheckList by language pair and error category.

<table><tr><td></td><td>en-ar</td><td>en-de</td><td>en-es</td><td>en-fr</td><td>en-hi</td><td>en-ja</td><td>en-ru</td><td>en-zh</td><td>all</td></tr><tr><td>Copy URLs</td><td>一</td><td>一</td><td></td><td></td><td>一</td><td>一</td><td></td><td></td><td>一</td></tr><tr><td>Copy DNTs</td><td>一</td><td>一</td><td>一</td><td>一</td><td>一</td><td>一</td><td>一</td><td>一</td><td>一</td></tr><tr><td>Long Numbers</td><td>0</td><td>3</td><td>2</td><td>2</td><td>6</td><td>2</td><td>5</td><td>0</td><td>20</td></tr><tr><td>Lead./Trail. Sp.</td><td>12</td><td>0</td><td>2</td><td>0</td><td>50</td><td>42</td><td>46</td><td>8</td><td>160</td></tr><tr><td>Extra Period</td><td>50</td><td>30</td><td>31</td><td>30</td><td>50</td><td>50</td><td>50</td><td>50</td><td>341</td></tr><tr><td>All Caps</td><td>0</td><td>50</td><td>50</td><td>50</td><td>0</td><td>0</td><td>50</td><td>0</td><td>200</td></tr><tr><td>Spanish  $\therefore \& \mathrm { i }$ </td><td>0</td><td>0</td><td>50</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>50</td></tr><tr><td>French NBSPs</td><td>0</td><td>0</td><td>0</td><td>50</td><td>0</td><td>0</td><td>0</td><td>0</td><td>50</td></tr><tr><td>Unicode Spaces</td><td>0</td><td>0</td><td>0</td><td>18</td><td>0</td><td>2</td><td>0</td><td>0</td><td>20</td></tr><tr><td>Totals</td><td>62</td><td>83</td><td>135</td><td>150</td><td>106</td><td>96</td><td>151</td><td>58</td><td>841</td></tr></table>

Table 8: Distribution of samples across categories in the public LocCheck split. Samples are distributed across the eight LocCheck language pairs that occur in WMT25 and BOUQuET.
<table><tr><td></td><td></td><td>+Tok</td><td>+MSE</td><td>+Tok+MSE</td><td>+MRL</td><td>+Tok+MRL</td><td>+Tok+Multi</td><td>+Tok+Multi+Aug</td></tr><tr><td>Copy URLs</td><td>78.8</td><td>80.8</td><td>82.7</td><td>78.8</td><td>34.6</td><td>28.8</td><td>76.9</td><td>78.8</td></tr><tr><td>Copy DNTs</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td></tr><tr><td>Long Numbers</td><td>85.2</td><td>81.3</td><td>97.9</td><td>96.8</td><td>37.5</td><td>33.1</td><td>89.3</td><td>97.0</td></tr><tr><td>Lead./Trail. Sp.</td><td>25.5</td><td>68.4</td><td>27.5</td><td>44.4</td><td>31.6</td><td>41.5</td><td>39.3</td><td>78.2</td></tr><tr><td>Extra Period</td><td>40.2</td><td>44.6</td><td>51.8</td><td>47.6</td><td>44.6</td><td>38.6</td><td>49.4</td><td>92.6</td></tr><tr><td>All Caps</td><td>22.3</td><td>26.7</td><td>64.3</td><td>63.3</td><td>63.7</td><td>76.7</td><td>68.3</td><td>99.7</td></tr><tr><td>Spanish  $\mathit { i } \ \& \ \mathrm { i }$ </td><td>96.0</td><td>96.0</td><td>78.0</td><td>72.0</td><td>96.0</td><td>92.0</td><td>82.0</td><td>90.0</td></tr><tr><td>French NBSPs</td><td>42.0</td><td>8.0</td><td>50.0</td><td>38.0</td><td>52.0</td><td>82.0</td><td>80.0</td><td>88.0</td></tr><tr><td>Unicode Spaces</td><td>34.2</td><td>11.1</td><td>39.2</td><td>55.4</td><td>36.3</td><td>85.9</td><td>71.1</td><td>86.6</td></tr></table>

Table 9: Performance of COMETKiwi models on the Localisation CheckList set by error category. Fine-tuning was done with the internal dataset throughout.

## E.2 Analysis of Results

Table 9 breaks down the performance of COMETKiwi variants, fine-tuned on internal data, by category. Long numbers show a strong improvement under MSE fine-tuning. Recall that we explicitly subtracted from the chrF scores assigned to segments failing this criterion, so the QE model has clearly learned to attend to this pattern due to this data augmentation.

As expected, the category of leading and trailing spaces is strongly helped by allowing the model’s tokeniser to see the whitespace in question. Finetuning harms performance here, likely not providing a useful signal for this category. After adding augmented examples to the training data the model learns to recognize the leading and trailing spaces difference with 78.2% accuracy.

Regarding the NBSP, merely adding it to the tokeniser harms performance because there is now a new, untrained token in many of the segments. Fine-tuning with MSE helps but does not restore performance to baseline—chrF++ does not distinguish the NBSP either. Here, however, contrastive and multi-task fine-tuning proves effective: Our localisation data clearly contains a number of segments where this type of edit was made, and consistently so. Adding augmented training data further raises performance to 88%. A similar picture emerges for the broader Unicode Spaces rule.

Accuracy on the ‘Extra Period’ category stays in a similar range across training approaches, again suggesting a scarcity of useful signal. This is supported by the result after adding augmentation, which helps the model to perfectly distinguish this phenomenon very well. Additionally, fine-tuning is relatively effective at making the model rate an all-caps translation of an all-caps source above the sentence-case version, but data augmentation is once again necessary to consistently demonstrate the rule.

The Spanish punctuation marks stand out somewhat: The base QE model already shows very high performance on this category. Fine-tuning with MSE hurts compared to the baseline, and even the multi-task training with data augmentation stays slightly behind the baseline, but this is likely noise due to a small sample size.

## F Additional Results

We check the general capabilities of our fine-tuned models on the ACES challenge set (Amrhein et al., 2023), primarily tracking the aggregated ACES-Score. Table 10 shows the ACES-Score (Amrhein

et al., 2023) achieved by fine-tuned metrics, a proxy for their general capabilities. It also lists their preference accuracy on the WMT24 test set. Table 11 lists the dataset sizes of our internal data per language pair, while Table 12 shows per-language-pair performance of fine-tuned metric and QE models on that same data. Tables 14 and 15 break down the per-language-pair performance of our QE models on LocHD.
<table><tr><td>Model</td><td>ACES</td><td>WMT24</td></tr><tr><td>GEMBA-ESA (gpt-4.1)</td><td>10.41</td><td>63.8</td></tr><tr><td>COMETKiwi2A 22</td><td>17.94</td><td>66.2</td></tr><tr><td>+ New Tok</td><td>17.88</td><td>66.2</td></tr><tr><td>Internal data fine-tuning</td><td></td><td></td></tr><tr><td>+ MSE</td><td>13.37</td><td>65.9</td></tr><tr><td> $+ \mathrm { N e w \ T o k + M S E }$ </td><td>14.58</td><td>65.5</td></tr><tr><td>+ MRL</td><td>4.34</td><td>64.5</td></tr><tr><td>+ New  $\mathrm { T o k } + \mathrm { M R L }$ </td><td>3.67</td><td>63.9</td></tr><tr><td>+ New  $\mathrm { T o k } + \mathrm { M S E } + \mathrm { M R L }$ </td><td>11.97</td><td>67.6</td></tr><tr><td>+ New  $\mathrm { T o k } + \mathrm { M S E } + \mathrm { M R L } + \mathrm { A u g }$ </td><td>12.60</td><td>67.2</td></tr><tr><td>ACED fine-tuning</td><td></td><td></td></tr><tr><td>+ MSE</td><td>13.56</td><td>65.8</td></tr><tr><td>+ New  ${ \mathrm { T o k } } + { \mathrm { M S E } }$ </td><td>14.37</td><td>66.2</td></tr><tr><td>+ MRL</td><td>17.24</td><td>67.8</td></tr><tr><td>+ New  $\mathrm { T o k } + \mathrm { M R L }$ </td><td>17.13</td><td>67.5</td></tr><tr><td>+ New  $\mathrm { T o k } + \mathrm { M S E } + \mathrm { M R L }$ </td><td>13.52</td><td>66.2</td></tr><tr><td>+  $\mathrm { N e w \ T o k + M S E + M R L + A u g }$ </td><td>14.01</td><td>66.0</td></tr></table>

Table 10: General metric capabilities, in terms of ACES-Score and preference accuracy (in percent) on WMT24.

<table><tr><td>Dataset</td><td>de-en</td><td>de-es</td><td>de-fr</td><td>en-ar</td><td>en-bg</td><td>en-bn</td><td>en-cs</td><td>en-da</td><td>en-de</td></tr><tr><td>Train (Continuous Scores)</td><td>0.48%</td><td>0.20%</td><td>0.21%</td><td>3.58%</td><td>0.70%</td><td>0.27%</td><td>1.93%</td><td>2.76%</td><td>4.60%</td></tr><tr><td>Train (Preferences)</td><td>0.37%</td><td>0.06%</td><td>0.05%</td><td>3.53%</td><td>0.45%</td><td>0.24%</td><td>1.62%</td><td>1.85%</td><td>4.81%</td></tr><tr><td>Dev (Continuous Scores)</td><td>0.57%</td><td></td><td></td><td>1.55%</td><td>0.04%</td><td>0.13%</td><td>0.55%</td><td>1.70%</td><td>2.84%</td></tr><tr><td>Dev (Preferences)</td><td>0.13%</td><td></td><td>0.32%</td><td>1.58%</td><td>0.58%</td><td>0.17%</td><td>5.32%</td><td>1.26%</td><td>1.45%</td></tr><tr><td>Test (Continuous Scores)</td><td>81</td><td>500</td><td>46</td><td>500</td><td></td><td>500</td><td>500</td><td>500</td><td>500</td></tr><tr><td>Test (Preferences)</td><td>185</td><td>500</td><td>24</td><td>500</td><td></td><td>500</td><td>500</td><td>500</td><td>500</td></tr><tr><td>Dataset</td><td>en-el</td><td>en-es</td><td>en-fa</td><td>en-fi</td><td>en-fr</td><td>en-he</td><td>en-hi</td><td>en-hr</td><td>en-hu</td></tr><tr><td>Train (Continuous Scores)</td><td>1.10%</td><td>5.09%</td><td>0.32%</td><td>5.66%</td><td>4.57%</td><td>0.47%</td><td>1.77%</td><td>0.84%</td><td>0.57%</td></tr><tr><td>Train (Preferences)</td><td>0.68%</td><td>5.00%</td><td>0.21%</td><td>6.03%</td><td>5.15%</td><td>0.67%</td><td>1.91%</td><td>1.42%</td><td>1.15%</td></tr><tr><td>Dev (Continuous Scores)</td><td>2.04%</td><td>1.90%</td><td>0.50%</td><td>0.09%</td><td>2.51%</td><td>2.32%</td><td>1.47%</td><td>0.19%</td><td>1.89%</td></tr><tr><td>Dev (Preferences)</td><td>2.57%</td><td>4.88%</td><td>0.01%</td><td>0.32%</td><td>6.65%</td><td>1.02%</td><td>0.07%</td><td>0.14%</td><td>1.69%</td></tr><tr><td>Test (Continuous Scores)</td><td>500</td><td>500</td><td></td><td>500</td><td>500</td><td>500</td><td>500</td><td></td><td>258</td></tr><tr><td>Test (Preferences)</td><td>500</td><td>500</td><td></td><td>500</td><td>500</td><td>500</td><td>500</td><td></td><td>110</td></tr><tr><td>Dataset</td><td>en-hy</td><td>en-id</td><td>en-it</td><td>en-ja</td><td>en-ko</td><td>en-lt</td><td>en-lv</td><td>en-mr</td><td>en-ms</td></tr><tr><td>Train (Continuous Scores)</td><td>0.10%</td><td>1.85%</td><td>4.04%</td><td>4.40%</td><td>3.97%</td><td>0.26%</td><td>0.25%</td><td>0.19%</td><td>1.29%</td></tr><tr><td>Train (Preferences)</td><td>0.30%</td><td>1.25%</td><td>5.43%</td><td>4.56%</td><td>4.06%</td><td>0.31%</td><td>0.01%</td><td>0.19%</td><td>0.53%</td></tr><tr><td>Dev (Continuous Scores)</td><td></td><td>1.74%</td><td>9.11%</td><td>6.77%</td><td>7.57%</td><td>0.43%</td><td>0.01%</td><td>1.25%</td><td></td></tr><tr><td>Dev (Preferences)</td><td></td><td>6.61%</td><td>3.61%</td><td>7.69%</td><td>14.06%</td><td>0.02%</td><td>0.33%</td><td></td><td>0.47%</td></tr><tr><td>Test (Continuous Scores)</td><td></td><td>500</td><td>500</td><td>500</td><td>500</td><td>95</td><td>500</td><td>5</td><td>61</td></tr><tr><td>Test (Preferences)</td><td></td><td>500</td><td>500</td><td>500</td><td>500</td><td>114</td><td>500</td><td>10</td><td>41</td></tr><tr><td>Dataset</td><td>en-nl</td><td>en-no</td><td>en-pl</td><td>en-pt</td><td>en-ro</td><td>en-ru</td><td>en-sk</td><td>en-sl</td><td>en-so</td></tr><tr><td>Train (Continuous Scores)</td><td>4.78%</td><td>2.97%</td><td>4.38%</td><td>4.23%</td><td>2.18%</td><td>4.18%</td><td>0.46%</td><td>0.31%</td><td>0.25%</td></tr><tr><td>Train (Preferences)</td><td>4.44%</td><td>2.27%</td><td>3.52%</td><td>4.77%</td><td>2.20%</td><td>3.75%</td><td>0.50%</td><td>0.27%</td><td>0.39%</td></tr><tr><td>Dev (Continuous Scores)</td><td>5.29%</td><td>1.58%</td><td>4.12%</td><td>3.43%</td><td>5.64%</td><td>0.85%</td><td>1.34%</td><td></td><td></td></tr><tr><td>Dev (Preferences)</td><td>5.52%</td><td>5.73%</td><td>2.52%</td><td>3.09%</td><td>10.64%</td><td>0.83%</td><td>0.77%</td><td>0.30%</td><td></td></tr><tr><td>Test (Continuous Scores)</td><td>500</td><td>500</td><td>500</td><td>500</td><td>500</td><td>500</td><td>500</td><td></td><td></td></tr><tr><td>Test (Preferences)</td><td>500</td><td>500</td><td>500</td><td>500</td><td>314</td><td>500</td><td>419</td><td></td><td></td></tr><tr><td>Dataset</td><td>en-sq</td><td>en-sv</td><td>en-sw</td><td>en-ta</td><td>en-te</td><td>en-th</td><td>en-tl</td><td>en-tr</td><td>en-uk</td></tr><tr><td>Train (Continuous Scores)</td><td>0.15%</td><td>2.49%</td><td>0.01%</td><td>0.18%</td><td>0.25%</td><td>3.84%</td><td>0.86%</td><td>1.56%</td><td>3.74%</td></tr><tr><td>Train (Preferences)</td><td>0.25%</td><td>1.83%</td><td>0.37%</td><td>0.18%</td><td>0.27%</td><td>4.01%</td><td>0.73%</td><td>2.41%</td><td>3.05%</td></tr><tr><td>Dev (Continuous Scores)</td><td></td><td>7.89%</td><td></td><td>0.67%</td><td></td><td>4.32%</td><td></td><td>6.70%</td><td>0.55%</td></tr><tr><td>Dev (Preferences)</td><td>0.00%</td><td>1.14%</td><td>一</td><td>0.12%</td><td></td><td>1.67%</td><td>0.02%</td><td>0.30%</td><td>0.25%</td></tr><tr><td>Test (Continuous Scores)</td><td></td><td>500</td><td>97</td><td></td><td>373</td><td>500</td><td></td><td>500</td><td>500</td></tr><tr><td>Test (Preferences)</td><td></td><td>500</td><td>62</td><td></td><td>367</td><td>500</td><td></td><td>500</td><td>500</td></tr><tr><td>Dataset</td><td>en-ur</td><td>en-vi</td><td>en-zh-Hans</td><td></td><td>en-zh-Hant</td><td>es-en</td><td>pt-en</td><td></td><td>Total</td></tr><tr><td>Train (Continuous Scores)</td><td>0.16%</td><td>2.07%</td><td></td><td>4.35%</td><td></td><td>1.25%</td><td>0.14%</td><td></td><td>100.0%</td></tr><tr><td>Train (Preferences)</td><td>0.16%</td><td>1.49%</td><td>5.48%</td><td></td><td>3.71% 4.18%</td><td>1.22%</td><td>0.43%</td><td></td><td>100.0%</td></tr><tr><td>Dev (Continuous Scores)</td><td>0.74%</td><td>0.34%</td><td>3.65%</td><td></td><td>2.69%</td><td>2.87%</td><td>0.14%</td><td></td><td>100.0%</td></tr><tr><td>Dev (Preferences)</td><td>0.59%</td><td>1.16%</td><td></td><td>1.55%</td><td>2.86%</td><td>0.00%</td><td></td><td></td><td>100.0%</td></tr><tr><td>Test (Continuous Scores)</td><td></td><td>500</td><td></td><td>500</td><td>500 500</td><td>500 327</td><td>381</td><td></td><td>17,397</td></tr><tr><td>Test (Preferences)</td><td></td><td>500</td><td></td><td>500</td></table>

Table 11: Internal dataset sizes by language pair.

<table><tr><td>Model</td><td>en-ar</td><td>en-de</td><td>en-es</td><td>en-fi</td><td>en-fr</td><td>en-ja</td><td>en-pl</td><td>en-pt</td><td>en-sv</td><td>en-tr</td></tr><tr><td>COMETKiwi2</td><td>52.2</td><td>54.2</td><td>51.4</td><td>54.4</td><td>48.6</td><td>49.0</td><td>50.4</td><td>50.0</td><td>55.0</td><td>51.6</td></tr><tr><td>+ New Tok</td><td>52.8</td><td>53.4</td><td>52.4</td><td>53.6</td><td>47.2</td><td>50.8</td><td>45.4</td><td>49.4</td><td>54.0</td><td>51.0</td></tr><tr><td>+ MSE</td><td>54.0</td><td>49.0</td><td>43.8</td><td>48.0</td><td>42.8</td><td>53.0</td><td>50.4</td><td>46.2</td><td>49.2</td><td>47.6</td></tr><tr><td>+ New Tok + MSE</td><td>51.8</td><td>47.2</td><td>44.0</td><td>48.8</td><td>43.0</td><td>51.2</td><td>51.0</td><td>44.0</td><td>48.4</td><td>48.0</td></tr><tr><td>+ MRL</td><td>73.2</td><td>73.6</td><td>65.0</td><td>64.6</td><td>70.0</td><td>66.6</td><td>67.2</td><td>72.4</td><td>63.4</td><td>57.6</td></tr><tr><td>+ New Tok + MRL</td><td>73.8</td><td>73.2</td><td>67.8</td><td>65.2</td><td>70.0</td><td>68.6</td><td>71.6</td><td>74.6</td><td>65.4</td><td>59.6</td></tr><tr><td>+ New Tok + Multi</td><td>61.6</td><td>58.0</td><td>55.0</td><td>55.8</td><td>54.8</td><td>63.4</td><td>59.8</td><td>59.0</td><td>51.4</td><td>56.4</td></tr><tr><td>+ New Tok + Multi + Aug</td><td>59.4</td><td>53.2</td><td>51.6</td><td>51.2</td><td>52.2</td><td>61.8</td><td>54.2</td><td>55.2</td><td>49.6</td><td>52.6</td></tr><tr><td></td><td>52.2</td><td>54.2</td><td>51.4</td><td>54.4</td><td>48.6</td><td>49.0</td><td>50.4</td><td>50.0</td><td>55.0</td><td>51.6</td></tr><tr><td>+ New Tok</td><td>52.8</td><td>53.4</td><td>52.4</td><td>53.6</td><td>47.2</td><td>50.8</td><td>45.4</td><td>49.4</td><td>54.0</td><td>51.0</td></tr><tr><td>+ MSE</td><td>58.4</td><td>61.2</td><td>52.4</td><td>61.4</td><td>50.4</td><td>56.8</td><td>55.4</td><td>55.2</td><td>55.6</td><td>44.4</td></tr><tr><td>+ New Tok + MSE</td><td>58.8</td><td>58.0</td><td>54.4</td><td>58.6</td><td>48.6</td><td>56.4</td><td>51.0</td><td>53.6</td><td>55.8</td><td>45.6</td></tr><tr><td>+ MRL</td><td>55.8</td><td>59.8</td><td>51.6</td><td>57.2</td><td>51.0</td><td>55.6</td><td>53.6</td><td>52.2</td><td>51.8</td><td>50.6</td></tr><tr><td>+ New Tok + MRL</td><td>55.2</td><td>59.2</td><td>52.0</td><td>55.8</td><td>52.0</td><td>57.0</td><td>51.2</td><td>53.2</td><td>51.8</td><td>50.6</td></tr><tr><td>+ New Tok + Multi</td><td>60.0</td><td>57.8</td><td>53.8</td><td>60.6</td><td>50.0</td><td>59.6</td><td>55.6</td><td>56.0</td><td>55.4</td><td>47.0</td></tr><tr><td>+ New Tok + Multi + Aug</td><td>59.0</td><td>60.2</td><td>54.6</td><td>61.2</td><td>50.8</td><td>58.0</td><td>53.8</td><td>54.4</td><td>57.6</td><td>45.6</td></tr></table>

Table 12: Finetuning on internal data—Preference accuracy on internal data, by language pair.

Table 13: Finetuning on ACED—Preference accuracy on internal data, by language pair.

<table><tr><td>Model</td><td>en-ar</td><td>en-de</td><td>en-es</td><td>en-fi</td><td>en-fr</td><td>en-ja</td><td>en-pl</td><td>en-pt</td><td>en-sv</td><td>en-tr</td></tr><tr><td>COMETKiwi22 i22</td><td>0.175</td><td>0.119</td><td>0.103</td><td>0.133</td><td>0.173</td><td>0.164</td><td>0.136</td><td>0.086</td><td>0.110</td><td>0.235</td></tr><tr><td>+ New Tok</td><td>0.175</td><td>0.130</td><td>0.095</td><td>0.127</td><td>0.156</td><td>0.160</td><td>0.159</td><td>0.098</td><td>0.107</td><td>0.219</td></tr><tr><td>+ MSE</td><td>0.155</td><td>0.137</td><td>0.132</td><td>0.152</td><td>0.225</td><td>0.133</td><td>0.096</td><td>0.110</td><td>0.134</td><td>0.214</td></tr><tr><td>+ New Tok + MSE</td><td>0.151</td><td>0.137</td><td>0.132</td><td>0.132</td><td>0.215</td><td>0.140</td><td>0.102</td><td>0.116</td><td>0.130</td><td>0.208</td></tr><tr><td>+ MRL</td><td>0.069</td><td>0.036</td><td>0.020</td><td>0.102</td><td>0.074</td><td>0.137</td><td>0.057</td><td>-0.000</td><td>0.004</td><td>0.149</td></tr><tr><td>+ New Tok + MRL</td><td>0.023</td><td>0.028</td><td>-0.003</td><td>0.070</td><td>0.018</td><td>0.122</td><td>0.027</td><td>-0.062</td><td>-0.041</td><td>0.128</td></tr><tr><td>+ New Tok + Multi</td><td>0.128</td><td>0.089</td><td>0.100</td><td>0.170</td><td>0.178</td><td>0.124</td><td>0.099</td><td>0.130</td><td>0.081</td><td>0.223</td></tr><tr><td> $+ \mathrm { N e w \ T o k + M u l t i + A u g }$ </td><td>0.121</td><td>0.096</td><td>0.101</td><td>0.159</td><td>0.120</td><td>0.130</td><td>0.111</td><td>0.147</td><td>0.101</td><td>0.214</td></tr></table>

Table 14: Finetuning on internal data—Global Kendall’s τ of QE models with human annotation z-scores, by language pair.

<table><tr><td>Model</td><td>en-ar</td><td>en-de</td><td>en-es</td><td>en-f</td><td>en-fr</td><td>en-ja</td><td>en-pl</td><td>en-pt</td><td>en-sv</td><td>en-tr</td></tr><tr><td>COMETKiwi22A :22</td><td>0.175</td><td>0.119</td><td>0.103</td><td>0.133</td><td>0.173</td><td>0.164</td><td>0.136</td><td>0.086</td><td>0.110</td><td>0.235</td></tr><tr><td>+ New Tok</td><td>0.175</td><td>0.130</td><td>0.095</td><td>0.127</td><td>0.156</td><td>0.160</td><td>0.159</td><td>0.098</td><td>0.107</td><td>0.219</td></tr><tr><td>+ MSE</td><td>0.133</td><td>0.113</td><td>0.086</td><td>0.110</td><td>0.135</td><td>0.148</td><td>0.080</td><td>0.037</td><td>0.123</td><td>0.228</td></tr><tr><td>+ New Tok + MSE</td><td>0.145</td><td>0.138</td><td>0.111</td><td>0.139</td><td>0.131</td><td>0.158</td><td>0.095</td><td>0.076</td><td>0.133</td><td>0.253</td></tr><tr><td>+ MRL</td><td>0.186</td><td>0.133</td><td>0.123</td><td>0.162</td><td>0.253</td><td>0.192</td><td>0.128</td><td>0.122</td><td>0.101</td><td>0.257</td></tr><tr><td>+ New Tok + MRL</td><td>0.180</td><td>0.131</td><td>0.124</td><td>0.148</td><td>0.221</td><td>0.193</td><td>0.143</td><td>0.128</td><td>0.107</td><td>0.249</td></tr><tr><td>+ New Tok + Multi</td><td>0.152</td><td>0.105</td><td>0.098</td><td>0.133</td><td>0.178</td><td>0.147</td><td>0.097</td><td>0.044</td><td>0.088</td><td>0.279</td></tr><tr><td>+ New Tok + Multi + Aug</td><td>0.162</td><td>0.133</td><td>0.112</td><td>0.151</td><td>0.182</td><td>0.157</td><td>0.126</td><td>0.089</td><td>0.127</td><td>0.254</td></tr></table>

Table 15: Finetuning on ACED—Global Kendall’s τ of QE models with human annotation z-scores, by language pair.