# Do Cantonese-Adapted Language Models Better Predict Cantonese Reading? A Cross-Model Eye-Tracking Evaluation

Ziqi Zhang<sup>1</sup> Emmanuele Chersoni<sup>1</sup> Mohammad Momenian<sup>2</sup>

<sup>1</sup>Department of Language Science and Technology, The Hong Kong Polytechnic University {zhangziqi.bram, emmanuelechersoni}@gmail.com

<sup>2</sup>Department of Linguistics, The University of Hong Kong momenian@hku.hk

## Abstract

Information-theoretic measures derived from autoregressive language models are widely used to characterize the expectations that shape human reading, but whether language-varietyspecific training improves such psycholinguistic alignment remains unclear.

This question is still open for Cantonese, where recent NLP evaluations reported mixed benefits from Cantonese-specific training relative to Mandarin-oriented or general-purpose models.

Using naturalistic Cantonese eye-tracking data, we compare two within-family adaptation contrasts: CKIP GPT-2 Tiny versus its lightly Cantonese-adapted JED351 derivative, and Qwen2.5-7B versus CantoneseLLM-7B, which underwent substantially more extensive Cantonese continued pretraining and instruction tuning. From each model, we derive lexical surprisal, POS surprisal, entropy before the target, and entropy reduction. Lexical surprisal and the joint four-metric model consistently favor CantoneseLLM-7B, followed by Qwen2.5-7B, CKIP, and JED351, whereas entropy reduction favors CKIP. These results suggest that more extensive Cantonese-specific training can be associated with stronger predictive fit, while model rankings also depend on the informationtheoretic measure being evaluated.

## 1 Introduction

Readers allocate different amounts of time to different words, making eye tracking a sensitive measure of online language processing (Attardo and Pickering, 2023). A central account of this variation is that processing difficulty increases when a word is unexpected in its context (Hale, 2001; Levy, 2008). Autoregressive language models operationalize this account by assigning a probability distribution to possible continuations. Their probabilities support not only lexical surprisal, but also measures of uncertainty before a word and probabilistic updating after it is observed (Hale, 2016; Pimentel et al.,

2023). Chinese reading, with its distinctive writing system and segmentation properties, has also motivated dedicated models of eye-movement control and word processing (Rayner et al., 2007; Li and Pollatsek, 2020).

Eye-tracking prediction has expanded from individual high-resource languages to multilingual evaluations (Hollenstein et al., 2021a,b, 2022; Salicchi et al., 2022), but regional and under-resourced language varieties remain comparatively underexplored. Cantonese provides an informative case because the value of explicit Cantonese-specific training is itself unsettled in current NLP research. Although Cantonese is widely spoken, its textual resources remain much scarcer than those available for Mandarin, and Cantonese NLP systems frequently rely on transfer from Mandarin-oriented models (Jiang et al., 2025; Min et al., 2025).

Nevertheless, recent evaluations provide mixed evidence on the need for such adaptation. Chinesedeveloped LLMs do not necessarily show broader advantages over Western-developed models for languages spoken in China, with their clearest advantage concentrated in Mandarin (Wang et al., 2026). Cantonese-specific evaluation likewise shows that large general-purpose models can remain highly competitive and that substantial limitations persist in Cantonese linguistic and cultural knowledge (Cheng et al., 2025). By contrast, Cantoneseadapted models have been found to perform best overall on a recent Cantonese NLU benchmark, although Mandarin models remain competitive on several tasks (Min et al., 2025); large-scale Cantonese continued pretraining and supervised finetuning have also produced strong gains on dedicated Cantonese benchmarks (Jiang et al., 2025). Taken together, these results suggest that matching model training to Cantonese does not guarantee a uniform advantage: the benefit appears to depend on the adaptation strategy and on the specific evaluation task.

Existing Cantonese NLP studies primarily evaluate performance on linguistic, reasoning, or cultural benchmarks. It remains unclear whether differences between Cantonese-adapted and more general models extend to the probabilistic expectations that best predict human online language processing. We address this question using the Cantonese component of MCFIX (Li et al., 2023), which contains natural reading (NR) and task-specific reading (TSR) measurements for a Cantonese translation of The Little Prince. We compare two within-family adaptation contrasts: CKIP GPT-2 Tiny Chinese versus its Cantonese-adapted JED351 derivative, and Qwen2.5-7B versus CantoneseLLM-7B, which was developed from Qwen2.5 through Cantoneseoriented continued pretraining and instruction tuning. For each model, we estimate four informationtheoretic predictors: lexical surprisal, POS surprisal, entropy before the target, and entropy reduction. Together, these measures characterize complementary aspects of predictive processing, from uncertainty before the target and lexical or syntacticcategory expectation at the target to changes in uncertainty after the target is observed.

Our study addresses three questions. First, do these information-theoretic measures improve prediction beyond conventional lexical, positional, POS, and dependency predictors? Second, within each model family, does Cantonese-specific training improve predictive value, and does the pattern differ between lighter and more extensive adaptation strategies? Third, do these patterns differ across eye-tracking measures, such as first fixation duration (FFD) and total fixation duration (TFD), and between reading tasks?

The main contributions of this study are threefold: (i) a matched comparison of two small and two large language models arranged as two withinfamily Cantonese-adaptation contrasts; (ii) a joint evaluation of lexical surprisal, POS surprisal, and two entropy-based measures that characterize complementary aspects of predictive processing; and (iii) a five-seed robustness analysis, reading-task comparison, and predictor-block ablations.

## 2 Background

## 2.1 Information-theoretic predictors

We selected four information-theoretic measures to distinguish complementary aspects of model-based expectation around each target word. Entropy before captures uncertainty prior to encountering the target; lexical surprisal quantifies prediction error for the observed word itself; POS surprisal captures prediction error at the syntactic-category level; and entropy reduction measures how observing the target changes uncertainty over the next continuation. We do not assume that these measures map one-toone onto discrete cognitive stages, but together they characterize pre-target anticipation, target-level expectation, and post-target probabilistic updating.

For a target word $w _ { t }$ and its preceding context $c _ { t }$ , lexical surprisal is calculated as:

$$
S ( w _ { t } ) = - \log _ { 2 } P ( w _ { t } \mid c _ { t } ) .\tag{1}
$$

It represents the information conveyed by the observed word: a low-probability continuation has high surprisal. Model-derived lexical surprisal has been widely used to predict reading behavior across languages and eye-tracking settings (Hollenstein et al., 2021b; Salicchi et al., 2023; Wilcox et al., 2023).

POS surprisal abstracts away from word identity. Given the target POS $z _ { t }$ and the lexical candidates assigned to that POS, we estimate the aggregate probability of that category and define

$$
S ( z _ { t } ) = - \log P ( z _ { t } \mid c _ { t } ) .\tag{2}
$$

Unlike a POS dummy, which only records the observed category, POS surprisal quantifies how expected that category is in the current context; related GPT-2-derived POS surprisal has been used to study syntactic-category prediction during natural language comprehension (Heilbron et al., 2022).

Entropy before measures the dispersion of the model’s next-token distribution before the target:

$$
H _ { \mathrm { b e f o r e } } = - \sum _ { v \in V } P ( v \mid c _ { t } ) \log _ { 2 } P ( v \mid c _ { t } ) .\tag{3}
$$

It captures anticipatory uncertainty rather than the unexpectedness of the word that actually occurs, and contextual entropy has been shown to predict reading-time variation beyond lexical surprisal (Pimentel et al., 2023; Wilcox et al., 2023).

Finally, we define the next-token entropy reduction metric as

$$
E R _ { t } = H _ { \mathrm { b e f o r e } } - H _ { \mathrm { a f t e r } } ,\tag{4}
$$

where $H _ { \mathrm { a f t e r } }$ is the entropy of the next-token distribution after adding the current word to the context. This operationalization measures the signed change in uncertainty over next-token continuations and should be distinguished from parse-state entropy reduction in probabilistic grammar. Related entropy-reduction measures have been used to model natural reading and naturalistic language comprehension (Lowder et al., 2018; Song et al., 2024).

## 2.2 Cantonese eye-tracking data

MCFIX provides parallel Mandarin and Cantonese eye-tracking corpora (Li et al., 2023). The Cantonese corpus was collected from 30 native speakers under two reading conditions: NR, in which participants read for comprehension, and TSR, in which they searched the text for specified information. The dataset includes first fixation duration (FFD), second fixation duration (SFD), and total fixation duration (TFD) at the word level.

The dataset also provides lexical, positional, POS, dependency, and neighbourhood-based annotations used as conventional predictors in our baseline model. These variables are motivated by prior work showing effects of frequency, word length, syntactic structure, and spillover on eye movements during Chinese and multilingual reading (Yan et al., 2006; Zang et al., 2018; van Schijndel and Schuler, 2015; Pollatsek et al., 2008). Cantonese frequency and neighbourhood information derives from Cifu (Lai and Winterstein, 2020). Li and colleagues further showed that linguistic and GPT-derived predictors are informative for MCFIX.

## 3 Data and Models

## 3.1 Data and outcomes

We used the traditional-character Cantonese portion of MCFIX. The source contained 5,072 NR rows and 5,071 TSR rows. Removing observations without an analyzable lexical target yielded 10,097 rows, matching the reported Cantonese counts of 5,050 (NR) and 5,047 (TSR). To compare all four language models on identical observations, we took the intersection on which every required metric was available. The frozen sample contained 10,011 task-level rows: 5,007 NR and 5,004 TSR observations. The excluded 86 rows account for less than 1% of the 10,097-row analysis set.

The outcomes were aggregate durations in milliseconds: first fixation duration (FFD), second fixation duration (SFD), and total fixation duration (TFD). NR and TSR observations were analyzed jointly, with reading task included as an indicator predictor. We report both overall cross-validated performance and performance calculated separately on the NR and TSR subsets.

## 3.2 Language models

Table 1 summarizes the comparison. The model pairs were chosen to provide two Cantoneseadaptation contrasts at different scales, keeping the architectures fixed. Within the small-model pair, JED351 descends from CKIP GPT-2 Tiny Chinese.<sup>1</sup> The intermediate JED351 base model<sup>2</sup> patched the CKIP tokenizer and embedding matrix with Cantonese characters but had not yet been trained on Cantonese; the model used here was subsequently fine-tuned for ten epochs on approximately 50 MB of Cantonese Wikipedia text. Within the large-model pair, CantoneseLLM-7B descends directly from Qwen2.5-7B, allowing a comparison between a broadly pretrained Chinese-capable base model and a derivative receiving substantially more extensive Cantonese-specific training.

According to the documentation for CantoneseLLM, the model underwent continued pretraining of Qwen2.5-7B on a large corpus drawn from publicly available Hong Kong news and Cantonese websites, followed by supervised instruction tuning on 75,000 pairs (hon9kon9ize, 2025). The comparisons thus represent substantially different Cantonese-adaptation strategies: JED351 combines vocabulary expansion with fine-tuning on a relatively limited amount of Cantonese text, whereas CantoneseLLM-7B combines extensive Cantonese continued pretraining with supervised instruction tuning.

## 4 Methods

## 4.1 Metric extraction

For each target word, the preceding withinsentence context was used to derive four information-theoretic measures from each language model. Multi-token lexical surprisal was calculated as the sum of the conditional surprisals of the target tokens. Entropy before was defined as the entropy of the model’s next-token distribution given the preceding context, whereas entropy after was computed after the complete target word was added; entropy reduction was defined as their difference. Lexical surprisal and both entropy measures were expressed in bits.

<table><tr><td>Model</td><td>Model family / scale</td><td>Primary characterization</td></tr><tr><td>CKIP</td><td>GPT-2 tiny</td><td>Traditional-Chinese base model for the small-model contrast</td></tr><tr><td>JED351</td><td>GPT-2 tiny</td><td>CKIP-derived; Cantonese vocabulary patching and Wikipedia fine-tuning</td></tr><tr><td>Qwen2.5-7B</td><td>Qwen2.5, 7B</td><td>Multilingual base model for the large-model contrast</td></tr><tr><td>CantoneseLLM-7B</td><td>Qwen2.5, 7B</td><td>Qwen2.5-derived; Cantonese continued pretraining and SFT</td></tr></table>

Table 1: Autoregressive models evaluated in this study, arranged as two within-family Cantonese-adaptation contrasts. Model descriptions follow the released model repositories and documentation, together with the Qwen2.5 technical report (Yang et al., 2024).

POS surprisal was constructed using a common candidate inventory from the Cifu Cantonese lexicon (Lai and Winterstein, 2020). Cifu words were assigned POS categories with PyCantonese (Lee et al., 2022), and the language-model probabilities of candidates sharing the target POS were aggregated before taking negative log probability. Using the same lexical candidate inventory for all four models makes the POS-surprisal construction directly comparable across models.

## 4.2 Regression design

We used CatBoost regression (Prokhorenkova et al., 2018), consistent with the strong performance of gradient-boosting methods with linguistic predictors in prior eye-tracking tasks (Hollenstein et al., 2021a; Salicchi et al., 2022). The no-LLM baseline contained reading task, POS dummies, linear distance to root, linear distance to head, dependency depth, neighbourhood size, current and previous frequency, current and previous syllable count, and relative word position.

For each eye-tracking outcome, we trained: (i) the no-LLM baseline; (ii) four models that added one LLM metric at a time; (iii) a model adding all four metrics; and (iv) leave-one-predictor-blockout models that removed each linguistic or LLM block from the full model. CatBoost used 1,000 iterations and internal random seed 10; all other parameters retained the library defaults.

We used word-level KFold cross-validation with five folds, shuffling, and split seeds 42–46. The same row identities and fold assignments were used for all four language models. For each seed, all reported metrics were calculated from concatenated out-of-fold predictions. Our primary measure was mean absolute error (MAE); we also retained mean squared error, root mean squared error, $R ^ { 2 }$ , Pearson correlation, and Spearman correlation.

For a single-metric addition, improvement was defined as

$$
\Delta \mathrm { M A E } = \mathrm { M A E } _ { \mathrm { b a s e l i n e } } - \mathrm { M A E } _ { \mathrm { a u g m e n t e d } } .\tag{5}
$$

Positive values indicate lower error. In leave-oneout ablation, contribution was defined as the MAE of the model without a block minus the MAE of the full four-metric model.

As FFD, SFD, and TFD have different baseline scales, we additionally computed a baselinenormalized improvement within each split seed:

$$
\% \mathrm { { \Delta \Delta M A E = 1 0 0 } \left( 1 - \frac { M A E _ { a u g } } { M A E _ { b a s e } } \right) . }\tag{6}
$$

We summarize this quantity in the appendix. It helps prevent cross-outcome comparisons from being driven solely by the larger numerical scale of the TFD metric.

## 5 Results

## 5.1 Incremental value of LLM metrics

Table 2 reports the central comparison. Every model–metric combination improves FFD and TFD on average. Results for SFD are smaller: entropy reduction is approximately null for JED351 and Qwen2.5-7B, but clearly positive for CKIP and slightly positive for CantoneseLLM-7B.

The gains are generally stable across split seeds but modest in absolute predictive terms: even the largest joint improvement is below 2% of baseline MAE after normalization. They should therefore be interpreted as incremental predictive contributions rather than large changes in overall accuracy.

Lexical surprisal shows a consistent ordering across all three outcomes: CantoneseLLM-7B yields the largest MAE reduction, followed by Qwen2.5-7B, CKIP, and JED351. For TFD, the corresponding gains are 0.683, 0.599, 0.249, and 0.229 ms. POS surprisal and entropy before also generally favor CantoneseLLM-7B. However, the full four-model ordering is clearest for lexical surprisal and does not generalize uniformly across metrics.

<table><tr><td>Metric</td><td>Measure</td><td>JED351</td><td>CKIP</td><td>Qwen2.5-7B</td><td>CantoneseLLM-7B</td></tr><tr><td>Lexical surprisal</td><td>FFD</td><td>+0.118</td><td>+0.196</td><td>+0.268</td><td>+0.297</td></tr><tr><td rowspan="4">POS surprisal</td><td>SFD</td><td>+0.064</td><td>+0.081</td><td>+0.152</td><td>+0.161</td></tr><tr><td>TFD</td><td>+0.229</td><td>+0.249</td><td>+0.599</td><td>+0.683</td></tr><tr><td>FFD</td><td>+0.118</td><td>+0.200</td><td>+0.276</td><td>+0.308</td></tr><tr><td>SFD</td><td>+0.002</td><td>+0.018</td><td>+0.015</td><td>+0.050</td></tr><tr><td rowspan="4">Entropy before</td><td>TFD</td><td>+0.124</td><td>+0.231</td><td>+0.321</td><td>+0.478</td></tr><tr><td>FFD</td><td>+0.147</td><td>+0.112</td><td>+0.163</td><td>+0.293</td></tr><tr><td>SFD</td><td>+0.008</td><td>+0.042</td><td>+0.016</td><td>+0.097</td></tr><tr><td>TFD</td><td>+0.133</td><td>+0.101</td><td>+0.175</td><td>+0.478</td></tr><tr><td rowspan="3">Entropy reduction</td><td>FFD</td><td>+0.152</td><td>+0.183</td><td>+0.153</td><td>+0.156</td></tr><tr><td>SFD</td><td>-0.008</td><td>+0.104</td><td>-0.003</td><td>+0.009</td></tr><tr><td>TFD</td><td>+0.159</td><td>+0.362</td><td>+0.163</td><td>+0.162</td></tr><tr><td rowspan="3">All four</td><td>FFD</td><td>+0.409</td><td>+0.504</td><td>+0.597</td><td>+0.729</td></tr><tr><td>SFD</td><td>+0.125</td><td>+0.197</td><td>+0.200</td><td>+0.267</td></tr><tr><td>TFD</td><td>+0.409</td><td>+0.651</td><td>+0.821</td><td>+1.251</td></tr></table>

Table 2: Incremental predictive value of each LLM-derived metric across both reading tasks. Values are mean ∆MAE across five split seeds (ms), where $\Delta \mathrm { M A E } = \mathrm { M A E } _ { \mathrm { b a s e l i n e } } - \mathrm { M A E } _ { \mathrm { a u g m e n t e d } } ;$ positive values indicate improvement. The largest gain in each model-comparison row is bold. Full seed-robustness statistics are reported in Appendix Table 3.

Entropy reduction shows a qualitatively different pattern. Although CKIP is not the strongest model for lexical surprisal, its entropy-reduction measure yields the largest mean improvement among all four models for FFD (0.183 ms), SFD (0.104 ms), and TFD (0.362 ms). This dissociation suggests that a model’s usefulness for estimating target-word probability need not coincide with its usefulness for estimating how strongly the target changes uncertainty over upcoming material.

Adding all four measures yields larger improvements than adding any single measure for every model and outcome, consistent with the four metrics providing complementary predictive information. The full CantoneseLLM-7B block reduces MAE by 0.729 ms for FFD, 0.267 ms for SFD, and 1.251 ms for TFD. Within each outcome, the mean joint-model improvements follow the ordering CantoneseLLM-7B > Qwen2.5-7B > CKIP > JED351. Figure 1 shows the same descriptive ordering after normalization by baseline MAE; exact values are reported in Appendix Table 4.

Normalization also changes the cross-outcome interpretation: although TFD shows the largest raw reduction for the two 7B models, the relative fourmetric improvement is largest for FFD in all four models (1.118–1.991%). Thus, the apparent magnitude of an outcome-level gain depends on whether it is expressed in milliseconds or relative to baseline error.

The full means, standard deviations, and numbers of positive split-seed effects are reported in Appendix Table 3.

## 5.2 Comparison between reading tasks

Figure 2 compares performance on the NR and TSR subsets of the out-of-fold predictions; complete single-metric and joint-model results can be seen in Appendix Tables 5 and 6. The taskstratified mean gains show a similar broad ordering: CantoneseLLM-7B produces the largest joint improvement in five of the six task–outcome combinations, and Qwen2.5-7B is generally second. The clearest descriptive task difference concerns FFD. For all four models, the four-metric block yields a larger mean FFD improvement in TSR than in NR; the TSR advantage is 0.296 ms for JED351, 0.114 ms for CKIP, 0.091 ms for Qwen2.5-7B, and 0.187 ms for CantoneseLLM-7B.

The pattern does not generalize uniformly to later measures. CantoneseLLM-7B also yields larger TSR improvements for SFD and TFD, whereas the other three models show slightly or substantially larger NR gains on these outcomes. The contrast is strongest for CKIP, whose joint SFD and TFD improvements are 0.257 and 0.778 ms in NR, compared with 0.138 and 0.523 ms in TSR. Thus, task-stratified differences depend jointly on the eye-tracking measure and the language model: TSR is associated with larger FFD improvements across all four models, whereas SFD and TFD show model-specific patterns.

![](images/f9297ea4f63d08e34519e73ef0f26db9d1e01a1d2141eddd3e13437c105fd778.jpg)

Figure 1: Baseline-normalized improvement for each LLM-derived metric and the joint four-metric model. Each cell is mean %∆MAE across five split seeds; positive values indicate improvement over the no-LLM baseline. Normalization is performed within each model, outcome, and seed before averaging.  
![](images/de3fe90605d9776846479baf6c9207723195b569742d9e19967d5d1b3e517310.jpg)  
Figure 2: Four-metric ∆MAE evaluated separately on NR and TSR observations from the jointly trained models. Points show the five-seed mean and horizontal bars show one standard deviation.

## 5.3 Unique contributions in the full model

Single-metric addition asks whether a metric improves prediction beyond the linguistic baseline, whereas ablation asks whether that metric retains a unique contribution when the other three LLMderived measures are already present. Figure 3 extends the ablation analysis to all linguistic and LLM predictor blocks. Current syllable count and relative word position rank first and second for all four models. Among the LLM-derived blocks, lexical surprisal has the largest mean conditional contribution for Qwen2.5-7B and CantoneseLLM-7B, entropy reduction for CKIP, and entropy before for JED351. Overall, the LLM-derived variables occupy the middle of the ranking, complementing rather than replacing the strongest form- and position-based predictors. Outcome-specific raw LLM ablations are reported in Appendix Table 7.

The conditional contributions of the LLMderived measures are generally smaller and less uniform than their single-metric addition gains, consistent with partial overlap among the four measures. Appendix Figure 6 quantifies this overlap. The strongest and most consistent association is, unsurprisingly, between entropy before and entropy reduction $( \rho = . 5 5 \mathrm { - . 6 7 }$ across models), whereas most other pairwise correlations are weak. CantoneseLLM additionally shows moderate associations of lexical surprisal with entropy before $( \rho = . 5 6 )$ and entropy reduction $( \rho = . 3 8 )$ . The four measures therefore share information without being interchangeable.

![](images/6b17c20cd361b9559eabcf55dd611388b5a1a390c569afa66ad6d9e8a9673759.jpg)  
Figure 3: Full-model leave-one-predictor-block-out contributions. Within each split seed and eye-tracking outcome, contribution is the MAE increase after removal divided by the complete model’s MAE and multiplied by 100. Cells average this percentage across five seeds and the three outcomes; rows are ordered by their mean across models. Positive values indicate that removal worsens prediction, and bold row labels identify LLM-derived metrics.

## 6 Discussion

## 6.1 Do LLM-derived metrics improve prediction beyond conventional predictors?

Our first question asked whether the four information-theoretic measures provide predictive information beyond conventional lexical, positional, POS, and dependency predictors. The results indicate that they do, although the gains are modest. Each metric improves prediction in at least some model–outcome combinations, and the joint four-metric model outperforms every corresponding single-metric model. At the same time, even the largest baseline-normalized joint improvement remains below 2%, indicating that these measures provide incremental rather than transformative gains in predictive accuracy.

The ablation analysis further clarifies their role. Current syllable count and relative word position remain the strongest predictor blocks across models, while the LLM-derived measures generally occupy the middle of the ranking. The informationtheoretic measures therefore complement rather than replace established linguistic predictors. Their conditional contributions are also smaller than their single-metric additions, consistent with partial overlap among the four measures.

These findings support the use of multiple theoretically distinct probability-derived measures when examining the relationship between languagemodel expectations and naturalistic reading behavior.

## 6.2 How does predictive value vary with Cantonese-specific training?

Our second question concerned whether Cantonesespecific training improves predictive value within each model family, and whether the pattern differs between lighter and more extensive adaptation strategies. The results support a qualified association rather than a uniform advantage from Cantonese adaptation. In the small-model comparison, JED351 does not show a general predictive advantage over its CKIP base model despite Cantonese vocabulary expansion and Wikipedia fine-tuning. In contrast, CantoneseLLM-7B, which underwent substantially more extensive Cantonese continued pretraining and supervised instruction tuning, yields the largest mean lexical-surprisal and joint-model improvements across all three eyetracking outcomes.

For the joint four-metric model, the JED351– CKIP contrast is negative for FFD, SFD, and TFD, providing no evidence of an overall advantage from the lighter Cantonese adaptation strategy. The large-model pair shows the opposite pattern: CantoneseLLM-7B exceeds Qwen2.5-7B by 0.132, 0.066, and 0.430 ms for FFD, SFD, and TFD, respectively, with positive contrasts across all five split seeds. Entropy before shows similarly consistent advantages, whereas the entropy-reduction contrasts remain close to zero. Lexical and POS surprisal also favor CantoneseLLM-7B on average, although less uniformly across seeds. The complete within-family contrasts are shown in Appendix Figure 5.

These comparisons do not isolate the effect of Cantonese training from every other training difference between the checkpoints. Within each pair, Cantonese adaptation involves multiple changes, including vocabulary modification, continued pretraining, or instruction tuning. Nor should the stronger performance of the 7B models be interpreted as evidence that scale alone improves psycholinguistic fit: more training and larger models have been reported to worsen surprisal fit in controlled English studies (Oh and Schuler, 2023; Oh et al., 2024), and similar results were found for other languages (e.g. Brazilian Portuguese, Alves (2025)). The present results therefore suggest that more extensive Cantonese-specific training can be associated with stronger predictive fit, while Cantonese adaptation by itself is not sufficient to guarantee an advantage.

## 6.3 Do the effects vary across eye-tracking measures and reading tasks?

Our third question asked whether the predictive patterns vary across eye-tracking outcomes and between natural and task-specific reading. The strongest raw improvements often occur for TFD, but this partly reflects its larger baseline-error scale: after baseline normalization, the joint four-metric improvement is largest for FFD in every model. Because FFD indexes the first fixation whereas SFD records a second fixation, these outcomes plausibly emphasize different stages of processing; second-pass fixation measures can be more sensitive to later structural demands (Conklin and Pellicer-Sánchez, 2016; Li et al., 2023).

The clearest task-related pattern concerns FFD. All four models show larger mean joint-model improvements in TSR than in NR for this measure, which is compatible with goal-directed information search increasing the relevance of lexical expectations during initial processing. However, task-specific conditioning does not necessarily improve fit (Gruteke Klein et al., 2024), and the pattern does not extend uniformly to SFD or TFD. CantoneseLLM-7B shows larger TSR gains for these later measures, whereas the other models often show larger NR gains.

The results suggest that the predictive value of LLM-derived information-theoretic measures may depend on both the reading objective and the eyetracking measure being modeled.

## 7 Conclusion

This study examined whether Cantonese-specific language-model training is associated with stronger prediction of Cantonese reading behavior. Across four information-theoretic measures, LLM-derived predictors provided consistent but modest improvements beyond conventional linguistic predictors. More importantly, Cantonese adaptation did not yield a uniform advantage: the lightly adapted JED351 model did not outperform its CKIP base model overall, whereas the more extensively adapted CantoneseLLM-7B consistently outperformed Qwen2.5-7B for lexical surprisal and the joint four-metric model. At the same time, entropy reduction showed a different cross-model pattern, with CKIP yielding the strongest gains.

Together, these results suggest that psycholinguistic alignment depends not simply on whether a model has been trained on more Cantonese, but on both the adaptation strategy and the aspect of predictive uncertainty being modeled. Better prediction of the upcoming word therefore does not necessarily imply better modeling of how linguistic input updates uncertainty during reading.

## Limitations

The model comparisons do not isolate the effect of Cantonese training. JED351 differs from CKIP in vocabulary, embeddings, and Cantonese Wikipedia fine-tuning, while CantoneseLLM-7B differs from Qwen2.5-7B in continued Cantonese pretraining and instruction tuning; the two pairs also differ in architecture and scale. The results therefore reflect broader adaptation strategies rather than any single training component.

The extraction pipeline tokenizes context and target separately, imposing a lexical boundary that differs from natural concatenated tokenization on 246 of 5,072 Qwen-family rows. The data are further limited to one literary translation and participant sample, and entropy values are not directly comparable across model–tokenizer systems with different continuation spaces.

## References

Diego Alves. 2025. Benchmarking language model surprisal for eye-tracking predictions in Brazilian Portuguese. In Proceedings ofthe First International Workshop on Gaze Data and Natural Language Processing, pages 7–17. INCOMA Ltd., Shoumen, BUL-GARIA.

Salvatore Attardo and Lucy Pickering. 2023. Eye Tracking in Linguistics. Bloomsbury Academic, London.

Tsz Chung Cheng, Chung Shing Cheng, Chaak-ming Lau, Eugene Lam, Wong Chun Yat, Hoi On Yu, and Cheuk Hei Chong. 2025. HKCanto-eval: A benchmark for evaluating Cantonese language understanding and cultural comprehension in LLMs. In Proceedings ofthe 29th Conference on Computational Natural Language Learning, pages 1–11. Association for Computational Linguistics.

Kathy Conklin and Ana Pellicer-Sánchez. 2016. Using eye-tracking in applied linguistics and second language research. Second Language Research, 32(3):453–467.

Keren Gruteke Klein, Yoav Meiri, Omer Shubi, and Yevgeni Berzak. 2024. The effect of surprisal on reading times in information seeking and repeated reading. In Proceedings of the 28th Conference on Computational Natural Language Learning, pages 219–230. Association for Computational Linguistics.

John Hale. 2001. A probabilistic Earley parser as a psycholinguistic model. In Second Meeting of the North American Chapter of the Association for Computational Linguistics.

John Hale. 2016. Information-theoretical complexity metrics. Language and Linguistics Compass, 10(9):397–412.

Micha Heilbron, Kristijan Armeni, Jan-Mathijs Schoffelen, Peter Hagoort, and Floris P. de Lange. 2022. A hierarchy of linguistic predictions during natural language comprehension. Proceedings ofthe National Academy ofSciences, 119(32):e2201968119.

Nora Hollenstein, Emmanuele Chersoni, Cassandra Jacobs, Yohei Oseki, Laurent Prévot, and Enrico Santus.

2022. CMCL 2022 shared task on multilingual and crosslingual prediction of human reading behavior. In Proceedings ofthe Workshop on Cognitive Modeling and Computational Linguistics, pages 121–129. Association for Computational Linguistics.

Nora Hollenstein, Emmanuele Chersoni, Cassandra L. Jacobs, Yohei Oseki, Laurent Prévot, and Enrico Santus. 2021a. CMCL 2021 shared task on eye-tracking prediction. In Proceedings ofthe Workshop on Cognitive Modeling and Computational Linguistics, pages 72–78. Association for Computational Linguistics.

Nora Hollenstein, Federico Pirovano, Ce Zhang, Lena Jäger, and Lisa Beinborn. 2021b. Multilingual language models predict human reading behavior. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 106–123. Association for Computational Linguistics.

hon9kon9ize. 2025. CantoneseLLMChat-v1.0-7B. Hugging Face model repository.

Jiyue Jiang, Alfred Kar Yin Truong, Yanyu Chen, Qinghang Bao, Sheng Wang, Pengan Chen, Jiuming Wang, Lingpeng Kong, Yu Li, and Chuan Wu. 2025. Developing and utilizing a large-scale Cantonese dataset for multi-tasking in large language models. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 1924–1944. Association for Computational Linguistics.

Regine Lai and Grégoire Winterstein. 2020. Cifu: A frequency lexicon of Hong Kong Cantonese. In Proceedings of the Twelfth Language Resources and Evaluation Conference, pages 3069–3077. European Language Resources Association.

Jackson L. Lee, Litong Chen, Charles Lam, Chaak Ming Lau, and Tsz-Him Tsui. 2022. PyCantonese: Cantonese linguistics and NLP in Python. In Proceedings of the Thirteenth Language Resources and Evaluation Conference, pages 6607–6611. European Language Resources Association.

Roger Levy. 2008. Expectation-based syntactic comprehension. Cognition, 106(3):1126–1177.

Junlin Li, Bo Peng, Yu-Yin Hsu, and Emmanuele Chersoni. 2023. Comparing and predicting eye-tracking data of Mandarin and Cantonese. In Proceedings of the Tenth Workshop on NLP for Similar Languages, Varieties and Dialects (VarDial 2023), pages 121– 132. Association for Computational Linguistics.

Xingshan Li and Alexander Pollatsek. 2020. An integrated model of word processing and eye-movement control during Chinese reading. Psychological Review, 127(6):1139–1162.

Matthew W. Lowder, Wonil Choi, Fernanda Ferreira, and John M. Henderson. 2018. Lexical predictability during natural reading: Effects of surprisal and entropy reduction. Cognitive Science, 42(S4):1166– 1183.

Junghyun Min, York Hay Ng, Sophia Chan, Helena Shunhua Zhao, and En-Shiun Annie Lee. 2025. CantoNLU: A benchmark for Cantonese natural language understanding. arXiv preprint arXiv:2510.20670.

Byung-Doh Oh and William Schuler. 2023. Transformer-based language model surprisal predicts human reading times best with about two billion training tokens. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 1915–1921. Association for Computational Linguistics.

Byung-Doh Oh, Shisen Yue, and William Schuler. 2024. Frequency explains the inverse correlation of large language models’ size, training data amount, and surprisal’s fit to reading times. In Proceedings of the 18th Conference ofthe European Chapter ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 2644–2663. Association for Computational Linguistics.

Tiago Pimentel, Clara Meister, Ethan G. Wilcox, Roger P. Levy, and Ryan Cotterell. 2023. On the effect of anticipation on reading times. Transactions of the Association for Computational Linguistics, 11:1624–1642.

Alexander Pollatsek, Barbara J. Juhasz, Erik D. Reichle, Debra Machacek, and Keith Rayner. 2008. Immediate and delayed effects of word frequency and word length on eye movements in reading: A reversed delayed effect of word length. Journal of Experimental Psychology: Human Perception and Performance, 34(3):726–750.

Liudmila Prokhorenkova, Gleb Gusev, Aleksandr Vorobev, Anna Veronika Dorogush, and Andrey Gulin. 2018. CatBoost: Unbiased boosting with categorical features. In Advances in Neural Information Processing Systems 31, pages 6638–6648.

Keith Rayner, Xingshan Li, and Alexander Pollatsek. 2007. Extending the E-Z Reader model of eye movement control to Chinese readers. Cognitive Science, 31(6):1021–1033.

Lavinia Salicchi, Emmanuele Chersoni, and Alessandro Lenci. 2023. A study on surprisal and semantic relatedness for eye-tracking data prediction. Frontiers in Psychology, 14:1112365.

Lavinia Salicchi, Rong Xiang, and Yu-Yin Hsu. 2022. HkAmsters at CMCL 2022 shared task: Predicting eye-tracking data from a gradient boosting framework with linguistic features. In Proceedings of the Workshop on Cognitive Modeling and Computational Linguistics, pages 114–120. Association for Computational Linguistics.

Ming Song, Jing Wang, and Qing Cai. 2024. The unique contribution of uncertainty reduction during naturalistic language comprehension. Cortex, 181:12–25.

Marten van Schijndel and William Schuler. 2015. Hierarchic syntax improves reading time prediction. In Proceedings of the 2015 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 1597–1605. Association for Computational Linguistics.

Andrea Wen-Yi Wang, Unso Eun Seo Jo, and David Mimno. 2026. Do Chinese models speak Chinese languages? In Proceedings ofthe 2026 ACM Conference on Fairness, Accountability, and Transparency, pages 3921–3941. Association for Computing Machinery.

Ethan Gotlieb Wilcox, Tiago Pimentel, Clara Meister, Ryan Cotterell, and Roger Levy. 2023. Testing the predictions of surprisal theory in 11 languages. Transactions of the Association for Computational Linguistics, 11:1451–1470.

Guoli Yan, Hongjie Tian, Xuejun Bai, and Keith Rayner. 2006. The effect of word and character frequency on the eye movements of Chinese readers. British Journal ofPsychology, 97(2):259–268.

An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, and 23 others. 2024. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115.

Chuanli Zang, Ying Fu, Xuejun Bai, Guoli Yan, and Simon P. Liversedge. 2018. Investigating word length effects in Chinese reading. Journal ofExperimental Psychology: Human Perception and Performance, 44(12):1831–1841.

## A Reproducibility Details

The regression analysis uses split seeds 42–46, five shuffled folds per seed, CatBoost internal seed 10, and 1,000 boosting iterations. Code will be released upon acceptance.

## B Supplementary Analyses

<table><tr><td>Metric</td><td>Measure</td><td>JED351</td><td>CKIP</td><td>Qwen2.5-7B</td><td>CantoneseLLM-7B</td></tr><tr><td rowspan="3">Lexical surprisal</td><td>FFD</td><td>+0.118 ± 0.031 +0.196 ± 0.048 +0.268 ± 0.079 (5/5)</td><td>(5/5)</td><td>(5/5)</td><td>+0.297 ± 0.054 (5/5)</td></tr><tr><td></td><td>+0.064 ± 0.016 +0.081 ± 0.020 +0.152 ± 0.030 (5/5)</td><td></td><td></td><td> $+ 0 . 1 6 \dot { 1 } \pm 0 . 0 2 5$ </td></tr><tr><td>SFD</td><td>+0.229 ± 0.036 +0.249 ± 0.112 +0.599 ± 0.043</td><td>(5/5)</td><td>(5/5)</td><td>(5/5) +0.683 ± 0.066</td></tr><tr><td rowspan="3">POS surprisal</td><td>TFD</td><td>(5/5) +0.118 ± 0.057 +0.200 ± 0.035 +0.276 ± 0.032</td><td>(5/5)</td><td>(5/5)</td><td>(5/5) +0.308 ± 0.037</td></tr><tr><td>FFD</td><td>(5/5) +0.002 ± 0.026 +0.018 ± 0.010 +0.015 ± 0.037</td><td>(5/5)</td><td>(5/5)</td><td>(5/5) +0.050 ± 0.020</td></tr><tr><td>SFD</td><td>(3/5)</td><td>(5/5)</td><td>(3/5) +0.124 ± 0.111 +0.231 ± 0.015 +0.321 ± 0.076</td><td>(5/5) +0.478 ± 0.054</td></tr><tr><td rowspan="3">Entropy before</td><td>TFD</td><td>(4/5) +0.147 ± 0.039 +0.112 ± 0.040 +0.163 ± 0.054</td><td>(5/5)</td><td>(5/5)</td><td>(5/5) +0.293 ± 0.023</td></tr><tr><td>FFD</td><td>(5/5) +0.008 ± 0.013 +0.042 ± 0.042 +0.016 ± 0.017</td><td>(5/5)</td><td>(5/5)</td><td>(5/5) +0.097 ± 0.012</td></tr><tr><td>SFD</td><td>(3/5)</td><td>(4/5) +0.133 ± 0.153 +0.101 ± 0.082 +0.175 ± 0.090</td><td>(5/5)</td><td>(5/5) +0.478 ± 0.060</td></tr><tr><td rowspan="3">Entropy reduction FFD</td><td>TFD</td><td>(4/5) +0.152 ± 0.013 +0.183 ± 0.038 +0.153 ± 0.027</td><td>(4/5)</td><td>(5/5)</td><td>(5/5)</td></tr><tr><td></td><td>(5/5) -0.008 ± 0.016 +0.104 ± 0.032 -0.003 ± 0.011</td><td>(5/5)</td><td>(5/5)</td><td>+0.156 ± 0.027 (5/5)</td></tr><tr><td>SFD</td><td>(2/5) +0.159 ± 0.085 +0.362 ± 0.090 +0.163 ± 0.064</td><td>(5/5)</td><td>(2/5)</td><td>+0.009 ± 0.032 (3/5)</td></tr><tr><td rowspan="3">All four</td><td>TFD</td><td>(5/5) +0.409 ± 0.078 +0.504 ± 0.067 +0.597 ± 0.078</td><td>(5/5)</td><td>(5/5)</td><td>+0.162 ± 0.085 (5/5)</td></tr><tr><td>FFD</td><td>(5/5) +0.125 ± 0.025 +0.197 ± 0.039 +0.200 ± 0.030</td><td>(5/5)</td><td>(5/5)</td><td>+0.729 ± 0.074 (5/5)</td></tr><tr><td>SFD</td><td>(5/5) (5/5)</td><td>(5/5) +0.409 ± 0.201 +0.651 ± 0.193 +0.821 ± 0.136 (5/5)</td><td>(5/5) (5/5)</td><td>+0.267 ± 0.034 (5/5) +1.251 ± 0.173</td></tr></table>

Table 3: Five-seed robustness of the pooled single-metric and joint-model additions. Each cell reports mean $\Delta \mathrm { M A E } \pm \mathrm { S D }$ in milliseconds; the second line gives the number of split seeds with $\Delta \mathrm { M A E } > 0$ out of five. The count is a descriptive stability summary, not a significance test.

<table><tr><td>Metric</td><td>Measure</td><td>JED351</td><td></td><td>CKIP</td><td></td><td>Qwen2.5-7B CantoneseLLM-7B</td></tr><tr><td rowspan="3">Lexical surprisal</td><td>FFD</td><td> $+ 0 . 3 2 3 \pm 0 . 0 8 4$ </td><td></td><td> $+ 0 . 5 3 4 \pm 0 . 1 3 1$ </td><td> $+ 0 . 7 3 2 \pm 0 . 2 1 5$ </td><td> $+ 0 . 8 1 0 \pm 0 . 1 4 7$ </td></tr><tr><td>SFD</td><td> $+ 0 . 2 7 1 \pm 0 . 0 6 9$ </td><td> $+ 0 . 3 4 5 \pm 0 . 0 8 4$ </td><td></td><td> $+ 0 . 6 4 9 \pm 0 . 1 2 8$ </td><td> $+ 0 . 6 8 4 \pm 0 . 1 0 5$ </td></tr><tr><td>TFD</td><td> $+ 0 . 3 2 3 \pm 0 . 0 4 9$ </td><td> $+ 0 . 3 5 0 \pm 0 . 1 5 8$ </td><td></td><td> $+ 0 . 8 4 4 \pm 0 . 0 6 3$ </td><td> $+ 0 . 9 6 3 \pm 0 . 0 9 3$ </td></tr><tr><td rowspan="3">POS surprisal</td><td>FFD</td><td> $+ 0 . 3 2 3 \pm 0 . 1 5 4$ </td><td> $+ 0 . 5 4 7 \pm 0 . 0 9 6$ </td><td></td><td> $+ 0 . 7 5 2 \pm 0 . 0 8 6$ </td><td> $+ 0 . 8 4 2 \pm 0 . 0 9 8$ </td></tr><tr><td>SFD</td><td> $+ 0 . 0 0 7 \pm 0 . 1 1 1$ </td><td> $+ 0 . 0 7 6 \pm 0 . 0 4 1$ </td><td></td><td> $+ 0 . 0 6 3 \pm 0 . 1 5 7$ </td><td> $+ 0 . 2 1 3 \pm 0 . 0 8 5$ </td></tr><tr><td>TFD</td><td> $+ 0 . 1 7 5 \pm 0 . 1 5 6$ </td><td> $+ 0 . 3 2 5 \pm 0 . 0 2 2$ </td><td></td><td> $+ 0 . 4 5 3 \pm 0 . 1 0 7$ </td><td> $+ 0 . 6 7 5 \pm 0 . 0 7 6$ </td></tr><tr><td rowspan="3">Entropy before</td><td>FFD</td><td> $+ 0 . 4 0 1 \pm 0 . 1 0 5$ </td><td> $+ 0 . 3 0 6 \pm 0 . 1 0 9$ </td><td></td><td> $+ 0 . 4 4 5 \pm 0 . 1 4 6$ </td><td> $+ 0 . 8 0 1 \pm 0 . 0 6 3$ </td></tr><tr><td>SFD</td><td> $+ 0 . 0 3 3 \pm 0 . 0 5 3$ </td><td> $+ 0 . 1 7 9 \pm 0 . 1 7 7$ </td><td></td><td> $+ 0 . 0 6 9 \pm 0 . 0 7 2$ </td><td> $+ 0 . 4 1 1 \pm 0 . 0 5 1$ </td></tr><tr><td>TFD</td><td> $+ 0 . 1 8 7 \pm 0 . 2 1 5$ </td><td> $+ 0 . 1 4 2 \pm 0 . 1 1 6$ </td><td></td><td> $+ 0 . 2 4 7 \pm 0 . 1 2 6$ </td><td> $+ 0 . 6 7 4 \pm 0 . 0 8 5$ </td></tr><tr><td rowspan="3">Entropy reduction</td><td>FFD</td><td> $+ 0 . 4 1 6 \pm 0 . 0 3 5$ </td><td> $+ 0 . 5 0 0 \pm 0 . 1 0 3$ </td><td></td><td> $+ 0 . 4 1 7 \pm 0 . 0 7 5$ </td><td> $+ 0 . 4 2 6 \pm 0 . 0 7 3$ </td></tr><tr><td>SFD</td><td> $- 0 . 0 3 2 \pm 0 . 0 6 7$ </td><td></td><td> $+ 0 . 4 4 3 \pm 0 . 1 3 6$ </td><td> $- 0 . 0 1 2 \pm 0 . 0 4 5$ </td><td> $+ 0 . 0 3 9 \pm 0 . 1 3 7$ </td></tr><tr><td>TFD</td><td> $+ 0 . 2 2 5 \pm 0 . 1 2 0$ </td><td></td><td> $+ 0 . 5 1 1 \pm 0 . 1 2 6$ </td><td> $+ 0 . 2 3 0 \pm 0 . 0 9 0$ </td><td> $+ 0 . 2 2 9 \pm 0 . 1 1 9$ </td></tr><tr><td rowspan="3">All four</td><td>FFD</td><td> $+ 1 . 1 1 8 \pm 0 . 2 1 2$ </td><td> $+ 1 . 3 7 7 \pm 0 . 1 8 3$ </td><td></td><td> $+ 1 . 6 3 1 \pm 0 . 2 0 8$ </td><td> $+ 1 . 9 9 1 \pm 0 . 2 0 0$ </td></tr><tr><td>SFD</td><td> $+ 0 . 5 3 4 \pm 0 . 1 0 4$ </td><td></td><td> $+ 0 . 8 3 9 \pm 0 . 1 6 4$ </td><td> $+ 0 . 8 5 3 \pm 0 . 1 2 7$ </td><td> $+ 1 . 1 3 5 \pm 0 . 1 4 7$ </td></tr><tr><td>TFD</td><td> $+ 0 . 5 7 7 \pm 0 . 2 8 2$ </td><td></td><td> $+ 0 . 9 1 7 \pm 0 . 2 7 2$ </td><td> $+ 1 . 1 5 7 \pm 0 . 1 9 0$ </td><td> $+ 1 . 7 6 4 \pm 0 . 2 4 4$ </td></tr></table>

Table 4: Baseline-normalized improvement for each LLM-derived metric. Values are mean percentage ∆MAE ± SD across the five split seeds, calculated within seed as $1 0 0 ( \mathrm { M A E _ { b a s e l i n e } - M A E _ { a u g m e n t e d } ) / M A E _ { b a s e l i n e } } .$ . Positive values indicate improvement.

<table><tr><td>Metric</td><td>Measure</td><td>JED351</td><td>CKIP</td><td>Qwen2.5-7B</td><td>CantoneseLLM-7B</td></tr><tr><td rowspan="3">Lexical surprisal</td><td>FFD</td><td>+0.045 ± 0.072 +0.204 ± 0.093 +0.239 ± 0.094</td><td></td><td>(5/5)</td><td>+0.237 ± 0.069 (5/5)</td></tr><tr><td></td><td>(4/5) +0.063 ± 0.012 +0.105 ± 0.034 +0.147 ± 0.025</td><td>(5/5)</td><td></td><td>+0.131 ± 0.024</td></tr><tr><td>SFD</td><td>(5/5)</td><td>(5/5) +0.115 ± 0.152 +0.261 ± 0.181 +0.555 ± 0.110</td><td>(5/5)</td><td>(5/5) +0.625 ± 0.117</td></tr><tr><td rowspan="3">POS surprisal</td><td>TFD</td><td>(3/5) +0.066 ± 0.082 +0.185 ± 0.069 +0.274 ± 0.025</td><td>(5/5)</td><td>(5/5)</td><td>(5/5) +0.309 ± 0.057</td></tr><tr><td>FFD</td><td>(4/5) +0.012 ± 0.031 +0.016 ± 0.044 +0.030 ± 0.042</td><td>(5/5)</td><td>(5/5)</td><td>(5/5) +0.061 ± 0.024</td></tr><tr><td>SFD</td><td>(3/5)</td><td>(4/5)</td><td>(4/5)</td><td>(5/5)</td></tr><tr><td rowspan="3">Entropy before</td><td>TFD</td><td>(4/5)</td><td>(5/5)</td><td>+0.147 ± 0.175+0.213 ± 0.059+0.299 ± 0.147 (5/5)</td><td>+0.473 ± 0.048 (5/5)</td></tr><tr><td>FFD</td><td>(5/5)</td><td>(5/5)</td><td>+0.140 ± 0.076 +0.083 ± 0.030 +0.139 ± 0.097 (5/5)</td><td>+0.248 ± 0.046 (5/5)</td></tr><tr><td>SFD</td><td>(2/5)</td><td>(4/5)</td><td>-0.001 ± 0.029 +0.058 ± 0.059 +0.035 ± 0.032 (5/5)</td><td>+0.089 ± 0.031 (5/5)</td></tr><tr><td rowspan="3">Entropy reduction FFD</td><td>TFD</td><td>(5/5)</td><td>+0.138 ± 0.089 +0.133 ± 0.152 +0.188 ± 0.132 (4/5)</td><td>(5/5)</td><td>+0.452 ± 0.111</td></tr><tr><td></td><td>(5/5)</td><td>+0.180 ± 0.026 +0.201 ± 0.029 +0.177 ± 0.051 (5/5)</td><td>(5/5)</td><td>(5/5) +0.152 ± 0.038 (5/5)</td></tr><tr><td>SFD</td><td>(5/5)</td><td>+0.032 ± 0.020 +0.127 ± 0.057 +0.028 ± 0.019 (5/5)</td><td>(5/5)</td><td>+0.027 ± 0.039 (4/5)</td></tr><tr><td rowspan="3">All four</td><td>TFD</td><td>(5/5)</td><td>+0.394 ± 0.189 +0.443 ± 0.081 +0.254 ± 0.164 (5/5)</td><td>(5/5)</td><td>+0.243 ± 0.123 (5/5)</td></tr><tr><td>FFD</td><td>(5/5)</td><td>+0.262 ± 0.096+0.447 ± 0.084+0.552 ± 0.073 (5/5)</td><td>(5/5)</td><td>+0.636 ± 0.097 (5/5)</td></tr><tr><td>SFD</td><td>(5/5)</td><td>(5/5)</td><td>+0.138 ± 0.015 +0.257 ± 0.071 +0.223 ± 0.035 (5/5)</td><td>+0.242 ± 0.055 (5/5)</td></tr><tr><td></td><td>TFD</td><td>+0.421 ± 0.196 +0.778 ± 0.150 +0.849 ± 0.156 (5/5)</td><td>(5/5)</td><td>(5/5)</td><td>+1.186 ± 0.127 (5/5)</td></tr></table>

Table 5: Complete NR task-stratified additions from the pooled models. Each cell reports mean $\Delta \mathbf { M } \mathbf { A } \mathbf { E } \pm \mathbf { S } \mathbf { D }$ in milliseconds, followed by the number of positive split-seed effects out of five. These are evaluations of NR out-of-fold observations from jointly trained NR+TSR models, not separately trained NR-only models.

![](images/7d3fdcd509a778b322c406891062ab0928285d511febaba067acc7499e82b7f7.jpg)

![](images/182ebcf6a0eaafff04712c7c8755627d4785f5c4e96f748603e40ab2bf8864f5.jpg)

![](images/239861b6f5eaa100ba42b9deff2048009c49532f074f5b885f47cfd6c707b441.jpg)  
(a) Lexical surprisal

![](images/369f3ed2072ba36f9de4f56126d559e06d5429168f18b3525248455b7dbd5446.jpg)

![](images/3d1366e1e1edd9439947a0dddec9c3daf07bfd31383803e6b547699459b4e00b.jpg)

![](images/bf21357624d1e837f26a35cbedce1251baf020f966be3e2dd74df9e1dc6cd2fd.jpg)  
(b) POS surprisal

![](images/47e70947050213dd4c16f441b3b9b279ad6242bc597809da2c267c221b4ff14e.jpg)  
(c) Entropy before

![](images/77b407bf7c423628dd6c64e30d94a49cf84952050a93d41cf2ba05cb0a9f4339.jpg)  
(d) Entropy reduction  
Figure 4: Seed-level ∆MAE distributions for the four LLM-derived metrics. Each panel compares JED351, CKIP, Qwen2.5-7B, and CantoneseLLM-7B. Each box contains the five split-seed values, which are also shown as individual points. Positive values indicate improvement over the no-LLM baseline.

![](images/a5a44d02656da62f1185f36d99cc21f7a281eb7728b0d75c674a45fd891133dc.jpg)  
Paired adaptation contrast, ΔΔMAE (ms)  
Negative: base model larger gain Positive: Cantonese-adapted model larger gain

Figure 5: Within-family Cantonese-adaptation contrasts. Values show the paired difference in ∆MAE between each Cantonese-adapted model and its base counterpart across the five split seeds. Negative values indicate larger predictive gains for the base model, whereas positive values indicate larger gains for the Cantonese-adapted model.
<table><tr><td>Metric</td><td>Measure</td><td>JED351</td><td>CKIP</td><td>Qwen2.5-7B</td><td>CantoneseLLM-7B</td></tr><tr><td rowspan="3">Lexical surprisal</td><td></td><td>+0.192 ± 0.037 +0.188 ± 0.018 +0.298 ± 0.071</td><td></td><td></td><td>+0.356 ± 0.067</td></tr><tr><td>FFD</td><td>(5/5) +0.064 ± 0.025 +0.057 ± 0.030 +0.157 ± 0.040</td><td>(5/5)</td><td>(5/5)</td><td>(5/5) +0.190 ± 0.057</td></tr><tr><td>SFD</td><td>(5/5)</td><td>(5/5)</td><td>(5/5)</td><td>(5/5)</td></tr><tr><td rowspan="4">POS surprisal</td><td>TFD</td><td>+0.344 ± 0.112 +0.237 ± 0.210 +0.642 ± 0.085 (5/5)</td><td>(4/5)</td><td>(5/5)</td><td>+0.741 ± 0.098 (5/5)</td></tr><tr><td>FFD</td><td>(5/5)</td><td>+0.171 ± 0.035+0.216 ± 0.043 +0.277 ± 0.045</td><td></td><td>+0.308 ± 0.038 (5/5)</td></tr><tr><td>SFD</td><td>(2/5)</td><td>(5/5) -0.009 ± 0.044 +0.020 ± 0.047 +0.000 ± 0.039</td><td>(5/5)</td><td>+0.039 ± 0.034</td></tr><tr><td>TFD</td><td>(3/5)</td><td>(3/5)</td><td>(2/5) +0.101 ± 0.146 +0.248 ± 0.081 +0.344 ± 0.060</td><td>(5/5) +0.484 ± 0.105</td></tr><tr><td rowspan="3">Entropy before</td><td></td><td></td><td>(5/5) +0.154 ± 0.045 +0.140 ± 0.063 +0.186 ± 0.040</td><td>(5/5)</td><td>(5/5) +0.339 ± 0.049</td></tr><tr><td>FFD</td><td>(5/5)</td><td>(5/5) +0.017 ± 0.031 +0.026 ± 0.034 -0.003 ± 0.045</td><td>(5/5)</td><td>(5/5) +0.105 ± 0.028</td></tr><tr><td>SFD</td><td>(4/5)</td><td>(4/5) +0.127 ± 0.229 +0.068 ± 0.161 +0.162 ± 0.115</td><td>(4/5)</td><td>(5/5) +0.504 ± 0.137</td></tr><tr><td rowspan="3"></td><td>TFD</td><td>(4/5) +0.125 ± 0.011 +0.165 ± 0.066 +0.129 ± 0.043</td><td>(3/5)</td><td>(5/5)</td><td>(5/5)</td></tr><tr><td>Entropy reduction FFD</td><td>(5/5)</td><td>(5/5) -0.047 ± 0.034 +0.081 ± 0.029 -0.033 ± 0.029</td><td>(5/5)</td><td>+0.159 ± 0.030 (5/5) -0.008 ± 0.045</td></tr><tr><td>SFD</td><td>(1/5)</td><td>(5/5) -0.076 ± 0.117 +0.281 ± 0.218 +0.072 ± 0.128</td><td>(0/5)</td><td>(2/5)</td></tr><tr><td rowspan="3">All four</td><td>TFD</td><td>(2/5)</td><td>(4/5)</td><td>(3/5)</td><td>+0.081 ± 0.081 (4/5)</td></tr><tr><td>FFD</td><td>(5/5)</td><td>(5/5)</td><td>+0.557 ± 0.074 +0.562 ± 0.080 +0.643 ± 0.093 (5/5) +0.113 ± 0.042 +0.138 ± 0.017 +0.178 ± 0.080</td><td>+0.823 ± 0.069 (5/5)</td></tr><tr><td>SFD</td><td>(5/5) (5/5)</td><td>(5/5) +0.397 ± 0.280 +0.523 ± 0.270 +0.793 ± 0.136 (5/5)</td><td>(5/5) (5/5)</td><td>+0.292 ± 0.070 (5/5) +1.317 ± 0.279</td></tr></table>

Table 6: Complete TSR task-stratified additions from the pooled models. Each cell reports mean ∆MAE ± SD in milliseconds, followed by the number of positive split-seed effects out of five. These are evaluations of TSR out-of-fold observations from jointly trained NR+TSR models, not separately trained TSR-only models.

<table><tr><td>Removed block</td><td>Measure</td><td>JED351</td><td>CKIP</td><td></td><td>Qwen2.5-7B CantoneseLLM-7B</td></tr><tr><td rowspan="3">Lexical surprisal</td><td>FFD</td><td> $+ 0 . 0 5 0 \pm 0 . 0 2 9$ </td><td> $+ 0 . 0 8 3 \pm 0 . 0 2 6$ </td><td> $+ 0 . 0 8 8 \pm 0 . 0 1 7$ </td><td> $+ 0 . 0 8 3 \pm 0 . 0 7 2$ </td></tr><tr><td>SFD</td><td> $+ 0 . 0 7 3 \pm 0 . 0 2 0$ </td><td> $+ 0 . 0 6 1 \pm 0 . 0 2 4$ </td><td> $+ 0 . 1 2 0 \pm 0 . 0 3 6$ </td><td> $+ 0 . 1 1 5 \pm 0 . 0 1 9$ </td></tr><tr><td>TFD</td><td> $+ 0 . 0 2 0 \pm 0 . 1 9 0$ </td><td> $+ 0 . 1 6 6 \pm 0 . 1 3 4$ </td><td> $+ 0 . 3 1 3 \pm 0 . 0 8 3$ </td><td> $+ 0 . 2 6 9 \pm 0 . 1 4 7$ </td></tr><tr><td rowspan="3">POS surprisal</td><td>FFD</td><td> $+ 0 . 0 2 7 \pm 0 . 0 6 4$ </td><td> $+ 0 . 0 8 4 \pm 0 . 0 5 2$ </td><td> $+ 0 . 0 9 7 \pm 0 . 0 5 3$ </td><td> $+ 0 . 1 4 6 \pm 0 . 0 5 9$ </td></tr><tr><td>SFD</td><td> $+ 0 . 0 0 1 \pm 0 . 0 2 7$ </td><td> $+ 0 . 0 2 6 \pm 0 . 0 2 8$ </td><td> $- 0 . 0 0 2 \pm 0 . 0 3 2$ </td><td> $+ 0 . 0 2 9 \pm 0 . 0 2 8$ </td></tr><tr><td>TFD</td><td> $- 0 . 0 0 3 \pm 0 . 1 3 4$ </td><td> $+ 0 . 1 1 2 \pm 0 . 1 5 7$ </td><td> $+ 0 . 0 3 8 \pm 0 . 1 4 1$ </td><td> $+ 0 . 1 8 8 \pm 0 . 0 6 0$ </td></tr><tr><td rowspan="3">Entropy before</td><td>FFD</td><td> $+ 0 . 0 9 1 \pm 0 . 0 5 3$ </td><td> $+ 0 . 0 7 0 \pm 0 . 0 4 6$ </td><td> $+ 0 . 0 6 3 \pm 0 . 0 6 5$ </td><td> $+ 0 . 0 7 2 \pm 0 . 0 4 3$ </td></tr><tr><td>SFD</td><td> $+ 0 . 0 5 0 \pm 0 . 0 1 5$ </td><td> $+ 0 . 0 2 9 \pm 0 . 0 2 3$ </td><td> $+ 0 . 0 2 3 \pm 0 . 0 2 5$ </td><td> $+ 0 . 0 5 2 \pm 0 . 0 2 6$ </td></tr><tr><td>TFD</td><td> $+ 0 . 1 2 3 \pm 0 . 1 2 2$ </td><td> $- 0 . 0 2 1 \pm 0 . 1 3 7$ </td><td> $- 0 . 0 2 9 \pm 0 . 1 1 8$ </td><td> $+ 0 . 2 1 5 \pm 0 . 0 4 4$ </td></tr><tr><td rowspan="3">Entropy reduction</td><td>FFD</td><td> $+ 0 . 1 0 2 \pm 0 . 0 3 0$ </td><td> $+ 0 . 1 4 8 \pm 0 . 0 4 3$ </td><td> $+ 0 . 1 4 3 \pm 0 . 0 4 4$ </td><td> $+ 0 . 1 6 3 \pm 0 . 0 8 8$ </td></tr><tr><td>SFD</td><td> $+ 0 . 0 1 6 \pm 0 . 0 1 8$ </td><td> $+ 0 . 0 7 9 \pm 0 . 0 1 8$ </td><td> $+ 0 . 0 4 7 \pm 0 . 0 3 3$ </td><td> $+ 0 . 0 3 6 \pm 0 . 0 4 3$ </td></tr><tr><td>TFD</td><td> $+ 0 . 1 3 2 \pm 0 . 0 9 9$ </td><td> $+ 0 . 1 8 5 \pm 0 . 0 9 6$ </td><td> $+ 0 . 0 8 6 \pm 0 . 1 9 5$ </td><td> $+ 0 . 2 1 3 \pm 0 . 1 0 7$ </td></tr></table>

Table 7: Leave-one-LLM-metric-out ablation from each full four-metric model. Values are mean increases in MAE after removal ± SD across split seeds (ms); positive values indicate a unique contribution conditional on all other predictors.

![](images/7c05aa084554504c9873b5ba4907d3ad0b854e86626a6fe68f7206acc6d091b2.jpg)

![](images/b08c5eb30f5c0edcc7608b1cb0d357e40c7102715e2071f286ff349a73004ee9.jpg)

![](images/84f51af9207960602bb4fbb80990dd85ee7d53b02bd4eca8bdbbff5cc734606e.jpg)  
Figure 6: Within-model Spearman correlations among the four LLM-derived metrics. Correlations are calculated over 5,007 unique canonical text positions after collapsing the duplicated NR and TSR observations. The fullprecision coefficients are provided in the supplementary CSV file.