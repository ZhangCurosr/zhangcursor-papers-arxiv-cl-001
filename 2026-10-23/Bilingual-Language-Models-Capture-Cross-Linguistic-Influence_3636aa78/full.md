# Bilingual Language Models Capture Cross-Linguistic Influence in Second-Language Reading

Abderrahmane Issam<sup>1</sup>, Irene E. Winther<sup>2</sup>, Stefan L. Frank<sup>2</sup>

<sup>1</sup>Maastricht University, <sup>2</sup>Radboud University abderrahmane.issam@maastrichtuniversity.nl {irene.winther, stefan.frank}@ru.nl

## Abstract

While monolingual language model (LM) surprisal is widely established as a predictor of reading times, it cannot capture the non-native (L2) reading dynamics that are shaped by the reader’s first language (L1). In this work, we investigate whether the cross-lingual transfer within bilingual models aligns with human L1- driven biases during L2 reading. We evaluate bilingual models on eye-tracking data of read ers from 12 diverse L1 backgrounds reading English texts. To isolate L1-specific transfer, we compare bilingual models trained on an L1 that matches the readers’ L1 background to both a monolingual English baseline and bilingual models trained on other L1s. We demonstrate that matching the readers’ L1 improves the prediction of L2 reading times. This predictive advantage is concentrated in late-stage eye-tracking measures, modulated by L1-L2 script overlap, and peaks among intermediateproficiency readers. These findings provide evidence that bilingual LMs capture aspects of cross-linguistic influence, and that such influence is present in L2 reading behavior.

## 1 Introduction

Language Models (LMs) are increasingly being investigated for their alignment with human cognitive processing (Armeni et al., 2017; Karamolegkou et al., 2023; Niu et al., 2026; Futrell and Mahowald, 2026). Within this domain, a prominent line of work evaluates LMs on eye-tracking data (Demberg and Keller, 2008; Frank, 2009; Goodkind and Bicknell, 2018; Oh and Schuler, 2023; Shain et al., 2024; Wilcox et al., 2023), resting on the wellestablished assumption that fixation durations reflect cognitive effort during sentence processing (Rayner, 1998; Schotter and Dillon, 2025). Under surprisal theory (Hale, 2001; Levy, 2008), the cognitive effort of processing a word is tied its predictability in context (surprisal). While substantial work shows that LM surprisal successfully predicts native-language (L1) reading times (Smith and Levy, 2013; Goodkind and Bicknell, 2018; Wilcox et al., 2020; Shain et al., 2024), much less is known regarding the ability of bilingual LMs to predict reading behavior in a non-native language (L2).

One candidate source of difference between native and non-native reading is cross-linguistic influence from the reader’s L1 (Kim, 2025; Alemán Bañón and Martin, 2021; Odlin, 1989; Tokowicz and MacWhinney, 2005; Tokowicz et al., 2024; Foucart, 2021). If such influence actively shapes L2 reading, a monolingual LM trained solely on the target language cannot, by definition, capture it.

In parallel, research in NLP demonstrates that multilingual models learn shared cross-lingual representations (Pires et al., 2019; Libovický et al., 2020; de Varda and Marelli, 2023), enabling zeroshot performance transfer across languages (Wu and Dredze, 2019; Conneau et al., 2020b,a). However, it remains an open question whether these representations translate into predictive biases that align with human L2 reading behavior.

In this work, we investigate whether bilingual LMs’ cross-lingual transfer aligns with human cross-linguistic influence during L2 reading. We evaluate bilingual GPT-2 models on the Multilingual Eye Movement Corpus (MECO L2) (Kuperman et al., 2025), using eye-tracking recordings from participants across 12 diverse L1 backgrounds reading English texts. To examine whether the duration of L1 training affects L2 prediction in the models, we train bilingual models across multiple Step of Exposure (SoE) stages, varying the moment during L1 learning at which L2 English is introduced. Hence, we manipulate the duration of initial L1 training and the amount of L2 exposure. We evaluate model surprisal across four readingtime metrics (first fixation duration, gaze duration, go-past time, and total fixation duration) to distinguish early lexical access from late-stage structural reanalysis and text integration (Rayner, 1998; Liversedge and Findlay, 2000; Inhoff, 1984).

Our investigation addresses two primary questions. First, we establish whether bilingual models’ surprisal explains variance in L2 reading times that a monolingual baseline cannot. Second, we compare L1-matched bilingual models (where the model’s initial training language matches the reader’s L1) to L1-mismatched controls (where the model is trained on a different L1) to determine whether this predictive advantage is driven by language-specific transfer rather than a generic multilingual effect.

We find a bilingual predictive advantage concentrated in late-stage reading measures, with little to no advantage in early ones. Furthermore, L1- matched models outperform mismatched models, indicating that the explained variance in L2 reading times is L1-specific. Finally, we observe that this advantage is modulated by L1-L2 script overlap and peaks for L2 readers with intermediate proficiency. In summary, these findings provide evidence that bilingual LMs are capable of modeling cross-linguistic influence, and that such influence is present in human L2 reading behavior.

## 2 Related Works

## 2.1 Bilingual Models for L2 Acquisition

The success of multilingual language models has prompted research into whether their internal crosslingual transfer mechanisms align with human cross-linguistic influence (Mueller et al., 2020; Papadimitriou and Jurafsky, 2020; Oba et al., 2023; Aoyama and Schneider, 2024; Yadavalli et al., 2023; Issam et al., 2026; Salhan et al., 2026). To simulate the dynamics of L2 acquisition, previous studies have modeled bilingual exposure by training on unbalanced L1 and L2 data (Dhar and Bisazza, 2021; Winther et al., 2021), introducing the L2 sequentially (Winther et al., 2021; Constantinescu et al., 2025; Arnett et al., 2025; Issam et al., 2026), freezing L1 representations during L2 exposure (Papadimitriou and Jurafsky, 2020; Yadavalli et al., 2023; Aoyama and Schneider, 2024), or using specialized training objectives (Oba et al., 2023). Previous research on bilingual LMs has investigated various effects of bilingual exposure, including syntactic transfer (Oba et al., 2023; Yadavalli et al., 2023; Issam et al., 2026), crosslinguistic priming (Frank, 2021; Michaelov et al.,

2023; Arnett et al., 2025), and L1 influence on L2 reading times (Aoyama and Schneider, 2024; Salhan et al., 2026). Building on this, we adopt the sequential training strategy of Issam et al. (2026) to train bilingual models for several L1s, and evaluate their ability to predict human L2 reading times.

## 2.2 Evaluating LMs on L2 Reading Times

While most work evaluating language models on eye-tracking data focuses on L1 reading (Demberg and Keller, 2008; Frank, 2009; Goodkind and Bicknell, 2018; Oh and Schuler, 2023; Shain et al., 2024; Wilcox et al., 2023), the release of bilingual speaker corpora such as CELER (Berzak et al., 2022) and MECO L2 (Kuperman et al., 2025) has enabled the study of non-native processing (de Varda and Marelli, 2022; Berzak and Levy, 2023; Aoyama and Schneider, 2024; Salhan et al., 2026). Most of this work, however, still relies on surprisal from monolingual models for the prediction of L2 reading times.

A smaller body of work has explored bilingual models for this purpose, but critical gaps remain. Salhan et al. (2026) lacked monolingual baselines and L1-mismatched controls, making it impossible to isolate L1-specific transfer. While Aoyama and Schneider (2024) included a monolingual baseline, they relied on frozen L1 representations and compared independent model fits rather than testing for additional variance explained. Furthermore, both studies omitted L1 control groups and restricted their evaluations primarily to early reading metrics like gaze duration. Our approach systematically addresses these limitations. First, we train fully unfrozen models, allowing bilingual representations to dynamically interact. Second, we measure each bilingual model’s variance explained over a monolingual baseline and test whether L1-matched surprisal explains variance beyond L1-mismatched surprisal within the same regression analysis. Empirically, we broaden the scope by evaluating L1 and L2 readers across four eye-tracking measures and 12 L1s, further examining how this predictive advantage is modulated by reader proficiency and script overlap.

## 3 Methods

## 3.1 Bilingual LM Training

We follow the training approach of Issam et al. (2026) to train bilingual GPT-2 LMs for each L1- L2 language pair. We acquire multilingual data from FineWeb (Penedo et al., 2024), using the 10- billion token sample (sample-10BT) for English and the language-specific splits from FineWeb-v2 (Penedo et al., 2025) for the L1s. We follow the approach of Issam et al. (2026) to minimize codeswitching by filtering out sentences using sentenceand word-level language detection. While completely mitigating code-switching remains an open challenge in NLP (Burchell et al., 2024), this filtering step is crucial. Because code-switching is known to enhance cross-lingual transfer (Qin et al., 2020; Wang et al., 2025), leaving it unfiltered could inflate the observed L1 influence and contaminate the monolingual baseline.

We vary the Step of Exposure (SoE) as the step at which we introduce the L2 to study how extended L1 training and delayed L2 exposure affect the prediction of L2 reading data. Increasing the SoE results in longer initial L1 training and less L2 training. Since we train all models for the same total amount of training steps, this mimics a comparison between bilingual speakers of the same age but with different ages of first exposure to the L2. Once the L2 is introduced, we continue training on the L1 data by interleaving L1 and L2 sequences. This prevents catastrophic forgetting and maintains the L1 influence on the L2 (Constantinescu et al., 2025; Arnett et al., 2025). Furthermore, it mirrors a realistic scenario where non-native speakers maintain strong knowledge of their L1, either due to critical period effects (Lenneberg, 1967; Bylund, 2019) or persistent L1 use during L2 learning.

We train a SentencePiece tokenizer (Kudo and Richardson, 2018) for each L1-L2 pair using a balanced subset of 10 million lines (5 million lines per language). Once trained, this tokenizer is used to process the full corpus of each language independently, packing the text into shuffled sequences of 256 tokens. Finally, from these fully tokenized corpora, we sample one billion tokens per language for LM training and one million tokens for evaluation.

We train all models for a total of 64k training steps. During monolingual training, batches consists of monolingual sequences, while bilingual training consists of interleaved L1 and L2 sequences in each batch. We train 4 SoE models for each L1-L2 pair by initiating bilingual training at training step 0, 16k, 32k and 48k. We use a fixed learning rate with 5K warmup similarly to Issam et al. (2026) to avoid confounding our variables, because a linear learning rate would lead to later SoE models being trained on L2 with a lower learning rate. Finally, besides the bilingual models, we train a monolingual baseline using an identical tokenizer to each bilingual pair exclusively on English data. In total, 55 GPT-2 models were trained in this study (1 monolingual and 4 bilingual SoE models for each of the 11 L1-L2 pairs). Detailed training hyperparameters are provided in Appendix B.

## 3.2 Evaluation Framework

We use MECO L2 (Kuperman et al., 2025), which provides reading data of English as a second language. The dataset includes eye-tracking data of participants from 13 L1 language backgrounds collected across 16 institutions. Rather than aggregating participants by L1, we keep each institution as a distinct subset. We include all L1 groups except the Basque subset, which we exclude because of insufficient data representation in FineWeb-v2. We also include the native English (En-UK) subset to serve as a native control group. Participants across all subsets read the same 12 English texts sourced from standardized reading assessments. See Appendix A for further details on the subsets.

We evaluate model predictions against four eyetracking measures:

1. First Fixation Duration: The duration of the first fixation on a word.

2. Gaze Duration: The sum of all fixation durations landing on a word before the eye leaves it for the first time.

3. Go-Past Time: The total time elapsed from the initial fixation on a word until the reader’s gaze moves to the right of that word. This includes all time spent regressing to earlier parts of the text.

4. Total Fixation Duration: The sum of all fixation durations on a word, including subsequent fixations if the eye returns to it.

These measures are thought to reflect progressively later stages of cognitive processing (Rayner, 1998; Liversedge and Findlay, 2000). Early measures, such as first fixation duration and gaze duration, are assumed to capture initial lexical access, including word form recognition and meaning retrieval. Late measures, such as go-past time and total fixation duration, are thought to reflect post-lexical processing, such as structural reanalysis and syntactic or semantic integration. Separating these metrics allows us to distinguish context-invariant word recognition from higher-level, context-dependent processing (Inhoff, 1984; Berzak and Levy, 2023).

We derive surprisal estimates from every LM trained for this study: 1 monolingual and 4 SoE bilingual LMs for each of the 11 L1-L2 language pairs (excluding native English and Basque subsets). We calculate the surprisal of each word as its negative log-probability. For words segmented into multiple subwords by the tokenizer, we sum the surprisals of the constituent tokens.

To evaluate the predictions, we fit linear mixedeffects models (LMMs) with crossed random intercepts for participants and words. The baseline fixed effects include log-frequency, word length, and word position in the sentence. We also include the log-frequency and length of the preceding word $( w _ { i - 1 } )$ to account for spillover effects. Word frequencies are extracted using the wordfreq library (Speer, 2022).

## 3.2.1 Model Comparisons

We test whether bilingual models capture crosslinguistic cognitive processing through two main comparisons.

Bilingual vs. Monolingual First, we compare each bilingual model against the monolingual English baseline. Our hypothesis is that if bilingual models capture cross-linguistic influence, their surprisal estimates will yield better predictions of nonnative reading times than a strictly monolingual model’s. To account for inherent differences in baseline predictability across the different language subsets, we use a pseudo- $R ^ { 2 }$ metric:

$$
R _ { \mathrm { p s e u d o } } ^ { 2 } = 1 - { \frac { L L _ { \mathrm { m o d e l } } } { L L _ { \mathrm { b a s e l i n e } } } }\tag{1}
$$

where $L L _ { \mathrm { b a s e l i n e } }$ is the log-likelihood of an LMM containing only the surface baseline features. We then calculate $\Delta R ^ { 2 } = R _ { \mathrm { b i l i n g u a l } } ^ { 2 } - R _ { \mathrm { m o n o l i n g u a l } } ^ { 2 }$ to measure the proportion of explainable variance gained by the bilingual model. We evaluate the LMs on the respective MECO L1 subsets and the native En-UK. The native subset serves as a control group to verify that any observed advantage of a bilingual model is specific to L2 reading rather than a general improved fit to reading times.

Matched vs. Mismatched Bilingual Models Second, we compare matched bilingual LMs to mismatched bilingual LMs. A bilingual LM is considered matched if its L1 matches the native language of the evaluated MECO subset; all other bilingual

LMs are considered mismatched. If bilingual LMs develop L1-specific transfer effects similar to those of human readers, matched LMs should predict reading times more accurately than mismatched LMs. To quantify this difference, we calculate the difference in log-likelihood $( \Delta L L )$ between two non-nested LMMs: one using surprisal from a matched bilingual LM and the other using surprisal from a mismatched LM.

To compute formal p-values, we also conduct nested regression model comparisons. We fit an encompassing model $( m _ { \mathrm { b o t h } } )$ containing both the matched and mismatched surprisal predictors. We then compare this encompassing model against restricted models containing either the matched or the mismatched predictor alone. All three models use identical baseline covariates and random effects structures, differing solely in their surprisal terms. The unique explanatory variance added by matched surprisal is calculated as $L L ( m _ { \mathrm { { b o t h } } } )$ $L L ( m _ { \mathrm { m i s m a t c h } } )$ . Because $m _ { \mathrm { m i s m a t c h } }$ is nested within m<sub>both</sub>, we evaluate statistical significance of this added variance using a likelihood ratio test.

## 3.3 Proficiency Modulation Analysis

To evaluate how reader proficiency modulates the predictive advantage of bilingual surprisal over monolingual baselines, we employ a two-stage regression inspired by Berzak and Levy (2023). In the first stage, we isolate reader-specific surprisal sensitivity across eye-tracking measures by fitting individual participant regressions that control for baseline features. For each reader, we compute the bilingual explanatory advantage $( \Delta R ^ { 2 } =$ $\mathrm { \bar { \it R } _ { b i l i n g u a l } ^ { 2 } } - \mathrm { \bar { \it R } _ { m o n o l i n g u a l } ^ { 2 } } )$ . In the second stage, we model $\Delta R ^ { 2 }$ as a function of reader L2 English proficiency, measured via LexTALE (Lemhöfer and Broersma, 2011). To ensure orthogonal linear and quadratic predictors, proficiency scores are mean-centered. We estimate parameters using Weighted Least Squares (WLS), where each reader is weighted by their total observation count.

To test whether proficiency moderation is linear or non-linear, we compare a linear model $( \Delta R ^ { 2 } \sim$ prof) with a quadratic model $( \Delta R ^ { 2 } \sim \mathrm { p r o f } + \mathrm { p r o f } ^ { 2 } )$ We select between them using $\Delta A I C$ and test the quadratic coefficient using HC3 heteroskedasticityconsistent standard errors (MacKinnon and White, 1985). To rule out confounding by differences across L1 subsets, we then refit the quadratic model with L1-subset fixed effects and standard errors clustered by L1 subset, and check whether the quadratic term remains significant.

![](images/fa3f0b9a987526d4e59ee9d07fcada0e7aff649058e56b75e5e95e2c139e9b0e.jpg)

![](images/4d3e96e9711e82b7d546cca67ca9721d0d049c5ef72f7742ec6130df0fc1cc5f.jpg)

![](images/0a55ebc1222298ada2cdba1b75c829100e1a836bb0be3ebf99f69e573af61b5a.jpg)

![](images/e932fc0bbf661c8b8ab964359ea1e1436f539d0ac7d399695533f8ba128735a5.jpg)  
L1 Readers (EN-UK) Mean (± 95% CI) L2 Readers (Main) Mean (± 95% CI) Individual Traces  
Figure 1: Difference in explained variance $\Delta R _ { \mathrm { p s e u d o } } ^ { 2 }$ (scaled by $1 0 ^ { 4 } )$ in four reading measures as a function of SoE. The blue line represents the mean across all L2 reader subsets, while the red line denotes the native English (En-UK) control group. Faint lines indicate individual model evaluations. Positive values indicate a predictive advantage for the bilingual surprisal. With the exception of first fixation duration, bilingual surprisal consistently provides greater predictive power for L2 than L1 readers.

## 4 Results and Analysis

## 4.1 Are L2 Models Better at Predicting L2 Data than Monolingual Models?

Figure 1 shows the difference in explained variance $( \Delta R _ { p s e u d o } ^ { 2 } )$ between the matched bilingual models and the monolingual baselines across four SoE stages. The results show a difference between early and late reading measures.

For first fixation duration, the bilingual surprisal provides no predictive advantage for L2 readers. The mean $\Delta R _ { p s e u d o } ^ { 2 }$ for L2 readers starts near zero and drops slightly below zero at the SoE=48k. In contrast, for the later reading measures, the bilingual model yields a clear advantage. In both gaze duration and go-past time, the L2 mean is strictly positive and increases consistently with SoE. In total fixation time, the L2 mean remains positive and stable across all SoEs.

Crucially, across gaze duration, go-past time, and total fixation duration, the native English control group stays close to zero and consistently below the L2 mean. This gap confirms that the bilingual models capture variance specific to non-native reading behavior, rather than offering a generic predictive improvement for all readers.

## 4.2 Are Matched Models Better Than Mismatched Models?

Figure 2 illustrates the non-nested per-word $\Delta L L$ between LMMs using matched and mismatched bilingual surprisals. To control for evaluation biases or tokenization artifacts, we include a monolingual baseline derived by treating each L1 monolingual LM as matched and comparing it against the other monolingual LMs. If the matched advantage is genuinely driven by bilingual cross-linguistic transfer, it should exceed this strictly monolingual baseline. Except for first fixation duration, the matched bilingual models consistently outperform their mismatched counterparts, and this advantage strengthens in later processing measures. Notably, go-past time is the only metric exhibiting a clear upward trend in matched model advantage as SoE increases. Although there is variance among individual subset traces (representing the mean per-word ∆LL between matched and mismatched models for each L1 subset), the matched bilingual models generally exceed the monolingual control.

The nested model results in Table 1 show whether this matched advantage provides unique explanatory variance. By evaluating both predictors simultaneously, we can assess which model contributes a greater share of unique information based on their mean ∆LL and significance ratios. These nested comparisons perfectly align with our non-nested findings. During early lexical access (first fixation duration), matched models offer no L1-specific predictive advantage, contributing less unique variance than mismatched models. In contrast, across all later reading measures (gaze duration, go-past time, and total fixation duration), the matched models contribute substantially more unique log-likelihood and achieve statistical significance at much higher rates than their mismatched counterparts.

Ultimately, even though the surprisal estimates from different bilingual models are highly correlated (see Appendix C for surprisal correlations), the matched models consistently explain significantly more unique variance during late-stage reading. These results indicate that bilingual LMs successfully capture aspects of L1-specific structural processing.

![](images/c1fe815917d2969801f1694efcbfcd6776fbeceea8ecf64ba64d68edcaf830e5.jpg)

![](images/8c30969513638a370b9589cc5525acf7d08f09e566cc6e7f5b7d85b8945ed033.jpg)

![](images/74d61a9046fc3b18187c1a3f087aada68a995c2c97649a0855f8da55c7d22c94.jpg)

![](images/f22417d482756c85d3f7e02091cd68079c6c391603bd229d13a0550dae9bafd1.jpg)  
L1 Mono Baseline (Mean ± 95% CI) L2 Bilingual Mean (± 95% CI) Individual Traces

Figure 2: Per-word ∆LL (scaled by $1 0 ^ { 4 } )$ between matched and mismatched models in four reading measures as a function of SoE. Solid lines and shaded bands represent subset means with 95% confidence intervals for the bilingual models (blue) and monolingual baseline (red), with faint blue lines indicating individual subset evaluations. Higher values indicate a stronger predictive advantage for the matched model relative the mismatched one. With the exception of first fixation duration, bilingual models match or outperform the monolingual baseline.
<table><tr><td>Metric</td><td colspan="2">Mean ∆LL</td><td colspan="2"> $\mathbf { S i g . }$  Ratio (p &lt; 0.05)</td></tr><tr><td></td><td>Match</td><td>Mism.</td><td>Match</td><td>Mism.</td></tr><tr><td>First Fixation</td><td>1.07</td><td>1.33</td><td>17.0%</td><td>23.2%</td></tr><tr><td>Gaze Duration</td><td>3.23</td><td>1.78</td><td>55.0%</td><td>30.4%</td></tr><tr><td>Go-Past</td><td>3.32</td><td>1.69</td><td>53.7%</td><td>30.8%</td></tr><tr><td>Total Fixation</td><td>6.53</td><td>4.00</td><td>77.3%</td><td>64.3%</td></tr></table>

Table 1: Nested model comparisons $( m _ { \mathrm { b o t h } } )$ evaluating the unique contribution of matched versus mismatched bilingual surprisal averaged across all SoE models and language subsets. With the exception of First Fixation, matched models’ mean $\Delta L L$ and significance ratio are higher than the mismatched models.

## 4.3 The Role of Script Overlap

A primary structural dimension separating the evaluated subsets is writing system overlap with English. We examine this effect by partitioning the results from Figure 2 by script. The Latin script group consists of 8 MECO subsets across 7 L1s, while the Non-Latin group consists of 6 subsets across 4 L1s (with the exception of go-past time, where the ch\_s subset is missing). See Appendix A for details on the L1 subsets and their scripts.

This analysis is motivated by prior NLP research showing that shared scripts facilitate cross-lingual transfer in LMs (Dhamecha et al., 2021; Moosa et al., 2023), as well as psycholinguistic studies examining the processing dynamics of reading across different writing systems (Miwa et al., 2013; Taylor and Mukai, 2023; Arab, 2026).

Figure 3 illustrates the resulting per-word $\Delta L L$ In first fixation duration, neither script group displays a bilingual advantage. Latin bilingual models hover near zero close to the monolingual baseline, while Non-Latin bilingual models decline sharply below zero as SoE increases, dropping well below their within-group baseline. In gaze duration, the groups diverge. Latin bilingual models remain consistently positive and comfortably above the monolingual baseline, whereas Non-Latin models hover near zero and below the baseline across all SoEs. For later measures (go-past time and total fixation duration), however, bilingual models in both groups perform comparably to or better than the monolingual baseline. Non-Latin subsets also display a pronounced, steady upward trend in go-past time as SoE increases.

## 4.4 Effect of Reader Proficiency

Previous work demonstrates that L2 proficiency non-monotonically modulates the predictive power of monolingual surprisal (Berzak and Levy, 2023). Inspired by this finding, we ask a novel question: how does the added predictive power of bilingual surprisal change with reader proficiency? We hypothesize an inverted-U relationship for this bilingual predictive advantage over the monolingual baseline. This advantage should be minimal at both extremes of the proficiency spectrum: beginners rely primarily on shallow, context-independent features such as word frequency (Diependaele et al., 2013; Berzak and Levy, 2023), while advanced readers no longer depend on L1 transfer as their reading behavior approaches that of a native speaker (Chen et al., 2026). Consequently, the bilingual advantage should peak for intermediate readers, who possess enough fluency to engage in contextual prediction but still rely heavily on L1 syntactic biases during processing.

![](images/ca146eae8e7b83e37491fe7f3b72a434afb1483db248b3fcd6510c76826c8ab5.jpg)  
L1 Mono Baseline (Mean ± 95% CI)  
L2 Bilingual Mean (± 95% CI)  
Individual Traces  
Figure 3: Per-word ∆LL (scaled by $1 0 ^ { 4 } )$ between matched and mismatched models in four reading measures as a function of SoE, grouped by Latin (top) and Non-Latin (bottom) L1 scripts. Solid lines and shaded bands denote subset means with 95% confidence intervals for the bilingual models (blue) and monolingual baseline (red). Faint blue lines indicate individual subset traces. Early measures (First Fixation and Gaze Duration) reveal a clear divergence in predictive behavior between the two script groups.

As illustrated in Figure 4, while early lexical access measures maintain a linear relationship (all $p _ { \mathrm { q u a d } } > 0 . 0 5 )$ , late processing measures (go-past time and total fixation duration) display a pronounced inverted-U trajectory that strongly favors the quadratic polynomial $( \Delta \mathrm { A I C } > 2 . 0 $ for SoE $\geq 1 6 \mathrm { k } ;$ Appendix D.2). Crucially, this curvature remains robust: go-past time shows significant quadratic modulation across all SoE (0.001 < $p _ { \mathrm { q u a d } } < 0 . 0 5 )$ , and total fixation duration reaches significance for SoE $\geq 1 6 \mathrm { k } \left( p _ { \mathrm { q u a d } } < 0 . 0 5 \right)$ . The bilingual advantage peaks for intermediate readers (65–80% LexTALE) and descends toward zero for the most advanced readers (> 90% LexTALE).

Our hypothesis of an inverted-U trajectory is thus confirmed, though it applies exclusively to late measures where contextual integration and syntactic reanalysis are prominent. Importantly, this pattern is not an artifact of LexTALE: evaluating a composite proficiency index extracted via Principal Component Analysis (PCA) across all five proficiency tests in MECO (see Appendix D.3) replicates this trajectory, confirming robust significance of non-monotonic moderation in go-past time (all SoEs) and total fixation duration (SoE=16k).

To contextualize this result, Appendix D.1 visualizes the absolute explanatory power $( R ^ { 2 } )$ of each model independently. As expected, both models exhibit an inverted-U trend across all metrics, corroborating Berzak and Levy (2023) by showing that general contextual prediction peaks for intermediate learners. Crucially, however, Figure 4 demonstrates that the differential advantage of the bilingual model $( \Delta R ^ { 2 } )$ follows this non-linear trajectory strictly during late measures. This indicates that while intermediate readers generally rely on broad contextual expectations, their late-stage processing is more strongly influenced by nativelanguage biases.

## 5 Discussion and Conclusion

In this work, we investigated whether bilingual LMs align with human cross-linguistic influence during L2 reading. Our findings demonstrate that bilingual LMs successfully capture aspects of L1- specific effects on L2 processing. Crucially, our two distinct evaluation frameworks—comparing bilingual LMs against a monolingual baseline (§4.1), and contrasting L1-matched against L1- mismatched bilingual LMs (§4.2)—yield consistent results.

Across both comparisons, we observed a progressive shift in predictive performance from early to late reading measures. The predictive advantage of bilingual surprisals is absent in first fixation duration, begins to emerge in gaze duration, and becomes most pronounced in late-stage measures such as go-past time and total fixation duration. Regarding the SoE, we generally did not observe a uniform effect on predictability across all measures. This lack of a generalized trend may occur because the impact of prolonged L1 training varies considerably across different L1-L2 language pairs. The notable exception is go-past time, which exhibits a consistent upward trend in predictability as SoE increases.

![](images/dc49908fd79d528185a2b2c5400574197e23c03dea4e80c05b02a30628611f0c.jpg)  
Figure 4: Bilingual surprisal advantage $\Delta R ^ { 2 }$ (scaled by $1 0 ^ { 4 } )$ across readers’ English proficiency (LexTALE %) in four reading measures. Curves show quadratic fits per SoE model; circular markers denote binned means (±95% CI) at $\mathrm { S o E { = } 4 8 k }$ . Faint dots indicate individual observations, and the horizontal dotted line denotes zero gain. Inset labels indicate the significance of the quadratic proficiency term in a WLS with L1-subset fixed effects and standard errors clustered by L1 subset $( ^ { * } p < 0 . 0 5 , ^ { * * } p < 0 . 0 1 , ^ { * * * } p < 0 . 0 0 1$ , <sup>ns</sup>not significant). While early measures show negligible non-linearity, late measures (Go-Past, Total Fixation) show an inverted-U trajectory peaking among intermediate-proficiency readers (65–80%).

Our analysis of writing system overlap (§4.3) reveals a clear divergence in early processing. In gaze duration, only Latin L1 models showed a predictive advantage, whereas both script groups competed with or outperformed the monolingual baseline in late measures. These results can be interpreted through the Bilingual Interactive Activation (BIA+) model (Dijkstra and van Heuven, 2002). When languages share a script, visual input can directly activate orthographic representations in both languages simultaneously, allowing cross-linguistic effects to emerge in early reading measures (Miwa et al., 2013). In contrast, when scripts differ, direct orthographic co-activation is blocked, delaying cross-linguistic influence until downstream phonological or semantic integration occurs (Taylor and Mukai, 2023; Miwa et al., 2013). Consequently, cross-linguistic effects in Non-Latin groups emerge only in late-stage measures (go-past time and total fixation duration). While our findings cannot definitively confirm these specific cognitive mechanisms, this script-dependent processing divergence presents a compelling avenue for future investigation.

Additionally, our results reveal that the bilingual predictive advantage is non-linearly modulated by reader proficiency (§4.4). We observed an inverted-U trajectory that peaks for intermediate L2 readers and diminishes for highly advanced learners. Importantly, this proficiency modulation is robustly constrained to late processing measures. This indicates that while intermediate learners rely heavily on broad contextual expectations, their reliance on L1-driven biases surfaces predominantly during late-stage processing.

Across all findings, go-past time exhibits the most consistent alignment with L1-specific processing. Unlike the other reading measures that group eye fixations solely by location, go-past time groups fixations by sequence in time, directly measuring the time required to detect and resolve processing breakdowns (Liversedge et al., 1998; Liversedge and Findlay, 2000; Rayner, 1998). By incorporating L1 structural patterns, the bilingual models successfully predict the regressive eye movements associated with L1-L2 conflicts, cautioning future research against relying solely on early reading measures.

In conclusion, our work bridges multilingual NLP and psycholinguistics by demonstrating that bilingual LMs internalize L1-specific predictive biases that mirror human non-native reading. Rather than a uniform advantage, we show that this crosslinguistic alignment is concentrated in late-stage structural processing and peaks among intermediate learners. Consequently, accurately modeling L2 reading requires evaluation frameworks that capture the full temporal complexity of non-native processing.

## Limitations

While we have attempted to rigorously evaluate bilingual LMs on L2 sentence processing, several limitations remain.

First, our evaluation is restricted to English as the target L2 and relies on a single eye-tracking corpus (MECO). Although MECO provides a rich diversity of L1 backgrounds, focusing exclusively on L2 English leaves it an open question whether our findings generalize to target languages with fundamentally different typological features, such as highly agglutinative languages or those with freer word order. Additionally, data collection for certain L1 subsets in MECO took place in regions where specific regional dialects or varieties of the language are spoken (e.g., Brazilian Portuguese). Because our bilingual models were trained on standard, general-domain corpora for each L1, this mismatch between standard pretraining text and regional variants may introduce noise and could explain some of the variation observed in individual model performance across subsets.

Second, while our script analysis groups languages based on their script overlap with English, this is only one axis of variation. Other typological factors, such as word order alignment and morphological similarity, likely affect L2 sentence processing. Future work should investigate how these broader structural overlaps independently modulate the predictive biases observed in L2 reading.

Third, our operationalization of reader proficiency relies on LexTALE, a metric shown to correlate with general L2 proficiency (Lemhöfer and Broersma, 2011) and widely adopted for this purpose (e.g., Diependaele et al. 2013; Cop et al. 2015; Caissie-Gilbert et al. 2026). However, its validity as a measure of global proficiency has been questioned (Puig Mayenco et al., 2023), as it is fundamentally a lexical measure. To address this, we demonstrated the robustness of our results using a PCA-derived composite score based on all proficiency metrics reported in the MECO corpus. Nevertheless, these combined metrics still primarily index lower-level skills such as vocabulary size and decoding efficiency. As a result, the extent to which other proficiency factors such as higherlevel comprehension or syntactic awareness modulate L1-driven predictive biases remains an open question.

Finally, we trained each model configuration using a single initialization seed. Although this precludes measuring variance across different weight initializations, we mitigate this limitation through the breadth of our experimental setup. Specifically, we trained four SoE bilingual models for each of the 11 L1s and compared them against corresponding monolingual baselines. By analyzing the consistency of these results across diverse languages and tracking how they evolve across SoE stages, we ensure that the observed bilingual advantage is a robust trend rather than an artifact of a specific model checkpoint.

## References

José Alemán Bañón and Clara Martin. 2021. The role of crosslinguistic differences in second language anticipatory processing: An event-related potentials study. Neuropsychologia, 155:107797.

Tatsuya Aoyama and Nathan Schneider. 2024. Modeling nonnative sentence processing with L2 language models. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing, pages 4927–4940, Miami, Florida, USA. Association for Computational Linguistics.

Sepideh Arab. 2026. A systematic review of eyetracking studies on L1 Persian and Arabic reading: Implications for L2 English reading comprehension. Iranian Journal of Language Teaching Research, 14(2):129–154.

Kristijan Armeni, Roel M. Willems, and Stefan L. Frank. 2017. Probabilistic language models in cognitive neuroscience: Promises and pitfalls. Neuroscience & Biobehavioral Reviews, 83:579–588.

Catherine Arnett, Tyler A. Chang, James A. Michaelov, and Ben Bergen. 2025. On the acquisition of shared grammatical representations in bilingual language models. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 20707–20726, Vienna, Austria. Association for Computational Linguistics.

Yevgeni Berzak and Roger Levy. 2023. Eye movement traces of linguistic knowledge in native and non-native reading. Open Mind, 7:179–196.

Yevgeni Berzak, Chie Nakamura, Amelia Smith, Emily Weng, Boris Katz, Suzanne Flynn, and Roger Levy. 2022. Celer: A 365-participant corpus of eye movements in L1 and L2 English reading. Open Mind, 6:41–50.

Laurie Burchell, Alexandra Birch, Robert Thompson, and Kenneth Heafield. 2024. Code-switched language identification is harder than you think. In Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pages 646–658, St. Julian’s, Malta. Association for Computational Linguistics.

Kenneth P. Burnham and David R. Anderson. 2002. Model Selection and Multimodel Inference: A Practical Information-Theoretic Approach, 2nd edition. Springer-Verlag, New York.

Emanuel Bylund. 2019. Age effects in language attrition. In The Oxford Handbook of Language Attrition. Oxford University Press.

Antonie Caissie-Gilbert, Dominic Guitard, and Jean Saint-Aubin. 2026. Bilingualism and individual differences in short-term ordered serial recall. Memory, 34(6):666–688. PMID: 41972492.

Lin Chen, Xiaoping Fang, and Charles Perfetti. 2026. Skilled reading in a second language: Core reading processes, L2 proficiency and first-language background. Bilingualism: Language and Cognition, page 1–13.

Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. 2020a. Unsupervised cross-lingual representation learning at scale. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 8440– 8451, Online. Association for Computational Linguistics.

Alexis Conneau, Shijie Wu, Haoran Li, Luke Zettlemoyer, and Veselin Stoyanov. 2020b. Emerging cross-lingual structure in pretrained language models. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 6022–6034, Online. Association for Computational Linguistics.

Ionut Constantinescu, Tiago Pimentel, Ryan Cotterell, and Alex Warstadt. 2025. Investigating critical period effects in language acquisition through neural language models. Transactions ofthe Associationfor Computational Linguistics, 13:96–120.

Uschi Cop, Denis Drieghe, and Wouter Duyck. 2015. Eye movement patterns in natural reading: A comparison of monolingual and bilingual reading of a novel. PloS one, 10:e0134008.

Andrea de Varda and Marco Marelli. 2022. The effects of surprisal across languages: Results from native and non-native reading. In Findings ofthe Association for Computational Linguistics: AACL-IJCNLP 2022, pages 138–144, Online only. Association for Computational Linguistics.

Andrea Gregor de Varda and Marco Marelli. 2023. Datadriven cross-lingual syntax: An agreement study with massively multilingual models. Computational Linguistics, 49(2):261–299.

Vera Demberg and Frank Keller. 2008. Data from eyetracking corpora as evidence for theories of syntactic processing complexity. Cognition, 109(2):193–210.

Tejas Dhamecha, Rudra Murthy, Samarth Bharadwaj, Karthik Sankaranarayanan, and Pushpak Bhattacharyya. 2021. Role of Language Relatedness in Multilingual Fine-tuning of Language Models: A Case Study in Indo-Aryan Languages. In Proceedings ofthe 2021 Conference on Empirical Methods in Natural Language Processing, pages 8584–8595, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Prajit Dhar and Arianna Bisazza. 2021. Understanding cross-lingual syntactic transfer in multilingual recurrent neural networks. In Proceedings of the 23rd Nordic Conference on Computational Linguistics (NoDaLiDa), pages 74–85, Reykjavik, Iceland (Online). Linköping University Electronic Press, Sweden.

Kevin Diependaele, Kristin Lemhöfer, and Marc Brysbaert. 2013. The word frequency effect in first- and second-language word recognition: A lexical entrenchment account. Quarterly Journal of Experimental Psychology, 66:843–863.

Ton Dijkstra and Walter van Heuven. 2002. The architecture of the bilingual word recognition system: From identification to decision. Bilingualism: Language and Cognition, 5:175–197.

Alice Foucart. 2021. Language prediction in second language: Does language similarity matter? In Edith Kaan and Theres Grüter, editors, Prediction in Second Language Processing and Learning, number 12 in Bilingual Processing and Acquisition, pages 91– 114. John Benjamins Publishing Company.

Stefan Frank. 2009. Surprisal-based comparison between a symbolic and a connectionist model of sentence processing. In Proceedings ofthe Annual Meeting ofthe Cognitive Science Society, volume 31.

Stefan Frank. 2021. Cross-language structural priming in recurrent neural network language models. In Proceedings of the Annual Meeting of the Cognitive Science Society, volume 43.

Richard Futrell and Kyle Mahowald. 2026. How linguistics learned to stop worrying and love the language models. Behavioral and Brain Sciences, 49:e198.

Adam Goodkind and Klinton Bicknell. 2018. Predictive power of word surprisal for reading times is a linear function of language model quality. In Proceedings of the 8th Workshop on Cognitive Modeling and Computational Linguistics (CMCL 2018), pages 10–18, Salt Lake City, Utah. Association for Computational Linguistics.

John Hale. 2001. A probabilistic Earley parser as a psycholinguistic model. In Second Meeting ofthe North American Chapter of the Association for Computational Linguistics.

Albrecht Werner Inhoff. 1984. Two stages of word processing during eye fixations in the reading of prose. Journal of Verbal Learning and Verbal Behavior, 23(5):612–624.

Abderrahmane Issam, Yusuf Can Semerci, Jan Scholtes, and Gerasimos Spanakis. 2026. A study of crosslinguistic influence in language models. Preprint, arXiv:2601.21587.

Antonia Karamolegkou, Mostafa Abdou, and Anders Søgaard. 2023. Mapping brains with language models: A survey. In Findings of the Association for Computational Linguistics: ACL 2023, pages 9748– 9762, Toronto, Canada. Association for Computational Linguistics.

Hyunwoo Kim. 2025. Crosslinguistic influence in bilingual morphosyntactic processing: Effects of language-common, language-contrasting, and language-specific information. Bilingualism: Language and Cognition, 28(1):172–184.

Taku Kudo and John Richardson. 2018. SentencePiece: A simple and language independent subword tokenizer and detokenizer for neural text processing. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 66–71, Brussels, Belgium. Association for Computational Linguistics.

Victor Kuperman, Sascha Schroeder, Cengiz Acartürk, Niket Agrawal, Dominick M. Alexandre, Lena S. Bolliger, Jan Brasser, César Campos-Rojas, Denis Drieghe, Dušica Filipovic Ður´ devi¯ c, and et al. 2025.´ New data on text reading in English as a second language: The wave 2 expansion of the Multilingual Eye-Movement Corpus (MECO). Studies in Second Language Acquisition, 47(2):677–695.

Kristin Lemhöfer and Mirjam Broersma. 2011. Introducing lextale: A quick and valid lexical test for advanced learners of english. Behavior Research Methods, 44:325–343.

Eric H. Lenneberg. 1967. The biological foundations of language. Hospital Practice, 2(12):59–67.

Roger Levy. 2008. Expectation-based syntactic comprehension. Cognition, 106(3):1126–1177.

Jindˇrich Libovický, Rudolf Rosa, and Alexander Fraser. 2020. On the language neutrality of pre-trained multilingual representations. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2020, pages 1663–1674, Online. Association for Computational Linguistics.

Simon P. Liversedge and John M. Findlay. 2000. Saccadic eye movements and cognition. Trends in Cognitive Sciences, 4(1):6–14.

Simon P. Liversedge, Kevin B. Paterson, and Martin J. Pickering. 1998. Chapter 3 - eye movements and measures of reading time. In Geoffrey Underwood, editor, Eye Guidance in Reading and Scene Perception, pages 55–75. Elsevier Science Ltd, Amsterdam.

James G MacKinnon and Halbert White. 1985. Some heteroskedasticity-consistent covariance matrix estimators with improved finite sample properties. Journal ofEconometrics, 29(3):305–325.

James A. Michaelov, Catherine Arnett, Tyler A. Chang, and Benjamin K. Bergen. 2023. Structural priming demonstrates abstract grammatical representations in multilingual language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 3703–3720, Singapore. Association for Computational Linguistics.

Koji Miwa, Ton Dijkstra, Patrick Bolger, and Harald Baayen. 2013. Reading English with Japanese in mind: Effects of frequency, phonology, and meaning in different-script bilinguals. Bilingualism: Language and Cognition, 17:445–463.

Ibraheem Muhammad Moosa, Mahmud Elahi Akhter, and Ashfia Binte Habib. 2023. Does transliteration help multilingual language modeling? In Findings of the Association for Computational Linguistics: EACL 2023, pages 670–685, Dubrovnik, Croatia. Association for Computational Linguistics.

Aaron Mueller, Garrett Nicolai, Panayiota Petrou-Zeniou, Natalia Talmina, and Tal Linzen. 2020. Cross-linguistic syntactic evaluation of word prediction models. In Proceedings ofthe 58th Annual Meeting of the Association for Computational Linguistics, pages 5523–5539, Online. Association for Computational Linguistics.

Qian Niu, Junyu Liu, Ziqian Bi, Pohsun Feng, Benji Peng, Keyu Chen, Ming Li, Kai Qi Yan, Yichao Zhang, Caitlyn Yin, Cheng Fei, Tianyang Wang, Yunze Wang, Silin Chen, Ming Liu, Ziyuan Qin, Riyang Bao, Xinyuan Song, and Zekun Jiang. 2026. Large language models and cognitive science: A comprehensive review of similarities, differences, and challenges. BIO Integration, 7.

Miyu Oba, Tatsuki Kuribayashi, Hiroki Ouchi, and Taro Watanabe. 2023. Second language acquisition of neural language models. In Findings of the Association for Computational Linguistics: ACL 2023, pages 13557–13572, Toronto, Canada. Association for Computational Linguistics.

Terence Odlin. 1989. Language Transfer. Cambridge Applied Linguistics. Cambridge University Press.

Byung-Doh Oh and William Schuler. 2023. Why does surprisal from larger transformer-based language models provide a poorer fit to human reading times? Transactions of the Association for Computational Linguistics, 11:336–350.

Isabel Papadimitriou and Dan Jurafsky. 2020. Learning Music Helps You Read: Using transfer to study linguistic structure in language models. In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 6829–6839, Online. Association for Computational Linguistics.

Guilherme Penedo, Hynek Kydlícek, Loubna Ben Al-ˇ lal, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro Von Werra, and Thomas Wolf. 2024. The fineweb datasets: decanting the web for the finest

text data at scale. In Proceedings ofthe 38th International Conference on Neural Information Processing Systems, NIPS ’24, Red Hook, NY, USA. Curran Associates Inc.

Guilherme Penedo, Hynek Kydlícek, Vinko Sabolˇ cec,ˇ Bettina Messmer, Negar Foroutan, Amir Hossein Kargaran, Colin Raffel, Martin Jaggi, Leandro Von Werra, and Thomas Wolf. 2025. Fineweb2: One pipeline to scale them all – adapting pre-training data processing to every language. Preprint, arXiv:2506.20920.

Telmo Pires, Eva Schlinger, and Dan Garrette. 2019. How multilingual is multilingual BERT? In Proceedings ofthe 57th Annual Meeting ofthe Associationfor Computational Linguistics, pages 4996–5001, Florence, Italy. Association for Computational Linguistics.

Eloi Puig Mayenco, Adel Chaouch-Orozco, Hong Liu, and Fernando Martín-Villena. 2023. The LexTALE as a measure of L2 global proficiency: A cautionary tale based on a partial replication of Lemhöfer and Broersma (2012). Linguistic Approaches to Bilingualism, 13:299–314.

Libo Qin, Minheng Ni, Yue Zhang, and Wanxiang Che. 2020. Cosda-ml: Multi-lingual code-switching data augmentation for zero-shot cross-lingual nlp. In Proceedings ofthe Twenty-Ninth International Joint Conference on Artificial Intelligence, IJCAI-20, pages 3853–3860. International Joint Conferences on Artificial Intelligence Organization. Main track.

Keith Rayner. 1998. Eye movements in reading and information processing: 20 years of research. Psychological Bulletin, 124:372–422.

Suchir Salhan, Catherine Arnett, James Michaelov, and Paula Buttery. 2026. Beetle: A bilingual model suite for modelling second-language processing. Preprint, arXiv:2609.22633.

Elizabeth Schotter and Brian Dillon. 2025. A beginner’s guide to eye tracking for psycholinguistic studies of reading. Behavior Research Methods, 57.

Cory Shain, Clara Meister, Tiago Pimentel, Ryan Cotterell, and Roger Levy. 2024. Large-scale evidence for logarithmic effects of word predictability on reading time. Proceedings of the National Academy of Sciences, 121(10):e2307876121.

Nathaniel J. Smith and Roger Levy. 2013. The effect of word predictability on reading time is logarithmic. Cognition, 128(3):302–319.

Robyn Speer. 2022. rspeer/wordfreq: v3.0.

Jamie Taylor and Yoichi Mukai. 2023. Bidirectional cross-linguistic influence with different-script languages: Evidence from eye tracking. Applied Psycholinguistics, 44(5):635–667.

Natasha Tokowicz and Brian MacWhinney. 2005. Implicit and explicit measures of sensitivity to violations in second language grammar: An event-related potential investigation. Studies in Second Language Acquisition, 27(2):173–204.

Natasha Tokowicz, Tessa Warren, and Leida Tolentino. 2024. The competition model and adult second language learning: Cross-language similarity and L2 morphosyntax learning. Language Teaching Research Quarterly, 44:71–83.

Zhijun Wang, Jiahuan Li, Hao Zhou, Rongxiang Weng, Jingang Wang, Xin Huang, Xue Han, Junlan Feng, Chao Deng, and Shujian Huang. 2025. Investigating and scaling up code-switching for multilingual language model pre-training. In Findings of the Association for Computational Linguistics: ACL 2025, pages 11032–11046, Vienna, Austria. Association for Computational Linguistics.

Ethan G. Wilcox, Jon Gauthier, Jennifer Hu, Peng Qian, and Roger Levy. 2020. On the predictive power of neural language models for human real-time comprehension behavior. In Proceedings ofthe Annual Meeting ofthe Cognitive Science Society, pages 1707– 1713.

Ethan G. Wilcox, Tiago Pimentel, Clara Meister, Ryan Cotterell, and Roger P. Levy. 2023. Testing the predictions of surprisal theory in 11 languages. Transactions ofthe Associationfor Computational Linguistics, 11:1451–1470.

Irene E. Winther, Yevgen Matusevych, and Martin J. Pickering. 2021. Cumulative frequency can explain cognate facilitation in language models. In Proceedings ofthe Annual Meeting ofthe Cognitive Science Society, volume 43, pages 2513–2519.

Shijie Wu and Mark Dredze. 2019. Beto, bentz, becas: The surprising cross-lingual effectiveness of BERT. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 833–844, Hong Kong, China. Association for Computational Linguistics.

Aditya Yadavalli, Alekhya Yadavalli, and Vera Tobin. 2023. SLABERT talk pretty one day: Modeling second language acquisition with BERT. In Proceedings ofthe 61st Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 11763–11777, Toronto, Canada. Association for Computational Linguistics.

## A MECO Subsets and Languages

Table 2 summarizes the L1 languages, writing scripts, total observations, and number of readers for each MECO L1 subset. While the exact sample sizes vary across individual reading measures due to missing eye-tracking data, every evaluated subset maintains a minimum of 7,639 samples per metric. The sole exception is the ch\_s subset, which completely lacks data for go-past time.

<table><tr><td>Subset</td><td>Language &amp; Site</td><td>Samples</td><td>Readers</td></tr><tr><td>hi_iiith</td><td>Hindi (Hyderabad)*</td><td>77,522</td><td>53</td></tr><tr><td>en_uk</td><td>English (UK)</td><td>66,041</td><td>50</td></tr><tr><td>bp</td><td>Brazilian Portuguese</td><td>62,608</td><td>54</td></tr><tr><td>ge_zu</td><td>German (Zurich)</td><td>58,949</td><td>47</td></tr><tr><td>hi_iitk</td><td>Hindi (Kanpur)*</td><td>58,642</td><td>45</td></tr><tr><td>ru_mo</td><td>Russian (Moscow)*</td><td>57,817</td><td>49</td></tr><tr><td>ic</td><td>Icelandic</td><td>56,821</td><td>45</td></tr><tr><td>ge_po</td><td>German (Potsdam)</td><td>49,604</td><td>43</td></tr><tr><td>sp_ch</td><td>Spanish (Chile)</td><td>49,434</td><td>45</td></tr><tr><td>ch_s</td><td>Chinese simplified*</td><td>47,685</td><td>46</td></tr><tr><td>se</td><td>Serbian*</td><td>46,220</td><td>43</td></tr><tr><td>ch_t</td><td>Chinese traditional*</td><td>44,900</td><td>43</td></tr><tr><td>da</td><td>Danish</td><td>25,815</td><td>25</td></tr><tr><td>no</td><td>Norwegian</td><td>19,025</td><td>22</td></tr><tr><td>tr</td><td>Turkish</td><td>13,797</td><td>14</td></tr></table>

Table 2: MECO L2 subsets, corresponding L1 languages and recording site (\* denotes Non-Latin), total number of samples (before filtering) and number of readers.

## B Hyperparameters

All models share the exact same GPT-2 architecture and training hyperparameters, which are detailed in Table 3.

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Layers</td><td>12</td></tr><tr><td>Embedding size</td><td>768</td></tr><tr><td>Hidden size Intermediate hidden size</td><td>768</td></tr><tr><td>Attention heads</td><td>3072 12</td></tr><tr><td>Attention head size</td><td>64</td></tr><tr><td>Activation function</td><td>GELU</td></tr><tr><td>Vocab size Max sequence length</td><td>50004</td></tr><tr><td>Position embedding</td><td>256 Absolute</td></tr><tr><td>Batch size</td><td>64</td></tr><tr><td>Train steps</td><td>64k</td></tr><tr><td>Learning rate decay</td><td>Constant</td></tr><tr><td>Warmup steps</td><td>5000</td></tr><tr><td>Learning rate</td><td> $1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Adam €</td><td> $1 \times 1 0 ^ { - 6 }$ </td></tr><tr><td></td><td></td></tr><tr><td>Adam  $\beta _ { 1 }$ </td><td>0.9</td></tr><tr><td>Adam  $\beta _ { 2 }$ </td><td>0.999</td></tr><tr><td>Dropout</td><td>0.1</td></tr><tr><td>Attention dropout</td><td>0.1</td></tr><tr><td></td><td></td></tr><tr><td>Seed</td><td>123</td></tr></table>

Table 3: Architectural and optimization hyperparameters for the GPT-2 bilingual models.

## C Bilingual Model Surprisal Correlation

Because all bilingual models in this study are trained and evaluated on English, their surprisal estimates for English text share substantial variance. We calculate the pairwise Pearson correlation of per-word surprisal across all L1 bilingual models using the 12 English texts from the MECO dataset. Figure 5 confirms this high overlap, showing that all pairwise correlations remain above 0.94. However, the mean correlation decreases slightly as SoE increases (from 0.96 to 0.95), suggesting that prolonged L1 exposure introduces small but noticeable L1-specific differences between models.

![](images/4c1d13d6ee7a8b7070bd008f2a010178c6f8674b30aa27736b82b713be690816.jpg)  
Figure 5: Pairwise Pearson correlation of English surprisal values between all L1 bilingual models across the SoE stages. The dark blue line represents the mean correlation, the shaded band shows 95% confidence interval, and the light blue lines show individual pairwise comparisons.

## D Proficiency Analysis

## D.1 Isolated effects of Bilingual and Monolingual Surprisal

In §4.4, we evaluated the unique advantage of bilingual surprisal over monolingual surprisal $( \Delta R ^ { 2 }$ $R _ { \mathrm { b i l i n g u a l } } ^ { 2 } - R _ { \mathrm { m o n o l i n g u a l } } ^ { 2 } )$ . To provide full context for this metric, Figure 6 displays the explanatory power of the monolingual and the bilingual surprisal models separately over the baseline (which contains only surface covariates such as word length and frequency).

The absolute surprisal from both monolingual and bilingual models follows a highly similar inverted-U trajectory across proficiency levels, peaking for intermediate readers. This replicates the findings of Berzak and Levy (2023), confirming that intermediate L2 readers rely most heavily on context-dependent expectations. Because the bilingual and monolingual LMs share a vast majority of their English linguistic representations (as evidenced by their high surprisal correlation), their individual lines overlap almost entirely. Crucially, in the late processing measures (go-past time and total fixation duration), the bilingual models consistently diverge from and peak slightly above the monolingual baseline. The subtraction methodology utilized in §4.4 isolates this specific gap, confirming that the isolated cross-linguistic transfer effect also scales non-linearly with reader proficiency.

![](images/d6cf245550faf304d47f13c4c2704bde2bc97ebece4cdbca95f6ecd11aa3cc8e.jpg)

![](images/df541d92683fad2078165878772120358758b057e0a5efee5684bb764dbae51a.jpg)

![](images/411ca9adde16cb10b99ac9884d151957539476c77e20d8f0bfb377725f561f58.jpg)

![](images/12d167b3feb790054595eeb34d3d6c62e93534398d08407e3e7dba0bfd4bec47.jpg)  
Figure 6: Explanatory power $( \Delta R ^ { 2 }$ over the baseline, scaled by $1 0 ^ { 4 } )$ of monolingual and of bilingual surprisal across readers’ English proficiency (LexTALE %) in four reading measures. Faint dots indicate individual observations, and the horizontal dotted line denotes zero gain. Inset labels indicate the significance of the quadratic proficiency term in a WLS with L1-subset fixed effects and standard errors clustered by L1 subset $( ^ { * } p < 0 . 0 5 , ^ { * * } p < 0 . 0 1 , ^ { * * * } p < 0 . 0 0 1$ <sup>ns</sup>not significant). Both bilingual and monolingual models exhibit a pronounced inverted-U relationship with proficiency.

## D.2 Quadratic Relationship Evaluation

Table 4 provides the statistical evaluation comparing linear and quadratic specifications for the effect of proficiency on the bilingual advantage $( \Delta R ^ { 2 } )$ . For early measures (first fixation duration and gaze duration), the quadratic term is not statistically significant, and the negative $\Delta \mathrm { A I C }$ values indicate that the simpler linear baseline is generally preferred. In contrast, for late measures (go-past time and total fixation duration), the quadratic model provides a better fit $( p _ { \mathrm { q u a d } } < 0 . 0 5$ and $\Delta \mathrm { A I C } > 2 . 0 )$ for models with $\mathrm { S o E } \geq 1 6 \mathrm { k }$ . The negative $\beta _ { \mathrm { q u a d } }$ coefficients align with the observed inverted-U trajectory. Support for this non-linear relationship tends to increase with longer L1 training; in go-past time, for instance, the $\Delta \mathrm { A I C }$ rises from +0.6 at SoE 0 to +9.5 at SoE 48k. This suggests that a stronger L1 influence on the L2 (due to extended L1 training and delayed L2 exposure) increases the predictive value of bilingual surprisal mostly for intermediate learners.

<table><tr><td>Metric</td><td>SoE</td><td>N</td><td> $\beta _ { \mathrm { q u a d } }$ </td><td>pquad</td><td>∆AIC</td></tr><tr><td rowspan="4">First Fixation</td><td>0</td><td>480</td><td>+0.002</td><td>0.464</td><td>-1.5</td></tr><tr><td>16k</td><td>480</td><td>+0.002</td><td>0.494</td><td>-1.7</td></tr><tr><td>32k</td><td>480</td><td>-0.001</td><td>0.793</td><td>-2.0</td></tr><tr><td>48k</td><td>480</td><td>+0.003</td><td>0.436</td><td>-1.5</td></tr><tr><td rowspan="4">Gaze Duration</td><td>0</td><td>480</td><td>-0.007</td><td>0.129</td><td>-0.2</td></tr><tr><td>16k</td><td>480</td><td>-0.007</td><td>0.093</td><td>+0.0</td></tr><tr><td>32k</td><td>480</td><td>-0.009</td><td>0.053</td><td>+0.8</td></tr><tr><td>48k</td><td>480</td><td>-0.002</td><td>0.703</td><td>-1.9</td></tr><tr><td rowspan="4">Go-Past</td><td>0</td><td>434</td><td>-0.010</td><td>0.055</td><td>+0.6</td></tr><tr><td>16k</td><td>434</td><td>-0.014</td><td>0.011</td><td>+2.5</td></tr><tr><td>32k</td><td>434</td><td>-0.022</td><td>&lt; 0.001</td><td>+6.9</td></tr><tr><td>48k</td><td>434</td><td>-0.029</td><td>&lt; 0.001</td><td>+9.5</td></tr><tr><td rowspan="4">Total Fixation</td><td>0</td><td>480</td><td>-0.011</td><td>0.074</td><td>+0.9</td></tr><tr><td>16k</td><td>480</td><td>-0.017</td><td>0.002</td><td>+4.5</td></tr><tr><td>32k</td><td>480</td><td>-0.018</td><td>0.003</td><td>+4.6</td></tr><tr><td>48k</td><td>480</td><td>-0.022</td><td>0.004</td><td>+5.0</td></tr></table>

Table 4: Model comparison evaluating non-linear proficiency moderation across eye-tracking measures. $\beta _ { \mathrm { q u a d } }$ denotes the estimated coefficient of the quadratic proficiency term, and $p _ { \mathrm { q u a d } }$ denotes its robust HC3 significance. $\Delta \mathrm { A I C } = \mathrm { A I C } _ { \mathrm { l i n e a r } } - \mathrm { A I C } _ { \mathrm { q u a d r a t i c } }$ , where values $> + 2 . 0$ indicate that the quadratic model improves on the linear baseline by more than the penalty for its extra parameter (Burnham and Anderson, 2002). Boldface marks exposure stages where both criteria favor the quadratic model $( p _ { \mathrm { q u a d } } < 0 . 0 5 , \Delta \mathrm { A I C > 2 . 0 ) }$

## D.3 Robustness to Composite Proficiency (PCA)

To verify that the non-linear proficiency moderation observed with LexTALE is not an artifact of this single metric, we evaluated whether these findings generalize across other proficiency measures reported in MECO (Kuperman et al., 2025). Specifically, we extracted a composite index via Principal

![](images/a9f982637fc1c824aa73e384b447522427dd098647aecf0f7b91b9617e49c666.jpg)

![](images/95d81989acdd5ac289996b7fe1462b1dc4c5ce2132e0d474edbf8762889722a9.jpg)

![](images/a97a9f383dc52019be4ac4bcf9a8d6bdeb17aa8518bd81c33976eee1e5599f37.jpg)

![](images/565b2b86f2c9838a371a43523f3fa7c1d65f53b9773b8bbee388f48fff1f906f.jpg)  
Figure 7: Bilingual surprisal advantage $( \Delta R ^ { 2 } = R _ { \mathrm { b i l i n g u a l } } ^ { 2 } - R _ { \mathrm { m o n o l i n g u a l } } ^ { 2 } ,$ scaled by $1 0 ^ { 4 } )$ modeled against composite L2 proficiency (PC1 extracted across LexTALE, Vocabulary, Spelling, TOWRE word, and TOWRE nonword; with 52.5% variance explained) in four reading measures. Curves depict second-order polynomial regressions across SoEs; markers denote empirical binned means (±95% CI) at SoE=48k. Inset labels indicate the significance of the quadratic proficiency term in a WLS with L1-subset fixed effects and standard errors clustered by L1 subset $( ^ { * } p < 0 . 0 5 , ^ { * * } p < 0 . 0 1 , ^ { * * * } p < 0 . 0 0 1$ , <sup>ns</sup>not significant). Faint points represent individual participants, and the dotted horizontal line marks zero gain.

Component Analysis (PCA) across five standardized assessments (LexTALE, vocabulary knowledge, spelling, TOWRE-2 test of words and nonwords), with the first principal component (PC1) accounting for 52.5% of the total variance.

As shown in Figure 7, modeling $\Delta R ^ { 2 }$ against PC1 closely mirrors the findings obtained with Lex-TALE alone. In early reading measures (first fixation duration and gaze duration), the quadratic term remains non-significant across all SoE (p>0.05), confirming our earlier findings using LexTALE. In contrast, late processing in go-past time robustly replicates the inverted-U trajectory across all SoE (p<0.05). For total fixation duration, the quadratic trajectory exhibits the same qualitative inverted-U curvature, although under strict L1-subset controls its statistical significance is attenuated relative to LexTALE, reaching significance only at SoE=16k. Overall, these results demonstrate that the bilingual processing advantage peaking among intermediate-proficiency readers in late-stage eyemovement measures generalizes across alternative proficiency assessments, particularly for go-past time.