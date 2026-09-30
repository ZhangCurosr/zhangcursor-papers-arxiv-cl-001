# Reader Proficiency Shapes Layer-wise Surprisal Profiles

Akio Hayakawa

Horacio Saggion

Universitat Pompeu Fabra Barcelona, Spain {akio.hayakawa, horacio.saggion}@upf.edu

## Abstract

Reading behaviour varies not only with linguistic input, but also with reader proficiency. In this study, we investigate whether the layerwise relationship between surprisal from large language models (LLMs) and human gaze behaviour differs across readers with different levels of proficiency and across gaze measures. Using eye-tracking data from the MECO L2 corpus, we compare readers with high and low vocabulary proficiency on first-pass gaze duration (FPGD) and total gaze duration (TGD). We quantify the distribution of the predictive power of surprisal across model layers using Predictive Depth. Across 12 tested LLMs, we find that readers with lower vocabulary proficiency tend to show deeper Predictive Depth for FPGD, while this difference is smaller for TGD. Also, TGD itself shows deeper Predictive Depth than FPGD in both proficiency groups. These patterns suggest that where predictive power is concentrated across LLM layers may be related to the timing and breadth of the reading processes captured by different gaze measures, and that this relationship can vary with reader proficiency. Our leave-one-out analysis further shows that the advantage of informative internal layers extends to unseen texts, although the practical improvements in prediction are limited. Overall, our results show that layerwise LLM surprisal provides a useful perspective on variation in reading behaviour across both reader groups and gaze measures.<sup>1</sup>

## 1 Introduction

While text difficulty is often treated as a fixed property of the text, understanding the reading process requires considering the interaction between readers and linguistic structures. This perspective is particularly important in fields such as text simplification, which aims to enhance reader comprehension by rewriting complex texts into more accessible forms (Siddharthan, 2014; Alva-Manchego et al., 2020). However, traditional methods for evaluating text difficulty often rely on surface-level metrics, such as Flesch-Kincaid Grading Levels (Kincaid et al., 1975), or comprehension accuracy (Agrawal and Carpuat, 2024). While these measures can indicate if a text is linguistically simple or if a reader eventually understands it, they provide limited information about how the text is processed during reading. This is of paramount importance because readers with different levels of proficiency may process the same text differently.

![](images/2c5ad388cc407ec58e144764fc5eed01287052c42bcba16de2d0304de63f3b50.jpg)  
Figure 1: Illustration of the distribution of predictive power along LLM layers. Readers with limited vocabulary knowledge (red) tend to show deeper Predictive Depth for FPGD compared to readers with higher vocabulary knowledge (blue). The difference in Predictive Depth is smaller for TGD, which itself shows deeper Predictive Depth than FPGD in both reader groups.

In this context, surprisal from large language models (LLMs) provides a useful way to examine human reading behaviour. While surprisal has been widely utilized to predict human reading times (Smith and Levy, 2013; Kuribayashi et al., 2022; Shain et al., 2024), recent findings suggest that different internal layers of LLMs have different predictive power for human reading times (Kuribayashi et al., 2025). Importantly, different reading measures are best predicted by surprisal from different parts of LLMs. For example, measures which capture relatively fast responses, such as gaze duration, tend to be better predicted by surprisal from shallower layers, whereas measures which capture slower responses show stronger predictive power in deeper layers. Among eye-tracking measures, firstpass gaze duration (FPGD) and total gaze duration (TGD) capture different spans of reading behaviour. FPGD reflects the initial pass over a word, while TGD includes later and repeated fixations. This pattern may be related to differences in the timing and breadth of cognitive processing captured by each measure.

While previous studies have mainly focused on native English speakers, the potential of this approach becomes evident when considering the diversity of reader proficiency. Cognitive models of reading suggest that less-proficient readers may engage in compensatory processing (Stanovich, 1980) when faced with linguistic obstacles, such as difficult vocabulary or syntax. In such cases, they may rely more on contextual information to support comprehension. This raises the possibility that reader proficiency may also be an important factor in determining which internal layers best explain reading behaviour.

In this study, we examine how the layer-wise relationship between LLM surprisal and reading behaviour varies with reader proficiency and gaze measure, focusing on FPGD and TGD. To summarise the layer-wise distribution of predictive power, we introduce Predictive Depth, which measures how the predictive power of surprisal is distributed across LLM layers.

We find that the layer-wise relationship between LLM surprisal and reading behaviour varies both with reader proficiency and gaze measure. For FPGD, readers with lower vocabulary proficiency tend to show deeper Predictive Depth, while this difference is substantially reduced for TGD. Also, TGD generally shows deeper Predictive Depth than FPGD across both proficiency groups. A leaveone-out analysis further shows that the predictive advantage of selected internal layers extends to unseen texts, although the improvements are small.

Our main contributions are as follows:

• We show that the layer depth at which surprisal best predicts reading behaviour varies with vocabulary proficiency.

• We show that this relationship also differs between FPGD and TGD, revealing a clear effect of gaze measure.

• We show that surprisal provides weak to modest predictive gains on unseen texts, with the selected internal layer generally outperforming the final layer.

## 2 Related Work

## 2.1 Cognitive Models of Reading

Reading involves the coordination of lower-level lexical access with higher-level semantic integration (Kintsch, 1988). A fundamental distinction between proficient and less-proficient readers lies in the automaticity of these processes. Proficient readers can access lexical information with relatively little conscious effort (LaBerge and Samuels, 1974). This automaticity leaves more cognitive resources available for higher-level semantic integration (Perfetti and Hart, 2002). In contrast, when lexical access is less automatic, readers may rely more on contextual information to support word recognition and comprehension (Stanovich, 1980). While contextual information can support comprehension, greater reliance on context may also require readers to integrate information across a broader context (Harrington and Sawyer, 1992; Carretti et al., 2009). Therefore, differences in proficiency can affect not only whether a text is understood, but also how linguistic information is processed during reading.

These differences are also reflected in eyemovement behaviour. Measures such as FPGD and TGD provide observable traces of the underlying cognitive processes involved in reading (Rayner, 1998). Examining variation in these measures can help us investigate how processes such as lexical access and semantic integration differ across readers with different levels of proficiency.

## 2.2 Layer-wise Surprisal and Reading Measures

In computational linguistics, surprisal is defined as the negative log-probability of a word given its preceding context (Hale, 2001; Levy, 2008). Surprisal from language models has long been used to model human reading behaviour (Smith and Levy, 2013; Kuribayashi et al., 2022; Shain et al., 2024). In recent cognitive modeling frameworks, the unique predictive power of surprisal is typically quantified by the increase in log-likelihood (∆LL) when surprisal is added to a regression model containing baseline linguistic features (Oh et al., 2022). More recently, studies have shown that surprisal from the final layer of modern large language models (LLMs) does not always provide the best explanation for human reading times (Oh and Schuler, 2023), and that surprisal from internal layers can improve predictive performance (Kuribayashi et al., 2025).

This has motivated closer examination of how predictive power varies across layer depth. Kuribayashi et al. (2025) examined several behavioural measures, including gaze durations, self-paced reading times, N400 responses, and MAZE processing times. They found that relatively fast measures such as gaze duration were better predicted by surprisal from shallower layers, whereas slower measures such as N400 and MAZE showed stronger predictive power in deeper layers. This suggests that the depth at which LLM surprisal is most predictive may vary with the temporal or processing scope of the human measure. Related probing studies have also shown that deeper layer representations tend to encode more contextualized information (Peters et al., 2018; Vulic et al.´ , 2020; Jin et al., 2025). These findings are consistent with the view that layer depth may reflect differences in the amount or scope of contextual information encoded across layers.

Recent studies further suggest that this relationship depends on the type of processing being examined. Kuribayashi et al. (2026) found that shallower layers better capture naturalistic reading, whereas deeper layers better capture processing difficulty in syntactically challenging constructions. Tsipidi et al. (2026) similarly found that the strongest predictors vary across eye-tracking measures, with representations from shallower layers performing particularly well for fast response measures and different patterns occurring for slower measures. These findings motivate examining how the predictive power of surprisal changes across layer depth and across gaze measures.

## 2.3 Research Questions

The literature reviewed above suggests two relevant sources of variation in reading behaviour. First, readers with different levels of proficiency may differ in their reliance on lexical and contextual information during reading. Second, the predictive power of LLM surprisal varies across layer depth and across measures that capture different aspects of the reading process. We therefore examine whether the layer-wise distribution of predictive power varies across reader proficiency and across gaze measures. We address this question through the following research questions:

• RQ1: Does reader proficiency affect the distribution of predictive power of LLM surprisal across layer depth?

• RQ2: Does this distribution differ between first-pass gaze duration and total gaze duration?

## 3 Method

We investigate the relationship between human gaze patterns and LLM internal representations through a layer-wise surprisal framework. We focus on non-native English readers for the experiment.

## 3.1 Dataset

We utilize the MECO L2 dataset (Waves 1 and 2) (Kuperman et al., 2023, 2025), which provides eye-tracking data from L2 English speakers with 21 different native language backgrounds who read the same English texts. The participants read 12 encyclopedic texts, ranging from 100 to 200 words in length, and answered comprehension questions after each text.

Our analysis uses two word-level gaze measures, FPGD and TGD. FPGD is defined as the total fixation time on a word during the first pass before eyes move away from that word, capturing relatively early processing during reading. TGD is defined as the total fixation time on a word, including all subsequent fixations, capturing gaze behaviour over a broader processing span. FPGD and TGD are computed over the same eye-tracking observations<sup>2</sup>. Both measures are log-transformed prior to statistical analysis, and log-FPGD and log-TGD are used as dependent variables in our main analyses.

## 3.2 Reader Profiling

To investigate the impact of vocabulary proficiency, we use LexTALE scores (Lemhöfer and Broersma, 2012), which are included in the MECO L2 metadata. LexTALE is a lexical decision task in which participants identify whether a given string is a real word or not, providing an efficient estimate of vocabulary size. Based on these scores, we categorize readers into quartiles and focus our analysis on the top (High-Lex) and bottom (Low-Lex) groups.

<table><tr><td></td><td>All</td><td>High-Lex</td><td>Low-Lex</td></tr><tr><td>LexTALE Score</td><td></td><td>≥ 83.75</td><td>&lt; 66.25</td></tr><tr><td>Group Size</td><td>1001</td><td>276</td><td>228</td></tr><tr><td>Total Trials</td><td>8597</td><td>2519</td><td>1845</td></tr></table>

Table 1: Statistics of reader groups from MECO L2. Trials are the number of times each participant read each text.

Table 1 summarizes the participants in the High-Lex and Low-Lex groups. These groups roughly correspond to advanced (C1-C2) and intermediate (B1-lower B2) proficiency levels, respectively (Lemhöfer and Broersma, 2012).

## 3.3 LLMs and Layer-wise Surprisal Calculation

We utilize the internal representations of several LLMs, including Gemma 3 (4B, 12B, 27B) (Gemma Team, 2025), Qwen 2.5 (7B, 14B, 32B, 72B) (Qwen Team, 2025), OLMo 2 (7B, 13B, 32B) (OLMo Team, 2025), and Llama 3.1 (8B, 70B) (Llama Team, 2024). We use base models instead of instruction-tuned models.

Building on Kuribayashi et al. (2025), we apply the Logit-Lens technique<sup>3</sup> to extract layer-wise information. This method applies the LLM’s final output head directly to the hidden states at each layer. This allows us to derive a probability distribution for each word and calculate its surprisal at every layer, rather than only at the final layer. We calculate these values by providing the full context of each text to the LLMs. For words split into multiple subwords, individual surprisal values are summed to represent the surprisal of the entire word.

Through this process, we obtain a surprisal value for every word at every layer of each LLM. These values serve as the basis for evaluating layer-wise predictive power in the following analysis.

## 3.4 Statistical Analysis

## 3.4.1 LMM Analysis

We use linear mixed-effects models (LMMs) (Baayen et al., 2008), a type of regression model, to quantify the layer-wise relationship between LLM surprisal and gaze duration while accounting for repeated observations from participants and reading trials. Here, the dependent variable is log-FPGD or log-TGD, and the explanatory variables include word length, word frequency, and surprisal from LLM layers. We conduct the layer-wise analyses separately for the High-Lex and Low-Lex reader groups.

For each reader group G and layer k, we compare a baseline LMM with a full LMM that additionally includes current-word surprisal. Let Y denote the log-transformed gaze measure, either log-FPGD or log-TGD. The baseline LMM is specified as:

$$
\begin{array} { r l } & { M _ { G , k } ^ { B a s e } : Y \sim L _ { w _ { i } } + L _ { w _ { i - 1 } } + L _ { w _ { i - 2 } } } \\ & { + F _ { w _ { i } } + F _ { w _ { i - 1 } } + F _ { w _ { i - 2 } } + S _ { w _ { i - 1 } , k } + S _ { w _ { i - 2 } , k } } \\ & { \qquad + \mathrm { R a n d o m } \mathrm { E f f e c t s } , } \end{array}
$$

where $w _ { i }$ denotes the current word, and $w _ { i - 1 }$ and $w _ { i - 2 }$ denote the two preceding words. L and F denote the word length (the number of characters in a word) and Zipf frequency, respectively. Frequency estimates are obtained using the wordfreq library (Speer, 2022). $S _ { w _ { i - 1 } , k }$ and $S _ { w _ { i - 2 } , k }$ denote the surprisal values of the two preceding words extracted from layer k. Random effects include random intercepts for participants and reading trials.

The full LMM adds the surprisal of the current word from the same layer k as the ninth variable:

$$
M _ { G , k } ^ { F u l l } : M _ { G , k } ^ { B a s e } + S _ { w _ { i } , k } .
$$

Including information from the preceding words in the baseline controls for spillover effects (Rayner, 1998) and allows us to isolate the unique contribution of current-word surprisal at each layer. The same random-effects structure is used for the baseline and full LMMs.

Since the baseline and full LMMs differ in their fixed-effects structure, we fit all LMMs using maximum-likelihood (ML) instead of restricted maximum-likelihood (REML) estimation for comparison. For each reader group G and layer k, we calculate the unique contribution of current-word surprisal as the increase in LMM log-likelihood:

$$
\Delta L L _ { G , k } = L L ( M _ { G , k } ^ { F u l l } ) - L L ( M _ { G , k } ^ { B a s e } ) .
$$

Here, LL(M) is the log-likelihood of the regression model M, which measures how well the regression model fits the observed data. Larger values of $\Delta L L _ { G , k }$ indicate a greater contribution of current-word surprisal to explaining the corresponding gaze measure at that layer. Since the full LMM is a nested version of the baseline LMM, the maximized $\Delta L L _ { G , k }$ is theoretically non-negative<sup>4</sup>. LMMs are fitted using the MixedLM class from the statsmodels library (Seabold and Perktold, 2010).

## 3.4.2 Predictive Depth

To summarize where the predictive power of surprisal is distributed across layers, we define Predictive Depth P D as the normalized weighted average of layers:

$$
P D _ { G } = \frac { \displaystyle \sum _ { k = 1 } ^ { N } k \cdot \Delta L L _ { G , k } } { \displaystyle N \cdot \sum _ { k = 1 } ^ { N } \Delta L L _ { G , k } } ,
$$

where the weights are $\Delta L L _ { G , k }$ and N is the total number of layers.

Predictive Depth represents the center of the layer-wise distribution of the predictive power of current-word surprisal to explain the corresponding gaze measure. Larger Predictive Depth indicates that the predictive power is distributed towards relatively deeper layers. We compute Predictive Depth separately for log-FPGD and log-TGD for comparing both across reader groups and across gaze measures.

## 4 Results

## 4.1 Layer-wise Predictive Profiles

Before comparing Predictive Depth across reader groups and gaze measures, we first examine the layer-wise predictive power of surprisal. As a result, surprisal from internal layers generally provides better explanations for gaze durations than surprisal from the final output layer, especially for log-FPGD, while the predictive power is not uniformly distributed across layers.

Figure 2 shows representative ∆LL curves from each tested LLM family for both FPGD and TGD, while the full set of plots can be found in Appendix A. Gemma 3 4B and OLMo 2 13B exhibit multiple peaks across layers, while Qwen 2.5 72B and Llama 3.1 70B show stronger predictive power towards deeper layers. The detailed profiles also differ between FPGD and TGD, with some LLMs showing stronger predictive power near the final layers for TGD. Despite these broad tendencies, the detailed profiles vary across LLMs and gaze measures, indicating that gaze-predictive surprisal is distributed differently across LLM architectures.

## 4.2 Impact of Vocabulary Proficiency

Vocabulary proficiency is systematically associated with the Predictive Depth of surprisal for FPGD. As summarized in Table 2, $P D _ { \mathrm { L o w } }$ is deeper than $P D _ { \mathrm { H i g h } }$ for 10 out of 12 tested models for FPGD, with a mean difference of +0.024. Although the magnitude of the difference varies across LLMs, the overall pattern is observed across multiple LLM families. As illustrated in Figure 2 (top), Low-Lex readers often show relatively stronger predictive power in deeper layers, shifting Predictive Depth towards greater depths.

This difference between proficiency groups is substantially weaker for TGD. $P D _ { \mathrm { L o w } }$ is deeper than $P D _ { \mathrm { H i g h } }$ in only 4 out of 12 LLMs, and the mean difference is +0.003. Therefore, while readers with less vocabulary tend to show deeper Predictive Depth for FPGD, the corresponding difference is small for TGD.

## 4.3 FPGD vs TGD

We also compare the Predictive Depth between FPGD and TGD within each reader group. As summarized in Table 2, TGD has a deeper Predictive Depth than FPGD for all 12 LLMs in High-Lex, with a mean difference of +0.050. The same tendency is observed in Low-Lex, where TGD has a deeper Predictive Depth than FPGD for 10 out of 12 LLMs, with a mean difference of +0.030. Therefore, the shift towards deeper layers is observed in both groups, but its magnitude is larger in High-Lex, consistent with the smaller proficiency difference observed in TGD.

## 5 Generalization to Unseen Texts

The preceding analyses show that the layer-wise relationship between LLM surprisal and reading behaviour varies with both vocabulary proficiency and gaze measure. An important question is whether these relationships generalize beyond the texts used for LMM fitting. To address this question, we conduct a leave-one-out (LOO) analysis. Since MECO L2 contains 12 texts, we fit the LMMs on 11 texts and evaluate the learned relationships on the remaining unseen text in each fold.

![](images/7885fa1530052e4831238668addce6c2a5905ae6566282a607b371cab2201fb8.jpg)  
Figure 2: Plots of layer-wise ∆LL for both FPGD (top) and TGD (bottom) from each LLM family. $\Delta L L$ values on the y-axis are normalized per token and multiplied by 1000 for better visualization.

<table><tr><td colspan="2" rowspan="2">LLM Size</td><td rowspan="2">N</td><td colspan="2">FPGD</td><td colspan="3">TGD</td><td colspan="2">TGD-FPGD High</td></tr><tr><td>Low High</td><td>L-H</td><td>Low</td><td>High</td><td>L-H</td><td>Low</td></tr><tr><td rowspan="3">Gemma 3</td><td>4B</td><td>34</td><td>0.668 0.553</td><td>+0.115</td><td>0.629</td><td>0.586</td><td>+0.043</td><td>-0.039 +0.033</td></tr><tr><td>12B</td><td>48</td><td>0.613 0.548</td><td>+0.065</td><td>0.678</td><td>0.599</td><td>+0.079</td><td>+0.065 +0.051</td></tr><tr><td>27B</td><td>62</td><td>0.523 0.549</td><td>-0.026</td><td>0.674</td><td>0.673</td><td>+0.001</td><td>+0.151 +0.124</td></tr><tr><td rowspan="4">Qwen 2.5</td><td>7B</td><td>28</td><td>0.558</td><td>0.549 +0.009</td><td>0.565</td><td>0.587</td><td>-0.022</td><td>+0.000</td></tr><tr><td>14B</td><td>48</td><td>0.520 0.507</td><td>+0.014</td><td>0.558</td><td>0.559</td><td>-0.001</td><td>+0.039 +0.038</td></tr><tr><td>32B</td><td>64</td><td>0.509 0.485</td><td>+0.024</td><td>0.545</td><td>0.540</td><td>+0.004</td><td>+0.052</td></tr><tr><td>72B</td><td>80</td><td>0.552 0.520</td><td>+0.031</td><td>0.558</td><td>0.560</td><td>-0.003</td><td>+0.036 +0.055 +0.006 +0.040</td></tr><tr><td rowspan="3">OLMo 2</td><td>7B</td><td>32</td><td>0.525</td><td>0.520 +0.005</td><td>0.545</td><td>0.560</td><td>-0.015</td><td>+0.020 +0.040</td></tr><tr><td>13B</td><td>40</td><td>0.499</td><td>0.492 +0.008</td><td>0.543</td><td>0.553</td><td>-0.010</td><td>+0.044 +0.061</td></tr><tr><td>32B</td><td>64</td><td>0.534 0.535</td><td>-0.001</td><td>0.561</td><td>0.575</td><td>-0.013</td><td>+0.027 +0.040</td></tr><tr><td rowspan="2">Llama 3.1</td><td>8B</td><td>32</td><td>0.640</td><td>0.620 +0.020</td><td>0.658</td><td>0.676</td><td>-0.018</td><td>+0.018 +0.056</td></tr><tr><td>70B</td><td>80</td><td>0.596</td><td>0.574 +0.021</td><td>0.586</td><td>0.589</td><td>-0.003</td><td>-0.010 +0.015</td></tr></table>

Table 2: Predictive Depth for FPGD and TGD in the Low-Lex and High-Lex groups across all tested LLMs. N represents the number of layers in each model. The last two columns show the difference in P D between TGD and FPGD for each group. Positive differences are shown in bold.

## 5.1 Leave-One-Out Evaluation

For this LOO analysis, we focus on FPGD, our primary gaze measure, to evaluate whether the main finding on vocabulary proficiency transfers to unseen texts. For each LOO fold, LMMs for High-Lex and Low-Lex readers are fitted and evaluated separately. The internal layer used for evaluation is selected exclusively from 11 training texts, based on the layer-wise $\Delta L L$ obtained from LMMs. Then, we evaluate the selected internal layer on the held-out text, and compare its predictive performance with that of the final LLM layer and a surface-feature baseline.

Specifically, we compare the following three LMMs:

$$
\begin{array} { r l } & { M ^ { \mathrm { S u r f a c e } } : \log \mathrm { F P G D } \sim L _ { w _ { i } } + L _ { w _ { i - 1 } } + L _ { w _ { i - 2 } } } \\ & { + F _ { w _ { i } } + F _ { w _ { i - 1 } } + F _ { w _ { i - 2 } } + \mathrm { R a n d o m ~ E f f e c t s } , } \end{array}
$$

$$
M ^ { \mathrm { F i n a l } } = M ^ { \mathrm { S u r f a c e } } + S _ { w _ { i } , N } + S _ { w _ { i - 1 } , N } + S _ { w _ { i - 2 } , N } ,
$$

$$
M ^ { \mathrm { S e l } } = M ^ { \mathrm { S u r f a c e } } + S _ { w _ { i } , k ^ { * } } + S _ { w _ { i - 1 } , k ^ { * } } + S _ { w _ { i - 2 } , k ^ { * } } ,
$$

where $M ^ { \mathrm { F i n a l } }$ denotes the LMM using the final LLM layer $N _ { \ast }$ , and $M ^ { S e l }$ denotes the LMM using the selected internal layer $k ^ { * }$ with the largest $\Delta L L$ on the training texts.

This predictive analysis slightly differs from the main layer-profile analysis. In the main analysis, previous-word surprisal is included in the baseline LMM so that ∆LL isolates the unique contribution of current-word surprisal. In the LOO evaluation, surprisal of current and previous words from the selected layer is used jointly to predict log-FPGD, since the goal of this analysis is to evaluate how well the layer representations predict FPGD on an unseen text.

In this LOO analysis, while LMM fitting on the 11 texts is performed on individual-level observations, evaluation on the held-out text is based on word-level averages within each proficiency group with fixed effects only. Therefore, the resulting scores reflect word-level variation at the group level rather than individual-level differences.

## 5.2 Evaluation Metrics

We evaluate predictive performance on the heldout text in two ways. First, we calculate mean absolute error (MAE) between the observed and predicted log-FPGD for each word. Second, we compute Spearman correlations between residual word-level log-FPGD and current-word surprisal from the selected internal layer and the final layer.

MAE measures how closely each LMM predicts the observed gaze durations for words in the heldout text. We calculate MAE for each LOO fold and then average the MAE values across the 12 folds. The correlation analysis evaluates whether layerspecific surprisal is associated with word-level differences that are not captured by the Surface LMM. To obtain these differences, we define residual log-FPGD $r _ { G , w }$ as:

$$
r _ { G , w } = \overline { { Y } } _ { G , w } - \hat { Y } _ { G , w } ^ { \mathrm { S u r f a c e } } ,
$$

where $\overline { { Y } } _ { G , w }$ is the observed log-FPGD averaged within each group G for word $w ,$ , and $\hat { Y } _ { g , w } ^ { \mathrm { S u r f a c e } } \ \mathrm { i s }$ the predicted log-FPGD from the Surface LMM. We then compute Spearman correlations between $r _ { G , w }$ and current-word surprisal separately for the selected internal layer and the final layer. We then report the mean correlation coefficients across the 12 folds.

## 5.3 Generalization Results

Table 3 summarizes the LOO evaluation results. Across LLMs, adding surprisal generally produces lower MAE than the Surface LMM. The Selected LMM achieves lower MAE than the Surface LMM for all 12 LLMs in both proficiency groups, and also outperforms the Final LMM in 21 out of 24 cases. However, the absolute reductions in MAE are small across LLMs and proficiency groups. Thus, although surprisal from the selected internal layer generally performs better on unseen texts, the overall improvement in MAE remains small.

A similar pattern is observed in the correlation analysis. Residual log-FPGD shows stronger correlations with surprisal from the selected internal layer than with surprisal from the final layer in 19 out of 24 cases across LLMs and proficiency groups. Nevertheless, the absolute correlation values remain weak to modest, with the strongest condition reaching approximately $\rho = 0 . 2 4$ , indicating that layer-specific surprisal captures only part of the word-level variation that remains after the Surface LMM. Together with the MAE results, this suggests that the layer-wise relationships identified on the training texts transfer to unseen texts, but that the strength of this transfer is limited.

<table><tr><td colspan="5">Low-Lex</td></tr><tr><td>LLM</td><td>MAE (×100)</td><td>Final Selected</td><td>Final</td><td> $\rho$  Selected</td></tr><tr><td rowspan="2">(Surface)</td><td></td><td>8.96</td><td></td><td></td></tr><tr><td>4B</td><td>8.94 8.87</td><td>0.089</td><td>0.117</td></tr><tr><td rowspan="2">Gemma 3 12B</td><td></td><td>8.95 8.87</td><td>-0.030</td><td>0.125</td></tr><tr><td>27B</td><td>8.95</td><td>8.88 -0.046</td><td>0.161</td></tr><tr><td rowspan="4">Qwen 2.5</td><td>7B</td><td>9.00 8.83</td><td>0.064</td><td>0.140</td></tr><tr><td>14B</td><td>8.94</td><td>8.93 -0.029</td><td>0.095</td></tr><tr><td>32B</td><td>8.90</td><td>8.89 0.149</td><td>0.161</td></tr><tr><td>72B</td><td>8.99</td><td>8.80 0.073</td><td>0.142</td></tr><tr><td rowspan="3">OLMo 2</td><td>7B</td><td>8.92 8.84</td><td>0.100</td><td>0.170</td></tr><tr><td>13B</td><td>8.92 8.95</td><td>0.118</td><td>0.116</td></tr><tr><td>32B</td><td>8.92 8.91</td><td>0.132</td><td>0.124</td></tr><tr><td rowspan="2">Llama 3.1</td><td>8B</td><td>8.90 8.86</td><td>0.098</td><td>0.168</td></tr><tr><td>70B 8.89</td><td>8.67</td><td>0.135</td><td>0.239</td></tr><tr><td colspan="5">High-Lex</td></tr><tr><td rowspan="2">LLM</td><td></td><td>|Final Selected|</td><td>Final</td><td>Selected</td></tr><tr><td></td><td>8.31</td><td></td><td></td></tr><tr><td rowspan="2">(Surface)</td><td>4B</td><td>8.29 8.25</td><td>-0.012</td><td>0.132</td></tr><tr><td>Gemma 3 12B 27B</td><td>8.29 8.30</td><td>8.28 -0.027 8.27 -0.044</td><td>0.113 0.125</td></tr><tr><td rowspan="4">Qwen 2.5</td><td>7B</td><td>8.31</td><td>8.19</td><td>0.100 0.141</td></tr><tr><td>14B</td><td>8.29</td><td>8.18 0.003</td><td>0.114</td></tr><tr><td>32B</td><td>8.20</td><td>8.27 0.137</td><td>0.128</td></tr><tr><td>72B</td><td>8.31</td><td>8.16 0.064</td><td>0.112</td></tr><tr><td rowspan="2">OLMo 2</td><td>7B</td><td>8.25 8.13</td><td>0.128</td><td>0.146</td></tr><tr><td>13B</td><td>8.26 8.21</td><td>0.147</td><td>0.141</td></tr><tr><td rowspan="2">Llama 3.1</td><td>32B</td><td>8.27 8.27</td><td></td><td>0.142 0.101</td></tr><tr><td>8B 70B</td><td>8.25</td><td>8.20</td><td>0.132 0.145</td></tr></table>

Table 3: Leave-one-out evaluation results for Low-Lex and High-Lex readers. Selected denotes the internal layer chosen based on $\Delta L L$ on the training texts. MAE values are multiplied by 100 for readability.

## 5.4 Illustrative Example

To illustrate the LOO results at the word level, Table 4 shows a example from Llama 3.1 70B, where the layer 46 of 80 is selected with the largest ∆LL on the training texts. The word improves with the largest absolute residual $r _ { G , w }$ under the Surface LMM and its surrounding words are shown.

<table><tr><td>Word</td><td>log-FPGD Obs. Surf. 1 Fin.</td><td>Sel.</td><td> $r _ { G , w }$  Surf. Fin.</td><td>Surprisal Sel.</td></tr><tr><td>related</td><td>5.82 5.73</td><td>5.73 5.71</td><td>0.09</td><td>1.98 2.63</td></tr><tr><td>note,</td><td>5.71 5.59</td><td>5.57 5.53</td><td>0.12 0.14</td><td>0.00</td></tr><tr><td>sleep</td><td>5.48 5.65</td><td>5.635.66</td><td>-0.17</td><td>1.32 13.01</td></tr><tr><td>also</td><td>5.41 5.52</td><td>5.51 5.49</td><td>-0.111.53</td><td>0.02</td></tr><tr><td>improves</td><td>5.63 5.88</td><td>5.87 5.87</td><td>-0.251.57</td><td>8.87</td></tr><tr><td>concentration</td><td>6.25 6.06</td><td>6.05 6.08</td><td>0.192.12</td><td>7.95</td></tr><tr><td>and</td><td>5.34 5.37</td><td>5.36 5.38</td><td>-0.041.72</td><td>9.51</td></tr><tr><td>mental</td><td>5.75 5.75</td><td>5.75 5.77</td><td>0.004.03</td><td>12.59</td></tr><tr><td>alertness.</td><td>6.23 6.00</td><td>5.99 5.97</td><td>0.23 2.08</td><td>2.48</td></tr></table>

Table 4: Illustrative example for Llama 3.1 70B. The selected internal layer is layer 46 out of 80. improves is highlighted as one of the words with the largest residual $r _ { G , w }$ under the Surface LMM. Obs. denotes observed log-FPGD, while Surf., Fin., and Sel., denote the Surface, Final, and Selected LMM predictions, respectively. For each word, the prediction closest to the observed value is in bold.

The three LMMs often show similar log-FPGD predictions, which is consistent with the small MAE differences observed in Table 3. For improves, the Surface LMM overestimates its log-FPGD. While surprisal values from the selected and final layers differ substantially, these two LMMs produce similar predictions. A similar pattern can be seen in the surrounding words, where the selected and final layers sometimes assign substantially different surprisal values, while the resulting log-FPGD remain close. This example illustrates that surprisal values can differ across layers, while the resulting differences in predictions remain small.

## 6 Discussion

## 6.1 Gaze Measure and Predictive Depth

One of the primary findings of this study is that Predictive Depth is generally deeper for TGD than for FPGD. This difference is highly consistent across LLMs, occurring in all tested LLMs for High-Lex and in most LLMs for Low-Lex. Since FPGD is restricted to first-pass reading while TGD also includes subsequent fixations, the two measures differ in both the timing and breadth of reading processes they capture. The deeper Predictive Depth observed for TGD suggests that surprisal from deeper layers becomes relatively more predictive as the gaze measure incorporates a broader span of processing.

One possible explanation is that this shift is related to the information encoded at different layer depths. Previous probing studies suggest that deeper LLM layers tend to encode more contextualized information (Peters et al., 2018; Vulic et al.´ , 2020; Jin et al., 2025). Therefore, TGD may be more sensitive than FPGD to surprisal representations that incorporate broader contextual information, as it includes processing that extends beyond the initial lexical access phase. This interpretation is consistent with the previous findings that relatively fast reading measures tend to be better predicted by shallower layers, whereas slower measures show stronger predictive power in deeper layers (Kuribayashi et al., 2025; Tsipidi et al., 2026).

Importantly, this pattern does not necessarily imply a direct mapping between layer depth and specific stages of human cognition. Rather, the consistent difference between FPGD and TGD suggests that layer depth may be related to the timing and breadth of the reading processes captured by different gaze measures.

## 6.2 Vocabulary Proficiency and Reading Processes

The difference between High-Lex and Low-Lex readers is more evident in FPGD, where Low-Lex readers tend to show deeper Predictive Depth than High-Lex readers. This difference becomes substantially weaker in TGD. Thus, vocabulary proficiency is more clearly reflected in first-pass reading than in the broader processing captured by TGD.

One possible interpretation is that readers with lower proficiency may rely on contextual information earlier when they encounter linguistic obstacles. Cognitive models of compensatory processing suggest that readers can recruit contextual information when faced with difficult vocabulary or syntax (Stanovich, 1980). Given that deeper layers tend to encode more contextualized information, the deeper Predictive Depth observed for Low-Lex readers in FPGD may be consistent with earlier reliance on such information during first-pass reading.

The smaller proficiency difference in TGD is also informative. Since TGD includes later and repeated fixations, it is likely to reflect a later stage of cognitive processing where contextual information has more opportunity to influence gaze behaviour in both proficiency groups. This would reduce the difference between High-Lex and Low-Lex readers in Predictive Depth, even if their first-pass reading

patterns differ.

## 6.3 Interpreting Predictive Depth

Predictive Depth should be interpreted as a summary of how the predictive power of surprisal is distributed across layers. A deeper Predictive Depth indicates that predictive power is weighted more towards deeper layers, but it does not imply greater reading difficulty or cognitive load. Likewise, our results do not assume a direct correspondence between LLM layers and specific stages of human cognition.

Instead, the observed findings suggest that layer depth may be informative about differences in the timing and breadth of the reading processes captured by gaze measures. The consistently deeper Predictive Depth for TGD than for FPGD supports this interpretation. Furthermore, the differences in FPGD between two proficiency groups suggest that the layer-wise distribution can vary across reader profiles. Predictive Depth provides a descriptive way to compare these layer-wise patterns without assigning fixed cognitive interpretations to each layer.

## 6.4 Transfer Beyond Training Texts

The leave-one-out analysis provides an external check on whether the layer-wise relationships identified in the main analysis extend beyond the texts used for fitting LMMs. Across LLMs, the selected internal layer generally performs better than the final layer on held-out texts, but the reduction in MAE is quite small in practice. The correlation analysis shows a similar pattern. Residual log-FPGD tends to correlate more strongly with surprisal from the selected internal layer than with that from the final layer, but the correlations remain weak to modest.

These results suggest that the advantage of internal layers is not limited to the texts on which the LMMs were fitted. At the same time, the small MAE improvements and modest correlations indicate that the transferable relationships are limited. The leave-one-out results provide out-of-sample support for the layer-wise analysis, but they do not imply that surprisal alone is sufficient for accurate prediction of gaze behaviour on unseen texts.

## 7 Conclusion

In this paper, we investigate how the layer-wise relationship between surprisal from LLMs and human reading behaviour varies with vocabulary proficiency and gaze measure. With respect to RQ1, vocabulary proficiency is associated with the layerwise distribution of predictive power, especially for FPGD. Across LLMs, Low-Lex readers tend to show deeper Predictive Depth than High-Lex readers for FPGD, while this difference is substantially smaller for TGD. Regarding RQ2, the distribution differs systematically between FPGD and TGD. TGD itself generally shows deeper Predictive Depth than FPGD in both groups. These findings suggest that where predictive power is concentrated across LLM layers may be related to the timing and breadth of the reading processes captured by different gaze measures, and that this relationship can vary with reader proficiency. Our leave-one-out analysis further shows that the advantage of selected internal layers extends to unseen texts, but the overall improvement in prediction is limited.

Future work should examine whether these findings generalize to other languages, datasets, and additional measures of reader proficiency. It will also be important to move beyond group-level comparisons and investigate whether individual reader characteristics can be incorporated directly into models of layer-wise surprisal and gaze behaviour.

## Limitations

While this study utilizes LexTALE for grouping readers, reading ability is composed of multiple aspects, and vocabulary knowledge is merely one component. Factors such as background knowledge and working memory capacity also play critical roles in the reading process. In addition, the analysis based on LexTALE groups does not capture continuous variation among readers. Future work could incorporate multiple proficiency measures at the individual level rather than discrete reader groups.

Also, our experiments are based on English reading data from the MECO L2 dataset. The layerwise relationships observed in this study may be influenced by the specific properties of the English language, particular texts in the corpus, or reader population in the dataset. Languages with different morphological and syntactic properties may show different patterns of layer-wise predictive power. Evaluating the same analysis across languages and datasets will be important for establishing the generalizability of the findings.

The linear mixed-effects models used in this study include random effects to account for repeated observations. However, richer randomeffects structures, such as random slopes or wordlevel predictors, may capture variation that is not represented in our study.

Predictive Depth is a summary statistic calculated from the distribution of ∆LL across layers. While layer depth is normalized within each LLM, equivalent normalized depth may not represent equivalent information across model architectures. Therefore, comparisons of Predictive Depth across LLMs should be interpreted as comparisons of layer-wise patterns rather than as evidence that particular depths have the same functional role across LLMs.

## Acknowledgments

This work is partially financed by the Ministerio de Ciencia, Innovación y Universidades, Agencia Estatal de Investigaciones: project CPP2023-010780 funded by MICIU/AEI/10.13039/501100011033 and by FEDER, UE (“Habilitando Modelos de Lenguaje Responsables e Inclusivos”). It also received funding from the European Union’s Horizon Europe research and innovation program under the Grant Agreement No. 101132431 (iDEM: Innovative and Inclusive Democratic Spaces for Deliberation and Participation). Views and opinions expressed are, however, those of the authors only and do not necessarily reflect those of the European Union. Neither the European Union nor the granting authority can be held responsible for them.

We used ChatGPT for language polishing and proofreading of text originally written by the authors.

## References

Sweta Agrawal and Marine Carpuat. 2024. Do text simplification systems preserve meaning? a human evaluation via reading comprehension. Transactions of the Associationfor Computational Linguistics, 12:432– 448.

Fernando Alva-Manchego, Carolina Scarton, and Lucia Specia. 2020. Data-driven sentence simplification: Survey and benchmark. Computational Linguistics, 46(1):135–187.

R.H. Baayen, D.J. Davidson, and D.M. Bates. 2008. Mixed-effects modeling with crossed random effects for subjects and items. Journal of Memory and Language, 59(4):390–412. Special Issue: Emerging Data Analysis.

Barbara Carretti, Erika Borella, Cesare Cornoldi, and Rossana De Beni. 2009. Role of working memory in explaining the performance of individuals with specific reading comprehension difficulties: A meta-analysis. Learning and Individual Differences, 19(2):246–251.

Gemma Team. 2025. Gemma 3 technical report. Preprint, arXiv:2503.19786.

John Hale. 2001. A probabilistic Earley parser as a psycholinguistic model. In Second Meeting of the North American Chapter of the Association for Computational Linguistics.

Michael Harrington and Mark Sawyer. 1992. L2 working memory capacity and l2 reading skill. Studies in second language acquisition, 14(1):25–38.

Mingyu Jin, Qinkai Yu, Jingyuan Huang, Qingcheng Zeng, Zhenting Wang, Wenyue Hua, Haiyan Zhao, Kai Mei, Yanda Meng, Kaize Ding, Fan Yang, Mengnan Du, and Yongfeng Zhang. 2025. Exploring concept depth: How large language models acquire knowledge and concept at different layers? In Proceedings of the 31st International Conference on Computational Linguistics, pages 558–573, Abu Dhabi, UAE. Association for Computational Linguistics.

J Peter Kincaid, Robert P Fishburne Jr, Richard L Rogers, and Brad S Chissom. 1975. Derivation of new readability formulas (automated readability index, fog count and flesch reading ease formula) for navy enlisted personnel. Defense Technical Information Center.

Walter Kintsch. 1988. The role of knowledge in discourse comprehension: a construction-integration model. Psychological review, 95(2):163–182.

Victor Kuperman, Sascha Schroeder, Cengiz Acartürk, Niket Agrawal, Dominick M Alexandre, Lena S Bolliger, Jan Brasser, César Campos-Rojas, Denis Drieghe, Dušica Filipovic Ður ´ devi ¯ c, and 1 others.´ 2025. New data on text reading in english as a second language: The wave 2 expansion of the multilingual eye-movement corpus (meco). Studies in Second Language Acquisition, 47(2):677–695.

Victor Kuperman, Noam Siegelman, Sascha Schroeder, Cengiz Acartürk, Svetlana Alexeeva, Simona Amenta, Raymond Bertram, Rolando Bonandrini, Marc Brysbaert, Daria Chernova, and 1 others. 2023. Text reading in english as a second language: Evidence from the multilingual eye-movements corpus. Studies in second language acquisition, 45(1):3–37.

Tatsuki Kuribayashi, Yohei Oseki, Ana Brassard, and Kentaro Inui. 2022. Context limitations make neural language models more human-like. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 10421–10436, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Tatsuki Kuribayashi, Yohei Oseki, Souhaib Ben Taieb, Kentaro Inui, and Timothy Baldwin. 2025. Large language models are human-like internally. Transactions ofthe Associationfor Computational Linguistics, 13:1743–1766.

Tatsuki Kuribayashi, Alex Warstadt, Yohei Oseki, and Ethan Gotlieb Wilcox. 2026. Dual alignment between language model layers and human sentence processing. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 46207–46223, San Diego, California, United States. Association for Computational Linguistics.

David LaBerge and S Jay Samuels. 1974. Toward a theory of automatic information processing in reading. Cognitive psychology, 6(2):293–323.

Kristin Lemhöfer and Mirjam Broersma. 2012. Introducing lextale: A quick and valid lexical test for advanced learners of english. Behavior research methods, 44(2):325–343.

Roger Levy. 2008. Expectation-based syntactic comprehension. Cognition, 106(3):1126–1177.

Llama Team. 2024. The llama 3 herd of models. Preprint, arXiv:2407.21783.

Byung-Doh Oh, Christian Clark, and William Schuler. 2022. Comparison of structural parsers and neural language models as surprisal estimators. Frontiers in Artificial Intelligence, 5:777963.

Byung-Doh Oh and William Schuler. 2023. Why does surprisal from larger transformer-based language models provide a poorer fit to human reading times? Transactions of the Association for Computational Linguistics, 11:336–350.

OLMo Team. 2025. 2 olmo 2 furious. Preprint, arXiv:2501.00656.

Charles A Perfetti and Lesley Hart. 2002. The lexical quality hypothesis. Precursors offunctional literacy, 11:67–86.

Matthew E. Peters, Mark Neumann, Luke Zettlemoyer, and Wen-tau Yih. 2018. Dissecting contextual word embeddings: Architecture and representation. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 1499– 1509, Brussels, Belgium. Association for Computational Linguistics.

Qwen Team. 2025. Qwen2.5 technical report. Preprint, arXiv:2412.15115.

Keith Rayner. 1998. Eye movements in reading and information processing: 20 years of research. Psychological bulletin, 124(3):372.

Skipper Seabold and Josef Perktold. 2010. statsmodels: Econometric and statistical modeling with python. In 9th Python in Science Conference.

Cory Shain, Clara Meister, Tiago Pimentel, Ryan Cotterell, and Roger Levy. 2024. Large-scale evidence for logarithmic effects of word predictability on reading time. Proceedings of the National Academy of Sciences, 121(10):e2307876121.

Advaith Siddharthan. 2014. A survey of research on text simplification. ITL-International Journal ofApplied Linguistics, 165(2):259–298.

Nathaniel J Smith and Roger Levy. 2013. The effect of word predictability on reading time is logarithmic. Cognition, 128(3):302–319.

Robyn Speer. 2022. rspeer/wordfreq: v3.0.

Keith E. Stanovich. 1980. Toward an interactivecompensatory model of individual differences in the development of reading fluency. Reading Research Quarterly, 16(1):32–71.

Eleftheria Tsipidi, Samuel Kiegeland, Francesco Ignazio Re, Tianyang Xu, Mario Giulianelli, Karolina Stanczak, and Ryan Cotterell. 2026. Probing for reading times. In Proceedings ofthe 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 12618–12642, San Diego, California, United States. Association for Computational Linguistics.

Ivan Vulic, Edoardo Maria Ponti, Robert Litschko,´ Goran Glavaš, and Anna Korhonen. 2020. Probing pretrained language models for lexical semantics. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 7222–7240, Online. Association for Computational Linguistics.

## A Plots of Predictive Power of Layer-wise Surprisal

![](images/010cb51b43d24d726f47d7f82bafe3c82bccede54dd4ef09e188ea97c6ab1f1a.jpg)  
Figure 3: ∆LL and Predictive Depth across all tested LLMs. As in Figure 2, the color blue refers to High-Lex, and the red refers to Low-Lex.

## B Implementation Details

## B.1 Surprisal Extraction

We use HuggingFace Transformers library to extract the log-probabilities for calculating surprisal values. All inputs are processed as plain text without additional prompt templates. While the interest areas of eye-tracking measurements often include adjacent punctuation marks, such as periods or quotation marks, we exclude these symbols from our surprisal analysis to focus on linguistic and cognitive processing of lexical items. In other words, for words followed by punctuation, its surprisal is calculated based solely on the constituent characters of the word itself.

## B.2 LMM Settings

To fit the linear mixed-effects models, we use the default optimizer lbfgs (Limited-memory BFGS) in the statsmodels library in Python. The maximum number of iterations is set to 1000. Other hyperparameters are set to default values.