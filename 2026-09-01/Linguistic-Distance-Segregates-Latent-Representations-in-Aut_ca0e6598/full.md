# Linguistic Distance Segregates Latent Representations in Automatic Speech Recognition Systems

Ting-Hui Cheng, Line Katrine Harder Clemmensen and Sneha Das Department of Applied Mathematics and Computer Science Technical University of Denmark {tiche, 1khc, sned}@dtu.dk

## Abstract

While automatic speech recognition (ASR) models have achieved remarkable improvements in recent years, performance disparities persist across different speaker populations. One such disparity is for speakers whose first languages (L1) are from families distant from English. This paper investigates the relationship between first language background and English ASR performance. Through empirical analysis, we observe that the correlation between speakers'L1 distance and ASR error rates yields a systematic effect on English Speech, with its strength varying across datasets and models. This association is statistically significant in a follow-up analysis accounting for dataset-level variation in Tweedie mixed-effects models (p < 0.001 across evaluated models). In addition, analysis of the latent space reveals a L1-based spatial segregation across deeper acoustic layers in the majority of evaluated architectures¹.

## 1 Introduction

Automatic Speech Recognition (ASR) systems are widely deployed in real-world applications. Ensuring robust performance across diverse speaker populations is therefore essential. Yet, performance disparities across speaker subgroups continue to be reported, raising questions about when and who the systems underperform for?

In this work, we define L1 speakers as those whose first language is English, and L2 speakers as those who acquired English as a second language. Prior work has documented ASR performance disparities across various speaker attributes. Differences have been observed with respect to gender (Attanasio et al., 2024) and racial background (Zolnoori et al., 2024). Beyond these demographic factors, L2 speakers exhibit higher Word Error Rates (WER) than L1 across commercial English ASR systems (Google, Amazon, Microsoft) (DiChristofano et al., 2022) and open-source models such as Whisper (Graham and Roll, 2024). Similar disparities have also been reported for Swedish (Cumbal et al., 2024) and Korean (Na et al., 2024).

Table 1: Overview of the datasets, including total samples (hours) and the number of accents.
<table><tr><td>Dataset</td><td>N. Samples (hr)</td><td>Accents</td></tr><tr><td>Speech Accent Archive (SAA) (Weinberger, 2015)</td><td>2,015 (15.5)</td><td>17</td></tr><tr><td>Fair-Speech (Veliche et al., 2024)</td><td>26,471 (54.5)</td><td>27</td></tr><tr><td>L2-ARCTIC (L2Arc) (Zhao et al., 2018)</td><td>26,867 (27.1)</td><td>6</td></tr><tr><td>EdAcc (Sanabria et al., 2023)</td><td>17,965 (32.7)</td><td>20</td></tr><tr><td>ALLSSTAR (Bradlow et al., 2010)</td><td>699 (20.9)</td><td>22</td></tr><tr><td>AFRISPEECH-200 (Afri200) (Olatunji et al., 2023)</td><td>6318 (18.77)</td><td>108</td></tr></table>

These performance gaps are commonly attributed to speaker characteristics, including geographic origin (Jahan et al., 2025) and firstlanguage influence (Chan et al., 2022). Prior research suggests that L1-L2 disparities may stem from systematic structural differences between languages. Linguistic research has shown that L1-L2 distance plays an important role in second language processing and acquisition (Kuperman, 2025). Similar effects have been observed in NLP systems, where linguistic proximity impacts crosslingual transfer performance, such as in automatic abusive language detection (Eronen et al., 2022). These findings suggest that linguistic distance may systematically impact model generalization across languages. However, its role in explaining disparities across ASR remains limited.

In this work, we examine whether L1-L2 ASR performance gaps on English speech are systematically associated with structural Linguistic Distance (LD) between speakers' L1 and English. Furthermore, we conduct a comparative analysis across distinct ASR architectures to evaluate their respective sensitivities to L2 speech. We observe that as input representations propagate into deeper layers, all evaluated models maintain a clear segregation between L2 acoustic representations and those of English L1 speakers. This systematic spatial separation implies that current state-of-theart architectures isolate accented features rather than robustly normalizing them, emphasizing the need for enhanced accent-invariant representation learning in future acoustic frameworks.

## 2 Related Work

Several studies have examined the impact of speakers' L1 on ASR performance, showing that recognition accuracy varies with speaker language and accent (Chan et al., 2022; Liu et al., 2025; Jahan et al., 2025). While prior work documents L1-L2 performance disparities, it has largely focused on quantifying these gaps. Issa et al. (2026) investigated Whisper large-v3's performance on L2 Egyptian Arabic speech and revealed existing correlation across WER and high accentedness (rated by human) but comprehensibility only had a marginal positive association. Bartelds et al. (2021) showed that there exists a correlation between acoustic distances in neural speech representation and human native-likeness judgments. Tsoukala et al. (2026) revealed a clear performance gradient with dialectal distance from Standard Modern Greek. Similar research has been shown in Finnish (Törö et al., 2025). Törö et al. (2025) further showed that language embeddings cluster meaningfully, with distances correlating with geographic and lexical distances across languages, while Prasad and Jyothi (2020) found that accentrelated information is largely encoded in early ASR layers. While prior studies link embeddings to linguistic and accent features, their relevance to L2- ASR performance disparities remains unexplored.

## 3 Methodology

Language linguistic distances: We evaluate two speaker populations: first language speakers in English (L1) and speakers who acquired English as a second language (L2). To quantify the linguistic dissimilarity between each speaker's first language and English, we apply method proposed by Vincent and Johannes (2020)2, which provides a single composite score of linguistic relatedness between languages for subsequent analysis. The resulting linguistic distance (LD) score ranges from 0 to 100 (identical to completely unrelated languages).

![](images/619037dfb37150bcbf662a25317d3fea8917722445091c81baffb39ed5f875d9.jpg)  
Figure 1: Coefficient (β) for Linguistic Distance (LD) relative to English ASR model performance in WER, WIL and SemDist metrics. Error bars represent 95% confidence intervals, and the value shown is the adjusted $R ^ { 2 }$ (\*: $p \ < \ 0 . 0 5$ \*\*: $p \ < \ 0 . 0 1$ \*\*\*: $p \ < \ 0 . 0 0 1 )$ Statistical assumptions were verified and met.

Regression model: We performed a regression analysis³ with $L D$ as the main parameter to quantify the impact of speakers’ L1 transfer on English ASR performance.

$$
Y _ { i } ^ { m e t r i c } = \beta _ { c o n s t } + \beta _ { L D } x _ { L D , i } + \epsilon ,\tag{1}
$$

where $Y _ { i } ^ { m e t r i c }$ is the mean of performance metric evaluated on language i; $\beta _ { c o n s t }$ represents constant intercept term, capturing the model's baseline error rate for L1 speakers; $x _ { L D , i }$ denotes the LD from language i to English; $\beta _ { L D }$ is the slope coefficient quantifying model's performance sensitivity to increasing linguistic distance; and $\epsilon _ { i } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } )$ is the residual error term. Both $Y _ { i } ^ { m e t r i c }$ and $x _ { L D , i }$ are normalized via Min-Max scaling⁴ to ensure a uniform baseline for comparison.

Generalized Linear Mixed Model: We conduct a follow-up investigation into the association between speakers' L1 LD from English and ASR performance across datasets by accounting for dataset differences. Towards that, we employ a generalized linear mixed model with a Tweedie likelihood⁵ (Ma and Jørgensen, 2007), shown in Equation 2. The Tweedie family is suitable for continuous, non-negative response variables and can accommodate a point mass at zero through its compound Poisson-Gamma representation, making it appropriate for modeling ASR errorrelated measures.

$$
\begin{array} { r l } & { Y _ { g i } ^ { \mathrm { W E R } } \mid u _ { g } \sim \mathrm { T w e e d i e } ( \mu _ { g i } , \phi , p ) , } \\ & { \quad \quad \log ( \mu _ { g i } ) = \beta _ { 0 } + \beta _ { 1 } x _ { L D , i } + u _ { g } , } \\ & { \quad \quad \quad u _ { g } \overset { \mathrm { i . i . d . } } { \sim } \mathcal { N } ( 0 , \sigma _ { u } ^ { 2 } ) , } \end{array}\tag{2}
$$

where $Y _ { g i } ^ { \mathrm { W E R } }$ denotes the WER for observation i in dataset $g , \mu _ { g i }$ is its conditional mean with $N g$ represents the number of observation within the group, $\sigma _ { u } ^ { 2 }$ is the variance component, $\phi$ is the dispersion parameter, and $p$ is the Tweedie power parameter constrained to $1 < p < 2$ . The conditional mean and variance are given by

$$
E \left[ Y _ { g i } ^ { \mathrm { W E R } } \mid u _ { g } \right] = \mu _ { g i } , \quad \mathrm { V a r } \left( Y _ { g i } ^ { \mathrm { W E R } } \mid u _ { g } \right) = \phi \mu _ { g i } ^ { p } .
$$

The coefficient $\beta _ { 1 }$ represents the population-level LD-WER association, while $u _ { g }$ captures datasetspecific deviations from overall intercept. The model trends for WIL and SemDist are presented in Appendix C.

Embedding distance and correlation: To evaluate how L2 phonetic variants are encoded dynamically within the internal representations across the network's depth, we compute the Spearman rank correlation coefficient (ρ) between $L D$ and the layer-specific (l) Embedding Distance (ED1) of L2 individual utterances relative to a English L1 speakers' utterance baseline. Following similar approaches utilized in recent literature to measure embedding distance (Durrheim et al., 2023; Bolukbasi et al., $2 0 1 6 ;$ Ohamadike et al., 2025) , we apply cosine distance as our core metric. For any given target utterance embedding vector $E _ { \mathrm { S } , l }$ , we calculate its cosine distance to the centroid of the L1 English reference embeddings, $C ( E _ { \mathrm { E } , l } )$ , as:

$$
E D _ { l } = 1 - { \frac { C ( E _ { \mathrm { E } , l } ) \cdot E _ { \mathrm { S } , l } } { \Vert C ( E _ { \mathrm { E } , l } ) \Vert \Vert E _ { \mathrm { S } , l } \Vert } }\tag{3}
$$

where the L1 English centroid $C ( E _ { \mathrm { E } , l } )$ represents the geometric mean of all L1's utterance embeddings in all evaluating datasets within the corresponding network layer l, calculated as:

$$
C ( E _ { \mathrm { E } , l } ) = \frac { 1 } { N _ { \mathrm { E } } } \sum _ { i = 1 } ^ { N _ { \mathrm { E } } } E _ { \mathrm { E } , l , i }\tag{4}
$$

![](images/64e1f51759ef3fd4cc7bdec72014983eb0230396c183a10f4ee75d958f6b68d2.jpg)  
Figure 2: Association between L1's Linguistic Distance (LD) from English and ASR performance (WER). Dots represent mean WER for each L1 group, and lines show the fitted relationships between WER and L1's LD from Tweedie mixed-effects models, with dataset as a random effect. All ASR models show a statistically significant positive effect of LD $( p < 0 . 0 0 1 )$ . Statistical assumptions were verified and met.

Here, $N _ { \mathrm { E } }$ denotes the total number of L1 speaker utterances, and $E _ { \mathrm { E } , l , i }$ represents the embedding vector of the i-th L1 speaker sample in layer l. The layer-wise Spearman rank correlation $( \rho _ { l } )$ between the embedding displacement distribution and the structural $L D$ is calculated as:

$$
\rho _ { l } = \frac { \mathrm { c o v } ( R [ L D ] , R [ E D _ { l } ] ) } { \sigma _ { R [ L D ] } \sigma _ { R [ E D _ { l } ] } }\tag{5}
$$

where $R [ L D ]$ and $R [ E D _ { l } ]$ represent the assigned ranks of the LD and $E D _ { l }$ vectors, respectively; cov(·) denotes the covariance of these ranks; and σ represents their respective standard deviations.

Experiment Setup: We choose to evaluate English-capable ASR models that are architecturally different to each other, to assess bias against L2 speakers, including the transformer-based Whisper (Radford et al., 2022) (Small, 244M, and Large, 1.5B), as well as the Conformer-based Parakeet-TDT-0.6B-v3 and Canary-1B-v2 (Sekoyan et al., 2025). Table 1 shows the selected datasets during our experiment which are chosen because of their rich L1 backgrounds among speakers who acquired English as a second language.

Beyond the WER, we incorporate Word Information Loss (WIL) (Morris et al., 2004) to better align our results with human perceptual judgment, and Semantic Distance (SemDist) (Kim et al., 2021) to capture errors that may be lexically significant but semantically minor. SemDist, specifically, uses RoBERTa-base to generate semantic embeddings of the reference and hypothesis transcripts, quantifying their similarity via cosine distance.

![](images/8d320fd93f2f3f862426c07d8f0739464c5f797b82db3e0060af3767d2985646.jpg)  
Figure 3: Layer-wise t-SNE visualizations of Whisper Small (perplexity = 30, 1,000 iterations, random state = 20): Linguistic clustering intensifies with model depth, with FairSpeech driving high-WER/low-LD observations.

## 4 Results

Correlation between LD and performance: As illustrated in Figure 1, LD exhibits a general positive association with all evaluated error metrics across nearly all datasets, with the sole exception of Fair-Speech. The association is particularly pronounced and statistically significant for the SAA dataset. This pattern suggests that ASR performance tends to deteriorate as the LD between speakers' L1 and English increases. Critically, this pattern remains stable across different model architectures, confirming that increased LD from English systematically drives higher downstream recognition error rates. Across all linear regressions, LD accounts for between 1% and 56% of the total performance variance. However, LD is an explanatory factor and its explanatory power varies substantially across datasets and model settings: LD explains a larger proportion of the variance in settings with higher adjusted R² values, while its explanatory power is weaker for datasets such as EdAcc or Afrispeech-200, (see Appendix A for further discussion). Overall, these results suggest that LD exerts a small-to-moderate yet systematic influence on model errors, given that ASR performance is simultaneously shaped by speaker-specific variables such as L2 proficiency, age, and gender.

Conversely, Fair-Speech dataset shows negative regression coefficients across all evaluated models and performance metrics, with the total variance explained by LD ranging between 2% and 31%. To investigate this anomaly, we conducted a preliminary human perception study6 (n = 4) to evaluate whether listeners could reliably distinguish between the corpus's designated L1 and L2 speaker cohorts. Our perceptual results reveal that the acoustic boundaries between L1 and L2 speakers in this dataset are remarkably thin. While the average classification accuracy across evaluators was 58.8%, binomial tests against a chance baseline (50%) demonstrated that only Evaluator 1 $( p \ = \ 0 . 0 0 3 )$ and Evaluator 3 $( p = 0 . 0 1 6 )$ performed significantly better than random guessing. Furthermore, the low inter-rater reliability (Fleiss’κ = 0.29) confirms a distinct lack of perceptual consensus among the human annotators. However, this should be interpreted as a preliminary perceptual check of whether the dataset's designated L1/L2 distinction was audible to listeners under our experimental task, rather than as an expert validation of speakers’ linguistic backgrounds or the accuracy of the dataset labels.

Collectively, these findings suggest that Fair-Speech corpus may contain substantial demographic and sociolinguistic variation that influences the observed relationship between LD and ASR performance. Since all participants, including L2 speakers, are recruited in the United States, their speech may exhibit greater convergence toward general American English. This potentially reduce the retention of the distinct phonological markers associated with their primary language, which may contributing to the perceptual ambiguity documented in our human validation study. At the same time, the substantial diversity within the corpus, including variation among English L1 speakers, suggests that factors beyond L1/L2 status may contribute to the observed WER patterns and may also explain the negative LD coefficients. Therefore, we view Fair-Speech as a counterexample demonstrating that English L1/L2 status and $L D$ are not necessarily the primary axes of ASR disparity in every corpus.

LD-ASR Association Across Datasets: To account for systematic differences in ASR performance across datasets arising from differences in recording conditions and task design, we further fitted Tweedie mixed-effects models with dataset-specific random intercepts. The results (Fig. 2) are consistent with our previous analysis, with LD showing a positive and statistically significant association with WER across the evaluated models. For example, in Whisper-Small, LD has positive association with WER $( \beta ~ = ~ 0 . 4 1 0 , ~ S E ~ = ~ 0 . 0 1 6 , ~ z ~ = ~ 2 5 . 2 0$ $p < 0 . 0 0 1 )$ . Given the log link, this corresponds to an approximately 52% increase in expected WER for a one-unit increase in LD, holding the dataset-specific random effect constant. The estimated dataset-level random-effect variance is $\hat { \sigma } _ { u } ^ { 2 } = 0 . 9 9 9 7 \ : ( S E = 0 . 7 0 2 )$ . Overall, this analysis provides additional support for the association between LD and ASR error, suggesting that the observed relationship persists after accounting dataset-level variation7.

Model layer depth influence on accent features: To investigate the origins of the disparity in LD-ASR performance within the latent space, we visualize the embeddings using t-SNE projections to identify potential separation based on speakers' L1. Taking Whisper-small as an example (Figure 3), linguistic clustering becomes increasingly pronounced in the deeper layers and remains evident in the final layers, indicating that L1-related structure is preserved in the representations. Furthermore, this clustering within the embedding space emerges as early as the first few layers, aligning with the higher error rates observed in early-stage representations. This phenomenon is not unique to Whisper architecture. A similar layer-wise spatial separation can also be observed in both Parakeet and Canary8.

To examine our hypothesis that LD is encoded across the model layers and is associated with ASR errors, we calculate the Spearmans correlation across both LD and $\mathrm { E D } _ { \mathrm { l } } ,$ and LD and Ymetric. Figure 4 (a) shows that Whisper, no matter the model size, has correlation between $L D$ and $\mathrm { E D } _ { \mathrm { l } }$ becomes pronounced from the early-to-middle layers and remains evident in the later layers. This pattern is also reflected in the relationship between $\mathrm { E D } _ { \mathrm { l } }$ and $\mathrm { Y } ^ { \mathrm { W E R } }$ showing in Figure 4 $\left( \mathbf { b } \right) ^ { 9 } .$ As for Parakeet/Canary, they both hold a high correlation across LD and ED1. However, CanaryV2 shows lower correlation across performance metrics. It aligns with Figure 1, where Canary generally has the lowest coefficient comparing to other models.

![](images/85c73cd7ca13664fcbb4c836bc0d01007a92c4fd5c2b2a86f9cc66b424e0c7e9.jpg)  
(a) Spearman correlation across Linguistic Distance (LD) and Embedding Distance (EDl).

![](images/b84994309d651b75513767ddd977b022aa70bdef8a78caaa959988593e6efcf2.jpg)  
(b) Spearman correlation across Word Error Rates $( Y ^ { \mathsf { W E R } } )$ and Embedding Distance (EDl).

Figure 4: Evolution of Spearman correlation metrics across network layers. Embedding space divergence relative to language distance $\rho ( L D , E D _ { l } )$ and Word Error Rates $\rho ( { \bar { Y } } ^ { \operatorname { W E R } } , E D _ { l } )$ intensify with layer depth. Due to non-normality, Spearman's rank correlation was used; All statistical assumptions were satisfied.

## 5 Conclusion

This work explores how Linguistic Distance (LD) impacts English Automatic Speech Recognition (ASR) accuracy across different architectures. We reveal that as a speaker's first language (L1) LD from English increase, ASR performance systematically degrades. This relationship remains after accounting for dataset-level variation in mixed-effects models. Fair-Speech provides a counterexample, highlighting that L1/L2 status and LD are not always the primary axes of the disparity. By mapping internal model dynamics, we show that L1/L2-related separation in acoustic embeddings persists across deeper neural network layers, suggesting that models retain rather than normalize accent-related variation.Crucially, CanaryV2 emerges as a notable exception on within this investigation, indicating greater robustness to accent diversity.

## 6 Limitations

While our study incorporates diverse datasets, the L2 samples from available dataset are still primarily drawn from specific geographic populations, particularly European and East Asian speakers. This leaves other regions underrepresented in our current analysis, potentially limiting the generalizability of our findings to all marginalized speaker groups.

Furthermore, our reliance on LD as a metric may not fully capture the nuance of diverse linguistic backgrounds. Dataset-provided or selfreported L1 labels can be ambiguous and may not provide a complete representation of a speaker's linguistic history. Specifically, the speech patterns of multilingual individuals often reflect blended influences that cannot be easily categorized by a single first language L1, but has been filtered out during the analysis. Treating L1 and LD as primary explanations for ASR disparities may therefore oversimplify complex linguistic histories and overlook substantial within-corpus variability, including L2 proficiency, age of acquisition and exposure to English. These factors may contribute to variation in the observed results and influence the relationship between LD and ASR performance, potentially affecting our conclusions.

Another limitation of this work is the lack of a universally standardized measure of LD. Although the approach proposed by Vincent and Johannes (2020) provides a single composite relatedness score that is convenient for our regression analysis, other commonly used resources, such as URIEL/lang2vec and WALS, capture a broader range of linguistic features but require additional choices regarding feature selection and aggregation. Consequently, the observed relationship between LD and ASR performance may depend, to some extent, on the particular distance measure adopted.

In our study, we focus on compare commonly used open-source ASR model. Although the selected models are multilingual or multilingualcapable, they are not designed the same way as massively multilingual system such as Omnilingual, or commericial system. Those models trained with broader multilingual coverage may encode L1-related variation differently and may reduce or reshape the disparities observed in our selected model.

Finally, since our empirical analysis is fully focuses on English speech, further investigation is still needed before generalize the effect of language distance to other target languages ASR system or broader setting.

## 7 Ethics Considerations

Demonstrating that ASR models separate speakers by L1 in their latent space reveals a dual-use risk. These internal representations could be extracted without consent to profile or surveil users based on their background. Furthermore, we show that optimizing only for output parity (WER, WIL, SemDist) hides this internal bias, allowing structurally biased models to be deployed while falsely appearing fair.

## References

Giuseppe Attanasio, Beatrice Savoldi, Dennis Fucci, and Dirk Hovy. 2024. Twists, humps, and pebbles: Multilingual speech recognition models exhibit gender performance gaps. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 21318–21340.

Martijn Bartelds, Wietse de Vries, Caitlin Richter, Mark Liberman, and Martijn Wieling. 2021. Measuring foreign accent strength using an acoustic distance measure. In 12th International Seminar on Speech Production, pages 17–20. Haskins Press.

Tolga Bolukbasi, Kai-Wei Chang, James Y Zou, Venkatesh Saligrama, and Adam T Kalai. 2016. Man is to computer programmer as woman is to homemaker? debiasing word embeddings. Advances in neural information processing systems, 29.

Ann R Bradlow, L Ackerman, L Burchfield, L Hesterberg, J Luque, and K Mok. 2010. Allsstar: Archive of 11 and 12 scripted and spontaneous transcripts and recordings. In Proceedings of the International Congress on Phonetic Sciences, volume 356, page 359.

May Pik Yu Chan, June Choe, Aini Li, Yiran Chen, Xin Gao, and Nicole R Holliday. 2022. Training and typological bias in asr performance for world englishes. In INTERSPEECH, pages 1273–1277.

Ronald Cumbal, Birger Moell, José Lopes, and Olof Engwall. 2024. You don't understand me!:

Comparing asr results for 11 and 12 speakers of swedish. arXiv preprint arXiv:2405.13379.

Alex DiChristofano, Henry Shuster, Shefali Chandra, and Neal Patwari. 2022. Global performance disparities between english-language accents in automatic speech recognition. arXiv preprint arXiv:2208.01157.

Kevin Durrheim, Maria Schuld, Martin Mafunda, and Sindisiwe Mazibuko. 2023. Using word embeddings to investigate cultural biases. British Journal of Social Psychology, 62(1):617–629.

Juuso Eronen, Michal Ptaszynski, Fumito Masui Masaki Arata, Gniewosz Leliwa, and Michal Wroczynski. 2022. Transfer language selection for zero-shot cross-lingual abusive language detection. Information Processing & Management, 59(4):102981.

Calbert Graham and Nathan Roll. 2024. Evaluating openai's whisper asr: Performance analysis across diverse accents and speaker traits. JASA Express Letters, 4(2).

Elsayed Issa, Mahmoud Ali, and Kevin Hirschi. 2026. Measuring linguistic bias in asr: Whisper largev3 on non-native speech versus human perception. Procedia Computer Science, 275:692–699.

Maliha Jahan, Priyam Mazumdar, Thomas Thebaud, Mark Hasegawa-Johnson, Jesús Villalba, Najim Dehak, and Laureano Moro-Velazquez. 2025. Unveiling performance bias in asr systems: A study on gender, age, accent, and more. In ICASSP 2025- 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5. IEEE.

Suyoun Kim, Duc Le, Weiyi Zheng, Tarun Singh, Abhinav Arora, Xiaoyu Zhai, Christian Fuegen, Ozlem Kalinli, and Michael L Seltzer. 2021. Evaluating user perception of speech recognition system quality with semantic distance metric. arXiv preprint arXiv:2110.05376.

Victor Kuperman. 2025. How does language distance affect reading fluency and comprehension in english as second language? Studies in Second Language Acquisition, 47(3):757–773.

Wei Liu, Yukun Xiong, Bin Hu, and Daehan Kwak. 2025. Evaluating automatic speech recognition models: How well do they handle accents? In International Conference on the AI Revolution, pages 447–458. Springer.

Renjun Ma and Bent Jørgensen. 2007. Nested generalized linear mixed models: an orthodox best linear unbiased predictor approach. Journal of the Royal Statistical Society Series B: Statistical Methodology, 69(4):625–641.

Andrew Cameron Morris, Viktoria Maier, and Phil D Green. 2004. From wer and ril to mer and wil: improved evaluation measures for connected speech recognition. In Interspeech, pages 2765–2768.

Jonghwan Na, Yeseul Park, and Bowon Lee. 2024. A comparative study on the biases of age, gender, dialects, and 12 speakers of automatic speech recognition for korean language. In 2024 Asia Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC), pages 1-6.

Nnaemeka Ohamadike, Kevin Durrheim, and Mpho Primus. 2025. Whose voice matters? word embeddings reveal identity bias in news quotes. EPJ Data Science, 14(1):30.

Tobi Olatunji, Tejumade Afonja, Aditya Yadavalli, Chris Chinenye Emezue, Sahib Singh, Bonaventure FP Dossou, Joanne Osuchukwu, Salomey Osei, Atnafu Lambebo Tonja, Naome Etori, and 1 others. 2023. Afrispeech-200: Pan-african accented speech dataset for clinical and general domain asr. Transactions of the Association for Computational Linguistics, 11:1669–1685.

Archiki Prasad and Preethi Jyothi. 2020. How accents confound: Probing for accent information in endto-end speech recognition systems. In Proceedings of the 58th annual meeting of the association for computational linguistics, pages 3739–3753.

Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. 2022. Robust speech recognition via large-scale weak supervision. arXiv preprint.

Ramon Sanabria, Nikolay Bogoychev, Nina Markl, Andrea Carmantini, Ondrej Klejch, and Peter Bell. 2023. The Edinburgh International Accents of English Corpus: Towards the Democratization of English ASR. In ICASSP 2023.

Monica Sekoyan, Nithin Rao Koluguri, Nune Tadevosyan, Piotr Zelasko, Travis Bartley, Nikolay Karpov, Jagadeesh Balam, and Boris Ginsburg. 2025. Canary-1b-v2 & parakeet-tdt-0.6 b-v3: Efficient and high-performance models for multilingual asr and ast. arXiv preprint arXiv:2509.14128.

Tuukka Törö, Antti Suni, and Juraj Šimko. 2025. Neighbors and relatives: How do speech embeddings reflect linguistic connections across the world? PLoS One, 20(8):e0330755.

Chara Tsoukala, Stavros Bompolas, Antigoni Margariti, Konstantina Panagiotou, Maria Elisavet Plaiti, Nefeli Tzanakaki, Petros Karatsareas, Angela Ralli, Antonios Anastasopoulos, and Stella Markantonatou. 2026. Extending asr evaluation resources for modern greek dialects. In Proceedings of the 13th Workshop on NLP for Similar Languages, Varieties and Dialects, pages 210–222.

Tuukka Törö, Antti Suni, Leena Dihingia, Juraj Šimko, and Priyankoo Sarmah. 2025. Exploring dialects with speech embeddings: Insights from two speech databases in assamese and finnish. In 2025 28th Conference of the Oriental COCOSDA International Committee for the Co-ordination and Standardisation of Speech Databases and Assessment Techniques (O-COCOSDA), pages 1–6.

Irina-Elena Veliche, Zhuangqun Huang, Vineeth Ayyat Kochaniyan, Fuchun Peng, Ozlem Kalinli, and Michael L Seltzer. 2024. Towards measuring fairness in speech recognition: Fair-speech dataset. arXiv preprint arXiv:2408.12734.

Beaufils Vincent and Tomin Johannes. 2020. Stochastic approach to worldwide language classification: the signals and the noise towards long-range exploration.

Jan Philip Wahle, Terry Ruas, Saif M Mohammad Norman Meuschke, and Bela Gipp. 2023. Ai usage cards: Responsibly reporting ai-generated content. In 2023 ACM/IEEE Joint Conference on Digital Libraries (JCDL), pages 282–284. IEEE.

Steven Weinberger. 2015. Speech accent archive. george mason university.

Guanlong Zhao, Sinem Sonsaat, Alif Silpachai, Ivana Lucic, Evgeny Chukharev-Hudilainen, John Levis, and Ricardo Gutierrez-Osuna. 2018. L2-arctic: A non-native english speech corpus. In Proc. Interspeech, page 2783–2787.

Maryam Zolnoori, Sasha Vergez, Zidu Xu, Elyas Esmaeili, Ali Zolnour, Krystal Anne Briggs, Jihye Kim Scroggins, Seyed Farid Hosseini Ebrahimabad, James M Noble, Maxim Topaz, and 1 others. 2024. Decoding disparities: evaluating automatic speech recognition system performance in transcribing black and white patient verbal communication with nurses in home healthcare. JAMIA open, 7(4):ooae130.

## A Weak Explanatory Power of LD

LD has weaker explanatory power in datasets such as EDACC and Afrispeech-200, where additional sources of variability may influence ASR performance beyond linguistic distance. EDACC contains spontaneous dyadic conversations that includes speaker overlap and background noise, while Afrispeech-200 includes clinical domain speech, for which ASR systems have been shown to perform worse than on more generaldomain speech (Olatunji et al., 2023). These dataset-specific characteristics reinforce the finding that systematic variation observed in English ASR performance cannot be attributed to LD alone. Instead, the explanatory power of LD should be interpreted alongside dataset design, sociolinguistic and demographic variation, multilingual speaker histories, and other corpuslevel factors.

## B Human Perception Study

To qualitatively verify the Fair-Speech dataset, an informal listening test was conducted with four volunteering participants based in Denmark with an English proficiency of at least C1. Each participant was presented with 100 randomized audio samples, consisting of 50 L1 and 50 L2 speakers. The participants were instructed to label each sample based solely on their perception of the speaker's language status, where L1 referred to speakers whose first language was English and L2 referred to speakers whose first language was not English. Prior to the task, participants were briefed on the study's objective: validating ASR model performance across diverse speaker backgrounds. They were also informed that their responses would be fully anonymized and used strictly for internal result verification, with no individual-level data to be publicly released.

## C Tweedie Mixed Effect Models for Alternative Performance Metrics

![](images/3182f77db5da725fc64a01545b17c536ba1e4a973033712b02e1d171bb00c500.jpg)

![](images/59f48e2f6498023f3e19c3ef4ed8019f6d760404365d7b9ce2d1ac22fe18f891.jpg)

![](images/da15f142e629824b07de9915911f5360351f1939fc1f73be8131546e56fc336f.jpg)

![](images/58c6ef4d3923cb59c8491f37ba848c83b08f88cee61a219e8f797cb5d78e1698.jpg)  
Linguistic Distance (LD)

(a) Association between L1's Linguistic Distance (LD) from English and ASR performance in Word Information Lost (YWIL).  
![](images/8625711b1faf95b0460c089e7f46e0adc41660470ba915399f829fa43801a469.jpg)

![](images/b8042dd3418d862cd2ea60d5141035c2537b97c4212e28e01deb6f08fce709bf.jpg)

![](images/adf0d20371cdc5db8bb77fe8afe22556dab8cd7be41b917def1622efa3e1f73d.jpg)

![](images/5618af7928f48d8d9ca36abfa9a3c82b61f1943223aef82be243097bd9b7327e.jpg)  
Linguistic Distance (LD)  
(b) Association between L1's Linguistic Distance (LD) from English and ASR performance in Semantic Distance (SemDist).  
Figure 5: The association between LD and ASR performance is consistent across alternative evaluation metrics, WIL and SemDist.

## D Comparative Embedding Analysis Across ASR Architectures

![](images/871abaee5bbec7e282969ea0499439cf7e3f3e6c2b46d6fe763fe0b751b73a38.jpg)

(a) Whisper Large (1.5B)  
![](images/4277c5e5f981c91b649496b1d75ebb83b293ddc100eb6eadaad4c1c09257371f.jpg)  
(b) Parakeet-TDT-v3 (0.6B)

![](images/21235b858a122988c788e42b7eba23249e63c34203708bd67d355c52eb1170e3.jpg)  
(a) Canary-v2 (1B)  
Figure 7: Layer-wise evolution of latent embeddings. t-SNE (perplexity = 30, 1,000 iterations, random state = 20) projections demonstrate the progressive segregation of L2 speech features across different ASR architectures. The observed patterns are consistent across alternative t-SNE settings (perplexity = 10, random state = 49; perplexity = 30, random state = 42).

## E Correlation Analysis for Alternative Performance Metrics

![](images/7ab57c63ee902c6d4688be27cdd1e565028bbbbfc1cd4ba4fb86a985a30e70b6.jpg)  
(a) Correlation between Word Information Lost $( Y ^ { \mathrm { W I L } } )$ and Embedding Distance (EDl).

![](images/30db3db071e6e56476e82b86d402214f149b016974a6b9a3fe7e70d9a567d81d.jpg)  
(b) Correlation between Semantic Distance $( Y ^ { \mathrm { S e m D i s t } } )$ and Embedding Distance (EDl).  
Figure 8: Evolution of Spearman correlation metrics across network layers. The plots illustrate consistent alignment between internal representation shifts and alternative performance metrics (WIL and SemDist)

F AI Usage card based on (Wahle et al., 2023)
<table><tr><td colspan="3">Al Usage Card</td></tr><tr><td colspan="3"></td></tr><tr><td colspan="2">CORRESPONDENCE(S) CONTACT(S) Ting-Hui Cheng tiche@dtu.dk</td><td>AFFILIATION(S) Technical University of Denmark</td></tr><tr><td></td><td>PROJECT NAME Linguistic Distance Segregates Latent Representations in Automatic Speech Recognition Systems</td><td>KEY APPLICATION(S) Evaluation of ASR robustness</td></tr><tr><td>MODEL(S) ChatGPT Gemini Claude</td><td>VERSION(S) 5.1, 5.6 Luna 2.5 Pro, 3 Pro Sonnet 5</td><td></td></tr><tr><td>WRITING ChatGPT, Gemini</td><td>GENERATING NEW TEXT BASED ON INSTRUCTIONS Not used</td><td>ASSISTING IN IMPROVING OWN CONTENT When existing text is paraphrased or improved.</td></tr><tr><td rowspan="2">CODING Claude,ChatGPT, Gemini</td><td>PARAPHRASING RELATED WORK Not used</td><td>PUTTING OTHER WORKS IN PERSPECTIVE Not used</td></tr><tr><td>GENERATING NEW CODE BASED ON DESCRIPTIONS OR EXISTING CODE When new code is generated based on instructions or prompts.</td><td>REFACTORING AND OPTIMIZING EXISTING CODE When existing or generated code is refactored or its performance optimized.</td></tr><tr><td rowspan="3">ETHICS</td><td>COMPARING ASPECTS OF EXISTING CODE Not used</td><td></td></tr><tr><td>WHAT ARE THE IMPLICATIONS OF USING AI FOR THIS PROJECT? To facilitate academic writing and improve efficiency.</td><td>WHAT STEPS ARE WE TAKING TO MITIGATE ERRORS OF AI FOR THIS PROJECT? All Al-assisted content is critically reviewed and verified by the authors.</td></tr><tr><td>WHAT STEPS ARE WE TAKING TO MINIMIZE THE CHANCE OF HARM OR INAPPROPRIATE USE OF AI FOR THIS PROJECT? Only to support writing and coding assistance and not used to generate experimental results or scientific conclusions.</td><td>THE CORRESPONDING AUTHORS VERIFY AND AGREE WITH THE MODIFICATIONS OR GENERATIONS OF THEIR USED AI-GENERATED CONTENT Yes</td></tr></table>