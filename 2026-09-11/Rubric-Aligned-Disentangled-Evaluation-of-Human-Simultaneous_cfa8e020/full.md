# Rubric-Aligned Disentangled Evaluation of Human Simultaneous Interpreting

Ziyu Zhang<sup>1,∗</sup>, Satoshi Nakamura<sup>2,3,∗,∗∗</sup>

<sup>1</sup> School of Data Science, The Chinese University of Hong Kong, Shenzhen, China <sup>2</sup> School of Artificial Intelligence, The Chinese University of Hong Kong, Shenzhen, China Shenzhen Loop Area Institute, China

ziyuzhang@cuhk.edu.cn, snakamura@cuhk.edu.cn

## Abstract

Human simultaneous interpreting (SI) is commonly assessed with analytic rubrics separating meaning transfer, delivery quality, and temporal synchrony, yet no automatic metric is designed for rubric-aligned segment-level SI evaluation. We construct a professionally annotated corpus of 1,101 SI segments with scores for meaning transfer (LQ), delivery quality (EXP), and perceived latency (LAT). We show that structured LLM prompting and scalar supervision collapse rubric dimensions, yielding near-zero correlation with human ratings and strong crossdimension coupling. To isolate supervision structure under identical backbone capacity, we introduce dual regression heads on a LoRA-adapted COMET-KIWI encoder. On a held-out talklevel test set, the model achieves Pearson correlations of 0.388 (LQ) and 0.301 (EXP), improving over frozen COMET-KIWI. Given low absolute rater agreement, we interpret results relative to human consistency and target stable ranking signals for formative assessment.

Index Terms: simultaneous interpreting evaluation, rubricaligned evaluation, disentangled neural metrics, COMET-KIWI adaptation, LLM-based evaluation, formative assessment

## 1. Introduction

Human simultaneous interpreting (HSI) involves professional interpreters producing target speech in real time while listening to the source. In professional and educational contexts, HSI performance is commonly assessed using analytic rubrics that separately evaluate meaning transfer, delivery quality, and temporal synchrony. In this work, we operationalize these dimensions at the segment level as LQ (semantic fidelity), EXP (delivery quality), and LAT (perceived synchrony).

Professional certification frameworks such as NAATI [1] and CATTI [2] employ multi-criteria analytic scoring that distinguishes semantic fidelity from delivery quality, while recent proposals such as SVIP [3] further formalize structured multidimensional SI evaluation. However, these frameworks operate at the interpreter level and do not provide fine-grained segment-level diagnostics, and SI assessment remains laborintensive and subject to inter-rater variability due to subjective judgment [4], creating a gap between professional standards and scalable automatic evaluation.

Existing MT evaluation metrics—including BLEU, ME-TEOR, BERTScore, BLEURT, COMET, and COMET-KIWI—were developed for written translation and typically collapse quality into a single scalar score [5, 6, 7, 8, 9, 10]. However, SI differs fundamentally from written translation due to incremental processing, temporal constraints, and frequent reformulation or omission [11, 12]. Prior work further reports discrepancies between MT-oriented metrics and human SI judgments under interpreting-specific phenomena such as summarization and latency effects [13, 14].

![](images/5ccf296d015bfede4505db0d55d1c086aa8618ff45cb71a870e45895ffd9ff9e.jpg)  
Figure 1: Overview of the rubric-aligned SI evaluation framework: segment-level rubric annotation (LQ/EXP/LAT) with talk-level split, dual-head COMET-KIWIfor LQ/EXP, and evaluation via correlation, human reliability, and disentanglement.

Recent work explores large language models (LLMs) as automatic evaluators for text generation [15, 16, 17]. We evaluate structured zero-shot and few-shot prompting strategies that enforce explicit scoring order (“LQ first, then EXP”). Nevertheless, LLM predictions exhibit strong cross-dimensional coupling (corr ≈ 0.90 on dev), suggesting that instruction-level control alone fails to preserve rubric structure.

Motivated by this observation, we hypothesize that under correlated analytic dimensions and rater-specific scale variability, scalar supervision collapses ranking structure even with identical encoder capacity. We therefore test whether parameter-level structural separation is necessary to preserve rubric-aligned signals.

To address this, we introduce a rubric-aligned neural evaluation framework that explicitly separates meaning transfer and delivery quality. We adopt COMET-KIWI [10] as a referencefree quality estimation backbone and adapt it to rubric supervision via lightweight LoRA [18] and disentangled regression heads (Figure 1). Our contributions are threefold:

1. A professionally annotated segment-level SI corpus (1,101 segments) with analytic rubric supervision.

2. Empirical evidence that prompt-based and scalar supervision collapse rubric dimensions.

3. A dual-head neural metric that improves correlation with professional ratings and yields stable ranking signals under noisy supervision.

Although we use a dual-head architecture, the central challenge is supervision structure under correlated-but-distinct rating dimensions, not conventional multi-task learning. Our goal is not to replace certification assessment, but to support formative segment-level feedback interpreted relative to human consistency.

## 2. Related Work

## 2.1. Automatic Metrics for Machine Translation

Most automatic evaluation metrics were developed for written machine translation and assume scalar quality representation. Classical metrics such as BLEU, TER, METEOR, and chrF [5, 19, 6, 20] rely on surface similarity, while neural metrics including BERTScore, BLEURT, YiSi, COMET and COMET-KIWI [7, 8, 21, 9, 10], leverage pretrained multilingual encoders to approximate human adequacy judgments.

## 2.2. Evaluation of Human Simultaneous Interpreting

HSI is produced incrementally under real-time constraints and is commonly analyzed through latency–quality trade-offs [22], often operationalized using measures such as Ear–Voice Span (EVS), a classical latency measure in simultaneous interpreting [23]. Analyses of simultaneous interpretation corpora, including large-scale sentence-aligned datasets [24], as well as speech translation resources such as BSTC, report discrepancies between MT-oriented metric scores and human SI judgments under interpreting-specific phenomena such as summarization and reformulation [14].

Professional assessment frameworks such as NAATI and CATTI employ analytic scoring that separates meaning transfer and delivery quality, while academic proposals such as SVIP [3] further formalize multi-dimensional SI evaluation principles under structured rubric design. However, prior work lacks neural evaluation models explicitly trained under rubric supervision to disentangle these dimensions in human SI.

## 2.3. LLM-Based Evaluation of Generated Text

Recent work explores LLMs as evaluators for text generation and SI, including GPT-3.5-based SI evaluation, promptbased scoring, and multi-agent debate frameworks [15, 25, 17]. Unlike reference-based machine-SI evaluation, our setting is reference-free human SI evaluation under professional analytic rubrics. Such LLM-based evaluation relies mainly on instruction-level control, raising concerns about rubric-level controllability.

## 3. Dataset and Rubric

The corpus combines publicly available BSTC-derived SI segments [13] with newly collected licensed TED-style conference recordings obtained under research consent. Transcripts were obtained from official transcripts where available and from ASR for collected recordings, followed by normalization. After combining all sources, we performed the train/dev/test split at the talk level to prevent information leakage across related discourse [26].

## 3.1. Dataset and Split

Each segment is double-rated by professional raters under the analytic rubric (Table1), and individual ratings are treated as separate supervision instances. The resulting split comprises 839 segments (48 talks) for training, 87 segments (5 talks) for development, and 169 segments (8 talks) for testing, covering both English→Chinese and Chinese→English directions. Pre-processing included semantic segmentation of source–interpretation pairs, transcript normalization, and removal of incomplete or unavailable segments prior to splitting.

Table 1: Segment-level rubric summary (0–3).
<table><tr><td>Dim. Criteria (summary)</td><td></td></tr><tr><td>LQ</td><td>3: &gt;80% meaning preserved; no major omis- sion/distortion. 2: ~60–70% meaning; some omissions/inaccuracies but main message in- tact. 1: ~40–50% meaning; major omis- sions/distortions. 0: &lt;30% meaning or mislead-</td></tr><tr><td>EXP</td><td>ing/reversed meaning. 3: fluent, idiomatic; minor errors only. 2: gen- erally fluent with noticeable awkwardness/errors. 1: frequent disfluency/grammar issues affecting delivery. 0: breakdown-level disfluency impedes</td></tr><tr><td>LAT</td><td>comprehension. 3: low perceived lag; well synchronized. 2: oc- casional lag; manageable. 1: frequent noticeable lag; disjointed flow. 0: severe persistent lag; dif- ficult to follow.</td></tr></table>

Table 2: Dataset statistics and talk-level split.
<table><tr><td></td><td>Train</td><td>Dev</td><td>Test</td></tr><tr><td>#Segments</td><td>839</td><td>87</td><td>169</td></tr><tr><td>#Talks</td><td>48</td><td>5</td><td>8</td></tr><tr><td>Split level Directions</td><td></td><td>Talk (no cross-talk overlap) En→Zh and Zh→En</td><td></td></tr></table>

## 3.2. From Professional Framework to Segment-Level Operationalization

Professional SI certification frameworks (e.g., NAATI, CATTI) and analytic proposals such as SVIP evaluate interpreting performance along separable dimensions including meaning transfer and delivery quality. Building on these rubric-based frameworks, we introduce a modeling-oriented segment-level operationalization that preserves analytic separation while enabling neural supervision. Scoring is performed on semantically segmented units, a deterministic gating rule sets EXP to zero when LQ indicates severe meaning failure, and latency is annotated impressionistically to capture perceived synchrony rather than precise acoustic timing.

## 3.3. Latency Dimension Analysis

The annotation protocol includes a perceived latency (LAT) dimension scored on a 0–3 scale alongside objective segmentlevel onset delay. Perceived latency shows substantial variation (mean = 2.13, std = 0.79) but weak correlation with objective delay (Pearson = -0.048; Spearman = -0.034), suggesting discourse-level synchrony judgments consistent with cognitiveload–based accounts of simultaneous interpreting [27, 28]. LAT additionally correlates with LQ (r = 0.38) and EXP (r = 0.43), indicating a latency–quality trade-off.

## 4. Method

We model rubric-aligned SI evaluation as multi-output regression, $f ( x , y )  ( \hat { s } _ { L Q } , \hat { s } _ { E X P } )$ , where x denotes the source and y the interpreted output. The COMET-KIWI encoder was initialized from pretrained weights, while regression heads were randomly initialized using scaled Xavier initialization. We focus on text-only prediction of LQ and EXP. Since EXP also depends on acoustic cues such as prosody, pauses, and fluency, our model is a text-based lower bound that captures transcriptvisible delivery signals. Modeling LAT likely requires multimodal timing cues and is left for future work.

Our approach relies on three assumptions: (i) rubric dimensions such as meaning transfer and delivery quality are analytically separable despite empirical correlation; (ii) individual rater supervision better captures scale variability than aggregated mean labels; and (iii) formative SI assessment primarily requires stable ranking signals rather than absolute agreement with any single rater.

## 4.1. Backbone and Dual-Head Architecture

We build on COMET-KIWI [10] (XLM-R Large) using pair encoding [CLS] source [SEP] hypothesis [SEP]. The original scalar regression head is replaced with two independent linear heads: $\hat { s } _ { L Q } ~ = ~ W _ { L Q } h \stackrel { - } { + } b _ { L Q }$ and sˆ<sub>EXP</sub> = $W _ { E X P } h + b _ { E X P } ,$ , where h denotes the [CLS] representation. The backbone contains approximately 550M parameters, while LoRA and regression heads introduce fewer than 1M additional trainable parameters.

## 4.2. Residual Prediction and Objective

To mitigate prediction mean-collapse under noisy supervision, we adopt residual prediction $\hat { s } _ { d } ~ = ~ \mu _ { d } + \Delta _ { d }$ for d ∈ $\{ L Q , E X P \}$ and optimize

$$
\mathcal { L } = \mathrm { M S E } _ { L Q } + w \mathrm { M S E } _ { E X P } + \lambda \mathcal { L } _ { v a r } ,
$$

with w=1.7 and λ=0.05. Predictions are clamped to [0, 3] for evaluation (optionally quantized to 0.5 steps).

## 4.3. LoRA Adaptation

Encoder adaptation is performed using LoRA [18] applied to attention Q, V projections $( r = 8 , \alpha = 1 6 .$ , dropout 0.1). Training follows a two-stage schedule: epoch 1 optimizes regression heads only, while subsequent epochs jointly update heads and LoRA parameters. Hyperparameters were selected via manual tuning on the development set. Best checkpoint was chosen by maximizing the sum of LQ and EXP Pearson correlations.

## 5. Experiments

## 5.1. Experimental Setup

Experiments are conducted on the SI corpus described in Section 3 using talk-level splits to ensure generalization to unseen talks. Training was performed on Google Colab using an NVIDIA L4 GPU for approximately 10 epochs, with each run completing within several hours depending on batch size and sequence length.

Evaluation metrics include Pearson correlation (r) as the primary measure of ranking alignment with human ratings, Spearman correlation (ρ) for robustness to scale differences, prediction standard deviation to detect collapse, and crossdimension correlation. Mean squared error (MSE) is additionally reported to capture absolute prediction deviation. Statistical significance is assessed using Fisher’s r-to-z transformation and bootstrap resampling (10,000 samples).

Table 3: Human inter-rater reliability on dual-annotated segments.
<table><tr><td>Dim.</td><td>r</td><td>95% CI</td><td> $\rho$ </td><td>ICC2</td><td>ICC3</td></tr><tr><td>LQ</td><td>.207</td><td>[.107,.303].196</td><td></td><td>.139</td><td>.343</td></tr><tr><td>EXP</td><td></td><td>.274[.173,.369].291</td><td></td><td>.140</td><td>.430</td></tr></table>

Table 4: LLM evaluation behavior on the dev set.
<table><tr><td>Dim.</td><td>Human std LLM std Human r LLM r</td><td></td><td></td><td></td></tr><tr><td>LQ/EXP</td><td>.69/.74</td><td>1.07/1.27</td><td>.56</td><td>.90</td></tr></table>

## 5.2. Baselines

Baselines are designed to isolate supervision structure under comparable backbone capacity: (i) Frozen COMET-KIWI: the pretrained wmt22-cometkiwi-da model applied without fine-tuning, with scalar predictions correlated separately with LQ and EXP; (ii) Frozen Encoder + Linear Heads: dual regression heads trained on frozen COMET-KIWI representations; (iii) Single-Head Fine-Tuning: LoRA-adapted scalar regression predicting the mean of LQ and EXP; (iv) Prompt-Based LLM Evaluation: structured zero-shot and few-shot rubric prompting with enforced scoring order (evaluated on the development set); (v) Mean Baseline: prediction of training-set mean scores.

## 5.3. Human Reliability Analysis

Given the subjective nature of SI evaluation, we quantify interrater reliability on overlapping dual-annotated segments.

## 5.3.1. Inter-Rater Agreement

On the dual-annotated subset (n=367 for LQ; n=345 for EXP), we estimate reliability using correlation- and variance-based measures. Because segments are annotated by different rater pairs, ICC is computed by treating the two ratings per segment as repeated measurements. ICC(2,1) captures absolute agreement, while ICC(3,1) captures consistency after rater scale shifts. Table 3 shows low absolute agreement but moderate consistency-level reliability.

## 5.3.2. Inter-rater pairwise correlation

To complement ICC, we compute pairwise Pearson correlations across rater pairs on overlapping subsets and average results across available combinations. Mean pairwise Pearson correlations are 0.264 (LQ), 0.286 (EXP), and 0.223 (LAT), providing a human–human reference ceiling for model correlation under subjective segment-level scoring.

## 6. Results

## 6.1. Prompt-Based LLM Evaluation (Dev Set)

Structured rubric prompting is evaluated on the development set $( n = 8 7 )$ . Despite explicit instructions enforcing dimension separation, prompt-based evaluation shows near-zero correlation with human ratings: zero-shot prompting yields negligible correlation, while few-shot prompting modestly improves LQ (Pearson = 0.13) but not EXP (Pearson = 0.01).

Table 5: Pearson correlation on development and test sets
<table><tr><td>Model</td><td>LQ (r)</td><td>EXP (r)</td></tr><tr><td colspan="3">Development Set</td></tr><tr><td>Prompt (zero-shot)</td><td>0.054</td><td>0.041</td></tr><tr><td>Prompt (few-shot)</td><td>0.130</td><td>0.010</td></tr><tr><td>Scalar fine-tune</td><td>0.092</td><td>-0.020</td></tr><tr><td colspan="3">Test Set</td></tr><tr><td>Mean baseline</td><td>0.000</td><td>0.000</td></tr><tr><td>Frozen COMET-KIWI</td><td>0.219</td><td>0.175</td></tr><tr><td>Dual-head (proposed)</td><td>0.388</td><td>0.301</td></tr></table>

Table 4 highlights two failure modes: predictions exhibit substantially higher variance than human ratings and strong cross-dimensional coupling (corr = 0.90 vs. human = 0.56), indicating collapse toward a single latent quality signal despite structured prompting.

## 6.2. Scalar Supervision Failure (Dev Set)

We next examine scalar supervision by fine-tuning a singlehead model to predict the mean of LQ and EXP. On the development set $\mathrm { ~  ~ { ~ ( ~ n ~ } ~ } = \mathrm { ~  ~ { ~ 8 7 ~ } ) ~ }$ , correlations remain near zero $( \mathrm { P e a r s o n } ( s , \mathrm { L Q } ) ~ = ~ 0 . 0 9 2$ , Pearson(s, $\mathrm { E X P } \ : \ : = \ : \ : - 0 . 0 2 0 ,$ Pearson(s, combined) = 0.050), indicating that collapsing rubric dimensions into a single objective obscures ranking structure.

## 6.3. Main results

Table 5 reports Pearson correlation with human ratings on development and test sets (n=169 for test). On the development set, prompt-based evaluation and scalar supervision yield nearzero correlations $( \mathrm { L Q } r { \leq } 0 . 1 3 0 , \mathrm { E X P } r { \leq } 0 . 0 4 1 )$ , indicating that both instruction-level prompting and scalar regression objectives fail to preserve rubric dimensionality.

On the held-out test set, the mean baseline shows no ranking ability, while frozen COMET-KIWI demonstrates moderate transfer (LQ r=0.219, EXP r=0.175), suggesting that translation-oriented representations partially capture SI quality signals. The proposed dual-head model substantially improves alignment with human ratings, achieving r=0.388 for LQ and r=0.301 for EXP, with gains over the frozen baseline statistically significant under bootstrap resampling (p<0.05).

Considering moderate consistency-level human reliability (ICC(3,1)=0.343 for LQ and 0.430 for EXP; Table 3), these results indicate that structured supervision enables recovery of stable ranking signals under subjective segment-level annotation, positioning model performance within the human–human correlation range observed under subjective segment-level scoring.

## 6.4. Qualitative Error Analysis

Representative examples (Table 6) illustrate correct separation of meaning transfer and delivery quality. Failure cases occur primarily in procedural segments containing multi-step instructions (F1–F2). In these examples, the interpretation preserves the overall workflow and action type but omits or distorts several individual steps, leading the model to assign high LQ despite incomplete content coverage. This pattern suggests that the model captures global procedural semantics while remaining insensitive to step-level information completeness.

Table 6: Representative qualitative examples.
<table><tr><td>ID</td><td>Interpretation (ex- cerpt)</td><td>Gold (LQ/EXP)</td><td>Pred (LQ/EXP)</td></tr><tr><td>E1</td><td>“...one becomes writer, one be- comes prisoner.&quot;</td><td>3/2</td><td>3/2</td></tr><tr><td>E2</td><td>&quot;Maybe it&#x27;s about economic shift...&quot;</td><td>2/2</td><td>2/2</td></tr><tr><td>F1</td><td>“...use SDK build process app...&quot;</td><td>to 1/1</td><td>3/2</td></tr><tr><td>F2</td><td>“...choose create new project...&quot;</td><td>1/1</td><td>2/2</td></tr></table>

## 6.5. Dimension Disentanglement Analysis

We examine cross-dimensional coupling on the test set, where the proposed model yields corr $( \hat { s } _ { L Q } , \hat { s } _ { E X P } ) ~ = ~ 0 . 5 2 9$ . For comparison, human ratings exhibit $\mathrm { c o r r } ( s _ { L Q } , s _ { E X P } ) = 0 . 5 6$ while prompt-based LLM evaluation shows near-complete coupling (≈ 0.90, Table 4). The proposed model therefore reproduces a coupling structure closely aligned with human judgment, in contrast to prompt-based evaluation that collapses both dimensions into a single latent quality signal. This finding indicates that parameter-level disentanglement preserves humanlike partial dependence between meaning transfer and delivery quality while avoiding artificial over-coupling.

## 7. Discussion

Results indicate that the main bottleneck in automatic SI evaluation lies in supervision structure: prompt-based and scalar objectives collapse rubric dimensions, while multi-head supervision preserves human-like coupling. The weak association between perceived latency and objective delay suggests that temporal judgments reflect discourse-level synchrony, motivating future multimodal modeling. Limitations include the textonly setting, moderate dataset size, rater scale variability, and Chinese–English focus. We target stable ranking signals for formative assessment rather than absolute agreement. Labels, guidelines, splits, and code will be released where permitted; restricted materials remain subject to original licenses and consent constraints.

## 8. Conclusion

We presented a rubric-aligned framework for segment-level SI evaluation under professional analytic scoring. Using 1,101 annotated segments, we showed that prompt-based and scalar supervision collapse rubric dimensions, yielding weak alignment with human ratings. A dual-head COMET-KIWI model with LoRA improved correlation (Pearson = 0.388 LQ, 0.301 EXP) while preserving human-like cross-dimensional coupling, indicating recovery of stable ranking signals under subjective annotation. The approach enables scalable formative feedback and rubric-aligned benchmarking for SI training. Future work will extend to multimodal latency modeling and broader crosslingual settings.

## 9. Acknowledgments

The authors thank Jason Zhang, a PhD student in Interpreting Studies at CUHK-Shenzhen, for valuable discussions on simultaneous interpreting annotation and evaluation design. We also thank Prof. Li Lan (School of Humanities and Social Science, CUHK-Shenzhen) for insightful feedback on rubric development and interpreting pedagogy perspectives.

We are grateful to the interpreters and annotators who contributed their time and expertise to data collection and rating. We additionally acknowledge the providers of publicly available datasets and licensed conference materials used in this study.

This work was supported by Project W2531054 of the National Natural Science Foundation of China, and the Program for Guangdong Introducing Innovative and Entrepreneurial Teams.

## 10. Generative AI Use Disclosure

Generative AI tools were used for limited auxiliary purposes, including language editing during manuscript preparation, automated assistance in dataset organization, and code debugging support (e.g., via Cursor). All research design, experimental procedures, analyses, and conclusions were performed and verified by the authors, who take full responsibility for the content of this paper in accordance with ISCA policy.

## 11. References

[1] National Accreditation Authority for Translators and Interpreters, “Naati certification testing guidelines,” https://www.naati.com. au/, 2024.

[2] China Foreign Languages Publishing Administration, “China accreditation test for translators and interpreters,” http://catti.net.cn/, 2023.

[3] S. Cheng, Y. Bao, Z. Huang, Y. Lu, N. Peng, L. Xu, R. Yu, R. Cao, Y. Du, T. Han, Y. Hu, Z. Li, S. Liu, S. Ma, S. Pan, J. Xiao, N. Xu, M. Yang, R. Ye, Y. Yu, J. Zhang, R. Zhang, W. Zhang, W. Zhu, L. Zou, L. Lu, Y. Wang, and Y. Wu, “Seed liveinterpret 2.0: End-to-end simultaneous speech-to-speech translation with your voice,” 2025. [Online]. Available: https://arxiv.org/abs/2507.17527

[4] C. Fantinuoli, Ed., Interpreting and technology, ser. Translation and Multilingual Natural Language Processing. Berlin: Language Science Press, 2018, no. 11.

[5] K. Papineni, S. Roukos, T. Ward, and W.-J. Zhu, “Bleu: a method for automatic evaluation of machine translation,” in Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics, P. Isabelle, E. Charniak, and D. Lin, Eds. Philadelphia, Pennsylvania, USA: Association for Computational Linguistics, Jul. 2002, pp. 311–318. [Online]. Available: https://aclanthology.org/P02-1040/

[6] S. Banerjee and A. Lavie, “METEOR: An automatic metric for MT evaluation with improved correlation with human judgments,” in Proceedings of the ACL Workshop on Intrinsic and Extrinsic Evaluation Measures for Machine Translation and/or Summarization, J. Goldstein, A. Lavie, C.-Y. Lin, and C. Voss, Eds. Ann Arbor, Michigan: Association for Computational Linguistics, Jun. 2005, pp. 65–72. [Online]. Available: https://aclanthology.org/W05-0909/

[7] T. Zhang, V. Kishore, F. Wu, K. Q. Weinberger, and Y. Artzi, “Bertscore: Evaluating text generation with bert,” 2020. [Online]. Available: https://arxiv.org/abs/1904.09675

[8] T. Sellam, D. Das, and A. Parikh, “BLEURT: Learning robust metrics for text generation,” in Proceedings of the 58th Annual Meeting of the Association for Computational

Linguistics, D. Jurafsky, J. Chai, N. Schluter, and J. Tetreault, Eds. Online: Association for Computational Linguistics, Jul. 2020, pp. 7881–7892. [Online]. Available: https://aclanthology. org/2020.acl-main.704/

[9] R. Rei, C. Stewart, A. C. Farinha, and A. Lavie, “COMET: A neural framework for MT evaluation,” in Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), B. Webber, T. Cohn, Y. He, and Y. Liu, Eds. Online: Association for Computational Linguistics, Nov. 2020, pp. 2685–2702. [Online]. Available: https://aclanthology.org/2020.emnlp-main.213/

[10] R. Rei, M. Treviso, N. M. Guerreiro, C. Zerva, A. C. Farinha, C. Maroti, J. G. C. de Souza, T. Glushkova, D. M. Alves, A. Lavie, L. Coheur, and A. F. T. Martins, “Cometkiwi: Ist-unbabel 2022 submission for the quality estimation shared task,” 2022. [Online]. Available: https://arxiv.org/abs/2209.06243

[11] D. Gile, Basic Concepts and Models for Interpreter and Translator Training: Revised edition, 11 2009.

[12] R. Setton and A. Dawrant, Conference Interpreting: A Complete Course, ser. Benjamins Translation Library. Amsterdam and Philadelphia: John Benjamins Publishing Company, 2016, vol. 120.

[13] R. Zhang, X. Wang, C. Zhang, Z. He, H. Wu, Z. Li, H. Wang, Y. Chen, and Q. Li, “Bstc: A large-scale chineseenglish speech translation dataset,” 2021. [Online]. Available: https://arxiv.org/abs/2104.03575

[14] S. Wein, T. I, C. Cherry, J. Juraska, D. Padfield, and W. Macherey, “Barriers to effective evaluation of simultaneous interpretation,” in Findings of the Association for Computational Linguistics: EACL 2024, Y. Graham and M. Purver, Eds. St. Julian’s, Malta: Association for Computational Linguistics, Mar. 2024, pp. 209–219. [Online]. Available: https://aclanthology.org/2024. findings-eacl.15/

[15] Y. Liu, D. Iter, Y. Xu, S. Wang, R. Xu, and C. Zhu, “G-eval: NLG evaluation using gpt-4 with better human alignment,” in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, H. Bouamor, J. Pino, and K. Bali, Eds. Singapore: Association for Computational Linguistics, Dec. 2023, pp. 2511–2522. [Online]. Available: https://aclanthology.org/2023.emnlp-main.153/

[16] G. Li, H. A. A. K. Hammoud, H. Itani, D. Khizbullin, and B. Ghanem, “Camel: Communicative agents for ”mind” exploration of large language model society,” 2023. [Online]. Available: https://arxiv.org/abs/2303.17760

[17] H. Li, Q. Dong, J. Chen, H. Su, Y. Zhou, Q. Ai, Z. Ye, and Y. Liu, “Llms-as-judges: A comprehensive survey on llm-based evaluation methods,” 2024. [Online]. Available: https://arxiv.org/abs/2412.05579

[18] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “Lora: Low-rank adaptation of large language models,” 2021. [Online]. Available: https: //arxiv.org/abs/2106.09685

[19] M. Snover, B. Dorr, R. Schwartz, L. Micciulla, and J. Makhoul, “A study of translation edit rate with targeted human annotation,” in Proceedings of the 7th Conference of the Association for Machine Translation in the Americas: Technical Papers. Cambridge, Massachusetts, USA: Association for Machine Translation in the Americas, Aug. 8-12 2006, pp. 223–231. [Online]. Available: https://aclanthology.org/2006.amta-papers. 25/

[20] M. Popovic, “chrF: character n-gram F-score for automatic MT´ evaluation,” in Proceedings of the Tenth Workshop on Statistical Machine Translation, O. Bojar, R. Chatterjee, C. Federmann, B. Haddow, C. Hokamp, M. Huck, V. Logacheva, and P. Pecina, Eds. Lisbon, Portugal: Association for Computational Linguistics, Sep. 2015, pp. 392–395. [Online]. Available: https://aclanthology.org/W15-3049/

[21] C.-k. Lo, “YiSi - a unified semantic MT quality evaluation and estimation metric for languages with different levels of available resources,” in Proceedings of the Fourth Conference on Machine Translation (Volume 2: Shared Task Papers, Day 1), O. Bojar, R. Chatterjee, C. Federmann, M. Fishel, Y. Graham, B. Haddow, M. Huck, A. J. Yepes, P. Koehn, A. Martins, C. Monz, M. Negri, A. Nev´ eol, M. Neves, M. Post, M. Turchi,´ and K. Verspoor, Eds. Florence, Italy: Association for Computational Linguistics, Aug. 2019, pp. 507–513. [Online]. Available: https://aclanthology.org/W19-5358/

[22] M. Elbayad, L. Besacier, and J. Verbeek, “Efficient wait-k models for simultaneous machine translation,” in Proc. Interspeech, Shanghai, China, 2020, pp. 2997–3001.

[23] F. Goldman-Eisler, “Segmentation of input in simultaneous translation,” Journal of Psycholinguistic Research, vol. 1, no. 2, pp. 127–140, 1972.

[24] K. Doi, K. Sudoh, and S. Nakamura, “Large-scale English-Japanese simultaneous interpretation corpus: Construction and analyses with sentence-aligned data,” in Proceedings of the 18th International Conference on Spoken Language Translation (IWSLT 2021), M. Federico, A. Waibel, M. R. Costa-jussa, J. Niehues, S. Stuker, and E. Salesky, Eds.\` Bangkok, Thailand (online): Association for Computational Linguistics, Aug. 2021, pp. 226–235. [Online]. Available: https://aclanthology.org/2021.iwslt-1.27/

[25] C. Fantinuoli and X. Wang, “Exploring the correlation between human and machine evaluation of simultaneous speech translation,” in Proceedings of the 25th Annual Conference of the European Association for Machine Translation (Volume 1), C. Scarton, C. Prescott, C. Bayliss, C. Oakley, J. Wright, S. Wrigley, X. Song, E. Gow-Smith, R. Bawden, V. M. Sanchez-´ Cartagena, P. Cadwell, E. Lapshinova-Koltunski, V. Cabarrao,˜ K. Chatzitheodorou, M. Nurminen, D. Kanojia, and H. Moniz, Eds. Sheffield, UK: European Association for Machine Translation (EAMT), Jun. 2024, pp. 327–336. [Online]. Available: https://aclanthology.org/2024.eamt-1.28/

[26] D. R. Roberts, V. Bahn, S. Ciuti, M. S. Boyce, J. Elith, G. Guillera-Arroita, S. Hauenstein, J. J. Lahoz-Monfort, B. Schroder, W. Thuiller, D. I. Warton, B. A. Wintle, F. Hartig,¨ and C. F. Dormann, “Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure,” Ecography, vol. 40, no. 8, pp. 913–929, 2017. [Online]. Available: https://nsojournals.onlinelibrary.wiley.com/doi/abs/10. 1111/ecog.02881

[27] D. Gile, “The effort models of interpreting,” in Basic Concepts and Models for Interpreter and Translator Training, ser. Benjamins Translation Library. Amsterdam and Philadelphia: John Benjamins Publishing Company, 2009, vol. 8, pp. 157–190.

[28] Y. Kano, K. Sudoh, and S. Nakamura, “Average token delay: A latency metric for simultaneous translation,” 2023. [Online]. Available: https://arxiv.org/abs/2211.13173