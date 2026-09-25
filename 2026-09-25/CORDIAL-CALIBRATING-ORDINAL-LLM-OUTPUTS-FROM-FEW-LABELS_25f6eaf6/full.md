# CORDIAL: CALIBRATING ORDINAL LLM OUTPUTS FROM FEW LABELS

Xiangwei Wang<sup>♡</sup>, Peng Wang<sup>♣</sup>, Saman Halgamuge<sup>♡</sup>

<sup>♡</sup>The University of Melbourne, Australia

## ABSTRACT

A large language model (LLM) can turn a text into a distribution over an ordered scale, but that distribution is a noisy measurement: saturated, compressed or exaggerated, and biased in a consistent direction. We propose CORDIAL, which treats the model’s output as a noisy reading of the true label and corrects it with a channel of five interpretable parameters. The channel is small enough for its posterior to be averaged from a handful of labels, and we prove that the resulting calibration preserves first-order stochastic order. On Amazon reviews and CMU-MOSEI transcripts with four LLMs, CORDIAL has the lowest log loss among nine calibrators in 76 of 80 settings with 5 to 100 labels; with 20 labels and the main 7B reader, it matches the strongest baseline using 28–54 labels. The same posterior lets us learn priors from other tasks and fuse several LLMs. Unrestricted calibrators such as Dirichlet calibration overtake it only as the calibration set grows into the hundreds or thousands.

Index Terms— Large language models, calibration, ordinal classification, Bayesian inference, label efficiency

## 1. INTRODUCTION

A large language model (LLM) can read a review or a spoken sentence and return a distribution over an ordered scale [1]. Such outputs are useful measurements but not calibrated statistics: a model may understand every sentence and still rate one star too generously [2, 3, 4], spread its mass too widely or too narrowly [5], or deserve less trust than its confidence suggests, since post-training degrades calibration [6, 7]. Treating the output as a noisy measurement of the true label turns calibration into channel estimation from very few labels, which an LLM is deployed to avoid collecting.

The usual answer is a scalar, a temperature or a reliability weight toward a population prior; we show that a scalar reliability cannot move mass to a neighboring class and so cannot correct a directional error. An unrestricted $K \times K$ matrix or Dirichlet calibration can, but spends $K ( K - 1 )$ or $K ( K { + } 1 )$ parameters on the few labels available. We propose CORDIAL (Calibrating ordinal outputs), a structured semantic channel whose five parameters, temperature, offset, gain, concentration, and strength, match the geometry of ordinal errors, in a mixture form that keeps the shape of the reading and <sup>♣</sup>Shanghai Jiao Tong University, China

a unimodal location form. With five parameters the posterior can be averaged over, which hedges against the estimation error that dominates with a few dozen labels, and the channels are stacked with logit-space calibrators.

Our contributions are: (C1) the Bayesian structured channel in two forms, a proof that it never reverses the ordinal order of readings, and its stacking with logit-space calibrators; (C2) a parameter-counting crossover, $n ^ { \star } = ( d _ { \mathrm { u } } - d _ { \mathrm { s } } ) / ( 2 \delta )$ for a measurable gap δ, that predicts the scale at which unrestricted calibrators take over; and (C3) three uses of the fiveparameter posterior, a prior learned across tasks, few-label fusion of several frozen LLMs, and a one-parameter per-user shift, with evidence on Amazon Reviews 2023 and CMU-MOSEI transcripts with four readers.

Related work. Temperature, vector, and Dirichlet scaling calibrate class probabilities from held-out labels [8, 9] and are the usual fix for miscalibrated LLM confidences [7], and contextual and batch calibration [2, 10] remove an LLM’s label bias without labels; all ignore the order of the classes. Ordinal-aware losses calibrate classifiers during training [11], and LLM raters on ordinal scales drift toward the middle [5] and depend on the order of options and demonstrations [12]; we correct a frozen reader after the fact. Confusion-matrix models of noisy observers go back to Dawid and Skene [13], with ordinal crowd labels aggregated by minimax conditional entropy [14], and human and model probabilities fused through calibrated confusion matrices [15]. Ordinal regression [16] maps features directly to the scale; supra-Bayesian pooling treats an expert’s statement as data with its own likelihood [17], which is the view taken here for an LLM.

## 2. CORDIAL: STRUCTURED CHANNELS

## 2.1. LLM outputs as noisy ordinal measurements

A LLM reads a text x and returns option logits $\ell ( x )$ over K ordered classes $y \in \{ 0 , \ldots , K - 1 \}$ , taken from the nexttoken scores of the answer options. The quantity of interest is the true ordinal label z. We treat the reading $q ( y \mid x ) =$ softmax $( \ell / \tau ) _ { y }$ as a noisy measurement of z and calibrate it through a row-stochastic channel T,

$$
{ \hat { p } } ( z \mid x ) = \sum _ { y } T ( z \mid y ) q ( y \mid x ) .\tag{1}
$$

The LLM stays frozen; estimating T from a few labeled texts is the calibration problem. The temperature τ belongs to the channel because frozen readers are saturated: on our benchmarks the largest probability of softmax(ℓ) averages 0.92– 0.96.

## 2.2. Why a scalar reliability is not enough

The usual channel trusts the reading with probability β and otherwise falls back on the label prior π, $T _ { \beta } ( z \mid y ) =$ $\beta \mathbf { 1 } \{ z = y \} + ( 1 - \beta ) \pi ( z )$ , so that $\hat { p } _ { \beta } = \beta q + ( 1 - \beta ) \pi$ Remark 1 (scalar rigidity). Since $\hat { p } _ { \beta } - \pi = \beta ( q - \pi )$ , a scalar channel only rescales the reading’s excess over the prior and cannot move it to a neighboring class. If a saturated reading $q = e _ { y }$ has label $y + 1$ , every scalar channel assigns that label $( 1 - \bar { \beta } ) \pi ( y + 1 ) \leq \pi ( y + 1 )$ , no more than ignoring the reader.

## 2.3. An affine channel with interpretable parameters

Ordinal readers err in structured ways: by a consistent offset, by compressing or stretching the scale, and by a consistent spread. With a Gaussian profile on the scale,

$$
G ( z \mid c ) = \frac { \exp [ - \kappa ( z - c ) ^ { 2 } ] } { \sum _ { v } \exp [ - \kappa ( v - c ) ^ { 2 } ] } ,\tag{2}
$$

we use two forms of one affine channel,

$$
\hat { p } _ { \mathrm { m i x } } = \rho \sum _ { y } q ( y \mid x ) G ( z \mid s y + d ) + ( 1 - \rho ) \pi ( z ) ,\tag{3}
$$

$$
\hat { p } _ { \mathrm { l o c } } = \rho G \left( z \mid s m ( x ) + d \right) + \left( 1 - \rho \right) \pi ( z ) ,\tag{4}
$$

where $\begin{array} { r } { m ( x ) \ = \ \sum _ { y } y q ( y \mid x ) } \end{array}$ is the reading’s mean. The parameters are $\phi = \bar { ( \tau , d , s , \kappa , \rho ) } \colon$ temperature, offset d, gain $s > 0 ( s > 1$ stretches a compressed reading, $s < 1$ contracts an exaggerated one), concentration $\kappa > 0 .$ , and strength $\rho$ of the reading against the smoothed label histogram π. The mixture form (3) keeps the shape of the reading, and $d = 0 , s = 1$ $\kappa  \infty$ recovers the scalar channel; the location form (4) is unimodal, which suits labels that are rounded averages. A vector-scaled variant replaces $\ell / \tau$ by $e ^ { \epsilon } \odot \ell / \tau + b$ with 2K more parameters shrunk toward zero.

Proposition 1 (ordinal coherence). Write $q _ { 1 } ~ \preceq ~ q _ { 2 } ~ ( \mathit { f i r s t } .$ order stochastic dominance) $\begin{array} { r } { i f \sum _ { z < t } q _ { 1 } ( z ) \geq \sum _ { z < t } q _ { 2 } ( z ) } \end{array}$ for all t. For any $\kappa > 0 , s > 0 , d , \bar { \rho } \in [ 0 , 1 ] ,$ , and π, both forms satisfy $q _ { 1 } \preceq q _ { 2 } \Rightarrow \hat { p } _ { 1 } \preceq \hat { p } _ { 2 }$ . The same holds for the posterior predictive provided that the tempered readings remain ordered for every temperature in the posterior support. Saturated readings of ordered classes meet this condition at every temperature. Calibration can change a reading’s confidence, bias, and spread but cannot turn a higher reading into a lower prediction; unrestricted calibrators can (Section 3.1). Proof. (a) For $c _ { 1 } < c _ { 2 } , G ( z \mid c _ { 2 } ) / G ( z \mid c _ { 1 } ) \propto e ^ { 2 \kappa ( c _ { 2 } - c _ { 1 } ) z }$ increases in $z , \ s \mathbf { o } \ G ( \cdot \mid c )$ increases in likelihood-ratio, hence stochastic, order in c [18]. (b) Mixture form: $c ( y ) = s y +$ d increases in y, so $\textstyle \sum _ { z < t } G ( z \mid c ( y ) )$ decreases in $y ,$ and its average is larger under $q _ { 1 }$ than under $q _ { 2 }$ . Location form: $\begin{array} { r } { m ( q ) \ = \ \sum _ { z } z q ( z ) } \end{array}$ increases under $\preceq , \ s o \ s m ( q _ { 1 } ) + d \leq$ $s m ( q _ { 2 } ) + d$ and (a) applies. (c) Mixing with the same π or averaging over the same posterior keeps the order. □

## 2.4. Bayesian estimation from few labels

With five parameters, unlike with an unrestricted matrix, the posterior can be averaged over cheaply. We place independent Gaussian priors on (log τ, d, log s, log κ, logit ρ) centered at a near-identity channel $( \tau = 2 , d = 0 , s = 1 , \kappa = 4 , \rho =$ 0.95) with standard deviations 1, 1, 0.5, 1.5, and 2, and $\epsilon \sim$ $\mathcal { N } ( 0 , 0 . 5 ^ { 2 } ) , b \sim \mathcal { N } ( 0 , 1 )$ . We predict with the posterior mean of pˆ over 300 importance-reweighted draws from the Laplace approximation at the mode (covariance inflated by 1.5<sup>2</sup>). The two forms on the two readings give four channels.

## 2.5. Stacking with logit-space calibrators

The channels are combined with three calibrators that act on the logits: temperature scaling, vector scaling [8], and proportional-odds regression [16] on the centered logits with an $L _ { 2 }$ penalty of $1 0 0 / n$

$$
\hat { p } ( z \mid x ) = \sum _ { m = 1 } ^ { 7 } \omega _ { m } \hat { p } _ { m } ( z \mid x ) ,\tag{5}
$$

with weights ω that maximize the out-of-fold log-likelihood on four folds of the n labels under a Dirichlet(1) prior, found by EM; with fewer than eight labels the weights are equal. The same stack without channels (the logit stack), with Bayesian scalar channels, or with full $K \times K$ channels shrunk toward the scalar one by cross-validated KL penalties isolates what the structure adds. Channel forms and priors were chosen on validation texts from the calibration pool.

## 2.6. When does structure pay?

Let δ be the approximation gap of the structured family, the expected log loss of its best member minus that of the best unrestricted calibrator, and $d _ { \mathrm { s } } = 5 , d _ { \mathrm { u } }$ the parameter counts. Under the regularity conditions for maximum-likelihood estimation of possibly misspecified models [19], each estimator’s expected excess log loss is its gap plus about $d / ( 2 n )$ , so structure is expected to win when

$$
n < n ^ { \star } = \frac { d _ { \mathrm { u } } - d _ { \mathrm { s } } } { 2 \delta } .\tag{6}
$$

Higher-order terms and shrinkage move the actual crossover, so (6) predicts its scale rather than bounding it; Section 3 checks this with a known δ.

## 2.7. What a five-parameter posterior enables

A learned prior. The parameters mean the same thing on every task, so their prior can be learned from other tasks by empirical Bayes. We fit each channel on the full calibration pools of the other datasets, with every reader, express d and log κ in units of the scale length $K - 1$ , and use the mean of these fits with standard deviation min $( \sigma _ { 0 } , \sqrt { v + \sigma _ { 0 } ^ { 2 } / 4 } )$ , where v is their variance and $\sigma _ { 0 }$ the default.

Several readers. R frozen LLMs are R sensors of one label. Each reader gets its own stack (5), and the R stacked predictions are combined by a second out-of-fold stack together with a multi-reader location channel $\begin{array} { r } { G ( z \mid s \sum _ { r } w _ { r } m _ { r } ( x ) + } \end{array}$ $d )$ , with a temperature per reader and weights w on the simplex. This channel has $2 R + 3$ parameters and is estimated in the same Bayesian way as the single-reader channels.

Personalization. For repeated texts from one author u, the affine channel with $s = \rho = 1$ , offset $\Delta _ { u }$ , and $\kappa  \infty$ shifts the population prediction by $\Delta _ { u }$ classes (clipped at the ends); $\Delta _ { u }$ maximizes the likelihood of u’s earlier labeled texts minus $\Delta _ { u } ^ { 2 } / ( 2 \sigma ^ { 2 } )$ , with σ cross-validated over users. By Remark 1, a per-user reliability could only rescale u’s evidence.

## 3. EXPERIMENTS

Benchmarks. Amazon Reviews 2023 [20] provides review text, a 1–5 star rating $( K = 5 )$ , and a reviewer identifier; we use three product domains (Digital Music, All Beauty, Software), each with 2,000 calibration and 4,000 report reviews from single-review users plus 400 repeat users with at least eight reviews (up to 16 each, in time order). CMU-MOSEI [21] provides transcribed video segments with an averaged sentiment in [−3, 3], rounded to seven levels $( K =$ 7); its train and validation splits (18,135 segments) form the calibration pool and its test split (4,653 segments from 675 videos) the report pool.

Readers. Frozen Qwen2.5-Instruct 3B, 7B (main), and 14B [22] and Llama-3.1-8B-Instruct [23] are prompted with the text and the K options. The 7B reader’s mode is right for 75%, 66%, and 68% of Amazon reviews and 32% of MOSEI segments, and its errors are directional (Fig. 2).

Methods and protocol. All calibrators use the same $n \in$ $\{ 5 , 1 0 , \ldots , 2 0 0 0 \}$ calibration labels (20 draws each) and are scored on the report pool by NLL and the ranked probability score (RPS), with probabilities floored at $1 0 ^ { - 4 }$ . Baselines are temperature and vector scaling [8], proportional-odds regression on the centered logits [16] with a cross-validated $L _ { 2 }$ penalty, Dirichlet calibration [9] with cross-validated offdiagonal and intercept penalties, and the three stacks of Section 2.5. Label-free batch calibration [10] does not lower the raw NLL. Paired 95% intervals use 10,000 bootstrap draws over report items and label subsets; a difference is resolved when its interval excludes zero. Fitting ours takes 7–30 s on one CPU core and predicting 0.2 ms per text.

Table 1. Test NLL at $n = 2 0$ and $n = 2 0 0$ calibration labels, 7B reader (means over 20 subsets; bold: best). CORDIAL adds the structured channels to the logit stack (LS).
<table><tr><td></td><td>Music</td><td>Beauty</td><td>Software</td><td></td><td>MOSEI</td></tr><tr><td>Method</td><td>20 200</td><td>20</td><td>200</td><td>20 200</td><td>20 200</td></tr><tr><td>Raw LLM</td><td>1.3391.339 2.052 2.052 2.093 2.093 5.806 5.806</td><td></td><td></td><td></td><td></td></tr><tr><td>Temperature</td><td>0.6930.5440.7660.7430.792 0.7631.6821.662</td><td></td><td></td><td></td><td></td></tr><tr><td>Vector scaling</td><td>0.907 0.438 0.953 0.5931.023 0.6751.7291.283</td><td></td><td></td><td></td><td></td></tr><tr><td>Ordinal regr.</td><td>0.787 0.454 0.789 0.5681.028 0.7151.5401.186</td><td></td><td></td><td></td><td></td></tr><tr><td>Dirichlet</td><td>0.982 0.4491.181 0.5391.3080.663 2.0201.221</td><td></td><td></td><td></td><td></td></tr><tr><td>Logit stack (LS) 0.519 0.426 0.654 0.551 0.742 0.661 1.3831.183</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LS + scalar ch.</td><td></td><td></td><td></td><td></td><td>0.5040.4290.6620.5540.7490.6641.3951.187</td></tr><tr><td>LS + full ch.</td><td>0.5280.4260.6590.5520.7480.6601.3971.186</td><td></td><td></td><td></td><td></td></tr><tr><td>CORDIAL</td><td></td><td></td><td></td><td></td><td>0.473 0.420 0.596 0.541 0.718 0.6631.330 1.172</td></tr></table>

## 3.1. Few labels: structure beats size

With the 7B reader, ours has the lowest NLL among nine calibrators, the eight in Table 1 and the fixed-penalty ordinal regression used inside the stack, at every budget from 5 to 100 labels on all four benchmarks (Fig. 1). With 20 labels it matches the logit stack with 28–54 labels and Dirichlet calibration with 79–132. Against the logit stack it gains 0.046, 0.057, 0.024, and 0.053 nats at n = 20 for Music, Beauty, Software, and MOSEI and 0.002–0.016 at $n = 1 0 0 .$ with every gain resolved except on Software beyond 20 labels; the same stack with full or scalar channels changes the logit stack’s NLL at n = 20 by −0.015 to +0.014 and trails ours by resolved margins of 0.031–0.067, so the gain comes from the structure rather than from more members. Posterior averaging adds up to 0.134 nats over the posterior mode with ten labels or fewer and nothing from 100 on. The RPS agrees, favoring ours over the logit stack by 0.008–0.017 at $n = 2 0$ and accuracy matches the logit stack. Grouped by the reader’s stated class (Fig. 2), ours is closest to the labels on all four benchmarks at $n = 2 0 !$ its class-weighted KL is 0.06–0.17, against $0 . 1 0 – 0 . 2 2$ for the logit stack, 0.31–0.55 for Dirichlet calibration, and 0.32–1.97 for the raw reader. The labels are ordered in y on every benchmark, and so, by Proposition 1, are the temperature-reading channels; fitted on 20 labels, vector scaling and Dirichlet calibration map saturated readings of some adjacent classes to predictions reversed by more than 0.05 in cumulative probability in 45–80% and 45–70% of fits, and ours in at most 10%.

The crossover. Dirichlet calibration, the strongest unrestricted calibrator, crosses zero in Fig. 1 and first leads with a resolved interval at 1000, 500, 500, and 2000 labels and at $n \ : = \ : 2 0 0 0$ by 0.018, 0.022, 0.042, and 0.009 nats. Using the large-n gaps as plug-in estimates of δ, Eq. (6) with $d _ { \mathrm { u } } = K ( K + 1 )$ gives $n ^ { \star } = 7 0 0$ , 565, 297, and 2746, all within a factor of two of the observed crossovers and preserving their ordering. With labels from a known channel outside the affine family $( \delta = 0 . 0 0 1 \mathrm { - } 0 . 0 9 )$ , maximum-likelihood excess losses follow $d / ( 2 n )$ ) with d ≈ 4 for the four-parameter channel and 1.1–1.3 K(K − 1) for the full matrix, overtaking at 1.2–2.7 n<sup>⋆</sup> without shrinkage and 0.2–1.3 n<sup>⋆</sup> with it.

![](images/3a7be04c2bedffba919aeeac75f864d9c66d0acf46d67949b6fba8de932adce3.jpg)

![](images/7abd8933ceb4abef76c13e241f07f0e8302557861b1563f9dad7cc387cdc427e.jpg)

![](images/42834fe78259b07015b6f75bdd2cf1e7d5ebba4da974cd2b98b76cfe2688c945.jpg)

![](images/fc70876560762f629a707d5cb186574ccaefd5947cc66f50edd3316d6a7769e9.jpg)  
CORDIAL (Δ = 0) Logit stack (LS) LS + full ch. Ordinal regr. Temperature Vector scaling Dirichlet

Fig. 1. Test NLL of each calibrator minus that of CORDIAL (above zero: CORDIAL better) against the number of calibration labels n, 7B reader, means over 20 label subsets; bands: paired 95% bootstrap intervals for the three strongest baselines. The logit stack (LS) combines temperature scaling, vector scaling, and ordinal regression. Curves above a panel’s range are cut.  
![](images/0d21c45c674ed48aca63ecdd8719f74f716ecea950fe5e65a5cb8d2683314249.jpg)  
Fig. 2. Label distributions given the 7B reader’s predicted class y (rows) over the true label z (columns; dashed: $z = y )$ 20 labels, mean over 20 subsets. Empirical: observed label frequencies. The raw reading is nearly a point mass at z = y; the labels shift up on Amazon and toward neutral on MOSEI.

Reader size and family. With the 3B, 14B, and Llama readers, CORDIAL is best in 56 of 60 settings with 5 to 100 labels, trails by at most 0.006 in the other four, and leads the logit stack at $n = 2 0$ by resolved margins of 0.012–0.086.

In-context examples. Used as in-context demonstrations for the 7B reader, the same 20 to 100 labels give accuracy within $- 0 . 0 6 { \mathrm { t o } } - 0 . 0 2$ of ours but a 2.3 to 3.3 times higher NLL.

## 3.2. What the posterior buys

A learned prior. With the prior learned from the other datasets, ours gains on average 0.017, 0.016, 0.011, 0.006, and 0.003 nats with 5, 10, 20, 50, and 100 labels over the twelve Amazon reader–domain pairs; the gain is resolved for

9 or 10 pairs at each budget, and no pair shows a resolved loss. On MOSEI, with a prior from Amazon alone, it costs 0.009–0.011 at $n \leq 1 0$ and gains 0.013–0.015 at 20–50.

Several readers. The four-reader fusion, designed on validation texts, beats the 7B reader alone in all 28 settings from 5 to 500 labels, by 0.008–0.090 nats, the reader chosen by cross-validation in all 28, and the same fusion without channels in 27, by 0.011–0.043 at n = 20; Dirichlet calibration of the concatenated readings trails by 0.55–1.5 nats there. On MOSEI with 10–20 labels Llama alone is 0.018–0.030 better; the multi-reader channel adds at most 0.006.

Across users. A per-user shift channel on the population fit (1,000 labels) gains, after eleven or more earlier texts of the user, 0.100, 0.046, 0.044, and 0.048 nats on Music, Beauty, Software, and MOSEI, all resolved. Over all texts it beats a per-user reliability by 0.013–0.041, a per-user temperature by 0.004–0.016, and a random-intercept ordinal regression by 0.021–0.029 on Amazon, tying it on MOSEI.

## 4. CONCLUSION

An LLM’s ordinal output can be viewed as a noisy sensor with systematic offset, gain, and concentration errors. Our fiveparameter Bayesian channel calibrates these distortions from few labels and naturally extends to learned priors, multiple readers, and user-specific adaptation.

## 5. REFERENCES

[1] Saurav Kadavath et al., “Language models (mostly) know what they know,” arXiv preprint arXiv:2207.05221, 2022.

[2] Zihao Zhao et al., “Calibrate before use: Improving few-shot performance of language models,” in Proc. International Conference on Machine Learning, 2021, pp. 12697–12706.

[3] Lianmin Zheng et al., “Judging LLM-as-a-judge with MT-Bench and Chatbot Arena,” in Advances in Neural Information Processing Systems, Datasets and Benchmarks Track, 2023.

[4] Wenxuan Zhang et al., “Sentiment analysis in the era of large language models: A reality check,” arXiv preprint arXiv:2305.15005, 2023.

[5] Jiaqing Zhang et al., “Auditing multimodal LLM raters: Central tendency bias in clinical ordinal scoring,” arXiv preprint arXiv:2605.16386, 2026.

[6] OpenAI, “GPT-4 technical report,” arXiv preprint arXiv:2303.08774, 2023.

[7] Katherine Tian et al., “Just ask for calibration: Strategies for eliciting calibrated confidence scores from language models fine-tuned with human feedback,” in Proc. Conference on Empirical Methods in Natural Language Processing, 2023, pp. 5433–5442.

[8] Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger, “On calibration of modern neural networks,” in Proc. International Conference on Machine Learning, 2017, vol. 70, pp. 1321–1330.

[9] Meelis Kull et al., “Beyond temperature scaling: Obtaining well-calibrated multi-class probabilities with Dirichlet calibration,” in Advances in Neural Information Processing Systems, 2019, vol. 32.

[10] Han Zhou et al., “Batch calibration: Rethinking calibration for in-context learning and prompt engineering,” in Proc. International Conference on Learning Representations, 2024.

[11] Daehwan Kim, Haejun Chung, and Ikbeom Jang, “Ordinal-aware calibration for ordinal classification,” arXiv preprint arXiv:2410.15658, 2024.

[12] Yu Wang, Zhe Zhou, Menglin Liu, and Ge Shi, “Are LLMs positionally consistent ordinal classifiers? A systematic evaluation,” arXiv preprint arXiv:2608.08869, 2026.

[13] A. P. Dawid and A. M. Skene, “Maximum likelihood estimation of observer error-rates using the EM algorithm,” Journal of the Royal Statistical Society: Series C (Applied Statistics), vol. 28, no. 1, pp. 20–28, 1979.

[14] Dengyong Zhou et al., “Aggregating ordinal labels from crowds by minimax conditional entropy,” in Proc. International Conference on Machine Learning, 2014, vol. 32, pp. 262–270.

[15] Gavin Kerrigan, Padhraic Smyth, and Mark Steyvers, “Combining human predictions with model probabilities via confusion matrices and calibration,” in Advances in Neural Information Processing Systems, 2021, vol. 34, pp. 4421–4434.

[16] Peter McCullagh, “Regression models for ordinal data,” Journal of the Royal Statistical Society: Series B, vol. 42, no. 2, pp. 109–142, 1980.

[17] Christian Genest and James V. Zidek, “Combining probability distributions: A critique and an annotated bibliography,” Statistical Science, vol. 1, no. 1, pp. 114–135, 1986.

[18] Moshe Shaked and J. George Shanthikumar, Stochastic Orders, Springer, 2007.

[19] Halbert White, “Maximum likelihood estimation of misspecified models,” Econometrica, vol. 50, no. 1, pp. 1–25, 1982.

[20] Yupeng Hou et al., “Bridging language and items for retrieval and recommendation,” arXiv preprint arXiv:2403.03952, 2024.

[21] AmirAli Bagher Zadeh et al., “Multimodal language analysis in the wild: CMU-MOSEI dataset and interpretable dynamic fusion graph,” in Proc. Annual Meeting of the Association for Computational Linguistics, 2018, pp. 2236–2246.

[22] An Yang et al., “Qwen2.5 technical report,” arXiv preprint arXiv:2412.15115, 2024.

[23] Abhimanyu Dubey et al., “The Llama 3 herd of models,” arXiv preprint arXiv:2407.21783, 2024.

## 6. COMPLIANCE WITH ETHICAL STANDARDS

This study reuses two public datasets, Amazon Reviews 2023 and CMU-MOSEI transcripts, with the user and speaker identifiers they provide. It involves no new participant recruitment or intervention.