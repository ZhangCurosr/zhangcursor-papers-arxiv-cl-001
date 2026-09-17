# Attention Dispersion as a Diagnostic Signal for Hallucination in Large Language Models

Shardul P. More Rajarambapu Institute of Technology Ishwarpur, Maharashtra, India shardulmore112@gmail.com

Tanuja S. Pawar Rajarambapu Institute of Technology Ishwarpur, Maharashtra, India tanujapawar203@gmail.com

## Abstract

Large Language Models (LLMs) frequently exhibit hallucinations, presenting a major barrier to reliability in complex reasoning tasks. While traditional detection methods rely on outputbased confidence metrics, these logits are often miscalibrated by modern alignment techniques. In this paper, we investigate the temporal volatility of internal attention mechanisms as an alternative diagnostic signal for hallucination that does not depend on output calibration. By introducing an unsupervised metric for attention dispersion, we show that epistemic uncertainty leaves a measurable trace within intermediate layers, where spikes in attention entropy are associated with reasoning breakdowns. We evaluate our approach on mathematical reasoning benchmarks (GSM8K and MATH-500) using the Qwen2.5 model family (1.5B and 3B parameters), finding statistically significant AUC improvements of up to +0.076 over output-based baselines across all tested conditions. These findings suggest that attention dispersion is a promising complement to traditional hallucination detection methods, requiring further investigation across broader model families and task domains.

## 1 Introduction

The rapid growth and integration of Large Language Models (LLMs) present a challenge in identifying hallucinations, in which the models are confident yet factually incorrect in their outputs. Initial approaches to uncertainty estimation heavily relied on the model’s output probabilities, leveraging maximum softmax probabilities as a baseline for detecting errors (Hendrycks and Gimpel, 2018). Further studies demonstrated that large language models can often evaluate the validity of their own claims through self-evaluation prompts (Kadavath et al., 2022). However, recent alignment strategies, such as Reinforcement Learning from Human Feedback (RLHF), often cause models to exhibit systematic, verbalized overconfidence regardless of actual response quality (Leng et al., 2025). Consequently, the output logits are structurally compromised as diagnostic indicators.

Recent methodologies have explored internal representations to bypass the corrupted logits. Classifiers such as SAPLMA (Azaria and Mitchell, 2023) are used to extract signals of truthfulness directly from hidden-layer activations. More importantly, these approaches often do not consider the dynamic, multi-step nature of sequential reasoning. Even though mid-depth layers act as critical information bottlenecks (Skean et al., 2025), the temporal fluctuations or volatility of the attention mechanism across these layers has not been thoroughly investigated as a diagnostic tool.

We hypothesize that hallucination is not just a static representation error, but a dynamic failure in reasoning stability. Grounded reasoning maintains a consistent attention trajectory through the depth of the transformer, whereas epistemic failure appears as erratic, highly dispersed attention shifts. Our contributions are threefold: (1) we propose the theoretical framing that the temporal dispersion of attention entropy can serve as a zero-shot, internal indicator of epistemic failure; (2) we introduce a metric, step\_attn\_std, to track reasoning volatility across intermediate layers without relying on output logits; and (3) we provide empirical validation on GSM8K and MATH-500, demonstrating that dynamic attention dispersion provides a predictive signal for hallucination detection without requiring additional probe training.

## 2 Related Work

For the purpose of enhancing the reliability of the reasoning of large language models and measuring their uncertainty, researchers have explored sampling-based methods. Self-Consistency was introduced as a method that selects the most frequently occurring answer among diverse reasoning paths, showcasing significant improvements in logical as well as general commonsense tasks (Wang et al., 2023). Similarly, Semantic Entropy was proposed to cluster multiple responses based on semantic equivalence, detecting hallucinations by measuring uncertainty across these clusters (Farquhar et al., 2024). While these strategies provide robust uncertainty signals, they require multiple generation passes per input. For instance, Semantic Entropy requires 5 to 10 generations, resulting in higher computational costs. To mitigate this overhead, recent work has proposed Semantic Entropy Probes (SEPs), which approximate semantic uncertainty directly from the hidden states of a single generation (Kossen et al., 2024).

Another line of research is verbalized confidence, where the language models are required to report the certainty level of a generated answer explicitly. Studies on the CalibratedMath benchmark showed that models can be trained to express uncertainty by reporting numerical probabilities or linguistic features describing something (Lin et al., 2022). Furthermore, confidence elicitation has been expanded to RLHF-aligned models, discovering that verbalized confidence can sometimes offer better calibration than the model’s conditional probabilities (Tian et al., 2023). However, while models can meaningfully estimate confidence in natural language, this signal remains dependent on the model’s explicit verbal outputs, which can still be likely influenced by alignment-induced overconfidence (Leng et al., 2025).

To bypass output-dependent metrics, mechanistic interpretability analyzes internal model computations. Previous work has mapped Transformer circuits to understand component interactions (Elhage et al., 2021) and identified middle-layer feedforward networks as essential mediators of factual prediction (Meng et al., 2022). These studies confirm that robust knowledge representations reside strictly within intermediate layers. Building on this foundation, we shift from static feature extraction to dynamically tracking the temporal instability of these attention mechanisms.

## 3 Methodology

Our objective is to quantify the knowledge-related stability of a Large Language Model (LLM) during sequential reasoning without relying on its output logits. To achieve this, we introduce a framework that measures the temporal dispersion of attention entropy across the network’s layers at each generation step.

## 3.1 Layer-Wise Attention Entropy

Consider a transformer model with L layers. At generation step t, the model attends to the preceding context of length k. For a given layer $l \in \{ 1 , \ldots , L \}$ and attention head $h \in \{ 1 , \ldots , H \}$ let $\alpha _ { l , h } ^ { ( t ) } \in \mathbb { R } ^ { k }$ denote the attention probability distribution over the context tokens.

To quantify the positional or spatial focus of the model at layer l, we first calculate the Shannon entropy of the attention distribution. We define the layer-wise attention entropy $E _ { l } ^ { ( t ) }$ as the average entropy across all H heads in layer l:

$$
E _ { l } ^ { ( t ) } = \frac { 1 } { H } \sum _ { h = 1 } ^ { H } \left( - \sum _ { i = 1 } ^ { k } \alpha _ { l , h , i } ^ { ( t ) } \log \alpha _ { l , h , i } ^ { ( t ) } \right)
$$

A low entropy value indicates a sharp, highly localized attention focus, while a high entropy value indicates broad, dispersed attention across the context.

## 3.2 Temporal Attention Dispersion

Because intermediate layers systematically compress semantic features (Skean et al., 2025), grounded reasoning should exhibit a wellconnected, stable trajectory of attention entropy through this depth. In contrast, we propose that epistemic failure disrupts this trajectory, causing volatile shifts in attention entropy across layers.

To measure this volatility, we calculate the standard deviation of the layer-wise entropies across the entire depth of the model. First, we define the mean attention entropy across all layers at step t:

$$
\bar { E } ^ { ( t ) } = \frac { 1 } { L } \sum _ { l = 1 } ^ { L } E _ { l } ^ { ( t ) }
$$

Nex $\mathbf { \delta } _ { \mathrm { ( t , ~ } }$ we define our primary metric, temporal attention dispersion (step\_attn\_std), as the standard deviation of these layer-wise entropies from the mean:

$$
s t e p _ { - } a t t n _ { - } s t d ^ { ( t ) } = \sqrt { \frac { 1 } { L } \sum _ { l = 1 } ^ { L } \left( E _ { l } ^ { ( t ) } - \bar { E } ^ { ( t ) } \right) ^ { 2 } }
$$

By extracting step\_attn ${ } _ { s t d } ( t )$ at each generation step, we obtain a dynamic, zero-shot signal of the model’s reasoning stability. This metric is computed strictly from the internal self-attention mechanisms, rendering it completely independent of miscalibrated output logits.

## 4 Experiments

To test our hypothesis that temporal attention dispersion is a reliable signal of epistemic failure, we evaluate our metrics against standard output-based measures of confidence across multiple reasoning datasets and model scales.

## 4.1 Datasets and Models

We select two complex mathematical reasoning benchmarks that require multi-step sequential logic, where output-level hallucination is highly prevalent: GSM8K (Cobbe et al., 2021), a dataset of high-quality grade-school math word problems, and MATH-500 (Lightman et al., 2023), a curated subset of the MATH dataset featuring highly complex, competition-level mathematics.

To evaluate the scalability and robustness of our approach, we extract internal states and outputs from two different models of the Qwen family(Yang et al., 2024): a 1.5B-parameter model (evaluated on both GSM8K and MATH-500) and a 3B-parameter model (evaluated on MATH-500). Ground-truth labels are extracted using a strict, stack-based LaTeX parsing algorithm to perfectly align model generations with boxed mathematical solutions.

## 4.2 Feature Extraction

For each generated trajectory, we aggregate the step-level metrics into sequence-level features to train a lightweight hallucination classifier. For output baseline features, we compute the mean and maximum of the predictive entropy at the output layer (mean\_out\_entropy, max\_out\_entropy), alongside the sequence-level output probability (out\_tau). For our proposed attention features, we aggregate the spatial attention entropy (mean\_attn\_entropy, max\_attn\_entropy) and sequence-level attention confidence (attn\_tau). Crucially, we also include the temporal dispersion of attention across layers, aggregated over the sequence (mean\_attn\_std).

## 4.3 Evaluation Protocol

We frame hallucination detection as a binary classification task where the objective is to predict reasoning failure (i.e., whether the final generated answer is incorrect). We train a standard Logistic Regression classifier using Scikit-learn (Pedregosa et al., 2018) with balanced class weights to predict failure based on the extracted features.

To ensure robust estimation of model performance and prevent overfitting, we employ a Repeated Stratified K-Fold cross-validation strategy (5 splits, 10 repeats). All features are standardized prior to training. We evaluate the models using the Area Under the Receiver Operating Characteristic Curve (ROC-AUC). To determine statistical significance between the proposed attention-based features and the output-based baselines, we compute 95% confidence intervals for the performance delta and apply a one-sided Wilcoxon signed-rank test.

## 5 Results and Analysis

Our evaluations suggest that internal attention dispersion provides a predictive, orthogonal signal for hallucination detection that consistently outperforms traditional output-based metrics on complex reasoning tasks.

## 5.1 Performance on Reasoning Benchmarks

Table 1 summarizes the classification performance across different benchmarks and model scales. The proposed attention-driven feature set consistently outperforms the baseline across all tested configurations.

Across the evaluated settings, attention dispersion appears increasingly informative in larger models and more challenging reasoning tasks. However, these observations are suggestive rather than conclusive, as model scale and reasoning complexity are not independently controlled in our experiments. While the proposed metric yields a modest gain on the GSM8K benchmark (+0.0093 AUC), the performance gain is larger on the more challenging MATH-500 benchmark, reaching +0.0486 AUC for the 1.5B model and +0.0759 AUC for the 3B model.

As further illustrated in Figure 1, the fusion of both signals (Combined) yields the highest predictive power, indicating that temporal attention dispersion captures failure modes not well reflected in output logits alone.

## 5.2 Mechanistic Feature Importance

To understand the drivers of this performance, we analyze the feature weights of the trained logistic regression classifier. The analysis reveals that mean\_attn\_entropy and max\_attn\_entropy carry significant weight, but critically, they act in opposing directions. A high maximum attention entropy predicts failure, aligning with our hypothesis that sudden, dispersed attention spikes indicate a breakdown in reasoning focus.

![](images/1b77a6aad96de74173ab6a01ba16b787321e2c2d20795e851e9a8831de540ad3.jpg)  
Figure 1: Out-of-Fold ROC Curves on MATH-500 (3B). The internal attention features outperform standard output metrics, while the combined model achieves the highest AUC.

![](images/759ba9346216c524639f7ea7b483a454a5127b00f318354a890408bcdc1b1a4d.jpg)  
Figure 2: Logistic Regression Coefficients on MATH-500 (3B). Positive weights strongly predict reasoning failure.

<table><tr><td>Benchmark</td><td>Model</td><td>Base AUC</td><td>Prop. AUC</td><td>Gain (95% CI)</td><td>p-value</td></tr><tr><td>GSM8K</td><td>Qwen2.5-1.5B</td><td>0.6897</td><td>0.6990</td><td>+0.0093 ([0.003, 0.015])</td><td> $1 . 6 6 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>MATH-500</td><td>Qwen2.5-1.5B</td><td>0.6259</td><td>0.6745</td><td>+0.0486 ([0.031, 0.067])</td><td> $5 . 4 9 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>MATH-500</td><td>Qwen2.5-3B</td><td>0.5817</td><td>0.6575</td><td>+0.0759 ([0.057, 0.095])</td><td> $5 . 6 5 \times 1 0 ^ { - 8 }$ </td></tr></table>

Table 1: Evaluation results showing AUC performance across models and datasets. The 95% confidence intervals (CI) and p-values (Wilcoxon signed-rank test) confirm the statistical significance of the gains.

The classifier’s reliance on these internal attention features, even when output metrics like out\_tau are available, suggests that attention volatility carries information about hallucination risk not captured by output confidence alone — a pattern consistent with our hypothesis, though further work across model families and task types is needed to establish generality.

## 6 Conclusion

We investigated the temporal volatility of internal attention mechanisms as a diagnostic signal for hallucination in LLMs. Our results suggest that epistemic uncertainty leaves a measurable trace within intermediate layers, offering a signal less exposed to the miscalibration affecting output logits. Using an unsupervised metric for attention dispersion, we found that spikes in attention entropy are associated with reasoning breakdowns. Evaluations on GSM8K and MATH-500 with the Qwen2.5 family show this internal signal consistently outperforms output-based baselines across all tested conditions, with gains that appear larger in more challenging configurations — though model scale and task complexity are not independently controlled in our design. These findings point toward internal computational signals as a promising complement to output-based hallucination detection. Future work will extend this framework to diverse architectures, non-mathematical domains, and direct comparison against sampling-based uncertainty methods, while exploring the causal drivers of these attention breakdowns.

## Limitations

While our findings suggest the potential of temporal attention dispersion as a diagnostic measure for hallucination, this study has several important limitations that require consideration:

Domain Specificity: Our empirical evaluation is strictly confined to mathematical reasoning tasks (GSM8K and MATH-500). While multi-step mathematics provides an excellent testbed for sequential logic, the observed attention dynamics need to be analyzed in other modalities of hallucination, such as factual fabrications in open-domain question answering, creative generation, or translation tasks.

Architectural Scope: The experiments were conducted solely on the Qwen2.5 model family (1.5B and 3B parameters). We have not yet verified whether these temporal attention dispersion patterns remain consistent across fundamentally different Transformer variants, such as Mixture-of-Experts (MoE) architectures, or models utilizing alternative attention mechanisms.

White-Box Requirement: Our proposed metric, step\_attn\_std, fundamentally relies on extracting intermediate attention distributions across all layers of the network. Consequently, this diagnostic framework is limited to open-weight models and cannot be applied to proprietary, black-box APIs (e.g., GPT-4, Claude) where only final textual outputs or restricted output logits are accessible.

Confounding Variables in Scaling: As noted in our analysis, the predictive advantage of our metric appeared greater on the larger model (3B) and the more complex benchmark (MATH-500). However, our experimental design did not independently control for model scale versus task complexity, limiting our ability to determine what is actually driving this trend.

Missing Empirical Baselines: While we position our work against sampling-based uncertainty methods such as Semantic Entropy and Semantic Entropy Probes in our related work, we do not include a direct empirical comparison against these methods in this study. Our evaluation is limited to output-logit-based baselines; establishing relative performance against these alternative approaches is left to future work.

Sample Size: Our evaluation, particularly for the 3B model on MATH-500 (n = 254), involves a limited sample size relative to the scale often used in large-scale uncertainty quantification studies. While our statistical tests indicate significant effects within this sample, further validation on larger evaluation sets would strengthen confidence in the generalizability of the reported effect sizes.

Step Alignment Coverage: Our step-level parsing algorithm did not achieve complete alignment coverage on MATH-500 due to its more complex LaTeX notation compared to GSM8K, potentially introducing selection effects if parsing failures are not independent of reasoning correctness.

## Acknowledgments

This research was conducted independently without external institutional funding. The authors gratefully acknowledge Kaggle for providing the computation required for the execution of the model evaluations. We disclose the use of generative AI tools during manuscript and pipeline preparation: Anthropic’s Claude Sonnet 5 was used to assist with code generation and pipeline development, and Google’s Gemini 3.1 Pro and Grammarly were used for proofreading and language polishing. All AI-assisted code, analysis, and text were reviewed, verified, and revised by the authors, who take full responsibility for the content of this manuscript.

## References

Amos Azaria and Tom Mitchell. 2023. The Internal State of an LLM Knows When It’s Lying. arXiv preprint. ArXiv:2304.13734 [cs.CL].

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training Verifiers to Solve Math Word Problems. arXiv preprint. ArXiv:2110.14168 [cs.LG].

Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Nova DasSarma, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, and 6 others. 2021. A Mathematical Framework for Transformer Circuits. Transformer Circuits Thread.

Sebastian Farquhar, Jannik Kossen, Lorenz Kuhn, and Yarin Gal. 2024. Detecting hallucinations in large language models using semantic entropy. Nature, 630(8017):625–630.

Dan Hendrycks and Kevin Gimpel. 2018. A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks. arXiv preprint. ArXiv:1610.02136 [cs.NE].

Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, Scott Johnston, Sheer El-Showk, Andy Jones, Nelson Elhage, Tristan Hume, Anna Chen, Yuntao Bai, Sam Bowman, Stanislav

Fort, and 17 others. 2022. Language Models (Mostly) Know What They Know. arXiv preprint. ArXiv:2207.05221 [cs.CL].

43 others. 2024. Qwen2 Technical Report. arXiv preprint. ArXiv:2407.10671 [cs.CL].

Jannik Kossen, Jiatong Han, Muhammed Razzak, Lisa Schut, Shreshth Malik, and Yarin Gal. 2024. Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs. arXiv preprint. ArXiv:2406.15927 [cs.CL].

Jixuan Leng, Chengsong Huang, Banghua Zhu, and Jiaxin Huang. 2025. Taming Overconfidence in LLMs: Reward Calibration in RLHF. arXiv preprint. ArXiv:2410.09724 [cs.CL].

Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. 2023. Let’s Verify Step by Step. arXiv preprint. ArXiv:2305.20050 [cs.LG].

Stephanie Lin, Jacob Hilton, and Owain Evans. 2022. Teaching Models to Express Their Uncertainty in Words. arXiv preprint. ArXiv:2205.14334 [cs.CL].

Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. 2022. Locating and Editing Factual Associations in GPT. In Advances in Neural Information Processing Systems, volume 35, pages 17359–17372. Curran Associates, Inc.

Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion, Olivier Grisel, Mathieu Blondel, Andreas Müller, Joel Nothman, Gilles Louppe, Peter Prettenhofer, Ron Weiss, Vincent Dubourg, Jake Vanderplas, Alexandre Passos, David Cournapeau, Matthieu Brucher, Matthieu Perrot, and Édouard Duchesnay. 2018. Scikitlearn: Machine Learning in Python. arXiv preprint. ArXiv:1201.0490 [cs.LG].

Oscar Skean, Md Rifat Arefin, Dan Zhao, Niket Patel, Jalal Naghiyev, Yann LeCun, and Ravid Shwartz-Ziv. 2025. Layer by Layer: Uncovering Hidden Representations in Language Models. arXiv preprint. ArXiv:2502.02013 [cs.LG].

Katherine Tian, Eric Mitchell, Allan Zhou, Archit Sharma, Rafael Rafailov, Huaxiu Yao, Chelsea Finn, and Christopher D. Manning. 2023. Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback. arXiv preprint. ArXiv:2305.14975 [cs.CL].

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2023. Self-Consistency Improves Chain of Thought Reasoning in Language Models. arXiv preprint. ArXiv:2203.11171 [cs.CL].

An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, and