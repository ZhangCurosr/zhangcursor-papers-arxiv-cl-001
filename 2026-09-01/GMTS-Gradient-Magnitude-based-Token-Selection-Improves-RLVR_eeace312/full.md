# GMTS: Gradient Magnitude-based Token Selection Improves RLVR Training for LLM Reasoning

Outongyi Lv<sup>1\*</sup> Yuanwei Zhang<sup>1\*</sup> Xiaoqun Zhang<sup>1,2†</sup>

<sup>1</sup>School of Mathematical Sciences, Shanghai Jiao Tong University <sup>2</sup>Institute of Natural Sciences, Shanghai Jiao Tong University {harry\_lv,sjtuzyw,xqzhang}@sjtu.edu.cn

## Abstract

Reinforcement learning (RL), particularly RL with Verifiable Rewards (RLVR), has recently emerged as a central paradigm for enhancing large language models’ (LLMs) reasoning abilities, demonstrating remarkable effectiveness across reasoning tasks. Recent studies suggest that high-entropy tokens play an exceptionally important role in model training, since training with only the highest 20% entropy tokens yields significant performance gains. However, why such high-entropy tokens are beneficial remains insufficiently understood. In this work, we find that although high-entropy tokens within one answer tend to correlate with large gradient magnitude, entropy alone fails to consistently reflect token importance across different answers, considering the variations in the answer-level reward signals. Based on this observation, we introduce the Gradient Magnitude-based Token Selection (GMTS) method to quantify token importance, which leverages the entropy–gradient connection to approximate gradient-magnitude rankings for token selection. We find that training on the top 20% tokens ranked by GMTS consistently outperforms entropy-based token selection across three reasoning domains and various model sizes, suggesting that GMTS provides a more fine-grained estimate of token contribution for RLVR training.<sup>1</sup>

## 1 Introduction

Large language models (LLMs) have recently shown remarkable capabilities across diverse domains, such as mathematics, image quality assessment, speech, and multimodality (Achiam et al., 2023; Wu et al., 2025; Shen et al., 2025; Bie et al.,

2026). Reinforcement learning (RL) has made substantial contributions to the advancement of LLMs, demonstrating outstanding effectiveness in enhancing performance (Ouyang et al., 2022; Lee et al., 2024; Shinn et al., 2023). DeepSeek introduced Group Relative Policy Optimization (GRPO) (Shao et al., 2024), a method that dispenses with the value model in favor of simple rule-based rewards and adopts group-level advantage estimation, leading to significant improvements on reasoning benchmarks. Based on GRPO, several variants, such as Dynamic sAmpling Policy Optimization (DAPO) (Yu et al., 2025) and Group Sequence Policy Optimization (GSPO) (Zheng et al., 2025), have been proposed, showing that RLVR provides a scalable and efficient paradigm for advancing the reasoning abilities of LLMs. Nevertheless, these methods adopt a uniform objective function across all tokens, without fully accounting for the fact that different tokens contribute unevenly to RLVR.

Recent studies have focused on distinguishing the token importance by designing token-level advantage functions (Yang et al., 2025; Liu et al., 2026; Ma et al., 2026), introducing new objective functions (Cui et al., 2025; Li et al., 2025), and filtering out low importance tokens (Wang et al., 2025c; Lv et al., 2026). Concurrently, several studies have also emphasized the central role of high token entropy in enhancing RLVR (Wang et al., 2025c; Cui et al., 2025; Meng et al., 2026), and controlling entropy provides a way to strengthen the RLVR reasoning ability. In these works, Wang et al. (2025c) shows that training with only the top 20% high-entropy tokens yields substantially better performance across the reasoning tasks than training with all tokens.

## However, can entropy alone sufficiently indicate token importance?

In this work, we observe that within a single answer, high-entropy tokens tend to be associated with large gradient magnitude. However, due to variations in reward signals and sample-specific characteristics, entropy alone fails to adequately capture token importance across different answers, whereas gradient magnitude provides a more reliable quantification. Based on this, we propose the Gradient Magnitude-based Token Selection (GMTS) method. In contrast to Entropy-based Token Selection (ETS) (Wang et al., 2025c) under the same settings, GMTS can more effectively distinguish token importance across different answers. Empirically, our experiments show that GMTS consistently improves performance across diverse reasoning tasks and model sizes. For example, GMTS achieves approximately 1-3 percentage point gains on math reasoning benchmarks. Furthermore, GMTS is also simple to implement and can be easily integrated into existing RLVR frameworks such as GRPO and DAPO. Since all the required components are already available during standard training, GMTS introduces minimal computational overhead. In summary, the contributions of this work are threefold:

![](images/2914bb0119406d7158b249f81cc04a79206d06ef05cce33120674ed4c5d38e78.jpg)  
Figure 1: Overall architecture of the GMTS framework: High-entropy tokens selected by ETS may contribute less to the token gradient or be affected by gradient clipping. By computing $\delta _ { i , t } ,$ , GMTS mitigates this problem. Moreover, GMTS can identify low-entropy tokens that are overlooked by ETS but still contribute more on token gradient.

• Establish the relation between token entropy and gradient magnitude. We observe that within a single answer, tokens with high entropy correlate strongly with large gradient magnitude, which explains why entropy is an effective indicator of token importance. However, we further demonstrate that this correlation does not hold consistently across different answers when their reward signals vary significantly. This motivates the use of gradient magnitude as a more robust and reliable metric for quantifying token importance.

• GMTS for quantifying token importance. Building upon the relationship between entropy and gradient magnitude, we propose GMTS for quantifying token importance. GMTS effectively retains the benefits of using entropy as a signal while overcoming its key limitations in quantifying token importance across different answers.

• Empirical studies across reasoning domains and model scales. Our empirical studies show that GMTS consistently outperforms the baselines across three reasoning domains and multiple model scales. Detailed experimental results and settings are provided in Section 4. These results demonstrate the effectiveness of GMTS in improving RLVR performance, as well as its adaptability across different tasks and model scales.

## 1.1 Related Works

Token-level advantage estimation. Chen et al. (2025) leverage the differences between positive and negative advantage-induced token-level signals, together with entropy, to redistribute token-level advantages. Tan and Pan (2025) assign entropy-weighted rewards to emphasize highentropy tokens and down-weight low-entropy ones, mitigating model over-updating. At the same time, Cheng et al. (2026) show that high-entropy tokens correlate with exploration-facilitating behaviors and directly incorporate entropy into advantage estimation. Wang et al. (2025a) focus on highimpact strategy tokens in reasoning and amplifies their learning signals by modifying the advantages, while Ma et al. (2026) use token-level future KL to reassess each token’s contribution.

Objective function design. Wang et al. (2025b) introduce an entropy-aware clipping strategy that applies stricter constraints to low-entropy tokens while relaxing updates for high-entropy ones. Cui et al. (2025) observe that decreasing entropy can lead to entropy collapse and address this issue by dynamically adjusting the entropy regularization strength. Building on this direction, Li et al. (2025) identify critical positions using high-entropy tokens, resample them to construct branched trajectories, and jointly train RLVR on both the original and branched trajectories. In addition, Yao et al. (2025) systematically study diversity in RLVR reasoning and incorporate diversity-aware training into the RLVR objective.

Low importance token filtering. Wang et al. (2025c) find that high-entropy tokens often serve as critical forking thinking tokens, and leverage this observation by selecting only the top 20% entropy tokens for RLVR training, achieving significant performance gains. GMTS also lies in this direction, filtering tokens with low gradient magnitude and using remaining tokens in RLVR training.

## 2 Preliminaries

## 2.1 LLM Formulation and Notations

Let $\mathcal { V } = \{ x ^ { 1 } , x ^ { 2 } , \ldots , x ^ { V } \}$ denote the finite vocabulary of tokens, where V is the size of the vocabulary. Given an input query q, an LLM parameterized by θ defines an autoregressive policy $\pi _ { \theta }$ that generates an output sequence $\pmb { o } \ = \ \left( o _ { 1 } , \ldots , o _ { T } \right)$ token by token. At step t, conditioned on the query and the previously generated tokens $\mathbf { \delta } _ { \pmb { o } _ { < } t }$ , the model predicts a distribution over the vocabulary $\pi _ { \boldsymbol { \theta } } ( \cdot  { | } \ J ,  { \boldsymbol { o } } _ { < t } ) = [ p _ { t , 1 } , p _ { t , 2 } , \dots , p _ { t , V } ]$ , from which the next token $o _ { t }$ is sampled or selected. This process continues until an end-of-sequence token is generated or the maximum generation length is reached. We define the entropy of the predictive distribution at step t as

$$
E _ { t } : = - \sum _ { k = 1 } ^ { V } p _ { t , k } \log p _ { t , k } .
$$

This notation will be used throughout our method.

## 2.2 RLVR Methods

This section briefly reviews two representative RLVR methods: GRPO and DAPO.

GRPO (Shao et al., 2024) is a PPO-style method (Schulman et al., 2017) that estimates advantages from a group of sampled responses, thereby removing the need for a separate value model. Given a query $q \sim \mathcal { D } ,$ , GRPO samples $G$ answers $[ o _ { 1 } , o _ { 2 } , \ldots , o _ { G } ]$ from $\pi _ { \mathrm { o l d } } .$ , with rewards $R = [ R _ { 1 } , R _ { 2 } , \ldots , R _ { G } ]$ to compute the advantage $\begin{array} { r } { A _ { i } = \frac { R _ { i } - \mathrm { m e a n } ( R ) } { \mathrm { s t d } ( R ) } } \end{array}$ . This response-level advantage is assigned to all tokens in $o _ { i } , \mathrm { i . e . , } A _ { i , t } = A _ { i }$ . The optimization objective of GRPO is:

$$
\operatorname* { m a x } _ { \theta } \mathbb { E } _ { q } \left[ \frac { 1 } { G } \sum _ { i = 1 } ^ { G } \frac { 1 } { | \pmb { \mathscr { o } } _ { i } | } \sum _ { t = 1 } ^ { | \pmb { \mathscr { o } } _ { i } | } \ell _ { i , t } ( \theta ) \right] ,\tag{1}
$$

where $\ell _ { i , t } ( \theta )$ denotes the token-level PPO-style objective:

$$
\begin{array} { r l } & { \ell _ { i , t } ( \theta ) : = \operatorname* { m i n } \big ( \mathrm { c l i p } ( r _ { i , t } ( \theta ) , 1 - \epsilon _ { 1 } , 1 + \epsilon _ { 2 } ) A _ { i , t } , } \\ & { \qquad r _ { i , t } ( \theta ) A _ { i , t } \big ) - \beta \cdot \mathbb { D } _ { i , t } ^ { K L } ( \theta ) . } \end{array}
$$

In this expression, $\mathbb { D } _ { i , t } ^ { K L } ( \theta )$ denotes the KL divergence between the current and reference predictive distributions $\pi _ { \boldsymbol { \theta } } ( \cdot \mathrm { ~  ~ \vert ~ } \boldsymbol { q } , \boldsymbol { o } _ { < t } )$ and $\pi _ { \mathrm { r e f } } ( \cdot \ |$ $q , o _ { < t } ) , \pi _ { \mathrm { r e f } }$ is the reference policy and $r _ { i , t } ( \theta ) =$ $\pi _ { \boldsymbol { \theta } } ( o _ { i , t } \vert \mathbf { q } , \pmb { o } _ { i , < t } ) / \pi _ { \mathrm { o l d } } ( o _ { i , t } \vert \mathbf { q } , \pmb { o } _ { i , < t } ) . \beta$ is the parameter for the KL penalty term and $\epsilon _ { 1 } , \epsilon _ { 2 }$ are clipping hyperparameters. In practice, following Schulman (2020), we estimate the per-token KL penalty using a single-sample unbiased estimator.

DAPO builds on GRPO with several practical modifications (Yu et al., 2025). Compared with GRPO, DAPO removes the KL penalty from the objective and adopts dynamic sampling to discard uninformative groups in which all responses are either correct or incorrect. It also uses token-level loss averaging and a larger upper clipping threshold $\epsilon _ { 2 }$ to improve training stability.

## 3 Methodology

However, both GRPO and DAPO treat tokens in the same response equally in training objectives by assigning the same response-level advantage to all tokens, thereby ignoring their unequal contributions to policy updates. In this section, we first analyze token-level differences from the perspective of gradient magnitude and connect them to token entropy. We then introduce GMTS, a gradient-magnitudebased token selection method for RLVR training.

![](images/5928b462b9d3b433a02581b5d08187147578841e4843a70ef9fb1a2115b1798f.jpg)

![](images/4344cb4ea7521180992328b699c411ee91dcefee11b5b837aad4a7c6c2645338.jpg)

![](images/8e36d3b08d530a8e3a28b427beb977247c256d4b38313873fc1f659eac891cee.jpg)  
Figure 2: We use DAPO to train Qwen2.5-math-1.5B on MATH-12K dataset under the setting of $G = 1 6$ . At training step 100, we selected two answers with different advantages in one question group, and computed the gradients $\nabla _ { \boldsymbol { \theta } } \ell _ { i , t } ( \boldsymbol { \theta } )$ $\nabla _ { \theta } \log \pi _ { \theta } ( o _ { i , t } )$ , and entropy $E _ { i , t }$ of all the tokens. With these collected data, we plotted entropy against the token-level log-probability gradients under the log scale (left), entropy against the true gradient magnitude under the log scale (middle), and the distributions of true gradient magnitude together with their 20th percentile boundaries (right).

![](images/cc440596aa00e273bbf8902a0d826cda8955ca3834363e9724feaf5cb5f427b8.jpg)

![](images/7f538ebad95018e79dbe56d802ae645ef0bcc6453bfa3075f7c8b7e0db90aed6.jpg)

![](images/6f8ffb7d132322ef0a70837cf8901a56b4f47d7b121cec12c1cb4121182a9715.jpg)  
Figure 3: We select 10 questions from each of MATH-500 and KodCode, and evaluate Qwen2.5-Math and Qwen2.5- Coder at the 1.5B and 7B scales on the math and code benchmarks, respectively. For each question, we generate 16 responses. Within each group of responses to the same question, we compute the Pearson correlation coefficient between log $\left( E _ { i , t } \right)$ and log $\left( \lVert \nabla _ { \theta } \log \pi _ { \theta } ( o _ { i , t } ) \rVert _ { 2 } \right)$ (left), and the Spearman rank correlation coefficient between the GMTS score $| E _ { i , t } \cdot \omega _ { i , t } ( \theta ) |$ and the true gradient magnitude $\| \nabla _ { \theta } \ell _ { i , t } ( \theta ) \| _ { 2 }$ (middle). We further plot the distribution of ε over all 160 generated responses on MATH-500 (right).

Gradient of each token. Following Yang et al. (2025), the gradient of the token-level objective $\ell _ { i , t } ( \theta )$ can be expressed as a scalar coefficient times the score function:

$$
\nabla _ { \boldsymbol { \theta } } \ell _ { i , t } ( \boldsymbol { \theta } ) = \omega _ { i , t } ( \boldsymbol { \theta } ) \cdot \nabla _ { \boldsymbol { \theta } } \log \pi _ { \boldsymbol { \theta } } ( o _ { i , t } ) ,
$$

where we use $\pi _ { \boldsymbol { \theta } } ( o _ { i , t } )$ to represent $\pi _ { \boldsymbol { \theta } } \big ( o _ { i , t } \big | \boldsymbol { q } , o _ { i , < t } \big )$ for ease of exposition. The coefficient $\omega _ { i , t } ( \theta )$ is given by

$$
\begin{array} { r l } & { \omega _ { i , t } ( \theta ) = r _ { i , t } ( \theta ) A _ { i , t } \cdot \mathbb { I } _ { \epsilon _ { 1 } , \epsilon _ { 2 } } ( r _ { i , t } ( \theta ) , A _ { i , t } ) } \\ & { \quad \quad + \beta \frac { \pi _ { \mathrm { r e f } } ( o _ { i , t } ) } { \pi _ { \theta } \left( o _ { i , t } \right) } - \beta . } \end{array}
$$

Here, $\mathbb { I } _ { \epsilon _ { 1 } , \epsilon _ { 2 } } ( \cdot , \cdot )$ denotes the clipping indicator function:

$$
\mathbb { I } _ { \epsilon _ { 1 } , \epsilon _ { 2 } } ( a , b ) = \left\{ \begin{array} { l l } { 0 } & { \mathrm { i f } b > 0 , a > 1 + \epsilon _ { 2 } , } \\ { 0 } & { \mathrm { i f } b < 0 , a < 1 - \epsilon _ { 1 } , } \\ { 1 } & { \mathrm { o t h e r w i s e } . } \end{array} \right.
$$

The decomposition of $\nabla _ { \boldsymbol { \theta } } \ell _ { i , t } ( \boldsymbol { \theta } )$ shows that $\omega _ { i , t } ( \theta )$ collects the effective token-level learning signals in RLVR, including the advantage value, clipping, and KL regularization terms. Therefore, it directly scales the policy gradient at each token and determines how strongly that token contributes to the overall parameter update. In DAPO, the KL term is omitted, so $\omega _ { i , t } ( \theta )$ reduces to $r _ { i , t } ( \theta ) A _ { i , t } \mathbb { I } _ { \epsilon _ { 1 } , \epsilon _ { 2 } } ( r _ { i , t } ( \theta ) , A _ { i , t } )$ When $\pi _ { \theta }$ remains close to $\pi _ { \mathrm { o l d } }$ and the clipping constraint is inactive, we have $r _ { i , t } ( { \boldsymbol { \theta } } ) \approx 1$ and hence $\omega _ { i , t } ( \theta )$ is approximately $A _ { i , t }$

Token gradient magnitude and entropy. As illustrated in the left panel of Figure 2, the magnitude of the token-level log-probability gradient $\nabla _ { \theta }$ log $\pi _ { \boldsymbol { \theta } } \big ( o _ { i , t } | \boldsymbol { q } , \boldsymbol { o } _ { < t } \big )$ exhibits an approximately linear relationship with token entropy when both quantities are plotted on a logarithmic scale. To better understand this phenomenon, we analyze the gradient structure in detail. Consider a query input $\pmb q$ and an output sequence $^ { o , }$ and focus on the t-th generated token $o _ { t }$ . Let $\boldsymbol { z } _ { t } = [ z _ { t , 1 } , z _ { t , 2 } , \ldots , z _ { t , V } ]$ denote the logits produced by the final layer of the LLM at time step t, and let $\pi _ { \boldsymbol { \theta } } ( \cdot \ | \ q , o _ { < t } ) =$ Softmax $( z _ { t } ) : = [ p _ { t , 1 } , p _ { t , 2 } , . . . , p _ { t , V } ]$ be the resulting predictive distribution over the vocabulary.

Proposition 3.1. Let $G _ { t }$ be the ℓ<sub>2</sub>-norm of the gradient of the log-probability log $\pi _ { \boldsymbol { \theta } } \big ( o _ { t } | \boldsymbol { q } , \boldsymbol { o } _ { < t } \big )$ with respect to the logits $z _ { t } , E _ { t }$ is the entropy ofdistribution $\pi _ { \boldsymbol { \theta } } ( \cdot | \boldsymbol { q } , \boldsymbol { o } _ { < t } )$ . Let $1 - \pi _ { \theta } ( o _ { t } | q , o _ { < t } ) = \varepsilon ,$ then we have:

$$
\operatorname* { l i m } _ { \varepsilon \to 0 } { \frac { \log G _ { t } } { \log E _ { t } } } = 1 .
$$

The detailed derivation for Theorem 3.1 is provided in Section A.2. It establishes a direct connection between token entropy and the magnitude of the log-probability gradient with respect to the logits. As shown by the distribution of $\varepsilon$ on the right panel of Figure 3, most tokens have ε values close to 0. This provides a theoretical explanation for the linearity observed in Figure 2, as well as its slope being close to 1. Furthermore, let $\textstyle { \frac { \partial z _ { t } } { \partial \theta } }$ denote the Jacobian of the logits with respect to the model parameters. By the chain rule $\nabla _ { \theta }$ log $\pi _ { \boldsymbol { \theta } } ( o _ { t } \mid$ $\begin{array} { r } { \pmb q , \pmb o < { t } ) = \frac { \partial \pmb z _ { t } } { \partial \theta } \cdot \nabla _ { \pmb z _ { i } } } \end{array}$ <sub>t</sub> log $\pi _ { \boldsymbol { \theta } } ( o _ { t } \mid \boldsymbol { q } , \boldsymbol { o } _ { < t } )$ , which implies that tokens with higher entropy tend to induce larger gradients with respect to the model parameters. This analysis provides a principled explanation for the empirical observation that high-entropy tokens may contribute more significantly to parameter updates in RLVR training (Wang et al., 2025c).

Gradient magnitude token selection. However, the coefficients $\omega _ { i , t } ( \theta )$ in the token gradients can exhibit substantial variability and are particularly influenced by the clipping constraints and the advantage values derived from answer-level reward signals. As shown in the middle and right panels of Figure 2, tokens belonging to answers with different advantages display notable differences in gradient magnitudes, along with distinct percentile distributions. These observations suggest that entropy alone is insufficient to reflect a token’s contribution in RLVR training. To address this limitation, we propose the GMTS method, which assesses token importance directly from the gradient magnitude. Since directly computing the true gradient norm is computationally intractable during RL training, we approximate $\nabla _ { \theta }$ log $\tau _ { \theta } ( o _ { i , t } | \pmb q , \pmb { o } _ { i , < t } )$ using token entropy. And we define the GMTS score of token $o _ { i , t }$ as:

$$
\delta _ { i , t } = | E _ { i , t } \cdot \omega _ { i , t } ( \boldsymbol { \theta } ) | ,
$$

which serves as a tractable measure of token importance. Based on that, the training objective of GMTS is as follows:

$$
\operatorname* { m a x } _ { \theta } \mathbb { E } _ { q } \left[ \frac { 1 } { S _ { \rho } } \sum _ { i = 1 } ^ { G } \sum _ { t = 1 } ^ { | o _ { i } | } \mathbb { I } [ \delta _ { i , t } \geq \tau _ { \rho } ] \cdot \ell _ { i , t } ( \theta ) \right] .\tag{2}
$$

where $\mathbb { I } ( \cdot )$ is the indicator function that evaluates to 1 if the condition inside holds and 0 otherwise. $\rho$ is a predefined ratio specifying the top proportion to be selected, and $\tau _ { \rho }$ is the corresponding threshold such that only tokens with $\delta _ { i , t } ( \theta ) \geq \tau _ { \rho }$ , comprising the top-ρ fraction of all tokens in the batch, are used to compute the gradient, $S _ { \rho }$ denotes the number of selected tokens in the group. Thus the loss is averaged only over the selected tokens.

From a computational perspective, calculating $\omega _ { i , t } ( \theta )$ involves the advantage signal $A _ { i , t }$ and the predicted probability from $\pi _ { \boldsymbol { \theta } } .$ π<sub>old</sub> and $\pi _ { \mathrm { r e f } }$ for each token. These quantities are already available during standard RLVR training. Therefore, the additional computational overhead introduced by GMTS is minimal. Moreover, GMTS can be readily integrated into other RLVR training frameworks, with the only difference being the specific formulation of the coefficient $\omega _ { i , t } ( \theta )$ . By leveraging only a small subset of influential tokens for RLVR updates, GMTS offers the potential for more efficient RLVR training.

Comparison with ETS. Both ETS and GMTS retain only the top-ρ fraction of tokens for policy updates, but they differ in how token importance is measured. ETS ranks tokens solely by entropy, whereas GMTS scores each token by $\delta _ { i , t } ,$ , which incorporate the effective scalar coefficient in the token gradient. This difference matters in three aspects: (1) Tokens with similar entropy can have very different effects when they belong to responses with different advantages. As shown in the middle panels of Figure 2, the same level of entropy may correspond to distinct gradient magnitudes due to different response-level reward signals. (2) PPOstyle clipping can suppress the policy-gradient term when the probability ratio falls outside the clipping range. In such cases, ETS may still select high-entropy tokens whose gradients are inactive or largely reduced, while GMTS downweights them through the clipping indicator in $\omega _ { i , t } ( \theta )$ . (3) When KL regularization is used, the effective gradient is also affected by the reference-policy correction term, which is not reflected by entropy alone. Thus, GMTS prioritizes tokens that contribute more directly to the actual RLVR update.

## 4 Numerical Experiments

## 4.1 Training and Evaluation Settings

We implement and evaluate GMTS and ETS within the verl framework<sup>2</sup> across GRPO and DAPO. Furthermore, we provide a standalone implementation<sup>3</sup> that not only gives a clear understanding of GMTS but also supports training under limited GPUs. In our experiments, unless otherwise specified, both ETS and GMTS are trained using the Top 20% of tokens, consistent with the selection ratio recommended by ETS (Wang et al., 2025c).

Training Details. We evaluate GMTS across three distinct domains: MATH, CODE, and COM-MONSENSE (CS). Primary results are reported for the 1.5B and 7B model variants, with additional experiments conducted on the 8B model specifically for MATH domain. We use the following training sets: For MATH domain, we utilize the MATH-12K (Lightman et al., 2023) dataset (1.5B and 7B) and DAPO-MATH-17K (Wang et al., 2025c) (8B). For CODE domain, we utilize the KodCode (Xu et al., 2025). For CS domain, we utilize the CS-QA (Talmor et al., 2019). Other details for training on these tasks are all summarized in Section A.3.

Evaluation Details. For MATH domain, we evaluate the 1.5B and 7B models on five benchmarks: AIME2024, AMC23, MATH-500, Minerva, and OlympiadBench. For the 8B model, we additionally report results on AIME2025, since it is particularly challenging for small models (1.5B and 7B). For CODE domain, we utilize standard benchmarks widely adopted in the Qwen3-Coder evaluation <sup>4</sup>: LiveCodeBench (202407 – 202411), MBPP, HumanEval, and Bigcode-Bench. For CS domain, we utilize the test of CS-QA and more general CS-QA2 (Talmor et al., 2022). For MATH and CS domains, we generate 16 candidate answers for each question and report the average accuracy as average@16. All generations are performed with the temperature of $T = 1 . 0$ . For CODE domain, we employ the greedy strategy for evaluation.

## 4.2 Main Results

In Figure 4, we illustrate the global performance of ETS and GMTS across different domains. Next, we show more details of these results on each domain.

## 4.2.1 Results on MATH Domain

We implement ETS/GMTS on GRPO/DAPO, and apply them to the Qwen2.5-math-1.5B and 7B model. All the detailed results are in Tables 1 and 2.

Tables 1 and 2 show that GMTS Top (20%) consistently outperforms ETS across both DAPO and GRPO backbones (1.5B-DAPO: +1.55, 1.5B-GRPO: +1.30, 7B-DAPO: +1.33, 7B-GRPO: +3.41). To evaluate the performance of GMTS on larger reasoning models, we extended our experiments to Qwen3-8B, using results from Wang et al. (2025c) as a baseline. As shown in Table 3, GMTS consistently delivers notable improvements over ETS across a wide range of math benchmarks, with substantial improvements in challenging tasks such as AIME2024 (+5.21) and AIME2025 (+3.75). With an overall average gain of 1.85%, these results demonstrate the generality of GMTS, suggesting it has the potential to scale effectively to larger and more complex reasoning models.

## 4.2.2 Results on CODE and CS Domains

We implement ETS/GMTS on GRPO/DAPO, for code domain, we apply them to the Qwen2.5-coder-1.5B and 7B model. For commonsense domain, we apply them to the Qwen2.5-1.5B-base and 7B model. All the detailed results are in Tables 4 and 5.

For HumanEval and MBPP, which consist of Base and Plus subsets, the table reports the average accuracy across both. Similarly, for BigCode-Bench, the reported values represent the mean accuracy across the Full and Hard subsets. As shown in Tables 4 and 5, GMTS achieves consistent performance gains within the code domain (1.5B-DAPO: +1.87, 1.5B-GRPO: +1.90, 7B-DAPO: +0.74, 7B-GRPO: +0.69). Furthermore, in the commonsense domain (1.5B-DAPO: +0.87, 1.5B-GRPO: +0.67, 7B-DAPO: +1.53, 7B-GRPO: +1.26). These results validate the generalization of the GMTS method across diverse reasoning tasks.

## 4.3 Ablation Study

Linearity and monotonicity consistency. We evaluated the linear and monotonic behavior of GMTS using Qwen2.5-math and Qwen2.5-coder at both the 1.5B and 7B scales. As shown in Figure 3 right and middle panels, the linear relationship consistently holds across domains and model scales, indicating that it is not an isolated empirical artifact. Moreover, GMTS exhibits strong monotonicity, suggesting that gradient-magnitude-based selection preserves the ordering of token contribu-

![](images/bf56fed27d308b154555147e0d4365585d1e9acd4f3ca7f289c8d1b3247d598f.jpg)  
Figure 4: Overall performance comparison. GMTS consistently outperforms GRPO and DAPO baselines as well as the ETS method across multiple model scales (1.5B and 7B).

Table 1: Evaluation on 5 math-reasoning benchmarks (ignore AIME2025) for ETS/GMTS with DAPO/GRPO on Qwen2.5-math-1.5B under average@16 accuracy (%).
<table><tr><td rowspan="2">Qwen2.5-math-1.5B</td><td>AIME2024</td><td>AMC23</td><td>MATH-500</td><td>Minerva</td><td>OlympiadBench</td><td>Avg.</td></tr><tr><td>2.70</td><td>18.75</td><td>23.40</td><td>6.20</td><td>14.10</td><td>13.03</td></tr><tr><td>DAPO</td><td>14.17</td><td>50.62</td><td>73.85</td><td>29.47</td><td>37.17</td><td>41.06</td></tr><tr><td>+ ETS</td><td>14.17</td><td>48.44</td><td>72.25</td><td>29.01</td><td>36.16</td><td>40.01</td></tr><tr><td>+ GMTS</td><td>14.58</td><td>50.63</td><td>73.83</td><td>30.51</td><td>38.23</td><td>41.56</td></tr><tr><td>GRPO</td><td>10.63</td><td>48.12</td><td>71.59</td><td>28.81</td><td>34.90</td><td>38.81</td></tr><tr><td>+ ETS</td><td>12.71</td><td>48.89</td><td>72.39</td><td>28.41</td><td>35.54</td><td>39.59</td></tr><tr><td>+ GMTS</td><td>11.88</td><td>53.33</td><td>73.06</td><td>30.11</td><td>36.09</td><td>40.89</td></tr></table>

Table 2: Evaluation on 5 math-reasoning benchmarks (ignore AIME2025) for ETS/GMTS with DAPO/GRPO on Qwen2.5-math-7B under average@16 accuracy (%).
<table><tr><td rowspan="2">Qwen2.5-math-7B</td><td>AIME2024</td><td>AMC23</td><td>MATH-500</td><td>Minerva</td><td>OlympiadBench</td><td>Avg.</td></tr><tr><td>7.29</td><td>21.09</td><td>34.45</td><td>6.09</td><td>10.92</td><td>15.97</td></tr><tr><td>DAPO</td><td>18.12</td><td>62.50</td><td>80.91</td><td>36.49</td><td>44.35</td><td>48.47</td></tr><tr><td>+ ETS</td><td>20.00</td><td>63.75</td><td>80.04</td><td>36.72</td><td>43.56</td><td>48.81</td></tr><tr><td>+ GMTS</td><td>25.00</td><td>65.16</td><td>80.40</td><td>37.39</td><td>42.75</td><td>50.14</td></tr><tr><td>GRPO</td><td>27.08</td><td>62.34</td><td>79.76</td><td>35.18</td><td>42.87</td><td>49.45</td></tr><tr><td>+ETS</td><td>19.17</td><td>60.67</td><td>77.72</td><td>33.65</td><td>40.96</td><td>46.43</td></tr><tr><td>+ GMTS</td><td>23.33</td><td>64.83</td><td>81.13</td><td>36.32</td><td>43.60</td><td>49.84</td></tr></table>

Table 3: Evaluation on 6 math-reasoning benchmarks for ETS/GMTS with DAPO on Qwen3-8B under average@16 accuracy (%).
<table><tr><td rowspan="2"></td><td>AIME2024</td><td>AIME2025</td><td>AMC23</td><td>MATH-500</td><td>Minerva</td><td>OlympiadBench</td><td>Avg.</td></tr><tr><td>Qwen3-8B 3.75</td><td>6.04</td><td>33.91</td><td>62.99</td><td>22.29</td><td>23.66</td><td>25.44</td></tr><tr><td>DAPO</td><td>33.33</td><td>25.42</td><td>77.81</td><td>89.24</td><td>39.77</td><td>56.67</td><td>53.71</td></tr><tr><td>+ ETS</td><td>34.58</td><td>26.25</td><td>77.19</td><td>89.70</td><td>40.26</td><td>57.43</td><td>54.23</td></tr><tr><td>+ GMTS</td><td>39.79</td><td>30.00</td><td>79.38</td><td>91.14</td><td>42.42</td><td>53.76</td><td>56.08</td></tr></table>

tions.

Bottom selection. To examine the role of low gradient magnitude tokens, we follow Wang et al. (2025c) to evaluate DAPO with bottom selection on Qwen2.5-math-1.5B. As shown in Figure 5 left and Table 6, GMTS consistently performs worse than ETS, with average drops of -1.08 and -1.28 at the

Table 4: Evaluation on 4 code-reasoning and 2 CS benchmarks for ETS/GMTS with DAPO/GRPO on Qwen2.5- coder-1.5B under greedy accuracy (%) (left) and Qwen2.5-base-1.5B under average@16 accuracy (%) (right).
<table><tr><td rowspan="2">1.5B Base Model</td><td>LiveCodeBench</td><td>HumanEval</td><td>MBPP</td><td>Bigcode-Bench</td><td>Avg.</td><td>CS-QA</td><td>CS-QA2</td><td>Avg.</td></tr><tr><td>1.33</td><td>22.04</td><td>12.21</td><td>4.20</td><td>9.95</td><td>10.26</td><td>4.14</td><td>7.20</td></tr><tr><td>DAPO</td><td>11.02</td><td>72.50</td><td>73.14</td><td>26.31</td><td>45.74</td><td>77.21</td><td>52.73</td><td>64.97</td></tr><tr><td>+ ETS</td><td>9.00</td><td>73.51</td><td>73.12</td><td>23.22</td><td>44.71</td><td>78.09</td><td>52.82</td><td>65.46</td></tr><tr><td>+ GMTS</td><td>11.02</td><td>74.72</td><td>73.49</td><td>27.06</td><td>46.58</td><td>79.15</td><td>53.51</td><td>66.33</td></tr><tr><td>GRPO</td><td>10.31</td><td>72.91</td><td>74.14</td><td>25.10</td><td>45.62</td><td>77.43</td><td>51.07</td><td>64.25</td></tr><tr><td>+ ETS</td><td>9.00</td><td>72.92</td><td>71.34</td><td>25.71</td><td>44.74</td><td>79.82</td><td>50.92</td><td>65.37</td></tr><tr><td>+ GMTS</td><td>11.02</td><td>75.03</td><td>74.21</td><td>26.33</td><td>46.64</td><td>79.95</td><td>52.16</td><td>66.04</td></tr></table>

Table 5: Evaluation on 4 code-reasoning and 2 CS benchmarks for ETS/GMTS with DAPO/GRPO on Qwen2.5- coder-7B under greedy accuracy (%) (left) and Qwen2.5-base-7B under average@16 accuracy (%) (right).
<table><tr><td rowspan="2">7B Base Model</td><td>LiveCodeBench</td><td>HumanEval</td><td>MBPP</td><td>Bigcode-Bench</td><td>Avg.</td><td>CS-QA</td><td>CS-QA2</td><td>Avg.</td></tr><tr><td>3.20</td><td>36.33</td><td>39.92</td><td>13.50</td><td>23.24</td><td>31.56</td><td>14.85</td><td>23.21</td></tr><tr><td>DAPO</td><td>19.94</td><td>85.10</td><td>79.02</td><td>36.84</td><td>55.20</td><td>84.70</td><td>66.57</td><td>75.64</td></tr><tr><td>+ ETS</td><td>19.06</td><td>84.39</td><td>77.18</td><td>38.42</td><td>54.76</td><td>83.44</td><td>66.09</td><td>74.77</td></tr><tr><td>+ GMTS</td><td>20.13</td><td>85.62</td><td>79.89</td><td>36.37</td><td>55.50</td><td>85.65</td><td>66.95</td><td>76.30</td></tr><tr><td>GRPO</td><td>20.00</td><td>84.21</td><td>76.71</td><td>36.31</td><td>54.31</td><td>84.00</td><td>62.39</td><td>73.20</td></tr><tr><td>+ ETS</td><td>17.31</td><td>85.46</td><td>77.02</td><td>37.98</td><td>54.44</td><td>84.17</td><td>63.82</td><td>74.00</td></tr><tr><td>+ GMTS</td><td>20.60</td><td>85.91</td><td>77.43</td><td>36.58</td><td>55.13</td><td>84.60</td><td>65.92</td><td>75.26</td></tr></table>

![](images/9f2b9e49504e84a0caa6ed3318f72813f1799af424b7cbd36e481bb6048b8fb4.jpg)

![](images/816563c7b04ff36c3361cb3b38ccded8cce232a271fad50419b1f73495105428.jpg)  
Figure 5: The average@16 performance of DAPO, DAPO + ETS Bottom (80%, 90%) and DAPO + GMTS Bottom (80%, 90%) on Qwen2.5-math-1.5B (left) and the average@16 overall performance of varying selected ratios for DAPO, DAPO + ETS Top, and DAPO + GMTS Top on Qwen2.5-math-1.5B (right).

80% and 90% ratios, respectively. This suggests that low gradient magnitude tokens contribute less to RLVR training, whereas low-entropy tokens may still contain useful signals.

Sensitivity to selection ratios. To evaluate the impact of selection intensity, we evaluate both DAPO and GRPO on Qwen2.5-math-1.5B with ratios 0.1, 0.2, 0.5, 0.7, and 0.9, with the results reported in Table 7 and Figure 5 right. It shows that GMTS consistently outperforms ETS in nine out of ten configurations. These findings suggest that the advantages of GMTS are not confined to a specific regime but extend across a broad spectrum of training conditions, providing a more reliable selection mechanism than ETS.

## 5 Conclusion

In this work, we propose GMTS for improving RLVR in LLM reasoning. Although gradient magnitude is correlated with entropy within one answer, entropy alone cannot fully capture token importance across different answers, where the contribution of each token to policy update can vary substantially. GMTS addresses this limitation by leveraging a gradient-magnitude proxy for token selection. Empirically, GMTS achieves consistent improvements over ETS and remains robust across multiple selection ratios, model scales, and evaluation domains. These results provide empirical evidence supporting GMTS as an effective and efficient choice for RLVR.

## Limitations

Despite the consistent performance gains of GMTS in RLVR, several limitations remain and point to directions for future research.

First, although extensive empirical evaluations support the effectiveness of GMTS, its theoretical foundation remains only partially understood. In this work, we provide preliminary evidence on the relationship between token entropy and gradient magnitude, but a rigorous theoretical justification for why GMTS outperforms ETS, along with a deeper analytical characterisation of its behaviour, remains an important direction for future work.

Second, due to computational constraints, we have not yet evaluated GMTS on larger models, such as 14B and 32B models. While our empirical results across different reasoning domains and model scales suggest that GMTS has the potential to extend to larger-scale models, a more comprehensive evaluation on such models is left for future work.

## Acknowledgments

We thank the SJTU supercomputer center for providing the computing services and the anonymous reviewers very much for their careful reading and valuable comments. This work was supported by the National Natural Science Foundation of China (Grant No. 125B2026) and the Natural Science Foundation of Chongqing, China (No. CSTB2023NSCQ-LZX0054).

## References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, and et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

Tiwei Bie, Maosong Cao, Xiang Cao, Bingsen Chen, Fuyuan Chen, Kun Chen, Lun Du, Daozhuo Feng, Haibo Feng, Mingliang Gong, and et al. 2026. Llada2. 1: Speeding up text diffusion via token editing. arXiv preprint arXiv:2602.08676.

Minghan Chen, Guikun Chen, Wenguan Wang, and Yi Yang. 2025. Seed-grpo: Semantic entropy enhanced grpo for uncertainty-aware policy optimization. arXiv preprint arXiv:2505.12346.

Daixuan Cheng, Shaohan Huang, Xuekai Zhu, Bo Dai, Xin Zhao, Zhenliang Zhang, and Furu Wei. 2026. Reasoning with exploration: An entropy perspective. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 30377–30385.

Ganqu Cui, Yuchen Zhang, Jiacheng Chen, Lifan Yuan, Zhi Wang, Yuxin Zuo, Haozhan Li, Yuchen Fan, Huayu Chen, Weize Chen, and et al. 2025. The entropy mechanism of reinforcement learning for reasoning language models. arXiv preprint arXiv:2505.22617.

Harrison Lee, Samrat Phatale, Hassan Mansoor, Thomas Mesnard, Johan Ferret, Kellie Ren Lu, Colton Bishop, Ethan Hall, Victor Carbune, Abhinav Rastogi, and et al. 2024. Rlaif vs. rlhf: Scaling reinforcement learning from human feedback with ai feedback. In International Conference on Machine Learning, pages 26874–26901. PMLR.

Qingbin Li, Rongkun Xue, Jie Wang, Ming Zhou, Zhi Li, Xiaofeng Ji, Yongqi Wang, Miao Liu, Zheming Yang, Minghui Qiu, and et al. 2025. Cure: Criticaltoken-guided re-concatenation for entropy-collapse prevention. arXiv preprint arXiv:2508.11016.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. 2023. Let’s verify step by step. In The Twelfth International Conference on Learning Representations.

Shih-Yang Liu, Xin Dong, Ximing Lu, Shizhe Diao, Peter Belcak, Mingjie Liu, Min-Hung Chen, Hongxu Yin, Yu-Chiang Frank Wang, Kwang-Ting Cheng, and et al. 2026. Gdpo: Group reward-decoupled normalization policy optimization for multi-reward rl optimization. arXiv preprint arXiv:2601.05242.

Outongyi Lv, Yanzhao Zheng, Yuanwei Zhang, Zhenghao Huang, Xingjun Wang, Baohua Dong, Hangcheng Zhu, and Yingda Chen. 2026. Which tokens matter? adaptive token selection for rlvr with the relative surprisal index. arXiv preprint arXiv:2606.31575.

Chiyu Ma, Shuo Yang, Kexin Huang, Jinda Lu, Haoming Meng, Shangshang Wang, Bolin Ding, Soroush Vosoughi, Guoyin Wang, and Jingren Zhou. 2026. Fipo: Eliciting deep reasoning with futurekl influenced policy optimization. arXiv preprint arXiv:2603.19835.

Haoming Meng, Kexin Huang, Shaohang Wei, Chiyu Ma, Shuo Yang, Xue Wang, Guoyin Wang, Bolin Ding, and Jingren Zhou. 2026. Sparse but critical: A token-level analysis of distributional shifts in rlvr fine-tuning of llms. In The Fourteenth International Conference on Learning Representations.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, and et al. 2022. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744.

John Schulman. 2020. Approximating kl divergence. John Schulman’s Homepage.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, and et al. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300.

Haozhan Shen, Peng Liu, Jingcheng Li, Chunxin Fang, Yibo Ma, Jiajia Liao, Qiaoli Shen, Zilun Zhang, Kangjia Zhao, Qianqian Zhang, and et al. 2025. Vlmr1: A stable and generalizable r1-style large visionlanguage model. arXiv preprint arXiv:2504.07615.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. 2023. Reflexion: Language agents with verbal reinforcement learning. Advances in Neural Information Processing Systems, 36:8634–8652.

Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. 2019. Commonsenseqa: A question answering challenge targeting commonsense knowledge. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4149–4158.

Alon Talmor, Ori Yoran, Ronan Le Bras, Chandra Bhagavatula, Yoav Goldberg, Yejin Choi, and Jonathan Berant. 2022. Commonsenseqa 2.0: Exposing the limits of ai through gamification. arXiv preprint arXiv:2201.05320.

Hongze Tan and Jianfei Pan. 2025. Gtpo and grpo-s: Token and sequence-level reward shaping with policy entropy. arXiv preprint arXiv:2508.04349.

Haozhe Wang, Qixin Xu, Che Liu, Junhong Wu, Fangzhen Lin, and Wenhu Chen. 2025a. Emergent hierarchical reasoning in llms through reinforcement learning. arXiv preprint arXiv:2509.03646.

Jiakang Wang, Runze Liu, Fuzheng Zhang, Xiu Li, and Guorui Zhou. 2025b. Stabilizing knowledge, promoting reasoning: Dual-token constraints for rlvr. arXiv preprint arXiv:2507.15778.

Shenzhi Wang, Le Yu, Chang Gao, Chujie Zheng, Shixuan Liu, Rui Lu, Kai Dang, Xionghui Chen, Jianxin Yang, Zhenru Zhang, and et al. 2025c. Beyond the 80/20 rule: High-entropy minority tokens drive effective reinforcement learning for llm reasoning. arXiv preprint arXiv:2506.01939.

Tianhe Wu, Jian Zou, Jie Liang, Lei Zhang, and Kede Ma. 2025. Visualquality-r1: Reasoning-induced image quality assessment via reinforcement learning to rank. arXiv preprint arXiv:2505.14460.

Zhangchen Xu, Yang Liu, Yueqin Yin, Mingyuan Zhou, and Radha Poovendran. 2025. Kodcode: A diverse, challenging, and verifiable synthetic dataset for coding. In Findings of the Association for Computational Linguistics: ACL 2025, pages 6980– 7008.

Zhihe Yang, Xufang Luo, Zilong Wang, Dongqi Han, Zhiyuan He, Dongsheng Li, and Yunjian Xu. 2025. Do not let low-probability tokens over-dominate in rl for llms. arXiv preprint arXiv:2505.12929.

Jian Yao, Ran Cheng, Xingyu Wu, Jibin Wu, and Kay Chen Tan. 2025. Diversity-aware policy optimization for large language model reasoning. arXiv preprint arXiv:2505.23433.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, and et al. 2025. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476.

Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, and et al. 2025. Group sequence policy optimization. arXiv preprint arXiv:2507.18071.

## A Appendix

## A.1 More Experimental Results

We provide additional complete numerical results for Section 4.3 in the main text, with detailed values reported in Tables 6 and 7. The Bottom selection experiment in Table 6 is designed to empirically examine whether the top ranked tokens under GMTS contain more training beneficial information than those selected by ETS. In addition, as shown in Table 7, GMTS clearly outperforms ETS in 9 out of 10 GRPO/DAPO settings, which empirically shows that GMTS generally remains superior to ETS across different top-selection ratios.

## A.2 Derivation of token gradient

Proposition 3.1: Let $G _ { t }$ be the ℓ<sub>2</sub>-norm of the gradient of the log-probability log $\pi _ { \boldsymbol { \theta } } { \big ( } o _ { t } | \boldsymbol { q } , \boldsymbol { o } _ { < t } { \big ) }$ with respect to the logits $z _ { t } , E _ { t }$ is the entropy of distribution $\pi _ { \boldsymbol { \theta } } ( \cdot | \boldsymbol { q } , \boldsymbol { o } _ { < t } )$ . Let $1 - \pi _ { \boldsymbol { \theta } } \big ( o _ { t } | \boldsymbol { q } , \boldsymbol { o } _ { < t } \big ) = \varepsilon$ , then we have:

$$
\operatorname* { l i m } _ { \varepsilon \to 0 } { \frac { \log G _ { t } } { \log E _ { t } } } = 1 .
$$

Proof. Let $p _ { t } = \pi _ { \theta } ( \cdot \mid q , o _ { < t } )$

$$
\begin{array} { c l } { \displaystyle \nabla _ { z } \log ( p _ { t , i } ) = \frac { 1 } { p _ { t , i } } \nabla _ { z } p _ { t , i } } \\ { \displaystyle = \frac { 1 } { p _ { t , i } } \big [ p _ { t , i } ( e _ { i } - p _ { t } ) \big ] } \\ { \displaystyle = e _ { i } - p _ { t } , } \end{array}
$$

Table 6: Evaluation on five math-reasoning benchmarks (ignore AIME2025) for ETS/GMTS with DAPO/GRPO on Qwen2.5-math-1.5B under average@16 accuracy (%).
<table><tr><td rowspan="2">Qwen2.5-math-1.5B</td><td>AIME2024</td><td>AMC23</td><td>MATH-500</td><td>Minerva</td><td>OlympiadBench</td><td>Avg.</td></tr><tr><td>2.70</td><td>18.75</td><td>23.40</td><td>6.20</td><td>14.10</td><td>13.03</td></tr><tr><td>DAPO</td><td>14.17</td><td>50.62</td><td>73.85</td><td>29.47</td><td>37.17</td><td>41.06</td></tr><tr><td>+ ETS Bottom (80%)</td><td>12.22</td><td>47.17</td><td>65.52</td><td>23.77</td><td>30.40</td><td>35.81</td></tr><tr><td>+ ETS Bottom (90%)</td><td>13.11</td><td>49.50</td><td>70.60</td><td>28.73</td><td>34.49</td><td>39.29</td></tr><tr><td>+ GMTS Bottom (80%)</td><td>12.67</td><td>46.67</td><td>63.11</td><td>22.08</td><td>29.14</td><td>34.73</td></tr><tr><td>+ GMTS Bottom (90%)</td><td>12.44</td><td>50.17</td><td>68.35</td><td>26.74</td><td>32.33</td><td>38.01</td></tr></table>

Table 7: Evaluation on five math-reasoning benchmarks for ETS/GMTS with DAPO/GRPO on Qwen2.5-math-1.5B under average@16 accuracy (%) in different selected ratios.
<table><tr><td rowspan="2">Qwen2.5-math-1.5B</td><td>AIME2024</td><td>AMC23</td><td>MATH-500</td><td>Minerva</td><td>OlympiadBench</td><td>Avg.</td></tr><tr><td>2.70</td><td>18.75</td><td>23.40</td><td>6.20</td><td>14.10</td><td>13.03</td></tr><tr><td>DAPO</td><td>14.17</td><td>50.62</td><td>73.85</td><td>29.47</td><td>37.17</td><td>41.06</td></tr><tr><td>+ ETS Top (90%)</td><td>13.96</td><td>51.72</td><td>73.97</td><td>30.36</td><td>24.58</td><td>38.92</td></tr><tr><td>+ GMTS Top (90%)</td><td>11.04</td><td>52.34</td><td>74.20</td><td>30.36</td><td>37.61</td><td>41.11</td></tr><tr><td>+ ETS Top (70%)</td><td>12.08</td><td>48.44</td><td>74.07</td><td>29.26</td><td>36.31</td><td>40.03</td></tr><tr><td> + GMTS Top (70%)</td><td>11.46</td><td>52.50</td><td>74.00</td><td>28.80</td><td>37.40</td><td>40.83</td></tr><tr><td>+ ETS Top (50%)</td><td>12.50</td><td>52.81</td><td>73.22</td><td>29.55</td><td>36.97</td><td>41.01</td></tr><tr><td>+ GMTS Top (50%)</td><td>13.54</td><td>53.44</td><td>73.66</td><td>29.89</td><td>36.50</td><td>41.41</td></tr><tr><td>+ ETS Top (10%)</td><td>12.71</td><td>46.88</td><td>70.03</td><td>27.08</td><td>32.93</td><td>37.93</td></tr><tr><td>+ GMTS Top (10%)</td><td>13.33</td><td>52.19</td><td>73.53</td><td>29.72</td><td>35.68</td><td>40.89</td></tr><tr><td>GRPO</td><td></td><td>48.12</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>10.63</td><td></td><td>71.59</td><td>28.81</td><td>34.90</td><td>38.81</td></tr><tr><td>+ ETS Top (90%) + GMTS Top (90%)</td><td>11.56 15.33</td><td>50.00 51.00</td><td>72.73 72.68</td><td>29.61</td><td>36.07</td><td>40.00</td></tr><tr><td>+ ETS Top (70%)</td><td></td><td></td><td></td><td>30.51</td><td>35.91</td><td>41.09</td></tr><tr><td>+ GMTS Top (70%)</td><td>11.88 12.92</td><td>50.62 52.50</td><td>73.51 74.22</td><td>30.31</td><td>36.61</td><td>40.59</td></tr><tr><td></td><td></td><td></td><td></td><td>30.61</td><td>37.11</td><td>41.47</td></tr><tr><td>+ ETS Top (50%) + GMTS Top (50%)</td><td>12.92 10.63</td><td>50.78 49.38</td><td>73.46 72.95</td><td>29.25</td><td>37.86</td><td>40.85</td></tr><tr><td></td><td></td><td></td><td></td><td>28.75</td><td>36.09</td><td>39.56</td></tr><tr><td>+ ETS Top (10%) + GMTS Top (10%)</td><td>10.63</td><td>45.94</td><td>70.00 72.24</td><td>27.30</td><td>32.97</td><td>37.37</td></tr><tr><td></td><td>15.00</td><td>49.84</td><td></td><td>29.60</td><td>35.65</td><td>40.47</td></tr></table>

where $e _ { i }$ is the unit vector whose i-th entry is 1 and all other entries are 0. Let $\pi _ { \boldsymbol { \theta } } \bigl ( o _ { t } | \boldsymbol { q } , \boldsymbol { o } _ { < t } \bigr ) = p _ { t , i }$ , we can get:

$$
G _ { t } = \sqrt { ( 1 - p _ { t , i } ) ^ { 2 } + \sum _ { j \neq i } p _ { t , j } ^ { 2 } } .
$$

According to $1 - \pi _ { \boldsymbol { \theta } } \big ( o _ { t } | \boldsymbol { q } , \boldsymbol { o } _ { < t } \big ) = \varepsilon$ , then we will have

$$
\begin{array} { r } { \log G _ { t } = \log ( \sqrt { \varepsilon ^ { 2 } \left( 1 + \displaystyle \sum _ { j \neq i } q _ { t , j } ^ { 2 } \right) } ) } \\ { = \log \varepsilon + \log ( \sqrt { 1 + \displaystyle \sum _ { j \neq i } q _ { t , j } ^ { 2 } } ) . } \end{array}
$$

where $\begin{array} { r } { q _ { t , k } = \frac { p _ { t , k } } { \varepsilon } } \end{array}$

For entropy $E _ { t }$ , we have:

$$
\begin{array} { l } { E _ { t } = - \displaystyle \sum _ { j } p _ { t , j } \log p _ { t , j } } \\ { = - p _ { t , i } \log p _ { t , i } - ( 1 - p _ { t , i } ) \log ( 1 - p _ { t , i } ) } \\ { \quad - \left( 1 - p _ { t , i } \right) \displaystyle \sum _ { j \neq i } \frac { p _ { t , j } } { 1 - p _ { t , i } } \log \displaystyle \frac { p _ { t , j } } { 1 - p _ { t , i } } } \end{array}
$$

Notice that $\begin{array} { r } { - \sum _ { j \neq i } \frac { p _ { t , j } } { 1 - p _ { t , i } } \log \frac { p _ { t , j } } { 1 - p _ { t , i } } } \end{array}$ is the entropy of the renormalized residual distribution after removing the i-th probability, we call it $E ( p _ { t } / p _ { t , i } )$ then we have:

$$
E _ { t } = - ( 1 - \varepsilon ) \log ( 1 - \varepsilon ) - \varepsilon \log \varepsilon + \varepsilon E ( p _ { t } / p _ { t , i } )
$$

Using the Taylor expansion log $( 1 - \varepsilon ) = - \varepsilon -$ $\frac { \varepsilon ^ { 2 } } { 2 } + { O } ( \varepsilon ^ { 3 } )$ . We get

$$
\begin{array} { l } { \displaystyle \log E _ { t } } \\ { = \log ( \varepsilon - \varepsilon \log \varepsilon + \varepsilon E ( p _ { t } / p _ { t , i } ) - \frac { \varepsilon ^ { 2 } } { 2 } + O ( \varepsilon ^ { 3 } ) ) } \\ { = \log \varepsilon + \log ( 1 - \log \varepsilon + E ( p _ { t } / p _ { t , i } ) - \displaystyle \frac { \varepsilon } { 2 } + O ( \varepsilon ^ { 2 } ) ) } \end{array}
$$

Thus, we have:

$$
\operatorname* { l i m } _ { \varepsilon \to 0 } { \frac { \log G _ { t } } { \log E _ { t } } }
$$

$\begin{array} { r } { \log \varepsilon + \log ( \sqrt { 1 + \sum _ { j \neq i } q _ { t , j } ^ { 2 } } ) } \end{array}$ lim ε→0 $\begin{array} { r } { \log \varepsilon + \log ( 1 - \log \varepsilon + E ( p _ { t } / p _ { t , i } ) - \frac { \varepsilon } { 2 } + O ( \varepsilon ^ { 2 } ) } \end{array}$ = 1.

## A.3 Training Details

For fairness and reproducibility, we fix the training seed to 0 in all the experiments, which are performed on eight NVIDIA H100 GPUs, each equipped with 80 GB of memory.

## Hyperparameter settings for Qwen2.5-math-1.5B and Qwen2.5-math-7B.

For the Qwen2.5-math-1.5B and Qwen2.5-math-7B models, the training configurations are as follows: the maximum response length is set to 2048 tokens, and the maximum prompt length to 1024 tokens. Each training batch contained 64 questions, with a mini-batch size of 64. For each question, 16 rollouts are sampled with a temperature of 1.0. The clipping parameters $\epsilon _ { 1 }$ and $\epsilon _ { 2 }$ are set to 0.2 and 0.28. The learning rate is fixed at $3 \times 1 0 ^ { - 5 }$ without applying any learning rate warm-up or scheduling.

Hyperparameter settings for Qwen3-8B.

Due to computational constraints, we could not fully adopt the settings of (Wang et al., 2025c), and therefore adjusted the maximum output length. Specifically, we set the maximum response length to 4096 tokens and the maximum prompt length to 1024 tokens. Each training batch contained 512 questions and a mini-batch size of 32, resulting in 16 gradient steps per training batch. For each question, 16 rollouts are sampled with a temperature of 1.0. The clipping parameters $\epsilon _ { 1 }$ and $\epsilon _ { 2 }$ are set to 0.2 and 0.28. The learning rate is fixed at $1 \times 1 0 ^ { - 6 }$ without applying any learning rate warmup or scheduling.

## Hyperparameter settings for Qwen2.5-coder-1.5B and Qwen2.5-coder-7B.

For the Qwen2.5-coder-1.5B and Qwen2.5- coder-7B models, the training configurations are as follows: the maximum response length is set to 32768 tokens, and the maximum prompt length to 2048 tokens. Each training batch contained 512 questions, with a mini-batch size of 32. For each question, 16 rollouts are sampled with a temperature of 1.0. The clipping parameters $\epsilon _ { 1 }$ and $\epsilon _ { 2 }$ are set to 0.2 and 0.28. The learning rate is fixed at $1 \times 1 0 ^ { - 6 }$ without applying any learning rate warmup or scheduling.

## Hyperparameter settings for Qwen2.5-base-1.5B and Qwen2.5-base-7B.

For the Qwen2.5-base-1.5B and Qwen2.5-base-)7B models, the training configurations are as follows: the maximum response length is set to 4096 tokens, and the maximum prompt length to 1024 tokens. Each training batch contained 512 questions, with a mini-batch size of 32. For each question, 16 rollouts are sampled with a temperature of 1.0. The clipping parameters $\epsilon _ { 1 }$ and $\epsilon _ { 2 }$ are set to 0.2 and 0.28. The learning rate is fixed at $1 \times 1 0 ^ { - 6 }$ without applying any learning rate warm-up or scheduling.

## Dialogue template

For the math and CS domains, we adopt the following dialogue template:

```perl
<| im_start |> system \ nPlease reason step
by step , and put your final answer
within \ boxed {}. <| im_end |>\n <| im_start |>
user \n"<| im_end |>\n <| im_start |> assistant
\n
```

For the code domain, we adopt the following dialogue template:

<| im\_start | > system \ nYou are Qwen ,   
created by Alibaba Cloud . You are a   
helpful assistant .<| im\_end |>\n <| im\_start   
|> user \n Please provide a self - contained   
Python script that solves the following

problem :\n { question }\ nYou will use the following starter code to write the solution to the problem $: \lnot \cdots  p \lor$ thon \n{ starter\_code }\n\`\`\`\n Please reason step by step , and put your code answer in a markdown code block .

## A.4 An Example for ETS Top (20%) and GMTS (20%) with DAPO on Qwen3-8B under AIME2025

In DAPO on Qwen3-8B, we selected an example under the ETS Top (20%) and GMTS Top (20%) settings to illustrate that the output of GMTS is not only more concise but also more accurate compared to ETS:

## Question and Ground-Truth

Question: Find the number of ordered pairs $( x , y ) ,$ where both x and y are integers between −100 and 100, inclusive, such that $1 2 x ^ { 2 } - x y - 6 y ^ { 2 } = 0 .$ Ground-Truth:

## ETS Top (20%)

Okay, so I need to find the number of ordered pairs $( x , y )$ where both x and y are integers between −100 and 100 inclusive, and they satisfy the equation

$$
1 2 x ^ { 2 } - x y - 6 y ^ { 2 } = 0 .
$$

Let me think about how to approach this.

First, the equation is quadratic in both x and y. Maybe I can factor it or find some relationship between x and y. Let me try rearranging the equation. Let me write it again:

$$
1 2 x ^ { 2 } - x y - 6 y ^ { 2 } = 0 .
$$

Hmm. Maybe I can factor this quadratic equation. Let me see. If I treat this as a quadratic in $x ,$ then maybe I can factor it or use the quadratic formula. Alternatively, maybe factor it as a quadratic in y. Let me try both approaches.

Let me try treating it as a quadratic in x. ${ \mathrm { S o } } ,$

$$
1 2 x ^ { 2 } - y x - 6 y ^ { 2 } = 0 .
$$

${ \bf { S o , } }$ in the form $a x ^ { 2 } + \underline$ bx $+ \ c = 0 ,$ where $a \ = \ 1 2 ,$ $b = - y .$ , and $c = - 6 y ^ { 2 }$ . Then, using quadratic formula:

$$
x = { \frac { y \pm { \sqrt { y ^ { 2 } - 4 \cdot 1 2 \cdot ( - 6 y ^ { 2 } ) } } } { 2 \cdot 1 2 } } .
$$

Calculating discriminant:

$$
D = y ^ { 2 } - 4 \cdot 1 2 \cdot ( - 6 y ^ { 2 } ) = y ^ { 2 } + 2 8 8 y ^ { 2 } = 2 8 9 y ^ { 2 } .
$$

So

$$
{ \sqrt { D } } = { \sqrt { 2 8 9 y ^ { 2 } } } = 1 7 | y | .
$$

Since y is an integer, |y| is non-negative, so $\sqrt { D }$ is $1 7 | y | .$

Therefore,

$$
x = { \frac { y \pm 1 7 | y | } { 2 4 } } .
$$

Case 1: $y \geq 0 .$ . Then $| y | = y . \mathrm { S o }$

$$
x = { \frac { y \pm 1 7 y } { 2 4 } } .
$$

So two possibilities:

$$
x = { \frac { y + 1 7 y } { 2 4 } } = { \frac { 1 8 y } { 2 4 } } = { \frac { 3 y } { 4 } } ,
$$

or

$$
x = { \frac { y - 1 7 y } { 2 4 } } = { \frac { - 1 6 y } { 2 4 } } = - { \frac { 2 y } { 3 } } .
$$

Case $2 \colon y < 0 .$ Then $| y | = - y .$ So

$$
x = { \frac { y \pm 1 7 ( - y ) } { 2 4 } } .
$$

Let’s compute:

First, with the plus sign:

$$
{ \frac { y + ( - 1 7 y ) } { 2 4 } } = { \frac { - 1 6 y } { 2 4 } } = - { \frac { 2 y } { 3 } } .
$$

With the minus sign:

$$
{ \frac { y - ( - 1 7 y ) } { 2 4 } } = { \frac { y + 1 7 y } { 2 4 } } = { \frac { 1 8 y } { 2 4 } } = { \frac { 3 y } { 4 } } .
$$

So regardless of the sign of $_ { y , }$ the solutions for x are

$$
x = { \frac { 3 y } { 4 } } \quad { \mathrm { o r } } \quad x = - { \frac { 2 y } { 3 } } .
$$

Therefore, for each $y , x$ must be either ${ \frac { 3 y } { 4 } } \ \mathrm { o r } - { \frac { 2 y } { 3 } }$ . But since x has to be an integer, this imposes conditions on y. Therefore, $3 y / 4$ must be integer $\mathrm { o r - } 2 y / 3$ must be integer. Therefore:

$$
3 \mid 2 y .
$$

But since 3 and 4 are coprime, ${ \mathrm { ~ 4 ~ } } | { \mathrm { ~ 3 } } y$ implies $4 \mid y .$ Similarly, 3 | 2y implies ${ \dot { 3 } } \mid y .$ Therefore, for x to be integer:

Either y is divisible by 4, or y is divisible by 3. Therefore, the solutions are:

\- For each y divisible by 4, x = 3y/4 is integer. - For each y divisible by $3 , x = - 2 y / 3$ is integer. If y is divisible by both 3 and 4 $( \mathrm { i . e . , } \mathrm { b y } 1 2 )$ , then both x’s would be integers.

Therefore, the total number of solutions would be the number of $y ^ { \prime } \mathbf { s }$ divisible by 4 plus the number of $y ^ { \prime } \mathbf { s }$ divisible by 3. However, we have to be careful not to double count the cases where y is divisible by both 3 and 4, i.e., by 12. Because those $y ^ { \prime } \mathbf { s }$ would be counted in both sets. Therefore, using inclusion-exclusion, total solutions would be:

$$
N = ( \# \{ y : 4 \mid y \} ) + ( \# \{ y : 3 \mid y \} ) - ( \# \{ y : 1 2 \mid y \} ) .
$$

$$
\lfloor 1 0 0 / 4 \rfloor \stackrel { - } { \times } 2 + 1 = 2 5 \times
$$

$$
2 + 1 = 5 1 .
$$

$$
3 \colon \lfloor 1 0 0 / 3 \rfloor \times
$$

$2 + 1 = 3 3 \times 2 + 1 = 6 7 .$ - Number of y divisible by   
$1 2 \colon \lfloor 1 0 0 / 1 2 \rfloor \times 2 + 1 = 8 \times 2 + 1 = 1 7 .$   
Thus   
N = 51 + 67 − 17 = 101.   
Check special cases:   
$\begin{array} { r } { { \bf { \nabla } } - y = 0 . } \end{array}$ Then the equation is $1 2 x ^ { 2 } = 0 \implies x = 0 .$   
So $( 0 , 0 )$ is one solution. This is correctly counted once   
by the formula.   
- y = 4. Then $x = 3 .$ Solution: $( 3 , 4 )$ works.   
- y = 3. Then x = −2. Solution: $( - 2 , 3 )$ works.   
- y = 12. Then x = 9 or $x = - 8 ,$ both valid.   
All checks out.   
Finally, also check x range: $\phantom { } _ { \mathrm { { s } } } - 1 0 0 \leq x \leq 1 0 0 .$ For   
$\begin{array} { r } { | y | \leq 1 0 0 , x = \frac { 3 y } { 4 } \mathrm { { o r } } x = - \frac { 2 y } { 3 } } \end{array}$ also lies in [−100, 100].   
Verified.   
101 101

## GMTS Top (20%)

Okay, so I need to find the number of ordered pairs $( x , y )$ where both x and y are integers between $- 1 \bar { 0 } 0$ and 100 inclusive, and they satisfy the equation $1 2 x ^ { 2 } - x y -$ $6 { y } ^ { 2 } = 0 .$ . Hmm, let me think about how to approach this. First, maybe I can try to factor the equation or simplify it somehow. Let me write down the equation again:

$$
1 2 x ^ { 2 } - x y - 6 y ^ { 2 } = 0 .
$$

Hmm, quadratic in terms of x and y. Maybe I can treat this as a quadratic equation in x or $y ?$ Let me try treating it as a quadratic in x. Let me rearrange the terms:

$$
1 2 x ^ { 2 } - x y - 6 y ^ { 2 } = 0 .
$$

If I consider this as a quadratic in $x ,$ then it’s of the form $a x ^ { 2 } + b x + c = 0 ,$ , where:

$$
a = 1 2 , \quad b = - y , \quad c = - 6 y ^ { 2 } .
$$

Then, using the quadratic formula,

$$
x = { \frac { - b \pm { \sqrt { b ^ { 2 } - 4 a c } } } { 2 a } } = { \frac { y \pm { \sqrt { y ^ { 2 } - 4 \cdot 1 2 \cdot ( - 6 y ^ { 2 } ) } } } { 2 4 } } .
$$

Simplify inside the square root:

$$
y ^ { 2 } - 4 \cdot 1 2 \cdot ( - 6 y ^ { 2 } ) = y ^ { 2 } + 2 8 8 y ^ { 2 } = 2 8 9 y ^ { 2 } .
$$

So

$$
{ \sqrt { 2 8 9 y ^ { 2 } } } = 1 7 | y | .
$$

Therefore,

$$
x = { \frac { y \pm 1 7 | y | } { 2 4 } } .
$$

Case 1: $y \geq 0 .$ . Then $| y | = y .$ Therefore:

$$
x = { \frac { y \pm 1 7 y } { 2 4 } } .
$$

So two possibilities:

$$
x = \frac { 1 8 y } { 2 4 } = \frac { 3 y } { 4 } , \quad \mathrm { o r } \quad x = \frac { - 1 6 y } { 2 4 } = - \frac { 2 y } { 3 } .
$$

Case 2: $y < 0 .$ Then $| y | = - y .$ Therefore:

$$
x = { \frac { y \pm 1 7 ( - y ) } { 2 4 } } .
$$

So: First possibility: $y + ( - 1 7 y ) = - 1 6 y , \mathrm { h e n c e } x = - { \frac { 2 y } { 3 } }$ $\begin{array} { r } { x = - \frac { 2 y } { 3 } . } \\ { \mathtt { e } x = \frac { 3 y } { 4 } . } \end{array}$ Second possibility: $y - ( - 1 7 y ) = 1 8 y ,$ henc

So regardless of the sign of y, the solutions for x are:

$$
x = { \frac { 3 y } { 4 } } \quad { \mathrm { o r } } \quad x = - { \frac { 2 y } { 3 } } .
$$

Therefore, the solutions to the equation are all pairs $( x , y )$ with $x , y \in [ - 1 0 0$ , 100] such that either $x \overset { \cdot } { = } 3 y / 4$ or $x = - 2 y / 3 ,$ with integrality conditions.

Case 1: $x = 3 y / 4 .$   
For x to be integer, y must be divisible by 4. Let $y = 4 k ,$   
then $x = 3 k$ . Range: $: - 1 0 0 \leq 4 k \leq 1 0 0 \implies - 2 5 \leq$   
$k \leq 2 5 .$ . That gives $2 5 - ( - 2 5 ) + 1 = 5 1$ solutions.   
Case 2: x = −2y/3.   
For x to be integer, y must be divisible by 3. Let y = 3m,   
then $x = - 2 m .$ . Range: $- 1 0 0 \leq 3 m \leq$ 100 =⇒   
$- 3 3 \leq m \leq 3 3$ . That gives $3 3 - ( - 3 3 ) + 1 = 6 7$   
solutions.

## Overlap check:

Suppose (x, y) satisfies both forms: $( x , y ) = ( 3 k , 4 k )$ and $( x , y ) = ( - 2 m , 3 m )$ . Then $3 k = - 2 m$ and 4k = 3m. Solving gives $k = 0 , m = 0 ,$ hence (0, 0). So there is exactly one overlap.

## Final count:

$$
5 1 + 6 7 - 1 = 1 1 7 .
$$