# Hyperparameter Scaling Laws Across MoE Sparsity

Changxin Tian, Kunlong Chen, Jia Liu, Ziqi Liu, Zhiqiang Zhang<sup>†</sup>, Jun Zhou<sup>†</sup>

Ling Team, Ant Group

<sup>†</sup>Corresponding author

Mixture-of-Experts (MoE) models expand model capacity without a proportional increase in training compute, but increasing sparsity makes reliable hyperparameter transfer challenging. In this work, we show that conventional hyperparameter scaling laws are insufficient for ultra-sparse MoEs: the optimal learning rate and batch size vary with activation ratio, and these shifts cannot be explained by either total or activated parameter count alone. To characterize this dependence, we conduct 1,800 pre-training runs spanning six activatedparameter scales and models with up to 6B total non-embedding parameters, processing approximately 20 trillion tokens at a cost of 200,000 equivalent H800 GPU-hours. Our results reconcile conflicting findings in prior work by revealing two scaling regimes. At fixed sparsity, the optimal batch size follows a power-law relationship with training tokens D, whereas the optimal learning rate scales with training compute C and remains robust to the allocation between model size and data. Across sparsity levels, the activation ratio A enters both relationships as an additional multiplicative power-law factor. These observations lead to unified hyperparameter scaling laws that transfer across MoE sparsity levels. Large-scale evaluation shows that the scaling form outperforms alternative functional forms. On a held-out ultra-sparse MoE with 12B total parameters and only 1/64 of its experts activated, the predicted hyperparameters remain close to the observed optima, supporting joint extrapolation across model scale and sparsity. Further experiments demonstrate transfer across expert granularities and isolate the effect of activation ratio from that of total expert count.

Date: September 9, 2026 Correspondence: {tianchangxin.tcx,lingyao.zzq,jun.zhoujun}@antgroup.com

![](images/2ff097b0809b3bd5e55b5e7693e2d388ef6b8d4b6f2cf6b4cdf8b3a850a59542.jpg)

![](images/d86ed4eacf34a7cfb8d15d3367b2658bb968f12ba0404c70704a4bea62007ab6.jpg)  
(a)

![](images/69605ee7943bba2d8d68d9e658bf0eb48e1c6f8fcbaf800ed2a8347be60c10a9.jpg)  
(b)

## 1 Introduction

Mixture-of-Experts (MoE) models (Shazeer et al., 2017; Team et al., 2026; Xu et al., 2026) have emerged as an important paradigm for scaling the capacity of large language models (LLMs), activating only a small subset of experts per token to expand the total parameter count without a proportional increase in training compute (Clark et al., 2022; Tian et al., 2026). As model sizes and training budgets continue to grow, hyperparameters such as the learning rate (LR) and batch size (BS) become increasingly important to training stability, convergence speed, and final performance (McCandlish et al., 2018; Bjorck et al., 2025; Zhang et al., 2025). Exhaustively tuning these hyperparameters at the target scale, however, is prohibitively expensive. Prior studies reduce this cost by establishing empirical scaling laws that relate the optimal learning rate and batch size to model size, dataset size, or compute budget (Kaplan et al., 2020; Hoffmann et al., 2022; Bi et al., 2024). Recent work has extended these laws to MoEs, but largely under conventional or fixed sparsity configurations (Ludziejewski et al., 2025; Li et al., 2025; Wang et al., 2024; Tian et al., 2026).

However, prior studies report conflicting findings on MoE hyperparameters. Some find that hyperparameters transfer robustly between dense models and sparse MoEs (Wang et al., 2024; Li et al., 2025), whereas others observe that MoEs favor larger batch sizes and lower learning rates (Ludziejewski et al., 2025; Tian et al., 2026). Existing scaling laws do not characterize how the optima vary continuously with sparsity, particularly in the ultra-sparse regime down to $A = 1 / 6 4$ Our experiments further show that conventional hyperparameter scaling laws are insufficient for ultra-sparse MoEs. As shown in the controlled experiments in Figure 1, the optimal learning rate and batch size shift between the dense model and the A = 1/64 MoE even when activated parameter count and training data are matched. This shift indicates that the activation ratio provides additional predictive information and should be explicitly incorporated into scaling laws.

To address this gap, we conduct 1,800 pre-training runs, systematically varying the activated non-embedding parameter count N, total non-embedding parameter count $N _ { \mathrm { t o t . } }$ , activation ratio A, and number of training tokens D. The experiments span six activated-parameter scales from approximately 10M to 324M and reach 6B total non-embedding parameters, processing approximately 20 trillion tokens at a cost of 200,000 equivalent H800 GPU-hours. For each configuration, we analytically compute the non-embedding FLOPs per token M and define training compute as $C = M D$ . At fixed sparsity, the optimal batch size and learning rate follow power laws in D and C, respectively. Across sparsity levels, A modifies their prefactors through a multiplicative power law. These findings yield the unified form $h ^ { * } ( X , A ) = \mathbf { \bar { k } } _ { h } X ^ { \gamma _ { h } } A ^ { \delta _ { h } }$ , which we compare with alternative forms using large-scale experimental data. On a frozen target with 324M activated parameters and 12B total parameters, the law jointly extrapolates beyond the fitting ranges in A, D, and C, outperforming existing scaling laws (Bi et al., 2024; Li et al., 2025) and predicting the hyperparameters closest to the observed optima. Overall, our main contributions are as follows:

• We reconcile conflicting findings in prior work within a unified experimental framework. At fixed sparsity, the optimal batch size and learning rate follow power laws in training tokens and training compute, respectively. Across sparsity levels, activation ratio provides an additional key predictor by multiplicatively modifying the prefactors of both laws.

• We derive unified hyperparameter scaling laws across MoE sparsity levels and validate their fit on large-scale experimental data through comparisons with alternative forms. The laws jointly extrapolate to a target beyond the ranges of the development data, predict hyperparameters close to the observed optima, and transfer robustly across the tested expert granularities.

## 2 Preliminaries

We first formulate optimal hyperparameter selection across MoE sparsity levels and then describe the controlled experimental setup used to isolate the effects of scale and sparsity.

## 2.1 Problem Formulation

We formalize optimal hyperparameter selection for MoEs across sparsity levels. Let N denote the number of non-embedding parameters activated per token, $N _ { \mathrm { t o t } }$ the total number of non-embedding parameters, D the number of training tokens, and M the analytically computed non-embedding FLOPs per token for each architecture. We further denote the number of experts activated per token and the total number of experts by $E _ { \mathrm { a c t } }$ and $E _ { \mathrm { t o t } }$ , respectively, and define

$$
A \equiv \frac { E _ { \mathrm { a c t } } } { E _ { \mathrm { t o t } } } , \qquad C \equiv M D ,\tag{1}
$$

where a smaller A indicates a sparser model. Following previous studies (DeepSeek-AI, 2024; Tian et al., 2026), we compute training compute as $C = M D$ rather than using the approximation $C \approx 6 N D$ (Kaplan et al., 2020). Thus, C denotes the analytically computed non-embedding training FLOPs, while N remains a parameter-scale descriptor and candidate predictive variable.

Let η denote the peak learning rate under a fixed schedule and B the global number of tokens processed per optimizer update. Holding the architecture family, data distribution, and all remaining training choices fixed, we write the validation cross-entropy after training on D tokens as

$$
\mathcal { L } ( \eta , B \mid N , N _ { \mathrm { t o t } } , M , D , A ) ,\tag{2}
$$

and define the optimal learning rate and batch size over the candidate search space as

$$
\left( \eta ^ { \ast } , B ^ { \ast } \right) \equiv \underset { \eta , B } { \arg \operatorname* { m i n } } \ : \mathcal { L } \left( \eta , B \ \middle | \ N , N _ { \mathrm { t o t } } , M , D , A \right) .\tag{3}
$$

Equation (3) defines the conceptual joint optimum over the two-dimensional LR–BS space. Because a finite grid does not directly reveal the continuous optimum, subsequent quantitative analyses consider both the observed optimum and the near-optimal set to reduce the effect of noise. Following prior work (Bi et al., 2024), we prespecify a default threshold of 0.1% and regard the hyperparameters of model configurations whose generalization error is no more than this threshold above the minimum as near-optimal. This formulation extends standard hyperparameter-scaling setups (Li et al., 2025; Ludziejewski et al., 2025; Zhou et al., 2026) by making MoE sparsity explicit. It does not assume in advance which scale variables best explain variation in $( \eta ^ { * } , B ^ { * } )$ ; we compare the predictive power of the candidate variables empirically in Section 3.

Following the power-law assumption adopted in prior hyperparameter-scaling studies (Bi et al., 2024; Li et al., 2025; Ludziejewski et al., 2025; Tian et al., 2026), we model the relation between an optimal hyperparameter $h ^ { * }$ and its corresponding scale variable X as

$$
h ^ { \ast } ( X ) = a X ^ { b } ,\tag{4}
$$

where a is the prefactor and b is the scaling exponent. Equivalently, log $h ^ { * } = \log a + b \log X ,$ , where log a is the intercept and b is the slope in log-log space.

![](images/256a756fcc99f34ede3969218af1954c7b500ad9d6820db77622ed1f68a95b0a.jpg)

![](images/03bd9762c80ad7f1231e502b66559c06047e3d29c6ad89be72a9fc97cd022a80.jpg)  
(a)

![](images/5e9505cce4e0838b24075e376d3a3e5adcd74c88cacc55456ac163f6b53df1ad.jpg)  
(b)  
Figure 2 Optimal hyperparameters still shift with sparsity at matched activated or total parameter count. Validation loss is shown against (a) learning rate η at a fixed batch size and (b) batch size B at a fixed learning rate. Within each subfigure, the left panel matches activated parameter count N, while the right panel matches total parameter count $N _ { \mathrm { t o t } }$ Color denotes activation ratio A. Larger markers show the minimum along each plotted curve.

## 2.2 Controlled Experimental Setup

To isolate the effects of model scale, training horizon, and sparsity, we construct six model scales and systematically vary N, $N _ { \mathrm { t o t } } , D ,$ and $A \in \{ 1 , 1 / 4 , 1 / 1 6 , 1 / 3 2 \}$ . The main scaling sweep uses the same mixed pre-training data, 4,096-token sequences, a hybrid linear-attention/MLA backbone (Qin et al., 2023; DeepSeek-AI, 2024; Li et al., 2026), and the Muon optimizer (Liu et al., 2025). Further details are provided in Appendix A.

For each experimental group, we search the peak learning rate η and global token batch size B. Here, B denotes the number of tokens processed per optimizer update. All fitting scales and activation ratios share the main grid listed in Table 7, while the final held-out configuration uses a separate search grid. Training follows a warmup–stable–decay (WSD) schedule (Hu et al., 2024): after a 1% warmup, the learning rate remains at its peak before a final 10% exponential decay. The complete model grid, compute budgets, and implementation controls are reported in Section A.2.

## 3 Optimal Hyperparameters Scaling Laws for Ultra-Sparse MoEs

In this section, we first evaluate activation ratio as a distinct predictive dimension for the optimal learning rate and batch size in MoEs. We then separate its predictive contribution from those of data and compute and derive unified scaling laws.

## 3.1 Beyond Parameter Counts: Sparsity Matters

Existing studies agree that model scale affects the optimal learning rate, but disagree on whether the relevant scale is the activated parameter count (Ludziejewski et al., 2025) or the total parameter count (Li et al., 2025). We first fix the batch size and test whether alignment by either parameter count removes the learning-rate shifts across sparsity levels. As shown in Figure 2a, the curve minima still shift across activation ratios at matched activated parameter count N (left). Alignment by total parameter count $N _ { \mathrm { t o t } }$ likewise fails to remove the shift (right). Thus, neither parameter count alone explains the variation in the optimal learning rate.

The appropriate scale variable for predicting the optimal batch size is similarly unsettled: some studies model it as a function of compute C (Bi et al., 2024; Team et al., 2025), while others tie it primarily to the number of training tokens D (Li et al., 2025). Figure 2b shows that, at a fixed learning rate, neither variable provides a unified explanation of the optimal batch size across sparsity levels. Even when $D , N ,$ and total training compute $C = M D$ are held fixed, $B ^ { * }$ still varies with A (left). Thus, neither training tokens nor compute alone explains the optimal batch size. Alignment by total parameter count $N _ { \mathrm { t o t } }$ leads to the same conclusion (right).

![](images/47c7d9b70bb88581b9f94464bc6a64e4091df83c4a36ea5c969778456180328e.jpg)

![](images/14af2231ed7bde40e1b1a6e8b2401caffdc599cb33b2aaa7eeeaab094236652c.jpg)  
(a)

![](images/7eede2d52076fa7051d9afea2bed529d9e22810cb65de07a63bbdf73f63e6ddf.jpg)

![](images/b3b0024a906d3275f763c35c2dd8068fd71c2c083269f23163fd99641184623f.jpg)  
(b)

Figure 3 Training compute organizes the optimal learning rate at fixed sparsity. (a) Comparison of $N , D ,$ , and C as candidate predictive variables for $\eta ^ { * }$ . (b) Power-law relation between $\eta ^ { * }$ and C on the representative $A = 1 / 3 2$ slice.  
![](images/2094408b523b36a09e45484347294a548d679b1c4ad4fe3444f04c9c4b44a47e.jpg)  
(a)

![](images/b5ee2b44e1edf947bec0bb02bdffa4abcbb7c3a8a0d94d9fda84eaba6f8fc093.jpg)

![](images/dbf03a0d334e75bda1492a6549cb9c53a204f539a3010a8685de823feb91359d.jpg)  
(b)  
Figure 4 Training tokens organize the optimal batch size at fixed sparsity. (a) Comparison of N, C, and D as candidate predictive variables for $B ^ { * }$ . (b) Power-law relation between ${ \hat { B } } ^ { * }$ and D on the representative $A = 1 / 3 2$ slice.

These results provide practical guidance for selecting candidate variables: the effect of sparsity is not absorbed by compute, activated parameter count, or total parameter count. We therefore use the activation ratio A to represent sparsity explicitly, distinguishing models with similar per-token compute but different expert capacities and supplying the predictive dimension needed for scaling across sparsity levels. We evaluate the contribution of A through grouped out-of-fold prediction on the observed two-dimensional surfaces.

Takeaway: Sparsity introduces an additional scaling dimension for optimal hyperparameters. Parameter counts, training tokens, and compute alone cannot explain the shifts in $( \eta ^ { * } , B ^ { * } )$ across sparsity levels; the activation ratio A must therefore be modeled explicitly.

## 3.2 Disentangling Compute, Data, and Sparsity Effects

The preceding subsection shows that the effect of activation ratio on optimal hyperparameters must be modeled explicitly. To separate this effect from the base scale dependencies, we first hold activation ratio fixed and use the estimated optima extracted from the observed two-dimensional loss surfaces to compare which scale variables best explain the optimal learning rate and batch size. We then model the additional effect of activation ratio separately.

![](images/4cb4ceffb20a9eb90d19abc8a0a20166129edf7d4ac33b54aee6e25d3d32eb66.jpg)  
Figure 5 Sparsity-dependent shifts in the optimal-learning-rate power law. Left: near-optimal $\eta ^ { * }$ and power-law fits versus C for four activation ratios; dashed extensions indicate extrapolation beyond each observed range. Middle: fitted exponent $b _ { \eta } ( A )$ versus $A ;$ the horizontal dashed line marks the mean, and the similar exponents motivate a shared-exponent model. Right: near-optimal learning rates at matched FLOPs. Small markers show individual observations, large markers show geometric means, and the solid line shows a multiplicative power-law fit in A.

Compute Scaling of Optimal Learning Rate At fixed sparsity, existing studies make different assumptions about the base scale of the optimal learning rate: some model it jointly with N and D (Bjorck et al., 2025; Li et al., 2025), whereas others use training compute C directly (Bi et al., 2024; Team et al., 2025). Under our definition $C = M D$ , the key question is whether $\eta ^ { * }$ remains sensitive to the allocation between per-token compute M and training duration D at fixed C. We retain N as a parameter-count baseline, compare $N , D ,$ and C as explanatory variables, and vary the $M / D$ allocation while holding C fixed. Figure 3a shows that neither N nor D alone organizes the optimal learning rates across configurations, whereas C provides a clearer relation. Moreover, at the same $C ,$ the optimal learning rate remains nearly unchanged despite large differences in the allocation of M and D. As shown in Figure 3b, $\eta ^ { * }$ follows a stable power law in C on the representative $A = 1 / 3 2$ slice. Using C in place of the separate variables N and D reduces the model degrees of freedom, making the scaling relation more stable and easier to fit.

Data Scaling of Optimal Batch Size At fixed sparsity, the base scale for the optimal batch size is likewise disputed: some studies use training compute C (Bi et al., 2024; Team et al., 2025), while others identify the number of training tokens D as the primary variable (Li et al., 2025; Bergsma et al., 2026). We apply the same controlled comparison used for the learning rate to evaluate whether $N , C ,$ or D most consistently organizes the observed batch-size optima. Figure 4a shows that $B ^ { * }$ is organized primarily by D, while neither N nor C yields a comparably stable relation. On the representative $A = 1 / 3 2$ slice, $B ^ { * }$ follows a stable power law in D, as shown in Figure 4b. Thus, longer training horizons favor larger global token batches, consistent with prior observations (Li et al., 2025; Bergsma et al., 2026).

Sparsity as an Additional Scaling Dimension To characterize how sparsity modifies these relations, we fit a separate scaling curve at each observed slice $A \in \{ 1 , 1 / 4 , 1 / 1 6 , 1 / 3 2 \}$ , compare the fitted coefficients, and analyze how the estimated optimal hyperparameters vary with A at fixed C or D.

Figure 5 summarizes the fitted coefficients across activation ratios. The learning-rate exponents $b _ { \eta } ( A )$ fluctuate around their mean, while at fixed $C ,$ log $\eta ^ { * }$ is approximately linear in $\log _ { 2 } A .$ equivalently $\eta ^ { * } \propto A ^ { \delta _ { \eta } }$ . Appendix Figure 10 shows the corresponding relationship across three additional fixed-compute slices. These descriptive results suggest that sparsity may act primarily through a multiplicative prefactor correction. Because a smaller A yields a lower optimal learning rate, the candidate multiplicative model has $\delta _ { \eta } > 0$ . Batch size exhibits the same pattern but shifts in the opposite direction. In Figure $^ { 6 , }$ the fitted exponents $b _ { B } ( A )$ likewise fluctuate around their mean, while at fixed $D ,$ log $B ^ { * }$ is approximately linear in $\log _ { 2 } A _ { . }$ , equivalently $B ^ { * } \propto A ^ { \delta _ { B } }$ . Because a smaller $A$ yields a larger optimal batch size, the candidate multiplicative model has $\delta _ { B } < 0$ Appendix Figure 11 shows the corresponding relationship across three fixed-token slices. This trend is qualitatively consistent with prior observations (Team et al., 2025; Ludziejewski et al., 2025).

![](images/31bb618b37c26f1cf22dc548c7004872618f517765992f9bf859278c99d28d2d.jpg)

![](images/6f2406d09f7fd8a8123ee3347a6199553d024617f8ed2e70b42239a1ca8ad00b.jpg)

![](images/24904d9fe105458aa68aba116346ea43747a5f6bfba7b107b3cf4d81c3c8df77.jpg)  
Figure 6 Sparsity-dependent shifts in the optimal-batch-size power law. Left: near-optimal $B ^ { * }$ and power-law fits versus D for four activation ratios; dashed extensions indicate extrapolation beyond each observed range. Middle: fitted exponent $b _ { B } ( A )$ versus $A ;$ the horizontal dashed line marks the mean, and the similar exponents motivate a shared-exponent model. Right: near-optimal batch sizes at matched training-token counts. Small markers show individual observations, large markers show geometric means, and the solid line shows a multiplicative power-law fit in A.

Our theoretical analysis shows that a simple gradient-noise model can explain both trends. Under balanced routing, each expert receives only about AB tokens per step, so decreasing A increases expert-side gradient noise and favors a larger global batch. Because the optimal batch size does not grow enough to keep the effective expert batch $A B ^ { * }$ constant, the optimal learning rate still decreases as A decreases. Appendix C provides the full derivation.

Overall, the results show that C and D organize the base power laws for the optimal learning rate and batch size, respectively, while the observed shifts along A support the separable correction $A ^ { \delta _ { h } }$ as a candidate form. Because four slices cannot rule out an interaction in which the exponent varies with A, Section 4 compares the shared-exponent and interaction forms using identical grouped folds rather than selecting a model from the slice plots alone.

Takeaway: Training compute $C ,$ training tokens $D ,$ and activation ratio A provide three complementary dimensions for predicting the optimal hyperparameters of MoEs.

• Base scaling at fixed sparsity. $B ^ { * }$ follows a power law in training tokens $D ,$ while $\eta ^ { * }$ follows a power law in compute $C = M D$ and is insensitive to the $M / D$ allocation at fixed C.

• Multiplicative sparsity correction. The observed shifts along activation ratio A support the separable factor $A ^ { \delta _ { h } }$ as a sparsity correction.

## 3.3 Unified Hyperparameter Scaling Laws for MoEs

Empirical Structure To characterize the relation between activation ratio and the base scale variables, we summarize three empirical observations from the preceding analysis:

1. At fixed $A , \eta ^ { * }$ follows a power law in C, while $B ^ { * }$ follows a power law in D. Thus, C and D serve as their respective base predictive variables.

2. Across activation ratios, the fitted exponents fluctuate around their respective means, supporting a shared-exponent candidate.

3. At fixed C, log η<sup>∗</sup> is approximately linear in $\log _ { 2 } A .$ , with $\delta _ { \eta } > 0 ;$ at fixed D, log $B ^ { * }$ is approximately linear in log A, with $\delta _ { B } < 0$ . Thus, the sparsity effect on each hyperparameter can be represented by a multiplicative power law in A.

Observation 1 supports C and D as the base predictive variables for learning rate and batch size, respectively. Observations 2 and 3 support sharing the base scaling exponent across activation ratios and applying a multiplicative correction in A to the prefactor. Based on these observations, we propose a unified hyperparameter formulation.

Unified Hyperparameter Formulation We express both laws with the common functional family

$$
h ^ { \ast } ( X , A ) = k _ { h } X ^ { \gamma _ { h } } A ^ { \delta _ { h } } , \qquad ( h , X ) \in \{ ( \eta , C ) , ( B , D ) \} .\tag{5}
$$

Here, $k _ { h }$ is a constant prefactor independent of scale and sparsity, $\gamma _ { h }$ is the scaling exponent for the base variable $X ,$ and $\delta _ { h }$ is the sparsity exponent for the activation ratio A. In log space,

$$
\log h ^ { * } = \log k _ { h } + \gamma _ { h } \log X + \delta _ { h } \log A .\tag{6}
$$

Specifically, learning rate and batch size use training compute C and training-token count D as their base scale variables, with multiplicative prefactor corrections in $A ,$ , where $\delta _ { \eta } > 0$ and $\delta _ { B } < 0 :$

$$
\eta ^ { * } ( C , A ) = k _ { \eta } C ^ { \gamma _ { \eta } } A ^ { \delta _ { \eta } } ,\tag{7}
$$

$$
B ^ { * } ( D , A ) = k _ { B } D ^ { \gamma _ { B } } A ^ { \delta _ { B } } .\tag{8}
$$

These formulations clearly separate the base power laws at fixed sparsity from the multiplicative sparsity corrections. The exponents $\gamma _ { \eta }$ and $\gamma _ { B }$ describe scaling with compute and training tokens, respectively, while $\delta _ { \eta }$ and $\delta _ { B }$ measure sensitivity to the activation ratio. The shared-exponent assumption is motivated by the preceding slice trends rather than established by the plots alone. Subsequent analysis compares it with a more flexible interaction form and uses grouped out-of-fold prediction to determine which functional form is better supported by the experimental results.

Comparison with Existing Scaling Laws Table 1 compares representative scaling laws in terms of the scale variables used for the optimal learning rate and batch size. Unlike existing formulations, ours uses C and D as the respective base predictive variables for learning rate and batch size, and uses the activation ratio A to explicitly describe continuous shifts across sparsity levels.

Table 1 Comparison of scale variables used to model optimal learning rates and batch sizes.
<table><tr><td>Method</td><td>Variables for  $\eta ^ { * }$ </td><td>Variables for B*</td></tr><tr><td>DeepSeek Law (Bi et al., 2024)</td><td>C</td><td>C</td></tr><tr><td>Microsoft Law (Bjorck et al., 2025)</td><td> $\left( N _ { \mathrm { t o t } } , D \right)$ </td><td></td></tr><tr><td>Joint MoE Scaling Law (Ludziejewski et al., 2025)</td><td> $\left( N , E _ { \mathrm { t o t } } \right)$ </td><td></td></tr><tr><td>Step Law (Li et al., 2025)</td><td> $\left( N _ { \mathrm { t o t } } , D \right)$ </td><td>D</td></tr><tr><td>Ours</td><td>(C, A)</td><td>(D, A)</td></tr></table>

## 4 Fitting and Predictive Validation

Next, we use large-scale experimental data to estimate the coefficients of the unified scaling laws developed above. We then evaluate their fit quality and robustness through grouped out-of-fold prediction, and finally perform a single-point consistency check for joint extrapolation in $A , D ,$ and C on a held-out target with 12B total parameters and an activation ratio of $A = 1 / 6 4$

## 4.1 Fitting Protocol and Fitted Laws

The formal analysis uses only development data at $A \in \{ 1 , 1 / 4 , 1 / 1 6 , 1 / 3 2 \}$ across the six activatedparameter scales. Functional-family selection, grouped cross-validation, and final coefficient fitting are all restricted to this set. Throughout fitting, cross-validation, fixed-C slicing, and held-out evaluation, compute is obtained from the configuration-specific analytical value as $C = M D$ Following prior work, we prespecify a default threshold of 0.1% and define the near-optimal set for each LR–BS loss surface as the observed grid points whose losses are no more than this threshold above the observed minimum, reducing sensitivity to noise in any single grid optimum. We transform each power law into a linear form in log space and fit its parameters by least squares. Appendix A.2 provides the complete experimental design.

Table 2 Fitted coefficients of the unified scaling laws defined in Equations (7) and (8).
<table><tr><td>Hyperparameter h</td><td>Input variables</td><td> $k _ { h }$ </td><td> $\gamma _ { h }$ </td><td> $\delta _ { h }$ </td></tr><tr><td>Learning rate η</td><td>(C, A)</td><td>0.8343</td><td>-0.1385</td><td>0.1361</td></tr><tr><td>Batch size B</td><td>(D, A)</td><td>6.4765</td><td>0.5181</td><td>-0.0841</td></tr></table>

Table 2 summarizes the fitted coefficients of the two laws. Here, $C = M D$ is measured in nonembedding training FLOPs, while D and B are measured in tokens; the coefficient values therefore depend on these units. The multiplicative sparsity exponents $\delta _ { \eta } ~ > ~ 0$ and $\delta _ { B } ~ < ~ 0$ show that decreasing A lowers the optimal learning rate and increases the optimal batch size.

## 4.2 Fit Quality and Robustness

To evaluate the fit quality and robustness of our scaling laws, we assess out-of-fold predictions using two grouped cross-validation schemes:

• Leave-one-activation-ratio-out (LOAO) holds out one of the four activation ratios and removes all LR–BS loss surfaces at that ratio across active-parameter scales. The optima are then re-extracted from the remaining activation ratios, and all coefficients are refitted.

• Leave-one-active-scale-out (LONO) holds out one of the six active-parameter scales and removes all LR–BS loss surfaces at that scale across activation ratios. The optima are then re-extracted from the remaining active-parameter scales, and all coefficients are refitted.

The purpose of LOAO and LONO is to evaluate which candidate functional family best captures the scaling of the estimated optimal hyperparameter coordinates, rather than to predict validation loss. We therefore use the absolute log-ratio errors in LR and BS as the primary metrics, while practical loss differences are evaluated separately on the final held-out target in Section 4.3. To prevent information leakage, we construct each fold at the loss-surface level and use only the training split to extract optima and estimate coefficients. For each hyperparameter $h ,$ let G denote the set of held-out groups and s index the eligible loss surfaces. We define the overall error as the equally weighted mean of the median absolute log -ratio error within each group:

![](images/4155956bd55cea95d9867f35117ba20efaa47088e3974943ad131607a0bd1f8f.jpg)  
Figure 7 Grouped prediction results for the multiplicative scaling laws. Each point represents an LR–BS loss surface. The left and middle columns show LOAO predictions with $A = 1 / \bar { 3 } 2$ held out and LONO predictions with the 158M scale held out, respectively. Solid lines denote equality, and dashed lines mark factor- $- 2 ^ { 0 . 5 }$ error bounds. The right column compares LOAO and LONO errors with conditional 95% paired surface-cluster bootstrap intervals.

$$
e _ { h } = \frac { 1 } { | \mathcal { G } | } \sum _ { g \in \mathcal { G } } \operatorname * { m e d i a n } _ { s \in \mathcal { g } } \left| \log _ { 2 } ( \hat { h } _ { s } / h _ { s } ^ { * } ) \right| ,\tag{9}
$$

To validate the functional form in Equation (5), Table 3 compares four candidate families, scale only, additive, log interaction, and multiplicative, using identical target loss surfaces and grouped folds. The log-interaction family allows the scaling exponent to vary with A. For LR, all three A-aware families outperform the scale-only law under both cross-validation schemes; for BS, the differences among candidate families are smaller and vary with the grouping scheme. The multiplicative law has the lowest point estimate on three of the four error measures, while log interaction is marginally lower for LR under LOAO (0.150 versus 0.153), indicating similar predictive performance. The left and middle columns of Figure 7 show the held-out A = 1/32 LOAO fold and the held-out 158M

Table 3 Hyperparameter prediction errors for the four candidate families. Each entry reports the LONO and LOAO absolute base-2 log-ratio errors, in that order.
<table><tr><td>Candidate family</td><td>Functional form</td><td>p</td><td>BS error</td><td>LR error</td></tr><tr><td>Scale only</td><td>kXγ</td><td>2</td><td>0.295/0.271</td><td>0.316/0.355</td></tr><tr><td>Additive</td><td> $k _ { X } X ^ { \gamma } + k _ { A } A ^ { \delta }$ </td><td>4</td><td>0.312/0.276</td><td>0.216/0.201</td></tr><tr><td>Log interaction</td><td> $k \bar { X } ^ { \beta _ { X } + \beta _ { X A } \operatorname { l o g } _ { 2 } { A } } A _ { A } \beta _ { A }$ </td><td>4</td><td>0.307/0.238</td><td>0.205/0.150</td></tr><tr><td>Multiplicative (Ours, selected)</td><td> $k X ^ { \gamma } A ^ { \delta }$ </td><td>3</td><td>0.282/0.221</td><td>0.176/0.153</td></tr></table>

![](images/418035631d4776e9a84b7d860f0360925444f6a85fe5f5b07552f8ab6a03274e.jpg)  
(a) Training-loss search surface.

![](images/305d40854488ba2edc78e38b2838ed28717af8c5bf4b827c9a63c52186ec3bce.jpg)  
(b) Validation-loss evaluation surface.  
Figure 8 Training and validation loss surfaces for the held-out joint-extrapolation target. The left panel shows the training loss used for the grid search, and the right panel shows the validation loss. All predictions are frozen before inspection of the held-out losses: the star marks our prediction, and the other markers show predictions from the comparison laws.

LONO fold, respectively; the right column summarizes the out-of-fold errors and conditional 95% paired surface-cluster bootstrap intervals for all four families. Given their comparable predictive performance, we select the multiplicative law as our working model because it uses one fewer parameter and offers a simpler interpretation. Fitted coefficients for the three A-aware candidate families are reported in Appendix B.2.

These results provide initial support for the proposed functional form. The next subsection further evaluates its joint extrapolation by applying it to a target beyond the fitting ranges in $A , D ,$ and C, while retaining N at the observed boundary scale.

## 4.3 Joint Extrapolation to a Held-Out Configuration

Held-out configuration. The final held-out target has $N = 3 2 4 \mathrm { M }$ activated parameters and $N _ { \mathrm { t o t } } = 1 2 \mathrm { B }$ total parameters, with $A = 1 / 6 4$ and $D = 1 5 9 \mathrm { B }$ training tokens. Its analytical M gives a training compute of $C = M D = 3 \times 1 0 ^ { 2 0 }$ FLOPs. While N is the largest scale observed in the fitting set, $A , D ,$ , and C all lie beyond their fitting ranges. The target losses are excluded from functional-family selection, cross-validation, and coefficient fitting until the prediction coordinates are frozen. The pre-specified grid in the holdout row of Appendix Table 7 serves as the search reference.

Baseline protocol. We evaluate existing scaling laws in two complementary ways. Publishedcoefficient transfer directly applies the original coefficients to test transfer across optimizers, training schedules, architectures, and parameter-count definitions, but is not included in the primary functional-form ranking. Refitted-family comparison uses a common optimum definition and grouped folds to select each family on the formal fitting set, then re-estimates its coefficients on the complete fitting set. All coefficients and exact prediction coordinates are frozen before the held-out losses are inspected. The joint exact-point comparison includes only the DeepSeek Law (Bi et al., 2024) and Step Law (Li et al., 2025), which predict both LR and BS; Joint MoE Scaling Laws (Ludziejewski et al., 2025) and Microsoft Law (Bjorck et al., 2025) are omitted because they lack a joint BS law.

Held-out evaluation. Table 4 reports each method’s exact predicted settings, frozen before inspection of the held-out losses, together with their gaps relative to our predicted setting. Figure 8 shows where these predictions lie on the training and validation loss surfaces. For method m, the relative training loss gap in per mille is defined as 1000 $( \widehat { \mathcal { L } } _ { m } - \widehat { \mathcal { L } } _ { \mathrm { o u r s } } ) / \widehat { \mathcal { L } } _ { \mathrm { o u r s } }$ . Results are reported separately for published-coefficient transfer and for refitting on the same samples used by our method. Further limitations are discussed in Section 6.

Table 4 Predictions on the held-out joint-extrapolation target. “Published” uses the original coefficients, while “refitted” re-estimates the same family using the same samples as our method. All methods are evaluated at $N = 3 2 4 \mathrm { M }$ $N _ { { \mathrm { t o t } } } = 1 2 \mathrm { B } , D = 1 5 9 \mathrm { B } , A = 1 / 6 4 ,$ , and $C = 3 \times 1 0 ^ { \dot { 2 } 0 }$ . Relative gaps are measured against our predicted setting.
<table><tr><td></td><td>Mode</td><td>Formula</td><td></td><td></td><td>Predicted LR Predicted BS Loss gap vs. ours</td></tr><tr><td rowspan="2">DeepSeek Law</td><td>Published</td><td> $\eta ^ { * } = 0 . 3 1 1 8 C ^ { - 0 . 1 2 5 0 }$   $\dot { B } ^ { * } = 0 . 2 9 2 0 C ^ { 0 . 3 2 7 1 }$ </td><td> $8 . 5 9 \times 1 0 ^ { - 4 }$ </td><td> $1 . 4 6 \times 1 0 ^ { 6 }$ </td><td>7.22 %o</td></tr><tr><td>Refitted</td><td> $\eta ^ { * } = 1 . 6 7 6 3 C ^ { - 0 . 1 6 1 9 }$   $\dot { B } ^ { * } = 1 8 2 . 9 9 5 1 C ^ { 0 . 2 0 3 8 }$ </td><td> $8 . 1 1 \times 1 0 ^ { - 4 }$ </td><td> $2 . 7 3 \times 1 0 ^ { 6 }$ </td><td>1.37 %o</td></tr><tr><td rowspan="3">Step Law</td><td>Published</td><td> $\begin{array} { l } { \eta ^ { \ast } = 1 . 7 9 0 0 N _ { \mathrm { t o t } } ^ { - 0 . 7 1 3 0 } D ^ { 0 . 3 0 7 0 } } \\ { B ^ { \ast } = 0 . 5 8 0 0 D ^ { 0 . 5 7 1 0 } } \end{array}$ </td><td> $3 . 2 0 \times 1 0 ^ { - 4 }$ </td><td> $1 . 4 4 \times 1 0 ^ { 6 }$ </td><td>9.02 %o</td></tr><tr><td>Refitted</td><td> $\eta ^ { * } = 0 . 0 6 3 8 N _ { \mathrm { t o t } } ^ { - 0 . 1 7 4 3 } D ^ { - 0 . 0 0 8 4 }$   $\dot { B } ^ { * } = 5 . 2 8 2 8 D ^ { \ddot { 0 } . 5 3 4 5 }$ </td><td> $8 . 9 9 \times 1 0 ^ { - 4 }$ </td><td> $5 . 1 3 \times 1 0 ^ { 6 }$ </td><td>1.03 %o</td></tr><tr><td></td><td> $\eta ^ { * } = 0 . 8 3 4 3 C ^ { - 0 . 1 3 8 5 } A ^ { 0 . 1 3 6 1 }$   $\stackrel { \prime } { B ^ { * } } = 6 . 4 7 6 5 D ^ { 0 . 5 1 8 1 } A ^ { - 0 . 0 8 4 1 }$ </td><td> $6 . 9 2 \times 1 0 ^ { - 4 }$ </td><td> $5 . 8 4 \times 1 0 ^ { 6 }$ </td><td></td></tr></table>

## 4.4 Expert-Granularity Transfer and Sparsity Control

We use two controlled comparisons to separate the effects of expert granularity and sparsity. The first holds the activation ratio A fixed while varying the active expert count, total expert count, and expert width, testing whether the activation-ratio relation transfers across expert granularities. The second holds the total expert count and total capacity fixed while changing the active expert count and hence A, testing whether the resulting sparsity shift follows our scaling laws.

Controlled comparisons. All three configurations share the backbone of the approximately 10Mactivated-parameter configuration in the main sweep and are trained on D = 4.56B tokens. Here, “10M” labels the backbone configuration rather than an exactly matched activated parameter count across the three controls. The reference configuration is $( E _ { \mathrm { a c t } } , E _ { \mathrm { t o t } } , h _ { \mathrm { M o E } } ) \ : = \ : ( 2 , 6 4 , 3 8 4 )$ corresponding to $A = 1 / 3 2$ . The expert-granularity control uses (4, 128, 192): it keeps A = 1/32 while doubling the active and total expert counts and halving the expert width, thereby matching activated and total capacity. The sparsity control uses (4, 64, 384): relative to the reference, it keeps the total expert count, expert width, $N _ { \mathrm { t o t } } ,$ , and total routed capacity fixed while increasing A to 1/16. Apart from these MoE expert settings, all three configurations share the remaining training settings and LR–BS search grid. Complete configurations are provided in Appendix A.3.

Results. We first compare the left and middle panels of Figure 9 to test transfer across expert granularities. The two capacity-matched configurations at fixed A have similar near-optimal validation-loss regions and LR–BS optimum locations. Thus, doubling both the active and total expert counts does not materially change the optimal hyperparameters, supporting transfer of the activation-ratio relation across the tested expert granularities. We then compare the left and right panels. Holding the total expert count, expert width, $N _ { \mathrm { t o t } }$ , and total capacity fixed while increasing A from 1/32 to 1/16 shifts both the optimal learning rate and batch size downward, consistent with the direction predicted by our scaling laws. Together, these controls show that the sparsity effect captured by A cannot be attributed solely to any single absolute expert count or to total capacity.

![](images/8c6eb61c48454d2cc59e9286ec4b9b1ce7062b991174e645009f02232ba645a5.jpg)  
Figure 9 Expert-granularity and sparsity controls using the 10M-activated-parameter backbone configuration. Left: the reference with $A = 1 / 3 2$ and $( E _ { \mathrm { a c t } } , E _ { \mathrm { t o t } } , h _ { \mathrm { M o E } } ) = ( 2 , 6 4 , 3 8 4 )$ . Middle: (4, 128, 192) with the same A, activated capacity, and total capacity, testing transfer across expert granularities. Right: (4, 64, 384) with the same total expert count $E _ { t o t } ,$ expert width, and total capacity as the reference, but $A = 1 / 1 6$ due to the larger top-n. This complements the main sweep, which changes A through $E _ { t o t } .$ . Each cell shows validation loss at one LR–BS grid point; stars mark near-optimal points.

## 5 Related Work

We review two lines of related work: optimization hyperparameter scaling and sparse MoE scaling.

Scaling Laws for Optimal Hyperparameters Scaling laws for optimal hyperparameters aim to transfer configurations identified in small-scale experiments reliably to larger training scales. Existing studies are either theory-driven or empirically driven. Theory-driven methods, represented by the µP framework (Yang et al., 2021) and its extensions (Yang et al., 2024; Dey et al., 2026; Peng et al., 2026), use scale-aware parameterizations to preserve hyperparameter transfer across changes in model width, depth, and other architectural dimensions. Such approaches generally rely on particular initialization or parameterization choices and focus on variation induced by architectural scaling. Empirically driven methods instead identify and fit scale dependence directly from training experiments. They show that optimal learning rates and batch sizes vary jointly with model size and training horizon (Bjorck et al., 2025; Bi et al., 2024; Zhang et al., 2025); subsequent work further characterizes the coupled scaling of batch size and weight decay (Bergsma et al., 2026). Together, these results establish predictable scaling behavior in optimization hyperparameters. We extend this empirical framework to hyperparameter scaling across MoE sparsity levels.

Scaling Hyperparameters for MoEs Recent studies have begun to examine whether hyperparameter rules transfer from dense models to Mixture-of-Experts architectures (Li et al., 2025; Tian et al., 2026; Ludziejewski et al., 2025; Zhou et al., 2026). Step Law reported that learning-rate and batch-size scaling laws are broadly robust across dense and MoE models (Li et al., 2025). Other studies observed that MoEs tend to favor larger global batch sizes and slightly lower learning rates than comparably scaled dense models (Tian et al., 2026; Ludziejewski et al., 2025). Meanwhile, work on MoE hyperparameter transfer showed that model width, depth, expert count, and expert width can be incorporated into a unified transfer parameterization (Jiang et al., 2026). However, existing studies largely compare dense models with a limited set of MoE configurations rather than explicitly modeling the continuous dependence of optimal hyperparameters on sparsity (Tian et al.,

2026; Li et al., 2025). Some joint scaling analyses also omit relevant factors such as the training horizon (Ludziejewski et al., 2025), while parameterization-based transfer has been validated over only a restricted range of sparsity configurations (Jiang et al., 2026). Consequently, a hyperparameter scaling law that transfers systematically across sparsity levels remains missing. This gap is critical for frontier ultra-sparse MoEs (Team et al., 2026; Xu et al., 2026), which span much wider ranges of expert count and activation ratio. Our work systematically quantifies the effect of sparsity on optimal hyperparameters and derives unified scaling laws that transfer across sparsity levels.

## 6 Limitations and Future Directions

Experimental and evaluation scope. Our evidence comes from a single hybrid linear-attention/MLA backbone, data mixture, Muon optimizer, and sigmoid auxiliary-loss-free routing setup, while the expert-granularity and sparsity controls cover only three configurations. Transfer to other model scales, architectures, optimizers, training schedules, routing mechanisms, and broader expert configurations therefore requires further validation. We also define optimal hyperparameters using validation loss, whose improvements may not translate uniformly to downstream capabilities (Gadre et al., 2025; Isik et al., 2025; Sun et al., 2026). Capability-oriented objectives and dataand efficiency-scaling laws for ultra-sparse MoEs remain important directions for future work.

Functional-form and extrapolation uncertainty. The grouped-cross-validation intervals overlap, and the number of available groups is limited; the current evidence therefore neither establishes a significant advantage for the no-interaction model nor rules out variation of the scaling exponent with activation ratio. The joint-extrapolation evaluation also contains only one target beyond the fitting ranges in A, D, and C, so it cannot isolate extrapolation along each variable. Although this study includes 1,800 pre-training runs, the cost of large-scale training prevents us from repeating the complete grid across multiple random seeds, as is common in lower-cost conventional machine-learning experiments. This resource constraint is broadly shared by large-scale scaling-law studies (Kaplan et al., 2020; Hoffmann et al., 2022; Ludziejewski et al., 2025; Tian et al., 2026). Consequently, each exact prediction is evaluated once against the minimum of noisy grid observations, making the observed loss gaps descriptive point estimates. These results do not guarantee applicability to arbitrary out-of-range targets or quantify predictive uncertainty beyond the observed domain.

## 7 Conclusion

We systematically study how the optimal learning rate and batch size of ultra-sparse MoEs vary with compute scale, training horizon, and sparsity. Our central finding is that conventional scale variables alone cannot describe hyperparameter shifts across sparsity levels: the activation ratio A must be treated as an additional predictive dimension. At fixed sparsity, the optimal learning rate follows a power law in training compute C = MD and remains stable across different M/D allocations at fixed C, while the optimal batch size follows a power law in training tokens D. Across sparsity levels, A modifies the prefactors of both relations through a multiplicative power law, yielding the unified form $h ^ { * } ( X , \bar { A } ) = k _ { h } X ^ { \gamma _ { h } } A ^ { \delta _ { h } }$ . Grouped prediction shows that this form outperforms the scale-only and additive alternatives. On the frozen target with 12B total parameters, the law jointly extrapolates beyond the fitting ranges in A, D, and $C ,$ with its prediction lying on the observed near-optimal loss plateau. Further controls show that the relation transfers across capacitymatched expert granularities. Overall, these results turn sparsity from an architectural attribute into an explicit predictor for hyperparameter selection, providing a transferable prescription for training ultra-sparse MoEs at scale.

## References

Shane Bergsma, Nolan Dey, Gurpreet Gosal, Gavia Gray, Daria Soboleva, and Joel Hestness. Power lines: Scaling laws for weight decay and batch size in llm pre-training. Advances in Neural Information Processing Systems, 38:125153–125188, 2026.

Xiao Bi, Deli Chen, Guanting Chen, Shanhuang Chen, Damai Dai, Chengqi Deng, Honghui Ding, Kai Dong, Qiushi Du, Zhe Fu, et al. Deepseek llm: Scaling open-source language models with longtermism. arXiv preprint arXiv:2401.02954, 2024.

Johan Bjorck, Alon Benhaim, Vishrav Chaudhary, Furu Wei, and Xia Song. Scaling optimal lr across token horizons. In International Conference on Learning Representations, volume 2025, pages 83640–83657, 2025.

Aidan Clark, Diego de Las Casas, Aurelia Guy, Arthur Mensch, Michela Paganini, Jordan Hoffmann, Bogdan Damoc, Blake Hechtman, Trevor Cai, Sebastian Borgeaud, et al. Unified scaling laws for routed language models. In International conference on machine learning, pages 4057–4086. PMLR, 2022.

DeepSeek-AI. Deepseek-v3 technical report, 2024. URL https://arxiv.org/abs/2412.19437.

Nolan Dey, Bin Zhang, Lorenzo Noci, Mufan Li, Blake Bordelon, Shane Bergsma, Cengiz Pehlevan, Boris Hanin, and Joel Hestness. Don’t be lazy: Completep enables compute-efficient deep transformers. Advances in Neural Information Processing Systems, 38:137707–137739, 2026.

Samir Yitzhak Gadre, Georgios Smyrnis, Vaishaal Shankar, Suchin Gururangan, Mitchell Wortsman, Rulin Shao, Jean Mercat, Alex Fang, Jeffrey Li, Sedrick Keh, et al. Language models scale reliably with over-training and on downstream tasks. In International Conference on Learning Representations, volume 2025, pages 67661–67682, 2025.

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022.

Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui, Xiang Long, Zhi Zheng, Yewei Fang, Yuxiang Huang, Weilin Zhao, et al. Minicpm: Unveiling the potential of small language models with scalable training strategies. arXiv preprint arXiv:2404.06395, 2024.

Berivan Isik, Natalia Ponomareva, Hussein Hazimeh, Dimitris Paparas, Sergei Vassilvitskii, and Sanmi Koyejo. Scaling laws for downstream task performance in machine translation. In International Conference on Learning Representations, volume 2025, pages 88769–88790, 2025.

Tianze Jiang, Blake Bordelon, Cengiz Pehlevan, and Boris Hanin. Hyperparameter transfer with mixture-of-expert layers. arXiv preprint arXiv:2601.20205, 2026.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.

Ang Li, Ben Liu, Bin Han, Bin Hu, Bin Jing, Binbin Hu, Bing Li, Cai Chen, Caizhi Tang, Changxin Tian, et al. Ling and ring 2.6 technical report: Efficient and instant agentic intelligence at trillion-parameter scale. arXiv preprint arXiv:2606.15079, 2026.

Houyi Li, Wenzhen Zheng, Qiufeng Wang, Hanshan Zhang, Zili Wang, Shijie Xuyang, Yuantao Fan, Zhenyu Ding, Haoying Wang, Ning Ding, et al. Predictable scale: Part i, step law–optimal hyperparameter scaling law in large language model pretraining. arXiv preprint arXiv:2503.04715, 2025.

Jingyuan Liu, Jianlin Su, Xingcheng Yao, Zhejun Jiang, Guokun Lai, Yulun Du, Yidao Qin, Weixin Xu, Enzhe Lu, Junjie Yan, et al. Muon is scalable for llm training. arXiv preprint arXiv:2502.16982, 2025.

Jan Ludziejewski, Maciej Pióro, Jakub Krajewski, Maciej Stefaniak, Michał Krutul, Jan Mała´snicki, Marek Cygan, Piotr Sankowski, Kamil Adamczewski, Piotr Miło´s, et al. Joint moe scaling laws: Mixture of experts can be memory efficient. arXiv preprint arXiv:2502.05172, 2025.

Sam McCandlish, Jared Kaplan, Dario Amodei, and OpenAI Dota Team. An empirical model of large-batch training. arXiv preprint arXiv:1812.06162, 2018.

Hongwu Peng, Ohiremen Dibua, Yuanjun Xiong, Yifan Gong, Jianming Zhang, and Yan Kang. Complete-mue: Optimal hyperparameter transfer and scaling for moe models. arXiv preprint arXiv:2605.23893, 2026.

Zhen Qin, Dong Li, Weigao Sun, Weixuan Sun, Xuyang Shen, Xiaodong Han, Yunshen Wei, Baohong Lv, Xiao Luo, Yu Qiao, et al. Transnormerllm: A faster and better large language model with improved transnormer. arXiv preprint arXiv:2307.14995, 2023.

Noam Shazeer. Glu variants improve transformer. arXiv preprint arXiv:2002.05202, 2020.

Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538, 2017.

Yusuxke Shibata, Takuya Kida, Shuichi Fukamachi, Masayuki Takeda, Ayumi Shinohara, Takeshi Shinohara, and Setsuo Arikawa. Byte pair encoding: A text compression scheme that accelerates pattern matching. 1999.

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

Quanen Sun, Changxin Tian, Ke Shi, Cai Chen, Cunyin Peng, Jia Liu, Kunlong Chen, and Zhiqiang Zhang. Supervalid: Capability-aligned ood validation for generalizable downstream scaling. arXiv preprint arXiv:2605.28179, 2026.

Kimi Team, Tongtong Bai, Yifan Bai, Yiping Bao, Jianfeng Cai, Xinyuan Cai, Peizhou Cao, Yuxuan Cao, Ziwei Chai, Y Charles, et al. Kimi k3: Open frontier intelligence. arXiv preprint arXiv:2607.24653, 2026.

Ling Team, Ang Li, Ben Liu, Binbin Hu, Bing Li, Bingwei Zeng, Borui Ye, Caizhi Tang, Changxin Tian, Chao Huang, et al. Every activation boosted: Scaling general reasoner to 1 trillion open language foundation. arXiv preprint arXiv:2510.22115, 2025.

Changxin Tian, Kunlong Chen, Jia Liu, Ziqi Liu, Zhiqiang Zhang, and Jun Zhou. Towards greater leverage: Scaling laws for efficient mixture-of-experts language models. In International Conference on Learning Representations, volume 2026, pages 29806–29843, 2026.

Siqi Wang, Zhengyu Chen, Bei Li, Keqing He, Min Zhang, and Jingang Wang. Scaling laws across model architectures: A comparative analysis of dense and moe models in large language models. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing, pages 5583–5595, 2024.

Anyi Xu, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, Chenchen Ling, et al. Deepseek-v4: Towards highly efficient million-token context intelligence. arXiv preprint arXiv:2606.19348, 2026.

Ge Yang, Edward Hu, Igor Babuschkin, Szymon Sidor, Xiaodong Liu, David Farhi, Nick Ryder, Jakub Pachocki, Weizhu Chen, and Jianfeng Gao. Tuning large neural networks via zero-shot hyperparameter transfer. Advances in Neural Information Processing Systems, 34:17084–17097, 2021.

Greg Yang, Dingli Yu, Chen Zhu, and Soufiane Hayou. Tensor programs vi: Feature learning in infinite depth neural networks. In International Conference on Learning Representations, volume 2024, pages 55099–55150, 2024.

Hanlin Zhang, Depen Morwani, Nikhil Vyas, Jingfeng Wu, Difan Zou, Udaya Ghai, Dean Foster, and Sham Kakade. How does critical batch size scale in pre-training? In International Conference on Learning Representations, volume 2025, pages 66756–66782, 2025.

Yunhua Zhou, Shuhao Xing, Junhao Huang, Xipeng Qiu, and Qipeng Guo. How to set the learning rate for large-scale pre-training? In Findings ofthe Associationfor Computational Linguistics: ACL 2026, pages 37344–37361, 2026.

## A Additional Experimental Details

This section supplements the experimental setup in Section 2.2. We describe the fixed training controls, followed by the model scales, compute budgets, and hyperparameter search spaces.

## A.1 Training Controls

Table 5 summarizes the training framework, tokenizer, backbone, optimization, and routing settings. All experiments use Megatron<sup>1</sup>, 4,096-token sequences, and the same internal mixture of web, book, and code data. Engineering settings such as numerical precision and parallelism strategy are held fixed across experiments and are not treated as study variables, so they are not listed separately.

Table 5 Training controls shared across experiments.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Training framework</td><td>Megatron</td></tr><tr><td>Sequence length</td><td>4,096 tokens</td></tr><tr><td>Tokenizer</td><td>Byte-level BPE (Shibata et al., 1999), vocabulary size 157,184</td></tr><tr><td>Normalization</td><td>RMSNorm with QK layer normalization</td></tr><tr><td>FFN activation</td><td>SwiGLU (Shazeer, 2020)</td></tr><tr><td>Backbone</td><td>Hybrid linear attention and multi-latent attention (MLA) (Li et al., 2026)</td></tr><tr><td>Position encoding</td><td>RoPE (Su et al., 2024)</td></tr><tr><td>Initialization</td><td>Standard deviation 0.006; chunk initialization with  $\alpha = 3 . 0$ </td></tr><tr><td>Optimizer</td><td>Muon (Liu et al., 2025)</td></tr><tr><td>LR schedule</td><td>WSD: 1% warmup and final 10% exponential decay (Hu et al., 2024)</td></tr><tr><td>Weight decay</td><td>0.1, including normalization parameters</td></tr><tr><td>Gradient clipping</td><td>Global norm 1.0</td></tr><tr><td>MoE routing</td><td>Sigmoid routing with Auxiliary-Loss-Free load balancing strategy (DeepSeek-AI, 2024)</td></tr></table>

## A.2 Model Configurations and Search Grids

Table 7 summarizes the model configurations and search grid used for formal functional-family selection and coefficient fitting, together with the final held-out target with 324M activated and 12B total parameters. The formal analysis, family selection, cross-validation, and coefficient fitting use only $A \in \left\{ 1 , 1 / 4 , 1 / 1 6 , 1 / 3 2 \right\}$ at the six activated-parameter scales, with main-grid compute budgets from $3 \times 1 0 ^ { 1 7 } \mathrm { t o } 1 0 ^ { 2 0 }$ FLOPs. The main grid searches $\eta \in \{ 5 , 7 , 1 0 , 1 4 , 2 0 , 2 8 , 4 0 , 5 6 \} \times 1 0 ^ { - 4 }$ and $\mathsf { \bar { B } } \in \{ 2 ^ { 1 7 } , 2 ^ { 1 8 } , \dots , 2 ^ { 2 3 } \}$ ; the held-out configuration instead searches $\eta \in \{ 3 . 6 , 5 , 7 , 1 0 , 1 4 , 2 0 \} \times$ $1 0 ^ { - 4 }$ and $B \in \{ 2 ^ { 1 9 } , 2 ^ { 2 0 } , \dots , 2 ^ { 2 3 } \}$ . Every compute budget and Target FLOPs entry is obtained as $C = M D$ using the analytical non-embedding FLOPs/token of the corresponding configuration. All fitting scales and activation ratios share the main grid. The final target retains the observed boundary scale $N = 3 2 4 \mathrm { M } ,$ but its $A = 1 / 6 4 , D = 1 5 9 \mathrm { B }$ , and $C = 3 \times 1 0 ^ { 2 0 }$ all lie beyond the fitting ranges, making it a single joint-extrapolation target in these three variables; this held-out configuration uses the separately specified search grid. Its losses remain uninspected until the prediction coordinates are frozen, and it enters none of the preceding operations. Under fixed top-2 routing, the five activation ratios correspond to 2, 8, 32, 64, and 128 total experts, respectively. To improve resource efficiency, we stop selected grid runs when intermediate results show that they are clearly outside the near-optimal region and unlikely to provide additional information. Accordingly, the loss surfaces in this paper denote the observed two-dimensional grids for each configuration and do not necessarily cover the full Cartesian product of the pre-specified LR–BS grid. Compute is calculated analytically for each hybrid linear-attention/MLA MoE configuration. Specifically, M counts non-embedding forward-and-backward FLOPs per token for that architecture and $C = M D$ gives non-embedding training FLOPs. The reported fits, grouped validation, fixed-C slices, held-out compute, and Target FLOPs entries all use this same accounting.

Table 6 Architecture controls for the six model scales. “S/L” gives MLA/linear-attention layers; the main grid activates two routed experts per token. Activated parameter counts and fitting roles are specified in Table 7.
<table><tr><td>Scale</td><td>Layers</td><td>Hidden size</td><td>Heads</td><td>Routed-expert width  $( h _ { \mathrm { M o E } } )$ </td><td> $\mathrm { S / L }$ </td></tr><tr><td>10M</td><td>10</td><td>256</td><td>4</td><td>384</td><td>2/8</td></tr><tr><td>20M</td><td>10</td><td>384</td><td>6</td><td>512</td><td>2/8</td></tr><tr><td>40M</td><td>12</td><td>512</td><td>8</td><td>640</td><td>3/9</td></tr><tr><td>80M</td><td>16</td><td>640</td><td>10</td><td>768</td><td>4/12</td></tr><tr><td>158M</td><td>20</td><td>768</td><td>12</td><td>1,024</td><td>4/16</td></tr><tr><td>324M</td><td>24</td><td>1,024</td><td>16</td><td>1,280</td><td>4/20</td></tr></table>

Table 7 Coefficient-fitting configurations and final joint-extrapolation holdout. These configurations support formal analysis, functional-family selection, grouped cross-validation, and coefficient estimation. The holdout uses a separate grid and retains the observed boundary scale $N = 3 2 4 \mathrm { M } ,$ , but lies beyond the fitting ranges in $A , D ,$ and $C ;$ it is excluded from all four operations and evaluated only after its prediction coordinates are frozen.
<table><tr><td>Scale</td><td>N</td><td>A</td><td> $E _ { \mathrm { a c t } }$ </td><td> $E _ { \mathrm { { t o t } } }$ </td><td> $N _ { \mathrm { t o t } }$ </td><td>D</td><td>Target FLOPs</td><td>Role in law fitting</td></tr><tr><td rowspan="4">10M</td><td rowspan="4">10M</td><td>1</td><td>2</td><td>2</td><td>10M</td><td rowspan="4">4.6B</td><td rowspan="4"> $3 \times 1 0 ^ { 1 7 }$ </td><td rowspan="4">Fit</td></tr><tr><td>1/4</td><td>2</td><td>8</td><td>27M</td></tr><tr><td>1/16</td><td>2</td><td>32</td><td>98M</td></tr><tr><td>1/32</td><td>2</td><td>64</td><td>193M</td></tr><tr><td rowspan="4">20M</td><td rowspan="4">20M</td><td>1</td><td>2</td><td>2</td><td>20M</td><td rowspan="4">7.8B</td><td rowspan="4"> $1 \times 1 0 ^ { 1 8 }$ </td><td rowspan="4">Fit</td></tr><tr><td>1/4</td><td>2</td><td>8</td><td>55M</td></tr><tr><td>1/16</td><td>2</td><td>32</td><td>197M</td></tr><tr><td>1/32</td><td>2</td><td>64</td><td>386M</td></tr><tr><td rowspan="4">40M</td><td rowspan="4">40M</td><td>1</td><td>2</td><td>2</td><td>40M</td><td rowspan="4">11.6B</td><td rowspan="4"> $3 \times 1 0 ^ { 1 8 }$ </td><td rowspan="4">Fit</td></tr><tr><td>1/4</td><td>2</td><td>8</td><td>111M</td></tr><tr><td>1/16</td><td>2</td><td>32</td><td>395M</td></tr><tr><td>1/32</td><td>2</td><td>64</td><td>772M</td></tr><tr><td rowspan="4">80M</td><td rowspan="4">80M</td><td>1</td><td>2</td><td>2</td><td>80M</td><td rowspan="4">19.6B</td><td rowspan="4"> $1 \times 1 0 ^ { 1 9 }$ </td><td rowspan="4">Fit</td></tr><tr><td>1/4</td><td>2</td><td>8</td><td>223M</td></tr><tr><td>1/16</td><td>2</td><td>32</td><td>790M</td></tr><tr><td>1/32</td><td>2</td><td>64</td><td>1.5B</td></tr><tr><td rowspan="4">158M</td><td rowspan="4">158M</td><td>1</td><td>2</td><td>2</td><td>158M</td><td rowspan="4">31.7B</td><td rowspan="4"> $3 \times 1 0 ^ { 1 9 }$ </td><td rowspan="4">Fit</td></tr><tr><td>1/4</td><td>2</td><td>8</td><td>442M</td></tr><tr><td>1/16</td><td>2</td><td>32</td><td>1.6B</td></tr><tr><td>1/32</td><td>2</td><td>64</td><td>3.1B</td></tr><tr><td rowspan="4">324M</td><td rowspan="4">324M</td><td>1</td><td>2</td><td>2</td><td>324M</td><td rowspan="4">52.9B</td><td rowspan="4"> $1 \times 1 0 ^ { 2 0 }$ </td><td rowspan="4">Fit</td></tr><tr><td>1/4</td><td>2</td><td>8</td><td>894M</td></tr><tr><td>1/16</td><td>2</td><td>32</td><td>3.2B</td></tr><tr><td>1/32</td><td>2</td><td>64</td><td>6.2B</td></tr><tr><td>324M</td><td>324M</td><td>1/64</td><td>2</td><td>128</td><td>12.2B</td><td>159.0B</td><td> $3 \times 1 0 ^ { 2 0 }$ </td></tr></table>

![](images/59ddb0fc0eacfe46029cfba86066ddb00fa51705b7b2c0c33aa922037143c201.jpg)  
Figure 10 Learning-rate dependence on sparsity across fixed-compute slices. Each panel shows near-optimal learning rates as a function of A at a fixed compute budget: $C = 3 \times 1 0 ^ { 1 7 } , \dot { 3 } \times 1 0 ^ { 1 8 }$ , and $2 \times \dot { 1 } 0 ^ { 1 9 }$ FLOPs from left to right. Faint markers denote individual observations, highlighted markers their geometric means, and solid lines show multiplicative power-law fits in A.

## A.3 Expert-Granularity-Controlled Configurations

The validation uses three configurations that share the approximately 10M-activated-parameter backbone, each trained on $D = 4 . 5 6 \mathrm { B }$ tokens. Here, “10M” labels the backbone configuration rather than an exactly matched activated parameter count across all three configurations. The reference $( E _ { \mathrm { a c t } } , E _ { \mathrm { t o t } } , h _ { \mathrm { M o E } } ) = ( 2 , 6 4 , 3 8 4 )$ and the expert-granularity control (4, 128, 192) share $A = 1 / 3 2$ and match activated and total capacity. The sparsity control (4, 64, 384) matches the reference in total expert count, expert width, $N _ { \mathrm { t o t } } ,$ and total routed capacity, but uses $A = 1 / 1 6$ . All configurations are excluded from coefficient estimation and functional-family selection.

Table 8 Held-out expert-granularity and sparsity controls using the approximately 10M-activated-parameter backbone configuration. No row contributes to coefficient estimation or model selection.
<table><tr><td>Configuration</td><td> $\left( E _ { \mathrm { a c t } } , E _ { \mathrm { t o t } } \right)$ </td><td> $h _ { \mathrm { M o E } }$ </td><td>A</td><td>Activated routed capacity</td><td>Total routed capacity</td><td>Role</td></tr><tr><td>Reference</td><td>(2,64)</td><td>384</td><td>1/32</td><td> $2 \times 3 8 4$ </td><td> $6 4 \times 3 8 4$ </td><td>Holdout</td></tr><tr><td>Granularity control</td><td>(4,128)</td><td>192</td><td>1/32</td><td> $4 \times 1 9 2$ </td><td> $1 2 8 \times 1 9 2$ </td><td>Holdout</td></tr><tr><td>Sparsity control</td><td>(4,64)</td><td>384</td><td>1/16</td><td> $4 \times 3 8 4$ </td><td> $6 4 \times 3 8 4$ </td><td>Holdout</td></tr></table>

## B Additional Experimental Results

This section reports supplementary sparsity slices and fitted coefficients for the candidate functional families.

## B.1 Sparsity Dependence Across Fixed-Scale Slices

The right panels of Figures 5 and 6 show the dependence of the near-optimal hyperparameters on the activation ratio at one fixed compute or token budget, respectively. Here, we extend this analysis to three fixed-C slices and three fixed-D slices to test whether the relationship persists across training scales. Across all slices, the sparsity-induced shifts are not specific to a single reference scale: over the compute and token ranges examined, both near-optimal learning rates and batch sizes follow the separable multiplicative power-law correction $A ^ { \delta _ { h } }$ used in the main text.

![](images/0b450b375032c152c60a289de4374fa0085ca33738ddf96639c5156faf1e9de1.jpg)  
Figure 11 Batch-size dependence on sparsity across fixed-token slices. Each panel shows near-optimal batch sizes as a function of A at a fixed training horizon: $D \stackrel { \cdot } { = } 4 . 7 \times 1 0 ^ { 9 } , 7 . 8 \times 1 0 ^ { 9 }$ , and $3 . 2 \times 1 0 ^ { 1 0 }$ tokens from left to right. Faint markers denote individual observations, highlighted markers their geometric means, and solid lines show multiplicative power-law fits in A.

## B.2 Candidate Functional-Form Coefficients

Table 9 reports the full-development-set coefficients and log-space fit metrics for all four candidate families. RMSE and $R ^ { 2 }$ are computed from base-2 log residuals on the same target-aligned surfaces, with lower RMSE and higher ${ \bf { \bar { \phi } } } _ { R ^ { 2 } }$ indicating better in-sample fit. These descriptive metrics are distinct from the grouped out-of-fold errors used for model selection in the main text. Although the additive family achieves the best in-sample metrics, the multiplicative law is selected because its grouped predictive performance is comparable while using one fewer parameter and admitting a simpler interpretation.

Table 9 Fitted coefficients and in-sample fit metrics for the four candidate families. RMSE and $R ^ { 2 }$ are evaluated in base-2 log space on the target-aligned development surfaces. Lower RMSE and higher $R ^ { 2 }$ indicate better in-sample fit.
<table><tr><td>Candidate family</td><td>Functional form</td><td>Target</td><td>Fitted coefficients</td><td>RMSE↓</td><td> $\overline { { R ^ { 2 } \uparrow } }$ </td></tr><tr><td rowspan="2">Scale only</td><td rowspan="2"> $k X ^ { \gamma }$ </td><td></td><td> $\operatorname { L R } \left( X = C \right) \quad k = 0 . 8 7 0 4 , \gamma = - 0 . 1 4 6 3$ </td><td>0.3634</td><td>0.3879</td></tr><tr><td></td><td> $\mathrm { B S } \left( X = D \right) k = 1 . 4 8 2 7 , \gamma = 0 . 5 8 9 4$ </td><td>0.3057</td><td>0.7104</td></tr><tr><td rowspan="2">Additive</td><td rowspan="2"> $k _ { X } X ^ { \gamma } + k _ { A } A ^ { \delta }$ </td><td></td><td> $\mathrm { L R } \left( X = C \right) \quad k _ { X } = 1 . 7 5 7 9 \times 1 0 ^ { 8 } , \gamma = - 0 . 6 3 6 2 , k _ { A } = 1 . 9 3 5 8 \times 1 0 ^ { - 3 } , \delta = 0 . 1 8 0 8$ </td><td>0.2358</td><td>0.7422</td></tr><tr><td></td><td> $\mathrm { B S } \left( X = D \right) \quad k _ { X } = 0 . 0 1 9 7 , \gamma = 0 . 7 6 5 1 , k _ { A } = 1 . 1 8 0 2 \times 1 0 ^ { 5 } , \delta = - 0 . 3 4 1 2$ </td><td>0.2644</td><td>0.7834</td></tr><tr><td rowspan="2">Log interaction</td><td rowspan="2"> $k X ^ { \beta _ { X } + \beta _ { X A } \log _ { 2 } { A } } A ^ { \beta _ { A } }$ </td><td></td><td> $\begin{array} { r l } { \mathrm { L R } \left( X = C \right) } & { { } k = 0 . 3 1 7 3 , \mathrm { ~ } \beta _ { X } = - 0 . 1 1 5 5 , \mathrm { ~ } \beta _ { A } = - 0 . 1 4 3 4 , \mathrm { ~ } \beta _ { X A } = 0 . 0 0 4 6 } \end{array}$ </td><td>0.2452</td><td>0.7214</td></tr><tr><td></td><td> $\begin{array} { r l } { \mathsf { B S } \left( X = D \right) } & { { } k = 0 . 1 2 7 1 , \beta _ { X } = 0 . 6 8 9 6 , \beta _ { A } = - 1 . 1 3 0 8 , \beta _ { X A } = 0 . 0 3 1 9 } \end{array}$ </td><td>0.2661</td><td>0.7805</td></tr><tr><td rowspan="2">Multiplicative (selected)</td><td rowspan="2"> $k X ^ { \gamma } A ^ { \delta }$ </td><td></td><td> $\operatorname { L R } \left( X = C \right) \quad k = 0 . 8 3 4 3 , \gamma = - 0 . 1 3 8 5 , \delta = 0 . 1 3 6 1$ </td><td>0.2463</td><td>0.7188</td></tr><tr><td></td><td> $\begin{array} { r l } { \operatorname { B S } \left( X = D \right) } & { { } k = 6 . 4 7 6 5 , \ \gamma = 0 . 5 1 8 1 , \delta = - 0 . 0 8 4 1 } \end{array}$ </td><td>0.2765</td><td>0.7630</td></tr></table>

## C Theoretical Analysis

This section gives a simple local model for the two sparsity trends observed in the main text: as the activation ratio A decreases, the optimal batch size increases and the optimal learning rate decreases. We use the notation of the main text, where $E _ { \mathrm { a c t } }$ and $E _ { \mathrm { { t o t } } }$ denote the number of experts activated per token and the total number of experts, respectively, and $A = E _ { \mathrm { a c t } } / E _ { \mathrm { t o t } }$

## C.1 Why Sparsity Increases the Optimal Batch Size

Consider an MoE layer with global token batch size B. Under balanced routing, each expert receives approximately AB tokens on average. Under an independent-sample approximation, the variance

of its gradient estimate is inversely proportional to this effective batch size:

$$
\mathrm { V a r } ( \widehat { g } _ { e } ) \propto \frac { 1 } { A B } .\tag{10}
$$

Thus, decreasing A increases expert-side gradient noise and favors a larger global batch. At a fixed training-token budget D, however, increasing B reduces the number of optimizer updates to $D / B$ Meanwhile, shared parameters still receive gradients from the full batch and are not thinned by sparse routing. We model this trade-off with the local approximation

$$
\Delta \mathcal { L } ( B ; D , A ) \simeq c _ { \mathrm { s t e p } } \left( \frac { B } { D } \right) ^ { p } + c _ { \mathrm { s h a r e d } } B ^ { - q } + c _ { \mathrm { e x p e r t } } ( A B ) ^ { - q } , \qquad p , q > 0 ,\tag{11}
$$

where all three coefficients are positive. The terms represent the penalties from fewer optimizer updates, gradient noise in shared parameters, and gradient noise in expert parameters, respectively. Minimizing over B gives

$$
B ^ { \ast } ( D , A ) = \left[ \frac { q \left( c _ { \mathrm { s h a r e d } } + c _ { \mathrm { e x p e r t } } A ^ { - q } \right) } { p c _ { \mathrm { s t e p } } } \right] ^ { \frac { 1 } { p + q } } D ^ { \frac { p } { p + q } } .\tag{12}
$$

In this model, the training-token exponent is $\gamma _ { B } = p / \left( p + q \right)$ and does not depend on A, consistent with the shared-exponent form used in the main text. The local elasticity with respect to A is

$$
\delta _ { B } ( A ) \equiv \frac { \partial \log { B ^ { * } } } { \partial \log { A } } = - \frac { q } { p + q } \frac { c _ { \mathrm { e x p e r t } } A ^ { - q } } { c _ { \mathrm { s h a r e d } } + c _ { \mathrm { e x p e r t } } A ^ { - q } } , \qquad - 1 < \delta _ { B } ( A ) < 0 .\tag{13}
$$

Thus, the optimal batch size increases as A decreases, but more slowly than $A ^ { - 1 }$ . The update-count penalty and shared-parameter noise both weaken the effect of sparse routing, consistent with the sublinear trend in Figure 11.

## C.2 Why Sparsity Decreases the Optimal Learning Rate

Equation (10) shows that expert-side gradient noise increases as the effective expert batch AB decreases. Under a local noise-scale approximation, the expert-side noise induced by one optimizer update scales as

$$
\begin{array} { r } { \tau _ { e } \propto \frac { \eta } { A B } , } \end{array}\tag{14}
$$

where $\eta$ is the learning rate. A smaller effective expert batch therefore favors a smaller optimal learning rate. We summarize the response of the optimizer and the buffering effect of shared parameters with an exponent $\rho \colon$

$$
\eta ^ { \ast } \propto ( A B ^ { \ast } ) ^ { \rho } , \qquad 0 < \rho \leq 1 .\tag{15}
$$

From Equation (13), the increase in $B ^ { * }$ is not large enough to offset the decrease in A, because

$$
\frac { \partial \log ( A B ^ { * } ) } { \partial \log A } = 1 + \delta _ { B } ( A ) > 0 .\tag{16}
$$

The corresponding local sparsity exponent for the learning rate is therefore

$$
\delta _ { \eta } ( A ) \equiv { \frac { \partial \log \eta ^ { * } } { \partial \log A } } = \rho { \left( 1 + \delta _ { B } { \left( A \right) } \right) } > 0 .\tag{17}
$$

As A decreases, the effective number of samples received by each expert still falls, so the optimal learning rate also decreases. The accompanying increase in $B ^ { * }$ partially offsets this effect, while shared parameters are not directly thinned by sparse routing; the resulting learning-rate shift is therefore typically modest.

The analysis relies on simplifying assumptions, including balanced routing, independent samples, and a local noise model. It explains the empirical signs $\delta _ { B } < 0$ and $\delta _ { \eta } > 0$ and the approximately shared base exponent for batch size across sparsity levels; it does not derive the fitted coefficients from first principles or require the sparsity exponents to be exactly constant. In particular, the base scaling of the learning rate with compute C remains determined empirically in the main text.