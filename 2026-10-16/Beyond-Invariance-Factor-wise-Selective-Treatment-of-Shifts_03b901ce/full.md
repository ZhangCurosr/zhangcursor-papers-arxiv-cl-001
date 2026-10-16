# Beyond Invariance: Factor-wise Selective Treatment of Shifts for Multimodal Learning

Guimin Hu<sup>1</sup>, Qiwei Peng<sup>2</sup>

<sup>1</sup>Guangdong University of Technology <sup>2</sup>University of Copenhagen rice.hu.x@gmail.com

## Abstract

Distribution shifts pose a fundamental challenge to robust multimodal learning. Existing methods typically promote invariance by sup pressing environment-dependent variations, implicitly applying a uniform treatment to shifted information. However, shifted information can have different predictive consequences: some contributes little to prediction, some re mains reliably informative, and some exhibits environment-dependent predictive effects. This raises a fundamental question: which shifted information should be suppressed, preserved, or attenuated? To address this question, we propose FaShift, a factor-wise shift framework that diagnoses latent representation factors ac cording to their distributional changes, inter vention effects, and cross-environment stability. By factorizing multimodal representations and evaluating these complementary properties, FaShift distinguishes non-predictive factors, stably predictive factors, and predictively useful but unstable factors. It then selectively suppresses, preserves, or attenuates their contributions, rather than enforcing uniform invariance. Experiments on MOSI, MOSEI, and CH-SIMS show that FaShift achieves the best ID accuracy and F1 across all three datasets. Compared with the strongest competitors reporting OOD results, it improves OOD accuracy/F1 by 1.5/3.6 points on MOSI and 1.7/2.7 points on MOSEI, while achieving the best reported OOD accuracy and F1 on CH-SIMS.

## 1 Introduction

Multimodal learning exploits heterogeneous modalities to capture complementary information, but data distributions can vary across domains, users, acquisition conditions, and time. Such distribution shifts may induce reliance on environment-specific correlations, degrading generalization to unseen environments. A common approach is invariant representation learning, which aims to reduce environment-dependent variation while preserving predictive information that generalizes across environments.

![](images/79fb3f70220ecc9762cef96dab5c07b18750191c3b97d69b59070402ee97bc99.jpg)

![](images/4607b50c063b3c391c758b02ca4fb57f65ccc14c65b0f34d91bc0c01ad554e2b.jpg)  
Figure 1: Heterogeneous factor shifts motivate shiftaware reliance. Left: Factor-wise shift and intervention effects. Right: Shift-aware factor reliance.

However, environment-dependent variation is not necessarily spurious. Shifted representations may retain predictive information, yet exhibit different predictive consequences across factors. Thus, the key challenge extends beyond identifying invariant information to assessing the predictive effects of shifted information and their stability across environments. This raises a fundamental question: how should shifted information be treated according to its predictive effects and crossenvironment stability?

To investigate this question at the factor level, we decompose multimodal representations into latent factors. As illustrated in Figure 1, these factors vary substantially in shift magnitude, predictive effects, and the cross-environment stability of those effects (left). Notably, similar shifts can induce markedly different changes in model predictions (right), indicating that shift magnitude alone is insufficient to determine how a factor should be treated. This motivatesfactor-wise diagnosis, which determines whether each shifted factor should be suppressed, preserved, or attenuated based on its predictive effects and their cross-environment stability.

Existing approaches address distribution shifts through invariant representation learning, causal adaptation, or evolving domain generalization (Mai and Han, 2026b; Zhong et al., 2025; He et al., 2025). Although these methods account for the potential predictive value of environment-dependent information, they generally emphasize learning transferable representations or adaptation mechanisms, with less focus on systematically diagnosing how individual shifted factors affect predictions and whether these effects remain stable across environments (Sun et al., 2022; Mai et al., 2023).

To address this question, we propose FaShift (Factor-wise Shift Discovery), a factor-level framework for diagnosing and selectively adapting representation shifts. FaShift first decomposes multimodal representations into latent factors and characterize each factor from three complementary perspectives: distributional shift, interventional effect, and cross-environment stability. Distributional shift measures how a factor’s representation changes across environments. We then intervene on each factor while keeping the remaining factors fixed and quantify the resulting change in prediction. Finally, we examine whether this interventional effect remains consistent across environments. These three characteristics distinguish whether a factor shifts, whether it affects prediction, and whether its predictive effect is stable across environments.

Based on these factor-wise characteristics, FaShift performs selective adaptation rather than enforcing universal invariance. Factors with negligible interventional effects are suppressed, while factors with meaningful and stable effects are retained. For factors with meaningful but unstable effects, FaShift attenuates their contributions according to their degree of cross-environment instability. In this way, FaShift identifies which shifted factors are consequential for prediction and determines whether their contributions should be suppressed, preserved, or attenuated according to the stability of their effects across environments. This factorwise treatment provides a shift-aware alternative to universal invariance for robust multimodal learning. Our contributions are summarized as follows:

• We show that shift magnitude alone is insufficient to determine factor treatment: factors with similar distributional shifts can have markedly different predictive consequences, motivating selective rather than uniform treatment of shifted information.

• We propose FaShift, a factor-wise framework that diagnoses individual representation factors through distributional shift, intervention effect, and cross-environment stability, and accordingly suppresses, preserves, or attenuates their contributions.

• Extensive experiments across multimodal datasets show that FaShift consistently improves OOD generalization while maintaining SOTA ID performance, demonstrating the effectiveness of selective treatment over universal invariance.

## 2 Related Work

## 2.1 Invariant Multimodal Learning

Distribution shifts have motivated invariant representation learning and domain generalization, which seek predictive relationships that remain stable across environments. Recent multimodal studies have incorporated invariant perspectives to improve robustness under such shifts. Domain Generalization via Causal Adjustment (Wang et al., 2024) employs backdoor adjustment to remove domain-specific confounding and learn domaininvariant representations. CmIR (Mai and Han, 2026b) disentangles modality representations into causal-invariant and environment-specific components, while SCLF (Cao et al., 2026) identifies stable causal latent features through invariance and stability constraints. Other approaches explore counterfactual intervention for robust emotion recognition (Zhong et al., 2025) and causal representation learning for domain generalization (Liang et al., 2026). These methods primarily seek invariant or stable representations by suppressing environmentdependent variations. In contrast, we ask how each shifted factor should be treated according to its intervention effect and cross-environment stability.

## 2.2 Distribution Shifts and Adaptation

Beyond invariant representation learning, causal adaptation studies have examined how predictive mechanisms transfer or evolve across environments. Causal domain adaptation methods model domain differences as changes in the underlying data-generating process and seek relationships that remain transferable across domains (Magliacane et al., 2018; Teshima et al., 2020; Yue et al., 2021). More recent work on evolving domain generalization further recognizes that environment-dependent factors may contain useful information, motivating the joint modeling of invariant and dynamic representations (Xie et al., 2024; He et al., 2025). These studies highlight that distribution shifts need not uniformly correspond to spurious variation. However, existing approaches primarily focus on transferring mechanisms or learning representations that generalize across evolving environments, without explicitly characterizing how individual factors change or whether such changes affect prediction. We instead take a shift-centric perspective, quantifying factor-wise shifts and intervention effects to determine which should be suppressed or preserved, rather than uniformly enforcing invariance.

## 3 FaShift

## 3.1 Problem Formulation

Given a multimodal observation X with label Y under an environment $\textit { E } \in \textit { \mathcal { E } } .$ , we aim to determine how environment-induced representation shifts should be treated for robust prediction. While conventional robust learning often seeks to suppress environment-dependent variations, such a strategy implicitly assumes that all shifts are undesirable. This raises a fundamental question: when a representation shifts across environments, should the shift be suppressed, preserved, or adaptively attenuated?

Each environment represents a distinct datagenerating condition, such as domain, population, acquisition setting, or temporal context. The key challenge is therefore to distinguish representation shifts that contribute to prediction from those that do not, and to determine their appropriate treatment across environments.

Shift Does Not Directly Determine Treatment. An environment-induced shift does not by itself determine how the corresponding factor should be treated. A factor may vary substantially across environments while having little effect on the prediction when intervened upon. Therefore,

$$
D _ { k } ^ { a , b } > 0 ; \neq ; | \Delta _ { k } ^ { a  b } | > 0 .\tag{1}
$$

Conversely, a shifted factor may exhibit a substantial intervention-based effect on the prediction:

$$
D _ { k } ^ { a , b } > 0 \quad \mathrm { a n d } \quad | \Delta _ { k } ^ { a  b } | > 0 .\tag{2}
$$

Thus, shift magnitude alone cannot determine factor treatment. Intervention effects indicate prediction relevance, while cross-environment stability determines whether relevant factors should be preserved or attenuated. This motivates our factorwise analysis for selective treatment.

![](images/7a8507d8418dffbb27dd33e773ea839f2bdb2ad98e730781ec380f325aa760ea.jpg)  
Figure 2: Visualization of factor-wise shift diagnosis in the PCA space of the learned latent factors. FaShift identifies heterogeneous shift patterns across latent factors, distinguishing nuisance, unstable predictive, and stable predictive shifts.

## 3.2 Stage I: Factor Discovery.

Factorized Representation Learning. We first encode and fuse the multimodal inputs into a joint representation using a multimodal encoder $f _ { \mathrm { m m } } .$ $\mathbf { H } = f _ { \mathrm { m m } } ( \mathcal { X } )$ , which is subsequently factorized into K latent factors:

$$
\mathbf { Z } = [ \mathbf { z } _ { 1 } , \ldots , \mathbf { z } _ { K } ] = { \mathcal { F } } _ { \boldsymbol { \theta } } ( \mathbf { H } ) .\tag{3}
$$

where $\mathcal { F } _ { \theta }$ denotes the factorization module and each $\mathbf { z } _ { k }$ captures a potentially distinct aspect of the predictive representation without requiring predefined semantic annotations. The resulting factorized representation is then mapped to the task output by a prediction head $g ( \cdot )$

$$
{ \hat { Y } } _ { Z } = g ( \mathbf { Z } ) .\tag{4}
$$

To reduce redundancy among factors, we minimize their pairwise statistical dependence using the Hilbert-Schmidt Independence Criterion (HSIC)(Gretton et al., 2005):

$$
\mathcal { L } _ { \mathrm { d e c } } = \sum _ { i \neq j } \mathrm { H S I C } ( \mathbf { z } _ { i } , \mathbf { z } _ { j } ) ,\tag{5}
$$

where HSIC measures statistical dependence between two random variables. The representation objective is

$$
\mathcal { L } _ { \mathrm { r e p } } = \ell ( \hat { Y } _ { Z } , Y ) + \lambda _ { \mathrm { d e c } } \mathcal { L } _ { \mathrm { d e c } } .\tag{6}
$$

where $\ell ( \cdot , \cdot )$ denotes the task-specific prediction loss. This stage learns a non-redundant factorized representation that preserves task-relevant information and supports factor-level intervention, without assuming environmental stability of individual factors. We assess factor redundancy by computing the Pearson correlation between the $\ell _ { 2 } \cdot$ -norm activations of each factor pair. Lower correlations indicate less overlap between factors (see Experiment section). The factorization is designed to preserve the task-relevant information in the original representation, such that the factorized representation Z retains predictive capability for Y . Specifically, we require a small gap between $\hat { Y } _ { Z }$ and ${ \hat { Y } } _ { H } = g ( \mathbf { H } )$ using the same prediction head.

## 3.3 Stage II: Factor-wise Shift Diagnosis

We characterize each factor by its distributional shift, intervention-based predictive relevance, and cross-environment stability. Given a source environment $E _ { a }$ and a target environment $E _ { b }$ , factorlevel interventions assess the predictive effects of cross-environment changes, while cross-modal factor interactions capture their joint effects. These statistics guide factor-wise treatment under distributional shifts.

Environment Construction. We construct training environments to capture heterogeneous conditions for factor-wise shift diagnosis. When reliable environmental attributes are available, such as speaker, participant, session, or temporal context, we use them directly to define environments. Otherwise, we construct pseudo-environments from the Stage-I factorized representations by computing the factor activation profile.

$$
\mathbf { s } ^ { ( n ) } = [ | \mathbf { z } _ { 1 } ^ { ( n ) } | _ { 2 } , \dots , | \mathbf { z } _ { K } ^ { ( n ) } | _ { 2 } ]\tag{7}
$$

for each training sample and cluster these profiles into groups using K-means. The resulting groups serve as pseudo-environments for shift diagnosis. Environment pairs $( E _ { a } , E _ { b } )$ are then used to assess factor-wise distributional shifts, intervention effects, and cross-environment stability. Environments are constructed using training data only, with validation data used for hyperparameter selection and the test set reserved for final evaluation.

Shift Discovery. For each factor $\mathbf { z } _ { k }$ , we measure its distributional shift between an environment pair $( E _ { a } , E _ { b } )$ by

$$
D _ { k } = \mathcal { D } \big ( P ( \mathbf { z } _ { k } | E _ { a } ) , P ( \mathbf { z } _ { k } | E _ { b } ) \big ) ,\tag{8}
$$

where D denotes a distribution discrepancy measure such as Maximum Mean Discrepancy (MMD)

<table><tr><td>Shift Type</td><td>Decision Criterion Treatment</td><td></td></tr><tr><td>Nuisance Shift</td><td> $\Delta _ { k } \le \tau _ { c }$ </td><td>Suppress</td></tr><tr><td>Stable Predictive Shift Unstable Predictive Shift</td><td> $\Delta _ { k } > \tau _ { c } , \ V _ { k } \leq \tau _ { v }$   $\Delta _ { k } > \tau _ { c } , \ V _ { k } > \tau _ { v }$ </td><td>Preserve Attenuate</td></tr></table>

Table 1: Shift categorization and corresponding selective treatment of representation factors. $\tau _ { c }$ and $\tau _ { v }$ are the thresholds for intervention effect and instability, respectively.

or Wasserstein distance. A factor is regarded as shifted when $D _ { k } > \tau _ { s }$

Factor-level Intervention Attribution. For a sample from environment $E _ { b } ,$ , we treat $\mathbf { z } _ { k }$ as the factor to be intervened on and identify the top-M nearest counterparts in a reference environment $E _ { a }$ by matching the non-intervened factors z<sub>−k</sub>. We then average their corresponding $\mathbf { z } _ { k }$ representations to obtain a reference factor $\mathbf { z } _ { k } ^ { a , * }$ , which is used to construct the counterfactual representation:

$$
\begin{array} { r } { { \bf z } ^ { k  a } = [ { \bf z } _ { 1 } ^ { b } , \cdot \cdot \cdot , { \bf z } _ { k } ^ { a , * } , \cdot \cdot \cdot , { \bf z } _ { K } ^ { b } ] . } \end{array}\tag{9}
$$

Here, only $\mathbf { z } _ { k }$ is replaced, while all non-intervened factors $\mathbf { z } _ { - k }$ remain unchanged. We measure the similarity between samples based on their nonintervened factors using cosine similarity, and select the top-M nearest counterparts from the reference environment. The relevance of the shift is quantified by the prediction change:

$$
\Delta _ { k } ^ { a \right. b } = d _ { Y } \big ( g ( \mathbf { z } ^ { b } ) , g ( \mathbf { z } ^ { k \left. a } ) \big ) ,\tag{10}
$$

where $d _ { Y }$ measures the difference between the factual and counterfactual predictions.

Joint Intervention Attribution. Factor-level intervention alone may not fully capture the joint effects of multimodal factors, whose combined influence can be complementary, redundant, or conflicting. We therefore further consider pairwise joint interventions to characterize cross-factor dependencies. Specifically, for a factor pair $( \mathbf { z } _ { i } , \mathbf { z } _ { j } )$ we simultaneously replace both factors with their reference counterparts:

$$
\mathbf { z } ^ { i j  a } = [ \mathbf { z } _ { 1 } ^ { b } , \ldots , \mathbf { z } _ { i } ^ { a , * } , \ldots , \mathbf { z } _ { j } ^ { a , * } , \ldots , \mathbf { z } _ { K } ^ { b } ]\tag{11}
$$

The joint intervention effect is defined as

$$
\Delta _ { i j } ^ { a \right. b } = d _ { Y } \left( g ( \mathbf { z } ^ { b } ) , g ( \mathbf { z } ^ { i j \left. a \right)} )\tag{12}
$$

We then quantify the cross-factor interaction as the deviation of the joint effect from the sum of the individual effects:

$$
I _ { i j } ^ { a  b } = \Delta _ { i j } ^ { a  b } - \Delta _ { i } ^ { a  b } - \Delta _ { j } ^ { a  b } .\tag{13}
$$

A positive $I _ { i j }$ indicates a synergistic interaction, while a negative value indicates a sub-additive interaction caused by redundancy, saturation, or conflicting effects. We analyze only cross-modal factor pairs to characterize their complementary, redundant, or conflicting dependencies.

Cross-environment Stability. We further examine whether the intervention effects remain consistent across environment pairs. For factor-level effects, we quantify relative variability as

$$
V _ { k } = \frac { \mathrm { S t d } ( \Delta _ { k } ) } { \mathrm { M e a n } ( \Delta _ { k } ) + \epsilon } .\tag{14}
$$

Similarly, we measure the cross-environment instability of factor interactions as

$$
V _ { i j } = \frac { \mathrm { S t d } ( I _ { i j } ) } { \mathrm { M e a n } ( | I _ { i j } | ) + \epsilon } , I _ { k } = \sum _ { j \neq k } V _ { k j } ,\tag{15}
$$

A smaller $V _ { k }$ or $I _ { k }$ indicates greater crossenvironment stability. Based on $( D _ { k } , \Delta _ { k } , V _ { k } )$ and the associated cross-factor interaction profiles, shifted factors are categorized and treated as summarized in Table 1 and Figure 2.

## 3.4 Stage III: Selective Shift Treatment

Based on the statistics obtained in Stage II, we determine the retention weight of each factor according to its intervention effect and cross-environment stability. Specifically, we define

$$
w _ { k } = \left\{ \begin{array} { l l } { 0 , } & { \Delta _ { k } \leq \tau _ { c } , } \\ { 1 , } & { \Delta _ { k } > \tau _ { c } , V _ { k } \leq \tau _ { v } , } \\ { \frac { \bar { \Delta } _ { k } } { \bar { \Delta } _ { k } + \lambda _ { v } S _ { k } + \epsilon } , } & { \Delta _ { k } > \tau _ { c } , V _ { k } > \tau _ { v } , } \end{array} \right.\tag{16}
$$

where $S _ { k } = \bar { V } _ { k } + \lambda _ { I } \bar { I } _ { k }$ aggregates the instability and interaction terms for factor k, $w _ { k } \in [ 0 , 1 ]$ is the retention weight of factor $k , \tau _ { c }$ and $\tau _ { v }$ are the thresholds for intervention effect and instability, respectively, $\bar { \Delta } _ { k } , ~ \bar { I } _ { k }$ and $\bar { V _ { k } }$ are the normalized intervention effect and instability. $\lambda _ { v }$ controls the penalty on instability.

The resulting retention weights are applied to the factorized representation as

$$
\begin{array} { r } { \widetilde { Z } _ { k } = w _ { k } Z _ { k } , \qquad \widetilde { Z } = [ \widetilde { Z } _ { 1 } , \dots , \widetilde { Z } _ { K } ] , } \end{array}\tag{17}
$$

with the final prediction given by $\widehat { Y } = g ( \widetilde { Z } )$

This selective treatment suppresses irrelevant shifts, preserves stable and task-relevant factors, and attenuates environmentally sensitive ones. The weights are derived directly from interventionbased statistics across training environments, without factor-level semantic annotations or an additional domain adaptation network.

## 4 Experiments

## 4.1 Environment Setting

We construct training environments from the natural data-generating structure of each dataset to expose the model to heterogeneous conditions for factor-wise shift diagnosis. We consider two representative types: Human-style environments, where each speaker or participant defines an environment and captures individual-specific multimodal characteristics; and Temporal-style environments, where samples are partitioned by recording time or session to capture temporal variations in multimodal characteristics. See Appendix ?? for further details on environment construction.

Unlike the environments defined for factor-level shift diagnosis and treatment policy learning, OOD settings refer to distribution-shift scenarios used exclusively for evaluation. This separation allows us to assess whether the treatment policy learned from naturally occurring training heterogeneity can generalize beyond the environments used for diagnosis.

Datasets. We evaluate our method under two settings: in-distribution (ID) and out-ofdistribution (OOD). For ID evaluation, we follow the standard train-dev-test splits of MOSEI (Zadeh et al., 2018), MOSI (Zadeh et al., 2016), CH-SIMS (Yu et al., 2020) and MUD3 (Wang et al., 2025a), where the training and test sets are drawn from the same underlying distribution. For OOD evaluation, we adopt the distribution-shifted settings introduced by Sun et al. (Sun et al., 2022) and Zhang et al. (Zhang et al., 2025) on the corresponding datasets. In our experiments, we define environments based on speakers for MOSI, MO-SEI, and CH-SIMS, with each speaker constituting one environment. For MU3D, we define temporal environments according to the recording period and more results are reported in Appendix B.

Metric. We use Acc-2 (Acc.) and F1 as the primary metrics, measuring classification accuracy and the balance between precision and recall, respectively.

Multimodal Baselines. We compare FaShift with representative multimodal learning methods, including Self-MM (Yu et al., 2021), C-MIB (Mai et al., 2023), DMD (Li et al., 2023), DEVA (Wu et al., 2025), AtCAF (Huang et al., 2025), EMOE (Fang et al., 2025), Boosting (Mai et al., 2024), ITHP (Xiao et al., 2024), DLF (Wang et al., 2025b), CaReFlow (Mai and Han, 2026a), and CmIR (Mai and Han, 2026b), covering diverse multimodal representation, interaction, fusion, and optimization strategies.

<table><tr><td rowspan="3">Method</td><td colspan="3">MOSI</td><td colspan="4">MOSEI</td><td colspan="4">CH-SIMS</td></tr><tr><td colspan="2">ID</td><td colspan="2">OOD</td><td colspan="2">ID</td><td colspan="2">OOD</td><td colspan="2">ID</td><td colspan="2">OOD</td></tr><tr><td>Acc.↑ F1↑</td><td></td><td>Acc.↑ F1↑</td><td>Acc.↑</td><td>F1↑</td><td>Acc.↑</td><td>F1↑</td><td>Acc.↑</td><td>F1↑</td><td>Acc.↑</td><td>F1↑</td></tr><tr><td>Self-MM</td><td>84.9</td><td>84.8</td><td>76.7</td><td>76.7</td><td>85.2</td><td>85.2</td><td>74.7 74.3</td><td>80.7</td><td>80.8</td><td>70.1*</td><td></td><td>70.1*</td></tr><tr><td>C-MIB</td><td>87.8</td><td>87.8</td><td>77.9</td><td>77.8</td><td>86.9</td><td>86.8</td><td>75.1</td><td>75.3</td><td></td><td>1</td><td></td><td></td></tr><tr><td>DMD</td><td>84.3</td><td>84.3</td><td>77.5</td><td>77.4</td><td>84.6</td><td>84.6</td><td>75.2</td><td>75.7</td><td></td><td></td><td></td><td></td></tr><tr><td>DEVA</td><td>86.3</td><td>86.3</td><td></td><td></td><td>86.1</td><td>86.2</td><td></td><td></td><td>79.6</td><td>80.3</td><td>69.7*</td><td>69.8*</td></tr><tr><td>AtCAF</td><td>88.6</td><td>88.5</td><td>78.1</td><td>77.2</td><td>87.0</td><td>86.8</td><td>75.3</td><td>75.7</td><td></td><td></td><td></td><td></td></tr><tr><td>EMOE</td><td>84.8</td><td>84.8</td><td>77.6</td><td>76.8</td><td>85.0</td><td>85.0</td><td>75.0</td><td>74.6</td><td>78.8*</td><td>78.5*</td><td>67.3*</td><td>67.3*</td></tr><tr><td>Boosting</td><td>88.5</td><td>88.4</td><td></td><td></td><td>86.5</td><td>86.5</td><td></td><td></td><td></td><td>1</td><td></td><td></td></tr><tr><td>ITHP</td><td>88.5</td><td>88.5</td><td></td><td></td><td>87.1</td><td>87.1</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DLF</td><td>88.7</td><td>88.6</td><td>79.7</td><td>77.5</td><td>87.5</td><td>87.4</td><td>76.8</td><td>76.2</td><td>80.4*</td><td>80.3*</td><td>71.2*</td><td>70.8*</td></tr><tr><td>CaReFlow</td><td>89.8</td><td>89.7</td><td></td><td></td><td>87.9</td><td>88.0</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CmIR</td><td>89.6</td><td>89.5</td><td></td><td></td><td>87.8</td><td>87.7</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>FaShift(Ours)</td><td>90.9±0.3 91.4±0.3</td><td></td><td>81.2±0.1 81.4±0.2</td><td></td><td>88.2±0.3 88.3±0.3</td><td></td><td>78.5±0.1 78.4±0.1</td><td></td><td></td><td>83.3±0.2 83.3±0.3</td><td></td><td>73.7±0.4 74.2±0.2</td></tr></table>

Table 2: Comparison of different methods under in-domain (ID) and out-of-distribution (OOD) settings on MOSI, MOSEI, and CH-SIMS. We report binary accuracy (Acc.) and F1 score. Results marked with an asterisk (\*) are reproduced using the authors’ publicly available code, while “–” indicates that results and codes are unavailable. Best results are highlighted in bold.

<table><tr><td rowspan="2">Rep.</td><td colspan="2">MOSEI</td><td colspan="2">CH-SIMS</td></tr><tr><td>Acc.</td><td>F1</td><td>Acc.</td><td>F1</td></tr><tr><td>Original H</td><td>84.4</td><td>84.2</td><td>75.6</td><td>75.7</td></tr><tr><td>Factorized Z</td><td>84.6</td><td>84.3</td><td>75.9</td><td>75.9</td></tr><tr><td>Gap</td><td>0.2</td><td>0.1</td><td>0.3</td><td>0.2</td></tr></table>

Table 3: Task information preservation in the factorized representation.

## 4.2 Implementation Details

All experiments are implemented in PyTorch and conducted on a single NVIDIA GeForce RTX 4080 GPU. Unless otherwise specified, the factorized representation contains K = 6 factors. The factorization module is optimized using AdamW with a learning rate of $4 \times 1 0 ^ { - 4 }$ , a weight decay of $2 \times 1 0 ^ { - 5 }$ , and a batch size of 128. The coefficient of the decorrelation loss is set to $\lambda _ { \mathrm { d e c } } = 5 0 0$ . For intervention-based shift analysis, we use the factor representations learned from the training data and construct reference samples from the corresponding environments. We set $M = 8 , \tau _ { c } = 0 . 2$ $\tau _ { s } = 0 . 0 2$ and $\tau _ { v } = 0 . 6$ , with all hyperparameters selected on the validation set. The prediction head and the factorization module are trained using the training split only, while all OOD data are reserved for evaluation. See Appendix A for more implementation details.

![](images/aa0dfdeca6608683dd9f0c130144a4ecc9d803c55e6fee4e11be4bce363233a1.jpg)

![](images/da277abf7c32ea0618ffc7528e614bf05df37764534d95bd3665a9d59d127c59.jpg)  
Figure 3: Visualization of the learned factors. Left: pairwise correlation matrix. Right: PCA projection of the learned factors.

## 4.3 ID and OOD Results

Table 2 compares FaShift with existing multimodal sentiment analysis methods under both indistribution (ID) and out-of-distribution (OOD) settings. For MOSI, MOSEI, and CH-SIMS, we use speaker-based environments to characterize factorwise shifts across speakers. FaShift achieves the best overall performance across all three datasets, with more pronounced gains under OOD evaluation. These results suggest that modeling factorwise shifts provides benefits beyond conventional ID fitting. FaShift characterizes shifts across speaker-defined environments and uses intervention effects and cross-environment stability to selectively suppress, preserve, or attenuate individual factors. This enables the model to retain predictive variations while reducing the influence of unstable or less relevant shifts, which contributes to stronger OOD robustness.

<table><tr><td>Setting</td><td>Variant</td><td>Acc.</td><td>F1</td></tr><tr><td>ID: MOSEI</td><td>w/o Factorization w/o Shift Diagnosis Uniform Treatment FaShift</td><td>84.4 84.6 83.8 88.2</td><td>84.2 84.3 83.6 88.3</td></tr><tr><td>ID: CH-SIMS</td><td>w/o Factorization w/o Shift Diagnosis Uniform Treatment FaShift</td><td>75.6 75.9 75.5 83.3</td><td>75.7 75.9 75.1 83.3</td></tr><tr><td>OOD: CH-SIMS</td><td>w/o Factorization w/o Shift Diagnosis Uniform Treatment FaShift</td><td>69.4 70.3 70.5 73.7</td><td>69.1 70.5 70.8 74.2</td></tr><tr><td>OOD: MOSEI</td><td>w/o Factorization w/o Shift Diagnosis Uniform Treatment FaShift</td><td>73.9 74.1 74.1 78.5</td><td>73.3 74.2 74.6 78.4</td></tr></table>

Table 4: Ablation study of FaShift under in-domain (ID) and distribution-shifted (OOD) settings.

![](images/348b01ab0a8345cf5a497651c4a75beae3b04d299677966d02f7cb8f523888e0.jpg)

![](images/2f54004291c6cf9e9bddd44c38f5f23143d9c8dff0fc9250543d17305d0abb4a.jpg)  
Figure 4: Factor-wise intervention effects and stability as a function of distribution shifts.

The consistent gains across the three datasets support FaShift’s premise that robust multimodal learning should distinguish heterogeneous factorwise shifts and treat them selectively, achieving a better balance between predictive performance and robustness.

## 4.4 Analysis of Learned Factors

We examine whether the proposed factorization preserves task-relevant information. As shown in Table 3, the factorized representation Z achieves performance close to the original representation H on both CH-SIMS and MOSEI, with gaps of only 0.2 and 0.1 points in Acc. and F1 on CH-SIMS, and 0.3 and 0.2 points on MOSEI, respectively. These small differences indicate that factorization incurs minimal loss of predictive information, while Z retains the task-relevant information in H for subsequent factor-level shift analysis.

We further visualize the learned factors in Figure 3. The PCA projection shows that the factors exhibit structured distributions in the latent space, while the pairwise correlation matrix indicates relatively weak correlations among most factor pairs. Together, these observations suggest that the learned factors capture complementary information with limited redundancy. Importantly, these analyses serve as empirical validation of the learned representation rather than direct evidence of factor independence. Combined with the task information preservation results, they support the use of Z as the basis for the subsequent factor-level shift analysis.

![](images/efc003616be9e474c27913e83311ec7b0e51f3145c311f6019aedef65d8d9513.jpg)

![](images/9299c4b41b8bd8064ac21d6178b6769e84810d069cc4a0daa83db85ec0f34629.jpg)  
Figure 5: Pairwise joint intervention effects under distribution shifts. Left: Joint effects versus additive expectations. Right: Pairwise interaction effects, with positive and negative values indicating reinforcement and antagonism, respectively.

## 4.5 Ablation Study

Table 4 presents an ablation study of FaShift under both ID and OOD settings. Removing factorization causes a substantial performance drop across all datasets. In this variant, the entire multimodal representation is treated as a single factor. Removing shift diagnosis further degrades performance, indicating that factorization alone is insufficient without characterizing factor-wise shift relevance and cross-environment stability. This variant directly preserves all learned factors, i.e., $w _ { k } = 1$ for all factors, without intervention-based shift diagnosis. Finally, Uniform Treatment retains both factorization and shift diagnosis but applies a shared treatment weight, i.e., $w _ { k } = \bar { w }$ for all factors. Its inferior OOD performance highlights the importance of factor-wise adaptive treatment under heterogeneous shifts.

Uniform Treatment outperforms the two ablated variants but remains consistently inferior to FaShift, particularly under OOD settings. This demonstrates that different factors require different treatments based on their intervention effects and stability. Overall, the results validate the complete factor-wise shift treatment pipeline, with factorization enabling fine-grained analysis, shift diagnosis identifying heterogeneous shifts, and adaptive treatment selectively retaining factors according to their properties.

## 4.6 Factor Shift Analysis

As shown in Figure 4, factors exhibit substantially different intervention effects and crossenvironment stability under comparable levels of shift. For example, $F _ { 4 }$ experiences the largest shift and also shows the strongest intervention effect, but its effect is less stable across environments. In contrast, $F _ { 0 }$ exhibits a relatively large shift and a strong intervention effect while maintaining higher stability. Other factors, such as $F _ { 2 }$ and $F _ { 3 }$ , undergo smaller shifts but show close intervention effects and high cross-environment stability. Based on their intervention effects and cross-environment stability, our framework classifies $F _ { 1 } , F _ { 3 } , F _ { 5 }$ as stable predictive factors, $F _ { 2 }$ as a nuisance factor, and $F _ { 0 } , F _ { 4 }$ as unstable predictive factors, thereby guiding their selective treatment.

We further investigate whether the effects of individual factors combine additively. Figure 5 compares pairwise joint intervention effects with their additive expectations. The joint effects deviate noticeably from the additive baseline, with both positive and negative interactions observed. Specifically, 53.3% of factor pairs exhibit positive interactions, while 46.7% exhibit negative interactions, with a mean absolute interaction magnitude of 0.084. The interaction map reveals heterogeneous cross-factor dependencies, including reinforcing and antagonistic effects, suggesting nonadditive factor interactions under distribution shifts and providing complementary evidence for factorlevel shift analysis.

## 4.7 Factor-wise Shift Heterogeneity

Figure 6 shows substantial heterogeneity in crossenvironment distribution shifts across the learned factors. $F _ { 4 }$ exhibits the largest shift magnitude under speaker-based environment changes, indicating that its distribution is particularly sensitive to speaker-specific variations. Although the factors are not explicitly assigned semantic meanings, this pattern suggests that $F _ { 4 }$ may encode speaker-dependent behavioral or expressive characteristics. $F _ { 0 }$ also shows a relatively pronounced shift, whereas $F _ { 2 } , F _ { 5 }$ exhibit comparatively smaller changes across environments. These results reveal heterogeneous sensitivities of individual factors to speaker-induced shifts. Since shift magnitude alone cannot determine their predictive relevance or stability, we jointly consider shift magnitude, intervention effects, and cross-environment stability to guide selective factor treatment.

![](images/4aa42fff55b0a1c95af02fc88e48296980e1c24f6f96dfa3cbf702aa5516df8a.jpg)  
Figure 6: Cross-environment shift magnitude across the learned factors.

## 5 Conclusion

In this work, we challenge the conventional view that distribution shifts in multimodal learning should be universally suppressed. We introduce an intervention-aware shift framework that factorizes multimodal representations and characterizes each factor by its distributional shift, intervention effect, and cross-environment stability. Based on these properties, the framework selectively suppresses irrelevant shifts while preserving or adapting to task-relevant ones. Extensive experiments across multiple multimodal benchmarks demonstrate that FaShift improves generalization to unseen environments while retaining sensitivity to meaningful changes. These results advocate moving beyond invariance toward factor-wise selective invariance, where each shift is treated according to its predictive relevance and cross-environment stability.

## Limitations

Despite the effectiveness of FaShift, several limi tations remain. First, the intervention-based analysis relies on latent factor replacement to estimate predictive sensitivity. Although this provides a practical way to characterize how factor perturba tions affect model predictions, it does not establish formal causal effects or guarantee causal identifiability. Future work could incorporate explicit causal structures or stronger intervention mechanisms to further validate the causal interpretation of factor-level shifts. Second, the learned latent factors are not explicitly aligned with predefined semantic concepts, and their identifiability may depend on the factorization objective and model initialization. While our empirical results demonstrate their utility for shift analysis, a more principled characterization of factor semantics and identifiability would strengthen the framework. Third, some experiments rely on pseudo-environments constructed from learned factor representations to simulate distribution shifts. Although we additionally evaluate FaShift under naturally structured temporal shifts, broader evaluation across real-world environment changes would provide stronger evidence of its generality. Finally, the current treatment policy relies on empirically selected thresholds to distinguish predictive relevance and stabil ity, and the joint factor interactions are currently used for analysis rather than directly incorporated into the treatment policy. Developing adaptive treatment criteria and explicitly modeling higher-order factor interactions are promising directions for fu ture work.

## References

Liang Cao, Weide Liu, Yan Qin, Zhenghua Chen, Zhuo Chen, and Bhushan Gopaluni. 2026. Sparse causal latent features for robust multimodal learning under distribution shifts. In Companion Proceedings of the ACM Web Conference 2026, pages 1366–1372.

Yiyang Fang, Wenke Huang, Guancheng Wan, Kehua Su, and Mang Ye. 2025. Emoe: Modality-specific enhanced dynamic emotion experts. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14314–14324.

Arthur Gretton, Olivier Bousquet, Alexander Smola, and Bernhard Schölkopf. 2005. Measuring statistical dependence with hilbert-schmidt norms. In International Conference on Algorithmic Learning Theory, pages 63–77. Springer.

Zhuo He, Shuang Li, Wenze Song, Longhui Yuan, Jian Liang, Han Li, and Kun Gai. 2025. Learning time-aware causal representation for model generalization in evolving domains. arXiv preprint arXiv:2506.17718.

Changqin Huang, Jili Chen, Qionghao Huang, Shijin Wang, Yaxin Tu, and Xiaodi Huang. 2025. Atcaf: Attention-based causality-aware fusion network for multimodal sentiment analysis. Information Fusion, 114:102725.

Yong Li, Yuanzhi Wang, and Zhen Cui. 2023. Decoupled multimodal distilling for emotion recognition. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6631– 6640.

Xusheng Liang, Lihua Zhou, Nianxin Li, Miao Xu, Ziyang Song, Dong Yi, Jinlin Wu, Jiawei Ma, Hongbin Liu, Zhen Lei, et al. 2026. Multimodal causalitydriven representation learning for generalizable medical image segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13670–13679.

Sara Magliacane, Thijs Van Ommen, Tom Claassen, Stephan Bongers, Philip Versteeg, and Joris M Mooij. 2018. Domain adaptation by using causal inference to predict invariant conditional distributions. Advances in neural information processing systems, 31.

Sijie Mai and Shiqin Han. 2026a. Careflow: Cyclic adaptive rectified flow for multimodal fusion. arXiv preprint arXiv:2602.19140.

Sijie Mai and Shiqin Han. 2026b. Learning invariant modality representation for robust multimodal learning from a causal inference perspective. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 45698–45721.

Sijie Mai, Ya Sun, Aolin Xiong, Ying Zeng, and Haifeng Hu. 2024. Multimodal boosting: Addressing noisy modalities and identifying modality contribution. IEEE Transactions on Multimedia, 26:3018–3033.

Sijie Mai, Ying Zeng, and Haifeng Hu. 2023. Multimodal information bottleneck: Learning minimal sufficient unimodal and multimodal representations. IEEE Transactions on Multimedia, 25:4121–4134.

Teng Sun, Wenjie Wang, Liqaing Jing, Yiran Cui, Xuemeng Song, and Liqiang Nie. 2022. Counterfactual reasoning for out-of-distribution multimodal sentiment analysis. In Proceedings of the 30th ACM International Conference on Multimedia, pages 15–23.

Takeshi Teshima, Issei Sato, and Masashi Sugiyama. 2020. Few-shot domain adaptation by causal mechanism transfer. In International Conference on Machine Learning, pages 9458–9469. PMLR.

Bichen Wang, Yixin Sun, Yanyan Zhao, and Bing Qin. 2025a. Beyond snapshots: A multimodal user-level dataset for depression detection in dynamic social media streams. In Proceedings ofthe 33rd ACM International Conference on Multimedia, pages 12912– 12918.

Pan Wang, Qiang Zhou, Yawen Wu, Tianlong Chen, and Jingtong Hu. 2025b. Dlf: Disentangled-languagefocused multimodal sentiment analysis. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 21180–21188.

Siyin Wang, Jie Zhou, Qin Chen, Qi Zhang, Tao Gui, and Xuan-Jing Huang. 2024. Domain generalization via causal adjustment for cross-domain sentiment analysis. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 5286–5298.

Shuwei Wu, Dong He, Xiaofeng Wang, Lei Wang, and Jian Dang. 2025. Enriching multimodal sentiment analysis through textual emotional descriptions of visual-audio content. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 1601–1609.

Xiongye Xiao, Gengshuo Liu, Gaurav Gupta, Defu Cao, Shixuan Li, Yaxing Li, Tianqing Fang, Mingxi Cheng, and Paul Bogdan. 2024. Neuro-inspired information-theoretic hierarchical perception for multimodal learning. In International Conference on Learning Representations.

Binghui Xie, Yongqiang Chen, Jiaqi Wang, Kaiwen Zhou, Bo Han, Wei Meng, and James Cheng. 2024. Enhancing evolving domain generalization through dynamic latent representations. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 16040–16048.

Wenmeng Yu, Hua Xu, Fanyang Meng, Yilin Zhu, Yixiao Ma, Jiele Wu, Jiyun Zou, and Kaicheng Yang. 2020. Ch-sims: A chinese multimodal sentiment analysis dataset with fine-grained annotation of modality. In Proceedings ofthe 58th Annual Meeting of the Association for Computational Linguistics, pages 3718–3727. Association for Computational Linguistics.

Wenmeng Yu, Hua Xu, Ziqi Yuan, and Jiele Wu. 2021. Learning modality-specific representations with selfsupervised multi-task learning for multimodal sentiment analysis. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pages 10790–10797.

Zhongqi Yue, Qianru Sun, Xian-Sheng Hua, and Hanwang Zhang. 2021. Transporting causal mechanisms for unsupervised domain adaptation. In 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pages 8579–8588. IEEE.

Amir Zadeh, Rowan Zellers, Eli Pincus, and Louis-Philippe Morency. 2016. Mosi: Multimodal corpus of sentiment intensity and subjectivity analysis in online opinion videos. arXiv preprint arXiv:1606.06259.

AmirAli Bagher Zadeh, Paul Pu Liang, Soujanya Poria, Erik Cambria, and Louis-Philippe Morency. 2018. Multimodal language analysis in the wild: Cmumosei dataset and interpretable dynamic fusion graph. In Proceedings of the 56th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 2236–2246. Association for Computational Linguistics.

Hanlei Zhang, Qianrui Zhou, Hua Xu, Jianhua Su, Roberto Evans, and Kai Gao. 2025. Multimodal classification and out-of-distribution detection for multimodal intent understanding. IEEE Transactions on Multimedia.

Guowei Zhong, Ruohong Huan, Mingzhen Wu, Ronghua Liang, and Peng Chen. 2025. Towards robust multimodal emotion recognition under missing modalities and distribution shifts. arXiv preprint arXiv:2506.10452.

## A More Implementation Details

Our framework follows a learn–diagnose–refine paradigm. First, we learn a factorized multimodal representation by decomposing the joint representation into non-redundant latent factors and optimizing them for the target task. After training, the encoder and prediction head are frozen. We then perform training-free shift diagnosis: for each factor, we measure its distributional shift across environments using a distribution discrepancy (e.g., MMD), intervene on the shifted factor while keeping other factors fixed, and quantify its interventionbased predictive relevance by the resulting prediction change. The consistency of intervention effects across environment pairs is further used to assess cross-environment stability. Based on these profiles, shifted factors are selectively treated: nuisance shifts are suppressed, stable consequential shift shifts are preserved, and unstable consequential shift shifts are adaptively refined. This design avoids universal invariance and instead retains or adjusts environmental variations according to their relevance to the task. The main hyperparameter settings are summarized in Table 5.

In summary, Stage I requires training, whereas Stage II performs factor-wise shift diagnosis without additional parameter training. Stage III applies the resulting treatment rules without introducing an additional adaptation network. For fair comparison, all methods use the same data splits and evaluation protocols. Unless otherwise stated, each experiment is repeated with 10 random seeds, and we report the average performance.

Environment Construction. We instantiate environments according to the natural data-generating structure of each dataset. For MOSI, MOSEI and CH-SIMS, each speaker defines an environment, capturing speaker-specific linguistic, acoustic, and visual characteristics. These environment definitions provide observed distributional contrastsfor subsequent factor-level shift diagnosis without using prediction labels to construct environments. Formally, for speaker $u ,$ we define

$$
E _ { u } = \{ x _ { u , v } \} _ { v = 1 } ^ { N _ { u } } ,\tag{18}
$$

where $x _ { u , v }$ denotes the v-th video sample of speaker u, and $N _ { u }$ is the total number of video samples associated with that speaker. Environment pairs are constructed from distinct speakers. The speaker-level target labels are used only for prediction and are not involved in environment construction.

For MUD3, since exact timestamps are unavailable but videos from each user are chronologically ordered, we group every 10 consecutive videos from the same user into an environment. Formally, we define

$$
E _ { u , t } = \{ x _ { u , v } \mid 1 0 ( t - 1 ) < v \leq \operatorname * { m i n } ( 1 0 t , N _ { u } ) \} ,\tag{19}
$$

where $x _ { u , v }$ denotes the v-th video of user u in chronological order, $N _ { u }$ is the total number of videos associated with user u, and t indexes the temporal environments. This construction approximates temporal variations in multimodal behavioral characteristics using the chronological order of video samples.

Design Principles for Factorization. Stage I does not aim to directly identify predictive factors. Instead, it learns a set of predictive, non-redundant, and intervenable latent factors. Stage II then leverages environmental variations and factor-level interventions to characterize their stability and shift properties. The factorization is guided by four principles:

1. Task relevance. The factorized representation should preserve the task-relevant information in the original multimodal representation, such that $g ( Z )$ maintains comparable predictive performance to $g ( H )$ .

<table><tr><td>Dataset</td><td>K</td><td>M</td><td> $\tau _ { s }$ </td><td> $\tau _ { c }$ </td><td> $\tau _ { v }$ </td><td> $\lambda _ { d e s c }$ </td></tr><tr><td>MOSI</td><td>6</td><td>8</td><td>0.02</td><td>0.2</td><td>0.6</td><td>500</td></tr><tr><td>MOSEI</td><td>6</td><td>8</td><td>0.02</td><td>0.2</td><td>0.6</td><td>500</td></tr><tr><td>CH-SIMS</td><td>8</td><td>4</td><td>0.05</td><td>0.3</td><td>0.6</td><td>400</td></tr><tr><td>MUD3</td><td>8</td><td>6</td><td>0.05</td><td>0.3</td><td>0.6</td><td>200</td></tr></table>

Table 5: Hyperparameter settings of FaShift across different multimodal datasets.
<table><tr><td></td><td colspan="2">ID</td><td colspan="2">OOD</td></tr><tr><td></td><td>Acc.</td><td>F1</td><td>Acc.</td><td>F1</td></tr><tr><td>(Wang et al., 2025a)</td><td>78.3</td><td>77.3</td><td>70.9</td><td>70.7</td></tr><tr><td>FaShift(ours)</td><td>82.4</td><td>82.7</td><td>74.1</td><td>74.4</td></tr></table>

Table 6: Results of MUD3 datasets.

2. Low redundancy. Different factors should capture sufficiently distinct information. We therefore encourage low statistical dependence between factor pairs using the HSIC-based regularization.

3. Intervenability. Each factor should constitute a distinct representation unit that can be individually replaced or modified, enabling factor-level interventions for subsequent shift analysis.

4. No predefined semantics. We do not impose semantic labels or modality-specific meanings on individual factors. Each factor is learned from the downstream task and may capture an arbitrary predictive pattern emerging from the multimodal input.

Reference Factor Construction. The reference factor $Z _ { k } ^ { a , * }$ is obtained by averaging the corresponding $Z _ { k }$ representations of the selected counterparts:

$$
Z _ { k } ^ { a , * } = \frac { 1 } { M } \sum _ { j \in \mathcal { N } _ { k } ^ { a } ( b ) } Z _ { k } ^ { j } ,\tag{20}
$$

where $\mathcal { N } _ { k } ^ { a } ( b )$ denotes the set of top-M matched samples from $E _ { a }$ . This averaged factor is used to replace $Z _ { k } ^ { b }$ while keeping all non-intervened factors $Z _ { - k } ^ { b }$ unchanged.

Factorized Representation. We use a pretrained multimodal encoder (Qwen3-Embedding- $\mathbf { \mathcal { A } B ^ { 1 } } )$ to obtain a joint representation $H \in \mathbb { R } ^ { d }$ , which is further decomposed into K compact factors, with each factor having dimension $d _ { f }$ . In our experiments, we set $K = 6$ and $d _ { f } = 3 2$ . The factorization module is trained jointly with the prediction head, while the resulting factorized representation is fixed during the subsequent factor-wise shift analysis.

Distribution Shift Estimation. For each environment $E _ { a }$ , we collect the empirical samples of the k-th factor as:

$$
\mathcal { Z } _ { k } ^ { a } = \{ z _ { k , i } : E _ { i } = E _ { a } \} .\tag{21}
$$

We then quantify the cross-environment shift using MMD (Maximum Mean Discrepancy):

$$
D _ { k } ^ { a , b } = \mathrm { M M D } \left( \mathcal { Z } _ { k } ^ { a } , \mathcal { Z } _ { k } ^ { b } \right) .\tag{22}
$$

We quantify the distributional shift of each factor using the maximum mean discrepancy (MMD) between its empirical distributions across environments. For each factor $z _ { k }$ , we compare its representations from a pair of environments and aggregate the pairwise MMD scores across all environment pairs.

Counterpart Matching. For each target sample from environment $E _ { b }$ and factor $z _ { k } .$ , we retrieve the top-M nearest samples from the reference environment $E _ { a }$ based on the similarity of their nonintervened representations $z _ { - k }$ . Specifically, we use cosine similarity to measure the similarity between the target representation and each candidate in $E _ { a } ,$ and select the M candidates with the highest similarity.

Counterfactual Intervention. All counterfactual interventions are performed without updating model parameters. For each shifted factor, we replace only the target factor with its referenceenvironment counterpart while keeping all remaining factors unchanged. The factual and counterfactual samples are then passed through the same prediction head to measure the intervention- induced prediction change.

Intervention Effect. We quantify the intervention effect $\Delta _ { k }$ as the $\ell _ { 1 }$ distance between the factual and counterfactual prediction probability vectors. This measure captures the magnitude of prediction changes induced by intervening on factor $F _ { k } ,$ providing a finer-grained assessment of its predictive influence than binary label flipping.

## B More Results

Results under Temporal Shifts. Table 6 presents the performance of FaShift on MUD3 under different environment settings. For the MUD3 dataset, we construct environments according to recording time. Specifically, samples are chronologically partitioned into distinct temporal groups, with each group treated as an environment. This construction captures temporal variations in multimodal characteristics and induces realistic distribution shifts across environments, providing a natural setting for evaluating model robustness to temporal changes.

<table><tr><td>Dataset</td><td>Environment</td><td>ID Acc-2 ↑ F1 ↑ Acc-2 ↑ F1 ↑</td><td>OOD</td><td></td></tr><tr><td>MOSEI</td><td>Observed Pseudo</td><td>88.2 82.5</td><td>88.3 78.5 82.6 75.7</td><td>78.4 74.5</td></tr><tr><td>CH-SIMS</td><td>Observed Pseudo</td><td>83.3 79.1</td><td>83.3 79.6</td><td>75.4 75.2 70.6 70.3</td></tr></table>

Table 7: Comparison between observed and pseudoenvironments for factor-wise shift diagnosis. For pseudo-environments, observed environmental attributes are excluded, and environments are constructed solely from Stage-I factor activation profiles.

![](images/b1ee248d79221c8e9e8320b50aa85da34e5c2c437fbf77142c7959a15f72acdf.jpg)  
Figure 7: Hyperparameter sensitivity of FaShift under ID and OOD settings. We vary the number of nearest neighbors $M .$ , intervention-effect threshold $\tau _ { c } ,$ and stability threshold $\tau _ { v }$ , while keeping other hyperparameters fixed.

FaShift consistently outperforms the compared method in all evaluation metrics, demonstrating improved predictive performance in both indistribution and distribution-shifted settings. Notably, FaShift maintains a clear advantage under OOD evaluation, indicating that its factor-wise shift treatment remains effective when multimodal characteristics vary across environments. These results further support the generalizability of FaShift to temporally structured distribution shifts beyond the benchmark datasets considered in the main experiments.

Pseudo-Environment Analysis. Pseudoenvironments provide an alternative operational grouping when reliable environmental attributes are unavailable. Table 7 evaluates FaShift using representation-derived pseudo-environments without relying on observed environmental attributes. Although pseudo-environments result in a moderate performance drop compared with naturally defined environments, FaShift maintains competitive OOD performance on both MOSEI and CH-SIMS v2. This indicates that factor-wise shift diagnosis and adaptive treatment remain effective when explicit environmental attributes are unavailable. The performance gap further suggests that naturally defined environments provide more informative heterogeneity, while pseudo-environments offer a viable alternative for datasets lacking reliable environmental attributes.

<table><tr><td>K</td><td>Avg. Factor Corr</td><td>OOD Acc-2 (%)</td></tr><tr><td>2</td><td>0.65</td><td>74.6</td></tr><tr><td>4</td><td>0.35</td><td>76.8</td></tr><tr><td>6</td><td>0.15</td><td>79.5</td></tr><tr><td>8</td><td>0.38</td><td>75.1</td></tr></table>

Table 8: Sensitivity of FaShift to the number of latent factors K on MOSEI.

Hyperparameter Sensitivity. Figure 7 evaluates the sensitivity of FaShift to the key hyperparameters $M , \tau _ { c } ,$ and $\tau _ { v }$ , while Table 8 investigates the effect of the number of latent factors K. The performance remains relatively stable across a broad range of values under both ID and OOD settings, indicating that FaShift is not highly sensitive to the choice of hyperparameters. In particular, the selected values of $M = 8 , \tau _ { c } = 0 . 2$ , and $\tau _ { v } = 0 . 6$ achieve consistently strong performance without requiring fine-grained tuning. These results suggest that the proposed factor-wise shift diagnosis and adaptive treatment are robust to moderate variations in the intervention and stability criteria.