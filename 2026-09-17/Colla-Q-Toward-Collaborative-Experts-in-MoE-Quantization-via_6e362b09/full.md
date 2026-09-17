# Colla-Q: Toward Collaborative Experts in MoE Quantization via Minimax Precision Balancing

Eunju Shin Ajou University, South Korea eunjushin@ajou.ac.kr

Jongbin Ryu\* Ajou University, South Korea jongbinryu@ajou.ac.kr

## Abstract

In this paper, we present a Mixture-of-Experts (MoE) quantization method based on activation entropy. Although quantization reduces memory and computational costs, it can substantially degrade performance. In particular, performance decline is pronounced in quantized MoE models, where individual experts have a small number of parameters that are sensitive to low-bit representation. Considering that MoE operates as an ensemble model with collaborative contributions from routed experts, a significant performance decline of a particular expert due to quantization can harm model performance. Therefore, we propose Colla-Q, a bitallocation framework to maintain balanced performance across experts through an activationentropy-based bit-width allocation algorithm. This approach encourages each expert to operate collaboratively in the quantized model, thereby 1) improving the overall MoE performance and 2) reducing the dependence on the calibration dataset. Since uniformly adjusting each expert’s performance facilitates robustness and stability of the MoE model, the proposed MoE quantization method can generalize more consistently across different calibration datasets. Our code is available at: https: //github.com/mmai-laboratory/Colla\_Q

## 1 Introduction

In recent large language models (LLMs), MoE has been widely adopted as a sparse architecture to reduce computational overhead (Jiang et al., 2024; Muennighoff et al., 2025). It reduces overhead by routing input tokens to relevant experts during inference (Shazeer et al., 2017). However, despite this efficiency, MoE models require substantial memory to load all expert parameters, including those that are not routed. This leads to excessive memory requirements as the model size increases. Therefore, mixed-precision quantization is an effective method to address this limitation of MoE. However, mixed-precision quantization methods developed for dense LLMs are not suitable for MoE due to the unique structure of MoE architectures.

![](images/60e446fbe57986694d0a482af423e32b997d1a7af748f50a99bb20eaec07695c.jpg)  
(a) Accuracy of ours and baseline

![](images/aba27a066062ded0279c4f85c08b4f8951b9fc5edf5b7c33d36fe55c2d36b7bb.jpg)  
(b) Performance across experts after quantization  
Figure 1: Preliminary study supporting our approach. The baseline refers to the quantized model with randomly assigned expert bit-widths. (a) Zero-shot performance (%) of Mixtral 8×7B under our method and baseline across different bit-widths. (b) The baseline results in imbalanced expert performance after quantization, whereas our method allocates more bits to preserve similar performance across experts. Strong experts tend to degrade less at low bit-widths. The x-axis shows the experts and their assigned bit-widths, and the y-axis shows the expert performance measured by our metric.

Unlike dense LLMs, MoE models consist of individual experts with relatively few parameters, each tending to specialize in different token contexts. As a result, experts are highly sensitive to low-bit quantization, and some experts experience significant performance degradation (Dong et al., 2019). This degradation is non-uniform across experts, which can amplify performance differences and impair the quality of the aggregated MoE output. This phenomenon is consistent with an ensemble perspective: previous studies (Caruana et al., 2004;

![](images/bc1dc5cd6a621f58ef967e32c45a98929c8c224b4df6d07c86a99cf55c0ff370.jpg)

![](images/cd3d414b86a12d4a3006d38a2caa1e0f945d4fc3b67e68f86141f27f5a3a0bb7.jpg)  
Figure 2: Conceptual illustration of the proposed method. (a) Allocating fewer bits to a lower-performing expert can significantly degrade its performance, leading to suboptimal performance of the aggregated output of the MoE block. (b) In contrast, our method allocates more bits to a lower-performing expert, thereby improving the aggregated output and preserving the overall quality of the MoE block output. Agg. denotes the aggregated MoE block output obtained by combining the outputs of routed experts.

An and Xu, 2023; Yao et al., 2025) show that a single weak learner can hinder collaborative expert contributions and, in turn, degrade the overall performance of the aggregated model. This perspective motivates our view of MoE architectures, as they aggregate the contributions of routed experts in an ensemble-like manner. We provide experimental results supporting these observations in Figure 1, which demonstrates the relationship between performance balancing across experts and overall model performance. Accordingly, this observation motivates assigning more appropriate bit-widths to weak experts in MoE quantization to reduce further performance degradation.

Previous studies (Huang et al., 2025; Li et al., 2024; Duanmu et al., 2025) have made similar efforts to optimize MoE bit-width allocation. They allocate different bit-widths to experts based on routing frequency. Huang et al. (2025) and Li et al. (2024) measure each expert’s importance by counting how often tokens are routed to it on a calibration dataset. Frequently routed experts are deemed more important due to higher utilization and therefore receive more bits. Duanmu et al. (2025) also uses per-expert routing results to guide bit-width allocation. Although these methods are effective for MoE quantization, they do not directly consider preserving the performance of weak experts. As shown in Figures 1b and 2, allocating more bits to weak experts suggests that promoting similar expert performance after quantization helps preserve the aggregated MoE block output. Moreover, their estimates of expert importance depend on the calibration distribution, as shown in Figure 3.

Therefore, we emphasize that maintaining balanced expert performance is important, as it supports collaborative expert contributions and ultimately improves overall model performance. The key question at this point is how to estimate expert performance for bit-width allocation. Since individual experts do not directly generate tokens, we need a label-free way to estimate expert performance from output activations. To this end, we propose an activation-entropy metric that estimates expert performance through the lens of activation variance, grounded in differential entropy theory. Based on this metric, we further introduce a minimax precision balancing algorithm to preserve balanced expert performance. We refer to this overall framework as Colla-Q.

Colla-Q also shows better generalization across calibration datasets. For practical quantization, this ability to generalize across calibration datasets is important because the inference distribution is often unknown in advance. However, routing frequency and weights used for bit-width allocation in MoE quantization tend to reflect properties of the calibration domain more strongly, as shown in Table 4. For example, when calibrating a model on a dataset with math problems, these routing-based metrics can place greater emphasis on experts specialized in numerical reasoning. By contrast, as shown in Figure 3, our method is less sensitive to the calibration domain because it allocates bitwidths based on expert performance estimated from output activations. As a result, this yields more consistent bit-width assignments and smaller performance variation across calibration datasets.

## 2 Related Work

## 2.1 Quantization for LLMs with Activation

Post-training quantization (PTQ) is a method that quantizes pre-trained models without the need for additional training, effectively reducing the memory footprint of large language models (LLMs). GPTQ (Frantar et al., 2023) is a representative PTQ method that enables low-bit quantization via layerwise optimization. It performs layer-wise optimization to minimize quantization error for each layer using calibration data. Meanwhile, recent methods such as LLM.int8() (Dettmers et al., 2022), OWQ (Lee et al., 2024), and AWQ (Lin et al., 2024) have shown that performance degradation can be reduced using activation statistics. In particular, AWQ identifies important channels based on activation statistics and scales their weights to reduce performance degradation during quantization. These results suggest that activation-derived signals are useful for enhancing the effectiveness of weight quantization.

## 2.2 Quantization for MoE-LLMs: From Allocation to Precision Balancing

Quantization is widely applied to MoE models to reduce the memory footprint while mitigating performance degradation. Across several studies (Li et al., 2024; Huang et al., 2025; Duanmu et al., 2025), mixed precision is applied to different MoE components, such as MoE blocks, experts, and linear layers within each expert. QuantMoE-Bench (Li et al., 2024) presents a structure-aware strategy and shows that mixed precision outperforms uniform bit-width under the same budget. Building on this strategy, PMQ (Huang et al., 2025) allocates per-expert bit-width using routing statistics from a calibration dataset, such as routing frequency and weight. MxMoE (Duanmu et al., 2025) considers linear-layer quantization sensitivity, expert activation patterns, and hardware constraints to allocate bit-width. Overall, prior work has designed mixed-precision bit-width for different MoE components. Most of these approaches determine expert importance using routing statistics from a calibration dataset, implicitly assuming that more frequently activated experts are more critical. In contrast, rather than allocate bit-width based on routing statistics, our minimax precision balancing approach actively equilibrates expert performance to maintain the integrity of the overall ensemble.

## 3 Method

Previous studies on ensemble models (Caruana et al., 2004; An and Xu, 2023; Yao et al., 2025) have demonstrated that when a single weak learner performs at an exceptionally low level, the performance of the aggregated ensemble model drops significantly. To address this performance drop in the quantized MoE model, we adopt an ensemble perspective as the motivation for our approach. Building on this perspective, we present Colla-Q, a bit-allocation framework with two components: 1) an activation-entropy metric for expert performance estimation and 2) a minimax precision balancing algorithm for bit allocation.

## 3.1 MoE from Ensemble Perspective

Formally, the output of a MoE block can be understood as a dynamic ensemble output of k routed experts. Given an input x, the layer output $\hat { \mathbf { y } }$ is derived from the weighted ensemble of the expert outputs as:

$$
\hat { \mathbf { y } } = \sum _ { e \in \hat { E } } g _ { e } ( \mathbf { x } ) \cdot e ( \mathbf { x } ) ,\tag{1}
$$

where $\hat { E }$ is the set of routed experts, $e ( \mathbf { x } )$ refers to the expert output, and $g _ { e } ( \mathbf { x } )$ indicates the corresponding routing weight. Let $R _ { e }$ represent the expected error of an expert, which quantifies its individual performance deficiency. Assuming the errors of distinct experts are uncorrelated, we can approximate the global error $\mathcal { E } _ { e n s }$ of the aggregated ensemble by the weighted sum of individual errors:

$$
\mathcal { E } _ { e n s } \approx \sum _ { e \in \hat { E } } ( \bar { g } _ { e } ( \mathbf { x } ) ) ^ { 2 } R _ { e } .\tag{2}
$$

In order to enhance the stability and robustness of the model, our goal is to minimize the global ensemble error $\mathcal { E } _ { e n s } .$ . Considering a fixed total amount of expert errors $\begin{array} { r } { ( i . e . , \sum _ { e \in \hat { E } } R _ { e } = \mathcal A ) } \end{array}$ for simplicity, we assume approximately uniform routing weights in expectation over many tokens, $\begin{array} { r } { \bar { g } _ { e } \triangleq \mathbb { E } _ { \mathbf { x } } [ g _ { e } ( \mathbf { x } ) ] \approx \frac { 1 } { | \hat { E } | } } \end{array}$ as the load-balancing objective, which encourages balanced expert utilization on average over many tokens (shown in Appendix A.1). Under these assumptions, we can structure the optimization problem as minimizing the following convex objective:

$$
\operatorname* { m i n } _ { \{ R _ { e } \} } \sum _ { e \in \hat { E } } R _ { e } ^ { 2 } \quad \mathrm { s . t . } \quad \sum _ { e \in \hat { E } } R _ { e } = \mathcal { A } .\tag{3}
$$

Since the objective function is strictly convex, we obtain the global minimum only when the errors across all routed experts are identical:

$$
R _ { e _ { i } } \approx R _ { e _ { j } } \approx \frac { \mathcal { A } } { | \hat { E } | } \quad \forall e _ { i } , e _ { j } \in \hat { E } .\tag{4}
$$

This analysis suggests that differences in expected errors among experts can increase the lower bound of the global ensemble error. Therefore, according to an ensemble perspective, we are motivated to encourage similar performance across all experts to strengthen the robustness of MoE.

## 3.2 Activation-Entropy of Experts

We introduce the activation-entropy metric in Colla-Q, which estimates expert performance without using token labels. Typically, language-model performance is measured by cross-entropy, which compares generated tokens with token labels. However, this approach is not feasible in our method, because experts do not directly generate tokens; instead, we evaluate each expert using its continuous-valued output activations $( i . e . ,$ expert FFN outputs), without access to token labels. Therefore, we employ the principle of differential entropy (Cover and Thomas, 2006; Malinin and Gales, 2018) that quantifies uncertainty to derive the entropy of experts, as follows:

$$
H ( x ) = \frac { 1 } { 2 } \ln ( 2 \pi e \sigma ^ { 2 } ) ,\tag{5}
$$

where $H ( x )$ is the differential entropy, π and e denote the constant values, and $\sigma ^ { 2 }$ represents the variance of x. Eq. 5 holds when x follows the Gaussian approximation $p ( x ) \sim \mathcal { N } ( \mu , \sigma ^ { 2 } )$ , and the activations generated by each expert are reasonably well approximated by this distribution (shown in Appendix A.2). Therefore, by applying Eq. 5, we can effectively express expert activation-entropy in terms of activation variance. Based on this, we define the activation-entropy-based proxy $\rho$ for expert performance by the following ratio:

$$
\rho = \frac { \exp ( 2 H _ { w i t h i n } ) } { \exp ( 2 H _ { t o t a l } ) } \propto \frac { \sigma _ { w i t h i n } ^ { 2 } } { \sigma _ { t o t a l } ^ { 2 } } ,\tag{6}
$$

where lower values of $\rho$ are associated with better expert performance. Here $H _ { w i t h i n }$ and $H _ { t o t a l }$ are the within-channel and total entropies of activations. By Eq. 5, entropy is determined by variance, so we compute the corresponding variances as:

$$
\begin{array} { r } { \sigma _ { \mathrm { w i t h i n } } ^ { 2 } = \displaystyle \frac { 1 } { | \mathcal { T } _ { e } | | \mathcal { C } | } \sum _ { t \in \mathcal { T } _ { e } } \sum _ { c \in \mathcal { C } } \big ( A _ { t , c } - \mu _ { c } \big ) ^ { 2 } , } \\ { \sigma _ { \mathrm { t o t a l } } ^ { 2 } = \displaystyle \frac { 1 } { | \mathcal { T } _ { e } | | \mathcal { C } | } \sum _ { t \in \mathcal { T } _ { e } } \sum _ { c \in \mathcal { C } } \big ( A _ { t , c } - \bar { \mu } \big ) ^ { 2 } , } \end{array}\tag{7}
$$

where $\mathcal { T } _ { e }$ is a token set routed to expert $e , { \mathcal { C } }$ denotes channels of the activations, and $A _ { t , c }$ is the activation at token t and channel c. The channelwise and global averages are denoted as $\mu _ { c } =$ $\textstyle { \frac { 1 } { | { \mathcal { T } } _ { e } | } } \sum _ { t \in { \mathcal { T } } _ { e } } A _ { t , c }$ and $\begin{array} { r } { \bar { \mu } = \frac { 1 } { | { \mathcal T } _ { e } | | { \mathcal C } | } \sum _ { t \in { \mathcal T } _ { e } } \sum _ { c \in { \mathcal C } } A _ { t , c } . } \end{array}$ The denominator in Eq. 6, total entropy and variance, signifies the expert’s capacity to represent activations in its global activation space. It measures the dynamic range of activations across neurons, reflecting the expert’s capacity to represent diverse features for the tokens routed to the expert. The numerator in Eq. 6, the within-channel entropy and variance, quantifies the uncertainty across channels. Each channel of the activations is hypothesized to act as a dedicated feature encoder for specific input patterns, and thus an expert is expected to produce predictable responses. A high within-channel variance suggests that an expert is less predictable, reflecting high uncertainty.

A related intuition also appears in random forest (Breiman, 2001), which explains the generalization ability of the ensemble model as the strength and correlation of decision trees. In random forests, the generalization ability is quantified by the ratio of the strength, which measures the uncertainty of decision trees, and the correlation, which evaluates their representation capacity. Therefore, the proposed ratio $\rho$ can effectively estimate expert performance. Lower values of $\rho$ suggest that an expert captures its unique channel-wise patterns more consistently while achieving high representational capacity. Additional empirical analysis of activation entropy is provided in Appendix A.3.

## 3.3 Minimax precision balancing

We introduce the second component of Colla-Q, a minimax precision balancing algorithm that uses the proposed activation-entropy metric to allocate bit-widths across experts. We begin by revisiting the objective function of previous MoE quantization methods (Huang et al., 2025) as follows:

$$
\underset { \ b { \mathbf { b } } } { \operatorname* { m i n } } \sum _ { i = 1 } ^ { N } \mathcal { L } ( b _ { e _ { i } } ) \quad \mathrm { s . t . } \quad \sum _ { i = 1 } ^ { N } b _ { e _ { i } } \leq B _ { t o t a l }\tag{8}
$$

where E denotes the expert set of a single MoE block, $b _ { e }$ is a bit-width of expert e, and $B _ { t o t a l }$ stands for the total bit-width assigned for the MoE block. $\mathcal { L } ( b _ { e } )$ is an objective function that should be minimized during the mixed-precision quantization process. Therefore, according to Eq. 8, previous mixed-precision quantization methods minimize the averaged loss of the objective functions across all experts. However, our approach targets the minimization of the objective function of the worst-performing expert, as follows:

$$
\mathbf { b } ^ { * } = \underset { \mathbf { b } } { \arg \operatorname* { m i n } } \left( \underset { i = 1 } { \overset { N } { \operatorname* { m a x } } } \mathcal { L } ( b _ { e _ { i } } ) \right)\tag{9}
$$

Under this problem definition, we aim to allocate optimal bit-widths $\mathbf { b } ^ { * } = \{ b _ { e _ { 1 } } , b _ { e _ { 2 } } , . . . , b _ { e _ { N } } \}$ to N experts within a MoE block, such that expert performance is balanced after mixed-precision quantization. To support collaboration among experts, we prioritize experts based on their full-precision performance estimated using the activation-entropy metric in the objective function as:

$$
\mathcal { L } ( b _ { e _ { i } } ) = \rho _ { e _ { i } } \times \underbrace { \Vert e _ { i } ( \mathbf { x } ) - \tilde { e } _ { i } ( \mathbf { x } ; b _ { e _ { i } } ) \Vert _ { 2 } ^ { 2 } } _ { \mathrm { Q u a n t i z a t i o n ~ e r r o r } } ,\tag{10}
$$

where $e _ { i } ( \mathbf { x } )$ denotes the full-precision output of expert $e _ { i } ,$ and $\tilde { e } _ { i } ( { \bf x } ; b _ { e _ { i } } )$ denotes the output after applying GPTQ quantization to $e _ { i }$ with bit-width $b _ { e _ { i } }$ The quantization error is also considered to capture further degradation of weak experts after quantization, which can affect the aggregated MoE output. Based on this objective function, we formalize this objective through the minimax precision balancing algorithm as shown in Algorithm 1. By recursively assigning additional bits to the worst-performing expert, our minimax precision balancing algorithm forces the ensemble into a balanced state.

As shown in Algorithm 1, after initialization, we recursively assign one additional bit to the worst-performing expert. This procedure emphasizes that the performance of a specific expert is not significantly degraded, unlike previous bitwidth allocation approaches, which optimize all experts equally to maximize average performance. In other words, we posit that balancing expert performance ultimately benefits the MoE, even if it does not maximize the average performance. This approach, which emphasizes collaboration among experts, improves MoE performance while reducing calibration-specific overfitting in bit allocation, thereby maintaining generalizability.

Algorithm 1 Minimax precision balancing   
1: Input   
Expert set: $E = \{ e _ { 1 } , \ldots , e _ { N } \}$   
Total budget and max bit-width: $B _ { \mathrm { t o t a l } } , \ : b _ { \mathrm { m a x } }$   
Activation-entropy weights: $\{ \rho _ { e _ { 1 } } , \dots , \rho _ { e _ { N } } \}$   
2: Output   
Bit-width allocation $\mathbf { b } ^ { * } = \{ b _ { e _ { 1 } } , \dots , b _ { e _ { N } } \}$   
3: Initialization:   
∀i, $b _ { e _ { i } } \gets 1$   
B<sub>current</sub> $ \textstyle \sum _ { i = 1 } ^ { N } b _ { e _ { i } }$   
4: while $B _ { \mathrm { c u r r e n t } } < B _ { \mathrm { t o t a l } }$ do   
5: ${ \mathcal { C } } \gets \left\{ i \mid b _ { e _ { i } } < b _ { \operatorname* { m a x } } \right\} \triangleright I d e n t i f y$ experts   
below max bit-budget   
6: if ${ \mathcal { C } } = \emptyset$ then   
7: break ▷ All experts reached $b _ { \mathrm { m a x } }$   
8: for all $c \in { \mathcal { C } }$ do   
9: $\mathcal { E } _ { c } \gets \parallel e _ { c } ( \mathbf { x } ) - \tilde { e } _ { c } ( \mathbf { x } ; b _ { e _ { c } } ) \parallel _ { 2 } ^ { 2 }$   
10: $\mathcal { L } _ { c } \gets \rho _ { e _ { c } } \times \mathcal { E } _ { c }$   
11: $k ^ { * } \gets \operatorname { a r g m a x } _ { c \in \mathcal { C } } ( \mathcal { L } _ { c } ) \triangleright$ Find the worst   
performing expert   
12: $b _ { e _ { k ^ { * } } }  b _ { e _ { k ^ { * } } } + 1$ ▷ Allocate 1 bit   
13: $B _ { c u r r e n t }  B _ { c u r r e n t } + 1$   
14: return $\mathbf { b } ^ { * }$

## 4 Experiments

## 4.1 Settings

Model and datasets. We conduct extensive experiments on three MoE models: Mixtral 8×7B (Jiang et al., 2024), DeepSeek-MoE-16B-Base (Dai et al., 2024), and Phi3.5-MoE (Abdin et al., 2024). The architectural details and parameter scales of each model are provided in Appendix A.4. We evaluate our method using EleutherAI’s LM Evaluation Harness (Gao et al., 2024) on eight zero-shot benchmarks: PIQA (Bisk et al., 2020), ARC-Easy and ARC-Challenge (Clark et al., 2018), BoolQ (Clark et al., 2019), HellaSwag (Zellers et al., 2019), WinoGrande (Sakaguchi et al., 2020), MathQA (Amini et al., 2019), and MMLU (Hendrycks et al., 2021). In addition, we use four calibration datasets: C4 (Raffel et al., 2020), MathQA, BoolQ, and the French subset of Lambada\_mul\_fr (Paperno et al., 2016).

Experimental Setup. We determine the bit-width allocation for our method using 128 sequences randomly sampled from C4. For the generalizability analysis, we repeat bit-width allocation using each calibration dataset (C4, Math, French, and QA). We then evaluate the proposed method against stateof-the-art MoE quantization methods. Across all experiments, we use GPTQ for weight quantization, with 128 sequences sampled from Wikitext2 (Merity et al., 2017) as the GPTQ calibration dataset. Additional implementation details for our method and the baselines are provided in Appendix A.4.

<table><tr><td>Bits</td><td>Method</td><td>MMLU</td><td>PIQA</td><td>ARC-e</td><td>ARC-c</td><td>BoolQ</td><td>HellaS.</td><td>Wino.</td><td>MathQA</td><td>Avg. (↑)</td></tr><tr><td>16</td><td>一</td><td>67.8</td><td>83.6</td><td>84.2</td><td>56.5</td><td>85.0</td><td>84.0</td><td>76.2</td><td>41.7</td><td>72.4</td></tr><tr><td>2</td><td>Uniform</td><td>30.2</td><td>60.7</td><td>47.1</td><td>25.7</td><td>62.3</td><td>41.9</td><td>52.8</td><td>22.4</td><td>42.9</td></tr><tr><td rowspan="5">2.54</td><td>OA-GPTQ</td><td>30.3</td><td>69.6</td><td>57.0</td><td>34.7</td><td>58.1</td><td>59.9</td><td>61.3</td><td>26.6</td><td>49.7</td></tr><tr><td>BSP</td><td>25.1</td><td>62.8</td><td>49.4</td><td>29.8</td><td>53.8</td><td>51.5</td><td>56.0</td><td>23.9</td><td>44.0</td></tr><tr><td>MxMoE</td><td>54.4</td><td>79.7</td><td>74.8</td><td>50.0</td><td>79.1</td><td>78.1</td><td>71.2</td><td>33.1</td><td>65.1</td></tr><tr><td>PMQ</td><td>56.4</td><td>80.5</td><td>77.1</td><td>51.3</td><td>82.5</td><td>79.0</td><td>74.0</td><td>39.2</td><td>67.5</td></tr><tr><td>Colla-Q</td><td>59.4</td><td>80.5</td><td>79.8</td><td>54.0</td><td>84.7</td><td>78.5</td><td>74.9</td><td>36.5</td><td>68.5</td></tr><tr><td rowspan="3">2.05</td><td>MxMoE</td><td>42.3</td><td>75.4</td><td>68.6</td><td>41.5</td><td>71.4</td><td>71.7</td><td>66.8</td><td>29.9</td><td>58.4</td></tr><tr><td>PMQ</td><td>46.8</td><td>79.2</td><td>73.1</td><td>48.4</td><td>80.6</td><td>75.0</td><td>71.3</td><td>31.8</td><td>63.3</td></tr><tr><td>Colla-Q</td><td>52.9</td><td>77.8</td><td>74.9</td><td>46.4</td><td>83.3</td><td>72.4</td><td>72.0</td><td>32.3</td><td>64.0</td></tr><tr><td rowspan="3">1.57</td><td>MxMoE</td><td>28.5</td><td>61.7</td><td>49.5</td><td>29.8</td><td>64.1</td><td>46.2</td><td>60.0</td><td>24.5</td><td>45.5</td></tr><tr><td>PMQ</td><td>32.3</td><td>72.4</td><td>62.5</td><td>37.9</td><td>73.6</td><td>63.2</td><td>66.4</td><td>26.8</td><td>54.5</td></tr><tr><td>Colla-Q</td><td>36.6</td><td>71.0</td><td>63.1</td><td>36.2</td><td>75.8</td><td>59.8</td><td>66.2</td><td>26.9</td><td>54.5</td></tr></table>

Table 1: Zero-shot task performance (%) of the quantized Mixtral 8×7B on eight benchmarks under different average bit-widths and MoE quantization methods. Bits denotes the average effective bit-widths for each quantization method. Avg. reports the averaged accuracy over all tasks.
<table><tr><td rowspan="2">Bits</td><td rowspan="2">Method</td><td colspan="3">DeepSeek-16B-Base</td><td colspan="3">Phi3.5-MoE</td></tr><tr><td>MMLU</td><td>C.S Avg.</td><td>Avg.</td><td>MMLU</td><td>C.S Avg.</td><td>Avg.</td></tr><tr><td>16</td><td>-</td><td>38.0</td><td>65.1</td><td>51.6</td><td>76.6</td><td>68.2</td><td>72.4</td></tr><tr><td rowspan="5">2.54</td><td>OA-GPTQ</td><td>33.3</td><td>61.9</td><td>47.6</td><td>43.8</td><td>53.4</td><td>48.6</td></tr><tr><td>BSP</td><td>29.7</td><td>56.7</td><td>43.2</td><td>39.4</td><td>44.9</td><td>42.2</td></tr><tr><td>MxMoE</td><td>29.8</td><td>62.1</td><td>46.0</td><td>43.9</td><td>51.3</td><td>47.6</td></tr><tr><td>PMQ</td><td>32.8</td><td>61.9</td><td>47.3</td><td>50.7</td><td>57.8</td><td>54.2</td></tr><tr><td>Colla-Q</td><td>33.4</td><td>63.2</td><td>48.3</td><td>51.8</td><td>58.4</td><td>55.1</td></tr><tr><td rowspan="3">2.05</td><td>MxMoE</td><td>26.4</td><td>57.8</td><td>42.1</td><td>27.8</td><td>42.3</td><td>35.1</td></tr><tr><td>PMQ</td><td>27.0</td><td>58.0</td><td>42.5</td><td>25.4</td><td>50.5</td><td>38.0</td></tr><tr><td>Colla-Q</td><td>27.1</td><td>59.2</td><td>43.1</td><td>28.3</td><td>52.9</td><td>40.6</td></tr><tr><td rowspan="3">1.57</td><td>MxMoE</td><td>24.0</td><td>45.4</td><td>34.7</td><td>23.7</td><td>38.8</td><td>31.3</td></tr><tr><td>PMQ</td><td>22.6</td><td>54.0</td><td>38.3</td><td>23.5</td><td>41.8</td><td>32.7</td></tr><tr><td>Colla-Q</td><td>23.3</td><td>54.7</td><td>39.0</td><td>24.3</td><td>43.3</td><td>33.8</td></tr></table>

Table 2: Zero-shot performance comparison of two MoE models (DeepSeek-16B-Base and Phi3.5-MoE) under different average bit-widths and MoE quantization methods. C.S Avg. denotes the mean accuracy over the seven non-MMLU benchmarks. Avg. is the averaged value of MMLU and C.S Avg. Full results can be found in Appendix A.5.

<table><tr><td>Bits</td><td>Method</td><td>MMLU</td></tr><tr><td>16</td><td>-</td><td>70.6</td></tr><tr><td rowspan="6">2.54</td><td>OA-GPTQ</td><td>58.1</td></tr><tr><td>BSP</td><td>51.7</td></tr><tr><td>MxMoE</td><td>59.1</td></tr><tr><td>PMQ</td><td>61.2</td></tr><tr><td>Colla-Q</td><td>62.5</td></tr><tr><td></td><td></td></tr></table>

Table 3: MMLU 5-shot task performance of Mixtral 8×7B under different MoE quantization methods at an average bit-width of 2.54. Additional results under other bit-width settings are provided in the Appendix A.5.

## 4.2 Experimental Results

SOTA comparison. Table 1 compares our method with Uniform GPTQ<sup>1</sup> and state-of-the-art (SOTA) mixed-precision MoE quantization methods on Mixtral 8×7B. The results show that our method performs favorably across most average bit-width settings. A similar trend is observed for DeepSeek-16B-Base and Phi3.5-MoE in Table 2, where our method consistently performs well. Our method exhibits almost no average performance drop at an average bit-width of 2.54 bits on Mixtral 8×7B and DeepSeek-16B-Base. We verify in-context performance using MMLU 5-shot on Mixtral 8×7B (Table 3), complementing the zero-shot comparisons. The conclusions remain unchanged under the 2.54-bit setting, indicating that our bit allocation remains effective in the few-shot setting.

![](images/9cd62bb8a160481a273b9ef2db34c058cd818cf2ec1dec5ac03c93d8bcfc71c0.jpg)  
Figure 3: Performance comparison of PMQ and Ours on Mixtral 8×7B and Phi3.5-MoE quantized to 1.57-bit with different calibration datasets (C4, Math, French, QA). Each subplot corresponds to a test dataset, the x-axis denotes the calibration dataset, and the y-axis shows zero-shot accuracy. Test Average denotes the mean accuracy over all three test datasets.

Generalizability. We analyze the generalizability of quantization methods across different calibration datasets. As shown in Table 4, PMQ exhibits lower cosine similarity between routing-based factor vectors (routing frequency and routing weight) computed from different calibration datasets. This result indicates that PMQ’s routing-based metrics lead to substantially different bit-width configurations depending on the calibration dataset, as the routing-based metric can change with the calibration distribution. In contrast, our method maintains high cosine similarity, resulting in consistent bit-width configurations across calibration datasets. Figure 3 also illustrates this property by comparing quantization performance across MoE models under different calibration datasets. The performance of PMQ deteriorates considerably when the calibration dataset differs from the test dataset; however, our method shows favorable performance across most calibration datasets. These results support the generalizability of our method across calibration datasets in the quantization process.

## 4.3 Ablation Study

We provide experimental results for the two main components of our method: the proposed activationentropy metric and the minimax precision balancing algorithm. Table 5 demonstrates that the activation-entropy metric performs favorably compared to the routing-based metric used in prior quantization methods. We conduct experiments in which implementation details are identical, differing only in the metric used in the objective function to determine the bit-width configuration. Under this ablation setting, our activation-entropy metric consistently achieves strong results. We validate the effectiveness of our minimax precision balancing algorithm in Table 6, where our method compares favorably with PMQ’s allocation algorithm. Considering our method is designed to prioritize activation-entropy rather than a routing-based metric, this result aligns with the ensemble perspective of MoE. Since we enhance the worst-performing expert by recursively adding bits, we can reduce the error of the aggregated output of multiple experts. Additional results isolating the effects of the metric and allocation strategy, together with further algorithmic analysis, are provided in Appendix A.6.

<table><tr><td rowspan="2">Calibration dataset</td><td colspan="2">Mixtral 8×7B</td><td colspan="2">Phi3.5-MoE</td></tr><tr><td>PMQ</td><td>Ours</td><td>PMQ</td><td>Ours</td></tr><tr><td>C4</td><td>86.0</td><td>98.8</td><td>72.1</td><td>94.0</td></tr><tr><td>Math</td><td>85.5</td><td>98.5</td><td>46.3</td><td>91.7</td></tr><tr><td>French</td><td>86.0</td><td>98.2</td><td>57.8</td><td>91.6</td></tr><tr><td>QA</td><td>72.0</td><td>98.9</td><td>72.7</td><td>94.8</td></tr></table>

Table 4: Average cosine similarity of routing-based metric from PMQ and our metric. For each calibration dataset (row), we compute the mean cosine similarity to the other three calibration datasets, excluding selfsimilarity. Details are provided in Appendix A.5.

<table><tr><td>Bits</td><td>Metric</td><td>MMLU</td><td>PIQA</td><td>ARC-e</td><td>ARC-c</td><td>BoolQ</td><td>HellaS.</td><td>Wino.</td><td>MathQA</td><td>Avg.</td></tr><tr><td rowspan="2">2.54</td><td>PMQ</td><td>58.6</td><td>81.2</td><td>78.0</td><td>52.2</td><td>82.2</td><td>79.5</td><td>74.0</td><td>35.7</td><td>67.7</td></tr><tr><td>Colla-Q</td><td>59.4</td><td>80.5</td><td>79.8</td><td>54.0</td><td>84.7</td><td>78.5</td><td>74.9</td><td>36.5</td><td>68.5</td></tr><tr><td rowspan="2">2.05</td><td>PMQ</td><td>45.5</td><td>77.6</td><td>74.9</td><td>47.2</td><td>82.8</td><td>74.8</td><td>71.6</td><td>32.0</td><td>63.3</td></tr><tr><td>Colla-Q</td><td>52.9</td><td>77.8</td><td>74.9</td><td>46.4</td><td>83.3</td><td>72.4</td><td>72.0</td><td>32.3</td><td>64.0</td></tr><tr><td rowspan="2">1.57</td><td>PMQ</td><td>33.1</td><td>70.3</td><td>63.9</td><td>36.8</td><td>71.4</td><td>59.8</td><td>67.2</td><td>26.5</td><td>53.6</td></tr><tr><td>Colla-Q</td><td>36.6</td><td>71.0</td><td>63.1</td><td>36.2</td><td>75.8</td><td>59.8</td><td>66.2</td><td>26.9</td><td>54.5</td></tr></table>

Table 5: Zero-shot performance of Mixtral 8×7B on eight benchmarks under different average bit-widths. In comparison, we use PMQ’s routing-based metric and activation-entropy metric with minimax precision balancing.
<table><tr><td>Bits</td><td>Algorithm</td><td>MMLU</td><td>PIQA</td><td>ARC-e</td><td>ARC-c</td><td>BoolQ</td><td>HellaS.</td><td>Wino.</td><td>MathQA</td><td>Avg.</td></tr><tr><td rowspan="2">2.54</td><td>PMQ</td><td>57.6</td><td>79.5</td><td>78.4</td><td>51.9</td><td>83.5</td><td>78.8</td><td>72.4</td><td>36.4</td><td>67.3</td></tr><tr><td>Colla-Q</td><td>59.4</td><td>80.5</td><td>79.8</td><td>54.0</td><td>84.7</td><td>78.5</td><td>74.9</td><td>36.5</td><td>68.5</td></tr><tr><td rowspan="2">2.05</td><td>PMQ</td><td>49.3</td><td>77.4</td><td>72.7</td><td>45.2</td><td>81.7</td><td>73.3</td><td>71.5</td><td>33.9</td><td>63.1</td></tr><tr><td>Colla-Q</td><td>52.9</td><td>77.8</td><td>74.9</td><td>46.4</td><td>83.3</td><td>72.4</td><td>72.0</td><td>32.3</td><td>64.0</td></tr><tr><td rowspan="2">1.57</td><td>PMQ</td><td>34.2</td><td>72.1</td><td>59.1</td><td>36.2</td><td>72.3</td><td>61.3</td><td>63.4</td><td>27.2</td><td>53.2</td></tr><tr><td>Colla-Q</td><td>36.6</td><td>71.0</td><td>63.1</td><td>36.2</td><td>75.8</td><td>59.8</td><td>66.2</td><td>26.9</td><td>54.5</td></tr></table>

Table 6: Zero-shot performance of Mixtral 8×7B with different bit-width allocation algorithms. We compare PMQ’s allocation algorithm and our minimax precision balancing using the proposed activation-entropy metric.

We further examine the role of activation entropy by comparing quantization error alone with activation-entropy-weighted quantization error in Table 7. Activation entropy serves as a label-free proxy for estimating expert performance from full-precision activations, while the expert-output quantization error captures quantization sensitivity. The combined objective consistently improves performance across all average bit-widths, with a larger gain in the extreme low-bit setting, indicating the benefit of jointly considering weak experts and quantization sensitivity. Also, we analyze our design choice in Table 8, where expertwise mixed-precision allocation yields better performance across all average bit-width settings, supporting our design choice. Full results for Tables 7 and 8 can be found in Appendix A.7.

Additionally, we measure the expertcomputation efficiency of Colla-Q using low-bit expert GEMM kernels and compare it with FP16 under the same routing workload. As shown in Figure 4, Colla-Q achieves higher TFLOPS and a 1.39×–1.48× expert-computation speedup across the evaluated average bit-widths. These results suggest that, beyond reducing memory usage while preserving performance, Colla-Q can also provide practical computation gains with low-bit expert GEMM kernels. This analysis focuses on expert computation rather than end-to-end inference, and detailed experimental settings are provided in Appendix A.4.

<table><tr><td>Bits</td><td>Term</td><td>MMLU</td><td>C.S Avg.</td><td>Avg.</td></tr><tr><td>16</td><td>-</td><td>67.8</td><td>73.0</td><td>70.4</td></tr><tr><td rowspan="2">2.54</td><td>Quantization error</td><td>58.3</td><td>69.2</td><td>63.7</td></tr><tr><td>Colla-Q</td><td>59.4</td><td>69.8</td><td>64.6</td></tr><tr><td rowspan="2">2.05</td><td>Quantization error</td><td>49.9</td><td>65.8</td><td>57.8</td></tr><tr><td>Colla-Q</td><td>52.9</td><td>65.6</td><td>59.3</td></tr><tr><td rowspan="2">1.57</td><td>Quantization error</td><td>35.7</td><td>56.0</td><td>45.8</td></tr><tr><td>Colla-Q</td><td>36.6</td><td>57.0</td><td>46.8</td></tr></table>

Table 7: Zero-shot performance of Mixtral 8×7B under different metric settings. We compare quantization error with activation-entropy-weighted quantization error using the same minimax precision balancing algorithm.
<table><tr><td>Bits</td><td>Components</td><td>MMLU</td><td>C.S Avg.</td><td>Avg.</td></tr><tr><td>16</td><td></td><td>67.8</td><td>73</td><td>70.4</td></tr><tr><td rowspan="2">2.54</td><td>Linear Layer</td><td>57.9</td><td>69.9</td><td>63.9</td></tr><tr><td>Expert</td><td>59.4</td><td>69.8</td><td>64.6</td></tr><tr><td rowspan="2">2.05</td><td>Linear Layer</td><td>49.1</td><td>65.1</td><td>57.1</td></tr><tr><td>Expert</td><td>52.9</td><td>65.6</td><td>59.3</td></tr><tr><td rowspan="2">1.57</td><td>Linear Layer</td><td>34.1</td><td>54.3</td><td>44.2</td></tr><tr><td>Expert</td><td>36.6</td><td>57.0</td><td>46.8</td></tr></table>

Table 8: Ablation study comparing our minimax precision balancing when applied to different components of Mixtral 8×7B. We compare expert-level allocation with linear-layer-level allocation across different average bitwidth settings under identical conditions.

![](images/e2fa989709c75e3772983bc4c4aab61f84752ef463708b6e8dc8577a019cc12e.jpg)  
Figure 4: Expert-computing efficiency of Colla-Q on Mixtral 8×7B. Bars show zero-shot accuracy across eight benchmarks, and the line shows TFLOPS under the routing workload. Average bit-widths denote allocation budgets with discrete expert-wise bit assignments.

## 5 Conclusion

We examine MoE from an ensemble perspective, highlighting how low-performing experts can adversely affect the overall performance of quantized MoE models. Based on this insight, we propose Colla-Q, a bit-allocation framework based on an activation-entropy approach that maintains balanced performance across experts under quantization. Unlike routing-based bit allocation strategies, Colla-Q estimates expert performance without token labels to reduce performance imbalance across experts. We first establish an activation-entropy metric to estimate expert performance without token labels. Using this metric, we develop a minimax precision balancing algorithm that assigns more bits to low-performing experts. In our experiments, Colla-Q minimizes quantization-induced performance loss while maintaining stable average performance across diverse MoE LLMs. Moreover, our method shows smaller variation under calibration-domain shifts, indicating better generalization. These results emphasize that, even in low-bit MoE quantization, maintaining balanced expert performance is important, as it supports collaborative expert contributions, thereby enhancing overall model performance and generalizability.

## Limitations

Our study demonstrates the effectiveness of the proposed method on several recent MoE language models under low-bit mixed-precision weightonly quantization. We evaluate Mixtral-8×7B, DeepSeek-16B-Base, and Phi3.5-MoE at average bit-widths ranging from 1.57 to 2.54 bits. While the results show consistent performance across MoE architectures and quantization settings, the empirical scope remains limited in two aspects.

First, our experiments focus on weightonly quantization. Extending Colla-Q to joint weight–activation quantization would be a meaningful direction, as balancing expert performance is not inherently limited to weight quantization. Second, our experiments are limited to three representative MoE backbones—Mixtral-8×7B, DeepSeek-MoE-16B-Base, and Phi3.5-MoE—with model sizes ranging from 16B to 46.7B parameters. These models were selected because they are widely used in prior MoE quantization and compression studies, enabling comparison under comparable settings. However, evaluation on newer MoE architectures and substantially larger models, such as those exceeding 100B parameters, remains limited by the considerable computational cost.

As an additional practical consideration, our expert-computation analysis shows improved efficiency with low-bit expert GEMM kernels, while the evaluation does not cover MoE-specific runtime overheads such as expert dispatch, memory movement, and small-GEMM scheduling. Since inference speed is as important as memory reduction in practice, end-to-end kernel and runtime optimization for mixed-precision MoE inference remains important. Taken together, extending Colla-Q to joint weight–activation quantization, broader and larger MoE architectures, and end-to-end optimization remain important directions for future work.

## Acknowledgements

This research was supported by the National Research Foundation of Korea (NRF), Electronics and Telecommunications Research Institute(ETRI), and Institute of Information & Communications Technology Planning & Evaluation (IITP), funded by the Korean government [26CS1100, Development of Proprietary Physical AI-based Small-scale Computers and Integrated Soft Suits], and the Korea government(MSIT) (RS-2024-00356486, RS-2026- 25617480, and IITP-2026-RS-2023-00255968).

## References

Marah Abdin, Jyoti Aneja, Hany Awadalla, Ahmed Awadallah, Ammar Ahmad Awan, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Jianmin Bao, Harkirat Behl, Alon Benhaim, Misha Bilenko, Johan Bjorck, Sébastien Bubeck, Martin Cai, Qin Cai, Vishrav Chaudhary, Dong Chen, Dongdong Chen, and 110 others. 2024. Phi-3 technical report: A highly capable language model locally on your phone. arXiv preprint arXiv:2404.14219.

Aida Amini, Saadia Gabriel, Shanchuan Lin, Rik Koncel-Kedziorski, Yejin Choi, and Hannaneh Hajishirzi. 2019. MathQA: Towards interpretable math word problem solving with operation-based formalisms. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 2357–2367. Association for Computational Linguistics.

Xiaomeng An and Sen Xu. 2023. A selective evolutionary heterogeneous ensemble algorithm for classifying imbalanced data. Electronic Research Archive, 31(5):2733–2757.

Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. 2020. PIQA: Reasoning about physical commonsense in natural language. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 34, pages 7432–7439.

Leo Breiman. 2001. Random forests. Machine Learning, 45(1):5–32.

Rich Caruana, Alexandru Niculescu-Mizil, Geoff Crew, and Alex Ksikes. 2004. Ensemble selection from libraries of models. In Proceedings of the Twenty-First International Conference on Machine Learning, page 18.

Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. 2019. BoolQ: Exploring the surprising difficulty of natural yes/no questions. In Proceedings ofthe 2019 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 2924–2936.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018. Think you have solved question answering? try ARC, the AI2 reasoning challenge. arXiv preprint arXiv:1803.05457.

Thomas M Cover and Joy A Thomas. 2006. Elements of information theory. John Wiley & Sons.

Damai Dai, Chengqi Deng, Chenggang Zhao, R.X. Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Y. Wu, Zhenda Xie, Y.K. Li, Panpan Huang, Fuli Luo, Chong Ruan, Zhifang Sui, and Wenfeng Liang. 2024. DeepSeekMoE: Towards ultimate

expert specialization in mixture-of-experts language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1280–1297.

Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer. 2022. LLM.int8(): 8-bit matrix multiplication for transformers at scale. In Advances in Neural Information Processing Systems, pages 30318–30332.

Zhen Dong, Zhewei Yao, Amir Gholami, Michael W Mahoney, and Kurt Keutzer. 2019. HAWQ: Hessian aware quantization of neural networks with mixedprecision. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 293– 302.

Haojie Duanmu, Xiuhong Li, Zhihang Yuan, Size Zheng, Jiangfei Duan, Xingcheng Zhang, and Dahua Lin. 2025. MxMoE: Mixed-precision quantization for MoE with accuracy and performance co-design. In Forty-second International Conference on Machine Learning.

Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. 2023. GPTQ: Accurate post-training quantization for generative pre-trained transformers. In International Conference on Learning Representations.

Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, and 5 others. 2024. A framework for few-shot language model evaluation.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021. Measuring massive multitask language understanding. In International Conference on Learning Representations.

Wei Huang, Yue Liao, Jianhui Liu, Ruifei He, Haoru Tan, Shiming Zhang, Hongsheng Li, Si Liu, and Xiaojuan Qi. 2025. Mixture compressor for mixture-ofexperts LLMs gains more. In International Conference on Learning Representations.

Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Sandeep Subramanian, Sophia Yang, and 7 others. 2024. Mixtral of experts. arXiv preprint arXiv:2401.04088.

Changhun Lee, Jungyu Jin, Taesu Kim, Hyungjun Kim, and Eunhyeok Park. 2024. OWQ: Outlier-aware weight quantization for efficient fine-tuning and inference of large language models. In Proceedings

of the AAAI Conference on Artificial Intelligence, volume 38, pages 13355–13364.

Pingzhi Li, Xiaolong Jin, Zhen Tan, Yu Cheng, and Tianlong Chen. 2024. QuantMoE-Bench: Examining post-training quantization for mixture-of-experts. arXiv preprint arXiv:2406.08155.

Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, and Song Han. 2024. AWQ: Activation-aware weight quantization for LLM compression and acceleration. In Proceedings ofMachine Learning and Systems, volume 6, pages 87–100.

Andrey Malinin and Mark Gales. 2018. Predictive uncertainty estimation via prior networks. In Advances in Neural Information Processing Systems, volume 31.

Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. 2017. Pointer sentinel mixture models. In International Conference on Learning Representations.

Niklas Muennighoff, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Jacob Morrison, Sewon Min, Weijia Shi, Evan Pete Walsh, Oyvind Tafjord, Nathan Lambert, Yuling Gu, Shane Arora, Akshita Bhagia, Dustin Schwenk, David Wadden, Alexander Wettig, Binyuan Hui, Tim Dettmers, Douwe Kiela, and 5 others. 2025. OLMoE: Open mixture-of-experts language models. In International Conference on Learning Representations.

Denis Paperno, Germán Kruszewski, Angeliki Lazaridou, Ngoc-Quan Pham, Raffaella Bernardi, Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel Fernández. 2016. The LAMBADA dataset: Word prediction requiring a broad discourse context. In Proceedings of the 54th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 1525–1534, Berlin, Germany. Association for Computational Linguistics.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal ofMachine Learning Research, 21(140):1–67.

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. 2020. WinoGrande: An adversarial Winograd schema challenge at scale. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 8732–8740.

Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. 2017. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. In International Conference on Learning Representations.

Yuxuan Yao, Han Wu, Mingyang Liu, Sichun Luo, Xiongwei Han, Jie Liu, Zhijiang Guo, and Linqi Song. 2025. Determine-then-ensemble: Necessity of top-k union for large language model ensembling. In International Conference on Learning Representations.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. 2019. HellaSwag: Can a machine really finish your sentence? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 4791–4800.

## Appendix

## A.1 Empirical Analysis of the Approximately Uniform Routing Weights

In Section 3.1, we assumed that the routing weights are approximately uniform in expectation over many tokens due to the load-balancing objective. To verify this assumption, we analyze the routing behavior of Mixtral 8×7B on 128 samples from C4 with sequence length 2048. Since Mixtral ${ \bf 8 } \times 7 { \bf B }$ adopts top-2 routing, $| \hat { E } ( x ) | = 2$ for each token, and the reference value is $1 / | \hat { E } ( x ) | = 0 . 5 $ . For each layer, we collect the router logits, identify the top-2 experts for each token, and compute the conditional average gate $\bar { g } _ { e }$ of each expert as:

$$
\bar { g } _ { e } = \frac { \sum _ { x } g _ { e } ( x ) \mathbf { 1 } [ e \in \hat { E } ( x ) ] } { \sum _ { x } \mathbf { 1 } [ e \in \hat { E } ( x ) ] } .\tag{11}
$$

We then compute the mean absolute difference (MAD) between $\bar { g } _ { e }$ and $1 / | \hat { E } ( x )$ | across experts. A smaller MAD indicates that routing weights are closer to the approximately uniform assumption.

<table><tr><td>Metric</td><td>E0</td><td>E1</td><td>E2</td><td>E3</td><td>E4</td><td>E5</td><td>E6</td><td>E7</td></tr><tr><td>Avg.</td><td>0.50</td><td>0.51</td><td>0.50</td><td>0.49</td><td>0.50</td><td>0.50</td><td>0.51</td><td>0.49</td></tr></table>

Table 9: Conditional average routing weight of each expert over many tokens, averaged over Layers 00-31.

<table><tr><td>Layer MAD</td><td></td><td></td><td>|Layer MAD |Layer MAD|</td><td></td><td></td><td></td><td>|Layer MAD</td></tr><tr><td>0</td><td>0.06</td><td>8</td><td>0.01</td><td>16</td><td>0.03</td><td>24</td><td>0.02</td></tr><tr><td>1</td><td>0.07</td><td>9</td><td>0.02</td><td>17</td><td>0.03</td><td>25</td><td>0.04</td></tr><tr><td>2</td><td>0.02</td><td>10</td><td>0.03</td><td>18</td><td>0.02</td><td>26</td><td>0.04</td></tr><tr><td>3</td><td>0.01</td><td>11</td><td>0.03</td><td>19</td><td>0.02</td><td>27</td><td>0.04</td></tr><tr><td>4</td><td>0.01</td><td>12</td><td>0.04</td><td>20</td><td>0.02</td><td>28</td><td>0.06</td></tr><tr><td>5</td><td>0.01</td><td>13</td><td>0.01</td><td>21</td><td>0.01</td><td>29</td><td>0.04</td></tr><tr><td>6</td><td>0.02</td><td>14</td><td>0.02</td><td>22</td><td>0.02</td><td>30</td><td>0.06</td></tr><tr><td>7</td><td>0.01</td><td>15</td><td>0.01</td><td>23</td><td>0.02</td><td>31</td><td>0.08</td></tr></table>

Table 10: Layer-wise MAD of the conditional average routing weights from the reference value of 0.5.

As shown in Tables 9 and 10, for Mixtral ${ \bf 8 } \times 7 { \bf B }$ the expert-wise conditional average routing weights are close to the reference value, and the layer-wise MAD is small in most layers. We further verify this assumption using the same analysis on DeepSeek-MoE-16B-Base and Phi3.5-MoE in Table 11. These results support the approximately uniform routingweight assumption across the evaluated MoE architectures with different expert counts and top-k routing structures.

<table><tr><td>Model</td><td>/ Top-k</td><td>#Experts Reference value</td><td>Expert- Layer-wise wise avg. MAD avg.</td><td></td></tr><tr><td>DeepSeek-MoE</td><td>64/6</td><td> $1 / 6 \approx 0 . 1 6 7$ </td><td>0.165</td><td>0.024</td></tr><tr><td>16B-Base Phi3.5-MoE</td><td>16/2</td><td> $1 / 2 = 0 . 5 0 0$ </td><td>0.496</td><td>0.030</td></tr></table>

Table 11: Conditional average routing weights for additional MoE architectures. Expert-wise avg. is the mean routing weight; Layer-wise MAD avg. is the mean absolute deviation from the reference value.

## A.2 Empirical Analysis of the Gaussian Approximation of Expert Activations

<table><tr><td>Metric</td><td>Channel-wise Expert-wise</td></tr><tr><td>Mixtral 8×7B</td></tr><tr><td>Number of units 32,768 channels 256 experts Mean  $L _ { 1 }$  to  $\mathcal { N } ( 0 , 1 )$  0.020 0.030 Median 0.020 0.020</td></tr><tr><td> $L _ { 1 }$  to  $\mathcal { N } ( 0 , 1 )$  Fraction with  $L _ { 1 } < 0 . 0 5$  0.99 0.92</td></tr><tr><td>DeepSeek-MoE-16B-Base</td></tr><tr><td>Number of units 228,096 channels 1,782 experts</td></tr><tr><td>Mean  $L _ { 1 } \mathrm { t o } \mathcal { N } ( 0 , 1 )$  0.018 0.015 Median 0.016 0.012</td></tr><tr><td> $L _ { 1 } \mathrm { t o } \mathcal { N } ( 0 , 1 )$  Fraction with  $L _ { 1 } < 0 . 0 5$  0.98 0.99</td></tr><tr><td>Phi3.5-MoE</td></tr><tr><td></td></tr><tr><td>Number of units 65,536 channels 512 experts</td></tr><tr><td>Mean  $L _ { 1 }$  to  $\mathcal { N } ( 0 , 1 )$  0.0256 0.029</td></tr><tr><td>Median</td></tr><tr><td> $L _ { 1 }$  to  $\mathcal { N } ( 0 , 1 )$  0.022 0.023 Fraction with  $L _ { 1 } < 0 . 0 5$  0.95 0.90</td></tr></table>

Table 12: Quantitative results for the Gaussianity of channel-wise and expert-wise expert FFN output activations. Lower L1 values indicate that the distribution is closer to the standard Gaussian distribution. Fraction with $\mathrm { L 1 < 0 . 0 5 }$ denotes the proportion of channels or experts with L1 distance below 0.05.

In Section 3.2, Eq. 5 holds when the expert activations follow the Gaussian approximation. To verify this condition, we collect expert output activations from the evaluated MoE models using 128 samples from C4, each with a sequence length of 2048. We conduct two analyses corresponding to $H _ { \mathrm { w i t h i n } }$ and $H _ { \mathrm { t o t a l } }$ in $\operatorname { E q . }$ . 6: channel-wise Gaussianity of $A _ { t , c }$ and expert-wise Gaussianity of pooled activations. For the channel-wise analysis, we sample 32 channels from each expert over four random seeds, while for the expert-wise analysis, we treat all activations from each expert as a single pooled distribution. We compare the distributions with the standard Gaussian using full-range $\ell _ { 1 }$ distance and the fraction of units below a fixed $\ell _ { 1 }$ threshold.

![](images/37ced313b623e2d6c51dddef587eae98cb1d81eba044356afb4ee71850f857b4.jpg)

![](images/2065ae0c2edbcab09f8ff75f4b8bdb2ba367e9226abbbb8a9a9bc2207a7021ae.jpg)  
Figure 5: Channel-wise and expert-wise Gaussianity of expert FFN output activations in Mixtral 8×7B. Activation dist. denotes the averaged standardized density over sampled channels (left) and the standardized density of pooled expert activations (right).

As shown in Table 12, the activation distributions remain close to the standard Gaussian across both channel-wise and expert-wise analyses. Figure 5 further shows that both the standardized channelwise activations and the expert-wise pooled activations exhibit Gaussian-like bell-shaped distributions. This indicates that the dominant structure of expert activations is well captured by the Gaussian approximation. These results support the assumption used in Eq. 5.

## A.3 Empirical Analysis of Activation-Entropy as an Expert Performance Proxy

To examine whether the proposed activationentropy metric $\rho$ provides a meaningful proxy for expert performance, we conduct a controlled expert-pair intervention. The observed negative relationship between activation entropy and the expert performance estimated through this intervention supports its use as a proxy for expert performance. Since expert quality in MoE does not have an observable ground truth, we assess expert performance operationally through downstream performance under controlled expert combinations.

Specifically, we randomly select three layers of Mixtral ${ \bf 8 } \times 7 { \bf B }$ , keep all other layers unchanged, and evaluate all 28 expert pairs in each target layer with routing weights fixed to 0.5 for equal contribution. We measure the downstream performance of each pair and define each expert’s performance as the average over the seven pairs containing that expert. We then compute the Spearman correlation between $\rho$ and the resulting expert-level performance scores. As shown in Table 13, $\rho$ exhibits negative correlations across all three layers, with an average Spearman correlation of −0.56. Since lower $\rho$ indicates better estimated expert performance, this consistent negative correlation supports the intended ordering of our metric and its use as a label-free allocation proxy for relative expert performance.

<table><tr><td>Layer</td><td>Spearman Correlation</td></tr><tr><td>1</td><td>-0.67</td></tr><tr><td>7</td><td>-0.57</td></tr><tr><td>11</td><td>-0.43</td></tr><tr><td>Average</td><td>-0.56</td></tr></table>

Table 13: Spearman correlation between activation entropy $\rho$ and expert-level performance obtained from the expert-pair intervention. A negative correlation indicates that lower $\rho$ tends to correspond to higher expert-level performance.

## A.4 Additional Experimental Setup

Baselines. Uniform quantization does not consider importance differences in the MoE structure, and instead assigns the same bit-width to all experts. BSP and OA-GPTQ (Li et al., 2024) are MoEblock-wise and expert-linear-layer-wise bit-width allocation methods, respectively. BSP estimates the importance of each MoE block using a separately trained predictor based on the cosine similarity of activations. OA-GPTQ computes linear-layer importance within each expert by scoring weight outliers. MxMoE (Duanmu et al., 2025) estimates the importance of each linear layer within an expert using quantization sensitivity and expert activation patterns. PMQ (Huang et al., 2025) is an expertwise bit-width allocation method, which uses a routing-based metric for bit-width allocation.

Implementation Details of Baselines. BSP and OA-GPTQ use Wikitext2 for calibration with 128 sequences of length 2048. BSP allocates 4-bit to the top 25% most important MoE blocks and 2- bit to the remaining blocks. OA-GPTQ allocates 4-bit to the top 25% most important linear layers within experts and 2-bit to the rest. Both achieve a target average bit-width of approximately 2.54 bits (in DeepSeek, shared-expert layers receive 4-bit only if they fall into the selected top blocks/linear layers). We reproduce the experimental results in model configurations that baseline methods (Li et al., 2024) do not report.

MxMoE uses Wikitext2 for calibration with 128 sequences, and we set the sequence length to 2048. We include all linear layers in the bit-width allocation, including those in the shared expert of DeepSeek, and we do not manually fix the sharedexpert bit-width. For a fair comparison, we quantize the attention and gating modules to 4-bit, consistent with our baselines. In low-bit settings, following the MxMoE setup for reproduction, we set the runtime-cost weighting parameter r to 1. Therefore, we can perform a fair comparison between MxMoE and ours under the same conditions.

<table><tr><td>Model</td><td></td><td>Layer #E</td><td>Top-k</td><td>(B)</td><td>Param. Mem. (GB)</td></tr><tr><td>Mixtral 8×7B</td><td>32</td><td>8</td><td>2</td><td>46.7</td><td>96.8</td></tr><tr><td>DeepSeek-16B-Base</td><td>28</td><td>64+2</td><td>6</td><td>16</td><td>30.4</td></tr><tr><td>Phi3.5-MoE</td><td>32</td><td>16</td><td>2</td><td>42</td><td>83.8</td></tr></table>

Table 14: Architectural and memory configurations of MoE models. #E denotes the number of experts in each model, and Top-k indicates the number of routed experts in a MoE block.

PMQ and our method use C4 as the calibration dataset with 128 sequences. For each MoE block, we apply GPTQ-based 4-bit quantization to the attention and gating modules. For DeepSeek-16B-Base, since the first layer is a dense MLP rather than an MoE block, we quantize it using 4-bit perchannel GPTQ as well. For models with shared experts, we fix the shared-expert bit-width to 2-bit when targeting average bit-widths of 1.57 and 2.05 bits, and to 4-bit for the 2.54-bit setting. Model architectures (e.g., the number of layers, experts, and Top-k routing) are summarized in Table 14, and we follow these configurations when applying the above quantization settings.

Expert-Computing Efficiency Evaluation. Following the kernel-level efficiency evaluation of Mx-MoE (Duanmu et al., 2025), we measure expert computation. We use the actual routing results of

<table><tr><td>Bits</td><td>Method</td><td>MMLU</td></tr><tr><td>16</td><td>一</td><td>70.6</td></tr><tr><td rowspan="5">2.54</td><td>OA-GPTQ</td><td>58.1</td></tr><tr><td>BSP</td><td>51.7</td></tr><tr><td>MxMoE</td><td>59.1</td></tr><tr><td>PMQ</td><td>61.2</td></tr><tr><td>Colla-Q</td><td>62.5</td></tr><tr><td rowspan="3">2.05</td><td>MxMoE</td><td>50.2</td></tr><tr><td>PMQ</td><td>49.8</td></tr><tr><td>Colla-Q</td><td>53.2</td></tr><tr><td rowspan="3">1.57</td><td>MxMoE</td><td>29.1</td></tr><tr><td>PMQ</td><td>33.4</td></tr><tr><td>Colla-Q</td><td>34.1</td></tr></table>

Table 15: MMLU 5-shot task performance of Mixtral 8×7B, comparing MoE quantization methods.

Mixtral 8×7B on 128 WikiText2 sequences with a sequence length of 512 and execute each expert according to its routed token count. The reported average bit-widths (e.g., 2.54 bits) denote effective model-level bit-widths obtained from discrete expert-wise bit assignments, with the attention and gating modules fixed at 4 bits, rather than fractionalbit kernels. Based on the bit-width assigned by Colla-Q, experts are executed using INT1×FP16, INT2×FP16, or INT3×FP16 weight-only GEMM kernels and compared with FP16 under the same routing workload. Measurements are conducted on a single NVIDIA RTX 3090 24GB GPU with 10 warm-up iterations and 30 repetitions. Attention and MoE routing/dispatch are excluded, and TFLOPS is computed from the expert-MLP FLOPs and measured kernel execution time.

## A.5 Additional Experimental Results

Complete MMLU Few-shot Results. Table 15 extends Table 3 by reporting MMLU 5-shot performance for Mixtral 8×7B at additional average bit-widths (2.05-bit and 1.57-bit), together with the 2.54-bit setting.

Complete Results Across All Benchmarks. We provide complete experimental results to complement Table 2 of the manuscript. Table 16 and Table 17 show the complete experimental results on eight benchmarks for DeepSeek-16B-Base and Phi3.5-MoE across different average bit-widths and MoE quantization methods.

Calibration Dataset Dependence. We provide experimental results regarding the calibration dataset dependence. Figure 6 visualizes the heatmaps of the routing-based and activation-entropy metrics across experts and MoE blocks for the two models Mixtral 8×7B and Phi3.5-MoE. They demonstrate that the activation-entropy metric produces similar patterns of the heatmaps across different calibration datasets. This result implies that our method operates without dependence on the particular calibration dataset, yielding generalized quantization results. Figure 7 provides the complete cosine-similarity heatmaps corresponding to the summary statistics reported in Table 4. In our experiments, we compute the cosine similarity as:

$$
S = \frac { 1 } { | \mathcal { L } | } \sum _ { l \in \mathcal { L } } \frac { { \mathbf { a } } _ { l } ^ { \top } { \mathbf { b } } _ { l } } { \| { \mathbf { a } } _ { l } \| _ { 2 } \| { \mathbf { b } } _ { l } \| _ { 2 } } ,\tag{12}
$$

where L denotes the set of MoE blocks, a<sub>l</sub> and b<sub>l</sub> denote the metric vectors (e.g., activation-entropy

<table><tr><td>Bits</td><td>Method</td><td>MMLU</td><td>PIQA</td><td>ARC-e</td><td>ARC-c</td><td>BoolQ</td><td>HellaS.</td><td>Wino.</td><td>MathQA</td><td>Avg.</td></tr><tr><td>16</td><td></td><td>38.0</td><td>80.0</td><td>76.0</td><td>47.7</td><td>72.3</td><td>77.4</td><td>71.0</td><td>31.5</td><td>61.7</td></tr><tr><td rowspan="5">2.54</td><td>OA-GPTQ</td><td>33.3</td><td>78.6</td><td>72.9</td><td>42.1</td><td>70.5</td><td>72.1</td><td>69.0</td><td>28.3</td><td>58.3</td></tr><tr><td>BSP</td><td>29.7</td><td>76.6</td><td>69.3</td><td>39.7</td><td>47.5</td><td>70.5</td><td>66.1</td><td>27.1</td><td>53.3</td></tr><tr><td>MxMoE</td><td>29.8</td><td>79.3</td><td>72.8</td><td>43.2</td><td>67.7</td><td>74.3</td><td>68.1</td><td>29.5</td><td>58.1</td></tr><tr><td>PMQ</td><td>32.8</td><td>78.9</td><td>70.1</td><td>41.8</td><td>74.2</td><td>71.0</td><td>67.9</td><td>29.3</td><td>58.2</td></tr><tr><td>Colla-Q</td><td>33.4</td><td>79.8</td><td>74.7</td><td>42.8</td><td>72.3</td><td>74.1</td><td>69.1</td><td>29.8</td><td>59.5</td></tr><tr><td rowspan="3">2.05</td><td>MxMoE</td><td>26.4</td><td>75.7</td><td>67.1</td><td>37.6</td><td>66.3</td><td>64.4</td><td>66.2</td><td>27.0</td><td>53.8</td></tr><tr><td>PMQ</td><td>27.0</td><td>77.3</td><td>64.4</td><td>38.6</td><td>67.5</td><td>68.0</td><td>65.1</td><td>25.3</td><td>54.1</td></tr><tr><td>Colla-Q</td><td>27.1</td><td>77.0</td><td>68.7</td><td>38.7</td><td>69.1</td><td>67.3</td><td>66.5</td><td>26.8</td><td>55.2</td></tr><tr><td rowspan="3">1.57</td><td>MxMoE</td><td>24.0</td><td>63.9</td><td>40.5</td><td>23.9</td><td>63.4</td><td>43.0</td><td>59.1</td><td>24.2</td><td>42.8</td></tr><tr><td>PMQ</td><td>22.6</td><td>73.0</td><td>62.0</td><td>34.6</td><td>63.0</td><td>58.3</td><td>63.4</td><td>23.5</td><td>50.1</td></tr><tr><td>Colla-Q</td><td>23.3</td><td>72.6</td><td>64.3</td><td>35.1</td><td>64.0</td><td>57.6</td><td>65.0</td><td>24.5</td><td>50.8</td></tr></table>

Table 16: Zero-shot task performance of DeepSeek-16B-Base on eight benchmarks under different average bitwidths and quantization methods.
<table><tr><td>Bits</td><td>Method</td><td>MMLU</td><td>PIQA</td><td>ARC-e</td><td>ARC-c</td><td>BoolQ</td><td>HellaS.</td><td>Wino.</td><td>MathQA</td><td>Avg.</td></tr><tr><td>16</td><td>-</td><td>76.6</td><td>78.2</td><td>65.0</td><td>53.6</td><td>88.4</td><td>79.1</td><td>75.9</td><td>37.3</td><td>69.3</td></tr><tr><td rowspan="5">2.54</td><td>OA-GPTQ</td><td>43.8</td><td>66.0</td><td>55.7</td><td>40.1</td><td>69.9</td><td>60.9</td><td>58.3</td><td>22.6</td><td>52.2</td></tr><tr><td>BSP</td><td>39.4</td><td>60.0</td><td>42.4</td><td>33.8</td><td>56.5</td><td>48.3</td><td>51.5</td><td>22.0</td><td>44.2</td></tr><tr><td>MxMoE</td><td>43.9</td><td>64.1</td><td>51.6</td><td>38.9</td><td>64.7</td><td>57.8</td><td>58.5</td><td>23.3</td><td>50.3</td></tr><tr><td>PMQ</td><td>50.7</td><td>71.6</td><td>58.6</td><td>42.3</td><td>76.6</td><td>70.0</td><td>62.9</td><td>22.5</td><td>56.9</td></tr><tr><td>Colla-Q</td><td>51.8</td><td>72.1</td><td>59.2</td><td>41.6</td><td>76.3</td><td>70.7</td><td>63.6</td><td>25.5</td><td>57.6</td></tr><tr><td rowspan="3">2.05</td><td>MxMoE</td><td>27.8</td><td>57.9</td><td>41.3</td><td>26.5</td><td>53.7</td><td>44.6</td><td>51.5</td><td>20.7</td><td>40.5</td></tr><tr><td>PMQ</td><td>25.4</td><td>64.6</td><td>50.5</td><td>36.0</td><td>59.9</td><td>60.4</td><td>60.2</td><td>22.0</td><td>47.4</td></tr><tr><td>Colla-Q</td><td>28.3</td><td>68.1</td><td>51.0</td><td>37.4</td><td>67.6</td><td>61.4</td><td>60.9</td><td>23.8</td><td>49.8</td></tr><tr><td rowspan="3">1.57</td><td>MxMoE</td><td>23.7</td><td>53.1</td><td>32.3</td><td>27.1</td><td>55.4</td><td>32.4</td><td>50.0</td><td>21.1</td><td>36.8</td></tr><tr><td>PMQ</td><td>23.5</td><td>56.8</td><td>37.9</td><td>31.4</td><td>50.6</td><td>43.4</td><td>53.1</td><td>19.4</td><td>39.5</td></tr><tr><td>Colla-Q</td><td>24.3</td><td>59.8</td><td>40.8</td><td>29.6</td><td>54.6</td><td>42.7</td><td>54.9</td><td>20.7</td><td>40.9</td></tr></table>

Table 17: Zero-shot task performance of Phi3.5-MoE on eight benchmarks under different average bit-widths and quantization methods.

![](images/04fbbb0394aca5f5256200ee7e2455a9d0383684620a8fd385513795bcc15fc5.jpg)

![](images/d4e2aae63440f3738ab858574626c7a75b3163dc382ce5471d2e22de32bec3e6.jpg)  
(a) Mixtral 8×7B’s factor vectors

![](images/8caa166fd7e92a60e4c60adedaf008924181627bf1aab090e25b4cfc7b4888b7.jpg)

![](images/a7b76eb799ad6635626ea56b7f8904343b715fd243eba9a45f7fd80489143635.jpg)  
(b) Phi3.5-MoE’s factor vectors  
Figure 6: Heatmaps of the PMQ’s routing-based metric (left) and our metric (right) computed from different calibration datasets on Mixtral 8×7B and Phi3.5-MoE models. For each calibration dataset, each heatmap shows the normalized values across MoE blocks.

metric). Using this cosine similarity across MoE blocks, we can confirm that our activation-entropy

metric produces consistent results across different calibration datasets, as shown in Figure 7. Further-

![](images/bc9557bd94cbd06585f6bfff22c04aef884472ff903c1397ae190521820eaf6c.jpg)  
(a) Mixtral 8×7B’s factor cosine heatmap

![](images/6893788f813cf5f14a64884b877f71215dd29f8c9f6503240c34e59c29afbb10.jpg)  
(b) Phi3.5-MoE’s factor cosine heatmap

Figure 7: Cosine similarity heatmaps of routing-based and our metrics across calibration datasets on Mixtral 8×7B and Phi3.5-MoE models. For each model, the left heatmap shows similarities between metrics obtained with PMQ and the right reports those obtained with ours. Cosine similarities in [0, 1] are multiplied by 100.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Calibration dataset</td><td colspan="4">Mixtral 8×7B</td><td colspan="4">Phi3.5-MoE</td></tr><tr><td>MathQA</td><td>Lambada_mul_fr</td><td>BoolQ</td><td>Avg.</td><td>MathQA</td><td>Lambada_mul_fr</td><td>BoolQ</td><td>Avg.</td></tr><tr><td rowspan="4">PMQ</td><td>C4</td><td>26.8</td><td>23.8</td><td>67.8</td><td>39.4</td><td>19.4</td><td>0.8</td><td>50.6</td><td>23.6</td></tr><tr><td>Math</td><td>27.0</td><td>22.9</td><td>68.2</td><td>39.4</td><td>22.1</td><td>0.1</td><td>54.0</td><td>25.4</td></tr><tr><td>French</td><td>25.4</td><td>25.7</td><td>72.1</td><td>41.1</td><td>19.3</td><td>4.4</td><td>51.6</td><td>25.1</td></tr><tr><td>QA</td><td>25.8</td><td>21.3</td><td>76.2</td><td>41.1</td><td>20.3</td><td>0.6</td><td>57.3</td><td>26.1</td></tr><tr><td rowspan="4">Ours</td><td>C4</td><td>26.9</td><td>26.8</td><td>76.5</td><td>43.5</td><td>22.7</td><td>6.8</td><td>60.6</td><td>30.0</td></tr><tr><td>Math</td><td>27.3</td><td>26.4</td><td>77.8</td><td>43.8</td><td>23.0</td><td>6.3</td><td>60.8</td><td>30.0</td></tr><tr><td>French</td><td>27.2</td><td>26.9</td><td>77.8</td><td>44.0</td><td>22.1</td><td>6.6</td><td>61.3</td><td>30.0</td></tr><tr><td>QA</td><td>27.1</td><td>26.3</td><td>77.6</td><td>43.7</td><td>22.2</td><td>6.5</td><td>61.5</td><td>30.0</td></tr></table>

Table 18: Zero-shot task performance under different calibration datasets (C4, Math, French, QA), comparing PMQ and ours for Mixtral 8×7B and Phi3.5-MoE in the 1.57-bit setting.

<table><tr><td rowspan="2">Bits</td><td rowspan="2">Metric</td><td colspan="2">Algorithm</td></tr><tr><td>ILP</td><td>Ours</td></tr><tr><td rowspan="2">2.54</td><td>PMQ</td><td>67.5</td><td>67.7</td></tr><tr><td>Ours</td><td>67.3</td><td>68.5</td></tr><tr><td rowspan="2">2.05</td><td>PMQ</td><td>63.3</td><td>63.3</td></tr><tr><td>Ours</td><td>63.1</td><td>64.0</td></tr><tr><td rowspan="2">1.57</td><td>PMQ</td><td>54.5</td><td>53.6</td></tr><tr><td>Ours</td><td>53.2</td><td>54.5</td></tr></table>

Table 19: Ablation study on Mixtral 8×7B comparing combinations of PMQ and activation entropy metrics with ILP and minimax precision balancing. Values report average performance across eight benchmarks.

more, Table 18 reports the corresponding zero-shot performance under each calibration dataset for both models.

## A.6 Additional Ablation Studies

Ablation of Metric and Allocation Strategy. To isolate the contributions of the metric and allocation strategy, we conduct a $2 \times 2$ ablation by crossapplying the PMQ routing-based and our activationentropy metrics with the PMQ ILP allocation and our minimax precision balancing. As shown in Table 19, minimax precision balancing consistently outperforms ILP with the activation-entropy metric across all bit-widths, while the gains are less consistent with the PMQ metric. These results support the complementary roles of activation entropy and minimax precision balancing in Colla-Q.

<table><tr><td>Bits</td><td>Minimax Precision Balancing (Ours)</td><td>Direct ILP w/ Colla-Q Objective</td></tr><tr><td>2.54</td><td>68.5</td><td>65.9</td></tr><tr><td>2.05</td><td>64.0</td><td>61.7</td></tr><tr><td>1.57</td><td>54.5</td><td>52.9</td></tr></table>

Table 20: Ablation study of average zero-shot performance on eight benchmarks for Mixtral 8×7B. We compare minimax precision balancing with Direct ILP using the same Colla-Q objective and bit budget.

Ablation of the Bit-Allocation Procedure. To evaluate the effectiveness of our minimax precision balancing procedure, we compare it with Direct ILP. For a fair comparison, we reformulate Direct ILP to optimize the same Colla-Q minimax objective, rather than the conventional sum-based objective, under the same bit budget. While Direct ILP determines the bit-width configuration through a single optimization, our method recursively allocates additional bits to the worst-performing expert. As shown in Table 20, minimax precision balancing consistently achieves higher downstream performance across all bit-widths. These results support the effectiveness of our precision-balancing procedure for discrete expert-wise bit allocation.

<table><tr><td>Bits</td><td>Term</td><td>MMLU</td><td>PIQA</td><td>ARC-e</td><td>ARC-c</td><td>BoolQ</td><td>HellaS.</td><td>Wino.</td><td>MathQA</td><td>Avg.</td></tr><tr><td>16</td><td>-</td><td>67.8</td><td>83.6</td><td>84.2</td><td>56.5</td><td>85.0</td><td>84.0</td><td>76.2</td><td>41.7</td><td>72.4</td></tr><tr><td rowspan="2">2.54</td><td>Quantization error</td><td>58.3</td><td>80.0</td><td>79.0</td><td>51.4</td><td>83.9</td><td>79.0</td><td>75.0</td><td>35.8</td><td>67.8</td></tr><tr><td>Colla-Q</td><td>59.4</td><td>80.5</td><td>79.8</td><td>54.0</td><td>84.7</td><td>78.5</td><td>74.9</td><td>36.5</td><td>68.5</td></tr><tr><td rowspan="2">2.05</td><td>Quantization error</td><td>49.9</td><td>78.2</td><td>73.6</td><td>47.8</td><td>83.0</td><td>73.2</td><td>72.1</td><td>32.7</td><td>63.8</td></tr><tr><td>Colla-Q</td><td>52.9</td><td>77.8</td><td>74.9</td><td>46.4</td><td>83.3</td><td>72.4</td><td>72.0</td><td>32.3</td><td>64.0</td></tr><tr><td rowspan="2">1.57</td><td>Quantization error</td><td>35.7</td><td>70.3</td><td>63.0</td><td>34.0</td><td>76.0</td><td>58.1</td><td>64.1</td><td>26.2</td><td>53.4</td></tr><tr><td>Colla-Q</td><td>36.6</td><td>71.0</td><td>63.1</td><td>36.2</td><td>75.8</td><td>59.8</td><td>66.2</td><td>26.9</td><td>54.5</td></tr></table>

Table 21: Zero-shot task performance of Mixtral 8×7B on eight benchmarks, comparing quantization error with Colla-Q under different average bit budgets.
<table><tr><td>Bits</td><td>Components</td><td>MMLU</td><td>PIQA</td><td>ARC-e</td><td>ARC-c</td><td>BoolQ</td><td>HellaS.</td><td>Wino.</td><td>MathQA</td><td>Avg.</td></tr><tr><td>16</td><td>-</td><td>67.8</td><td>83.6</td><td>84.2</td><td>56.5</td><td>85.0</td><td>84.0</td><td>76.2</td><td>41.7</td><td>72.4</td></tr><tr><td rowspan="2">2.54</td><td>Linear Layer</td><td>57.9</td><td>80.6</td><td>80.7</td><td>53.4</td><td>84.9</td><td>78.8</td><td>74.7</td><td>36.4</td><td>68.4</td></tr><tr><td>Expert</td><td>59.4</td><td>80.5</td><td>79.8</td><td>54.0</td><td>84.7</td><td>78.5</td><td>74.9</td><td>36.5</td><td>68.5</td></tr><tr><td rowspan="2">2.05</td><td>Linear Layer</td><td>49.1</td><td>78.3</td><td>72.2</td><td>45.2</td><td>81.7</td><td>72.8</td><td>72.8</td><td>33.0</td><td>63.1</td></tr><tr><td>Expert</td><td>52.9</td><td>77.8</td><td>74.9</td><td>46.4</td><td>83.3</td><td>72.4</td><td>72.0</td><td>32.3</td><td>64.0</td></tr><tr><td rowspan="2">1.57</td><td>Linear Layer</td><td>34.1</td><td>71.1</td><td>61.8</td><td>34.5</td><td>63.1</td><td>56.6</td><td>65.8</td><td>27.2</td><td>51.8</td></tr><tr><td>Expert</td><td>36.6</td><td>71.0</td><td>63.1</td><td>36.2</td><td>75.8</td><td>59.8</td><td>66.2</td><td>26.9</td><td>54.5</td></tr></table>

Table 22: Zero-shot task performance of Mixtral 8×7B on eight benchmarks, comparing mixed-precision bit-width allocation across components (Linear Layer vs. Expert) under different average bit budgets.

## A.7 Complete Results for Ablation Study

We offer the complete experimental results of our ablation studies in Tables 7 and 8 of the manuscript. Tables 21 and 22 further extend these results by reporting the full benchmark outcomes for activationentropy-weighted quantization error and expertwise mixed-precision allocation. These additional results allow a more detailed assessment of the consistency of the observed trends across tasks.