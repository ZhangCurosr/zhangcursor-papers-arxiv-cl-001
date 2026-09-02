# Trust Your Guide Only When Certain: Uncertainty-Aware Sparse Alignment at Inference Time

Zeen Zhu<sup>1</sup>, Zhuo Li<sup>1</sup>, Weiyang Guo<sup>1</sup>, Liye Zhao<sup>2</sup>, Haibing Di<sup>2</sup>,

Yequan Wang<sup>3</sup>, Jing Li<sup>1#</sup>

<sup>1</sup>Harbin Institute of Technology, Shenzhen, China <sup>2</sup>Huawei Technologies Co., Ltd.

<sup>3</sup>Beijing Academy of Artificial Intelligence, China. zhuzeen24@gmail.com jingli.phd@hotmail.com

## Abstract

A prominent paradigm in inference-time alignment employs lightweight supervisors to steer Large Language Models (LLMs). Through empirical analysis, we identify a structural mismatch in this paradigm: weak supervisors exhibit pervasive high entropy across the vast majority of tokens, yet prevailing dense intervention approaches mandate supervision at every decoding step. This leads to frequent low-confidence interventions that can disrupt valid base-model reasoning and incur substantial utility costs. To resolve this, we propose TUSA (Trust-based Uncertainty Sparse Alignment). Moving away from continuous oversight, TUSA reframes alignment as a dynamic arbitration process, introducing an uncertainty-aware arbiter that authorizes intervention only when two conditions are met: the supervisor is confident and the token is semantically salient. This mechanism effectively filters out uncertainty-driven noise and redundant supervision. Extensive experiments across multiple models and benchmarks show that TUSA consistently improves both safety alignment and general helpfulness. By bypassing approximately 50% of alignment steps, it not only enhances safety preference by up to 15.6%, but also boosts general preference rates by up to 12.0% compared to the dense baseline, demonstrating that selective, high-precision alignment can outperform continuous supervision. Our code is available at this url.

WARNING: This paper contains potentially offensive and harmful text.

## 1 Introduction

Aligning Large Language Models (LLMs) with human values is essential for safe deployment. While training-time alignment approaches such as RLHF (Christiano et al., 2017; Ouyang et al., 2022) and DPO (Rafailov et al., 2023) have been highly effective, they require expensive data collection and parameter modifications to large model weights. As models scale, such retraining becomes increasingly computationally demanding.

![](images/d0afcba9a0700d63597f7f694ce455639f30c8fe7b8b6b106bbf048ea3f6583c.jpg)  
Figure 1: Dense vs. Sparse Intervention. While dense intervention (top) scrutinizes every token, our sparse strategy (bottom) intervenes at safety-critical decision points. Benign descriptors like “chemical” are bypassed to avoid unnecessary over-correction.

These limitations have motivated a paradigm shift toward Inference-Time Alignment (Li et al., 2023a; Shi et al., 2024; Huang et al., 2025a). Instead of updating model parameters, this paradigm steers the generation process of a frozen Base Model through intervention techniques. This reframes alignment as a dynamic control problem, enabling modular, controllable, and resource-efficient safety enforcement without the prohibitive cost of full-scale retraining.

A prominent direction in this paradigm follows the Weak-to-Strong Guidance principle (Mudgal et al., 2024; Khanov et al., 2024; Yuan et al., 2025; Zhang et al., 2025): a lightweight, specialized supervisor model (the “Weak Specialist”) supervises the generation of a massive, general-purpose model (the “Strong Generalist”). Systems such as MARA (Zhang et al., 2025) exemplify this approach, showing that a parameter-efficient Micro-Agent can effectively steer substantially larger Base Models through token-level intervention. However, these methods rely on Dense Intervention, requiring the Specialist to inspect and potentially override every output token. As illustrated in Figure 1, this strategy enforces continuous supervision across the entire sequence—including benign function words and clear semantic flows—forcing the Base Model to be continuously redirected.

![](images/294b9f89369cb1c33a21d9628f56c2572b4653c96fb027eca3b54c40ffbf3231.jpg)  
(a) Uncertainty distribution across domains.

![](images/d7709493aff452a2f7d1394cc65821c9e452f7b7a89ed17a4054e69aef03eee0.jpg)  
(b) Uncertainty over generation steps.  
Figure 2: Uncertainty Profile of the Weak Specialist. (a) Entropy consistently hovers near the maximum (ln 2) across all domains. (b) High uncertainty persists across decoding steps, rendering dense intervention unreliable.

Dense intervention operates on the premise that the Weak Specialist can provide reliable guidance at every decoding step. However, our empirical analysis suggests this assumption is frequently challenged. Figure 2 illustrates the specialist’s entropy profile. As visualized in Figure 2a, the Weak Specialist displays consistently elevated uncertainty across diverse domains, encompassing both general tasks and safety benchmarks, where entropy values frequently approach the theoretical maximum. Furthermore, Figure 2b indicates that this ambiguity is not transient but often persists throughout the generation trajectory. Consequently, enforcing interventions during these low-confidence states creates a structural Confidence–Intervention Mismatch, which risks introducing cognitive noise rather than yielding meaningful alignment gains.

This reveals a fundamental question for inference-time alignment: Should the Weak Specialist intervene only when it is genuinely confident, and otherwise allow the Strong Generalist to proceed unhindered? In this work, we answer this question and introduce a trust-based sparse alignment framework that adaptively arbitrates when intervention is necessary.

We contend that alignment should follow a Trust-Based Arbitration principle: the Base Model’s distribution is preserved unless the Weak Specialist exhibits both high confidence and clear necessity for intervention.

To instantiate this, we introduce TUSA (Trustbased Uncertainty Sparse Alignment), which substitutes dense supervision with a training-free Cognitive Arbiter. As shown in Figure 3, by fusing uncertainty and semantic saliency, the arbiter constructs an adaptive activation boundary, authorizing interventions only when the specialist is confident and the candidate token is semantically rich.

Our contributions are summarized as follows:

• Empirical Insight: We identify a structural Confidence–Intervention Mismatch, showing that weak supervisors often exhibit high entropy. Current dense methods typically operate independently of this uncertainty, applying supervision uniformly across all decoding steps, which risks introducing noise during low-confidence states.

• Methodological Framework: We propose TUSA, a training-free framework for adaptive alignment. Central to this is the Cognitive Arbiter, which synthesizes specialist competence and semantic saliency to compute a Joint Necessity Score. This mechanism establishes a dynamic trust boundary, authorizing supervision strictly when the specialist exhibits high confidence on semantically significant tokens.

• Performance: Extensive evaluations across the Mistral and Llama families demonstrate that TUSA reduces intervention overhead by approximately 50% compared to the dense baseline. Simultaneously, it boosts the Preference Rate by up to 12.0% on general benchmarks and achieves gains of up to 15.6% on safety benchmarks, validating that selective intervention yields superior outcomes compared to continuous supervision.

![](images/dcd363d7159f5e4891c01902a17382f0f2025f38035b64e3b171ee779f53c3dc.jpg)  
Figure 3: The TUSA Framework. Alignment is applied only when the Weak Specialist is both confident and necessary, transforming dense supervision into selective steering.

## 2 Related Work

Small-Model Guided Generation. Leveraging lightweight models to guide or supervise large language models has emerged as an effective paradigm for scalable oversight (Burns et al., 2024). Early approaches such as Contrastive Decoding (Li et al., 2023b) and Proxy Tuning (Liu et al., 2024) influence generation through inference-time logit manipulation, while more recent work extends this paradigm to preference optimization (Zhu et al., 2025) and robustness transfer under uncertainty (Dong et al., 2025). Collectively, these methods demonstrate that small or specialized models can meaningfully influence stronger generators, motivating broader exploration of modular and inference-time supervision mechanisms.

Inference-Time Alignment. Building on the Weak to Strong principle, inference-time alignment focuses on steering frozen base models through external control signals, avoiding costly retraining procedures such as RLHF (Ouyang et al., 2022) or DPO (Rafailov et al., 2023). Existing approaches span internal activation steering (Subramani et al., 2022; Turner et al., 2023; Zou et al., 2023) and decoding-time methods such as PPLM (Dathathri et al., 2020) and DExperts (Liu et al., 2021). More recent modular frameworks further instantiate this paradigm, including contrastive decoding variants like DoLa (Chuang et al., 2024), reward-guided generation methods such as GENARM (Xu et al., 2025) and PARM (Lin et al., 2025), as well as micro alignment systems like MARA (Zhang et al., 2025) and Aligner (Ji et al., 2024). Many of these methods apply token-level supervision throughout the generation process, highlighting the importance of deciding when and how external signals should be applied.

Uncertainty Quantification in Alignment. Model uncertainty is widely used to improve generation reliability, commonly estimated via self-consistency checks (Kadavath et al., 2022; Lin et al., 2024b). Prior work distinguishes linguistic ambiguity from factual error through methods such as Semantic Entropy (Kuhn et al., 2023) and SelfCheck-GPT (Manakul et al., 2023), and density-based methods (Vazhentsev et al., 2025). In alignment settings, approaches including ConfPO (Yoon et al., 2025) and uncertainty-aware decoding (Huang et al., 2025b) incorporate confidence estimates to guide preference filtering or generation control, motivating complementary signals that capture tokenlevel informativeness.

## 3 Methodology: Trust-based Uncertainty Sparse Alignment

We propose TUSA (Trust-based Uncertainty Sparse Alignment), a framework designed to mitigate the interference caused by uncertain supervisors by treating alignment as a sparse decision process. Central to this approach is a Cognitive Arbiter, which dynamically regulates the interventions of the Weak Specialist (small supervisor, $\pi _ { \phi } )$ on the Strong Generalist (base model, π ).

## 3.1 Problem Formulation

Training-time alignment of LLMs embeds safety constraints into parameters θ by maximizing an expected reward objective:

$$
\operatorname* { m a x } _ { \theta } \mathbb { E } _ { \mathbf { x } \sim \mathcal { D } , \mathbf { y } \sim \pi _ { \theta } } \left[ R ( \mathbf { x } , \mathbf { y } ) - \beta \mathrm { K L } ( \pi _ { \theta } \| \pi _ { \mathrm { r e f } } ) \right] ,\tag{1}
$$

where D is the optimization dataset, R encodes alignment preferences, and $\beta$ scales the regularization against reference $\pi _ { \mathrm { r e f } }$

Inference-time alignment shifts the control locus from parameter updates to the decoding stage: the base model parameters θ remain fixed, and alignment is enforced by modulating the generation process during decoding. Under this paradigm, alignment can be viewed as a sequence of decisions over whether and how external guidance should influence the next-token distribution. A common formulation defines the dense aligned policy π<sub>dense</sub> by combining the base policy with an auxiliary policy $\pi _ { \phi }$ at every decoding step:

$$
\pi _ { \mathrm { d e n s e } } ( y _ { t } \mid \mathbf { x } , \mathbf { y } _ { < t } ) \propto \mathcal { F } ( \pi _ { \theta } , \pi _ { \phi } ) .\tag{2}
$$

where $\mathcal { F }$ represents a functional mapping that combines signals from both policies to form the final aligned distribution.

To enable precise control, we generalize this to a conditional formulation governed by a gate $g _ { t }$ . Specifically, given a candidate $\hat { y } _ { t } \sim \pi _ { \theta } ( \cdot \ |$ $\mathbf { x } , \mathbf { y } _ { < t } )$ proposed by the base model, the gate $g _ { t } =$ $\mathbb { I } ( R _ { \mathrm { j o i n t } } ( t , \hat { y } _ { t } ) \geq \tau )$ determines whether to invoke the aligned policy $\pi _ { \mathrm { d e n s e } }$ or directly accept yˆ<sub>t</sub>:

$$
y _ { t } \sim { \left\{ \begin{array} { l l } { \pi _ { \mathrm { d e n s e } } ( \cdot \mid \mathbf { x } , \mathbf { y } _ { < t } ) } & { { \mathrm { i f ~ } } g _ { t } = 1 \quad { \mathrm { ( I n t e r v e n e ) } } , } \\ { \delta _ { \hat { y } _ { t } } } & { { \mathrm { i f ~ } } g _ { t } = 0 \quad { \mathrm { ( B y p a s s ) } } . } \end{array} \right. }\tag{3}
$$

where $\delta _ { \hat { y } _ { t } }$ is a degenerate distribution placing all probability mass on $\hat { y } _ { t }$ . Under this mechanism, dense intervention is a special case where $g _ { t } \equiv 1$ while TUSA achieves efficiency by activating the aligned distribution only when $R _ { \mathrm { j o i n t } } ( t , \hat { y } _ { t } )$ signals a necessity for intervention.

## 3.2 The Cognitive Arbiter: Quantifying Trust

## 3.2.1 Signal I: Cognitive Confidence

The raw logits of the lightweight alignment actuator may exhibit poor calibration. To obtain a reliable competence measure, we first apply temperature scaling (Guo et al., 2017) to the raw logits $\bar { l _ { t } } \in \mathbb { R } ^ { | \nu | }$ at step t. The calibrated policy $\pi _ { \phi } ^ { \prime }$ is given by:

$$
\pi _ { \phi } ^ { \prime } ( y _ { t } ) = \mathrm { S o f t m a x } ( l _ { t } / \lambda ) ,\tag{4}
$$

where λ is the temperature hyperparameter. A key observation in Figure 2 is that the Weak Specialists often exhibit high entropy (near-uniform distribution) when facing out-of-distribution contexts. Therefore, we quantify cognitive confidence $C _ { t }$ not as raw entropy, but as the Kullback-Leibler (KL) divergence (Kullback and Leibler, 1951) between the calibrated policy and a Maximum Entropy proxy distribution $P _ { \mathrm { p r o x y } }$ (a uniform distribution over the action space A):

$$
C _ { t } = D _ { K L } ( \pi _ { \phi } ^ { \prime } ( \cdot | t ) | | P _ { \mathrm { p r o x y } } ) .\tag{5}
$$

Expanding this term reveals its relationship to the Shannon entropy H:

$$
\begin{array} { r l r } & { } & { C _ { t } = \displaystyle \sum _ { a \in \mathcal { A } } \pi _ { \phi } ^ { \prime } ( a \vert t ) \log \frac { \pi _ { \phi } ^ { \prime } ( a \vert t ) } { P _ { \mathrm { p r o x y } } ( a ) } } \\ & { } & { ~ = - \mathcal { H } ( \pi _ { \phi } ^ { \prime } ) + \log \vert \mathcal { A } \vert } \end{array}\tag{6}
$$

A higher $C _ { t }$ signals that the Weak Specialist is deviating significantly from a random guess, indicating a distinct, confident preference. This metric effectively differentiates valid safety preferences from the cognitive noise.

## 3.2.2 Signal II: Semantic Saliency

Cognitive certainty is necessary but insufficient; dense intervention often wastes computation on syntactically deterministic tokens. For instance, scrutinizing high-frequency functional words like “the” or $^ { 6 6 } \mathrm { o f } ^ { 9 }$ contributes negligible safety value while inflating inference latency. To address this, we introduce Semantic Saliency $S _ { t }$ , grounded in the token’s Global Information Surprisal. We approximate this using Inverse Document Frequency (IDF) (Sparck Jones, 1972):

$$
\mathcal { T } _ { \mathrm { g l o b a l } } ( y _ { t } ) \approx \log \frac { N } { 1 + d f ( y _ { t } ) } ,\tag{7}
$$

where N is the total document count and $d f ( y _ { t } )$ is the document frequency. To utilize this as a gating signal, we project the surprisal value into a normalized probability space [0, 1]:

$$
S _ { t } = \frac { \mathcal { T } _ { \mathrm { g l o b a l } } ( y _ { t } ) - \mathcal { T } _ { \mathrm { m i n } } } { \mathcal { T } _ { \mathrm { m a x } } - \mathcal { T } _ { \mathrm { m i n } } } .\tag{8}
$$

This metric acts as a semantic high-pass filter, selectively suppressing intervention on low-surprisal syntactic glue (where $S _ { t }  0 )$ while preserving sensitivity to high-information concepts that drive the narrative trajectory. For further discussion on Semantic Saliency, please refer to Appendix D.2.

## 3.2.3 Joint Necessity Estimation

The Cognitive Arbiter fuses the two signals into a single scalar, the Joint Necessity Score $R _ { \mathrm { j o i n t } } ( t )$

$$
R _ { \mathrm { j o i n t } } ( t ) = C _ { t } \cdot S _ { t }\tag{9}
$$

This multiplicative coupling enforces a rigorous trust boundary: intervention $( g _ { t } = 1 )$ is authorized only when the Weak Specialist is both cognitively certain and the token is semantically salient. This effectively filters out incompetent guessing and trivial micromanagement, realizing the principle of Trust-Based Consultation.

## 3.3 Adaptive Execution Protocol

A static threshold is suboptimal due to the varying baseline entropy across contexts (Kuhn et al., 2023). To address this, we employ an Adaptive Thresholding Mechanism that calibrates the boundary based on the local generation trajectory.

We maintain a sliding window $H _ { t }$ $\{ R _ { \mathrm { j o i n t } , t - K } , \dots , R _ { \mathrm { j o i n t } , t - 1 } \}$ of the last K tokens. The dynamic threshold $\tau _ { t }$ is computed as the scaled moving average of this history:

$$
\tau _ { t } = \alpha \cdot \mathbb { E } _ { k \in [ t - K , t - 1 ] } [ R _ { \mathrm { j o i n t } , k } ] = \alpha \cdot \left( \frac { 1 } { | H _ { t } | } \sum _ { R \in H _ { t } } R \right)\tag{10}
$$

The hyperparameter α controls the safetyefficiency trade-off: a lower $\alpha$ lowers the barrier (favoring high safety recall), while a higher α enforces stricter sparsity (favoring inference speed and precision). Meanwhile, the Window Size K regulates the baseline’s inertia: a larger K provides a stable historical context to smooth out transient noise, whereas a smaller K allows for rapid adaptation to sudden semantic shifts.

This adaptive mechanism ensures TUSA robustness to distribution shifts: high-entropy states naturally suppress $R _ { j o i n t }$ toward zero. While $\tau _ { t }$ tracks this floor, our multiplicative design filters background noise, authorizing intervention only when relative necessity spikes break the local baseline.

## 3.4 Algorithmic Instantiation

TUSA pipeline (Algorithm 1) operates as follows: For a candidate token $y _ { t } ^ { ( 1 ) }$ sampled from $\pi _ { \boldsymbol { \theta } } .$ , we compute $R _ { \mathrm { j o i n t } }$ . If $R _ { \mathrm { j o i n t } } < \tau _ { t }$ , the system triggers the Trust Pathway, bypassing the Specialist and directly adopting the base model’s output:

$$
y _ { t }  y _ { t } ^ { ( 1 ) } \quad ( \mathrm { i f } R _ { \mathrm { j o i n t } } < \tau _ { t } )\tag{11}
$$

Otherwise $( g _ { t } = 1 ) , \pi _ { \phi }$ evaluates $y _ { t } ^ { ( 1 ) }$

## 4 Experiments

## 4.1 Experimental Setup

Models and Datasets. We evaluate TUSA using a diverse suite of Strong Generalist base models, including the Llama family (Llama-3.1-8B, Llama-3.2-3B) (AI@Meta, 2024) and the Mistral-7B series (v0.1, v0.2, v0.3) (Jiang et al., 2023). These models are guided by the 4M-parameter microagent (the Weak Specialist) from MARA (Zhang et al., 2025). Our evaluation protocol spans two distinct domains. For Safety Alignment, we utilize three standard benchmarks: PKU-SafeRLHF (Ji et al., 2025), BeaverTails (Ji et al., 2023), and HarmfulQA (Bhardwaj and Poria, 2023). For General Capabilities, we assess the impact of alignment interventions on AlpacaEval (Li et al., 2023c) and JustEval (Lin et al., 2024a) for instruction following. For more detailed information, please refer to Appendix B.1.

Algorithm 1 TUSA Execution Protocol   
Require: Base Model $\pi _ { \boldsymbol { \theta } } .$ , Weak Specialist $\pi _ { \phi } .$   
Safety Coeff α   
1: Initialize history buffer $H \gets \{ \tau _ { 0 } \}$   
2: for $t = 1$ to $T$ do   
3: Get $h _ { t }$ and Top-k candidates $\nu _ { k }$ from $\pi _ { \boldsymbol { \theta } } ;$   
$y _ { t } \gets \mathrm { P o p } ( \mathcal { V } _ { k } )$   
4: $R _ { t } \gets \mathrm { C a l c J o i n t R i s k } ( \pi _ { \phi } ( h _ { t } ) , y _ { t } )$   
5: $\tau _ { t }  \alpha$ · DynamicThreshold $( H )$   
6: // High Risk: Apply Guidance   
7: if $R _ { t } \ge \tau _ { t }$ then   
8: while arg max $\pi _ { \phi } ( h _ { t } ) = = \mathbb { R }$ EJECT and   
$R _ { t } \ge \tau _ { t }$ do   
9: $y _ { t } \gets \mathrm { P o p } ( \mathcal V _ { k } ) ;$   
10: $R _ { t } \gets$ CalcJointRisk $( \pi _ { \phi } ( h _ { t } ) , y _ { t } )$   
11: end while   
12: end if   
13: // Low Risk: Trust Base Model   
14: Update H with $R _ { t } ;$ Append $y _ { t }$ to output   
15: end for

Baselines. We compare TUSA against three representative setups: (1) the original Base Model; (2) MARA (Zhang et al., 2025), representing the established dense intervention method; and (3) ConfPO (Yoon et al., 2025), a leading method for conditional training-time preference optimization alignment frameworks. Unlike our training-free inference-time approach, ConfPO updates model parameters during training and does not rely on a Weak Specialist during decoding. This inclusion allows us to benchmark TUSA against a strong uncertainty-aware parameter-updating paradigm. The details of our baselines are provided in Appendix B.2.

Unified Evaluation Framework. To ensure a unified evaluation standard and reproducibility across all experiments, we employ the open-source Beaver-7B (Dai et al., 2024) suite as our judge, which provides decoupled supervisory signals for distinct model attributes. Specifically, we use beaver-cost to evaluate safety (Harmlessness)

<table><tr><td rowspan="2">Base Model</td><td rowspan="2">TUSA (ours) vs. baseline</td><td colspan="3">SafeRLHF</td><td colspan="3">BeaverTails</td><td colspan="3">HarmfulQA</td></tr><tr><td>H↑</td><td>Ha↑</td><td>P↑</td><td>H↑</td><td>Ha↑</td><td>P↑</td><td>H↑</td><td>Ha↑</td><td>P↑</td></tr><tr><td rowspan="3">Mistral-7B-v0.1</td><td>Base Model</td><td>-0.5</td><td>27.1</td><td>13.6</td><td>10.6</td><td>22.6</td><td>16.6</td><td>-3.7</td><td>36.0</td><td>22.1</td></tr><tr><td>ConfPO</td><td>1.5</td><td>25.1</td><td>13.6</td><td>11.4</td><td>22.4</td><td>17.0</td><td>-5.5</td><td>27.7</td><td>20.5</td></tr><tr><td>MARA</td><td>36.2</td><td>-5.5</td><td>15.6</td><td>28.3</td><td>-11.6</td><td>8.4</td><td>17.2</td><td>-7.2</td><td>5.1</td></tr><tr><td rowspan="3">Mistral-7B-v0.2</td><td>Base Model</td><td>6.5</td><td>2.5</td><td>4.5</td><td>5.3</td><td>15.3</td><td>10.3</td><td>-4.1</td><td>20.9</td><td>8.4</td></tr><tr><td>ConfPO</td><td>-12.6</td><td>12.6</td><td>-0.5</td><td>10.4</td><td>10.4</td><td>10.4</td><td>4.5</td><td>14.1</td><td>9.4</td></tr><tr><td>MARA</td><td>13.6</td><td>4.5</td><td>9.1</td><td>28.9</td><td>-19.6</td><td>4.7</td><td>11.9</td><td>-6.4</td><td>2.9</td></tr><tr><td rowspan="3">Mistral-7B-v0.3</td><td>Base Model</td><td>14.6</td><td>0.0</td><td>7.0</td><td>7.1</td><td>-1.6</td><td>2.9</td><td>10.3</td><td>17.6</td><td>13.9</td></tr><tr><td>ConfPO</td><td>6.5</td><td>-2.0</td><td>2.0</td><td>7.1</td><td>-0.9</td><td>3.3</td><td>5.7</td><td>13.9</td><td>9.8</td></tr><tr><td>MARA</td><td>6.0</td><td>-1.5</td><td>2.5</td><td>7.1</td><td>-10.1</td><td>-1.4</td><td>9.0</td><td>-5.7</td><td>0.8</td></tr><tr><td rowspan="3">Llama 3.1-8B</td><td>Base Model</td><td>14.6</td><td>3.5</td><td>9.1</td><td>22.3</td><td>-0.7</td><td>10.7</td><td>6.0</td><td>3.0</td><td>4.5</td></tr><tr><td>ConfPO</td><td>15.6</td><td>3.0</td><td>9.1</td><td>23.4</td><td>-8.6</td><td>7.4</td><td>7.8</td><td>-5.3</td><td>1.2</td></tr><tr><td>MARA</td><td>3.0</td><td>3.5</td><td>3.5</td><td>-2.0</td><td>12.6</td><td>5.4</td><td>3.5</td><td>11.1</td><td>7.4</td></tr><tr><td rowspan="3">Llama 3.2-3B</td><td>Base Model</td><td>28.1</td><td>-1.0</td><td>13.6</td><td>19.1</td><td>12.9</td><td>16.1</td><td>24.8</td><td>12.9</td><td>18.9</td></tr><tr><td>ConfPO</td><td>24.6</td><td>-3.5</td><td>10.6</td><td>18.4</td><td>14.0</td><td>16.3</td><td>22.5</td><td>12.7</td><td>17.6</td></tr><tr><td>MARA</td><td>3.0</td><td>4.0</td><td>4.0</td><td>8.0</td><td>-0.6</td><td>3.7</td><td>0.4</td><td>1.6</td><td>1.0</td></tr><tr><td>Average</td><td>一</td><td>10.7</td><td>4.8</td><td>7.8</td><td>13.7</td><td>3.8</td><td>8.8</td><td>7.4</td><td>9.8</td><td>9.6</td></tr></table>

Table 1: Relative performance improvements (%). The second column lists the baseline method against which TUSA is compared. Positive values indicate TUSA outperforms the baseline. Metrics: Helpful (H), Harmless (Ha) and Preference (P).

and beaver-reward to assess utility (Helpfulness) and overall quality across datasets. Comprehensive analysis of the judge’s reliability is provided in Appendix E.

Evaluation Metrics. We report the Guidance Proportion to quantify the sparsity during inference. Specifically, it represents the ratio of guided tokens to the total sequence length. For performance quality, we adopt the Preference Rate (w) to quantify the improvement over the baseline. This metric accounts for ties and is calculated as:

$$
w = \frac { N _ { \mathrm { w i n } } - N _ { \mathrm { l o s e } } } { N _ { \mathrm { w i n } } + N _ { \mathrm { t i e } } + N _ { \mathrm { l o s e } } } \times 1 0 0 \%\tag{12}
$$

where $N _ { \mathrm { w i n } } , N _ { \mathrm { t i e } }$ , and $N _ { \mathrm { l o s e } }$ represent the counts of wins, ties, and losses in pairwise comparisons against the base model. We report this metric across three dimensions: Harmlessness, Helpfulness, and Overall Preference. More information about our evaluation is provided in Appendix B.3.

## 4.2 Main Results

Safety Alignment. Table 1 reports the Preference Rate across benchmarks, where TUSA shows consistent superiority. On average, it attains gains of +7.8% on SafeRLHF, +8.8% on BeaverTails, and +9.6% on HarmfulQA. As detailed via absolute Win/Lose counts in Appendix C.4, the improvement magnitude correlates with base model intrinsic alignment, which is exemplified by the substantial +15.6% gain achieved on the less-aligned Mistral-v0.1 compared with Mistral-v0.3, while surpassing ConfPO by 13.6%. This confirms that uncertainty-aware sparsity effectively mitigates the safety-capability trade-off. By selectively intervening only at critical junctures, TUSA ensures rigorous safety compliance while preserving, or even actively enhancing, the intrinsic helpfulness and general reasoning capabilities of the base model.

<table><tr><td rowspan="2">Model</td><td colspan="3">AlpacaEval</td><td colspan="3">JustEval</td></tr><tr><td>H↑</td><td>Ha↑</td><td>P↑</td><td>H↑</td><td>Ha↑</td><td>P↑</td></tr><tr><td>Mistral-v0.1 Mistral-v0.2</td><td>4.5 7.0</td><td>-1.5 11.0</td><td>1.5 9.0</td><td>12.0 5.4</td><td>-6.6 -0.6</td><td>2.4 2.4</td></tr><tr><td>Mistral-v0.3 Llama 3.1</td><td>5.5 2.0</td><td>-3.0 14.0</td><td>1.5 8.0</td><td>1.8 5.4</td><td>3.0 13.1</td><td>2.4 9.0</td></tr><tr><td>Llama 3.2</td><td>3.0</td><td>6.0</td><td>4.5</td><td>17.4</td><td>7.2</td><td>12.0</td></tr><tr><td>Average</td><td>4.4</td><td>5.3</td><td>4.9</td><td>8.4</td><td>3.2</td><td>5.6</td></tr></table>

Table 2: Performance improvement of TUSA over the dense baseline on general benchmarks.

General Capabilities. As shown in Table 2, TUSA consistently outperforms the dense baseline, achieving average preference gains of +4.9% on AlpacaEval and +5.6% on JustEval. Notably, Helpfulness improves significantly (e.g., +17.4% on JustEval for Llama 3.2-3B), indicating that our sparse mechanism effectively bypasses the overcorrection typical of dense supervision while simultaneously achieving Harmlessness improvements (e.g., +14.0%). TUSA successfully preserves the base model’s reasoning and instruction-following abilities by restricting interventions strictly to safety-critical contexts.

![](images/56e0de4530499a13497aeb0da2b7505fc97eb1247a8392b7d315e6f9503b4511.jpg)  
(a) Inference Latency Comparison across models.

![](images/b8480a4a17d865e6d80c4fd7a02b8af7e504237adac121195cff38baf9e0fc34.jpg)  
(b) Best Guidance Proportion across benchmarks.  
Figure 4: Analysis of Computational Efficiency and Adaptive Intervention Dynamics.

Efficiency and Adaptive Sparsity. Figure 4 illustrates the computational efficiency and intervention dynamics. As shown in Figure 4a, TUSA maintains a latency comparable to the dense baseline. Rather than radically accelerating the unaligned base model, TUSA aims to balance general utility and safety within essentially the similar inference time budget as dense supervision. This is achieved because the constant, lightweight overhead of the Cognitive Arbiter is effectively offset by completely bypassing heavy guided decoding operations on benign tokens. Detailed componentwise profiling in Appendix D.3 confirms this efficiency, demonstrating an over 70% reduction in heavy Weak Specialist evaluations and revealing that the comparable end-to-end latency is inherently dominated by the base model’s basic generation cost. Figure 4b validates the efficacy of our sensitivity calibration. Under an optimized trust boundary, the system exhibits highly contextdependent behaviors: it maintains high vigilance on safety benchmarks (e.g., ∼60% on SafeRLHF)

![](images/e8e2585a50d8497af205143c2c16626e1d3e24e8b775c564424642a0dc50cf33.jpg)  
Figure 5: Distribution of intervened tokens across Partof-Speech (POS) categories. TUSA shifts focus toward content-rich tokens compared to the dense baseline.

![](images/8016212deed20902dc9582ae4b6969216fe15067211fe415e34ab6d17c8985e7.jpg)  
Figure 6: Qualitative visualization of the Joint Necessity of different tokens.

while relaxing constraints on general utility tasks (dropping to ∼45% on AlpacaEval). This confirms that our mechanism successfully differentiates varying risk levels and allocates the computational budget where it is most needed for effective alignment.

## 4.3 Mechanism Analysis

Semantic Saliency Verification. Figure 5 presents a comparative analysis of the Part-of-Speech (POS) distribution for guided tokens. As illustrated, TUSA exhibits a pronounced shift in intervention focus toward content-rich categories: the proportion of intervened NOUN and VERB tokens increases significantly compared to the baseline, indicating a targeted emphasis on the entities and actions central to the narrative. Conversely, interventions on functional categories—specifically DET, ADP, and PUNCT—are drastically reduced, often dropping to less than half of the baseline frequency. This structural redistribution confirms that the Cognitive Arbiter effectively isolates semantic saliency from syntactic scaffolding. By allocating the guidance budget strictly to pivotal concepts while bypassing low-risk syntactic redundancies, the system ensures that computational resources are concentrated solely on the decision points most critical for safety alignment.

![](images/71139104336f9488a2e74d91de5b6a58e28b5eaf753ca169172fa70394b9512d.jpg)  
(a) α on SafeRLHF

![](images/08d9f78d60fd7d887c299b965b852c0f40bb76a2e850664ceb25b20745f83790.jpg)  
(b) α on BeaverTails

![](images/11b6c2f1082e1c3acd42af197ae7403fefec93e47826125b1a2b8cff11acbb73.jpg)  
(c) λ on SafeRLHF

![](images/e00e55d4e30a6d4b42ae78057d5afc4a990aa25893b43daf5d622cda31d4905b.jpg)  
(d) λ on BeaverTails

Figure 7: Hyperparameter ablation on Mistral-v0.1. (a)-(b) show the impact of Safety Coefficient α, and (c)-(d) show Temperature Coefficient λ. The red dashed line denotes the Guidance Proportion.  
![](images/ed975d140dbd92f242e0bf255abed9c15af76b0724fe7c21e44283dcb2171f1a.jpg)  
(a) α on SafeRLHF

![](images/8d621b97b19cbfb6d8ada977b83e846da52123947ac2cb539d83722b8e5f2ebd.jpg)  
(b) α on BeaverTails

![](images/b9ab418845542d21580d7603ba8877aa5dd517792245ef8d1f6c84b249d28ca7.jpg)  
(c) λ on SafeRLHF

![](images/1ac50f8a287d35539f67c703619eaea9bcea5dae3d5504fc0beecb269bad63fe.jpg)  
(d) λ on BeaverTails

Figure 8: Hyperparameter ablation on Llama-3.1-8B. (a)-(b) show the impact of safety coefficient α, and (c)-(d) show temperature coefficient λ. The red dashed line denotes the Guidance Proportion.  
![](images/0b40005b693d191817cdf5e0382e4be59242b7ed79fef27613fc5fb7723330d7.jpg)  
(a) Llama-3.1-8B

![](images/ff7fdb257f9664699b2c7586da646a197fd4993ef89f0b6feabe5e70f8b56acf.jpg)  
(b) Mistral-v0.3-7B  
Figure 9: Visualization of the Dynamic Arbitration mechanism. The purple line represents the Joint Risk score $( R _ { \mathrm { j o i n t } } )$ , and the dashed orange line indicates the adaptive Dynamic Threshold (τ ).

What does TUSA Skip? Figure 6 qualitatively validates TUSA’s filtering. The high-necessity cluster accurately isolates safety-critical concepts, prioritizing tokens like “trafficking”, “atomic”, and “racist” that exhibit both high Specialist Confidence and Semantic Saliency. In sharp contrast, low scores are assigned almost exclusively to benign function words and numerals (e.g., “the”, “is”, “5”). This confirms that the arbiter effectively bypasses syntactic redundancy, concentrating supervision solely on genuine, high-impact risks.

Dynamic Arbitration Dynamics. Figure 2 establishes a critical premise: the Weak Specialist operates with a persistent noise profile, rendering absolute confidence unreliable. Addressing this, Figure 9 demonstrates our solution: the dynamic threshold actively tracks this local uncertainty baseline. This mechanism effectively performs background subtraction across the entire decoding sequence, shifting arbitration from relying on absolute confidence to detecting relative semantic spikes. Consequently, interventions are triggered strictly by genuine risk signals (purple peaks) that significantly exceed the prevailing ambient noise.

## 4.4 Ablation Studies

Effectiveness of Trust Strategy. To verify that our performance gains stem from the strategic selection of tokens rather than mere sparsity, we compared TUSA against a baseline that applies guidance to random tokens at equivalent sparsity levels. As detailed in Table 3, TUSA consistently outperforms the stochastic approach across all base models and benchmarks. Notably, we achieve average preference improvements of 12.2% on BeaverTails and 7.9% on SafeRLHF. These results confirm that effective alignment requires precision, not just reduction: the Cognitive Arbiter successfully isolates critical risk points that random selection misses, proving that where we intervene is as crucial as how much we intervene.

Impact of Semantic Saliency. Table 4 empirically validates the role of IDF. The “w/o IDF” variant (relying solely on raw confidence C<sub>t</sub>) yields limited gains (< 5%) as it fails to filter syntactic noise from raw entropy. In contrast, TUSA consistently achieves robust improvements, proving that filtering low-value tokens shifts alignment focus onto critical positions, ultimately driving substantial gains in safety and utility.

<table><tr><td rowspan="2">Base Model</td><td colspan="3">Preference Rate (%)↑</td></tr><tr><td>SafeRLHF</td><td>BeaverTails</td><td>HarmfulQA</td></tr><tr><td>Mistral-7B-v0.1</td><td>18.1</td><td>15.6</td><td>14.8</td></tr><tr><td>Mistral-7B-v0.2</td><td>3.0</td><td>19.7</td><td>1.4</td></tr><tr><td>Mistral-7B-v0.3</td><td>8.0</td><td>6.0</td><td>1.0</td></tr><tr><td>Llama 3.1-8B</td><td>2.5</td><td>6.3</td><td>4.1</td></tr><tr><td>Llama 3.2-3B</td><td>8.0</td><td>13.6</td><td>8.6</td></tr><tr><td>Average</td><td>7.9</td><td>12.2</td><td>6.0</td></tr></table>

Table 3: Performance improvements of TUSA over random guidance. Values indicate the percentage increase in preference rate (%).
<table><tr><td rowspan="2">Method</td><td colspan="5">Safety Coefficient α</td></tr><tr><td>0.4</td><td>0.6 0.8</td><td>1.0</td><td>1.2</td><td>1.4</td></tr><tr><td>w/o IDF</td><td>0.0</td><td>2.5 1.0</td><td>1.5</td><td>4.5</td><td>2.5</td></tr><tr><td>TUSA (ours)</td><td>8.0</td><td>11.0</td><td>6.5 10.0</td><td>10.0</td><td>4.5</td></tr></table>

Table 4: Ablation on Semantic Saliency. Values denote the preference rate improvement (%) over dense baseline. The “w/o IDF” variant relies solely on confidence (C<sub>t</sub>) without semantic weighting (S<sub>t</sub>).

We evaluate the sensitivity and performanceefficiency trade-offs across varying safety coefficients α and temperatures λ on Mistral-v0.1 (Figure 7) and Llama-3.1-8B (Figure 8). Across both backbones, increasing α tightens filtering criteria, slashing the Guidance Proportion from over 80% to under 20% while maintaining a consistently high Preference Rate. This confirms that TUSA robustly eliminates redundant interventions across architectures. Temperature λ controls arbiter sensitivity to uncertainty. With optimal ranges originally identified on a held-out validation set, test-set evaluations show stable gains around default λ ≈ 1.1, while naturally aligning with model-specific calibration—peaking at λ ≈ 1.6–1.8 for Mistral-v0.1 and λ ≈ 1.0–1.2 for Llama-3.1-8B. Properly reflecting the model’s confidence distribution reliably maximizes safety-capability gains. Extended analyses are in Appendix C.1.

## 5 Conclusion

We introduced TUSA (Trust-based Uncertainty Sparse Alignment) to reconcile safety alignment with general text-generation capabilities. Moving beyond dense intervention, our framework leverages a lightweight Cognitive Arbiter to dynamically quantify uncertainty, employing an adaptive threshold to selectively intervene on genuine risk spikes. By filtering redundant guidance on benign tokens, TUSA avoids excessive behavioral over-correction on safe contexts. Crucially, this selective intervention surpasses dense methods by protecting base model intelligence without sacrificing core safety quality. This work demonstrates that precision-driven sparse alignment outperforms dense intervention, providing a high-fidelity alignment framework for broader generative tasks.

## Limitations

Despite the promising results, our framework entails certain limitations that merit discussion. First, due to computational resource constraints, our empirical validation primarily focuses on models ranging from 3B to 8B parameters. While TUSA demonstrates significant gains at this scale, extending the framework to larger foundation models remains an important future direction to verify scalability. Second, our method currently operates under a white-box assumption, requiring access to the model’s output logits for uncertainty estimation. This precludes direct application to closed-source, API-based models where internal states are inaccessible. Finally, as with all weak-to-strong generalization paradigms, the system’s ultimate upper bound is correlated with the weak supervisor’s capability. While effective for standard safety risks, there remains a theoretical challenge in handling extremely subtle or complex adversarial scenarios where the weak supervisor itself may lack the necessary nuance, leaving room for future exploration in more advanced alignment transfer mechanisms.

## Ethical Considerations

This work aims to democratize rigorous safety alignment by significantly reducing computational overhead, aligning with Green AI principles. However, we acknowledge that the underlying framework is objective-agnostic: malicious actors could theoretically invert the guidance to steer models toward harmful content, and any inherent biases in the specialist model may be amplified during intervention. Therefore, we strongly advise practitioners to rigorously audit the fairness and calibration of the guiding model prior to deployment, ensuring that efficiency gains do not compromise ethical integrity.

## Acknowledgements

This work was supported in part by National Natural Science Foundation of China (62476070), Shenzhen Science and Technology Program (JCYJ2024

1202123503005, GXWD20231128103232001, Z DSYS20230626091203008, KQTD20240729102 154066), Department of Science and Technology of Guangdong (2024A1515011540).

## References

AI@Meta. 2024. Llama 3 model card.

Rishabh Bhardwaj and Soujanya Poria. 2023. Redteaming large language models using chain of utterances for safety-alignment. arXiv preprint arXiv:2308.09662.

Collin Burns, Pavel Izmailov, Jan Hendrik Kirchner, Bowen Baker, Leo Gao, Leopold Aschenbrenner, Yining Chen, Adrien Ecoffet, Manas Joglekar, Jan Leike, Ilya Sutskever, and Jeffrey Wu. 2024. Weakto-strong generalization: Eliciting strong capabilities with weak supervision. In Proceedings of the 41st International Conference on Machine Learning (ICML), pages 4971–5012.

Paul F. Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, and Dario Amodei. 2017. Deep reinforcement learning from human preferences. In Proceedings ofthe 31st International Conference on Neural Information Processing Systems (NeurIPS), pages 4299–4307.

Yung-Sung Chuang, Yujia Xie, Hongyin Luo, Yoon Kim, James R Glass, and Pengcheng He. 2024. DoLa: Decoding by contrasting layers improves factuality in large language models. In Proceedings of the 12th International Conference on Learning Representations (ICLR).

Juntao Dai, Xuehai Pan, Ruiyang Sun, Jiaming Ji, Xinbo Xu, Mickel Liu, Yizhou Wang, and Yaodong Yang. 2024. Safe RLHF: Safe reinforcement learning from human feedback. In Proceedings of the Twelfth International Conference on Learning Representations (ICLR).

Sumanth Dathathri, Andrea Madotto, Janice Lan, Jane Hung, Eric Frank, Piero Molino, Jason Yosinski, and Rosanne Liu. 2020. Plug and play language models: A simple approach to controlled text generation. In Proceedings ofthe 8th International Conference on Learning Representations (ICLR).

Junhao Dong, Cong Zhang, Xinghua Qu, Zejun Ma, Piotr Koniusz, and Yew-Soon Ong. 2025. Robust SuperAlignment: Weak-to-strong robustness generalization for vision-language models. In Proceedings ofthe 39th Annual Conference on Neural Information Processing Systems (NeurIPS).

Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. 2017. On calibration of modern neural networks. In Proceedings ofthe 34th International Conference on Machine Learning (ICML), pages 1321– 1330.

Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. 2018. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In Proceedings of the Thirty-Fifth International Conference on Machine Learning (ICML), pages 1861–1870. Pmlr.

James Y. Huang, Sailik Sengupta, Daniele Bonadiman, Yi-an Lai, Arshit Gupta, Nikolaos Pappas, Saab Mansour, Katrin Kirchhoff, and Dan Roth. 2025a. DeAL: Decoding-time alignment for large language models. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL), pages 26280–26300.

Yuheng Huang, Jiayang Song, Zhijie Wang, Shengming Zhao, Huaming Chen, Felix Juefei-Xu, and Lei Ma. 2025b. Look before you leap: An exploratory study of uncertainty analysis for large language models. IEEE Transactions on Software Engineering (TSE).

Hakan Inan, Kartikeya Upasani, Jianfeng Chi, Rashi Rungta, Krithika Iyer, Yuning Mao, Michael Tontchev, Qing Hu, Brian Fuller, Davide Testuggine, and Madian Khabsa. 2023. Llama guard: Llm-based input-output safeguard for human-ai conversations.

Jiaming Ji, Boyuan Chen, Hantao Lou, Donghai Hong, Borong Zhang, Xuehai Pan, Juntao Dai, Tianyi Qiu, and Yaodong Yang. 2024. Aligner: Efficient alignment by learning to correct. In Proceedings of the 38th International Conference on Neural Information Processing Systems (NeurIPS).

Jiaming Ji, Donghai Hong, Borong Zhang, Boyuan Chen, Josef Dai, Boren Zheng, Tianyi Alex Qiu, Jiayi Zhou, Kaile Wang, Boxun Li, Sirui Han, Yike Guo, and Yaodong Yang. 2025. PKU-SafeRLHF: Towards multi-level safety alignment for LLMs with human preference. In Proceedings of the 63rd Annual Meeting ofthe Associationfor Computational Linguistics (ACL), pages 31983–32016.

Jiaming Ji, Mickel Liu, Josef Dai, Xuehai Pan, Chi Zhang, Ce Bian, Boyuan Chen, Ruiyang Sun, Yizhou Wang, and Yaodong Yang. 2023. BeaverTails: Towards improved safety alignment of LLM via a human-preference dataset. In Proceedings of the 37th International Conference on Neural Information Processing Systems (NeurIPS), pages 24678–24704.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2023. Mistral 7b. Preprint, arXiv:2310.06825.

Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, Scott Johnston, Sheer El Showk, Andy Jones, Nelson Elhage, Tristan Hume, Anna Chen,

Yuntao Bai, Sam Bowman, Stanislav Fort, and 17 others. 2022. Language models (mostly) know what they know. arXiv preprint arXiv:2207.05221.

Maxim Khanov, Jirayu Burapacheep, and Yixuan Li. 2024. ARGS: Alignment as reward-guided search. In Proceedings ofthe 12th International Conference on Learning Representations (ICLR).

Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar. 2023. Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation. In Proceedings ofthe 11th International Conference on Learning Representations (ICLR).

Solomon Kullback and Richard A Leibler. 1951. On information and sufficiency. The annals of mathematical statistics, 22(1):79–86.

Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, and Martin Wattenberg. 2023a. Inferencetime intervention: Eliciting truthful answers from a language model. In Proceedings ofthe 37th International Conference on Neural Information Processing Systems (NeurIPS), pages 41451–41530.

Xiang Lisa Li, Ari Holtzman, Daniel Fried, Percy Liang, Jason Eisner, Tatsunori Hashimoto, Luke Zettlemoyer, and Mike Lewis. 2023b. Contrastive Decoding: Open-ended text generation as optimization. In Proceedings ofthe 61st Annual Meeting ofthe Associationfor Computational Linguistics (ACL), pages 12286–12312.

Xuechen Li, Tianyi Zhang, Yann Dubois, Rohan Taori, Ishaan Gulrajani, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. 2023c. Alpacaeval: An automatic evaluator of instruction-following models. https://github.com/tatsu-lab/alpaca\_eval.

Baijiong Lin, Weisen Jiang, Yuancheng Xu, Hao Chen, and Ying-Cong Chen. 2025. PARM: Multi-objective test-time alignment via preference-aware autoregressive reward model. In Proceedings ofthe 42nd International Conference on Machine Learning (ICML).

Bill Yuchen Lin, Abhilasha Ravichander, Ximing Lu, Nouha Dziri, Melanie Sclar, Khyathi Chandu, Chandra Bhagavatula, and Yejin Choi. 2024a. The unlocking spell on base LLMs: Rethinking alignment via in-context learning. In Proceedings of the Twelfth International Conference on Learning Representations (ICLR).

Zhen Lin, Shubhendu Trivedi, and Jimeng Sun. 2024b. Generating with confidence: Uncertainty quantification for black-box large language models. Transactions on Machine Learning Research (TMLR).

Alisa Liu, Xiaochuang Han, Yizhong Wang, Yulia Tsvetkov, Yejin Choi, and Noah A. Smith. 2024. Tuning language models by proxy. In Proceedings of the 1st Conference on Language Modeling (COLM).

Alisa Liu, Maarten Sap, Ximing Lu, Swabha Swayamdipta, Chandra Bhagavatula, Noah A Smith, and Yejin Choi. 2021. DExperts: Decoding-time controlled text generation with experts and anti-experts. In Proceedings ofthe 59th Annual Meeting ofthe Associationfor Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, pages 6691–6706.

Potsawee Manakul, Adian Liusie, and Mark Gales. 2023. SelfCheckGPT: Zero-resource black-box hallucination detection for generative large language models. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 9004–9017.

Sidharth Mudgal, Jong Lee, Harish Ganapathy, YaGuang Li, Tao Wang, Yanping Huang, Zhifeng Chen, Heng-Tze Cheng, Michael Collins, Trevor Strohman, Jilin Chen, Alex Beutel, and Ahmad Beirami. 2024. Controlled decoding from language models. In Proceedings of the 41st International Conference on Machine Learning (ICML).

OpenAI. 2024. GPT-4o system card. arXiv preprint arXiv:2410.21276.

OpenAI. 2025. gpt-oss-120b & gpt-oss-20b model card. Preprint, arXiv:2508.10925.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F. Christiano, Jan Leike, and Ryan Lowe. 2022. Training language models to follow instructions with human feedback. In Proceedings of the 36th International Conference on Neural Information Processing Systems (NeurIPS), pages 27730–27744.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D. Manning, Stefano Ermon, and Chelsea Finn. 2023. Direct preference optimization: Your language model is secretly a reward model. In Proceedings of the 37th International Conference on Neural Information Processing Systems (NeurIPS), pages 53728– 53741.

Ruizhe Shi, Yifang Chen, Yushi Hu, Alisa Liu, Hannaneh Hajishirzi, Noah A. Smith, and Simon S. Du. 2024. Decoding-time language model alignment with multiple objectives. In Proceedings ofthe 38th International Conference on Neural Information Processing Systems (NeurIPS), pages 48875–48920.

Karen Sparck Jones. 1972. A statistical interpretation of term specificity and its application in retrieval. Journal ofdocumentation, 28(1):11–21.

Nishant Subramani, Nivedita Suresh, and Matthew E. Peters. 2022. Extracting latent steering vectors from pretrained language models. In Findings of the Associationfor Computational Linguistics: ACL 2022, pages 566–581.

Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J Vazquez, Ulisse Mini, and Monte MacDiarmid. 2023. Steering language models with activation engineering. arXiv preprint arXiv:2308.10248.

Artem Vazhentsev, Lyudmila Rvanova, Ivan Lazichny, Alexander Panchenko, Maxim Panov, Timothy Baldwin, and Artem Shelmanov. 2025. Token-level density-based uncertainty quantification methods for eliciting truthfulness of large language models. In Proceedings ofthe 2025 Conference ofthe Nations of the Americas Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (NAACL), pages 2246–2262.

Yuancheng Xu, Udari Madhushani Sehwag, Alec Koppel, Sicheng Zhu, Bang An, Furong Huang, and Sumitra Ganesh. 2025. GenARM: Reward guided generation with autoregressive reward model for test-time alignment. In Proceedings ofthe 13th International Conference on Learning Representations (ICLR).

Hee Suk Yoon, Eunseop Yoon, Mark A Hasegawa-Johnson, Sungwoong Kim, and Chang D Yoo. 2025. ConfPO: Exploiting policy model confidence for critical token selection in preference optimization. In Proceedings of the 42nd International Conference on Machine Learning (ICML).

Yige Yuan, Teng Xiao, Yunfan Li, Bingbing Xu, Shuchang Tao, Yunqi Qiu, Huawei Shen, and Xueqi Cheng. 2025. Inference-time alignment in continuous space. In Proceedings ofthe 39th International Conference on Neural Information Processing Systems (NeurIPS).

Yang Zhang, Yu Yu, Bo Tang, Yu Zhu, Chuxiong Sun, Wenqiang Wei, Jie Hu, Zipeng Xie, Zhiyu Li, Feiyu Xiong, and Edward Chung. 2025. Token-level accept or reject: A micro alignment approach for large language models. In Proceedings of the 34th International Joint Conference on Artificial Intelligence (IJCAI), pages 8375–8383.

Wenhong Zhu, Zhiwei He, Xiaofeng Wang, Pengfei Liu, and Rui Wang. 2025. Weak-to-Strong Preference Optimization: Stealing reward from weak aligned model. In Proceedings of the 13th International Conference on Learning Representations (ICLR).

Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, Shashwat Goel, Nathaniel Li, Michael J. Byun, Zifan Wang, Alex Mallen, Steven Basart, Sanmi Koyejo, Dawn Song, Matt Fredrikson, and 2 others. 2023. Representation Engineering: A top-down approach to AI transparency. arXiv preprint arXiv:2310.01405.

## Appendix

A Implementation Details 14   
A.1 The Weak Specialist: Micro-Agent Architecture . 14   
A.2 Construction of Semantic Saliency Atlas . 14   
B Experiment Details 15   
B.1 Models and Datasets 15   
B.2 Baselines 16   
B.3 Evaluation Protocol 16   
B.4 Parameter Setting 16   
B.5 Computing Resources . 17   
C Additional Experiment Results 18   
C.1 Additional Ablation Studies . . 18   
C.2 Analysis of Optimal Guidance Proportion 18   
C.3 Statistical Significance Analysis 19   
C.4 Detailed Win-Rate Statistics 19   
D Analysis and Discussion 19   
D.1 Analysis between Trust and Entropy 19   
D.2 Analysis and Further Discussion of S<sub>t</sub> 20   
D.3 Detailed Analysis of Computational Overhead and Sparsity 21   
E LLM judge quality validation 22   
F Prompt 22   
G Case Study 23   
H The Use of Large Language Models 23

## A Implementation Details

To ensure reproducibility, we provide a comprehensive description of the model architectures, training configurations, and inference hyperparameters used in TUSA.

## A.1 The Weak Specialist: Micro-Agent Architecture

Network Specification. To instantiate the Weak Specialist $( \pi _ { \phi } )$ , we adopt the model architecture from the MARA framework (Zhang et al., 2025). Unlike traditional alignment methods that fine-tune the entire large language model (LLM), MARA introduces a lightweight, decoupled Micro-Alignment Model (MAM) designed to perform token-level binary classification.

• Input Layer: Accepts the last hidden state vector $\mathbf { h } _ { t } \in \mathbb { R } ^ { d _ { m o d e l } }$ of the current token generated by the frozen Base Model (π<sub>θ</sub>).

• Hidden Layers: Consists of three fully connected layers with hidden dimensions of [4096, 1024, 256].

• Output Layer: Projects the features to a 2- dimensional logit vector corresponding to the discrete actions A = {ACCEPT, REJECT}.

With a total parameter count of approximately 4M, the micro-agent incurs negligible memory overhead compared to the multi-billion parameter Base Model.

Inference Logic. During the dense intervention phase, the agent computes the probability of acceptance for a candidate token $y _ { t }$ as:

$$
P ( \operatorname { A C C E P T } | y _ { t } ) = \operatorname { S o f t m a x } ( \operatorname { M L P } ( \mathbf { h } _ { t } ) ) _ { [ 0 ] }\tag{13}
$$

In our TUSA framework, this agent serves as the execution actuator, but it is invoked selectively based on the Cognitive Arbiter’s decision.

We integrate the pre-trained MARA micro-agent into the TUSA framework as the conditional execution unit. The complete inference process, which couples the Cognitive Arbiter (for trust assessment) with the Micro-Agent (for risk assessment), is formalized in Algorithm 2.

The algorithm highlights how TUSA transforms the standard “generate-then-check” loop into a “trust-or-check” conditional flow.

## A.2 Construction of Semantic Saliency Atlas

To strictly quantify the Semantic Saliency (S<sub>t</sub>) of each token, we constructed a static importance atlas based on global corpus statistics. This process ensures that the saliency metric reflects the general information content of a token rather than its local context probability.

Corpus Selection and Sampling. We utilized the Common Corpus dataset<sup>1</sup>, a large-scale, opendomain text collection. To construct a representative yet computationally manageable statistic set, we employed a stratified sampling strategy. Specifically, we sampled data from 10 distinct subdomains. From each sub-domain, we randomly selected 10 Parquet files, resulting in a diverse aggregate corpus of 100 files. This ensures the IDF statistics are robust to domain shifts.

IDF Calculation Protocol. The calculation pipeline proceeds as follows:

1. Tokenization: We utilized the tokenizer of the target Base Model (e.g., Llama-3.1-8B-Instruct) to process the raw text. This ensures the calculated statistics map one-to-one with the model’s vocabulary.

2. Document Frequency (DF) Statistics: For each document d in the sampled corpus, we extracted the set of unique token IDs, denoted as $\gamma _ { d } .$ . We then computed the document frequency df(t) for every token t in the vocabulary V, defined as the total count of documents containing that token:

$$
\mathrm { d f } ( t ) = \sum _ { d \in \mathcal { D } } \mathbb { I } ( t \in \mathcal { V } _ { d } )\tag{14}
$$

where $\mathcal { D }$ represents the total set of processed documents and I(·) is the indicator function.

3. Inverse Document Frequency (IDF): The final raw saliency score is computed using the smooth IDF formula:

$$
\mathrm { I D F } ( t ) = \log \left( \frac { N } { 1 + \mathrm { d f } ( t ) } \right)\tag{15}
$$

where $N = | \mathcal { D } |$ is the total number of documents.

Algorithm 2 TUSA Inference with Micro-Agent   
Require: Base Model $\pi \theta$ (Strong Generalist), Micro-Agent $\pi _ { \phi }$ (Weak Specialist), Prompt sequence x, Sensitivity α, Window   
size K, Sampling size k.   
Ensure: Aligned sequence y.   
1: Initialize generation history $\mathbf { y } _ { < 1 }  \mathbf { x } .$   
2: Initialize necessity history buffer $H \gets \{ \tau _ { 0 } \}$ . {Initialize with prior to avoid division by zero}   
3: for $t = 1 , 2 , \dots , \mathbf { \dot { } } T$ do   
4: // Stage 1: Base Generation   
5: Get logits and hidden state: $P _ { \theta } ( \cdot ) , \mathbf { h } _ { t } \gets \pi _ { \theta } ( \mathbf { y } _ { < t } )$   
6: Get Top-k candidates: ${ \mathcal { V } } _ { k } \gets \mathrm { T o p } { - } k ( P _ { \theta } )$   
7: $y _ { t } \gets \bar { \mathsf { P o p } } ( \mathcal { V } _ { k } )$   
8: // Stage 2: Trust-Based Arbitration   
9: Compute Cognitive Confidence C<sub>t</sub> from Micro-Agent:   
10: $\mathbf { l } _ { \phi _ { . } } ^ { \setminus }  \pi _ { \phi } ( \bar { \mathbf { h } } _ { t } )$ {Forward pass only, no decision yet}   
11: $\bar { \pi _ { \phi } ^ { \prime } }  \mathsf { S }$ oftmax ${ \left( 1 _ { \phi } / \lambda \right) }$   
12: $C _ { t } \gets \log | \mathcal { A } | - \mathcal { H } ( \pi _ { \phi } ^ { \prime } )$ {Confidence via KL divergence}   
13: Calculate Dynamic Threshold:   
14: $\begin{array} { r } { \tau _ { t }  \alpha \cdot \frac { 1 } { | H | } \sum _ { v \in H } v } \end{array}$   
15: Candidate Evaluation: {Entry point for resampled y<sub>t</sub>}   
16: Compute Semantic Saliency: $\bar { S } _ { t } \gets \mathrm { I }$ ookupIDF(y<sub>t</sub>)   
17: Calculate Joint Necessity: $\mathbf { \bar { \phi } } _ { R _ { \mathrm { j o i n t } } }  C _ { t } \cdot S _ { t }$   
18: // Stage 3: Adaptive Execution   
19: if $R _ { \mathrm { j o i n t } } < \tau _ { t }$ then   
20: Path A: Trust (Bypass)   
21: Keep y<sub>t</sub> directly.   
22: else   
23: Path B: Intervention   
24: Decode action: $a _ { t } \gets$ argmax(l<sub>ϕ</sub>)   
25: if $a _ { t } = = { \mathrm { R E J E C T } }$ then   
26: y<sub>t</sub> ← Pop(V<sub>k</sub>) {Take the next highest probability token}   
27: Goto Candidate Evaluation   
28: end if   
29: end if   
30: Update history $H  H \cup \{ R _ { \mathrm { j o i n t } } \}$ (maintain max size K)   
31: Append y<sub>t</sub> to y<sub><t+1</sub>   
32: end for   
33: return $\mathbf { y } _ { < T + 1 }$

Deployment. The computed IDF values were compiled into a static lookup table (CSV file). During inference, this table allows for O(1) retrieval of the saliency score for any generated token ID.

## B Experiment Details

## B.1 Models and Datasets

To comprehensively validate the effectiveness of TUSA in resolving the competence mismatch paradox, we conduct experiments across a diverse range of models and a dual-domain benchmark suite. Throughout these experiments, we strictly adhere to the usage terms and licenses of all open-source models and datasets employed in this work.

Base Models (The Strong Generalists). We evaluate our framework on five state-of-the-art open-source Large Language Models (LLMs) to ensure universality. These include:

• Llama-3 Family (AI@Meta, 2024): We utilize Meta-Llama-3.1-8B-Instruct and the compact Llama-3.2-3B-Instruct.

• Mistral Family (Jiang et al., 2023): We cover the evolution of the Mistral-7B series by including Mistral-7B-Instruct-v0.1, v0.2, and v0.3.

All base models are kept frozen during inference. The Weak Specialist is the pre-trained 4Mparameter MARA micro-agent (Zhang et al., 2025), which guides these strong generalists.

Benchmarks (Dual-Domain). We design a rigorous evaluation protocol covering both safety alignment and general capabilities to measure the tradeoff between alignment efficacy and inference overhead.

Safety Benchmarks: To assess the model’s adherence to safety constraints, we employ three standard datasets:

• PKU-SafeRLHF (Ji et al., 2025): A large-scale dataset containing expert-annotated preference pairs labeled for helpfulness and harmlessness. We randomly sample 200 prompts from the test set for evaluation.

• BeaverTails (Ji et al., 2023): Focuses on identifying and mitigating toxicity. We evaluate on a balanced subset of 700 instances covering diverse harm categories (e.g., hate speech, violence).

• HarmfulQA (Bhardwaj and Poria, 2023): A collection of red-teaming prompts derived from LLM generation, designed to elicit harmful content across 10 topics. We sample 490 high-risk prompts to test robustness.

General Capability Benchmarks: To quantify the impact of intervention on general intelligence, we evaluate on:

• AlpacaEval (Li et al., 2023c): Assessing general instruction-following abilities. We randomly sample 200 prompts from the test set for evaluation.

• JustEval (Lin et al., 2024a): A comprehensive evaluation benchmark, which is strategically partitioned into 800 problem-solving instances and 200 safety-focused tests to evaluate both utility and robustness. We randomly sample 200 prompts from the test set for evaluation.

## B.2 Baselines

We compare TUSA against three distinct inference paradigms:

• Upstream LLM: The original instruction-tuned base model without any inference-time intervention, serving as the lower bound for safety and upper bound for utility.

• MARA (Dense Intervention) (Zhang et al., 2025): The representative dense alignment framework. MARA employs a rigorous accept-reject mechanism at every decoding step: the microagent continuously scrutinizes the base model’s proposal, explicitly deciding to either “accept” the current token or “reject” and override it with a safety-aligned alternative. This serves as the dense baseline representing supervision without dynamic filtering.

• ConfPO (Inference-Adapted) (Yoon et al., 2025): A state-of-the-art training-time alignment method, ConfPO utilizes uncertainty estimation to selectively filter noisy signals during preference optimization.

## B.3 Evaluation Protocol

Unified Evaluation Framework. To ensure a consistent evaluation standard and guarantee reproducibility, we strictly employ the open-source Beaver-7B suite (Dai et al., 2024) as our automated judge, rather than relying on proprietary APIs which are subject to version shifts and opacity. Specifically, we utilize beaver-7b-reward to evaluate Helpfulness (assessing utility and instruction following) and beaver-7b-cost to evaluate Harmlessness (measuring safety violations). Comprehensive analysis of the judge’s reliability is provided in Appendix E.

Evaluation Metrics. To rigorously assess the alignment quality, we employ the beaver-7b-v1.0-reward model to quantify Helpfulness (utility) and the beaver-7b-v1.0-cost model to measure Harmlessness (safety). Unlike standard win-rate metrics that evaluate dimensions in isolation, we adopt a strict dominance criterion to penalize the safety-via-refusal shortcut—where models achieve high safety scores simply by rejecting all instructions. Accordingly, the comparison logic is defined as follows:

• Win $( N _ { \mathrm { w i n } } )$ : Registered if and only if the approach achieves superior performance in both helpfulness and harmlessness compared to the baseline.

• Loss (N ): Recorded if the approach underperforms in both dimensions.

• Tie (N<sub>tie</sub>): Assigned to all trade-off scenarios (e.g., safer but less helpful, or vice versa) where superiority is inconclusive.

This stringent metric ensures that a high win-rate reflects a genuine Pareto improvement rather than a compromise between safety and utility.

## B.4 Parameter Setting

Training Parameters of MARA. In this section, we provide the detailed hyperparameter configurations used to train the MARA micro-agent via the Soft Actor-Critic (SAC) (Haarnoja et al., 2018) algorithm. Table 5 and Table 6 list the specific settings employed for the Llama-family (Llama-3.1-8B, Llama-3.2-3B) and Mistral-family (Mistralv0.1, v0.2, v0.3) models, respectively. The parameters are categorized into four groups: (1) Optimization and Training, detailing learning rates and batch sizes; (2) Network Architecture and Buffer, describing the actor-critic structure and replay buffer capacity; (3) Algorithm Coefficients, specifying the SAC-specific entropy targets and reward multipliers; and (4) Generation Strategy, listing the decoding parameters used during inference.

Table 5: Hyperparameter Settings of Llama Models.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Optimization &amp; Training</td><td></td></tr><tr><td>Train Epoch</td><td>1</td></tr><tr><td>Max Episode</td><td>2100</td></tr><tr><td>Batch Size</td><td>1024</td></tr><tr><td>Actor Learning Rate</td><td> $3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Critic Learning Rate</td><td> $3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Alpha Learning Rate Update Time</td><td> $3 \times 1 0 ^ { - 4 }$  10</td></tr><tr><td>Network Architecture &amp; Buffer State Dimension</td><td></td></tr><tr><td>Action Dimension</td><td>4096 2</td></tr><tr><td>Hidden Dimension</td><td>1024</td></tr><tr><td>Buffer Capacity</td><td> $1 \times { 1 0 } ^ { 6 }$ </td></tr><tr><td>Replace Tau</td><td>0.005</td></tr><tr><td></td><td></td></tr><tr><td>Algorithm Coefficients (Entropy/Reward/KL)</td><td></td></tr><tr><td>Alpha</td><td></td></tr><tr><td></td><td>0.01</td></tr><tr><td>Target Entropy Factor</td><td>0.6</td></tr><tr><td>Target Entropy Type</td><td></td></tr><tr><td>Reward Multiplier</td><td>log</td></tr><tr><td></td><td> $1 . 0 , - 1 . 0$ </td></tr><tr><td>Reward Type</td><td>kl_div</td></tr><tr><td>Reward Baseline</td><td>0.0</td></tr><tr><td>KL Control Coefficient  $\mathbf { \Pi } ( \mathbf { k l \_ c t l } )$ </td><td>0.1</td></tr><tr><td>KL Penalty Type</td><td>kl</td></tr><tr><td>Generation Strategy</td><td></td></tr><tr><td>Max New Token</td><td>512</td></tr><tr><td></td><td></td></tr><tr><td>Temperature</td><td>0.8</td></tr><tr><td>Top-K</td><td>40</td></tr><tr><td>Top-P</td><td>0.95</td></tr></table>

Inference Parameters. This section details the hyperparameter settings employed during the inference and evaluation phases. To prevent data leakage, inference hyperparameters (α, λ) were tuned on separate validation splits and frozen during testing. TUSA and the dense MARA baseline share the identical, frozen pre-trained checkpoint. Tables 9 and 10 outline the general decoding configurations and sparse intervention-specific parameters, respectively. Unless otherwise noted, the sliding uncertainty calculation uses a fixed context window size of $K = 1 0$

## B.5 Computing Resources

All experiments were conducted on a computing server equipped with dual Intel Xeon Platinum 8352V CPUs (2.10GHz), providing a total of 72 physical cores and 144 logical threads. The system features approximately 1TB of system RAM and is accelerated by 8 NVIDIA L20 GPUs, each equipped with 48GB of VRAM. The software environment is configured with NVIDIA driver version 570.148.08 and the CUDA 12.8 toolkit.

Table 6: HyperparameterSettings of Mistral Models.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Optimization &amp; Training Train Epoch Max Episode Batch Size</td><td>1 2100 1024</td></tr><tr><td>Actor Learning Rate Critic Learning Rate Alpha Learning Rate Update Time</td><td> $3 \times 1 0 ^ { - 4 }$   $3 \times 1 0 ^ { - 4 }$   $3 \times 1 0 ^ { - 4 }$  10</td></tr><tr><td>Network Architecture &amp; Buffer State Dimension Action Dimension Hidden Dimension Buffer Capacity Replace Tau</td><td>4096 2  $1 0 2 4$   $1 \times 1 0 ^ { 6 }$  0.005</td></tr><tr><td>Algorithm Coefficients (Entropy/Reward/KL) Alpha Target Entropy Factor Target Entropy Type</td><td>0.01 0.6 log</td></tr><tr><td>Reward Multiplier Reward Type Max New Token</td><td>2.0, -1.0 kl_div</td></tr><tr><td>Reward Baseline KL Control Coefficient (kl_ctl) KL Penalty Type Generation Strategy</td><td>0.0 0.1 kl</td></tr></table>

Table 7: Hyperparameter configurations for the ConfPO training on Llama Models.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td colspan="2">Optimization</td></tr><tr><td>Optimizer Precision Learning Rate LR Scheduler Warmup Ratio Train Batch Size (per device)</td><td>DeepSpeed ZeRO-2 bfloat16  $2 . 0 \times 1 0 ^ { - 8 }$  Cosine 0.1</td></tr><tr><td>Gradient Accumulation Steps Effective Batch Size Num GPUs Epochs</td><td>2 8 64 4</td></tr><tr><td>Gradient Checkpointing</td><td>1 True</td></tr><tr><td colspan="2">Model &amp; Data Base Model</td></tr><tr><td>Max Sequence Length</td><td>Llama-3.2-3B-Instruct 1024</td></tr><tr><td>Max Prompt Length Attention Implementation</td><td>900 Flash Attention 2</td></tr><tr><td colspan="2">Algorithm (ConfPO/SimPO) Loss Type SimPO</td></tr></table>

Table 8: Hyperparameter configurations for the ConfPO training on Mistral Models.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td colspan="2">Optimization</td></tr><tr><td>Precision Learning Rate LR Scheduler Warmup Ratio Train Batch Size (per device)</td><td>bfloat16  $2 . 0 \times 1 0 ^ { - 8 }$   $\operatorname { C o s i n e }$  0.1 2</td></tr><tr><td colspan="2">Effective Batch Size</td></tr><tr><td>Num GPUs Epochs</td><td>4 1</td></tr><tr><td>Gradient Checkpointing</td><td>True</td></tr><tr><td colspan="2">Model &amp; Data</td></tr><tr><td>Base Model Max Sequence Length</td><td>Mistral-7B-Instruct-v0.1 1024</td></tr><tr><td>Max Prompt Length Attention Implementation</td><td>900 Flash Attention 2</td></tr><tr><td colspan="2">Algorithm (ConfPO/SimPO)</td></tr><tr><td></td><td></td></tr><tr><td>Loss Type</td><td>SimPO</td></tr><tr><td>β (Beta)</td><td>2.0</td></tr><tr><td>γ (Gamma)</td><td>0.5</td></tr></table>

Table 9: General Inference and Evaluation Parameters.
<table><tr><td>Parameter Value</td></tr><tr><td>Evaluation Configuration Evaluation Action</td></tr><tr><td>generate Evaluation Mode</td></tr><tr><td>proxy Serial Action True</td></tr><tr><td>Eval From Start True</td></tr><tr><td>Model &amp; Strategy Settings</td></tr><tr><td>State Dimension [4096, 3072]</td></tr><tr><td>Policy Model Type [1lama, mistral]</td></tr><tr><td>State Transition v0</td></tr><tr><td>Default Action Index</td></tr><tr><td>0</td></tr><tr><td>Proxy Strategy top1</td></tr><tr><td>Generation Hyperparameters</td></tr><tr><td>Max New Token 512</td></tr><tr><td>Temperature 0.8</td></tr><tr><td>Top-K 40</td></tr><tr><td>Top-P 0.95</td></tr></table>

## C Additional Experiment Results

## C.1 Additional Ablation Studies

Ablation of Window Size. To assess the sensitivity of the Cognitive Arbiter to the length of local history, we conducted an additional ablation study on the sliding window size K. This parameter determines the number of preceding tokens considered when calculating the local uncertainty statistics required for the dynamic threshold τ . Figure 10 presents the performance variations on the SafeRLHF benchmark across different window sizes for both Llama-3.1-8B and Mistral-v0.1-7B. We observe that for the majority of models, both guidance proportion and alignment performance reach a plateau when $K \geq 1 0$ . While Llama-3.1- 8B exhibits a slight performance peak at $K = 1 5$ the overall sensitivity to this parameter remains low. Therefore, to maintain a unified and simplified experimental setting, we fix the window size at K = 10 for all models reported in this work.

## C.2 Analysis of Optimal Guidance Proportion

Table 11 presents the empirical guidance proportions required to achieve peak performance across diverse model families and tasks. The results reveal that the optimal intervention frequency is highly dynamic, ranging from as low as 27.43% (Mistralv0.3 on AlpacaEval) to 66.98% (Mistral-v0.1 on SafeRLHF). This variance confirms that the Cognitive Arbiter successfully tailors its aggressiveness based on the specific uncertainty profile of each model-task pair, rather than enforcing a fixed computational budget. Furthermore, we observe an inverse correlation between base model capability and intervention frequency. Notably, the stronger Llama-3.1-8B model requires less intervention on safety benchmarks (∼ 30–50%) compared to the Mistral-v0.1 baseline (∼ 60–70%). This suggests that TUSA effectively identifies when the base model aligns well with safety constraints intrinsically, thereby stepping back to preserve inference speed and reasoning flow.

<table><tr><td>Model</td><td>SafeRLHF</td><td>BeaverTails</td><td>HarmfulQA</td><td>AlpacaEval JustEval</td><td></td></tr><tr><td>Llama-3.1-8B</td><td>0.6 / 1.1</td><td>1.2 / 1.15</td><td>1.2 / 1.1</td><td>0.5 / 1.1</td><td>0.7 / 1.1</td></tr><tr><td>Llama-3.2-3B</td><td>0.42 / 1.1</td><td>0.38 / 1.1</td><td>0.38 / 1.02</td><td>0.7 / 1.1</td><td>0.8 / 1.1</td></tr><tr><td>Mistral-v0.1-7B</td><td>0.5 / 1.8</td><td>0.5 / 1.15</td><td>0.4 / 1.05</td><td>0.7 / 1.1</td><td>0.5 / 1.1</td></tr><tr><td>Mistral-v0.2-7B</td><td>0.5 / 1.1</td><td>0.4 / 1.1</td><td>0.52 / 1.05</td><td>0.7 / 1.1</td><td>1.1 / 1.1</td></tr><tr><td>Mistral-v0.3-7B</td><td>0.53 / 1.12</td><td>0.48 / 1.16</td><td>0.4 / 1.06</td><td>1.5 / 1.1</td><td>0.9 / 1.1</td></tr></table>

Table 10: Specific hyperparameter settings for sparse intervention across different models and benchmarks. The values are presented in the format of Safety Coefficient / Temperature Coefficient (α / λ).
<table><tr><td rowspan="2">Model</td><td colspan="3">Safety Benchmarks</td><td colspan="2">General Capabilities</td></tr><tr><td>SafeRLHF</td><td>BeaverTails HarmfulQA</td><td></td><td>AlpacaEval JustEval</td><td></td></tr><tr><td>Mistral-v0.1-7B</td><td>66.98</td><td>58.98</td><td>66.58</td><td>47.35</td><td>56.68</td></tr><tr><td>Mistral-v0.2-7B</td><td>59.10</td><td>65.06</td><td>53.74</td><td>46.87</td><td>37.84</td></tr><tr><td>Mistral-v0.3-7B</td><td>59.91</td><td>62.91</td><td>64.65</td><td>27.43</td><td>40.28</td></tr><tr><td>Llama-3.1-8B</td><td>52.13</td><td>32.19</td><td>31.24</td><td>56.47</td><td>47.44</td></tr><tr><td>Llama-3.2-3B</td><td>62.81</td><td>65.27</td><td>57.72</td><td>47.11</td><td>43.04</td></tr></table>

Table 11: The optimal Guidance Proportion (%) for each model across different datasets. Higher values indicate a higher frequency of intervention by the specialist model.

## C.3 Statistical Significance Analysis

To verify the empirical reliability and robustness of the alignment gains achieved by TUSA, we conduct a variance analysis across multiple evaluation benchmarks and baseline model architectures. Reporting variance estimates serves as an empirical indicator to assess whether our selective intervention strategy introduces unstable performance fluctuations under varying test distributions and diverse capacity baselines.

The empirical variance analysis demonstrates that TUSA’s performance improvements are highly consistent and resilient to random fluctuations. As detailed in Table 12, the vast majority of our improvements across Mistral and Llama variants on SafeRLHF, BeaverTails, and HarmfulQA are highly statistically significant (mostly $p < 0 . 0 0 1$ or $p \ < \ 0 . 0 0 0 1 )$ . This confirms that the performance gains introduced by TUSA are statistically reliable and not due to random variance.

## C.4 Detailed Win-Rate Statistics

In this section, we provide the granular breakdown of the comparative evaluation between TUSA and the baseline methods in Tables 1– 2. Tables 18– 28 detail the specific counts for Win, Lose, and Tie outcomes across the three evaluation dimensions: Helpfulness, Harmlessness, and the composite Preference Rate. The statistics are calculated based on the valid samples successfully generated and evaluated for each benchmark. The specific counts of valid data points included in the final analysis are as follows:

• Safety Benchmarks: SafeRLHF (N = 199), BeaverTails (N = 700), and HarmfulQA (N = 488).

• General Utility Benchmarks: AlpacaEval (N = 200) and JustEval (N = 167).

## D Analysis and Discussion

## D.1 Analysis between Trust and Entropy

To formally justify using the KL-divergence as a trust metric, we derive its mathematical relationship with the Shannon entropy of the weak specialist. By defining the proxy distribution $P _ { \mathrm { p r o x y } }$ as a uniform distribution over the action space A, the trust score $C _ { t }$ can be expanded as follows:

<table><tr><td rowspan="2">Base Model</td><td rowspan="2">TUSA (ours) vs. baseline</td><td colspan="3">Preference Rate (%) ± Std. Dev.</td></tr><tr><td>SafeRLHF</td><td>BeaverTails</td><td>HarmfulQA</td></tr><tr><td rowspan="3">Mistral-7B-v0.1</td><td>Base Model</td><td> $1 3 . 6 \pm 0 . 2 ^ { * * * }$ </td><td> $1 6 . 6 \pm 0 . 8 ^ { * * * }$ </td><td> $2 2 . 1 \pm 0 . 6 ^ { * * * }$ </td></tr><tr><td>ConfPO</td><td> $1 3 . 6 \pm 0 . 1 ^ { * * * }$ </td><td> $1 7 . 0 \pm 0 . 6 ^ { * * * }$ </td><td> $2 0 . 5 \pm 0 . 3 ^ { * * * }$ </td></tr><tr><td>MARA</td><td> $1 5 . 6 \pm 0 . 1 ^ { * * * }$ </td><td> $8 . 4 \pm 0 . 4 ^ { * * * }$ </td><td> $5 . 1 \pm 0 . 3 ^ { * * * }$ </td></tr><tr><td rowspan="3">Mistral-7B-v0.2</td><td>Base Model</td><td> $4 . 5 \pm 0 . 2 ^ { * * * }$ </td><td> $1 0 . 3 \pm 0 . 4 ^ { * * * }$ </td><td> $8 . 4 \pm 0 . 4 ^ { * * * }$ </td></tr><tr><td>ConfPO</td><td> $- 0 . 5 \pm 0 . 0$ </td><td> $1 0 . 4 \pm 0 . 2 ^ { * * * }$ </td><td> $9 . 4 \pm 0 . 2 ^ { * * * }$ </td></tr><tr><td>MARA</td><td> $9 . 1 \pm 0 . 2 ^ { * * * }$ </td><td> $4 . 7 \pm 0 . 4 ^ { * * * }$ </td><td> $2 . 9 \pm 0 . 4 ^ { * * }$ </td></tr><tr><td rowspan="3">Mistral-7B-v0.3</td><td>Base Model</td><td> $7 . 0 \pm 0 . 3 ^ { * * * }$ </td><td> $2 . 9 \pm 0 . 4 ^ { * * }$ </td><td> $1 3 . 9 \pm 0 . 5 ^ { * * * }$ </td></tr><tr><td>ConfPO</td><td> $2 . 0 \pm 0 . 2 ^ { * * }$ </td><td> $3 . 3 \pm 0 . 3 ^ { \ast \ast }$ </td><td> $9 . 8 \pm 0 . 3 ^ { * * * }$ </td></tr><tr><td>MARA</td><td> $2 . 5 \pm 0 . 3 ^ { * * }$ </td><td> $- 1 . 4 \pm 0 . 4$ </td><td> $0 . 8 \pm 0 . 4 ^ { * }$ </td></tr><tr><td rowspan="3">Llama 3.1-8B</td><td>Base Model</td><td> $9 . 1 \pm 0 . 3 ^ { * * * }$ </td><td> $1 0 . 7 \pm 0 . 4 ^ { * * * }$ </td><td> $4 . 5 \pm 0 . 4 ^ { * * }$ </td></tr><tr><td>ConfPO</td><td> $9 . 1 \pm 0 . 3 ^ { * * * }$ </td><td> $7 . 4 \pm 0 . 3 ^ { * * * }$ </td><td> $1 . 2 \pm 0 . 2 ^ { * * }$ </td></tr><tr><td>MARA</td><td> $3 . 5 \pm 0 . 4 ^ { * * }$ </td><td> $5 . 4 \pm 0 . 2 ^ { * * * }$ </td><td> $7 . 4 \pm 0 . 4 ^ { * * * }$ </td></tr><tr><td rowspan="3">Llama 3.2-3B</td><td>Base Model</td><td> $1 3 . 6 \pm 0 . 4 ^ { * * * }$ </td><td> $1 6 . 1 \pm 0 . 4 ^ { * * * }$ </td><td> $1 8 . 9 \pm 0 . 4 ^ { * * * }$ </td></tr><tr><td>ConfPO</td><td> $1 0 . 6 \pm 0 . 4 ^ { * * * }$ </td><td> $1 6 . 3 \pm 0 . 4 ^ { * * * }$ </td><td> $1 7 . 6 \pm 0 . 4 ^ { * * * }$ </td></tr><tr><td>MARA</td><td> $4 . 0 \pm 0 . 3 ^ { \ast \ast }$ </td><td> $3 . 7 \pm 0 . 4 ^ { * * }$ </td><td> $1 . 0 \pm 0 . 3 ^ { * }$ </td></tr></table>

Table 12: Statistical significance analysis of the relative Preference Rate (%) improvements. Results are computed over 3 independent trials. p-values are derived from a one-tailed t-test and denoted as superscripts: ${ } ^ { * } p < 0 . 0 5 .$ $^ { * * } p < 0 . 0 1 , ^ { * * * } p < 0 . 0 0 1 , ^ { * * * * } p < 0 . 0 0 0 1$ . Standard deviations are reported alongside means (Mean ± SD).

$$
\begin{array} { r l } { C _ { t } ^ { \prime } = D _ { K L } [ \pi _ { \phi } ^ { \prime } ( \cdot | t ) ] P _ { \mathrm { R e a y } } \rangle } \\ { = } & { \sum _ { \phi \in A } ^ { \pi _ { \phi } ^ { \prime } ( a | t ) } \log \frac { \pi _ { \phi } ^ { \prime } ( a | t ) } { P _ { \mathrm { F o r y } } ^ { \prime } ( a | t ) } } \\ { = } & { \sum _ { \phi \in A } ^ { \pi _ { \phi } ^ { \prime } ( a | t ) } \log \pi _ { \phi } ^ { \prime } ( a | t ) } \\ { \underbrace { \phantom { \sum _ { \phi \in A } ^ { \phi } \pi _ { \phi } ^ { \prime } ( a | t ) } } _ { - \mathcal { H } ( x _ { \phi } ^ { \prime } ) } } \\ { - \sum _ { \phi \in \mathcal { Q } } \pi _ { \phi } ^ { \prime } ( a | t ) \log \underbrace { P _ { \mathrm { P o r y } } ( a ) } _ { \mathcal { H } | \mathcal { A } | } } \\ { - \mathcal { H } ( \pi _ { \phi } ^ { \prime } ) - \log ( | \mathcal { A } | ^ { - 1 } ) } \\ { = } & { \log | \mathcal { A } | - \mathcal { H } ( \pi _ { \phi } ^ { \prime } ) } \end{array}
$$

## D.2 Analysis and Further Discussion of $S _ { t }$

(16)

This derivation reveals a direct linear relationship between the trust score $C _ { t }$ and the negative entropy of the specialist’s policy. Mathematically, it implies that maximizing the divergence from a uniform prior is equivalent to minimizing the predictive entropy. Consequently, a higher $C _ { t }$ strictly corresponds to higher model confidence (lower uncertainty), validating its use as a robust, densitybased indicator for identifying safe intervention points.

This section presents a discussion and empirical analysis regarding the lexical Prior, boundary cases, task generalization, and alternative estimator choices for the Inverse Document Frequency (IDF) formulation used within our Semantic Saliency framework.

Lexical Prior Justification. The Cognitive Arbiter’s static IDF prior serves as a training-free, global high-pass filter to eliminate low-surprisal syntactic redundancy. Empirical analysis confirms that bypassed tokens comprise purely grammatical scaffolding, such as articles (the), prepositions (of), conjunctions (and), pronouns (it), and auxiliaries (is). Intervening on these functional tokens wastes computation, as alignment traits are rarely dictated by syntactic glue. Removing this noise concentrates the steering budget exclusively on contentrich nouns and verbs that directly drive alignment.

Joint Mechanism for High-Frequency Tokens. To ensure high-frequency but semantically pivotal tokens (e.g., not, never) are not erroneously bypassed by low static scores, our Joint Necessity Score introduces a dynamic safeguard. The supervisor’s predictive variance actively compensates for static lexical limitations through multiplicative coupling. When critical negation or exception markers appear in contexts where the specialist exhibits high confidence, a sufficiently large $C _ { t }$ can compensate for their relatively low—but nonzero—static saliency scores, allowing the joint necessity score to cross the adaptive threshold.

![](images/643b581b7d701139e4b18f43ead30650185519a3b654633a9824d87209bf1e71.jpg)  
(a) Mistral-v0.1-7B

![](images/8c899c7d74e7f58fed93e3c1c80d0e0ef60163b48ea564ea38fbfa5cc0c27749.jpg)  
(b) Llama-3.1-8B  
Figure 10: Ablation study on the Sliding Window Size K. The plots illustrate how varying the historical context length impacts the performance of (a) Mistral-v0.1-7B and (b) Llama-3.1-8B on the SafeRLHF benchmark.

Task Generalization via Decoupling. TUSA explicitly decouples intervention timing (when to steer) from the alignment objective (what to enforce), allowing the static filter to generalize naturally across domains. Downstream criteria— whether safety, helpfulness, or humor—are handled exclusively by the task-specific Micro-Agent. By isolating structural decision points via statistical rarity rather than high-level semantics, the Cognitive Arbiter requires no task-specific re-training, achieving universal plug-and-play applicability.

Semantic Saliency Estimators. To address estimator selection, Table 13 compares our Static IDF against alternative paradigms across contextawareness, inference overhead, and architectural requirements. While Intrinsic methods (e.g., selfattention or logit entropy) restrict deployment to white-box models, Dynamic estimators (e.g., proxy language models) require synchronous auxiliary forward passes, completely undermining sparse intervention latency advantages. Consequently, our choice of Static IDF serves as an efficient, modelagnostic high-pass filter. Crucially, its lack of contextual adaptation is actively compensated by the supervisor’s dynamic confidence $C _ { t }$ within the Joint Necessity Score $( R _ { \mathrm { j o i n t } } = C _ { t } \cdot S _ { t } )$ . This multiplicative coupling allows high-risk spikes in $C _ { t }$ to override low static saliency $S _ { t }$

## D.3 Detailed Analysis of Computational Overhead and Sparsity

In the TUSA framework, a lightweight Micro-Agent confidence probe is executed at each eligible decoding step to evaluate the cognitive confidence $( C _ { t } )$ . The sparsity of our method is achieved by skipping the computationally heavy, candidatelevel accept/reject and resampling pipeline, rather than eliminating this initial lightweight confidence check.

To quantify the computational overhead, we profile TUSA against the Dense MARA baseline using Llama-3.1-8B-Instruct across 9,484 eligible decoding steps on a single NVIDIA H100 GPU. The comparison of intervention metrics is detailed in Table 14.

As shown in Table 14, TUSA reduces candidatelevel Micro-Agent evaluations by 70.18%. This efficiency stems from successfully bypassing the heavy candidate-level arbitration and resampling pipeline at 62.48% of the decoding positions.

To further analyze the end-to-end latency performance, we present a component-wise latency breakdown in Table 15.

The profiling reveals that the dominant basemodel generation and ranking pipelines consume 93.81% of the total execution time. Because this baseline inference cost inherently remains unchanged, the substantial reductions in micro-agent calls do not translate into a proportional end-to-end speedup. Ultimately, TUSA optimizes the safetyutility trade-off within a dense-comparable latency budget by bypassing active candidate-level correction at approximately half of the decoding positions.

<table><tr><td>Estimator Paradigm</td><td>Context-Awareness</td><td>Latency</td><td>Access</td></tr><tr><td>Static Prior (e.g., IDF)</td><td>Low</td><td>O(1) Lookup</td><td>Black-box</td></tr><tr><td>Lexical (e.g., POS Tagger)</td><td>Medium-Low</td><td>Extremely Low</td><td>Black-box</td></tr><tr><td>Intrinsic (e.g., Logit Èntropy)</td><td>High</td><td>Medium (VRAM bounded)</td><td>White-box Only</td></tr><tr><td>Dynamic (e.g., Dynamic LLM)</td><td>High</td><td>High (Auxiliary Forward)</td><td>Black-box</td></tr></table>

Table 13: Taxonomy and comparison of semantic saliency estimation paradigms during inference.
<table><tr><td>Metric</td><td>Dense MARA</td><td>TUSA (ours)</td><td>Reduction</td></tr><tr><td>Steps entering full candidate arbitration</td><td>100%</td><td>37.52%</td><td>62.48%</td></tr><tr><td>Micro-Agent evaluations per eligible step</td><td>5.44</td><td>1.62</td><td>70.18%</td></tr></table>

Table 14: Comparison of intervention frequency and evaluation overhead between Dense MARA and TUSA.

## E LLM judge quality validation

To ensure a unified and scalable evaluation standard across all experiments, we employ the Beaver-7B suite (Dai et al., 2024) as our primary automated judge. The rationale for this selection is multifaceted, encompassing considerations of evaluation reliability, alignment granularity, and computational efficiency.

1. Multi-Dimensional Value Alignment. Built on the Alpaca architecture via the Safe RLHF paradigm (Dai et al., 2024), Beaver-7B leverages human-annotated preference data across over ten safety constraints. This rigorous training enables it to reliably decouple and independently assess response helpfulness and harmlessness with humanlike nuance.

2. Transparency and Reproducibility. Unlike opaque, proprietary closed-source models prone to silent version updates, Beaver-7B provides a fully open-source evaluation ecosystem. Its publicly accessible weights, datasets, and hyperparameters ensure a deterministic and verifiable evaluation pipeline for the research community.

3. Resource Efficiency. Beaver-7B strikes an optimal balance between judgment capability and computational cost. Its 7B scale ensures minimal VRAM and inference overhead, allowing for scalable, high-throughput sampling and multi-round evaluations in resource-constrained environments without API rate bottlenecks.

To validate this 7B-scale judge, we benchmarked Beaver-7B against Llama-Guard-4-12B (Inan et al., 2023), GPT-OSS-120B (OpenAI, 2025), and GPT-4o (OpenAI, 2024) using 100 human-annotated responses, each evaluated across both helpfulness and harmlessness dimensions. The human evaluation was conducted by volunteer researchers within our institution who were fully informed of the academic objectives. As shown in Table 16, Beaver-7B exhibits high human alignment. On Harmlessness, it achieves an 87.0% agreement rate, outperforming Llama-Guard (82.0%) and competing closely with GPT-4o (91.0%). On Helpfulness, Beaver-7B achieves the highest agreement (81.0%), surpassing both GPT-OSS-120B (75.0%) and GPT-4o (79.0%), confirming its credibility as a robust evaluator.

Additionally, we conducted a cross-judge robustness check by comparing Beaver-7B against GPT-4 on Mistral-7B-v0.1. As summarized in Table 17, despite minor variations in absolute delta values across helpfulness (H), harmlessness (Ha), and preference rate (P), the relative performance trends of TUSA remain highly consistent. This marginal discrepancy proves our preference signals reflect genuine algorithmic enhancements rather than evaluator bias.

These findings strongly validate our experimental design. They confirm that Beaver-7B’s specialized safety-aligned training endows it with evaluation capabilities that rival, and in some dimensions exceed, those of vastly larger or proprietary models. Consequently, utilizing Beaver-7B provides a scientifically rigorous, highly scalable, and reproducible proxy for human preferences.

## F Prompt

To ensure a robust and unbiased automated evaluation, we meticulously designed the prompts for the LLM-as-a-Judge system. Following Ji et al. (2024), the prompt templates for evaluating Harmlessness and Helpfulness are presented in Figure 11 and Figure 12, respectively.

The design of these prompts incorporates several critical evaluation principles:

• Decoupled Evaluation: A common flaw in automated evaluation is the conflation of safety and utility, where judges penalize the helpfulness score of harmless refusals. To mitigate this, our helpfulness prompt explicitly instructs the judge to “view utility and safety as two separate, unrelated aspects” and to strictly disregard safety-related factors when assessing utility.

<table><tr><td>TUSA Component</td><td>Mean ms / Step</td><td>Share of Wall Time</td></tr><tr><td>Policy Generation and Ranking (Base)</td><td>50.34</td><td>93.81%</td></tr><tr><td>Micro-Agent Confidence Probe</td><td>0.93</td><td>1.74%</td></tr><tr><td>Micro-Agent Decision Loop</td><td>0.57</td><td>1.05%</td></tr><tr><td>Risk Computation</td><td>0.23</td><td>0.42%</td></tr></table>

Table 15: Latency breakdown of TUSA components per eligible decoding step.
<table><tr><td>Judge</td><td>Helpful Agr. (%) ↑</td><td>Harmless Agr. (%) ↑</td></tr><tr><td>Beaver-7B</td><td>81.0</td><td>87.0</td></tr><tr><td>Llama-Guard-4-12B</td><td></td><td>82.0</td></tr><tr><td>GPT-OSS-120B</td><td>75.0</td><td>89.0</td></tr><tr><td>GPT-4o</td><td>79.0</td><td>91.0</td></tr></table>

Table 16: Judgment agreement of different automated evaluators with human experts. We report the simple agreement rate (%). The results show that Beaver-7B consistently aligns with high-capability judges, justifying its use as a scalable evaluator.

<table><tr><td>Evaluator</td><td>H (↑)</td><td>Ha (↑)</td><td>P (↑)</td></tr><tr><td>Beaver-7B</td><td>+36.2%</td><td>-5.5%</td><td>+15.6%</td></tr><tr><td>GPT-4</td><td>+25.8%</td><td>-2.0%</td><td>+17.6%</td></tr></table>

Table 17: Cross-judge alignment discrepancy analysis on the Mistral-7B-v0.1 base model (SafeRLHF dataset), comparing relative metric shifts (%) against baseline under different automated evaluators.

• Forced Differentiation (Tie Minimization): Default LLM judges often exhibit a “tie bias,” frequently outputting “Equal” when faced with closely matched responses. We explicitly instruct the judge to “make as many determinations as possible that they are not equal.” This compels the evaluator to scrutinize subtle qualitative differences, yielding higher-resolution preference signals.

• Structured Chain-of-Thought and Parsing: The prompt mandates that the evaluator acts as a domain expert and provides a detailed reasoning process prior to drawing a conclusion. Furthermore, the final decision must adhere to a strict regular expression format (e.g., [[responseA]]), which eliminates ambiguities during large-scale automated evaluation.

## G Case Study

Figure 13 illustrates a representative response to a harmful query. Despite the high-risk context, TUSA intervenes on only 9 out of 58 tokens. Crucially, the Cognitive Arbiter operates with high semantic precision, invoking the specialist strictly for pivotal safety concepts (e.g., “illegal”, “theft”, “deception”) and the decisive refusal verb (“assist”). Furthermore, intervening on “presence” demonstrates constructive redirection, steering the narrative from a deceptive persona toward a safe alternative. The remaining syntax, logic, and benign transitions are successfully offloaded to the base model. This confirms that TUSA enforces rigorous safety boundaries by intervening only at critical semantic junctures, avoiding dense computational redundancy.

## H The Use of Large Language Models

In the preparation of this manuscript, Large Language Models (LLMs) were utilized solely for the purpose of linguistic refinement and grammatical correction to enhance readability. All core methodologies, experimental designs, and scientific conclusions were conceived and developed exclusively by the human authors. No AI tools were used to generate new scientific insights or formulate the substance of the arguments. The authors bear full responsibility for the accuracy and integrity of the content presented herein.

![](images/35c35b368fa819c6020668e2dbf92239484958f869057b7c4b31e97cf9504f41.jpg)  
Figure 11: The prompt template used for evaluating the Harmlessness (Safety) of model responses. The structured prompt forces the LLM judge to output its final decision in a parsable format.

![](images/47b80ef964cdcbc9466725c552f9a505c212f4bd30e13dd609e37399ce52c1d9.jpg)  
Figure 12: The prompt template used for evaluating the Helpfulness (Utility) of model responses. The prompt explicitly instructs the judge to decouple safety considerations from utility assessment.

![](images/bfec76b479c762f9052d906eb8dbadfdd963f1e2a2b7da818ce2176f9ab0b7f0.jpg)

Figure 13: Case Study of Sparse Intervention. We highlight the intervened tokens in orange.
<table><tr><td rowspan="2">Model</td><td colspan="3">Helpful</td><td colspan="3">Harmless</td><td colspan="3">Preference Rate</td></tr><tr><td>Win</td><td>Lose</td><td>Tie</td><td>Win</td><td>Lose</td><td>Tie</td><td>Win</td><td>Lose</td><td>Tie</td></tr><tr><td>Llama 3.1-8B</td><td>114</td><td>85</td><td>0</td><td>103</td><td>96</td><td>0</td><td>60</td><td>42</td><td>97</td></tr><tr><td>Llama 3.2-3B</td><td>124</td><td>68</td><td>7</td><td>95</td><td>97</td><td>7</td><td>66</td><td>39</td><td>94</td></tr><tr><td>Mistral v0.1-7B</td><td>99</td><td>100</td><td>0</td><td>126</td><td>72</td><td>1</td><td>60</td><td>33</td><td>106</td></tr><tr><td>Mistral v0.2-7B</td><td>106</td><td>93</td><td>0</td><td>102</td><td>97</td><td>0</td><td>53</td><td>44</td><td>102</td></tr><tr><td>Mistral v0.3-7B</td><td>114</td><td>85</td><td>0</td><td>98</td><td>98</td><td>3</td><td>56</td><td>42</td><td>101</td></tr></table>

Table 18: Model Performance Comparison: Win, Lose, and Tie Statistics (TUSA vs. Base Model on SafeRLHF dataset).

<table><tr><td rowspan="2">Model</td><td colspan="3">Helpful</td><td colspan="3">Harmless</td><td colspan="3">Preference Rate</td></tr><tr><td>Win</td><td>Lose</td><td>Tie</td><td>Win</td><td>Lose</td><td>Tie</td><td>Win</td><td>Lose</td><td>Tie</td></tr><tr><td>Llama 3.1-8B</td><td>427</td><td>271</td><td>2</td><td>346</td><td>351</td><td>3</td><td>220</td><td>145</td><td>335</td></tr><tr><td>Llama 3.2-3B</td><td>395</td><td>261</td><td>44</td><td>373</td><td>283</td><td>44</td><td>233</td><td>120</td><td>347</td></tr><tr><td>Mistral v0.1-7B</td><td>387</td><td>313</td><td>0</td><td>428</td><td>270</td><td>2</td><td>220</td><td>104</td><td>376</td></tr><tr><td>Mistral v0.2-7B</td><td>368</td><td>331</td><td>1</td><td>403</td><td>296</td><td>1</td><td>195</td><td>123</td><td>382</td></tr><tr><td>Mistral v0.3-7B</td><td>375</td><td>325</td><td>0</td><td>344</td><td>355</td><td>1</td><td>173</td><td>153</td><td>374</td></tr></table>

Table 19: Model Performance Comparison: Win, Lose, and Tie Statistics (TUSA vs. Base Model on BeaverTails dataset).

<table><tr><td rowspan="2">Model</td><td colspan="3">Helpful</td><td colspan="3">Harmless</td><td colspan="3">Preference Rate</td></tr><tr><td>Win</td><td>Lose</td><td>Tie</td><td>Win</td><td>Lose</td><td>Tie</td><td>Win</td><td>Lose</td><td>Tie</td></tr><tr><td>Llama 3.1-8B</td><td>250</td><td>221</td><td>17</td><td>248</td><td>233</td><td>7</td><td>137</td><td>115</td><td>236</td></tr><tr><td>Llama 3.2-3B</td><td>300</td><td>179</td><td>9</td><td>271</td><td>208</td><td>9</td><td>187</td><td>95</td><td>206</td></tr><tr><td>Mistral v0.1-7B</td><td>232</td><td>250</td><td>6</td><td>330</td><td>154</td><td>4</td><td>141</td><td>33</td><td>314</td></tr><tr><td>Mistral v0.2-7B</td><td>234</td><td>254</td><td>0</td><td>294</td><td>192</td><td>2</td><td>135</td><td>94</td><td>259</td></tr><tr><td>Mistral v0.3-7B</td><td>268</td><td>218</td><td>2</td><td>287</td><td>201</td><td>0</td><td>154</td><td>86</td><td>248</td></tr><tr><td>Llama 3.1-8B</td><td>102</td><td>96</td><td>1</td><td>103</td><td>96</td><td>0</td><td>57</td><td>50</td><td>92</td></tr><tr><td>Llama 3.2-3B</td><td>95</td><td>89</td><td>15</td><td>96</td><td>88</td><td>15</td><td>61</td><td>53</td><td>85</td></tr><tr><td>Mistral v0.1-7B</td><td>135</td><td>63</td><td>1</td><td>94</td><td>105</td><td>0</td><td>61</td><td>30</td><td>108</td></tr><tr><td>Mistral v0.2-7B</td><td>113</td><td>86</td><td>0</td><td>104</td><td>95</td><td>0</td><td>53</td><td>35</td><td>111</td></tr><tr><td>Mistral v0.3-7B</td><td>105</td><td>93</td><td>1</td><td>98</td><td>101</td><td>0</td><td>49</td><td>44</td><td>106</td></tr></table>

Table 20: Model Performance Comparison: Win, Lose, and Tie Statistics (TUSA vs. Base Model on HarmfulQA dataset).

Table 21: Model Performance Comparison: Win, Lose, and Tie Statistics (TUSA vs. MARA on SafeRLHF dataset).

<table><tr><td rowspan="2">Model</td><td colspan="3">Helpful</td><td colspan="3">Harmless</td><td colspan="3">Preference Rate</td></tr><tr><td>Win</td><td>Lose</td><td>Tie</td><td>Win</td><td>Lose</td><td>Tie</td><td>Win</td><td>Lose</td><td>Tie</td></tr><tr><td>Llama 3.1-8B</td><td>343</td><td>357</td><td>0</td><td>393</td><td>305</td><td>2</td><td>197</td><td>159</td><td>344</td></tr><tr><td>Llama 3.2-3B</td><td>325</td><td>269</td><td>106</td><td>295</td><td>299</td><td>106</td><td>201</td><td>175</td><td>324</td></tr><tr><td>Mistral v0.1-7B</td><td>449</td><td>251</td><td>0</td><td>309</td><td>390</td><td>1</td><td>192</td><td>133</td><td>375</td></tr><tr><td>Mistral v0.2-7B</td><td>451</td><td>249</td><td>0</td><td>281</td><td>418</td><td>1</td><td>159</td><td>126</td><td>415</td></tr><tr><td>Mistral v0.3-7B</td><td>375</td><td>325</td><td>0</td><td>314</td><td>385</td><td>1</td><td>157</td><td>167</td><td>376</td></tr></table>

Table 22: Model Performance Comparison: Win, Lose, and Tie Statistics (TUSA vs. MARA on BeaverTails dataset).

<table><tr><td rowspan="2">Model</td><td colspan="3">Helpful</td><td colspan="3">Harmless</td><td colspan="3">Preference Rate</td></tr><tr><td>Win</td><td>Lose</td><td>Tie</td><td>Win</td><td>Lose</td><td>Tie</td><td>Win</td><td>Lose</td><td>Tie</td></tr><tr><td>Llama 3.1-8B</td><td>252</td><td>235</td><td>1</td><td>271</td><td>217</td><td>0</td><td>152</td><td>116</td><td>220</td></tr><tr><td>Llama 3.2-3B</td><td>231</td><td>229</td><td>28</td><td>234</td><td>226</td><td>28</td><td>131</td><td>126</td><td>231</td></tr><tr><td>Mistral v0.1-7B</td><td>283</td><td>199</td><td>6</td><td>223</td><td>258</td><td>7</td><td>129</td><td>104</td><td>255</td></tr><tr><td>Mistral v0.2-7B</td><td>273</td><td>215</td><td>0</td><td>227</td><td>258</td><td>3</td><td>109</td><td>95</td><td>284</td></tr><tr><td>Mistral v0.3-7B</td><td>266</td><td>222</td><td>0</td><td>230</td><td>258</td><td>0</td><td>114</td><td>110</td><td>264</td></tr></table>

Table 23: Model Performance Comparison: Win, Lose, and Tie Statistics (TUSA vs. MARA on HarmfulQA dataset).

<table><tr><td rowspan="2">Model</td><td colspan="3">Helpful</td><td colspan="3">Harmless</td><td colspan="3">Preference Rate</td></tr><tr><td>Win</td><td>Lose</td><td>Tie</td><td>Win</td><td>Lose</td><td>Tie</td><td>Win</td><td>Lose</td><td>Tie</td></tr><tr><td>Llama 3.1-8B</td><td>115</td><td>84</td><td>0</td><td>102</td><td>96</td><td>1</td><td>55</td><td>37</td><td>107</td></tr><tr><td>Llama 3.2-3B</td><td>121</td><td>72</td><td>6</td><td>93</td><td>100</td><td>6</td><td>67</td><td>46</td><td>86</td></tr><tr><td>Mistral v0.1-7B</td><td>101</td><td>98</td><td>0</td><td>124</td><td>74</td><td>1</td><td>59</td><td>32</td><td>108</td></tr><tr><td>Mistral v0.2-7B</td><td>86</td><td>111</td><td>2</td><td>112</td><td>87</td><td>0</td><td>46</td><td>47</td><td>106</td></tr><tr><td>Mistral v0.3-7B</td><td>106</td><td>93</td><td>0</td><td>97</td><td>101</td><td>1</td><td>47</td><td>43</td><td>109</td></tr><tr><td>Llama 3.1-8B</td><td>432</td><td>268</td><td>0</td><td>320</td><td>380</td><td>0</td><td>204</td><td>152</td><td>344</td></tr><tr><td>Llama 3.2-3B</td><td>392</td><td>263</td><td>45</td><td>377</td><td>279</td><td>44</td><td>231</td><td>117</td><td>352</td></tr><tr><td>Mistral v0.1-7B</td><td>390</td><td>310</td><td>0</td><td>428</td><td>271</td><td>1</td><td>218</td><td>99</td><td>383</td></tr><tr><td>Mistral v0.2-7B</td><td>386</td><td>313</td><td>1</td><td>386</td><td>313</td><td>1</td><td>203</td><td>130</td><td>367</td></tr><tr><td>Mistral v0.3-7B</td><td>375</td><td>325</td><td>0</td><td>346</td><td>352</td><td>2</td><td>178</td><td>155</td><td>367</td></tr></table>

Table 24: Model Performance Comparison: Win, Lose, and Tie Statistics (TUSA vs. ConfPO on SafeRLHF dataset).

Table 25: Model Performance Comparison: Win, Lose, and Tie Statistics (TUSA vs. ConfPO on BeaverTails dataset).

<table><tr><td rowspan="2">Model</td><td colspan="3">Helpful</td><td colspan="3">Harmless</td><td colspan="3">Preference Rate</td></tr><tr><td>Win</td><td>Lose</td><td>Tie</td><td>Win</td><td>Lose</td><td>Tie</td><td>Win</td><td>Lose</td><td>Tie</td></tr><tr><td>Llama 3.1-8B</td><td>262</td><td>224</td><td>2</td><td>231</td><td>257</td><td>0</td><td>132</td><td>126</td><td>230</td></tr><tr><td>Llama 3.2-3B</td><td>295</td><td>185</td><td>8</td><td>271</td><td>209</td><td>8</td><td>184</td><td>98</td><td>206</td></tr><tr><td>Mistral v0.1-7B</td><td>228</td><td>255</td><td>5</td><td>286</td><td>151</td><td>51</td><td>140</td><td>40</td><td>308</td></tr><tr><td>Mistral v0.2-7B</td><td>255</td><td>233</td><td>0</td><td>278</td><td>209</td><td>1</td><td>145</td><td>99</td><td>244</td></tr><tr><td>Mistral v0.3-7B</td><td>258</td><td>230</td><td>0</td><td>278</td><td>210</td><td>0</td><td>140</td><td>92</td><td>256</td></tr></table>

Table 26: Model Performance Comparison: Win, Lose, and Tie Statistics (TUSA vs. ConfPO on HarmfulQA dataset).

<table><tr><td rowspan="2">Model</td><td colspan="3">Helpful</td><td colspan="3">Harmless</td><td colspan="3">Preference Rate</td></tr><tr><td>Win</td><td>Lose</td><td>Tie</td><td>Win</td><td>Lose</td><td>Tie</td><td>Win</td><td>Lose</td><td>Tie</td></tr><tr><td>Llama 3.1-8B</td><td>102</td><td>98</td><td>0</td><td>114</td><td>86</td><td>0</td><td>71</td><td>55</td><td>74</td></tr><tr><td>Llama 3.2-3B</td><td>101</td><td>95</td><td>4</td><td>104</td><td>92</td><td>4</td><td>61</td><td>52</td><td>87</td></tr><tr><td>Mistral v0.1-7B</td><td>104</td><td>95</td><td>1</td><td>98</td><td>101</td><td>1</td><td>56</td><td>53</td><td>91</td></tr><tr><td>Mistral v0.2-7B</td><td>107</td><td>93</td><td>0</td><td>111</td><td>89</td><td>0</td><td>66</td><td>48</td><td>86</td></tr><tr><td>Mistral v0.3-7B</td><td>105</td><td>94</td><td>1</td><td>96</td><td>102</td><td>2</td><td>57</td><td>54</td><td>89</td></tr></table>

Table 27: Model Performance Comparison: Win, Lose, and Tie Statistics (TUSA vs. MARA on AlpacaEval dataset).

<table><tr><td rowspan="2">Model</td><td colspan="3">Helpful</td><td colspan="3">Harmless</td><td colspan="3">Preference Rate</td></tr><tr><td>Win</td><td>Lose</td><td>Tie</td><td>Win</td><td>Lose</td><td>Tie</td><td>Win</td><td>Lose</td><td>Tie</td></tr><tr><td>Llama 3.1-8B</td><td>88</td><td>79</td><td>0</td><td>94</td><td>72</td><td>1</td><td>62</td><td>47</td><td>58</td></tr><tr><td>Llama 3.2-3B</td><td>98</td><td>69</td><td>0</td><td>89</td><td>77</td><td>1</td><td>57</td><td>37</td><td>73</td></tr><tr><td>Mistral v0.1-7B</td><td>93</td><td>73</td><td>1</td><td>77</td><td>88</td><td>2</td><td>42</td><td>38</td><td>87</td></tr><tr><td>Mistral v0.2-7B</td><td>88</td><td>79</td><td>0</td><td>83</td><td>84</td><td>0</td><td>41</td><td>37</td><td>89</td></tr><tr><td>Mistral v0.3-7B</td><td>85</td><td>82</td><td>0</td><td>86</td><td>81</td><td>0</td><td>42</td><td>38</td><td>87</td></tr></table>

Table 28: Model Performance Comparison: Win, Lose, and Tie Statistics (TUSA vs. MARA on JustEval dataset).