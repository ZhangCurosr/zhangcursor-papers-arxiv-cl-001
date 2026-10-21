# Why Static Probes Fail: Context Determines Deception Feature Composition

Yao Huang<sup>1,2∗</sup>, Siyuan Chen<sup>1∗</sup>, Zhenghan Zhou<sup>1</sup>, Yinpeng Dong<sup>1,2†</sup>

<sup>1</sup>College of Artificial Intelligence, Tsinghua University, Beijing 100083, China <sup>2</sup>Shanghai Qi Zhi Institute, Shanghai 200232, China

huangyao26@mails.tsinghua.edu.cn, dongyinpeng@mail.tsinghua.edu.cn

## Abstract

With increasing autonomy, Large Language Models (LLMs) may engage in deceptive behaviors, such as scheming to pursue misaligned goals or sandbagging to escape oversight. Monitoring outputs alone is insufficient, as models can produce seemingly benign responses while harboring misaligned internal objectives. While mechanistic approaches like linear probing are promising, they rely on the assumption that deception corresponds to a single direction in representation space. We demonstrate that such single-direction probes fail to generalize across diverse deception scenarios. To bridge this gap, we propose representing deception as a multi-dimensional subspace rather than a single direction. Our method utilizes iterative orthogonal decomposition to identify complementary directions that capture distinct facets of deceptive intent. To effectively leverage these directions for detection, we find that static combinations prove brittle due to strong domain dependence across different scenarios. We therefore introduce a domain-adaptive gating mechanism that dynamically composes the extracted directions through few-shot adaptation at test time. With only 10 examples, our approach achieves an average AUROC of 0.97 and 94% recall at 5% FPR across all evaluated datasets, substantially outperforming existing methods.

## 1 Introduction

As LLMs are granted increasing autonomy in high-stakes domains, concerns have grown regarding their potential for deceptive behavior [5, 6, 14, 16, 27]. Deception refers to cases in which a model strategically produces misleading signals to advance objectives that are misaligned with the intentions of its developers or users. Such behaviors can take multiple forms, including in-context scheming [19], where hidden goals are pursued during deployment; sandbagging [3, 16], in which performance is intentionally degraded during evaluation to evade oversight; and the concealment of long-term objectives to avoid modification during training [8, 11, 21, 25]. These risks render deception a distinct safety challenge requiring targeted countermeasures. However, training-based mitigation approaches [24], such as enforcing alignment objectives through reinforcement learning, may introduce new failure modes, including reward hacking and superficially compliant yet strategically misaligned behaviors [2, 13]. In contrast, detection-based methods [10, 12] provide a complementary and lightweight line of defense that does not require modifying the model, enabling monitoring, timely intervention when preventive measures fail.

Most existing detection approaches rely on analyzing model outputs [12, 17]. However, this strategy is limited because a sufficiently capable model can produce benign-looking responses while internally representing misaligned goals, masking deceptive intent. This limitation has motivated a shift toward mechanistic interpretability methods [18, 29] that probe models’ internal representations rather than surface behavior alone. A prominent approach is linear probing [10], where classifiers are trained on intermediate activations to identify directions in the representation space associated with deceptive behavior. In principle, this enables detection even when outputs appear compliant.

Nevertheless, representation-based detection methods often rely on a strong and under-examined assumption that deception corresponds to a single, universal direction in the representation space [10, 29]. This assumption treats deception as a monolithic construct with consistent internal signatures across tasks and contexts, potentially overlooking the diverse and domain-dependent nature of deceptive strategies in advanced LLMs. As demonstrated in Table 1, such single-direction formulations generalize poorly across datasets and suffer substantial performance degradation when applied to heterogeneous deceptive scenarios, indicating that internal representations of deception are not universally aligned across domains.

Motivated by this phenomenon, we aim to design a robust detector capable of identifying deceptive behaviors across diverse datasets and their corresponding domains. Rather than modeling deception with a single vector, we represent it as a multi-dimensional subspace in which different directions capture complementary deception-related features. This formulation naturally raises two challenges: first, how to identify multiple separable directions that meaningfully characterize distinct aspects of deception; and second, how to combine these directions effectively, given that deceptive behaviors vary substantially across datasets and domains. Static combination strategies, such as simple adding, are insufficient in this setting, as directions that are informative in one domain may be irrelevant or noisy in another, motivating the need for adaptive composition mechanisms.

To address this, we propose Dynamic Deception Subspace Navigation (DDSN), a novel method that dynamically navigates the deception subspace to precisely capture LLM deception under varying dataset conditions. First, we use Iterative Orthogonal Subspace Decomposition (IOSD) to extract a set of complementary directions, ensuring that each subsequent component captures residual deceptive information orthogonal to previous ones. Second, to handle the challenge of domain variability, we propose a Domain-Adaptive Gating Mechanism. Rather than relying on fixed weights, this mechanism leverages test-time few-shot adaptation to dynamically combine the extracted directions based on the specific characteristics of the dataset. This enables the model to focus on the most relevant deceptive features for the given domain. Experimentally, our method demonstrates exceptional efficiency and robustness: with just 10 adaptation examples, it achieves an average AUROC of 0.97 and a recall of 95% at a 5% FPR across evaluated datasets, significantly outperforming existing static probing methods. The 10-sample requirement is reasonable in practice: the detector transfers across related domains without re-adaptation (Figure 3), so new examples are only needed when a substantial context shift occurs, which can be flagged by a lightweight off-the-shelf classifier.

Our main contributions are summarized as follows:

➊ We reveal that deception in LLMs manifests differently across various datasets, challenging the prevailing assumption of a universal, single-direction detector. Our empirical analysis demonstrates that single-direction probes fail to generalize, highlighting the need for a more nuanced approach to deception detection.

➋ We introduce Dynamic Deception Subspace Navigation (DDSN), a novel method that combines Iterative Orthogonal Subspace Decomposition with an Adaptive Gating Network. This approach effectively disentangles complementary deceptive features and dynamically adjusts their importance based on dataset characteristics, addressing the limitations of static detection.

➌ Extensive evaluations across multiple diverse datasets demonstrate that our method significantly outperforms existing baselines. DDSN achieves superior detection accuracy and generalization with minimal adaptation data, requiring only 10 examples per dataset.

## 2 Related Works

## 2.1 Deceptive Behaviors in LLMs

Recent research has unveiled the complex landscape of deceptive behaviors in LLMs, leading to the development of various datasets and evaluation methods. For instance, PKU-DeceptionBench [16] develops 180 scenarios covering five categories of LLM deception: sycophancy, strategic deception, honesty evasion, alignment faking, and sandbagging. THU-DeceptionBench [14] establishes 150 scenarios across five real-world domains, exploring intrinsic behavioral patterns and extrinsic factors influencing deception. The MASK dataset [22] investigates deception induced by pressure prompts. [11, 15, 19, 23] contribute datasets focusing on various forms of deception, including concealment of insider information, denial of policy violations, and attempts to circumvent oversight mechanisms. [3] introduce scenarios for sandbagging, where models intentionally provide incorrect responses to specific queries. [10], building on [20], develop datasets for roleplay scenarios simulating human-like strategic deception. These diverse datasets and evaluation methods, each with its unique focus, collectively reveal significant variability in deceptive behaviors across contexts, highlighting the challenge of detecting subtle forms of deception in LLMs and underscoring the need for nuanced, context-aware approaches to deception detection in LLMs.

Dynamic	Deception	Subspace	Navigation	(DDSN)  
![](images/4fb19ef3a991655bb787483fd34609bcb508efbde3798632ce28f6a88ccb182e.jpg)  
Figure 1: Overview of the Dynamic Deception Subspace Navigation (DDSN) framework. The framework consists of three stages. Phase 1: Subspace Construction. Iterative Orthogonal Subspace Decomposition (IOSD) is applied to extract a set of orthogonal directions that capture complementary components of the deception subspace. Phase 2: Domain-Adaptive Gating. A gating network is trained using base-domain data and a small number of target-domain samples to learn contextdependent weighting over the extracted directions. Phase 3: Test-time Inference. At inference, input activations are projected onto the learned directions, and the final deception score is computed as a dynamically weighted combination of the resulting scores.

## 2.2 Detection for LLM Deception

Prior works have explored approaches to detect deception in LLMs, categorized into black-box and white-box methods. Black-box approaches primarily focus on model outputs. [20] develops a classifier using responses to multiple-choice follow-up questions. [12, 17] utilize chain-of-thought monitoring. While these methods offer valuable insights, they may not fully capture the complex nature of deceptive behaviors in LLMs, especially when models maintain a facade of appropriate responses despite internal misalignment. This challenge underscores the importance of white-box techniques that can directly examine the model’s internal representations. Addressing this, [4] propose an innovative unsupervised approach that identifies directions in the embedding space associated with logical consistency across negated statements. Complementing this, [18] explore the efficacy of straightforward statistical methods, demonstrating that even simple difference-in-means probes can yield meaningful results in detecting deceptive patterns. Recent work by [10] advances this field by introducing linear probes that monitor model activations, evaluating their effectiveness in realistic scenarios of deceptive behavior. Despite these advancements, the generalizability of these methods remains limited, often relying on the assumption of a universal direction effective across all datasets.

## 3 Preliminaries

## 3.1 Autoregressive Language Model

An autoregressive language model predicts the next token in a sequence based on the preceding tokens. Given an input sequence $\mathbf { x } = ( x _ { 1 } , \dots , x _ { T } )$ , the model generates hidden states at each layer. The hidden state h $\mathbf { \chi } ^ { \bullet } \in \mathbb { R } ^ { D }$ of a token at layer l can be calculated as:

$$
\mathbf { h } _ { l } = \mathbf { h } _ { l - 1 } + \mathbf { a } _ { l } + \mathbf { m } _ { l }\tag{1}
$$

$$
\mathbf { m } _ { l } = \mathbf { W } _ { \mathrm { o u t } } ^ { l } \sigma ( \mathbf { W } _ { \mathrm { i n } } ^ { l } \gamma ( \mathbf { h } _ { l - 1 } + \mathbf { a } _ { l } ) )\tag{2}
$$

where a<sub>l</sub> and m<sub>l</sub> represent the outputs of the attention block and the feed-forward network (FFN) layer; $\mathbf { W } _ { \mathrm { i n } } ^ { l }$ and $\mathbf { \bar { W } } _ { \mathrm { o u t } } ^ { i }$ are the weight matrices of FFN layers; σ is the activation function, and γ denotes the layer normalization.

For our analysis of deceptive language patterns, we focus on extracting the hidden states from a specific layer l. The activation matrix $\dot { \mathbf { X } } ^ { l } \in \mathbb { R } ^ { \dot { N } \times D }$ for an input sequence at layer l is formed by concatenating these hidden states:

$$
\mathbf { X } ^ { l } = [ \mathbf { h } _ { 1 } ^ { l } ; \mathbf { h } _ { 2 } ^ { l } ; \ldots \vdots \mathbf { h } _ { N } ^ { l } ]\tag{3}
$$

where $\mathbf { h } _ { i } ^ { l }$ is the hidden state of the i-th sample at layer l. This activation matrix serves as the foundation for our subsequent analysis, capturing the model’s internal representation of the input sequence at a specific layer.

## 3.2 Linear Probing

Linear probing is a technique used to analyze the information encoded in model activations at a specific layer. In our context, we employ linear probing to investigate how deception-related information is represented within the activation space of language models at a chosen layer l.

Given a set of activations $\mathbf { X } ^ { l } \in \mathbb { R } ^ { N \times D }$ from layer l and corresponding deception labels $\mathbf { Y } \in \mathbb { R } ^ { N }$ where N is the number of samples, linear probing aims to find a weight vector w $\in \mathbb { R } ^ { D }$ that optimally maps the activations to the labels. This is typically achieved by minimizing a loss:

$$
\mathcal { L } ( \mathbf { w } ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \ell ( \mathbf { w } ^ { T } \mathbf { x } _ { i } ^ { l } , y _ { i } ) + \lambda \| \mathbf { w } \| ^ { 2 }\tag{4}
$$

where ℓ is a suitable loss function (e.g., cross-entropy for classification), $\mathbf { x } _ { i } ^ { l }$ is the i-th activation sample from layer $l , y _ { i }$ is its label, and λ is a regularization parameter.

Then, through optimization, we can obtain the optimal $\mathbf { w } ,$ , which can be expressed as:

$$
\mathbf { w } ^ { * } = \arg \operatorname* { m i n } _ { \mathbf { w } } \left( \mathcal { L } ( \mathbf { w } ) \right)\tag{5}
$$

The performance of this linear probe provides insights into the linear separability of deception-related features within the model’s activation space at layer l. However, traditional linear probing may not capture the full complexity of deceptive behaviors, as it assumes that all relevant information can be extracted through a single direction in the activation space. This limitation motivates our proposed Iterative Orthogonal Subspace Decomposition method, which we will introduce in the following section to address the multi-faceted nature of deception in language.

## 4 Methodology

## 4.1 Iterative Orthogonal Subspace Decomposition

To identify complementary directions that capture deceptive information, we draw inspiration from the match pursuit algorithm in [7] and propose an Iterative Orthogonal Subspace Decomposition (IOSD) approach. IOSD decomposes the deception subspace into a series of orthogonal components, each capturing residual deception-related features orthogonal to previous components.

Let $\mathbf { X } ^ { l } \in \mathbb { R } ^ { N \times D }$ be the set of activations from layer l of the model, where N is the number of samples and D is the dimensionality of the activation space. IOSD finds a sequence of orthogonal directions $\left\{ \mathbf { w } _ { 1 } , \mathbf { w } _ { 2 } , \ldots , \mathbf { w } _ { K } \right\}$ , such that each $\mathbf { w } _ { i }$ captures a distinct component of deceptive information. We iteratively perform linear probing in the orthogonal complement space of previously found directions:

$$
\begin{array} { c } { \mathbf { w } _ { i } = \underset { \mathbf { w } } { \arg \operatorname* { m i n } } \left( \mathcal { L } ( \mathbf { w } ) \right) } \\ { \mathbf { s u b j e c t \ t o \quad } \mathbf { w } ^ { T } \mathbf { w } _ { j } = 0 , \quad \forall j < i } \end{array}\tag{6}
$$

where $\mathcal { L } ( \mathbf { w } )$ is the same loss as in the linear probing section. After obtaining $\mathbf { w } _ { i }$ , we update $\mathbf { X } ^ { l }$ :

$$
\mathbf { X } _ { i + 1 } ^ { l } = \mathbf { X } ^ { l } - \mathbf { w } _ { i } \mathbf { w } _ { i } ^ { T } \mathbf { X } ^ { l }\tag{7}
$$

This process is repeated until the norm of $\mathbf { w } _ { i }$ falls below a predefined threshold ϵ, or until a maximum number of iterations K is reached. The threshold ϵ is introduced because when the norm of $\mathbf { w } _ { i }$ becomes very small, it indicates that the remaining information in the orthogonal complement space is no longer strongly related to deception, and further iterations may not yield meaningful directions.

The resulting set $\left\{ \mathbf { w } _ { 1 } , \mathbf { w } _ { 2 } , \ldots , \mathbf { w } _ { K } \right\}$ captures different aspects of deceptive information in the activation space. By iteratively extracting these directions from the orthogonal complement space, IOSD can uncover multiple facets of deception that may not be captured by a single linear probe.

Next, we present two key theorems that highlight the unique characteristics of the IOSD method:

Theorem 4.1 (Hierarchical Feature Capture in IOSD). Let $\mathbf { X } ^ { l } \in \mathbb { R } ^ { N \times D }$ be the set ofactivationsfrom layer l ofa model, and $\mathbf { y } \in \{ 0 , 1 \} ^ { N }$ be the corresponding deception labels. The Iterative Orthogonal Subspace Decomposition $( I O S D )$ method extracts a set oforthogonal directions $\left\{ \mathbf { w } _ { 1 } , \mathbf { w } _ { 2 } , \ldots , \mathbf { w } _ { K } \right\}$ such thatfor any pair ofdirections $\mathbf { w } _ { i }$ and ${ \bf w } _ { j }$ where $i < j$

$$
I ( \mathbf { w } _ { i } ^ { T } \mathbf { X } ^ { l } ; \mathbf { y } ) \geq I ( \mathbf { w } _ { j } ^ { T } \mathbf { X } ^ { l } ; \mathbf { y } )\tag{8}
$$

where $I ( \cdot ; \cdot )$ denotes mutual information.

Theorem 4.2 (Non-linear Approximation of IOSD). Given the set ofdirections $\left\{ \mathbf { w } _ { 1 } , \mathbf { w } _ { 2 } , \ldots , \mathbf { w } _ { K } \right\}$ extracted by the IOSD methodfrom activations $\mathbf { X } ^ { l } \in \mathbb { R } ^ { N \times D }$ , for any continuous function $f : \mathbb { R } ^ { D } $ R and $\epsilon > 0$ , there exists a set ofcoefficients $\left\{ a _ { 1 } , . . . , a _ { K } \right\}$ such that:

$$
\left\| \boldsymbol { f } ( \mathbf { X } ^ { l } ) - \sum _ { i = 1 } ^ { K } a _ { i } ( \mathbf { w } _ { i } ^ { T } \mathbf { X } ^ { l } ) \right\| _ { 2 } < \epsilon\tag{9}
$$

Moreover, as more directions are included, the approximation error monotonically decreases.

These properties make IOSD particularly well-suited for our task of analyzing and detecting deception in language models. The hierarchical nature of feature extraction allows us to focus on the most salient deception patterns first, while still capturing more subtle cues. This characteristic ensures that our sequential iteration process does not miss any important directions, as each step captures the most significant remaining information. The ability to approximate non-linear patterns means that IOSD can capture complex deception strategies that might not be evident from simple linear analysis. Together, these characteristics enable IOSD to provide a comprehensive and nuanced understanding of deception in language model outputs, facilitating more effective detection and analysis strategies.

## 4.2 Domain-Adaptive Gating Mechanism

To address domain variability in deceptive behaviors, we then introduce the Domain-Adaptive Gating Mechanism. This mechanism dynamically combines the directions extracted by Iterative Orthogonal Subspace Decomposition (IOSD) based on the characteristics of the dataset, allowing the model to focus on the most relevant deceptive features through test-time few-shot adaptation.

Using the set of directions $\left\{ \mathbf { w } _ { 1 } , \mathbf { w } _ { 2 } , \ldots , \mathbf { w } _ { K } \right\}$ extracted by IOSD, the Domain-Adaptive Gating Mechanism computes a set of domain-dependent weights $\{ \alpha _ { 1 } , \alpha _ { 2 } , \ldots , \alpha _ { K } \}$ . During test time, we collect a small support set of approximately 10 labeled examples for each new domain, denoted as $\begin{array} { r } { \boldsymbol { S } = \{ ( \mathbf { x } ^ { ( l ) } , y ^ { ( \hat { l } ) } ) \} _ { l = 1 } ^ { N } } \end{array}$ with $N \approx 1 0$ . The gating mechanism performs few-shot adaptation by updating the gating network parameters $\theta$ to minimize the following loss over the support set:

$$
\mathcal { L } _ { \mathrm { a d a p t } } ( \theta ) = \sum _ { l = 1 } ^ { N } \left( \sum _ { i = 1 } ^ { K } \alpha _ { i } ( \mathbf { x } ^ { ( l ) } ) \mathbf { w } _ { i } ^ { \top } \mathbf { x } ^ { ( l ) } - y ^ { ( l ) } \right) ^ { 2 } .\tag{10}
$$

where $Y _ { i }$ represents the target deception label associated with the direction $\mathbf { w } _ { i }$ . The domain-dependent weights $\alpha _ { i } \bar { ( \mathbf { x } ^ { l } ) }$ are computed through a gating network $G _ { \theta } \colon$

$$
\alpha _ { i } ( \mathbf { x } ^ { l } ) = G _ { \theta } ( \mathbf { x } ^ { l } , \mathbf { w } _ { i } )\tag{11}
$$

The parameters θ of the gating network are updated using gradient descent on the few-shot examples, allowing the model to adapt quickly to new domains. After adapting on these 10 samples, the final prediction for a test input is computed by aggregating the weighted directions:

$$
\hat { Y } = \sum _ { i = 1 } ^ { K } \alpha _ { i } ( \mathbf { x } ^ { l } ) \mathbf { w } _ { i } ^ { T } \mathbf { x } ^ { l }\tag{12}
$$

Table 1: Deception Detection Results. Performance comparison across different datasets of Static and Adaptive methods on LLaMA-3.3-70B-Instruct and Qwen3-32B, highlighting the benefits of adaptive approaches under domain shifts. Bolded entries indicate our method. Dashes (“-”) denote datasets where few-shot adaptation is infeasible due to excessively long input sequences.
<table><tr><td rowspan="2">Method</td><td colspan="2">AI Liar</td><td colspan="2">|Insider Trading</td><td colspan="2">Insider D.D.</td><td colspan="2">Sandbagging</td><td colspan="2">MASK</td><td colspan="2">PKU-Bench</td><td colspan="2">THU-Bench</td></tr><tr><td>|AUC Rec FPR</td><td></td><td>|AUC Rec FPR</td><td></td><td>AUC Rec FPR</td><td></td><td>|AUC Rec FPR</td><td></td><td>AUC Rec FPR</td><td>|AUC Rec FPR</td><td></td><td></td><td>AUC Rec FPR</td></tr><tr><td colspan="10">Llama-3.3-70B-Instruct</td><td colspan="7"></td></tr><tr><td>Linear Probe</td><td>0.96</td><td>1.00 0.31</td><td>1.00 1.00</td><td>1.00 0.97</td><td>1.00</td><td>0.96 0.98</td><td>1.00 0.92</td><td>0.78</td><td>1.00 0.92</td><td>0.80</td><td>0.67</td><td>0.24</td><td>0.61 0.84</td><td>0.90 0.75</td></tr><tr><td>Difference of Means</td><td>0.95</td><td>0.98 0.19</td><td>0.98 1.00</td><td>1.00 0.90</td><td>0.97 0.54</td><td>1.00</td><td>1.00 0.21</td><td>0.77</td><td>0.59 0.24</td><td>0.86</td><td>0.81</td><td>0.26</td><td>0.88</td><td>0.52</td></tr><tr><td>Linear Discriminant Analysis</td><td>0.96</td><td>0.87 0.08</td><td>0.99 0.99</td><td>0.21 0.94</td><td>0.27</td><td>0.00 0.90</td><td>1.00 0.78</td><td>0.77</td><td>0.56 0.20</td><td>0.80</td><td>0.63</td><td>0.23 0.68</td><td>0.76</td><td>0.60</td></tr><tr><td>LLM-as-Judge</td><td>0.97</td><td>0.69 0.01</td><td>0.99 0.97</td><td>0.03 0.87</td><td>0.86 0.25</td><td>0.99</td><td>0.71 0.01</td><td>0.97</td><td>0.85 0.05</td><td>0.81</td><td>0.26</td><td>0.00</td><td>0.97 0.73</td><td>0.00</td></tr><tr><td>Linear Probe (Few)</td><td>0.96</td><td>1.00 0.31</td><td>1.00 1.00</td><td>1.00</td><td>0.97 1.00 0.96</td><td>0.99</td><td>1.00 0.88</td><td>0.79</td><td>0.94 0.62</td><td>0.81</td><td>0.67</td><td>0.19</td><td>0.61 0.89</td><td>0.75</td></tr><tr><td>LLM-as-Judge (Few) Ours</td><td>0.98 0.95</td><td>0.63 0.00 0.94 0.28</td><td>1.00 0.96 0.00</td><td></td><td>0.98 0.96 0.08</td><td>1.00</td><td>1.00 0.20</td><td>0.97 0.93</td><td>0.94 0.05</td><td>0.86</td><td>0.93</td><td>0.44 0.00 0.15</td><td>0.99 0.89 0.98 0.91</td><td>0.00 0.07</td></tr><tr><td colspan="10"></td><td colspan="7">0.87 0.13 0.94</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>Qwen3-32B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Linear Probe</td><td>0.90</td><td>0.97 0.32</td><td>0.66 0.30</td><td>0.01</td><td>0.53 0.03</td><td>0.01 1.00</td><td>1.00 0.02</td><td>0.70</td><td>0.76 0.47</td><td>0.78</td><td>0.68</td><td>0.22</td><td>0.64 0.67</td><td>0.41</td></tr><tr><td>Difference of Means</td><td>0.86</td><td>0.30 0.02</td><td>0.36 0.00</td><td>0.00</td><td>0.40 0.00</td><td>0.00 1.00</td><td>0.13 0.00</td><td>0.59</td><td>0.03 0.08</td><td>0.84</td><td>0.36</td><td>0.02 0.93</td><td>0.33</td><td>0.01 0.07</td></tr><tr><td>Linear Discriminant Analysis</td><td>0.90</td><td>0.57 0.03</td><td>0.69 0.04</td><td>0.00 0.60</td><td>0.00</td><td>0.00 1.00</td><td>0.74 0.00</td><td>0.60</td><td>0.09 0.06</td><td>0.77 0.79</td><td>0.20</td><td>0.03 0.84</td><td>0.47</td><td></td></tr><tr><td>LLM-as-Judge</td><td>0.96</td><td>0.63 0.01</td><td>0.78 0.48</td><td>0.05</td><td>0.55 0.18</td><td>0.11 1.00</td><td>0.87 0.00</td><td>0.97</td><td>0.84 0.06</td><td></td><td>0.32</td><td>0.00</td><td>0.94 0.65</td><td>0.00</td></tr><tr><td>Linear Probe (Few)</td><td>0.90</td><td>0.96 0.32</td><td>0.66 0.30</td><td>0.01</td><td>0.53 0.03</td><td>0.01 1.00</td><td>1.00</td><td>0.02 0.70</td><td>0.76</td><td>0.47 0.78</td><td>0.68</td><td>0.22</td><td>0.64 0.67</td><td>0.41</td></tr><tr><td>LLM-as-Judge (Few)</td><td>0.97</td><td>0.71 0.01</td><td></td><td></td><td></td><td></td><td></td><td>0.97</td><td>0.90 0.11</td><td>0.92</td><td>0.36</td><td>0.00</td><td>1.00</td><td>0.58 0.00</td></tr><tr><td>Ours</td><td></td><td>0.91 0.90 0.23</td><td>0.98 0.52 0.00</td><td></td><td>0.93 0.34 0.02</td><td></td><td>1.00 1.00 0.05</td><td>0.88</td><td>0.63 0.11</td><td>0.86</td><td>0.68 0.14</td><td></td><td>0.97</td><td>0.82 0.03</td></tr></table>

This adaptive approach enables our model to dynamically adjust its focus based on each test-time domain, leading to more accurate and robust detection across diverse scenarios. By leveraging the complementary directions from IOSD and adapting them through few-shot fine-tuning on a small set of domain-specific examples, the Domain-Adaptive Gating Mechanism captures the multi-faceted nature of deceptive language more effectively than static approaches.

## 5 Experiments

## 5.1 Experimental Settings

Models. For fair comparison with prior methods and because deceptive behavior in frontier-scale models is of greater concern, we follow [10] and use Llama-3.3-70B-Instruct [9] as our main model. As a frontier-scale, open-source model, it represents a realistic and high-impact setting in which deceptive capabilities are both plausible and safety-critical. We additionally include Qwen-3-32B [28] to evaluate effectiveness across different model families.

Datasets. We divide our data into three functional categories: Source Domains for subspace extraction, Target Domains for adaptation and evaluation, and a Control Domain for threshold calibration.

Source Domains (Training). We extract the initial deception subspace using two datasets that serve as the basis for IOSD: (1) Instructed Pairs [29]: Unlike standard behavioral datasets, this uses an off-policy approach. Each sample contains a contrastive instruction (prompting honesty or deception) followed by a ground-truth fact [1]. We use activations from the fact tokens (truncating the last 5) to capture the internal preparation for deception before the generation of falsehoods. (2) Roleplaying Scenarios [10]: Simple roleplay setups where the model adopts a deceptive persona. These sources provide the primary basis vectors for our subspace.

Target Domains (Evaluation & Adaptation). We evaluate our method on a diverse set of datasets covering different types of complex deceptive behaviors. For Domain-Adaptive Gating, we use 10 randomly selected examples from each dataset for few-shot adaptation and reserve the remaining samples for evaluation.

• Strategic Scheming: We use Insider Trading dataset [23] (Report and Confirmation subsets), where models conceal illegal trading. We also include Sandbagging [3], where models strategically underperform to evade evaluation.

• Instrumental Deception: We use AI Liar [20]. Here, agents lie to maximize rewards (e.g., a doctor prescribing inferior treatment) without explicit "Yes/No" forced choices.

• Context-Conflicted Honesty: Using the MASK dataest [22], we could test scenarios where agents are given ground-truth facts but are pressured by user prompts to contradict them.

![](images/6548debfc7cbb653b2dee6796dac01e2b09f73644b27f67e3f8e5a9f36b82cb5.jpg)  
Figure 2: ROC Analysis. Mean ROC curves averaged across all target datasets, showing that our method maintains high recall at low false positive rates on honest and control data.

Table 2: Score Aggregation Results. Performance comparison of different methods for aggregating token-level scores into the detection score, where w and s denote the window size and stride.
<table><tr><td>Method</td><td>AUROC</td><td>Rec@5%</td><td>FPR@5%</td></tr><tr><td>Mean</td><td>0.97</td><td>0.94</td><td>0.13</td></tr><tr><td>Max</td><td>0.94</td><td>0.83</td><td>0.13</td></tr><tr><td>Windoww=8,s=4</td><td>0.94</td><td>0.80</td><td>0.10</td></tr><tr><td>Windoww=48,s=24</td><td>0.95</td><td>0.86</td><td>0.10</td></tr><tr><td>Windoww=192,s=128</td><td>0.96</td><td>0.90</td><td>0.10</td></tr></table>

• Comprehensive Benchmarks: To ensure coverage, we include PKU-DeceptionBench [16] (covering alignment faking, sycophancy, etc.) and the Level 2 scenarios from THU-DeceptionBench [14]. The latter involves sophisticated prompts mimicking social engineering attacks, which differ from simple instruction following.

Control Domain (Calibration). To calibrate the false positive rate (FPR), we follow [10] and use 10k samples from the Alpaca dataset [26]. As it contains standard instruction-following tasks, we treat it as a benign baseline for setting thresholds that avoid flagging helpful outputs as deceptive.

Baselines. We compare our method against a diverse set of baselines, including supervised linear probing, simple mechanistic approaches, and external black-box monitoring. For linear probing, we follow [10] and train a static logistic regression probe on raw activations from the Instructed Pairs dataset, which the original paper identifies as the supervision source yielding the strongest generalization performance among available alternatives. To ensure a fair comparison with adaptive baselines, we additionally evaluate a few-shot variant of the linear probe that incorporates the same target-domain examples used by few-shot black-box monitoring. We further include two mechanistic baselines based on activation statistics: Difference of Means and Linear Discriminant Analysis (LDA) [18]. Finally, we evaluate GPT-4o-based black-box monitoring as an external judge operating solely on behavioral outputs, considering both a standard prompt-based setting and a few-shot variant augmented with target-domain examples.

Evaluation Metrics. We report three metrics to evaluate both discriminative power and reliability. First, we use the AUROC to measure separability between deceptive and honest representations in a threshold-independent manner. To reflect deployment scenarios where false alarms must be tightly controlled, we additionally report deceptive recall at a 5% false positive rate on benign control samples (Recall@5%FPR), which quantifies the fraction of deceptive examples detected under a strict error constraint. Finally, to assess generalization beyond the control distribution, we report the false positive rate on the target honest samples at the same calibrated threshold (FPR@5%FPR), measuring how often honest behaviors are misclassified when the detector maintains a 5% FPR.

## 5.2 Main Results

Comparison with Baselines. Across datasets with diverse and complex deceptive behaviors (Table 1, Figure 2), baseline methods exhibit clear limitations. Linear probing generalizes poorly to benchmarks with heterogeneous deception patterns (e.g., Mask, PKU-DeceptionBench, THU-DeceptionBench) and often attains high deceptive recall only at the cost of extremely high false positive rates on honest samples. Adding few-shot supervision yields only marginal improvements and does not resolve this trade-off. Difference of Means and LDA occasionally outperform linear probes, but still suffer from elevated false positives, indicating that a single global direction is insufficient to capture deception across diverse contexts. Black-box monitoring reduces false positives, yet degrades on implicit or compositional deception (e.g., PKU-DeceptionBench), where detection remains limited even with few-shot prompting. In contrast, our method consistently achieves strong and balanced performance across all datasets and model families, reaching average AUROC of 0.97 on Llama-3.3- 70B-Instruct and 0.93 on Qwen3-32B, while maintaining high recall with acceptable false positive rates. Although Qwen3-32B shows increased confusion on transaction-related insider scenarios due to partial representation overlap, our approach still surpasses all baselines by a substantial margin.

![](images/17374c2e96fa492732cefd62bf9ae7a7270599572d4459f3fb5d59713bb21462.jpg)  
Figure 3: Cross-Dataset Generalization Analysis. Violin plots of deception score distributions under a cross-dataset setting, where the detector is adapted using few-shot examples from a single dataset and evaluated on the remaining benchmarks without further tuning. Adaptation is performed on one of AI Liar, PKU-DeceptionBench, or THU-DeceptionBench. Each row corresponds to the dataset used for adaptation, and each column shows the resulting performance on a target dataset.

![](images/d9f94710304c0f4d3ed0ca24328d350011531f90aef4f78cef63d45d1cffcf66.jpg)  
Figure 4: Gating Variation across Domains. Mean gating weights over seven orthogonal directions extracted from LLaMA-3.3-70B-Instruct under the Domain-Adaptive Gating Mechanism. Although Direction 1 consistently receives the highest weight as the dominant component, its activation varies substantially across datasets, and the auxiliary directions exhibit pronounced domain-specific reweighting, indicating that a single universal direction is insufficient for cross-domain adaptation.

Overall, these results demonstrate that our method provides a more reliable detection signal across diverse deception types.

Score Aggregation Stratgies. As demonstrated in Table 2, we compare several strategies for aggregating token-level anomaly scores into the detection score, including simple averaging, maximum pooling, and sliding-window aggregation with different window sizes and strides. We find that maxbased aggregation consistently underperforms mean aggregation in terms of AUROC and deceptive recall, indicating that relying on the single highest token-level score is insufficient to characterize sequence-level deception. Sliding-window methods partially mitigate this issue by requiring localized consistency within a window, but their performance remains sensitive to window size and stride, and they generally trade off recall for lower FPR. In contrast, mean aggregation achieves the strongest overall separability and the highest deceptive recall while maintaining comparable false positive rates, suggesting that deception in most cases is better reflected as a distributed signal across the sequence rather than isolated spikes. Based on this consistent advantage and its simplicity, we adopt mean aggregation as the default scoring strategy in all experiments.

Generalization Analysis. Although we use few-shot adaptation to improve robustness on diverse datasets, we find that the resulting detector also generalizes well even when adaptation is performed on only a single dataset and then applied to other datasets without further tuning. This behavior is consistent with our multi-dimensional subspace formulation, which captures a richer set of deception related features than a single direction and therefore transfers more reliably across settings. As shown in Figure 3, we adapt the method using examples from one dataset and directly evaluate it on three distinct benchmarks, including AI Liar, PKU-DeceptionBench, and THU-DeceptionBench. While this cross-dataset setting is slightly weaker than the per-dataset few-shot results in Table 1, the detector still exhibits clear separation and maintains strong recall on most datasets, indicating meaningful transfer beyond the adaptation domain.

![](images/dd426ed5f2ed5b2b834528f9cc688561a376e060f50c17d1f97a127875cc3b91.jpg)  
Figure 5: Deception Detection Examples. Each panel shows a deceptive prompt and the model response, with token-level deception scores visualized as heatmaps. Left: fraudulent product marketing. Middle: tax evasion justification. Right: climate misinformation news.

## 5.3 Gating Adaptation Analysis

As illustrated in Figure 4, we extract seven orthogonal directions from LLaMA-3.3-70B-Instruct with IOSD. Direction 1 consistently receives the primary gating weight across datasets, which is reasonable since the first extracted direction captures the most prominent and globally shared component of the learned representation. However, its activation magnitude still exhibits substantial variation across datasets, indicating that even this dominant direction is modulated in a domain-dependent manner. Moreover, the remaining directions show dataset-specific gating patterns, with different domains assigning higher importance to different auxiliary components. These observations demonstrate that cross-domain adaptation cannot be adequately characterized by a single universal direction, but instead relies on the dynamic reweighting of multiple sub-components within the learned subspace.

## 6 Discussion

Our results highlight several advantages of domain-adaptive subspace navigation over static deception detection. As illustrated in Figure 5, few-shot gating adaptation enables the detector to dynamically reweight deception-related directions in a domain-specific manner, leading to more stable and interpretable score trajectories across different stages of deceptive behavior. In particular, the detector localizes deceptive spans more precisely, often assigns elevated scores prior to the first explicit deceptive statement, and appropriately reduces scores once the model transitions back to honest behavior. These dynamics suggest that adaptive detectors can capture early intent signals and temporal structure in a way that static, domain-agnostic directions cannot.

## 7 Conclusion

In conclusion, this paper introduces a novel approach to detecting deception in LLMs by leveraging Dynamic Deception Subspace Navigation (DDSN). By moving beyond static probing methods and incorporating iterative orthogonal decomposition and domain-adaptive gating, the proposed method effectively captures diverse deceptive behaviors across various datasets. Experimental results demonstrate the superiority of DDSN, achieving high accuracy with minimal adaptation data. This work provides a valuable step towards more reliable and robust detection mechanisms for LLMs, enhancing model safety and interpretability before deployment in the real-world applications.

## References

[1] Amos Azaria and Tom Mitchell. The internal state of an llm knows when it’s lying. arXiv preprint arXiv:2304.13734, 2023.

[2] Bowen Baker, Joost Huizinga, Leo Gao, Zehao Dou, Melody Y Guan, Aleksander Madry, Wojciech Zaremba, Jakub Pachocki, and David Farhi. Monitoring reasoning models for misbehavior and the risks of promoting obfuscation. arXiv preprint arXiv:2503.11926, 2025.

[3] Joe Benton, Misha Wagner, Eric Christiansen, Cem Anil, Ethan Perez, Jai Srivastav, Esin Durmus, Deep Ganguli, Shauna Kravec, Buck Shlegeris, et al. Sabotage evaluations for frontier models. arXiv preprint arXiv:2410.21514, 2024.

[4] Collin Burns, Haotian Ye, Dan Klein, and Jacob Steinhardt. Discovering latent knowledge in language models without supervision. arXiv preprint arXiv:2212.03827, 2022.

[5] Boyuan Chen, Sitong Fang, Jiaming Ji, Yanxu Zhu, Pengcheng Wen, Jinzhou Wu, Yingshui Tan, Boren Zheng, Mengying Yuan, Wenqi Chen, et al. Ai deception: Risks, dynamics, and controls. arXiv preprint arXiv:2511.22619, 2025.

[6] Yanda Chen, Joe Benton, Ansh Radhakrishnan, Jonathan Uesato, Carson Denison, John Schulman, Arushi Somani, Peter Hase, Misha Wagner, Fabien Roger, et al. Reasoning models don’t always say what they think. arXiv preprint arXiv:2505.05410, 2025.

[7] Valérie Costa, Thomas Fel, Ekdeep Singh Lubana, Bahareh Tolooshams, and Demba Ba. From flat to hierarchical: Extracting sparse representations with matching pursuit. arXiv preprint arXiv:2506.03093, 2025.

[8] Carson Denison, Monte MacDiarmid, Fazl Barez, David Duvenaud, Shauna Kravec, Samuel Marks, Nicholas Schiefer, Ryan Soklaski, Alex Tamkin, Jared Kaplan, et al. Sycophancy to subterfuge: Investigating reward-tampering in large language models. arXiv preprint arXiv:2406.10162, 2024.

[9] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv e-prints, pages arXiv–2407, 2024.

[10] Nicholas Goldowsky-Dill, Bilal Chughtai, Stefan Heimersheim, and Marius Hobbhahn. Detecting strategic deception with linear probes. In Forty-second International Conference on Machine Learning, 2025.

[11] Ryan Greenblatt, Carson Denison, Benjamin Wright, Fabien Roger, Monte MacDiarmid, Sam Marks, Johannes Treutlein, Tim Belonax, Jack Chen, David Duvenaud, et al. Alignment faking in large language models. arXiv preprint arXiv:2412.14093, 2024.

[12] Melody Y Guan, Miles Wang, Micah Carroll, Zehao Dou, Annie Y Wei, Marcus Williams, Benjamin Arnav, Joost Huizinga, Ian Kivlichan, Mia Glaese, et al. Monitoring monitorability. arXiv preprint arXiv:2512.18311, 2025.

[13] Rohan Gupta and Erik Jenner. Rl-obfuscation: Can language models learn to evade latent-space monitors? arXiv preprint arXiv:2506.14261, 2025.

[14] Yao Huang, Yitong Sun, Yichi Zhang, Ruochen Zhang, Yinpeng Dong, and Xingxing Wei. Deceptionbench: A comprehensive benchmark for AI deception behaviors in real-world scenarios. In The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2025.

[15] Olli Järviniemi and Evan Hubinger. Uncovering deceptive tendencies in language models: A simulated company ai assistant. arXiv preprint arXiv:2405.01576, 2024.

[16] Jiaming Ji, Wenqi Chen, Kaile Wang, Donghai Hong, Sitong Fang, Boyuan Chen, Jiayi Zhou, Juntao Dai, Sirui Han, Yike Guo, et al. Mitigating deceptive alignment via self-monitoring. arXiv preprint arXiv:2505.18807, 2025.

[17] Tomek Korbak, Mikita Balesni, Elizabeth Barnes, Yoshua Bengio, Joe Benton, Joseph Bloom, Mark Chen, Alan Cooney, Allan Dafoe, Anca Dragan, et al. Chain of thought monitorability: A new and fragile opportunity for ai safety. arXiv preprint arXiv:2507.11473, 2025.

[18] Samuel Marks and Max Tegmark. The geometry of truth: Emergent linear structure in large language model representations of true/false datasets. arXiv preprint arXiv:2310.06824, 2023.

[19] Alexander Meinke, Bronson Schoen, Jérémy Scheurer, Mikita Balesni, Rusheb Shah, and Marius Hobbhahn. Frontier models are capable of in-context scheming. arXiv preprint arXiv:2412.04984, 2024.

[20] Lorenzo Pacchiardi, Alex J Chan, Sören Mindermann, Ilan Moscovitz, Alexa Y Pan, Yarin Gal, Owain Evans, and Jan Brauner. How to catch an ai liar: Lie detection in black-box llms by asking unrelated questions. arXiv preprint arXiv:2309.15840, 2023.

[21] Alexander Pan, Erik Jones, Meena Jagadeesan, and Jacob Steinhardt. Feedback loops with language models drive in-context reward hacking. arXiv preprint arXiv:2402.06627, 2024.

[22] Richard Ren, Arunim Agarwal, Mantas Mazeika, Cristina Menghini, Robert Vacareanu, Brad Kenstler, Mick Yang, Isabelle Barrass, Alice Gatti, Xuwang Yin, et al. The mask benchmark: Disentangling honesty from accuracy in ai systems. arXiv preprint arXiv:2503.03750, 2025.

[23] Jérémy Scheurer, Mikita Balesni, and Marius Hobbhahn. Large language models can strategically deceive their users when put under pressure. arXiv preprint arXiv:2311.07590, 2023.

[24] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

[25] Abhay Sheshadri, John Hughes, Julian Michael, Alex Mallen, Arun Jose, Fabien Roger, et al. Why do some language models fake alignment while others don’t? arXiv preprint arXiv:2506.18032, 2025.

[26] Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Stanford alpaca: An instruction-following llama model. https://github.com/tatsu-lab/stanford\_alpaca, 2023.

[27] Kai Wang, Yihao Zhang, and Meng Sun. When thinking llms lie: Unveiling the strategic deception in representations of reasoning models. arXiv preprint arXiv:2506.04909, 2025.

[28] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

[29] Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, et al. Representation engineering: A top-down approach to ai transparency. arXiv preprint arXiv:2310.01405, 2023.