# Unifying Detection and Adaptation in Task-Free Continual Learning

Dezheng Han<sup>1</sup>, Anbang Zhang<sup>2</sup>, Zhihao Zhu<sup>3,4</sup>, Shuaishuai Guo<sup>1,\*</sup>

<sup>1</sup>School of Control Science and Engineering, Shandong University, Jinan, China

<sup>2</sup>Department of ECE, The Hong Kong University of Science and Technology, Hong Kong

<sup>3</sup>National Medicine-Engineering Interdisciplinary Industry-Education Integration Innovation Platform,

Shandong University, Jinan, China

<sup>4</sup>Shandong Key Laboratory: Magnetic Field-free Medicine & Functional Imaging,

Qilu Hospital of Shandong University, Jinan, China

{dezhenghan, 202415670}@mail.sdu.edu.cn, azhangay@connect.ust.hk, shuaishuai\_guo@sdu.edu.cn

## Abstract

To mitigate catastrophic forgetting in downstream continual learning (CL) for large language models (LLMs), existing methods typically constrain parameter updates or introduce task-specific adaptation modules. However, these methods often rely on explicit task boundaries during training, limiting their applicability to realistic task-free scenarios. In this paper, we propose a Fisher-guided unified (FiUni) framework for batch-level task detection and parameter-efficient continual adaptation. FiUni is motivated by a key observation about the Fisher information matrix (FIM) of pre-trained models: the orthogonality among the principal subspaces of its Kronecker-Factored Approximate Curvature (K-FAC) approximation, estimated from a small number of downstream task samples, can reflect the similarity between different tasks. Based on this observation, Fi-Uni constructs FIM-derived frozen subspaces to guide low-rank adaptation (LoRA), while matching the Fisher principal subspace of each incoming batch window with historical subspaces. This enables FiUni to adaptively determine whether to reuse existing knowledge, expand a related subspace, or create a new subspace, dynamically balancing knowledge sharing and task isolation. Experiments show that FiUni can effectively infer latent batch-level task affiliations and achieve competitive performance against advanced task-aware CL methods with fewer trainable parameters.

## 1 Introduction

With the remarkable generalization capabilities of large language models (LLMs) (Raffel et al., 2020; Grattafiori et al., 2024; Yang et al., 2025) across diverse downstream tasks, enabling LLMs to continuously adapt to dynamic environments has emerged as a critical research problem (Shi et al., 2025).

However, direct full-parameter fine-tuning for continual learning (CL) (Zhou et al., 2024; Wang et al., 2024) suffers from prohibitive computational and storage costs, and inevitably leads to severe catastrophic forgetting (McCloskey and Cohen, 1989), where the model loses performance on old tasks after learning new ones. Parameter-efficient finetuning (PEFT) (Han et al., 2024; Wang et al., 2025), which updates only a small number of original or additional parameters, provides a more feasible solution to this problem. Following this idea, existing parameter-efficient continual learning methods typically manage the low-dimensional parameter update spaces of different tasks, thereby reducing inter-task interference while maintaining fine-tuning efficiency (Wang et al., 2023a; Liao et al., 2026; Biswas et al., 2026).

Despite promising performance in multi-task continual learning, existing methods typically rely on task-aware assumptions, where task boundaries or IDs are available for allocating and switching parameter subspaces. In contrast, real-world continual learning is often task-free, with data arriving in an online stream without explicit boundary annotations or task identities (Aljundi et al., 2019). Thus, the model must autonomously infer latent task structure while adapting its parameters to incoming data. Although recent methods exploit model-internal task-related signals to relax this assumption (Du et al., 2024), they mainly focus on task-label-free forgetting mitigation, rather than explicitly unifying task identification with the isolation and sharing of PEFT subspaces.

To address these issues, we need a meaningful geometric signal that can characterize the relationships among different incoming data and further guide parameter-efficient adaptation. Fisher geometry (ichi Amari, 1998) provides a promising perspective for this purpose. Recent studies have shown that the Kronecker-Factored Approximate Curvature (K-FAC) (Martens and Grosse, 2015)

![](images/a5fb3bf858c07bacf20e7b2c9401b83f6998da422dc880c8b3d6875816680fb6.jpg)  
Figure 1: Fisher principal subspaces encode task similarity. For each downstream task, we estimate the K-FAC approximation and extract the top-r principal subspaces of its gradient and activation covariance factors. Higher subspace similarity indicates more similar task geometry, while lower similarity indicates stronger orthogonality.

of the Fisher information matrix (FIM), estimated from a small number of downstream samples of different tasks on a pre-trained model, can serve as effective fixed low-rank update bases to guide LoRA along parameter-sensitive directions (Han and Guo, 2026). This motivates us to further explore whether Fisher principal subspaces can be used to model latent task relationships among different downstream data.

Under this perspective, we discover that in the K-FAC (Martens and Grosse, 2015) approximation of the FIM estimated on a few downstream samples, the orthogonality of the principal subspaces, spanned by the top-r eigenvectors of the gradient and activation covariance factors, naturally reflects the geometric similarity between distinct tasks, as shown in Figure 1. Specifically, data from identical or related tasks tend to exhibit higher Fisher principal subspace similarity, whereas those from unrelated tasks tend towards orthogonality. This indicates that for pre-trained models, the Fisher principal subspace not only characterizes the sensitive update directions for the current data but also serves as a geometric signal to identify the latent task affiliation of each batch in an online data stream. Here, a latent task does not necessarily align perfectly with human-annotated discrete tasks, but rather represents an implicit learning phase perceived by the pre-trained model, characterized by similar learning requirements and adaptation behaviors.

Motivated by the above observation, we propose a Fisher geometry-based unified task-free framework, named FiUni, for the continual parameterefficient fine-tuning scenario. The key idea of Fi-Uni is to utilize the same Fisher principal subspace signal to simultaneously perform batch-level latent task detection and LoRA subspace construction. Specifically, FiUni maintains a storehouse of historical Fisher subspaces during online training and matches the Fisher principal subspace of each incoming batch against this storehouse. The resulting subspace similarity serves as a unified geometric signal to adaptively determine whether the current batch should reuse an existing subspace, expand a related subspace, or be assigned to a new subspace. Thus, FiUni jointly models latent task detection and parameter-efficient adaptation within a single framework, enabling the model to dynamically balance knowledge sharing and task-specific isolation without relying on explicit task boundaries or IDs.

The main contributions of this work are summarized as follows:

• We reveal the intrinsic task geometry within Fisher/K-FAC principal subspaces. By utilizing only a small number of downstream samples to estimate the Fisher/K-FAC principal subspace, we depict meaningful taskgeometric structures, thereby providing an effective signal for measuring task similarity.

• We propose FiUni, a task-free continual parameter-efficient fine-tuning framework that unifies latent task shift detection and LoRA subspace construction via a single geometric signal. Furthermore, we design an adaptive decision mechanism to dynamically balance knowledge sharing and task isolation based on the matching degree between current data and historical Fisher subspaces.

• Experimental results demonstrate that FiUni can effectively infer the latent task affiliation of each incoming batch without relying on task boundaries or IDs, and achieve competitive performance compared to state-of-the-art task-aware continual learning baselines while requiring fewer trainable parameters.

## 2 Related Work

CL (Zhou et al., 2024; Wang et al., 2024) aims to enable models to sequentially obtain knowledge from continuous data streams or tasks while preserving performance on previously learned information. The primary challenge in this paradigm is catastrophic forgetting, i.e., adapting to new tasks tends to overwrite or disrupt parameter representations crucial for prior knowledge (French, 1999; McCloskey and Cohen, 1989). To mitigate this issue, existing research broadly categorizes CL approaches into regularization-based, replaybased, and dynamic architecture-based methods. Regularization-based techniques (Kirkpatrick et al., 2017; Zenke et al., 2017; Aljundi et al., 2018) typically preserve crucial parameters of previous tasks by restricting updates on new tasks. Replay-based strategies (Lopez-Paz and Ranzato, 2017; Shin et al., 2017) alleviate forgetting by saving a small number of old task samples or generating historical task samples during the training procedure of new tasks. Moreover, dynamic architecture-based schemes (Lee et al., 2017; Li et al., 2019) progressively expand the network topology to provide additional capacity for new knowledge. Recently, the development of LLMs has shifted the research focus of CL from traditional lightweight classifiers to massive pre-trained models (Shi et al., 2025; Wu et al., 2024).

This shift makes parameter efficiency a central concern. PEFT (Han et al., 2024; Wang et al., 2025) provides a feasible approach by freezing most pretrained parameters and introducing only a small number of trainable parameters for downstream adaptation. Typical approaches include Adapters (Houlsby et al., 2019), Prefix Tuning (Li and Liang, 2021), and LoRA (Hu et al., 2022). LoRA introduces low-rank update branches into weight matrices, and these updates can be merged back after training without additional inference latency. Building on this low-rank formulation, recent works have further attempted to impose pre-defined update subspaces on LoRA to reduce trainable parameters while maintaining adaptation performance (Kopiczko et al., 2024; Bałazy et al., 2025; Gao et al., 2024). For instance, FiLoRA (Han and Guo, 2026) utilizes the K-FAC (Martens and Grosse, 2015) approximation of the FIM from a small number of downstream samples, to extract principal subspaces as fixed low-rank bases, thereby guiding LoRA to perform efficient adaptation in parametersensitive directions. This indicates that the Fisher principal subspace can characterize the important update directions of the current task-specific parameters and provide a more geometrically meaningful subspace selection for efficient parameter fine-tuning.

In CL scenarios, PEFT provides a natural way to manage task-specific adaptation parameters, which can be independently stored, composed, or isolated. Building upon this property, existing continual PEFT methods typically enforce orthogonal constraints to separate the low-rank update spaces of different tasks, thereby reducing inter-task interference (Wang et al., 2023a; Qiao and Mahdavi, 2024). Furthermore, recent methods have moved beyond strict isolation and explicitly model shared and taskspecific components across tasks to improve transfer ability (Liao et al., 2026; Biswas et al., 2026). Although current continual PEFT schemes demonstrate strong performance in task-aware settings, their reliance on explicit task boundaries or task IDs for module management limits their applicability to task-free online scenarios. While recent tasklabel-free methods have explored model-internal signals to reduce the dependence on task identities (Du et al., 2024), they mainly focus on update modulation rather than explicitly coordinating latent task discovery with parameter-efficient subspace organization. Therefore, how to unify task-free latent structure identification and adaptive PEFT subspace management remains underexplored.

The intrinsic difference between FiUni and the above methods lies in its ability to determine the allocation of incoming knowledge at the batch level within a task-free setting. Furthermore, rather than passively mitigating inter-task interference through auxiliary regularization terms during the training process, FiUni actively confines parameter updates to low-rank subspaces determined by Fisher geometry during the subspace construction stage. Importantly, unlike methods that depend on explicit task boundaries to dictate knowledge sharing and isolation strategies, the mechanisms of knowledge reuse and task isolation in FiUni emerge naturally from the Fisher geometry induced by the pre-trained model on a minimal set of downstream samples, rather than being explicitly specified by humanannotated task partitions.

## 3 Methodology

## 3.1 Fisher Subspace Similarity

We first introduce how FiUni measures the geometric similarity between different downstream data batches through Fisher principal subspaces. Consider a linear layer in a pre-trained model with weight matrix $W \in \mathbb { R } ^ { m \times n }$ . For a data window B, the Fisher information matrix of this layer can be approximated by K-FAC (Martens and Grosse, 2015) as the Kronecker product of two covariance factors: $F _ { W } \approx { \mathcal { G } } \otimes A .$ , where

$$
\begin{array} { r } { \boldsymbol { \mathcal { A } } = \mathbb { E } _ { \boldsymbol { x } \sim \boldsymbol { B } } [ \boldsymbol { x } \boldsymbol { x } ^ { \top } ] , \mathcal { G } = \mathbb { E } _ { \boldsymbol { x } \sim \boldsymbol { B } } [ \delta \delta ^ { \top } ] . } \end{array}\tag{1}
$$

Here, x denotes the input activation of the layer, and δ denotes the back-propagated gradient with respect to the layer output. The activation covariance factor $\mathcal { A }$ captures the dominant directions in the input feature space, while the gradient covariance factor $\mathcal { G }$ captures the dominant sensitive directions in the output gradient space. Together, they characterize the Fisher geometry induced by the current data on the parameter updates of this layer.

To obtain a compact task representation, we perform eigendecomposition on A and G, and select the $\mathrm { \ t o p { - } } r _ { \mathrm { d e t } }$ eigenvectors corresponding to the largest eigenvalues, where $r _ { \mathrm { d e t } }$ denotes the detection rank used for Fisher subspace matching:

$$
V _ { \mathcal { B } } = \mathrm { T o p E i g } ( \mathcal { A } , r _ { \mathrm { d e t } } ) , U _ { \mathcal { B } } = \mathrm { T o p E i g } ( \mathcal { G } , r _ { \mathrm { d e t } } ) ,\tag{2}
$$

where $V _ { B } \in \mathbb { R } ^ { n \times r _ { \mathrm { d e t } } }$ and $U _ { B } \in \mathbb { R } ^ { m \times r _ { \mathrm { d e t } } }$ . We refer to $( U _ { B } , V _ { B } )$ as the Fisher principal subspace of the current data window.

For two data windows $B _ { i }$ and $B _ { j }$ , we measure their geometric similarity by the overlap between

their Fisher principal subspaces. Specifically, the gradient-side and activation-side subspace similarities are defined as

$$
s _ { U } ( B _ { i } , B _ { j } ) = \frac { 1 } { r _ { \mathrm { d e t } } } \left\| U _ { i } ^ { \top } U _ { j } \right\| _ { F } ^ { 2 } ,\tag{3}
$$

$$
s _ { V } ( B _ { i } , B _ { j } ) = \frac { 1 } { r _ { \mathrm { d e t } } } \left\| V _ { i } ^ { \top } V _ { j } \right\| _ { F } ^ { 2 } .\tag{4}
$$

The final Fisher subspace similarity is defined as

$$
s ( \beta _ { i } , \beta _ { j } ) = \frac { s _ { U } ( \beta _ { i } , \beta _ { j } ) + s _ { V } ( \beta _ { i } , \beta _ { j } ) } { 2 } .\tag{5}
$$

For a multi-layer model, we compute the above similarity over multiple selected target layers and modules, and average them as the overall task similarity:

$$
s _ { i j } = \frac { 1 } { | \mathcal { L } | } \sum _ { \ell \in \mathcal { L } } s ^ { ( \ell ) } ( B _ { i } , B _ { j } ) ,\tag{6}
$$

where $\mathcal { L }$ denotes the set of layers and modules used for Fisher subspace estimation. This similarity can be interpreted as the average principaldirection overlap between two Fisher geometries. A larger $s _ { i j }$ suggests that two data windows may come from the same or related latent task phases, while a smaller value indicates stronger orthogonality and may imply the emergence of a new latent phase. Therefore, this metric can be used to detect latent task changes in an online data stream by measuring task similarity between different data windows.

## 3.2 FiUni Framework Design

Based on the observation that Fisher principal subspaces reflect task similarity, we propose FiUni, a unified task-free continual learning framework for latent task detection and parameter-efficient adaptation, as shown in Figure 2. FiUni considers an online data stream where training data arrive sequentially in batches:

$$
\mathcal { D } = \{ B _ { 1 } , B _ { 2 } , \ldots , B _ { n } , \ldots \} ,\tag{7}
$$

During training, we maintain a historical Fisher subspace pool:

$$
\begin{array} { r } { { \cal { S } } = \{ ( U _ { k } , V _ { k } , R _ { k } ) \} _ { k = 1 } ^ { K } , } \end{array}\tag{8}
$$

where K denotes the number of discovered latent phases. For the k-th phase, $U _ { k }$ and $V _ { k }$ are frozen bases derived from Fisher/K-FAC principal subspaces, and $R _ { k }$ is the corresponding trainable core matrix. For a given layer, FiUni represents the effective model parameter as

![](images/80f3a646dc2d697ca9b9e3422a8ee9bcf129e328afd8b7d6316e4ef8eb99f5e4.jpg)  
Figure 2: Overview of FiUni. For each incoming batch in a task-free stream, FiUni estimates its Fisher principal subspace and matches it with the historical subspace pool. Based on the subspace similarity, FiUni adaptively decides to create a New subspace, Expand a related subspace, or Reuse an existing one.

$$
W _ { \mathrm { e f f } } = W _ { 0 } + \sum _ { k = 1 } ^ { K } U _ { k } R _ { k } V _ { k } ^ { \top } ,\tag{9}
$$

where $W _ { 0 }$ is the frozen pre-trained weight. Unlike standard LoRA, which directly learns a full low-rank decomposition, FiUni fixes the left and right update bases using Fisher principal subspaces and only trains the compact intermediate matrix $R _ { k }$ , thereby restricting parameter updates to Fishersensitive directions of the current data.

When a new data window $B _ { n }$ arrives, FiUni first estimates its Fisher principal subspace by extracting the top-r principal directions $( U _ { \mathrm { c u r } } , V _ { \mathrm { c u r } } )$ for subsequent LoRA subspace construction.

It then computes the similarity between the current subspace and all historical subspaces using the $\mathrm { t o p } { - } r _ { \mathrm { d e t } }$ principal directions, with $r _ { \mathrm { d e t } } \le r _ { \mathrm { : } }$ , and takes the maximum value as the matching degree between the current window and historical knowledge:

$$
s _ { n , \mathrm { m a x } } = \operatorname* { m a x } _ { k \in \{ 1 , \dots , K \} } s \left( ( U _ { \mathrm { c u r } } , V _ { \mathrm { c u r } } ) , ( U _ { k } , V _ { k } ) \right) .\tag{10}
$$

This maximum similarity reflects whether the current data can be explained by existing Fisher subspaces. A larger $s _ { n , \mathrm { m a x } }$ indicates that the current data is highly related to a historical phase, while a smaller value suggests that the current data may correspond to a new latent task phase.

Based on the matching degree between the current window and historical subspaces, FiUni adopts a three-state decision mechanism: REUSE, EX-PAND, NEW. This mechanism is controlled by two thresholds, $\tau _ { \mathrm { l o w } }$ and $\tau _ { \mathrm { h i g h } }$

When $s _ { n , \mathrm { m a x } } \geq \tau _ { \mathrm { h i g h } } .$ , FiUni regards the current data as highly matched with an existing phase, and thus reuses the corresponding LoRA subspace by continuing to update its core matrix $R _ { k }$ . When $\tau _ { \mathrm { l o w } } ~ \le ~ s _ { n , \mathrm { m a x } } ~ < ~ \tau _ { \mathrm { h i g h } }$ , FiUni treats the current data as related to an existing phase with moderate drift. It therefore expands the most similar FiLoRA subspace by removing the overlapping components of $( U _ { \mathrm { c u r } } , V _ { \mathrm { c u r } } )$ , orthogonalizing the remaining directions, and supplementing them with random directions to increase the subspace rank by $\Delta r$ . When $s _ { n , \mathrm { m a x } } < \tau _ { \mathrm { l o w } }$ , FiUni considers the current data geometrically distinct from historical phases and allocates a new LoRA subspace using the current Fisher principal subspace:

$$
U _ { K + 1 } = U _ { \mathrm { c u r } } , V _ { K + 1 } = V _ { \mathrm { c u r } } ,\tag{11}
$$

and initializes a new trainable core matrix $R _ { K + 1 }$ The new tuple $\left( U _ { K + 1 } , V _ { K + 1 } , R _ { K + 1 } \right)$ is then added to the historical subspace pool.

To improve online detection stability and reduce false triggers caused by sample noise, FiUni adopts a two-window confirmation mechanism. For two consecutive batches/windows, FiUni performs the corresponding REUSE or NEW operation only when both satisfy the same decision condition, making online task-shift detection more robust.

## 3.3 Geometric Reuse and Isolation in FiUni

Unlike methods that rely on given task boundaries to actively isolate adaptation modules and predefine shared or isolated parameters, knowledge sharing and task isolation in FiUni both arise from Fisher geometry itself, and parameter updates are restricted to the corresponding geometric subspaces.

Taking the output-gradient Fisher subspace U as an example, for the current historical subspace set S, the common overlapping component among these subspaces corresponds to reusable shared directions:

$$
U _ { S } ^ { \mathrm { s h a r e } } = \bigcap _ { k = 1 } ^ { K } U _ { k } .\tag{12}
$$

For any subspace $U _ { i }$ and the remaining subspace pool $U _ { \neg i } ,$ the orthogonal component corresponds to more task-specific isolated directions for the latent task:

$$
U _ { i } ^ { \mathrm { i s o } } = ( I - U _ { \neg i } U _ { \neg i } ^ { \top } ) U _ { i } .\tag{13}
$$

The overlapping components support knowledge reuse across related phases, while the orthogonal residuals provide isolation for new or drifting knowledge. Therefore, FiUni does not simply switch modules after detecting task boundaries. Instead, it performs geometric matching at the batch level and adaptively balances reuse, expansion, and isolation throughout the online data stream.

In practice, estimating Fisher statistics and performing eigendecomposition for all parameter blocks would introduce considerable computational overhead, while our ablation experiments show that the discriminative ability of Fisher subspace similarity is not very sensitive to the specific choice of layers or modules, but is mainly affected by the number of samples used for estimation and the $r _ { \mathrm { d e t } }$ . Therefore, FiUni performs subspace matching only on a small number of selected layers. This enables FiUni to perform efficient online detection with limited overhead, making it practical for continual parameter-efficient fine-tuning of large language models.

## 4 Experiment

## 4.1 Benchmarks and Data Setup

We evaluate FiUni on three CL benchmarks. Unlike conventional task-aware continual learning, we focus on the more challenging task-free online setting, where training data arrive sequentially as a batch stream and the model has no access to explicit task boundaries or task IDs during training.

Standard CL Benchmark (SC) consists of four text classification datasets: AG News, Amazon Reviews, DBpedia, and Yahoo Answers (Zhang et al., 2015). Following prior work (Wang et al., 2023a), we organize them into three task orders, denoted as Orders 1–3, to evaluate continual adaptation over short classification sequences.

Long Sequence Benchmark (LS) extends SC to a longer and more diverse stream with 15 tasks (Razdaibiedina et al., 2023). We sample a fixed number of training examples for each task and construct three task orders, denoted as Orders 4–6. Compared with SC, LS contains more heterogeneous task types and more complex task transitions.

TRACE Benchmark (Wang et al., 2023b) contains diverse LLM tasks, including multiple-choice QA, multilingual understanding, code generation, and mathematical reasoning, denoted as Order 7.

## 4.2 Task Similarity

We first examine the core observation of this work: Fisher subspace similarity s, estimated from a small number of downstream samples, can reflect geometric similarity between tasks. Specifically, we use T5-base on the LS benchmark and sample 32 examples from each task to compute the pairwise Fisher subspace similarity s defined in Section 3.1. We then analyze whether the resulting similarity structure is consistent with task semantics or task families.

As shown in Figure 3 (a), the Fisher subspace similarity matrix exhibits clear task structure. Data from identical or related tasks usually show higher subspace similarity, while unrelated tasks tend to be more orthogonal. For example, MNLI, CB, and RTE, which are all related to language inference, show relatively high similarity. Among sentimentrelated tasks, IMDB, SST-2, Yelp, and Amazon are all sentiment-oriented, but Yelp and Amazon exhibit stronger similarity, likely because both are rating-style review tasks.

We further evaluate the within-task consistency of this signal through repeated sampling, as shown in Figure 3 (b). For each task, we independently sample two groups of examples and estimate their Fisher principal subspaces separately. The results show that two subsets from the same task usually produce highly similar Fisher subspaces, indicating that samples within the same task tend to share consistent Fisher-geometric characteristics. Finally, Figure 3 (c) compares within-task self-similarity and task-to-other similarity. Most tasks exhibit substantially higher self-similarity than cross-task similarity. In particular, similarities between identical or related tasks are mostly concentrated in the range of 0.6–0.9, whereas low-similarity or unrelated task pairs mostly fall below 0.6. This suggests that appropriate thresholds can separate similar tasks from unrelated ones, supporting the feasibility of Fisher subspace similarity as a batch-level signal for latent task identification.

![](images/f391dbb5e15bd8bb3b2b70d1682eb37cdc9e6ea6f31a4e441c87ec556a99a596.jpg)  
(a)

![](images/0dfbbf286a6c1e559507f243afaed08063982dc2e60822949326a9f8652ac778.jpg)  
(b)

![](images/091f3be5c55a4eca24f8d0a12bc391df670aaaa7129b9eae578e2b106fb2b895.jpg)  
(c)

![](images/f4442bc170cf35823b3a09bddf02b9404604dc19f001670571f8c83e1efe98d6.jpg)  
(d)  
Figure 3: Empirical analysis of Fisher subspace similarity and FiUni decisions on T5-base. Panels (a)–(c) estimate Fisher subspaces using 32 samples per task over all linear layers with $r _ { d e t } = 8 \colon$ (a) shows cross-task Fisher subspace similarity, (b) shows within-task self-similarity from two independent samplings, and (c) compares within-task similarity with task-to-other similarity. Panel (d) shows the batch-level decisions produced by FiUni on a task-free stream using 32-shot Fisher estimation, selected Q/K/V/O layers, and thresholds $\tau _ { \mathrm { l o w } } = 0 . 6 5$ and $\tau _ { \mathrm { h i g h } } = 0 . 8 5$

## 4.3 Task Detection

After verifying that Fisher subspaces reflect task similarity, we further analyze the batch-level decision process of FiUni in an online task-free stream. We take up to 1,000 samples from each task in the

LS benchmark and concatenate them into a sequential training stream. During detection, the model has no access to task IDs or task boundaries, and makes decisions batch by batch.

Figure 3 (d) shows the batch-level decision trajectory of FiUni in the task-free stream. When the stream remains within the same task or moves to a related task, FiUni usually tends to reuse an existing subspace or perform a lightweight expansion. When the stream enters a more distinct stage, FiUni can trigger the allocation of a new subspace. For example, within the SST-2 stage, FiUni expands the current subspace when the internal data variation becomes larger. In contrast, when the stream transitions from CB to WiC or from WiC to COPA, FiUni creates a new subspace.

We also observe that FiUni’s decisions do not always coincide with human-defined task boundaries. For some related tasks, FiUni may reuse or expand an existing subspace instead of immediately creating a new one. For instance, when the stream moves from MNLI to CB, FiUni expands the subspace associated with MNLI rather than allocating a new subspace. During the Amazon stage,

<table><tr><td rowspan="2">Methods</td><td colspan="4">Standard CL Benchmark (SC)</td><td rowspan="2"></td><td colspan="4">Long Sequence Benchmark (LS)</td><td rowspan="2">TRACE Order 7</td></tr><tr><td>Order 1</td><td>Order 2</td><td>Order 3</td><td>Avg</td><td>Order 4</td><td>Order 5</td><td>Order 6</td><td>Avg</td></tr><tr><td rowspan="12">T5SLaarge</td><td></td><td>39.5</td><td>31.9</td><td>46.6</td><td>39.3</td><td>4.9</td><td>3.5</td><td>4.2</td><td>4.2</td><td>12.1</td></tr><tr><td>SeqLoRA† SeqLoRAReplay</td><td>74.0</td><td>73.1</td><td>73.0</td><td>73.3</td><td>74.2</td><td>72.7</td><td>73.9</td><td>73.6</td><td>34.0</td></tr><tr><td>IncLoRA</td><td>63.4</td><td>62.2</td><td>65.1</td><td>63.6</td><td>63.0</td><td>57.9</td><td>60.4</td><td>60.5</td><td></td></tr><tr><td>EWC (Kirkpatrick et al., 2017)</td><td>46.3</td><td>45.3</td><td>52.1</td><td>47.9</td><td>44.9</td><td>44.0</td><td>45.4</td><td>44.8</td><td></td></tr><tr><td>L2P (Wang et al., 2022)</td><td>60.3</td><td>61.7</td><td>61.1</td><td>60.7</td><td>57.5</td><td>53.8</td><td>56.9</td><td>56.1</td><td></td></tr><tr><td>LFPT5 (Qin and Joty, 2022)</td><td>67.6</td><td>72.6</td><td>77.9</td><td>72.7</td><td>70.4</td><td>68.2</td><td>69.1</td><td>69.2</td><td></td></tr><tr><td>O-LoRA (Wang et al., 2023a)</td><td>73.2</td><td>72.4</td><td>70.4</td><td>72.0</td><td>69.9</td><td>68.5</td><td>65.3</td><td>67.9</td><td>23.1</td></tr><tr><td>+ MIGU (Du et al., 2024)</td><td>73.5</td><td>71.4</td><td>70.0</td><td>71.6</td><td>65.4</td><td>65.2</td><td>65.2</td><td>65.3</td><td></td></tr><tr><td>SpaRTA (Liao et al., 2026)</td><td>73.7</td><td>70.5</td><td>73.8</td><td>72.7</td><td>71.5</td><td>70.5</td><td>68.0</td><td>70.0</td><td>16.7</td></tr><tr><td>+ Replay</td><td>77.0</td><td>75.6</td><td>75.2</td><td>75.9</td><td>75.6</td><td>73.2</td><td>74.1</td><td>74.3</td><td>36.5</td></tr><tr><td>FiUni (ours)†</td><td>75.6</td><td>76.5</td><td>73.0</td><td>75.0</td><td>67.1</td><td>67.0</td><td>70.84</td><td>68.3</td><td>32.9</td></tr><tr><td>SeqLoRA†</td><td>75.88</td><td>74.40</td><td>74.35</td><td>74.86</td><td>67.81</td><td>65.93</td><td>63.80</td><td>65.85</td><td>28.23</td></tr><tr><td>L--8B</td><td>O-LoRA (Wang et al., 2023a) 69.39</td><td>67.58</td><td>71.44</td><td>69.46</td><td>69.54</td><td>64.42</td><td>66.50</td><td>66.82</td><td>28.45</td></tr><tr><td>SpaRTA (Liao et al., 2026)</td><td>76.10</td><td>75.69</td><td>75.52</td><td>75.77</td><td>71.34</td><td>70.77</td><td>72.95</td><td>71.69</td><td>31.33</td></tr><tr><td>+ Replay</td><td>77.56</td><td>77.01</td><td>76.03</td><td>76.83</td><td>73.10</td><td>73.05</td><td>74.17</td><td>73.44</td><td>34.16</td></tr><tr><td>SeqLoRAReplay</td><td>77.02</td><td>76.53</td><td>76.19</td><td>76.58</td><td>72.03</td><td>72.96</td><td>75.38</td><td>73.46</td><td>33.02</td></tr><tr><td>ELLA (Biswas et al., 2026)</td><td>77.80</td><td>77.20</td><td>77.70</td><td>77.57</td><td>72.87</td><td>72.84</td><td>76.82</td><td>74.18</td><td>33.29</td></tr><tr><td>FiUni (ours)†</td><td>78.00</td><td>78.20</td><td>77.49</td><td>77.91</td><td>75.74</td><td>74.74</td><td>77.50</td><td>75.99</td><td>55.21</td></tr></table>

Table 1: Comparison results on Standard CL Benchmark (SC), Long Sequence Benchmark (LS), and TRACE. FiUni results are averaged over three random seeds. Methods marked with <sup>†</sup> do not rely on task boundaries during training. Gray-colored methods use replay data, generated pseudo-data, or external memory during continual learning. Bold indicates the best result, and underline indicates the second-best result among non-replay methods.

FiUni directly reuses or continues training on the subspace previously triggered by Yelp. This is consistent with our definition of latent tasks. Such behavior allows FiUni to avoid mechanically allocating a separate subspace for every dataset-level task, reducing redundant parameter growth while promoting knowledge reuse across related tasks.

## 4.4 Overall Results

To verify the effectiveness of FiUni in detecting latent tasks, we compare it with multiple task-aware continual learning methods. Since task boundaries are unavailable in our stream setting, we only consider overall accuracy (Chaudhry et al., 2018) $\begin{array} { r } { \operatorname { O A } _ { T } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } a _ { T , t } } \end{array}$ , where ${ a } _ { T , t }$ denotes the performance on task t after training on the complete stream. We do not consider FWT (Lopez-Paz and Ranzato, 2017) or BWT (Ke and Liu, 2023), as these metrics rely on task boundaries to measure forgetting and knowledge transfer.

As shown in Table 1, FiUni achieves strong results across different benchmarks and backbone models, and outperforms strict orthogonal isolation methods such as O-LoRA in all settings. This demonstrates the effectiveness of using fixed Fisher subspaces for task sharing and isolation. We observe that FiUni brings relatively limited improvement on LS with T5-Large. A possible reason is that T5-Large has a lower hidden dimension, and to keep the FiLoRA parameter scale competitive with standard LoRA, the available rank capacity can be consumed earlier in the long task stream, thereby restricting the expansion space for later phases.

In contrast, LLaMA-3.1-8B provides a larger representation space and richer subspace capacity, where FiUni achieves more consistent gains across the three benchmarks. Overall, without using task boundaries or task IDs, FiUni still matches or surpasses state-of-the-art task-aware continual PEFT methods. This shows that latent task differences can be effectively captured by the Fisher geometry of downstream data on the pre-trained model, enabling effective continual adaptation even without explicit task information.

## 5 Conclusion

We propose FiUni, a Fisher geometry-based framework for task-free continual parameter-efficient fine-tuning in online data streams. FiUni unifies batch-level latent task identification and LoRA subspace construction through Fisher principal subspaces, enabling adaptive reuse, expansion, and allocation of LoRA subspaces without explicit task boundaries or task IDs. Experiments demonstrate that Fisher subspace similarity captures meaningful and stable task geometry, and that FiUni achieves competitive performance against strong task-aware baselines with substantially fewer trainable parameters. Our results highlight Fisher geometry as an effective and practical signal for scalable task-free continual adaptation of LLMs.

## 6 Limitations

FiUni still has several limitations. First, the detection stage estimates Fisher/K-FAC statistics on the pre-trained model, which introduces an extra forward and backward pass. Although this cost is moderate because detection uses only a few samples and selected layers, it may still be noticeable in large-scale online training. A promising future direction is to further study the relationship between task streams and the Fisher factors estimated on the pre-trained model combined with already learned LoRA modules, so that detection can potentially be performed with fewer additional passes or reused statistics. Second, computing the activation and gradient covariance factors also introduces additional memory overhead, since activations and gradients need to be cached for selected modules. Although FiUni avoids constructing the full Fisher matrix, more memory-efficient estimation strategies, such as streaming covariance updates, lowprecision statistics, or randomized eigendecomposition, could further improve scalability. Finally, due to computational and memory constraints, our experiments are limited to models up to 8B parameters. We therefore have not fully assessed the behavior and overhead of FiUni on larger-scale models such as 70B LLMs. Further scaling experiments would be needed to verify its efficiency and robustness in such settings.

## 7 Ethical Considerations

FiUni is a method for parameter-efficient continual adaptation and does not introduce new data collection, human annotation, or user-facing decision procedures. All experiments are conducted on publicly available benchmarks following standard research protocols. Since FiUni does not require storing previous task samples for replay, it does not add extra data-retention requirements beyond the training data used in each benchmark. As with standard fine-tuning methods for large language models, practical deployment should follow the data usage policies and safety requirements of the target application.

## 8 AI writing statement

This paper utilized AI assistance for language polishing of the manuscript, including vocabulary correction and spell checking.

## References

Rahaf Aljundi, Francesca Babiloni, Mohamed Elhoseiny, Marcus Rohrbach, and Tinne Tuytelaars. 2018. Memory aware synapses: Learning what (not) to forget. In Proceedings of the European Conference on Computer Vision.

Rahaf Aljundi, Klaas Kelchtermans, and Tinne Tuytelaars. 2019. Task-free continual learning. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11254–11263.

Klaudia Bałazy, Mohammadreza Banaei, Karl Aberer, and Jacek Tabor. 2025. LoRA-XS: Low-rank adaptation with extremely small number of parameters. Computing Research Repository, arXiv:2405.17604. Version 3.

Shristi Das Biswas, Yue Zhang, Anwesan Pal, Radhika Bhargava, and Kaushik Roy. 2026. ELLA: Efficient lifelong learning for adapters in large language models. In Proceedings of the 19th Conference of the European Chapter ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 1907–1924, Rabat, Morocco.

Arslan Chaudhry, Puneet K. Dokania, Thalaiyasingam Ajanthan, and Philip H. S. Torr. 2018. Riemannian walk for incremental learning: Understanding forgetting and intransigence. In Proceedings ofthe European Conference on Computer Vision (ECCV), pages 556–572, Munich, Germany. Springer.

Wenyu Du, Shuang Cheng, Tongxu Luo, Zihan Qiu, Zeyu Huang, Ka Chun Cheung, Reynold Cheng, and Jie Fu. 2024. Unlocking continual learning abilities in language models. In Findings ofthe Association for Computational Linguistics: EMNLP 2024, pages 6503–6522, Miami, Florida, USA.

Robert M. French. 1999. Catastrophic forgetting in connectionist networks. Trends in Cognitive Sciences, 3(4):128–135.

Ziqi Gao, Qichao Wang, Aochuan Chen, Zijing Liu, Bingzhe Wu, Liang Chen, and Jia Li. 2024. Parameter-efficient fine-tuning with discrete fourier transform. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 14884–14901.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, and 1 others. 2024. The Llama 3 herd of models. Computing Research Repository, arXiv:2407.21783.

Dezheng Han and Shuaishuai Guo. 2026. FiLoRA: Parameter-efficient fine-tuning with fisher information-guided low-rank adaptation. IEEE Signal Processing Letters, 33:604–608.

Zeyu Han, Chao Gao, Jinyang Liu, Jeff Zhang, and Sai Qian Zhang. 2024. Parameter-efficient finetuning for large models: A comprehensive survey. Computing Research Repository, arXiv:2403.14608.

Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin de Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. 2019. Parameter-efficient transfer learning for NLP. In Proceedings of the 36th International Conference on Machine Learning, pages 2790–2799, Long Beach, USA.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-rank adaptation of large language models. In Proceedings ofthe 10th International Conference on Learning Representations, pages 12513–12525, Virtual.

Shun ichi Amari. 1998. Natural gradient works efficiently in learning. Neural Computation, 10(2):251– 276.

Zixuan Ke and Bing Liu. 2023. Continual learning of natural language processing tasks: A survey. Computing Research Repository, arXiv:2211.12701.

James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A. Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, Demis Hassabis, Claudia Clopath, Dharshan Kumaran, and Raia Hadsell. 2017. Overcoming catastrophic forgetting in neural networks. Proceedings ofthe National Academy of Sciences, 114(13):3521–3526.

Dawid J. Kopiczko, Tijmen Blankevoort, and Yuki M. Asano. 2024. VeRA: Vector-based random matrix adaptation. In Proceedings of the 12th International Conference on Learning Representations, pages 37947–37967, Vienna, Austria.

Jeongtae Lee, Jaehong Yoon, Eunho Yang, and Sung Ju Hwang. 2017. Lifelong learning with dynamically expandable networks. Computing Research Repository, arXiv:1708.01547.

Xiang Lisa Li and Percy Liang. 2021. Prefix-tuning: Optimizing continuous prompts for generation. In Proceedings ofthe 59th Annual Meeting ofthe Associationfor Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 4582– 4597, Virtual.

Xilai Li, Yingbo Zhou, Tianfu Wu, Richard Socher, and Caiming Xiong. 2019. Learn to grow: A continual structure learning framework for overcoming catastrophic forgetting. In Proceedings ofthe 36th International Conference on Machine Learning, volume 97

of Proceedings of Machine Learning Research, pages 3925–3934.

Huanxuan Liao, Shizhu He, Yupu Hao, Jun Zhao, and Kang Liu. 2026. Spectral disentanglement: Rankaware task adaptation for rehearsal-free continual learning in LLMs. In Proceedings of the 64th Annual Meeting ofthe Associationfor Computational Linguistics, San Diego, California.

David Lopez-Paz and Marc’Aurelio Ranzato. 2017. Gradient episodic memory for continual learning. In Advances in Neural Information Processing Systems, volume 30.

James Martens and Roger Grosse. 2015. Optimizing neural networks with kronecker-factored approximate curvature. In Proceedings of the 32nd International Conference on Machine Learning, pages 2408–2417, Lille, France.

Michael McCloskey and Neal J. Cohen. 1989. Catastrophic interference in connectionist networks: The sequential learning problem. In Gordon H. Bower, editor, Psychology of Learning and Motivation, volume 24, pages 109–165. Academic Press.

Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, and 2 others. 2019. Pytorch: An imperative style, high-performance deep learning library. In Advances in Neural Information Processing Systems, volume 32.

Fuli Qiao and Mehrdad Mahdavi. 2024. Learn more, but bother less: Parameter efficient continual learning. In Advances in Neural Information Processing Systems, volume 37, pages 97476–97498. Curran Associates, Inc.

Chengwei Qin and Shafiq Joty. 2022. LFPT5: A unified framework for lifelong few-shot language learning based on prompt tuning of T5. In Proceedings of the 10th International Conference on Learning Representations.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal ofMachine Learning Research, 21(140):1–67.

Anastasia Razdaibiedina, Yuning Mao, Rui Hou, Madian Khabsa, Mike Lewis, and Amjad Almahairi. 2023. Progressive prompts: Continual learning for language models. In Proceedings of the 11th International Conference on Learning Representations.

Haizhou Shi, Zihao Xu, Hengyi Wang, Weiyi Qin, Wenyuan Wang, Yibin Wang, Zifeng Wang, Sayna Ebrahimi, and Hao Wang. 2025. Continual learning

of large language models: A comprehensive survey. ACM Computing Surveys, 58(5).

Hanul Shin, Jung Kwon Lee, Jaehong Kim, and Jiwon Kim. 2017. Continual learning with deep generative replay. In Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc.

Alex Wang, Yada Pruksachatkun, Nikita Nangia, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman. 2019a. SuperGLUE: A stickier benchmark for general-purpose language understanding systems. In Advances in Neural Information Processing Systems, volume 32, Vancouver, Canada.

Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman. 2019b. GLUE: A multi-task benchmark and analysis platform for natural language understanding. In Proceedings of the 7th International Conference on Learning Representations, pages 1786–1805, New Orleans, USA.

Liyuan Wang, Xingxing Zhang, Hang Su, and Jun Zhu. 2024. A comprehensive survey of continual learning: Theory, method and application. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(8):5362–5383.

Luping Wang, Sheng Chen, Linnan Jiang, Shu Pan, Runze Cai, Sen Yang, and Fei Yang. 2025. Parameter-efficient fine-tuning in large models: A survey of methodologies. Computing Research Repository, arXiv:2410.19878.

Xiao Wang, Tianze Chen, Qiming Ge, Han Xia, Rong Bao, Rui Zheng, Qi Zhang, Tao Gui, and Xuanjing Huang. 2023a. Orthogonal subspace learning for language model continual learning. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 10658–10671, Singapore.

Xiao Wang, Yuansen Zhang, Tianze Chen, Songyang Gao, Senjie Jin, Xianjun Yang, Zhiheng Xi, Rui Zheng, Yicheng Zou, Tao Gui, Qi Zhang, and Xuanjing Huang. 2023b. TRACE: A comprehensive benchmark for continual learning in large language models. Computing Research Repository, arXiv:2310.06762.

Zifeng Wang, Zizhao Zhang, Chen-Yu Lee, Han Zhang, Ruoxi Sun, and Xiaoqi Ren. 2022. Learning to prompt for continual learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 139–149, New Orleans, LA, USA.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Remi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, and 3 others. 2020. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 Conference on Empirical

Methods in Natural Language Processing: System Demonstrations, pages 38–45.

Tongtong Wu, Linhao Luo, Yuan-Fang Li, Shirui Pan, Thuy-Trang Vu, and Gholamreza Haffari. 2024. Continual learning for large language models: A survey. Computing Research Repository, arXiv:2402.01364.

An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, and 23 others. 2025. Qwen2.5 technical report. Computing Research Repository, arXiv:2412.15115.

Friedemann Zenke, Ben Poole, and Surya Ganguli. 2017. Continual learning through synaptic intelligence. In Proceedings of the 34th International Conference on Machine Learning, volume 70, pages 3987–3995.

Xiang Zhang, Junbo Zhao, and Yann LeCun. 2015. Character-level convolutional networks for text classification. In Advances in Neural Information Processing Systems, volume 28. Curran Associates, Inc.

Da-Wei Zhou, Qi-Wei Wang, Zhi-Hong Qi, Han-Jia Ye, De-Chuan Zhan, and Ziwei Liu. 2024. Classincremental learning: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(12):9851–9873.

## Appendix

This appendix provides additional details and analyses to support the main paper. We organize the appendix into three parts. First, we provide a theoretical justification for why Fisher information can be used to characterize task-dependent geometry, and further explain why the K-FAC principal subspaces of the activation and gradient covariance factors can reflect task similarity. Second, we present detailed experimental settings, including benchmark construction, implementation details, hyperparameters, and evaluation protocols. Third, we report additional experimental results and ablation studies to further analyze the behavior of FiUni under different configurations.

## A Theoretical Justification

In this section, we provide a theoretical justification for why Fisher information can be used to identify different latent tasks, and why its K-FAC approximation allows us to use the principal subspaces of the activation and gradient covariance factors as a practical task-similarity signal. We first discuss the full Fisher information matrix, and then derive the connection under the K-FAC approximation.

## A.1 Full Fisher Information as Task Geometry

Consider a pre-trained model with parameters $\theta _ { 0 }$ and a downstream task distribution $\mathcal { D } _ { t }$ . For a sample $\boldsymbol z = ( x , y ) \sim \mathcal D _ { t }$ , let $\ell ( z ; \theta _ { 0 } )$ denote the training loss, and define the task-induced gradient as

$$
\begin{array} { r } { g _ { t } ( z ) = \nabla _ { \theta } \ell ( z ; \theta ) \big | _ { \theta = \theta _ { 0 } } . } \end{array}\tag{14}
$$

The Fisher information matrix (ichi Amari, 1998) induced by task $\mathcal { D } _ { t }$ is

$$
F _ { t } = \mathbb { E } _ { z \sim \mathcal { D } _ { t } } \left[ g _ { t } ( z ) g _ { t } ( z ) ^ { \top } \right] .\tag{15}
$$

This matrix captures the second-order geometry of the loss around the pre-trained model. For a small parameter perturbation $\Delta \theta ,$ , the local change of the task loss or predictive distribution can be approximated by a quadratic form:

$$
\Delta \mathcal { L } _ { t } \approx \frac { 1 } { 2 } \Delta \theta ^ { \top } F _ { t } \Delta \theta .\tag{16}
$$

Therefore, directions with large Fisher eigenvalues correspond to parameter directions to which task $\mathcal { D } _ { t }$ is most sensitive. In other words, the dominant eigenspace of $F _ { t }$ describes the main update directions required by this task on the pre-trained model.

Let

$$
Q _ { t } = \mathrm { T o p E i g } ( F _ { t } , r )\tag{17}
$$

be the subspace spanned by the top-r eigenvectors of $F _ { t }$ . For two tasks $\mathcal { D } _ { a }$ and $\mathcal { D } _ { b }$ , their Fisher principal subspaces $Q _ { a }$ and $Q _ { b }$ reflect whether the two tasks require similar parameter update directions. A natural similarity measure is

$$
s _ { F } ( a , b ) = \frac { 1 } { r } \left. Q _ { a } ^ { \top } Q _ { b } \right. _ { F } ^ { 2 } .\tag{18}
$$

If the two tasks induce similar gradient distributions on the pre-trained model, then their Fisher matrices are close, and their principal subspaces are also close. If the two tasks require very different update directions, their Fisher principal subspaces tend to have smaller overlap.

This can be formalized using standard eigenspace perturbation results. Suppose the top-r eigenspace of $F _ { a }$ has eigengap $\gamma _ { a } .$ , and $F _ { b }$ is a perturbation of $F _ { a }$ . Then the distance between their principal subspaces satisfies

$$
\| \sin \Theta ( Q _ { a } , Q _ { b } ) \| _ { 2 } \leq C \frac { \| F _ { a } - F _ { b } \| _ { 2 } } { \gamma _ { a } } ,\tag{19}
$$

where $C$ is a constant and $\Theta ( Q _ { a } , Q _ { b } )$ denotes the principal angles between the two subspaces. This indicates that if two tasks induce similar Fisher matrices, their principal subspaces remain aligned; if their Fisher matrices differ significantly, their principal subspaces become less aligned or more orthogonal.

Thus, the full Fisher information matrix provides a task-dependent geometric representation: its leading eigenspace captures the parameter-sensitive directions of a task, and the overlap between two Fisher principal subspaces reflects the similarity between the corresponding tasks.

## A.2 K-FAC Approximation of the Layer-wise Fisher Matrix

Although the full Fisher information matrix provides a principled task geometry, it is prohibitively expensive to compute and store for large language models. Therefore, we use the K-FAC (Martens and Grosse, 2015) approximation to obtain a tractable layer-wise representation.

Consider a linear layer with weight matrix

$$
W \in \mathbb { R } ^ { m \times n } .\tag{20}
$$

Let $x \in \mathbb { R } ^ { n }$ be the input activation of this layer, and let $\delta \in \mathbb { R } ^ { m }$ be the back-propagated gradient with respect to the layer output. The gradient of the loss with respect to W is

$$
\nabla _ { W } \ell = \delta x ^ { \top } .\tag{21}
$$

After vectorization, this gradient can be written as

$$
\operatorname { v e c } ( \nabla _ { W } \ell ) = x \otimes \delta .\tag{22}
$$

The layer-wise Fisher matrix is then

$$
F _ { W } = \mathbb { E } \left[ \mathrm { v e c } ( \nabla _ { W } \ell ) \mathrm { v e c } ( \nabla _ { W } \ell ) ^ { \top } \right] .\tag{23}
$$

K-FAC approximates this Fisher matrix by factorizing the second-order statistics of activations and output gradients:

$$
F _ { W } \approx { \mathcal { G } } \otimes { \mathcal { A } } ,\tag{24}
$$

where

$$
\begin{array} { r } { \boldsymbol { \mathcal { A } } = \mathbb { E } \left[ \boldsymbol { x } \boldsymbol { x } ^ { \top } \right] , } \\ { \boldsymbol { \mathcal { G } } = \mathbb { E } \left[ \boldsymbol { \delta } \boldsymbol { \delta } ^ { \top } \right] . } \end{array}\tag{25}
$$

Here, A captures the dominant directions in the input activation space, while G captures the dominant directions in the output-gradient space. Together, they approximate the layer-wise Fisher geometry induced by the current data.

## A.3 Why the Principal Subspaces of A and G Reflect Task Similarity

We now explain why the top eigenvectors of $\mathcal { A }$ and G can be used to reflect task similarity under the K-FAC approximation.

Let

$$
\begin{array} { l } { { A v _ { i } = \alpha _ { i } v _ { i } , } } \\ { { g _ { u _ { j } } = \beta _ { j } u _ { j } . } } \end{array}\tag{26}
$$

Then the Kronecker product $\mathcal { G } \otimes \mathcal { A }$ has eigenvectors of the form $u _ { j } \otimes v _ { i }$ , with eigenvalues $\beta _ { j } \alpha _ { i } \mathbf { : }$

$$
( \mathcal { G } \otimes \mathcal { A } ) ( u _ { j } \otimes v _ { i } ) = \beta _ { j } \alpha _ { i } ( u _ { j } \otimes v _ { i } ) .\tag{27}
$$

Therefore, the dominant eigenspace of the K-FAC Fisher matrix is determined by the dominant eigenspaces of $\mathcal { G }$ and A. This means that the leading eigenvectors of the gradient covariance factor and the activation covariance factor provide a compact approximation to the principal geometry of the layer-wise Fisher matrix.

For a data window or task B, we define

$$
\begin{array} { r } { U _ { B } = \mathrm { T o p E i g } ( \mathcal { G } _ { B } , r ) , } \\ { V _ { B } = \mathrm { T o p E i g } ( A _ { B } , r ) . } \end{array}\tag{28}
$$

Here, $U _ { B }$ represents the dominant output-gradient directions, and $V _ { B }$ represents the dominant inputactivation directions. Since the dominant directions of $\mathcal { G } \otimes \mathcal { A }$ are constructed from combinations of $U _ { B }$ and $V _ { B }$ , the pair $( U _ { B } , V _ { B } )$ can serve as a tractable representation of the Fisher principal geometry.

Now consider two data distributions or tasks $\mathcal { D } _ { a }$ and $\mathcal { D } _ { b }$ . If they are similar, they tend to produce similar activation patterns and similar outputgradient patterns on the same pre-trained model. This implies that

$$
\begin{array} { r } { A _ { a } \approx A _ { b } , } \end{array}\tag{29}
$$

and

$$
{ \mathcal { G } } _ { a } \approx { \mathcal { G } } _ { b } .\tag{30}
$$

Under eigengap conditions, the top eigenspaces of these matrices will also be close. Thus, $V _ { a }$ will be close to $V _ { b } .$ , and $U _ { a }$ will be close to $U _ { b }$ . Conversely, if two tasks induce very different activation or gradient covariance structures, their top eigenspaces will have smaller overlap.

This provides the theoretical basis for using the principal subspaces of A and $\mathcal { G }$ to compare tasks. Instead of explicitly constructing the full Fisher eigenspace, FiUni measures the overlap between the two K-FAC factors:

$$
s _ { U } ( B _ { i } , B _ { j } ) = \frac { 1 } { r _ { \mathrm { d e t } } } \left\| U _ { i } ^ { \top } U _ { j } \right\| _ { F } ^ { 2 } ,\tag{31}
$$

$$
s _ { V } ( B _ { i } , B _ { j } ) = \frac { 1 } { r _ { \mathrm { d e t } } } \left\| V _ { i } ^ { \top } V _ { j } \right\| _ { F } ^ { 2 } .\tag{32}
$$

The final similarity is

$$
s ( \ B _ { i } , B _ { j } ) = \frac { s _ { U } ( B _ { i } , B _ { j } ) + s _ { V } ( B _ { i } , B _ { j } ) } { 2 } .\tag{33}
$$

This similarity is a surrogate for the overlap between the dominant eigenspaces of the K-FAC Fisher matrices. A high value means that the two data windows share similar activation and gradient principal directions, and therefore induce similar Fisher geometry. A low value means that their sensitive update directions are more orthogonal, suggesting that they correspond to different latent tasks or learning phases.

## A.4 Implication for Task-free Continual Learning

The above analysis shows that Fisher information contains task-dependent geometry at two levels. At the full-matrix level, the Fisher principal subspace captures the dominant parameter directions to which a task is most sensitive. Under the K-FAC approximation, this geometry is represented by the principal subspaces of the activation covariance factor A and the gradient covariance factor G.

This explains why the Fisher/K-FAC principal subspaces can be used for latent task identification in an online task-free stream. If an incoming batch belongs to an existing latent task, its Fisher principal subspace should have high similarity with historical subspaces. If it corresponds to a new or substantially different latent phase, its Fisher subspace should have low similarity to all historical subspaces. Intermediate similarity naturally corresponds to partial overlap, motivating the expansion operation in FiUni.

Therefore, Fisher geometry provides a unified signal for both detecting the latent task affiliation of each incoming batch and constructing the corresponding low-rank adaptation subspace.

Connection to subspace Methods. The above analysis also provides a possible explanation for why subspace methods, such as SeqLoRA and O-LoRA, can be effective in continual learning. We conjecture that, around a pre-trained model, different downstream tasks do not explore the entire parameter space uniformly, but instead tend to update along Fisher-sensitive directions induced by their task data. As a result, the effective update space used by each task may occupy only a low-dimensional subspace of the full parameter space. When different tasks correspond to distinct Fisher-sensitive directions, stochastic gradient descent may naturally push their updates toward partially separated or nearly orthogonal regions during training.

From this perspective, orthogonal constraints and subspace allocation mechanisms may not create task isolation from scratch; rather, they may reinforce the implicit geometric separation induced jointly by the pre-trained model and downstream tasks. Thus, the effectiveness of these methods may not solely come from the imposed orthogonality constraints, but may also rely on a taskrelated geometric structure that already exists near the pre-trained model: different tasks correspond to different dominant Fisher directions, while related tasks share partially overlapping directions. This perspective suggests that parameter isolation and knowledge reuse in continual learning can be understood as explicit ways of exploiting this pretraining-induced geometric structure.

## B System-level Explanation of FiUni

Notably, the Fisher subspace similarity based detection mechanism is not tied to a specific LoRA adaptation design. Since it only relies on the geometric matching between the Fisher principal subspace of the current batch and historical subspaces, it can serve as a standalone upstream module for latent task identification and be plugged into other continual learning methods or task-expert selection scenarios. FiUni further combines this detection mechanism with Fisher-guided LoRA subspace construction to enable task-free continual parameter-efficient fine-tuning.

## C Experimental Settings

## C.1 Benchmarks

We evaluate FiUni on three continual learning benchmarks: Standard CL Benchmark (SC) (Zhang et al., 2015), Long Sequence Benchmark (LS) (Razdaibiedina et al., 2023), and TRACE (Wang et al., 2023b). These benchmarks cover task streams of different lengths, domains, and task formats. SC focuses on short text classification sequences, LS extends the setting to longer and more heterogeneous classification-oriented task streams by incorporating additional tasks from GLUE (Wang et al., 2019b) and SuperGLUE (Wang et al., 2019a), and TRACE further introduces diverse LLM-oriented tasks such as multilingual understanding, code completion, mathematical reasoning, and question answering. The benchmark composition and task orders are summarized in Tables 2, 3, and 4.

For the SC and LS benchmarks, we follow the same data sampling protocol as prior studies (Wang et al., 2023a; Liao et al., 2026). Specifically, we use the same task orders and the same sampled training and evaluation splits provided in their official code repositories, ensuring that both training and testing data are consistent with existing baselines for fair comparison.

In addition to hiding task boundaries, our setting differs from task-aware continual learning in how the data stream is constructed. We concatenate datasets from different tasks into a single continuous stream. As a result, around a manually defined task boundary, a training batch may contain samples from two substantially different datasets. This makes our task-free benchmark closer to realistic online scenarios and more challenging than the standard task-aware setting.

<table><tr><td>Dataset Name</td><td>Category</td><td>Task</td><td>Domain</td><td>Metric</td></tr><tr><td>Yelp</td><td>CL Benchmark</td><td>Sentiment Analysis</td><td>Yelp Reviews</td><td>Accuracy</td></tr><tr><td>Amazon</td><td>CL Benchmark</td><td>Sentiment Analysis</td><td>Amazon Reviews</td><td>Accuracy</td></tr><tr><td>DBpedia</td><td>CL Benchmark</td><td>Topic Classification</td><td>Wikipedia</td><td>Accuracy</td></tr><tr><td>Yahoo</td><td>CL Benchmark</td><td>Topic Classification</td><td>Yahoo Q&amp;A</td><td>Accuracy</td></tr><tr><td>AG News</td><td>CL Benchmark</td><td>Topic Classification</td><td>News</td><td>Accuracy</td></tr><tr><td>MNLI</td><td>GLUE</td><td>Natural Language Inference</td><td>Various</td><td>Accuracy</td></tr><tr><td>QQP</td><td>GLUE</td><td>Paraphrase Detection</td><td>Quora</td><td>Accuracy</td></tr><tr><td>RTE</td><td>GLUE</td><td>Natural Language Inference</td><td>News, Wikipedia</td><td>Accuracy</td></tr><tr><td>SST-2</td><td>GLUE</td><td>Sentiment Analysis</td><td>Movie Reviews</td><td>Accuracy</td></tr><tr><td>WiC</td><td>SuperGLUE</td><td>Word Sense Disambiguation</td><td>Lexical Databases</td><td>Accuracy</td></tr><tr><td>CB</td><td>SuperGLUE</td><td>Natural Language Inference</td><td>Various</td><td>Accuracy</td></tr><tr><td>COPA</td><td>SuperGLUE</td><td>Question Answering</td><td>Blogs, Encyclopedia</td><td>Accuracy</td></tr><tr><td>BoolQA</td><td>SuperGLUE</td><td>Boolean Question Answering</td><td>Wikipedia</td><td>Accuracy</td></tr><tr><td>MultiRC</td><td>SuperGLUE</td><td>Question Answering</td><td>Various</td><td>Accuracy</td></tr><tr><td>IMDB</td><td>SuperGLUE</td><td>Sentiment Analysis</td><td>Movie Reviews</td><td>Accuracy</td></tr></table>

Table 2: Task details of the Long Sequence Benchmark (LS). The first five datasets correspond to the standard CL benchmark, while the remaining tasks are drawn from GLUE, SuperGLUE, and IMDB.
<table><tr><td>Dataset</td><td>Source</td><td>Avg. Len.</td><td>Metric</td><td>Language</td><td>#Data</td></tr><tr><td colspan="6">Domain-specific</td></tr><tr><td>ScienceQA</td><td>Science</td><td>210</td><td>Accuracy</td><td>English</td><td>5,000</td></tr><tr><td>FOMC</td><td>Finance</td><td>51</td><td>Accuracy</td><td>English</td><td>5,000</td></tr><tr><td>MeetingBank</td><td>Meeting</td><td>2853</td><td>ROUGE-L</td><td>English</td><td>5,000</td></tr><tr><td colspan="6">Multi-lingual</td></tr><tr><td>C-STANCE</td><td>Social Media</td><td>127</td><td>Accuracy</td><td>Chinese</td><td>5,000</td></tr><tr><td>20Minuten</td><td>News</td><td>382</td><td>SARI</td><td>German</td><td>5,000</td></tr><tr><td colspan="6">Code Completion</td></tr><tr><td>Py150</td><td>GitHub</td><td>422</td><td>Edit Similarity</td><td>Python</td><td>5,000</td></tr><tr><td colspan="6">Mathematical Reasoning</td></tr><tr><td>NumGLUE-cm</td><td>Math</td><td>32</td><td>Accuracy</td><td>English</td><td>5,000</td></tr><tr><td>NumGLUE-ds</td><td>Math</td><td>21</td><td>Accuracy</td><td>English</td><td>5,000</td></tr></table>

Table 3: Task details of TRACE. The benchmark covers domain-specific tasks, multilingual tasks, code completion, and mathematical reasoning.

## C.2 Baselines

We compare FiUni with a broad set of continual learning and parameter-efficient fine-tuning baselines. These baselines cover sequential LoRA training, replay-based methods, regularization-based continual learning, prompt-based continual learning, and recent orthogonal or subspace-based continual PEFT methods.

SeqLoRA. SeqLoRA sequentially trains LoRA modules on the task stream without preserving previous task knowledge or allocating task specific subspaces. It serves as a simple continual PEFT baseline that directly applies LoRA to sequential learning.

SeqLoRAReplay. SeqLoRAReplay extends SeqLoRA with experience replay. During training on new tasks, it reuses a small number of stored examples from previous tasks to reduce forgetting.

IncLoRA. IncLoRA incrementally introduces new LoRA parameters for newly encountered tasks. Instead of using a single shared LoRA module, it expands adaptation parameters along the task sequence to provide additional capacity for new knowledge.

EWC (Kirkpatrick et al., 2017). Elastic Weight Consolidation is a regularization based continual learning method. It estimates parameter importance, typically using Fisher information, and penalizes changes to parameters important for previous tasks.

L2P (Wang et al., 2022). Learning to Prompt maintains a pool of learnable prompts and selects input relevant prompts during training. It adapts to new tasks by reusing and updating prompt knowledge from previous tasks.

LFPT5 (Qin and Joty, 2022). LFPT5 continuously trains soft prompts for T5 based continual learning. The learned prompts are used both for task solving and pseudo sample generation, which supports experience replay.

<table><tr><td>Benchmark</td><td>Order</td><td>Task Sequence</td></tr><tr><td rowspan="3">Standard CL Benchmark</td><td>1</td><td> $\mathrm { D B p e d i a }  \mathrm { A m a z o n }  \mathrm { Y a h o o }  \mathrm { A G N e w s }$ </td></tr><tr><td>2</td><td> $\mathrm { D B p e d i a }  \mathrm { A m a z o n }  \mathrm { A G N e w s }  \mathrm { Y a h o o }$ </td></tr><tr><td>3</td><td> $\mathrm { Y a h o o  A m a z o n  A G N e w s  D B p e d i a }$ </td></tr><tr><td rowspan="3">Long Sequence Benchmark</td><td>4</td><td> $\mathrm { M N L I } \to \mathrm { C B } \to \mathrm { W i C } \to \mathrm { C O P A } \to \mathrm { Q Q P } \to \mathrm { B o o l Q A } \to \mathrm { R T E } \to \mathrm { I M D B } \to $   $\mathrm { Y e l p } \to \mathrm { A m a z o n } \to \mathrm { S S T } _ { } 2 \to \mathrm { D B p e d i a } \to \mathrm { A G ~ N e w s } \to \mathrm { M u l t i R C } \to \mathrm { Y a h o o }$ </td></tr><tr><td>5</td><td> ${ \mathrm { M u l t i R C } } \to { \mathrm { B o o l Q A } } \to { \mathrm { W i C } } \to { \mathrm { M N L I } } \to { \mathrm { C B } } \to { \mathrm { C O P A } } \to { \mathrm { Q Q P } } \to { \mathrm { R T E } } \to$   $\mathrm { I M D B } \to \mathrm { S S T - 2 } \to \mathrm { D B p e d i a } \to \mathrm { A G } \mathrm { N e w s } \to \mathrm { Y e l p } \to \mathrm { A m a z o n } \to \mathrm { Y a h o o }$ </td></tr><tr><td>6</td><td> ${ \mathrm { Y e l p } } \to { \mathrm { A m a z o n } } \to { \mathrm { M N L I } } \to { \mathrm { C B } } \to { \mathrm { C O P A } } \to { \mathrm { Q Q P } } \to { \mathrm { R T E } } \to { \mathrm { I M D B } } \to$   $\mathrm  S S \hat { T } - 2  D B p e d i a  A G N e w s  Y a h o o  M u l t i R C  B o o l Q A  W i C$ </td></tr><tr><td>TRACE</td><td>7</td><td> $\mathbf { C } \cdot S \mathbf { T A N C E }  \mathbf { F O M C }  \mathbf { M e e t i n g B a n k }  \mathbf { P y } 1 5 0  \mathbf { S c i e n c e Q A } $   $\mathrm { N u m G L U E \mathrm { - } c m }  \mathrm { N u m G L U E \mathrm { - } d s }  2 0 \mathrm { M i n u t e n }$ </td></tr></table>

Table 4: Task orders used in our continual learning experiments. Orders 1–3 correspond to SC, Orders 4–6 correspond to LS, and Order 7 corresponds to TRACE.

O-LoRA (Wang et al., 2023a). O-LoRA reduces task interference by enforcing orthogonality among the low rank update spaces of different tasks. In this way, task specific parameter updates are isolated in separate subspaces.

MIGU (Du et al., 2024). MIGU is a rehearsal free and task label free continual learning method. It exploits the magnitude distribution of outputs in linear layers and updates parameters with large output magnitudes to reduce gradient conflicts across tasks.

SpaRTA (Liao et al., 2026). SpaRTA is a continual PEFT method that manages task specific and shared adaptation components. It aims to reduce forgetting while preserving transfer across related tasks.

ELLA (Biswas et al., 2026). ELLA is a continual PEFT baseline that models task relationships in the adaptation space. It encourages knowledge sharing across related tasks while maintaining task specific components to reduce interference.

## C.3 Implementation Details

We implement all experiments based on Transformers 4.57.6 (Wolf et al., 2020) and PyTorch 2.6.0 (Paszke et al., 2019). To ensure consistency across different methods, all baselines and FiUni are evaluated under the same software environment and hardware configuration. All experiments are conducted on a single NVIDIA RTX 6000 Ada Generation GPU. All models are initialized in bfloat16 (bf16) precision to reduce memory usage and computational overhead. Unless otherwise specified, all methods use the same backbone model, task orders, data splits, and evaluation protocol within each benchmark to ensure fair comparison.

For the SC and LS benchmarks, we concatenate all training data according to the predefined task orders and feed them into the model as a continuous batch stream. This setting simulates an online task-free continual learning scenario, where data arrive sequentially and the model must update itself based only on the current batch. During training, the model has no access to task boundaries or task IDs, and therefore cannot rely on explicit task-switching signals to create, select, or reset adaptation modules.

For the TRACE benchmark, different tasks involve different batch size requirements. Therefore, for data-format compatibility, we organize training batches task by task rather than directly mixing all samples into a single stream. However, this does not provide task-aware supervision to the model: task identities and explicit task-switching signals are still hidden during training. For each incoming batch, its latent task affiliation and the corresponding subspace management decision are determined adaptively by FiUni based on Fisher subspace similarity.

We summarize the hyperparameters used for different backbones and benchmarks in Tables 5 and 6. Here, r denotes the LoRA/FiLoRA subspace rank, $r _ { \mathrm { d e t } }$ denotes the detection rank used for Fisher subspace matching, and $\Delta r$ denotes the additional rank introduced during expansion. The thresholds $\tau _ { \mathrm { l o w } }$ and $\tau _ { \mathrm { h i g h } }$ control the New, Expand, and Reuse decisions. We use $\mathcal { L } _ { \mathrm { a d a p t } }$ for the layers used in adaptation, ${ \mathcal { L } } _ { \mathrm { d e t } }$ for the layers used in Fisher-based detection, and M for the target modules. The cooldown steps $C _ { \mathrm { e x p a n d } }$ and $C _ { \mathrm { n e w } }$ prevent Expand and New operations from being triggered too frequently.

<table><tr><td rowspan="2">Hyperparameter</td><td colspan="3">T5-large</td><td colspan="3">Llama-3.1-8B</td></tr><tr><td>SC</td><td>LS</td><td>TRACE</td><td>SC</td><td>LS</td><td>TRACE</td></tr><tr><td>Batch size</td><td></td><td></td><td>32</td><td></td><td></td><td></td></tr><tr><td>Learning rate</td><td></td><td>1e-3</td><td></td><td></td><td>1e-4</td><td></td></tr><tr><td>Epochs</td><td></td><td></td><td>1</td><td></td><td></td><td></td></tr><tr><td>FiLoRA rank r</td><td></td><td>32</td><td></td><td></td><td>128</td><td></td></tr><tr><td>Dropout</td><td>0</td><td>0.05</td><td>0</td><td>0.1</td><td>0.1</td><td>0</td></tr><tr><td>Adaptation layers  $\mathcal { L } _ { \mathrm { a d a p t } }$ </td><td></td><td></td><td>all</td><td></td><td></td><td></td></tr><tr><td>Target modules  $\mathcal { M }$ </td><td></td><td> ${ \bf q , k , v , o }$ </td><td></td><td></td><td>q,v</td><td></td></tr></table>

Table 5: Training and adaptation hyperparameters for FiUni. Batch size denotes the effective batch size, computed as the raw batch size multiplied by the number of gradient accumulation steps.
<table><tr><td rowspan="2">Hyperparameter</td><td colspan="3">T5-large</td><td colspan="3">Llama-3.1-8B</td></tr><tr><td>SC</td><td>LS</td><td>TRACE</td><td>SC</td><td>LS</td><td>TRACE</td></tr><tr><td>Detection rank  $r _ { \mathrm { d e t } }$ </td><td></td><td>4</td><td></td><td></td><td>8</td><td></td></tr><tr><td>Expansion rank  $\Delta r$ </td><td>0</td><td>4</td><td>4</td><td>8</td><td>8</td><td>8</td></tr><tr><td>Tlow</td><td></td><td>0.5</td><td></td><td></td><td>0.6</td><td></td></tr><tr><td>Thigh</td><td></td><td></td><td>0.7</td><td></td><td></td><td></td></tr><tr><td>Detection layers  $\mathcal { L } _ { \mathrm { d e t } }$ </td><td></td><td>23</td><td></td><td></td><td>31</td><td></td></tr><tr><td>Expand cooldown  $C _ { \mathrm { e x p a n d } }$ </td><td>50</td><td>200</td><td>100</td><td>40</td><td>40</td><td>40</td></tr><tr><td>New cooldown  $C _ { \mathrm { n e w } }$ </td><td>30</td><td>10</td><td>100</td><td>10</td><td>10</td><td>10</td></tr></table>

Table 6: Detection and decision hyperparameters for FiUni.

## D Additional Results

## D.1 Ablation on Fisher Subspace Similarity Computation

We further analyze how the computation of Fisher subspace similarity s is affected by different design choices, including the detection rank $r _ { \mathrm { d e t } }$ , the number of samples used for Fisher estimation, the selected modules, and the selected layers. Unless otherwise specified, the ablation is conducted on the LS benchmark using the same similarity computation protocol as in the main paper. Each figure reports both the task-level Fisher subspace similarity matrix and the scatter plot comparing within-task self-similarity with task-to-other similarity.

Effect of detection rank. Figure 4 shows the effect of using different detection ranks. When $r _ { \mathrm { d e t } }$ is too small, the estimated Fisher subspace may not contain enough task-discriminative directions, leading to weaker separation between self-similarity and task-to-other similarity. $\mathrm { \bf A s } \ r _ { \mathrm { d e t } }$ increases, the similarity structure becomes more stable and better reflects task-level relationships. However, using an overly large rank may also introduce less informative directions, making the similarity signal less concentrated. This motivates the use of a moderate detection rank in FiUni.

Effect of Fisher estimation samples. Figure 5 studies the influence of the number of samples used for estimating the Fisher/K-FAC factors. With very few samples, the estimated Fisher statistics are noisier, and the resulting similarity signal becomes less stable. Increasing the number of samples improves the consistency of within-task self-similarity and makes the separation from unrelated tasks clearer. These results show that Fisher subspace similarity can be estimated from a small number of downstream samples, while a sufficient sample size is still helpful for robust task discrimination.

Effect of module selection. Figure 6 compares different choices of Transformer modules for Fisher subspace estimation. The results indicate that the similarity pattern is relatively robust across different module selections. In particular, attentionrelated modules already provide meaningful taskdiscriminative signals, suggesting that FiUni does not require estimating Fisher statistics on all modules to obtain useful task geometry. This supports our design of performing detection only on selected modules to reduce computation.

![](images/8eec928fcb44c3645749b340b183fa1dc4e3bbc8e64428545cdb5e9d4c7b6c4b.jpg)  
Figure 4: Ablation on the detection rank $r _ { \mathrm { d e t } }$ for Fisher subspace similarity computation. A moderate rank provides a stable separation between within-task self-similarity and task-to-other similarity.

Effect of layer selection. Figure 7 evaluates whether Fisher subspace similarity depends strongly on the choice of layers. We observe that different layer selections produce broadly consistent task-similarity patterns. This suggests that the task-discriminative Fisher geometry is not limited to a specific layer, and reliable detection can be achieved using only a subset of layers. Therefore, FiUni can reduce online detection overhead by computing Fisher subspace similarity on selected layers rather than all parameter blocks.

Overall, these ablations show that Fisher subspace similarity is a robust signal for capturing tasklevel geometry. Although the quality of the signal is affected by the number of estimation samples and the detection rank, it remains relatively stable across different module and layer selections. This supports FiUni’s practical design of using a moderate detection rank, a small number of Fisher estimation samples, and only selected modules/layers for efficient online task-free detection. Meanwhile, although the overall similarity values decrease under extremely few-shot estimation and the separation between related and unrelated tasks becomes less pronounced, the within-task similarity still remains the highest for most tasks. This suggests that Fisher subspace similarity can still preserve a certain degree of task-discriminative signal in low-sample regimes, providing a practical basis for sample-efficient batch-level detection in downstream continual adaptation.

## D.2 Training Dynamics and Decision Trajectory

We further analyze the online behavior of FiUni by jointly examining the training loss and the corresponding batch-level decision trajectory. Figures 8 and 9 present the results of T5-large on the LS and SC benchmarks, respectively. In each figure, the upper panel shows the training loss along the sequential data stream, and the lower panel shows the corresponding Reuse, Expand, and New decisions.

Across both benchmarks, the decision trajectory follows the training dynamics in a meaningful yet nontrivial manner. When the incoming data become substantially less compatible with previously accumulated knowledge, the loss often rises and FiUni correspondingly activates New or Expand to introduce additional subspace capacity. By contrast, when the stream remains relatively stable, the decisions are dominated by Reuse, meaning that the current batches can be effectively absorbed by existing Fisher subspaces. Such a pattern is particularly visible on the LS benchmark, where several major shifts in the stream are accompanied by noticeable changes in the loss curve and the appearance of newly allocated subspaces.

At the same time, FiUni is not merely reacting to abrupt loss changes. Compared with methods that rely on sudden loss variation for task-boundary detection, FiUni operates at a finer batch-level granularity through Fisher subspace matching. Consequently, its decision process is able to capture latent task transitions even when the loss curve does not exhibit a pronounced spike. For example, in the SC benchmark, the transitions from Amazon to Yahoo and from Yahoo to AG do not lead to particularly sharp loss peaks, yet FiUni still introduces new subspaces at the corresponding phases. A similar phenomenon appears in the LS benchmark, such as the transitions from Amazon to SST-2 and from AG to MultiRC, where the loss changes only mildly while the decision trajectory still detects the underlying shift in task geometry.

![](images/dc64e5549c3fc666f6143b1fc8e5e02100cb594b0ce0c29587f4e64a0c29aeee.jpg)

![](images/f5afeb705751188c0c2e62742810920f2cd85e68a7250b7a55f4ada54a6523eb.jpg)

![](images/b9c6e47d505617db658183d98f6d3e66f171361d2f43ed13a9fc97c0e376c224.jpg)

![](images/4e38892191fe2059e8d88c4d5848095191e7afc76dc6f0cdddcd5515168915df.jpg)  
Shot\_2

![](images/cf72f5190d1ffe0735ddfd9cba7c72b8912c30d6759265880dc9697103f7c55e.jpg)  
Shot\_8

![](images/b9cb7c0db0b7555f07d4d36e4a28c64ea6e084d05373d4866e4390cfa1a59596.jpg)  
Shot\_32  
Figure 5: Ablation on the number of samples used for Fisher estimation. More samples lead to more stable Fisher subspaces and clearer separation between within-task and cross-task similarities.

Another noteworthy characteristic is that FiUni does not simply replicate manually defined task boundaries. Some transitions are handled through Reuse or Expand rather than immediately opening a new subspace, especially when adjacent tasks remain geometrically related. Likewise, within a single human-defined task, FiUni may occasionally expand an existing subspace when the incoming batches reveal additional internal variation. This behavior is consistent with our notion of latent tasks: the model tracks its own perceived learning phases rather than mechanically following datasetlevel segmentation.

In addition, in the SC benchmark, we observe several Expand decisions within the DBpedia stage. Although DBpedia is treated as a single datasetlevel task, it covers a wide range of entity categories and topic types, which can induce noticeable batch-level variation in Fisher geometry. These batches may not be fully explained by the initial DBpedia subspace, but they remain related enough to avoid opening entirely new subspaces. FiUni therefore expands the existing subspace to provide additional capacity for intra-task variation while preserving knowledge reuse. The different behavior of DBpedia in SC and LS further reflects the context-dependent nature of FiUni’s decisions. Rather than assigning a fixed decision pattern to each dataset, FiUni compares every incoming batch with the current historical Fisher subspace pool. In SC, DBpedia appears when the historical pool is still small, making its intra-task variation more likely to fall into the intermediate-similarity region and trigger Expand. In LS, however, DBpedia is encountered after many previous tasks have already populated the subspace pool, so some DBpedia batches can be better explained by existing related subspaces and are thus handled through Reuse. This behavior further supports that FiUni tracks model-perceived latent learning phases rather than simply following dataset names or manually defined task boundaries.

Taken together, the loss-decision analysis provides an intuitive view of how FiUni manages subspaces throughout online continual adaptation.

![](images/ea2b460f2f4751cb41bcc1998a4646e5f91e2e4d525acf59a05663872e7eaafb.jpg)

![](images/b935c51bb4cd779efb0c35670fd45131a543831eb3a1676234e2615735043f48.jpg)

![](images/068f4cd5997ad8956b9c0c35c8739de0faf1e268b46a31c7f7ab92d70b7cb21a.jpg)

![](images/319d9e36982e55784693e2685495e480c95af06ed5c0eb0d8ad2ae5f0551d530.jpg)

![](images/ff7c3d215cd85bdc711baae49399f042fc5091e8f17441ce9a20b93199b1ed47.jpg)  
Module\_{q,k,v,o,wi,wo}

![](images/bd40cef30713c5d20cae57725a92220c41eb2211c9261308e5a8c11fdab03d51.jpg)  
Module\_{q,o}  
Figure 6: Ablation on module selection for Fisher subspace similarity computation. Different module choices preserve similar task-level structures, showing that the similarity signal is not tied to a single module configuration.

Large shifts in the stream are often accompanied by subspace creation or expansion, while stable periods are mainly handled through knowledge reuse. More importantly, the decision mechanism is not limited to coarse signals derived from loss fluctuations, but can also resolve finer latent transitions at the batch level, which is especially valuable in realistic task-free settings where explicit task boundaries are unavailable.

## D.3 Dynamic Trainable Parameter Analysis

We further analyze the number of trainable parameters from a dynamic perspective on the LS benchmark with T5-large. Different from task-aware methods that typically allocate or optimize a fixed amount of adaptation parameters for each task, Fi-Uni adjusts its trainable parameter budget according to the online decision trajectory. Specifically, a Reuse decision does not introduce additional subspace parameters, while Expand only increases the capacity of an existing subspace by a small amount. A New decision allocates a new LoRA subspace only when the incoming batch is geometrically distinct from all historical subspaces. Therefore, the trainable parameter count of FiUni grows adaptively with the data stream rather than linearly following the number of manually defined tasks.

<table><tr><td>Metric</td><td>O-LoRA</td><td>SpaRTA</td><td>FiUni (ours)</td></tr><tr><td>Trainable Params</td><td>35.39M</td><td>44.24M</td><td>3.62M</td></tr></table>

Table 7: Trainable parameter comparison on the LS benchmark. For FiUni, we report the dynamic trainable parameter count obtained on Order 4.

For comparison, according to the hyperparameter settings reported in prior work (Wang et al., 2023a; Liao et al., 2026), we compute the accumulated trainable parameters of O-LoRA and SpaRTA on T5-large, while for FiUni we directly count the dynamically allocated trainable parameters after training on Order 4. The comparison is summarized in Table 7.

The results show that FiUni uses substantially fewer trainable parameters than the fixed task-level allocation adopted by strong task-aware baselines. Since FiUni only trains the core matrix within each Fisher-guided LoRA subspace, the trainable parameter count is substantially reduced at the subspace level. On top of this, the adaptive mechanism further helps control parameter growth: related batches can share existing subspaces, moderately shifted batches only expand the most relevant subspace, and new subspaces are created only when necessary. As a result, FiUni avoids assigning a full set of adaptation parameters to every dataset-level task, while still preserving enough flexibility to handle distributional changes in the online stream. This adaptive parameter allocation is especially beneficial for long-sequence continual learning, where blindly adding a complete adapter for each task would lead to rapid parameter growth.

![](images/061e1237e236c8620aee8694d52662e3a2d33a983882ee7d583adefc907ba964.jpg)

![](images/1237e90057efecc7e691d218d144d6ef0c94c829a669facb432afe32e42b47fd.jpg)

![](images/6750163144d4e7e169fdcb970f6cc5114e1174b2d58d96048d11568bb6c3be76.jpg)

![](images/2117c301f983c26b50f24b859c30dface52e897420c63b8febfc9b1a47013e4b.jpg)

![](images/5a3c7b7f9864a85654014a9e27da44506ad80c629f7ee1095e740e2a083d74e2.jpg)  
Layer\_11

![](images/b626c92d7664174efeb163a7b77cd1d29628f536781edeec27f0d9edc985dd0f.jpg)  
Layer\_all  
Figure 7: Ablation on layer selection for Fisher subspace similarity computation. The resulting similarity structures remain broadly consistent across different layer choices, supporting efficient detection with selected layers.

![](images/b7a5f093072098b31070f4ddfafb9e1f38c187bbca3503b994f3c25840ae145a.jpg)  
Figure 8: Training loss and FiUni decision trajectory on the LS benchmark with T5-large. The upper panel shows the training loss along the online data stream, and the lower panel shows the corresponding batch-level Reuse, Expand, and New decisions.

![](images/f78b29be827f2491ca6a37bca0b36581c9c526a5125e364dfcefefe21c6ac79d.jpg)  
Figure 9: Training loss and FiUni decision trajectory on the SC benchmark with T5-large. Even when some task transitions do not produce obvious loss spikes, FiUni can still identify the corresponding latent shifts through Fisher subspace matching and trigger appropriate batch-level decisions.