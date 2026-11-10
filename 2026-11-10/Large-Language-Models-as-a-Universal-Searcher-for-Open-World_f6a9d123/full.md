# Large Language Models as a Universal Searcher for Open-World Automated Machine Learning

Yuxuan Hu, Jihao Liu, Weikang Shi, Xijie Huang, Linda Wei, Mingjie Zhan, Aojun Zhou, Rui Liu, and Hongsheng Li, Member, IEEE

Abstract—Developing effective machine learning models requires complex manual tuning, while current Automated Machine Learning (AutoML) frameworks remain domain-specific and isolate model architecture, training hyperparameter, and data ratio optimization. Here, we bridge this gap by proposing a universal, large language model (LLM)- based AutoML framework that jointly optimizes these configurations for the first time without the need for task-specific fine-tuning. We propose NCode, a unified numerical encoding scheme that translates diverse AutoML design choices into universal discrete sequences and effectively reframes the optimization process as a unified ranking task. Trained solely on optimization trajectories derived from a limited, singledomain vision task, the proposed LLM-based searcher demonstrates impressive cross-task, cross-modality and joint search-space optimization generalizability, achieving state-of-the-art performance across image classification, image segmentation, speaker verification and textto-image benchmarks. Crucially, it exhibits superior scalability in the large-scale training and joint optimization of a billion-parameter Diffusion Transformer, requiring 896 GPU-days and attaining a state-of-the-art GenEval score of 0.91. By leveraging reinforcement learning to instill Chain-of-Thought reasoning, we provide a systematic way to interpret the LLM-based searcher’s internal decision-making mechanisms, revealing its optimization strategy shifted from the early-stage diversitydriven exploration to the late-stage anchor exploitation. Our proposed framework provides a scalable, generalizable, and interpretable pathway toward general-purpose open-world machine learning optimization.

Index Terms—Automated machine learning, open-world generalization, joint optimization, large language models.

## 1 INTRODUCTION

ECENT advancements in Machine Learning (ML) have vision, natural language processing, and audio analysis. Despite the progress, developing effective ML models remains a complex endeavor, often requiring extensive human experience in designing network architectures, tuning hyperparameters, and organizing training data. Consequently, Automated Machine Learning (AutoML) has emerged as a pivotal solution, aiming to automate such design choices of ML pipelines with minimal human intervention.

While intensive explorations in AutoML have been made, particularly in Neural Architecture Search (NAS) for architecture design [1, 2, 3] and Hyperparameter Optimization (HPO) for determining training hyperparameters [4], the field still lacks a general-purpose AutoML framework. This gap has become increasingly challenging in the era of foundation models. Currently, the realization of such an open-domain system is impeded by three fundamental obstacles:

• Open-world problem generalization. Existing AutoML approaches are typically optimized for specific search spaces, lacking the flexibility to generalize across diverse tasks, datasets and architectures. For instance, the popular gradient-based NAS method DARTS [3] achieves 97.24% accuracy on the CIFAR-10 test set within its original search space, yet suffers a severe performance collapse to 39.77% when evaluated on the widely used NAS-Bench-201 benchmark [5]. Even though both settings share identical dataset and optimizers, this sharp decline underscores the inherent fragility of current NAS and AutoML techniques when subjected to structural shifts in the search space. Consequently, this limitation hinders the application of AutoML methods in open-world problems that data distributions, architecture search spaces and task definitions are novel or unseen.

• Intricate interdependence of model optimization choices. The optimal performance of a model is the result of the complex interplay between model architectures, training hyperparameters, and data ratios, etc. Such highly non-linear interactions are difficult to optimize in isolation. Existing AutoML methods fail to capture the synergies and often lead to sub-optimal local minima when components are optimized separately.

• The prohibitive cost of AutoML training data acquisition. Creating AutoML training datasets is computationally expensive and time-consuming, as it involves the repetitive process of model training and performance evaluation for every new task. The high cost hinders the development and scalability of learningbased AutoML methods.

To overcome these barriers, the field is witnessing a paradigm shift from specialized optimization algorithms toward more generalizable approaches. In particular, Large Language Models (LLMs) were leveraged to optimize AutoML problems to take advantage of their strong reasoning capabilities [6, 7]. However, they remain heavily dependent on task-specific training, fine-tuning, and manual prompt engineering. As a result, there has been little deployment of AutoML frameworks that can navigate open-world problems spanning across multiple architectures, tasks, and modalities.

In this work, we introduce an open-world LLM-based AutoML framework that is capable of jointly optimizing the complex combination of model architectures, training hyperparameters, data ratios for different tasks without taskspecific fine-tuning for the first time. To enable the optimization over the multiple intrinsically diverse search spaces, we introduce NCode, an encoding scheme that maps heterogeneous design choices to their respective indices, representing them as unified numerical sequences (Fig. 1b,c). By converting diverse AutoML tasks into a universal discrete optimization problem, NCode allows LLMs to easily generalize to unseen open-world domains without domain-specific fine-tuning (Fig. 1a). To train the LLM-based searcher, we curate optimization trajectories from the NAS-Bench-301 dataset [8], which provides model architectures and their corresponding image classification performances on the CIFAR-10 dataset. We model the model arch instances as a densely connected Directed Acyclic Graph (DAG) and apply stochastic edge pruning to generate diverse, localized architecture search spaces for rich trajectory formulation. Furthermore, for the LLM-based searcher’s outputs, we reframe the discrete NCode optimization as a ranking problem, where the LLM identifies superior candidates from a candidate pool rather than directly predicting their NCodes, a setting demonstrated to be significantly simpler to learn than directly generating full NCodes. Consequently, despite being trained exclusively on single-domain vision data, our LLM searcher generalizes effectively to cross-domain joint optimization tasks where data collection is typically costprohibitive.

A preliminary version of our system that focused only on model architecture search was presented in EMNLP2025 [9]. The current study unveils the unified LLM-based searcher’s capabilities in significantly more complex, heterogeneous search spaces and mastering the synergy between model architectures, training hyperparameters, and data ratios across seven domains, four tasks and three modalities (vision, text, audio) without fine-tuning. More importantly, our LLM-based searcher is trained to explain its optimization rationales by learning explainable Chain-of-Thought (CoT)

via Group Relative Policy Optimization (GRPO) [10]. Our analysis revealed how the model navigates the solution space for optimization and offers critical insights. Extensive experiments provide compelling evidence and demonstrate that our framework achieves state-of-the-art performance across a diverse suite of open-world machine learning problems.

We highlight our major contributions as follows:

1) Unified Problem Representation: We introduce a universal numerical encoding that maps model architectures, hyperparameters, and data ratios into a unified discrete space, allowing the AutoML framework to jointly optimize diverse design choices across distinct search spaces.

2) Search Trajectory Curation: We introduce a pruningbased architecture sampling strategy to curate rich, diverse optimization trajectories, mitigating the prohibitive cost of AutoML data collection. By simultaneously reframing the LLM’s objective from direct sequence generation to conditional ranking, we boost the searcher’s capacity to generalize across complex optimization landscapes.

3) Extensive Empirical Validation: By deploying our approach across real-world machine learning tasks in diverse scenarios, and achieving state-of-the-art (SOTA) performances, we demonstrate the framework’s robust zero-shot generalizability.

4) Interpretability via RL-driven Chain-of-Thought: To unveil the internal decision-making mechanism of the LLM-based searcher, we leverage Group Relative Policy Optimization (GRPO) to instill Chain-of-Thought (CoT) reasoning without requiring human annotation. This self-explaining capability provides critical insights into the model’s learned optimization strategies, specifically revealing a dynamic shift from early-stage diversitydriven exploration to late-stage anchor exploitation, and highlights the inherent limitations of conventional discrete optimization methods in existing AutoML frameworks.

## 2 RELATED WORK

## 2.1 Traditional Automated Machine Learning

Automated Machine Learning (AutoML) has evolved significantly, progressing from early reinforcement learning [1, 11] and gradient-based Neural Architecture Search (NAS) [3] to Transformer-based Hyperparameter Optimization (HPO) [4]. More recently, the field has increasingly embraced advanced generative paradigms, such as leveraging multi-conditioned graph diffusion models [12] to sample candidate neural structures. Despite these advancements, conventional AutoML approaches remain inherently tailored to predefined, task-specific search spaces. Furthermore, even contemporary generative NAS models typically treat architecture search and training configuration tuning as disjoint problems. This isolation fails to capture the intricate synergies among heterogeneous components, restricting their applicability in open-world scenarios involving unseen modalities or structural shifts. In contrast to optimizing components in silos, our framework introduces

![](images/a425794d6b87abbff255eb8c9fa6183d11441bd5f5fa30ebed68faea9ebf9268.jpg)

(c) Search Space NCode Encoding
<table><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1><img src="images/75a8dc9a416b96d53d21b213c10b9e4bea1bb07ea2a40de1d85c67eb29001b1b.jpg"/></td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>4 Cross-attn- 3072 1e-5- 4.0</td></tr></table>

Fig. 1. Overview of the proposed LLM-based open-world AutoML framework and the universal NCode representation. (a) Conventional AutoML paradigms typically require separate adaptation of the model for different tasks. In contrast, our proposed open-world LLM-based search paradigm uses universal encoding to jointly optimize model architectures, training configurations, and data combinations across tasks, domains, and modalities. (b) NCode maps heterogeneous AutoML design choices, including architectures, hyperparameters, and data sampling weights, into unified discrete numerical sequences. (c) Illustrative encoding process for models with varying topologies, operations, training hyperparameters, and data ratios. Note that the depicted codebook is a simplified excerpt; the functional codebook used in our experiments is more extensive.

NCode to unify these heterogeneous design choices into a shared numerical space, enabling holistic joint optimization.

## 2.2 LLM-Based Automated Machine Learning.

Recent years have witnessed a growing interest in leveraging Large Language Models (LLMs) for Neural Architecture Search (NAS) and Automated Machine Learning (AutoML), with representative frameworks including AutoML-GPT [6], LLMatic [13], and CoLLM-NAS [7]. Despite these advances, existing LLM-based searchers suffer from key limitations: they either rely heavily on task-specific fine-tuning or formulate search as a direct textual generation task, which is prone to producing invalid configurations and exhibiting high learning complexity in vast discrete spaces. While our preliminary work, LM-Searcher [9], initiated the exploration of cross-domain NAS via unified numerical encoding, it remained restricted to architecture-only search. To overcome these bottlenecks, the proposed framework unifies heterogeneous configurations into a discrete NCode space and reframes the search as a conditional candidate ranking problem. This design achieves robust zero-shot joint optimization of architectures, hyperparameters, and data ratios across diverse domains and modalities without task-specific adaptation.

## 2.3 Explainability and Chain-of-Thought Reasoning.

Traditional AutoML algorithms and neural search agents are frequently criticized for their black-box nature, providing minimal intrinsic rationale for their optimization trajectories [14]. While Chain-of-Thought (CoT) prompting [15] and reinforcement learning algorithms like Group Relative Policy Optimization (GRPO) [10] have boosted logical reasoning in LLMs, applying self-explaining CoT to interpret AutoML search logic remains under-explored. To bridge this gap, the proposed framework leverages GRPO to posttrain the LLM-based searcher to articulate its reasoning steps in natural language. By taxonomizing the resulting CoT outputs into distinct behavioral patterns, we provide interpretability into LLM-driven optimization mechanisms.

![](images/4f2d0877826793f3c0937ddf768f73d7e76fca7ab576f1d9ee41ee120db8e950.jpg)  
Fig. 2. Overview of post-training the LLM with CoT reasoning capabilities using Group Relative Policy Optimization. A Qwen3-8B model with native thinking abilities is fine-tuned on our curated dataset via reinforcement learning to articulate its reasoning paths. Given the system prompt, historical context, and candidate configurations represented as NCodes, the policy model generates multiple candidate responses. The total reward is calculated as a weighted sum of the accuracy reward (whether the model selects the optimal configuration from the candidate pool) and the format reward (whether the model outputs a complete reasoning process and the final answer in the correct format).

## 3 THE LLM-BASED SEARCHER FRAMEWORK

## 3.1 Overview

To enable the joint optimization of complex AutoML design choices including model architectures, training hyperparameters, and data ratios, we propose a universal, LLM-based open-world search framework. The core of our framework is a unified numerical encoding scheme termed NCode, which maps heterogeneous configurations into discrete sequences. During the search phase, the LLM-based searcher iteratively analyzes a historical buffer of evaluated NCode-reward pairs to autonomously identify and select the most promising configuration from an unevaluated candidate pool. Rather than relying on domain-specific finetuning, our LLM-based searcher is trained to implicitly learn optimization trajectories and reasoning strategies, allowing it to seamlessly generalize to unseen configurations and modalities.

## 3.2 Universal NCode Representation

To unify the representation of heterogeneous AutoML design choices (e.g., model architecture, training hyperparameters and data ratios) and task domains, we uniformly encode all configurations into discrete numerical sequences termed NCode. Formally, let a given AutoML search space be denoted as S. An AutoML configuration $a \in S$ can be defined by a set of n decisions, which may represent operational primitives, network connectivity, training modulespecific hyperparameters or data ratios. We define an encoding function ${ \hat { f } } : S  { \mathcal { C } } ,$ where C is the universal space of NCode sequences. The function maps the configuration to a string of discrete indices, denoted as $c = ( c _ { 1 } , c _ { 2 } , \ldots , c _ { n } )$ where each element $c _ { i }$ is a numerical index corresponding to a selected configuration option of the i-th configuration.

NCode enables the joint states of model topologies, operations, training hyperparameters, and data ratios. To represent network topologies, networks can be modeled as Directed Acyclic Graphs (DAGs), where each edge represents an operation and each node represents a summed feature map. For instance (Fig. 1b), in the NAS-Bench-201 search space, each architecture is defined by assigning one of five distinct operations to each of the six edges within a cell. The NCode representation establishes a mapping between these operations and a set of integers. Notably, by framing the search as a black-box combinatorial optimization problem, NCode natively extends to jointly encode other heterogeneous configurations. Fig. 1c illustrates how NCode jointly encodes model architectures, training hyperparameters, and data ratios.

## 3.3 Optimization Trajectory Curation

To effectively leverage LLMs as a general-purpose searcher for AutoML, we curated a dataset of optimization trajectories to capture the sequential, causal steps of iterative search processes. Specifically, each optimization trajectory comprises a sequence of AutoML configuration–reward pairs, with architectures represented by our unified NCode scheme. At each iteration, the search algorithm utilizes historically evaluated pairs as context to sample a new model architecture, which is subsequently evaluated to derive its performance metrics. A weighted sum of these metrics serves as the reward signal for the given architecture.

Instead of collecting AutoML joint optimization samples directly, which is time-consuming and computationally expensive, we sampled model-reward pairs from the established NAS dataset NAS-Bench-301 that includes approximately $1 0 ^ { 2 1 }$ distinct model architectures with their performance on CIFAR-10 classification task. To augment the diversity of the architectural search space within our optimization data, we introduced a pruning-based architecture sampling strategy that generates varied sub-spaces from a single overarching search space. As illustrated in Fig. 3, the comprehensive architecture search space $s$ is modeled as a fully connected DAG. We independently prune each edge with a probability $p \ = \ 0 . 5$ . This stochastic independent pruning inherently defines a localized sub-search space $\bar { \mathcal { S } } _ { s u b } \subset \mathcal { S }$ . Iterating this independent pruning generates a multitude of distinct sub-search spaces, yielding a training dataset with highly varied structural distributions.

![](images/a50c9b73326ab2ac46e8eaacf7bb16d6f7d8fbac926d280b74f0fb8b353a4cc9.jpg)  
Fig. 3. Pipeline of the subspace-sampled data curation strategy. Given an entire architecture search space modeled as a directed acyclic graph (DAG), stochastic edge pruning is applied to generate structurally diverse, localized sub-spaces $( S _ { s u b } )$ . Within each isolated sub-space, a subset of 40<sub>model architectures is sampled and evaluated to pair their unified NCode representations with corresponding performance metrics. To construct the</sub> training trajectories, these samples are partitioned into a sequential contextual history (H) of evaluated trials and a pool of unassessed candidates (C). This pipeline reframes the large-scale optimization problem into a tractable ranking task, training the model to select the candidate with the highest expected performance conditioned on historical context.

Given the immense scale of the architectural search space, which comprises approximately $1 0 ^ { 2 1 }$ architectures, identifying a single optimal architecture to serve as an absolute ground truth is computationally intractable and poses a significant learning challenge for the model. Consequently, we reframed the objective of the LLM-based searcher from direct prediction to candidate ranking. Within each isolated sub-space $\mathcal { S } _ { s u b }$ , we randomly sample a set of N models (where $N \in [ 1 0 0 , 2 0 0 ] $ ) alongside their corresponding performance metrics. The contextual history H, comprising a sequence of previously evaluated NCode representations and their rewards, is provided to the LLM-based searcher. The model is then tasked with selecting the most promising NCode from a candidate pool C of unassessed configurations sampled from the same sub-space.

Through this methodology, we curated a robust dataset comprising approximately 228,000 optimization trajectory instances. This extensive dataset boosts the LLM’s capacity to optimize AutoML configurations in a domain-agnostic manner.

## 3.4 Interpretability through LLM CoT Reasoning

To unveil the LLM-based searcher’s reasoning process for AutoML optimization, we utilized Group Relative Policy Optimization (GRPO) [10] to fine-tune the Qwen3-8B [16] LLM as self-explaining searcher which learns to navigate the optimal AutoML configuration guided by its own CoT reasoning. We then analyzed the CoT reasoning explained in natural language on the NAS-Bench-201 [5] and GenEval [17] datasets, extracting its reasoning trajectories during the search phase for subsequent interpretability analysis.

Policy Optimization for AutoML Search. We cast the LLMbased AutoML search as a Markov Decision Process (MDP) designed to optimize decision-making trajectories within the universal NCode space. For a given search context x (comprising historical trials and current candidates), the searcher acts as a policy $\pi _ { \theta }$ that generates a token sequence y representing the combined reasoning (CoT) and the final configuration selection. To efficiently navigate this complex discrete search space without the memory overhead of an explicit value network, we implement Group Relative Policy Optimization (GRPO) [10]. The reward for GRPO is a weighted sum of accuracy reward, verifying the selection of the optimal NCode candidate, and format reward, enforcing the structural integrity of the reasoning process required for interpretability analysis.

Taxonomy of Chain-of-Thought Reasoning Behaviors We employ LLM-based searcher with CoT capability to search on NAS-Bench-201 for image classification and GenEval for text-to-image generation. Experiments yielded 2,500 search steps complete with the model’s intermediate reasoning process on each task. To systematically analyze the searcher’s reasoning trajectories, we taxonomized the generated Chain-of-Thought outputs into six distinct behavioral patterns. The definitions of these patterns, as shown in Fig. 4, are as follows:

• Positional Anchoring: This pattern occurs when the searcher explicitly identifies that the presence of a specific operational digit at a designated sequence position is strongly correlated with extreme (either high or low) performance outcomes. The LLM uses this single-point heuristic as an anchor for subsequent candidate selection.

• Global Component Distribution: This applies when the searcher bases its decision on the global distribution of operations. The model observes that the aggregate count of specific functional digits across the entire configuration sequence systematically correlates with overall model performance, regardless of their absolute positions.

![](images/346d5f82e5e40b25d6534aa17d02e2b6be720b3530dedbafe9363a595fe932e5.jpg)  
Fig. 4. Representative examples of the LLM’s reasoning process categorized by six distinct analytical patterns (e.g., Positional Anchoring, Combinatorial Synergy).

• Neighborhood Mutation: We define this behavior as an evolutionary search strategy where the LLM selects a historically high-performing configuration as a baseline (“parent”) and targets candidates that represent minimal structural perturbations (e.g., modifying only one or two sequence digits). This reflects an exploitation tactic focused on exploring the immediate local optimum neighborhood.

• Combinatorial Synergy: This describes instances where the reasoning process emphasizes the combinatorial interaction between multiple positions. The searcher identifies that specific multi-digit combinations (e.g., a coupled pair of operations at positions 3 and 5) yield a synergistic effect on performance that cannot be explained by individual digits alone.

• Hierarchical Sequence Progression: This categorizes behavior where the LLM detects monotonically increasing or decreasing numerical patterns within specific sub-regions of the configuration string. The model interprets these ordinal trends as reflections of smooth or abrupt hierarchical transitions in the network topology that systematically impact the target metric.

• Operational Heterogeneity: This pattern emerges when the searcher evaluates candidates based on their digital heterogeneity. By observing that highperforming models typically encompass a broader vocabulary of unique functional digits compared to homogeneous, underperforming ones, the LLM deliberately favors configurations that integrate a diverse array of numerical components.

## 3.5 Zero-Shot Inference Pipeline

Once the LLM-based searcher is trained on the curated trajectory dataset (as detailed in Sec. 3.3). It can be directly deployed to jointly optimize architectures, hyperparameters, and data ratios for unseen domains, novel search spaces, and distinct modalities without any task-specific gradient updates. This zero-shot inference pipeline acts as a closedloop Markov Decision Process, illustrated in Fig. 5, and comprises the following core stages:

1) Cold-Start Initialization: To initialize the historical context for the LLM, we employ a Random Search strategy to sample a small set of initial configurations (e.g., $K _ { \mathrm { i n i t } } = 5 )$ . These configurations are mapped to their respective discrete sequences via the universal NCode <sup>ˆ</sup>f. They are subsequently evaluated on the target task to obtain their performance metrics, which are aggregated and normalized into scalar rewards to form the initial historical buffer H.

2) Candidate Pool Generation: At each search iteration, we generate a candidate pool C of unassessed configurations. To ensure both exploration and exploitation, this pool is populated by applying an evolutionary algorithm (i.e., structural mutations on historically highperforming NCodes) alongside random sampling.

3) LLM-Guided Reasoning and Selection: The current historical buffer H (comprising evaluated NCodereward pairs) and the candidate pool C are integrated into a standardized prompt. The frozen LLM-based searcher processes this context, identifies and outputs the most promising candidate c<sup>∗</sup> from the pool.

4) Proxy Evaluation and Context Update: To mitigate the prohibitive computational cost of fully training largescale models during the search phase, we implement an early-stopping proxy evaluation protocol. The selected candidate is decoded back into the actual machine learning pipeline and trained for a minimal number of steps or epochs (e.g., 1 epoch for segmentation, 2500 steps for text-to-image synthesis). The derived surrogate reward r<sup>∗</sup> is then paired with its NCode c<sup>∗</sup> and appended to the historical buffer H.

![](images/f073317ff015d7761bd614fc311ce5d24453103f61866c11f40f6b286a52addf.jpg)  
Fig. 5. Overview of our iterative LLM-based search framework. The training configurations encompassing hyperparameters, architectures, and data ratios are uniformly encoded as discrete numerical sequences named as NCodes. These NCodes are paired with their normalized performance metrics to serve as rewards, forming a continuously updated historical buffer of evaluated trials. During the search phase, a combination of an evolutionary algorithm and random search generates a pool of new candidate AutoML configurations. The LLM-based searcher analyzes the historical trials and identifies the most promising candidate from the candidate pool. The selected candidate undergoes training and evaluation, and the resulting performance metric is stored in the historical buffer. This iterative loop with NCodes is task- and domain-agnostic, enabling the LLM to zero-shot adapt to new AutoML domains without task-specific fine-tuning.

5) Bounded Buffer Management: To respect the LLM’s maximum token context window and maintain highquality context, the capacity of the historical buffer is bounded. Once the number of evaluated pairs exceeds a predefined threshold (e.g., M = 200), a filtering mechanism is triggered to retain only the top-M performing configurations, discarding sub-optimal trials.

This iterative cycle continues until a predefined computational budget is exhausted. The overall procedure is formalized in Algorithm 1.

## 4 EXPERIMENTS

## 4.1 Overview

To comprehensively evaluate the effectiveness, zero-shot transferability, scalability, and interpretability of our proposed LLM-based searcher, we conduct extensive empirical evaluations across four distinct machine learning scenarios. Specifically, we evaluate our framework on: (1) Cross-Domain Image Segmentation to test PEFT design and hyperparameter optimization under zero-shot domain shifts; (2) Cross-Modality Speaker Verification to assess cross-modal generalization from vision to audio; (3) Large-Scale Text-to-Image Synthesis to demonstrate joint architecture, hyperparameter, and data-ratio search at scale; and (4) NAS Benchmark (NAS-Bench-201) to verify standard neural architecture search performance. For each paradigm, we present the detailed experimental settings followed by the quantitative results. Furthermore, we investigate the interpretability of the LLM-based searcher’s internal decision-making process, supported by thorough ablation studies and qualitative case studies. Below, we first outline the common implementation details before detailing each specific task.

## Implementation Details

• LLM-based Searcher Training: To build the primary search agent, the LLaMA-3.1-8B base model [18] undergoes full-parameter supervised fine-tuning (SFT). The training is anchored on the curated dataset of approximately 228,000 optimization trajectory instances. To robustly capture expansive sequences of historical trial rewards alongside candidate configurations, we employ an extended context window with a strict token cutoff length of 20,480. The SFT process is executed for 3 epochs with a learning rate of $1 . 0 \times 1 0 ^ { - 5 }$ , regulated by a cosine learning rate scheduler following a 10% warmup phase. To ensure memory efficiency during full-parameter optimization, training is distributed across a single node equipped with 8 GPUs using the DeepSpeed ZeRO-2 framework in bfloat16 mixed precision, configured with a per-device batch size of 1 and a gradient accumulation step of 2.

To analyze the searcher’s internal decision-making process, we post-train a Qwen3-8B model using Group Relative Policy Optimization (GRPO). For this reinforcement learning stage, we subsample 2,000 representative trajectory instances from the primary 228,000-instance dataset and execute policy training for 3 epochs. To facilitate relative advantage estimation within the algorithmic framework of GRPO, the policy model is configured to generate a group of 8 independent candidate responses for each individual prompt sample. The computationally intensive training is scaled across a distributed 4-node cluster equipped with 8 GPUs per node (totaling 32 GPUs), running the DeepSpeed ZeRO-3 framework with CPU offloading in bfloat16 mixed precision. The execution configuration utilizes a per-device batch size of 1 matched with a gradient accumulation step of 4. To mitigate the immense token overhead of long-form Chain-of-Thought (CoT) generation while preserving search alignment, we employ Parameter-Efficient Fine-Tuning (PEFT) via Low-Rank Adaptation (LoRA) targeted at the attention projection modules (q\_proj and v\_proj). The maximum prompt and completion boundaries are constrained to 10,240 and 8,192 tokens respectively, allowing sufficient capacity for structural rationalization. A vLLM inference engine is colocated with the training framework to accelerate group-based relative reward sampling.

Algorithm 1 LLM-based Zero-Shot AutoML Search Pipeline   
Input: Target task $\tau ,$ search space $s ,$ universal NCode   
encoder ${ \hat { f } } ,$ proxy evaluator $E _ { p r o x y } ,$ frozen LLM policy $\pi _ { \theta } ,$   
search budget $T ,$ buffer capacity M, initial samples $K _ { i n i t }$   
Output: The optimal AutoML configuration   
$a ^ { * }$   
1: Initialize historical buffer H $ \emptyset$   
2: // Stage 1: Cold-Start Initialization   
3: for $i = 1$ to $K _ { i n i t }$ do   
4: Sample configuration $a _ { i } \sim$ Random(S)   
5: Encode NCode sequence: $c _ { i } \gets \hat { f } ( a _ { i } )$   
6: Proxy evaluation on target task: $r _ { i } \gets E _ { p r o x y } ( a _ { i } , \mathcal { T } )$   
7: Update history: ${ \mathcal { H } } \gets { \tilde { \mathcal { H } } } \cup \{ ( c _ { i } , r _ { i } ) \}$   
8: end for   
9: // Stage 2: Iterative Search Phase   
10: for $t = K _ { i n i t } + 1$ to $T$ do   
11: Generate candidate pool $A _ { c a n d }$ via Evolutionary $\&$   
Random Search   
12: Encode pool: $\mathcal { C } _ { c a n d } \gets \{ \hat { f } ( a ) \ | \ a \in \mathcal { A } _ { c a n d } \}$   
13: // Stage 3: LLM-Guided Selection   
14: Construct prompt string $\mathcal { P }$ using H and $\mathcal { C } _ { c a n d }$   
15: LLM reasoning and selection: $c ^ { * }  \pi _ { \theta } ( \mathcal { P } )$   
16: Decode to target configuration: $a ^ { * }  \hat { f } ^ { - 1 } ( c ^ { * } )$   
17: // Stage 4: Proxy Evaluation   
18: Execute early-stopping evaluation: $r ^ { * }$ ←   
$E _ { p r o x y } ( a ^ { * } , { \mathcal { T } } )$   
19: Update history: ${ \mathcal { H } } \gets { \mathcal { H } } \cup \{ ( c ^ { * } , r ^ { * } ) \}$   
20: // Stage 5: Bounded Buffer Management   
21: if $\vert \mathcal { H } \vert > M$ then   
22: H ← Top-M elements in H ranked by r   
23: end if   
24: end for   
25: return $a _ { b e s t } ~  ~ \mathrm { a r g } \mathrm { m a x } _ { ( c , r ) \in \mathcal { H } } r$ (Decoded back to   
config space)

• Search Phase: To initialize the AutoML configuration search within our framework, we utilized Random Search to sample and evaluate five configurations. This generated an initial set of five NCode-reward pairs for the LLM-based searcher to analyze. In each subsequent iteration, the LLM proposed an NCode, which was then decoded into a specific training configuration. To ensure computational efficiency during searches for image segmentation and speaker verification, we applied early stopping after a single epoch and used the partially trained model to derive a surrogate performance measure. For large-scale text-to-image synthesis, we jointly searched model architectures, training hyperparameters, and data sampling ratios using a surrogate model scaled down to half the block depth and hidden dimension size of the target model, with training truncated at 2,500 steps. The overall reward was calculated as the weighted average of all metrics and subsequently normalized across all evaluated configurations. Notably, for metrics where lower values denote superior performance, their contribution to the final reward was negated. Throughout the search, we maintained a historical buffer of up to 200 NCodereward pairs. Once the evaluated pairs exceeded this threshold, we filtered the context provided to the LLM, retaining only the top-200 performing configurations.

• Retraining Phase: Once the optimal configurations are identified by the LLM-based searcher during the earlystopping proxy evaluation, the selected configurations undergo a complete retraining phase for rigorous evaluation. Since these optimal configurations (encompassing architectures, hyperparameters, and data ratios) represent the direct outcomes of our search process, the specific retraining implementation details are inherently task-specific. Therefore, we detail the complete retraining setups within the respective Search Configuration sections of each task (Sections 4.2–4.4).

## 4.2 Cross-Domain Image Segmentation

## 4.2.1 Experimental Setting

• Datasets. To evaluate the cross-task and cross-domain generalizability of our proposed LLM-based searcher, we utilized a diverse suite of image segmentation datasets encompassing multiple distinct domains. The search phase was conducted in the medical imaging domain using the Kvasir-SEG and CVC-612 datasets. Specifically, Kvasir-SEG and CVC-612 evaluate the model’s capability in gastrointestinal polyp segmentation using endoscopic and colonoscopy imagery, respectively. To test zero-shot transferability, the optimal configuration discovered during search was evaluated on seven datasets across domains. In addition to the two datasets, these include ISIC2017, which focuses on the precise boundary delineation of skin lesions for melanoma detection; CAMO, which features natural imagery of camouflaged objects that artfully blend into their surroundings; SBU, which tasks the model with differentiating true structural boundaries from complex shadow masks in everyday scenes; Leaf, an agricultural dataset focused on delineating individual plant leaves from overlapping foliage backgrounds; and Road, which utilizes high-resolution remote sensing imagery to evaluate the extraction of continuous topological road networks across vast landscapes.

TABLE 1  
Joint optimization of model architecture and hyperparameters achieves state-of-the-art zero-shot transferability across diverse image segmentation domains. Results are reported as mean ± standard error over five independent runs. LM-Searcher performs architecture search in a zero-shot setting with respect to the target segmentation domain. Higher values indicate better performance for $S _ { \alpha } , E _ { \phi } ,$ Jac, $F _ { \beta } ^ { \omega }$ , and IoU, whereas lower values indicate better performance for BER. Results for the architecture-only configuration, denoted as Ours $( \mathsf { A r c } ) [ \breve { \mathsf { 9 } } ]$ , are adopted from our preliminary conference version [9] to serve as a baseline for demonstrating the benefits of joint optimization
<table><tr><td rowspan="2">Method</td><td colspan="2">Kvasir-SEG</td><td colspan="2"> $\overline { { \mathrm { C V C } { \cdot } 6 1 2 } }$ </td><td colspan="2"> $\overline { { \mathrm { I S I C } 2 0 1 7 } }$ </td><td colspan="2"> $\overline { { \mathrm { C A M O } } }$ </td><td> $\overline { { \mathrm { { \_ { S B U } } } } }$ </td><td>Leaf</td><td>Road</td></tr><tr><td> $S _ { \alpha } \uparrow$ </td><td> $E _ { \phi } \uparrow$ </td><td> $S _ { \alpha } \uparrow$ </td><td> $E _ { \phi } \uparrow$ </td><td> $\mathrm { J a c } \uparrow$ </td><td> $S _ { \alpha } \uparrow$ </td><td> $E _ { \phi } \uparrow$ </td><td> $F _ { \beta } ^ { \omega } \uparrow$ </td><td> $\mathrm { B E R } \downarrow$ </td><td> $\mathrm { I o U } \uparrow$ </td><td> $\mathrm { I o U \uparrow }$ </td></tr><tr><td>BitFit [19]</td><td> $\overline { { 9 0 . 8 \pm 0 . 5 7 } }$ </td><td> $\overline { { 9 3 . 8 \pm 0 . 9 8 } }$ </td><td> $\overline { { 8 9 . 0 \pm 0 . 4 0 } }$ </td><td> $\overline { { 9 1 . 6 \pm 0 . 9 8 } }$ </td><td> $\overline { { 7 6 . 4 \pm 0 . 4 5 } }$ </td><td> $\overline { { 8 6 . 8 \pm 0 . 3 3 } }$ </td><td> $\overline { { 9 0 . 7 \pm 0 . 2 8 } }$ </td><td> $\overline { { 8 1 . 5 \pm 0 . 1 9 } }$ </td><td> $\overline { { 3 . 2 \pm 0 . 1 3 } }$ </td><td> $\overline { { 7 1 . 4 \pm 1 . 1 5 } }$ </td><td> $\overline { { 6 0 . 6 \pm 0 . 1 5 } }$ </td></tr><tr><td>Adapter [20]</td><td> $9 1 . 2 \pm 0 . 2 3$ </td><td> $9 4 . 0 \pm 0 . 1 6$ </td><td> $8 9 . 3 \pm 0 . 4 3$ </td><td> $9 2 . 0 \pm 0 . 6 3$ </td><td> $7 6 . 7 \pm 0 . 6 6$ </td><td> $8 7 . 7 \pm 0 . 1 0$ </td><td> $9 1 . 3 \pm 0 . 4 0$ </td><td> $8 2 . 8 \pm 0 . 3 5$ </td><td> $2 . 8 \pm 0 . 0 9$ </td><td> $7 2 . 1 \pm 0 . 4 7$ </td><td> $6 1 . 5 \pm 0 . 1 1$ </td></tr><tr><td>VPT [21]</td><td> $9 1 . 5 \pm 0 . 2 3$ </td><td> $9 4 . 3 \pm 0 . 0 6$ </td><td> $9 1 . 0 \pm 0 . 9 4$ </td><td> $9 3 . 7 \pm 1 . 4 1 $ </td><td> $7 6 . 9 \pm 0 . 9 4$ </td><td> $8 7 . 4 \pm 0 . 6 0$ </td><td> $9 1 . 4 \pm 0 . 6 8$ </td><td> $8 2 . 1 \pm 0 . 7 5$ </td><td> $2 . 7 \pm 0 . 0 6$ </td><td> $7 3 . 6 \pm 0 . 2 6$ </td><td> $6 0 . 2 \pm 1 . 8 7$ </td></tr><tr><td>LST [22]</td><td> $8 9 . 7 \pm 0 . 2 5$ </td><td> $9 3 . 3 \pm 0 . 3 7$ </td><td> $8 9 . 4 \pm 0 . 3 7$ </td><td> $\overline { { 9 2 . 4 \pm 0 . 5 4 } }$ </td><td> $7 6 . 4 \pm 1 . 0 5$ </td><td> $8 3 . 3 \pm 0 . 2 8$ </td><td> $8 8 . 0 \pm 0 . 2 3 $ </td><td> $7 7 . 1 \pm 0 . 0 2$ </td><td> $3 . 2 \pm 0 . 0 1$ </td><td> $7 0 . 2 \pm 0 . 8 7$ </td><td> $6 0 . 2 \pm 0 . 2 6$ </td></tr><tr><td>SAM-Adapter [23]</td><td> $8 9 . 6 \pm 0 . 2 4$ </td><td> $9 2 . 5 \pm 0 . 1 0$ </td><td> $8 9 . 6 \pm 0 . 2 2$ </td><td> $9 2 . 4 \pm 1 . 0 6$ </td><td> $7 6 . 1 \pm 0 . 4 5$ </td><td> $8 5 . 6 \pm 0 . 2 6$ </td><td> $8 9 . 6 \pm 0 . 5 5$ </td><td> $7 9 . 8 \pm 0 . 8 9$ </td><td> $3 . 1 \pm 0 . 0 6$ </td><td> $7 1 . 4 \pm 0 . 2 0$ </td><td> $6 0 . 6 \pm 0 . 0 6$ </td></tr><tr><td>SSF [24]</td><td> $9 1 . 3 \pm 0 . 8 7$ </td><td> $9 3 . 9 \pm 1 . 4 9$ </td><td> $8 9 . 6 \pm 0 . 3 7$ </td><td> $9 1 . 9 \pm 0 . 7 9$ </td><td> $7 6 . 6 \pm 0 . 1 9$ </td><td> $8 7 . 5 \pm 0 . 1 1$ </td><td> $9 1 . 4 \pm 0 . 1 6$ </td><td> $8 2 . 6 \pm 0 . 1 2$ </td><td> $3 . 2 \pm 0 . 0 5$ </td><td> $7 1 . 5 \pm 0 . 6 3$ </td><td> $6 1 . 6 \pm 0 . 0 3$ </td></tr><tr><td>LoRA [25]</td><td> $9 1 . 2 \pm 0 . 2 8$ </td><td> $9 3 . 8 \pm 0 . 2 2$ </td><td> $9 0 . 7 \pm 0 . 0 4$ </td><td> $9 2 . 5 \pm 0 . 4 1$ </td><td> $7 6 . 6 \pm 0 . 2 3$ </td><td> $8 8 . 0 \pm 0 . 2 4 $ </td><td> $9 1 . 9 \pm 0 . 4 2$ </td><td> $8 2 . 8 \pm 0 . 1 6$ </td><td> $2 . 7 \pm 0 . 0 8$ </td><td> $7 3 . 7 \pm 0 . 2 0$ </td><td> $6 2 . 2 \pm 0 . 2 1$ </td></tr><tr><td> $\mathrm { C o n v - L o R A } \left[ 2 6 \right]$ </td><td> $9 1 . 9 \pm 0 . 6 4$ </td><td> $9 4 . 3 \pm 0 . 7 7$ </td><td> $9 0 . 1 \pm 0 . 3 1$ </td><td> $9 2 . 3 \pm 0 . 4 2$ </td><td> $7 7 . 0 \pm 0 . 5 1$ </td><td> $8 8 . 6 \pm 0 . 1 2 $ </td><td> $9 2 . 4 \pm 0 . 1 7$ </td><td> $8 3 . 1 \pm 0 . 1 9$ </td><td> $2 . 8 \pm 0 . 0 2$ </td><td> $7 3 . 9 \pm 0 . 2 0$ </td><td> $6 2 . 1 \pm 0 . 2 3$ </td></tr><tr><td>Ours (Arc) [9]</td><td> $9 2 . 4 \pm 0 . 4 3$ </td><td> $9 4 . 7 \pm 0 . 4 7$ </td><td> $9 1 . 1 \pm 0 . 4 7$ </td><td> $9 3 . 2 \pm 0 . 7 0$ </td><td> $\overline { { 7 7 . 6 \pm 0 . 4 0 } }$ </td><td> $\underline { { 8 8 . 9 \pm 0 . 0 4 } }$ </td><td> $9 2 . 7 \pm 0 . 1 5$ </td><td> $\underline { { 8 3 . 9 \pm 0 . 3 2 } }$ </td><td> $2 . 6 \pm 0 . 0 4$ </td><td> $7 4 . 2 \pm 0 . 2 8$ </td><td> $\underline { { 6 2 . 6 \pm 0 . 0 4 } }$ </td></tr><tr><td>Ours (Arc+Hyp)</td><td> $\mathbf { 9 2 . 7 \pm 0 . 4 3 }$ </td><td> $\mathbf { 9 5 . 5 \pm 0 . 5 5 }$ </td><td> $\mathbf { 9 2 . 6 \pm 0 . 5 6 }$ </td><td> $\mathbf { 9 4 . 3 \pm 0 . 8 0 }$ </td><td> ${ \bf 7 7 . 6 \pm 0 . 5 7 }$ </td><td> $\overline { { { \bf 8 9 . 3 \pm 0 . 2 6 } } }$ </td><td> $\bf { 9 3 . 0 \pm 0 . 1 2 }$ </td><td> $\mathbf { 8 3 . 9 \pm 0 . 3 2 }$ </td><td> $\overline { { { \bf 2 . 6 \pm 0 . 0 5 } } }$ </td><td> $\overline { { 7 5 . 0 \pm 0 . 2 8 } }$ </td><td> $\pm \bf { 0 . 2 . 6 \pm 0 . 1 1 }$ </td></tr></table>

(BER) is specifically applied to evaluate the performance on the SBU dataset. It ensures an unbiased assessment for shadow detection by equally weighting the classification error rates of both shadow and non-shadow regions.

• Search Configuration. The search space comprises both model architectures and training hyperparameters. For architecture, we used the Segment Anything Model (SAM) as the base segmenter and searched the PEFT design by assigning the ranks of 32 LoRA modules in the ViT encoder from {3, 6, 12, 24}, yielding a search space of approximately $\mathrm { i 0 ^ { 1 9 } }$ architectures. The hyperparameter search space incorporates the focal loss γ, weight decay, and learning rate. To efficiently navigate this expansive space, we implement an early-stopping proxy evaluation protocol during the search phase, where each candidate configuration is trained for only 1 epoch to derive a surrogate reward. The discovered optimal configuration is subsequently subjected to full retraining for 30 epochs, establishing a highly efficient searchto-full-training workload ratio of 1:30. Comprehensive details regarding the search space are provided in $\mathsf { A p - }$ pendix Table 1. For the subsequent retraining phase, we applied the identified best-performing AutoML configurations to each downstream dataset. Specifically, we fine-tuned the Segment Anything Model (SAM) [27] using our optimal LoRA [25] configuration. Training was conducted on a single GPU, utilizing a global batch size of $^ { 4 , }$ a learning rate of $1 \times 1 0 ^ { - 3 } .$ , a weight decay of $3 \times 1 0 ^ { - 5 }$ , and $\mathfrak { a } \gamma$ value of 2.0.

• Evaluation Metrics. To comprehensively assess the zero-shot transferability of the searched configurations across diverse target domains, we employ a multifaceted evaluation protocol tailored to the morphological characteristics of each dataset. For all metrics except the Balanced Error Rate (BER), higher values indicate superior performance.

– Structural and Alignment Metrics: For datasets requiring strict foreground object consistency and structural integrity (Kvasir-SEG, CVC-612, and CAMO), we utilize the S-measure $( S _ { \alpha } )$ [28] and Enhanced-alignment measure $( E _ { \phi } )$ [29].

– Spatial Overlap and Region Metrics: For tasks where exact spatial overlap is the most critical success factor (ISIC2017, Leaf, and Road), we rely on the widely adopted Jaccard Index (Jac). Additionally, the weighted F-measure $( F _ { \beta } ^ { \omega } )$ is evaluated across multiple datasets to adaptively emphasize errors in highly salient regions.

Detailed mathematical formulations for all the aforementioned evaluation metrics are provided in $\mathsf { A p - }$ pendix Section 2.

– Balanced Error Metrics: The Balanced Error Rate

## 4.2.2 Experimental Results

As detailed in Table 1, the optimal configuration discovered by our LLM-based searcher, denoted as Ours (Arc+Hyp), establishes a new state-of-the-art across all evaluated datasets and metrics. Notably, despite the search phase being exclusively anchored in the medical imaging domain, the resulting configuration exhibits competitive zero-shot generalizability, successfully transferring to natural (CAMO, SBU), agricultural (Leaf), and remote sensing (Road) imagery. It consistently outperforms both domain-agnostic PEFT baselines (e.g., LoRA [25], Adapter, VPT [21]) and task-specific adaptations (e.g., SAM-Adapter [23], Conv-LoRA [26]). For instance, on the $C \mathrm { V C } { - 6 1 2 }$ dataset, our method achieves an $S _ { \alpha }$ of 92.6 and an $E _ { \phi }$ of 94.3, yielding absolute improvements of 2.5 and 2.0 percentage points, respectively, over the strongest baseline, Conv-LoRA [26]. Furthermore, ablation studies validate our joint search paradigm: simultaneously searching architecture and hyperparameters (Ours (Arc+Hyp)) consistently surpasses architecture-only search (Ours (Arc)) across all benchmarks.

## 4.3 Cross-Modality Speaker Verification

## 4.3.1 Experimental Setting

• Datasets. We validated the cross-modality generalization capability of our LLM-based searcher on the audio modality evaluation suite 3D-Speaker-Toolkit [35]. The 3D-Speaker corpus is a large-scale dataset containing over 10,000 speakers, which are partitioned into training and test sets. We evaluated the models across three challenging tracks: multi-device, multi-distance, and multi-dialect. The evaluation split comprises 240 unseen speakers, 18,782 utterances, 11 dialects, and 15.42 hours of speech.

• Search Configuration. Our framework navigates a comprehensive search space that comprises both model architectures and training hyperparameters for the speaker verification task. The architectural search space dictates the model’s capacity and feature resolution, specifically targeting the Mel-filterbank input dimensions, the base channel width of the ERes2NetV2 backbone, and the final speaker embedding size. Concurrently, the hyperparameter search space includes the data augmentation probability, learning rate, optimizer momentum, and weight decay. To maintain computational efficiency across this joint space, we implement an early-stopping proxy evaluation protocol during the search phase, truncating each candidate’s training at only 1 epoch to derive a surrogate reward. The discovered optimal configuration is subsequently subjected to full retraining for a total of 77 epochs—comprising 70 epochs of standard training followed by a 7-epoch large-margin fine-tuning stage—establishing a highly efficient search-to-full-training workload ratio of approximately 1:77. The exact search space for each design choice is detailed in Appendix Table 1. For the retraining phase, the optimal model architecture was trained on the 3D-Speaker-Toolkit dataset for 70 epochs. We employed a cosine learning rate scheduler with an initial learning rate of 0.3, a weight decay of $5 \times 1 0 ^ { - 5 }$ , a momentum of 0.85, and a data augmentation probability of 0.4. The resulting model was subsequently finetuned utilizing a large-margin strategy for an additional seven epochs, maintaining the same hyperparameter settings.

TABLE 2  
The proposed searcher demonstrates superior zero-shot cross-modality generalization on the 3D-Speaker verification benchmark. Lower values indicate better performance for all metrics. “Average” denotes the mean EER across the cross-device, cross-distance, and cross-dialect tracks.
<table><tr><td rowspan="2">Method</td><td colspan="2">Multi-Device</td><td colspan="2">Multi-Distance</td><td colspan="2">Multi-Dialect</td><td colspan="2">Average</td></tr><tr><td>EER(%)↓</td><td>MinDCF↓</td><td>EER(%)↓</td><td>MinDCF↓</td><td>EER(%)↓</td><td>MinDCF↓</td><td>EER(%)↓</td><td>MinDCF↓</td></tr><tr><td>Res2Net [30]</td><td>8.03</td><td>0.707</td><td>9.67</td><td>0.781</td><td>14.11</td><td>0.920</td><td>10.60</td><td>0.802</td></tr><tr><td>ResNet34 [31]</td><td>7.29</td><td>0.689</td><td>8.98</td><td>0.762</td><td>12.81</td><td>0.906</td><td>9.69</td><td>0.786</td></tr><tr><td>ECAPA-TDNN [32]</td><td>8.55</td><td>0.728</td><td>12.15</td><td>0.814</td><td>12.24</td><td>0.915</td><td>10.98</td><td>0.819</td></tr><tr><td>ERes2Net [33]</td><td>7.12</td><td>0.657</td><td>9.82</td><td>0.749</td><td>12.10</td><td>0.866</td><td>9.68</td><td>0.757</td></tr><tr><td> $\mathrm { C A M + + } \ [ 3 4 ]$ </td><td>7.17</td><td>0.669</td><td>9.84</td><td>0.722</td><td>11.78</td><td>0.844</td><td>9.60</td><td>0.745</td></tr><tr><td>ERes2NetV2 [35]</td><td>6.82</td><td>0.628</td><td>9.35</td><td>0.723</td><td>11.93</td><td>0.826</td><td>9.37</td><td>0.726</td></tr><tr><td>OptFormer [4]</td><td>7.05</td><td>0.668</td><td>11.54</td><td>0.802</td><td>12.01</td><td>0.842</td><td>10.20</td><td>0.771</td></tr><tr><td>Ours</td><td>6.88</td><td>0.629</td><td>9.10</td><td>0.697</td><td>11.66</td><td>0.819</td><td>9.21</td><td>0.715</td></tr></table>

• Evaluation Metrics. We assessed the performance of the discovered configurations utilizing two standard metrics within the speaker verification domain: the Equal Error Rate (EER) and the minimum of the normalized detection cost function (minDCF). For both metrics, lower values signify superior detection and recognition performance. The optimal configuration was fully retrained and evaluated across five independent runs, from which we report the mean and standard deviation.

## 4.3.2 Experimental Results

As reported in Table 2, our approach achieves an Equal Error Rate (EER) of 9.21 and a Minimum Detection Cost Function (minDCF) of 0.715 on average across all tracks, notably surpassing the performance of the current state-of-the-art manually-tuned ERes2NetV2. Furthermore, our framework significantly outperforms the competitive AutoML baseline,

OptFormer [4]<sup>1</sup>, which attains a higher average EER of 10.20 and a minDCF of 0.771. This demonstrates that our method can effectively generalize across modalities without any further fine-tuning, considering the LLM-based searcher was trained solely on image modality data and zero-shot transferred to the 3D-Speaker audio dataset [35].

![](images/ce8c1d5ccf98767a434ec9e4f7ebd37a4512868095e049f4714806a03e34eac2.jpg)  
Fig. 6. Performance comparison of our image generative model using configurations optimized by the proposed LLM-based searcher against baseline methods on the GenEval benchmark. Our method achieves the highest overall score while using significantly fewer parameters than competing state-of-the-art models.

## 4.4 Large-Scale Text-to-Image Synthesis

## 4.4.1 Experimental Setting

• Heterogeneous Training Corpus. To train the DiT model, we constructed a diverse image-text corpus and dynamically search for the optimal data distribution ratios during training. This corpus amalgamates several high-quality datasets, each providing image-caption pairs with unique visual-semantic priors:

– BLIP3o (Short, Long, JourneyDB, and 60k variants) [50]: Provides highly descriptive, synthetically enriched captions that enhance fine-grained semantic alignment.

TABLE 3  
The optimized Diffusion Transformer achieves state-of-the-art compositional text-to-image alignment on the GenEval benchmark with high parameter efficiency. Arrows (↑) indicate that higher scores correspond to better performance.
<table><tr><td>Model</td><td># Params</td><td>Single object ↑</td><td>Two object ↑</td><td>Counting ↑</td><td>Colors ↑</td><td>Position ↑</td><td>Attribute binding ↑</td><td>Overall ↑</td></tr><tr><td>SDv2.1 [36]</td><td>0.9B</td><td>0.98</td><td>0.51</td><td>0.44</td><td>0.85</td><td>0.07</td><td>0.17</td><td>0.50</td></tr><tr><td>SDXL [37]</td><td>2.6B</td><td>0.96</td><td>0.74</td><td>0.39</td><td>0.85</td><td>0.15</td><td>0.23</td><td>0.55</td></tr><tr><td>SD3-medium [38]</td><td>2B</td><td>0.99</td><td>0.94</td><td>0.72</td><td>0.89</td><td>0.33</td><td>0.60</td><td>0.74</td></tr><tr><td>FLUX.1-dev [39]</td><td>12B</td><td>0.99</td><td>0.81</td><td>0.79</td><td>0.74</td><td>0.20</td><td>0.47</td><td>0.67</td></tr><tr><td>Emu3-Gen [40]</td><td>8B</td><td>0.98</td><td>0.71</td><td>0.34</td><td>0.81</td><td>0.17</td><td>0.21</td><td>0.54</td></tr><tr><td>JanusFlow [41]</td><td>2B</td><td>0.97</td><td>0.59</td><td>0.45</td><td>0.83</td><td>0.53</td><td>0.42</td><td>0.63</td></tr><tr><td>Janus-Pro-7B [42]</td><td>7B</td><td>0.99</td><td>0.89</td><td>0.59</td><td>0.90</td><td>0.79</td><td>0.66</td><td>0.80</td></tr><tr><td>OmniGen [43]</td><td>3.8B</td><td>0.98</td><td>0.84</td><td>0.66</td><td>0.74</td><td>0.40</td><td>0.43</td><td>0.68</td></tr><tr><td>BAGEL [44]</td><td>7B</td><td>0.98</td><td>0.95</td><td>0.84</td><td>0.95</td><td>0.78</td><td>0.77</td><td>0.88</td></tr><tr><td>Lumina-Image 2.0 [45]</td><td>2.6B</td><td></td><td>0.87</td><td>0.67</td><td></td><td></td><td>0.62</td><td>0.73</td></tr><tr><td>HiDream-I1-Full [46]</td><td>17B</td><td>1.00</td><td>0.98</td><td>0.79</td><td>0.91</td><td>0.60</td><td>0.72</td><td>0.83</td></tr><tr><td>Qwen-Image [47]</td><td>20B</td><td>0.99</td><td>0.92</td><td>0.89</td><td>0.88</td><td>0.76</td><td>0.77</td><td>0.87</td></tr><tr><td>GPT Image 1 [High] [48]</td><td></td><td>0.99</td><td>0.92</td><td>0.85</td><td>0.92</td><td>0.75</td><td>0.61</td><td>0.84</td></tr><tr><td>Seedream 3.0 [49]</td><td></td><td>0.99</td><td>0.96</td><td>0.91</td><td>0.93</td><td>0.47</td><td>0.80</td><td>0.84</td></tr><tr><td>OptFormer [4]</td><td>4B</td><td>0.99</td><td>0.90</td><td>0.76</td><td>0.92</td><td>0.73</td><td>0.74</td><td>0.84</td></tr><tr><td>Ours</td><td>1.1B</td><td>0.99</td><td>0.96</td><td>0.85</td><td>0.96</td><td>0.86</td><td>0.85</td><td>0.91</td></tr></table>

– LAION-high-resolution [51]: A large-scale, webscraped dataset ensuring broad coverage of realworld concepts and natural image distributions.

– Midjourney-23m-2: Comprises high-fidelity, artistically curated synthetic images to elevate the overall aesthetic quality of the generated outputs.

– X2I2 [52]: A dataset designed to improve complex instruction-guided generation.

– Echo-4o [53]: A synthetic dataset generated by GPT-4o, providing clean supervision to complement rare scenarios and complex compositions less represented in real-world data.

• Search Configuration. Our LLM-based searcher navigates a unified configuration space to jointly optimize the model architecture, training hyperparameters, and data sampling ratios. Architecturally, we adopt the MM-DiT as the foundational backbone, searching over the hidden dimension size, as well as the depths of the single- and double-stream blocks. The hyperparameter search space encompasses the learning rate, weight decay, and the optimizer parameters β and ϵ. Finally, to optimize the training data composition, the sampling weight for each dataset during training is independently tuned within a range of 1.0 to 6.0. To scale this framework to the immense computational demands of large-scale pre-training, we execute the search phase using a surrogate model scaled down to half the block depth and hidden dimension size of the target model, with training truncated at 2,500 steps. The optimal configuration is then scaled up to the full 1.3-billion-parameter architecture and trained for a total of 100,000 steps across three distinct resourceintensive stages, establishing a highly efficient searchto-full-training step ratio of approximately 1:40. Detailed descriptions of the search space are provided in Appendix Table 3. For the full retraining phase, we conducted a large-scale pre-training experiment, training a 1.3-billion-parameter model across 128 GPUs for approximately seven days. The training process consisted of three stages. In the first stage, all training images were resized to a $2 5 6 \times 2 5 6$ resolution. The model was trained for 85,000 steps on BLIP3o-JourneyDB, BLIP3o-Long, BLIP3o-Short, LAION-highresolution, Midjourney-23m-2, and X2I2, with data sampling ratios of 1.0, 2.0, 3.0, 5.0, 1.0, and 4.0, respectively. We utilized a learning rate of $1 \times 1 0 ^ { - 3 } .$ , a β<sub>2</sub> parameter of 0.99, an ϵ of $1 \times 1 0 ^ { - 1 5 }$ , and a weight decay of 0.1. In the second stage, we scaled the image resolution to $5 1 2 \times 5 1 2$ and continually trained the model for 5,000 steps using identical hyperparameters and data ratios. Finally, the model was fine-tuned on the BLIP3o-60k and Echo-4o datasets for 10,000 steps.

• Evaluation Metrics. We benchmarked the optimized DiT model using GenEval [17] to rigorously assess fine-grained compositional alignment at the instance level. We systematically evaluated the model across fundamental visual primitives, including single/multiobject rendering, object counting, spatial positioning, coloring, and attribute binding. The overall GenEval score is reported as the unweighted macro-average of the accuracies across these distinct sub-tasks, ensuring equal evaluation weight for each compositional skill. For evaluation, we generate four samples per prompt using an inference step number of 50 and a Classifier-Free Diffusion Guidance (CFG) scale of 5.0.

## 4.4.2 Experimental Results

Table 3 illustrates the performance comparison between our optimized DiT model and established text-to-image synthesis works on the GenEval [17] metric. Notably, our approach achieves the highest GenEval score of 0.91 compared with existing state-of-the-art methods. We also benchmarked our approach against another AutoML method, OptFormer [4], within the identical search space, outperforming it by an absolute margin of 0.07. Moreover, our model demonstrates remarkable parameter and data efficiency: despite comprising only 1.3 billion parameters and being trained on merely 50 billion tokens, it exceeds the performance of the 20- billion-parameter Qwen-Image model by 0.04. Furthermore, it surpasses the BAGEL model, which relies on 7 billion parameters and a massive 5 trillion training tokens, by a margin of 0.03 (Fig. 6). These findings underscore the robust generalization capability and scalability of our method in large-scale machine learning training scenarios.

## 4.5 NAS Benchmark

## 4.5.1 Experimental Setting

• Datasets. While traditional NAS methods are highly specialized for image classification tasks, we benchmark our approach on NAS-Bench-201 to verify that our domain-agnostic LLM-based searcher matches their performance in standard image classification, serving as a prerequisite for our open-world AutoML evaluations. Every architecture within the NAS-Bench-201 search space has been fully trained and evaluated across three distinct image classification datasets: CIFAR-10, CIFAR-100, and ImageNet-16-120. By querying this predefined dataset, we bypass the immense computational overhead of repetitively training candidate architectures from scratch.

• Search Space Configuration. NAS-Bench-201 offers an algorithm-agnostic, fixed search space based on a neural cell topology represented as a DAG. The cell architecture comprises four nodes and five predefined operation options (zeroize, skip connection, 1-by-1 convolution, 3-by-3 convolution, and 3-by-3 average pooling), which yields a comprehensive search space of 15,625 unique neural architecture candidates.

• Evaluation Protocol. We executed five independent search trials of 200 steps per dataset. The reported metrics represent the average classification accuracy across the five top-tier architectures, each selected based on the highest validation accuracy attained during its individual run.

## 4.5.2 Experimental Results

As illustrated in Table 4, our proposed LLM-based searcher demonstrates highly competitive performance across the CIFAR-10, CIFAR-100, and ImageNet-16-120 datasets without any task-specific fine-tuning. Although our framework is designed as a general-purpose, open-world optimizer rather than an architecture-specialized NAS algorithm, it yields performance highly comparable to state-of-the-art specialized methods. Specifically, on ImageNet16-120, our zero-shot searcher achieves a test accuracy of 46.51%, noticeably outperforming the top conventional NAS baseline DiNAS [12] (45.41%) as well as earlier LLM-based searchers such as LLMatic [13] (45.87%) and GENIUS [54] (44.96%), while performing closely on par with the specialized CoLLM-NAS [7] (46.79%). These results confirm that our task-agnostic paradigm retains strong search efficiency on standard architectural benchmarks, providing a solid prerequisite for multi-modal open-world AutoML tasks.

## 4.6 Interpretability of LLM-based Searcher’s Reasoning Rationales

## 4.6.1 Experimental Setting

To unveil the proposed LLM-based searcher’s reasoning process in optimizing heterogeneous AutoML configurations, we post-train a Qwen3-8B model with CoT capabilities using our curated optimization trajectory dataset with GRPO (Fig. 2). Crucially, as validated in our ablation studies (Section 4.7), this Qwen3-8B-based searcher exhibits performance parity with our primary LLaMA-3.1- 8B backbone, establishing it as a faithful proxy to decode the framework’s decision-making logic. We employed the LLM-based searcher with CoT capability to search for image classification on NAS-Bench-201 and text-to-image synthesis on GenEval, generating optimization trajectories alongside natural language reasoning processes for explainability. These “reasoning processes” were then taxonomized using Gemini-2.5-Pro to identify six major optimization strategies learned by the LLM-based searcher (Fig. 7): Positional Anchoring (e.g., fixing high-performing digit at critical position), Combinatorial Synergy (e.g., combinational effect of digits across positions), Neighborhood Mutation (e.g., making slight perturbations on evaluated NCode), Global Component Distribution (e.g., tracking the occurrence of specific digit), Hierarchical Sequence Progression (increasing/decreasing trends at local regions) and Operational Heterogeneity (diversity of digits in NCodes).

TABLE 4  
Performance comparison on the NAS-Bench-201 benchmark. For each of the five experimental runs, we select the architecture with the highest validation accuracy and report its corresponding average test accuracy. The evaluation results of ours are adapted from our preliminary conference paper [9].
<table><tr><td rowspan="2">Method</td><td colspan="2">CIFAR10</td><td colspan="2">CIFAR100</td><td colspan="2">ImageNet16-120</td></tr><tr><td>Validation</td><td>Test</td><td>Validation</td><td>Test</td><td>Validation</td><td>Test</td></tr><tr><td colspan="7">Conventional</td></tr><tr><td>Random Search [55]</td><td>91.12</td><td>93.89</td><td>72.08</td><td>72.07</td><td>45.87</td><td>45.98</td></tr><tr><td>DARTS [3]</td><td>39.77</td><td>54.30</td><td>38.57</td><td>15.61</td><td>18.87</td><td>16.32</td></tr><tr><td>PC-DARTS [56]</td><td>89.96</td><td>93.41</td><td>67.12</td><td>67.48</td><td>40.83</td><td>41.31</td></tr><tr><td>SNAS [57]</td><td>90.10</td><td>92.77</td><td>69.69</td><td>69.34</td><td>42.84</td><td>43.16</td></tr><tr><td>SGNAS [58]</td><td>90.18</td><td>93.53</td><td>70.28</td><td>70.31</td><td>44.65</td><td>44.98</td></tr><tr><td>DiNAS [12]</td><td>91.61</td><td>94.37</td><td>73.49</td><td>73.51</td><td>46.66</td><td>45.41</td></tr><tr><td colspan="7">LLM-based</td></tr><tr><td>GENIUS [54]</td><td>91.07</td><td>93.79</td><td>70.96</td><td>70.91</td><td>45.29</td><td>44.96</td></tr><tr><td>LLMatic [13]</td><td>91.42</td><td>94.26</td><td>71.41</td><td>71.62</td><td>44.98</td><td>45.87</td></tr><tr><td>CoLLM-NAS [7]</td><td>91.59</td><td>94.37</td><td>73.44</td><td>73.44</td><td>46.62</td><td>46.79</td></tr><tr><td>Ours</td><td>91.52</td><td>94.20</td><td>72.82</td><td>72.96</td><td>46.48</td><td>46.51</td></tr></table>

## 4.6.2 Experimental Results

Subsequent analysis of strategy distributions, state transitions, and temporal dynamics reveals that the LLM intrinsically develops a sophisticated exploration-exploitation paradigm, progressively shifting from broad structural exploration to the greedy exploitation of key architectural anchors.

Analyzing the temporal evolution of these reasoning patterns unveils a reasonable search logic rather than a random application of strategies (Fig. 7). Initially, the searcher predominantly employs Neighborhood Mutation to navigate the configuration space, selecting candidate NCodes that introduce localized structural perturbations to historically high-performing baselines. As the search progresses, the probability of utilizing this Neighborhood Mutation gradually diminishes, giving way to exploitative patterns that leverage historical context. Crucially, this transition adapts to the complexity of the target task. On the simpler NAS-Bench-201 image classification benchmark, the searcher gradually shifts to the Positional Anchoring strategy, indicating a learned tendency to fix specific, high-yielding modules at critical topological positions. Conversely, on the highly complex GenEval text-to-image synthesis task, the model gradually shifts toward Combinatorial Synergy. By focusing on multi-digit patterns within NCodes, it demonstrates an ability to untangle the intricate combinatorial effects of heterogeneous configurations.

![](images/41a86aaccc138ad3c1828a4991aa74dd42b3666ca133aaf1e2f9d0bdf9a91989.jpg)

![](images/b62a2f3dafa34c2051d4cc816596cc16822f49b85eb2ded1c9f47d33cce87317.jpg)  
Fig. 7. Temporal transition dynamics of the reasoning pattern distributions throughout the search phase (steps 0–180). The stacked area charts illustrate how the LLM dynamically shifts its reliance on different reasoning strategies over time across the NAS-Bench-201 (left) and GenEval (right) benchmarks.

TABLE 5  
Ablation performance comparison of different methods on the GenEval benchmark.
<table><tr><td>Method</td><td># Params</td><td>Single object ↑</td><td>Two object ↑</td><td>Counting ↑</td><td>Colors ↑</td><td>Position ↑</td><td>Attribute binding ↑</td><td>Overall ↑</td></tr><tr><td>Only search architecture</td><td>4.1B</td><td>0.98</td><td>0.93</td><td>0.78</td><td>0.94</td><td>0.76</td><td>0.77</td><td>0.86</td></tr><tr><td>Search hyperparameter and data ratio</td><td>1.8B</td><td>0.99</td><td>0.95</td><td>0.79</td><td>0.92</td><td>0.83</td><td>0.86</td><td>0.89</td></tr><tr><td>w/o searched hyperparameter and data ratio</td><td>1.1B</td><td>0.99</td><td>0.89</td><td>0.68</td><td>0.91</td><td>0.76</td><td>0.72</td><td>0.82</td></tr><tr><td>w/o searched architecture</td><td>1.8B</td><td>0.99</td><td>0.91</td><td>0.75</td><td>0.92</td><td>0.76</td><td>0.73</td><td>0.84</td></tr><tr><td>Ours</td><td>1.1B</td><td>0.99</td><td>0.96</td><td>0.85</td><td>0.96</td><td>0.86</td><td>0.85</td><td>0.91</td></tr></table>

Furthermore, the trajectory of Global Component Distribution, wherein the searcher evaluates configurations based on the macro-level frequency of specific operational digits rather than their absolute positions, reveals distinct adaptations. In the homogeneous NAS-Bench-201 search, the utilization of this strategy progressively intensifies, scaling up from an initial 10% to nearly 20%. In sharp contrast, for the heterogeneous GenEval task, this pattern is kept consistently subdued, hovering steadily at a probability of approximately 5%. This divergence offers compelling evidence of the system’s open-world generalizability: while the model learns to capitalize on global frequency heuristics as a stable descriptor for homogeneous model architecture search, it correctly recognizes that joint optimization across architecture, data, and hyperparameters demands a strict departure from simple global counting. Ultimately, these dynamics demonstrate that the LLM-based searcher does not merely memorize patterns; rather, it effectively navigates search spaces by adaptively combining different strategies to pinpoint optimal configurations.

## 4.7 Ablation Studies

To validate the effectiveness of our approach and the resulting configurations, we compared our method against several baselines. As shown in Table 5, our joint architecture and hyperparameter search approach outperforms both architecture-only search and hyperparameter/data-ratioonly search by overall scores of 0.05 and 0.02, respectively. This highlights the critical importance of jointly optimizing model architectures, training hyperparameters, and data ratios. Furthermore, when applying the searched AutoML configurations without the searched architecture or without the hyperparameter and data configurations (reverting to manually designed baselines), the overall score drops by 0.07 and 0.09, respectively. These results demonstrate that our LLM-based searcher successfully navigates the optimal combination of all configurations, rather than treating their optimization as isolated problems.

Furthermore, to investigate the efficacy of the key components within our framework and evaluate the impact of different LLM backbones, we conduct a comprehensive ablation study on the NAS-Bench-201 benchmark (table 6). As delineated in Table 6, removing the pruning-based architecture sampling strategy (w/o pruning-based sample) incurs a noticeable performance degradation across all three datasets (CIFAR-10, CIFAR-100, and ImageNet16-120), validating its critical role in augmenting trajectory diversity and structural distribution. More significantly, when disabling the objective reframing mechanism (w/o objective reframing) and forcing the model to directly generate full NCodes rather than ranking candidates, we observe a catastrophic performance collapse (e.g., the test accuracy on ImageNet16-

FLUX.1-dev

a photo of a red orange and a purple broccoli

a photo of a brown giraffe and a white stop sign

a photo of a purple wine glass and a black apple a photo of a vase right of a horse a photo of a bird left of a couch

![](images/825e0af30abc9000c316331880050c470da45020613e4a4bc8db0bd419f6bbea.jpg)  
Fig. 8. Qualitative comparison of text-to-image synthesis results on the GenEval benchmark. We compare our optimized model against FLUX.1-dev and OptFormer across complex prompts requiring precise coloring and attribute binding (Columns 1–3), as well as spatial reasoning (Columns 4–5).

TABLE 6  
Ablation performance on NAS-Bench-201 using different models and training strategies.
<table><tr><td rowspan="2">Config</td><td colspan="2">CIFAR10</td><td colspan="2">CIFAR100</td><td colspan="2">ImageNet16-120</td></tr><tr><td>Validation</td><td>Test</td><td>Validation</td><td>Test</td><td>Validation</td><td>Test</td></tr><tr><td>w/o pruning-based sample</td><td>90.83</td><td>93.39</td><td>71.08</td><td>70.98</td><td>44.83</td><td>43.98</td></tr><tr><td>w/o objective reframing</td><td>89.10</td><td>92.16</td><td>69.33</td><td>69.51</td><td>43.38</td><td>42.53</td></tr><tr><td>LLM-based Searcher (LLaMA-3.1-8B)</td><td>91.52</td><td>94.20</td><td>72.82</td><td>72.96</td><td>46.48</td><td>46.51</td></tr><tr><td>LLM-based Searcher (Qwen3-8B)</td><td>91.45</td><td>94.23</td><td>73.05</td><td>73.13</td><td>46.50</td><td>46.37</td></tr></table>

120 plummets from 46.51% to 42.53%). This sharp decline firmly demonstrates that reframing the discrete optimization into a tractable ranking task substantially alleviates the learning burden of the LLM searcher.

In addition, we compare the search efficiency of the LLM-based searchers built upon LLaMA-3.1-8B and Qwen3- 8B. The empirical results indicate that both models achieve highly comparable and competitive performance across all validation and test sets (e.g., reaching 72.96% and 73.13% test accuracy on CIFAR-100, respectively). This comparable performance proves that instilling Chain-of-Thought (CoT) reasoning capabilities via Group Relative Policy Optimization (GRPO) does not jeopardize the core optimization efficacy. To optimize overall search efficiency and circumvent the substantial computational overhead associated with token-heavy CoT generation, we deploy the LLaMA-3.1- 8B searcher to execute all primary search environments reported in this study. Concurrently, the Qwen3-8B searcher is leveraged exclusively for the interpretability analysis to unpack and systematically interpret the internal decisionmaking mechanisms and temporal strategic transitions of the universal search agent.

## 4.8 Case Studies

To further evaluate the fine-grained compositional text-toimage alignment of our optimized Diffusion Transformer, we present qualitative case studies comparing our method against the 12-billion-parameter FLUX.1-dev [39] and the AutoML approach OptFormer [4]. As illustrated in Fig. 8, our method demonstrates superior semantic understanding and instruction adherence across complex compositional prompts involving coloring, attribute binding, and spatial relationships

Specifically, in the first two columns evaluating color attribute binding, FLUX.1-dev struggles to overcome strong visual priors and assign atypical colors to objects. It generates a standard orange instead of the requested ”red orange” and defaults to a standard red stop sign despite the prompt explicitly specifying a ”white stop sign”. In contrast, our method faithfully adheres to the instructions, rendering both the red orange and the white stop sign accurately. In the third column (”a photo of a purple wine glass and a black apple”), FLUX.1-dev misinterprets the attribute binding by applying the purple color to the liquid wine rather than the glass itself. Conversely, our model successfully disentangles these concepts to generate a purpletinted glass.

Furthermore, our method excels in understanding spatial dynamics and relational object associations. In the fourth column (”a photo of a vase right of a horse”), FLUX.1-dev fails to comprehend the distinct object relationship, suffering from concept bleeding where it erroneously generates a horse made of a porcelain material matching the vase. Similarly, in the fifth column (”a photo of a bird left of a couch”), FLUX.1-dev fails to reflect the correct relative spatial positioning requested by the user. Our model correctly captures these spatial and compositional dynamics in both instances. Finally, while OptFormer demonstrates a moderate degree of instruction adherence, it suffers from severe geometric distortions and a lack of photorealism. For instance, OptFormer generates a horse missing a leg (Column 4) and produces severely deformed, unrealistic couch structures (Column 5). These qualitative results underscore our LLM-based search framework’s ability to discover model configurations that yield high-fidelity, photorealistic, and highly instruction-compliant visual generations.

## 5 CONCLUSION AND FUTURE WORK

In this paper, we presented a universal LLM-based AutoML framework designed to jointly optimize model architectures, training hyperparameters, and data ratios across open-world scenarios without task-specific fine-tuning. By introducing NCode, a unified numerical encoding scheme, we mapped heterogeneous design choices into universal discrete sequences and reframed the optimization process as a conditional candidate ranking problem. Trained exclusively on trajectory data curated from a single-domain vision dataset, our searcher demonstrated remarkable zeroshot generalizability across seven domains, four tasks, and three modalities (vision, text, and audio). Experimental results validated its state-of-the-art efficiency and scalability, notably optimizing a 1.3-billion-parameter Diffusion Transformer to attain a GenEval score of 0.91.

Furthermore, by instilling Chain-of-Thought reasoning through Group Relative Policy Optimization (GRPO), we demystified the searcher’s internal decision-making mechanisms, revealing an adaptive transition from early-stage diversity-driven exploration to late-stage anchor exploitation. This work establishes a scalable, generalizable, and interpretable pathway toward autonomous, general-purpose machine learning optimization. Future work will explore extending the searcher to Vision-Language Models (VLMs) to integrate direct visual rewards for richer generative quality assessment.

## REFERENCES

[1] B. Zoph and Q. Le, “Neural architecture search with reinforcement learning,” in International Conference on Learning Representations, 2017.

[2] B. Zoph, V. Vasudevan, J. Shlens, and Q. V. Le, “Learning transferable architectures for scalable image recognition,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 8697–8710.

[3] H. Liu, K. Simonyan, and Y. Yang, “Darts: Differentiable architecture search,” arXiv preprint arXiv:1806.09055, 2018.

[4] Y. Chen, X. Song, C. Lee, Z. Wang, R. Zhang, D. Dohan, K. Kawakami, G. Kochanski, A. Doucet, M. Ranzato et al., “Towards learning universal hyperparameter optimizers with transformers,” Advances in Neural Information Processing Systems, vol. 35, pp. 32 053–32 068, 2022.

[5] X. Dong and Y. Yang, “Nas-bench-201: Extending the scope of reproducible neural architecture search,” arXiv preprint arXiv:2001.00326, 2020.

[6] S. Zhang, C. Gong, L. Wu, X. Liu, and M. Zhou, “Automl-gpt: Automatic machine learning with gpt,” arXiv preprint arXiv:2305.02499, 2023.

[7] Z. Li, Z. Lin, and Y. Wang, “Collm-nas: Collaborative large language models for efficient knowledgeguided neural architecture search,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026, pp. 3273–3282.

[8] A. Zela, J. Siems, L. Zimmer, J. Lukasik, M. Keuper, and F. Hutter, “Surrogate nas benchmarks: Going beyond the limited search spaces of tabular nas benchmarks,” arXiv preprint arXiv:2008.09777, 2020.

[9] Y. Hu, J. Liu, K. Wang, J. Zheng, W. Shi, M. Zhang, Q. Dou, R. Liu, A. Zhou, and H. Li, “Lm-searcher: Cross-domain neural architecture search with llms via unified numerical encoding,” in Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, 2025, pp. 9419–9432.

[10] Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. Li, Y. Wu et al., “Deepseekmath: Pushing the limits of mathematical reasoning in open language models,” arXiv preprint arXiv:2402.03300, 2024.

[11] A. Zela, A. Klein, S. Falkner, and F. Hutter, “Towards automated deep learning: Efficient joint neural architecture and hyperparameter search,” arXiv preprint arXiv:1807.06906, 2018.

[12] R. Asthana, J. Conrad, Y. Dawoud, M. Ortmanns, and V. Belagiannis, “Multi-conditioned graph diffusion for neural architecture search,” arXiv preprint arXiv:2403.06020, 2024.

[13] M. U. Nasir, S. Earle, J. Togelius, S. James, and C. Cleghorn, “Llmatic: neural architecture search via large language models and quality diversity optimization,” in proceedings of the Genetic and Evolutionary Computation Conference, 2024, pp. 1110–1118.

[14] F. Hutter, H. Hoos, and K. Leyton-Brown, “An efficient approach for assessing hyperparameter importance,” in International conference on machine learning. PMLR, 2014, pp. 754–762.

[15] J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V. Le, D. Zhou et al., “Chain-of-thought prompting elicits reasoning in large language models,” Advances in neural information processing systems, vol. 35, pp. 24 824–24 837, 2022.

[16] A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv, C. Zheng, D. Liu, F. Zhou, F. Huang, F. Hu, H. Ge, H. Wei, H. Lin, J. Tang, J. Yang, J. Tu, J. Zhang, J. Yang, J. Yang, J. Zhou, J. Zhou, J. Lin, K. Dang, K. Bao, K. Yang, L. Yu, L.-C. Deng, M. Li, M. Xue, M. Li, P. Zhang, P. Wang, Q. Zhu, R. Men, R. Gao, S.-Q. Liu, S. Luo, T. Li, T. Tang, W. Yin, X. Ren, X. Wang, X. Zhang, X. Ren, Y. Fan, Y. Su, Y.-C. Zhang, Y. Zhang, Y. Wan, Y. Liu, Z. Wang, Z. Cui, Z. Zhang, Z. Zhou, and Z. Qiu, “Qwen3 technical report,” 2025. [Online]. Available: https://api.semanticscholar.org/CorpusID:278602855

[17] D. Ghosh, H. Hajishirzi, and L. Schmidt, “Geneval: An object-focused framework for evaluating text-to-image alignment,” ArXiv, vol. abs/2310.11513, 2023. [Online]. Available: https://api.semanticscholar.org/CorpusID: 264288728

[18] AI@Meta, “Llama 3 model card,” 2024. [Online]. Available: https://github.com/meta-llama/llama3/ blob/main/MODEL CARD.md

[19] E. B. Zaken, S. Ravfogel, and Y. Goldberg, “Bitfit: Simple parameter-efficient fine-tuning for transformerbased masked language-models,” arXiv preprint arXiv:2106.10199, 2021.

[20] N. Houlsby, A. Giurgiu, S. Jastrzebski, B. Morrone, Q. De Laroussilhe, A. Gesmundo, M. Attariyan, and S. Gelly, “Parameter-efficient transfer learning for nlp,” in International conference on machine learning. PMLR, 2019, pp. 2790–2799.

[21] M. Jia, L. Tang, B.-C. Chen, C. Cardie, S. Belongie, B. Hariharan, and S.-N. Lim, “Visual prompt tuning, in European conference on computer vision. Springer, 2022, pp. 709–727.

[22] Y.-L. Sung, J. Cho, and M. Bansal, “Lst: Ladder sidetuning for parameter and memory efficient transfer learning,” Advances in Neural Information Processing Systems, vol. 35, pp. 12 991–13 005, 2022.

[23] T. Chen, L. Zhu, C. Deng, R. Cao, Y. Wang, S. Zhang, Z. Li, L. Sun, Y. Zang, and P. Mao, “Samadapter: Adapting segment anything in underperformed scenes,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 3367– 3375.

[24] D. Lian, D. Zhou, J. Feng, and X. Wang, “Scaling & shifting your features: A new baseline for efficient model tuning,” Advances in Neural Information Processing Systems, vol. 35, pp. 109–123, 2022.

[25] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “Lora: Low-rank adaptation of large language models,” arXiv preprint arXiv:2106.09685, 2021.

[26] Z. Zhong, Z. Tang, T. He, H. Fang, and C. Yuan, “Convolution meets lora: Parameter efficient finetuning for segment anything model,” in International Conference on Learning Representations, vol. 2024, 2024, pp. 26 755– 26 779.

[27] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.- Y. Lo et al., “Segment anything,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 4015–4026.

[28] D.-P. Fan, M.-M. Cheng, Y. Liu, T. Li, and A. Borji, “Structure-measure: A new way to evaluate foreground maps,” International Journal of Computer Vision, vol. 129, pp. 2622 – 2638, 2017. [Online]. Available: https://api.semanticscholar.org/CorpusID:22726592

[29] D.-P. Fan, C. Gong, Y. Cao, B. Ren, M.-M. Cheng, and A. Borji, “Enhanced-alignment measure for binary foreground map evaluation,” in International Joint Conference on Artificial Intelligence, 2018. [Online]. Available: https://api.semanticscholar.org/CorpusID: 44072899

[30] S.-H. Gao, M.-M. Cheng, K. Zhao, X.-Y. Zhang, M.-H. Yang, and P. Torr, “Res2net: A new multi-scale backbone architecture,” IEEE transactions on pattern analysis and machine intelligence, vol. 43, no. 2, pp. 652–662, 2019.

[31] B. Koonce, “ResNet 34,” in Convolutional Neural Networks with Swift for TensorFlow: Image Recognition and Dataset Categorization. Berkeley, CA: Apress, 2021, pp. 51–61.

[32] B. Desplanques, J. Thienpondt, and K. Demuynck, “Ecapa-tdnn: Emphasized channel attention, propagation and aggregation in tdnn based speaker verification,” arXiv preprint arXiv:2005.07143, 2020.

[33] Y. Chen, S. Zheng, H. Wang, L. Cheng, Q. Chen, and J. Qi, “An enhanced res2net with local and global feature fusion for speaker verification,” arXiv preprint arXiv:2305.12838, 2023.

[34] H. Wang, S. Zheng, Y. Chen, L. Cheng, and Q. Chen, “Cam++: A fast and efficient network for speaker verification using context-aware masking,” arXiv preprint arXiv:2303.00332, 2023.

[35] Y. Chen, S. Zheng, H. Wang, L. Cheng, T. Zhu, R. Huang, C. Deng, Q. Chen, S. Zhang, W. Wang et al., “3d-speaker-toolkit: An open-source toolkit for multimodal speaker verification and diarization,” in ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2025, pp. 1–5.

[36] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “High-resolution image synthesis with latent diffusion models,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 10 684–10 695.

[37] D. Podell, Z. English, K. Lacey, A. Blattmann, T. Dockhorn, J. Muller, J. Penna, and R. Rombach, “Sdxl:¨ Improving latent diffusion models for high-resolution image synthesis,” in International Conference on Learning Representations, vol. 2024, 2024, pp. 1862–1874.

[38] P. Esser, S. Kulal, A. Blattmann, R. Entezari, J. Muller,¨ H. Saini, Y. Levi, D. Lorenz, A. Sauer, F. Boesel et al., “Scaling rectified flow transformers for high-resolution image synthesis,” arXiv preprint arXiv:2403.03206, 2024.

[39] B. F. Labs, “Flux,” https://github.com/ black-forest-labs/flux, 2024.

[40] X. Wang, X. Zhang, Z. Luo, Q. Sun, Y. Cui, J. Wang, F. Zhang, Y. Wang, Z. Li, Q. Yu et al., “Emu3:

Next-token prediction is all you need,” arXiv preprint arXiv:2409.18869, 2024.

[41] Y. Ma, X. Liu, X. Chen, W. Liu, C. Wu, Z. Wu, Z. Pan, Z. Xie, H. Zhang, X. Yu et al., “Janusflow: Harmonizing autoregression and rectified flow for unified multimodal understanding and generation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 7739–7751.

[42] X. Chen, Z. Wu, X. Liu, Z. Pan, W. Liu, Z. Xie, X. Yu, and C. Ruan, “Janus-pro: Unified multimodal understanding and generation with data and model scaling,” arXiv preprint arXiv:2501.17811, 2025.

[43] S. Xiao, Y. Wang, J. Zhou, H. Yuan, X. Xing, R. Yan, C. Li, S. Wang, T. Huang, and Z. Liu, “Omnigen: Unified image generation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 13 294–13 304.

[44] C. Deng, D. Zhu, K. Li, C. Gou, F. Li, Z. Wang, S. Zhong, W. Yu, X. Nie, Z. Song et al., “Emerging properties in unified multimodal pretraining,” arXiv preprint arXiv:2505.14683, 2025.

[45] Q. Qin, L. Zhuo, Y. Xin, R. Du, Z. Li, B. Fu, Y. Lu, X. Li, D. Liu, X. Zhu et al., “Lumina-image 2.0: A unified and efficient image generative framework,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 20 031–20 042.

[46] Q. Cai, Y. Li, Y. Pan, T. Yao, and T. Mei, “Hidream-i1: An open-source high-efficient image generative foundation model,” in Proceedings ofthe 33rd ACM International Conference on Multimedia, 2025, pp. 13 636–13 639.

[47] C. Wu, J. Li, J. Zhou, J. Lin, K. Gao, K. Yan, S.-m. Yin, S. Bai, X. Xu, Y. Chen et al., “Qwen-image technical report,” arXiv preprint arXiv:2508.02324, 2025.

[48] OpenAI, “Gpt image 1,” https://developers.openai. com/api/docs/models/gpt-image-1, 2025.

[49] Y. Gao, L. Gong, Q. Guo, X. Hou, Z. Lai, F. Li, L. Li, X. Lian, C. Liao, L. Liu et al., “Seedream 3.0 technical report,” arXiv preprint arXiv:2504.11346, 2025.

[50] J. Chen, Z. Xu, X. Pan, Y. Hu, C. Qin, T. Goldstein, L. Huang, T. Zhou, S. Xie, S. Savarese, L. Xue, C. Xiong, and R. Xu, “Blip3-o: A family of fully open unified multimodal models-architecture, training and dataset,” ArXiv, vol. abs/2505.09568, 2025. [Online]. Available: https://api.semanticscholar.org/CorpusID:278602689

[51] C. Schuhmann, R. Beaumont, R. Vencu, C. Gordon, R. Wightman, M. Cherti, T. Coombes, A. Katta, C. Mullis, M. Wortsman, P. Schramowski, S. Kundurthy, K. Crowson, L. Schmidt, R. Kaczmarczyk, and J. Jitsev, “Laion-5b: An open large-scale dataset for training next generation image-text models,” ArXiv, vol. abs/2210.08402, 2022. [Online]. Available: https://api.semanticscholar.org/CorpusID:252917726

[52] C. Wu, P. Zheng, R. Yan, S. Xiao, X. Luo, Y. Wang, W. Li, X. Jiang, Y. Liu, J. Zhou, Z. Liu, Z. Xia, C. Li, H. Deng, J. Wang, K. Luo, B. Zhang, D. Lian, X. Wang, Z. Wang, T. Huang, and Z. Liu, “Omnigen2: Exploration to advanced multimodal generation,” ArXiv, vol. abs/2506.18871, 2025. [Online]. Available: https://api.semanticscholar.org/CorpusID:279999713

[53] J. Ye, D. Jiang, Z. Wang, L. Zhu, Z. Hu, Z. Huang, J. He, Z. Yan, J. Yu, H. Li, C. He, and W. Li,

“Echo-4o: Harnessing the power of gpt-4o synthetic images for improved image generation,” ArXiv, vol. abs/2508.09987, 2025. [Online]. Available: https: //api.semanticscholar.org/CorpusID:280642445

[54] M. Zheng, X. Su, S. You, F. Wang, C. Qian, C. Xu, and S. Albanie, “Can gpt-4 perform neural architecture search?” arXiv preprint arXiv:2304.10970, 2023.

[55] L. Li and A. Talwalkar, “Random search and reproducibility for neural architecture search,” in Uncertainty in artificial intelligence. PMLR, 2020, pp. 367–377.

[56] Y. Xu, L. Xie, X. Zhang, X. Chen, G.-J. Qi, Q. Tian, and H. Xiong, “Pc-darts: Partial channel connections for memory-efficient architecture search,” arXiv preprint arXiv:1907.05737, 2019.

[57] S. Xie, H. Zheng, C. Liu, and L. Lin, “Snas: stochastic neural architecture search,” in International Conference on Learning Representations, 2019.

[58] S.-Y. Huang and W.-T. Chu, “Searching by generating: Flexible and efficient one-shot nas with architecture generator,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 983– 992.