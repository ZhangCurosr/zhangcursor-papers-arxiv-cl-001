# MNN-Pocket-0.3B Technical Report

MNN Team<sup>1</sup>

<sup>1</sup>Taobao and Tmall Group of Alibaba

## Abstract

This technical report presents MNN-Pocket-0.3B, a compact 300M-parameter dense language model designed for eficient on-device deployment and supporting context lengths of up to 32K tokens. With an INT4 deployment footprint of only 173 MB and decoding throughput of up to 466 tokens/s on iPad Pro M5 (Metal), it can be seamlessly integrated into applications on edge devices such as smartphones and smart glasses.

MNN-Pocket-0.3B is developed through the co-design of its architecture, data, training pipeline, and inference system. Hardware-aware architecture selection and dependency-aware vocabulary pruning balance model capability with on-device memory and compute constraints. A four-stage pre-training pipeline covering approximately 4.1 trillion high-quality Chinese and English tokens equips the model with broad knowledge and reasoning capabilities. This is followed by a three-stage post-training pipeline comprising instruction fine-tuning, multi-teacher on-policy distillation, and quantization-aware training. Finally, hardware-specific optimizations in the MNN inference engine enable fast, memory-eficient inference and long-context processing.

Extensive evaluations demonstrate that MNN-Pocket-0.3B achieves frontier-level performance among models of comparable size, with particularly strong results in Chinese and English knowledge and reasoning, mathematics, coding, and instruction following. It consistently outperforms similarly sized open-source models and even surpasses several 0.5B-parameter models on instruction-following benchmarks.

We release MNN-Pocket-0.3B with open weights and ready-to-use MNN deployment packages. We hope it will provide a practical foundation for capable, responsive, and memory-eficient edge applications while advancing research and development in on-device language models.

Resources: GitHub

Hugging Face ModelScope

![](images/16a184bf414c44aab59c6b714d59f0e0c024ed581205c8a183bcc9db447ef3e8.jpg)  
Figure 1 MNN-Pocket-0.3B Model Overview.

## Contents

Introduction 3   
2 Architecture 4   
2.1 Design Objectives   
2.2 Scaling On-Device Dense Model .   
2.3 Tokenizer Design 5   
3 Pre-Training 6   
<sup>3.1</sup> <sup>Principles</sup> <sup>for</sup> <sup>On-device</sup> <sup>LLM</sup> <sup>Pre-training</sup> <sup>.</sup> <sup>.</sup> <sup>.</sup> <sup>.</sup>3.2 Multi-dimensional Pre-training Data Engineering   
3.3 Pre-training Recipe .   
4 Post-Training . 8   
4.1 Supervised Fine-Tuning (SFT) 8   
4.1.1 SFT Data Mixture 8   
4.1.2 SFT Loop Engine 9   
4.2 Multi-Teacher On-Policy Distillation 10   
4.2.1 Domain-Specialized Teachers and Routing 10   
4.2.2 GKD-Style OPD with Reverse KL 10   
4.2.3 Self-OPD with Quantization-Aware Training 10   
4.3 Non-Autoregressive Decision Modeling 11   
5 On-Device Inference and Deployment . 12   
5.1 Eficient Inference with MNN 12   
5.2 On-Device Eficiency Evaluation 12   
5.3 Speculative Decoding 13   
6 Evaluation 14   
6.1 Experimental Setup 14   
6.2 Post-trained Model Results 15   
6.3 Pre-trained Model Results 15   
6.4 MOPD Training Gains 16   
Applications . 17   
7.1 MNN-Pocket-0.3B for E-Commerce Review Opinion Mining 17   
7.1.1 Dataset and Training Setup 18   
7.1.2 Evaluation 18   
7.2 An Eficient On-Device Voice Assistant with MNN-Pocket-0.3B 19   
7.2.1 System Pipeline 19   
7.2.2 Evaluation 20   
8 Conclusion and Limitations 21   
Contributions and Acknowledgments 21

## 1 Introduction

On-device intelligence has already created substantial value in applications such as search and recommendation, short-form video, live streaming, and extended reality. In the era of large language models, a unified on-device model has the potential to consolidate multiple task-specific models while delivering low-latency responses, reducing cloud inference costs, protecting user privacy, supporting ofline operation, and simplifying model maintenance and iteration.

Recent work has explored compact LLMs ranging from sub-billion to a few billion parameters. SmolLM [20], LFM [1, 31], and MiniCPM [35] focus specifically on small language models, while major open model families—including Gemma [14], Qwen [38], Hunyuan [21], and ERNIE [3]—ofer variants as small as 0.3–0.6B parameters. Despite this progress, deployment in large-scale consumer applications remains challenging. Such applications must support diverse mobile hardware under stringent resource budgets, often requiring 200–300 MB of resident memory and less than 300 MB of peak memory, while targeting generation speeds approaching 200 tokens/s. Existing models rarely balance memory footprint, inference speed, and task performance simultaneously. Addressing this gap requires an edge-first approach in which the model and inference system are jointly optimized across architecture, data, training, quantization, and runtime execution.

We therefore introduce MNN-Pocket-0.3B, a compact 300M-parameter dense language model explicitly designed for on-device deployment. It supports context lengths of up to 32K tokens, occupies only 173 MB with INT4 weights, and achieves decoding throughput of up to 466 tokens/s on iPad Pro M5 (Metal). These properties make MNN-Pocket-0.3B practical for integration into resource-constrained applications on smartphones, smart glasses, and other edge platforms.

MNN-Pocket-0.3B is built around four key innovations:

• Edge-first architecture–tokenizer co-design. Guided by on-device measurements with MNN [23, 43], we select a 300M-parameter dense decoder-only architecture and refine its depth–width allocation to balance model quality, memory footprint, and decoding throughput. We complement this design with a compact 64K bilingual tokenizer derived through dependency-aware vocabulary pruning, balancing sequence compression against embedding memory and output-projection cost.

• Data-centric, stage-aware pre-training. We curate and synthesize Chinese, English, math and code data, grouping samples by domain, quality, knowledge density, dificulty, and style. Proxy runs and model-update interpolation screen candidate recipes, which are refined through direct training on data mixtures. The four-stage pipeline coordinates data mixtures and learning-rate schedules across approximately 4.1 trillion tokens to build knowledge and reasoning capabilities within a compact model.

• Capability-centric, deployment-aware post-training. A three-stage pipeline combines supervised fine-tuning with the SFT Loop Engine, multi-teacher on-policy distillation (MOPD), and quantizationaware training (QAT). MOPD routes student-generated trajectories to domain-specialized teachers, consolidating knowledge, instruction following, mathematics, and coding capabilities into a single compact model. Finally, Self-OPD with QAT aligns the quantization-aware student with its frozen BF16 counterpart to mitigate quantization-induced degradation for W4A16 deployment.

• Model–runtime optimization for efficient edge deployment. MNN-Pocket-0.3B is deployed with MNN through standardized model assets and a unified inference interface with hardware-specific backends. Phase-aware low-bit kernels, fused Transformer operators, and backend-specific attention and KV-cache implementations enable eficient INT4 inference, achieving 8,721 tokens/s in prefill and 466 tokens/s in autoregressive decoding on an iPad Pro M5 with Metal. Optional speculative decoding further accelerates generation, with gains depending on the device, backend, and workload.

The remainder of this report is organized as follows. Section 2 presents the architecture–tokenizer co-design of the MNN-Pocket-0.3B backbone. Sections 3 and 4 describe the pre-training and post-training pipelines, respectively. Section 5 details deployment with MNN-LLM, including runtime optimizations and evaluations of inference speed and memory eficiency. Section 6 reports comprehensive benchmark results across Chinese and English knowledge and reasoning, mathematics, coding, and instruction following. Finally, Section 7 discusses potential applications of MNN-Pocket-0.3B on edge platforms.

## 2 Architecture

## 2.1 Design Objectives

On-device LLM design requires a diferent objective from frontier-scale models: instead of maximizing capability under abundant resources, we optimize jointly for model quality, training eficiency, inference latency, memory footprint, and deployment simplicity within tight resource budgets.

• Low-latency and memory-efficient inference. For on-device deployment (e.g., smartphones and wearables), low latency and small memory footprint are primary requirements. This difers fundamentally from training, which is typically performed on multi-GPU clusters optimized for throughput and scale. As a result, an architecture that performs slightly better during training may still be unsuitable for deployment if it introduces irregular memory access, increases KV-cache overhead, or is poorly supported by optimized inference kernels. We prioritize designs that execute eficiently on practical inference engines and hardware backends.

• High quality under tight parameter budgets. Compared with hundred-billion-parameter models, models in the sub-billion-parameter regime are more sensitive to how parameters are allocated across attention, feed-forward layers, embeddings, and depth [32]. Our goal is therefore not merely to reduce parameter count, but to maximize capability under a limited parameter budget and deployment-friendly constraints.

• Training efficiency and optimization stability. For on-device LLMs, training on GPUs is intrinsically less eficient: reduced model and activation sizes often lead to lower memory occupancy and Model Flops Utilization (MFU), making it harder to fully utilize modern accelerators. The architecture should therefore support stable optimization, eficient hardware utilization, and predictable scaling under moderate batch sizes and limited token budgets.

• Simplicity and extensibility. Although modern Transformer variants [4, 24] can improve eficiency in specific scenarios, such as long-context modeling, they often introduce greater training complexity and weaker compatibility with on-device deployment stacks. Our goal is to adopt a design that is easy to train, eficient to deploy on edge hardware, and extensible to future model variants.

Taken together, our goal is to identify the most efective on-device LLM configuration by jointly balancing quality, eficiency, latency, and flexibility.

## 2.2 Scaling On-Device Dense Model

We adopt an architecture scaled down from the dense Qwen3 model [38] as the backbone of MNN-Pocket-0.3B. While mixture-of-experts (MoE) architectures can increase model capacity without proportionally increasing the number of active parameters, they require a substantially larger total parameter footprint and introduce routing and non-contiguous expert access [7]. These costs are unfavorable for batch-one on-device decoding, where memory capacity, bandwidth, and access locality are more critical than peak compute throughput.

We select the model architecture through a two-stage process. We first determine the overall parameter scale based on deployment memory and inference speed, and then, within the fixed parameter budget, jointly optimize the internal configuration for model capability and hardware eficiency. We scale the dense model architecture from ∼50M to ∼1.3B parameters and deploy each candidate on an M5-based iPad Pro using the MNN Metal backend [23] under identical inference settings. As shown in Figure 2(a), larger models incur higher peak memory usage and lower decoding throughput. We set a 300 MB peak-memory budget to limit the inference footprint under mobile hardware and application constraints, together with a 200 tokens/s decoding target for real-time interaction. Under W4A16 quantization and representative input and output lengths of 256 tokens, the approximately 300M-parameter model is the largest candidate satisfying both constraints. We therefore select this scale and optimize its internal architecture in the second stage.

Within the fixed 300M-parameter budget, we further optimize the internal architecture by jointly varying depth and width while adjusting the feed-forward and attention dimensions to maintain comparable model sizes. Figure 2(b) compares representative 16-, 28-, and 36-layer configurations. Increasing depth from 16 to 28 layers substantially reduces training loss and improves evaluation performance, with only moderate memory and decoding overhead. Further increasing depth to 36 layers yields no additional capability gain, while reducing decoding throughput from 355 to 323 tokens/s. We therefore select the 28-layer configuration, which approaches the capability of the deeper model while providing a more favorable eficiency trade-of. The final configuration of MNN-Pocket-0.3B is summarized in Figure 2(c).

![](images/6dcc7e8960b151d14ca55a0f0d73a7134824f5d49d56517b0e7c5e379a9a9ac6.jpg)  
Figure 2 Hardware-aware architecture selection. (a) Model-scale selection on iPad Pro M5 using the MNN Metal backend. The approximately 300M-parameter model is the largest tested candidate with peak memory below 300 MB and decoding throughput above 200 tokens/s; shading marks the feasible tested candidates. (b) Architecture search at a comparable 300M-parameter budget. The 28-layer model achieves the same evaluation average as the 36-layer model, with lower training loss, lower peak memory, and higher prefill and decoding throughput. (c) Final architecture hyperparameters of MNN-Pocket-0.3B.

## 2.3 Tokenizer Design

Tokenizer design is particularly consequential for on-device LLMs. A larger vocabulary improves sequence compression but increases the parameter and memory costs of the embedding and output layers, which constitute a substantial fraction of a sub-billion-parameter model. It also enlarges the output projection and therefore increases decoding latency. To balance tokenization eficiency with deployment cost, we derive a compact 64K vocabulary from the Qwen3 tokenizer [38] while preserving its coverage of Chinese, English, and specialized domains.

We construct a 6-GiB corpus spanning pre-training, supervised fine-tuning, and constraint-following data, with broad coverage of Chinese, English, code, mathematics, and instruction-oriented formats. After tokenizing the corpus with the original tokenizer, we rank tokens by frequency and greedily select high-frequency entries under the vocabulary budget. All special tokens and 256 byte-fallback tokens are retained, together with the complete BPE merge-dependency chain of every selected token. This dependency-aware pruning preserves valid segmentation and closely approximates the behavior of the original tokenizer without introducing the instability or initialization mismatch associated with training a new tokenizer from scratch.

Finally, we remap the retained tokens to a contiguous ID space. The resulting tokenizer reduces the vocabulary from approximately 151K to 64K tokens—a reduction of about 57%—thereby substantially decreasing embedding and output-layer parameters as well as vocabulary-dependent decoding overhead. Meanwhile, it retains efective tokenization across Chinese and English text, code, mathematics, and instruction-following data, providing a favorable capability–eficiency trade-of for on-device deployment.

Figure 3 summarizes the training pipeline built on the architecture and tokenizer described above. The following sections detail the pre-training and post-training stages.

![](images/2e7bac5959540f09f1d05e45c9b3863fc967bba35ea382156c1d7a4483ced882.jpg)  
Figure 3 Training pipeline overview. Four-stage pre-training produces MNN-Pocket-0.3B-Base, followed by supervised fine-tuning (SFT) on 3 million high-quality question–answer pairs to obtain MNN-Pocket-0.3B-SFT. Four domain-specialized reinforcement learning (RL) teachers covering knowledge, instruction following, mathematics, and code guide MOPD of the SFT student to obtain MNN-Pocket-0.3B. Quantization-aware training (QAT) further converts the consolidated model into MNN-Pocket-0.3B-INT4 for on-device deployment. The consolidated model also initializes MNN-Pocket-0.3B-Dec for non-autoregressive decision modeling.

## 3 Pre-Training

In the sub-billion-parameter regime, better data engineering and training schedules matter much more than simply increasing training tokens. This section outlines the principles guiding our pre-training design, then presents the data engineering pipeline, and finally describes the training recipe for MNN-Pocket-0.3B-Base.

## 3.1 Principles for On-device LLM Pre-training

Tight parameter budgets impose severe capacity constraints, making model capability more sensitive to data quality, dificulty, and domain balance. Our objective is therefore to curate, synthesize, and organize higher-quality data under a fixed tokens-per-parameter (TPP) ratio. The pre-training design of MNN-Pocket-0.3B-Base is guided by the following principles:

• Quality over Scale. On-device LLMs are less tolerant to noisy, redundant, or weakly informative tokens, making data quality rather than scale a primary determinant of final capability.

• Controlled Knowledge Density and Difficulty. Efective pre-training data should be not only high-quality but also suficiently informative and appropriately learnable. We therefore control data attributes such as knowledge density and dificulty to support more precise filtering, analysis, construction, and mixture.

• Stage-aware Domain Balance. Since domains compete more strongly for representation in capacitylimited small LLMs, and diferent domains often favor diferent training stages and settings, efective pre-training especially requires finer-grained control to achieve a well-balanced domain composition, with mixtures adjusted across training stages.

## 3.2 Multi-dimensional Pre-training Data Engineering

Following these principles, we build a pre-training pipeline that treats data as a structured, schedulable asset rather than a uniformly filtered corpus. After cleaning and deduplication, we apply structure-aware rewriting and normalization to semi-structured sources, including question-answer pairs, tables, and lists, to expose their underlying semantics. For Math and Code, we use frontier models to enrich selected samples with relevant concepts, intermediate reasoning steps, and explanatory comments. This processing improves knowledge density and learnability, allowing a capacity-limited model to acquire useful structure from fewer tokens.

![](images/aaf9b5037d12787ac754c60d5ca04a3f0b67c1fb8a3fd9c640bf6a68f5324396.jpg)

![](images/bd0cf4519cedf630a45c1390d4e1778532ddee4991204832cd2c7a591391f3aa.jpg)

![](images/a63c25a8e9dcb7f951c5416678a3c4f275cedd3c09d2ddd09c1380c95015cf6c.jpg)  
Figure 4 Pre-training data composition across training stages. From outer to inner, the concentric rings show domain, style, knowledge density, dificulty, and quality distributions. Knowledge density (D1–D3), dificulty (H1–H3), and quality (Q1–Q3) are each grouped into three levels, from low to high. Percentages are normalized independently within each dimension using recorded token weights. Center labels indicate the training token budget for each stage, in trillions (T) or billions (B).

To support fine-grained control, each document is annotated with the following attributes:

• Domain classifies each document hierarchically, covering broad domains such as English, Chinese, Math, and Code, and finer-grained subdomains (e.g., algebra and geometry within Math).

• Quality reflects the general cleanliness, coherence, and expected learning value of the sample.

• Knowledge density uses signals such as content diversity, entity and concept density, and redundancy to distinguish informative content from verbose or weakly informative data.

• Difficulty measures sample complexity using signals such as prerequisite level, reasoning depth, and domain-specific complexity.

• Style captures format-related properties such as plain prose, QA, structured lists, or tabularized text.

Rather than collapsing these properties into a single scalar score, we use them to form multi-dimensional joint buckets over quality, density, dificulty, style, and domain. This allows the corpus to be analyzed and mixed at a much finer granularity than conventional source-level or score-threshold filtering. During pre-training, these buckets become the basic units for stage-wise mixture design, enabling us to determine not only whether a sample is useful, but also when it is useful and in what proportion.

## 3.3 Pre-training Recipe

Pre-training a 300M-parameter model requires tighter coordination between data composition and optimization than simply scaling down recipes developed for larger LLMs. In particular, domain proportions cannot be optimized independently of their internal quality, knowledge-density, and dificulty distributions: changing these attributes within one domain can alter its optimal share relative to other domains. We therefore formulate pre-training recipe design as a joint search over the complete multi-dimensional data distribution, together with stage boundaries, learning rates, and token budgets.

For each stage t, we partition the corpus into joint buckets $\boldsymbol { b } = ( m , q , d , h , s )$ , corresponding to domain, quality, knowledge density, dificulty, and style, and define its data recipe as a sampling distribution p (b) Rather than tuning individual dimensions sequentially, we jointly search the bucket weights, learning-rate schedule, token budget, and stage boundary. Let $\theta _ { t } ^ { ( 0 ) }$ denote the checkpoint entering stage t. From this shared initialization, short proxy runs train models on diferent candidate recipes under controlled budgets, yielding proxy checkpoints $\theta _ { t } ^ { ( i ) }$ and updates $\Delta \theta _ { t } ^ { ( i ) } = \theta _ { t } ^ { ( i ) } - \theta _ { t } ^ { ( 0 ) }$ . We interpolate these updates as

Table 1 Training hyperparameters for each pre-training stage of MNN-Pocket-0.3B-Base.
<table><tr><td>Hyperparameter</td><td>Stable Stage 1</td><td>Stable Stage 2</td><td>Short Decay</td><td>Long Decay</td></tr><tr><td>Context length</td><td>4,096</td><td>4,096</td><td>4,096</td><td>32,768</td></tr><tr><td>Total tokens</td><td>~3T</td><td>~800B</td><td>~200B</td><td>~100B</td></tr><tr><td>Global batch size</td><td>1024</td><td>1024</td><td>1024</td><td>128</td></tr><tr><td>Tokens per step</td><td>4.2M</td><td>4.2M</td><td>4.2M</td><td>4.2M</td></tr><tr><td>Peak learning rate</td><td> $3 \times 1 0 ^ { - 4 }$ </td><td> $1 \times 1 0 ^ { - 4 }$ </td><td> $4 \times 1 0 ^ { - 5 }$ </td><td> $4 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>LR schedule</td><td>Constant</td><td>Constant</td><td>Cosine</td><td>Cosine</td></tr><tr><td>Warmup steps</td><td>3,000</td><td>3,000</td><td>1000</td><td>1000</td></tr><tr><td>Hardware (GPUs)</td><td>32</td><td>32</td><td>32</td><td>32</td></tr><tr><td>Wall-clock time</td><td>7–8 weeks</td><td>~2 weeks</td><td>~2–3 days</td><td>~2–3 days</td></tr><tr><td>Optimizer</td><td colspan="4">AdamW  $( \beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 5 , \epsilon = 1 0 ^ { - 1 5 } )$ </td></tr><tr><td>Weight decay</td><td colspan="4">0.1</td></tr><tr><td>Gradient clipping</td><td colspan="4">1.0</td></tr></table>

$$
\widetilde { \theta } _ { t } ( \alpha ) = \theta _ { t } ^ { ( 0 ) } + \sum _ { i } \alpha _ { i } \Delta \theta _ { t } ^ { ( i ) } , \qquad \sum _ { i } \alpha _ { i } = 1 ,\tag{1}
$$

to coarsely screen a broader mixture space. Since model interpolation only approximates the trajectory of training on an actual mixture, it is not used for final recipe selection. Instead, we perform a fine-grained search around the strongest candidates by training on their actual bucket-level mixtures and jointly refining the sampling weights and optimization settings. The final candidates are evaluated using convergence, validation loss, downstream capability, and cross-domain regression.

This procedure yields a highly data-eficient pre-training recipe that achieves frontier-level performance at the 300M scale, as demonstrated in Section 6. Figure 4 and Table 1 summarize the multi-dimensional data composition and optimization hyperparameters, respectively, across the pre-training stages of MNN-Pocket-0.3B-Base.

## 4 Post-Training

We adapt MNN-Pocket-0.3B-Base for on-device instruction following through a three-stage post-training pipeline: supervised fine-tuning with the SFT Loop Engine, multi-teacher on-policy distillation for capability consolidation, and quantization-aware training for eficient low-precision deployment.

## 4.1 Supervised Fine-Tuning (SFT)

Foundational SFT combines curated open-source data with targeted synthesis from the SFT Loop Engine (Section 4.1.2) to form the final training mixture, supporting on-device instruction following while aligning with foundational capabilities and deployment scope.

## 4.1.1 SFT Data Mixture

The core corpus is drawn from open-source datasets covering mathematical reasoning, programming, knowledge QA, and open-ended generation. We supplement it with 116.6K targeted synthetic examples from the SFT Loop Engine. After data filtering, the resulting mixture provides broad task coverage while supporting targeted additions for capability gaps. The final recipe contains approximately 3.16M samples and 1.83B assistant tokens, with all percentages normalized by assistant token count. Table 2 summarizes the mixture from two complementary perspectives: taxonomy family and capability.

Table 2 Token-based composition of the final filtered SFT data recipe across taxonomy families and capabilities.
<table><tr><td>Category</td><td>Samples</td><td>Tokens</td><td>Percent</td></tr><tr><td colspan="4">Family</td></tr><tr><td>Multilingual Creation &amp; Open-Ended Generation</td><td>607K</td><td>289M</td><td>15.81%</td></tr><tr><td>Mathematics &amp; Quantitative Reasoning</td><td>722K</td><td>393M</td><td>21.54%</td></tr><tr><td>Programming &amp; Algorithmic Coding</td><td>801K</td><td>433M</td><td>23.70%</td></tr><tr><td>Application Development &amp; General QA</td><td>247K</td><td>148M</td><td>8.08%</td></tr><tr><td>General Chinese-Language Tasks</td><td>450K</td><td>337M</td><td>18.43%</td></tr><tr><td>STEM QA &amp; Scientific Computation</td><td>334K</td><td>227M</td><td>12.45%</td></tr><tr><td colspan="4">Capability</td></tr><tr><td>Code</td><td>970K</td><td>558M</td><td>30.58%</td></tr><tr><td>Knowledge QA</td><td>747K</td><td>482M</td><td>26.41%</td></tr><tr><td>Mathematics</td><td>799K</td><td>443M</td><td>24.26%</td></tr><tr><td>Creative Writing</td><td>378K</td><td>250M</td><td>13.67%</td></tr><tr><td>Instruction Following</td><td>90.0K</td><td>23.6M</td><td>1.29%</td></tr><tr><td>Others</td><td>178K</td><td>69.2M</td><td>3.79%</td></tr><tr><td>Total</td><td>3.16M</td><td></td><td>1.83B 100.00%</td></tr></table>

Our post-training data system uses a three-level taxonomy comprising 1,511 fine-grained categories, 154 mid-level categories, and six families, providing a consistent basis for data versioning, quota assignment, and synthesis.

## 4.1.2 SFT Loop Engine

The compact size of MNN-Pocket-0.3B keeps the resource cost of supervised fine-tuning (SFT) modest, making repeated training and refinement practical. Building on this, we develop the SFT Loop Engine to explore recursive self-improvement (RSI) through iterative cycles of training, evaluation, and targeted data synthesis.

As shown in Figure 5, a lightweight main agent coordinates two skills: Train & evaluate and Refine data. The training and evaluation skill assembles the current data mixture from base and incremental or replacement data, trains an SFT checkpoint, and collects evaluation results. Within the data refinement skill, Subagent A uses the post-training data system to map failed responses and their generation traces to fine-grained categories. Subagent B aggregates failure attributions and historical training-data statistics to assign per-category and per-language synthesis quotas. Subagent C generates targeted synthetic examples without access to the original evaluation items, then performs deduplication and leakage screening. The main agent maintains only workflow state and model-version decisions, and the resulting data update is returned to the training skill to close the loop for the next SFT iteration.

![](images/8bc297cabdd378c95a99b131784163e4d8f05c2a8f5a5255e57035757c5e5644.jpg)  
Figure 5 SFT Loop Engine. An agent invokes the training/evaluation (1) and data refinement (2) skills. Subagents categorize failures (A), aggregate results and allocate quotas (B), and synthesize, deduplicate, and screen data for leakage (C). The refined data feeds the next SFT iteration.

## 4.2 Multi-Teacher On-Policy Distillation

Building on the SFT student, the remaining challenge is to strengthen code, mathematical reasoning, knowledge, and instruction following within a fixed on-device parameter budget. Joint multi-domain reinforcement learning (RL) must reconcile potentially competing updates, while sequential domain-wise tuning risks making the final policy dependent on training order. We therefore adopt Multi-Teacher On-Policy Distillation (MOPD) to consolidate domain experts independently trained through RL into a single student on student-generated trajectories, as shown in Step 1 of Figure 6. The consolidated BF16 student is then compressed to a lowprecision student and recovered by Self-OPD with quantization-aware training, shown in Step 2 of the same figure.

## 4.2.1 Domain-Specialized Teachers and Routing

We construct four domain-specialized teacher routes: code, mathematics, knowledge, and instruction following, listed as the four same-origin teachers inside the routed-teacher box of Step 1 in Figure 6. The teachers are initialized from a common base model and further optimized using domain-specific RL objectives. They share the same architecture, tokenizer, vocabulary, and chat template, which reduces distribution mismatch during distillation.

Each training sample inherits a domain label d from its source dataset, which deterministically selects one frozen teacher $q _ { d } ^ { T }$ . Within a mixed-domain batch, samples are grouped by label for teacher inference, and the returned logits are restored to their original order to supervise the shared student.

## 4.2.2 GKD-Style OPD with Reverse KL

OPD typically uses either a PG objective or a GKD objective. The PG formulation uses the diference between the teacher and student log probabilities of the sampled token as a detached advantage. GKD instead matches their token distributions at each prefix generated by the student, providing supervision over multiple candidate tokens. We use GKD because this denser signal has lower variance and provides more stable training and better performance for our compact student.

For the distillation loss, we choose reverse KL for distribution matching. Unlike the mode-covering forward KL, reverse KL emphasizes the teacher’s dominant modes, enabling more selective knowledge transfer to the capacity-constrained student. Let $\nu _ { d , t } ^ { K }$ denote the teacher’s Top-K candidates, and let $\bar { p } _ { \theta , t }$ and $\bar { q } _ { d , t } ^ { T }$ be the student and teacher distributions renormalized over this set. The token-level distillation loss is

$$
\ell _ { t } ^ { \mathrm { R K L } } = D _ { \mathrm { K L } } \left( \bar { p } _ { \theta , t } \parallel \bar { q } _ { d , t } ^ { T } \right) = \sum _ { v \in \mathcal { V } _ { d , t } ^ { K } } \bar { p } _ { \theta , t } ( v ) \log \frac { \bar { p } _ { \theta , t } ( v ) } { \bar { q } _ { d , t } ^ { T } ( v ) } .\tag{2}
$$

The overall MOPD objective is

$$
\mathcal { L } _ { \mathrm { M O P D } } ( \boldsymbol { \theta } ) = \mathbb { E } _ { \boldsymbol { ( x , d ) } \sim \mathcal { D } } \left[ \frac { 1 } { | \boldsymbol { y } | } \sum _ { t = 1 } ^ { | \boldsymbol { y } | } \ell _ { t } ^ { \mathrm { R K L } } \right] .\tag{3}
$$

## 4.2.3 Self-OPD with Quantization-Aware Training

After MOPD, we freeze the consolidated BF16 model as a self-teacher and initialize a low-precision student from the same checkpoint through post-training quantization $( \mathrm { P T Q } )$ , as shown in Step 2 of Figure 6. The student uses MNN-compatible asymmetric INT4 weight quantization with a group size of 64 and 16-bit activations (W4A16), with shared quantized weights for the token embedding and LM head. To mitigate quantization-induced degradation, we combine Self-OPD with quantization-aware training (QAT), aligning the student and frozen teacher distributions on student-generated prefixes using the Top-K reverse-KL objective in Section 4.2.2. Only the student is updated and retained for deployment.

![](images/e94b0492e5f6234b5a68afe58b9fe51391382e95fb10304c2eae00d768c015fd.jpg)  
Figure 6 Two-step consolidation and low-precision recovery. Both steps run the same on-policy distillation loop. Step 1 (left): Multi-Teacher On-Policy Distillation (MOPD), in which four same-origin RL teachers—code, mathematics, knowledge, and instruction following—are routed by domain label and the selected teacher prefills the MNN-Pocket-0.3B student’s own rollout, so that domain-routed Top-K reverse-KL distillation supervises the student in BF16. Step 2 (right): Self-OPD with quantization-aware training (QAT), which instead uses a single frozen teacher: MNN-Pocket-0.3B in BF16 serves as the self-teacher, and MNN-Pocket-0.3B-INT4 in W4A16 form, initialized by post-training quantization, is aligned to it on its own rollout.

## 4.3 Non-Autoregressive Decision Modeling

Following Kev and Laya [12, 36], we adapt MNN-Pocket-0.3B into MNN-Pocket-0.3B-Dec, a decision model using a shared causal Transformer backbone, task-specific LoRA adapters, and a pointer-based decision head.

![](images/c32ed932832f61cfae7e8909f88461f39d045a2c880eb68d6805cd0d4a8cac98.jpg)  
Figure 7 MNN-Pocket-0.3B-Dec architecture. Left: abridged Typed Decisions [33] customer-service input. Right: the shared model architecture with dataset reference probabilities shown for illustration. Markers are semantic aliases of reserved tokens.

Let $\mathbf { h } _ { \mathrm { d e c } }$ and $\mathbf { h } _ { i }$ be final-RMSNorm states at the decision and option-readout tokens. Shared afine projections and scaled dot-product normalization give

$$
\mathbf { q } = W _ { Q } \mathbf { h } _ { \mathrm { d e c } } + \mathbf { b } _ { Q } , \quad \mathbf { k } _ { i } = W _ { K } \mathbf { h } _ { i } + \mathbf { b } _ { K } , \quad p _ { i } = \mathrm { s o f t m a x } _ { i } \left( \left[ \mathbf { q } ^ { \top } \mathbf { k } _ { j } / \sqrt { d _ { p } } \right] _ { j = 1 } ^ { M } \right) .\tag{4}
$$

Here $W _ { Q } , W _ { K }$ and $\mathbf { b } _ { Q } , \mathbf { b } _ { K }$ are projection weights and biases; $d _ { p } = 2 5 6$ is the projection width, M is the valid-option count, and $p _ { i }$ is option i’s probability. The decision path reads all outputs in one forward pass, bypassing the retained LM head and token-by-token decoding.

## 5 On-Device Inference and Deployment

## 5.1 Efficient Inference with MNN

MNN-Pocket-0.3B is deployed on MNN [23, 43], an open-source on-device inference engine maintained by the MNN team. At runtime, an input prompt is first encoded by the model tokenizer and processed in a prompt-prefill pass that materializes the KV cache. Generation then proceeds autoregressively, consuming one token at a time while reusing the cached states. The generated token IDs are incrementally decoded and streamed to the application. This execution path requires no network connection and keeps both model computation and user data on the device.

The two inference phases have diferent computational characteristics. Prefill exposes parallelism across the prompt and is dominated by matrix–matrix multiplication and attention, whereas batch-one decoding repeatedly streams the model weights and is more sensitive to memory bandwidth and kernel-dispatch overhead. MNN therefore uses phase-aware low-bit projection kernels, fused Transformer operators, and backend-specific attention and KV-cache implementations. Beyond these kernel-level optimizations, MNN also supports speculative decoding, which we apply to on-device generation in Section 5.3. The exported MNN graph provides a common integration interface across devices, while the runtime maps its operators to the available hardware backend. Dedicated NPU deployment may additionally require backend-specific graph conversion, calibration, and compiled model assets.

Unless otherwise stated, the deployment results in this section use MNN-Pocket-0.3B-INT4, obtained through the quantization-aware training procedure in Section 4.2.3. Here we focus on the systems impact of quantization—model size, runtime memory, and inference latency.

## 5.2 On-Device Efficiency Evaluation

We evaluate on-device language models on the iPad Pro M5 and Snapdragon 8 Gen 5 using the MNN Metal, OpenCL, and CPU backends. The models run on MNN 3.6.2 with four CPU threads, low-precision and low-memory modes, and INT4 quantization. Greedy decoding is used to reduce sampling randomness and improve reproducibility. Following llama-bench, each configuration is warmed up once and then measured five times. The tables report the median throughput; prefill and decode are measured independently from an empty KV cache.

Memory and Deployment Footprint Table 3 compares the deployment footprint and peak Metal resident memory usage of each model at a context length of 512 tokens. The MNN deployment footprint includes model weights, the tokenizer, the computation graph, and inference configuration. Peak memory is measured as the maximum Metal resident memory during model loading and inference, minus the baseline measured before model loading. On the iPad Pro (M5), MNN-Pocket-0.3B requires a deployment footprint of 173 MB and reaches a peak memory usage of 228 MB above baseline.

Inference Throughput Table 4 reports inference throughput for 512 token prefill and decode on the iPad Pro M5 and Snapdragon 8 Gen 5. On the iPad Pro M5, MNN-Pocket-0.3B achieves 8,721 tok/s prefill and 466 tok/s decode with Metal, and 2,228 tok/s prefill and 231 tok/s decode with CPU. On Snapdragon 8 Gen 5, it achieves 3,403 tok/s prefill and 138 tok/s decode with OpenCL, and 930 tok/s prefill and 251 tok/s decode with CPU.

Overall, MNN-Pocket-0.3B combines a small deployment footprint and low peak memory usage with high prefill and decode throughput, demonstrating strong eficiency for on-device deployment.

Table 3 Memory and deployment footprint of on-device LLMs on iPad Pro (M5) at 512 token context.
<table><tr><td>Metric</td><td>MNN-Pocket 0.3B</td><td>Gemma-3 270M</td><td>ERNIE-4.5 0.3B</td><td>LFM2.5 350M</td><td>SmolLM2 360M</td><td>MiniCPM4 0.5B</td><td>Hunyuan 0.5B</td><td>Qwen3 0.6B</td></tr><tr><td>MNN inference file size (MB)</td><td>173</td><td>162</td><td>207</td><td>202</td><td>206</td><td>247</td><td>307</td><td>340</td></tr><tr><td>Peak Memory (MB)</td><td>228</td><td>272</td><td>287</td><td>258</td><td>257</td><td>305</td><td>449</td><td>427</td></tr></table>

Table 4 Inference speed (tok/s) of on-device LLMs across diferent platforms at 512 token context.
<table><tr><td>Device</td><td>Backend Stage</td><td></td><td>MNN-Pocket 0.3B</td><td>Hunyuan 0.5B</td><td>Qwen3 0.6B</td><td>0.5B</td><td>MiniCPM4 SmolLM2 360M</td><td>ERNIE-4.5 0.3B</td><td>LFM2.5 350M</td><td>Gemma-3 270M</td></tr><tr><td rowspan="4">iPad Pro (M5)</td><td></td><td>Prefill</td><td>8721</td><td>5330</td><td>6027</td><td>6973</td><td>7067</td><td>7218</td><td>8026</td><td>9519</td></tr><tr><td>Metal</td><td>Decode</td><td>466</td><td>262</td><td>286</td><td>357</td><td>397</td><td>395</td><td>504</td><td>455</td></tr><tr><td rowspan="2">CPU</td><td>Prefill</td><td>2228</td><td>1474</td><td>1396</td><td>1818</td><td>1824</td><td>2185</td><td>2234</td><td>3793</td></tr><tr><td>Decode</td><td>231</td><td>149</td><td>135</td><td>197</td><td>187</td><td>224</td><td>262</td><td>290</td></tr><tr><td rowspan="4">Android (8 Gen 5)</td><td rowspan="2">OpenCL</td><td>Prefill</td><td>3403</td><td>1757</td><td>1627</td><td>2368</td><td>2291</td><td></td><td></td><td>4209</td></tr><tr><td>Decode</td><td>138</td><td>88</td><td>93</td><td>117</td><td>127</td><td>2532 119</td><td>3070 175</td><td>118</td></tr><tr><td rowspan="2">CPU</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Prefill Decode</td><td>930 251</td><td>556 137</td><td>522 126</td><td>682 189</td><td>719 199</td><td>853 216</td><td>889 249</td><td>1953 293</td></tr></table>

## 5.3 Speculative Decoding

To further improve batch-one decoding eficiency, we combine weight quantization with DFlash2 [22]. The draft proposes a block of tokens in parallel, which the target then verifies in a single forward pass [25].

Draft architecture selection Methods such as EAGLE [29], DFlash [5], and DSpark [9] use target hidden states to construct proposals and apply a language-model head to obtain token distributions. With a compact 0.3B target, minimizing draft computation and vocabulary-projection overhead is particularly important. An autoregressive draft requires a draft forward pass and an LM-head computation for each proposed token. DFlash2 instead generates a proposal block in one draft forward pass and computes logits for all proposal positions in a batch. Our implementation shares the target’s token embedding and LM head and uses the position-weighted loss of D-PACE [44].

Draft training We train DFlash2 drafts using either the full-precision MNN-Pocket-0.3B model or its QAT counterpart to provide training signals, yielding the MNN-trained draft and the MNN-INT4-trained draft, respectively. Training is built on SpecForge [27]. Both pipelines use prompts from Open-PerfectBlend [45] and 500K post-training samples with the corresponding target. Unless otherwise stated, speculative-decoding experiments use block size B=8, batch size 1, and greedy decoding. We report mean accepted length per decoding round, denoted AL, and on-device decode throughput. Speedup is measured against autoregressive decoding with the same target, backend, and workload. We use six prompt-only benchmarks: MATH-500 [30] and AIME25 [34] for math, MBPP [2] and HumanEval [6] for code, and MT-Bench [48] and Arena-Hard [28] for chat.

Acceptance results Table 5 evaluates cross-target draft transfer, target-matched training, and draft precision across all six benchmarks with a maximum generation length of 256 tokens. Under the MNN-INT4 verification model, the mean accepted length of the MNN-trained draft is lower than that of the MNN-INT4-trained draft by 1.06% in BF16 and 1.94% in INT4. For the fully trained drafts evaluated here, the mismatch between training source and verification model incurs a modest acceptance penalty that remains acceptable for deployment.

Table 5 Mean accepted length across draft–target configurations for MNN-Pocket-0.3B.
<table><tr><td rowspan="2">Draft Precision Source</td><td rowspan="2">Training</td><td rowspan="2">Verification Model</td><td colspan="2">Math</td><td colspan="2">Code</td><td colspan="2">Chat</td><td rowspan="2">Avg.</td></tr><tr><td>MATH-500</td><td>AIME25</td><td>MBPP</td><td>HumanEval</td><td>MT-Bench</td><td>A-Hard</td></tr><tr><td>BF16</td><td>MNN</td><td>MNN</td><td>5.34</td><td>4.59</td><td>5.55</td><td>5.83</td><td>3.55</td><td>3.18</td><td>4.67</td></tr><tr><td>BF16</td><td>MNN</td><td>MNN-INT4</td><td>5.30</td><td>4.49</td><td>5.57</td><td>5.75</td><td>3.62</td><td>3.16</td><td>4.65</td></tr><tr><td>BF16</td><td>MNN-INT4</td><td>MNN-INT4</td><td>5.34</td><td>4.55</td><td>5.64</td><td>5.82</td><td>3.70</td><td>3.18</td><td>4.70</td></tr><tr><td>INT4</td><td>MNN</td><td>MNN-INT4</td><td>5.16</td><td>4.37</td><td>5.48</td><td>5.65</td><td>3.50</td><td>3.06</td><td>4.54</td></tr><tr><td>INT4</td><td>MNN-INT4</td><td>MNN-INT4</td><td>5.22</td><td>4.47</td><td>5.55</td><td>5.76</td><td>3.65</td><td>3.10</td><td>4.63</td></tr></table>

Table 6 Performance with MNN-Pocket-0.3B MNN-INT4 target and the INT4 MNN-trained draft.
<table><tr><td>Device</td><td></td><td></td><td></td><td>Workload AR tok/s Mean AL Spec. tok/s Speedup</td><td></td></tr><tr><td rowspan="4">iPad Pro (M5)</td><td>Math</td><td rowspan="4">435</td><td>4.49</td><td>709</td><td>1.63×</td></tr><tr><td>Code</td><td>5.73</td><td>895</td><td>2.06×</td></tr><tr><td>Chat</td><td>3.51</td><td>547</td><td>1.26×</td></tr><tr><td>Avg.</td><td>4.57</td><td>717</td><td>1.65×</td></tr><tr><td rowspan="4">Android (8 Gen 5)</td><td>Math</td><td rowspan="4">135</td><td>4.48</td><td>159</td><td>1.17×</td></tr><tr><td>Code</td><td>5.72</td><td>186</td><td>1.38×</td></tr><tr><td>Chat</td><td>3.49</td><td>119</td><td>0.88×</td></tr><tr><td>Avg.</td><td>4.56</td><td>155</td><td>1.15×</td></tr></table>

On-device evaluation Table 6 reports the on-device evaluation of the release configuration on AIME25, HumanEval, and MT-Bench, with a maximum generation length of 256 tokens. Speculative decoding achieves an average decode speedup of 1.65× on iPad Pro and 1.15× on Snapdragon 8 Gen 5, with math and code workloads benefiting more than chat. Despite similar mean accepted lengths across devices, the actual decode speedup varies with device, backend, and workload characteristics.

## 6 Evaluation

## 6.1 Experimental Setup

Benchmarks We evaluate the post-trained MNN-Pocket-0.3B on 11 benchmarks spanning Chinese and English knowledge, general reasoning, mathematics, coding, and instruction following: C-Eval [19], CMMLU [26], MMLU-Redux [13] and BBH [41] for knowledge; GSM8K [11], MGSM-zh [40], and MATH-500 [30] for mathematics; HumanEval [6] and MBPP [2] for coding; and IFEval [49] and IFEval-zh, our Chinese translation of IFEval, for instruction following. For the pre-trained MNN-Pocket-0.3B-Base, we additionally include HellaSwag [46], ARC-Easy [10], and GPQA-Diamond [39] for general capabilities, and CRUX-O [17] for coding.

Baseline Models We compare MNN-Pocket-0.3B with seven open-source language models: Gemma-3- 270M [14], ERNIE-4.5-0.3B [3], LFM2.5-350M [31], SmolLM2-360M [20], Hunyuan-0.5B [21], MiniCPM4-

Table 7 Performance of post-trained MNN-Pocket-0.3B model and other baselines across 11 benchmarks.
<table><tr><td colspan="8">~300M Models ~500M Models</td></tr><tr><td>Benchmark</td><td>MNN-Pocket MNN-Pocket 0.3B</td><td>0.3B-INT4</td><td>Gemma-3 270M</td><td>ERNIE-4.5 0.3B</td><td>LFM2.5 350M</td><td>SmolLM2 360M</td><td>Hunyuan 0.5B</td><td>MiniCPM4 0.5B</td><td>O.G3</td></tr><tr><td>Average</td><td>50.00</td><td>49.59</td><td>6.42</td><td>31.99</td><td>31.28</td><td>18.43</td><td>31.22</td><td>46.81</td><td>44.28</td></tr><tr><td>Knowledge</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>C-Eval</td><td>38.86</td><td>40.27</td><td>6.84</td><td>32.17</td><td>32.10</td><td>23.77</td><td>35.29</td><td>67.98</td><td>44.06</td></tr><tr><td>CMMLU</td><td>40.11</td><td>39.65</td><td>4.94</td><td>33.91</td><td>31.65</td><td>24.87</td><td>35.69</td><td>66.04</td><td>44.50</td></tr><tr><td>MMLU-Redux</td><td>44.00</td><td>43.40</td><td>13.58</td><td>32.68</td><td>41.01</td><td>24.17</td><td>36.29</td><td>52.85</td><td>51.63</td></tr><tr><td>BBH</td><td>47.38</td><td>46.63</td><td>8.09</td><td>31.07</td><td>26.03</td><td>27.74</td><td>54.65</td><td>47.61</td><td>41.94</td></tr><tr><td>IF</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>IFEval</td><td>73.20</td><td>74.12</td><td>20.15</td><td>44.92</td><td>72.09</td><td>32.16</td><td>24.58</td><td>52.13</td><td>56.93</td></tr><tr><td>IFEval-zh</td><td>74.59</td><td>72.49</td><td>17.02</td><td>32.87</td><td>44.06</td><td>25.64</td><td>25.87</td><td>30.30</td><td>50.35</td></tr><tr><td>Math</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GSM8K</td><td>54.36</td><td>52.84</td><td>0.00</td><td>31.24</td><td>33.66</td><td>11.60</td><td>36.01</td><td>42.46</td><td>45.49</td></tr><tr><td>MGSM-zh</td><td>51.20</td><td>48.80</td><td>0.00</td><td>25.20</td><td>18.00</td><td>3.60</td><td>34.80</td><td>37.60</td><td>40.00</td></tr><tr><td>MATH500</td><td>35.40</td><td>35.40</td><td>0.00</td><td>21.40</td><td>14.80</td><td>4.60</td><td>35.80</td><td>23.00</td><td>45.20</td></tr><tr><td>Code</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>HumanEval</td><td>53.66</td><td>53.66</td><td>0.00</td><td>38.41</td><td>14.63</td><td>0.00</td><td>14.63</td><td>48.17</td><td>36.59</td></tr><tr><td>MBPP</td><td>37.20</td><td>38.20</td><td>0.00</td><td>28.00</td><td>16.00</td><td>24.60</td><td>9.80</td><td>46.80</td><td>30.40</td></tr></table>

0.5B [35], and Qwen3-0.6B [38]. These baselines span approximately 0.27B to 0.6B parameters, covering both models of comparable size and larger alternatives. Table 7 places MNN-Pocket-0.3B and its INT4 variant first, followed by baselines grouped into approximately 300M- and 500M-parameter classes. For the base-model evaluation, we use the corresponding base or pre-trained versions of these models where available, as reported in Table 8.

## 6.2 Post-trained Model Results

We show the evaluation results of MNN-Pocket-0.3B and baseline models in Table 7. For knowledge assessment, we use the zero-shot chain-of-thought (CoT) variants of C-Eval, CMMLU, and MMLU-Redux. General reasoning is evaluated on BBH with 3-shot CoT prompting. Mathematical reasoning is evaluated on GSM8K and MATH500 with 4-shot CoT prompting, and on the Chinese subset of MGSM with 8-shot CoT prompting. For coding and instruction following, we use HumanEval, MBPP, IFEval, and IFEval-zh in the zero-shot setting.

MNN-Pocket-0.3B achieves the best average score of 50.00, outperforming the strongest ∼300M baseline, ERNIE-4.5-0.3B, by 18.0 points. Despite having 1.7–2× fewer parameters, it surpasses the larger ∼500M baselines, MiniCPM4-0.5B (46.81) and Qwen3-0.6B (44.28). The advantage is not concentrated on a single capability. Among the full-precision models, MNN-Pocket-0.3B ranks among the top three on all eleven benchmarks and achieves the best score on five of them, spanning instruction following (IFEval 73.20 and IFEval-zh 74.59), bilingual mathematical reasoning (GSM8K 54.36 and MGSM 51.20), and code generation (HumanEval 53.66). The baselines are markedly more specialized: LFM2.5-350M matches MNN-Pocket-0.3B on English IFEval (72.09) but falls to 14.80 on MATH500 and 14.63 on HumanEval, MiniCPM4-0.5B is strong on knowledge-intensive tasks (C-Eval 67.98) yet reaches only 30.30 on IFEval-zh, and Hunyuan-0.5B leads on BBH (54.65) but scores 24.58 on IFEval. This balanced profile is consistent with our post-training design, in which the SFT Loop Engine (Section 4.1.2) turns evaluation failures into targeted training data and multi-teacher on-policy distillation (Section 4.2.1) consolidates four domain specialists into a single student.

## 6.3 Pre-trained Model Results

We show the evaluation results of MNN-Pocket-0.3B-Base and baseline models in Table 8. C-Eval, CMMLU, MMLU-Redux, and GPQA-Diamond are evaluated in the 5-shot setting, HellaSwag in the 10-shot setting, and ARC-Easy in the zero-shot setting. For mathematics, we use 4-shot prompting on GSM8K and MATH500 and 8-shot prompting on MGSM-zh. HumanEval and CRUX-O are evaluated in the zero-shot setting.

Table 8 Performance of pre-trained MNN-Pocket-0.3B-Base model and other baselines across 11 benchmarks.
<table><tr><td colspan="7">~300M Models ~500M Models</td></tr><tr><td>Benchmark</td><td>MNN-Pocket 0.3B-Base</td><td>Gemma-3 270M Base</td><td>ERNIE-4.5 0.3B Base</td><td>LFM2.5 350M Base</td><td>SmolLM2 360M Base</td><td>Hunyuan 0.5B-Pretrain</td><td>Qwen3 0.6B Base</td></tr><tr><td>Average</td><td>33.58</td><td>16.94</td><td>34.00</td><td>29.86</td><td>22.25</td><td>42.60</td><td>44.86</td></tr><tr><td>General Tasks</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>C-Eval</td><td>36.55</td><td>23.25</td><td>40.56</td><td>35.66</td><td>22.07</td><td>57.95</td><td>54.68</td></tr><tr><td>CMMLU</td><td>38.13</td><td>25.26</td><td>39.54</td><td>36.78</td><td>25.30</td><td>54.54</td><td>53.29</td></tr><tr><td>MMLU-Redux</td><td>47.07</td><td>24.47</td><td>46.59</td><td>48.44</td><td>30.19</td><td>59.34</td><td>57.45</td></tr><tr><td>HellaSwag</td><td>38.25</td><td>29.29</td><td>36.33</td><td>36.65</td><td>43.03</td><td>36.87</td><td>40.98</td></tr><tr><td>ARC-Easy</td><td>65.82</td><td>57.70</td><td>65.03</td><td>70.16</td><td>70.59</td><td>66.75</td><td>65.57</td></tr><tr><td>GPQA-Diamond</td><td>27.27</td><td>25.76</td><td>24.24</td><td>21.72</td><td>21.72</td><td>25.25</td><td>30.30</td></tr><tr><td>Math</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GSM8K</td><td>26.00</td><td>0.00</td><td>31.92</td><td>19.79</td><td>4.55</td><td>42.99</td><td>58.38</td></tr><tr><td>MGSM-zh</td><td>17.60</td><td>0.00</td><td>21.60</td><td>16.00</td><td>1.60</td><td>45.20</td><td>48.40</td></tr><tr><td>MATH500</td><td>16.80</td><td>0.00</td><td>17.20</td><td>16.20</td><td>5.20</td><td>23.00</td><td>29.20</td></tr><tr><td>Code</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>HumanEval</td><td>28.66</td><td>0.61</td><td>27.44</td><td>11.59</td><td>0.00</td><td>28.05</td><td>28.66</td></tr><tr><td>CRUX-O</td><td>27.27</td><td>0.00</td><td>23.50</td><td>15.50</td><td>20.50</td><td>28.63</td><td>26.50</td></tr></table>

MNN-Pocket-0.3B-Base achieves an average score of 33.58, ranking second among the approximately 300M models and trailing ERNIE-4.5-0.3B Base (34.00) by only 0.42 points. It exceeds LFM2.5-350M Base, SmolLM2-360M Base, and Gemma-3-270M Base by 3.72, 11.33, and 16.64 points, respectively. Within this size group, it ranks among the top two on ten of the eleven benchmarks and leads on GPQA-Diamond (27.27), HumanEval (28.66), and CRUX-O (27.27), showing competitive performance across both general and specialized tasks. The larger Qwen3-0.6B Base and Hunyuan-0.5B-Pretrain models retain higher overall averages of 44.86 and 42.60, respectively, with stronger knowledge and mathematical reasoning results. Nevertheless, MNN-Pocket-0.3B-Base matches Qwen3-0.6B Base on HumanEval (28.66) and exceeds it on CRUX-O (27.27 vs. 26.50). These results establish a competitive pre-training foundation at the 300M scale, while leaving room for further gains in knowledge and mathematics.

## 6.4 MOPD Training Gains

Figure 8 compares the final MOPD model against its SFT initialization on the same 11 benchmarks as Table 7. MOPD improves ten tasks and maintains the MBPP score at 37.20, raising the macro-average from 35.79 to 50.00. The largest gains occur on MGSM (+28.00), IFEval-zh (+25.17), and MATH500 (+22.40). These results show improvements across knowledge, instruction following, mathematics, and code generation.

MNN-Pocket-0.3B-INT4 achieves an equal-weight average of 49.59 across the same 11 tasks, only 0.41 points below its BF16 counterpart (50.00), indicating that our quantization approach largely preserves floating-point performance. The Self-OPD and quantization-aware training procedure is described in Section 4.2.3.

Overall, the results validate the post-training pipeline: supervised fine-tuning with the SFT Loop Engine and capability consolidation produce a single 0.3B student whose abilities are broad rather than specialized, while the deployment path keeps the teachers out of inference.

![](images/bf18b431e04e6f70ec621bb317bbc03fb028f54009bda424d3c178b872e94d66.jpg)  
Figure 8 Benchmark evolution from the SFT initialization to the final MOPD model. Gray-blue bars denote SFT scores, and blue segments denote gains after MOPD. Final scores match Table 7.

## 7 Applications

To assess the practical value of MNN-Pocket-0.3B beyond benchmark scores, we evaluate it in two realistic workflows: (1) a structured E-Commerce Review Opinion Mining task requiring joint span extraction and fine-grained classification under strict format constraints, and (2) a fully on-device, streaming Voice Assistant deployed on mobile hardware to evaluate real-time conversational latency and interactive quality.

## 7.1 MNN-Pocket-0.3B for E-Commerce Review Opinion Mining

To assess the semantic understanding of the model trained in this work under a controlled, externally scored setting, we evaluate it on the E-Commerce Review Opinion Mining track of the Tianchi Classic Leaderboard.<sup>1</sup> The corpus consists of Chinese cosmetics reviews in which brand mentions are masked with \*\* for anonymity, so brand identity is unavailable as a cue. Given a review, a system must extract every opinion expressed in it as a quadruple (AspectTerms, OpinionTerms, Categories, Polarities). The aspect and opinion terms must be reproduced verbatim from the review, the category is drawn from thirteen predefined classes, and the polarity is one of positive, neutral, and negative. The annotation guidelines allow either term to be absent, in which case the field is filled with the sentinel \_, so a system must decide both what is being evaluated and whether it is stated explicitly at all.

Table 9 shows the prompt and an example instance. The system message fixes the assistant persona, the user turn states the extraction task, enumerates the admissible categories and polarities, provides one worked example, and finally supplies the review to be processed; the target is the quadruple list. The first quadruple names an aspect explicitly and falls in the packaging category, whereas the second has no aspect term: the field is left empty in the target and is written back as the sentinel \_ at submission time, yet the opinion must still be assigned a category — here the overall class — and a polarity. Note also that one review routinely yields several quadruples, all of which have to be recovered.

We adopt this task as a probe of semantic understanding for three reasons. First, it couples span-level grounding with fine-grained classification: a prediction is credited only when all four fields are simultaneously correct, so partial pattern matching on sentiment-bearing words does not score. Second, polarity in this corpus is frequently carried by mild or concessive expressions rather than explicit sentiment lexicon, which penalises models that rely on surface cues. Third, the output is a strictly formatted structured list, so the task jointly measures instruction following and format stability under a fixed decoding budget. Because all submissions are scored by the organiser against held-out labels, the resulting comparison across language backbones is external and reproducible.

Table 9 Prompt and example for E-Commerce Review Opinion Mining.
<table><tr><td>System</td><td>你是一个聪明的AI电商助手。</td></tr><tr><td>Input</td><td>请在商品评论中抽取： 1.评论中被评价的商品属性（AspectTerms），并为其归入对应类目（Categories)。可选类目 为：「&#x27;包装&#x27;，&#x27;成分&#x27;，&#x27;尺寸&#x27;，&#x27;服务&#x27;，&#x27;功效&#x27;，&#x27;价格&#x27;，&#x27;气味&#x27;，’使用体验&#x27;，&#x27;物流&#x27;，’新鲜度&#x27;，’真伪&#x27;，&#x27;整体&#x27;，</td></tr><tr><td></td><td>&#x27;其他]; 2.消费者用以评价的表述（OpinionTerms），并判定其情感倾向（Polarities）。可选倾向为：[&#x27;正 面&#x27;，&#x27;负面&#x27;，&#x27;中性&#x27;]。 若评论未明确提及被评价的属性，AspectTerms置为空字符串”。</td></tr><tr><td></td><td>供参考的评论示例：’很好，遮暇功能差一些’</td></tr><tr><td></td><td>输出：[{&#x27;AspectTerms&#x27;：”,&#x27;OpinionTerms&#x27;：’很好&#x27;，&#x27;Categories&#x27;:’整体’，&#x27;Polarities&#x27;：’正面’},</td></tr><tr><td></td><td>{&#x27;AspectTerms&#x27;:’遮暇功能&#x27;,&#x27;OpinionTerms&#x27;:&#x27;差一些’,&#x27;Categories&#x27;:&#x27;功效&#x27;,&#x27;Polarities&#x27;:&#x27;负面&#x27;}] 下面是需要处理的评论：包装太随便了，感觉很不好</td></tr><tr><td></td><td>输出：</td></tr><tr><td>Output</td><td></td></tr><tr><td></td><td>[{&#x27;AspectTerms&#x27;:&#x27;包装’,&#x27;OpinionTerms&#x27;:&#x27;太随便了&#x27;,&#x27;Categories&#x27;:&#x27;包装’，&#x27;Polarities&#x27;:’负面&#x27;},</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr></table>

## 7.1.1 Dataset and Training Setup

The training split contains 3,229 reviews annotated with 6,633 quadruples, an average of 2.05 opinions per review. The test split contains 2,237 reviews whose labels are withheld; scores are obtained by submitting predictions to the leaderboard. The thirteen categories are packaging, ingredients, size, service, eficacy, price, scent, user experience, logistics, freshness, authenticity, overall, and other, and the three polarities are positive, neutral, and negative; both label sets are fixed by the training split. The oficial annotation additionally records the character ofsets of every aspect and opinion span, but the evaluation ignores position information and scores the quadruples alone, so we discard the ofsets and supervise the quadruple list only. We convert the oficial annotation tables into a single-turn supervised fine-tuning corpus of 3,229 examples. Each user turn states the extraction task, enumerates the thirteen categories and the three polarities, and provides one worked example; the target is the gold quadruple list rendered as a Python literal. All backbones share the same system prompt and the same instruction template, and no development split is held out. At submission time the generated list is flattened into the oficial id, AspectTerms, OpinionTerms, Categories, Polarities format, empty fields are written back as the sentinel \_, and reviews for which no quadruple is produced receive a placeholder row so that every identifier appears exactly once.

All backbones are fine-tuned with LoRA-based supervised fine-tuning under ms-swift [47] in bfloat16 on a single AMD accelerator. We train for three epochs with a learning rate of $1 \times 1 0 ^ { - 4 }$ , a warmup ratio of 0.05, a maximum sequence length of 4,096, and a per-device batch size of 16 without gradient accumulation. Data, prompt template, system prompt, and every optimisation hyperparameter are held identical across backbones<sup>2</sup>, so the diferences reported below are attributable to the pretrained model rather than to per-model tuning. For backbones with an optional reasoning mode we disable it, matching the training distribution.

## 7.1.2 Evaluation

Table 10 reports F1 on the oficial test split. A predicted quadruple is credited only when all four fields match a gold quadruple exactly, and F1 is micro-averaged over the split.

MNN-Pocket-0.3B leads at 80.15, ahead of the runner-up Qwen3-1.7B (78.59) by 1.56 points with less than a fifth of its parameters, and of Qwen3-0.6B (77.47) by 2.68 points with half. It also beats both 0.5B systems (Hunyuan-0.5B by 2.88 points and MiniCPM4-0.5B by 4.88) and the identically sized ERNIE-4.5-0.3B by 3.42, making it both the strongest and the most parameter-eficient point in the comparison — the regime that matters for on-device deployment. The field spans 12.61 points and its ordering does not track capacity: the weakest system, SmolLM2-360M, is not the smallest, LFM2.5-350M outscores MiniCPM4-0.5B, and a 0.3B backbone beats every larger one, indicating that once in-domain supervision is available the binding constraint is adaptability rather than capacity. Since a quadruple is credited only when its aspect span, opinion span, category, and polarity are all correct at once, MNN-Pocket-0.3B’s clear lead shows that it sustains span-level grounding, fine-grained classification, and stable-format instruction following together — at less than a fifth of the parameters of the next-best model, precisely the property on-device deployment requires.

Table 10 E-Commerce Review Opinion Mining results. All backbones are LoRA fine-tuned with the identical recipe of Section 7.1.1 and scored on the oficial test split; a quadruple counts as correct only under exact match on all four fields. Parameter counts are the nominal sizes given by the model names. Rows are ordered by F1.
<table><tr><td>Language backbone</td><td>Params</td><td>F1 (%) ↑</td></tr><tr><td>MNN-Pocket-0.3B</td><td>0.3B</td><td>80.15</td></tr><tr><td>Qwen3-1.7B</td><td>1.7B</td><td>78.59</td></tr><tr><td>Qwen3-0.6B</td><td>0.6B</td><td>77.47</td></tr><tr><td>Hunyuan-0.5B</td><td>0.5B</td><td>77.27</td></tr><tr><td>ERNIE-4.5-0.3B</td><td>0.3B</td><td>76.73</td></tr><tr><td>LFM2.5-350M</td><td>0.35B</td><td>76.19</td></tr><tr><td>MiniCPM4-0.5B</td><td>0.5B</td><td>75.27</td></tr><tr><td>LFM2.5-230M</td><td>0.23B</td><td>73.14</td></tr><tr><td>Gemma-3-270M</td><td>0.27B</td><td>72.23</td></tr><tr><td>SmolLM2-360M</td><td>0.36B</td><td>67.54</td></tr></table>

## 7.2 An Efficient On-Device Voice Assistant with MNN-Pocket-0.3B

To assess the deployment feasibility and real-time interaction capability of the model trained in this work, we built a fully on-device speech-to-speech dialogue system. The system uses MNN-Pocket-0.3B as its language backbone and integrates streaming automatic speech recognition (ASR), voice activity detection (VAD), semantic turn-end detection, streaming text generation, incremental text-to-speech (TTS) synthesis, and barge-in support. All inference and audio processing are performed locally, eliminating network-induced latency and dependence on cloud inference services while keeping user speech and dialogue data on the device.

## 7.2.1 System Pipeline

Figure 9 shows the overall system pipeline. All neural-network modules are deployed using the MNN inference framework. Speech recognition is performed by an INT8-quantized X-ASR [15] model, text responses are generated by the INT4-quantized MNN-Pocket-0.3B model trained in this work, and speech is synthesized by MOSS-TTS-Nano [16].

User speech is captured by the microphone and processed by the audio front end for acoustic echo cancellation, noise suppression, and automatic gain control. The processed audio is then fed to Silero VAD [42] for speech activity detection and X-ASR for streaming transcription. Turn-end detection follows a two-stage strategy. Silero VAD first identifies candidate endpoints from speech activity, after which Smart Turn [37] evaluates whether the current utterance is semantically complete. Combined with transcript-stability checks, this strategy reduces the risk of premature turn termination that may arise when endpoint detection relies solely on silence duration.

![](images/9bfa09b36af488487b0f812d873c224aa02f674209114567cf3f3aace3da1fa7.jpg)  
Figure 9 Fully on-device speech-to-speech dialogue pipeline. Processed microphone audio feeds Silero VAD and INT8 X-ASR for speech activity detection and streaming transcription. Transcript-stability checks and Smart Turn determine when to commit the user utterance to INT4 MNN-Pocket-0.3B. Generated text is segmented and normalized for incremental synthesis by MOSS-TTS-Nano, with audio chunks queued for playback. Generation, synthesis, and playback run asynchronously; new user speech interrupts all three, while the playback signal provides a reference for acoustic echo cancellation. All neural modules run locally with MNN.

Once the user turn is committed, the finalized utterance is passed to the MNN-Pocket-0.3B model for streaming text generation. The generated text is incrementally segmented and normalized into text chunks, which are then fed to MOSS-TTS-Nano for incremental speech synthesis. The resulting audio chunks are placed in a playback queue for sequential output. Text generation, speech synthesis, and audio playback operate asynchronously, allowing playback to begin before the full response has been generated.

During playback, the system continues to monitor incoming speech. When new user speech is detected, it cancels ongoing text generation and speech synthesis and stops audio playback to support barge-in. The playback signal is also fed back to the audio front end as a reference for acoustic echo cancellation.

## 7.2.2 Evaluation

To assess the on-device eficiency, inference speed, and runtime latency profile of the proposed voice assistant, we conduct comprehensive evaluations across various mobile devices. We report both the CPU-based computeonly theoretical latency and the measured production latency. The theoretical latency assumes warmed-up models, immediate endpoint commitment, and synthesis from the earliest available text fragment. The production configuration introduces additional safeguards for ASR tail completeness, semantic turn detection, and TTS prosody stability. Table 11 summarizes the overall performance. The theoretical latency estimates exclude endpointing delays and text-bufering safeguards, whereas measured latencies are reported as the 50th and 95th percentiles under warmed-up conditions.

To comprehensively assess the proposed system’s spoken dialogue capability and intelligent conversational performance on standard voice benchmarks, we evaluate all language backbones using a unified setup with an X-ASR front end (160-ms chunk size and 2-s trailing silence). The system takes speech as input and simultaneously produces text and speech responses. Open-ended tasks are scored by Qwen3.7-Max with three votes per sample. Table 12 reports results on nine VoiceBench [8] tasks. MNN-Pocket-0.3B achieves the highest overall score of 45.44, surpassing the second-best backbone, LFM2.5-350M, by 2.17 points. It ranks first on BBH, IFEval, and AdvBench, demonstrating competitive reasoning, instruction-following, and safety

Table 11 On-device performance of the MNN-Pocket-0.3B Voice Assistant.
<table><tr><td>Metric</td><td>iPhone</td><td>iPad</td><td>Android</td></tr><tr><td>Device / SoC</td><td>13 Pro / A15</td><td>Pro / M5</td><td>Canoe / 8Gen5</td></tr><tr><td>Model storage (MB) ↓</td><td>2,018</td><td>2,018</td><td>2,018</td></tr><tr><td>Peak memory (MB) ↓</td><td>1,591</td><td>1,858</td><td>1,704</td></tr><tr><td>ASR RTF ↓</td><td>0.073</td><td>0.047</td><td>0.045</td></tr><tr><td>LLM decoding speed (token/s) ↑</td><td>130.2</td><td>324.9</td><td>231.7</td></tr><tr><td>TTS RTF ↓</td><td>0.653</td><td>0.366</td><td>0.651</td></tr><tr><td>Theoretical TTFT (ms) ↓</td><td>~169</td><td>~46</td><td>~144</td></tr><tr><td>TTFT P50/P95 (ms) ↓</td><td>568  / 585</td><td>472 /481</td><td>523 / 535</td></tr><tr><td>Theoretical TTFA (ms) ↓</td><td>~485</td><td>~171</td><td>~358</td></tr><tr><td>TTFA P50/P95 (ms) ↓</td><td>1,055 / 1,113</td><td>600 / 618</td><td>935 / 1,022</td></tr></table>

Table 12 VoiceBench results of the MNN-Pocket-0.3B Voice Assistant and other models.
<table><tr><td>Language backbone</td><td>Common</td><td>WildVoice</td><td>SD-QA</td><td>OBQA</td><td>MMSU</td><td>BBH</td><td>IFEval</td><td>AdvBench</td><td>AlpacaEval</td><td>Overall</td></tr><tr><td>MNN-Pocket-0.3B</td><td>40.23</td><td>36.72</td><td>5.75</td><td>32.75</td><td>27.91</td><td>58.90</td><td>64.11</td><td>97.69</td><td>44.91</td><td>45.44</td></tr><tr><td>Gemma3-270M</td><td>37.17</td><td>31.54</td><td>6.43</td><td>25.93</td><td>25.02</td><td>49.40</td><td>18.32</td><td>93.27</td><td>38.56</td><td>36.18</td></tr><tr><td>ERNIE-4.5-0.3B</td><td>37.33</td><td>32.50</td><td>4.87</td><td>28.57</td><td>25.63</td><td>50.60</td><td>33.98</td><td>39.42</td><td>38.46</td><td>32.37</td></tr><tr><td>LFM2.5-350M</td><td>47.17</td><td>39.96</td><td>11.06</td><td>22.86</td><td>25.76</td><td>46.30</td><td>57.17</td><td>88.65</td><td>50.47</td><td>43.27</td></tr><tr><td>SmolLM2-360M</td><td>37.93</td><td>32.26</td><td>15.78</td><td>21.98</td><td>24.59</td><td>46.40</td><td>32.59</td><td>23.46</td><td>41.20</td><td>30.69</td></tr><tr><td>Hunyuan-0.5B</td><td>22.67</td><td>22.19</td><td>2.02</td><td>29.67</td><td>28.24</td><td>48.70</td><td>18.06</td><td>23.85</td><td>23.73</td><td>24.35</td></tr><tr><td>MiniCPM4-0.5B</td><td>43.90</td><td>35.69</td><td>10.67</td><td>28.79</td><td>31.13</td><td>49.70</td><td>44.04</td><td>58.65</td><td>43.76</td><td>38.48</td></tr><tr><td>Qwen3-0.6B</td><td>45.87</td><td>38.77</td><td>11.15</td><td>35.82</td><td>32.76</td><td>49.60</td><td>47.73</td><td>60.96</td><td>47.70</td><td>41.15</td></tr></table>

performance among the evaluated backbones.

## 8 Conclusion and Limitations

We present MNN-Pocket-0.3B, an on-device language model that combines architecture–tokenizer co-design, staged training, and MNN runtime optimization to balance capability, memory, and latency. It delivers strong benchmark performance and supports practical opinion mining and streaming voice interaction.

However, knowledge-intensive tasks and complex reasoning remain challenging, and deployment eficiency varies across devices and workloads. Our evaluations focus primarily on Chinese and English and a limited set of applications and hardware. Future work will strengthen model reliability, broaden evaluation coverage, improve end-to-end eficiency, and explore on-device agents, search and recommendation, user intent understanding, and other e-commerce applications within the Taobao mobile app.

## 9 Contributions and Acknowledgments

MNN-Pocket-0.3B represents MNN’s exploration of on-device LLM training and deployment and is the result of the sustained, collective eforts of all our team members and interns. This work was also supported by the Key Research and Development Program of Zhejiang Province (Grant No. 2025C01026).

Project Leaders Zhiwen Chen, Zhaode Wang, Chengfei Lv, Gang Yu

Contributors (Ordered by the last name) Juncan Deng, Haotian Dong, Zhewei Fang, Canan Huang, Zhoulin Ji, Zefeng Sun, Gexing Tan, Yunfeng Wang, Yufan Xiong, Huangxiang Zhang, Weijun Zhang, Yuxin Zhang

## References

[1] Liquid AI. Lfm2.5-350m: No size left behind. Liquid AI Blog, 2026. www.liquid.ai/blog/lfm2-5-350m-no-size-leftbehind.

[2] Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, and Charles Sutton. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021. URL https://arxiv.org/abs/2108.07732.

[3] Baidu-ERNIE-Team. Ernie 4.5 technical report. 2025. URL https://ernie.baidu.com/blog/publication/ ERNIE\_Technical\_Report.pdf.

[4] Ruisheng Cao, Mouxiang Chen, Jiawei Chen, Zeyu Cui, Yunlong Feng, Binyuan Hui, Yuheng Jing, Kaixin Li, Mingze Li, Junyang Lin, et al. Qwen3-coder-next technical report. arXiv preprint arXiv:2603.00729, 2026.

[5] Jian Chen, Yesheng Liang, and Zhijian Liu. DFlash: Block difusion for flash speculative decoding. In International Conference on Machine Learning (ICML), 2026. URL https://arxiv.org/abs/2602.06036.

[6] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021. URL https://arxiv.org/abs/2107. 03374.

[7] Yanbei Chen, Hanxian Huang, Ernie Chang, Jacob Szwejbka, Digant Desai, Zechun Liu, Vikas Chandra, and Raghuraman Krishnamoorthi. Mobilemoe: Scaling on-device mixture of experts. arXiv preprint arXiv:2605.27358, 2026.

[8] Yiming Chen, Xianghu Yue, Chen Zhang, Xiaoxue Gao, Robby T Tan, and Haizhou Li. Voicebench: Benchmarking llm-based voice assistants. Transactions of the Association for Computational Linguistics, 14:378–398, 2026.

[9] Xin Cheng, Xingkai Yu, Chenze Shao, Jiashi Li, Yunfan Xiong, Yi Qian, et al. DSpark: Confidence-scheduled speculative decoding with semi-autoregressive generation. arXiv preprint arXiv:2607.05147, 2026. URL https: //arxiv.org/abs/2607.05147.

[10] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv:1803.05457v1, 2018.

[11] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

[12] Convai Innovations. Laya: Non-autoregressive decision modeling. https://github.com/NandhaKishorM/laya, 2026. Source revision 432312b3; accessed October 14, 2026.

[13] Aryo Pradipta Gema, Joshua Ong Jun Leang, Giwon Hong, Alessio Devoto, Alberto Carlo Maria Mancino, Rohit Saxena, Xuanli He, Yu Zhao, Xiaotang Du, Mohammad Reza Ghasemi Madani, Claire Barale, Robert McHardy, Joshua Harris, Jean Kaddour, Emile van Krieken, and Pasquale Minervini. Are we done with mmlu? 2024.

[14] Gemma Team. Gemma 3 technical report. 2025. URL https://arxiv.org/abs/2503.19786.

[15] Gilgamesh-J. X-ASR. https://github.com/Gilgamesh-J/X-ASR. GitHub repository. Accessed: 2026-09-11.

[16] Yitian Gong, Botian Jiang, Yiwei Zhao, Yucheng Yuan, Kuangwei Chen, Yaozhou Jiang, Cheng Chang, Dong Hong, Mingshu Chen, Ruixiao Li, Yiyang Zhang, Yang Gao, Hanfu Chen, Ke Chen, Songlin Wang, Xiaogui Yang, Yuqian Zhang, Kexin Huang, ZhengYuan Lin, Kang Yu, Ziqi Chen, Jin Wang, Zhaoye Fei, Qinyuan Cheng, Shimin Li, and Xipeng Qiu. Moss-tts technical report, 2026. URL https://arxiv.org/abs/2603.18090.

[17] Alex Gu, Baptiste Rozière, Hugh Leather, Armando Solar-Lezama, Gabriel Synnaeve, and Sida I. Wang. Cruxeval: A benchmark for code reasoning, understanding and execution. arXiv preprint arXiv:2401.03065, 2024.

[18] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. arXiv preprint arXiv:2103.03874, 2021. URL https://arxiv.org/abs/2103.03874.

[19] Yuzhen Huang, Yuzhuo Bai, Zhihao Zhu, Junlei Zhang, Jinghan Zhang, Tangjun Su, Junteng Liu, Chuancheng Lv, Yikai Zhang, Jiayi Lei, Yao Fu, Maosong Sun, and Junxian He. C-eval: A multi-level multi-discipline chinese evaluation suite for foundation models. arXiv preprint arXiv:2305.08322, 2023.

[20] Hugging Face Team. Smollm2: When smol goes big – data-centric training of a small language model. 2025. URL https://arxiv.org/abs/2502.02737.

[21] Hunyuan Team. Hunyuan: Hunyuan-0.5b-instruct. 2025. URL https://huggingface.co/tencent/Hunyuan-0. 5B-Instruct.

[22] Inco AI. DFlash 2: Keep drafting parallel. Inco AI Blog, August 2026. URL https://inco.ai/blog/dflash2/. Accessed: 2026-09-08.

[23] Xiaotang Jiang, Huan Wang, Yiliu Chen, Ziqi Wu, Lichuan Wang, Bin Zou, Yafeng Yang, Zongyang Cui, Yu Cai, Tianhang Yu, et al. Mnn: A universal and eficient inference engine. Proceedings of Machine Learning and Systems, 2:1–13, 2020.

[24] Kimi Team. Kimi K3: Open frontier intelligence. arXiv preprint arXiv:2607.24653, 2026. URL https://arxiv. org/abs/2607.24653.

[25] Yaniv Leviathan, Matan Kalman, and Yossi Matias. Fast inference from transformers via speculative decoding. In International Conference on Machine Learning (ICML), 2023. URL https://arxiv.org/abs/2211.17192.

[26] Haonan Li, Yixuan Zhang, Fajri Koto, Yifei Yang, Hai Zhao, Yeyun Gong, Nan Duan, and Timothy Baldwin. Cmmlu: Measuring massive multitask language understanding in chinese. 2024. URL https://arxiv.org/abs/ 2306.09212.

[27] Shenggui Li, Chao Wang, Yikai Zhu, Yubo Wang, Fan Yin, Shuai Shi, Yefei Chen, Xiaomin Dong, Qiaoling Chen, Jin Pan, et al. SpecForge: A flexible and eficient open-source training framework for speculative decoding. arXiv preprint arXiv:2603.18567, 2026. URL https://arxiv.org/abs/2603.18567.

[28] Tianle Li, Wei-Lin Chiang, Evan Frick, Lisa Dunlap, Banghua Wu, Banghua Zhu, Joseph E. Gonzalez, and Ion Stoica. From crowdsourced data to high-quality benchmarks: Arena-Hard and BenchBuilder pipeline. arXiv preprint arXiv:2406.11939, 2024. URL https://arxiv.org/abs/2406.11939.

[29] Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. EAGLE: Speculative sampling requires rethinking feature uncertainty. In International Conference on Machine Learning (ICML), 2024. URL https://arxiv.org/ abs/2401.15077.

[30] Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. arXiv preprint arXiv:2305.20050, 2023. URL https://arxiv.org/abs/2305.20050. Source of the MATH-500 re-solve subset of the MATH benchmark [18].

[31] Liquid AI. Lfm2 technical report. 2025. URL https://arxiv.org/abs/2511.23404.

[32] Zechun Liu, Changsheng Zhao, Forrest Iandola, Chen Lai, Yuandong Tian, Igor Fedorov, Yunyang Xiong, Ernie Chang, Yangyang Shi, Raghuraman Krishnamoorthi, Liangzhen Lai, and Vikas Chandra. MobileLLM: Optimizing sub-billion parameter language models for on-device use cases. In Proceedings of the 41st International Conference on Machine Learning, pages 32431–32454, 2024.

[33] LocalLLaMA. Typed Decisions. https://huggingface.co/datasets/LocalLLaMA/typed-decisions, 2026. Dataset revision d0e2f0c42fef; accessed October 14, 2026.

[34] Mathematical Association of America. American invitational mathematics examination (AIME) 2025. https: //www.maa.org/math-competitions/aime, 2025. Competition problems used as a math-reasoning benchmark. Accessed: 2026-09-16.

[35] MiniCPM Team. Minicpm4: Ultra-eficient llms on end devices. 2025. URL https://arxiv.org/abs/2506.07900.

[36] Jared Palmer. Kev: Open-source decision models with pointer readout. https://github.com/jaredpalmer/kev, 2026. Source revision 0a6d3faf; accessed October 14, 2026.

[37] Pipecat. Smart Turn v3. https://huggingface.co/pipecat-ai/smart-turn-v3. Hugging Face model repository. Accessed: 2026-09-11.

[38] Qwen Team. Qwen3 technical report. 2025. URL https://arxiv.org/abs/2505.09388.

[39] David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. Gpqa: A graduate-level google-proof q&a benchmark. 2023. URL https: //arxiv.org/abs/2311.12022.

[40] Freda Shi, Mirac Suzgun, Markus Freitag, Xuezhi Wang, Suraj Srivats, Soroush Vosoughi, Hyung Won Chung, Yi Tay, Sebastian Ruder, Denny Zhou, Dipanjan Das, and Jason Wei. Language models are multilingual chain-of-thought reasoners. 2022. URL https://arxiv.org/abs/2210.03057.

[41] Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V Le, Ed H Chi, Denny Zhou, and Jason Wei. Challenging big-bench tasks and whether chain-of-thought can solve them. arXiv preprint arXiv:2210.09261, 2022.

[42] Silero Team. Silero vad: pre-trained enterprise-grade voice activity detector (vad), number detector and language classifier. https://github.com/snakers4/silero-vad, 2024.

[43] Zhaode Wang, Jingbang Yang, Xinyu Qian, Shiwen Xing, Xiaotang Jiang, Chengfei Lv, and Shengyu Zhang. Mnn-llm: A generic inference engine for fast large language model deployment on mobile devices. page 1–7, December 2024. doi: 10.1145/3700410.3702126. URL http://dx.doi.org/10.1145/3700410.3702126.

[44] Tianyu Wu, Yu Yao, Zhenting Qi, Han Zheng, Zhuohan Wang, Haoran Ma, Lawrence Liao, Himabindu Lakkaraju, Ju Li, and Yilun Du. D-PACE: Dynamic position-aware cross-entropy for parallel speculative drafting. arXiv preprint arXiv:2605.18810, 2026. URL https://arxiv.org/abs/2605.18810.

[45] Tengyu Xu, Eryk Helenowski, Karthik Abinav Sankararaman, Di Jin, Kaiyan Peng, Eric Han, Shaoliang Nie, Chen Zhu, Hejia Zhang, Wenxuan Zhou, Zhouhao Zeng, Yun He, Karishma Mandyam, Arya Talabzadeh, Madian Khabsa, Gabriel Cohen, Yuandong Tian, Hao Ma, Sinong Wang, and Han Fang. The Perfect Blend: Redefining RLHF with mixture of judges. arXiv preprint arXiv:2409.20370, 2024. URL https://arxiv.org/abs/2409.20370.

[46] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? 2019.

[47] Yuze Zhao, Jintao Huang, Jinghan Hu, Xingjun Wang, Yunlin Mao, Daoze Zhang, Hong Zhang, Zeyinzi Jiang, Zhikai Wu, Baole Ai, Ang Wang, Wenmeng Zhou, and Yingda Chen. Swift:a scalable lightweight infrastructure for fine-tuning. 2025. URL https://arxiv.org/abs/2408.05517.

[48] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging LLM-as-a-judge with MT-Bench and chatbot arena. arXiv preprint arXiv:2306.05685, 2023. URL https://arxiv.org/abs/2306.05685.

[49] Jefrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. Instruction-following evaluation for large language models. 2023. URL https://arxiv.org/abs/2311.07911.