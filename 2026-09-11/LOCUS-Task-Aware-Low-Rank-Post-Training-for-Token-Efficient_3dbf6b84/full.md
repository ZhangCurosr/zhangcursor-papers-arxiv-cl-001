# LOCUS: Task-Aware Low-Rank Post-Training for Token-Efficient Language Generation

Dongfang Zhao (dzhao@uw.edu)

## Abstract

Large language model serving costs scale directly with output sequence length, yet standard preference alignment often inflates response verbosity without improving utility. We study whether the parameterization of posttraining updates affects generation length: lowrank subspaces alter sequence length without modifying the alignment loss. We present LOCUS, a method that selects a task-aware low-rank adaptation subspace to minimize output-token cost subject to a utility constraint. Within this subspace, post-training retains the native preference objective with a frozen backbone. Across Anthropic HH-RLHF dialogue preferences, we evaluate two ∼3B decoder backbones, Pythia-2.8B and Qwen2.5- 3B, against protocol-matched full-parameter DPO and DrDPO branches and the released SamPO checkpoint. LOCUS reduces continuation length by up to 39.84% on Pythia-2.8B and by 14.87–17.58% on Qwen2.5-3B while updating only 0.24–0.28% of model parameters, with no material change in the internal preference diagnostic.

## 1 Introduction

Serving large language models requires substantial computational and memory resources, with perrequest latency and serving expenses governed by the number of autoregressively generated output tokens (Pope et al., 2023; Kwon et al., 2023). In production environments, each additional decoded token requires an iterative forward pass through the entire model depth, consuming memory bandwidth and expanding key-value cache storage (Leviathan et al., 2023; Dao et al., 2022). Consequently, generating overly verbose responses directly diminishes operational throughput and escalates deployment costs. Developing techniques that produce concise generations while maintaining strict answer quality represents a central objective for practical language model serving.

Post-training preference alignment methods align models with human values (Ouyang et al., 2022; Rafailov et al., 2023). Algorithms such as Direct Preference Optimization (Rafailov et al., 2023), Distributionally Robust DPO (Wu et al., 2025), and down-sampled divergence methods (Lu et al., 2024) successfully steer model outputs toward preferred behaviors. However, when optimized across all backbone parameters, these objectives frequently suffer from verbosity bias, wherein models learn that longer responses correlate with higher preference scores (Singhal et al., 2024; Saito et al., 2023; Park et al., 2024). These observations motivate examining whether the parameterization of preference updates affects generation length under an unchanged objective.

Prior attempts to curtail output length typically introduce ad-hoc regularizers or prompt modifications, yet these interventions exhibit clear deficiencies. Methods that apply heuristic length penalties or negative length rewards risk distorting the underlying preference formulation, leading to premature termination or degraded response quality (Li et al., 2026). Similarly, prompting models for conciseness relies on fragile instruction following that degrades under distribution shifts (Zhou et al., 2023). These approaches modify the learning objective or inference interface, leaving the effect of update parameterization on output length open to investigation.

Our insight toward this problem is that the parameterization of post-training updates can affect output-length dynamics: low-rank adaptation subspaces can alter sequence length without modifying the alignment objective. Low-rank adaptation (LoRA) (Hu et al., 2022) confines trainable updates to factorized matrices, making the update subspace a concrete variable for studying generation length. By freezing the pretrained backbone and confining parameter updates to structured low-rank trajectories, the model lands on a materially different quality-token operating point under the exact original preference loss.

![](images/7e897b461070b61578743bf404766757d8ccc8f8dfd52fbc956e4ebee7d81ccf.jpg)  
Figure 1: Token and accuracy summary on Pythia-2.8B.

This paper turns the above insight into a posttraining method, namely Length Optimization for Concise Utility-Preserving Sequences (LOCUS). LOCUS selects a task-aware low-rank adaptation subspace on development data to minimize outputtoken cost subject to a utility constraint. Within each candidate subspace, adaptation uses the native post-training objective with a frozen backbone. LOCUS leaves the underlying loss formulation intact: it adds no length normalization beyond what a native objective already specifies, no explicit token penalty, and no brevity prompting. With the objective fixed, LOCUS treats the low-rank subspace configuration (spanning adapter rank, module placement, layer scope, and training checkpoint step) as a structured design variable.

Figure 1 summarizes the primary empirical results of LOCUS across three preferenceoptimization objective families on the Anthropic Helpful and Harmless (HH) benchmark (Bai et al., 2022) using Pythia-2.8B (Biderman et al., 2023). Across the three HH comparisons on Pythia-2.8B, LOCUS reduces continuation tokens by 20.73% under DPO, 25.29% under DrDPO, and 39.84% under SamPO with negligible accuracy deltas (≤ 0.13 pp), updating only 0.28% of parameters. Furthermore, controlled DPO and DrDPO comparisons on Qwen2.5-3B (Yang et al., 2024) show 14.87% and 17.58% token reductions, respectively, under 0.24% trainable parameters.

In summary, this paper makes the following contributions:

• We formulate token-efficient preference posttraining as a task-aware low-rank subspace selection problem subject to an explicit utility constraint. The formulation preserves the native alignment objective; Section 3 defines the problem and selection procedure.

• We analyze the trainable-parameter count and the exact inference-time equivalence between unmerged and merged low-rank updates. Section 4 gives the propositions, proofs, and parameter accounting.

• We evaluate LOCUS on Anthropic HH-RLHF dialogue preferences with two ∼3B decoder backbones, Pythia-2.8B and Qwen2.5-3B, against protocol-matched full-parameter DPO and DrDPO branches and the released SamPO checkpoint. The protocol-matched branches yield 25.29% and 17.58% DrDPO reductions on Pythia-2.8B and Qwen2.5-3B, and continued SamPO adaptation yields 39.84%, using a small trainable subspace; the complete protocol and results appear in Section 5.

## 2 Related Work

Preference Optimization and Alignment. Preference alignment techniques train language models to align with human preferences using pairwise comparison data. Reinforcement Learning from Human Feedback (Christiano et al., 2017; Stiennon et al., 2020; Ouyang et al., 2022) optimizes reward models using policy gradient techniques. Direct Preference Optimization (Rafailov et al., 2023) eliminates the separate reward modeling stage by deriving a closed-form substitution of the reward function into the policy objective. Subsequent methods extend direct preference optimization through alternative objectives. For example, IPO (Azar et al., 2024) introduces regularized linear losses, whereas KTO (Ethayarajh et al., 2024) models prospect theory. DrDPO (Wu et al., 2025) instead addresses reference distribution shifts. Prior methods optimize objectives in the full parameter space or modify the loss formulation; LOCUS preserves the native preference loss while altering only the optimization subspace.

Length Bias and Output Conciseness. Language models aligned through preference optimization frequently develop severe length bias, generating unnecessarily long responses to maximize rewards (Singhal et al., 2024; Saito et al., 2023;

Dubois et al., 2023). Several studies investigate this phenomenon: LR-DPO (Park et al., 2024) disentangles length from quality through score adjustments, while SamPO (Lu et al., 2024) down-samples reference sequence length to diminish length reliance. $\mathrm { G R ^ { 3 } }$ (Li et al., 2026) applies group relative reward rescaling during reinforcement learning to alleviate length inflation. Prior approaches introduce adhoc length regularization terms or heuristic prompting that distort the target preference distribution, whereas LOCUS achieves token efficiency through subspace parameterization without modifying the underlying objective.

Parameter-Efficient Fine-Tuning. Parameterefficient fine-tuning (PEFT) adapts large language models by updating a small subset of parameters while freezing base weights. Prominent methodologies include adapter insertion (Houlsby et al., 2019), prefix tuning (Li and Liang, 2021), prompt tuning (Lester et al., 2021), and low-rank adaptation (Hu et al., 2022). Recent works expand lowrank tuning through quantization (Dettmers et al., 2023), weight decomposition (Liu et al., 2024), and adaptive rank allocation (Zhang et al., 2023). Theoretical investigations establish that language model adaptation possesses low intrinsic dimensionality, indicating that overparameterized updates are redundant for task learning (Aghajanyan et al., 2021; Sharma et al., 2024). Conventional parameterefficient tuning treats low-rank adaptation solely as a resource-saving approximation to full fine-tuning; in contrast, LOCUS treats the adaptation subspace as a Pareto design variable to control generation length while preserving task utility.

## 3 Methodology

Figure 2 summarizes the LOCUS procedure in three stages. First, LOCUS attaches low-rank adapters to selected modules. The pretrained backbone remains frozen, which confines parameter updates to those adapters. The adapters are then trained with the original preference objective, allowing the experiment to test whether the update parameterization changes generation length without adding a length penalty or changing the loss.

Second, LOCUS trains a small set of candidate adapters that differ in rank, target modules, target layers, or training step. It evaluates each candidate on development data and selects the shortest one that stays within the allowed utility difference from the baseline. When a confirmation split is available, the selected adapter must pass that check before the held-out test evaluation.

![](images/0b7328f95269792f80a2ae8d89dd25733f84005a7bd530e36fc9e9b5251296a3.jpg)  
Figure 2: LOCUS procedure from native-objective training to selected low-rank deployment.

Finally, LOCUS prepares the selected adapter for inference. The adapter can be merged into the backbone for a single deployment, or kept separate when one backbone serves multiple task adapters. The formal notation used in Figure 2, together with the native DPO and DrDPO objectives, is defined in Section 3.1.

## 3.1 Objective-Preserving Adaptation

We denote an input prompt by $x \in \mathcal { X }$ and an autoregressively generated continuation by $y =$ $( y _ { 1 } , y _ { 2 } , \dotsc , y _ { T } ) \in \mathcal { V }$ . We consider an autoregressive language model parameterized by frozen pretrained backbone weights $W _ { 0 } \in \mathbb { R } ^ { d _ { 1 } \times d _ { 2 } }$ . Given a task dataset D with pairwise preference comparisons $( x , y _ { w } , y _ { l } )$ , where $y _ { w }$ denotes the preferred response and $y _ { l }$ denotes the dispreferred response, standard alignment methods optimize a base objective ${ \mathcal { L } } _ { \mathrm { b a s e } } ( \theta )$ across all model parameters. For example, under Direct Preference Optimization (Rafailov et al., 2023), defining the implicit reward proxy $r _ { \theta } ( x , y ) = \beta \log [ \pi _ { \theta } ( y \mid x ) / \pi _ { \mathrm { r e f } } ( y \mid x ) ]$ , the

Algorithm 1 LOCUS Subspace Selection   
Require: $\mathcal { L } _ { \mathrm { b a s e } } , W _ { 0 } , \mathcal { D } _ { t } ^ { \mathrm { t r a i n } } , \mathcal { D } _ { t } ^ { \mathrm { s e l } } , \mathcal { D } _ { t } ^ { \mathrm { c o n f } } , \epsilon _ { t } , \mathcal { C }$   
Ensure: Selected adapter $c ^ { * }$ and policy $\pi _ { c ^ { * } }$   
1: $Q _ { t } ^ { \mathrm { b a s e } } \gets \mathrm { E v a l U t i l i t y } ( \pi _ { \mathrm { b a s e } } , ^ { \bullet } D _ { t } ^ { \mathrm { s e l } } \Big )$   
2: for $c \in { \mathcal { C } }$ do   
3: $\Delta \bar { W } _ { c } \gets \mathrm { T r a i n } ( \mathcal { L } _ { \mathrm { b a s e } } , W _ { 0 } , \mathcal { D } _ { t } ^ { \mathrm { t r a i n } } , c )$   
4: $( T _ { t } ( c ) , Q _ { t } ( c ) ) \gets \mathrm { E v a l } ( \pi _ { c } , { \mathcal D } _ { t } ^ { \mathrm { s e l } } )$   
5: end for   
6: ${ \mathcal { C } } _ { \mathrm { f e a s } } \gets \{ c \in { \mathcal { C } } \ | \ Q _ { t } ( c ) \ \geq Q _ { t } ^ { \mathrm { b a s e } } - \epsilon _ { t } \}$   
7: $c ^ { * } \gets$ arg min $\mathrel { \mathop : } \in { \mathcal { C } } _ { \mathrm { f e a s } } T _ { t } ( c )$   
8: $\mathbf { i f } \ \mathcal { D } _ { t } ^ { \mathrm { c o n f } }$ exists then   
9: $\mathsf { \bar { ( } } T _ { \mathrm { c o n f } } , Q _ { \mathrm { c o n f } } ) \gets \mathrm { E v a l } ( \pi _ { c ^ { * } } , { \mathcal { D } } _ { t } ^ { \mathrm { c o n f } } )$   
10: $Q _ { \mathrm { c o n f } } ^ { \mathrm { b a s e } }  \mathrm { E v a l U t i l i t y } ( \pi _ { \mathrm { b a s e } } , D _ { t } ^ { \mathrm { c o n f } } )$   
11: $T _ { \mathrm { c o n f } } ^ { \mathrm { b a s e } }  \mathrm { E v a l L e n g t h } ( \pi _ { \mathrm { b a s e } } , { \mathcal D } _ { t } ^ { \mathrm { c o n f } } )$   
12: u $ [ Q _ { \mathrm { c o n f } } \geq Q _ { \mathrm { c o n f } } ^ { \mathrm { b a s e } } - \epsilon _ { t } ]$   
13: $\ell \gets [ T _ { \mathrm { c o n f } } < T _ { \mathrm { c o n f } } ^ { \mathrm { b a s e } } ]$   
14: if u ∧ ℓ then   
15: return Frozen adapter $c ^ { * }$ and model $\pi _ { c ^ { * } }$   
16: else   
17: return Fallback baseline π<sub>base</sub>   
18: end if   
19: else   
20: return Unconfirmed adapter $c ^ { * }$   
21: end if

objective is:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { D P O } } ( \boldsymbol { \theta } ) = - \mathbb { E } _ { \mathcal { D } } \left[ \log \sigma \left( r _ { \boldsymbol { \theta } } ( x , y _ { w } ) - r _ { \boldsymbol { \theta } } ( x , y _ { l } ) \right) \right] , } \end{array}\tag{1}
$$

where $\beta > 0$ controls the divergence penalty from the reference policy $\pi _ { \mathrm { r e f } }$ . The function σ denotes the sigmoid. Under Distributionally Robust DPO (Wu et al., 2025), the objective robustifies against sample-level reference divergence:

$$
\mathcal { L } _ { \mathrm { D r D P O } } ( \theta ) = - \beta ^ { \prime } \log \left( \frac { 1 } { B } \sum _ { i = 1 } ^ { B } e ^ { - \ell _ { i } ( \theta ) / \beta ^ { \prime } } \right)\tag{2}
$$

where $\ell _ { i } ( \theta )$ denotes the individual pairwise loss for sample i in a microbatch of size B. The parameter $\beta ^ { \prime }$ controls distributional robustness.

In LOCUS, we leave the native objective $\mathcal { L } _ { \mathrm { b a s e } }$ strictly intact. We restrict trainable updates through the LoRA parameterization in Eq. (3):

$$
W = W _ { 0 } + \Delta W = W _ { 0 } + \frac { \alpha } { r } B A ,\tag{3}
$$

where $A \in \mathbb { R } ^ { r \times d _ { 2 } } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } )$ and $B \in \mathbb { R } ^ { d _ { 1 } \times r }$ is zero-initialized. The adaptation uses rank $r \ll \operatorname* { m i n } ( d _ { 1 } , d _ { 2 } )$ with scaling factor $\alpha .$ . We define an adaptation subspace configuration as a tuple $c \ = \ ( r , \alpha , \mathcal { P } , \mathcal { L } , s ) \ \in \ \mathcal { C }$ The sets $\mathcal { P } \subseteq$ $\{ \mathbf { Q } , \mathbf { K } , \mathbf { V } , \mathbf { O } , \mathbf { M } \mathbf { L } \mathbf { P } \}$ and $\mathcal { L } \subseteq \{ 1 , \ldots , L \}$ specify the targeted weight modules and layer indices, respectively. For these targets, s identifies the training optimization checkpoint step. The resulting selection procedure is summarized in Algorithm 1.

It first measures baseline utility on the task’s selection split. Each candidate adapter is then trained with the same native objective and evaluated on that split for continuation length and preference utility. LOCUS discards candidates whose utility falls beyond the allowed tolerance, then selects the shortest remaining candidate. When a confirmation split is available, the selected candidate must also reduce length without exceeding the utility tolerance; otherwise, LOCUS falls back to the baseline. If no confirmation split is available, the selected adapter is returned as unconfirmed.

## 3.2 Task-Aware Constrained Selection

Different subspace configurations $c \in { \mathcal { C } }$ produce distinct behavioral trajectories during autoregressive decoding. Define $T _ { t } ( c )$ as the mean continuation token length produced by configuration c under greedy decoding on task t. The corresponding task utility is denoted by $Q _ { t } ( c )$ . For candidate selection, $Q _ { t } ( c )$ is evaluated using held-out chosen-versusrejected sequence log-probability preference accuracy over $( x , y _ { w } , y _ { l } ) \in \mathcal { D } _ { t }$

$$
Q _ { t } ( c ) = \frac { 1 } { | \mathscr { D } _ { t } | } \sum _ { ( x , y _ { w } , y _ { l } ) } \mathbb { I } ( \pi _ { c } ( y _ { w } \mid x ) > \pi _ { c } ( y _ { l } \mid x ) ) .\tag{4}
$$

LOCUS identifies the optimal adaptation configuration $c _ { t } ^ { * }$ by solving a constrained minimization problem:

$$
c _ { t } ^ { * } = \arg \operatorname* { m i n } _ { c \in \mathcal { C } } T _ { t } ( c )\tag{5}
$$

where $Q _ { t } ^ { \mathrm { b a s e } }$ is the benchmark utility achieved by the protocol-appropriate baseline on the same task. The tolerance $\epsilon _ { t } \geq 0$ specifies the allowed utility decrease from that baseline (set to 1.0 percentage points in this work).

To prevent split overfitting, LOCUS partitions development data into disjoint selection $( \mathbf { \nabla } { \mathcal { D } } _ { t } ^ { \mathrm { s e l } } )$ and confirmation $( \mathcal { D } _ { t } ^ { \mathrm { c o n f } } )$ splits. Candidate configurations in C are screened on $\mathcal { D } _ { t } ^ { \mathrm { s e l } }$ via Eq. (5); when available, the top candidate $c ^ { * }$ is verified on $\mathcal { D } _ { t } ^ { \mathrm { c o n f } }$ before being frozen for held-out evaluation.

## 3.3 Complexity and Serving Modalities

The trainable parameter and optimizer-state footprint of LOCUS scales with the low-rank factors. For a targeted module, low-rank adaptation exposes $r ( d _ { 1 } + d _ { 2 } )$ trainable entries; full-parameter adaptation exposes $d _ { 1 } d _ { 2 }$ . Consequently, gradient storage and optimizer states shrink with the trainable set. Under an Adam-style two-state optimizer, the primary adapter requires two state tensors over 7.86M entries for Pythia-2.8B or 7.37M for Qwen2.5-3B. This accounting assumes two optimizer states per trainable entry; actual state storage depends on each experiment’s optimizer. Because the frozen backbone also requires forward computation and activation backpropagation, total memory and training FLOPs require separate assessment.

During serving, LOCUS supports two deployment modalities. In single-tenant deployments, low-rank weights are pre-merged via $W ^ { * } = W _ { 0 } +$ ${ } _ { r } ^ { \alpha } B ^ { * } A ^ { * }$ prior to execution, incurring zero adapterinduced latency or parameter overhead. In multitenant deployments, a single frozen backbone dynamically serves multiple task adapters, though active adapters retain bookkeeping and memory overhead not benchmarked here.

## 4 Analysis

## 4.1 Trainable Parameterization

Because LOCUS restricts every update to a lowrank subspace, the size of that restriction deserves a precise statement. Full-parameter fine-tuning gives the optimizer one free parameter per entry of a weight matrix, whereas the low-rank parameterization gives it only the entries of the two factors: on $\mathrm { ~ a ~ } 2 5 6 0 \times 2 5 6 0$ module at $r = 1 6 , 8 1 , 9 2 0$ trainable entries rather than $6 { , } 5 5 3 { , } 6 0 0$ . The proposition below states the count for a general module, with the remark that follows bounding what the count can be taken to mean.

Proposition 1 (Trainable parameter-count reduction). Let $W \in \mathbb { R } ^ { d _ { 1 } \times d _ { 2 } }$ be a targeted linear module and let r be a positive integer. Full-parameter fine-tuning exposes $d _ { 1 } d _ { 2 }$ trainable scalar entries, whereas the LoRA parameterization $W = W _ { 0 } +$ $( \alpha / r ) B A$ , with fixed $W _ { 0 } ,$ , fixed α $\neq 0 , \boldsymbol { B } \in \mathbb { R } ^ { d _ { 1 } \times r } ,$ and $A \ \in \ \mathbb { R } ^ { r \times d _ { 2 } }$ , exposes $r ( d _ { 1 } + d _ { 2 } )$ trainable factor entries. Thus the factor parameterization hasfewer exposed trainable entries exactly when $r ( d _ { 1 } + d _ { 2 } ) < d _ { 1 } d _ { 2 }$

Proof. The complete proof, including the distinction between exposed factor entries and the intrinsic dimension of the represented update set, is given in Appendix A. □

Remark 1 (Interpretation of the count). The proposition is an implementation-level parameter-count statement. It does not claim that the low-rank factors select particular semantic or verbosity-specific directions, nor that the count alone determines generation length, preference accuracy, total memory, or training speed.

For the selected Pythia-2.8B attention configuration $( L = 3 2 , d = 2 5 6 0 , r = 1 6 )$ , the combined adapter accounts for $N _ { \mathrm { a t t n } } ~ = ~ L r ( ( 3 d +$ $d ) + ( d + d ) ) = 7 , 8 6 4 , 3 2 0$ trainable parameters (0.2826% of the model). Similarly, for Qwen2.5- 3B (L = 36, r = 16 on q\_proj, k\_proj, v\_proj, o\_proj), the adapter comprises 7,372,800 trainable parameters (0.2383% of 3.09B). These calculations explain the parameter counts in Table 1. Translating those counts into end-to-end memory or latency requires measurements of the execution path.

## 4.2 Inference-Time Merge Equivalence

A second property is required for the deployment discussion: after training, LOCUS may either keep an adapter branch or merge its update into the corresponding frozen linear module. The following proposition identifies the exact condition under which these two inference representations compute the same function.

Proposition 2 (Exact equivalence after adapter merging). Consider a linear module with input h, frozen weight $W _ { 0 } ,$ and a LoRA update $( \alpha / r ) B A$ At inference time, with adapter dropout disabled, the unmerged computation and the computation using the merged weight $W ^ { * } = W _ { 0 } + \left( \alpha / r \right)$ BA produce exactly the same module output for every input h.

Proof. The complete algebraic proof and its assumptions are given in Appendix A. □

Remark 2 (Scope of merge equivalence). The equality concerns the output of a linear module in inference mode. Deployments that produce this same output can have different latency and memory costs because an unmerged adapter adds computation and bookkeeping. Furthermore, finite-precision or quantized implementations may introduce small numerical differences.

## 5 Evaluation

## 5.1 Experimental Setup

Backbones. We evaluate LOCUS on two decoder-only backbone families at roughly 3B parameter scale: Pythia-2.8B (Biderman et al., 2023), based on GPT-NeoX (Black et al., 2022), and Qwen2.5-3B (Yang et al., 2024), a modern Llamastyle architecture with grouped-query attention, SwiGLU (Shazeer, 2020), and RMSNorm (Zhang and Sennrich, 2019). Appendix Table 1 details the architectural differences and target modules for each backbone.

Baselines. We compare LOCUS with a protocolmatched full-parameter branch while fixing the training pool, starting checkpoint, native objective, and data splits. Controlled DPO (Rafailov et al., 2023) and DrDPO (Wu et al., 2025) first train a shared full-parameter supervised fine-tuned (SFT) checkpoint, then compare full-parameter preference optimization with low-rank LOCUS from that checkpoint. SamPO (Lu et al., 2024) instead continues from the official released Pythia-2.8B HH-RLHF Iterative SamPO checkpoint, while Qwen2.5-3B repeats the shared-SFT comparison under both native objectives. Appendix Table 2 makes every starting point and branch parameterization explicit.

Datasets. Primary evaluation uses Anthropic Helpful and Harmless (HH-RLHF) (Bai et al., 2022), an open-ended multi-turn dialogue preference benchmark. DPO and DrDPO use 8,552 frozen test pairs, while SamPO uses its established 256-example split. The additional datasets are Anthropic HH-RLHF harmless-base for safety refusal and Orca DPO (Mukherjee et al., 2023) for single-turn instruction following. Each uses a separate 256-example development evaluation; neither is presented as an external test result. Appendix Table 3 records these roles.

Platform. All training and inference runs use one NVIDIA A100-PCIE-80GB GPU with bfloat16 mixed precision. Activation checkpointing is enabled for the DPO and DrDPO full-parameter runs with 512-token sequences, whereas the SamPO comparison follows the published recipe. The approximately 3B scale makes the full-vs-low-rank protocol reproducible on this hardware.

Metrics. We measure continuation length and held-out preference accuracy. Generation uses greedy decoding (do\_sample = False, max\_new\_tokens = 256). We count continuation tokens exclusive of the end-of-sequence (EOS) token and record 256-token limit hits. Preference accuracy compares chosen and rejected sequence logprobabilities under the evaluated policy (Eq. (4)).

![](images/36f25b3d9a84ac2c7627adf749f7583ce94981d4aaf1d172c23ba86971bc8ba9.jpg)

![](images/bf30629146911d84b4d90ddc9a546775a523baee2721698b3e4cef4acb58b016.jpg)  
Figure 3: Constrained checkpoint selection.

![](images/cdb24eb176cc8da9f51d54c0d563d529a66672721d69a2ed6acb976f1689a36c.jpg)

![](images/3054ac38eb55640bda6e4897588e24bd6e860acac896de833e16e1b3dab4ae09.jpg)  
Figure 4: Continuation length CDF.

We choose this metric because fixed preference labels and an explicit scoring rule enable reproducible model comparisons.

## 5.2 Baseline Comparisons

Figure 3 reports the measured 250-, 500-, and 750-step selection-dev checkpoints for DPO and DrDPO, with full-parameter baselines at 146.76 and 150.48 tokens. For DPO on Pythia-2.8B, both branches train on 160,544 Anthropic HH pairs with β = 0.1 from the same SFT checkpoint; DPO-FULL updates all weights, whereas DPO-LOCUS uses a low-rank adapter. On the 8,552 frozen test pairs, LOCUS reduces mean continuation length from 137.67 to 109.12 tokens (20.73%) and changes preference accuracy from 48.40% to 48.39% (−0.01 pp), using 7,864,320 trainable parameters (0.2826%). Despite nonmonotonic trajectories, utility-constrained selection chooses step 750 in both comparisons.

Figure 4 presents the continuation-length CDF across the 8,552 frozen test prompts. Under Distributionally Robust DPO (DrDPO) on Pythia-2.8B (Wu et al., 2025), DRDPO-FULL updates all weights while DRDPO-LOCUS updates only its adapter from the same SFT checkpoint, using $\beta \ : = \ : 0 . 1$ and $\beta ^ { \prime } = 1 . 0$ . LOCUS reduces mean length from 145.61 to 108.79 tokens (25.29%) and changes preference accuracy from 48.39% to 48.26% (−0.13 pp); the selected step-750 candidate reaches 26.70% reduction on selection-dev. The median falls from 100 to 52 tokens under DPO and from 125 to 52 under DrDPO, while limit hits fall by 24.70% and 30.98%.

For SamPO, LOCUS starts from the official Pythia-2.8B Anthropic HH Iterative SamPO checkpoint (Lu et al., 2024), which incorporates pertoken length normalization into preference optimization. On the 256-example HH evaluation split, the released checkpoint produces a mean continuation of 132.77 tokens with 53.52% preference accuracy. We train a low-rank adapter with $r = 1 6$ and $\alpha = 3 2$ on attention modules under the native SamPO objective $( \beta = 0 . 0 5 )$ . As summarized in Figure 1 and Appendix Table 4, LOCUS reduces mean continuation length from 132.77 to 79.88 tokens (52.89 tokens; 39.84%) while preserving 53.52% preference accuracy $( \Delta = 0 . 0 0 \mathrm { p p } )$ .

## 5.3 Rank Sensitivity

Using the Pythia-2.8B backbone, we measure sensitivity to adapter rank $r ~ \in ~ \{ 4 , 8 , 1 6 , 3 2 \}$ on a 64-example Anthropic HH screening subset under attention adaptation with the SamPO objective $( \alpha / r = 2 )$ . Rank controls the dimensionality of the trainable update, so we keep the target modules, objective, data split, and scaling rule fixed while varying only r. This screen uses a separate subset from the candidate pool C evaluated in Section 5, so it characterizes rank sensitivity without redetermining that selection. Figure 5 reports the resulting token changes and internal preference diagnostics.

Because the objective, attention placement, data split, and step budget are fixed, this screen isolates the effect of rank within the candidate adapter subspace. At rank $r = 4$ , the model exhibits length inflation (+11.74% tokens), whereas $r \ = \ 8$ produces a small reduction of 2.32%. The reduction increases to 15.95% at $r ~ = ~ 1 6$ and 25.10% at $r = 3 2$ (Figure 5(a)), so the measured length effect varies materially across the tested ranks. The internal preference diagnostic remains 45.31% for $r \in \{ 4 , 8 , 1 6 \}$ } and rises to 46.88% at r = 32 (Figure 5(b)); within this screen, token reduction therefore does not follow a monotonic change in that diagnostic.

![](images/f87dd8701358dde52bb4daf6f1ffb065e0abf1c4a0e58f02fb11fa4792fbdf4e.jpg)

![](images/19baa52f012d127109a5bd708f349438ad2474b7869f18e4d4bcbccc0ba4fa51.jpg)  
Figure 5: Sensitivity to adapter rank.

![](images/65247c1c1747ab4393ca20bcbfc3761252a6e1c0dfa26aebb0245bc21e9f9234.jpg)

![](images/72161ccde7499d05c98c3ca3ae6b936120108595ea0501befc1e7049b1f88190.jpg)  
Figure 6: Target-module ablation.

## 5.4 Target-Module Ablation

Using Pythia-2.8B, we perform a controlled ablation over which existing linear projections receive LoRA updates on a 64-example Anthropic HH screening subset with $r = 1 6 .$ , keeping the backbone and layer coverage fixed. The five candidate target sets are query-key-value projections only (QKV), attention output projections only (O), their combination (Attention-All), feed-forward network projections only (MLP), and all linear projections (All-Linear).

Figure 6 compares token reduction with trainable parameter count across the five target sets. QKV and output-only adaptation yield 8.09% and 4.96% reductions, while Attention-All reaches 15.95%. MLP adaptation yields 5.75%; All-Linear reaches 13.88% with 2.7× more trainable parameters than Attention-All (20.97M vs. 7.86M). The internal preference diagnostic remains 45.31% across all placements.

![](images/a2f714eda940d7816a5d43aca7cb31b7fe8a0e8f4971e5e8c5f1cecce1b394bc.jpg)  
Figure 7: Cross-task evaluations.

![](images/f3ba29c445d6058980bec9757c6b517d72139d2d0f884333fd218933ded8887c.jpg)  
Δ Accuracy (pp)

![](images/ca67d99833ef3807684c90f2dfb7d48b1ba92df5b71aa42680e1e8513c3515e9.jpg)

## 5.5 Cross-Task Generality

![](images/825a193cafe6a17040c1b836efff961b8cd3b1884e59b0d4167216bb83428d75.jpg)  
Figure 8: Cross-backbone token and accuracy results.

We test whether the measured token reduction extends beyond the HH dialogue setting when the preference data and task semantics change. Using Pythia-2.8B, we evaluate the HH comparison alongside Anthropic HH-RLHF harmless-base and the Orca DPO instruction-following dataset (Figure 7). These evaluations cover dialogue, safety, and instruction-following settings, each with a taskspecific adapter; the HH row reports the 8,552-pair frozen-test result, Harmless and Orca DPO each use a 256-example development split, and the preference numbers remain internal chosen-vs-rejected diagnostics rather than external quality evaluations.

Figure 7(a) reports task-specific length changes. The HH comparison reduces mean continuation length from 137.67 to 109.12 tokens (20.73%). The Harmless evaluation reduces it from 137.53 to 102.75 tokens (25.29%), while the Orca DPO evaluation reduces it from 175.95 to 35.25 tokens (79.97%). Figure 7(b) reports the corresponding internal preference diagnostics: HH changes from 48.40% to 48.39% (−0.01 pp), Harmless changes from 57.03% to 58.20% (+1.17 pp), and Orca DPO changes from 69.14% to 82.81% (+13.67 pp).

## 5.6 Cross-Backbone Transfer to Qwen2.5-3B

We test whether the token-reduction effect transfers across architectural families on Qwen2.5-3B (Yang et al., 2024), a distinct decoder backbone family at comparable ∼3B scale. Following the identical experimental discipline, both branches begin from a single shared full-parameter SFT checkpoint trained for one epoch on the 160,288 Anthropic HH training pairs. Candidate low-rank adapters $( r = 1 6 , \alpha = 3 2$ on the attention projections of all 36 layers; 7.37M parameters, 0.2383%) are screened on the 256-pair selection split and confirmed on a disjoint 256-pair split before frozen

held-out evaluation.

Figure 8 reports controlled frozen test results on the 8,552 Anthropic HH test pairs for both DPO and DrDPO alongside the established Pythia-2.8B baselines. Under DPO $( \beta = 0 . 1 )$ , LOCUS (step 250) reduces mean continuation length from 108.26 to 92.16 tokens (14.87% reduction; median 75 to 51 tokens), while held-out preference accuracy changes from 48.69% to 48.64% (−0.05 pp delta). Under DrDPO $( \beta = 0 . 1 , \beta ^ { \prime } = 1 . 0 )$ , LO-CUS (step 500) reduces mean continuation length from 111.58 to 91.97 tokens (17.58% reduction; median 81 to 52 tokens), with preference accuracy changing from 48.64% to 48.55% (−0.09 pp delta). Both comparisons meet the strong positive criterion of at least 2% token reduction with a utility delta of at least −1.0 pp.

## 6 Conclusion

This paper presents LOCUS, which freezes the pretrained backbone and selects a low-rank adaptation subspace while preserving the native preference objective. On Anthropic HH-RLHF, we compare LOCUS with protocol-matched full-parameter DPO and DrDPO branches and with the released SamPO checkpoint on Pythia-2.8B, then repeat the comparisons on Qwen2.5-3B. Each comparison matches the objective, checkpoint, data split, and decoding protocol, measuring continuation length and internal chosen-vs-rejected preference accuracy. The protocol-matched DrDPO branch yields 25.29% and 17.58% reductions on Pythia-2.8B and Qwen2.5-3B, and continued SamPO adaptation yields 39.84%, all while updating fewer than 0.3% of model parameters. Additional development evaluations extend the measured reductions to safety and instruction-following data under the same internal preference diagnostic.

## Limitations

LOCUS selects a low-rank adaptation subspace for post-training, with the following limitations. The evaluation covers two decoder-only backbone families at roughly 3B scale, Pythia-2.8B and Qwen2.5- 3B, under greedy decoding. Larger models and sampling-based inference remain untested.

The candidate pool C uses four discrete ranks and coarse projection groupings. Continuous rank allocation, finer layer selection, and gradientinformed pruning remain outside the study.

The cross-task evaluations broaden coverage across dialogue, safety, and instruction following, but do not predict the reduction available on an unseen task.

## References

Armen Aghajanyan, Sonal Gupta, and Luke Zettlemoyer. 2021. Intrinsic dimensionality explains the effectiveness of language model fine-tuning. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics, pages 7319– 7328.

Mohammad Gheshlaghi Azar, Zhaohan Daniel Guo, Bilal Piot, Remi Munos, Mark Rowland, Michal Valko, and Daniele Calandriello. 2024. A general theoretical paradigm to understand learning from human preferences. In Proceedings of the International Confer ence on Artificial Intelligence and Statistics, volume 238, pages 4447–4455.

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, Nicholas Joseph, Saurav Kadavath, Jackson Kernion, Tom Conerly, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Tristan Hume, and 12 others. 2022. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862.

Stella Biderman, Hailey Schoelkopf, Quentin Gregory Anthony, Herbie Bradley, Kyle O’Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, Aviya Skowron, Lintang Sutawika, and Oskar van der Wal. 2023. Pythia: A suite for analyzing large language models across training and scaling. In Proceedings ofthe 40th International Conference on Machine Learning, pages 2397–2430.

Sid Black, Stella Biderman, Eric Hallahan, Quentin Anthony, Leo Gao, Laurence Golding, Horace He, Connor Leahy, Kyle McDonell, Jason Phang, Michael Pieler, USVSN Sai Prashanth, Shivanshu Purohit, Laria Reynolds, Jonathan Tow, Ben Wang, and

Samuel Weinbach. 2022. GPT-NeoX-20B: An opensource autoregressive language model. In Proceedings ofBigScience Episode #5 – Workshop on Challenges & Perspectives in Creating Large Language Models, pages 95–136.

Paul F. Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. 2017. Deep reinforcement learning from human preferences. In Advances in Neural Information Processing Systems, volume 30, pages 4299–4307.

Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. 2022. FlashAttention: Fast and memory-efficient exact attention with IO-awareness. In Advances in Neural Information Processing Systems, volume 35, pages 16344–16359.

Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. 2023. QLoRA: Efficient finetuning of quantized LLMs. In Advances in Neural Information Processing Systems, volume 36, pages 10088–10115.

Yann Dubois, Xuechen Li, Rohan Taori, Tianyi Zhang, Ishaan Gulrajani, Jimmy Ba, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. 2023. Alpaca-Farm: A simulation framework for methods that learn from human feedback. In Advances in Neural Information Processing Systems, volume 36, pages 30039– 30069.

Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, and Douwe Kiela. 2024. Model alignment as prospect theoretic optimization. In Proceedings ofthe 41st International Conference on Machine Learning, pages 12634–12651.

Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin De Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. 2019. Parameter-efficient transfer learning for NLP. In Proceedings ofthe 36th International Conference on Machine Learning, pages 2790–2799.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-rank adaptation of large language models. In Proceedings of the International Conference on Learning Representations.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient memory management for large language model serving with PagedAttention. In Proceedings ofthe 29th Symposium on Operating Systems Principles, pages 611–626.

Brian Lester, Rami Al-Rfou, and Noah Constant. 2021. The power of scale for parameter-efficient prompt tuning. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 3045–3059.

Yaniv Leviathan, Matan Kalman, and Yossi Matias. 2023. Fast inference from transformers via speculative decoding. In Proceedings of the 40th International Conference on Machine Learning, pages 19274–19286.

Xiang Lisa Li and Percy Liang. 2021. Prefix-tuning: Optimizing continuous prompts for generation. In Proceedings ofthe 59th Annual Meeting ofthe Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, pages 4582–4597.

Zichao Li, Jie Lou, Fangchen Dong, Zhiyuan Fan, Mengjie Ren, Hongyu Lin, Xianpei Han, Debing Zhang, Le Sun, Yaojie Lu, and Xing Yu. 2026. Tackling length inflation without trade-offs: Group relative reward rescaling for reinforcement learning. arXiv preprint arXiv:2603.10535.

Shih-Yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo Molchanov, Yu-Chiang Frank Wang, Kwang-Ting Cheng, and Min-Hung Chen. 2024. DoRA: Weightdecomposed low-rank adaptation. In Proceedings of the 41st International Conference on Machine Learning, pages 32100–32121.

Junru Lu, Jiazheng Li, Siyu An, Meng Zhao, Yulan He, Di Yin, and Xing Sun. 2024. Eliminating biased length reliance of direct preference optimization via down-sampled KL divergence. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 1047–1067, Miami, Florida, USA. Association for Computational Linguistics.

Subhabrata Mukherjee, Arindam Mitra, Ganesh Jawahar, Sahaj Agarwal, Hamid Palangi, and Ahmed Awadallah. 2023. Orca: Progressive learning from complex explanation traces of GPT-4. arXiv preprint arXiv:2306.02707.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F. Christiano, Jan Leike, and Ryan Lowe. 2022. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems, volume 35, pages 27730–27744.

Ryan Park, Rafael Rafailov, Stefano Ermon, and Chelsea Finn. 2024. Disentangling length from quality in direct preference optimization. In Findings of the Associationfor Computational Linguistics: ACL 2024, pages 4998–5017, Bangkok, Thailand. Association for Computational Linguistics.

Reiner Pope, Sholto Douglas, Aakanksha Chowdhery, Jacob Devlin, James Bradbury, Jonathan Heek, Kefan Xiao, Shivani Agrawal, and Jeff Dean. 2023. Efficiently scaling transformer inference. In Proceedings of Machine Learning and Systems, volume 5, pages 606–624.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D. Manning, Stefano Ermon, and Chelsea Finn. 2023. Direct preference optimization: Your language model is secretly a reward model. In Advances in Neural Information Processing Systems, volume 36, pages 53728–53741.

Keita Saito, Akifumi Wachi, Koki Wataoka, and Youhei Akimoto. 2023. Verbosity bias in preference labeling by large language models. arXiv preprint arXiv:2310.10076.

Pratyusha Sharma, Jordan T. Ash, and Dipendra Misra. 2024. The truth is in there: Improving reasoning in language models with layer-selective rank reduction. In Proceedings of the International Conference on Learning Representations.

Noam Shazeer. 2020. GLU variants improve transformer. arXiv preprint arXiv:2002.05202.

Prasann Singhal, Tanya Goyal, Jiacheng Xu, and Greg Durrett. 2024. A long way to go: Investigating length correlations in RLHF. In First Conference on Language Modeling.

Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F. Christiano. 2020. Learning to summarize with human feedback. In Advances in Neural Information Processing Systems, volume 33, pages 3008–3021.

Junkang Wu, Yuexiang Xie, Zhengyi Yang, Jiancan Wu, Jiawei Chen, Jinyang Gao, Bolin Ding, Xiang Wang, and Xiangnan He. 2025. Towards robust alignment of language models: Distributionally robustifying direct preference optimization. In Proceedings of the International Conference on Learning Representations.

An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, et al. 2024. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115.

Biao Zhang and Rico Sennrich. 2019. Root mean square layer normalization. In Advances in Neural Information Processing Systems, volume 32, pages 12360– 12371.

Qingru Zhang, Minshuo Chen, Alexander Bukharin, Pengcheng He, Yu Cheng, Weizhu Chen, and Tuo Zhao. 2023. Adaptive budget allocation for parameter-efficient fine-tuning. In Proceedings of the International Conference on Learning Representations.

Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. 2023. Instruction-following evaluation for large language models. arXiv preprint arXiv:2311.07911.

## A Formal Proofs of Propositions 1 and 2

This appendix gives the formal proofs of Propositions 1 and 2, together with remarks clarifying their scope.

Proof of Proposition 1. Let $\mathcal { W } = \mathbb { R } ^ { d _ { 1 } \times d _ { 2 } }$ denote the space of matrices that can be updated by a targeted module. Under full-parameter fine-tuning, write the update as $\Delta W \in \mathcal { W }$ . For the standard matrix basis $\{ E _ { i j } : 1 \le i \le d _ { 1 } , 1 \le j \le d _ { 2 } \}$ every update has the unique expansion

$$
\Delta W = \sum _ { i = 1 } ^ { d _ { 1 } } \sum _ { j = 1 } ^ { d _ { 2 } } \delta _ { i j } E _ { i j } .
$$

The coefficients $\delta _ { i j }$ are independent trainable scalars. Therefore, the full-parameter representation exposes exactly $d _ { 1 } d _ { 2 }$ trainable entries.

For LoRA, define the factor parameter space

$$
\mathcal { P } = \mathbb { R } ^ { d _ { 1 } \times r } \times \mathbb { R } ^ { r \times d _ { 2 } }
$$

and the update map $\phi : \mathcal { P }  \mathcal { W }$ by

$$
\phi ( B , A ) = { \frac { \alpha } { r } } B A .
$$

The first factor contains $d _ { 1 } r$ scalar entries and the second contains $r d _ { 2 }$ scalar entries. Because $\alpha / r \neq$ 0 is fixed, it introduces no trainable scalar and does not change the number of entries exposed to the optimizer. Therefore,

$$
\dim _ { \operatorname { e n t r i e s } } ( { \mathcal { P } } ) = d _ { 1 } r + r d _ { 2 } = r ( d _ { 1 } + d _ { 2 } ) .
$$

Moreover, for every $( B , A ) \in { \mathcal { P } }$ and indices $i , j ,$

$$
\phi ( B , A ) _ { i j } = \frac { \alpha } { r } \sum _ { k = 1 } ^ { r } B _ { i k } A _ { k j } ,
$$

so rank $( \phi ( B , A ) ) \leq r .$ This establishes both the claimed factor-entry count and the rank constraint on the represented update.

The factorization map need not be injective: for every invertible $G \in \mathbb { R } ^ { r \times r }$

$$
\phi ( B G , G ^ { - 1 } A ) = { \frac { \alpha } { r } } B G G ^ { - 1 } A = \phi ( B , A ) .
$$

The identity shows that distinct factor pairs can represent the same matrix. At full-rank factors, this change-of-basis redundancy has $r ^ { 2 }$ dimensions. Accounting for that redundancy gives a local dimension of $r ( d _ { 1 } + d _ { 2 } - r )$ for the represented rank-r manifold. The optimizer stores gradients and (when applicable) state tensors for the individual entries of B and A. Consequently, the implementation-level trainable-parameter count remains $r ( d _ { 1 } + d _ { 2 } )$

Comparing this count with the $d _ { 1 } d _ { 2 }$ independent entries of full fine-tuning gives a strict reduction if and only if

$$
r ( d _ { 1 } + d _ { 2 } ) < d _ { 1 } d _ { 2 } .
$$

This completes the proof.

□

Remark 3 (Interpretation and scope). The proposition is a parameter-count identity; it does not claim that LoRA projects updates onto semantic or verbosity-specific singular directions. The usual initialization with $B = 0$ makes the initial effective update zero, while later update directions are determined by optimization through the factorized parameterization. Consequently, parameter reduction alone does not imply a particular change in generation length, preference accuracy, total memory, or training speed; those quantities are measured empirically in Section 5.

ProofofProposition 2. Let the common bias term, if present, be b. The unmerged inference path computes

$$
z _ { \mathrm { u n m e r g e d } } = W _ { 0 } h + \frac { \alpha } { r } B ( A h ) + b .
$$

By associativity and distributivity of matrix multiplication,

$$
{ \begin{array} { r } { z _ { \mathrm { u n m e r g e d } } = \left( W _ { 0 } + { \frac { \alpha } { r } } B A \right) h + b } \\ { = W ^ { * } h + b = z _ { \mathrm { m e r g e d } } . } \end{array} }
$$

The equality holds for every admissible input vector $h$ and does not depend on the values or rank of B and $A ,$ provided $r \neq 0$ and the scalar α is fixed. If the module is embedded in a deterministic downstream network whose parameters and subsequent operations are unchanged, equal module outputs induce equal downstream activations and logits by composition. The condition that adapter dropout is disabled is required because a stochastic dropout mask inserted into the unmerged branch is not represented by the fixed merged matrix. □

Remark 4 (Deployment scope). The result establishes exact algebraic equivalence for a single merged linear module in inference mode. It does not establish an end-to-end systems benefit: an unmerged multi-tenant deployment still incurs adapter-specific computation and bookkeeping, while numerical effects from quantization or finite-precision operation order may produce small implementation-level differences.

## B Experimental Reference Tables

Table 1 describes the two backbone architectures and their attention-adapter parameterizations. Pythia uses fused query-key-value projections, whereas Qwen uses separate query, key, and value projections with grouped-query attention. Consequently, the same attention-adaptation scope exposes different parameter counts: 7.86M for Pythia and 7.37M for Qwen. The table relates these counts to the module layouts used in the cross-backbone comparison.

Table 2 defines the starting checkpoint and update parameterization of each comparison branch. For controlled DPO and DrDPO, both branches start from a shared full-parameter SFT checkpoint and retain the same native preference objective. The full-parameter branch updates the backbone, whereas LOCUS updates its low-rank adapter. The SamPO row identifies a separate continued-adaptation comparison against the official released checkpoint.

Table 3 distinguishes the training pools, development splits, and final evaluation sets. The controlled HH comparisons use development data for selection, with separate confirmation where specified, before evaluation on the 8,552 frozen test pairs. The SamPO comparison uses its established 256-example evaluation split. The additional Anthropic Harmless and Orca DPO rows identify the safety-refusal and instruction-following datasets. Their evaluation scope is limited to the corresponding 256-example development splits; no external frozen-test claim is made for these rows.

## C Additional Experimental Results

This appendix records the primary HH comparisons and the cross-task development evaluations reported in Section 5. Table 4 consolidates the primary Anthropic HH comparisons across both backbone families and the three preference objectives. For DPO and DrDPO, both branches start from the same SFT checkpoint; for SamPO, LOCUS adapts the released SamPO checkpoint. The table reports the measured token and internal preferencediagnostic changes without introducing an additional evaluation protocol.

Table 5 gives the exact values behind the crosstask comparison. The Anthropic Helpful and Harmless row reproduces the 8,552-pair frozen-test result from Table 4; the Harmless and Orca DPO rows use 256-example development evaluations. In every row, the preference values are internal chosen-vs-rejected diagnostics rather than external quality measurements.

<table><tr><td>Backbone</td><td>Architecture</td><td>LOCUS targets</td><td>Trainable parame- ters</td></tr><tr><td>Pythia-2.8B</td><td>2560; parallel attention/MLP blocks. ules.</td><td>GPT-NeoX; 32 layers; hidden size Fused QKV and attention output mod- 7.86M (0.2826%).</td><td></td></tr><tr><td>Qwen2.5-3B</td><td>layers; hidden size 2048; grouped- o_proj. query attention; SwiGLU; RM-</td><td>Modern Llama-style architecture; 36 Separate q_proj, k_proj, v_proj, and 7.37M (0.2383%).</td><td></td></tr></table>

Table 1: Backbones and backbone-specific LOCUS parameterizations.
<table><tr><td>Comparison</td><td>Starting point</td><td>Full-parameter branch</td><td>Low-rank LOCUS branch</td></tr><tr><td>Pythia DPO</td><td>Shared full-parameter SFT.</td><td>objective.</td><td>Full-parameter DPO with native Low-rank DPO with same objec- tive and splits.</td></tr><tr><td>Pythia DrDPO</td><td>Shared full-parameter SFT.</td><td>objective.</td><td>Full-parameter DrDPO with native Low-rank DrDPO with same ob- jective and splits.</td></tr><tr><td>Pythia SamPO</td><td>checkpoint.</td><td>Official released SamPO Not a same-SFT comparison.</td><td>Continued low-rank adaptation under native SamPO.</td></tr><tr><td></td><td>Qwen DPO/DrDPO Shared full-parameter SFT.</td><td>Full-parameter DPO/DrDPO.</td><td>Low-rank LOCUS; selection and confirmation precede frozen test evaluation.</td></tr></table>

Table 2: Full-parameter and low-rank branches used in the primary comparisons.
<table><tr><td>Dataset / protocol</td><td>Training pool</td><td>Selection / confirmation</td><td></td><td>Frozen evaluation</td></tr><tr><td>HH / Pythia DPO</td><td>160,800-pair pool.</td><td>One 256-example development 8,552 test pairs. split.</td><td></td><td></td></tr><tr><td>HH / Pythia DrDPO</td><td>160,800-pair pool.</td><td>Disjoint 256-example selection and 8,552 test pairs. confirmation splits.</td><td></td><td></td></tr><tr><td>HH / SamPO</td><td>training baseline.</td><td>Released checkpoint; no re- Established 256-example evaluation 256 examples. split.</td><td></td><td></td></tr><tr><td>HH / Qwen validation</td><td>example development splits. example confirmation.</td><td>160,288 pairs after two 256- 256-example selection and 256- 8,552 test pairs.</td><td></td><td></td></tr><tr><td>Harmless dataset</td><td>42,130 harmless-base train- split.</td><td>HH-RLHF One 256-example development Development evaluation</td><td>only.</td><td></td></tr><tr><td>Orca DPO dataset</td><td>ing pairs; safety refusal. 12,112 Intel/orca_dpo_pairs training pairs; instruction following.</td><td>split.</td><td>context-filtered One 256-example development Development evaluation only.</td><td></td></tr></table>

Table 3: Datasets, development splits, and frozen evaluation sets.

<table><tr><td>Protocol and backbone</td><td>Tokens</td><td>Reduction</td><td>Preference accuracy</td><td>Accuracy change; trainable parameters</td></tr><tr><td>SamPO / Anthropic Helpful and Harmless (Pythia-2.8B)</td><td>132.77→79.88</td><td>+39.84%</td><td>53.52→53.52%</td><td>0.00 pp; 7.86M/0.28%</td></tr><tr><td>DPO / Anthropic Helpful and Harmless (Pythia-2.8B)</td><td>137.67→109.12</td><td>+20.73%</td><td>48.40→48.39%</td><td>-0.01 pp; 7.86M/0.28%</td></tr><tr><td>DrDPO / Anthropic Helpful and Harmless (Pythia-2.8B)</td><td>145.61→108.79</td><td>+25.29%</td><td>48.39→48.26%</td><td>-0.13 pp; 7.86M/0.28%</td></tr><tr><td>DPO / Anthropic Helpful and Harmless (Qwen2.5-3B)</td><td>108.26→92.16</td><td>+14.87%</td><td>48.69→48.64%</td><td>-0.05 pp; 7.37M/0.24%</td></tr><tr><td>DrDPO / Anthropic Helpful and Harmless (Qwen2.5-3B)</td><td>111.58→91.97</td><td>+17.58%</td><td>48.64→48.55%</td><td>-0.09 pp; 7.37M/0.24%</td></tr></table>

Table 4: Primary results across both backbone families.

<table><tr><td>Dataset</td><td>Split</td><td>Base tokens</td><td>LOCUS tokens</td><td>Reduction</td><td>Base accuracy</td><td>LOCUS accuracy</td><td>Accuracy change</td></tr><tr><td>Anthropic Helpful and Harmless</td><td>8,552 test</td><td>137.67</td><td>109.12</td><td>20.73%</td><td>48.40%</td><td>48.39%</td><td>-0.01 pp</td></tr><tr><td>Anthropic Harmless</td><td>256 dev.</td><td>137.53</td><td>102.75</td><td>25.29%</td><td>57.03%</td><td>58.20%</td><td>+1.17 pp</td></tr><tr><td>Orca DPO</td><td>256 dev.</td><td>175.95</td><td>35.25</td><td>79.97%</td><td>69.14%</td><td>82.81%</td><td>+13.67 pp</td></tr></table>

Table 5: Cross-task token and accuracy comparisons.