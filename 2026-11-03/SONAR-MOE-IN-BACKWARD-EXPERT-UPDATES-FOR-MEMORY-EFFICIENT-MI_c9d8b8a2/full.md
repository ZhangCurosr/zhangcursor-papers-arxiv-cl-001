# SONAR-MOE: IN-BACKWARD EXPERT UPDATES FOR MEMORY-EFFICIENT MIXTURE-OF-EXPERTS TRAINING

Avni Mittal <sup>1</sup> Avinash Anand <sup>2</sup> Ashutosh Kumar <sup>3</sup> Dikshant Kukreja <sup>4</sup> Kritarth Prasad <sup>2</sup> Chaitanya Lakhchaura <sup>5</sup> Rishitej Reddy Vyalla <sup>5</sup> Erik Cambria <sup>6</sup> Zhengkui Wang <sup>2</sup> Rajiv Ratn Shah <sup>3</sup>

## ABSTRACT

Most parameters of a Mixture-of-Experts (MoE) model belong to its experts, and standard training keeps every expert gradient in GPU memory until the optimizer step. This memory grows with the number and size of the experts, so it rises as MoE models scale. We introduce SONAR-MOE, which updates each expert tensor during the backward pass and frees its gradient right away. It brings optimizer-in-backward from dense models to batched MoE expert tensors, so the GPU never holds more than one layer of expert gradients. When we fine-tune OLMoE-1B-7B and Granite-3.1-3B-A800M in bf16 with two expert-kernel backends, SONAR-MOE lowers peak memory by 19% to 22%. Its updates are bitwise identical to standard training without expert gradient clipping, and held-out loss stays within 0.003 of all baselines. Each step also takes 17% to 23% less time than standard training with global-norm clipping. Updating whole batched tensors rather than individual experts is what keeps SONAR-MOE fast, as it avoids a 20% to 72% slowdown while using the same memory. SONAR-MOE matches optimizer-in-backward over all parameters in memory and speed while still clipping the dense parameters.

## 1 INTRODUCTION

Mixture-of-Experts (MoE) models scale their parameter count without a matching increase in per-token compute, since each token is routed to only a few of many experts (Shazeer et al., 2017; Lepikhin et al., 2021; Fedus et al., 2022; Jiang et al., 2024a; DeepSeek-AI, 2024). The cost is paid in memory. A model that activates one billion parameters per token still has to keep the weights, gradients and optimizer state of every parameter during training. In OLMoE-1B-7B (Muennighoff et al., 2024), for example, the experts hold 6.44B of the 6.92B parameters, and their bf16 gradients alone take 12.9 GB. On a single GPU, this gradient memory alone can decide whether fine-tuning fits at all.

Much of this gradient memory is held for no reason. A standard training step runs the whole backward pass and only then calls the optimizer, so the gradient of the last MoE layer waits in memory until the backward pass has reached the first layer. An element-wise optimizer such as AdamW (Loshchilov & Hutter, 2019) does not need this wait, since the update of a tensor depends only on that tensor’s gradient and optimizer state. Prior work on dense models fuses the update into the backward pass (Lv et al., 2024b;a; Lewandowski & Kosson, 2023), and PyTorch provides a hook that fires as soon as a parameter’s gradient is ready (Paszke et al., 2019). Updating every parameter this way frees all gradient memory, but it gives up global-norm clipping, because no gradient survives to the end of the step. In an MoE the experts hold almost all of the gradient memory, so it is enough to update only the experts early.

We introduce SONAR-MOE, which updates the experts inside the backward pass. It attaches a hook to each batched expert tensor of every MoE layer, and when the tensor’s gradient is ready, the hook applies a fused AdamW update and frees the gradient, so the GPU holds at most one layer of expert gradients at any time (Figure 2). All other parameters keep the standard path with global-norm clipping. The placement of the hooks also matters. Expert weights can be stored per expert or as batched tensors (Gale et al., 2023; Tan et al., 2024; Guo et al., 2025), and SONAR-MOE hooks the batched tensors. With E experts per layer and L MoE layers, per-expert hooks fire 2EL times per step, while one hook per batched weight tensor fires only 2L times. On OL-MoE (E=64, L=16) this is 32 hook calls per step instead of 2,048, for the same update and the same memory.

In bf16 fine-tuning, SONAR-MOE lowers peak memory from 55.6 to 43.6 GB on OLMoE and by 21.4% on Granite-

![](images/cd11b713528fc6dc9770eb808860889999fee274850248cb0570472e1339b6b1.jpg)

![](images/e26601bc3d2c4a1774ee4863e247bbd88c3f70b7f146e11548acb737961a62f4.jpg)

![](images/3af7084d517f42661eec995148a154e9d1aff5e9a1ba8c0e3effaf7dd0c6f8ad.jpg)  
Figure 1. Alpaca fine-tuning of OLMoE-1B-7B and Granite-3.1-3B-A800M for 2,000 steps with the SonicMoE and ScatterMoE back ends, standard training with global-norm clipping (baseline) vs. SONAR-MOE. (a) Training loss on OLMoE; (b) peak GPU memory; (c) throughput averaged over steps 201–2,000.

3.1-3B-A800M (Figure 1). Its updates are bitwise identical to those of standard training without expert-gradient clipping, and held-out loss stays within 0.003 of every baseline. Each step is also 19–23% shorter than in standard training with global-norm clipping (Table 1). Two effects shorten the step. The expert gradients skip the global-norm clipping kernels, which gives most of the saving, and on Granite the in-backward updates fill GPU time that would otherwise sit idle, which cuts a further 11.6% at the same update rule. Applying optimizer-in-backward to all parameters reaches the same memory and speed but cannot clip any gradient. SONAR-MOE matches it on both while keeping global-norm clipping for the dense parameters. Our contributions are:

• SONAR-MOE, which updates each batched expert tensor inside the backward pass and frees its gradient at once. Retained expert-gradient memory is bounded by one layer, peak memory falls by 21.6% (OLMoE) and 21.4% (Granite), and the updates are bitwise identical to standard training without expert-gradient clipping (§4, §5.2).

• A speedup over standard training, with steps 19–23% shorter than under global-norm clipping, and a kernellevel profile that explains it. The expert gradients skip the clipping kernels, and on Granite the AdamW kernels run in otherwise idle GPU time (§5.3). Hooking batched tensors rather than single experts keeps this speed, since per-expert hooks make steps 20–72% slower (§4.3).

• A direct comparison with all-parameter optimizer-inbackward, which SONAR-MOE matches in memory and speed while keeping global-norm clipping, and a measured operating range across batch sizes, gradient checkpointing, 8-bit AdamW and gradient accumulation (§5.5, §5.6).

## 2 RELATED WORK

MoE training systems. Systems work on MoE training has made the expert computation faster through better kernels (Gale et al., 2023; Tan et al., 2024; Guo et al., 2025), better parallelism and communication across devices (Rajbhandari et al., 2022; Hwang et al., 2023; He et al., 2022; Jiang et al., 2024b; Zhang et al., 2025), and a cheaper expert backward pass (Liu et al., 2023; Zhang et al., 2026). Dataparallel training overlaps the gradient all-reduce with the backward pass (Li et al., 2020). None of this changes when gradients are released.

Training memory. ZeRO partitions optimizer state, gradients and parameters across ranks (Rajbhandari et al., 2020), and ZeRO-Offload moves them to CPU memory (Ren et al., 2021). Activation checkpointing and reversible blocks reduce activation memory (Chen et al., 2016; Liu et al., 2025), 8-bit optimizers shrink optimizer state (Dettmers et al., 2022), and Collage trains fully in bf16 without fp32 master weights (Yu et al., 2024). Other methods train fewer parameters or a smaller subspace: LoRA and QLoRA train lowrank adapters (Hu et al., 2021; Dettmers et al., 2023), ESFT fine-tunes only the experts most relevant to a task (Wang et al., 2024), GaLore projects gradients to a low-rank subspace (Zhao et al., 2024), and BAdam updates one block at a time (Luo et al., 2024). SONAR-MOE updates every expert in full and changes only when its gradient is freed, so these savings add to ours (§5.5).

Optimizer updates in the backward pass. The closest work fuses the optimizer update into the backward pass of dense models. LOMO does this with SGD (Lv et al., 2024b), AdaLomo adds adaptive learning rates (Lv et al., 2024a), Lewandowski & Kosson (2023) run the optimizer step during back-propagation, and GaLore uses per-layer updates to avoid storing weight gradients (Zhao et al., 2024). FORGE fuses gradient computation with optimizer updates in registers (Kukreja et al., 2026), whereas SONAR-MOE uses hooks to update and free materialized gradients of batched expert tensors. PyTorch’s post-accumulate hooks make this available for every parameter (Paszke et al., 2019), which we call all-parameter optimizer-in-backward. We do not claim the in-backward update itself, only the choice of where to apply it. Hooking every parameter frees all gradient memory but rules out global-norm clipping (Pascanu et al., 2013), while hooking only the batched expert tensors frees almost as much and keeps clipping for the rest of the model.

![](images/db77b4c285ac22b763d69c3db429579cef2185b0f8ac03fa2c8e90b674bf6fd6.jpg)  
Figure 2. The SONAR-MOE backward pass. Left: a per-expert layout launches 2E weight-gradient kernels per layer, while the batched layout computes all experts with one grouped GEMM per weight tensor. Middle: in standard training, expert gradients pile up layer by layer until the optimizer step, reaching $M _ { \mathrm { f l o o r } }$ . Right: a post-accumulate hook updates each batched tensor and frees its gradient as soon as it is ready, so retained gradient memory stays at the scale of one layer.

## 3 BACKGROUND

MoE layers. Let d be the hidden size, n the expert intermediate size, E the number of experts, K the number of experts used per token and L the number of MoE layers. A router scores each token $x _ { t }$ , keeps the top-K experts $S _ { t }$ and mixes their outputs with weights $w _ { e } ( x _ { t } )$ from the router softmax:

$$
y _ { t } = \sum _ { e \in S _ { t } } w _ { e } ( x _ { t } ) \mathrm { E x p e r t } _ { e } ( x _ { t } ) .\tag{1}
$$

Each expert is a gated MLP with a fused gate/up projection of size 2n × d and a down projection of size d × n. Existing MoE implementations store the experts of a layer in two batched tensors, $W _ { \mathsf { u p } } ^ { ( l ) } \in \mathbb { R } ^ { E \times 2 n \times d }$ and $W _ { \mathrm { d o w n } } ^ { ( l ) } \in \mathbb { R } ^ { E \times d \times n }$ and compute each weight gradient with one grouped GEMM over all experts instead of E separate GEMMs (Gale et al., 2023; Tan et al., 2024; Guo et al., 2025). SONAR-MOE keeps this layout and attaches its hooks to these two tensors.

Memory in a training step. A training step keeps four kinds of data on the GPU. Parameters and optimizer state stay resident across steps. Activations are freed during backward as soon as they have been used. Gradients stay in memory until the optimizer step, so by the end of backward every gradient of the model is held at once. In our setting, weights, gradients and AdamW states are all kept in bf16, so the gradients take as much memory as the weights. This gradient memory is what SONAR-MOE releases early.

## 4 METHOD

## 4.1 Gradient Memory in Standard Training

Write $| W _ { e } ^ { ( l ) } |$ for the number of weights of expert e in layer l. A standard step keeps every expert gradient until the optimizer runs, so at the end of backward the GPU holds

$$
M _ { \mathrm { H o o r } } = \sum _ { l = 1 } ^ { L } \sum _ { e = 1 } ^ { E } | W _ { e } ^ { ( l ) } |\tag{2}
$$

gradient elements for the experts alone. For OLMoE $\scriptstyle ( L = 1 6 .$ $E { = } 6 4 , d { = } 2 0 4 8 , n { = } 1 0 2 4 )$ this is 6.44B elements, 12.9 GB in bf16. None of these gradients is needed after its own update, and for an element-wise optimizer that update could have happened the moment the gradient was complete.

## 4.2 In-Backward Expert Updates

SONAR-MOE removes the floor with PyTorch’s register post accumulate grad hook, which calls a function as soon as a parameter’s gradient has been fully accumulated. We register one hook on each batched expert tensor, $W _ { \boldsymbol { \mathrm { u p } } } ^ { ( l ) }$ and $W _ { \mathrm { d o w n } } ^ { ( l ) } ,$ , and remove these tensors from the end-of-step optimizer. When the hook fires, it applies a fused AdamW update to the tensor using its gradient and optimizer state, and then sets the gradient to None so its memory can be reused. All other parameters, such as attention, embeddings and routers, keep the usual end-of-step update. Algorithm 1 summarizes the training

Algorithm 1 SONAR-MOE training step.   
1: Setup: for each MoE layer l, register a hook on $W _ { \boldsymbol { \mathrm { u p } } } ^ { ( l ) }$ and   
$W _ { \mathrm { d o w n } } ^ { ( l ) }$ (2L hooks); exclude them from the end-of-step opti  
mizer.   
2: Hook(W): fused AdamW update of W with W.grad; set   
W.grad ← NONE.   
3: Set the learning rate η<sub>t</sub> of the hooked and end-of-step optimiz  
ers.   
4: Forward pass as usual.   
5: for each MoE layer $l = L , \ldots , 1$ in backward do   
6: $\nabla W _ { \mathrm { d o w n } } ^ { ( l ) } \gets$ one grouped GEMM over all experts   
7: Hoo $\mathsf { \Omega } _ { \mathsf { a } } ( W _ { \mathsf { d o w n } } ^ { ( l ) } )$ ▷ update, then free   
8: $\nabla W _ { \boldsymbol { \mathsf { u p } } } ^ { ( l ) } \gets$ one grouped GEMM over all experts   
9: Hook $( W _ { \boldsymbol { \mathsf { u p } } } ^ { ( l ) } )$ ▷ update, then free   
10: pass the input gradient to layer $l - 1$   
11: end for   
12: Optimizer step on non-expert parameters only.

step.

Because the backward pass visits the MoE layers one at a time, each expert gradient is freed before the next layer’s gradients are computed. The retained expert-gradient memory is therefore bounded by one layer:

$$
M _ { \mathrm { f l o o r } } ^ { \mathrm { h o o k } } \leq \operatorname* { m a x } _ { l } \sum _ { e = 1 } ^ { E } | W _ { e } ^ { ( l ) } | \approx \frac { M _ { \mathrm { f l o o r } } } { L } .\tag{3}
$$

For OLMoE this is a 16-fold reduction, from 12.9 GB to 0.8 GB (one layer). Without gradient accumulation, this is the same update that standard training applies when its clipping excludes the expert tensors, and we verify that the two are bitwise identical (Table 4). Because the hooks read the learning rate during the backward pass, the learning rate for step t must be set before backward rather than after it.

## 4.3 Hook Granularity

The same schedule can be built on per-expert parameters, with one hook and one optimizer per expert slice, so each step makes 2EL optimizer calls instead of 2L. Since AdamW is element-wise, one fused update on an $( E , 2 n , d )$ tensor gives exactly the same result as E fused updates on its slices. What changes is the number of hook calls per step:

$$
N _ { \mathrm { h o o k s } } ^ { \mathrm { p e r - e x p e r t } } = 2 E L , \qquad N _ { \mathrm { h o o k s } } ^ { \mathrm { b a t c h e d } } = 2 L .\tag{4}
$$

Every call has a fixed cost, because autograd has to schedule a Python callback and the callback launches its own small kernels. If $\tau _ { \mathrm { h o o k } }$ is this fixed cost, the per-expert layout pays $2 E L \tau _ { \mathrm { h o o k } }$ per step and the batched layout pays 2L $\tau _ { \mathrm { h o o k } }$ . For OLMoE this means 2,048 calls instead of 32, each doing E times less work. Modern MoE designs favour many small experts (Dai et al., 2024; DeepSeek-AI, 2024), which makes the gap grow with E. On OLMoE (E=64), perexpert hooks add 23.6 ms per step (+20%), about 12 µs per extra optimizer call. On Granite (E=40, 32 layers) they add 61.1 ms (+72%), about $2 4 \mu \mathrm { s }$ per extra call. Both layouts use the same memory and the per-expert layout is slower (Table 3), so SONAR-MOE attaches the hooks to the batched tensors.

## 4.4 Clipping and Gradient Accumulation

Updating during backward changes two things that training loops usually take for granted, and our evaluation keeps both visible.

Global-norm clipping. Clipping by the global gradient norm needs every gradient before any update, but SONAR-MOE has already updated and freed the expert gradients by then. We therefore compare with two standard baselines, one that clips all gradients and one that clips only the nonexpert gradients. The second performs exactly our update (§5.1).

Gradient accumulation. Gradient accumulation sums the gradients of A microbatches and applies one optimizer step to their mean $\begin{array} { r } { \bar { g } = \frac { 1 } { A } \sum _ { a } g _ { a } } \end{array}$ . A post-accumulate hook fires as soon as each microbatch’s gradient is ready, so the hook schedule applies one AdamW update per microbatch instead. These are not equivalent to a single update using the combined gradients, because both the weights and AdamW’s optimizer state change between microbatches. The first and second moments can both differ between these schedules. The standard schedule updates $v _ { t }$ once per step with $\bar { g } ^ { 2 }$ , while the hook schedule updates it A times with the individual $g _ { a } ^ { 2 } \mathrm { . }$

$$
\begin{array} { r l } { \displaystyle } & { \displaystyle \boldsymbol { v } _ { t } ^ { \mathrm { s t d } } = \beta _ { 2 } \boldsymbol { v } _ { t - 1 } + \left( 1 - \beta _ { 2 } \right) \bar { \boldsymbol { g } } ^ { 2 } , } \\ & { \displaystyle \boldsymbol { v } _ { t } ^ { \mathrm { h o o k } } = \beta _ { 2 } ^ { A } \boldsymbol { v } _ { t - 1 } + \left( 1 - \beta _ { 2 } \right) \sum _ { a = 1 } ^ { A } \beta _ { 2 } ^ { A - a } \boldsymbol { g } _ { a } ^ { 2 } . } \end{array}\tag{5}
$$

For independent microbatch gradient noise of equal variance, averaging A microbatches divides the noise variance in $\bar { g }$ by A, whereas the hook schedule uses the individual noisy gradients in its second-moment updates. The two schedules coincide at A=1. We therefore use A=1 in all main experiments and evaluate A>1 separately in §5.6. Table 8 gives the measured cost of these per-microbatch updates.

## 5 EXPERIMENTS

## 5.1 Experimental Setup

Models and data. We pick two public MoE models of different sizes and expert counts that both run on batched expert kernels. We fine-tune allenai/ OLMoE-1B-7B-0924 (Muennighoff et al., 2024) (16 MoE layers, 64 experts, top-8) and ibm-granite/ granite-3.1-3b-a800m-base (IBM Granite Team, 2024) (32 MoE layers, 40 experts) on 8,000 Alpaca examples (Taori et al., 2023) for 2,000 steps. A held-out set of 500 further examples (indices 8000–8499 of the same seed-42 shuffle) is disjoint from training. We use a batch size of 2, sequences of at most 256 tokens and no gradient accumulation. The learning rate peaks at $2 \times 1 0 ^ { - 5 }$ after 100 warm-up steps and decays with a cosine schedule toward $2 \times 1 0 ^ { - 6 }$ AdamW uses $\beta { = } ( 0 . 9 , 0 . 9 5 ) , { \epsilon } { = } 1 0 ^ { - 8 }$ and no weight decay. Training is in bf16, with weights, gradients and AdamW states all kept in bf16 and no fp32 master weights. All arms use seed 42 and the same data order, which we check by hashing the first batches.

Baselines. Standard training with global clipping runs the backward pass, clips the global gradient norm of all parameters to 1.0 and takes one fused AdamW step. This is the usual default. Standard training without expert-gradient clipping is the same but clips only the non-expert parameters. It performs exactly the SONAR-MOE update, so any speed difference between the two comes from when the update happens rather than from the update itself. All-parameter optimizer-in-backward updates every parameter in its own backward hook, so no clipping is possible. We also run SONAR-MOE with per-expert hooks, one optimizer per expert slice, as an ablation.

Backends. We run every method with two expert-kernel libraries that store experts in batched tensors, SonicMoE (Guo et al., 2025) and ScatterMoE (Tan et al., 2024). Table 1 reports both; the other experiments use SonicMoE.

Metrics and hardware. We report the median step time over steps 201–2,000, with a GPU synchronization at the end of each step, throughput in supervised response tokens per second (about 111 per step), and peak allocated and peak reserved GPU memory. Same-seed repeats vary by at most 1.4% in step time and 0.0016 in held-out loss. Runs use two NVIDIA RTX PRO 6000 Blackwell GPUs (96 GB) with PyTorch 2.11 and CUDA 13.1. All arms of one model and backend ran on the same GPU, and the two GPUs have different power limits, so we compare step times only within a model and backend. Downstream accuracy is measured with lm-evaluation-harness (Gao et al., 2021) on the full test sets of ARC-Challenge (Clark et al., 2018) and HellaSwag (Zellers et al., 2019) (0-shot, acc norm) and MMLU (Hendrycks et al., 2020) (5-shot, accuracy).

## 5.2 Main Results

SONAR-MOE saves the memory that the mechanism predicts (Table 1). With SonicMoE, peak memory falls by 12.0 GB on OLMoE and 5.8 GB on Granite, which is the expert-gradient buffer of 12.9 and 6.0 GB minus the one layer that is still held. Held-out loss stays within 0.003 of every baseline. Speed needs a more careful reading. Standard training without expert-gradient clipping is as fast as SONAR-MOE on OLMoE (115.9 vs. 116.1 ms), so on OLMoE the whole speedup over standard training comes from not clipping the expert gradients. On Granite SONAR-MOE is a further 11.6% faster than that baseline, which §5.3 traces to idle GPU time. All-parameter optimizerin-backward matches SONAR-MOE in both memory and speed and differs only in giving up global-norm clipping on the dense parameters. ScatterMoE shows the same pattern. SONAR-MOE lowers peak memory by 21.1% on OLMoE and 19.1% on Granite, is as fast as standard training without expert-gradient clipping on OLMoE (133.2 vs. 133.3 ms), and is 5.9% faster than that baseline on Granite (88.2 vs. 93.7 ms), a smaller gain than with SonicMoE.

![](images/50015be6bd7c01a9db350feb1c5008ebc3a2d2ec9e0ef2883f5e308110e1cd02.jpg)  
Figure 3. Where step time goes (ms per step): torch.profiler over steps 300–309. Profiling adds CPU overhead, so absolute times are higher than in Table 1; compare bars only. “Clip” covers the norm and scaling kernels of global-norm clipping, which take 1.8–1.9 ms on OLMoE and 1.1–1.2 ms on Granite when the expert gradients are not clipped. “Fwd+bwd” covers the forward and backward kernels.

Loss alone does not show whether a fine-tuned model is still useful, so we also evaluate the OLMoE checkpoints after 2,000 steps on downstream tasks (Table 2). Fine-tuning raises ARC-Challenge accuracy by 3–5 points and leaves HellaSwag and MMLU almost unchanged. The two training methods differ by 0.016 on ARC-Challenge, about one standard error, and by at most 0.001 on HellaSwag and MMLU. We evaluate only OLMoE and omit GSM8K, where this model stays near 0.1 accuracy after Alpaca fine-tuning.

## 5.3 Sources of the Speedup

Three effects explain the step times (Figure 3). First, globalnorm clipping of the expert gradients costs about 25 ms of kernel time per step on OLMoE (26.9 vs. 1.8 ms). It reads the 12.9 GB of gradients to compute the norm, then reads and writes them again to rescale them. This accounts for nearly all of the difference between standard training and SONAR-MOE on OLMoE. The cost scales with the number of expert parameters rather than with the batch, which is why it matters most for small batches (§5.5). Second, on Granite the GPU work is the same with and without inbackward updates (57.7 vs. 57.4 ms busy), but idle time falls from 65 to 42 ms. Granite’s backward pass is limited by kernel-launch overhead, and the in-backward AdamW kernels run in these gaps. OLMoE leaves little idle time, so it gains almost nothing from this. Third, attaching hooks per expert keeps the memory saving but adds 23.6 ms per step on OLMoE, about 12 µs per extra optimizer call (Table 3).

Table 1. Alpaca fine-tuning for 2,000 steps with batch size 2 and seed 42 on two expert-kernel backends; every row uses the same data order. ∆ is relative to standard training with global clipping on the same model and backend. Step time is the median over steps 201–2,000. Same-seed repeats vary by at most 1.4% in step time and 0.0016 in held-out loss. All rows of one model and backend ran on one GPU; the OLMoE SonicMoE rows ran on a different GPU from all other rows, so compare step times only within a model and backend. Bold marks the lowest memory and step time per model and backend.
<table><tr><td>Model</td><td>Backend</td><td>Method</td><td>Peak alloc. GB (∆)</td><td>Peak res. GB</td><td>Step ms (∆)</td><td>Tok/s</td><td>Held-out loss</td></tr><tr><td rowspan="5">OLMoE-1B-7B</td><td rowspan="4">SonicMoE</td><td>Standard (global clip)</td><td>55.55</td><td>56.42</td><td>142.8</td><td>768</td><td>1.2577</td></tr><tr><td>Standard (no expert clip)</td><td>55.55 (0.0%)</td><td>56.42</td><td> $1 1 5 . 9 \left( - 1 8 . 8 \% \right)$ </td><td>958</td><td>1.2591</td></tr><tr><td>Optimizer-in-backward (all)</td><td>43.35 (-22.0%)</td><td>44.24</td><td> $1 1 4 . 4 \left( - 1 9 . 9 \% \right)$ </td><td>970</td><td>1.2612</td></tr><tr><td>SONAR-MOE</td><td>43.57 (-21.6%)</td><td>44.24</td><td>116.1 (−18.7%)</td><td>948</td><td>1.2596</td></tr><tr><td rowspan="4">ScatterMoE</td><td>Standard (global clip)</td><td>55.55</td><td>57.17</td><td>160.2</td><td>688</td><td>1.2585</td></tr><tr><td>Standard (no expert clip)</td><td>55.55 (0.0%)</td><td>57.17</td><td>133.3 (−16.7%)</td><td>825</td><td>1.2601</td></tr><tr><td>Optimizer-in-backward (all)</td><td>43.60 (-21.5%)</td><td>44.79</td><td>131.0 (−18.2%)</td><td>841</td><td>1.2610</td></tr><tr><td>SONAR-MOE</td><td>43.81 (-21.1%)</td><td>44.80</td><td>133.2 (−16.8%)</td><td>826</td><td>1.2601</td></tr><tr><td rowspan="5">Granite-3.1-3B</td><td rowspan="3">SonicMoE</td><td>Standard (global clip)</td><td>26.91</td><td>28.03</td><td>109.1</td><td>1111</td><td>1.1625</td></tr><tr><td>Standard (no expert clip)</td><td>26.91 (0.0%)</td><td>28.02</td><td>95.5 (−12.4%)</td><td>1275</td><td>1.1643</td></tr><tr><td>Optimizer-in-backward (all)</td><td>21.15 (−21.4%)</td><td>21.90</td><td>86.5 (−20.7%)</td><td>1421</td><td>1.1630</td></tr><tr><td>SONAR-MOE</td><td>21.14 (−21.4%)</td><td>22.03</td><td>84.4 (−22.7%)</td><td>1450</td><td>1.1616</td></tr><tr><td rowspan="3">ScatterMoE</td><td>Standard (global clip)</td><td>26.89</td><td>28.49</td><td>106.8</td><td>1151</td><td>1.1591</td></tr><tr><td>Standard (no expert clip) Optimizer-in-backward (all)</td><td>26.89 (0.0%)</td><td>28.49</td><td>93.7 (−12.3%)</td><td>1230</td><td>1.1582</td></tr><tr><td>SONAR-MOE</td><td>21.77 (−19.1%) 21.75 (–19.1%)</td><td>22.40 22.34</td><td>90.0 (−15.8%) 88.2 (−17.4%)</td><td>1362 1390</td><td>1.1585 1.1582</td></tr></table>

Table 2. Downstream accuracy on OLMoE after 2,000 steps (full test sets). Standard errors are 0.015 for ARC-Challenge and 0.004 for HellaSwag and MMLU in every row.
<table><tr><td>Checkpoint</td><td>ARC-C</td><td>HellaSwag</td><td>MMLU</td></tr><tr><td>Pretrained</td><td>0.482</td><td>0.771</td><td>0.527</td></tr><tr><td>Standard (global clip)</td><td>0.532</td><td>0.771</td><td>0.521</td></tr><tr><td>SONAR-MOE</td><td>0.516</td><td>0.772</td><td>0.521</td></tr></table>

Table 3. Hook granularity. Both layouts hold the same memory; per-expert hooks make 2EL optimizer calls per step instead of 2L.
<table><tr><td>Model</td><td>Hooks</td><td>Calls</td><td>Peak GB</td><td>Step ms</td><td>Held-out</td></tr><tr><td>OLMoE</td><td>batched</td><td>32</td><td>43.57</td><td>116.1</td><td>1.2596</td></tr><tr><td>OLMoE</td><td>per-expert</td><td>2048</td><td>43.58</td><td>139.7</td><td>1.2600</td></tr><tr><td>Granite</td><td>batched</td><td>64</td><td>21.14</td><td>84.4</td><td>1.1616</td></tr><tr><td>Granite</td><td>per-expert</td><td>2560</td><td>21.37</td><td>145.5</td><td>1.1606</td></tr></table>

## 5.4 Exactness and Training Quality

Update exactness. Without gradient accumulation, SONAR-MOE performs the same arithmetic as standard training that clips only the non-expert gradients. We train OLMoE for three steps with a learning rate of $1 0 ^ { - 4 }$ large enough that bf16 updates are visible, and compare the weights of two runs element by element (Table 4). SONAR-MOE and standard training without expert-gradient clipping give exactly the same 6.44B expert weights. They differ in 6 of the 477M other weights, and two runs of the same baseline also differ in 6 elements, so this is ordinary run-to-run noise, not an effect of the schedule. Per-expert and batched hooks also give the same expert weights, and all-parameter optimizer-in-backward gives exactly the same weights as standard training without clipping.

Table 4. Weights that differ between two runs after three steps on OLMoE (6.44B expert weights, 477M other weights).
<table><tr><td>Runs compared</td><td>Expert weights Other weights</td></tr><tr><td>Same baseline, run twice</td><td>0</td></tr><tr><td>SONAR-MoE vs. standard (no expert clip)</td><td>0</td></tr><tr><td>SoNAR-MoE: per-expert vs. batched hooks</td><td>0</td></tr><tr><td>Opt.-in-backward (all) vs. standard (no clip)</td><td>0</td></tr><tr><td>Standard: global clip vs. no expert clip</td><td>0 3.04B (47%) 91.5M (19%)</td></tr></table>

Clipping the expert gradients, as standard training does, changes 47% of the expert weights after three steps, yet changes held-out loss on OLMoE by only 0.0014 after 2,000 steps (Table 1). AdamW divides each update by a running estimate of the gradient scale, which largely cancels a uniform rescaling.

(a) Peak memory  
![](images/eab2b782f527510f4a069046f317d9d0cac8c65d9ef780a5ddf4ff59c3e7ded0.jpg)  
(b) Step time

![](images/9deea5e2a8a4da09cc3db654b60b2de6bdf91087ccae57083e821106e68b059d.jpg)  
Figure 4. Batch size on OLMoE. (a) Peak allocated memory (filled bars) and peak reserved memory (dashed outlines) for standard training and SONAR-MOE. The bold labels give the saving in allocated memory. (b) Step time in ms for each method. Sequences are padded to 256 tokens (512–8,192 tokens per step) so that kernel autotuning happens once, in warm-up.

Pure-bf16 updates. Because weights are stored in bf16, an update smaller than half a unit in the last place of a weight is rounded away. Table 5 shows that after 2,000 steps most weights still hold their pretrained values, and that the two methods change the same share of weights.

Table 5. Share of weights that differ from their pretrained values after 2,000 steps of bf16 training on OLMoE.
<table><tr><td>Method</td><td>Expert weights</td><td>Other weights</td></tr><tr><td>Standard (global clip)</td><td>19.3%</td><td>22.4%</td></tr><tr><td>SONAR-MOE</td><td>19.3%</td><td>22.5%</td></tr></table>

## 5.5 Batch Size and Memory Composition

The cost of clipping expert gradients is fixed per step (22– 29 ms), so the speedup over standard training shrinks from 18% at batch size 2 to 8% at batch size 32 (Figure 4). With the same update rule, SONAR-MOE and standard training are within ±1.1% at batch sizes 2–16, and at batch size

Table 6. Memory composition on OLMoE at batch size 2 (300-step runs). Peak allocated and reserved memory in GB, step time in ms, and whether 300 steps completed with the PyTorch allocator capped at 46 GB (standing for a 48 GB GPU) or 30.5 GB (a 32 GB GPU). GC = gradient checkpointing; opt.-in-bwd = all-parameter optimizer-in-backward.
<table><tr><td colspan="4"></td><td colspan="2">Runs under cap</td></tr><tr><td>Setup</td><td>Alloc.</td><td>Res.</td><td>ms</td><td>46 GB</td><td>30.5 GB</td></tr><tr><td>Standard</td><td>55.55</td><td>56.40</td><td>140.8</td><td>OOM</td><td></td></tr><tr><td>Standard + GC</td><td>55.48</td><td>56.16</td><td>155.6</td><td></td><td></td></tr><tr><td>SONAR-MOE</td><td>43.56</td><td>44.22</td><td>111.8</td><td>√</td><td></td></tr><tr><td>SONAR-MOE + GC</td><td>43.23</td><td>44.03</td><td>130.1</td><td></td><td></td></tr><tr><td>Opt.-in-bwd</td><td>43.35</td><td>44.24</td><td>109.9</td><td>√</td><td></td></tr><tr><td>8-bit AdamW</td><td>41.93</td><td>43.13</td><td>186.0</td><td></td><td>OOM</td></tr><tr><td>8-bit + SONAR-MoE</td><td>29.94</td><td>30.52</td><td>165.4</td><td></td><td>√</td></tr><tr><td>8-bit + SONAR-MoE + GC</td><td>29.61</td><td>29.78</td><td>189.5</td><td></td><td>√</td></tr><tr><td>8-bit + opt.-in-bwd</td><td>29.74</td><td>30.69</td><td>165.0</td><td></td><td>OOM</td></tr></table>

32 SONAR-MOE is 4.0% faster. The memory saving also shrinks with batch size, because activations, not gradients, set the peak at large batches. The saving in peak allocated memory is 12.0 GB at batch size 2, 9.4 GB at 8, 5.2 GB at 16 and none at 32, where only 2.2 GB of reserved memory is saved. SONAR-MOE helps most when batches are small relative to the model. In practice the saving can be read off Eq. 2: it equals the expert-gradient buffer as long as gradients, not activations, set the peak.

Gradient checkpointing does not lower peak memory at batch size 2, because activations are small, and only costs time (+10–16%, Table 6). 8-bit AdamW and SONAR-MOE save different memory, optimizer states and gradients, so they combine. Together they need about 30 GB and run under a 30.5 GB cap, where 8-bit AdamW alone runs out of memory. All-parameter optimizer-in-backward gives the same savings. It also runs under the 46 GB cap, and with 8-bit AdamW its peak allocated memory matches ours (29.7 vs. 29.9 GB). 8-bit AdamW is slower per step (186 vs. 141 ms).

8-bit AdamW quality. For the combination that fits OL-MoE fine-tuning on a 32 GB GPU, we also check that quality is preserved. We use the setup of Table 1 (2,000 steps, batch size 2, seed 42) but store the AdamW states in 8 bits with bitsandbytes AdamW8bit (Dettmers et al., 2022). With SONAR-MOE, each expert tensor gets its own 8-bit optimizer inside its hook, and the other parameters use one 8-bit optimizer at the end of the step with global-norm clipping, as in Table 1. Held-out loss stays within 0.003 of standard training (Table 7), so the memory saving of 8-bit AdamW and SONAR-MOE comes at no visible cost in quality.

Table 7. Held-out loss on OLMoE after 2,000 steps with 8-bit AdamW, compared with standard training from Table 1.
<table><tr><td>Configuration</td><td>Held-out loss</td></tr><tr><td>Standard (global clip)</td><td>1.2577</td></tr><tr><td>8-bit AdamW</td><td>1.2574</td></tr><tr><td>8-bit AdamW + SONAR-MoE</td><td>1.2601</td></tr></table>

Table 8. Gradient accumulation with A microbatches per step (OL-MoE, 500 steps). Step time in ms; peak allocated memory in GB for standard training (both variants) / SONAR-MOE. SONAR-MOE makes A updates per step.
<table><tr><td>A Std. (global) Std. (no exp.) SoNAR-MoE Peak GB</td><td></td></tr><tr><td>143.4</td><td>117.3 116.455.55 / 43.57</td></tr><tr><td>217.8 191.9</td><td>229.4 57.21 / 44.31</td></tr><tr><td>2 4 368.5</td><td>341.7 451.7 57.21 / 44.31</td></tr><tr><td>8 665.8 639.6</td><td>895.657.21 / 44.32</td></tr></table>

## 5.6 Gradient Accumulation

As described in §4.4, hooks fire once per microbatch, so with A microbatches SONAR-MOE takes A optimizer steps where the baseline takes one. Table 8 sweeps A ∈ {1, 2, 4, 8}. At A=1 SONAR-MOE is as fast as standard training without expert-gradient clipping (116.4 vs. 117.3 ms). With accumulation it runs the optimizer A times per step and is slower than standard training with global clipping, by 5% at A=2, 23% at A=4 and 35% at A=8, while still saving 12.9 GB. Each extra update is a full AdamW pass over the 6.44B expert weights, about 37 ms at every A, so the overhead grows linearly with A. The method also becomes a different optimizer, so we recommend SONAR-MOE for A=1, or for settings where memory is the binding constraint and the change of update rule is acceptable.

## 6 DISCUSSION

Practical memory gains. The most direct benefit is that a run fits where it did not before. Fine-tuning OLMoE-1B-7B in our setup peaks at 55.6 GB with standard training and at 43.6 GB with SONAR-MOE, which fits on a 48 GB GPU, since it runs under a 46 GB allocator cap where standard training runs out of memory. With 8-bit AdamW it fits a 32 GB GPU (30.5 GB cap), where 8-bit AdamW alone does not. All-parameter optimizer-in-backward reaches the same peak memory and also fits under the 46 GB cap. Gradient checkpointing does not reduce peak memory at batch size 2 (§5.5), and the saving shrinks as batches grow, because activations then set the peak.

Operating conditions. Our results point to a simple rule. Early release is a good default when four conditions hold: the optimizer is element-wise, gradients are not accumulated across microbatches, the experts are stored in batched tensors, and the expert gradients are not needed for globalnorm clipping. The memory saving is largest at small batch sizes. A speed benefit appears only when the baseline clips expert gradients or when the backward pass leaves the GPU idle. With accumulation the method becomes a different and slower optimizer (§5.6), and with per-expert hooks it keeps the memory saving but loses time (§5.3). We think this kind of explicit operating range is more useful to practitioners than a single headline number, and we have tried to report every regime we measured, including the ones where the method does not help.

Why hook granularity is overlooked. Optimizer-inbackward methods were developed for dense models, where each weight matrix is one parameter and the question of hook granularity never arises. In MoE code it does arise. Expert weights are often stored as a list of per-expert modules for readability, or are split into per-expert views when a checkpoint is loaded. Both cases silently multiply the hook count by E. Our ablation shows that this keeps the memory saving intact but adds 20% (OLMoE) and 72% (Granite) step time (Table 3), so the problem is easy to overlook if only memory is measured.

## 7 CONCLUSION

In MoE training, the gradients of all experts usually sit in memory until the optimizer step, even though each one is needed only for its own update. SONAR-MOE updates every batched expert tensor as soon as its gradient is ready and frees the gradient right away, which bounds the retained expert-gradient memory by one layer. At batch size 2 it lowers peak memory by 21.6% on OLMoE and 21.4% on Granite, and its updates are bitwise identical to standard training without expert-gradient clipping. Its speed advantage over standard training comes from not clipping the expert gradients and, on models whose backward pass leaves the GPU idle, from running the optimizer kernels in that idle time. Hooks attached per expert save the same memory but make each step 20–72% slower, so early release should be attached to batched expert tensors. All-parameter optimizerin-backward matches the method in memory and speed. Natural next steps are to support global-norm clipping without keeping all gradients in memory, and to support data parallelism by updating after the gradient all-reduce. We are also extending SONAR-MOE to inference, where expert computation can be started speculatively before the router’s decision is final and reconciled against the exact routing, so that the same early-release principle speeds up both training and inference of MoE models.

## LIMITATIONS

Gradient clipping. Standard global-norm clipping needs all gradients before any update, while SONAR-MOE updates experts during backward. Compatibility can be restored by delaying the update and keeping, offloading or recomputing the gradients after the norm is known, or by clipping each expert tensor by its own norm inside the hook. We plan to evaluate these options in the next version. Gradient accumulation. With A≥2 the method applies A updates per step and becomes a different optimizer that is 5–35% slower than standard training with global clipping (§5.6), so our main results use A=1. The schedule is exact for element-wise optimizers such as AdamW, SGD or Lion, and extending it to optimizers that need statistics across tensors is left to future work. Scope of the evidence. All results use one GPU, and under data parallelism the update would run after the gradient all-reduce, which we will add by integrating with the communication hooks. The memory saving is largest at small batch sizes and shrinks as activations dominate. We train in bf16 and test models up to 7B parameters on Alpaca fine-tuning with one seed, using same-seed repeats to measure noise; larger models, more datasets and more seeds are planned for the next experiments.

## REFERENCES

Chen, T., Xu, B., Zhang, C., and Guestrin, C. Training deep nets with sublinear memory cost. arXiv preprint arXiv:1604.06174, 2016. URL https://arxiv. org/abs/1604.06174.

Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A., Schoenick, C., and Tafjord, O. Think you have solved question answering? Try ARC, the AI2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.

Dai, D., Deng, C., Zhao, C., Xu, R. X., Gao, H., Chen, D., Li, J., et al. DeepSeekMoE: Towards ultimate expert specialization in mixture-of-experts language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL), 2024. URL https://arxiv.org/abs/2401.06066.

DeepSeek-AI. DeepSeek-V3 technical report. arXiv preprint arXiv:2412.19437, 2024. URL https:// arxiv.org/abs/2412.19437.

Dettmers, T., Lewis, M., Shleifer, S., and Zettlemoyer, L. 8-bit optimizers via block-wise quantization. In International Conference on Learning Representations (ICLR), 2022. URL https://arxiv.org/abs/ 2110.02861.

Dettmers, T., Pagnoni, A., Holtzman, A., and Zettlemoyer, L. QLoRA: Efficient finetuning of quantized LLMs.

In Advances in Neural Information Processing Systems (NeurIPS), pp. 10088–10115, 2023.

Fedus, W., Zoph, B., and Shazeer, N. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. Journal ofMachine Learning Research, 23 (120):1–39, 2022. URL https://arxiv.org/abs/ 2101.03961.

Gale, T., Narayanan, D., Young, C., and Zaharia, M. MegaBlocks: Efficient sparse training with mixture-ofexperts. In Proceedings of Machine Learning and Systems (MLSys), 2023. URL https://arxiv.org/ abs/2211.15841.

Gao, L., Tow, J., Biderman, S., Black, S., DiPofi, A., Foster, C., Golding, L., Hsu, J., McDonell, K., Muennighoff, N., et al. A framework for few-shot language model evaluation. Zenodo, 2021.

Guo, W., Mishra, M., Cheng, X., Stoica, I., and Dao, T. SonicMoE: Accelerating MoE with IO and tile-aware optimizations. arXiv preprint arXiv:2512.14080, 2025.

He, J., Zhai, J., Antunes, T., Wang, H., Luo, F., Shi, S., and Li, Q. FasterMoE: Modeling and optimizing training of large-scale dynamic pre-trained models. In Proceedings ofthe 27th ACM SIGPLAN Symposium on Principles and Practice ofParallel Programming (PPoPP), pp. 120–134, 2022.

Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., and Steinhardt, J. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300, 2020.

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. LoRA: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.

Hwang, C., Cui, W., Xiong, Y., Yang, Z., Liu, Z., Hu, H., Wang, Z., Salas, R., Jose, J., Ram, P., et al. Tutel: Adaptive mixture-of-experts at scale. In Proceedings ofMachine Learning and Systems (MLSys), 2023. URL https://arxiv.org/abs/2206.03382.

IBM Granite Team. Granite 3.0 language models, October 2024. URL https:// github.com/ibm-granite/granite-3. 0-language-models/.

Jiang, A. Q., Sablayrolles, A., Roux, A., Mensch, A., Savary, B., Bamford, C., Chaplot, D. S., de las Casas, D., Bou Hanna, E., Bressand, F., et al. Mixtral of experts. arXiv preprint arXiv:2401.04088, 2024a. URL https://arxiv.org/abs/2401.04088.

Jiang, C., Tian, Y., Jia, Z., Zheng, S., Wu, C., and Wang, Y. Lancet: Accelerating mixture-of-experts training via whole graph computation-communication overlapping. Proceedings of Machine Learning and Systems, 6:74–86, 2024b.

Kukreja, D., Prasad, K., Anand, A., Wang, Z., Cambria, E., Liu, T., Ng, A. B., See, S., and Chatterjee, B. FORGE: Fused on-register gradient elimination for memory-efficient LLM training. arXiv preprint arXiv:2606.22932, 2026.

Lepikhin, D., Lee, H., Xu, Y., Chen, D., Firat, O., Huang, Y., Krikun, M., Shazeer, N., and Chen, Z. GShard: Scaling giant models with conditional computation and automatic sharding. In International Conference on Learning Representations, 2021. URL https://arxiv.org/abs/ 2006.16668.

Lewandowski, B. and Kosson, A. Memory efficient mixedprecision optimizers. arXiv preprint arXiv:2309.12381, 2023. URL https://arxiv.org/abs/2309. 12381.

Li, S., Zhao, Y., Varma, R., Salpekar, O., Noordhuis, P., Li, T., Paszke, A., Smith, J., Vaughan, B., Damania, P., et al. PyTorch distributed: Experiences on accelerating data parallel training. arXiv preprint arXiv:2006.15704, 2020.

Liu, L., Gao, J., and Chen, W. Sparse backpropagation for MoE training. arXiv preprint arXiv:2310.00811, 2023. URL https://arxiv.org/abs/2310.00811.

Liu, N., Yang, J., Cai, K., and Wang, K. RevFFN: Memory-efficient full-parameter fine-tuning of mixtureof-experts LLMs with reversible blocks. arXiv preprint arXiv:2512.20920, 2025. URL https://arxiv. org/abs/2512.20920.

Loshchilov, I. and Hutter, F. Decoupled weight decay regularization. In International Conference on Learning Representations (ICLR), 2019. URL https://arxiv. org/abs/1711.05101.

Luo, Q., Yu, H., and Li, X. BAdam: A memory efficient full parameter optimization method for large language models. In Advances in Neural Information Processing Systems (NeurIPS), 2024. URL https://arxiv.org/abs/ 2404.02827.

Lv, K., Yan, H., Guo, Q., Lv, H., and Qiu, X. AdaLomo: Low-memory optimization with adaptive learning rate. In Findings of the Association for Computational Linguistics: ACL 2024, pp. 12486–12502, 2024a. URL https://arxiv.org/abs/2310.10195.

Lv, K., Yang, Y., Liu, T., Guo, Q., and Qiu, X. Full parameter fine-tuning for large language models with limited resources. In Ku, L.-W., Martins, A., and Srikumar, V. (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 8187–8198, Bangkok, Thailand, August 2024b. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.445. URL https: //aclanthology.org/2024.acl-long.445/.

Muennighoff, N., Soldaini, L., Groeneveld, D., Lo, K., Morrison, J., Min, S., Shi, W., Walsh, P., Tafjord, O., Lambert, N., et al. OLMoE: Open mixture-of-experts language models. arXiv preprint arXiv:2409.02060, 2024. URL https://arxiv.org/abs/2409.02060.

Pascanu, R., Mikolov, T., and Bengio, Y. On the difficulty of training recurrent neural networks. In Proceedings of the 30th International Conference on Machine Learning (ICML), 2013. URL https://arxiv.org/ abs/1211.5063.

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Kopf, A., Yang, E., DeVito, Z., Rai-¨ son, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., and Chintala, S. PyTorch: An imperative style, high-performance deep learning library. In Advances in Neural Information Processing Systems (NeurIPS), 2019. URL https://arxiv.org/abs/1912.01703.

Rajbhandari, S., Rasley, J., Ruwase, O., and He, Y. ZeRO: Memory optimizations toward training trillion parameter models. In Proceedings of the International Conferencefor High Performance Computing, Networking, Storage and Analysis (SC), 2020. URL https://arxiv. org/abs/1910.02054.

Rajbhandari, S., Li, C., Yao, Z., Zhang, M., Aminabadi, R. Y., Awan, A. A., Rasley, J., and He, Y. DeepSpeed-MoE: Advancing mixture-of-experts inference and training to power next-generation AI scale. In Proceedings of the 39th International Conference on Machine Learning (ICML), 2022. URL https://arxiv.org/abs/ 2201.05596.

Ren, J., Rajbhandari, S., Aminabadi, R. Y., Ruwase, O., Yang, S., Zhang, M., Li, D., and He, Y. ZeRO-Offload: Democratizing billion-scale model training. In 2021 USENIX Annual Technical Conference (USENIX ATC 21), pp. 551–564, 2021.

Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., and Dean, J. Outrageously large neural networks: The sparsely-gated mixture-of-experts

layer. In International Conference on Learning Representations, 2017. URL https://arxiv.org/abs/ 1701.06538.

Tan, S., Shen, Y., Panda, R., and Courville, A. Scattered mixture-of-experts implementation. arXiv preprint arXiv:2403.08245, 2024.

Taori, R., Gulrajani, I., Zhang, T., Dubois, Y., Li, X., Guestrin, C., Liang, P., and Hashimoto, T. B. Stanford Alpaca: An instruction-following LLaMA model, 2023. URL https://github.com/tatsu-lab/ stanford\_alpaca.

Wang, Z., Chen, D., Dai, D., Xu, R., Li, Z., and Wu, Y. Let the expert stick to his last: Expert-specialized finetuning for sparse architectural large language models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 784–801, 2024.

Yu, T., Gupta, G., Gopalswamy, K., Mamidala, A., Zhou, H., Huynh, J., Park, Y., Diamant, R., Deoras, A., and Huan, L. Collage: Light-weight low-precision strategy for LLM training. arXiv preprint arXiv:2405.03637, 2024.

Zellers, R., Holtzman, A., Bisk, Y., Farhadi, A., and Choi, Y. HellaSwag: Can a machine really finish your sentence? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL), pp. 4791–4800, 2019.

Zhang, J., Liu, Y., Yan, S., Deng, L., Cao, J., Yang, S., Ni, M., Xue, B., and Li, S. MoEBlaze: Breaking the memory wall for efficient MoE training on modern GPUs. arXiv preprint arXiv:2601.05296, 2026. URL https: //arxiv.org/abs/2601.05296.

Zhang, S., Zheng, N., Lin, H., Jiang, Z., Bao, W., Jiang, C., Hou, Q., Cui, W., Zheng, S., Chang, L.-W., Chen, Q., and Liu, X. COMET: Fine-grained computationcommunication overlapping for mixture-of-experts. In Proceedings of Machine Learning and Systems (MLSys), 2025.

Zhao, J., Zhang, Z., Chen, B., Wang, Z., Anandkumar, A., and Tian, Y. GaLore: Memory-efficient LLM training by gradient low-rank projection. arXiv preprint arXiv:2403.03507, 2024.