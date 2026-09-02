# Behaviorally Effective LoRA Writes Are Sparse and Structured

Haruto Sato, Yuki Tanaka, Ren Nakamura, Aoi Kobayashi, Mei Ito Independent Researchers

## Abstract

Low-rank adaptation fixes the rank of the update, but it does not identify which parts of a trained write actually carry behavior. We study that question directly and show that behaviorally effective LoRA writes are sparse, structured, and far more concentrated than the raw low-rank parameterization suggests.

We use LEARNED-BASIS LORA, a learned-basis continuation recipe, to expose that structure. The recipe warms up an unconstrained adapter, converts its learned write columns into a modulewise orthonormal basis, freezes that basis, and continues training inside the constrained parameterization. Across 14 exact switches from unconstrained to constrained form, held-out accuracy is unchanged at the conversion step and reconstructed write matrices differ by at most 0.25% relative Frobenius error. Samestate continuation then shows that the same trained checkpoint develops differently under different write subspaces, establishing write geometry as a causal state variable. A no-retraining projection test shows that useful write signal stays inside the learned write space and largely disappears from random or frozen-activation PCA controls.

The concentration pattern is strong at both local and global scales. Across GSM8K, MathQA, and AQuA, permodule top-k continuation reaches its optimum at $k ~ \in ~ \{ 2 , 4 \}$ in all twelve seed-level cases we test. A stricter global ranking test shows that learned top-16 and top-32 subsets outperform matched random subsets, especially on GSM8K/Qwen and MathQA/Qwen. Single-direction ablations further reveal a sparse set of late q\_proj, o\_proj, and down\_proj components with outsized behavioral impact.

The contribution of the paper is a structural mechanism result. Trained LoRA adapters do not use their write budget uniformly. Most of the downstream effect is carried by a small, late, and structured subset of write components, and a learned-basis constrained continuation exposes that organization clearly enough to measure it.

## 1 Introduction

Parameter-efficient fine-tuning has become the standard way to adapt large language models under strict memory, storage, and optimizer budgets [Houlsby et al., 2019, Li and Liang, 2021, Lester et al., 2021, Liu et al., 2021, Zaken et al., 2022, Mahabadi et al., 2021, Hu et al., 2021, Dettmers et al., 2023, Ding et al., 2023, Zhang et al., 2023, Kopiczko et al., 2023]. Most PEFT work studies scalar capacity decisions: rank, parameter budget, quantization level, or layer allocation. A growing line instead studies update geometry through weight decomposition, singular-structure initialization, orthogonal constraints, and structured subspaces [Liu et al., 2024a, Meng et al., 2024, Wang et al., 2024, She et al., 2025, Liu et al., 2024b, Wang et al., 2023, Lion et al., 2025, Li et al., 2025, Yang et al., 2025, Xiao et al., 2026, Chang et al., 2026]. These papers show that geometry matters. They leave a sharper question open: after a LoRA adapter has been trained, which parts of its write update actually carry task behavior?

Recent systems in adjacent areas increasingly treat model capacity as a selectively allocated resource. EXACT redistributes long-context supervision toward underexposed long-range targets [Zhu et al., 2026a]. HeLa-Mem consolidates agent experience through associative memory structure [Zhu et al., 2026b]. GapSight learns when a vision-language model should revisit a local visual region [Zhu et al., 2026c]. FCPRAG performs retrieval-conditioned fusion over passage-specific LoRA adapters [Zhu et al., 2026d]. Learning Less Is More shows that premature upper-layer attention specialization can distort pretraining dynamics before lower-layer features stabilize [Zhu et al., 2026e]. The shared pattern is selective allocation: the decisive signal occupies only part of the available computation or parameter space. We ask whether LoRA writes follow the same pattern.

A standard LoRA adapter can spend its limited rank on any output direction that improves training loss, including prompt-format quirks, splitspecific shortcuts, or brittle heuristics. If transfer depends on only a small part of that update, rank alone is a poor description of what the adapter learned. We therefore separate two objects: the low-rank code the adapter computes and the output subspace into which it writes. The central claim of this paper is direct: behaviorally effective LoRA writes are sparse and structured.

We study this with a projected low-rank parameterization that freezes a module-wise write basis and continues training inside that subspace. Our main trainable instantiation, LEARNED-BASIS LORA, first warms up an unconstrained FULL adapter, orthonormalizes its learned write columns into a basis, and then continues training with the basis fixed. This construction lets us run four clean tests. Exact conversion verifies that the switch into constrained form is lossless. Samestate basis swap tests whether write geometry is a causal state variable. No-retraining projection tests where the useful signal lies inside a trained FULL update. Local and global concentration analyses measure how much of the learned write space is actually needed.

The evidence supports the same conclusion across several intervention levels. Across 14 exact switches, held-out accuracy is unchanged at the conversion step and reconstructed write matrices differ by at most 0.25% relative Frobenius error. Same-state continuation from a shared checkpoint shows large separations among candidate bases, with the learned basis best in every benchmark-backbone pair we test. No-retraining projection preserves almost all FULL performance inside the learned component and removes it from random or frozen-activation PCA controls. Across GSM8K, MathQA, and AQuA, the best per-module continuation always appears at k ∈ {2, 4} across twelve seed-level cases. Global top-16 and top-32 learned subsets also outperform matched random subsets, and the largest single-direction effects cluster in late q\_proj, o\_proj, and down\_proj modules.

The main text centers on real-task mechanism evidence. The synthetic graph benchmark stays in the appendix, where it isolates shortcut shift and structural length shift under controlled conditions. Its role is to explain why write geometry can matter once the real-task diagnostics establish that the constrained parameterization is viable.

## Our contributions are:

1. We identify write geometry as a causal state variable of PEFT: from the same trained checkpoint, different write subspaces lead to different futures.

2. We show that trained LoRA writes are sharply concentrated: per-module optima appear at top-2 or top-4 learned directions, and learned global subsets outperform matched random subsets.

3. We localize the strongest effects to a sparse set of late q\_proj, o\_proj, and down\_proj components, and we use LEARNED-BASIS LORA as a mechanism probe that exposes that organization on real reasoning benchmarks.

## 2 Related Work

## 2.1 PEFT and Low-Rank Adaptation

PEFT methods reduce adaptation cost by freezing most backbone parameters and learning only a small task-specific interface. Early adapter work showed that small residual modules can recover much of the benefit of full fine-tuning [Houlsby et al., 2019]. Later variants broadened that idea through prompt tuning, prefix tuning, P-Tuning v2, BitFit, Compacter, and LoRA-style low-rank updates [Li and Liang, 2021, Lester et al., 2021, Liu et al., 2021, Zaken et al., 2022, Mahabadi et al., 2021, Hu et al., 2021]. QLoRA, AdaLoRA, and VeRA further optimized that interface for quantization, budget allocation, and storage efficiency [Dettmers et al., 2023, Zhang et al., 2023, Kopiczko et al., 2023]. This literature establishes low-rank adaptation as an efficient training interface. Our paper keeps that background fixed and asks a more structural question: once a strong task-specific LoRA update has been learned, how much of it is actually behaviorally essential?

## 2.2 Geometry-Aware and Spectral Variants of LoRA

The closest thread is the growing line of geometryaware LoRA variants. Some methods change the parameterization itself, for example by separating direction and magnitude as in Liu et al. [2024a]. Others initialize LoRA from spectral structure in pretrained weights, such as Meng et al. [2024], or emphasize different parts of that structure, such as Wang et al. [2024]. A third group explicitly constrains or learns subspaces through orthogonal, butterfly, or manifold structure, including She et al. [2025], Liu et al. [2024b], Wang et al. [2023], Lion et al. [2025], and Li et al. [2025]. More recent work updates or diversifies subspaces during training, as in Yang et al. [2025], Chang et al. [2026], and Xiao et al. [2026]. These papers are highly relevant because they show that update geometry is a first-order design choice rather than an incidental implementation detail.

The paper uses a geometry-constrained adapter as an instrument to expose the internal structure of the trained write update. The central question is: once a useful write space has formed, how concentrated is the behaviorally effective part of the update inside that space? This is why the core experiments are no-retraining projection, same-state basis-swap continuation, and top-k continuation inside the learned basis.

Recent post-hoc work is even closer in spirit. LoRA-Squeeze learns a larger adapter and compresses it into a smaller one [Vulic et al.´ , 2026]. Spectral Surgery reweights singular directions inside a trained adapter while keeping the learned subspace fixed [Tian et al., 2026]. Both methods assume that trained adapters contain internal redundancy. Our paper supplies direct behavioral evidence for that assumption and identifies where the effective write signal is concentrated.

## 2.3 Low-Dimensional Fine-Tuning and Representation Subspaces

Several results suggest that fine-tuning behaves as a low-dimensional phenomenon [Aghajanyan et al., 2020]. That observation helps explain why low-rank or sparse parameterizations can work at all, but it does not by itself identify which low-dimensional directions matter for behavior. Our method is also related to representationlevel intervention methods such as Wu et al. [2024a], which edit or finetune behavior through low-dimensional hidden-state interventions, and to projection-based techniques such as iterative nullspace projection and concept erasure [Ravfogel et al., 2020, Belrose et al., 2023]. The conceptual overlap is that subspaces in hidden space can have semantic or functional significance. The difference is that we use this idea as a persistent training-time write constraint inside LoRA and then ask a behavior-level question: how many learned write directions are actually needed for the best continuation?

## 2.4 Structured Control, Routing, and Selective Allocation

The broader representation-engineering literature shows that linear directions in hidden space can strongly influence behavior without modifying the full model weights [Subramani et al., 2022, Turner et al., 2023, Zou et al., 2023, Meng et al., 2022]. Adapter composition and routing work makes a related point in parameter space: MoLE and MixLoRA combine multiple LoRA experts through learned control [Wu et al., 2024b, Li et al., 2024], and FCPRAG performs sample-level fusion over passage-specific LoRA injections [Zhu et al., 2026d]. Recent work beyond standard PEFT applies the same selective-allocation principle in other forms. EXACT redistributes longcontext supervision [Zhu et al., 2026a], HeLa-Mem consolidates agent memory through associative structure [Zhu et al., 2026b], GapSight learns when a vision-language model should revisit local evidence [Zhu et al., 2026c], and Learning Less Is More shows that upper-layer attention timing can hurt pretraining when specialization outruns lower-layer feature formation [Zhu et al., 2026e]. These papers solve different tasks, but they share the idea that useful signal is localized and should be routed, weighted, or constrained selectively.

## 2.5 Where the Present Claim Sits

Taken together, these prior works already imply that geometry matters in adaptation. Our claim sits one step downstream. We ask what fraction of a trained write update is behaviorally essential once geometry has already been learned. The answer supported by our interventions is that the task-effective part of a trained LoRA write update is sparse and structured. Exact conversion, no-retraining projection, same-state basis-swap continuation, local top-k continuation, and global top-M selection all point to that same conclusion.

## 3 Method

## 3.1 Constrained Write Subspaces

Let $h _ { \ell } \in \mathbb { R } ^ { d }$ be a hidden state at layer ℓ. Standard LoRA adds a low-rank update

$$
\Delta h _ { \ell } = B _ { \ell } A _ { \ell } h _ { \ell } ,\tag{1}
$$

where $A _ { \ell } \in \mathbb { R } ^ { r \times d }$ and $B _ { \ell } \in \mathbb { R } ^ { d \times r }$ . This constrains rank, but it leaves the output directions of the write unconstrained.

We separate two questions:

1. What is the low-rank code the adapter computes?

2. In which output subspace is that code allowed to write?

To make the second question explicit, we fix a module-wise orthonormal basis $\bar { U _ { \ell } } \in \bar { \mathbb { R } ^ { d \times k } }$ with $U _ { \ell } ^ { \top } U _ { \ell } = I$ and parameterize

$$
\Delta h _ { \ell } = U _ { \ell } C _ { \ell } A _ { \ell } h _ { \ell } ,\tag{2}
$$

where $C _ { \ell } \in \mathbb { R } ^ { k \times r }$ is learned and $U _ { \ell }$ is frozen. Every adapter write is therefore confined to span(U<sub>ℓ</sub>). We refer to this parameterization as WRITE-SUBSPACE LORA.

## 3.2 Frozen-Basis Baselines

We study two direct ways to choose $U _ { \ell }$ before training.

Random basis. Sample a Gaussian matrix in $\mathbb { R } ^ { d \times k }$ and orthonormalize it. This baseline keeps the same bottleneck size but removes any geometry signal.

Frozen-activation PCA basis. Collect frozen activations on the train split, form the activation matrix $H _ { \ell } \in \mathbb { R } ^ { n \times d }$ for each target layer, center it, and keep its top-k right singular vectors. This basis reflects directions already used by the frozen backbone on the task distribution, but it is estimated before adaptation begins.

These baselines are scientifically useful because they answer whether any generic bottleneck helps and whether a purely frozen representation geometry is already sufficient.

## 3.3 Warmup-Induced Basis Extraction

Instead of choosing $U _ { \ell }$ from the frozen backbone alone, LEARNED-BASIS LORA estimates it from a partially trained FULL adapter. Suppose a warmup stage has produced a target-module write matrix $\mathrm { \Delta } _ { B _ { \ell } ^ { \mathrm { w a r m u p } } } ^ { \mathrm { w a r m u p } }$ . We then set

$$
U _ { \ell } = \mathrm { o r t h } ( B _ { \ell } ^ { \mathrm { w a r m u p } } ) , \qquad C _ { \ell } = U _ { \ell } ^ { \top } B _ { \ell } ^ { \mathrm { w a r m u p } } ,\tag{3}
$$

where orth $( B _ { \ell } ^ { \mathrm { w a r m u p } } )$ is an orthonormal basis for the column space of the warmup write. Then

$$
B _ { \ell } ^ { \mathrm { w a r m u p } } = U _ { \ell } C _ { \ell } ,\tag{4}
$$

so the warmup update can be rewritten exactly as a constrained update:

$$
B _ { \ell } ^ { \mathrm { w a r m u p } } A _ { \ell } h _ { \ell } = U _ { \ell } C _ { \ell } A _ { \ell } h _ { \ell } .\tag{5}
$$

This conversion matters because it lets training continue from the warmup solution without changing the represented update at the switch step. Any later gain or loss must therefore come from the frozen write space used after the switch.

## 3.4 A Trainable Recipe

Instead of estimating the basis entirely from the frozen backbone, we estimate it from a partially trained FULL adapter.

Stage 1: warmup. Train an unconstrained FULL adapter for $\bar { T _ { \mathrm { w a r m u p } } } \mathrm { s t e p s . }$

Stage 2: freeze the learned write basis. For each target module, extract the orthonormal basis of the warmup adapter’s learned output column space, convert the warmup solution into constrained form, and continue training only the constrained parameters.

This produces our main algorithm, LEARNED-BASIS LORA. The idea is simple: learn a useful write space first, then freeze it and continue training inside it. The method interpolates between two extremes:

1. freezing the basis too early, before the warmup adapter has learned useful write directions;

2. never freezing the basis at all, which reduces to FULL LoRA.

The empirical question is what this recipe reveals about the internal organization of the trained write update.

## 3.5 Why Timing and Module Alignment Matter

Two implementation details are central in the real benchmark. First, the basis must be module-wise and layer-aligned. Reusing one basis across nonaligned adapter layers mixes incompatible write spaces and weakens continuation. Second, the basis must be frozen late enough to reflect a meaningful adaptation trajectory. The experiments below show that an early warmup can produce a basis that is exact for the current weak adapter and still poor for the final task, while a later warmup yields a much stronger constrained continuation.

This decomposition yields four falsifiable structural predictions. First, warmup-to-subspace conversion should be representationally continuous. Second, if write geometry is a causal state variable, the same trained checkpoint should have different futures under different write subspaces. Third, if the useful update is concentrated, only a very small number of learned directions should be needed inside each module, and a smaller-thanfull learned subset should still dominate globally. Fourth, if the effect is structured, the strongest components should cluster in a limited set of late write modules. The experiments are organized around these predictions.

## 3.6 Interpretation

The paper studies a structural claim. For PEFT, rank leaves out an important variable: the directions into which the adapter is allowed to write. Those directions change held-out behavior, and the behaviorally effective part of the trained update is highly non-uniform. Random and frozen-activation PCA are frozen basis estimators; LEARNED-BASIS LORA tests whether learning the basis during adaptation is enough to expose a sparse and structured subset of write components that carries most of the downstream effect.

## 4 Experimental Setup

## 4.1 Main Real Benchmarks

The main text uses six partial real benchmarks: GSM8K-partial [Cobbe et al., 2021], CommonsenseQA [Talmor et al., 2019], StrategyQA [Geva et al., 2021], AQuA [Ling et al., 2017], ARC-Challenge [Clark et al., 2018], and MathQA [Amini et al., 2019]. Each benchmark is evaluated on two 3B instruction-tuned backbones, Qwen2.5- 3B-Instruct [Qwen Team, 2025] and Llama-3.2- 3B-Instruct via the Llama 3 family [Dubey et al., 2024]. The main split uses 1,024 train examples, 128 validation examples, and up to 256 test examples for every benchmark-backbone pair; AQuA uses its full 254-example test split. All reported runs use seeds 39 and 40, the last four target layers, and the shared target-module family q\_proj, o\_proj, and down\_proj. We report held-out test accuracy in the main text and keep per-seed detail only for the timing diagnostic in the appendix.

The paper’s primary evidence consists of four mechanism analyses: exact warmup-to-subspace conversion, same-state basis-swap continuation, no-retraining counterfactual projection, and concentration analysis via per-module top-k, global top-M, and single-direction ablations. The multibenchmark table serves as scope evidence that the constrained parameterization remains viable beyond the intervention tests.

## 4.2 Prompting and Readout

All rows within a given benchmark share exactly the same prompt and readout rule. GSM8K uses a chain-of-thought prompt that requires a final sentence of the form “The final answer is <number>.” and is scored by direct numeric extraction. CommonsenseQA, AQuA, and ARC-Challenge use a single-letter answer format, StrategyQA uses a yes/no format, and MathQA uses a workedsolution prompt that ends with “The final answer is <letter>.” The main table does not mix prompt recipes across methods within a benchmark.

## 4.3 Compared Methods

We compare seven families of runs.

1. Frozen base evaluation.

2. FULL LoRA as a larger unconstrained reference.

3. RANDOM-SUBSPACE.

4. PCA-SUBSPACE.

5. DORA.

6. PISSA.

7. LEARNED-BASIS LORA, the learned-basis warmup-and-freeze recipe.

The main parameter counts are as follows. On Qwen, FULL uses 967,680 trainable parameters, RANDOM-SUBSPACE and PCA-SUBSPACE use 972,352, DORA and PISSA use 682,496, and LEARNED-BASIS LORA uses 682,596. On Llama, FULL uses 991,232, RANDOM-SUBSPACE and PCA-SUBSPACE use 990,784, DORA and PISSA use 599,040, and LEARNED-BASIS LORA uses 599,384. The most decisionrelevant related-work comparison is therefore between LEARNED-BASIS LORA and the budgetmatched DORA/PISSA rows, while FULL serves as a higher-budget unconstrained reference.

## 4.4 Timing Study

The trainable method is also evaluated in a separate switch-analysis setting with explicit warmup schedules. The main timing comparison studies a 100-step warmup followed by 300 constrained steps, and a 300-step warmup followed by 100 constrained steps. This ablation is secondary diagnostic evidence: it shows that an early basis can be representationally exact for the current weak adapter and still be poor for later continuation.

## 4.5 Counterfactual Projection Test

On GSM8K-partial we add a no-retraining intervention test on top of trained FULL adapters from the main benchmark configuration. For each target module, we replace the learned write matrix by either its projection onto a candidate basis or the orthogonal residual left after that projection. We then re-evaluate the resulting model with no further optimization. This isolates write-space localization from training dynamics: if a candidate basis contains the useful signal, the projectedparallel model should retain most of the FULL performance and the orthogonal residual should fail.

![](images/6167d067b1e097fff0ac718087a16b3760cfd280371086d98c4f4c33808e4c06.jpg)  
Figure 1: Overview of the paper’s mechanism pipeline. A trained LoRA adapter does not use its write budget uniformly. We warm up an unconstrained adapter, orthonormalize its module-wise write columns into a learned basis, freeze that basis, and continue optimization in constrained form. Same-state basis swap, no-retraining projection, and top-k continuation then test whether behavior is concentrated inside a small learned write subspace.

## 4.6 Same-State Continuation and Top-k Continuation

We also run a stronger continuation test. Starting from a fixed trained warmup checkpoint, we convert the same source state into constrained form under different bases and continue training with the same remaining budget and hyperparameters. For the learned basis itself, we then study concentration at two levels. First, we keep only the top-k learned directions inside each target module and continue training inside that smaller subspace. Second, we rank directions globally via single-direction ablations and measure whether globally selected top-M learned subsets outperform matched random subsets. These concentration analyses are reported on GSM8K, MathQA, and AQuA with seeds 39 and 40.

## 4.7 Appendix-Only Controlled World

The synthetic graph benchmark stays in the appendix. It isolates shortcut shift and structural length shift under controlled conditions and explains why write geometry can matter once the real-task diagnostics establish that the constrained parameterization itself is viable.

## 5 Results

## 5.1 Write Geometry Is a Causal State Variable

We begin with the strongest intervention in the paper. Table 1 starts from the same warmup checkpoint, rewrites that identical trained state into constrained form under different bases, and continues training with the same remaining budget and hyperparameters. The learned same-task basis is best in every benchmark-backbone pair we test and beats the strongest control by more than 0.03 in five of the six rows. This establishes write geometry as a causal state variable: the same trained state has different futures under different write subspaces.

The effect is sharpest on GSM8K/Qwen, where learned continuation reaches 0.5703 while the strongest control remains at 0.3223, and it remains substantial on GSM8K/Llama at 0.6348 versus 0.5000. MathQA shows the same pattern on both backbones. StrategyQA/Qwen still favors the learned basis by a wide margin, while StrategyQA/Llama shows the weakest separation and marks the current boundary of the claim. Once a useful state has formed, different subspace factorizations of that same state support different futures.

<table><tr><td>Case</td><td>Learned</td><td>Random</td><td>PCA</td><td>Cross-task</td></tr><tr><td>GSM8K / Qwen</td><td>0.5703</td><td>0.3223</td><td>0.3203</td><td>0.3047</td></tr><tr><td>GSM8K / Llama</td><td>0.6348</td><td>0.5000</td><td>0.4883</td><td>0.4941</td></tr><tr><td>MathQA / Qwen</td><td>0.4980</td><td>0.1094</td><td>0.1152</td><td>0.1172</td></tr><tr><td>MathQA / Llama</td><td>0.4590</td><td>0.4023</td><td>0.3809</td><td>0.4258</td></tr><tr><td>StrategyQA / Qwen</td><td>0.6621</td><td>0.5645</td><td>0.5625</td><td>0.5625</td></tr><tr><td>StrategyQA / Llama</td><td>0.6895</td><td>0.6660</td><td>0.6719</td><td>0.6621</td></tr></table>

Table 1: Same-state basis-swap continuation from a fixed warmup checkpoint. For each case we start from the same trained FULL state, convert it into constrained form under four different bases, and continue training with the same remaining budget and hyperparameters. Reported values are held-out test accuracy averaged over seeds 39 and 40. The learned same-task basis is best in every row and beats the strongest control by more than 0.03 in five of the six benchmark-backbone pairs.

## 5.2 Counterfactual Projection Localizes the Useful Write Signal

Table 2 turns basis overlap into a no-retraining intervention test. Starting from a trained GSM8K FULL adapter, we replace each module-wise write matrix by either its projection onto a candidate basis or the orthogonal residual left after that projection, then re-evaluate the model with no further optimization. As a sanity check, projecting onto the final FULL basis itself exactly reproduces FULL on both backbones.

The learned basis behaves qualitatively differently from random and frozen-activation PCA. On Qwen, its projected-parallel model keeps 95.9% of the FULL write energy and still reaches 0.5312 test accuracy, while the orthogonal residual keeps only 4.1% of the energy and falls to 0.2930. On Llama, the same split is even sharper: the learned parallel component keeps 95.2% of the write energy and reaches 0.6094, only 0.0098 below FULL, whereas the orthogonal residual drops to 0.4707. The useful signal in the trained FULL update is therefore localized inside the learned write space.

Random and frozen-activation PCA show the opposite pattern. Their projected-parallel models collapse to 0.3105–0.3125 on Qwen and 0.4766– 0.5078 on Llama, while their orthogonal complements recover nearly all of FULL. Useful write signal is concentrated inside the learned write sub space and lies almost entirely outside the guessed frozen bases.

<table><tr><td colspan="3">Qwen2.5-3B-Instruct</td></tr><tr><td>Basis family</td><td>Parallel kept</td><td>Orthogonal residual</td></tr><tr><td>Learned basis</td><td>0.5312 (95.9%)</td><td>0.2930 (4.1%)</td></tr><tr><td>Random</td><td>0.3125 (0.6%)</td><td>0.6172 (99.4%)</td></tr><tr><td>Frozen-act PCA</td><td>0.3105 (0.8%)</td><td>0.6113 (99.3%)</td></tr></table>

Llama-3.2-3B-Instruct
<table><tr><td>Basis family</td><td>Parallel kept</td><td>Orthogonal residual</td></tr><tr><td>Learned basis</td><td>0.6094 (95.2%)</td><td>0.4707 (4.8%)</td></tr><tr><td>Random</td><td>0.4766 (0.3%)</td><td>0.6328 (99.7%)</td></tr><tr><td>Frozen-act PCA</td><td>0.5078 (0.9%)</td><td>0.6387 (99.1%)</td></tr></table>

Table 2: No-retraining counterfactual projection test on GSM8K-partial. Starting from a trained FULL adapter, we replace each module-wise write matrix by either its projection onto a candidate basis (parallel kept) or the complementary residual (orthogonal residual), then re-evaluate the model without further optimization. Parenthesized values report the corresponding fraction of FULL write energy. The learned basis keeps most of the useful write signal in its parallel component, whereas random and frozen-activation PCA bases leave almost all useful signal in the orthogonal residual. Projecting onto the final FULL basis itself exactly recovers FULL and is used as a sanity check in the text rather than a separate row.

<table><tr><td>Case</td><td>Learned parallel</td><td>Learned orthogonal</td><td>Random parallel</td><td>PCA parallel</td></tr><tr><td>MathQA / Qwen</td><td>0.4746 (97.1%)</td><td>0.1035 (2.9%)</td><td>0.1309 (0.6%)</td><td>0.1172 (2.9%)</td></tr><tr><td>MathQA / Llama</td><td>0.4609 (99.5%)</td><td>0.4023 (0.5%)</td><td>0.3984 (0.4%)</td><td>0.3867 (1.9%)</td></tr><tr><td>StrategyQA / Qwen</td><td>0.6699 (97.6%)</td><td>0.5625 (2.4%)</td><td>0.5625 (0.5%)</td><td>0.5625 (1.2%)</td></tr></table>

Table 3: Additional counterfactual projection cases in the same mechanism analysis. Each row starts from a trained FULL adapter on the stated benchmark, projects the learned write update onto the learned basis or onto a guessed frozen basis, and re-evaluates without further training. Parenthesized values report the retained fraction of FULL write energy. Together with the GSM8K counterfactual table above, these rows show the same write-localization pattern: the learned basis keeps most of the useful behavior, while guessed frozen bases do not.

Table 3 extends the same counterfactual analysis to three additional cases.

On MathQA/Qwen, the learned parallel component reaches 0.4746 while the orthogonal residual drops to 0.1035, and the random/PCA parallel controls remain at only 0.1309 and 0.1172. On MathQA/Llama, the learned parallel component reaches 0.4609, while the orthogonal residual drops to 0.4023 and the random/PCA parallel controls remain lower at 0.3984 and 0.3867. On StrategyQA/Qwen, the learned parallel component reaches 0.6699, while the orthogonal residual and both guessed-basis parallel controls all sit at 0.5625. Across all four cases, most of the useful FULL write signal survives projection onto the learned basis, and very little survives projection onto guessed frozen bases.

<table><tr><td>Case</td><td>Top-2 Top-4</td><td>Top-8</td></tr><tr><td>GSM8K / Qwen</td><td>0.5547 0.5488</td><td>0.5352</td></tr><tr><td>GSM8K / Llama MathQA / Qwen</td><td>0.6230</td><td>0.6348 0.6328 0.5273</td></tr><tr><td>MathQA / Llama</td><td>0.5195</td><td>0.5020</td></tr><tr><td></td><td>0.4766</td><td>0.4609 0.4512</td></tr><tr><td>AQuA / Qwen AQuA / Llama</td><td>0.3543 0.2539</td><td>0.3602 0.3543 0.2539 0.2539</td></tr></table>

Table 4: Same-state per-module top-k continuation under the learned basis. Starting from the same warmup checkpoint, we keep only the top-k learned directions inside each target module and continue training. Reported values are held-out test accuracy averaged over seeds 39 and 40. Across the twelve seed-level cases underlying this table, the best continuation is always achieved at $k \in \{ 2 , 4 \} ; k = 8$ is never uniquely necessary.

## 5.3 Local Concentration Inside Modules

The next question is how much of the learned basis is actually needed inside each module. Table 4 shows the answer. Across GSM8K, MathQA, and AQuA, the best per-module continuation always appears at $k \in \{ \bar { 2 } , 4 \}$ across all twelve seed-level cases we test. k = 8 is never uniquely necessary.

This is the clearest local concentration result in the paper. On GSM8K/Qwen, top-2 continuation reaches 0.5547, beating top-4 at 0.5488 and top-8 at 0.5352. On MathQA/Llama, top-2 reaches 0.4766, above top-4 at 0.4609 and top-8 at 0.4512. Even when top-4 is slightly better than top-2, as on GSM8K/Llama and MathQA/Qwen, the optimum still lies inside a very small per-module subset. Within modules, only a few learned directions are behaviorally effective.

## 5.4 Global Concentration and Sparse High-Impact Components

Table 5 asks a stricter question. If we rank learned directions globally across modules rather than keeping top-k directions inside every module, does the concentration effect survive? The answer is yes, but at a larger budget. On GSM8K/Qwen, learned global top-32 reaches 0.5527 while a matched random subset reaches only 0.3203. On MathQA/Qwen, the same comparison is 0.5039 versus 0.1621. The gaps are smaller on AQuA and on some Llama rows, but the learned global subsets still dominate in most settings. Global concentration survives ranking, although the useful global budget is larger than the per-module top-2/top-4 budget.

<table><tr><td>Case</td><td>Learned-4</td><td>Random-4</td><td>Learned-32</td><td>Random-32</td></tr><tr><td>GSM8K / Qwen</td><td>0.4414</td><td>0.3086</td><td>0.5527</td><td>0.3203</td></tr><tr><td>GSM8K / Llama</td><td>0.5137</td><td>0.4902</td><td>0.5859</td><td>0.4824</td></tr><tr><td>MathQA / Qwen</td><td>0.3359</td><td>0.1074</td><td>0.5039</td><td>0.1621</td></tr><tr><td>MathQA / Llama</td><td>0.4219</td><td>0.3926</td><td>0.4375</td><td>0.4199</td></tr><tr><td>AQuA / Qwen</td><td>0.3543</td><td>0.3248</td><td>0.3681</td><td>0.3445</td></tr><tr><td>AQuA / Llama</td><td>0.2165</td><td>0.2165</td><td>0.2539</td><td>0.2146</td></tr></table>

Table 5: Global top-M counterfactuals from singledirection ranking. We rank learned directions globally across modules, keep only the top-M learned directions, and compare against matched random subsets. Reported values are mean test accuracy over seeds 39 and 40. The learned subsets consistently outperform matched random subsets on the strongest arithmeticstyle cases, showing that the concentration effect survives global selection, although at a larger scale than the per-module top-2/top-4 result.

<table><tr><td>Case</td><td>Strongest component</td><td>Mean ∆</td><td>Flips</td><td>Dominant bias</td></tr><tr><td>GSM8K / Qwen</td><td>L30 q-proj d0</td><td>-0.1426</td><td>48.5</td><td>multistep / ratio</td></tr><tr><td>GSM8K / Llama</td><td>L20 o_proj d0</td><td>-0.0078</td><td>16.5</td><td>multistep / ratio</td></tr><tr><td>MathQA / Qwen</td><td>L28 down_proj d0</td><td>-0.0957</td><td>52.0</td><td>ratio</td></tr><tr><td>MathQA / Llama</td><td>L20 down_proj d0</td><td>-0.0312</td><td>48.5</td><td>ratio</td></tr><tr><td>AQuA / Qwen</td><td>L28 down_proj d0</td><td>-0.0138</td><td>15.0</td><td>ratio / multistep</td></tr><tr><td>AQuA / Llama</td><td>L20 down_proj d0</td><td>-0.0177</td><td>13.5</td><td>add/sub / ratio</td></tr></table>

Table 6: Highest-impact single directions from seedpooled ablations. ‘Mean $\Delta ^ { \prime }$ is the mean test-accuracy change relative to the corresponding FULL model when that one direction is removed. The strongest directions cluster in late q\_proj, o\_proj, and down\_proj modules rather than spreading uniformly across the adapter.

Table 6 adds the second structural piece: where the strongest components live.

Single-direction ablations identify a sparse set of late q\_proj, o\_proj, and down\_proj directions whose removal causes disproportionately large drops. The sharpest Qwen cases are latelayer q\_proj and down\_proj components; the strongest Llama cases shift across o\_proj and down\_proj, but still remain sparse and late. Additional rotated-basis controls are mixed, so the stable object is the sparse effect and its latemodule localization rather than one privileged coordinate system inside the learned subspace.

## 5.5 Timing Is Secondary Diagnostic Evidence

Table 7 provides supporting diagnostic evidence. It summarizes a timing study in the switchanalysis setting.

When the basis is frozen after only 100 warmup steps, the trainable constrained model is poor on Qwen and only modest on Llama. This is the clearest evidence that the write subspace can be exact for the current warmup adapter and still be the wrong subspace for the final task. A 300- step warmup changes the regime. On Qwen, the warmup FULL model already reaches 51.56% test accuracy in the diagnostic setting, and the constrained continuation improves it further to 53.13%. On Llama, the same 300-step schedule produces a constrained model at 60.16%, essentially matching the warmup model and closing most of the gap to the 60.94% FULL reference in that setting.

Warmup schedule Qwen warmup FULL Qwen LEARNED-BASIS LORA Llama warmup FULL Llama LEARNED-BASIS LORA 100 + 300 <sup>0.3711</sup><sub>0.5156</sub> 0.3145 0.5059 <sup>0.5508</sup><sub>0.6016</sub> 300 + 100 0.5313 0.6016 Table 7: Timing ablation for the trainable constrained recipe in the diagnostic switch-analysis setting. Each row reports held-out test accuracy averaged over seeds 39 and 40. Freezing the basis after only 100 warmup steps is too early, especially on Qwen. A 300-step warmup moves the method into a different regime: the constrained continuation improves over the weaker early-freeze schedule and preserves nearly all of the late-warmup performance. A separate Llama-only 350- step diagnostic run matched rather than improved the 300-step schedule and is discussed in the appendix.  
![](images/ac0e3ecd1430857f640a7854211fba776b1b68d66cd5271bc9f75069644a401d.jpg)  
Figure 2: Timing ablation for the trainable method. Freezing after 100 steps hurts because the learned write space is still immature. Freezing after 300 steps lands in a different regime: the constrained continuation improves over the warmup model on Qwen and preserves nearly all of it on Llama.

This timing result shows why a useful write space must first be formed. It does not identify which components inside that space are behaviorally essential. The same-state swap, projection, local top-k, and global top-M results answer that stronger structural question.

## 5.6 Main Real-Benchmark Table

Table 8 shows that the learned-basis recipe remains competitive beyond the mechanism probes. The table is split into a Qwen panel and a Llama panel so that backbone effects remain visible.

The first pattern comes from GSM8K. On both backbones, the guessed frozen bases are weak: RANDOM-SUBSPACE and PCA-SUBSPACE sit far below the strongest tuned PEFT rows. On Qwen, FULL is best at 0.6133, while DORA reaches 0.6035 and LEARNED-BASIS LORA ties PISSA at 0.5664. On Llama, LEARNED-BASIS LORA is the strongest row at 0.6367, narrowly above DORA at 0.6348 and PISSA at 0.6289. These cells match the mechanism picture above: once the learned write space has formed, a constrained continuation can stay strong.

Qwen2.5-3B-Instruct
<table><tr><td>Method</td><td>GSM8K</td><td>CSQA</td><td>StrategyQA</td><td>AQuA</td><td>ARC-Challenge</td><td>MathQA</td></tr><tr><td>Frozen base</td><td>0.3242</td><td>0.7773</td><td>0.5625</td><td>0.3228</td><td>0.8242</td><td>0.1094</td></tr><tr><td>FULL</td><td>0.6133</td><td>0.7813</td><td>0.6582</td><td>0.3720</td><td>0.8125</td><td>0.5000</td></tr><tr><td>RANDOM-SUBSPACE</td><td>0.3242</td><td>0.7754</td><td>0.6367</td><td>0.3346</td><td>0.8242</td><td>0.1191</td></tr><tr><td>PCA-SUBSPACE</td><td>0.2734</td><td>0.7734</td><td>0.6543</td><td>0.3386</td><td>0.8242</td><td>0.1758</td></tr><tr><td>DoRA</td><td>0.6035</td><td>0.7773</td><td>0.6348</td><td>0.3484</td><td>0.8223</td><td>0.4727</td></tr><tr><td>PISSA</td><td>0.5664</td><td>0.7852</td><td>0.6367</td><td>0.3602</td><td>0.8125</td><td>0.4727</td></tr><tr><td>LEARNED-BASIS LORA</td><td>0.5664</td><td>0.7793</td><td>0.6621</td><td>0.3445</td><td>0.8223</td><td>0.4980</td></tr></table>

<table><tr><td colspan="6">Method</td></tr><tr><td></td><td>GSM8K</td><td>CSQA</td><td>StrategyQA</td><td></td><td>AQuA ARC-Challenge</td><td>MathQA</td></tr><tr><td>Frozen base</td><td>0.4844</td><td>0.7188</td><td>0.6758</td><td>0.2441</td><td>0.7227</td><td>0.3711</td></tr><tr><td>FULL</td><td>0.6191</td><td>0.7305</td><td>0.6680</td><td>0.2657</td><td>0.7109</td><td>0.4707</td></tr><tr><td>RANDOM-SUBSPACE</td><td>0.5020</td><td>0.7227</td><td>0.6699</td><td>0.2441</td><td>0.7246</td><td>0.4023</td></tr><tr><td>PCA-SUBSPACE</td><td>0.5020</td><td>0.7285</td><td>0.6484</td><td>0.2362</td><td>0.7188</td><td>0.4648</td></tr><tr><td>DoRA</td><td>0.6348</td><td>0.7305</td><td>0.6797</td><td>0.2854</td><td>0.7207</td><td>0.4395</td></tr><tr><td>PISSA</td><td>0.6289</td><td>0.7383</td><td>0.6504</td><td>0.2500</td><td>0.7070</td><td>0.4629</td></tr><tr><td>LEARNED-BASIS LORA</td><td>0.6367</td><td>0.7344</td><td>0.6855</td><td>0.2677</td><td>0.7227</td><td>0.4766</td></tr></table>

Table 8: Extended main real-benchmark table, reporting held-out test accuracy averaged over seeds 39 and 40. The table is split into two backbone-specific panels to keep backbone effects readable. RANDOM-SUBSPACE and PCA-SUBSPACE are fixed-beforetraining guessed-basis controls, DORA and PISSA are official PEFT baselines, and LEARNED-BASIS LORA is the default learned-basis continuation row. GSM8K retains the prompt-aligned results used throughout the main benchmark.

The second pattern comes from the five additional benchmarks. On CommonsenseQA, PISSA remains strongest on both backbones, with LEARNED-BASIS LORA staying close. On StrategyQA, LEARNED-BASIS LORA stays strongest on both backbones. On AQuA, FULL is strongest on Qwen and DORA is strongest on Llama. ARC-Challenge is near ceiling: on Qwen the strongest tied rows are Frozen base, RANDOM-SUBSPACE, and PCA-SUBSPACE at 0.8242, while LEARNED-BASIS LORA reaches 0.8223 and matches DORA; on Llama, RANDOM-SUBSPACE is highest at 0.7246 and LEARNED-BASIS LORA ties the base model at 0.7227. MathQA is more favorable: FULL is strongest on Qwen at 0.5000 with LEARNED-BASIS LORA close at 0.4980, while LEARNED-BASIS LORA is strongest on Llama at 0.4766, above FULL at 0.4707. The broad pattern is a competitive constrained recipe on several reasoning-heavy cells, while near-ceiling multiple-choice tasks remain less informative about the mechanism.

## 5.7 What the Remaining Gap Means

The two backbones support different parts of the story. Llama is the cleanest end-to-end case for LEARNED-BASIS LORA in the main table: it leads on GSM8K, both StrategyQA cells, and MathQA, and it stays near the top on CommonsenseQA and ARC-Challenge. Qwen is more mixed: LEARNED-BASIS LORA leads on StrategyQA, stays within 0.0020 of FULL on MathQA, and reaches 0.8223 on ARC-Challenge, while FULL remains strongest on GSM8K and AQuA. The remaining challenge is to learn and freeze the basis robustly enough that the trainable recipe wins more consistently on the hardest Qwen cells.

## 5.8 What the Appendix Adds

The appendix serves two purposes. First, it records the per-seed real-benchmark numbers so that the timing story is auditable. Second, it preserves the synthetic graph evidence in the right role. In that controlled world, constrained write spaces improve shortcut-shift and length-shift generalization much more cleanly than they do on GSM8K. Those gains explain why write geometry is worth studying in the first place.

## 6 Discussion

The paper’s main result is simple: LoRA does not use its write budget uniformly. A small, structured subset of write components carries most of the behavior. The evidence is consistent across several intervention levels: same-state continuation shows that write geometry is causal, projection shows that useful signal is localized to the learned write space, local and global concentration tests show that the effect is sharply nonuniform, and single-direction ablations show that the strongest components cluster in a sparse set of late write modules.

This changes how to interpret write geometry in PEFT. The central question is how much of the learned write update is actually needed for behavior. On the current arithmetic-style reasoning family, the useful update is substantially smaller and more structured than the full trained low-rank write, with especially strong concentration inside modules and a weaker but still real concentration at the global level.

This perspective also clarifies what the smaller parameter counts mean. The parameter savings relative to the larger FULL reference are real: LEARNED-BASIS LORA uses 29.5% fewer trainable parameters on Qwen and 39.5% fewer on Llama. More importantly, the results show that rank budget and behavioral effectiveness are different objects. Recent post-hoc compression and spectrum-editing methods already suggest that trained adapters contain internal redundancy [Vulic et al.´ , 2026, Tian et al., 2026]. Our interventions make that point behaviorally explicit and localize redundancy at the level of write directions. Structured or routed adapter systems can be read through the same lens: they shape where the model is allowed to write, combine, or preserve task signal [Xiao et al., 2026, Chang et al., 2026, Wu et al., 2024b, Li et al., 2024, Zhu et al., 2026d].

## 6.1 Limitations

The paper does not claim a unique canonical basis-column mechanism. Rotated controls are mixed, which indicates that the stable object is the sparse structured write effect rather than one privileged coordinate system inside the learned subspace. The global concentration effect is also weaker than the per-module concentration effect, so the current evidence does not reduce the whole model to only two or four global directions. Finally, this is a structural mechanism paper rather than a full semantic explanation paper. We identify sparse high-impact write components and where they live, but we do not explain every component in natural-language algorithmic terms.

## 7 Conclusion

This paper isolates the geometry of LoRA writes after adaptation and shows that the behaviorally effective part of a trained update is sparse and structured. Projection shows that useful signal is localized to the learned write space. Same-state continuation shows that different write subspaces give the same trained checkpoint different futures. Per-module top-k continuation shows strong local concentration, global top-M selection shows that this concentration survives global ranking, and single-direction ablations show that a sparse set of late q\_proj, o\_proj, and down\_proj components carries outsized behavioral impact.

The contribution is a structural mechanism result with practical consequences. Geometryconstrained low-rank adaptation remains viable on real tasks, and the stronger finding is how little of the learned write space is actually needed. Future PEFT methods should allocate budget toward high-impact write components instead of treating every direction in a low-rank update as equally valuable. LEARNED-BASIS LORA is useful because it exposes that internal structure clearly enough to measure it.

## References

Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin de Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. Parameter-efficient transfer learning for NLP, 2019. URL https://arxiv.org/abs/ 1902.00751.

Xiang Lisa Li and Percy Liang. Prefix-tuning: Optimizing continuous prompts for generation, 2021. URL https://arxiv.org/abs/2101.00190.

Brian Lester, Rami Al-Rfou, and Noah Constant. The power of scale for parameter-efficient prompt tuning, 2021. URL https://arxiv.org/ abs/2104.08691.

Xiao Liu, Kaixuan Ji, Yicheng Fu, Weng Lam Tam, Zhengxiao Du, Zhilin Yang, and Jie Tang. P-tuning v2: Prompt tuning can be comparable to fine-tuning universally across scales and tasks, 2021. URL https://arxiv.org/abs/2110. 07602.

Elad Ben Zaken, Yoav Goldberg, and Shauli Ravfogel. Bitfit: Simple parameter-efficient finetuning for transformer-based masked languagemodels, 2022. URL https://arxiv.org/abs/2106. 10199.

Rabeeh Karimi Mahabadi, James Henderson, and Sebastian Ruder. Compacter: Efficient lowrank hypercomplex adapter layers, 2021. URL https://arxiv.org/abs/2106.04647.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models, 2021. URL https://arxiv.org/abs/2106.09685.

Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. QLoRA: Efficient finetuning of quantized LLMs, 2023. URL https://arxiv.org/abs/2305.14314.

Ning Ding, Yujia Qin, Guang Yang, Fuchao Wei, Zonghan Yang, Yusheng Su, Ergui Zhao, Yue Hu, Cong Sun, Dahua Lin, et al. Parameterefficient fine-tuning of large-scale pre-trained language models: A survey, 2023. URL https: //arxiv.org/abs/2303.15647.

Qingru Zhang, Minshuo Chen, Alexander Bukharin, Nikos Karampatziakis, Pengcheng He, Yu Cheng, Weizhu Chen, and Tuo Zhao. AdaLoRA: Adaptive budget allocation for parameter-efficient fine-tuning, 2023. URL https://arxiv.org/abs/2303.10512.

Dawid J. Kopiczko, Tijmen Blankevoort, and Yuki M. Asano. VeRA: Vector-based random matrix adaptation, 2023. URL https: //arxiv.org/abs/2310.11454.

Shih-Yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo Molchanov, Yu-Chiang Frank Wang, Kwang-Ting Cheng, and Min-Hung Chen. DoRA: Weight-decomposed low-rank adaptation, 2024a. URL https://arxiv.org/abs/2402. 09353.

Fanxu Meng, Zhaohui Wang, and Muhan Zhang. PiSSA: Principal singular values and singular vectors adaptation of large language models, 2024. URL https://arxiv.org/abs/2404.02948.

Hanqing Wang, Yixia Li, Shuo Wang, Guanhua Chen, and Yun Chen. MiLoRA: Harnessing minor singular components for parameterefficient LLM finetuning, 2024. URL https: //arxiv.org/abs/2406.09044.

Yifei She, Xinhao Wei, and Yulong Wang. Dis-LoRA: Task-specific low-rank adaptation via orthogonal basis from singular value decomposition. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 13740–13755. Association for Computational Linguistics, 2025. doi: 10. 18653/v1/2025.emnlp-main.694. URL https: //aclanthology.org/2025.emnlp-main.694/.

Weiyang Liu, Zeju Qiu, Yao Feng, Yuliang Xiu, Yuxuan Xue, Longhui Yu, Haiwen Feng, Zhen Liu, Juyeon Heo, Songyou Peng, Yandong Wen, Michael J. Black, Adrian Weller, and Bernhard Schölkopf. Parameter-efficient orthogonal finetuning via butterfly factorization, 2024b. URL https://arxiv.org/abs/2311.06243.

Xiao Wang, Tianze Chen, Qiming Ge, Han Xia, Rong Bao, Rui Zheng, Qi Zhang, Tao Gui, and Xuanjing Huang. Orthogonal subspace learning for language model continual learning, 2023. URL https://arxiv.org/abs/2310.14152.

Kai Lion, Liang Zhang, Bingcong Li, and Niao He. PoLAR: Polar-decomposed low-rank adapter representation, 2025. URL https:// arxiv.org/abs/2506.03133.

Zhizhong Li, Sina Sajadmanesh, Jingtao Li, and Lingjuan Lyu. StelLA: Subspace learning in low-rank adaptation using stiefel manifold, 2025. URL https://arxiv.org/abs/2510.01938.

Haodong Yang, Lei Wang, and Md Zakir Hossain. SRLoRA: Subspace recomposition in low-rank adaptation via importance-based fusion and reinitialization, 2025. URL https: //arxiv.org/abs/2505.12433.

Xi Xiao, Chenrui Ma, Yunbei Zhang, Chen Liu, Zhuxuanzi Wang, Yanshu Li, Lin Zhao, Guosheng Hu, Tianyang Wang, and Hao Xu. Not all directions matter: Toward structured and task-aware low-rank adaptation, 2026. URL https://arxiv.org/abs/2603.14228.

Yupeng Chang, Yuan Wu, and Yi Chang. SOS-LoRA: Static orthogonal-subspace low-rank adaptation with fixed multi-scale scaling, 2026. URL https://arxiv.org/abs/2607.16252.

Jinchang Zhu, Jindong Li, Chengyu Zou, Rong Fu, Chao Wang, Haowei He, and Menglin Yang. Where does long-context supervision actually go? effective-context exposure balancing, 2026a. URL https://arxiv.org/abs/2605. 10544.

Jinchang Zhu, Jindong Li, Chenghao Zhang, Jiacheng Liu, and Menglin Yang. HeLa-Mem: Hebbian learning and associative memory for LLM agents, 2026b. URL https://arxiv.org/abs/ 2604.16839.

Jinchang Zhu, Rong Fu, Yi Ding, Chenghao Wu, Ying Liu, and Menglin Yang. Learning to look again: Loss-gap supervision for free-form crop routing in vision-language models, 2026c. URL https://arxiv.org/abs/2608.21762.

Jinchang Zhu, Jindong Li, Yi Ding, Xiaojian Nie, Rong Fu, Shuangyong Song, Haowei He, and Menglin Yang. FCPRAG: Fusion-controller parametric retrieval-augmented generation for stable multi-passage LoRA injection, 2026d. URL https://arxiv.org/abs/2608.21750.

Jinchang Zhu, Jindong Li, Yuwen Hao, Chengyu Zou, Rong Fu, and Menglin Yang. Learning less is more: Premature upper-layer attention specialization hurts language model pretraining, 2026e. URL https://arxiv.org/abs/2605. 10504.

Ivan Vulic, Adam Grycner, Quentin de Larous-´ silhe, and Jonas Pfeiffer. LoRA-squeeze: Simple and effective post-tuning and in-tuning compression of LoRA modules, 2026. URL https://arxiv.org/abs/2602.10993.

Zailong Tian, Yanzhe Chen, Zhuoheng Han, and Lizi Liao. Spectral surgery: Training-free refinement of LoRA via gradient-guided singular value reweighting, 2026. URL https: //arxiv.org/abs/2603.03995.

Armen Aghajanyan, Luke Zettlemoyer, and Sonal Gupta. Intrinsic dimensionality explains the effectiveness of language model fine-tuning, 2020. URL https://arxiv.org/abs/2012.13255.

Zhengxuan Wu, Aryaman Arora, Zheng Wang, Atticus Geiger, Dan Jurafsky, Christopher D. Manning, and Christopher Potts. ReFT: Representation finetuning for language models, 2024a. URL https://arxiv.org/abs/2404.03592.

Shauli Ravfogel, Yanai Elazar, Yoav Goldberg, and Ryan Cotterell. Null it out: Guarding protected attributes by iterative nullspace projection. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, 2020.

Nora Belrose, David Schneider-Joseph, Shauli Ravfogel, Ryan Cotterell, Edward Raff, and Stella Biderman. LEACE: Perfect linear concept erasure in closed form, 2023. URL https: //arxiv.org/abs/2306.03819.

Nishant Subramani, Sam Bowman, and Dan Klein. Extracting latent steering vectors from pretrained language models, 2022. URL https: //arxiv.org/abs/2205.05124.

Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J. Vazquez, Ulisse Mini, and Monte MacDiarmid. Steering language models with activation engineering, 2023. URL https://arxiv.org/abs/2308.10248.

Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, Shashwat Goel, Nathaniel Li, Michael J. Byun, Zifan Wang, Alex Mallen, Steven Basart, Sanmi Koyejo, Dawn Song, Matt Fredrikson, J. Zico Kolter, and Dan Hendrycks. Representation engineering: A top-down approach to AI transparency, 2023. URL https://arxiv.org/abs/2310.01405.

Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing factual associations in GPT, 2022. URL https://arxiv. org/abs/2202.05262.

Xun Wu, Shaohan Huang, and Furu Wei. Mixture of LoRA experts, 2024b. URL https://arxiv. org/abs/2404.13628.

Dengchun Li, Yingzi Ma, Naizheng Wang, Zhengmao Ye, Zhiyuan Cheng, Yinghao Tang, Yan Zhang, Lei Duan, Jie Zuo, Cal Yang, and Mingjie Tang. MixLoRA: Enhancing large language models fine-tuning with lorabased mixture of experts, 2024. URL https: //arxiv.org/abs/2404.15159.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems, 2021. URL https://arxiv. org/abs/2110.14168.

Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. Commonsenseqa: A question answering challenge targeting commonsense knowledge. In Proceedings of the 2019 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4149– 4158, Minneapolis, Minnesota, 2019. Association for Computational Linguistics. doi: 10.

18653/v1/N19-1421. URL https://aclanthology. org/N19-1421/.

Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot, Dan Roth, and Jonathan Berant. Did aristotle use a laptop? a question answering benchmark with implicit reasoning strategies. Transactions ofthe Associationfor Computational Linguistics, 9:346–361, 2021. doi: 10. 1162/tacl\_a\_00370. URL https://aclanthology. org/2021.tacl-1.21/.

Wang Ling, Dani Yogatama, Chris Dyer, and Phil Blunsom. Program induction by rationale generation: Learning to solve and explain algebraic word problems. In Proceedings ofthe 55th Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), pages 158–167, Vancouver, Canada, 2017. Association for Computational Linguistics. doi: 10.18653/v1/P17-1015. URL https://aclanthology.org/P17-1015/.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try ARC, the AI2 reasoning challenge, 2018. URL https://arxiv.org/abs/ 1803.05457.

Aida Amini, Saadia Gabriel, Shanchuan Lin, Rik Koncel-Kedziorski, Yejin Choi, and Hannaneh Hajishirzi. MathQA: Towards interpretable math word problem solving with operationbased formalisms. In Proceedings ofthe 2019 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 2357–2367, Minneapolis, Minnesota, 2019. Association for Computational Linguistics. doi: 10.18653/ v1/N19-1245. URL https://aclanthology.org/ N19-1245/.

Qwen Team. Qwen2.5 technical report, 2025. URL https://arxiv.org/abs/2412.15115.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The Llama 3 herd of models, 2024. URL https://arxiv.org/ abs/2407.21783.

<table><tr><td>Seed</td><td>Base</td><td>FULL</td><td>RANDOM-SUBSPACE</td><td>PCA-SUBSPACE</td><td>Warmup FULL</td><td>LEARNED-BASIS LORA</td></tr><tr><td>39</td><td>0.2266</td><td>0.4688</td><td>0.2813</td><td>0.2773</td><td>0.5156</td><td>0.5430</td></tr><tr><td>40</td><td>0.2266</td><td>0.5469</td><td>0.1992</td><td>0.2813</td><td>0.5156</td><td>0.5195</td></tr></table>

Table 9: Per-seed GSM8K-partial test accuracy for the diagnostic Qwen2.5-3B-Instruct timing study. “Warmup FULL” refers to the 300-step unconstrained warmup used by LEARNED-BASIS LORA before basis freezing. The key pattern is stable in this diagnostic setting: both seeds end with constrained models above the corresponding FULL baseline.

<table><tr><td>Seed</td><td>Base</td><td>FULL</td><td>RANDOM-SUBSPACE</td><td>PCA-SUBSPACE</td><td>Warmup FULL</td><td>LEARNED-BASIS LORA</td></tr><tr><td>39</td><td>0.4570</td><td>0.6055</td><td>0.5039</td><td>0.4648</td><td>0.5820</td><td>0.5781</td></tr><tr><td>40</td><td>0.4570</td><td>0.6133</td><td>0.4531</td><td>0.4883</td><td>0.6211</td><td>0.6250</td></tr></table>

Table 10: Per-seed GSM8K-partial test accuracy for the diagnostic Llama-3.2-3B-Instruct timing study. The 300-step warmup gives one seed slightly above FULL and one seed slightly below it, which explains why the mean trainable result is a near-match in this diagnostic setting.

## A Additional Real-Benchmark Details

This appendix records the diagnostic realbenchmark runs that underlie the mechanism sections in the main text. These runs use train/validation/test sizes of 1,024/128/256, seeds 39 and 40, and a shared target-module family over the last four layers. They should be read as mechanism diagnostics rather than as the main benchmark table.

The appendix now has two jobs. First, it preserves the timing diagnostics that explain why a useful learned write space exists. Second, it records the continuation diagnostics that show how little of that write space is actually needed once it has formed.

## A.1 Timing Diagnostics

In this diagnostic setting, Qwen is the clearest end-to-end success case and Llama is more conservative, but both show the same high-level pattern: an early basis can be exact for the current weak adapter and still be poor for later continuation, while a later learned basis is much stronger. These timing records explain why the learned write space should be treated as a meaningful object in the first place.

## A.2 Same-State Basis-Swap Diagnostics

The main text reports seed-averaged same-state basis-swap continuation results across GSM8K, MathQA, and StrategyQA. The per-seed records matter because they show that the learned sametask basis is consistently above the strongest guessed-basis control in nearly every arithmeticstyle case. The weakest separation appears on

<table><tr><td>Method</td><td>Trainable params</td><td>Val</td><td>Shortcut shift</td><td>Length shift</td></tr><tr><td>FULL (r = 16)</td><td>1,359,872</td><td>0.8281</td><td>0.8444</td><td>0.6764</td></tr><tr><td>FULL (budget-match)</td><td>967,680</td><td>0.8815</td><td>0.8965</td><td>0.7220</td></tr><tr><td>RANDOM-SUBSPACE</td><td>967,424</td><td>0.8854</td><td>0.9043</td><td>0.7533</td></tr><tr><td>PROBE-SUBSPACE</td><td>967,424</td><td>0.8958</td><td>0.9043</td><td>0.7546</td></tr><tr><td>PCA-SUBSPACE</td><td>967,424</td><td>0.9036</td><td>0.9147</td><td>0.7624</td></tr></table>

Table 11: Appendix-only synthetic graph summary in the controlled shortcut/length-shift world. A fixed write bottleneck helps most strongly when shortcut shift and structural length shift can be separated cleanly.

StrategyQA/Llama, which is why the manuscript treats that pair as a boundary case.

## A.3 Top-k Continuation Diagnostics

The strongest new mechanism result is the samestate top-k continuation analysis. Starting from the same learned-basis warmup checkpoint, we retain only the top-k learned directions and continue training. The main text reports the seed-averaged results on GSM8K, MathQA, and AQuA. The corresponding per-seed records show two robust facts. Best continuation does not require the full learned subspace, and the optimum appears in a very small range, typically top-2 or top-4, instead of increasing monotonically with larger k.

## B Synthetic Graph Evidence as Mechanism Support

The synthetic graph world remains valuable because it exposes two failure modes that are difficult to isolate cleanly in a real benchmark: shortcut-sensitive surface cues and structural length shift. We keep it in the appendix because the paper’s main claim is anchored in real-task evidence.

The pattern in Table 11 is substantially stronger than the GSM8K result. Under controlled shortcut and length shifts, all constrained variants beat the fixed-rank FULL baseline, and PCA-SUBSPACE also beats the near-budget FULL control on all three held-out metrics. The synthetic world therefore acts as a microscope for the mechanism: when task structure and shortcut-sensitive directions can be separated cleanly, constraining writes is a strong inductive bias. On real benchmarks, the same bias still helps, but the gains are smaller because the separation is noisier and the update needs are broader.