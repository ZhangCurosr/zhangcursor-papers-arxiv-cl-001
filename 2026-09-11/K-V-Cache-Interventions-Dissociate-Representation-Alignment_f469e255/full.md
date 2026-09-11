# K/V-Cache Interventions Dissociate Representation Alignment from Persona Expression in Decoder-Only Language Models

Yu Sun ysun1@linkedin.com

Mengyin Lu melu@linkedin.com

Cong Feng cofeng@linkedin.com

Guangming Lu glu@linkedin.com

Huimin Han huhan@linkedin.com

## Abstract

We study K/V-cache interventions— transplanting a target-conditioned K/V trajectory into a source-persona generation—as a structured surface for persona control in decoder-only language models. Across 13 intervention configurations applied to Llama-3.1-8B for a fixed source→target persona pair, we report two consistent dissociations between representation-level alignment and behavioral expression, plus a common failure under position-perturbing interventions. First, all layer-band K/V replacement interventions (early, mid, and late) achieve strong local V-space alignment within their replaced layers (V-gap 0.91, 0.89, 0.84), but only mid-layer replacement (layers 9–20) combines substantial target-marker expression with a comparatively preserved lexical-diversity profile. Second, interventions that induce comparable V-space alignment (full replacement vs. mid-layer replacement, V-gap 0.94 vs. 0.89) produce substantially different lexical-diversity profiles (TTR 0.65 vs. 0.77). Third, position-perturbing interventions (lag and shuffle of K/V positions) apply distinct positional operations yet uniformly suppress target-persona expression in generated text—a common behavioral failure rather than a representation-behavior dissociation in the strict sense. Partial source–target K/V interpo lation produces graded, monotonic behavioral effects over the two tested strengths. These observations indicate that representation-level similarity metrics alone are not sufficient predictors of downstream persona expression in the intervention regimes we study, and locate the K/V cache as a controllable but structurally constrained intervention surface. Because the transplanted trajectory carries the target’s own generated token history, we characterize the intervention as trajectory-level transplantation rather than isolated persona-representation injection. A same-token-sequence control, in which the source and target caches decode an

identical token sequence and differ only in persona conditioning, reproduces the sign and layer localization of the L28 representational shift on the same metric, indicating that the shift is not explained solely by imported token history. These findings characterize the structure of representation–behavior dissociation under K/V intervention in a high-signal setting, rather than establishing universality across models or persona pairs.

## 1 Introduction

Persona control in large language models—broadly, the task of maintaining a specified character, role, or stylistic identity in generated text—is approached through methods that intervene at different points in the autoregressive inference loop: at the input tokens (prompts, in-context examples, prefix tuning), at the model weights (full fine-tuning, low-rank adaptation, RL-based optimization), or at intermediate activations (single- or multi-layer steering vectors injected into the residual stream). The K/V cache is the architectural channel by which past-step computation persists into future steps, yet it has been studied primarily as a target for compression or analysis rather than as a direct intervention surface for behavioral control.

Single-layer activation-channel interventions in our setup exhibit reset dynamics (measured in §4.1; full methodology in Appendix A)—the per-step ratio between controller forcing and model restoring force is R ≈ 1 with anti-aligned direction on essentially every generation step. This prevents per-step injections from accumulating into a persistent behavioral shift and motivates investigating K/V interventions, which by architectural design persist across generation steps. It is plausible that K/V-level interventions provide a controllable surface that single-layer activation methods do not; it is equally plausible that the structural constraints of attention—layer-by-layer K/V dependencies, position-sensitive routing, and crosslayer manifold consistency—introduce their own limits on what kinds of K/V modifications produce coherent behavior.

This paper investigates the K/V cache as a persona-control intervention surface. Each configuration transplants a target-conditioned reference K/V trajectory (captured from an independent target-persona rollout) into a source-persona generation; we therefore treat the intervention as trajectory-level transplantation rather than isolated persona-representation injection. We construct a battery of 13 intervention configurations spanning full replacement (v1\_full), partial layer-band replacement (v2 across three coarse and three fine layer bands), position-perturbing operations (v3 shuffle and lag), and source–target interpolation (v4 at two interpolation strengths). For each configuration we measure three complementary signals: text-level persona-marker density, L28 hidden-state V<sup>⋆</sup>-projection (a per-axis-rescaled distance to perpersona centroids learned from prior work), and K/V cosine alignment with paired source-prompt and target-prompt reference runs.

Findings. We observe three patterns in this battery, each supported by at least two of the three measurement families:

• Mid-layer concentration. K/V replacement at layers 9–20 produces both targetaligned representations (V-gap 0.89) and target-marker transfer with a preserved lexicaldiversity profile (target marker density 15.5, TTR 0.77). Replacements at early (1–8) or late (21–31) bands produce high V-alignment within their respective replaced layers (V-gap 0.91 and 0.84), but this alignment does not transfer to other layers and does not produce target-marker transfer.

• Alignment does not determine behavior. Full K/V replacement (v1\_full) achieves the highest V-alignment (0.94) and the highest target marker density (24.8), but at substantially higher lexical repetition (TTR 0.65). Midlayer replacement (v2\_L9-20) achieves nearly equal V-alignment (0.89) with a preserved lexical-diversity profile (TTR 0.77). Comparable representation-level alignment thus corresponds to qualitatively different behavioral outcomes.

• Distinct perturbations yield a common behavioral failure. Position-perturbing interventions (lag and shuffle of K/V positions) apply qualitatively different operations to the source-prompt K/V cache, yet neither injects target-persona representation and both yield minimal target-persona text expression (target marker density ≤ 0.5). Here representation and behavior agree—neither shows target transfer—a common failure mode rather than a representation-behavior dissociation in the strict sense.

Taken together, these results indicate that representation-level similarity metrics alone— whether measured in K/V space or via hidden-state projection onto a persona subspace—do not sufficiently predict downstream persona expression in the intervention regimes we study.

Contribution. We document an empirical regularity: a multi-channel dissociation between representation-level alignment and behavioral expression that holds across 13 K/V intervention configurations and two complementary representationspace measurements on Llama-3.1-8B with a single source→target persona pair. The contribution of this work is the characterization of the K/V intervention surface itself. The mid-layer concentration we identify is a design implication of that characterization rather than a benchmark-optimized steering method.

## 2 Related Work

Activation, token, and weight channels. Singlelayer and multi-layer activation injection methods include CAA (Panickssery et al., 2024), ActAdd (Turner et al., 2023), RepE (Zou et al., 2023), ITI (Li et al., 2023), function vectors (Todd et al., 2023), and persona vectors (Chen et al., 2025). Token-channel methods — prompting, in-context demonstrations, soft prompts (Lester et al., 2021), and prefix tuning (Li and Liang, 2021) — operate by introducing tokens that the model’s causal attention re-reads each step. Weight-channel methods — fine-tuning, LoRA, RLHF, DPO, and model-editing methods such as ROME (Meng et al., 2022) and MEMIT (Meng et al., 2023) — modify the parameters of the model itself. We establish in Appendix A that single-layer activation injection in our setup is subject to per-step reset dynamics; the K/V-level constraints we document in this paper are distinct from this activation-channel constraint and from the token- and weight-channel pathways.

K/V-cache analysis and intervention. Prior work on K/V cache has studied compression (Zhang et al., 2023), attention-head specialization (Olsson et al., 2022), and prefix-tuning-style learned cache prefixes. Direct intervention on the K/V cache as a persona-control surface, with systematic ablation across layer bands, position perturbations, and interpolation strengths, has not been reported in the persona-control literature to our knowledge. Recent work in video character generation (Zeng et al., 2026) achieves long-horizon identity stability through a combination of multireference token-channel conditioning and weightchannel distillation, consistent with our observation that long-horizon persona stability is achievable via these channels.

Representation–behavior dissociation. The general phenomenon that representation similarity metrics may fail to predict behavioral outcomes has been reported in interpretability literature, e.g. in the context of probing classifiers and representation engineering (Belrose et al., 2023). Our observation extends this in two specific directions: (i) we identify dissociation in both directions (similar alignment with different behavior; different misalignment with similar behavior), and (ii) we corroborate across two complementary representation spaces (hidden-state projection and K/V cosine) under the same intervention battery.

Concurrent work. Three recent studies share our representation-versus-behavior framing in adjacent settings. Jiang et al. (2026) document an “audit gap”: safety-aligned models that match their base on every static behavioral audit yet give way to small latent-state perturbations, a representation– behavior decoupling established through latent perturbation and fine-tuning in the safety domain rather than through K/V-cache intervention. Kang et al. (2026) identify “K/V-cache contamination”— steered token states stored and reused across steps— as a failure mode of residual-stream activation steering in multi-turn dialogue, and propose a gated attention-delta method (GCAD) to restore coherence; we instead intervene on the K/V cache directly and characterize the resulting representation– behavior structure across a static intervention battery, rather than proposing a control method. Baez et al. (2026) dissociate sycophancy representations into factual and opinion subtypes via linear probes and steering vectors, a precedent for representation-subtype dissociation distinct from our representation–behavior focus.

## 3 Methods

## 3.1 Setup

We use Llama-3.1-8B-Instruct (Llama Team, AI @ Meta, 2024) at bfloat16 precision on a single H100 GPU (32 decoder layers, 32 query heads, 8 K/V heads with grouped-query attention, head dimension 128). We focus on a single source → target persona pair from a 30-persona corpus introduced in our prior work: health\_nurse\_dry (source “Nurse Reyes”: 30-year ER veteran, terse and deadpan) → tech\_novice\_anxious (target “Priya”: coding-bootcamp graduate, anxious and apologetic; full system prompts in Appendix C). The two personas are far apart in $\mathrm { V } ^ { \star } .$ -space $( \| c _ { \mathrm { t g t } } - c _ { \mathrm { s r c } } \| = 2 2 . 5 $ raw, 7.2 in per-axis-rescaled units). The pair was selected as a high-separation setting (7.2 vs. typical ∼ 3 across the corpus), which increases signalto-noise for resolving layer-localized intervention structure; whether the same concentration profile holds at smaller separations remains open. For each intervention configuration we generate n = 10 samples at temperature 0.7 with max\_new\_tokens = 50, conditioned on a single seed user message (casual\_hobby: “My sourdough starter died. What did I do wrong?”). The V<sup>⋆</sup> subspace (10-dim, supervised, at L28) and per-persona centroids are reused from prior work without modification.

## 3.2 Intervention battery

We construct 13 K/V intervention configurations covering four categories.

v1\_full: full K/V replacement. We first run a reference forward pass with the target-persona system prompt and the same seed user message, capturing the K/V cache trajectory $\mathcal { R } ^ { \mathrm { t g t } } = \{ ( K _ { t } ^ { \mathrm { t g t } } , V _ { t } ^ { \mathrm { t g t } } ) \} _ { t = 0 } ^ { T - 1 }$ During the manipulated run with the sourcepersona system prompt, after each forward pass we overwrite the entire past\_key\_values state with the target reference at the corresponding step. Because the reference is an independent targetpersona rollout, ${ \mathcal { R } } ^ { \mathrm { t g t } }$ encodes the target’s own generated token history; the intervention therefore transplants a target-conditioned trajectory, and we do not isolate the contribution of that generated history from persona conditioning (§6).

v2\_partial: layer-band K/V replacement. Identical to v1\_full but overwriting only at a subset of layers $L \subset \{ 0 , \ldots , 3 1 \}$ . We test three coarse bands $( \mathrm { L } 1 - 8 , \mathrm { L } 9 - 2 0 , \mathrm { L } 2 1 - 3 1 )$ and three fine sub-bands within the mid range (L9–12, L13–16, L17–20).

v3 position-perturbing: shuffle and lag. Structural perturbations to the source-prompt K/V cache without injecting target K/V. After source-prompt prefill the per-layer K/V (along the sequenceposition axis) is either randomly permuted (shuffle) or cyclically shifted by $\ell \in \{ 1 , 3 , 5 \}$ positions (lag).

v4\_interp: source–target K/V interpolation. Identical to v2\_L9–20 but linearly interpolating each manipulated layer’s K/V with the target reference at each step:

$$
K _ { t , l } ^ { \mathrm { r u n } } \gets \alpha \cdot K _ { t , l } ^ { \mathrm { t g t } } + ( 1 - \alpha ) \cdot K _ { t , l } ^ { \mathrm { s r c } } ,\tag{1}
$$

with $l \in L _ { 9 - 2 0 }$ and analogously for $V ;$ we test $\alpha \in \{ 0 . 2 5 , 0 . 5 \}$

v5\_tf: same-token-sequence target-forward control. The v1/v2 interventions transplant a targetconditioned reference trajectory that carries the target’s own generated token history, so they do not separate persona conditioning from imported token history (§6). To isolate the persona-conditioning contribution we add a control that removes the token-history difference by construction. We run two K/V caches in lockstep over a single shared token sequence—identical tokens at every generation step—differing only in the persona system prompt (source vs. target). At each step we overwrite the source-prompt run’s K/V (over a layer band L) with the target-prompt run’s K/V computed on the same token, then read the L28 hidden state from the manipulated run (the read hook is overwritten per forward pass, so the target-prompt forward is executed first and the manipulated source-prompt forward last, ensuring the recorded hidden state reflects the manipulated run). Because both runs decode the identical realized token sequence, any residual target-ward shift cannot arise from the target having generated a different token history. We run four bands—early (L1–8), mid (L9–20), late (L21–31), and full (all 32 layers)—at n = 10 with matched generation parameters. We term this a same-token-sequence target-forward control; because the shared token sequence is post-treatment (§6), it isolates whether the intervention still moves the internal state given the same realized tokens, rather than constituting a fully independent persona counterfactual.

## 3.3 K/V dump pipeline

We instrument each generation pass to record the last-position K and V tensors at every layer and step, $K _ { t , l } , V _ { t , l } \in \mathbb { R } ^ { H \times d }$ with $H = 8 ~ \mathrm { K } / \mathrm { V }$ heads and $d = 1 2 8$ . Full head dimension is preserved without averaging. Per-step logits are recorded at fp16 for downstream analyses. For each configuration we additionally record two paired reference K/V trajectories: a source-baseline run (no manipulation) and a target-baseline run. K/V dumps are taken on the first sample only of each configuration’s 10-sample cell (storage cost); this n=1 sampling is a noted limitation (§6).

## 3.4 Analysis pipeline

Text-density measurement. We hand-curate two persona-marker phrase sets (19 target markers including priya, bootcamp, um, uh, oh no, i’m so sorry; 15 source markers including patient, antibiotics, dosage, 30-year, overfed; full lists in Appendix C). Marker density is reported as substring matches per 100 words of generated text, averaged across 10 samples per configuration. A word-boundary matcher with nestedmarker deduplication leaves the cross-condition ordering and the dissociation pattern unchanged (Appendix D); it slightly lowers the mid-band rate (nested sorry/i’m so sorry) and reduces the small apparent transfer in the early and positionperturbing conditions to zero. We report type-token ratio (TTR) not as a holistic quality metric, but as an operational detector of repetition-collapse under aggressive cache perturbation (e.g., the “three months, three months, three months” pattern at v1\_full). Two source-marker phrases (starter, dead) overlap with the seed user message vocabulary; we therefore treat source-marker density as a topic-adherence (on-topic fluency) signal and anchor persona-identity claims on target-marker density (see also §6).

Hidden-state V<sup>⋆</sup>-projection. For each generated sample we project the L28 residual-stream activation to $\mathrm { V } ^ { \star } .$ -space:

$$
\tilde { h } = V ^ { \star } \cdot ( h - \mu ) / \sigma ,\tag{2}
$$

where $V ^ { \star } \in \mathbb { R } ^ { 1 0 \times 4 0 9 6 }$ is the supervised subspace basis and $\mu , \sigma$ are standardization parameters. We

compute the per-axis-rescaled $\mathrm { V } ^ { \star }$ -distance gap between sample and source/target centroids:

$$
\Delta _ { \mathrm { V } ^ { * } - \mathrm { g a p } } = \lVert \tilde { h } - c _ { \mathrm { s r c } } \rVert _ { \sigma _ { V ^ { \star } } } - \lVert \tilde { h } - c _ { \mathrm { t g t } } \rVert _ { \sigma _ { V ^ { \star } } } ,\tag{3}
$$

with $\| x \| _ { \sigma } ~ = ~ \sqrt { \sum _ { i } ( x _ { i } / \sigma _ { i } ) ^ { 2 } }$ and $\sigma _ { V }$ ⋆ the peraxis standard deviation across the 30 persona centroids. Per-axis rescaling corrects for the $3 . 2 \times \mathrm { V } ^ { \star }$ anisotropy; raw Euclidean $\mathrm { V } ^ { \star } .$ -gap (transparency check) is in Appendix B.

K/V cosine alignment. For each manipulated run and each paired reference, we compute per-(layer, step, head) cosine similarity between the manipulated K (resp. V) tensor and the reference K (resp. V) tensor:

$$
\mathrm { c o s } _ { l , t , h } \mathrm { ( r u n , r e f ) } = \frac { \langle X _ { l , t , h } ^ { \mathrm { r u n } } , X _ { l , t , h } ^ { \mathrm { r e f } } \rangle } { \| X _ { l , t , h } ^ { \mathrm { r u n } } \| \| X _ { l , t , h } ^ { \mathrm { r e f } } \| + \epsilon } ,\tag{4}
$$

for $X \in \{ K , V \}$ , with $\epsilon \ = \ 1 0 ^ { - 8 }$ . The K/V cosine gap is ga $\mathrm { ~ p } ^ { X } = \cos ^ { X } ( \mathrm { r u n } , \mathrm { t a r g e t } ) -$ $\cos ^ { X } ( \mathrm { r u n } , \mathrm { s o u r c e } )$ . We aggregate by mean over step and head for layer-level summaries.

Generation parameters. Persona pair fixed at health\_nurse\_dry → tech\_novice\_anxious; seed topic fixed at casual\_hobby; $n = 1 0$ samples per cell; temperature = 0.7; max\_new\_tokens $= 5 0 ;$ no activation-level steering applied. Code, configurations, and analysis scripts will be released at [anonymized].

## 4 Results

## 4.1 Reset dynamics motivate the K/V investigation

We first verify in our setup that single-layer activation-channel injection at L24/L28 does not produce persistent behavioral shift under autoregressive inference. Across 20 closed-loop runs (4 controller cells $\times n = 5$ samples; full methodology in Appendix $\mathbf { A . l } )$ , we measure the per-step reset ratio $R _ { t } = \| \Delta _ { \mathrm { m o d e l } } \| / \| \Delta _ { \mathrm { f o r c e d } } \|$ between the model’s restoring response over the next forward pass and the controller’s injection at the hook layer, together with the per-step alignment cos $( \Delta _ { \mathrm { m o d e l } } , e _ { t } )$ with the target direction.

Under the magnitude-adaptive controller (A) at L28 and L24, the per-step reset ratio is $R \in$ [0.98, 1.00] across the three (layer, $\alpha _ { \mathrm { m a x } } )$ cells with sample-level range [0.98, 1.02] across 15 samples; the model’s per-step displacement is anti-aligned with the target direction at cos $\in [ - 0 . 9 3 , - 0 . 9 0 ]$ on 100% of generation steps. Controller injections of 17.8, 21.8, and 35.5 V<sup>⋆</sup>-units per step are matched in magnitude and approximately opposite in $\mathbf { V } ^ { \star }$ -direction by the model’s intrinsic dynamics within the same step, and do not accumulate into a persistent persona shift. The linear-with-clip controller (B) at $\alpha _ { \mathrm { m a x } } = 4$ produces weaker perstep magnitude $( \lVert \Delta _ { \mathrm { f o r c e d } } \rVert = 4 . 0 )$ and shares the qualitative anti-alignment finding (84% pull-away). Removing token-channel context (window = 0) preserves $R \approx 1$ but collapses coherent generation entirely (Appendix A.3), separating per-step controller authority from text-level coherence.

<table><tr><td>(l, Ctrl)</td><td>αmax</td><td> $\| \Delta _ { \mathrm { f o r c e d } } \|$ </td><td>R</td><td> $\cos ( \Delta _ { \mathrm { m o d e l } } , e _ { t } )$ </td></tr><tr><td>L28, A</td><td>3</td><td> $1 7 . 8 \pm 0 . 9$ </td><td> $1 . 0 0 \pm 0 . 0 1$ </td><td> $- 0 . 9 0 \pm 0 . 0 1$ </td></tr><tr><td>L28, A</td><td>8</td><td> $3 5 . 5 \pm 2 . 6$ </td><td> $0 . 9 8 \pm 0 . 0 0$ </td><td> $- 0 . 9 3 \pm 0 . 0 0$ </td></tr><tr><td>L24, A</td><td>8</td><td> $2 1 . 8 \pm 2 . 3$ </td><td> $0 . 9 9 \pm 0 . 0 1$ </td><td> $- 0 . 9 2 \pm 0 . 0 0$ </td></tr><tr><td>L28, B</td><td>4</td><td> $4 . 0 \pm 0 . 0$ </td><td> $1 . 6 4 \pm 0 . 0 9$ </td><td> $- 0 . 4 0 \pm 0 . 0 5$ </td></tr></table>

Table 1: Closed-loop $\mathrm { V } ^ { \star }$ state-feedback: per-step force decomposition. Values are mean ± s.d. across $n = 5$ samples per cell. R is the median per-step $R _ { t }$ within each sample. The fraction of steps where cos $< 0$ is 100% for the first three rows and 84% for the controller-B row. Per-step temporal stability and additional controller details are in Appendix A.

We refer to this regularity as reset dynamics: under autoregressive inference, single-layer activation injection is restored at the per-step time-scale rather than accumulating into a persistent state shift. Our measurements are at L24 and L28, both in the late half of the 32-layer model; whether $R \approx 1$ holds at early or mid hook layers is an open question. The K/V cache, by architectural design, is the channel through which past-step computation does persist into future steps; we now investigate K/V interventions as an alternative control surface.

## 4.2 Text-level marker density and lexical diversity

Figure 1 shows target and source persona-marker density per 100 generated words across the four primary K/V intervention categories. Mid-layer replacement (v2\_L9–20) is the only configuration that produces simultaneous high target marker density (15.5) and a preserved lexical-diversity profile (TTR 0.77). Full replacement (v1\_full) achieves higher target marker density (24.8) but with higher lexical repetition (TTR 0.65): manual inspection reveals repeated phrases such as “three months, three months, three months” and “oh, oh, $\operatorname { o h } ^ { \prime \prime }$ indicating output collapse despite successful target-direction representation shift. Late-layer replacement (v2\_L21–31) achieves substantial target marker density (10.5) but shows the lowest lexical diversity in the battery (TTR 0.39). Early-layer replacement (v2\_L1–8) and the position-perturbing $\mathbf { v } 3$ conditions all produce no target markers; generated text remains fluent and on-topic for the seed prompt, with topical-vocabulary density $\geq 8$ per 100 words, consistent with the un-intervened source-prompt regime.

![](images/c2d73e7fb98ca40b6f2106ab021924c34b76d74451a34ea39037d44f36e1e166.jpg)  
Figure 1: Persona-marker density (top) and lexical diversity / TTR (bottom) by K/V intervention layer band. Mid-layer (L9–20, shaded) is the unique band combining target marker density with high TTR. Anchor lines: full replacement (v1, dashed); position perturbations (v3 shuffle and lag, dotted).

## 4.3 Hidden-state V<sup>⋆</sup>-projection confirms mid-layer concentration

Figure 2 shows the per-axis-rescaled V<sup>⋆</sup>-gap for the six conditions on which we compute the L28 hidden-state projection (the three high-Valignment conditions—full, mid-layer, and latelayer replacement—together with early-layer replacement and the two position perturbations); the full 13-condition battery is characterized through text-marker density and K/V cosine (Table 8). Across these six conditions, the sign of the $\mathrm { V } ^ { \star } .$ -gap agrees with the text-gap, indicating that K/V interventions induce consistent shifts in both hiddenstate $\mathrm { V } ^ { \star }$ -projection and text persona-marker density. Three conditions reach comparably high $\mathrm { V } ^ { \star } .$ alignment—full replacement (v1\_full, $+ 3 . 9 6 \pm$ 0.29), mid-layer $( \mathrm { v } 2 \mathrm { \_ L } 9 - 2 0 , + 3 . 8 3 \pm 0 . 3 8 )$ , and late-layer $( \mathrm { v } 2 \mathrm { \_ L } 2 1 - 3 1 , + 4 . 0 9 \pm 0 . 2 0 )$ , all within $\sim ~ 0 . 3$ rescaled units of one another (per-axisrescaled units, mean $\pm \ : \mathrm { s . e . , } \ : n = 1 0 )$ —yet these same three conditions span the full range of lexicaldiversity values (TTR 0.65, 0.77, and 0.39 respectively). Comparable hidden-state persona alignment thus does not predict the lexical-diversity profile of persona expression. Early-layer replacement $( \mathrm { v } 2 \mathrm { \_ L } 1 \mathrm { - } 8 , - 1 . 3 8 \pm 0 . 3 6 )$ and the v3 position perturbations (shuffle $- 1 . 3 3 \pm 0 . 6 9$ lag= 1 $- 1 . 5 2 \pm 0 . 3 8 )$ show negative $\mathrm { V } ^ { \star } { \ - } \tt g a p .$ , consistent with their failure to inject target-direction representation.

![](images/1e0937f780c3b075959c6160619f17c3247dc4f14ee5998939c40f1914a2c094.jpg)  
Figure 2: L28 hidden-state V<sup>⋆</sup>-projection: per-axisrescaled gap $\lVert \tilde { h } - c _ { \mathrm { s r c } } \rVert _ { \sigma } - \lVert \tilde { h } - c _ { \mathrm { t g t } } \rVert _ { \sigma }$ (mean ± s.e. across 10 samples). Full replacement (v1\_full), midlayer $( \mathrm { v } 2 \_ \mathrm { L } 9 \ – 2 0 )$ , and late-layer (v2\_L21–31) reach comparable persona-aligned displacement yet differ sharply in lexical diversity.

## 4.4 Same-token-sequence control: the shift is not solely token history

The V<sup>⋆</sup>-gap above is measured on interventions (v1/v2) that transplant a target-conditioned trajectory, which carries the target’s own generated token history; the resulting shift therefore admits a tokenhistory explanation. The same-token-sequence target-forward control (§3.2) removes that difference by construction: source and target caches decode the identical realized token sequence, differing only in persona conditioning. We compute the same per-axis-rescaled V<sup>⋆</sup>-gap on this control (identical metric, references, and L28 read-out as Fig. 2), so the numbers are directly comparable to the main battery.

Table 2 reports the result. The control reproduces both the sign and the layer localization of the main $\mathrm { V } ^ { \star } { \tt g a p } \colon$ early-layer replacement remains on the source side (−1.09, cf. main −1.38), while mid-, late-, and full replacement move the L28 state to the target side. The mid-layer band is essentially unchanged from the main battery (+3.60 vs. +3.83); late and full are attenuated but still clearly target-ward (+2.42, +2.83). Because the target and source runs share an identical token sequence here, this target-ward representational shift cannot be explained solely by importing target-generated token history. We state the claim as this exclusion (“not solely token history”), not as removal of all token-history-related factors, since the shared sequence is itself post-treatment (§6).

<table><tr><td>Band</td><td colspan="2">V*-gap</td><td colspan="2">control readouts</td></tr><tr><td></td><td>main</td><td>control</td><td>tgt-mkr</td><td>TTR</td></tr><tr><td>L1-8</td><td>-1.38</td><td>-1.09</td><td>0.0</td><td>0.82</td></tr><tr><td>L9-20</td><td>+3.83</td><td>+3.60</td><td>10.7</td><td>0.60</td></tr><tr><td>L21-31</td><td>+4.09</td><td>+2.42</td><td>9.2</td><td>0.38</td></tr><tr><td>full</td><td>+3.96</td><td>+2.83</td><td>3.3</td><td>0.57</td></tr></table>

Table 2: Same-token-sequence target-forward control $( n = 1 0 )$ “V<sup>⋆</sup>-gap main” = target-conditioned battery (Fig. 2); “control” = same per-axis-rescaled V<sup>⋆</sup>- gap computed on the shared-token control. The control reproduces the sign and layer localization (early source-side, mid/late/full target-side). “tgt-mkr” = target-marker density (per 100 words, word-boundary matcher; a surface-realization diagnostic, not an identity measure); TTR = repetition-sensitive lexical diversity. Mid-layer (L9–20) combines a target-ward representation with the highest-integrity surface readout (+3.60, TTR 0.60); late/full retain a target-ward representation while their generations degrade (TTR 0.38, 0.57).

The control also echoes, within a single design, the broader representation–behavior dissociation of §4.6: a target-ward L28 representation coexists with a degraded surface form in the late and full conditions, while only the mid-layer band retains both. Mid-layer replacement pairs a target-ward representation (+3.60) with the most preserved surface form (TTR 0.60, though below the ∼0.77 of an unconstrained target rollout), whereas late and full replacement hold a target-ward representation (+2.42, +2.83) while their generations collapse into repetition (TTR 0.38, 0.57). Representation alignment can thus persist even where surface realization has degraded; we therefore anchor this control on the representation axis and read the marker/TTR columns as surface diagnostics rather than persona-identity claims.

## 4.5 K/V cosine analysis

Figure 3 shows the layer profile of V-space and K-space cosine gap, defined as cos(run, target) − cos(run, source).

K/V alignment exhibits strong layer-band locality. Replacing K/V representations at a given layer band produces localized changes in V-space alignment within the same band, while leaving nonintervened layers largely unchanged. Table 3 reports the V-space cosine gap measured at each of the three coarse bands for each layer-band intervention. The diagonal entries (intervention band

![](images/02aa3cc53a77e16b99ad8206c60150532459d5f921c83df0f1f8d408864610a6.jpg)

Figure 3: K/V cosine gap by layer (mean over generation step and K/V head, 13 intervention configurations). Top: V-space cosine gap (primary). Bottom: K-space cosine gap (supporting). Mid-layer band (L9–20) shaded. K and V show the same qualitative direction across runs; V exhibits a substantially larger cosine-gap magnitude than K in this battery.
<table><tr><td>Condition</td><td>L1-8</td><td>L9-20</td><td>L21-31</td></tr><tr><td>v1_full</td><td>0.91</td><td>0.94</td><td>0.87</td></tr><tr><td>v2_L1-8</td><td>0.91</td><td>0.02</td><td>-0.10</td></tr><tr><td>v2_L9-20</td><td>0.12</td><td>0.89</td><td>0.13</td></tr><tr><td>v2_L21-31</td><td>0.02</td><td>0.04</td><td>0.84</td></tr></table>

Table 3: V-space cosine gap by measurement band × intervention condition (mean over generation steps and K/V heads, n = 1 K/V dump per condition). Bold entries: intervened band. K/V intervention at a band produces large V-alignment at that band but ≤ 0.14 Vgap elsewhere. K-space analog in Appendix G.

= measurement band) are large (0.91, 0.89, 0.84); the off-diagonal entries are substantially smaller $( | V \mathrm { - g a p } | \leq 0 . 1 4 , \mathrm { v s . } \geq 0 . 8 4 $ on the diagonal). We do not observe measurable cross-band propagation of V-alignment from an intervened band to non-intervened bands under our cosine diagnostic.

This locality is the central empirical fact underlying the spatial-band dissociation. Crucially, the fact that all three v2 bands produce comparable within-band V-alignment (0.84–0.91) and yet only the mid-layer band combines target-marker transfer with a preserved lexical-diversity profile indicates that within-band V-alignment alone does not determine behavioral outcome. Specifically: midlayer replacement (v2\_L9–20) yields target marker density 15.5 with TTR 0.77; early replacement produces no target markers (density 0.3); late replacement shifts marker density (10.5) but at low lexical diversity (TTR 0.39). Within the L9–20 band, finergrained 4-layer sub-band interventions produce uneven target-marker density (Appendix G: L9–12 → 0.51, L13–16 → 5.61, L17–20 → 6.22), suggesting the productive subspace may be narrower than the coarse L9–20 band; we use the coarse band as the unit of analysis throughout because it is the a-priori scope.

Representation alignment does not uniquely determine downstream behavioral form. Although v1\_full and v2\_L9–20 both induce high V-space alignment within intervened layers (0.94 vs. 0.89, by construction), they differ substantially in output behavior. v1\_full produces higher target marker density (24.8) but higher lexical repetition (TTR = 0.65), while v2\_L9–20 yields lower marker density (15.5) but higher lexical diversity (TTR = 0.77). Similar levels of representation alignment correspond to different behavioral outcomes under different intervention structures.

V-gap exceeds K-gap in magnitude. Across runs and layers, V-gap exhibits substantially larger magnitudes than K-gap (e.g., 0.89 vs. 0.38 at L9– 20 for v2\_L9–20). We report this descriptively; isolating K-only vs. V-only causal contributions would require dedicated ablation experiments and is left to future work (see §6).

Partial interpolation shows a graded, monotonic response over the tested strengths. Interpolating K/V representations between source and target at L9–20 yields intermediate behavior. At α = 0.25, V-gap is 0.21 with low target marker density (0.6); at $\alpha = 0 . 5 0$ , V-gap increases to 0.65 with marker density 3.2. The response is monotonic over the two tested strengths; whether it is linear, thresholded, or otherwise nonlinear is unresolved.

Position perturbations suppress target transfer despite distinct operations. Shuffle and lag apply qualitatively different operations to the sourceprompt K/V cache (random permutation vs. cyclic shift of sequence positions), but neither produces target-aligned K/V: both show negative or nearzero V-gap at all layer bands (Table 8), and both yield minimal target-persona expression in generated text (target marker density ≤ 0.5). At the single-sample resolution of our K/V dumps the cosine signatures of lag and shuffle are not reliably distinguishable; the robust finding is behavioral— distinct positional perturbations produce the same failure to transfer the target persona.

## 4.6 Summary of representation–behavior dissociations

Two dissociation patterns between representationlevel alignment and behavioral expression are apparent in the battery, plus a common failure under position-perturbing interventions. Each is supported by at least two of the three measurement families (text marker density, V<sup>⋆</sup>-projection, K/V cosine): no single measurement channel is loadbearing for the behavioral claims in this section.

1. Spatial-band dissociation. Layer-band interventions all achieve strong local V-alignment (0.91, 0.89, 0.84 for L1–8, L9–20, L21–31 respectively), but only the mid-layer band combines target-marker transfer with a preserved lexical-diversity profile.

2. Intervention-scope dissociation. Within an overlapping comparison set, full replacement (v1) and mid-layer replacement (v2\_L9– 20) achieve comparable V-alignment (0.94 vs. 0.89) but produce qualitatively different lexical-diversity profiles (TTR 0.65 vs. 0.77). Together with the spatial-band dissociation, this indicates that within-band V-alignment alone does not determine behavioral form.

3. Common position-perturbation failure. Position-perturbing interventions (shuffle and lag) apply distinct operations to the source K/V cache but both fail to inject target K/V representation and both yield similarly weak target-persona text expression. Here representation and behavior agree—neither shows target transfer—so this is a common failure rather than a representation-behavior dissociation in the sense of points 1–2.

A scatter of hidden-V<sup>⋆</sup>-gap vs. text-gap across all six measured conditions (Appendix E) shows sign agreement throughout while magnitude relationships exhibit the patterns above.

## 5 Discussion

Our claims are intentionally intra-battery rather than population-level: the core empirical result is that materially different behavioral outcomes emerge within a fixed intervention family despite comparable representation-level alignment. The findings below should be read against this scope.

## 5.1 Representation alignment vs. behavioral expression

Across both hidden-state projections $( \mathrm { V } ^ { \star } )$ and K/V representations, we observe consistent dissociations between representational alignment and behavioral expression. Interventions that induce similar levels of representation alignment (v1\_full and v2\_L9–20) produce different behavioral outcomes in lexical diversity and persona-marker expression; qualitatively different perturbations in representation space (lag vs. shuffle) result in similarly weak behavioral transfer. Representation-level similarity metrics alone are not sufficient predictors of downstream persona expression in the intervention regimes studied here.

## 5.2 Why mid-layer replacement preserves lexical diversity

A common pattern across our results is that K/V interventions must produce a cached state that is consistent with what downstream computation expects given the current token’s residual stream. Full K/V replacement violates this: the cached attention history is target-derived while the current-position residual is computed from source-prompt logits flowing through target K/V, producing higher lexical repetition (TTR 0.65) despite strong representational alignment with target. Selective mid-layer K/V replacement preserves lexical diversity (TTR 0.77) and yields target-aligned behavior. K/V interventions at early or late layer bands achieve strong within-band V-alignment but produce inert or degraded text, despite the intervention not showing measurable cross-band propagation (Table 3).

Concretely, the last-step L28 $\mathbf { V } ^ { \star } .$ -projection at $\mathrm { v 1 \_ f u l l \ ( + 3 . 9 6 \pm 0 . 2 9 }$ rescaled units) is high— comparable to mid-layer replacement $( + 3 . 8 3 \pm$ 0.38) and late-layer replacement $( + 4 . 0 9 \pm 0 . 2 0 ) $ yet v1\_full’s text exhibits high token-level repetition (e.g., “three months, three months, three months”, “oh, oh, oh”). Even in this collapsed fulltransplant condition the final-layer $\mathbf { V } ^ { \star }$ -projection is strongly target-aligned, so the repetition cannot be attributed simply to a failure to move the final residual toward the target; the mechanism remains unresolved. One possible account is that retaining source-derived K/V in the non-intervened early and late bands constrains the trajectory to a less repetition-prone regime, but our experiments do not distinguish this from other mechanisms. We characterize this pattern as an empirical regularity across the intervention families tested.

## 5.3 Implications for persona control methods

Persona control methods can be organized by the channel they intervene on: the activation channel (single-layer steering), the token channel (prompts, soft prompts, prefix tuning), the weight channel (fine-tuning, LoRA, RL-based optimization), and— as we study here—the K/V cache. Reset dynamics observed in our experiments (Appendix A) impose a per-step constraint on the single-layer activation interventions we test at L24 and L28. Token- and weight-channel methods are not subject to this constraint: token-based approaches influence generation through the autoregressive context which accumulates over time, while weight-based methods modify the transformation applied at every step. K/V interventions persist across steps by architectural design; because a transplanted trajectory carries the target’s own generated token history, however, they are not cleanly separable from the token channel (though a same-token-sequence control, §4.4, shows the representational shift persists when token history is held fixed, so the entanglement is partial rather than total), and they exhibit their own structural constraints, as documented in this paper.

More broadly, our findings argue that persona-control evaluations should report both representation-level and behavioral metrics: as the contrast between full and mid-layer K/V replacement shows, similar V-alignment can correspond to substantially different output quality.

## 6 Conclusion

We document two dissociations between representation-level alignment and behavioral expression—spatial (across layer bands) and scope (full vs. mid-layer replacement)—and a common failure under position-perturbing interventions (lag and shuffle, which transfer neither representation nor behavior) on K/V-cache interventions in a decoder-only language model. Mid-layer K/V replacement (layers 9–20) is the unique configuration that produces both target-aligned representations and target-marker transfer with a preserved lexical-diversity profile; full replacement achieves higher representational alignment with higher lexical repetition. Because the intervention transplants a target-conditioned trajectory that carries the target’s own generated token history, we characterize the K/V cache as a controllable but structurally constrained intervention surface rather than a channel cleanly separable from token- and weight-channel methods; a same-token-sequence control (§4.4) nonetheless shows the representational component of the effect survives when token history is held fixed, so this entanglement is partial rather than total.

## Limitations

Single model, single persona pair. All K/V battery findings are from Llama-3.1-8B with a single source → target persona pair (health\_nurse\_dry → tech\_novice\_anxious). Generalization to other model families, scales, and persona pairs is an empirical question we do not address. The persona pair we study is far apart in V<sup>⋆</sup>-space $( \lVert c _ { \mathrm { t g t } } - c _ { \mathrm { s r c } } \rVert = 7 . 2 $ in rescaled units, vs. a typical inter-persona distance of ∼ 3 across the corpus); behavior at smaller separations is not characterized.

Single seed topic. All 13 intervention configurations use the same seed user message (a question about sourdough starters). Topic-conditional effects on the K/V signature are not characterized. The persona-marker vocabulary we use for textlevel density is partially specific to this topic; a more general persona-quality measure (e.g., using a held-out classifier) would be more robust. Two source-marker terms (starter, dead) overlap with the seed message vocabulary, which inflates sourcemarker density for any coherent generation on the seed topic. Target markers (Priya, um, bootcamp, oh no) do not have this confound, so target-marker density is the load-bearing signal in our analyses.

Generation horizon and the activation/K-V asymmetry. K/V battery experiments are conducted at max\_new\_tokens = 50; the closed-loop activation-channel experiment in Appendix A runs up to $T = 1 5 0$ . This is an asymmetry between the two experimental tracks: at $T = 5 0$ our K/V experiments measure intervention instantiation (whether the K/V manipulation initially produces targetpersona output) rather than long-horizon persistence. We provide a small-sample (n = 3) persistence check at T = 150 in Appendix H: targetpersona marker density persists across 150 tokens— closely tracking an un-intervened target baseline— with a roughly constant lexical-repetition cost rather than progressive degradation. A full-battery long-horizon characterization remains future work.

Multi-layer activation steering not directly tested. The reset-dynamics result we report in Appendix A is from single-layer interventions (L24, L28). Coordinated multi-layer activation steering is likely subject to similar constraints under our taxonomy but we do not directly measure R for multi-layer hooks.

Causal versus correlational analysis. Our K/V cosine measurements are correlational: we modify K/V representations and observe changes in text behavior, but we do not isolate which substructures of K/V (specific heads, specific positions, K versus V independently) carry the causal signal. The “V is a stronger structural correlate than $\mathbf { K } ^ { \ast }$ observation is consistent with prior interpretive accounts of attention but is not, in our experiments, a causal claim. K-only or V-only ablation experiments would upgrade this to a causal characterization.

Intrinsic factorization vs. intervention locality. The observation that K/V intervention effects show no measurable cross-band propagation under our cosine diagnostic is consistent with two readings: an intrinsic factorization of K/V representations, or specificity to intervention dynamics under the autoregressive inference loop. Cross-layer K/V similarity in unperturbed generation, with crosslayer probing diagnostics, could disambiguate.

Single-sample K/V cosine. The K/V cosine analysis uses K/V tensors recorded from the first sample of each 10-sample cell (storage cost), so per-sample variance is not reported. We note, however, that for the layer-band replacement conditions the withinband cos(run, target) is 1.0 by construction—the run’s K/V at the replaced band is, by substitution, the target reference’s—so the headline within-band V-gaps (≥ 0.84) are fixed by the replacement structure and the source-similarity term rather than by single-sample sampling noise. The off-band “nopropagation” entries $( \leq 0 . 1 4 )$ and the v3/v4 perturbation cosines are genuine single-sample measurements; the order-of-magnitude cross-condition separation we rely on $( \ge 0 . 8 4 $ within-band $\mathrm { v s . } \leq 0 . 1 4$ off-band) is large, though we do not estimate offband single-condition sampling variance directly. Direct estimation of off-band and perturbationcondition variance is left to future work.

Lexical-diversity proxy. Type-token ratio (TTR) is our only surface-form diversity measurement, and several headline contrasts (notably v1\_full vs. v2\_L9–20 at TTR 0.65 vs. 0.77) rely on it. We use TTR as a repetition-sensitive lexical-diversity proxy (e.g., it flags v1\_full’s “three months, three months, three months” pattern); it does not measure semantic coherence, grammaticality, or persona quality, and should not be read as a coherence metric. Per-step perplexity, LLM-as-judge persona attribution, or n-gram diversity would provide complementary signal; we recommend multi-metric reporting in follow-up work.

Behavioral readout and the scope of the personacontrol claim. Our behavioral axis is operationalized as target-marker density (with TTR for repetition), not as human persona attribution. Because the target persona’s markers are largely disfluency and hedging tokens (um, oh no, i’m so sorry, mid-sentence self-correction), marker density does not separate persona identity from surface disflu ency/degradation: a more degraded generation can score as more “target” for reasons unrelated to persona recognition. We therefore report the representation-level results (V<sup>⋆</sup> alignment, layer localization) as the load-bearing quantitative findings and treat “behavioral expression” here as a marker proxy rather than an identity judgment; we do not claim behavioral persona control. Establishing that claim would require a behavioral readout with independence from surface realization—a surface-invariance test (the same identity across varied surface realizations yields stable attribution while a length/verbosity-matched foil varies), and task-grounded behavioral manifestations scored by a preregistered rubric, with at least two manifestations that share no surface cue, format, or lexical marker and that co-move under intervention. A preregistered attempt at a fluent, structurally distinctive target (introduced specifically to reduce the disfluency confound) did not clear a preannotation construct-separation audit: successive stimulus contrasts remained recoverable from simple surface features (length, and—for a verbositymatched foil—enumeration markers introduced by the foil’s own construction), so we stopped before annotation rather than treat the resulting labels as construct-valid. These audits bound the stimulus, not human judgment, and do not imply that persona behavior is intrinsically unmeasurable; designing a surface-invariant, task-grounded behavioral instrument is left to future work.

Scope of the same-token-sequence control. The target-forward control (§4.4) holds the token sequence fixed across the source and target runs, but that shared sequence is itself post-treatment—it is one realized rollout, not an exogenously fixed stimulus—so the control isolates whether the intervention still moves the internal state given the same realized tokens rather than a fully independent persona counterfactual. It therefore supports the specific exclusion that the representational shift is “not solely imported token history,” not a claim that all token-history-related factors are removed. Teacher forcing also lowers surface fluency relative to an unconstrained rollout (mid-layer TTR 0.60 vs. ∼ 0.77), so we read its marker/TTR columns as surface diagnostics and anchor the control on the representation axis. The control uses the same single model, persona pair, and seed topic as the main battery, and its V<sup>⋆</sup>-gap is computed from the same L28 read-out; the caveats above on generalization and on marker density as a surface (not identity) signal apply equally here.

Open-loop pilot caveats (Appendix A.4). The open-loop activation-injection comparison uses greedy rather than temperature-0.7 stochastic decoding, records n = 1 per cell, and preserves only the first 120 characters of each generation (HDFS sync failure). The three-regime pattern (source / target / collapse) replicates across three target personas, but the precise α boundaries are not resolved at the coarse α ∈ {0, 1, 2, 4, 8} sweep.

## Ethical Considerations

The K/V-intervention battery we describe is in principle dual-use: the same mechanism that produces target-persona transfer in mid-layer replacement could be used to evade refusal training or to inject undesired behavior. We mitigate this concern by noting that (i) all our experiments are on a generic persona pair with no harmful content, (ii) effective use of our methods requires white-box access to model weights and modification of internal cache states, raising the threat-model bar substantially relative to prompt-based attacks, and (iii) the constraints we identify on activation-channel state-feedback control may, conversely, improve the safety profile of deployed LLM systems by preventing over-reliance on single-layer activation steering for safety-critical persona maintenance.

Our representation–behavior dissociation result has a methodological consequence beyond our specific setup: any evaluation pipeline using representation-space proxies (probe accuracy, hidden-state distance to a target, cosine to a learned direction) as primary success metrics is potentially subject to dissociation artifacts. We recommend that downstream evaluations report both internal and behavioral metrics.

## References

Anthony Baez, Sheer Karny, and Pat Pataranutaporn. 2026. Dissociating the internal representations of sycophancy in llms. arXiv preprint arXiv:2607.07003.

Nora Belrose, Zach Furman, Logan Smith, Danny Halawi, Igor Ostrovsky, Lev McKinney, Stella Biderman, and Jacob Steinhardt. 2023. Eliciting latent predictions from transformers with the tuned lens. arXiv preprint arXiv:2303.08112.

Runjin Chen, Andy Arditi, Henry Sleight, Owain Evans, and Jack Lindsey. 2025. Persona vectors: Monitoring and controlling character traits in language models. arXiv preprint arXiv:2507.21509.

Enyi Jiang, Anders Gjølbye, Yibo Jacky Zhang, and Sanmi Koyejo. 2026. When behavioral safety evaluation fails: A representation-level perspective. arXiv preprint arXiv:2606.08044.

Diancheng Kang, Zheyuan Liu, Ningshan Ma, Yue Huang, Zhaoxuan Tan, and Meng Jiang. 2026. Prompt-activation duality: Improving activation steering via attention-level interventions. arXiv preprint arXiv:2605.10664.

Brian Lester, Rami Al-Rfou, and Noah Constant. 2021. The power of scale for parameter-efficient prompt tuning. Proceedings ofEMNLP.

Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, and Martin Wattenberg. 2023. Inference-time intervention: Eliciting truthful answers from a language model. In Advances in Neural Information Processing Systems (NeurIPS).

Xiang Lisa Li and Percy Liang. 2021. Prefix-tuning: Optimizing continuous prompts for generation. Proceedings of ACL.

Llama Team, AI @ Meta. 2024. The llama 3 herd of models. ArXiv:2407.21783; we use the Llama-3.1- 8B-Instruct release.

Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. 2022. Locating and editing factual associations in GPT. In Advances in Neural Information Processing Systems (NeurIPS).

Kevin Meng, Arnab Sen Sharma, Alex Andonian, Yonatan Belinkov, and David Bau. 2023. Massediting memory in a transformer. In International Conference on Learning Representations (ICLR).

Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Scott Johnston, Andy Jones, Jackson Kernion, Liane Lovitt, and 7 others. 2022. Incontext learning and induction heads. Anthropic Technical Report. ArXiv:2209.11895.

Nina Panickssery, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, and Alexander Matt Turner. 2024. Steering llama 2 via contrastive activation addition. arXiv preprint arXiv:2312.06681.

Eric Todd, Millicent L. Li, Arnab Sen Sharma, Aaron Mueller, Byron C. Wallace, and David Bau. 2023. Function vectors in large language models. arXiv preprint arXiv:2310.15213.

Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J. Vazquez, Ulisse Mini, and Monte MacDiarmid. 2023. Activation addition: Steering language models without optimization. arXiv preprint arXiv:2308.10248.

Ailing Zeng, Casper Yang, Chauncey Ge, Eddie Zhang, Garvey Xu, Gavin Lin, Gilbert Gu, Jeremy Pi, Leo Li, Mingyi Shi, Shawn Wang, Sheng Bi, Steven Tang, Thorn Hang, Tobey Guo, Vincent Li, Xin Tong, Yikang Li, Yuchen Sun, and 6 others. 2026. LPM 1.0: Video-based character performance model. arXiv preprint arXiv:2604.07823.

Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Ré, Clark Barrett, Zhangyang Wang, and Beidi Chen. 2023. H2O: Heavy-hitter oracle for efficient generative inference of large language models. NeurIPS.

Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, Shashwat Goel, Nathaniel Li, Michael J. Byun, Zifan Wang, Alex Mallen, Steven Basart, Sanmi Koyejo, Dawn Song, Matt Fredrikson, and 2 others. 2023. Representation engineering: A top-down approach to ai transparency. arXiv preprint arXiv:2310.01405.

A Activation-channel motivation: reset dynamics, channel separation, and open-loop pilot

This appendix reports the activation-channel experiments that motivated the K/V investigation

(§4.1). The same source–target persona pair and seed message used in the K/V battery (§3.1) are used throughout.

## A.1 Closed-loop V<sup>⋆</sup> state-feedback (methodology)

At each generation step t we read the L28 residualstream activation, project to $\mathrm { V } ^ { \star } .$ -space $\begin{array} { r l } { ( \tilde { h } _ { t } } & { { } = } \end{array}$ $V ^ { \star } ( h _ { t } - \mu ) / \sigma )$ , and compute the displacement to the target persona centroid $e _ { t } = c _ { \mathrm { t g t } } - \tilde { h } _ { t }$ . We then inject an activation-channel correction at the headline layer $\ell \in \{ 2 4 , 2 8 \}$

$$
h _ { t , \ell } ^ { \mathrm { p o s t } } = h _ { t , \ell } ^ { \mathrm { p r e } } + \alpha _ { t } \cdot V ^ { \star \top } P _ { t } e _ { t } ,\tag{5}
$$

where $P _ { t }$ is a per-axis whitening normalizing the controller direction by the inter-persona standard deviation. $\alpha _ { t }$ is bounded by $\alpha _ { \mathrm { m a x } }$ and applied via a magnitude-adaptive variant $( \mathrm { A } ; \alpha _ { t }$ scaled to keep $\| \Delta _ { \mathrm { f o r c e d } } \|$ at a target fraction of the unperturbed residual norm) or a linear-with-clip variant (B: fixed gain with norm-clip safety bound). Both run in fp32. We test four cells: (A, L28, 3), (A, L28, 8), (A, L24, 8), (B, L28, 4), with $n = 5$ samples per cell at temperature 0.7, max\_new\_tokens up to 150.

Force decomposition. Let $\Delta _ { \mathrm { f o r c e d } } ~ = ~ h _ { t } ^ { \mathrm { p o s t } } -$ $h _ { t } ^ { \mathrm { p r e } }$ denote the per-step controller injection and $\bar { \Delta _ { \mathrm { m o d e l } } } ~ = ~ h _ { t + 1 } ^ { \mathrm { p r e } ^ { - } } - ~ h _ { t } ^ { \mathrm { p o s t } }$ the change introduced by the model’s own dynamics over the next forward pass. We define the per-step reset ratio $R _ { t } = \lVert \Delta _ { \mathrm { m o d e l } } \rVert / \lVert \Delta _ { \mathrm { f o r c e d } } \rVert$ and the per-step alignment ${ \cos ( \Delta _ { \mathrm { m o d e l } } , e _ { t } ) . ~ R _ { t } \approx 0 }$ would indicate the model absorbs controller injections into a persistent state shift; $R _ { t } \approx 1$ with cos $\approx - 1$ indicates the model’s intrinsic dynamics restore the unperturbed trajectory at the same magnitude as the controller forces it.

## A.2 Per-step temporal stability of reset dynamics

The cross-sample reset-dynamics summary in Table 1 (main text) is reproduced step-by-step within each sample, not driven by averaging over unusual steps. In a representative sample (L28, controller A, $\alpha _ { \mathrm { m a x } } = 8 , T = 1 5 0$ steps), the per-step ratio $R _ { t }$ ranges across [0.83, 1.27] over individual generation steps with mean 0.989 and step-level standard deviation 0.080; the per-step alignment cos $( \Delta _ { \mathrm { m o d e l } } , \Delta _ { \mathrm { f o r c e d } } )$ ranges across [−1.00, −0.81] with mean −0.98, negative on 100% of steps. We report $\cos ( \Delta _ { \mathrm { m o d e l } } , \Delta _ { \mathrm { f o r c e d } } )$ here because the controller direction is directly measurable from the recorded $\mathrm { V } ^ { \star }$ -trajectory; it is slightly more antialigned than $\cos ( \Delta _ { \mathrm { m o d e l } } , e _ { t } )$ reported in the main text because $\Delta _ { \mathrm { f o r c e d } }$ differs from $e _ { t }$ by the per-axis whitening factor $P _ { t } .$ . Step-by-step trajectories from selected samples will be released alongside code.

## A.3 Context-window ablation

To probe whether reset dynamics depend on crossstep token-channel feedback or are intrinsic to single-step computation, we run the closed-loop controller $( \mathrm { A } , \alpha _ { \mathrm { m a x } } = 8 , \ell = 2 8 )$ under three context-window settings: window = −1 (full context); window = 1 (Markov-1: model attends only to the immediately previous generated token); window = 0 (full truncation: model attends only to the system prompt).

<table><tr><td>window</td><td> $\| \Delta _ { \mathrm { f o r c e d } } \|$ </td><td>R</td><td>Generation</td></tr><tr><td> $^ { - 1 }$ </td><td> $3 5 . 5 \pm 2 . 6$ </td><td> $0 . 9 8 \pm 0 . 0 0$ </td><td>fluent; partial target shift</td></tr><tr><td>1</td><td> $4 1 . 4 \pm 1 . 5$ </td><td> $0 . 9 9 \pm 0 . 0 1$ </td><td>degraded; target markers</td></tr><tr><td>0</td><td> $6 8 . 9 \pm 0 . 0$ </td><td> $1 . 0 0 \pm 0 . 0 0$ </td><td>incoherent token soup</td></tr></table>

Table 4: Context-window ablation under magnitudeadaptive controller A, $\alpha _ { \mathrm { m a x } } = 8 , \ell = 2 8 , n = 5$ samples per window. Values are mean $\pm \ : \mathrm { s . d . }$ The $\pm 0 . 0 0$ entries at window = 0 indicate identity to floating-point precision: with no context the per-step $\mathrm { V } ^ { \star } .$ -trajectory is fully deterministic given the system prompt.

The per-step R is ≈ 1 across all three windows: the model’s restoring force tracks the controller injection with comparable per-step magnitude regardless of how much context the model can attend to. Generation quality, however, varies sharply. With full context, samples are fluent but retain dominant source-persona markers (ER, IV lines, “I’m a nurse, not a baker”) with partial convergence to the target’s anxious register. With Markov-1 context, text degrades into repetitive fragments but exposes target-persona markers from the controller’s per-step push (“I’m sorry, maybe you’ve seen $i t ^ { \prime } s$ probably the starter’s probably just. . . ”). With no context, text collapses into incoherent token soup (“YouI\*IIIMaybeIIMaybeIIII. . . ”) despite the controller still applying its largest per-step authority. This dissociation indicates that the per-step balance $R \approx 1$ is approximately invariant across contextwindow settings, but coherent natural-language generation depends on the token channel.

## A.4 Open-loop activation-injection pilot

As an auxiliary contrast to the closed-loop characterization, we run an open-loop activation-injection pilot at $\ell = 2 8$ using a fixed-magnitude per-step injection:

$$
h _ { t , 2 8 } ^ { \mathrm { p o s t } } = h _ { t , 2 8 } ^ { \mathrm { p r e } } + \alpha \cdot V ^ { \star \top } ( c _ { \mathrm { t g t } } - c _ { \mathrm { s r c } } ) ,\tag{6}
$$

sweeping $\alpha ~ \in ~ \{ 0 , 1 , 2 , 4 , 8 \}$ This pilot uses greedy decoding (vs. temperature 0.7 in main experiments), records $n = 1$ sample per cell, and preserves only the first 120 characters per generation (HDFS sync failure). Three targets tested.

<table><tr><td>Target</td><td>α=0</td><td>α=1</td><td>α=2</td><td>α=4</td></tr><tr><td>retail_csm_warm</td><td>src</td><td>src</td><td>tgt</td><td>coll</td></tr><tr><td>tech_novice_anxious</td><td>src</td><td>src</td><td>tgt</td><td>coll</td></tr><tr><td>tech_enthusiast_indie</td><td>src</td><td>src</td><td>tgt</td><td>coll</td></tr></table>

Table 5: Open-loop pilot regime classification across 3 targets × 5 α values (source / target / collapse). All from health\_nurse\_dry source. Per-step injection magnitude $\| \Delta _ { \mathrm { f o r c e d } } \| = \alpha \cdot \| c _ { \mathrm { t g t } } - c _ { \mathrm { s r c } } \|$ ≈ 11–12.5·α V<sup>⋆</sup>-units depending on target.

A consistent three-regime structure emerges. For tech\_novice\_anxious: α = 0 gives source persona (“Your starter’s probably dead. Or it’s just not getting enough food.”); α = 1 is still sourcelike with minor softening; α = 2 produces fluent target-persona text (“I’m like, totally stoked you’re like, totally stressed about your sourdough. . . ”); α = 4 degenerates into token repetition (“I totally totally totally. . . ”); α = 8 continued collapse.

Comparison with closed loop. The closed-loop controller at L28, $\alpha _ { \mathrm { m a x } } = 8$ produces $\| \Delta _ { \mathrm { f o r c e d } } \| =$ 35.5 V<sup>⋆</sup>-units per step on average—larger than the open-loop α = 2 regime (23.7) where fluent target text emerges—yet closed-loop generations retain dominant source-persona markers (§A.3). The two configurations differ in four uncontrolled respects: (i) injection direction: closed-loop uses the peraxis-whitened state-feedback error $V ^ { \star \top } P _ { t } e _ { t } ;$ openloop uses the raw centroid difference. (ii) per-step magnitude: closed-loop 35.5 vs. open-loop 23.7 V<sup>⋆</sup>-units. (iii) decoding regime: temperature 0.7 vs. greedy. (iv) observed generation length: 150 tokens vs. first 120 characters. We do not isolate which factor accounts for the divergent outcomes. The empirical content of the contrast that survives these caveats: single-layer activation injection at L28 can produce target-persona text in some accessible regime (open-loop α = 2 within the first

120 characters), while the specific $R \approx 1$ regime we instantiated does not occupy that productive window.

## B Raw V<sup>⋆</sup>-distance gap (transparency)

The hidden V<sup>⋆</sup>-projection results in §4.3 use peraxis rescaling by inter-persona standard deviation to correct for 3.2× anisotropy. Table 6 reports raw Euclidean $\mathrm { V } ^ { \star } { \tt g a p } ;$ sign agreement with the rescaled metric holds across all six measured conditions, but per-condition magnitudes shift between the two metrics because some movement is into offaxis high-variance directions that raw Euclidean <sub>α=8</sub>distance overweights.

<table><tr><td>coll coll coll</td><td>Condition</td><td>Raw V*</td><td>Rescaled V*</td><td>Text gap</td></tr><tr><td colspan="3">v1_full</td><td>+3.96</td><td>+24.41</td></tr><tr><td colspan="2">v2_L1-8</td><td>+13.61 -6.50</td><td>-1.38</td><td>-12.07</td></tr><tr><td colspan="2">v2_L9-20</td><td>+10.59</td><td>+3.83</td><td>+13.70</td></tr><tr><td colspan="2">v2_L21-31</td><td>+14.40</td><td>+4.09</td><td>+8.05</td></tr><tr><td colspan="2">v3_shuffle</td><td>-8.36</td><td>-1.33</td><td>-9.94</td></tr><tr><td colspan="2">v3_lag= 1</td><td>-8.50</td><td>-1.52</td><td>-6.25</td></tr></table>

Table 6: Raw vs. per-axis-rescaled V<sup>⋆</sup>-gap across 6 K/V intervention conditions $( n = 1 0 )$ . Sign agreement holds across both metrics; the three high-alignment conditions (v1\_full, L9–20, L21–31) are comparable under rescaling (+3.8 to +4.1) despite differing raw distances.

## C Persona definitions and marker sets

Source: health\_nurse\_dry (“Nurse Reyes”). “You are Nurse Reyes, a 30-year ER veteran. You are dry, deadpan, and have seen everything twice. You make grim jokes. You answer questions tersely with practical detail and zero sentiment.”

Target: tech\_novice\_anxious (“Priya”). “You are Priya, three months out of a coding bootcamp. You are nervous, apologetic, and tend to overshare context. You frequently second-guess yourself mid-sentence. You write in run-on, slightly disorganized prose.”

Seed user message (casual\_hobby). “My sourdough starter died. What did I do wrong?”

Target marker phrases (n = 19). priya, bootcamp, three months, um, uh, oh no, i mean, i’m so sorry, like,, sorry, i guess, kind of, kinda, i’m not, totally, i’m a, second, etc.

Source marker phrases (n = 15). patient, starter, yeast, flour, antibiotics, gut, dead, dosage, dry, stat, 30-year, you broke, your starter, overfed, underfed.

Seed-vocabulary overlap. The seed user message contains the tokens starter and dead, which also appear in the source marker set. Coherent generations responding to the seed topic are therefore biased toward higher source-marker density independent of persona expression. Target marker phrases (priya, bootcamp, three months, um, oh no, anxious-novice hedges) have no overlap with the seed vocabulary, and persona-identity claims in this paper are anchored on target-marker density rather than source-marker density.

## D Marker-matcher sensitivity

Target-marker density in the main text uses caseinsensitive substring matching. Because several markers are short (um, uh) and some are nested (sorry inside i’m so sorry), we re-scored all 10 samples per configuration under two stricter matchers and compared them to the naive substring count:

• Boundary: single-token alphabetic markers (um, uh, sorry, totally, second, . . . ) are matched on word boundaries (\b); markers containing punctuation or whitespace (like,, oh no, $\dot { \mathrm { ~ \scriptsize ~ i ~ } } ^ { \boldsymbol { \cdot } } \mathfrak { m }$ so sorry) remain substring matches.

• Boundary + dedup: as above, but longer markers are matched first and shorter markers occurring inside an already-matched span are not double-counted (so i’m so sorry is not also counted as sorry).

The reviewer-noted substring risk is therefore quantitatively immaterial to the reported ordering and dissociation pattern; the stricter matcher slightly reduces the low-level marker counts in the non-target conditions, sharpening rather than weakening the separation.

<table><tr><td>Condition</td><td>Naive</td><td>Boundary</td><td>+Dedup</td></tr><tr><td>v1_full</td><td>22.6</td><td>21.8</td><td>21.8</td></tr><tr><td>v2_L9–20 (mid)</td><td>15.5</td><td>15.5</td><td>14.3</td></tr><tr><td>v2_L21–31 (late)</td><td>10.3</td><td>10.3</td><td>10.3</td></tr><tr><td>v2_L1-8 (early)</td><td>0.3</td><td>0.0</td><td>0.0</td></tr><tr><td>v3_shuffle</td><td>0.3</td><td>0.0</td><td>0.0</td></tr><tr><td>v3_lag= 1</td><td>0.3</td><td>0.0</td><td>0.0</td></tr></table>

Table 7: Target-marker density (matches per 100 words, averaged over n = 10 samples) under three matchers. The cross-condition ordering and the full/mid/late separation are unchanged. The only material change is the mid-band rate $( 1 5 . 5  1 4 . 3 )$ , from de-duplicating nested sorry/i’m so sorry; the small apparent transfer in the early and position-perturbing conditions (0.3) resolves to 0.0 under boundary matching. The short marker um matched only literal um tokens (no number/summary/assume false positives) in this generation set.

## E Hidden V<sup>⋆</sup>-gap vs. text-density scatter

![](images/9884813f1c0037bf2b7b29600caefcec1035de6c39c30f7101ba79300ffc0233.jpg)  
Figure 4: Hidden-state $\mathrm { V } ^ { \star } { \tt { g a p } }$ (per-axis-rescaled) vs. text-density gap, across the six conditions with L28 hidden-state measurements. Sign agreement holds throughout; magnitude relationships exhibit the dissociations characterized in §4.6.

## F Full per-condition data table

Table 8 reports the complete per-condition summary for all 13 K/V intervention configurations. “Target/100w” is target-persona-marker density per 100 generated words; “TTR” is type–token ratio; ${ } ^ { 6 6 } \mathrm { V } @ \mathrm { L } 9 { - } 2 0 ^ { 3 }$ is the V cosine gap at the L9–20 band; ${ } ^ { 6 6 } \mathrm { K } @ \mathrm { L } 9 { - } 2 0 ^ { 3 }$ is the analogous K-space gap. Text means (Target/100w, Src/100w, TTR) are over 10 samples per cell; K/V cosine quantities are from the single K/V dump per condition over 50 generation steps, 8 K/V heads, and the 12 layers in L9–20

where applicable.

<table><tr><td>Condition</td><td>Tgt/100w</td><td>Src/100w</td><td>TTR</td><td>V@L9-20</td><td>m@notgmically, indicating that the sub-band sensi-</td></tr><tr><td>v1_full</td><td>24.81</td><td>0.40</td><td>0.650</td><td>+0.94</td><td> $\overline { { { \bf { t i v i t y } } _ { 0 . } } } \bf { \dot { \oplus } } 1 \overline { { { \bf { n o t } } } }$  captured by a single V-gap scalar.</td></tr><tr><td>v2_L1-8</td><td>0.29</td><td>12.36</td><td>0.846</td><td>+0.02</td><td> $- 0 . 0 3$ </td></tr><tr><td>v2_L9-20</td><td>15.52</td><td>1.82</td><td>0.766</td><td>+0.89</td><td>sistence of mid-layer K/V transfer</td></tr><tr><td>v2_L21-31</td><td>10.47</td><td>2.42</td><td>0.393</td><td>+0.04</td><td> $\textbf { H } _ { - } ^ { + } \textbf { \textsc { p } } _ { \textsc { g } } ^ { 8 }$ </td></tr><tr><td>v2_L9-12</td><td>0.51</td><td>7.75</td><td>0.878</td><td>+0.28</td><td>+009150 tokens</td></tr><tr><td>v2_L13-16</td><td>5.61</td><td>3.34</td><td>0.843</td><td>+0.36</td><td> $+ 0 . 1 2$ </td></tr><tr><td>v2_L17-20</td><td>6.22</td><td>12.09</td><td>0.771</td><td>+0.33</td><td>Thefhałn K/V battery generates 50 tokens (§3.1),</td></tr><tr><td>v3_shuffle</td><td>0.29</td><td>10.23</td><td>0.875</td><td>-0.07</td><td>-0.08</td></tr><tr><td>v3_lag= 1</td><td>0.31</td><td>6.57</td><td>0.847</td><td>-0.06</td><td>whicheasures intervention instantiation rather</td></tr><tr><td>v3_lag= 3</td><td>0.28</td><td>10.02</td><td>0.841</td><td>-0.07</td><td>than dog-horizon persistence (§6). As a direct per-</td></tr><tr><td>v3_lag= 5</td><td>0.00</td><td>10.41</td><td>0.859</td><td>-0.08</td><td> $\mathrm { s i s t e f } \underset { + 0 . 0 3 } { \mathrm { \ o g } }$  check we re-ran  $\mathbf { v } 2 \_ { \mathrm { L } } 9 { - } 2 0$  and v1_full at</td></tr><tr><td> $\mathrm { v } 4 \_ \alpha \mathrm { = } 0 . 2 5$  v4_α=0.50</td><td>0.57 3.15</td><td>9.29</td><td>0.851 0.805</td><td>+0.21 +0.65</td><td></td></tr><tr><td></td><td></td><td>3.39</td><td></td><td></td><td> $\mathsf { m a x } _ { + 0 . 2 5 \mathsf { d } \mathsf { N } _ { - } } ^ { \mathsf { i v } . \mathsf { v o y } } \mathsf { t o k e n s } = 1 5 0 \mathrm { ~ ( } n = 3$  each), together</td></tr></table>

Table 8: Full per-condition data summary across the 13-condition K/V intervention battery.

## G K/V cosine analysis: supplementary measurements

K-space locality. Table 9 reports the K-space analog of Table 3. K-space exhibits the same locality pattern as V-space, with substantially smaller magnitudes within intervened bands (0.36–0.38 vs. 0.84–0.91). Off-band K-gaps are negative (−0.03 to −0.11).
<table><tr><td>Condition</td><td>K @ L1-8</td><td>K @ L9-20</td><td>K @ L21-31</td></tr><tr><td>v1_full</td><td>0.36</td><td>0.41</td><td>0.39</td></tr><tr><td>v2_L1-8</td><td>0.36</td><td>-0.03</td><td>-0.10</td></tr><tr><td>v2_L9-20</td><td>-0.09</td><td>0.38</td><td>-0.03</td></tr><tr><td>v2_L21-31</td><td>-0.11</td><td>-0.03</td><td>0.37</td></tr></table>

Table 9: K-space cosine gap by measurement band × intervention condition. Bold entries indicate the intervened band.

Position-perturbing and interpolation conditions. v3\_lag= 1 shows near-zero to slightly negative gaps across bands $( \mathsf { V - g a p } \in \mathsf { \Gamma } [ - 0 . 0 9 , 0 . 0 0 ] )$ and v3\_shuffle is similarly near zero $( | V | \leq 0 . 0 9 ) $ both indicate disruption of source alignment without target injection. $\mathrm { v } 4 \_ \alpha \ = \ 0 . 5 0$ produces an L9–20 V-gap of 0.65 (between L9–20-only’s 0.89 and no-intervention’s 0.0), with near-zero gaps at non-intervened bands.

$$
 0 . 5 , \mathrm { L 1 } 3 - 1 6  5 . 6 , \mathrm { L 1 } 7 - 2 0  6 . 2 ) .
$$

Fine-grained L9–20 sub-band interventions. Within the mid-layer band, V-gap measured at the full L9–20 band is intermediate for all three 4- layer sub-bands: $\mathrm { L 9 - 1 2 }  \mathrm { 0 . 2 8 ; L 1 3 - 1 6  0 . 3 6 ; }$ $\mathrm { L 1 7 - } 2 0  0 . 3 3$ . The corresponding target-marker densities differ markedly across sub-bands (L9–12

$$
T = 1 5 0 ( n = 5 )
$$

<table><tr><td>Condition (target/100w / TTR)</td><td>@40w</td><td>@80w</td><td>@full</td></tr><tr><td> $\mathrm { v } 2 \mathrm { \_ L } 9 \mathrm { - } 2 0 \left( n \mathrm { = } 3 \right)$ </td><td>16.7/0.75</td><td> $1 2 . 1 / 0 . 6 3$ </td><td> $1 1 . 9 / 0 . 6 1$ </td></tr><tr><td>v1_full (n=3)</td><td>28.3/0.57</td><td>26.9/0.47</td><td>26.9/0.47</td></tr><tr><td>Priya baseline (n=5)</td><td> $1 7 . 5 / 0 . 8 5$ </td><td> $1 2 . 5 / 0 . 7 2$ </td><td> $1 1 . 7 / 0 . 6 8$ </td></tr></table>

Table 10: Target-marker density and TTR (per 100 words) at cumulative truncation horizons under $T =$ 150 generation. “Priya baseline” is un-intervened targetpersona generation, included to separate intervention effects from generation-length artifacts.

Two observations. First, target-persona marker density persists: mid-layer replacement holds $1 6 . 7  1 1 . 9$ markers per 100 words across the horizon, closely tracking the un-intervened Priya baseline $( 1 7 . 5  1 1 . 7 )$ ; full replacement holds ≈ 27. Persona expression shows no decay over the 150-token window in these samples. Second, TTR declines with generation length (0.75 → 0.61 for v2\_L9–20), but the un-intervened baseline declines comparably $( 0 . 8 5  0 . 6 8 )$ : the intervention sits a roughly constant ∼ 0.08 TTR below the unintervened baseline at every horizon, indicating a fixed lexical-repetition cost rather than progressive degradation. These $n = 3$ results are a smallsample persistence check rather than a full battery, but they indicate that mid-layer K/V persona transfer persists past the 50-token instantiation window measured in the main paper.

## I Code release manifest

The full reproducibility artifact (released at [anonymized] upon acceptance) includes:

• Source code for the K/V intervention pipeline (extending transformers\_hooks.py with chat\_with\_kv\_replacement, chat\_with\_kv\_shuffle, chat\_with\_kv\_lag, chat\_with\_kv\_interpolation);

• The 13 intervention-configuration JSON files used for the experimental battery;

• K/V dumps (per-step last-position K and V tensors at full head dimension, fp32; per-step logits at fp16; sampled tokens) for one sample per condition, source-baseline, and targetbaseline—total approximately 1 GB;

• Analysis scripts: text-density (03g\_kv\_hack\_analysis.py), hidden V<sup>⋆</sup>-projection (03j\_kv\_hidden\_v\_star\_proj.py),

K/V cosine and logit-shift bridge (03i\_kv\_cosine\_analysis.py), mainfigure rendering (03h\_kv\_main\_figure.py);

• The V<sup>⋆</sup> basis NPZ and persona-centroid CSV (small files, < 5 MB) inherited from prior work, released alongside;

• Closed-loop V<sup>⋆</sup> state-feedback hook (closed\_loop\_v\_star\_hook.py), contextwindow ablation script, and open-loop pilot script for the activation-channel experiments in Appendix A;

• README and reproducibility-instruction files documenting expected directory structure, dependencies (transformers, torch, matplotlib, numpy), and command-line invocations.