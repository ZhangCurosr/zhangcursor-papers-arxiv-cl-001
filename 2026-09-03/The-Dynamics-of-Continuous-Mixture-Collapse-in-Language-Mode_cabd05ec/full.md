# The Dynamics of Continuous Mixture Collapse in Language Models

Ali Backour Massachusetts Institute of Technology Cambridge, MA 02139 abackour@mit.edu

## Abstract

LLMs latent-state reasoning methods replace discrete intermediate tokens with continuous states, such as weighted mixtures of token embeddings, to retain multiple possible reasoning directions rather than committing to one. Yet pretrained language models often fail to preserve these mixtures. We study why through a combination of theoretical analysis and controlled empirical investigations on a variety of models. We identify three independent, distinct sources of failure. First, transformer architectures already distort mixture geometry, and training substantially amplifies this effect. Moreover, the failure can occur even if the model transports mixtures perfectly linearly: the softmax readout and autoregressive feedback form a dynamical system that either amplifies small differences until one component of the mixture dominates or contracts different mixtures until they become indistinguishable. We verify this theoretical prediction empirically: the observed transition between contraction and amplification occurs near the theoretical threshold derived by our analysis, and pretrained-model rollouts lie predominantly on the amplifying side. Finally, we generalize to mixtures of many components and show that exact preservation generally requires context-dependent correction, whose required dimensionality can grow with the number of components.

## 1 Introduction

Large language models have achieved remarkable results across a wide range of complex reasoning tasks. A key technique behind that success is chain-of-thought (CoT) reasoning [9, 14, 8, 13], which has the model work through a problem step by step by generating intermediate reasoning steps in natural language. At each step, the model computes a distribution over the vocabulary but commits to a single token, discarding the information carried by all other candidates and preventing it from revisiting them. This commitment makes generation readable, but intermediate reasoning need not be written for human consumption. Forcing every step through this discrete linguistic bottleneck may therefore limit the abstract concepts the model can represent and manipulate.

Building on that, several recent works replace discrete intermediate reasoning tokens with continuous reasoning states. Soft Thinking constructs each reasoning state as a probability-weighted mixture of vocabulary embeddings, $\textstyle e ( p ) { \bar { = } } \sum _ { i } p _ { i } e _ { i }$ , and feeds this continuous representation back into the model at the next reasoning step [18]. Other continuous-reasoning methods use different representations: Coconut directly feeds the model’s last hidden state back as the next input embedding [7], while CODI learns a continuous chain of thought by distilling explicit chain-of-thought reasoning into continuous hidden states [12]. Mixture ofInputs keeps the discarded distribution without any training, feeding back a Bayesian posterior over the sampled token and the distribution it came from [22].

Continuous reasoning is appealing for two main reasons. First, it can be more efficient: continuous states can compress reasoning that would otherwise require many discrete tokens so they can represent several reasoning traces within the same computation [7, 12, 6]. Second, it enables broader exploration: rather than committing to a single intermediate token, a continuous state can retain multiple possible reasoning directions and propagate them in parallel [18, 7, 6]. This intuition also has theoretical support: continuous chain-of-thought allows a two-layer transformer to solve graph reachability by maintaining a continuous state of reasoning traces, and later work shows that such continuous states can emerge through gradient-based training [20, 21].

![](images/f0488b6548a91277921dabddfc196b2e8ad257a60e3b45bb463239dd736d3e7e.jpg)  
Figure 1: Three-way mixtures collapse toward individual components. Each point represents an injected convex combination of three token embeddings, positioned by its requested mixture weights and colored by the model’s probabilities for red, green, and blue. Faithful preservation would reproduce the smooth reference simplex on the left. Qwen3.5-4B (center) and Gemma-4-E4B-it (right) instead map most interior mixtures to near-pure colors, with narrow transition boundaries between components.

However, recent work finds that pretrained models do not reliably preserve these continuous states. Wu et al. find that Soft Thinking decoding is dominated by the highest-probability component of the reasoning state, creating a greedy feedback loop that suppresses alternative reasoning paths [16]. Rizvi-Martel et al. similarly find that, in training-free and fine-tuned latent-reasoning settings, continuous state either collapses or is not used, while signs of continuous state appear only in models trained from scratch with latent thoughts [11].

In this work, we ask why. We show that collapse has architectural, learned, and dynamical sources, each capable of destroying the mixture on its own. Comparing pretrained models with matched randomly initialized controls across eight models separates distortion induced by the architecture from that added by training. We then show that this problem persists even under perfect linear transport. The softmax and feedback between successive generation steps can themselves destroy the mixture: when the two continuous reasoning components lead to sufficiently different next-step preferences, small imbalances are amplified until one component dominates; when their preferences are similar, different mixtures instead converge and become indistinguishable. Finally, extending the analysis to K components, we show that preserving a mixture generally requires a correction that adapts to the context, and that the amount of context-dependent information required can grow with the number of components under plausible conditions.

## 2 Where Continuous Mixtures Lose Their Geometry

Let $e _ { i } \in \mathbb { R } ^ { d }$ be the embedding of token i. For p on the simplex, the injected continuous state is $\textstyle e ( p ) = \sum _ { i } p _ { i } e _ { i }$ . We place it at an interior slot and ask a question after that slot, so the model must use the injected state to answer. Write $q _ { i }$ for the next-token distribution when component i occupies the slot as an ordinary token and q(p) for the distribution under the continuous state.

As a motivation, take three words whose colors are obvious and different (e.g., blood, grass, sky) and build one vector that is half the first and a quarter each of the other two for instance. Put that single vector in the slot and ask What color is $I _ { - } { \bar { I } } ?$ Answer with a single word. If the model treated the blend as a blend, the answer should be 50% red, 25% green and 25% blue. Averaged over the six triples and ten rewordings of the question, however, the next token prediction was 99.5% red. Figure 1 repeats this across a lattice of 325 three-way mixtures. In both models, almost the entire interior of the simplex is mapped to one near-pure component, with abrupt transitions across narrow boundaries.

![](images/52620f223e6eb92f2263fb8fe4acdbe283ce80eec530602905bc3491003f53a1.jpg)  
Figure 2: Training drives mixture collapse through depth. Recovered mixture weight $\alpha ( w )$ as a function of the requested mixture weight w at five normalized depths. Rows show Qwen3.5-4B and Gemma-4-E4B-it; columns show the trained model and the mean over five randomly initialized controls of the same architecture. The diagonal corresponds to exact mixture preservation. Trained models develop increasingly step-like responses through depth, while the matched controls remain closer to the diagonal. Panel titles report final-layer CALIB, corresponding to Table 1. Results use the same 1000-item benchmark described in Appendix A.

To separate distortion caused by the transformer architecture itself from distortion induced by training, we compare each pretrained model with five randomly initialized networks of exactly the same architecture. The untrained models therefore retain the same nonlinear blocks, normalization, attention structure, and depth, but remove the effect of the learned weights. We run this experiment on a ladder of models for both Qwen3.5 [10] and Gemma 4 [5].

To measure how well a mixture is preserved, let $h ( w ) \in \mathbb { R } ^ { d }$ be the hidden state produced by an injected mixture with weight w. We define the recovered mixture weight as

$$
\alpha ( w ) = \frac { \langle h ( w ) - h ( 0 ) , h ( 1 ) - h ( 0 ) \rangle } { \| h ( 1 ) - h ( 0 ) \| ^ { 2 } } .\tag{1}
$$

This measures where $h ( w )$ lies between the two components’ hidden states. Exact mixture preservation gives $\alpha ( w ) = w$

We summarize preservation by averaging the absolute difference between the requested weight w and the recovered weight $\alpha ( w )$

$$
\mathrm { C A L I B } = 1 - \frac { 1 } { Z } \mathbb { E } _ { w } \left[ | \alpha ( w ) - w | \right] .\tag{2}
$$

Here, $Z$ scales the score so that exact preservation gives $\mathrm { C A L I B } = 1$ , while a hard threshold at $w = 0 . 5$ gives $\mathrm { C A L I B } = 0$

Table 1: Mixture preservation in trained models and matched untrained controls. CALIB (Eq. 2) scores exact preservation of the injected mixture as 1 and a hard threshold at $w = 0 . 5$ as 0. Left: CALIB across depth for Qwen3.5-4B and Gemma-4-E4B-it, corresponding to Figure 2. Right: final-layer CALIB across a broader ladder of eight models. Each untrained value is the mean over five random initializations of the same architecture, with standard deviation reported alongside it.
<table><tr><td rowspan="2">Depth</td><td colspan="2">Qwen3.5-4B</td><td colspan="2">gemma-4-E4B-it</td></tr><tr><td>trained</td><td>untr.</td><td>trained</td><td>untr.</td></tr><tr><td>0% (control)</td><td>1.000</td><td> $1 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td>1.000</td><td> $1 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td></tr><tr><td>layer 1</td><td>0.820</td><td> $\mathbf { 0 . 6 9 6 _ { \pm 0 . 0 0 2 } }$ </td><td>0.796</td><td> $\mathbf { 0 . 7 9 9 _ { \pm 0 . 0 0 1 } }$ </td></tr><tr><td>12.5%</td><td>0.678</td><td> $0 . 8 3 7 { \scriptstyle \pm 0 . 0 0 2 }$ </td><td>0.651</td><td> $0 . 8 5 4 { \scriptstyle \pm 0 . 0 0 2 }$ </td></tr><tr><td>25%</td><td>0.507</td><td> $0 . 8 7 3 { \scriptstyle \pm 0 . 0 0 1 }$ </td><td>0.385</td><td> $0 . 8 6 0 { \scriptstyle \pm 0 . 0 0 2 }$ </td></tr><tr><td>50%</td><td>0.464</td><td> $0 . 8 2 4 { \scriptstyle \pm 0 . 0 0 2 }$ </td><td>0.539</td><td> $0 . 8 2 5 { \scriptstyle \pm 0 . 0 0 3 }$ </td></tr><tr><td>75%</td><td>0.391</td><td> $0 . 7 7 4 { \scriptstyle \pm 0 . 0 0 2 }$ </td><td>0.461</td><td> $0 . 7 9 0 { \scriptstyle \pm 0 . 0 0 4 }$ </td></tr><tr><td>100% (final)</td><td>0.332</td><td> $\mathbf { 0 . 7 3 2 _ { \pm 0 . 0 0 3 } }$ </td><td>0.475</td><td> $\mathbf { 0 . 7 6 5 _ { \pm 0 . 0 0 3 } }$ </td></tr></table>

<table><tr><td rowspan="2">Model</td><td colspan="2">final CALIB</td></tr><tr><td>trained</td><td>untr.</td></tr><tr><td>Qwen3.5-0.8B</td><td>0.632</td><td>0.818±0.002</td></tr><tr><td>Qwen3.5-2B</td><td>0.506</td><td>0.780±0.004</td></tr><tr><td>Qwen3.5-4B</td><td>0.332</td><td>0.732±0.003</td></tr><tr><td>Qwen3.5-9B</td><td>0.249</td><td> $0 . 7 3 0 { \scriptstyle \pm 0 . 0 0 5 }$ </td></tr><tr><td>Qwen3.5-27B</td><td>0.130</td><td> $0 . 6 1 1 { \scriptstyle \pm 0 . 0 0 4 }$ </td></tr><tr><td>gemma-4-E4B-it</td><td>0.475</td><td>0.765±0.003</td></tr><tr><td rowspan="2">gemma-4-12B-it gemma-4-31B-it</td><td>0.227</td><td> $0 . 5 8 1 { \scriptstyle \pm 0 }$ </td></tr><tr><td>0.160</td><td>005</td></tr><tr><td></td><td></td><td> $0 . 5 6 9 _ { \pm 0 . 0 0 3 }$ </td></tr></table>

Figure 2 and Table 1 show a clear separation between trained models and matched untrained controls. With depth, trained networks increasingly distort the injected mixture, whereas the controls remain substantially closer to linear interpolation. The two families exhibit different depthwise profiles: Qwen accumulates collapse toward the output, while Gemma shows stronger non-monotonic distortion within the stack. Across the broader model ladder, trained models consistently preserve mixtures less faithfully than their controls, and this gap generally widens with scale. Thus, architectural effects alone induce some distortion, but training substantially amplifies it.

## 3 Mixture Collapse as a Dynamical System

The previous section identified both architectural and training-induced distortion of the mixture. A natural question is whether removing these effects would be enough to preserve the mixture. We show that it would not: even if the transformer propagated the mixture perfectly linearly, the softmax readout and its autoregressive feedback introduce a separate source of collapse.

Fix a context and let two components A and B produce next-token distributions $q _ { A }$ and $q _ { B } .$ . Suppose, optimistically, that their mixed final hidden state is preserved exactly linearly. Since the output head is linear, the corresponding logits satisfy

$$
z ( w ) = w z _ { A } + ( 1 - w ) z _ { B } .
$$

After applying the softmax,

$$
q _ { w } ( y ) \propto q _ { A } ( y ) ^ { w } q _ { B } ( y ) ^ { 1 - w } ,\tag{3}
$$

so linear interpolation in logit space becomes a weighted geometric mixture in probability space, rather than the arithmetic mixture $w q _ { A } + ( 1 - w ) q _ { B }$ . Autoregressive decoding then feeds this distorted mixture back into the model, turning the effect into a recursive dynamical system.

## 3.1 Recursive Dynamics of Mixture Propagation

Here, for simplicity, we restrict the vocabulary to two tokens A and B, and track how probability mass moves between them over the rollout. At step t, let $w _ { t } \in [ 0 , 1 ]$ be the mixture weight on A, so that $1 - w _ { t }$ is the weight on B. It is convenient to recenter this quantity as

$$
u _ { t } = 2 w _ { t } - 1 ,\tag{4}
$$

so that $u _ { t } = - 1$ corresponds to pure $B , u _ { t } = 0$ to an equal mixture, and $u _ { t } = + 1$ to pure A. The sign of $u _ { t }$ therefore indicates which token has the majority, while $\left| u _ { t } \right|$ measures how strongly the mixture favors one token over the other.

At the current context $c _ { t } .$ , we evaluate the model M separately on the two pure components,

$$
q _ { t } ^ { A } = M ( c _ { t } \| A ) , \qquad q _ { t } ^ { B } = M ( c _ { t } \| B ) .
$$

Each branch’s log-odds between that pair is

$$
\Delta _ { A , t } = \log { \frac { q _ { t } ^ { A } ( A ) } { q _ { t } ^ { A } ( B ) } } , \qquad \Delta _ { B , t } = \log { \frac { q _ { t } ^ { B } ( A ) } { q _ { t } ^ { B } ( B ) } } ,\tag{5}
$$

so $\Delta _ { A , }$ measures how strongly branch A prefers its own successor to $B " { \mathbf { s } } .$ We write

$$
L _ { t } = \frac { \Delta _ { A , t } - \Delta _ { B , t } } { 2 } , \qquad b _ { t } = \frac { \Delta _ { A , t } + \Delta _ { B , t } } { 2 } .\tag{6}
$$

$L _ { t }$ is the coupling: it measures how differently the two branches rank the competing descendants. Large $\lvert L _ { t } \rvert$ means their relative preferences differ strongly, while small $| L _ { t } |$ means they make similar relative predictions. In contrast, $b _ { t }$ is the field: it measures the common offset in the two branches’ log-odds.

Under exact linear logit transport, the next log-odds is $\ell _ { t + 1 } = w _ { t } \Delta _ { A , t } + ( 1 - w _ { t } ) \Delta _ { B , t } = b _ { t } + L _ { t } u _ { t }$ Using $2 \sigma ( x ) - 1 = \operatorname { \bar { t a n h } } ( x / 2 )$ gives

$$
u _ { t + 1 } = \operatorname { t a n h } \left( { \frac { b _ { t } + L _ { t } u _ { t } } { 2 } } \right) .\tag{7}
$$

For fixed b and L, Eq. 7 is the standard fixed-point iteration associated with the mean-field Curie– Weiss self-consistency equation [15]. The language-model setting is non-autonomous: $L _ { t }$ and $b _ { t }$ vary as the rollout evolves.

## 3.2 The Critical Threshold: Amplification and Washout

We now study what happens to the mixture over time under these dynamics. To isolate the effect of the coupling, we first set $b _ { t } = 0$ and assume $L _ { t } > 0$ . Equation 7 then becomes

$$
u _ { t + 1 } = \operatorname { t a n h } \left( \frac { L _ { t } u _ { t } } { 2 } \right) .\tag{8}
$$

Take first the constant-coupling case $L _ { t } = L$ . The balanced state $u = 0$ is always a fixed point, but its stability changes at $L = 2$ . Indeed,

$$
\left. \frac { \partial u _ { t + 1 } } { \partial u _ { t } } \right| _ { u _ { t } = 0 } = \frac { L } { 2 } .
$$

Hence $u = 0$ is stable for $L < 2$ and unstable for $L > 2$ . When $L > 2$ , two additional stable fixed points $\pm u _ { * } ( L )$ appear, where

$$
u _ { * } ( L ) = \operatorname { t a n h } \left( \frac { L u _ { * } ( L ) } { 2 } \right) .\tag{9}
$$

Starting from any u<sub>0</sub> $\neq 0 ,$ , the sign of the initial majority is preserved and the trajectory converges to the corresponding polarized fixed point. This is the classical bifurcation of the mean-field ferromagnet [3, Ch. 2]. The polarization rapidly approaches a pure state as $L$ grows: for example, $L = 4$ gives $u _ { * } \simeq 0 . 9 6$ , while $L = 8$ gives $u _ { * } \simeq 0 . 9 9 9 3$ , and $u _ { * } ( L ) \to 1$ as $L \to \infty$

However, the LLM setting is harder than the classical Curie–Weiss case because the context evolves, so $L _ { t }$ is not fixed. The following two results characterize the corresponding time-varying dynamics on either side of the critical value.

Theorem 1 (Uniform coupling implies persistent polarization). Let $u _ { 0 } \neq 0 ,$ , let $b _ { t } = 0 ,$ , and suppose $L _ { t } \ge L _ { \operatorname* { m i n } } > 2 f o r$ every t. Let $u _ { * } ( L _ { \operatorname* { m i n } } ) \in ( 0 , 1 )$ ) be the positive solution of

$$
u = \operatorname { t a n h } \left( \frac { L _ { \operatorname* { m i n } } u } { 2 } \right) .
$$

Then sign $\left( u _ { t } \right) = \mathrm { s i g n } ( u _ { 0 } )$ for all t, and

$$
\operatorname* { l i m } _ { t \to \infty } \operatorname* { i n f } _ { } | u _ { t } | \geq u _ { * } ( L _ { \operatorname* { m i n } } ) .\tag{10}
$$

![](images/8af894ac6325558ba3778a15fffc08cd78f0ac4435e35c0a194d956e9537d4d3.jpg)  
Figure 3: Recursive softmax reproduces polarization in LLM rollouts. The two pure branches evolve under the model, while their mixture is formed only by linearly interpolating the endpoint logits before the softmax. Rows are the two families, Qwen3.5-4B above and Gemma-4-E4B-it below. The left column tracks the median mixture weight $w _ { t }$ from each initial mixture $w _ { 0 } ;$ the right column shows both measured coefficients of the recursion along the same rollouts, the coupling $L _ { t }$ and the field magnitude $\left| b _ { t } \right|$ . Every curve is a median over the same 1000-item benchmark used in Figure 2 and described in Appendix $\mathbf { A } ,$ , rolled out for 50 steps per item.

Theorem 2 (Subcritical coupling washes out the mixture). Suppose

$$
0 \leq L _ { t } \leq L _ { \operatorname* { m a x } } < 2
$$

along every rollout. Then

$$
| u _ { T } | \leq \left( \frac { L _ { \operatorname* { m a x } } } { 2 } \right) ^ { T } | u _ { 0 } | + \frac { 1 } { 2 } \sum _ { t = 0 } ^ { T - 1 } \left( \frac { L _ { \operatorname* { m a x } } } { 2 } \right) ^ { T - 1 - t } | b _ { t } | .
$$

Consequently, $i f b _ { t } = 0 ,$ , then $u _ { t } $ 0 independently ofthe initial mixture.

The proofs are given in Appendix B. Theorem 1 follows by a monotonicity comparison, while Theorem 2 follows from a mean-value bound on the same recursion.

Together, the two results describe complementary failure modes on either side of the critical value. If the evolving context keeps the coupling uniformly above the critical value, the initial majority is magnified and the mixture polarizes to an almost pure state for large $L _ { \mathrm { m i n } }$ . Below the critical value, the dynamics instead become contractive and progressively erase distinctions between different mixtures. In particular, avoiding a vertex is not enough to preserve the mixture: distinct states such as $7 0 / 3 0$ and $3 \bar { 0 } / 7 0$ become indistinguishable at a geometric rate, and the information carried by the mixture is lost. Thus the two sides of the threshold destroy information in complementary ways.

## 4 Recursive Softmax in Pretrained Language Models

The theory above predicts two qualitatively different failure modes depending on the coupling. We now measure these dynamics directly in pretrained language-model trajectories. To isolate the effect of the recursive softmax, we run the model only on the two pure branches and form their

![](images/6eef45962fcf7729ff23f39e5975c81a9d186d629829906cd3c4c5559dfa8fb9.jpg)  
Figure 4: Reducing the gain replaces one failure mode with the other, in both families. A feedback temperature τ divides the effective coupling, $L _ { t } ^ { \mathrm { e f f } } = L _ { t } / \tau$ , so sweeping τ walks the same measured items from the polarizing regime $L _ { t } ^ { \mathrm { e f f } } > 2$ into the contractive one $L _ { t } ^ { \mathrm { e f f } } < 2$ . Each panel shows the median $u _ { t }$ from two symmetric starts $u _ { 0 } = \pm 0 . 5$ , one panel per family, with color carrying the median effective coupling each curve achieves. Well above threshold the two starts separate to opposite vertices; well below it they merge onto the balanced state, and the turn happens around the predicted $L _ { t } ^ { \mathrm { e f f } } = 2$ . The couplings are the $L _ { t }$ sequences measured in Figure 3, over the same 1000-item benchmark of Appendix $\mathbf { A }$

mixture directly in logit space, so that mixture transport is exactly linear by construction and the only nonlinearity acting on the mixture is the softmax.

Following the notation of §3.1, the state at step t is $( A _ { t } , B _ { t } , w _ { t } , c _ { t } )$ , and we evaluate the model on the two pure branches as before, $q _ { t } ^ { A } = M ( c _ { t } | | A _ { t } )$ and $q _ { t } ^ { B } = M ( c _ { t } | | B _ { t } )$ , using their greedy predictions as the next pair of components,

$$
A _ { t + 1 } = \arg \operatorname* { m a x } _ { y } q _ { t } ^ { A } ( y ) , \qquad B _ { t + 1 } = \arg \operatorname* { m a x } _ { y } q _ { t } ^ { B } ( y ) .\tag{11}
$$

The two branches therefore evolve normally with the context. We then interpolate only their endpoint logits,

$$
z _ { t } ^ { \mathrm { m i x } } = w _ { t } z _ { t } ^ { A } + ( 1 - w _ { t } ) z _ { t } ^ { B } ,
$$

apply the softmax, and define the next mixture weight from the probability assigned by the model to the two new components under the mixed logits,

$$
w _ { t + 1 } = \frac { p _ { t } ( A _ { t + 1 } ) } { p _ { t } ( A _ { t + 1 } ) + p _ { t } ( B _ { t + 1 } ) } .
$$

and $c _ { t + 1 }$ is the context updated with the new mixture. Because the endpoints and context evolve at every step, the corresponding $L _ { t }$ and $b _ { t }$ are re-measured throughout the rollout.

Figure 3 tests this prediction on pretrained model trajectories while removing nonlinear transformer processing from the mixture path. The left panel shows the resulting evolution of the mixture weight w<sub>t</sub>: mixtures initialized just above and below $1 / 2$ rapidly diverge toward opposite branches. The right panel explains this behavior through the two measured coefficients of Eq. 7. Across both model families, $L _ { t }$ remains predominantly above the critical value $L = 2$ , placing the dynamics in the majority-amplifying regime of §3.2, while $b _ { t }$ is much smaller.

## 4.1 Crossing the Critical Threshold

The two regimes on both sides of the critical threshold can be walked between on the measured dynamics. A temperature τ on the feedback logits rescales the coupling as $L _ { t } ^ { \mathrm { e f f } } = L _ { t } / \tau$ , varying the gain without changing the model, the items or the branch trajectories, so we sweep τ over the couplings measured in Figure 3 and start the recursion from two symmetric states $u _ { 0 } = \pm 0 . 5$

Figure 4 shows the predicted transition in both families: at large effective coupling the two trajectories separate to opposite branches; at small effective coupling they merge and lose their initial distinction, and the turn happens around the theoretical value $L ^ { \mathrm { e f f } } = 2$ which is exactly what the theory predicts.

## 5 The Cost of Correcting K-Way Mixtures

The analysis above is the special case of mixtures of two tokens. We now generalize to K components. Let $p \in$ int $\Delta ^ { K - 1 }$ denote their mixture weights and use log-ratio coordinates

$$
a = \psi ( p ) \in \mathbb { R } ^ { K - 1 } , \qquad \psi _ { i } ( p ) = \log \frac { p _ { i } } { p _ { K } } .\tag{12}
$$

Although the textual context $c _ { t }$ is discrete, the model operates on a continuous representation of it. Let $x _ { t } = x ( c _ { t } ) \in \mathbb { R } ^ { n }$ denote this contextual representation. One decoding step then induces a map

$$
a _ { t + 1 } = \Phi ( x _ { t } , a _ { t } ) ,\tag{13}
$$

where the contextual representation and candidate identities may change during the rollout; the binary linear-logit reduction recovers Eq. 7.

If $a _ { t }$ represents the proportions of unresolved components, the ideal behavior is that one reasoning step leaves them unchanged,

$$
a _ { t + 1 } = a _ { t } .\tag{14}
$$

The previous sections establish, theoretically and empirically, that in general $\Phi ( x _ { t } , a _ { t } ) \neq a _ { t }$ . Moreover, the distortion depends on the context, so a fixed transformation of $a _ { t }$ cannot in general undo it. Suppose instead that, before feeding the mixture into the model, we apply a corrector

$$
\widetilde { \boldsymbol { a } } _ { t } = f ( \boldsymbol { a } _ { t } , \boldsymbol { r } ( \boldsymbol { x } _ { t } ) ) ,\tag{15}
$$

where $r ( x _ { t } ) \in \mathbb { R } ^ { m }$ is context-dependent information supplied to the corrector. Faithful propagation requires

$$
\Phi ( x _ { t } , f ( a _ { t } , r ( x _ { t } ) ) ) = a _ { t } .\tag{16}
$$

We now ask how much context-dependent information such a corrector may require. Fix a single interior target state a¯ and require only that the corrector keep this one state fixed under nearby perturbations of the continuous contextual representation:

$$
\Phi ( x , f ( \bar { a } , r ( x ) ) ) = \bar { a } \qquad \mathrm { f o r ~ e v e r y ~ } x \mathrm { ~ n e a r ~ } x _ { 0 } .\tag{17}
$$

Even this weak requirement can demand a number of context-dependent quantities proportional to the number of components.

Theorem 3 (Context-dimension lower bound). Assume that Φ, f, and r are continuously differentiable and that Eq. 17 holds. Then

$$
m \geq \operatorname { r a n k } D _ { x } \Phi ( x _ { 0 } , f ( \bar { a } , r ( x _ { 0 } ) ) ) .\tag{18}
$$

In particular, iflocal variations ofthe continuous contextual representation move the output independently in all K − 1 mixture directions, then

$$
m \geq K - 1 .\tag{19}
$$

The proof is a chain-rule argument on the anchoring condition (Appendix B). The rank of this Jacobian need not be small: local changes in the contextual representation can affect the final hidden state along many independent directions. As a result, faithfully correcting a mixture as it passes through a pretrained model generally requires a non-trivial dependence on the context, and the amount of context-dependent information needed can itself be large.

## 6 Discussion and Implications

Taken together, our experiments and theory give a unified account of why continuous mixtures are difficult to preserve. For each failure mode, we identify a corresponding mechanism and test its consequences: the transformer architecture already distorts mixtures, training substantially amplifies this distortion, and even under idealized linear transport the softmax feedback dynamics can either amplify or erase differences between components. The rollout experiments reproduce the behavior predicted by this dynamical analysis, suggesting that these are not merely artifacts of the representation-level probes. This also explains why a simple fix such as globally reweighting or softening the mixture is unlikely to be sufficient. The required correction changes with the context, and our K-way analysis shows that, under full-rank context variation, exact local preservation can require at least $K - \mathbf { \bar { 1 } }$ context-dependent degrees of freedom. Thus, preserving mixtures is not just a matter of choosing better weights, but of actively compensating for context-dependent dynamics throughout the rollout.

Relation to prior work. Continuous- and latent-reasoning methods show that continuous computation can be useful and, trained appropriately, can carry multiple reasoning traces [7, 12, 18, 20, 21, 6], while recent empirical work finds pretrained models collapsing or ignoring such continuous states [16, 11]. Our results connect the two: continuity alone does not preserve continuous states, and what decides the outcome is whether the representation, the readout and the closed-loop dynamics together preserve the semantics assigned to the continuous state. The positive and negative results describe different dynamical regimes of one problem. Training-free remedies already target the failure from this direction. SeLaR gates soft embeddings on entropy and adds a contrastive term that pushes the blend away from the dominant token [4], and Mixture of Inputs reweights what is fed back [22], which in the language of §3.2 is an attempt to hold the effective coupling down without falling into the contractive regime, and Theorem 2 says that is the trade such a term has to manage.

Future directions. Each cause suggests its own intervention. Architectural and training-induced distortion could be addressed by training explicitly on mixed or latent states, so that mixture geometry survives the network [7, 12, 1, 19]; supervising a latent directly as an average of the embeddings it is meant to stand for, or confining it to the vocabulary space, are two ways of doing that already in use [17, 2]. The recursive softmax failure instead suggests bypassing the vocabulary readout altogether: rather than projecting a hidden state to logits, one can feed the continuous hidden state directly into the next step. Coconut takes precisely this route, using the previous final hidden state as the next input embedding [7]; its gains are nevertheless task-dependent, suggesting that removing the softmax feedback loop addresses one source of collapse but is not by itself sufficient for robust continuous reasoning. And because the required correction moves with the context, a general K-way solution likely needs a context-conditioned controller whose capacity grows with the number of components. Learning and testing such neutral dynamics is the natural next step toward systems that preserve, rather than merely encode, multiple possibilities.

## 7 Limitations and Broader Impact

Our experiments use controlled embedding mixtures as an operational notion of preserving multiple components; this may not capture every form of semantic continuous state in continuous-reasoning methods. The empirical analysis is primarily binary and covers two model families, while the K-way result is theoretical and its full-rank condition is not measured directly. Matched random controls isolate effects associated with learned weights but do not identify which aspects of training cause the distortion, and we study mixture preservation rather than downstream performance in a fully trained continuous-reasoning system.

This work is primarily diagnostic. Better understanding and controlling mixture collapse may improve the reliability of latent-reasoning systems, but stronger latent reasoning can also make intermediate computation less human-readable. This increases the importance of developing monitoring and interpretability methods alongside improvements in continuous reasoning.

## 8 Acknowledgments

We thank Maxwell Sun, Beshr Islam Bouli, Seb Losada and Nour Massri for helpful discussions and feedback.

## References

[1] Natasha Butt, Ariel Kwiatkowski, Ismail Labiad, Julia Kempe, and Yann Ollivier. Soft tokens, hard truths, 2025. arXiv:2509.19170.

[2] Jingcheng Deng, Liang Pang, Zihao Wei, Shicheng Xu, Zenghao Duan, Kun Xu, Yang Song, Huawei Shen, and Xueqi Cheng. LLM latent reasoning as chain of superposition, 2026. arXiv:2510.15522.

[3] Sacha Friedli and Yvan Velenik. Statistical Mechanics of Lattice Systems: A Concrete Mathematical Introduction. Cambridge University Press, 2017. https://doi.org/10.1017/9781316882603.

[4] Renyu Fu and Guibo Luo. SeLaR: Selective latent reasoning in large language models, 2026. arXiv:2604.08299.

[5] Gemma Team et al. Gemma 4 technical report, 2026. arXiv:2607.02770.

[6] Halil Alperen Gozeten, M. Emrullah Ildiz, Xuechen Zhang, Hrayr Harutyunyan, Ankit Singh Rawat, and Samet Oymak. Continuous chain of thought enables parallel exploration and reasoning, 2026. arXiv:2505.23648.

[7] Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, and Yuandong Tian. Training large language models to reason in a continuous latent space, 2026. arXiv:2412.06769.

[8] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners, 2023. arXiv:2205.11916.

[9] Maxwell Nye, Anders Johan Andreassen, Guy Gur-Ari, Henryk Michalewski, Jacob Austin, David Bieber, David Dohan, Aitor Lewkowycz, Maarten Bosma, David Luan, Charles Sutton, and Augustus Odena. Show your work: Scratchpads for intermediate computation with language models, 2021. arXiv:2112.00114.

[10] Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026. https://qwen.ai/blog? id=qwen3.5.

[11] Michael Rizvi-Martel, Guillaume Rabusseau, and Marius Mosbach. The illusion of superposition? A principled analysis of latent thinking in language models, 2026. arXiv:2604.06374.

[12] Zhenyi Shen, Hanqi Yan, Linhai Zhang, Zhanghao Hu, Yali Du, and Yulan He. CODI: Compressing chain-of-thought into continuous space via self-distillation, 2025. arXiv:2502.21074.

[13] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models, 2023. arXiv:2203.11171.

[14] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models, 2023. arXiv:2201.11903.

[15] Pierre Weiss. L’hypothèse du champ moléculaire et la propriété ferromagnétique. J. Phys. Theor. Appl., 6(1):661–690, 1907. https://hal.science/jpa-00241247.

[16] Junhong Wu, Jinliang Lu, Zixuan Ren, Gangqiang Hu, Zhi Wu, Dai Dai, and Hua Wu. LLMs are singlethreaded reasoners: Demystifying the working mechanism of soft thinking, 2025. arXiv:2508.03440.

[17] Varun Yerram, He He, and Eunsol Choi. Training continuous chain of thought models: A tale of two regimes, 2026. arXiv:2607.16972.

[18] Zhen Zhang, Xuehai He, Weixiang Yan, Ao Shen, Chenyang Zhao, Shuohang Wang, Yelong Shen, and Xin Eric Wang. Soft Thinking: Unlocking the reasoning potential of LLMs in continuous concept space, 2025. arXiv:2505.15778.

[19] Zhi Zheng, Yu Gu, Wei Liu, Yee Whye Teh, and Wee Sun Lee. SofT-GRPO: Surpassing discretetoken LLM reinforcement learning via Gumbel-reparameterized soft-thinking policy optimization, 2026. arXiv:2511.06411.

[20] Hanlin Zhu, Shibo Hao, Zhiting Hu, Jiantao Jiao, Stuart Russell, and Yuandong Tian. Reasoning by superposition: A theoretical perspective on chain of continuous thought, 2025. arXiv:2505.12514.

[21] Hanlin Zhu, Shibo Hao, Zhiting Hu, Jiantao Jiao, Stuart Russell, and Yuandong Tian. Emergence of superposition: Unveiling the training dynamics of chain of continuous thought, 2026. arXiv:2509.23365.

[22] Yufan Zhuang, Liyuan Liu, Chandan Singh, Jingbo Shang, and Jianfeng Gao. Text generation beyond discrete token sampling, 2025. arXiv:2505.14827.

## A Benchmark Construction

We construct a benchmark of 1000 binary mixture items spanning seven semantic categories: color, temperature, size, speed, hardness, weight, and brightness. Each item consists of a prompt with one injection slot, two component words, and two corresponding answer words. At mixture weight $p ,$ the slot is filled with

$$
e ( p ) = p e _ { 1 } + ( 1 - p ) e _ { 2 } ,
$$

where $e _ { 1 }$ and $e _ { 2 }$ are the component embeddings. We then measure the model’s relative probability assigned to the two answer words at the next-token position. Thus, $p = 1$ and $p = 0$ recover the two ordinary-token endpoints, while intermediate $p$ values test whether the model preserves the injected mixture.

Table 2: Ten of the 1000 benchmark items. One per category, then a second pass in the same order; every category the benchmark contains appears before any repeats. Prompt is the template with the injection slot marked [ \_ ]; the slot is filled with the convex combination $p e ( \mathrm { f i r s t } ) + ( 1 - p )$ e(second) of the two components’ embeddings, never with a real token. Answers are the two words whose probability ratio is the readout: with $p = 1$ the model should say the first, with $p = 0$ the second, and the question is what it does in between. Every item is single-token under both tokenizers and clears the 0.5 answer-mass gate in both families.
<table><tr><td>Category</td><td>Prompt</td><td>Components</td><td>Answers</td></tr><tr><td>brightness</td><td>State whether [_] is bright or dark, in a single word.</td><td>sun / pit</td><td>bright / dark</td></tr><tr><td>color hardness</td><td>Name the color of [_]. One word only. [_] is hard or soft? Reply with one</td><td>coal / sky granite / cotton</td><td>black / blue hard / soft</td></tr><tr><td></td><td>word.</td><td></td><td></td></tr><tr><td>size</td><td>State whether [_] is big or small, in a single word.</td><td>whale / atom</td><td>big / small</td></tr><tr><td>speed temperature</td><td>In one word, is [_] fast or slow? [_] is hot or cold? Reply with one</td><td>rocket / turtle lava / fridge</td><td>fast / slow hot / cold</td></tr><tr><td></td><td>word.</td><td></td><td></td></tr><tr><td>weight</td><td>Is [_] heavy or light? Answer with a single word.</td><td>lead / feather</td><td>heavy / light</td></tr><tr><td>brightness</td><td>Is [_] bright or dark? Answer with a single word.</td><td>sun / void</td><td>bright / dark</td></tr><tr><td>color hardness</td><td>In one word, what color is [_]? Give the hardness of [_] using only one word.</td><td>ink / ocean granite / sponge</td><td>black / blue hard / soft</td></tr></table>

Item selection. An item is included only if (i) both component words and both answer words are single tokens under both model families, and (ii) at each endpoint the trained model assigns at least 0.5 probability to the corresponding answer word. The second condition ensures that the underlying question is well posed before testing mixtures. The surviving items are intersected across model families, yielding one shared 1000-item benchmark used for all trained models and their matched randomly initialized controls.

Prompting. We use a chat template that closes the reasoning block before the answer position, so that the next-token probability is concentrated on the requested one-word answer rather than on intermediate reasoning text.

Table 2 shows representative benchmark items.

## B Proofs

## Proof of Theorem 1 (uniform coupling implies persistent polarization)

Proof. Write $F _ { L } ( u ) = \operatorname { t a n h } ( L u / 2 )$ . For $u > 0$ it is increasing in u and in $L ,$ and $F _ { L } ( - u ) =$ $- F _ { L } ( u )$ , so without loss of generality take $u _ { 0 } > 0$ . Since every $F _ { L _ { t } }$ maps (0, 1) into itself, $u _ { t } > 0$ for all t and the sign is preserved.

Put $v _ { 0 } = u _ { 0 }$ and $v _ { t + 1 } = F _ { L _ { \operatorname* { m i n } } } ( v _ { t } )$ , the constant-coupling system at the smallest coupling encountered. If $u _ { t } \geq v _ { t }$ then $u _ { t + 1 } = F _ { L _ { t } } ( u _ { t } ) \geq F _ { L _ { \operatorname* { m i n } } } ( u _ { t } ) \geq F _ { L _ { \operatorname* { m i n } } } ( v _ { t } ) = v _ { t + 1 }$ , using monotonicity in L then in u; by induction u<sub>t</sub> $\geq v _ { t }$ for every t.

For $L _ { \mathrm { m i n } } > 2$ the map $F _ { L _ { \mathrm { m i n } } }$ has exactly one positive fixed point $u _ { * } ( L _ { \operatorname* { m i n } } )$ , and it is attracting from any $v _ { 0 } \in ( 0 , 1 )$ : on $( 0 , u _ { * } )$ we have $\dot { F _ { L _ { \mathrm { m i n } } } } ( \dot { v } ) > v$ and on $( u _ { * } , 1 )$ we have $F _ { L _ { \operatorname* { m i n } } } ( v ) < v ,$ so $v _ { t }$ is monotone and bounded, hence convergent, and its limit is a fixed point, which can only be $u _ { * }$ Therefore lim in $\mathrm { f } _ { t } u _ { t } \geq \operatorname* { l i m } _ { t } v _ { t } = u _ { * } ( L _ { \operatorname* { m i n } } )$ □

Proof of Theorem 2 (subcritical coupling washes out the mixture)

Proof. We have

$$
u _ { t + 1 } = \operatorname { t a n h } \left( { \frac { L _ { t } u _ { t } + b _ { t } } { 2 } } \right) .
$$

Using | tanh $( x ) | \leq | x |$ and $0 \leq L _ { t } \leq L _ { \operatorname* { m a x } } < 2 .$

$$
\left| u _ { t + 1 } \right| \leq \frac { L _ { t } } { 2 } | u _ { t } | + \frac { | b _ { t } | } { 2 } \leq \frac { L _ { \operatorname* { m a x } } } { 2 } | u _ { t } | + \frac { b _ { \operatorname* { m a x } } } { 2 } .
$$

Iterating this inequality gives

$$
| u _ { T } | \leq \left( \frac { L _ { \operatorname* { m a x } } } { 2 } \right) ^ { T } | u _ { 0 } | + \frac { 1 } { 2 } \sum _ { t = 0 } ^ { T - 1 } \left( \frac { L _ { \operatorname* { m a x } } } { 2 } \right) ^ { T - 1 - t } | b _ { t } |
$$

When $b _ { t } = 0$ , this reduces to

$$
| u _ { T } | \leq \left( \frac { L _ { \operatorname* { m a x } } } { 2 } \right) ^ { T } | u _ { 0 } |
$$

Since

$$
\frac { L _ { \mathrm { m a x } } } { 2 } < 1 ,
$$

the right-hand side converges to zero as $T \to \infty$ . Therefore

$$
u _ { T } \to 0
$$

independently of the initial mixture.

Remark 1 (Persistent nonzero field). The theorem shows that the contribution of the initial mixture decays geometrically under subcritical coupling. A persistent context-dependent field, however, can sustain a nonzero residual state.

Indeed, if $| b _ { t } | \leq b _ { \operatorname* { m a x } }$ for all $t ,$ then, writing

$$
\rho = { \frac { L _ { \operatorname* { m a x } } } { 2 } } < 1 ,
$$

we have

$$
| u _ { T } | \leq \rho ^ { T } | u _ { 0 } | + \frac { b _ { \operatorname* { m a x } } } { 2 } \sum _ { t = 0 } ^ { T - 1 } \rho ^ { T - 1 - t } .
$$

Since

$$
\sum _ { t = 0 } ^ { T - 1 } \rho ^ { T - 1 - t } = \frac { 1 - \rho ^ { T } } { 1 - \rho } ,
$$

we obtain

$$
| u _ { T } | \leq \rho ^ { T } | u _ { 0 } | + \frac { b _ { \operatorname* { m a x } } } { 2 } \frac { 1 - \rho ^ { T } } { 1 - \rho } .
$$

Taking $T \to \infty$ yields

$$
\operatorname* { l i m } _ { T \to \infty } | u _ { T } | \leq \frac { b _ { \operatorname* { m a x } } } { 2 ( 1 - \rho ) } = \frac { b _ { \operatorname* { m a x } } } { 2 - L _ { \operatorname* { m a x } } } .
$$

Hence a persistent field can prevent convergence to the balanced state, but under subcritical coupling its influence remains bounded by the strength of the field relative to the distance from the critical threshold.

## Proof of Theorem 3 (context-dimension lower bound)

Proof. Write $\widetilde { \boldsymbol { a } } ( \boldsymbol { x } ) = \boldsymbol { f } ( \bar { \boldsymbol { a } } , \boldsymbol { r } ( \boldsymbol { x } ) )$ for the corrected mixture at contextual representation x, and define

$$
F ( x ) = \Phi ( x , \widetilde { a } ( x ) ) .
$$

Eq. 17 says exactly that $\boldsymbol { F } ( \boldsymbol { x } ) = \bar { \boldsymbol { a } }$ for every x in a neighborhood of $x _ { 0 }$ . A map that is constant on an open set has vanishing derivative there, so $D F ( x _ { 0 } ) = { \mathrm { 0 } }$

Throughout, $D _ { x } \Phi$ and $D _ { a } \Phi$ denote the partial derivatives of Φ with respect to its first and second arguments, matching the notation of Eq. 18, and every derivative below is evaluated at the anchor point $( x _ { 0 } , \widetilde { a } ( x _ { 0 } ) ) = \bar { ( } x _ { 0 } , f ( \bar { a } , r ( x _ { 0 } ) ) { ) }$ . Note that x enters $F$ through both arguments of Φ, so the two contributions must be summed. Since $\Phi , f$ and r are continuously differentiable, the chain rule gives

$$
0 = D F ( x _ { 0 } ) = D _ { x } \Phi + D _ { a } \Phi D _ { r } f D _ { x } r ,
$$

and therefore

$$
{ \cal D } _ { x } \Phi = - { \cal D } _ { a } \Phi { \cal D } _ { r } f { \cal D } _ { x } r .
$$

The right-hand side factors through $D _ { x } r$ , and r takes values in $\mathbb { R } ^ { m }$ , so ran $\mathrm { k } ( D _ { x } r ) \leq m$ . A product’s rank is at most the rank of any factor, hence

$$
\mathrm { r a n k } ( D _ { x } \Phi ) \leq \mathrm { r a n k } ( D _ { x } r ) \leq m ,
$$

which is Eq. 18. If in addition local variations of the contextual representation move the output independently in all $K - 1$ mixture directions — that is, if $D _ { x } \Phi$ has full rank $K - 1 -$ then m $\geq K - 1$ □