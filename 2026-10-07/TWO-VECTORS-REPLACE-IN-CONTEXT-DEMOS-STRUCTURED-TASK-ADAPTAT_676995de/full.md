# TWO VECTORS REPLACE IN-CONTEXT DEMOS: STRUCTURED TASK ADAPTATION VIA EMBEDDINGS

Xi Ding<sup>♠</sup> Naichen Shi<sup>♡</sup> Jiawei Zhang<sup>♠</sup>

<sup>♠</sup>University of Wisconsin–Madison <sup>♡</sup>Northwestern University

## ABSTRACT

In-context learning (ICL) adapts frozen large multimodal models (LMMs) to new tasks from a few demonstrations (demos), but re-encodes them at every query, where each demo image adds up to hundreds of visual tokens. Demo-free methods remove this cost with a compact task state. However, they add it at locations searched per task or at every decoder layer, where task parameters grow with depth. Moreover, inserted tokens or keys cannot change how the original prompt divides its attention within a layer. To address these issues, we propose Structured Task Adaptation via Embeddings (STAVE), which replaces demos with two task-specific vectors added to existing input embeddings. Specifically, a readout vector updates the answer-producing tokens and a context vector updates the other structural token groups. Both are trained with answer labels on prompts with and without demos. We justify these design choices theoretically using a first-order analysis of the loss and a margin bound. Extensive experiments on six LMMs and five large language models show that STAVE matches or outperforms state-of-the-art methods on multimodal tasks with far fewer task parameters and surpasses 15-shot ICL and prior task vectors on 18 text tasks, all at zero-shot inference cost.

## 1 Introduction

In-context learning (ICL) lets a frozen large language model (LLM) perform a new task from a few input–output demonstrations (demos) placed in its prompt (Brown et al., 2020; Dong et al., 2024). Large multimodal models (LMMs) inherit this ability (Alayrac et al., 2022; Laurençon et al., 2024b; Li et al., 2024b), so one frozen model can handle visual question answering (VQA) or image captioning without training. This flexibility is costly, since every query re-encodes its demos (Liu et al., 2022a) and each demo image expands into up to hundreds of visual tokens. Its accuracy also depends on which demos are chosen and in what order they appear (Zhao et al., 2021). We study how to adapt a frozen model to a task at the prompt length and cost of zero-shot inference.

Demo-free methods store the task in a compact state added to the frozen model at inference. Extracted task vectors take this state from demo activations. The task vector (TV) (Hendel et al., 2023) and the function vector (FV) (Todd et al., 2024) add one vector at an intermediate layer, while MTV (Huang et al., 2024) and STV (Ma et al., 2026) place it at attention heads selected for each task. Trained interventions instead optimize the state with supervision. LIVE (Peng et al., 2024) learns a shift vector for every decoder layer, and MimIC (Jiang et al., 2025) and HiFICL (Li et al., 2026) learn per-head shifts or virtual keys and values in every layer to approximate the effect of demos on the query. Prompt tuning (PT) (Lester et al., 2021) inserts learned virtual tokens into the input prompt.

These methods remove the demos at inference, yet two limitations remain. First, the task state enters either at a location searched for each task or at every decoder layer. A vector added at an intermediate layer cannot act through the blocks before it, and its layer or heads must be chosen for each task. Trained interventions avoid this choice by adding a shift to every layer, so their task parameters and per-query computation grow with model depth. Second, methods that insert tokens or keys, such as PT and HiFICL, cannot change how the original prompt divides its attention within a layer. For fixed queries and keys of the original tokens, added keys scale all their attention weights by one factor, so the relative attention between the image and the question stays unchanged (Petrov et al., 2024). PT must also choose where to insert its tokens, before the prompt, after it, or before its final token, and each inserted token lengthens the prompt and its computation.

To address these limitations, we ask two questions: where a demo-free update should enter a frozen model, and how it can be shared across tokens without losing answer-relevant directions. For the first, only an update at the input embeddings acts through every decoder block, and to first order an unrestricted input update can reproduce the first-answer effect of shifts in every layer. Added to existing tokens, the update keeps the prompt length and changes their attention queries and keys, so it can reweight the attention between the image and the question. For the second, structural token groups, such as the question text and the query image, recur in every prompt of a task, so one vector per group serves every query. Sharing has a cost: the final prompt token predicts the first answer token, whereas the other tokens reach it only through attention and still hold query-specific evidence, and a vector shared with many weak-gradient tokens, or with tokens whose gradients point in other directions, has a diluted first-order loss decrease.

Based on this analysis, we propose Structured Task Adaptation via Embeddings (STAVE), which adds two task-specific vectors to the input embeddings of a frozen LLM or LMM. A readout vector updates only the final prompt token and the answer cue, and a context vector covers the remaining groups. Both are trained with answer labels instead of an ICL teacher, on queries with and without demos, and added once to the demo-free prompt.

Contributions. ❶ Framework. We propose STAVE, which replaces in-context demos with a readout vector and a context vector added to the input embeddings of a frozen model. Its task parameters do not grow with model depth, and it runs at zero-shot inference cost. ❷ Theoretical Explanation. We show that only an input update reaches every decoder block, but one vector shared there is limited in norm and direction. Weak-gradient tokens such as image tokens dilute its norm, and its single direction cannot follow readout and context gradients that differ. At equal total norm, two separately normalized vectors never give a smaller firstorder loss decrease, and their advantage grows as these gradients diverge. We also bound the margin cost of tying groups and show when training with and without demos lowers the demo-free loss. ❸ Extensive Evaluations. On multimodal and text tasks, STAVE matches or outperforms per-layer interventions with far fewer task

![](images/31fb0b8755efe1fa07e96156d5a85fe1d8ea418ae61c530bb9d2c5f6695c6884.jpg)  
Figure 1: STAVE performance and efficiency. Bars are relative to STAVE, and params use a log scale (Appendix E.3).

parameters and surpasses ICL and task vectors without demos at inference (Figure 1). ❹ Insight. The gain comes from the readout/context separation, not a modality split or a second vector. The readout vector sets a shared answer format, the context vector adds query-specific evidence, and together they mainly elicit answers the model already holds.

## 2 Motivating Analysis

We answer the two questions of Section 1 for an additive prompt update, which changes a frozen decoder’s output without demos. Its reach decides where it should enter the model, and thefirst-order cost ofsharing decides how to share it across tokens without losing answer-relevant directions.

Proposition 2.1 (Reach of a prompt update). Fix a prompt and teacher-forced answer prefixes in a differentiable causal decoder with $N _ { \mathrm { l a y e r } }$ blocks and a positionwise final head. Depth zero denotes input embeddings and depth ℓ the output of block ℓ. (i) A prompt-only update at depth ℓ can act through later blocks and the head. After the last block, with stored keys and values unchanged, only thefinal prompt token can affect answer logits, and only the first prediction. (ii) If all intervening own-token Jacobians are invertible, any infinitesimal change of all depth-ℓ prompt states has a unique unrestricted per-token realization at an earlier depth.

Reach. Thus input injection can affect every block. By part (ii), an unrestricted per-prompt input update matches the first-answer effect of simultaneous per-layer shifts to first order (Appendix B.1).

Adding to existing tokens also keeps the sequence length. At a fixed attention layer, added keys scale all weights on the original tokens by one common factor when their queries, keys, and positional scores stay fixed, which preserves the ratios between them (Petrov et al., 2024). An embedding update can change those queries and keys and hence reorder the weights (Appendix B.2). However, a per-prompt update does not transfer to new queries, so reuse requires shared vectors.

Sharing. Fix a differentiable loss and let $g _ { t } \in \mathbb { R } ^ { d }$ be its gradient at input embedding t, where d is the hidden dimension. For a nonempty token set S and budget $\rho > 0$ , one vector shared as $w / \sqrt { | S | }$ has total update norm $\lVert \boldsymbol { w } \rVert _ { 2 }$ . Its largest

first-order loss decrease is

$$
\Gamma ( S ) = \frac { \rho } { \sqrt { | S | } } \left\| \sum _ { t \in S } g _ { t } \right\| _ { 2 } \leq \rho \left( \sum _ { t \in S } \| g _ { t } \| _ { 2 } ^ { 2 } \right) ^ { 1 / 2 } .\tag{1}
$$

Free per-token updates of equal total norm attain the right side. Equality holds only when all $g _ { t }$ agree. For disjoint nonempty A, B, two vectors at equal total budget attain $\sqrt { \Gamma ( A ) ^ { 2 } + \Gamma ( B ) ^ { 2 } } \geq \Gamma ( A \cup B )$ . The gap splits into a norm term, zero when the mean gradients of A and B have equal norm, and a direction term that grows as the cosine between their summed gradients falls (Proposition C.2(iii)). If the gradients on $\boldsymbol { B }$ vanish, one shared vector attains only $\sqrt { | \mathcal { A } | / | \mathcal { A } \cup \mathcal { B } | } \Gamma ( \mathcal { A } )$ (Proposition C.2). Figure 2 measures this in Idefics2 on Flowers before training (Appendix C.4). One vector on all tokens splits its update by token count, so image tokens, with almost no gradient, receive most of it, and the final token and answer cue, with 76% of the gradient, receive under 2% (Figure 2(a)).

Design. STAVE therefore separates a readout set A, the final prompt token and answer cue, from the remaining context tokens B. The final token supplies the first-token readout (Proposition 2.1(i)), and A has the largest mean gradient per token (Figure 2(b)), so a vector on $\mathcal { A }$ alone is not diluted by B. Context tokens reach the readout only through attention and have a descent direction nearly

![](images/525bbc990a4abd535ad612c9e92cf921fe7cc07422fbe1db4c41c72852bd9cea.jpg)  
Figure 2: One vector falls short and two do better.

orthogonal to that of A (Appendix C.4), so one shared vector cannot point along both. Along these directions, the held-out loss levels off with the readout vector alone, whereas each added context vector norm lowers the whole curve (Figure 2(c)). Section 3 builds both vectors from reusable structural token groups and bounds the margin cost of tying groups within each vector (Theorem 3.1).

## 3 Method

STAVE adds two task-specific vectors to existing embeddings of a frozen autoregressive model. A readout vector updates the final prompt token and answer cue, and a context vector updates the other eligible tokens. Both are trained with and without demos and used without them (Figure 3).

## 3.1 Structural embedding updates

Let $f _ { \theta }$ have frozen parameters θ and hidden dimension d. Prompt $p$ has input $E ( p ) \in \mathbb { R } ^ { T ( p ) \times d }$ , where $T ( p )$ counts tokens after modality expansion. A fixed, label-free map $s _ { t } ( p ) \in \{ 0 , \ldots , R \}$ assigns each token to a structural token group or to zero for exclusion. The $R = 7$ groups are LAST (final prompt token), ANSWER-CUE (template tokens introducing the answer before it), FIRST (designated first token), IMAGE-GATE (query-image delimiters), QUERY-IMAGE (expanded query-image embeddings), QUESTION-TEXT (remaining query text), and PREFIX-TEXT (instruction prefix).

Beginning-of-sequence (BOS) tokens, padding, answer tokens, and unmatched tokens are excluded. Demo tokens are also excluded, except FIRST when the template places it in a demo. FIRST is the first attended non-BOS token, or the first query-text token in captioning. Appendix C.1 gives the precedence and template conventions. Remaining tokens are eligible, and groups may be empty.

For $1 \leq q \leq R$ learned vectors, let $a ( k ) \in \{ 1 , \ldots , q \}$ assign group $k$ to a vector and set $a ( 0 ) = 0$ . Let $n _ { k } ( p )$ count group $k$ and $\begin{array} { r } { N _ { r } ( p ) = \sum _ { k : a ( k ) = r } n _ { k } ( p ) } \end{array}$ count all tokens assigned to vector r. Define

$$
\widetilde { E } ( p , W ) = E ( p ) + H ( p ) W , \qquad H _ { t r } ( p ) = \left\{ { \displaystyle { 1 \{ a ( s _ { t } ( p ) ) = r \} } / { \sqrt { N _ { r } ( p ) } } } , \quad N _ { r } ( p ) > 0 , \right.\tag{2}
$$

Here $W \in \mathbb { R } ^ { q \times d }$ holds one task’s vectors as rows, $H ( p ) \in \mathbb { R } ^ { T ( p ) \times q }$ distributes them, and 1 is the indicator. STAVE sets $q = 2 \colon$ LAST and ANSWER-CUE share the readout vector, and the other five groups share the context vector. Each vector is normalized once by its merged token count $N _ { r } ( p )$

Nonempty columns of $H ( p )$ have disjoint supports and unit norm. Hence $\| H ( p ) W \| _ { F } \leq \| W \| _ { F } ,$ with equality when $N _ { r } ( p ) > 0$ for every $r .$ Replication thus cannot grow the injected energy. At $W = 0 ;$ , the original computation is recovered exactly. For a differentiable loss $F ,$ the gradient of vector r is $\begin{array} { r } { N _ { r } ( p ) ^ { - 1 / 2 } \sum _ { t : a ( s _ { t } ( p ) ) = r } \nabla _ { \widetilde { E } _ { t } } F } \end{array}$ , where $\widetilde { E } _ { t }$ is the updated embedding of token $t . \ \mathrm { A t } \ W = 0 , \rho$ times its norm equals Γ of that vector’s token set in Equation 1 (Appendix C.2).

![](images/f2b70a54610dede68f20963491be30c64a8d91fb15574e3b8f034da3dd15656b.jpg)  
Figure 3: Illustration of the proposed framework STAVE in the multimodal setting.

## 3.2 Margin cost of tying groups

To compare tying with per-group updates, fix finitely many prompts and correct answer prefixes with equal group counts $n _ { k }$ . Scale each group’s vector by $1 / \sqrt { n _ { k } }$ and stack them into $v \in \mathbb { R } ^ { R d }$ , where empty groups have no effect. Equation 2 restricts v to a count-weighted subspace $\mathcal { U } _ { q }$ with orthogonal projector $P _ { q } .$ . With zero empty-row parameters, the learned-row norm is $\| v \| _ { 2 }$ (Appendix C.5).

Let $h _ { j } ( v )$ be correct-token minus competing-token logit margin $j ,$ for a fixed collection of $N \geq 1$ comparisons across these contexts. For radius $\rho > 0$ , define

$$
m _ { q } ( \rho ) = \operatorname* { m a x } _ { v \in \mathcal { U } _ { q } , \| v \| _ { 2 } \leq \rho } \operatorname* { m i n } _ { \substack { 1 \leq j \leq N } } h _ { j } ( v ) , \qquad \delta _ { q } = \operatorname* { m a x } _ { j } \| ( I - P _ { q } ) \nabla h _ { j } ( 0 ) \| _ { 2 } .\tag{3}
$$

Here $m _ { q }$ is the best worst-case margin in the tied class, I is the identity, and $\delta _ { q }$ measures sensitivity to discarded group contrasts. Define $m _ { R }$ using the unrestricted group-coordinate ball.

Theorem 3.1 (Cost of tying structural token groups). Under the common-count setup above, suppose each $h _ { j }$ is differentiable on a neighborhood of $\{ v : \| v \| _ { 2 } \overset { } { \leq } \rho \rbrace $ and its gradient is κ-Lipschitz on that ball, with $\kappa \geq 0$ . Then

$$
0 \leq m _ { R } ( \rho ) - m _ { q } ( \rho ) \leq \rho \delta _ { q } + \kappa \rho ^ { 2 } .\tag{4}
$$

For affine margins, $\delta _ { q } = 0$ suffices for equal attainable margins.

The proof projects an unrestricted update onto $\mathcal { U } _ { q }$ and bounds the omitted linear response and curvature. For readout/- context sharing, $\delta _ { q }$ measures count-weighted gradient variation within the two sets, and their mutual contrast is kept. Small $\delta _ { q }$ and κ therefore give a small margin cost.

Equation 2 and the objective are defined for any group counts. When group proportions vary across prompts, the theorem applies within each common-count collection.

## 3.3 Paired supervision

For each training query $x _ { i }$ with answer $y _ { i }$ , render $p _ { i } ^ { \mathrm { s r c } } = \mathrm { r e n d e r } ( D _ { i } , x _ { i } )$ with demos $D _ { i }$ and $p _ { i } ^ { \mathrm { t g t } } = \mathrm { r e n d e r } ( \emptyset , x _ { i } )$ without them. Both retain the query modality. For answer $y = ( a _ { 1 } , \dotsc , a _ { L } )$ of length $L \geq 1$ , use

$$
\ell ( y \mid p , W ) = - \frac { 1 } { L } \sum _ { l = 1 } ^ { L } \log p _ { \theta } ( a _ { l } \mid \widetilde { E } ( p , W ) , a _ { < l } ) ,\tag{5}
$$

$$
{ \mathcal { L } } ( W ) = { \mathcal { L } } _ { \mathrm { s r c } } ( W ) + { \mathcal { L } } _ { \mathrm { t g t } } ( W ) , \qquad { \mathcal { L } } _ { b } ( W ) = { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } \ell ( y _ { i } \mid p _ { i } ^ { b } , W ) .\tag{6}
$$

Here $p _ { \theta }$ is the frozen next-token distribution, $a _ { < l }$ is the correct prefix, n counts training query–answer pairs, and $b \in \{ \mathrm { s r c } , \mathrm { t g t } \}$ selects a branch. The two sequence cross-entropy (CE) losses are equally weighted. Both branches of

each paired minibatch contribute before an optimizer step. The vectors start at zero, and backbone parameters and answer-token input embeddings get no direct update.

The target branch supervises deployment inputs. The source branch fits the same vectors with demos present, using answer labels rather than soft labels or hidden-state targets. This adds a compatibility requirement, and Proposition 3.2 bounds its effect on the target loss after one gradient step.

Proposition 3.2 (Target descent under paired supervision). Stack the learned vectors into $w \in \mathbb { R } ^ { q d }$ and write $g _ { b } = \nabla _ { w } \mathcal { L } _ { b } ( w )$ and $g = g _ { \mathrm { s r c } } + g _ { \mathrm { t g t } } . \mathrm { I f } \nabla \mathcal { L } _ { \mathrm { t g t } }$ is β-Lipschitz on the segment from w to $w - \eta g$ , where $\beta \geq 0$ and $\eta > 0 .$ , then

$$
\mathcal { L } _ { \mathrm { t g t } } ( w - \eta g ) - \mathcal { L } _ { \mathrm { t g t } } ( w ) \leq - \eta \langle g _ { \mathrm { t g t } } , g \rangle + \frac { \beta \eta ^ { 2 } } { 2 } \| g \| _ { 2 } ^ { 2 } .\tag{7}
$$

Thus $\| g _ { \mathrm { t g t } } \| _ { 2 } ^ { 2 } + \langle g _ { \mathrm { t g t } } , g _ { \mathrm { s r c } } \rangle > 0$ gives target descent for sufficiently small gradient steps within the smoothness region. Positive source–target alignment also improves the first-order decrease relative to a target-only step. Appendix D.1 proves both statements and gives finite-step conditions.

For the whole objective, restrict $W$ to a Frobenius ball and define the source–target compatibility gap $\mathcal { C } _ { \mathrm { s r c | t g t } }$ as the smallest excess source loss among target minimizers. Any $\widehat { W }$ within $\varepsilon _ { \mathrm { o p t } } \geq 0$ of min L then has excess target loss at most $\mathcal { C } _ { \mathrm { s r c | t g t } } + \varepsilon _ { \mathrm { o p t } }$ , and the gap is zero when the two branches share a minimizer (Appendix D). Under independent target sampling and bounded-loss and Lipschitz assumptions, Appendix D.3 adds a uniform estimation term with covering dimension $q d = 2 d$ . This separates representation size from compatibility, optimization, and statistical error.

## 3.4 Demo-free inference and cost

In LLMs, STAVE updates frozen input embeddings. In LMMs, it updates merged text and image embeddings after the frozen visual encoder and connector. At inference, add both vectors once during demo-free prefill. Generated embeddings receive no update. Storage is 2d task-specific scalars, independent of query count and demo length. Group counts are built from template metadata in $O ( T ( \bar { p ) } )$ time, and adding the scaled vectors costs $O ( T ( p ) d )$ operations. Prompt length and decoder prefill dimensions equal those ofzero-shot inference, as does key–value (KV) cache shape at a fixed generated length. Training is a one-time cost: both branches run the frozen model and backpropagate to the two vectors, with no backbone optimizer states or weight updates (Appendix E).

## 4 Experiments

## 4.1 Setup

Datasets, backbones, and protocols. We evaluate six frozen LMMs, LLaVA-Interleave-7B (Li et al., 2024b), Idefics2-8B (Laurençon et al., 2024b), Qwen-VL-7B (Bai et al., 2023), InternVL3.5-8B (Wang et al., 2025), Qwen2.5- VL-7B (Bai et al., 2025), and Idefics3-8B (Laurençon et al., 2024a). The tasks are VQA with VQAv2 (Goyal et al., 2017), OK-VQA (Marino et al., 2019), and VizWiz (Gurari et al., 2018), image captioning with the COCO Karpathy split (Lin et al., 2014; Karpathy & Fei-Fei, 2015), and fine-grained classification with DTD (Cimpoi et al., 2014), Flowers (Nilsback & Zisserman, 2008), and CUB (Wah et al., 2011). We also evaluate five frozen LLMs, Pythia 2.8B, 6.9B, and 12B (Biderman et al., 2023), GPT-J 6B (Wang & Komatsuzaki, 2021), and LLaMA 7B (Touvron et al., 2023), on the TV benchmark, the 18 algorithmic, translation, linguistic, and knowledge tasks of Hendel et al. (2023). Each comparison follows the most common protocol for its benchmark. Table 1 follows MimIC (Jiang et al., 2025) and HiFICL (Li et al., 2026), with 1,000 training samples that also form the demo pool, evaluation on the validation split, and results from the best epoch. Tables 2 and 10 in the appendix follow MTV (Huang et al., 2024) and STV (Ma et al., 2026), with a held-out test split. Table 3 follows Hendel et al. (2023), with at most 256 training, 64 development, and 200 test queries per task. All trained methods answer without demos at test time. COCO is scored by CIDEr (Vedantam et al., 2015) and all other tasks by accuracy.

Compared methods. Besides zero-shot and n-shot ICL, we compare STAVE with three families of methods. Parameter efficient fine-tuning includes LoRA (Hu et al., 2021), prefix tuning (Li & Liang, 2021), and PT (Lester et al., 2021) in three modes, with its tokens before the prompt (PT-Pre), after it (PT-App), or before the final token (PT-BL). Task vectors include TV (Hendel et al., 2023), FV (Todd et al., 2024), ICV (Liu et al., 2024b), I2CL (Li et al., 2025b), MTV, STV, and SITE (Park et al., 2026), whose values come from demo activations, and the last four learn which heads, values, or scales to use. Trained per-layer interventions include LIVE (Peng et al., 2024), MimIC, and HiFICL. The three families differ in where they act. Fine-tuning updates weights or adds virtual tokens, task vectors inject demo activations at one layer, in every layer, or at selected heads, and trained interventions learn parameters inside every

<table><tr><td>LLaVA</td><td># Params (M)</td><td>VQAv2</td><td>OK-VQA</td><td>COCO</td></tr><tr><td>Zero-shot</td><td></td><td> $2 7 . 6 5 _ { \pm 0 . 0 0 }$ </td><td> $1 2 . 9 3 _ { \pm 0 . 0 0 }$ </td><td> $1 0 7 . 2 5 _ { \pm 0 . 0 0 }$ </td></tr><tr><td>8-shot ICL</td><td></td><td> $6 8 . 6 1 _ { \pm 0 . 2 0 }$ </td><td> $4 8 . 4 3 _ { \pm 0 . 4 2 }$ </td><td> $1 1 8 . 8 8 _ { \pm 0 . 7 7 }$ </td></tr><tr><td>16-shot ICL</td><td></td><td> $6 6 . 2 1 \pm 0 . 4 8$ </td><td> $4 4 . 4 7 _ { \pm 0 . 4 6 }$ </td><td> $1 2 4 . 5 2 { \scriptstyle \pm 1 . 0 5 }$ </td></tr><tr><td>32-shot ICL</td><td></td><td> $6 2 . 6 0 { \scriptstyle \pm 1 . 0 1 }$ </td><td> $3 9 . 3 9 _ { \pm 0 . 5 2 }$ </td><td> $1 2 4 . 5 6 { \scriptstyle \pm 0 . 7 3 }$ </td></tr><tr><td>LoRA</td><td>19.7 (×2399.00)</td><td> $7 4 . 6 5 _ { \pm 0 . 4 7 }$ </td><td> $5 3 . 0 8 { \scriptstyle \pm 0 . 4 1 }$ </td><td> $1 2 9 . 4 0 { \scriptstyle \pm 1 . 0 1 }$ </td></tr><tr><td>PT-Pre</td><td>0.082 (× 10.00)</td><td> $7 4 . 7 3 { \scriptstyle \pm 0 . 0 9 }$ </td><td> $5 4 . 2 2 _ { \pm 0 . 1 7 }$ </td><td> $1 2 9 . 7 2 { \scriptstyle \pm 0 . 0 8 }$ </td></tr><tr><td>PT-App</td><td>0.082 (×10.00)</td><td> $7 4 . 6 8 _ { \pm 0 . 2 4 }$ </td><td> $5 3 . 4 6 _ { \pm 0 . 3 3 }$ </td><td> $1 2 7 . 9 8 _ { \pm 0 . 8 0 }$ </td></tr><tr><td>PT-BL</td><td>0.082 (×10.00)</td><td> $7 4 . 4 7 { \scriptstyle \pm 0 . 2 1 }$ </td><td> $5 3 . 9 1 _ { \pm 0 . 5 0 }$ </td><td> $1 2 8 . 3 3 { \scriptstyle \pm 0 . 4 7 }$ </td></tr><tr><td>LIVE</td><td> $0 . 1 3 \left( \times 1 6 . 0 0 \right)$ </td><td> $7 5 . 1 5 _ { \pm 0 . 0 3 }$ </td><td> $5 3 . 6 9 _ { \pm 0 . 1 9 }$ </td><td> $1 2 9 . 7 5 _ { \pm 0 . 4 0 }$ </td></tr><tr><td>MimIC</td><td>17.0 (×2080.13)</td><td> $7 5 . 1 9 _ { \pm 0 . 1 6 }$ </td><td> $5 5 . 7 4 { \scriptstyle \pm 0 . 2 5 }$ </td><td> $1 2 8 . 0 9 _ { \pm 0 . 2 5 }$ </td></tr><tr><td>HiFICL</td><td> $2 . 2 \ : ( \times 2 7 2 . 0 0 )$ </td><td> $7 5 . 3 6 { \scriptstyle \pm 0 . 3 0 }$ </td><td> $5 4 . 5 0 { \scriptstyle \pm 0 . 5 2 }$ </td><td> $\underline { { 1 3 0 . 2 4 } } \pm \mathrm { 0 . 3 2 }$ </td></tr><tr><td>STAVE</td><td> $\mathbf { 0 . 0 0 8 } \left( \mathbf { \times 1 . 0 0 } \right)$ </td><td> ${ \bf 7 6 . 0 7 { \scriptstyle \pm 0 . 1 0 } }$ </td><td> ${ \bf 5 5 . 8 1 { _ { \pm 0 . 1 0 } } }$ </td><td> $\mathbf { 1 3 0 . 5 3 { \scriptstyle \pm 1 . 1 7 } }$ </td></tr><tr><td>Idefics2-8B</td><td># Params (M)</td><td>VQAv2</td><td> $\mathrm { O K ^ { - } V Q A }$ </td><td>COCO</td></tr><tr><td>Zero-shot</td><td></td><td>47.61±0.00</td><td> $2 9 . 5 2 _ { \pm 0 . 0 0 }$ </td><td> $8 0 . 5 4 _ { \pm 0 . 0 0 }$ </td></tr><tr><td>8-shot ICL</td><td></td><td> $6 7 . 4 4 { \scriptstyle \pm 0 . 3 0 }$ </td><td> $5 7 . 3 5 _ { \pm 0 . 2 0 }$ </td><td> $1 1 9 . 2 5 _ { \pm 0 . 4 2 }$ </td></tr><tr><td>16-shot ICL</td><td></td><td> $6 7 . 5 6 { \scriptstyle \pm 0 . 7 4 }$ </td><td> $5 6 . 2 4 _ { \pm 0 . 4 6 }$ </td><td> $1 2 9 . 5 3 _ { \pm 0 . 4 4 }$ </td></tr><tr><td>32-shot ICL</td><td></td><td> $5 3 . 7 7 _ { \pm 0 . 7 6 }$ </td><td> $5 2 . 0 5 _ { \pm 1 . 1 7 }$ </td><td> $1 2 2 . 3 6 _ { \pm 0 . 3 2 }$ </td></tr><tr><td>LoRA</td><td>17.6 (×2147.00)</td><td> $7 1 . 1 1 { \scriptstyle \pm 0 . 8 7 }$ </td><td> $5 9 . 2 3 _ { \pm 0 . 2 1 }$ </td><td> $\underline { { 1 3 2 . 8 4 } } { \scriptstyle \pm 0 . 5 7 }$ </td></tr><tr><td>PT-Pre</td><td>0.082 (× 10.00)</td><td>72.75±0.14</td><td> $5 9 . 2 0 { \scriptstyle \pm 0 . 4 4 }$ </td><td>131.03±1.51</td></tr><tr><td> $\mathrm { P T - A p p }$ </td><td>0.082 (× 10.00)</td><td> $7 0 . 7 5 { \scriptstyle \pm 0 . 5 6 }$ </td><td> $5 5 . 8 3 { \scriptstyle \pm 1 . 7 8 }$ </td><td> $1 3 1 . 2 0 { \scriptstyle \pm 1 . 5 9 }$ </td></tr><tr><td>PT-BL</td><td>0.082 (× 10.00)</td><td> $7 1 . 0 4 { \scriptstyle \pm 0 . 3 8 }$ </td><td> $5 7 . 8 2 _ { \pm 0 . 8 4 }$ </td><td> $1 3 1 . 1 3 { \scriptstyle \pm 1 . 1 2 }$ </td></tr><tr><td>LIVE</td><td>0.13 (×16.00)</td><td> $6 9 . 7 3 { \scriptstyle \pm 1 . 3 0 }$ </td><td> $5 8 . 2 2 _ { \pm 0 . 4 0 }$ </td><td> $1 2 8 . 2 0 { \scriptstyle \pm 0 . 8 4 }$ </td></tr><tr><td>MimIC</td><td>0.26 (×32.13)</td><td> $7 1 . 3 4 _ { \pm 0 . 0 5 }$ </td><td> $5 9 . 6 9 _ { \pm 0 . 3 7 }$ </td><td> $1 3 2 . 4 2 _ { \pm 0 . 1 3 }$ </td></tr><tr><td>HiFICL</td><td>2.2 (×272.00)</td><td> $7 1 . 6 1 { \scriptstyle \pm 0 . 6 8 }$ </td><td> $\underline { { 6 0 . 2 9 } } { \scriptstyle \pm 0 . 5 3 }$ </td><td> $1 2 8 . 6 9 _ { \pm 0 . 1 4 }$ </td></tr><tr><td>STAVE</td><td>0.008 (×1.00)</td><td> $7 3 . 4 3 _ { \pm 0 . 0 6 }$ </td><td> ${ \bf 6 1 . 0 0 _ { \pm 0 . 4 1 } }$ </td><td> $\mathbf { 1 3 4 . 1 8 _ { \pm 0 . 8 2 } }$ </td></tr></table>

Table 1: Comparison on VQAv2, OK-VQA, and COCO (left), with qualitative examples (right). # Params: low , medium , high . Best: bold, second: underlined.  
![](images/ddd5397d5df5092a786fc2ca515688f52fd257ec21a945de637d5ad3aa3f89c9.jpg)

![](images/bcf8e8fae083420a0dbc0d9ec7e03d2bc810a6f45d1008e703d3464b918012e7.jpg)

![](images/e8c7c6545b159a9699270c525768a5f527c174a51e477b6473ed9d628c887110.jpg)

Table 2: Comparison on VQA and classification datasets.
<table><tr><td></td><td colspan="2"></td><td colspan="4">Idefics2-8B</td><td colspan="2"></td><td colspan="2">Qwen-VL-7B</td><td></td><td></td><td></td></tr><tr><td>Method</td><td># Params (M)</td><td>VizWiz</td><td>OK-VQA</td><td>DTD</td><td>Flowers</td><td>CUB</td><td>Avg.</td><td>VizWiz</td><td>OK-VQA</td><td>DTD</td><td>Flowers</td><td>CUB</td><td>Avg.</td></tr><tr><td>Zero-shot</td><td>=</td><td> $3 7 . 7 4 _ { \pm 0 . 0 0 }$ </td><td>49.44±0.00</td><td> $9 0 . 2 3 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $8 7 . 5 9 _ { \pm 0 . 0 0 }$ </td><td>87.84±0.00</td><td>70.57±0.00</td><td> $4 5 . 4 9 _ { \pm 0 . 0 0 }$ </td><td>57.51±0.00</td><td> $8 3 . 7 7 _ { \pm 0 . 0 0 }$ </td><td>73.88±0.00</td><td> $9 1 . 1 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td>70.35±0.00</td></tr><tr><td>4-shot ICL</td><td>=</td><td> $4 3 . 2 4 _ { \pm 0 . 1 1 }$ </td><td> $5 0 . 3 8 { \scriptstyle \pm 0 . 1 9 }$ </td><td> $8 8 . 8 7 _ { \pm 0 . 1 2 }$ </td><td> $8 0 . 2 2 _ { \pm 0 . 4 0 }$ </td><td> $8 5 . 9 1 _ { \pm 0 . 1 2 }$ </td><td> $6 9 . 7 2 _ { \pm 0 . 0 5 }$ </td><td> $4 6 . 3 6 { \scriptstyle \pm 0 . 3 2 }$ </td><td> $6 0 . 9 2 _ { \pm 0 . 0 7 }$ </td><td> $8 4 . 4 4 _ { \pm 0 . 2 5 }$ </td><td> $8 7 . 1 7 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $8 8 . 7 3 _ { \pm 0 . 2 5 }$ </td><td> $7 3 . 5 3 _ { \pm 0 . 1 2 }$ </td></tr><tr><td>PT-Pre</td><td>0.082 (×10.00)</td><td> $6 6 . 1 9 _ { \pm 1 . 2 6 }$ </td><td> $5 4 . 4 9 _ { \pm 2 . 5 7 }$ </td><td> $9 4 . 6 4 \pm 0 . 3 5$ </td><td> $8 7 . 5 0 { \scriptstyle \pm 0 . 4 1 }$ </td><td> $9 4 . 2 5 _ { \pm 0 . 5 3 }$ </td><td> $7 9 . 4 1 _ { \pm 0 . 7 7 }$ </td><td> $6 3 . 8 3 _ { \pm 0 . 8 2 }$ </td><td> $5 0 . 5 6 { \scriptstyle \pm 0 . 7 3 }$ </td><td> $9 4 . 4 4 _ { \pm 0 . 8 5 }$ </td><td> $9 3 . 8 3 _ { \pm 0 . 2 6 }$ </td><td> $9 5 . 6 5 _ { \pm 0 . 7 1 }$ </td><td> $7 9 . 6 6 _ { \pm 0 . 2 9 }$ </td></tr><tr><td>PT-App</td><td> $0 . 0 8 2 \left( \times 1 0 . 0 0 \right)$ </td><td> $6 4 . 5 6 { \scriptstyle \pm 2 . 4 3 }$ </td><td> $5 5 . 1 0 { \scriptstyle \pm 0 . 5 6 }$ </td><td> $9 4 . 1 4 { \scriptstyle \pm 0 . 0 7 }$ </td><td> $8 9 . 6 0 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $9 5 . 1 0 { \scriptstyle \pm 0 . 1 5 }$ </td><td> $7 9 . 7 0 { \scriptstyle \pm 0 . 6 3 }$ </td><td> $6 0 . 1 9 { \scriptstyle \pm 3 . 2 6 }$ </td><td> $3 7 . 8 4 { \scriptstyle \pm 4 . 1 1 }$ </td><td> $9 3 . 1 0 { \scriptstyle \pm 0 . 9 6 }$ </td><td> $9 3 . 5 6 { \scriptstyle \pm 0 . 2 8 }$ </td><td> $9 4 . 1 4 { \scriptstyle \pm 0 . 4 3 }$ </td><td> $7 5 . 7 7 { \scriptstyle \pm 1 . 0 3 }$ </td></tr><tr><td>PT-BL</td><td>0.082 (× 10.00)</td><td> $6 6 . 3 5 { \scriptstyle \pm 0 . 2 1 }$ </td><td> $5 6 . 5 8 { \scriptstyle \pm 0 . 9 8 }$ </td><td> $9 4 . 1 3 _ { \pm 0 . 1 9 }$ </td><td> $8 8 . 9 7 _ { \pm 0 . 5 2 }$ </td><td> $9 4 . 5 5 { \scriptstyle \pm 1 . 5 4 }$ </td><td> $8 0 . 1 2 _ { \pm 0 . 2 2 }$ </td><td> $6 0 . 3 0 { \scriptstyle \pm 4 . 6 4 }$ </td><td> $4 4 . 3 5 { \scriptstyle \pm 3 . 6 1 }$ </td><td> $9 3 . 4 4 _ { \pm 0 . 2 8 }$ </td><td> $9 3 . 4 9 _ { \pm 0 . 2 9 }$ </td><td> $9 4 . 7 3 { \scriptstyle \pm 0 . 5 0 }$ </td><td> $7 7 . 2 6 _ { \pm 0 . 4 8 }$ </td></tr><tr><td>TV</td><td> $\mathbf { 0 . 0 0 4 } \left( \mathbf { \times 0 . 5 0 } \right)$ </td><td> $4 6 . 2 9 _ { \pm 6 . 9 0 }$ </td><td> $5 0 . 2 3 _ { \pm 0 . 9 0 }$ </td><td> $9 0 . 5 8 { \scriptstyle \pm 0 . 4 9 }$ </td><td> $8 7 . 5 2 _ { \pm 0 . 0 2 }$ </td><td> $8 7 . 9 9 _ { \pm 0 . 2 6 }$ </td><td> $7 2 . 5 2 { \scriptstyle \pm 1 . 4 1 }$ </td><td> $4 9 . 3 7 _ { \pm 2 . 4 1 }$ </td><td> $5 9 . 0 3 _ { \pm 0 . 6 8 }$ </td><td> $9 1 . 0 0 { \scriptstyle \pm 1 . 8 0 }$ </td><td> $9 3 . 5 9 _ { \pm 0 . 0 6 }$ </td><td> $9 2 . 6 9 { \scriptstyle \pm 1 . 3 6 }$ </td><td> $7 7 . 1 4 _ { \pm 0 . 8 3 }$ </td></tr><tr><td>ICV</td><td> $0 . 1 3 \ : ( \times 1 6 . 0 0 )$ </td><td> $3 7 . 7 5 _ { \pm 0 . 0 2 }$ </td><td> $4 9 . 4 9 _ { \pm 0 . 0 3 }$ </td><td> $9 0 . 3 0 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $8 7 . 8 0 _ { \pm 0 . 3 5 }$ </td><td> $8 7 . 8 6 _ { \pm 0 . 0 2 }$ </td><td> $7 0 . 6 4 _ { \pm 0 . 0 7 }$ </td><td></td><td> $5 7 . 8 1 _ { \pm 0 . 3 9 }$ </td><td> $9 0 . 0 7 _ { \pm 0 . 1 2 }$ </td><td> $9 2 . 2 2 _ { \pm 0 . 0 2 }$ </td><td> $9 1 . 5 1 { \scriptstyle \pm 0 . 0 1 }$ </td><td> $7 5 . 4 4 _ { \pm 0 . 0 8 }$ </td></tr><tr><td>I2CL</td><td> $0 . 2 6 \left( \times 3 2 . 0 2 \right)$ </td><td> $4 2 . 1 5 _ { \pm 5 . 8 7 }$ </td><td> $4 7 . 4 1 _ { \pm 1 . 9 9 }$ </td><td></td><td>92.59±0.86 89.25±0.32</td><td> $9 3 . 4 0 _ { \pm 1 . 7 3 }$ </td><td>72.96±1.01</td><td> $4 5 . 5 7 _ { \pm 0 . 1 0 }$   $5 3 . 2 2 { \scriptstyle \pm 2 . 0 4 }$ </td><td> $5 1 . 5 5 { \scriptstyle \pm 3 . 0 2 }$ </td><td> $9 0 . 7 7 { \scriptstyle \pm 1 . 6 2 }$ </td><td>93.75±0.25</td><td> $9 3 . 3 6 _ { \pm 0 . 9 7 }$ </td><td> $7 6 . 5 3 { \scriptstyle \pm 0 . 2 6 }$ </td></tr><tr><td>MTV</td><td> $0 . 1 3 \overset { ^ { \cdot } } { ( } \times 1 6 . 0 0 \overset { ^ { \cdot } } { ) }$ </td><td> $5 0 . 8 5 _ { \pm 4 . 9 4 }$ </td><td> $5 0 . 3 0 { \scriptstyle \pm 0 . 2 4 }$ </td><td> $9 3 . 7 3 _ { \pm 0 . 2 6 }$ </td><td> $8 9 . 4 6 _ { \pm 0 . 1 7 }$ </td><td>94.15±0.72</td><td> $7 5 . 7 0 { \scriptstyle \pm 1 . 0 6 }$ </td><td>52.53±6.36 57.68±1.17 94.27±0.21</td><td></td><td></td><td> $9 3 . 7 3 _ { \pm 0 . 3 0 }$ </td><td>94.66±0.15</td><td>78.57±1.35</td></tr><tr><td>STV</td><td> $\mathbf { 0 . 0 0 8 \left( \times 1 . 0 0 \right) }$ </td><td> $6 1 . 3 9 { \scriptstyle \pm 0 . 2 0 }$ </td><td>49.44±0.00</td><td> $9 1 . 5 7 { \scriptstyle \pm 1 . 5 3 }$ </td><td> $8 7 . 8 3 { \scriptstyle \pm 0 . 4 1 }$ </td><td> $9 3 . 3 8 _ { \pm 1 . 0 1 }$ </td><td> $7 6 . 7 2 { \scriptstyle \pm 0 . 3 5 }$ </td><td> $5 8 . 1 5 { \scriptstyle \pm 0 . 9 0 }$ </td><td> $5 8 . 1 7 _ { \pm 0 . 7 5 } ^ { - }$ </td><td> $7 6 . 3 2 { \scriptstyle \pm 8 . 9 0 }$ </td><td> $8 8 . 6 4 \pm 5 . 4 7 $ </td><td> $9 1 . 1 0 { \stackrel { - } { \pm } } 0 . 0 0$ </td><td> $7 4 . 4 8 { \scriptstyle \pm 3 . 1 3 }$ </td></tr><tr><td>LIVE</td><td> $0 . 1 3 \ : ( \times 1 6 . 0 0 )$ </td><td> $5 9 . 7 4 _ { \pm 0 . 4 5 }$ </td><td> $5 4 . 1 2 _ { \pm 0 . 1 7 }$ </td><td> $9 3 . 6 9 _ { \pm 0 . 1 1 }$ </td><td> $8 7 . 8 0 { \scriptstyle \pm 0 . 9 8 }$ </td><td> $9 3 . 3 6 _ { \pm 0 . 4 9 }$ </td><td> $7 7 . 7 4 _ { \pm 0 . 2 5 }$ </td><td> $6 2 . 4 0 _ { \pm 0 . 4 4 }$ </td><td> $5 3 . 8 5 _ { \pm 0 . 2 7 }$ </td><td> $7 8 . 9 1 _ { \pm 0 . 1 0 }$ </td><td> $8 6 . 3 3 _ { \pm 0 . 2 2 }$ </td><td> $8 6 . 8 7 _ { \pm 0 . 4 1 }$ </td><td> $7 3 . 6 7 _ { \pm 0 . 1 4 }$ </td></tr><tr><td>MimIC</td><td> $0 . 2 6 \overset { cdot } { ( } \times 3 2 . 1 3 \overset { . } { ) }$ </td><td> $6 8 . 4 6 _ { \pm 1 . 7 2 }$ </td><td> $5 9 . 3 0 { \scriptstyle \pm 0 . 7 0 }$ </td><td> $9 4 . 6 1 _ { \pm 0 . 6 0 }$ </td><td> $9 0 . 4 5 _ { \pm 3 . 5 4 }$ </td><td> $9 3 . 3 0 { \scriptstyle \pm 1 . 2 2 }$ </td><td> $8 1 . 2 2 { \scriptstyle \pm 0 . 3 5 }$ </td><td> $6 7 . 8 3 _ { \pm 0 . 8 9 } ^ { - }$ </td><td> $5 5 . 2 1 _ { \pm 0 . 9 5 }$ </td><td> $9 5 . 1 6 _ { \pm 0 . 5 2 }$ </td><td> $9 5 . 7 9 _ { \pm 1 . 0 0 }$ </td><td> $9 6 . 3 5 { \scriptstyle \pm 0 . 4 7 }$ </td><td> $8 2 . 0 7 _ { \pm 0 . 5 4 }$ </td></tr><tr><td>HiFICL</td><td> $2 . 2 \left( \times 2 7 2 . 0 0 \right)$ </td><td> $6 5 . 5 6 { \scriptstyle \pm 2 . 0 5 }$ </td><td> $5 9 . 2 4 { \scriptstyle \pm 1 . 1 9 }$ </td><td> $9 2 . 2 9 { \scriptstyle \pm 0 . 7 5 }$ </td><td> $9 1 . 9 3 { \scriptstyle \pm 0 . 7 3 }$ </td><td>95.18±0.39</td><td> $8 0 . 8 4 { \scriptstyle \pm 0 . 7 4 }$ </td><td> $6 7 . 1 0 { \scriptstyle \pm 1 . 4 4 }$ </td><td> $5 5 . 7 8 { \scriptstyle \pm 1 . 2 7 }$ </td><td> $\mathbf { 9 5 . 4 7 { \scriptstyle \pm 0 . 2 7 } }$ </td><td> $\mathbf { 9 7 . 2 9 4 0 . 5 6 }$ </td><td> $\mathbf { 9 7 . 0 5 _ { \pm 0 . 3 4 } }$ </td><td> $\underline { { 8 2 . 5 4 } } \pm \mathrm { 0 . 2 5 }$ </td></tr><tr><td>STAVE</td><td>0.008 (×1.00)</td><td> ${ \bf 6 9 . 6 3 _ { \pm 0 . 3 7 } }$ </td><td> ${ \bf 5 9 . 5 9 4 0 . 5 0 }$ </td><td> $\mathbf { 9 5 . 3 6 _ { \pm 0 . 4 7 } }$ </td><td> $\mathbf { 9 5 . 6 7 _ { \pm 0 . 4 2 } }$ </td><td> $9 3 . 5 0 { \scriptstyle \pm 0 . 7 2 }$ </td><td> $\mathbf { 8 2 . 7 5 _ { \pm 0 . 1 9 } }$ </td><td> ${ \bf 6 8 . 6 0 _ { \pm 0 . 5 5 } }$ </td><td> ${ \bf 6 2 . 3 9 2 0 . 1 2 }$ </td><td> $9 5 . 2 0 { \scriptstyle \pm 0 . 3 2 }$ </td><td> $9 6 . 3 0 { \scriptstyle \pm 0 . 2 3 }$ </td><td> $9 5 . 5 6 { \scriptstyle \pm 0 . 2 6 }$ </td><td> $\mathbf { 8 3 . 6 1 _ { \pm 0 . 1 4 } }$ </td></tr></table>

decoder layer. STAVE instead trains two vectors that are added once to the input embeddings, without changing the prompt length or adding computation inside the decoder layers.

## 4.2 Implementation details

STAVE learns only the two rows of W, which start at zero in single precision, while the backbone stays frozen in half precision. Each source prompt holds one to eight demos. We optimize the vectors with AdamW (Loshchilov & Hutter, 2017) under a cosine learning-rate schedule. Generally, each training step uses $K = 4$ paired examples and a learning rate of $1 0 ^ { - 3 }$ on LMM tasks, and K = 8 and $1 0 ^ { - 1 }$ on LLM tasks. We run all compared methods from their released code with the evaluation data, prompts, metrics, and decoding of each protocol. All main experiments are run three times, and we report the mean and standard deviation of the three runs. Additional details, including backbones, protocol settings, hyperparameter sensitivity, and prompt examples, are reported in Appendix F.

Table 3: Comparison on the TV benchmark (18 text tasks).
<table><tr><td></td><td></td><td colspan="3">Vanilla ICL</td><td></td><td></td><td colspan="3">PT</td><td colspan="3"></td><td>一</td></tr><tr><td>Model</td><td>Zero-shot</td><td> $n { = } 1$ </td><td> $n { = } 5$ </td><td> $n { = } 1 5$ </td><td>LoRA</td><td>Prefix</td><td>Pre</td><td>App</td><td>BL</td><td>FV</td><td>TV</td><td>SITE</td><td>STAVE</td></tr><tr><td>Pythia 2.8B</td><td> $1 3 . 1 _ { \pm 0 . 0 }$ </td><td> $6 0 . 4 _ { \pm 0 . 6 }$ </td><td> $8 5 . 6 { \scriptstyle \pm 0 . 4 }$ </td><td> $9 0 . 6 { \scriptstyle \pm 0 . 2 }$ </td><td> $7 9 . 1 _ { \pm 5 . 1 }$ </td><td> $6 1 . 8 { \scriptstyle \pm 1 . 6 }$ </td><td> $7 0 . 4 { \scriptstyle \pm 5 . 7 }$ </td><td> $7 9 . 4 { \scriptstyle \pm 6 . 7 }$ </td><td> $7 8 . 4 { \scriptstyle \pm 5 . 2 }$ </td><td> $4 0 . 7 _ { \pm 2 . 1 }$ </td><td> $7 5 . 6 { \scriptstyle \pm 0 . 1 }$ </td><td> $8 9 . 9 { \scriptstyle \pm 0 . 2 }$ </td><td> ${ \bf 9 1 . 2 _ { \pm 0 . 3 } }$ </td></tr><tr><td>Pythia 6.9B</td><td> $1 0 . 9 { \scriptstyle \pm 0 . 0 }$ </td><td> $6 5 . 0 { \scriptstyle \pm 0 . 9 }$ </td><td> $8 6 . 6 { \scriptstyle \pm 0 . 5 }$ </td><td> $9 0 . 6 { \scriptstyle \pm 0 . 1 }$ </td><td> $7 8 . 1 \pm 2 . 4$ </td><td> $6 5 . 7 _ { \pm 2 . 6 }$ </td><td> $7 0 . 5 { \scriptstyle \pm 6 . 2 }$ </td><td> $8 3 . 5 { \scriptstyle \pm 1 . 8 }$ </td><td> $7 6 . 1 \pm 8 . 2$ </td><td> $4 2 . 4 _ { \pm 3 . 0 }$ </td><td> $7 5 . 5 { \scriptstyle \pm 0 . 2 }$ </td><td> $9 0 . 2 { \scriptstyle \pm 0 . 1 }$ </td><td> ${ \bf 9 1 . 4 _ { \pm 0 . 2 } }$ </td></tr><tr><td>Pythia 12B</td><td> $1 1 . 4 _ { \pm 0 . 0 }$ </td><td> $6 4 . 5 { \scriptstyle \pm 0 . 3 }$ </td><td> $8 7 . 0 { \scriptstyle \pm 0 . 4 }$ </td><td> $9 1 . 0 { \scriptstyle \pm 0 . 1 }$ </td><td> $8 1 . 5 { \scriptstyle \pm 4 . 6 }$ </td><td> $5 5 . 2 { \scriptstyle \pm 3 . 8 }$ </td><td> $7 1 . 8 { \scriptstyle \pm 4 . 1 }$ </td><td> $8 2 . 9 { \scriptstyle \pm 1 . 0 }$ </td><td> $7 3 . 9 { \scriptstyle \pm 3 . 9 }$ </td><td> $4 2 . 2 { \scriptstyle \pm 0 . 8 }$ </td><td> $7 5 . 0 { \scriptstyle \pm 0 . 4 }$ </td><td> $9 1 . 1 { \pm } 0 . 4 $ </td><td> ${ \bf 9 1 . 2 _ { \pm 0 . 5 } }$ </td></tr><tr><td>LLaMA 7B</td><td> $1 7 . 1 _ { \pm 0 . 0 }$ </td><td> $6 7 . 8 { \scriptstyle \pm 0 . 5 }$ </td><td> $8 8 . 4 _ { \pm 0 . 2 }$ </td><td> $9 0 . 6 { \scriptstyle \pm 0 . 3 }$ </td><td> $7 6 . 1 \pm 2 . 7$ </td><td> $7 0 . 2 { \scriptstyle \pm 6 . 8 }$ </td><td> $5 9 . 9 { \scriptstyle \pm 3 . 8 }$ </td><td> $8 7 . 5 { \scriptstyle \pm 2 . 5 }$ </td><td> $8 2 . 5 { \scriptstyle \pm 4 . 7 }$ </td><td> $5 9 . 4 { \scriptstyle \pm 0 . 5 }$ </td><td> $8 0 . 4 { \scriptstyle \pm 0 . 9 }$ </td><td> $\mathbf { 9 2 . 0 _ { \pm 0 . 2 } }$ </td><td> $9 1 . 9 { \scriptstyle \pm 0 . 4 }$ </td></tr><tr><td>GPT-J 6B</td><td> $1 0 . 2 { \scriptstyle \pm 0 . 0 }$ </td><td> $6 3 . 6 { \scriptstyle \pm 0 . 4 }$ </td><td> $8 6 . 3 { \scriptstyle \pm 0 . 3 }$ </td><td> $9 1 . 0 { \scriptstyle \pm 0 . 2 }$ </td><td> $7 3 . 7 _ { \pm 4 . 0 }$ </td><td> $6 6 . 5 { \scriptstyle \pm 6 . 1 }$ </td><td> $7 7 . 8 { \scriptstyle \pm 3 . 6 }$ </td><td> $8 6 . 0 { \scriptstyle \pm 0 . 8 }$ </td><td> $8 4 . 9 { \scriptstyle \pm 3 . 2 }$ </td><td> $5 6 . 3 { \scriptstyle \pm 1 . 7 }$ </td><td> $7 2 . 4 { \scriptstyle \pm 0 . 6 }$ </td><td> $9 1 . 8 { \scriptstyle \pm 0 . 2 }$ </td><td> $\mathbf { 9 1 . 9 } _ { \pm 0 . 4 }$ </td></tr><tr><td>Avg.</td><td> $1 2 . 5 { \scriptstyle \pm 0 . 0 }$ </td><td> $6 4 . 3 { \scriptstyle \pm 0 . 3 }$ </td><td> $8 6 . 8 { \scriptstyle \pm 0 . 3 }$ </td><td> $9 0 . 8 { \scriptstyle \pm 0 . 1 }$ </td><td> $7 7 . 7 _ { \pm 0 . 4 }$ </td><td> $6 3 . 9 { \scriptstyle \pm 0 . 3 }$ </td><td> $7 0 . 1 { \scriptstyle \pm 3 . 4 }$ </td><td> $8 3 . 9 { \scriptstyle \pm 1 . 3 }$ </td><td> $7 9 . 1 \pm 2 . 6$ </td><td> $4 8 . 2 { \scriptstyle \pm 0 . 9 }$ </td><td> $7 5 . 8 { \scriptstyle \pm 0 . 3 }$ </td><td> $9 1 . 0 { \scriptstyle \pm 0 . 1 }$ </td><td> ${ \bf 9 1 . 5 _ { \pm 0 . 3 } }$ </td></tr></table>

## 4.3 Main results

Comparison with trained methods on LMMs. Tables 1 and 2 test whether an update at the input embeddings alone can match methods that learn a shift in every decoder layer. STAVE has the best score on every column of Table 1 and the best average on both backbones of Table 2, and it leads on five of six columns with three more recent LMMs (Appendix G.1). It stores 14 to 2,080 times fewer parameters than LIVE, MimIC, and HiFICL and keeps zero-shot latency (Figure 1), since an input update already acts through every block (Proposition 2.1). PT also trains at the input, but it inserts virtual tokens and its best mode varies with the backbone and dataset. STAVE outperforms all three PT modes on average with 10 times fewer parameters and without lengthening the prompt. With over 2,000 times more parameters, LoRA still trails STAVE on every column of Table 1.

Comparison with ICL and task vectors. STAVE outperforms ICL with up to 32 demos on every column of Table 1 while using none, and ICL peaks at 8 or 16 demos on VQA. In Table 2, TV and ICV add demo activations without training and stay near zero-shot on Idefics2. I2CL learns their scales, and MTV and STV learn which heads receive them, yet all trail STAVE on average, even STV at the same size, so fitting the values to the answer matters more than choosing where to add them.

Comparison on LLMs. Table 3 tests whether the design transfers to LLMs. STAVE has the highest average on the TV benchmark and exceeds 15-shot ICL on every LLM without demos at inference. SITE, which edits head activations in every layer, comes closest, yet STAVE is slightly higher with an input update alone. TV and FV remain far below. LoRA, prefix tuning, and PT train more parameters yet stay below 5-shot ICL on every LLM, and in Tables 2 and 3 PT varies far more across runs than STAVE, whose zero-initialized vectors start at the exact zero-shot computation.

## 4.4 Ablation studies

Structural token groups. Figure 4(a) changes only how groups are assigned to vectors. The readout/context split is best on COCO and tied for best on VQAv2. One vector is never enough, and one for all tokens is weakest on COCO, as the dilution in Equation 1 predicts. Two vectors also fall short if the answer cue joins the context vector (Last / rest) or they split by modality (Text / image). Both put readout tokens in one vector with context tokens of smaller gradient per token (Figure 2(b)), so the gain comes from separating answer-producing tokens from their context. Three or seven vectors add parameters without a gain, consistent with Theorem 3.1. Appendix H adds equalnorm random splits and count normalization variants.

![](images/9b31bf7496411241bebc54f55d4bfddc6b6eff94c188555d41796def251dc9ff.jpg)

Injection depth. Figure 4(b) adds the same vectors after a decoder block instead of at the input (Appendix H.7). The embedding injection is best or tied for best on all four curves, and the retained gain shrinks with depth, falling below zero-shot after the last block on two of them. This matches Proposition 2.1: a later update reaches fewer blocks, and after the last block it can change only the first answer token, and only through the final token.

![](images/8374c316ba96fc23161b6d98c51e2e84b74e678f70a7af3f4cc6c4ece2cfe8b2.jpg)  
Figure 4: Ablations. (a) Groups split into 1, 2, 3, or 7 vectors, mean of Idefics2 and LLaVA. (b) Injection depth.

Paired supervision. Table 4 drops one branch of Equation 6. Paired training scores highest in every row, so demos in training also help the demo-free prompt, as Proposition 3.2 allows for aligned branch gradients. The source branch reuses each labeled query in a second prompt, and since paired vectors exceed target-only vectors on the demo-free prompt in every row, this extra signal costs no target performance, consistent with a small compatibility gap $\mathcal { C } _ { \mathrm { s r c | t g t } }$ (Section 3.3). Each single branch fits only the prompt format it sees. Source-only vectors are lowest for every LMM and dataset. Target-only vectors lose 2.6 points on the TV benchmark with one demo, lose most of their LLaVA COCO score with demos, and hallucinate more on Idefics2 COCO (Appendices H.1 and H.2). MimIC and HiFICL, which also train only on demo-free prompts, fall below 5 CIDEr on LLaVA COCO once a demo is present. Paired vectors hold in both formats, so one checkpoint serves prompts with and without demos.

Table 4: Training-objective ablation.
<table><tr><td>Backbone / Dataset</td><td>Paired</td><td>Target-only</td><td>Source-only</td></tr><tr><td>Idefics2 VQAv2</td><td>73.46%</td><td>73.32%</td><td>72.41%</td></tr><tr><td>LLaVA VQAv2</td><td>76.12%</td><td>76.01%</td><td>74.98%</td></tr><tr><td>Idefics2 COCO</td><td>134.38</td><td>134.01</td><td>131.34</td></tr><tr><td>LLaVA COCO</td><td>131.78</td><td>129.80</td><td>129.11</td></tr><tr><td>TV bench</td><td>92.07%</td><td>91.53%</td><td>79.67%</td></tr><tr><td>w/1 demo</td><td>92.23%</td><td>88.90%</td><td>92.10%</td></tr></table>

Table 5: Hallucination on COCO.
<table><tr><td>Idefics2 COCO</td><td>CIDEr↑</td><td>CHAIRs ↓ CHAIRi↓ Recall ↑</td><td></td><td></td></tr><tr><td>Zero-shot 8-shot ICL</td><td>80.54 119.07</td><td>3.44 3.06</td><td>2.84 2.26</td><td>35.73 41.76</td></tr><tr><td>LoRA</td><td></td><td></td><td></td><td></td></tr><tr><td>LIVE</td><td>132.37 127.62</td><td>2.46 2.56</td><td>1.68</td><td>43.64 43.15</td></tr><tr><td>MimIC</td><td>132.38</td><td>3.48</td><td>1.83 2.28</td><td>45.62</td></tr><tr><td>HiFICL</td><td>128.62</td><td>3.30</td><td>2.10</td><td>45.98</td></tr><tr><td>STAVE</td><td>133.27</td><td>2.24</td><td>1.53</td><td>44.20</td></tr></table>

![](images/1aebead945a6c16cfdea2ad663ff231c56ea6f76284da97b11585b796e430d3f.jpg)  
Figure 7: Data efficiency across training set sizes.

## 4.5 Analysis

Query-specific answers from one update. The same two vectors are added to every prompt, yet the answer changes with the query (Figure 5). Without the update, LLaVA predicts a template token for almost every VQAv2 question. STAVE moves the states of each answer type in a different direction, and within object questions it separates the categories they ask about, because the frozen blocks process the vectors together with each query. Table 1 shows examples.

What each vector contributes. Figure 6(a) adds the Idefics2 VQAv2 vectors one at a time to held-out prompts without retraining and reads the reference first token. Without the update, this token often already ranks first among the task’s answers, yet the model almost never generates it. The context vector alone ranks it first more often but rarely changes the output, whereas the readout vector alone makes the model answer. Both vectors score highest. Figure 6(b) shows why the vectors play different roles. On VQAv2 and OK-VQA, most of the change that the readout vector makes to the final-token state is common to all prompts, whereas the change that the context vector adds becomes specific to each prompt from the middle layers on. The readout vector thus sets a shared answer format, and the context vector supplies queryspecific evidence for choosing the answer. Appendix I adds more tests of each vector.

![](images/323338ea248e9d20b2ea2935df04ca5bc14450a21927d750ee7e7335e9e9e8ca.jpg)  
Figure 5: Final-token states of LLaVA on VQAv2 validation questions, without and with STAVE.  
No update Context vector Readout vector Both vectors

![](images/4a2957b3b5be357f00d7cbe5b5d0df0df11e1f592f56e0983022138d29a25be7.jpg)

![](images/225d20746a0c26140398790fe380e7bb1bb38760f5dafcecec8eef2a8f7862cb.jpg)  
Figure 6: Effect of each learned vector.

Why a small update is sufficient. STAVE’s 2d parameters draw on what the frozen backbone already holds (Figure 6(a)) and leave it intact. On Qwen-VL OK-VQA, PT and all per-layer interventions in Table 2 fall below zero-shot, while STAVE exceeds it (Appendix F.3). The small update makes STAVE data efficient (Figure 7). With a tenth of the training samples it loses a few points, whereas MimIC and HiFICL lose far more on Idefics2, consistent with the risk bound of Section 3.3, whose estimation term depends on the covering dimension 2d, not on the number of decoder layers.

![](images/165e68f53fb36f3c00719df1a86256d1f577387f6052c104681eb62c99f615b7.jpg)

![](images/57aadff33c763fb18fa0cb4e93f021eec98984143dee22bfc74321822c047b4c.jpg)  
Figure 8: Gradient cosine before training (a) and performance after training (b) on the TV benchmark.

Separate vectors help most when gradients disagree. Proposition C.2(iii) predicts when STAVE’s two vectors matter. For the readout set A and the context tokens B, the first-order decrease of one shared vector grows with the cosine between the two sets’ summed gradients, whereas that of two vectors does not depend on it, so their gap widens as the cosine falls. Figure 8 tests this prediction on the 18 tasks of the TV benchmark with each of the five LLMs. Before training, we measure this cosine for each task and LLM, and after training we compare STAVE with a single vector shared by all eligible tokens. The cosines lie between −0.15 and 0.20 (Figure 8(a)), so the readout and context tokens ask for nearly orthogonal updates in every LLM. As predicted, the shared vector performs better as the cosine grows, whereas STAVE performs similarly across the whole range (Figure 8(b)). STAVE’s advantage is thus largest where the two gradients disagree most, 43.7 points in the lowest third of cosines against 27.5 points in the highest, and the cosine is negatively rank-correlated with this advantage for each of the five LLMs. Theorem 3.1 gives the same picture for margins. Its bound grows with the gradient component that a tied class discards, and one vector also discards the readout–context contrast that STAVE keeps. Because even the highest cosines stay far below one, the condition under which one vector suffices does not arise in these models, and the second vector lets STAVE perform well whether or not the two token sets agree.

Fine-grained classification. STAVE leads on Idefics2 DTD and Flowers, by 3.7 points on Flowers, and stays within 1.7 points of the best method on the other four columns, all led by HiFICL. The test classes of these two-choice tasks never appear as training queries, so correcting a zero-shot error means separating similar classes that no training query contains. HiFICL can add query-specific visual evidence in every layer with 272 times more parameters, whereas the two vectors of STAVE act only through the frozen blocks and draw on the classes the backbone already recognizes.

Object hallucination. Table 5 checks what drives the CIDEr gain on Idefics2 COCO. STAVE has the highest CIDEr and lowest CHAIRs and CHAIRi (Rohrbach et al., 2018), the shares of captions and of mentions that name absent objects. MimIC and HiFICL recall more objects, but their CHAIRs stays near the zero-shot level, while STAVE lowers it by a third, so its gain comes from more accurate mentions. In the COCO example of Table 1, STAVE names the chicken, whereas MimIC and HiFICL describe only a plate of food.

Inference cost. Table 6 measures the per-query cost of the Idefics2 VQAv2 checkpoints of Table 1 on one NVIDIA H200. STAVE adds its vectors before the first decoder block, so its decoder computation is that of zero-shot inference (Appendix E.1). ICL re-encodes its demos at every query, and with 32 demos its time to first token (TTFT) is 60 times that of STAVE, and its memory grows with the demo count. The trained methods keep the prompt length but add computation inside the decoder layers, which slows every decoded token. STAVE has the lowest cost in every column of both panels.

Table 6: Inference cost on Idefics2 VQAv2.
<table><tr><td>vs. ICL</td><td></td><td>Tokens TTFT (ms)</td><td>Mem. (GiB)</td></tr><tr><td rowspan="3">Zero-shot 8-shot ICL</td><td>128</td><td>98</td><td rowspan="3">16.5 29.8</td></tr><tr><td>1,026</td><td>1,430</td></tr><tr><td>16-shot ICL 1,922</td><td>2,854</td></tr><tr><td>32-shot ICL</td><td>3,717</td><td>5,877</td><td>43.0 68.7</td></tr><tr><td>STAVE</td><td>128</td><td>98</td><td>16.5</td></tr><tr><td rowspan="2">vs. trained</td><td>State (KiB)</td><td></td><td rowspan="2">TTFT (ms) Decode (ms/tok)</td></tr><tr><td>68,704</td><td></td></tr><tr><td>LoRA LIVE</td><td>512</td><td>116 101</td><td rowspan="2">33.6 27.4</td></tr><tr><td>MimIC</td><td>1,028</td><td></td></tr><tr><td></td><td></td><td>106</td><td>31.6</td></tr><tr><td>HiFICL</td><td>8,704</td><td>108</td><td>34.3</td></tr><tr><td>STAVE</td><td>32</td><td>98</td><td>23.9</td></tr></table>

## 5 Related Work

ICL puts demos in every query’s prompt (Brown et al., 2020), and in LMMs each demo image adds many visual tokens (Laurençon et al., 2024b). Extracted task vectors reuse demo activations as one vector at one layer (Hendel et al., 2023; Todd et al., 2024), at heads selected per task (Huang et al., 2024; Ma et al., 2026; Park et al., 2026), or in every layer with fixed or learned scales (Liu et al., 2024b; Li et al., 2025b), and STAVE beats all seven on average, even STV at equal size. Learned task vectors instead train one vector from the answer at a chosen layer and token (Yang et al., 2026). STAVE also learns from the answer, but on tokens set by prompt structure, so it needs no layer, head, or token search. Trained interventions add per-layer vectors (Peng et al., 2024), per-head shifts (Jiang et al., 2025), attention-logit biases (Talemi et al., 2026), or virtual keys and values (Li et al., 2026) to every decoder layer, and all four train only on demo-free prompts, three against an ICL teacher. STAVE stores 2d input scalars regardless of depth, adds no per-layer computation, and scores higher on average. Its teacher-free training with and without demos beats demo-free training (Table 4). PT and prefix tuning insert new tokens or attention memory (Lester et al., 2021; Li & Liang, 2021; Gupta et al., 2026), and within a layer added keys scale a token’s original attention weights by one factor, so its relative attention over image and text stays fixed (Petrov et al., 2024). STAVE writes onto existing tokens, so it can change this relative attention, keeps the zero-shot prompt length, and outperforms 20-token PT with 10 times fewer parameters. Steering vectors and ReFT also edit existing hidden states, but at chosen layers and tokens (Turner et al., 2023; Wu et al., 2024). Adapters and LoRA modify decoder modules per task (Houlsby et al., 2019; Hu et al., 2021; Liu et al., 2022a), whereas STAVE swaps two input vectors, shares every module across tasks, and outperforms LoRA with orders of magnitude fewer parameters. Appendix A extends this discussion.

## 6 Conclusion

We presented STAVE, which replaces in-context demos with two task-specific vectors added to the input embeddings of a frozen LLM or LMM. This is the only depth that reaches every decoder block, and the vectors are added to existing tokens, one for the answer-producing tokens and one for the other five groups. Trained with and without demos, they match or outperform per-layer interventions on six LMMs with 14 to 2,080 times fewer task parameters and exceed 15-shot ICL on five LLMs, while keeping the prompt length, latency, and memory of zero-shot inference. The ablations show that the gain comes from separating the answer-producing tokens from their context, not from more vectors or a modality split, and that training with demos also helps the demo-free prompt. Our analysis suggests that the vectors mainly make the frozen model output answers it already represents, and their captions name fewer absent objects. Training is a one-time cost but backpropagates through the frozen decoder, so its memory grows with the backbone, and it requires access to the weights. Since each task adds only two vectors and no prompt tokens, one deployed backbone can serve many tasks. Extending structural token groups to multi-image and video prompts is a natural next step.

## References

Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems, 35:23716–23736, 2022.

Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond, 2023. URL https://arxiv.org/abs/2308.12966.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report, 2025. URL https://arxiv.org/abs/2502.13923.

Stella Biderman, Hailey Schoelkopf, Quentin Gregory Anthony, Herbie Bradley, Kyle O’Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, et al. Pythia: A suite for analyzing large language models across training and scaling. In International conference on machine learning, pp. 2397–2430. PMLR, 2023.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

Wang Cai, Hsiu-Yuan Huang, Zhixiang Wang, and Yunfang Wu. Beyond demonstrations: Dynamic vector construction from latent representations. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pp. 5853–5868, 2025.

Mircea Cimpoi, Subhransu Maji, Iasonas Kokkinos, Sammy Mohamed, and Andrea Vedaldi. Describing textures in the wild. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 3606–3613, 2014.

Xi Ding and Lei Wang. Do language models understand time? In Companion Proceedings of the ACM on Web Conference 2025, pp. 1855–1868, 2025.

Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Jingyuan Ma, Rui Li, Heming Xia, Jingjing Xu, Zhiyong Wu, Baobao Chang, et al. A survey on in-context learning. In Proceedings of the 2024 conference on empirical methods in natural language processing, pp. 1107–1128, 2024.

Yuxin Dong, Jiachen Jiang, Zhihui Zhu, and Xia Ning. Understanding task vectors in in-context learning: Emergence, functionality, and limitations. In International Conference on Learning Representations, volume 2026, pp. 19393– 19428, 2026.

Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 6904–6913, 2017.

Akash Gupta, Amos Storkey, and Mirella Lapata. Meta-adaptive prompt distillation for few-shot visual question answering. In International Conference on Learning Representations, volume 2026, pp. 120800–120837, 2026.

Danna Gurari, Qing Li, Abigale J Stangl, Anhong Guo, Chi Lin, Kristen Grauman, Jiebo Luo, and Jeffrey P Bigham. Vizwiz grand challenge: Answering visual questions from blind people. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 3608–3617, 2018.

Roee Hendel, Mor Geva, and Amir Globerson. In-context learning creates task vectors. In Findings ofthe Association for Computational Linguistics: EMNLP 2023, pp. 9318–9333, 2023.

Alberto Hojel, Yutong Bai, Trevor Darrell, Amir Globerson, and Amir Bar. Finding visual task vectors. In European Conference on Computer Vision, pp. 257–273. Springer, 2024.

Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin De Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. Parameter-efficient transfer learning for nlp. In International conference on machine learning, pp. 2790–2799. PMLR, 2019.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.

Brandon Huang, Chancharik Mitra, Assaf Arbelle, Leonid Karlinsky, Trevor Darrell, and Roei Herzig. Multimodal task vectors enable many-shot multimodal in-context learning. Advances in Neural Information Processing Systems, 37: 22124–22153, 2024.

Dongfu Jiang, Xuan He, Huaye Zeng, Cong Wei, Max Ku, Qian Liu, and Wenhu Chen. Mantis: Interleaved multi-image instruction tuning. arXiv preprint arXiv:2405.01483, 2024.

Yuchu Jiang, Jiale Fu, Chenduo Hao, Xinting Hu, Yingzhe Peng, Xin Geng, and Xu Yang. Mimic in-context learning for multimodal tasks. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, pp. 29825–29835, 2025.

Joonseong Kang, Soojeong Lee, Subeen Park, Sumin Park, Taero Kim, Jihee Kim, Ryunyi Lee, and Kyungwoo Song. Adaptive task vectors for large language models. arXiv preprint arXiv:2506.03426, 2025.

Andrej Karpathy and Li Fei-Fei. Deep visual-semantic alignments for generating image descriptions. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 3128–3137, 2015.

Hugo Laurençon, Andrés Marafioti, Victor Sanh, and Léo Tronchon. Building and better understanding vision-language models: insights and future directions. arXiv preprint arXiv:2408.12637, 2024a.

Hugo Laurençon, Léo Tronchon, Matthieu Cord, and Victor Sanh. What matters when building vision-language models? Advances in Neural Information Processing Systems, 37:87874–87907, 2024b.

Brian Lester, Rami Al-Rfou, and Noah Constant. The power of scale for parameter-efficient prompt tuning. In Proceedings of the 2021 conference on empirical methods in natural language processing, pp. 3045–3059, 2021.

Dongfang Li, Zhenyu Liu, Xinshuo Hu, Zetian Sun, Baotian Hu, and Min Zhang. In-context learning state vector with inner and momentum optimization. Advances in Neural Information Processing Systems, 37:7797–7820, 2024a.

Feng Li, Renrui Zhang, Hao Zhang, Yuanhan Zhang, Bo Li, Wei Li, Zejun Ma, and Chunyuan Li. Llava-next-interleave: Tackling multi-image, video, and 3d in large multimodal models. arXiv preprint arXiv:2407.07895, 2024b.

Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, and Martin Wattenberg. Inference-time intervention: Eliciting truthful answers from a language model. Advances in Neural Information Processing Systems, 36:41451– 41530, 2023.

Xiang Lisa Li and Percy Liang. Prefix-tuning: Optimizing continuous prompts for generation. In Proceedings ofthe 59th Annual Meeting ofthe Associationfor Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pp. 4582–4597, 2021.

Xiaoyu Li, Yuhang Liu, Xuanshuo Kang, Zheng Luo, Fangqi Lou, Xiaohua Wu, and Zihan Xiong. Hificl: High-fidelity in-context learning for multimodal tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3069–3078, 2026.

Yanshu Li, Yi Cao, Hongyang He, Qisen Cheng, Xiang Fu, Xi Xiao, Tianyang Wang, and Ruixiang Tang. M<sup>2</sup>IV: Towards efficient and fine-grained multimodal in-context learning via representation engineering. arXiv preprint arXiv:2504.04633, 2025a.

Zhuowei Li, Zihao Xu, Ligong Han, Yunhe Gao, Song Wen, Di Liu, Hao Wang, and Dimitris Metaxas. Implicit in-context learning. In International Conference on Learning Representations, volume 2025, pp. 101644–101666, 2025b.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision, pp. 740–755. Springer, 2014.

Haokun Liu, Derek Tam, Mohammed Muqeeth, Jay Mohta, Tenghao Huang, Mohit Bansal, and Colin A Raffel. Fewshot parameter-efficient fine-tuning is better and cheaper than in-context learning. Advances in Neural Information Processing Systems, 35:1950–1965, 2022a.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023a.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pp. 26296–26306, 2024a.

Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, and Graham Neubig. Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language processing. ACM computing surveys, 55(9): 1–35, 2023b.

Sheng Liu, Haotian Ye, Lei Xing, and James Y. Zou. In-context vectors: Making in context learning more effective and controllable through latent space steering. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp (eds.), Proceedings ofthe 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pp. 32287–32307. PMLR, 21–27 Jul 2024b. URL https://proceedings.mlr.press/v235/liu24bx.html.

Xiao Liu, Kaixuan Ji, Yicheng Fu, Weng Tam, Zhengxiao Du, Zhilin Yang, and Jie Tang. P-tuning: Prompt tuning can be comparable to fine-tuning across scales and tasks. In Proceedings ofthe 60th Annual Meeting ofthe Association for Computational Linguistics (Volume 2: Short Papers), pp. 61–68, 2022b.

Xiao Liu, Yanan Zheng, Zhengxiao Du, Ming Ding, Yujie Qian, Zhilin Yang, and Jie Tang. Gpt understands, too. AI open, 5:208–215, 2024c.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.

Grace Luo, Trevor Darrell, and Amir Bar. Vision-language models create cross-modal task representations. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaff, and Jerry Zhu (eds.), Proceedings ofthe 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, pp. 41157–41176. PMLR, 13–19 Jul 2025. URL https://proceedings.mlr. press/v267/luo25c.html.

Ziyu Ma, Chenhui Gou, Yiming Hu, Yong Wang, Bohan Zhuang, and Jianfei Cai. Where and what matters: Sensitivityaware task vectors for many-shot multimodal in-context learning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pp. 7892–7900, 2026.

Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. Ok-vqa: A visual question answering benchmark requiring external knowledge. In Proceedings ofthe IEEE/cvfconference on computer vision and pattern recognition, pp. 3195–3204, 2019.

Jesse Mu, Xiang Li, and Noah Goodman. Learning to compress prompts with gist tokens. Advances in Neural Information Processing Systems, 36:19327–19352, 2023.

Maria-Elena Nilsback and Andrew Zisserman. Automated flower classification over a large number of classes. In 2008 Sixth Indian conference on computer vision, graphics & image processing, pp. 722–729. IEEE, 2008.

Samet Oymak, Ankit Singh Rawat, Mahdi Soltanolkotabi, and Christos Thrampoulidis. On the role of attention in prompt-tuning. In International Conference on Machine Learning, pp. 26724–26768. PMLR, 2023.

Jungwon Park, Jimyeong Kim, Changin Choi, and Wonjong Rhee. Soft head selection for injecting icl-derived task embeddings. In Findings of the Association for Computational Linguistics: ACL 2026, pp. 27161–27214, 2026.

Yingzhe Peng, Chenduo Hao, Xinting Hu, Jiawei Peng, Xin Geng, and Xu Yang. Live: Learnable in-context vector for visual question answering. Advances in Neural Information Processing Systems, 37:9773–9800, 2024.

Aleksandar Petrov, Philip Torr, and Adel Bibi. When do prompting and prefix-tuning work? a theory of capabilities and limitations. In International Conference on Learning Representations, volume 2024, pp. 6031–6054, 2024.

Guanghui Qin and Jason Eisner. Learning how to ask: Querying lms with mixtures of soft prompts. In Proceedings ofthe 2021 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, pp. 5203–5212, 2021.

Nina Rimsky, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, and Alexander Turner. Steering llama 2 via contrastive activation addition. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 15504–15522, 2024.

Anna Rohrbach, Lisa Anne Hendricks, Kaylee Burns, Trevor Darrell, and Kate Saenko. Object hallucination in image captioning. In Proceedings of the 2018 conference on empirical methods in natural language processing, pp. 4035–4045, 2018.

Baturay Saglam, Xinyang Hu, Zhuoran Yang, Dionysis Kalogerias, and Amin Karbasi. Learning task representations from in-context learning. In Findings of the Association for Computational Linguistics: ACL 2025, pp. 6634–6663, 2025.

Melanie Sclar, Yejin Choi, Yulia Tsvetkov, and Alane Suhr. Quantifying language models’ sensitivity to spurious features in prompt design or: How i learned to start worrying about prompt formatting. In International Conference on Learning Representations, volume 2024, pp. 25055–25083, 2024.

Nishant Subramani, Nivedita Suresh, and Matthew E Peters. Extracting latent steering vectors from pretrained language models. In Findings of the Association for Computational Linguistics: ACL 2022, pp. 566–581, 2022.

Niloufar Alipour Talemi, Hossein Kashiani, and Fatemeh Afghah. Hyper-icl: Attention calibration with hyperbolic anchor distillation for multimodal in-context learning. arXiv preprint arXiv:2606.04434, 2026.

Eric Todd, Millicent Li, Arnab Sen Sharma, Aaron Mueller, Byron Wallace, and David Bau. Function vectors in large language models. In International conference on learning representations, volume 2024, pp. 17282–17333, 2024.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J Vazquez, Ulisse Mini, and Monte MacDiarmid. Steering language models with activation engineering. arXiv preprint arXiv:2308.10248, 2023.

Ramakrishna Vedantam, C Lawrence Zitnick, and Devi Parikh. Cider: Consensus-based image description evaluation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 4566–4575, 2015.

Constantin Venhoff, Iván Arcuschin, Philip Torr, Arthur Conmy, and Neel Nanda. Understanding reasoning in thinking language models via steering vectors. arXiv preprint arXiv:2506.18167, 2025.

Catherine Wah, Steve Branson, Peter Welinder, Pietro Perona, and Serge Belongie. The caltech-ucsd birds-200-2011 dataset. 2011.

Ben Wang and Aran Komatsuzaki. Gpt-j-6b: A 6 billion parameter autoregressive language model, 2021.

Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, Zhaokai Wang, Zhe Chen, Hongjie Zhang, Ganlin Yang, Haomin Wang, Qi Wei, Jinhui Yin, Wenhao Li, Erfei Cui, Guanzhou Chen, Zichen Ding, Changyao Tian, Zhenyu Wu, Jingjing Xie, Zehao Li, Bowen Yang, Yuchen Duan, Xuehui Wang, Zhi Hou, Haoran Hao, Tianyi Zhang, Songze Li, Xiangyu Zhao, Haodong Duan, Nianchen Deng, Bin Fu, Yinan He, Yi Wang, Conghui He, Botian Shi, Junjun He, Yingtong Xiong, Han Lv, Lijun Wu, Wenqi Shao, Kaipeng Zhang, Huipeng Deng, Biqing Qi, Jiaye Ge, Qipeng Guo, Wenwei Zhang, Songyang Zhang, Maosong Cao, Junyao Lin, Kexian Tang, Jianfei Gao, Haian Huang, Yuzhe Gu, Chengqi Lyu, Huanze Tang, Rui Wang, Haijun Lv, Wanli Ouyang, Limin Wang, Min Dou, Xizhou Zhu, Tong Lu, Dahua Lin, Jifeng Dai, Weijie Su, Bowen Zhou, Kai Chen, Yu Qiao, Wenhai Wang, and Gen Luo. Internvl3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency, 2025. URL https://arxiv.org/abs/2508.18265.

Yihan Wang, Jatin Chauhan, Wei Wang, and Cho-Jui Hsieh. Universality and limitations of prompt tuning. Advances in Neural Information Processing Systems, 36:75623–75643, 2023.

Zhengxuan Wu, Aryaman Arora, Zheng Wang, Atticus Geiger, Dan Jurafsky, Christopher D Manning, and Christopher Potts. Reft: Representation finetuning for language models. Advances in Neural Information Processing Systems, 37: 63908–63962, 2024.

Haolin Yang, Hakaze Cho, Kaize Ding, and Naoya Inoue. Task vectors, learned not extracted: Performance gains and mechanistic insights. In International Conference on Learning Representations, volume 2026, pp. 74846–74903, 2026.

Renrui Zhang, Jiaming Han, Chris Liu, Peng Gao, Aojun Zhou, Xiangfei Hu, Shilin Yan, Pan Lu, Hongsheng Li, and Yu Qiao. Llama-adapter: Efficient fine-tuning of language models with zero-init attention. arXiv preprint arXiv:2303.16199, 2023.

Yiyuan Zhang, Handong Li, Jing Liu, and Xiangyu Yue. Learning beyond still frames: Scaling vision-language models with video. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 22425–22435, 2025.

Haozhe Zhao, Zefan Cai, Shuzheng Si, Xiaojian Ma, Kaikai An, Liang Chen, Zixuan Liu, Sheng Wang, Wenjuan Han, and Baobao Chang. Mmicl: Empowering vision-language model with multi-modal in-context learning. In International Conference on Learning Representations, volume 2024, pp. 14942–14980, 2024.

Zihao Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh. Calibrate before use: Improving few-shot performance of language models. In International conference on machine learning, pp. 12697–12706. Pmlr, 2021.

Yucheng Zhou, Xiang Li, Qianning Wang, and Jianbing Shen. Visual in-context learning for large vision-language models. In Findings ofthe Associationfor Computational Linguistics: ACL 2024, pp. 15890–15902, 2024.

Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, et al. Representation engineering: A top-down approach to ai transparency. arXiv preprint arXiv:2310.01405, 2023.

A Extended related work 16   
B Injection depth and inserted tokens 18   
B.1 Proof of Proposition 2.1 . 18   
B.2 Attention ratios at a fixed layer 18   
C Structural token groups and vector sharing 19   
C.1 Partition conventions and modality handling 19   
C.2 Normalized geometry and gradients 19   
C.3 First-order cost of sharing . 20   
C.4 Sharing cost before training . 20   
C.5 Proof of Theorem 3.1 21   
D Paired supervision and demo-free risk   
D.1 Proof of Proposition 3.2 .   
D.2 Compatibility of the empirical objectives .   
D.3 Demo-free risk guarantees   
E Inference cost   
E.1 Measured inference cost   
E.2 Stored task state .   
E.3 Performance overview settings   
F Implementation details   
F.1 Backbones .   
F.2 VQAv2, OK-VQA, and COCO   
F.3 VizWiz, OK-VQA, and fine-grained classification 29   
F.4 TV benchmark 29   
F.5 Hyperparameter sensitivity 30   
F.6 Dataset and prompt examples 31   
G Additional results 33   
G.1 More recent LMMs 33   
G.2 VQAv2 answer types 33   
H Ablation details 34   
H.1 Demos in the prompt on COCO Captioning   
H.2 Object hallucination by training objective   
H.3 Training objective on the TV benchmark .   
H.4 Structural token groups .   
H.5 Allocation of a shared update   
H.6 Count normalization   
H.7 Injection depth   
Analysis of the learned vectors 40   
I.1 Promoted tokens 40   
I.2 States and decoded tokens   
I.3 What each learned vector changes   
I.4 Answer-prefix probabilities   
I.5 Qualitative examples   
J Limitations and future work 48   
K Impact statement 48

## A Extended related work

Multimodal ICL and its cost. Visual instruction tuning connects image understanding with language generation (Liu et al., 2023a; 2024a), and interleaved image–text training together with multi-image instruction tuning supports richer multimodal contexts (Laurençon et al., 2024b; Jiang et al., 2024; Zhao et al., 2024). These models inherit demo-based adaptation from language models (Brown et al., 2020), but each demo image adds visual processing and many visual tokens to every query. Visual In-Context Learning retrieves demos, summarizes their images, and composes textual demos (Zhou et al., 2024), yet the demos remain in every prompt. Demo-free methods remove them and store the task in a compact state. They differ in where this state enters the frozen model: as extracted or trained decoder states, as added input tokens or attention memory, or, in STAVE, as updates to existing input embeddings. STAVE keeps only the query image and text at inference, so it never encodes a demo image for a new query and runs with the prompt length and cache of zero-shot inference.

Extracted visual and multimodal task vectors. Visual task vectors in MAE-VQGAN show that averaged attentionhead activations can specify visual tasks without input–output examples (Hojel et al., 2024), and cross-modal activation patching shows that LMMs encode task information shared across text and image inputs (Luo et al., 2025). MTV averages demo-conditioned head activations and uses reinforcement learning to select the heads that receive them (Huang et al., 2024). STV selects context-sensitive heads from activation differences and then uses reinforcement learning to pick values from clustered activation banks (Ma et al., 2026). In both, the values are restricted to demo activations, and every task needs a search over head locations. STAVE learns its values directly with answer supervision on tokens fixed by prompt structure, which removes the head search and the activation bank and lets the readout and context vectors be optimized jointly for the same answer. Because the update enters before the first decoder layer, one pair of vectors reaches every layer’s attention, whereas a head-level vector acts only where it is inserted. STAVE outperforms MTV and STV in every comparison, including STV at an equal parameter count.

Distillation through decoder interventions. LIVE learns one vector and one scale per decoder layer by matching the output distribution of a demo-conditioned model, with an added answer loss (Peng et al., 2024). M<sup>2</sup>IV assigns separate vectors to the attention and feed-forward branches of every layer and combines output distillation, answer supervision, and a branch-correlation objective (Li et al., 2025a). MimIC learns per-head shifts after attention with query-dependent scaling and aligns hidden states with the demo-conditioned model layer by layer (Jiang et al., 2025). Hyper-ICL adds low-rank, query-modulated biases to attention logits and aligns intermediate features with a demo-conditioned teacher in hyperbolic space (Talemi et al., 2026). These parameters sit in every decoder layer, so their storage grows with depth, their objectives match teacher outputs or intermediate features, and the student sees only demo-free prompts. STAVE stores 2d scalars regardless of depth, 14 to 2,080 times fewer than LIVE and MimIC, adds no per-layer computation at inference, and scores higher on average. Its two vectors are trained with answer CE on prompts with and without demos, with no teacher target, and this paired objective improves demo-free accuracy over training on demo-free prompts alone.

Teacher-free virtual attention context. HiFICL places low-rank virtual KV pairs in every attention head, whose interaction with each query mixes learned context values into the original attention output, and trains them with the task loss alone, as STAVE does (Li et al., 2026). STAVE instead changes the embeddings of existing readout and context tokens before the decoder. It stores 2d scalars, over 200 times fewer than HiFICL, adds no attention-memory entries, and scores higher on average on every LMM we evaluate. Its readout vector also changes the attention query of the token that predicts the answer, so it can re-weight attention over the query’s own image and text tokens, whereas virtual keys and values can only mix a learned context into the original attention output.

Extracted and adaptive task representations in language models. TV takes the prompted residual state at one layer (Hendel et al., 2023), FV sums the outputs of task-relevant attention heads (Todd et al., 2024), and in-context vectors derive steering directions from latent input–output differences (Liu et al., 2024b). Later methods optimize injection coefficients or weighted combinations of head outputs (Li et al., 2025b; Saglam et al., 2025), generate a vector for each query (Kang et al., 2025), or refine it with inner and momentum optimization (Li et al., 2024a). SITE learns soft head-selection weights over ICL-derived embeddings (Park et al., 2026), and dynamic vector construction segments extracted representations and learns their injection locations (Cai et al., 2025). Learned task vectors replace extraction with answer-supervised training of a vector at a chosen layer and token (Yang et al., 2026), and analyses of extracted task vectors identify limits on representing high-rank task mappings (Dong et al., 2026). STAVE also learns its vectors from the answer, but places them at the input over tokens defined by prompt structure, so its tokens need no per-task search and one structural partition transfers across tasks and backbones. Its normalization keeps the update energy from growing with the number of tokens a vector updates, which matters in LMMs, where one image expands into hundreds of tokens. On the TV benchmark, STAVE exceeds TV and FV on every model and reaches the highest average accuracy, above SITE and 15-shot ICL, without any demo at inference. The vector-sharing tradeoffs of this parameterization are analyzed in Appendix C.5.

Task adaptation at added input tokens. PT prepends learned vectors to the input embeddings of a frozen model and approaches full fine-tuning as model scale grows (Lester et al., 2021). P-tuning generates continuous prompts at template-defined locations with a small prompt encoder (Liu et al., 2024c), deep prompt tuning and prefix tuning supply prompt vectors or KV pairs at every decoder layer (Liu et al., 2022b; Li & Liang, 2021), and gist tokens compress an instruction into the activations of a few added tokens (Mu et al., 2023). In LMMs, MAPD meta-learns soft prompts together with an attention mapper over visual features and adapts them from a few labeled examples at test time (Gupta et al., 2026). Surveys of prompting collect further variants (Liu et al., 2023b). All of these methods add tokens or attention-memory entries whose placement is a design choice, and language models are sensitive to prompt formatting (Sclar et al., 2024). Theory shows that added prefix tokens only bias the attention output of the original tokens without changing their relative attention (Petrov et al., 2024), characterizes the expressive limits of bounded-length prompts (Wang et al., 2023), and traces how a tuned prompt acts through a single attention layer (Oymak et al., 2023). STAVE also keeps the decoder frozen and stores the task at the input, but differs in three respects. It writes onto existing tokens selected by a structural partition shared across tasks and backbones, so no insertion point is chosen and the prompt keeps its zero-shot length and cache. Its readout vector changes the attention query of the token that predicts the answer, so it can change the relative attention over the original image and text tokens, which added keys cannot do at a fixed layer. A zero update recovers the original computation, so training starts from the frozen model’s own predictions, whereas an added token with a zero embedding still enters the attention normalizer, and LLaMA-Adapter obtains this property only through a separately learned zero-initialized gate (Zhang et al., 2023). A two-token soft prompt stores as many parameters as STAVE, so these differences come from where the parameters enter rather than from capacity. STAVE outperforms 20-token PT on LMMs with 10 times fewer parameters and outperforms PT and prefix tuning on the TV benchmark. Appendix B.2 analyzes the attention-level consequence.

Updates to existing representations. Steering vectors add a fixed direction to the residual stream at chosen layers (Subramani et al., 2022; Turner et al., 2023; Rimsky et al., 2024; Zou et al., 2023), inference-time intervention shifts selected attention-head outputs (Li et al., 2023), and ReFT learns low-rank interventions on the hidden states of a fixed set of prefix and suffix tokens in selected layers (Wu et al., 2024). Mixtures of soft prompts initialize each prompt vector from a template word and tune it in place (Qin & Eisner, 2021). These methods choose a layer, a direction, or a set of tokens per task, and the steering approaches take their directions from activation contrasts rather than from the task answer. STAVE also updates existing tokens but acts at the input embeddings, so one update reaches every layer’s attention and no layer is chosen. Moving the same update to the output of a decoder layer does not improve it in our depth ablation. Its tokens follow prompt structure, including the expanded image embeddings and the final prompt token, and its two shared vectors are scaled by the number of tokens each updates and learned from the answer on paired prompts.

Weight and module adaptation. Adapters and LoRA learn task-specific modules or low-rank weight updates (Houlsby et al., 2019; Hu et al., 2021), and few-shot parameter-efficient fine-tuning has been proposed as an alternative to supplying demos repeatedly (Liu et al., 2022a). These methods also keep the pretrained weights frozen, but their task parameters are spread across decoder layers and grow with depth and rank. STAVE stores two embedding vectors, shares every decoder module across tasks, switches tasks by replacing two vectors, and outperforms LoRA with orders of magnitude fewer parameters.

## B Injection depth and inserted tokens

The derivative statements concern the real-valued decoder map at fixed tokenization, attention masks, positional indices, and correct answer prefixes. Modality features are frozen, and any stochastic computation is fixed or disabled. These statements do not differentiate through token selection or numerical quantization.

## B.1 Proof of Proposition 2.1

Fix a nonempty prompt with $T$ attended tokens. Let $H _ { \ell } \in \mathbb { R } ^ { T \times d }$ contain all its states after block $\ell ,$ with $H _ { 0 }$ the input embeddings. A prompt-only intervention at depth ℓ changes these states before later blocks are evaluated. It does not alter answer-token states at that depth or retroactively modify keys and values already formed in blocks $1 , \ldots , \ell .$ After the last block, the output computation consists only of positionwise normalization, if present, and a positionwise vocabulary head.

Proof. For part (i), the prompt keys and values used by block k are computed from $H _ { k - 1 }$ . An addition to $H _ { \ell }$ therefore leaves all prompt keys and values in blocks $k \leq \ell$ unchanged and can affect later blocks only. The first answer token is predicted from the final prompt token. Every subsequent teacher-forced answer token is predicted from an answer-prefix token. At $\ell = N _ { \mathrm { l a y e r } }$ , no later block can transmit a prompt-state change to these answer tokens. The positionwise head can therefore depend on the intervention only through the final prompt state, and only for the first prediction. Its derivative there may also vanish. Since the context vector excludes the final token, it has zero sequence-loss gradient at this depth.

For part (ii), take depths $0 \le \ell ^ { \prime } < \ell \le N _ { \mathrm { l a y e r } }$ . Causality makes the Jacobian of each intervening block, restricted to all prompt states, block lower triangular in token order. Its diagonal d $\times d$ blocks are the own-token derivatives. By assumption, these blocks are invertible. Their product, the Jacobian $\mathsf { T } _ { \ell , \ell ^ { \prime } }$ from the row-stacked entries of $H _ { \ell ^ { \prime } }$ to those of $H _ { \ell } .$ is thus invertible. For any desired infinitesimal change $\Delta \in \mathbb { R } ^ { T d }$ at depth $\ell ,$ the unique prompt-state perturbation producing it to first order is $\mathsf { T } _ { \ell , \ell ^ { \prime } } ^ { - 1 } \Delta$ . The first-answer logits factor through $H _ { \ell }$ and the remaining prompt computation, so the two interventions have the same first-order effect on those logits. □

Several depths. For simultaneous prompt-state shifts $\Delta _ { \ell }$ at multiple depths, all evaluated about the same unmodified forward pass, let $\ T _ { \ell } = \ T _ { \ell , 0 }$ and ${ \mathsf { T } } _ { 0 } = I .$ . The input perturbation $\sum \ell \mathsf { T } _ { \ell } ^ { - 1 } \Delta _ { \ell }$ reproduces their combined first-order first-answer effect by the chain rule and linearity, given nonsingularity along every involved path. Part (ii) concerns unrestricted perturbations over all attended prompt tokens and the first answer token. It characterizes what the input interface can reach, while the two-vector parameterization is analyzed in Appendix C.

## B.2 Attention ratios at a fixed layer

Fix one attention head and an attention query $\xi \in \mathbb { R } ^ { d _ { h } }$ , where $d _ { h }$ is the head dimension. For visible token i, let $k _ { i }$ and $u _ { i }$ be its key and value, and let its finite attention score be $s _ { i } = \xi ^ { \top } k _ { i } / \sqrt { d _ { h } } + c _ { i }$ , with fixed positional term $c _ { i }$ . Partition the visible tokens into nonempty original and added sets $Q$ and $D$ . Writing $\textstyle Z _ { Q } = \sum _ { i \in Q } e ^ { s _ { i } }$ and $\begin{array} { r } { Z _ { D } = \sum _ { i \in D } e ^ { s _ { i } } } \end{array}$ gives

$$
o = ( 1 - \lambda ) o ^ { Q } + \lambda o ^ { D } , \qquad \lambda = \frac { Z _ { D } } { Z _ { Q } + Z _ { D } } , \qquad o ^ { Q } = \frac { \sum _ { i \in Q } e ^ { s _ { i } } u _ { i } } { Z _ { Q } } , \quad o ^ { D } = \frac { \sum _ { i \in D } e ^ { s _ { i } } u _ { i } } { Z _ { D } } .\tag{B.1}
$$

This is the single-head decomposition used to motivate shift-based adaptation (Jiang et al., 2025). All quantities here belong to the same forward computation. In a multilayer decoder, $o ^ { Q }$ need not equal the output on a separately rendered demo-free prompt, because the original states and positional scores can also change.

Added keys and changed queries. If the original scores remain fixed when keys are added, every original weight $\alpha _ { i } = e ^ { s _ { i } } / \dot { Z } _ { Q }$ becomes $\bar { \alpha } _ { i } ^ { \prime } = \bar { Z } _ { Q } \alpha _ { i } / ( Z _ { Q } + Z _ { D } \bar { ) }$ . Hence $\alpha _ { i } ^ { \prime } / \alpha _ { i } ^ { \prime } = \alpha _ { i } / \alpha _ { j }$ for $i , j \in Q$ (Petrov et al., 2024). In contrast, holding keys and positional terms fixed while changing the query to $\xi + \epsilon b$ gives

$$
\left. \frac { d } { d \epsilon } \log \frac { \alpha _ { i } ( \epsilon ) } { \alpha _ { j } ( \epsilon ) } \right| _ { \epsilon = 0 } = \frac { ( k _ { i } - k _ { j } ) ^ { \top } b } { \sqrt { d _ { h } } } ,\tag{B.2}
$$

where $b \in \mathbb { R } ^ { d _ { h } }$ is the query-change direction. This follows by subtracting the two scores, since their common normalizer cancels. An update at an existing token can change queries, keys, and values without inserting tokens, and Equation B.2 isolates the query response that added keys lack at a fixed layer.

## C Structural token groups and vector sharing

Write $S \in \{ 0 , 1 \} ^ { R \times q }$ for the assignment matrix, with $S _ { k r } = { \bf 1 } \{ a ( k ) = r \}$ and exactly one nonzero entry per row. The transpose of learned row r for task τ is denoted by $v _ { \tau , r } \in \mathbb { R } ^ { d }$ , and the task subscript is suppressed on $W$ . The independent-group configuration learns R vectors, while STAVE ties the groups into the readout and context vectors as in Equation 2.

## C.1 Partition conventions and modality handling

The partition uses the serialized prompt and template metadata, not the answer label, model predictions, or learned attention scores. Counts refer to decoder-input tokens after image expansion. Demo spans include their inputs, answers, and demo-specific formatting. They are excluded before any span is assigned, with the FIRST token as the only exception. BOS tokens, padding, supervised answer tokens, and generated answer tokens are also excluded.

Endpoints and precedence. LAST is the final prompt token, the last attended token before the answer. FIRST is the first attended non-BOS token of the rendered prompt. When the template has an instruction prefix, as in VizWiz and VQAv2, this token lies in the shared prefix in both renderings and no demo token is updated. When the template has no prefix, it opens the query in the demo-free rendering and the first demo in the source rendering, and that single token receives the context vector while every other demo token is unchanged. On prompts that begin with an image, it is the opening delimiter of that image. The caption experiments instead use the first query-text token in both renderings. Tokens are assigned in the order LAST, FIRST, ANSWER-CUE, IMAGE-GATE, QUERY-IMAGE, QUESTION-TEXT, and PREFIX-TEXT, and the first matching label is kept. The same precedence is used for both renderings.

Spans. ANSWER-CUE consists of the template tokens that introduce the answer, such as “Answer:”, except the final prompt token, which forms LAST. IMAGE-GATE holds explicit delimiters around the query image, not its expanded visual embeddings, which form QUERY-IMAGE. QUESTION-TEXT is the remaining text of the query, such as the question in VQA. PREFIX-TEXT is the template’s instruction span outside demos. Tokens not identified by the fixed metadata remain unchanged. A group with no eligible tokens has count zero and contributes no update.

Modality handling. Text prompts use the frozen input-embedding layer. In LMMs, the frozen visual encoder and connector supply image embeddings at the hidden dimension of the decoder. STAVE operates on the merged decoderinput sequence, not on image pixels, visual encoder activations, or unexpanded placeholder tokens. Image groups are absent in text-only prompts, and IMAGE-GATE can also be absent in a multimodal template. Both paired renderings retain the query and its modality. These conventions specify an unambiguous operator.

## C.2 Normalized geometry and gradients

For prompt $p ,$ let $\mathcal { T } ( p ) = \{ k : n _ { k } ( p ) > 0 \}$ and $\mathcal { I } ( p ) = \{ r : N _ { r } ( p ) > 0 \}$ denote the active groups and rows. Define $G _ { t k } ( p ) = \mathbf { 1 } \{ s _ { t } ( p ) = k \} / \sqrt { n _ { k } ( p ) }$ for $k \in \mathcal { T } ( p )$ , with zero columns otherwise. Let $\Lambda ( p )$ be diagonal, with

$$
\Lambda _ { k k } ( p ) = \left\{ \begin{array} { l l } { \sqrt { n _ { k } ( p ) / N _ { a ( k ) } ( p ) } , } & { n _ { k } ( p ) > 0 , } \\ { 0 , } & { n _ { k } ( p ) = 0 . } \end{array} \right.\tag{C.1}
$$

An active group always has $N _ { a ( k ) } ( p ) > 0$ , so every division is well-defined.

Proposition C.1 (Normalized geometry under vector sharing). The injection operator satisfies

$$
\begin{array} { c } { { \displaystyle H ( p ) = G ( p ) \Lambda ( p ) S , \qquad G ( p ) ^ { \top } G ( p ) = \mathrm { d i a g } \left( { \bf 1 } \{ k \in \mathcal { Z } ( p ) \} \right) , } } \\ { { \displaystyle H ( p ) ^ { \top } H ( p ) = \mathrm { d i a g } \left( { \bf 1 } \{ r \in \mathcal { I } ( p ) \} \right) , \qquad \| H ( p ) W \| _ { F } ^ { 2 } = \displaystyle \sum _ { r \in \mathcal { I } ( p ) } \| v _ { \tau , r } \| _ { 2 } ^ { 2 } \leq \| W \| _ { F } ^ { 2 } . } } \end{array}\tag{C.2}
$$

Moreover, $H ( p ) H ( p ) ^ { \top }$ is the orthogonal projector onto corrections constant on each active row’s token set and zero elsewhere.

Proof. For an active group k assigned to row $r ,$ the corresponding product of scales is $n _ { k } ^ { - 1 / 2 } \sqrt { n _ { k } / N _ { r } } = N _ { r } ^ { - 1 / 2 }$ Excluded tokens and inactive groups contribute zero. This proves $\bar { H } = G \Lambda S$ . Active columns of G have disjoint supports and squared norm $n _ { k } / n _ { k } = 1$ , and the same argument gives squared norm $N _ { r } / N _ { r } = 1$ for active columns of H. Taking the trace of $W ^ { \top } H ^ { \top } H W$ proves the norm identity. The orthonormal active columns of H span exactly the stated correction space, so their outer products sum to its orthogonal projector. □

Row gradients. For any differentiable loss $F$ on $\widetilde { E } = E + H W$ , the chain rule gives

$$
\nabla _ { W } F = H ^ { \top } \nabla _ { \widetilde { E } } F , \qquad \nabla _ { v _ { \tau , r } } F = \frac { 1 } { \sqrt { N _ { r } } } \sum _ { t : a ( s _ { t } ) = r } \nabla _ { \widetilde { E } _ { t } } F \quad ( N _ { r } > 0 ) .\tag{C.3}
$$

The row gradient is zero when $N _ { r } = 0$ , and $\| \nabla _ { W } F \| _ { F } \leq \| \nabla _ { \widetilde { E } } F \| _ { F }$ . Normalization controls replication-induced energy, not gradient invariance: if all token gradients in a row equal $^ { g , }$ its gradient is $\sqrt { N _ { r } } g$

## C.3 First-order cost of sharing

Fix one prompt and a differentiable loss $F ,$ evaluated at zero update. Let $g _ { t } = \nabla _ { \widetilde { E } _ { t } } F \in \mathbb { R } ^ { d }$ , and fix $\rho > 0$ . For a nonempty eligible set S, define $\Gamma ( S )$ as in Equation 1.

Proposition C.2 (First-order cost of sharing). (i) The largest first-order decrease from an update w $/ \sqrt { | { \cal S } | }$ at every token in S, with $\| w \| _ { 2 } \leq \rho ,$ , is Γ(S). The right side of Equation 1 is the optimum for free per-token updates of total Frobenius norm at most $\rho .$ Equality holds exactly when all $g _ { t }$ in $s$ are equal.

(ii) For disjoint nonempty A, B, two separately normalized vectors with combined squared norm at most $\rho ^ { 2 }$ attain

$$
\Gamma _ { \mathrm { s p l i t } } = \sqrt { \Gamma ( A ) ^ { 2 } + \Gamma ( B ) ^ { 2 } } \geq \operatorname* { m a x } \{ \Gamma ( A \cup B ) , \Gamma ( A ) , \Gamma ( B ) \} .
$$

If $g _ { t } = 0$ on B, then $\Gamma ( A \cup B ) = \sqrt { | A | / | A \cup B | } \Gamma ( A )$

(iii) Let cos θ be the cosine between $\textstyle \sum _ { t \in { \mathcal { A } } } g _ { t }$ and $\textstyle \sum _ { t \in B } g _ { t }$ , set to one if either sum is zero. Then

$$
\Gamma _ { \mathrm { s p l i t } } ^ { 2 } - \Gamma ( \mathcal { A } \cup \mathcal { B } ) ^ { 2 } = \frac { \left( \sqrt { | \mathcal { B } | } \Gamma ( \mathcal { A } ) - \sqrt { | \mathcal { A } | } \Gamma ( \mathcal { B } ) \right) ^ { 2 } + 2 \sqrt { | \mathcal { A } | | \mathcal { B } | } \Gamma ( \mathcal { A } ) \Gamma ( \mathcal { B } ) ( 1 - \cos \theta ) } { | \mathcal { A } | + | \mathcal { B } | } .
$$

The first term, a norm gap, vanishes exactly when the mean gradients of $\mathcal { A }$ and B have equal norm. The second term, a direction gap, grows as cos θ falls for fixed $\Gamma ( A )$ and Γ(B).

Proof. The shared update has directional derivative $\langle w , \sum _ { t \in \mathcal { S } } g _ { t } \rangle / \sqrt { | { \cal S } | }$ . Minimizing this linear functional on the radius- $- \rho$ ball yields −Γ(S). Applying the same argument to the stacked free token updates gives the right side of Equation 1. With $\begin{array} { r } { \overline { { g } } = | \dot { S } | ^ { - 1 } \sum _ { t \in S } ^ { } g _ { t } } \end{array}$ , the identity

$$
\sum _ { t \in S } \| g _ { t } \| _ { 2 } ^ { 2 } - \frac { 1 } { | S | } \left\| \sum _ { t \in S } g _ { t } \right\| _ { 2 } ^ { 2 } = \sum _ { t \in S } \| g _ { t } - \overline { { g } } \| _ { 2 } ^ { 2 }
$$

proves the inequality and its equality condition.

For two vectors, the stacked gradient has norm $\sqrt { \Gamma ( \mathcal { A } ) ^ { 2 } + \Gamma ( \mathcal { B } ) ^ { 2 } } / \rho ,$ giving the stated optimum. A one-vector update w on $S = A \cup B$ is realized by $w _ { \mathcal { A } } = \sqrt { | \mathcal { A } | / | \mathcal { S } | }$ w and $\boldsymbol { w } \boldsymbol { B } = \sqrt { | B | / | S | } \boldsymbol { w }$ , with the same total norm. Thus splitting cannot reduce the optimum for this fixed prompt. Setting either vector to zero yields the other two lower bounds. The zero-gradient statement follows because $\bar { B }$ changes only the normalization factor, not the gradient sum.

For (iii), write $G _ { A }$ and $G _ { B }$ for the two gradient sums, so that $\| G _ { \cal A } \| _ { 2 } = \sqrt { | { \cal A } | } \Gamma ( { \cal A } ) / \rho , \| G _ { \cal B } \| _ { 2 } = \sqrt { | { \cal B } | } \Gamma ( { \cal B } ) / \rho ,$ and $\langle G _ { \mathcal { A } } , G _ { \mathcal { B } } \rangle = \| G _ { \mathcal { A } } \| _ { 2 } \| G _ { \mathcal { B } } \| _ { 2 }$ cos θ. Expanding $\Gamma ( A \cup B ) ^ { 2 } = \rho ^ { 2 } \| G _ { A } + G _ { B } \| _ { 2 } ^ { 2 } / ( | A | + | B | )$ ) gives

$$
( | A | + | B | ) \Gamma ( A \cup B ) ^ { 2 } = | A | \Gamma ( A ) ^ { 2 } + | B | \Gamma ( B ) ^ { 2 } + 2 \sqrt { | A | | B | } \Gamma ( A ) \Gamma ( B ) \cos \theta .
$$

Subtracting this from $( | \mathcal { A } | + | \mathcal { B } | ) \Gamma _ { \mathrm { s p l i t } } ^ { 2 }$ and completing the square proves the identity. The first term is zero exactly when $\| G _ { A } \| _ { 2 } / | A | = \| G _ { B } \| _ { 2 } / | B |$ □

The dilution statement is exact under its zero-gradient hypothesis for a fixed prompt. Appendix H.5 measures it across image resolutions.

## C.4 Sharing cost before training

Figure 2 measures the quantities of Proposition C.2 in the frozen Idefics2 backbone of Table 2 on Flowers. No vector is trained and no optimizer is involved. The measurement uses 512 demo-free samples from the training pool of that table, 256 to fit the update directions and 256 others to measure the loss. For each sample, $g _ { t }$ is the zero-update gradient of the answer CE of Equation 5 with respect to input embedding t. A prompt holds one token in each of LAST, ANSWER-CUE, FIRST, and IMAGE-GATE, 64 QUERY-IMAGE tokens, and 37.5 QUESTION-TEXT tokens on average, and no PREFIX-TEXT. The readout set A is LAST and ANSWER-CUE, and the context set B holds the other eligible tokens.

Update and gradient shares. One vector shared by all eligible tokens adds $w / \sqrt { | S | }$ to each of them, so the share of $\lVert \boldsymbol { w } \rVert _ { 2 } ^ { 2 }$ that a group receives equals its share of the tokens. For the gradient, let $\hat { g } _ { k }$ average $n _ { k } ^ { - 1 / 2 } \textstyle \sum _ { t : s _ { t } = k } g _ { t }$ over all 512 samples. By Equation C.3, gˆ is the mean gradient of an independent vector for group k, and $\rho \| \hat { g } _ { k } \| _ { 2 }$ is its value of Γ. Panel (a) reports each group’s share of $\sum _ { k } \| \hat { g } _ { k } \| _ { 2 } ^ { 2 }$ . The readout set receives 1.9% of the shared update and holds 76% of the gradient. QUERY-IMAGE receives 61% of the update and holds 0.002% of the gradient, so it comes close to the zero-gradient case of Proposition C.2(ii): it enlarges the normalization count without adding to the gradient sum. On the held-out samples, the first-order loss decrease per unit update norm is 12.5 along the direction of one shared vector and 27.1 along that of the readout vector alone.

Gradient per token. Panel (b) reports, for each token set S, the norm of $| S | ^ { - 1 } \textstyle \sum _ { \scriptstyle t \in S } g _ { t }$ after this vector is averaged over the 512 samples. The readout set reaches 19.2, 3.5 times FIRST, the largest context group at 5.5, and QUERY-IMAGE reaches 0.02. The readout value is below that of ANSWER-CUE (30.2) and of LAST (22.4) alone, because the mean gradients of these two tokens are nearly orthogonal, so their average is shorter than either of them.

Adding context. Panel (c) uses the unit descent directions $u _ { A }$ and $u _ { B }$ of the readout and context vectors, each the negative mean zero-update gradient of its vector over the fitting samples, scaled to unit norm. On the held-out samples, the readout vector is set to $a u \mathbf { \Theta } _ { A }$ and the context vector to c u<sub>B</sub>, and the panel reports the mean answer CE in nats. The zero update gives 3.05. With $c = 0$ , the loss falls to 1.12 at $a = 0 . 2$ and 1.07 at $a = 0 . 3$ . Each context norm $c \in \{ 0 . 1 , \bar { 0 . 2 } , 0 . \bar { 3 } \}$ lowers the whole curve, and $c = 0 . 3$ reaches 0.55 at $a = 0 . 2$ . The two directions have cosine similarity 0.03, so one vector on $\mathcal { A } \cup \mathcal { B }$ cannot point along both. All directions come from zero-update gradients and all norms stay at most 0.3, so the figure describes the neighborhood of the zero update.

## C.5 Proof of Theorem 3.1

Tied coordinates. Fix the common group counts used in the theorem, and let $w \in \mathbb { R } ^ { q d }$ stack the transposed rows of W. The normalized independent-group coordinates of its tied update are

$$
v = Q _ { q } w , \qquad Q _ { q } = ( \Lambda S ) \otimes I _ { d } , \qquad \mathcal { U } _ { q } = \cot ( Q _ { q } ) , \qquad P _ { q } = Q _ { q } Q _ { q } ^ { \top } .\tag{C.4}
$$

Here ⊗ is the Kronecker product and $I _ { d }$ is the identity on $\mathbb { R } ^ { d }$ . The active columns of $\Lambda S$ are orthonormal, by summing $n _ { k } / N _ { \tau }$ <sub>r</sub> over groups assigned to row $r ,$ and inactive columns are zero. Hence $P _ { q }$ is an orthogonal projector, even with empty rows. For each $v \in \mathcal { U } _ { q } , w = Q _ { q } ^ { \top }$ v realizes v with inactive coordinates zero and $\| w \| _ { 2 } = \| v \| _ { 2 }$ . Inactive full-group coordinates have no effect on the margins and can also be set to zero.

Interpreting $\delta _ { q } .$ . To interpret $\delta _ { q } .$ , let $g _ { j k } = \nabla _ { v _ { k } } h _ { j } ( 0 ) / \sqrt { n _ { k } }$ for active group $k ,$ where $v _ { k } \in \mathbb { R } ^ { d }$ is its independent vector. This is the mean gradient with respect to that group’s input embeddings. For active row $r ,$ put $\overline { { { g } } } _ { j r } \ : = \ :$ $\scriptstyle \sum _ { k : a ( k ) = r , n _ { k } > 0 } ( n _ { k } / N _ { r } ) g _ { j k }$ . Projection onto $\mathcal { U } _ { q }$ replaces each gradient block by $\sqrt { n _ { k } } \overline { { g } } _ { j r } , { \bf s } { \bf o }$

$$
\delta _ { q } ^ { 2 } = \operatorname* { m a x } _ { j } \sum _ { \substack { r : N _ { r } > 0 k : a ( k ) = r , n _ { k } > 0 } } n _ { k } \| g _ { j k } - \overline { { g } } _ { j r } \| _ { 2 } ^ { 2 } .\tag{C.5}
$$

Thus readout/context sharing discards only within-row group contrasts. This identity does not assume that their gradients agree.

ProofofTheorem 3.1. The tied feasible set is contained in the unrestricted group-coordinate ball, so $m _ { q } ( \rho ) \leq m _ { R } ( \rho )$ Continuity and compactness give an unrestricted maximizer $v ^ { \star }$ . Its projection $\bar { \overline { { v } } } = P _ { q } v ^ { \star }$ is tied-feasible and has norm at most $\rho .$

Gradient Lipschitzness along every segment from zero to a point in the ball gives

$$
h _ { j } ( \boldsymbol { v } ) = h _ { j } ( 0 ) + \nabla h _ { j } ( 0 ) ^ { \top } \boldsymbol { v } + r _ { j } ( \boldsymbol { v } ) , \qquad | r _ { j } ( \boldsymbol { v } ) | \leq \frac { \kappa } { 2 } \| \boldsymbol { v } \| _ { 2 } ^ { 2 } .\tag{C.6}
$$

Indeed, the remainder is the integral of $\langle \nabla h _ { j } ( t v ) - \nabla h _ { j } ( 0 ) , v \rangle$ over $t \in [ 0 , 1 ]$ . Since $v ^ { \star } - \overline { { v } }$ is orthogonal to $\mathcal { U } _ { q }$

$$
\begin{array} { r l } & { h _ { j } ( \overline { { v } } ) \geq h _ { j } ( v ^ { \star } ) - \| ( I - P _ { q } ) \nabla h _ { j } ( 0 ) \| _ { 2 } \| v ^ { \star } - \overline { { v } } \| _ { 2 } - \frac { \kappa } { 2 } \big ( \| v ^ { \star } \| _ { 2 } ^ { 2 } + \| \overline { { v } } \| _ { 2 } ^ { 2 } \big ) } \\ & { \qquad \geq h _ { j } ( v ^ { \star } ) - \rho \delta _ { q } - \kappa \rho ^ { 2 } . } \end{array}
$$

Taking the minimum over the finite margin collection and using feasibility of v proves the upper bound. Affine margins admit $\kappa = 0$ , which gives equality of attainable margins when $\delta _ { q } = 0$ □

From margins to cross-entropy. For any desired margin $\gamma \in \mathbb { R }$ , an unrestricted update attaining every margin at least $\gamma + \rho \delta _ { q } + \bar { \kappa } \rho ^ { 2 }$ implies the existence of a tied update attaining every margin at least γ on these contexts. If the collection includes every competing token for each supervised prefix in a vocabulary of size $C \geq 2$ , the tied logits $z \in \mathbb { R } ^ { C }$ at each prefix satisfy $z _ { a } - z _ { c } \ge \gamma$ for every $c \neq a .$ , where a is the correct token. Thus

$$
- \log \operatorname { s o f t m a x } ( z ) _ { a } = \log \left( 1 + \sum _ { c \neq a } e ^ { z _ { c } - z _ { a } } \right) \leq c \gamma : = \log { \left( 1 + ( C - 1 ) e ^ { - \gamma } \right) } .\tag{C.7}
$$

Averaging preserves this CE bound.

Varying counts. The theorem requires a single $Q _ { q }$ for all its contexts. Equal group counts suffice, and so do common normalized within-row proportions. With varying proportions, the shared row w induces different full-group vectors $Q _ { q } ( p ) \imath$ w. For example, with two groups, one scalar row, and count profiles (1, 1) and (1, 4), these vectors are ${ ( w / \sqrt { 2 } , w / \sqrt { 2 } ) }$ and $( w / \sqrt { 5 } , 2 w / \sqrt { 5 } )$ . No single full-group vector realizes both corrections for nonzero w. Conversely, the full-group vector $( 1 , 0 )$ is not tied on the first profile. Thus the two shared embedding-correction families need not be nested across varying-count prompts, and the theorem applies to each count profile separately.

## D Paired supervision and demo-free risk

Equation 6 is a sum of answer-supervised source and target CE losses. It contains no hidden-state matching or soft-label objective. Throughout the whole-sample results below, n denotes the number of training query–answer pairs, not the optimizer’s minibatch size. The source renderings are fixed for these comparisons. The same deterministic arguments apply to a source loss averaged over a prescribed demo-sampling rule whenever that averaged loss satisfies the stated continuity or differentiability assumptions.

## D.1 Proof of Proposition 3.2

Proof. Let $w ^ { + } = w - \eta g$ . Integrating the target gradient along the segment gives

$$
\begin{array} { r l r } {  { \mathcal { L } _ { \mathrm { t g t } } ( w ^ { + } ) - \mathcal { L } _ { \mathrm { t g t } } ( w ) = - \eta \int _ { 0 } ^ { 1 } \langle \nabla \mathcal { L } _ { \mathrm { t g t } } ( w - t \eta g ) , g \rangle d t } } \\ & { } & { \leq - \eta \langle g _ { \mathrm { t g t } } , g \rangle + \frac { \beta \eta ^ { 2 } } { 2 } \| g \| _ { 2 } ^ { 2 } , ~ } \end{array}
$$

which proves Equation 7.

Finite-step conditions. Write $a = \langle g _ { \mathrm { t g t } } , g \rangle = \| g _ { \mathrm { t g t } } \| _ { 2 } ^ { 2 } + \langle g _ { \mathrm { t g t } } , g _ { \mathrm { s r c } } \rangle$ . If $a > 0$ , then $g \neq 0$ . For $\beta > 0$ , a step satisfying $0 < \eta < 2 a / ( \beta \| g \| _ { 2 } ^ { 2 } )$ and the segment smoothness assumption strictly lowers target loss. For $\beta = 0$ , every positive step satisfying that assumption does so.

Comparison with a target-only step. For comparison with a target-only step, additionally assume the same gradient Lipschitz bound on the segment from w to $w - \eta g _ { \mathrm { t g t } }$ . Applying the integral remainder bound in the other direction yields

$$
\mathcal { L } _ { \mathrm { t g t } } ( w - \eta g _ { \mathrm { t g t } } ) - \mathcal { L } _ { \mathrm { t g t } } ( w ) \geq - \eta \Vert g _ { \mathrm { t g t } } \Vert _ { 2 } ^ { 2 } - \frac { \beta \eta ^ { 2 } } { 2 } \Vert g _ { \mathrm { t g t } } \Vert _ { 2 } ^ { 2 } .
$$

Subtracting this from the paired-step upper bound gives

$$
\begin{array} { r l } & { \displaystyle \mathcal { L } _ { \mathrm { t g t } } ( w - \eta g ) - \mathcal { L } _ { \mathrm { t g t } } ( w - \eta g _ { \mathrm { t g t } } ) \leq - \eta \langle g _ { \mathrm { t g t } } , g _ { \mathrm { s r c } } \rangle } \\ & { \displaystyle \qquad + \frac { \beta \eta ^ { 2 } } { 2 } \big ( \| g \| _ { 2 } ^ { 2 } + \| g _ { \mathrm { t g t } } \| _ { 2 } ^ { 2 } \big ) . } \end{array}\tag{D.1}
$$

If $c = \langle { g } _ { \mathrm { t g t } } , g _ { \mathrm { s r c } } \rangle > 0$ and $\beta > 0 ,$ , the paired step is strictly better at the same step size whenever $0 ~ < ~ \eta ~ <$ $2 c / [ \beta ( \| g \| _ { 2 } ^ { 2 } + \| g _ { \mathrm { t g t } } \| _ { 2 } ^ { 2 } ) ]$ and both segment assumptions hold. For $\beta = 0$ , positivity of c suffices on those segments. These comparisons concern plain gradient steps from the same parameter value.

## D.2 Compatibility of the empirical objectives

Fix $\rho > 0$ and the comparison class $\mathcal { W } _ { q , \rho } = \{ W \in \mathbb { R } ^ { q \times d } : \| W \| _ { F } \leq \rho \}$ . Assume both whole-sample branch losses are continuous on it. Define

$$
{ \mathcal L } _ { b } ^ { \star } = \operatorname* { m i n } _ { W \in { \mathcal W } _ { q , \rho } } { \mathcal L } _ { b } ( W ) , \qquad { \mathcal L } _ { \mathrm { s r c l t g t } } = \operatorname* { m i n } _ { W \in \arg \operatorname* { m i n } _ { U \in { \mathcal W } _ { q , \rho } } { \mathcal L } _ { \mathrm { t g t } } ( U ) } { \mathcal L } _ { \mathrm { s r c } } ( W ) - { \mathcal L } _ { \mathrm { s r c } } ^ { \star } .\tag{D.2}
$$

The nonnegative gap $\mathcal { C } _ { \mathrm { s r c | t g t } }$ is the source-loss sacrifice required to remain target-optimal. The ball and the target minimizer set are nonempty compact sets, so all displayed minima exist.

Proposition D.1 (Target cost of incompatible supervision). If $\widehat { W } \in \mathcal { W } _ { q , \rho }$ satisfies $\begin{array} { r } { \mathcal L ( \widehat W ) \leq \operatorname* { m i n } _ { W \in \mathcal W _ { q , \rho } } \mathcal L ( W ) + \varepsilon _ { \mathrm { o p t } } } \end{array}$ for $\varepsilon _ { \mathrm { o p t } } \geq 0$ , then

$$
0 \leq \mathcal { L } _ { \mathrm { t g t } } ( \widehat { W } ) - \mathcal { L } _ { \mathrm { t g t } } ^ { \star } \leq \mathcal { C } _ { \mathrm { s r c } | \mathrm { t g t } } + \varepsilon _ { \mathrm { o p t } } .\tag{D.3}
$$

Proof. Choose a target minimizer $W ^ { \dagger }$ attaining the source minimum in Equation D.2. Approximate paired optimality implies

$$
\mathcal { L } _ { \mathrm { t g t } } ( \widehat { W } ) + \mathcal { L } _ { \mathrm { s r c } } ( \widehat { W } ) \leq \mathcal { L } _ { \mathrm { t g t } } ^ { \star } + \mathcal { L } _ { \mathrm { s r c } } ( W ^ { \dagger } ) + \varepsilon _ { \mathrm { o p t } } .
$$

Use $\mathcal { L } _ { \mathrm { s r c } } ( \widehat { W } ) \geq \mathcal { L } _ { \mathrm { s r c } } ^ { \star }$ and subtract $\mathcal { L } _ { \mathrm { t g t } } ^ { \star }$ to obtain the upper bound. The lower bound follows from the definition of the target minimum. □

Shared minimizers. The symmetric source bound follows by exchanging branches. If the two losses share a minimizer, both compatibility gaps vanish and an approximate paired minimizer is within $\varepsilon _ { \mathrm { o p t } }$ of each branch optimum on the stated ball.

A counterexample. Shared answer labels do not ensure compatible responses. For a scalar binary-logit example, margins $h _ { \mathrm { s r c } } ( w ) = w$ and $h _ { \mathrm { t g t } } ( w ) = - w$ give losses $\log ( 1 + e ^ { - w } )$ and $\log ( 1 + e ^ { w } )$ . Their gradients cancel at zero, and no w gives both margins a positive value. Replacing the source margin by 2w makes a paired gradient step at zero increase target loss. The compatibility gap $\mathcal { C } _ { \mathrm { s r c | t g t } }$ is the quantity that separates this case from compatible supervision.

## D.3 Demo-free risk guarantees

Let $n \geq 1$ and let $( x _ { i } , y _ { i } ) _ { i = 1 } ^ { n }$ be independent, identically distributed query–answer pairs from the deployment distribution of one task. The frozen model, target rendering, structural partition, row assignment, and radius $\rho > 0$ are fixed independently of this sample. Assume the losses are measurable in the example, and define

$$
{ \mathcal { R } } _ { \operatorname { t g t } } ( W ) = \mathbb { E } _ { ( x , y ) } \ell ( y \mid \operatorname { r e n d e r } ( \varnothing , x ) , W ) .\tag{D.4}
$$

The empirical mean is $\mathcal { L } _ { \mathrm { t g t } }$ in Equation 6. Source demos may be reused or chosen using the training sample: the concentration argument uses only the n independent target examples. Paired renderings, answer tokens, and repeated minibatches do not increase this sample count.

Assumption D.2 (Bounded and Lipschitz target losses). There are constants $0 < M _ { \ell } < \infty$ and $0 \leq G _ { \ell } < \infty$ such that, for every target example in the distribution’s support and all $W , W ^ { \prime } \in \mathcal { W } _ { q , \rho } ,$ , its loss lies in $[ 0 , M _ { \ell } ]$ and

$$
| \ell ( y ~ \vert ~ p , W ) - \ell ( y ~ \vert ~ p , W ^ { \prime } ) | \leq G _ { \ell } \| H ( p ) ( W - W ^ { \prime } ) \| _ { F } , \qquad p = \mathrm { r e n d e r } ( \emptyset , x ) .\tag{D.5}
$$

These conditions are uniform over examples and the comparison class. Uniform bounds on logits and input derivatives over the ball supply such constants. The radius is an analysis device, not an extra penalty, clipping rule, or projection in STAVE.

Theorem D.3 (Uniform target-risk deviation). Under Assumption D.2, for fixed $\epsilon \in ( 0 , \rho ]$ and $\delta \in ( 0 , 1 )$ , with probability at least 1 − δ,

$$
\operatorname* { s u p } _ { W \in \mathcal { W } _ { q , \rho } } | \mathcal { R } _ { \mathrm { t g t } } ( W ) - \mathcal { L } _ { \mathrm { t g t } } ( W ) | \leq \mathfrak { B } _ { n } ( \epsilon , \delta ) : = 2 G _ { \ell } \epsilon + M _ { \ell } \sqrt { \frac { q d \log ( 1 + 2 \rho / \epsilon ) + \log ( 2 / \delta ) } { 2 n } } .\tag{D.6}
$$

Here ϵ is the covering radius, δ the failure probability, and $\mathfrak { B } _ { n }$ the deviation bound.

Proof. Proposition C.1 makes every target loss $G _ { \ell ^ { - 1 } }$ Lipschitz in $\| W - W ^ { \prime } \| _ { F }$ . The comparison class is a Euclidean ball of dimension qd. A maximal set separated by more than ϵ is an ϵ-net. Its disjoint open balls of radius $\epsilon / 2$ lie inside a ball of radius $\rho + \epsilon / 2 .$ , so a volume comparison bounds the net size by $( 1 + 2 \rho / \epsilon ) ^ { q d }$

For completeness, if $X \in [ 0 , M _ { \ell } ]$ , the log moment-generating function $\psi ( s ) = \log \mathbb { E } e ^ { s ( X - \mathbb { E } X ) }$ has $\psi ( 0 ) = \psi ^ { \prime } ( 0 ) = 0$ and $\psi ^ { \prime \prime } { \overset { . } { ( s ) } } \leq M _ { \ell } ^ { 2 } / 4$ . The second derivative is a variance under an exponentially tilted distribution, still supported on $[ 0 , M _ { \ell } ]$ , and any such variance is at most its second moment about $M _ { \ell } / 2 .$ , hence at most $M _ { \ell } ^ { 2 } / 4$ . Integrating twice gives $\dot { \psi } ( s ) \dot { \leq } s ^ { 2 } M _ { \ell } ^ { 2 } / 8$ . Independence and the exponential Markov inequality, optimized over s for each tail, therefore give at every fixed net point

$$
\begin{array} { r } { \operatorname* { P r } \big ( | { \mathcal L } _ { \mathrm { t g t } } - { \mathcal R } _ { \mathrm { t g t } } | > t \big ) \le 2 \exp ( - 2 n t ^ { 2 } / M _ { \ell } ^ { 2 } ) . } \end{array}
$$

A union bound over the net yields the square-root term in Equation D.6. Every point of the ball is within ϵ of a net point. The empirical and population losses each change by at most $G _ { \ell } \epsilon .$ , proving the result. Continuity in W permits the supremum to be taken over a countable dense subset, so the event is measurable. □

Corollary D.4 (Conditional target risk of paired training). On the event of Theorem D.3, suppose the empirical source loss is continuous on $\mathcal { W } _ { q , \rho }$ and $\widehat { W }$ satisfies the radius and paired optimization condition of Proposition D.1. Then

$$
\mathcal { R } _ { \mathrm { t g t } } ( \widehat { W } ) - \operatorname* { i n f } _ { W \in \mathcal { W } _ { q , \rho } } \mathcal { R } _ { \mathrm { t g t } } ( W ) \leq \mathcal { C } _ { \mathrm { s r c } | \mathrm { t g t } } + \varepsilon _ { \mathrm { o p t } } + 2 \mathfrak { B } _ { n } ( \epsilon , \delta ) .\tag{D.7}
$$

Proof. Uniform deviation and Proposition D.1 give

$$
\mathcal { R } _ { \mathrm { t g t } } ( \widehat { W } ) \leq \mathcal { L } _ { \mathrm { t g t } } ^ { \star } + \mathcal { C } _ { \mathrm { s r c } | \mathrm { t g t } } + \varepsilon _ { \mathrm { o p t } } + \mathfrak { B } _ { n } .
$$

The same event implies $\mathcal { L } _ { \mathrm { t g t } } ^ { \star } \leq \operatorname* { i n f } _ { W } \mathcal { R } _ { \mathrm { t g t } } ( W ) + \mathfrak { B } _ { n }$ . Combining the two inequalities proves the claim.

Capacity term. The uniform event covers any sample-dependent choice of $\widehat { W }$ in the fixed ball, including one fitted using source prompts. The capacity term uses 2d coordinates for STAVE and 7d for independent group vectors, a comparison made at common radius and loss constants. No common-count assumption is needed for this statistical result. When the class is chosen from $M _ { \mathrm { c f g } }$ predetermined configurations, a union bound replaces $\log ( 2 / \delta )$ by $\log ( 2 M _ { \mathrm { c f g } } / \delta )$

## E Inference cost

Prompt tokens and storage. Let $T _ { \mathrm { s r c } }$ and $T _ { \mathrm { t g t } }$ be decoder-input lengths with and without demos for the same query, including expanded visual tokens. STAVE inference uses exactly $T _ { \mathrm { t g t } }$ prompt tokens. Counts and assignments can be constructed in $O ( T _ { \mathrm { t g t } } )$ time from fixed structural metadata. Adding the scaled vectors costs $O ( T _ { \mathrm { t g t } } \bar { d } )$ operations and can be performed in place. Storage is 2d task-specific scalars, independent of query count and demo length. No independent full-group matrix needs to be stored for inference.

KV cache. For a decoder with $N _ { \mathrm { l a y e r } }$ layers, $h _ { \mathrm { k v } }$ KV heads per layer, and head dimension $d _ { h }$ , a conventional uncompressed prompt cache stores

$$
\mathrm { K V E l e m e n t s } ( T ) = 2 N _ { \mathrm { l a y e r } } T h _ { \mathrm { k v } } d _ { h } .\tag{E.1}
$$

Here $T$ is the number of cached tokens and the factor two counts keys and values. For equal precision, batch size, and output length, STAVE and zero-shot inference have the same cache shape. The values in that cache differ. Relative to keeping demos, the saved prompt-cache elements are $2 N _ { \mathrm { l a y e r } } ( T _ { \mathrm { s r c } } - T _ { \mathrm { t g t } } ) h _ { \mathrm { k v } } d _ { h }$ when the source prompt is longer. Both caches grow identically per subsequently generated token, although the number of generated tokens can differ.

Prefill computation. Under conventional dense attention, the prefill attention term is $O ( N _ { \mathrm { l a y e r } } T ^ { 2 } d )$ and the positionwise decoder terms are linear in $T$ for a fixed architecture. STAVE preserves the target prompt’s decoder dimensions and adds only the embedding-update pass. A soft prompt of length P instead changes the attention-size term from $T ^ { 2 }$ to $( T + P ) ^ { \bar { 2 } }$ . A KV prefix adds attention-memory entries even when it computes no new attention queries.

Visual preprocessing and latency. The visual encoder and connector process the query image as in zero-shot inference. Demo removal avoids the demo prompt tokens and, when demos would otherwise be encoded at inference, their visual preprocessing. Measured latency also depends on kernels, precision, batching, and stopping behavior, so Appendix E.1 runs every method under one setting.

Training cost. Both paired branches run the frozen model, and gradients propagate through the decoder to the input vectors. Backbone optimizer states and weight updates are absent, while activation storage and backward computation remain, and the source branch includes demos. This one-time training cost is separate from the per-query inference cost above.

## E.1 Measured inference cost

Table 7 reports measured per-query cost for the Idefics2-8B VQAv2 entries of Table 1. Every method loads the checkpoint that produces the score in that table and runs on the same H200 inside one job, with batch size 1, fp16 weights, three beams, and the same 100 validation samples after warmup. TTFT is one prefill call with a single new token and is the like-for-like measure of prompt cost. Because the methods emit answers of different lengths, the Fixed-20 column forces exactly 20 generated tokens for every method. TTFT and the decode column time the generation call alone, with decode taken from the 20-token generation time, while Fixed-20 is end to end and also includes the processor pass over the prompt text and images. State is the stored task state. STAVE only adds its vectors to the input embeddings before the first decoder block, so its decoder computation during prefill and decoding is that of zero-shot inference. The zero-shot and STAVE rows therefore share one value in the TTFT, decode, and Fixed-20 columns, the mean of their two measurements, which differ by less than 1%.

STAVE keeps the prompt length, prefill FLOPs, and KV cache of zero-shot inference, at 128 tokens. MimIC, HiFICL, and LIVE run the same prompt through an attention wrapper, which multiplies TTFT by 1.03 to 1.10 relative to STAVE. LoRA keeps the prompt but adds low-rank branches in the decoder, the visual encoder, and the connector, which multiplies TTFT by 1.18 and the per-token decode cost by 1.41. ICL with 32 demos needs 60 times the TTFT, 31 times the prefill FLOPs, and 52 GiB more peak memory than STAVE.

STAVE has the lowest stored state of any trained method here and matches the TTFT, per-token decode cost, and fixed-length latency of zero-shot inference, below every other row, and Table 1 gives it the highest accuracy on this model and dataset, so nothing in this comparison trades accuracy for speed against it.

Figures 9 and 10 report six cost dimensions against the demo count on Idefics2-8B VizWiz and OK-VQA and Qwen-VL-7B OK-VQA, measured on one A100 40GB with batch size 1, fp16 weights, greedy decoding, at most 10 new tokens, and a shared stop rule, as medians over 500 test queries. They use the full group set. Prompt tokens, prefill FLOPs, KV cache, TTFT, and peak activation memory depend only on the prompt and are identical for the two-vector configuration. Answer latency also depends on the generated answer length and includes the CPU pass that builds the group assignment (4 to 14 ms). Qwen-VL with 32 demos exceeds the 40 GB memory and is omitted. On all three model and dataset pairs, STAVE keeps the prompt tokens, prefill FLOPs, KV cache, and TTFT of zero-shot inference, while ICL cost grows with every added demo. At 16 demos, ICL multiplies the TTFT of STAVE by 8 to 13 and its peak activation memory by 14 to 17.

Table 7: Measured per-query inference cost on Idefics2-8B VQAv2.
<table><tr><td>Method</td><td>State (KiB)</td><td>Prompt tok.</td><td>TTFT (ms)</td><td>Decode (ms/tok)</td><td>Fixed-20 (ms)</td><td>Peak mem. (GiB)</td></tr><tr><td>Zero-shot</td><td>0</td><td>128</td><td>98</td><td>23.9</td><td>560</td><td>16.5</td></tr><tr><td>8-shot ICL</td><td>0</td><td>1,026</td><td>1,430</td><td>24.0</td><td>2,021</td><td>29.8</td></tr><tr><td>16-shot ICL</td><td>0</td><td>1,922</td><td>2,854</td><td>24.4</td><td>3,605</td><td>43.0</td></tr><tr><td>32-shot ICL</td><td>0</td><td>3,717</td><td>5,877</td><td>26.1</td><td>6,933</td><td>68.7</td></tr><tr><td>LoRA (r=16)</td><td>68,704</td><td>128</td><td>116</td><td>33.6</td><td>763</td><td>16.5</td></tr><tr><td>LIVE</td><td>512</td><td>128</td><td>101</td><td>27.4</td><td>629</td><td>16.5</td></tr><tr><td>MimIC</td><td>1,028</td><td>128</td><td>106</td><td>31.6</td><td>713</td><td>16.5</td></tr><tr><td>HiFICL</td><td>8,704</td><td>128</td><td>108</td><td>34.3</td><td>767</td><td>16.5</td></tr><tr><td>STAVE</td><td>32</td><td>128</td><td>98</td><td>23.9</td><td>560</td><td>16.5</td></tr></table>

## E.2 Stored task state

Table 8 lists the task-specific state each method stores per task and backbone at hidden dimension d = 4096. STAVE stores two vectors. The full group set stores one vector per structural token group. ICV stores one vector per layer, added after the MLP of each block. MTV stores one vector per attention head across layers, and LIVE stores one vector and one scale per layer. I2CL stores one attention-output vector and one MLP-output vector per layer, with two scales for each. MimIC and HiFICL store trained attention modules, and LoRA stores low-rank weight updates. Parameter counts are read back from the loaded module of each method and match the main tables. Sizes assume fp32 storage. The Idefics2 MimIC entry is the original encoder, which adds one shift vector per layer rather than learning a projection of the query states. The largest state here, LoRA on LLaVA, is 2,399 times that of STAVE.

## E.3 Performance overview settings

Figure 1 uses reported results from four tables. From Table 1, it shows LLaVA and Idefics2-8B on VQAv2, OK-VQA, and COCO. From Table 2, it shows Idefics2-8B on VizWiz, DTD, and Flowers and Qwen-VL-7B on VizWiz and OK-VQA. From Table 3, it shows Pythia 2.8B. TV benchmark accuracy aggregates its 18 tasks, weighted by test queries. Bar labels show the reported accuracy percentages or COCO CIDEr scores at the precision printed in each table. Each performance bar’s length is the ratio of its score to STAVE’s score for that model and dataset, with the inner edge representing zero.

![](images/34173a6769a33a0c70ab274ca76b37d59d98da207fc7b600fef7b30165cfe08e.jpg)  
Figure 9: Prompt tokens, prefill FLOPs, and KV cache against the demo count.

The params sector uses the Idefics2-8B parameter counts of Table 1. Its bar length is $1 - \log _ { 1 0 } ( P _ { m } / P _ { \mathrm { S T A V E } } ) / 4$ of the full length, where $P _ { m }$ is method m’s parameter count, so STAVE has the full length and each factor of 10 removes a quarter of it. The latency sector shows the fixed-20 latencies printed in Table 7. Its bar length is $t _ { \mathrm { S T A V E } } / t _ { m }$ of the full length, where $t _ { m }$ is method m’s latency, so a longer bar is faster. These measurements are for Idefics2-8B VQAv2 under the settings of that table. ICL uses 16 demos in the Table 1 and latency sectors, four in Table 2, and 15 on the TV benchmark. LoRA uses rank 8 on the TV benchmark and rank 16 elsewhere. Every value in the overview is copied from the cited table.

![](images/942cd51294e1d3dd6a5d2c5e976251041afdc11a218d589377365c45d924e2d8.jpg)  
Figure 10: TTFT, answer latency, and peak activation memory against the demo count.

## F Implementation details

## F.1 Backbones

Table 9 lists the eleven frozen backbones used in the experiments. The hidden dimension d fixes the size of the learned matrix, which holds 2d parameters on every backbone, and $N _ { \mathrm { l a y e r } }$ is the number of decoder blocks. Image tokens is the number of decoder-input tokens one query image occupies after the frozen visual encoder and connector expand it, under the image settings used here. The readout vector covers at most a few tokens on every backbone, so these counts are the scale the context vector is normalized against.

Idefics2 and Idefics3 run without image splitting, and Qwen2.5-VL images are resized to 448 pixels, so each of the six LMMs has a fixed count. Idefics2 appears under two checkpoints, HuggingFaceM4/idefics2-8b-base in Table 1 and HuggingFaceM4/idefics2-8b in Table 2, written Idefics2-8B-base and Idefics2-8B-instruct where both appear, and the model written as LLaVA in Table 1 is llava-hf/llava-interleave-qwen-7b-hf.

Table 8: Stored task state per task and backbone (d = 4096, fp32).
<table><tr><td>Method</td><td>Stored state</td><td>Parameters</td><td>Size</td></tr><tr><td>Zero-shot, ICL</td><td>none</td><td>0</td><td>0</td></tr><tr><td>Single vector (TV, FV)</td><td>one embedding vector</td><td>4,096</td><td>16 KiB</td></tr><tr><td>ICV</td><td>one vector per layer</td><td>131,072</td><td>512 KiB</td></tr><tr><td>MTV</td><td>one vector per head</td><td>131,072</td><td>512 KiB</td></tr><tr><td>LIVE</td><td>one vector and one scale per layer</td><td>131,104</td><td>512 KiB</td></tr><tr><td>I2CL</td><td>two vectors and four scales per layer</td><td>262,272</td><td>1.0 MiB</td></tr><tr><td>MimIC (Idefics2 / LLaVA)</td><td>attention modules</td><td>263,168 / 17,040,384</td><td>1.0 MiB / 65.0 MiB</td></tr><tr><td>HiFICL</td><td>attention modules</td><td>2,228,224</td><td>8.5 MiB</td></tr><tr><td>LoRA (Idefics2 / LLaVA)</td><td>low-rank weight updates</td><td>17,588,224 / 19,652,608</td><td>67.1 MiB / 75.0 MiB</td></tr><tr><td>STAVE</td><td>two embedding vectors</td><td>8,192</td><td>32 KiB</td></tr></table>

The other LMM checkpoints are Qwen/Qwen-VL-Chat, OpenGVLab/InternVL3\_5-8B, Qwen/Qwen2. 5-VL-7B-Instruct, and HuggingFaceM4/Idefics3-8B-Llama3. The LLM checkpoints are the EleutherAI pythia-2.8b, pythia-6.9b, and pythia-12b releases together with EleutherAI/gpt-j-6B and huggyllama/llama-7b. Panel (a) names the language model each backbone is built from, and several released checkpoints are instruction-tuned: the Idefics3 decoder starts from Llama-3.1-8B-Instruct, the LLaVA-Interleave decoder starts from Qwen1.5-7B-Chat, and Qwen-VL and Qwen2.5-VL are used in their chat and instruct releases. Training and evaluation run on NVIDIA A100 (40GB) and H200 GPUs.

Table 9: Frozen backbones.  
(a) LMMs
<table><tr><td>Backbone</td><td>Language model</td><td>Vision encoder and connector</td><td>d</td><td> $N _ { \mathrm { l a y e r } }$ </td><td>Img.</td></tr><tr><td>Idefics2-8B</td><td>Mistral-7B</td><td>SigLIP-SO400M, perceiver resampler</td><td>4096</td><td>32</td><td>64</td></tr><tr><td>LLaVA-Interleave-7B</td><td>Qwen1.5-7B</td><td>SigLIP-SO400M, MLP projector</td><td>4096</td><td>32</td><td>729</td></tr><tr><td>Qwen-VL-7B</td><td>Qwen-7B</td><td>OpenCLIP ViT-bigG, cross-attention</td><td>4096</td><td>32</td><td>256</td></tr><tr><td>InternVL3.5-8B</td><td>Qwen3-8B</td><td>InternViT-300M, pixel shuffle, MLP</td><td>4096</td><td>36</td><td>256</td></tr><tr><td>Qwen2.5-VL-7B</td><td>Qwen2.5-7B</td><td>Native-resolution ViT, patch merger</td><td>3584</td><td>28</td><td>256</td></tr><tr><td>Idefics3-8B</td><td>Llama-3.1-8B</td><td>SigLIP-SO400M, pixel shuffle, MLP</td><td>4096</td><td>32</td><td>169</td></tr></table>

(b) LLMs
<table><tr><td>Backbone</td><td>d</td><td> $N _ { \mathrm { l a y e r } }$ </td><td>Family</td></tr><tr><td>Pythia 2.8B</td><td>2560</td><td>32</td><td>GPT-NeoX</td></tr><tr><td>Pythia 6.9B</td><td>4096</td><td>32</td><td>GPT-NeoX</td></tr><tr><td>Pythia 12B</td><td>5120</td><td>36</td><td>GPT-NeoX</td></tr><tr><td>GPT-J 6B</td><td>4096</td><td>28</td><td>GPT-J</td></tr><tr><td>LLaMA 7B</td><td>4096</td><td>32</td><td>LLaMA</td></tr></table>

## F.2 VQAv2, OK-VQA, and COCO

Table 1 follows the protocol of MimIC and HiFICL, and all methods, STAVE included, share the same data, prompts, and decoding. The training set is the first 1,000 samples of the released shuffle at seed 3407, which is also the demo pool. The trained methods are evaluated without demos on the full evaluation set, with 10,000 VQAv2 questions, the 5,046 validation questions of the official OK-VQA v1.1 release, and 5,000 COCO Karpathy validation images. ICL draws the demos of each query from the same pool without replacement, with seeds 41, 42, and 3407. Zero-shot and 8-shot ICL are scored on the full evaluation set, and 16-shot and 32-shot ICL on the first 1,000 samples of the shuffled evaluation order. Zero-shot has no demos and deterministic decoding, so the three seeds give the same score. The prompts are the released templates, an instruction-style prefix for the VQA tasks and a caption-style prefix for COCO (Appendix F.6). Decoding uses three beams on both backbones, at most 20 new tokens, length penalty 0, and batch size 1. Post-processing and scorers are the released ones, and CIDEr uses pycocoevalcap with CoreNLP 3.4.1. Published CIDEr values reported as fractions are multiplied by 100.

MimIC and HiFICL report the best-performing epoch, and we follow this protocol. Tables 2 and 10 instead select the checkpoint on validation samples (Appendix F.3). STAVE trains its $2 d = 8 { , } 1 9 2$ parameters for five epochs with the source and target CE losses weighted 1 to 1, using AdamW with weight decay $1 \dot { 0 } ^ { - 3 }$ and a cosine schedule with 10% warmup. The learning rate is $1 \bar { 0 } ^ { - 3 }$ or $3 \times 1 0 ^ { - 3 }$ set per model and dataset, each training step uses $K = 4$ paired examples $( K = 1 6$ on LLaVA COCO), and the source prompt holds one demo on OK-VQA and eight on COCO.

All compared methods run with the authors’ released code. LIVE (Peng et al., 2024), MimIC (Jiang et al., 2025), and HiFICL (Li et al., 2026) follow the settings of their papers. PT follows the setting of Lester et al. (2021) with vocabulary initialization. The three PT modes learn 20 virtual tokens each on the prompt without demos and train for five epochs with AdamW. We tune the PT learning rate between $1 0 ^ { - 3 }$ and $3 \times 1 0 ^ { - 2 }$ . PT is sensitive to the learning rate and hard to tune, and a learning rate that trains stably at one random seed can collapse at another seed on the same model and dataset.

LoRA appears in Tables 1 and 3 with different ranks and settings. In Table 1 it follows the LoRA configuration released with HiFICL. It applies rank 16 to the attention query, key, value, and output projections of the language model and, on Idefics2, of the perceiver connector, and to the query, key, and value projections of the visual encoder. The MLP connector of LLaVA is not adapted. It trains with AdamW at learning rate $5 \times 1 0 ^ { - 4 }$ , weight decay $1 0 ^ { - 3 }$ , and a cosine schedule with 10% warmup. It uses batch size 2 with 8 accumulation steps, the setting of the released LoRA script. It follows the five-epoch schedule. The rank 8 setting of Table 3 is given in Appendix F.4.

## F.3 VizWiz, OK-VQA, and fine-grained classification

Tables 2 and 10 follow the protocol of MTV (Huang et al., 2024) and STV (Ma et al., 2026), and every method uses the STAVE split. The pool is a subsample of the training split drawn with seed 41, the validation block follows the fitting block, and the test set is the full held-out split. MimIC draws its teacher demos only from its training records. On VizWiz and OK-VQA the validation questions are those of STAVE.

Table 2 pairs a weaker and a stronger frozen checkpoint. Qwen-VL-7B, used in its chat release, is the stronger one on the two VQA datasets, with zero-shot accuracy of 45.49% on VizWiz and 57.51% on OK-VQA against 37.74% and 49.44% for Idefics2-8B. Four demos raise its average from 70.35% to 73.53%, whereas they lower the Idefics2-8B average from 70.57% to 69.72%. The pair therefore tests whether the learned vectors help both when the frozen model leaves much room for improvement and when it already answers many questions correctly. STAVE has the highest VizWiz and OK-VQA accuracy and the best average with both checkpoints. On Qwen-VL-7B OK-VQA, PT and the three trained per-layer interventions fall below zero-shot, while STAVE raises accuracy from 57.51% to 62.39%.

All compared methods start from the authors’ released code, and we adapt each one to the datasets and protocol of these tables for a fair comparison. Following their papers, MimIC and HiFICL train for 10 epochs with AdamW (Loshchilov & Hutter, 2017) at learning rate $5 \times \mathrm { \bar { 1 0 } ^ { - 3 } }$ and weight decay $1 0 ^ { - 3 }$ , batch size 2 with 2 accumulation steps, a cosine schedule with 10% warmup, and gradient clipping at 1.0. MimIC uses a four-shot teacher with loss weights 0.5 for CE and 1 for alignment. HiFICL uses rank 8 and eight virtual tokens. Decoding is greedy. Each of the 10 epochs is scored on the validation samples with the test metric, ties break by the lower mean teacher-forced answer CE and then by the earlier epoch, and only the selected epoch is evaluated on the test split.

STAVE trains its 2d parameters with the source and target CE losses weighted 1 to 1, using AdamW with weight decay 0 and a cosine schedule that decays the learning rate to zero. The source prompt holds one demo on VizWiz and OK-VQA and two on DTD, Flowers, and CUB. Each run takes 250 to 4,000 training steps of $K = 4$ or $K = 8$ paired examples, with a learning rate between $1 0 ^ { - 3 }$ and $1 0 ^ { - 2 }$ set per model and dataset. As for MimIC and HiFICL, and unlike Table 1, the checkpoint is selected on the validation samples, and only the selected checkpoint is evaluated on the test split. The three PT modes in Table 2 also learn 20 virtual tokens each on the prompt without demos. They train for 500 training steps of $K = 4$ samples with AdamW at weight decay $1 0 ^ { - 2 }$ and a cosine schedule after a linear warmup, with the CE of the first answer token as the loss. The learning rate is $1 0 ^ { - 2 }$ in most entries. When a run collapsed at one seed, that seed was trained again at another learning rate between $3 \times 1 0 ^ { - 4 }$ and $5 \times 1 0 ^ { - 3 }$ , so the three seeds of an entry do not always share one learning rate.

## F.4 TV benchmark

Table 3 uses the 18 tasks of the TV benchmark (Hendel et al., 2023) with five frozen LLMs. Each task has one split, drawn with seed 41, of at most 256 training, 64 development, and 200 test queries. Every baseline uses the same prompts, training samples, and test queries as STAVE, and every per-task choice of a baseline is made on the demo-free development split, never on the test split. Each baseline runs with the authors’ released code and follows the settings of its paper. TV and FV (Todd et al., 2024) choose the injection layer per task, as in their papers. SITE (Park et al., 2026)

![](images/b4f4068277bf01305c2004ecbb5dc5fd0efd2cffcbecd93490f2ac32ba07b85c.jpg)  
STAVE Strongest baseline

![](images/6bce1a74454002b9f4630b8dba6ec434fa74c5176d2e34211cedd982b9595146.jpg)

![](images/13d59a7517b5562c50a35c6fea6ee75f607b6efe1c74c55a33582ae1bdd192ba.jpg)

![](images/870030a3e9bd717d7200de7d0bfbfa203ef1311da0449659dfcb32ac922e7edd.jpg)  
Figure 11: Learning-rate sensitivity on four model and dataset pairs of Table 1.

![](images/94739d68dd0f17a5dd3ce5e8019d0fce086f5d32a44e50bf7a9138b44c9cb454.jpg)

![](images/374b437d6c1f3e50fab3d707c4f67daa25d9ac03986cae5eb723293870b9c184.jpg)

![](images/587675fee4cc998475ec7c1ceaa7535c92e28c6495f22be410fb5c718d7bafb6.jpg)  
Figure 12: Score after each training Figure 13: Batch size $K$ on two Figure 14: Number of source demos epoch, relative to the strongest base- Idefics2 datasets (axes start above on Idefics2 VQAv2. line. zero).

runs all three stages, with the task embedding averaged over 50 prompts of 10 demos. PT again learns 20 virtual tokens. LoRA follows Hu et al. (2021) and applies rank 8 with scaling $\alpha = 1 6$ to the attention query and value projections of every decoder block, or to the fused query, key, and value projection on Pythia. It trains with AdamW at weight decay $0 . 0 1 { \overset { \cdot } { , } } K = 8 .$ , and at most 500 training steps with early stopping on the development loss, and its learning rate is $1 0 ^ { - \overline { { 4 } } }$ or $1 0 ^ { - 3 } ,$ , chosen per task by the development loss. STAVE picks its configuration per task by the lowest demo-free development CE, with one demo in the source prompt. The chosen configurations use AdamW with weight decay 0.01 on 84 of the 90 backbone and task pairs and 0 on the rest, 500 or 1,000 training steps, $K = 8$ paired examples per step on 74 pairs and 16 or 32 on the rest, and a learning rate between 0.01 and 0.5. The context vector uses 0.003 to 0.3 times the learning rate of the readout vector. The average weights each task by its number of test queries.

## F.5 Hyperparameter sensitivity

Figures 11 to 14 vary one training setting at a time on four model and dataset pairs of Table 1, LLaVA VQAv2 and Idefics2 VQAv2, OK-VQA, and COCO. Every run uses seed 41 and the 1,000 training samples of that table and reports its best epoch, the checkpoint selection used for its entries. Unless a figure varies them, runs use K = 4 paired examples per training step in Equation $^ { 6 , }$ five epochs, weight decay 0.001, and one source demo on OK-VQA or eight on COCO. Figures 12 and 13 use the learning rate with the highest score in Figure 11. The strongest baseline for a model and dataset is the best non-STAVE entry of its column in Table 1. Scores are accuracy (%) for VQAv2 and OK-VQA and CIDEr for COCO.

Learning rate. STAVE stays above the strongest baseline over most of the learning-rate range in Figure 11. On LLaVA VQAv2 every learning rate from $5 \times 1 0 ^ { - 4 } \mathrm { t o } \ 5 \times 1 0 ^ { - 3 }$ is above it, and on Idefics2 COCO every one except $1 0 ^ { - 3 }$ . On Idefics2 VQAv2 the score decreases as the learning rate grows and remains above the strongest baseline up to $1 0 ^ { - 3 }$ Idefics2 OK-VQA peaks at $1 0 ^ { - 3 }$ and exceeds the strongest baseline from $7 \times 1 0 ^ { - 4 } t o 2 \times 1 0 ^ { - 3 }$

Training epochs. Figure 12 follows the score after every epoch. All four model and dataset pairs are above the strongest baseline from the fourth epoch on, including the last one, so the gain over the strongest baseline does not depend on selecting a single epoch.

Batch size. Figure 13 varies K on the two Idefics2 datasets and compares STAVE with itself. K = 4 gives the highest score on both datasets. With the learning rate and the number of epochs fixed, a larger K means fewer training steps per epoch, 63 at K = 16 against 250 at K = 4, and the OK-VQA score decreases as K grows.

Number of source demos. Figure 14 varies the number of source demos on Idefics2 VQAv2 from one to eight, with every other setting fixed. The score changes by at most 0.40 points, from 73.06% to 73.46%, and does not rise or fall with the number of demos. Every setting remains above the strongest baseline, PT-Pre at 72.75%. The number of source demos therefore has little effect on STAVE.

## F.6 Dataset and prompt examples

This subsection shows the decoded, demo-free query block and its reference answer. The VQA and captioning queries come from the evaluation splits. The source branch used during training places demos before the same query block. Angle-bracketed image markers mark where the image shown beside the prompt is placed. Each visual sample is identified by its dataset ID or original file path.

## F.6.1 VQA and captioning

## VQAv2: question ID 575633002

![](images/3de2e541280c2043d385acd87ba2a6164ec594380ef1064aafa96ae7d040673d.jpg)

## Decoded LLaVA-Interleave prompt

Instruction: provide an answer to the question. Use the image to answer.

Image:<image>Question:How many sides does the STOP sign have?

Reference answer: 8. The ten annotations are {8, 8, 2, 8, 8, 8, 8, 8, 8, 8}.

## OK-VQA: question ID 4516835

![](images/7e8e53cd95691fe9fdbd61d624fdcbd8555cc196776f08afc250dd87fad7ccb3.jpg)

## Decoded prompt

<image>What black veggie is on this pizza? Answer:

Reference answer: olives. Eight annotators answered “olives” and two answered “olive.”

## VizWiz: question ID VizWiz\_val\_00001816

![](images/9bb49b70076320de058c296d4f374172705c91543f900b3b37be9264e593a905.jpg)

## Decoded prompt

First carefully understand the given examples.

Then use the given image and answer the question in the same way as the examples.

If the question can not be answered, respond unanswerable. <image>What is this? Answer:

Reference answer: coffee creamer. The ten annotations include “coffee creamer,” “coffee mate,” and more specific variants such as “french vanilla coffee creamer.”

![](images/cc17823172b03721c59971c2e6eea9b7e36bb05ba39e969915ec84ac947b156a.jpg)

## F.6.2 Fine-grained visual classification

The three classification datasets use the same two-choice query format, and evaluation uses the same format on the test split. Each example below is the first query of the fitting split, which the target branch sees without demos. The options are shuffled deterministically. The answer is an option letter, while the class name below makes the label explicit.

![](images/d9ab0428693c78db33ff7064f1cb652c34df9bc114bb18853bd9bbc7c850f312.jpg)  
F.6.3 TV benchmark tasks

Every TV benchmark task uses the demo-free template example:<input>->. The answer tokens follow the arrow during supervision. In the translation and linguistic tasks, the input and the answer each start with one space, as in example: chien->, and a pair is kept only when its answer is a single token for the backbone’s tokenizer. The algorithmic and knowledge tasks add no space and apply no single-token filter. The entries below are input-output pairs that remain after filtering on every backbone, or records from the deterministic algorithmic task generators.

Each task draws up to 200 test queries, and a task with a small input space yields fewer. Next letter and Previous letter have 25 possible inputs and the two case tasks have 26, so their test splits hold 12 or 13 queries. The single-token filter also shrinks some translation and linguistic tasks. English to Spanish holds 33 to 110 test queries depending on the backbone, while French to English and Spanish to English hold 200. An unweighted mean over the 18 tasks would give a 12-query split the same influence as a 200-query split, so every TV benchmark average in this paper weights each task by its number of test queries. The reported figure is therefore the accuracy over the pooled test queries of the suite, 2,138 to 2,202 per backbone. Every column uses the same weighting, and selection is unaffected because it is made per task on the development split.

<table><tr><td colspan="2">Algorithmic tasks</td></tr><tr><td>Task Prompt</td><td></td></tr><tr><td>Next letter</td><td>example:a-&gt; example:b-&gt;</td></tr><tr><td>Previous letter example:a,b,c-&gt;</td><td></td></tr><tr><td>List first</td><td></td></tr><tr><td>List last</td><td></td></tr><tr><td>To uppercase</td><td></td></tr><tr><td>To lowercase</td><td></td></tr></table>

<table><tr><td colspan="4">Translation tasks</td></tr><tr><td colspan="4"></td></tr><tr><td>Task</td><td>Prompt</td><td></td><td>Reference answer</td></tr><tr><td>French to English</td><td>example: chien-&gt;</td><td></td><td>dog</td></tr><tr><td>Spanish to English</td><td>example: perro-&gt;</td><td></td><td>dog</td></tr><tr><td>English to French</td><td>example: good-&gt;</td><td></td><td>bon</td></tr><tr><td>English to Spanish</td><td>example: two-&gt;</td><td></td><td>dos</td></tr></table>

<table><tr><td>Task</td><td>Prompt</td><td>Reference answer</td></tr><tr><td>Present to gerund</td><td>example: run-&gt;</td><td>running</td></tr><tr><td>Present to past</td><td>example: run-&gt;</td><td>ran</td></tr><tr><td>Plural to singular</td><td>example: children-&gt;</td><td>child cold</td></tr><tr><td>Antonym</td><td>example: hot-&gt;</td><td></td></tr></table>

<table><tr><td colspan="3">Knowledge tasks</td></tr><tr><td>Task</td><td>Prompt</td><td>Reference answer</td></tr><tr><td>Country to capital</td><td>example:Germany-&gt;</td><td>Berlin</td></tr><tr><td>Person to language</td><td>example:Gilad Atzmon-&gt;</td><td>Hebrew</td></tr><tr><td>Location to continent</td><td>example:Paris-&gt;</td><td>Europe</td></tr><tr><td>Location to religion</td><td>example:Edwin of Northumbria-&gt;</td><td>Christian</td></tr></table>

## G Additional results

## G.1 More recent LMMs

Table 10 extends the VizWiz and OK-VQA comparison of Table 2 to InternVL3.5-8B, Qwen2.5-VL-7B, and Idefics3- 8B, under the protocol of Appendix F.3, and each trained method is run three times. STAVE has the best mean on five of the six columns, with 14 to 36 times fewer parameters than LIVE and MimIC and 238 to 306 times fewer than HiFICL, and it exceeds 4-shot ICL by more than 10 points on every column.

## G.2 VQAv2 answer types

Figure 15 splits the VQAv2 scores of Table 1 by the three official answer types on all 10,000 evaluation questions, using one run of each method. Each point is the accuracy gain of STAVE over LIVE, MimIC, or HiFICL, with a paired

Table 10: Comparison on VizWiz and OK-VQA with more recent LMMs.
<table><tr><td></td><td colspan="3">InternVL3.5-8B</td><td colspan="3">Qwen2.5-VL-7B</td><td colspan="3">Idefics3-8B</td></tr><tr><td>Method</td><td># Params (M)</td><td>VizWiz</td><td>OK-VQA</td><td># Params (M)</td><td>VizWiz</td><td>OK-VQA</td><td># Params (M)</td><td>VizWiz</td><td>OK-VQA</td></tr><tr><td>Zero-shot</td><td>一</td><td>27.7</td><td>19.4</td><td>一</td><td>48.1</td><td>54.5</td><td></td><td>29.9</td><td>45.2</td></tr><tr><td>4-shot ICL</td><td></td><td>56.5</td><td>47.4</td><td>一</td><td>54.5</td><td>43.0</td><td></td><td>37.6</td><td>45.9</td></tr><tr><td>LIVE</td><td>0.15 (×18.00)</td><td> $6 3 . 5 4 _ { \pm 0 . 7 1 }$ </td><td> $4 9 . 9 8 _ { \pm 0 . 1 3 }$ </td><td> $0 . 1 0 \left( \times 1 4 . 0 0 \right)$ </td><td> $5 2 . 9 7 _ { \pm 9 . 0 8 }$ </td><td> $5 5 . 3 1 _ { \pm 0 . 3 3 }$ </td><td> $0 . 1 3 \ : ( \times 1 6 . 0 0 )$ </td><td> $4 6 . 9 3 _ { \pm 1 . 9 6 }$ </td><td> $3 9 . 2 5 _ { \pm 1 . 0 3 }$ </td></tr><tr><td>MimIC</td><td>0.30 (×36.14)</td><td> $6 6 . 7 9 _ { \pm 1 . 4 2 }$ </td><td> $5 7 . 3 6 { \scriptstyle \pm 0 . 4 2 }$ </td><td> $0 . 2 0 \left( \times 2 8 . 1 1 \right)$ </td><td> $7 3 . 4 4 _ { \pm 0 . 8 7 }$ </td><td> $6 4 . 5 5 { \scriptstyle \pm 0 . 4 7 }$ </td><td> $0 . 2 6 \left( \times 3 2 . 1 3 \right)$ </td><td>66.71±0.86</td><td> $5 7 . 2 2 { \scriptstyle \pm 0 . 2 2 }$ </td></tr><tr><td>HiFICL</td><td>2.5 (×306.00)</td><td> $6 8 . 0 9 { \scriptstyle \pm 0 . 4 1 }$ </td><td> $5 8 . 7 4 { \scriptstyle \pm 0 . 1 7 }$ </td><td> $1 . 7 \left( \times 2 3 8 . 0 0 \right)$ </td><td> $\underline { { 7 1 . 7 4 } } \pm 0 . 5 1$ </td><td> $5 9 . 8 1 _ { \pm 0 . 5 3 }$ </td><td> $2 . 2 \ : ( \times 2 7 2 . 0 0 )$  —</td><td> $6 5 . 6 0 { \scriptstyle \pm 0 . 7 7 }$ </td><td> $5 6 . 5 3 { \scriptstyle \pm 0 . 6 4 }$ </td></tr><tr><td>STAVE</td><td>0.008 (×1.00)</td><td> ${ \bf 6 8 . 3 7 { \scriptstyle \pm 0 . 7 1 } }$ </td><td> ${ \bf 5 8 . 9 4 _ { \pm 0 . 1 1 } }$ </td><td> $\mathbf { 0 . 0 0 7 \left( \times 1 . 0 0 \right) }$ </td><td>—  $7 1 . 3 4 _ { \pm 0 . 2 4 }$ </td><td> ${ \bf 6 4 . 9 1 _ { \pm 0 . 2 9 } }$ </td><td> $\mathbf { 0 . 0 0 8 } \left( \mathbf { \times 1 . 0 0 } \right)$ </td><td> ${ \bf 6 8 . 6 6 _ { \pm 0 . 1 8 } }$ </td><td> ${ \bf 5 8 . 7 4 _ { \pm 0 . 3 7 } }$ </td></tr></table>

bootstrap 95% interval over questions (2,000 resamples). STAVE matches or exceeds every baseline on every answer type on both backbones. The other type holds half of the questions and most open-vocabulary answers, and there the gain is 1.3 to 2.6 points, with an interval that excludes zero against every baseline on both backbones.

(a)  
![](images/488377b50fa43bfcade3ad9aa56e11192706b2e6e86eb95740c34e52e7bb9c77.jpg)

(b)  
(c)  
![](images/6ee7ece1fa858da3be2b9f1cd0899987b38e7d9e004449dc7b76132b5c6b0480.jpg)

![](images/68825508a8e39d1dbf5c3b3a190253fab4d9e670210cf4af4a84fd5f865c275d.jpg)  
10,000 VQAv2 questions per backbone: yes/no 3,654, number 1,386, other 4,960. Error bars: paired bootstrap 95% CI, 2,000 resamples.  
Figure 15: VQAv2 accuracy gain of STAVE over each trained baseline, by answer type.

## H Ablation details

## H.1 Demos in the prompt on COCO Captioning

Table 1 evaluates every method without demos. Table 11 puts demos back into the prompt at inference and changes nothing else. Every method is read from its own published checkpoint on the first 1,000 images of the COCO Karpathy validation split, and the demos of a query are drawn from the same pool in the same order for every method, so a column compares methods on identical prompts. Columns are the number of demos, and the mismatched column gives four demos whose captions belong to other images. Avg. drop averages the change from the demo-free score over the five demo conditions.

MimIC and HiFICL lose almost all captioning ability on LLaVA as soon as one demo is present, falling from about 135 CIDEr to below 5, and their loss does not shrink as demos are added. On Idefics2, MimIC holds its score within about 3 CIDEr of the demo-free one until the demos are mismatched, and HiFICL does not. Paired STAVE stays within 6 CIDEr of its demo-free score in every condition on both backbones, and within 0.7 CIDEr on Idefics2, including under mismatched demos. Target-only training falls to between 11 and 26 CIDEr on LLaVA, and the paired objective repairs that case.

The released evaluator turns a generation into a caption by cutting it at the first of a list of stop words, and the HiFICL Idefics2 entries at four and at mismatched demos hold 130 and 160 generations of 1,000 that open with a template marker, which places a stop word at the first token and leaves an empty caption. Those two entries are scored after one leading marker is removed, which recovers 93 and 59 captions. The rest are empty generations, so both entries stay far below the demo-free score.

Table 11: COCO CIDEr with demos added to the prompt at inference (first 1,000 Karpathy validation images).
<table><tr><td>Method</td><td>0</td><td>1</td><td>2</td><td>4</td><td>8</td><td>mismatched</td><td>Avg. drop</td></tr><tr><td colspan="8">LLaVA-Interleave, COCO</td></tr><tr><td>MimIC</td><td>134.6</td><td>3.3</td><td>3.0</td><td>3.4</td><td>3.9</td><td>4.3</td><td>-131.0</td></tr><tr><td>HiFICL</td><td>138.0</td><td>3.4</td><td>3.0</td><td>3.4</td><td>4.5</td><td>4.0</td><td>-134.4</td></tr><tr><td>STAVE (source-only)</td><td>134.0</td><td>135.2</td><td>134.8</td><td>135.8</td><td>137.4</td><td>135.7</td><td>+1.8</td></tr><tr><td>STAVE (target-only)</td><td>135.2</td><td>11.3</td><td>21.8</td><td>21.6</td><td>25.5</td><td>15.0</td><td>-116.2</td></tr><tr><td>STAVE (paired)</td><td>138.8</td><td>133.9</td><td>134.7</td><td>135.5</td><td>134.4</td><td>133.0</td><td>-4.5</td></tr><tr><td colspan="8">Idefics2, COCO</td></tr><tr><td>MimIC</td><td>140.4</td><td>139.3</td><td>139.1</td><td>137.3</td><td>137.6</td><td>94.5</td><td>-10.8</td></tr><tr><td>HiFICL</td><td>137.7</td><td>114.6</td><td>129.7</td><td>66.5</td><td>136.0</td><td>21.4</td><td>-44.1</td></tr><tr><td>STAVE (source-only)</td><td>138.1</td><td>140.4</td><td>141.4</td><td>140.8</td><td>141.0</td><td>139.8</td><td>+2.6</td></tr><tr><td>STAVE (target-only)</td><td>141.5</td><td>141.8</td><td>142.0</td><td>142.4</td><td>143.4</td><td>138.1</td><td>+0.0</td></tr><tr><td>STAVE (paired)</td><td>141.8</td><td>141.1</td><td>141.6</td><td>142.5</td><td>142.1</td><td>142.1</td><td>+0.0</td></tr></table>

## H.2 Object hallucination by training objective

Table 12 scores the Idefics2 COCO captions of the three objectives of Table 4 with the CHAIR metrics of Table 5. Paired training has the highest recall, and its captions name absent objects less often than target-only captions on both CHAIR measures. Source-only captions name absent objects least but recall the fewest objects.

Table 12: Object hallucination by training objective on Idefics2 COCO.
<table><tr><td>Objective</td><td>CHAIRs↓</td><td>CHAIRi↓</td><td>Recall ↑</td></tr><tr><td>Paired</td><td>2.62</td><td>1.76</td><td>44.81</td></tr><tr><td>Target-only</td><td>2.84</td><td>1.93</td><td>44.47</td></tr><tr><td>Source-only</td><td>2.36</td><td>1.64</td><td>43.08</td></tr></table>

## H.3 Training objective on the TV benchmark

The paired objective has two branches. The source branch trains the learned vectors inside a one-demo prompt and the target branch trains them in the demo-free prompt. Table 13 removes one branch at a time. STAVE-source sets the target weight to zero, STAVE-target sets the source weight to zero, and STAVE-paired is the configuration of Table 3. Each entry reuses the configuration that Table 3 selected for that backbone and task, with no further tuning, and is evaluated twice on the same test queries: without a demo, and with the same learned vectors placed in a one-demo prompt. Accuracy weights each task by its test queries. The TV bench rows of Table 4 average the three LLMs of Table 13.

Each control holds only in the prompt format it was trained in. STAVE-source loses 2.4 to 25.4 points when the demo is removed, and STAVE-target loses up to 4.3 points when one is added. STAVE-paired stays within half a point of itself in both formats and is above each control in the format that control was not trained for. The one entry where a control is ahead is Pythia 6.9B with a demo, where STAVE-source is 0.2 points above STAVE-paired. The same pattern holds on Pythia 2.8B and Pythia 12B, where STAVE-source loses 4.4 and 24.5 points without a demo. SITE is listed for reference and is read the same way. It loses 0.8 to 1.0 points when a demo is added, a milder form of the STAVE-target pattern, so STAVE-paired is the only entry that holds its accuracy in both formats.

## H.4 Structural token groups

Figure 4(a) and Table 14 hold the training configuration fixed and vary the group-to-vector assignment or the updated tokens. Every configuration trains for five epochs at the learning rate and demo count of the Table 1 entry of its model and dataset, and reports its best epoch, the checkpoint selection used for that entry. Configuration names list the tokens covered by each learned vector, with vectors separated by a slash. The readout tokens are the final prompt token and the answer cue. The context tokens are all other eligible tokens. The image tokens are the query-image embeddings. In “Text / image”, the text tokens are all eligible tokens outside the query image, and in “Readout / text / image”, they exclude the readout tokens. “Readout / context” is the configuration used in the main experiments. A configuration marked “only” learns one vector and leaves all other tokens unchanged.

Table 13: Training objective ablation on the TV benchmark (accuracy, %). Best objective in bold.
<table><tr><td></td><td colspan="3">Pythia 6.9B</td><td colspan="2">GPT-J6B</td><td colspan="3">LLaMA 7B</td></tr><tr><td>Method</td><td>w/o demo w/ 1 demo</td><td></td><td>Δ</td><td>w/o demo w/ 1 demo</td><td>Δ</td><td>w/o demo w/ 1 demo</td><td></td><td>Δ</td></tr><tr><td>SITE</td><td>90.4</td><td>89.4</td><td>-1.0</td><td>91.6 90.6</td><td>-1.0</td><td>92.1</td><td>91.3</td><td>-0.8</td></tr><tr><td>STAVE-source</td><td>66.6</td><td>92.0</td><td>+25.4</td><td>82.8 92.3</td><td>+9.5</td><td>89.6</td><td>92.0</td><td>+2.4</td></tr><tr><td>STAVE-target</td><td>91.3</td><td>88.1</td><td>-3.2</td><td>91.7 87.4</td><td>-4.3</td><td>91.6</td><td>91.2</td><td>-0.4</td></tr><tr><td>STAVE-paired</td><td>91.6</td><td>91.8</td><td>+0.2</td><td>92.3 92.4</td><td>+0.1</td><td>92.3</td><td>92.5</td><td>+0.2</td></tr></table>

Table 14: Group ablation on COCO (CIDEr) and VQAv2 (accuracy, %). Best: bold, second: underlined.
<table><tr><td></td><td colspan="2">COCO</td><td colspan="2">VQAv2</td></tr><tr><td>Configuration</td><td>Idefics2</td><td>LLaVA</td><td>Idefics2</td><td>LLaVA</td></tr><tr><td colspan="5">One vector</td></tr><tr><td>Context only</td><td>132.65</td><td>129.57</td><td>72.84</td><td>75.51</td></tr><tr><td>Readout only</td><td>132.24</td><td>129.85</td><td>71.78</td><td>75.86</td></tr><tr><td>All tokens</td><td>131.95</td><td>129.18</td><td>72.75</td><td>75.74</td></tr><tr><td colspan="5">Two vectors</td></tr><tr><td>Readout / context</td><td>134.38</td><td>131.78</td><td>73.46</td><td>76.12</td></tr><tr><td>Last / rest</td><td>133.28</td><td>128.98</td><td>73.33</td><td>75.77</td></tr><tr><td>Text / image</td><td>133.35</td><td>129.12</td><td>72.67</td><td>75.98</td></tr><tr><td colspan="5">Three vectors</td></tr><tr><td>Readout / text / image</td><td>133.79</td><td>130.79</td><td>73.62</td><td>75.96</td></tr><tr><td colspan="5">One vector per group</td></tr><tr><td>Per group</td><td>133.91</td><td>130.19</td><td>72.72</td><td>75.84</td></tr></table>

One-vector configurations. “All tokens” ties every eligible group to one vector. “Readout only” learns the readout vector and leaves the context tokens unchanged. “Context only” learns the context vector and leaves the readout tokens unchanged. Excluded groups are zeroed rather than reassigned. The LLaVA prompt has no image delimiters, so its IMAGE-GATE group is empty. Demos, BOS tokens, padding, and answer tokens remain excluded in every configuration.

Two-vector, three-vector, and per-group configurations. “Last / rest” learns one vector at the final prompt token and one vector shared by every other eligible token, so the answer cue joins the context tokens instead of the final token. “Text / image” learns one vector for the text tokens, including IMAGE-GATE, and one vector for the image tokens. “Readout / text / image” keeps the readout vector and splits the context tokens into one vector for the query-image tokens and one vector for the other context tokens. “Per group” is the full group set, with a separate vector for every structural token group. Every configuration recomputes the count normalization on its merged groups, so each learned vector is divided by the square root of the number of tokens it updates in the prompt.

What the comparisons support. Figure 4(a) shows the mean of the two backbones in Table 14, on axes that start above zero. Overall, “Readout / context” is the best configuration. One learned vector is not enough: “All tokens”, “Readout only”, and “Context only” are below it for every model and dataset, by up to 1.7 points on Idefics2 VQAv2 and up to 2.6 CIDEr on LLaVA COCO. Two-vector partitions that mix readout and context tokens are also below it for every model and dataset, “Text / image” by up to 2.7 CIDEr and “Last / rest” by up to 2.8 CIDEr. More vectors bring no consistent gain. “Per group” is below it for every model and dataset, and “Readout / text / image” is above it only on Idefics2 VQAv2, by 0.16 points, inside the paired-bootstrap interval against “Readout / context”, which spans about half a point on VQAv2 (2,000 resamples over validation samples). Appendix I.3 examines what each vector changes in the trained matrix.

## H.5 Allocation of a shared update

Resolution sweep. Figure 16 tests the dilution of Equation 1 by changing only the number of image tokens. From 64 to 4,096 image tokens, one vector shared by all eligible tokens loses 81% of its first-order decrease on OK-VQA and stays within about 3% of the prediction with zero gradient at the image tokens of Proposition C.2. The readout/context split at the same total norm keeps its effect.

The figure uses the frozen Qwen2.5-VL-7B backbone of   
Table 10 with the demo-free OK-VQA prompt of that table.   
Each image is resized to $s \times s$ pixels with s from 224 to   
1792, which gives $( s / 2 8 ) ^ { 2 }$ image tokens and leaves every text   
token unchanged. Besides the image, the prompt holds one   
FIRST token, one IMAGE-GATE token, about nine question   
tokens, and two readout tokens, ANSWER-CUE and LAST.   
The same 160 training-pool samples are used at every size,   
96 to fit the update directions and 64 to measure the loss. For   
each configuration, the direction is the negative mean zero  
update gradient of its vector over the fitting samples, scaled   
to unit total norm, so that it changes the input embeddings by   
Figure 16: One shared vector against the readout/contextone in Frobenius norm. The reported value is the first-order one in Frobenius norm. The reported value is the first-order

![](images/866161b15f884d88bfec444b075134b4ba638a6ab4ae7cd98e6281e50370c6ec.jpg)

split as the image grows (Qwen2.5-VL, OK-VQA).decrease of the answer loss of Equation 5 along this direction, averaged over the held-out samples. The prediction with zero gradient at the image tokens is the value of one vector on the text tokens, multiplied by $\sqrt { N _ { \mathrm { t e x t } } / N _ { 1 } }$ , where $N _ { \mathrm { t e x t } }$ counts the eligible text tokens and $N _ { 1 }$ all eligible tokens. The readout/context configuration ties LAST and ANSWER-CUE to one vector and all other eligible tokens, including the image, to a second vector. Two vectors on random token sets with the readout and context counts, under three partition seeds, never exceed one shared vector by more than 0.3%, so the gain comes from separating the readout tokens rather than from the second vector. Updates of norm 0.1 give the same pattern in the actual loss. At 64 image tokens one shared vector lowers it by 0.047 nats and the readout/context split by 0.115 nats, and at 4,096 image tokens by 0.0002 and 0.167 nats. Smaller norms are not informative on this backbone, because bf16 rounding moves the loss of a single sample by up to about 0.1 nats under any perturbation of its embeddings. Table 15 lists the values for OK-VQA and for VizWiz, whose prompt also carries a 39-token instruction, so that the shared vector falls later there. Real photos fall inside this range. The default Qwen2.5-VL processor limits each image to 12,845,056 pixels and rounds its sides to multiples of 28. On 1,500 OK-VQA images drawn at random from the same pool, it gives 234 to 391 image tokens between the 10th and 90th percentiles, with a median of 345. On 1,500 VizWiz images the median is 1,610 and the 90th percentile 2,552.

Native resolution. The LLaVA measurements below use the frozen LLaVA backbone of Table 1 on VQAv2, OK-VQA, and COCO Captioning, with the prompt templates and structural token groups of the main experiments. All prompts are demo-free, the loss is the answer CE of Equation 5 in nats per answer token, and all images come from the training pool, never from an evaluation split. Each configuration learns one or two vectors, which are the rows of W in Equation 2.

Effect of one token. The eligible tokens form three sets: the readout tokens (LAST and ANSWER-CUE), the query-image tokens, and the remaining context tokens, which are all text tokens in the LLaVA prompts. For each set, one vector is tied to its tokens with the count normalization of Equation 2, and all other tokens stay unchanged. On 300 prompts per task, the negative mean gradient of this vector at zero update is estimated on one half and scaled to unit norm, and the first-order loss decrease along this direction is averaged over the other half. Dividing it by the square root of the number of tokens in the set gives the average decrease for a unit shift at one token of the set. A readout token exceeds a query-image token by a factor of about 19,000 on VQAv2 and OK-VQA and about 14,000 on COCO. With the same halves, query-image tokens contribute 0.2% of the first-order decrease of one shared vector on VQAv2 and 0.04% on COCO, although they form 95% and 99% of the eligible tokens.

Equal update size. For each configuration, the update direction is the negative mean zero-update gradient of its vectors over 96 images, scaled to unit total norm. Because the columns of $H ( \bar { p } )$ in Equation 2 are orthonormal when every row of W is nonempty, vectors of total norm $\rho$ change the input embeddings by exactly $\rho$ in Frobenius norm. The loss is measured on 64 other images at $\rho = 0 . 0 1$ , a value fixed before the measurement, so no optimizer or learning rate is involved. The random configurations assign the eligible tokens of each prompt at random to two vectors with the readout and context counts, under three partition seeds. Table 16 reports the resulting loss decreases. At the same update norm, the readout/context split lowers the held-out loss 3.4 to 7.5 times as much as one shared vector, while random splits with the same counts stay within 5% of it.

Table 15: First-order loss decrease per unit update as the image grows (Qwen2.5-VL).
<table><tr><td>Image tok.</td><td>Size (px)</td><td>One shared</td><td>Predicted</td><td>Readout / context</td><td>Ratio</td><td>Image share (%)</td></tr><tr><td>OK-VQA</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>64</td><td>224</td><td>0.408</td><td>0.409</td><td>1.137</td><td>2.8</td><td>1.95</td></tr><tr><td>144</td><td>336</td><td>0.270</td><td>0.271</td><td>0.956</td><td>3.5</td><td>1.80</td></tr><tr><td>256</td><td>448</td><td>0.274</td><td>0.276</td><td>1.337</td><td>4.9</td><td>1.41</td></tr><tr><td>576</td><td>672</td><td>0.209</td><td>0.209</td><td>1.897</td><td>9.1</td><td>1.27</td></tr><tr><td>1,024</td><td>896</td><td>0.178</td><td>0.184</td><td>1.320</td><td>7.4</td><td>1.35</td></tr><tr><td>2,304</td><td>1344</td><td>0.112</td><td>0.113</td><td>1.702</td><td>15.2</td><td>0.68</td></tr><tr><td>4,096</td><td>1792</td><td>0.077</td><td>0.078</td><td>1.630</td><td>21.2</td><td>0.74</td></tr><tr><td>VizWiz</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>64</td><td>224</td><td>1.082</td><td>1.089</td><td>1.397</td><td>1.3</td><td>0.01</td></tr><tr><td>144</td><td>336</td><td>0.936</td><td>0.947</td><td>1.399</td><td>1.5</td><td>0.01</td></tr><tr><td>256</td><td>448</td><td>0.856</td><td>0.867</td><td>1.501</td><td>1.8</td><td>0.02</td></tr><tr><td>576</td><td>672</td><td>0.806</td><td>0.820</td><td>2.125</td><td>2.6</td><td>0.04</td></tr><tr><td>1,024</td><td>896</td><td>0.877</td><td>0.897</td><td>3.352</td><td>3.8</td><td>0.04</td></tr><tr><td>2,304</td><td>1344</td><td>0.550</td><td>0.564</td><td>3.100</td><td>5.6</td><td>0.04</td></tr><tr><td>4,096</td><td>1792</td><td>0.443</td><td>0.453</td><td>3.213</td><td>7.3</td><td>0.06</td></tr></table>

One shared: one vector on all eligible tokens. Predicted: the same vector with zero gradient at the image tokens. Readout / context: one vector on LAST and ANSWER-CUE and one on all other eligible tokens, with the same total norm. Ratio: readout / context over one shared. Image share: part of the one-shared value contributed by image tokens.

Table 16: Held-out loss decrease at update norm 0.01 on LLaVA. Ratios are relative to one shared vector.
<table><tr><td>Task</td><td>One shared vector (nats)</td><td>Readout / context</td><td>Random splits</td></tr><tr><td>VQAv2</td><td>0.113</td><td>7.52×</td><td>0.99-1.00×</td></tr><tr><td>OK-VQA</td><td>0.094</td><td>7.31×</td><td>1.01-1.03×</td></tr><tr><td>COCO</td><td>0.028</td><td>3.43×</td><td> $0 . 9 5 \substack { - 1 . 0 1 \times }$ </td></tr></table>

## H.6 Count normalization

Table 17 varies only the scale applied to each learned vector and keeps every other setting of the Table 1 entry of the model and dataset. Write $n _ { k }$ for the number of tokens of structural token group k in the prompt and $N _ { r }$ for the merged count of the groups that share vector r. “Vector count” divides a vector by $\sqrt { N _ { r } }$ and is the normalization used in all other experiments. “Group count” divides each token by the square root of its own group count, so a vector that ties several groups is not renormalized after merging. “Vector mean” divides by $N _ { r }$ . “None” applies no scale. Every configuration uses the two-vector assignment, paired supervision, five epochs, and the learning rate and demo count of the model and dataset, and reports its best epoch.

Table 17: Count normalization ablation on VQAv2 (accuracy, %) and COCO (CIDEr).
<table><tr><td colspan="2"></td><td colspan="2">VQAv2</td><td colspan="2">COCO</td></tr><tr><td>Normalization</td><td>Scale</td><td>Idefics2</td><td>LLaVA</td><td>Idefics2</td><td>LLaVA</td></tr><tr><td>Vector count</td><td> $1 / \sqrt { N _ { r } }$ </td><td>73.46</td><td>76.12</td><td>134.38</td><td>131.78</td></tr><tr><td>Group count</td><td> $1 / \sqrt { n _ { k } }$ </td><td>72.36</td><td>75.73</td><td>133.14</td><td>129.10</td></tr><tr><td>Vector mean</td><td> $1 / N _ { r }$ </td><td>72.10</td><td>75.87</td><td>133.80</td><td>130.09</td></tr><tr><td>None</td><td>1</td><td>67.61</td><td>74.96</td><td>132.98</td><td>130.23</td></tr></table>

What the comparisons support. Vector count is the best entry for every model and dataset. Group count is below vector count for every model and dataset, by 1.10 points on Idefics2 VQAv2 and by 2.68 CIDEr on LLaVA COCO, which supports computing the scale on merged groups rather than on the original groups. Removing the scale is the most damaging choice on VQAv2 for both models, by 5.85 points on Idefics2 and 1.16 points on LLaVA. The readout vector covers the final token and the answer cue, while the context vector covers all image tokens, 64 on Idefics2 and

729 on LLaVA, so an unscaled update injects far more energy through the context vector than through the readout vector at the same per-coordinate step.

## H.7 Injection depth

The depth ablation replaces the embedding injection by an update after a decoder block while retaining the readout vector, context vector, group assignments, and prompt-only mask. The Qwen-VL depths use the VizWiz and OK-VQA entries in Table 2. Both backbones have 32 decoder blocks. A displayed depth k/8 $k / 8$ means injection after block 4k for $k = 1 , \dots , 8$ . Depth zero denotes the input embeddings.

Raw scores and retained gain. Figure 17 contains all four raw curves. VQA scores are accuracies in percentages. Captioning uses CIDEr (Vedantam et al., 2015), on the scale reported in Table 1. The retained gain in Figure 4(b) is computed as

$$
{ \mathrm { R e t a i n e d G a i n } } ( \ell ) = { \frac { s _ { \ell } - s _ { \mathrm { z e r o } } } { s _ { \mathrm { e m b } } - s _ { \mathrm { z e r o } } } } \times 1 0 0 \%\tag{H.1}
$$

Here $s _ { \ell }$ is the score at injection depth $\ell , s _ { \mathrm { z e r o } }$ is the unadapted zero-shot score, and $s _ { \mathrm { e m b } }$ is the embedding endpoint. A negative retained gain means the depth-specific score is below zero-shot.

Embedding injection is best or tied for best in every curve. Later Idefics2 scores degrade sharply, while Qwen-VL retains more of its embedding gain on VizWiz than on OK-VQA. At depth one, after the last block, Proposition 2.1 explains why the context vector receives zero gradient and why later teacher-forced stopping decisions are not directly controllable.

![](images/89cefba2066ceecc1abd9702c50d30fb53794ebe733ff7095df1160c48abd251.jpg)

![](images/155ae4f7c1aee83113a564b94c7d46f7792f2951acdda11226754a3880e96040.jpg)

![](images/cf76e3e716772c31fe578b40df11354fe639d394ccd6edb72bec21c2bc921927.jpg)

![](images/710027d7e89ce10a20a41460bf18acd9670afd28aa679552a63d7fd4d6126920.jpg)  
Figure 17: Scores at each injection depth on Idefics2 (top) and Qwen-VL (bottom).

## I Analysis of the learned vectors

## I.1 Promoted tokens

The learned vectors are added to input embeddings and act through the frozen decoder, so Table 18 describes them by their effect on the next-token distribution. For a category with demo-free test prompts $p _ { 1 } , \ldots , p _ { n }$ , the promotion of a vocabulary token u is

$$
\Delta ( u ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \Big [ p _ { \theta } \big ( u \mid \widetilde { E } ( p _ { i } , W ) \big ) - p _ { \theta } \big ( u \mid E ( p _ { i } ) \big ) \Big ] ,\tag{I.1}
$$

where both distributions are taken at the final prompt token. Spellings with and without a leading space are merged, and whitespace tokens are omitted. Each cell lists the four tokens with the largest ∆(u) over 30 to 476 test prompts. A token is black when, lowercased and stripped of punctuation, it begins at least one reference answer of the category, and gray otherwise.

The checkpoints are the seed-41 runs of Tables 1 and 3 and the reported entries of Table 2. Prompts are rendered as in the corresponding evaluation. VQAv2 and VizWiz categories are assigned from the question wording, the VizWiz unanswerable column uses the answerability label of the dataset, and the OK-VQA columns use its annotated knowledge categories. No category depends on a model prediction.

The same two vectors are added to every prompt of a task, yet the promoted tokens change with the category and are the answers it asks for. Over the 68 categories of the eight VQA checkpoints, excluding the catch-all other category, 95.9% of the five most promoted tokens begin a reference answer. The table shows a subset of these categories. Adding demos without the learned vectors, as in the source prompt $p _ { i } ^ { \mathrm { s r c } }$ of paired training, gives 91.8% on the same categories, with four demos on VQAv2 and one on the other tasks. On the 90 TV benchmark checkpoints the two rates are 92.2% and 82.2%.

Several entries are the first subword of a longer answer, for example gir for giraffe, sam and son for Samsung and Sony, vit for vitamin, and un for unanswerable. LLaVA answers without a leading space, and its tokenizer then splits more answers, as in base, ten, and soc for baseball, tennis, and soccer.

## I.2 States and decoded tokens

The state of a prompt is the output of the last decoder block at the final prompt token, computed without the update and with the STAVE update.

On LLaVA VQAv2 we leave out the catch-all other category, which keeps 1,585 validation questions. Without the update, the next token is <|im\_start|> for 99.1% of them. In a principal component analysis (PCA) of the states of both conditions, the first component explains 50% of the variance and has cosine 0.99 with the difference between the two condition means, so Figure 5(a) shows the second and third components. Along them the no-update states stay in one cluster, and the STAVE states leave it in a different direction for each answer type. Panel (b) refits the PCA on the STAVE states of the six object categories. The no-update states still encode the question type: in the full space, a nearest-centroid classifier recovers it with 99.2% accuracy under five-fold cross-validation.

Figure 18(a) decodes the state after each block with the final normalization layer and the output projection of the frozen model, for three questions of Figure 5(b). We call an answer found late and lost when the reference first token, with and without a leading space, is the top decoded token after some block in the last 40% of the decoder but not at the output. Without the update this holds for 10% to 58% of the test prompts of the seven LMM checkpoints in panel (e) and for 15% of the TV benchmark prompts. STAVE outputs that token for 73% to 95% of these prompts. For these prompts the answer is present inside the frozen decoder, and the learned vectors keep it through the last blocks.

Panels (b) to (d) project the states of all 18 TV benchmark tasks onto the two leading principal components of the STAVE states, and the no-update states onto the same axes. Several tasks share test prompts exactly, which we detect as prompts with identical no-update states. A shared prompt has one no-update state and one next token, while STAVE moves it with the vectors of each task. List first and list last share all 200 test prompts. With the vectors of each task, STAVE answers both tasks correctly on all of them on every LLM, against at most 2.5% for the frozen model. Present to gerund and present to past share 9 to 29 prompts per LLM, with 100% against 0%. Panel (f) counts every pair of tasks with a shared test prompt, 265 to 277 per LLM. STAVE answers both tasks correctly for 92% to 94% of them and the frozen model for at most 2.6%. On GPT-J 6B and Pythia 12B the cards show next letter and to uppercase in place of the translation pair.

Figure 19 applies the same decoding to the first answer token on the Idefics2 and LLaVA VQAv2 and COCO checkpoints of Table 1, with 128 held-out demo-free prompts per model and dataset and a prompt with four demos and no update as a reference. Without the update, on three of the four model and dataset pairs the correct token is the top decoded token before the last block for 33% to 48% of the prompts and is lost at the output. With both vectors it stays top-1 through the output for 64.8% to 70.3% of the prompts, against 55.5% to 64.1% with four demos. In panel (b), the change that the vectors cause in the state stays between 0.7 and 1.3 times the norm of the state from block 16 to the output, so the frozen blocks do not suppress the update. In panel (c), the attention from the final token to the query image, averaged over layers, rises with the vectors for every model and dataset.

Table 18: Tokens most promoted by STAVE at the final prompt token. Gray tokens begin no reference answer of the column.  
(a) VQAv2, Table 1 checkpoints
<table><tr><td>Model</td><td>Color</td><td>Sport</td><td>Animal</td></tr><tr><td>Idefics2-8B-base LLaVA</td><td>blue, white, black, brown white, black, blue, red</td><td>tennis, baseball, soccer, sk base, ten, soc, sk</td><td>cat, horse, gir, z cat, g, dog, horse</td></tr><tr><td>Model</td><td>Food</td><td>Vehicle</td><td>Time or weather</td></tr><tr><td>Idefics2-8B-base LLaVA</td><td>pizza, hot, sandwich, cake pizza, hot, grass, sand</td><td>bus, motor, truck, train bus, bike, motor, pickup</td><td>afternoon, winter, morning, sun winter, day, summer, s</td></tr></table>

(b) OK-VQA, Table 1 and Table 2 checkpoints
<table><tr><td>Model</td><td>Vehicles and transport</td><td>Brands and products</td><td>Objects and material</td></tr><tr><td>Idefics2-8B-base</td><td>stop, double, fighter, dies</td><td>sam, son, apple, laptop</td><td>t, clock, wood, plastic</td></tr><tr><td>Idefics2-8B-instruct Qwen-VL-7B</td><td>stop, double, vol, c</td><td>apple, son, sam, computer</td><td>t, bear, wood, metal</td></tr><tr><td></td><td>bo, schw, east, stop</td><td>apple, c, schw, monitor</td><td>sit, cut, suit, leather</td></tr><tr><td>Model</td><td>Sports and recreation</td><td>Cooking and food</td><td>Weather and climate</td></tr><tr><td>Idefics2-8B-base</td><td>ski, surf, tennis, sk</td><td>b, vit, c, cake</td><td>sun, rain, winter, cold</td></tr><tr><td>Idefics2-8B-instruct</td><td>ski, tennis, sk, surf</td><td>b, c, vit, f</td><td>rain, sun, cloud, cold</td></tr><tr><td>Qwen-VL-7B</td><td>surf, fall, ser, run</td><td>carrot, lunch, spinach, k</td><td>dusk, str, rain, star</td></tr></table>

(c) VizWiz, Table 2 checkpoints
<table><tr><td>Model</td><td>Unanswerable</td><td>Identify the object</td><td></td><td>Color</td></tr><tr><td>Idefics2-8B-instruct</td><td>un, no, yes, windows</td><td></td><td>un, laptop, computer, tv</td><td>white, grey, black, blue</td></tr><tr><td>Qwen-VL-7B</td><td>un, priority, pink, unknown</td><td></td><td>un, computer, sweet, mac</td><td>un, white, black, blue</td></tr></table>

(d) TV benchmark, Table 3 checkpoints
<table><tr><td>Model</td><td>Location to continent</td><td>Location to religion</td><td colspan="3">Person to language</td></tr><tr><td>Pythia 6.9B</td><td>An, Europe, Asia, Af</td><td>Muslim, Christian, Jewish, Chinese</td><td></td><td>English, French, Italian, Spanish</td><td></td></tr><tr><td>Pythia 12B</td><td>An, Europe, Asia, Af</td><td>Muslim, Christian, Jewish, Jew</td><td></td><td>English, French, Italian, Spanish</td><td></td></tr><tr><td>GPT-J 6B</td><td>Ant, Europe, Asia, Af</td><td>Muslim, Christian, Jewish, Chinese</td><td></td><td>English, French, Spanish, Italian</td><td></td></tr><tr><td>Model</td><td>Country to capital</td><td>Present to gerund</td><td></td><td>English to French</td><td></td></tr><tr><td>Pythia 6.9B</td><td>Paris, Val, Pr, Ber</td><td>staying, blessing, begging, providing</td><td></td><td>par, comme, m, toujours</td><td></td></tr><tr><td>Pythia 12B</td><td>Paris, Ber, Pr, Val</td><td>rushing, staying, handing, guiding</td><td></td><td>comme, par, image, toujours</td><td></td></tr><tr><td>GPT-J6B</td><td>Paris, Ber, Pr, Val</td><td>singing, making, banning, baking</td><td></td><td>dire, par, personnel, pour</td><td></td></tr></table>

## I.3 What each learned vector changes

Figures 6 and 20 apply the deployed vectors of the Idefics2 entries of Table 1 to held-out demo-free prompts with no update, with one of the two vectors, and with both vectors. No vector is retrained, so the four conditions differ only in which vectors are added. Figure 6(a) and Figure 20(a) read the first answer token on 1,000 prompts per task, for VQAv2 and OK-VQA. A token counts as correct when it matches the answer of one annotator, the reference used for training, so these rates are lower than the VQA scores of Table 1, which give full credit to any answer given by at least three annotators. VQAv2 number questions are left out because every number answer begins with the same token on this backbone. The left group of bars asks whether the correct token is the most probable among the first tokens of the task’s answers, and the right group whether it is the generated token.

On VQAv2 and OK-VQA, the context vector alone makes the correct answer the most probable task answer for 10 and 14 more percentage points of the prompts, yet it is generated for only 0.3% and 10.5% of them, because without the readout vector the most probable next token is usually not an answer. The readout vector alone makes the model

every task pair with a shared test prompt, 265 to 277 per LLM

green: the reference first token. 晩 is Chinese for evening.

(a) LLaVA on VQAv2: top decoded token after each block
<table><tr><td rowspan=1 colspan=2>block</td><td rowspan=1 colspan=1>22</td><td rowspan=1 colspan=1>23</td><td rowspan=1 colspan=1>24</td><td rowspan=1 colspan=1>25</td><td rowspan=1 colspan=1>26</td><td rowspan=1 colspan=1>27</td><td rowspan=1 colspan=1>28</td><td rowspan=1 colspan=1>29</td><td rowspan=1 colspan=1>30</td><td rowspan=1 colspan=2>31     output</td></tr><tr><td rowspan=3 colspan=2></td><td rowspan=1 colspan=1>E</td><td rowspan=1 colspan=1>E</td><td rowspan=1 colspan=1>E</td><td rowspan=1 colspan=1>E</td><td rowspan=1 colspan=1>E</td><td rowspan=1 colspan=1>E</td><td rowspan=1 colspan=1>E</td><td rowspan=1 colspan=1>E</td><td rowspan=1 colspan=1>E</td><td rowspan=1 colspan=1>football</td><td rowspan=1 colspan=1>&lt;|im_start|&gt;</td></tr><tr><td rowspan=1 colspan=1>football</td><td rowspan=1 colspan=1>football</td><td rowspan=1 colspan=1>football</td><td rowspan=1 colspan=1>football</td><td rowspan=1 colspan=1>football</td><td rowspan=1 colspan=1>football</td><td rowspan=1 colspan=1>football</td><td rowspan=1 colspan=1>football</td><td rowspan=1 colspan=1>football</td><td rowspan=1 colspan=1>football</td><td rowspan=1 colspan=1>football</td></tr><tr><td rowspan=2 colspan=1></td><td rowspan=2 colspan=1></td><td rowspan=1 colspan=1>answer</td><td rowspan=1 colspan=1>answer</td><td rowspan=1 colspan=1>answer</td><td rowspan=1 colspan=1>E</td><td rowspan=1 colspan=1>E</td><td rowspan=1 colspan=1>E</td><td rowspan=1 colspan=1>E</td><td rowspan=1 colspan=1>E</td><td rowspan=1 colspan=1>living</td><td rowspan=1 colspan=1>living</td><td rowspan=1 colspan=1>&lt;|im_start|&gt;</td></tr><tr><td rowspan=1 colspan=1>living</td><td rowspan=1 colspan=1>living</td><td rowspan=1 colspan=1>living</td><td rowspan=1 colspan=1>living</td><td rowspan=1 colspan=1>living</td><td rowspan=1 colspan=1>living</td><td rowspan=1 colspan=1>living</td><td rowspan=1 colspan=1>living</td><td rowspan=1 colspan=1>living</td><td rowspan=1 colspan=1>living</td><td rowspan=1 colspan=1>living</td></tr><tr><td rowspan=2 colspan=2></td><td rowspan=1 colspan=1>it</td><td rowspan=1 colspan=1>it</td><td rowspan=1 colspan=1>it</td><td rowspan=1 colspan=1>night</td><td rowspan=1 colspan=1>晚</td><td rowspan=1 colspan=1>E</td><td rowspan=1 colspan=1>E</td><td rowspan=1 colspan=1>E</td><td rowspan=1 colspan=1>E</td><td rowspan=1 colspan=1>at</td><td rowspan=1 colspan=1>&lt;|im_start|&gt;</td></tr><tr><td rowspan=1 colspan=1>night</td><td rowspan=1 colspan=1>night</td><td rowspan=1 colspan=1>night</td><td rowspan=1 colspan=1>night</td><td rowspan=1 colspan=1>night</td><td rowspan=1 colspan=1>night</td><td rowspan=1 colspan=1>night</td><td rowspan=1 colspan=1>night</td><td rowspan=1 colspan=1>night</td><td rowspan=1 colspan=1>night</td><td rowspan=1 colspan=1>night</td></tr></table>

(b) LLaMA 7B, 18 text tasks  
![](images/67e4a5fd42bf0bb7640995f3c8a77e34651c7d7c18de7af0bb45637864c61c5c.jpg)

<table><tr><td>Algorithmic: one prompt, two tasks</td><td></td></tr><tr><td>example:u,x,e,g-&gt; No update</td><td>0.13</td></tr><tr><td>• List first</td><td></td></tr><tr><td>List last</td><td>1.00</td></tr><tr><td>→ g 200 char s both right</td><td>1.00 ST4V5 100%/ ne undete 2%</td></tr></table>

![](images/f986c1497017c416211461afbb11d0bf4e05df823bccba19dec06f1bc58779d7.jpg)  
200 shared prompts, both right: STAVE 100%, no update 2%

First Principal Component  
![](images/b1d30c8316b8f565bfdde877c59941c5483087c1df824073af8fc0d75420815e.jpg)

<table><tr><td colspan="2">Translation: one prompt, two tasks</td></tr><tr><td>example:music-&gt;</td><td></td></tr><tr><td> No update</td><td>→music</td></tr><tr><td>● Englísh to French</td><td>0.03 → musique 1.00</td></tr><tr><td>English to Spanish</td><td>1.00</td></tr><tr><td colspan="2">→música 33 shared prompts, both right: STAVE 76%, no update 0%</td></tr></table>

![](images/9b8b130dbb74394359cf7d4555aac0e3bc94ba05b9af3176034356a4f815321b.jpg)

<table><tr><td>Linguistic: one prompt, two tasks</td></tr><tr><td>example:do-&gt;</td></tr><tr><td>No update →do 0.05</td></tr><tr><td>● Present to gerund →doing 1.00</td></tr><tr><td>Present to past →did 1.00</td></tr><tr><td>9 shared prompts, both right: STAVE 100%, no update 0%</td></tr></table>

(c) GPT-J 6B, 18 text tasks  
9 shared prompts, both right: STAVE 100%, no update 0%

![](images/4943988200f31d51d2dedffc4929b63f02ea4a9d525612e0f1299021aa4ef01c.jpg)

<table><tr><td>Algorithmic: one prompt, two tasks</td></tr><tr><td>example:z,e,g,i-&gt;</td></tr><tr><td>No update →f 0.04</td></tr><tr><td>• List first →z 1.00</td></tr><tr><td>List last →i 1.00</td></tr><tr><td>200 shared prompts, both right: STAVE 100%, no update 0%</td></tr></table>

![](images/6d00fde4e7cd347e0b2efb09ad144147b53f4b10f97f05c69e575df7a6c672c1.jpg)

First Principal Component  
![](images/79812e9bc56b42d2f772e50d4818efc34f727f3d9c46e8f75616d5b5d62358d8.jpg)

<table><tr><td colspan="2">Algorithmic: one prompt, two tasks</td></tr><tr><td>example:o-&gt;</td><td>0.07</td></tr><tr><td>No update</td><td>→0 →p</td></tr><tr><td>Next letter</td><td>0.99</td></tr><tr><td>To uppercase →0</td><td>1.00</td></tr><tr><td colspan="2">12 shared prompts, both right: STAVE 100%, no update 0%</td></tr></table>

![](images/8c22c8caefd77919f34f8a00f303663403bcd786dc2f65ba25787f6338b5d0d0.jpg)

<table><tr><td>Linguistic: one prompt, two tasks</td></tr><tr><td>example:take-&gt;</td></tr><tr><td>No update →take 0.05</td></tr><tr><td>● Presènt to gerund →taking 1.00</td></tr><tr><td>Present to past → took 1.00</td></tr><tr><td>28 shared prompts, both right: STAVE 100%, no update 0%</td></tr></table>

(d) Pythia 12B, 18 text tasks  
![](images/ec180211a8df7770e8f1c5580bf4f293cf7b86261f308629f4a46aa2856004c6.jpg)

![](images/08fd01b08e044e2e394facd4b68992513797fac0358cb4ded5ab6ee10509a852.jpg)

<table><tr><td>Algorithmic: one prompt, two tasks</td></tr><tr><td>example:z,e,g,i-&gt;</td></tr><tr><td>No update →Z 0.08</td></tr><tr><td>• List first →z 1.00</td></tr><tr><td>List last →i 1.00</td></tr><tr><td>200 shared prompts, both right: STAVE 100%, no update 2%</td></tr></table>

![](images/d62f27910aba34e8a663846b687340c904c9945bcf698443b967793c9f0559e3.jpg)

<table><tr><td>Algorithmic: one prompt, two tasks</td></tr><tr><td>example:0-&gt;</td></tr><tr><td>No update → 0</td></tr><tr><td>0.09 Next letter →p 0.99</td></tr><tr><td>To uppercase →0 1.00</td></tr><tr><td>12 shared prompts, both right: STAVE 100%, no update 0%</td></tr></table>

![](images/e2600e3a6f31175ffb7aed688d236e326c1a523c4689bbe3e092d14649c38a6a.jpg)  
(e) Answers found in a late block, lost at the output  
found: the answer is the top token after a block in the last 40% of the decode

![](images/44c2f0be4213462b33fd8c959d0de58b08c7058bd08d1c78967326f3223a2535.jpg)

(f) Prompts shared by two tasks: both answers right  
![](images/7e17655b1e3bc1bec211f4f736629eea61d9b2064d27e12f1d09fe28a3032134.jpg)  
Figure 18: Final-token analysis beyond Figure 5. (a) Top decoded token after each block. (b) to (d) TV benchmark states. (e) Answers found late and lost. (f) Shared prompts answered correctly for both tasks.

(b)

No update 4 demos, no update Context vector only Readout vector only Both vectors (STAVE)

(c)  
(a)  
![](images/9c06a69fe28843185a3621b30d502e98a458cf2369a6792329aa3b304166ab7c.jpg)

![](images/c558022f3ecf3f8a128b7a6a01b5828e14557a5718ee021f3653a69ef3b2a297.jpg)

![](images/0313d095cb8e1e533123e3bd7d3e0c044a39bb0ecc133ec2dd86214770ae06d9.jpg)

![](images/9b9537fe01166f981d0fdbecc4893c24603d3756dfb7d857275bb27793d25f77.jpg)

![](images/7039cfe2f1bacd7e6e8acc3da5728f89d54a9221af8ef0a8800838f7f2af00b3.jpg)

![](images/39191d0a0d4979dd98b7d61b6ea5195df6a33cae898f38903dbd29dd5a06b0aa.jpg)

![](images/ab931c82d53dd2f4372ad537ae977181a09fdbc4fd6d1b23bf20b4a5473be4e9.jpg)

![](images/ff46f96d502ef05a249706a627fee809b00bf69484f2292efc10f13107cc2a29.jpg)

![](images/7176376d04602df89d03a5514c3581f2810cbf2e37892f2e370743416d307786.jpg)

![](images/4545d340fc4f4259093cdb144e69215dc3040afde470999c66b42679e4f3b6e6.jpg)

![](images/4d28d62dc0d94a97353c687bf09fcee27e6cda10ae2029deb25be46e1834df22.jpg)

![](images/762c46e3bb262258ff4221f08419aa95aa75d538b7fd08602dddbdf89fe695d2.jpg)  
Figure 19: Layerwise effect of the learned vectors at the final prompt token. (a) Correct first token decoded as top-1. (b) Relative change of the state. (c) Attention to the query image.

answer. Both vectors give the highest rate on both measures, so the improvement that the context vector brings to the choice among answers reaches the generated answer mostly when the readout vector is present. In Figure 6(b), for each prompt, let ∆ be a change of the state at the final prompt token. The figure reports the share $\| \bar { \Delta } \| _ { 2 } ^ { 2 } / \overline { { \| \Delta \| _ { 2 } ^ { 2 } } }$ that is common to all prompts, where the bars denote means over prompts.

Figure 20(b) to (d) show what the prompt-specific change of the context vector does. On VQAv2 yes/no question whose reference is no, the readout vector alone still answers yes for 45% of them, and adding the context vector lowers this rate to 26% while the rate on questions whose reference is yes stays near 80%. In panel (c), the readout vector is kept at every layer, while the tokens after the context see the change that the context vector makes to the context states only inside one block of four decoder layers. Layers 12 to 15 alone recover 94% of the yes/no accuracy gain and layers 16 to 19 alone 65%, while layers 20 to 31 recover almost none. In layers 12 to 15 the context vector also raises the attention from the final prompt token to the query image the most. Panel (d) replaces the query image by the image of another evaluation sample and keeps the question and the reference answer. On all three tasks the answer loss rises more when the context vector is present, by 0.23, 0.16, and 0.06 nats per token, and every paired 95% bootstrap interval excludes zero. The context vector therefore makes the answer depend more on the query image, and the readout vector makes this dependence visible in the generated answer.

## I.4 Answer-prefix probabilities

Figure 21 compares the LLaVA and Idefics2 entries of Table 1 and the Idefics2 and Qwen-VL entries of Table 2. In each of its four rows, STAVE has the largest share of examples above the diagonal and the highest mean answer-prefix probability of the four trained methods. Table 1 rows compare VQAv2 and OK-VQA against PT-Pre, using its 20-token checkpoint. Table 2 rows compare VizWiz and OK-VQA against zero-shot. Each row uses the corresponding table’s method checkpoints and 512 fixed seed-41 evaluation examples per dataset, paired across methods. The two OK-VQA protocols are kept separate and assigned different colors.

Each point compares the probability that a continuation begins with an annotated answer. We sum disjoint referenceprefix probabilities after removing duplicate token sequences and extensions of shorter reference prefixes. We include tokenizations with and without a leading space, without an end-of-sequence token or length normalization. Each method retains its reported inference prompt. The axes use a logit scale with probabilities displayed as percentages. Marginal histograms show sample counts, and dashed ellipses summarize the empirical 90% sample spread. Probabilities below 0.0001% are clipped for display only. Each row is read separately, since its reference method and protocol differ.

![](images/805afed94d8e3be027abdf4db25065ecc23fe78ba824915180bc1edd87023e59.jpg)  
Figure 20: More tests of each learned vector on held-out Idefics2 prompts.

## I.5 Qualitative examples

Figures 22 and 23 compare STAVE with LIVE, MimIC, and HiFICL on demo-free evaluation examples. The VQA examples come from VQAv2, OK-VQA, and VizWiz with the checkpoints of Tables 1 and 2, and each is a question that STAVE answers correctly and all three baselines answer incorrectly. The captions come from the Idefics2 and LLaVA COCO checkpoints of Table 1. STAVE answers questions that depend on text in the image, as for the bill and the toothpaste, and its captions name the object that the reference captions mention where the baselines name a more common one, for example a girl instead of a doll or a book instead of a laptop.

![](images/bcce8b62e9b9c87f0be1ffc05a79bc1094f5b8a073a1a0965e746b6eadd1a557.jpg)

![](images/34b49043f89cc5ac52b6a36e4fbde9468e40070a79edadb27224409cca40c924.jpg)

![](images/f73171fb9a51508a4d0f3b73abb33ef2a2525b06eec59d98f1de5a5f333f2288.jpg)

![](images/5b1051f5e2dbdd1d650229561cdcad5adab626e63829c1ab4cdb5ce128daf230.jpg)

![](images/689dad905f048abf2a8259d69d64e8f6a8c040c34354708fdc9a259a6e14aa59.jpg)

![](images/df9fe583d4362f4e2b1edeeeb45b0ba598fe8aadeb22d1bd91f0296624445ff1.jpg)

![](images/b05226c3e5b3745889ce6438aa1636987beb6340bcd30d1d869c99d80033273f.jpg)

![](images/470ca5c68c36bf5bff36bea2a41b5b7124bd6be8084891985e0a2d31ea4774da.jpg)

![](images/94dc0de1354d87875977309031ff3da20749ace6656854acbf47bd8250bb51f7.jpg)

![](images/495b11730baf126fcb50f100ed41f37761e71e4227e74fd6cfc3adf5ec445c9a.jpg)

![](images/c2fc95a19f8fe8e99fc6f0ed6b558462024b5ef108f60c1648ef533efd1ea5fe.jpg)

![](images/3f765f7bbe7131d91b867f9efd00d30300e21e333736df907fc848f7357a9ab0.jpg)

![](images/c37c760cb24d48b69ea35c8de9a29f319e6dc6aae43f0b3d98a292e08fd4f9b0.jpg)

![](images/4691d0da4a0ffa4a759cf6b7d410c4e5b45e7a3e12ef5f070e6af64d2eee432f.jpg)

![](images/71988173b1f829cbe7e855d37fd519c9bc2aa89217f6b0ab261898da3a7aa944.jpg)

![](images/955abf1947e1a72738ac544dab2e46706ffe75618bcc9bf8e8e51dbca1c75c1d.jpg)  
Figure 21: Answer-prefix probabilities against PT-Pre (top two rows) and zero-shot (bottom two rows).

<table><tr><td colspan="1" rowspan="4"><img src="images/8bb314f5b79c1a98068859e835b717a8f78e02c789a3c315dec16a90ecddbe50.jpg"/></td><td colspan="1" rowspan="4">What denomination is this bill?| STAVE 20                  VLIVE  2 dollar               xMimIC 2 dollar               xHiFICL 2 dollar               ×</td><td colspan="1" rowspan="4"><img src="images/6e03eb1cde2e25b3e17efc00bd2499de8a364f8ceee6984125a8c840692c9ec2.jpg"/></td><td colspan="1" rowspan="1">STAVE: A doll sitting on a table next to aclock.</td></tr><tr><td colspan="1" rowspan="1">LIVE: A young girl sitting in front of aclock.</td></tr><tr><td colspan="1" rowspan="1">MimIC: A little girl sitting in front of aclock.</td></tr><tr><td colspan="1" rowspan="1">HiFICL: A litte girl sitting in front of aclock.</td></tr><tr><td colspan="1" rowspan="1"><img src="images/cb392d5d40ea8b641de5a0922b687692738d238d7c59e76b391f042fddbe8137.jpg"/></td><td colspan="1" rowspan="1">What black veggie is on thispizza?| STAVE olives                VLIVE  onion.                xMimlC onion                ×HiFICL mushroom             x</td><td colspan="1" rowspan="1"><img src="images/eafd3cb45d7c3ae276a9b9f21fe124e36ca98c9dd3438c26a0a292096f64983a.jpg"/></td><td colspan="1" rowspan="1">STAVE: A person sitting on a bed with alaptop.LIVE: A woman is laying in bed readinga book.MimIC: A woman sitting on a bed readinga book.HiFICL: A woman sitting on a bed readinga book.</td></tr><tr><td colspan="2" rowspan="7">What is a baby cow called?STAVE calf                 VLIVE  lamb                ×<img src="images/c385870b9743ebcf90ce1fa378cc3f899001a89201e2e631c519eb05cf4194f4.jpg"/>MimlC lamb                xHiFICL lamb                xHow many sides does the STOPsign have?STAVE 8                   VLIVE  4                   x<img src="images/629feda6e06a906e4f8f1c8f3c4be733f18cbfc44346e4dfcc1abb45816d2e47.jpg"/>MimlC 4                   xHiFICL 4                   xWhere is the sign pointing?| STAVE down                VLIVE  park                 x<img src="images/c34e715167e2a1d348320371ca4052f31474fb567fcb2868b87d8c1e1a0414db.jpg"/>MimlC park                 xHiFICL park                 x</td><td colspan="1" rowspan="2"><img src="images/71c446e89b74cbe8d0f41d3cfdc8b619b4489625534ca119e59edf530990c4f9.jpg"/></td><td colspan="1" rowspan="1">STAVE: A computer desk with a laptopand a monitor.</td></tr><tr><td colspan="1" rowspan="1">LIVE: A laptop computer sitting on adesk next to a keyboard.MimIC: A desktop computer sitting on topof a desk.HiFICL: A desktop computer sitting on topof a wooden desk.</td></tr><tr><td colspan="1" rowspan="3"><img src="images/526f61251685ddcf5dd79c7e469eeac033932143fda075c01a09f301c91e0d03.jpg"/></td><td colspan="1" rowspan="1">STAVE: A young boy wearing a cape andeating a cookie.</td></tr><tr><td colspan="1" rowspan="1">LIVE: A young boy dressed as Robinthe Boy Wonder.MimIC: A young boy wearing a cape andeating a banana.</td></tr><tr><td colspan="1" rowspan="1">HiFICL: A young boy dressed as a superhero eating a banana.</td></tr><tr><td colspan="1" rowspan="2"><img src="images/cba1a4c570364b5ab2f321d82b9513eb6e5dc92e7aae6d30903996fbd20b7f3b.jpg"/></td><td colspan="1" rowspan="1">STAVE: A boat docked next to a bunchof bicycles.</td></tr><tr><td colspan="1" rowspan="1">LIVE: Bicycles are parked on the side ofa canal.MimIC: Bicycles are parked along theside of a canal.HiFICL: Bicycles are parked along theside of a canal.</td></tr><tr><td colspan="1" rowspan="3"><img src="images/47d807f63f29943739f5d041f5943242a7c9e3d3a97c968c31154eebfbb3049b.jpg"/></td><td colspan="1" rowspan="3">Is this some kind of bone? Or ahorn, by chance? STAVE horn                VLIVE  bone                xMimlC bone                ×HiFICL bone</td><td colspan="1" rowspan="3"><img src="images/8a49b8c7efa901cf3bf13ab4e32c070f6a7a7bb7c3b8017fc1f0340856b55c1d.jpg"/></td><td colspan="1" rowspan="1">STAVE: A slice of pie on a plate with afork.</td></tr><tr><td colspan="1" rowspan="1">LIVE:A slice of cake on a plate.MimIC: A piece of cake sitting on a platewith a fork.</td></tr><tr><td colspan="1" rowspan="1">HiFICL: A slice of cake sitting on top of aplate.</td></tr><tr><td colspan="1" rowspan="1"><img src="images/70c11f0fdcb47623cd224cfafe09c326e890a2ed9188249a8e580d4ffa4b8e66.jpg"/></td><td colspan="1" rowspan="1">What long strap can be tied tothe object around the animal'sneck?STAVE leash                VLIVE  collar.                xMimlC collar                xHiFICL collar                x</td><td colspan="1" rowspan="1"><img src="images/1b09a7cf9d5641ec969ccc086f6d0336414bd3b97fa6c4c56224aaf01d6dcb47.jpg"/></td><td colspan="1" rowspan="1">STAVE: A little girl standing in front of aChristmas stocking.LIVE: A young girl holding a chocolatecovered donut.MimIC: A little girl holding a remotecontrol in her hand.HiFICL: A little girl holding a remotecontrol in her hand.</td></tr><tr><td colspan="1" rowspan="2"><img src="images/c90eaf88d04067180de6ac867c935a2889acbb2ff9bb6203e2dc46b164787871.jpg"/></td><td colspan="1" rowspan="2">What is this type of mack truckused for?STAVE cement              VLIVE  fuel                 ×MimlC fuel                 xHiFICL fuel                  x</td><td colspan="1" rowspan="2"><img src="images/d6b9644e8528263361498aa0a970f6ceb532116e8ccfe555850b7023be80f2b7.jpg"/><img src="images/d5abfce00d680855dcad0f8c84fa72958baed65dc231898fc6c187790c76032e.jpg"/></td><td colspan="1" rowspan="1">STAVE: a chair with a backpack on it anda table with food on it</td></tr><tr><td colspan="1" rowspan="1">LIVE: A pile of food sitting on top of achair.MimIC: A pile of food sitting on top of achair.HiFICL: A pile of food sitting on top of achair.</td></tr><tr><td colspan="1" rowspan="2"><img src="images/be113c6c1aa7cc984d40521f128bf35615e18a563624b5f75f174617bc28267c.jpg"/></td><td colspan="2" rowspan="2">What is the name of thetoothpaste?STAVE crest                VLIVE  Colgate              ×               <img src="images/7a3adda346f9a1fb496a5a492ca8cca0295c1abd95fb8e99b7f4e2cf8ddf5a1a.jpg"/>MimlC colgate               xHiFICL colgate               x</td><td colspan="1" rowspan="1">STAVE: A clock hanging from theceiling in a room.</td></tr><tr><td colspan="1" rowspan="1">LIVE: A clock mounted on a wall in aroom.MimIC: A large clock mounted to theside of a building.HiFICL: A clock mounted to the side ofa building.</td></tr><tr><td colspan="1" rowspan="2"><img src="images/bde06f6eac94d003658a33b49c5ccdacdb97d06c105514616ed9a3116a5cf5e9.jpg"/></td><td colspan="1" rowspan="2">What is the zebra doing?STAVE laying down            JLIVE  cleaning              xMimlC cleaning              xHiFICL cleaning              x</td><td colspan="1" rowspan="2"><img src="images/b91bcbf5f42e9b7726793dc376ac6222a1a5f670aea9242d8e269214011ce746.jpg"/></td><td colspan="1" rowspan="1">STAVE: A man sitting on a couch holdinga sandwich.</td></tr><tr><td colspan="1" rowspan="1">LIVE:A man sitting on a couch holdinga piece of bread.MimIC: A man sitting on a couch holdinga piece of bread.HiFICL: A man sitting on a couch holdinga loaf of bread.</td></tr></table>

Figure 22: More qualitative examples on VQA (left) and COCO (right), set 1.

Figure 23: More qualitative examples on VQA (left) and COCO (right), set 2.

## J Limitations and future work

STAVE runs at zero-shot inference cost, and its two vectors are trained once per task by backpropagation through the frozen decoder, so training memory and time grow with the backbone. STAVE learns each task from labeled training samples. With a tenth of them it loses a few points, less than MimIC and HiFICL on Idefics2 (Figure 7). Training and injection act on the input embeddings, so STAVE requires access to the model weights.

Our LMM experiments cover prompts with one query image and short answers or captions. A natural next step is to apply STAVE to interleaved multi-image and video prompts (Li et al., 2024b; Zhang et al., 2025) and to tasks that require temporal understanding (Ding & Wang, 2025). Another application is long-form reasoning, where steering vectors already change the behavior of thinking language models (Venhoff et al., 2025). More broadly, a task stored in two vectors adds no prompt tokens and only 2d parameters, so one deployed backbone can serve many downstream tasks, including on devices with limited memory.

## K Impact statement

This work aims to make task adaptation of frozen LLMs and LMMs cheaper. STAVE removes demos from the prompt at inference, which lowers the compute, memory, and energy spent on each query, and it stores a task in two vectors, so many tasks can share one deployed backbone. Because the backbone stays frozen, STAVE inherits the biases and failure modes of the pretrained model, including hallucinated content in generated captions, and the same low cost also applies to tasks trained on harmful data. Deployments should therefore keep the safeguards and usage policies of the underlying model.