# On Trajectory-Aware Training for Masked Diffusion Language Models

Manuel Madeira<sup>†</sup>, Amitis Shidani, Alice Bizeul, Victor Turrisi, Louis Béthune, Bhavika Devnani<sup>‡</sup>, Dan Busbridge, Pierre Ablin, João Monteiro

Apple, <sup>†</sup>EPFL, <sup>‡</sup>Georgia Institute of Technology

Masked diffusion models (MDMs) generate text by unmasking several tokens per step, but they are trained and sampled under different conditions. The model is trained on randomly masked sequences, whereas inference follows a trajectory shaped by the model’s own predictions. Additionally, each step has no access to what the previous one computed. Recent methods narrow these limitations from separate angles, leaving open how these choices interact. We introduce PUMBA, a unified framework for trajectory-aware training that trains the denoiser on consecutive steps of policy-induced trajectories, passes information between steps, and optimizes them jointly by backpropagation through time. A controlled study of this design space shows that i) exact train–inference alignment fails due to local overfitting, whereas a looser alignment still brings training masks closer to those seen at inference; ii) passing continuous information outperforms discrete gradient estimators through the commitment at each step; and iii) performance improves as backpropagation through time spans more steps, which we support theoretically. Combined, these components match the best checkpoint of a same-size autoregressive model. Building on these findings, we scale PUMBA to supervised fine-tuning of LLaDA-8B, where it improves the trade-off between performance and number of function evaluations (NFEs) in both full-canvas and block diffusion generation. At matched performance, it needs up to 22% fewer NFEs than standard fine-tuning with twice the budget in full-canvas generation, and up to 26% fewer than standard fine-tuning for the same number of steps in block diffusion.

Correspondence: manuel.madeira@epfl.ch; bdevnani3@gatech.edu; {amitis\_shidani, abizeul, v\_turrisi, l\_bethune, dbusbridge, p\_ablin, jmonteiro2}@apple.com Contributions: Work done while MM and BD were interns at Apple. Date: September 30, 2026

## 1 Introduction

Masked difusion models (MDMs) (Shi et al., 2024; Sahoo et al., 2024) generate text by iteratively unmasking several tokens at a time, and have recently been scaled up (Nie et al., 2025; Ye et al., 2025; Inception Labs, 2025; Wu et al., 2026b; Nie et al., 2026; Bie et al., 2025). These models are trained to unmask partially masked sequences with randomly drawn masks. At inference, however, an unmasking policy uses the model’s predictions to decide which positions to reveal iteratively. This disagreement between training and inference opens two gaps. First, the model and policy jointly shape the inference trajectory, whose states difer from the training distribution (Figure 1). Second, since training sequences are drawn independently, the model never learns to use information from previous steps. Recent work addresses these gaps in diferent ways: by passing continuous information between steps (Jo et al., 2026; Hu et al., 2026); by training along trajectories induced by the inference policy, known as Progressive Unmasking (PU)<sup>1</sup> (Kim et al., 2026); and by training on several consecutive steps jointly (Xia et al., 2026; Rozonoyer et al., 2026).

We unify these approaches in a single formulation of trajectory-aware training, which we name PUMBA (Progressive UnMasking with Backpropagation Across steps), where the denoiser is trained on consecutive steps of a trajectory rather than isolated sequences (Section 3). It spans three axes: the policy that builds each training trajectory; the carry, i.e. the information one step passes to the next; and the window W of consecutive steps whose losses are optimized jointly by backpropagation through time (BPTT). Prior methods each address some of these axes.

![](images/5f83cbf6c2b33e90dba306519ba861fffbdb00dcd81d50f1139a8a613dc2b843.jpg)  
Figure 1 From MDM training to PUMBA. Each row shows the inputs to the denoiser $f _ { \theta }$ over consecutive steps; the bottom row corresponds to inference, where the policy g commits the model’s own samples, errors included. MDM trains on one independently masked $\mathbf { x } _ { t }$ . PU follows the trajectory of $g$ with ground-truth tokens, the carry passes the hidden state $h _ { j }$ to the next step, and BPTT backpropagates the loss through $h _ { j }$ within a window of W steps. Right: the axes each row uses.

To map this space, we first study a controlled setting where we train on TinyGSM (Liu et al., 2023) and evalu ate on GSM8K (Cobbe et al., 2021), analyzing how each axis contributes to reasoning performance (Section 3). First, we identify a local overfitting phenomenon that prevents PU from revealing as few tokens per step in training as at inference (Section 3.1). We show that even when training reveals several times more tokens per step, PU still reduces the train–inference mask discrepancy. Second, we demonstrate that a continuous carry is the most efective way to cross the discrete commitment at each unmasking step (Section 3.2). Third, performance improves as W grows, which we support theoretically. Crucially, the three axes compound: each adds generative performance at every decoding budget. Together they match the performance of an autoregressive (AR) model of the same size trained on the same data, while decoding more than one token per step.

Building on these findings, we scale PUMBA to the supervised fine-tuning (SFT) setting using LLaDA-8B-Base as the base model (Nie et al., 2025), in both full-canvas and block difusion formulations (Section 4). To the best of our knowledge, this is the first trajectory-aware SFT at this scale (full tuning of all 8B parameters, 4096-token sequences, on a 1.8B-token general-purpose instruction dataset). PUMBA suits $\mathrm { S F T }$ , which is data-constrained: further epochs soon yield little. Guided by the controlled study and the failure modes of PU at this scale, we design a PUMBA instantiation that pushes the performance versus number of function evaluations (NFEs) Pareto frontier beyond standard SFT on reasoning and instruction following. In full-canvas generation, 4000 steps of PUMBA improve performance three times as much as doubling the SFT budget, in less training time. At matched performance, PUMBA needs up to 22% fewer NFEs than an SFT run twice as long in full-canvas generation, and up to 26% fewer in block difusion than an MDM control trained as long.

## 2 Background

Notation. Let be a finite vocabulary containing a mask token $m \in \mathcal { V } .$ , and let $\mathbf { x } = ( x ^ { 1 } , \ldots , x ^ { L } ) \in \mathcal { V } ^ { L }$ be a sequence of length L with masked positions $\mathcal { M } ( \mathbf { x } ) = \{ i : x ^ { i } = m \}$ . We write $\Delta ^ { d }$ for the probability simplex in $\mathbb { R } ^ { d }$ and sg[ ] for stop-gradient.

Training of MDMs. The noising process of masked difusion interpolates between a clean sequence $\mathbf { x } _ { \mathrm { 0 } }$ at level t = 0 and the fully masked sequence at $t \ : = \ : 1$ . Under the linear schedule, it replaces each token of $\mathbf { x } _ { \mathrm { 0 } }$ by m independently with probability t (Sahoo et al., 2024; Shi et al., 2024). An MDM learns a denoiser $f _ { \theta } : \mathcal { V } ^ { L } \to \mathsf { \bar { ( } } \Delta ^ { | \mathcal { V } | } ) ^ { L }$ that returns a distribution $f _ { \theta } ^ { i } ( \cdot \mid \mathbf { x } _ { t } )$ over the clean token at each position i. The denoiser infers t from the masking ratio of ${ \bf x } _ { t } ,$ so it needs no time input (Gat et al., 2024; Sahoo et al., 2024; Amin et al., 2025). Training maximizes an evidence lower bound on the log-likelihood that reduces to a weighted cross-entropy over the masked positions:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { M D M } } = \mathbb { E } _ { \mathbf { x } _ { 0 } , t , \mathbf { x } _ { t } } \left[ \ell ( \mathbf { x } _ { 0 } , \mathbf { x } _ { t } ) \right] , \qquad \ell ( \mathbf { x } _ { 0 } , \mathbf { x } _ { t } ) = \frac { 1 } { t } \sum _ { i \in \mathcal { M } ( \mathbf { x } _ { t } ) } - \log f _ { \boldsymbol { \theta } } ^ { i } ( x _ { 0 } ^ { i } \mid \mathbf { x } _ { t } ) . } \end{array}\tag{2.1}
$$

The factor $1 / t$ follows from the bound and normalizes the sum over the $L \cdot t$ positions masked in expectation (Bethune et al., 2026). Each training step draws $\mathbf { x } _ { \mathrm { 0 } }$ and $t \sim \mathcal { U } [ 0 , 1 ]$ , masks $\mathbf { x } _ { \mathrm { 0 } }$ independently, and calls the denoiser once.

Inference in MDMs. Generation starts from the fully masked sequence and reveals tokens over multiple steps. We index steps by j and write $\mathbf { x } _ { t _ { j } }$ for the sequence after step j, whose level is its masking ratio $t _ { j } = | \mathcal { M } ( \mathbf { x } _ { t _ { j } } ) | / L$ Each step applies an unmasking policy g that selects $S = g ( \mathbf { x } _ { t _ { j } } ) \subseteq \mathcal { M } ( \mathbf { x } _ { t _ { j } } )$ and fills every $i \in S$ with a token drawn from $f _ { \theta } ^ { i } ( { \bf \cdot } \mid { \bf x } _ { t _ { j } } )$ . Revealing several positions at once trades exactness for speed by sampling from the product of their marginals rather than the joint distribution (Webb et al., 2026). Choosing the most confident positions keeps this product closer to the joint distribution and improves samples at a fixed number of steps (Nie et al., 2025; Wu et al., 2026b). The top-u policy (Chang et al., 2022) reveals the u masked positions with highest confidence $c _ { i } = \operatorname* { m a x } _ { v \in \mathcal { V } } f _ { \theta } ^ { i } ( v \mid \mathbf { x } _ { t _ { j } } )$ . We use this policy for the controlled experiments (Section 3). For larger MDMs, Fast-dLLM (Wu et al., 2026b,a) is an efective policy that reveals every position with $c _ { i } \geq \tau .$ or the single most confident one if none qualifies. Combined with block difusion decoding, which proceeds from left to right, block by block (Arriola et al., 2025), it preserves generation quality at far fewer denoiser calls, or number of function evaluations $( N F E s )$ , per sample. We therefore adopt it as the inference policy for LLaDA in Section 4. See Appendix A for an extended version.

## 3 Elucidating the Design Space of PUMBA

We map the design space of PUMBA in a controlled setting following Kim et al. (2026): we train a 125Mparameter bidirectional transformer from scratch on TinyGSM (Liu et al., 2023) for 20 epochs and evaluate zero-shot on GSM8K (Cobbe et al., 2021) with the top-u policy (Section 2). All variants share the same architecture, data, batch size, and learning rate.

## 3.1 Training–Inference Alignment via Progressive Unmasking

Standard MDM training masks sequences independently of the model, whereas inference visits the sequences that the denoiser and the unmasking policy dictate. Training and inference therefore encounter diferent distributions of masking patterns. The patterns at inference time depend on the reveal count u. In practice, the efective inference u is small, since otherwise the product of marginals becomes a poor approximation of the joint distribution of the unmasked tokens, and generation quality degrades (Webb et al., 2026). For example, on TinyGSM, the accuracy of every variant we train peaks at $u \in \{ 2 , 3 , 4 \}$ and roughly halves by u=16 (Figure 7, left). Similar trends hold for larger models (Kang et al., 2026). The regime of interest is therefore a rather small u. Ideally, training would visit the same sequences as inference does at this u.

Progressive unmasking (PU) (Kim et al., 2026) aims at exactly this: instead of the independent corruptions of Eq. (2.1), it trains on the trajectory that the inference policy itself produces. For each sample, training starts from the fully masked sequence. At each optimizer step, we compute the loss on the current sequence and then use the inference-time unmasking policy to reveal more positions. The resulting sequence is used in the next step, and this continues until the sample is fully revealed, after which a new sample is used. Many consecutive training examples thus come from the same trajectory rather than from multiple data samples with independent mask patterns. Each step still requires one denoiser call, and trajectories can be batched and progressively unmasked in parallel (Figure 2, left).

Formulation and training objective. We use the top-u policy (Section 2), which matches the TinyGSM decoding policy: at each step, we reveal the u most confident masked positions. We use the MDM loss of Eq. (2.1) unchanged, including the $1 / t$ factor. Here, t is the realized masking ratio, so the loss is normalized by the number of masked tokens. Kim et al. (2026) show that, although PU sequences are not draws from the original noising process, this construction preserves the unique minimizer of the MDM objective.

![](images/5a93659216528d08dde0cad1758425d900742f640770472c06bd1d302d09a3c7.jpg)

![](images/b9f2f4962d5e021ae58639ef33335e0f5e12c42cce5408a78001bb7744a93748.jpg)

![](images/c5e4fb2c12a16a36e6c38f1c20e6240f7ea19499e305d669ab23fe3c5a6daa3e.jpg)  
Figure 2 Local overfitting of PU at small training u on TinyGSM. Left: batch rows over steps (shade: share of masked tokens); MDM draws a fresh sequence and mask per row at every step, PU keeps each sequence for $J \approx L / u$ steps. Middle: completion loss of a sequence inserted into the batch, relative to the validation set (thick while in the batch, thin after). Right: training and validation completion losses; PU and MDM training losses use diferent masks and are not comparable.

Revealed positions are teacher-forced. For stability, a trajectory advances by committing the ground-truth token (hence teacher-forced) at each selected position rather than using the predictions of the denoiser. The mask pattern is therefore model-dependent, but the clean tokens are not. Under the policies we consider, a committed position is never revisited, so a sampled token (student forcing; Bengio et al., 2015) would introduce an error that could not be corrected later. The loss would still ask for the ground-truth $\mathbf { x } _ { 0 }$ at the remaining masked positions, so these targets could contradict the visible context. We investigate student forcing and its failure cases in PU in Appendix C.3.

Local overfitting at small u. At small u, the targeted inference regime, PU breaks two assumptions of stochastic optimization: each sequence is reused J times instead of once, and consecutive batches share all but a fraction $1 / J$ of their sequences instead of being independent. To see how the model responds, we launch short TinyGSM training runs at fixed u, where every trajectory lasts exactly J steps (Appendix D.1), and track the completion loss of held-out sequences as they enter and leave the batch, i.e., the loss on the second half of the answer given the prompt and the first half. We observe local overfitting (Figure 2, middle): at $u { = } 2 .$ , the completion loss of a sequence falls sharply within about 50 steps of entering the batch, and even rises again before the sequence leaves it: the model soon fits the sequence on the masks of its trajectory, after which the sequence provides no gradient for its revealed tokens, which are forgotten (Appendix D.3). The drop shrinks as u grows and, as expected, vanishes with MDM. As in regular overfitting, the validation loss moves opposite to the training loss (Figure 2, right); in these runs, PU trails MDM on validation completion loss for every $u \le 8 \ \mathrm { ( T a b l e \ 9 ) }$ , so local overfitting prevents training PU at the small u used at inference (see Appendix D for an in-depth study).

What about a larger u? Since local overfitting fades as u grows (Table 9), training at a larger u than we decode with may keep part of the alignment of PU without overfitting locally. The original PUMA recipe does this by committing a random number of positions per step, which is annealed over training, and additionally committing every position whose confidence exceeds 0.9 (Appendix B.2). Its efective training u, the number of generated positions divided by the number of forward passes per sequence, is about 133 at the start of training and 67 at the end, against u=2 at inference. We use this recipe for the runs in Figures 3 and 7.

Alignment under a mismatched u. To test whether training at a larger u than inference still reduces the train– inference mismatch, we compare the masking patterns seen in training under the PUMA recipe with those met during generation.

Definition 1 (Mask discrepancy). For a sample $\mathbf { x } _ { \mathrm { 0 } }$ and a masking ratio t, let $P _ { \mathrm { t r a i n } }$ and $P _ { \mathrm { i n f } }$ be the distributions of masks $\mathbf { m } \in \{ 0 , 1 \bar  \} ^ { L }$ at ratio t produced by the training procedure and by the sampler, respectively.

The mask discrepancy is

$$
\begin{array} { r l } & { D _ { \mathrm { m a s k } } ( t ) = \mathbb { E } _ { \mathbf { x } _ { 0 } } \big [ \mathrm { M M D } _ { k } ^ { 2 } ( P _ { \mathrm { t r a i n } } , P _ { \mathrm { i n f } } ) \big ] , } \\ & { \quad k ( \mathbf { m } , \mathbf { m } ^ { \prime } ) = \exp \bigl ( - d _ { H } ( \mathbf { m } , \mathbf { m } ^ { \prime } ) / \sigma \bigr ) , } \end{array}
$$

where MMD is maximum mean discrepancy (Gretton et al., 2012) and $d _ { H }$ the Hamming distance.

The kernel k is strictly positive definite on $\{ 0 , 1 \} ^ { L }$ , so $D _ { \mathrm { m a s k } } ( t ) = 0$ exactly when training and inference produce the same masks at ratio t. Larger values mean that inference visits mask patterns rarely seen in training. $D _ { \mathrm { m a s k } }$ depends only on which positions are masked, not on the value of the token or the carry, so it isolates the masking distribution shift. Appendix C.4 gives the estimator we use in practice. Thus, although local overfitting forces a much larger training $u ,$ the PUMA recipe still reduces $D _ { \mathrm { m a s k } }$ at $u { = } 2$ well below MDM (Figure 3). The information propagation techniques of Section 3.2 do not undo the alignment of PU, inheriting this benefit.

![](images/3855195b06db3f827960bb89f1a0e64751a12adbd3dce4c5b4ef7a6a24ee4daa.jpg)  
Figure 3 Mask discrepancy at $u { = } 2$ across $t ;$ lower is better.

## 3.2 Propagating Information Across Denoising Steps

In PU, each step influences the input seen by the next step, but the objective trains each step only to make its own predictions, while the next step receives only the committed tokens. We argue that a denoising step should not only predict its own tokens, but also produce an output that enables subsequent steps to operate efectively, especially since committed positions are never revisited. Training for this requires a channel that passes information from one step to the next, and a gradient that flows back through it. The channel can be a continuous carry that bypasses the discrete commitment (Jo et al., 2026; Hu et al., 2026), or a gradient estimator through the committed tokens themselves, which we call a discrete bridge (Jang et al., 2017; Maddison et al., 2017; Bengio et al., 2013; Williams, 1992). We first describe the carry and how we train it, then compare it with the discrete bridges.

Passing information forward. The carry can be the hidden state of the denoiser (Jo et al., 2026) or the probability-weighted input embedding of each prediction (Hu et al., 2026). We adopt the carry design from loopholing (Jo et al., 2026) for its simplicity and compactness, and do not ablate other designs. The denoiser thus also takes the previous hidden state as input and returns its own:

$$
\big ( f _ { \boldsymbol { \theta } } ( \cdot  { | } \mathbf { x } _ { t _ { j } } , h _ { j } ) , h _ { j + 1 } \big ) = f _ { \mathrm { c a r r y } } ( \mathbf x _ { t _ { j } } , h _ { j } ) .\tag{3.1}
$$

The carry $h _ { j } \in \mathbb { R } ^ { L \times d }$ is the last hidden state before the output projection, one vector of width $d \ll | \nu |$ per position (Appendix B). Each trajectory starts from a zero carry, and each inference step passes its carry to the next. The carry thus provides a weak form of student forcing: it is produced by the model at the previous step and reflects its imperfect predictions, while the visible tokens, and hence the targets, stay consistent with $\mathbf { x } _ { \mathrm { 0 } }$

Passing gradient back. When each update trains on a single step $\left( W { = } 1 \right)$ , the model learns to use the carry it receives but not to produce a useful one, since no term of the objective depends on the carry’s efect on the next step. We therefore compute the loss on $W$ consecutive steps of the same trajectory in one update and backpropagate through the carries between them. The loss at each step then reaches the preceding steps of the window, so the carry is also optimized for the predictions that follow. As a trajectory usually exceeds W steps, we split it into consecutive windows, each with its own update, and gradients do not cross between windows. The resulting procedure (Algorithm 1) is truncated backpropagation through time (BPTT).

Table 1 Peak GSM8K accuracy (%) for diferent BPTT techniques at $W { = } 2$ on TinyGSM.
<table><tr><td>PU + REINFORCE</td><td>41.2</td></tr><tr><td>PU + Gumbel-softmax</td><td>42.4</td></tr><tr><td>PU + Straight-through</td><td>44.7</td></tr><tr><td> $\mathrm { P U } + \mathrm { C a r r y }$ </td><td>51.2</td></tr><tr><td></td><td></td></tr></table>

Comparing the channels. To compare the carry with the discrete bridges, we fix the window at W=2 and replace the carry with a relaxed categorical (Gumbel-softmax) (Jang et al., 2017; Maddison et al., 2017), a straight-through estimator (Bengio et al., 2013), and a score-function (REINFORCE) estimator (Williams, 1992) with a leave-one-out baseline (Kool et al., 2019). To remove confounders and keep the training distribution stationary, all runs of this comparison keep the PUMA recipe at its initial setting instead of annealing it, under the same budget.

The carry outperforms all discrete bridges (Table 1; training curves in Appendix C.2), so we adopt it as the default strategy.

Implementation. The carry passes through a zero-initialized LayerNorm (Ba et al., 2016) and is added to the token embeddings, so training starts from the unmodified denoiser. MDM training has no predecessor to take a carry from, so loopholing obtains one by self-conditioning (Chen et al., 2023; Jabri et al., 2023), with an extra forward pass that adds about 30% training time (Jo et al., 2026). Under PU, the predecessor exists, so we simply cache its carry, one $L \times d$ tensor per trajectory. Each update averages the losses of its window, so compute and memory grow linearly in W while the update scale does not. Algorithm 1 in Appendix B.1 summarizes the resulting algorithm, PUMBA.

Why a longer window helps. The comparison above fixes $W { = } 2$ without exploring how large this window should be. The following result bounds how close the learned sampler gets to the best achievable one as a function of W (proofs in Appendix E).

Proposition 1 (Informal). Assume teacher-forced commits and a reveal rule that matches inference.

(i) No gradient flows through committed tokens. Without a carry (or $W { = } 1 )$ , a step’s loss cannot reward earlier steps; with BPTT, it reaches at most W 1 of the earlier steps of its window.

(ii) Assume the carry is contractive with factor $\gamma < 1$ . Let $\theta _ { W }$ be a point at which the truncated gradient vanishes and the loss satisfies a Polyak–Łojasiewicz condition (Karimi et al., 2016). Then

$$
\mathrm { K L } ( p ^ { \star } \| p _ { \theta _ { W } } ) - \operatorname* { i n f } _ { \theta } \mathrm { K L } ( p ^ { \star } \| p _ { \theta } ) \leq C \Big ( \frac { J } { W } - 1 \Big ) ^ { 2 } ,
$$

where J is the number of decoding steps and C does not depend on $W$

The first part of Proposition 1 explains why discrete gradient estimators do not match the carry (Table 1), while the second part shows that the gap to the best achievable sampler shrinks as the window grows, and vanishes when the window covers the whole trajectory. If the carry-augmented sampler can approximate $p ^ { \star }$ arbitrarily well, a large enough window then yields a sampler closer to $p ^ { \star }$ than any AR model with nonzero error (Corollary 1). Part (i) holds along any teacher-forced trajectory; part (ii) holds when training and inference share the reveal rule, and Figure 3 and Table 2 test empirically whether its prediction carries over to the smaller inference u. We discuss the scope of the bound in more detail in Appendix E.

## 3.3 Composing the Design Choices

We now combine the choices of Sections 3.1 and 3.2 and compare them with MDM and AR baselines. To increase alignment, we train a standard MDM, PU, PU with the carry $( W { = } 1 )$ , and PU with the carry trained by BPTT over $W \in \{ 2 , 4 , 8 \}$ steps. As controls, we train MDM for $2 \times$ and $8 \times$ as many epochs, which matches the number of denoiser forward and backward passes of BPTT at $W { = } 2$ and $W { = } 8$ , since each BPTT update runs W of them. We also train an AR model of the same size on the same data.

The three design choices compose: each one adds accuracy on top of the previous ones, at u=2 (Table 2) and at every other decoding budget we evaluate (Figure 7). At a matched number of denoiser passes, MDM trained 2 or 8 longer remains substantially below the corresponding BPTT at W=2 and W=8, respectively. Accuracy also keeps increasing with the window, as Proposition 1 predicts. Together, the three choices reach the best AR checkpoint while revealing more than one token per step. Accuracy during training and pass@k are in Appendix C.

Table 2 Best GSM8K accuracy (%) at $u { = } 2 ;$ MDM and PU: mean ± s.d. over three seeds.
<table><tr><td>AR</td><td>55.3</td></tr><tr><td>MDM</td><td>34.8 ± 2.5</td></tr><tr><td>MDM ×2 epochs</td><td>42.6</td></tr><tr><td>MDM ×8 epochs</td><td>43.6</td></tr><tr><td>PU  $\mathrm { P U } + \mathrm { C a r r y } ,$   $W { = } 1$ </td><td>40.2 ± 0.3</td></tr><tr><td> $W { = } 2$ </td><td>44.4</td></tr><tr><td> $W { = } 4$ </td><td>48.7</td></tr><tr><td> $W { = } 8$ </td><td>52.6 55.6</td></tr></table>

## 4 A PUMBA Recipe for Large-Scale SFT

Following Section 3, our SFT recipe for LLaDA-8B-Base (Nie et al., 2025) trains on inference-policy trajectories at a larger u than at inference, avoiding local overfitting while keeping training masks aligned, and trains the carry by BPTT over a window W.

Progressive unmasking with variable lengths. TinyGSM samples have similar lengths, so the PUMA recipe, which reveals a fixed length fraction per step, unmasks a similar number of tokens for all of them. SFT responses instead range from 14 tokens at the 10th percentile to 778 at the 90th (Appendix B.2). We fix the minimum number of tokens per step: every trajectory starts fully masked, and each step reveals the u most confident positions, plus every other position with confidence above τ=0.9.

Setup and evaluation. To separate the efect due to PUMBA from the model simply learning the data distribution, we use two stages: standard SFT of LLaDA-8B-Base, during which the model captures most of the target distribution, followed by a Post-SFT stage where we apply PUMBA with diferent parameters and the corresponding baselines. Since the original LLaDA SFT dataset is not publicly available, we use the SFT dataset from the OLMo 3 post-training pipeline (Olmo Team, 2025). We follow typical SFT practice: the prompt is fully revealed and only the response is maskable (Nie et al., 2025; Ye et al., 2025). Additional details are present in Appendix B.

We evaluate along two axes: the model’s learned ability to follow instructions, and potential regression on other capabilities. For instruction following, we use IFEval (Zhou et al., 2023), which measures the central capability targeted by our general SFT. To check that other capabilities are preserved, we additionally evaluate on GSM8K (Cobbe et al., 2021) (math) and MBPP (Austin et al., 2021b) (code). We refer to the mean of the three benchmarks as the score. We decode with Fast-dLLM (Wu et al., 2026b) in blocks of size 32 and report each model’s score–NFE curve over its threshold τ  0.7, 0.8, 0.9, 1 , from one token per step (τ=1) to more in parallel. We consider two settings: full-canvas and block difusion.

## 4.1 Full-Canvas Generation

In the full-canvas setting, the model attends to the full sequence bidirectionally. We SFT the model with the vanilla MDM objective for 1.5 epochs ( 25k training steps) at a global batch size of 128 sequences, over which the model sees 2.71 B tokens. From this checkpoint, we enter the Post-SFT stage and train for an additional 4000 steps. We compare PU and PU with the carry trained by BPTT over a window W, where W=1 passes the carry without gradient, and sweep the training u and W. As baselines, we continue the 1.5-epoch checkpoint with MDM for the same 4000 steps (Post-SFT MDM), which rules out gains from the additional updates alone, and double the SFT budget to 3 epochs, which shows that MDM training has nearly saturated on this data. We report results for u 32, 64 with remaining values present in Appendix F.

Neither baseline improves much: Post-SFT MDM stays close to its starting checkpoint, and the 3-epoch SFT, despite twice the training, improves only slightly and only at higher NFEs, suggesting that MDM training has nearly saturated on this data. In contrast, PUMBA improves the score while reducing NFEs, with a fraction of the SFT step budget. Its best configuration (u=32, W=8) gains 2.6 points, over three times the gain of doubling the SFT budget, and at matched score PUMBA needs up to 22% fewer NFEs than the 3-epoch SFT (Figure 4).

## 4.2 Block Diffusion

In the block difusion setting, the model generates the response block by block, attending block-causally to previous blocks and bidirectionally within the current one. This attention pattern departs from LLaDA’s fully bidirectional pre-training, and we find that it requires a longer SFT to reach competitive performance.

![](images/efdd40864f47c3857cbd79d5bbe289321e8397f54057b3a32b597f03f1eab4b5.jpg)  
Figure 4 Post-SFT of LLaDA-8B for full canvas (top row) and block difusion (bottom row). In each legend, we set in bold the SFT checkpoint the Post-SFT runs of that setting start from.

We therefore run SFT for 6 epochs, using the vectorized implementation of Arriola et al. (2025), which trains all blocks of a sequence in a single forward pass, with blocks of size 32 to match the Fast-dLLM inference policy (Wu et al., 2026b).

Post-SFT starts from this checkpoint and, as in the full-canvas setting, runs for 4000 steps for each composition of PU and BPTT. In the full-canvas setup, the reference for longer MDM training was a run with twice the SFT budget, which here would require 12 epochs. We instead use the 3-epoch checkpoint of the same run, so that the step from 3 to 6 epochs measures what doubling SFT yields. Since blocks have a fixed size of 32, u fixes the number of steps per block, which upper-bounds the window: e.g., u=16 implies $W \le 2$ We report results for $u \in \{ 8 , 1 6 \}$ in Figure 4; the remaining values are in Appendix F. Consistent with the full-canvas setting, PUMBA improves over the MDM baselines. PU alone and the carry (W=1) bring modest gains, but BPTT with wider windows yields a clear improvement. The best configuration in Figure 4 (u=16, W=2) gains 2.2 points in 4000 steps, more than the 50k SFT steps from 3 to 6 epochs, and at matched score it needs 26% fewer NFEs than the Post-SFT MDM control. The PUMBA variants also consistently reduce NFEs, a trend not observed with progressively longer SFT (see Table 12 for best scores per variant).

## 4.3 Failure Modes at Small Training u

Local overfitting at scale. Random masking acts as implicit data augmentation, which favors MDM over AR models when data is repeated (Prabhudesai et al., 2025). However, these masks are drawn independently of the unmasking policy, so most orderings seen in training never occur at inference. PU concentrates training on the orderings the inference policy produces, yielding a more targeted and data-eficient signal. At moderate u and matched compute, PU-based variants tend to outperform the MDM baselines. This advantage, however, holds only while u is large enough: the smaller $u ,$ the longer each trajectory, and the fewer distinct samples the model sees in a fixed number of steps. Figure 5a measures this cost directly: at full-canvas u=1 a run sees only 0.6% of samples visited by MDM, rising to 26% at u=128. Block difusion is less exposed, reaching 100% by $u { = } 3 2 .$ , since the parallel-block implementation stratifies unmasking: u=1 reveals one token per block rather than one in the entire sequence. As u decreases, fewer distinct samples are seen and each stays longer in the batch (Figure 25), until the model overfits locally and training collapses (Figure 5b). At the other extreme, PU at u=128 only matches the Post-SFT MDM control: each step reveals a large part of the response, so the trajectory reduces to a few coarse steps. The best training u needs to be small enough to align with inference and large enough to allow the model to learn.

![](images/0e1881024792b65bd9bd4a1be8bf27d8132c648c71ae72a0a3b1e67d25dfef61.jpg)

![](images/43a7d0468164b711e0b842fc0aa25bb00b12263dac0fb8426c1a23f6ffc3130f.jpg)

![](images/c52c3813eec7c5b68e9cc9b3b6ea05495c23bcf8a0276aaaec4484456f7f9110.jpg)  
SFT, 1.5 epochs PU u=1 PU u=4 PU u=16 PU u=64 MDM PU u=2 PU u=8 PU u=32 PU u=128  
Figure 5 Trade-of between train–inference alignment and sample diversity. (a) Sample diversity for PU with varying u, measured as the number of distinct training samples visited, as a percentage of the Post-SFT MDM control’s. (b) Performance as a function of NFEs for the full-canvas setting, at full extent (left) and zoomed into the dashed box (right).

## 5 Related Work

Masked diffusion language models. Difusion language models operate on continuous representations (Li et al., 2022; Dieleman et al., 2022) or discrete tokens (Hoogeboom et al., 2021; Austin et al., 2021a; Campbell et al., 2022; Lou et al., 2024). Masked difusion simplifies the absorbing-state formulation (Shi et al., 2024; Sahoo et al., 2024) and scales to large language models through autoregressive conversion or training from scratch (Gong et al., 2025; Ye et al., 2025; Nie et al., 2025). Block Difusion is a particularly efective difusion formulation for language modeling, combining autoregressive generation across blocks with parallel denoising within them (Arriola et al., 2025). We study how to post-train these models for the states their decoders induce.

Structured inference. MDM inference policies choose both the reveal order and the degree of parallelism. Common criteria include confidence (Ghazvininejad et al., 2019; Chang et al., 2022; Wu et al., 2026b), probability margins (Kim et al., 2025), and predictive entropy (Ben-Hamu et al., 2025); others use explicit planners (Liu et al., 2025; Peng et al., 2025) or learned unmasking policies (Jazbec et al., 2026). These policies induce structured trajectories whose intermediate states difer from the random corruptions used in conventional training, motivating our focus on trajectory alignment.

Train–inference alignment. Training on expert states while generating from model predictions is a longstanding challenge (Ross et al., 2011; Bengio et al., 2015); in MDMs, the shift concerns both revealed positions and token content. PUMA trains on progressive teacher-forced unmasking trajectories (Kim et al., 2026), while PAPL reweights the loss for the decoding planner (Peng et al., 2026). Other approaches leverage task rewards along denoising trajectories (He et al., 2025; Wang et al., 2026) or supervise model-induced states through on-policy distillation (Ren et al., 2026; Su et al., 2026; Xu et al., 2026). Relatedly, SDTT distills many-step denoising into fewer sampling steps (Deschenaux & Gulcehre, 2025). We investigate alignment directly within supervised instruction fine-tuning, without task-specific reward optimization or distillation.

State propagation across denoising steps. Loopholing preserves latent information through self-conditioning (Jo et al., 2026). RCD and DifusionGemma instead feed back the probability-weighted input embedding of each prediction (Hu et al., 2026; DifusionGemma Team, 2026). MetaState adds persistent working memory to a frozen backbone (Xia et al., 2026). Closely related to PUMBA, Relay passes diferentiable per-token representations between steps and trains them with truncated BPTT (Rozonoyer et al., 2026), but uses a twostep unroll (W=2 in our notation) in all its experiments. Our study combines trajectory-aligned supervised post-training with continuous state propagation, examines each component’s contribution, and characterizes the trade-of between alignment and learnability. Appendix G details the comparison with the closest work.

## 6 Conclusion

We improve masked difusion training by aligning it with the trajectories encountered during generation. Building on progressive unmasking methods, we cache the denoiser’s hidden state across steps and use truncated BPTT to train each step for its contribution to subsequent predictions. We study the components of the resulting method, PUMBA, on TinyGSM and show we can scale PUMBA to general-purpose instructionfollowing SFT on LLaDA-8B, improving the performance–NFE frontier under both full-canvas and block difusion. In particular, the Post-SFT recipe we employ is practical: it pushes a model beyond what additional SFT allows, and the resulting reduction in denoising steps amortizes this extra training stage, which is short, since a small number of training steps sufices to realize the gains of PUMBA. Our findings highlight the need to balance trajectory alignment with the diversity required for efective learning.

Limitations and Future Work. We identify local overfitting at small reveal counts, caused by reusing each training sequence over many consecutive steps, and ways to mitigate it; turning these findings into training procedures that sustain closer inference alignment across model scales, and across decoding policies beyond the confidence-based ones we study, remains open. Extending the carry’s weak student forcing to sampled tokens did not improve on teacher forcing in preliminary experiments (Appendix C.3), since irreversible wrong commitments leave the context inconsistent with the target; remasking could address this. Finally, we use a minimal carry to isolate the benefits of trajectory alignment and BPTT; richer carries, such as the fixedsize working memory of MetaState (Xia et al., 2026), and BPTT-specific gradient checkpointing, selective gradient propagation, or adaptive windows could improve the trade-of between training cost, memory, and credit-assignment horizon.

## Acknowledgments

We thank Marco Cuturi, Oscar Davis, Eleonora Gualdoni, Michael Klein, Tatiana Likhomanenko, Barry Theobald, and Russ Webb for their helpful feedback and critical discussions throughout the process of writing this paper; Okan Akalin, Brian Gamp, Denise Hui, Li Li, Cindy Liu, Evan Samanas, Guillaume Seguin, and the wider Apple infrastructure team for assistance with developing scalable, fault-tolerant code. Names are in alphabetical order by last name within group.

## References

Alan Nawzad Amin, Nate Gruver, and Andrew Gordon Wilson. Why masking difusion works: Condition on the jump schedule for improved discrete difusion. In Proc. of NeurIPS, 2025.

Marianne Arriola, Aaron Gokaslan, Justin T. Chiu, Zhihan Yang, Zhixuan Qi, Jiaqi Han, Subham Sekhar Sahoo, and Volodymyr Kuleshov. Block difusion: Interpolating between autoregressive and difusion language models. In Proc. of ICLR, 2025.

Jacob Austin, Daniel D. Johnson, Jonathan Ho, Daniel Tarlow, and Rianne van den Berg. Structured denoising difusion models in discrete state-spaces. In Proc. of NeurIPS, 2021a.

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, and Charles Sutton. Program synthesis with large language models, 2021b. URL https://arxiv.org/abs/2108.07732.

Jimmy Lei Ba, Jamie Ryan Kiros, and Geofrey E. Hinton. Layer normalization, 2016. URL https://arxiv.org/abs/1607. 06450.

Heli Ben-Hamu, Itai Gat, Daniel Severo, Niklas Nolte, and Brian Karrer. Accelerated sampling from masked difusion models via entropy bounded unmasking. In Proc. of NeurIPS, 2025.

Samy Bengio, Oriol Vinyals, Navdeep Jaitly, and Noam Shazeer. Scheduled sampling for sequence prediction with recurrent neural networks. In Proc. of NeurIPS, 2015.

Yoshua Bengio, Nicholas Léonard, and Aaron Courville. Estimating or propagating gradients through stochastic neurons for conditional computation, 2013. URL https://arxiv.org/abs/1308.3432.

Louis Bethune, Victor Turrisi, Bruno Kacper Mlodozeniec, Pau Rodriguez Lopez, Lokesh Boominathan, Nikhil Bhendawade, Amitis Shidani, Joris Pelemans, Theo X. Olausson, Devon Hjelm, et al. The design space of tri-modal masked difusion models, 2026. URL https://arxiv.org/abs/2602.21472.

Tiwei Bie, Maosong Cao, Kun Chen, Lun Du, Mingliang Gong, Zhuochen Gong, Yanmei Gu, Jiaqi Hu, Zenan Huang, Zhenzhong Lan, et al. LLaDA2.0: Scaling up difusion language models to 100B, 2025. URL https://arxiv.org/abs/2512. 15745.

Andrew Campbell, Joe Benton, Valentin De Bortoli, Thomas Rainforth, George Deligiannidis, and Arnaud Doucet. A continuous time framework for discrete denoising models. In Proc. of NeurIPS, 2022.

Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T. Freeman. MaskGIT: Masked generative image transformer. In Proc. of CVPR, 2022.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code, 2021. URL https://arxiv.org/abs/2107.03374.

Ting Chen, Ruixiang Zhang, and Geofrey E. Hinton. Analog bits: Generating discrete data using difusion models with self-conditioning. In Proc. of ICLR, 2023.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems, 2021. URL https://arxiv.org/abs/2110.14168.

Justin Deschenaux and Caglar Gulcehre. Beyond autoregression: Fast LLMs via self-distillation through time. In Proc. of ICLR, 2025.

Sander Dieleman, Laurent Sartran, Arman Roshannai, Nikolay Savinov, Yaroslav Ganin, Pierre H. Richemond, Arnaud Doucet, Robin Strudel, Chris Dyer, Conor Durkan, Curtis Hawthorne, Rémi Leblond, Will Grathwohl, and Jonas Adler. Continuous difusion for categorical data, 2022. URL https://arxiv.org/abs/2211.15089.

DifusionGemma Team. DifusionGemma technical report, 2026. URL https://arxiv.org/abs/2608.00146.

Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jefrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighof, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. The language model evaluation harness, July 2024. URL https://zenodo.org/records/12608602.

Itai Gat, Tal Remez, Neta Shaul, Felix Kreuk, Ricky T. Q. Chen, Gabriel Synnaeve, Yossi Adi, and Yaron Lipman. Discrete flow matching. In Proc. of NeurIPS, 2024.

Marjan Ghazvininejad, Omer Levy, Yinhan Liu, and Luke Zettlemoyer. Mask-Predict: Parallel decoding of conditional masked language models. In Proc. of EMNLP, 2019.

Shansan Gong, Shivam Agarwal, Yizhe Zhang, Jiacheng Ye, Lin Zheng, Mukai Li, Chenxin An, Peilin Zhao, Wei Bi, Jiawei Han, Hao Peng, and Lingpeng Kong. Scaling difusion language models via adaptation from autoregressive models. In Proc. of ICLR, 2025.

Arthur Gretton, Karsten M. Borgwardt, Malte J. Rasch, Bernhard Schölkopf, and Alexander Smola. A kernel twosample test. Journal of Machine Learning Research, 13(25):723–773, 2012.

Tiankai Hang, Shuyang Gu, Chen Li, Jianmin Bao, Dong Chen, Han Hu, Xin Geng, and Baining Guo. Eficient difusion training via Min-SNR weighting strategy. In Proc. of ICCV, 2023.

Haoyu He, Katrin Renz, Yong Cao, and Andreas Geiger. MDPO: Overcoming the training-inference divide of masked difusion language models, 2025. URL https://arxiv.org/abs/2508.13148.

Emiel Hoogeboom, Didrik Nielsen, Priyank Jaini, Patrick Forré, and Max Welling. Argmax flows and multinomial difusion: Learning categorical distributions. In Proc. of NeurIPS, 2021.

Yuezhou Hu, Harman Singh, Monishwaran Maheswaran, Haocheng Xi, Coleman Hooper, Jintao Zhang, Aditya Tomar, Michael W. Mahoney, Sewon Min, Mehrdad Farajtabar, Kurt Keutzer, Amir Gholami, and Chenfeng Xu. Residual context difusion language models. In Proc. of ICML, 2026.

Inception Labs. Mercury: Ultra-fast language models based on difusion, 2025. URL https://arxiv.org/abs/2506.17298.

Allan Jabri, David J. Fleet, and Ting Chen. Scalable adaptive computation for iterative generation. In Proc. of ICML, 2023.

Eric Jang, Shixiang Gu, and Ben Poole. Categorical reparameterization with Gumbel-Softmax. In Proc. of ICLR, 2017.

Metod Jazbec, Theo X. Olausson, Louis Béthune, Pierre Ablin, Michael Kirchhof, João Monteiro, Victor Turrisi, Jason Ramapuram, and Marco Cuturi. Learning unmasking policies for difusion language models. In Proc. of ICML, 2026.

Mingyu Jo, Jaesik Yoon, Justin Deschenaux, Caglar Gulcehre, and Sungjin Ahn. Loopholing discrete difusion: Deterministic bypass of the sampling wall. In Proc. of ICLR, 2026.

Wonjun Kang, Kevin Galim, Seunghyuk Oh, Minjae Lee, Yuchen Zeng, Shuibai Zhang, Coleman Hooper, Yuezhou Hu, Hyung Il Koo, Nam Ik Cho, et al. ParallelBench: Understanding the trade-ofs of parallel decoding in difusion LLMs. In Proc. of ICLR, 2026.

Hamed Karimi, Julie Nutini, and Mark Schmidt. Linear convergence of gradient and proximal-gradient methods unde the Polyak-Łojasiewicz condition. In Proc. of ECML PKDD, 2016.

Jaeyeon Kim, Kulin Shah, Vasilis Kontonis, Sham M. Kakade, and Sitan Chen. Train for the worst, plan for the best: Understanding token ordering in masked difusions. In Proc. of ICML, 2025.

Jaeyeon Kim, Jonathan Geuter, David Alvarez-Melis, Sham Kakade, and Sitan Chen. Stop training for the worst: Progressive unmasking accelerates masked difusion training. In Proc. of ICML, 2026.

Wouter Kool, Herke van Hoof, and Max Welling. Buy 4 REINFORCE samples, get a baseline for free! In ICLR Workshop on Deep Reinforcement Learning Meets Structured Prediction, 2019. URL https://openreview.net/forum?id= r1lgTGL5DE.

Xiang Li, John Thickstun, Ishaan Gulrajani, Percy Liang, and Tatsunori B. Hashimoto. Difusion-LM improves controllable text generation. In Proc. of NeurIPS, 2022.

Bingbin Liu, Sebastien Bubeck, Ronen Eldan, Janardhan Kulkarni, Yuanzhi Li, Anh Nguyen, Rachel Ward, and Yi Zhang. TinyGSM: Achieving >80% on GSM8k with small language models, 2023. URL https://arxiv.org/abs/2312. 09241.

Sulin Liu, Juno Nam, Andrew Campbell, Hannes Stärk, Yilun Xu, Tommi S. Jaakkola, and Rafael Gómez-Bombarelli. Think while you generate: Discrete difusion with planned denoising. In Proc. of ICLR, 2025.

Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete difusion modeling by estimating the ratios of the data distribution. In Proc. of ICML, 2024.

Chris J. Maddison, Andriy Mnih, and Yee Whye Teh. The Concrete distribution: A continuous relaxation of discrete random variables. In Proc. of ICLR, 2017.

Boris Mityagin. The zero set of a real analytic function. Mathematical Notes, 107:529–530, 2020. doi: 10.1134/ S0001434620030189.

Zanlin Ni, Shenzhi Wang, Yang Yue, Tianyu Yu, Weilin Zhao, Yeguo Hua, Tianyi Chen, Jun Song, Cheng Yu, Bo Zheng, and Gao Huang. The flexibility trap: Rethinking the value of arbitrary order in difusion language models. In Proc. of ICML, 2026.

Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Large language difusion models. In Proc. of NeurIPS, 2025.

Shen Nie, Qiyang Min, Shaoxuan Xu, Zihao Huang, Yuxuan Song, Yong Shan, Yankai Lin, Wayne Xin Zhao, Chongx uan Li, and Ji-Rong Wen. Improved large language difusion models, 2026. URL https://arxiv.org/abs/2606.25331.

Olmo Team. Olmo 3, 2025. URL https://arxiv.org/abs/2512.13961.

Fred Zhangzhi Peng, Zachary Bezemek, Sawan Patel, Jarrid Rector-Brooks, Sherwood Yao, Avishek Joey Bose, Alexander Tong, and Pranam Chatterjee. Path planning for masked difusion model sampling, 2025. URL https://arxiv.org/abs/2502.03540.

Fred Zhangzhi Peng, Zachary Bezemek, Jarrid Rector-Brooks, Shuibai Zhang, Anru R. Zhang, Michael Bronstein, Alexander Tong, and Avishek Joey Bose. Planner aware path learning in difusion language models training. In Proc. of ICLR, 2026.

Mihir Prabhudesai, Mengning Wu, Amir Zadeh, Katerina Fragkiadaki, and Deepak Pathak. Difusion beats autoregressive in data-constrained settings. In Proc. of NeurIPS, 2025.

Haolin Ren, Ziyang Huang, Chenhao Yuan, Jun Zhao, and Kang Liu. Trace-based on-policy distillation for masked difusion language models, 2026. URL https://arxiv.org/abs/2607.16872.

Stéphane Ross, Geofrey Gordon, and Drew Bagnell. A reduction of imitation learning and structured prediction to no-regret online learning. In Proc. of AISTATS, 2011.

Benjamin Rozonoyer, Jacopo Minniti, Dhruvesh Patel, Neil Band, Avishek Joey Bose, Tim G. J. Rudner, and Andrew McCallum. Learned relay representations for forward-thinking discrete difusion models, 2026. URL https://arxiv.org/ abs/2605.22967.

Subham S. Sahoo, Marianne Arriola, Yair Schif, Aaron Gokaslan, Edgar Marroquin, Justin T. Chiu, Alexander Rush, and Volodymyr Kuleshov. Simple and efective masked difusion language models. In Proc. of NeurIPS, 2024.

Jiaxin Shi, Kehang Han, Zhe Wang, Arnaud Doucet, and Michalis K. Titsias. Simplified and generalized masked difusion for discrete data. In Proc. of NeurIPS, 2024.

Xingyu Su, Jacob Helwig, Shubham Parashar, Atharv Chagi, Lakshmi Jotsna, Degui Zhi, James Caverlee, Dileep Kalathil, and Shuiwang Ji. Data-eficient autoregressive-to-difusion language models via on-policy distillation, 2026. URL https://arxiv.org/abs/2606.06712.

Yinjie Wang, Ling Yang, Bowen Li, Ye Tian, Ke Shen, and Mengdi Wang. Revolutionizing reinforcement learning framework for difusion large language models. In Proc. of ICLR, 2026.

Russ Webb, Amitis Shidani, Alice Bizeul, and Dan Busbridge. Limits of confidence in difusion, 2026. URL https: //arxiv.org/abs/2609.20581.

Ronald J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning, 8(3):229–256, 1992.

Chengyue Wu, Hao Zhang, Shuchen Xue, Shizhe Diao, Yonggan Fu, Zhijian Liu, Pavlo Molchanov, Ping Luo, Song Han, and Enze Xie. Fast-dLLM v2: Eficient block-difusion LLM. In Proc. of ICLR, 2026a.

Chengyue Wu, Hao Zhang, Shuchen Xue, Zhijian Liu, Shizhe Diao, Ligeng Zhu, Ping Luo, Song Han, and Enze Xie. Fast-dLLM: Training-free acceleration of difusion LLM by enabling KV cache and parallel decoding. In Proc. of ICLR, 2026b.

Kejing Xia, Mingzhe Li, Lixuan Wei, Zhenbang Du, Xiangchi Yuan, Dachuan Shi, Qirui Jin, and Wenke Lee. MetaState: Persistent working memory enhances reasoning in discrete difusion language models. In Proc. of COLM, 2026.

Shijian Xu, Andrea Miele, Metod Jazbec, Volker Roth, Eric Nalisnick, and Ilija Bogunovic. Temporal self-distillation: Faster inference in discrete difusion language models, 2026. URL https://arxiv.org/abs/2609.15177.

Jiacheng Ye, Zhihui Xie, Lin Zheng, Jiahui Gao, Zirui Wu, Xin Jiang, Zhenguo Li, and Lingpeng Kong. Dream 7B: Difusion large language models, 2025. URL https://arxiv.org/abs/2508.15487.

Jefrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. Instruction-following evaluation for large language models, 2023. URL https://arxiv.org/abs/2311.07911.

## Appendix Contents

A Extended background 15   
B Training procedure and experimental setup 16   
B.1 PUMBA training procedure 16   
B.2 Progressive unmasking: the two constructions 16   
B.3 TinyGSM . 17   
B.4 LLaDA-8B 18   
B.5 Compute 21   
C Additional TinyGSM results 22   
C.1 Accuracy and pass@k . 22   
C.2 Carry and gradient estimators 25   
C.3 From teacher forcing to student forcing 26   
C.4 Estimating the mask discrepancy 27   
D Local overfitting at small training u 29   
D.1 Setup 29   
D.2 Results 29   
D.3 Why the loss rises again before the sequence leaves the batch 31   
E Theory for the carry and the BPTT window 33   
Additional results on LLaDA 37   
F.1 Full-canvas SFT schedule 37   
F.2 Full-canvas BPTT 38   
F.3 Block diffusion SFT 40   
F.4 Block diffusion u sweep and BPTT 42   
F.5 The limit case of u equal to the block size in block diffusion . 44   
F.6 Model collapse as a function of training u 44   
F.7 Relevance of confidence thresholding at training time 47   
F.8 Comparison with published LLaDA results 51   
G Relation to the closest work 52

## A Extended background

This section gives the full version of Section 2.

Notation. Let be a finite vocabulary containing a mask token $m \in \mathcal { V }$ , and let $\mathbf { x } = ( x ^ { 1 } , \ldots , x ^ { L } ) \in \mathcal { V } ^ { L }$ be a sequence of length $L ,$ , where $x ^ { i }$ is its i-th token and $\mathcal { M } ( \mathbf { x } ) = \{ i : x ^ { i } = m \}$ its set of masked positions. For any d $\ l \in \mathbb { N }$ , define the probability simplex $\Delta ^ { d } : = \{ \mathbf { p } \in \mathbb { R } _ { > 0 } ^ { d } : \mathbf { 1 } _ { d } ^ { \top } \mathbf { p } = 1 \}$ , and write $\operatorname { C a t } ( \mathbf { p } )$ for the categorical distribution with parameter $\mathbf { p } \in \Delta ^ { | \nu | }$ and $\mathbf { e } _ { v } \in \Delta ^ { | \nu | }$ for the one-hot vector at $v \in \mathcal V$ . Finally, sg[ ] denotes stop-gradient.

Masked diffusion. Masked difusion defines a generative model over sequences through a forward process that masks tokens and a backward process that unmasks them. The forward, or noising, process is a family of corruptions indexed by a level $t \in [ 0 , 1 ]$ that interpolates between a clean sequence $\mathbf { x } _ { 0 } \in \mathcal { V } ^ { L }$ at $t = 0$ and the fully masked sequence $( m , \ldots , m )$ at $t = 1$ . It factorizes over positions, with $q ( x _ { t } ^ { i } \mid x _ { 0 } ^ { i } ) =$ Ca $\left( \alpha _ { t } \mathbf { e } _ { x _ { 0 } ^ { i } } + \left( 1 - \alpha _ { t } \right) \mathbf { e } _ { m } \right)$ , so that $x _ { 0 } ^ { i }$ is retained with probability $\alpha _ { t }$ and replaced by m otherwise. The coeficient $\alpha _ { t }$ decreases monotonically from $\alpha _ { 0 } = 1$ to $\alpha _ { 1 } = 0$ , and a common choice is the linear schedule $\alpha _ { t } = 1 - t$ (Sahoo et al., 2024; Shi et al., 2024). The backward, or denoising, process runs from $t = 1$ to $t ~ = ~ 0$ and fills in the masked positions. Its transitions are defined by the posterior $q ( x _ { 0 } ^ { i } \mid \mathbf { x } _ { t } )$ over the clean token at a masked position, which depends on the unknown data distribution and is intractable. Masked difusion models approximate this posterior with a learned denoiser and generate data by iterating the resulting transitions from the fully masked sequence.

Training of MDMs. Since masking is absorbing and position-wise, the masking ratio of $\mathbf { x } _ { t }$ determines the level $t ,$ so the posterior depends on $\mathbf { x } _ { t }$ only through its mask pattern and revealed tokens (Gat et al., 2024; Sahoo et al., 2024; Amin et al., 2025). The denoiser therefore takes only the corrupted sequence as input, $f _ { \theta } : \mathcal { V } ^ { L } \to ( \Delta ^ { | \mathcal { V } | } ) ^ { L }$ , and returns one categorical distribution per position, $f _ { \theta } ^ { i } ( \cdot \mid \mathbf { x } _ { t } ) \in \Delta ^ { | \nu | }$ . Fitting $f _ { \theta }$ maximizes an evidence lower bound on the data log-likelihood, which for this corruption reduces to a weighted cross-entropy over the masked positions:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { M D M } } = \mathbb { E } _ { \mathbf { x } _ { 0 } , t , \mathbf { x } _ { t } } \left[ \ell ( \mathbf { x } _ { 0 } , \mathbf { x } _ { t } ) \right] , \qquad \ell ( \mathbf { x } _ { 0 } , \mathbf { x } _ { t } ) = \frac { 1 } { t } \sum _ { i \in \mathcal { M } ( \mathbf { x } _ { t } ) } - \log f _ { \boldsymbol { \theta } } ^ { i } ( x _ { 0 } ^ { i } \mid \mathbf { x } _ { t } ) . } \end{array}\tag{A.1}
$$

The factor $1 / t$ is the weight the bound takes under the linear schedule $\alpha _ { t } = 1 - t .$ . It divides the sum by the Lt positions masked in expectation, so heavily masked sequences do not dominate the loss (Bethune et al., 2026). Each training step needs a single denoiser call. It draws a clean sequence $\mathbf { x } _ { \mathrm { 0 } }$ and a level $t \sim \mathcal { U } [ 0 , 1 ]$ masks each token independently with probability $1 - \alpha _ { t }$ , and computes the loss on the predictions at the masked positions.

Inference in MDMs. With a perfect denoiser, revealing one position at a time samples the joint distribution exactly by the chain rule, since each new token is conditioned on all tokens committed so far. Revealing several positions at once is faster, but samples from the product of their marginals instead (Webb et al., 2026). We index these steps by j and write $\mathbf { x } _ { t _ { j } }$ for the sequence after step j, whose masking ratio $t _ { j } = | \mathcal { M } ( \mathbf { x } _ { t _ { j } } ) | / L$ gives its level. Each step applies an unmasking policy g that selects $S = g ( \mathbf { x } _ { t _ { j } } ) \subseteq \mathcal { M } ( \mathbf { x } _ { t _ { j } } )$ and fills every $i \in S$ with a draw from $f _ { \theta } ^ { i } ( { \bf \cdot } \mid { \bf x } _ { t _ { j } } )$ , giving $\mathbf { x } _ { t _ { j + 1 } }$

Revealing positions with more certain marginals keeps the product closer to the joint and gives better samples at a fixed number of steps (Nie et al., 2025; Wu et al., 2026b). Unmasking policies include hand-designed rules (Kim et al., 2025; Ben-Hamu et al., 2025; Ye et al., 2025) and learned policies (Jazbec et al., 2026). The top-u policy (Chang et al., 2022) reveals the u masked positions with the highest confidence $c _ { i } = \operatorname* { m a x } _ { v \in \mathcal { V } } f _ { \theta } ^ { i } ( v \mid \mathbf { x } _ { t _ { j } } )$ For large MDMs, Fast-dLLM (Wu et al., $^ { 2 0 2 6 \mathrm { b } , \mathrm { a } ) }$ is an efective policy that reveals every position with $c _ { i } \geq \tau ,$ or the single most confident position if none exceeds the threshold. The number of positions revealed per step thus follows the confidence of the marginals, and every step reveals at least one token, so generation terminates. Wu et al. (2026b) pair it with semi-autoregressive decoding, which generates the sequence block by block (Arriola et al., 2025). Together, the two preserve generation quality at far fewer denoiser calls per sample, or number of function evaluations (NFEs).

## B Training procedure and experimental setup

In this section, we describe the data, task construction, training protocol, and evaluation procedure for each experimental setting. Unless stated otherwise, comparisons within a setting use the same data, tokenizer, optimizer configuration, sequence length, and number of updates; only the construction of the training sequences changes between training procedures.

## B.1 PUMBA training procedure

In this section, we give the pseudocode of PUMBA (Algorithm 1).

Algorithm 1 PUMBA training for one sample. In practice, we batch and advance several trajectories in parallel, each   
restarting from a fresh sample once fully revealed.   
Require: Carry-augmented denoiser f (Eq. (3.1)); unmasking policy g (Section 3.1); unroll window W   
1: repeat   
2: Draw $\mathbf { x } _ { \mathrm { 0 } }$ from the dataset; $\mathbf { x } \gets ( m , \dots , m ) ;$ $h  0$ ▷ Fresh trajectory, fully masked   
3: while $\mathcal { M } ( \mathbf { x } ) \neq \emptyset$ do ▷ One window, and one optimizer step, per iteration   
4: $\mathcal { L }  0 ; \quad w  0$   
5: while $w < W$ and $\mathcal { M } ( \mathbf { x } ) \neq \emptyset$ do   
6: $( f _ { \theta } ( \cdot \mid \mathbf { x } , h ) , \ h ^ { \prime } )  f _ { \mathrm { c a r r y } } ( \mathbf { x } , h )$ ▷ Forward pass   
7: $\boldsymbol { \mathcal { L } } \gets \boldsymbol { \mathcal { L } } + \ell ( \mathbf { x } _ { 0 } , \mathbf { \dot { x } } , h )$ ▷ Eq. (2.1), with the carry as extra input   
8: $S \gets g ( \mathbf { x } , h )$ ▷ Position selection: no gradient through g   
9: $x ^ { i }  x _ { 0 } ^ { i }$ for $i \in S$ ▷ Revealed positions are teacher-forced   
10: $h  h ^ { \prime } ; ~ w  w + 1$ ▷ Gradient flows through the carry   
11: end while   
12: Update θ from $\nabla _ { \boldsymbol { \theta } } ( \mathcal { L } / w )$   
13: $h  \mathrm { s g } [ h ]$ ▷ Gradient does not cross the window boundary   
14: end while   
15: until the training budget is exhausted

Clarification on the loss. The loss of each window is averaged over its steps, which keeps the update scale independent of W. Since the $1 / t$ factor of Eq. (2.1) normalizes each step’s loss per masked token, all steps of a window contribute equally regardless of their masking ratio.

## B.2 Progressive unmasking: the two constructions

The LLaDA-8B runs (Section 4) use the confidence-threshold construction: each step commits every masked position whose confidence exceeds τ, and at least u positions. The TinyGSM runs (Section 3) instead follow the construction of Kim et al. (2026) as published. Table 3 compares the two, and the paragraphs below give the details. In both, confidence is the maximum probability over the vocabulary, computed in fp32 from the forward pass that produces the loss, and committed positions receive the ground-truth token.

The stage-indexed construction of Kim et al. (2026), used for TinyGSM. The K stages split the revealed fraction of a trajectory into equal intervals: a trajectory is at stage p when a fraction in $[ p / K , ( p + 1 ) / K )$ of its maskable positions is revealed. Each step nominally advances a trajectory by one stage, so a trajectory lasts about K steps. A new trajectory starts at stage 0 with a small fraction of its positions already revealed; these positions are drawn uniformly at random, not by confidence. A step taken at stage p samples a target fraction from the next interval and commits the most confident masked positions until that target is met. The number of positions committed per step is therefore random, with mean close to $L / K$ . The target is capped at $L - 1 .$ , so the reveal step alone never completes a trajectory. Every other masked position with confidence above τ is then committed as well, and the stage is recomputed from the revealed fraction. A trajectory can therefore skip stages, although retirement is still decided by the stage before this recomputation. The maskable region includes the EOS padding that fills the canvas to 512 tokens (the TinyGSM sequence length), so the padding counts toward L and enters the loss. In practice, the threshold commits most of the padding within about one step, so a trajectory lasts far fewer than K steps: about 3.5 at $K = 1 2$ and 7 at $K = 4 2$

Table 3 The two progressive unmasking constructions. L is the number of maskable positions of a sample and K the number of stages. A trajectory is at stage p when a fraction in $[ p / K , ( p + 1 ) / K )$ of its L positions is revealed.
<table><tr><td></td><td>TinyGSM, Kim et al. (2026)</td><td>LLaDA-8B, Section 4</td></tr><tr><td>Start of a trajectory</td><td>A random fraction  $r \sim \mathcal { U } [ 0 , 1 / K )$  of the L positions revealed, chosen uniformly at</td><td>Fully masked</td></tr><tr><td>Progress index</td><td>random Stage  $p \in \{ 0 , \ldots , K - 1 \}$ </td><td>None</td></tr><tr><td>Positions committed per step</td><td>Enough to reach round(rL) revealed positions,  $r \sim \mathcal { U } [ ( p + 1 ) / K , ( p + 2 ) / K ) .$  capped at  $L - 1$ </td><td>The u most confident</td></tr><tr><td>Confidence threshold</td><td>Every other position above  $\tau = 0 . 9 ;$  the stage is then recomputed from the revealed</td><td>Every other position above  $\tau = 0 . 9$ </td></tr><tr><td>Schedule</td><td>fraction K raised stepwise from 12 to 42 (Table 4)</td><td>u fixed per run (Table 7)</td></tr><tr><td>Retirement</td><td>After the step taken at stage  $K - 1$ </td><td>Once the response and its EOS are fully revealed</td></tr><tr><td>Maskable region</td><td>Response, including the EOS padding to 512 tokens</td><td>Response, its first EOS, and padding (Appendix B.4.1)</td></tr></table>

Our construction, used for LLaDA-8B. Every trajectory starts fully masked, and there is no stage index and no schedule. The prompt is always visible and never masked. Each step commits the u most confident masked positions, together with every other masked position whose confidence exceeds $\tau = 0 . 9$ . This is the same set as committing every position above τ and topping up with the next most confident positions until at least u are committed. A trajectory is retired once its response and EOS are fully revealed, and a fresh, fully masked sample takes its place. On the full canvas, padding inside the attended window is also maskable, so some of the u positions committed at a step can be padding, which does not enter the loss.

Why we depart from the published construction. Progressive unmasking aims to train on the sequences that the inference policy visits (Section 3.1). The published construction approximates them with a schedule: a trajectory nominally lasts about K steps, so each step commits about $L / K$ positions, whatever the length L of the sample. A confidence-threshold decoder instead commits a number of positions per step that does not depend on L, so longer responses take more steps. The two agree only when L is nearly the same for every sample. On TinyGSM, L includes the EOS padding to 512 tokens and varies only with the prompt length, so we keep the published construction and its curriculum on K, and our PU baseline is the recipe of Kim et al. (2026). On Dolci-Instruct-SFT, the response length ranges from 14 tokens at the 10th percentile to 778 at the 90th (Figure 6). At $K = 4 2$ , the final value of that curriculum, a 14-token response would receive about 0.3 commits per step, so most of its steps would commit nothing beyond the positions above τ, while a 778-token response would receive about 18.5 commits per step. No single K therefore matches the decoding policy across the mixture. Our construction drops the schedule and takes its reveal rule from the decoding policy itself: every trajectory starts fully masked, as decoding does, and each step commits by the same confidence-threshold rule, with τ and u as its only parameters. It also removes the random reveal at stage 0 and the sampled commit count, which have no counterpart at inference.

## B.3 TinyGSM

Dataset and task. The TinyGSM experiments provide the controlled small-scale setting used to isolate the contribution of progressive unmasking, the on-trajectory hidden-state carry, and BPTT.

We train on TinyGSM (Liu et al., 2023), holding out 2% for validation, and evaluate on the 1,319 problems of the GSM8K test set (Cobbe et al., 2021). The prompt is always visible; only the response is masked and enters the loss, as typically adopted in the SFT settings (Nie et al., 2025; Ye et al., 2025). Each TinyGSM response is a Python function, so we score a generation by executing the function it defines in a sandbox with a 1-second limit and comparing its return value with the GSM8K answer, exactly for integers and to within $1 0 ^ { - 3 }$ for floats. We report this accuracy with EMA weights. Decoding is greedy and reveals the u most confident masked positions at each step.

Compared procedures. All difusion variants share the architecture and start from random initialization. For W > 1, we lower the per-GPU batch size and accumulate gradients so that the efective batch stays at 256. MDM 2 and 8 train for 40 and 160 epochs. The AR model has the same size ( 125M parameters), uses causal attention, and is trained with next-token cross-entropy on the same data, averaged over the response tokens up to and including the first EOS. The denoiser, tokenizer, output parameterization, and loss shape are held fixed across the difusion runs, whose loss also covers the EOS padding. Jo et al. (2026) propose carry dropout, which uses the carry only a fraction p of the time during training and helps for p  [0.5, 0.9] in their experiments. Our TinyGSM runs do not use it (p = 1), so the carry is always passed on. We also follow the curriculum on K of Kim et al. (2026): K starts at 12 and rises by 3 every 30k updates from update 60k, reaching 42 at update 330k. Each GPU keeps 32 trajectories, which start at stages spread evenly over $\{ 0 , \ldots , K - 1 \}$ . Every change of K re-initializes all trajectories this way, with a zero carry.

Table 4 Hyperparameters for the TinyGSM experiments.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Architecture</td><td>Bidirectional transformer, 14 layers, hidden size 512, MLP size 1536, 8 heads, tied embeddings (≈125M parameters)</td></tr><tr><td>Tokenizer</td><td>Qwen2, restricted to its first 151,645 ids; id 151,644 serves as the mask token</td></tr><tr><td>Maximum sequence length</td><td>512</td></tr><tr><td>Optimizer</td><td>AdamW, learning rate  $3 \times 1 0 ^ { - 4 }$  , linear warmup over 1000 updates, then cosine decay to 0</td></tr><tr><td>Weight decay / gradient clipping</td><td>0.01 / 1.0</td></tr><tr><td>Batch size / training length</td><td>256 sequences / 20 epochs (≈907k updates) 0.9999</td></tr><tr><td>EMA decay PU schedule</td><td>Stage-indexed, K raised stepwise from 12 to 42 over the first 330k</td></tr><tr><td></td><td>updates; confidence threshold 0.9 (Appendix B.2) Last hidden state before the output projection, added to the token</td></tr><tr><td>Carry</td><td>embeddings through a zero-initialized LayerNorm; zero at the first step; stored in fp32. Dropout p = 1</td></tr><tr><td>BPTT window W</td><td>{1, 2, 4, 8}</td></tr></table>

## B.4 LLaDA-8B

The LLaDA-8B experiments follow the two-stage procedure of Section 4. The SFT stage fine-tunes LLaDA-8B-Base (Nie et al., 2025) with the MDM objective. The Post-SFT stage continues from an SFT checkpoint with one of the compared training procedures. Table 5 lists the shared settings, Appendices B.4.1 and B.4.2 those specific to each stage, and Appendix B.4.3 the evaluation.

Dataset. Both stages train on Dolci-Instruct-SFT from the OLMo 3 post-training pipeline (Olmo Team, 2025). The mixture contains instruction-following, mathematical, code, conversational, and tool-use examples. We use the full mixture, about 2.15M (prompt, response) pairs. Prompts are rendered as plain-text role labels (user:, assistant:) without a BOS token, and the response is followed by a trainable EOS that supplies the stop signal. Each sample is truncated to a 4096-token canvas by reserving the response and its EOS first and left-truncating the prompt into the remainder. Truncation therefore never removes response tokens unless the response alone exceeds the canvas (which almost never happens; only 1645 samples); it removes 16.2% of all prompt tokens, concentrated in the 2.5% of samples that do not fit the canvas. Figure 6 shows the prompt and response lengths before truncation, and Table 8 reports the token budget after it.

![](images/383ad0541ebafb7f6f7f22ff8fb81ee2d12aa3ae1b1c40434ffd75ae4ffa0e67.jpg)  
Figure 6 Empirical CDFs of prompt, response, and total length per sample in Dolci-Instruct-SFT, before truncation to the 4096-token canvas (vertical dotted line). Response length includes the terminal EOS. The horizontal dotted line marks the 97.5% of samples that fit the canvas.

The benchmarks in Appendix B.4.3 are used only for downstream assessment. We run no separate contamination check: every compared run trains on the same data, so any overlap with the benchmarks is shared by all of them.

Objective. The prompt is never masked, but not every masked position enters the loss: the response and its first EOS always do, and whether padding does depends on the setting (Appendix B.4.1). The MDM objective weights each sample by 1/t (Eq. (2.1)), and we cap this factor at $\gamma = 5 ,$ , following the min-SNR weighting of Hang et al. (2023). In our runs, the cap slightly stabilizes training. An auxiliary z-loss is weighted $1 0 ^ { - 5 }$ , and the loss is normalized by the number of positions it is computed on, not by the maskable ones.

Table 5 Settings shared by the SFT and Post-SFT stages of LLaDA-8B.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Model</td><td>LLaDA-8B-Base architecture and tokenizer; hidden size 4096; 126,464 tokens, with the existing &lt;|mdm_mask|&gt; as the mask token</td></tr><tr><td>Data</td><td>Dolci-Instruct-SFT, full mixture, about 2.15M (prompt, response) pairs</td></tr><tr><td>Canvas</td><td>4096 tokens; response and EOS reserved first, prompt left-truncated</td></tr><tr><td>Optimizer</td><td>AdamW  $( \beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 5 , \epsilon = 1 0 ^ { - 8 } )$  , weight decay 0.01, global gradient-norm clipping at 1.0</td></tr><tr><td>Learning-rate schedule</td><td>Linear warmup from 0, then cosine decay to 0</td></tr><tr><td>Precision</td><td>bfloat16 parameters, gradient reduction, and outputs</td></tr><tr><td>Global batch size</td><td>128 sequences, 2 per GPU on 64 GPUs; no gradient accumulation</td></tr><tr><td>Parallelism</td><td>Hybrid-sharded data parallelism: parameters sharded within each node of 8 GPUs, replicated across 8 nodes</td></tr><tr><td>Hardware</td><td>8 nodes of 8 NVIDIA H100 80GB, except where Tables 6 and 7 state B200</td></tr><tr><td>MDM loss weighting</td><td>min  $( 1 / t , 5 ) ;$  z-loss  $1 0 ^ { - 5 } ;$  normalized by the positions in the loss</td></tr></table>

## B.4.1 SFT stage

The SFT stage trains with the MDM objective in the same attention pattern the model later decodes with (Table 6). Full-canvas SFT attends bidirectionally over the whole canvas. As in the SFT of Nie et al. (2025), each local batch is padded with EOS only up to its longest sequence rather than to the full canvas. In preliminary runs, we observed this made performance vary less with the canvas length. Unlike Nie et al. (2025), we keep these padding tokens out of the loss: they are attended and can be masked, but only the first EOS after the response is trained on. In LLaDA, EOS and padding are the same token, and when we also trained on all the padding of the full canvas, the model collapsed to EOS or to very short responses, an efect Nie et al. (2025) also report. Block difusion SFT follows the vectorized block MDM algorithm of Arriola et al. (2025) with blocks of 32 tokens. Each sample holds a clean and a noised copy of its response in one sequence of twice the length, and the attention mask lets each noised block attend to itself and to the clean blocks before it, so all blocks train in one forward pass. We interleave the two copies block by block, which leaves this attention pattern unchanged. Each response is rounded up to a whole number of blocks with EOS, and this filler enters the loss. Both settings select the peak learning rate by a sweep at 1.5 epochs (Appendices F.1 and F.3).

Table 6 SFT stage of LLaDA-8B. Wall-clock time is the training run alone, without evaluation.
<table><tr><td></td><td>Full canvas</td><td>Block diffusion</td></tr><tr><td>Attention</td><td>Bidirectional</td><td>Block-causal, bidirectional within blocks of 32</td></tr><tr><td>Peak learning rate</td><td> $5 \times 1 0 ^ { - 6 }$ </td><td> $5 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>Warmup</td><td>2000 updates</td><td>2000 updates</td></tr><tr><td>Training length</td><td>1.5 epochs, 25,195 updates; 3-epoch control</td><td>6 epochs, 100,781 updates; 1.5- and 3-epoch controls</td></tr><tr><td>Padding</td><td>Attended and maskable, not in the loss</td><td>EOS filler to the block edge, in the loss</td></tr><tr><td>Wall-clock (reported run)</td><td>9.4 h on 64 H100</td><td>48.6 h on 64 B200</td></tr></table>

## B.4.2 Post-SFT stage

Every Post-SFT run trains for 4000 updates at a peak learning rate of $2 . 5 \times 1 0 ^ { - 6 }$ , with a linear warmup over the first 200 updates (5% of the run) and cosine decay to 0. This is half the SFT peak, and together with the new warmup it keeps the continuation from the SFT checkpoint as smooth as the 4000-update horizon allows. Full-canvas runs start from the 1.5-epoch full-canvas SFT checkpoint, and block difusion runs start from the 6-epoch block difusion SFT checkpoint. Within each setting, every Post-SFT run therefore shares its warm start, data, recipe, and number of updates, and only the construction of the training sequences changes. The Post-SFT MDM control uses the same recipe with the MDM training algorithm.

Progressive unmasking follows the construction in Appendix B.2. Each GPU keeps two trajectories, so that the 64 GPUs advance 128 trajectories per update, and every trajectory advances at every update. Under block difusion training, each block of 32 positions is its own trajectory, and all blocks of a sample advance in lockstep, so a block is retired after at most 32/u steps. The trajectories are not checkpointed, so a resumed run starts from fresh, fully masked ones. The PU loss weights each sequence by min $( 1 / t _ { k } , 5 )$ , where $t _ { k }$ is the realized masked fraction of the positions in the loss: the response up to and including its first EOS on the full canvas, and each block separately under block difusion. It is rescaled by the fraction of sequences in the batch that still have masked positions in the loss, and it is averaged over the W forward passes of a window. The carry is the last hidden state after the final normalization, one vector per position, and it is added to the token embeddings through a LayerNorm whose weight and bias start at zero. This LayerNorm is new at the start of Post-SFT and trains at the same learning rate as the rest of the model. The carry is zero at the first step of a trajectory and is stored in fp32 between updates; the sum with the token embeddings is cas back to bfloat16 before the first transformer block. Carry dropout is not used (p = 1).

## B.4.3 Evaluation

We measure verifiable instruction following with IFEval (Zhou et al., 2023), mathematical reasoning with GSM8K (Cobbe et al., 2021), and program synthesis with MBPP (Austin et al., 2021b). All three run in lm-eval 0.4.9.1 (Gao et al., 2024). We report prompt-level strict accuracy on IFEval (0-shot), exact match with flexible answer extraction on GSM8K (8-shot), and pass@1 on MBPP (3-shot). The score is the mean of the three. Decoding uses the Fast-dLLM confidence-threshold policy (Wu et al., 2026b). It runs left to right in blocks of 32 tokens for every model, at temperature $T = 0 ,$ , and sweeps the confidence threshold $\tau \in \{ 0 . 7 , 0 . 8 , 0 . 9 , 1 \}$ . τ = 1 recovers one-token-per-step confidence decoding. The generation budget is 1024 tokens for GSM8K and MBPP and 1280 tokens for IFEval, the values lm-eval sets per task. Carry models pass the carry from each decoding step to the next. On the full canvas, the carry also passes from one block to the next. Under block difusion, we reset the carry to zero at the start of every block, to match training, where the vectorized layout of Appendix B.4.1 advances all blocks of a sample in parallel from a zero carry.

Table 7 Post-SFT stage of LLaDA-8B. Wall-clock time is the training run alone, without evaluation, on 64 H100 unless stated.
<table><tr><td></td><td>Full canvas</td><td>Block diffusion</td></tr><tr><td>Warm start</td><td>1.5-epoch SFT (25,195 updates)</td><td>6-epoch SFT (100,781 updates)</td></tr><tr><td>Learning rate</td><td>Peak  $2 . 5 \times 1 0 ^ { - 6 }$ </td><td>, 200 warmup updates, cosine to 0, 4000 updates</td></tr><tr><td>Confidence threshold τ</td><td>0.9</td><td>0.9</td></tr><tr><td>Reveal count u</td><td>{32, 64} reported</td><td>{8, 16} per block reported</td></tr><tr><td>BPTT window W</td><td>{1, 2, 4, 8}</td><td>{1, 2, 4} at u = 8; {1, 2} at u = 16</td></tr><tr><td>Carry</td><td>Final hidden state; zero-initialized LayerNorm; zero at the first step; fp32 storage; p = 1</td><td></td></tr><tr><td>Wall-clock</td><td> $\mathrm { M D M \ 1 . 7 h ; \ a t \ } u = 3 2 \colon \mathrm { P U \ 1 . 8 h } ,$   $W = 1 ~ 2 . 2 \mathrm { h } , W = 2 ~ 4 . 2 \mathrm { h } , W = 4$   $7 . 7 \mathrm { h } , W = 8 \ 1 4 . 4 \mathrm { h }$ </td><td> $\mathrm { M D M ~ 3 . 1 ~ h } ; \mathrm { ~ a t ~ } u = 1 6 ; \mathrm { ~ P U ~ 3 . 0 ~ h } ,$   $W = 1 ~ 3 . 0 \mathrm { h } , W = 2 ~ 5 . 6 \mathrm { h } ; \mathrm { a t } ~ u = 8 \mathrm { : }$   $W = 4 ~ 6 . 9 \mathrm { h ~ o n ~ } 6 4 ~ \mathrm { B } 2 0 0$ </td></tr></table>

Prompts use one of two formats, both plain text without special tokens such as BOS or EOS. The plain prompt format concatenates the few-shot demonstrations and the query into a single string, as lm-eval does by default. The chat prompt format renders them as alternating turns with the same user: and assistant: labels as our SFT data, and ends with assistant:. We report all results in the chat prompt format, except the base-model check in Appendix F.8.

Table 8 Token budget of LLaDA-8B SFT, per epoch over Dolci-Instruct-SFT, with the total of the reported run in parentheses: 1.5 epochs on the full canvas (25,195 updates of 128 sequences) and 6 epochs under block difusion (100,781 updates). Counts are measured over one epoch of the training dataloader and, under block difusion, of the vectorized layout the training step builds from it. Content is prompt plus response plus the terminal EOS afte truncation to the 4096-token canvas, EOS filler completes the last block of each response, in the loss counts the positions the loss is computed on, and attended counts the positions the model attends over; under block difusion, these are the context and the clean and noised copies of the response. Total canvas counts every sample as a full canvas, whatever part of it the sample uses: 4096 positions on the full canvas and 8192 under block difusion, whose layout holds the clean and noised copies side by side.
<table><tr><td></td><td>Full canvas</td><td>Block diffusion</td></tr><tr><td>Content (prompt + response + EOS)</td><td>1.81 B (2.71 B)</td><td>1.81 B (10.84 B)</td></tr><tr><td>of which prompt</td><td>1.22 B (1.83 B)</td><td>1.22 B (7.34 B)</td></tr><tr><td>of which response + EOS</td><td>0.58 B (0.88 B)</td><td>0.58 B (3.50 B)</td></tr><tr><td>EOS filler</td><td></td><td>0.03 B (0.20 B)</td></tr><tr><td>In the loss</td><td>0.58 B (0.88 B)</td><td>0.62 B (3.71 B)</td></tr><tr><td>Attended</td><td>2.32 B (3.49 B)</td><td>2.46 B (14.75 B)</td></tr><tr><td>Total canvas</td><td>8.81 B (13.21 B)</td><td>17.61 B (105.68 B)</td></tr></table>

## B.5 Compute

The runs reported in this paper used about 85,000 H100 and 7,300 B200 GPU-hours. The TinyGSM experiments account for about 36,000 H100 GPU-hours, and the LLaDA-8B experiments for about 49,000 H100 and 7,300 B200 GPU-hours, of which evaluation takes about 24,000 H100 GPU-hours.

## C Additional TinyGSM results

## C.1 Accuracy and pass@k

Figure 7 complements Table 2 with the accuracy of each variant across decoding budgets u and during training at u=2.

![](images/2cdb6031ab0ec4414a2dde29a79c8f315f26d77641aaa28e08f8b7d194905453.jpg)

![](images/9026c3aff08d64a9c16a77ea81340bb93fc0a07c29aea8334b7ab503d8adc6c0.jpg)  
Figure 7 GSM8K accuracy on TinyGSM. Left: final checkpoint against the number of tokens revealed per step, u. Right: accuracy at u=2 during training; the gray area marks the annealing phase of the PUMA recipe. Dotted lines: AR at its final and best checkpoints.

We investigate the efect of the number of sampling tokens per denoising step u in Figure 8 during training. We observe that the trend and ordering of the models remain the same, with carry $( + \ W { = } 8 )$ performance matching AR’s best performance. Note that AR’s performance decays with more epochs of training as expected. We also study pass@k as an alternative metric to accuracy. Ni et al. (2026) show that pass@k can be used to compare sample diversity across diferent methods, such as AR and MDM. We change the sampling temperature $T \in \{ 0 . 2 , 0 . 7 , 1 . 0 \}$ . Figure 9 shows the result of this experiment. We observe that PU, PU with carry, and BPTT have a positive efect on this metric. While AR seems to be less sensitive to temperature and performs better at higher temperatures, we see that at lower temperatures, BPTT with higher values of W can outperform AR.

![](images/8a1b27b614fd20e4dd74efffa7388815018bd354f72d14468470f38ca795a020.jpg)  
(a)

![](images/55feb3adf657789a47cb559ed6bc83a3caea5f8eac2e15b89b7dacf832d9731f.jpg)  
(b)

![](images/8dc2fe0fa853c923a40fa8b7dc565600c15030e0459ac159be868e76af369b73.jpg)  
(c)

![](images/10c7f3ad3b18f1573612b9cfa2b610b8fa36f572b11028fcdeaede81d8b89711.jpg)  
(d)  
Figure 8 GSM8K accuracy during training at u=2 (top) and u=3 (bottom). Right: zoom on the dashed window. Shaded: ramp of the PU schedule.

![](images/cbcc991825a8aeb25942589bba975d0320e77046166ae44b4a7a8737908a8124.jpg)  
(a)

![](images/1f1e7bafbb4396ff056dbb296bf0ade58c957e726734975b212146526adfac2d.jpg)  
(b)

![](images/f79a71f37f47203bc7493c0fb2045636f60cc71db958f87ad9d3edf24d84267b.jpg)  
(c)

![](images/5d1fe83d26e1d5b703be21c41272f83a9c73eba0f776be0a269d2dded619eed2.jpg)  
(d)

![](images/de7577cb7910bca4fd976fdbb407f4e0ffb31e63e76d4074853ad43f72c84612.jpg)  
(e)

![](images/ab0a1aa8d28adc97936bdcf68b90f8b460bb455f442dc0f28fff23c2fd2c2604.jpg)  
(f)  
MDM PU PU + Carry, W=1

Figure 9 pass@k from 100 samples per problem (Chen et al., 2021), at u=2 (top) and u=3 (bottom). AR leads at higher temperatures while carry with W=8 has comparable performance at lower temperatures; the ordering among difusion variants is unchanged with PU improving over MDM, carry and BPTT having more positive efects in addition to PU.

## C.2 Carry and gradient estimators

Figure 10 shows the training curves behind Table 1, together with the runs that separate the carry from the multi-step objective, and Figure 11 evaluates their latest checkpoints across decoding budgets u. All runs keep the PUMA recipe at its initial setting instead of annealing it (Section 3.2). They share the model, an efective batch size of 256, and a learning rate of $3 \times 1 0 ^ { - 4 }$ , and are evaluated zero-shot on GSM8K. We sweep the Gumbel-softmax temperature over 0.5, 1, 2 and the REINFORCE weight over 0.1, 1, 10 , with a leave-one-out baseline and normalized advantages, $\mathrm { i . e . , }$ to reduce the variance of gradients, each sequence’s reward is compared with the average reward of the other sequences in the batch, and the resulting diferences are rescaled to unit standard deviation. The carry without BPTT (W=1) tracks PU throughout training, and supervising two steps per update without a carry adds little, whereas the carry trained by BPTT at $W { = } 2$ and $W { = } 4$ separates from both early in training (Figure 10, left). No discrete bridge exceeds the two-step objective without a carry (right), so the gain of Table 1 comes from training the carry through the following steps rather than from supervising several steps per update. The same ordering holds at every decoding budget we evaluate, and the gap narrows at $u { = } 1 6 ,$ , where every variant degrades (Figure 11).

![](images/0467a23f195ed1cf4d702494933f7f2fe1e99ad82e548c63087d1fc5f99462bb.jpg)  
Figure 10 GSM8K accuracy at u=2 during training. Left: MDM, PU, the carry without BPTT (W=1), a two-step objective without a carry, and the carry with BPTT (W=2, 4). Right: the carry and three gradient estimators at $W { = } 2 .$ . The W=4 run is at 63% of its training budget.

![](images/1868ad0107854e920e8377c869ecd2e10e7fdb07255a785b9ca40ed655340bc7.jpg)  
Figure 11 GSM8K accuracy against the number of tokens revealed per step, u, at the latest checkpoint of the runs of Figure 10.

## C.3 From teacher forcing to student forcing

PU writes ground-truth tokens at the revealed positions (Section 3.1), whereas at inference these positions hold the model’s own predictions. We test whether scheduled sampling (Bengio et al., 2015), the standard remedy for this mismatch, improves PU on TinyGSM.

Setup. We train PU without the carry, with the architecture and training budget of Table 4. For simplicity, we fix the number of stages to K=12 instead of raising it from 12 to 42 as in Kim et al. (2026), so that the student-forcing schedule is the only schedule that changes during training. Each revealed position receives the ground-truth token with probability ε and the model’s argmax prediction otherwise, with a straight-through gradient through the predicted tokens. The ratio ε follows the inverse-sigmoid decay of Bengio et al. (2015) with a floor,

$$
\varepsilon ( s ) = \operatorname* { m a x } \Bigl ( 0 . 1 , \ \frac { k } { k + \exp ( s / k ) } \Bigr ) ,
$$

where s is the optimizer step after a 1,000-step warmup and k sets how fast training moves from teacher to student forcing. We sweep eight values of k from 15,542 to 98,862, with one run per value, and report GSM8K accuracy at u=2 as the mean of the last five evaluations.

Results. Figure 12 shows that no schedule improves on the PU baseline (39.0%). Accuracy increases with the average ratio of ground-truth tokens ε¯ and reaches the baseline only as ε¯ approaches 1. The two closest runs (37.5% and 38.7%) have $\bar { \varepsilon } = 0 . 9 7$ and 0.99: their schedules never reach the floor and are almost pure teacher forcing. Below $\bar { \varepsilon } \approx 0 . 9$ , every run falls under the MDM baseline (34.3%), and at $\bar { \varepsilon } = 0 . 2 4$ accuracy drops to 13.7%. In additional runs, detaching the predicted tokens instead of using the straight-through gradient changes accuracy by less than one point, so the drop comes from the predicted tokens in the context rather than from the gradient estimator.

Why the carry avoids this failure. Committed positions are never revisited, so a wrong predicted token stays in the context for the rest of the trajectory, while the loss still asks for $\mathbf { x } _ { \mathrm { 0 } }$ at the remaining masked positions. The model is then trained to predict targets that contradict its own context. The carry exposes each step to its predecessor’s imperfect computation through a continuous input, while the visible tokens stay consistent with $\mathbf { x } _ { \mathrm { 0 } }$ (Section 3.2). The model can therefore learn how much to rely on the carry, and BPTT trains the predecessor to produce a useful one.

We do not rule out a student-forcing schedule that outperforms teacher forcing, for example one that remasks wrong commitments, but none of the schedules we tried did.

![](images/19aa3c976009d45aff03c42a04039171f71d02bc97db349787356519a1a8a8e0.jpg)  
Figure 12 Scheduled sampling on PU (TinyGSM). Right: ratio of ground-truth tokens ε during training for each schedule; the dot on the y-axis marks its average ε¯. Left: GSM8K accuracy at u=2 against $\bar { \varepsilon } ,$ at the same height as the corresponding schedule. $\varepsilon = 1$ is pure teacher forcing.

## C.4 Estimating the mask discrepancy

This section details how we estimate $D _ { \mathrm { m a s k } }$ (Definition 1) for a given checkpoint.

Masks and kernel. Prompt positions are never masked, so we restrict masks to the N response positions of a validation problem, $\mathbf { m } \in \{ 0 , 1 \} ^ { N }$ , where $m ^ { i } = 1$ if position i is masked. The Hamming distance counts the positions on which two masks difer,

$$
d _ { H } ( { \bf m } , { \bf m } ^ { \prime } ) = \sum _ { i = 1 } ^ { N } \bigl | m ^ { i } - m ^ { \prime i } \bigr | .
$$

We set the kernel bandwidth relative to the response length, $\sigma = 0 . 2 N$ , so that the kernel has the same scale for short and long responses. We evaluate the masking ratios $t \in \{ 0 . 1 , 0 . 2 , \ldots , 0 . 9 \}$

Inference side. We run the sampler used for evaluation: greedy decoding that, at each step, reveals the u most confident masked positions and fills them with the most likely token. For carry models, the carry starts at zero and is passed from step to step, as at evaluation. This sampler is deterministic, so $P _ { \mathrm { i n f } }$ is a point mass at a single mask $\mathbf { m } _ { \mathrm { i n f } }$ . We take $\mathbf { m } _ { \mathrm { i n f } }$ to be the mask at the first step j at which the masked fraction reaches t,

$$
\frac { | \mathcal { M } ( \mathbf { x } _ { t _ { j } } ) | } { N } \leq t .
$$

For $u > 1$ , this mask can have up to u  1 fewer masked positions than tN. This slightly increases $D _ { \mathrm { m a s k } }$ in the same way for all methods.

Training side. Our MDM implementation first draws the number of masked positions and then masks a uniformly random subset of that size. At ratio t, we therefore draw $N _ { \mathrm { t r } } ~ = ~ 3 2$ masks, each masking a uniformly random subset of round(tN) response positions. The PU construction of Kim et al. (2026) does not allow sampling a mask at a requested ratio: masks arise along trajectories whose reveal counts vary. We therefore run the training construction itself on $\mathbf { x } _ { \mathrm { 0 } }$ and collect the masks it produces.

1. We create a pool of 2K trajectories, where K is the number of stages the checkpoint was trained with. Every trajectory is a copy of $\mathbf { x } _ { \mathrm { 0 } }$ and starts at a random stage, as in training.

2. We advance the pool with the training update: reveal the most confident positions, reveal every position whose confidence exceeds 0.9, and restart a trajectory from a new random stage once it is complete. For carry models, the carry is passed along each trajectory; for $W > 1$ , the pool is advanced once per window of W steps, as in Algorithm 1.

3. At every step, we record the mask of each trajectory and assign it to the nearest ratio t on the grid, if it lies within 0.05 of it. Each ratio keeps at most $N _ { \mathrm { t r } } = 3 2$ masks.

We run each pool for K+1 steps and use up to five independent pools, stopping once every ratio holds 32 masks. At the checkpoints we report, all ratios are full for all problems.

Estimator. Let $\mathbf { m } _ { 1 } , \ldots , \mathbf { m } _ { N _ { \mathrm { t } } }$ be the training masks at ratio t. The unbiased MMD estimator (Gretton et al., 2012) compares the average kernel value within the training masks with the average kernel value between training and inference masks. Since $P _ { \mathrm { i n f } }$ is a point mass, its within-set term equals $k ( { \bf m } _ { \mathrm { i n f } } , { \bf m } _ { \mathrm { i n f } } ) = 1$ , and the estimator becomes

$$
\widehat { \widetilde \mathrm { M M D } } _ { k } ^ { 2 } = \frac { 1 } { N _ { \mathrm { t r } } ( N _ { \mathrm { t r } } - 1 ) } \sum _ { i \neq j } k ( \mathbf { m } _ { i } , \mathbf { m } _ { j } ) \ + \ 1 \ - \ \frac { 2 } { N _ { \mathrm { t r } } } \sum _ { i = 1 } ^ { N _ { \mathrm { t r } } } k ( \mathbf { m } _ { i } , \mathbf { m } _ { \mathrm { i n f } } ) .\tag{C.1}
$$

The first term is large when the training masks are similar to each other. The last term is large when they are similar to the inference mask. $D _ { \mathrm { m a s k } } ( t )$ is the average of this estimate over 50 problems from the validation split, using the EMA weights at the final checkpoint.

Binning control. MDM masks have exactly round(tN) masked positions, whereas PU masks are grouped within 0.05 of t and therefore have varying counts. To check that this diference does not drive the comparison, we also build MDM masks with the same procedure as PU: we draw i.i.d. MDM masks at random ratios and group them in the same way. At u=2, this re-binned control difers from MDM by at most 0.0015, so we omit it from Figure 3; the gap between MDM and PU is therefore not an artifact of the grouping.

## D Local overfitting at small training u

This section details the TinyGSM experiments behind Figure 2 (Section 3.1). We describe the runs and how we follow individual sequences (Appendix D.1), report the sweep over u and the ablations that separate the causes of the failure (Appendix D.2), and explain why, at u=2, the loss of a sequence rises again before the sequence leaves the batch (Appendix D.3).

## D.1 Setup

Runs. We use the model, data, optimizer and batch size of Appendix B.3, with three changes that isolate the efect of u. First, each run lasts 3,000 updates: the learning rate warms up linearly over 600 updates and then stays at $3 \times 1 0 ^ { - 4 }$ , so that every insertion (below) happens at the same learning rate. Second, the number of stages K is fixed within a run, with $K \in \{ 2 3 1 , 1 1 5 , 5 7 , 7 , 4 \}$ , that is $K \approx L / u$ for $u \in \{ 2 , 4 , 8 , 6 4 , 1 2 8 \}$ and the $L \approx 4 6 0$ maskable positions of a sample. Third, the stage-indexed construction of Appendix B.2 runs without its confidence threshold $\tau ,$ so that no trajectory skips a stage: every trajectory has $J = K$ decoding steps, and every sequence stays in the batch for exactly J updates. The 32 trajectories of a GPU start at distinct stages drawn at random from $\{ 0 , \ldots , K - 1 \}$ when $K > 3 2$ , and cover all stages evenly otherwise. There is no carry and no BPTT, and MDM follows the same schedule. Each configuration is trained once.

Inserted sequences. We follow 16 sequences from the validation split, which are not trained on otherwise. Twelve of them are inserted into the batch, four at each of updates 750, 1,500 and 2,250, and the other four are never inserted. Under PU, an inserted sequence takes the place of the next sequence to enter the pool of one GPU and starts at stage 0; it is then trained on for exactly J consecutive updates, as one of the 256 sequences of each batch. Under MDM, it replaces one sequence of one batch, for a single update. In a separate run at $u { = } 2$ , we also followed the next 16 training sequences to enter the pool, at their natural times. They respond like the sequences inserted in the same run, with a completion loss 65% lower 20 updates after entry, so inserted sequences behave like ordinary training sequences.

Completion loss. To measure how well the model predicts a sequence, we reveal its prompt and the first half of its answer, and average the cross-entropy over the rest of the answer, up to and including its first EOS; the padding that follows is not included in the loss. This mask is the same throughout the run, and the loss is computed with the raw weights, not the EMA, after every update. The validation completion loss averages the same quantity over 256 other validation sequences, computed every 20 updates, and we report its mean over the last 200 updates. The relative completion loss of an inserted sequence is its completion loss divided by its mean over the 20 updates before insertion, divided by the same ratio for the validation completion loss, which removes the drift shared by all sequences. Figure 2 (middle) averages its logarithm over the 12 inserted sequences and smooths it over 9 updates; the bands show one standard error.

## D.2 Results

Sweep over u. Table 9 summarizes the runs of Figure 2, and Figure 13 (left) plots their validation completion loss against u. As u decreases, each sequence stays longer in the batch and a run visits fewer distinct sequences, 3.6k at $u { = } 2$ against 768k for MDM. The inserted sequences are then fitted more deeply while they are in the batch, and the validation completion loss is higher: PU beats MDM when each sequence stays in the batch for only a few steps, but trails it for $u \leq 8$ . At every $u ,$ the gain has disappeared 300 updates after the sequence leaves the batch. The training loss moves in the opposite direction $( { \mathrm { F i g u r e ~ 2 } } , \ r i g h t )$ : at $u { = } 2 ,$ it falls to 0.07, while the validation completion loss is 6.92, the highest of the sweep.

Ablations. Table 10 changes one aspect of PU at a time, at $u \in \{ 2 , 4 , 8 \}$ , and Figure 14 follows the inserted sequences under the first two changes at u=2.

• PU schedule, MDM masks. Each sequence still stays in the batch for J consecutive updates, but every update draws a fresh MDM mask for it instead of advancing its trajectory. The inserted sequences are fitted even more deeply, with a relative completion loss of 0.02 at $u { = } 2$ that is still 0.52 300 updates after they leave, and the validation loss is as high as with PU or higher. The failure therefore does not come from the masks of PU (Figure 14, middle).

Table 9 Runs of Figure 2. Distinct sequences: training sequences visited over the 3,000 updates. Relative completion loss of the inserted sequences: its minimum, its value when the sequence leaves the batch (J updates after insertion, one update for MDM), and its value 300 updates later. Validation completion loss: mean over the last 200 updates.
<table><tr><td rowspan="2"></td><td rowspan="2">J</td><td rowspan="2">Distinct sequences</td><td colspan="3">Relative completion loss</td><td rowspan="2">Validation completion loss</td></tr><tr><td>Minimum</td><td>At exit</td><td>300 later</td></tr><tr><td> $\mathrm { P U } , u { = } 2$ </td><td>231</td><td>3.6k</td><td>0.54</td><td>0.88</td><td>0.98</td><td>6.92</td></tr><tr><td> $\mathrm { P U } , u { = } 4$ </td><td>115</td><td>6.9k</td><td>0.70</td><td>0.71</td><td>0.98</td><td>6.58</td></tr><tr><td> $\mathrm { P U } , u { = } 8$ </td><td>57</td><td>13.7k</td><td>0.75</td><td>0.75</td><td>0.98</td><td>5.72</td></tr><tr><td> $\mathrm { P U } , u { = } 6 4$ </td><td>7</td><td>110k</td><td>0.97</td><td>0.99</td><td>1.00</td><td>4.37</td></tr><tr><td> $\mathrm { P U } , u { = } 1 2 8$ </td><td>4</td><td>192k</td><td>0.98</td><td>1.00</td><td>1.03</td><td>4.36</td></tr><tr><td>MDM</td><td></td><td>768k</td><td>0.98</td><td>0.98</td><td>1.00</td><td>4.72</td></tr></table>

![](images/52c283b492494672136e831fdc4da4f35b2ea5c1b076050018e71a9832f7d3b0.jpg)

![](images/5cc2902ce02b53f39023bb69bd6656c7bd694198f51613e946ba1a1622c0a66d.jpg)  
Figure 13 Complements to Figure 2. Left: validation completion loss at the end of training. PU beats MDM when each sequence stays in the batch for only a few steps, but degrades as u shrinks, and training on the PU schedule with MDM’s random masks does not help. Right: loss of the trajectory of an inserted sequence on the answer tokens it still masks, while the sequence is in the batch. At u=2, the trajectory fits its answer within a quarter of its stay.

• Spread visits. Each GPU keeps a pool of 3,000 32/J sequences, and each update advances the next 32 of them along their trajectories, so that a sequence is visited once every 3,000/J updates, about 13 at u=2. The trajectories, the number of visits per sequence and the number of distinct sequences are the same as with PU, but consecutive batches share no sequence. The validation loss is higher than with PU at every u: the model memorizes the whole pool over the run, with a training loss of 0.03 at u=2, instead of one batch at a time (Figure 14, right). Reuse is therefore what costs, and the consecutive visits of PU only make the overfitting local.

• Initial stages 0 to 31. Instead of drawing the initial stages at random, the 32 trajectories of a GPU start at stages 0 to 31. Since every trajectory lasts exactly J updates, the whole pool then moves along the trajectory as a single window of 32 consecutive stages, and all losses oscillate with period J. The validation loss is much higher at $u \leq 4 ;$ at $u { = } 8 ,$ where the window covers 32 of the 57 stages, it is unchanged. This is the noise-level concentration of Section 4.3, which here lasts for the whole run because no trajectory skips a stage.

• No 1/t factor. Every masked token gets the same weight, instead of the per-sequence normalization of Eq. (2.1). This deepens the drop at $u { = } 2 ,$ , to a minimum relative completion loss of 0.36, but changes the validation loss by at most 0.16 at every u, including $u \in \{ 6 4 , 1 2 8 \}$ and MDM.

• Batch of 128 sequences. Halving the batch makes each sequence a larger share of each update and halves the number of distinct sequences. The drop deepens, to 0.37 at u=2, and the validation loss is higher at every u.

![](images/c8b7c4d1bb16a30502189ee3422fb3f591b3a4dc4d6a8fd19b3ac552e155e0d9.jpg)  
Figure 14 Inserted sequences under two ablations of PU, at $u { = } 2 .$ . Each colored curve is the mean completion loss of the four sequences inserted at the same update, smoothed over 9 updates. With MDM’s masks, an inserted sequence is almost memorized during its stay and forgotten after it; with spread visits, the model memorizes the inserted sequences over the rest of the run while the validation loss rises

Table 10 Validation completion loss when one aspect of PU is changed at a time (MDM: 4.72).
<table><tr><td></td><td>u=2</td><td>u=4</td><td>u=8</td></tr><tr><td>PU</td><td>6.92</td><td>6.58</td><td>5.72</td></tr><tr><td>PU schedule, MDM masks</td><td>7.61</td><td>6.67</td><td>6.23</td></tr><tr><td>Spread visits</td><td>9.70</td><td>7.18</td><td>6.03</td></tr><tr><td>Initial stages 0 to 31</td><td>9.71</td><td>7.64</td><td>5.73</td></tr><tr><td>No 1/t factor</td><td>7.08</td><td>6.44</td><td>5.81</td></tr><tr><td>Batch of 128 sequences</td><td>8.22</td><td>7.12</td><td>6.24</td></tr></table>

## D.3 Why the loss rises again before the sequence leaves the batch

At $u { = } 2 ,$ the relative completion loss of an inserted sequence bottoms out about 50 updates after insertion, and three quarters of the drop are gone when the sequence leaves the batch, 231 updates after insertion (Figure 2, middle). At u=4 and u=8, it keeps falling until the sequence leaves (Table 9). Diagnostics logged in further runs at the same settings explain this diference.

The trajectory fits its answer early. While an inserted sequence is in the batch, we log the loss of its trajectory on the answer tokens that are still masked, which is what training minimizes on this sequence, restricted to its answer (Figure 13, right). At u=2, this loss falls below 0.1 nats 55 updates after insertion, while 95% of the answer is revealed only after 137 updates. From then on, the sequence contributes almost no gradient, and the updates on the other sequences of the batch erase what it had taught the model. At u=4, the trajectory fits its answer only 112 updates into its 115-update stay, and at u=8 it does not fit it at all (1.7 nats when the sequence leaves), so the completion loss keeps falling until the sequence leaves and rises only afterwards. Revealing positions in a random order instead of by confidence still fits the trajectory early, 26 updates after insertion at u=2, and keeps a smaller rebound, so the reveal order is not its cause.

A revealed token is no longer maintained. In two further runs at u=2 with 24 inserted sequences, we record the stage at which each position is revealed, and follow the completion loss at each position of the second half of the answer, whose tokens are the targets of the completion loss. Grouped by the stage at which they are revealed, the tokens of each group start degrading at their own reveal (Figure 15, left). Under confidence order, the 1,179 tokens revealed between stages 60 and 170 keep improving while they are still masked, by 0.62 0.11 nats between the bottom of the curve and their reveal, and their loss rises by $1 . 1 0 \pm 0 . 0 7$ nats in the 50 updates after their reveal (Figure 15, right). Under random order, masked and revealed tokens both degrade by about 0.45 nats per 50 updates once the trajectory has fitted its answer, and the reveal does not change this rate. A token’s prediction is therefore maintained only while the token remains a target that the trajectory has not fitted yet. Under confidence order, the tokens that remain masked are the ones the model is least confident about, so they keep receiving gradient while the revealed ones are forgotten.

![](images/c238deb103a8bdc5650f1a7a7ec2d795686f586a2d4fe296f65ed4c55cc9eee0.jpg)

![](images/bccd3dcfd11b01f3d5e9b6b8aa68638f7511a1c1e25fa34da09b740d3f80b20e.jpg)  
Figure 15 A token is forgotten once its trajectory reveals it (u=2). Tokens: the second-half answer tokens of the inserted sequences, the targets of the completion loss. Left: tokens grouped into quartiles of the stage at which they are revealed; change in their completion loss relative to before insertion. Right: tokens revealed at stages 60 to 170; change relative to the 10 updates before their reveal (mean ± one standard error). Under confidence order, each group degrades from its own reveal on; under random order, the reveal makes no diference, and the four groups of the left panel (not shown) coincide before insertion and do not turn up at their reveal.

## E Theory for the carry and the BPTT window

Setting. We use the notation of Sections 2 and 3. The prompt is fixed and omitted from the notation. Let Ω be the set of response positions. For a set of positions $A ,$ we write $\mathbf { x } ^ { A } = ( x ^ { i } ) _ { i \in A }$ . Let $p ^ { \star }$ be the data distribution of the clean response $\mathbf { x } _ { 0 } ^ { \Omega }$ . The carry-augmented denoiser $\mathrm { ( E q . ~ ( 3 . 1 ) ) }$ returns a distribution per position and a new carry:

$$
f _ { \mathrm { c a r r y } } ( \mathbf { x } , h ) = \big ( f _ { \boldsymbol { \theta } } ( \cdot \mid \mathbf { x } , h ) , h ^ { \prime } \big ) .
$$

The sampler starts from $\mathbf { x } _ { t _ { 0 } }$ , with the response fully masked, and $h _ { 0 } = 0$ . At each step $j ,$ it proceeds as follows.

1. It runs the denoiser on the current sequence and carry:

$$
\big ( f _ { \boldsymbol { \theta } } ( \cdot  { | } \mathbf x _ { t _ { j } } , h _ { j } ) , h _ { j + 1 } \big ) = f _ { \mathrm { c a r r y } } ( \mathbf x _ { t _ { j } } , h _ { j } ) .
$$

2. The unmasking policy selects $S _ { j } = g ( \mathbf { x } _ { t _ { j } } , h _ { j } )$ , the u most confident positions of $\mathcal { M } ( \mathbf { x } _ { t _ { j } } )$ (or all remaining ones, if fewer), breaking ties by index.

3. It fills each selected position independently:

$$
x ^ { i } \sim f _ { \theta } ^ { i } ( \cdot \mid \mathbf { x } _ { t _ { j } } , h _ { j } ) , \qquad i \in S _ { j } .
$$

We write $A _ { j } = \Omega \backslash \mathcal { M } ( \mathbf { x } _ { t _ { j } } )$ for the response positions revealed before step $j .$ The sampler stops after $J$ steps, when no masked position remains, and we denote by $p _ { \theta }$ the resulting distribution over responses. Training follows the same policy $g$ but commits ground-truth tokens (Algorithm 1). We write $\mathcal { L } ^ { \left( j \right) }$ for the loss at step $j ,$ and the total loss is

$$
\mathcal { L } _ { \infty } = \sum _ { j < J } \mathcal { L } ^ { ( j ) } .
$$

Assumption 1 (Denoiser). The denoiser never predicts the mask token and puts positive mass on every other token: for all θ, x, h and $i ,$

$$
f _ { \theta } ^ { i } ( m \mid \mathbf { x } , h ) = 0 , \qquad f _ { \theta } ^ { i } ( v \mid \mathbf { x } , h ) > 0 \quad { \mathrm { f o r ~ } } v \in \mathcal { V } \setminus \{ m \} .
$$

Moreover, for every fixed input sequence x, the map $( \theta , h ) \mapsto f _ { \mathrm { c a r r y } } ( \mathbf { x } , h )$ is real-analytic.

The first part holds for a softmax restricted to $\nu \backslash \{ m \}$ , i.e., with the mask logit set to $- \infty$ . The second part holds for transformers built from analytic operations, such as attention, softmax, SiLU or GELU activations, and LayerNorm or RMSNorm with $\epsilon > 0$ , but not for ReLU activations. Since the data contain no mask token, $p ^ { \star }$ is supported on $( \mathcal { V } \setminus \{ m \} ) ^ { \Omega }$ , which is finite.

Regular parameters. Since $S _ { j }$ depends on θ through the confidences, $\mathcal { L } _ { \infty }$ can jump where a selection changes, and it is not diferentiable there. We call θ regular if, for every $\mathbf { x } _ { \mathrm { 0 } }$ in the support of $p ^ { \star }$ and every step $j$ of its teacher-forced trajectory with $| \mathcal { M } ( \mathbf { x } _ { t _ { j } } ) | > u$ , the u-th largest confidence in $\mathcal { M } ( \mathbf { x } _ { t _ { j } } )$ is strictly larger than the $( u { + } 1 ) { - } \mathrm { t h }$ . The confidences are continuous in θ (Assumption 1) and the support of $p ^ { \star }$ is finite, so, by induction on $j ,$ every set $S _ { j }$ is constant in a neighborhood of a regular θ. In that neighborhood, $\mathcal { L } _ { \infty }$ is a finite sum of analytic functions, and $\nabla { \mathcal { L } } _ { \infty } ( \theta )$ is the gradient obtained by backpropagating through all carries with the sets $S _ { j }$ held fixed.

Assumption 2 (Non-degenerate selection). The set of non-regular θ has Lebesgue measure zero.

This holds, for instance, if no two token probabilities are identical as functions of θ: by Assumption 1, each of them is real-analytic in $\theta ,$ and the zero set of a real-analytic function that is not identically zero has Lebesgue measure zero (Mityagin, 2020).

For simplicity, we assume that every trajectory has J steps and that W divides J. Training splits the trajectory into consecutive windows of $W$ steps and detaches the carry at each window boundary (Algorithm 1). At a regular $\theta ,$ we denote by $g _ { W } ( \theta )$ the resulting truncated gradient, with the sets $S _ { j }$ held fixed; in particular, $g _ { J } = \nabla \mathcal { L } _ { \infty }$

Proposition 2 (Likelihood identity). Define the teacher-forced loss on revealed positions,

$$
\widetilde { \mathcal { L } } ( \boldsymbol { \theta } ) = \mathbb { E } _ { \mathbf { x } _ { 0 } \sim p ^ { \star } } \biggl [ \sum _ { j < J } \sum _ { i \in S _ { j } } - \log f _ { \boldsymbol { \theta } } ^ { i } \left( x _ { 0 } ^ { i } \mid \mathbf { x } _ { t _ { j } } , h _ { j } \right) \biggr ] ,
$$

where the trajectory is built from $\mathbf { x } _ { \mathrm { 0 } }$ by teacher forcing. Then

$$
\widetilde { \mathcal { L } } ( \boldsymbol { \theta } ) = H ( p ^ { \star } ) + \mathrm { K L } \big ( p ^ { \star } \| p _ { \boldsymbol { \theta } } \big ) .
$$

Proof. By induction on $j ,$ the pair $( \mathbf { x } _ { t _ { j } } , h _ { j } )$ and the set $S _ { j }$ are deterministic functions of the revealed values $\mathbf { x } ^ { A _ { j } }$ . Indeed, the initial state depends only on the prompt, and the state at step $j { + 1 }$ is computed from the state at step $j$ and the values written at $S _ { j }$ . Hence each response $\mathbf { x } ^ { \Omega }$ is produced by exactly one sampling path, namely the one that selects the sets $S _ { 0 } , S _ { 1 } , . . .$ . computed from x. Since the draws within a step are independent,

$$
p _ { \theta } ( \mathbf { x } ^ { \Omega } ) = \prod _ { j < J } \prod _ { i \in S _ { j } } f _ { \theta } ^ { i } \big ( x ^ { i } \mid \mathbf { x } _ { t _ { j } } , h _ { j } \big ) .
$$

By Assumption 1, the sampler never writes the mask token, so $p _ { \theta }$ is a probability distribution on $( \mathcal { V } \backslash \{ m \} ) ^ { \Omega }$ which contains the support of $p ^ { \star }$ . Taking log and the expectation under $p ^ { \star }$ gives the claim. All terms are finite because $f _ { \theta } ^ { i } ( v \mid \cdot ) > 0$ for every v $\neq$ m (Assumption 1). □

Assumption 3 (Contractive carry). There exist constants $\gamma < 1 , C _ { h }$ and $C _ { L }$ such that, for all $\theta ,$ every x<sub>0</sub> in the support of $p ^ { \star }$ and every step $j$ of its teacher-forced trajectory,

$$
\bigg \| \frac { \partial h _ { j + 1 } } { \partial h _ { j } } \bigg \| \leq \gamma , \qquad \bigg \| \frac { \partial h _ { j + 1 } } { \partial \theta } \bigg \| \leq C _ { h } , \qquad \bigg \| \frac { \partial \mathcal { L } ^ { ( j ) } } { \partial h _ { j } } \bigg \| \leq C _ { L } .
$$

Proposition 3 (Truncation bias). Under Assumptions 1 and ${ \mathcal { B } } ,$ for every regular $\theta ,$

$$
\left\| g _ { W } ( \theta ) - \nabla \mathcal { L } _ { \infty } ( \theta ) \right\| \le B \Big ( \frac { J } { W } - 1 \Big ) , \qquad B = \frac { C _ { L } C _ { h } } { ( 1 - \gamma ) ^ { 2 } } .
$$

Proof. Since $\theta$ is regular, both $g _ { W } ( \theta )$ and $\nabla { \mathcal { L } } _ { \infty } ( \theta )$ are computed with the sets $S _ { j }$ held fixed, and they difer only in the carry paths that cross a window boundary. Unrolling the carry, the gradient of $\mathcal { L } ^ { \left( j \right) }$ contains, for each lag $1 \leq r \leq j$ , the term

$$
\frac { \partial \mathcal { L } ^ { ( j ) } } { \partial h _ { j } } \Big ( \prod _ { s = 1 } ^ { r - 1 } \frac { \partial h _ { j - s + 1 } } { \partial h _ { j - s } } \Big ) \frac { \partial h _ { j - r + 1 } } { \partial \theta } .
$$

By Assumption $^ { 3 , }$ its norm is at most $C _ { L } C _ { h } \gamma ^ { r - 1 }$ . Let $p \in \{ 0 , \ldots , W - 1 \}$ be the position of step $j$ in its window. The truncated gradient keeps exactly the lags $r \leq p ,$ so the dropped terms of step $j$ sum to at most

$$
\sum _ { r > p } C _ { L } C _ { h } \gamma ^ { r - 1 } = \frac { C _ { L } C _ { h } } { 1 - \gamma } \gamma ^ { p } .
$$

In the first window, $p = j$ and nothing is dropped. Each of the remaining $J / W - 1$ windows contributes at most

$$
\frac { C _ { L } C _ { h } } { 1 - \gamma } \sum _ { p = 0 } ^ { W - 1 } \gamma ^ { p } \ \leq \ \frac { C _ { L } C _ { h } } { ( 1 - \gamma ) ^ { 2 } } = B .
$$

Definition 2 (Polyak–Łojasiewicz inequality). Let F be a function with infimum $F ^ { \star } = \operatorname* { i n f } _ { \theta } F ( \theta )$ . It satisfies the Polyak–Łojasiewicz inequality at θ with constant $\mu > 0$ if F is diferentiable at θ and

$$
\begin{array} { r } { \frac { 1 } { 2 } \| \nabla F ( \theta ) \| ^ { 2 } \geq \mu \left( F ( \theta ) - F ^ { \star } \right) . } \end{array}
$$

It satisfies the Polyak–Łojasiewicz inequality (Karimi et al., 2016) if this holds at every $\theta .$

The PL inequality does not require convexity. When it holds everywhere, the gradient is small only near the minimum value, and every stationary point is a global minimizer. Since $\mathcal { L } _ { \infty }$ jumps where the selected sets change, it cannot satisfy the inequality everywhere in general, and Theorem 1 only requires it at the point of interest.

Theorem 1 (Window and approximation). Assume Assumptions 1 and 3 and $\begin{array} { r } { \mathcal { L } _ { \infty } = \widetilde { \mathcal { L } } _ { \ast } } \end{array}$ . Let $\theta _ { W }$ be a regular point with $g _ { W } ( \theta _ { W } ) = 0$ , at which $\mathcal { L } _ { \infty }$ satisfies the $P L$ inequality (Definition 2) with constant $\mu .$ Then

$$
\mathrm { K L } \big ( p ^ { \star } \| p _ { \theta _ { W } } \big ) - \operatorname* { i n f } _ { \theta } \mathrm { K L } \big ( p ^ { \star } \| p _ { \theta } \big ) \ \leq \ \frac { B ^ { 2 } } { 2 \mu } \Big ( \frac { J } { W } - 1 \Big ) ^ { 2 } .
$$

The bound decreases as W grows and vanishes at $W = J$

Proof. Since $\theta _ { W }$ is regular and $g _ { W } ( \theta _ { W } ) = 0$ , Proposition 3 gives

$$
\| \nabla \mathcal { L } _ { \infty } ( \theta _ { W } ) \| \le B \Big ( \frac { J } { W } - 1 \Big ) .
$$

The PL inequality at $\theta _ { W }$ then yields

$$
\mathcal { L } _ { \infty } ( \theta _ { W } ) - \operatorname* { i n f } _ { \theta } \mathcal { L } _ { \infty } \ \leq \ \frac { \| \nabla \mathcal { L } _ { \infty } ( \theta _ { W } ) \| ^ { 2 } } { 2 \mu } \ \leq \ \frac { B ^ { 2 } } { 2 \mu } \Big ( \frac { J } { W } - 1 \Big ) ^ { 2 } .
$$

By Proposition $2 , { \mathcal { L } } _ { \infty }$ and the KL difer only by the constant $H ( p ^ { \star } )$ , which gives the claim.

Corollary 1 (Comparison with an autoregressive model). Let q be a distribution over responses, for instance that of an autoregressive model, with error $\delta = \mathrm { K L } ( p ^ { \star } \| q ) > 0$ , and let $\begin{array} { r } { \varepsilon ^ { \star } = \operatorname* { i n f } _ { \theta } \mathrm { K L } ( p ^ { \star } \| p _ { \theta } ) } \end{array}$ . Assume that, for every window W dividing J, the hypotheses of Theorem 1 hold at $\theta _ { W }$ with the same constant µ. If $\varepsilon ^ { \star } < \delta$ then $\mathrm { K L } ( p ^ { \star } \| p _ { \theta _ { W } } ) < \delta$ for every such W with

$$
W > \frac { J } { 1 + \sqrt { 2 \mu ( \delta - \varepsilon ^ { \star } ) } / B } ,
$$

and $W = J$ always qualifies. In particular, under the zero-infimum condition $\varepsilon ^ { \star } = 0$ , a large enough window beats q for every $\delta > 0$

Proof. By Theorem 1, $\begin{array} { r } { \mathrm { K L } ( p ^ { \star } \| p _ { \theta _ { W } } ) \leq \varepsilon ^ { \star } + \frac { B ^ { 2 } } { 2 \mu } \big ( \frac { J } { W } - 1 \big ) ^ { 2 } } \end{array}$ . The condition on W is equivalent to $\frac { J } { W } - 1 <$ $\sqrt { 2 \mu ( \delta - \varepsilon ^ { \star } ) } / B$ , so the second term is below $\delta - \varepsilon ^ { \star }$ . Since the right-hand side of the condition is smaller than $J , W = J$ satisfies it. □

The zero-infimum condition requires the carry-augmented sampler to approximate $p ^ { \star }$ arbitrarily well under the reveal rule g. For $u = 1$ , the chain rule makes this a capacity assumption. For $u > 1$ , each step draws its u tokens independently given the state, so $p _ { \theta }$ factorizes within each step and $\varepsilon ^ { \star }$ is generally positive; the corollary then applies only to AR models with $\delta > \varepsilon ^ { \star }$

Scope of the bound. Both parts assume that training reveals positions as inference does. Part (i) holds along any teacher-forced trajectory, so it does not depend on the training u. Part (ii), however, bounds the sampler that decodes with the training reveal rule, including its u and its number of steps J. Local overfitting forces us to train at a much larger u than we decode with (efective training $u \approx 1 3 3$ to 67 on TinyGSM, against $u { = } 2$ at inference). So the bound does not directly cover the sampler we evaluate, which takes more steps and visits diferent states. We read it as a guarantee for an idealized, aligned setting. Whether the benefit carries over to the smaller inference u is an empirical question: Figure 3 addresses it for masking patterns, and Table 2 for accuracy.

Theorem 2 (Credit assignment). Assume Assumption 1. Fix $\mathbf { x } _ { \mathrm { 0 } }$ in the support of $p ^ { \star }$ and its teacher-forced trajectory. Let step j produce the carry $h _ { j + 1 ; }$ , and consider the loss of step $j + r , r \geq 1$ steps later.

(i) If the only state passed between steps is the committed tokens (no carry), then at every regular $\theta ,$ and hence under Assumption 2 for almost every $\theta ,$ the total derivative $\mathrm { d } \dot { \mathcal { L } } ^ { ( j + r ) } / \mathrm { d } \theta$ equals the partial derivative with the trajectory held fixed.

(ii) With the carry, let θ be regular and let $\sigma = ( S _ { 0 } , \ldots , S _ { j + r - 1 } )$ be the sets it selects. If steps j and $j + r$ lie in the same window, the truncated gradient of $\mathcal { L } ^ { ( j + r ) }$ contains the term

$$
G _ { j , r } ^ { \sigma } ( \theta ) = \frac { \partial \mathcal { L } ^ { ( j + r ) } } { \partial h _ { j + r } } \Big ( \prod _ { s = 1 } ^ { r - 1 } \frac { \partial h _ { j + r - s + 1 } } { \partial h _ { j + r - s } } \Big ) \frac { \partial h _ { j + 1 } } { \partial \theta } ,
$$

where the product is ordered from $s = 1$ on the ${ \it l e f t }$ to $s = r - 1$ on the $r i g h t ,$ and all factors are evaluated along the trajectory defined by σ. As a function of θ with σ fixed, $G _ { j , r } ^ { \sigma }$ is real-analytic on the whole parameter space. Hence $G _ { i , r } ^ { \sigma }$ is either identically zero or nonzero for almost every $\theta ,$ and the second case holds as soon as $G _ { j , r } ^ { \sigma } ( \check { \theta } ^ { \prime } ) \neq 0$ at a single $\theta ^ { \prime }$ . If steps j and $j + r$ lie in diferent windows, this term is absent. In particular, the loss of a step reaches only earlier steps of its own window, so $r \leq W - 1$

Proof. (i) Under teacher forcing, the committed tokens are the ground-truth values $x _ { 0 } ^ { i }$ , which do not depend on θ. Without a carry, the input $\mathbf { x } _ { t _ { j + } }$ therefore depends on θ only through the sets $S _ { 0 } , \ldots , S _ { j + r - 1 }$ . At a regular θ, these sets are constant in a neighborhood of $\theta$ (see the paragraph on regular parameters). The input $\mathbf { x } _ { t _ { j + r } }$ is thus locally constant, and $\bar { \mathrm { d } } \mathcal { L } ^ { ( j + r ) } / \mathrm { d } \theta$ equals the partial derivative with the trajectory held fixed. Under Assumption 2, regular points form a set of full measure.

(ii) The sets σ are constant near $\theta ,$ so near θ the carries $h _ { j + 1 } , \dotsc , h _ { j + r }$ and the loss $\mathcal { L } ^ { ( j + r ) }$ are computed along the fixed trajectory defined by σ. Within a window, the carry is never detached, and the chain rule along the path

$$
\theta  h _ { j + 1 }  \cdot \cdot \cdot  h _ { j + r }  \mathcal { L } ^ { ( j + r ) }
$$

gives $G _ { j , r } ^ { \sigma }$ . With σ fixed, each carry is a composition of the maps $( \theta , h ) \mapsto f _ { \mathrm { c a r r y } } ( \mathbf { x } _ { t _ { k } } , h )$ , which are realanalytic by Assumption 1. The loss is analytic in the predicted probabilities, since they are positive at the target tokens. Partial derivatives and products of real-analytic functions are real-analytic, so $G _ { j , r } ^ { \sigma }$ is real-analytic on the whole parameter space. The dichotomy follows from the fact that the zero set of a realanalytic function that is not identically zero has Lebesgue measure zero (Mityagin, 2020). At initialization, the zero-initialized LayerNorm maps every carry to zero, so $\partial \mathcal { L } ^ { ( j + r ) } / \partial h _ { j + r } = 0$ and $G _ { j , i } ^ { \sigma }$ vanishes; the same holds at every θ where this LayerNorm has zero gain and bias. Initialization alone therefore does not tell which of the two cases holds. If the path crosses a window boundary, the carry is detached there, and the term is absent. □

Theorem 1 assumes that training uses the inference policy g and applies the loss only to the positions each step reveals. Our TinyGSM training uses the stage-indexed schedule of Kim et al. (2026) and applies the loss to all masked positions with the weighting of Eq. (2.1), so the theorem describes an idealized version of it. In particular, $p _ { \theta }$ in Proposition 2 and Theorem 1 is the sampler that decodes with the training reveal rule. Whenever inference uses a smaller $u ,$ as in all our experiments, the bound applies to that sampler and not to the one we evaluate. The theorem also treats $g _ { W }$ as the sum of the window gradients at a single $\theta ,$ whereas Algorithm 1 divides each window’s loss by its length and updates $\theta$ after every window, so it characterizes the fixed points of the idealized full-trajectory update rather than the iterates of Algorithm 1. The theorem bounds the excess KL of the sampler at points where the truncated gradient vanishes, and this bound decreases with $W ;$ it says nothing directly about the accuracy of the trained model. Theorem 2 concerns exact gradients. Straight-through, relaxed and score-function estimators provide surrogate signals, which is why they can help but do not match the carry (Section 3.2). Finally, with teacher-forced commits, the gradient through the carry trains each step to pass useful information forward, not to correct its own committed errors.

## F Additional results on LLaDA

## F.1 Full-canvas SFT schedule

All full-canvas Post-SFT runs start from an SFT checkpoint trained for 1.5 epochs (25k steps) with a cosine schedule. We choose this length because SFT has largely saturated by then: doubling the schedule to 3 epochs (50k steps) raises the score at τ = 1 by 0.7 points, from 60.9 to 61.6 (Figure 16 and Table 11). At this length, we select the peak learning rate from $\{ 1 0 ^ { - 8 } , 1 0 ^ { - 7 } , 1 0 ^ { - 6 } , 5 \times 1 0 ^ { - 6 } , 1 0 ^ { - 5 } , 1 0 ^ { - 4 } \}$ (Figure 17). The selection criterion is the score at $\tau = 1$ , which decodes one token per step, as difusion language models are usually benchmarked (Nie et al., 2025; Ye et al., 2025). The peak learning rate $5 \times 1 0 ^ { - 6 }$ scores highest (60.9, against 60.5 at $1 0 ^ { - 5 } )$ , and its checkpoint is the starting point of all full-canvas Post-SFT runs. The Post-SFT MDM control, which continues this checkpoint for 4000 steps, scores 61.2 against its 60.9; we attribute the small change to the short warm-up and re-annealing of the learning rate. All scores use the chat prompt format (Appendix B.4); Appendix F.8 compares them with published LLaDA results. IFEval is the benchmark that gains the most from SFT, from 23.3 to 68.0 at $\tau = 1$ , against 60.3 to 75.9 on GSM8K and 36.6 to 38.8 on MBPP (Table 13). This is consistent with the instruction-following focus of the Dolci-Instruct-SFT data.

![](images/9361115ddbade34da649ec73371f8bc0c885b1ec7846f327a3d5e57cb1a88869.jpg)  
Figure 16 Full-canvas SFT of LLaDA-8B-Base for 1.5 and 3 epochs, against SFT step. Columns: decode threshold τ. Rows: score, then each benchmark. The star marks LLaDA-8B-Base.

![](images/deff70ef71e961101922e0cc5838c8bdb9112f51297a561248292644e2a95dc7.jpg)  
Figure 17 Full-canvas SFT for 1.5 epochs, against peak learning rate. Columns: decode threshold τ. Rows: score, then each benchmark.

## F.2 Full-canvas BPTT

Table 11 summarizes the best score of every full-canvas Post-SFT configuration.

Figure 18 reports full-canvas BPTT for $u \in \{ 1 6 , 3 2 , 6 4 , 1 2 8 \}$ and $W \in \{ 1 , 2 , 4 , 8 \}$ , for the score and for each benchmark. Every carry configuration reaches a higher best score than the Post-SFT MDM control (61.2). The gain over PU without carry is not monotonic in W, but the best window improves on PU at every u. This improvement grows with u, from +0.1 at u = 16 to +1.4 at u = 128, mostly because PU alone weakens as u grows: at u = 128 it only matches the Post-SFT MDM control (Section 4.3). The best score overall is reached at u = 32 (Table 11). Below u = 16, PU without carry already falls under both baselines (59.4 at $u = 8$ and 47.6 at u = 4, Figure 29), a consequence of the small-u optimization issues of Section 4.3, so we do not extend BPTT there.

Table 11 Best score of each full-canvas Post-SFT variant, for every training u of the BPTT sweep, with the change relative to the departure SFT checkpoint in parentheses. Best and 2nd best highlighted.
<table><tr><td>Full canvas</td><td> $u { = } 1 6$ </td><td> $u { = } 3 2$ </td><td> $u { = } 6 4$ </td><td> $u { = } 1 2 8$ </td></tr><tr><td>PU</td><td> $6 2 . 5 \ : ( + 1 . 6 )$ </td><td> $6 2 . 8 \left( + 1 . 9 \right)$ </td><td> $6 1 . 9 \left( + 1 . 0 \right)$ </td><td> $6 1 . 2 \ : ( + 0 . 3 )$ </td></tr><tr><td> $\mathrm { P U } + \mathrm { C a r r y } , W { = } 1$ </td><td> $6 1 . 9 \ : ( + 1 . 0 ) $ </td><td> $6 2 . 6 \ : ( + 1 . 7 )$ </td><td> $6 2 . 4 \ : ( + 1 . 5 )$ </td><td> $6 1 . 7 \left( + 0 . 8 \right)$ </td></tr><tr><td> $W { = } 2$ </td><td> $6 1 . 8 \left( + 0 . 9 \right)$ </td><td> $6 1 . 4 \ : ( + 0 . 5 )$ </td><td> $6 2 . 5 \ : ( + 1 . 6 )$ </td><td> $6 2 . 0 \left( + 1 . 1 \right)$ </td></tr><tr><td> $W { = } 4$ </td><td> $6 1 . 9 \ : ( + 1 . 0 ) $ </td><td> $\underline { { 6 3 . 0 } } \left( + 2 . 1 \right)$ </td><td> $6 2 . 6 \ : ( + 1 . 7 )$ </td><td> $6 2 . 6 \ : ( + 1 . 7 )$ </td></tr><tr><td> $W { = } 8$ </td><td> $6 2 . 6 \ : _ { ( + 1 . 7 ) }$ </td><td> ${ \bf 6 3 . 5 _ { \left( + 2 . 6 \right) } }$ </td><td> $\underline { { 6 3 . 0 \left( + 2 . 1 \right) } }$ </td><td> $6 2 . 4 \ : ( + 1 . 5 )$ </td></tr><tr><td>SFT, 1.5 epochs (departure)</td><td colspan="4">60.9</td></tr><tr><td>SFT, 3 epochs</td><td colspan="4"> $6 1 . 6 \ : ( + 0 . 7 )$ </td></tr><tr><td>Post-SFT MDM</td><td colspan="4"> $6 1 . 2 \ : ( + 0 . 3 )$ </td></tr></table>

![](images/7626a1926d60f72929e606ce43c33b44b09b83c093a266b212d18b15af0582ec.jpg)  
Full canvas, 1.5 epochs Post-SFT MDM PU + Carry, W=1 PU + Carry, W=4 PU + Carry, W=8 Full canvas, 3 epochs Post-SFT PU PU + Carry, W=2  
Figure 18 Full-canvas Post-SFT with BPTT, against NFEs. Columns: train u. Rows: score, then each benchmark. Each curve sweeps the decode threshold $\tau \in \{ 0 . 7 , 0 . 8 , 0 . 9 , 1 \}$

## F.3 Block diffusion SFT

As on the full canvas, we select the learning rate for block difusion SFT at 1.5 epochs (25k steps), with the score at τ = 1 as criterion (Figure 19). The peak learning rate $5 \times 1 0 ^ { - 6 }$ again scores highest (57.8, against 57.4 at 10<sup>−5</sup>), and all block difusion SFT runs use it. Unlike the runs in Figure 20, these sweep runs do not train on the EOS padding that completes the last block of each response, so their scores difer slightly from those runs.<sup>2</sup>

Unlike the full canvas, block difusion SFT has not saturated at 1.5 epochs. The score at $\tau = 1$ rises from 57.2 at 1.5 epochs to 59.8 at 3 epochs and 61.4 at 6 epochs (Figure 20), against a 0.7-point gain from 1.5 to 3 epochs on the full canvas. We attribute the slower convergence to the mismatch between the block-wise attention pattern, causal across blocks and bidirectional within the current block, and the fully bidirectional attention of LLaDA’s pre-training. We therefore start the block difusion Post-SFT runs from the 6-epoch checkpoint (100k steps). As on the full canvas, IFEval gains the most, from 23.3 to 65.4 at 6 epochs, against 60.3 to 77.0 on GSM8K and 36.6 to 41.6 on MBPP.

Figure 21 compares the block difusion checkpoints at 1.5, 3, and 6 epochs with the full-canvas checkpoints at 1.5 and 3 epochs, as a function of NFEs. At 6 epochs, the block difusion checkpoint lies above both fullcanvas checkpoints over most of the NFE range: it reaches 61.1 at 153 NFEs, which the 3-epoch full-canvas checkpoint matches only at 204 NFEs. The exception is the 3-epoch full-canvas checkpoint at τ = 1, which scores 0.2 points higher (61.6) at 223 NFEs, against 196. We attribute this to block difusion SFT training with the same block-wise generation order as the block-32 decoding used at evaluation.

![](images/63342f5b0015eacee701e4dd3f674f283f5d66ce945d05634c6069c5025ba530.jpg)  
Figure 19 Block difusion SFT for 1.5 epochs, against peak learning rate. Columns: decode threshold τ . Rows: score, then each benchmark.

![](images/e54864bcb3739ccd77572de4debd9e8d46f57b66f7068592aedb2609cf13e688.jpg)  
Figure 20 Block difusion SFT of LLaDA-8B-Base for 1.5, 3, and 6 epochs, against SFT step. Columns: decode threshold τ. Rows: score, then each benchmark. The star marks LLaDA-8B-Base.

![](images/5c9c1495c734d11869f89e9043bdf9db6975fd31d225f511e3ad4d103444d235.jpg)  
Figure 21 Score against NFEs for the SFT checkpoints and LLaDA-8B-Base. Each curve sweeps the decode threshold $\tau \in \{ 0 . 7 , 0 . 8 , 0 . 9 , 1 \}$ . Left: full range. Right: fine-tuned checkpoints only.

## F.4 Block diffusion u sweep and BPTT

Figure 22 reports block difusion PU Post-SFT for $u \in \{ 1 , 2 , 4 , 8 , 1 6 \}$ , starting from the 6-epoch SFT checkpoint. Unlike the full canvas, no value of u fully collapses, although $u \in \{ 1 , 2 \}$ still trail the others in the score–NFE trade-of.

All blocks of a sample are unmasked in parallel, so its trajectory lasts at most $\lceil 3 2 / u \rceil$ steps. With the trainingtime threshold, which commits more than u tokens per step when the model is confident, a trajectory at $u = 1$ lasts 16.0 steps on average, against 160.2 on the full canvas.

Figure 23 reports block difusion BPTT for $u \in \{ 4 , 8 , 1 6 \}$ . The window cannot exceed the trajectory length, so $W \leq \lceil 3 2 / u \rceil \colon u = 4$ admits $W \in \{ 1 , 2 , 4 , 8 \} , u = 8$ admits $W \in \{ 1 , 2 , 4 \}$ , and $u = 1 6$ admits $W \in \{ 1 , 2 \}$ The best window improves on PU without carry by 1.0–1.5 points at every u, while the carry-only variant $( W = 1 )$ does not. Table 12 summarizes the best score of every block difusion Post-SFT configuration.

Table 12 Best score of each block difusion Post-SFT variant, for every training u of the BPTT sweep, with the change relative to the departure SFT checkpoint in parentheses. Best and 2nd best highlighted.
<table><tr><td>Block diffusion</td><td></td><td></td><td>u=16</td></tr><tr><td></td><td>u=4</td><td>u=8</td><td></td></tr><tr><td>PU  $\mathrm { P U } + \mathrm { C a r r y } , W { = } 1$ </td><td> $6 2 . 8 \ : ( + 1 . 4 )$ </td><td> $6 1 . 9 \ : ( + 0 . 5 )$ </td><td> $6 2 . 1 \left( + 0 . 8 \right)$   $6 2 . 1 \ : ( + 0 . 7 )$ </td></tr><tr><td> $W { = } 2$ </td><td> $6 1 . 1 \ : ( - 0 . 2 )$ </td><td> $6 2 . 0 \ : ( + 0 . 6 ) $ </td><td></td></tr><tr><td> $W { = } 4$ </td><td> ${ \bf 6 3 . 8 _ { \left( + 2 . 5 \right) } }$   $6 2 . 3 \ : ( + 0 . 9 )$ </td><td> $6 2 . 0 \left( + 0 . 6 \right)$   $6 2 . 9 \ : ( + 1 . 5 )$ </td><td> $\underline { { 6 3 . 6 } } \left( + 2 . 2 \right)$ </td></tr><tr><td> $W { = } 8$ </td><td> $6 2 . 1 \left( + 0 . 8 \right)$ </td><td></td><td></td></tr><tr><td></td><td colspan="3"></td></tr><tr><td>SFT, 6 epochs (departure)</td><td colspan="3">61.4</td></tr><tr><td>SFT, 3 epochs</td><td colspan="3"> $5 9 . 9 \left( - 1 . 4 \right)$ </td></tr><tr><td> $\mathrm { P o s t - S F T \ M D M }$ </td><td colspan="3"> $6 2 . 0 \ : _ { ( + 0 . 7 ) }$ </td></tr></table>

![](images/4fc1f78605e0c696921a9317a8430678490af31551e945ddacc333b1e497457f.jpg)  
Figure 22 Score against NFEs for block difusion PU Post-SFT across u. Each curve sweeps the decode threshold $\tau \in \{ 0 . 7 , 0 . 8 , 0 . 9 , 1 \}$

![](images/c984121ef12d9b9883eeae750c21cd3419ef1eba1ff74e93bba64b8e2bc73e8c.jpg)  
Block diffusion, 3 epochs Post-SFT MDM PU + Carry, W=1 PU + Carry, W=4 Block diffusion, 6 epochs Post-SFT PU PU + Carry, W=2 PU + Carry, W=8  
Figure 23 Block difusion Post-SFT with BPTT, against NFEs. Columns: u. Rows: score, then each benchmark. Each curve sweeps the decode threshold $\tau \in \{ 0 . 7 , 0 . 8 , 0 . 9 , 1 \}$

## F.5 The limit case of u equal to the block size in block diffusion

With a block size of 32, a training step at $u = 3 2$ commits the whole block, so every trajectory lasts one step. PU at $u = 3 2$ is therefore a limit case of PU: it only trains on fully masked blocks that follow a clean prefix. At inference, every block starts from such a sequence, except that the model generated the prefix. The later reveals within a block receive no Post-SFT training at $u = 3 2 .$ , so the model makes them with what it learned in pre-training and SFT.

PU at $u = 3 2$ reaches 63.1 at $\tau = 1$ , against 62.0 for Post-SFT MDM (Figure 24). Its score–NFE curve also lies above those of PU at every other u (Figure 22). We believe this gain comes from the short Post-SFT horizon of 4000 steps: the model improves its first reveal of each block, and pre-training and SFT still let it complete the block. The later reveals receive no training signal, so we expect a longer Post-SFT at $u = 3 2$ to collapse.

We cannot apply BPTT at $u = 3 2$ because the trajectory length is 1. Figure 24 compares PU at $u = 3 2$ with PU at $u = 1 6 .$ alone and with BPTT at $W = 2$ , the configurations reported in Section 4.2. We observe that the score–NFE trade-of of PU at $u = 1 6$ with BPTT at $W = 2$ dominates that of PU at $u = 3 2$ , although PU alone at $u = 1 6$ scores lower (62.1 against 63.1). The best reveal count without BPTT is therefore not necessarily the best one with it: a smaller u with BPTT can match or exceed a larger u that rules BPTT out. Our experiments do not isolate why $u { = } 3 2$ suits PU alone; the cause may lie in our setup (model, dataset, and benchmarks), in the optimization issues of small u (Section 4.3 and Appendix F.6), or in the decode block of 32.

![](images/54525fbda5f317fe1e51eedba07b6f0425b816f0199c65c569ca23840d8637d7.jpg)  
Figure 24 Score against NFEs for block difusion Post-SFT with PU at $u = 3 2 ,$ and with PU at $u = 1 6$ alone and with BPTT at $W = 2 ,$ against the SFT checkpoints and Post-SFT MDM. Each curve sweeps the decode threshold $\tau \in \{ 0 . 7 , 0 . 8 , 0 . 9 , 1 \}$

## F.6 Model collapse as a function of training u

In Section 4.3, we present the efect of decreasing the training u on the learning dynamics. Below, we provide more evidence of the challenges brought by small training u, which we trace to three factors: sample diversity, noise diversity, and the objective mismatch.

Sample diversity. PU-style training causes consecutive steps to visit similar sequences: a sequence not yet fully unmasked at step t reappears at step t + 1, one unmasking step further along its trajectory. At a fixed token budget, training therefore covers a smaller set of distinct sequences (Figure 5a) and keeps each of them in the batch for many consecutive steps, which is what makes the model overfit locally (Section 4.3). Figure 25 measures how long sequences stay in the batch, by the Jaccard similarity—the intersection over union—between the sequences visited by two steps $\Delta$ apart. We estimate it from the dynamics observed on a single machine in a multi-GPU run and extrapolate to the full run. Small training u yields substantially higher overlap that persists over large $\Delta .$ , whereas larger u shows a faster decay as $\Delta$ increases. Confidence collapse dampens the efect: by committing several high-confidence positions at once, it allows sequences to complete in fewer steps and be replaced sooner.

![](images/0c3d1e302575c5402e72bb5ab7e723d2134061ac425b7b7dc1358bc72d1edc2b.jpg)  
∆ = 1 ∆ = 10 ∆ = 100 MDM  
Figure 25 Intersection over union of samples within the training batch $\Delta$ training steps apart across training u: (a) PU training without confidence collapse; (b) PU training with confidence collapse; (c) PU training without confidence collapse when sampling batches from a pool of training trajectories 10× the size of the batch.

Noise diversity. PU-style training iterates through sequences by progressively unmasking them, starting from fully masked sequences. Early in training the number of visible tokens across a batch is therefore concentrated near 0. MDMs are not subject to this efect, as noise levels are drawn i.i.d. per sample at every training step. Figure 26 illustrates this behavior. We track the standard deviation, across training batches logged every 10 steps, of the number of unmasking steps each sequence has undergone. For all training u, the standard deviation starts near zero as all the sequences are fully masked and grows as sequences complete and are renewed at diferent times. What changes between u is the pace at which this mixing happens: for u=1, the progression toward reaching a steady state is longer than for u=32 and u=128, for which it can happen at a pace the logging frequency cannot track. While this mixing is slower at low u, it afects only the first few hundred training steps, so we expect its impact on the general performance to diminish as the number of training steps increases. Two further factors shape this mixing. Sequences difer in length and are completed after diferent numbers of steps, so the within-batch noise-level distribution broadens as training progresses. In block difusion, the parallel-block implementation unmasks u tokens per block, so a small training u corresponds to a higher efective number of tokens unmasked per step across the sequence.

Objective mismatch. PU-style training computes the loss over all masked tokens, akin to MDMs. The weight given to uncommitted positions is therefore not set directly but follows from u: as u decreases, the ratio of supervised to committed positions grows, and the gradient is increasingly dominated by predictions at positions the step does not commit. A single hyperparameter thus governs both the granularity of the trajectory and the balance of supervision between committed and uncommitted positions. Interestingly, Xia et al. (2026) propose an alternative loss which decouples the two, defining the loss as a weighted sum of the term over all masked tokens and the term over committed tokens alone.

## F.6.1 Mitigating the gap at small train u

We attempt to mitigate the efect of training with a small u by targeting the three factors identified above: sample diversity, noise diversity, and the objective mismatch.

Sample diversity. We maintain a pool of P training sequences, ten times larger than the training batch size B. At each training step s, rather than advancing the sequences seen at s 1, we advance the next B sequences in the pool. Consecutive steps therefore visit non-overlapping sets of sequences (Figure 25 c) and the number of distinct sequences seen over training increases. This comes at a cost in memory: the pool holds P sequences and their masks rather than B, and the overhead grows further with carry and BPTT, which store additional tensors per sequence. In practice, this bounds the pool size we can reasonably use.

![](images/b775b12abac87f5c6a3ab597eb34b69d0543fbcd28c90dd3a7da744ba20a30cd.jpg)  
Figure 26 Standard deviation of the number of unmasking steps within a batch across training, for PU-style training (top) and PU-style training with randomly initialized masking ratios (bottom), at u = 1 (left), u = 32 (middle) and u = 128 (right). All runs use full-canvas difusion without confidence collapse. The dotted line refers to the mean computed on the last 3,000 training steps.

Noise diversity. We initialize training from partially unmasked sequences, with masking ratios sampled uniformly. Figure 26 shows the efect on early training at u=1: PU with random initial masking ratios maintains a higher within-batch standard deviation than PU from the first step onward.

Objective mismatch. Finally, we explicitly balance the contributions of the tokens to be unmasked at the current step against those of the remaining masked tokens. Xia et al. (2026) propose a weighted sum of two cross-entropy terms, one over all masked tokens and one over the tokens revealed at the current step:

$$
\mathcal { L } _ { \mathrm { M S } } ^ { k } = w _ { k } \bigg [ \lambda \ell _ { k } ^ { \mathrm { d e n s e } } + \left( 1 - \lambda \right) \ell _ { k } ^ { \mathrm { r e v e a l } } \bigg ] , \qquad w _ { k } = \frac { n _ { k } } { N _ { m } } ,\tag{F.1}
$$

where $n _ { k }$ is the number of positions revealed at step $k , N _ { m }$ the total number of maskable positions, and $\ell _ { k } ^ { \mathrm { d e n s e } }$ and $\ell _ { k } ^ { \mathrm { r e v e a l } }$ the cross-entropy losses averaged over all masked tokens and over the tokens revealed at step k, respectively. We adapt Eq. (F.1) as

$$
\begin{array} { r } { \mathcal { L } _ { P U + } ^ { k } = \Big [ \lambda \ell _ { k } ^ { \mathrm { r e v e a l } } + \left( 1 - \lambda \right) \ell _ { k } ^ { \mathrm { d i f f e r } } \Big ] , } \end{array}\tag{F.2}
$$

where $\ell _ { k } ^ { \mathrm { d i f f e r } }$ is averaged over the tokens that remain masked after step k. We do not ablate λ thoroughly, but instead ofer a proof of concept: we set $\lambda = 3 2 / L$ , where L is the total number of masked positions per sequence, matching the relative weight that the best-performing PU run (u=32) assigns to revealed versus still-masked tokens.

In Figure 27, we apply these changes to PU-style full-canvas training, where the top-u tokens are unmasked at each step; at inference we use the Fast-dLLM policy and vary the confidence threshold τ . Together, these changes substantially reduce the performance gap at small u without closing it. Additionally, these tricks cost memory and tuning efort and only partially mitigate collapse; the simpler alternative remains to choose a training u that trades of training–inference alignment against healthy dynamics.

These three efects make it impractical to do PU at small u, while this is the regime typically targeted at inference (e.g., unmask 8 tokens at a time). Beyond memory and tuning, the adjustments above are also harder to integrate with complementary techniques such as the carry or BPTT.

Block diffusion reduces reuse. Block difusion allays this trade-of. With blocks of b tokens, all advancing in parallel, a sequence stays in the batch for at most $\lceil b / u \rceil$ steps instead of about $L / u ,$ so that at the same u it is reused $L / b$ times less (Section 4.2 and Appendix B.4.2). This is consistent with block difusion being less exposed to the loss of sample diversity (Figure 5a) and with no value of u fully collapsing in that setting (Appendix F.4).

![](images/2840c43e6f1c5222dde5863c055332c82589072f3b5cf8eb68f7f149be84f100.jpg)  
Figure 27 Mitigating model collapse in PU-style training at small training u: we compare PU, trained for full-canvas generation, top-u unmasking with u = 1, with PU+, which integrates tricks aiming at mitigating issues found in PU at small u. Inference here follows the Fast-dLLM policy with confidence threshold $\tau \in \{ 0 . 7 , 0 . 8 , 0 . 9 , 1 \}$

## F.7 Relevance of confidence thresholding at training time

All PU and BPTT runs outside this subsection use a training-time confidence threshold $\tau _ { \mathrm { t r a i n } } = 0 . 9$ , the default of Kim et al. (2026), who find PU mostly insensitive to the threshold except for very extreme values $( \mathrm { e . g . } , \tau \le 0 . 6 5 )$ . Each training step commits every masked position whose confidence exceeds $\tau _ { \mathrm { t r a i n } }$ and, if fewer than u qualify, the most confident remaining positions up to u (Section 4 and Appendix B.2). Here we compare it with a threshold-free variant that commits exactly u positions per step.

The threshold has two efects. First, it brings training trajectories closer to those of the confidence-threshold decoding used at inference. Second, it acts on the small-u optimization issues of Section 4.3: positions the model already predicts confidently are committed at once, so the model spends fewer steps on tokens it already knows and trajectories end sooner. As a result, a run visits more distinct samples (Figure 28). At $u = 1 ,$ , a run with the threshold visits 0.6% of the samples the Post-SFT MDM control visits, against 0.3% without. The diference shrinks as u grows and is within the uncertainty bands from $u = 8$ on (2.5% against 2.3%). We estimate the number of samples a run visits from the mean number of response tokens revealed per step, divided by the mean response length measured on the Post-SFT MDM control.<sup>3</sup>

Figures 29 and 30 report the efect on score. The threshold helps most where PU is limited by small u: it adds 13.2–13.8 points at $u = 4$ and 2.4–5.4 points at $u = 8 ,$ across decode thresholds. For $u \in \{ 1 6 , 3 2 , 6 4 \}$ the diference lies between 0.3 and +2.0 points, and at u = 128 between 0.8 and +0.3. $\mathrm { A t } ~ u \le 2$ , both variants collapse below 6 points. These comparisons are at matched u and $\tau ,$ not at matched NFEs, since the threshold also changes the NFEs a run needs at a given τ. Figure 31 accounts for cost. Without the threshold, no value of u clearly dominates both the 1.5-epoch SFT checkpoint and the Post-SFT MDM control across NFEs. With it, several values of u, for example $u \in \{ 3 2 , 6 4 \}$ , lie above both baselines over large ranges of NFEs (Figure 5b).

![](images/758b5091eec6c2ee9f4cc479cd7ea7d20daf328bd8873ce0e8df8f434d9ddd25.jpg)  
Figure 28 Distinct training samples visited as a percentage of Post-SFT MDM, for full-canvas PU with and without the training-time threshold. Bands: one standard error.

![](images/2bca04d1186626a5fc1e56cdf3b472ba420d39e60559c1c98a64cdf315890964.jpg)  
Figure 29 Full-canvas PU Post-SFT across u, with $( \tau _ { \mathrm { t r a i n } } = 0 . 9 )$ and without the training-time threshold. Columns: decode threshold τ. Top: score. Bottom: diference, with minus without, in points at matched train u and τ; NFEs are not matched.

![](images/28ac2455c1e3d1c10d5b554f4a9a2222c075550f6bd84635883cc8d2b45a17a3.jpg)  
Figure 30 Per-benchmark scores for the runs in Figure 29. Columns: decode threshold τ. Rows: score, then each benchmark.

![](images/372e02d922e24e7f2896ecb9b5a109348106cba12d8b014705f372d7d4f6d3e9.jpg)  
PU u=1 PU u=4 PU u=16 PU u=64 Full canvas, 1.5 epochs PU u=2 PU u=8 PU u=32 PU u=128 Post-SFT MDM  
Figure 31 Score against NFEs for full-canvas PU Post-SFT across $u ,$ without the training-time threshold. Each curve sweeps the decode threshold $\tau \in \{ 0 . 7 , 0 . 8 , 0 . 9 , 1 \}$ . Left: full range. Right: scores above 50%.

## F.8 Comparison with published LLaDA results

Table 13 compares our measurements with published LLaDA-8B values. These are the only external numbers in the paper; every other comparison is between runs that share the data, decoding policy, and evaluation harness. With the plain prompt format (Appendix B.4), our harness reproduces the published base figures. All other results use the chat prompt format (Appendix B.4), which matches the format of our SFT data. With it, the base model scores 10.1 points lower on GSM8K, 3.2 lower on MBPP, and 2.4 higher on IFEval. The prompt format is fixed across all our runs, so this shift does not afect our comparisons, but our absolute scores should not be compared directly with published ones. Published figures also disagree with each other: for LLaDA-8B-Instruct on GSM8K, Nie et al. (2025) report 69.4 with full-canvas decoding and 77.5 with block-32 decoding, and Ye et al. (2025) report 78.6.

The checkpoints our Post-SFT runs start from score 75.9 and 77.0 on GSM8K, 38.8 and 41.6 on MBPP, and 68.0 and 65.4 on IFEval. These are in the same range as the published LLaDA-8B-Instruct figures, and higher on IFEval than LLaDA-8B-Instruct as evaluated by Ye et al. (2025) (59.9).

Table 13 LLaDA-8B: published figures against our own measurements. The published rows are reference only and are not protocol-matched to anything else in this paper. Shot counts are given per row as IFEval/GSM8K/MBPP, with n/r where the source does not report a value. Our rows decode with Fast-dLLM at τ = 1, left to right in blocks of 32 tokens, and report IFEval 0-shot prompt-level strict accuracy, GSM8K 8-shot exact match, and MBPP 3-shot pass@1. The two LLaDA-8B-Base rows difer only in the prompt format; the two SFT rows are the checkpoints the Post-SFT runs start from, evaluated in the chat format. The last two published rows are LLaDA as evaluated by Ye et al. (2025), not Dream’s own model.
<table><tr><td>Model</td><td>Source</td><td>Shots</td><td>IFEval</td><td>GSM8K</td><td>MBPP</td></tr><tr><td colspan="6">Published, reference only</td></tr><tr><td>LLaDA-8B-Base</td><td>LLaDA</td><td> $\mathrm { n / r / 4 / 4 }$ </td><td> $\mathbf { n } / \mathbf { r }$ </td><td>70.3</td><td>40.0</td></tr><tr><td>LLaDA-8B-Instruct, full canvas</td><td>LLaDA</td><td> $\mathrm { n } / \mathrm { r } / 4 / 4$ </td><td> $\mathbf { n } / \mathbf { r }$ </td><td>69.4</td><td>41.0</td></tr><tr><td>LLaDA-8B-Instruct, block 32</td><td>LLaDA</td><td> $\mathrm { n } / \mathrm { r } / 4 / 4$ </td><td> $\mathbf { n } / \mathbf { r }$ </td><td>77.5</td><td>34.2</td></tr><tr><td>LLaDA-8B-Base</td><td>Dream</td><td> $\mathrm { n / r / 8 / 4 }$ </td><td> $\mathbf { n } / \mathbf { r }$ </td><td>70.9</td><td>39.0</td></tr><tr><td>LLaDA-8B-Instruct</td><td>Dream</td><td> $\mathrm { n / r }$ </td><td>59.9</td><td>78.6</td><td>34.2</td></tr><tr><td colspan="6">Ours, LLaDA-8B-Base at τ = 1</td></tr><tr><td>Plain prompt format</td><td>This work</td><td>0/8/3</td><td>20.9</td><td>70.4</td><td>39.8</td></tr><tr><td>Chat prompt format</td><td>This work</td><td>0/8/3</td><td>23.3</td><td>60.3</td><td>36.6</td></tr><tr><td colspan="6">Ours, LLaDA-8B-Base after SFT on Dolci-Instruct-SFT, at τ = 1</td></tr><tr><td>Full canvas, 1.5 epochs</td><td>This work</td><td>0/8/3</td><td>68.0</td><td>75.9</td><td>38.8</td></tr><tr><td>Block diffusion, 6 epochs</td><td>This work</td><td>0/8/3</td><td>65.4</td><td>77.0</td><td>41.6</td></tr></table>

## G Relation to the closest work

Our contribution concerns how to train masked difusion models for the trajectories they encounter during generation: which states to train on, what information to propagate between them, and how far to propagate the learning signal. PUMA (Kim et al., 2026), Loopholing (Jo et al., 2026), MetaState (Xia et al., 2026), and Relay (Rozonoyer et al., 2026) establish complementary foundations for these questions. Building on them, we study trajectory alignment with the inference policy and temporal credit assignment via BPTT together, evaluate their contribution to general-purpose instruction post-training, and investigate the conditions under which closer alignment remains learnable.

PUMA: from trajectory construction to temporal credit assignment. Kim et al. (2026) establish progressive unmasking as an efective training strategy, with a marginal-agreement guarantee under idealized posterior sampling. We build on their teacher-forced trajectories: the policy selects reveal positions, but ground-truth commitments still difer from an imperfect model’s sampled content. Our carry adds the predecessor’s modelproduced representation, providing a weak form of student forcing and a diferentiable path from later losses to earlier computations. PUMA also recognizes that long chains can provide redundant supervision and motivates a coarse-to-fine schedule partly by the unreliable policy early in training. Its schedule increases the number of stages, reducing the fraction of each sequence revealed per step, whereas inference includes two- and three-token steps.<sup>4</sup> Its schedule ablations already suggest that matching the reveal order need not mean matching its granularity. Our post-training results show that small-count degradation persists with a pretrained, instruction-tuned denoiser. An informed policy does not by itself ensure a learnable trajectory. We trace it to local overfitting caused by sample reuse (Section 4.3), and also examine masking-level diversity and the balance of supervision on committed versus uncommitted positions (Appendix F), connecting schedule sensitivity to concrete learning dynamics. PUMA’s large-model experiments establish efectiveness in a specialized setting: LoRA adaptation of a code model, evaluated on HumanEval and MBPP. Our full-model, general instruction post-training tests a broader question: whether the gains extend to instruction following while retaining reasoning and code capabilities, under both full-canvas and block difusion.

Loopholing: from self-conditioned context to trajectory-trained state. Jo et al. (2026) establish the value of retaining continuous information across discrete sampling. They name the underlying limitation the sampling wall: collapsing each predicted distribution to a single token discards what the model knew about the alternative tokens, and the carry lets part of this information survive the commitment. We adopt their simple, empirically efective hidden-state carry to isolate the role of training and trajectory alignment with minimal architectural changes. Their self-conditioning pass takes a corrupted sequence and an empty carry, then supplies its detached output (with stop-gradient) to a second prediction on the same sequence. Our cached carry instead comes from the preceding rollout step, which itself consumed its predecessor’s carry after trajectory initialization. We therefore hypothesize that this recurrent history better matches inference-time conditioning than a context generated in isolation, while recognizing that teacher-forced token content still leaves a mismatch. A stored carry that crosses an optimizer update was also produced by the parameters before that update. Loopholing reports approximately 30% additional training time. Our caching approach removes its auxiliary context-generation pass, thus avoiding this additional training cost, and replaces it with storage of one hidden-width vector per token, rather than a vocabulary-sized distribution. This is a compact repre sentation relative to logits, although total storage still scales with sequence length and the number of active trajectories. Within each unroll window, we can also train the computation that produced the carry through subsequent losses. PUMBA thus develops a multi-step training direction, combining recurrent conditioning with temporal credit assignment. The removed auxiliary pass and the additional memory and computation of BPTT are separate costs, and we provide more details on this trade-of between them in Table 7.

RCD and DiffusionGemma: distribution-derived carries. Residual Context Difusion (RCD; Hu et al., 2026) and DifusionGemma (DifusionGemma Team, 2026) carry a diferent quantity from the backbone hidden state we adopt from Loopholing. Both pass the expected input embedding $\mathbf { E } ^ { \top } \mathbf { p }$ to the next step, where $\mathbf { p } \in \Delta ^ { \vert \nu \vert }$ is the predicted distribution at a position and $\mathbf { E } \in \mathbb { R } ^ { | \nu | \times d }$ is the input embedding table. The resulting vector $\mathbf { E } ^ { \top } \mathbf { p } \in \mathbb { R } ^ { d }$ has the same width as the hidden-state carry. RCD entropy-weights this residual. It trains the target model using residuals supplied by a frozen reference model, then recursively feeds back the target model’s own predictions at inference. DifusionGemma applies a temperature to the logits before computing p and passes the expected embedding through a feedforward network, which yields a self-conditioning signal for the next denoising step. Adapting such carries to our trajectory-trained setting is an interesting direction.

MetaState: memory architecture and trajectory alignment. Xia et al. (2026) address the information island: continuous computations are lost between denoising steps when only discrete token decisions persist. Their fixed-size working memory improves reasoning with a frozen backbone. Their objective also combines supervision on all masked positions with a term focused on the positions about to be revealed. This emphasis connects to trajectory alignment, and we investigate a loss following that design in our small-u analysis (Section 4.3 and Appendix F). Both approaches train recurrent state through unrolling, but MetaState uses random reveal orderings and Dirichlet–Multinomial reveal counts, whereas its decoder is confidence-based. PUMBA follows the model’s unmasking policy and updates the denoiser itself, thus better aligning with our training–inference alignment objective. MetaState uses four unrolled steps by default and reports broadly stable performance over $K \in \{ 3 , 4 , 5 \}$ . These steps partition the entire training trajectory; our window W instead truncates gradients along trajectories that can span multiple updates. MetaState trains on 50k examples from the general-purpose Tülu-3 mixture and evaluates mathematical reasoning and code generation. Our distinction is not general-purpose training data alone: we study full-model instruction post-training and explicitly evaluate instruction following alongside reasoning and code. Their work advances memory architecture and eficiency, while we fix a minimal carry to isolate training and alignment choices. Therefore, we do not compare the architectures directly; combining more efective or compact memories with PUMBA is an important direction for future work (Section 6).

Relay: shared mechanism, complementary evidence and analysis. Concurrent work by Rozonoyer et al. (2026) is particularly close: it combines model-selected, teacher-forced rollouts with diferentiable per-token state and truncated BPTT. It includes rollout-only and detached-carry (i.e., with stop-gradient) controls in both Sudoku and Fast-dLLM v2 (1.5B). The latter is adapted on a code-and-mathematics mixture and evaluated on HumanEval and MBPP. Our setting is general instruction post-training of LLaDA-8B: we evaluate instruction following as the target capability and mathematical reasoning and code generation for capability retention, under both full-canvas and block difusion. Relay’s results also motivate studying the components across settings: the detached carry gives a substantial gain on Sudoku, while the language-model results show a diferent balance between rollout accuracy and the NFE benefit of temporal training. Their large-model study establishes efectiveness for a two-step unroll within block difusion. We broaden this evidence by jointly varying training-trajectory granularity and the unroll horizon at 8B scale, under both full-canvas and block decoding: the question is how reliably the benefits extend across post-training configurations, and what controls their magnitude. In particular, we characterize when closer alignment improves the quality– NFE frontier, when repeated states and concentrated masking levels impair learning, and how targeted interventions mitigate these efects (Section 4.3 and Appendix F). This analysis provides practical guidance for carry-based post-training beyond establishing that a diferentiable channel can help.