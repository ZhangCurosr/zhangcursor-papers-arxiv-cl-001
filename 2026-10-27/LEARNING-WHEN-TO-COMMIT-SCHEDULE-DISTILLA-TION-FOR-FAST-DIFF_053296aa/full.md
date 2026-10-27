# LEARNING WHEN TO COMMIT: SCHEDULE DISTILLA-TION FOR FAST DIFFUSION LANGUAGE MODELS

Aidar Myrzakhan<sup>1</sup>, Zhiqiang Shen<sup>1</sup>

<sup>1</sup>VILA Lab, MBZUAI

{aidar.myrzakhan,zhiqiang.shen}@mbzuai.ac.ae

## ABSTRACT

Diffusion large language models (dLLMs) can generate multiple tokens in parallel, but often require many decoding rounds to complete a response. Efficient generation depends not only on which tokens are easy to predict, but also on how committing to them helps predict the remaining positions. A highly predictable token may provide little useful context, while revealing a less predictable token can help several other positions become predictable sooner. We propose dSpark, a new schedule distillation-based framework for learning when to commit tokens for faster diffusion language generation. To achieve this, we construct unmasking schedules that prioritize tokens whose reveal makes other positions easier to predict, reducing uncertainty earlier in decoding. We then distill these schedules into the model to enable faster parallel sampling, with a KL-divergence regularizer to help preserve generation quality. We extensively evaluate our approach across mathematical reasoning and code generation tasks. On GSM8K, our method achieves remarkable 10.12× and 5.12× speedups over the various baselines on LLaDA-8B-Instruct and Dream-7B-Instruct, respectively, while maintaining superior task performance. Our code is available at https://github.com/VILA-Lab/dSpark.

## 1 INTRODUCTION

Large language models (LLMs) have achieved remarkable capabilities across natural language understanding (Brown et al., 2020; Ouyang et al., 2022; Achiam et al., 2023; Touvron et al., 2023; Grattafiori et al., 2024; Team et al., 2023), and reasoning (Kojima et al., 2022; Jaech et al., 2024; Guo et al., 2025), yet efficient inference remains a fundamental challenge. Most LLMs adopt an autoregressive (AR) paradigm, producing each token conditioned on all previous tokens. This strict left-toright dependency limits decoding parallelism: generating L tokens typically requires L sequential steps. Diffusion large language models (dLLMs) offer an appealing alternative by generating sequences through iterative denoising. From a partially or fully masked sequence, a dLLM predicts multiple masked positions in parallel and progressively commits them over a series of denoising rounds. Recent models such as LLaDA (Nie et al., 2025) and Dream (Ye et al., 2025) demonstrate that this paradigm can attain strong language modeling, reasoning, and coding capabilities while enabling flexible generation orders and much greater potential for parallel decoding (Ben-Hamu et al., 2025; Wu et al., 2026b; Chen et al., 2026a). However, realizing this potential requires reducing the many iterative model evaluations needed to turn a masked sequence into a complete response.

A central factor governing dLLM inference efficiency is the unmasking schedule: which token positions should be committed at each decoding round and which should remain masked for subsequent refinement. Given the predictive distributions over all masked positions, a decoder may reveal one token, a fixed number of tokens, or a dynamically selected subset in each iteration. Committing more positions increases parallelism and reduces the number of decoding rounds, but overly aggressive commitments may introduce irreversible errors that contaminate the context used to predict othe tokens. Conversely, conservative schedules preserve generation quality but sacrifice much of the computational advantage of parallel prediction. Existing decoding strategies therefore commonly rely on prediction confidence, e.g., committing positions whose maximum token probabilities are highest, under the intuition that highly confident predictions are safer to reveal. While effective as a simple heuristic, this strategy treats each masked position largely by its current local predictability and does not explicitly model how revealing it will change the difficulty of subsequent predictions.

We argue that this distinction between predictability and utility is crucial for efficient diffusion language generation. The easiest token to predict is not necessarily the most useful token to commit. Consider two masked positions i and j. Position i may already have an extremely peaked predictive distribution, yet revealing it may add almost no information about the rest of the sequence. In contrast, position j may be moderately less certain but encode a critical semantic, syntactic, or reasoning decision, once j is revealed, the uncertainty of several other positions may decrease substantially. This is particularly important in structured generation tasks such as mathematical reasoning and code synthesis, where intermediate decisions induce strong dependencies among downstream tokens. A schedule based solely on instantaneous confidence may therefore spend early commitments on safe but informationally redundant tokens, leaving influential positions unresolved for many more rounds. The effective decoding problem is thus not simply which tokens can be predicted now, but which tokens should be committed now so that the remaining sequence becomes easier to predict.

Motivated by this observation, we propose dSpark, a novel framework for learning when to commit tokens during diffusion language generation via schedule distillation. Our key idea is to construct informative unmasking schedules according to the downstream effect of each commitment rather than its confidence alone. We evaluate candidate commitments by how much revealing them helps reduce predictive uncertainty over the remaining positions. Tokens whose revelation makes multiple unresolved positions easier to determine are prioritized earlier, allowing useful context to propagate more rapidly. Decoding can thus be viewed as a context-construction process: every committed token not only resolves one position but also modifies the prediction problem of all remaining positions. By accounting for this effect, our schedules favor commitments with high future predictive value, producing trajectories that reduce sequence-level uncertainty faster than conventional confidence-driven schedules. This enables substantially more useful progress within each expensive model evaluation.

Constructing better schedules alone, however, can add computation if sophisticated token selection must be repeated during inference. We therefore further distill the behavior induced by these schedules into the model itself. The constructed schedules provide training trajectories that expose the model to commitment orders in which informative tokens are revealed early and the remaining predictions are progressively simplified. After distillation, the model can exploit these learned dependencies under efficient parallel sampling without expensive search over candidate schedules at every decoding step. Importantly, aggressively optimizing for fast commitment can alter the model’s original predictive behavior and degrade generation quality. To mitigate this, we introduce a KL-divergence regularizer that keeps the distilled model close to the original model distribution while learning to operate under accelerated schedules. Schedule distillation thus shifts computation from repeated inference-time schedule optimization into training, yielding a practical mechanism for jointly improving parallelism and preserving the generation capability of the underlying dLLM.

We extensively evaluate dSpark on two representative models, LLaDA-8B-Instruct and Dream-7B-Instruct, across mathematical reasoning and code generation benchmarks. The results demonstrate that learning when to commit is a powerful yet underexplored dimension for accelerating dLLMs. On GSM8K, our approach achieves 10.12× and 5.12× speedups over the baselines for LLaDA and Dream, respectively, while maintaining superior task performance. This shows that generation efficiency depends not merely on how many tokens are predicted in parallel, but on how effectively each commitment restructures the prediction problem for later rounds. Our main contributions are:

• We identify a key limitation of existing diffusion language model decoding strategies: token confidence alone does not capture the downstream utility of a commitment. We show that revealing a less predictable but more informative token can reduce uncertainty over multiple remaining positions and lead to faster overall decoding.

• We propose schedule distillation, a new strategy that constructs unmasking schedules by prioritizing tokens according to how much their commitment simplifies future predictions. We further distill these informative schedules into the model, enabling efficient parallel decoding without requiring expensive schedule optimization at inference time.

• We extensively validate our method across mathematical reasoning and code generation tasks on multiple dLLMs. On GSM8K, it achieves up to 10.12× speedup on LLaDA-8B-Instruct and 5.12× on Dream-7B-Instruct while maintaining strong generation quality.

## 2 RELATED WORK

Diffusion Language Models. Discrete diffusion models generate sequences by reversing a corruption process, and subsequent work developed stronger objectives for language modeling, including SEDD, MDLM, and RADD (Austin et al., 2021a; Lou et al., 2024; Sahoo et al., 2024; Ou et al., 2025). LLaDA (Nie et al., 2025) scaled masked diffusion to an instruction-following language model trained from scratch, while Dream (Ye et al., 2025) demonstrated strong reasoning and coding performance with a model initialized from an autoregressive LLM. Other work combines diffusion within blocks with generation across blocks or scales diffusion models further through conversion from autoregressive models (Arriola et al., 2025; Bie et al., 2025; Zhu et al., 2026). These advances establish capable diffusion backbones, but their iterative generation process leaves substantial room to improve inference efficiency.

Efficient Diffusion Language Decoding. Existing methods reduce either the cost or the number of decoding steps. Fast-dLLM (Wu et al., 2026b), dKV-Cache (Ma et al., 2025a), Elastic-Cache (Nguyen-Tri et al., 2026), and dInfer (Ma et al., 2025b) improve cache reuse and inference execution, while D2F (Wang et al., 2026) and Fast-dLLM v2 (Wu et al., 2026a) use block-based generation to support efficient parallel decoding. Other approaches distill generation into fewer steps or enable more aggressive token updates (Deschenaux & Gulcehre, 2025; Sahoo et al., 2025; Chen et al., 2026a). Closest to our work, dParallel (Chen et al., 2026b) uses certainty-forcing distillation to make tokens ready earlier, whereas d3LLM (Qian et al., 2026) distills pseudo-trajectories and applies multi-block decoding. Training-free approaches such as LEAP (Zhang et al., 2026) and Ripple-Pivot Search (Ye et al., 2026) use lookahead to identify tokens worth committing early.

## 3 PRELIMINARIES

## 3.1 MASKED DIFFUSION LANGUAGE MODELS

Masked diffusion language models (MDLMs) generate text by learning to recover tokens from partially masked sequences (Austin et al., 2021a). For a clean sequence $\mathbf { \bar { \boldsymbol { x } } } _ { 0 } \in \mathcal { V } ^ { L }$ , the linear forward process independently replaces each token with the absorbing state MASK with probability t:

$$
q _ { t \mid 0 } ( x _ { t } \mid x _ { 0 } ) = \prod _ { i = 1 } ^ { L } q _ { t \mid 0 } ( x _ { t } ^ { i } \mid x _ { 0 } ^ { i } ) = \prod _ { i = 1 } ^ { L } \mathrm { C a t } \Bigl ( x _ { t } ^ { i } ; ( 1 - t ) \delta _ { x _ { 0 } ^ { i } } + t \delta _ { \mathrm { \tt M A S K } } \Bigr ) \ ,\tag{1}
$$

where $\nu$ is the token vocabulary, MASK $\notin \mathcal { V } ,$ , Cat denotes a categorical distribution, and $\delta _ { v }$ is a point mass at v. The noise level $t \in [ \bar { 0 } , 1 ]$ interpolates between the clean sequence and complete masking. A bidirectional denoiser $p _ { \theta } ( x _ { 0 } ^ { i } \mid x _ { t } )$ is trained to reconstruct masked tokens with the reweighted cross-entropy objective (Sahoo et al., 2024; Lou et al., 2024)

$$
\mathcal { L } _ { \mathrm { M D M } } ( \theta ; x _ { 0 } ) = \int _ { 0 } ^ { 1 } \frac { 1 } { t } \mathbb { E } _ { q _ { t \mid 0 } ( x _ { t } \mid x _ { 0 } ) } \left[ \sum _ { i : x _ { t } ^ { i } = \mathrm { M A S K } } - \log p _ { \theta } ( x _ { 0 } ^ { i } \mid x _ { t } ) \right] \mathrm { d } t .\tag{2}
$$

This formulation underlies large-scale diffusion language models such as LLaDA (Nie et al., 2025) and Dream-7B (Ye et al., 2025), which predict masked tokens in parallel during iterative generation.

## 3.2 PARALLEL DECODING AND UNMASKING SCHEDULES

Given a fixed prompt, decoding starts from a fully masked response. Let $s _ { r }$ denote the partially decoded sequence at round r and $\mathcal { M } _ { \tau }$ the masked response positions eligible for commitment, such as those within an active block or window. A forward pass computes $p _ { \theta } ( \cdot \mid s _ { r } , j )$ for each $j \in \mathcal { M } _ { r } .$ The decoder selects a nonempty set $S _ { r } \subseteq { \mathcal { M } } _ { r }$ , for example using confidence thresholds (Nie et al., 2025) or predictive-entropy budgets (Ben-Hamu et al., 2025). Under greedy token assignment, the sequence is updated as

$$
s _ { r + 1 } ^ { j } = \left\{ \begin{array} { l l } { \underset { v \in \mathcal { V } } { \arg \operatorname* { m a x } } p _ { \theta } ( v  { \mid s _ { r } , j ) } , } & { j \in S _ { r } , } \\ { s _ { r } ^ { j } , } & { j \not \in S _ { r } . } \end{array} \right.\tag{3}
$$

An unmasking schedule $\pi = ( S _ { 0 } , \ldots , S _ { T - 1 } )$ specifies which positions are revealed in each round.

![](images/e583894e13de04ea4d85f317edfeca770411f451ab14b7cdafe7ef4b61f1c1ea.jpg)  
(a) Surprisal drop at other positions

![](images/0c5835e3941df11b289699ddd6a82d380f1c4e71e3c532b302d022574ee1072c.jpg)  
(b) Newly ready positions

![](images/c92c8059df4266c61f77d64ec9e60fec0c666e15cf3533557b7ef1db4c618208.jpg)  
(c) Unc. Area vs. forward passes  
Figure 1: The most predictable token is not the most useful one to reveal. On correct GSM8K responses from LLaDA, we reveal one masked token $y _ { i }$ and measure its effect on the other masked positions. In (a) and (b), each dot is one reveal, and the gray region marks uncertain tokens $( p _ { \theta _ { 0 } } < 0 . 5 )$ which confidence-based decoders defer. (a) Mean surprisal drop at the other positions (nats). (b) Net number of other positions that become ready, i.e., whose top-1 prediction matches the reference. (c) Responses with a larger uncertainty area A (Eq. (4)) need more forward passes.

## 4 METHODOLOGY

## 4.1 MOTIVATION

The order in which tokens are revealed can affect how quickly a dLLM completes a response. Each reveal provides context for the remaining positions, but some tokens help more than others. We measure this with the frozen base model $p _ { \theta _ { 0 } }$ on a correct response $y .$ In a partially decoded state $s ,$ the model’s uncertainty about the target token at a masked position j is its surprisal $h _ { j } ( s ) =$ $- \log p _ { \theta _ { 0 } } ( y _ { j } \mid s , j )$ . Fig. 1a and 1b show that revealing less predictable target tokens (larger $h _ { i } )$ can reduce uncertainty at other positions and help more predictions match the correct response. We call a position ready when the model’s most likely token matches the ground truth. Revealing an uncertain token makes 5.1× more other positions ready than revealing a confident one (Fig. 1b) and lowers their surprisal $4 . 7 \times$ more (Fig. 1a). A useful schedule should therefore consider how each token helps predict the remaining positions, as well as how confident the model is about that token.

What matters for efficient decoding is not only how much uncertainty a reveal removes, but how early it removes it. Following the masked-token surprisal sum inside the brackets of Eq. (2), we measure remaining reference uncertainty and its accumulated area as

$$
H ( s _ { r } ; y ) = \sum _ { j \in \mathcal { M } _ { r } } h _ { j } ( s _ { r } ) , \qquad A ( \pi ; y ) = \sum _ { r = 0 } ^ { T - 1 } H ( s _ { r } ; y ) .\tag{4}
$$

The area adds up the model’s remaining uncertainty at every decoding round. If revealing a token early makes other tokens easier to predict, their uncertainty is lower in the following rounds and the total area is smaller. Two schedules can therefore produce the same response but have different areas. Figure 2a illustrates this: under the natural schedule, the remaining uncertainty stays high for many rounds before it falls, whereas promoting tokens that help the other positions makes it drop earlier and shifts the whole curve to the left, removing the hatched area from A. Since the response is fixed, this gain comes only from the order of reveals. A smaller area also means faster completion. Responses with larger areas under the natural schedule tend to require more forward passes (Fig. 1c), and optimized schedules finish verified responses in fewer rounds, never take longer, and gain the most on the responses that need the most rounds (Fig. 2b). Our method starts from the natural schedule and searches for tokens whose earlier reveal helps other positions become ready sooner.

## 4.2 UNCERTAINTY-GUIDED SCHEDULE CONSTRUCTION

The uncertainty-area objective in Eq. (4) motivates revealing tokens that make the remaining positions easier to predict early in the trajectory. Figure 3 summarizes our framework. Given a correct response $y ,$ we construct its unmasking trajectory offline with the frozen model $p _ { \theta _ { 0 } }$ . The response remains fixed, and we change only the order in which its tokens are revealed. At each round, we compare the candidate reveals by how much they reduce uncertainty at the other masked positions, as illustrated on the right of Figure 3. This gives a local selection rule based on the context a token provides, rather than only on how predictable that token already is.

![](images/dfa88f95fb10c78e13c7ed94e42bfcbe783904e791b1f8ca3aaeff0467a3861d.jpg)  
(a) Uncertainty area

![](images/1a7de8c1d9368cb291a2b8125603e8219434646a280d7e6d074eeeb6cda2d206.jpg)  
(b) Completion of verified responses  
Figure 2: Promoting helpful tokens makes uncertainty fall earlier and responses finish sooner. Both schedules reveal the same response, in a different order. (a) The hatched region is the part of the uncertainty area $\mathcal { A }$ (Eq. (4)) that the optimized schedule removes. (b) Share of responses completed by round r. The hatched area shows the mean number of rounds saved.

At round $r ,$ we consider plausible target tokens among the masked positions $\mathcal { M } _ { r }$ in state $s _ { r } ,$

$$
\begin{array} { r } { \mathcal { C } _ { r } = \left\{ i \in \mathcal { M } _ { r } \mid p _ { \theta _ { 0 } } ( y _ { i } \mid s _ { r } , i ) \geq p _ { \operatorname* { m i n } } , \right. \left. \mathrm { r a n k } _ { \theta _ { 0 } } ( y _ { i } \mid s _ { r } , i ) \leq k \right\} . } \end{array}\tag{5}
$$

Here $p _ { \mathrm { m i n } }$ is a minimum target token probability and k is the maximum allowed rank among vocabulary tokens. For each candidate i, let $s _ { r } ^ { ( i ) }$ be $s _ { r }$ with $y _ { i }$ revealed and all other positions unchanged. Using the per-position surprisal $h _ { j } ( s ) \ = \ - \log p _ { \theta _ { 0 } } ( y _ { j } \mid s , j )$ , we select the candidate that most reduces uncertainty at other positions,

$$
i _ { r } ^ { * } \in \underset { i \in \mathcal { C } _ { r } } { \arg \operatorname* { m a x } } \sum _ { j \in \mathcal { M } _ { r } \backslash \{ i \} } \left[ h _ { j } \big ( s _ { r } \big ) - h _ { j } \big ( s _ { r } ^ { ( i ) } \big ) \right] .\tag{6}
$$

The score measures the reduction in uncertainty at the other masked positions, excluding the revealed position i itself. We promote the candidate with the largest positive score. If no candidate has a positive score, we skip promotion. Repeating these steps until the response is fully revealed produces $\tau ^ { * } ( y ) = ( s _ { 0 } ^ { * } , \ldots , s _ { T } ^ { * } )$ , where $s _ { r } ^ { * }$ is the state after r rounds. This optimized trajectory is illustrated on the left of Figure 3. We pair its intermediate states $s _ { r } ^ { * }$ with the correct response $y$ to train the student p<sub>θ</sub> to predict the remaining masked tokens.

## 4.3 SCHEDULE DISTILLATION

We train the student on intermediate states $s _ { r } ^ { * }$ from the optimized trajectory $\tau ^ { * } ( y )$ . The correct response provides the token targets, while the schedule determines which tokens are visible in each training example. The student, therefore, learns to predict the remaining tokens from contexts in which informative positions have been revealed earlier.

Let $\mathcal { D } _ { + }$ denote the distribution of correct responses y and states $s _ { r } ^ { * }$ sampled from $\tau ^ { * } ( y )$ . We combine cross-entropy supervision with confidence sharpening. At each state $s _ { r } ^ { * } , {  { \mathcal { M } } } _ { \gamma }$ contains the eligible masked positions and $\mathcal { R } _ { r } \subseteq { \mathcal { M } } _ { \mathrm { ~  ~ } }$ those where the student’s most likely token matches the target token $y _ { j }$ . The positive-trajectory objective is

$$
\begin{array} { r } { \mathcal { L } _ { + } ( \theta ) = \mathbb { E } _ { ( s _ { r } ^ { * } , y ) \sim \mathcal { D } _ { + } } \big [ \ell _ { \mathrm { C E } } ( s _ { r } ^ { * } , y ) + \ell _ { \mathrm { s h a r p } } ( s _ { r } ^ { * } , y ) \big ] , } \end{array}\tag{7}
$$

where $\ell _ { \mathrm { C E } }$ trains the student to reconstruct the masked tokens of the correct response $y ,$ using crossentropy averaged over $\mathcal { M } _ { r } .$ . Training on intermediate states from our optimized trajectory encourages the student to follow its unmasking progression during decoding. The sharpening term $\ell _ { \mathrm { s h a r p } }$ averages the predictive entropy $\mathcal { H } ( p _ { \theta } ( \cdot  { | } s _ { r } ^ { * } , j ) )$ ) over currently correct positions $j \in \mathcal { R } _ { r } ,$ where $\begin{array} { r } { \mathcal { H } ( q ) = - \sum _ { v \in \mathcal { V } } q ( v ) \log q ( v ) } \end{array}$ . Minimizing this term strengthens confidence in correct predictions, supporting earlier commitments during decoding.

![](images/ad4933e77988d8f0d04a8875e430be23628fc0ff39ec801716d2763e5a274038.jpg)  
Figure 3: Overview of dSpark. $L e f t { \mathrm { : } }$ for a fixed correct response $y ,$ the frozen teacher $p _ { \theta _ { 0 } }$ builds an unmasking trajectory that reveals the same tokens in a new order. Masked cells are shaded by surprisal, with darker cells more uncertain. The grid below gives the round in which each token is revealed. Promoting $^ { 6 6 } 1 2 ^ { , 5 }$ first makes the other tokens predictable sooner, so the response is complete after 3 rounds instead of 6. Right: single-step promotion. Each plausible candidate $i \in \mathcal { C } _ { r }$ is revealed from the same state $s _ { r } .$ . Gray and green bars show the surprisal at the other masked positions before and after the reveal. The candidate with the largest positive total drop (Eq. (6)) is promoted and committed with the ready positions. Bottom: the student is trained on promoted trajectories $( \mathcal { L } _ { + } )$ with KL to $p _ { \theta _ { 0 } }$ on failed ones (L ).

The sharpening term, however, cannot tell right from wrong on its own. It is applied only where the student is already correct on the constructed trajectories, but the parameters it updates are shared, so the added confidence carries over to states the student reaches on unsuccessful generation paths. On those states, more confidence can only hurt, so the student must stay where the base model was. We therefore retain failed trajectories for selective KL regularization. Response-level failure does not identify which individual tokens are incorrect, so we use the frozen base model’s distribution as the target rather than the failed response’s tokens. Let $\mathcal { D } _ { - }$ denote the distribution of state–position pairs sampled for anchoring from these trajectories. The negative-trajectory loss is

$$
\begin{array} { r } { \mathcal { L } _ { - } ( \theta ) = \mathbb { E } _ { ( s ^ { - } , j ) \sim \mathcal { D } _ { - } } \left[ D _ { \mathrm { K L } } \left( p _ { \theta _ { 0 } } ( \cdot \vert s ^ { - } , j ) \Vert p _ { \theta } ( \cdot \vert s ^ { - } , j ) \right) \right] , } \end{array}\tag{8}
$$

where both models receive the same state $s ^ { - }$ and the base model remains frozen. This anchoring limits changes where reliable token-level supervision is unavailable, while leaving the constructed correct trajectories free of the KL penalty. The full training objective is

$$
\begin{array} { r } { \mathcal { L } ( \boldsymbol { \theta } ) = \mathcal { L } _ { + } ( \boldsymbol { \theta } ) + \beta \mathcal { L } _ { - } ( \boldsymbol { \theta } ) , } \end{array}\tag{9}
$$

where $\beta$ controls the regularization strength. Together, these objectives encourage earlier confident commitments for faster parallel decoding while maintaining high task accuracy.

## 5 EXPERIMENTS

## 5.1 EXPERIMENTAL SETUP

Training Data. We build a separate distillation corpus for each backbone. For both models, we generate the trajectories from GSM8K (Cobbe et al., 2021), PRM12K (Lightman et al., 2024), NuminaMath-CoT (LI et al., 2024), and AceCode (Zeng et al., 2025) using a maximum length of 256 tokens, with a block size of 32. Overall, the training data contains 52k correct and 38k incorrect samples for LLaDA, and 80k correct and 57k incorrect for Dream. During training, we use all correct trajectories and subsample the incorrect ones to a 4:1 ratio, yielding 52k correct and 13k incorrect trajectories for LLaDA, and 80k correct and 20k incorrect for Dream.

Implementation Details. In our experiments we use LLaDA-Instruct (Nie et al., 2025) and Dream Instruct (Ye et al., 2025) models. We fine-tune the models using LoRA (Hu et al., 2021). We use a multi-block decoding strategy similar to d3LLM (Qian et al., 2026). To create a set of target tokens in Eq. 5, we use $p _ { m i n } = 0 . 0 2$ and k = 10. We set β = 1 in the training objective in Eq. 9. Training configurations and hardware details are provided in the Appendix.

Evaluation Details. Our evaluation covers mathematical reasoning on GSM8K (0-shot) (Cobbe et al., 2021) and MATH (4-shot) (Hendrycks et al., 2021), and code generation on HumanEval (0-shot) (Chen et al., 2021) and MBPP (3-shot) (Austin et al., 2021b). We report accuracy for math and pass@1 for code, together with tokens per forward (TPF) and accuracy under parallelism (AUP) (Qian et al., 2026). The results are averaged over three independent evaluation runs. We use lm-eval-harness (Gao et al., 2024).

Baselines. We evaluate our method against competitive dLLM baselines. These include the vanilla LLaDA (Nie et al., 2025) and Dream (Ye et al., 2025) models and the training-free Fast-dLLM (Wu et al., 2026b). We also compare with methods that fine-tune the model for faster decoding, namely dParallel (Chen et al., 2026b), d3LLM (Qian et al., 2026), and D2F (Wang et al., 2026).

## 5.2 MAIN RESULTS

Table 1 shows that dSpark achieves the highest TPF and AUP across all four benchmarks on both backbones. The results demonstrate a consistent improvement in the accuracy–parallelism trade-off, with the relative contributions of accuracy retention and increased parallelism varying across tasks.

Results on LLaDA. On GSM8K, dSpark achieves 10.12× the decoding parallelism of the base model while maintaining the base model’s accuracy. Compared to d3LLM, it further increases parallelism while closing the accuracy gap relative to the base model, resulting in a 33.3% improvement in AUP. On MATH, dSpark similarly preserves base-model performance while achieving the highest TPF and AUP. The code benchmarks show that the benefits extend beyond mathematical reasoning. On HumanEval, dSpark matches d3LLM’s pass@1 while decoding more tokens per forward pass. On MBPP, it improves both pass@1 and TPF over d3LLM. Across all four tasks, dSpark make better use of parallel decoding than the competing methods, with the highest AUP in every case.

Results on Dream. On GSM8K, dSpark achieves 5.12× the base model’s TPF while improving accuracy by 1.5%. Relative to d3LLM, the larger parallelism gains occur on code generation: TPF increases by 30.4% on MBPP and 36.4% on HumanEval, accompanied by higher pass@1 on both benchmarks. These improvements give dSpark a substantial AUP lead on the two code tasks and show that its advantage extends across backbones. On MATH, the gain over d3LLM is smaller, but dSpark still achieves the highest TPF and AUP. On MBPP, dSpark achieves the highest TPF and AUP of all compared methods, while also improving pass@1 over d3LLM.

Accuracy–parallelism trade-off. Figure 4 shows why dSpark leads in AUP. Methods can decode more tokens per forward pass by accepting a sharp drop in accuracy. dSpark achieves higher TPF while maintaining strong accuracy. This is clearest on GSM8K, where it achieves both the highest TPF and the highest accuracy on LLaDA and Dream. On HumanEval, it exceeds d3LLM’s TPF without losing pass@1 on LLaDA and improves both measures on Dream. The other plots show the same overall pattern: dSpark moves decoding toward greater parallelism while preserving enough task performance to make that gain useful. These results are consistent with the goal of dSpark, which encourages the model to reveal informative tokens earlier so that more of the remaining positions can be predicted together. The KL regularizer helps dSpark retain accuracy under aggressive parallel decoding, giving it an advantage over competing methods at higher TPF.

Table 1: Results by benchmark and backbone. Accuracy, tokens per forward (TPF), and accuracy under parallelism (AUP) for LLaDA-8B-Instruct and Dream-7B-Instruct.
<table><tr><td rowspan="2">Benchmark</td><td rowspan="2">Method</td><td colspan="3">LLaDA-8B-Instruct</td><td colspan="3">Dream-7B-Instruct</td></tr><tr><td>Acc.↑</td><td>TPF↑</td><td>AUP↑</td><td>Acc.↑</td><td>TPF↑</td><td>AUP↑</td></tr><tr><td rowspan="5">GSM8K</td><td>Base model</td><td>78.8</td><td>1.00</td><td>78.80</td><td>82.1</td><td>1.00</td><td>82.10</td></tr><tr><td>Fast-dLLM (Wu et al., 2026b)</td><td>75.5</td><td>2.91</td><td>199.24</td><td>80.0</td><td>1.52</td><td>122.01</td></tr><tr><td>D2F (Wang et al., 2026)</td><td>74.2</td><td>2.83</td><td>190.87</td><td>80.2</td><td>1.71</td><td>132.39</td></tr><tr><td>dParallel (Chen et al., 2026b)</td><td>75.6</td><td>7.09</td><td>469.03</td><td>82.8</td><td>3.06</td><td>246.01</td></tr><tr><td>d3LLM (Qian et al., 2026)</td><td>74.2</td><td>9.05</td><td>571.28</td><td>80.7</td><td>4.97</td><td>379.09</td></tr><tr><td></td><td>dSpark</td><td>78.9</td><td>10.12</td><td>761.67</td><td>83.6</td><td>5.12</td><td>424.28</td></tr><tr><td rowspan="5">MATH</td><td>Base model</td><td>31.7</td><td>1.00</td><td>31.70</td><td>39.5</td><td>1.00</td><td>39.50</td></tr><tr><td>Fast-dLLM (Wu et al., 2026b)</td><td>30.8</td><td>1.91</td><td>46.24</td><td>38.4</td><td>1.71</td><td>53.87</td></tr><tr><td>D2F (Wang et al., 2026)</td><td>28.5</td><td>2.44</td><td>45.96</td><td>38.0</td><td>1.64</td><td>51.32</td></tr><tr><td>dParallel (Čhen et al., 2026b)</td><td>30.3</td><td>3.16</td><td>64.57</td><td>38.1</td><td>2.93</td><td>76.76</td></tr><tr><td>d3LLM (Qian et al., 2026)</td><td>30.3</td><td>5.75</td><td>107.59</td><td>38.6</td><td>4.07</td><td>101.47</td></tr><tr><td></td><td>dSpark</td><td>32.0</td><td>6.03</td><td>119.55</td><td>38.9</td><td>4.31</td><td>106.58</td></tr><tr><td rowspan="6">MBPP</td><td>Base model</td><td>41.7</td><td>1.00</td><td>41.70</td><td>57.0</td><td>1.00</td><td>57.00</td></tr><tr><td>Fast-dLLM (Wu et al., 2026b)</td><td>38.2</td><td>2.20</td><td>57.33</td><td>54.5</td><td>1.17</td><td>62.75</td></tr><tr><td>D2F (Wang et al., 2026)</td><td>38.0</td><td>1.98</td><td>50.70</td><td>53.8</td><td>1.31</td><td>67.52</td></tr><tr><td>dParallel (Chen et al., 2026b)</td><td>40.2</td><td>2.27</td><td>59.46</td><td>56.2</td><td>2.35</td><td>113.76</td></tr><tr><td>d3LLM (Qian et al., 2026)</td><td>39.7</td><td>4.56</td><td>91.88</td><td>55.2</td><td>2.89</td><td>137.54</td></tr><tr><td>dSpark</td><td>40.0</td><td>5.52</td><td>104.94</td><td>56.0</td><td>3.77</td><td>171.43</td></tr><tr><td rowspan="5">HumanEval</td><td>Base model</td><td>38.1</td><td>1.00</td><td>38.10</td><td>55.2</td><td>1.00</td><td>55.20</td></tr><tr><td>Fast-dLLM (Wu et al., 2026b)</td><td>37.8</td><td>2.39</td><td>52.30</td><td>54.6</td><td>1.45</td><td>66.76</td></tr><tr><td>D2F (Wang et al., 2026)</td><td>38.2</td><td>2.71</td><td>60.59</td><td>54.0</td><td>1.39</td><td>66.78</td></tr><tr><td>dParallel (Chen et al., 2026b)</td><td>39.0</td><td>4.95</td><td>83.88</td><td>54.3</td><td>2.30</td><td>91.46</td></tr><tr><td>d3LLM (Qian et al., 2026)</td><td>39.6</td><td>5.95</td><td>96.58</td><td>57.1</td><td>3.19</td><td>129.15</td></tr><tr><td>dSpark</td><td></td><td>39.6</td><td>6.73</td><td>106.19</td><td>57.3</td><td>4.35</td><td>167.17</td></tr></table>

Confidence and False Positives. Distillation for parallel decoding makes a model confident enough to commit many tokens per forward pass. The risk is that it also becomes confident about wrong tokens, which a threshold decoder then commits. Table 8 measures this on the natural schedules of 250 test GSM8K samples, recording at every round the prediction at every masked position. The three distilled models are confident about a similar share of correct predictions. Among wrong predictions, however, d3LLM and dParallel are confident for 22.9% and 21.7%, while dSpark is confident for 15.5%, so its confident predictions are correct 91% of the time, against 82% for the two baselines. Figure 5 breaks these predictions into confidence bins. The base model is well calibrated: in every bin, its accuracy is close to its confidence. dParallel and d3LLM are overconfident at every confidence level; for example, their predictions made with a confidence between 0.8 and 0.9 are correct only 51% and 52% of the time. dSpark stays much closer to the diagonal, at 65% in the same bin, although it places slightly more predictions in the top bin than the two baselines. Its expected calibration error is also substantially lower than theirs. This more reliable confidence helps explain why dSpark can commit tokens aggressively without the same loss of accuracy.

## 5.3 ABLATION STUDY

Ablation Study on Training Recipe. To examine the contribution of each component, we ablate schedule distillation and KL regularization on LLaDA, with results shown in Table 2. Using ordinary trajectory training without optimized schedule and KL, it achieves the accuracy of 74.9 and TPF of 7.6 on GSM8K, and 37.9 accuracy and 5.2 TPF on Humaneval. Using the schedule alone substantially increases TPF on both GSM8K and HumanEval, but accuracy declines as the model commits more tokens in parallel. KL regularization alone has the opposite effect: it improves accuracy but reduces TPF, leading to lower AUP than the variant without either component. Combining the two gives the best balance. The full recipe retains 10.1 TPF on GSM8K and 6.7 TPF on HumanEval while raising accuracy to 78.9% and 39.6%, respectively. It achieves the highest AUP on both tasks. These results show that the schedule is responsible for the parallelism gain, while KL regularization helps prevent that gain from coming at the expense of accuracy.

Tokens per forward (TPF)  
![](images/9d7d1cb0dbb7897f436faa01efdd6bcb571a427e5a1b43533b57f4c580da88d4.jpg)  
Figure 4: dSpark extends the accuracy–parallelism frontier on every benchmark and backbone. Accuracy versus TPF for LLaDA (top) and Dream (bottom). Markers are the operating points of Table 1. Each curve connects a method’s accuracy at TPF = 1 to its operating point, the two points that make AUP. A flat curve means the method gains parallelism at little cost in accuracy, and a steep drop means it trades accuracy for speed. The dashed gray line is the Pareto frontier.

Table 2: Ablation study on different training recipes of dSpark on the LLaDA-8B-Instruct.
<table><tr><td rowspan="2">Schedule</td><td rowspan="2">KL Regularization</td><td colspan="3">GSM8K</td><td colspan="3">HumanEval</td></tr><tr><td>TPF ↑</td><td>Acc ↑</td><td>AUP↑</td><td>TPF ↑</td><td>Acc ↑</td><td>AUP↑</td></tr><tr><td></td><td></td><td>7.6</td><td>74.9</td><td>463.4</td><td>5.2</td><td>37.9</td><td>82.2</td></tr><tr><td>√</td><td></td><td>11.4</td><td>74.2</td><td>718.0</td><td>8.3</td><td>35.1</td><td>105.8</td></tr><tr><td></td><td>√</td><td>6.3</td><td>78.0</td><td>458.4</td><td>3.7</td><td>40.2</td><td>74.3</td></tr><tr><td>√</td><td>√</td><td>10.1</td><td>78.9</td><td>761.7</td><td>6.7</td><td>39.6</td><td>106.2</td></tr></table>

## Ablation study on KL regularization data.

To validate our choice of training mixture, we ablate the ratio of correct to incorrect trajectories on LLaDA and evaluate the resulting models on GSM8K, as shown in Table 3. Without incorrect trajectories, the model reaches 11.4 TPF but only 74.19% accuracy. Adding incorrect samples improves accuracy, although a larger share generally reduces parallelism. We choose a 4:1 ratio because it retains 10.12 TPF while raising accuracy to 78.94%. Increasing the share further to 1:1 brings only a small gain in accuracy, while TPF falls to 7.90. At 2:1, both accuracy and TPF are lower than at 4:1, reaching 78.04% and 9.77, respectively. We also test using a separate, disjoint set of correct trajectories for KL regularization, keeping the same ratio of 4:1. Compared with using incorrect trajectories at the same ratio, accuracy falls from 78.94% to 77.10%, and TPF drops by 30.5%. Together, these results support using incorrect trajectories for KL regularization at a 4:1 ratio, which provides a strong balance between accuracy and parallel decoding.

Table 3: Accuracy and TPF on LLaDA GSM8K with different ratios of correct to incorrect training trajectories.
<table><tr><td>Data Ratio</td><td>Acc (%) ↑</td><td>TPF↑</td></tr><tr><td>No inc. data</td><td>74.19</td><td>11.4</td></tr><tr><td>16:1</td><td>75.89</td><td>11.15</td></tr><tr><td>8:1</td><td>76.80</td><td>10.92</td></tr><tr><td>4:1</td><td>78.94</td><td>10.12</td></tr><tr><td>2:1</td><td>78.04</td><td>9.77</td></tr><tr><td>1:1</td><td>79.15</td><td>7.90</td></tr></table>

## 6 CONCLUSION

In this work, we studied efficient decoding for diffusion large language models from the perspective of when to commit predicted tokens. We showed that token confidence alone is insufficient for effective scheduling, since some less predictable tokens can provide more useful context for resolving the remaining sequence. We introduce dSpark enabled by schedule distillation, which constructs informative unmasking schedules based on the downstream utility of token commitments and distills these schedules into the model for efficient parallel inference. Experiments on mathematical reasoning and code generation show substantial acceleration, including 10.12× and 5.12× speedups on GSM8K for LLaDA-8B-Instruct and Dream-7B-Instruct, respectively, while maintaining strong task performance. Our results highlight unmasking schedules as an important learnable component of diffusion language modeling and suggest that better commitment strategies can substantially narrow the gap between the theoretical parallelism and practical efficiency of dLLMs.

## ETHICS STATEMENT

This work focuses on improving the inference efficiency of diffusion large language models and does not involve human subjects or the collection of sensitive personal data. Our method does not introduce new model capabilities beyond those of the underlying pretrained models and therefore inherits their existing limitations and potential risks, including biased, harmful, or inaccurate generations. By reducing the number of decoding steps required during inference, the proposed approach can also help lower computational cost and energy consumption.

## REPRODUCIBILITY STATEMENT

We provide detailed descriptions of the proposed method, training objectives, decoding procedures, experimental settings, and evaluation protocols in the main paper and supplementary material. We release the implementation code and relevant configurations to facilitate reproducibility.

## AI USE STATEMENT

Generative AI tools were used only for language polishing and improving the clarity of the manuscript. All technical content, experimental design, analysis, and conclusions were developed and verified by the authors.

## REFERENCES

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Marianne Arriola, Aaron Gokaslan, Justin Chiu, Zhihan Yang, Zhixuan Qi, Jiaqi Han, Subham Sahoo, and Volodymyr Kuleshov. Block diffusion: Interpolating between autoregressive and diffusion language models. In International Conference on Learning Representations, volume 2025, pp. 50726–50753, 2025.

Jacob Austin, Daniel D Johnson, Jonathan Ho, Daniel Tarlow, and Rianne Van Den Berg. Structured denoising diffusion models in discrete state-spaces. Advances in neural information processing systems, 34:17981–17993, 2021a.

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021b.

Heli Ben-Hamu, Itai Gat, Daniel Severo, Niklas S Nolte, and Brian Karrer. Accelerated sampling from masked diffusion models via entropy bounded unmasking. Advances in Neural Information Processing Systems, 38:55981–56007, 2025.

Tiwei Bie, Maosong Cao, Kun Chen, Lun Du, Mingliang Gong, Zhuochen Gong, Yanmei Gu, Jiaqi Hu, Zenan Huang, Zhenzhong Lan, et al. Llada2. 0: Scaling up diffusion language models to 100b. arXiv preprint arXiv:2512.15745, 2025.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

Zigeng Chen, Gongfan Fang, Xinyin Ma, Ruonan Yu, and Xinchao Wang. Dmax: Aggressive parallel decoding for dllms. arXiv preprint arXiv:2604.08302, 2026a.

Zigeng Chen, Gongfan Fang, Xinyin Ma, Ruonan Yu, and Xinchao Wang. dparallel: Learnable parallel decoding for dllms. In International Conference on Learning Representations, volume 2026, pp. 53193–53211, 2026b.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Justin Deschenaux and Caglar Gulcehre. Beyond autoregression: Fast llms via self-distillation through time. In International Conference on Learning Representations, volume 2025, pp. 5007– 5045, 2025.

Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. The language model evaluation harness, 07 2024. URL https://zenodo.org/records/12608602.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300, 2020.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. NeurIPS, 2021.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.

Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.

Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. Advances in neural information processing systems, 35:22199–22213, 2022.

Jia LI, Edward Beeching, Lewis Tunstall, Ben Lipkin, Roman Soletskyi, Shengyi Costa Huang, Kashif Rasul, Longhui Yu, Albert Jiang, Ziju Shen, Zihan Qin, Bin Dong, Li Zhou, Yann Fleureau, Guillaume Lample, and Stanislas Polu. Numinamath. [https://huggingface. co/AI-MO/NuminaMath-CoT](https://github.com/project-numina/ aimo-progress-prize/blob/main/report/numina\_dataset.pdf), 2024.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In International Conference on Learning Representations, volume 2024, pp. 39578–39601, 2024.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.

Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete diffusion modeling by estimating the ratios of the data distribution. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp (eds.), Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 32819–32848. PMLR, 21–27 Jul 2024. URL https://proceedings.mlr. press/v235/lou24a.html.

Xinyin Ma, Runpeng Yu, Gongfan Fang, and Xinchao Wang. dkv-cache: The cache for diffusion language models. Advances in Neural Information Processing Systems, 38:149009–149033, 2025a.

Yuxin Ma, Lun Du, Lanning Wei, Kun Chen, Qian Xu, Kangyu Wang, Guofeng Feng, Guoshan Lu, Lin Liu, Xiaojing Qi, et al. dinfer: An efficient inference framework for diffusion language models. arXiv preprint arXiv:2510.08666, 2025b.

Quan Nguyen-Tri, Mukul Ranjan, and Zhiqiang Shen. Attention is all you need for kv cache in diffusion llms. In International Conference on Learning Representations, volume 2026, pp. 34915– 34946, 2026.

Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Large language diffusion models. Advances in Neural Information Processing Systems, 38:50608–50646, 2025.

Jingyang Ou, Shen Nie, Kaiwen Xue, Fengqi Zhu, Jiacheng Sun, Zhenguo Li, and Chongxuan Li. Your absorbing discrete diffusion secretly models the conditional distributions of clean data. In International Conference on Learning Representations, volume 2025, pp. 64972–65009, 2025.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35: 27730–27744, 2022.

Yu-Yang Qian, Junda Su, Lanxiang Hu, Peiyuan Zhang, Zhijie Deng, Peng Zhao, and Hao Zhang. d3llm: Ultra-fast diffusion llm using pseudo-trajectory distillation. arXiv preprint arXiv:2601.07568, 2026.

Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. Zero: Memory optimizations toward training trillion parameter models. In SC20: International Conference for High Performance Computing, Networking, Storage and Analysis, pp. 1–16, 2020. doi: 10.1109/SC41405. 2020.00024.

Subham S Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin T Chiu, Alexander Rush, and Volodymyr Kuleshov. Simple and effective masked diffusion language models. Advances in Neural Information Processing Systems, 37:130136–130184, 2024.

Subham Sekhar Sahoo, Justin Deschenaux, Aaron Gokaslan, Guanghan Wang, Justin T Chiu, and Volodymyr Kuleshov. The diffusion duality. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaff, and Jerry Zhu (eds.), Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 52584–52619. PMLR, 13–19 Jul 2025. URL https://proceedings.mlr.press/v267/sahoo25a.html.

Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

Xu Wang, Chenkai Xu, Yijie Jin, Jiachun Jin, Hao Zhang, Kai Yu, and Zhijie Deng. Diffusion llms can do faster-than-ar inference via discrete diffusion forcing. In International Conference on Learning Representations, volume 2026, pp. 2282–2299, 2026.

Chengyue Wu, Hao Zhang, Shuchen Xue, Shizhe Diao, Yonggan Fu, Zhijian Liu, Pavlo Molchanov, Ping Luo, Song Han, and Enze Xie. Fast-dllm v2: Efficient block-diffusion llm. In International Conference on Learning Representations, volume 2026, pp. 128353–128370, 2026a.

Chengyue Wu, Hao Zhang, Shuchen Xue, Zhijian Liu, Shizhe Diao, Ligeng Zhu, Ping Luo, Song Han, and Enze Xie. Fast-dllm: Training-free acceleration of diffusion llm by enabling kv cache and parallel decoding. In International Conference on Learning Representations, volume 2026, pp. 57027–57051, 2026b.

Jiacheng Ye, Zhihui Xie, Lin Zheng, Jiahui Gao, Zirui Wu, Xin Jiang, Zhenguo Li, and Lingpeng Kong. Dream 7b: Diffusion large language models. arXiv preprint arXiv:2508.15487, 2025.

Yushi Ye, Xu Chen, Haoyun Jiang, Jinsong Lan, Haihong Tang, Bo Han, Ivor Tsang, Yanfeng Wang, Bo Zheng, and Jiangchao Yao. Ripple-pivot search: Active parallel decoding for diffusion large language models. arXiv preprint arXiv:2608.11742, 2026.

Huaye Zeng, Dongfu Jiang, Haozhe Wang, Ping Nie, Xiaotong Chen, and Wenhu Chen. ACE-CODER: Acing coder RL via automated test-case synthesis. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.), Proceedings ofthe 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 12023–12040, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176- 251-0. doi: 10.18653/v1/2025.acl-long.587. URL https://aclanthology.org/2025. acl-long.587/.

Haohui Zhang, Zhiye Wang, Xiaoying Gan, Xinbing Wang, and Bo Jiang. Leap: Unlocking dllm parallelism via lookahead early-convergence token detection. arXiv preprint arXiv:2605.10980, 2026.

Fengqi Zhu, Rongzhen Wang, Shen Nie, Xiaolu Zhang, Chunwei Wu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Llada 1.5: Variance-reduced preference optimization for large language diffusion models. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 11425–11460, 2026.

## APPENDIX

## A ADDITIONAL RESULTS

## A.1 WALL-CLOCK THROUGHPUT AND KV CACHING

TPF counts forward passes, but a forward pass with a longer context costs more time. We measure wall-clock throughput on NVIDIA A100-SXM4 GPUs with batch size 1. Table 4 therefore reports the wall-clock throughput of dSpark on LLaDA, without a KV cache and with the prefix and dual caches of Fast-dLLM (Wu et al., 2026b). dSpark runs with both caches. With the short 0-shot prompt, the caches barely change throughput (253 and 213 tokens/s, against 249 without a cache), since they also increase the number of forward passes (from 24.8 to 29.4 with the prefix cache). With a 5-shot prompt, the prefix cache raises throughput from 92 to 155 tokens/s (1.7×) at similar accuracy. At 512 tokens, the caches lower accuracy by 1.1–2.7 points without raising throughput. Caching is therefore most useful for long prompts, and dSpark reaches 249 tokens/s without it.

We also implement a static KV cache that runs every decoding step as a captured CUDA graph. It keeps the keys and values of the prompt and of committed tokens and, at each step, recomputes only the active window, the block behind it and a rotating slice of 64 cached positions, so that every cached position is refreshed periodically. With this cache, dSpark generates 313 tokens/s at 79.2% accuracy with the 0-shot prompt, 260 tokens/s with the 5-shot prompt and 300 tokens/s at 512 tokens, 1.3–2.8× its throughput without a cache, with accuracy within 1.3 points of it (Table 4).

Table 4: Wall-clock throughput of dSpark with KV caching (LLaDA-8B-Instruct, GSM8K, batch size 1). Prefix and Dual are the caches of Fast-dLLM (Wu et al., 2026b), and Static is our CUDAgraph cache. TPS: tokens per second.
<table><tr><td>Setting</td><td>KV cache</td><td>Acc.↑</td><td>TPF↑</td><td>TPS↑</td></tr><tr><td rowspan="4">0-shot, 256 tokens</td><td>None</td><td>78.94</td><td>10.12</td><td>249</td></tr><tr><td>Prefix</td><td>77.86</td><td>8.52</td><td>253</td></tr><tr><td>Dual</td><td>78.17</td><td>7.94</td><td>213</td></tr><tr><td>Static (ours)</td><td>79.23</td><td>9.51</td><td>313</td></tr><tr><td rowspan="4">5-shot, 256 tokens</td><td>None</td><td>79.68</td><td>9.60</td><td>92</td></tr><tr><td>Prefix</td><td>80.29</td><td>8.07</td><td>155</td></tr><tr><td>Dual</td><td>80.97</td><td>7.45</td><td>142</td></tr><tr><td>Static (ours)</td><td>79.08</td><td>8.53</td><td>260</td></tr><tr><td rowspan="4">0-shot, 512 tokens</td><td>None</td><td>84.00</td><td>11.69</td><td>168</td></tr><tr><td>Prefix</td><td>81.35</td><td>7.94</td><td>153</td></tr><tr><td>Dual</td><td>82.94</td><td>7.28</td><td>160</td></tr><tr><td>Static (ours)</td><td>82.71</td><td>9.86</td><td>300</td></tr></table>

Table 5 shows that dSpark’s gains in decoding parallelism translate into faster inference at batch size 1. On GSM8K, it achieves 249 tokens per second (TPS), a 16.6× improvement over the base LLaDA model and 13.2% higher throughput than d3LLM. It is also the most accurate method in the table, scoring 78.94% compared with 78.82% for LLaDA and 74.21% for d3LLM. Thus, dSpark’s higher TPF corresponds to a wall-clock speedup while preserving the backbone’s task performance.

Table 5: Accuracy and wall-clock throughput against the baselines on GSM8K for LLaDA models with batch size 1.
<table><tr><td>Model</td><td>Acc.↑</td><td>TPF↑</td><td>TPS↑</td></tr><tr><td>LLaDA</td><td>78.82</td><td>1.00</td><td>15</td></tr><tr><td>Fast-dLLM</td><td>75.51</td><td>2.91</td><td>94</td></tr><tr><td>D2F</td><td>74.24</td><td>2.83</td><td>89</td></tr><tr><td>dParallel</td><td>75.62</td><td>7.09</td><td>165</td></tr><tr><td>d3LLM</td><td>74.21</td><td>9.05</td><td>220</td></tr><tr><td>dSpark</td><td>78.94</td><td>10.12</td><td>249</td></tr></table>

## A.2 GENERATION LENGTH

Table 6: The gains of dSpark hold across generation lengths. Accuracy (%) and TPF on LLaDA for maximum generation length L. Bold: best in each column.
<table><tr><td rowspan="2"></td><td rowspan="2">Method</td><td colspan="2"> $L = 1 2 8$ </td><td colspan="2"> $L = 2 5 6$ </td><td colspan="2"> $L = 5 1 2$ </td></tr><tr><td>Acc.↑</td><td>TPF↑</td><td>Acc.↑</td><td>TPF↑</td><td>Acc.↑</td><td>TPF↑</td></tr><tr><td rowspan="4">GSM8K</td><td>LLaDA</td><td>72.86</td><td>1.00</td><td>78.82</td><td>1.00</td><td>82.26</td><td>1.00</td></tr><tr><td>dParallel</td><td>69.53</td><td>6.28</td><td>75.62</td><td>7.09</td><td>80.89</td><td>8.66</td></tr><tr><td>d3LLM</td><td>68.06</td><td>8.31</td><td>74.21</td><td>9.05</td><td>78.92</td><td>10.73</td></tr><tr><td>dSpark</td><td>73.31</td><td>8.85</td><td>78.94</td><td>10.12</td><td>84.00</td><td>11.69</td></tr><tr><td rowspan="4">MATH</td><td>LLaDA</td><td>30.86</td><td>1.00</td><td>31.74</td><td>1.00</td><td>34.26</td><td>1.00</td></tr><tr><td>dParallel</td><td>27.48</td><td>2.78</td><td>30.31</td><td>3.16</td><td>30.36</td><td>3.39</td></tr><tr><td>d3LLM</td><td>28.62</td><td>4.93</td><td>30.34</td><td>5.75</td><td>30.46</td><td>6.28</td></tr><tr><td>dSpark</td><td>30.50</td><td>4.96</td><td>32.03</td><td>6.03</td><td>35.64</td><td>6.79</td></tr></table>

Effect of Generation Length. Table 6 compares performance at maximum generation lengths of 128, 256, and 512 tokens. dSpark achieves the highest TPF in every setting and the highest accuracy in five of six settings. On GSM8K, it matches or exceeds the base LLaDA model’s accuracy at every length while increasing TPF from 8.85 at $L = 1 2 8$ to 11.69 at $L = 5 1 2$ . On MATH, dSpark is more accurate than both accelerated baselines at all three lengths and reaches the highest accuracy overall at $L = 2 5 6$ and $L = 5 1 2$ . Its advantage is especially clear at $L = 5 1 2$ , where it achieves 35.64% accuracy and 6.79 TPF, compared with 30.46% and 6.28 TPF for d3LLM. These results show that dSpark’s gains in parallelism persist as the generation length increases, while accuracy remains strong across both tasks.

## A.3 GENERAL KNOWLEDGE

General Knowledge. We evaluate dSpark on MMLU (Hendrycks et al., 2020) and ARC-C (Clark et al., 2018) to check whether training for faster decoding affects performance beyond math and code. As shown in Table 7, dSpark matches the base LLaDA model’s 63.0% accuracy on MMLU and scores 47.2% on ARC-C, which is only 0.4 drop from base model. This suggests that dSpark largely preserves the backbone’s general knowledge performance.

Table 7: General knowledge is preserved. Accuracy (%) of LLaDA-8B-Instruct before and after dSpark training.
<table><tr><td></td><td>LLaDA</td><td>dSpark</td></tr><tr><td>MMLU</td><td>63.0</td><td>63.1</td></tr><tr><td>ARC-C</td><td>47.6</td><td>47.2</td></tr></table>

## B TRAINING DETAILS

We fine-tune both backbones with LoRA (Hu et al., 2021) on all attention and feed-forward projections, keeping the pretrained weights frozen, and merge the adapters into the backbone before evaluation. Table 9 lists the configuration. The two backbones share every optimization setting and differ only in their training corpora. We train each model on four NVIDIA A100-SXM4 GPUs with data parallelism under DeepSpeed ZeRO stage 2 (Rajbhandari et al., 2020), without CPU offloading.

## C EVALUATION DETAILS

Decoding for dSpark. We use a multi-block decoding strategy (Qian et al., 2026) with a confidence threshold of 0.90 to 0.95 (Wu et al., 2026b), which enables decoding both the current and next blocks in parallel. All models decode greedily with at most 256 new tokens and a block size of 32. A new block enters the window once the last active block is 50% decoded and becomes fully active once its predecessor is 50% decoded. We use no KV cache in the main results.

Table 8: dSpark is less often confident when it is wrong. Top-1 predictions at every masked position and decoding round along the natural schedules of 250 GSM8K samples.
<table><tr><td></td><td>Base model</td><td>dParallel</td><td>d3LLM</td><td>dSpark</td></tr><tr><td>Correct predictions with  $p \geq 0 . 9 ( \% ) \uparrow$ </td><td>51.6</td><td>74.4</td><td>74.3</td><td>76.1</td></tr><tr><td>Wrong predictions with  $p \geq 0 . 9 ( \% ) \downarrow$ </td><td>2.5</td><td>21.7</td><td>22.9</td><td>15.5</td></tr><tr><td>Predictions with  $p \geq 0 . 9$  that are correct (%)↑</td><td>97.6</td><td>81.7</td><td>82.2</td><td>90.8</td></tr><tr><td>Expected calibration error (%)↓</td><td>4.6</td><td>19.2</td><td>20.4</td><td>8.2</td></tr></table>

![](images/e4666d903e8f6c210b5b6737b21faf0fde4d0c6c24cf9dd956da6783ff800b1e.jpg)  
Figure 5: dSpark stays much closer to calibration than the baselines it is compared with. Reliability diagrams for the predictions of Table 8 in 10 equal-width confidence bins. Top: accuracy per bin. The dashed diagonal marks perfect calibration, and the red hatched area is the gap between confidence and accuracy. Bottom: share of predictions in each bin.

Metrics. Tokens per forward (TPF) is the number of generated tokens, excluding end-of-sequence padding, divided by the number of forward passes, both summed over the benchmark. Accuracy under parallelism (AUP) (Qian et al., 2026) integrates accuracy over parallelism. For points $( \rho _ { 1 } , y _ { 1 } ) , \ldots , ( \rho _ { n } , y _ { n } )$ sorted by TPF $\rho ,$ with accuracy y, and keeping only points with $y _ { i } \geq y _ { 1 } - 5$

$$
\mathrm { A U P } = \rho _ { 1 } y _ { 1 } + \sum _ { i = 2 } ^ { n } \frac { \rho _ { i } - \rho _ { i - 1 } } { 2 } \big ( y _ { i } W ( y _ { i } ) + y _ { i - 1 } W ( y _ { i - 1 } ) \big ) , \qquad W ( y ) = \operatorname* { m i n } \Bigl ( e ^ { - \alpha ( 1 - y / y _ { \operatorname* { m a x } } ) } , 1 \Bigr ) ,\tag{10}
$$

where $y _ { \mathrm { m a x } }$ is the highest accuracy of any method on the benchmark and $\alpha = 3$ . Each method contributes its one-token-per-forward point $( \rho = 1 )$ and its reported operating point.

## D ADDITIONAL ABLATIONS

## D.1 CANDIDATE FILTER

A promoted token is committed before the frozen model would commit it on its own, so it is, by construction, a token the model is not yet confident about. The filter in Eq. (5) therefore does not ask for confidence. It only asks that the reference token be plausible to the model: among its $k = 1 0$ most likely tokens at that position, with a probability of at least $p _ { \mathrm { m i n } } = 0 . 0 2 , \mathrm { i . e . }$ , a surprisal of at most about 4 nats, the range shown in Figure 1. The two bounds complement each other. When the model’s distribution is flat, many tokens exceed 2% and the rank bound applies; when it is peaked, the tenth-ranked token can have a negligible probability and the probability bound applies.

Table 9: Training configuration of dSpark.
<table><tr><td colspan="2">LLaDA-8B-Instruct</td><td>Dream-7B-Instruct</td></tr><tr><td>LoRA rank r / α</td><td colspan="2">256 / 256</td></tr><tr><td>LoRA target modules</td><td>q_proj, k_proj, v_proj, attn_out, ff_proj, up_proj, ff_out</td><td>q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj</td></tr><tr><td>Optimizer</td><td colspan="2">AdamW (Loshchilov &amp; Hutter, 2017)</td></tr><tr><td>Learning rate (schedule) Weight decay / gradient clipping</td><td colspan="2"> $2 \times 1 0 ^ { - 5 }$  (constant) 0.01 / 1.0</td></tr><tr><td>Epochs</td><td colspan="2">6</td></tr><tr><td>Batch size</td><td colspan="2">64</td></tr><tr><td>Max. sequence length</td><td colspan="2">512</td></tr><tr><td>KL weight β (Eq. 9)</td><td colspan="2">1</td></tr><tr><td>Precision</td><td colspan="2">bfloat16</td></tr></table>

Why not a stricter filter. The tokens that help the other positions most are rarely the confident ones. As shown in Figure 1, revealing an uncertain token $( p _ { \theta _ { 0 } } < 0 . 5 )$ makes 5.1× more other positions ready than revealing a confident one and lowers their surprisal 4.7× more. A filter that asks for confidence would therefore exclude the most useful promotions. Promoting these low-probability tokens also does not cost accuracy (Table 10).

Why not a looser filter. Below these bounds, the model has effectively ruled the reference token out given the current context. Promoting such a token would make the schedule depend on knowing the verified answer rather than on the model’s own uncertainty, and would train the student to commit a token that its context does not support. The bounds also keep the search small, since every candidate is scored with its own forward pass in Eq. (6). We set $p _ { \mathrm { m i n } }$ and k from these considerations and did not tune them in end-to-end training.

To test whether low-probability promotions cost accuracy, we take 240 states at which the schedule promotes a token, 80 per probability range, and finish decoding either by committing the promoted token or by committing nothing. Table 10 shows that committing the promoted token leaves accuracy unchanged within sampling noise while reducing the forward passes needed to finish, by 6.5 for the least likely tokens.

Table 10: Committing a low-probability promoted token keeps accuracy and saves forward passes. From 240 states where our schedule promotes a token (80 per probability range), we finish decoding with (Promoted) or without (Not promoted) committing it.
<table><tr><td rowspan="2">Probability of promoted token</td><td colspan="2">Accuracy (%)</td><td colspan="2">Forward passes to finish</td></tr><tr><td>Promoted</td><td>Not promoted</td><td>Promoted</td><td>Not promoted</td></tr><tr><td>[0.02, 0.1)</td><td>80.0</td><td>80.0</td><td>31.6</td><td>38.1</td></tr><tr><td>[0.1, 0.3)</td><td>85.0</td><td>86.3</td><td>24.8</td><td>28.6</td></tr><tr><td>[0.3, 1]</td><td>86.3</td><td>83.8</td><td>25.1</td><td>27.7</td></tr></table>

## D.2 TOKEN PROMOTION STRATEGY.

We compare the proposed surprisal-reduction criterion Eq. (6) with the natural schedule, random promotion of a plausible token, and promotion of the highest-surprisal token on GSM8K as shown in Table 11. Without KL regularization, our criterion reaches 11.4 TPF, compared with 7.6 for the natural schedule and 8.3 for random promotion. Promoting the highest-surprisal token reaches 10.1 TPF but lowers accuracy to 73.3%. With KL regularization, our criterion achieves the highest TPF and accuracy, at 10.1 and 78.9%, respectively. It improves both measures over highest-surprisal promotion (8.9 TPF and 77.8% accuracy) and raises TPF substantially over random promotion (7.5) while maintaining similar accuracy. These results support selecting tokens for how much they sim plify the remaining predictions, rather than for their own surprisal alone.

Table 11: Effect of the token promotion strategy on LLaDA GSM8K, with and without KL regularization. Accuracy is reported as a percentage.
<table><tr><td rowspan="2">Promotion strategy</td><td colspan="2">Without KL</td><td colspan="2">With KL</td></tr><tr><td>TPF↑</td><td>Acc.↑</td><td>TPF↑</td><td>Acc.↑</td></tr><tr><td>Natural schedule</td><td>7.6</td><td>74.9</td><td>6.3</td><td>78.0</td></tr><tr><td>Random plausible token</td><td>8.3</td><td>75.1</td><td>7.5</td><td>78.7</td></tr><tr><td>Highest-surprisal token</td><td>10.1</td><td>73.3</td><td>8.9</td><td>77.8</td></tr><tr><td>Surprisal reduction (Eq. 6)</td><td>11.4</td><td>74.2</td><td>10.1</td><td>78.9</td></tr></table>

## D.3 KL WEIGHT

Sensitivity to the KL Weight. We vary the KL regularization weight $\beta$ on LLaDA and report the results in Table 12. Without KL regularization, the model reaches the highest TPF but has the lowest accuracy on both GSM8K and MATH-500. Increasing $\beta$ to 1 raises

accuracy from 74.19% to 78.94% on GSM8K and from 32.81% to 34.40% on MATH-500, while retaining 10.12 and 4.96 TPF, respectively. Raising β further to 2 adds only 0.16 accuracy on GSM8K while reducing TPF from 10.12 to 8.1. On MATH-500, accuracy remains unchanged and TPF also falls. We therefore use $\beta = \mathrm { i }$ for its stronger balance between accuracy and parallel decoding.

Table 12: Effect of the KL weight β on LLaDA accuracy and TPF for GSM8K and MATH-500. Bold indicates the selected setting.
<table><tr><td rowspan="2"> $\beta$ </td><td colspan="2">GSM8K</td><td colspan="2">MATH-500</td></tr><tr><td>Acc (%)↑</td><td>TPF↑</td><td>Acc (%)↑</td><td>TPF↑</td></tr><tr><td>0</td><td>74.19</td><td>11.4</td><td>32.81</td><td>6.03</td></tr><tr><td>0.5</td><td>75.74</td><td>10.8</td><td>33.22</td><td>5.36</td></tr><tr><td>1</td><td>78.94</td><td>10.12</td><td>34.40</td><td>4.96</td></tr><tr><td>2</td><td>79.10</td><td>8.1</td><td>34.40</td><td>4.61</td></tr></table>

## E QUALITATIVE EXAMPLE

In the following, we present the example of generation from our model and different baselines for LLaDA.

Prompt: A raspberry bush has 6 clusters of 20 fruit each and 67 individual fruit scattered across the bush. How many raspberries are there total? Reference answer: 187

## Base model (LLaDA-8B-Instruct)

To determine the total number of raspberries on the bush, we need to add the number of raspberries in the clusters to the number of individual raspberries scattered across the bush. Let’s break this down step by step: 1. Calculate the number of raspberries in the clusters: – There are 6 clusters. Each cluster contains 20 raspberries. – Therefore, the total number of raspberries in the clusters is: $6 \times 2 0 ~ = ~ 1 2 0 .$ 2. Add the number of individual raspberries scattered across the bush: – There are 67 individual raspberries scattered across the bush. – Therefore, the total number of raspberries is: $1 2 0 + 6 7 = 1 8 7 .$ So, the total number of raspberries on the bush is 187 .

Forward passes: 199 Tokens: 199 Correct

## d3LLM

To determine the total number of raspberries on the bush, we need to calculate the number of raspberries in the clusters and the number of individual raspberries separately, and then sum these amounts. First, let’s calculate the number of raspberries in the clusters: – There are 6 clusters. – Each cluster contains 20 raspberries. – Therefore, the total number of raspberries in the clusters is: 6 clusters × 20 raspberries per cluster = 120 raspberries. Next, we add the number of individual raspberries: – There are 67 individual raspberries scattered across the bush. – Therefore, the total number of raspberries is: 120 raspberries + 67 raspberries = 187 raspberries. Thus, the total number of raspberries on the bush is 187

Forward passes: 23 Tokens: 216 Correct

## dParallel

To determine the total number of raspberries on the bush, we need to add the number of raspberries in the clusters and the number of individual raspberries scattered across the bush. 1. First, calculate the number of raspberries in the clusters: – There are 6 clusters, and each cluster contains 20 raspberries. – Therefore, the number of raspberries in the clusters is: 6 × 20 = 120. 2. Next, add the number of individual raspberries scattered across the bush: – There are 67 individual raspberries scattered across the bush. – Therefore, the total number of raspberries is: $1 2 0 + 6 7 = 1 8 7$ Thus, the total number of raspberries on the bush is 187 .

Forward passes: 25 Tokens: 188 Correct

## dSpark (ours)

To determine the total number of raspberries on the bush, we need to follow these steps: 1. Calculate the number of raspberries in the clusters. 2. Add the number of individual raspberries to the number of raspberries in the clusters. First, let’s calculate the number of raspberries in the clusters: – There are 6 clusters. – Each cluster contains 20 raspberries. – Therefore, the total number of raspberries in the clusters is: 6 × 20 = 120. Next, we add the number of individual raspberries to the number of raspberries in the clusters: – There are 67 individual raspberries. – Therefore, the total number of raspberries is: 120 + 67 = 187. Thus, the total number of raspberries on the bush is 187

Forward passes: 17 Tokens: 202 Correct

Prompt: John wins an award at work. The award has a 1 time monetary reward of\$5000 and a 5% raise in salary. Ifhe makes 2000 a week, every week ofthe year and he got the award on thefirst day ofthe year how much money did he make that year? Reference answer: 114,200

## Base model (LLaDA-8B-Instruct)

To determine how much money John made that year, we need to consider both his one-time monetary reward and his annual salary after the 5% raise. 1. One-Time Monetary Reward: John receives a one-time monetary reward of \$5000. 2. Annual Salary After Raise: John’s weekly salary is \$2000. A 5% raise in his salary means he will earn: 2000 × 0.05 = 1000 dollars per week. Therefore, his annual salary is: 2000 × 52 = 104000 dollars. 3. Total Earnings for the Year: Adding the one-time monetary reward to his annual salary, we get: 5000+104000 = 109000 dollars. Thus, the total amount of money John made that year is 109000

Forward passes: 256 Tokens: 255 Wrong

## d3LLM

To determine how much money John made that year, we need to break down his earnings into two parts: the one-time monetary reward and his increased salary. 1. One-Time Monetary Reward: John receives a one-time monetary reward of \$5000. 2. Increased Salary: John receives a 5% raise in his salary. His original salary is \$2000 per week. – Calculate the raise amount: 2000 × 0.05 = 1000 – Add the raise to his original salary: 2000 + 1000 = 3000 – Calculate the total salary for the year: 3000 × 52 = 156000 3. Total Earnings for the Year: – Add the onetime monetary reward to the total salary for the year: 5000 + 16000 = 161000. Therefore, the total amount of money John made that year is \boxed{161000 [cut off at 256 tokens]

Forward passes: 46 Tokens: 256 Wrong

## dParallel

To determine how much money John made in the year, we need to consider both the one-time monetary reward and the percentage increase in his salary. 1. One-time Monetary Reward: John receives a one-time monetary reward of \$5000. 2. Percentage Increase in Salary: John receives a 5% raise in his salary. His original salary is \$2000 per week. 3. Total Annual Salary: There are 52 weeks in a year. Therefore, his annual salary is: 2000 × 52 = 104000. 4. Total Earnings for the Year: Adding the one-time monetary reward to his total annual salary: 5000 + 104000 = 109000. Thus, the total amount of money John made that year is 109000

Forward passes: 45 Tokens: 225 Wrong

## dSpark (ours)

To determine how much money John made that year, we need to break down the problem into two parts: the one-time monetary reward and the 5% raise in his salary. 1. One-Time Monetary Reward: John receives a one-time monetary reward of \$5000. 2. 5% Raise in Salary: John’s original salary is \$2000 per week. A 5% raise in his salary is: 0.05 × 2000 = 100. Therefore, his new salary is: 2000 + 100 = 2100. There are 52 weeks in a year, so his annual salary is: 2100×52 = 109200. Now, we add the one-time monetary reward to his annual salary to find the total amount he made: 5000 + 109200 = 114200 .

Forward passes: 28 Tokens: 255 Correct