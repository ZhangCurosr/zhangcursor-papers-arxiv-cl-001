# Measured Coupling for Parallel Decoding in Diffusion Language Models

Tal Oved Tel Aviv University

Lior Wolf Tel Aviv University

## Abstract

Diffusion language models can write many tokens per step, but must choose which tokens are safe to commit together. Mean-field decoding penalizes pairs of positions whose predictions look alike. We replace this proxy with the coupling itself: how much committing one token changes the model’s confidence at another. Cheap probes inside the main forward pass measure it, with no training or extra model, and it plugs into the same mean-field rule. On three models and four benchmarks, our method needs far fewer steps than mean-field decoding and is faster in most settings, also with a key-value cache, at an average accuracy cost of 1.1 points. Shuffling the couplings or replacing them with a uniform bonus loses this gain.

## 1 Introduction

Large language models (Achiam et al., 2023; Bai et al., 2023; Comanici et al., 2025; Liu et al., 2024) write text one token at a time. Each new token needs a full forward pass, so the time to answer grows with the length of the answer. Discrete diffusion language models (Austin et al., 2021a; Li et al., 2022; Nie et al., 2026; Zhu et al., 2026; Ye et al., 2025) offer a parallel alternative. They treat generation as iterative denoising: the answer starts fully masked, and every step can fill in several positions at once. Recent models scale this idea to strong, large systems (Bie et al., 2025; 2026), which makes fast parallel decoding a practical goal.

The hard part is deciding which tokens to commit together. A diffusion model predicts every masked position independently, from the same context. Two tokens can each look certain on their own and still form a wrong pair, because neither was predicted with the other in view (Ringel et al., 2026; Luo et al., 2026; Zhang et al., 2026). Confidence thresholds commit only positions whose top prediction is very likely (Wu et al., 2026). Dependency-aware decoders estimate which positions interact, from attention maps (Luo et al., 2026; Kim et al., 2026) or with a small trained predictor (Ringel et al., 2026). Mean-field parallel decoding (Zoabi et al., 2026) takes a training-free view based on an energy model. Each masked position gets a commit score, a pairwise term penalizes committing two positions whose predictive distributions are similar, measured by the Jensen-Shannon divergence, and a few mean-field updates decide which positions to commit.

In this work, we tackle the question of what pairwise quantity should this energy use. The quantity that matters is the effect of committing token j on the confidence of position i. We call it the coupling $\Delta _ { i j }$ (Figure 1). It can be positive, when j makes i less certain, or negative, when j supports i. Two positions with similar distributions may not affect each other at all, and two positions with different distributions may depend strongly on each other, for example the digits of one number.

The coupling can be measured with the model itself: reveal j, run the model, and read the new confidence at i. Done naively, this costs one forward pass per candidate. We make it cheap. A probe only needs to recompute the current block. The rest of the sequence reuses the keys and values from the step’s main forward pass. This approximation is accurate (Pearson 0.96 to 0.97 with exact probes) and reduces the work of a probe from the whole sequence to the masked positions of one block. We then run the probes inside the main forward pass of each step, so the model weights are read once for both.

![](images/194948b632fdafa853d0b87ad6d326cf6ed3be3605a508254fadfb83d5364fc2.jpg)

![](images/bbaf74874bfbf97bfd8eaac3b2010480a965b81b205e6cb8611663294c4fd3c7.jpg)  
Figure 1: Overview of measured coupling. Top: mean-field decoding penalizes two positions whose predicted distributions overlap (left). We instead measure how much revealing token j changes the margin of token i, which shows support as well as conflict (right). Bottom: one decoding step. The main tokens and K probe rows, one for each of the most confident candidates, run in a single forward pass. Each probe row reveals one candidate and recomputes only the masked positions, reading the main keys and values of the same layer everywhere else. The resulting margins give the coupling $\Delta _ { i j } = c _ { i } - c _ { i } ^ { ( j ) }$ , which enters the mean-field update, and positions with $q _ { i } \geq \tau$ are committed for the next step.

## Our contributions are:

• Measured coupling. We define the pairwise term of parallel decoding as the change in log-margin caused by committing another token, including support as well as conflict.

• Fused block-local probes. A probe recomputes only the masked positions of the current block against the keys and values of the main pass, and all probes run inside that same pass. This makes measuring the coupling cheap enough to pay off in wall-clock time.

• A reading of the mean-field rule. As an interpretation, under an additive model of conditional margins the mean-field fixed point with measured coupling equals the sigmoid of each token’s expected margin given the others (Proposition 1), and a standard contraction condition guarantees a unique fixed point (Proposition 2).

• Results. On LLaDA-8B, LLaDA-1.5, and Dream, measured coupling needs the fewest steps on every benchmark and has the shortest time per answer in most of them. Its accuracy is 1.1 points below mean-field decoding on average and ranges from 6.1 points lower to 1.8 points higher; the drop is significant on MATH for two of the three models, and the largest drops are on Dream HumanEval and LLaDA-1.5 MBPP.

## 2 Background

Decoding a masked diffusion model. The sequence x holds a prompt followed by masked positions. Decoding fills the answer block by block. At each step the model reads x and returns a distribution $\pi _ { i }$ for every masked position i in the current block. Let U be these positions. For each $i \in U$ we take the most likely token $v _ { i } =$ arg max $\pi _ { i }$ and its log-margin

$$
c _ { i } = \log \pi _ { i } ( v _ { i } ) - \operatorname* { m a x } _ { v \neq v _ { i } } \log \pi _ { i } ( v ) .\tag{1}
$$

The decoder commits a set $S \subseteq U$ , writes $v _ { S }$ into x, and repeats until the block is full. The number of decoding steps, called function evaluations (NFE), sets the cost.

Mean-field parallel decoding. Zoabi et al. (2026) attach a binary variable $s _ { i }$ to each $i \in U$ , with $s _ { i } = 1$ meaning commit. They define the energy

$$
E ( s ) = - \sum _ { i } c _ { i } s _ { i } + \sum _ { i < j } D _ { i j } s _ { i } s _ { j } ,\tag{2}
$$

where $D _ { i j }$ is one minus the normalized Jensen-Shannon divergence between $\pi _ { i }$ and $\pi _ { j }$ . A factorized distribution q with $q _ { i } = P ( s _ { i } = 1 )$ gives the mean-field update

$$
q _ { i }  \sigma \Big ( c _ { i } - \sum _ { j \neq i } D _ { i j } q _ { j } \Big ) ,\tag{3}
$$

run for two iterations from $q = \sigma ( c )$ . Positions with $q _ { i } \geq \tau$ are committed. This rule is training-free and fast, and it is the method we build on.

## 3 Measured Coupling

## 3.1 The coupling

Definition 1 (Coupling). For positions $i , j \in U ,$ , let $c _ { i } ^ { ( j ) }$ be the log-margin of the same token $v _ { i }$ at position i after revealing $x _ { j } = v _ { j }$ . The coupling ofj on i is

$$
\begin{array} { r } { \Delta _ { i j } = c _ { i } - c _ { i } ^ { ( j ) } . } \end{array}\tag{4}
$$

A positive $\Delta _ { i j }$ means that committing j lowers the confidence at $i ,$ so committing both together is risky. A negative $\Delta _ { i j }$ means that j supports i: once $j$ is known, i becomes more certain. The coupling is directional, since the effect of $j$ on i can differ from the effect of i on $j .$ . We keep its sign. Support is what allows many tokens to be committed in one step.

We plug the coupling into the same update as (3):

$$
q _ { i }  \sigma \Big ( c _ { i } - \sum _ { j \neq i } \Delta _ { i j } q _ { j } \Big ) .\tag{5}
$$

## 3.2 What the update computes

The next result gives a simple reading of this update. Write $c _ { i } ( x _ { S } )$ for the log-margin of $v _ { i }$ at position i after revealing the tokens $v _ { S }$ .

Proposition 1 (Expected margin). Assume that conditional margins are additive in the revealed tokens,

$$
c _ { i } ( x _ { S } ) = c _ { i } - \sum _ { j \in S } \Delta _ { i j } f o r a l l S \subseteq U \setminus \{ i \} .\tag{6}
$$

Let the other positions be committed with probabilities $\operatorname* { P r } ( s _ { j } = 1 ) = q _ { j }$ , independently or not. Then the expected margin ofi is

$$
\mathbb { E } _ { q } \left[ c _ { i } ( x _ { S } ) \right] = c _ { i } - \sum _ { j \neq i } \Delta _ { i j } q _ { j } ,\tag{7}
$$

and everyfixed point of (5) satisfies $q _ { i } = \sigma \big ( \mathbb { E } _ { q } [ c _ { i } ( x _ { S } ) ] \big ) f$ or all i.

Proof. Under $\begin{array} { r } { ( 6 ) , c _ { i } ( x _ { S } ) = c _ { i } - \sum _ { j \neq i } \Delta _ { i j } s _ { j } } \end{array}$ . Taking expectations and using $\mathbb { E } [ s _ { j } ] = q _ { j }$ gives the first claim, by linearity of expectation. A fixed point of (5) satisfies $q _ { i } = \sigma ( c _ { i } - \textstyle \sum _ { j } \Delta _ { i j } q _ { j } )$ , which is the sigmoid of this expectation. □

In words, each token’s commit probability becomes a function of how confident it is expected to be once the other likely commits are in place. With $\Delta$ in place of D, the mean-field rule of Zoabi et al. (2026) gains this direct reading. The similarity matrix D can be seen as a cheap stand-in for $\Delta$

The proposition interprets the update; it does not show that the committed set is safe. Since $\Delta$ is not symmetric, the update is a fixed-point rule with the mean-field form, and no longer solves a pairwise energy. The value $\sigma ( c _ { i } )$ compares only the top two tokens, so it is not a calibrated probability of being correct, and $\sigma ( \mathbb { E } [ c _ { i } ] )$ differs from $\mathbb { E } [ \sigma ( c _ { i } ) ]$ . The proposition also averages over random commits, while the decoder commits every position above τ after two iterations. Assumption (6) is a first-order model, and our probes give estimates of each $\Delta _ { i j }$ (Section 3.4), so additivity and estimation error both enter the final decision. We test additivity directly in Appendix J. For the sets the decoder commits, the additive prediction and the jointly measured margin fall on the same side of τ for 90% to 93% of the remaining positions, and for 95% to 99% when two or three tokens are committed. The sum overstates both support and conflict, and more so for large sets.

## 3.3 When the update has a unique answer

Proposition 2 (Contraction). Let $\begin{array} { r } { F ( q ) _ { i } = \sigma \big ( c _ { i } - \sum _ { j \neq i } \Delta _ { i j } q _ { j } \big ) } \end{array}$ and $\begin{array} { r } { \| \Delta \| _ { \infty } = \operatorname* { m a x } _ { i } \sum _ { j \neq i } | \Delta _ { i j } | . } \end{array}$ . If $\| \Delta \| _ { \infty } < 4$ , then $F$ is a contraction in the $\ell _ { \infty }$ norm. It has a uniquefixed point, and the iteration $q  F ( q )$ converges to it geometricallyfrom any start.

Proof. The sigmoid satisfies $| \sigma ^ { \prime } ( z ) | \le 1 / 4$ . By the mean value theorem, $| F ( q ) _ { i } - F ( q ^ { \prime } ) _ { i } | \ \leq$ $\begin{array} { r } { \frac { 1 } { 4 } \sum _ { j } | \Delta _ { i j } | | q _ { j } ^ { - } q _ { j } ^ { \prime } | \leq \frac { 1 } { 4 } \| \Delta \| _ { \infty } \| q - q ^ { \prime } \| _ { \infty } } \end{array}$ . The Banach fixed point theorem completes the proof.

When couplings are large, synchronous updates can swing between states. A simple example is two strongly conflicting tokens that rise together, fall together, and rise again. This explains why the number of mean-field iterations matters in practice (Zoabi et al., 2026). Like mean-field decoding, we run the update for two iterations at every decoding step. In practice the measured couplings are strong, and the contraction condition holds in about a quarter of the decoding steps. The condition is therefore sufficient and rarely met in practice. Running the update for 100 iterations changes the commit decision in about one step in seven. When the update oscillates, two iterations need not be close to a fixed point.

## 3.4 Cheap block-local probes

Measuring $\Delta _ { i j }$ directly needs one forward pass per candidate $j ,$ over the full sequence. We reduce this cost with the key-value cache. The main forward pass of a step already computes keys and values for every position. A probe for candidate $j$ writes $v _ { j }$ at position $j$ and recomputes only the positions of the current block that are still masked. All other positions, inside or outside the block, reuse the keys and values of the main pass. The probes for all candidates run as one batch.

This is the same approximation that Fast-dLLM uses for its dual cache (Wu et al., 2026). Here it is applied to probes. It treats the representations of all other positions as fixed while j is revealed. Section 4.6 shows that the resulting couplings agree closely with exact probes.

Probes inside the main pass. A separate probe call costs a fixed amount, mostly for reading the model weights, even when it holds few tokens. We remove this cost by running the probes inside the main forward pass of the next step. All token-wise layers (normalization, projections, MLP, and output head) process the main tokens and the probe tokens together, so each weight is read once. Only attention is split. The main tokens use the usual bidirectional attention. Each probe row attends to its own keys at the masked positions and to the main keys and values of the same layer everywhere else, including the block positions that are already committed. Since the main logits of a step are not known before the pass, the probes use the proposals $v _ { j }$ and the confidence ranking from the previous step, including the predictions for the next block at a block boundary. A probe whose token has changed since then is dropped. At the first step of an answer, the probes run in one extra call after the main pass, so an answer takes NFE + 1 calls. We probe the $K$ most confident candidates, which carry most of the weight $q _ { j }$ in the mean-field sum, and set $\Delta _ { i j } = 0$ for the others.

Algorithm 1 One decoding step with measured coupling   
Require: Sequence x, masked positions $\overline { { \mathcal { U } _ { t } } }$ of the current block, proposals $v _ { j }$ and margins $c _ { j }$ from   
the previous step, parameters τ , R, K   
Ensure: Commit set $\mathcal { C } _ { t }$   
$/ / l .$ Probe construction   
$\mathcal { P } $ the $K$ positions $j \in \mathcal { U } _ { t }$ with the largest $c _ { j }$   
$x ^ { ( j ) }$ ← the tokens at $\mathcal { U } _ { t }$ with $x _ { j } : = v _ { j }$ , for all $j \in \mathcal P$   
$/ / 2 .$ Oneforward passfor x and all probes   
$\{ \ell _ { i } \} , \{ \ell _ { i } ^ { ( j ) } \}  f _ { \theta } ( x , \{ x ^ { ( j ) } \} _ { j \in \mathcal { P } } )$ ▷ probes reuse the main keys and values outside U   
${ \dot { / / } } 3 .$ Local confidence   
$\pi _ { t , i } \gets$ softmax(ℓ<sub>i</sub>), $v _ { i } \gets$ arg max $\pi _ { t , i } ( \boldsymbol { v } )$ for all $i \in \mathcal { U } _ { t }$   
$c _ { i } \gets \log \pi _ { t , i } ( v _ { i } ) - \operatorname* { m a x } _ { v \neq v _ { i } }$ log $\pi _ { t , i } ( \boldsymbol { v } )$ , and $c _ { i } ^ { ( j ) }$ from $\ell _ { i } ^ { ( j ) }$ with the same token $v _ { i }$   
$/ / 4 .$ Measured coupling   
$\Delta _ { i j }  c _ { i } - c _ { i } ^ { ( j ) }$ for $j \in \mathcal { P }$ whose probed token is still $v _ { j } .$ , and $\Delta _ { i j }  0$ otherwise; $\Delta _ { i i }  0$   
$/ / { \check { 5 } } .$ Mean-field relaxation   
$q ^ { ( 0 ) }  \sigma ( \dot { c } )$   
for $r = 0$ to $R - 1$ do   
$q ^ { ( r + 1 ) } \gets \sigma \big ( c - \Delta q ^ { ( r ) } \big )$   
end for   
// 6. Thresholding   
$\mathcal { C } _ { t } \gets \{ i \in \mathcal { U } _ { t } : q _ { i } ^ { ( R ) } \geq \tau \}$ ; if it is empty, $\mathcal { C } _ { t } \gets .$ {arg max<sub>i</sub> $q _ { i } ^ { ( R ) } \}$   
return $\mathcal { C } _ { t }$

Algorithm 1 summarizes one step. The method adds no parameters and needs no training data. Code, including the fused probe kernels, exact configurations, evaluation and timing scripts, and per-example outputs, is available at https://github.com/TalOved1/measured-coupling.

## 4 Experiments

## 4.1 Setup

We use LLaDA-8B-Instruct (Nie et al., 2026) and the DAWN code base (Luo et al., 2026), which the mean-field paper also uses. We also evaluate LLaDA-1.5 (Zhu et al., 2026) with the same protocol, and Dream-v0-Instruct-7B (Ye et al., 2025), a model from a different family. We evaluate on GSM8K (Cobbe et al., 2021) (1,319 problems, 5-shot), MATH (Hendrycks et al., 2021) (5,000 problems, 4-shot with the Minerva prompts (Lewkowycz et al., 2022)), HumanEval (Chen et al., 2021) (164 problems, 0-shot), and MBPP (Austin et al., 2021b) (500 problems, 3-shot). The generation length is 256 tokens. MATH answers are checked for mathematical equivalence with math-verify, and HumanEval answers are post-processed with the sanitizer from the DAWN code base.

We compare against one-token-per-step decoding, Fast-dLLM with threshold 0.9 (Wu et al., 2026), DAWN (Luo et al., 2026), and mean-field decoding with the published LLaDA setting (block 20, $\tau = 0 . 8 5$ , two iterations) (Zoabi et al., 2026). Our method uses the same block size (20) and number of iterations (two) as mean-field decoding. All methods are greedy. Section 4.2 explains how we chose the remaining settings of our method.

We report accuracy, the average number of function evaluations (NFE) per answer, and the step reduction over one-token-per-step decoding (256/NFE). Throughput (tokens per second, TPS) depends on the GPU, so we measure it for all methods in the same job on the same A100 GPU. We compare accuracy with mean-field decoding on the same questions: we report the paired difference with a 95% confidence interval (paired Wald interval), and use a paired exact McNemar test where we call a difference significant. When an interval includes zero, the data cannot tell which method is more accurate; it does not show that they are equally accurate.

Table 1: Results on LLaDA-8B-Instruct. Step reduction is 256/NFE, relative to one-token-per-step decoding, and is not a runtime speedup. TPS (tokens per second) is measured for all methods of a benchmark in one job on the same GPU (GSM8K on its first 200 questions, MATH on 105 problems with 15 per subject, HumanEval and MBPP on the full test sets). Time is the wall-clock time per answer in seconds, from the same job. Bold marks the best value in each efficiency column. Ours is measured coupling, MF is mean-field decoding, and one token is one-token-per-step decoding. The last column is the accuracy difference from mean-field decoding in points, with a paired 95% confidence interval.
<table><tr><td>Bench.</td><td>Method</td><td>Acc. ↑</td><td>NFE↓</td><td>Step red. ↑</td><td>TPS ↑</td><td></td><td>Time (s) ↓ ∆ vs. MF [95% CI]</td></tr><tr><td rowspan="5">GSM8K</td><td>One token</td><td>77.94</td><td>256.0</td><td>1.0×</td><td>7.2</td><td>32.42</td><td></td></tr><tr><td>Fast-dLLM</td><td>78.17</td><td>78.5</td><td>3.3×</td><td>23.2</td><td>10.00</td><td></td></tr><tr><td>DAWN</td><td>77.79</td><td>55.8</td><td>4.6×</td><td>31.7</td><td>7.29</td><td></td></tr><tr><td>MF</td><td>76.57</td><td>65.4</td><td>3.9×</td><td>28.6</td><td>8.10</td><td></td></tr><tr><td>Ours</td><td>76.27</td><td>40.1</td><td>6.4×</td><td>40.3</td><td>5.86</td><td>-0.3 [-2.5, +1.9]</td></tr><tr><td rowspan="5">MATH</td><td>One token</td><td>33.82</td><td>256.0</td><td>1.0×</td><td>9.4</td><td>26.21</td><td></td></tr><tr><td>Fast-dLLM</td><td>33.36</td><td>97.0</td><td>2.6×</td><td>24.0</td><td>10.32</td><td></td></tr><tr><td>DAWN</td><td>32.30</td><td>72.6</td><td>3.5×</td><td>30.8</td><td>8.01</td><td></td></tr><tr><td>MF</td><td>32.02</td><td>77.2</td><td>3.3×</td><td>30.8</td><td>8.18</td><td></td></tr><tr><td>Ours</td><td>31.18</td><td>47.9</td><td>5.3×</td><td>41.5</td><td>6.05</td><td>-0.8 [-1.8, +0.2]</td></tr><tr><td rowspan="5">HumanEval DAWN</td><td>One token</td><td>41.46</td><td>256.0</td><td>1.0×</td><td>19.8</td><td>12.27</td><td></td></tr><tr><td>Fast-dLLM</td><td>42.68</td><td>78.9</td><td>3.2×</td><td>64.5</td><td>3.76</td><td></td></tr><tr><td></td><td>37.80</td><td>61.6</td><td>4.2×</td><td>80.6</td><td>2.97</td><td></td></tr><tr><td>MF</td><td>37.20</td><td>61.7</td><td>4.1×</td><td>80.4</td><td>2.93</td><td></td></tr><tr><td>Ours</td><td>39.02</td><td>40.3</td><td>6.4×</td><td>83.7</td><td>2.77</td><td>+1.8 [−4.1, +7.8]</td></tr><tr><td rowspan="5">MBPP</td><td>One token</td><td>30.00</td><td>256.0</td><td>1.0×</td><td>6.4</td><td>26.92</td><td></td></tr><tr><td>Fast-dLLM</td><td>30.20</td><td>70.6</td><td>3.6×</td><td>23.4</td><td>7.41</td><td></td></tr><tr><td>DAWN</td><td>31.00</td><td>50.5</td><td>5.1×</td><td>30.7</td><td>5.42</td><td></td></tr><tr><td>MF</td><td>30.20</td><td>51.6</td><td>5.0×</td><td>28.6</td><td>5.31</td><td></td></tr><tr><td>Ours</td><td>29.00</td><td>32.3</td><td>7.9×</td><td>31.7</td><td>4.00</td><td>-1.2 [−4.9, +2.5]</td></tr></table>

## 4.2 How we chose the settings

Our method has three settings: the threshold τ, the number of probes K, and whether the probes run in a separate call or inside the main pass. We follow the protocol of mean-field decoding (Zoabi et al., 2026): we choose the settings on the first 100 questions of the GSM8K training split, separately for each model family. The candidate with the highest held-out accuracy wins, and ties go to the candidate with fewer steps. This selects $K = 1 \bar { 2 }$ probes inside the main pass with $\tau = 0 . 9$ for LLaDA-8B-Instruct, which LLaDA-1.5 shares, and K = 20 with τ = 0.95 for Dream. As a check on code, we also scored the candidates on the validation split of MBPP (90 problems), which does not overlap with the MBPP or HumanEval test sets; on LLaDA the two lists overlap but differ, since the candidates changed during development. On Dream it selects the same setting. On LLaDA its best threshold is slightly higher, τ = 0.95, two problems ahead of τ = 0.9. Tables 13 and 14 in the appendix list every candidate scored on each set.

All baselines use their published settings. Mean-field decoding uses the LLaDA setting of Zoabi et al. (2026) (block 20, τ = 0.85, two iterations), and Fast-dLLM and DAWN use the settings of the DAWN code base. On Dream, mean-field decoding uses the default Dream setting of Zoabi et al. (2026) (block $2 0 , \tau = 0 . 9 5$ , two iterations), and Fast-dLLM and DAWN use the Dream settings of the DAWN code base. Their test results are therefore not tuned on the test sets either.

During development we also evaluated other variants on the test sets, and Section 4.4 reports several of them as analysis. The main table uses only the settings chosen by the rule above. The candidate list grew once during the project: we added the probe variant that recomputes only the masked positions, and applied the same rule again to the full list. All reported results use the setting selected from the full list.

Table 2: Results on LLaDA-1.5. Our method uses the setting chosen on LLaDA-8B-Instruct, with no new tuning. Columns and TPS measurement as in Table 1. Bold marks the best value in each efficiency column.
<table><tr><td>Bench.</td><td>Method</td><td>Acc. ↑</td><td>NFE↓</td><td>Step red. ↑</td><td>TPS ↑</td><td></td><td>Time (s) ↓ ∆ vs. MF [95% CI]</td></tr><tr><td rowspan="5">GSM8K</td><td>One token</td><td>80.59</td><td>256.0</td><td>1.0×</td><td>6.9</td><td>31.82</td><td></td></tr><tr><td>Fast-dLLM</td><td>81.12</td><td>75.4</td><td>3.4×</td><td>22.5</td><td>9.70</td><td></td></tr><tr><td>DAWN</td><td>81.35</td><td>54.4</td><td>4.7×</td><td>30.7</td><td>7.11</td><td></td></tr><tr><td>MF</td><td>81.35</td><td>64.1</td><td>4.0×</td><td>28.2</td><td>7.78</td><td></td></tr><tr><td>Ours</td><td>80.36</td><td>40.2</td><td>6.4×</td><td>37.3</td><td>5.93</td><td>-1.0 [-2.9, +0.9]</td></tr><tr><td rowspan="5">MATH</td><td>One token</td><td>33.96</td><td>256.0</td><td>1.0×</td><td>8.8</td><td>25.62</td><td></td></tr><tr><td>Fast-dLLM</td><td>33.96</td><td>95.6</td><td>2.7×</td><td>22.2</td><td>10.15</td><td></td></tr><tr><td>DAWN</td><td>32.20</td><td>72.7</td><td>3.5×</td><td>28.4</td><td>7.95</td><td></td></tr><tr><td>MF</td><td>33.28</td><td>77.3</td><td>3.3×</td><td>28.8</td><td>7.88</td><td></td></tr><tr><td>Ours</td><td>32.02</td><td>49.5</td><td>5.2×</td><td>37.4</td><td>6.09</td><td>-1.3[-2.3, −0.2]</td></tr><tr><td rowspan="5">HumanEval</td><td>One token</td><td>43.29</td><td>256.0</td><td>1.0×</td><td>7.1</td><td>11.77</td><td></td></tr><tr><td>Fast-dLLM</td><td>43.29</td><td>99.9</td><td>2.6×</td><td>18.3</td><td>4.56</td><td></td></tr><tr><td>DAWN</td><td>40.85</td><td>77.2</td><td>3.3×</td><td>23.5</td><td>3.58</td><td></td></tr><tr><td>MF</td><td>38.41</td><td>72.4</td><td>3.5×</td><td>25.2</td><td>3.29</td><td></td></tr><tr><td>Ours</td><td>40.24</td><td>44.4</td><td>5.8×</td><td>27.8</td><td>2.83</td><td>+1.8 [-3.4, +7.0]</td></tr><tr><td rowspan="5">MBPP</td><td>One token</td><td>37.40</td><td>256.0</td><td>1.0×</td><td>2.4</td><td>26.58</td><td></td></tr><tr><td>Fast-dLLM</td><td>37.40</td><td>42.1</td><td>6.1×</td><td>14.1</td><td>4.49</td><td></td></tr><tr><td>DAWN</td><td>35.80</td><td>30.0</td><td>8.5×</td><td>19.5</td><td>3.29</td><td></td></tr><tr><td>MF</td><td>39.80</td><td>38.3</td><td>6.7×</td><td>16.2</td><td>3.96</td><td></td></tr><tr><td>Ours</td><td>37.40</td><td>28.5</td><td>9.0×</td><td>18.4</td><td>3.58</td><td>-2.4 [−6.0, +1.2]</td></tr></table>

## 4.3 Main results

Table 1 shows the main results. Measured coupling needs the fewest steps on every benchmark, more than a third fewer than mean-field decoding. Its accuracy differs from mean-field decoding by −1.2 to +1.8 points, and every 95% confidence interval of the paired difference includes zero.

The step savings carry over to wall-clock time. On the same GPU, our method has the highest throughput of all methods on every benchmark, most of all on GSM8K, whose long prompts make the main pass dominate a step. Appendix C compares the methods at matched accuracy, with every threshold timed on the same GPU. On GSM8K and MBPP our method reaches a given accuracy in less time than mean-field decoding where both reach it, and on HumanEval the two curves cross. With the dual key-value cache of Fast-dLLM for all methods, our method at block 32 is the fastest on GSM8K and MBPP (Appendix D).

Figure 3 in Appendix E shows the full trade-off between accuracy and steps, sweeping the threshold of each method. On GSM8K, HumanEval, and MBPP our curve lies to the left of the mean-field curve: it reaches a given accuracy with far fewer steps, at accuracies up to 76.7% on GSM8K. On MBPP it also rises above the best mean-field point.

Table 2 repeats the comparison on LLaDA-1.5, with the setting chosen on LLaDA-8B-Instruct and no new tuning. Our method again needs the fewest steps on every benchmark, 1.3 to 1.6 times fewer than mean-field decoding. Its accuracy differs from mean-field decoding by −2.4 to +1.8 points. On MATH it is 1.3 points lower, and this difference is significant $( p = 0 . 0 2 )$ . Its throughput is 1.10 to 1.32 times that of mean-field decoding, and it is the highest of all methods on GSM8K, MATH, and HumanEval. On MBPP, DAWN is slightly faster (19.5 against 18.4 tokens per second).

Table 3 reports Dream-v0-Instruct-7B, with the setting calibrated on Dream. Our method needs 1.7 to 2.2 times fewer steps than mean-field decoding on every benchmark. It is less accurate than mean-field decoding on all four benchmarks, by 0.6 to 6.1 points. The largest drop is on HumanEval, where the confidence interval is wide (164 problems), and the drop on MATH (2.1 points) is significant $( p < 0 . 0 0 1 )$ ). It has the shortest time per answer on GSM8K, MATH, and HumanEval, and on MBPP DAWN is slightly faster. Tokens per second counts only tokens before the end-of-text token, and our answers end earlier (Appendix C), so our gain shows more in time per answer.

Table 3: Results on Dream-v0-Instruct-7B. Our method uses the setting calibrated on Dream (K = 20, τ = 0.95). Columns and measurement as in Table 1. Bold marks the best value in each efficiency column.
<table><tr><td>Bench.</td><td>Method</td><td>Acc. ↑</td><td>NFE↓</td><td>Step red. ↑</td><td>TPS ↑</td><td>Time (s) ↓</td><td>∆ vs. MF [95% CI]</td></tr><tr><td rowspan="5">GSM8K</td><td>One token</td><td>75.06</td><td>256.0</td><td>1.0×</td><td>4.8</td><td>27.17</td><td></td></tr><tr><td>Fast-dLLM</td><td>74.98</td><td>62.2</td><td>4.1×</td><td>19.1</td><td>6.71</td><td></td></tr><tr><td>DAWN</td><td>72.25</td><td>52.0</td><td>4.9×</td><td>22.3</td><td>5.79</td><td></td></tr><tr><td>MF</td><td>76.19</td><td>69.2</td><td>3.7×</td><td>17.0</td><td>7.41</td><td></td></tr><tr><td>Ours</td><td>75.13</td><td>41.4</td><td>6.2×</td><td>21.1</td><td>5.67</td><td>−1.1 [−3.2, +1.1]</td></tr><tr><td rowspan="5">MATH</td><td>One token</td><td>38.18</td><td>256.0</td><td>1.0×</td><td>11.2</td><td>22.37</td><td></td></tr><tr><td>Fast-dLLM</td><td>38.34</td><td>115.0</td><td>2.2×</td><td>23.5</td><td>10.65</td><td></td></tr><tr><td>DAWN</td><td>37.70</td><td>94.4</td><td>2.7×</td><td>27.9</td><td>8.82</td><td></td></tr><tr><td>MF</td><td>37.88</td><td>100.0</td><td>2.6×</td><td>21.6</td><td>9.03</td><td></td></tr><tr><td>Ours</td><td>35.76</td><td>51.9</td><td>4.9×</td><td>29.4</td><td>5.97</td><td>-2.1 [-3.2, −1.0]</td></tr><tr><td rowspan="5">HumanEval</td><td>One token</td><td>52.44</td><td>256.0</td><td>1.0×</td><td>16.3</td><td>10.44</td><td></td></tr><tr><td>Fast-dLLM</td><td>55.49</td><td>90.7</td><td>2.8×</td><td>38.8</td><td>3.83</td><td></td></tr><tr><td>DAWN</td><td>54.27</td><td>75.5</td><td>3.4×</td><td>46.2</td><td>3.16</td><td></td></tr><tr><td>MF</td><td>54.27</td><td>76.0</td><td>3.4×</td><td>35.7</td><td>3.21</td><td></td></tr><tr><td>Ours</td><td>48.17</td><td>40.1</td><td>6.4×</td><td>34.0</td><td>2.59</td><td>-6.1 [-13.2, +1.0]</td></tr><tr><td rowspan="5">MBPP</td><td>One token</td><td>54.00</td><td>256.0</td><td>1.0×</td><td>5.5</td><td>20.39</td><td></td></tr><tr><td>Fast-dLLM</td><td>56.00</td><td>63.6</td><td>4.0×</td><td>26.9</td><td>5.17</td><td></td></tr><tr><td>DAWN</td><td>56.80</td><td>42.9</td><td>6.0×</td><td>30.1</td><td>3.55</td><td></td></tr><tr><td>MF</td><td>54.20</td><td>77.2</td><td>3.3×</td><td>25.6</td><td>6.22</td><td></td></tr><tr><td>Ours</td><td>53.60</td><td>34.4</td><td>7.4×</td><td>18.6</td><td>3.79</td><td>−0.6 [−4.0, +2.8]</td></tr></table>

Calibration narrows the drops. We keep one setting per model family, chosen like mean-field decoding on GSM8K training questions, and τ trades steps for accuracy. Stricter thresholds narrow the drops and keep most of the step savings: on MATH, τ = 0.99 on Dream and 0.95 on LLaDA-1.5 are within 0.8 and 0.1 points of mean-field decoding (95% CI [−1.9, +0.3] and [−1.2, +0.9], so at most 2 points lower) with 39% and 29% fewer steps (Appendix G). On Dream HumanEval, $\tau = 0 . 9 7$ $( K = 1 2 )$ narrows the gap to 3.7 points (CI [−10.8, +3.5]) with 45% fewer steps (Appendices E and F).

## 4.4 Where the savings come from

Appendix B isolates the pairwise term at a common threshold. Our coupling needs clearly fewer steps than both the confidence-only rule and mean-field decoding, because it detects support: tokens that confirm each other are committed together. Table 4 tests whether the measured pairs matter, or whether any added support would do. Each control changes only the coupling of our method, on the same measured pairs, with all other settings fixed. Shuffling the measured values across pairs keeps their distribution and breaks accuracy, by 36 points on GSM8K and 20 on MBPP. A constant support bonus, tuned on held-out GSM8K questions, keeps accuracy and needs 44% to 46% more steps, and keeping the measured conflicts with a tuned constant support needs 53% to 72% more. The gain therefore comes from knowing which pairs support each other. The two signs work together: keeping only support collapses accuracy, and keeping only conflicts needs about as many steps as mean-field decoding. Averaging the two directions where both are measured costs 2 to 3 points (significant on MBPP), so its direction carries some information.

The coupling matters most when decoding is pushed hard. At about 34 steps on GSM8K, the confidence-only rule drops to 62% accuracy, while our method keeps 75% (Figure 3).

## 4.5 Ablations

Appendix H varies the number of probes and the block size on GSM8K. More probes save steps at the same accuracy. Larger blocks need more probes to keep the same share of candidates measured; with that, block 32 matches mean-field accuracy with far fewer steps, while block 64 loses accuracy. From

Table 4: Controls on the measured coupling, LLaDA-8B-Instruct, full test sets (accuracy % / NFE). Each row changes only the coupling of our method on the measured pairs. ∆ is the accuracy difference from our method with a paired 95% confidence interval. The constants $( b = 0 . 0 2$ and 0.05) were chosen on 100 GSM8K training questions. Appendix I defines each control.
<table><tr><td></td><td colspan="2">GSM8K</td><td colspan="2">MBPP</td></tr><tr><td>Coupling</td><td>Acc. / NFE</td><td>∆ [95% CI]</td><td>Acc. / NFE</td><td>∆ [95% CI]</td></tr><tr><td>Measured (ours)</td><td>76.27 / 40.1</td><td></td><td>29.00 / 32.3</td><td></td></tr><tr><td>Shuffled across pairs</td><td>40.11 / 44.7</td><td>-36.2[-39.1, -33.2]</td><td>8.60 / 30.9</td><td>-20.4[-24.2, -16.6]</td></tr><tr><td>Support only</td><td>53.37 / 39.8</td><td>-22.9[-25.7, -20.1]</td><td>15.40 / 29.3</td><td>-13.6 [-16.9, -10.3]</td></tr><tr><td>Conflict only</td><td>78.24 / 72.2</td><td>+2.0 [−0.2, +4.1]</td><td>32.20 / 51.9</td><td>+3.2 [+0.0, +6.4]</td></tr><tr><td>Partially symmetrized</td><td>74.22 / 37.4</td><td>-2.1 [−4.1, +0.0]</td><td>26.00 / 30.8</td><td>-3.0 [-5.5, -0.5]</td></tr><tr><td>Constant bonus</td><td>76.72 / 58.6</td><td>+0.5 [−1.7, +2.6]</td><td>29.80 / 46.4</td><td>+0.8 [−2.2, +3.8]</td></tr><tr><td>Conflicts + constant support</td><td>77.33 / 68.9</td><td>+1.1 [−1.1, +3.2]</td><td>32.80 / 49.3</td><td>+3.8 [+0.7, +6.9]</td></tr></table>

one to four mean-field iterations, our steps change little and no accuracy difference is significant, while mean-field decoding alternates between short and long schedules with the parity of R (Appendix L).

## 4.6 Probe fidelity and cost

Our probes agree closely with exact full-sequence probes. During greedy decoding on GSM8K and HumanEval, the two couplings have a Pearson correlation of 0.96 to 0.97, and they agree in sign on 97% of the pairs with a clear effect $( | \Delta _ { i j } | > 0 . 5 ;$ Appendix A). About half of the candidate pairs are measured (51% on GSM8K, 55% on MBPP; the rest keep $\Delta _ { i j } = 0 )$ , and 31% and 23% of probes are dropped because their token changed. On LLaDA, at a similar step budget, a confidence threshold commits three to four times as many tokens that are overturned by the rest of their commit set (Appendix K). The probes are also cheap. Exact probes cost about nine forward passes per step, and a separate probe call about half a pass, mostly for reading the weights. Inside the main pass, a step costs 1.2 to 1.5 times a mean-field step, and our method needs 26% to 55% fewer steps. A variant that learns the coupling from the decoder’s next step, with no probes, saves fewer steps (Appendix M).

## 5 Related Work

Confidence thresholds (Wu et al., 2026), entropy bounds, and dilated schedules (Ben-Hamu et al., 2025; Luxembourg et al., 2026) decide how many positions to commit at once. Mean-field decoding adds a similarity penalty and a variational update (Zoabi et al., 2026). DAWN and dependency-aware parallel decoding build a dependency graph from attention maps and choose an independent set (Luo et al., 2026; Kim et al., 2026). DEMASK trains a predictor of the total-variation effect of revealing a token and gives a guarantee on the joint error (Ringel et al., 2026); our coupling is close to this quantity, and its predictor is not public. AXON reveals supportive anchors on top of an existing decoder, approximating their value with attention (Abu Ayoub et al., 2026); it could run on top of ours, and our probes measure a similar quantity directly. PUNT tests independence with extra model calls and prunes every dependent token (Azangulov et al., 2026). Conditional influence is therefore not new; we add a training-free way to measure it online, with sign and direction, cheaply enough to coordinate parallel commits. Other work verifies drafts with an autoregressive model or with the model itself (Israel et al., 2026; Han et al., 2026), verifies drafted future steps (Pan et al., 2026; Shlomi et al., 2026), or remasks committed tokens (Wang et al., 2026); these ideas are complementary to ours. In speculative decoding, DCOCO trains a small module to score adjacent draft tokens (Eisenstadt et al., 2026); we measure such dependencies without training.

## 6 Limitations

A step with probes costs 1.2 to 1.5 times a mean-field step (1.4 to 1.5 times with the cache), so the wall-clock gain is smaller than the step gain; a leaner probe kernel or cheaper coupling estimation would widen it. The selected settings trade accuracy for speed on MATH and Dream HumanEval (Section 4.3). Mutual support shows self-consistency, not correctness, so wrong tokens can reinforce each other. All answers have 256 tokens. With a key-value cache (Appendix D), our GSM8K accuracy is below both baselines, and at block 20 we are slightly slower than mean-field decoding.

## 7 Conclusion

The pairwise term of parallel decoding can be measured instead of approximated, cheaply, by probes inside the main pass. The measured coupling shows support as well as conflict, and on three models it needs far fewer steps than mean-field decoding with no training, with and without a key-value cache. Shuffled couplings and a uniform bonus both lose this gain.

## References

Giries Abu Ayoub, Mario Barbara, Lluís Pastor-Pérez, Tanja Bien, Aneesh Barthakur, Alaa Maalouf, and Loay Mualem. Supportive token revealing for fast diffusion language model decoding. arXiv preprint arXiv:2606.04236, 2026.

Josh Achiam, Steven Adler, Sandhini Agarwal, et al. GPT-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Jacob Austin, Daniel D. Johnson, Jonathan Ho, Daniel Tarlow, and Rianne van den Berg. Structured denoising diffusion models in discrete state-spaces. In Advances in Neural Information Processing Systems, 2021a.

Jacob Austin et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021b.

Iskander Azangulov, Teodora Pandeva, Niranjani Prasad, Javier Zazo, and Sushrut Karmalkar. Parallel sampling from masked diffusion models via conditional independence testing. In International Conference on Learning Representations, volume 2026, pages 140485–140521, 2026.

Jinze Bai, Shuai Bai, Yunfei Chu, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.

Heli Ben-Hamu, Itai Gat, Daniel Severo, Niklas Nolte, and Brian Karrer. Accelerated sampling from masked diffusion models via entropy bounded unmasking. In Advances in Neural Information Processing Systems, 2025.

Tiwei Bie, Maosong Cao, Kun Chen, et al. LLaDA2.0: Scaling up diffusion language models to 100B. arXiv preprint, 2025.

Tiwei Bie, Maosong Cao, Xiang Cao, et al. LLaDA2.1: Speeding up text diffusion via token editing. arXiv preprint, 2026.

Mark Chen et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

Karl Cobbe et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

Roy Eisenstadt, Harel Lidar, Tal Gonen, Omer Luxembourg, Itamar Zimerman, Lior Wolf, Eli David, and Ilan Naiman. Contextual correction of parallel drafts for speculative decoding. arXiv preprint arXiv:2610.12570, 2026.

Ligong Han, Hao Wang, Han Gao, Kai Xu, and Akash Srivastava. S2D2: Fast decoding for diffusion LLMs via training-free self-speculation. arXiv preprint arXiv:2603.25702, 2026.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. In Advances in Neural Information Processing Systems, Datasets and Benchmarks Track, 2021.

Daniel Israel, Guy Van den Broeck, and Aditya Grover. Accelerating diffusion LLMs via adaptive parallel decoding. Advances in neural information processing systems, 38:52870–52888, 2026.

Bumjun Kim, Dongjae Jeon, Moongyu Jeon, and Albert No. DAPD: Dependency-aware parallel decoding via attention for diffusion LLMs. arXiv preprint arXiv:2603.12996, 2026.

Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, Yuhuai Wu, Behnam Neyshabur, Guy Gur-Ari, and Vedant Misra. Solving quantitative reasoning problems with language models. In Advances in Neural Information Processing Systems, 2022.

Xiang Lisa Li, John Thickstun, Ishaan Gulrajani, Percy Liang, and Tatsunori Hashimoto. Diffusion-LM improves controllable text generation. In Advances in Neural Information Processing Systems, 2022.

Aixin Liu, Bei Feng, Bing Xue, et al. DeepSeek-V3 technical report. arXiv preprint arXiv:2412.19437, 2024.

Lizhuo Luo, Zhuoran Shi, Jiajun Luo, Zhi Wang, Shen Ren, Wenya Wang, and Tianwei Zhang. DAWN: Dependency-aware fast inference for diffusion LLMs. arXiv preprint arXiv:2602.06953, 2026.

Omer Luxembourg, Haim Permuter, and Eliya Nachmani. Plan for speed: Dilated scheduling for masked diffusion language models. In International Conference on Machine Learning, 2026.

Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Large language diffusion models. Advances in Neural Information Processing Systems, 38:50608–50646, 2026.

Tianxiang Pan, Baitao Gong, Mo Guang, Hongwei Yong, Tianpeng Jiang, Yaqian Li, Zheng Cao, and Kaiwen Long. Trajectory-level speculative decoding for diffusion language models. arXiv preprint arXiv:2608.27514, 2026.

Liran Ringel, Ameen Ali, and Yaniv Romano. Dependency-guided parallel decoding in discrete diffusion language models. arXiv preprint arXiv:2604.02560, 2026.

Eliezer Shlomi, Eliron Rahimi, Avi Mendelson, and Chaim Baskin. On the SPOT: Enhancing speculative decoding for diffusion language models via commit-order prediction. arXiv preprint arXiv:2610.22543, 2026.

Guanghan Wang, Yair Schiff, Subham Sahoo, and Volodymyr Kuleshov. Remasking discrete diffusion models with inference-time scaling. Advances in Neural Information Processing Systems, 38: 147282–147339, 2026.

Chengyue Wu, Hao Zhang, Shuchen Xue, Zhijian Liu, Shizhe Diao, Ligeng Zhu, Ping Luo, Song Han, and Enze Xie. Fast-dLLM: Training-free acceleration of diffusion LLM by enabling KV cache and parallel decoding. In International Conference on Learning Representations, volume 2026, pages 57027–57051, 2026.

Jiacheng Ye, Zhihui Xie, Lin Zheng, Jiahui Gao, Zirui Wu, Xin Jiang, Zhenguo Li, and Lingpeng Kong. Dream 7B: Diffusion large language models. arXiv preprint arXiv:2508.15487, 2025.

Shaorong Zhang, Longxuan Yu, Rob Brekelmans, Luhan Tang, Salman Asif, and Greg Ver Steeg. Generation order and parallel decoding in masked diffusion models: An information-theoretic perspective. arXiv preprint arXiv:2602.00286, 2026.

Fengqi Zhu, Rongzhen Wang, Shen Nie, Xiaolu Zhang, Chunwei Wu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. LLaDA 1.5: Variance-reduced preference optimization for large language diffusion models. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 11425–11460, 2026.

Tamim Zoabi, Ameen Ali, Liran Ringel, and Lior Wolf. Mean-field parallel decoding for discrete diffusion language models. arXiv preprint arXiv:2606.15805, 2026.

## A Implementation details

A probe row holds one copy of the still-masked positions of the current block, with the probed candidate j set to its proposal $v _ { j }$ and the other positions left masked. In every layer, the main tokens and all probe rows go through the same normalization, projections, MLP, and output head in one batch. Each probe query attends to the main keys and values of the same layer at every position it does not recompute, and to the keys and values of its own row at the positions it does; it never sees another probe row. The main keys at the recomputed positions are masked out for the probe queries. Rotary embeddings use each position’s index in the sequence, so a probe token sits where it would in the main pass. Main tokens use the model’s usual bidirectional attention, so the main logits match a plain forward pass up to bfloat16 rounding.

The fidelity numbers in Section 4.6 come from 3 test questions of GSM8K (5-shot) and 3 of HumanEval, decoded greedily for three blocks of 20; we measure at the start and middle of each block, giving 18 states and 4,230 pairs per task. The contraction statistics in Section 3.3 come from 1,805 steps on GSM8K and 1,662 on HumanEval. With $K = 1 2$ , a fused step uses 0.26 GB more peak memory than a plain step on GSM8K and 0.14 GB more on HumanEval, on top of 14.9 GB of weights. Timings come from one job per benchmark on one A100 (80 GB), with batch size 1 and bfloat16. Where our method is timed at the start and end of a job, the two times differ by at most 1.3%, except in one LLaDA-1.5 GSM8K job (16%); for that benchmark we report the median over three jobs.

## B Effect of the pairwise term

Table 5 isolates the pairwise term. All rows use the same block size, schedule, and threshold $\tau = 0 . 8 5$ the published threshold of mean-field decoding, so the pairwise term is the only difference. The first row has no pairwise term, so each position is committed on its own confidence. The second row is mean-field decoding with the similarity coupling, as in Table 1. The third row is our method with $\tau = 0 . 8 5$ in place of its selected $\tau = 0 . 9$ , which is why its numbers differ from Table 1.

Table 5: Effect of the pairwise term (accuracy % / NFE). All rows use block 20 and $\tau = 0 . 8 5$ , so only the pairwise term changes. Our row uses the selected probes $( K = 1 2 ,$ , inside the main pass) at this common threshold.
<table><tr><td>Pairwise term</td><td>GSM8K</td><td>HumanEval</td><td>MBPP</td></tr><tr><td>None (confidence only)</td><td>78.09 / 52.1</td><td>35.98 / 49.6</td><td>26.80 / 42.7</td></tr><tr><td>Similarity D (MF)</td><td>76.57 / 65.4</td><td>37.20 / 61.7</td><td>30.20 / 51.6</td></tr><tr><td>Measured ∆ (ours)</td><td>76.42 / 37.4</td><td>34.76 / 37.7</td><td>28.60 / 30.7</td></tr></table>

The similarity term makes decoding more careful: it adds steps relative to the confidence-only rule, and it helps accuracy on MBPP (+3.4 points, $p = 0 . 0 2 7 )$ . The measured coupling moves in the other direction on steps. At the same threshold it needs 24% to 43% fewer steps than either rule.

## C Accuracy against time per answer

Figure 2 sweeps the threshold of our method and of mean-field decoding on LLaDA-8B-Instruct and plots accuracy against wall-clock time per answer. All points of a benchmark are timed in one job on the same A100 GPU (GSM8K on its first 200 questions, HumanEval on all 164, MBPP on its first 200), with batch size 1 and bfloat16; accuracy is from the full test sets. We timed our method at $\tau = 0 . 9$ at the start and at the end of each job, and the two times differ by 0.6% to 3.1%. On GSM8K and MBPP our curve lies to the left of the mean-field curve; on GSM8K it stops at 76.7%, slightly below the best mean-field point (77.5%). On HumanEval the curves cross: our method is faster at the higher accuracies and mean-field decoding at the lower ones. Our answers are also shorter. Counting tokens before the end-of-text token, they are 5% to 56% shorter than those of mean-field decoding on Dream, 17% shorter on LLaDA-8B MBPP, and within about 5% in the other LLaDA settings.

![](images/ee2357733e9aa13a9d7f7f29567e06f11e3b4b20ed0509c4e13d3c4e80576f9e.jpg)  
Figure 2: Accuracy against wall-clock time per answer on LLaDA-8B-Instruct. Curves sweep τ from 0.75 to 0.95 for our method and for mean-field decoding. Gray points are Fast-dLLM and DAWN at their published settings. All points of a panel are timed on the same GPU.

## D Results with a key-value cache

We also run every method with the dual cache of Fast-dLLM (Wu et al., 2026). A full forward pass at the start of each block fills the cache, and every later step of the block runs only the block against it. All three decoders share one implementation of this step, captured as a CUDA graph, with one fused attention kernel that applies the rotary embedding on load; our probe rows run in the same pass and kernel. All methods use block 32, where the block-start passes cost the same for every method. Table 6 reports our method with K = 4 probes and $\tau = 0 . 9$ on LLaDA-8B-Instruct. Accuracy is from the full test sets, and time is measured for all methods in one job on the same GPU after a warm-up of 20 questions (GSM8K on its first 200 questions, MBPP on all 500). Our method needs 29% to 31% fewer steps than mean-field decoding and 49% to 55% fewer than Fast-dLLM, and has the shortest time per answer on both benchmarks. Its accuracy is higher than both baselines on MBPP, by 4.0 points over mean-field decoding $( p = 0 . 0 3 )$ , and lower on GSM8K. This setting is the fastest of the cached grid we ran $( K \in \{ 4 , 8 \bar { \} } , \tau \in \{ 0 . 8 5 , 0 . 9 \}$ , blocks 20 and 32). The setting chosen by our rule on GSM8K training questions uses block 20 and K = 8 (GSM8K 75.1%, MBPP 27.6%); at block 20 every method pays more block starts, and ours is 3% to 5% slower than mean-field decoding at the same block.

Table 6: Methods with the dual key-value cache on LLaDA-8B-Instruct, block 32 (full test sets). Time is the wall-clock time per answer in seconds. Step time excludes the full passes at block starts, which cost about the same for every method (1.04 s per answer on GSM8K, 0.86 to 0.88 s on MBPP).
<table><tr><td>Bench.</td><td>Method</td><td>Acc. ↑</td><td>NFE↓</td><td>Time (s) ↓</td><td>Step time (s) ↓</td></tr><tr><td rowspan="3">GSM8K</td><td>Fast-dLLM</td><td>79.38</td><td>85.6</td><td>2.31</td><td>1.27</td></tr><tr><td>MF</td><td>77.03</td><td>61.6</td><td>2.06</td><td>1.02</td></tr><tr><td>Ours</td><td>75.44</td><td>43.6</td><td>1.96</td><td>0.92</td></tr><tr><td rowspan="3">MBPP</td><td>Fast-dLLM</td><td>26.60</td><td>68.5</td><td>1.84</td><td>0.96</td></tr><tr><td>MF</td><td>24.00</td><td>44.9</td><td>1.56</td><td>0.70</td></tr><tr><td>Ours</td><td>28.00</td><td>30.9</td><td>1.51</td><td>0.65</td></tr></table>

## E Accuracy against steps on LLaDA-8B

![](images/0b2c31f860c1780e8c0f8710d7761c2f7b60cf7d1e4681fda95301e08f4fcdde.jpg)  
Figure 3: Accuracy against the number of steps per answer on LLaDA-8B-Instruct. Curves sweep the threshold τ: 0.75 to 0.95 for our method and for mean-field decoding, and 0.7 to 0.85 for the confidence-only rule (0.8 and 0.85 on code). Gray points are Fast-dLLM and DAWN at their published settings; the dashed line is one-token-per-step decoding.

## F Accuracy against steps on Dream

Figure 4 sweeps the threshold on Dream-v0-Instruct-7B. Our curve reaches its accuracy with about half the steps of mean-field decoding, and on MBPP it rises above the mean-field curve.

![](images/92121e9758515adc93efb7fa14381df7808344401434df3e0bc83e249adf931e.jpg)  
Figure 4: Accuracy against steps per answer on Dream-v0-Instruct-7B. Curves sweep τ over 0.9, 0.95, and 0.97 at block 20, and up to 0.99 for our method on MATH. Our curve uses K = 12. The star is our calibrated setting from Table 3. Gray points are Fast-dLLM and DAWN; the dashed line is one-token-per-step decoding.

## G Where the MATH gap comes from

We paired every MATH problem between our method and mean-field decoding, using the saved answers of the main runs. The gap is spread over all difficulty levels and subjects instead of concentrating on the hardest problems, where long and sensitive derivations would stress the additive model most. Repetition loops occur at similar rates for both methods and explain at most 0.2 points of the gap. On Dream, a stricter threshold narrows the gap: at τ = 0.99 our method comes within 0.8 points of mean-field decoding (37.1% against 37.9%, with K = 12 and with $K = 2 0 )$ with nearly 40% fewer steps, and the difference is not significant (Figure 4). The gap therefore depends on the speed setting chosen on held-out data, and a stricter threshold reduces it at the cost of more steps.

## H Number of probes and block size

Table 7 varies the number of probes K and the block size on GSM8K, with all other settings fixed at the selected ones.

Number of probes. Accuracy stays the same for every K we tried, while each extra probe saves steps. More measured couplings let the decoder commit more tokens together, at no cost in accuracy.

Block size. At blocks 16 and 20, our method matches mean-field decoding with far fewer steps, and at block 16 it is slightly more accurate. With a fixed K, larger blocks leave most candidates unprobed: their columns of $\Delta$ are zero, so their influence on other positions is not measured, although they still receive measured influences from the probed candidates. Accuracy drops. Scaling K with the block size keeps the share of probed candidates constant. This restores accuracy at block 32, where our method again matches mean-field decoding with far fewer steps. Very large blocks such as 64 need more than additional probes, and we leave them to future work.

Table 7: Ablations on GSM8K (accuracy % / NFE). Left: number of probes K $( \tau = 0 . 9$ , block 20). Right: block size, for our method with $K = 1 2$ , with K scaled with the block $( K = 1 2$ at blocks 16 and 20, K = 20 at 32, K = 40 at 64), and for mean-field decoding $( \tau = 0 . 8 5 )$
<table><tr><td>K</td><td>Ours</td><td>Block</td><td>Ours (K = 12)</td><td>Ours (scaled K)</td><td>MF</td></tr><tr><td>4</td><td>76.42 / 45.0</td><td>16</td><td>77.03 / 43.2</td><td>77.03 / 43.2</td><td>76.95 / 69.0</td></tr><tr><td>8</td><td>76.27 / 41.3</td><td>20</td><td>76.27 / 40.1</td><td>76.27 / 40.1</td><td>76.57 / 65.4</td></tr><tr><td>12</td><td>76.27 / 40.1</td><td>32</td><td>72.55 / 34.2</td><td>76.19 / 33.6</td><td>76.42 / 56.4</td></tr><tr><td>20</td><td>76.27 / 38.8</td><td>64</td><td>67.70 / 29.1</td><td>67.70 / 27.4</td><td>75.44 / 48.3</td></tr></table>

## I Definition of the controls

Every control in Table 4 changes only the measured off-diagonal entries of $\Delta ,$ , so it has the same coverage as our method. Shuffled permutes these values at random with a fixed seed. Support only sets $\bar { \Delta _ { i j } } \gets \operatorname* { m i n } ( \Delta _ { i j } , 0 )$ , and conflict only sets $\Delta _ { i j } \gets \operatorname* { m a x } ( \Delta _ { i j } , 0 )$ . Partially symmetrized replaces $\Delta _ { i j }$ by $( \Delta _ { i j } + \Delta _ { j i } ) \big / 2$ where both directions are measured and keeps the other entries, so the matrix is not fully symmetric. Constant bonus sets every measured entry to −b; conflicts + constant support keeps $\Delta _ { i j } > 0$ and sets every measured $\Delta _ { i j } < \dot { 0 } { \mathrm { t o } } - b$

## J Testing the additive model

The decoder sums single-token effects over the committed set $S$ (Equation 6). We test this on the sets it actually commits. At every step with $| S | \ge 2$ , we measure each $\Delta _ { i j }$ exactly with one full forward pass $\mathsf { p e r } j \in S$ , and the true margin of each remaining position with all of S revealed in one more pass. A flip means that the additive prediction and the true margin, after the sigmoid, fall on different sides of $\tau .$ . Flips are rare for small sets and grow with |S| (Table 8). The sum overstates both signs. Where support dominates, the predicted margin is about 5.5 and the true one about 3.5, so support from several tokens is partly redundant. Where conflict dominates, the prediction is about −16 and the true margin about −3.5, so conflicts also saturate. For the committed tokens themselves, with the rest of their set revealed (Appendix K), the additive and the exact margin fall on the same side of τ for 96% of tokens on GSM8K and 97% on MBPP.

## K Commit decisions

We decode 50 test questions of GSM8K and of MBPP with our method and, at every step, compute the commit set on the same state with other couplings (Table 9). Block-local probes for all candidates change 4% of the decisions of exact probes, and measuring only the probed columns accounts for most of the rest. Most changed decisions hold back a token that exact probes would commit, which costs steps more than accuracy.

Table 8: Additive prediction against the jointly measured margin, LLaDA-8B-Instruct with our method $( K = 1 2 , \tau = 0 . 9 ;$ K scaled with the block at 32 and 64). Share of remaining (uncommitted) positions whose decision at τ flips, by the size of the committed set. For the committed tokens, see the text.
<table><tr><td>Run</td><td>Positions</td><td> $| S | = 2 \mathrm { - 3 }$ </td><td> $| S | = 4 { - } 7$ </td><td> $| S | \ge 8$ </td><td>All</td></tr><tr><td>GSM8K, block 20</td><td>7,880</td><td>3.7%</td><td>7.8%</td><td>15.6%</td><td>10.0%</td></tr><tr><td>MBPP, block 20</td><td>7,981</td><td>4.5%</td><td>7.3%</td><td>14.2%</td><td>9.4%</td></tr><tr><td>GSM8K, block 32</td><td>8,199</td><td>2.4%</td><td>5.7%</td><td>9.2%</td><td>7.0%</td></tr><tr><td>GSM8K, block 64</td><td>13,147</td><td>1.4%</td><td>3.7%</td><td>10.8%</td><td>7.2%</td></tr></table>

We then check each committed token against the rest of its commit set: we reveal the other tokens of the set, leave the token masked, and test whether its committed value is still the most likely one (Table 10). End-of-text padding is excluded. At the same number of steps, a confidence threshold commits three to four times as many overturned tokens as our method, and the gap is largest for small sets. Slower decoders with 50% to 65% more steps are overturned slightly less often. On Dream $( K = 2 0 , \tau = 0 . 9 5 )$ , the threshold at a similar step budget is overturned 3.6 times as often on MATH and 1.4 times as often on HumanEval.

Table 9: Commit decisions on the same decoding states, against exact full-sequence probes for every candidate (50 test questions each, our method with $K = 1 2$ and $\tau = 0 . 9 ;$ steps with only end-of-text padding left are excluded).
<table><tr><td></td><td colspan="2">GSM8K</td><td colspan="2">MBPP</td></tr><tr><td>Coupling</td><td>Same set</td><td>Decisions changed</td><td>Same set</td><td>Decisions changed</td></tr><tr><td>Block-local, all candidates</td><td>70%</td><td>4.2%</td><td>71%</td><td>4.4%</td></tr><tr><td>Exact, probed columns only</td><td>55%</td><td>10.6%</td><td>55%</td><td>11.2%</td></tr><tr><td>Ours (inside the main pass)</td><td>50%</td><td>11.8%</td><td>47%</td><td>12.5%</td></tr><tr><td>No coupling</td><td>33%</td><td>27.3%</td><td>36%</td><td>25.2%</td></tr></table>

Table 10: Committed tokens overturned by the rest of their commit set, by set size (50 test questions each; LLaDA-8B-Instruct unless marked Dream). Conf. is the confidence threshold.
<table><tr><td>Task</td><td>Decoder</td><td>T</td><td>NFE</td><td> $| S | = 2 \mathrm { - } 3$ </td><td> $| S | = 4 { - } 7$ </td><td> $| S | \ge 8$ </td><td>All</td></tr><tr><td rowspan="4">GSM8K</td><td>Ours</td><td>0.9</td><td>41.1</td><td>0.4%</td><td>0.8%</td><td>0.8%</td><td>0.8%</td></tr><tr><td>Conf.</td><td>0.75</td><td>41.9</td><td>6.6%</td><td>4.9%</td><td>1.7%</td><td>2.9%</td></tr><tr><td>Conf.</td><td>0.9</td><td>66.3</td><td>0.2%</td><td>0.3%</td><td>0.2%</td><td>0.2%</td></tr><tr><td>MF</td><td>0.85</td><td>68.0</td><td>0.5%</td><td>0.3%</td><td>0.1%</td><td>0.2%</td></tr><tr><td rowspan="4">MBPP</td><td>Ours</td><td>0.9</td><td>35.0</td><td>1.2%</td><td>0.8%</td><td>0.7%</td><td>0.8%</td></tr><tr><td>Conf.</td><td>0.8</td><td>41.5</td><td>5.0%</td><td>3.3%</td><td>1.0%</td><td>2.3%</td></tr><tr><td>Conf.</td><td>0.9</td><td>53.5</td><td>0.8%</td><td>0.4%</td><td>0.1%</td><td>0.3%</td></tr><tr><td>MF</td><td>0.85</td><td>55.2</td><td>1.3%</td><td>0.1%</td><td>0.0%</td><td>0.3%</td></tr><tr><td rowspan="3">Dream MATH</td><td>Ours</td><td>0.95</td><td>60.9</td><td>0.5%</td><td>0.5%</td><td>0.4%</td><td>0.4%</td></tr><tr><td>Conf.</td><td>0.8</td><td>61.6</td><td>3.4%</td><td>1.5%</td><td>1.0%</td><td>1.5%</td></tr><tr><td>MF</td><td>0.95</td><td>116.8</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td></tr><tr><td rowspan="3">Dream HumanEval</td><td>Ours</td><td>0.95</td><td>28.9</td><td>0.6%</td><td>1.1%</td><td>0.5%</td><td>0.7%</td></tr><tr><td>Conf.</td><td>0.85</td><td>30.3</td><td>2.7%</td><td>1.0%</td><td>0.4%</td><td>1.0%</td></tr><tr><td>MF</td><td>0.95</td><td>56.4</td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td></tr></table>

## L Number of mean-field iterations

Table 11 varies the number of mean-field iterations R on Dream-v0-Instruct-7B, with all other settings as in Table 3. For our method, the number of steps stays within 40 to 43 on GSM8K and 34 to 36 on MBPP for every R. Accuracy varies by up to 1.2 points on GSM8K and 2.4 points on MBPP, and no

difference from R = 2 is significant. Mean-field decoding needs 1.4 to 1.6 times more steps at odd R than at even R, which matches the oscillation reported by Zoabi et al. (2026). All main results use $R = 2 ,$ , as in mean-field decoding.

Table 11: Number of mean-field iterations R on Dream-v0-Instruct-7B (accuracy % / NFE). The main results use R = 2 (bold).
<table><tr><td rowspan="2"></td><td colspan="2">Ours  $( K = 2 0 , \tau = 0 . 9 5 )$ </td><td colspan="2">Mean-field (τ = 0.95)</td></tr><tr><td>GSM8K Acc. % / NFE</td><td>MBPP Acc. % / NFE</td><td>GSM8K Acc. % / NFE</td><td>MBPP Acc. % / NFE</td></tr><tr><td>R 1</td><td>75.36 / 42.9</td><td>56.00 / 36.1</td><td>75.66 / 113.3</td><td></td></tr><tr><td>2</td><td>75.13 / 41.4</td><td>53.60 / 34.4</td><td>76.19 / 69.2</td><td>57.80 / 118.3 54.20 / 77.2</td></tr><tr><td>3</td><td>74.15 / 40.2</td><td>56.00 / 34.3</td><td>75.89 / 110.3</td><td>58.40 / 113.9</td></tr><tr><td>4</td><td>74.22 / 39.8</td><td>54.80 / 33.8</td><td>75.82 / 70.2</td><td>54.00 / 79.2</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table>

## M Learned coupling from the next step

The decoder’s own next step contains information about the coupling. After a set S is committed, the next forward pass shows whether each remaining position changes its choice. This gives an exact label for the event that committing that position together with S would have changed the answer, at no extra cost. We fit a logistic model of this event on simple features of position pairs (distance and side, similarity of the two predictions, whether the tokens match, and the uncertainty at i), by ridge-regularized Newton steps, online and without training data, and commit positions whose predicted error probability is below a risk level $\rho .$

This variant needs no extra forward passes and is useful on its own. With ρ tuned on GSM8K training questions, it uses fewer steps than mean-field decoding on GSM8K, HumanEval, and MBPP, with accuracy within 1.4 points. Its predicted flip rates match the observed rates within a few points for every committed-set size. This calibration is about flips, and Appendix J tests the additive margins directly. Its labels come only from positions that were not committed, so they rarely cover confident tokens. Probes reach those tokens directly, which explains the larger savings of measured coupling.

Table 12 reports this variant on LLaDA-8B-Instruct. We chose $\rho = 0 . 0 5$ on the first 100 GSM8K training questions, the same held-out set as for our main method. It uses fewer steps than mean-field decoding on all three benchmarks. Its accuracy is 0.2 points higher on GSM8K, equal on HumanEval, and 1.4 points lower on MBPP.

Table 12: Learned coupling from the next step on LLaDA-8B-Instruct (block 20, $\rho = 0 . 0 5 )$ . Meanfield decoding is as in Table 1.
<table><tr><td rowspan="2">Method</td><td colspan="2">GSM8K</td><td colspan="2">HumanEval</td><td colspan="2">MBPP</td></tr><tr><td>Acc. % ↑</td><td>NFE↓</td><td>Acc. % ↑</td><td>NFE↓</td><td>Acc. % ↑</td><td>NFE↓</td></tr><tr><td>Mean-field (MF)</td><td>76.57</td><td>65.4</td><td>37.20</td><td>61.7</td><td>30.20</td><td>51.6</td></tr><tr><td>Learned coupling</td><td>76.72</td><td>53.1</td><td>37.20</td><td>54.2</td><td>28.80</td><td>48.7</td></tr></table>

## N Held-out selection results

Table 13: Every candidate setting of our method scored on each held-out set. “Masked only” probes recompute only the still-masked positions, and “separate call” probes run after the main pass instead of inside it. The selected setting, chosen on GSM8K, is in bold. MBPP validation is a check on code. Mean-field decoding is shown for reference only; its setting is the published one.
<table><tr><td>Held-out set</td><td>Probes</td><td>T</td><td>Accuracy ↑</td><td>NFE↓</td></tr><tr><td rowspan="10">GSM8K train (100)</td><td>12, inside main pass, masked only</td><td>0.90</td><td>91.0</td><td>39.3</td></tr><tr><td>12, inside main pass, masked only</td><td>0.85</td><td>84.0</td><td>37.2</td></tr><tr><td>8, inside main pass, masked only</td><td>0.90</td><td>88.0</td><td>40.1</td></tr><tr><td>8, inside main pass, masked only</td><td>0.85</td><td>86.0</td><td>38.4</td></tr><tr><td>8, separate call</td><td>0.85</td><td>90.0</td><td>34.3</td></tr><tr><td>8, separate call</td><td>0.90</td><td>85.0</td><td>36.2</td></tr><tr><td>12, separate call</td><td>0.85</td><td>84.0</td><td>32.8</td></tr><tr><td>12, separate call</td><td>0.90</td><td>86.0</td><td>34.7</td></tr><tr><td>20, inside main pass</td><td>0.85</td><td>84.0</td><td>35.6</td></tr><tr><td>20, inside main pass 12, inside main pass, masked only</td><td>0.90 0.93</td><td>82.0</td><td>38.0</td></tr><tr><td>12, inside main pass, masked only</td><td></td><td></td><td>86.0</td><td>42.0</td></tr><tr><td>12, inside main pass, masked only</td><td>0.95 0.97</td><td>83.0 81.0</td><td></td><td>42.8 45.5</td></tr><tr><td>12, inside main pass, masked only</td><td>0.90</td><td>35.56</td><td></td><td></td></tr><tr><td>8, inside main pass, masked only</td><td>0.90</td><td>30.00</td><td></td><td>32.7</td></tr><tr><td>20, inside main pass, masked only</td><td>0.90</td><td>30.00</td><td></td><td>33.5</td></tr><tr><td></td><td></td><td>0.90</td><td>28.89</td><td>33.5</td></tr><tr><td></td><td>12, inside main pass</td><td>0.85</td><td></td><td>32.9</td></tr><tr><td>MBPP validation (90)</td><td>12, inside main pass</td><td></td><td>27.78</td><td>32.0</td></tr><tr><td></td><td>20, inside main pass</td><td>0.85</td><td>27.78</td><td>31.3</td></tr><tr><td>12, separate call</td><td>20, inside main pass</td><td>0.90</td><td>27.78</td><td>33.5</td></tr><tr><td></td><td></td><td>0.90</td><td>26.67</td><td>29.6</td></tr><tr><td></td><td>8, separate call</td><td>0.90</td><td>25.56</td><td>30.8</td></tr><tr><td></td><td>12, inside main pass, masked only</td><td>0.93</td><td>34.44</td><td>33.8</td></tr><tr><td></td><td>12, inside main pass, masked only</td><td>0.95</td><td>37.78</td><td>35.2</td></tr><tr><td></td><td>12, inside main pass, masked only</td><td>0.97</td><td>37.78</td><td>37.1</td></tr><tr><td>Mean-field (reference)</td><td></td><td>0.85</td><td>30.00</td><td>52.2</td></tr></table>

Table 14: Held-out candidates on Dream-v0-Instruct-7B, with the same held-out sets and rule as Table 13. The selected setting, chosen on GSM8K, is in bold; MBPP validation selects the same one. Mean-field decoding is shown for reference only; its setting is the published Dream one.
<table><tr><td>Held-out set</td><td>Probes</td><td>T</td><td>Accuracy ↑</td><td>NFE↓</td></tr><tr><td rowspan="7">GSM8K train (100)</td><td>20, inside main pass, masked only</td><td>0.95</td><td>74.0</td><td>42.2</td></tr><tr><td>20, inside main pass, masked only</td><td>0.90</td><td>73.0</td><td>37.9</td></tr><tr><td>8, inside main pass, masked only</td><td>0.95</td><td>71.0</td><td>42.8</td></tr><tr><td>12, inside main pass, masked only</td><td>0.90</td><td>71.0</td><td>37.6</td></tr><tr><td>8, inside main pass, masked only</td><td>0.90</td><td>70.0</td><td>38.2</td></tr><tr><td>12, inside main pass, masked only</td><td>0.97</td><td>69.0</td><td>44.4</td></tr><tr><td>8, inside main pass, masked only</td><td>0.97</td><td>68.0</td><td>45.2</td></tr><tr><td>20, inside main pass, masked only 12, inside main pass, masked only</td><td>0.97 0.95</td><td>68.0</td><td></td><td>44.1</td></tr><tr><td>Mean-field (reference)</td><td>0.95</td><td></td><td>66.0</td><td>42.1</td></tr><tr><td>20, inside main pass, masked only</td><td></td><td></td><td>78.0</td><td>68.6</td></tr><tr><td>12, inside main pass, masked only</td><td></td><td>0.95 0.97</td><td>54.44</td><td>34.1</td></tr><tr><td></td><td></td><td></td><td>54.44</td><td>34.9</td></tr><tr><td rowspan="7">MBPP validation (90)</td><td>20, inside main pass, masked only</td><td>0.90</td><td>53.33</td><td>30.4</td></tr><tr><td>20, inside main pass, masked only</td><td>0.97</td><td>53.33</td><td>35.5</td></tr><tr><td>12, inside main pass, masked only</td><td>0.95</td><td>53.33</td><td>34.1</td></tr><tr><td>8, inside main pass, masked only</td><td>0.97</td><td>52.22</td><td>35.2</td></tr><tr><td>8, inside main pass, masked only</td><td>0.95</td><td>50.00</td><td>32.9</td></tr><tr><td>8, inside main pass, masked only</td><td>0.90</td><td>48.89</td><td>30.3</td></tr><tr><td>12, inside main pass, masked only</td><td>0.90</td><td>47.78</td><td>30.4</td></tr><tr><td></td><td>Mean-field (reference)</td><td>0.95</td><td>56.67</td><td>76.0</td></tr></table>