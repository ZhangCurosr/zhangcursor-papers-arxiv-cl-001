# HILO-RD: CROSS-BUDGET SELF-DISTILLATION FOR MASKED DIFFUSION LANGUAGE MODELS

Xiang Li<sup>1</sup> Chengting Yu<sup>2</sup> Jungang Li<sup>1</sup> Yilun Zheng<sup>3</sup> Junzhuo Li<sup>1</sup> Xingyao Zhang<sup>4</sup> Wenbo Su<sup>4</sup> Xuming Hu<sup>1,5</sup>

<sup>1</sup>The Hong Kong University of Science and Technology (Guangzhou) <sup>2</sup>Zhejiang University <sup>3</sup>Nanyang Technological University <sup>4</sup>Alibaba Group <sup>5</sup>The Hong Kong University of Science and Technology

## ABSTRACT

We propose HiLo-RD (High-to-Low Budget Representation Distillation) to improve masked diffusion language models (MDLMs) under a limited generation budget. Unlike autoregressive models, MDLMs choose both what to generate and where to commit it. This creates two coupled challenges: predicting useful content before reasoning is complete, and deciding when to fix answer tokens that shape later predictions. We identify budget-inducedpremature commitment: even when a correct solution fits, the model can fix answer fragments near the window boundary before resolving intermediate steps. These fragments become context for later predictions and redirect subsequent reasoning. Existing posttraining methods do not directly transfer the stronger trajectories that the same model follows with a larger window. HiLo-RD addresses this gap by (1) pairing large-window teacher states with cropped small-window views and (2) matching their hidden representations at committed positions. This supervision guides both token prediction and commitment order without an external teacher. Across mathematics benchmarks, HiLo-RD improves low-budget accuracy and shortens responses; these gains extend to code generation without code-domain training. On GSM8K, accuracy rises by 9.4 percentage points as mean readable length falls by about 25%. Attention diagnostics and fixed-budget order interventions support the proposed mechanism. Together, these results show that large-window reasoning can improve small-window decoding.

## 1 INTRODUCTION

Masked diffusion language models (MDLMs) offer a route beyond left-to-right generation by reconstructing masked text under bidirectional attention (Sahoo et al., 2024; Nie et al., 2025; 2026). In pure diffusion decoding, the model predicts candidates throughout a response window and chooses which positions to commit at each update. This allows reasoning to unfold across denoising steps and potentially across independent subproblems in parallel (Svete & Sabharwal, 2026). For practical decoding, however, the response window and the number of updates are finite. The model must therefore decide both what to write and when to fix each part of the answer within a limited budget.

Consider a mathematical question for which the same model gives different answers under two response-window sizes. With the smaller window, it can fix answer fragments near the end before filling the reasoning in between, even though a correct solution would fit. With a larger window, it may defer the answer and reach a shorter, correct solution. We call this failure budget-induced premature commitment. Figure 1 shows the broader aggregate pattern: fewer responses end near the larger window’s limit, while accuracy rises and mean readable length falls. This raises our central question: Ifa larger window yields shorter, more accurate answers, can we distill that advantage into a smaller one?

(a)  
![](images/fee9f60d3b922ca66e65f1c101a8817c4c5b9a575a7b3e3025bb16d2db123fe2.jpg)

(b)  
![](images/7d7f289414d2f3930c1a153e931253a9fbf8fa6cefff9371fde51c460a7eb9d7.jpg)

![](images/8d57ce6d44b7cbacdcc30b0974bf93e5d60839b03127f1a604972f2eb19813e7.jpg)  
Figure 1: More room, shorter and more accurate answers. At $L = T = 5 1 2$ (blue) and $L = T = 1 0 2 4$ (orange), (a) shows response-length shares in 32- token bins on common axes and (b) compares accuracy and mean readable length. Values are approximate readings from the archived aggregate plot.

This example exposes two coupled challenges in low-budget diffusion reasoning. First, the model must choose what to predict while intermediate steps are still unresolved. Second, it must decide when to commit an answer fragment, because committed tokens cannot be revised and become context for every later prediction. A premature answer can thus shape the reasoning subsequently filled around it. Existing inference-time methods adjust the available space or restrict the order of generation (Li et al., 2025; Arriola et al., 2025); post-training methods learn from demonstrations, outcome rewards, or self-distillation targets (Zhao et al., 2025; 2026; Dat et al., 2026). These approaches improve decoding in different ways, but do not directly transfer the same model’s more effective large-window computation into its small-window decisions.

We investigate how an early commitment changes the subsequent trajectory. Before the model fixes the response tail, attention concentrates near the small-window boundary. Once tail tokens are committed, unresolved positions attend more to them and less to the established reasoning prefix. This feedback makes later predictions depend on an answer selected before its supporting steps are complete. In a set of 155 selected cases where native low-budget decoding is wrong and native high-budget decoding is correct, temporarily delaying tail commitments at the original low budget recovers correct answers in 43.2% of cases, compared with 30.3% for a matched order control. These diagnostics connect the budget, commitment order, and downstream reasoning, and show that a better path can remain available without increasing the deployment budget.

We summarize our main contributions as follows.

❶ We identify budget-induced premature commitment: a small window can induce an early answer even when a correct solution fits. Attention diagnostics and fixed-budget interventions trace how such commitments influence subsequent reasoning.

❷ We introduce HiLo-RD (High-to-Low Budget Representation Distillation). It aligns largewindow teacher states with their cropped small-window views and matches hidden representations at committed positions, using the model’s own trajectories to guide low-budget decoding.

❸ At the same small inference budget, HiLo-RD raises GSM8K accuracy from 68.84% to 78.24% while reducing mean readable length from 362.8 to 271.3 tokens. The gains extend to code generation without code-domain training, and successive distillation improves the low-budget quality– length trade-off on a fixed training subset.

Overall, HiLo-RD transfers the same model’s larger-window computation into more accurate, more concise small-window responses. The results show that the way a diffusion model commits its answer is a useful target for improving reasoning under a limited generation budget.

## 2 PRELIMINARIES AND RELATED WORK

## 2.1 MASKED DIFFUSION DECODING

We consider pure diffusion decoding with LLaDA-8B-Instruct (Nie et al., 2025), a masked diffusion language model (MDLM). The full response is decoded as one bidirectional block. Once a token is committed, it cannot be revised.

Write the prompt and response state before each update as

$$
\mathbf { q } = [ q _ { 1 } , \dots , q _ { P } ] , \qquad \mathbf { y } _ { t } = [ y _ { t , 1 } , \dots , y _ { t , L } ] , \qquad \mathbf { y } _ { 0 } = [ { \mathrm { M A S K } } ] ^ { L } . \qquad 
$$

A response position � has input index $P + j$ . The masked and observed (committed) positions are

$$
M _ { t } = \{ j \in [ L ] : y _ { t , j } = [ \mathtt { M A S K } ] \} , \qquad C _ { t } = [ L ] \backslash \mathcal { M } _ { t } ,\tag{1}
$$

where $[ L ] = \left\{ 1 , \ldots , L \right\}$ . For vocabulary $\boldsymbol { \mathbf { \mathit { \Sigma } } } _ { \mathcal { \mathbf { V } } . }$ , the model with parameters � predicts at each masked position

$$
\hat { y } _ { t , j } = \arg \operatorname* { m a x } _ { \nu \in \mathcal { V } } p _ { \theta } ( \nu \mid \mathbf { q } , \mathbf { y } _ { t } , j ) .
$$

A confidence rule ranks masked positions, suppressing EOS/EOT candidates in the reference decoder. Selected positions receive their predicted tokens; all other positions remain unchanged. Selection need not follow response order, and the ranking score need not equal token probability.

A budget specifies the response-window length and planned number of updates. A run can terminate after � updates:

$$
B \triangleq ( L , T ) , \qquad 1 \leq R \leq T \leq L .
$$

The full decoder fills the window in � updates. When $L = T ,$ , it commits one position per update; when $T \ < \ L ,$ at least one update commits multiple positions. Enlarging the window changes both the visible context and the positions competing for commitment.

## 2.2 DECODING BUDGETS AND REALIZED COMPUTATION

At each update, the decoder processes the prompt and full response window afresh. For $N = P + L$ input positions, dense forward cost (including the vocabulary projection) scales as

$$
F \big ( N \big ) = \Theta \Big ( n _ { \ell } \left[ N d ^ { 2 } + N d d _ { \mathrm { f f } } + N ^ { 2 } d \right] + N d | \mathcal { V } | \Big ) .\tag{2}
$$

Here $n _ { \ell }$ is the number of layers, � the hidden width, and $d _ { \mathrm { f f } }$ the feed-forward width. Bidirectional states can change when new response tokens are committed, so this decoder does not reuse a KV cache across updates (Vaswani et al., 2017; Nie et al., 2025). Using the forward-pass cost as a proxy gives

$$
C ( \mathbf { q } , B ; R ) \triangleq R F ( P + L ) , \qquad C _ { \mathrm { f u l l } } ( \mathbf { q } , B ) \triangleq T F ( P + L ) .\tag{3}
$$

Each avoided update saves one full-window forward. Section 4.2 specifies the stopping rules.

## 2.3 RELATED WORK

Diffusion language models. Masked diffusion models predict missing tokens from visible context, allowing positions to be committed out of order (Sahoo et al., 2024; Nie et al., 2025). LLaDA and iLLaDA train from scratch, while DiffuLLaMA and Dream adapt autoregressive checkpoints (Nie et al., 2026; Gong et al., 2025; Ye et al., 2025). Block diffusion orders generation between blocks (Arriola et al., 2025); WeDLM uses causal attention for cache-friendly recovery (Liu et al., 2025). Our single-block setting keeps all masked response positions eligible at each update.

Post-training methods. Masked SFT learns from partially masked references; Diffu-GRPO uses outcome rewards (Nie et al., 2025; Zhao et al., 2025). OPSD and dOPSD self-distill sampled trajectories (Zhao et al., 2026; Dat et al., 2026), while T3D distills trajectories to reduce denoising steps (Zhang et al., 2026). HiLo-RD pairs a teacher state with its cropped small-window view and matches their hidden states to improve low-budget reasoning.

## 3 BUDGET-INDUCED PREMATURE COMMITMENT AND ITS CONSEQUENCES

A response window can fit a correct solution yet still change when a masked diffusion language model fixes its answer. We study 155 paired cases selected for an incorrect answer at $L = T = 5 \bar { 1 } 2$ and a correct answer from the same model at $L = T = 1 0 2 4$ . In the smaller window, the model commits an answer fragment near the response tail before the intervening reasoning is complete. We connect this failure to boundary-focused attention, subsequent attention to the committed tail, and recovery when tail commitment is delayed within the original budget.

## 3.1 BOUNDARY ATTENTION PRECEDES PREMATURE COMMITMENT

Figure 2 examines attention immediately before the first ordinary tail commitment. Under lowbudget decoding, a band forms near the response-window boundary across the middle and later layers while intermediate positions remain unresolved. Appending 512 visible [MASK] positions to the same partial state and query weakens this band.

![](images/c87f776db07fa8c244b0bd4b71141f735f0b8cf45e312e886b783be46a19363d.jpg)

![](images/2a0cb14e7cea0333cc4b10e65de0a8c72606d45357a2dba110514eb06220cbe5.jpg)

![](images/6290601bd9e90536831e2542fa1efe155c26f14308e000f45ecb0edba1ec2989.jpg)  
Figure 2: Boundary-focused attention precedes tail commitment. Visible-window expansion relaxes this focus at the same state and query (left vs. middle), while native high-budget decoding attends more to earlier response positions at the matched update (right). Panels share an attention scale; red dashed lines mark the original low-budget boundary.

Native high-budget decoding, followed from initialization to the matched update, instead attends more to earlier decoded reasoning. The within-state comparison isolates the effect of visible response space; the native trajectory shows the computation reached when that space is available throughout decoding. The smaller window changes attention before the first tail token is fixed.

## 3.2 COMMITTED TAIL CONTENT ATTRACTS SUBSEQUENT ATTENTION

Once the tail is committed, it becomes part of the bidirectional context for unresolved positions.

Figure 3 follows the same unfinished front positions across 145 valid traces. Immediately after the first tail commitment, their mean attention to the tail rises by 1.91 percentage points, while attention to the established prefix falls by 0.41 points. Along the subsequent decoding trajectory, these changes reach +4.19 and −1.87 points at +32 updates. The early answer fragment thus enters the context used to complete the still-masked reasoning steps.

![](images/da1d8140a1564156c0b92cc9fb0c2a386710000ad42c3512f7f028f4987fe30f.jpg)  
Figure 3: Attention shifts toward the committed tail. Changes from the precommitment update, averaged across layers and 145 valid traces. The dashed line marks commitment; 0 is immediately after it.

## 3.3 TAIL DELAY RESTORES CORRECT ANSWERS

A fixed-budget intervention tests whether postponing that decision opens a better continuation. Tail Delay excludes the original responsetail positions during the first 10% of remaining updates. A Matched-order Control perturbs selection at the same update indices but excludes its own current highest-scoring position, without a tail-specific rule. Both branches start from the same pre-event state and retain the $L = T = 5 1 2$ window, predictor, and decoding schedule.

Tail Delay reaches the correct answer in 67 of the 155 selected cases (43.2%), compared with 47 (30.3%) for the order control, a paired gain of 12.9 percentage points (95% bootstrap inter-

![](images/2637a639e052131184b2a325963f13dfb383e972fcfff8a0eddd3679b07c7ee5.jpg)  
Figure 4: Tail delay improves low-budget recovery. Correct continuations in 155 cases selected for an incorrect native 512-token answer and a correct native 1024-token answer. Both interventions use $L = T = 5 1 2$

val: 7.7–18.7). The advantage over a matched order perturbation supports postponing the tail as the useful change. Successful paths are accessible within the smaller window. This motivates HiLo-RD to transfer the high-budget model’s commitment-aware representations into low-budget decoding.

## 4 HiLo-RD: CROSS-BUDGET REPRESENTATION SELF-DISTILLATION

## 4.1 CROSS-BUDGET REPRESENTATION DISTILLATION

HiLo-RD transfers the commitment-aware representations of the model’s own high-budget trajectories into low-budget decoding. A frozen teacher and a trainable student start from the same checkpoint and receive the same prompt �. Their budgets are $B _ { T } = \left( L _ { T } , T _ { T } \right)$ and $B _ { S } = \left( L _ { S } , T _ { S } \right)$ , where $L _ { S } < L _ { T }$ . The high-budget teacher follows a trajectory that is less affected by Budget-Induced Premature Commitment; its states therefore encode more reliable decisions about both what to predict and where to commit.

![](images/45cbe60a9e7556d7d6db431004785dffbf16d02524ee980d4575255ddefc39a7.jpg)  
Figure 5: HiLo-RD overview. Left: Budget-Induced Premature Commitment. For the same prompt, low-budget decoding concentrates attention near its response boundary $P _ { \mathrm { l o w } }$ , fixes an incorrect answer early, and produces a response that fills its allocated window. With a larger budget, attention remains on earlier reasoning positions; the same model commits the correct answer at the final decoding stage and produces a much shorter response. Colors indicate normalized commitment time. Right: Cross-budget representation self-distillation. A frozen teacher processes a highbudget state, while the trainable student receives the crop ending at $P _ { \mathrm { l o w } }$ . Hidden-state MSE over the cumulative teacher-committed set $C _ { t + 1 }$ , including newly and previously committed positions, distills the reliable high-budget trajectory encoded in the teacher representations into the low-budget student. The trained student follows a coherent low-budget path and reaches the correct answer.

For each retained teacher state $y _ { t }$ , the student receives its cropped response state $y _ { t , 1 : L _ { S } }$ . The overlapping positions preserve their token values and position indices, yielding an exact cross-budget correspondence. Let $H _ { \theta }$ denote the final normalized hidden states before the shared vocabulary head. At each response position $j \leq L _ { S }$ , the paired representations are

$$
h _ { t , j } ^ { T } = H _ { \bar { \theta } } ( q , y _ { t } ) _ { P + j } ,\tag{4}
$$

$$
h _ { t , j } ^ { S } = H _ { \theta } ( q , y _ { t , 1 : L _ { S } } ) _ { P + j } ,\tag{5}
$$

where $\bar { \theta }$ is the frozen teacher. The teacher representation is computed over its high-budget trajectory before cropping, whereas the student must produce its representation from the low-budget view.

We supervise the cumulative positions committed by the teacher after update �: $m _ { t , j } = 1 \{ j \in C _ { t + 1 } \}$ and $\begin{array} { r } { n _ { t } = \sum _ { j = 1 } ^ { L _ { S } } m _ { t , j } } \end{array}$ . This includes both previously committed positions and newly selected positions whose inputs in � remain masked. For $n _ { t } > 0 ,$ , HiLo-RD minimizes

$$
\ell _ { t } = \frac { 1 } { d n _ { t } } \sum _ { j = 1 } ^ { L _ { S } } m _ { t , j } \left\| h _ { t , j } ^ { S } - \mathrm { s g } [ h _ { t , j } ^ { T } ] \right\| _ { 2 } ^ { 2 } ,\tag{6}
$$

where � is the hidden dimension and sg stops the teacher gradient. We average this loss over valid retained states and contributing trajectories. The high-budget trajectory is encoded directly in each teacher representation $h _ { t , j } ^ { T } \colon$ it reflects the trajectory’s bidirectional context, commitment history, and its reliable decisions about both what to predict and where to commit. Minimizing Eq. (6) forces the low-budget representation $h _ { t , j } ^ { S }$ to internalize this high-budget trajectory information at the corresponding position. HiLo-RD is therefore a trajectory distillation method at the representation level, rather than a loss on isolated tokens or separately specified commitment labels. The shared vocabulary head maps the resulting representation alignment to locally aligned token distributions; Appendix B.3 gives the corresponding bound. Since commitment confidence is also derived from these representations, the same objective improves both token prediction and commitment order.

## 4.2 EFFICIENT TRAJECTORY ACQUISITION

Full-completion decoding continues until every allocated position is filled, even when the readable response is already complete. An early EOT alone is insufficient to stop: an EOT token may be committed before earlier reasoning positions have been resolved. We instead require a clean EOT: the leftmost committed EOT and every preceding response position must be committed. With unchanged updates and irreversible commitment, further decoding cannot alter the response extracted before that EOT (Proposition 1).

Teacher acquisition can additionally stop when its first �<sub>S</sub> response positions are filled, since these define the supervision window. This cap preserves the completed prefix but may truncate a longer solution; it is distinct from response-preserving clean-EOT stopping. Before termination, all positions in the teacher’s full window remain eligible for commitment. Both rules are applied during rollout, so they avoid forwards rather than merely discarding stored states.

If stopping reduces the realized updates from � to �, the cost becomes $R F ( P + L )$ instead of $T F ( P + L )$ under Section 2.2. Each forward still processes the full window. Efficient acquisition therefore reduces continuation cost, while distillation improves the quality attainable at a smaller budget. Exact preservation assumes matched decoding and first-EOT extraction; implementation-specific extraction details are given in Appendix C.3.

## 5 EXPERIMENTS

We evaluate HiLo-RD through four research questions covering low-budget reasoning quality, transfer, rollout efficiency, and iterative self-distillation.

RQ1 (Performance): Can HiLo-RD distill high-budget reasoning into accurate, compact low-budget decoding?

RQ2 (Generalization): Do these decoding gains generalize to out-of-domain code generation?

RQ3 (Efficiency): Can efficient trajectory acquisition cut pure-diffusion rollout cost without quality loss?

RQ4 (Self-Evolving): Can successive self-distillation drive self-evolving quality–budget refinement?

## 5.1 EXPERIMENTAL SETUP

Data Preparation. We train separately on the full GSM8K training set (Cobbe et al., 2021) and the original MATH training set (Hendrycks et al., 2021), and evaluate the corresponding checkpoints on the GSM8K test set and MATH500 (Lightman et al., 2024), respectively. For out-of-domain evaluation, we directly test mathematics-trained checkpoints on HumanEval (Chen et al., 2021) and MBPP (Austin et al., 2021), without additional code-domain training.

Baselines. We compare HiLo-RD with the original LLaDA-8B-Instruct (vanilla) (Nie et al., 2025) and four post-training methods: masked supervised fine-tuning (Masked SFT), Diffu-GRPO (Zhao et al., 2025), on-policy self-distillation (OPSD) (Zhao et al., 2026), and dOPSD (Dat et al., 2026). All comparisons use pure diffusion decoding. For vanilla and HiLo-RD, we additionally compare full decoding with the fast-rollout stopping rule in Section 4.2 to assess its effect on quality and efficiency. Section 2.3 situates these post-training objectives.

Table 1: Low-budget reasoning and transfer. All methods use the same low-budget pure diffusion setting and fast rollout. Acc. and Pass@1 are percentages; RTL is mean readable tokens, and Time is seconds per request. Domain separates mathematics and code; HiLo-RD identifies our method; Best marks the highest task score or shortest response per benchmark. Bold and underlined values mark the best and second-best distinct values in each column, including ties.
<table><tr><td></td><td colspan="3">GSM8K</td><td colspan="3">MATH500</td></tr><tr><td>Method</td><td>Acc.↑</td><td>RTL↓</td><td>Time↓</td><td>Acc.↑</td><td>RTL↓</td><td>Time↓</td></tr><tr><td colspan="7">In-domain mathematics reasoning</td></tr><tr><td>vanilla</td><td>68.84</td><td>362.8</td><td>73.35</td><td>28.40</td><td>479.6</td><td>31.09</td></tr><tr><td>Masked SFT</td><td>70.13</td><td>366.8</td><td>72.10</td><td>30.60</td><td>478.3</td><td>31.03</td></tr><tr><td>Diffu-GRPO</td><td>72.93</td><td>325.4</td><td>63.53</td><td>29.20</td><td>482.1</td><td>31.19</td></tr><tr><td>OPSD</td><td>69.90</td><td>360.7</td><td>73.25</td><td>30.20</td><td>478.5</td><td>30.96</td></tr><tr><td>dOPSD</td><td>70.05</td><td>363.9</td><td>73.47</td><td>29.40</td><td>478.4</td><td>30.96</td></tr><tr><td>HiLo-RD</td><td>78.24</td><td>271.3</td><td>71.71</td><td>32.40</td><td>458.2</td><td>29.90</td></tr><tr><td></td><td colspan="3">HumanEval</td><td colspan="3">MBPP</td></tr><tr><td>Method</td><td>Pass@1↑</td><td>RTL↓</td><td>Time↓</td><td>Pass@1↑</td><td>RTL↓</td><td>Time↓</td></tr><tr><td colspan="7">Out-of-domain code generation</td></tr><tr><td>vanilla</td><td>45.73</td><td>510.0</td><td>35.59</td><td>38.60</td><td>151.0</td><td>19.51</td></tr><tr><td>Masked SFT</td><td>47.56</td><td>510.0</td><td>35.53</td><td>37.40</td><td>148.9</td><td>19.36</td></tr><tr><td>Diffu-GRPO</td><td>48.78</td><td>510.0</td><td>35.54</td><td>38.40</td><td>127.1</td><td>16.61</td></tr><tr><td>OPSD</td><td>46.95</td><td>508.4</td><td>35.52</td><td>38.00</td><td>146.9</td><td>19.11</td></tr><tr><td>dOPSD</td><td>50.00</td><td>510.0</td><td>35.64</td><td>37.80</td><td>143.1</td><td>18.73</td></tr><tr><td>HiLo-RD</td><td>49.39</td><td>508.2</td><td>35.45</td><td>41.40</td><td>84.7</td><td>11.20</td></tr></table>

Training and Evaluation. We perform parameter-efficient adaptation with LoRA (Hu et al., 2022), optimizing the adapter parameters with AdamW (Loshchilov & Hutter, 2019) while keeping the backbone weights frozen. The HiLo-RD training runs use eight NVIDIA A100 80GB GPUs. We measure mathematical reasoning performance using accuracy and code-generation performance using pass@� in the main comparison. We also report mean readable token length (RTL) and mean wall-clock time per request to characterize response length and inference cost. Additional training and evaluation details are provided in Appendix C.

## 5.2 LOW-BUDGET PERFORMANCE AND CROSS-DOMAIN TRANSFER (RQ1 & RQ2)

The results and comparisons of the model performance using different methods under the same lowbudget pure diffusion setting with fast rollout are shown in Table 1. HiLo-RD improves over vanilla on all four benchmarks while producing shorter readable responses.

In-domain mathematical reasoning. We observe that HiLo-RD improves low-budget mathematical reasoning across datasets of different difficulty, achieving the highest accuracy and shortest readable responses among the compared methods on both GSM8K and MATH500. The joint improvement matters because it shows that better reasoning need not come from extending the readable solution: higher-budget representations can instead help the same model use its smaller window more effectively. Compared with the other post-training objectives, representation self-distillation yields a more favorable balance between correctness and response length. This supports the motivation developed in Section 3: useful computation exposed by a larger budget can guide low-budget token predictions and commitment decisions, rather than requiring the larger budget at deployment. These gains align with the mechanism of Budget-Induced Premature Commitment: stronger lowbudget reasoning no longer requires longer readable responses, although the attainable degree of compression remains task-dependent.

Out-of-domain code generation. The benefits extend to code generation without further codedomain training, suggesting that the learned improvement is not confined to mathematics-specific answer patterns. On MBPP, HiLo-RD combines the strongest pass@� with the shortest responses.

On HumanEval, it remains competitive with dOPSD, which has a slight advantage in pass@�, while the reduction in readable length is marginal. This contrast shows that quality transfer need not be accompanied by equally strong compression across domains. Nevertheless, improvement over vanilla on both benchmarks supports the transferability of the decoding behavior learned through mathematics-based representation distillation. The same model’s higher-budget computation thu supplies useful supervision beyond the training domain, without teacher-only reference solutions.

## 5.3 EFFICIENT TRAJECTORY ACQUISITION (RQ3)

Efficient trajectory acquisition reduces unnecessary decoding after response completion. Figure 6 compares full and fast rollout for vanilla and HiLo-RD; within each dataset, movement toward the upper left indicates higher quality and lower latency, while smaller linear marker size indicates shorter RTL. Fast rollout reduces the recorded latency most clearly on GSM8K and MBPP, with little change on HumanEval. In particular, the GSM8K HiLo-RD pair has identical accuracy and RTL under full and fast rollout. This illustrates the distinction between the two efficiency gains: distillation improves what the model generates, whereas completion-aware stopping can save computation without shortening the completed response. Reported timing differences remain subject to crossserver normalization.

![](images/b893d3baa10af0ff33def4d3533f9444a4532b508dd2b64e2fadc73f2f3da0d0.jpg)  
Figure 6: Quality–latency trade-off. Color denotes dataset; shape and fill identify model and rollout. Linear marker size represents RTL. Lines pair Full/Fast results. Timings await normalization.

## Why savings vary across tasks. The

task-dependent pattern is consistent with the cost reduction from $T F ( P + L )$ to $R F ( P + L )$ : early completion leaves more updates to skip, and each skipped update saves a forward pass over the full prompt and response window. Short MBPP responses leave substantial unnecessary continuation, whereas HumanEval responses nearly fill the window. Longer prompts, such as those used for GSM8K, also increase the saving per skipped update when other settings are fixed. RTL alone does not determine the stopping step, since commits outside the eventual readable prefix also consume updates. Thus, the observed efficiency depends jointly on response completion, commitment order, and input length. Under matched decoding and extraction, the stopping rule preserves the completed prefix (Proposition 1); applied to teacher acquisition, it similarly avoids continuation after the student-sized supervision prefix is complete. This makes efficient acquisition complementary to representation distillation, rather than a substitute for learning better low-budget decoding.

## 5.4 TOWARD SELF-EVOLVING QUALITY–BUDGET REFINEMENT (RQ4)

We investigate whether HiLo-RD can bootstrap its own supervision across successive distillation cycles. Starting from the unadapted model C0, each cycle uses the preceding model to acquire new high-budget trajectories and distills them back into low-budget decoding. All cycles reuse the same fixed subset of 1,000 GSM8K training examples, so improvements arise from recursively refining the model’s own cross-budget supervision rather than introducing additional data. Figure 7 reports C0–C3, with exact measurements provided in Appendix D.

Successive distillation progressively improves the quality–budget frontier. From C0 to C3, lowbudget accuracy rises while readable length and latency fall; high-budget decoding remains above C0 accuracy and becomes progressively shorter and faster. Because every cycle reuses the same data, these coupled gains arise from recursively improving the model’s own supervision. This provides a concrete path toward self-evolving budget adaptation without an external teacher or additional training examples.

![](images/9267acbdab465e73c831b81511bbb689b30c21d455ba1f3ac23ffcc7faca02d1.jpg)

![](images/4298340f1dc0fa57b6c66c67ac58ad446c4e07d8bc82609dc9d003cca99d1ddb.jpg)

![](images/606336d22d042a1ebc645daa48cab7922b0637acd77540b1de891e0ae5d3c3ce.jpg)  
Figure 7: Successive cross-budget self-distillation on GSM8K. Each cycle reuses the same 1,000 training examples and acquires new high-budget supervision from the preceding model. Across C0– C3, low-budget accuracy rises while readable length and latency fall; high-budget decoding remain stronger than C0 and becomes progressively more efficient.

## 6 CONCLUSION AND FUTURE WORK

We identify Budget-Induced Premature Commitment in masked diffusion language models: under a low yet sufficient response budget, the model can fix response-tail fragments before the reasoning that supports them is resolved. Decoding traces, attention diagnostics, and commitment-order interventions trace how this early decision redirects subsequent bidirectional reasoning and degrades the resulting trajectory. Guided by this mechanism, HiLo-RD distills representations from the same model’s reliable high-budget trajectories into low-budget decoding. The resulting model reasons more accurately with shorter responses across mathematical reasoning and code generation, realizing Think Wide, Reason Lean; completed-prefix stopping further avoids unnecessary decoding work.

Successive distillation further improves the quality–budget frontier using the model’s own trajectories, opening a path toward self-evolving diffusion reasoning. Incorporating richer intermediate supervision may help sustain this refinement across longer sequences of self-distillation cycles.

## AI USE STATEMENT

In this work, we used generative AI tools solely to polish the manuscript. We did not use generative AI tools to generate or analyze experimental results, produce research artifacts, or make scientific claims. At least two authors independently conducted a detailed review of every AI-assisted passage, checking its accuracy, clarity, and consistency with the reported methods, experiments, and claims. We take responsibility for the final content of this work, including text, claims, or artifacts produced with the aid of generative AI.

## REFERENCES

Marianne Arriola, Subham Sekhar Sahoo, Aaron Gokaslan, Zhihan Yang, Zhixuan Qi, Jiaqi Han, Justin T Chiu, and Volodymyr Kuleshov. Block diffusion: Interpolating between autoregressive and diffusion language models. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=tyEyYT267x.

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, and Charles Sutton. Program synthesis with large language models, 2021. URL https://arxiv.org/abs/2108.07732.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob Mc-Grew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. Evaluating large language models trained on code, 2021. URL https://arxiv.org/abs/2107.03374.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. CoRR, abs/2110.14168, 2021. URL https://arxiv.org/abs/2110.14168.

Phuong Tuan Dat, Qi Li, and Xinchao Wang. dopsd: On-policy self-distillation for diffusion language models. arXiv preprint arXiv:2607.04428, 2026.

Shansan Gong, Shivam Agarwal, Yizhe Zhang, Jiacheng Ye, Lin Zheng, Mukai Li, Chenxin An, Peilin Zhao, Wei Bi, Jiawei Han, Hao Peng, and Lingpeng Kong. Scaling diffusion language models via adaptation from autoregressive models. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id= j1tSLYKwg8.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. In J. Vanschoren and S. Yeung (eds.), Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks, volume 1, 2021. URL https://datasets-benchmarks-proceedings.neurips.cc/paper\_files/ paper/2021/file/be83ab3ecd0db773eb2dc1b0a17836a1-Paper-round2.pdf.

Edward J Hu, yelong shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id= nZeVKeeFYf9.

Jinsong Li, Xiaoyi Dong, Yuhang Zang, Yuhang Cao, Jiaqi Wang, and Dahua Lin. Beyond fixed: Training-free variable-length denoising for diffusion large language models, 2025. URL https: //arxiv.org/abs/2508.00819.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview. net/forum?id=v8L0pN6EOi.

Aiwei Liu, Minghua He, Shaoxun Zeng, Sijun Zhang, Linhao Zhang, Chuhan Wu, Wei Jia, Yuan Liu, Xiao Zhou, and Jie Zhou. Wedlm: Reconciling diffusion language models with standard causal attention for fast inference, 2025. URL https://arxiv.org/abs/2512.22737.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id= Bkg6RiCqY7.

Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, JUN ZHOU, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Large language diffusion models. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview. net/forum?id=KnqiC0znVF.

Shen Nie, Qiyang Min, Shaoxuan Xu, Zihao Huang, Yuxuan Song, Yong Shan, Yankai Lin, Wayne Xin Zhao, Chongxuan Li, and Ji-Rong Wen. Improved large language diffusion models, 2026. URL https://arxiv.org/abs/2606.25331.

Subham Sekhar Sahoo, Marianne Arriola, Aaron Gokaslan, Edgar Mariano Marroquin, Alexander M Rush, Yair Schiff, Justin T Chiu, and Volodymyr Kuleshov. Simple and effective masked diffusion language models. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview.net/forum?id=L4uaAR4ArM.

Anej Svete and Ashish Sabharwal. On the reasoning abilities of masked diffusion language models. In The Fourteenth International Conference on Learning Representations, 2026. URL https: //openreview.net/forum?id=BVnIsh4Nz1.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in Neural Information Processing Systems, 30, 2017. URL https://arxiv.org/abs/1706.03762.

Jiacheng Ye, Zhihui Xie, Lin Zheng, Jiahui Gao, Zirui Wu, Xin Jiang, Zhenguo Li, and Lingpeng Kong. Dream 7b: Diffusion large language models. arXiv preprint arXiv:2508.15487, 2025.

Tunyu Zhang, Xinxi Zhang, Ligong Han, Haizhou Shi, Xiaoxiao He, Zhuowei Li, Hao Wang, Kai Xu, Akash Srivastava, Chengzhi Mao, Hao Wang, Vladimir Pavlovic, and Dimitris N. Metaxas. Few-step diffusion language models via trajectory self-distillation, 2026. URL https://arxiv.org/abs/2602.12262.

Siyan Zhao, Devaansh Gupta, Qinqing Zheng, and Aditya Grover. d1: Scaling reasoning in diffusion large language models via reinforcement learning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum? id=7ZVRlBFuEv.

Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, and Aditya Grover. Self-distilled reasoner: On-policy self-distillation for large language models. In Forty-third International Conference on Machine Learning, 2026. URL https://openreview.net/forum? id=Jpxfof0EaS.

## A BUDGET-INDUCED PREMATURE COMMITMENT: DIAGNOSTIC PROTOCOLS AND ADDITIONAL EVIDENCE

## A.1 MAIN DIAGNOSTIC COHORT AND CONTROLLED COMPARISONS

The three diagnostics in Figures 2–4 use 155 randomly sampled cases exhibiting Budget-Induced Premature Commitment during low-budget decoding. These paired cases have an incorrect native $L = T = 5 1 2$ response and a correct native $L = T = 1 0 2 4$ response. All three diagnostics use this same cohort; recovery rates in Figure 4 are measured within these failure/success pairs. Figure 1 reports the population-level length and accuracy patterns. The complementary pilot and illustrative traces below use separate cohorts.

Window expansion before commitment. At the first ordinary response-tail commitment in each low-budget trace, we evaluate the immediately preceding state using the query at the selected tail position. The extended view appends 512 visible MASK positions while retaining the prompt, original token identities, position indices, and query. Native high-budget decoding is replayed to the same per-case update index and uses its own next-selected position as the query. The extension therefore tests the same state’s sensitivity to visible response space, while the native comparison characterizes the computation reached along a high-budget trajectory. Attention is averaged over heads within each case and then across cases, using 16-position generation-key bins, all 32 layers, and a shared linear scale. The highlighted interval is response positions 500–520. Values are raw attention percentages over each condition’s full key set. Shared-key normalization and fixed-target confidence probes below separate changes in attention normalization from changes in commitment preference.

Response tail and replay state. Let � be the realized response extent, ending at the first spatial EOT, otherwise the first EOS, or the last ordinary token if neither is present. With zero-based response indices, the tail set is

$$
{ \mathcal { E } } = \{ e - \lceil 0 . 1 e \rceil , \ldots , e - 1 \} , \qquad t _ { \star } = \operatorname* { m i n } \{ t : j _ { t } \in { \mathcal { E } } , ~ v _ { t } \notin \{ \mathrm { ~ [ M A S K ] } , \mathrm { E O S } , \mathrm { E O T } \} \} .\tag{7}
$$

Here $( j _ { t } , \nu _ { t } )$ is the position and token committed at update �. This tail follows the realized response; the allocated-window boundary is at �. They lie close together in boundary-filling low-budget traces. Replay constructs $\mathbf { X } _ { \star }$ using only commitments with $t < t _ { \star }$ , leaving the event position masked. The original tail set remains fixed in subsequent probes and interventions.

Algorithm 1 Same-state visible-window expansion   
Input: Model $f _ { \theta } ,$ prompt q, low-budget trace �, lengths $L _ { \ell } < L _ { h }$   
Output: Attention and prediction measurements before the next commitment   
1: Identify tail � and first ordinary tail event $( t _ { \star } , j _ { \star } , \upsilon _ { \star } )$ in �   
2: x ← [q; [MASK]<sup>�ℓ</sup>]   
3: for each commitment $( t , j _ { t } , \upsilon _ { t } )$ in � with $t < t _ { \star }$ do   
4: Set response position $j _ { t }$ in $\mathbf { X } _ { \star }$ to �<sub>�</sub>   
5: end for   
6: $\mathbf { x } ^ { + } \gets [ \mathbf { x } _ { \star } ; [ \mathsf { M A S K } ] ^ { L _ { h } - L _ { \ell } } ]$   
7: $\mathcal { D } _ { \ell } \gets f _ { \theta } ( \mathbf { x } _ { \star } )$ ⊲ Original window   
8: $\mathcal { D } _ { + }  f _ { \theta } ( \mathbf { x } ^ { + } )$ ⊲ Visible extension   
9: $\mathcal { D } _ { 0 } \gets f _ { \theta } ( \mathbf { x } ^ { + } ;$ ; appended keys hidden) ⊲ Padding control   
10: Exclude appended positions from selection in $\mathcal { D } _ { 0 }$   
11: Fix the front candidate using $\mathcal { D } _ { \ell }$   
12: for $\mathcal { D } \in \{ \mathcal { D } _ { \ell } , \mathcal { D } _ { + } , \mathcal { D } _ { 0 } \}$ do   
13: Record attention at query $j _ { \star }$ and probabilities of the fixed targets   
14: Record commitment scores and the next eligible position   
15: Compare scores restricted to the original masked positions   
16: end for   
17: return Recorded measurements

Appended visible masks participate in bidirectional attention and are eligible for the diagnostic nextposition selection. The additional shared-position comparison restricts this competition to the original window. These are single-forward probes; native high-budget trajectories are decoded separately from initialization.

Attention after commitment. The replay retains fixed front queries selected from unfilled positions and fixed tail/prefix key sets at Before, After, and offsets $+ 2 , + 4 , + 8 , + 1 6 , + 3 2$ . Each value is a paired change from Before, averaged over heads and queries within a case and then over valid cases. Coverage is 145/155 cases. Immediately after commitment, tail attention increases by 1.908 percentage points on average, with increases in 140/145 cases; prefix attention changes by −0.413 points. $\mathbf { A } \mathbf { t } + 3 2$ , the respective changes are +4.194 and −1.875 points. Later offsets describe the evolving trajectory, including additional natural commitments; the immediate Before/After pair measures the local change around the first tail commitment.

Interventions on commitment order. Low Budget retains the archived low-budget schedule. Tail Delay prevents selection of the original response-tail positions for the first 10% of the remaining low-budget updates. Matched-order Control changes decisions at the same intervention offsets by excluding its own current highest-scoring position, without a tail-specific rule. Both branches start from x<sub>★</sub> and retain $L = T = 5 1 2$ , one commitment per update, and the full decoding schedule. Correct outcomes number 67 and 47, respectively. The paired difference is 12.903 percentage points, with a 95% interval of [7.742,18.710] from 2,000 bootstrap resamples of questions. This tests recovery through commitment order at fixed budget.

Matched intervention schedule. The construction in Algorithm 2 sets $R ~ = ~ T ~ - ~ t _ { \star }$ and $D \ =$ max(1, ⌈0.1�⌉). Let Top(x, A) return the highest-confidence currently masked position in eligible set A, and let $\widehat { \nu } _ { j } ( \mathbf { x } )$ be its predicted token. The same predictor and confidence rule are used in both branches. Excluding a position changes eligibility for that update; it leaves all committed tokens intact.

Algorithm 2 Tail Delay and Matched-order Control   
Input: Pre-event state x<sub>★</sub>, original tail E, remaining updates �, delay �   
Output: Tail-delay continuation $\mathbf { x } ^ { \mathrm { d e l a y } }$ and matched-order continuation $\mathbf { x } ^ { \mathrm { c o n t r o l } }$   
1: $\mathbf { x } \gets \mathbf { x _ { \star } } ; \mathcal { K } \gets \emptyset$   
2: for $k = 0 , \ldots , R - 1 { \bf d o }$ ⊲ Tail Delay   
3: $\mathcal { M }  \{ j : x _ { j } = [ \mathtt { M A S K } ] \} ; i  \mathrm { T o p } ( \mathbf { x } , \mathcal { M } ) ; j  i$   
4: i $\mathbf { f } \ k < D$ and $M \backslash \mathcal { E } \neq \emptyset$ then   
5: $j \gets \mathrm { T o p } ( \mathbf { x } , M \setminus \mathcal { E } )$   
6: end if   
7: if � ≠ � then   
8: ${ \mathcal { K } } \gets { \mathcal { K } } \cup \{ k \}$ ⊲ Record changed decisions   
9: end if   
10: $x _ { j } \gets \widehat { \nu } _ { j } ( { \bf x } )$   
11: end for   
12: $\mathbf { x } ^ { \mathrm { { d e l a y } } } \gets \mathbf { x } ; \mathbf { x } \gets \mathbf { x } _ { \star }$   
13: for $k = 0 , \ldots , R - 1$ do ⊲ Matched-order Control   
14: $\mathcal { M }  \{ j : x _ { j } = [ \mathtt { M A S K } ] \} ; i  \mathrm { T o p } ( \mathbf { x } , \mathcal { M } ) ; j  i$   
15: if � ∈ K and M \ {�} ≠ ∅ then   
16: � ← Top(x, M \ {�})   
17: end if   
18: $x _ { j } \gets \widehat { \nu } _ { j } ( { \bf x } )$   
19: end for   
20: $\mathbf { x } ^ { \mathrm { c o n t r o l } }  \mathbf { x }$   
21: return $\mathbf { x } ^ { \mathrm { d e l a y } } , \mathbf { x } ^ { \mathrm { c } }$ ontrol

The control’s excluded position is recomputed from its own state and can itself lie in the tail. Thus, the comparison is between a tail-defined eligibility constraint and a matched schedule of currentwinner exclusions. When an exclusion would leave no candidate, the decoder uses its default winner and records the fallback. The reference decoder predicts tokens by argmax, suppresses EOS/EOT in the confidence calculation, masks already committed positions, and then chooses one eligible position. These order interventions operate within the original window throughout.

## A.2 COMPLEMENTARY CHECKS ON VISIBLE SPACE AND TAIL READING

Visible masks versus invisible padding. This check uses 20 low-budget failure/high-budget short-success pairs. At the state immediately before the first ordinary tail commitment, we retain the same prompt, committed contents, event-tail query, and token targets across the original window, visible-mask expansion, and invisible padding. Visible expansion reduces mean tail-target probability from 46.30% to 0.25%, while invisible padding gives 46.24%. The fixed front target changes from 47.65% to 52.12% with visible expansion and to 47.79% with invisible padding. The next selected position moves away from the original tail in 19/20 cases, and the confidence margin reverses even when competition is restricted to original positions. These comparisons isolate the effect of visible generation space from added tensor length and competition from new positions. Algorithm 1 specifies the construction.

Temporary restriction of tail reading. A complementary intervention blocks front queries from attending to committed ordinary tail keys in every layer for an initial interval. It repairs one case and breaks one among the 20 pairs, leaving aggregate accuracy unchanged. This intervention restricts access to already committed content; the order interventions in Figure 4 instead postpone its commitment while retaining normal bidirectional attention.

## A.3 IRREVERSIBLE COMMITMENT AND REACHABLE COMPLETIONS

For a partial response, let

$$
\mathcal { F } _ { t } ( L ) = \{ \mathbf { y } \in \mathscr { V } ^ { L } : y _ { j } = y _ { t , j } \mathrm { ~ f o r ~ e v e r y ~ } j \in C _ { t } \} . \qquad \mathcal { F } _ { t + 1 } ( L ) \subseteq \mathcal { F } _ { t } ( L ) .\tag{8}
$$

If $\mathcal { G } ( \mathbf { q } , L )$ denotes correct completions under a specified evaluator and extraction rule, then $\mathcal { F } _ { t + 1 } ( L ) \cap \mathcal { G } ( \mathbf { q } , L ) \subseteq \mathcal { F } _ { t } ( L ) \cap \mathcal { G } ( \mathbf { q } , L )$ . A wrong fixed fragment can remove compatible correct continuations; its content also changes subsequent predictions and position scores. Since both correct and incorrect commitments shrink the set, the relevant question is which successful continuations remain reachable. The order interventions test whether a different sequence of commitments can recover successful paths at the same budget. Budget-Induced Premature Commitment denotes premature boundary-near commitment while the intervening reasoning remains unresolved despite sufficient response space.

## A.4 PAIRED TRACE EXAMPLES AND VISUALIZATION PROTOCOL

The paired traces in Figures 8 and 9 show 16 GSM8K responses at $L = T = 5 1 2$ and $L = T = 1 0 2 4$ The supplied semantic-answer audit verifies an incorrect low-budget answer and a correct highbudget answer in each pair. Selection additionally requires the low-budget answer to be fully committed within the first 35% of response decoding, the high-budget answer in the final 10%, and a tail-before-prefix inversion score of at least 0.85 and at most 0.08, respectively. These examples illustrate an outcome and ordering contrast, not the prevalence of Budget-Induced Premature Commitment.

Reading the figures. Tokens follow final response order. Color encodes $( s + 1 ) / R _ { \mathrm { r e s p } }$ , where � is the stored zero-based commitment update and $R _ { \mathrm { r e s p } }$ is the exported response-completion step. Purple is early, yellow is late, and gray marks special tokens. Normalization is separate for each response: equal colors do not imply equal update counts or equal fractions of the configured budget. The ribbon covers the realized response; text excerpts show its beginning and end. Headers report readable token length (RTL), while panel widths are rescaled for layout. RTL is 510 in every selected low-budget case and ranges from 162 to 355 at the high budget; neither RTL nor ribbon width is a count of model calls.

The low-budget ribbons expose early answer commitments followed by middle infilling. The paired high-budget responses commit their answers near completion and are shorter. For example, gsm8k test 0314 fixes the wrong answer 20 early at the low budget but reaches 30 near highbudget completion. The audited answer, rather than an automatically extracted number, is displayed. Extraction false negatives such as gsm8k test 0527 are excluded. In gsm8k test 1248, the reviewed low-budget answer is 251 whereas the archived grader extracts 15; both are incorrect relative to 750. This audit prevents grading artifacts from being presented as reasoning failures without retroactively changing benchmark scores.

![](images/4adcf32abd34572b83382a665263bf6e76c209d9dfafd5f4f6dbb2db749d5f57.jpg)  
Figure 8: Paired premature-commitment traces, Cases 1–8. Left: low budget; right: high budget. Headers give RTL. Ribbons and excerpts encode commitment timing in final response order: purple is early, yellow late, and gray denotes special tokens. Timing is normalized per response, and panel widths are rescaled. Cases are selected for contrasting outcomes and commitment order.

![](images/363a19930601ae164ff7312afe3337c6dac1b370cd6c4621092f84a4e7584bc9.jpg)  
Figure 9: Paired premature-commitment traces, Cases 9–16. The encoding follows Figure 8. Low-budget incorrect answers are fixed before much of the intervening reasoning; the paired highbudget answers are correct and committed near completion. RTL provides the absolute length com parison.

## B HILO-RD: OBJECTIVE DETAILS AND LOCAL GUARANTEES

## B.1 TRAJECTORY SAMPLING AND OBJECTIVE NORMALIZATION

Within a training phase, the frozen teacher generates

$$
\begin{array} { r } { { \boldsymbol \xi } = ( { \bf y } _ { 0 } , \ldots , { \bf y } _ { R } ) \sim { \sf P } _ { \bar { \theta } } ^ { B _ { \mathrm { T } } } ( \cdot \mid { \bf q } ) , \qquad R \leq T _ { \mathrm { T } } . } \end{array}\tag{9}
$$

The reference fast-acquisition implementation retains at most $K = 8$ states, including the initial all-MASK state. Each paired forward uses the pre-update state $\mathbf { y } _ { t } \mathbf { . }$ the teacher processes the full window and the student processes its crop. Supervision eligibility is determined after the corresponding update:

$$
m _ { t , j } = { \bf 1 } \{ j \in C _ { t + 1 } \} , \qquad n _ { t } = \sum _ { j = 1 } ^ { L _ { \mathrm { S } } } m _ { t , j } .\tag{10}
$$

This cumulative mask includes previously committed positions and positions newly committed in the update. A newly selected position can therefore be supervised while its input is still masked. Earlier commitments receive repeated supervision as surrounding content develops. The loss in Eq. (6) normalizes by $d n _ { t } ,$ so a later state does not receive more weight solely because more positions have been filled.

Let $\underline { { \tau } } _ { + } ( \xi )$ contain the retained states with $n _ { t } > 0$ . We average their losses:

$$
\ell _ { \xi } ( \boldsymbol { \theta } ) = \frac { 1 } { | \mathcal { I } _ { + } ( \xi ) | } \sum _ { t \in \mathcal { I } _ { + } ( \xi ) } \ell _ { t } ( \boldsymbol { \theta } ; \boldsymbol { \xi } ) ,\tag{11}
$$

with $\ell _ { \xi } = 0$ if the set is empty. For a batch $\mathcal { B }$ and a trajectory gate $g ( \xi , \mathbf { q } , a ) \in \{ 0 , 1 \}$

$$
\mathcal { L } _ { \mathrm { H i L o - R D } } ( \theta ; \mathcal { B } ) = \frac { \sum _ { ( \mathbf { q } , a , \xi ) \in \mathcal { B } } g ( \boldsymbol { \xi } , \mathbf { q } , a ) \ell _ { \boldsymbol { \xi } } ( \theta ) } { \operatorname* { m a x } \{ 1 , \sum _ { ( \mathbf { q } , a , \boldsymbol { \xi } ) \in \mathcal { B } } g ( \boldsymbol { \xi } , \mathbf { q } , a ) \} } .\tag{12}
$$

The reported positive training recipe accepts teacher responses whose extracted answers pass verification. An empty accepted batch contributes zero signal. The answer label selects rollouts; it is not supplied as teacher input or used as the representation target. This filter certifies the extracted answer, not every intermediate reasoning step. Setting $g \equiv 1$ yields an unfiltered variant without correctness labels. No gradient passes through the sampled tokens, commitment decisions, or frozen teacher.

## B.2 COMPLETED-PREFIX STOPPING

The clean-EOT rule and the prefix-cap rule in Section 4.2 share a preservation property under irreversible decoding.

Proposition 1 (Completed-prefix preservation) Considerfull andfast decoders with the same initialization, update schedule, tie-breaking, and coupled randomness. Assume that committed tokens are never revised. If fast decoding stops after � updates, its states and actions through � equal the corresponding prefix offull decoding. Every fully committed response prefix is unchanged by continuing the full decoder. If the leftmost committed EOT is at position � and all preceding positions are committed, extraction strictly before the first EOT returns the same token sequence after either run.

Proof. Before termination, both decoders apply identical updates to identical states. Induction gives trajectory-prefix equality. Irreversibility then preserves every token in the completed prefix. Under clean EOT, all positions through � are fixed, so no earlier EOT can subsequently appear and extraction before � is unchanged. □

Stopping at a completed cap $c < L$ preserves only the first � positions and may truncate a solution that continues beyond them. The teacher can use $c \ = \ L _ { S }$ while retaining full-window attention and allowing commitments outside that prefix before termination. The evaluation driver’s default cap instead equals its allocated response length. Both rules must execute inside the decoding loop; discarding later states after a complete rollout does not save forward passes.

The output guarantee requires the stated extraction rule. The archived evaluation driver decodes the returned response array with skip special tokens=True, without explicitly cutting at the first EOT. Already committed post-EOT text can then enter grading. Prefix preservation alone does not establish identical scores for that extractor; raw outputs, extracted answers, and stop reasons must be compared under a common protocol.

## B.3 REPRESENTATION MATCHING AND TOKEN DISTRIBUTIONS

The vocabulary head translates a representation difference into a logit difference. The following bound formalizes this local relation without assuming identical autonomous decoding paths.

Proposition 2 (Local token-distribution control) $L e t V = \vert \mathcal { V } \vert$ and let $\boldsymbol { z } ^ { \mathrm { T } } = \boldsymbol { W } \boldsymbol { h } ^ { \mathrm { T } } + \boldsymbol { b }$ and $z ^ { \mathrm { S } } =$ $W h ^ { S } + b$ befinite logits under the samefixed head � $\in \mathbb { R } ^ { V \times d } . \stackrel { \cdot } { F o r } \tau > 0 ,$ , define $p _ { \tau } ^ { \mathrm { T } } =$ softmax $( z ^ { \mathrm { T } } / \tau )$ $p _ { \tau } ^ { \mathrm { S } } = \mathrm { s o f t m a x } ( z ^ { \mathrm { S } } / \tau ) , \mathsf { C } = I _ { V } - \mathbf { 1 } \mathbf { 1 } ^ { \top } / V ,$ and $e = \| h ^ { \mathrm { S } } - h ^ { \mathrm { T } } \| _ { 2 } ^ { 2 } / d$ . Then

$$
D _ { \mathrm { K L } } ( p _ { \tau } ^ { \mathrm { T } } \| p _ { \tau } ^ { \mathrm { S } } ) \leq \frac { d \| \mathsf { C } W \| _ { \mathrm { o p } } ^ { 2 } } { 4 \tau ^ { 2 } } e .\tag{13}
$$

The same bound holds with the KL arguments exchanged.

Proof. Let $\begin{array} { r } { A ( u ) = \log \sum _ { \nu = 1 } ^ { V } e ^ { u _ { \nu } } } \end{array}$ . Its gradient is softmax, and its Hessian is $\mathrm { d i a g } ( p ) - p p ^ { \top }$ , where $p = \operatorname { s o f t m a x } ( u )$ . For any unit vector �, the Hessian quadratic form is the variance of �<sub>�</sub> under $i \sim p$ so

$$
w ^ { \top } \nabla ^ { 2 } A ( u ) w \leq \frac { ( \operatorname* { m a x } _ { i } w _ { i } - \operatorname* { m i n } _ { i } w _ { i } ) ^ { 2 } } { 4 } \leq \frac { 1 } { 2 } .\tag{14}
$$

Thus, � has a 1/2-Lipschitz gradient and

$$
\begin{array} { r } { A ( \nu ) - A ( u ) - \nabla A ( u ) ^ { \top } ( \nu - u ) \leq \frac { 1 } { 4 } \| \nu - u \| _ { 2 } ^ { 2 } . } \end{array}\tag{15}
$$

Set $u = \mathsf { C } z ^ { \mathrm { T } } / \tau$ and $\upsilon = \mathsf { C z } ^ { \mathrm { { S } } } / \tau$ . Centering leaves softmax unchanged. Substitution gives

$$
D _ { \mathrm { K L } } ( p _ { \tau } ^ { \mathrm { T } } \| p _ { \tau } ^ { \mathrm { S } } ) = A ( \nu ) - A ( u ) - \nabla A ( u ) ^ { \top } ( \nu - u )\tag{16}
$$

$$
\leq \frac { \| \mathsf { C } ( z ^ { \mathrm { { S } } } - z ^ { \mathrm { { T } } } ) \| _ { 2 } ^ { 2 } } { 4 \tau ^ { 2 } }\tag{17}
$$

$$
\mathrm { ~  ~ \Gamma ~ } = \frac { \| { \bf C } W ( h ^ { \mathrm { S } } - h ^ { \mathrm { T } } ) \| _ { 2 } ^ { 2 } } { 4 \tau ^ { 2 } } \leq \frac { d \| { \bf C } W \| _ { \mathrm { o p } } ^ { 2 } } { 4 \tau ^ { 2 } } e .\tag{18}
$$

Exchanging � and � proves the reverse-direction bound; the common bias cancels.

Applying the objective’s nonnegative averaging weights gives the corresponding average bound on supervised pairs. The result concerns finite raw logits and does not cover hard logit suppression or discrete selection directly. If the teacher’s unique top logit has margin $\gamma \ > \ 0$ , the sufficient condition $\| \mathsf { C } W \| _ { \mathrm { o p } } \| h ^ { \mathrm { S } } - h ^ { \mathrm { T } } \bar { \| } _ { 2 } < \gamma / 2$ also preserves its token argmax: every centered logit changes by less than half the margin. Neither statement guarantees position-selection agreement, because other competing positions may not be supervised and autonomous student states may differ.

## B.4 WHAT THE SMALLER-WINDOW REPRESENTATION CAN RETAIN

Fix a teacher phase and let � be the distribution over supervised position pairs induced by the rollout, filter, state sampling, and normalization in Eq. (12). Write $U = ( \mathbf { q } , \mathbf { y } _ { t , 1 : L _ { S } } , j )$ for the student-visible state and queried position, and $Z = h _ { t , j } ^ { \mathrm { T } }$ for the target; the student does not receive � as a separate input. For square-integrable � and any square-integrable predictor $f ( U )$

$$
\begin{array} { r } { \mathbb { E } _ { \mu } \| f ( U ) - Z \| _ { 2 } ^ { 2 } = \mathbb { E } _ { \mu } \| f ( U ) - \mathbb { E } _ { \mu } [ Z \mid U ] \| _ { 2 } ^ { 2 } } \\ { + \mathbb { E } _ { \mu } \| Z - \mathbb { E } _ { \mu } [ Z \mid U ] \| _ { 2 } ^ { 2 } . } \end{array}\tag{19}
$$

To see this, expand around $m ( U ) = \mathbb { E } _ { \mu }  { [ Z \mid U ] }$ ; the cross term has zero conditional expectation given �. Hence the unrestricted optimum is $f ^ { * } ( U ) = \mathbb { E } _ { \mu } \left[ Z \mid U \right]$ . The objective transfers the component of higher-budget representations predictable from the shared prompt and cropped state. It need not reconstruct every detail that depends on omitted response positions. A parameterized student may incur approximation and optimization error, and this regression identity does not guarantee reasoning quality on its autonomous rollouts.

## B.5 OPTIONAL OUTCOME-CONDITIONED REPULSION

These variants supplement the positive objective and are considered separately, with the other repulsion branch disabled. Repulsion is disabled in the reference configuration. Let $D _ { \mathrm { H } }$ be the higherbudget trajectory distance in Eq. (11), and let $g _ { \mathrm { H } }$ indicate final-answer correctness. A failed-highbudget variant uses

$$
\mathcal { L } _ { \mathrm { f a i l e d - H } } = \mathrm { A v g } _ { g _ { \mathrm { H } } = 1 } D _ { \mathrm { H } } - \lambda _ { \mathrm { H } } \mathrm { A v g } _ { g _ { \mathrm { H } } = 0 } \operatorname* { m i n } ( D _ { \mathrm { H } } , m _ { \mathrm { H } } ) .\tag{20}
$$

The two groups are normalized separately, and empty groups contribute zero. Because $- \operatorname* { m i n } ( D , m ) = [ m - D ] _ { + } - m$ , the negative term is gradient-equivalent to margin repulsion, which stops increasing separation beyond �<sub>H</sub>.

A separate unfiltered-positive variant draws an additional failed rollout with fewer updates:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { f a i l e d - L } } = \mathrm { A v g } _ { \mathcal { B } } \left[ D _ { \mathrm { H } } + \lambda _ { \mathrm { L } } ( 1 - g _ { \mathrm { L } } ) \left[ m _ { \mathrm { L } } - D _ { \mathrm { L } } \right] _ { + } \right] . } \end{array}\tag{21}
$$

Here $D _ { \mathrm { { L } } }$ uses that rollout’s own paired states; its penalty is averaged over the whole batch rather than only the failures. In the supplied implementation, this negative teacher retains the long window and reduces the step count, e.g., $( L _ { \mathrm { n e g } } , \bar { T } _ { \mathrm { n e g } } ) = ( 1 0 2 4 , 1 2 8 )$ , not (128, 128). Removing $1 - g _ { \mathrm { { L } } }$ disables failure-only gating. Applying the high-budget correctness gate instead gates both terms on the rank, producing a distinct recipe; the failed-high-budget branch and that gate use mutually exclusive code paths. Representation repulsion does not guarantee a lower probability for a specified erroneous trajectory.

## C IMPLEMENTATION AND EVALUATION DETAILS

## C.1 TRAINING CONFIGURATION AND DEPLOYMENT

The supplied GSM8K reference configuration is recorded in Table 2. The main experiments train separate checkpoints on the full GSM8K and original MATH training sets and evaluate them on GSM8K and MATH500, respectively. Code evaluation uses mathematics-trained checkpoints without further code-domain adaptation. The configuration below describes the documented HiLo-RD run; it is not a complete configuration record for every dataset and baseline.

Table 2: GSM8K reference training configuration. Teacher and Student distinguish the two budget views. The remaining settings document trajectory selection and adapter training for the fulltraining-set run. The 1,000-example successive-distillation study uses a separate training subset.
<table><tr><td>Component</td><td>Setting</td></tr><tr><td>Backbone</td><td>LLaDA-8B-Instruct</td></tr><tr><td>Teacher budget</td><td> $L _ { \mathrm { T } } = T _ { \mathrm { T } } = 1 0 2 4$ </td></tr><tr><td>Student training window</td><td> $L _ { \mathrm { S } } = 5 1 2$ </td></tr><tr><td>Teacher response-prefix cap</td><td> $c = L _ { S }$ </td></tr><tr><td>Retained states</td><td>Up to  $K = 8 ,$  including the initial state</td></tr><tr><td>Representation</td><td>Final RMSNorm, before vocabulary projection</td></tr><tr><td>Supervision positions</td><td>Post-update cumulative committed positions</td></tr><tr><td>Trajectory filter</td><td>Teacher final-answer correctness</td></tr><tr><td>Adapter</td><td>LoRA rank 16, scaling 32</td></tr><tr><td>Learning rate / warmup</td><td>10−⁵ / 50 optimizer steps</td></tr><tr><td>Evaluated checkpoint</td><td>Optimizer step 900</td></tr><tr><td>Training pool</td><td>GSM8K train, 7,473 examples</td></tr><tr><td>Checkpoint interval</td><td>100 optimizer steps</td></tr><tr><td>Training hardware</td><td>Eight NVIDIA A100 80GB GPUs</td></tr><tr><td>Auxiliary repulsion</td><td>Disabled</td></tr></table>

Only the student’s LoRA parameters are optimized; the teacher, backbone, and vocabulary head remain fixed. At deployment, the adapter can be merged into the backbone and the student decodes autonomously. The number of sampled training states does not determine its deployment step budget $T _ { \mathrm { { S } } }$

## C.2 COMPUTATION AND LATENCY

Equation (2) describes dense recomputation without classifier-free guidance. It does not assume that quadratic attention dominates at the studied lengths or exclude implementations with partial reuse. For variable-window decoding, the corresponding cost is $\textstyle \sum _ { t = 0 } ^ { R - 1 } F ( { \bar { P } } + L _ { t } )$

Measured latency also depends on hardware, kernels, batching, and prompt length. The reported Full/Fast timings include results from different servers and remain subject to normalization; their ratios are not controlled speedup estimates. Mean per-request latency is distinct from dataset throughput and total multi-GPU wall time. Teacher-acquisition savings require stopping the decoding loop, rather than retaining fewer states after a full rollout.

## C.3 READABLE LENGTH, EXTRACTION, AND PASS@�

The archived evaluation driver measures RTL by retokenizing the decoded response after removing special tokens. This can differ from the number of occupied response positions or the length extracted before the first EOT. RTL is neither the configured window size nor the realized number of forward passes: out-of-prefix commitments also consume updates. The prefix-preservation guarantee in Appendix B.2 therefore requires an explicit extraction convention when comparing Full and Fast outputs.

For a problem with � independently sampled completions and � passing all code tests, the standard estimator is

$$
{ \widehat { \operatorname { p a s s } @ k } } = 1 - { \frac { { \binom { n - c } { k } } } { \binom { n } { k } } } , \qquad n \geq k .\tag{22}
$$

The dataset score averages this quantity over problems. Its interpretation depends on the sample count, decoding distribution, and code-test protocol; a per-completion latency is not the total cost of obtaining a multi-sample pass@� score.

## D SUCCESSIVE DISTILLATION: PROCEDURE AND MEASUREMENTS

Each cycle merges the preceding adapter into the backbone before initializing the next frozen teacher and trainable student. If $\theta ^ { ( r - \widetilde { 1 } ) }$ is the merged model, cycle � uses $\bar { \theta } ^ { ( r ) } \ = \ \theta ^ { ( r - 1 ) }$ and a student initialized from the same weights with a new adapter $\Delta ^ { ( r ) }$ . The resulting merged model supplies the next cycle’s supervision. This extension changes the teacher between cycles; the teacher remains fixed within each cycle.

The measurements underlying Figure 7 are provided in Table 3. Every cycle reuses the same 1,000 GSM8K training examples selected with a fixed seed. C0 is the unadapted model, and C1–C3 denote successive cycles; this sequence is separate from the full-training-set main comparison.

Table 3: Successive distillation improves the low-budget quality–length trade-off. C0–C3 reuse the same 1,000 GSM8K training examples. Accuracy is in percent, RTL is mean readable tokens, and Time is seconds per request. C3 identifies the final cycle; Best marks the highest accuracy or shortest response within each budget. Bold indicates the best value in each column, including ties.
<table><tr><td rowspan="2">Cycle</td><td colspan="3">Low budget</td><td colspan="3">High budget</td></tr><tr><td>Acc.↑</td><td>RTL↓</td><td>Time↓</td><td>Acc.↑</td><td>RTL↓</td><td>Time↓</td></tr><tr><td>CO (vanilla)</td><td>68.84</td><td>362.8</td><td>73.35</td><td>79.61</td><td>278.4</td><td>66.05</td></tr><tr><td>C1</td><td>74.37</td><td>309.3</td><td>60.16</td><td>81.05</td><td>250.7</td><td>59.18</td></tr><tr><td>C2</td><td>76.19</td><td>275.5</td><td>53.90</td><td>81.05</td><td>235.7</td><td>55.63</td></tr><tr><td>C3</td><td>78.09</td><td>254.2</td><td>49.71</td><td>80.74</td><td>222.9</td><td>52.47</td></tr></table>

Low-budget accuracy improves monotonically from C0 to C3, accompanied by consistent reductions in readable length and latency. High-budget decoding also becomes substantially more efficient while retaining an accuracy advantage over C0. The improvement at both budgets allows each adapted model to generate more effective supervision for the following cycle, supporting successive cross-budget distillation as a practical mechanism for self-evolving quality–budget refinement.