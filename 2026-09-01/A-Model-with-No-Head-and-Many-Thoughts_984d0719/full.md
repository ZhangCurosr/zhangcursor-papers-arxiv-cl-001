# A Model with No Head and Many Thoughts

Nikita Koriagin<sup>1,\*</sup>, Yaroslav Aksenov<sup>2,†</sup>, George Bredis<sup>2</sup>, Gleb Gerasimov<sup>2</sup>, Nikita Balagansky<sup>2</sup>, Daniil Gavrilov<sup>2</sup>

<sup>1</sup>Yandex, <sup>2</sup>T-Tech

<sup>\*</sup>Work done while at T-Tech. <sup>†</sup>Correspondence: Yaroslav Aksenov, y.o.aksenov@t-tech.dev

## Abstract

Large language models decode by projecting hidden states through a large vocabulary head at every step. This operation is computationally costly and forces all reasoning to be expressed in discrete tokens. We introduce Soft Latent Thinking, a method that replaces the LM head during reasoning with a lightweight projector, enabling autoregressive rollout in embedding space where reasoning steps remain continuous rather than tokenized. Experiments on DeepSeek-Qwen-1.5B and LLaMA-3.2-3B show that Soft Latent Thinking consistently improves pass@k across all k while reducing per-step compute during chain-of-thought. Our method achieves the highest pass@32 among all soft-thinking approaches, demonstrating that effective reasoning can be carried out in continuous space without discrete token generation.

## 1 Introduction

Chain-of-thought (CoT) reasoning improves language model performance on complex problems by allocating extra computation to intermediate steps before producing a final answer. Soft thinking (Zhang et al., 2025) generalizes CoT by carrying out these intermediate steps in continuous embedding space rather than emitting discrete tokens, enabling richer state transitions and more flexible credit assignment. Recent methods (Butt et al., 2025; Zheng et al., 2025) further combine soft thinking with reinforcement learning (RL), yielding consistent gains over discrete-token RL on reasoning tasks.

A key limitation of existing soft-thinking approaches is that they still route every reasoning step through thefull vocabulary head. Concretely, at each step they form a V -way distribution, and use it to produce a weighted mixture of token embeddings. This design has two undesirable consequences: it (i) ties latent reasoning states to discrete token semantics (since the state must lie in the span of token embeddings), and (ii) keeps the dominant computational cost of the V -dimensional projection and normalization, which becomes a bottleneck for long reasoning traces.

We introduce Soft Latent Thinking (SLT), a soft-thinking variant that replaces the vocabulary projection only during reasoning with a compact latent projector. Instead of producing a distribution over V tokens, SLT maps a hidden state $h _ { t }$ to coefficients over a learned latent basis of size $K \ll V$ and directly synthesizes the next reasoning state in embedding space:

$$
h _ { t } \ \mapsto \ a _ { t } \in \mathbb { R } ^ { K } , \qquad z _ { t } = B a _ { t } \in \mathbb { R } ^ { d } ,
$$

where $B \in \mathbb { R } ^ { d \times K }$ is the latent basis and $z _ { t }$ is the continuous reasoning state fed to the next step. For stability, we initialize B from frequent domain token embeddings, but we allow both B and the projector to evolve under RL. For the final response, we keep standard token decoding unchanged; SLT only alters the internal reasoning operator. This replacement reduces per-step compute from a V -way projection to a K-dimensional mapping, and removes the requirement that intermediate reasoning states must be expressible as mixtures of discrete token embeddings.

To make SLT trainable under RL, we define a tractable per-step likelihood for the latent reasoning action via the sampled Gumbel variables used in the continuous selection over the K basis directions. This provides a well-defined policy for policy-gradient updates despite operating on continuous intermediate states. In practice, we find that training is reliable with lightweight adaptation: updating only the SLT projector (and basis) or optionally combining it with LoRA (Hu et al., 2021) on a set of backbone weights, without full model retraining.

![](images/c7afe1e7e2e83d31cd54bf9801ef409c084810b9a2ba3fc985e3aba9b171a7c9.jpg)  
(a)

![](images/3bc2cbaa417c8bbac36b160ff81d0a2e877a8685fba91145ce474129a5848789.jpg)

![](images/8530f574c95c560e92c3353ccf781cc06af02ebedebfeca8fcf2138857e6e444.jpg)  
(c)  
Figure 1: Comparison of decoding methods. (a) Standard decoding projects hidden states through the LM head to sample discrete tokens. (b) Soft thinking computes vocabulary probabilities via the LM head, then forms soft embeddings as weighted mixtures over all tokens. (c) Our method replaces the LM head with a lightweight projector (encoder-decoder) producing soft embeddings directly without full vocabulary projection.

We evaluate SLT on DeepSeek-R1-Distill-Qwen-1.5B (DeepSeek-AI, 2025) and LLaMA-3.2-3B-Instruct (Grattafiori et al., 2024) across five mathematical reasoning benchmarks. Compared to SofT-GRPO, SLT improves average pass@32 while using fewer reasoning tokens and a cheaper per-step reasoning operator. We also analyze the diversity– accuracy tradeoff induced by the latent operator: individual samples can be slightly less precise than full-vocabulary soft thinking, but the increased diversity across rollouts improves coverage at higher k, which primarily drives the pass@32 gains.

## 2 Related Work

Chain-of-thought (CoT) prompting improves language model performance on complex reasoning tasks by encouraging step-by-step intermediate computations before producing final answers (Wei et al., 2022; Kojima et al., 2022). These works showed that allocating multiple steps of internal reasoning often leads to better outcomes than direct prediction.

Recent work has explored reasoning in continuous embedding space rather than generating discrete tokens at each step. These methods allow models to maintain latent reasoning trajectories without committing to surface forms during intermediate steps. Coconut (Hao et al., 2024) introduces a recurrent hidden-state reasoning loop, where each hidden state is fed back as the next input, while Diffusion-of-Thought (Ye et al., 2024) refines reasoning through continuous denoising. A distinct class of approaches, known as soft thinking, constructs intermediate steps as weighted mixtures of token embeddings (Zhang et al., 2025). This allows smooth transitions across token semantics but can lead to degenerate deterministic behavior. Wu et al. address this by applying Gumbel-Softmax sampling to promote diversity in reasoning paths (Wu et al., 2025; Jang et al., 2017).

Recent work applies reinforcement learning with verifiable rewards (RLVR) to enhance CoT reasoning. Group Relative Policy Optimization (GRPO) (Shao et al., 2024) samples groups of trajectories per query and updates the policy to favor higherreward samples, yielding strong results on reasoning benchmarks. Initial attempts to combine soft thinking with GRPO underperform discrete-token counterparts (Butt et al., 2025), as soft tokens are deterministic given the logits, limiting exploration of alternative reasoning paths. SofT-GRPO (Zheng et al., 2025) addresses this through Gumbel reparameterization, introducing stochasticity while enabling policy gradients. We build on SofT-GRPO but introduce a dedicated projector that decouples soft token generation from the LM head, reducing per-step compute while improving sample diversity.

## 3 Background

## 3.1 Overview

Standard autoregressive decoding computes at each step:

$$
p ( x _ { t } | \boldsymbol { x } _ { < t } ) = \mathrm { s o f t m a x } ( \boldsymbol { W } _ { \mathrm { h e a d } } \cdot \boldsymbol { h } _ { t } ) ,
$$

where $h _ { t } \in \mathbb { R } ^ { d }$ is the hidden state and $W _ { \mathrm { h e a d } } ~ \in$ $\mathbb { R } ^ { V \times d }$ projects to the full vocabulary.

## 3.2 Soft Thinking Methods

Soft Thinking. Zhang et al. (2025) propose generating soft tokens as probability-weighted mixtures of embeddings. At each step:

$$
s _ { t } = \sum _ { i = 1 } ^ { V } p _ { t , i } \cdot e _ { i } ,
$$

where $e _ { i }$ is the embedding of token i, and $p _ { t } =$ softmax $( W _ { \mathrm { h e a d } } \cdot h _ { t } ) \in \mathbb { R } ^ { V }$ . The soft token $s _ { t }$ is fed as input to the next step. This enables continuous reasoning without training, but computes a full softmax over $V$ at each step.

Stochastic Soft Thinking. Wu et al. (2025) identify that vanilla soft thinking suffers from a greedy pitfall—models rely predominantly on the highest-probability token. They introduce Gumbel-Softmax sampling to encourage exploration of diverse reasoning paths. Given logprobabilities log $p _ { t , i } \in \mathbb { R } ^ { V }$ over the vocabulary, Gumbel-Softmax sampling computes:

$$
g _ { t , i } = \log p _ { t , i } + \epsilon _ { t , i } , \quad \epsilon _ { t , i } \sim \mathrm { G u m b e l } ( 0 , 1 ) ,\tag{1}
$$

$$
y _ { t , i } = \frac { \exp ( g _ { t , i } / \tau _ { g } ) } { \sum _ { j = 1 } ^ { V } \exp ( g _ { t , j } / \tau _ { g } ) } ,\tag{2}
$$

where $\tau _ { g }$ is a temperature parameter. The soft token is then $\begin{array} { r } { s _ { t } = \sum _ { i = 1 } ^ { V } y _ { t , i } \cdot e _ { i } } \end{array}$

SofT-GRPO. Zheng et al. (2025) adapt GRPOstyle RL to the soft-thinking setting, where each reasoning step outputs a continuous vector $s _ { t } \in \mathbb { R } ^ { d }$ instead of a discrete token. Two issues must be solved to make policy gradients workable: (i) plain soft thinking is effectively deterministic given the logits, so there is little exploration of alternative reasoning paths; and (ii) GRPO/PPO (Schulman et al., 2017) needs a per-step log-probability (or importance ratio), but a soft token $s _ { t }$ is not sampled from a categorical distribution.

To obtain stochastic but valid soft tokens, SofT-GRPO samples a Gumbel–Softmax mixture over the vocabulary and then maps it to an embedding. For a context $c _ { t } ~ = ~ [ Q , ( s _ { 1 } , \ldots , s _ { t - 1 } ) ]$ , rollout uses the behavior policy $\pi _ { \theta _ { \mathrm { o l d } } }$ to form token probabilities, injects i.i.d. Gumbel noise, and produces a soft token.

The central trick for gradient evaluation is that SofT-GRPO does not try to assign a probability density directly to $s _ { t } .$ Instead, it assigns likelihood to the underlying Gumbel variables (equivalently, to $g _ { t } )$ . Given a candidate policy $\pi _ { \theta }$ producing $p _ { t , i } = \pi _ { \theta } ( i \mid c _ { t } )$ , the stored $g _ { t , i }$ is distributed as a standard Gumbel shifted by $\log p _ { t , i }$ , which yields the per-step log-likelihood

$$
\begin{array} { c } { \displaystyle \log p _ { \theta } ( \boldsymbol { g } _ { t } \mid \boldsymbol { c } _ { t } ) = \sum _ { i = 1 } ^ { V } \Big [ - ( \boldsymbol { g } _ { t , i } - \log p _ { t , i } ) } \\ { \displaystyle - \exp \big ( - ( \boldsymbol { g } _ { t , i } - \log p _ { t , i } ) \big ) \Big ] . } \end{array}\tag{3}
$$

Crucially, during the update $g _ { t , i }$ is treated as fixed because it is part of the sampled trajectory produced in rollout (under $\theta _ { \mathrm { o l d } } )$ . Equivalently, the underlying noise $\epsilon _ { t , i } = g _ { t , i } - \log p _ { t , i } ^ { \mathrm { o l d } }$ is fixed. Therefore $g _ { t , i }$ does not depend on θ inside the gradient computation: the only θ-dependent quantity above is $p _ { t , i } = \pi _ { \theta } ( i \mid c _ { t } )$

Finally, SofT-GRPO plugs this likelihood into a GRPO/PPO-style importance ratio for the softthinking steps:

$$
\begin{array} { c } { { \log r _ { t } = \log p _ { \theta } ( g _ { t } \mid c _ { t } ) - \log p _ { \theta _ { \mathrm { o h } } } ( g _ { t } \mid c _ { t } ) } } \\ { { { } ~ } } \\ { { { } ~ = \displaystyle \sum _ { i = 1 } ^ { V } \Big [ - ( g _ { t , i } - \log p _ { t , i } ) } } \\ { { { } ~ } } \\ { { { } ~ - \exp \big ( - ( g _ { t , i } - \log p _ { t , i } ) \big ) \Big ] } } \\ { { { } ~ - \displaystyle \sum _ { i = 1 } ^ { V } \Big [ - \epsilon _ { t , i } - \exp ( - \epsilon _ { t , i } ) \Big ] . } } \end{array}\tag{4}
$$

The second sum is constant with respect to $\theta ,$ so optimizing the GRPO surrogate increases the likelihood of the stored Gumbel variables (and thus the stored soft-thinking trajectory) under $\pi _ { \theta }$ when its final answer receives higher reward. Answer tokens A are handled exactly as in standard GRPO (categorical log-probabilities and ratios), while only the soft-thinking prefix uses the Gumbel-based ratio above; clipping and a KL penalty to a reference policy are then applied in the usual way to stabilize updates.

## 4 Soft Latent Thinking

Our method, Soft Latent Thinking, replaces the expensive vocabulary projection during reasoning with a dedicated latent projector that maps the model hidden state directly to a soft embedding. Unlike SofT-GRPO (Zheng et al., 2025), which forms soft tokens as mixtures over the full vocabulary embeddings, our projector learns an independent, lower-cardinality latent space of size $K \ll V$ and decodes it back to $\mathbb { R } ^ { d }$ . This decoupling allows soft tokens to carry information that is not constrained to discrete token semantics, while reducing per-step compute by operating over $K \approx \mathrm { 1 2 k - 2 4 k }$ instead of V ≈ 150k.

Table 1: Performance comparison on five mathematical reasoning benchmarks. We evaluate two base models under discrete-token chain-of-thought and soft-thinking reasoning patterns. Metrics @1 denote Mean@32 (average Pass@1 across 32 runs), while @16 and @32 denote Pass@16 and Pass@32 respectively. Best result per metric/dataset is underlined, best average is bolded, second-best average is shaded. All values scaled by 100.
<table><tr><td rowspan="2">Dataset Metrics</td><td colspan="3">AIME2024</td><td colspan="3">AIME2025</td><td colspan="3">AMC23</td><td colspan="3">MATH-500</td><td colspan="3">GSM8K</td><td colspan="3">Average</td></tr><tr><td>@1</td><td>@16</td><td>@32</td><td>@1</td><td>@16</td><td>@32</td><td>@1</td><td>@16</td><td>@32</td><td>@1</td><td>@16</td><td>@32</td><td>@1</td><td>@16</td><td>@32</td><td>@1 @16</td><td>@32</td></tr><tr><td colspan="14">DeepSeek-R1-Distill-Qwen-1.5B Base LLM</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>Discrete-Token CoT Reasoning Pattern</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>No-Finetune</td><td>30.6</td><td>70.0</td><td>73.3</td><td>23.0</td><td>46.7</td><td>53.3 70.7</td><td>92.5</td><td>95.0</td><td>84.6</td><td>97.8</td><td>97.8</td><td>81.5</td><td>95.8</td><td>96.7</td><td>58.09</td><td>80.54</td><td>83.23</td></tr><tr><td>+ GRPO</td><td>31.8</td><td>66.7</td><td>76.7</td><td>25.3</td><td>46.7</td><td>46.7</td><td>77.3 95.0</td><td>95.0</td><td>87.1</td><td>97.4</td><td>97.8</td><td>84.9</td><td>95.1</td><td>95.8</td><td>61.28</td><td>80.16</td><td>82.39</td></tr><tr><td colspan="14">Soft-Thinking Reasoning Pattern</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>No-Finetune</td><td>27.3</td><td>66.7</td><td>70.0</td><td>23.8</td><td>46.7</td><td>53.3</td><td>69.9</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>82.41</td></tr><tr><td>+ GRPO</td><td>29.2</td><td>70.0</td><td>73.3</td><td>25.4</td><td>46.7</td><td>53.3 75.8</td><td>95.0 95.0</td><td>95.0 95.0</td><td>79.4 86.3</td><td>93.2 96.8</td><td>96.6 98.2</td><td>81.0 84.9</td><td>94.6 95.6</td><td>97.1 96.4</td><td>56.28 60.31</td><td>79.23 80.81</td><td>83.26</td></tr><tr><td>+ SofT-GRPO</td><td>32.6</td><td>76.7</td><td>80.0</td><td>26.1</td><td>50.0</td><td>53.3 76.4</td><td>97.5</td><td>97.5</td><td>86.3</td><td>97.4</td><td>98.0</td><td>85.5</td><td>96.1</td><td>97.0</td><td>61.39</td><td>83.54</td><td>85.18</td></tr><tr><td>+ Ours</td><td>28.7</td><td>74.3</td><td>80.0</td><td>20.5</td><td>48.3</td><td>56.0 72.3</td><td>97.5</td><td>100.0</td><td>83.9</td><td>96.7</td><td>97.6</td><td>81.2</td><td>96.5</td><td>97.5</td><td>57.32</td><td>82.66</td><td>86.22</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>LLaMA-3.2-3B-Instruct Base LLM</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="14"></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>No-Finetune</td><td></td><td></td><td></td><td></td><td></td><td></td><td>Discrete-Token CoT Reasoning Pattern</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>+ GRPO</td><td>4.4 7.3</td><td>20.0 23.3</td><td>26.7</td><td>0.3</td><td>0.3</td><td>1.0</td><td>18.3 65.0</td><td>75.0</td><td>38.1</td><td>75.6</td><td></td><td>84.0 67.9</td><td>92.1</td><td>94.6</td><td>25.79</td><td>50.61</td><td>56.26</td></tr><tr><td></td><td></td><td></td><td>26.7</td><td>0.5</td><td>3.3</td><td>3.3</td><td>27.3 62.5</td><td>67.5</td><td>48.3</td><td>77.2</td><td>82.6</td><td>79.6</td><td>95.4</td><td>96.5</td><td>32.60</td><td>52.35</td><td>55.32</td></tr><tr><td>Soft-Thinking Reasoning Pattern</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>No-Finetune + GRPO</td><td>3.4</td><td>16.7 20.0</td><td>16.7</td><td>0.2</td><td>6.7</td><td>6.7</td><td>17.6 70.0</td><td>77.5</td><td>36.7</td><td>76.0</td><td>81.4</td><td>66.9</td><td>91.6</td><td>94.7</td><td>24.96</td><td>52.18</td><td>55.39</td></tr><tr><td>+ SofT-GRPO</td><td>8.0 7.7</td><td></td><td>23.3</td><td>0.7</td><td>3.3</td><td>10.0</td><td>27.3 70.0</td><td>75.0</td><td>47.8</td><td>76.8</td><td>81.8</td><td>79.2</td><td>94.8</td><td>96.3</td><td>32.60</td><td>53.00</td><td>57.28</td></tr><tr><td>+ Ours</td><td>7.1</td><td>23.3 28.1</td><td>26.7 36.7</td><td>0.3 0.3</td><td>10.0 5.0</td><td>10.0 10.0</td><td>31.3 67.5 19.1 64.2</td><td>67.5 75.0</td><td>47.2 39.6</td><td>77.6 78.6</td><td>83.4 84.6</td><td>77.6 66.8</td><td>96.4 95.5</td><td>97.7 97.2</td><td>32.83 26.58</td><td>54.96 54.28</td><td>57.06 60.70</td></tr></table>

At soft-thinking step $t ,$ let $h _ { t } \in \mathbb { R } ^ { d }$ denote the last-layer hidden state (before the LM head) produced by the base model given the current context. The projector produces a soft embedding $s _ { t } \in \mathbb { R } ^ { d }$ via a compressed softmax and Gumbel–Softmax sampling:

Encoder. The encoder is a single linear map that compresses the hidden state into K logits:

$$
z _ { t } = W _ { \mathrm { e n c } } h _ { t } .
$$

Sampling. We first compute probabilities from the encoder logits:

$$
p _ { t , i } = \frac { \exp ( z _ { t , i } ) } { \sum _ { j = 1 } ^ { K } \exp ( z _ { t , j } ) } .\tag{5}
$$

We then apply Gumbel-Softmax (Eqs. 1–2) over K categories, yielding mixture weights $\boldsymbol { y } _ { t } \in \mathbb { R } ^ { K }$

Decoder. The decoder maps the latent mixture y<sub>t</sub> back to the model embedding space:

$$
s _ { t } = W _ { \mathrm { d e c } } ^ { \top } y _ { t } .\tag{6}
$$

Algorithm 1 Soft Latent Thinking training step.   
Require: prompt Q, policies $\theta _ { \mathrm { o l d } } , \theta ,$ projectors   
$\mathbf { \bar { \rho } } ( W _ { \mathrm { e n c } } ^ { o l d } , W _ { \mathrm { d e c } } ^ { o l \mathrm { \hat { d } } } )$ and (W<sub>enc</sub>, W<sub>dec</sub>), temperature   
$\tau _ { g }$   
Rollout   
1: $c _ { 1 }  Q , \quad B  \emptyset$   
2: for $t = 1 , \dots , T _ { \mathrm { m a x } }$ do   
3: $h _ { t } \gets f _ { \theta _ { \mathrm { o l d } } } ( c _ { t } )$   
4: p<sub>t</sub> ← softmax $( W _ { \mathrm { e n c } } ^ { o l d } h _ { t } )$   
5: $\epsilon _ { t } \sim \mathrm { G u m b e l } ( 0 , 1 ) ^ { K } , \quad g _ { t }  \log p _ { t } + \epsilon _ { t }$   
6: y<sub>t</sub> ← softmax $( g _ { t } / \tau _ { g } )$   
7: $s _ { t } \gets ( W _ { \mathrm { d e c } } ^ { o l d } ) ^ { \top } y _ { t }$   
8: $\begin{array} { r } { \mathcal { B }  \dot { B } \cup \{ \big ( c _ { t } , g _ { t } , \epsilon _ { t } , s _ { t } \big ) \} , \quad c _ { t + 1 }  [ c _ { t } ; s _ { t } ] } \end{array}$   
9: if stop(s<sub>t</sub>) then ▷ aligned with </think> or \boxed   
10: break   
11: end if   
12: end for   
13: Decode answer with LM head; receive reward R   
Update   
14: for $( c _ { t } , g _ { t } , \epsilon _ { t } , s _ { t } ) \in B$ do   
15: $h _ { t } ^ { \theta }  f _ { \theta } ( c _ { t } )$   
16: $p _ { t } ^ { \theta } \gets$ softmax $\mathrm { W _ { e n c } } h _ { t } ^ { \theta } )$   
17: log r<sub>t</sub> ← log p<sub>θ</sub>(g<sub>t</sub> | c<sub>t</sub>) − log p<sub>θ</sub> (g<sub>t</sub> | c<sub>t</sub>) ▷ Eq. 3   
18: end for   
19: Optimize clipped GRPO objective using {log r<sub>t</sub>}, answer  
token ratios, and R

## 4.1 Initialization

Let $\boldsymbol { \mathcal { K } } = \{ k _ { 1 } , \ldots , k _ { K } \}$ be the indices of the K most frequent tokens in the target domain (or another chosen token subset). We initialize the projector from the pretrained LM head $W _ { \mathrm { h e a d } } \in \mathbb { R } ^ { V \times d }$ and the input embedding table $E \in \mathbb { R } ^ { V \times d }$ by copying the corresponding rows:

![](images/803d816dd5c1700162196826f9e5ea666ad9d6caa175787e8d28a72698ffc34a.jpg)  
Figure 2: Cumulative distribution of token frequencies on mathematical reasoning data. A small subset of tokens covers most occurrences: the top 5000 tokens account for almost 99% of the distribution.

$$
\begin{array} { r l } & { W _ { \mathrm { e n c } } [ i , : ]  W _ { \mathrm { h e a d } } [ k _ { i } , : ] , } \\ & { W _ { \mathrm { d e c } } [ i , : ]  E [ k _ { i } , : ] , \quad \quad i \in \{ 1 , \dots , K \} . } \end{array}\tag{7}
$$

(8)

After initialization, $W _ { \mathrm { e n c } }$ and $W _ { \mathrm { d e c } }$ are trained and are not tied to $W _ { \mathrm { h e a d } }$ or E.

The token-frequency distribution used for initialization is shown in Figure 2; it is computed on mathematical reasoning data (the math subset of OpenThoughts-114k (Guha et al., 2025), containing DeepSeek-generated (DeepSeek-AI, 2025) reasoning traces). The distribution is highly concentrated, but although the top 5k tokens cover over 99% of occurrences, our ablations (Table 7) show that K ≈ 12k yields optimal performance, suggesting the model benefits from access to rarer tokens during reasoning.

## 4.2 Training

We train with the SofT-GRPO objective (Zheng et al., 2025). During rollout, we sample softthinking trajectories using the policy parameters $\theta _ { \mathrm { o l d } }$ . At each soft-thinking step t, we compute encoder logits $z _ { t } ^ { o l d } = W _ { e n c } ( \theta _ { o l d } ) h _ { t }$ , apply Gumbel-Softmax sampling, and store the perturbed log probs $g _ { t }$ together with the decoder produced soft embedding $s _ { t }$ (which is used as the next-step input). During the policy update, we replay the stored soft embeddings $\left( s _ { 1 } , \ldots , s _ { t - 1 } \right)$ to form the same context and recompute the projector probabilities under the current parameters θ:

$$
\begin{array} { r } { { \mathbf \xi } _ { z _ { t } } = W _ { \mathrm { e n c } } ( \theta ) h _ { t } , \quad } \\ { p _ { t , i } = \displaystyle \frac { \exp ( z _ { t , i } ) } { \sum _ { j = 1 } ^ { K } \exp ( z _ { t , j } ) } . } \end{array}
$$

Following SofT-GRPO, we evaluate the likelihood of the stored Gumbel-perturbed log probs $g _ { t } ~ =$ $( g _ { t , 1 } , \ldots , g _ { t , K } )$ under the current policy and evaluate the Gumbel likelihood (Eq. 3 with V replaced by K). In this computation, $g _ { t , i }$ is treated as fixed because it is part of the sampled trajectory produced during rollout (under $\theta _ { \mathrm { o l d } } ) ;$ only $p _ { t , i }$ depends on θ. The corresponding per-step importance ratio used by the GRPO surrogate is

$$
\log r _ { t } = \log p _ { \theta } ( g _ { t } \mid c _ { t } ) - \log p _ { \theta _ { \mathrm { o l d } } } ( g _ { t } \mid c _ { t } ) ,\tag{9}
$$

where the second term is constant w.r.t. θ for a fixed rollout sample. For the final answer tokens, we use the standard categorical log-probability ratios from the LM head, as in discrete-token GRPO.

## 4.3 Inference

Generation begins with a <think> token when available. At each reasoning step, the projector produces a soft embedding $s _ { t }$ , which is fed to the next model step. Reasoning terminates when the soft embedding is sufficiently aligned with the </think> token embedding:

$$
\cos ( s _ { t } , e _ { < / \mathrm { t h i n k } > } ) > \delta .\tag{10}
$$

The model then switches to standard autoregressive decoding to generate the final answer. For models without explicit thinking boundary tokens (e.g., some LLaMA-style checkpoints), we use an alternative stopping criterion. Reasoning terminates when the soft embedding is sufficiently close to the \boxed token embedding:

$$
\cos ( s _ { t } , e _ { \setminus \mathrm { b o x e d } } ) > \gamma .\tag{11}
$$

Stopping-threshold sensitivity. The stopping rule uses a cosine-similarity threshold to decide when the latent reasoning phase should hand control back to ordinary token decoding. In preliminary sensitivity checks, we observed that the cosine similarity to the boundary embedding stays low during intermediate reasoning, typically below 0.2, and rises sharply near the point where the model begins to produce the final answer. This makes the stopping rule relatively insensitive to moderate changes in δ or γ, although a full sweep over thresholds remains an important robustness check.

## 5 Experiments

We implement Soft Latent Thinking on two base models: DeepSeek-R1-Distill-Qwen-1.5B (DeepSeek-AI, 2025) and LLaMA-3.2-3B-Instruct (Grattafiori et al., 2024).

Table 2: Token efficiency on mathematical reasoning benchmarks . #Token represents total tokens generated across all queries, while #Token represents tokens generated for correctly solved queries only.
<table><tr><td rowspan="2">Dataset Metrics</td><td colspan="2">AIME2024</td><td colspan="2">AIME2025 #Token</td><td colspan="2">AMC23</td><td colspan="2">MATH-500</td><td colspan="2">GSM8K</td><td colspan="2">Average</td></tr><tr><td>#Token</td><td></td><td>#Token_c</td><td>#Token_c</td><td>#Token</td><td>#Token_c</td><td>#Token</td><td>#Token_c</td><td>#Token</td><td>#Token_c</td><td>#Token</td><td>#Token_c</td></tr><tr><td colspan="10">DeepSeek-R1-Distill-Qwen-1.5B Base LLM</td></tr><tr><td></td><td></td><td></td><td></td><td>Discrete-Token CoT Reasoning Pattern</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>No-Finetune + GRPO</td><td>16241.6 8927.6</td><td>14997.8 8535.6</td><td>16416.3 8039.4</td><td>13448.8 6414.2</td><td>10052.2 5001.3</td><td>9394.2 4672.8</td><td>5616.6 3106.1</td><td>5368.1 2958.5</td><td>1839.3 1417.1</td><td>1772.5 1370.6</td><td>10033.2 5298.3</td><td>8996.3 4790.3</td></tr><tr><td colspan="10">Soft-Thinking Reasoning Pattern</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>No-Finetune</td><td>17857.8</td><td>16191.3</td><td>17569.4</td><td>14582.4</td><td>11269.9</td><td>10482.0</td><td>4015.9</td><td>3888.3</td><td>1699.3</td><td>1649.9</td><td>10482.5</td><td>9358.8</td></tr><tr><td>+ GRPO</td><td>9383.2</td><td>8934.5</td><td>8007.2</td><td>7325.8</td><td>5203.7</td><td>4894.4</td><td>3233.6</td><td>3131.7</td><td>1385.5</td><td>1349.9</td><td>5442.7</td><td>5127.2</td></tr><tr><td>+ SofT-GRPO</td><td>11039.6</td><td>10756.1</td><td>10519.6</td><td>7831.3</td><td>5900.2</td><td>5630.4</td><td>3549.5</td><td>3399.2</td><td>1577.6</td><td>1542.5</td><td>6517.3</td><td>5831.9</td></tr><tr><td>+ Ours</td><td>10280.8</td><td>7006.2</td><td>9805.2</td><td>7630.9</td><td>5252.6</td><td>4423.4</td><td>3379.5</td><td>2815.0</td><td>1646.9</td><td>1342.6</td><td>6073.0</td><td>4643.6</td></tr></table>

Projector. The projector consists of an encoder $W _ { \mathrm { e n c } } \in \mathbb { R } ^ { K \times d }$ and decoder $W _ { \mathrm { d e c } } \in \mathbb { R } ^ { K \times d } .$ . We set $K = 8 d ,$ giving $K = 1 2 2 8 8$ for Qwen (d = 1536) and $K = 2 4 5 7 6$ for LLaMA (d = 3072).

Training. We train on the DeepScaleR dataset (Luo et al., 2025) using Soft-GRPO with outcomebased rewards. For both models we freeze the backbone and train LoRA adapters (rank 64) on all attention and MLP modules jointly with the projector.

Baselines. We compare against: (1) the base model without fine-tuning, (2) the base model trained with standard GRPO, (3) Soft-Thinking without fine-tuning (Zhang et al., 2025), (4) Soft-Thinking with GRPO, and (5) SofT-GRPO (Zheng et al., 2025).

## 5.1 Main Results

The main results of Soft Latent Thinking are shown in Table 1. We evaluate on five mathematical reasoning benchmarks: AIME2024, AIME2025 (Art of Problem Solving, 2025), AMC23 (AI-MO, 2024), MATH-500 (Hendrycks et al., 2021), and GSM8K (Cobbe et al., 2021).

Soft Latent Thinking primarily improves the multi-sample accuracy–efficiency tradeoff rather than uniformly dominating at every sampling budget. On DeepSeek-R1-Distill-Qwen-1.5B, we achieve 86.22 average pass@32 compared to 83.23 for the base model and 85.18 for SofT-GRPO. On LLaMA-3.2-3B-Instruct, we achieve 60.70 average pass@32 compared to 56.26 for the base model and 57.06 for SofT-GRPO.

The gains are most pronounced at higher k, suggesting that decoupling soft token generation from the LM head—combined with Gumbel-Softmax sampling—encourages more diverse reasoning paths across rollouts. This also clarifies the deployment regime: SLT is most useful when multiple rollouts are affordable or already required, such as RL training and pass@k inference.

Table 3: Preliminary AIME2024 larger-model sanity check on Qwen3.5-9B. Results are from one untuned configuration and should be interpreted as evidence of transfer stability rather than as a definitive scaling study.
<table><tr><td>Model</td><td>Avg. tokens</td><td>Correct tokens</td><td>Pass@1</td><td>Pass@16</td><td>Pass@32</td></tr><tr><td>Qwen3.5-9B + projector</td><td>9288.1</td><td>7378.2</td><td>68.4</td><td>92.8</td><td>93.3</td></tr><tr><td>Base Qwen3.5-9B</td><td>23292.1</td><td>21778.6</td><td>76.4</td><td>93.2</td><td>93.3</td></tr></table>

## 5.2 Preliminary Larger-Model Check

To probe whether the behavior transfers beyond the 1.5B–3B setting, we additionally ran a single untuned experiment on Qwen3.5-9B (Qwen Team, 2026). This check is not intended as a full-scale evaluation, because we did not perform scale-specific hyperparameter tuning or a complete baseline sweep. Nevertheless, as Table 3 shows, it reproduces the qualitative pattern observed in smaller models: the projector model is weaker at low k, but reaches the same pass@32 while using substantially fewer reasoning tokens.

## 5.3 Out-of-Domain Evaluation

We evaluate out-of-domain behavior on GPQA Diamond (science) (Rein et al., 2024) and HumanEval (code) (Chen et al., 2021). The corresponding pass@k curves are shown in Figure 3, and the full numerical results are reported in Table 9. We evaluate three settings: base model, LoRA with projector enabled, and LoRA with standard soft thinking (full vocabulary, no projector).

On GPQA, the projector transfers well, achieving accuracy comparable to baselines. We attribute this to vocabulary overlap between mathematical and scientific reasoning (shared use of numbers, symbols, and formal notation).

![](images/bce13ffbfa72b63ba0c65e147ae13446eb5f2a10ab5b9c35f025e0d1e427679d.jpg)

![](images/349cdd12e3912344c89930ea40215405fdbe92e2f7828916a52ab5274ccbbc42.jpg)  
Figure 3: Out-of-domain evaluation on GPQA Diamond (science reasoning) and HumanEval (code generation) using DeepSeek-R1-Distill-Qwen-1.5B trained on mathematical reasoning tasks. SLT<sup>†</sup> indicates the projector is disabled at inference time, reverting to standard soft thinking with full-vocabulary decoding.

On HumanEval, the math-initialized projector degrades performance, as code requires different vocabulary (keywords, syntax, identifiers). However, disabling the projector and using standard soft thinking with the jointly trained LoRA achieves performance comparable to the strongest baselines. This demonstrates that LoRA adapters trained jointly with the projector remain effective for outof-domain tasks when the projector is bypassed. Mechanistically, the degradation is expected: code generation relies on Python keywords, identifiers, syntax tokens, and indentation markers, whereas the projector is initialized from tokens frequent in mathematical reasoning.

Crucially, when the projector is disabled, performance matches or exceeds SofT-GRPO on outof-domain tasks (97.0 vs 95.5 on GPQA, 92.7 vs 94.5 on HumanEval). This enables a practical deployment strategy: use the projector for in-domain tasks to gain efficiency and pass@k improvements, and disable it for out-of-domain tasks with no performance penalty relative to the strongest baseline.

The full numerical results are reported in Table 9.

## 5.4 Token Efficiency

Table 2 presents token usage across benchmarks. Soft Latent Thinking uses fewer tokens on average compared to SofT-GRPO, while achieving higher pass@k at higher k. This suggests that reasoning in continuous embedding space allows for more compact reasoning chains without sacrificing accuracy. The reduced token count, combined with the lower per-step FLOPs from bypassing the full vocabulary projection, results in overall computational savings during chain-of-thought generation.

## 5.5 Computational Efficiency

The primary advantage of Soft Latent Thinking is reduced compute during chain-of-thought generation. At each reasoning step, the standard approach computes:

$$
\mathrm { L M h e a d F L O P s } = 2 \times d \times V
$$

Our projector instead computes:

Projector $\mathrm { F L O P s } = 2 \times d \times K + 2 \times K \times d = 4 d K$ For $d = 1 5 3 6 , V = 1 5 0 \mathbf { k }$ , and $K \ : = \ : 1 6 \mathbf { k }$ , this yields a reduction of $\begin{array} { r } { \frac { 2 d V } { 4 d K } = \frac { V } { 2 K } \approx 5 \times } \end{array}$ on the vocabulary projection step.

We also measured prototype serving throughput in a mini-SGLang (Zheng et al., 2024) implementation. To be conservative, Table 4 reports projector scale 8, the largest and least favorable projector among the tested settings. Even in this setting, throughput remains above the vanilla baseline across the tested models and batch sizes. We separate these end-to-end serving numbers from graph-level decode speed: the latter shows an approximately 1.05× decode speedup across tested models and is the cleaner measurement of the algorithmic gain from replacing the LM-head reasoning step.

Table 4: Prototype serving throughput with projector scale 8. TPS denotes generated tokens per second.
<table><tr><td>Model</td><td>Batch</td><td>Vanilla TPS</td><td>Ours TPS</td><td>Speedup</td></tr><tr><td>Llama-3.2-1B</td><td>1</td><td>675.4</td><td>723.3</td><td>1.071×</td></tr><tr><td>Llama-3.2-1B</td><td>4</td><td>2637.1</td><td>2823.1</td><td>1.071×</td></tr><tr><td>Llama-3.2-1B</td><td>8</td><td>5038.2</td><td>5349.2</td><td>1.062×</td></tr><tr><td>Llama-3.2-3B</td><td>1</td><td>299.3</td><td>314.7</td><td>1.051×</td></tr><tr><td>Llama-3.2-3B</td><td>4</td><td>1176.7</td><td>1236.2</td><td>1.051×</td></tr><tr><td>Llama-3.2-3B</td><td>8</td><td>2317.8</td><td>2436.6</td><td>1.051×</td></tr><tr><td>Llama-3.1-8B</td><td>1</td><td>160.2</td><td>165.3</td><td>1.032×</td></tr><tr><td>Llama-3.1-8B</td><td>4</td><td>519.9</td><td>652.9</td><td>1.256×</td></tr><tr><td>Llama-3.1-8B</td><td>8</td><td>1113.7</td><td>1266.1</td><td>1.137×</td></tr></table>

## 5.6 Ablations

All ablation studies are conducted on AIME2024 using DeepSeek-R1-Distill-Qwen-1.5B unless otherwise specified.

Effect of projector training. We ablate the projector by replacing trained weights with initial ones (Eq. 7) while keeping LoRA fixed. As shown in Table 5, this degrades performance across all pass@k and increases response length, indicating the projector learns more efficient reasoning paths despite minimal weight change.

We also compare our projector against standard soft thinking using the same LoRA weights (Table 5). Full vocabulary soft thinking achieves higher pass@1 but lower pass@16, with similar pass@32. This suggests the compressed vocabulary introduces beneficial stochasticity: while individual samples may be less precise, diversity across samples improves, yielding better coverage at higher k.

Table 5: Projector variant ablation with fixed LoRA weights. Full ST uses standard soft thinking over full vocabulary V.
<table><tr><td>Setting</td><td>@1</td><td>@16</td><td>@32</td><td>#Token</td></tr><tr><td>Ours (trained projector)</td><td>28.7</td><td>74.3</td><td>80.0</td><td>10280</td></tr><tr><td>Init projector</td><td>28.3</td><td>70.1</td><td>76.7</td><td>11383</td></tr><tr><td>Full ST</td><td>29.5</td><td>72.3</td><td>80.0</td><td>11242</td></tr></table>

Vocabulary compression at inference. We evaluate vocabulary compression without training by using an initialized projector (pruned LM head and embedding table to K tokens) at inference time. Table 6 compares this against the full vocabulary baseline.

Without fine-tuning, the pruned projector degrades performance compared to full vocabulary soft thinking, indicating that vocabulary compression alone hurts. When applied to the SofT-GRPO checkpoint, the pruned projector shows the same degradation. This demonstrates that vocabulary compression at inference alone is insufficient—the model must be trained with the compressed vocabulary to effectively reason through the bottleneck.

Table 6: Vocabulary compression at inference without training.
<table><tr><td>Setting</td><td>@1</td><td>@16</td><td>@32</td><td>#Token</td></tr><tr><td colspan="5">No fine-tuning</td></tr><tr><td>Full ST</td><td>27.3</td><td>66.7</td><td>70.0</td><td>17858 6370</td></tr><tr><td>Init projector</td><td>11.3</td><td>56.3</td><td>63.3</td><td></td></tr><tr><td colspan="5">SofT-GRPO checkpoint</td></tr><tr><td>Full ST</td><td>32.6</td><td>76.7</td><td>80.0</td><td>11039</td></tr><tr><td>Init projector</td><td>15.8</td><td>60.6</td><td>70.0</td><td>6408</td></tr></table>

Projector size. Table 7 ablates the projector size K. Small K ≈ 1.5k underperforms due to limited capacity. K ≈ 6k achieves the best pass@1, while K ≈ 12k achieves the best pass@16 and pass@32. This suggests a trade-off: smaller projectors produce more focused selections benefiting single attempts, while larger projectors enable greater diversity for multi-sample evaluation.

Table 7: Ablation on projector size K.
<table><tr><td>K</td><td>@1</td><td>@16</td><td>@32</td><td>#Token</td></tr><tr><td>1*1536</td><td>20.4</td><td>58.2</td><td>66.7</td><td>18694</td></tr><tr><td>4*1536</td><td>29.7</td><td>62.0</td><td>66.7</td><td>14881</td></tr><tr><td>8*1536</td><td>28.7</td><td>74.3</td><td>80.0</td><td>10280</td></tr></table>

Projector and backbone training. Table 8 ablates the training configuration. Training only the projector with a frozen backbone yields weaker performance than the base model, as the backbone cannot adapt to process soft embeddings. Joint training of projector and LoRA achieves the best results, allowing both components to co-adapt. We do not evaluate freezing the projector while training only LoRA, as this reduces to SofT-GRPO constrained to K tokens—strictly less expressive than full vocabulary SofT-GRPO.

Table 8: Projector and backbone training ablation.
<table><tr><td>Projector</td><td>Backbone</td><td>@1</td><td>@16</td><td>@32</td><td>#Token</td></tr><tr><td>Trained</td><td>LoRA</td><td>28.7</td><td>74.3</td><td>80.0</td><td>10280</td></tr><tr><td>Trained</td><td>Frozen</td><td>26.0</td><td>65.4</td><td>70.0</td><td>12256</td></tr><tr><td>Init</td><td>Frozen</td><td>11.3</td><td>56.3</td><td>63.3</td><td>6370</td></tr></table>

Temperature. We explored the Gumbel-Softmax temperature $\tau _ { g }$ during training and inference. For training, $\tau _ { g } = 0 . 1$ worked best for LLaMA-3.2-3B-Instruct while $\tau _ { g } = 0 . 5$ worked best for DeepSeek-R1-Distill-Qwen-1.5B, suggesting that the optimal training temperature may depend on the base model. For inference, both models performed best with $\tau _ { g } = 0 . 5$ , indicating that moderate stochasticity during generation encourages exploration of diverse reasoning paths.

Table 9: Out-of-domain evaluation on GPQA Diamond and HumanEval. † indicates projector disabled, reverting to full-vocabulary soft thinking.
<table><tr><td>Dataset Metrics</td><td colspan="3">GPQA Diamond</td><td colspan="3">HumanEval</td></tr><tr><td></td><td>@1</td><td>@8</td><td>@32</td><td>@1</td><td>@8</td><td>@32</td></tr><tr><td colspan="7">DeepSeek-R1-Distill-Qwen-1.5B</td></tr><tr><td colspan="7">Discrete-Token CoT Reasoning</td></tr><tr><td>No-Finetune</td><td>36.7</td><td>84.3</td><td>96.0</td><td>68.1</td><td>87.2</td><td>93.9</td></tr><tr><td>+ GRPO</td><td>35.4</td><td>77.8</td><td>93.4</td><td>72.2</td><td>90.9</td><td>94.5</td></tr><tr><td colspan="7">Soft-Thinking Reasoning</td></tr><tr><td>No-Finetune</td><td>36.0</td><td>83.8</td><td>97.0</td><td>67.2</td><td>89.6</td><td>92.7</td></tr><tr><td>+ GRPO</td><td>36.5</td><td>81.3</td><td>94.4</td><td>71.8</td><td>89.0</td><td>95.1</td></tr><tr><td>+ SofT-GRPO</td><td>37.3</td><td>82.8</td><td>95.5</td><td>71.2</td><td>88.4</td><td>94.5</td></tr><tr><td>+ Ours</td><td>34.2</td><td>82.7</td><td>95.5</td><td>54.0</td><td>77.9</td><td>87.2</td></tr><tr><td>+ Ours†</td><td>35.4</td><td>83.0</td><td>97.0</td><td>70.0</td><td>88.0</td><td>92.7</td></tr></table>

## 6 Discussion

Training efficiency. Replacing the full vocabulary projection (V ≈ 150k) with a compressed projector $( K \approx 1 2 – 2 4 \mathbf { k } )$ reduces this operation by ∼5–10× per reasoning step. End-to-end gains are smaller because attention and MLP layers remain, but the savings compound over long traces and many GRPO rollouts.

Decoupled embeddings. Standard soft thinking uses the embedding table both for latent reasoning and discrete output, which may constrain expressiveness. Our projector decouples these roles: its decoder learns embeddings optimized for latent reasoning rather than token identity.

LoRA as decoding adapter. LoRA adapts the frozen backbone to process projector-generated soft embeddings instead of discrete token embeddings. This helps explain why out-of-domain knowledge is preserved when the projector is disabled.

Modularity. Our approach separates reasoning capacity (projector) from domain adaptation (LoRA). This enables a plug-and-play setup: different projectors can be trained for different domains (math, code, science) while sharing the same frozen backbone. At inference, users can select the appropriate projector for their task or disable it entirely for general-purpose use.

Exploration via stochasticity. The Gumbel-Softmax mechanism introduces controlled stochasticity during both training and inference. Our results show this benefits pass@k at higher k, as the model explores diverse reasoning paths across samples. This contrasts with full vocabulary soft thinking, which achieves higher pass@1 but lower pass@16 due to more deterministic selections.

## 7 Conclusion

We introduced Soft Latent Thinking, a method that enables language models to reason in continuous embedding space through a lightweight projector operating over a compressed vocabulary. By replacing the full vocabulary projection with a small learned projector, we reduce per-step FLOPs during chain-of-thought while achieving competitive or superior performance.

Our experiments on DeepSeek-R1-Distill-Qwen-1.5B and LLaMA-3.2-3B-Instruct demonstrate that Soft Latent Thinking outperforms SofT-GRPO and other baselines at higher pass@k, with improved token efficiency. The approach requires only LoRA adapters on a frozen backbone, preserving out-ofdomain capabilities and enabling modular deployment.

Our analysis reveals that the projector learns subtle but impactful adjustments to token selection, and that the compressed vocabulary introduces beneficial stochasticity for exploring diverse reasoning paths.

## 8 Limitations

The projector is initialized on domain-specific tokens (e.g., mathematical vocabulary), which limits out-of-domain transfer when the projector is enabled. On HumanEval (code), performance degrades with the math-initialized projector, though disabling it and using standard soft thinking with LoRA preserves performance. Future work could explore multi-domain projectors and better domainagnostic initializations. A preliminary PCA/SVD initialization over the embedding table performed poorly in our early experiments, suggesting that useful reasoning traces may occupy a specialized subspace rather than a high-variance global embedding subspace.

## References

AI-MO. 2024. AIMO validation AMC dataset. https://huggingface.co/datasets/AI-MO/ aimo-validation-amc.

Art of Problem Solving. 2025. AIME problems and solutions. https://artofproblemsolving. com/wiki/index.php/AIME\_Problems\_and\_ Solutions.

Natasha Butt, Ariel Kwiatkowski, Ismail Labiad, Julia Kempe, and Yann Ollivier. 2025. Soft tokens, hard truths. arXiv preprint arXiv:2509.19170.

Mark Chen et al. 2021. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168.

DeepSeek-AI. 2025. DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning. Nature, 645(8081):633–638.

Aaron Grattafiori et al. 2024. The Llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Etash Guha et al. 2025. OpenThoughts: Data recipes for reasoning models. arXiv preprint arXiv:2506.04178.

Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, and Yuandong Tian. 2024. Training large language models to reason in a continuous latent space. arXiv preprint arXiv:2412.06769.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring mathematical problem solving with the MATH dataset. arXiv preprint arXiv:2103.03874.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. LoRA: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685.

Eric Jang, Shixiang Gu, and Ben Poole. 2017. Categorical reparameterization with Gumbel-Softmax. In International Conference on Learning Representations.

Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. 2022. Large language models are zero-shot reasoners. arXiv preprint arXiv:2205.11916.

Michael Luo, Sijun Tan, Justin Wong, Xiaoxiang Shi, William Y. Tang, Manan Roongta, Colin Cai, Jeffrey Luo, Tianjun Zhang, Li Erran Li, Raluca Ada Popa,

and Ion Stoica. 2025. DeepScaleR: Surpassing O1- Preview with a 1.5B model by scaling RL. Notion Blog.

Qwen Team. 2026. Qwen3.5: Towards native multimodal agents. Qwen Blog.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. 2024. GPQA: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. 2024. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. 2022. Chain-of-thought prompting elicits reasoning in large language models. arXiv preprint arXiv:2201.11903.

Junhong Wu, Jinliang Lu, Zixuan Ren, Gangqiang Hu, Zhi Wu, Dai Dai, and Hua Wu. 2025. LLMs are single-threaded reasoners: Demystifying the working mechanism of soft thinking. arXiv preprint arXiv:2508.03440.

Jiacheng Ye, Shansan Gong, Liheng Chen, Lin Zheng, Jiahui Gao, Han Shi, Chuan Wu, Xin Jiang, Zhenguo Li, Wei Bi, and Lingpeng Kong. 2024. Diffusion of thoughts: Chain-of-thought reasoning in diffusion language models. arXiv preprint arXiv:2402.07754.

Zhen Zhang, Xuehai He, Weixiang Yan, Ao Shen, Chenyang Zhao, Shuohang Wang, Yelong Shen, and Xin Eric Wang. 2025. Soft thinking: Unlocking the reasoning potential of LLMs in continuous concept space. arXiv preprint arXiv:2505.15778.

Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark Barrett, and Ying Sheng. 2024. SGLang: Efficient execution of structured language model programs. arXiv preprint arXiv:2312.07104.

Zhi Zheng, Yu Gu, Wei Liu, Yee Whye Teh, and Wee Sun Lee. 2025. SofT-GRPO: Surpassing discrete-token LLM reinforcement learning via gumbel-reparameterized soft-thinking policy optimization. arXiv preprint arXiv:2511.06411.