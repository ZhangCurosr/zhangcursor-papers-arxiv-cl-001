# MICROMOUSE: MASKING HARMFUL SUPERVISION IN LONG COT VIA RELATIVE POTENTIAL

Xiaowei Zhu<sup>1,2</sup>, Hengzhu Tang<sup>3</sup>, Zhenyu Zhang<sup>3∗</sup>, Shuohuan Wang<sup>3</sup>, Yubing Ren<sup>1,2†</sup>, Yanan Cao<sup>1,2†</sup>

<sup>1</sup>Institute of Information Engineering, Chinese Academy of Sciences, Beijing, China <sup>2</sup>School of Cyber Security, University of Chinese Academy of Sciences, Beijing, China <sup>3</sup>Baidu Inc., Beijing, China

{zhuxiaowei, renyubing, caoyanan}@iie.ac.cn {tanghengzhu, zhangzhenyu07, wangshuohuan}@baidu.com

## ABSTRACT

Long chain-of-thought (CoT) trajectories have become an important source of supervision for training large reasoning models, yet conventional supervised finetuning (SFT) typically optimizes all tokens in these trajectories indiscriminately. However, even successful long CoTs may contain harmful supervision, motivating the need for methods that can automatically identify and mask such signals during SFT. Inspired by the MicroMouse maze-solving process, we view a long CoT as a search trajectory from the input problem toward the target answer. This perspective motivates maintaining a goal-relative measure, analogous to a Micro-Mouse’s spatial coordinates, to quantify how close the current reasoning state is to the target answer. Building on this perspective, we propose MICROMOUSE, an automated selective-SFT framework for masking harmful supervision in long CoTs. In MICROMOUSE we introduce a Relative Potential (REP) to measure how closely each reasoning state supports the gold answer. Unlike the perplexity of the gold answer alone, REP measures relative support by contrasting the gold answer with a matched foil. This contrast cancels likelihood variations shared by the pair, yielding a more stable signal of progress toward the target answer. By tracking the REP trajectory, MICROMOUSE identifies consecutive potential decreases as erroneous exploration and post-peak reasoning as redundant wandering, and masks these harmful supervision during SFT. Extensive experiments across multiple student model scales and reasoning benchmarks show that MICROMOUSE consistently outperforms Full-CoT SFT and existing selective-SFT baselines, yielding average relative gains of 5.5% in greedy accuracy and 2.6% in Pass@1 relative to Full-CoT SFT. Meanwhile, MICROMOUSE improves reasoning efficiency by reducing average output length by 8.8%, while identifying harmful supervision in only 1.7 seconds per long CoT on average.

## 1 INTRODUCTION

Recent large reasoning models (LRMs) have demonstrated strong performance on complex reasoning tasks by generating extended chains of thought (CoTs) and leveraging increased inferencetime computation (OpenAI, 2024; Guo et al., 2025). CoTs enable models to solve complex problems through verbalized intermediate reasoning steps, substantially improving performance across reasoning-intensive tasks (Wei et al., 2022). Long CoTs have increasingly become an important source of supervision for supervised fine-tuning (SFT), where student models learn reasoning be haviors from high-quality reasoning trajectories (Ye et al., 2025).

However, prevailing full-CoT SFT optimizes all tokens in a reasoning trajectory uniformly, regardless of their actual learning value. This treatment can be suboptimal for long-form CoTs, as even successful trajectories may contain harmful supervision such as erroneous exploration and redundant wandering. Indiscriminately learning from these steps can reinforce undesirable reasoning patterns, thereby degrading both reasoning performance and inference efficiency (Chen et al., 2025a; Cui et al., 2025; Li et al., 2026). This motivates an automated framework that identifies such harmful supervision and selectively masks its contribution to the SFT objective.

![](images/197420d13600e2270351f2c9b80502b6618c0eb2456d7a5eb39cd05fba1dc1c2.jpg)  
Figure 1: Overview of MicroMouse, a framework consisting of three stages.

As shown in Figure 1 (left), our motivation is inspired by the MicroMouse maze-solving problem (Wyard-Scott & Meng, 1995). During the search phase, a MicroMouse explores an unknown maze through a sequence of movements, where erroneous forward and backtracking decisions (i.e., passing the goal and returning) may occur. During the dash phase, it instead follows the computed fastest route to the destination. We draw an analogy between this search trajectory and a long CoT: reasoning steps may likewise contain erroneous exploration or redundant wandering after the target answer has been reached. This inspires a new perspective: maintaining a goal-relative measure throughout reasoning, analogous to a Micromouse’s spatial coordinates, to quantify how close each reasoning step is to the target answer and guide SFT to mask erroneous exploration and redundant wandering.

Building on this intuition, we propose MICROMOUSE, an automated selective-SFT framework for masking harmful supervision in long CoTs. Given a long CoT, we first partition it into a sequence of reasoning steps using naturally occurring transition keywords. We then introduce the Relative Potential (REP) to quantify how close each reasoning state is to the target answer. Perplexity provides a simple and intuitive measure of how strongly a reasoning prefix supports the gold answer, but the gold answer’s perplexity alone captures only absolute confidence and ignores how competing answers change. REP instead contrasts the gold answer with a matched foil through the difference between their log perplexities, equivalently the log-ratio of their perplexities. This relative formulation cancels likelihood variations shared by the matched pair and more directly tracks changes in the model’s preference for the target answer. By tracking the evolution of REP along the trajectory, we systematically categorize reasoning steps into Correct Progress, Necessary Exploration, Erroneous Exploration, and Redundant Wandering. During SFT, tokens belonging to erroneous or redundant steps are masked from the loss while remaining visible in the context.

Extensive experiments show that MICROMOUSE consistently outperforms Full-CoT SFT across student models of different scales and a broad range of reasoning benchmarks. Averaged over Qwen3- 4B, Qwen3-8B, and Qwen3-14B, MICROMOUSE achieves relative improvements of 5.5% in accuracy under greedy decoding and 2.6% in Pass@1 under temperature sampling, while also clearly surpassing existing selective-SFT baselines. In addition, MICROMOUSE reduces the average output length by 8.8%. The harmful supervision identification is also efficient, requiring only 1.7 seconds on average for a long CoT containing 22 reasoning steps.

## 2 METHODOLOGY

## 2.1 PRELIMINARY

Problem Definition. We focus on reasoning tasks with objectively verifiable target answers, including mathematical reasoning and multiple-choice reasoning. A long-form CoT $\begin{array} { r l } { \tau } & { { } = } \end{array}$ $\{ S _ { 1 } , S _ { 2 } \ldots , S _ { n } \}$ typically consists of multiple reasoning steps serving different roles, such as problem decomposition, intermediate derivation, exploration, self-correction, and verification. Even when the CoT eventually reaches the correct answer, it may still contain harmful supervision, such as persistent erroneous exploration and redundant wandering after the solution has effectively been reached. Indiscriminately optimizing all such steps may reinforce undesirable reasoning behaviors and impair both reasoning quality and efficiency. Our goal is to identify reasoning steps that should not contribute to supervision and selectively exclude their token-level losses during SFT.

Relative Potential. Relative Potential (REP) is designed to measure how close the current reasoning state is to the gold answer along a long CoT. Prior work (Xiao et al., 2025; Cui et al., 2025) often uses the perplexity of the gold answer to estimate how strongly a reasoning prefix supports the target answer. However, gold-answer perplexity alone only reflects the absolute likelihood of the target and does not capture how the probability of other answers changes. In particular, the gold answe may become more likely while a nearby incorrect answer increases even more, in which case perplexity would suggest improved progress despite a deterioration in the model’s relative preference for the gold answer. Motivated by this limitation, we instead adopt a contrastive view and compare the conditional log-likelihood of the gold answer with that of a foil answer (Warstadt et al., 2020). The foil is constructed as a minimal contrast to the gold answer: it preserves the answer format and token length while introducing a minimal rule-based perturbation (Gardner et al., 2020) that makes the answer incorrect (see details in Appendix A), e.g., 18 → 19, $A  B ,$ , or $\textstyle { \frac { 4 } { 5 } } \to { \frac { 3 } { 5 } }$ . Given a scoring model M, we define the Relative Potential at reasoning step k as

$$
\phi _ { k } ^ { ( M ) } = \frac { 1 } { L } \sum _ { j = 1 } ^ { L } \left[ \log p _ { M } \left( y _ { j } ^ { * } \mid q , S _ { \le k } , r , y _ { < j } ^ { * } \right) - \log p _ { M } \left( \tilde { y } _ { j } \mid q , S _ { \le k } , r , \tilde { y } _ { < j } \right) \right] = - \log \frac { \mathrm { P P L } _ { k } ^ { \mathrm { g o l d } } } { \mathrm { P P L } _ { k } ^ { \mathrm { f o l l } } } ,\tag{1}
$$

where $q$ denotes the input question, r is an answer probe, and $p _ { M } ( \cdot )$ denotes the conditional probability assigned by the scoring model M, and L is the shared token length of the gold answer $y ^ { * }$ and its foil ${ \tilde { y } } .$ Thus, ϕ<sub>k</sub> represents the normalized conditional log-likelihood advantage of the gold answer over its foil, or equivalently, their log perplexity ratio. A larger REP indicates stronger relative support for the gold answer. Compared with gold-answer PPL alone, REP reduces shared likelihood variations, yielding a more robust estimate of changes in the reasoning state’s support for the gold answer. We provide a formal analysis of these advantages of REP in Appendix B.

## 2.2 OVERVIEW

Given a long-form reasoning trajectory, we first partition it into reasoning steps at naturally occurring transition keywords, yielding $\mathcal { T } \doteq \{ S _ { 1 } , S _ { 2 } , \ldots , S _ { n } \}$ . MICROMOUSE then identifies harmful reasoning steps for supervision through three stages, as shown in Figure 1.

Stage 1: Relative Potential Computation. We compute the Relative Potential at each reasoning step to estimate the progress of the current reasoning state toward the target answer.

Stage 2: Harmful Supervision Step Identification. We analyze the changes in REP across reasoning steps to specifically identify erroneous exploration and redundant wandering.

Stage 3: Selective Loss Masking in SFT. Based on the identified reasoning steps, we construct a selective supervision mask and apply it to the loss optimization of downstream SFT.

## 2.3 STAGE 1: RELATIVE POTENTIAL COMPUTATION

Given a segmented reasoning $\operatorname { C o T } \mathcal { T } = \{ S _ { 1 } , S _ { 2 } , \dots , S _ { n } \}$ , we compute the Relative Potential for each reasoning step sequentially. For the k-th reasoning step, we first construct the input query by concatenating the original question $q ,$ the reasoning prefix $S _ { \leq k }$ , and a fixed answer probe $r ,$ instantiated as $< / \mathrm { t h i n k } > \ S _ { 0 }$ , the final answer $\mathrm { i } s \setminus \mathsf  \backslash b o x e d \{ $ . The resulting input can be written as $[ q ; S _ { \leq k } ; r ] .$ . We feed it into the scoring model M and compute the REP $\phi _ { k } ^ { ( M ) }$ from the conditional log-likelihood difference between the gold answer $y ^ { * }$ and its foil ${ \tilde { y } } ,$ following Eq. 1. Repeating this procedure for $k = 1 , \dots , n$ yields the complete potential trajectory:

$$
\Phi ^ { ( M ) } = \{ \phi _ { 1 } ^ { ( M ) } , \phi _ { 2 } ^ { ( M ) } , \dots , \phi _ { n } ^ { ( M ) } \} ,\tag{2}
$$

which characterizes the evolution of the reasoning state’s proximity to the target answer.

## 2.4 STAGE 2: HARMFUL SUPERVISION STEP IDENTIFICATION

Given the REP trajectory $\Phi ^ { ( M ) } = \{ \phi _ { 1 } ^ { ( M ) } , \phi _ { 2 } ^ { ( M ) } , . . . , \phi _ { n } ^ { ( M ) } \}$ , we use the potential change $\Delta \phi _ { k } ^ { ( M ) } = $ $\phi _ { k } ^ { ( M ) } - \phi _ { k - 1 } ^ { ( M ) }$ to quantify the effect of reasoning step $S _ { k }$ , and define $k ^ { \operatorname* { m a x } } = \arg \operatorname* { m a x } _ { k } \phi _ { k } ^ { ( M ) }$ as the step reaching the maximum REP. Let $\sigma$ denote the REP noise scale estimated from the lag-1 autocovariance of its increments (see details in Appendix A).We categorize all reasoning steps into four categories:

## (a) Correct Progress:

$$
S _ { \mathrm { p r o g r e s s } } = \left\{ S _ { k } \mid k \leq k ^ { \operatorname* { m a x } } , \Delta \phi _ { k } ^ { ( M ) } \geq - \sigma \right\} .\tag{3}
$$

These steps either increase the REP or introduce only minor fluctuations within the noise scale, and are retained for supervision.

## (b) Necessary Exploration:

$$
\begin{array} { r } { \mathcal { S } _ { \mathrm { e x p l o r e } } = \left\{ S _ { k } ~ | ~ k \leq k ^ { \operatorname* { m a x } } , ~ \Delta \phi _ { k } ^ { ( M ) } < - \sigma , ~ \Delta \phi _ { k - 1 } ^ { ( M ) } \geq - \sigma , ~ \Delta \phi _ { k + 1 } ^ { ( M ) } \geq - \sigma \right\} . } \end{array}\tag{4}
$$

These isolated potential decreases may reflect necessary local deviations toward the target answer, so we conservatively retain them for supervision.

## (c) Erroneous Exploration:

$$
\mathcal { S } _ { \mathrm { e r r o r } } = \left\{ S _ { k } ~ | ~ k \leq k ^ { \operatorname* { m a x } } , ~ \Delta \phi _ { k } ^ { ( M ) } < - \sigma , ~ \left( \Delta \phi _ { k - 1 } ^ { ( M ) } < - \sigma ~ \vee ~ \Delta \phi _ { k + 1 } ^ { ( M ) } < - \sigma \right) \right\} .\tag{5}
$$

These steps belong to consecutive potential decreases, indicating persistent movement away from the target answer, and are masked.

## (d) Redundant Wandering:

$$
S _ { \mathrm { r e d u n d a n t } } = \left\{ S _ { k } \mid k > k ^ { \operatorname* { m a x } } \right\} .\tag{6}
$$

These steps occur after the highest-potential reasoning step and may correspond to redundant reasoning that fails to terminate in time, and are therefore masked.

Accordingly, MICROMOUSE keeps $S _ { \mathrm { p r o g r e s s } } \cup S _ { \mathrm { e x p l o r e } }$ while masking $S _ { \mathrm { e r r o r } } \cup S _ { \mathrm { r e d u n d a n t } }$

## 2.5 STAGE 3: SELECTIVE LOSS MASKING IN SFT

Directly removing harmful reasoning steps from the training trajectory may break the continuity of the reasoning CoT and disrupt the contextual dependencies required by subsequent reasoning. We therefore preserve the complete reasoning CoT and selectively mask the corresponding token losses during SFT. Given an input token sequence $T = \{ t _ { 1 } , t _ { 2 } , \ldots , t _ { l } \}$ , the MICROMOUSE objective is formulated as

$$
\mathcal { L } _ { \mathrm { M i c r o M o u s e } ( \theta ) } = - \frac { 1 } { \sum _ { i = 1 } ^ { l } \mathbb { I } ( t _ { i } ) } \sum _ { i = 1 } ^ { l } \mathbb { I } ( t _ { i } ) \log P _ { \theta } \left( t _ { i } \mid q , t _ { < i } ; \theta \right) ,\tag{7}
$$

where $\mathbb { I } ( t _ { i } ) = 0 \mathrm { i f } t _ { i }$ belongs to a reasoning step in $S _ { \mathrm { e r r o r } } \cup S _ { \mathrm { r e d u n d a n t } }$ , and $\mathbb { I } ( t _ { i } ) = 1$ otherwise.

## 3 EXPERIMENTS

## 3.1 EXPERIMENTAL SETUP

Datasets. We use the official LIMO training set (Ye et al., 2025), which contains 817 carefully curated mathematical problems paired with long-form reasoning CoTs, as the supervision data for all SFT experiments. Following a similar evaluation protocol to Ye et al. (2025), we evaluate the result ing models on a diverse suite of reasoning benchmarks covering both in-domain and out-of-domain generalization. The in-domain benchmarks include MATH500 (Hendrycks et al., 2021), AMC23, and AIME24, which closely match the mathematical reasoning distribution of the training data. We further include HMMT26 as a recent competition-mathematics benchmark to assess generalization to newly released and unseen problems. For out-of-domain evaluation, we consider GPQA-Diamond (Rein et al., 2023), Minerva (Lewkowycz et al., 2022), and OlympiadBench (He et al., 2024), which cover challenging scientific, quantitative, and Olympiad-level reasoning problems.

Metrics. We evaluate reasoning performance under greedy decoding and temperature sampling. Greedy decoding uses exact-match accuracy (Acc), while temperature sampling reports Pass@1 and Pass@6, measuring average correctness and the probability of obtaining at least one correct solution within six samples, respectively.

Implementation Details. During training, we use DeepSeek-R1-Distill-Qwen-1.5B as the default scoring model M to compute the Relative Potential and derive the selective supervision masks. We conduct SFT on three dense Qwen3 models, namely Qwen3-4B-Base, Qwen3-8B-Base, and Qwen3-14B-Base. All models are trained for 9 epochs on the 817 long-CoT trajectories from LIMO with a learning rate of $1 \times 1 0 ^ { - 5 }$ . For a fair comparison, Full CoT SFT, MICROMOUSE, and other existing baselines use exactly the same training data and hyperparameters. During evaluation, we evaluate all models under a zero-shot CoT setting with a maximum generation length of 32,768 tokens. Greedy decoding uses $T = 0 ,$ while temperature sampling uses $T = 0 . 6$ . For MATH500, GPQA-Diamond, Minerva, and OlympiadBench, we sample 6 responses per problem. For the smaller AMC23, AIME24, and HMMT26 benchmarks, we increase the number of sampled responses to 16 per problem to obtain a more stable estimate of average performance. All training and evaluation experiments are conducted on 8 NVIDIA H100 GPUs.

## 3.2 MAIN RESULTS

Table 1 presents a comparison across seven reasoning benchmarks, three model scales, and two decoding settings. Compared with the corresponding base models before SFT, both fine-tuning strategies substantially improve reasoning performance, while MICROMOUSE delivers more consistent gains over Full CoT SFT across evaluation settings. Specifically, averaged across the three model scales, MICROMOUSE achieves a relative improvement of 5.5% in accuracy under greedy decoding, and relative improvements of 2.6% and 0.9% in Pass@1 and Pass@6 under temperature sampling. The gains are particularly pronounced for Qwen3-8B, where MICROMOUSE improves over Full CoT SFT by 9.9% under greedy decoding and 4.2% in Pass@1 under temperature sampling.

We further observe that the improvement is generally larger under greedy decoding than under temperature sampling. One plausible explanation is that greedy decoding directly reflects changes to the model’s highest-probability reasoning trajectory, whereas stochastic sampling explores a broader set of candidate trajectories and consequently reduces the sensitivity of aggregate metrics to improvements concentrated around the dominant decoding path. Nevertheless, MICROMOUSE retains consistent gains under temperature sampling, indicating that its benefits are not restricted to a single deterministic decoding trajectory and remain robust under more diverse generation settings.

Performance across Problem Difficulty. To investigate whether MICROMOUSE provides different benefits across problem difficulty, we further break down the MATH500 results on Qwen3-8B according to the five difficulty levels. Following the easy-to-hard split adopted by Sun et al. (2024), we group Levels 1–3 as easy problems and Levels 4–5 as hard problems, as shown in Table 2. Overall, MICROMOUSE improves over Full CoT SFT from 84.4% to 86.5% in Pass@1 and from 95.4% to 96.0% in Pass@6, corresponding to relative improvements of approximately 2.5% and

Table 1: Main results across model scales and reasoning benchmarks. We compare Base, Full CoT SFT, and MICROMOUSE on four in-domain and three out-of-domain benchmarks. Overall reports accuracy under greedy decoding and Pass@1/Pass@6 under temperature sampling. “–” indicates that Pass@6 is not applicable to greedy decoding. The best results are highlighted in bold.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td colspan="4">In-Domain</td><td colspan="3">Out-of-Domain</td><td colspan="2">Overall</td></tr><tr><td>MATH500</td><td>AMC23</td><td>AIME24</td><td>HMMT26</td><td>GPQA</td><td>Minerva</td><td>Olympiad</td><td>Acc./P@1</td><td>P@6</td></tr><tr><td colspan="10">Greedy Decoding (T = 0)</td></tr><tr><td></td><td>Base</td><td>62.6</td><td>47.5</td><td>16.7</td><td>3.0</td><td>17.2</td><td>19.9</td><td>32.6</td><td>28.5</td><td>一</td></tr><tr><td>Qwen3-4B</td><td>Full CoT SFT</td><td>70.8</td><td>47.5</td><td>13.3</td><td>12.1</td><td>35.4</td><td>21.3</td><td>38.8</td><td>34.2</td><td>一</td></tr><tr><td></td><td>MICROMOUSE</td><td>71.6</td><td>55.0</td><td>13.3</td><td>12.1</td><td>36.9</td><td>21.0</td><td>39.3</td><td>35.6↑4.2%</td><td>一</td></tr><tr><td>Qwen3-8B</td><td>Base</td><td>76.2</td><td>62.5</td><td>13.3</td><td>9.1</td><td>41.4</td><td>29.4</td><td>40.3</td><td>38.9</td><td>一</td></tr><tr><td></td><td>Full CoT SFT</td><td>76.0</td><td>57.5</td><td>30.0</td><td>9.1</td><td>42.9</td><td>30.1</td><td>41.6</td><td>41.0</td><td>一</td></tr><tr><td></td><td>MICROMOUSE</td><td>80.4</td><td>70.0</td><td>30.0</td><td>12.1</td><td>44.9</td><td>30.9</td><td>47.4</td><td>45.119.9%</td><td></td></tr><tr><td></td><td>Base</td><td>81.0</td><td>65.0</td><td>13.3</td><td>12.1</td><td>42.9</td><td>32.0</td><td>44.9</td><td>41.6</td><td>一</td></tr><tr><td></td><td>Qwen3-14B Full CoT SFT</td><td>91.4</td><td>85.0</td><td>50.0</td><td>24.2</td><td>55.6</td><td>36.0</td><td>60.4</td><td>57.5</td><td></td></tr><tr><td></td><td>MICROMOUSE</td><td>90.8</td><td>87.5</td><td>43.3</td><td>36.4</td><td>58.6</td><td>36.0</td><td>59.9</td><td> $\pmb { 5 8 . 9 } \uparrow 2 . 5 \%$ </td><td>一</td></tr><tr><td colspan="10">Temperature Sampling (T = 0.6)</td></tr><tr><td></td><td>Base</td><td>35.8</td><td>26.6</td><td>7.2</td><td>3.2</td><td>7.2</td><td>12.1</td><td>20.0</td><td>16.0</td><td>40.5</td></tr><tr><td>Qwen3-4B</td><td>Full CoT SFT</td><td>81.0</td><td>65.3</td><td>20.0</td><td>11.9</td><td>41.1</td><td>30.9</td><td>47.3</td><td>42.5</td><td>62.2</td></tr><tr><td></td><td>MICROMOUSE</td><td>80.5</td><td>65.0</td><td>23.8</td><td>15.2</td><td>42.6</td><td>28.9</td><td>46.4</td><td> $4 3 . 2 ^ { \uparrow 1 . 6 \% }$ </td><td> $6 2 . 4 ^ { \uparrow 0 . 3 \% }$ </td></tr><tr><td></td><td>Base</td><td>72.9</td><td>52.5</td><td>11.9</td><td>5.1</td><td>40.7</td><td>26.0</td><td>37.3</td><td>35.2</td><td>57.2</td></tr><tr><td>Qwen3-8B</td><td>Full CoT SFT</td><td>84.4</td><td>71.4</td><td>30.6</td><td>19.1</td><td>48.7</td><td>34.0</td><td>52.3</td><td>48.6</td><td>68.9</td></tr><tr><td></td><td>MICROMOUSE</td><td>86.5</td><td>73.4</td><td>32.9</td><td>21.8</td><td>51.5</td><td>35.8</td><td>54.5</td><td> $\mathbf { 5 0 . 9 } ^ { \uparrow 4 . 7 \% }$ </td><td> $\mathbf { 7 0 . 3 } ^ { \uparrow 2 . 0 \% }$ </td></tr><tr><td></td><td>Base</td><td>73.8</td><td>53.0</td><td>12.9</td><td>8.9</td><td>42.8</td><td>29.6</td><td>38.8</td><td>37.1</td><td>59.1</td></tr><tr><td></td><td>Qwen3-14B Full CoT SFT</td><td>91.5</td><td>85.2</td><td>46.9</td><td>29.7</td><td>57.3</td><td>35.2</td><td>61.1</td><td>58.1</td><td>73.5</td></tr><tr><td></td><td>MICROMOUSE</td><td>91.3</td><td>85.0</td><td>50.8</td><td>31.1</td><td>59.9</td><td>35.3</td><td>59.6</td><td> ${ \pmb 5 9 . \pmb 0 ^ { \uparrow 1 . 5 \% } }$ </td><td> $7 3 . 8 ^ { \uparrow 0 . 4 \% }$ </td></tr></table>

Table 2: Qwen3-8B temperature sampling performance across MATH500 difficulty levels.
<table><tr><td rowspan="2">Setting</td><td rowspan="2">Metric</td><td colspan="4">Easy</td><td colspan="3">Hard</td><td rowspan="2">Total Avg.</td></tr><tr><td></td><td>Level 1 Level 2 Level 3</td><td></td><td>Avg.</td><td>Level 4 Level 5</td><td></td><td>Avg.</td></tr><tr><td rowspan="2">Full CoT SFT</td><td>Pass@1</td><td>89.1</td><td>93.9</td><td>91.0</td><td>91.8</td><td>82.6</td><td>73.1</td><td>77.7</td><td>84.4</td></tr><tr><td>Pass@6</td><td>95.3</td><td>98.9</td><td>99.0</td><td>98.3</td><td>95.3</td><td>90.3</td><td>92.7</td><td>95.4</td></tr><tr><td rowspan="2">MICROMOUSE</td><td>Pass@1</td><td>92.6</td><td>92.0</td><td>92.1</td><td> $\pmb { 9 2 . 2 } ^ { \uparrow 0 . 4 \% }$ </td><td>85.8</td><td>77.2</td><td> $\mathbf { 8 1 . 4 } ^ { \uparrow 4 . 8 \% }$ </td><td> $8 6 . 5 ^ { \uparrow 2 . 5 \% }$ </td></tr><tr><td>Pass@6</td><td>100.0</td><td>98.9</td><td>98.1</td><td>98.710.4%</td><td>96.1</td><td>91.0</td><td> $9 3 . 5 ^ { \uparrow 0 . 9 \% }$ </td><td> $\mathbf { 9 6 . 0 } ^ { \uparrow 0 . 6 \% }$ </td></tr></table>

0.6%. More importantly, the benefit becomes more pronounced as problem difficulty increases. MI-CROMOUSE improves Pass@1 over Full CoT SFT by only 0.4% on easy problems, but by 4.8% on hard problems, indicating substantially larger gains as problem difficulty increases. A similar tendency is observed for Pass@6, although the margin is smaller due to the near-saturated performance. This pattern is consistent with our intuition that harder problems generally involve more complex and exploratory reasoning, increasing the likelihood of persistent erroneous exploration and redundant reasoning. By selectively suppressing supervision from such reasoning steps, MICROMOUSE provides a larger advantage on problems that require more challenging reasoning trajectories.

## 3.3 COMPARISON WITH EXITING METHODS

In this section, we compare MICROMOUSE on Qwen3-8B against five selective-SFT baselines: (1) First-Correct (Chen et al., 2025a), which retains the segment where the correct answer first appears together with all preceding segments; (2) Segment-Entropy (Li et al., 2026), which prioritizes segments with higher aggregated token-level entropy, indicating greater information contribution; (3) Segment-PPL (Cui et al., 2025), which estimates segment importance by measuring the increase in CoT perplexity after removing a reasoning segment; (4) Gemini Annotate, which directly uses Gemini-3.6-Flash to annotate the importance of each reasoning step, with the annotation prompt

Full CoT SFT First-Correct Segment-Entropy Segment-PPL Gemini Annotate Segment Selective MicroMouse

![](images/9d18caad0a6f31a0688de6b376242dbcc80f8ecf1f396ae4b7e515ae3c869641.jpg)  
Figure 2: Comparison with other selective-SFT baselines on Qwen3-8B. The top row reports reasoning performance under greedy decoding and temperature sampling, while the bottom row compares output length and the computational cost of masking harmful supervision.

provided in the Appendix C; and (5) Segment Selective (Wang et al., 2026b), which identifies important segments by aggregating token-level Integrated Gradients (IG) signals. For Segment-Entropy, Segment-PPL, Segment Selective, and MICROMOUSE, we use the same scoring model and match the supervision budget to ensure a fair comparison.

As shown in the upper row of Figure 2, we compare MICROMOUSE with Full CoT SFT and five selective-SFT baselines across seven reasoning benchmarks. Several selective-SFT strategies consistently outperform Full CoT SFT, providing further evidence that long CoTs contain harmful supervision and that masking such signals can yield stable gains in reasoning performance. Among the six selective-SFT methods, MICROMOUSE achieves the best overall performance under both decoding settings. These results further suggest that measuring each reasoning step by its relative potential enables more accurate identification and masking of harmful supervision in long CoTs. Interestingly, Gemini Annotate achieves the second-best overall performance, indicating that direct reasoning-step annotation with a strong LLM is also effective for selective supervision.

Reasoning Efficiency. The bottom row of Figure 2 further compares the average output length of Full CoT SFT and the six selective-SFT strategies under both greedy decoding and temperature sampling. First-Correct, Segment-Entropy, and Segment-PPL produce output lengths comparable to Full CoT SFT, whereas Gemini Annotate, Segment Selective, and MICROMOUSE yield noticeably shorter generations. In particular, MICROMOUSE reduces the average output length by 9.1% under greedy decoding and 8.4% under temperature sampling relative to Full CoT SFT. Notably, MICROMOUSE-SFT improves reasoning performance while producing shorter generations. This joint improvement suggests that masking harmful supervision during SFT encourages the student model to learn more effective and concise reasoning behaviors, reducing erroneous exploration and redundant wandering while improving reasoning performance. Although Gemini Annotate also reduces output length, MICROMOUSE achieves stronger performance, resulting in a better trade-off.

Computational Overhead. The bottom-right panel of Figure 2 compares the average time cost of the annotation-free methods on all 817 LIMO trajectories, each containing about 22 reasoning steps. Segment-Entropy is the fastest because it requires only one forward pass to compute tokenlevel entropy, whereas Segment-PPL repeatedly removes reasoning steps and recomputes perplexity, resulting in a much higher cost of 94.7 seconds per trajectory. Segment Selective takes 5.2 seconds due to token-level IG computation and aggregation. In contrast, MICROMOUSE only evaluates the conditional log-likelihoods of the gold answer and its matched foil at each step, requiring 1.7 seconds per trajectory and running about 3× faster than Segment Selective. Although Segment-Entropy is cheaper, MICROMOUSE yields substantially stronger downstream performance, achieving a better balance between supervision quality and computational cost for large scale long CoT SFT.

## 3.4 ABLATION STUDY

We investigate the contributions of individual MICROMOUSE components and the effect of replacing the scoring model, with results summarized in Table 3. For component ablations, we consider four variants: (1) Random Mask, which randomly masks reasoning segments until matching the supervision budget of MICRO-MOUSE; (2) w/o Mask $S _ { \mathrm { e r r o r } } ,$ which does not mask reasoning steps identified as erroneous exploration; (3) w/o Mask $S _ { \mathrm { r e d u n d a n t } }$ , which does not mask reasoning steps identified as re-

Table 3: Ablation studies on Qwen3-8B.
<table><tr><td></td><td></td><td colspan="2">Greedy</td><td colspan="3">Sampling</td></tr><tr><td>Variant</td><td>Keep (%)</td><td></td><td>Acc. Len. (k)</td><td></td><td>P@1 P@6 Len. (k)</td><td></td></tr><tr><td>Full CoT SFT</td><td>100.0</td><td>41.0</td><td>18.94</td><td></td><td>48.6 68.9</td><td>14.06</td></tr><tr><td>Random Mask</td><td>80.9</td><td>41.7</td><td>17.89</td><td>49.1</td><td>68.2</td><td>14.01</td></tr><tr><td>MICROMOUSE</td><td>80.9</td><td>45.1</td><td>17.22</td><td>50.9</td><td>70.3</td><td>12.88</td></tr><tr><td>w/o Mask  $S _ { \mathrm { e r r o r } }$ </td><td>83.5</td><td>44.4</td><td>16.67</td><td>50.7</td><td>70.2</td><td>12.84</td></tr><tr><td>w/o Mask Sredundant</td><td>97.1</td><td>38.9</td><td>19.34</td><td>49.0</td><td>69.0</td><td>14.11</td></tr><tr><td>w/ Mask  $\scriptstyle S _ { \mathrm { e x p l o r e } }$ </td><td>74.1</td><td>41.8</td><td>18.06</td><td>48.9</td><td>68.2</td><td>13.81</td></tr><tr><td>w/ Qwen3-4B</td><td>85.5</td><td>41.9</td><td>18.27</td><td></td><td>49.0 70.2</td><td>14.18</td></tr></table>

dundant wandering; and (4) w/Mask $S _ { \mathrm { e x p l o r e } } ,$ which additionally masks reasoning segments exhibiting a single-step decrease in REP. We further replace the default DeepSeek-R1-Distill-Qwen-1.5B with Qwen3-4B-Instruct to assess the sensitivity of MICROMOUSE to the choice of scoring model.

The component ablations reveal that the gains of MICROMOUSE arise primarily from which reasoning steps are excluded from supervision, rather than from simply reducing the amount of training signal. Random Mask, despite using the same supervision budget as MICROMOUSE, yields only marginal changes over Full CoT SFT, while remaining substantially below MICROMOUSE across all performance metrics. Removing the masking of either erroneous exploration or redundant wandering also degrades performance, supporting the necessity of suppressing these two types of harmful supervision. In particular, failing to mask redundant wandering results in relative performance drops of 13.7% in greedy accuracy, 3.4% in Pass@1, and 1.9% in Pass@6 under temperature sampling. Conversely, additionally masking reasoning steps with only a single-step decrease in REP causes a marked performance degradation. This result supports our design choice of preserving isolated potential decreases, which may correspond to necessary intermediate exploration rather than genuinely harmful supervision signals. Finally, replacing the default scoring model with Qwen3-4B-Instruct still yields consistent improvements over Full CoT SFT, including a 1.9% relative gain in Pass@6. This result indicates that the effectiveness of MICROMOUSE is not restricted to the default score and remains stable when using another alternative scoring model.

## 3.5 CASE STUDY

Figure 3 visualizes the Relative Potential trajectory of MICROMOUSE on LIMO sample #539 and compares its step-wise supervision decisions with those of other selective-SFT baselines. The original reasoning trajectory is provided in Appendix D. Several observations emerge from this example. First, an isolated decrease in Relative Potential does not necessarily indicate harmful supervision. At Step 5, the model reaches the correct answer through a logical jump, causing a temporary decrease in Relative Potential. Although the argument at this step is insufficiently justified, it still advances toward the target answer, and the subsequent step begins to further verify the result. Second, consecutive decreases in Relative Potential correspond more clearly to erroneous exploration. At Steps 7–8, the reasoning repeatedly recomputes an already established counting argument and then transitions into hesitation without introducing new progress. Third, reasoning steps after the Relative Potential reaches its peak exhibit clear redundant wandering. By Step 12, the model has repeatedly cross-validated the derivation and reached the target answer with high confidence. Nevertheless, Steps 13–15 continue to revisit alternative formulations and perform additional sanity checks, despite no longer contributing substantive progress toward the solution.

Taken together, this example provides empirical support for MICROMOUSE’s identification of erroneous exploration and redundant wandering based on the evolution of REP. It also illustrates why isolated potential drops should be treated conservatively, as they may correspond to necessary exploration rather than genuinely harmful supervision. In contrast, other selective-SFT baselines are either overly aggressive or permissive, leading to less accurate supervision selection on this example.

![](images/c7f8b5b2748424291af77a40a169ef4951ce82fdb5f1e36898bdb31bec745bd3.jpg)  
Figure 3: Case study of step-wise supervision analysis on LIMO #539. The figure visualizes the Relative Potential trajectory and compares the supervision decisions of MICROMOUSE with selective-SFT baselines across all reasoning steps.

## 4 RELATED WORK

Reasoning Efficiency of Large Reasoning Models. Large reasoning models (LRMs) achieve strong performance through extended chains of thought (CoTs), but long-form reasoning often incurs substantial computational overhead and redundant reasoning (Sui et al., 2025; Wang et al., 2026a). Existing approaches improve reasoning efficiency either at inference time or through training. Prompt- and budget-based methods explicitly constrain test-time computation (Han et al., 2025; Renze & Guven, 2024), while latent-reasoning methods reduce explicit verbalized computation by performing intermediate reasoning in continuous representations (Hao et al., 2025). Training-based methods instead refine or compress CoT supervision, for example by identifying critical reasoning steps using perplexity-based criteria (Cui et al., 2025). In contrast, MICROMOUSE targets the quality rather than the length of long-CoT supervision, preserving the complete trajectory as context while excluding harmful reasoning units from loss optimization.

Selective SFT and Importance Measurement for Reasoning CoTs. Recent work challenges the conventional SFT objective that optimizes all target tokens uniformly. Rho-1 selectively applies loss to high-utility tokens identified by a reference model (Lin et al., 2025), Goldfish Loss randomly excludes subsets of target tokens from optimization (Hans et al., 2024), and Token Cleaning removes low-quality token-level supervision during SFT (Pang et al., 2025). At the step level, ATLAS identifies critical steps in agent trajectories and fine-tunes only on selected actions (Chen et al., 2025b), while GLM-5 (GLM-5-Team et al., 2026) masks erroneous trajectory segments while retaining them as context for subsequent correction. For long reasoning CoTs, the central challenge is therefore to determine which parts of a trajectory should contribute to supervision. Existing methods operate primarily at token or segment granularity. Token-level approaches use signals such as logits, entropy, gradients, or contrastive differences to identify informative or error-critical tokens (Cheng et al., 2025; Wang et al., 2025). Since individual tokens may not correspond to semantically complete reasoning decisions, segment-level methods instead evaluate coherent reasoning units using perplexity (Cui et al., 2025), step entropy (Li et al., 2026), or perturbation-based attribution (Xu et al., 2025), while Wang et al. (2026b) aggregate token-level Integrated Gradients into segment-level attribution signals. Rather than assigning each segment an isolated importance score, MICROMOUSE tracks the evolution of Relative Potential along the reasoning trajectory to identify erroneous exploration and redundant wandering, and masks the corresponding harmful supervision during SFT.

## 5 CONCLUSION

In this work, we propose MICROMOUSE, an automated selective-SFT framework for masking harmful supervision in long CoTs. MICROMOUSE maintains a Relative Potential to measure how close each reasoning state is to the target answer, enabling the identification of erroneous exploration and redundant wandering along long reasoning trajectories. Unlike single answer log-perplexity measures, Relative Potential is defined through the conditional log-likelihood difference between the gold answer and a matched foil answer, which reduces shared likelihood fluctuations and provides a more stable signal for tracking reasoning progress. Extensive experiments s show that MICRO-MOUSE consistently outperforms Full CoT SFT and achieves stronger overall performance than existing selective-SFT baselines. It also delivers clear advantages in reasoning efficiency and computational overhead. Overall, MICROMOUSE provides an effective and scalable approach for iden tifying and masking harmful supervision in large-scale long-CoT SFT.

## REFERENCES

Xingyu Chen, Jiahao Xu, Tian Liang, Zhiwei He, Jianhui Pang, Dian Yu, Linfeng Song, Qiuzhi Liu, Mengfei Zhou, Zhuosheng Zhang, Rui Wang, Zhaopeng Tu, Haitao Mi, and Dong Yu. Do NOT think that much for 2+3=? on the overthinking of long reasoning models. In Forty-second International Conference on Machine Learning, 2025a. URL https://openreview.net/ forum?id=MSbU3L7V00.

Zhixun Chen, Ming Li, Yuxuan Huang, Yali Du, Meng Fang, and Tianyi Zhou. ATLAS: Agent tuning via learning critical steps. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.), Findings of the Association for Computational Linguistics: ACL 2025, pp. 25334–25349, Vienna, Austria, July 2025b. Association for Computational Linguistics. ISBN 979-8-89176-256-5. doi: 10.18653/v1/2025.findings-acl.1299. URL https: //aclanthology.org/2025.findings-acl.1299/.

Daixuan Cheng, Shaohan Huang, Xuekai Zhu, Bo Dai, Wayne Xin Zhao, Zhenliang Zhang, and Furu Wei. Reasoning with exploration: An entropy perspective, 2025. URL https://arxiv. org/abs/2506.14758.

Yingqian Cui, Pengfei He, Jingying Zeng, Hui Liu, Xianfeng Tang, Zhenwei Dai, Yan Han, Chen Luo, Jing Huang, Zhen Li, Suhang Wang, Yue Xing, Jiliang Tang, and Qi He. Stepwise perplexityguided refinement for efficient chain-of-thought reasoning in large language models. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.), Findings of the Association for Computational Linguistics: ACL 2025, pp. 18581–18597, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-256-5. doi: 10.18653/ v1/2025.findings-acl.956. URL https://aclanthology.org/2025.findings-acl. 956/.

Matt Gardner, Yoav Artzi, Victoria Basmov, Jonathan Berant, Ben Bogin, Sihao Chen, Pradeep Dasigi, Dheeru Dua, Yanai Elazar, Ananth Gottumukkala, Nitish Gupta, Hannaneh Hajishirzi, Gabriel Ilharco, Daniel Khashabi, Kevin Lin, Jiangming Liu, Nelson F. Liu, Phoebe Mulcaire, Qiang Ning, Sameer Singh, Noah A. Smith, Sanjay Subramanian, Reut Tsarfaty, Eric Wallace, Ally Zhang, and Ben Zhou. Evaluating models’ local decision boundaries via contrast sets. In Trevor Cohn, Yulan He, and Yang Liu (eds.), Findings of the Association for Computational Linguistics: EMNLP 2020, pp. 1307–1323, Online, November 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.findings-emnlp.117. URL https: //aclanthology.org/2020.findings-emnlp.117/.

GLM-5-Team, :, Aohan Zeng, Xin Lv, Zhenyu Hou, Zhengxiao Du, Qinkai Zheng, Bin Chen, Da Yin, Chendi Ge, Chenghua Huang, Chengxing Xie, Chenzheng Zhu, Congfeng Yin, Cunxiang Wang, Gengzheng Pan, Hao Zeng, Haoke Zhang, Haoran Wang, Huilong Chen, Jiajie Zhang, Jian Jiao, Jiaqi Guo, Jingsen Wang, Jingzhao Du, Jinzhu Wu, Kedong Wang, Lei Li, Lin Fan, Lucen

Zhong, Mingdao Liu, Mingming Zhao, Pengfan Du, Qian Dong, Rui Lu, Shuang-Li, Shulin Cao, Song Liu, Ting Jiang, Xiaodong Chen, Xiaohan Zhang, Xuancheng Huang, Xuezhen Dong, Yabo Xu, Yao Wei, Yifan An, Yilin Niu, Yitong Zhu, Yuanhao Wen, Yukuo Cen, Yushi Bai, Zhongpei Qiao, Zihan Wang, Zikang Wang, Zilin Zhu, Ziqiang Liu, Zixuan Li, Bojie Wang, Bosi Wen, Can Huang, Changpeng Cai, Chao Yu, Chen Li, Chengwei Hu, Chenhui Zhang, Dan Zhang, Daoyan Lin, Dayong Yang, Di Wang, Ding Ai, Erle Zhu, Fangzhou Yi, Feiyu Chen, Guohong Wen, Hailong Sun, Haisha Zhao, Haiyi Hu, Hanchen Zhang, Hanrui Liu, Hanyu Zhang, Hao Peng, Hao Tai, Haobo Zhang, He Liu, Hongwei Wang, Hongxi Yan, Hongyu Ge, Huan Liu, Huanpeng Chu, Jia’ni Zhao, Jiachen Wang, Jiajing Zhao, Jiamin Ren, Jiapeng Wang, Jiaxin Zhang, Jiayi Gui, Jiayue Zhao, Jijie Li, Jing An, Jing Li, Jingwei Yuan, Jinhua Du, Jinxin Liu, Junkai Zhi, Junwen Duan, Kaiyue Zhou, Kangjian Wei, Ke Wang, Keyun Luo, Laiqiang Zhang, Leigang Sha, Liang Xu, Lindong Wu, Lintao Ding, Lu Chen, Minghao Li, Nianyi Lin, Pan Ta, Qiang Zou, Rongjun Song, Ruiqi Yang, Shangqing Tu, Shangtong Yang, Shaoxiang Wu, Shengyan Zhang, Shijie Li, Shuang Li, Shuyi Fan, Wei Qin, Wei Tian, Weining Zhang, Wenbo Yu, Wenjie Liang, Xiang Kuang, Xiangmeng Cheng, Xiangyang Li, Xiaoquan Yan, Xiaowei Hu, Xiaoying Ling, Xing Fan, Xingye Xia, Xinyuan Zhang, Xinze Zhang, Xirui Pan, Xu Zou, Xunkai Zhang, Yadi Liu, Yan dong Wu, Yanfu Li, Yidong Wang, Yifan Zhu, Yijun Tan, Yilin Zhou, Yiming Pan, Ying Zhang, Yinpei Su, Yipeng Geng, Yong Yan, Yonglin Tan, Yuean Bi, Yuhan Shen, Yuhao Yang, Yujiang Li, Yunan Liu, Yunqing Wang, Yuntao Li, Yurong Wu, Yutao Zhang, Yuxi Duan, Yuxuan Zhang, Zezhen Liu, Zhengtao Jiang, Zhenhe Yan, Zheyu Zhang, Zhixiang Wei, Zhuo Chen, Zhuoer Feng, Zijun Yao, Ziwei Chai, Ziyuan Wang, Zuzhou Zhang, Bin Xu, Minlie Huang, Hongning Wang, Juanzi Li, Yuxiao Dong, and Jie Tang. Glm-5: from vibe coding to agentic engineering, 2026. URL https://arxiv.org/abs/2602.15763.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Hanwei Xu, Honghui Ding, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jingchang Chen, Jingyang Yuan, Jinhao Tu, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaichao You, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Mingxu Zhou, Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, R. L. Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, S. S. Li, Shuang Zhou, Shaoqing Wu, Tao Yun, Tian Pei, Tianyu Sun, T. Wang, Wangding Zeng, Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, W. L. Xiao, Wei An, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, X. Q. Li, Xiangyue Jin, Xiaojin Shen, Xiaosha Chen, Xiaowen Sun, Xiaoxiang Wang, Xinnan Song, Xinyi Zhou, Xianzu Wang, Xinxia Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Yang Zhang, Yanhong Xu, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Yu, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yuan Ou, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yunfan Xiong, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Y. X. Zhu, Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Ying Tang, Yukun Zha, Yuting Yan, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhicheng Ma, Zhigang Yan, Zhiyu Wu, Zihui Gu, Zijia Zhu, Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Zizheng Pan, Zhen Huang, Zhipeng Xu, Zhongyu Zhang, and Zhen Zhang. Deepseek-r1 incentivizes reasoning in llms through reinforcement learning. Nature, 645(8081):633–638, 2025. ISSN 1476-4687. doi: 10.1038/s41586-025-09422-z. URL http://dx.doi.org/10.1038/s41586-025-09422-z.

Tingxu Han, Zhenting Wang, Chunrong Fang, Shiyu Zhao, Shiqing Ma, and Zhenyu Chen. Tokenbudget-aware LLM reasoning. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.), Findings of the Association for Computational Linguistics: ACL 2025, pp. 24842–24855, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-256-5. doi: 10.18653/v1/2025.findings-acl.1274. URL https: //aclanthology.org/2025.findings-acl.1274/.

Abhimanyu Hans, John Kirchenbauer, Yuxin Wen, Neel Jain, Hamid Kazemi, Prajwal Singhania, Siddharth Singh, Gowthami Somepalli, Jonas Geiping, Abhinav Bhatele, and Tom Goldstein. Be like a goldfish, don’t memorize! mitigating memorization in generative LLMs. In The Thirtyeighth Annual Conference on Neural Information Processing Systems, 2024. URL https:// openreview.net/forum?id=DylSyAfmWs.

Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason E Weston, and Yuandong Tian. Training large language models to reason in a continuous latent space. In Second Conference on Language Modeling, 2025. URL https://openreview.net/forum?id= Itxz7S4Ip3.

Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, Jie Liu, Lei Qi, Zhiyuan Liu, and Maosong Sun. OlympiadBench: A challenging benchmark for promoting AGI with olympiad-level bilingual multimodal scientific problems. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 3828–3850, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.211. URL https://aclanthology.org/2024. acl-long.211/.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. CoRR, abs/2103.03874, 2021. URL https://arxiv.org/abs/2103.03874.

Aitor Lewkowycz, Anders Johan Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Venkatesh Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, Yuhuai Wu, Behnam Neyshabur, Guy Gur-Ari, and Vedant Misra. Solving quantitative reasoning problems with language models. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho (eds.), Advances in Neural Information Processing Systems, 2022. URL https://openreview.net/forum?id=IFXTZERXdM7.

Zeju Li, Jianyuan Zhong, Ziyang Zheng, Xiangyu Wen, Zhijian Xu, Yingying Cheng, Fan Zhang, and Qiang Xu. Making slow thinking faster: Compressing LLM chain-of-thought via step entropy. In The Fourteenth International Conference on Learning Representations, 2026. URL https: //openreview.net/forum?id=cGLqQfS5wH.

Zhenghao Lin, Zhibin Gou, Yeyun Gong, Xiao Liu, Yelong Shen, Ruochen Xu, Chen Lin, Yujiu Yang, Jian Jiao, Nan Duan, and Weizhu Chen. Rho-1: Not all tokens are what you need, 2025. URL https://arxiv.org/abs/2404.07965.

OpenAI. Openai o1 system card. CoRR, abs/2412.16720, 2024. doi: 10.48550/ARXIV.2412.16720. URL https://doi.org/10.48550/arXiv.2412.16720.

Jinlong Pang, Na Di, Zhaowei Zhu, Jiaheng Wei, Hao Cheng, Chen Qian, and Yang Liu. Token cleaning: Fine-grained data selection for LLM supervised fine-tuning. In Forty-second Interna tional Conference on Machine Learning, 2025. URL https://openreview.net/forum? id=tXkOUS3vLS.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. Gpqa: A graduate-level google-proof q&a benchmark, 2023. URL https://arxiv.org/abs/2311.12022.

Matthew Renze and Erhan Guven. The benefits of a concise chain of thought on problem-solving in large language models. In 2024 2nd International Conference on Foundation and Large Language Models (FLLM), pp. 476–483. IEEE, November 2024. doi: 10.1109/fllm63129.2024.10852493. URL http://dx.doi.org/10.1109/FLLM63129.2024.10852493.

Yang Sui, Yu-Neng Chuang, Guanchu Wang, Jiamu Zhang, Tianyi Zhang, Jiayi Yuan, Hongyi Liu, Andrew Wen, Shaochen Zhong, Na Zou, Hanjie Chen, and Xia Hu. Stop overthinking: A survey on efficient reasoning for large language models. Transactions on Machine Learning Research, 2025. ISSN 2835-8856. URL https://openreview.net/forum?id=HvoG8SxggZ.

Zhiqing Sun, Longhui Yu, Yikang Shen, Weiyang Liu, Yiming Yang, Sean Welleck, and Chuang Gan. Easy-to-hard generalization: Scalable alignment beyond human supervision, 2024. URL https://arxiv.org/abs/2403.09472.

Rui Wang, Hongru Wang, Boyang Xue, Yixia Li, Jianhui Pang, Shudong Liu, Yi Chen, Jiahao Qiu, Derek Fai Wong, Guanhua Chen, Heng Ji, and Kam-Fai Wong. Harnessing the reasoning economy: A survey of efficient reasoning for large language models, 2026a. URL https: //arxiv.org/abs/2503.24377.

Shenzhi Wang, Le Yu, Chang Gao, Chujie Zheng, Shixuan Liu, Rui Lu, Kai Dang, Xiong-Hui Chen, Jianxin Yang, Zhenru Zhang, Yuqiong Liu, An Yang, Andrew Zhao, Yang Yue, Shiji Song, Bowen Yu, Gao Huang, and Junyang Lin. Beyond the 80/20 rule: High-entropy minority tokens drive effective reinforcement learning for LLM reasoning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum? id=yfcpdY4gMP.

Siyuan Wang, Yanchen Liu, and Xiang Ren. Segment-level attribution for selective learning of long reasoning traces. In The Fourteenth International Conference on Learning Representations, 2026b. URL https://openreview.net/forum?id=C1UD4FLFPL.

Alex Warstadt, Alicia Parrish, Haokun Liu, Anhad Mohananey, Wei Peng, Sheng-Fu Wang, and Samuel R. Bowman. BLiMP: The benchmark of linguistic minimal pairs for English. Transactions of the Association for Computational Linguistics, 8:377–392, 2020. doi: 10.1162/ tacl a 00321. URL https://aclanthology.org/2020.tacl-1.25/.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, brian ichter, Fei Xia, Ed Chi, Quoc V Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (eds.), Advances in Neural Information Processing Systems, volume 35, pp. 24824–24837. Curran Associates, Inc., 2022. doi: 10.52202/ 068431-1800. URL https://proceedings.neurips.cc/paper\_files/paper/ 2022/file/9d5609613524ecf4f15af0f7b31abca4-Paper-Conference.pdf.

L. Wyard-Scott and Q.-H.M. Meng. A potential maze solving algorithm for a micromouse robot. In IEEE Pacific Rim Conference on Communications, Computers, and Signal Processing. Proceed ings, pp. 614–618, 1995. doi: 10.1109/PACRIM.1995.520440.

Yang Xiao, Jessie Wang, Ruifeng Yuan, Chunpu Xu, Kaishuai Xu, Wenjie Li, and Pengfei Liu. LIMOPro: Reasoning refinement for efficient and effective test-time scaling. In The Thirtyninth Annual Conference on Neural Information Processing Systems, 2025. URL https:// openreview.net/forum?id=W9Y0jtf45v.

Zhentao Xu, Fengyi Li, Albert C. Chen, and Xiaofeng Wang. ProCut: LLM prompt compression via attribution estimation. In Saloni Potdar, Lina Rojas-Barahona, and Sebastien Montella (eds.), Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: Industry Track, pp. 285–309, Suzhou (China), November 2025. Association for Computational Linguistics. ISBN 979-8-89176-333-3. doi: 10.18653/v1/2025.emnlp-industry.20. URL https: //aclanthology.org/2025.emnlp-industry.20/.

Yixin Ye, Zhen Huang, Yang Xiao, Ethan Chern, Shijie Xia, and Pengfei Liu. LIMO: Less is more for reasoning. In Second Conference on Language Modeling, 2025. URL https:// openreview.net/forum?id=T2TZ0RY4Zk.

## A FOIL CONSTRUCTION AND NOISE-SCALE ESTIMATION

## A.1 FOIL CONSTRUCTION

The potential function is read out as a length-normalised log-likelihood ratio between the gold answer and a foil: an answer string that matches the gold one in token count and surface format but carries a minimally different, incorrect value. Matching length and format holds everything except value-correctness fixed, so the ratio measures only whether the reasoning prefix has pinned down the right value; taking the nearest wrong value makes the foil the hardest available competitor, which keeps the signal sensitive to reasoning progress instead of saturating early. Answers are numeric: an integer answer v is perturbed to v + 1, falling back to v − 1 whenever v + 1 would change the digit count or equal zero, and a fractional answer is perturbed by incrementing its numerator.

```python
import re
def make_foil(ans):
'''Minimally-perturbed counterfactual answer:
same digit count, same format, nearest value, but wrong.'''
a = str(ans).strip()
m = re.fullmatch(r'-?\d+', a) # integer
answers
if m:
v = int(a)
d = v + 1 if v + 1 != 0 else v + 2 # never use 0 as
a foil
if len(str(abs(d))) != len(str(abs(v))): # keep digit
count stable
d = v - 1 if v - 1 != 0 else v - 2
return str(d)
num, den = re.fullmatch(r'(-?\d+)/(\d+)', a).groups() # simple
fractions
return '%d/%s' % (int(num) + 1, den)
```

## A.2 NOISE SCALE AS THE DESCENT THRESHOLD

Because ϕ is estimated by a finite forward pass of a scorer model, each increment $\Delta \phi _ { k } = \phi _ { k } - $ $\phi _ { k - 1 }$ carries estimation jitter on top of genuine progress, so a threshold is needed before a negative increment may be called a descent. Writing the observed potential as signal plus independent noise, $\phi _ { k } = s _ { k } + \eta _ { k }$ with $\eta _ { k }$ i.i.d. of variance $\sigma ^ { 2 } .$ , first differencing turns the noise into the MA(1) term $\eta _ { k } ~ - ~ \eta _ { k - 1 }$ , whose lag-1 autocovariance is exactly $- \sigma ^ { 2 }$ , while a smooth signal drift contributes approximately zero at lag 1:

$$
\mathrm { C o v } \left( \Delta \phi _ { k } , \Delta \phi _ { k + 1 } \right) \ \approx \ - \sigma ^ { 2 } .
$$

A negative lag-1 autocovariance is therefore the fingerprint of the noise alone, and we take $\sigma = \sqrt { - \mathrm { C o v } ( \Delta \phi _ { k } , \Delta \phi _ { k + 1 } ) }$ as the noise scale, which serves directly as the descent threshold: a step counts as a significant descent only when $\Delta \phi _ { k } < - \sigma ,$ , whereas a step with $| \Delta \phi _ { k } | \le \sigma$ is indistinguishable from noise and is left to be learned, neither breaking a run nor counting toward its length. Since σ shares the units of $\Delta \phi ,$ , the criterion introduces no tunable threshold and adapts to each trace.

```python
import numpy as np
def noise_scale(phi):
'''sigma from the lag-1 autocovariance of the increments:
Cov(d_k, d_{k+1}) = -sigmaˆ2 for i.i.d. noise.'''
d = np.diff(np.asarray(phi, dtype=np.float64))
x = d - d.mean()
c1 = float(np.mean(x[:-1] x[1:])) # lag-1
autocovariance
return float(np.sqrt(max(0.0, -c1)))
# the noise scale is the threshold itself
# significant descent: d < -sigma
# noise-level step (|d| <= sigma): learned, does not break or extend a
run
```

## B WHY REP PROVIDES A ROBUST SIGNAL OF REASONING PROGRESS

We provide a formal interpretation of REP from the perspective of perplexity and show why its change provides a more robust directional signal than gold-answer perplexity alone.

REP as Relative Perplexity. For a candidate answer a, define its length-normalized negative loglikelihood at reasoning step k as

$$
\mathcal { N } _ { k } ^ { a } = - \frac { 1 } { L } \sum _ { j = 1 } ^ { L } \log p _ { M } \left( a _ { j } \mid q , S _ { \leq k } , r , a _ { < j } \right) .\tag{8}
$$

By definition, $\mathrm { P P L } _ { k } ^ { a } = \exp ( \mathcal { N } _ { k } ^ { a } )$ . Therefore, Eq. 1 can be equivalently written as

$$
\phi _ { k } = \mathcal { N } _ { k } ^ { \mathrm { f o i l } } - \mathcal { N } _ { k } ^ { \mathrm { g o l d } } = \log \frac { \mathrm { P P L } _ { k } ^ { \mathrm { f o i l } } } { \mathrm { P P L } _ { k } ^ { \mathrm { g o l d } } } .\tag{9}
$$

Hence, REP measures the perplexity of the foil relative to the gold answer. A larger $\phi _ { k }$ means that the scoring model finds the gold answer relatively easier to predict than its matched foil.

Robustness to Common-Mode Likelihood Fluctuations. Consider a common-mode perturbation that affects both candidates under the same reasoning prefix. We decompose their observed log perplexities as

$$
\mathrm { l o g } \mathrm { P P L } _ { k } ^ { a } = \bar { \mathcal { N } } _ { k } ^ { a } + c _ { k } + \epsilon _ { k } ^ { a } , \qquad a \in \{ \mathrm { g o l d } , \mathrm { f o i l } \} ,\tag{10}
$$

where $\bar { \mathcal { N } } _ { k } ^ { a }$ denotes the candidate-specific signal of interest, $c _ { k }$ is a prefix-dependent fluctuation shared by both candidates, and $\epsilon _ { k } ^ { a }$ is candidate-specific residual noise.

Using gold-answer PPL alone gives

$$
\log \mathrm { P P L } _ { k } ^ { \mathrm { g o l d } } = \bar { \mathcal { N } } _ { k } ^ { \mathrm { g o l d } } + c _ { k } + \epsilon _ { k } ^ { \mathrm { g o l d } } ,\tag{11}
$$

so its variation is directly contaminated by the common term $c _ { k }$ . In contrast, substituting Eq. 10 into Eq. 9 yields

$$
\phi _ { k } = \bar { \mathcal { N } } _ { k } ^ { \mathrm { f o i l } } - \bar { \mathcal { N } } _ { k } ^ { \mathrm { g o l d } } + \epsilon _ { k } ^ { \mathrm { f o i l } } - \epsilon _ { k } ^ { \mathrm { g o l d } } ,\tag{12}
$$

where the shared fluctuation $c _ { k }$ cancels exactly. Therefore, when likelihood variation is dominated by common effects shared by the matched gold–foil pair, REP isolates their relative preference more faithfully than either candidate’s absolute PPL.

The matched construction of the foil further supports this assumption: the gold and foil share the same reasoning prefix, answer format, and token length, differing only by a minimal answer perturbation. Their nuisance likelihood fluctuations are therefore expected to be strongly correlated, making contrastive cancellation particularly effective.

Directional Meaning of $\Delta \phi _ { k }$ . To make the meaning of REP change explicit, define the perplexity improvement of a candidate answer a after incorporating reasoning step $S _ { k }$ as

$$
I _ { k } ^ { a } = \log \frac { \mathrm { P P L } _ { k - 1 } ^ { a } } { \mathrm { P P L } _ { k } ^ { a } } , \qquad a \in \{ \mathrm { g o l d } , \mathrm { f o i l } \} .\tag{13}
$$

A positive $I _ { k } ^ { a }$ indicates that the candidate becomes easier to predict after adding $S _ { k }$ . From $\operatorname { E q . }$ . 1,

$$
\begin{array} { r l } & { \Delta \phi _ { k } = \phi _ { k } - \phi _ { k - 1 } } \\ & { \qquad = \log \frac { \mathrm { P P L } _ { k } ^ { \mathrm { f o i l } } } { \mathrm { P P L } _ { k } ^ { \mathrm { g o l d } } } - \log \frac { \mathrm { P P L } _ { k - 1 } ^ { \mathrm { f o i l } } } { \mathrm { P P L } _ { k - 1 } ^ { \mathrm { g o l d } } } } \\ & { \qquad = \log \frac { \mathrm { P P L } _ { k - 1 } ^ { \mathrm { g o l d } } } { \mathrm { P P L } _ { k } ^ { \mathrm { g o l d } } } - \log \frac { \mathrm { P P L } _ { k - 1 } ^ { \mathrm { f o i l } } } { \mathrm { P P L } _ { k } ^ { \mathrm { f o i l } } } } \\ & { \qquad = I _ { k } ^ { \mathrm { g o l d } } - I _ { k } ^ { \mathrm { f o i l } } . } \end{array}\tag{14}
$$

Therefore, $\Delta \phi _ { k } > 0$ means that the gold answer improves in predictability more than its matched foil, while $\Delta \phi _ { k } < 0$ means that the foil improves relatively more. Importantly, a positive $\Delta \phi _ { k }$ does not require the foil perplexity to increase; both candidates may become easier to predict, as long as the gold answer improves by a larger relative margin. Thus, $\Delta \phi _ { k }$ directly measures whether reasoning step $S _ { k }$ strengthens or weakens the model’s relative support for the target answer.

## C PROMPT FOR GEMINI-BASED SEGMENT ANNOTATION

We use the following prompt to instruct Gemini-3.6-Flash to annotate whether each reasoning step should contribute to SFT supervision.

You are helping curate training data for supervised fine-tuning (SFT)   
of a smaller student reasoning model. You are given a math PROBLEM,   
its correct FINAL ANSWER, and a reasoning trace that has been split   
into indexed SEGMENTS. For EACH segment decide whether it is NECESSARY   
for the student to learn.   
label 1 = NECESSARY: it carries an essential logical step, key   
insight, correct setup, or computation that genuinely advances the   
derivation toward the answer.   
label 0 = UNNECESSARY: it is redundant, repetitive, filler, rambling   
self-talk, hesitation, backtracking that leads nowhere, restating   
known facts, or an ERRONEOUS step that does not help (or could   
mislead) the final correct solution.   
Judge each segment on its marginal contribution; a fluent trace can   
still contain many skippable segments.   
=== PROBLEM ===   
<question>   
=== CORRECT FINAL ANSWER ===   
<answer>   
=== REASONING SEGMENTS (indices 0..N-1) ===   
[SEGMENT 0]   
<segment 0 text>   
[SEGMENT 1]   
<segment 1 text>   
[SEGMENT N-1]   
<segment N-1 text>   
=== TASK ===   
Label all N segments (indices 0..N-1).   
OUTPUT REQUIREMENT: Return ONLY a single JSON object, no prose, no   
markdown fences, with EXACTLY two keys:   
"labels": an object mapping every segment index (as string) to 0 or   
1.   
"reasons": an object mapping every segment index (as string) to a   
concise (<=25 words) English reason.   
You MUST include an entry for every segment index shown.   
Example:   
{"labels": {"0": 1, "1": 0}, "reasons": {"0": "sets up the key   
substitution", "1": "restates prior step, redundant"}}

## D CASE STUDY: ORIGINAL PROBLEM AND FULL REASONING TRAJECTORY

This appendix provides the original problem and the complete long-form reasoning trajectory of LIMO #539 used in our case study. We preserve the original reasoning content verbatim and organize it according to the same reasoning-step segmentation used in our analysis.

## D.1 ORIGINAL PROBLEM

There is a set of 1000 switches, each of which has four positions, called \$A, B, C\$ , and   
\$D\$ . When the position of any switch changes, it is only from \$A\$ to \$B\$ , from \$B\$ to   
\$C\$ , from \$C\$ to \$D\$ , or from \$D\$ to \$A\$ . Initially each switch is in position \$A\$   
. The switches are labeled with the 1000 different integers \$(2ˆ{x})(3ˆ{y})(5ˆ{z})\$ ,   
where \$x, y\$ , and \$z\$ take on the values \$0, 1, \ldots, 9\$ . At step \$i\$ of a 1000-   
step process, the \$i\$ -th switch is advanced one step, and so are all the other switches   
whose labels divide the label on the \$i\$ -th switch. After step 1000 has been   
completed, how many switches will be in position \$A\$ ?

## Gold Answer. 650

## D.2 FULL REASONING TRAJECTORY

## Reasoning Step 0. Reasoning Step 0.

Okay, let's try to tackle this problem. So, we have 1000 switches, each labeled with a   
number of the form 2ˆx 3ˆy 5ˆz where x, y, z range from 0 to 9. Each switch starts   
in position A, and every time a switch is toggled, it cycles through A→B→C→D→A and   
so on.   
The process is a 1000-step process where in step i, we advance the i-th switch and all   
switches with labels that divide the label of the i-th switch. After completing all 1000   
steps, we need to find how many switches are back in position A.   
First, let's make sure I understand the setup correctly. Each switch is uniquely identified   
by (x, y, z) where each of x, y, z is between 0 and 9 inclusive. The label is 2ˆx <sub>\*</sub> 3ˆy   
<sub>\*</sub> 5ˆz. So, the number of switches is 10<sub>\*</sub>10<sub>\*</sub>10 = 1000, which checks out.   
In each step i, we toggle the i-th switch and all switches with labels that divide the i-th   
switch's label. So, for example, if the i-th switch has label 2ˆ3 <sub>\*</sub> 3ˆ2 <sub>\*</sub> 5ˆ1, then any   
switch with label 2ˆa 3ˆb 5ˆc where a ≤ 3, b ≤ 2, c ≤ 1 will be toggled. Each   
toggle moves the switch one position forward (A→B, B→C, etc.), and after four toggles,   
it cycles back to A.   
Therefore, the final position of a switch depends on how many times it was toggled modulo 4.   
If it was toggled 0 times mod 4, it's in A; 1 mod 4, it's in B; etc. So, switches that   
are toggled a multiple of 4 times will be back in A.   
Hence, our problem reduces to: For each switch, count the number of multiples of its label   
in the list of 1000 labels, then compute how many switches have this count ≡ 0 mod 4.

## Reasoning Step 1.

## Reasoning Step 2.

Wait, but actually, no. Because the count C\_j = (10 - a)(10 - b)(10 - c). We need C\_j ≡ 0   
mod 4. So, we need the product (10 - a)(10 - b)(10 - c) ≡ 0 mod 4.   
But to clarify, (10 - a)(10 - b)(10 - c) must be congruent to 0 modulo 4. So, the product   
must be divisible by 4.   
Therefore, for each triple (a, b, c), compute (10 - a)(10 - b)(10 - c) and check if it's   
divisible by 4. The total number of such triples is the answer.   
So, the problem becomes: How many triples (a, b, c) with a, b, c ∈ {0,1,...,9} are such that   
(10 - a)(10 - b)(10 - c) is divisible by 4?

## Reasoning Step 3.

Alternatively, since a, b, c ∈ {0,...,9}, 10 - a ∈ {1,...,10}, same for 10 - b and 10 - c.   
Therefore, let's let x = 10 - a, y = 10 - b, z = 10 - c, so x, y, z ∈ {1, 2, ..., 10}. Then,   
we need to count the number of triples (x, y, z) where x, y, z ∈ {1, ..., 10} and xyz   
≡ 0 mod 4.   
But the original problem counts each (a, b, c) once, so since x = 10 - a, a is determined by   
x: a = 10 - x. Similarly for y and z. Therefore, each (x, y, z) corresponds to exactly   
one (a, b, c). So, the number of such triples is the same as the number of (x, y, z)   
with x, y, z ∈ {1, ..., 10} and xyz divisible by 4.   
Therefore, our problem is equivalent to: How many triples (x, y, z) with x, y, z ∈ {1, 2,   
..., 10} have xyz divisible by 4?   
So, the answer will be the number of such triples, which is the same as the number of   
switches that are toggled a multiple of 4 times, hence ending up back in position A.   
So, the problem simplifies to: How many triples (x, y, z) with each component from 1 to 10   
inclusive have their product divisible by 4?   
To solve this, perhaps it's easier to compute the total number of triples (10<sub>\*</sub>10<sub>\*</sub>10 = 1000)   
and subtract the number of triples where the product is not divisible by 4. So, total -   
number of triples with product ≡ 1, 2, or 3 mod 4. But since 4 divides the product if   
and only if the product has at least two factors of 2. So, equivalently, the product is   
divisible by 4 iff the product has at least two 2s in its prime factorization.   
Therefore, we can think in terms of the number of triples (x, y, z) where the exponents of 2   
in x, y, z sum to at least 2. But perhaps another approach is better.

## Reasoning Step 4.

Alternatively, note that in the numbers 1 to 10, the even numbers are 2, 4, 6, 8, 10. The   
odd numbers are 1, 3, 5, 7, 9. Let's categorize each number as:   
Odd: 1, 3, 5, 7, 9 (numbers not divisible by 2)   
- Even but not divisible by 4: 2, 6, 10 (numbers divisible by 2 but not by 4)   
- Divisible by 4: 4, 8 (numbers divisible by 4)   
Let me check:   
- 1: odd   
- 2: even, not divisible by 4   
- 3: odd   
- 4: divisible by 4   
- 5: odd   
- 6: even, not divisible by 4   
- 7: odd   
- 8: divisible by 4   
- 9: odd   
- 10: even, not divisible by 4   
So, counts:   
- Odd numbers (O): 5   
- Even but not divisible by 4 (E): 3 (numbers 2,6,10)   
- Divisible by 4 (F): 2 (numbers 4,8)   
Therefore, for each x, y, z, they can be O, E, or F.

We need to compute the number of triples (x, y, z) where the product xyz is divisible by 4.   
Let's think in terms of the contributions from each component.   
The product xyz will have a certain number of factors of 2. For the product to be divisible   
by 4, the total number of factors of 2 must be at least 2.   
Each O contributes 0 factors of 2.   
Each E contributes 1 factor of 2 (since they are divisible by 2 but not 4).   
Each F contributes 2 factors of 2 (since they are divisible by 4).   
Therefore, the total number of factors of 2 in the product is:   
If a component is O: 0   
If E: 1   
If F: 2   
So, for each triple (X, Y, Z), where each is O, E, or F, the total number of factors of 2 is   
:   
Total = factors from X + factors from Y + factors from Z.   
We need Total ≥ 2.   
Therefore, the number of triples where Total ≥ 2 is equal to the total number of triples   
minus the number of triples where Total < 2, which is Total = 0 or 1.   
Total = 0: all three components are O. So, O, O, O.   
Total = 1: exactly one component is E or F contributing 1 or 2 factors, but total factors   
sum to 1. But wait, if a component is F, it contributes 2 factors, which would already   
make Total ≥ 2. So, to get Total = 1, we need exactly one of the components to be E (   
contributing 1 factor) and the rest to be O. Because if any component is F, it would add   
2, which would overshoot.   
Therefore, Total = 1 is only possible when exactly one component is E and the other two are   
O.   
Therefore, the number of triples where Total < 2 is:   
Number of O,O,O triples + number of triples with exactly one E and two O's.   
Compute these:   
Number of O,O,O triples: since there are 5 O's in each position. So, 5<sub>\*</sub>5<sub>\*</sub>5 = 125.   
Number of triples with exactly one E and two O's: For each position (x, y, z), we can have E   
in x, O in y and z; E in y, O in x and z; or E in z, O in x and y. For each case, the   
number is (number of E choices) <sub>\*</sub> (number of O choices)ˆ2. There are 3 E's, 5 O's. So,   
for each single position being E, it's 3 5 5. Since there are 3 positions, total is   
3 3 5 5 = 225.   
Therefore, total number of triples with Total < 2 is 125 + 225 = 350.   
Therefore, the number of triples with Total ≥ 2 is total triples - 350 = 1000 - 350 = 650.

## Reasoning Step 5.

Wait, but hold on. Wait, total triples are 10 10 10 = 1000. Wait, no, in our transformed   
problem, x, y, z are from 1 to 10, which corresponds to the original 1000 switches.   
Therefore, the total number of triples is indeed 1000. So, 1000 - 350 = 650. Therefore,   
650 switches are toggled a number of times divisible by 4, hence ending in position A.

Reasoning Step 6.   
But wait, hold on. Wait, no. Wait, in the transformed problem, we have x, y, z each from 1   
to 10, corresponding to (10 - a), (10 - b), (10 - c). So, the count of triples (x, y, z)   
where xyz is divisible by 4 is 650, which would mean 650 switches are toggled a number   
of times ≡ 0 mod 4, hence in position A.   
But the answer is 650? Let me verify.

## Reasoning Step 7.

Wait, let's check if the logic is correct.   
First, in the transformed variables, we have x, y, z from 1 to 10, and we need xyz divisible   
by 4. So, we categorize numbers 1-10 into O, E, F as before.   
Total triples: 1000.   
Number of bad triples (Total < 2 factors of 2):   
- All O: 5 5 5 = 125   
Exactly one E and two O's: 3 positions 3 E numbers 5 O 5 O = 3 3 25 = 225   
Total bad: 125 + 225 = 350   
Thus, good triples: 1000 - 350 = 650   
Therefore, the answer is 650.   
But let me think again. Wait, but hold on. When we have a component that is F (divisible by   
4), it contributes 2 factors of 2. So, if there's at least one F in the triple, then the   
total factors of 2 is at least 2. Similarly, if there are two E's, since each E   
contributes 1 factor of 2, two E's would give 2 factors. Similarly, one E and one F   
would give 3 factors, etc.   
Therefore, the number of triples with product divisible by 4 is:   
- All triples with at least one F: Let's compute this.   
Number of triples with at least one F: total triples - triples with no F.   
Triples with no F: all components are either O or E. There are (5 + 3) = 8 choices per   
component. So, 8 8 8 = 512. But within these, some may still have products divisible by   
4. Wait, no. Wait, if there are no Fs, then all components are either O or E. So, the   
factors of 2 come only from E's, which contribute 1 each. So, to get the product   
divisible by 4, we need at least two E's. So, in the absence of F's, the number of   
triples with product divisible by 4 is the number of triples with at least two E's.

## Reasoning Step 8.

Wait, this seems conflicting with the previous approach. Let me check.

## Reasoning Step 9.

Alternatively, maybe I should use inclusion-exclusion properly.   
Total number of triples with product divisible by 4 = number of triples with at least two   
factors of 2.   
Which can happen in two distinct cases:   
1. At least one F (each F ives two factors), so even if the rest are O, the total factors   
would be at least 2.   
2. No F's, but at least two E's (each E gives one factor, so two E's give two factors).   
Therefore, total number of good triples = (number of triples with at least one F) + (number   
of triples with no F's and at least two E's).   
Compute each part.   
First, number of triples with at least one F: total triples - triples with no F's. As before   
, triples with no F's: each component can be O or E, 8 choices per component. So 8ˆ3 =   
512. Therefore, triples with at least one F = 1000 - 512 = 488.   
Second, number of triples with no F's and at least two E's: Within the 512 triples with no F   
's, how many have at least two E's.   
In triples with no F's, each component is O or E.   
Number of such triples with at least two E's = total triples with no F's - triples with   
fewer than two E's.   
Which is 512 - [triples with zero E's + triples with one E].

Number of triples with zero E's (all O's): 5ˆ3 = 125.   
Number of triples with one E: 3 positions 3 E's 5ˆ2 O's = 3 3 25 = 225.   
Therefore, triples with at least two E's and no F's: 512 - 125 - 225 = 512 - 350 = 162.   
Therefore, total good triples = 488 (with at least one F) + 162 (with no F's but two E's) =   
650.   
Same as before. Therefore, 650 is the correct count. Therefore, the answer is 650.

## Reasoning Step 10.

But wait, let me cross-verify with another method.   
Alternative approach: For each triple (x, y, z), the product xyz is divisible by 4 iff:   
Either:   
- At least two of x, y, z are even, or   
- At least one of x, y, z is divisible by 4.

## Reasoning Step 11.

Wait, actually, not exactly. Let's think:   
A number is divisible by 4 if:   
- It has at least two factors of 2. So, in the product xyz, the total number of factors of 2   
must be at least two.   
Each even number contributes at least one factor of 2, and numbers divisible by 4 contribute   
two.   
So, the product's number of factors of 2 is the sum over each component's factors of 2.   
Therefore, possible scenarios:   
1. At least one component is divisible by 4 (i.e., contributes 2 factors), and the rest can   
be anything. Then, total factors is at least 2.   
2. No components are divisible by 4, but at least two components are even (i.e., each   
contributes 1 factor), totaling 2.   
Therefore, total good triples are:   
Number of triples with at least one divisible by 4 + number of triples with no divisible by   
4 but at least two even.   
Compute each:   
First, number of triples with at least one divisible by 4: as before, 1000 - 512 = 488.   
Second, number of triples with no divisible by 4 but at least two even: In this case, all   
components are either O or E (not F). So, components can be O or E.   
Number of such triples: 8ˆ3 = 512 (as before).   
Within these, number of triples with at least two E's: which is 162, as before.   
Therefore, total is 488 + 162 = 650. Same result.

## Reasoning Step 12.

Alternatively, perhaps even another way. Let's compute the number of triples where product is not divisible by 4, i.e., product is 1 or 2 mod 4.

Product is 2 mod 4: exactly one component is 2 mod 4 and the rest are odd. But wait, in our   
numbers, E is numbers divisible by 2 but not 4, i.e., 2 mod 4 or 6 mod 4 (which is 2 mod   
4) or 10 mod 4 (which is 2 mod 4). So, E numbers are 2, 6, 10, which are all 2 mod 4.   
Therefore, if exactly one component is E (2 mod 4) and the rest are O (1 mod 4), then the   
product is 2 1 1 = 2 mod 4. So, number of such triples is 3 positions 3 E's 5 O's   
5 O's = 3<sub>\*</sub>3<sub>\*</sub>5<sub>\*</sub>5 = 225.   
Therefore, total bad triples: 125 + 225 = 350. Hence, good triples: 1000 - 350 = 650. Same   
answer.   
Therefore, regardless of the approach, we get 650. So, the answer should be 650.   
But let me verify with an example.   
Take a smaller case. Suppose we have numbers from 1 to 2 instead of 1 to 10, with similar   
rules. Let's see if the formula works.   
But maybe that's overcomplicating. Alternatively, check if the counts make sense.   
Given that 5 numbers are odd, 3 are E (2 mod 4), and 2 are F (0 mod 4). So, total 10 numbers   
Compute the number of triples (x, y, z) with product divisible by 4.   
Using the formula, total triples 1000, bad triples 350, good triples 650. Hmm. Alternatively   
, perhaps we can compute the exact number.

## Reasoning Step 13.

Alternatively, think of the multiplicative contributions.   
For each position x, y, z:   
- The probability that x is O is 5/10 = 1/2.   
- The probability that x is E is 3/10.   
- The probability that x is F is 2/10 = 1/5.   
But since we are counting exact numbers, not probabilities, but maybe using generating   
functions.   
But perhaps overkill. Since all methods so far point to 650, and the logic seems consistent,   
I think 650 is the correct answer.   
But to be thorough, let's check a specific case.   
Take a switch with label 1 (i.e., x=y=z=0). Then, the number of multiples is (10 - 0)<sub>\*</sub>(10 -   
0) (10 - 0) = 1000. So, 1000 toggles. 1000 mod 4 is 0, so this switch would be in A. But   
according to our count, does (x, y, z) = (10, 10, 10) correspond to (a, b, c) = (0, 0,   
0). Then, the product xyz = 10<sub>\*</sub>10<sub>\*</sub>10 = 1000. 1000 mod 4 is 0, so it's counted as a good   
triple. Therefore, this switch is included in the 650.   
Similarly, take a switch with label 2ˆ9 3ˆ9 5ˆ9. Its number of multiples is (10 - 9) (10   
- 9)<sub>\*</sub>(10 - 9) = 1<sub>\*</sub>1<sub>\*</sub>1 = 1. So, 1 toggle. 1 mod 4 is 1, so it's in position B. Therefore   
, not in A. So, this switch is excluded from the 650, which is correct.

## Reasoning Step 14.

Another test case: take a switch with label 2ˆ8 (i.e., x=8, y=0, z=0). Then, the number of multiples is (10 - 8)<sub>\*</sub>(10 - 0)<sub>\*</sub>(10 - 0) = 2<sub>\*</sub>10<sub>\*</sub>10 = 200. 200 mod 4 is 0, so this switch would be in A. Indeed, in our transformed variables, x = 10 - a = 2, y = 10 - b = 10, z = 10 - c = 10. The product xyz = 2<sub>\*</sub>10<sub>\*</sub>10 = 200. 200 is divisible by 4, so this triple is counted in the 650.

## Reasoning Step 15.

Another test case: take a switch with label 2ˆ1 <sub>\*</sub> 3ˆ0 <sub>\*</sub> 5ˆ0 (i.e., x=1, y=0, z=0). Then, the   
number of multiples is (10 - 1)<sub>\*</sub>(10 - 0)<sub>\*</sub>(10 - 0) = 9<sub>\*</sub>10<sub>\*</sub>10 = 900. 900 mod 4 is 900 / 4   
= 225, remainder 0. So, 900 mod 4 = 0. Therefore, this switch is in A. In transformed   
variables, x = 9, y = 10, z = 10. The product is 9 10 10 = 900, which is divisible by 4.   
Wait, 9 is odd, 10 is even. 9<sub>\*</sub>10<sub>\*</sub>10 = 9<sub>\*</sub>100 = 900. 900 / 4 = 225. Yes, divisible by 4.   
So, even though x is odd (9), the product is still divisible by 4 because y and z are

both even. So, this is accounted for in our count. Since y and z are 10, which are E numbers (divisible by 2 but not 4). So, each contributes 1 factor of 2. Therefore, total factors of 2 is 2, so product is divisible by 4. Hence, included in the 650.

Therefore, all test cases check out.

Therefore, I'm confident that the answer is 650.

Final Answer \boxed{650}