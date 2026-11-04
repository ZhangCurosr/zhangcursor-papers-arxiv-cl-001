# SELF-PRIVILEGED FINE-TUNING

Yangrong Hu<sup>1,\*</sup>, Xueyu Zhou<sup>1,\*</sup>, Yue Wu<sup>1</sup>, Jian Huang<sup>1,2,†</sup>

<sup>1</sup>Department of Data Science and Artificial Intelligence

<sup>2</sup>Department of Applied Mathematics

The Hong Kong Polytechnic University

{yangrong.hu,xueyu.zhou,yue0301.wu}@connect.polyu.hk

j.huang@polyu.edu.hk

## ABSTRACT

Supervised fine-tuning (SFT) is a key component of language model post-training, enabling effective task adaptation and serving as an important foundation for subsequent reinforcement learning. Recent methods seek to improve SFT by introducing token-level weights into the training objective, where these weights are typically derived from a model’s output distribution at each token position. To better understand how such weighting schemes affect SFT, we analyze their optimization objectives and theoretically characterize the target distributions they induce. Our analysis shows that dynamic weighting methods, such as Dynamic Fine-Tuning (DFT), induce a heavily sharpened distribution, whereas static weighting methods fit a target distribution proportional to the product of the data distribution and a frozen prior. These findings suggest that static weighting provides a more stable training target, while its effectiveness critically depends on the quality of the prior distribution. Building on this observation, we propose Self-Privileged Fine-Tuning (SPFT), which uses a frozen copy of the initial model conditioned on privileged information (such as ground truth for mathematical questions) as a prior to provide guidance for token-level weighting during fine-tuning. Through a Bayesian decomposition, we further show that SPFT reweights the standard static prior according to how informative each target token is for predicting the groundtruth outcome, thereby assigning larger weights to more predictive tokens. Experiments on mathematical reasoning and code generation show that SPFT improves over standard SFT and other baselines across the main settings.

## 1 INTRODUCTION

Supervised fine-tuning (SFT) is a fundamental component of language-model post-training and commonly serves as the foundation for subsequent reinforcement learning (RL) (Ouyang et al., 2022; Touvron et al., 2023). Whereas RL learns from reward signals assigned to model-generated responses, SFT directly learns from supervised datasets by maximizing the likelihood of target responses, which enables pretrained models to better adapt to downstream tasks such as instruction following, mathematical reasoning and code generation.

As standard SFT assigns equal weight to all target tokens, recent studies have explored improving SFT by introducing non-uniform token weights. Wu et al. (2026) reweight each token according to its probability under the current model, giving larger updates to tokens that are better supported by the model. Building on this formulation, Zhu et al. (2026) introduce an anchoring term based on KL divergence to stabilize optimization, while Lin et al. (2026) introduce temperature scaling to control the magnitude of token weights. Despite their different designs, these methods derive weights from the model being optimized, causing the weighting signal to evolve dynamically throughout training. More recently, Wang et al. (2026) take a different approach by computing token weights from a frozen reference model, resulting in a static weighting scheme. While these methods demonstrate the effectiveness of token reweighting, the mechanisms underlying their improvements remain insufficiently understood.

To better understand these methods, we analyze their loss functions and induced optimization objectives in Section 3. For dynamic weighting methods such as Dynamic Fine-Tuning (DFT) (Wu et al., 2026), the target distribution evolves with the model throughout training, leading to progressive sharpening and increasing concentration on high-probability modes. In contrast, static weighting such as PriFT (Wang et al., 2026) induces a fixed target distribution proportional to the product of the data distribution and the prior distribution. This characterization makes the quality of the prior central to static token weighting.

Motivated by this observation, we ask whether the model itself can provide a more informative prior for static weighting without relying on an external teacher. To this end, we propose Self-Privileged Fine-Tuning (SPFT). SPFT uses a frozen copy of the initial model conditioned on privileged information as a prior to guide token-level weighting during fine-tuning. A Bayesian decomposition shows that the resulting privileged prior modifies the standard static prior according to the predictive contribution of each target token to the ground-truth outcome, so that more informative tokens receive larger weights. Experiments on mathematical reasoning and code generation show that SPFT consistently outperforms standard SFT and other token-weighting baselines across the main settings.

Our contributions are:

• We provide a theoretical characterization of token-weighted supervised fine-tuning, showing that dynamic weighting progressively sharpens the model distribution, while static weighting optimizes toward a fixed distribution jointly determined by the data distribution and a frozen prior.

• We propose Self-Privileged Fine-Tuning (SPFT), which constructs a more informative static prior by conditioning a frozen copy of the initial model on privileged information available during training. A Bayesian decomposition further shows that SPFT upweights tokens according to their contribution to predicting the ground-truth outcome.

• We evaluate SPFT on mathematical reasoning and code generation, where it consistently improves over standard SFT and existing token-weighting baselines. Further ablation studies validate the importance of a fixed privileged prior and examine other key design choices in SPFT.

## 2 RELATED WORK

Token-Reweighted Supervised Fine-Tuning. SFT remains a stable and data-efficient posttraining paradigm, but its generalization can lag behind reinforcement learning in some settings (Chu et al., 2025; Shenfeld et al., 2025). RL methods such as PPO and verifiable-reward reasoning training optimize the model through on-policy rollouts and rewards, yet online sampling introduces substantial computational cost (Schulman et al., 2017; Shao et al., 2024; Guo et al., 2025; Sheng et al., 2025; Swamy et al., 2025). In contrast, offline preference optimization methods such as DPO avoid explicit online rollouts (Rafailov et al., 2023). Recent studies introduce token-level reweighting as an offline bridge between SFT and RL (Wu et al., 2026; Zhu et al., 2026). Existing methods can be broadly divided into online dynamic weighting and anchored or prior-guided static weighting. DFT (Wu et al., 2026), temperature-based variants (Lin et al., 2026), and related uncertainty or confidence-based schemes use statistics from the evolving model during training, which can improve performance on difficult reasoning examples. Anchored SFT (Zhu et al., 2026) and PriFT (Wang et al., 2026) improve stability through reference anchoring or a frozen pretrained prior, but their signals remain unconditioned on the task outcome. SPFT instead conditions the frozen reference model on the verified outcome, producing a goal-directed process signal while remaining offline and independent of the evolving model policy.

Privileged Information and Knowledge Distillation. Knowledge distillation traditionally matches a teacher’s soft predictions on fixed training prefixes (Hinton et al., 2015). Autoregressive methods such as MiniLLM (Gu et al., 2024) and Generalized Knowledge Distillation (GKD) (Agarwal et al., 2024) address the mismatch between fixed prefixes and student-generated trajectories through on-policy sampling, but require online rollouts and, in GKD, an external teacher. On-Policy Self-Distillation (OPSD) (Zhao et al., 2026) removes the external-teacher requirement by conditioning on a reference solution as privileged information. SPFT adopts this idea offline, using verified outcomes to construct token weights for SFT without on-policy sampling or full-vocabulary distribution matching.

## 3 THEORETICAL ANALYSIS OF TOKEN-WEIGHTED FINE-TUNING

In this section, we analyze the optimization behavior induced by different token-weighting strategies. We first examine dynamic weighting and characterize how its induced target distribution evolves during training. We then analyze static weighting and show that it instead defines a fixed target distribution determined by the data distribution and the prior.

Notation Let x denote the prompt and $y = ( y _ { 1 } , y _ { 2 } , . . . , y _ { T } )$ denote the corresponding output. We denote the underlying data distribution by $p _ { \mathrm { d a t a } } ( x , y )$ . Let $p _ { \theta _ { 0 } } ( y \mid x )$ denote the initial model’s distribution and $p _ { \theta _ { k } } ( y \mid x )$ the distribution at optimization step k.

## 3.1 DYNAMIC WEIGHTING LEADS TO PROGRESSIVE DISTRIBUTION SHARPENING

We first analyze dynamic weighting through DFT (Wu et al., 2026), where the training weight is determined by the current model and therefore evolves throughout training. The objective of DFT is given by

$$
\mathcal { L } _ { \mathrm { D F T } } ( \theta ) = - \mathbb { E } _ { ( x , y ) \sim p _ { \mathrm { d a t a } } } \left[ \sum _ { t } \operatorname { s g } [ p _ { \theta } ( y _ { t } \mid x , y _ { < t } ) ] \log p _ { \theta } ( y _ { t } \mid x , y _ { < t } ) \right] ,\tag{1}
$$

where $\mathrm { s g } [ \cdot ]$ denotes the stop-gradient operator.

At the k-th optimization step, the weighting term is evaluated using the current model $p _ { \theta _ { k } }$ and treated as fixed during this step. Accordingly, the objective at the k-th step can be written as

$$
\mathcal { L } _ { \mathrm { D F T } } ^ { ( k ) } ( \theta ) = - \mathbb { E } _ { ( x , y ) \sim p _ { \mathrm { d a t a } } } \left[ \sum _ { t } \operatorname { s g } [ p _ { \theta _ { k } } ( y _ { t } \mid x , y _ { < t } ) ] \log p _ { \theta } ( y _ { t } \mid x , y _ { < t } ) \right] .\tag{2}
$$

For a given token context $( x , y _ { < t } )$ , the optimal solution to the objective in Eq. 2 satisfies

$$
p _ { \theta _ { k + 1 } } ( y _ { t } \mid x , y _ { < t } ) \propto p _ { \mathrm { d a t a } } ( y _ { t } \mid x , y _ { < t } ) p _ { \theta _ { k } } ( y _ { t } \mid x , y _ { < t } ) .\tag{3}
$$

As optimization proceeds, the target distribution at each token context depends on the model obtained from the preceding step, causing the optimization target to evolve continuously throughout training. When each optimization step reaches the corresponding optimum, repeatedly applying Eq. 3 gives

$$
p _ { \theta _ { k + 1 } } ( y _ { t } \mid x , y _ { < t } ) \propto p _ { \theta _ { 0 } } ( y _ { t } \mid x , y _ { < t } ) p _ { \mathrm { d a t a } } ( y _ { t } \mid x , y _ { < t } ) ^ { k + 1 } .\tag{4}
$$

For two candidate tokens $v _ { 1 }$ and $v _ { 2 }$ under the same context $( x , y _ { < t } )$ , the probability ratio satisfies

$$
\frac { p _ { \theta _ { k + 1 } } ( v _ { 1 } \mid x , y _ { < t } ) } { p _ { \theta _ { k + 1 } } ( v _ { 2 } \mid x , y _ { < t } ) } = \frac { p _ { \theta _ { 0 } } ( v _ { 1 } \mid x , y _ { < t } ) } { p _ { \theta _ { 0 } } ( v _ { 2 } \mid x , y _ { < t } ) } \left( \frac { p _ { \mathrm { d a t a } } ( v _ { 1 } \mid x , y _ { < t } ) } { p _ { \mathrm { d a t a } } ( v _ { 2 } \mid x , y _ { < t } ) } \right) ^ { k + 1 } .\tag{5}
$$

Therefore, whenever $p _ { \mathrm { d a t a } } ( v _ { 1 } \mid x , y _ { < t } ) > p _ { \mathrm { d a t a } } ( v _ { 2 } \mid x , y _ { < t } )$ , the relative probability assigned to v<sub>1</sub> grows exponentially with the number of optimization steps. As a result, tokens with higher conditional probability under the data distribution are progressively amplified relative to lower-probability tokens, causing the next-token distribution to become increasingly sharp throughout training.

Proposition 1 (Limiting Distribution of DFT). Assume that the maximizer is unique, and let $v ^ { * } =$ arg ma $\mathrm { x } _ { v } p _ { \mathrm { d a t a } } ( v \mid x , y _ { < t } )$ . Also assume that the initial model assigns nonzero probability to this mode, $i . e . , p _ { \theta _ { 0 } } ( v ^ { * } \mid x , y _ { < t } ) > 0$ . Then,

$$
\operatorname* { l i m } _ { k  \infty } p _ { \theta _ { k } } ( v \mid x , y _ { < t } ) = \mathbf { 1 } [ v = v ^ { * } ] .\tag{6}
$$

Proposition 1 shows that DFT progressively sharpens the next-token distribution and eventually concentrates all probability mass on the unique mode of the data distribution. This analysis also suggests that similar sharpening behavior may arise in other dynamic weighting methods whose evolving weights progressively emphasize high-probability regions of the data distribution.

![](images/6b0303495a572df78a62f29e2c6821938a5f3465bc9c813dc90c3c5e8d6fe93d.jpg)

![](images/dcdcbdaa30f694cc85219ccc1670d867c88a137aee5848bddd1abcbf78542bcf.jpg)

![](images/0c4b792173dc3b27189b9b3a969ec44fa4c09d2bcaafec674a710aae2d0cfdab.jpg)  
Figure 1: Next-token distribution evolution of SFT, DFT, and SPFT at four training steps.

![](images/509e3dd2932af24853fbfb7ccf139432d22a88e6c13b5ca92ecb9346e8f65490.jpg)

Consistent with this analysis, Figure 1 tracks the next-token distribution at four training steps and provides an illustration of the sharpening behavior. While SFT remains comparatively diffuse, DFT rapidly concentrates nearly all probability mass on the top-ranked token, whereas SPFT further sharpens the distribution relative to SFT, but in a selective manner: tokens receiving stronger support from the privileged prior are amplified more substantially, while alternative tokens retain non-trivial probability mass.

## 3.2 STATIC WEIGHTING INDUCES A FIXED PRODUCT DISTRIBUTION

We next consider static weighting methods, where the token weights are determined by a fixed reference model and remain unchanged throughout training. Let $p _ { \mathrm { p r i o r } }$ denote the frozen reference distribution. The corresponding weighted fine-tuning objective can be written as

$$
\mathcal { L } _ { \mathrm { s t a t i c } } ( \theta ) = - \mathbb { E } _ { ( x , y ) \sim p _ { \mathrm { d a t a } } } \left[ \sum _ { t } p _ { \mathrm { p r i o r } } ( y _ { t } \mid x , y _ { < t } ) \log p _ { \theta } ( y _ { t } \mid x , y _ { < t } ) \right] .\tag{7}
$$

Proposition 2 (Optimal Distribution of Static Weighting). For a fixed token context $( x , y _ { < t } )$ , the optimal solution to the objective in $E q . ~ 7$ satisfies

$$
p _ { \theta } ^ { * } ( v \mid x , y _ { < t } ) \propto p _ { \mathrm { d a t a } } ( v \mid x , y _ { < t } ) p _ { \mathrm { p r i o r } } ( v \mid x , y _ { < t } ) .\tag{8}
$$

Proposition 2 shows that static weighting defines a fixed target distribution determined jointly by the data distribution and the reference distribution. In particular, for any two candidate tokens $v _ { 1 }$ and $v _ { 2 }$ under the same context,

$$
\frac { p _ { \theta } ^ { * } ( v _ { 1 } \mid x , y _ { < t } ) } { p _ { \theta } ^ { * } ( v _ { 2 } \mid x , y _ { < t } ) } = \frac { p _ { \mathrm { d a t a } } ( v _ { 1 } \mid x , y _ { < t } ) } { p _ { \mathrm { d a t a } } ( v _ { 2 } \mid x , y _ { < t } ) } \frac { p _ { \mathrm { p r i o r } } ( v _ { 1 } \mid x , y _ { < t } ) } { p _ { \mathrm { p r i o r } } ( v _ { 2 } \mid x , y _ { < t } ) } .\tag{9}
$$

Thus, the relative preference between tokens is determined by both their support under the data distribution and their probability under the prior distribution. Since $p _ { \mathrm { p r i o r } }$ remains unchanged during training, the induced target distribution is fixed throughout optimization, avoiding the continuously evolving target induced by dynamic weighting.

This characterization makes the quality of the prior distribution a central design choice for static weighting. An informative prior can steer optimization toward more useful target tokens, whereas an inaccurate prior may distort the induced target distribution. For instance, PriFT (Wang et al., 2026) uses the frozen initial model as the prior, yielding a stable but outcome-agnostic weighting signal. This observation motivates the design of a more informative prior that incorporates additional task-relevant information while remaining fixed throughout training.

## 4 SELF-PRIVILEGED FINE-TUNING

Building on the theoretical analysis in Section 3, we present the formulation of Self-Privileged Fine-Tuning (SPFT). We first describe how privileged conditioning provides an informative reference prior and derive the token-weighted training objective. We then present a detailed implementation of SPFT in Section 4.2 and Section 4.3.

## 4.1 METHOD FORMULATION AND OBJECTIVE FUNCTION

As established in Proposition 2, static token-weighted fine-tuning induces a target distribution proportional to the product of the data distribution and a fixed prior:

$$
p _ { \theta } ^ { * } ( v \mid x , y _ { < t } ) \propto p _ { \mathrm { d a t a } } ( v \mid x , y _ { < t } ) p _ { \mathrm { p r i o r } } ( v \mid x , y _ { < t } ) .\tag{10}
$$

Prior work such as PriFT (Wang et al., 2026) instantiates this prior using the frozen initial model, $p _ { \mathrm { p r i o r } } ( y _ { t } \mid x , y _ { < t } ) = p _ { \theta _ { 0 } } ( y _ { t } \mid x , y _ { < t } )$ . While this provides a stable and readily available prior, it may provide limited information about how strongly an intermediate token $y _ { t }$ contributes to the eventual ground-truth outcome, since it is conditioned only on the problem prompt x and the preceding tokens y<sub><t</sub>.

Drawing inspiration from On-Policy Self-Distillation (OPSD) (Zhao et al., 2026), we incorporate privileged information available during training into the conditioning context of the frozen initial model. Given privileged information c associated with the ground-truth outcome, this yields a prior conditioned on privileged information:

$$
q _ { t } \triangleq p _ { \theta _ { 0 } } ( y _ { t } \mid x , c , y _ { < t } ) .\tag{11}
$$

Proposition 3 (Privileged Prior as Outcome-Reweighted Prior). Assume that the relevant conditional probabilities are nonzero and let $q _ { t } = p _ { \theta _ { 0 } } ( y _ { t } \mid x , c , y _ { < t } )$ denote the prior conditioned on privileged information c. Then, by Bayes’ rule,

$$
q _ { t } = \frac { p _ { \theta _ { 0 } } ( c \mid x , y _ { \le t } ) } { p _ { \theta _ { 0 } } ( c \mid x , y _ { < t } ) } p _ { \theta _ { 0 } } ( y _ { t } \mid x , y _ { < t } ) .\tag{12}
$$

Proposition 3 shows that privileged conditioning refines the base-model prior by multiplying it with an outcome-likelihood ratio. Conditioned on the preceding prefix, this ratio measures the incremental contribution of the current token $y _ { t }$ to the ground-truth outcome c. A ratio greater than one indicates that $y _ { t }$ contributes positively to deriving the ground-truth outcome, thereby increasing its privileged weight, whereas a ratio below one indicates a weaker contribution toward the outcome and thus reduces its weight.

Using the privileged prior $q _ { t }$ as the token weight, SPFT optimizes the following objective

$$
\mathcal { L } _ { \mathrm { S P F T } } ( \theta ) = - \mathbb { E } _ { ( x , y , c ) \sim p _ { \mathrm { d a t a } } } \left[ \sum _ { t } q _ { t } \log p _ { \theta } ( y _ { t } \mid x , y _ { < t } ) \right] .\tag{13}
$$

The privileged information c is used only to determine the token weights $q _ { t }$ , while the model being optimized remains conditioned on the original prompt x and prefix $y _ { < t }$ . Thus, privileged information affects training through the weighting signal rather than through the model’s forward input.

Following Proposition 2, the optimal distribution induced by SPFT satisfies

$$
p _ { \theta , \mathrm { S P F T } } ^ { * } ( v \mid x , y _ { < t } ) \propto p _ { \mathrm { d a t a } } ( v \mid x , y _ { < t } ) p _ { \theta _ { 0 } } ( v \mid x , c , y _ { < t } ) .\tag{14}
$$

Using the Bayesian decomposition, this target can be further written as

$$
p _ { \theta , \mathrm { S P F T } } ^ { * } ( v \mid x , y _ { < t } ) \propto p _ { \mathrm { d a t a } } ( v \mid x , y _ { < t } ) p _ { \theta _ { 0 } } ( v \mid x , y _ { < t } ) \frac { p _ { \theta _ { 0 } } ( c \mid x , y _ { < t } , v ) } { p _ { \theta _ { 0 } } ( c \mid x , y _ { < t } ) } .\tag{15}
$$

Compared with PriFT, SPFT therefore retains the original base-model prior while further modulating it according to the contribution of each candidate token to the privileged outcome. Tokens that make the ground-truth outcome more likely receive additional emphasis, whereas tokens that contribute less receive relatively smaller weights. In this way, SPFT preserves the fixed-target property of static weighting while making the induced target distribution more outcome-aware.

To quantify how different weighting schemes affect the learned policy distribution, Figure 2 tracks top-1 confidence, token-level entropy, and KL divergence from the initial base model during training. DFT rapidly increases top-1 confidence while sharply reducing entropy, indicating strong concentration on a narrow set of continuations. Its substantially larger KL divergence further shows that the resulting policy moves farther away from the base model distribution. By contrast, SFT maintains a relatively flat token distribution, with higher entropy and lower top-1 confidence throughout training. This suggests that uniform token weighting provides limited selective emphasis among candidate continuations and induces comparatively mild distributional change. SPFT exhibits an intermediate behavior: it increases confidence and reduces entropy relative to SFT, indicating stronger emphasis on selected tokens, while maintaining a much smaller degree of concentration and distri butional drift than DFT. This is consistent with SPFT selectively amplifying tokens supported by the privileged prior rather than uniformly driving the model toward an increasingly sharp distribution.

![](images/d6d2d347ae31ad931b102108af8de4efb6b863a51b7e621d4e5250c503e136b3.jpg)

![](images/71799e98d5900127420a0f3f823e60f422cb8d973247bef2263f57d7fb232648.jpg)

![](images/0cc8a78eb8b06d449fb76bbc21e0a56e347745c3f42cdbc0d1ee347e05a6db05.jpg)  
Figure 2: Distributional evolution of SFT, DFT, and SPFT during training, measured by token-level entropy, top-1 confidence, and KL divergence from the initial model. DFT rapidly sharpens the output distribution and moves farthest from the base model, while SPFT produces a more controlled increase in confidence and reduction in entropy with substantially smaller distributional drift.

## 4.2 PRIVILEGED PROMPT CONSTRUCTION ACROSS DOMAINS

Given a training instance $( x , y , c )$ , the trainable model receives only the standard task prompt and prefix $h _ { t } = ( x , y _ { < t } )$ , whereas the frozen reference model evaluates the sequence under the privileged context $h _ { t } ^ { \mathrm { P } } = \left( x , c , y _ { < t } \right) $ , incorporating the privileged information c within a domain-specific instruction template.

The form of privileged information c differs across reasoning domains. In mathematical problem solving, c is intuitively instantiated as the ground-truth numerical answer extracted from boxed solutions, guiding the reference model to assess whether intermediate derivations align with the target result. In program synthesis, however, code tasks lack an obvious concise numerical answer. Simply providing the entire code reference implementation as c causes severe code token leakage. To provide semantic guidance without leaking surface syntax, we extract high-level conceptual guid ance via lightweight static Abstract Syntax Tree (AST) parsing and pattern matching, which identifies overarching algorithmic paradigms such as dynamic programming, breadth-first search, or two pointers, detects key abstract data structures such as heaps, deques, or state arrays, and isolates boundary invariants such as early return conditions. This clean-rule abstraction conveys algorithmic intent without exposing implementation details, enabling the privileged prior to reward critical algorithmic choices while maintaining generalization. The full conditioning interfaces are provided in Appendix B.

## 4.3 SEQUENCE-RELATIVE WEIGHTING (SPFT-SR)

Absolute privileged likelihoods $q _ { t }$ can vary with overall problem difficulty, obscuring which tokens are relatively well supported within a response. SPFT-SR therefore centers the privileged prior probabilities $q _ { t }$ over the T valid response tokens and retains only positive deviations:

$$
\mu = { \frac { 1 } { T } } \sum _ { t = 1 } ^ { T } q _ { t } , \qquad \sigma = { \sqrt { { \frac { 1 } { T } } \sum _ { t = 1 } ^ { T } ( q _ { t } - \mu ) ^ { 2 } + \epsilon } } , \qquad r _ { t } = \left[ { \frac { q _ { t } - \mu } { \sigma } } \right] _ { + } .\tag{16}
$$

The score $r _ { t }$ selects tokens with above-average likelihood under the privileged prior within the response: tokens with $q _ { t } \leq \mu$ have zero score. This reflects relative outcome-conditioned support rather than guaranteed semantic importance. More details are provided in Appendix A.4. Before the

Table 1: Mathematical reasoning results across five benchmarks. Each benchmark reports Avg@8 / Pass@8, and the final pair gives the unweighted five-benchmark average. External baselines include ASFT (Zhu et al., 2026), TALR (Lin et al., 2026), and PriFT (Wang et al., 2026) whenever a matched result is available.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td colspan="2">MATH-500</td><td colspan="2">Minerva Math</td><td colspan="2">OlympiadBench</td><td colspan="2">AIME24</td><td colspan="2">AMC23</td><td colspan="2">Average</td></tr><tr><td>Avg@8</td><td>Pass@8</td><td>Avg@8</td><td>Pass@8</td><td>Avg@8</td><td>Pass@8</td><td>Avg@8</td><td>Pass@8</td><td>Avg@8</td><td>Pass@8</td><td>Avg@8</td><td>Pass@8</td></tr><tr><td rowspan="8">Qwen2.5 Math (7B)</td><td>Base</td><td>60.38</td><td>87.40</td><td>16.18</td><td>42.28</td><td>27.76</td><td>54.07</td><td>14.58</td><td>36.67</td><td>49.69</td><td>85.00</td><td>33.72</td><td>61.08</td></tr><tr><td>SFT</td><td>67.85</td><td>89.40</td><td>24.45</td><td>47.43</td><td>30.96</td><td>55.85</td><td>5.42</td><td>20.00</td><td>40.62</td><td>75.00</td><td>33.86</td><td>57.54</td></tr><tr><td>DFT</td><td>68.32</td><td>81.20</td><td>25.46</td><td>35.66</td><td>33.04</td><td>48.00</td><td>6.25</td><td>16.67</td><td>46.56</td><td>67.50</td><td>35.93</td><td>49.81</td></tr><tr><td>ASFT</td><td>70.32</td><td>88.00</td><td>22.75</td><td>38.24</td><td>34.50</td><td>57.04</td><td>7.08</td><td>16.67</td><td>47.81</td><td>80.00</td><td>36.49</td><td>55.99</td></tr><tr><td>TALR</td><td>65.88</td><td>85.20</td><td>22.56</td><td>44.85</td><td>29.20</td><td>53.48</td><td>5.83</td><td>16.67</td><td>35.62</td><td>75.00</td><td>31.82</td><td>55.04</td></tr><tr><td>PriFT</td><td>72.08</td><td>89.20</td><td>19.49</td><td>40.07</td><td>31.93</td><td>55.56</td><td>14.58</td><td>36.67</td><td>55.31</td><td>77.50</td><td>38.68</td><td>59.80</td></tr><tr><td>SPFT</td><td>74.05</td><td>89.40</td><td>26.61</td><td>47.43</td><td>37.48</td><td>59.11</td><td>15.83</td><td>40.00</td><td>59.69</td><td>87.50</td><td>42.73</td><td>64.69</td></tr><tr><td>SPFT-SR</td><td>72.05</td><td>87.40</td><td>30.15</td><td>50.00</td><td>36.89</td><td>56.44</td><td>14.58</td><td>36.67</td><td>57.81</td><td>82.50</td><td>42.30</td><td>62.60</td></tr><tr><td rowspan="8">Qwen3 8B Base</td><td>Base</td><td>74.98</td><td>90.00</td><td>31.85</td><td>53.68</td><td>39.94</td><td>59.56</td><td>12.92</td><td>33.33</td><td>53.75</td><td>80.00</td><td>42.69</td><td>63.31</td></tr><tr><td>SFT</td><td>70.28</td><td>87.60</td><td>21.92</td><td>34.93</td><td>33.07</td><td>56.44</td><td>10.83</td><td>20.00</td><td>45.94</td><td>82.50</td><td>36.41</td><td>56.29</td></tr><tr><td>DFT</td><td>74.92</td><td>87.00</td><td>22.93</td><td>30.51</td><td>39.61</td><td>57.33</td><td>16.67</td><td>30.00</td><td>56.56</td><td>87.50</td><td>42.14</td><td>58.47</td></tr><tr><td>ASFT</td><td>76.55</td><td>89.80</td><td>23.48</td><td>31.62</td><td>41.35</td><td>60.30</td><td>16.25</td><td>26.67</td><td>61.56</td><td>85.00</td><td>43.84</td><td>58.68</td></tr><tr><td>TALR</td><td>72.42</td><td>88.00</td><td>23.12</td><td>36.76</td><td>36.78</td><td>56.74</td><td>14.58</td><td>26.67</td><td>49.06</td><td>85.00</td><td>39.19</td><td>58.63</td></tr><tr><td>PriFT</td><td>80.47</td><td>92.00</td><td>30.56</td><td>39.71</td><td>45.89</td><td>64.74</td><td>23.75</td><td>43.33</td><td>62.50</td><td>90.00</td><td>48.63</td><td>65.96</td></tr><tr><td>SPFT</td><td>80.00</td><td>92.20</td><td>28.08</td><td>38.97</td><td>45.07</td><td>63.26</td><td>20.83</td><td>40.00</td><td>65.00</td><td>90.00</td><td>47.80</td><td>64.89</td></tr><tr><td>SPFT-SR</td><td>81.10</td><td>92.60</td><td>29.64</td><td>40.81</td><td>46.50</td><td>65.93</td><td>23.33</td><td>46.67</td><td>68.12</td><td>92.50</td><td>49.74</td><td>67.70</td></tr></table>

per-response rescaling described in Appendix A.4, the score-weighted objective has the form

$$
\mathcal { L } _ { \mathrm { S P F T - S R } } ( \theta ) = - \mathbb { E } _ { ( x , y , c ) \sim p _ { \mathrm { d a t a } } } \left[ \sum _ { t } r _ { t } \log p _ { \theta } ( y _ { t } \mid x , y _ { < t } ) \right] .\tag{17}
$$

## 5 EXPERIMENTS

## 5.1 EXPERIMENTAL SETUP

Mathematical reasoning. We train on a 10K random subsampled dataset from NuminaMath-CoT (Li et al., 2024). We use a global batch size of 32, a learning rate of $1 0 ^ { - 5 }$ with cosine decay and 5% warmup, one epoch, and a maximum sequence length of 3,072 tokens. We evaluate Qwen2.5-Math-1.5B and Qwen2.5-Math-7B (Qwen Team, 2024), together with Qwen3-8B-Base (Qwen Team, 2025). The evaluation uses the five-benchmark suite consisting of MATH-500 (Hendrycks et al., 2021), Minerva Math (Lewkowycz et al., 2022), OlympiadBench (He et al., 2024), AIME 2024 (Mathematical Association of America, 2024), and AMC 2023 (Mathematical Association of America, 2023), with eight samples per problem, temperature 0.7, top-p 0.8, evaluation seed 0, and the unified qwen-boxed protocol. For Qwen3-8B, the trained-model comparison uses the strictly matched non-thinking evaluation protocol.

Code generation. For code, we train on a 10K randomly subsampled dataset from OpenCodeInstruct (Ahmad et al., 2025) with a global batch size of 16, a learning rate of $5 \times 1 0 ^ { - 5 }$ , one epoch, and a maximum sequence length of 2,048 tokens. We evaluate Qwen2.5-Coder-1.5B (Hui et al., 2024) using EvalPlus (Liu et al., 2023) with greedy decoding. The reported metrics include the original HumanEval (Chen et al., 2021) and MBPP (Austin et al., 2021) benchmarks, together with their EvalPlus-extended HumanEval+ and MBPP+ variants (Liu et al., 2023), and their four-task average.

Baselines. We compare SPFT with the pretrained model (Base), standard supervised fine-tuning (SFT) (Chung et al., 2024), dynamic token reweighting (DFT) (Wu et al., 2026), anchored SFT (ASFT) (Zhu et al., 2026), temperature-based reweighting (TALR) (Lin et al., 2026), and frozenprior weighting (PriFT) (Wang et al., 2026). All trained methods use the same data, initialization, optimization budget, and evaluation protocol.

![](images/29ea1f3e94097326e63d5a1f46f8654bccfaa50ac7431252f3336798169405fd.jpg)  
(a) Pass@k

![](images/909c101a67f952dda6fd4658872fe5354dbd8e0c6b55ee980302b30e83d0adaa.jpg)  
(b) Maj@k  
Figure 3: Response-level sampling scaling on AMC 2023. Increasing k reveals whether a method can obtain a correct solution from multiple samples (Pass@k) and whether correct solutions become sufficiently dominant for majority voting (Maj@k). SPFT combines high discovery probability with strong consensus across the sampling budget.

Table 2: Qwen2.5-Coder-1.5B EvalPlus results. All values are greedy Pass@1 percentages; ∆ reports the change in four-task Code Avg relative to SFT (49.03%).
<table><tr><td>Method</td><td>HumanEval</td><td>HumanEval+</td><td>MBPP</td><td>MBPP+</td><td>Code Avg</td><td>∆ vs SFT</td></tr><tr><td>Base</td><td>43.9</td><td>36.6</td><td>69.2</td><td>58.6</td><td>52.08</td><td>1</td></tr><tr><td>SFT</td><td>41.5</td><td>37.2</td><td>63.2</td><td>54.2</td><td>49.03</td><td>0.00</td></tr><tr><td>ASFT</td><td>49.4</td><td>43.3</td><td>67.2</td><td>55.8</td><td>53.92</td><td>+4.89</td></tr><tr><td>PriFT</td><td>56.1</td><td>50.6</td><td>67.7</td><td>57.4</td><td>57.95</td><td>+8.92</td></tr><tr><td>TALR</td><td>59.8</td><td>52.4</td><td>67.5</td><td>56.3</td><td>59.00</td><td>+9.97</td></tr><tr><td>SPFT</td><td>56.7</td><td>53.0</td><td>70.6</td><td>58.5</td><td>59.70</td><td>+10.67</td></tr></table>

## 5.2 EXPERIMENTAL RESULTS

Mathematical Results. Table 1 summarizes the main mathematical results. Each benchmark reports Avg@8 and Pass@8, and the final pair reports the five-benchmark average Avg@8 and Pass@8. SPFT-SR improves over SFT on both metrics at the two main reported model scales.

The response-level sampling curves in Figure 3 complement the token-level diagnostics in Figure 2. Pass@k measures whether at least one of k sampled solutions is correct, while Maj@k measures correctness under majority voting. SPFT maintains strong growth in both metrics as k increases, indicating that it preserves useful output diversity while also increasing agreement around correct solutions. This is evidence for controlled sharpening rather than the near-deterministic collapse induced by DFT; Pass@k and Maj@k are response-level diagnostics, not direct token-level entropy measurements.

Code Results. Table 2 reports the Qwen2.5-Coder-1.5B comparison. The final column reports the improvement in four-task Code Avg relative to SFT. SPFT achieves the best four-task Code Avg among the trained methods, 59.70%, improving over SFT by 10.67 percentage points.

## 6 ABLATION STUDIES

We use two ablations in this section to isolate the design choices behind SPFT. We first test whether the privileged reference must remain frozen and then vary the form of privileged information in code generation. A complementary comparison between scalar target-token weighting, as in DFT (Wu et al., 2026), and full-vocabulary knowledge distillation (Hinton et al., 2015) is reported in Appendix C.1.

Table 3: Frozen versus online privileged reference for Qwen2.5-Math-1.5B. Each benchmark reports Avg@8 / Pass@8, and the final pair gives the unweighted five-benchmark average.
<table><tr><td rowspan="2">Reference</td><td colspan="2">MATH-500</td><td colspan="2">Minerva Math</td><td colspan="2">OlympiadBench</td><td colspan="2">AIME24</td><td colspan="2">AMC23</td><td colspan="2">Average</td></tr><tr><td> $\operatorname { A v g } ( \varnothing 8$ </td><td>Pass@8</td><td>Avg@8</td><td>Pass@8</td><td>Avg@8</td><td>Pass@8</td><td>Avg@8</td><td>Pass@8</td><td> $\operatorname { A v g } ( \varnothing 8$ </td><td>Pass@8</td><td> $\operatorname { A v g } ( \varnothing 8$ </td><td>Pass@8</td></tr><tr><td>Online (θt)</td><td>60.30</td><td>84.00</td><td>21.65</td><td>51.10</td><td>23.61</td><td>47.70</td><td>5.42</td><td>16.67</td><td>38.12</td><td>75.00</td><td>29.82</td><td>54.89</td></tr><tr><td>Frozen (θ0)</td><td>64.75</td><td>87.40</td><td>20.82</td><td>40.44</td><td>28.94</td><td>51.26</td><td>7.08</td><td>16.67</td><td>43.12</td><td>82.50</td><td>32.94</td><td>55.65</td></tr><tr><td>Frozen – Online</td><td>+4.45</td><td>+3.40</td><td>-0.83</td><td>-10.66</td><td>+5.33</td><td>+3.56</td><td>+1.66</td><td>0.00</td><td>+5.00</td><td>+7.50</td><td>+3.12</td><td>+0.76</td></tr></table>

Table 4: Privileged-information form ablation for Qwen2.5-Coder-1.5B. All rows use the same SPFT training objective; only the privileged prompt content changes.
<table><tr><td>Privileged information</td><td>HumanEval</td><td>HumanEval+</td><td>MBPP</td><td>MBPP+</td><td>Code Avg</td></tr><tr><td>SPFT (Clean Rule)</td><td>56.7</td><td>53.0</td><td>70.6</td><td>58.5</td><td>59.70</td></tr><tr><td>SPFT (Full Code)</td><td>58.5</td><td>54.3</td><td>63.9</td><td>54.1</td><td>57.70</td></tr><tr><td>SPFT (Unit Tests)</td><td>53.0</td><td>48.2</td><td>68.2</td><td>56.6</td><td>56.50</td></tr><tr><td>SPFT (Suffix-Half)</td><td>57.3</td><td>51.8</td><td>66.2</td><td>56.9</td><td>58.05</td></tr></table>

## 6.1 FROZEN VERSUS ONLINE PRIVILEGED REFERENCE

Our theoretical analysis identifies the reference weight as part of the optimization target rather than as a detached implementation detail. With a frozen privileged reference model, the weight is fixed throughout training:

$$
q _ { t } ^ { \mathrm { f r o z e n } } = p _ { \theta _ { 0 } } ( y _ { t } \mid x , c , y _ { < t } )\tag{18}
$$

The corresponding static objective preserves a stable reference prior, so the model is optimized against the same privileged signal at every update. The online variant removes this invariance by evaluating the privileged branch with the current parameters:

$$
q _ { t } ^ { \mathrm { o n l i n e } } = \operatorname { s g } [ p _ { \theta } ( y _ { t } \mid x , c , y _ { < t } ) ] .\tag{19}
$$

The reference weight then updates as the model optimizes. As established in Section 3, a frozen reference provides a stationary statistical target that ensures stable optimization. Table 3 shows the comparison between the two variants. The consistent aggregate improvement supports the theoretical prediction that a fixed reference prior is more stable than a reference that evolves dynamically alongside the model during optimization.

## 6.2 PRIVILEGED-INFORMATION FORM ABLATION

The main SPFT code result uses Clean Rule as privileged information, which conveys algorithmic intent without exposing the reference implementation. More details are provided in Section B.2. To isolate the role of the privileged-information representation, we compare Clean Rule against three alternatives: the complete reference code, unit-test information, and a suffix-half fragment of the reference code. Full-code conditioning supplies the most direct answer-level signal but can leak surface syntax and boilerplate. Unit-test information provides outcome-related constraints without exposing the entire implementation. Suffix-Half provides a partial textual reference and tests whether a local code fragment is sufficient. Clean Rule instead supplies a compact abstraction of algorithmic paradigms, data structures, and boundary guards. Table 4 shows that Clean Rule achieves the highest Code Avg score, which supports the use of high-level algorithmic guidance rather than direct code leakage.

## 7 CONCLUSION AND LIMITATIONS

We presented Self-Privileged Fine-Tuning (SPFT), an offline token-reweighted SFT framework that uses privileged information available during training but hidden from the model at inference. A frozen copy of the initial model conditions on privileged information, such as a verified answer or a structured description of algorithmic intent, and assigns scalar weights to target tokens while the trainable model receives only the ordinary task prompt. This design provides outcome-aware supervision without an external teacher model or online rollouts. We further developed a theoretical framework that distinguishes dynamic and static weighting, showing why a frozen reference is needed to preserve a fixed optimization target and avoid the moving, self-reinforcing signal caused by online updates. Experiments on mathematical reasoning and code generation, together with targeted ablations, demonstrate the effectiveness of SPFT and support these design choices.

SPFT currently relies on useful verifiable or structurally extractable privileged information, and our evaluation is limited to mathematics and code. Future work should investigate how to construct and validate appropriate privileged information in other domains, especially when outcomes are noisy, partial, subjective, or long-horizon. Extending SPFT to broader tasks and modalities, while improving robustness to imperfect privileged signals, is an important direction.

## ETHICS STATEMENT

This work does not raise significant ethical concerns beyond those commonly associated with research on language-model training. It does not involve human subjects, private or sensitive personal data, or deployment in high-stakes settings. We use publicly available datasets and open-source models, and we did not identify additional ethical risks specific to the methods or experiments presented here. The authors take responsibility for the ethical conduct and content of this work.

## AI ASSISTANCE DISCLOSURE

With the exception of limited implementation-related code assistance described below, we did not use generative AI tools to generate synthetic datasets, develop theoretical models or conceptual frameworks, formulate mathematical claims, provide critical ingredients for proofs, assist in writing proofs, propose or refine hypotheses, design or provide feedback on the research methodology or experiments, translate content, clean or reformat datasets, perform qualitative or thematic analysis, or analyze and interpret experimental results. Generative AI tools were used to assist with manuscript drafting and editing, including improving readability, grammar, spelling, and phrasing. They were also used in a limited capacity during implementation, primarily to identify potential bugs and vulnerabilities; the method design and algorithmic decisions were made by the authors. The authors reviewed all AI-assisted text for accuracy and consistency and verified the resulting implementation through manual inspection and testing. The authors take full responsibility for the final content of this paper, including all AI-assisted contributions.

## REPRODUCIBILITY STATEMENT

The assumptions and theoretical results are described in Section 3 and the appendix. The SPFT method and its implementation details are presented in Section 4, while the datasets, hyperparameters, training configurations, baselines, and evaluation protocols are described in Sections 5–6 and the appendix. These details are provided to facilitate reproduction of our results.

## ACKNOWLEDGMENTS

The authors acknowledge support from The Hong Kong Polytechnic University (Research Grant P0046811).

## REFERENCES

Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos Garea, Matthieu Geist, and Olivier Bachem. On-policy distillation of language models: Learning from selfgenerated mistakes. In International Conference on Learning Representations, 2024.

Wasi Uddin Ahmad, Aleksander Ficek, Mehrzad Samadi, Jocelyn Huang, Vahid Noroozi, Somshubra Majumdar, and Boris Ginsburg. OpenCodeInstruct: A large-scale instruction tuning dataset for code LLMs, 2025. URL https://arxiv.org/abs/2504.04030.

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

Tianzhe Chu, Yuexiang Zhai, Jihan Yang, Shengbang Tong, Saining Xie, Dale Schuurmans, Quoc V. Le, Sergey Levine, and Yi Ma. SFT memorizes, RL generalizes: A comparative study of foundation model post-training. In Proceedings ofthe 42nd International Conference on Machine Learning, volume 267, pp. 10818–10838, 2025. URL https://proceedings.mlr.press/ v267/chu25c.html.

Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. Scaling instruction-finetuned language models. Journal ofMachine Learning Research, 25(70):1–53, 2024.

Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang. MiniLLM: Knowledge distillation of large language models. In International Conference on Learning Representations, 2024.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, et al. OlympiadBench: A challenging benchmark for promoting AGI with olympiad-level bilingual multimodal scientific problems. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics, pp. 3828–3850, 2024.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. In Advances in Neural Information Processing Systems, 2021.

Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. In NeurIPS Deep Learning and Representation Learning Workshop, 2015.

Binyuan Hui, Jian Yang, Zeyu Cui, Jiaxi Yang, Dayiheng Liu, Lei Zhang, Tianyu Liu, Jiajun Zhang, Bowen Yu, Keming Lu, et al. Qwen2.5-Coder technical report. arXiv preprint arXiv:2409.12186, 2024.

Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, et al. Solving quantitative reasoning problems with language models. In Advances in Neural Information Processing Systems, volume 35, pp. 3843–3857, 2022.

Jia Li, Edward Beeching, Lewis Tunstall, Ben Lipkin, Roman Soletskyi, Shengyi Costa Huang, Kashif Rasul, Longhui Yu, Albert Jiang, Ziju Shen, Zihan Qin, Bin Dong, Li Zhou, Yann Fleureau, Guillaume Lample, and Stanislas Polu. NuminaMath, 2024.

Jiacheng Lin, Zhongruo Wang, Kun Qian, Tian Wang, Arvind Srinivasan, Hansi Zeng, Ruochen Jiao, Xie Zhou, Jiri Gesi, Dakuo Wang, Yufan Guo, Kai Zhong, Weiqi Zhang, Sujay Sanghavi, Changyou Chen, Hyokun Yun, and Lihong Li. SFT doesn’t always hurt general capabilities: Revisiting domain-specific fine-tuning in LLMs. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id= ru93xpQFi1.

Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, and Lingming Zhang. Is your code generated by ChatGPT really correct? rigorous evaluation of large language models for code generation. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https:// openreview.net/forum?id=1qvx610Cu7.

Mathematical Association of America. AMC 2023 competition problems. American Mathematics Competitions, 2023.

Mathematical Association of America. AIME 2024 competition mathematical problems. American Invitational Mathematics Examination, 2024.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems, volume 35, 2022.

Qwen Team. Qwen2.5-Math technical report, 2024. URL https://arxiv.org/abs/2409. 12122.

Qwen Team. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In Advances in Neural Information Processing Systems, volume 36, pp. 53728–53741, 2023.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Idan Shenfeld, Jyothish Pari, and Pulkit Agrawal. RL’s razor: Why online reinforcement learning forgets less. arXiv preprint arXiv:2509.04259, 2025.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. HybridFlow: A flexible and efficient RLHF framework. In Proceedings of the 2025 European Conference on Computer Systems, pp. 1279–1297, 2025.

Gokul Swamy, Sanjiban Choudhury, Wen Sun, Zhiwei Steven Wu, and J Andrew Bagnell. All roads lead to likelihood: The value of reinforcement learning in fine-tuning. arXiv preprint arXiv:2503.01067, 2025.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open founda tion and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

Ke Wang, Shuangqi Li, Mathieu Salzmann, and Pascal Frossard. PriFT: Prior-support guided supervised fine-tuning, 2026. URL https://arxiv.org/abs/2606.09396.

Yongliang Wu, Yizhou Zhou, Ziheng Zhou, Yingzhe Peng, Xinyu Ye, Xinting Hu, Wenbo Zhu, Lu Qi, Ming-Hsuan Yang, and Xu Yang. On the generalization of SFT: A reinforcement learning perspective with reward rectification. In The Fourteenth International Conference on Learning Representations, 2026. URL https://proceedings.iclr.cc/paper\_files/paper/2026/hash/ 2e53c02ea028cbf603f4b6b47fef3d97-Abstract-Conference.html.

Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, and Aditya Grover. Self-distilled reasoner: On-policy self-distillation for large language models. arXiv preprint arXiv:2601.18734, 2026.

He Zhu, Junyou Su, Peng Lai, Ren Ma, Wenjia Zhang, Linyi Yang, and Guanhua Chen. Anchored supervised fine-tuning. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=PORko7QT64.

## A MATHEMATICAL PROOFS AND WEIGHTING IMPLEMENTATION

## A.1 PROOF OF PROPOSITION 1 (STEPWISE OPTIMUM AND LIMITING DISTRIBUTION OF DFT)

For a fixed prefix context $( x , y _ { < t } )$ , let $p _ { \mathrm { d a t a } } ( v \mid x , y _ { < t } )$ denote the conditional next-token distribution of the demonstration dataset over the vocabulary V, and let $p _ { \theta _ { k } } ( v \mid x , y _ { < t } )$ denote the model’s distribution at optimization step k. At step k, the DFT objective in Eq. 2 decomposes across contexts $( x , y _ { < t } )$ . For a given context, minimizing the expected weighted negative log-likelihood with respect to the next-token distribution $p _ { \theta } ( \cdot \mid x , y _ { < t } )$ corresponds to solving:

$$
\operatorname* { m i n } _ { p _ { \theta } ( \cdot | x , y _ { < t } ) \in \Delta ( \mathcal { V } ) } - \sum _ { v \in \mathcal { V } } p _ { \mathrm { d a t a } } ( v \mid x , y _ { < t } ) p _ { \theta _ { k } } ( v \mid x , y _ { < t } ) \log p _ { \theta } ( v \mid x , y _ { < t } ) .\tag{20}
$$

Define the unnormalized nonnegative weight $u _ { k } ( v ) \ \triangleq \ p _ { \mathrm { d a t a } } ( v \ \vert \ x , y _ { < t } ) p _ { \theta _ { k } } ( v \ \vert \ x , y _ { < t } )$ and the partition constant $\begin{array} { r } { Z _ { k } ( x , y _ { < t } ) = \sum _ { v ^ { \prime } \in \mathcal { V } } { u } _ { k } ^ { - } ( v ^ { \prime } ) } \end{array}$ . Provided $Z _ { k } ( x , y _ { < t } ) > 0 .$ , normalizing $u _ { k }$ yields a valid probability distribution $\tilde { p } _ { k } ( v ) \stackrel { - } { = } u _ { k } ( v ) / Z _ { k } ( x , y _ { < t } )$ . By Gibbs’ inequality (nonnegativity of KL divergence),

$$
- \sum _ { v \in \mathcal { V } } u _ { k } ( v ) \log p _ { \theta } ( v \mid x , y _ { < t } ) = Z _ { k } ( x , y _ { < t } ) \Bigl [ D _ { \mathrm { K L } } \bigl ( \tilde { p } _ { k } ( \cdot ) \parallel p _ { \theta } ( \cdot \mid x , y _ { < t } ) \bigr ) + H ( \tilde { p } _ { k } ) \Bigr ] ,\tag{21}
$$

which is uniquely minimized when $p _ { \theta _ { k + 1 } } ( v  { \mid } x , y _ { < t } ) = \tilde { p } _ { k } ( v ) \propto p _ { \mathrm { d a t a } } ( v  { \mid } x , y _ { < t } ) p _ { \theta _ { k } } ( v  { \mid } x , y _ { < t } ) ,$ establishing Eq. 3.

By induction over $k \geq 0 .$ , repeatedly applying Eq. 3 yields the closed-form iterate in Eq. 4:

$$
p _ { \theta _ { k } } ( v \mid x , y _ { < t } ) = \frac { p _ { \theta _ { 0 } } ( v \mid x , y _ { < t } ) p _ { \mathrm { d a t a } } ( v \mid x , y _ { < t } ) ^ { k } } { \sum _ { v ^ { \prime } \in \mathcal { V } } p _ { \theta _ { 0 } } ( v ^ { \prime } \mid x , y _ { < t } ) p _ { \mathrm { d a t a } } ( v ^ { \prime } \mid x , y _ { < t } ) ^ { k } } .\tag{22}
$$

Let $v ^ { * } = \arg \operatorname* { m a x } _ { v \in \mathcal { V } } p _ { \mathrm { d a t a } } ( v ~ | ~ x , y _ { < t } )$ be the unique mode of $p _ { \mathrm { d a t a } } ( \cdot \ | \ x , y _ { < t } )$ , with $p _ { \theta _ { 0 } } ( v ^ { * } \ |$ $x , y _ { < t } ) > 0$ . Dividing both numerator and denominator by $p _ { \mathrm { d a t a } } ( v ^ { * } \mid x , y _ { < t } ) ^ { k }$ gives:

$$
p _ { \theta _ { k } } ( v \mid x , y _ { < t } ) = \frac { p _ { \theta _ { 0 } } ( v \mid x , y _ { < t } ) \left( \frac { p _ { \mathrm { d a t a } } ( v \mid x , y _ { < t } ) } { p _ { \mathrm { d a t a } } ( v ^ { * } \mid x , y _ { < t } ) } \right) ^ { k } } { p _ { \theta _ { 0 } } ( v ^ { * } \mid x , y _ { < t } ) + \sum _ { v ^ { \prime } \not = v ^ { * } } p _ { \theta _ { 0 } } ( v ^ { \prime } \mid x , y _ { < t } ) \left( \frac { p _ { \mathrm { d a t a } } ( v ^ { \prime } \mid x , y _ { < t } ) } { p _ { \mathrm { d a t a } } ( v ^ { * } \mid x , y _ { < t } ) } \right) ^ { k } } .\tag{23}
$$

Since $v ^ { * }$ is the unique maximizer, every $v ^ { \prime } \neq v ^ { * }$ satisfies $p _ { \mathrm { d a t a } } ( v ^ { \prime } \mid x , y _ { < t } ) / p _ { \mathrm { d a t a } } ( v ^ { * } \mid x , y _ { < t } ) < 1$ so the corresponding ratio raised to the k-th power converges to 0 as $k  \infty$ . Consequently, $\begin{array} { r } { \operatorname* { l i m } _ { k  \infty } p _ { \theta _ { k } } ( \boldsymbol { \bar { v } } \mid \boldsymbol { x } , \boldsymbol { \bar { y _ { < t } } } ) = \mathbf { 1 } [ v = v ^ { * } ] } \end{array}$ , completing the proof of Proposition 1. ■

## A.2 PROOF OF PROPOSITION 2 (OPTIMAL DISTRIBUTION OF STATIC WEIGHTING)

Consider the static token-weighted fine-tuning objective in Eq. 7, where $p _ { \mathrm { r e f } } ( v \mid x , y _ { < t } )$ is a frozen reference distribution independent of θ:

$$
\mathcal { L } _ { \mathrm { s t a t i c } } ( \theta ) = - \mathbb { E } _ { ( x , y ) \sim p _ { \mathrm { d a t a } } } \left[ \sum _ { t } p _ { \mathrm { r e f } } ( y _ { t } \mid x , y _ { < t } ) \log p _ { \theta } ( y _ { t } \mid x , y _ { < t } ) \right] .\tag{24}
$$

Conditioned on any fixed prefix $( x , y _ { < t } )$ in the support of $p _ { \mathrm { d a t a } }$ , the pointwise expected loss over the next token $v \in \mathcal V$ is:

$$
\ell _ { \mathrm { s t a t i c } } \big ( p _ { \theta } ( \cdot \mid x , y _ { < t } ) \big ) = - \sum _ { v \in \mathcal { V } } p _ { \mathrm { d a t a } } ( v \mid x , y _ { < t } ) p _ { \mathrm { r e f } } ( v \mid x , y _ { < t } ) \log p _ { \theta } ( v \mid x , y _ { < t } ) .\tag{25}
$$

Let $\begin{array} { r } { Z _ { \mathrm { r e f } } ( x , y _ { < t } ) = \sum _ { v ^ { \prime } \in \mathcal { V } } p _ { \mathrm { d a t a } } ( v ^ { \prime } \mid x , y _ { < t } ) p _ { \mathrm { r e f } } ( v ^ { \prime } \mid x , y _ { < t } ) } \end{array}$ denote the overlap partition constant. Defining the normalized product distribution

$$
p _ { \theta } ^ { * } ( v \mid x , y _ { < t } ) = \frac { p _ { \mathrm { d a t a } } ( v \mid x , y _ { < t } ) p _ { \mathrm { r e f } } ( v \mid x , y _ { < t } ) } { Z _ { \mathrm { r e f } } ( x , y _ { < t } ) } ,\tag{26}
$$

we can rewrite the pointwise objective up to an additive constant independent of $\theta$ as:

$$
\begin{array} { r l } & { \ell _ { \mathrm { s t a t i c } } \big ( p _ { \theta } ( \cdot \mid x , y _ { < t } ) \big ) = Z _ { \mathrm { r e f } } ( x , y _ { < t } ) D _ { \mathrm { K L } } \big ( p _ { \theta } ^ { * } ( \cdot \mid x , y _ { < t } ) \mid \mid p _ { \theta } ( \cdot \mid x , y _ { < t } ) \big ) } \\ & { \phantom { = } + Z _ { \mathrm { r e f } } ( x , y _ { < t } ) H \big ( p _ { \theta } ^ { * } ( \cdot \mid x , y _ { < t } ) \big ) . } \end{array}\tag{27}
$$

By Gibbs’ inequality, the KL divergence is uniquely minimized at zero when $p _ { \theta } ( v \mid x , y _ { < t } ) =$ $p _ { \theta } ^ { * } ( v \mid x , y _ { < t } ) \propto p _ { \mathrm { d a t a } } ( v \mid x , y _ { < t } ) p _ { \mathrm { r e f } } ( v \mid x , y _ { < t } )$ . Because $p _ { \mathrm { r e f } }$ is held fixed across optimization steps, $p _ { \theta } ^ { * }$ is stationary throughout training. ■

## A.3 PROOF OF PROPOSITION 3 AND CONNECTION TO OUTCOME-CONDITIONED INFERENCE

Let $( x , y , c )$ denote a problem prompt x, a demonstration sequence $y ~ = ~ ( y _ { 1 } , \dots , y _ { T } )$ , and the verified outcome c. Under the frozen reference model $p _ { \theta _ { 0 } }$ , applying Bayes’ rule to the sequencelevel conditional probability $p _ { \theta _ { 0 } } ( y \mid x , c )$ gives:

$$
\ p _ { \theta _ { 0 } } ( y \mid x , c ) = { \frac { p _ { \theta _ { 0 } } ( y , c \mid x ) } { p _ { \theta _ { 0 } } ( c \mid x ) } } = { \frac { p _ { \theta _ { 0 } } ( c \mid x , y ) } { p _ { \theta _ { 0 } } ( c \mid x ) } } p _ { \theta _ { 0 } } ( y \mid x ) ,\tag{28}
$$

which establishes the sequence-level ratio relation. Similarly, conditioning on the prefix $y _ { < t }$ at step $t \in \{ 1 , \ldots , T \}$ and applying Bayes’ rule to the event $( y _ { t } , c )$ given $( x , y _ { < t } )$ yields:

$$
\begin{array} { r l } { q _ { t } \ \triangleq \ p _ { \theta _ { 0 } } ( y _ { t } \mid x , c , y _ { < t } ) = \displaystyle { \frac { p _ { \theta _ { 0 } } ( y _ { t } , c \mid x , y _ { < t } ) } { p _ { \theta _ { 0 } } ( c \mid x , y _ { < t } ) } } } \\ & { \hphantom { \frac { p _ { \theta _ { 0 } } ( c \mid x , y _ { \le t } ) } { p _ { \theta _ { 0 } } ( c \mid x , y _ { < t } ) } } = \displaystyle { \frac { p _ { \theta _ { 0 } } ( c \mid x , y _ { \le t } ) } { p _ { \theta _ { 0 } } ( c \mid x , y _ { < t } ) } p _ { \theta _ { 0 } } ( y _ { t } \mid x , y _ { < t } ) } = \displaystyle { \frac { p _ { \theta _ { 0 } } ( c \mid x , y _ { \le t } ) } { p _ { \theta _ { 0 } } ( c \mid x , y _ { < t } ) } p _ { t } ^ { \mathrm { p r i o r } } } , } \end{array}\tag{29}
$$

which establishes Eq. 12.

## A.4 IMPLEMENTATION AND INTERPRETATION OF SPFT-SR WEIGHTS

As introduced in Section 4.3, SPFT-SR operates on detached privileged reference probabilities $q _ { t } =$ $p _ { \theta _ { 0 } } ( y _ { t } \mid x , c , y _ { < t } )$ rather than log-probabilities. Consider one sample $( x , y , c )$ with $T$ valid response tokens. To compare privileged confidence within this response, we compute

$$
\mu = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } q _ { t } , \qquad \sigma = \sqrt { \frac { 1 } { T } \sum _ { t = 1 } ^ { T } ( q _ { t } - \mu ) ^ { 2 } + \epsilon } ,\tag{30}
$$

where $\epsilon = 1 0 ^ { - 8 }$ ensures numerical stability. The clipped sequence-relative score is

$$
r _ { t } = \left[ \frac { q _ { t } - \mu } { \sigma } \right] _ { + } .\tag{31}
$$

Only tokens with above-average privileged confidence receive positive scores. We rescale them to have mean weight one over the response:

$$
w _ { t } = \frac { r _ { t } } { \operatorname* { m a x } \Bigl ( \frac { 1 } { T } \sum _ { u = 1 } ^ { T } r _ { u } , \epsilon \Bigr ) } .\tag{32}
$$

When $\begin{array} { r } { \sum _ { u = 1 } ^ { T } r _ { u } \le \epsilon . } \end{array}$ , the implementation instead uses uniform weights $w _ { t } = 1$ to avoid a zerogradient update. The resulting weights are detached before multiplying the model’s token-level cross-entropy loss. This is a sequence-relative calibration transformation, not an exact estimator of a log-ratio soft advantage.

## B PRIVILEGED INFORMATION CONSTRUCTION AND CONDITIONING INTERFACES

This section provides the exact prompt templates, static AST extraction rules, and concrete walkthrough examples for both mathematical reasoning and program synthesis, as well as the exact specifications of all four privileged-information (PI) variants evaluated in the code ablation study (Section 6.2, Table 4).

## B.1 MATHEMATICAL REASONING CONDITIONING INTERFACE (OPSD ANSWER)

In mathematical reasoning on NuminaMath-CoT 10K, the ground-truth terminal answer c is extracted from the final \boxed{...} expression of the verified solution y. The frozen privileged reference model receives c via the opsd answer template, whereas the target model receives only the standard problem prompt:

Problem: {problem}   
Here is a correct reference final answer to this problem:   
=== Reference Answer Begin ===   
{ground\_truth\_answer}   
=== Reference Answer End ===   
Use the reference final answer only as privileged guidance.   
Independently derive a valid solution; do not merely copy   
the answer. Think step by step, explore different approaches,   
and don’t be afraid to backtrack or reconsider if something   
doesn’t work out:   
Please reason step by step, and put your final answer   
within \boxed{}.

## B.2 PROGRAM SYNTHESIS PRIVILEGED INFORMATION: CLEAN RULE SPECIFICATION

Motivation. A complete reference implementation is an overly detailed privileged signal: it exposes function signatures, variable names, formatting, and routine control-flow tokens in addition to the algorithmic idea. Such surface-form information can make the reference model highly confident on boilerplate tokens and dilute the relative emphasis placed on algorithmic decisions. Clean Rule PI instead provides a compact description of the solution’s structure without supplying the implementation itself.

High-level construction. Clean Rule PI is obtained by lightweight deterministic static analysis of the verified reference solution. The analyzer identifies broad algorithmic paradigms (for example, dynamic programming, search, graph traversal, greedy or heap-based processing, hashing, sorting, and bit manipulation), records the main abstract data structures, and extracts a small number of boundary or early-return conditions when present. These signals are deduplicated and formatted as a short structural descriptor with three possible fields: Paradigms, Key Data Structures, and Boundary Invariants. The extraction is offline, deterministic, and does not call an external language model. The resulting descriptor is supplied only to the frozen privileged reference model; the target model and the inference-time model receive the original problem without the privileged description.

Concrete example. The following example illustrates the transformation from a verified implementation to its abstract privileged signal. The implementation is used only as the source for extraction; the privileged reference model receives only the second item.

## • Ground-truth reference:

```python
def twoSum(nums, target):
seen = {}
for i, num in enumerate(nums):
comp = target - num
if comp in seen:
return [seen[comp], i]
seen[num] = i
return []
```

## • Extracted Clean Rule PI:

Paradigms: Iterative Simulation

Table 5: Direct token weighting versus full-vocabulary KL for Qwen2.5-Math-1.5B. Each benchmark reports Avg@8 / Pass@8, and the final pair gives the unweighted five-benchmark average.
<table><tr><td rowspan="2">Objective</td><td colspan="2">MATH-500</td><td colspan="2">Minerva Math</td><td colspan="2">OlympiadBench</td><td colspan="2">AIME24</td><td colspan="2">AMC23</td><td colspan="2">Average</td></tr><tr><td>Avg@8</td><td>Pass@8</td><td>Avg@8</td><td>Pass@8</td><td>Avg@8</td><td>Pass@8</td><td>Avg@8</td><td>Pass@8</td><td>Avg@8</td><td>Pass@8</td><td>Avg@8</td><td>Pass@8</td></tr><tr><td>SPFT</td><td>64.75</td><td>87.40</td><td>20.82</td><td>40.44</td><td>28.94</td><td>51.26</td><td>7.08</td><td>16.67</td><td>43.12</td><td>82.50</td><td>32.94</td><td>55.65</td></tr><tr><td>Forward KL</td><td>41.23</td><td>81.80</td><td>9.97</td><td>37.50</td><td>19.02</td><td>46.37</td><td>3.75</td><td>16.67</td><td>20.00</td><td>65.00</td><td>18.79</td><td>49.47</td></tr><tr><td>Reverse KL</td><td>31.55</td><td>76.40</td><td>8.36</td><td>32.35</td><td>15.96</td><td>44.59</td><td>1.67</td><td>6.67</td><td>16.88</td><td>60.00</td><td>14.88</td><td>44.00</td></tr></table>

$$
\begin{array} { r l } & { \mathrm { K e y ~ \mathsf { \ D a t a } ~ \ s t r u c t u r e s : ~ \mathsf { H a s h } ~ \mathsf { M a p } ~ / ~ \mathsf { \Sigma } ~ S e t } } \\ & { \mathrm { B o u n d a r y ~ \mathsf { \ L n v a r i a n t s } : ~ \mathsf { C h e c k e d } ~ \mathsf { f o r } ~ \mathsf { \Sigma } ~ [ ~ ' ~ c o m p ~ \mathsf { i n } ~ \mathsf { s e e n } ~ ' ~ ] } } \end{array}
$$

## C ADDITIONAL ABLATION STUDIES

## C.1 DIRECT TOKEN WEIGHTING VERSUS FULL-VOCABULARY KD

SPFT uses the privileged reference model only to assign a scalar weight to the verified target token, as in token-reweighted objectives such as DFT (Wu et al., 2026). Standard knowledge distillation instead matches the reference and target distributions over the entire vocabulary at every prefix (Hinton et al., 2015). For a valid response position t, let

$$
q _ { t } ( v ) = p _ { \theta _ { 0 } } ( v \mid x , c , y _ { < t } ) , \qquad p _ { t } ( v ) = p _ { \theta } ( v \mid x , y _ { < t } ) .\tag{33}
$$

A full-vocabulary forward-KL control minimizes

$$
\mathcal { L } _ { \mathrm { K D } } ^ {  } ( \theta ; x , y , c ) = \sum _ { t = 1 } ^ { T } D _ { \mathrm { K L } } \big ( q _ { t } ( \cdot ) \| p _ { t } ( \cdot ) \big ) ,\tag{34}
$$

while the reverse-KL direction provides a complementary divergence control:

$$
\mathcal { L } _ { \mathrm { K D } } ^ {  } ( \theta ; x , y , c ) = \sum _ { t = 1 } ^ { T } D _ { \mathrm { K L } } \big ( p _ { t } ( \cdot ) \| q _ { t } ( \cdot ) \big ) .\tag{35}
$$

Both controls require matching a complete vocabulary distribution at every training prefix. This creates a potential train–test mismatch: training observes prefixes from the verified response through teacher forcing, whereas inference generates prefixes from the model’s own policy. Once a generated prefix deviates from the reference trajectory, the model is evaluated in contexts that were not represented by the KD loss. SPFT has a narrower supervision interface. It evaluates the privileged reference model on the same offline reference prefix but transfers only a scalar importance signal to the observed target token. As established in Section 3, this weighted SFT objective induces a target distribution proportional to the product of the data distribution and the reference distribution. It therefore avoids requiring the model to reproduce the reference’s entire conditional distribution on each prefix.

Table 5 shows a large gap between direct token weighting and full-vocabulary distribution matching. Forward KL achieves average Avg@8 / Pass@8 scores of 18.79% / 49.47%, while reverse KL achieves 14.88% / 44.00%, compared with 32.94% / 55.65% for SPFT. Thus, the completevocabulary objectives lose 14.15 / 6.18 and 18.06 / 11.65 percentage points in Avg@8 / Pass@8, respectively.

## D COMPLETE EXPERIMENTAL DETAILS AND EXTENDED RESULTS

## D.1 TRAINING AND EVALUATION HYPERPARAMETERS

Table 6 summarizes the training and evaluation configurations used across all mathematical reasoning and code generation experiments in Section 5 and Section 6. All methods within each domain share identical training data, batch sizes, learning rate schedules, sequence lengths, and evaluation decoding protocols.

Table 6: Summary of training and evaluation hyperparameters for mathematical reasoning and code generation.
<table><tr><td>Configuration Item</td><td>Mathematical Reasoning</td><td>Code Generation</td></tr><tr><td>Training dataset Backbone models</td><td>NuminaMath-CoT 10K (9,637 verified pairs)</td><td>OpenCodeInstruct 10K (10,000 verified pairs) Qwen2.5-Coder-1.5B</td></tr><tr><td>Privileged prompt style (c)</td><td>Qwen2.5-Math-1.5B, Qwen2.5-Math-7B, Qwen3-8B-Base opsd_answer (boxed final answer)</td><td>code_rule-paradigm (Clean Rule AST summary)</td></tr><tr><td>Global batch size</td><td></td><td></td></tr><tr><td></td><td>32</td><td>16</td></tr><tr><td>Peak learning rate</td><td> $1 . 0 \times 1 0 ^ { - 5 }$ </td><td> $5 . 0 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>LR schedule &amp; warmup</td><td>Cosine decay, 5% linear warmup</td><td>Cosine decay, 5% linear warmup</td></tr><tr><td>Training duration</td><td>1 epoch (301 optimization steps)</td><td>1 epoch (625 optimization steps)</td></tr><tr><td>Maximum sequence length</td><td>3,072 tokens</td><td>2,048 tokens</td></tr><tr><td>Optimizer &amp; precision</td><td>AdamW (β1 = 0.9, β2 = 0.999), BF16 + FSDP</td><td>AdamW (β1 = 0.9, β2 = 0.999), BF16 + FSDP</td></tr><tr><td>Evaluation benchmarks</td><td>MATH-500, Minerva, OlympiadBench, AIME24, AMC23</td><td>EvalPlus: HumanEval, HumanEval+, MBPP, MBPP+</td></tr><tr><td>Evaluation decoding</td><td>n = 8 samples, temperature 0.7, top-p 0.8, seed 0</td><td>Greedy decoding  $( T = 0 , \operatorname { P a s s } @ 1 )$ </td></tr><tr><td>Reported metrics</td><td>Avg@8, Pass@8 (Table 1); Maj@8 (Table 8)</td><td>Pass@1 on 4 tasks and unweighted Code Avg (Table 2)</td></tr></table>

Table 7: Additional mathematical reasoning results in the same format as Table 1.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td colspan="2">MATH-500</td><td colspan="2">Minerva Math</td><td colspan="2">OlympiadBench</td><td colspan="2">AIME24</td><td colspan="2">AMC23</td><td colspan="2">Average</td></tr><tr><td>Avg@8</td><td>Pass@8</td><td>Avg@8</td><td>Pass@8</td><td>Avg@8</td><td>Pass@8</td><td>Avg@8</td><td>Pass@8</td><td>Avg@8</td><td>Pass@8</td><td>Avg@8</td><td>Pass@8</td></tr><tr><td rowspan="8">Qwen2.5-Math 1.5B</td><td>Base</td><td>44.15</td><td>82.20</td><td>8.04</td><td>26.84</td><td>23.81</td><td>50.37</td><td>6.25</td><td>13.33</td><td>25.00</td><td>67.50</td><td>21.45</td><td>48.05</td></tr><tr><td>SFT</td><td>55.88</td><td>82.40</td><td>15.40</td><td>37.87</td><td>22.37</td><td>49.48</td><td>5.00</td><td>23.33</td><td>31.25</td><td>60.00</td><td>25.98</td><td>50.62</td></tr><tr><td>DFT</td><td>62.92</td><td>80.00</td><td>22.24</td><td>38.60</td><td>24.41</td><td>43.56</td><td>4.58</td><td>10.00</td><td>34.06</td><td>60.00</td><td>29.64</td><td>46.43</td></tr><tr><td>ASFT</td><td>59.20</td><td>82.40</td><td>18.98</td><td>40.81</td><td>23.28</td><td>46.52</td><td>4.17</td><td>13.33</td><td>32.81</td><td>70.00</td><td>27.69</td><td>50.61</td></tr><tr><td>TALR</td><td>56.60</td><td>82.60</td><td>17.97</td><td>41.54</td><td>22.22</td><td>48.00</td><td>1.67</td><td>6.67</td><td>31.56</td><td>62.50</td><td>26.00</td><td>48.26</td></tr><tr><td>PriFT</td><td>57.75</td><td>83.20</td><td>13.28</td><td>38.24</td><td>26.85</td><td>48.44</td><td>5.42</td><td>26.67</td><td>35.62</td><td>77.50</td><td>27.78</td><td>54.81</td></tr><tr><td>SPFT</td><td>64.75</td><td>87.40</td><td>20.82</td><td>40.44</td><td>28.94</td><td>51.26</td><td>7.08</td><td>16.67</td><td>43.12</td><td>82.50</td><td>32.94</td><td>55.65</td></tr><tr><td>SPFT-SR</td><td>67.30</td><td>85.80</td><td>20.82</td><td>41.91</td><td>30.80</td><td>51.70</td><td>9.17</td><td>23.33</td><td>43.44</td><td>80.00</td><td>34.31</td><td>56.55</td></tr><tr><td rowspan="8">Qwen3 1.7B-Base</td><td>Base</td><td>32.07</td><td>76.00</td><td>11.35</td><td>33.82</td><td>15.85</td><td>40.74</td><td>2.92</td><td>13.33</td><td>15.62</td><td>50.00</td><td>15.56</td><td>42.78</td></tr><tr><td>SFT</td><td>53.10</td><td>81.40</td><td>15.53</td><td>36.40</td><td>20.46</td><td>45.93</td><td>4.58</td><td>16.67</td><td>25.94</td><td>62.50</td><td>23.92</td><td>48.58</td></tr><tr><td>DFT</td><td>54.70</td><td>75.80</td><td>18.01</td><td>31.99</td><td>19.70</td><td>40.74</td><td>3.33</td><td>16.67</td><td>29.38</td><td>55.00</td><td>25.02</td><td>44.04</td></tr><tr><td>ASFT</td><td>55.40</td><td>80.80</td><td>15.81</td><td>33.46</td><td>21.48</td><td>45.93</td><td>2.50</td><td>13.33</td><td>24.06</td><td>57.50</td><td>23.85</td><td>46.20</td></tr><tr><td>TALR</td><td>53.08</td><td>80.00</td><td>16.18</td><td>37.87</td><td>21.20</td><td>45.19</td><td>3.75</td><td>13.33</td><td>30.00</td><td>67.50</td><td>24.84</td><td>48.78</td></tr><tr><td>PriFT</td><td>56.72</td><td>81.60</td><td>18.01</td><td>37.50</td><td>22.26</td><td>45.63</td><td>3.33</td><td>13.33</td><td>28.75</td><td>65.00</td><td>25.81</td><td>48.61</td></tr><tr><td>SPFT</td><td>56.15</td><td>81.00</td><td>16.64</td><td>41.54</td><td>22.67</td><td>45.63</td><td>2.92</td><td>10.00</td><td>30.94</td><td>67.50</td><td>25.86</td><td>49.13</td></tr><tr><td>SPFT-SR</td><td>57.75</td><td>80.20</td><td>19.30</td><td>35.66</td><td>23.09</td><td>45.93</td><td>7.08</td><td>20.00</td><td>30.00</td><td>62.50</td><td>27.44</td><td>48.86</td></tr></table>

## D.2 ADDITIONAL MODEL-SCALE RESULTS

Table 7 uses the same per-benchmark format as Table 1. Each benchmark reports Avg@8 / Pass@8, and the final pair gives the unweighted five-benchmark average.

## D.3 MAJORITY-VOTE (MA J@8) RESULTS FOR THE MAIN MATHEMATICAL COMPARISON

Table 8 reports the per-benchmark majority-vote accuracy (Maj@8) and its unweighted fivebenchmark average. It uses exactly the same eight sampled trajectories per problem as the Avg@8 / Pass@8 results in Table 1. Across the three main model backbones, SPFT and SPFT-SR achieve the highest average majority-vote accuracy, improving over standard SFT by +8.10 percentage points on Qwen2.5-Math-1.5B (41.67% vs. 33.57%), +8.94 percentage points on Qwen2.5-Math-7B (51.12% vs. 42.18%), and +12.78 percentage points on Qwen3-8B-Base (54.94% vs. 42.16%). The Qwen3- 1.7B-Base block is included as an additional model-scale comparison.

Table 8: Majority-vote accuracy (Maj@8) across all five mathematical reasoning benchmarks and their unweighted average for the main comparisons and the additional Qwen3-1.7B-Base modelscale comparison.
<table><tr><td>Model</td><td>Method</td><td>MATH-500</td><td>Minerva</td><td>Olympiad</td><td>AIME24</td><td>AMC23</td><td>Average</td></tr><tr><td rowspan="9">Qwen2.5-Math-1.5B</td><td>Base</td><td>56.20</td><td>6.62</td><td>31.11</td><td>13.33</td><td>27.50</td><td>26.95</td></tr><tr><td>SFT</td><td>68.00</td><td>20.59</td><td>30.07</td><td>6.67</td><td>42.50</td><td>33.57</td></tr><tr><td>DFT</td><td>67.20</td><td>25.37</td><td>27.85</td><td>6.67</td><td>35.00</td><td>32.42</td></tr><tr><td>ASFT</td><td>67.60</td><td>24.26</td><td>30.07</td><td>3.33</td><td>42.50</td><td>33.55</td></tr><tr><td>TALR</td><td>68.20</td><td>25.74</td><td>31.26</td><td>3.33</td><td>45.00</td><td>34.71</td></tr><tr><td>PriFT</td><td>69.20</td><td>14.34</td><td>33.63</td><td>6.67</td><td>40.00</td><td>32.77</td></tr><tr><td>SPFT</td><td>74.00</td><td>27.57</td><td>34.81</td><td>13.33</td><td>55.00</td><td>40.94</td></tr><tr><td>SPFT-SR</td><td>76.00</td><td>23.90</td><td>37.63</td><td>13.33</td><td>57.50</td><td>41.67</td></tr><tr><td>Base</td><td>75.00</td><td>19.49</td><td>34.22</td><td>26.67</td><td>65.00</td><td>44.08</td></tr><tr><td rowspan="8">Qwen2.5-Math-7B</td><td>SFT</td><td>76.60</td><td>33.09</td><td>41.19</td><td>10.00</td><td>50.00</td><td>42.18</td></tr><tr><td>DFT</td><td>73.00</td><td>26.47</td><td>36.44</td><td>6.67</td><td>47.50</td><td>38.02</td></tr><tr><td>ASFT</td><td>77.40</td><td>25.00</td><td>41.63</td><td>6.67</td><td>65.00</td><td>43.14</td></tr><tr><td>TALR</td><td>75.20</td><td>26.84</td><td>38.52</td><td>10.00</td><td>45.00</td><td>39.11</td></tr><tr><td>PriFT</td><td>79.20</td><td>25.37</td><td>38.81</td><td>20.00</td><td>67.50</td><td>46.18</td></tr><tr><td>SPFT</td><td>81.40</td><td>29.41</td><td>43.11</td><td>26.67</td><td>75.00</td><td>51.12</td></tr><tr><td>SPFT-SR</td><td>77.80</td><td>36.40</td><td>43.11</td><td>23.33</td><td>67.50</td><td>49.63</td></tr><tr><td>Base</td><td>82.60</td><td>40.07</td><td>48.89</td><td>20.00</td><td>72.50</td><td>52.81</td></tr><tr><td rowspan="8">Qwen3-8B-Base</td><td>SFT</td><td>76.80</td><td>23.90</td><td>39.26</td><td>13.33</td><td>57.50</td><td>42.16</td></tr><tr><td>DFT</td><td>79.00</td><td>23.53</td><td>43.11</td><td>23.33</td><td>60.00</td><td>45.79</td></tr><tr><td>ASFT</td><td>81.00</td><td>24.63</td><td></td><td></td><td>82.50</td><td></td></tr><tr><td>TALR</td><td>79.00</td><td>23.53</td><td>46.37 42.81</td><td>23.33 23.33</td><td>67.50</td><td>51.57</td></tr><tr><td>PriFT</td><td>85.80</td><td>31.99</td><td></td><td></td><td>72.50</td><td>47.23</td></tr><tr><td>SPFT</td><td>84.80</td><td>28.68</td><td>50.67 50.81</td><td>26.67 33.33</td><td>75.00</td><td>53.53</td></tr><tr><td>SPFT-SR</td><td>85.20</td><td>30.51</td><td>52.30</td><td>26.67</td><td>80.00</td><td>54.52</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>54.94</td></tr><tr><td rowspan="7">Qwen3-1.7B-Base</td><td>Base</td><td>38.60</td><td>13.24</td><td>21.33</td><td>0.00</td><td>17.50</td><td>18.13</td></tr><tr><td>SFT</td><td>65.60</td><td>20.96</td><td>28.74</td><td>3.33</td><td>32.50</td><td>30.23</td></tr><tr><td>DFT</td><td>61.60</td><td>20.96</td><td>25.04</td><td>3.33</td><td>30.00</td><td>28.19</td></tr><tr><td>ASFT</td><td>65.40</td><td>17.65</td><td>28.74</td><td>3.33</td><td>37.50</td><td>30.52</td></tr><tr><td>TALR</td><td>65.40</td><td>20.59</td><td>28.74</td><td>3.33</td><td>45.00</td><td>32.61</td></tr><tr><td>PriFT</td><td>66.00</td><td>21.69</td><td>30.81</td><td>6.67</td><td>40.00</td><td>33.03</td></tr><tr><td>SPFT</td><td>65.20</td><td>19.12 25.00</td><td>29.33</td><td>3.33</td><td>47.50</td><td>32.90</td></tr><tr><td></td><td>SPFT-SR</td><td>66.80</td><td></td><td>30.52</td><td>10.00</td><td>42.50</td><td>34.96</td></tr></table>