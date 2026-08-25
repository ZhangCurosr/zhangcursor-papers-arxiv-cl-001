# DIAG: Diagnostic Iterative Alignment and Generation for Data-Efficient Mathematical Preference Distillation

Guhan Chen<sup>1</sup>\*, Songtao Tian<sup>1∗</sup>, Bohan Li<sup>2</sup>, Hejin Wang<sup>1</sup>, Yexin Xie<sup>3</sup>, Zixiong Yu<sup>1,3†</sup> <sup>1</sup>Tsinghua University <sup>2</sup>Kyoto University <sup>3</sup>Nanjing University chen-gh23@mails.tsinghua.edu.cn xieye-xin@qq.com {tiansongtao.2020,libh19,wanghj20,yuzx19}@tsinghua.org.cn

## Abstract

Iterative preference optimization is essential for aligning Large Language Models on mathematical reasoning tasks, yet its efficiency is often throttled by signal scarcity: as the model improves, static problem sets become increas ingly mismatched to the model’s evolving competence, producing rollouts that are either too easy or too hard and therefore non-informative, which leads to a scarcity of valid preference pairs. We propose DIAG, a Diagnostic Iterative Alignment and Generation framework that adaptively reshapes the practice distribution to increase informative supervision and focus training near the student’s current competence boundary. DIAG consists of two phases: (1) diagnosing valid preference-pair yield to calibrate the exploration–exploitation trade-off and allocate topic quotas via an Empirical Bayes shrinkage estimator, thereby prioritizing high yield concepts; and (2) generating targeted practice, where a teacher synthesizes variants from the student’s failure traces. We further provide a theoretical view interpreting DIAG as a teacher-mediated approximation to KLregularized reweighting of the practice distribution toward the student’s competence boundary, where valid preference-pair yield is maximized. Experiments show that DIAG boosts yield across iterations and delivers stronger reasoning performance under an iso-effective training budget, demonstrating that it can distill more informative preference supervision for mathematical reasoning.

## 1 Introduction

Artificial intelligence (AI) has advanced at an unprecedented pace in recent years, spanning natural language processing (Zhao et al., 2023), computer vision (Cui et al., 2021; Tian et al., 2026), multimodal understanding and generation (Gemini Team, 2025; Yu et al., 2024), and intelligent agents (Shinn et al., 2023; Wang et al., 2026a,b). This progress has been driven by sustained advances in learning paradigms (Ouyang et al., 2022; Wu et al., 2026), model architectures (Vaswani et al., 2017; Yu et al., 2026b), data scaling (Yuan et al., 2023), and theoretical advances in understanding learning dynamics (Lai et al., 2023; Yu et al., 2025) and generalization (Li et al., 2024b; Chen et al., 2024a). Within this broader development, Large Language Models (LLMs) have emerged as a central paradigm, achieving increasingly strong performance across a wide range of tasks and becoming a foundation for modern AI applications (Ouyang et al., 2022). Their rapid development has consequently attracted growing interest in how to improve their reasoning, instruction following, and interaction with users and environments.

Against this backdrop, the landscape of LLMs has undergone a dramatic shift toward heterogeneity, with model scales, architectures, and deployment regimes ranging from massive frontier systems (GPT: OpenAI, 2024; Llama: Grattafiori et al., 2024) to specialized, lightweight variants (Phi-4- Mini: Microsoft, 2025; Gemini Nano: Gemini Team, 2025) optimized for edge deployment. This heterogeneity poses a challenge for model alignment: static training sets are increasingly unlikely to match the learning needs of models at different capability levels. Instances that are informative for one model may be too easy, too difficult, or even misleading for another, reducing the yield of supervision and potentially exacerbating hallucinationprone behavior (Gekhman et al., 2024).

Consequently, the alignment paradigm is shifting toward online, preference-based optimization loops (Rafailov et al., 2023; Pang et al., 2024). By generating rollouts on-policy, these online loops ensure that the learning signal is calibrated to the current reasoning state of the student model. This capability is crucial for maintaining effective alignment across heterogeneous models. Under this paradigm, supervision typically arises from natural contrasts: the student samples multiple rollouts, and a preference pair is formed only when the set contains both correct and incorrect trajectories. However, the efficacy of online rollouts remains fundamentally capped by the nature of the underlying prompt pool. Even if rollouts are generated on-policy and updated in real time, a fixed practice distribution may still fail to sample problems that match the model’s current capability, thereby producing too few informative preference pairs and wasting computational resources. Existing remedies often rely on brute-force scaling: increasing rollout width, expanding the static prompt pool, or using large-scale synthetic datasets (Yu et al., 2023; Toshniwal et al., 2024; Chen et al., 2024c).

![](images/4bb10bada810250c7de007ba58d6c1a5c9eebf1b6284a14c0b23f50836c01724.jpg)  
Figure 1: An illustrative example of DIAG. Left: The student directly compares likelihoods while ignoring prior probabilities, producing an incorrect posterior. Right: DIAG diagnoses the misconception from the student’s erroneous reasoning trace and generates a targeted follow-up with extreme priors that makes the error more salient, yielding more informative preference pairs for alignment.

A seemingly more adaptive alternative is active learning, which selects high-uncertainty prompts from a candidate pool (Settles, 2009). Yet for iterative preference optimization, active learning remains practice-agnostic in a crucial sense: it typically selects among existing prompts rather than generating targeted follow-ups tailored to the student model’s current state. As a result, it can still produce redundant queries or abrupt difficulty shifts, worsening preference-pair scarcity rather than alleviating it. This problem is conceptually a curriculum control problem. Classical curriculum learning and self-paced learning show that how examples are scheduled can be as important as how many examples are seen (Bengio et al., 2009; Kumar et al., 2010). However, applying curriculum ideas to iterative preference optimization introduces an additional challenge: the feedback signal is sparse, noisy at low sample counts, and nonstationary across iterations.

In this paper, we propose DIAG, a data-efficient framework for iterative preference distillation in mathematical reasoning. Our core insight is simple: conditioning teacher generation on student mistakes can improve both the yield ofpreference pairs and the quality of preference supervision. As illustrated in Figure 1, instead of treating the practice distribution as static, DIAG closes the loop between student outcomes and next-iteration practice. It follows a Diagnose-and-Generate paradigm: (i) diagnose how much usable preference signal the current curriculum yields; (ii) adaptively balance exploration and exploitation; (iii) allocate topic budgets using Empirical Bayes shrinkage to avoid low-count volatility (Efron and Morris, 1973); and (iv) generate targeted follow-up questions by conditioning a teacher on the student’s recent mistakes, focusing computation near the decision boundary through uncertainty-weighted resampling rather than static pool selection (Settles, 2009). This design increases the quantity of effective preference pairs while improving the quality of supervision by making new questions explicitly target observed failure modes. Our contributions are as follows:

• We formalize iterative preference distillation in verifiable mathematical reasoning as a signalyield problem, and empirically show that static practice distributions can lead to a rapid decline in natural contrastive supervision.

• We present DIAG, which couples a signal-aware pacing rule with Empirical Bayes topic scoring to robustly steer generation toward high-yield regions without over-reacting to low-count noise.

• We demonstrate that mistake-conditioned generation provides a simple and effective way to concentrate practice near the student’s decision boundary, yielding higher-quality preference pairs with greater information density per unit budget than practice-agnostic generation.

## 2 Related Work

Iterative Preference Optimization in Reasoning Post-training for LLM alignment has expanded from online RLHF pipelines (Schulman et al., 2017; Ouyang et al., 2022) to simpler preferencebased objectives such as DPO (Rafailov et al., 2023), which are easier to scale into iterative loops. Recent iterative variants for reasoning (e.g., IRPO, Pang et al. 2024; SVPO, Chen et al. 2024b) focus on improving the update rule or pair construction. Recent methods such as SAI-DPO dynamically select training data according to the model’s evolving reasoning abilities (Rao et al., 2026a). In contrast, DIAG diagnoses preference-pair yield and further generates targeted practice from the student’s erroneous reasoning traces. DIAG addresses an orthogonal bottleneck: even with a fixed verifier and rollout width, the practice distribution can quickly degenerate into all-correct or all-wrong regimes, starving the learner of usable preference pairs. In the spirit of curriculum learning and prioritized replay (Bengio et al., 2009; Kumar et al., 2010; Schaul et al., 2015; Jiang et al., 2021), our contribution is not a new objective but a signal-aware mechanism that allocates training budget where valid preference pairs are densest.

On-Policy Distillation (OPD) Recent on-policy distillation methods transfer teacher knowledge at the signal level: OPSD (Zhao et al., 2026) and SDPO (Hübotter et al., 2026) let the same model play teacher and student under different contexts and minimise a token-level KL divergence, while OPCD (Ye et al., 2026) enriches the teacher representation with auxiliary context. These approaches operate on fixed tasks or prompts and improve the supervisory distribution over the same problem. DIAG operates at an orthogonal, data level: the teacher analyses the student’s erroneous reasoning trace, diagnoses the underlying misconception, and synthesises a new question that forces the student to confront it, thereby reshaping the training distribution rather than the per-token supervision.

Reasoning Data Synthesis Throughout the lifecycle of large language models, data construction, curation, and feedback play central roles (Rao et al., 2025). Existing work has explored diverse data synthesis paradigms, including seed-based generation (Wang et al., 2023; Xu et al., 2023), seedless generation (Xu et al., 2025), and structured or design-guided synthesis (Yu et al., 2026c; Liu et al., 2026). However, these approaches generally overlook the value of targeted feedback. In contrast, DIAG starts from the student’s erroneous reasoning traces and uses targeted follow-up questions to continuously generate preference pairs with greater information content.

Mistake-Driven Augmentation Closest to our approach, LLM2LLM (Lee et al., 2024) prompts a teacher to generate new samples from student errors, but primarily conditions on the failed question rather than the student’s erroneous reasoning trace. Meanwhile, Rao et al. (2026b) improve exploration efficiency by mining and correcting failed reasoning trajectories, turning unsuccessful explorations into more informative training signals. Beyond reasoning-data augmentation, recent agent research has emphasized the value of execution traces, failure histories, and provenance for diagnosing erroneous states and guiding corrective actions (Yu et al., 2026a). In contrast, DIAG does not repair persistent agent memory; instead, it uses erroneous reasoning traces to diagnose misconceptions and generate targeted follow-up questions, integrating this mistake-driven augmentation with iterative preference distillation. Rather than correcting existing failures, it creates new questions near the decision boundary to improve the efficiency and informativeness of contrastive supervision.

## 3 Methods

## 3.1 Overview

As shown in Figure 2, we present DIAG, a framework that improves data efficiency in iterative preference distillation for mathematical reasoning.

![](images/c7a23099fa28bc69f1cfa7f8cc63b856a14069a986fff4ce6af139ed412cfe22.jpg)  
Figure 2: The DIAG Framework for Iterative Preference Distillation. DIAG identifies the student’s competence boundary and uncertainty, and generates targeted practice to sustain informative preference pairs, improving both supervision quality and data efficiency for math reasoning alignment.

Rather than training on a fixed prompt pool (Static Practice), DIAG adopts a Diagnose-and-Generate paradigm: after each DPO iteration it uses the student’s failures to adaptively reshape the practice distribution for the next round.

Concretely, the method operates in two coupled phases (see Algorithm 1 for the overall procedure):

(1) Diagnose & Allocate: estimate the effective pair yield from the previous iteration, set an exploration–exploitation budget split, and assign topic-level quotas via utility estimates smoothed by Empirical Bayes shrinkage to avoid noisy swings in the low-sample regime.

(2) Generate Practice: perform weighted resampling over recent mistakes within each topic and query a teacher to generate new follow-up problems conditioned on the failed question and the student’s incorrect response.

## 3.2 Preliminaries: Natural Preference Signal

Let $\pi _ { \theta }$ denote the student policy, and let $V ( y ; x ) \in$ {0, 1} denote a task verifier. For each prompt $x ,$ we draw K stochastic rollouts

$$
\{ y ^ { ( k ) } \} _ { k = 1 } ^ { K } \sim \pi _ { \theta } ( \cdot \mid x )
$$

and obtain correctness labels $\{ r ^ { ( k ) } \} _ { k = 1 } ^ { K }$ , where $r ^ { ( k ) } = V ( y ^ { ( k ) } ; x )$

Natural Pair Existence A prompt x is valid for constructing preference pairs if its rollouts contain at least one correct and one incorrect response:

$$
\mathbb { I } _ { \mathrm { v a l i d } } ( \boldsymbol { x } ) \triangleq \mathbb { I } \Big ( \exists i , j : \ r ^ { ( i ) } = 1 \ \wedge \ r ^ { ( j ) } = 0 \Big ) .\tag{1}
$$

Define the signal yield of a practice pool $P _ { t }$ as

$$
\rho _ { t } \triangleq \frac { 1 } { | P _ { t } | } \sum _ { x \in P _ { t } } \mathbb { I } _ { \mathrm { v a l i d } } ( x ) .\tag{2}
$$

When $\rho _ { t }$ collapses (e.g., many prompts yield allcorrect or all-wrong rollout sets), iterative preference learning experiences gradient starvation. DIAG explicitly adapts the practice generation process to maintain high $\rho _ { t }$ over iterations.

## 3.3 Phase I: Diagnose & Allocate

At each iteration, DIAG first diagnoses the amount of usable preference signal produced by the current practice distribution, then allocates the generation budget across topics accordingly. This phase combines signal-aware pacing with Empirical Bayes topic scoring to produce an adaptive, targeted, and noise-resilient generation plan.

Signal-aware pacing At the beginning of iteration t, DIAG uses the yield measured on the previous practice pool $P _ { t - 1 }$ to set the exploration rate:

$$
\alpha _ { t } \triangleq \lambda \big ( 1 - \rho _ { t - 1 } \big ) ,\tag{3}
$$

Algorithm 1 DIAG Distillation Loop   
1: Input: Topic set ${ \mathcal { Z } } ,$ Difficulty levels $\mathcal { D } _ { \mathbf { \lambda } }$   
teacher $\mathcal { M } _ { \mathrm { t e a c h } }$ , student $\pi _ { \theta } .$ , rollout width K,   
per-iter budget B   
2: Init: $N _ { z }  0 , C _ { z }  0 \forall z ; \ \mathcal { H } _ { 0 }  \emptyset ; \ \alpha _ { 1 }  1$   
3: for iteration $t = 1$ to T do   
4: // Phase I: Diagnose & Allocate   
5: $B _ { t } ^ { \mathrm { r n d } }  \lfloor \alpha _ { t } B \rfloor , ~ B _ { t } ^ { \mathrm { e x p } }  B - B _ { t } ^ { \mathrm { r n d } }$   
6: Compute scores $v _ { z } ; \quad \pi _ { \mathrm { s a m p l e r } } ( z ) \propto v _ { z }$   
7: $\mathcal { D } _ { t }  \emptyset , \ P _ { t }  \emptyset , \ \mathcal { H } _ { t }  \mathit { \tilde { \emptyset } }$   
8: // Phase II: Generate Practice   
9: for $j = 1$ to B do   
10: if $j \leq B _ { t } ^ { \mathrm { r n d } }$ then   
11: Sample $( \boldsymbol { z } , \boldsymbol { d } ) \sim \mathcal { U } ( \mathcal { Z } \times \mathcal { D } )$ ▷ explore   
12: $x \sim \mathcal { M } _ { \mathrm { t e a c h } } ( \cdot \mid z , d )$   
13: else   
14: Sample z ∼ π<sub>sampler</sub>;   
15: Get $( x _ { 0 } , d , y _ { \mathrm { e r r } } ) \sim \mathcal { H } _ { t - 1 } | _ { z }$ ▷ exploit   
16: $x \sim \mathcal { M } _ { \mathrm { t e a c h } } ( \cdot \mid z , d , x _ { 0 } , y _ { \mathrm { e r r } } )$   
17: end if   
18: $a ^ { \star }  \mathcal { M } _ { \mathrm { t e a c h } } ^ { \mathrm { a n s } } ( x )$ ▷ teacher solves   
19: Roll out $\{ y _ { k } \} _ { k = 1 } ^ { K } \sim \pi _ { \theta } ( \cdot \ | \ x ) ; \ r _ { k } \ \gets$   
$\mathcal { V } ( y _ { k } ; x , a ^ { \star } )$   
20: $P _ { t }  P _ { t } \cup \{ x \} ; \ N _ { z }  N _ { z } + 1$   
21: $\mathbf { i f } \exists i , j : r _ { i } = 1 , r _ { j } = 0$ then   
22: $y _ { w }  y _ { i } , \quad y _ { l }  y _ { j }$   
23: $\mathcal { D } _ { t }  \mathcal { D } _ { t } \cup \{ ( x , y _ { w } , y _ { l } ) \}$   
24: $C _ { z }  C _ { z } + 1$   
25: $\mathcal { H } _ { t }  \mathcal { H } _ { t } \cup \{ ( x , z , d , y _ { l } ) \}$   
26: end if   
27: end for   
28: Update $\pi \theta$ by minimizing L<sub>DPO</sub> on $\mathcal { D } _ { t }$   
29: $\rho _ { t }  \vert \{ x \in P _ { t } : \mathbb { I } _ { \mathrm { v a l i d } } ( x ) = 1 \} \vert / \vert P _ { t } \vert$   
30: $\alpha _ { t + 1 }  \lambda ( 1 - \rho _ { t } )$ ▷ next-iter explore rate   
31: end for

so that exploration increases when informative natural pairs are scarce and decreases when the current curriculum already produces abundant on-policy contrastive signal. By construction, $\alpha _ { t } \in [ 0 , \lambda ] \subseteq$ [0, 1], where $\lambda \in ( 0 , 1 ]$ is a fixed scaling factor controlling how aggressively the sampler shifts budget in response to the measured yield.

Budget split Given a fixed generation budget of B questions per iteration, we allocate

$$
B _ { t } ^ { \mathrm { r n d } } \triangleq \left\lfloor \alpha _ { t } B \right\rfloor , \quad B _ { t } ^ { \mathrm { e x p } } \triangleq B - B _ { t } ^ { \mathrm { r n d } } ,\tag{4}
$$

where $B _ { t } ^ { \mathrm { r n d } }$ is assigned to uniform exploration over the topic–difficulty space $( z , d )$ , and $\boldsymbol { B } _ { t } ^ { \mathrm { e x p } }$ is assigned to the exploitation stream. Intuitively, when $\rho _ { t - 1 }$ is low, we expand coverage to search for regions that can generate usable preference signal; when $\rho _ { t - 1 }$ is high, we concentrate the exploitation budget on high-yield topics and recent mistakes to maximize sample efficiency. In practice, this acts as a damping mechanism: when the model encounters a “plateau” with low signal yield, the system automatically allocates more budget to broad exploration to discover new learnable frontiers. See Appendix E for details.

Empirical Bayes topic scoring We maintain pertopic statistics to estimate which topics are most likely to yield informative natural preference pairs, and use these estimates to guide the allocation of the exploitation budget $\boldsymbol { B } _ { t } ^ { \mathrm { e x p } }$ . Let $z \in { \mathcal { Z } }$ denote a topic and let x be a question tagged with topic z. For each attempted question, we update

$$
N _ { z }  N _ { z } + 1 , \qquad C _ { z }  C _ { z } + \mathbb { I } _ { \mathrm { v a l i d } } ( x ) ,\tag{5}
$$

where $N _ { z }$ counts how many questions of topic z have been attempted, while $C _ { z }$ counts how often topic z produced a usable natural contrastive signal.

We model the latent utility $p _ { z }$ (the probability that topic z yields a valid natural pair) as a Beta random variable with Empirical Bayes prior:

$$
\begin{array} { r } { p _ { z } \sim \mathrm { B e t a } ( \alpha _ { 0 } , \beta _ { 0 } ) , \qquad } \\ { p _ { z } | ( C _ { z } , N _ { z } ) \sim \mathrm { B e t a } ( \alpha _ { 0 } + C _ { z } , \beta _ { 0 } + N _ { z } - C _ { z } ) . } \end{array}
$$

Here, $\alpha _ { 0 }$ and $\beta _ { 0 }$ are prior pseudo-counts controlling the amount of shrinkage toward the global valid rate. Let $\bar { p }$ denote the global mean valid rate across topics, and let $\kappa > 0$ denote the total prior strength. We set $\alpha _ { 0 } = \kappa \bar { p }$ and $\beta _ { 0 } = \kappa ( 1 - \bar { p } )$ . The posterior mean gives a Stein-type shrinkage score:

$$
\begin{array} { r } { v _ { z } \triangleq \mathbb { E } [ p _ { z } \mid C _ { z } , N _ { z } ] = \frac { C _ { z } + \kappa \bar { p } } { N _ { z } + \kappa } . } \end{array}\tag{6}
$$

Topic sampling policy We sample topics for the exploitation stream proportionally to $v _ { z } \colon$

$$
\pi _ { \mathrm { s a m p l e r } } ( z ) = v _ { z } / \sum _ { z ^ { \prime } \in \mathcal { Z } } v _ { z ^ { \prime } } .\tag{7}
$$

This Empirical Bayes scheduler prevents the curriculum from over-reacting to low-count noise while focusing generation on topics that repeatedly yield informative on-policy supervision.

## 3.4 Phase II: Generate Practice

Given the budget split and topic sampler from Phase I, we next construct the practice pool for the following iteration. The exploration stream samples uniformly, while the exploitation stream conditions on the student’s recent failures to generate targeted follow-up questions.

Mistake buffer At iteration t, we collect a buffer of boundary mistake cases:

$$
\mathcal { H } _ { t } \triangleq \{ ( x , z , d , y _ { \mathrm { e r r } } ) \} ,\tag{8}
$$

where $( z , d )$ denotes topic and difficulty, and $y _ { \mathrm { e r r } }$ is an incorrect trace from a prompt that also admitted a correct rollout (i.e., a valid preference pair). Only such mixed-outcome prompts enter $\mathcal { H } _ { t }$ , as they lie near the student’s competence boundary. Degenerate episodes still update the signal-yield and topic statistics; when their mass grows, the pacing rule shifts budget toward exploration.

Trace-aware question synthesis For each exploitation slot, we first sample a topic $z \sim$ $\pi _ { \mathrm { s a m p l e r } } ( z )$ , then retrieve a recent mistake case $( x , z , d , y _ { \mathrm { e r r } } ) \ \in \ \mathcal { H } _ { t - }$ <sub>−1</sub> with the same topic (and matched difficulty), via uncertainty-weighted resampling (Settles, 2009), and query the teacher to generate a targeted variant:

$$
x ^ { \prime } \sim \mathcal { M } _ { \mathrm { t e a c h } } ( \cdot \mid z , ~ d , ~ x , ~ y _ { \mathrm { e r r } } , ~ \mathrm { i n s t r u c t i o n } ) .\tag{9}
$$

The instruction enforces: (i) preserve the underlying concept (same z), (ii) match difficulty $d ,$ and (iii) alter surface form (numbers/context) to avoid memorization while probing the same failure mode. For the exploration stream, questions are generated from metadata alone without mistake conditioning. The resulting questions together form the next practice pool $P _ { t }$

Theoretical interpretation Intuitively, generating from mistakes focuses computation near the student’s competence boundary. Appendix A.1 formalizes this: for rollout width K, the probability that a prompt yields a valid preference pair is $\ d s _ { 1 } - \ d p _ { } ^ { K } - \ d s _ { } ( 1 - \ d p _ { } ) ^ { K }$ , which peaks at $p = 1 / 2$ and vanishes in the all-correct and allwrong regimes. Appendix A.2 then derives a KL-regularized reweighting that exponentially upweights high-yield prompts. DIAG approximates this ideal update via Empirical-Bayes topic allocation and teacher generation conditioned on recent boundary mistakes.

Policy update Once the practice pool $P _ { t }$ is constructed via Phases I–II, we collect preference pairs by labeling student rollouts based on ground-truth correctness. The student policy $\pi _ { \theta }$ is then updated by minimizing the standard DPO objective:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { D P O } } ( \pi _ { \theta } , \pi _ { \mathrm { r e f } } ) = - \mathbb { E } _ { ( x , y _ { w } , y _ { l } ) \sim \mathcal { D } } } \\ & { \qquad \Big [ \log \sigma \big ( \beta [ h _ { \theta } ( x , y _ { w } ) - h _ { \theta } ( x , y _ { l } ) ] \big ) \Big ] , } \end{array}\tag{10}
$$

where $h _ { \theta }$ is the implicit reward difference and $\pi _ { \mathrm { r e f } }$ is a fixed reference policy for the current iteration.

## 4 Experiments

## 4.1 Setup

Benchmarks We evaluate on two standard math benchmarks GSM8K (Cobbe et al., 2021) and MATH500 (Hendrycks et al., 2021b), together with harder testbeds: MinervaMath (Lewkowycz et al., 2022), Gaokao2023En (Liao et al., 2024), College Math (Tang et al., 2024), OlympiadBench (He et al., 2024), AIME24 (AoPS, 2024), and AMC23 (AoPS, 2023). We use greedy decoding for all benchmarks except AMC23 & AIME24, whose small test sizes make single-run accuracy more sensitive to sampling noise. For these two benchmarks, we sample 8 solutions per problem with temperature $T = 0 . 1$ and top- ${ \cdot } p = 0 . 9 5$ , and report the mean pass@1 accuracy to reduce evaluation variance.

Student Models We evaluate three open-weight student models with approximately 7B–8B parameters: Qwen2.5-Math-7B (Yang et al., 2024), Qwen3-8B-Base (Yang et al., 2025), and Llama-3.1-8B-Instruct (Grattafiori et al., 2024), spanning multiple model series and capability profiles.

Budget and fairness All methods share the same total student-sample budget. We run IDPO (Guo et al., 2024; Pang et al., 2024) for 7 iterations with $3 \times 1 0 ^ { 3 }$ prompts per iteration, keeping the decoding protocol identical across methods. We employ Qwen3-235B (Int4) as the teacher model.

Practice pool construction (what differs) We compare three pool-construction strategies under the same student-sample budget:

• Static Dataset Pool (Numina). We use the opensource NuminaMath dataset (Li et al., 2024a) as a fixed practice pool, without online regeneration.

• Static Gen (Teacher). The teacher generates all $T \times B$ questions upfront using only metadata such as topic and difficulty. The resulting pool is fixed and partitioned equally across iterations.

• Ours: Mistake-Conditioned. At each iteration, the teacher regenerates the pool by conditioning on the student’s failures (Sec. 3).

## 4.2 Main Results

Table 1 reports the full results. Despite identical student-sample budgets, DIAG achieves the strongest average performance across all three student models. The gains are most pronounced on harder testbeds, where generic practice pools are more likely to enter degenerate regimes, reducing the availability of natural preference supervision.

<table><tr><td rowspan="2">Method</td><td colspan="3">Standard</td><td colspan="2">Exam</td><td colspan="3">Competition (Hard)</td><td rowspan="2">Avg.</td></tr><tr><td>GSM8K</td><td>MATH 500</td><td>Minerva Math</td><td>Gaokao 2023en</td><td>College Math</td><td>Olympiad Bench</td><td>AIME24 (Avg@8)</td><td>AMC23 (Avg@8)</td></tr><tr><td colspan="10">Qwen2.5-Math-7B</td></tr><tr><td>BASE (Start)</td><td>84.2</td><td>70.2</td><td>26.5</td><td>58.7</td><td>35.0</td><td>33.8</td><td>24.2</td><td>54.7</td><td>48.4</td></tr><tr><td>NUMINA → IDPO</td><td>85.4</td><td>70.4</td><td>29.8</td><td>59.0</td><td>37.8</td><td>35.9</td><td>22.5</td><td>56.6</td><td>49.7</td></tr><tr><td>STATIC GEN → IDPO</td><td>86.5</td><td>69.6</td><td>33.8</td><td>60.8</td><td>38.7</td><td>35.7</td><td>22.1</td><td>59.7</td><td>50.9</td></tr><tr><td>DIAG (Ours)</td><td>87.1</td><td>70.8</td><td>34.9</td><td>61.0</td><td>41.9</td><td>38.4</td><td>25.0</td><td>63.1</td><td>52.8</td></tr><tr><td colspan="10">Qwen3-8B-Base</td></tr><tr><td>BASE (Start)</td><td>89.1</td><td>70.0</td><td>35.3</td><td>57.4</td><td>36.3</td><td>33.9</td><td>8.8</td><td>50.0</td><td>47.6</td></tr><tr><td>NUMINA → IDPO</td><td>90.9</td><td>73.4</td><td>34.9</td><td>62.3</td><td>38.1</td><td>40.1</td><td>8.3</td><td>56.2</td><td>50.5</td></tr><tr><td>STATIC GEN → IDPO</td><td>91.1</td><td>76.6</td><td>42.6</td><td>63.1</td><td>39.1</td><td>40.0</td><td>10.4</td><td>52.2</td><td>51.9</td></tr><tr><td>DIAG (Ours)</td><td>92.0</td><td>79.0</td><td>40.8</td><td>64.7</td><td>39.7</td><td>41.3</td><td>14.6</td><td>57.5</td><td>53.7</td></tr><tr><td colspan="10">Llama-3.1-8B-Instruct</td></tr><tr><td>BASE (Start)</td><td>84.8</td><td>47.6</td><td>25.7</td><td>41.6</td><td>28.8</td><td>14.5</td><td>7.1</td><td>21.2</td><td>33.9</td></tr><tr><td>NUMINA → IDPO</td><td>86.3</td><td>50.8</td><td>27.9</td><td>43.4</td><td>28.9</td><td>15.7</td><td>5.8</td><td>24.4</td><td>35.4</td></tr><tr><td>STATIC GEN → IDPO</td><td>86.9</td><td>50.2</td><td>29.4</td><td>42.1</td><td>30.0</td><td>15.6</td><td>6.7</td><td>26.6</td><td>35.9</td></tr><tr><td>DIAG (Ours)</td><td>88.4</td><td>48.2</td><td>31.6</td><td>45.2</td><td>32.1</td><td>17.6</td><td>7.5</td><td>28.4</td><td>37.4</td></tr></table>

Table 1: Main Results on Mathematical Reasoning (Fixed Total Budget). We compare four data sources under the same total student-sample budget: (BASE) the starting checkpoint, (NUMINA) IDPO on a static open-source pool, (STATIC GEN) IDPO on a fixed teacher-generated pool conditioned on (topic, difficulty), and DIAG (Ours) IDPO on online teacher generation conditioned on (failed question, wrong answer, topic, difficulty).

<table><tr><td rowspan="2">Task</td><td colspan="2">Sampling Strategies</td></tr><tr><td>w/o Explore</td><td>w/o EB Ours</td></tr><tr><td>GSM8K</td><td>87.6</td><td>86.4 87.1</td></tr><tr><td>MATH500</td><td>70.4 70.6</td><td>70.8</td></tr><tr><td>MinervaMath</td><td>35.7 35.3</td><td>34.9</td></tr><tr><td>Gaokao2023</td><td>59.5</td><td>59.7 61.0</td></tr><tr><td>Olympiad</td><td>37.5</td><td>38.7 38.4</td></tr><tr><td>CollegeMath</td><td>42.4</td><td>40.3 41.9</td></tr><tr><td>AIME24</td><td>24.2</td><td>24.2 25.0</td></tr><tr><td>AMC23</td><td>58.1 59.4</td><td>63.1</td></tr><tr><td>Avg.</td><td>51.9 (-0.9)</td><td>51.8 (-1.0) 52.8</td></tr></table>

Table 2: Ablation on Sampling / Scheduling. We ablate the uniform exploration stream (w/o Explore) and EB smoothing (w/o EB) under a fixed rollout budget.

## 4.3 Ablation Study

Sampling & Scheduling We ablate the sampling & scheduling design on Qwen2.5-Math-7B under a fixed rollout budget: (i) w/o Explore: remove the uniform exploration stream, allocating the entire budget to score-based exploitation; (ii) w/o Empirical Bayes (EB): replace EB shrinkage with raw per-topic rates; (iii) Ours: the full scheduler.

Table 2 shows that both components matter in different ways. Removing exploration yields comparable scores on standard benchmarks but exhibits brittleness on competition-style evaluation (AMC23: 58.1 vs 63.1). Removing EB shrinkage leads to broad degradation on most benchmarks (Avg 51.8), confirming that naive frequency-based scoring is unstable in the low-sample regime.

<table><tr><td>Method / Variant</td><td>Avg.</td><td>∆</td></tr><tr><td>Full DIAG</td><td>52.8</td><td></td></tr><tr><td>Impact of conditioning on failed question w/o Wrong-QA (no failed context)</td><td></td><td></td></tr><tr><td></td><td>51.3</td><td>-1.5</td></tr><tr><td>Impact of conditioning on wrong answer/trace</td><td></td><td></td></tr><tr><td>w/o Wrong-A (no wrong answer)</td><td>52.1 -0.7</td><td></td></tr></table>

Table 3: Ablation on Teacher Conditioning Signals. ∆ denotes performance drop relative to the full method.

Teacher Conditioning Signals We isolate which components of the teacher conditioning signal matter for targeted generation, all on Qwen2.5-Math-7B with identical budget and topic/difficulty controls: (i) w/o Wrong-QA: teacher conditioned only on topic/difficulty (no failed context); (ii) w/o Wrong-A: teacher receives the failed question but not the student’s wrong answer; (iii) Ours: full conditioning on topic + difficulty + failed question + wrong answer.

As shown in Table 3, performance increases monotonically as richer error context is provided. Removing the failed context (−1.5) weakens the teacher’s ability to anchor generation to the exact skill being tested. Removing the wrong answer (−0.7) confirms that error-aware generation is not equivalent to simple topic-conditioned rewriting: observing the student’s mistake helps the teacher probe the same misconception more reliably.

![](images/c8aa115718b1e1ebe94b327db5596b929355265780ab45e8e12e682f27b461cc.jpg)  
Figure 3: Effective sample yield across iterations. DIAG maintains a higher signal yield $\rho _ { t }$ than baselines and avoids degeneration into uninformative regimes.

## 4.4 Cross-Task Generalization

To verify that DIAG does not incur an alignment tax, we evaluate Qwen3-8B-Base on BBH (Suzgun et al., 2023), HumanEval (Chen et al., 2021), MMLU (Hendrycks et al., 2021a), and TruthfulQA (Lin et al., 2022). Table 4 shows that DIAG maintains comparable performance on knowledge and truthfulness benchmarks while showing mild positive transfer on reasoning and code generation.

<table><tr><td rowspan="2">Method</td><td rowspan="2">BBH</td><td rowspan="2">Human Eval</td><td rowspan="2">MMLU</td><td colspan="2">TruthfulQA</td></tr><tr><td>MC1</td><td>MC2</td></tr><tr><td>Qwen3-8B-Base</td><td>75.5</td><td>64.6</td><td>76.9</td><td>33.5</td><td>51.1</td></tr><tr><td>+ Static Gen</td><td>75.6↑</td><td>64.6</td><td>76.9</td><td>33.6↑</td><td>51.2↑</td></tr><tr><td>+ DIAG</td><td>75.6↑</td><td>64.7↑</td><td>76.9</td><td>33.9↑</td><td>51.0</td></tr></table>

Table 4: Cross-Task Generalization and Alignment Tax Evaluation. Bold denotes the best result; <sup>↑</sup> indicates improvement over the base model.

## 5 Analysis

## 5.1 Effective Sample Yield over Training

A central claim of our method is that optimizing the practice distribution improves sample efficiency under a fixed rollout budget. Following (2), we track the effective yield $\rho _ { t } ,$ defined as the fraction of questions whose K rollouts contain both correct and incorrect outcomes, across iterations on Qwen2.5-Math-7B.

![](images/ef3726e5b8de2a6db320ae0463d82bdf309a603e0b3e733f96e07680e224e280.jpg)  
Figure 4: Pass-rate distribution. Our method concentrates more mass in the intermediate region, reducing both all-correct and all-wrong degeneracy.

Figure 3 reports an extended 10-iteration diagnostic run: our method increases $\rho _ { t }$ from ∼0.59 to ∼0.83 (+24 pp) over 10 iterations, whereas both Static Gen and static Numina gradually decline to around 0.55, confirming that targeted generation keeps the training distribution near the student’s moving competence boundary.

## 5.2 Pass-Rate Distribution

The yield gap above suggests that the key advantage comes from where the model practices. We analyze the pass-rate distribution by estimating $\begin{array} { r } { \hat { p } ( x ) \triangleq \frac { 1 } { K } \sum _ { k = 1 } ^ { K } r _ { k } ( x ) } \end{array}$ for each generated question. Mass near $\hat { p } ( x ) \approx 1 \ \mathrm { o r } \ \hat { p } ( x ) \approx 0$ indicates wasted budget on trivial or intractable questions, respectively; both extremes cause gradient starvation. Figure 4 shows that our method concentrates more mass in the intermediate band, where the student is neither always correct nor always wrong. This suggests that mistake-conditioned generation acts as a calibration operator: it improves difficulty matching rather than simply making questions harder.

## 5.3 Iso-Effective Comparison

Our sampler produces more mixed episodes, which could confound improvements due purely to data quantity. To disentangle quantity from quality, we subsample each method to the same number (∼8k) of effective training instances and train the same Qwen2.5-Math-7B student. We additionally include two error-driven baselines under the same DPO framework: LLM2LLM-style (Lee et al., 2024), which generates new questions from student failures but without yield-aware scheduling (originally SFT; we use DPO for fair comparison); and SPIN (Chen et al., 2024d), which constructs self-play preference pairs on a fixed prompt pool.

<table><tr><td>Method</td><td>Eff. Samples</td><td>Avg.</td><td>∆</td></tr><tr><td>Qwen2.5-Math-7B</td><td>一</td><td>48.4</td><td>一</td></tr><tr><td>+ SPIN (Chen et al., 2024d)</td><td>8k</td><td>48.9</td><td>+0.5</td></tr><tr><td>+ NuminaMath (Li et al., 2024a)</td><td>8k</td><td>49.0</td><td>+0.6</td></tr><tr><td>+ Static IDPO (Zhang et al., 2025)</td><td>8k</td><td>49.7</td><td>+1.3</td></tr><tr><td>+ LLM2LLM (Lee et al., 2024)</td><td>8k</td><td>50.1</td><td>+1.7</td></tr><tr><td>+ Ours</td><td>8k</td><td>50.7</td><td>+2.3</td></tr></table>

Table 5: Iso-effective comparison. All methods use ∼8k effective instances and the same gradient steps. LLM2LLM-style adopts error-driven generation but trains with DPO for fair comparison.

As shown in Table 5, our method remains consistently better even when the effective sample count is held constant. SPIN (+0.5), operating on fixed prompts, confirms that prompt-level curriculum matters beyond response-level contrast. LLM2LLM-style (+1.7) benefits from error-driven generation but conditions the teacher only on the failed question; DIAG’s additional conditioning on the student’s wrong reasoning trace enables follow-ups that target the specific misconception, and its EB-guided scheduling further improves signal yield. This gap is consistent with the analysis in Appendix A.1 and A.5: boundary prompts produce more mixed rollouts, placing the model in a higher-curvature, more informative update region.

## 6 Conclusion

We showed that the dominant bottleneck in iterative preference distillation for math reasoning is signal scarcity: as the student improves, static practice pools degenerate into uninformative regimes. To address this, we proposed DIAG, a Diagnoseand-Generate framework that reshapes the practice distribution via signal-aware pacing, Empirical-Bayes topic prioritization, and mistake-conditioned generation. Across diverse students and challenging math evaluations, DIAG consistently improves both data efficiency and final reasoning performance, with gains persisting even under isoeffective training, indicating higher-quality supervision beyond increased signal yield.

## Limitations

DIAG diagnoses signal scarcity in real time via Empirical Bayes shrinkage and uses mistakeconditioned generation to produce targeted followup problems near the student model’s moving competence boundary. Two main limitations remain: (i) the Diagnose-and-Generate loop relies on a capable teacher model for question synthesis, which may not be available or affordable in all settings; (ii) our evaluation is restricted to mathematical reasoning with reliable verifiers, and it is unclear how well the framework transfers to domains where correctness cannot be automatically checked.

## Ethics Statement

This work complies with the ACL Ethics Policy. All experiments are conducted using publicly available datasets and widely accessible models, with proper citations provided throughout. Our method does not involve human subjects, private data, or any form of data annotation beyond automated answer verification. We acknowledge that LLMs, including those trained with our framework, may still produce incorrect or misleading reasoning steps. While DIAG improves mathematical reasoning reliability, users should independently verify model outputs before relying on them in high-stakes or real-world decision-making scenarios.

## References

Art of Problem Solving AoPS. 2023. AMC Problems and Solutions. https://artofproblemsolving. com/wiki/index.php/AMC\_12\_Problems\_and\_ Solutions. Accessed: 2025-09-30.

Art of Problem Solving AoPS. 2024. AIME Problems and Solutions. https://artofproblemsolving. com/wiki/index.php/AIME\_Problems\_and\_ Solutions. Accessed: 2025-09-30.

Yoshua Bengio, Jérôme Louradour, Ronan Collobert, and Jason Weston. 2009. Curriculum learning. In Proceedings of the 26th Annual International Conference on Machine Learning, ICML 2009, Montreal, Quebec, Canada, June 14-18, 2009, volume 382 of ACM International Conference Proceeding Series, pages 41–48. ACM.

Guhan Chen, Yicheng Li, and Qian Lin. 2024a. On the impacts of the random initialization in the neural tangent kernel theory. In Advances in Neural Information Processing Systems, volume 37, pages 35909–35944. Curran Associates, Inc.

Guoxin Chen, Minpeng Liao, Chengxi Li, and Kai Fan. 2024b. Step-level value preference optimization for mathematical reasoning. arXiv preprint arXiv:2406.10858.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, and 39 others. 2021. Evaluating large language models trained on code. Preprint, arXiv:2107.03374.

Nuo Chen, Ning Wu, Jianhui Chang, Linjun Shou, and Jia Li. 2024c. Controlmath: Controllable data generation promotes math generalist models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 12201–12217.

Zixiang Chen, Yihe Deng, Huizhuo Yuan, Kaixuan Ji, and Quanquan Gu. 2024d. Self-play fine-tuning converts weak language models to strong language models. arXiv preprint arXiv:2401.01335.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, and 1 others. 2021. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, and 1 others. 2025. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261.

Quan Cui, Boyan Zhou, Yu Guo, Weidong Yin, Hao Wu, Osamu Yoshie, and Yubo Chen. 2021. Contrastive vision-language pre-training with limited resources. arXiv preprint arXiv:2112.09331.

Bradley Efron and Carl Morris. 1973. Stein’s estimation rule and its competitors—an empirical bayes approach. Journal ofthe American Statistical Association, 68(341):117–130.

Zorik Gekhman, Gal Yona, Roee Aharoni, Matan Eyal, Amir Feder, Roi Reichart, and Jonathan Herzig. 2024. Does fine-tuning LLMs on new knowledge encourage hallucinations? In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing, pages 7765–7784, Miami, Florida, USA. Association for Computational Linguistics.

Gemini Team. 2025. Gemini: A family of highly capable multimodal models. Preprint, arXiv:2312.11805.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, and 1 others. 2024. The llama 3 herd of models. Preprint, arXiv:2407.21783.

Shangmin Guo, Biao Zhang, Tianlin Liu, Tianqi Liu, Misha Khalman, Felipe Llinares, Alexandre Rame, Thomas Mesnard, Yao Zhao, Bilal Piot, and 1 others. 2024. Direct language model alignment from online ai feedback. arXiv preprint arXiv:2402.04792.

Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, Jie Liu, Lei Qi, Zhiyuan Liu, and Maosong Sun. 2024. OlympiadBench: A challenging benchmark for promoting AGI with olympiad-level bilingual multimodal scientific problems. In Proceedings ofthe 62nd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 3828–3850, Bangkok, Thailand. Association for Computational Linguistics.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021a. Measuring massive multitask language understanding. In International Conference on Learning Representations.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021b. Measuring mathematical problem solving with the math dataset. Preprint, arXiv:2103.03874.

Jonas Hübotter, Frederike Lübeck, Lejs Behric, Anton Baumann, Marco Bagatella, Daniel Marta, Ido Hakimi, Idan Shenfeld, Thomas Kleine Buening, Carlos Guestrin, and Andreas Krause. 2026. Reinforcement learning via self-distillation. arXiv preprint arXiv:2601.20802.

Minqi Jiang, Edward Grefenstette, and Tim Rocktäschel. 2021. Prioritized level replay. In International Conference on Machine Learning, pages 4940–4950. PMLR.

M Kumar, Benjamin Packer, and Daphne Koller. 2010. Self-paced learning for latent variable models. Advances in neural information processing systems, 23.

Jianfa Lai, Zixiong Yu, Songtao Tian, and Qian Lin. 2023. Generalization ability of wide residual networks. Preprint, arXiv:2305.18506.

Xin Lai, Zhuotao Tian, Yukang Chen, Senqiao Yang, Xiangru Peng, and Jiaya Jia. 2024. Step-dpo: Step-wise preference optimization for long-chain reasoning of llms. arXiv preprint arXiv:2406.18629.

Nicholas Lee, Thanakul Wattanawong, Sehoon Kim, Karttikeya Mangalam, Sheng Shen, Gopala Anumanchipalli, Michael Mahoney, Kurt Keutzer, and Amir Gholami. 2024. Llm2llm: Boosting llms with novel iterative data enhancement. In Findings of the Associationfor Computational Linguistics: ACL 2024, pages 6498–6526.

Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, Yuhuai Wu, Behnam Neyshabur, Guy Gur-Ari, and Vedant Misra. 2022. Solving quantitative reasoning problems with language models. Preprint, arXiv:2206.14858.

Jia Li, Edward Beeching, Lewis Tunstall, Ben Lipkin, Roman Soletskyi, Shengyi Huang, Kashif Rasul,

Longhui Yu, Albert Q Jiang, Ziju Shen, and 1 others. 2024a. Numinamath: The largest public dataset in ai4maths with 860k pairs of competition math problems and solutions. Hugging Face repository, 13(9):9.

Yicheng Li, Zixiong Yu, Guhan Chen, and Qian Lin. 2024b. On the eigenvalue decay rates of a class of neural-network related kernel functions defined on general domains. Journal ofMachine Learning Research, 25(82):1–47.

Minpeng Liao, Chengxi Li, Wei Luo, Wu Jing, and Kai Fan. 2024. MARIO: MAth reasoning with code interpreter output - a reproducible pipeline. In Findings of the Associationfor Computational Linguistics: ACL 2024, pages 905–924, Bangkok, Thailand. Association for Computational Linguistics.

Stephanie Lin, Jacob Hilton, and Owain Evans. 2022. TruthfulQA: Measuring how models mimic human falsehoods. In Proceedings ofthe 60th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 3214–3252, Dublin, Ireland. Association for Computational Linguistics.

Weize Liu, Yongchi Zhao, Yijia Luo, Mingyu Xu, Jiaheng Liu, Yanan Li, Xiguo Hu, ZhiqiBai, Yuchi Xu, Wenbo Su, and Bo Zheng. 2026. DESIGNER: Design-logic-guided multidisciplinary data synthesis for LLM reasoning. In The Fourteenth International Conference on Learning Representations.

Microsoft. 2025. Phi-4-mini technical report: Compact yet powerful multimodal language models via mixture-of-loras. Preprint, arXiv:2503.01743.

OpenAI. 2024. Gpt-4 technical report. Preprint, arXiv:2303.08774.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, and 1 others. 2022. Training language models to follow instructions with human feedback. NeurIPS, 35:27730–27744.

Richard Yuanzhe Pang, Weizhe Yuan, He He, Kyunghyun Cho, Sainbayar Sukhbaatar, and Jason Weston. 2024. Iterative reasoning preference optimization. Advances in Neural Information Processing Systems, 37:116617–116637.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. 2023. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36:53728–53741.

Jun Rao, Xuebo Liu, Hexuan Deng, Zepeng Lin, Zixiong Yu, Jiansheng Wei, Xiaojun Meng, and Min Zhang. 2026a. Dynamic sampling that adapts: Selfaware iterative data persistent optimization for mathematical reasoning. In Findings of the Association for Computational Linguistics: ACL 2026, pages 28337–28350, San Diego, California, United States. Association for Computational Linguistics.

Jun Rao, Xuebo Liu, Haotian Yan, Junjie Shen, Haosi Mo, Yanghaopeng Dong, Zihao Yan, Ziyi Wang, Zepeng Lin, Xiaojun Meng, Zixiong Yu, Liqun Deng, Jiansheng Wei, Yunhe Wang, and Min Zhang. 2025. A data-centric perspective on the lifecycle of large language models. TechRxiv, 2025(1220).

Jun Rao, Zixiong Yu, Xuebo Liu, Guhan Chen, Jing Li, Hejin Wang, Jiansheng Wei, Xiaojun Meng, and Min Zhang. 2026b. Mining or synthesis? rethinking exploration efficiency in iterative alignment of mathematical reasoning. Preprint, arXiv:2602.05370.

Tom Schaul, John Quan, Ioannis Antonoglou, and David Silver. 2015. Prioritized experience replay. arXiv preprint arXiv:1511.05952.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.

Burr Settles. 2009. Active learning literature survey.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik R Narasimhan, and Shunyu Yao. 2023. Reflexion: language agents with verbal reinforcement learning. In Thirty-seventh Conference on Neural Information Processing Systems.

Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc Le, Ed Chi, Denny Zhou, and Jason Wei. 2023. Challenging BIG-bench tasks and whether chain-of-thought can solve them. In Findings ofthe Associationfor Computational Linguistics: ACL 2023, pages 13003–13051, Toronto, Canada. Association for Computational Linguistics.

Zhengyang Tang, Xingxing Zhang, Benyou Wang, and Furu Wei. 2024. Mathscale: Scaling instruction tuning for mathematical reasoning. In ICML, pages 47885–47900.

Songtao Tian, Guhan Chen, Bohan Li, Jingyi Ma, and Zixiong Yu. 2026. Curvature-adaptive consistency flow matching: Autonomous trajectory optimization via reinforcement learning. Preprint, arXiv:2606.22394.

Shubham Toshniwal, Ivan Moshkov, Sean Narenthiran, Daria Gitman, Fei Jia, and Igor Gitman. 2024. Openmathinstruct-1: A 1.8 million math instruction tuning dataset. Advances in Neural Information Processing Systems, 37:34737–34774.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc.

Yiqi Wang, Zihao Yan, Jiaqi Zhang, Zhangkai Wu, Mingkai Zheng, Zequn Sun, Yanming Zhu, and Taotao Cai. 2026a. Map-graph: Provenance-aware shared memory for multi-agent workflows. Preprint, arXiv:2608.10509.

Yiqi Wang, Jiaqi Zhang, Taotao Cai, Zirui Liu, Qingqiang Sun, Zequn Sun, Zhangkai Wu, Manqing Dong, Mingkai Zheng, Xuefei Yin, and Yanming Zhu. 2026b. From agent traces to trust: A survey of evidence tracing and execution provenance in llm agents. Preprint, arXiv:2606.04990.

Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A Smith, Daniel Khashabi, and Hannaneh Hajishirzi. 2023. Self-instruct: Aligning language models with self-generated instructions. In Proceedings of the 61st annual meeting of the association for computational linguistics (volume 1: long papers), pages 13484–13508.

Wenxun Wu, Yuanyang Li, Guhan Chen, Linyue Wang, and Hongyang Chen. 2026. Tool-augmented policy optimization: Synergizing reasoning and adaptive tool use with reinforcement learning. Preprint, arXiv:2510.07038.

Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, and Daxin Jiang. 2023. Wizardlm: Empowering large language models to follow complex instructions. arXiv preprint arXiv:2304.12244.

Zhangchen Xu, Fengqing Jiang, Luyao Niu, Yuntian Deng, Radha Poovendran, Yejin Choi, and Bill Yuchen Lin. 2025. Magpie: Alignment data synthesis from scratch by prompting aligned LLMs with nothing. In The Thirteenth International Conference on Learning Representations.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, and 1 others. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong Tu, Jingren Zhou, Junyang Lin, Keming Lu, Mingfeng Xue, Runji Lin, Tianyu Liu, Xingzhang Ren, and Zhenru Zhang. 2024. Qwen2.5-math technical report: Toward mathematical expert model via self-improvement. Preprint, arXiv:2409.12122.

Tianzhu Ye, Li Dong, Xun Wu, Shaohan Huang, and Furu Wei. 2026. On-policy context distillation for language models. Preprint, arXiv:2602.12275.

Bihui Yu, Dawei Liu, Huiyang Shi, GuiYong Chang, Jingxuan Wei, Linzhuang Sun, Songtao Tian, and Liping Bu. 2024. Sam-wav2lip++: Enhancing behavioral realism in synthetic agents through audio-driven speech and action refinement. In 2024 IEEE International Conference on Systems, Man, and Cybernetics (SMC), pages 2999–3006.

Caili Yu, Yiqi Wang, Jiaqi Zhang, Yiqun Duan, Mingkai Zheng, Zhangkai Wu, Kaize Shi, and Taotao Cai. 2026a. From faulty memories to corrected actions: Dependency-guided rollback repair for memoryaugmented agents. Preprint, arXiv:2608.10502.

Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. 2023. Metamath: Bootstrap your own mathematical questions for large language models. arXiv preprint arXiv:2309.12284.

Zixiong Yu, Guhan Chen, Jianfa Lai, Bohan Li, and Songtao Tian. 2026b. Branch scaling manifests as implicit architectural regularization for improving generalization in overparameterized resnets. In Fortythird International Conference on Machine Learning.

Zixiong Yu, Jun Rao, Guhan Chen, Songtao Tian, Bohan Li, Jiansheng Wei, Min Zhang, and Xiaojun Meng. 2026c. MathAgent: Adversarial evolution of constraint graphs for mathematical reasoning data synthesis. In Findings of the Association for Computational Linguistics: ACL 2026, pages 28298–28321, San Diego, California, United States. Association for Computational Linguistics.

Zixiong Yu, Songtao Tian, and Guhan Chen. 2025. Divergence of neural tangent kernel in classification problems. In The Thirteenth International Conference on Learning Representations.

Zheng Yuan, Hongyi Yuan, Chengpeng Li, Guanting Dong, Keming Lu, Chuanqi Tan, Chang Zhou, and Jingren Zhou. 2023. Scaling relationship on learning mathematical reasoning with large language models. arXiv preprint arXiv:2308.01825.

Hanning Zhang, Jiarui Yao, Chenlu Ye, Wei Xiong, and Tong Zhang. 2025. Online-dpo-r1: Unlocking effective reasoning without the ppo overhead. Notion Blog.

Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, and Aditya Grover. 2026. Self-distilled reasoner: On-policy self-distillation for large language models. Preprint, arXiv:2601.18734.

Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, and 1 others. 2023. A survey of large language models. arXiv preprint arXiv:2303.18223.

## A Theoretical Perspective: Signal Yield and Practice-Space Reweighting

This appendix provides a compact theoretical view of DIAG. The main idea is that useful natural preference pairs are most likely to arise near the student’s competence boundary, where the model sometimes succeeds and sometimes fails. DIAG can then be interpreted as a teacher-mediated approximation to an ideal practice-distribution reweighting that moves probability mass toward such high-yield regions.

## A.1 Signal Yield Peaks at the Competence Boundary

Let $\pi _ { \boldsymbol { \theta } } ( \boldsymbol { y } \quad | \quad \boldsymbol { x } )$ be the student policy and let $V ( y ; x ) \in \{ 0 , 1 \}$ be the verifier. For a prompt x, define the student’s pass probability as

$$
p _ { \theta } ( x ) = \operatorname* { P r } _ { y \sim \pi _ { \theta } ( . | x ) } \left[ V ( y ; x ) = 1 \right] .\tag{11}
$$

With rollout width $K \geq 2 ,$ , a valid natural preference pair exists when at least one rollout is correct and at least one rollout is incorrect. Thus the probability that x yields a valid pair is

$$
u _ { \theta , K } ( x ) = 1 - p _ { \theta } ( x ) ^ { K } - \bigl ( 1 - p _ { \theta } ( x ) \bigr ) ^ { K } .\tag{12}
$$

Proposition A.1 For any $K \geq 2$ , the function

$$
u _ { K } ( p ) = 1 - p ^ { K } - ( 1 - p ) ^ { K } , \quad p \in [ 0 , 1 ] ,\tag{13}
$$

is uniquely maximized at $p = 1 / 2 .$ , with value

$$
u _ { K } ( 1 / 2 ) = 1 - 2 ^ { 1 - K } .\tag{14}
$$

Moreover, $u _ { K } ( 0 ) = u _ { K } ( 1 ) = 0 .$

Proof The derivative is

$$
u _ { K } ^ { \prime } ( p ) = K \big [ ( 1 - p ) ^ { K - 1 } - p ^ { K - 1 } \big ] ,\tag{15}
$$

which vanishes only at $p = 1 / 2$ . The second derivative is

$$
u _ { K } ^ { \prime \prime } ( p ) = - K ( K - 1 ) \big [ p ^ { K - 2 } + ( 1 - p ) ^ { K - 2 } \big ] < 0
$$

for $p \in ( 0 , 1 )$ and $K \geq 2$ , so $u _ { K }$ is strictly concave and has a unique maximum at $p = 1 / 2$ . The boundary values follow directly. □

This proposition formalizes why all-correct and all-wrong prompts are inefficient: both regimes have low probability of producing usable preference pairs. Prompts near the competence boundary, where $p _ { \theta } ( x )$ is intermediate, provide denser supervision under the same rollout budget.

For a practice distribution $\mu ,$ define the expected valid-pair yield as

$$
J _ { \theta , K } ( \mu ) = \mathbb { E } _ { x \sim \mu } \left[ u _ { \theta , K } ( x ) \right] .\tag{16}
$$

If M prompts are sampled in an iteration, the expected number of effective DPO training instances is $M J _ { \theta , K } ( \mu )$

## A.2 Ideal KL-Regularized Practice Reweighting

The previous result suggests that an ideal practice distribution should upweight prompts with large $u _ { \theta , K } ( x )$ . Let $\mu _ { \mathrm { r e f } }$ be the distribution induced by a metadata-only teacher generator. Consider the KL-regularized update $( \eta > 0 )$

$$
\begin{array} { r l } & { \mu _ { \eta } \in \arg \underset { \mu \ll \mu _ { \mathrm { r e f } } } { \operatorname* { m a x } } \left\{ \mathbb { E } _ { x \sim \mu } \big [ u _ { \theta , K } ( x ) \big ] \right. } \\ & { \left. \qquad - \frac { 1 } { \eta } D _ { \mathrm { K L } } \big ( \mu \parallel \mu _ { \mathrm { r e f } } \big ) \right\} . } \end{array}\tag{17}
$$

Proposition A.2 The unique optimizer of (17) is

$$
\begin{array} { r } { \mu _ { \eta } ( x ) = \frac { \mu _ { \mathrm { r e f } } ( x ) \exp \big ( \eta u _ { \theta , K } ( x ) \big ) } { Z _ { \eta } } , } \\ { Z _ { \eta } = \mathbb { E } _ { x \sim \mu _ { \mathrm { r e f } } } \left[ \exp \big ( \eta u _ { \theta , K } ( x ) \big ) \right] . } \end{array}\tag{18}
$$

Proof By the calculus of variations, the stationarity condition of Eq. (17) gives

$$
u _ { \theta , K } ( x ) - \frac { 1 } { \eta } \left( \log \frac { \mu ( x ) } { \mu _ { \mathrm { r e f } } ( x ) } + 1 \right) + c = 0 ,\tag{19}
$$

where c enforces normalization. Rearranging yields

$$
\mu ( x ) \propto \mu _ { \mathrm { r e f } } ( x ) \exp { \left( \eta u _ { \theta , K } ( x ) \right) } ,\tag{20}
$$

which gives Eq. (18).

Eq. (18) is an idealized exponential reweighting of the practice distribution toward high-yield prompts. DIAG does not execute this update exactly, since estimating $u _ { \theta , K } ( x )$ for a large candidate space would be expensive. Instead, it approximates this reweighting using observable signals from the previous iteration: topic-level valid-pair statistics and local mistake contexts.

## A.3 Mistake-Conditioned Generation as an Amortized Approximation

Let z denote prompt metadata such as topic and difficulty, and let $\nu _ { t } ( z )$ be the scheduler over metadata at iteration t. A metadata-only generator induces

a kernel $T _ { 0 } ( x \mid z )$ , while the mistake-conditioned teacher induces

$$
T _ { \phi } ( x \mid z , m ) ,\tag{21}
$$

where m contains a failed question and a wrong student trace. In DIAG, mistake contexts are drawn from recent prompts that produced valid natural preference pairs. These prompts are empirical boundary cases: the student has both successful and failed rollouts on them. Degenerate all-correct or all-wrong prompts are counted in yield statistics and affect pacing, but are not used as exploitation anchors.

Let $q _ { t } ( m \mid z )$ be the empirical distribution over mistake contexts for topic z. The metadata-only and DIAG-induced practice distributions are

$$
\begin{array} { r } { \mu _ { 0 } ( x ) = \mathbb { E } _ { z \sim \nu _ { t } } \bigl [ T _ { 0 } ( x \mid z ) \bigr ] , } \end{array}\tag{22}
$$

$$
\mu _ { \mathrm { D I A G } } ( x ) = \mathbb { E } _ { z \sim \nu _ { t } } \mathbb { E } _ { m \sim q _ { t } ( \cdot | z ) } \big [ T _ { \phi } ( x \mid z , m ) \big ]\tag{23}
$$

For each metadata slice z, define the local yield gain

$$
\begin{array} { r l } & { \Delta _ { z } = \mathbb { E } _ { m \sim q _ { t } ( \cdot | z ) } \mathbb { E } _ { x \sim T _ { \phi } ( \cdot | z , m ) } \big [ u _ { \theta , K } ( x ) \big ] } \\ & { \qquad - \mathbb { E } _ { x \sim T _ { 0 } ( \cdot | z ) } \big [ u _ { \theta , K } ( x ) \big ] . } \end{array}\tag{24}
$$

Then, by linearity of expectation,

$$
J _ { \theta , K } ( \mu _ { \mathrm { D I A G } } ) - J _ { \theta , K } ( \mu _ { 0 } ) = \mathbb { E } _ { z \sim \nu _ { t } } [ \Delta _ { z } ] .\tag{25}
$$

Thus, DIAG improves expected valid-pair yield whenever mistake-conditioned generation produces less degenerate follow-up prompts than metadataonly generation on average. This condition is not assumed as a theorem about all teachers; rather, it is the empirical property evaluated through the yield and pass-rate analyses in the main text.

## A.4 Empirical-Bayes Stabilization

DIAG uses topic-level statistics to allocate the exploitation budget. For topic z, let $N _ { z }$ be the number of attempted prompts and $C _ { z }$ the number that yielded valid natural pairs. The raw rate $C _ { z } / N _ { z }$ can be noisy when $N _ { z }$ is small. The Empirical-Bayes score is

$$
v _ { z } = \frac { C _ { z } + \kappa \bar { p } } { N _ { z } + \kappa } = \frac { N _ { z } } { N _ { z } + \kappa } . \frac { C _ { z } } { N _ { z } } + \frac { \kappa } { N _ { z } + \kappa } . \bar { p } ,\tag{26}
$$

where $\bar { p }$ is the global valid-pair rate and κ is the prior strength. Thus $v _ { z }$ shrinks low-count topic estimates toward the global mean, preventing the scheduler from overreacting to a few lucky samples. As $N _ { z }$ grows, the topic-specific empirical rate dominates.

## A.5 Connection to DPO Curvature

The signal-yield view is also consistent with the local geometry of the DPO loss. For a pair $( x , y _ { w } , y _ { l } )$ let

$$
s _ { \theta } = h _ { \theta } ( x , y _ { w } ) - h _ { \theta } ( x , y _ { l } ) ,\tag{27}
$$

$$
\ell ( s _ { \theta } ) = - \log \sigma ( \beta s _ { \theta } ) .\tag{28}
$$

With respect to $a = \beta s _ { \theta }$ , the curvature is

$$
\frac { \partial ^ { 2 } \ell } { \partial a ^ { 2 } } = \sigma ( a ) \bigl ( 1 - \sigma ( a ) \bigr ) ,\tag{29}
$$

which is maximized at $a = 0$ . Hence, among valid preference pairs, updates are locally most sensitive when the model is uncertain between the preferred and dispreferred responses. This does not equate prompt-level pass probability with pairwise DPO uncertainty, but it supports the same design principle: practice near the student’s moving boundary tends to produce more informative supervision.

## B Complexity Analysis

We analyze computational cost through the lens of effective supervision. Let $C _ { \mathrm { r o l l } }$ be the unit cost of one student rollout and $C _ { \mathrm { t e a c h } }$ the unit cost of one teacher generation. At iteration t, we use a practice set $P _ { t }$ of size $| P _ { t } | = M$ and sample K rollouts per question, so the student-side cost per iteration is

$$
\mathcal { C } _ { t } ^ { \mathrm { s t u } } = M K C _ { \mathrm { r o l l } } .\tag{30}
$$

We focus on verifiable preference learning where a question x yields a usable DPO pair iff it contains both correct and incorrect rollouts $( \mathrm { E q . } ( 1 ) )$ . Recall the effective (valid-pair) yield is defined as

$$
\rho _ { t } \triangleq \frac { 1 } { M } \sum _ { x \in P _ { t } } \mathbb { I } _ { \mathrm { v a l i d } } ( x ) ,\tag{31}
$$

following Eq. (2). The expected number of effective training pairs in iteration t is

$$
\mathbb { E } [ | { \cal D } _ { t } | ] = { \cal M } \rho _ { t } .\tag{32}
$$

Compute-matched teacher budget. In our experiments, STATIC GEN and DIAG are matched by the same total number of teacher-generated questions. Concretely, over $T$ iterations both methods use TM teacher-generated questions, implying identical total teacher cost:

$$
{ \mathcal C } _ { \mathrm { t o t a l } } ^ { \mathrm { t e a c h } } \ = \ T M { \cal C } _ { \mathrm { t e a c h } } .\tag{33}
$$

The difference is not the amount of teacher compute, but how this fixed teacher budget is spent: STATIC GEN generates a fixed pool using only (topic, difficulty), while DIAG allocates the same number of generations online and conditions on the student’s recent failures to reshape the practice distribution.

Cost per effective pair (effective gradient cost). Since the student rollout budget per iteration is fixed, the key efficiency question is: how many of these rollouts produce gradient-bearing preference pairs? Define the per-iteration compute cost per effective pair as

$$
\begin{array} { l } { \displaystyle \mathcal { E } _ { t } \triangleq \frac { \mathcal { C } _ { t } ^ { \mathrm { s t u } } + \mathcal { C } _ { t } ^ { \mathrm { t e a c h } } } { \mathbb { E } [ | \mathcal { D } _ { t } | ] } = \frac { M K C _ { \mathrm { r o l l } } + \mathcal { C } _ { t } ^ { \mathrm { t e a c h } } } { M \rho _ { t } } } \\ { \displaystyle \qquad = \frac { K C _ { \mathrm { r o l l } } } { \rho _ { t } } + \frac { \mathcal { C } _ { t } ^ { \mathrm { t e a c h } } } { M \rho _ { t } } . } \end{array}\tag{34}
$$

Under compute matching, $\mathcal { C } _ { t } ^ { \mathrm { t e a c h } }$ is the same order for both methods (and the total teacher budget is identical), therefore the dominant factor is the yield $\rho _ { t }$ . When a static pool (or practice-agnostic generation) degenerates into all-correct or all-wrong regimes, $\rho _ { t }$ decreases and $\mathcal { E } _ { t }$ increases, meaning a growing fraction of rollouts are wasted with no contrastive supervision.

DIAG improves effective sample efficiency. By explicitly optimizing the practice distribution to stay near the student’s moving decision boundary, DIAG sustains a substantially higher $\rho _ { t }$ across iterations (Sec. 5). Plugging this into Eq. (34) yields a strictly lower compute-per-effective-pair:

$$
\begin{array} { r l } & { \rho _ { t } ^ { \mathrm { D I A G } } > \rho _ { t } ^ { \mathrm { S T A T I C ~ G E N } } } \\ & { \qquad \implies \mathcal { E } _ { t } ^ { \mathrm { D I A G } } < \mathcal { E } _ { t } ^ { \mathrm { S T A T I C ~ G E N } } . } \end{array}\tag{35}
$$

Equivalently, for the same fixed rollout budget MK, DIAG produces more effective preference pairs $M \rho _ { t }$ and therefore more gradient signal per unit compute. This explains why DIAG achieves higher final performance without increasing either student rollout compute or the total teachergeneration budget.

## C Empirical Bayes Derivation and Implementation Details

This appendix provides a brief derivation of the Empirical Bayes (EB) topic score used in Phase I and summarizes the corresponding implementation choices.

Beta–Binomial model. For each topic z, let $p _ { z }$ denote the latent probability that a sampled question from topic z yields a valid natural preference pair (Eq. 1). Given per-topic counts $( C _ { z } , N _ { z } )$ (Eq. 5), we model

$$
\begin{array} { c } { { p _ { z } \sim \mathrm { B e t a } ( \alpha _ { 0 } , \beta _ { 0 } ) , } } \\ { { C _ { z } \mid ( p _ { z } , N _ { z } ) \sim \mathrm { B i n o m i a l } ( N _ { z } , p _ { z } ) . } } \end{array}\tag{36}
$$

By conjugacy, the posterior is

$$
\begin{array} { r } { p _ { z } \mid ( C _ { z } , N _ { z } ) \sim \mathrm { B e t a } \big ( \alpha _ { 0 } + C _ { z } , \qquad } \\ { \beta _ { 0 } + N _ { z } - C _ { z } \big ) , } \end{array}\tag{37}
$$

and the posterior mean yields the EB score

$$
v _ { z } \triangleq \mathbb { E } [ p _ { z } \mid C _ { z } , N _ { z } ] = \frac { C _ { z } + \alpha _ { 0 } } { N _ { z } + \alpha _ { 0 } + \beta _ { 0 } } .\tag{38}
$$

Empirical Bayes prior from global statistics. We set the Beta prior using the global valid-pair rate p¯ and a smoothing strength κ (prior sample size):

$$
\alpha _ { 0 } \triangleq \kappa \bar { p } , \qquad \beta _ { 0 } \triangleq \kappa \big ( 1 - \bar { p } \big ) .\tag{39}
$$

Substituting into Eq. 38 gives the shrinkage estimator used in the main text:

$$
v _ { z } = \frac { C _ { z } + \kappa \bar { p } } { N _ { z } + \kappa } .\tag{40}
$$

When $N _ { z } \ll \kappa ,$ , the score shrinks towards ${ \bar { p } } ,$ preventing over-reaction to low-count noise; as $N _ { z }$ grows, the empirical rate $C _ { z } / N _ { z }$ dominates.

Implementation details. We compute $\bar { p }$ as the running global average of $\mathbb { I } _ { \mathrm { v a l i d } } ( x )$ over all attempted questions up to the current iteration (equivalently, $\begin{array} { r } { { \bar { p } } = \sum _ { z } C _ { z } / \sum _ { z } N _ { z } } \end{array}$ using the same statistics as Eq. 5). We fix κ as a small constant in all experiments, which controls the strength of shrinkage in the early, noisy regime.

## D Prompts

Prompt Templates. We present the prompt templates employed for the Diagnose-and-Generate loop in Figure 6 and Figure 5. For the teacher model, the instruction explicitly incorporates the Mistake Context, comprising the failed question and the erroneous solution trace, instructing the model to synthesize a variant that probes the same underlying misconception while altering surface features. Regarding student rollouts, we employ a standard format to elicit reasoning. To harmonize the reasoning structure across heterogeneous model architectures, we adopted an adaptive decoding strategy: for models requiring explicit structural guidance, the generation was primed with a sequential prefix (e.g., “Step 1:” (Lai et al., 2024)) to initiate the deduction process; for models with strong intrinsic instruction adherence, standard generation was utilized.

![](images/f144d0779be967f692e08fc3e035b784dc16bb769022fd0da6ff5b5d169d820b.jpg)  
Figure 5: Prompt for Student Model and Teacher Model Answer Generation.

## E Implementation Details

We provide the detailed hyperparameters and system configurations used in our experiments to ensure reproducibility.

## E.1 Training Configuration

All student models (Qwen2.5-Math-7B, Qwen3- 8B-Base, Llama-3.1-8B) were trained using the DPO objective with the AdamW optimizer, a cosine learning rate schedule with peak learning rate $5 \times 1 0 ^ { - 7 }$ , no weight decay, global batch size 128 (gradient accumulation 16), and 2 epochs per iteration. For DPO, we set the KL penalty $\beta = 0 . 1$ , use sigmoid loss with label smoothing 0.1, and freeze the previous iteration’s checkpoint as the reference policy. Sequences are truncated at 4096 tokens (prompts at 1000), and training uses bfloat16 precision with gradient checkpointing enabled.

Regarding data attributes, we employ a hierarchical taxonomy annotated by Gemini-2.5 (Comanici et al., 2025). This includes five primary domains: Algebra, Geometry, Calculus, Discrete Mathematics, and Probability. Each contains several fine-grained sub-topics (e.g., Polynomials, Modular Arithmetic), along with four difficulty levels ranging from Elementary to Expert. For mistake-conditioned generation, failure traces are sampled from the mistake buffer H<sub>t−1</sub> via an uncertainty-weighted probability proportional to $1 - \operatorname* { m a x } ( \hat { p } , 1 - \hat { p } )$ to prioritize boundary cases, which is standard in (Settles, 2009). The pacing factor λ in (3) is set to 0.5 by default.

## E.2 Generation & Rollout Configuration

The quality of the feedback loop relies heavily on the decoding strategies employed during the rollout phase.

• Student Rollouts: For collecting preference data, we sample K = 8 responses per prompt with temperature T = 1.0, top- ${ \cdot } p = 0 . 9 9$ , and top-k = 50 to ensure diversity in reasoning paths.

• Teacher Generation: For the targeted curriculum synthesis, the teacher model uses temperature T = 1.0 with top-p = 0.99 for question generation. For answer generation, we use greedy decoding $( T = 0 )$ to maximize correctness.

![](images/c4032a48264f74dfa6356e26e4206f85e5e841bcbb0f0c71a774007da6de439c.jpg)  
Figure 6: Prompt for Teacher Model Question Generation.

• Verifier: The verifier adopts rule-based evaluation (Zhang et al., 2025) to determine correctness.

## E.3 Cross-Task Evaluation Implementation Details

To ensure reproducibility, we adopt standard evaluation protocols for all benchmarks. For BBH and MMLU, we utilize deterministic greedy decoding (T = 0) with a single rollout. For TruthfulQA, we follow the standard MC1 and MC2 validation setup, calculating scores based on the log-probabilities of candidate answers. For HumanEval, we employ a sampling-based approach: we generate k = 20 solutions per problem with a temperature of $T = 0 . 1$ and report the average pass rate.

F Case Studies

## F.1 Ignoring Interior Singularities in Improper Integrals

A persistent failure mode among small models is mechanically applying the Fundamental Theorem of Calculus without checking whether the integrand is continuous on the integration interval. In the seed question below, the student is asked to evaluate $\textstyle \int _ { - 1 } ^ { 1 } 1 / x ^ { 2 }$ dx. The integrand has a singularity at $x = 0 .$ , making the integral divergent, but the student simply computes the antiderivative −1/x, substitutes the endpoints, and reports −2, which is a negative value for a non-negative integrand, an inconsistency that goes entirely unnoticed.

The follow-up shifts the singularity from $x = 0$ to x $\mathbf { \Sigma } = \mathbf { \Sigma } _ { 2 } \in \left( 1 , 4 \right)$ by asking the student to evaluate $\textstyle \int _ { 1 } ^ { 4 } 1 / ( x - 2 ) ^ { 2 } d x$ . This defeats the shallow heuristic “watch out when $x = 0$ is in the interval” while preserving the same underlying misconception. The naïve FTC evaluation now yields $- 3 / 2$ again negative for a strictly positive integrand, so even a minimal sign-check would catch the error. Students who have internalized the continuity prerequisite immediately output 0 (divergent); those who have not will consistently report −3/2, producing a clean binary split in rollout outcomes.

## F.2 Base-Rate Neglect in Bayesian Inference

The second case targets a different kind of oversight: ignoring prior probabilities when computing a posterior. Two machines have different production shares (60% vs. 40%) and defect rates (5% vs. 3%). The student is asked for $P ( A \mid$ defective) and normalizes the likelihoods directly, arriving at $5 / ( 5 + 3 ) = 5 / 8$ . The correct answer is $5 / 7$ obtained by weighting each likelihood by its prior. Notably, the student’s self-check $( 5 / 8 + 3 / 8 = 1 )$ is tautological, since any normalization sums to one, yet it gives a false sense of correctness.

To expose this misconception more sharply, the follow-up keeps the same defect rates but pushes the priors to an extreme: Machine A now produces only 1% of items while Machine B produces 99%. The likelihood-only shortcut yields the same $5 / 8 \approx 6 2 . 5 \%$ , but the correct Bayesian posterior drops to $0 . 0 0 0 5 / 0 . 0 3 0 2 \approx 1 . 6 6 \%$ , which is off by a factor of nearly $4 0 \times$ . The dramatic gap ensures that any model still ignoring the prior will produce an answer that is obviously wrong, rather than merely slightly off.

Evaluate the integral $\int _ { 1 } ^ { 4 } { \frac { 1 } { ( x - 2 ) ^ { 2 } } } d x$ . If the integral diverges, enter 0.   
Correct Answer: 0   
Trap answer via direct FTC: $\begin{array} { r } { \left[ - \frac { 1 } { x - 2 } \right] _ { 1 } ^ { 4 } = - \frac { 1 } { 2 } - 1 = - \frac { 3 } { 2 } . } \end{array}$

## Student Answer (Incorrect) – Improper Integral

Step 1: Find the antiderivative of $\textstyle { \frac { 1 } { x ^ { 2 } } }$ . We know that $\textstyle { \frac { 1 } { x ^ { 2 } } } = x ^ { - 2 }$ , so the antiderivative is $\textstyle { \frac { x ^ { - 1 } } { - 1 } } = - { \frac { 1 } { x } } .$

Step 2: Apply the Fundamental Theorem of Calculus. That is $\begin{array} { r } { \int _ { - 1 } ^ { 1 } { \frac { 1 } { x ^ { 2 } } } d x = \left[ - { \frac { 1 } { x } } \right] _ { - 1 } ^ { 1 } . } \end{array}$

Step 3: Evaluate at the bounds. Upper bound: $- { \frac { 1 } { 1 } } = - 1$ . Lower bound: $- { \frac { 1 } { - 1 } } = 1$

Step 4: Compute the result.

$$
\left[ - \frac { 1 } { x } \right] _ { - 1 } ^ { 1 } = ( - 1 ) - ( 1 ) = - 2 .
$$

The answer is $\boxed { - 2 } .$

## Teacher-Generated Follow-up (Targeted) – Improper Integral

## Student Answer (Incorrect) – Bayesian Inference

Step 1: Identify the given information. Machine A: defect rate = 5%. Machine B: defect rate =   
3%. We are told a defective item was found, and we need $P ( A \mid$ defective).   
Step 2: Since the item is already known to be defective, we need to determine which machine is   
more likely to have produced it. We can compare the defect rates directly. The probability of being   
defective given Machine A is 5%, and given Machine B is 3%.   
Step 3: Normalize to get the posterior.   
P(defective | A) 0.05 5   
P(A | defective) =   
P(defective | A) + P(defective | B) 0.05 + 0.03 8   
Step 4: Verify. $\textstyle { \frac { 5 } { 8 } } + \frac { 3 } { 8 } = 1$ The probabilities sum to 1, so the answer is consistent. The answer is   
${ \Bigg [ } { \frac { 5 } { 8 } } { \Bigg ] } .$

## Teacher-Generated Follow-up (Targeted) – Bayesian Inference

A factory has two machines. Machine A produces 1% of all items and has a 5% defect rate. Machine B produces 99% of all items and has a $3 \%$ defect rate. If a randomly selected item is found to be defective, what is the probability it was produced by Machine A?

Correct Answer: $\begin{array} { r } { \frac { 0 . 0 0 0 5 } { 0 . 0 3 0 2 } \approx 1 . 6 6 \% . } \end{array}$

Trap answer via likelihood-only normalization: $\begin{array} { r } { \frac { 5 } { 5 + 3 } = \frac { 5 } { 8 } = 6 2 . 5 \% . } \end{array}$

Figure 7: Case Studies. Top pair (Improper Integral): The student applies FTC across a singularity at $x = 0 ,$ arriving at −2; the follow-up shifts the singularity to $x = 2 ,$ where the trap value $- 3 / 2$ is negative for a strictly positive integrand. Bottom pair (Bayesian Inference): The student normalizes likelihoods without prior weighting $( 5 / 8$ instead of $5 / 7 ) ;$ the follow-up pushes priors to 1% vs. 99%, where the same shortcut gives 62.5% but the correct posterior is ≈ 1.66%.