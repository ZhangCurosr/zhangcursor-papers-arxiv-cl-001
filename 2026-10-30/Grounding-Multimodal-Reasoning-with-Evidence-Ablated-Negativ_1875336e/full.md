# Grounding Multimodal Reasoning with Evidence-Ablated Negatives

Chenxi Liu Department of Computer Science University of Maryland, College Park College Park, MD 20742 cxliu539@umd.edu

Yanshuo Chen Department of Computer Science University of Maryland, College Park College Park, MD 20742 cys@umd.edu

Tianyi Xiong Department of Computer Science University of Maryland, College Park College Park, MD 20742 txiong23@umd.edu

Heng Huang Department of Computer Science University of Maryland, College Park College Park, MD 20742 heng@umd.edu

## Abstract

Reinforcement learning with verifiable rewards (RLVR) has become a powerful tool for improving multimodal reasoning. However, standard outcome supervision suffers from credit assignment ambiguity, as a model may receive positive rewards by exploiting language priors or dataset shortcuts rather than grounding its reasoning in the requisite visual evidence. This makes high-quality negative responses critical, since effective negatives should remain plausible under the current policy while exposing the shortcut that should be suppressed. In this paper, we propose EVAN, a multimodal reinforcement learning framework utilizing EVidence-Ablated Negatives to train LMMs against their own language-prior failures. EVAN first identifies highly visually dependent examples using a scaled image information gain metric and preserves data diversity with a mixed-ranking strategy. It then constructs evidence-ablated images by masking the most discriminative image patches, eliciting plausible but visually ungrounded responses from the current policy. During policy optimization, these negative responses are incorporated as bounded auxiliary signals under the clean image, penalizing shortcut learning while maintaining RL stability. Experiments across challenging multimodal reasoning benchmarks demonstrate that EVAN mitigates modality imbalance and improves over strong RLVR baselines. Our code is available at https://github.com/DawnLIU35/EVAN.git.

## 1 Introduction

Large Multimodal Models (LMMs) have made substantial progress by coupling vision encoders with pre-trained Large Language Models (LLMs), enabling strong performance on diverse visual understanding and reasoning tasks [Liu et al., 2023, 2024a, Bai et al., 2025b, Tong et al., 2024, Chen et al., 2024b, OpenAI, 2023, Xiong et al., 2024, Chen et al., 2023, 2024a, Xiong et al., 2026a,c]. More recently, Reinforcement Learning with Verifiable Rewards (RLVR), often instantiated with Group Relative Policy Optimization (GRPO) [Shao et al., 2024], has emerged as a promising posttraining paradigm for improving model reasoning through online rollouts and task-level correctness signals [Guo et al., 2025, Chen et al., 2025, Shen et al., 2025, Zheng et al., 2025, Liu et al., 2026, 2025a, Jia et al., 2026]. However, verifiable rewards typically assess only whether the final answer is correct, without indicating whether a correct prediction is genuinely grounded in the visual input or instead driven by language priors, dataset shortcuts, or spurious question–answer correlations. As a result, LMMs can still suffer from modality imbalance, where the language backbone dominates the reasoning process while visual evidence is underutilized [Liu et al., 2024b, Jiang et al., 2024]. This failure mode is especially problematic for visual reasoning, since a model may obtain correct rewards during training yet remain weakly grounded in the input image, leading to brittle perception and hallucinated responses [Yu et al., 2024, Zhao et al., 2023].

A natural way to address modality imbalance is to intervene on the visual input and test whether the model’s behavior changes accordingly. Prior work perturbs, masks, or contrasts images to reduce language bias in visual question answering [Niu et al., 2021, Liang et al., 2020], and recent multimodal RL methods use perturbed images to estimate visual dependency or adjust policy updates [Wang et al., 2025b, Huang et al., 2025a, Liu et al., 2025c]. These studies highlight the value of visual interventions, but they do not necessarily produce informative negative samples for RLVR. Most interventions are still coarse and task-agnostic. They perturb images without first selecting examples whose solutions strongly require visual evidence, and the perturbations are not optimized to expose the model’s most plausible language-prior failure for the given query. Consequently, the induced negatives can be too easy, out of distribution, or only weakly connected to the shortcut that causes modality imbalance. This limitation is important because recent analyses of RLVR show that penalizing incorrect responses can itself provide a strong learning signal, and that plausible negative trajectories are more useful than arbitrary failures [Zhu et al., 2025, Di et al., 2026]. These findings shift the focus from simply perturbing images to constructing high-quality negatives. This motivates a targeted approach that first identifies visually dependent examples and then ablates query-critical visual evidence to elicit hard, plausible, and visually ungrounded negative responses for policy optimization.

In this paper, we propose EVAN, a multimodal reinforcement learning framework that learns from EVidence-Ablated Negatives to encourage LMMs to ground their reasoning in query-critical visual evidence. The key idea is to expose the model’s language-prior shortcuts only on examples where visual evidence is truly necessary. To this end, EVAN first estimates scaled image information gain (scaled-IIG) for each training example and applies a mixed-ranking strategy to select visually dependent samples while preserving data diversity. For each selected sample, EVAN identifies the image patches that most support the verified response and constructs an evidence-ablated image by masking these query-critical patches. The evidence-ablated image is then used to sample plausible but visually ungrounded responses from the current policy, which serve as hard negative responses. During policy optimization, EVAN incorporates these online evidence-ablated negatives together with clean-image rollouts. The evidence-ablated negatives are optimized under the clean image, so the policy learns to suppress language-prior failures when the relevant visual evidence is available. To maintain RL stability, EVAN filters out low-probability or overly long evidence-ablated responses and assigns bounded negative advantages, preventing uninformative negatives from dominating policy updates. Experiments on challenging visual reasoning benchmarks show that EVAN improves multimodal reasoning performance and yields more visually grounded behavior.

Overall, our contributions can be summarized as follows:

• We propose EVAN, a multimodal reinforcement learning framework that identifies visually dependent examples with scaled-IIG and constructs evidence-ablated images by masking querycritical patches. These images elicit plausible but visually ungrounded responses from the current policy.

• We introduce a stable training procedure for EVAN that incorporates online evidence-ablated negatives into RLVR. The negatives are filtered for clean-image plausibility and optimized under the clean image with bounded negative advantages.

• We evaluate EVAN on challenging multimodal reasoning benchmarks and show that it improves over strong RLVR baselines. Ablations and further analyses validate the roles of visually dependent example selection, evidence-ablated image construction, and stable negative optimization.

## 2 Related Work

RLVR for Multimodal Reasoning. Reinforcement learning has become a central post-training tool for improving reasoning and alignment in large language models [Ouyang et al., 2022, Jaech et al., 2024, Guo et al., 2025, Team et al., 2025]. Recent work extends Reinforcement Learning with Verifiable Rewards (RLVR), often instantiated with GRPO [Shao et al., 2024], to multimodal reasoning tasks such as visual math reasoning and academic question answering [Chen et al., 2025, Shen et al., 2025, Zheng et al., 2025, Meng et al., 2025, Huang et al., 2025b, Peng et al., 2025, Yang et al., 2025, Yu et al., 2025a, Zhan et al., 2025, Liu et al., 2025d]. These methods mainly optimize final-answer correctness from online rollouts. However, outcome rewards do not reveal whether a correct answer is grounded in visual evidence or produced from language priors. Recent studies further show that negative responses can provide important RLVR signals, especially when they are plausible rather than arbitrary failures [Zhu et al., 2025, Di et al., 2026]. EVAN is complementary to standard multimodal RLVR. It preserves clean-image policy optimization while introducing online evidence-ablated negatives that expose modality imbalance.

Visual Interventions for Modality Imbalance. Large Multimodal Models often inherit strong linguistic priors from pre-trained language backbones [Liu et al., 2023, 2024a, Bai et al., 2025b, Tong et al., 2024, Chen et al., 2024b]. This can lead to modality imbalance, where the model answers from language priors or dataset shortcuts instead of visual evidence [Liu et al., 2024b, Jiang et al., 2024, Xiong et al., 2026b]. Prior work studies this issue through hallucination analysis, counterfactual VQA, and visual grounding evaluation [Zhao et al., 2023, Yu et al., 2024, Niu et al., 2021]. Other methods intervene on images by masking, corrupting, cropping, or contrasting visual inputs to reduce language bias or construct dispreferred responses [Liang et al., 2020, Pi et al., 2024, Zhou et al., 2024, Wang et al., 2024a]. Recent perception-aware multimodal RL methods also use perturbed images to regularize policy updates [Wang et al., 2025b, Huang et al., 2025a, Liu et al., 2025c,b] or impose KL constraints between original and code-edited chart images [Zhang et al., 2026]. However, these signals may not effectively reflect the language-prior failures of the online policy alone training, or are limited to specific data such as charts. In contrast, EVAN first selects visually dependent examples with scaled-IIG and then constructs robust evidence-ablated images by masking query-critical patches. The resulting images elicit hard negative responses from the current policy, which are filtered and optimized under the clean image with bounded negative advantages.

## 3 Preliminaries

Reinforcement Learning. Group Relative Policy Optimization (GRPO) [Shao et al., 2024, Guo et al., 2025] estimates advantages from a group of sampled responses without training an additional critic. Given a query $q ,$ the behavior policy $\pi _ { \theta _ { \mathrm { o l d } } }$ generates a group of G candidate responses $\{ o _ { i } \} _ { i = 1 } ^ { G } .$ Each response receives a reward $R _ { i } .$ , and its advantage is computed by centering and normalizing the rewards within the group:

$$
\hat { A } _ { i } = \frac { R _ { i } - \operatorname* { m e a n } ( \{ R _ { i } \} _ { i = 1 } ^ { G } ) } { \mathrm { s t d } ( \{ R _ { i } \} _ { i = 1 } ^ { G } ) } .\tag{1}
$$

All tokens of $o _ { i }$ share this response-level advantage. We further adopt Decoupled Clip and Dynamic sAmpling Policy Optimization (DAPO) [Yu et al., 2025b] for stable long-form reasoning optimization. DAPO uses asymmetric clipping to control policy updates and applies dynamic sampling to keep only groups that contain both correct and incorrect responses. For readability, we omit the image input in the following preliminary objective and restore it in the multimodal formulation of EVAN. The resulting token-level objective is:

$$
\begin{array} { r l } { \mathcal { T } _ { \mathrm { { D A P O } } } ( \theta ) = } & { \mathbb { E } _ { ( q , o ^ { * } ) \sim Q , \{ o _ { i } \} _ { i = 1 } ^ { G } \sim \pi _ { \theta _ { \mathrm { o l d } } } ( \cdot | q ) } \left[ \frac { 1 } { \sum _ { i = 1 } ^ { G } \left| o _ { i } \right| } \displaystyle \sum _ { i = 1 } ^ { G } \sum _ { t = 1 } ^ { \left| o _ { i } \right| } \right. } \\ & { \left. \operatorname* { m i n } \left( r _ { i , t } ( \theta ) \hat { A } _ { i } , \mathrm { { \ c l i p } } \Bigl ( r _ { i , t } ( \theta ) , 1 - \varepsilon _ { \mathrm { l o w } } , 1 + \varepsilon _ { \mathrm { h i g h } } \Bigr ) \hat { A } _ { i } \right) \right] } \\ & { \mathrm { ~ s . t . ~ } 0 < \left| \{ o _ { i } \ | \mathrm { \bf ~ i s } _ { - } \mathrm { e q u i v a l e n t } ( o ^ { * } , o _ { i } ) \} \right| < G , } \end{array}\tag{2}
$$

where $\begin{array} { r } { r _ { i , t } ( \theta ) = \frac { \pi _ { \theta } \left( o _ { i , t } | { q , o _ { i , < t } } \right) } { \pi _ { \theta _ { \mathrm { o l d } } } \left( o _ { i , t } | { q , o _ { i , < t } } \right) } } \end{array}$ denotes the token-level importance ratio, and $o ^ { * }$ is the ground-truth answer used for reward computation.

## 4 Methodology

EVAN is designed to mitigate modality imbalance by training LMMs against hard negative responses that reveal language-prior shortcuts. The framework contains two stages. First, we mine visually dependent examples and construct evidence-ablated images by masking query-critical visual patches. These images selectively weaken the evidence needed to support the verified response, prompting the current policy to generate plausible but visually ungrounded responses. Second, we incorporate the resulting online evidence-ablated negatives into GRPO-based policy optimization with stability controls. The negatives are filtered and optimized under the clean image with bounded negative advantages, allowing the model to suppress shortcut-driven responses while preserving standard clean-image rollout optimization. The full procedure of EVAN is summarized in Algorithm 1.

Algorithm 1 EVAN: Learning from EVidence-Ablated Negatives   
Require: Training set $\mathcal D = \{ ( q _ { j } , I _ { j } , o _ { j } ^ { * } ) \} _ { j = 1 } ^ { N }$ , policy $\pi _ { \theta } ,$ selection ratio $\tau ,$ mask ratio $\rho$   
Ensure: Updated policy $\pi \theta$   
1: Stage 1: Evidence-ablated image construction   
2: for each example $( q _ { j } , I _ { j } , o _ { j } ^ { * } ) \in \bar { \mathcal { D } }$ do   
3: Compute the visual-dependency score $s _ { j } = \mathrm { s I I G } ( o _ { j } ^ { * } , q _ { j } , I _ { j } )$   
4: end for   
5: Select a visually dependent subset $S _ { \mathrm { a d v } }$ with RankMix using ratio $\tau .$   
6: for each selected example $j \in  { S _ { \mathrm { a d v } } }$ do   
7: Compute patch scores from the absolute gradient of $\mathrm { _ { 3 I I G } } ( o _ { j } ^ { * } , q _ { j } , I _ { j } )$ with respect to image patches.   
8: Mask the top $\rho$ fraction of patches to construct the evidence-ablated image $I _ { j } ^ { \mathrm { a d v } }$   
9: end for   
10: Form $\mathcal { D } _ { \mathrm { t r a i n } } = \mathcal { D } \cup \{ ( q _ { j } , I _ { j } , I _ { j } ^ { \mathrm { a d v } } , o _ { j } ^ { * } ) \mid j \in S _ { \mathrm { a d v } } \}$   
11: Stage 2: Stable policy optimization   
12: for each training step do   
13: Set $\pi _ { \theta _ { \mathrm { o l d } } }  \pi _ { \theta }$ and sample a mini-batch $B \subseteq { \mathcal { D } } _ { \mathrm { t r a i n } }$   
14: for each example $j \in B$ do   
15: Generate clean rollouts $\mathcal { O } _ { j } ^ { c } = \{ o _ { j , i } ^ { c } \} _ { i = 1 } ^ { G } \sim \pi _ { \theta _ { \mathrm { o l d } } } ( \cdot \mid q _ { j } , I _ { j } ) .$   
16: Compute rewards and DAPO advantages $\{ \hat { A } _ { j , i } ^ { c } \} _ { i = 1 } ^ { G }$ for the clean rollout group.   
17: Initialize the clean-image update set $\mathcal { U } _ { j } \gets \{ ( o _ { j , i } ^ { c } , \hat { A } _ { j , i } ^ { c } , I _ { j } ) \} _ { i = 1 } ^ { G }$   
18: if an evidence-ablated image $I _ { j } ^ { \mathrm { a d v } }$ is available then   
19: Generate evidence-ablated rollouts $\widetilde { \mathcal { O } } _ { j } ^ { a } = \{ \tilde { o } _ { j , k } ^ { a } \} _ { k = 1 } ^ { M } \sim \pi _ { \theta _ { \mathrm { o l d } } } ( \cdot \vert q _ { j } , I _ { j } ^ { \mathrm { a d v } } ) ,$   
20: Randomly select one incorrect evidence-ablated rollout as the candidate $o _ { j } ^ { a }$ , if any exists   
21: if $o _ { j } ^ { a }$ passes the clean-image likelihood and response length filters then   
22: Assign $o _ { j } ^ { a }$ the bounded negative advantage $\begin{array} { r } { \hat { A } _ { j } ^ { a } = - \operatorname* { m i n } _ { i } | \hat { A } _ { j , i } ^ { c } | . } \end{array}$   
23: Add it to the clean-image update set: $\mathcal { U } _ { j }  \mathcal { U } _ { j } \cup \{ ( o _ { j } ^ { a } , \hat { A } _ { j } ^ { a } , I _ { j } ) \}$   
24: end if   
25: end if   
26: end for   
27: Update π<sub>θ</sub> on $\begin{array} { r } { \mathcal { U } = \bigcup _ { j \in \mathcal { B } } \mathcal { U } _ { j } } \end{array}$ with the DAPO objective.   
28: end for

## 4.1 Evidence-ablated Data Mining

The goal of evidence-ablated data mining is to identify examples whose verified responses require visual evidence and construct evidence-ablated images that expose language-prior failures. Applying visual interventions uniformly to all samples is inefficient. Different examples require different degrees of visual information, and perturbing weakly visual-dependent samples may provide little useful training signal. Meanwhile, overly destructive interventions may simply make the image uninformative and produce off-distribution failures. EVAN therefore follows a targeted procedure. It first estimates the visual dependency of each example, then selects a category-balanced subset with high dependency, and finally masks query-critical patches that suppress the same dependency signal under the original query.

Scaled image information gain. We use scaled image information gain (scaled-IIG), denoted by sIIG in equations, to measure how much a response relies on the image. Let $\boldsymbol { \mathcal { D } } = \{ ( q _ { j } , I _ { j } , o _ { j } ) \} _ { j = 1 } ^ { N }$ be the candidate pool, where $q _ { j }$ is the query, $I _ { j }$ is the image, and $o _ { j }$ is the response used for scoring.

For a response o, we first compute its average log-likelihood under image I:

$$
\ell _ { \theta } ( o ; q , I ) = { \frac { 1 } { | o | } } \sum _ { t = 1 } ^ { | o | } \log \pi _ { \theta } ( o _ { t } \mid q , I , o _ { < t } ) .\tag{3}
$$

We then define the visual-dependency score of example $j$ as the scaled likelihood difference between the original image and a blank image $I _ { \varnothing } { \mathrm { : } }$

$$
s _ { j } = \mathrm { s I I G } ( o _ { j } , q _ { j } , I _ { j } ) = \frac { \ell _ { \theta } ( o _ { j } ; q _ { j } , I _ { j } ) - \ell _ { \theta } ( o _ { j } ; q _ { j } , I _ { \varnothing } ) } { - \ell _ { \theta } ( o _ { j } ; q _ { j } , I _ { \varnothing } ) + \eta } .\tag{4}
$$

Here η is a small numerical stabilizer. A larger $s _ { j }$ means that replacing the image with a blank input causes a larger relative drop in response likelihood. Thus, a high score indicates that the response is supported by visual evidence rather than mainly by language priors. This score is model-centered: it measures how much the LMM actually relies on the image, rather than assuming visual dependency from the task category or dataset annotation.

Balanced selection with RankMix. Selecting only the globally highest scaled-IIG examples can reduce data diversity. The top-ranked samples may concentrate in a few categories where blank-image likelihood drops are especially large. To preserve both visual dependency and category coverage, we use RankMix, which combines global and category-wise rankings.

Let $c _ { j }$ denote the category of example $j ,$ let $\mathcal { T } = \{ 1 , \ldots , N \}$ be the full index $\operatorname { s e t } ,$ , and let $\mathcal { T } _ { c } = \{ j$ $c _ { j } = c \}$ be the index set for category c. For any index set A, let $\mathrm { T o p } _ { \tau } ( \mathcal { A } ; s )$ return the top τ fraction of examples in $\mathcal { A }$ ranked by scaled-IIG score $s .$ . RankMix first forms a core set of examples that are strong under both global and category-wise rankings:

$$
\begin{array} { r } { \mathcal { C } = \left\{ j \in \mathbb { Z } \mid j \in \mathrm { T o p } _ { \tau } ( \mathbb { Z } ; s ) \mathrm { a n d } j \in \mathrm { T o p } _ { \tau } ( \mathbb { Z } _ { c _ { j } } ; s ) \right\} . } \end{array}\tag{5}
$$

This criterion keeps examples that are globally high-dependency while also being competitive within their own categories. If the core set is smaller than the target size, RankMix fills the remaining slots using a mixed rank score. Let $r ( j ; A )$ be the descending scaled-IIG rank of example $j$ within an index set ${ \mathcal { A } } .$ , with smaller ranks indicating larger scores. The fill score is defined as

$$
m _ { j } = \left( 1 - \frac { r ( j ; \mathcal { T } ) } { | \mathcal { T } | } \right) \left( 1 - \frac { r ( j ; \mathcal { T } _ { c _ { j } } ) } { | \mathcal { T } _ { c _ { j } } | } \right) .\tag{6}
$$

A larger $m _ { j }$ indicates that the example ranks highly both globally and within its category. The final selected set is formed by $\mathcal { C }$ plus the highest-scoring remaining examples according to $m _ { j }$ until the target size is reached. Since $m _ { j }$ ranks every example outside ${ \mathcal { C } } ,$ a category with no example in $\mathcal { C }$ can still contribute examples that rank highly within that category. In this way, RankMix keeps the mining pool focused on high visual dependency while avoiding diversity collapse.

Scaled-IIG-guided adversarial masking. After selecting visually dependent examples, we construct evidence-ablated images by masking the regions that most affect the same scaled-IIG signal used for selection. The goal is not to make the whole image unreadable. Instead, the intervention should remove high-influence visual evidence in a localized patch-level manner while keeping the textual query unchanged. For a selected example, let $o _ { j } ^ { * }$ denote a verified correct target response. We first compute how sensitive the scaled-IIG score is to each visual patch. Let $\mathcal { P } _ { j }$ be the set of non-overlapping patches that partition the image, and let $P _ { j , r } \in \mathcal { P } _ { j }$ denote the r-th patch. For each patch, we aggregate the absolute scaled-IIG gradient:

$$
a _ { j , r } = \sum _ { u \in P _ { j , r } } \left. \frac { \partial \operatorname { s I I G } ( o _ { j } ^ { * } , q _ { j } , I _ { j } ) } { \partial I _ { j } ( u ) } \right. .\tag{7}
$$

Here, u indexes a pixel location in $I _ { j }$ , and $I _ { j } ( u )$ denotes the corresponding pixel value. We then select the top $\rho$ fraction of patches by $a _ { j , i }$ <sub>r</sub> and set their pixels to black:

$$
I _ { j } ^ { \mathrm { a d v } } ( u ) = \left\{ \begin{array} { l l } { 0 , } & { u \in \bigcup _ { P \in M _ { j } } P , \quad } \\ { I _ { j } ( u ) , } & { \mathrm { o t h e r w i s e } , } \end{array} \right. \ \mathcal { M } _ { j } = \mathrm { T o p } _ { \rho } ( \mathcal { P } _ { j } ; a ) .\tag{8}
$$

This masking operation aligns the image intervention with the mining criterion. Rather than applying generic corruption or relying on fragile pixel-level perturbations, it uses scaled-IIG to localize the visual evidence most responsible for a valid target response and removes that evidence explicitly. Because the intervention operates at the patch level, it provides an evidence ablation while still targeting the evidence used by the current model. Conditioned on $I _ { j } ^ { \mathrm { a d v } }$ , the model is more likely to fall back on high-probability textual patterns and dataset priors under the same query.

We store the evidence-ablated image together with its corresponding clean example:

$$
{ \mathcal { D } } _ { \mathrm { a d v } } = \{ ( q _ { j } , I _ { j } , I _ { j } ^ { \mathrm { a d v } } ) \mid j \in S _ { \mathrm { a d v } } \} ,\tag{9}
$$

where $S _ { \mathrm { a d v } }$ denotes the RankMix-selected index set. The prepared evidence-ablated entries are appended to the original training set and later used to sample online evidence-ablated negatives:

$$
\mathcal { D } _ { \mathrm { t r a i n } } = \mathcal { D } \cup \mathcal { D } _ { \mathrm { a d v } } .\tag{10}
$$

## 4.2 Stable Policy Optimization

Evidence-ablated images are used to elicit hard negative responses that reveal language-prior shortcuts. However, these responses are sampled under weakened visual evidence, so directly treating them as ordinary clean-image on-policy rollouts can introduce unstable gradients. EVAN addresses this by using evidence-ablated rollouts as bounded auxiliary signals. The clean-image rollout group follows the standard DAPO update, while examples with prepared evidence-ablated images receive an additional evidence-ablated branch. Only one filtered incorrect response from this branch is appended to the actor update.

Evidence-ablated image rollout. For a selected example $( q _ { j } , I _ { j } , o _ { j } ^ { * } )$ with evidence-ablated image $I _ { j } ^ { \mathrm { a d v } }$ , EVAN samples an auxiliary rollout group under the evidence-ablated visual condition:

$$
\widetilde { \mathcal { O } } _ { j } ^ { a } = \{ \tilde { o } _ { j , k } ^ { a } \} _ { k = 1 } ^ { M } , \quad \tilde { o } _ { j , k } ^ { a } \sim \pi _ { \theta _ { \mathrm { o l d } } } ( \cdot \vert q _ { j } , I _ { j } ^ { \mathrm { a d v } } ) .\tag{11}
$$

We retain only evidence-ablated rollouts that have incorrect final answers:

$$
\mathcal { O } _ { j } ^ { a } = \left\{ \tilde { o } _ { j , k } ^ { a } \in \widetilde { \mathcal { O } } _ { j } ^ { a } \mid \mathrm { i } \mathbf { s } _ { - } \mathrm { e q u i v a l e n t } ( o _ { j } ^ { * } , \tilde { o } _ { j , k } ^ { a } ) = 0 \right\} .\tag{12}
$$

Since the visual evidence supporting the verified response has been weakened, these incorrect responses serve as candidate language-prior failures under the same query.

Rollout filtering. A wrong response generated under the evidence-ablated image is useful only if it remains plausible under the clean visual context. If it is far outside the clean-image policy distribution, penalizing it may introduce noisy gradients rather than correct a realistic shortcut. We therefore filter evidence-ablated candidates by comparing their clean-image likelihood with the clean rollout group. Let $\mathcal { O } _ { i } ^ { c }$ denote the clean rollout group, and let $\ell _ { \theta _ { \mathrm { o l d } } } ( o ; q , I )$ denote the token-mean log probability defined above. The evidence-ablated candidates that pass the filter are

$$
\mathcal { O } _ { j , \mathrm { k e e p } } ^ { a } = \left\{ o ^ { a } \in O _ { j } ^ { a } \mid \ell _ { \theta _ { \mathrm { o l d } } } ( o ^ { a } ; q _ { j } , I _ { j } ) > \operatorname* { m i n } _ { o ^ { c } \in O _ { j } ^ { c } } \ell _ { \theta _ { \mathrm { o l d } } } ( o ^ { c } ; q _ { j } , I _ { j } ) , \quad | o ^ { a } | \leq \operatorname* { m a x } _ { o ^ { c } \in O _ { j } ^ { c } } | o ^ { c } | \right\} .\tag{13}
$$

This relative criterion keeps evidence-ablated negatives that are more likely than the weakest clean rollout under the clean image. The length constraint removes evidence-ablated negatives that are longer than all clean rollouts for the same prompt, keeping appended negatives comparable to the clean rollout group. For each prompt with a non-empty $\bar { \mathcal { O } } _ { j } ^ { a }$ , we randomly select one candidate $o _ { i } ^ { a }$ from $\mathcal { O } _ { j } ^ { a }$ and append it only if $\mathbf { \bar { \Phi } } _ { O _ { j } ^ { a } } \in \mathbf { \bar { O } } _ { j , \mathrm { k e e p } } ^ { a }$ . Otherwise, no evidence-ablated negative is appended, and no other candidate is tried. This ensures that each training example contributes at most one online evidence-ablated negative, keeping the auxiliary signal bounded.

Bounded negative advantage. The selected evidence-ablated negative response is appended to the actor batch, but it does not change the clean group statistics. Let $\{ \hat { A } _ { j , i } ^ { c } \} _ { i = } ^ { G }$ be the DAPO advantages of the clean rollout group. We assign the evidence-ablated negative a bounded negative advantage:

$$
\hat { A } _ { j } ^ { a } = - \operatorname* { m i n } _ { 1 \leq i \leq G } \left| \hat { A } _ { j , i } ^ { c } \right| .\tag{14}
$$

This design uses the clean rollout group as a proxy for the current sample’s uncertainty. Higher outcome variance among clean rollouts indicates greater sample-level uncertainty and warrants a stronger auxiliary signal, whereas low outcome variance suggests that the model’s behavior on this sample is already stable, so EVAN applies only a mild correction to avoid dominating the update. Appendix A.3 illustrates this behavior under binary rewards.

The retained evidence-ablated negative is appended to the clean rollout group and optimized in the same actor update. Although o<sup>a</sup> is generated from the evidence-ablated image $I _ { j } ^ { \mathrm { a d v } }$ , it is paired with the clean image $I _ { j }$ during optimization, which is the same visual condition used for clean rollouts. Its importance ratio is therefore computed as

$$
r _ { j , t } ^ { a } ( \theta ) = \frac { \pi _ { \theta } \big ( o _ { j , t } ^ { a } \mid q _ { j } , I _ { j } , o _ { j , < t } ^ { a } \big ) } { \pi _ { \theta _ { \mathrm { o l d } } } \big ( o _ { j , t } ^ { a } \mid q _ { j } , I _ { j } , o _ { j , < t } ^ { a } \big ) } .\tag{15}
$$

Clean-image responses use their original DAPO advantages, while the appended evidence-ablated negative uses the bounded negative advantage $\hat { A } _ { i } ^ { a }$ . This formulation penalizes a plausible but visually ungrounded response under the clean visual context, where the relevant evidence is available. It preserves standard clean-image policy optimization while adding a bounded response-level signal against shortcut learning.

## 5 Experiments

In this section, we first introduce the implementation details, including training details, datasets, evaluation protocol and baseline methods. Subsequently, we present our main results comparing EVAN with baseline methods on several visual reasoning benchmarks, demonstrating the effectiveness of EVAN. In addition, the ablation study provides a closer look at EVAN and verifies the contributions of its individual components. Lastly, we include additional experimental results for further analysis.

## 5.1 Implementation Details

Training details. Following recent studies that apply GRPO to train LMMs [Wang et al., 2025b, Huang et al., 2025a], we adopt Qwen2.5-VL-3B-Instruct, Qwen2.5-VL-7B-Instruct [Bai et al., 2025b], and Qwen3-VL-8B-Instruct [Bai et al., 2025a] as backbone models and train them on the ViRL39K dataset [Wang et al., 2025a]. We train each model for 2 epochs with a learning rate of $1 \times 1 0 ^ { - 6 }$ and a prompt-generation batch size of 256, generating 8 responses per prompt. We set the maximum response length to 4096 for both the 3B and 7B models, and to 8192 for the 8B model. To construct evidence-ablated images, we select $\tau = 2 0 \%$ of the training set and mask 50% of the most influential patches using 16 × 16 patches. All experiments are conducted using the verl framework [Sheng et al., 2024]. Additional details are provided in the Appendix.

Baselines: We compare our models against a comprehensive set of state-of-the-art open-source reasoning LMMs with publicly released model weights, including VL-Rethinker [Wang et al., 2025a], R1-ShareVL [Yao et al., 2025], NoisyRollout [Liu et al., 2025c], PAPO [Wang et al., 2025b], BiPS [Zhang et al., 2026], and VPPO [Huang et al., 2025a]. We also train the base model with DAPO and use it as our baseline.

Evaluation benchmark: Following recent multimodal RL studies [Wang et al., 2025b], our evaluation benchmark covers mathematical and geometric reasoning, including Geometry3K [Lu et al., 2021], MathVista [Lu et al., 2023], MathVerse [Zhang et al., 2024], We-Math [Qiao et al., 2025], DynaMath [Zou et al., 2024], MathVision [Wang et al., 2024b], and MMK12 [Meng et al., 2025]; multi-disciplinary multimodal reasoning with MMMU-Pro [Yue et al., 2025]; and logical reasoning with LogicVista [Xiao et al., 2024]. We use exact-match evaluation against ground-truth answers and exclude datasets or instances with free-form answers that require LLM-as-a-judge evaluation. For all benchmarks, we report average accuracy@8 with an inference temperature of 1.0, using a single fixed evaluation pipeline for all models, where accuracy@8 averages the pass@1 accuracy over eight independently sampled responses per question.

## 5.2 Benchmark Comparisons

Table 1 presents the main comparison with open-source reasoning LMMs. Under the same Qwen2.5- VL backbones, EVAN improves over the DAPO baseline from 50.3 to 51.6 average accuracy at the 3B scale and from 56.5 to 57.9 at the 7B scale. This controlled comparison isolates the effect of evidence-ablated negatives, since EVAN keeps the clean-image RLVR pipeline while adding only a bounded auxiliary signal from evidence-ablated rollouts. The gains are also stable across scales: EVAN improves DAPO on eight of nine benchmarks for 3B and on most benchmarks for 7B. On Qwen3- VL-8B, EVAN also improves its matched DAPO baseline from 71.9 to 73.0 average accuracy and outperforms it on eight of nine benchmarks, indicating that the gains extend to another model family. Compared with prior open-source reasoning LMMs, EVAN achieves the best average performance in each size group, including a 0.9 point improvement over the strongest 7B baseline, VPPO-7B, and a 1.8 point improvement over VPPO-8B. These results support our main claim that plausible negatives induced by removing query-critical visual evidence provide useful training signal beyond standard outcome supervision, helping the policy reduce language-prior shortcuts while preserving strong clean-image reasoning performance.

Table 1: Comparison with state-of-the-art open-source reasoning LMMs on multimodal reasoning benchmarks. All methods are evaluated under the same evaluation configuration. The best performance is marked in bold.
<table><tr><td>Model</td><td>Geometry3K</td><td>MathVista</td><td>MathVerse</td><td>We-Math</td><td>DynaMath</td><td>MathVision</td><td>MMK12</td><td>MMMU-Pro</td><td>LogicVista</td><td>Avg.</td></tr><tr><td colspan="9">Qwen2.5-VL-3B-Instruct</td></tr><tr><td>PAPO-3B</td><td>36.9</td><td>63.3</td><td>60.5</td><td>62.4</td><td>57.2</td><td>26.2</td><td>62.9</td><td>28.4</td><td>43.2</td><td>49.0</td></tr><tr><td>Qwen2.5-VL-3B</td><td>19.4</td><td>46.3</td><td>33.5</td><td>29.9</td><td>40.4</td><td>18.2</td><td>36.3</td><td>19.8</td><td>32.1</td><td>30.7</td></tr><tr><td>+DAPO</td><td>38.8</td><td>62.4</td><td>60.4</td><td>60.5</td><td>59.3</td><td>26.5</td><td>72.2</td><td>29.9</td><td>43.0</td><td>50.3</td></tr><tr><td>+EVAN (ours)</td><td>39.9</td><td>62.7</td><td>62.8</td><td>64.6</td><td>60.6</td><td>27.6</td><td>71.5</td><td>31.0</td><td>43.3</td><td>51.6</td></tr><tr><td colspan="9">Qwen2.5-VL-7B-Instruct</td></tr><tr><td>VL-Rethinker-7B</td><td>39.5</td><td>61.0</td><td>67.7</td><td>67.7</td><td>61.1</td><td>30.3</td><td>64.5</td><td>36.5</td><td>45.5</td><td>52.6</td></tr><tr><td>R1-ShareVL-7B</td><td>42.8</td><td>68.8</td><td>67.2</td><td>69.3</td><td>62.6</td><td>29.0</td><td>66.1</td><td>35.7</td><td>47.0</td><td>54.3</td></tr><tr><td>NoisyRollout-7B</td><td>44.9</td><td>68.7</td><td>67.5</td><td>70.8</td><td>63.9</td><td>29.0</td><td>59.8</td><td>36.4</td><td>48.1</td><td>54.3</td></tr><tr><td>PAPO-7B</td><td>44.5</td><td>67.9</td><td>68.5</td><td>68.2</td><td>64.8</td><td>28.8</td><td>79.6</td><td>36.2</td><td>47.5</td><td>56.2</td></tr><tr><td>BiPS-7B</td><td>41.2</td><td>70.9</td><td>69.0</td><td>70.2</td><td>66.4</td><td>30.7</td><td>76.2</td><td>37.7</td><td>47.3</td><td>56.6</td></tr><tr><td>VPPO-7B</td><td>45.2</td><td>62.9</td><td>70.3</td><td>70.1</td><td>66.3</td><td>31.0</td><td>80.8</td><td>37.6</td><td>48.4</td><td>57.0</td></tr><tr><td>Qwen2.5-VL-7B</td><td>36.1</td><td>63.9</td><td>38.6</td><td>46.6</td><td>56.1</td><td>18.6</td><td>43.4</td><td>25.9</td><td>42.8</td><td>41.3</td></tr><tr><td>+DAPO</td><td>45.5</td><td>66.2</td><td>69.2</td><td>68.4</td><td>65.8</td><td>29.0</td><td>81.9</td><td>37.0</td><td>45.8</td><td>56.5</td></tr><tr><td>+EVAN (ours)</td><td>46.0</td><td>69.0</td><td>69.1</td><td>69.7</td><td>66.5</td><td>31.3</td><td>82.2</td><td>38.8</td><td>48.8</td><td>57.9</td></tr><tr><td colspan="9">Qwen3-VL-8B-Instruct</td><td></td></tr><tr><td>VPPO-8B</td><td>71.7</td><td>70.3</td><td>83.7</td><td>83.6</td><td>78.9</td><td>56.3</td><td>79.1</td><td>52.9</td><td>64.0</td><td>71.2</td></tr><tr><td>Qwen3-VL-8B</td><td>65.2</td><td>68.7</td><td>69.4</td><td>77.2</td><td>73.4</td><td>38.0</td><td>69.1</td><td>40.3</td><td>61.3</td><td>62.5</td></tr><tr><td>+DAPO</td><td>74.6</td><td>71.9</td><td>84.0</td><td>83.9</td><td>76.9</td><td>56.7</td><td>84.1</td><td>52.3</td><td>62.4</td><td>71.9</td></tr><tr><td>+EVAN (ours)</td><td>76.6</td><td>74.9</td><td>85.0</td><td>84.9</td><td>77.6</td><td>57.0</td><td>85.0</td><td>52.0</td><td>63.6</td><td>73.0</td></tr></table>

## 5.3 Ablation Study

We conduct ablation studies on Qwen2.5-VL-3B to isolate the main components of EVAN. All variants use the same training and evaluation configuration.

RankMix Selection. We ablate the selection strategy used by EVAN under a fixed 20% data budget, keeping the evidence-ablated image construction and policy optimization unchanged. The Random variant uniformly samples 20% of the training examples and then applies the same $\rho = 5 0 \%$ scaled-IIG-guided masking as the other variants. As shown in Table 2, selecting examples only by global scaled-IIG underperforms random selection, suggesting that high visual dependency alone can reduce data diversity. Category-wise selection improves the average score by preserving category coverage. RankMix further improves the average score to 51.6 by balancing global visual dependency with category-level coverage. This supports our choice of mixed ranking for selecting examples that are both visually dependent and sufficiently diverse.

Adversarial masking. We ablate the image construction strategy used for evidence-ablated rollouts. As shown in Table 3, substituting the evidence-ablated image with the clean image in the auxiliary branch yields an average score of 50.7. This shows that extra rollouts alone explain only a small fraction of the gain of EVAN. We further evaluate pixel-level perturbations by applying Fast Gradient

Table 2: Ablation study of RankMix Selection with Qwen2.5-VL-3B. All variants use the same 20% data budget and differ only in the selection strategy.
<table><tr><td>Variant</td><td>Geometry3K</td><td>MathVista</td><td>MathVerse</td><td>We-Math</td><td>DynaMath</td><td>MathVision</td><td>MMK12</td><td>MMMU-Pro</td><td>LogicVista</td><td>Avg.</td></tr><tr><td>Random</td><td>38.5</td><td>62.4</td><td>62.5</td><td>65.1</td><td>60.1</td><td>27.1</td><td>72.2</td><td>30.0</td><td>42.0</td><td>51.1</td></tr><tr><td>Global</td><td>39.8</td><td>60.8</td><td>60.9</td><td>61.9</td><td>59.2</td><td>27.5</td><td>72.3</td><td>28.8</td><td>41.7</td><td>50.3</td></tr><tr><td>Category</td><td>38.4</td><td>63.5</td><td>63.5</td><td>64.8</td><td>59.6</td><td>26.8</td><td>72.7</td><td>29.3</td><td>42.9</td><td>51.3</td></tr><tr><td>RankMix (EVAN)</td><td>39.9</td><td>62.7</td><td>62.8</td><td>64.6</td><td>60.6</td><td>27.6</td><td>71.5</td><td>31.0</td><td>43.3</td><td>51.6</td></tr></table>

Table 3: Ablation on evidence-ablated image construction with Qwen2.5-VL-3B. All variants use the same selected examples and differ only in the image used for evidence-ablated rollouts.
<table><tr><td>Variant</td><td>Geometry3K</td><td>MathVista</td><td>MathVerse</td><td>We-Math</td><td>DynaMath</td><td>MathVision</td><td>MMK12</td><td>MMMU-Pro</td><td>LogicVista</td><td>Avg.</td></tr><tr><td>Clean Image</td><td>39.8</td><td>63.5</td><td>60.8</td><td>61.3</td><td>59.8</td><td>26.9</td><td>72.8</td><td>29.2</td><td>42.4</td><td>50.7</td></tr><tr><td>Random Patch</td><td>39.9</td><td>62.0</td><td>62.1</td><td>61.4</td><td>59.6</td><td>27.2</td><td>72.9</td><td>29.9</td><td>42.8</td><td>50.8</td></tr><tr><td>FGSM</td><td>39.6</td><td>61.8</td><td>60.6</td><td>64.1</td><td>59.8</td><td>26.5</td><td>73.2</td><td>29.6</td><td>42.3</td><td>50.8</td></tr><tr><td>Adv. mask (EVAN)</td><td>39.9</td><td>62.7</td><td>62.8</td><td>64.6</td><td>60.6</td><td>27.6</td><td>71.5</td><td>31.0</td><td>43.3</td><td>51.6</td></tr></table>

Table 4: Effect of the masking ratio ρ on Qwen2.5-VL-3B. All variants use the same training recipe and differ only in the fraction of masked patches. $\rho = 1 0 0 \%$ masks the entire image.
<table><tr><td>Variant</td><td>Geometry3K</td><td>MathVista</td><td>MathVerse</td><td>We-Math</td><td>DynaMath</td><td>MathVision</td><td>MMK12</td><td>MMMU-Pro</td><td>LogicVista</td><td>Avg.</td></tr><tr><td>EVAN (ρ = 25%)</td><td>39.6</td><td>59.9</td><td>61.5</td><td>64.9</td><td>61.0</td><td>26.8</td><td>72.7</td><td>31.1</td><td>42.8</td><td>51.1</td></tr><tr><td>EVAN (ρ = 50%)</td><td>39.9</td><td>62.7</td><td>62.8</td><td>64.6</td><td>60.6</td><td>27.6</td><td>71.5</td><td>31.0</td><td>43.3</td><td>51.6</td></tr><tr><td>EVAN (ρ = 75%)</td><td>39.4</td><td>61.6</td><td>62.2</td><td>63.7</td><td>60.2</td><td>27.9</td><td>73.0</td><td>30.7</td><td>43.5</td><td>51.4</td></tr><tr><td>EVAN (ρ = 100%)</td><td>39.5</td><td>63.2</td><td>63.1</td><td>65.0</td><td>59.1</td><td>26.5</td><td>71.3</td><td>29.4</td><td>41.0</td><td>50.9</td></tr></table>

Sign Method (FGSM) [Goodfellow et al., 2014, Liu et al., 2025b], which only marginally improves the average score to 50.8. This suggests that pixel-level adversarial perturbations are insufficiently stable in the online RL setting, where the continuously evolving policy renders locally optimized perturbations rapidly uninformative. In contrast, the scaled-IIG-guided masking achieves an average score of 51.6, validating our design choice of employing patch-level evidence ablation to construct evidence-ablated images for hard negative generation. Replacing scaled-IIG-guided masking with random patch masking at the same 50% ratio yields an average score of only 50.8, suggesting that the gain comes from localizing query-critical evidence rather than from the masked area alone.

Masking ratio. We further vary the masking ratio ρ of scaled-IIG-guided masking while keeping the training and evaluation settings unchanged. As shown in Table 4, masking 50% of the patches achieves the best average score of 51.6, outperforming the 25%, 75%, and 100% settings by 0.5, 0.2, and 0.7 points, respectively. A lower ratio may leave part of the query-critical evidence visible, whereas $\rho = 1 0 0 \%$ blacks out the entire image while retaining the image tokens, so the evidenceablated rollouts are generated without any visual evidence. This fully black setting yields the lowest average score, consistent with our motivation in Section 4.1 that overly destructive interventions may make the image uninformative and produce off-distribution failures. These results support our choice of $\rho = 5 0 \%$ , which removes query-critical evidence while preserving the remaining visual context.

Stability controls. We ablate the two stability controls used for evidence-ablated negatives. As shown in Figure 1, removing Bounded negative advantage (BNA) causes late-stage entropy collapse and a sharp increase in gradient norm, indicating that the negative signal can dominate the clean RL update. Removing rollout filtering also destabilizes training, with PPO KL and response length growing rapidly. In contrast, EVAN keeps these dynamics within stable ranges. These results show that evidence-ablated negatives require both bounded advantages and clean-image plausibility filtering to provide useful training signal without destabilizing online RL.

## 5.4 Further Analysis

We provide further analysis to better understand the behavior of EVAN during training. Additional results on filtering behavior, selection ratio, and selected-data distribution are provided in the appendix.

![](images/1a40244f4fc5c6bcc57b30ba2ed02ade9e4b5edc83344d4df8039d923d05f0a1.jpg)

![](images/3a4e292c8acf7bf4e6c0232e78a1643ae335abcb80f396c0bb80632424ebd76e.jpg)

Figure 1: (Left) Bounded negative advantage (BNA) ablation on Qwen2.5-VL-3B, with entropy on the left axis and gradient norm on the right axis. (Right) rollout filtering ablation on Qwen2.5-VL-7B, with PPO KL Divergence on the left axis and response length on the right axis.  
![](images/3baf400cad77b29232a3eaf75e5be03409d0c01175eed18e542ae3bebb87cac3.jpg)

![](images/964d78bb805e3e7419955fb3dfc4342951a957faab09e617f8e9a422965db70a.jpg)  
Figure 2: Training dynamics of the EVidence-Ablated Negatives branch. (Left) Number of prompts with prepared evidence-ablated images in each training step. (Right) Number of retained evidenceablated negative responses appended to the clean-image update after filtering.

Evidence-ablated negatives. Figure 2 shows the training dynamics of the evidence-ablated negatives. Although evidence-ablated images are prepared for the selected 20% of training examples, only a small number of evidence-ablated responses are retained after filtering and appended to the clean-image update. This shows that EVAN uses a sparse auxiliary signal rather than adding many extra training responses. Together with the improved benchmark performance, this suggests that the gain does not come from substantially increasing the amount of training data, but from using a small set of high-quality evidence-ablated negatives as additional supervision.

## 6 Conclusion

We presented EVAN, a multimodal RL framework that improves visual reasoning by learning from evidence-ablated negatives. EVAN selects visually dependent examples with scaled-IIG, constructs evidence-ablated images by removing query-critical visual evidence, and incorporates the resulting negatives into clean-image RLVR with stability controls. Experiments on challenging multimodal reasoning benchmarks show that EVAN consistently improves over strong RLVR baselines across model scales. Ablations show that these gains come from localizing query-critical evidence rather than from extra rollouts or the masked area alone, and that bounded negative advantages and rollout filtering are both needed for stable training. The gains also extend to Qwen3-VL-8B and transfer beyond structured reasoning to chart understanding, general visual question answering, and counterfactual visual grounding, with no added parameters or computation at inference time. These results suggest that targeted evidence ablation provides useful auxiliary supervision for reducing language-prior shortcuts in LMMs.

## References

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025a.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025b.

Liang Chen, Lei Li, Haozhe Zhao, Yifan Song, and Vinci. R1-v: Reinforcing super generalization ability in vision-language models with less than \$3. https://github.com/Deep-Agent/R1-V, 2025. Accessed: 2025-02-02.

Ruibo Chen, Tianyi Xiong, Yihan Wu, Guodong Liu, Zhengmian Hu, Lichang Chen, Yanshuo Chen, Chenxi Liu, and Heng Huang. Gpt-4 vision on medical image classification–a case study on covid-19 dataset. arXiv preprint arXiv:2310.18498, 2023.

Ruibo Chen, Yihan Wu, Lichang Chen, Guodong Liu, Qi He, Tianyi Xiong, Chenxi Liu, Junfeng Guo, and Heng Huang. Your vision-language model itself is a strong filter: Towards high-quality instruction tuning with data selection. arXiv preprint arXiv:2402.12501, 2024a.

Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 24185–24198, 2024b.

Timothée Darcet, Maxime Oquab, Julien Mairal, and Piotr Bojanowski. Vision transformers need registers. In International conference on learning representations, volume 2024, pages 2632–2652, 2024.

Zixiang Di, Jinyi Han, Shuo Zhang, Ying Liao, Zhi Li, Xiaofeng Ji, Yongqi Wang, Zheming Yang, Ming Gao, Bingdong Li, et al. Not all negative samples are equal: Llms learn better from plausible reasoning. arXiv preprint arXiv:2602.03516, 2026.

Michal Golovanevsky, William Rudman, Michael A Lepori, Amir Bar, Ritambhara Singh, and Carsten Eickhoff. Pixels versus priors: Controlling knowledge priors in vision-language models through visual counterfacts. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 24848–24863, 2025.

Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572, 2014.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Siyuan Huang, Xiaoye Qu, Yafu Li, Yun Luo, Zefeng He, Daizong Liu, and Yu Cheng. Spotlight on token perception for multimodal reinforcement learning. arXiv preprint arXiv:2510.09285, 2025a.

Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Zhe Xu, Yao Hu, and Shaohui Lin. Vision-r1: Incentivizing reasoning capability in multimodal large language models. arXiv preprint arXiv:2503.06749, 2025b.

Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6693–6702. IEEE, 2019.

Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.

Yuqi Jia, Hangsheng Zhang, Chenxi Liu, Junjie Liang, and Xun Chen. Black-box red-teaming of multi-agent systems via reinforcement learning, 2026. URL https://openreview.net/forum? id=jd6sUOwz3H.

Songtao Jiang, Yan Zhang, Ruizhe Chen, Yeying Jin, and Zuozhu Liu. Modality-fair preference optimization for trustworthy mllm alignment. arXiv preprint arXiv:2410.15334, 2024.

Seil Kang, Jinyeong Kim, Junhyeok Kim, and Seong Jae Hwang. See what you are told: Visual attention sink in large multimodal models. In International Conference on Learning Representations, volume 2025, pages 87676–87703, 2025.

Zujie Liang, Weitao Jiang, Haifeng Hu, and Jiaying Zhu. Learning to contrast the counterfactual samples for robust visual question answering. In Proceedings of the 2020 conference on empirical methods in natural language processing (EMNLP), pages 3285–3292, 2020.

Chenxi Liu, Junjie Liang, Yuqi Jia, Bochuan Cao, Yang Bai, Heng Huang, and Xun Chen. Explore data left behind in reinforcement learning for reasoning language models. arXiv preprint arXiv:2511.04800, 2025a.

Chenxi Liu, Tianyi Xiong, Yanshuo Chen, Ruibo Chen, Yihan Wu, Junfeng Guo, Tianyi Zhou, and Heng Huang. Modality-balancing preference optimization of large multimodal models by adversarial negative mining. arXiv preprint arXiv:2506.08022, 2025b.

Chenxi Liu, Yanshuo Chen, Ruibo Chen, Tianyi Xiong, Tong Zheng, and Heng Huang. Learning from self-debate: Preparing reasoning models for multi-agent debate. arXiv preprint arXiv:2601.22297, 2026.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.

Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, January 2024a. URL https:// llava-vl.github.io/blog/2024-01-30-llava-next/.

Shi Liu, Kecheng Zheng, and Wei Chen. Paying more attention to image: A training-free method for alleviating hallucination in lvlms. In European Conference on Computer Vision, pages 125–140. Springer, 2024b.

Xiangyan Liu, Jinjie Ni, Zijian Wu, Chao Du, Longxu Dou, Haonan Wang, Tianyu Pang, and Michael Qizhe Shieh. Noisyrollout: Reinforcing visual reasoning with data augmentation. arXiv preprint arXiv:2504.13055, 2025c.

Yuqi Liu, Bohao Peng, Zhisheng Zhong, Zihao Yue, Fanbin Lu, Bei Yu, and Jiaya Jia. Segzero: Reasoning-chain guided segmentation via cognitive reinforcement. arXiv preprint arXiv:2503.06520, 2025d.

Pan Lu, Ran Gong, Shibiao Jiang, Liang Qiu, Siyuan Huang, Xiaodan Liang, and Song-Chun Zhu. Inter-gps: Interpretable geometry problem solving with formal language and symbolic reasoning. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 6774–6786, 2021.

Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating math reasoning in visual contexts with gpt-4v, bard, and other large multimodal models. CoRR, abs/2310.02255, 5:15, 2023.

Ahmed Masry, Jia Qing Tan, Shafiq Joty, Enamul Hoque, et al. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. In Findings of the association for computational linguistics: ACL 2022, pages 2263–2279, 2022.

Fanqing Meng, Lingxiao Du, Zongkai Liu, Zhixiang Zhou, Quanfeng Lu, Daocheng Fu, Tiancheng Han, Botian Shi, Wenhai Wang, Junjun He, et al. Mm-eureka: Exploring the frontiers of multimodal reasoning with rule-based reinforcement learning. arXiv preprint arXiv:2503.07365, 2025.

Yulei Niu, Kaihua Tang, Hanwang Zhang, Zhiwu Lu, Xian-Sheng Hua, and Ji-Rong Wen. Counterfactual vqa: A cause-effect look at language bias. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12700–12710, 2021.

OpenAI. Gpt-4v(ision) system card, 2023. URL https://api.semanticscholar.org/ CorpusID:263218031.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730– 27744, 2022.

Yingzhe Peng, Gongrui Zhang, Miaosen Zhang, Zhiyuan You, Jie Liu, Qipeng Zhu, Kai Yang, Xingzhong Xu, Xin Geng, and Xu Yang. Lmm-r1: Empowering 3b lmms with strong reasoning abilities through two-stage rule-based rl. arXiv preprint arXiv:2503.07536, 2025.

Renjie Pi, Tianyang Han, Wei Xiong, Jipeng Zhang, Runtao Liu, Rui Pan, and Tong Zhang. Strengthening multimodal large language model with bootstrapped preference optimization. In European Conference on Computer Vision, pages 382–398. Springer, 2024.

Runqi Qiao, Qiuna Tan, Guanting Dong, MinhuiWu MinhuiWu, Chong Sun, Xiaoshuai Song, Jiapeng Wang, Zhuoma Gongque, Shanglin Lei, Yifan Zhang, et al. We-math: Does your large multimodal model achieve human-like mathematical reasoning? In Proceedings of the 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 20023–20070, 2025.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Haozhan Shen, Peng Liu, Jingcheng Li, Chunxin Fang, Yibo Ma, Jiajia Liao, Qiaoli Shen, Zilun Zhang, Kangjia Zhao, Qianqian Zhang, Ruochen Xu, and Tiancheng Zhao. Vlm-r1: A stable and generalizable r1-style large vision-language model. arXiv preprint arXiv:2504.07615, 2025.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024.

Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, et al. Kimi k1. 5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599, 2025.

Peter Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Adithya Jairam Vedagiri IYER, Sai Charitha Akula, Shusheng Yang, Jihan Yang, Manoj Middepogu, Ziteng Wang, et al. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. Advances in Neural Information Processing Systems, 37:87310–87356, 2024.

Fei Wang, Wenxuan Zhou, James Y Huang, Nan Xu, Sheng Zhang, Hoifung Poon, and Muhao Chen. mdpo: Conditional preference optimization for multimodal large language models. arXiv preprint arXiv:2406.11839, 2024a.

Haozhe Wang, Chao Qu, Zuming Huang, Wei Chu, Fangzhen Lin, and Wenhu Chen. Vl-rethinker: Incentivizing self-reflection of vision-language models with reinforcement learning. arXiv preprint arXiv:2504.08837, 2025a.

Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Houxing Ren, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset. Advances in Neural Information Processing Systems, 37:95095–95169, 2024b.

Zhenhailong Wang, Xuehang Guo, Sofia Stoica, Haiyang Xu, Hongru Wang, Hyeonjeong Ha, Xiusi Chen, Yangyi Chen, Ming Yan, Fei Huang, et al. Perception-aware policy optimization for multimodal reasoning. arXiv preprint arXiv:2507.06448, 2025b.

Yijia Xiao, Edward Sun, Tianyu Liu, and Wei Wang. Logicvista: Multimodal llm logical reasoning benchmark in visual contexts. arXiv preprint arXiv:2407.04973, 2024.

Tianyi Xiong, Xiyao Wang, Dong Guo, Qinghao Ye, Haoqi Fan, Quanquan Gu, Heng Huang, and Chunyuan Li. Llava-critic: Learning to evaluate multimodal models. 2024. URL https: //arxiv.org/abs/2410.02712.

Tianyi Xiong, Yi Ge, Ming Li, Zuolong Zhang, Pranav Kulkarni, Kaishen Wang, Qi He, Zeying Zhu, Chenxi Liu, Ruibo Chen, et al. Multi-crit: Benchmarking multimodal judges on pluralistic criteria-following. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8641–8652, 2026a.

Tianyi Xiong, Shihao Wang, Guilin Liu, Yi Dong, Ming Li, Heng Huang, Jan Kautz, and Zhiding Yu. Phycritic: Multimodal critic models for physical ai. arXiv preprint arXiv:2602.11124, 2026b.

Tianyi Xiong, Zhengyuan Yang, Xiaofei Wang, Chung-Ching Lin, Ruichun Ma, Kevin Lin, Zhendong Wang, Linjie Li, Chenxi Liu, Ruibo Chen, et al. Rubrics as visual-repair context for self-evolving ui-to-code generation. arXiv preprint arXiv:2608.24138, 2026c.

Yi Yang, Xiaoxuan He, Hongkun Pan, Xiyan Jiang, Yan Deng, Xingtao Yang, Haoyu Lu, Dacheng Yin, Fengyun Rao, Minfeng Zhu, et al. R1-onevision: Advancing generalized multimodal reasoning through cross-modal formalization. arXiv preprint arXiv:2503.10615, 2025.

Huanjin Yao, Qixiang Yin, Jingyi Zhang, Min Yang, Yibo Wang, Wenhao Wu, Fei Su, Li Shen, Minghui Qiu, Dacheng Tao, et al. R1-sharevl: Incentivizing reasoning capability of multimodal large language models via share-grpo. arXiv preprint arXiv:2505.16673, 2025.

En Yu, Kangheng Lin, Liang Zhao, Jisheng Yin, Yuang Peng, Haoran Wei, Jianjian Sun, Chunrui Han, Zheng Ge, Xiangyu Zhang, Daxin Jiang, Jingyu Wang, and Wenbing Tao. Perception r1: Pioneering perception policy with reinforcement learning. arXiv preprint arXiv:2504.07954, 2025a.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025b.

Tianyu Yu, Yuan Yao, Haoye Zhang, Taiwen He, Yifeng Han, Ganqu Cui, Jinyi Hu, Zhiyuan Liu, Hai-Tao Zheng, Maosong Sun, et al. Rlhf-v: Towards trustworthy mllms via behavior alignment from fine-grained correctional human feedback. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13807–13816, 2024.

Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Botao Yu, Ge Zhang, Huan Sun, et al. Mmmu-pro: A more robust multi-discipline multimodal understanding benchmark. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15134–15186, 2025.

Yufei Zhan, Yousong Zhu, Shurong Zheng, Hongyin Zhao, Fan Yang, Ming Tang, and Jinqiao Wang. Vision-r1: Evolving human-free alignment in large vision-language models via vision-guided reinforcement learning. arXiv preprint arXiv:2503.18013, 2025.

Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Yu Qiao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? In European Conference on Computer Vision, pages 169–186. Springer, 2024.

Shuoshuo Zhang, Yizhen Zhang, Jingjing Fu, Lei Song, Jiang Bian, Yujiu Yang, and Rui Wang. See less, see right: Bi-directional perceptual shaping for multimodal reasoning. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 33499–33509, 2026.

Zhiyuan Zhao, Bin Wang, Linke Ouyang, Xiaoyi Dong, Jiaqi Wang, and Conghui He. Beyond hallucinations: Enhancing lvlms through hallucination-aware direct preference optimization. arXiv preprint arXiv:2311.16839, 2023.

Yaowei Zheng, Junting Lu, Shenzhi Wang, Zhangchi Feng, Dongdong Kuang, and Yuwen Xiong. Easyr1: An efficient, scalable, multi-modality rl training framework. https://github.com/ hiyouga/EasyR1, 2025.

Yiyang Zhou, Chenhang Cui, Rafael Rafailov, Chelsea Finn, and Huaxiu Yao. Aligning modalities in vision large language models via preference fine-tuning. arXiv preprint arXiv:2402.11411, 2024.

Xinyu Zhu, Mengzhou Xia, Zhepei Wei, Wei-Lin Chen, Danqi Chen, and Yu Meng. The surprising effectiveness of negative reinforcement in llm reasoning. arXiv preprint arXiv:2506.01347, 2025.

Chengke Zou, Xingang Guo, Rui Yang, Junyu Zhang, Bin Hu, and Huan Zhang. Dynamath: A dynamic visual benchmark for evaluating mathematical reasoning robustness of vision language models. arXiv preprint arXiv:2411.00836, 2024.

## Appendix

## A Experimental Details

## A.1 Training Details

We use DAPO [Yu et al., 2025b] as both the baseline method and the underlying optimizer for EVAN. For all experiments, we disable the KL penalty by setting the KL coefficient to 0, and use asymmetric clipping with $\epsilon _ { \mathrm { l o w } } = 0 . 2$ and $\epsilon _ { \mathrm { h i g h } } = 0 . 2 8$ . The maximum response length is set to 4,096 tokens for the 3B and 7B models and 8,192 tokens for the 8B model. At each rollout step, we generate responses for 256 prompts and optimize the policy with a mini-batch size of 32, which gives 8 gradient updates per rollout step. Training runs for 300 prompt-generation steps, corresponding to 2 epochs over the training dataset. For EVAN, we generate 2 responses for each evidence-ablated image with the 3B model and 4 responses with the 7B and 8B models to obtain auxiliary candidates. Experiments are conducted on 4 × NVIDIA H200 GPUs, and the training run of Qwen2.5-VL-7B takes 36.05 hours with DAPO and 36.74 hours with EVAN.

## A.2 Training Prompt and Reward

For all training rollouts, we append a fixed instruction to each question to standardize the response format:

You first think through the reasoning process as an internal monologue, enclosed within <think> </think> tags. Then, provide your final answer enclosed within \boxed{}.

This instruction encourages the model to separate its reasoning process from the final answer, and also makes answer extraction more reliable.

The reward contains an accuracy term and a format term. The accuracy term checks whether the final boxed answer matches the ground-truth answer, while the format term checks whether the response follows the required structure with <think> and </think> tags and a final \boxed{} answer. The total reward is computed as

$$
R ( o ) = 0 . 9 \cdot R _ { \mathrm { a c c } } ( o ) + 0 . 1 \cdot R _ { \mathrm { f m t } } ( o ) ,\tag{16}
$$

where $R _ { \mathrm { a c c } } ( o )$ denotes the accuracy score and $R _ { \mathrm { f m t } } ( o )$ denotes the format score. This reward design prioritizes answer correctness while keeping a small incentive for producing parseable responses.

## A.3 Bounded Negative Advantage under Binary Rewards

Consider binary rewards and let p be the fraction of correct responses in the clean rollout group, where dynamic sampling keeps $0 < p < 1$ . The group mean is p and the group standard deviation is $\sqrt { p ( 1 - p ) }$ , so Eq. (1) assigns $\sqrt { ( 1 - p ) / p }$ to each correct response and $- { \sqrt { p / ( 1 - p ) } }$ to each incorrect response. Eq. (14) therefore gives

$$
\hat { A } _ { j } ^ { a } = - \operatorname * { m i n } \left\{ \sqrt { \frac { 1 - p } { p } } , \sqrt { \frac { p } { 1 - p } } \right\} .\tag{17}
$$

When most clean rollouts are incorrect $( p < 0 . 5 )$ , the evidence-ablated negative receives the same advantage as a clean incorrect response. When most clean rollouts are correct $( p > 0 . 5 )$ , its magnitude equals the advantage of a clean correct response and is smaller than the penalty on a clean incorrect response. The magnitude reaches its maximum of 1 at $p = 0 . 5$ , where the clean outcomes are most uncertain, and approaches 0 as p approaches 0 or 1, where the behavior of the model on the example is already stable. The same values hold for the reward in Appendix A.2 whenever all responses in the group receive the same format score, because the rewards then take only two values and the advantage in Eq. (1) does not change when a constant is added to all rewards or when all rewards are multiplied by a positive constant. Table 5 lists these values for the group size $G = 8$ used in our experiments.

Table 5: Advantages under binary rewards for a clean rollout group of $G = 8$ responses. Each column corresponds to a number of correct responses in the clean rollout group, and each row gives the advantage assigned to one type of response.
<table><tr><td>Correct clean responses</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td></tr><tr><td>Clean correct response</td><td>2.65</td><td>1.73</td><td>1.29</td><td>1.00</td><td>0.77</td><td>0.58</td><td>0.38</td></tr><tr><td>Clean incorrect response</td><td>-0.38</td><td>-0.58</td><td>-0.77</td><td>-1.00</td><td>-1.29</td><td>-1.73</td><td>-2.65</td></tr><tr><td>Evidence-ablated negative</td><td>-0.38</td><td>-0.58</td><td>-0.77</td><td>-1.00</td><td>-0.77</td><td>-0.58</td><td>-0.38</td></tr></table>

## B Additional Experiments

## B.1 Confidence intervals of main results.

To assess the stability of the main results in Table 1, Table 6 reports 95% confidence intervals for DAPO and EVAN on every benchmark at both the 3B and 7B scales. Each interval is a Student’s t interval with seven degrees of freedom, computed from the eight stochastic pass@1 evaluations underlying accuracy@8 for the same fixed checkpoint. The interval of the average score is computed in the same way from the eight per-evaluation averages over all nine benchmarks. EVAN outperforms DAPO in average accuracy at both scales, and its 95% confidence interval lies entirely above that of DAPO, with $5 1 . { \overset { - } { 6 } } \pm 0 . 1 8$ versus $5 0 . 3 \pm 0 . 2 5$ for 3B and $5 7 . 9 \pm 0 . 2 3$ versus $5 6 . 5 \pm 0 . { \dot { 1 } } 6$ for 7B. This shows that the overall gains are stable under stochastic decoding.

Table 6: 95% confidence intervals of DAPO and EVAN on Qwen2.5-VL-3B and Qwen2.5-VL-7B. Each interval is computed from the eight stochastic pass@1 evaluations of the same checkpoint.
<table><tr><td>Model Geometry3K MathVista MathVerse We-Math DynaMath MathVision MMK12 MMMU-Pro LogicVista</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>Avg.</td></tr><tr><td colspan="8">Qwen2.5-VL-3B-Instruct</td></tr><tr><td></td><td></td><td></td><td></td><td>DAPO 38.8±0.80 62.4±0.52 60.4±0.2860.5±0.47 59.3±0.29 26.5±0.26 72.2±0.24</td><td></td><td></td><td> $2 9 . 9 { \scriptstyle \pm 0 . 4 6 }$ </td><td>43.0±0.93</td><td> $5 0 . 3 { \scriptstyle \pm 0 . 2 5 }$ </td></tr><tr><td>EVAN</td><td></td><td></td><td></td><td>39.9±0.66 62.7±0.45 62.8±0.50 64.6±0.42 60.6±0.17 27.6±0.58 71.5±0.40 31.0±0.61 43.3±1.14 51.6±0.18</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="8"></td></tr><tr><td></td><td></td><td></td><td></td><td>Qwen2.5-VL-7B-Instruct</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DAPO</td><td> $4 5 . 5 { \pm } 0 . 9 9$ </td><td></td><td></td><td>66.2±0.5869.2±0.4668.4±0.3265.8±0.12 29.0±0.2681.9±0.24</td><td></td><td></td><td> $3 7 . 0 { \scriptstyle \pm 0 . 4 4 }$ </td><td>45.8±1.19</td><td> $5 6 . 5 { \scriptstyle \pm 0 . 1 6 }$ </td></tr><tr><td>EVAN</td><td></td><td></td><td></td><td> $4 6 . 0 \pm 1 . 0 8 6 9 . 0 \pm 0 . 4 6 6 9 . 1 \pm 0 . 5 8 6 9 . 7 \pm 0 . 3 3 6 6 . 5 \pm 0 . 4 5 3 3 1 . 3 \pm 0 . 3 5 8 2 . 2 \pm 0 . 2 4 3 8 . 8 \pm 0 . 5 0 4 8 . 8 \pm 1 . 0 7$ </td><td></td><td></td><td></td><td></td><td> $5 7 . 9 { \pm } 0 . 2 3 $ </td></tr></table>

## B.2 Transfer beyond structured reasoning.

To evaluate whether the gains of EVAN extend beyond structured reasoning, Table 7 reports the results of the Qwen2.5-VL-3B DAPO and EVAN checkpoints on the full ChartQA [Masry et al., 2022], GQA [Hudson and Manning, 2019], and Visual CounterFact [Golovanevsky et al., 2025] evaluation sets. Each score is the mean sampled pass@1 over eight evaluations, using relaxed accuracy for ChartQA and exact-match accuracy for GQA. For Visual CounterFact, we use the direct counterfactual-grounding condition (CF+this), in which a counterfactual image is paired with an instance-specific question and the correct answer follows the modified visual attribute rather than the corresponding world-knowledge prior. EVAN outperforms DAPO on all three benchmarks and improves the pooled Visual CounterFact accuracy from 81.61 to 85.01. These results show that EVAN transfers beyond structured reasoning to chart understanding, general visual question answering, and counterfactual visual grounding.

Table 7: Transfer of the Qwen2.5-VL-3B DAPO and EVAN checkpoints beyond structured reasoning.
<table><tr><td>Model</td><td>ChartQA</td><td>GQA</td><td colspan="3">Visual CounterFact (CF+this)</td></tr><tr><td></td><td></td><td></td><td>Color</td><td>Size</td><td>Pooled</td></tr><tr><td>DAPO</td><td>81.74</td><td>54.46</td><td>84.61</td><td>79.57</td><td>81.61</td></tr><tr><td>EVAN</td><td>82.55</td><td>55.83</td><td>85.04</td><td>84.99</td><td>85.01</td></tr></table>

## B.3 Computational cost.

Table 8 compares the wall-clock cost of the matched 300-step Qwen2.5-VL-7B runs of DAPO and EVAN on the same 4 × NVIDIA H200 GPUs. The rollout time of EVAN includes both the standard clean-image generation and the auxiliary evidence-ablated generation. Filtering the evidence-ablated candidates is a lightweight host-side selection that takes negligible time relative to generation and is included in the total training time. EVAN increases the total training time by 1.9%, and its cumulative rollout time is lower than that of DAPO because the sampled response lengths and the number of dynamic-sampling batches differ between the two runs. In addition, EVAN requires a one-time offline preprocessing step, which takes about 2.89 hours on ${ 8 \times \mathrm { N V I D I A } }$ H200 GPUs for scaled-IIG scoring and 1.98 hours on $4 \times \mathrm { N V I D I A }$ H200 GPUs for scaled-IIG-guided mask construction. The auxiliary branch is removed after training, so EVAN adds no parameters or computation at inference time.

Table 8: Training cost of the matched 300-step Qwen2.5-VL-7B runs on 4 × NVIDIA H200 GPUs, excluding the one-time preprocessing of EVAN.
<table><tr><td>Method</td><td>Total training time (h)</td><td>Cumulative rollout time (h)</td></tr><tr><td>DAPO</td><td>36.05</td><td>≈22.0</td></tr><tr><td>EVAN</td><td>36.74</td><td>≈21.2</td></tr></table>

## B.4 Rollout filtering ratio during training.

As illustrated in Figure 3, both the length filter and the log-probability (logp) filter play a crucial role in curating high-quality training signals. The length filter (left) consistently ensures that evidenceablated negatives remain comparable in scale to clean rollouts, while the logp filter (right) effectively rejects evidence-ablated responses that are out-of-distribution or lack plausibility under the clean visual context. The stable filtering ratios observed across both 3B and 7B models demonstrate that our approach effectively suppresses low-quality or unstable gradients by filtering out uninformative failure cases.

![](images/f1537af5746674a9c1c87983da3e81cc23ea6ea9c8bd33fff547bf0014a2f3fa.jpg)

![](images/c78a154c3a5f42b26ac081cb74dbcb4ce27b8d9016d9a38322f128919cb84b5d.jpg)  
Figure 3: Rollout filtering ratios during training. The fraction of candidates that pass the length filter (left) and the clean-image likelihood filter (right) for EVAN-3B and EVAN-7B.

## B.5 Quality of retained negatives.

Masking query-critical patches intentionally makes the original answer unverifiable, which is the condition under which a language-biased policy still answers from priors. To verify that the retained negatives remain task-directed rather than describing the mask, we audit the evidence-ablated responses of the EVAN run on Qwen2.5-VL-3B, where 3,906 of the 22,864 evidence-ablated candidates, or 17.1%, pass all filters and are used for training. We use GPT-5.5 to inspect 250 retained responses and 100 responses removed by the clean-image likelihood filter, and we also manually review 50 retained responses. As shown in Table 9, all retained responses remain semantically relevant and task-directed in both audits, and none describes the mask or refuses to answer because the visual evidence is obscured. In contrast, 40 of the 100 responses removed by the likelihood filter are pathological, with 30 containing degenerate text and 10 explicitly describing the mask. These results show that the likelihood filter removes mask-induced pathological responses and keeps plausible task-directed failures as negatives.

Table 9: Semantic audit of evidence-ablated responses from the EVAN run on Qwen2.5-VL-3B. Retained responses pass all filters and are used for training. Pathological responses contain degenerate text or explicitly describe the mask.
<table><tr><td>Responses</td><td>Reviewer</td><td>Audited</td><td>Pathological</td><td>Degenerate text</td><td>Describes mask</td></tr><tr><td>Retained</td><td>GPT-5.5</td><td>250</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Retained</td><td>Human</td><td>50</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Removed by likelihood filter</td><td>GPT-5.5</td><td>100</td><td>40</td><td>30</td><td>10</td></tr></table>

## B.6 Effect of selection ratio.

We investigate the impact of the RankMix selection ratio τ on model performance. As shown in Table 10, EVAN achieves its best average performance with a sparse selection ratio of τ = 20%. Notably, increasing the ratio to 30% leads to a performance degradation, with the average score dropping to 50.8. This trend suggests that applying evidence-ablated negatives to an excessively large portion of the dataset may be counterproductive. It indicates that the strategy of compressing language priors should be selectively applied only to examples with high visual dependency. These results further validate our scaled-IIG-guided mining approach, demonstrating that focusing on a small but high-quality subset of visually dependent data is more effective than uniform intervention across the entire dataset.

Table 10: Effect of the RankMix selection ratio τ on Qwen2.5-VL-3B. All variants use the same training recipe and differ only in the fraction of selected visually dependent examples.
<table><tr><td>Variant</td><td>Geometry3K MathVista</td><td></td><td>MathVerse</td><td>We-Math</td><td>DynaMath</td><td>MathVision</td><td>MMK12</td><td>MMMU-Pro</td><td>LogicVista</td><td>Avg.</td></tr><tr><td>EVAN (τ = 10%)</td><td>39.5</td><td>63.2</td><td>63.2</td><td>63.9</td><td>59.3</td><td>27.4</td><td>72.4</td><td>30.1</td><td>44.6</td><td>51.5</td></tr><tr><td>EVAN (τ = 20%)</td><td>39.9</td><td>62.7</td><td>62.8</td><td>64.6</td><td>60.6</td><td>27.6</td><td>71.5</td><td>31.0</td><td>43.3</td><td>51.6</td></tr><tr><td>EVAN (τ = 30%)</td><td>38.8</td><td>63.7</td><td>60.3</td><td>60.3</td><td>61.1</td><td>27.5</td><td>72.1</td><td>30.3</td><td>43.0</td><td>50.8</td></tr></table>

## B.7 Scaled-IIG Change after Image Perturbation

Table 11 evaluates how image perturbation changes the visual-dependency signal on the RankMixselected data. Adversarial masking reduces both scaled-IIG and model accuracy after intervention, indicating that it weakens the visual evidence used by the model while retaining a non-trivial level of task solvability. In contrast, FGSM causes a much sharper drop in both metrics, suggesting that pixel-level perturbation is overly destructive for online negative generation. These results support the use of adversarial masking as a controlled intervention for constructing evidence-ablated images.

Table 11: Scaled-IIG and model accuracy before and after image perturbation on the RankMix 20% selected data.
<table><tr><td>Variant</td><td>Scaled-IIG Before</td><td>Scaled-IIG After</td><td>Acc. Before (%)</td><td>Acc. After (%)</td></tr><tr><td>Adv. mask</td><td>0.595</td><td>0.455</td><td>88.3</td><td>66.3</td></tr><tr><td>FGSM</td><td>0.595</td><td>0.093</td><td>88.3</td><td>35.3</td></tr></table>

## B.8 Scaled-IIG and response correctness.

We further examine whether scaled-IIG reflects response correctness on 5,000 ViRL39K questions, sampling five responses per question from each of Qwen2.5-VL-3B-Instruct, Qwen2.5-VL-7B-Instruct, Qwen3-VL-8B-Instruct, and Qwen3-VL-8B-Thinking. As shown in Table 12, using the scaled-IIG of each response to predict its correctness yields an AUC above 0.5 for every model, with a mean of 0.585. Responses in the highest scaled-IIG quintile are also more accurate than those in the lowest quintile for every model, with a mean accuracy of 69.9% versus 52.9%, a 17.0 point gap. These results show that scaled-IIG is positively associated with response correctness and support its use as a visual-dependency signal.

Table 12: Scaled-IIG and response correctness on 5,000 ViRL39K questions with five sampled responses per question. AUC measures how well the scaled-IIG of each response predicts its correctness. Top 20% and Bottom 20% report the accuracy (%) of the responses with the highest and lowest scaled-IIG, and Gap is their difference.
<table><tr><td>Model</td><td>AUC</td><td>Top 20%</td><td>Bottom 20%</td><td>Gap</td></tr><tr><td>Qwen2.5-VL-3B-Instruct</td><td>0.566</td><td>43.6</td><td>28.5</td><td>15.1</td></tr><tr><td>Qwen2.5-VL-7B-Instruct</td><td>0.583</td><td>63.5</td><td>41.0</td><td>22.6</td></tr><tr><td>Qwen3-VL-8B-Instruct</td><td>0.592</td><td>85.8</td><td>71.7</td><td>14.1</td></tr><tr><td>Qwen3-VL-8B-Thinking</td><td>0.600</td><td>86.6</td><td>70.5</td><td>16.2</td></tr><tr><td>Mean</td><td>0.585</td><td>69.9</td><td>52.9</td><td>17.0</td></tr></table>

<table><tr><td rowspan=1 colspan=5>Subway train schedule</td></tr><tr><td rowspan=1 colspan=1>River Avenue</td><td rowspan=1 colspan=1>9:15 A.M.</td><td rowspan=1 colspan=1>10:00 A.M.</td><td rowspan=1 colspan=1>10:15A.M.</td><td rowspan=1 colspan=1>11:00 A.M.</td></tr><tr><td rowspan=1 colspan=1>Jackson Street</td><td rowspan=1 colspan=1>10:00 A.M.</td><td rowspan=1 colspan=1>10:45 A.M.</td><td rowspan=1 colspan=1>11:00 A.M.</td><td rowspan=1 colspan=1>11:45 A.M.</td></tr><tr><td rowspan=1 colspan=1>Ash Street</td><td rowspan=1 colspan=1>10:30 A.M.</td><td rowspan=1 colspan=1>11:15 A.M.</td><td rowspan=1 colspan=1>11:30 A.M.</td><td rowspan=1 colspan=1>12:15 P.M.</td></tr><tr><td rowspan=1 colspan=1>Central Boulevard</td><td rowspan=1 colspan=1>10:45 A.M.</td><td rowspan=1 colspan=1>11:30 A.M.</td><td rowspan=1 colspan=1>11:45 A.M.1</td><td rowspan=1 colspan=1>2:30 P.M.</td></tr><tr><td rowspan=1 colspan=1>Davis Boulevard</td><td rowspan=1 colspan=1>11:15 A.M.</td><td rowspan=1 colspan=1>12:00 P.M.</td><td rowspan=1 colspan=1>12:15 P.M.</td><td rowspan=1 colspan=1>1:00 P.M.</td></tr><tr><td rowspan=1 colspan=1>Church Road</td><td rowspan=1 colspan=1>11:30 A.M.</td><td rowspan=1 colspan=1>12:15 P.M.</td><td rowspan=1 colspan=1>12:30 P.M.</td><td rowspan=1 colspan=1>1:15 P.M.</td></tr><tr><td rowspan=1 colspan=1>Forest Street</td><td rowspan=1 colspan=1>12:30 P.M.</td><td rowspan=1 colspan=1>1:15 P.M.</td><td rowspan=1 colspan=1>1:30 P.M.</td><td rowspan=1 colspan=1>2:15 P.M.</td></tr><tr><td rowspan=1 colspan=1>Sycamore Avenue</td><td rowspan=1 colspan=1>1:00 P.M.</td><td rowspan=1 colspan=1>1:45 P.M.</td><td rowspan=1 colspan=1>2:00 P.M.</td><td rowspan=1 colspan=1>2:45 P.M.</td></tr></table>

![](images/ce78697eba5bf54994789caa5621858e5a6d54ffdf40d80edc219940cb628d71.jpg)  
Figure 4: Case study on a train-schedule question. Left: clean image. Right: evidence-ablated image after adversarial masking.

## B.9 Distribution of selected data.

Table 13 compares the category distribution of the full training set, the globally ranked subset, and the subsets selected by RankMix. Global top-20% selection concentrates heavily on a few categories with high scaled-IIG, especially Tables/Charts, while substantially reducing coverage of other categories. In contrast, RankMix preserves a much broader category distribution across different selection ratios and remains closer to the full training pool. This result supports our design choice that mixed global and category-wise ranking can focus on visually dependent examples without causing severe diversity collapse.

Table 13: Category distribution of selected training examples.
<table><tr><td>Selection</td><td>Non-Geo Math</td><td>Geometric</td><td>Tables/Charts</td><td>Science</td><td>Spatial</td><td>Broader STEM</td><td>Social Science</td><td>Commonsense</td><td>Total</td></tr><tr><td>Full dataset</td><td>14901</td><td>10631</td><td>6189</td><td>3475</td><td>1922</td><td>956</td><td>676</td><td>120</td><td>38870</td></tr><tr><td>RankMix 10%</td><td>1122</td><td>382</td><td>1349</td><td>428</td><td>298</td><td>164</td><td>121</td><td>23</td><td>3887</td></tr><tr><td>RankMix 20%</td><td>2471</td><td>1131</td><td>2235</td><td>857</td><td>521</td><td>304</td><td>214</td><td>41</td><td>7774</td></tr><tr><td>RankMix 30%</td><td>3860</td><td>2092</td><td>2974</td><td>1241</td><td>720</td><td>423</td><td>295</td><td>56</td><td>11661</td></tr><tr><td>Global Top-20%</td><td>1244</td><td>306</td><td>4125</td><td>720</td><td>529</td><td>463</td><td>307</td><td>80</td><td>7774</td></tr></table>

## B.10 Case Study

Figures 4, 5, 6, and 7 show representative examples of train-schedule, counting, table, and genetics questions, respectively. The images in Figures 4, 5, and 6 are densely filled with content, and the image in Figure 7 contains large blank areas. For Figure 4, we show the responses of Qwen2.5-VL-3B and Qwen2.5-VL-7B on the evidence-ablated image. For Figures 5, 6, and 7, the response is generated by the Qwen2.5-VL-3B EVAN policy on the evidence-ablated image and passes both the likelihood filter and the length filter, so it is used as a negative in the clean-image update.

## Question of Figure 4.

Look at the train schedule. Colleen just missed the 10:15 AM train at River Avenue.   
Calculate the waiting time until the next train arrives.

## Qwen2.5-VL-3B on the evidence-ablated image.

<think>   
The table shows the train schedule at River Avenue.   
Look at the first two trains on the left side of the table,   
the first train departs at 10:00 AM and the second train departs at 10:15 AM.   
So, the waiting time until the next train arrives is from 00:00 to 00:15. Therefore, the waiting time is 15 minutes.   
The waiting time is 15 minutes.   
</think>   
\boxed{15}

## Qwen2.5-VL-7B on the evidence-ablated image.

<think>

The train schedule shows the times for a freight train at different stations. Colleen missed the 10:15 AM train at River Avenue. To find the waiting time until the next train arrives, I need to look at the next station in the schedule, which is Jackson Street.

The next train at Jackson Street is scheduled for 10:45 A.M., which is 30 minutes after the missed train at River Avenue.

Therefore, the waiting time until the next train arrives is 30 minutes. </think>

\boxed{30 minutes}

![](images/37ac3bbffea5c85cfab9bab8b9c89d01d81a6b6d43d966061d4756cc82350a3a.jpg)

![](images/ca0a641c250c98e0fa5e30194a98fbff42a21a15958f58d307f3ceec38f4ea8a.jpg)  
Figure 5: Case study on a counting question. Top: clean image. Bottom: evidence-ablated image after adversarial masking.

## Counting question of Figure 5.

How many triangles are there of each color, and what is the total number of triangles?

## Retained negative from Qwen2.5-VL-3B.

<think>

To determine the number of triangles of each color and the total number of triangles in the given sequence, we need to carefully count the triangles of each color. Let’s analyze the sequence step by step:

1. \*\*Identify the colors and count the triangles:\*\*

\- Purple triangles: △, △

\- Orange triangles: △, △, △

\- Teal triangles: △, △, △, △, △, △, △

\- Blue triangles: △, △

\- Green triangles: △

2. \*\*Count each color:\*\*

- Purple triangles: There are 2 purple triangles.   
- Orange triangles: There are 3 orange triangles.   
- Teal triangles: There are 7 teal triangles.   
- Blue triangles: There are 2 blue triangles.   
- Green triangles: There are 2 green triangles.

3. \*\*Summarize the count:\*\*

\- Purple: 2

\- Orange: 3

Now, we sum these counts to find the total number of triangles:

```javascript
2(purple) + 3(orange) + 7(teal) + 2(blue) + 2(green) = 16
```

So the total number of triangles is 16.

</think>

\boxed{16}

<table><tr><td rowspan=1 colspan=3>Train schedule</td></tr><tr><td rowspan=1 colspan=1>Location</td><td rowspan=1 colspan=1>Arrive</td><td rowspan=1 colspan=1>Depart</td></tr><tr><td rowspan=1 colspan=1>Arcadia</td><td rowspan=1 colspan=1>7:50 A.M.</td><td rowspan=1 colspan=1>8:00 A.M.</td></tr><tr><td rowspan=1 colspan=1>Summerfield</td><td rowspan=1 colspan=1>9:00 A.M.</td><td rowspan=1 colspan=1>9:05 A.M.</td></tr><tr><td rowspan=1 colspan=1>Weston</td><td rowspan=1 colspan=1>10:20 A.M.</td><td rowspan=1 colspan=1>10:35 A.M.</td></tr><tr><td rowspan=1 colspan=1>Kensington</td><td rowspan=1 colspan=1>12:00 P.M.</td><td rowspan=1 colspan=1>12:10 P.M.</td></tr><tr><td rowspan=1 colspan=1>Burlington</td><td rowspan=1 colspan=1>12:55 P.M.</td><td rowspan=1 colspan=1>1:10 P.M.</td></tr><tr><td rowspan=1 colspan=1>Hampton</td><td rowspan=1 colspan=1>2:30 P.M.</td><td rowspan=1 colspan=1>2:35 P.M.</td></tr><tr><td rowspan=1 colspan=1>Vindale</td><td rowspan=1 colspan=1>4:20 P.M.</td><td rowspan=1 colspan=1>4:30 P.M.</td></tr><tr><td rowspan=1 colspan=1>Dayton</td><td rowspan=1 colspan=1>6:25 P.M.</td><td rowspan=1 colspan=1>6:40 P.M.</td></tr></table>

![](images/2ba783da20acda1ea3b710327b53a7142283ae96dc6067885f28babab1b46cd0.jpg)  
Figure 6: Case study on a table question. Left: clean image. Right: evidence-ablated image after adversarial masking.

## Table question of Figure 6.

How long is the train scheduled to stop at Burlington?

## Retained negative from Qwen2.5-VL-3B.

<think>

To determine the length of the train stop at Burlington, we need to find the difference between the departure time and the arrival time at Burlington.

From the schedule:

\- The train arrives at Burlington at 11:00 P.M.

\- The train departs from Burlington at 10:00 P.M.

The difference between these two times will give us the duration of the stop. Le caleulate it fel1

\- Departure time: 10:00 P.M.

\- The difference between 11:00 P.M. and 10:00 P.M. is 1 hour.

So, the train is scheduled to stop for 1 hour at Burlington.

</think>

## Genetics question of Figure 7.

Based on this information, what is Moxie’s phenotype for the wool color trait?   
A. black wool   
B. white wool

![](images/da5886175221b1fc6e8ba0fc6c2ce6dd78e6dfe3dfccc3aa61ba551fc280ea46.jpg)  
Figure 7: Case study on a genetics question. Top: clean image. Bottom: evidence-ablated image after adversarial masking.

## Retained negative from Qwen2.5-VL-3B.

<think>

To determine Moxie’s phenotype for the wool color trait, we need to   
understand the genetics involved. The problem states that the gene for the wool color trait (let’s denote the alleles as L for white wool and l for black wool), and the l allele is the recessive allele because it is for black wool. The fact that Moxie has the homogenous (homogeneous) genotypic condition means that Moxie’s genotype is either LL or ll. Specifically: - If Moxie has the genotype LL, this indicates that Moxie has the white allele (L) dominant over the black allele (l), because when both alleles are the same, it results in the dominant trait. Since in this problem we need to determine Moxie’s phenotype, and LL represents white wool, Moxie will have white wool as it is homozygous dominant for the white allele.   
- If Moxie’s genotype is ll, this indicates that Moxie has the recessive allele for black wool. However, the problem states that Moxie is a sheep with the homogeneous genotypic condition regarding the wool color trait which means Moxie could either be homozygous dominant (LL) or homozygous recessive (ll). Since the problem specifically says it is Moxie’s   
homogenous genotypic condition for the wool color trait, and since L is dominant over l in this scenario, the homozygous condition (LL) will result in the dominant phenotype which is white wool.   
Since Moxie has the genotype LL (assuming Moxie to be of this genotype), it will have white wool. The homozygous dominant condition (LL) results in the white phenotype.   
</think>   
\boxed{B}

Analysis. In all four examples, the masks remove the query-critical evidence while the responses stay fluent and task-directed. In the clean image of Figure 4, the River Avenue row lists 9:15 AM, 10:00 AM, 10:15 AM, and 11:00 AM, so the correct waiting time after missing the 10:15 AM train is 45 minutes. After adversarial masking, both models produce fluent but incorrect answers under the same question. The two failures are plausible, but both reflect language-prior shortcuts rather than correct table grounding. The 3B model reduces the problem to a simple subtraction between a nearby visible pair of times and ignores the requirement to find the next train after 10:15 AM at the same station. The 7B model follows a different shortcut and interprets “next” as the next station in the table, leading it to subtract across rows instead of staying on River Avenue. In both cases, the weakened visual evidence breaks the row-and-column alignment needed for correct reasoning, and the models fall back to familiar textual heuristics for schedule questions. In Figure 5, the clean image contains 20 triangles in six colors, including two pink triangles in the upper row. The mask hides both pink triangles and parts of several other cells. The response counts each color and sums the counts, which is the right procedure, but it never mentions pink and reaches a total of 16 instead of

20. In Figure 6, the mask covers the Burlington arrival time, and only the fragment 0 P.M. of the departure time remains visible. The response reads the departure as 10:00 P.M., fills in an arrival of 11:00 P.M., and reports a one-hour stop. In the clean image, the train arrives at Burlington at 12:55 P.M. and departs at 1:10 P.M., so the correct answer is 15 minutes. In Figure 7, the evidence-ablated image hides the genotype ll of Moxie, while the word homozygous and the statement that black wool is recessive remain visible. The response correctly notes that a homozygous sheep can be either LL or ll, but it then fills in the dominant genotype LL and answers white wool. Under the clean image, Moxie has the genotype ll and therefore black wool, so this fluent response follows a textual template rather than the visual evidence. These examples illustrate the type of fluent but visually ungrounded negative response that EVAN is designed to elicit and suppress during training.

Masks on blank regions. In images with large blank areas, such as Figure 7, part of the masked patches fall on the blank background. This happens because the patch score in Eq. (7) is a gradient magnitude, which measures how strongly the scaled-IIG score responds to the pixels of a patch rather than how much visual content the patch contains. Vision transformers repurpose tokens in low-information background areas to store global image information [Darcet et al., 2024], and large multimodal models allocate high attention to a few visual tokens that are irrelevant to the text [Kang et al., 2025]. Blank patches whose tokens play these roles can therefore receive high patch scores even though they show no visual content. In content-dense images such as Figures 5 and 6, the masked patches fall on the content itself, and in Figure 7 the mask still hides the genotype ll needed to answer the question.

## C Limitations

This work shares limitations common to current large multimodal models and RLVR methods. Although benchmark improvements indicate better performance on evaluated image-text reasoning tasks, they do not fully resolve the broader challenge of reliable visual grounding in more diverse or open-ended settings. In addition, RLVR is inherently tied to the reward design and evaluation protocol, which capture only part of multimodal reasoning quality and may not fully reflect faithfulness, robustness, or calibration. More broadly, strong results on curated benchmarks should not be interpreted as a complete solution to the general limitations of current multimodal reasoning systems.

## D Broader Impact

This work studies improved visual grounding for multimodal reasoning models. More reliable use of visual evidence may benefit applications such as education, scientific assistance, and accessibility tools. At the same time, stronger multimodal reasoning does not remove model errors and may increase over-trust in outputs that appear well grounded. More capable multimodal systems could also be misused in high-stakes or persuasive settings. We therefore view this work as a step toward more reliable LMMs, but not as a guarantee of safety or correctness in deployment.