# Beyond the Stability-Exploration Dilemma: Environmental Regularization for LLM Policy Optimization

Xianlei Zhou<sup>1</sup>, Xiangdi Meng<sup>1</sup>, Yu He<sup>1,2</sup>\*, Tianyu Qi<sup>3</sup>, Shuyan Guan<sup>1</sup>, Xianli Zhang<sup>1</sup>, Jian Zhang<sup>4</sup>, Xin Li<sup>1</sup>, Qika Lin<sup>5</sup>, Jun Liu<sup>2</sup>

<sup>1</sup>AMAP, Alibaba Group <sup>2</sup>Xi’an Jiaotong University <sup>3</sup>JD.com <sup>4</sup>Beijing Normal University <sup>5</sup>National University of Singapore {zhouxianlei.zxl,gaolin.hy}@alibaba-inc.com

## Abstract

Policy optimization (PO) for Large Language Models faces a stability–exploration trade-off, currently mediated by an action-side Policy-KL regularizer. This puts practitioners in a double bind: keeping Policy-KL constrains response behavior and consumes the action-side exploration budget, while dropping it leaves the optimization without an explicit drift control. We argue for an alternative that breaks the dilemma by moving regularization to the input side. As training progresses, the distribution over training queries induced by the current policy drifts unchecked from its pre-RL reference distribution.

Concretely, Environment-Regularized Policy Optimization (ERPO) introduces a Query-KL (QKL) term that bounds this query distribution shift, together with a dataset-static reference-derived per-query weight that biases each per-query update toward queries typical under the reference. The QKL gradient flows strictly through the query likelihood; the response score function used by policy-gradient estimators does not appear in the QKL term, so QKL exerts no direct gradient pressure on the response distribution—exploration is preserved. ERPO plugs into GRPO/PPO/REINFORCEstyle pipelines without additional forward passes. On six mathematical reasoning benchmarks, ERPO replaces the standard Policy-KL regularizer while achieving effective control over query distribution drift, delivering stronger accuracy and substantially more stable behavior under high-temperature decoding and longhorizon training.Our source code are available at https://github.com/alibaba/ERPO.

## 1 Introduction

Policy optimization (PO) methods have become the de facto recipe for post-training large language models (LLMs), spanning trust-region style updates (TRPO/PPO) and preference-based objectives (DPO) together with broader RLHF/RLAIF variants (Schulman et al., 2015b, 2017; Ouyang et al., 2022; Bai et al., 2022; Rafailov et al., 2023). Despite impressive progress in mathematical reasoning and beyond, practitioners still face a persistent dilemma: how to trade off training stability against effective exploration. In long-horizon runs, optimization noise and distribution shift tend to accumulate, leading to oscillations and occasional collapses.

We argue that a key and under-controlled source of instability is environment non-stationarity induced by the query distribution. Even with a fixed training corpus, the model’s own sequence likelihood over training queries co-evolves with the policy: as θ updates, the likelihood the model assigns to each prompt drifts, altering the effective training environment and amplifying gradient variance. This input-side non-stationarity mirrors classic RL settings in which either the initial-state distribution or the transition kernel drifts over time; non-stationary and robust RL therefore advocate explicit distributional control (Padakandla, 2021; Iyengar, 2005; Nilim and El Ghaoui, 2005). A related lesson from imitation learning is that policy updates induce covariate (state) shift, motivating interactive data aggregation such as DAgger/AggreVaTe (Ross et al., 2011; Ross and Bagnell, 2014).

Recent LLM work formalizes prompt distributions (EVA, Align-Pro (Ye et al., 2024; Trivedi et al., 2025)) or reweights training data (StablePrompt, WPO (Kwon et al., 2024; Zhou et al., 2024)), while mainstream RLHF PO focuses on action-side Policy-KL to an SFT reference (Schulman et al., 2015b, 2017; Ouyang et al., 2022); neither directly constrains the query distribution. Empirically (Figure 1), under a fixed Policy-KL budget the batch-estimated Query-KL rises steadily while Policy-KL stays flat—the policy-induced query distribution $\rho _ { \theta }$ drifts unchecked from its pre-RL reference $\rho _ { \theta _ { 0 } }$

![](images/448bea9de6305024b7883a08700847643ae098a0f997fd1761254c10fb7511b0.jpg)  
Figure 1: KL losses during GRPO training. The Query-KL (dark) rises while the Policy-KL (light) stays low, showing action-only KL does not stabilize the query process.

We treat the model’s query likelihood as the input-side statistic to regularize. We introduce Query-KL regularization (QKL), a penalty on the KL divergence between the current policy-induced query distribution $\rho _ { \theta }$ and a pre-RL reference $\rho _ { \theta _ { 0 } }$ , limiting inter-round drift of this likelihood while leaving the action space free to explore. In parallel, we propose a lightweight reference-derived per-query weight that biases each per-query update toward queries typical under $\rho _ { \theta _ { 0 } }$ , reducing estimator variance and improving robustness under high-temperature decoding—where LLMs are especially sensitive to the long tail of decoding distributions (Holtzman et al., 2020; Wang et al., 2023). Both components are estimator-agnostic and compatible with GRPO/PPO/REINFORCE-style implementations with minimal changes. Figure 2 sketches ERPO: on top of GRPO we replace the usual Policy-KL with a Query-KL term, and during advantage computation we reweight per-query contributions by a reference-derived prior, yielding an environmentaware update while preserving action-side exploration.

We make four main contributions. (1) Queryenvironment control: We treat the model’s query likelihood as the regularizable input-side statistic, combining Query-KL (QKL) to bound its drift from a pre-RL reference $\rho _ { \theta _ { 0 } }$ with a dataset-static reference-derived per-query weight to reduce variance and tame high-temperature behavior. (2) Estimator-agnostic instantiation: The method adds a QKL term and a reference-derived perquery weight on top of GRPO/PPO/REINFORCEstyle pipelines with minimal changes. (3) Stability evaluation: We assess RL stability via multitemperature sampling paired with a multi-metric suite (Pass@k, Pass@1, Avg@k), enabling comprehensive capability and robustness evaluation. (4) Empirical gains: Across diverse reasoning benchmarks, the approach consistently improves accuracy.

## 2 Related Works

## 2.1 Reinforcement Learning with Verifiable Rewards (RLVR)

Reinforcement Learning with Verifiable Rewards represents a paradigm shift from traditional RLHF approaches by leveraging automatically verifiable outcomes rather than human preference annotations. This approach is particularly powerful for domains where ground truth can be objectively determined, such as mathematical reasoning, code generation, and logical problem solving. Models like AlphaCode (Li et al., 2022) and recent mathematical reasoning (Jeannotte and Kieran, 2017; Xia et al., 2025) systems leverage execution results and correctness verification as direct reward signals, eliminating the need for expensive human annotation.

Process Reward Models (PRMs) have emerged as a sophisticated extension of RLVR, where intermediate steps in reasoning processes are evaluated and rewarded based on their correctness (Uesato et al., 2022; Lightman et al., 2023). Recent developments include tool-augmented reasoning systems (Schick et al., 2023) and self-verification approaches (Kojima et al., 2022), which combine language models with external verification tools to enable automatic reward computation for broader task domains.

While RLVR provides scalable and consistent training signals compared to subjective human preferences, it introduces unique challenges in handling high variance from sparse rewards and potential reward hacking behaviors. These stability issues motivate the need for robust training methodologies that can effectively leverage verifiable rewards while maintaining training stability.

## 2.2 Reinforcement Learning Stability in Language Model Training

The stability of reinforcement learning algorithms in language model training has become a critical research area due to unique challenges posed by discrete action spaces, large parameter spaces, and complex reward landscapes (Sutton et al., 1998).

![](images/4c5983cf6757201ec240663675236c705650f238c8ccad930fce7e3fa54a1b59.jpg)  
Figure 2: The Proposed ERPO Overview. (a) For each query, the policy and reference induce current and reference query samplers, and we pre-compute a Query-KL to penalize environment drift. (b) For each query, the policy samples a response group scored by the reward model to produce the standard GRPO learning signal. (c) On top of GRPO we replace response-KL with pre-computed Query-KL and weight within-query advantages by the query’s occurrence probability, yielding an environment-aware update.

Recent works have identified specific stability issues including reward hacking (Gao et al., 2023) and the alignment tax problem (Dai et al., 2025), where policy optimization can degrade downstream performance while improving target metrics. Distribution shift during training has been recognized as a fundamental source of instability in policy gradient methods (Reddy et al., 2020). In language model contexts, this manifests as shifts in the query distribution during training, leading to high variance in gradient estimates and potential policy collapse (Wen et al., 2024). Existing approaches primarily focus on action-space regularization through trust region methods (Schulman et al., 2015a) and KL divergence penalties between current and reference policies.

Despite progress in understanding RL stability issues, there remains a notable gap in explicitly managing the input query distribution during training. Most current approaches focus on output regularization rather than addressing environmental shifts at the input level, leaving query distribution management as an underexplored avenue for improving training stability.

## 3 Preliminaries

## 3.1 RLVR setting and notation

We consider a standard generative–verify (RLVR) setting for training a large language model (LLM) with parameters $\theta .$ Given a query $q \in \mathcal { Q }$ , the LLM defines a response policy $\pi _ { \theta } ( o \mid q )$ over $o \in \mathcal { O }$ and a verifier returns a scalar reward $g ( q , o ) \in \mathbb { R }$ The underlying RL objective is

$$
J _ { \mathrm { R L } } ( \theta ) = \mathbb { E } _ { q \sim \rho _ { t r a i n } , o \sim \pi _ { \theta } ( \cdot | q ) } \left[ g ( q , o ) \right] ,\tag{1}
$$

where $\rho _ { \mathrm { t r a i n } }$ denotes the query distribution in the training set.

In practice, group-based or advantage-based variants of Eq. (1) are widely used (e.g., GRPO (Shao et al., 2024), RLOO (Ahmadian et al., 2024), DAPO (Yu et al., 2025)); our method is compatible with any such estimator. For concreteness in experiments we adopt the group-relative formulation of GRPO:

$$
A _ { \theta } ^ { \mathrm { G R P O } } ( q , o ^ { ( k ) } ) = \frac { g \bigl ( q , o ^ { ( k ) } \bigr ) - \mathrm { m e a n } ( \mathbf { g } ) } { \mathrm { s t d } ( \mathbf { g } ) } ,\tag{2}
$$

where $\mathbf { g } = \{ g ( q , o ^ { ( k ) } ) \} _ { k = 1 } ^ { K }$ are the rewards of K responses sampled from $\pi _ { \boldsymbol { \theta } } ( \cdot \ | \ \boldsymbol { q } )$ for the same query.

## 3.2 Policy-induced query distribution and environment drift

During LLM-RL training, the queries seen at each step are jointly shaped by the training corpus, curriculum design, difficulty filters, and active sampling schedulers.

Rather than tying the analysis to such pipelinespecific factors, we introduce a policy-only notion: $\rho _ { \theta }$ is the policy-induced query distribution, defined by $\rho _ { \theta } ( q ) = P _ { \theta } ( q )$ — the model’s autoregressive sequence likelihood of $q$ treated as a self-terminating token sequence (see Section 4 for the autoregressive form). This model-induced distribution is distinct from the exogenous $\rho _ { \mathrm { t r a i n } } .$ , which supplies queries to training batches and remains fixed in our experiments.

By construction, $\rho _ { \theta }$ depends only on $\theta ,$ and the pre-RL reference $\rho _ { \theta _ { 0 } }$ inherits whatever data composition shaped $\pi _ { \theta _ { 0 } }$ (pretraining, SFT, or prior RL stages). As $\theta$ is updated, $\rho _ { \theta }$ drifts away from $\rho _ { \theta _ { 0 } }$ without explicit constraint, this drift is unchecked. We call this policy-induced environment drift, and quantify it by the KL divergence

$$
\begin{array} { r l } & { \mathrm { E n v S h i f t } ( \theta ) : = \mathrm { K L } \big ( \rho _ { \theta } \big \| \rho _ { \theta _ { 0 } } \big ) } \\ & { \qquad = \mathbb { E } _ { q \sim \rho _ { \theta } } \Big [ \log \frac { \rho _ { \theta } ( q ) } { \rho _ { \theta _ { 0 } } ( q ) } \Big ] . } \end{array}\tag{3}
$$

Empirically (Figure 1), under a fixed action-level KL budget the batch-estimated Query-KL keeps rising throughout training while the response-level Policy-KL stays nearly flat—constraining only the action distribution does not stabilize the input/query process. This motivates the two components in Section 4: a query-level KL that bounds environment drift, and a per-query weighting scheme that stabilizes the empirical objective under arbitrary pipeline biases.

## 4 Method

We treat the pre-RL model-induced query distribution $\rho _ { \theta _ { 0 } }$ as the reference environment. As θ changes, $\rho _ { \theta }$ may drift while the exogenous ρ<sub>train</sub> remains fixed; our Environment-Regularized Policy Optimization (ERPO) bounds the former while leaving response-side exploration unconstrained.

Working assumption (A1). We treat as a premise—not a theorem—that maintaining alignment between the model-induced $\rho _ { \theta }$ and its pre-RL reference $\rho _ { \theta _ { 0 } }$ better preserves the generalization capabilities inherited from prior training stages than allowing unconstrained drift. A1 is examined empirically in Section 5.

Query likelihood. Our method relies on the autoregressive sequence likelihood the model assigns to a query. For a query $q$ tokenized as $( x _ { 1 } , \dots , x _ { T } )$ , write $\begin{array} { r } { P _ { \theta } ( q ) \triangleq \prod _ { t } P _ { \theta } ( x _ { t } \mid x _ { < t } ) } \end{array}$ for this sequence likelihood and $\ell _ { \theta } ( q ) \triangleq$ log $P _ { \theta } ( q ) =$ $\sum$ log $P _ { \theta } ( x _ { t } \mid x _ { < t } )$ for its log form, with $P _ { \theta _ { 0 } } ( q )$ and $\ell _ { \theta _ { 0 } } ( q )$ defined analogously under the reference.

The reference $\ell _ { \theta _ { 0 } }$ is computed once over the training set and cached, while $\ell _ { \theta }$ is read off the per-step PG forward pass; the cached $\ell _ { \theta _ { 0 } }$ table and the per-step $\ell _ { \theta }$ are the only inputs ERPO needs beyond what the underlying PG estimator already computes (see Appendix F for full computational notes).

## 4.1 Population objective and empirical loss

Let $\begin{array} { r l r } { \bar { g } _ { \theta } ( q ) } & { { } : = } & { \mathbb { E } _ { o \sim \pi _ { \theta } ( . | q ) } [ g ( q , o ) ] } \end{array}$ be the perquery expected reward, so that $J _ { \mathrm { R L } } ( \theta )$ $\mathbb { E } _ { q \sim \rho _ { \mathrm { t r a i n } } } [ \bar { g } _ { \theta } ( q ) ]$ . ERPO adds a query-level regularizer $\mathcal { R } _ { \mathrm { q u e r y } } ( \theta )$ penalizing the divergence between the model-induced $\rho _ { \theta }$ and $\rho _ { \theta _ { 0 } }$ :

$$
J _ { \mathrm { E R P O } } ( \theta ) : = J _ { \mathrm { R L } } ( \theta ) - \alpha \mathcal { R } _ { \mathrm { q u e r y } } ( \theta ) ,\tag{4}
$$

where $\alpha \ > \ 0$ controls regularization strength. On a mini-batch $B = \{ q _ { i } \} _ { i = 1 } ^ { m }$ , we additionally reweight per-query contributions by a dataset-static, reference-derived weight $w _ { B } ( q )$ (Section 4.2); together with a batch-level KL estimate $\widehat { \mathcal { R } } _ { \mathrm { q u e r y } } ( \theta )$ this gives the empirical loss

$$
\begin{array} { c l } { { \widehat { L } _ { \mathrm { E R P O } } ( \theta ) : = - \displaystyle \frac { 1 } { m } \sum _ { q \in { \cal B } } w _ { \cal B } ( q ) \bar { g } _ { \theta } ( q ) } } \\ { { + \alpha \widehat { \mathcal { R } } _ { \mathrm { q u e r y } } ( \theta ) . } } \end{array}\tag{5}
$$

$w _ { B }$ enters only at the estimator level; it shapes the SGD update direction toward queries typical under $\rho _ { \theta _ { 0 } }$ but does not enter the target J<sub>ERPO</sub>.

## 4.2 Query-level KL and query reweighting

Query-level KL. We instantiate the regularizer as the KL from the current query distribution to the reference:

$$
\begin{array} { r l } & { \mathcal { R } _ { \mathrm { q u e r y } } ( \theta ) : = \mathrm { K L } \big ( \rho _ { \theta } \big | \big | \rho _ { \theta _ { 0 } } \big ) } \\ & { \qquad = \mathbb { E } _ { q \sim \rho _ { \theta } } \big [ \log \rho _ { \theta } ( q ) - \log \rho _ { \theta _ { 0 } } ( q ) \big ] . } \end{array}\tag{6}
$$

QKL decouples regularization from exploration: the penalty acts solely on the query distribution through $\ell _ { \theta } ( q )$ , while imposing no direct constraint on $\pi _ { \theta } ( o \mid q )$ . Conventional Policy-KL, in contrast, restricts response behavior and consumes the action-side exploration budget.

Proposition 1 (Structural Decoupling). For autoregressive $\pi \theta ,$ , the gradient of $\mathcal { R } _ { \mathrm { q u e r y } } ( \theta )$ admits the closedform

$$
\begin{array} { r l } & { \nabla _ { \boldsymbol { \theta } } \mathcal { R } _ { \mathrm { q u e r y } } ( \boldsymbol { \theta } ) = \mathbb { E } _ { \boldsymbol { q } \sim \rho _ { \boldsymbol { \theta } } } \Bigl [ \bigl ( \ell _ { \boldsymbol { \theta } } ( \boldsymbol { q } ) - \ell _ { \boldsymbol { \theta _ { 0 } } } ( \boldsymbol { q } ) \bigr ) } \\ & { \qquad \quad \cdot \nabla _ { \boldsymbol { \theta } } \ell _ { \boldsymbol { \theta } } ( \boldsymbol { q } ) \Bigr ] , } \end{array}\tag{7}
$$

whichflows strictly through $\nabla _ { \boldsymbol { \theta } } \ell _ { \boldsymbol { \theta } } ( \boldsymbol { q } )$ ; the response score function $\nabla _ { \theta }$ log $\pi _ { \theta } ( o \mid q )$ used by PG estimators does not appear in the QKL loss, so the regularizer exerts no direct gradient pressure on the response distribution. Proofin Appendix H.

In our fixed-dataset implementation, we evaluate a K3-style mini-batch surrogate using the per-step $\ell _ { \theta } ( q )$ and cached $\ell _ { \theta _ { 0 } } ( q )$ . It regularizes model-induced query-likelihood drift without treating model likelihood as external query frequency. The resulting $\widehat { \mathcal { R } } _ { \mathrm { q u e r y } } ( \theta )$ is used in Eq. (5).

Query reweighting. Instead of optimizing the standard RL objective in Eq. (1) under the empirical query distribution $\rho _ { \mathrm { t r a i n } }$ , ERPO targets the reference-aligned query prior $\rho _ { \theta _ { 0 } }$ :

$$
J _ { \mathrm { E R P O } } ( \theta ) = \mathbb { E } _ { q \sim \rho _ { \theta _ { 0 } } , o \sim \pi _ { \theta } ( \cdot | q ) } [ g ( q , o ) ] .\tag{8}
$$

Because stochastic training samples queries from ρ<sub>train</sub>, this objective can be written as an importance-weighted objective with ideal weight

$$
w ^ { \star } ( q ) = \frac { \rho _ { \theta _ { 0 } } ( q ) } { \rho _ { \mathrm { t r a i n } } ( q ) } .\tag{9}
$$

Under approximately uniform sampling from a dataset of size N, $w ^ { \star } ( q ) = N \rho _ { \theta _ { 0 } } ( q )$ , so the query weight only needs to preserve the relative ordering of the cached reference-induced query prior. We therefore use a bounded, dataset-static weight $w ( q ) \propto \ell _ { \theta _ { 0 } } ( q )$ , leading to

$$
\begin{array} { r l } & { L _ { \mathrm { E R P O } } ( \theta ) = - \mathbb { E } { \underset { o \sim \pi _ { \theta } ( \cdot | q ) } { \mathbb { E } } } \left[ w ( q ) g ( q , o ) \right] } \\ & { \quad \quad \quad + \alpha \mathcal { R } _ { \mathrm { q u e r y } } ( \theta ) . } \end{array}\tag{10}
$$

The derivation is given in Appendix I.

PG-compatible surrogate. ERPO is agnostic to the inner PG estimator: any surrogate of the form $\nabla _ { \boldsymbol { \theta } } \bar { g } _ { \boldsymbol { \theta } } ( \boldsymbol { q } ) \approx \mathbb { E } _ { o } [ u _ { \boldsymbol { \theta } } ( \boldsymbol { q } , o ) A _ { \boldsymbol { \theta } } ^ { \star } ( \boldsymbol { q } , o ) \nabla _ { \boldsymbol { \theta } }$ log $\pi _ { \boldsymbol { \theta } } ( _ { O } \mid q ) ]$ recovers GRPO $( u _ { \theta } \equiv 1 , A ^ { \star }$ from Eq. (2)), PPO (clipped ratios), or REINFORCE (sample reward) by appropriate choice of $u _ { \theta } , A _ { \theta } ^ { \star }$ . Plugging into Eq. (5) gives the ERPO mini-batch PG loss

$$
\begin{array} { l } { \displaystyle \mathcal { L } _ { \mathrm { P G } } ( \theta ) : = - \frac { 1 } { m } \sum _ { q \in B } \frac { w _ { B } ( q ) } { K } \sum _ { o \in \mathcal { G } ( q ) } u _ { \theta } ( q , o ) A _ { \theta } ^ { \star } ( q , o ) } \\ { \displaystyle ~ + \alpha \widehat { \mathcal { R } } _ { \mathrm { q u e r y } } ( \theta ) , } \end{array}\tag{11}
$$

where $\mathcal G ( q )$ is the K-response group sampled for q. Per-algorithm specializations (GRPO/PPO/REINFORCE) and the resulting loss decompositions are summarized in Appendix G.

## 5 Experiments

## 5.1 Experimental Setup

Training We conduct experiments on mathematical reasoning tasks using Level 3–5 problems from the MATH dataset (Hendrycks et al., 2021), totaling approximately 8.5K examples. These are used to evaluate our proposed ERPO method, in comparison with the vanilla GRPO baseline. As described in Appendix A, the model must wrap its intermediate reasoning in <think></think> tags, and place the final answer inside \boxed {}.

Evaluation We follow standard practice and assess performance on six widely used benchmarks: AIME24, AIME25, AMC, MATH500 (Hendrycks et al., 2021), Minerva (Lewkowycz et al., 2022), and OlympiadBench (He et al., 2024). Prior work typically reports Avg@K (Yu et al., 2025), Pass@1 (Liu et al., 2025), and Pass@K (Hao et al., 2025) after RLVR training, often without specifying or controlling the inference-time sampling temperature. This omission can substantially affect reported performance and render results across studies not directly comparable. In preliminary experiments, we found that inference-time sampling temperature has a significant impact on performance, and that the effect intensifies as training progresses. To control for this factor, we fix the number of training steps across all models and evaluate at temperatures from 0.1 to 1.5; performance is then aggregated over this range.

Implementation Details We conduct all experiments using the EasyR1 framework (Zheng et al., 2025), training the Qwen2.5-Math-7B and Qwen2.5-32B model (Yang et al., 2024; Qwen et al., 2025) with both GRPO and ERPO algorithms. Following prior work (Liu et al., 2025), we set the maximum sequence length to 3K tokens. For each problem, we sample eight responses at an inference temperature of 1.0. The rollout batch size is set to 512, and the update batch size to 128, for a total of 240 training steps. Token-level loss is applied throughout training. To ensure a fair comparison, we adopt the default KL divergence coefficient of 0.01.

## 5.2 Main Results

Figure 3 summarizes Avg@32 accuracy on six mathematical reasoning benchmarks, averaged over sampling temperatures from 0.1 to 1.5. ERPO consistently outperforms GRPO, with gains of up to 14.9% and an overall average improvement of 6.2%, highlighting its enhanced capability. Table 1 presents the detailed results for each benchmark, grouped by evaluation metric (e.g., Pass@1, Pass@K).

For both GRPO and ERPO, the prompts are identical to those used during training, whereas the Qwen base model adopts the default configuration from Dr.GRPO (Liu et al., 2025) to ensure optimal performance. Consistent with the aggregated results in Figure 3 and Table 1, ERPO surpasses GRPO across all evaluation metrics, achieving improvements of 6.2% in Avg@32, 3.64% in Pass@32, and 5.69% in Pass@1. We also applied the concept of ERPO to other RLVR algorithms and observed similarly effective gains; details are provided in Appendix E.

![](images/15e1589200cdaa7fd87af5c973be3fdf59dd9bfaadae7d25844ee0bc45fa61b4.jpg)  
Figure 3: Avg@32 over Sampling Temperatures on Mathematical Reasoning Tasks

Table 1: Performance comparison across mathematical reasoning benchmarks. Best results per column are highlighted in bold.
<table><tr><td colspan="8">Mean Avg@32</td></tr><tr><td>Method</td><td>AIME24</td><td>AIME25</td><td>AMC</td><td>MATH500</td><td>Minerva</td><td>Olympiad</td><td>Avg.</td></tr><tr><td>Base</td><td>0.087</td><td>0.030</td><td>0.246</td><td>0.340</td><td>0.058</td><td>0.099</td><td>0.143</td></tr><tr><td>GRPO</td><td>0.174</td><td>0.072</td><td>0.398</td><td>0.528</td><td>0.207</td><td>0.266</td><td>0.274</td></tr><tr><td>ERPO</td><td>0.218</td><td>0.110</td><td>0.478</td><td>0.677</td><td>0.214</td><td>0.316</td><td>0.336</td></tr><tr><td colspan="8">Mean Pass@32</td></tr><tr><td>Base</td><td>0.373</td><td>0.206</td><td>0.674</td><td>0.764</td><td>0.349</td><td>0.411</td><td>0.463</td></tr><tr><td>GRPO</td><td>0.471</td><td>0.287</td><td>0.768</td><td>0.850</td><td>0.516</td><td>0.558</td><td>0.575</td></tr><tr><td>ERPO</td><td>0.509</td><td>0.342</td><td>0.820</td><td>0.904</td><td>0.500</td><td>0.593</td><td>0.611</td></tr><tr><td colspan="8">Mean Pass@1</td></tr><tr><td>Base</td><td>0.090</td><td>0.038</td><td>0.264</td><td>0.342</td><td>0.062</td><td>0.099</td><td>0.149</td></tr><tr><td>GRPO</td><td>0.169</td><td>0.084</td><td>0.398</td><td>0.533</td><td>0.201</td><td>0.263</td><td>0.275</td></tr><tr><td>ERPO</td><td>0.207</td><td>0.091</td><td>0.477</td><td>0.679</td><td>0.217</td><td>0.320</td><td>0.332</td></tr></table>

## 5.3 Training Dynamics

Figure 4 illustrates the training dynamics of the ERPO method. For both approaches, the sampling accuracy on the training set remains largely consistent; however, their divergence from the reference model exhibits markedly different trajectories.

In GRPO, constraints are imposed on the action distribution, causing the query distribution to drift away from the reference model at a substantially faster rate. Consequently, the KL divergence at the query level is an order of magnitude greater than at the policy level.

This imbalance leads to pronounced discrepancies in performance between the training and evaluation datasets. In contrast, ERPO applies constraints directly to the query distribution and adjusts the loss according to the probability of the given problem. This design both limits the degree of divergence from the reference model during training and, by leveraging the independence between the problem and the response, allows unconstrained exploration at the policy level. A quantitative analysis can be found in Appendix C. As a result, ERPO achieves superior generalization performance on general problems.

To assess the stability of long-term RL training, we scale the training steps up to 1K and monitor changes in model performance over time. As shown in the figure 5 and 6, GRPO remains stable for sampling temperatures below 1.0 until approximately 240 steps (epoch=15). However, a pronounced performance degradation is first observed in the high-temperature sampling regime after 400 steps, and subsequently propagates to encompass sampling across all temperatures as the steps increase.

![](images/8246d20222af5fc94832c2fb4c29205d6ea3279bf75cc5ebf2a655c2848e1a21.jpg)

![](images/746d7494bcd43c519fb0a5bc4aeac249dafc1c4db7edce3cbcfceee91661db58.jpg)

![](images/d823dc9ff9b4f3ab6b60712046d769ef2ceb2b3e3634ca2dfd09037dc00c70c8.jpg)

![](images/ac1149d6d7296ae04f70bf3d88ef0357766223954e9b94d556d6aa0aaa02df14.jpg)

![](images/75c16aaf8c4db7a991f97158d51725ff0f992918f44e459a7096ddc97ec1cca2.jpg)  
Figure 4: Training Dynamics on ERPO

![](images/1216553562c761b8a169492de1aa4fc12f634e6b5bcea534a8a94a616314e0b3.jpg)

![](images/0fc973a37332e5ef80f4ff719577858a3320f7d28b4606b4d2f0df9028076e0b.jpg)

![](images/1bbab58c87e0e3ba9a0b04c09f1203765090396b49dfd698bf8aca9e161692c2.jpg)

![](images/8b3389f6385f97723913173a8afa3cda1ccae8620629386f7f2aac3aa6978f15.jpg)  
Figure 5: Training Dynamics on Long-term RL  
Figure 6: Performance Variation Across Training Steps

In contrast, ERPO exhibits a modest performance decline; however, the overall deterioration is substantially smaller, and its performance even improves within the high-temperature range. Figure 6 presents the complete training trajectories for both GRPO and ERPO. Although ERPO is not entirely immune to the collapse phenomenon that may occur during extended training—manifested as a sudden increase in entropy and a loss of sampling capability—it consistently outperforms vanilla GRPO and achieves a comparable degree of policy distribution constraint without relying on an explicit policy-based KL divergence term.

## 5.4 Analysis

Ablation Study We conduct ablation studies on the MATH500 benchmarks to assess reasoning efficiency. Table 2 summarizes the results for several commonly used sampling temperatures. Figure 7 further provides the complete performance–temperature variation curves across different experimental settings, along with the corresponding training dynamics.

Mechanisms Without modifying other hyperparameters, replacing the policy-based KL divergence with query-based KL divergence yields the best overall performance<sup>1</sup>, with an average improvement of 15.9% over GRPO. In contrast, GRPO with policy-based KL divergence shows its highest performance only at a temperature of 1.0 (see Figure 7(a)).

To further investigate, an ablation study is conducted on the two mechanisms of ERPO with their effects evaluated using KL divergence and entropy (Table 3 and Figure 7(d)). The term $w _ { B ( s ) }$ downweights gradients from low-probability queries, which often lead to low-probability responses, thereby increasing gradient variance and entropy (Quantitative analysis is provided in Appendix D). Introducing $w _ { B ( s ) }$ allows sufficient training while concurrently reducing the policy KL divergence.

Different regularization strengths α also exert a significant influence on performance. As the constraint strength increases $( \mathrm { e } . \mathrm { g } . , \alpha = 5 \times 1 0 ^ { - 2 } )$ the model achieves further improvements in overall performance (see Table 2). It is worth noting that we did not conduct an exhaustive search for the optimal α; instead, we retained the default value to ensure a relatively fair comparison.

Table 2: Performance Comparison Under Different Experimental Settings
<table><tr><td rowspan="2">Base Model</td><td rowspan="2">Method</td><td rowspan="2"> $_ { \pmb { \alpha } }$ </td><td rowspan="2"> $w ( s )$ </td><td rowspan="2">Rollout Count</td><td colspan="6">Temperature Metrics</td></tr><tr><td>0.1</td><td>0.6</td><td>1</td><td>1.5</td><td> $\leq 1 . 0$ </td><td>1.2–1.5</td></tr><tr><td>Baseline</td><td></td><td></td><td></td><td></td><td>52.40</td><td>46.80</td><td>32.80</td><td>0.40</td><td>44.44</td><td>6.15</td></tr><tr><td rowspan="5">Qwen-7B</td><td>GRPO</td><td> $1 \times 1 0 ^ { - 2 }$ </td><td></td><td>8</td><td>66.80</td><td>68.40</td><td>73.80</td><td>0.40</td><td>68.80</td><td>12.50</td></tr><tr><td>GRPO*</td><td> $1 \times 1 0 ^ { - 2 }$ </td><td> $\checkmark$ </td><td>8</td><td>78.20</td><td>76.80</td><td>71.20</td><td>7.60</td><td>76.14</td><td>23.79</td></tr><tr><td>ERPO</td><td> $1 \times 1 0 ^ { - 2 }$ </td><td></td><td>8</td><td>81.60</td><td>81.60</td><td>79.00</td><td>2.60</td><td>80.90</td><td>38.00</td></tr><tr><td></td><td> $1 \times 1 0 ^ { - 2 }$ </td><td>√</td><td>8</td><td>79.40</td><td>80.60</td><td>75.20</td><td>8.60</td><td>78.74</td><td>37.90</td></tr><tr><td>ERPO</td><td> $5 \times 1 0 ^ { - 3 }$ </td><td>√</td><td>8</td><td>53.80</td><td>60.60</td><td>66.20</td><td>15.40</td><td>59.94</td><td>39.30</td></tr><tr><td rowspan="2">Qwen-7B under n = 16</td><td></td><td> $5 \times 1 0 ^ { - 2 }$ </td><td>√</td><td>8</td><td>78.80</td><td>81.00</td><td>76.00</td><td>15.00</td><td>79.00</td><td>43.35</td></tr><tr><td>GRPO</td><td> $1 \times 1 0 ^ { - 2 }$ </td><td></td><td>16 16</td><td>73.00</td><td>79.20</td><td>75.00</td><td>10.60</td><td>75.22</td><td>39.75</td></tr><tr><td rowspan="2">Qwen-32B</td><td>ERPO</td><td> $1 \times 1 0 ^ { - 2 }$ </td><td> $\checkmark$ </td><td></td><td>80.40</td><td>78.80</td><td>74.40</td><td>56.20</td><td>77.82</td><td>66.25</td></tr><tr><td>GRPO ERPO</td><td> $1 \times 1 0 ^ { - 2 }$   $1 \times 1 0 ^ { - 2 }$ </td><td> $\checkmark$ </td><td>8 8</td><td>81.60 85.00</td><td>82.40 84.80</td><td>81.20 83.60</td><td>25.20 80.80</td><td>81.62 84.60</td><td>57.20 82.80</td></tr></table>

Note: Within the Qwen-7B $( n = 8 )$ setting, the best result in each column is highlighted in bold, while the secondbest result is underlined. No highlighting is applied to the Qwen-7B $( n = 1 6 )$ or Qwen-32B results. The first three Qwen-7B rows constitute the ablation study, comparing GRPO, GRPO<sup>∗</sup>, and ERPO without query reweighting. The following three rows constitute the hyperparameter study, evaluating ERPO with query reweighting under different values of α. The columns $\leq 1 . 0$ and 1.2–1.5 report the mean accuracy (Acc) over the corresponding temperature ranges. The w(s) column indicates whether query reweighting is applied (✓) or not (—). GRPO<sup>∗</sup> denotes GRPO using only query reweighting.

![](images/d78acb328e455c0eb88e428ebfe68c7b7a49bbe2ebe2e43cf1e13b3303b920df.jpg)  
(a)

![](images/6d20e83890cf8e14e3e7501680b273fdb1ccc2471854b1fe6b110c766cce33ca.jpg)  
(b)

![](images/448b89c68fa29a7ac00dd2e0e236ac6cb23e16974ce859bdf6f88de0c7620666.jpg)  
(c)

![](images/d711f969f947305f173a0aba930a31ab3dc8771eefea040c82977e7068810f02.jpg)  
(d)  
Figure 7: Pass@1 accuracy and training dynamics under different settings: (a)–(c) Model performance at various temperatures on MATH500; (d) Policy-KL divergence variation with GRPO using only Query-KL.

Table 3: Influence of Query-KL and Query-Reweighting on Training Stability
<table><tr><td>Method</td><td>Query-KL</td><td>Policy-KL</td><td>Entropy</td></tr><tr><td>GRPO  $( \mathbf { A v g } @ ^ { ( a ) } { 3 } )$ </td><td>0.9679</td><td>0.0601</td><td>0.5063</td></tr><tr><td>GRPO  $\mathrm { w } / w _ { B ( s ) }$ </td><td>0.5933</td><td>0.0113</td><td>0.2782</td></tr><tr><td>GRPO w/Query-KL</td><td>0.0041</td><td>0.1001</td><td>0.5674</td></tr><tr><td>ERPO (Avg@3)</td><td>0.0828</td><td>0.0728</td><td>0.4244</td></tr></table>

Rollouts We also analyze the effect of the number of samples per query. By increasing the sampling number to 16, we achieve the best performance, with the average Pass@1 rising to 74.6%. A higher sampling count also significantly improves sampling stability at high temperatures (see Table 2), without a noticeable increase in divergence from the reference model. Moreover, increasing the sampling count facilitates ERPO-based models in acquiring the correct reasoning format

more effectively. <sup>2</sup>

## 6 Conclusion

By analyzing the coupling between the environment and the policy space in large language models, we decouple parameter regularization from the optimization objective during training. Specifically, we employ query-level KL divergence to bound the drift of the policy-induced query distribution $\rho _ { \theta }$ from a pre-RL reference $\rho _ { \theta _ { 0 } }$ further reweight the advantage by a dataset-static reference-derived per-query weight, biasing updates toward queries typical under $\rho _ { \theta _ { 0 } }$ and preventing premature convergence to suboptimal solutions. Experiments across multiple mathematical reasoning benchmarks demonstrate that the proposed ERPO method can achieve comparable

KL divergence control without explicit policy regularization, while delivering superior performance. Furthermore, by sampling at different temperatures, we examine the evolution of sampling capability over long-term RL training, providing additional evidence of ERPO’s stability during training.

## Limitations

Our experiments focus primarily on mathematical reasoning benchmarks and Qwen-family models, so the extent to which ERPO transfers to broader instruction-following, dialogue, code-generation, and multilingual settings remains to be validated. ERPO also relies on estimating query-level likelihoods or prevalence statistics during training; the quality and computational cost of these estimates may vary with the data-selection mechanism and model scale. Finally, we did not conduct an exhaustive sweep over the regularization coefficient, leaving more systematic hyperparameter analysis for future work.

## References

Arash Ahmadian, Chris Cremer, Matthias Gallé, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet Üstün, and Sara Hooker. 2024. Back to basics: Revisiting reinforce style optimization for learning from human feedback in llms. Preprint, arXiv:2402.14740.

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, Nicholas Joseph, Saurav Kadavath, Jackson Kernion, Tom Conerly, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Tristan Hume, and 12 others. 2022. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862.

Juntao Dai, Taiye Chen, Yaodong Yang, Qian Zheng, and Gang Pan. 2025. Mitigating reward overoptimization in rlhf via behavior-supported regularization. arXiv preprint arXiv:2503.18130.

Leo Gao, John Schulman, and Jacob Hilton. 2023. Scaling laws for reward model overoptimization. In International Conference on Machine Learning, pages 10835–10866. PMLR.

Yaru Hao, Li Dong, Xun Wu, Shaohan Huang, Zewen Chi, and Furu Wei. 2025. On-policy rl with optimal reward baseline. arXiv preprint arXiv:2505.23585.

Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu,

Xu Han, Yujie Huang, Yuxiang Zhang, and 1 others. 2024. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems. arXiv preprint arXiv:2402.14008.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Math. Cornell University - arXiv,Cornell University - arXiv.

Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. 2020. The curious case of neural text degeneration. In International Conference on Learning Representations (ICLR).

Garud N. Iyengar. 2005. Robust dynamic programming. Mathematics of Operations Research, 30(2):257– 280.

Doris Jeannotte and Carolyn Kieran. 2017. A conceptual model of mathematical reasoning for school mathematics. Educational Studies in mathematics, 96(1):1–16.

Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. 2022. Large language models are zero-shot reasoners. Advances in neural information processing systems, 35:22199– 22213.

Minchan Kwon, Gaeun Kim, Jongsuk Kim, Haeil Lee, and Junmo Kim. 2024. Stableprompt: Automatic prompt tuning using reinforcement learning for large language model. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 9868–9884, Miami, Florida, USA. Association for Computational Linguistics.

Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, and 1 others. 2022. Solving quantitative reasoning problems with language models. Advances in neural information processing systems, 35:3843–3857.

Yujia Li, David Choi, Junyoung Chung, Nate Kushman, Julian Schrittwieser, R’emi Leblond, Tom Eccles, James Keeling, Felix Gimeno, Agustin Dal Lago, and 1 others. 2022. Competition-level code generation with alphacode. Science, 378(6624):1092–1097.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. 2023. Let’s verify step by step. In The Twelfth International Conference on Learning Representations.

Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. 2025. Understanding r1-zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783.

Arnab Nilim and Laurent El Ghaoui. 2005. Robust control of markov decision processes with uncertain transition matrices. Operations Research, 53(5):780– 798.

Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. 2022. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems (NeurIPS).

Sindhu Padakandla. 2021. A survey of reinforcement learning algorithms for dynamically varying environments. ACM Computing Surveys, 54(6):127:1– 127:25.

Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, and 25 others. 2025. Qwen2.5 technical report. Preprint, arXiv:2412.15115.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. 2023. Direct preference optimization: Your language model is secretly a reward model. In Advances in Neural Information Processing Systems (NeurIPS).

Siddharth Reddy, Anca Dragan, Sergey Levine, Shane Legg, and Jan Leike. 2020. Learning human objectives by evaluating hypothetical behavior. In International conference on machine learning, pages 8020–8029. PMLR.

Stephane Ross and J. Andrew Bagnell. 2014. Reinforcement and imitation learning via interactive no-regret learning. arXiv preprint arXiv:1406.5979.

Stephane Ross, Geoffrey J. Gordon, and J. Andrew Bagnell. 2011. A reduction of imitation learning and structured prediction to no-regret online learning. In Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics (AISTATS), volume 15 of Proceedings of Machine Learning Research, pages 627–635, Fort Lauderdale, FL, USA. PMLR.

Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. 2023. Toolformer: Language models can teach themselves to use tools. Advances in Neural Information Processing Systems, 36:68539–68551.

John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. 2015a. Trust region policy optimization. In International conference on machine learning, pages 1889–1897. PMLR.

John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan, and Pieter Abbeel. 2015b. Trust region policy optimization. In Proceedings of the 32nd International Conference on Machine Learning (ICML), volume 37 of Proceedings ofMachine Learning Research, pages 1889–1897, Lille, France. PMLR.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300.

Richard S Sutton, Andrew G Barto, and 1 others. 1998. Reinforcement learning: An introduction. MIT press Cambridge.

Prashant Trivedi, Souradip Chakraborty, Avinash Reddy, Vaneet Aggarwal, Amrit Singh Bedi, and George K. Atia. 2025. Align-pro: A principled approach to prompt optimization for llm alignment. In Proceedings ofthe AAAI Conference on Artificial Intelligence (AAAI).

Jonathan Uesato, Nate Kushman, Ramana Kumar, Francis Song, Noah Siegel, Lisa Wang, Antonia Creswell, Geoffrey Irving, and Irina Higgins. 2022. Solving math word problems with process-and outcomebased feedback. arXiv preprint arXiv:2211.14275.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2023. Self-consistency improves chain of thought reasoning in language models. In International Conference on Learning Representations (ICLR).

Jiaxin Wen, Ruiqi Zhong, Akbir Khan, Ethan Perez, Jacob Steinhardt, Minlie Huang, Samuel R Bowman, He He, and Shi Feng. 2024. Language models learn to mislead humans via rlhf. arXiv preprint arXiv:2409.12822.

Shijie Xia, Xuefeng Li, Yixin Liu, Tongshuang Wu, and Pengfei Liu. 2025. Evaluating mathematical reasoning beyond accuracy. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 27723–27730.

An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong Tu, Jingren Zhou, Junyang Lin, and 1 others. 2024. Qwen2. 5-math technical report: Toward mathematical expert model via self-improvement. arXiv preprint arXiv:2409.12122.

Feng Yao, Liyuan Liu, Dinghuai Zhang, Chengyu Dong, Jingbo Shang, and Jianfeng Gao. 2025. Your efficient rl framework secretly brings you off-policy rl training.

Ziyu Ye, Rishabh Agarwal, Tianqi Liu, Rishabh Joshi, Sarmishta Velury, Quoc V. Le, Qijun Tan, and Yuan Liu. 2024. Scalable reinforcement posttraining beyond static human prompts: Evolving alignment via asymmetric self-play. arXiv preprint arXiv:2411.00062.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, and 1 others. 2025. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476.

Yaowei Zheng, Junting Lu, Shenzhi Wang, Zhangchi Feng, Dongdong Kuang, and Yuwen Xiong. 2025. Easyr1: An efficient, scalable, multi-modality rl training framework. https://github.com/hiyouga/ EasyR1.

Wenxuan Zhou, Ravi Agrawal, Shujian Zhang, Sathish Reddy Indurthi, Sanqiang Zhao, Kaiqiang Song, Silei Xu, and Chenguang Zhu. 2024. Wpo: Enhancing rlhf with weighted preference optimization. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 8328–8340, Miami, Florida, USA. Association for Computational Linguistics.

## A Prompt

{{ content | trim }} You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed {}.

## B Variation of Metrics with Temperature

Figure 8 illustrates the model performance across different evaluation metrics and sampling temperatures. Our approach reduces the performance gap between different sampling temperatures, while increasing the likelihood of sampling correct outputs.

## C Analysis of Reward Hacking

In the course of our experiments, we observed a severe reward hacking phenomenon when training with the baseline GRPO method. Specifically, while the model consistently achieved high rewards on the training data, its performance on the evaluation set often plateaued or even degraded during the later stages of optimization. This pronounced discrepancy suggests that the model overfits to the specific characteristics of the reward signal during training sampling, failing to generalize to the standard decoding distribution used during inference.

To quantitatively investigate this issue, we monitored the Train–Evaluation Consistency throughout the training process. We periodically evaluated both the training accuracy and the evaluation accuracy every ten optimization steps. To ensure the robustness of our inference metrics, evaluation was conducted using vLLM under two different Tensor Parallelism settings (TP1 and TP2), a factor which has been shown in previous work (Yao et al., 2025) to impact model performance.

Table 4 summarizes the average performance gap across six key checkpoints (Steps 40, 80, 120, 160, 200, and 240). The results confirm our hypothesis:

• GRPO exhibits a substantial average gap of 6.47%, indicating a significant misalignment between training and inference behaviors. Notably, as shown in the detailed trajectories in Table 5, GRPO’s evaluation accuracy drops sharply at Step-240 (from ≈ 75% to 58.4%) despite maintaining high training accuracy, a classic signature of reward hacking.

• ERPO, in contrast, demonstrates superior consistency. It reduces the average Train–Eval gap by approximately 51% (from 6.47% to 3.14%).

This significant reduction in the performance gap indicates that ERPO effectively regularizes the training process, preventing the model from exploiting spurious patterns in the reward function and ensuring that improvements in training translate reliably to inference performance.

## D Correlation analysis between query and response probabilities

We sampled over 8K questions from the training dataset at a temperature of 1.0 and independently computed the negative log-likelihood (NLL) for both the prompt and response parts:

$$
\mathrm { N L L } ( x ) = - \sum \log p ( x )
$$

where x denotes the generation probabilities of tokens in the prompt or response. The resulting histogram is shown in Figure 9. We observe a positive correlation between the NLL of the prompt and that of the response. For 95% of the training samples (where the NLL of the prompt is less than 300), the correlation coefficient is close to 1. Lowprobability responses appearing in positive samples contribute to an increase in entropy during training.

## E ERPO On Different Algorithms

Additional experiments were conducted on DAPO(Yu et al., 2025) and RLOO(Ahmadian et al., 2024) with and without the global KL divergence constraint, yielding absolute improvements of 10.24% and 2.28% at temperatures below 1.0, respectively (see Table 6). These findings demonstrate that the proposed method can achieve significant gains when applied to other RLVR algorithms.

## F Query Likelihood and the Cached Reference

This appendix expands the query likelihood machinery and computational notes referenced from Section 4.

Autoregressive sequence likelihood. For a query q with tokenization $\boldsymbol { x } = ( x _ { 1 } , \dots , x _ { T } )$ , the model’s sequence log-likelihood under parameters

![](images/0036bb77ba3813e5badd8389747a5305a7fd04122d653095861b167241be1643.jpg)

![](images/df926c089584bb6ec60502e4c087de98dd4e5684259da569fda650416ec92001.jpg)

![](images/94e28d4210005ba85ff3fa8eab2b6aa555a1621caf4623f142e3fd196415bcb9.jpg)  
Figure 8: Variation of Metrics with Temperature

Table 4: Quantification of Reward Hacking via Train–Inference Gap. The table compares the average accuracy during training sampling versus inference decoding. A larger gap indicates severe reward hacking (overfitting to training dynamics). ERPO reduces this gap by ≈ 51%, demonstrating robust generalization.
<table><tr><td>Method</td><td>Avg. Train Acc</td><td>Avg. Eval@TP1</td><td>Avg. Eval@TP2</td><td>Gap@TP1</td><td>Gap@TP2</td><td>Avg. Gap</td></tr><tr><td>GRPO</td><td>77.3</td><td>69.8</td><td>70.1</td><td>7.5</td><td>5.5</td><td>6.47</td></tr><tr><td>ERPO</td><td>77.1</td><td>74.1</td><td>73.8</td><td>3.0</td><td>3.3</td><td>3.14</td></tr><tr><td>Improvement</td><td>一</td><td>一</td><td>一</td><td>↓ 4.5 (60%)</td><td>↓ 2.2 (40%)</td><td>↓ 3.33 (51%)</td></tr></table>

Likelihood Relationship between Query and Response  
![](images/e79992e9a5f2791eea64e21dee85561f6bf33bfb35ee7d60507982a8cc29885b.jpg)  
Figure 9: Likelihood Relationship between Query and Response

θ is

$$
\ell _ { \theta } ( q ) \ \triangleq \ \log P _ { \theta } ( x ) \ = \ \sum _ { t = 1 } ^ { T } \log P _ { \theta } ( x _ { t } \mid x _ { < t } ) .\tag{12}
$$

This is well-defined for any token sequence, regardless of whether $q$ is sampled on-policy from $\rho _ { \theta }$ , drawn from a fixed dataset, or written by a human; computing $\ell _ { \theta } ( q )$ requires a single forward pass of the LLM. The reference $\ell _ { \theta _ { 0 } } ( q )$ is defined analogously under $\pi _ { \theta _ { 0 } }$

Computational notes governing ERPO’s zerooverhead design. Two distinctions in how $\ell _ { \theta }$ and $\ell _ { \theta _ { 0 } }$ are obtained make ERPO essentially free relative to the underlying PG estimator:

$\ell _ { \theta _ { 0 } } ( q )$ is computed once over the entire training set using the reference model $\pi _ { \boldsymbol { \theta } _ { 0 } } ,$ prior to RL, and cached. The per-query table is reused throughout training and does not change. Under backpropagation it is treated as a constant.

$\ell _ { \theta } ( q )$ is computed per training step using the current θ. This forward pass is already performed by the underlying PG estimator (for per-query advantage evaluation), so QKL and QW reuse it at no additional forward cost.

The cached $\ell _ { \theta _ { 0 } }$ table and the per-step $\ell _ { \theta }$ are the only inputs ERPO needs beyond what the underlying PG estimator already computes. In particular, the query weight $w _ { B } ( q )$ from Eq. (9) is fully precomputed and contributes no gradient or extra forward pass; only the QKL estimator $\widehat { \mathcal { R } } _ { \mathrm { q u e r y } } ( \theta )$ uses the perstep $\ell _ { \theta } ( q )$ , and even there the value is read from the existing PG forward pass.

Table 5: Trajectory of Train vs. Inference Accuracy. Detailed performance recorded at 40-step intervals. Note the divergence in GRPO at Step-240, where Eval accuracy drops significantly while Train accuracy remains high—a clear sign of reward hacking. ERPO maintains consistency throughout.
<table><tr><td>Model / Metric</td><td>Step-0</td><td>Step-40</td><td>Step-80</td><td>Step-120</td><td>Step-160</td><td>Step-200</td><td>Step-240</td><td>Avg.</td></tr><tr><td>GRPO (Baseline)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Train Acc</td><td>44.48</td><td>76.41</td><td>76.09</td><td>78.29</td><td>78.95</td><td>79.44</td><td>76.70</td><td>72.91</td></tr><tr><td>Eval@TP1</td><td>31.20</td><td>73.20</td><td>76.00</td><td>73.40</td><td>72.20</td><td>73.60</td><td>58.40</td><td>65.43</td></tr><tr><td>Eval@TP2</td><td>30.60</td><td>74.00</td><td>74.80</td><td>74.20</td><td>76.60</td><td>75.80</td><td>66.20</td><td>67.46</td></tr><tr><td>ERPO (Ours)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Train Acc</td><td>44.55</td><td>75.63</td><td>76.53</td><td>78.66</td><td>80.63</td><td>78.41</td><td>81.46</td><td>73.70</td></tr><tr><td>Eval@TP1</td><td>31.20</td><td>74.00</td><td>77.20</td><td>77.00</td><td>78.40</td><td>78.60</td><td>78.40</td><td>70.69</td></tr><tr><td>Eval@TP2</td><td>30.60</td><td>72.60</td><td>77.00</td><td>76.60</td><td>77.40</td><td>78.60</td><td>80.20</td><td>70.43</td></tr></table>

Table 6: Performance Comparison Under Different Experimental Settings
<table><tr><td rowspan="2">Base Model Method</td><td rowspan="2"></td><td rowspan="2">D_KL</td><td rowspan="2">Rollout</td><td colspan="6">Temperature</td></tr><tr><td>0.1</td><td>0.6</td><td>1</td><td>1.5</td><td>≤1.0</td><td>1.2-1.5</td></tr><tr><td>Baseline</td><td></td><td></td><td>一</td><td>52.4</td><td>46.8</td><td>32.8</td><td>0.4</td><td>44.44</td><td>6.15</td></tr><tr><td rowspan="6">Qwen-7B</td><td>GRPO</td><td>Policy</td><td>8 16</td><td>66.8 73.0</td><td>68.4 79.2</td><td>73.8 75.0</td><td>0.4 10.6</td><td>68.8 75.22</td><td>12.5 39.75</td></tr><tr><td>ERPO</td><td>Query</td><td>8 16</td><td>79.4 80.4</td><td>80.6 78.8</td><td>75.2 74.4</td><td>8.6 56.2</td><td>78.74 (+9.94) 77.82 (+2.6)</td><td>37.9 66.25</td></tr><tr><td>DAPO DAPO+ERPO</td><td>Query</td><td>8</td><td>62.0 80.2</td><td>77.4 79.4</td><td>65.4 75.8</td><td>5.6 20.2</td><td>68.16 78.4 (+10.24)</td><td>14.0 36.93</td></tr><tr><td>RLOO</td><td>Policy</td><td>8</td><td>77.6</td><td>78.4</td><td>75.4</td><td>12.4</td><td>77.28</td><td>35.93</td></tr><tr><td>RLOO+ERPO</td><td>Query</td><td></td><td>81.2 81.6</td><td>80.4 82.4</td><td>79.4</td><td>17.6</td><td>79.56 (+2.28)</td><td>40.8</td></tr><tr><td>GRPO ERPO</td><td>Policy Query</td><td>8</td><td>85.0</td><td>84.8</td><td>81.2 83.6</td><td>25.2 80.8</td><td>81.62 84.6 (+2.98)</td><td>57.2 82.8</td></tr></table>

Note: The columns ≤1.0 and 1.2–1.5 show the mean accuracy (Acc) over the corresponding temperature ranges. Values in parentheses indicate improvements over baseline methods. Best results per column are highlighted in bold, second-best results are underlined.

## G PG-Compatible Surrogate: Per-Algorithm Specializations

The ERPO mini-batch loss ${ \mathcal { L } } _ { \mathrm { P G } } ( \theta )$ in Eq. (11) is instantiated by a particular choice of actionlevel weight $u _ { \theta } ( q , o )$ and advantage $A _ { \theta } ^ { \star } ( q , o )$ , summarized below. The two ERPO modifications— adding the query-level KL term and reweighting the outer per-query sum by $w _ { B } ( q )$ —are orthogonal to the choice of $( u _ { \theta } , A _ { \theta } ^ { \star } )$ and act only on the outer query loop.

GRPO. Set $u _ { \theta } \equiv 1$ and $A _ { \theta } ^ { \star } = A _ { \theta } ^ { \mathrm { G R P O } }$ from Eq. (2). Substituting yields

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { E R P O - G R P O } } ( \theta ) : = - \displaystyle \frac { 1 } { m } \sum _ { q \in B } \frac { w _ { B } ( q ) } { K } \sum _ { o \in \mathcal { G } ( q ) } } \\ { A _ { \theta } ^ { \mathrm { G R P O } } ( q , o ) \log \pi _ { \theta } ( o \mid q ) } \\ { + \alpha \widehat { \mathcal { R } } _ { \mathrm { q u e r y } } ( \theta ) . } \end{array}\tag{13}
$$

All GRPO engineering details (reward normalization, group size K, sampling temperature, etc.) remain unchanged.

PPO. We set

$$
u _ { \theta } ( q , o ) = \mathrm { c l i p } \left( \frac { \pi _ { \theta } ( o \mid q ) } { \pi _ { \theta _ { \mathrm { o l d } } } ( o \mid q ) } , 1 - \epsilon , 1 + \epsilon \right)
$$

and $A _ { \theta } ^ { * }$ to the (GAE-based or standard) PPO advantage. The inner sum recovers the PPO clipped surrogate, with the outer $w _ { B } ( q )$ and the added α $\widehat { \mathcal { R } } _ { \mathrm { q u e r y } }$ providing the ERPO wrapping.

REINFORCE. Set $u _ { \theta } ~ \equiv ~ 1$ and $A _ { \theta } ^ { \star } ( q , o ) ~ =$ $g ( q , o ) - b _ { \theta } ( q )$ for an arbitrary baseline $b _ { \theta } ( q ) ;$ the inner sum recovers the sample-wise REINFORCE estimator.

Drop-in modification recipe. Given an existing PG-style training pipeline using any of the above estimators, applying ERPO requires three steps: (i) precompute $\ell _ { \theta _ { 0 } }$ once over the dataset (Appendix F); (ii) replace the per-query outer weight $1 / m$ with $w _ { B } ( q ) / m$ from Eq. (9); (iii) add $\chi \widehat { \mathcal { R } } _ { \mathrm { q u e r y } } ( \theta )$ (K3- estimated from the per-step $\ell _ { \theta }$ and the cached $\ell _ { \theta _ { 0 } } )$ to the loss. No additional forward passes, no architectural changes, and no modification to the inner PG estimator’s clip/baseline logic are required.

## H Proof of Proposition 1: Structural Decoupling of Regularization and Exploration

This appendix provides the formal proof of Proposition 1 stated in Section 4.2.

Proposition 1 (restated). For the autoregressive generation process parameterized $b y \theta ,$ , the gradient of the Query-KL regularizer $\mathcal { R } _ { \mathrm { q u e r y } } ( \theta )$ strictly flows through the query log-likelihood ∇ ℓ (q) and is entirely orthogonal to the response policy score function $\nabla _ { \theta }$ log $\pi _ { \boldsymbol { \theta } } ( o \mid q )$

Proof. By definition,

$$
\mathcal { R } _ { \mathrm { q u e r y } } ( \theta ) = \sum _ { q } \rho _ { \theta } ( q ) \bigl ( \log \rho _ { \theta } ( q ) - \log \rho _ { \theta _ { 0 } } ( q ) \bigr ) .
$$

Differentiating with respect to $\theta$ and applying the product rule,

$$
\begin{array} { l } { { \displaystyle \nabla _ { \theta } \mathcal { R } _ { \mathrm { q u e r y } } ( \theta ) } } \\ { ~ = \sum _ { q } \nabla _ { \theta } \rho _ { \theta } ( q ) \big ( \log \rho _ { \theta } ( q ) } \\ { ~ - \log \rho _ { \theta _ { 0 } } ( q ) \big ) } \\ { { \displaystyle ~ + \sum _ { q } \rho _ { \theta } ( q ) \nabla _ { \theta } \big ( \log \rho _ { \theta } ( q ) } } \\ { ~ - \log \rho _ { \theta _ { 0 } } ( q ) \big ) . } \end{array}\tag{14}
$$

Second sum collapses. Since $\rho _ { \theta _ { 0 } }$ does not depend on θ, we have $\nabla _ { \theta }$ log $\rho \theta _ { 0 } ( q ) = 0$ . Applying the identity $\nabla _ { \theta }$ log $\rho _ { \theta } ( q ) = \nabla _ { \theta } \rho _ { \theta } ( q ) / \rho _ { \theta } ( q )$ to the

remaining term gives

$$
\begin{array} { l l l } { \displaystyle \sum _ { q } \rho _ { \theta } ( q ) \nabla _ { \theta } \log \rho _ { \theta } ( q ) = \sum _ { q } \nabla _ { \theta } \rho _ { \theta } ( q ) } \\ { \displaystyle \qquad } \\ { \displaystyle \qquad } \\ { \displaystyle \qquad } \\ { \displaystyle \qquad } \end{array}\tag{15}
$$

where the third equality uses the fact that $\rho _ { \theta }$ is a probability distribution and therefore sums to one identically in θ.

First sum simplifies via the log-derivative trick. Applying $\nabla _ { \theta } \rho _ { \theta } ( q ) \ : = \ : \rho _ { \theta } ( q ) \nabla _ { \theta } \log \rho _ { \theta } ( q )$ and writing $\ell _ { \theta } ( q ) : = \log \rho _ { \theta } ( q )$ and $\ell _ { \theta _ { 0 } } ( q ) \ : =$ log $\rho _ { \theta _ { 0 } } ( q )$ , we obtain the closed-form gradient

$$
\begin{array} { r l } & { \nabla _ { \theta } \mathcal { R } _ { \mathrm { q u e r y } } ( \theta ) } \\ & { \quad \quad = \mathbb { E } _ { q \sim \rho _ { \theta } } \Big [ \big ( \ell _ { \theta } ( q ) - \ell _ { \theta _ { 0 } } ( q ) \big ) } \\ & { \qquad \cdot \nabla _ { \theta } \ell _ { \theta } ( q ) \Big ] . } \end{array}\tag{16}
$$

Structural conclusion. The right-hand side of Eq. (16) depends on θ only through the query log-likelihood $\ell _ { \theta } ( q )$ (and its gradient $\nabla _ { \boldsymbol { \theta } } \ell _ { \boldsymbol { \theta } } ( \boldsymbol { q } ) ) .$ no factor of the response policy score function $\nabla _ { \theta }$ log $\pi _ { \theta } ( o \mid q )$ appears. Equivalently, perturbations to $\pi _ { \theta } ( o \mid q )$ that leave the query marginal $\rho _ { \theta }$ unchanged incur zero QKL gradient and are unconstrained by $\mathcal { R } _ { \mathrm { q u e r y } }$ . This establishes the structural decoupling between input-environment regularization (carried by QKL) and response-side exploration (left untouched). □

## I Derivation of Query Reweighting

The standard RL objective optimizes the expected reward under the empirical training query distribution $\rho _ { \mathrm { t r a i n } }$ . ERPO instead considers a referencealigned objective in which queries are sampled from the cached reference-induced query prior $\rho _ { \theta _ { 0 } }$

$$
J _ { \mathrm { E R P O } } ( \theta ) = \mathbb { E } _ { q \sim \rho _ { \theta _ { 0 } } , o \sim \pi _ { \theta } ( \cdot | q ) } \left[ g ( q , o ) \right] .\tag{17}
$$

Expanding the expectation gives

$$
J _ { \mathrm { E R P O } } ( \theta ) = \sum _ { q } \rho _ { \theta _ { 0 } } ( q ) \sum _ { o } \pi _ { \theta } ( o \mid q ) g ( q , o ) .\tag{18}
$$

However, stochastic training samples queries from the empirical distribution $\rho _ { \mathrm { t r a i n } }$ , rather than directly from $\rho _ { \theta _ { 0 } }$ . To express the same objective under

$\rho _ { \mathrm { t r a i n } }$ , we multiply and divide by $\rho _ { \mathrm { t r a i n } } ( q )$

$$
\begin{array} { l }  \displaystyle { J _ { \mathrm { E R P O } } ( \theta ) = \sum _ { q } \rho _ { \mathrm { t r a i n } } ( q ) \frac { \rho _ { \theta _ { 0 } } ( q ) } { \rho _ { \mathrm { t r a i n } } ( q ) } \sum _ { o } \pi _ { \theta } ( o \mid q ) g ( q _ { \star } } \\ { \displaystyle = \mathbb { E } _ { q \sim \rho _ { \mathrm { t r a i n } } , o \sim \pi _ { \theta } ( \cdot \mid q ) } \left[ w ^ { \star } ( q ) g ( q , o ) \right] , } \end{array}\tag{19}
$$

where the ideal importance weight is

$$
w ^ { \star } ( q ) = \frac { \rho _ { \theta _ { 0 } } ( q ) } { \rho _ { \mathrm { t r a i n } } ( q ) } .\tag{20}
$$

If training queries are sampled approximately uniformly from a dataset of size N, then $\rho _ { \mathrm { t r a i n } } ( q ) = 1 / N$ . Therefore,

$$
w ^ { \star } ( q ) = \frac { \rho \theta _ { 0 } ( q ) } { \rho _ { \mathrm { t r a i n } } ( q ) } = N \rho _ { \theta _ { 0 } } ( q ) .\tag{21}
$$

Since the multiplicative constant N is shared by all queries, it does not affect the relative weighting among queries. Thus, up to a global constant,

$$
w ^ { \star } ( q ) \propto \rho _ { \theta _ { 0 } } ( q ) .\tag{22}
$$

In practice, ERPO estimates the referenceinduced query prior using a cached reference score $\ell _ { \theta _ { 0 } } ( q )$ , which assigns larger values to queries that are more favored under the pretrained reference model. Hence we use a bounded, dataset-static query weight $w ( q )$ that is monotone in $\ell _ { \theta _ { 0 } } ( q )$

$$
w ( q ) \propto \ell _ { \theta _ { 0 } } ( q ) .\tag{23}
$$

Substituting this query weight into the negative optimization objective yields the ERPO training loss

$$
\begin{array} { r l r } & { } & { L _ { \mathrm { E R P O } } ( \theta ) = - \mathbb { E } _ { q \sim \rho _ { \mathrm { t r a i n } } , o \sim \pi _ { \theta } ( \cdot | q ) } \left[ w ( q ) g ( q , o ) \right] } \\ & { } & { ~ + \alpha \mathcal { R } _ { \mathrm { q u e r y } } ( \theta ) . ~ } \end{array}
$$

where $\mathcal { R } _ { \mathrm { q u e r y } } ( \theta )$ denotes the query-level regularization term and α controls its strength.

In practice, the exact reference-induced query prior $\rho _ { \theta _ { 0 } } ( q )$ is not directly available on a finite training set. Therefore, we use the cached referencemodel log-probability as a practical monotone proxy for the relative query preference induced by the reference model. This approximation only aims to preserve the relative ordering of queries under the reference model, rather than to estimate the absolute density value. We emphasize that this implementation is not intended as an unbiased densityratio estimator; rather, it is a bounded monotone approximation to the ideal query importance weight in Eq. (20).

For each query $q _ { i } ,$ , let log $p _ { \theta _ { 0 } } ( q _ { i } )$ denote its cached sequence log-probability under the refo)erence model. We first define its negative logprobability as

$$
s _ { i } = - \log p _ { \theta _ { 0 } } ( q _ { i } ) .\tag{25}
$$

A larger reference-model probability corresponds to a smaller $s _ { i } .$ . To obtain a weight whose magnitude increases with the reference-model query probability, we use the inverse normalized negative log-probability. For a dataset with N queries, let

$$
\bar { s } = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } s _ { j } .\tag{26}
$$

We define the unbounded query weight as

$$
\tilde { w } ( q _ { i } ) = \frac { \bar { s } } { s _ { i } } .\tag{27}
$$

This normalization makes the weight dimensionless and centers its scale around the dataset average, while preserving the monotonic relationship that higher-probability queries receive larger weights. Finally, to reduce the variance of importance reweighting and prevent any single query from dominating optimization, we clip the weight to a fixed range:

$$
w ( q _ { i } ) = \mathrm { c l i p } \left( \tilde { w } ( q _ { i } ) , 0 , 2 \right) .\tag{28}
$$