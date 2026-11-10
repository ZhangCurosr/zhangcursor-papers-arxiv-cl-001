# Robust Domain-Mixture Scheduling for Multi-Teacher On-Policy Distillation

Yankai Chen <sup>1</sup> <sup>2</sup> Sida Lin <sup>3</sup> Yifan Li <sup>4</sup> Hanjun Luo <sup>5</sup> Hanrong Zhang <sup>6</sup> Zhuohan Xie <sup>2</sup> Bowei He <sup>1</sup> <sup>2</sup> Wei-Chieh Huang <sup>6</sup> Meng Ding <sup>7</sup> Wei Xue <sup>3</sup> Yike Guo <sup>3</sup> Philip S. Yu <sup>6</sup> Xue Liu <sup>1</sup> <sup>2</sup>

## Abstract

Multi-teacher on-policy distillation (MOPD) merges the capabilities of multiple domain-expert teachers, such as mathematical reasoning, code generation, and instruction following, into a single compact student. Since all domains share a limited training budget, how to allocate it across domains directly determines the quality of distillation. Existing methods allocate the budget by each domain’s learning progress, but offer no guarantee for lagging domains. We therefore introduce distributionally robust optimization (DRO) into MOPD to prioritize the worst-performing domain. However, the per-domain training signals are estimated from a few rollouts and are inherently noisy, so directly applying DRO mistakes noise for the worst case, lets occasional spikes hijack the budget, and protects inherently hard domains rather than truly regressing ones. To address these issues, we propose RDS-MOPD, which keeps the DRO objective and redesigns its three parts: it shifts the allocation only when differences clearly exceed the noise, bounds the influence of any single anomaly, and defines the worst case as the regression from each domain’s own best level, all with theoretical guarantees. Experiments show that RDS-MOPD outperforms recent competing methods and ranks first on most mathematical reasoning, code generation, and instruction-following benchmarks.

## 1. Introduction

As large language model (LLM) services become deeply embedded in various real-world applications, users often issue requests of vastly different types within one continuous session, and expect the same model to both handle every task well and keep the conversation coherent (Zhao et al., 2024). Deploying a dedicated model for each type of task multiplies the inference and maintenance cost, and breaks the coherence of the session whenever the task switches. Consequently, merging multiple expert capabilities into a single compact student model has become a fundamental requirement for large-scale deployment (Hinton et al., 2015). Multi-teacher on-policy distillation (MOPD) is a promising post-training paradigm toward this goal (Ma et al., 2026; Gao et al., 2026; Sun et al., 2026b), where the student learns from multiple domain-expert teachers simultaneously on its own sampled responses. Compared to offline sequence-level distillation, on-policy distillation provides dense supervision directly on the student’s own distribution, which alleviates the distribution shift between training and inference (Agarwal et al., 2024; Gu et al., 2024).

In MOPD, all domains (e.g., mathematics, code, and instruction following (IF)) share a limited training budget. Hence, the domain mixture, i.e., how much the student learns from each teacher at every step, directly determines the quality of distillation. Open-MOPD (Gao et al., 2026) diagnoses the capability imbalance in multi-teacher distillation, and proposes to balance the token shares and follow the perdomain KL gaps. More recently, D<sup>3</sup>-MOPD (Sun et al., 2026b) dynamically allocates the budget according to the learning progress of each domain (i.e., the remaining KL gap and its descent velocity), steering the budget toward the domains that benefit the most. However, these methods all aim to accelerate the overall progress: they only favor “which domain improves the fastest” and offer no guarantee for lagging or regressing domains. In multi-capability merging, what users perceive is often the weakest capability, and the lag of any single capability directly hurts the service quality.

To this end, we introduce distributionally robust optimization (DRO) into domain-mixture scheduling for MOPD.

DRO makes decisions against the most unfavorable case within an uncertainty set, and thereby prioritizes the worstperforming group (Sagawa et al., 2020; Lam, 2016). This idea naturally matches the goal of “leaving no capability behind”. Different from prior work that determines the pretraining data mixture offline with a proxy model (Xie et al., 2023), to the best of our knowledge, this is the first work that applies DRO to online mixture scheduling in MOPD. Nevertheless, directly applying DRO to MOPD (referred to as Vanilla DRO-MOPD) is confronted with a fundamental issue: every decision of DRO relies on the estimated per-domain training signals, which are computed from a limited number of rollouts per domain and are inherently noisy. As shown in Figure 1(a), on the training trajectory of our Vanilla DRO-MOPD implementation, the weight of the code domain surges after a KL burst and stays high until the end of training, even though code is no longer the worst-performing domain. Therefore, it is non-trivial to make reliable robust decisions under noisy training signals, and Vanilla DRO-MOPD mainly suffers from the following three instabilities (diagnosed in §3.1).

• Worst-Case Misjudgment. Vanilla DRO-MOPD moves its weights at the same rate whether the evidence is strong or weak, so it systematically up-weights the domain whose noisy estimate happens to be high. On the above trajectory, the current worst domain changes 52 times within 145 training steps, and on more than half (54%) of the non-spike steps, the gap between the worst and runner-up domains is below one noise standard deviation.

• Spike Hijacking. Every observation enters the weights of Vanilla DRO-MOPD in proportion to its magnitude and is carried forward by all later updates, so an abnormal KL spike can seize a large share of the budget. On the above trajectory, the KL of the code domain rises to about 30× its typical level within a dozen steps, and its weight nearly doubles and stays high until the end.

• Misplaced Protection. DRO is intended to protect the weak domains, but it compares the magnitude of losses, while the KL values given by different teachers naturally differ in scale. As a result, the budget flows to the domains with inherently large KL rather than the truly regressing ones: with the raw KL as the loss, Vanilla DRO-MOPD eventually allocates the whole budget to the code domain, whereas the IF domain, which regresses the most from its own best level, receives almost no protection.

To tackle these challenges, we propose a novel framework, namely Robust Domain-Mixture Schedulingfor Multi-Teacher On-Policy Distillation (RDS-MOPD). Our key observation is that the three instabilities do not stem from the idea of DRO itself; instead, Vanilla DRO-MOPD builds all three of its parts (i.e., the deviation radius, the nominal allocation, and the definition of the “worst case”) upon unreliable signals. Hence, RDS-MOPD keeps the DRO objective and redesigns its three parts with three modules. Specifically, (1) to handle worst-case misjudgment, we propose Noise-aware Radius Calibration (NRC), which calibrates the deviation radius to the noise level, so that the evidence strength decides how far the allocation may deviate. It lowerbounds the temperature of the nominal allocation via the online-estimated noise floor, so that differences within the noise range only cause bounded share changes, while truly significant ones still receive a decisive response. (2) To handle spike hijacking, we design Spike-robust Allocation Estimation (SAE), which rebuilds the nominal allocation from KL observations clipped by a log-domain Huber rule, bounding the influence of spikes while still following persistent real changes. Meanwhile, SAE re-estimates the nominal allocation at every update rather than accumulating it from previous decisions, so an anomaly no longer persists in the budget. (3) To handle misplaced protection, we propose Worst-case Domain Reweighting (WDR), which redefines the “worst case” as each domain’s regression from its own best level, and exactly solves DRO around the progressdriven nominal allocation. It thus reweights the domains toward the truly regressing ones with a bounded amount of extra budget. As shown in Figure 1(b), under the same setting, RDS-MOPD keeps re-allocating the budget throughout training, and no domain is locked in. Theoretically, we prove that NRC and SAE guarantee bounded share changes and bounded influence of anomalies, respectively, and that WDR exactly solves a KL-ball DRO problem.

![](images/2bc95932f628dc5ff99c75875a3c2723b1b76997c5248dd47b9656be0c332879.jpg)  
Figure 1. Domain weights on the training trajectory. (a) Vanilla DRO-MOPD: a code KL burst (shaded) lifts the code weight from 0.30 to 0.56, which then locks in (hatched); (b) RDS-MOPD keeps re-allocating the budget without lock-in.

We distill a 3B student model from three domain-expert teachers for mathematics, code, and instruction following, and conduct extensive experiments on six benchmarks. The results show that RDS-MOPD outperforms recent competing methods, including Open-MOPD (Gao et al., 2026) and $\mathrm { D ^ { 3 } }$ -MOPD (Sun et al., 2026b), and ranks first on the majority of benchmarks. RDS-MOPD also makes no large share reversal in training, and in a stress test that injects noise into the code teacher’s signal, it still keeps a stable mixture and protects the learning progress of the other domains. These results demonstrate both the superiority of RDS-MOPD and the effectiveness of its modules. To summarize, our main contributions are threefold:

![](images/fe15ba4e6d3c1ccc60084b54a3c649aeb6a49c2a1e4627fedd45238d04d9db11.jpg)  
Figure 2. Illustration of our RDS-MOPD framework (left) and its robust scheduler (right).

• To the best of our knowledge, we are the first to introduce DRO into domain-mixture scheduling for MOPD, and formally diagnose three instabilities of Vanilla DRO-MOPD under noise.

• We propose RDS-MOPD, a single DRO structure whose three modules fix these instabilities with theoretical guarantees.

• Extensive experiments show that RDS-MOPD ranks first on most benchmarks, and validate the effectiveness of its modules.

## 2. Preliminaries

We introduce MOPD and its domain-mixture scheduling problem. Bold lowercase letters denote vectors (e.g., q) and calligraphic letters denote sets $( \mathbf { e . g . } , \mathcal { D } ) . ~ \Delta ^ { K }$ denotes the probability simplex over K domains, $\mathbf { u } = \mathbf { 1 } / K$ the uniform allocation (1 is the all-ones vector), and $\operatorname { K L } ( \cdot \| \cdot )$ the Kullback–Leibler divergence. We index training steps by s and scheduling updates by t (omitted when clear from the context), and summarize the notations in Table 3 of Appendix A.

## 2.1. Multi-Teacher On-Policy Distillation

Let D be the set of domains (here, $\mathcal { D } = \{ \mathrm { m a t h } , \mathrm { c o d e } , \mathrm { I F } \} )$ with $K \ = \ | \mathcal { D } |$ Each domain d has a prompt pool $\mathcal { P } _ { d }$ and an RL-trained expert teacher $\pi _ { \phi _ { d } } .$ . MOPD distills all teachers into one student policy $\pi _ { \theta } .$ and routes each prompt x to the teacher of its domain $d ( x )$ . Given x, the student samples a response $y \sim \pi _ { \theta } ( \cdot \mid x )$ on-policy, and the routed teacher scores it with a single prefill. At each position $j$ of $y ,$ , the teacher computes the reverse KL $D _ { j }$ from the next-token distribution of the student to its own, restricted to the student’s top-k tokens (Gao et al., 2026). The dense token-level reward $- D _ { j }$ serves as the advantage of a criticfree policy-gradient update (Schulman et al., 2017). The distillation loss $\mathcal { L } _ { d } ( \theta )$ is the expected per-token $D _ { j }$ over the domain-d prompts and their on-policy responses. At training step s, the per-domain KL signal is computed as:

$$
m _ { d } ^ { ( s ) } = \frac { 1 } { \left| \mathcal { B } _ { d } \right| } \sum _ { y \in \mathcal { B } _ { d } } \frac { 1 } { \left| y \right| } \sum _ { j = 1 } ^ { \left| y \right| } D _ { j } ,\tag{1}
$$

where $B _ { d }$ holds the domain-d responses of step s, and $| y |$ is the length of y. As a one-batch estimate of $\mathcal { L } _ { d } , m _ { d } ^ { ( s ) }$ reuses the rewards already computed by the trainer (no extra forward pass), but is noisy, as each batch holds a limited number of rollouts per domain.

## 2.2. Domain-Mixture Scheduling

Every few training steps, a scheduler updates the domain mixture $\mathbf { q } ^ { ( t ) } \in \Delta ^ { \bar { \boldsymbol { K } } }$ from the observed KL signals. Each batch then samples prompts in proportion to $\mathbf { q } ^ { ( t ) }$ , with a random jitter (Appendix D). Between two scheduling updates, the student thus minimizes $\begin{array} { r } { \sum _ { d } q _ { d } \mathcal { L } _ { d } ( \theta ) ; } \end{array}$ ; q only decides how many prompts each domain receives, and the per-token loss is unchanged. Some schedulers fix the mixture before training, e.g., by prompt-pool size (Ma et al., 2026), or determine it offline with a proxy model (Xie et al., 2023). Others adapt to each domain’s learning progress online, e.g., Open-MOPD (Gao et al., 2026) reweights the per-domain rewards by token-share balancing and KL-gap following, and D<sup>3</sup>-MOPD (Sun et al., 2026b) resamples domains by the remaining KL gap times its descent velocity.

## 3. Our RDS-MOPD Methodology

We first formulate DRO for MOPD and diagnose Vanilla DRO-MOPD (§3.1), then present RDS-MOPD with its three modules (§3.2, Figure 2) and its overall procedure (§3.3).

## 3.1. DRO for MOPD

## 3.1.1. DRO FORMULATION FOR MOPD

DRO decides against the most unfavorable distribution within an uncertainty set around a nominal one (Ben-Tal et al., 2013; Duchi & Namkoong, 2021). In MOPD, where the student minimizes $\sum _ { d } q _ { d } \mathcal { L } _ { d }$ (§2.2), a DRO scheduler picks the most unfavorable mixture, up-weighting domains with a higher estimated loss. With a KL penalty around a nominal allocation $\mathbf { p } ,$ the DRO decision is:

$$
\mathbf { q } = \underset { \mathbf { q } ^ { \prime } \in \mathcal { U } } { \arg \operatorname* { m a x } } \ : \Big \{ \langle \mathbf { q } ^ { \prime } , \hat { \pmb { \ell } } \rangle - \frac { 1 } { \eta } \mathrm { K L } \big ( \mathbf { q } ^ { \prime } \ : \lVert \ : \mathbf { p } \big ) \Big \} ,\tag{2}
$$

where $\mathcal { U } \subseteq \Delta ^ { K }$ is the feasible set. Eqn. (2) has three parts: (i) the deviation radius, set by η (a larger η lets q move farther); (ii) the nominal allocation p, the anchor that q moves from; and (iii) the definition of the worst case, i.e., the loss estimate <sup>ˆ</sup>ℓ of $( \mathcal { L } _ { d } ) _ { d \in \mathcal { D } }$

Vanilla DRO-MOPD applies online Group DRO (Sagawa et al., 2020) to MOPD with each domain as a group, where q may reweight either the losses or the sampling. At every training step s (so t = s), it solves Eqn. (2) with $\mathcal { U } = \Delta ^ { \bar { K } }$ whose closed form is $q _ { d } \propto p _ { d } e ^ { \eta \hat { \ell } _ { d } }$ . Its nominal allocation is the previous weights, $\mathbf { p } = \mathbf { q } ^ { ( s - 1 ) }$ with $\mathbf { q } ^ { ( 0 ) } = \mathbf { u }$ , and its loss is the KL signal normalized by its initial level $I _ { d }$ (early mean of $m _ { d } ) , \hat { \ell } _ { d } ^ { ( s ) } = m _ { d } ^ { ( s ) } / I _ { d }$ , giving the exponentiatedgradient update:

$$
q _ { d } ^ { ( s ) } \propto q _ { d } ^ { ( s - 1 ) } e ^ { \eta \hat { \ell } _ { d } ^ { ( s ) } } .\tag{3}
$$

## 3.1.2. INSUFFICIENCY OF VANILLA DRO-MOPD

We model each observation as its true level times a noise factor, i.e., $m _ { d } ^ { ( s ) } = \bar { m } _ { d } ^ { ( s ) } e ^ { \xi _ { d } ^ { ( s ) } }$ , where the zero-mean log-noise $\xi _ { d }$ may be heavy-tailed and differ in scale across domains. Unrolling Eqn. (3) from $\mathbf { q } ^ { ( 0 ) } = \mathbf { u }$ , taking the log-ratio of two domains d and d<sup>′</sup> (which cancels the normalization), and writing $e ^ { \xi _ { d } } = 1 + ( e ^ { \xi _ { d } } - 1 )$ ) splits it into a signal term and a noise term summed over steps $s ^ { \prime } \leq s \colon$

$$
\begin{array} { r l } & { \log \displaystyle \frac { q _ { d } ^ { ( s ) } } { q _ { d ^ { \prime } } ^ { ( s ) } } = \underbrace { \eta \sum _ { s ^ { \prime } \leq s } \Big ( \frac { \bar { m } _ { d } ^ { ( s ^ { \prime } ) } } { I _ { d } } - \frac { \bar { m } _ { d ^ { \prime } } ^ { ( s ^ { \prime } ) } } { I _ { d ^ { \prime } } } \Big ) } _ { \mathrm { s i g n a l } } } \\ & { \quad \quad + \underbrace { \eta \sum _ { s ^ { \prime } \leq s } \Big ( \frac { \bar { m } _ { d } ^ { ( s ^ { \prime } ) } } { I _ { d } } \big ( e ^ { \xi _ { d } ^ { ( s ^ { \prime } ) } } - 1 \big ) - \frac { \bar { m } _ { d ^ { \prime } } ^ { ( s ^ { \prime } ) } } { I _ { d ^ { \prime } } } \big ( e ^ { \xi _ { d ^ { \prime } } ^ { ( s ^ { \prime } ) } } - 1 \big ) \Big ) } _ { \mathrm { n o i s e } } . } \end{array}\tag{4}
$$

Eqn. (4) shows how each part breaks under the noise. (1) Worst-case misjudgment (deviation radius). The radius, fixed by $\eta ,$ ignores the noise level, so the noise term accumulates like a random walk whose spread grows with the number of steps, and even drifts toward the noisier domain since $\mathbb { E } [ e ^ { \xi } ] \ge 1 . ( 2 )$ Spike hijacking (nominal allocation). Anchoring on the previous weights accumulates every observation, so a single observation $m _ { d } ^ { ( s _ { 0 } ) } = M$ leaves a term $\eta M / I _ { d }$ in the log-weight ratio of domain d to any other domain at all later steps, which grows linearly in M and never fades (e.g., the code burst in Figure 1(a)). (3) Misplaced protection (definition of the worst case). Let $\bar { m } _ { d } ^ { \star }$ be the lowest true level of domain d so far; the true level then splits into three factors:

$$
\bar { m } _ { d } = \underbrace { I _ { d } } _ { \mathrm { t e a c h e r ~ s c a l e } } \cdot \underbrace { \frac { \bar { m } _ { d } ^ { \star } } { I _ { d } } } _ { \mathrm { d e s c e n t } } \cdot \underbrace { \frac { \bar { m } _ { d } } { \bar { m } _ { d } ^ { \star } } } _ { \mathrm { r e g r e s s i o n } } .\tag{5}
$$

Normalizing by $I _ { d }$ removes only the teacher scale, so a domain that descended far but then regresses still ranks below a barely descended one. Thus, the problem lies not in the robust objective but in its three parts, calling for a noise-aware radius, a spike-robust nominal allocation, and a worst case defined by regression alone.

## 3.2. RDS-MOPD

RDS-MOPD keeps the robust objective and redesigns its three parts. At its t-th update, it solves:

$$
\mathbf { q } ^ { ( t ) } = \underset { \mathbf { q } \in \mathcal { U } } { \arg \operatorname* { m a x } } ~ \Big \{ \langle \mathbf { q } , \mathbf { h } \rangle - \frac { 1 } { \eta } \mathrm { K L } \big ( \mathbf { q } \| ~ \mathbf { p } \big ) \Big \} ,\tag{6}
$$

where p is the nominal allocation, h the excess loss, η the tilt strength, and U the feasible set, i.e., Eqn. (2) with h as the loss.

The nominal allocation is recomputed at every update from the progress score (Sun et al., 2026b) $r _ { d } = v _ { d } \ell _ { d } / I _ { d }$ of each domain, where $\ell _ { d }$ is the exponential moving average (EMA) of the clipped KL of domain d (§3.2.2), $\ell _ { d } / I _ { d }$ is the remaining gap, and the descent velocity $v _ { d } \geq 0$ is the relative decrease of $\ell _ { d }$ over recent steps (Appendix D):

$$
\mathbf { p } = f \mathbf { 1 } + ( 1 - K f ) \mathrm { s o f t m a x } ( \mathbf { r } / \tau ) ,\tag{7}
$$

where $f$ is the share floor and τ the temperature, so a domain with a large remaining gap that still descends fast receives a larger share. The softmax solves Eqn. (2) with r, u, and τ in place of <sup>ˆ</sup>ℓ, p, and $1 / \eta$ . Hence, τ sets the radius that turns noisy scores into shares (a larger τ keeps p closer to u), while η only bounds the step from p to q.

Accordingly, three modules redesign the three parts: (i) NRC calibrates the radius τ to the noise level (§3.2.1); (ii) SAE re-estimates p at every update from clipped observations instead of the previous weights (§3.2.2); (iii) WDR defines the worst case as each domain’s excess loss h over its best level and solves Eqn. (6) exactly (§3.2.3).

![](images/9bf13e32fab735075d8a545fa24e5d292db25538a792acb67b27031d689df0e2.jpg)  
Figure 3. Three modules on the mixture simplex. (a) NRC sizes the KL ball around u by evidence; without it, p flips between vertices. (b) SAE keeps p bounded after a code spike; without it, p is dragged to the code vertex. (c) WDR’s solution $\mathbf { q } ^ { \star }$ lies where the highest iso-line of $\langle \mathbf { q } , \mathbf { h } \rangle$ touches the KL ball of p.

## 3.2.1. NOISE-AWARE RADIUS CALIBRATION (NRC)

NRC fixes worst-case misjudgment: a difference that does not clearly exceed the noise should not cause a large change in the allocation. A common choice is the max-normalized temperature $\tau = T \cdot r _ { \operatorname* { m a x } }$ (Sun et al., 2026b), where $T$ is the base temperature and $r _ { \operatorname* { m a x } } = \operatorname* { m a x } _ { d } r _ { d }$ is the score of the leading domain (the leader). The softmax input is then $( \mathbf { r } / r _ { \operatorname* { m a x } } ) / T$ , so the leader always has a normalized score of 1, whether its lead is far beyond the noise or only a noise artifact. Near a plateau, the velocities and thus $r _ { \mathrm { m a x } }$ approach 0, so the leader switches with the noise, and each switch moves a large share. NRC therefore calibrates the radius to the uncertainty of the leading score.

Noise-Calibrated Radius. We estimate the scale of the log-noise $\xi _ { d } ~ ( \ S 3 . 1 . 2 )$ of each domain from the recent logdifferences of its clipped observations (§3.2.2). The delta method then carries this scale to the standard error (SE) of each score (Appendix D). Let σˆ be the SE of the leading score, and ϵ the noise-floor multiplier. NRC lower-bounds $r _ { \mathrm { m a x } }$ by the noise floor ϵσˆ:

$$
\tau = T \cdot \operatorname* { m a x } \left( r _ { \operatorname* { m a x } } , \epsilon \hat { \sigma } \right) .\tag{8}
$$

Thus, as the evidence weakens, τ stays large, and p moves smoothly toward u instead of following a noisy leader (Figure 3(a)).

Proposition 3.1 (Bounded Share Change below the Noise Level). Let p be given by Eqns. (7) and (8) with $\epsilon > 0$ Suppose that, at two consecutive updates, all scores satisfy $0 \leq r _ { d } \leq \zeta \hat { \sigma }$ for some $0 < \zeta \leq \epsilon ,$ with $\hat { \sigma } > 0$ taken at the same update. In this case, no score exceeds $\zeta S E s ,$ , and the noisefloor is active. Then every nominal share changes by at most $\frac { ( 1 - K f ) \zeta } { 2 \epsilon T }$

Proposition 3.1 (proved in Appendix H.1) guarantees that differences within the noise level only cause bounded share changes, and the weaker the evidence, the smaller the change.

## 3.2.2. SPIKE-ROBUST ALLOCATION ESTIMATION (SAE)

SAE fixes spike hijacking: one abnormal observation should not dominate the allocation. Re-estimating the nominal allocation at every update stops a spike from accumulating, but the spike still enters every statistic built on linear averages $( \boldsymbol { \mathrm { e } } . \boldsymbol { \mathrm { g } } . , I _ { d } , \ell _ { d } ,$ , and $v _ { d } )$ in proportion to its size. SAE therefore first clips every new observation into a range around the recent median.

Log-Domain Huber Clipping. Since the noise is multiplicative (§3.1.2), we clip each new observation $m _ { d } ^ { ( s ) }$ in the log domain, i.e., by a ratio to the median of the raw observations over the previous H steps, $\mu _ { d } ^ { ( s ) } = $ median $\{ m _ { d } ^ { ( s - H ) } , \dots , m _ { d } ^ { ( s - 1 ) } \}$ (fewer in warm-up):

$$
\tilde { m } _ { d } ^ { ( s ) } = \mathrm { c l i p } ( m _ { d } ^ { ( s ) } , e ^ { - c } \mu _ { d } ^ { ( s ) } , e ^ { c } \mu _ { d } ^ { ( s ) } ) ,\tag{9}
$$

where $\mathrm { c l i p } ( x , a , b )$ restricts x to $[ a , b ]$ , and c is the clipping threshold. Normal fluctuations pass unchanged, and a spike keeps its direction while its size is bounded. This is the Huber ψ-function (Huber, 1964) in the log domain, and all other statistics of RDS-MOPD are computed on the clipped sequence. Since the median uses the raw observations, a single anomaly cannot drag the center.

Proposition 3.2 (Bounded Influence and Trend Preservation). Let $H \geq 3$ and $c > 0 .$ . (i) Every clipped observation lies in $[ e ^ { - c } \mu _ { d } ^ { ( s ) } , e ^ { c } \mu _ { d } ^ { ( s ) } ] f o r s > 1$ . (ii) Replacing one raw observation $m _ { d } ^ { ( s _ { 0 } ) }$ with $s _ { 0 } > H$ by any $M > 0$ changes the clipped sequence only at steps $s _ { 0 } , \ldots , s _ { 0 } + H$ , and only within a range independent of M. Hence any EMA level ofthe clipped sequence stays in a range independent ofM, whereas without clipping it grows linearly in M. (iii) If the observations are noise-free and jump at step s<sub>0</sub> from one constant level to another, the clipped sequence equals the raw onefrom step $s _ { 0 } + \lfloor H / 2 \rfloor + 1 o n$

Hence, a single spike, however large, only has a bounded and short-lived effect, keeping p in a bounded region (Figure 3(b)), while a persistent real change still passes through after a short delay (proofs in Appendix H.2). A sustained burst is thus treated as a real change, whose effect is bounded by WDR (Theorem 3.3(ii)) and fades once it ends.

## 3.2.3. WORST-CASE DOMAIN REWEIGHTING (WDR)

WDR fixes misplaced protection: the worst case should be a regressing domain, not an inherently hard one. The nominal allocation alone does not protect such a domain either: a rising KL clips its $v _ { d }$ to 0, giving it the smallest share. WDR thus redefines the worst case and solves the DRO problem around p.

Excess Loss. We measure whether a domain is regressing by its excess loss over its own best level:

$$
h _ { d } = \operatorname* { m i n } \Big ( \log \frac { \ell _ { d } } { \ell _ { d } ^ { \star } } , \log \kappa \Big ) \in [ 0 , \log \kappa ] ,\tag{10}
$$

where $\ell _ { d } ^ { \star }$ is the lowest EMA level of domain d so far (current update included) and $\kappa \geq 1$ is the excess cap. Here, $h _ { d }$ is the log regression factor of Eqn. (5) on the EMA, capped to bound the tilt in Eqn. (11). It is 0 at the best level and grows as d regresses; as a ratio within one domain, it depends on neither the KL scale nor $I _ { d }$

Robust Best Response in a KL Ball. Given p, WDR solves Eqn. (6) with the feasible set $\mathcal { U } = \{ \mathbf { q } \in \Delta ^ { K } : q _ { d } \geq \lambda _ { d } \}$ where $\lambda _ { d } = \operatorname* { m i n } ( p _ { d } , \lambda _ { \mathrm { l o } } )$ and $\lambda _ { \mathrm { l o } }$ is the coverage floor: the tilt never pushes a domain below min $\left( p _ { d } , \lambda _ { \mathrm { l o } } \right)$ , and p itself stays feasible. Without the floor, the solution is the closed form of §3.1.1 with $\hat { \ell } _ { d }$ replaced by $h _ { d } \left( \propto p _ { d } e ^ { \eta h _ { d } } \right)$ ; the floor lifts any domain below $\lambda _ { d }$ to $\lambda _ { d } \mathbf { : }$

$$
q _ { d } = \operatorname* { m a x } { \left( \lambda _ { d } , \nu p _ { d } e ^ { \eta h _ { d } } \right) } ,\tag{11}
$$

where $\nu > 0$ is the unique normalizer with $\textstyle \sum _ { d } q _ { d } = 1$ (computed exactly in Appendix D). Each share is thus multiplied by $e ^ { \eta h _ { d } } \in [ 1 , \kappa ^ { \eta } ]$ : the tilt is idle when all domains are at their best, and a more regressing domain gets more budget, up to $\kappa ^ { \eta }$ times.

Theorem 3.3 (Exact KL-Ball DRO Solution). Let $\mathbf { p } \in \Delta ^ { K }$ with $\mathbf { p } > \mathbf { 0 } , \mathbf { h } \in [ 0 , \log \kappa ] ^ { K } , \eta > 0 , \kappa \geq 1 ,$ , and $\textstyle \sum _ { d } \lambda _ { d } < 1$ (i) Exact KL-ball DRO solution. Eqn. (6) has a unique maximizer, given by Eqn. (11), which also maximizes ⟨q<sup>′</sup>, h⟩ over the KL ball $\{ \mathbf { q } ^ { \prime } \in \mathcal { U } : \mathrm { K L } ( \mathbf { q } ^ { \prime } \| \mathbf { p } ) \leq \mathrm { K L } ( \mathbf { q } \| \mathbf { p } ) \}$ around p. (ii) Bounded deviation without starvation. $\kappa ^ { - \eta } \leq q _ { d } / p _ { d } \leq \kappa ^ { \eta }$ for all d, and $q _ { d } \ge \operatorname* { m i n } ( f , \lambda _ { \mathrm { l o } } )$ for p of Eqn. (7) with $K f \le 1$ . (iii) Targeted protection. If all $h _ { d }$ are equal, $\mathbf { q } = \mathbf { p } ;$ otherwise, every domain with the largest $h _ { d }$ has $q _ { d } \geq p _ { d }$

Intuitively, WDR is an exactly solved DRO decision (proofs in Appendix H.3): within a KL ball around p, q attains the largest expected excess loss, where the highest iso-line of $\langle \mathbf { q } , \mathbf { h } \rangle$ touches the ball (Figure 3(c)).

## 3.3. Overall Procedure

SAE clips each new KL observation at every training step (Eqn. (9)); at every scheduling update, RDS-MOPD computes r, σˆ, and h on the clipped sequences, estimates p (Eqns. (7) and (8)), and tilts it into q (Eqn. (11); Algorithm 1 in Appendix D). Setting $\epsilon = 0 , c = \infty , \mathrm { o r } \eta = 0$ (then ${ \textbf { q } } = { \textbf { p } }$ disables NRC, SAE, or WDR, respectively (§4.3.1). An update only reads the K KL sequences already computed by the trainer (Eqn. (1)) and costs $O ( K ^ { 2 } )$ , with no extra forward pass or GPU memory (§4.5).

## 4. Experiments

We evaluate RDS-MOPD by answering the following questions:

• RQ1: How does RDS-MOPD perform on math, code, and instruction following, compared with MOPD methods with a fixed or adaptive domain mixture and training-free model merging?

• RQ2: How does each module of RDS-MOPD contribute to the performance, and does it resolve its corresponding instability?

• RQ3: Can RDS-MOPD keep a robust mixture and protect the other domains when one domain teacher is corrupted in the middle of training?

• RQ4: Does RDS-MOPD incur a large time overhead?

## 4.1. Experimental Setups

## 4.1.1. BENCHMARKS AND EVALUATION METRICS

We evaluate all methods on six benchmarks, two per domain teacher: AIME24 and AIME25 (Mathematical Association of America, 2025) for mathematical reasoning, LiveCodeBench v5 and v6 (LCB v5/v6) (Jain et al., 2025) for code generation, and IFEval (Zhou et al., 2023) and IFBench (Pyatkin et al., 2025) for instruction following (IF). Each domain pairs an earlier benchmark with a more recent one, so that both in-domain fitting and generalization to newer distributions are examined. We report avg@64 on AIME24/25, avg@10 on LCB v5/v6, and avg@4 of the strict prompt-level accuracy on IFEval/IFBench (higher is better for all). Benchmark statistics and evaluation configurations are given in Appendix B.

## 4.1.2. TRAINING DATA AND MODELS

We build our testbed upon Open-MOPD (Gao et al., 2026), to our knowledge the only recent MOPD work that releases its complete training resources. We distill the SmolLM3- 3B (Hugging Face, 2025) student, initialized from the released MixSFT checkpoint (Student-SFT), from the released RL-Math, RL-Code, and RL-IF teachers. The prompt pool contains 86,931 prompts, i.e., 17,917 math prompts from DAPO (Yu et al., 2025), 23,667 code prompts from PrimeIntellect (Prime Intellect, 2025) and TACO (Li et al., 2023), and 45,347 IF prompts from Nemotron-IF-RL (Gao et al., 2026), where each prompt is routed to the teacher of its domain. Per-domain statistics are given in Appendix B.

## 4.1.3. COMPETING METHODS

We compare RDS-MOPD with two groups of competing methods: (1) state-of-the-art MOPD methods with a fixed or adaptive domain mixture, i.e., Vanilla MOPD (Ma et al., 2026), Open-MOPD (Gao et al., 2026), D<sup>3</sup>-MOPD (Sun et al., 2026b), and Vanilla DRO-MOPD (Sagawa et al., 2020) (§3.1); and (2) training-free model merging, i.e.,

Table 1. Overall performance (%, mean ± std over 3 seeds; training-free methods and reference rows have no std) under the BO (top) and BB (bottom) checkpoint selection criteria. Bold/underline: best/second-best among the ranked methods (gray reference rows excluded); ${ } ^ { 6 6 } \mathrm { A v g } .$ Rank” (↓): mean per-benchmark rank over the six benchmarks; “Recovery” (↑): (Total − SFT)/(RouteRL − SFT) following Open-MOPD, where Total is the mean of the three domain averages; p-value: one-sided Wilcoxon test of RDS-MOPD vs. each method over 18 benchmark-seed pairs, treating pairs as independent (Appendix E).
<table><tr><td rowspan="2">Method</td><td colspan="2">Math</td><td colspan="2">Code LCB v6</td><td colspan="2">Instruction Following</td><td rowspan="2"> $\overline { { \mathrm { A v g . } } }$ </td><td rowspan="2">Recovery (%, ↑)</td><td rowspan="2">p-value</td></tr><tr><td></td><td>AIME24 AIME25</td><td>LCB v5</td><td>IFEval</td><td>IFBench</td><td> $\mathbf { R a n k } \left( \downarrow \right)$ </td></tr><tr><td>Student-SFT</td><td>15.10 24.38</td><td>15.16</td><td>15.99</td><td>15.49</td><td>66.82</td><td>18.50</td><td>一</td><td>0.0</td><td></td></tr><tr><td>RouteRL</td><td>24.32</td><td>22.99</td><td></td><td>21.94</td><td>73.57</td><td>22.50</td><td></td><td>100.0</td><td></td></tr><tr><td colspan="10">Best-Overall (BO)</td></tr><tr><td>ParamMerge-Avg</td><td>21.20</td><td>21.67</td><td>20.00</td><td>19.43</td><td>70.52</td><td>20.00</td><td>7.00</td><td>60.4</td><td> $\overline { { { 3 . 8 \times 1 0 } ^ { - 6 } } }$ </td></tr><tr><td>ParamMerge-TA</td><td>23.33</td><td>24.11</td><td>22.46</td><td>21.26</td><td>73.20</td><td>22.83</td><td>5.00</td><td>94.1</td><td> $7 . 6 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>Vanilla MOPD</td><td> $\overline { { 2 3 . 2 5 \pm 0 . 2 9 } }$ </td><td> $2 3 . 3 0 \pm 0 . 2 4$ </td><td> $\overline { { 2 1 . 4 2 \pm 0 . 1 2 } }$ </td><td> $\overline { { 2 1 . 3 0 \pm 0 . 2 4 } }$ </td><td> $\overline { { 7 2 . 4 1 \pm 0 . 2 9 } }$ </td><td> $\overline { { 2 1 . 0 3 \pm 0 . 1 3 } }$ </td><td>5.83</td><td>83.6</td><td> $\overline { { { 3 . 8 \times 1 0 ^ { - 6 } } } }$ </td></tr><tr><td>Open-MOPD</td><td> $2 4 . 6 4 \pm 0 . 1 4$ </td><td> $2 4 . 7 0 \pm 0 . 2 4$ </td><td> $2 2 . 4 6 \pm 0 . 0 6$ </td><td> $2 2 . 3 1 \pm 0 . 1 3$ </td><td> $7 4 . 2 6 \pm 0 . 1 7$ </td><td> ${ \bf 2 3 . 7 2 \pm 0 . 0 9 }$ </td><td>2.67</td><td>105.6</td><td> $4 . 2 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>D³-MOPD</td><td> $2 4 . 7 7 \pm 0 . 8 1$ </td><td> $2 4 . 9 0 \pm 0 . 3 7$ </td><td> $2 2 . 9 1 \pm 0 . 7 4$ </td><td> $\overline { { 2 2 . 2 9 \pm 0 . 4 6 } }$ </td><td> $7 4 . 1 8 \pm 0 . 2 6$ </td><td> $2 3 . 4 2 \pm 0 . 8 0$ </td><td>2.58</td><td>106.5</td><td> $9 . 7 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Vanilla DRO-MOPD</td><td> $2 3 . 8 5 \pm 0 . 2 6$ </td><td> $2 4 . 6 9 \pm 0 . 2 7$ </td><td> $2 3 . 3 7 \pm 0 . 2 3$ </td><td> $2 2 . 0 7 \pm 0 . 1 2$ </td><td> $7 3 . 9 7 \pm 0 . 0 7$ </td><td> $2 3 . 4 2 \pm 0 . 1 4$ </td><td>3.42</td><td>103.9</td><td> $1 . 6 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>RDS-MOPD</td><td> $\mathbf { \delta } ^ { 2 6 . 7 2 \pm 0 . 1 6 }$ </td><td> $\mathbf { \overline { { { 2 6 . 0 9 \pm 0 . 1 9 } } } }$ </td><td> $\overline { { 2 3 . 7 9 \pm 0 . 1 9 } }$ </td><td> $\mathbf { \overline { { { 2 3 . 0 1 \pm 0 . 0 7 } } } }$ </td><td> $\overline { { 7 4 . 7 4 \pm 0 . 1 0 } }$ </td><td> $\overline { { 2 3 . 0 8 \pm 0 . 3 0 } }$ </td><td>1.50</td><td>118.1</td><td>1</td></tr><tr><td colspan="10">Best-per-Benchmark (BB)</td></tr><tr><td>ParamMerge-Avg</td><td>21.20</td><td>21.67</td><td>20.00</td><td>19.43</td><td>70.52</td><td>20.00</td><td>7.00</td><td>60.4</td><td> $3 . 8 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>ParamMerge-TA</td><td>23.33</td><td>24.11</td><td>22.46</td><td>21.26</td><td>73.20</td><td>22.83</td><td>5.33</td><td>94.1</td><td> $3 . 8 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>Vanilla MOPD</td><td> $\overline { { 2 3 . 3 4 \pm 0 . 1 4 } }$ </td><td> $\overline { { 2 3 . 6 5 \pm 0 . 0 9 } }$ </td><td> $\overline { { 2 1 . 7 4 \pm 0 . 0 6 } }$ </td><td> $2 1 . 3 5 \pm 0 . 1 5$ </td><td> $\overline { { 7 2 . 6 3 \pm 0 . 2 1 } }$ </td><td> $\overline { { 2 1 . 2 5 \pm 0 . 1 4 } }$ </td><td>5.67</td><td>86.5</td><td> $\overline { { { 3 . 8 \times 1 0 ^ { - 6 } } } }$ </td></tr><tr><td>Open-MOPD</td><td> $2 4 . 8 1 \pm 0 . 0 3$ </td><td> $2 4 . 9 0 \pm 0 . 0 6$ </td><td> $2 2 . 5 9 \pm 0 . 0 7$ </td><td> $2 2 . 3 8 \pm 0 . 0 7$ </td><td> ${ \underline { { 7 4 . 5 5 \pm 0 . 1 4 } } }$ </td><td> ${ \bf 2 3 . 8 1 \pm 0 . 1 7 }$ </td><td>2.58</td><td>107.8</td><td> $9 . 7 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>D³-MOPD</td><td> $2 4 . 9 5 \pm 0 . 5 6$ </td><td> $2 5 . 1 1 \pm 0 . 5 5$ </td><td> $2 2 . 9 1 \pm 0 . 7 4$ </td><td> $\underline { { 2 2 . 3 8 \pm 0 . 3 2 } }$ </td><td> $\overline { { 7 4 . 5 4 \pm 0 . 4 7 } }$ </td><td> $2 3 . 5 0 \pm 0 . 8 2$ </td><td>2.58</td><td>108.7</td><td> $9 . 7 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Vanilla DRO-MOPD</td><td> $2 4 . 1 7 \pm 0 . 0 6$ </td><td> $2 4 . 8 6 \pm 0 . 0 6$ </td><td> $2 3 . 5 1 \pm 0 . 0 3$ </td><td> $2 2 . 1 9 \pm 0 . 0 9$ </td><td> $7 4 . 0 3 \pm 0 . 0 0$ </td><td> $2 3 . 5 6 \pm 0 . 1 0$ </td><td>3.33</td><td>106.1</td><td> $1 . 6 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>RDS-MOPD</td><td> $\mathbf { \delta } ^ { 2 6 . 7 2 \pm 0 . 1 6 }$  </td><td> $\mathbf { \widetilde { 2 6 . 0 9 \pm 0 . 1 9 } }$ </td><td> $\overline { { 2 3 . 9 5 \pm 0 . 2 2 } }$ </td><td> $\mathbf { 2 3 . 2 0 \pm 0 . 1 1 }$ </td><td> $\overline { { 7 4 . 8 8 \pm 0 . 0 3 } }$  一</td><td> $\overline { { 2 3 . 1 7 \pm 0 . 1 7 } }$ </td><td>1.50</td><td>119.5</td><td>一</td></tr></table>

## 4.1.4. IMPLEMENTATION AND EVALUATION

ParamMerge-Avg (Wortsman et al., 2022) and ParamMerge-TA (Ilharco et al., 2023). We also report RouteRL (Gao et al., 2026) as a routed-teacher reference, which is not a single deployable model and is excluded from the ranking. Details are given in Appendix C.

All MOPD methods share the same training code and sampler, and differ mainly in the mixture rule. They use the dense top-16 teacher reverse KL as the per-token reward, and are trained for 150 steps with 1,024 prompts per step; the scheduling methods start from q<sup>(0)</sup> = (0.4, 0.4, 0.2) and update it every 10 steps, except Open-MOPD and Vanilla DRO-MOPD, which keep this sampling mixture and reweight the per-domain rewards and losses, respectively. Following $\mathrm { D ^ { 3 } - M O P D }$ , we report the mean and standard deviation over three seeds under two checkpoint selection criteria. Best-Overall Performance (BO) takes the single checkpoint with the best mean domain score of each run, and Best-per-Benchmark Performance (BB) takes the best checkpoint on each benchmark. As a scale-free summary, we also report the average per-benchmark rank (Avg. Rank; details in Appendix E).

## 4.2. Overall Performance (RQ1)

As shown in Table 1, we have the following threefold observations.

(1) Training-free model merging can hardly fuse the capabilities of multiple teachers. ParamMerge-Avg ranks last on all six benchmarks, and ParamMerge-TA still falls behind all the adaptive MOPD methods (Avg. Rank of

5.00 and 5.33 under BO and BB). This is reasonable as the domain-specific RL updates of the teachers conflict and partly cancel out when merged in the parameter space, whereas MOPD aligns the student with each teacher token by token on its own responses. Yet MOPD alone does not guarantee this advantage: Vanilla MOPD with a fixed mixture even ranks below ParamMerge-TA (5.83 vs. 5.00 under BO), whereas the adaptive methods (Open-MOPD, D<sup>3</sup>-MOPD, and RDS-MOPD) outperform Vanilla MOPD on all six benchmarks under both criteria. This indicates that the key lies in allocating the domain budget.

(2) Vanilla DRO-MOPD is unstable under noisy training signals. Although it up-weights the worst domain, Vanilla DRO-MOPD falls behind the progress-driven Open-MOPD and D<sup>3</sup>-MOPD on four of six benchmarks (AIME24, AIME25, LCB v6, and IFEval), with an Avg. Rank of 3.42 and 3.33 under BO and BB. This is mainly because, besides the three instabilities of §3.1, reweighting the losses under a fixed sampling mixture only rescales the gradients and amplifies the KL noise of the up-weighted domain. This indicates that robust scheduling should protect the domain regressing from its own best level rather than the one with the largest loss.

(3) Our RDS-MOPD consistently outperforms the existing methods. RDS-MOPD ranks first on five of six benchmarks with the lowest Avg. Rank of 1.50 under both BO and BB, clearly ahead of D<sup>3</sup>-MOPD and Open-MOPD (2.58– 2.67). The gains are largest on math, i.e., 1.95 and 1.19 points over the best baseline on AIME24 and AIME25 under BO, and also hold on code (e.g., 0.70 points on LCB v6) and IFEval. The only exception is IFBench, whose 58 constraint types are designed to be out-of-distribution (Pyatkin et al., 2025) and are all absent from our IF training prompts, which may limit the adaptability of the distilled student to them. Even so, all adaptive MOPD methods, including RDS-MOPD, exceed the IF teacher (22.50) on IFBench, whereas on the in-distribution IFEval, RDS-MOPD achieves the best result. RDS-MOPD also surpasses the domain teachers (RouteRL) on all six benchmarks, with all seed standard deviations within 0.30, whereas those of D<sup>3</sup>-MOPD reach 0.81 and 0.74 on AIME24 and LCB v5 under BO. This is mainly because the existing progress-driven methods act on progress scores estimated from noisy KL curves (e.g., D<sup>3</sup>-MOPD normalizes its softmax by the maximum score). Hence, noise-induced leader switches (Appendix F) and transient KL spikes can pull a large share of prompts toward one domain. The three modules of RDS-MOPD bound exactly these effects. This indicates that the bottleneck of adaptive scheduling is robustness to noisy signals rather than adaptivity itself. RDS-MOPD outperforms every competing method on at least 15 of the 18 benchmark-seed pairs.

Table 2. Ablation study of RDS-MOPD (%), with the relative change compared to RDS-MOPD in parentheses (variants are single runs with seed 0).
<table><tr><td>Benchmark</td><td>RDS-MOPD</td><td>w/oNRC</td><td>w/o SAE</td><td>w/oWDR</td></tr><tr><td colspan="5">Best-Overall (BO)</td></tr><tr><td>AIME24</td><td>26.72</td><td>25.47 (↓4.68%)</td><td>26.04 (↓2.54%)</td><td>26.20 (↓1.95%)</td></tr><tr><td>AIME25</td><td>26.09</td><td>25.62 (↓1.80%)</td><td>25.36 (↓2.80%)</td><td>25.89 (↓0.77%)</td></tr><tr><td>LCB v5</td><td>23.79</td><td>22.93 (↓3.61%)</td><td>23.47 (↓1.35%)</td><td>23.65 (↓0.59%)</td></tr><tr><td>LCB v6</td><td>23.01</td><td>22.51 (↓2.17%)</td><td>23.37 (↑1.56%)</td><td>22.86 (↓0.65%)</td></tr><tr><td>IFEval</td><td>74.74</td><td>74.58 (↓0.21%)</td><td>74.82 (↑0.11%)</td><td>75.74 (↑1.34%)</td></tr><tr><td>IFBench</td><td>23.08</td><td>22.75 (↓1.43%)</td><td>22.50 (↓2.51%)</td><td>23.00 (↓0.35%)</td></tr><tr><td colspan="5">Best-per-Benchmark (BB)</td></tr><tr><td>AIME24</td><td>26.72</td><td> $\overline { { 2 6 . 3 0 \left( \downarrow 1 . 5 7 \% \right) } }$ </td><td>26.04 (↓2.54%)</td><td>26.20 (↓1.95%)</td></tr><tr><td>AIME25</td><td>26.09</td><td>25.62 (↓1.80%)</td><td>25.36 (↓2.80%)</td><td>25.89 (↓0.77%)</td></tr><tr><td>LCB v5</td><td>23.95</td><td>23.59 (↓1.50%)</td><td>23.47 (↓2.00%)</td><td>23.65 (↓1.25%)</td></tr><tr><td>LCB v6</td><td>23.20</td><td> $2 3 . 2 0 ~ \mathrm { ( 0 . 0 0 \% ) }$ </td><td>23.37 (↑0.73%)</td><td> $2 2 . 8 6 ( \scriptstyle \downarrow 1 . 4 7 \% )$ </td></tr><tr><td>IFEval</td><td>74.88</td><td> $7 4 . 5 8 ~ ( \downarrow 0 . 4 0 \% )$ </td><td>74.82 (↓0.08%)</td><td> $7 5 . 7 4 \ ( \uparrow 1 . 1 5 \% )$ </td></tr><tr><td>IFBench</td><td>23.17</td><td> $2 2 . 7 5 \ ( \downarrow 1 . 8 1 \% )$ </td><td> $2 3 . 4 2 \ ( \uparrow 1 . 0 8 \% )$ </td><td> $2 3 . 0 8 ~ ( \downarrow 0 . 3 9 \% )$ </td></tr></table>

## 4.3. Empirical Module Analyses (RQ2)

## 4.3.1. ABLATION STUDY

To verify each module, we build three variants of RDS-MOPD, each disabling exactly one module by its hyperparameter (§3.3) and keeping all other hyper-parameters unchanged. As shown in Table 2, disabling any module degrades RDS-MOPD on most benchmarks.

• w/o NRC drops on all six benchmarks under BO, most notably by 4.68% and 3.61% on AIME24 and LCB v5, as the scheduler without the noise floor pushes the budget toward a domain that leads by chance, i.e., the worst-case misjudgment. This implies the criticalness of deciding when to respond under noisy signals.

• w/o SAE drops on four of six benchmarks under both criteria, most notably by 2.54% and 2.80% on AIME24 and AIME25. Without clipping, transient KL spikes of the code domain enter the statistics of the nominal allocation in proportion to their size, distorting the allocation and squeezing the other domains. This implies the criticalness of bounding the influence of any single anomaly.

![](images/e25cf4d0ddc9d89f90a1dac32f81b5ef15dc4b3fa5055041fdfb4cc10f4ffa43.jpg)

(a) Worst-case misjudgment  
![](images/99ff73a0fdbb83009b6fd07852ddb6454bee5c41ce20115fa08287aa07cd1c3b.jpg)

![](images/b9db6eb3a7037bf5ecdde12ecdb6b630c41cfb76ea164544d7342fa5abd4e374.jpg)  
(b) Spike hijacking  
(c) Misplaced protection  
Figure 4. RDS-MOPD vs. its variants without the corresponding module (“w/o module”), replayed on the same recorded trajectory. (a) Share of each domain vs. its progress gap in noise standard deviations (shaded: within one std); (b) share shift remaining 40 steps after a KL spike (lower is better); (c) share of the most regressing domain (higher is better).

• w/o WDR drops on all benchmarks except IFEval under both criteria, e.g., by 1.95% on AIME24 and 1.47% on LCB v6 under BB, as the nominal allocation gives no extra protection to a domain regressing from its best level. This implies the criticalness of each domain’s own best level as the reference.

## 4.3.2. MODULE STABILITY ANALYSES

To understand why each module works, we examine whether it resolves its instability of Vanilla DRO-MOPD (§3.1). We replay RDS-MOPD and its variants without each module on the same recorded trajectory, so that any difference in decisions is attributable to this module alone. As shown in Figure 4, we have the following threefold observations.

(1) NRC resolves the worst-case misjudgment caused by noise. As shown in Figure 4(a), without NRC, the shares swing between 0.15 and 0.65 even when the gap between domains is below one noise standard deviation, i.e., the worst-case misjudgment. With NRC, they stay between 0.17 and 0.40 within the noise range, yet still rise to 0.57 on average once the gap exceeds two noise standard deviations. As a result, the share follows the gap much more closely (Spearman correlation of 0.83 vs. 0.55). This indicates that RDS-MOPD responds only to real learning gaps, while still adjusting the mixture decisively.

(2) SAE prevents spikes from hijacking the training budget. As shown in Figure 4(b), without SAE, the share shift caused by an injected one-step KL spike grows with the spike size and lasts long after the spike disappears, i.e., the spike hijacking. The shift is still 0.027 after 40 steps under a 64× spike. With SAE, it stays at 0.002 regardless of the spike size, and its maximum right after the spike is bounded at 0.033, compared with 0.172 without SAE. This indicates that the influence of any single anomaly on RDS-MOPD is both bounded and short-lived.

![](images/26a9af1bfc72b4a0ff0a4919e9ed169973dd10291295137d6c4d54dcc5bded55.jpg)  
Figure 5. RDS-MOPD vs. $\mathrm { D } ^ { 3 } { \cdot } \mathrm { M O P D }$ with the code teacher corrupted from step 30 (shaded). (a) Share of code prompts (dashed: RDS-MOPD without corruption); (b) math KL relative to its level at step 30 (lower is better).

(3) WDR protects the truly regressing domains. As shown in Figure 4(c), without WDR, the domain regressing the most from its own best level receives only 0.277 of the budget on average, even less than the 0.398 of the domains close to their best levels, i.e., the misplaced protection. With WDR, it receives more budget at 12 of the 13 updates from step 30, and its share rises to 0.382 on average. This indicates that RDS-MOPD protects the truly regressing domains.

## 4.4. Robustness under a Corrupted Teacher (RQ3)

In practice, the signal of a domain teacher may become unreliable during training, e.g., due to a faulty scoring service. To test whether RDS-MOPD stays reliable in such cases, we corrupt the code teacher from step 30 on. Specifically, we add zero-mean noise (re-sampled at every call) to its token log-probabilities within the student’s top-16 candidates, which inflates its KL signal by about 2.5×. We then compare RDS-MOPD with D<sup>3</sup>-MOPD under the same seed and data stream for 60 steps, which cover the first three scheduling updates after the corruption. As shown in Figure 5, we have the following twofold observations.

(1) RDS-MOPD keeps a stable mixture under the corrupted teacher. As shown in Figure 5(a), the mixture of RDS-MOPD is almost unaffected by the corruption. Its code share deviates from that of its uncorrupted run by only 0.011 on average over steps 30–60, and averages 0.325 over steps 41–60. In contrast, D<sup>3</sup>-MOPD allocates only 0.179 to code and moves the released budget almost entirely to math (+0.145 vs. −0.006 for RDS-MOPD). This is mainly because the inflated KL raises the excess loss of code, so WDR keeps a bounded protection on it. Meanwhile, NRC treats the remaining score differences within the noise level as no evidence for re-allocation, so RDS-MOPD is not dragged by an unreliable teacher.

![](images/c4daa4889fb1a6a31e290251e063bf80462ab0517c6cd6b83df3e5b81f34feb0.jpg)  
Figure 6. Wall-clock time per training step of MOPD methods.

(2) RDS-MOPD preserves the learning progress of all domains. RDS-MOPD allocates about 1.8× as many prompts to the corrupted code domain as D<sup>3</sup>-MOPD, yet ends with a lower code KL (0.0164 vs. 0.0225 over steps 56–60). Meanwhile, as shown in Figure 5(b), its math KL decreases to 0.66× its level at step 30, compared with 0.79× for D<sup>3</sup>- MOPD. Thus, RDS-MOPD neither abandons the corrupted domain nor sacrifices clean ones.

## 4.5. Efficiency Analysis (RQ4)

We measure the per-step wall-clock time of the MOPD methods on the same hardware (16 A800 GPUs; stages in Figure 6) and the time of the scheduling decision, with the following twofold observations.

(1) The scheduling decision of RDS-MOPD introduces negligible overhead. The RDS-MOPD scheduler takes only 0.039 milliseconds per step on average, more than seven orders of magnitude below a training step of about 750 seconds. This is mainly because all three modules operate on a few per-domain scalars, with no extra forward pass or GPU memory. Thus, the robustness of RDS-MOPD comes from its decision rule, not extra computation.

(2) The end-to-end training cost of RDS-MOPD is on par with or even lower than that of existing methods. Over steps 2–84, RDS-MOPD takes 747.5 seconds per step on average, vs. 758.2, 764.5, and 762.6 seconds for Open-MOPD, D<sup>3</sup>-MOPD, and Vanilla DRO-MOPD. Over the full 150 steps, it is 5.6% faster per step than D<sup>3</sup>-MOPD, as rollout takes about 45% of each step and the student of RDS-MOPD produces shorter responses (7,130 vs. 8,127 tokens on average). Thus, RDS-MOPD distills better at no extra cost.

## 5. Related Work

## 5.1. Multi-Teacher and On-Policy Distillation

Knowledge distillation fits a student to the teacher’s output distribution (Hinton et al., 2015), and early multi-teacher methods fuse the soft labels or intermediate representations of multiple teachers (You et al., 2017), or select the teachers for each sample by reinforcement learning (Yuan et al., 2021). For LLMs, sequence-level distillation (Kim & Rush, 2016) suffers from the distribution shift between training and inference, which on-policy distillation (OPD) alleviates by supervising the student on its own sampled responses (Agarwal et al., 2024; Gu et al., 2024), e.g., with the reverse KL divergence to avoid overestimating the lowprobability regions of the teacher (Gu et al., 2024), and has become a standard post-training stage (Lu & Thinking Machines Lab, 2025; Yang et al., 2025), as its per-token supervision is denser than the sequence-level rewards of RL. Extending it to multiple domain-expert teachers leads to MOPD (Ma et al., 2026), which merges the capabilities of domain-specific RL teachers into one student. Subsequent work improves MOPD by balancing the capabilities of different teachers via token-share balancing and KL-gap following (Gao et al., 2026), weighting or gating teacher supervision by its reliability (Sun et al., 2026a; Zhang et al., 2026), resolving the conflicts among teachers (Tian et al., 2026; Chen et al., 2026), and distilling at the representation level (Fang et al., 2026). Most of them keep the domain mixture fixed and refine how the teacher supervision is weighted or combined. Closer to our work, D<sup>3</sup>-MOPD (Sun et al., 2026b) schedules the domain mixture by the learning progress of each domain, i.e., its remaining KL gap times its descent velocity, MT-GRPO (Ramesh et al., 2026) reweights tasks online toward the worst-performing task in multi-task RL, and concurrent work (Liu et al., 2026) sets model-merging coefficients before training by mini mizing the worst per-domain residual. Different from these works, RDS-MOPD introduces DRO into the online domainmixture scheduling of MOPD and makes it robust to noise.

## 5.2. Distributionally Robust Optimization

DRO optimizes the worst-case loss within an uncertainty set around the nominal distribution (Ben-Tal et al., 2013; Duchi & Namkoong, 2021), where the uncertainty set can be defined by, e.g., the KL divergence, whose worst case admits a closed-form exponential tilting (Lam, 2016), the Wasserstein distance (Esfahani & Kuhn, 2018), or a CVaR-type tail risk (Rockafellar & Uryasev, 2000). Group DRO (Sagawa et al., 2020) instantiates this idea over predefined groups by up-weighting high-loss groups online, and has been widely adopted, e.g., in topic-robust language modeling (Oren et al., 2019), data mixture optimization for pre-training (Xie et al., 2023), group-robust preference alignment (Ramesh et al.,

2024), and worst-group knowledge distillation (Kenfack et al., 2025; Wang et al., 2023; Vilouras et al., 2023). Among them, DoReMi (Xie et al., 2023) runs Group DRO on a small proxy model to determine the pre-training domain weights offline. However, DRO is known to be sensitive to outliers, since the worst case tends to be dominated by them, and DORO (Zhai et al., 2021) thus discards a fraction of the highest-loss samples. This motivates combining DRO with robust statistics against heavy-tailed noise, e.g., the Huber estimator (Huber, 1964) and median-of-means (Lugosi & Mendelson, 2019). Following this line, RDS-MOPD fixes three concrete instabilities of DRO under the noisy signals of MOPD within a single DRO structure.

## 6. Conclusion and Future Work

In this paper, we propose RDS-MOPD, a robust domainmixture scheduling framework for multi-teacher on-policy distillation. It resolves the instabilities of Vanilla DRO-MOPD under noisy signals with three modules, backed by theoretical guarantees. Extensive experiments on mathematical reasoning, code generation, and instruction-following benchmarks demonstrate the effectiveness and robustness of RDS-MOPD with negligible computational overhead. As for future work, we plan to extend the robust allocation from the domain level to finer granularities, e.g., the prompt level.

## References

Agarwal, R., Vieillard, N., Zhou, Y., et al. On-policy distillation of language models: Learning from self-generated mistakes. In ICLR, 2024.

Ben-Tal, A., den Hertog, D., Waegenaere, A. D., Melenberg, B., and Rennen, G. Robust solutions of optimization problems affected by uncertain probabilities. Management Science, 2013.

Chen, T., Ou, J., Liu, Z., Tang, R., Liang, J., and Li, H. Counteraction-aware multi-teacher on-policy distillation for general capability recovery with domain preservation. arXiv preprint arXiv:2605.27115, 2026.

Duchi, J. C. and Namkoong, H. Learning models with uniform performance via distributionally robust optimization. Ann. Stat., 2021.

Esfahani, P. M. and Kuhn, D. Data-driven distributionally robust optimization using the Wasserstein metric: performance guarantees and tractable reformulations. Math. Program., 2018.

Fang, Z., Hong, S., Yang, J., et al. Latent-MOPD: Latent multi-teacher on-policy distillation. arXiv preprint arXiv:2610.02381, 2026.

Gao, H., Chi, H., Yan, Y., et al. Open-MOPD: Diagnosing and fixing capability imbalance in multi-teacher on-policy distillation. arXiv preprint arXiv:2608.19098, 2026.

Gu, Y., Dong, L., Wei, F., and Huang, M. MiniLLM: Knowledge distillation of large language models. In ICLR, 2024.

Hinton, G., Vinyals, O., and Dean, J. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.

Huber, P. J. Robust estimation of a location parameter. Ann. Math. Stat., 1964.

Hugging Face. SmolLM3: Smol, multilingual, long-context reasoner. Hugging Face blog, 2025.

Ilharco, G., Ribeiro, M. T., Wortsman, M., Schmidt, L., Hajishirzi, H., and Farhadi, A. Editing models with task arithmetic. In ICLR, 2023.

Jain, N., Han, K., Gu, A., et al. LiveCodeBench: Holistic and contamination free evaluation of large language models for code. In ICLR, 2025.

Kenfack, P., A¨ıvodji, U., and Kahou, S. E. Adaptive group robust ensemble knowledge distillation. TMLR, 2025.

Kim, Y. and Rush, A. M. Sequence-level knowledge distillation. In EMNLP, 2016.

Lam, H. Robust sensitivity analysis for stochastic systems. Math. Oper. Res., 2016.

Li, R., Fu, J., Zhang, B., et al. TACO: Topics in algorithmic code generation dataset. arXiv preprint arXiv:2312.14852, 2023.

Liu, K., Zhuang, Z., Ma, H., et al. A fisher-geometric view of multi-teacher on-policy distillation. arXiv preprint arXiv:2610.26525, 2026.

Lu, K. and Thinking Machines Lab. On-policy distillation. Thinking Machines Lab: Connectionism (blog post), 2025.

Lugosi, G. and Mendelson, S. Mean estimation and regression under heavy-tailed distributions: A survey. Found. Comput. Math., 2019.

Ma, W., Wei, J., Zhao, L., et al. MOPD: Multi-teacher on-policy distillation for capability integration in LLM post-training. arXiv preprint arXiv:2606.30406, 2026.

Mathematical Association of America. American invitational mathematics examination (AIME) 2024 and 2025, 2025.

Oren, Y., Sagawa, S., Hashimoto, T. B., and Liang, P. Distributionally robust language modeling. In EMNLP, 2019.

Prime Intellect. SYNTHETIC-1: Two million collaboratively generated reasoning traces from DeepSeek-R1. Prime Intellect blog, 2025.

Pyatkin, V., Malik, S., Graf, V., et al. Generalizing verifiable instruction following. arXiv preprint arXiv:2507.02833, 2025.

Ramesh, S. S., Hu, Y., Chaimalas, I., et al. Group robust preference optimization in reward-free RLHF. In NeurIPS, 2024.

Ramesh, S. S., Ji, X., Zimmer, M., et al. Multi-task GRPO: Reliable LLM reasoning across tasks. In ICML, 2026.

Rockafellar, R. T. and Uryasev, S. Optimization of conditional value-at-risk. J. Risk, 2000.

Sagawa, S., Koh, P. W., Hashimoto, T. B., and Liang, P. Distributionally robust neural networks for group shifts: On the importance of regularization for worst-case generalization. In ICLR, 2020.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal policy optimization algorithms. CoRR, abs/1707.06347, 2017. URL https://arxiv. org/abs/1707.06347.

Sun, J., Zheng, M., Song, M., et al. Distill what you trust: Reliability-aware multi-teacher on-policy distillation. arXiv preprint arXiv:2609.23697, 2026a.

Sun, Z., Zhang, Z., Zhao, F., et al. D<sup>3</sup>-MOPD: Dynamic domain ScheDuling for efficient multi-teacher distillation. arXiv preprint arXiv:2608.24987, 2026b.

Tian, S., Liu, Y., Liu, X., et al. Beyond domain-routed supervision: Cross-teacher guarding for multi-teacher on-policy distillation. arXiv preprint arXiv:2610.13521, 2026.

Vilouras, K., Liu, X., Sanchez, P., O’Neil, A. Q., and Tsaftaris, S. A. Group distributionally robust knowledge distillation. In MLMI@MICCAI, 2023.

Wang, S., Narasimhan, H., Zhou, Y., Hooker, S., Lukasik, M., and Menon, A. K. Robust distillation for worstclass performance: on the interplay between teacher and student objectives. In UAI, 2023.

Wortsman, M., Ilharco, G., Gadre, S. Y., et al. Model soups: Averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. In ICML, 2022.

Xie, S. M., Pham, H., Dong, X., et al. DoReMi: Optimizing data mixtures speeds up language model pretraining. In NeurIPS, 2023.

Yang, A., Li, A., Yang, B., et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

You, S., Xu, C., Xu, C., and Tao, D. Learning from multiple teacher networks. In KDD, 2017.

Yu, Q., Zhang, Z., Zhu, R., et al. DAPO: An open-source LLM reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

Yuan, F., Shou, L., Pei, J., et al. Reinforced multi-teacher selection for knowledge distillation. In AAAI, 2021.

Zhai, R., Dan, C., Kolter, Z., and Ravikumar, P. DORO: distributional and outlier robust optimization. In ICML, 2021.

Zhang, Z., Sun, Z., Zhao, F., et al. Verify before you distill: Prompt-level teacher gating for on-policy distillation. arXiv preprint arXiv:2609.02998, 2026.

Zhao, W., Ren, X., Hessel, J., Cardie, C., Choi, Y., and Deng, Y. WildChat: 1m ChatGPT interaction logs in the wild. In ICLR, 2024.

Zhou, J., Lu, T., Mishra, S., et al. Instruction-following evaluation for large language models. arXiv preprint arXiv:2311.07911, 2023.

## A. Notations

Table 3. Notations and meanings.
<table><tr><td rowspan=1 colspan=1>Notation</td><td rowspan=1 colspan=1>Meaning</td></tr><tr><td rowspan=1 colspan=1> $\Delta ^ { K } , \mathbf { u } , \mathbf { 1 } , \odot$ </td><td rowspan=1 colspan=1>probability simplex; uniform allocation 1/K;all-ones vector; element-wise product</td></tr><tr><td rowspan=1 colspan=1> $\overline { { \mathcal { D } , \ d , \ K } }$ </td><td rowspan=1 colspan=1>domain set, a domain, number of domains (K = |D|)</td></tr><tr><td rowspan=1 colspan=1> $\pi _ { \boldsymbol { \theta } } , \pi _ { \boldsymbol { \phi } _ { d } } , \mathcal { P } _ { d }$ </td><td rowspan=1 colspan=1>student; RL teacher routed to domain d; prompt pool of d</td></tr><tr><td rowspan=1 colspan=1> $\underline { { \boldsymbol { x } , \boldsymbol { y } , | \boldsymbol { y } | , \boldsymbol { j } } }$ </td><td rowspan=1 colspan=1>prompt; on-policy student response; its length; token position</td></tr><tr><td rowspan=1 colspan=1> $D _ { j } , k , \mathcal { L } _ { d } ( \theta )$ </td><td rowspan=1 colspan=1>top-k reverse KL at position j (reward −Dj);top-k size; distillation loss of d</td></tr><tr><td rowspan=1 colspan=1> $s , t , \Delta , B , S$ </td><td rowspan=1 colspan=1>training step; scheduling update;update interval; prompts per batch; total steps</td></tr><tr><td rowspan=1 colspan=1> $m _ { d } ^ { ( s ) } , \tilde { m } _ { d } ^ { ( s ) } , B _ { d }$ </td><td rowspan=1 colspan=1>per-domain KL observation of d at step s; its clipped valuein SAE; domain-d responses in the batch</td></tr><tr><td rowspan=1 colspan=1> $\overline { { { \bar { m } } _ { d } ^ { ( s ) } , \bar { m } _ { d } ^ { \star } , \xi _ { d } ^ { ( s ) } } }$ </td><td rowspan=1 colspan=1>true KL level of d at step s; best true level so far; log-noise</td></tr><tr><td rowspan=1 colspan=1> $\mathbf { q } ^ { ( t ) } , \mathbf { p } ^ { ( t ) }$ </td><td rowspan=1 colspan=1>domain mixture at update t;nominal allocation (progress-driven in RDS-MOPD)</td></tr><tr><td rowspan=1 colspan=1> $\overline { { \hat { \ell } _ { d } ^ { ( s ) } } }$ </td><td rowspan=1 colspan=1>loss estimate in Eqn. $\overline { { ( 2 ) \left( m _ { d } ^ { ( s ) } / I _ { d } \right. } }$ in Vanilla DRO-MOPD)</td></tr><tr><td rowspan=1 colspan=1> $\overline { { \mathbf { } I _ { d } , \mathbf { \nabla } \ell _ { d } , \mathbf { \nabla } \ell _ { d } ^ { \star } } }$ </td><td rowspan=1 colspan=1>initial KL level; current EMA level; best (lowest) level so far</td></tr><tr><td rowspan=1 colspan=1> $g _ { d } , v _ { d } , r _ { d } , r _ { \operatorname* { m a x } }$ </td><td rowspan=1 colspan=1>remaining gap; descent velocity;progress score $( r _ { d } = g _ { d } v _ { d } ) ;$ largest score</td></tr><tr><td rowspan=1 colspan=1> $\overline { { W , \ n _ { 0 } , \ K _ { p } , \ J } }$ </td><td rowspan=1 colspan=1>EMA window; points for $\overline { { I _ { d } ; } }$ max./used rolling deltas</td></tr><tr><td rowspan=1 colspan=1> $\ell _ { d } ^ { \lfloor \ j \rfloor } , \omega , \varrho _ { d , j }$ </td><td rowspan=1 colspan=1>EMA of window j; EMA weights; ratio $\overline { { \ell _ { d } ^ { [ j ] } / \ell _ { d } ^ { [ j + 1 ] } } }$ </td></tr><tr><td rowspan=1 colspan=1> $\overline { { T , \tau ^ { ( t ) } , f , \alpha , z _ { d } } }$ </td><td rowspan=1 colspan=1>base/effective temperature; share floor; jitter range/value</td></tr><tr><td rowspan=1 colspan=1> $\sigma _ { d } , \hat { \sigma } ^ { ( t ) } , \ : N _ { \sigma } , \epsilon$ </td><td rowspan=1 colspan=1>log-noise scale; SE of the leading score;noise window; noise-floor multiplier</td></tr><tr><td rowspan=1 colspan=1> $\overline { { \mu _ { d } ^ { ( s ) } , H , c } }$ </td><td rowspan=1 colspan=1>running median; median window; Huber clipping threshold</td></tr><tr><td rowspan=1 colspan=1> $\overline { { h _ { d } , \kappa , \eta , \lambda _ { \mathrm { l o } } } }$ </td><td rowspan=1 colspan=1>excess loss; excess cap; DRO (tilt) strength; coverage floor</td></tr><tr><td rowspan=1 colspan=1> $\mathcal { U } , \lambda _ { d } , \nu$ </td><td rowspan=1 colspan=1>feasible set; per-domain floor; normalizing constant</td></tr><tr><td rowspan=1 colspan=1> $\overline { { \zeta , \ s _ { 0 } , \ M } }$ </td><td rowspan=1 colspan=1>score bound in SEs; step of an anomaly or jump; anomaly size</td></tr><tr><td rowspan=1 colspan=1> $\rho , \gamma$ </td><td rowspan=1 colspan=1>KL radius; dual variable (proof of Theorem 3.3)</td></tr></table>

## B. Benchmark and Training Data Details

Table 4 lists the six benchmarks: AIME24/25 (Mathematical Association of America, 2025) contain the 2024/2025 AIME problems with integer answers judged by exact match. LiveCodeBench v5/v6 (Jain et al., 2025) collect competitive programming problems by release window and judge each solution by hidden test cases, which alleviates contamination. IFEval (Zhou et al., 2023) contains programmatically verifiable constraints (e.g., length, format, and keywords), and IFBench (Pyatkin et al., 2025) adds novel out-of-distribution constraints; a response is correct only if it satisfies all constraints under the strict prompt-level criterion. We score each IF sample against its own response, unlike the released script of Open-MOPD (Gao et al., 2026), which gives all samples of a prompt one verdict. All metrics are pass@1 estimated by avg@#Samp., with thinking mode, top-p = 0.95, and at most 30k generated tokens, and each run is summarized over its checkpoints by BO and BB (Appendix E).

We build our testbed on Open-MOPD (Gao et al., 2026), to our knowledge the only recent MOPD work that releases its complete training resources (SFT student, domain teachers, and prompts), so all methods share the same student, teachers, and data. The student is SmolLM3-3B (Hugging Face, 2025) initialized from the released MixSFT checkpoint (Student-SFT), and the teachers RL-Math, RL-Code, and RL-IF are obtained by domain-specific RL on top of Student-SFT. The prompt pool (Table 4) draws math from DAPO (Yu et al., 2025), code from PrimeIntellect (Prime Intellect, 2025) and TACO (Li et al., 2023), and IF from Nemotron-IF-RL (Gao et al., 2026). All trained methods share this pool and differ mainly in the domain mixture $\mathbf { q } ^ { ( t ) }$ (except Vanilla DRO-MOPD, Appendix D).

Table 4. Benchmarks and training data per domain (#Prob.: problems; #Samp.: samples per problem, scored by avg@#Samp.; Temp.: sampling temperature).
<table><tr><td rowspan="2">Domain</td><td colspan="4">Evaluation</td><td colspan="3">Training</td></tr><tr><td>Benchmark #Prob. #Samp. Temp.</td><td></td><td></td><td></td><td>Source</td><td>#Prompts</td><td>Teacher</td></tr><tr><td rowspan="2">Math</td><td>AIME24</td><td>30</td><td>64</td><td>0.6</td><td>DAPO</td><td>17,917</td><td>RL-Math</td></tr><tr><td>AIME25</td><td>30</td><td>64</td><td>0.6</td><td></td><td></td><td></td></tr><tr><td rowspan="2">Code</td><td>LCB v5 LCB v6</td><td>167</td><td>10 10</td><td>1.0 1.0</td><td>PrimeIntellect + TACO</td><td>23,667</td><td>RL-Code</td></tr><tr><td></td><td>175</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2">IF</td><td>IFEval</td><td>541</td><td>4</td><td>1.0</td><td> $\overline { { { \mathrm { N e m o t r o n } } { \cdot } } }$ </td><td>45,347</td><td>RL-IF</td></tr><tr><td>IFBench</td><td>300</td><td>4</td><td>1.0</td><td></td><td></td><td></td></tr></table>

## C. Baseline Details

Vanilla MOPD (Ma et al., 2026; Sun et al., 2026b) fixes the mixture to each domain’s share of the prompt pool. Open-MOPD (Gao et al., 2026) adds token-share balancing, KL-gap following, and a reward refresh within every inner update. Vanilla DRO-MOPD (Sagawa et al., 2020) normalizes each per-domain loss by its initial level; its strength η is set as in Appendix D. D<sup>3</sup>-MOPD (Sun et al., 2026b) samples each domain by the softmax of its remaining KL gap times its descent velocity, with a share floor. ParamMerge-Avg (Wortsman et al., 2022) averages the parameters of the three teachers. ParamMerge-TA (Ilharco et al., 2023) adds the scaled sum of the teachers’ task vectors (differences from the SFT student) to the student; we report the better of scaling factors 0.5 and 1.0. The reference rows of Table 1 are excluded from the ranking: Student-SFT is the starting point of all trained methods, and RouteRL (Gao et al., 2026) answers each prompt with its domain teacher (RL-Math for AIME24/25, RL-Code for LCB v5/v6, and RL-IF for IFEval/IFBench), which needs an oracle router and all teachers.

## D. Implementation Details

Training. All methods use the dense top-16 reverse KL of the routed teacher as the per-token reward, B = 1,024 prompts per step, a mini-batch size of 256, 150 steps, a learning rate of $1 . 5 \times 1 0 ^ { - 6 }$ , and at most 16k response tokens. Trained methods start from ${ \bf q } ^ { ( 0 ) } = ( 0 . 4 , 0 . 4 , 0 . 2 )$ (math : code : IF), except Vanilla MOPD (pool-proportional), and the scheduling methods update the mixture every $\Delta \ = \ 1 0$ steps; Open-MOPD and Vanilla DRO-MOPD reweight the per-domain rewards and losses instead (§3.1.1). Each batch jitters the mixture as $\tilde { q } _ { d } \propto q _ { d } ( 1 + z _ { d } )$ with $z _ { d } \sim \mathrm { U n i f } [ - \alpha , \alpha ]$ and draws $B \tilde { q } _ { d }$ prompts by the largestremainder rule. All MOPD methods, including the ablation variants, share one training code and sampler and differ mainly in the mixture rule.

Hyper-parameters. RDS-MOPD uses $\epsilon = 4 , N _ { \sigma } = 4 0$ (NRC), $c = 0 . 5 , H = 5 ( \mathrm { S A E } ) , \eta = 1 , \kappa = 2 , \lambda _ { \mathrm { l o } } = 0 . 2$ (WDR), $W = 1 0 , n _ { 0 } = 5 , K _ { p } = 3 , T = 0 . 5 , f = 0 . 1 0 ,$ and $\alpha = 0 . 3 0$ (sensitivity in Appendix G). Baselines use their official settings; Vanilla DRO-MOPD updates the domain weights by exponentiated gradient on the per-domain loss normalized by its initial level, with strength η = 0.03.

Scheduler Formulas. The progress score $r _ { d } = g _ { d } v _ { d }$ follows D<sup>3</sup>-MOPD (Sun et al., 2026b): $g _ { d } ~ = ~ \ell _ { d } / I _ { d } .$ , and v<sub>d</sub> averages the relative decrease of $\ell _ { d } ^ { [ j ] }$ , the EMA over the j-th most recent window of W clipped observations $( \ell _ { d } = \ell _ { d } ^ { [ 0 ] } )$ , for $j \ < \ J \ \leq \ K _ { p } ,$ , clipped at 0. NRC estimates the log-noise scale over the last $N _ { \sigma }$ observations as $\begin{array} { r } { \sigma _ { d } = \frac { 1 . 4 8 2 6 } { \sqrt { 2 } } \operatorname * { m e d i a n } _ { s ^ { \prime } } \vert \log \tilde { m } _ { d } ^ { ( s ^ { \prime } ) } - \log \tilde { m } _ { d } ^ { ( s ^ { \prime } - 1 ) } \vert } \end{array}$ , and the delta method gives

$$
\begin{array} { l } { \displaystyle \mathrm { S E } ( \boldsymbol { v } _ { d } ) = \frac { \sqrt 2 \sigma _ { d } \| \omega \| _ { 2 } } { J } \Big ( \sum _ { j = 0 } ^ { J - 1 } \varrho _ { d , j } ^ { 2 } \Big ) ^ { 1 / 2 } , } \\ { \displaystyle \mathrm { S E } ( r _ { d } ) = \Big ( \big ( { v _ { d } } \sigma _ { d } \| \omega \| _ { 2 } g _ { d } \big ) ^ { 2 } + \big ( g _ { d } \mathrm { S E } ( \boldsymbol { v } _ { d } ) \big ) ^ { 2 } \Big ) ^ { 1 / 2 } , } \end{array}\tag{12}
$$

where ω are the EMA weights over one window and $\varrho _ { d , j } =$ $\ell _ { d } ^ { [ j ] } / \ell _ { d } ^ { [ j + 1 ] } ; \hat { \sigma } = \mathrm { S E } ( r _ { d _ { \operatorname* { m a x } } } )$ with $d _ { \operatorname* { m a x } } = \arg \operatorname* { m a x } _ { d } r _ { d } .$ During warm-up $( s \leq H )$ , the median of SAE uses the available observations. WDR tilts and normalizes p, then iteratively clamps the coordinates below their floors and rescales the others, which returns Eqn. (11) exactly within K rounds (Appendix H.3).

Algorithm 1: Training procedure of RDS-MOPD   
Input: Student $\pi _ { \boldsymbol { \theta } } ;$ teachers $\{ \pi _ { \phi _ { d } } \} ;$ pools $\{ \mathcal { P } _ { d } \} ;$ ; batch size   
B; steps S; interval $\Delta ;$ hyper-parameters   
(Appendix D).   
Output: Distilled student $\pi _ { \theta } .$   
1 $\mathbf { q }  \mathbf { q } ^ { ( 0 ) } ;$   
2 for s = 1, . . . , S do   
3 Sample B prompts from $\{ \mathcal { P } _ { d } \}$ by the jittered q;   
4 Roll out, score with the routed teachers, and update π   
with the reward $- D _ { j }$ (§2.1);   
5 foreach $d \in \mathcal { D }$ do   
6 Read $m _ { d } ^ { ( s ) }$ by Eqn. (1) and clip it into $\tilde { m } _ { d } ^ { ( s ) }$ by   
Eqn. (9) ; // SAE   
7 if s is a scheduling step (every ∆ steps after 2W   
observations per domain) then   
8 Get $\ell _ { d } , v _ { d } , r _ { d } ,$ and σˆ on $\{ \tilde { m } _ { d } \}$ (Appendix D);   
9 Get τ by Eqn. (8) and p by Eqn. (7) ; // NRC   
10 Update $\mathcal { l } _ { d } ^ { \star }$ and get $h _ { d }$ by Eqn. (10);   
11 $\mathbf { q } $ the solution of Eqn. (11) ; // WDR   
12 return π<sub>θ</sub>;

Evaluation Setup. For each of the three seeds, we evaluate the checkpoints at steps 90, 100, 120, and 150, summarize them by the BO and BB criteria (Appendix E), and apply the same rule to all methods. The main runs use two nodes of eight NVIDIA A800-80GB GPUs, the ablation variants one such node, and the stress test four GPUs, with Python 3.12, PyTorch 2.8.0 (CUDA 12.8), verl 0.7.0, vLLM 0.11.0, and Transformers 4.57.

## E. Evaluation Details

Each benchmark is scored at every evaluated checkpoint (steps 90, 100, 120, and 150), and each run is summarized by two criteria. Best-Overall (BO) selects the single checkpoint with the best mean of the three domain scores (each averaging its two benchmarks) and reports all six benchmarks there; it reflects one deployable model. Best-per-Benchmark (BB) takes the best score of each benchmark over all checkpoints separately. We report the mean and standard deviation over three seeds; training-free methods and reference rows have identical results under both criteria. As a scale-free summary, Avg. Rank (↓) ranks all ranked methods on each benchmark by their 3-seed means (1 is best, ties in the unrounded means averaged) and averages over the six benchmarks, per criterion. For significance, as three seeds are too few, we run the one-sided Wilcoxon signed-rank test between RDS-MOPD and each competitor over all 18 benchmark-seed pairs (six benchmarks × three seeds), per criterion, treating pairs as independent. RDS-MOPD wins on at least 15 out of the 18 pairs against every competitor.

![](images/8fd1bb76b8388c6d55d5eb1b43e7e23776d1fe6c2f099a79bcf477cebbad8049.jpg)  
Figure 7. Domain shares during training (seed 0). Filled/open triangles: noise-driven/other large reversals of $\mathbf { D ^ { 3 } - M O P D } .$ , marked at the most-changed domain; RDS-MOPD has none.

## F. Mixture Dynamics

We compare the real training runs of RDS-MOPD and D<sup>3</sup>-MOPD (same seed and hardware). An update is a large share reversal if any domain share changes by more than 0.3, and the mean share change of a run averages $\begin{array} { r } { \frac { 1 } { 2 } \| \mathbf { q } ^ { ( t ) } - \mathbf { q } ^ { ( t - 1 ) } \| _ { 1 } } \end{array}$ <sub>1</sub>. A large reversal is noise-driven if the same scheduler reproduces it (same update, domain, and direction) in fewer than half of 100 bootstrap replicas of the KL trajectory, resampling residuals around a 21-step running median.

(1) D<sup>3</sup>-MOPD oscillates the mixture with noise. It makes 7 large share reversals in 14 updates, 5 of which are noisedriven (reappearing in only 13–28% of the replicas), e.g., from step 100 to 140 the math share alternates between 0.175 and 0.651 (Figure 7(a)) while the math KL stays within 0.002–0.006. This is mainly because its temperature shrinks with the progress scores (from 0.086 at step 20 to about 0.006 after step 90), so batch-level KL fluctuations alone move most of the budget, i.e., the worst-case misjudgment of §3.1.

(2) RDS-MOPD keeps the mixture responsive yet stable. It makes no large share reversal and cuts the mean share change from 0.258 to 0.118 (54%) and the maximum from 0.478 to 0.285 (Figure 7), while still following real gaps, e.g., the IF share rises to 0.643 at step 140 as only IF still descends. This is mainly because NRC calibrates the radius to the estimated noise level.

## G. Hyper-parameter Sensitivity

We vary $\begin{array} { r l r } { \epsilon } & { { } \in } & { \left\{ 0 , 1 , 2 , 4 , 8 , 1 6 \right\} } \end{array}$ (NRC) and $( \eta , \kappa ) \in$ $\{ 0 , 0 . 5 , 1 , 2 \} \times \{ 1 . 5 , 2 , 3 \}$ (WDR) with the others at their defaults (Appendix D); $\epsilon = 0$ and $\eta = 0$ are $\mathtt { w } / \mathtt { o }$ NRC and w/o WDR. We replay the scheduler on the recorded KL trajectory of RDS-MOPD (150 steps, 14 updates; Figure 8) to isolate the scheduling decisions, with reversals as in Appendix F. The lagging domain at an update is the one whose KL regresses most from its own best level.

![](images/92bfa337fa917c63cea6bb85b01dda820b5eaee03c28af9a5e337750f9fed224.jpg)

![](images/8125d9e8c222ceab16a89a5fb71e3346ef89abac06bcb99227db44970a52e861.jpg)  
Figure 8. Sensitivity to ϵ of NRC (a, b) and η, κ of WDR (c, d) (▲: default $\epsilon = 4 , \eta = 1 , \kappa = 2 ) \colon$ (a) noise-driven reversals per bootstrap replica (mean ± std); (b) mean share change per update; (c) lagging-domain share; (d) max final share (dashed: $1 - 2 \lambda _ { \mathrm { l o } } .$ i.e., two domains at $\lambda _ { \mathrm { l o } } )$

(1) A moderate noise floor reduces noise-driven reversals without freezing the mixture. Raising ϵ from 0 to 4 cuts noise-driven reversals on the recorded trajectory from 3 to 0 and the mean share change from 0.200 to 0.118 (41%); ϵ = 4 is the smallest value removing all large reversals (5, 4, and 3 for ϵ = 0, 1, and 2) while keeping a largest singleupdate change of 0.285, whereas $\epsilon = 8$ and 16 (0.04 and 0.00 noise-driven reversals) suppress it to 0.158 and 0.161, making the scheduler sluggish.

(2) The tilt strength η controls how much budget goes to the lagging domain. Its share grows monotonically with η under every κ, e.g., 0.277, 0.327, 0.382, and 0.499 for $\eta = 0 , 0 . 5 , 1 .$ , and 2 under $\kappa = 2 ,$ so the default η = 1 adds 0.105 at no extra reversals.

(3) The excess cap κ prevents over-concentration. Under $\eta = 1$ , the largest final share grows from 0.393 to 0.464 and 0.565 for κ = 1.5, 2, and 3, and η = 2, κ = 3 ends at (0.200, 0.200, 0.600) with two domains on the floor $\lambda _ { \mathrm { l o } } = 0 . 2$ and 1 noise-driven reversal, whereas the default $\kappa = 2$ ends at (0.288, 0.249, 0.464).

## H. Proofs

## H.1. Proof of Proposition 3.1

Proposition 3.1 is part (i); the full statement adds (ii) for a given $\hat { \sigma } > 0$ and any $\mathbf { r } , \mathbf { r } ^ { \prime } \geq \mathbf { 0 } , \| \mathbf { p } ( \mathbf { r } ) - \mathbf { p } ( \mathbf { r } ^ { \prime } ) \| _ { 1 } \leq$ $\frac { 2 ( \bar { 1 } - K f ) } { T \epsilon \hat { \sigma } } \| \mathbf { r } - \mathbf { r } ^ { \prime } \| _ { \infty } ;$ and (iii) with $\epsilon \ = \ 0 .$ Eqn. (8) reduces to the max-normalized temperature. Under the maxnormalized temperature, such scores can instead change a share by $( 1 - \bar { K f } ) \frac { e ^ { 1 / T } - 1 } { e ^ { 1 / T } + K - 1 }$ for any $\zeta > 0$

Lemma H.1 (ℓ<sub>1</sub> Sensitivity of Softmax). For any $\mathbf { x } , \mathbf { x } ^ { \prime } \in$ $\begin{array} { r } { \mathbb { R } ^ { K } , \| \mathrm { s o f t m a x } ( \mathbf { x } ) - \mathrm { s o f t m a x } ( \mathbf { x } ^ { \prime } ) \| _ { 1 } \leq \frac { 1 } { 2 } \mathrm { o s c } ( \mathbf { x } - \mathbf { x } ^ { \prime } ) } \end{array}$ , where $\begin{array} { r } { \mathrm { o s c } ( \mathbf { y } ) = \operatorname* { m a x } _ { d } y _ { d } - \operatorname* { m i n } _ { d } y _ { d } } \end{array}$ . Moreover, any $\pmb { \Delta } \in \mathbb { R } ^ { K }$ with $\textstyle \sum _ { d } \Delta _ { d } = 0$ satisfies max<sub>d</sub> $| \Delta _ { d } | \leq \frac { 1 } { 2 } \| \Delta \| _ { 1 }$

Proof. Let $\delta = { \bf x } ^ { \prime } - { \bf x }$ and ${ \pmb \pi } ( \beta ) = \mathrm { s o f t m a x } ( { \bf x } + \beta { \pmb \delta } )$ for $\beta \in [ 0 , 1 ] ;$ then $\begin{array} { r } { \frac { d } { d \beta } \pi _ { d } ( \beta ) = \pi _ { d } ( \beta ) \big ( \delta _ { d } - \langle \pmb { \pi } ( \beta ) , \pmb { \delta } \rangle \big ) } \end{array}$ so $\begin{array} { r } { \| \frac { d } { d \beta } \pmb { \pi } ( \beta ) \| _ { 1 } = \mathbb { E } _ { d \sim \pmb { \pi } ( \beta ) } | \delta _ { d } - \bar { \delta } | } \end{array}$ with $\bar { \delta } \ = \ \langle \pi ( \beta ) , \delta \rangle$ the mean absolute deviation of $X ~ = ~ \delta _ { d }$ supported on $[ \delta _ { \mathrm { m i n } } , \delta _ { \mathrm { m a x } } ]$ with $\delta _ { \mathrm { m a x } } - \delta _ { \mathrm { m i n } } = \sec ( \delta ) ( \delta _ { \mathrm { m a x } } = \delta _ { \mathrm { m i n } }$ is trivial). By convexity, $( z - \bar { \delta } ) ^ { + } \leq ( \delta _ { \mathrm { m a x } } - \bar { \delta } ) ( z -$ $\delta _ { \mathrm { m i n } } \big ) / ( \delta _ { \mathrm { m a x } } - \delta _ { \mathrm { m i n } } )$ on $[ \delta _ { \mathrm { m i n } } , \delta _ { \mathrm { m a x } } ]$ , so $\mathbb { E } | X - { \bar { \delta } } | \ =$ $2 \mathbb { E } ( X - \bar { \delta } ) ^ { + } \leq 2 ( \delta _ { \operatorname* { m a x } } - \bar { \delta } ) ( \bar { \delta } - \delta _ { \operatorname* { m i n } } ) / ( \delta _ { \operatorname* { m a x } } - \delta _ { \operatorname* { m i n } } ) \leq$ $( \delta _ { \mathrm { m a x } } - \delta _ { \mathrm { m i n } } ) / 2$ . Integrating over β gives the first claim; for the second, the positive and negative entries of $\pmb { \Delta }$ each sum to $\frac { 1 } { 2 } \| \Delta \| _ { 1 }$ in absolute value. □

(i) Let $b ^ { ( t ^ { \prime } ) } ~ = ~ \epsilon \hat { \sigma } ^ { ( t ^ { \prime } ) } ~ > ~ 0$ for $t ^ { \prime } \in \{ t - 1 , t \}$ . Since $r _ { \mathrm { m a x } } ^ { ( t ^ { \prime } ) } \leq \zeta \hat { \sigma } ^ { ( t ^ { \prime } ) } \leq b ^ { ( t ^ { \prime } ) }$ , Eqn. (8) gives $\tau ^ { ( t ^ { \prime } ) } = T b ^ { ( t ^ { \prime } ) }$ , so the softmax input $\mathbf { x } ^ { ( t ^ { \prime } ) } = \mathbf { r } ^ { ( \dot { t } ^ { \prime } ) } / ( T \epsilon \bar { \sigma } ^ { ( t ^ { \prime } ) } )$ lies in $[ 0 , \zeta / ( \epsilon T ) ] ^ { K }$ and $\sec ( \mathbf { x } ^ { ( t ) } - \mathbf { x } ^ { ( t - 1 ) } ) \ \leq \ 2 \zeta / ( \epsilon T )$ . By Lemma H.1, $\| \mathbf { p } ^ { ( t ) } - \mathbf { p } ^ { ( t - 1 ) } \| _ { 1 } \ \leq \ ( 1 - K \dot { f } ) \dot { \zeta } / ( \epsilon T )$ , so (both allocations sum to one) $| p _ { d } ^ { ( t ) } - p _ { d } ^ { ( t - 1 ) } | \leq ( 1 - K f ) \zeta / ( 2 \epsilon T )$ , i.e., $0 . 7 \zeta / 4 = 0 . 1 7 5 \zeta$ for $K = 3 , f = 0 . 1 , T = 0 . 5 , \mathrm { a n d } \epsilon = 4$ For the max-normalized response, $\mathbf { r } ^ { ( t - 1 ) } = ( \delta , 0 , \ldots , 0 )$ and $\mathbf { r } ^ { ( t ) } = ( 0 , \delta , 0 , \ldots , 0 )$ with $0 ~ < ~ \delta ~ \leq ~ \zeta \hat { \sigma }$ normalize to $\mathbf { e } _ { 1 }$ and e<sub>2</sub> for any δ, so the first share drops from $f +$ $\begin{array} { r } { ( 1 - K f ) \frac { e ^ { 1 / T } } { e ^ { 1 / T } + K - 1 } = 0 . 6 5 1 \mathrm { t o } f + ( 1 - K f ) \frac { 1 } { e ^ { 1 / T } + K - 1 } = } \end{array}$ 0.175 by default, i.e., by $\begin{array} { r } { ( 1 - K f ) \frac { e ^ { 1 / T } - 1 } { e ^ { 1 / T } + K - 1 } = 0 . 4 7 6 } \end{array}$

(ii) Let $b = \epsilon \hat { \sigma } > 0 , n ( { \bf r } ) = \operatorname* { m a x } ( \| { \bf r } \| _ { \infty } , b ) \geq b ,$ and $x ( \mathbf { r } ) \ = \ \mathbf { r } / n ( \mathbf { r } )$ , so that the softmax input is $\chi ( \mathbf { r } ) / T$ (note $\| \mathbf { r } \| _ { \infty } = r _ { \operatorname* { m a x } }$ for $\textbf { r } \geq \mathbf { 0 } )$ . Then $\chi ( { \bf r } ) - \chi ( { \bf r } ^ { \prime } ) = $ $( \mathbf { r } - \mathbf { r } ^ { \prime } ) / n ( \mathbf { r } ) + \mathbf { r } ^ { \prime } \big ( n ( \mathbf { r } ^ { \prime } ) - n ( \mathbf { r } ) \big ) / \big ( n ( \mathbf { r } ) n ( \mathbf { r } ^ { \prime } ) \big )$ . Both terms are bounded by $\| \mathbf { \bar { r } } - \mathbf { r } ^ { \prime } \| _ { \infty } / b \mathrm { ~ i n ~ } \ell _ { \infty } ,$ the second since $\| \mathbf { r } ^ { \prime } \| _ { \infty } / n ( \mathbf { r } ^ { \prime } ) \leq 1$ and $n ( \cdot )$ is 1-Lipschitz in $\ell _ { \infty }$ . Hence the oscillation of the softmax input difference is at most $4 \| \mathbf { r } - \mathbf { r } ^ { \prime } \| _ { \infty } / ( b T )$ , and Lemma H.1 with the factor $( 1 - K f )$ gives the claim. For the max-normalized response, replacing b by min $( r _ { \operatorname* { m a x } } , r _ { \operatorname* { m a x } } ^ { \prime } )$ gives a constant that diverges as the maximum score tends to zero, as the example in (i) realizes. (iii) With $\epsilon = 0 , \operatorname* { m a x } ( r _ { \operatorname* { m a x } } ^ { ( t ) } , 0 ) = r _ { \operatorname* { m a x } } ^ { ( t ) }$ for $\mathbf { r } ^ { ( t ) } \geq \mathbf { 0 } , \mathrm { i . e . }$ the max-normalized temperature. □

## H.2. Proof of Proposition 3.2

(i) Directly from Eqn. (9). (ii) Primes denote contaminated quantities; L and U are the smallest and largest other observations in $[ s _ { 0 } - H , s _ { 0 } + H ]$ . As the median uses raw observations, $\mu _ { d } ^ { \prime ( s ) } \neq \mu _ { d } ^ { ( s ) }$ only if $s \in ( s _ { 0 } , s _ { 0 } + H ]$ , and for $s > s _ { 0 } { + H }$ the window and current observation are clean, so $\tilde { m } _ { d } ^ { \prime ( s ) } = \tilde { m } _ { d } ^ { ( s ) }$ . For $s = s _ { 0 }$ , the window $\{ s _ { 0 } { - } H , \ldots , s _ { 0 } { - } 1 \}$ is clean (as $s _ { 0 } ~ > ~ H )$ and lies in $[ s _ { 0 } - H , s _ { 0 } + H ]$ , so $\mu _ { d } ^ { \prime ( s _ { 0 } ) } \in [ L , U ]$ , and (i) gives $\tilde { m } _ { d } ^ { \prime ( s _ { 0 } ) } \in [ e ^ { - c } L , e ^ { c } U ]$ for any M. For $s \in ( s _ { 0 } , s _ { 0 } + H ]$ , the window contains the contaminated value and $H - 1 \geq 2$ clean values in $[ L , U ]$ . The median involves only ranks in $[ 2 , H - 1 ]$ (as $H \geq 3 ) !$ : the k-th order statistic with $k = ( H + 1 ) / 2$ for odd H, or the mean of the k-th and $( k + 1 )$ )-th with $k = H / 2$ for even $H$ Each such order statistic lies between two order statistics of the remaining $H - 1$ values, so $\mu _ { d } ^ { \prime ( s ) } \in [ L , U ]$ . The clean $m _ { d } ^ { ( s ) } \in [ L , U ]$ , and clipping moves it toward $\mu _ { d } ^ { \prime ( s ) }$ , so $\tilde { m } _ { d } ^ { \prime ( s ) } \in [ L , U ] \subseteq [ e ^ { - c } L , e ^ { c } U ]$ . An EMA level is a convex combination of clipped observations, each equal to its clean counterpart or in $[ e ^ { - c } L , e ^ { c } U ]$ , so it lies in an interval independent of $M ;$ without clipping, the EMA weight $\omega _ { i } > 0$ of step $s _ { 0 }$ makes it at least $\omega _ { i } M$

(iii) Let $m _ { d } ^ { ( s ) } = A$ for $s < s _ { 0 }$ and $m _ { d } ^ { ( s ) } = A ^ { \prime }$ for $s \geq s _ { 0 }$ For $s \geq s _ { 0 } + \lfloor H / 2 \rfloor + 1$ , the window of $\mu _ { d } ^ { ( s ) }$ (of size at most H) contains at least $\lfloor H / 2 \rfloor + 1$ values equal to $A ^ { \prime }$ (a strict majority), so $\mu _ { d } ^ { ( s ) } = A ^ { \prime } = m _ { d } ^ { ( s ) }$ lies in the clipping range and $\tilde { m } _ { d } ^ { ( s ) } = m _ { d } ^ { ( s ) }$ □

## H.3. Proof of Theorem 3.3

The full statement adds: in (i), the maximizer is the KL projection of $\mathbf { p } \odot e ^ { \eta \mathbf { h } }$ (normalized) onto $\mathcal { U } ; \mathbf { \Xi } ( \mathrm { i i } )$ also gives $q _ { d } ~ \ge ~ \operatorname* { m a x } ( \lambda _ { d } , \kappa ^ { - \eta } p _ { d } )$ and $\mathrm { K L } ( \mathbf { q } \parallel \mathbf { p } ) \mathbf { \beta } \leq \mathbf { \beta } \eta$ log κ; in (i) with $\rho ~ = ~ \mathrm { K L } ( \mathbf { q } \parallel \mathbf { p } )$ , when the floor is inactive, $\begin{array} { r } { \operatorname* { s u p } _ { \mathrm { K L } ( \mathbf { q } ^ { \prime } \parallel \mathbf { p } ) \leq \rho } \langle \mathbf { q } ^ { \prime } , \mathbf { h } \rangle = \operatorname* { i n f } _ { \gamma > 0 } \left\{ \gamma \rho + \gamma \log \mathbb { E } _ { d \sim \mathbf { p } } [ e ^ { h _ { d } / \gamma } ] \right\} } \end{array}$ attained at $\gamma = 1 / \eta$ . Let $Z = \langle \mathbf { p } , e ^ { \eta \mathbf { h } } \rangle \geq 1$ (as $h _ { d } \geq 0 )$ and $\mathbf { w } = \mathbf { p } \odot e ^ { \eta \mathbf { h } } / Z \in \Delta ^ { K }$

(i), unique maximizer. For any $\mathbf { q } ~ \in ~ \Delta ^ { K }$ , expanding $\operatorname { K L } ( \mathbf { q } \parallel \mathbf { w } )$ gives

$$
\begin{array} { r } { \langle \mathbf { q } , \mathbf { h } \rangle - \frac { 1 } { \eta } \mathrm { K L } ( \mathbf { q } \parallel \mathbf { p } ) = \frac { 1 } { \eta } \log Z - \frac { 1 } { \eta } \mathrm { K L } ( \mathbf { q } \parallel \mathbf { w } ) . } \end{array}\tag{13}
$$

Hence Eqn. (6) is the KL projection $\mathrm { m i n } _ { \mathbf { q } \in \mathcal { U } } \mathrm { K L } ( \mathbf { q } \parallel \mathbf { w } )$ onto the convex U, unique by strict convexity. With multipliers χ and $\beta _ { d }$ of the simplex and floor constraints, the KKT conditions $\mathrm { o g } ( q _ { d } / w _ { d } ) + 1 + \chi - \beta _ { d } = 0 , \beta _ { d } \geq 0 ,$ and $\beta _ { d } ( q _ { d } \ : - \ : \lambda _ { d } ) = 0$ give $q _ { d } ~ = ~ a w _ { d }$ if $q _ { d } \ > \lambda _ { d } ,$ and $q _ { d } ~ = ~ \lambda _ { d } ~ \ge ~ a w _ { d }$ otherwise, with $\ a \ = \ e ^ { - 1 - \chi }$ , i.e., $q _ { d } = \operatorname* { m a x } ( \lambda _ { d } , a w _ { d } )$ , which is Eqn. (11) with $\nu = a / Z$ $\begin{array} { r } { G ( \boldsymbol { a } ) = \sum _ { d } \operatorname* { m a x } ( \lambda _ { d } , \boldsymbol { a w } _ { d } ) } \end{array}$ is continuous and nondecreasing with $\begin{array} { r } { G ( 0 ) = \sum _ { d } \lambda _ { d } < 1 \leq G ( 1 ) } \end{array}$ , so $G ( a ) = 1$ has a root in (0, 1], and it is unique: at any root, some d has $a w _ { d } \geq \lambda _ { d }$ (otherwise $\begin{array} { r } { G ( { a } ) = \sum _ { d } \lambda _ { d } < 1 ) } \end{array}$ , and as $w _ { d } > 0$ $G ( a ^ { \prime } ) \geq G ( a ) + ( a ^ { \prime } - a ) w _ { d } > 1$ for all $a ^ { \prime } > a$

Exactness of the implementation. From w (Appendix D), the unclamped coordinates after round k are $b _ { k } w _ { d } , b _ { 0 } =$ 1. Clamping only raises coordinates, so $b _ { k + 1 } < b _ { k } ;$ the clamped set never shrinks, and the unclamped set is never empty as $\textstyle \sum _ { d } \lambda _ { d } \ < \ 1$ , so the procedure stops within K rounds, where every unclamped d satisfies $b _ { k } w _ { d } \geq \lambda _ { d } .$ , and every clamped d, clamped at some round $j \leq k$ , satisfies $b _ { k } w _ { d } \leq b _ { j } w _ { d } < \lambda _ { d }$ . Hence $\mathbf { q } = \operatorname* { m a x } ( \lambda , b _ { k } \mathbf { w } )$ with a unit sum, and $b _ { k } = a$ by uniqueness.

(i), KL ball. For any $\mathbf { q } ^ { \prime } \in \mathcal { U }$ with $\begin{array} { r } { \mathrm { K L } ( \mathbf { q } ^ { \prime } \parallel \mathbf { p } ) \leq \rho , } \end{array}$ the optimality of $\mathbf { q }$ in Eqn. (6) gives $\langle { \bf q } ^ { \prime } , { \bf h } \rangle \leq \langle { \bf q } , { \bf h } \rangle +$ $\begin{array} { r } { \frac 1 \eta \left( \mathrm { K L } ( \mathbf { q } ^ { \prime } \parallel \mathbf { p } ) - \boldsymbol { \rho } \right) \leq \left. \mathbf { q } , \mathbf { h } \right. } \end{array}$ , and q is feasible, so it solves the constrained problem. For the dual form, Eqn. (13) with $1 / \gamma$ in place of η gives, for any $\gamma > 0$ and $\mathrm { K L } ( \mathbf { q } ^ { \prime } \parallel \mathbf { p } ) \leq$ $\rho , \langle { \mathbf { q } ^ { \prime } } , \mathbf { h } \rangle \ \leq \ \gamma \mathrm { K L } ( { \mathbf { q } ^ { \prime } } \| { \mathbf { p } } ) + \gamma \log \mathbb { E } _ { \mathbf { p } } [ e ^ { h _ { d } / \gamma } ] \ \leq \ \gamma \rho + $ γ log $\mathbb { E } _ { \mathbf { p } } [ e ^ { h _ { d } / \gamma } ] , \mathrm { i . e . }$ , weak duality (Lam, 2016). When the floor is inactive, $\textbf { q } = \textbf { w }$ and Eqn. (13) gives $\left. \mathbf { q } , \mathbf { h } \right. =$ $\begin{array} { r } { \frac { 1 } { \eta } \rho + \frac { 1 } { \eta } \log Z , \mathrm { i . e . } } \end{array}$ , equality at $\gamma = 1 / \eta$

(ii), bounded deviation. Let $F = \{ d : a w _ { d } \geq \lambda _ { d } \}$ and C its complement; $F ~ \neq ~ \emptyset$ as $\textstyle \sum _ { d } \lambda _ { d } \ < \ 1$ . Let $\psi _ { d } ~ = ~ a e ^ { \eta h _ { d } } / Z ,$ , so that $q _ { d } / p _ { d } ~ = ~ \psi _ { d }$ for $d \in F .$ , and $q _ { d } / p _ { d } = \lambda _ { d } / p _ { d } \in ( \psi _ { d } , 1 ]$ for $d \in C$ . Upper bound: for $d \in { \cal F } , q _ { d } / p _ { d } = \psi _ { d } \le e ^ { \eta h _ { d } } \le \kappa ^ { \eta }$ as $a \leq 1 \leq Z ;$ for $d \in C , q _ { d } / p _ { d } \leq 1$ . Lower bound: if ma $\mathrm { x } _ { d \in F } \psi _ { d } < 1$ all ratios are below one on F and at most one on $C ,$ contradicting $\begin{array} { r } { \sum _ { d } p _ { d } ( q _ { d } / p _ { d } ) \ = \ 1 ; } \end{array}$ hence some $d ^ { \prime } \in F$ has $\psi _ { d ^ { \prime } } \geq 1$ , and for every $\begin{array} { r } { d , q _ { d } / p _ { d } \ge \psi _ { d } = \psi _ { d ^ { \prime } } e ^ { \eta ( h _ { d } - h _ { d ^ { \prime } } ) } \ge } \end{array}$ $\kappa ^ { - \eta }$ , since $h _ { d } - h _ { d ^ { \prime } } \geq - \log \kappa$ . Finally, $\operatorname { K L } ( \mathbf { q } \parallel \mathbf { p } ) =$ $\begin{array} { r } { \sum _ { d } q _ { d } \log ( q _ { d } / p _ { d } ) \leq \operatorname* { m a x } _ { d } \log ( q _ { d } / p _ { d } ) \leq \eta } \end{array}$ log κ. (ii), no starvation. $q _ { d } \ge \lambda _ { d } = \operatorname* { m i n } ( p _ { d } , \lambda _ { \mathrm { l o } } )$ by feasibility, and $p _ { d } \geq f$ by Eqn. (7) as $K f \le 1$ , so $q _ { d } \ge \operatorname* { m i n } ( f , \lambda _ { \mathrm { l o } } )$

(iii) If $h _ { d } \equiv h _ { 0 }$ , then $\mathbf { w } = \mathbf { p } \in \mathcal { U } , \mathrm { s o } \mathbf { q } = \mathbf { p }$ . Let $d ^ { \dagger } \in$ arg ma $\mathrm { x } _ { d } h _ { d }$ , so that $\psi _ { d ^ { \dagger } } = \mathrm { m a x } _ { d } \psi _ { d } . \mathrm { I f } d ^ { \dagger } \in C$ , then every $d \in F$ has $q _ { d } / p _ { d } = \psi _ { d } \leq \psi _ { d ^ { \dagger } } < \lambda _ { d ^ { \dagger } } / p _ { d ^ { \dagger } } \leq 1$ , and every $d \in C$ has $q _ { d } / p _ { d } \leq 1$ , contradicting $\begin{array} { r } { \sum _ { d } p _ { d } ( q _ { d } / p _ { d } ) = 1 } \end{array}$ Hence $d ^ { \dagger } \in F$ , and by (ii), $q _ { d ^ { \dagger } } / p _ { d ^ { \dagger } } = \mathrm { m a x } _ { d \in F } \psi _ { d } \geq$ 1. □